"""Distinct restart format for the pressure-balanced active/RCR validation.

Ordinary drivers reject this producer until the phase driver is integrated.
Retains current force-load history and the old cap geometry, never refits force.
"""
from dataclasses import replace, asdict
import json
from pathlib import Path
import numpy as np
import torch
from . import checkpoint as base
from .config import AFSEIConfig
from .driver import CoupledState, StrainHistory
from .tissue import TissueState
from .protocol import CycleState
from .buffered_load import ActiveLoadState
from .windkessel import WindkesselParameters
from ..config import _decode
from ..mesh_io import ImportedSolidMesh
from ..mac.paper_coupling import PaperState

PRODUCER = 'afsei-mac-balanced-interval-validation'


def save(path,driver,state,progress=None):
    driver.validate_control(state)
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.working')
    try:
        base.save(temporary,driver,state.ep,progress)
        with np.load(temporary,allow_pickle=False) as archive:
            data = {k:archive[k] for k in archive.files}
        meta = json.loads(str(data.pop('metadata'))); meta.pop('sha256')
        meta.update(producer=PRODUCER, balanced=dict(target_volume_ml=state.target_volume_ml,
            mode=state.mode,arterial_pc_mmhg=state.arterial_pc_mmhg,windkessel=asdict(state.windkessel),
            outlet_open=state.outlet_open,accepted_port_steps=state.accepted_port_steps,
            cap_mode=state.cap_mode,filling_pressure_mmhg=state.filling_pressure_mmhg),
            balanced_execution=dict(level=driver.pressure_mode_level,
                volume_transfer='integrated-p1-gradient-v1',
                guard_cells=driver.pressure_guard_cells,one_way=driver.one_way,
                cap_patch_width_cells=driver.cap_patch_width_cells,
                valve_resistance_mmhg_s_per_ml=driver.valve_resistance_mmhg_s_per_ml,
                port_geometry=asdict(driver.port_geometry) if driver.port_geometry else None),
            circulation_ports_connected=state.mode=='rcr',full_phase_controller=False)
        data['cap_pressure'] = state.cap_pressure.detach().cpu().numpy()
        data['cap_geometry'] = state.cap_geometry.detach().cpu().numpy()
        if hasattr(driver,'checkpoint_metadata'):
            meta.update(driver.checkpoint_metadata(state))
        meta['sha256'] = base.digest(meta,data)
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,metadata=json.dumps(meta,allow_nan=False),**data)
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def load(path,device='cuda',*,cardiac=False):
    from .mac_balanced_dynamic import BalancedAFSEIStepper, BalancedState
    from ..circulation.geometry import PortGeometry
    expected=PRODUCER
    extra={}; state_class=BalancedState; driver_class=BalancedAFSEIStepper
    if cardiac:
        from .mac_cycle import PRODUCER as expected,CardiacState,CardiacStepper
        state_class=CardiacState; driver_class=CardiacStepper
    with np.load(path,allow_pickle=False) as archive:
        data = {k:archive[k] for k in archive.files}
    try:
        meta = json.loads(str(data.pop('metadata'))); checksum = meta.pop('sha256')
        if (meta.get('producer') != expected or meta.get('schema') != 1
                or base.digest(meta,data) != checksum or not base.REQUIRED <= set(data)
                or set(data)-base.REQUIRED-base.OPTIONAL
                or not {'cap_pressure','cap_geometry'} <= set(data)):
            raise ValueError('invalid balanced checkpoint/schema/checksum')
        for name,a in data.items():
            if a.dtype != (np.int64 if name in base.INTEGER else np.float64) or not np.isfinite(a).all():
                raise ValueError('invalid balanced checkpoint array values')
        tensor = lambda n: torch.as_tensor(data[n],device=device)
        optional = lambda n: tensor(n) if n in data else None
        mesh = ImportedSolidMesh(*(tensor(n) for n in base.MESH),meta['vertex_count'],meta['mesh_metadata'])
        cfg = _decode(AFSEIConfig,meta['config']); settings = meta['balanced_execution']
        if settings.get('volume_transfer') != 'integrated-p1-gradient-v1':
            raise ValueError('checkpoint uses a different volume transfer discretization')
        if cardiac: extra=meta['cardiac_execution']
        driver = driver_class(mesh,cfg,level=settings['level'],**extra)
        driver.pressure_guard_cells = settings['guard_cells']; driver.one_way = settings['one_way']
        driver.cap_patch_width_cells = settings['cap_patch_width_cells']
        driver.valve_resistance_mmhg_s_per_ml = settings['valve_resistance_mmhg_s_per_ml']
        driver.port_geometry = PortGeometry(**settings['port_geometry']) if settings['port_geometry'] else None
        fsi = PaperState(meta['step'],meta['time'],tensor('x'),tuple(tensor(f'velocity_{c}') for c in range(3)),
            tensor('pressure'),tensor('force'),meta['force_time'],pressure_time=meta['pressure_time'],
            previous_x=optional('previous_x'),previous_dt=meta['previous_dt'])
        tissue = TissueState(*(tensor(n) for n in ('voltage_v','gpb','land')))
        strain = StrainHistory(*(tensor(n) for n in ('stretch','zs','zw')))
        cycle = CycleState(**meta['cycle'],anchor=optional('cycle_anchor'),resting_land=optional('resting_land'))
        active = ActiveLoadState(*(tensor('load_'+n) for n in ('previous_x','xs','xw','zs_old','zw_old')),
            **meta['active_load'],anchor=optional('force_anchor'))
        ep = CoupledState(meta['tick'],fsi,tissue,strain,cycle,active,tensor('ionic_current'),meta['rejected_attempts'])
        control = dict(meta['balanced']); control['windkessel'] = WindkesselParameters(**control['windkessel'])
        if cardiac: control.update(meta['cardiac'])
        state = state_class(ep,cap_pressure=tensor('cap_pressure'),cap_geometry=tensor('cap_geometry'),**control)
        driver.validate_control(state); driver.model.validate(fsi.x)
        driver.ode._validate(tissue,strain.stretch,strain.zs,strain.zw,fsi.time)
        e = driver._fsi_stepper(driver.maximum_ticks); e.grid.check_velocity(fsi.velocity)
        if (fsi.pressure.shape != e.grid.shape or fsi.force.shape != fsi.x.shape
                or state.ionic_current.shape != tissue.voltage_v.shape
                or state.rejected_attempts < 0 or cycle.mode not in ('zero','land','relaxation')
                or (fsi.step > 0 and (fsi.previous_dt != active.dt_s
                    or not torch.equal(fsi.previous_x,active.previous_x)))):
            raise ValueError('invalid balanced checkpoint history/shapes')
        driver.buffer.stage(active)
        if not torch.allclose(driver.force_for_state(state),fsi.force,rtol=1e-9,atol=1e-7):
            raise ValueError('balanced checkpoint force history mismatch')
        return driver,state,meta['progress']
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError('malformed balanced checkpoint') from exc
