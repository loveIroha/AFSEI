"""Self-contained, checksummed EP/FSI checkpoints, including force history.

No pickle, external mesh dependency or reuse of an AFSI-only checkpoint.
Publication is atomic and the schema hashes array names, shapes and dtypes.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ..config import _decode
from ..mesh_io import ImportedSolidMesh
from ..mac.paper_coupling import PaperState
from .config import AFSEIConfig
from .driver import AFSEIStepper,CoupledState,StrainHistory
from .tissue import TissueState
from .protocol import CycleState
from .buffered_load import ActiveLoadState

MESH = ('X','cells','faces','facet_tags','boundary_cells','boundary_local_facets','fiber','sheet')
INTEGER = {'cells','faces','facet_tags','boundary_cells','boundary_local_facets'}
REQUIRED = {*MESH,'x','pressure','force','velocity_0','velocity_1','velocity_2','voltage_v','gpb','land',
            'stretch','zs','zw','ionic_current','load_previous_x','load_xs','load_xw','load_zs_old','load_zw_old'}
OPTIONAL = {'previous_x','cycle_anchor','resting_land','force_anchor'}


def digest(metadata,data):
    h = hashlib.sha256(json.dumps(metadata,sort_keys=True,allow_nan=False).encode())
    for name in sorted(data):
        a = data[name]
        h.update(name.encode()); h.update(a.dtype.str.encode())
        h.update(json.dumps(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def save(path,driver,state,progress=None):
    driver._check_clock(state)
    array = lambda v: v.detach().cpu().numpy()
    data = {name:array(getattr(driver.mesh,name)) for name in MESH}
    data.update({name:array(getattr(state.fsi,name)) for name in ('x','pressure','force')})
    data.update({f'velocity_{c}':array(v) for c,v in enumerate(state.fsi.velocity)})
    data.update({name:array(getattr(state.tissue,name)) for name in ('voltage_v','gpb','land')})
    data.update({name:array(getattr(state.strain,name)) for name in ('stretch','zs','zw')})
    data['ionic_current'] = array(state.ionic_current)
    data.update({f'load_{name}':array(getattr(state.force_load,name))
                 for name in ('previous_x','xs','xw','zs_old','zw_old')})
    for name,value in (('previous_x',state.fsi.previous_x),('cycle_anchor',state.cycle.anchor),
                       ('resting_land',state.cycle.resting_land),('force_anchor',state.force_load.anchor)):
        if value is not None: data[name] = array(value)
    metadata = dict(producer='afsei-ep-fsi',schema=1,units='cm-g-s; V; microA/cm2; GPB-ms-mV',
        config=asdict(driver.config),tick=state.tick,step=state.fsi.step,time=state.fsi.time,
        force_time=state.fsi.force_time,pressure_time=state.fsi.pressure_time,previous_dt=state.fsi.previous_dt,
        cycle=dict(mode=state.cycle.mode,closure_tick=state.cycle.closure_tick,
                   freeze_tick=state.cycle.freeze_tick,stable_ticks=state.cycle.stable_ticks),
        active_load=dict(dt_s=state.force_load.dt_s,use_land=state.force_load.use_land,
                         ramp_factor=state.force_load.ramp_factor),
        vertex_count=driver.mesh.vertex_count,mesh_metadata=driver.mesh.metadata,
        rejected_attempts=state.rejected_attempts,progress=progress or {},
        circulation_ports_connected=False)
    metadata['sha256'] = digest(metadata,data)
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,metadata=json.dumps(metadata,allow_nan=False),**data)
    temporary.replace(path)


def load(path,device='cuda'):
    with np.load(path,allow_pickle=False) as archive:
        data = {name:archive[name] for name in archive.files}
    try:
        metadata = json.loads(str(data.pop('metadata')))
        checksum = metadata.pop('sha256')
        if (metadata.get('producer') != 'afsei-ep-fsi' or metadata.get('schema') != 1
                or metadata.get('units') != 'cm-g-s; V; microA/cm2; GPB-ms-mV'
                or metadata.get('circulation_ports_connected') is not False
                or not REQUIRED <= set(data) or set(data)-REQUIRED-OPTIONAL
                or digest(metadata,data) != checksum):
            raise ValueError('invalid AFSEI checkpoint/schema/checksum; an AFSI-only checkpoint cannot resume EP')
        for name,value in data.items():
            if value.dtype != (np.int64 if name in INTEGER else np.float64) or not np.isfinite(value).all():
                raise ValueError('invalid AFSEI checkpoint array type or values')
        config = _decode(AFSEIConfig,metadata['config'])
        tensor = lambda name: torch.as_tensor(data[name],device=device)
        mesh = ImportedSolidMesh(*(tensor(n) for n in MESH),metadata['vertex_count'],metadata['mesh_metadata'])
        driver = AFSEIStepper(mesh,config)
        optional = lambda name: tensor(name) if name in data else None
        fsi = PaperState(metadata['step'],metadata['time'],tensor('x'),tuple(tensor(f'velocity_{c}') for c in range(3)),
            tensor('pressure'),tensor('force'),metadata['force_time'],pressure_time=metadata['pressure_time'],
            previous_x=optional('previous_x'),previous_dt=metadata['previous_dt'])
        tissue = TissueState(*(tensor(n) for n in ('voltage_v','gpb','land')))
        strain = StrainHistory(*(tensor(n) for n in ('stretch','zs','zw')))
        cycle = CycleState(**metadata['cycle'],anchor=optional('cycle_anchor'),resting_land=optional('resting_land'))
        active = ActiveLoadState(*(tensor('load_'+n) for n in ('previous_x','xs','xw','zs_old','zw_old')),
            **metadata['active_load'],anchor=optional('force_anchor'))
        state = CoupledState(metadata['tick'],fsi,tissue,strain,cycle,active,tensor('ionic_current'),metadata['rejected_attempts'])
        driver._check_clock(state)
        driver.model.validate(fsi.x)
        driver.ode._validate(tissue,strain.stretch,strain.zs,strain.zw,fsi.time)
        grid = driver._fsi_stepper(driver.maximum_ticks).grid
        grid.check_velocity(fsi.velocity)
        if (fsi.pressure.shape != grid.shape or fsi.force.shape != fsi.x.shape
                or state.ionic_current.shape != tissue.voltage_v.shape or (strain.stretch <= 0).any()
                or type(state.rejected_attempts) is not int or state.rejected_attempts < 0
                or cycle.mode not in ('zero','land','relaxation')
                or any(type(v) is not int for v in (cycle.closure_tick,cycle.freeze_tick,cycle.stable_ticks))
                or not -1 <= cycle.closure_tick <= state.tick or not -1 <= cycle.freeze_tick <= state.tick
                or cycle.stable_ticks < 0):
            raise ValueError('invalid AFSEI checkpoint clocks, shapes or cycle')
        if cycle.mode == 'relaxation' and (cycle.anchor is None or cycle.closure_tick < 0):
            raise ValueError('relaxation requires an accepted closure anchor')
        if cycle.resting_land is not None and cycle.resting_land.shape != tissue.land.shape:
            raise ValueError('invalid pre-stimulus Land snapshot shape')
        if cycle.anchor is not None and (cycle.anchor.shape != driver.buffer.anchor.shape or (cycle.anchor < 0).any()):
            raise ValueError('invalid closure anchor shape/values')
        if fsi.step > 0 and (fsi.previous_x is None or fsi.previous_dt != active.dt_s
                or not torch.equal(fsi.previous_x,active.previous_x)):
            raise ValueError('checkpoint force history must use the pre-step geometry')
        driver.buffer.stage(active)
        if not torch.allclose(fsi.force,driver.model.force(fsi.x,fsi.time),rtol=1e-9,atol=1e-7):
            raise ValueError('checkpoint force disagrees with saved active trial history')
        return driver,state,metadata['progress']
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError('malformed AFSEI checkpoint') from exc
