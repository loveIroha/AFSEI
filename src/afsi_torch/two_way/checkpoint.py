"""Self-contained numeric-only three-field checkpoints; no exterior fluid.

Stores nodal solid velocity/inertia and actual fitted blood velocity/pressure.
Earlier one-way/IB checkpoints have different physics and cannot resume here.
"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from ..ep.checkpoint import MESH,INTEGER as EP_INTEGER
from ..ep.driver import CoupledState,StrainHistory
from ..ep.tissue import TissueState
from ..ep.protocol import CycleState
from ..ep.buffered_load import ActiveLoadState
from ..ep.config import AFSEIConfig, decode_checkpoint_config
from ..config import _decode
from ..mesh_io import ImportedSolidMesh
from ..moving_cavity.geometry import MovingCavityMesh
from ..moving_cavity.flow import ALEState,ALEConfig
from ..moving_cavity.checkpoint import atomic,validate as validate_blood
from ..circulation.checkpoint import digest
from .driver import ThreeFieldStepper,ThreeFieldState
from .mechanics import SolidState,MechanicalConfig

BLOOD_MESH=('cells','faces','cell_faces','signs','tags','left','right','reference_solid')
SOLID=('x','velocity','force')
EP=('voltage_v','gpb','land','stretch','zs','zw','ionic_current')
LOAD=('previous_x','xs','xw','zs_old','zw_old')
BLOOD=('solid_x','previous_solid_x','velocity','pressure','pc_mmhg','q_out_ml_s','port_pressure_cgs')
REQUIRED={*('mesh__'+n for n in MESH),*('solid__'+n for n in SOLID),*('ep__'+n for n in EP),
    *('load__'+n for n in LOAD),*('blood_mesh__'+n for n in BLOOD_MESH),*('blood__'+n for n in BLOOD),
    'blood_mesh__map_indices','blood_mesh__map_values'}
OPTIONAL={'solid__previous_x','cycle__anchor','cycle__resting_land','load__anchor'}
INTEGER={*('mesh__'+n for n in EP_INTEGER),*('blood_mesh__'+n for n in ('cells','faces','cell_faces','tags','left','right','map_indices'))}


def validate(driver,state):
    e=state.ep; s=e.fsi; b=state.blood
    driver._check_clock(e); driver.model.validate(s.x)
    driver.ode._validate(e.tissue,e.strain.stretch,e.strain.zs,e.strain.zw,s.time)
    validate_blood(driver.blood_driver,b)
    if e.tick!=b.tick or s.step!=b.step or not torch.equal(s.x,b.solid_x): raise ValueError('three-field checkpoint clocks/geometries differ')
    for x in (s.x,s.velocity,s.force,e.ionic_current,e.strain.stretch,e.strain.zs,e.strain.zw):
        if x.dtype!=torch.float64 or x.device!=driver.mesh.X.device or not torch.isfinite(x).all(): raise ValueError('invalid three-field tensor')
    if (s.velocity.shape!=s.x.shape or s.force.shape!=s.x.shape or e.ionic_current.shape!=e.tissue.voltage_v.shape
        or e.strain.stretch.shape!=e.tissue.voltage_v.shape or e.strain.zs.shape!=e.strain.stretch.shape
        or e.strain.zw.shape!=e.strain.stretch.shape or (e.strain.stretch<=0).any()
        or type(e.rejected_attempts)!=int or e.rejected_attempts<0 or e.cycle.mode not in ('zero','land','relaxation')):
        raise ValueError('invalid three-field shapes/cycle')
    cycle=e.cycle
    if (any(type(v)!=int for v in (cycle.closure_tick,cycle.freeze_tick,cycle.stable_ticks))
        or not -1<=cycle.closure_tick<=e.tick or not -1<=cycle.freeze_tick<=e.tick or cycle.stable_ticks<0): raise ValueError('invalid cycle clocks')
    if cycle.mode=='relaxation' and (cycle.anchor is None or cycle.closure_tick<0): raise ValueError('relaxation needs closure anchor')
    if cycle.resting_land is not None and cycle.resting_land.shape!=e.tissue.land.shape: raise ValueError('invalid Land resting snapshot')
    if cycle.anchor is not None and (cycle.anchor.shape!=driver.buffer.anchor.shape or (cycle.anchor<0).any()): raise ValueError('invalid closure anchor')
    if s.step>0:
        if (s.previous_x is None or s.previous_dt!=e.force_load.dt_s or s.previous_dt!=b.last_dt_s
            or not torch.equal(s.previous_x,e.force_load.previous_x) or not torch.equal(s.previous_x,b.previous_solid_x)
            or not torch.allclose(s.velocity,(s.x-s.previous_x)/s.previous_dt,rtol=1e-10,atol=1e-10)):
            raise ValueError('solid velocity/force history disagrees with previous accepted geometry')
    driver.buffer.stage(e.force_load)
    if not torch.allclose(s.force,driver.model.force(s.x,s.time),rtol=1e-9,atol=1e-7): raise ValueError('solid force disagrees with checkpoint active history')


def save(path,driver,state,progress=None):
    validate(driver,state); e=state.ep; s=e.fsi; b=state.blood; m=driver.blood_driver.mesh
    a=lambda x:x.detach().cpu().numpy()
    data={'mesh__'+n:a(getattr(driver.mesh,n)) for n in MESH}
    data.update({'solid__'+n:a(getattr(s,n)) for n in SOLID})
    data.update({'ep__'+n:a(getattr(e.tissue,n)) for n in ('voltage_v','gpb','land')})
    data.update({'ep__'+n:a(getattr(e.strain,n)) for n in ('stretch','zs','zw')}); data['ep__ionic_current']=a(e.ionic_current)
    data.update({'load__'+n:a(getattr(e.force_load,n)) for n in LOAD})
    data.update({'blood_mesh__'+n:a(getattr(m,n)) for n in BLOOD_MESH})
    coo=m.mapping.to_sparse_coo().coalesce(); data.update(blood_mesh__map_indices=a(coo.indices()),blood_mesh__map_values=a(coo.values()))
    data.update({'blood__'+n:a(getattr(b,n)) for n in BLOOD})
    for name,value in (('solid__previous_x',s.previous_x),('cycle__anchor',e.cycle.anchor),('cycle__resting_land',e.cycle.resting_land),('load__anchor',e.force_load.anchor)):
        if value is not None: data[name]=a(value)
    meta=dict(producer='afsei-two-way-three-field',schema=1,two_way=True,exterior_box_fluid=False,
        units='cm-g-s; V; GPB-ms-mV; pressure dyn/cm2; Pc mmHg',ep_config=asdict(driver.config),
        blood_config=asdict(driver.blood_driver.config),mechanical_config=asdict(driver.mechanics.config),
        vertex_count=driver.mesh.vertex_count,mesh_metadata=driver.mesh.metadata,blood_mesh_metadata=m.metadata,map_shape=list(m.mapping.shape),
        tick=e.tick,step=s.step,time=s.time,force_time=s.force_time,pressure_time=s.pressure_time,previous_dt=s.previous_dt,
        rejected_attempts=e.rejected_attempts,cycle=dict(mode=e.cycle.mode,closure_tick=e.cycle.closure_tick,freeze_tick=e.cycle.freeze_tick,stable_ticks=e.cycle.stable_ticks),
        active_load=dict(dt_s=e.force_load.dt_s,use_land=e.force_load.use_land,ramp_factor=e.force_load.ramp_factor),
        blood=dict(opened=list(b.opened),has_ejected=b.has_ejected,last_dt_s=b.last_dt_s),progress=progress or {})
    atomic(path,meta,data)


def load(path,device='cuda'):
    with np.load(path,allow_pickle=False) as f: data={n:f[n] for n in f.files}
    try:
        meta=json.loads(str(data.pop('metadata'))); sha=meta.pop('sha256')
        if (meta.get('producer')!='afsei-two-way-three-field' or meta.get('schema')!=1 or meta.get('two_way') is not True
            or meta.get('exterior_box_fluid') is not False
            or meta.get('units')!='cm-g-s; V; GPB-ms-mV; pressure dyn/cm2; Pc mmHg'
            or not REQUIRED<=set(data) or set(data)-REQUIRED-OPTIONAL
            or digest(meta,data)!=sha): raise ValueError('wrong three-field schema/producer/checksum; old IB/one-way checkpoints cannot resume')
        for n,v in data.items():
            if v.dtype!=(np.int64 if n in INTEGER else np.float64) or not np.isfinite(v).all(): raise ValueError('invalid checkpoint arrays')
        t=lambda n:torch.as_tensor(data[n],device=device); opt=lambda n:t(n) if n in data else None
        solid=ImportedSolidMesh(*(t('mesh__'+n) for n in MESH),meta['vertex_count'],meta['mesh_metadata'])
        mapping=torch.sparse_coo_tensor(t('blood_mesh__map_indices'),t('blood_mesh__map_values'),tuple(meta['map_shape']),check_invariants=True).coalesce().to_sparse_csr()
        blood_mesh=MovingCavityMesh(mapping,*(t('blood_mesh__'+n) for n in BLOOD_MESH),meta['blood_mesh_metadata'])
        d=ThreeFieldStepper(solid,decode_checkpoint_config(meta['ep_config']),blood_mesh,_decode(ALEConfig,meta['blood_config']),MechanicalConfig(**meta['mechanical_config']))
        s=SolidState(meta['step'],meta['time'],*(t('solid__'+n) for n in SOLID),meta['force_time'],meta['pressure_time'],opt('solid__previous_x'),meta['previous_dt'])
        e=CoupledState(meta['tick'],s,TissueState(*(t('ep__'+n) for n in ('voltage_v','gpb','land'))),
            StrainHistory(*(t('ep__'+n) for n in ('stretch','zs','zw'))),
            CycleState(**meta['cycle'],anchor=opt('cycle__anchor'),resting_land=opt('cycle__resting_land')),
            ActiveLoadState(*(t('load__'+n) for n in LOAD),**meta['active_load'],anchor=opt('load__anchor')),
            t('ep__ionic_current'),meta['rejected_attempts'])
        b=ALEState(meta['step'],meta['tick'],t('blood__solid_x'),t('blood__velocity'),t('blood__pressure'),
                t('blood__pc_mmhg'),t('blood__q_out_ml_s'),t('blood__port_pressure_cgs'),tuple(meta['blood']['opened']),
                meta['blood']['has_ejected'],t('blood__previous_solid_x'),meta['blood']['last_dt_s'])
        state=ThreeFieldState(e,b); validate(d,state)
        return d,state,meta['progress']
    except (KeyError,TypeError,IndexError,RuntimeError) as exc: raise ValueError('malformed three-field checkpoint') from exc
