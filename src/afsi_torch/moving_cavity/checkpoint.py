"""Numeric-only moving-flow and atomic paired one-way EP/ALE checkpoints."""
from dataclasses import asdict
from pathlib import Path
import json,tempfile
import numpy as np
import torch
from .geometry import MovingCavityMesh,swept_geometry
from .flow import ALEConfig,ALEStepper,ALEState
from .operators import ALEOperators
from .bridge import OneWayBridge,OneWayState
from ..circulation.checkpoint import digest
from ..ep.windkessel import WindkesselParameters
from ..fluid.solvers import SolverOptions

INTEGER={'cells','faces','cell_faces','tags','left','right','map_indices'}
FIELDS=INTEGER|{'map_values','signs','reference_solid','solid_x','previous_solid_x','velocity','pressure','pc_mmhg','q_out_ml_s','port_pressure_cgs'}


def atomic(path,meta,data):
    meta=dict(meta); meta['sha256']=digest(meta,data)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream: np.savez_compressed(stream,metadata=json.dumps(meta,allow_nan=False),**data)
    tmp.replace(path)


def validate(driver,state):
    m=driver.mesh; c=driver.config
    if (type(state.step)!=int or state.step<0 or type(state.tick)!=int or state.tick<round(c.start_time_s/c.clock_dt_s)
        or type(state.has_ejected)!=bool or len(state.opened)!=2 or any(type(v)!=bool for v in state.opened)
        or state.previous_solid_x is None or state.previous_solid_x.shape!=state.solid_x.shape
        or not np.isfinite(state.last_dt_s) or state.last_dt_s<=0 or abs(state.last_dt_s/c.clock_dt_s-round(state.last_dt_s/c.clock_dt_s))>1e-7
        or state.velocity.shape!=(len(m.faces),3) or state.pressure.shape!=(len(m.cells),)
        or state.pc_mmhg.shape!=() or state.q_out_ml_s.shape!=(2,) or state.port_pressure_cgs.shape!=(2,)):
        raise ValueError('invalid moving-cavity clock/shapes')
    for x in (state.solid_x,state.previous_solid_x,state.velocity,state.pressure,state.pc_mmhg,state.q_out_ml_s,state.port_pressure_cgs):
        if x.dtype!=torch.float64 or x.device!=m.reference_solid.device or not torch.isfinite(x).all(): raise ValueError('invalid moving-cavity tensor')
    g=swept_geometry(m,state.previous_solid_x,state.solid_x,state.last_dt_s); op=ALEOperators(m,g,state.opened)
    scale=max(state.q_out_ml_s.abs().max().item(),1.)
    if (not torch.allclose(op.flux(state.velocity),state.q_out_ml_s,atol=1e-8,rtol=1e-9)
        or op.divergence(state.velocity).abs().max()>1e-7*scale
        or not torch.allclose(state.velocity[op.fixed],g.grid_velocity[op.fixed],rtol=1e-10,atol=1e-10)):
        raise ValueError('checkpoint violates moving-wall flux or incompressibility')


def save(path,driver,state,progress=None):
    validate(driver,state); m=driver.mesh; arr=lambda x:x.detach().cpu().numpy()
    data={n:arr(getattr(m,n)) for n in ('cells','faces','cell_faces','signs','tags','left','right','reference_solid')}
    coo=m.mapping.to_sparse_coo().coalesce(); data.update(map_indices=arr(coo.indices()),map_values=arr(coo.values()))
    data.update({n:arr(getattr(state,n)) for n in ('solid_x','previous_solid_x','velocity','pressure','pc_mmhg','q_out_ml_s','port_pressure_cgs')})
    meta=dict(producer='afsei-moving-cavity-ale',schema=1,units='cm-g-s; mmHg; outward relative ml/s',
        config=asdict(driver.config),mesh_metadata=m.metadata,map_shape=list(m.mapping.shape),step=state.step,tick=state.tick,
        opened=list(state.opened),has_ejected=state.has_ejected,last_dt_s=state.last_dt_s,progress=progress or {},two_way=False)
    atomic(path,meta,data)


def load(path,device='cuda'):
    with np.load(path,allow_pickle=False) as f: data={n:f[n] for n in f.files}
    try:
        meta=json.loads(str(data.pop('metadata'))); checksum=meta.pop('sha256')
        if (meta.get('producer')!='afsei-moving-cavity-ale' or meta.get('schema')!=1 or meta.get('two_way') is not False
            or meta.get('units')!='cm-g-s; mmHg; outward relative ml/s' or set(data)!=FIELDS or digest(meta,data)!=checksum):
            raise ValueError('wrong moving-cavity producer/schema/checksum')
        for n,a in data.items():
            if a.dtype!=(np.int64 if n in INTEGER else np.float64) or not np.isfinite(a).all(): raise ValueError('invalid ALE array')
        t=lambda n:torch.as_tensor(data[n],device=device)
        raw=meta['config']; raw['windkessel']=WindkesselParameters(**raw['windkessel']); raw['solver']=SolverOptions(**raw['solver'])
        cfg=ALEConfig(**raw)
        mapping=torch.sparse_coo_tensor(t('map_indices'),t('map_values'),tuple(meta['map_shape']),check_invariants=True).coalesce().to_sparse_csr()
        m=MovingCavityMesh(mapping,*(t(n) for n in ('cells','faces','cell_faces','signs','tags','left','right','reference_solid')),meta['mesh_metadata'])
        driver=ALEStepper(m,cfg)
        state=ALEState(meta['step'],meta['tick'],*(t(n) for n in ('solid_x','velocity','pressure','pc_mmhg','q_out_ml_s','port_pressure_cgs')),
            tuple(meta['opened']),meta['has_ejected'],t('previous_solid_x'),meta['last_dt_s'])
        validate(driver,state); return driver,state,meta['progress']
    except (KeyError,TypeError,RuntimeError) as exc: raise ValueError('invalid moving-cavity checkpoint') from exc


def save_pair(path,bridge,state,progress=None):
    from ..ep import checkpoint as epcheck
    if state.ep.tick!=state.blood.tick or not torch.equal(state.ep.fsi.x,state.blood.solid_x): raise ValueError('paired checkpoint clocks/geometries differ')
    with tempfile.TemporaryDirectory(prefix='afsei-pair-') as d:
        ep_path=Path(d)/'ep.npz'; blood_path=Path(d)/'blood.npz'
        epcheck.save(ep_path,bridge.ep_driver,state.ep); save(blood_path,bridge.blood_driver,state.blood)
        data={}
        for prefix,p in (('ep__',ep_path),('ale__',blood_path)):
            with np.load(p,allow_pickle=False) as f: data.update({prefix+n:f[n] for n in f.files})
    atomic(path,dict(producer='afsei-one-way-ep-ale',schema=1,two_way=False,progress=progress or {}),data)


def load_pair(path,device='cuda'):
    from ..ep import checkpoint as epcheck
    with np.load(path,allow_pickle=False) as f: data={n:f[n] for n in f.files}
    meta=json.loads(str(data.pop('metadata'))); checksum=meta.pop('sha256')
    if (meta.get('producer')!='afsei-one-way-ep-ale' or meta.get('schema')!=1 or meta.get('two_way') is not False
        or digest(meta,data)!=checksum or any(not n.startswith(('ep__','ale__')) for n in data)
        or 'ep__metadata' not in data or 'ale__metadata' not in data): raise ValueError('wrong paired producer/schema/checksum')
    with tempfile.TemporaryDirectory(prefix='afsei-pair-') as d:
        paths=[]
        for prefix,name in (('ep__','ep.npz'),('ale__','ale.npz')):
            p=Path(d)/name; np.savez_compressed(p,**{n[len(prefix):]:a for n,a in data.items() if n.startswith(prefix)}); paths.append(p)
        ep,e,_=epcheck.load(paths[0],device); fluid,b,_=load(paths[1],device)
    bridge=OneWayBridge(ep,fluid)
    if e.tick!=b.tick or not torch.equal(e.fsi.x,b.solid_x): raise ValueError('paired endpoint clocks/geometries differ')
    return bridge,OneWayState(e,b),meta['progress']
