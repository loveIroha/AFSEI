"""Atomic, numeric-only checkpoints for the distinct rigid-cavity subproblem."""
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .geometry import CavityDomain
from .flow import CavityFlowConfig,CavityFlowStepper,CavityFluidState
from ..mac.grid import MACGrid
from ..ep.windkessel import WindkesselParameters
from ..fluid.solvers import SolverOptions

FIELDS={'fluid','tag_0','tag_1','tag_2','frame_center','frame_basis','pressure',
        'velocity_0','velocity_1','velocity_2','pc_mmhg','q_out_ml_s','port_pressure_cgs'}
UNITS='cm-g-s; mmHg; ml/s; outward port flux'


def digest(meta,data):
    h=hashlib.sha256(json.dumps(meta,sort_keys=True,allow_nan=False).encode())
    for n in sorted(data):
        a=data[n]; h.update(n.encode()); h.update(a.dtype.str.encode())
        h.update(json.dumps(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def validate(driver,state):
    d=driver.domain; cfg=driver.config; g=d.grid
    if (type(state.step)!=int or state.step<0 or state.time_s!=state.step*cfg.dt_s
        or type(state.ports_open)!=bool or state.pressure.shape!=g.shape or state.pc_mmhg.shape!=()
        or state.q_out_ml_s.shape!=(2,) or state.port_pressure_cgs.shape!=(2,)):
        raise ValueError('invalid cavity clocks/shapes')
    g.check_velocity(state.velocity)
    tensors=(*state.velocity,state.pressure,state.pc_mmhg,state.q_out_ml_s,state.port_pressure_cgs)
    if any(v.dtype!=torch.float64 or v.device!=d.fluid.device or not torch.isfinite(v).all() for v in tensors):
        raise ValueError('finite float64 cavity tensors required')
    op=driver.operators(state.ports_open)
    if (any((u[~a]!=0).any() for u,a in zip(state.velocity,op.active))
        or (state.pressure[~d.fluid]!=0).any()
        or not torch.allclose(op.flux(state.velocity),state.q_out_ml_s,atol=1e-10,rtol=1e-10)
        or op.divergence(state.velocity).abs().max()>1e-7):
        raise ValueError('invalid cavity checkpoint flux, walls or incompressibility')
    if state.ports_open and cfg.one_way and (state.q_out_ml_s[1]<-1e-10 or state.q_out_ml_s[0]>1e-10):
        raise ValueError('invalid one-way port flow')
    if state.step>0:
        from ..ep.windkessel import MMHG_TO_DYN_CM2 as C
        wk=cfg.windkessel
        expected=torch.stack((state.pc_mmhg.new_tensor(cfg.atrium_mmhg)+cfg.mv_resistance*state.q_out_ml_s[0],
            state.pc_mmhg+(wk.rc_mmhg_s_per_ml+cfg.av_resistance)*state.q_out_ml_s[1]))*C
        if not torch.allclose(expected,state.port_pressure_cgs,rtol=1e-10,atol=1e-9):
            raise ValueError('invalid checkpoint valve/RCR boundary pressure')


def save(path,driver,state,progress=None):
    validate(driver,state); d=driver.domain
    array=lambda x:x.detach().cpu().numpy()
    data={name:array(getattr(d,name)) for name in ('fluid','frame_center','frame_basis')}
    data.update({f'tag_{c}':array(t) for c,t in enumerate(d.tags)})
    data.update({f'velocity_{c}':array(v) for c,v in enumerate(state.velocity)})
    data.update({n:array(getattr(state,n)) for n in ('pressure','pc_mmhg','q_out_ml_s','port_pressure_cgs')})
    meta=dict(producer='afsei-rigid-cavity-rcr',schema=1,units=UNITS,config=asdict(driver.config),
        grid=asdict(d.grid),domain_metadata=d.metadata,step=state.step,time_s=state.time_s,
        ports_open=state.ports_open,progress=progress or {})
    meta['sha256']=digest(meta,data); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream: np.savez_compressed(stream,metadata=json.dumps(meta,allow_nan=False),**data)
    tmp.replace(path)


def load(path,device='cuda'):
    with np.load(path,allow_pickle=False) as archive: data={n:archive[n] for n in archive.files}
    try:
        meta=json.loads(str(data.pop('metadata'))); checksum=meta.pop('sha256')
        if (meta.get('producer')!='afsei-rigid-cavity-rcr' or meta.get('schema')!=1
            or meta.get('units')!=UNITS or set(data)!=FIELDS or digest(meta,data)!=checksum):
            raise ValueError('wrong cavity checkpoint producer/schema/checksum; EP/FSI checkpoints are separate')
        for n,a in data.items():
            expected=np.bool_ if n=='fluid' else np.int64 if n.startswith('tag_') else np.float64
            if a.dtype!=expected or not np.isfinite(a).all(): raise ValueError('invalid cavity array dtype/value')
        raw=meta['config'].copy(); raw['windkessel']=WindkesselParameters(**raw['windkessel']); raw['solver']=SolverOptions(**raw['solver'])
        cfg=CavityFlowConfig(**raw); grid=MACGrid(**{k:tuple(v) for k,v in meta['grid'].items()})
        tensor=lambda n:torch.as_tensor(data[n],device=device)
        domain=CavityDomain(grid,tensor('fluid'),tuple(tensor(f'tag_{c}') for c in range(3)),tensor('frame_center'),tensor('frame_basis'),meta['domain_metadata'])
        driver=CavityFlowStepper(domain,cfg)
        state=CavityFluidState(meta['step'],meta['time_s'],tuple(tensor(f'velocity_{c}') for c in range(3)),
            *(tensor(n) for n in ('pressure','pc_mmhg','q_out_ml_s','port_pressure_cgs')),meta['ports_open'])
        validate(driver,state); return driver,state,meta['progress']
    except (KeyError,TypeError,RuntimeError) as exc:
        raise ValueError('invalid cavity checkpoint') from exc
