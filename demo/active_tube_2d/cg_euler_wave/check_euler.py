"""GPU check of GPB57 Euler against independent one-step RHS and Land."""
from pathlib import Path
import sys,json
import torch
R=Path('/mnt/large2/gjh/AFSEI');sys.path.insert(0,str(R/'src'))
from afsi_torch.ep import cuda_tissue,cuda_euler
from afsi_torch.ep.tissue import initial_state
from afsi_torch.ep.land import land_ode_step
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
v=torch.tensor([-86.,-81.5456,-60.,-40.,.1,30.,50.],device='cuda');state=initial_state(v)
y=state.gpb;l=state.land;ones=torch.ones_like(v);z=torch.zeros_like(v);dt=.001
saved=y.clone();dy,_=cuda_tissue.rhs(y,v,0.)
oy,ol,oi,guards=cuda_euler.advance(y,l,v,ones,z,z,0.,dt,1,True)
ref=y+dt*dy;lr=land_ode_step(l,ref[:,37]*1000,ones,z,z,dt);_,ir=cuda_tissue.rhs(ref,v,dt)
assert torch.equal(saved,y) and int(guards.max())==0
g=float((oy-ref).abs().max());lg=float((ol-lr).abs().max());ig=float((oi-ir).abs().max())
assert g<1e-12 and lg<1e-12 and ig<1e-8,(g,lg,ig)
# 2ms voltage-clamp at seven voltages and independent 0.5us step reference.
a=cuda_euler.advance(y,l,v,ones,z,z,0.,.001,2000,True)
b=cuda_euler.advance(y,l,v,ones,z,z,0.,.0005,4000,True)
assert int(a[3].max())==0 and int(b[3].max())==0
assert torch.isfinite(a[0]).all() and (a[0][:,30:38]>0).all()
rel=lambda x,y:float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp_min(1e-20))
report=dict(passed=True,one_step_GPB_max_abs=g,one_step_Land_max_abs=lg,one_step_current_max_abs=ig,
    clamp_2ms_state_relative_L2=rel(a[0],b[0]),clamp_2ms_land_relative_L2=rel(a[1],b[1]),clamp_2ms_current_relative_L2=rel(a[2],b[2]),
    scope='GPU original GPB57 and Land, Euler1us vs0.5us at frozen voltages; not a complete FSI accuracy verification')
assert report['clamp_2ms_current_relative_L2']<.02 and report['clamp_2ms_land_relative_L2']<.02
# Original RHS is undefined at exactly these removable singularities.
special=torch.tensor([0.,-5.,-47.13],device='cuda');s=initial_state(special);u=torch.ones_like(special);zero=torch.zeros_like(u)
sol=cuda_euler.advance(s.gpb,s.land,special,u,zero,zero,0.,dt,1,True)
assert int(sol[3].max())==0 and all(torch.isfinite(x).all() for x in sol[:3])
plus=cuda_euler.advance(s.gpb,s.land,special+1e-4,u,zero,zero,0.,dt,1,True)
minus=cuda_euler.advance(s.gpb,s.land,special-1e-4,u,zero,zero,0.,dt,1,True)
error=float((sol[2]-.5*(plus[2]+minus[2])).abs().max());assert error<1e-5,error
report['removable_singularities_current_symmetric_limit_max_abs']=error
(R/'results/tube_cg_euler_wave_20261031/euler_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
