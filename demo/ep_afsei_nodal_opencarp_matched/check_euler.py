"""Independent CPU RHS recurrence versus CUDA Euler; no PDE rerun."""
from pathlib import Path
import sys,json,importlib.util
import numpy as np,torch
ROOT=Path('/mnt/large2/gjh/AFSEI');base=ROOT/'demo/ep_afsei_nodal_opencarp_matched'
sys.path.insert(0,str(ROOT/'demo/cell_optimization_20261030'))
from fused import native
spec=importlib.util.spec_from_file_location('original_grandi',base/'reference_cell.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
old=reference.module();new=native();n=257
initial=torch.as_tensor(np.load(base/'native_initial_state.npz')['gpb'])
y=initial.expand(-1,n).contiguous();v=torch.linspace(-85.21,77.06,n)
state=y.clone();summed=torch.zeros(n);dy=torch.empty_like(y);cur=torch.empty(n)
for _ in range(20):
    old.rhs(state,v,dy,cur);state=state+.001*dy
    old.rhs(state,v,dy,cur);summed+=cur
yg=y.cuda();vg=v.cuda();out=torch.empty_like(yg);I=torch.empty_like(vg);guards=torch.empty(n,device='cuda',dtype=torch.int64)
new.euler(yg,vg,out,I,guards,.001,20)
gap=float((out.cpu()-state).abs().max());igap=float((I.cpu()-summed/20).abs().max())
assert torch.allclose(out.cpu(),state,rtol=1e-12,atol=1e-12),gap
assert torch.allclose(I.cpu(),summed/20,rtol=1e-12,atol=1e-11),igap
assert not guards.any() and torch.equal(yg.cpu(),y)
# Final nonpositive concentrations are reported, never silently accepted.
names=np.load(base/'native_initial_state.npz')['state_names'].tolist();bad=y.clone();bad[names.index('Ki')]=-1
new.euler(bad.cuda(),vg,out,I,guards,.001,1)
assert (guards>0).all()
report=dict(passed=True,nodes=n,ode_dt_us=1,substeps=20,maximum_state_difference=gap,
    maximum_interval_current_difference=igap,input_unchanged=True,positive_concentration_guard_verified=True,
    reference='Independent CPU recurrence using original official RHS; explicit state update then endpoint ionic current')
(base/'euler_validation_20261031.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
