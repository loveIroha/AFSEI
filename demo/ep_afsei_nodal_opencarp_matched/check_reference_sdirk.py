from pathlib import Path
import importlib.util,json,sys,time
import numpy as np,torch
root=Path(__file__).resolve().parent
old=root.parent/'ep_opencarp_comparison'
spec=importlib.util.spec_from_file_location('native_grandi_initialization',old/'shared_cell.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
from reference_cell import module
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
meta=json.loads((root/'reference_cell_provenance.json').read_text());names=meta['states'];ns=len(names)
oldnames=[n for _,n in json.loads((old/'shared_cell_provenance.json').read_text())['state_fields']]
indices=[oldnames.index(n) for n in names];cai=names.index('Cai')
native_model=native.module();m=module()
yall=torch.empty((len(oldnames),1));v0=torch.empty(1);native_model.initialize(yall,v0,.02)
y0=yall[indices].clone();y0[cai]/=1000
np.savez(root/'native_initial_state.npz',gpb=y0.numpy(),voltage_mv=v0.numpy(),state_names=np.array(names))
V=torch.tensor([-85.,-80.,-40.,-20.,0.,40.]);y=y0.expand(-1,len(V)).contiguous()
d=torch.empty_like(y);I=torch.empty_like(V);m.rhs(y,V,d,I)
yd=y.cuda();vd=V.cuda();dd=yd.clone();Id=vd.clone();m.rhs(yd,vd,dd,Id)
rhs_gap=float((d-dd.cpu()).abs().max());current_gap=float((I-Id.cpu()).abs().max())
assert rhs_gap<1e-7 and current_gap<1e-9,(rhs_gap,current_gap)
rl=y.clone();rI=I.clone();dh=1e-7;m.rl_check(y,V,rl,rI,dh)
fd=(rl-y)/dh;fd_ratio=float(((fd-d).abs()/(1e-6+1e-3*torch.maximum(fd.abs(),d.abs()))).max())
assert float((rI-I).abs().max())<1e-10
assert fd_ratio<=1.,fd_ratio
z=y.clone();zD=yd.clone();g=torch.empty(len(V),dtype=torch.int64);gD=g.cuda()
m.advance(y,V,z,I,g,.002,10);m.advance(yd,vd,zD,Id,gD,.002,10)
step_gap=float((z-zD.cpu()).abs().max());step_current_gap=float((I-Id.cpu()).abs().max())
assert step_gap<1e-7 and step_current_gap<1e-8 and not g.any() and not gD.any()
# Independent RK4 check at a fixed Vm; reference only, not the benchmark method.
Vr=torch.tensor([-80.,-20.,40.]);yr=y0.expand(-1,len(Vr)).contiguous();duration=.02;h=.00002
def rhs(value):
    dy=torch.empty_like(value);cur=torch.empty(len(Vr));m.rhs(value,Vr,dy,cur);return dy
for k in range(round(duration/h)):
    k1=rhs(yr);k2=rhs(yr+h/2*k1);k3=rhs(yr+h/2*k2);k4=rhs(yr+h*k3)
    yr=yr+h/6*(k1+2*k2+2*k3+k4)
sd=y0.expand(-1,len(Vr)).contiguous();c=torch.empty(len(Vr));guard=torch.empty(len(Vr),dtype=torch.int64)
errors=[]
for hsd in [.002,.001]:
    final=sd.clone();m.advance(sd,Vr,final,c,guard,hsd,round(duration/hsd))
    errors.append(float(torch.linalg.vector_norm(final-yr)))
assert errors[1]<errors[0],errors
report=dict(passed=True,state_count=ns,initial_voltage_mv=float(v0[0]),initial_calcium_um=float(y0[cai,0])*1000,
    rhs_cpu_gpu_max_gap=rhs_gap,current_cpu_gpu_max_gap=current_gap,
    official_RL_infinitesimal_derivative_error_ratio=fd_ratio,
    sdirk_cpu_gpu_max_state_gap=step_gap,sdirk_cpu_gpu_max_current_gap=step_current_gap,
    sdirk_vs_RK4_state_errors_at_h002_h001=errors,
    numerical_method='Original AFSEI SDIRK arithmetic, FP64; RK4/RL used only for independent checks',
    physical_model='Official Grandi RHS and complete native initial state; exact unit conversion for Cai',
    production_gpb_modified=False)
(root/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
