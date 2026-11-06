"""Diagnostic only. Does NOT change production integrators or run a PDE."""
from pathlib import Path
import sys,importlib.util,json,time
import numpy as np,torch
from fused import native as optimized_native
ROOT=Path('/mnt/large2/gjh/AFSEI');here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference',ROOT/'demo/ep_afsei_nodal_opencarp_matched/reference_cell.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
old=ref.module();new=optimized_native()
initial=np.load(ROOT/'demo/ep_afsei_nodal_opencarp_matched/native_initial_state.npz')['gpb']
N=1777377;P=8;ids=np.linspace(0,N-1,P,dtype=int)
nativeV=np.memmap(ROOT/'results/opencarp_gpb_h0p0625_20261030/simulation_v2/vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N))
times=np.arange(2500)*.02
forcing=torch.as_tensor(np.column_stack([np.interp(times,np.arange(51),nativeV[:,i]) for i in ids]),device='cuda')
cases={'SDIRK_2us':(old.advance,.002,10),'SDIRK_0p5us':(old.advance,.0005,40),
    'Euler_2us':(new.euler,.002,10),'Euler_1us':(new.euler,.001,20)}
ys={key:torch.as_tensor(initial,device='cuda').expand(-1,P).contiguous() for key in cases}
outs={k:torch.empty_like(v) for k,v in ys.items()};cur={k:torch.empty(P,device='cuda') for k in ys}
guards={k:torch.empty(P,device='cuda',dtype=torch.int64) for k in ys}
currents={k:torch.empty((2500,P),device='cuda') for k in ys};snap=[];maxstate={k:0. for k in ys}
start=time.perf_counter()
for step,V in enumerate(forcing):
    for key,(fn,dt,substeps) in cases.items():
        fn(ys[key],V,outs[key],cur[key],guards[key],dt,substeps)
        currents[key][step]=cur[key];ys[key],outs[key]=outs[key],ys[key]
    if (step+1)%50==0:
        for key in ys:
            assert not guards[key].any() and torch.isfinite(ys[key]).all(),(key,step)
            scale=1e-9+1e-3*torch.maximum(ys[key].abs(),ys['SDIRK_0p5us'].abs())
            maxstate[key]=max(maxstate[key],float(((ys[key]-ys['SDIRK_0p5us']).abs()/scale).max()))
    if step in [0,49,99,249,499,999,1749,2499]:
        snap.append((ys['SDIRK_0p5us'].cpu().numpy().T.copy(),V.cpu().numpy().copy()))
    if (step+1)%500==0:print('Clamp replay', (step+1)*.02,'ms',flush=True)
reference=currents['SDIRK_0p5us'].cpu().numpy();metrics={}
for key in cases:
    a=currents[key].cpu().numpy();d=a-reference
    metrics[key]=dict(current_RMSE_ua_cm2=float(np.sqrt(np.mean(d*d))),
        current_relative_L2_error=float(np.linalg.norm(d)/np.linalg.norm(reference)),
        current_max_error_ua_cm2=float(abs(d).max()),max_state_scaled_error=maxstate[key],
        final_state_max_absolute_error=float((ys[key]-ys['SDIRK_0p5us']).abs().max()),all_finite=True)
# Local Jacobian stability margins at sampled valid states and frozen voltages.
S=np.concatenate([a for a,V in snap]);V=np.concatenate([V for a,V in snap]);m,ns=S.shape
eps=np.maximum(abs(S)*1e-6,1e-11)
plus=np.repeat(S[:,None,:],ns,axis=1);minus=plus.copy()
for j in range(ns):plus[:,j,j]+=eps[:,j];minus[:,j,j]-=eps[:,j]
assert np.isfinite(plus).all()
vp=torch.as_tensor(np.repeat(V,ns),device='cuda');fp=torch.empty((ns,m*ns),device='cuda');fm=fp.clone();I=torch.empty_like(vp)
old.rhs(torch.as_tensor(plus.reshape(-1,ns).T.copy(),device='cuda'),vp,fp,I)
old.rhs(torch.as_tensor(minus.reshape(-1,ns).T.copy(),device='cuda'),vp,fm,I)
J=(fp-fm).cpu().numpy().reshape(ns,m,ns).transpose(1,0,2)/(2*eps[:,None,:])
eigenvalues=np.linalg.eigvals(J);damped=eigenvalues[np.real(eigenvalues)<-1e-6]
limits=-2*np.real(damped)/abs(damped)**2;limit=float(limits.min())
report=dict(production_algorithm_changed=False,diagnostic_only=True,physical_model='Native Grandi FP64',
    forcing='8 actual openCARP mesh-node voltage traces, 50 ms, interpolated from saved 1 ms output; no PDE rerun',
    reference='SDIRK at 0.5 us; temporal refinement reference, not an exact solution',
    metrics=metrics,local_jacobian_samples=m,min_local_forward_Euler_step_limit_us=limit*1000,
    tested_Euler_step_us=2,local_Euler_linear_stability_condition_satisfied=bool(.002<=limit),
    caveat='Sampled frozen-voltage linear stability is necessary evidence, not a global stability or propagation-accuracy guarantee.',
    replay_wall_s=time.perf_counter()-start)
# Identical-input GPU throughput, no PDE: compare to existing baseline.
y=torch.as_tensor(initial,device='cuda').expand(-1,N).contiguous();z=torch.empty_like(y)
v=torch.linspace(-85.31,55.72,N,device='cuda');I=torch.empty_like(v);g=torch.empty(N,device='cuda',dtype=torch.int64)
for _ in range(2):new.euler(y,v,z,I,g,.002,10)
torch.cuda.synchronize();samples=[]
for _ in range(5):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();new.euler(y,v,z,I,g,.002,10);b.record();b.synchronize();samples.append(a.elapsed_time(b))
report['Euler_full_node_timing']=dict(nodes=N,median_ms=float(np.median(samples)),samples_ms=samples)
report['baseline_SDIRK_ms']=json.loads((here/'validation.json').read_text())['performance']['native_baseline']['median_ms']
report['raw_Euler_speedup_over_baseline_SDIRK']=report['baseline_SDIRK_ms']/float(np.median(samples))
(here/'explicit_feasibility.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
