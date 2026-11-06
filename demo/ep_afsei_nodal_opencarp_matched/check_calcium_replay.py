"""Reference ODE replay under the completed Euler run's actual probe Vm.

No PDE is rerun. This separates ionic integration error from propagation.
"""
from pathlib import Path
import importlib.util,json,time,os
os.environ.update(CUDA_VISIBLE_DEVICES='1',CUDA_HOME='/mnt/large2/gjh/conda-envs/afsei',TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4')
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/mnt/large2/gjh/AFSEI');base=ROOT/'demo/ep_afsei_nodal_opencarp_matched'
out=ROOT/'results/ep_afsei_nodal_euler1us_20261031';dest=out/'comparison'
assert json.loads((out/'report.json').read_text())['status']=='completed'
spec=importlib.util.spec_from_file_location('original_reference',base/'reference_cell.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference);model=reference.module()
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
tr=np.load(out/'traces.npz');ct=np.load(out/'cell_traces.npz');initial=np.load(base/'native_initial_state.npz')
names=initial['state_names'].tolist();cai=names.index('Cai');n=len(tr['probe_ids'])
y=torch.as_tensor(initial['gpb']).expand(-1,n).contiguous();next_y=torch.empty_like(y)
current=torch.empty(n);guards=torch.empty(n,dtype=torch.int64)
ca=[y[cai].numpy().copy()*1000];currents=[];begin=time.perf_counter()
for k,v in enumerate(tr['voltage_mv'][:-1]):
    model.advance(y,torch.as_tensor(v),next_y,current,guards,.0005,40)
    if guards.any():raise RuntimeError(f'Reference ODE replay failed at {k}')
    ca.append(next_y[cai].numpy().copy()*1000);currents.append(current.numpy().copy());y,next_y=next_y,y
ca=np.array(ca);currents=np.array(currents);delta=ct['calcium_um']-ca
relative=float(np.linalg.norm(delta)/np.linalg.norm(ca));imax=float(abs(ct['ionic_current_ua_cm2']-currents).max())
current_relative=float(np.linalg.norm(ct['ionic_current_ua_cm2']-currents)/np.linalg.norm(currents))
final=np.load(out/'final_gpb_state.npy',mmap_mode='r')[:,tr['probe_ids']]
report=dict(reference='CPU SDIRK 0.5 us, frozen actual Euler Vm; same official GPB RHS/initial state',
    scope='4 recorded probe nodes for 50 ms, no PDE rerun; this is temporal ionic accuracy, not full spatial convergence or Land validation',
    calcium_relative_L2_difference=relative,calcium_max_absolute_difference_um=float(abs(delta).max()),
    calcium_final_max_difference_um=float(abs(delta[-1]).max()),
    current_relative_L2_difference=current_relative,current_max_absolute_difference_ua_cm2=imax,
    final_allstate_max_absolute_difference=float(abs(final-y.numpy()).max()),
    probe_ids=tr['probe_ids'].tolist(),wall_s=time.perf_counter()-begin)
np.savez(dest/'calcium_reference_replay.npz',time_ms=tr['time_ms'],calcium_reference_um=ca,
    calcium_euler_um=ct['calcium_um'],reference_interval_current_ua_cm2=currents,
    euler_interval_current_ua_cm2=ct['ionic_current_ua_cm2'],probe_ids=tr['probe_ids'])
(dest/'calcium_reference_report.json').write_text(json.dumps(report,indent=2))
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for j,ax in enumerate(axes.flat):
    ax.plot(tr['time_ms'],ca[:,j],color='#364553',label='SDIRK 0.5 us, same imposed Vm')
    ax.plot(tr['time_ms'],ct['calcium_um'][:,j],'--',color='#e0802b',label='Euler 1 us, full EP run')
    ax.set(title=f'Node {tr["probe_ids"][j]}',xlabel='Time [ms]',ylabel='Cai [uM]');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle(f'Ionic ODE accuracy under the same actual Vm | calcium relative L2 difference {relative*100:.4f}%',fontsize=13)
for ext in ['png','pdf']:fig.savefig(dest/f'calcium_comparison.{ext}',dpi=160)
print(json.dumps(report,indent=2),flush=True)
