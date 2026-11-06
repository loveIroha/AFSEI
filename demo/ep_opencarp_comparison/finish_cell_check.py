from pathlib import Path
import json,numpy as np,torch
from shared_cell import module
root=Path('/mnt/large2/gjh/AFSEI/demo/ep_opencarp_comparison/single_cell_validation')
a=np.fromfile(root/'BENCH_REG.Vm.bin',dtype=np.float64)
with np.load(root/'gpu_free_cell.npz') as z:post_stim=z['voltage_mv']
# bench applies the current pulse to Vm before evaluating the ionic model.
# Tissue Electrics applies it after the Ionics module and before diffusion.
module_=module();meta=json.loads((root.parent/'shared_cell_provenance.json').read_text())
y=torch.empty((meta['state_count'],1),device='cuda',dtype=torch.float64);v=torch.empty(1,device='cuda',dtype=torch.float64)
module_.initialize(y,v,.02);z=y.clone();current=v.clone();guard=torch.empty(1,device='cuda',dtype=torch.int64)
trace=[float(v)]
for k in range(2500):
    if k<100:v.add_(.02*35.71)
    module_.step(y,v,z,current,guard,.02);v.add_(current,alpha=-.02)
    y,z=z,y;trace.append(float(v))
b=np.array(trace)
np.savez(root/'gpu_native_bench_order.npz',time_ms=np.arange(len(b))*.02,voltage_mv=b)
report=json.loads((root/'initial_check.json').read_text())
report.update(native_samples=len(a),gpu_samples=len(b),native_first_values=a[:5].tolist(),gpu_first_values=b[:5].tolist(),
    native_bench_stimulus_order='stimulus before ionic evaluation; matched here; tissue benchmark keeps Ionics then stimulus then diffusion',
    native_vm_peak_mv=float(a.max()))
tests={}
for offset in [-1,0,1]:
    ia=max(0,-offset);ib=max(0,offset);n=min(len(a)-ia,len(b)-ib)
    tests[str(offset)]={'max_gap_mv':float(abs(a[ia:ia+n]-b[ib:ib+n]).max()),'rmse_mv':float(np.sqrt(np.mean((a[ia:ia+n]-b[ib:ib+n])**2)))}
report['sample_alignment_checks']=tests
report['free_voltage_check_passed']=min(t['max_gap_mv'] for t in tests.values())<.01
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='native_outputs'},indent=2))
if not report['free_voltage_check_passed']:raise RuntimeError('Native free-cell trajectory mismatch; inspect before full simulation')
