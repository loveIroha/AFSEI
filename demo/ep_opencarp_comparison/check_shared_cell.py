from pathlib import Path
import json,re,subprocess,time
import numpy as np,torch
from shared_cell import module
root=Path(__file__).resolve().parent
m=module();meta=json.loads((root/'shared_cell_provenance.json').read_text());ns=meta['state_count']
torch.set_num_threads(4)
cpu=torch.empty((ns,6),dtype=torch.float64);cv=torch.empty(6,dtype=torch.float64)
gpu=torch.empty_like(cpu,device='cuda');gv=torch.empty_like(cv,device='cuda')
m.initialize(cpu,cv,.02);m.initialize(gpu,gv,.02)
assert torch.equal(cpu,gpu.cpu()) and torch.equal(cv,gv.cpu())
ci=torch.empty_like(cv);gi=torch.empty_like(gv);cg=torch.empty(6,dtype=torch.int64);gg=torch.empty(6,device='cuda',dtype=torch.int64)
cn=cpu.clone();gn=gpu.clone();max_current=0.;max_state=0.
for k in range(100):
    v=torch.tensor([-85,-80,-40,-20,0,40],dtype=torch.float64)
    m.step(cpu,v,cn,ci,cg,.02);m.step(gpu,v.cuda(),gn,gi,gg,.02)
    max_current=max(max_current,float((ci-gi.cpu()).abs().max()))
    max_state=max(max_state,float((cn-gn.cpu()).abs().max()))
    cpu,cn=cn,cpu;gpu,gn=gn,gpu
assert max_current<1e-4 and max_state<1e-4,(max_current,max_state)
case=root/'single_cell_validation';case.mkdir(exist_ok=True)
native=['/mnt/large2/gjh/opencarp-runtime/v20.0/squashfs-root/usr/bin/bench','--imp','Grandi',
    '--dt','0.02','--duration','50','--stim-curr','35.71','--stim-dur','2','--stim-start','0',
    '--dt-out','0.02','--fout',str(case/'native'),'--validate']
p=subprocess.run(native,cwd=case,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(case/'native.log').write_text(re.sub(r'https://[^\s/@]+:[^\s/@]+@git\.opencarp\.org/','https://git.opencarp.org/',p.stdout))
if p.returncode:raise RuntimeError('Native bench validation run failed')
# Isolated cell: exactly the common current update and Euler Vm used by native limpet.
y=torch.empty((ns,1),device='cuda',dtype=torch.float64);v=torch.empty(1,device='cuda',dtype=torch.float64)
m.initialize(y,v,.02);z=y.clone();I=v.clone();g=torch.empty(1,device='cuda',dtype=torch.int64)
trace=[float(v)]
for step in range(2500):
    m.step(y,v,z,I,g,.02)
    v.add_((35.71 if step<100 else 0.)-I,alpha=.02)
    y,z=z,y;trace.append(float(v))
np.savez(case/'gpu_free_cell.npz',time_ms=np.arange(2501)*.02,voltage_mv=trace)
report=dict(cpu_gpu_fixed_voltage_max_current_gap=max_current,cpu_gpu_fixed_voltage_max_state_gap=max_state,
    state_count=ns,native_outputs=[p.name for p in case.iterdir()],gpu_free_cell_initial_mv=trace[0],gpu_free_cell_peak_mv=max(trace))
(case/'initial_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
