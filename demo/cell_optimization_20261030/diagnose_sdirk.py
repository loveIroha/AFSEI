"""Separate unchanged-SDIRK implementation costs from algorithm RHS work."""
from pathlib import Path
import sys,importlib.util,json,gc
import numpy as np,torch
ROOT=Path('/mnt/large2/gjh/AFSEI');here=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from afsi_torch.ep import cuda_tissue
from afsi_torch.ep.initial import GPB_INITIAL
from staged_cell import NodeSDIRK
spec=importlib.util.spec_from_file_location('old_reference',ROOT/'demo/ep_afsei_nodal_opencarp_matched/reference_cell.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
native=reference.module();cuda_tissue.build();production=torch.ops.afsei_ep_cuda
init=torch.as_tensor(np.load(ROOT/'demo/ep_afsei_nodal_opencarp_matched/native_initial_state.npz')['gpb'],device='cuda')
n=257;y=init.expand(-1,n).contiguous();yp=torch.tensor(GPB_INITIAL,device='cuda').expand(n,-1).contiguous()
land=torch.tensor([0.,0.,0.,1.],device='cuda').expand(n,-1).contiguous()
sn=NodeSDIRK(n,native=True,dt_ms=.002,substeps=10)
sp=NodeSDIRK(n,native=False,dt_ms=.002,substeps=10,advance_land=True)
one=torch.ones(n,device='cuda');zero=torch.zeros_like(one);delay=torch.linspace(0,1.8,n,device='cuda')
a=y.clone();b=y.clone();oa=y.clone();current=zero.clone();guard=torch.empty(n,device='cuda',dtype=torch.int64)
pa=yp.clone();pb=yp.clone();la=land.clone();lb=land.clone();gaps=[0.,0.,0.,0.]
for step in range(100):
    t=step*.02;e=t-delay;V=torch.where(e<0,-80.9763+.05*t,25+54*torch.exp(-torch.clamp(e,min=0)/.35))
    native.advance(a,V,oa,current,guard,.002,10);nb=sn.advance(b,V)
    po=production.advance(pa,la,V,one,zero,zero,t,.002,10,True);pn=sp.advance(pb,V,lb,one,zero,zero)
    assert not guard.any() and not nb[3].any() and not po[3].any() and not pn[3].any()
    for j,(old,new) in enumerate([(oa,nb[0]),(current,nb[2]),(po[0],pn[0]),(po[1],pn[1])]):
        gaps[j]=max(gaps[j],float((old-new).abs().max()))
        assert torch.allclose(old,new,rtol=2e-10,atol=2e-11),(step,j,gaps[j])
    a,oa=oa,a;b=nb[0];pa,la=po[:2];pb,lb=pn[:2]
saved=b.clone();sn.advance(b,V);assert torch.equal(saved,b)
report=dict(validation_passed=True,gaps=dict(native_state=gaps[0],native_current=gaps[1],production_state=gaps[2],Land=gaps[3]),
    numerical_method='Unchanged SDIRK; each node keeps its own stopping flag; autonomous endpoint gradient reused',
    output_ownership_verified=True,production_default_changed=False)
(here/'sdirk_diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
del sn,sp,a,b,oa,pa,pb,po,pn,nb;gc.collect();torch.cuda.empty_cache()
def timeit(fn):
    for _ in range(2):fn()
    torch.cuda.synchronize();vals=[]
    for _ in range(5):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();result=fn();b.record();b.synchronize();vals.append(a.elapsed_time(b))
    return dict(median_ms=float(np.median(vals)),samples_ms=vals)
N=1777377;Y=init.expand(-1,N).contiguous();V=torch.linspace(-85.31,55.72,N,device='cuda')
d=torch.empty_like(Y);I=torch.empty_like(V)
node=NodeSDIRK(N,native=True,dt_ms=.002,substeps=10)
report['native_single_RHS']=timeit(lambda:native.rhs(Y,V,d,I))
report['native_staged_SDIRK']=timeit(lambda:node.advance(Y,V))
old=json.loads((here/'validation.json').read_text())['performance']
report['native_baseline_SDIRK']=old['native_baseline']
del node,Y,V,d,I;gc.collect();torch.cuda.empty_cache()
report['production_sizes']={}
for N,substeps in [(26889,25),(229425,25),(1777377,10)]:
    Y=torch.tensor(GPB_INITIAL,device='cuda').expand(N,-1).contiguous()
    V=torch.linspace(-85.31,55.72,N,device='cuda');one=torch.ones_like(V);zero=torch.zeros_like(V)
    L=torch.tensor([0.,0.,0.,1.],device='cuda').expand(N,-1).contiguous()
    node=NodeSDIRK(N,native=False,dt_ms=.002,substeps=substeps,advance_land=True)
    d=torch.empty_like(Y)
    item=dict(nodes=N,substeps=substeps,staged=timeit(lambda:node.advance(Y,V,L,one,zero,zero)),
        one_RHS=timeit(lambda:production.rhs(Y,V,0.)))
    item['baseline']=old['production_baseline'] if N==1777377 else timeit(lambda:production.advance(Y,L,V,one,zero,zero,0.,.002,substeps,True))
    item['speedup']=item['baseline']['median_ms']/item['staged']['median_ms']
    report['production_sizes'][str(N)]=item
    del node,Y,V,L,one,zero,d;gc.collect();torch.cuda.empty_cache()
report['native_speedup']=report['native_baseline_SDIRK']['median_ms']/report['native_staged_SDIRK']['median_ms']
report['timing_scope']='GPU event time including output publication, 2 warmups + 5 samples. Baselines at 1777377 reused from the first candidate; smaller production cases are newly measured.'
(here/'sdirk_diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
