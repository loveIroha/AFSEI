from pathlib import Path
import sys,importlib.util,json,time,argparse
import numpy as np,torch
ROOT=Path('/mnt/large2/gjh/AFSEI');here=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from afsi_torch.ep import cuda_tissue
from afsi_torch.ep.initial import GPB_INITIAL
parser=argparse.ArgumentParser();parser.add_argument('--compact',action='store_true');parser.add_argument('--fused',action='store_true');args=parser.parse_args()
if args.fused:
    from fused import native,production
elif args.compact:
    from compact import native,production
else:
    from candidates import native,production
reuse=args.compact or args.fused
report_path=here/('validation_fused.json' if args.fused else 'validation_compact.json' if args.compact else 'validation.json')
spec=importlib.util.spec_from_file_location('old_reference_cell',ROOT/'demo/ep_afsei_nodal_opencarp_matched/reference_cell.py')
oldmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmod)
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
old=oldmod.module();new=native();cuda_tissue.build();pold=torch.ops.afsei_ep_cuda;pnew=production()
init=np.load(ROOT/'demo/ep_afsei_nodal_opencarp_matched/native_initial_state.npz')['gpb']
gaps={};n=1024
v=torch.linspace(-86.23,78.12,n,device='cuda');y=torch.as_tensor(init,device='cuda').expand(-1,n).contiguous()
ys=y.clone();dy=torch.empty_like(y);dz=dy.clone();i=torch.empty_like(v);j=i.clone()
old.rhs(y,v,dy,i);new.rhs(y,v,dz,j)
gaps['native_RHS_max']=float((dy-dz).abs().max());gaps['native_current_max']=float((i-j).abs().max())
assert torch.allclose(dy,dz,rtol=1e-12,atol=1e-12) and torch.allclose(i,j,rtol=1e-12,atol=1e-12)
a=y.clone();b=y.clone();oa=y.clone();ob=y.clone();ga=torch.empty(n,dtype=torch.int64,device='cuda');gb=ga.clone()
py=torch.tensor(GPB_INITIAL,device='cuda').expand(n,-1).contiguous();pa=py.clone();pb=py.clone()
la=torch.tensor([0.,0.,0.,1.],device='cuda').expand(n,-1).contiguous();lb=la.clone()
stretch=torch.linspace(.92,1.16,n,device='cuda');zs=torch.linspace(-.1,.1,n,device='cuda');zw=zs*.3
# Replay a physiologically shaped voltage-clamp wave with heterogeneous delays.
delay=torch.linspace(0,1.8,n,device='cuda');maxnative=0.;maxprod=0.;maxland=0.;maxcur=0.
for k in range(100):
    t=k*.02;elapsed=t-delay
    V=torch.where(elapsed<0,-80.9763+.05*t,25+54*torch.exp(-torch.clamp(elapsed,min=0)/.35))
    old.advance(a,V,oa,i,ga,.002,10);new.advance(b,V,ob,j,gb,.002,10)
    assert not ga.any() and not gb.any()
    maxnative=max(maxnative,float((oa-ob).abs().max()));maxcur=max(maxcur,float((i-j).abs().max()))
    assert torch.allclose(oa,ob,rtol=2e-11,atol=2e-12) and torch.allclose(i,j,rtol=2e-11,atol=2e-11)
    a,oa=oa,a;b,ob=ob,b
    ra=pold.advance(pa,la,V,stretch,zs,zw,t,.002,10,True)
    rb=pnew.advance(pb,lb,V,stretch,zs,zw,t,.002,10,True)
    assert not ra[3].any() and not rb[3].any()
    maxprod=max(maxprod,float((ra[0]-rb[0]).abs().max()));maxland=max(maxland,float((ra[1]-rb[1]).abs().max()))
    assert all(torch.allclose(x,z,rtol=2e-11,atol=2e-11) for x,z in zip(ra[:3],rb[:3]))
    pa,la=ra[:2];pb,lb=rb[:2]
assert torch.equal(y,ys) and torch.equal(py,torch.tensor(GPB_INITIAL,device='cuda').expand(n,-1))
gaps.update(native_2ms_state_max=maxnative,production_2ms_state_max=maxprod,Land_2ms_state_max=maxland,
    native_2ms_current_max=maxcur,input_states_unchanged=True)
result=dict(validation_passed=True,gaps=gaps,nodes=n,trajectory_ms=2,methods_unchanged=True,precision='float64')
report_path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
# Full benchmark node count; identical state/voltage inputs for each pair.
N=1777377;yv=torch.as_tensor(init,device='cuda').expand(-1,N).contiguous();out=torch.empty_like(yv)
V=torch.linspace(-85.31,55.72,N,device='cuda');I=torch.empty_like(V);guard=torch.empty(N,dtype=torch.int64,device='cuda')
def timing(fn):
    for _ in range(2):fn()
    torch.cuda.synchronize();times=[]
    for _ in range(5):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record();value=fn();end.record();end.synchronize();times.append(begin.elapsed_time(end))
    return dict(median_ms=float(np.median(times)),min_ms=min(times),max_ms=max(times),samples_ms=times)
perf={}
saved=json.loads((here/'validation.json').read_text())['performance'] if reuse else {}
for name,m in [('baseline',old),('cached',new)]:
    perf['native_'+name]=saved['native_baseline'] if reuse and name=='baseline' else timing(lambda:m.advance(yv,V,out,I,guard,.002,10))
del yv,out;torch.cuda.empty_cache()
yp=torch.tensor(GPB_INITIAL,device='cuda').expand(N,-1).contiguous()
land=torch.tensor([0.,0.,0.,1.],device='cuda').expand(N,-1).contiguous();one=torch.ones_like(V);zero=torch.zeros_like(V)
for name,m in [('baseline',pold),('cached',pnew)]:
    perf['production_'+name]=saved['production_baseline'] if reuse and name=='baseline' else timing(lambda:m.advance(yp,land,V,one,zero,zero,0.,.002,10,True))
result['performance']=perf;result['timing_nodes']=N;result['gpu']=torch.cuda.get_device_name()
result['timing_scope']='CUDA events, identical heterogeneous frozen-voltage inputs, 10 unchanged SDIRK substeps, two warmups and five timed calls per implementation; production includes Land.'
result['speedups']={name:perf[name+'_baseline']['median_ms']/perf[name+'_cached']['median_ms'] for name in ['native','production']}
result['baseline_timing_reused_from_first_candidate']=reuse
if args.fused:
    small={}
    baselines=json.loads((here/'sdirk_diagnosis.json').read_text())['production_sizes']
    for count in [26889,229425]:
        sy=torch.tensor(GPB_INITIAL,device='cuda').expand(count,-1).contiguous()
        sl=torch.tensor([0.,0.,0.,1.],device='cuda').expand(count,-1).contiguous()
        sv=torch.linspace(-85.31,55.72,count,device='cuda');one=torch.ones_like(sv);zero=torch.zeros_like(sv)
        small[str(count)]=dict(optimized=timing(lambda:pnew.advance(sy,sl,sv,one,zero,zero,0.,.002,25,True)),baseline=baselines[str(count)]['baseline'])
    result['small_production']=small
report_path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
