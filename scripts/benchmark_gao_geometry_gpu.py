"""Same exact clipped P1 volume/gradient/Jacobian, CPU versus GPU."""
import argparse,json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.ep.gao_tracts import LVTractDomain
from afsi_torch.ep.tract_storage import TractStorage
from afsi_torch.ep.tract_storage_gpu import GPUTractStorage


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('output',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    d,s,_=restart.load(a.checkpoint,cardiac=True);e=d._fsi_stepper(d.maximum_ticks)
    cap=outlet_rows(e,s.fsi.x).prepared.cap;domain=LVTractDomain(e.grid,cap,d.gao_tract_anchor,mode=s.mode)
    old,x=s.fsi.previous_x,s.fsi.x;v=torch.cos(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x)
    report=dict(scope='read-only exact geometry benchmark; no physical step',checkpoint=a.checkpoint,backends={})
    results=[]
    for cls in (TractStorage,GPUTractStorage):
        storage=cls(cap,domain)
        def gradient():storage.cached=None;return storage.discrete(old,x)
        def hessian():storage.cached['jacobian']=None;return storage.derivative(old,x)
        def timed(fn,count=3):
            torch.cuda.synchronize();t=perf_counter()
            for _ in range(count):fn()
            torch.cuda.synchronize();return (perf_counter()-t)/count
        g=gradient();action=hessian();hv=action(v);vol=storage.volume(x)
        report['backends'][cls.__name__]=dict(volume_seconds=timed(lambda:storage.volume(x)),gradient_seconds=timed(gradient),jacobian_seconds=timed(hessian),volume_ml=vol,topology_builds=getattr(storage,'topology_builds',None),fallback_count=getattr(storage,'fallback_count',None))
        results.append((g,hv,vol))
    report['differences']=dict(gradient_max=float((results[0][0]-results[1][0]).abs().max()),jacobian_action_max=float((results[0][1]-results[1][1]).abs().max()),volume_ml=abs(results[0][2]-results[1][2]))
    torch.testing.assert_close(results[0][0],results[1][0],rtol=1e-7,atol=1e-10)
    torch.testing.assert_close(results[0][1],results[1][1],rtol=1e-6,atol=1e-8)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
