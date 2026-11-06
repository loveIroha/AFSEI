"""Read-only timing of the added exact tract geometry operations."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.ep.gao_tracts import LVTractDomain
from afsi_torch.ep.tract_storage import TractStorage


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('output',type=Path)
    args=p.parse_args();torch.set_num_threads(4)
    d,s,_=restart.load(args.checkpoint,cardiac=True);e=d._fsi_stepper(d.maximum_ticks)
    cap=outlet_rows(e,s.fsi.x).prepared.cap
    domain=LVTractDomain(e.grid,cap,d.gao_tract_anchor,mode=s.mode)
    storage=TractStorage(cap,domain);old=s.fsi.previous_x;x=s.fsi.x
    def timed(fn,count=3):
        torch.cuda.synchronize();start=perf_counter()
        for _ in range(count):fn()
        torch.cuda.synchronize();return (perf_counter()-start)/count
    def gradient():storage.cached=None;storage.discrete(old,x)
    gradient();storage.derivative(old,x)
    grad_seconds=timed(gradient)
    def hessian():storage.cached['jacobian']=None;storage.derivative(old,x)
    hess_seconds=timed(hessian)
    volume_seconds=timed(lambda:storage.volume(x))
    report=dict(scope='read-only geometry microprofile; no physical step',checkpoint=args.checkpoint,
        rim_nodes=len(storage.nodes),volume_seconds=volume_seconds,
        finite_step_gradient_seconds=grad_seconds,geometry_jacobian_seconds=hess_seconds)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
