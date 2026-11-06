"""Same masked pressure matrix and V cycle, different smoothing kernel scheduling."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.gao_tracts import prepare_cut_flow
from afsi_torch.mac.execution import tensor_kernel


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('output',type=Path)
    args=p.parse_args();torch.set_num_threads(4)
    d,s,_=restart.load(args.checkpoint,cardiac=True)
    e=d._fsi_stepper(d.maximum_ticks);prepare_cut_flow(e,s.fsi,s)
    solver=e.flow.pressure_solver;exact=s.fsi.pressure;rhs=solver.actions[0](exact)
    def timed(fn,n):
        torch.cuda.synchronize();started=perf_counter();out=None
        for _ in range(n):out=fn()
        torch.cuda.synchronize();return (perf_counter()-started)/n,out
    def measure():
        solver.solve(rhs);torch.cuda.synchronize()
        z=solver.precondition(rhs)
        prec,_=timed(lambda:solver.precondition(rhs),10)
        rows=[];solutions=[]
        for name,guess in [('zero',None),('warm',.99*exact)]:
            seconds,(x,info)=timed(lambda:solver.solve(rhs,guess),3)
            rows.append(dict(initial=name,seconds=seconds,**info));solutions.append(x)
        return dict(precondition_seconds=prec,solves=rows),z,solutions
    baseline,z0,x0=measure()
    old_graph=solver.graph;old_smoothers=solver.smoothers
    split=[]
    for op,diag in zip(solver.actions,solver.diagonals):
        def single(x,b,op=op,diag=diag):return x+(2/3)*(b-op(x))/diag
        kernel=tensor_kernel(single,rhs.device)
        def smooth(x,b,kernel=kernel):
            for _ in range(solver.options.smooth):x=kernel(x,b)
            return x
        split.append(smooth)
    solver.smoothers=split;solver.graph=None
    variant,z1,x1=measure()
    report=dict(scope='pressure-only execution benchmark; same matrix/relaxation/coarsening/PCG tolerances',
        shape=list(rhs.shape),reference=baseline,split_sweeps=variant,
        vcycle_relative_difference=float((z1-z0).norm()/z0.norm()),
        solution_relative_differences=[float((a-b).norm()/a.norm()) for a,b in zip(x0,x1)],
        speedups=[a['seconds']/b['seconds'] for a,b in zip(baseline['solves'],variant['solves'])])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
