"""Lossless geometry-coefficient storage benchmark for the same pressure V cycle."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.gao_tracts import prepare_cut_flow
from afsi_torch.mac.sealed_pressure import action
from afsi_torch.mac.execution import tensor_kernel


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('output',type=Path)
    args=parser.parse_args();torch.set_num_threads(4)
    d,s,_=restart.load(args.checkpoint,cardiac=True)
    e=d._fsi_stepper(d.maximum_ticks);prepare_cut_flow(e,s.fsi,s)
    solver=e.flow.pressure_solver;exact=s.fsi.pressure;rhs=solver.actions[0](exact)
    def timed(fn,count):
        torch.cuda.synchronize();start=perf_counter();result=None
        for _ in range(count):result=fn()
        torch.cuda.synchronize();return (perf_counter()-start)/count,result
    def measure():
        solver.solve(rhs);torch.cuda.synchronize()
        z=solver.precondition(rhs)
        prec,_=timed(lambda:solver.precondition(rhs),10)
        rows=[];xx=[]
        for name,guess in [('zero',None),('warm',.99*exact)]:
            seconds,(x,info)=timed(lambda:solver.solve(rhs,guess),3)
            rows.append(dict(initial=name,seconds=seconds,**info));xx.append(x)
        return dict(precondition_seconds=prec,solves=rows),z,xx
    baseline,z0,x0=measure()
    old_action=solver.actions[0];old_graph=solver.graph
    old_actions=solver.actions;old_smoothers=solver.smoothers
    compressed=[];new_actions=[];smoothers=[];storage=[]
    for h,m,diag in zip(solver.spacings,solver.masks,solver.diagonals):
        cm=[]
        for v in m:
            small=v.bool()
            if not torch.equal(small.to(v.dtype),v):
                small=v.float()
                if not torch.equal(small.to(v.dtype),v):small=v
            cm.append(small)
        cm=tuple(cm);compressed.append(cm)
        storage.append(dict(before_bytes=sum(v.numel()*v.element_size() for v in m),
            after_bytes=sum(v.numel()*v.element_size() for v in cm),dtypes=[str(v.dtype) for v in cm]))
        op=lambda x,h=h,m=cm:action(x,h,m)
        def smooth(x,b,h=h,m=cm,diag=diag):
            for _ in range(solver.options.smooth):x=x+(2/3)*(b-action(x,h,m))/diag
            return x
        new_actions.append(tensor_kernel(op,rhs.device));smoothers.append(tensor_kernel(smooth,rhs.device))
    solver.actions=new_actions;solver.smoothers=smoothers;solver.graph=None
    variant,z1,x1=measure()
    operator_error=float((solver.actions[0](exact)-old_action(exact)).abs().max())
    report=dict(scope='same FP64 pressure matrix/V cycle/PCG; geometry coefficients compacted only after exact roundtrip check',
        reference=baseline,compact_geometry=variant,coefficient_storage=storage,
        operator_max_abs_difference=operator_error,vcycle_relative_difference=float((z1-z0).norm()/z0.norm()),
        solution_relative_differences=[float((a-b).norm()/a.norm()) for a,b in zip(x0,x1)],
        speedups=[a['seconds']/b['seconds'] for a,b in zip(baseline['solves'],variant['solves'])])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
