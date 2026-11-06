"""Uncommitted one-interval operator diagnosis at an existing cut checkpoint."""
import argparse,json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep import mac_balanced_dynamic as dynamic
from afsi_torch.nonlinear import NonlinearFailure


class DiagnosticFinished(Exception):pass


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('output',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    d,s,_=restart.load(a.checkpoint,cardiac=True)
    original=dynamic.newton;report=dict(checkpoint=a.checkpoint,accepted_time_ms=s.fsi.time*1000,physical_state_published=False)
    def diagnose(residual,z,**kwargs):
        problem=residual.__self__;flow=problem.driver.flow;solver=flow.pressure_solver
        start=perf_counter()
        try:
            result=original(residual,z,**kwargs);report['trial_newton_converged']=True
        except NonlinearFailure as exc:
            result=exc.result;report.update(trial_newton_converged=False,trial_failure=str(exc))
        report['trial_newton_seconds']=perf_counter()-start
        report['trial_newton_history']=result.history
        report['trial_newton_tolerance']=result.tolerance
        base=result.x.clone();initial=s.fsi.pressure.clone();options=solver.cg_options
        def evaluate(value,reset=False):
            problem.last_z=None
            if reset:problem.pressure_initial=initial
            return problem.residual(value).clone()
        cases=[]
        for factor,reset in ((1.,False),(1.,True),(.01,False),(.001,False)):
            solver.cg_options=replace(options,rtol=options.rtol*factor,atol=options.atol*factor)
            problem.pressure_initial=initial
            values=[];pressure_info=[];start=perf_counter()
            for _ in range(5):
                values.append(evaluate(base,reset))
                pressure_info.append(problem.last_data[6]['pressure'])
            torch.cuda.synchronize();n=problem.mechanical_size
            cases.append(dict(pressure_rtol=solver.cg_options.rtol,pressure_atol=solver.cg_options.atol,
                reset_pressure_each_evaluation=reset,seconds=perf_counter()-start,
                residual_norms=[float(v.norm()) for v in values],
                max_repeat_vector_difference=float(torch.stack([v-values[0] for v in values]).norm(dim=1).max()),
                max_repeat_mechanical_difference=max(float((v[:n]-values[0][:n]).norm()) for v in values),
                max_repeat_cap_difference=max(float((v[n:]-values[0][n:]).norm()) for v in values),
                pressure_solves=pressure_info))
        report['repeatability']=cases
        solver.cg_options=replace(options,rtol=options.rtol*.001,atol=options.atol*.001)
        evaluate(base,True);action=problem.linearization(base)
        ids=torch.arange(len(base),device=base.device,dtype=base.dtype)
        v=torch.sin(ids);n=problem.mechanical_size;v[:n]/=n**.5
        v[n:]*=1333.22368421/problem.cap_scale
        jv=action(v);report['jacobian_fd']=[]
        for eps in (1e-3,1e-4,1e-5):
            fd=(evaluate(base+eps*v,True)-evaluate(base-eps*v,True))/(2*eps)
            report['jacobian_fd'].append(dict(epsilon=eps,relative_error=float((fd-jv).norm()/jv.norm()),
                mechanical_error=float((fd[:n]-jv[:n]).norm()),cap_error=float((fd[n:]-jv[n:]).norm())))
        solver.cg_options=options
        raise DiagnosticFinished()
    dynamic.newton=diagnose
    try:d.step(s,target_tick=s.tick+d.maximum_ticks,diagnostics=True)
    except DiagnosticFinished:pass
    finally:dynamic.newton=original
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='trial_newton_history'},indent=2))


if __name__=='__main__':main()
