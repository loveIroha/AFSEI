"""Isolated algebraic volume-increment trial; no physical step is published."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep import mac_balanced_dynamic as dynamic
from afsi_torch.ep.tract_storage_gpu import GPUTractStorage
from afsi_torch.nonlinear import NonlinearFailure


class Finished(Exception):pass


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('iterate');p.add_argument('output',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    driver,state,_=restart.load(a.checkpoint,cardiac=True)
    with np.load(a.iterate,allow_pickle=False) as z:
        if int(z['last_accepted_tick'])!=state.tick:raise ValueError('checkpoint mismatch')
        saved=torch.as_tensor(z['unknown'],device=state.fsi.x.device).clone()
    report=dict(physical_state_published=False,accepted_time_ms=state.fsi.time*1000)
    original_newton=dynamic.newton;original_discrete=GPUTractStorage.discrete

    def stable_discrete(storage,old,x):
        out=original_discrete(storage,old,x);c=storage.cached
        if c.get('reference',False) or c.get('stable_increment',False):return out
        inc=storage.stable_increment(c['r0'],c['r1'])
        if inc is None:return out
        numerator=inc-(c['gm']*c['dy']).sum()
        floor=64*torch.finfo(x.dtype).eps*torch.maximum(torch.maximum(abs(c['volume_old']),abs(c['volume_new'])),inc.new_tensor(1.))
        corrected=(c['square']>1e-20)&(abs(numerator)>floor)
        coefficient=torch.where(corrected,numerator/c['safe_square'],torch.zeros_like(numerator))
        g=c['gm']+coefficient*c['dy']
        c.update(numerator=numerator,coefficient=coefficient,corrected=corrected,gradient=g,jacobian=None,
                 stable_increment=True,volume_increment=inc)
        out=torch.zeros_like(x);out[storage.nodes]=g;return out

    def diagnose(residual,initial,**kwargs):
        problem=residual.__self__;storage=problem.cut_storage;solver=problem.driver.flow.pressure_solver
        pressure_seed=state.fsi.pressure.clone();n=problem.mechanical_size
        def evaluate(z):
            problem.last_z=None;problem.pressure_initial=pressure_seed.clone()
            return residual(z).clone()
        x=problem.unpack(saved)[1]
        original_discrete(storage,problem.state.x,x);c=storage.cached
        inc=storage.stable_increment(c['r0'],c['r1'])
        report['geometry_before']=dict(numerator=float(c['numerator']),square=float(c['square']),
            corrected=bool(c['corrected']),direct_increment=float(c['volume_new']-c['volume_old']),
            stable_increment=None if inc is None else float(inc),
            stable_numerator=None if inc is None else float(inc-(c['gm']*c['dy']).sum()))
        storage.cached=None;GPUTractStorage.discrete=stable_discrete
        r=evaluate(saved);report['stable_initial']=dict(total=float(r.norm()),mechanical=float(r[:n].norm()),cap=float(r[n:].norm()))
        report['pressure_options']=dict(rtol=solver.cg_options.rtol,atol=solver.cg_options.atol)
        started=perf_counter()
        try:
            result=original_newton(residual,saved.clone(),**dict(kwargs,linearization_factory=problem.linearization))
            ok,reason=True,None
        except NonlinearFailure as exc:result,ok,reason=exc.result,False,str(exc)
        r=evaluate(result.x);y,end_x,_=problem.unpack(result.x)
        dy=end_x-problem.state.x
        delta=storage.volume(end_x)-storage.volume(problem.state.x)
        work=float((storage.discrete(problem.state.x,end_x)*dy).sum())
        rows=problem.rows;velocity=problem.last_data[4];dt=problem.driver.flow.dt
        physical_cap=rows.flow(velocity)-rows.material_flow(y/dt)
        report.update(converged=ok,reason=reason,seconds=perf_counter()-started,
            residual=result.residual_norm,tolerance=result.tolerance,cold_residual=float(r.norm()),
            mechanical_residual=float(r[:n].norm()),cap_residual=float(r[n:].norm()),
            closed_cap_flux_ml_s=float(physical_cap[:-2].norm()),
            chain_rule_error_ml=work-delta,
            total_volume_rate_ml_s=float((problem.load.volume(end_x)-problem.load.volume(problem.state.x)+delta)/dt),
            history=result.history)
        raise Finished()

    dynamic.newton=diagnose
    try:driver.step(state,target_tick=state.tick+1,diagnostics=True)
    except Finished:pass
    finally:
        dynamic.newton=original_newton;GPUTractStorage.discrete=original_discrete
        a.output.parent.mkdir(exist_ok=True,parents=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
