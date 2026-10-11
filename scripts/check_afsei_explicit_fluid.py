"""Isolated explicit-viscosity MAC projection probe; NOT an explicit FSI driver.

Use the actual LV checkpoint velocity. Freeze the common advective departure
and apply no added force. Both candidates retain the original pressure matrix.
This measures fluid work and checks diffusion/CFL bounds, not elastic stability.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.mac.execution import tensor_kernel
from afsi_torch.mac.open_boundary import velocity_laplacian
from afsi_torch.mac.grid import divergence


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=30)
    a=p.parse_args()
    if a.repeats<1: p.error('positive repeat count required')
    a.output.mkdir(parents=True,exist_ok=False)
    d,state,_=restart.load(a.input); rows=[]
    for dt in (1e-4,5e-5):
        e=d._fsi_stepper(round(dt/d.protocol.dt_ep_s)); f=e.flow; grid=e.grid
        departure=f.advect(state.fsi.velocity)
        predict=tensor_kernel(lambda u: tuple(v+dt*f.mu/f.rho*velocity_laplacian(v,c,grid.spacing)
            for c,v in enumerate(u)),state.fsi.x.device)
        def explicit():
            star=predict(departure)
            pressure,info=f.pressure_solver.solve(-f.rho/dt*divergence(star,grid.spacing))
            return f._project(star,pressure),pressure,info
        def backward(): return f.solve_rhs(departure)
        times={}; results={}
        for name,operation in (('backward_euler',backward),('explicit_viscosity_projection',explicit)):
            for _ in range(3): operation()
            torch.cuda.synchronize(); start=perf_counter()
            for _ in range(a.repeats): result=operation()
            torch.cuda.synchronize()
            times[name]=(perf_counter()-start)/a.repeats
            results[name]=result
        u=results['explicit_viscosity_projection'][0]; reference=results['backward_euler'][0]
        dt_limit=f.rho/(2*f.mu*sum(1/h**2 for h in grid.spacing))
        cfl=dt*sum(v.abs().max().item()/h for v,h in zip(state.fsi.velocity,grid.spacing))
        div=divergence(u,grid.spacing)
        delta=sum((v-w).square().sum() for v,w in zip(u,reference)).sqrt()
        norm=sum(v.square().sum() for v in reference).sqrt()
        rows.append(dict(dt_s=dt,diffusion_dt_limit_s=dt_limit,diffusion_number=dt/dt_limit,
            old_velocity_cfl=cfl,seconds=times,fluid_block_speedup=times['backward_euler']/times['explicit_viscosity_projection'],
            relative_velocity_difference=delta.item()/max(norm.item(),1e-30),
            max_divergence_s_inv=div.abs().max().item(),pressure_check=results['explicit_viscosity_projection'][2],
            finite_fields=all(torch.isfinite(v).all().item() for v in (*u,results['explicit_viscosity_projection'][1]))))
    report=dict(scope=__doc__,input=str(a.input),rows=rows,explicit_full_fsi_stability_established=False,
        coupled_dt_acceptance_established=False,fluid_shape=grid.shape,spacing_cm=grid.spacing)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
