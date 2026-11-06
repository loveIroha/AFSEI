"""Frozen-geometry hydrostatic and work-pair tests on original operators.

All forces here are manufactured test data, not new production loads. No
EP/ODE, Newton, physical advance, state publication or parameter rewrite.
"""
import argparse,hashlib,json
from pathlib import Path
from time import perf_counter
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.gao_tracts import prepare_cut_flow
from afsi_torch.ep.mac_pressure_audit import cavity_masks
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.mac.sealed_pressure import gradient,divergence
from afsi_torch.mac.open_boundary import velocity_laplacian,face_weights


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--work-only',action='store_true',help='Check work identities without repeating hydrostatic solves')
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);start=perf_counter();before=digest(a.checkpoint)
    d,s,_=restart.load(a.checkpoint,cardiac=True);x=s.fsi.x
    e=d._fsi_stepper(round(s.force_load.dt_s/d.protocol.dt_ep_s))
    prepare_cut_flow(e,s.fsi,s);st=e.transfer.prepare(x);cap=st.rows;flow=e.flow
    _,core=cavity_masks(x,e.model.endo_faces,e.grid,guard_cells=d.pressure_guard_cells)
    report=dict(time_ms=s.fsi.time*1000,physical_time_advanced=False,
        checkpoint_sha256=before,geometry='577.9 ms endpoint frozen; no velocity history advanced',
        manufactured_load='total nodal force = P*mode, closed cap multipliers = P*unit; not actual Land/material equilibrium',
        pressure_work_tests=[],cap_pair_tests=[],hydrostatic_cases=[])
    report['unit_target_core_mmhg']=dict(mean=float(st.potential[core].mean()/C),
        minimum=float(st.potential[core].min()/C),maximum=float(st.potential[core].max()/C))
    def publish():
        target=a.output/'report.json';tmp=target.with_suffix('.json.working')
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(target)
    def work(u,v):return sum((p*q*face_weights(e.grid,c,x)).sum() for c,(p,q) in enumerate(zip(u,v)))
    def lifting(f,p_mm):
        out=list(f.grid.zeros(device=x.device))
        for plane,value in zip(f.planes,p_mm):
            sign=-1. if plane.side==0 else 1.
            out[plane.axis].select(plane.axis,plane.side).add_(
                sign*2*C/f.grid.spacing[plane.axis]*value*plane.aperture.to(x.dtype))
        return tuple(out)
    flow.prescribe([1.])
    manufactured_gradient=tuple(g+b for g,b in zip(st.gradient,st.cut_unit_lift))
    e.transfer.reset_warm_start()
    density,_,_=e.transfer.relative_parts(st.mode,st.cap_unit_pressure,st,rows=cap,check_finite=False)
    report['nodal_cap_to_gradient_relative_error']=float((sum((v-w).square().sum() for v,w in zip(density,manufactured_gradient))
        /sum(v.square().sum() for v in manufactured_gradient)).sqrt())

    # Exact SBP identity, including half boundary control-volume weights.
    for label,raw,pfield,pvalue in (('saved_fields',s.fsi.velocity,s.fsi.pressure,
        json.loads((a.checkpoint.parent/'report.json').read_text())['records'][-1]['balanced']['prescribed_cut_pressure_mmhg']),
        ('smooth_test',tuple(torch.sin(e.grid.coordinates(c,device=x.device)[...,0]+.3*e.grid.coordinates(c,device=x.device)[...,2])
            for c in range(3)),st.potential,1.)):
        u=flow.domain.mask(raw);g=gradient(pfield,e.grid.spacing,flow.domain.masks);b=lifting(flow,[pvalue])
        lhs=work(u,tuple(v+w for v,w in zip(g,b)))+e.grid.volume*(pfield*divergence(u,e.grid.spacing)).sum()
        rhs=C*pvalue*flow.flux(u).sum()
        error=float(abs(lhs-rhs));scale=max(float(abs(lhs)+abs(rhs)),1.)
        report['pressure_work_tests'].append(dict(name=label,relative_error=error/scale,
            absolute_error_erg_s=error,left_power_erg_s=float(lhs),
            cut_power_erg_s=float(rhs),passed=error/scale<1e-10))
    e.transfer.reset_warm_start();nodal,_=e.transfer.interpolate(s.fsi.velocity,st,check_finite=False);nodal=nodal.clone()
    for label,bp in (('uniform_closed_cap',st.cap_unit_pressure),
        ('signed_closed_patches',C*torch.sin(torch.arange(cap.count,device=x.device,dtype=x.dtype)))):
        bp=bp.clone();bp[-2:]=0.;reaction=cap.reaction(bp)
        e.transfer.reset_warm_start();density,_,_=e.transfer.relative_parts(reaction,bp,st,rows=cap,check_finite=False)
        fluid=work(s.fsi.velocity,density);material=(reaction*nodal).sum();flux=(bp*cap.flow(s.fsi.velocity)).sum()
        relative=float(abs(fluid-material+flux))/max(float(abs(fluid)+abs(material)+abs(flux)),1.)
        report['cap_pair_tests'].append(dict(name=label,relative_error=relative,passed=relative<1e-7))
    publish();print(json.dumps(dict(stage='work_pairs_checked',**report),indent=2),flush=True)
    if a.work_only:
        report.update(checkpoint_unchanged=digest(a.checkpoint)==before,
            elapsed_seconds=perf_counter()-start,completed=True,work_only=True,
            diagnostic_boundary_lift_dtype=str(x.dtype))
        publish()
        return

    def measure(f,u,p,scale,name,seconds,info):
        e.transfer.reset_warm_start();v,_=e.transfer.interpolate(u,st,check_finite=False)
        rates=cap.flow(u)-cap.material_flow(v)
        lower=rates[:-2]/cap.area[:-2];w=cap.area[:-2]/cap.area[:-2].norm()
        lower=lower+15*w*(w*lower).sum()
        residual=torch.sqrt((f.dt*v).square().sum()+lower.square().sum())
        error=p-scale*st.potential
        result=dict(name=name,dt_s=f.dt,pressure_scale_mmhg=scale,seconds=seconds,
            max_velocity_cm_s=max(float(q.abs().max()) for q in u),
            nodal_velocity_norm_cm_s=float(v.norm()),outlet_flow_ml_s=float(f.flux(u).sum()),
            closed_cap_relative_flow_norm_ml_s=float(rates[:-2].norm()),
            manufactured_coupled_residual=float(residual),
            original_outer_atol=d.config.fsi.nonlinear.atol,
            pressure_core_error_mmhg=float(error[core].mean()/C),
            pressure_field_relative_error=float(error.norm()/(scale*st.potential.norm())),
            pressure_info=info.get('pressure',info),
            helmholtz_residuals=info.get('helmholtz_residuals'),
            helmholtz_tolerances=info.get('helmholtz_tolerances'))
        report['hydrostatic_cases'].append(result);publish()
        print(json.dumps(result),flush=True)
    for interval in (2,1):
        endpoint=e if interval==2 else d._fsi_stepper(interval)
        if endpoint is not e:prepare_cut_flow(endpoint,s.fsi,s)
        f=endpoint.flow;zero=f.grid.zeros(device=x.device)
        grad=manufactured_gradient
        for name in ('current_gradient','BE_compatible_manufactured_control','projection_only_control'):
            f.prescribe([1.]);tick=perf_counter()
            if name=='projection_only_control':
                star=tuple(f.dt/f.rho*v for v in grad)
                p0,info=f.pressure_solver.solve(-f.rho/f.dt*divergence(star,f.grid.spacing))
                gp=gradient(p0,f.grid.spacing,f.domain.masks)
                u0=tuple(v-f.dt/f.rho*g for v,g in zip(star,gp))
                u,p=f._combine(u0,p0,f.pressures)
            else:
                force=grad if name=='current_gradient' else tuple(m*(v-f.dt*f.mu/f.rho*velocity_laplacian(v,c,f.grid.spacing))
                    for c,(v,m) in enumerate(zip(grad,f.domain.masks)))
                u,p,info=f.advance(zero,force,None)
            measure(f,u,p,1.,name,perf_counter()-tick,info)
        if interval==2:
            f.prescribe([80.]);tick=perf_counter()
            u,p,info=f.advance(zero,tuple(80*v for v in grad),None)
            measure(f,u,p,80.,'current_gradient_80mmhg',perf_counter()-tick,info)
    first=next(r for r in report['hydrostatic_cases'] if r['name']=='current_gradient' and r['dt_s']==1e-4)
    half=next(r for r in report['hydrostatic_cases'] if r['name']=='current_gradient' and r['dt_s']==5e-5)
    report.update(max_velocity_half_to_full_ratio=half['max_velocity_cm_s']/max(first['max_velocity_cm_s'],1e-300),
        checkpoint_unchanged=digest(a.checkpoint)==before,elapsed_seconds=perf_counter()-start,completed=True,
        control_interpretation='BE-compatible forcing applies H to the same gradient solely to identify predictor/projection splitting; it is NOT a proposed production-force replacement.')
    publish()
    cases=report['hydrostatic_cases'];labels=[r['name'].replace('_',' ')+'\n'+str(r['dt_s'])+' s' for r in cases]
    fig,ax=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    ax[0].barh(labels,[r['max_velocity_cm_s'] for r in cases]);ax[0].set_xscale('log')
    ax[0].set(xlabel='Max velocity [cm/s]',title='Manufactured static response, original operators')
    ax[1].barh(labels,[max(r['manufactured_coupled_residual'],1e-30) for r in cases]);ax[1].set_xscale('log')
    ax[1].axvline(d.config.fsi.nonlinear.atol,color='r',ls='--',label='Original outer absolute tolerance')
    ax[1].set(xlabel='Fixed-pose manufactured residual',title='Diagnostic only; no physical step accepted');ax[1].legend(fontsize=8)
    for xaxis in ax:xaxis.grid(alpha=.2)
    fig.suptitle('Same 577.9 ms geometry: hydrostatic / port-work controls')
    fig.savefig(a.output/'static_port.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
