"""Read-only saved-state audit plus one diagnostic step; production is untouched."""
import json,hashlib,sys
from pathlib import Path
import torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.cut_diagnostics import fixed_tract_momentum
from afsi_torch.ep.active_diagnostics import active_metrics
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from scripts.analyze_lv_transport import flux_budget

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    root=Path('results/afsei_source_timing_neck_two_cycles_20261029')
    path=root/'phase_checkpoints/02_beat1_ejection_150ms_sample_tick00013199.npz'
    out=Path('results/lv_basal_audit_20261030/force_decomposition');out.mkdir(parents=True,exist_ok=True)
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    d,s,_=restart.load(path,cardiac=True)
    print('loaded',s.fsi.time,flush=True)
    new,info=d.step(s,target_tick=s.tick+d.maximum_ticks,diagnostics=True)
    e=d._fsi_stepper(round(info['accepted_dt_s']/d.protocol.dt_ep_s));p=e._last_cut_problem
    x,_,volume,force,u,pressure,_,nodal,density,_,_=p.last_data
    domain=e.flow.domain;grid=e.grid;dt=info['accepted_dt_s'];geometric=(x-s.fsi.x)/dt
    budget=domain.budget(u,p.rows.prepared,geometric)
    budget={k:float(v) if isinstance(v,torch.Tensor) else v for k,v in budget.items()}
    b=info['balanced'];budget['tract_storage_rate_ml_s']=b['tract_volume_rate_ml_s']
    budget['cavity_volume_rate_ml_s']=float((volume-p.load.volume(s.fsi.x))/dt)
    budget['cap_relative_over_tract_ml_s']=budget['cap_over_tract_absolute_ml_s']-budget['cap_over_tract_motion_ml_s']
    budget['cut_vs_moving_cap_storage_residual_ml_s']=budget['cut_outward_ml_s']-budget['cap_relative_over_tract_ml_s']+b['tract_volume_rate_ml_s']
    prepared=p.rows.prepared
    budget['all_cap_relative_by_label_ml_s']=prepared.relative_flow(u,geometric).cpu().tolist()
    budget['all_cap_absolute_by_label_ml_s']=prepared.rows.flow(u).cpu().tolist()
    budget['all_cap_motion_by_label_ml_s']=prepared.material_flow(geometric).cpu().tolist()
    regions={}
    for name,mask,k0,k1 in [('fixed_union',domain.neck_apertures[0]|domain.neck_apertures[1],domain.bottom_index,domain.shared_top_index),
                          ('av_neck',domain.neck_apertures[1],domain.bottom_index,domain.shared_top_index),
                          ('upper_av',domain.apertures[1],domain.shared_top_index+1,grid.shape[2])]:
        regions[name]=flux_budget(u,mask,k0,k1,grid,e.flow.rho)
    momentum={}
    for name,bottom,mask in [('neck_to_cut',domain.bottom_index,domain.neck_apertures[1]),
                             ('upper_same_columns',domain.shared_top_index+1,domain.neck_apertures[1]),
                             ('upper_full',domain.shared_top_index+1,domain.apertures[1])]:
        momentum[name]=fixed_tract_momentum(e.flow,p.state.velocity,p.departure,u,pressure,density,
            axis=2,bottom=bottom,aperture=mask)
    # Separate original IB interpolation from the volume/pressure-mode correction.
    transfer=e.transfer
    stencil=p.current_stencil
    base,base_info=transfer.base.interpolate(u,stencil.original)
    corrected,_=transfer.interpolate(u,stencil)
    from afsi_torch.ep.mac_transport import closed_volume_gradient
    grad=closed_volume_gradient(x,p.load.faces,p.load.rim)
    modes=dict(base_cavity_rate_ml_s=float((grad*base).sum()),
        corrected_cavity_rate_ml_s=float((grad*corrected).sum()),
        geometric_cavity_rate_ml_s=float((grad*geometric).sum()),
        correction_max_cm_s=float((corrected-base).norm(dim=1).max()),
        base_max_cm_s=float(base.norm(dim=1).max()),
        geometric_max_cm_s=float(geometric.norm(dim=1).max()))
    from afsi_torch.ep.mac_compact_cap import add_cap_lift
    raw,_=transfer.base.spread(force,stencil.original)
    cap_force=add_cap_lift(p.rows,-p.last_cap_pressure,tuple(torch.zeros_like(v) for v in density),check_finite=False)
    corrected_force=tuple(v-r-c for v,r,c in zip(density,raw,cap_force))
    def integrate_force(field,bottom):
        mask=domain.neck_apertures[1];h=grid.spacing[2]
        weights=x.new_full((grid.shape[2]-bottom+1,),h);weights[0]*=.5;weights[-1]*=.5
        return float(-(field[2][:,:,bottom:]*mask[:,:,None]*weights).sum()/mask.sum()/C)
    force_parts={name:dict(neck_to_cut_mmhg=integrate_force(value,domain.bottom_index),
        upper_mmhg=integrate_force(value,domain.shared_top_index+1))
        for name,value in [('original_ib_spread',raw),('cap_lift',cap_force),('pressure_mode_correction',corrected_force)]}
    # Independent source formula, evaluated at exactly the same state and geometry.
    load=d.buffer;g=load.geometry;F=d.model.element_gradient(x);f=d.mesh.fiber
    ff=torch.einsum('eij,ej->ei',F,f);of=torch.einsum('eij,ej->ei',load.previous_F,f)
    lam=ff.norm(dim=1)[:,None];old=of.norm(dim=1)[:,None];lc=lam.clamp_max(1.2)
    h=(1+2.3*(lc+lc.clamp_max(.87)-1.87)).clamp_min(0)
    xs,xw,zs,zw=load.fields.unbind(-1);tau=new.force_load.dt_s
    zs=zs*np.exp(-40.14*tau)+10*(lam-old)*(-np.expm1(-40.14*tau)/(40.14*tau))
    zw=zw*np.exp(-405.86*tau)+10*(lam-old)*(-np.expm1(-405.86*tau)/(405.86*tau))
    ta=(4.8e6*h*(xs*(zs+1)+xw*zw)).clamp_min(0)
    coefficient=(g.weights*ta*(1+4.9*(lam.clamp_min(1)-1))).sum(-1)
    directional=torch.einsum('eaJ,eJ->ea',g.gradients[:,0],f)
    local=-torch.einsum('e,ei,ea->eai',coefficient,ff,directional)
    source=torch.zeros_like(x).index_add_(0,g.cells.reshape(-1),local.reshape(-1,3))
    actual=load.force(x)
    active=dict(metrics=active_metrics(d.model,load,x),source_force_relative_error=float((source-actual).norm()/source.norm()),
        ta_max_error_cgs=float((ta-load.tension(x)).abs().max()),
        scope='identical state and geometry, independent source formula; not independent solid-only trajectory')
    report=dict(time_ms=new.fsi.time*1000,checkpoint=str(path),budget=budget,fixed_regions=regions,
        momentum=momentum,force_parts=force_parts,interpolation=modes,active=active,balanced=b,
        source_sha256=checksum,source_unchanged=checksum==hashlib.sha256(path.read_bytes()).hexdigest(),
        diagnostic_accepted_step_only=True,production_modified=False)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='balanced'},indent=2),flush=True)

if __name__=='__main__':main()
