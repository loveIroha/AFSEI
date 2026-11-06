"""Isolated paired short probes; never resume the production ledger after geometry edits."""
import argparse,json,sys,time,subprocess,hashlib,traceback
from pathlib import Path
import numpy as np
from dataclasses import replace
ROOT=Path('results/lv_connection_verify_20261030')
SOURCE=Path('results/afsei_source_timing_neck_two_cycles_20261029/phase_checkpoints/02_beat1_ejection_150ms_sample_tick00013199.npz')

def prepare():
    ROOT.mkdir(exist_ok=True)
    with np.load(SOURCE) as a:data={k:a[k] for k in a.files}
    np.savez_compressed(ROOT/'input.npz',**data)
    (ROOT/'provenance.json').write_text(json.dumps(dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        common_change='EP backend AMG -> optimized Jacobi-PCG; same nodal/CN/model/tolerances',
        cases=['baseline','patch2','collar5'],production_modified=False),indent=2))

def run(case):
    import torch
    from afsi_torch.ep import mac_balanced_checkpoint as restart
    from afsi_torch.ep.gao_tracts import prepare_cut_flow
    from afsi_torch.ep.mac_balanced_dynamic import BalancedAFSEIStepper
    from afsi_torch.ep.cut_diagnostics import fixed_tract_momentum
    from afsi_torch.ep.mac_compact_cap import add_cap_lift
    from afsi_torch.ep.mac_ports import _stencil
    from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
    torch.set_num_threads(4)
    out=ROOT/case;out.mkdir(exist_ok=True);report=dict(case=case,status='running',records=[])
    started=time.perf_counter()
    try:
        d,s,_=restart.load(ROOT/'input.npz',cardiac=True)
        numerics=replace(d.config.ep_numerics,backend='torch_pcg_jacobi_light')
        d.ep.close();d.ep.numerics=numerics
        d.config=replace(d.config,ep_numerics=numerics)
        d.ep.prepare(d.protocol.dt_ep_s)
        e=d._fsi_stepper(d.maximum_ticks);prepare_cut_flow(e,s.fsi,s)
        oldstorage=e.flow.domain.storage.volume(s.fsi.x)
        if case=='patch2':d.cap_patch_width_cells=2.
        if case=='collar5':
            d.cut_geometry=dict(d.cut_geometry,shared_collar_cells=5)
            for ep in d._steppers.values():ep._gao_tract_flows={}
            prepare_cut_flow(e,s.fsi,s)
        domain=e.flow.domain
        mismatch=max(float((v-w).abs().max()) for v,w in zip(s.fsi.velocity,domain.mask(s.fsi.velocity)))
        report.update(initial_mask_difference_cm_s=mismatch,
            initial_storage_change_ml=domain.storage.volume(s.fsi.x)-oldstorage,
            patch_width=d.cap_patch_width_cells,geometry=domain.metadata,
            start_time_ms=s.fsi.time*1000,ep_backend=d.config.ep_numerics.backend)
        if mismatch>1e-8:
            report.update(status='incompatible_initial_velocity',interpretation='New geometry blocks nonzero initial velocities; no short trajectory or causal pressure comparison is accepted.')
        else:
            target=s.tick+10  # 0.5 ms, five ordinary FSI steps; EP dt remains 0.05 ms.
            while s.tick<target:
                old=s;t=time.perf_counter()
                s,info=BalancedAFSEIStepper.step(d,s,target_tick=target,diagnostics=True)
                endpoint=d._fsi_stepper(round(info['accepted_dt_s']/d.protocol.dt_ep_s));p=endpoint._last_cut_problem
                x,_,_,force,u,pressure,_,nodal,density,_,_=p.last_data
                domain=endpoint.flow.domain;mask=domain.neck_apertures[1]
                momentum=fixed_tract_momentum(endpoint.flow,p.state.velocity,p.departure,u,pressure,density,
                    axis=2,bottom=domain.bottom_index,aperture=mask)
                cap=add_cap_lift(p.rows,-p.last_cap_pressure,tuple(torch.zeros_like(v) for v in density),check_finite=False)
                raw,_=endpoint.transfer.base.spread(force,p.current_stencil.original)
                correction=tuple(a-b-c for a,b,c in zip(density,raw,cap))
                def integral(field):
                    bottom=domain.bottom_index;h=endpoint.grid.spacing[2]
                    weights=x.new_full((endpoint.grid.shape[2]-bottom+1,),h);weights[0]*=.5;weights[-1]*=.5
                    return float(-(field[2][:,:,bottom:]*mask[:,:,None]*weights).sum()/mask.sum()/C)
                # Sample cap lift on the virtual open patches. Nonzero support alone is not a leak.
                prepared=p.rows.prepared;normal=prepared.area_vectors/prepared.area_vectors.norm(dim=1)[:,None]
                values=torch.stack([(v.flatten()[ids]*w).sum(-1) for v,(ids,w) in zip(cap,_stencil(endpoint.grid,prepared.points))],-1)
                opened=prepared.cap.labels>0
                cap_open_rms=float(((values*normal).sum(-1)[opened].square().mean()).sqrt())
                b=info['balanced']
                row=dict(time_ms=s.fsi.time*1000,step_seconds=time.perf_counter()-t,
                    volume_ml=b['volume_ml'],core_mmhg=b['pressure_core']['mean_mmhg'],outlet_mmhg=b['boundary_pressure_mmhg'],
                    q_ml_s=b['cut_outward_flux_ml_s'],mass_error_ml_s=b['mass_error_ml_s'],
                    work_error=b['boundary_work_relative_error'],transfer_error=b['transfer_power_relative_error'],
                    momentum=momentum,cap_mmhg=integral(cap),mode_mmhg=integral(correction),
                    open_patch_cap_force_density_rms_dyn_cm3=cap_open_rms,
                    nonlinear_iterations=info['nonlinear']['iterations'],rejections=info.get('rejected_attempts',[]),
                    ep=info.get('ep',{}))
                report['records'].append(row);print(case,json.dumps(row),flush=True)
                (out/'report.json').write_text(json.dumps(report,indent=2))
            report['status']='passed'
    except Exception as exc:
        report.update(status='failed',failure=str(exc),traceback=traceback.format_exc())
    report['elapsed_seconds']=time.perf_counter()-started
    (out/'report.json').write_text(json.dumps(report,indent=2));print(case,report['status'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case');args=parser.parse_args()
    if args.case:run(args.case)
    else:
        prepare()
        for name in ['baseline','patch2','collar5']:
            subprocess.run([sys.executable,__file__,'--case',name],check=True)
