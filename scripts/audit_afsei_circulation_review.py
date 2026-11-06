"""Read-only CPU audit of saved AFSEI circulation histories and active fields.

No physical advancement, fluid response, EP/ODE integration or solver changes.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import argparse
import numpy as np

C = 1333.22368421


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(p, start):
    records = []
    previous = start
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            b, c = r['balanced'], r['cardiac']
            t = r['time_ms']/1000
            records.append(dict(t=t, dt=t-previous, beat=c['beat'],
                phase=c['solved_phase'], next_phase=c['phase'], mode=b['mode'],
                outlet_open=b['outlet_open'],
                v=b['volume_ml'], tract=b.get('tract_volume_ml', 0.),
                pc=b['arterial_pc_mmhg'], oldpc=b['arterial_pc_old_mmhg'],
                qin=b['inlet_flow_ml_s'], qout=b.get('physical_outlet_flow_ml_s', r.get('q_out_ml_s',0.)),
                p=b['pressure_core']['mean_mmhg'], port=b['boundary_pressure_mmhg'],
                pin=b.get('supply_pressure_mmhg'), wall=b['wall_pressure_mmhg'],
                ta=r['tension_max_cgs']/1e4, factor=c['unload_factor'],
                events=c['events'], mass=b['mass_error_ml_s'],
                cumulative_mass=c['cumulative_mass_error_ml'],
                edv=c['edv_ml'], esv=c['esv_ml'],
                rcr_residual=b.get('rcr_balance_ml_s'),
                sheets=b.get('internal_pressure_sheets_mmhg'),
                cap_reference=b['cap_reference_mmhg'],
                momentum=r.get('cut_momentum_budget'),
                land=r['ep']['land_advanced']))
            previous=t
    return records


def aggregate(records, initial_pc, params):
    rc,rp,cap,pv=(params[k] for k in ('rc_mmhg_s_per_ml','rp_mmhg_s_per_ml',
                                     'compliance_ml_per_mmhg','venous_mmhg'))
    oldpc=initial_pc
    errors=dict(pc_update_mmhg=0., old_pc_continuity_mmhg=0., outlet_law_mmhg=0.,
                mass_ml_s=0., cumulative_mass_ml=0., nonfilling_wall_mmhg=0.,
                ejection_cap_reference_mmhg=0., internal_pressure_sheet_mmhg=0.,
                closed_outlet_flow_ml_s=0., closed_valve_positive_pressure_gap_mmhg=0.)
    for r in records:
        dt=r['dt']; q=r['qout'] if r['mode']=='rcr' else 0.
        expected=(cap*r['oldpc']+dt*(q+pv/rp))/(cap+dt/rp)
        errors['pc_update_mmhg']=max(errors['pc_update_mmhg'],abs(r['pc']-expected))
        errors['old_pc_continuity_mmhg']=max(errors['old_pc_continuity_mmhg'],abs(r['oldpc']-oldpc))
        oldpc=r['pc']
        if r['mode']=='rcr' and r['outlet_open']:
            errors['outlet_law_mmhg']=max(errors['outlet_law_mmhg'],abs(r['port']-r['pc']-rc*r['qout']))
        elif r['mode']=='rcr':
            errors['closed_outlet_flow_ml_s']=max(errors['closed_outlet_flow_ml_s'],abs(r['qout']))
            errors['closed_valve_positive_pressure_gap_mmhg']=max(
                errors['closed_valve_positive_pressure_gap_mmhg'],r['port']-r['pc'])
        if r['phase']!='FILLING':
            errors['nonfilling_wall_mmhg']=max(errors['nonfilling_wall_mmhg'],abs(r['wall']))
        if r['phase']=='EJECTION':
            errors['ejection_cap_reference_mmhg']=max(errors['ejection_cap_reference_mmhg'],abs(r['cap_reference']))
        if r['sheets'] is not None:
            errors['internal_pressure_sheet_mmhg']=max(errors['internal_pressure_sheet_mmhg'],
                float(np.max(np.abs(r['sheets']))))
        errors['mass_ml_s']=max(errors['mass_ml_s'],abs(r['mass']))
        errors['cumulative_mass_ml']=max(errors['cumulative_mass_ml'],abs(r['cumulative_mass']))
    beats=[]
    for beat in sorted({r['beat'] for r in records}):
        rows=[r for r in records if r['beat']==beat]
        # A pacing boundary is solved in the preceding physical phase.
        # Its beat label is the new beat; assign it by its right endpoint below.
        start=.5+(beat-1)*.9; end=start+.9
        rows=[r for r in records if start < r['t']+1e-12 and r['t']<=end+1e-12
              and r['t']>start+1e-12]
        if not rows: continue
        A=1.; B=0.
        for r in rows:
            a=cap/(cap+r['dt']/rp)
            B=a*B+r['dt']*r['qout']/(cap+r['dt']/rp)
            A*=a
        phases={}
        for name in ('ISOVOLUMIC_CONTRACTION','EJECTION','ISOVOLUMIC_RELAXATION','FILLING'):
            rr=[r for r in rows if r['phase']==name]
            if not rr:continue
            z=dict(start_ms=(rr[0]['t']-rr[0]['dt'])*1000,end_ms=rr[-1]['t']*1000,
                duration_ms=sum(r['dt'] for r in rr)*1000,
                first_record_volume_ml=rr[0]['v'],
                volume_end_ml=rr[-1]['v'],
                inflow_ml=sum(r['dt']*r['qin'] for r in rr),
                outflow_ml=sum(r['dt']*r['qout'] for r in rr),
                wall_pressure_time_mmhg_s=sum(r['dt']*r['wall'] for r in rr),
                last_qin_ml_s=rr[-1]['qin'],last_core_mmhg=rr[-1]['p'],
                last_wall_mmhg=rr[-1]['wall'],last_ta_max_kpa=rr[-1]['ta'])
            if name=='FILLING':
                positive=[r for r in rr if r['wall']>1e-8]
                supplied=[r for r in rr if r['pin'] is not None and r['pin']>=8.-1e-8]
                z.update(wall_positive_duration_ms=sum(r['dt'] for r in positive)*1000,
                    full_inlet_pressure_duration_ms=sum(r['dt'] for r in supplied)*1000,
                    gao_original_wall_pressure_time_mmhg_s=4.8)
                z['relative_to_gao_original_wall_pressure_time']=z['wall_pressure_time_mmhg_s']/4.8
            phases[name]=z
        duration=sum(r['dt'] for r in rows)
        avgq=sum(r['dt']*r['qout'] for r in rows)/duration
        peak=max(rows,key=lambda r:r['p'])
        ta_peak=max(rows,key=lambda r:r['ta'])
        full_period=abs(duration-.9)<1e-8
        beats.append(dict(beat=beat,start_ms=(rows[0]['t']-rows[0]['dt'])*1000,end_ms=rows[-1]['t']*1000,
            duration_s=duration,pc_start_mmhg=rows[0]['oldpc'],pc_end_mmhg=rows[-1]['pc'],
            mean_outflow_ml_s=avgq,
            frozen_flow_periodic_pc_at_start_mmhg=pv+B/(1-A) if full_period else None,
            frozen_flow_period_map_A=A, frozen_flow_period_map_B_mmhg=B,
            steady_mean_outlet_for_frozen_flow_mmhg=pv+(rc+rp)*avgq if full_period else None,
            peak_core_mmhg=peak['p'],peak_core_time_ms=peak['t']*1000,
            peak_ta_max_kpa=ta_peak['ta'],peak_ta_time_ms=ta_peak['t']*1000,phases=phases))
    momentum=[]
    for r in records:
        m=r['momentum']
        if m is not None:
            momentum.append(dict(t_ms=r['t']*1000,core_minus_cut=r['p']-m['cut_mmhg'],
                core_minus_neck=m['core_minus_neck_mmhg'],neck_minus_cut=m['measured_drop_mmhg'],
                terms=m['terms_mmhg'],consistent=m['consistent_with_scheme'],
                closure_error=m['closure_error_mmhg']))
    samples=[]
    if momentum:
        for target in (578.,620.,644.35,700.,750.,789.95):
            samples.append(min(momentum,key=lambda r:abs(r['t_ms']-target)))
    unload=[]
    for r in records:
        if 'OUTLET_CLOSURE_TA_SNAPSHOT' in r['events']:
            t0=r['t'];anchor=r['ta'];rr=[x for x in records if t0<x['t']<=t0+.08+1e-12]
            unload.append(dict(start_ms=t0*1000,end_ms=(t0+.08)*1000,
                max_factor_error=max((abs(x['factor']-max(0.,1-(x['t']-t0)/.08)) for x in rr),default=0.),
                max_tension_error_kpa=max((abs(x['ta']-anchor*max(0.,1-(x['t']-t0)/.08)) for x in rr),default=0.)))
    return dict(rows=len(records),errors=errors,beats=beats,momentum_samples=samples,unloading=unload,
        momentum_all_consistent=all(r['consistent'] for r in momentum),
        events=[dict(t_ms=r['t']*1000,events=r['events']) for r in records if r['events']],
        time_constant_s=rp*cap,
        first_tract_ml=records[0]['tract'],last_tract_ml=records[-1]['tract'])


def weighted(values, weights):
    v=np.broadcast_to(values,weights.shape).ravel();w=weights.ravel()
    order=np.argsort(v);cdf=np.cumsum(w[order]);cdf/=cdf[-1]
    return dict(mean=float(v@w/w.sum()),min=float(v.min()),max=float(v.max()),
        p05=float(np.interp(.05,cdf,v[order])),p50=float(np.interp(.5,cdf,v[order])),
        p95=float(np.interp(.95,cdf,v[order])))


def active_snapshot(p, logged_max=None):
    from itertools import permutations
    import torch
    from afsi_torch.ep.land import land_tension
    before=sha(p)
    with np.load(p,allow_pickle=False) as z:
        m=json.loads(str(z['metadata']))
        X,cells,x,fiber,previous=(z[k] for k in ('X','cells','x','fiber','load_previous_x'))
        nodal=np.stack([z[k] for k in ('load_xs','load_xw','load_zs_old','load_zw_old')],axis=-1)
        anchor=z['force_anchor'] if 'force_anchor' in z else None
        tags=z['facet_tags'];faces=z['faces'];current_z=z['zs'];current_w=z['zw']
    if len(nodal)!=len(X):raise ValueError('this review expects the unrefined EP mesh')
    D=(X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1)
    grad=np.array([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])@np.linalg.inv(D)
    F=np.einsum('eai,eaJ->eiJ',x[cells],grad)
    previousF=np.einsum('eai,eaJ->eiJ',previous[cells],grad)
    v=np.einsum('eij,ej->ei',F,fiber);oldv=np.einsum('eij,ej->ei',previousF,fiber)
    lam=np.linalg.norm(v,axis=1)[:,None];oldlam=np.linalg.norm(oldv,axis=1)[:,None]
    bary=[];weights=[]
    for orbit,w in [((.25,)*4,.030283678097089182),
        ((0.,1/3,1/3,1/3),.006026785714285717),
        ((8/11,1/11,1/11,1/11),.011645249086028967),
        ((.4334498464263357,)*2+(.0665501535736643,)*2,.010949141561386449)]:
        for b in sorted(set(permutations(orbit))):bary.append(b);weights.append(w)
    weights=np.abs(np.linalg.det(D))[:,None]*np.array(weights)
    fields=np.einsum('qa,eak->eqk',np.array(bary),nodal[cells])
    xs,xw,zs,zw=fields.transpose(2,0,1)
    active=m['active_load'];dt=active['dt_s']
    lc=np.minimum(lam,1.2);h=np.maximum(0,1+2.3*(lc+np.minimum(lc,.87)-1.87))
    zs_new=zs*np.exp(-40.14*dt)+10*(-np.expm1(-40.14*dt)/(40.14*dt))*(lam-oldlam)
    zw_new=zw*np.exp(-405.86*dt)+10*(-np.expm1(-405.86*dt)/(405.86*dt))*(lam-oldlam)
    ta=np.maximum(0,4.8e6*h*(xs*(zs_new+1)+xw*zw_new))
    # Independent NumPy evaluation of the source UFL formula versus production reference.
    expected=land_tension(*(torch.from_numpy(a) for a in (lam,oldlam,xs,xw,zs,zw)),dt).numpy()
    reference_error=float(np.max(np.abs(ta-expected)))
    if not active['use_land']:
        ta=np.zeros_like(ta) if anchor is None else anchor*active['ramp_factor']
    factor=1+4.9*(np.maximum(lam,1)-1)
    J=np.linalg.det(F)[:,None]
    stress=ta*factor*lam**2/J
    power=-float(np.sum(weights*ta*factor*np.sum(v*(v-oldv)/dt,axis=1)[:,None]))/1e7
    lam_zero=.5*(1.87-1/2.3)
    base_nodes=np.unique(faces[tags==3]);dx=x[base_nodes]-X[base_nodes]
    center_xy=np.array(m['config']['fsi']['basal_center_cm'])
    radial=X[base_nodes,:2]-center_xy
    radial/=np.linalg.norm(radial,axis=1)[:,None]
    circ=np.stack((-radial[:,1],radial[:,0]),axis=-1)
    result=dict(checkpoint=str(p),sha256_before=before,sha256_after=sha(p),time_ms=m['time']*1000,
        use_land=active['use_land'],ramp_factor=active['ramp_factor'],
        independent_source_formula_max_abs_error_dyn_cm2=reference_error,
        stretch=weighted(lam,weights),land_length_factor=weighted(h,weights),
        land_zero_length_threshold=lam_zero,
        reference_volume_fraction_below_lambda_087=float(np.sum(weights*(lam<.87))/weights.sum()),
        reference_volume_fraction_zero_length_factor=float(np.sum(weights*(h<=0))/weights.sum()),
        tension_kpa=weighted(ta/1e4,weights),
        active_fiber_cauchy_stress_kpa=weighted(stress/1e4,weights*J),
        active_force_power_w=power,J=weighted(J,weights),
        fiber_norm_min=float(np.linalg.norm(fiber,axis=1).min()),fiber_norm_max=float(np.linalg.norm(fiber,axis=1).max()),
        no_negative_applied_tension=bool(np.min(ta)>=0),
        basal_mean_axial_displacement_mm=float(dx[:,2].mean()*10),
        basal_max_abs_axial_displacement_mm=float(np.abs(dx[:,2]).max()*10),
        basal_circumferential_rms_displacement_mm=float(np.mean(np.sum(dx[:,:2]*circ,axis=1)**2)**.5*10),
        current_distortion_zs_range=[float(current_z.min()),float(current_z.max())],
        current_distortion_zw_range=[float(current_w.min()),float(current_w.max())])
    if logged_max is not None:result['tension_max_vs_log_error_kpa']=float(np.max(ta)/1e4-logged_max)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-active',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path.cwd();out=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        physical_time_advanced=False,solver_modified=False,new_fluid_solves=0)
    old=root/'results/afsei_two_cycles_coarse_900ms_20261023'
    new=root/'results/gao_cut_cycle_validation_20261026/ejection_until_closure_20261028'
    out['cases']={};sources=[]
    for name,path,start,fn in [('old_internal_port',old,.5,'checkpoint.npz'),
                              ('new_physical_cut',new,.5779,'last_accepted_checkpoint.npz')]:
        with np.load(path/fn,allow_pickle=False) as z:m=json.loads(str(z['metadata']))
        cfg=json.loads((path/'configuration.json').read_text())
        rows=read_rows(path/'steps.jsonl',start)
        initial=rows[0]['oldpc'];params=m['balanced']['windkessel']
        audit=aggregate(rows,initial,params)
        audit.update(path=str(path),parameters=params,config=cfg.get('cardiac_execution'),
            cut_boundary=bool(m.get('cut_boundary')),initial_pc_mmhg=initial,
            complete_beat_count=2 if name.startswith('old') else 0)
        out['cases'][name]=audit
        for f in (path/'configuration.json',path/'steps.jsonl'):
            sources.append(dict(path=str(f),size_bytes=f.stat().st_size,sha256=sha(f)))
        if name.startswith('new'):newlastta=rows[-1]['ta']
    out['active_snapshots']=[]
    reused=json.loads(args.reuse_active.read_text())['active_snapshots'] if args.reuse_active else []
    for p,maximum in [
        (root/'results/gao_cut_cycle_validation_20261026/ejection_20ms_20261028/checkpoint.npz',None),
        (new/'last_accepted_checkpoint.npz',newlastta),
        (old/'diagnostics/tick_00027998/checkpoint.npz',0.),
        (old/'diagnostics/tick_00045998/checkpoint.npz',0.)]:
        if reused:
            a=next(r for r in reused if r['checkpoint']==str(p))
            if sha(p)!=a['sha256_after']:raise ValueError('reused active input changed')
            a['land_zero_length_threshold']=.5*(1.87-1/2.3)
            out['active_snapshots'].append(a)
        else:out['active_snapshots'].append(active_snapshot(p,maximum))
    out['input_manifest']=sources
    production=['src/afsi_torch/ep/'+s for s in ['cut_cycle.py','gao_tracts.py','gao_filling.py',
        'mac_balanced_dynamic.py','mac_pressure_mode.py','land.py','driver.py','source_mechanics.py','windkessel.py']]
    production+=['src/afsi_torch/mac/cut_plane.py','src/afsi_torch/mac/sealed_flow.py']
    out['source_manifest']=[dict(path=s,sha256=sha(root/s)) for s in production]
    out['limitations']=[
        'Only saved accepted trajectories; no new physical simulation or parameter sensitivity.',
        'A frozen-flow periodic RCR fixed point is not a periodic solution of coupled AFSEI.',
        'Momentums are discrete neck-to-cut budgets; not a proof of spatial convergence.',
        'No matched solid-only quadrature active fields were available; no force insufficiency claim.',
        'Current source hashes identify review state, not historical runtime source hashes.',
        'New cut run has no closure/unloading/filling/second beat; these remain physically unvalidated.'
    ]
    (args.output/'audit.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output/'audit.json'),cases={k:v['errors'] for k,v in out['cases'].items()},
        active=[dict(time_ms=r['time_ms'],tension_kpa=r['tension_kpa'],stretch=r['stretch'])
                for r in out['active_snapshots']]),indent=2))


if __name__=='__main__':main()
