"""Reuse the completed kappa=.4 trajectory; analyze one new .425 window.

CPU postprocessing only. Shared 29 normal-plane sections and reference-volume
J metrics are imported from the preceding validated analysis, not redefined.
"""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np,meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_available_volume_results import geometry_data,measure

ROOT=Path('/mnt/large2/gjh/AFSEI/results')
OUT=ROOT/'volume_kappa_sensitivity_20261025'
OLD=ROOT/'volumetric_stabilization_20261025'


def lumen_volume(x,geo):
    """Same oriented inner-surface + diagnostic-cap geometry as production."""
    rings=geo['rings'];center=x[rings].mean((0,1));p=x[geo['inner_faces']]-center
    value=np.sum(p[:,0]*np.cross(p[:,1],p[:,2]))
    for ring,sign in ((rings[0],-1.),(rings[-1],1.)):
        q=x[ring]-center
        value+=sign*np.sum(q.mean(0)*np.cross(q,np.roll(q,-1,axis=0)))
    return float(value/6)


def rel(a,b):
    return float(np.linalg.norm((a-b).ravel())/max(np.linalg.norm(b.ravel()),1e-30))


def main():
    design=json.loads((OUT/'design.json').read_text());candidate=OUT/'candidate'
    run=json.loads((candidate/'report.json').read_text())
    if run['status']!='completed':raise RuntimeError('Candidate did not finish; do not mark successful')
    baseline=Path(design['baseline']);geo=geometry_data(baseline)
    old_rows=json.loads((OLD/'results_analysis/frame_metrics.json').read_text())['stabilized_solid2']
    old={round(r['time_ms'],7):r for r in old_rows if 280-1e-7<=r['time_ms']<=300+1e-7}
    old_h={}
    for h in json.loads((baseline/'continued_300ms/history.json').read_text()):
        if .28-1e-10<=h['time_s']<=.3+1e-10:old_h[round(h['time_s']*1000,7)]=h
    source=np.load(design['checkpoint']);initial=np.load(candidate/'initial_state.npz')
    names={k:k for k in ['reference','cells','x','previous_x','u','v','w','pressure','voltage_v','gpb','land','zs','zw']}
    names['resting_tension_dyn_cm2']='resting_tension'
    input_checks={k:bool(np.array_equal(source[k],initial[v])) for k,v in names.items()}
    if not all(input_checks.values()):raise AssertionError('An accepted initial field was altered')
    history=json.loads((candidate/'history.json').read_text());hist={h['step']:h for h in history}
    rows=[]
    for e in ET.parse(candidate/'vtk/solid.pvd').findall('.//DataSet'):
        local=float(e.attrib['timestep']);step=round(local/design['dt_s']);t=280+local*1000
        m=meshio.read(candidate/'vtk'/e.attrib['file'])
        if not np.array_equal(m.cells_dict['tetra'],geo['cells']):raise AssertionError('FE cell ordering changed')
        r=dict(time_ms=t,**measure(m.points,m.cell_data_dict['J_min']['tetra'],geo,2))
        r.update(lumen_volume_ml=lumen_volume(m.points,geo),maximum_applied_tension_kpa=hist[step]['maximum_applied_tension_kpa'],
            nonlinear_iterations=hist[step]['nonlinear_iterations'],gmres_iterations=hist[step]['gmres_iterations'],
            fluid_solves=hist[step]['fluid_solves'],residual_to_tolerance=hist[step]['residual_to_tolerance'])
        rows.append(r)
    shared=[(old[round(r['time_ms'],7)],r) for r in rows if round(r['time_ms'],7) in old]
    assert len(shared)==3 and all(abs(b['time_ms']-s['time_ms'])<1e-7 for b,s in shared)
    comparisons=[]
    for b,s in shared:
        k=round(s['time_ms'],7);h=old_h[k]
        old_lumen=h['lumen_volume_ml'];new_lumen=s['lumen_volume_ml']
        comparisons.append(dict(time_ms=s['time_ms'],
            J_rms_baseline=b['J_rms_error'],J_rms_candidate=s['J_rms_error'],
            fraction_J_error_gt10pct_baseline=b['reference_fraction_abs_Jerror_gt10pct'],
            fraction_J_error_gt10pct_candidate=s['reference_fraction_abs_Jerror_gt10pct'],
            wall_volume_percent_baseline=b['wall_volume_change_percent'],wall_volume_percent_candidate=s['wall_volume_change_percent'],
            normal_diameter_mm_baseline=b['sampled_min_normal_diameter_mm'],normal_diameter_mm_candidate=s['sampled_min_normal_diameter_mm'],
            normal_diameter_relative_change_percent=100*(s['sampled_min_normal_diameter_mm']/b['sampled_min_normal_diameter_mm']-1),
            lumen_volume_ml_baseline=old_lumen,lumen_volume_ml_candidate=new_lumen,
            lumen_volume_relative_change_percent=100*(new_lumen/old_lumen-1),
            bending_mm_baseline=b['max_bending_from_end_line_mm'],bending_mm_candidate=s['max_bending_from_end_line_mm'],
            max_tension_difference_kpa=abs(s['maximum_applied_tension_kpa']-h['maximum_applied_tension_dyn_cm2']/1e4),
            nonlinear_iterations_baseline=h['nonlinear_iterations'],nonlinear_iterations_candidate=s['nonlinear_iterations'],
            gmres_iterations_baseline=h['gmres_iterations'],gmres_iterations_candidate=s['gmres_iterations'],
            fluid_solves_baseline=h['fluid_solves'],fluid_solves_candidate=s['fluid_solves']))
        if abs(s['time_ms']-280.)<1e-7:
            # This is the common INITIAL state, not an additional trial step.
            for prefix in ['nonlinear_iterations','gmres_iterations','fluid_solves']:
                comparisons[-1][prefix+'_baseline']=None
                comparisons[-1][prefix+'_candidate']=None
    # Compare full final electrical arrays, not just extrema.
    final=np.load(candidate/'final_state.npz');old_final=np.load(baseline/'continued_300ms/final_state.npz')
    electrical={}
    for k in ['voltage_v','gpb','land']:
        electrical[k]=dict(max_abs=float(abs(final[k]-old_final[k]).max()),relative_l2=rel(final[k],old_final[k]))
    electrical['max_voltage_difference_mv']=electrical['voltage_v']['max_abs']*1000
    electrical['max_calcium_difference_um']=float(abs(final['gpb'][:,37]-old_final['gpb'][:,37]).max()*1000)
    pvd_counts={}
    for kind in ['solid','fluid','ep']:
        entries=ET.parse(candidate/'vtk'/f'{kind}.pvd').findall('.//DataSet');pvd_counts[kind]=len(entries)
        if not all((candidate/'vtk'/e.attrib['file']).is_file() for e in entries):raise AssertionError('Broken PVD reference')
    unique=all(s.get('enclosing_loops',0)==1 for r in rows for s in r['sections'])
    checks=dict(initial_fields_bitwise_preserved=all(input_checks.values()),all_400_steps_accepted=run['accepted_steps']==400,
        all_J_positive=run['minimum_detF']>0,nonlinear_acceptance=run['max_sampled_residual_ratio']<=1,
        all_29_sections_valid=all(r['valid_sections']==29 for r in rows),unique_selected_sections=unique,
        full_electrical_voltage_unchanged=electrical['max_voltage_difference_mv']<=1e-5,
        full_electrical_calcium_unchanged=electrical['max_calcium_difference_um']<=1e-8,
        pvd_complete=len(set(pvd_counts.values()))==1 and pvd_counts['ep']==run['vtk_frames'])
    endpoint=comparisons[-1];result=dict(status='completed',numerical_checks_passed=all(checks.values()),
        scope=design['scope'],design=design,checks=checks,initial_field_checks=input_checks,
        common_times_ms=[r['time_ms'] for r in comparisons],comparisons=comparisons,electrical_final=electrical,
        endpoint_J_rms_reduction_percent=100*(1-endpoint['J_rms_candidate']/endpoint['J_rms_baseline']),
        endpoint_fraction_gt10pct_reduction_percent=100*(1-endpoint['fraction_J_error_gt10pct_candidate']/endpoint['fraction_J_error_gt10pct_baseline']),
        window_observed_compute_seconds_per_step=run['warmed_seconds_per_step'],
        total_compute_seconds=run['compute_seconds'],setup_seconds=run['setup_seconds'],
        maximum_sampled_residual_ratio=run['max_sampled_residual_ratio'],pvd_frames=pvd_counts,
        full_new_parameter_from_rest_validated=False,formal_spatial_convergence_established=False,
        matched_accuracy_speedup_established=False,
        limitations=['Only280-300ms after an explicit numerical-parameter switch; initial state evolved with old parameter.',
                     'Baseline full-window runtime and GPU occupancy differ: no speedup or net-overhead ratio.',
                     'Newton/GMRES counts are at shared diagnostic samples, not totals over all400steps.',
                     'The normal diameter is the shared central70% sampled minimum, not a unique anatomical definition.'])
    first=comparisons[0]
    initial_geometry_equal=(abs(first['J_rms_candidate']-first['J_rms_baseline'])<1e-12
        and abs(first['normal_diameter_mm_candidate']-first['normal_diameter_mm_baseline'])<1e-8
        and abs(first['lumen_volume_ml_candidate']-first['lumen_volume_ml_baseline'])<1e-12)
    result['checks']['initial_geometric_metrics_equal']=bool(initial_geometry_equal)
    result['numerical_checks_passed']=bool(all(result['checks'].values()))
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2))
    (OUT/'candidate_frame_metrics.json').write_text(json.dumps(rows,indent=2))
    # Four parallel metrics: mean volume, local RMS, mechanics and displaced lumen.
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.2),layout='constrained')
    old_plot=[dict(**r,lumen_volume_ml=old_h[t]['lumen_volume_ml']) for t,r in sorted(old.items())]
    specs=[('J_rms_error','Reference-volume RMS of J - 1'),
           ('reference_fraction_abs_Jerror_gt10pct','Reference wall fraction |J-1| > 10%'),
           ('sampled_min_normal_diameter_mm','Minimum normal-section diameter (mm)'),
           ('lumen_volume_ml','Lumen volume (mL)')]
    for ax,(key,label) in zip(axes.flat,specs):
        for rs,color,ls,name in [(old_plot,'#68737d','--','Existing baseline:53.85kPa'),(rows,'#176d9c','-','Candidate:73.08kPa')]:
            ax.plot([r['time_ms'] for r in rs],[r[key] for r in rs],color=color,ls=ls,marker='o',ms=3,label=name)
        ax.set(xlabel='Absolute electrical time (ms)',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Fine-wall numerical bulk sensitivity: same accepted 280ms state',fontsize=13)
    fig.supxlabel('Only the numerical bulk parameter changes. Existing baseline reused; no EP/fluid/solid refinement.',fontsize=9)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'comparison.{ext}',dpi=190)
    plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)
    if not result['numerical_checks_passed']:raise AssertionError('A declared numerical check failed')


if __name__=='__main__':main()
