"""Read-only comparison of planar muscle-stretch models."""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def inspect_case(path):
    report=json.loads((path/'report.json').read_text())
    history=json.loads((path/'history.json').read_text())
    final=history[-1]
    config=json.loads((path/'config.resolved.json').read_text())
    initial=config['geometry']['lumen_height_cm']
    volume_change=final['budget_fixed_end_volume_ml']-final['budget_initial_moving_volume_ml']
    budget=final['budget_fixed_control_imbalance_ml']
    missing=[];frames=0
    for p in (path/'vtk').glob('*.pvd'):
        for e in ET.parse(p).getroot().iter('DataSet'):
            frames+=1
            if not (p.parent/e.attrib['file']).exists():missing.append(str(p.parent/e.attrib['file']))
    result=dict(status=report['status'],time_ms=report.get('time_s',final['time_s'])*1000,
        last_sample_ms=final['time_s']*1000,accepted_steps=report['accepted_steps'],
        setup_seconds=report.get('setup_seconds'),elapsed_seconds=report.get('elapsed_seconds'),
        warmed_seconds_per_step=(report.get('seconds_per_step_excluding_first_10') if report['status']=='completed' else None),
        min_gap_cm=final['minimum_gap_cm'],min_gap_reduction_percent=100*(1-final['minimum_gap_cm']/initial),
        mid_gap_reduction_percent=100*(1-final['mid_gap_cm']/initial),
        mean_gap_reduction_percent=100*(1-final['mean_gap_cm']/initial),
        active_traction_max_kpa=final.get('active_traction_max_dyn_cm2',np.nan)/1e4,
        active_traction_mean_kpa=final.get('active_traction_mean_dyn_cm2',np.nan)/1e4,
        nodal_active_tension_max_kpa=final['applied_tension_increment_max_dyn_cm2']/1e4,
        fixed_control_change_ml=volume_change,budget_error_ml=budget,
        budget_error_fraction_of_volume_change=(abs(budget)/abs(volume_change) if abs(volume_change)>1e-14 else None),
        max_sampled_divergence_s_inv=report.get('max_sampled_divergence_s_inv'),
        vtk_dataset_count=frames,missing_vtk_files=missing,config=config)
    if report['status'] in ('completed','stopped-no-contact-limit'):
        with np.load(path/'final_state.npz') as s,np.load(path/'snapshots.npz') as a:
            result['minimum_snapshot_gap_ratio']=float(np.min(np.linalg.norm(a['x'][:,a['upper']]-a['x'][:,a['lower']],axis=-1))/initial)
            if 'x' in s:
                result['checkpoint_geometry_matches_last_frame']=bool(np.array_equal(s['x'],a['x'][-1]))
                result['checkpoint_time_matches_last_frame']=bool(float(s['time_s'])==float(a['times'][-1]))
    return result,history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--control',type=Path)
    p.add_argument('--prefix',type=Path,help='completed prefix reused by the continued run')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    baseline,h0=inspect_case(args.baseline);new,h1=inspect_case(args.new)
    # Earlier cases did not log actual traction; its saved-force audit is
    # carried explicitly from the independently verified 60 ms analysis.
    if abs(baseline['time_ms']-60)<1e-8:
        baseline['active_traction_max_kpa']=.8720883852397374
        baseline['active_traction_mean_kpa']=.06406907511510956
    result=dict(scope='planar reduced actuator validation; not quantitative paper reproduction',
        baseline=baseline,new=new,single_variable_comparison=False)
    if args.prefix:
        prefix,_=inspect_case(args.prefix);result['prefix']=prefix
        result['production_solve_seconds']=prefix['elapsed_seconds']+new['elapsed_seconds']
    with np.load(args.baseline/'snapshots.npz') as old:
        times=[];errors=[]
        for folder in ([args.prefix,args.new] if args.prefix else [args.new]):
            with np.load(folder/'snapshots.npz') as fresh:
                for j,t in enumerate(fresh['times']):
                    candidates=np.where(np.isclose(old['times'],t,rtol=0,atol=1e-12))[0]
                    if len(candidates):
                        i=int(candidates[0]);times.append(float(t)*1000)
                        errors.append(float(np.max(np.abs(old['voltage'][i]-fresh['voltage'][j]))))
        result['matched_voltage_times_ms']=times
        result['max_voltage_difference_mv_at_shared_times']=max(errors,default=None)
    if args.control:
        control,hc=inspect_case(args.control);result['no_stimulus_control']=control
        result['no_stimulus_max_sampled_gap_change_cm']=max(abs(r['minimum_gap_cm']-hc[0]['minimum_gap_cm']) for r in hc)
    if args.prefix:
        import os
        merged={}
        for kind in ('solid','fluid','ep'):
            datasets={}
            for folder in (args.prefix,args.new):
                path=folder/'vtk'/f'{kind}.pvd'
                if path.exists():
                    for e in ET.parse(path).getroot().iter('DataSet'):
                        datasets[float(e.attrib['timestep'])]=path.parent/e.attrib['file']
            if datasets:
                doc=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
                collection=ET.SubElement(doc,'Collection')
                for t,path in sorted(datasets.items()):
                    if not path.exists():raise ValueError(str(path))
                    ET.SubElement(collection,'DataSet',timestep=str(t),group='',part='0',file=os.path.relpath(path,args.output))
                ET.ElementTree(doc).write(args.output/f'{kind}.pvd',encoding='utf-8',xml_declaration=True)
                merged[kind]=len(datasets)
        result['merged_pvd_frames']=merged
    rows=[]
    for label,history in [('legacy_gap',h0),('isometric_hoop',h1)]:
        for row in history:
            rows.append(dict(model=label,time_ms=row['time_s']*1000,
                min_gap_cm=row['minimum_gap_cm'],
                active_traction_mean_kpa=row.get('active_traction_mean_dyn_cm2',np.nan)/1e4,
                budget_error_ml=row['budget_fixed_control_imbalance_ml']))
    with (args.output/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=True),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for label,h,c in [('Legacy gap proxy',h0,baseline['config']),('Isometric omitted hoop',h1,new['config'])]:
        t=np.array([r['time_s'] for r in h])*1000
        gap=np.array([r['minimum_gap_cm'] for r in h])/c['geometry']['lumen_height_cm']
        axes[0,0].plot(t,(1-gap)*100,label=label)
        axes[1,0].plot(t,np.array([r['budget_fixed_control_imbalance_ml'] for r in h])*1000,label=label)
        axes[1,1].plot(t,[r.get('active_muscle_stretch_min',v) for r,v in zip(h,gap)],label=label)
    axes[0,1].plot([r['time_s']*1000 for r in h1],
        [r['active_traction_mean_dyn_cm2']/1e4 for r in h1],color='C1',label='Isometric omitted hoop')
    axes[0,1].scatter([baseline['last_sample_ms']],[baseline['active_traction_mean_kpa']],
        color='C0',label='Legacy: audited final load')
    axes[0,0].set(ylabel='Minimum-gap reduction (%)',title='Inward contraction')
    axes[0,1].set(ylabel='Reference-length mean traction (kPa)',title='Actual actuator load')
    axes[1,0].set(ylabel='Cumulative budget difference (microlitre)',title='Volume / flow consistency')
    axes[1,1].set(ylabel='Minimum muscle stretch',title='Explicit reduced-model assumption')
    for ax in axes.flat:ax.set_xlabel('Time (ms)');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(args.output/'comparison.png',dpi=180);fig.savefig(args.output/'comparison.svg')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='config'} if isinstance(v,dict) else v
        for k,v in result.items()},indent=2,allow_nan=True))


if __name__=='__main__':main()
