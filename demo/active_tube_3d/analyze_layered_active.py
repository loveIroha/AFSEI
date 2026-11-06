"""Read existing active/passive component probes; never advance a solver."""
import argparse
import csv
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def read_case(folder):
    report=json.loads((folder/'report.json').read_text())
    history=json.loads((folder/'history.json').read_text())
    first,last=history[0],history[-1]
    initial=first['lumen_volume_ml']
    fixed_change=last['budget_fixed_end_volume_ml']-first['budget_fixed_end_volume_ml']
    valid_scale=abs(fixed_change)>1e-12*initial
    missing=[]
    for file in (folder/'vtk').glob('*.pvd'):
        for row in ET.parse(file).findall('.//DataSet'):
            if not (file.parent/row.attrib['file']).exists():missing.append(str(file.parent/row.attrib['file']))
    data=dict(status=report['status'],accepted_steps=last['step'],duration_ms=last['probe_time_ms'],
        relative_lumen_volume_change=(last['lumen_volume_ml']-initial)/initial,
        peak_displacement_um=max(r['max_total_displacement_cm'] for r in history)*10000,
        peak_end_displacement_um=max(r['maximum_end_displacement_cm'] for r in history)*10000,
        min_jelly_detF=min(r['cardiac-jelly_min_detF'] for r in history),
        min_myocardium_detF=min(r['myocardium_min_detF'] for r in history),
        peak_tension_kpa=max(r['maximum_applied_tension_dyn_cm2'] for r in history)/10000,
        fixed_control_volume_change_ml=fixed_change,
        integrated_net_inflow_ml=last['budget_integrated_net_inflow_ml'],
        final_fixed_control_imbalance_ml=last['budget_fixed_control_imbalance_ml'],
        max_abs_fixed_control_imbalance_ml=max(abs(r['budget_fixed_control_imbalance_ml']) for r in history),
        fixed_imbalance_fraction_of_volume_change=abs(last['budget_fixed_control_imbalance_ml']/fixed_change) if valid_scale else None,
        max_geometric_identity_error_ml=max(abs(r['budget_fixed_geometric_identity_error_ml']) for r in history),
        max_surface_quadrature_difference_ml_s=max(r.get('budget_surface_quadrature_difference_ml_s',0.) for r in history),
        max_nonlinear_residual_ratio=max((r['nonlinear_residual']/r['nonlinear_tolerance'] for r in history if r['nonlinear_tolerance']>0),default=0.),
        step_seconds=[r['step_seconds'] for r in history[1:]],
        fluid_solves=sum(r['fluid_solves'] for r in history[1:]),
        gmres_iterations=sum(r['gmres_iterations'] for r in history[1:]),
        helmholtz_sweeps=sum(r['helmholtz_sweeps'] for r in history[1:]),missing_vtk_references=missing)
    data['helmholtz_pcg_iterations']=sum(r.get('helmholtz_pcg_iterations',0) for r in history[1:])
    data['seconds_by_viscous_solver']={name:[r['step_seconds'] for r in history[1:] if r.get('viscous_solver','jacobi')==name]
        for name in ('jacobi','pcg')}
    return history,data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--active-subdir',default='active')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    prep=json.loads((args.root/'prepared_ep.json').read_text())
    runs={};summary=dict(scope='active-load verification with fixed-geometry EP preparation',
        physiological_restart=False,quantitative_paper_replication=False,preparation=prep,cases={})
    for name in ('active','passive'):
        folder=args.root/(args.active_subdir if name=='active' else name)
        if not (folder/'report.json').exists():continue
        history,data=read_case(folder);runs[name]=history;summary['cases'][name]=data
        keys=sorted(set().union(*(r.keys() for r in history)))
        with (args.output/f'{name}_history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(history)
    for name,path in [('matched_fluid_comparison',args.root/'viscous_comparison/report.json'),
                      ('surface_refinement',args.root/'surface_flux_level5.json'),
                      ('regional_wall_flux',args.root/'surface_flux_regions.json')]:
        if path.exists():summary[name]=json.loads(path.read_text())
    summary['active_combined_collections']=[]
    for kind in ('solid','fluid','ep'):
        entries={}
        for folder in (args.root/'active_ramp',args.root/args.active_subdir):
            file=folder/'vtk'/f'{kind}.pvd'
            if not file.exists():continue
            for item in ET.parse(file).findall('.//DataSet'):
                entries[float(item.attrib['timestep'])]=file.parent/item.attrib['file']
        if entries:
            doc=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
            collection=ET.SubElement(doc,'Collection')
            for time,file in sorted(entries.items()):
                ET.SubElement(collection,'DataSet',timestep=str(time),group='',part='0',file=os.path.relpath(file,args.output))
            name=f'active_{kind}.pvd'
            ET.ElementTree(doc).write(args.output/name,encoding='utf-8',xml_declaration=True)
            summary['active_combined_collections'].append(name)
    (args.output/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    axes[0,0].plot([r['electrical_time_ms'] for r in prep['history']],
                   [r['tension_kpa'] for r in prep['history']],'-o',ms=3)
    axes[0,0].axhline(prep['target_tension_kpa'],ls='--',color='gray',label='Preparation target')
    axes[0,0].set(xlabel='Electrical time (ms)',ylabel='Active tension (kPa)',title='Fixed-geometry electrical preparation')
    for name,h in runs.items():
        t=np.array([r['probe_time_ms'] for r in h]);v=np.array([r['lumen_volume_ml'] for r in h])
        axes[0,1].plot(t,100*(v-v[0])/v[0],'-o',ms=3,label=name)
        axes[1,1].plot(t,[r['maximum_end_displacement_cm']*10000 for r in h],'-o',ms=3,label=name)
        if name=='active':
            fixed=np.array([r['budget_fixed_end_volume_ml'] for r in h])
            net=np.array([r['budget_integrated_net_inflow_ml'] for r in h])
            axes[1,0].plot(t,-(fixed-fixed[0])*1e6,'-o',ms=3,label='Fixed-control volume decrease')
            axes[1,0].plot(t,-net*1e6,'--o',ms=3,label='Integrated net expelled fluid')
    axes[0,1].set(xlabel='Probe time (ms)',ylabel='Lumen volume change (%)',title='Active vs zero-active-load control')
    axes[1,0].set(xlabel='Probe time (ms)',ylabel='Volume (nL)',title='Fixed-end control volume and port flux')
    axes[1,1].set(xlabel='Probe time (ms)',ylabel='Maximum end displacement (µm)',title='Finite spring ends remain approximate')
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(args.output/f'active_validation.{ext}',dpi=180)
    print(json.dumps(summary['cases'],indent=2))


if __name__=='__main__':main()
