"""Bulk-modulus sensitivity probe; same shear modulus and existing constitutive law.

The changed bulk resistance is started from rest to avoid an artificial
jump in stored volumetric stress. Split at 260 ms to retain a pre-bending accepted checkpoint.
"""
import argparse,json,os,traceback
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from afsi_torch.cycle_checkpoint import atomic_json


def compare(root,baseline):
    import meshio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from afsi_torch.simulation.circular_tube_config import resolve_config
    from afsi_torch.solids.circular_tube import generate_circular_tube
    folders=[f for f in [root/'prefix_260ms',root/'continued_350ms'] if (f/'history.json').exists()]
    out=root/'analysis';out.mkdir(exist_ok=True)
    merged={};reports=[]
    for f in folders:
        reports.append(json.loads((f/'report.json').read_text()))
        for row in json.loads((f/'history.json').read_text()):merged.setdefault(row['step'],row)
    h=[merged[k] for k in sorted(merged)]
    base=[r for r in json.loads((baseline/'analysis/history.json').read_text()) if r['time_s']<=.35000001]
    c=json.loads((folders[0]/'config.resolved.json').read_text());g,_,_,_=resolve_config(c)
    m=generate_circular_tube(g);rings=m.inner_rings.numpy()
    oldc=json.loads((baseline/'active_120ms/config.resolved.json').read_text())
    bulk=c['solid']['poisson_ratio']!=oldc['solid']['poisson_ratio']
    label_new='Higher bulk modulus' if bulk else '10x spring'
    change='Bulk modulus increased with shear modulus held fixed' if bulk else 'End-spring stiffness: 1e6 -> 1e7 dyn/cm3'
    geometry=[];missing=[]
    for kind in ['solid','fluid','ep']:
        entries={}
        for f in folders:
            p=f/'vtk'/f'{kind}.pvd'
            for e in ET.parse(p).findall('.//DataSet'):
                time=float(e.attrib['timestep']);source=p.parent/e.attrib['file']
                entries.setdefault(time,source)
                if not source.exists():missing.append(str(source))
        xml=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian');collection=ET.SubElement(xml,'Collection')
        for time,path in sorted(entries.items()):
            ET.SubElement(collection,'DataSet',timestep=repr(time),group='',part='0',file=os.path.relpath(path,out))
            if kind=='solid':
                x=meshio.read(path).points;q=x[rings];center=q.mean(1)
                shift=np.linalg.norm(center[:,:2]-np.array(g.center_xy_cm),axis=1)
                gap=np.linalg.norm(q-np.roll(q,len(rings[0])//2,axis=1),axis=2)
                geometry.append(dict(time_ms=time*1000,max_centerline_shift_mm=float(shift.max()*10),
                    minimum_3d_opposed_chord_mm=float(gap.min()*10)))
        ET.ElementTree(xml).write(out/f'{kind}.pvd',encoding='utf-8',xml_declaration=True)
    last=folders[-1]/'final_state.npz'
    if last.exists():
        with np.load(last) as a:
            t=float(a['time_s']);x=a['x'];q=x[rings];center=q.mean(1)
        if not geometry or t*1000>geometry[-1]['time_ms']+1e-8:
            geometry.append(dict(time_ms=t*1000,max_centerline_shift_mm=float(np.linalg.norm(center[:,:2]-np.array(g.center_xy_cm),axis=1).max()*10),minimum_3d_opposed_chord_mm=float(np.linalg.norm(q-np.roll(q,len(rings[0])//2,axis=1),axis=2).min()*10)))
    old=json.loads((baseline/'analysis/geometry_audit.json').read_text())['saved_frame_metrics']
    old=[r for r in old if r['time_ms']<=350.00001]
    info=dict(status='completed' if all(r['status']=='completed' for r in reports) else 'stopped-by-guard',changed_physical_parameter=change,
        end_time_ms=max(r['time_s'] for r in reports)*1000,
        baseline_max_centerline_shift_mm=max(r['maximum_transverse_centerline_shift_mm'] for r in old),
        probe_max_centerline_shift_mm=max(r['max_centerline_shift_mm'] for r in geometry),
        baseline_first_saved_shift_over_0p1_mm_ms=next((r['time_ms'] for r in old if r['maximum_transverse_centerline_shift_mm']>.1),None),
        probe_first_saved_shift_over_0p1_mm_ms=next((r['time_ms'] for r in geometry if r['max_centerline_shift_mm']>.1),None),
        baseline_max_centerline_shift_common_window_mm=max(r['maximum_transverse_centerline_shift_mm'] for r in old if r['time_ms']<=max(p['time_s'] for p in reports)*1000),
        baseline_max_end_displacement_mm=max(r['maximum_end_displacement_cm'] for r in base)*10,
        probe_max_end_displacement_mm=max(r['maximum_end_displacement_cm'] for r in h)*10,
        baseline_min_equivalent_radius_ratio=min(r['minimum_radius_ratio'] for r in base),
        probe_min_equivalent_radius_ratio=min(r['minimum_radius_ratio'] for r in h),
        probe_min_detF=min(r['minimum_detF'] for r in h),
        baseline_max_wall_volume_loss_percent=100*(1-min(r['wall_volume_cm3'] for r in base)/base[0]['wall_volume_cm3']),
        probe_max_wall_volume_loss_percent=100*(1-min(r['wall_volume_cm3'] for r in h)/h[0]['wall_volume_cm3']),
        probe_max_sampled_residual_ratio=max(r['nonlinear_residual']/max(r['nonlinear_tolerance'],1e-30) for r in h),
        solve_seconds=sum(r['elapsed_seconds'] for r in reports) if all('elapsed_seconds' in r for r in reports) else None,missing_pvd_targets=missing,
        interpretation='Boundary sensitivity diagnostic, not proof of physical or numerical origin and not a matched performance benchmark')
    atomic_json(out/'summary.json',info);atomic_json(out/'history.json',h);atomic_json(out/'geometry.json',geometry)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    axs[0,0].plot([r['time_ms'] for r in old],[r['maximum_transverse_centerline_shift_mm'] for r in old],label='Original spring')
    axs[0,0].plot([r['time_ms'] for r in geometry],[r['max_centerline_shift_mm'] for r in geometry],label=label_new)
    axs[0,0].set(ylabel='Centerline transverse shift (mm)',title='Lateral bending')
    for rows,label in [(base,'Original spring'),(h,label_new)]:
        t=[r['time_s']*1000 for r in rows]
        axs[0,1].plot(t,[r['minimum_radius_ratio'] for r in rows],label=label)
        axs[1,0].plot(t,[r['maximum_end_displacement_cm']*10 for r in rows],label=label)
        axs[1,1].plot(t,[100*(r['wall_volume_cm3']/rows[0]['wall_volume_cm3']-1) for r in rows],label=label)
    axs[0,1].set(ylabel='Minimum equivalent radius / reference',title='Contraction')
    axs[1,0].set(ylabel='Maximum end-node displacement (mm)',title='Finite support motion')
    axs[1,1].set(ylabel='Wall volume change (%)',title='Structural volume preservation')
    for ax in axs.flat:ax.set_xlabel('Time (ms)');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'support_comparison.png',dpi=170);plt.close(fig)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('choose an empty output directory')
    a.output.mkdir(parents=True)
    status=a.output/'status.json'
    c=json.loads(a.config.read_text());old=json.loads((a.baseline/'active_120ms/config.resolved.json').read_text())
    def normalized(v):
        v=json.loads(json.dumps(v));v.pop('description');v['time'].pop('end_time_s');v['solid'].pop('young_modulus_dyn_cm2');v['solid'].pop('poisson_ratio')
        return v
    mu=lambda s:s['young_modulus_dyn_cm2']/(2*(1+s['poisson_ratio']))
    if normalized(c)!=normalized(old) or not np.isclose(mu(c['solid']),mu(old['solid']),rtol=1e-13) or c['solid']['poisson_ratio']!=.45:
        raise ValueError('only bulk modulus may change; shear modulus and all other parameters must match')
    bulk=lambda s:s['young_modulus_dyn_cm2']/(3*(1-2*s['poisson_ratio']))
    atomic_json(a.output/'case_comparison.json',dict(original_shear_modulus=mu(old['solid']),new_shear_modulus=mu(c['solid']),original_bulk_modulus=bulk(old['solid']),new_bulk_modulus=bulk(c['solid']),all_other_physical_and_numerical_parameters_equal=True))
    try:
        from afsi_torch.simulation.circular_tube import run
        atomic_json(status,dict(status='running-prefix',pid=os.getpid(),target_ms=350.))
        prefix=json.loads(json.dumps(c));prefix['time']['end_time_s']=.26
        run(prefix,a.output/'prefix_260ms',device='cuda:0')
        atomic_json(status,dict(status='running-continuation',pid=os.getpid(),reached_ms=260.,target_ms=350.))
        run(c,a.output/'continued_350ms',device='cuda:0',resume=a.output/'prefix_260ms/final_state.npz')
        info=compare(a.output,a.baseline)
        atomic_json(status,dict(status='completed',pid=os.getpid(),summary=info));print(json.dumps(info,indent=2),flush=True)
    except Exception as exc:
        analysis=None
        try:
            if (a.output/'prefix_260ms/history.json').exists():analysis=compare(a.output,a.baseline)
        except Exception as analysis_exc:analysis={'analysis_error':str(analysis_exc)}
        atomic_json(status,dict(status='failed',pid=os.getpid(),error=str(exc),partial_analysis=analysis))
        traceback.print_exc();raise


if __name__=='__main__':main()
