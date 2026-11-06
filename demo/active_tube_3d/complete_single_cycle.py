"""Finish the authorized single-tube experiment without repeating its prefix.

Wait only for the named prefix to finish, verify exact accepted-state recovery,
then continue the identical model at the identical dt. No failed-step restart
or automatic model retuning is performed. Never manages unrelated processes.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET
import numpy as np
from afsi_torch.cycle_checkpoint import atomic_json


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from afsi_torch.simulation.circular_tube_config import resolve_config
    from afsi_torch.solids.circular_tube import generate_circular_tube
    from afsi_torch.simulation.tube_validation import contraction_metrics
    folders=[root/'active_120ms',root/'active_continued_800ms']
    rows={}
    reports=[]
    for folder in folders:
        reports.append(json.loads((folder/'report.json').read_text()))
        for row in json.loads((folder/'history.json').read_text()):rows.setdefault(row['step'],row)
    h=[rows[k] for k in sorted(rows)]
    out=root/'analysis';out.mkdir(exist_ok=True)
    c=json.loads((folders[-1]/'config.resolved.json').read_text());g,_,_,_=resolve_config(c)
    mesh=generate_circular_tube(g)
    with np.load(folders[-1]/'final_state.npz') as a:
        final_geometry=contraction_metrics(a['reference'],a['x'],mesh.inner_rings.numpy(),g.center_xy_cm)
    radius=np.array([r['minimum_radius_ratio'] for r in h]);volume=np.array([r['lumen_volume_ml'] for r in h])
    ta=np.array([r['maximum_applied_tension_dyn_cm2'] for r in h]);times=np.array([r['time_s'] for r in h])
    imax=int(np.argmin(radius));ipeak=int(np.argmax(ta))
    missing=[];frames={}
    for kind in ['solid','fluid','ep']:
        entries={}
        for folder in folders:
            p=folder/'vtk'/f'{kind}.pvd'
            for item in ET.parse(p).findall('.//DataSet'):
                t=float(item.attrib['timestep']);source=p.parent/item.attrib['file']
                if not source.exists():missing.append(str(source))
                entries.setdefault(t,os.path.relpath(source,out))
        xml=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
        col=ET.SubElement(xml,'Collection')
        for t,path in sorted(entries.items()):ET.SubElement(col,'DataSet',timestep=repr(t),group='',part='0',file=path)
        ET.ElementTree(xml).write(out/f'{kind}.pvd',encoding='utf-8',xml_declaration=True)
        frames[kind]=len(entries)
    solve=sum(r['elapsed_seconds'] for r in reports);setup=sum(r['setup_seconds'] for r in reports)
    info=dict(status='completed',planned_end_ms=800.,reached_time_ms=float(times[-1]*1000),
        sampled_peak_tension_kpa=float(ta[ipeak]/1e4),sampled_peak_tension_time_ms=float(times[ipeak]*1000),
        sampled_maximum_radius_reduction_percent=float(100*(1-radius[imax])),
        sampled_maximum_section_area_reduction_percent=float(100*(1-radius[imax]**2)),
        sampled_maximum_contraction_time_ms=float(times[imax]*1000),
        sampled_minimum_equivalent_diameter_mm=float(2*g.inner_radius_cm*10*radius.min()),
        sampled_maximum_node_displacement_mm=max(r['max_total_displacement_cm'] for r in h)*10,
        final_radius_ratio=float(radius[-1]),final_volume_relative_change=float(volume[-1]/volume[0]-1),
        final_tension_fraction_of_peak=float(ta[-1]/max(ta.max(),1e-30)),
        final_geometry=final_geometry,solve_seconds=solve,setup_seconds=setup,
        accepted_steps=sum(r['accepted_steps_this_run'] for r in reports),
        mean_seconds_per_step=solve/sum(r['accepted_steps_this_run'] for r in reports),
        recovery_metrics=dict(radius_within_5_percent=bool(abs(radius[-1]-1)<.05),
            volume_within_5_percent=bool(abs(volume[-1]/volume[0]-1)<.05),
            tension_below_1_percent=bool(ta[-1]<.01*ta.max())),
        literature_replication=False,convergence_validated=False,
        missing_pvd_targets=missing,pvd_frames=frames)
    atomic_json(out/'summary.json',info);atomic_json(out/'history.json',h)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    axs[0,0].plot(times*1000,ta/1e4);axs[0,0].set(ylabel='Applied tension maximum (kPa)',title='Isometric Land amplitude; natural relaxation')
    axs[0,1].plot(times*1000,100*(1-radius));axs[0,1].set(ylabel='Minimum equivalent radius reduction (%)',title='Contraction and recovery')
    axs[1,0].plot(times*1000,volume/volume[0]);axs[1,0].axhline(1,color='gray',ls=':');axs[1,0].set(ylabel='Lumen volume / reference volume')
    axs[1,1].plot(times*1000,[r['voltage_max_mv'] for r in h],label='Maximum Vm');axs[1,1].plot(times*1000,[r['voltage_min_mv'] for r in h],label='Minimum Vm');axs[1,1].set(ylabel='Voltage (mV)');axs[1,1].legend()
    for ax in axs.flat:ax.set_xlabel('Time (ms)');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'full_cycle.png',dpi=180);plt.close(fig)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--wait-seconds',type=float,default=7200)
    a=p.parse_args();root=a.root;prefix=root/'active_120ms';manifest=root/'full_run_status.json'
    c=json.loads(a.config.read_text())
    if c['time']['end_time_s']!=.8 or c['active'].get('stretch_feedback')!='isometric-driver':
        raise ValueError('this orchestration entry expects the explicit 800 ms single-tube configuration')
    atomic_json(manifest,dict(status='waiting-for-prefix',pid=os.getpid(),prefix=str(prefix)))
    try:
        started=time.monotonic()
        while True:
            report=prefix/'report.json';state=prefix/'final_state.npz'
            if report.exists():
                r=json.loads(report.read_text())
                if r['status']=='failed':raise RuntimeError('prefix failed; see its report and log')
                if r['status']=='completed' and state.exists():
                    with np.load(state) as check:
                        if abs(float(check['time_s'])-.12)>1e-12:raise RuntimeError('unexpected prefix clock')
                    break
            if time.monotonic()-started>a.wait_seconds:raise TimeoutError('prefix did not finish within the specified wait')
            time.sleep(10)
        from afsi_torch.simulation.circular_tube import run
        check_config=json.loads(json.dumps(c));check_config['time']['end_time_s']=.12
        check_dir=root/'checkpoint_roundtrip'
        run(check_config,check_dir,device=a.device,resume=state)
        with np.load(state) as old,np.load(check_dir/'final_state.npz') as new:
            equal={k:bool(np.array_equal(old[k],new[k])) for k in old.files}
        atomic_json(root/'checkpoint_roundtrip.json',dict(passed=all(equal.values()),fields=equal))
        if not all(equal.values()):raise RuntimeError('checkpoint roundtrip changed accepted physical state')
        atomic_json(manifest,dict(status='running',pid=os.getpid(),start_ms=120.,target_ms=800.,checkpoint_roundtrip_passed=True))
        run(c,root/'active_continued_800ms',device=a.device,resume=state)
        summary=summarize(root)
        atomic_json(manifest,dict(status='completed',pid=os.getpid(),summary=summary))
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        atomic_json(manifest,dict(status='failed',pid=os.getpid(),error=str(exc)))
        traceback.print_exc();raise


if __name__=='__main__':main()
