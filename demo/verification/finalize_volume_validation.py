"""One-shot dependent postprocess for the two new 300ms runs; no simulation.

Waits for these specific jobs only. No parameter changes, retries, alerts,
or edits to earlier datasets. Failure remains failure, never a passing run.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/large2/gjh/AFSEI/results');OUT=ROOT/'volumetric_stabilization_20261025/from_rest_300ms'

def save(path,data):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(path)

def history(folder):
    rows={}
    for phase in ('prefix_60ms','continued_300ms'):
        for h in json.loads((folder/phase/'history.json').read_text()):rows[round(h['time_s'],10)]=h
    return [rows[t] for t in sorted(rows)]

def metrics(rows):
    reference=rows[0]['wall_volume_cm3']
    return dict(final_time_ms=rows[-1]['time_s']*1000,
        max_sampled_wall_volume_error_percent=100*max(abs(h['wall_volume_cm3']/reference-1) for h in rows),
        final_wall_volume_change_percent=100*(rows[-1]['wall_volume_cm3']/reference-1),
        minimum_detF=min(h['minimum_detF'] for h in rows),maximum_detF=max(h['maximum_detF'] for h in rows),
        minimum_projected_diameter_mm=20*np.sqrt(min(h['minimum_section_area_cm2'] for h in rows)/np.pi),
        peak_tension_kpa=max(h['maximum_applied_tension_dyn_cm2'] for h in rows)/1e4,
        peak_tension_time_ms=max(rows,key=lambda h:h['maximum_applied_tension_dyn_cm2'])['time_s']*1000)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    started=time.monotonic()
    while True:
        states={}
        for f in (1,2):
            p=OUT/f'solid{f}/status.json';states[str(f)]=json.loads(p.read_text()) if p.exists() else dict(stage='starting')
        if any(v.get('stage')=='failed' for v in states.values()):
            save(OUT/'validation_status.json',dict(stage='failed',cases=states,parameters_retried=False));return
        if all(v.get('stage')=='completed' for v in states.values()):break
        save(OUT/'validation_status.json',dict(stage='running',cases=states,full_validation_complete=False))
        if not args.wait:return
        if time.monotonic()-started>14400:
            save(OUT/'validation_status.json',dict(stage='postprocess-timeout',simulations_not_interrupted=True,cases=states));return
        time.sleep(30)
    results={};series={}
    for f,old in ((1,'coarse_box2'),(2,'fine_box2')):
        original=ROOT/'domain_space_campaign_20261025'/old;new=OUT/f'solid{f}'
        b,s=history(original),history(new);bm,sm=metrics(b),metrics(s)
        assert abs(sm['final_time_ms']-300)<1e-7
        pair={};lookup={round(h['time_s'],10):h for h in b}
        pair['maximum_common_tension_difference_kpa']=max(abs(h['maximum_applied_tension_dyn_cm2']-
            lookup[round(h['time_s'],10)]['maximum_applied_tension_dyn_cm2'])/1e4 for h in s)
        a=np.load(new/'continued_300ms/final_state.npz');X=a['reference'];cells=a['cells'];x=a['x']
        D=(X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1)
        d=(x[cells[:,1:]]-x[cells[:,:1]]).transpose(0,2,1)
        vol=np.linalg.det(D)/6;J=np.linalg.det(d)/np.linalg.det(D)
        sm['final_J_rms_error']=float(np.sqrt((vol*(J-1)**2).sum()/vol.sum()))
        centers=x[a['inner_rings']].mean(1)
        sm['final_centerline_transverse_displacement_mm']=float(np.linalg.norm(centers[:,:2]-.7,axis=1).max()*10)
        reports=[json.loads((new/p/'report.json').read_text()) for p in ('prefix_60ms','continued_300ms')]
        sm['compute_wall_seconds_including_output']=sum(r['elapsed_seconds'] for r in reports)
        sm['observed_seconds_per_step']=sm['compute_wall_seconds_including_output']/6000
        pair.update(baseline=bm,stabilized=sm)
        results[f'solid{f}']=pair;series[f'solid{f}']=dict(baseline=b,stabilized=s)
    save(OUT/'comparison.json',dict(stage='completed',from_rest=True,results=results,
        formal_spatial_convergence_passed=False,matched_accuracy_performance_speedup_established=False,
        limitations=['Two solid spaces only; this report does not set a spatial-convergence pass.',
                     'Projected diameter requires caution if bending is large.',
                     'Different discrete formulations and possible shared GPU0 occupancy; timings are observations.']))
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for f,color in ((1,'#176d9c'),(2,'#be602b')):
        for name,ls in (('baseline','--'),('stabilized','-')):
            rows=series[f'solid{f}'][name];t=[h['time_s']*1000 for h in rows];V0=rows[0]['wall_volume_cm3']
            axes[0].plot(t,[100*(h['wall_volume_cm3']/V0-1) for h in rows],ls=ls,color=color,label=f'Solid {f} / {name}')
            axes[1].plot(t,[20*np.sqrt(h['minimum_section_area_cm2']/np.pi) for h in rows],ls=ls,color=color)
            axes[2].plot(t,[h['maximum_applied_tension_dyn_cm2']/1e4 for h in rows],ls=ls,color=color)
    for ax,label in zip(axes,['Wall-volume change (%)','Minimum x-y projected diameter (mm)','Maximum input tension (kPa)']):
        ax.set(xlabel='Time from rest (ms)',ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('From-rest 300ms stabilization comparison; original baselines reused')
    for ext in ('png','pdf'):fig.savefig(OUT/f'comparison.{ext}',dpi=180)
    plt.close(fig);save(OUT/'validation_status.json',dict(stage='completed',from_rest_300ms_completed=True,
        comparison='comparison.json',formal_spatial_convergence_passed=False))

if __name__=='__main__':main()
