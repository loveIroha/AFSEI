"""Fair-window analysis of the existing bulk-control run, no simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);a=p.parse_args()
    base=json.loads((a.baseline/'analysis/history.json').read_text())
    probe=json.loads((a.probe/'analysis/history.json').read_text())
    bg=json.loads((a.baseline/'analysis/geometry_audit.json').read_text())['saved_frame_metrics']
    pg=json.loads((a.probe/'analysis/geometry.json').read_text())
    end=.3
    stats={}
    for name,rows,geo,field in [('baseline',base,bg,'maximum_transverse_centerline_shift_mm'),('bulk_control',probe,pg,'max_centerline_shift_mm')]:
        h=[r for r in rows if r['time_s']<=end+1e-12];g=[r for r in geo if r['time_ms']<=end*1000+1e-8]
        v0=h[0]['wall_volume_cm3'];last=h[-1]
        ratio=min(r['minimum_radius_ratio'] for r in h)
        stats[name]=dict(common_window_end_ms=last['time_s']*1000,
            wall_volume_loss_at_300ms_percent=100*(1-last['wall_volume_cm3']/v0),
            max_wall_volume_loss_common_window_percent=100*(1-min(r['wall_volume_cm3'] for r in h)/v0),
            centerline_shift_at_300ms_mm=g[-1][field],
            minimum_equivalent_diameter_mm=6*ratio,
            maximum_equivalent_diameter_reduction_percent=100*(1-ratio),
            maximum_section_area_reduction_percent=100*(1-ratio**2),
            minimum_sampled_detF=min(r['minimum_detF'] for r in h),
            maximum_sampled_detF=max(r['maximum_detF'] for r in h),
            maximum_end_displacement_mm=max(r['maximum_end_displacement_cm'] for r in h)*10,
            maximum_sampled_courant=max(r['courant'] for r in h),
            max_sampled_residual_ratio=max(r['nonlinear_residual']/max(r['nonlinear_tolerance'],1e-30) for r in h))
    bystep={r['step']:r for r in base}
    keys=['voltage_min_mv','voltage_max_mv','calcium_min_um','calcium_max_um','maximum_applied_tension_dyn_cm2']
    difference={key:max(abs(r[key]-bystep[r['step']][key]) for r in probe if r['step'] in bystep) for key in keys}
    stats['maximum_absolute_shared_sample_difference']=difference
    stats['bulk_status']=json.loads((a.probe/'status.json').read_text())['status']
    stats['bulk_stop_time_ms']=json.loads((a.probe/'continued_350ms/report.json').read_text())['time_s']*1000
    stats['scope']='Common 0-300 ms window. Identical shear modulus, support, electrical and active input. No independent-reference or convergence validation.'
    out=a.probe/'analysis';(out/'common_window.json').write_text(json.dumps(stats,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for name,rows,geo,field,label in [('baseline',base,bg,'maximum_transverse_centerline_shift_mm','Original case'),('bulk_control',probe,pg,'max_centerline_shift_mm','Higher bulk modulus')]:
        h=[r for r in rows if r['time_s']<=end+1e-12];g=[r for r in geo if r['time_ms']<=end*1000+1e-8];t=[r['time_s']*1000 for r in h]
        axs[0,0].plot([r['time_ms'] for r in g],[r[field] for r in g],label=label)
        axs[0,1].plot(t,[100*(r['wall_volume_cm3']/h[0]['wall_volume_cm3']-1) for r in h],label=label)
        axs[1,0].plot(t,[100*(1-r['minimum_radius_ratio']) for r in h],label=label)
        axs[1,1].plot(t,[r['maximum_end_displacement_cm']*10 for r in h],label=label)
    axs[0,0].set(ylabel='Transverse centerline shift (mm)',title='Lateral response')
    axs[0,1].set(ylabel='Wall volume change (%)',title='Volume preservation improves')
    axs[1,0].set(ylabel='Equivalent diameter reduction (%)',title='Strong contraction remains')
    axs[1,1].set(ylabel='Maximum end-node displacement (mm)',title='Finite end supports')
    for ax in axs.flat:ax.set(xlabel='Time (ms)',xlim=(0,300));ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Common 0-300 ms window; bulk-control run later stopped at 304.95 ms')
    fig.tight_layout();fig.savefig(out/'bulk_common_window.png',dpi=175);plt.close(fig)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
