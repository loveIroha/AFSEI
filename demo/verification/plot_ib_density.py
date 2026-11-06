"""Plot existing frozen-operator results; no solvers or time advancement."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True,type=Path);a=p.parse_args()
    r=json.loads(a.report.read_text());rows=r['records'];comparison=r['comparison']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.2),layout='constrained');pos=np.arange(2);colors=['#337e98','#bd6b3e']
    specs=[([v['point_count']/1e6 for v in rows],'Interaction quadrature points (million)','%.3f'),
           ([1000*v['prepare_interpolate_spread_median_s'] for v in rows],'Prepare + interpolate + spread (ms)','%.2f'),
           ([v['stencil_storage_bytes']/1024**2 for v in rows],'Logical storage per stencil (MiB)','%.1f'),
           ([v['structure_divergence_rms_s_inv'] for v in rows],'Structural velocity divergence RMS (1/s)','%.6f'),
           ([v['relative_volume_rate_s_inv'] for v in rows],'Instantaneous relative wall-volume rate (1/s)','%.7f')]
    for ax,(values,label,fmt) in zip(axes.flat,specs):
        bars=ax.bar(pos,values,color=colors,width=.6);ax.bar_label(bars,fmt=fmt,padding=4,fontsize=9)
        ax.set_xticks(pos,['Density 2','Density 3']);ax.set_ylabel(label)
        if min(values)<0:ax.set_ylim(min(values)*1.22,0)
        else:ax.set_ylim(0,max(values)*1.21)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax=axes[1,2]
    values=[comparison['velocity_relative_current_L2']*100,comparison['structure_divergence_relative_L2']*100,
            comparison['spread_force_relative_eulerian_L2']*100]
    bars=ax.bar(np.arange(3),values,color=['#337e98','#7a688f','#bd6b3e']);ax.bar_label(bars,fmt='%.4f%%',padding=4,fontsize=9)
    ax.set_xticks(np.arange(3),['Velocity','Divergence','Spread force']);ax.set_ylabel('Field L2 difference relative to density 3 (%)')
    ax.set_ylim(0,max(values)*1.22);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Frozen 280ms state: higher IB quadrature density changes transfer little and costs more',fontsize=13)
    fig.supxlabel('Same 25,920-node P1 wall, 64^3 MAC grid, kernel, mass matrix and fields. No time steps. Timing is for isolated cached operators.',fontsize=9)
    for ext in ['png','pdf','svg']:fig.savefig(a.report.parent/f'comparison.{ext}',dpi=190)
    plt.close(fig)


if __name__=='__main__':main()
