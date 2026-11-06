"""Plot existing frozen-field results; no numerical advancement."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True,type=Path);a=p.parse_args()
    r=json.loads(a.report.read_text());rows=r['levels'];q=r['common_quadrature']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.5),layout='constrained')
    indices=np.arange(3);colors=['#5a708f','#308b9b','#bd6947']
    labels=[f"Factor {x['factor']}\n{x['nodes']:,} nodes" for x in rows]
    error=[100*x['common_metrics']['velocity_relative_l2_difference_from_regularized_field'] for x in rows]
    bars=axes[0].bar(indices,error,color=colors,width=.65)
    axes[0].bar_label(bars,fmt='%.2f%%',padding=4)
    axes[0].set(title='Velocity approximation improves',ylabel='Relative L2 difference from regularized velocity (%)',ylim=(0,max(error)*1.19))
    div=[x['common_metrics']['rms_divergence_s_inv'] for x in rows]
    bars=axes[1].bar(indices,div,color=colors,width=.65)
    axes[1].bar_label(bars,fmt='%.2f',padding=4)
    axes[1].axhline(q['delta_divergence_rms_s_inv'],color='#262626',ls='--',label='Kernel field before FE projection')
    axes[1].set(title='Divergence does not converge to zero',ylabel='Current-volume RMS divergence (1/s)',ylim=(0,max(div)*1.35))
    axes[1].legend(loc='upper right',fontsize=8.5)
    means=[x['common_metrics']['mean_divergence_s_inv'] for x in rows]
    bars=axes[2].bar(indices,means,color=colors,width=.65)
    axes[2].bar_label(bars,fmt='%.3f',padding=4)
    axes[2].axhline(q['delta_divergence_mean_s_inv'],color='#262626',ls='--')
    axes[2].axhline(0,color='#888',lw=.7)
    axes[2].set(title='Instantaneous wall-volume rate',ylabel='(1/V) integral div(v) dV (1/s)',ylim=(min(means)*1.25,.15))
    for ax in axes:
        ax.set_xticks(indices,labels);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Same 120 ms wall geometry, same saved MAC velocity, unchanged IB operator',fontsize=14)
    fig.supxlabel(f"No time advancement. MAC divergence RMS over the fluid box: {r['native_mac_divergence']['rms_s_inv']:.2e} 1/s. "
                  'Dashed lines: analytical regularized field.',fontsize=9)
    for ext in ('png','pdf','svg'):
        fig.savefig(a.report.parent/f'comparison.{ext}',dpi=190)
    plt.close(fig)

if __name__=='__main__':main()
