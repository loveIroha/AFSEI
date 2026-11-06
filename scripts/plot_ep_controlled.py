"""Three-way activation/waveform plots and isolated discretization/kernel timing."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES=['consistent_tc','nodal_tc','nodal_afsei']
LABELS=['A: consistent P1 / TorchCor CG','B: nodal P1 / TorchCor CG','C: nodal P1 / AFSEI CG']
COLORS=['#64748b','#0ea5e9','#f97316']


def main():
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();root=Path(a.folder)
    report=json.loads((root/'report.json').read_text());cfg=json.loads((root/'config.json').read_text())
    fig,axes=plt.subplots(2,4,figsize=(18,8),layout='constrained')
    edges,edge_axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    summary=[]
    for row,dim in enumerate([2,3]):
        data=[np.load(root/f'{dim}d'/name/'fields.npz') for name in NAMES]
        X=data[0]['X'];plane=np.ones(len(X),dtype=bool) if dim==2 else abs(X[:,2])<1e-12
        lo=min(np.nanmin(z['activation_s'][plane])*1000 for z in data);hi=max(np.nanmax(z['activation_s'][plane])*1000 for z in data)
        for col,(z,label,color) in enumerate(zip(data,LABELS,COLORS)):
            a=axes[row,col];im=a.tricontourf(X[plane,0],X[plane,1],z['activation_s'][plane]*1000,
                levels=np.linspace(lo,hi,17),cmap='turbo')
            a.set(xlabel='x (cm)',ylabel='y (cm)');a.set_title(f'{dim}D {label}\nActivation (ms)'+(' at z=0' if dim==3 else ''),fontsize=10)
            a.set_aspect('equal');fig.colorbar(im,ax=a,shrink=.6)
            traces=z['traces_v'];times=np.arange(len(traces))*cfg['dt_ep_s']*1000
            # Dashed C remains visible when same-equation B/C overlap.
            axes[row,3].plot(times,traces[:,1]*1000,label=label,color=color,ls='--' if col==2 else '-',lw=1.8 if col==2 else 1.4)
            line=np.all(abs(X[:,1:])<1e-12,axis=1)
            edge_axes[row].plot(X[line,0],z['activation_s'][line]*1000,label=label,color=color,ls='--' if col==2 else '-')
        axes[row,3].set(xlabel='Time (ms)',ylabel='Voltage (mV)',title=f'{dim}D far-corner action potential')
        axes[row,3].legend(fontsize=7);axes[row,3].grid(alpha=.2)
        edge_axes[row].set(xlabel='x (cm)',ylabel='Activation time (ms)',title=f'{dim}D edge from stimulated corner')
        edge_axes[row].legend(fontsize=8);edge_axes[row].grid(alpha=.2)
    fig.suptitle('Same GPB + 200 ms conditioning | old benchmark conductivities | CN dt = 0.05 ms',fontsize=12)
    fig.savefig(root/'activation_and_waveforms.png',dpi=180);fig.savefig(root/'activation_and_waveforms.pdf');plt.close(fig)
    edges.savefig(root/'activation_edges.png',dpi=180);plt.close(edges)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,case in enumerate(report['cases']):
        dim=case['dimension'];summary_row={'dimension':dim}
        for row,scope in enumerate(['trajectory','actual_ionic']):
            values=case['methods'] if row==0 else case['frozen_replay'][scope]
            vals=[values[name]['pde_mean_ms'] for name in NAMES]
            errors=None
            if row==1:
                errors=np.array([[v-min(values[name]['repeat_ms']),max(values[name]['repeat_ms'])-v]
                                 for name,v in zip(NAMES,vals)]).T
            bars=axes[row,col].bar(['A: standard/TC','B: nodal/TC','C: nodal/AFSEI'],vals,color=COLORS,
                                  yerr=errors,capsize=4)
            axes[row,col].bar_label(bars,fmt='%.3f',padding=3);axes[row,col].set_ylim(0,max(vals)*1.25)
            axes[row,col].set(ylabel='PDE time per step (ms)',title=f'{dim}D '+('full GPB trajectory' if row==0 else 'identical frozen V / GPB current / stimulus'))
            axes[row,col].grid(axis='y',alpha=.2);axes[row,col].set_axisbelow(True)
            summary_row[scope]={'spatial_speedup_A_over_B':vals[0]/vals[1],
                'afsei_vs_tc_same_nodal_cost_C_over_B':vals[2]/vals[1],'overall_A_over_C':vals[0]/vals[2]}
        summary_row['zero_ionic_replay']={name:case['frozen_replay']['zero_ionic'][name]['pde_mean_ms'] for name in NAMES}
        summary.append(summary_row)
    fig.suptitle('A/B: mass-quadrature benefit with shared CG | C/B: independent CG implementation cost\nCommon PDE wrapper, initial guess and true-residual criterion',fontsize=12)
    fig.savefig(root/'performance_decomposition.png',dpi=180);fig.savefig(root/'performance_decomposition.pdf');plt.close(fig)
    (root/'decomposition.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    # Primary comparison requested by user: standard TorchCor FEM versus nodal FEM.
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,dim in enumerate([2,3]):
        data=[np.load(root/f'{dim}d'/name/'fields.npz') for name in NAMES[:2]]
        X=data[0]['X'];plane=np.ones(len(X),dtype=bool) if dim==2 else abs(X[:,2])<1e-12
        lo=min(np.nanmin(z['activation_s'][plane])*1000 for z in data);hi=max(np.nanmax(z['activation_s'][plane])*1000 for z in data)
        for col,(z,label,color) in enumerate(zip(data,['TorchCor standard P1 FEM','Nodal P1 FEM'],COLORS[:2])):
            im=axes[row,col].tricontourf(X[plane,0],X[plane,1],z['activation_s'][plane]*1000,
                                       levels=np.linspace(lo,hi,17),cmap='turbo')
            axes[row,col].set(xlabel='x (cm)',ylabel='y (cm)')
            axes[row,col].set_title(f'{dim}D {label}\nActivation (ms)'+(' at z=0' if dim==3 else ''),fontsize=11)
            axes[row,col].set_aspect('equal');fig.colorbar(im,ax=axes[row,col],shrink=.65)
            trace=z['traces_v'];axes[row,2].plot(np.arange(len(trace))*cfg['dt_ep_s']*1000,trace[:,1]*1000,label=label,color=color)
        axes[row,2].set(xlabel='Time (ms)',ylabel='Voltage (mV)',title=f'{dim}D far-corner action potential')
        axes[row,2].legend(fontsize=8);axes[row,2].grid(alpha=.2)
    fig.suptitle('Standard FEM vs nodal FEM | identical GPB, mesh, CN and CG kernel',fontsize=12)
    fig.savefig(root/'standard_vs_nodal_activation_waveforms.png',dpi=180)
    fig.savefig(root/'standard_vs_nodal_activation_waveforms.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    x=np.arange(2)
    for method,label,color,offset in zip(NAMES[:2],['TorchCor standard P1 FEM','Nodal P1 FEM'],COLORS[:2],[-.18,.18]):
        vals=[r['frozen_replay']['actual_ionic'][method]['pde_mean_ms'] for r in report['cases']]
        errors=np.array([[v-min(r['frozen_replay']['actual_ionic'][method]['repeat_ms']),
                          max(r['frozen_replay']['actual_ionic'][method]['repeat_ms'])-v] for v,r in zip(vals,report['cases'])]).T
        bars=axes[0].bar(x+offset,vals,.35,color=color,label=label,yerr=errors,capsize=4)
        axes[0].bar_label(bars,fmt='%.3f ms',padding=4)
    axes[0].set(ylabel='PDE time per step (ms)',title='Identical frozen V / GPB current / stimulus',xticks=x,xticklabels=['2D: 2,349 nodes','3D: 30,537 nodes'])
    axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.25);axes[0].legend(fontsize=8)
    ratios=[r['actual_ionic']['spatial_speedup_A_over_B'] for r in summary]
    bars=axes[1].bar(x,ratios,color=COLORS[1],width=.55);axes[1].bar_label(bars,fmt='%.3fx',padding=4)
    axes[1].axhline(1,ls='--',color=COLORS[0]);axes[1].set(ylabel='Standard FEM time / nodal FEM time',title='Measured PDE speedup with shared CG',xticks=x,xticklabels=['2D','3D'],ylim=(0,max(ratios)*1.25))
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('FP64 | CN 0.05 ms | old benchmark parameters | same TorchCor CG kernel',fontsize=12)
    fig.savefig(root/'standard_vs_nodal_performance.png',dpi=180);fig.savefig(root/'standard_vs_nodal_performance.pdf');plt.close(fig)


if __name__=='__main__':main()
