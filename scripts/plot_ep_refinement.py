"""Two-level comparison of standard P1 and nodal P1 using shared TorchCor CG."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES=['consistent_tc','nodal_tc']
LABELS=['TorchCor standard P1 FEM','Nodal P1 FEM']
COLORS=['#64748b','#0ea5e9']


def main():
    p=argparse.ArgumentParser();p.add_argument('--coarse',required=True);p.add_argument('--fine',required=True);a=p.parse_args()
    coarse=Path(a.coarse);fine=Path(a.fine)
    reports=[json.loads((r/'report.json').read_text()) for r in [coarse,fine]]
    cfg=json.loads((fine/'config.json').read_text());summary=[]
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    conv,ca=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for row,dim in enumerate([2,3]):
        sets=[[np.load(root/f'{dim}d'/name/'fields.npz') for name in NAMES] for root in [coarse,fine]]
        metrics=[]
        for root,pair,report in zip([coarse,fine],sets,reports):
            r=next(c for c in report['cases'] if c['dimension']==dim)
            t=[r['methods'][name]['far_corner_activation_ms'] for name in NAMES]
            lat=[z['activation_s'] for z in pair];mask=np.isfinite(lat[0])&np.isfinite(lat[1]);delta=lat[1][mask]-lat[0][mask]
            metrics.append({'spacing_mm':json.loads((root/'config.json').read_text())['spacing_cm']*10,
                'standard_far_ms':t[0],'nodal_far_ms':t[1],'far_delay_ms':t[1]-t[0],
                'far_delay_percent':(t[1]/t[0]-1)*100,'activation_relative_l2_percent':float(np.linalg.norm(delta)/np.linalg.norm(lat[0][mask])*100),
                'activation_mean_abs_ms':float(abs(delta).mean()*1000),
                'pde_trajectory_ms':{name:r['methods'][name]['pde_mean_ms'] for name in NAMES},
                'pde_replay_ms':{name:r['frozen_replay']['actual_ionic'][name]['pde_mean_ms'] for name in NAMES},
                'replay_speedup':r['frozen_replay']['actual_ionic'][NAMES[0]]['pde_mean_ms']/r['frozen_replay']['actual_ionic'][NAMES[1]]['pde_mean_ms']})
        X=sets[1][0]['X'];plane=np.ones(len(X),dtype=bool) if dim==2 else abs(X[:,2])<1e-12
        values=[z['activation_s'][plane]*1000 for z in sets[1]];lo=min(np.nanmin(v) for v in values);hi=max(np.nanmax(v) for v in values)
        for col,(z,label,color) in enumerate(zip(sets[1],LABELS,COLORS)):
            im=axes[row,col].tricontourf(X[plane,0],X[plane,1],values[col],levels=np.linspace(lo,hi,17),cmap='turbo')
            axes[row,col].set(xlabel='x (cm)',ylabel='y (cm)');axes[row,col].set_title(f'{dim}D {label}\nActivation (ms)'+(' at z=0' if dim==3 else ''),fontsize=11)
            axes[row,col].set_aspect('equal');fig.colorbar(im,ax=axes[row,col],shrink=.65)
            trace=z['traces_v'];axes[row,2].plot(np.arange(len(trace))*cfg['dt_ep_s']*1000,trace[:,1]*1000,label=label,color=color)
        axes[row,2].set(xlabel='Time (ms)',ylabel='Voltage (mV)',title=f'{dim}D far-corner action potential');axes[row,2].legend(fontsize=8);axes[row,2].grid(alpha=.2)
        for level,pair in enumerate(sets):
            for k,(z,color) in enumerate(zip(pair,COLORS)):
                Xk=z['X'];line=np.all(abs(Xk[:,1:])<1e-12,axis=1)
                ca[row].plot(Xk[line,0],z['activation_s'][line]*1000,color=color,ls='--' if level==0 else '-',label=f'{LABELS[k]}, {metrics[level]["spacing_mm"]:g} mm')
        ca[row].set(xlabel='x (cm)',ylabel='Activation time (ms)',title=f'{dim}D mesh refinement');ca[row].legend(fontsize=7);ca[row].grid(alpha=.2)
        # Same spatial locations for within-method coarse/fine convergence.
        shape=tuple(round(L/cfg['spacing_cm'])+1 for L in cfg['lengths_cm'][:dim])
        restriction=tuple(slice(None,None,2) for _ in range(dim))
        within={}
        for name,z0,z1 in zip(NAMES,sets[0],sets[1]):
            lat0=z0['activation_s'];lat1=z1['activation_s'].reshape(shape)[restriction].reshape(-1)
            valid=np.isfinite(lat0)&np.isfinite(lat1)
            within[name]={'far_change_percent_relative_fine':float((lat0[-1]/lat1[-1]-1)*100),
                          'activation_relative_l2_change_percent':float(np.linalg.norm(lat0[valid]-lat1[valid])/np.linalg.norm(lat1[valid])*100)}
        summary.append({'dimension':dim,'levels':metrics,'within_method_refinement':within,
                        'fine_far_delay_below_5percent':abs(metrics[-1]['far_delay_percent'])<5,
                        'fine_field_l2_difference_below_5percent':metrics[-1]['activation_relative_l2_percent']<5})
    fig.suptitle('0.125 mm EP mesh | same GPB, CN dt=0.05 ms and TorchCor CG kernel',fontsize=12)
    fig.savefig(fine/'refined_activation_and_waveforms.png',dpi=180);fig.savefig(fine/'refined_activation_and_waveforms.pdf');plt.close(fig)
    conv.savefig(fine/'activation_mesh_refinement.png',dpi=180);plt.close(conv)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    labels=['2D\n0.25 mm','2D\n0.125 mm','3D\n0.25 mm','3D\n0.125 mm'];x=np.arange(4)
    for k,(name,color,label) in enumerate(zip(NAMES,COLORS,LABELS)):
        vals=[r['pde_replay_ms'][name] for d in summary for r in d['levels']]
        bars=axes[0].bar(x+(k-.5)*.35,vals,.35,color=color,label=label);axes[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='PDE time per step (ms)',title='Identical frozen V / GPB current / stimulus');axes[0].legend(fontsize=8)
    delay=[r['far_delay_percent'] for d in summary for r in d['levels']]
    bars=axes[1].bar(x,delay,color=['#94a3b8','#0ea5e9']*2);axes[1].bar_label(bars,fmt='%.2f%%',padding=4)
    axes[1].axhline(5,color='#ef4444',ls='--',label='5% acceptance threshold')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Far-corner activation delay (%)',title='(Nodal time / standard FEM time - 1) x 100');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(fine/'refinement_performance_and_delay.png',dpi=180);plt.close(fig)
    (fine/'refinement_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
