"""Plot completed 0.125 mm PDE measurements; never run a solver."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def case(report):
    if report['status'] != 'completed':
        raise ValueError('completed report required')
    selected = [row for row in report['cases'] if abs(row['spacing_mm']-.125)<1e-10]
    if len(selected) != 1:
        raise ValueError('exactly one 0.125 mm case required')
    return selected[0]


def row(value, label, color):
    return dict(label=label, color=color, source_method=value['method'],
                mean_ms=value['mean_pde_ms'], repeat_ms=value['repeat_pde_ms'],
                mean_iterations=value.get('mean_iterations', value.get('representative_mean_iterations')))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cn-report', type=Path, required=True)
    parser.add_argument('--full-report', type=Path, required=True)
    parser.add_argument('--two-level-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args(argv)
    paths=[args.cn_report,args.full_report,args.two_level_report]
    reports=[json.loads(path.read_text(encoding='utf-8')) for path in paths]
    cases=[case(report) for report in reports]
    signature=lambda report,c:(report['gpu'],report['dtype'],c['node_count'],c['cell_count'],
                              c['dt_ep_ms'],c['time_scheme'],report['settings']['snapshot_times_ms'],
                              report['settings']['rtol'],report['settings']['atol'])
    if any(signature(r,c)!=signature(reports[0],cases[0]) for r,c in zip(reports,cases)):
        raise ValueError('common benchmark conditions differ')
    old={v['method']:v for v in cases[0]['variants']}
    batches=[
        dict(title='Earlier controlled ablation',source_report=str(paths[0].resolve()),
             note='Common PDE wrappers; TorchCor CG kernel',methods=[
                 row(old['torchcor_p1_jacobi'],'Standard P1\nJacobi-PCG','#64748b'),
                 row(old['nodal_torchcor_jacobi'],'Nodal P1\nlight Jacobi-PCG','#38bdf8'),
                 row(old['afsei_nodal_amgx'],'Nodal P1\nAMG (SIZE_2)','#f97316')]),
        dict(title='Native pair: full AMG',source_report=str(paths[1].resolve()),
             note='Original electric statements vs AFSEI step',methods=[
                 row(cases[1]['methods'][0],'TorchCor native\nP1 + Jacobi-CG','#64748b'),
                 row(cases[1]['methods'][1],'AFSEI nodal P1\nfull AMG (SIZE_8)','#f97316')]),
        dict(title='Native pair: two-level AMG',source_report=str(paths[2].resolve()),
             note='Original electric statements vs AFSEI step',methods=[
                 row(cases[2]['methods'][0],'TorchCor native\nP1 + Jacobi-CG','#64748b'),
                 row(cases[2]['methods'][1],'AFSEI nodal P1\ntwo-level AMG','#f97316')])]
    args.output.mkdir(parents=True,exist_ok=True)
    summary=dict(status='completed',spacing_mm=.125,node_count=cases[0]['node_count'],
                 cell_count=cases[0]['cell_count'],batches=batches,
                 scope='PDE-only saved measurements; no new PDE or ODE solves',
                 limitation='Separate batches and baseline interfaces; speed ratios valid within each batch only')
    (args.output/'finest_comparison.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(args.output.resolve()/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5.2),layout='constrained',sharey=True)
    maximum=max(max(m['repeat_ms']) for b in batches for m in b['methods'])*1.2
    for ax,batch in zip(axes,batches):
        methods=batch['methods'];means=np.array([m['mean_ms'] for m in methods])
        errors=np.array([[m['mean_ms']-min(m['repeat_ms']),max(m['repeat_ms'])-m['mean_ms']]
                         for m in methods]).T
        bars=ax.bar(np.arange(len(methods)),means,color=[m['color'] for m in methods],
                    width=.62,yerr=errors,capsize=4)
        ax.bar_label(bars,fmt='%.3f ms',padding=5,fontsize=10)
        ax.set(xticks=np.arange(len(methods)),xticklabels=[m['label'] for m in methods],
               title=batch['title'],ylim=(0,maximum))
        ax.tick_params(axis='x',labelsize=9)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.text(.5,.97,batch['note'],transform=ax.transAxes,ha='center',va='top',fontsize=8)
    axes[0].set_ylabel('Warm PDE time per step (ms)')
    fig.suptitle('0.125 mm | 229,425 nodes | RTX 4090 | FP64 | CN dt = 0.05 ms',fontsize=14)
    fig.supxlabel('Separate experiment batches; compare speed ratios within each panel. Error bars: block min-max.',fontsize=10)
    fig.savefig(args.output/'finest_pde_comparison.png',dpi=200)
    fig.savefig(args.output/'finest_pde_comparison.pdf')
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output),solver_calls=0)))


if __name__=='__main__':
    main()
