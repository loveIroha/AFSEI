"""Compare saved standard/nodal P1 timings with the same TorchCor CG kernel."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    source=json.loads(args.report.read_text(encoding='utf-8'))
    if (source['status']!='completed' or source['settings']['time_scheme']!='crank_nicolson'
            or source['settings']['demo']!='ep-pde-methods-benchmark'):
        raise ValueError('completed shared-kernel CN benchmark required')
    cases=sorted(source['cases'],key=lambda c:-c['spacing_mm'])
    if len(cases)!=3 or any(abs(c['spacing_mm']-h)>1e-10 for c,h in zip(cases,(.5,.25,.125))):
        raise ValueError('exactly the three requested mesh levels required')
    rows=[]
    for c in cases:
        methods={v['method']:v for v in c['variants']}
        pair=[methods[name] for name in ('torchcor_p1_jacobi','nodal_torchcor_jacobi')]
        if any(v['max_true_residual_to_tolerance']>1 for v in pair):
            raise ValueError('common true residual validation failed')
        speedup=pair[0]['mean_pde_ms']/pair[1]['mean_pde_ms']
        rows.append(dict(spacing_mm=c['spacing_mm'],node_count=c['node_count'],cell_count=c['cell_count'],
                         dt_ep_ms=c['dt_ep_ms'],methods=pair,speedup=speedup,
                         time_reduction_percent=100*(1-1/speedup)))
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(status='completed',source_report=str(args.report.resolve()),
                source_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                gpu=source['gpu'],dtype=source['dtype'],settings=source['settings'],torchcor=source['torchcor'],
                cases=rows,solver_calls=0,
                scope='Same TorchCor Jacobi-PCG kernel and shared PDE wrapper; different P1 mass quadrature',
                timing=source['timing_scope'],
                limitation='Saved same-batch CN replay; not the independent AFSEI PCG or native electric-statement pair')
    (args.output/'same_solver_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(args.output.resolve()/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    positions=np.arange(len(rows))
    labels=[f'{c["spacing_mm"]:g} mm\n{c["node_count"]:,} nodes' for c in rows]
    for k,(label,color) in enumerate((('TorchCor standard P1-Galerkin + Jacobi-PCG','#64748b'),
                                     ('Nodal P1 + same Jacobi-PCG kernel','#38bdf8'))):
        means=[c['methods'][k]['mean_pde_ms'] for c in rows]
        errors=np.array([[m-min(c['methods'][k]['repeat_pde_ms']),max(c['methods'][k]['repeat_pde_ms'])-m]
                         for m,c in zip(means,rows)]).T
        bars=axes[0].bar(positions+(k-.5)*.35,means,.35,color=color,label=label,yerr=errors,capsize=4)
        axes[0].bar_label(bars,fmt='%.3f ms',padding=5,fontsize=9)
    axes[0].set(ylabel='Warm PDE time per step (ms)',title='Same solver: standard P1 vs nodal P1',
                xticks=positions,xticklabels=labels)
    axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.22)
    axes[0].legend(fontsize=8,loc='upper right')
    ratios=[c['speedup'] for c in rows]
    bars=axes[1].bar(positions,ratios,color='#38bdf8',width=.6)
    axes[1].bar_label(bars,fmt='%.3fx',padding=5,fontsize=10)
    axes[1].axhline(1,color='#64748b',ls='--')
    axes[1].set(ylabel='Standard P1 time / nodal P1 time',title='Measured speedup with shared CG kernel',
                xticks=positions,xticklabels=labels,ylim=(0,max(ratios)*1.22))
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('RTX 4090 | FP64 | CN dt = 0.05 ms | same true residual criterion',fontsize=14)
    fig.supxlabel('Same TorchCor CG kernel and PDE wrapper | excludes ODE and assembly | error bars: block min-max',fontsize=9)
    fig.savefig(args.output/'same_solver_pde_comparison.png',dpi=200)
    fig.savefig(args.output/'same_solver_pde_comparison.pdf')
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output),solver_calls=0,
                         cases=[dict(h=r['spacing_mm'],speedup=r['speedup'],time_reduction_percent=r['time_reduction_percent']) for r in rows])))


if __name__=='__main__':main()
