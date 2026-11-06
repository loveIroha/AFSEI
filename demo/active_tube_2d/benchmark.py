"""Same-model, same-discretization GPU implementation comparison (not external validation)."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np


def problem_contract(config):
    """Only execution backends may differ; physical AND numerical inputs stay fixed."""
    result=copy.deepcopy(config)
    result.pop('execution',None)
    for key in ('pressure_backend','mass_backend','ib_backend'):
        result['solver'].pop(key,None)
    return result


def compare_runs(reference, candidate, *, rtol=5e-5):
    reference,candidate=Path(reference),Path(candidate)
    configs=[json.loads((p/'config.resolved.json').read_text()) for p in (reference,candidate)]
    if problem_contract(configs[0]) != problem_contract(configs[1]):
        raise ValueError('model/mesh/time step/tolerances/output differ: timing comparison rejected')
    reports=[json.loads((p/'report.json').read_text()) for p in (reference,candidate)]
    if any(r['status']!='completed' for r in reports):
        raise ValueError('both trajectories must complete')
    if any(reports[0][key]!=reports[1][key] for key in ('accepted_steps','ep_steps','time_s','gpu')):
        raise ValueError('trajectory length or GPU differs')
    metrics={}
    def assess(name,a,b,atol):
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'invalid {name} field')
        error=float(np.linalg.norm((b-a).ravel()))
        scale=float(np.linalg.norm(a.ravel()))
        tolerance=atol*np.sqrt(a.size)+rtol*scale
        metrics[name]=dict(linf=float(np.max(np.abs(b-a))),l2_relative=error/max(scale,1e-30),
                           l2_error=error,l2_tolerance=float(tolerance),passed=bool(error<=tolerance))
    with np.load(reference/'snapshots.npz') as a, np.load(candidate/'snapshots.npz') as b:
        for key in ('reference','cells','lower','upper','times'):
            if not np.array_equal(a[key],b[key]):
                raise ValueError(f'{key} differs: no interpolation or time shifting permitted')
        # Displacement error is normalized by displacement, not large reference coordinates.
        assess('displacement_cm',a['x']-a['reference'],b['x']-b['reference'],1e-9)
        for name,atol in (('voltage',1e-6),('u',1e-8),('v',1e-8),('p',1e-6)):
            assess(name,a[name],b[name],atol)
    with np.load(reference/'final_state.npz') as a, np.load(candidate/'final_state.npz') as b:
        # Compare GPB variables separately: concentrations must not be hidden by voltage scale.
        for j in range(a['gpb'].shape[1]):
            assess(f'gpb_{j}',a['gpb'][:,j],b['gpb'][:,j],1e-12)
        for name in ('land','zs','zw','force'):
            assess(name,a[name],b[name],1e-9)
    histories=[json.loads((p/'history.json').read_text()) for p in (reference,candidate)]
    if [r['time_s'] for r in histories[0]] != [r['time_s'] for r in histories[1]]:
        raise ValueError('diagnostic times differ')
    for name in ('minimum_gap_cm','mid_gap_cm','lumen_mid_flow_cm2_s'):
        assess(name,*[np.array([r[name] for r in h]) for h in histories],1e-9)
    encoded=json.dumps(problem_contract(configs[0]),sort_keys=True,separators=(',',':')).encode()
    passed=all(v['passed'] for v in metrics.values())
    return dict(passed=passed,problem_sha256=hashlib.sha256(encoded).hexdigest(),metrics=metrics,
                external_literature_validation=False,
                speedup=(reports[0]['warmed_compute_seconds']/reports[1]['warmed_compute_seconds']) if passed else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--end-ms',type=float)
    args=parser.parse_args()
    if args.repeats<1:
        parser.error('repeats must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    base=json.loads(args.config.read_text(encoding='utf-8'))
    if args.end_ms is not None:
        base['time']['end_time_s']=args.end_ms/1000
    if base['time']['end_time_s']<=base['stimulus']['start_s']+base['stimulus']['duration_s']:
        parser.error('comparison must include post-stimulus coupled dynamics')
    base['output']['write_vtk']=False  # Field snapshots still saved, outside timed blocks.
    root=Path(__file__).resolve().parents[1]
    pairs=[]
    timings={'optimized':[],'reference':[]}
    for repeat in range(args.repeats):
        order=('optimized','reference') if repeat%2==0 else ('reference','optimized')
        for profile in order:
            config=copy.deepcopy(base)
            config['execution']={'profile':profile}
            if profile=='optimized':
                config['solver'].update(pressure_backend='graph',mass_backend='graph',ib_backend='cuda')
            else:
                config['solver'].update(pressure_backend='reference',mass_backend='pcg',ib_backend='reference')
            config_path=(args.output/f'{profile}_{repeat}.json').resolve()
            config_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
            run_path=(args.output/f'{profile}_{repeat}').resolve()
            with (args.output/f'{profile}_{repeat}.log').open('w') as log:
                subprocess.run([sys.executable,str(root/'run.py'),str(config_path),'--output',str(run_path)],
                               stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
            report=json.loads((run_path/'report.json').read_text())
            timings[profile].append(report['warmed_compute_seconds']/report['warmed_compute_steps'])
            print(f'{profile} repeat={repeat}: {timings[profile][-1]:.6g} s/step',flush=True)
        comparison=compare_runs(args.output/f'reference_{repeat}',args.output/f'optimized_{repeat}')
        pairs.append(comparison)
        (args.output/f'comparison_{repeat}.json').write_text(json.dumps(comparison,indent=2)+'\n')
        if not comparison['passed']:
            raise RuntimeError('field comparison failed; no performance claim is accepted')
    summary=dict(scope='AFSEI GPU implementation optimization only; not versus an external FEM code',
        passed=True,repeats=args.repeats,seconds_per_step=timings,
        median_speedup=statistics.median(timings['reference'])/statistics.median(timings['optimized']),
        problem_sha256=pairs[0]['problem_sha256'],external_literature_validation=False,
        note='First 10 physical steps excluded; late compilation, if any, remains charged. '
             'Use repeated runs to expose cache/order effects. I/O excluded equally; setup reported separately.')
    (args.output/'benchmark.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
