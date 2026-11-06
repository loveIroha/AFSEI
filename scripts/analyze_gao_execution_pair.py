"""Compare two same-checkpoint short runs, including their complete saved fields."""
import json
from pathlib import Path
import sys
import numpy as np

root=Path(sys.argv[1]);cases={};arrays={}
other=sys.argv[2] if len(sys.argv)>2 else 'pressure_compact_step'
destination=sys.argv[3] if len(sys.argv)>3 else 'execution_pair_summary.json'
for name in ('pressure_reference_step',other):
    path=root/name;r=json.loads((path/'report.json').read_text());rows=r['records']
    with np.load(path/'checkpoint.npz',allow_pickle=False) as a:
        meta=json.loads(str(a['metadata']))
        arrays[name]={k:a[k] for k in ('x','pressure','force','voltage_v','gpb','land','stretch','zs','zw',
                                      'velocity_0','velocity_1','velocity_2')}
    with np.load(path/'initial_checkpoint.npz',allow_pickle=False) as a:
        initial=json.loads(str(a['metadata']))
    cases[name]=dict(status=r['status'],steps=r['accepted_steps'],elapsed_seconds=r['elapsed_seconds'],
        mean_step_seconds=float(np.mean([x['step_seconds'] for x in rows])),
        mean_after_first_seconds=float(np.mean([x['step_seconds'] for x in rows[1:]])),
        mean_fluid_solves=float(np.mean([x['nonlinear']['fluid_solves'] for x in rows])),
        mean_newton_iterations=float(np.mean([x['nonlinear']['iterations'] for x in rows])),
        total_preconditioner_setup_seconds=sum(x['nonlinear']['preconditioner_setup_seconds'] for x in rows),
        new_rejected_attempts=meta['rejected_attempts']-initial['rejected_attempts'],
        compact_geometry=meta['cut_boundary']['compact_geometry'],
        tract_geometry_gpu=meta['cut_boundary'].get('tract_geometry_gpu',False),
        max_mass_error_ml_s=r['max_mass_error_ml_s'],
        max_cumulative_mass_error_ml=r['max_cumulative_mass_error_ml'],
        advanced_restart_roundtrip=r['advanced_restart_roundtrip'])
names=list(arrays)
diff={k:float(np.max(np.abs(arrays[names[0]][k]-arrays[names[1]][k]))) for k in arrays[names[0]]}
old,new=cases.values()
summary=dict(scope='three accepted 0.05ms intervals per variant on same GPU; not a long performance benchmark',
    cases=cases,max_abs_field_differences=diff,
    elapsed_speedup=old['elapsed_seconds']/new['elapsed_seconds'],
    accepted_step_speedup=old['mean_step_seconds']/new['mean_step_seconds'],
    after_first_step_speedup=old['mean_after_first_seconds']/new['mean_after_first_seconds'])
(root/destination).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
