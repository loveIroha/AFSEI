import json
import sys
from pathlib import Path

base=Path(sys.argv[1] if len(sys.argv)>1 else 'results/gao_connection_pair_20261026')
for name in ('legacy','physical_cut'):
    path=base/name/'report.json'
    if not path.exists(): path=base/name/'progress.json'
    if not path.exists():
        print(json.dumps(dict(case=name,status='initializing')))
        continue
    try:
        r=json.loads(path.read_text())
    except json.JSONDecodeError:
        print(json.dumps(dict(case=name,status='progress being written')))
        continue
    row=r['records'][-1] if r['records'] else {}
    b=row.get('balanced',{})
    print(json.dumps(dict(case=name,status=r.get('status','running'),
        steps=r['accepted_steps'],time_ms=row.get('time_ms'),
        pressure_core_mmhg=b.get('pressure_core',{}).get('mean_mmhg'),
        pressure_boundary_mmhg=b.get('boundary_pressure_mmhg'),
        volume_ml=b.get('volume_ml'),step_seconds=row.get('step_seconds'),
        hydrostatic_max_speed_cm_s=r.get('hydrostatic_probe',{}).get('max_fluid_speed_cm_s'),
        rim_mean_axial_change_cm=row.get('rim_mean_axial_change_cm'),
        failure=r.get('failure'))))
