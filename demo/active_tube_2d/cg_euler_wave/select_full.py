from pathlib import Path
import json,sys
R=Path('/mnt/large2/gjh/AFSEI');sys.path.insert(0,str(R/'src'))
from afsi_torch.simulation.active_tube import validate_config
O=R/'results/tube_cg_euler_wave_20261031';p=R/'demo/active_tube_2d/configs/tube_cg_euler_wave_gpu.json'
pilot=json.loads((O/'pilot/report.json').read_text());assert pilot['status']=='completed' and pilot['time_s']==.1
c=json.loads(p.read_text());assert c['active']['tension_scale']==.02
c['active']['tension_scale']=.002;validate_config(c)
p.write_text(json.dumps(c,indent=2));(O/'configuration.json').write_text(json.dumps(c,indent=2))
(O/'load_selection.json').write_text(json.dumps(dict(pilot_scale=.02,full_scale=.002,pilot_duration_ms=100,
    pilot_minimum_gap_cm=pilot['final']['minimum_gap_cm'],full_planned_duration_ms=600,
    reason='Pilot reached 73.3% local closure at100ms. Use a smaller explicit actuator scale for a longer noncontact wave demonstration; no prescribed moving-load waveform or displacement.',
    unchanged='Geometry, Neo-Hookean material, supports, fluid MAC/IB, RC boundary, GPB/Land equations, spatial support and stimulus'),indent=2))
print('Selected full-run active scale .002; all other physical fields retained')
