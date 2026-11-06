from pathlib import Path
import sys,json,hashlib
R=Path('/mnt/large2/gjh/AFSEI');sys.path.insert(0,str(R/'src'))
from afsi_torch.simulation.active_tube import validate_config
base=R/'demo/active_tube_2d';out=R/'results/tube_cg_euler_wave_20261031';out.mkdir(exist_ok=False)
c=json.loads((base/'configs/tube_plane_strain_contraction_gpu.json').read_text())
c['ep']['numerics'].update(discretization='consistent_p1',diffusion_quadrature='consistent_p1',time_scheme='crank_nicolson')
c['ep']['numerics']['solver'].update(check_every=5,recompute_every=40)
c['ep']['ode_method']='explicit_euler'
# Same physical conductivity/Cm/chi as the recent standard EP benchmark.
common=json.loads((R/'results/ep_opencarp_same_mesh_20261030/common_configuration.json').read_text())
c['ep']['parameters']=common['parameters']
c['time'].update(fsi_dt_s=1e-5,ep_dt_s=2e-5,ode_dt_ms=.001,end_time_s=.6,stop_on_no_contact_limit=True)
c['active']['tension_scale']=.02
c['output'].update(log_every=200,field_every=200,write_vtk=True,wave_sections=True)
validate_config(c)
(base/'configs/tube_cg_euler_wave_gpu.json').write_text(json.dumps(c,indent=2))
(out/'configuration.json').write_text(json.dumps(c,indent=2))
short=json.loads(json.dumps(c));short['time']['end_time_s']=.02
(base/'configs/tube_cg_euler_wave_check_gpu.json').write_text(json.dumps(short,indent=2))
print(json.dumps(dict(output=str(out),config=str(base/'configs/tube_cg_euler_wave_gpu.json'),planned_duration_ms=600,GPB_model='Original tube GPB57, unchanged parameters and initial state; not the distinct native Grandi54 benchmark adapter',force='Paired upper/lower inward Land traction, scale .02',FSI='Existing 2D explicit-lagged MAC+MG+IB+P2 Neo-Hookean unchanged'),indent=2))
