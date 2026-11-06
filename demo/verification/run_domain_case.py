"""Fresh-rest domain-size study; same h, aligned lattice and wall geometry."""
from pathlib import Path
import argparse,json
from afsi_torch.simulation.circular_tube import run

p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);c=json.loads(a.config.read_text())
factor=c.get('solid_refinement_factor',1)
assert factor in (1,2) and c['fluid']['shape']==[64,64,64]
assert c['fluid']['lengths_cm']==[2.8,2.8,2.8] and c['fluid']['origin_cm']==[-.7,-.7,0.]
status=a.output/'status.json'
def save(**data):
    t=status.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(status)
try:
    prefix=json.loads(json.dumps(c));prefix['time']['end_time_s']=.06
    save(stage='running-prefix',end_s=.06)
    r=run(prefix,a.output/'prefix_60ms')
    assert r['status']=='completed' and r['ep_nodes']==3936 and r['solid_cells']==15360*factor**3
    save(stage='running-continuation',reached_s=.06,end_s=.3)
    r=run(c,a.output/'continued_300ms',resume=a.output/'prefix_60ms/final_state.npz')
    save(stage='completed',report=r)
except Exception as exc:save(stage='failed',error=str(exc));raise
