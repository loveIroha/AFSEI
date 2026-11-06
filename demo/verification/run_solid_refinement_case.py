"""Two-stage, fresh-rest solid-only refinement case; no replay of old physics."""
from pathlib import Path
import argparse,json
from afsi_torch.simulation.circular_tube import run

p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
c=json.loads(a.config.read_text())
assert c['solid_refinement_factor']==2
status=a.output/'status.json'
def save(**data):status.write_text(json.dumps(data,indent=2))
try:
    prefix=json.loads(json.dumps(c));prefix['time']['end_time_s']=.06
    save(stage='running-prefix',end_s=.06)
    r=run(prefix,a.output/'prefix_60ms')
    assert r['status']=='completed' and r['ep_nodes']==3936 and r['solid_cells']==122880
    save(stage='running-continuation',reached_s=.06,end_s=.3)
    r=run(c,a.output/'continued_300ms',resume=a.output/'prefix_60ms/final_state.npz')
    save(stage='completed',report=r)
except Exception as exc:
    save(stage='failed',error=str(exc));raise
