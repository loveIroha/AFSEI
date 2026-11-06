"""Run the explicit dual-tract LV case from a JSON configuration."""
import argparse,json
from pathlib import Path
from afsi_torch.simulation.lv_dual_port_cycle import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('configuration',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    print(json.dumps(run(json.loads(a.configuration.read_text()),a.output,device=a.device),indent=2))
