"""Run one prespecified full tube trajectory with a reusable short prefix.

The prefix is part of this same from-rest initial value problem. No different
mesh checkpoint is interpolated or used as a substitute for a full trajectory.
"""
from pathlib import Path
import argparse,hashlib,json,os,time
import torch
from afsi_torch.simulation.circular_tube import run

def write(path,value):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2));p.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();c=json.loads(args.config.read_text());out=args.output
    if out.exists() and any(out.iterdir()):raise ValueError('new empty case output required')
    out.mkdir(parents=True);torch.set_num_threads(2)
    write(out/'config.json',c)
    original={k:c[k] for k in ('solid','active','ep','stimulus','fsi')}
    write(out/'contract.json',dict(from_rest=True,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        ep_nodes=3936,ep_mesh_refined=False,solid_factor=c['solid_refinement_factor'],
        mac_shape=c['fluid']['shape'],time_dt_s=c['time']['fsi_dt_s'],end_s=c['time']['end_time_s'],
        fixed_model_sha256=hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest(),
        source_sha256={str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in [
            Path('src/afsi_torch/solids/refined_circular_tube.py'),Path('src/afsi_torch/simulation/circular_tube.py')]}))
    try:
        prefix=json.loads(json.dumps(c));prefix['time']['end_time_s']=.001
        prefix['output'].update(log_every=10,field_every=20,checkpoint_every=20)
        write(out/'status.json',dict(stage='initial-prefix',end_ms=1))
        r=run(prefix,out/'prefix_1ms')
        if r['status']!='completed':raise RuntimeError('prefix did not complete')
        write(out/'prefix_resources.json',dict(max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_memory_reserved_bytes=torch.cuda.max_memory_reserved(),report=r))
        if r['ep_nodes']!=3936 or r['solid_cells']!=15360*c['solid_refinement_factor']**3:
            raise RuntimeError('mesh-size contract failed')
        torch.cuda.empty_cache()
        write(out/'status.json',dict(stage='full-continuation',reached_ms=1,end_ms=300))
        r=run(c,out/'continued_300ms',resume=out/'prefix_1ms/final_state.npz')
        if r['status']!='completed' or abs(r['time_s']-.3)>1e-10:raise RuntimeError('full trajectory incomplete')
        write(out/'status.json',dict(stage='completed',report=r))
    except Exception as e:
        write(out/'status.json',dict(stage='failed',error=repr(e),settings_relaxed=False));raise

if __name__=='__main__':main()
