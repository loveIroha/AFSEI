"""Run an AFSEI cycle job from one explicit JSON configuration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration',type=Path)
    parser.add_argument('--duration-ms',type=float,help='validation override only')
    parser.add_argument('--output',type=Path,help='validation override only')
    args=parser.parse_args()
    job=json.loads(args.configuration.read_text(encoding='utf-8'))
    allowed={'physical_gpu','preload','resume','reference_checkpoint','cycles','checkpoint_every',
        'ep_config','ep_refinement_factor','ep_refinement_cache','ode_step_ms','spatial_config',
        'circulation_config','port_config','diagnostics_config','output','reuse_step_tangent','fsi_config',
        'ode_recovery_config','period_ms','filling_hold_ms'}
    if set(job)-allowed:raise ValueError('unknown job configuration keys')
    if job['physical_gpu'] not in (0,1):raise ValueError('physical GPU must be 0 or 1')
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(job['physical_gpu'])
    command=[sys.executable,'-u',str(root/'scripts/run_afsei_cycle.py')]
    for key,value in job.items():
        if key=='physical_gpu' or value is None:continue
        if key=='output' and args.output:value=str(args.output)
        if key=='reuse_step_tangent':
            command.append('--reuse-step-tangent' if value else '--no-reuse-step-tangent');continue
        command.extend(('--'+key.replace('_','-'),str(value)))
    if args.duration_ms is not None:
        command.extend(('--duration-ms',str(args.duration_ms)))
        command=[v for v in command]
    # Validation uses the same equations/config but only in-range snapshots.
    if args.duration_ms is not None and '--diagnostics-config' in command:
        i=command.index('--diagnostics-config');del command[i:i+2]
    print(json.dumps(dict(configuration=str(args.configuration.resolve()),command=command,
        physical_gpu=job['physical_gpu']),ensure_ascii=False),flush=True)
    raise SystemExit(subprocess.call(command,cwd=root,env=env))


if __name__=='__main__':main()
