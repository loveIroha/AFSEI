"""Decompose recorded accepted steps after an AFSEI diagnostic run."""
import argparse
import json
from pathlib import Path
import torch
from afsi_torch.ep.mac_snapshot_diagnostics import audit_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path,help='cycle output containing diagnostics/index.json')
    parser.add_argument('--tick',type=int,action='append',help='optional selected EP clock ticks')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--force',action='store_true',help='recompute an already verified snapshot analysis')
    args=parser.parse_args()
    torch.set_num_threads(4)
    index=json.loads((args.output/'diagnostics/index.json').read_text())
    result=[]
    for item in index['snapshots']:
        if args.tick and item['tick'] not in args.tick:continue
        directory=args.output/'diagnostics'/f"tick_{item['tick']:08d}"
        cached=directory/'pressure_diagnostics.json'
        reuse=(not args.force and cached.exists() and cached.stat().st_mtime>=max(
            (directory/'checkpoint.npz').stat().st_mtime,(directory/'history.npz').stat().st_mtime))
        report=json.loads(cached.read_text()) if reuse else audit_snapshot(directory,device=args.device)
        if not report['reconstruction_passed'] or report['tick']!=item['tick']:
            raise RuntimeError('snapshot analysis failed or refers to a different accepted step')
        result.append(report)
        print(json.dumps(dict(time_ms=report['time_ms'],core=report['core_pressure_mmhg'],
            boundary=report['boundary_multiplier_mmhg'],reconstruction=report['reconstruction'],cached=reuse),allow_nan=False),flush=True)
    if not result:raise SystemExit('no requested snapshots have been recorded')
    name='snapshot_analysis_selected.json' if args.tick else 'snapshot_analysis.json'
    (args.output/'diagnostics'/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
