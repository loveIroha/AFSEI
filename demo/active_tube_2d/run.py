"""Run a configured planar active-tube demonstration using AFSEI solvers."""
import argparse
import json
from pathlib import Path
from afsi_torch.simulation.active_tube import run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--end-ms',type=float)
    parser.add_argument('--passive',action='store_true',help='zero active load; retain the same EP and passive FSI')
    parser.add_argument('--no-stimulus',action='store_true',help='keep GPB/Land and mechanics, but inject no electrical stimulus')
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.end_ms is not None:
        config['time']['end_time_s']=args.end_ms/1000
    report=run(config,args.output,passive=args.passive,no_stimulus=args.no_stimulus)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    main()
