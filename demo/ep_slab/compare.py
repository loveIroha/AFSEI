"""Plot and compare existing slab results; no GPU computation is repeated."""
import argparse
import json
from afsi_torch.simulation.ep_tissue_analysis import compare_runs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',help='completed slab result directories')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    report=compare_runs(args.runs,args.output)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    main()
