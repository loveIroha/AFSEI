"""Compare an existing GPB/ORd pair with exactly equal external conditions."""
import argparse,json
from afsi_torch.simulation.ep_tissue_analysis import compare_cell_models


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs=2)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    report=compare_cell_models(args.runs,args.output)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__': main()
