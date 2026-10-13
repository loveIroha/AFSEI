"""Regenerate tissue figures from saved outputs without repeating the solve."""
import argparse
from afsi_torch.simulation.ep_tissue_analysis import regenerate_plots


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',help='completed slab result directory')
    args=parser.parse_args(argv)
    regenerate_plots(args.result)
    print(f'Figures regenerated: {args.result}',flush=True)


if __name__=='__main__':
    main()
