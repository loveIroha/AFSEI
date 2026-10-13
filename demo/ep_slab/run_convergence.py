"""Same experiment on multiple meshes; reuse the case entry point and EP solver."""
import argparse
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from afsi_torch.config import load_config
from afsi_torch.ep.experiment import EPTissueConfig
from afsi_torch.simulation.ep_tissue import run
from afsi_torch.simulation.ep_tissue_analysis import compare_runs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/santiago_gpb.json')
    parser.add_argument('--spacing-mm',type=float,nargs='+',default=[.5,.25,.125])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--vtk',action=argparse.BooleanOptionalAction,default=False)
    args=parser.parse_args(argv)
    if len(set(args.spacing_mm))!=len(args.spacing_mm) or any(h<=0 or not np.isfinite(h) for h in args.spacing_mm):
        parser.error('mesh spacings must be positive, finite and distinct')
    config=load_config(args.config,EPTissueConfig)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('choose a new or empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    destinations=[]
    for spacing in args.spacing_mm:
        case=replace(config,domain=replace(config.domain,spacing_cm=spacing/10),
                     output=replace(config.output,write_vtk=args.vtk))
        destination=args.output/f'h_{spacing:g}mm'
        run(case,destination,device=args.device)
        destinations.append(destination)
    compare_runs(destinations,args.output/'comparison')
    print(json.dumps(dict(status='completed',output=str(args.output)),indent=2))


if __name__=='__main__':
    main()
