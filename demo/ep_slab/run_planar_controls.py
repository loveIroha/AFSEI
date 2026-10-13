"""High-resolution planar-wave controls using the same P1/AmgX/cell solvers."""
import argparse,json,csv
from dataclasses import asdict,replace
from pathlib import Path
from afsi_torch.config import load_config
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.ep.experiment import EPTissueConfig
from afsi_torch.simulation.ep_tissue import run
from afsi_torch.simulation.ep_tissue_analysis import compare_cell_models,render_planar_wave_comparison


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spacing-mm',nargs='+',type=float,default=[.0625,.03125])
    parser.add_argument('--resume-study',action='store_true')
    args=parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume_study:
        parser.error('choose a fresh output or resume complete matching cases')
    args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).parent/'configs'
    rows=[]
    for spacing in args.spacing_mm:
        paths=[]
        for model in ('gpb','ord2011_epi'):
            base=load_config(root/f'planar_{model}.json',EPTissueConfig)
            config=replace(base,domain=replace(base.domain,spacing_cm=spacing/10,
                           axis_spacing_cm=(spacing/10,.1,.1)))
            destination=args.output/f'planar_h{spacing:g}_{model}'
            if (destination/'report.json').exists():
                actual=load_config(destination/'config_resolved.json',EPTissueConfig)
                if actual!=config: raise ValueError('cached planar case has different conditions')
                report=json.loads((destination/'report.json').read_text())
                if report['status']!='completed': raise ValueError('cached planar case is incomplete')
                print(f'Reuse completed case: {destination}',flush=True)
            else:
                print(f'Run: {destination}',flush=True)
                report=run(config,destination)
            probes=report['probes']; t0=probes[1]['activation_ms']; t1=probes[3]['activation_ms']
            velocity=1000/(t1-t0) if t0 is not None and t1 is not None and t1>t0 else None
            rows.append(dict(spacing_mm=spacing,cell_model=model,cv_cm_s=velocity,
                              t_x05_ms=t0,t_x15_ms=t1,activated_fraction=report['activated_fraction']))
            paths.append(destination)
        folder=args.output/f'planar_h{spacing:g}_comparison'
        if not (folder/'report.json').exists(): compare_cell_models(paths,folder)
        atomic_json(args.output/'progress.json',dict(status='running',cases=rows))
    atomic_json(args.output/'report.json',dict(status='completed',cases=rows,
        cv_definition='1cm / activation delay between x=0.5cm and x=1.5cm',
        scope='planar-wave verification; separate from the 3D corner-stimulated Santiago slab'))
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    render_planar_wave_comparison(args.output)
    print(json.dumps(dict(status='completed',cases=rows),indent=2),flush=True)


if __name__=='__main__': main()
