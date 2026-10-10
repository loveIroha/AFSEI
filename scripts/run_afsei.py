"""Run/restart EP-GPB-Land with the original implicit MAC/IB fluid-solid solver."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from afsi_torch.config import TimeConfig,load_config
from afsi_torch.paper_lv import PaperLVConfig
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.simulation.afsei import run


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fsi-config',help='PaperLV JSON; default: configs/afsei_mac.json')
    p.add_argument('--source-dir')
    p.add_argument('--output')
    p.add_argument('--resume')
    p.add_argument('--duration-ms',type=float,default=2.,help='additional duration, on the 0.05ms EP grid')
    p.add_argument('--fluid-shape',type=int,help='override MAC grid size; default preset is 128 cubed')
    p.add_argument('--mechanical-dt-ms',type=float)
    p.add_argument('--nonlinear-solver',choices=('jfnk','newton','anderson-newton'))
    p.add_argument('--profile',action='store_true',help='synchronize GPU for per-stage timing; includes cold starts and retries')
    p.add_argument('--start-ms',type=float,default=500.)
    p.add_argument('--mechanics',choices=('source','inherited'),default='source',
                   help='fresh runs: source EP-project parameters or inherited AFSI parameters; resume restores saved parameters')
    args = p.parse_args(argv)
    if args.resume and any(value is not None for value in
            (args.fsi_config,args.source_dir,args.fluid_shape,args.mechanical_dt_ms,args.nonlinear_solver)):
        p.error('resume restores the saved MAC solver configuration; fresh-run overrides cannot be used')
    return args


def fresh_config(args):
    path = args.fsi_config or Path(__file__).resolve().parents[1]/'configs/afsei_mac.json'
    fsi = load_config(path,PaperLVConfig)
    dt = fsi.time.dt if args.mechanical_dt_ms is None else args.mechanical_dt_ms/1000
    fsi = replace(fsi,time=TimeConfig(dt,dt))
    if args.source_dir is not None: fsi = replace(fsi,source_dir=args.source_dir)
    if args.fluid_shape is not None: fsi = replace(fsi,fluid=replace(fsi.fluid,shape=(args.fluid_shape,)*3))
    if args.nonlinear_solver is not None: fsi = replace(fsi,nonlinear_solver=args.nonlinear_solver)
    factory = AFSEIConfig.source if args.mechanics == 'source' else AFSEIConfig
    return factory(fsi=fsi,start_time_s=args.start_ms/1000)


def main():
    args = parse_args()
    config = None if args.resume else fresh_config(args)
    report = run(config=config,output=args.output,resume=args.resume,duration_s=args.duration_ms/1000,
                 profile=args.profile)
    print(json.dumps({k:report[k] for k in ('status','accepted_steps','reached_time_s','ep_steps',
        'ode_substeps','rejected_attempts','elapsed_seconds','fluid_shape','direction_mapping',
        'fluid_discretization','fsi_solver','performance','mechanics_profile',
        'source_mechanics_parameters_aligned','ep_fsi_time_integration_implemented',
        'preload_performed','prescribed_endocardial_pressure_mmhg',
        'windkessel_ports_connected','three_field_integration_implemented')},indent=2))


if __name__ == '__main__': main()
