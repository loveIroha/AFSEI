"""Run/restart the short AFSEI EP + fluid/solid integration experiment."""
import argparse
import json
from afsi_torch.config import FluidConfig,TimeConfig,OutputConfig
from afsi_torch.paper_lv import PaperLVConfig
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.simulation.afsei import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',default='/mnt/large2/gjh/realistic_left_ventricle')
    p.add_argument('--output')
    p.add_argument('--resume')
    p.add_argument('--duration-ms',type=float,default=2.,help='additional duration, on the 0.05ms EP grid')
    p.add_argument('--fluid-shape',type=int,default=32,help='coarse-grid integration check; not a resolution study')
    p.add_argument('--mechanical-dt-ms',type=float,default=.1)
    p.add_argument('--start-ms',type=float,default=500.)
    p.add_argument('--mechanics',choices=('source','inherited'),default='source',
                   help='fresh runs: source EP-project parameters or inherited AFSI parameters; resume restores saved parameters')
    args = p.parse_args()
    config = None
    if not args.resume:
        dt = args.mechanical_dt_ms/1000
        fsi = PaperLVConfig(source_dir=args.source_dir,time=TimeConfig(dt,dt),
            fluid=FluidConfig((args.fluid_shape,)*3,(13.,)*3,rho=1.,mu=.04),nonlinear_solver='newton',
            output=OutputConfig(log_every=5,checkpoint_every=20,output_every=20,write_vtk=False))
        factory = AFSEIConfig.source if args.mechanics == 'source' else AFSEIConfig
        config = factory(fsi=fsi,start_time_s=args.start_ms/1000)
    report = run(config=config,output=args.output,resume=args.resume,duration_s=args.duration_ms/1000)
    print(json.dumps({k:report[k] for k in ('status','accepted_steps','reached_time_s','ep_steps',
        'ode_substeps','rejected_attempts','elapsed_seconds','fluid_shape','direction_mapping',
        'mechanics_profile','source_mechanics_parameters_aligned','ep_fsi_time_integration_implemented',
        'windkessel_ports_connected','three_field_integration_implemented')},indent=2))


if __name__ == '__main__': main()
