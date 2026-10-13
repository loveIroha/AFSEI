"""Santiago Figure 6 geometry with AFSEI GPB; fixed tissue, nodal FEM + AMG."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from afsi_torch.config import load_config
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.ep import MonodomainParameters, MonodomainNumerics
from afsi_torch.ep.experiment import (EPTissueConfig,TissueDomainConfig,TissueStimulus,
    EPExperimentTime,TissueBoundary,TissueInitial,TissueObservations,TissueOutput,TissueReference)
from afsi_torch.simulation.ep_tissue import run


# CASE DEFINITION: cm, s, V, microA/cm2, microS/cm, microF/cm2.
# Published Table 1 mS/cm -> microS/cm: multiply by 1000 exactly once.
# GPB retains AFSEI's pinned initialization and existing CUDA ODE integrator.
CONFIG=EPTissueConfig(
    case_name='santiago2018_fig6_gpb_fixed_slab',
    domain=TissueDomainConfig(lengths_cm=(2.,.7,.3),origin_cm=(0.,0.,0.),spacing_cm=.025,
                             fiber=(1.,0.,0.),sheet=(0.,1.,0.)),
    parameters=MonodomainParameters(capacitance_uf_per_cm2=1.,surface_to_volume_per_cm=.1,
                                    conductivity_us_per_cm=(1.17,.39,.39)),
    ep_numerics=MonodomainNumerics('nodal_p1','amgx_pcg_amg'),
    time=EPExperimentTime(dt_ep_s=5e-5,dt_ode_ms=.002,end_time_s=.08),
    stimulus=TissueStimulus(minimum_cm=(0.,0.,0.),maximum_cm=(.15,.15,.15),
        amplitude_ua_per_cm2=50000./1400.,start_s=0.,duration_s=.002),
    boundary=TissueBoundary(),initial=TissueInitial(),
    observations=TissueObservations(activation_threshold_mv=0.,slice_axis=2,slice_coordinate_cm=0.,
        line_axis=0,line_anchor_cm=(0.,0.,0.),
        probes_cm=((0.,0.,0.),(.5,0.,0.),(1.,0.,0.),(1.5,0.,0.),(2.,0.,0.),(2.,.7,.3))),
    output=TissueOutput(log_every=100,trace_every=10,vtk_every=200,write_vtk=True,write_plots=True),
    reference=TissueReference(doi='10.1002/cnm.3140',figure='6; section 3.1; Table 1',
        comparison_scope='Santiago slab geometry and bulk parameters; AFSEI GPB instead of O Hara Rudy',
        parameter_source='Table 1: conductivity (0.00117,0.00039,0.00039) mS/cm; Sv=0.1/cm; Cm=1 microF/cm2',
        stimulus_source='Niederer 2011 DOI 10.1098/rsta.2011.0139: 1.5mm cube, 2ms; '
            'membrane equivalent 50000/1400 microA/cm2. Retain this membrane current, '
            'not its volumetric 50000 microA/cm3 when using Santiago Sv=0.1/cm.'),
)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,help='standard JSON experiment configuration')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--mesh-spacing-mm',type=float,help='structured brick spacing, in mm')
    parser.add_argument('--duration-ms',type=float,help='fixed experiment end time, in ms')
    parser.add_argument('--ep-backend',choices=('amgx_pcg_amg','torch_pcg_jacobi'))
    parser.add_argument('--vtk',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--plots',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--write-config',type=Path,help='write resolved template JSON and exit')
    args=parser.parse_args(argv)
    config=load_config(args.config,CONFIG) if args.config else CONFIG
    changes={}
    if args.mesh_spacing_mm is not None:
        changes['domain']=replace(config.domain,spacing_cm=args.mesh_spacing_mm/10)
    if args.duration_ms is not None:
        changes['time']=replace(config.time,end_time_s=args.duration_ms/1000)
    if args.ep_backend is not None:
        changes['ep_numerics']=replace(config.ep_numerics,backend=args.ep_backend)
    flags={name:value for name,value in (('write_vtk',args.vtk),('write_plots',args.plots)) if value is not None}
    if flags: changes['output']=replace(config.output,**flags)
    config=replace(config,**changes)
    if args.write_config:
        atomic_json(args.write_config,dict(demo='ep-slab',**asdict(config)))
        return config
    if args.output is None:
        from datetime import datetime
        args.output=Path('results')/f'ep_slab_{datetime.now():%Y%m%d_%H%M%S_%f}'
    print(f'Output directory: {args.output}',flush=True)
    report=run(config,args.output,device=args.device)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    main()
