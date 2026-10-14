"""Controlled GPB/ORd study; declared JSON, reuse completed cases on resume."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import numpy as np
from afsi_torch.config import load_config,_decode
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.ep.experiment import EPTissueConfig
from afsi_torch.simulation.ep_tissue import run
from afsi_torch.simulation.ep_tissue_analysis import compare_cell_models


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/model_comparison.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume-study',action='store_true',help='reuse only complete matching cases')
    args=parser.parse_args(argv)
    study=json.loads(args.config.read_text())
    allowed={'demo','schema_version','base_config','models','spacing_mm','reference_diffusion_cm2_s',
             'literal_table_check','time_refinement','write_vtk_spacing_mm'}
    if set(study)-allowed or study.get('demo')!='ep-slab-model-comparison' or study.get('schema_version')!=1:
        parser.error('invalid declared study schema')
    if study['models']!=['gpb','ord2011_epi']: parser.error('study compares GPB and original epicardial ORd')
    base=load_config(args.config.parent/study['base_config'],EPTissueConfig)
    if base.initial.state!='model_native_rest': parser.error('use common native initialization policy')
    derived=np.array(base.parameters.conductivity_us_per_cm)/(
        base.parameters.surface_to_volume_per_cm*base.parameters.capacitance_uf_per_cm2)
    if not np.allclose(derived,study['reference_diffusion_cm2_s'],rtol=1e-14,atol=0):
        parser.error('base tissue parameters must encode the declared reference diffusion')
    output=args.output
    if output.exists() and any(output.iterdir()) and not args.resume_study:
        parser.error('choose a new output or explicitly resume the study')
    output.mkdir(parents=True,exist_ok=True)
    saved=output/'study_resolved.json'
    resolved=dict(study=study,base=asdict(base))
    if saved.exists():
        cached=json.loads(saved.read_text())
        cached['base']['ep_numerics'].setdefault('time_scheme','backward_euler')
        # Fill new optional defaults when resuming results from older schemas.
        cached['base']=asdict(_decode(EPTissueConfig,cached['base']))
        if json.loads(json.dumps(cached))!=json.loads(json.dumps(resolved)):
            parser.error('resume cannot change the declared study')
    atomic_json(saved,resolved)
    protocols=[]
    for spacing in study['spacing_mm']:
        protocols.append((f'reference_h{spacing:g}',spacing,study['reference_diffusion_cm2_s'],base.time))
    literal=study['literal_table_check']
    if literal['enabled']:
        protocols.append(('literal_table',literal['spacing_mm'],literal['diffusion_cm2_s'],base.time))
    refined=study['time_refinement']
    if refined['enabled']:
        protocols.append(('time_refined',refined['spacing_mm'],study['reference_diffusion_cm2_s'],
                          replace(base.time,dt_ep_s=refined['dt_ep_s'],dt_ode_ms=refined['dt_ode_ms'])))
    comparisons=[]
    for name,spacing,diffusion,times in protocols:
        paths=[]
        for model in study['models']:
            config=replace(base,cell_model=model,domain=replace(base.domain,spacing_cm=spacing/10),time=times,
                parameters=base.parameters if diffusion==study['reference_diffusion_cm2_s'] else
                    replace(base.parameters,conductivity_us_per_cm=tuple(
                    d*base.parameters.surface_to_volume_per_cm*base.parameters.capacitance_uf_per_cm2 for d in diffusion)),
                output=replace(base.output,write_vtk=spacing==study['write_vtk_spacing_mm']))
            destination=output/f'{name}_{model}'
            if (destination/'report.json').exists():
                report=json.loads((destination/'report.json').read_text())
                data=json.loads((destination/'config_resolved.json').read_text());data.pop('demo',None)
                data['ep_numerics'].setdefault('time_scheme','backward_euler')
                actual=json.loads(json.dumps(asdict(_decode(EPTissueConfig,data))))
                if report['status']!='completed' or actual!=json.loads(json.dumps(asdict(config))):
                    raise ValueError(f'existing case does not match: {destination}')
                print(f'Reuse completed case: {destination}',flush=True)
            else:
                print(f'Run: {destination}',flush=True)
                run(config,destination,device=args.device)
            paths.append(destination)
        comparison=output/f'{name}_comparison'
        if (comparison/'report.json').exists():
            report=json.loads((comparison/'report.json').read_text())
        else:
            report=compare_cell_models(paths,comparison)
        comparisons.append(dict(protocol=name,report=report))
        atomic_json(output/'study_progress.json',dict(status='running',completed_protocols=comparisons))
    atomic_json(output/'report.json',dict(status='completed',protocols=comparisons,
                                         strict_santiago_reproduction=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for model,axis in zip(study['models'],axes):
        for name,h,_,_ in protocols:
            if not name.startswith('reference_'): continue
            curve=np.loadtxt(output/f'{name}_{model}'/'activation_diagonal.csv',delimiter=',',skiprows=1)
            axis.plot(curve[:,0],curve[:,-1],label=f'{h:g} mm')
        axis.set(title=model,xlabel='Distance along body diagonal (cm)',ylabel='Activation time (ms)')
        axis.grid(alpha=.25);axis.legend()
    fig.savefig(output/'model_mesh_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(status='completed',output=str(output)),indent=2),flush=True)


if __name__=='__main__': main()
