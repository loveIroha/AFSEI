"""Compare existing fixed-tissue outputs without repeating any solve."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from ..cycle_checkpoint import atomic_json


def _physical_signature(config):
    data=json.loads(json.dumps(config))
    for key in ('demo','case_name','reference','output'):
        data.pop(key,None)
    data['domain'].pop('spacing_cm',None)
    return data


def compare_runs(paths, output, *, write_plot=True):
    folders=[Path(p) for p in paths]
    if len(folders)<2:
        raise ValueError('at least two existing mesh runs required')
    configs=[json.loads((p/'config_resolved.json').read_text()) for p in folders]
    if any(_physical_signature(c)!=_physical_signature(configs[0]) for c in configs[1:]):
        raise ValueError('compare the same equations, stimulus, boundaries, time grid and observations')
    spacings=[c['domain']['spacing_cm']*10 for c in configs]
    if len(set(spacings))!=len(spacings):
        raise ValueError('mesh spacings must be distinct')
    reports=[json.loads((p/'report.json').read_text()) for p in folders]
    if any(r['status']!='completed' for r in reports):
        raise ValueError('only completed runs may be compared')
    curves=[np.loadtxt(p/'activation_line.csv',delimiter=',',skiprows=1,ndmin=2) for p in folders]
    axis=configs[0]['observations']['line_axis']
    transverse=[a for a in range(3) if a!=axis]
    if any(not np.allclose(curve[:,transverse],curves[0][0,transverse],rtol=0,atol=1e-12) for curve in curves):
        raise ValueError('resolved observation lines must coincide across meshes')
    folder=Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('comparison output directory must be new or empty')
    folder.mkdir(parents=True,exist_ok=True)
    finest=int(np.argmin(spacings)); reference=curves[finest]
    cases=[]
    for p,h,r,curve in zip(folders,spacings,reports,curves):
        # Interpolate the finer numerical solution only where all line nodes activated.
        # This is a mesh comparison, not an error estimate against an exact solution.
        delta=None
        if np.isfinite(reference[:,-1]).all() and np.isfinite(curve[:,-1]).all():
            values=np.interp(curve[:,axis],reference[:,axis],reference[:,-1])
            delta=float(np.max(np.abs(curve[:,-1]-values)))
        cases.append(dict(path=str(p),spacing_mm=h,node_count=r['node_count'],cell_count=r['cell_count'],
            activated_fraction=r['activated_fraction'],last_activation_ms=r['last_activation_ms'],
            line_endpoint_activation_ms=float(curve[-1,-1]) if np.isfinite(curve[-1,-1]) else None,
            max_line_difference_from_finest_ms=delta,integration_seconds=r['integration_seconds'],
            stimulus_nodal_quadrature_volume_cm3=r['stimulus_nodal_quadrature_volume_cm3']))
    result=dict(status='completed',finest_spacing_mm=spacings[finest],
        spatial_convergence_established=False,strict_paper_model_reproduction=False,cases=cases)
    atomic_json(folder/'report.json',result)
    with (folder/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        columns=[k for k in cases[0] if k!='path']
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
        writer.writeheader(); writer.writerows(cases)
    if write_plot:
        os.environ.setdefault('MPLCONFIGDIR',str(folder.resolve()/'.matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
        for h,curve in zip(spacings,curves):
            ax.plot(curve[:,axis],curve[:,-1],label=f'{h:g} mm')
        ax.set(xlabel=f'{"xyz"[axis]} (cm)',ylabel='Activation time (ms)',
               title='GPB fixed slab: spatial resolution comparison')
        ax.grid(alpha=.25); ax.legend()
        fig.savefig(folder/'mesh_comparison.png',dpi=180); plt.close(fig)
    return result


def regenerate_plots(folder):
    """Use saved nodal fields and traces; no mesh assembly or ODE/PDE solve."""
    from ..config import load_config
    from ..ep.experiment import EPTissueConfig
    from ..geometry.tissue_slab import TissueSlabMesh
    from .ep_tissue import observation_indices,_plots
    import torch
    folder=Path(folder)
    config=load_config(folder/'config_resolved.json',EPTissueConfig)
    with np.load(folder/'mesh.npz',allow_pickle=False) as saved:
        X=saved['coordinates_cm']; cells=saved['cells']; shape=tuple(int(n) for n in saved['shape'])
    mesh=TissueSlabMesh(torch.from_numpy(X),torch.from_numpy(cells),shape,
                       tuple(L/n for L,n in zip(config.domain.lengths_cm,shape)))
    probes,line,slice_index=observation_indices(mesh,config.domain,config.observations)
    with np.load(folder/'final_state.npz',allow_pickle=False) as final:
        activation=final['activation_time_s']
    with np.load(folder/'voltage_snapshots.npz',allow_pickle=False) as saved:
        snapshot_steps=np.rint(saved['time_s']/config.time.dt_ep_s).astype(int)
        snapshots=saved['voltage_v']
    trace=np.loadtxt(folder/'probes.csv',delimiter=',',skiprows=1,ndmin=2)
    n=len(probes)
    os.environ.setdefault('MPLCONFIGDIR',str(folder.resolve()/'.matplotlib'))
    _plots(folder,X,tuple(n+1 for n in shape),config.observations,slice_index,line,activation,
           trace[:,0]/1000,trace[:,1:n+1]/1000,trace[:,n+1:]/1000,probes,
           snapshot_steps,snapshots,config.time.dt_ep_s)
