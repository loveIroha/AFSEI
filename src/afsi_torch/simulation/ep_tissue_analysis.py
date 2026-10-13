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
           snapshot_steps,snapshots,config.time.dt_ep_s,model_name=config.cell_model)


def sample_slab_p1(X,shape,points,field):
    """Interpolate a P1 nodal scalar in the exact Freudenthal tetrahedral mesh."""
    shape=np.asarray(shape,dtype=int)
    lo=X.min(0); hi=X.max(0); spacing=(hi-lo)/shape
    logical=(np.asarray(points)-lo)/spacing
    if ((logical< -1e-10)|(logical>shape+1e-10)).any():
        raise ValueError('interpolation points must be inside the slab')
    base=np.minimum(np.maximum(np.floor(logical).astype(int),0),shape-1)
    fractions=np.clip(logical-base,0,1)
    order=np.argsort(-fractions,axis=1,kind='stable')
    sorted_fractions=np.take_along_axis(fractions,order,axis=1)
    bary=np.column_stack((1-sorted_fractions[:,0],sorted_fractions[:,0]-sorted_fractions[:,1],
                          sorted_fractions[:,1]-sorted_fractions[:,2],sorted_fractions[:,2]))
    nodes=np.tile(base[:,None,:],(1,4,1))
    index=np.arange(len(points))
    nodes[index,1,order[:,0]]+=1
    nodes[:,2]=nodes[:,1]
    nodes[index,2,order[:,1]]+=1
    nodes[:,3]=base+1
    ids=(nodes[:,:,0]*(shape[1]+1)+nodes[:,:,1])*(shape[2]+1)+nodes[:,:,2]
    values=field[ids]
    # Zero-weight unactivated vertices must not contaminate exact-node samples.
    return (np.where((bary>1e-14)|np.isfinite(values),values,0)*bary).sum(1)


def _cell_signature(config):
    data=json.loads(json.dumps(config))
    data['domain'].setdefault('axis_spacing_cm',None)
    for key in ('demo','case_name','reference','output','cell_model'):
        data.pop(key,None)
    return data


def compare_cell_models(paths,output,*,write_plot=True):
    """Strict model-only comparison; keep mesh spacing and all external settings."""
    folders=[Path(p) for p in paths]
    configs=[json.loads((p/'config_resolved.json').read_text()) for p in folders]
    if len(folders)!=2 or {c['cell_model'] for c in configs}!={'gpb','ord2011_epi'}:
        raise ValueError('compare source epicardial GPB against original epicardial ORd 2011')
    if _cell_signature(configs[0])!=_cell_signature(configs[1]):
        raise ValueError('cell comparison requires exactly equal external conditions, including mesh and time steps')
    reports=[json.loads((p/'report.json').read_text()) for p in folders]
    if any(r['status']!='completed' for r in reports): raise ValueError('completed results required')
    meshes=[]; states=[]; curves=[]
    for folder,c in zip(folders,configs):
        with np.load(folder/'mesh.npz',allow_pickle=False) as m:
            mesh={k:m[k] for k in ('coordinates_cm','cells','shape','stimulus_region')}
        meshes.append(mesh)
        with np.load(folder/'final_state.npz',allow_pickle=False) as saved:
            activation=saved['activation_time_s']
        states.append(activation)
        origin=np.asarray(c['domain']['origin_cm']); lengths=np.asarray(c['domain']['lengths_cm'])
        points=origin+np.linspace(0,1,401)[:,None]*lengths
        curve=sample_slab_p1(mesh['coordinates_cm'],mesh['shape'],points,activation)
        distance=np.linalg.norm(points-origin,axis=1)
        table=np.column_stack((distance,points,curve*1000))
        np.savetxt(folder/'activation_diagonal.csv',table,delimiter=',',
                   header='distance_cm,x_cm,y_cm,z_cm,activation_ms',comments='')
        curves.append(table)
    for key in meshes[0]:
        if not np.array_equal(meshes[0][key],meshes[1][key]):
            raise ValueError('resolved meshes and stimulus masks must match exactly')
    folder=Path(output)
    if folder.exists() and any(folder.iterdir()): raise ValueError('choose a new comparison directory')
    folder.mkdir(parents=True,exist_ok=True)
    summaries={}
    for cfg,r,curve in zip(configs,reports,curves):
        start,end=curve[0,-1],curve[-1,-1]
        initialization=r.get('initialization')
        if initialization is None:
            # Earlier same-protocol outputs retain the conditioned state in row zero.
            trace=np.loadtxt(folders[configs.index(cfg)]/'probes.csv',delimiter=',',skiprows=1,ndmin=2)
            n=len(cfg['observations']['probes_cm'])
            initialization=dict(policy=cfg['initial']['state'],conditioning_s=cfg['initial']['conditioning_s'],
                conditioned_voltage_mv=float(trace[0,1]),conditioned_calcium_um=float(trace[0,n+1]),
                values_read_from_initial_probe_row=True)
        summaries[cfg['cell_model']]=dict(path=str(folders[configs.index(cfg)]),
            activation_start_ms=float(start) if np.isfinite(start) else None,
            activation_diagonal_end_ms=float(end) if np.isfinite(end) else None,
            propagation_duration_ms=float(end-start) if np.isfinite(start+end) else None,
            last_activation_ms=r['last_activation_ms'],activated_fraction=r['activated_fraction'],
            initialization=initialization,mean_seconds_per_ep_step=r['mean_seconds_per_ep_step'])
    gpb,ord=summaries['gpb'],summaries['ord2011_epi']
    duration_ratio=None
    if gpb['propagation_duration_ms'] and ord['propagation_duration_ms']:
        duration_ratio=ord['propagation_duration_ms']/gpb['propagation_duration_ms']
    result=dict(status='completed',only_cell_model_differs=True,
        external_conditions=_cell_signature(configs[0]),
        external_conditions_sha256=__import__('hashlib').sha256(json.dumps(_cell_signature(configs[0]),sort_keys=True).encode()).hexdigest(),
        models=summaries,ord_to_gpb_propagation_duration_ratio=duration_ratio,
        strict_santiago_reproduction=False,
        diagonal_observation='P1 interpolation of nodal first-upward-crossing times along the body diagonal',
        observation_endpoints_cm=[configs[0]['domain']['origin_cm'],
            (np.array(configs[0]['domain']['origin_cm'])+configs[0]['domain']['lengths_cm']).tolist()])
    atomic_json(folder/'report.json',result)
    if write_plot:
        os.environ.setdefault('MPLCONFIGDIR',str(folder.resolve()/'.matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        upper=max(float(np.nanmax(a))*1000 for a in states if np.isfinite(a).any())
        for axis,mesh,activation,cfg in zip(axes,meshes,states,configs):
            shape=tuple(int(n)+1 for n in mesh['shape'])
            z=np.ma.masked_invalid(activation.reshape(shape)[:,:,0].T*1000)
            lengths=cfg['domain']['lengths_cm']; origin=cfg['domain']['origin_cm']
            image=axis.imshow(z,origin='lower',extent=(origin[0],origin[0]+lengths[0],origin[1],origin[1]+lengths[1]),
                              aspect='equal',cmap='turbo',vmin=0,vmax=upper)
            axis.set(title=cfg['cell_model'],xlabel='x (cm)',ylabel='y (cm)')
        fig.colorbar(image,ax=list(axes[:2]),label='Activation time (ms)',shrink=.8)
        for curve,cfg in zip(curves,configs):
            axes[2].plot(curve[:,0],curve[:,-1],label=cfg['cell_model'])
        axes[2].set(xlabel='Distance along body diagonal (cm)',ylabel='Activation time (ms)',
                    title='Equal external conditions')
        axes[2].grid(alpha=.25); axes[2].legend()
        fig.savefig(folder/'model_comparison.png',dpi=180);plt.close(fig)
    return result


def render_cell_comparison_3d(paths,output,times_ms=(10.,20.,40.)):
    """Render saved full 3D surface voltage; does not rerun the solvers."""
    if len(paths)!=2 or not times_ms:
        raise ValueError('two model cases and at least one snapshot required')
    os.environ.setdefault('MPLCONFIGDIR',str(Path(output).resolve().parent/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,6.5),layout='constrained')
    axes=[]; norm=Normalize(-90,50); cmap=plt.get_cmap('turbo')
    for row,path in enumerate(paths):
        folder=Path(path)
        config=json.loads((folder/'config_resolved.json').read_text())
        with np.load(folder/'mesh.npz',allow_pickle=False) as m:
            X=m['coordinates_cm'];shape=tuple(int(n)+1 for n in m['shape'])
        grid=np.arange(len(X)).reshape(shape)
        triangles=[]
        for axis in range(3):
            for side in (0,shape[axis]-1):
                face=np.take(grid,side,axis=axis)
                a=face[:-1,:-1].ravel();b=face[1:,:-1].ravel();c=face[:-1,1:].ravel();d=face[1:,1:].ravel()
                triangles.extend((np.column_stack((a,b,d)),np.column_stack((a,d,c))))
        faces=np.concatenate(triangles)
        with np.load(folder/'voltage_snapshots.npz',allow_pickle=False) as snapshots:
            time=snapshots['time_s']*1000;voltage=snapshots['voltage_v']*1000
        for column,t in enumerate(times_ms):
            index=int(np.argmin(np.abs(time-t)))
            axis=fig.add_subplot(2,len(times_ms),row*len(times_ms)+column+1,projection='3d')
            colors=cmap(norm(voltage[index,faces].mean(1)))
            axis.add_collection3d(Poly3DCollection(X[faces],facecolors=colors,edgecolors='none',rasterized=True))
            lo=X.min(0);hi=X.max(0)
            axis.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),
                     xlabel='x (cm)',ylabel='y (cm)',zlabel='z (cm)',
                     xticks=np.linspace(lo[0],hi[0],3),yticks=np.linspace(lo[1],hi[1],3),
                     zticks=np.linspace(lo[2],hi[2],3),
                     title=f'{config["cell_model"]}: {time[index]:g} ms')
            axis.tick_params(labelsize=8,pad=1)
            axis.set_box_aspect(hi-lo);axis.view_init(elev=25,azim=-60);axes.append(axis)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes,label='Voltage (mV)',shrink=.7)
    fig.savefig(output,dpi=180);plt.close(fig)


def render_planar_wave_comparison(folder):
    """Plot measured planar LAT and CV from already completed controls."""
    folder=Path(folder)
    report=json.loads((folder/'report.json').read_text())
    if report['status']!='completed': raise ValueError('completed planar study required')
    os.environ.setdefault('MPLCONFIGDIR',str(folder.resolve()/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    styles=('-', '--', ':', '-.')
    for model,color in (('gpb','tab:blue'),('ord2011_epi','tab:orange')):
        rows=sorted((c for c in report['cases'] if c['cell_model']==model),key=lambda c:c['spacing_mm'])
        for i,row in enumerate(rows):
            h=row['spacing_mm']
            curve=np.loadtxt(folder/f'planar_h{h:g}_{model}'/'activation_line.csv',delimiter=',',skiprows=1)
            axes[0].plot(curve[:,0],curve[:,-1],color=color,ls=styles[i%len(styles)],label=f'{model}, {h:g} mm')
        axes[1].plot([r['spacing_mm'] for r in rows],
                     [np.nan if r['cv_cm_s'] is None else r['cv_cm_s'] for r in rows],
                     color=color,marker='o',label=model)
    axes[0].set(xlabel='x (cm)',ylabel='Activation time (ms)',title='Planar wave: equal external conditions')
    axes[1].set(xlabel='Longitudinal mesh spacing (mm)',ylabel='Conduction velocity (cm/s)',
                title='CV between x = 0.5 and 1.5 cm')
    for ax in axes: ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(folder/'planar_wave_comparison.png',dpi=180);plt.close(fig)
