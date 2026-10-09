"""Validate idealized basal ports + RCR in a FIXED real LV cavity (no EP/FSI)."""
import argparse,csv,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from afsi_torch.mesh_io import read_solid_mesh
from afsi_torch.circulation import PortGeometry,from_lv_mesh,CavityFlowConfig,CavityFlowStepper
from afsi_torch.circulation import checkpoint


def write_vtk(folder,domain,state):
    """Export the actual staircase cavity, including its tagged port faces."""
    import meshio
    g=domain.grid; shape=np.array(g.shape); h=np.array(g.spacing); origin=np.array(g.origin)
    indices=np.indices(tuple(shape+1)).reshape(3,-1).T
    points=indices*h+origin
    points=points@domain.frame_basis.cpu().numpy()+domain.frame_center.cpu().numpy()
    ids=np.arange(len(points)).reshape(tuple(shape+1))
    cells=np.argwhere(domain.fluid.cpu().numpy()); i,j,k=cells.T
    hexa=np.stack((ids[i,j,k],ids[i+1,j,k],ids[i+1,j+1,k],ids[i,j+1,k],
        ids[i,j,k+1],ids[i+1,j,k+1],ids[i+1,j+1,k+1],ids[i,j+1,k+1]),1)
    pressure=state.pressure.cpu().numpy()[i,j,k]/1333.22368421
    velocity=[]
    for c,u in enumerate(state.velocity):
        a=[slice(None)]*3; b=a.copy(); a[c]=slice(None,-1); b[c]=slice(1,None)
        avg=.5*(u[tuple(a)]+u[tuple(b)])
        velocity.append(avg.cpu().numpy()[i,j,k])
    world_velocity=np.stack(velocity,1)@domain.frame_basis.cpu().numpy()
    meshio.write(folder/'cavity_final.vtu',meshio.Mesh(points,[('hexahedron',hexa)],
        cell_data={'pressure_mmhg':[pressure],'velocity_cm_s':[world_velocity]}))
    left,right=domain.adjacent(); quads=[]; labels=[]
    for c,(l,r,t) in enumerate(zip(left,right,domain.tags)):
        for face in np.argwhere((l^r).cpu().numpy()):
            other=[a for a in range(3) if a!=c]; vertices=[]
            for a,b in ((0,0),(1,0),(1,1),(0,1)):
                node=face.copy(); node[other[0]]+=a; node[other[1]]+=b; vertices.append(ids[tuple(node)])
            quads.append(vertices); labels.append(int(t[tuple(face)]))
    meshio.write(folder/'cavity_walls_ports.vtu',meshio.Mesh(points,[('quad',np.array(quads))],
        cell_data={'port_tag_0_wall_1_mv_2_av':[np.array(labels)]}))
    if 'ports_local_cm' in domain.metadata:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap,BoundaryNorm
        top=np.where(domain.fluid[:,:,-1].cpu().numpy(),domain.tags[2][:,:,-1].cpu().numpy(),-1)
        colors=ListedColormap(['white','#dedede','#268bd2','#dc5a4d'])
        fig,ax=plt.subplots(figsize=(7,6))
        ax.imshow(top.T,origin='lower',interpolation='nearest',cmap=colors,norm=BoundaryNorm([-.5-1,-.5,.5,1.5,2.5],4),
            extent=[origin[0],origin[0]+g.lengths[0],origin[1],origin[1]+g.lengths[1]])
        ports=domain.metadata['ports_local_cm']
        for label,color in (('mv','#268bd2'),('av','#dc5a4d')):
            center=ports[label+'_center']; radius=ports[label+'_radius']
            ax.add_patch(plt.Circle(center,radius,fill=False,color=color,linestyle='--',linewidth=1.5))
            ax.text(center[0],center[1],label.upper(),ha='center',va='center',weight='bold')
        ax.set(xlabel='Basal frame x (cm)',ylabel='Basal frame y (cm)',title='Idealized basal ports: resolved faces and nominal discs')
        ax.set_aspect('equal'); fig.tight_layout(); fig.savefig(folder/'idealized_ports.png',dpi=180); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__,epilog='On resume all geometry/physics parameters are restored from the checkpoint; fresh-run parameter options are ignored.')
    p.add_argument('--source-dir',default='/mnt/large2/gjh/realistic_left_ventricle')
    p.add_argument('--output',required=True); p.add_argument('--resume')
    p.add_argument('--steps',type=int,default=20,help='additional steps on resume')
    p.add_argument('--spacing-cm',type=float,default=.18); p.add_argument('--dt-ms',type=float,default=1.)
    p.add_argument('--atrium-mmhg',type=float,default=8.5); p.add_argument('--initial-pc-mmhg',type=float,default=8.)
    p.add_argument('--mv-center-cm',nargs=2,type=float,default=(.65,0.)); p.add_argument('--av-center-cm',nargs=2,type=float,default=(-.65,0.))
    p.add_argument('--mv-radius-cm',type=float,default=.55); p.add_argument('--av-radius-cm',type=float,default=.40)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda'); p.add_argument('--write-vtk',action='store_true')
    args=p.parse_args()
    if args.steps<1: p.error('steps must be positive')
    output=Path(args.output)
    if output.exists() and any(output.iterdir()): p.error('output must be new or empty; use a new folder on resume')
    output.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); torch.set_num_threads(4)
    if args.resume:
        driver,state,previous=checkpoint.load(args.resume,args.device)
    else:
        folder=Path(args.source_dir)
        mesh=read_solid_mesh(folder/'mesh_scale.xdmf',units='cm',boundaries=folder/'boundaries.xml',tag_map={2:1,1:2,3:3},device=args.device)
        ports=PortGeometry(tuple(args.mv_center_cm),tuple(args.av_center_cm),args.mv_radius_cm,args.av_radius_cm,args.spacing_cm)
        domain=from_lv_mesh(mesh,ports)
        driver=CavityFlowStepper(domain,CavityFlowConfig(dt_s=args.dt_ms/1000,atrium_mmhg=args.atrium_mmhg,initial_pc_mmhg=args.initial_pc_mmhg))
        state=driver.initial_state(); previous={}
    domain=driver.domain; start_step=state.step; history=[]
    checkpoint.save(output/'checkpoint.npz',driver,state)
    try:
        for _ in range(args.steps):
            state,diag=driver.step(state); history.append(diag)
            if state.step%5==0: print(f"Cavity step={state.step}, t={state.time_s*1000:.3f}ms, Qin={diag['q_in_ml_s']:.6g}, Qout={diag['q_out_ml_s']:.6g}ml/s, Pc={diag['pc_mmhg']:.6g}mmHg, energy_error={diag['total_energy_relative_error']:.3g}",flush=True)
    finally:
        checkpoint.save(output/'checkpoint.npz',driver,state,{'accepted_this_run':state.step-start_step})
        if history:
            with (output/'history.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=history[0]); writer.writeheader(); writer.writerows(history)
    if args.write_vtk: write_vtk(output,domain,state)
    report=dict(status='completed',scope='fixed real LV unsteady Stokes / idealized ports / flow-driven RCR validation subproblem',
        elapsed_seconds=time.perf_counter()-started,accepted_steps=state.step-start_step,total_steps=state.step,time_s=state.time_s,
        device=str(domain.fluid.device),fluid_shape=domain.grid.shape,fluid_cells=int(domain.fluid.sum()),config=asdict(driver.config),geometry=domain.metadata,
        resolved_port_area_cm2={label:sum(int((t==i).sum())*domain.grid.volume/h for t,h in zip(domain.tags,domain.grid.spacing)) for label,i in (('mv',1),('av',2))},
        final=history[-1],max_mass_error_ml_s=max(abs(d['mass_error_ml_s']) for d in history),
        max_divergence_s_inv=max(d['max_divergence_s_inv'] for d in history),max_wall_leak_cm_s=max(d['wall_leak_cm_s'] for d in history),
        max_total_energy_relative_error=max(d['total_energy_relative_error'] for d in history),
        max_fluid_energy_relative_error=max(d['fluid_energy_relative_error'] for d in history),
        fixed_cavity=True,fluid_ports_connected=True,windkessel_fluid_subproblem_connected=True,
        anatomical_valves_present=False,ep_advanced=False,myocardium_motion_advanced=False,preload_performed=False,
        windkessel_ports_connected_to_afsei_driver=False,three_field_integration_implemented=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()
