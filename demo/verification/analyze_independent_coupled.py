"""Plots and VTK export of completed paired fields; no solver advancement."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np,torch,meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.paper_coupling import PaperState
from afsi_torch.mac.output import MACWriter

ROOT=Path('/mnt/large2/gjh/AFSEI/results/independent_coupled_20261025')
torch.set_num_threads(2)

def main():
    report=json.loads((ROOT/'report.json').read_text());assert report['passed']
    c=json.loads((ROOT/'model.json').read_text());g,fsi,_,_=resolve_config(c)
    mesh=generate_circular_tube(g);X=mesh.X.numpy();cells=mesh.cells.numpy()
    runs={k:dict(np.load(ROOT/f'{k}_fields.npz')) for k in ['candidate','reference']}
    ode=dict(np.load(ROOT/'ode_fields.npz'));steps=report['accepted_steps']
    time=np.arange(steps+1)*fsi.time.dt;metrics={};histories={};errors={}
    for name,a in runs.items():
        points=a['x'][:,mesh.inner_rings.numpy(),:2];points=points-points.mean(2,keepdims=True)
        q=np.roll(points,-1,axis=2);area=.5*(points[...,0]*q[...,1]-points[...,1]*q[...,0]).sum(2)
        diameter=6*np.sqrt((area/area[0]).min(1))
        disp=np.linalg.norm(a['x']-X,axis=-1).max(1)*10
        energy=np.zeros(steps+1)
        for i in range(3):
            v=a[f'velocity_{i}'];w=np.ones(v.shape[1:]);key=[slice(None)]*3;key[i]=0;w[tuple(key)]=.5;key[i]=-1;w[tuple(key)]=.5
            energy+=.5*c['fluid']['density_g_per_cm3']*np.prod(np.array(c['fluid']['lengths_cm'])/c['fluid']['shape'])*(v*v*w).sum((1,2,3))
        inv=np.linalg.inv((X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1));grad=np.concatenate((-inv.sum(1,keepdims=True),inv),axis=1)
        refvol=abs(np.linalg.det((X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1)))/6
        F=np.einsum('teai,eaj->teij',a['x'][:,cells],grad);J=np.linalg.det(F);volume=(J*refvol).sum(1)
        histories[name]=dict(time_ms=(time*1000).tolist(),diameter_mm=diameter.tolist(),max_displacement_mm=disp.tolist(),kinetic_energy_erg=energy.tolist(),voltage_max_mv=(a['voltage'].max(1)*1000).tolist(),wall_volume_ml=volume.tolist())
        metrics[name]=dict(max_total_displacement_mm=float(disp.max()),minimum_equivalent_diameter_mm=float(diameter.min()),
            max_fluid_speed_component_cm_s=float(max(abs(a[f'velocity_{i}']).max() for i in range(3))),
            min_J=float(J.min()),max_J=float(J.max()),final_wall_volume_change_pct=float(100*(volume[-1]/volume[0]-1)))
    for field in ['displacement','velocity','pressure']:
        curve=[]
        for j in range(1,steps+1):
            if field=='displacement':a=runs['candidate']['x'][j]-X;b=runs['reference']['x'][j]-X
            elif field=='pressure':a=runs['candidate']['pressure'][j];b=runs['reference']['pressure'][j]
            else:
                a=np.concatenate([runs['candidate'][f'velocity_{k}'][j].ravel() for k in range(3)])
                b=np.concatenate([runs['reference'][f'velocity_{k}'][j].ravel() for k in range(3)])
            curve.append(float(np.linalg.norm((a-b).ravel())/max(np.linalg.norm(b.ravel()),1e-30)))
        errors[field]=curve
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for name,label,color,style in [('candidate','AFSEI GPU','#0072B2','-'),('reference','Independent EP/FSI','#D55E00','--')]:
        a=histories[name]
        for ax,key,ylabel in [(axs[0,0],'diameter_mm','Minimum equivalent diameter (mm)'),(axs[0,1],'kinetic_energy_erg','Eulerian kinetic energy (erg)'),(axs[1,0],'voltage_max_mv','Maximum voltage (mV)')]:
            ax.plot(a['time_ms'],a[key],style,color=color,lw=2 if name=='candidate' else 1.5,label=label)
            ax.set(xlabel='Elapsed verification time (ms)',ylabel=ylabel);ax.legend(fontsize=9)
    for field,v in errors.items():axs[1,1].semilogy(time[1:]*1000,np.maximum(v,1e-16),label=field)
    axs[1,1].set(xlabel='Elapsed verification time (ms)',ylabel='Relative discrete-field difference');axs[1,1].legend()
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.suptitle('40-step coupled trajectory: same discretization, independent EP/FSI implementation')
    fig.text(.5,.025,'288 P1 nodes, 768 tets, MAC 16x16x32; dt=50 us. Separate ODE states, shared GPB/Land CUDA kernel.\nNonzero electrical initial condition; this is a 2 ms verification, not a full-cycle or spatial-convergence claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95))
    for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    # Output uses the existing writer only; it contributes no numerical solve.
    solid=CircularTubeSolid(mesh,g,**{k:v for k,v in c['solid'].items() if k!='end_condition'},
        **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},dt_s=fsi.time.dt,end_support=c['solid']['end_condition'])
    grid=MACGrid(tuple(c['fluid']['shape']),tuple(c['fluid']['lengths_cm']),tuple(c['fluid']['origin_cm']))
    for name,a in runs.items():
        folder=ROOT/name/'vtk';writer=MACWriter(folder,solid,grid,pressure_description='physical open-box projection pressure; p=0 at all exterior faces')
        ep_frames=[]
        for j in range(0,steps+1,5):
            tensor=lambda v:torch.from_numpy(v.copy())
            state=PaperState(j,float(time[j]),tensor(a['x'][j]),tuple(tensor(a[f'velocity_{k}'][j]) for k in range(3)),
                tensor(a['pressure'][j]),tensor(a['force'][j]),float(time[j]),pressure_time=float(time[j]))
            writer.write(state);filename=f'ep_{j:06d}.vtu'
            meshio.write(folder/filename,meshio.Mesh(a['x'][j],[('tetra',cells)],point_data={
                'voltage_mV':a['voltage'][j]*1000,'calcium_uM':ode[f'{name}_gpb'][j,:,37]*1000,
                'land_XS':ode[f'{name}_land'][j,:,0],'land_input_stretch':np.ones(len(X))}),binary=True)
            ep_frames.append((float(time[j]),filename))
        root=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian');collection=ET.SubElement(root,'Collection')
        for t,filename in ep_frames:ET.SubElement(collection,'DataSet',timestep=repr(t),group='',part='0',file=filename)
        ET.ElementTree(root).write(folder/'ep.pvd',encoding='utf-8',xml_declaration=True)
    initial=dict(np.load(ROOT/'inputs.npz'));initial['gpb']=ode['candidate_gpb'][0];initial['absolute_ode_start_s']=report['contract']['ode_absolute_start_s']
    seed=np.load(report['contract']['seed']);select=np.arange(41*3*32).reshape(41,3,32)[::5,::2,::2].ravel()
    initial['zs']=seed['zs'][select];initial['zw']=seed['zw'][select]
    np.savez_compressed(ROOT/'initial_state_complete.npz',**initial)
    (ROOT/'analysis.json').write_text(json.dumps(dict(trajectory_metrics=metrics,histories=histories,errors=errors,
       vtk_scope='9 saved frames per field per path; PVD clock is elapsed verification time; ODE clock starts at 60 ms'),indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
