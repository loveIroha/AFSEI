"""Standard-config tube windows; call production EP/ODE/FSI solvers.

Mechanical reference geometry and zero fluid velocity are NEW initial data.
Electrical fields are read from a named checkpoint. This is not a mechanical
restart of that checkpoint and not a complete activation/recovery cycle.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,time,hashlib,os,xml.etree.ElementTree as ET
import numpy as np
import torch
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube,section_areas
from afsi_torch.solids.refined_circular_tube import refine_circular_tube
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.ep.monodomain import Monodomain
from afsi_torch.ep.tissue import TissueState,GPBLand
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.mac.output import MACWriter
from afsi_torch.mac.grid import divergence


def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();cfg=json.loads(args.config.read_text());c=cfg['model'];out=args.output
    if out.exists() and any(out.iterdir()):raise ValueError('choose a new empty output directory')
    out.mkdir(parents=True,exist_ok=True);save_json(out/'config.resolved.json',cfg)
    torch.set_num_threads(2);setup_start=time.perf_counter();g,fsi,par,num=resolve_config(c)
    factor=c.get('solid_refinement_factor',1)
    coarse=generate_circular_tube(g,device='cuda:0');mesh=coarse;field_transfer=None
    if factor==2:mesh,field_transfer=refine_circular_tube(coarse,g)
    seed=Path(cfg['electrical_initial_state']);a=np.load(seed)
    saved_mechanics=cfg.get('initial_mechanics','reference-zero-fluid')=='saved-state-stress-switch'
    if saved_mechanics:
        if not (np.array_equal(a['reference'],mesh.X.cpu().numpy()) and np.array_equal(a['cells'],mesh.cells.cpu().numpy())):
            raise ValueError('saved mechanical mesh mismatch')
    elif not (np.array_equal(a['reference'],coarse.X.cpu().numpy()) and np.array_equal(a['cells'],coarse.cells.cpu().numpy())):
        raise ValueError('electrical seed must match the unchanged original EP mesh exactly')
    absolute_start=float(a['time_s'])
    if abs(absolute_start-cfg['absolute_ode_start_s'])>1e-12:raise ValueError('absolute ODE clock mismatch')
    get=lambda k:torch.as_tensor(a[k].copy(),device=mesh.X.device,dtype=mesh.X.dtype)
    tissue=TissueState(get('voltage_v'),get('gpb'),get('land'))
    zs,zw,rest=get('zs'),get('zw'),get('resting_tension_dyn_cm2')
    if c['active']['stretch_feedback']!='isometric-driver' or float(zs.abs().max())>1e-12 or float(zw.abs().max())>1e-12:
        raise ValueError('this isolated verification requires the staged isometric branch and zero distortion history')
    one=torch.ones_like(zs);zero=torch.zeros_like(one)
    solid=CircularTubeSolid(mesh,g,**{k:v for k,v in c['solid'].items() if k!='end_condition'},
        **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},
        dt_s=fsi.time.dt,end_support=c['solid']['end_condition'])
    solid.field_transfer=field_transfer;solid.set_reference(rest);solid.stage(mesh.X,tissue,zs,zw)
    driver=BEIBStepper(solid,fsi,mesh.X.device);state=driver.initialize(mesh.X)
    if saved_mechanics:
        solid.stage(get('x'),tissue,zs,zw)
        state=replace(state,x=get('x'),previous_x=get('previous_x'),
            velocity=tuple(get(k) for k in ('u','v','w')),pressure=get('pressure'),
            force=solid.force(get('x'),0.),force_time=0.,pressure_time=0.)
    radial=coarse.X[:,:2]-coarse.X.new_tensor(g.center_xy_cm);radial/=radial.norm(dim=1,keepdim=True)
    axial=coarse.X.new_tensor([0.,0.,1.]).expand_as(coarse.X)
    sheet=torch.cat((radial,torch.zeros_like(radial[:,:1])),1)
    ep=Monodomain(coarse.X,coarse.cells,axial,sheet,parameters=par,numerics=num);ep.prepare(fsi.time.dt)
    ode=GPBLand(fsi.time.dt,c['time']['ode_dt_ms'])
    writer=MACWriter(out/'vtk',solid,driver.grid,pressure_description='MAC Eulerian Lagrange multiplier; outer box p=0; do not interpret wall pressure without extra stress')
    frames=[];history=[];blocks=[];completed=0;steps=round(fsi.time.end_time/fsi.time.dt)
    log_every=c['output']['log_every'];warmup=cfg['timing_warmup_steps']
    if steps%log_every or warmup!=log_every:raise ValueError('timing uses complete fixed blocks; first block excluded')
    reference_volume=float(solid.volumes.sum())
    array=lambda v:v.detach().cpu().numpy()
    np.savez_compressed(out/'initial_state.npz',reference=array(mesh.X),cells=array(mesh.cells),
        ep_reference=array(coarse.X),ep_cells=array(coarse.cells),voltage_v=array(tissue.voltage_v),
        gpb=array(tissue.gpb),land=array(tissue.land),zs=array(zs),zw=array(zw),resting_tension=array(rest),
        x=array(state.x),previous_x=array(state.previous_x) if state.previous_x is not None else array(state.x),
        u=array(state.velocity[0]),v=array(state.velocity[1]),w=array(state.velocity[2]),pressure=array(state.pressure),
        absolute_ode_start_s=absolute_start)
    contract=dict(initial_mechanics=('saved accepted x, previous_x, MAC velocity and pressure; recomputed force after explicit stress switch'
                                    if saved_mechanics else 'reference geometry; zero MAC velocity and pressure storage'),
        initial_electrical_state='complete saved EP/GPB/Land state; no new stimulus',
        absolute_ode_start_s=absolute_start,electrical_seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
        local_window_s=fsi.time.end_time,stress_formulation_switched_at_start=saved_mechanics,
        original_single_formulation_trajectory=False,
        solid_nodes=len(mesh.X),solid_cells=len(mesh.cells),ep_nodes=len(coarse.X),
        fluid_shape=list(driver.grid.shape),fluid_spacing_cm=driver.grid.spacing,
        dt_s=fsi.time.dt,physical_gpu=os.environ.get('AFSEI_GPU'),gpu=torch.cuda.get_device_name(),
        timing_scope='compute blocks excluding diagnostics/VTK; discard first block; distinct stress discretizations, not a matched-accuracy speedup')
    save_json(out/'contract.json',contract)

    def record(info=None):
        J=torch.linalg.det(solid.element_gradient(state.x));d=solid.diagnostics(state.x)
        area=section_areas(state.x,mesh);centers=state.x[mesh.inner_rings].mean(1)
        n=(info or {}).get('nonlinear',{})
        row=dict(time_ms=state.time*1000,absolute_ode_time_ms=(absolute_start+state.time)*1000,step=state.step,
            wall_volume_change_percent=100*(d['wall_volume_cm3']/reference_volume-1),
            J_rms_difference_from_one=float(((J-1).square()*solid.volumes).sum().div(solid.volumes.sum()).sqrt()),
            minimum_detF=d['minimum_detF'],maximum_detF=d['maximum_detF'],
            minimum_projected_diameter_mm=float(20*(area.min()/torch.pi).sqrt()),
            maximum_centerline_transverse_displacement_mm=float((centers[:,:2]-centers.new_tensor(g.center_xy_cm)).norm(dim=1).max()*10),
            maximum_applied_tension_kpa=d['maximum_applied_tension_dyn_cm2']/1e4,
            max_mac_divergence_s_inv=float(divergence(state.velocity,driver.grid.spacing).abs().max()),
            max_velocity_cm_s=max(float(v.abs().max()) for v in state.velocity),
            nonlinear_iterations=n.get('iterations',0),residual_to_tolerance=n.get('residual_norm',0)/max(n.get('tolerance',1),1e-30),
            fluid_solves=n.get('fluid_solves',0),gmres_iterations=n.get('gmres_iterations',0),pressure_cycles=n.get('pressure_cycles',0))
        history.append(row);save_json(out/'history.json',history)
        writer.write(state)
        filename=f'ep_{state.step:06d}.vtu';writer.meshio.write(out/'vtk'/filename,
            writer.meshio.Mesh(array(state.x[:len(coarse.X)]),[('tetra',array(coarse.cells))],
                point_data=dict(voltage_mV=1000*array(tissue.voltage_v),calcium_uM=1000*array(tissue.gpb[:,37]),land_XS=array(tissue.land[:,0]))),
            file_format='vtu',binary=True,compression='zlib')
        frames.append((state.time,filename));root=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
        collection=ET.SubElement(root,'Collection')
        for t,name in frames:ET.SubElement(collection,'DataSet',timestep=repr(t),group='',part='0',file=name)
        ET.ElementTree(root).write(out/'vtk/ep.pvd',encoding='utf-8',xml_declaration=True)
        print(json.dumps(row),flush=True)

    def save_state():
        np.savez_compressed(out/'final_state.npz',x=array(state.x),reference=array(mesh.X),cells=array(mesh.cells),
            inner_rings=array(mesh.inner_rings),u=array(state.velocity[0]),v=array(state.velocity[1]),w=array(state.velocity[2]),
            pressure=array(state.pressure),voltage_v=array(tissue.voltage_v),gpb=array(tissue.gpb),land=array(tissue.land),
            time_s=state.time,absolute_ode_start_s=absolute_start,step=state.step)

    try:
        record();torch.cuda.synchronize();setup_seconds=time.perf_counter()-setup_start
        block_start=time.perf_counter()
        for step in range(1,steps+1):
            next_tissue,_,_=ode.step(ep,tissue,one,zs,zw,absolute_start+(step-1)*fsi.time.dt,zero)
            solid.stage(state.x,next_tissue,zs,zw)
            sample=step%log_every==0 or step==steps
            trial,info=driver.step(state,diagnostics=sample)
            state,tissue=trial,next_tissue;completed=step
            if sample:
                torch.cuda.synchronize();seconds=time.perf_counter()-block_start
                blocks.append(dict(first_step=step-log_every+1,last_step=step,steps=log_every,seconds=seconds,
                    seconds_per_step=seconds/log_every,excluded_warmup=step<=warmup))
                record(info);save_json(out/'timing_blocks.json',blocks)
                block_start=time.perf_counter()
        save_state()
        timed=[b for b in blocks if not b['excluded_warmup']]
        report=dict(status='completed',accepted_steps=completed,contract=contract,
            setup_seconds=setup_seconds,compute_seconds=sum(b['seconds'] for b in blocks),
            warmed_seconds_per_step=sum(b['seconds'] for b in timed)/sum(b['steps'] for b in timed),
            max_sampled_wall_volume_error_percent=max(abs(h['wall_volume_change_percent']) for h in history),
            final_wall_volume_change_percent=history[-1]['wall_volume_change_percent'],
            minimum_projected_diameter_mm=min(h['minimum_projected_diameter_mm'] for h in history),
            max_sampled_J_rms_error=max(h['J_rms_difference_from_one'] for h in history),
            minimum_detF=min(h['minimum_detF'] for h in history),maximum_detF=max(h['maximum_detF'] for h in history),
            max_sampled_residual_ratio=max(h['residual_to_tolerance'] for h in history),
            vtk_frames=len(frames),volumetric_stabilization=solid.volumetric_stabilization)
        save_json(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)
    except Exception as exc:
        save_state();save_json(out/'report.json',dict(status='failed',accepted_steps=completed,error=str(exc),contract=contract));raise
    finally:ep.close()


if __name__=='__main__':main()
