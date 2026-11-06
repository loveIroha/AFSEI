"""Offline diagnostics of one accepted MAC step, including its actual history.

No time advance. Component responses use the existing linear BE fluid solver;
they are an algebraic decomposition, not independent physiological pressures.
"""
from dataclasses import replace
import gc
import json
from pathlib import Path
import numpy as np
import torch
from . import mac_balanced_checkpoint as restart
from .mac_preload import MACEndocardialPressure
from .mac_transport import closed_volume_gradient
from .mac_rcr_coupled import outlet_rows
from .mac_cap_coupled import closed_rows
from .mac_ports import _stencil
from .mac_pressure_audit import cavity_masks
from .windkessel import MMHG_TO_DYN_CM2 as C


class CellSampler:
    """Trilinear cell-field samples; no clamping or extrapolation."""
    def __init__(self, grid, points):
        local=(points-points.new_tensor(grid.origin))/points.new_tensor(grid.spacing)-.5
        low=torch.floor(local).long();fraction=local-low
        if ((low<0).any() or (low+1>=low.new_tensor(grid.shape)).any()):
            raise ValueError('pressure/traction sampling requires interior cell support')
        self.entries=[]
        for i in (0,1):
            for j in (0,1):
                for k in (0,1):
                    index=low+low.new_tensor([i,j,k])
                    weight=torch.ones(len(points),device=points.device,dtype=points.dtype)
                    for axis,bit in enumerate((i,j,k)):
                        weight*=fraction[:,axis] if bit else 1-fraction[:,axis]
                    self.entries.append((index,weight))

    def __call__(self, field):
        out=field.new_zeros(len(self.entries[0][1]))
        for index,weight in self.entries:
            out+=weight*field[index[:,0],index[:,1],index[:,2]]
        return out


def normal_viscous_stress(grid, velocity, normals, sampler, viscosity):
    """2 mu n^T grad(u) n; MAC diagonals and centered cross derivatives."""
    stress=normals.new_zeros(len(normals))
    for component,u in enumerate(velocity):
        size=grid.shape[component]
        left=u.narrow(component,0,size);right=u.narrow(component,1,size)
        center=(left+right)*.5
        for axis in range(3):
            derivative=((right-left)/grid.spacing[axis] if axis==component else
                torch.gradient(center,spacing=(grid.spacing[axis],),dim=(axis,))[0])
            stress+=2*viscosity*normals[:,component]*normals[:,axis]*sampler(derivative)
    return stress


def _mean(value,weight):
    return ((value*weight).sum()/weight.sum()).item()


@torch.no_grad()
def audit_snapshot(directory, *, device='cuda'):
    directory=Path(directory)
    d,s,_=restart.load(directory/'checkpoint.npz',device=device,cardiac=True)
    if d.spatial_coupling!='legacy-balanced' or d.fluid_solver!='be-projection':
        raise ValueError('this decomposition targets the recorded legacy BE projection branch')
    with np.load(directory/'history.npz',allow_pickle=False) as z:
        history=json.loads(str(z['metadata']))
        tensor=lambda name:torch.as_tensor(z[name],device=s.fsi.x.device)
        previous=tuple(tensor(f'previous_velocity_{c}') for c in range(3))
        departure=tuple(tensor(f'departure_velocity_{c}') for c in range(3))
    interval=s.tick-history['previous_tick']
    if (history['tick']!=s.tick or interval<=0 or
            abs(history['dt_s']-s.force_load.dt_s)>1e-12):
        raise ValueError('snapshot and previous velocity clocks disagree')
    e=d._fsi_stepper(interval);grid=e.grid
    old=s.cap_geometry;x=s.fsi.x
    st=e.transfer.prepare(old)
    mode=s.cap_mode or s.mode
    rows=(outlet_rows(e,old,label=1 if mode=='filling' else 2)
          if mode!='closed' else closed_rows(e,old))
    load=MACEndocardialPressure(d.model)
    gbar=(closed_volume_gradient(old,load.faces,load.rim)+
          4*closed_volume_gradient((old+x)/2,load.faces,load.rim)+
          closed_volume_gradient(x,load.faces,load.rim))/6
    cap_cov=st.rows.reaction(torch.full_like(st.rows.area,C))/C
    m=-C*(gbar-cap_cov)
    st=replace(st,mode=m,dual=m/m.square().sum())
    _,core=cavity_masks(x,d.model.endo_faces,grid,guard_cells=d.pressure_guard_cells)
    p=rows.prepared
    current_points,current_area=p.cap.surface(x)
    frozen_area=p.area_vectors.norm(dim=-1)
    area_current=current_area.norm(dim=-1)
    material=(x-old)/history['dt_s']
    point_velocity=torch.stack([(u.flatten()[ids]*w).sum(-1)
        for u,(ids,w) in zip(s.fsi.velocity,_stencil(grid,p.points))],-1)
    relative=point_velocity-p.cap.material_points(material)
    regions={};samplers={}
    offsets=history['sample_offsets_cells']
    signed=tuple(-k for k in reversed(offsets))+tuple(offsets)
    for label,name in enumerate(('closed_cap','mv','av')):
        take=p.cap.labels==label
        a=frozen_area[take];normal=p.area_vectors[take]/a[:,None]
        speed=(relative[take]*normal).sum(-1)
        q=(a*speed).sum()
        regions[name]=dict(frozen_area_cm2=a.sum().item(),current_area_cm2=area_current[take].sum().item(),
            relative_flow_ml_s=q.item(),mean_relative_normal_speed_cm_s=(q/a.sum()).item(),
            relative_normal_speed_rms_cm_s=(_mean(speed.square(),a))**.5,
            max_abs_relative_normal_speed_cm_s=speed.abs().max().item())
        # Pressure layers are sampled on the two artificial ports. The cap
        # background includes points too close to the curved myocardial wall.
        if label==0:continue
        points=torch.cat([p.points[take]+k*min(grid.spacing)*normal for k in signed])
        sampler=CellSampler(grid,points)
        normals=normal.repeat(len(signed),1)
        pressure=sampler(s.fsi.pressure).reshape(len(signed),-1)/C
        viscous=normal_viscous_stress(grid,s.fsi.velocity,normals,sampler,e.flow.mu).reshape(len(signed),-1)/C
        layers={}
        for index,k in enumerate(signed):
            layers[str(k)]=dict(pressure_mmhg=_mean(pressure[index],a),
                viscous_normal_stress_mmhg=_mean(viscous[index],a),
                compressive_normal_stress_mmhg=_mean(pressure[index]-viscous[index],a))
        regions[name]['layers']=layers
        regions[name]['finite_distance_jumps']={str(k):dict(
            pressure_mmhg=layers[str(-k)]['pressure_mmhg']-layers[str(k)]['pressure_mmhg'],
            compressive_normal_stress_mmhg=layers[str(-k)]['compressive_normal_stress_mmhg']-
                                          layers[str(k)]['compressive_normal_stress_mmhg']) for k in offsets}
        samplers[name]=(sampler,len(signed),a)

    d.buffer.stage(s.force_load)
    full=d.model.force(x,s.fsi.time)
    d.buffer.stage(replace(s.force_load,use_land=False,anchor=None,ramp_factor=0.))
    passive=d.model.force(x,s.fsi.time)
    d.buffer.stage(s.force_load)
    parts=dict(passive_and_basal=passive,ep_land_active=full-passive,
        prescribed_wall=load.force(x,s.wall_pressure_mmhg),
        cap_material_reaction=rows.reaction(s.cap_pressure+s.cap_reference_mmhg*C))
    force=sum(parts.values())
    pressure_sum=torch.zeros_like(s.fsi.pressure)
    velocity_sum=grid.zeros(device=x.device,dtype=x.dtype)
    components={}

    def response(name,rhs,known=None,alpha=0.):
        nonlocal pressure_sum,velocity_sum
        u,rp,_=e.flow.solve_rhs(rhs)
        component_p=rp if known is None else rp+known
        pressure_sum=pressure_sum+component_p
        velocity_sum=tuple(a+b for a,b in zip(velocity_sum,u))
        layers={}
        for region,(sampler,count,a) in samplers.items():
            values=sampler(component_p).reshape(count,-1)/C
            layers[region]={str(k):_mean(values[i],a) for i,k in enumerate(signed)}
        components[name]=dict(core_mean_mmhg=component_p[core].mean().item()/C,
            pressure_mode_amplitude_mmhg=alpha,port_layers_pressure_mmhg=layers)

    response('previous_velocity_lift',previous)
    response('transport_increment',tuple(a-b for a,b in zip(departure,previous)))
    total_alpha=0.
    for name,f in parts.items():
        alpha=(st.dual*f).sum();total_alpha+=alpha
        e.transfer.reset_warm_start()
        density,_=e.transfer.base.spread(f-alpha*st.mode,st.original)
        lift=rows.lift(torch.full_like(rows.area,alpha*C))
        rhs=tuple(e.flow.dt/e.flow.rho*(a+b) for a,b in zip(density,lift))
        response(name,rhs,alpha*st.potential,alpha.item())
        components[name]['solid_power_erg_s']=(f*material).sum().item()
    response('cap_fluid_traction',tuple(-e.flow.dt/e.flow.rho*a for a in rows.lift(s.cap_pressure)))
    pressure_error=(pressure_sum-s.fsi.pressure).norm()/s.fsi.pressure.norm().clamp_min(1.)
    velocity_error=sum((a-b).square().sum() for a,b in zip(velocity_sum,s.fsi.velocity)).sqrt()
    velocity_scale=sum(a.square().sum() for a in s.fsi.velocity).sqrt().clamp_min(1.)
    error=dict(pressure_relative_l2=pressure_error.item(),
        pressure_core_mean_error_mmhg=(pressure_sum-s.fsi.pressure)[core].mean().item()/C,
        velocity_relative_l2=(velocity_error/velocity_scale).item(),
        material_force_relative_l2=((force-s.fsi.force).norm()/s.fsi.force.norm().clamp_min(1.)).item())
    checks=all(error[k]<1e-7 for k in ('pressure_relative_l2','velocity_relative_l2','material_force_relative_l2'))
    known=total_alpha*st.potential
    out=dict(time_ms=s.fsi.time*1000,tick=s.tick,dt_s=history['dt_s'],mode=mode,
        source_checkpoint=str(directory/'checkpoint.npz'),
        core_pressure_mmhg=s.fsi.pressure[core].mean().item()/C,
        logged_core_pressure_mmhg=history['accepted_balanced']['pressure_core']['mean_mmhg'],
        boundary_multiplier_mmhg=s.cap_pressure[-1].item()/C,
        pressure_mode_core_mmhg=known[core].mean().item()/C,
        remaining_core_pressure_mmhg=(s.fsi.pressure-known)[core].mean().item()/C,
        relative_volume_rate_ml_s=(gbar*material).sum().item(),
        grid_spacing_cm=grid.spacing,regions=regions,components=components,
        reconstruction=error,reconstruction_passed=checks,
        fluid_linear_responses=7,time_advanced=False,
        interpretation='Discrete BE response superposition; components are not independent physical pressures or dissipation. Layer jumps use finite offsets and are not interface trace residuals.')
    path=directory/'pressure_diagnostics.json';path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    del d,s,e,st,parts,force,full,passive,pressure_sum,velocity_sum,previous,departure
    gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    if not checks:raise RuntimeError(f'offline step reconstruction failed: {error}')
    return out
