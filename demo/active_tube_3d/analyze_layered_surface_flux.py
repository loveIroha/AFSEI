"""Refine only diagnostic surface quadrature on saved accepted states (CPU).

No flow/solid mesh or solver is changed and no PDE/ODE is advanced.
This separates RT0 surface integration error from material-wall/RT0 flux gap.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.sealed_ports import SealedDomain
from afsi_torch.simulation.tube_budget import TubeBudget,TriangleSurface


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--levels',type=int,nargs='+',default=[2,3,4])
    p.add_argument('--regions',action='store_true',help='split wall flux into the two end zones and interior')
    a=p.parse_args();torch.set_num_threads(8)
    c=json.loads((a.case/'config.resolved.json').read_text());g,_,_,_=resolve_config(c)
    mesh=generate_circular_tube(g)
    grid=MACGrid(tuple(c['fluid']['shape']),tuple(c['fluid']['lengths_cm']),tuple(c['fluid']['origin_cm']))
    domain=SealedDomain(grid,g,c['hydraulic_domain'],mesh.X,dimension=3)
    with np.load(a.case/'final_state.npz') as s:
        get=lambda key:torch.from_numpy(s[key].copy())
        old,new=get('previous_x'),get('x');velocity=tuple(get(k) for k in ('u','v','w'))
    b=TubeBudget(mesh,domain,initial_x=old);dt=c['time']['fsi_dt_s'];b.accepted(old,new,velocity,dt)
    old,new,rates,_=b.last
    def old_motion(surface):
        v=(new[surface.indices]-old[surface.indices]).mean(1)/dt
        return float((v*surface.area_vectors(old)).sum())
    old_wall_motion=old_motion(b.wall)
    fixed_change=float(b.fixed_volume(new)-b.fixed_volume(old))
    rows=[]
    def flux(surface,level):
        if level<=4:return surface.flux(old,velocity,grid,level)
        return sum(TriangleSurface(ids,mesh.X).flux(old,velocity,grid,level)
                   for ids in surface.indices.split(64))
    for level in a.levels:
        if not 0<=level<=6:raise ValueError('diagnostic levels must be between zero and six')
        wall=float(flux(b.wall,level));conn=float(flux(b.connectors,level))
        row=dict(level=level,subtriangles_per_face=4**level,wall_rt0_flux_ml_s=wall,connector_rt0_flux_ml_s=conn,
            divergence_surface_quadrature_defect_ml_s=wall+conn+float(rates[1]-rates[0]),
            wall_motion_minus_rt0_ml_s=float(rates[3])-wall,
            old_geometry_wall_motion_minus_rt0_ml_s=old_wall_motion-wall,
            connector_motion_minus_rt0_ml_s=float(rates[4])-conn)
        rows.append(row);print(json.dumps(row),flush=True)
    result=dict(scope='only saved final step; quadrature refinement, not mesh refinement',case=str(a.case),
        wall_motion_old_geometry_ml_s=old_wall_motion,
        geometric_time_increment_ml_s=float(rates[3])-old_wall_motion,
        dt_s=dt,fixed_control_volume_change_ml=fixed_change,net_inflow_ml_s=float(rates[0]-rates[1]),
        fixed_control_increment_imbalance_ml_s=fixed_change/dt-float(rates[0]-rates[1]),rows=rows,
        note='RT0 differs from regularized IB interpolation; wall flux gap is a spatial diagnostic, not nonlinear solver residual')
    if a.regions:
        z=mesh.X[b.wall.indices,2].mean(1)-g.axial_start_cm
        width=2*g.length_cm/g.axial_elements
        selected=[z<width,(z>=width)&(z<=g.length_cm-width),z>g.length_cm-width]
        result['regions']=[]
        for name,mask in zip(('first_two_axial_elements','interior','last_two_axial_elements'),selected):
            s=TriangleSurface(b.wall.indices[mask],mesh.X)
            fluid=float(flux(s,max(a.levels)));motion=float(s.swept(old,new)/dt)
            old_rate=old_motion(s)
            result['regions'].append(dict(name=name,faces=int(mask.sum()),motion_flux_ml_s=motion,
                rt0_fluid_flux_ml_s=fluid,motion_minus_rt0_ml_s=motion-fluid,
                old_geometry_motion_ml_s=old_rate,old_geometry_motion_minus_rt0_ml_s=old_rate-fluid,
                geometric_time_increment_ml_s=motion-old_rate))
        result['reference_end_zone_width_um']=width*10000
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
