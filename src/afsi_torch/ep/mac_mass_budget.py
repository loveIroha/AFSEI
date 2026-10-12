"""Read-only closed LV surface mass audit on the unchanged Cartesian MAC grid.

Peskin traces measure interface slip with the existing IB kernel. An RT0
trace is an independent diagnostic of the MAC cell divergence; it never
replaces IB interpolation or the simulation velocity. Surface integration
is refined in chunks, without a fitted blood-volume mesh or fluid solve.
"""
from dataclasses import dataclass,asdict
from math import isfinite
import torch
from .mac_ports import _stencil,_subdivision
from ..p1 import cavity_rim,cavity_volume
from ..mac.grid import divergence


@dataclass(frozen=True)
class MassAuditTargets:
    # Explicit engineering diagnostic targets, not physiological standards
    # or new nonlinear/time-step acceptance tolerances.
    signed_mass_ml_s: float = .01
    absolute_interface_leak_ml_s: float = .1
    quadrature_change_ml_s: float = .001

    def __post_init__(self):
        if any(not isfinite(v) or v<=0 for v in asdict(self).values()):
            raise ValueError('positive finite diagnostic targets required')


def peskin_trace(grid,velocity,points):
    return torch.stack([(u.reshape(-1)[ids]*weights).sum(-1)
        for u,(ids,weights) in zip(velocity,_stencil(grid,points))],-1)


def rt0_trace(grid,velocity,points):
    """Cellwise affine normal component, with exactly the MAC divergence.

    Normal velocity is continuous across cell faces; tangential components
    may jump. Surface quadrature must therefore still be convergence-tested.
    """
    scaled=(points-points.new_tensor(grid.origin))/points.new_tensor(grid.spacing)
    cells=torch.floor(scaled).to(torch.int64)
    if (cells<0).any() or (cells>=cells.new_tensor(grid.shape)).any():
        raise ValueError('RT0 diagnostic point outside MAC box')
    fraction=scaled-cells
    values=[]
    for c,u in enumerate(velocity):
        shape=grid.face_shape(c)
        index=(cells[:,0]*shape[1]+cells[:,1])*shape[2]+cells[:,2]
        stride=(shape[1]*shape[2],shape[2],1)[c]
        flat=u.reshape(-1)
        values.append((1-fraction[:,c])*flat[index]+fraction[:,c]*flat[index+stride])
    return torch.stack(values,-1)


class ClosedCavityMassAudit:
    def __init__(self,reference_x,endocardial_faces,grid,*,chunk_points=32768,rim=None):
        if (reference_x.ndim!=2 or reference_x.shape[1]!=3 or reference_x.dtype!=torch.float64
                or not torch.isfinite(reference_x).all() or type(chunk_points) is not int
                or chunk_points<16384):
            raise ValueError('finite float64 nodal geometry and chunk_points>=16384 required')
        self.X,self.faces,self.grid=reference_x,endocardial_faces,grid
        self.rim=cavity_rim(endocardial_faces) if rim is None else rim
        if (self.rim.ndim!=2 or self.rim.shape[1]!=2 or self.rim.dtype!=torch.int64
                or self.rim.device!=reference_x.device):
            raise ValueError('matching oriented cavity rim required')
        self.chunk_points=chunk_points
        self.rules={}
        if cavity_volume(reference_x,self.faces,self.rim)<=0:
            raise ValueError('positive blood-outward closed volume required')

    def validate_geometry(self,nodal):
        if nodal.shape!=self.X.shape or nodal.dtype!=self.X.dtype or nodal.device!=self.X.device:
            raise ValueError('matching closed cavity geometry required')
        valid=torch.isfinite(nodal).all() & (cavity_volume(nodal,self.faces,self.rim)>0)
        if not valid:
            raise ValueError('finite positive blood-outward closed geometry required')

    def triangles(self,nodal):
        # Source endocardial normals point into blood; reverse them. The
        # rim's original oriented edges then produce a blood-outward cap.
        wall=nodal[self.faces[:,[0,2,1]]]
        center=nodal[self.rim[:,0]].mean(0)
        cap=torch.cat((nodal[self.rim],center[None,None].expand(len(self.rim),1,3)),1)
        return wall,cap

    @staticmethod
    def area(triangles):
        return .5*torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])

    def rule(self,level):
        if level not in self.rules:
            three=self.X.new_tensor([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
            self.rules[level]=torch.einsum('ra,qab->qrb',three,_subdivision(level,self.X)).reshape(-1,3)
        return self.rules[level]

    def region(self,triangles,material_triangles,velocity,level):
        bary=self.rule(level); q=len(bary)
        batch=max(1,self.chunk_points//q)
        fields={k:[] for k in ('fluid_flux','material_flux','relative_flux','absolute_relative_flux',
                              'rt0_fluid_flux','rt0_absolute_relative_flux','area','normal_slip_square','max_normal_slip')}
        for start in range(0,len(triangles),batch):
            nodes=triangles[start:start+batch]; material=material_triangles[start:start+batch]
            area=self.area(nodes); weight=area/q
            points=torch.einsum('qa,fai->fqi',bary,nodes)
            w=torch.einsum('qa,fai->fqi',bary,material)
            u=peskin_trace(self.grid,velocity,points.reshape(-1,3)).reshape_as(points)
            conservative=rt0_trace(self.grid,velocity,points.reshape(-1,3)).reshape_as(points)
            fluid=(u*weight[:,None]).sum(-1)
            motion=(w*weight[:,None]).sum(-1)
            relative=fluid-motion
            rt0_fluid=(conservative*weight[:,None]).sum(-1)
            point_area=weight.norm(dim=-1)
            slip=relative/point_area[:,None]
            values=dict(fluid_flux=fluid.sum(1),material_flux=motion.sum(1),
                relative_flux=relative.sum(1),absolute_relative_flux=relative.abs().sum(1),
                rt0_fluid_flux=rt0_fluid.sum(1),rt0_absolute_relative_flux=(rt0_fluid-motion).abs().sum(1),
                area=area.norm(dim=-1),normal_slip_square=(slip.square()*point_area[:,None]).sum(1),
                max_normal_slip=slip.abs().amax(1))
            for key,value in values.items(): fields[key].append(value)
        fields={key:torch.cat(value) for key,value in fields.items()}
        report={key+'_ml_s':fields[key].sum().item() for key in
            ('fluid_flux','material_flux','relative_flux','absolute_relative_flux','rt0_fluid_flux','rt0_absolute_relative_flux')}
        report.update(area_cm2=fields['area'].sum().item(),
            normal_slip_rms_cm_s=(fields['normal_slip_square'].sum()/fields['area'].sum()).sqrt().item(),
            normal_slip_max_cm_s=fields['max_normal_slip'].max().item(),
            quadrature_points=len(triangles)*q,subdivision_level=level)
        return report,fields

    @torch.no_grad()
    def evaluate(self,previous_x,x,velocity,*,dt,levels=(0,1,2),targets=None,return_fields=False):
        if (not isfinite(dt) or dt<=0 or len(levels)<2 or tuple(sorted(set(levels)))!=tuple(levels)
                or any(type(k) is not int or not 0<=k<=4 for k in levels)):
            raise ValueError('positive dt and >=2 distinct increasing levels in 0..4 required')
        for value in (previous_x,x):
            if (value.shape!=self.X.shape or value.dtype!=self.X.dtype or value.device!=self.X.device
                    or not torch.isfinite(value).all()):
                raise ValueError('finite matching nodal history required')
        self.grid.check_velocity(velocity)
        if any(v.dtype!=self.X.dtype or v.device!=self.X.device or not torch.isfinite(v).all() for v in velocity):
            raise ValueError('finite float64 matching MAC velocity required')
        targets=targets or MassAuditTargets()
        w=(x-previous_x)/dt
        moving=self.triangles(w)
        old=self.triangles(previous_x); current=self.triangles(x); middle=self.triangles((previous_x+x)/2)
        volume_old=cavity_volume(previous_x,self.faces,self.rim)
        volume_new=cavity_volume(x,self.faces,self.rim)
        if volume_old<=0 or volume_new<=0:
            raise ValueError('positive closed volumes required')
        secant=(volume_new-volume_old)/dt
        def material_rate(surface):
            return sum((self.area(t)*v.mean(1)).sum() for t,v in zip(surface,moving))
        old_rate,new_rate=material_rate(old),material_rate(current)
        swept=(old_rate+4*material_rate(middle)+new_rate)/6
        levels_report=[]; last_fields={}
        for level in levels:
            # Long radial cap fans need more points than the LV wall triangles.
            wall,wall_fields=self.region(current[0],moving[0],velocity,level)
            cap,cap_fields=self.region(current[1],moving[1],velocity,level+2)
            fluid=wall['fluid_flux_ml_s']+cap['fluid_flux_ml_s']
            material=wall['material_flux_ml_s']+cap['material_flux_ml_s']
            relative=wall['relative_flux_ml_s']+cap['relative_flux_ml_s']
            levels_report.append(dict(level=level,wall=wall,cap=cap,
                closed_fluid_flux_ml_s=fluid,closed_material_flux_ml_s=material,
                closed_relative_flux_ml_s=relative,
                closed_absolute_relative_flux_ml_s=wall['absolute_relative_flux_ml_s']+cap['absolute_relative_flux_ml_s'],
                rt0_closed_fluid_flux_ml_s=wall['rt0_fluid_flux_ml_s']+cap['rt0_fluid_flux_ml_s'],
                flux_decomposition_error_ml_s=abs(fluid-material-relative)))
            last_fields=dict(wall=wall_fields,cap=cap_fields)
        last,prior=levels_report[-1],levels_report[-2]
        changes={key:abs(last[key]-prior[key]) for key in
            ('closed_relative_flux_ml_s','closed_absolute_relative_flux_ml_s','rt0_closed_fluid_flux_ml_s')}
        for name in ('wall','cap'):
            for key in ('relative_flux_ml_s','absolute_relative_flux_ml_s'):
                changes[name+'_'+key]=abs(last[name][key]-prior[name][key])
        quadrature_passed=all(changes[k]<=targets.quadrature_change_ml_s for k in changes if not k.startswith('rt0_'))
        geometry_error=abs((swept-secant).item())
        budget_passed=(levels[-1]>=2 and quadrature_passed and geometry_error<=1e-7
            and abs(last['closed_relative_flux_ml_s'])<=targets.signed_mass_ml_s
            and abs(last['closed_fluid_flux_ml_s'])<=targets.signed_mass_ml_s
            and last['closed_absolute_relative_flux_ml_s']<=targets.absolute_interface_leak_ml_s)
        report=dict(scope='read-only closed LV wall plus material cap; no physical port flow',
            dt_s=dt,volume_previous_ml=volume_old.item(),volume_current_ml=volume_new.item(),
            secant_volume_rate_ml_s=secant.item(),swept_surface_material_rate_ml_s=swept.item(),
            swept_geometry_identity_error_ml_s=geometry_error,
            old_geometry_material_rate_ml_s=old_rate.item(),current_geometry_material_rate_ml_s=new_rate.item(),
            endpoint_vs_secant_material_rate_ml_s=(new_rate-secant).item(),
            mac_max_divergence_s_inv=divergence(velocity,self.grid.spacing).abs().max().item(),
            levels=levels_report,last_level_change_ml_s=changes,targets=asdict(targets),
            targets_scope='engineering diagnostic targets; do not alter step acceptance',
            peskin_quadrature_converged_to_targets=quadrature_passed,
            rt0_quadrature_converged_to_target=changes['rt0_closed_fluid_flux_ml_s']<=targets.quadrature_change_ml_s,
            surface_mass_audit_passed=budget_passed,
            rt0_trace_used_only_for_diagnostic=True,source_state_modified=False,
            windkessel_actual_port_flow_connected=False,three_field_integration_implemented=False)
        return (report,last_fields) if return_fields else report
