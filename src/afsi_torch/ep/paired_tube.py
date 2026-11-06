"""Reduced, fixed-y paired actuator on the reference lumen surface.

t0 = -scale * (h0/R0) * DeltaTa * taper * n0_y * e_y [dyn/cm2].
The normal points out of the lumen. Reference-area integration gives nodal
forces in dyn. This is an external EP-driven squeeze, not active hoop PK1,
not a fluid pressure, and not a prescribed displacement/travelling waveform.
Three-point triangle quadrature integrates P1 test/trial products exactly
when the axial taper is constant. The load is dead with respect to mechanics;
GPB/Land states are staged once per EP step, outside Newton iterations.
"""
import torch
from .land import land_tension
from ..solids.p1_model import BoundaryForce


class PairedTransverseTraction:
    def __init__(self, mesh, geometry, *, scale, taper_cm, inactive_cm, dt_s):
        self.mesh, self.geometry, self.scale, self.dt = mesh, geometry, scale, dt_s
        self.faces = mesh.inner_faces
        self.N = mesh.X.new_tensor([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
        vertices = mesh.X[self.faces]
        cross = torch.linalg.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])
        twice_area = cross.norm(dim=-1)
        self.area = twice_area/2
        self.normal_y = cross[:, 1]/twice_area
        center = vertices.mean(1)[:, :2]-mesh.X.new_tensor(geometry.center_xy_cm)
        if not ((cross[:, :2]*center).sum(-1) > 0).all():
            raise ValueError('paired traction requires lumen-outward inner faces')
        zq = torch.einsum('qa,eai->eqi', self.N, vertices)[..., 2]-geometry.axial_start_cm
        a = (torch.minimum(zq, geometry.length_cm-zq)-inactive_cm)/taper_cm
        self.taper = torch.sin(.5*torch.pi*a.clamp(0, 1)).square()
        self.reference = mesh.X.new_zeros(len(mesh.X))
        self.local_force = torch.zeros_like(vertices)
        self.pressure = torch.zeros_like(zq)
        self.increment = torch.zeros_like(zq)
        # Persistent fields are copied in place so compiled FSI sees every update.
        self.boundary = BoundaryForce(self.faces, lambda nodes, fields, time: fields[0], (self.local_force,))
        nt = geometry.circumferential_elements
        upper = torch.arange(1, nt//2, device=mesh.X.device)
        self.upper = mesh.inner_rings[:, upper]
        self.lower = mesh.inner_rings[:, nt-upper]
        self.reference_gap = mesh.X[self.upper, 1]-mesh.X[self.lower, 1]
        self.pole_top, self.pole_bottom = mesh.inner_rings[:, nt//4], mesh.inner_rings[:, 3*nt//4]

    def set_reference(self, tension):
        self.reference.copy_(tension)

    @torch.no_grad()
    def stage(self, tissue):
        # This variant deliberately uses the same lambda=1 isometric assumption
        # as the planar actuator. Nodal states are interpolated onto the surface.
        fields = torch.einsum('qa,eak->eqk', self.N, tissue.land[self.faces, :2])
        one, zero = torch.ones_like(self.pressure), torch.zeros_like(self.pressure)
        baseline = torch.einsum('qa,ea->eq', self.N, self.reference[self.faces])
        self.increment.copy_((land_tension(one, one, fields[..., 0], fields[..., 1], zero, zero, self.dt)-baseline).clamp_min(0))
        self.pressure.copy_(self.scale*(self.geometry.wall_thickness_cm/self.geometry.inner_radius_cm)*self.increment*self.taper)
        ty = -self.pressure*self.normal_y[:, None]
        fy = self.area[:, None]*torch.einsum('qa,eq->ea', self.N, ty)/3
        self.local_force.zero_()
        self.local_force[..., 1].copy_(fy)

    def valid_gap(self, x, minimum_ratio):
        return (x[self.upper, 1]-x[self.lower, 1]) > minimum_ratio*self.reference_gap

    def diagnostics(self, x):
        gap = x[self.pole_top, 1]-x[self.pole_bottom, 1]
        inward = torch.stack((self.mesh.X[self.pole_top, 1]-x[self.pole_top, 1],
                             x[self.pole_bottom, 1]-self.mesh.X[self.pole_bottom, 1]))
        return dict(minimum_vertical_gap_cm=float(gap.min()),
                    maximum_pole_inward_displacement_cm=float(inward.max()),
                    maximum_paired_pressure_dyn_cm2=float(self.pressure.max()),
                    maximum_paired_traction_dyn_cm2=float((self.pressure*self.normal_y[:, None]).abs().max()),
                    maximum_applied_tension_dyn_cm2=float((self.increment*self.taper*self.scale).max()),
                    active_force_resultant_dyn=self.local_force.sum((0, 1)).tolist())
