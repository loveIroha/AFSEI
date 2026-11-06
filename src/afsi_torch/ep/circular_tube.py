"""Circumferential Land stress on a 3D Neo-Hookean P1 wall.

The EP conductivity axis is independent of the mechanical muscle axis. History
buffers are staged once per trial step; force/tangent evaluations never advance
ODEs or publish distortion history.
"""
from math import isfinite
import torch
from .land import land_tension, active_pk1
from ..materials import neo_hookean_pk1
from ..solids.p1_model import P1Solid, BoundaryForce
from ..solids.circular_tube import section_areas, lumen_volume
from ..p1 import prepare_p1
from ..solids.volumetric_stabilization import validate_volumetric_stabilization, deviatoric_logj_pk1


class CircularTubeSolid(P1Solid):
    def __init__(self, mesh, geometry, *, young_modulus_dyn_cm2, poisson_ratio,
                 end_spring_dyn_cm3, tension_scale, end_taper_cm, dt_s,
                 minimum_radius_ratio=.2, stretch_feedback='circumferential',
                 end_support='axial-reference-surface-spring', volumetric_stabilization=None,
                 collar_support=None, land_reference_ratio=1., mechanism='circumferential-land-pk1'):
        if mechanism not in ('circumferential-land-pk1', 'paired-transverse-land-traction'):
            raise ValueError('unsupported tube active mechanism')
        if mechanism == 'paired-transverse-land-traction' and (stretch_feedback != 'isometric-driver' or land_reference_ratio != 1. or geometry.circumferential_elements % 4):
            raise ValueError('paired actuator requires isometric lambda=1 and circumferential count divisible by four')
        self.mechanism = mechanism
        self.paired_load = None
        if isinstance(land_reference_ratio, bool) or not isfinite(land_reference_ratio) or not 0 < land_reference_ratio <= 1.2:
            raise ValueError('Land reference ratio must be positive and at most the model length cap 1.2')
        self.land_reference_ratio = float(land_reference_ratio)
        if stretch_feedback not in ('circumferential', 'isometric-driver'):
            raise ValueError('unsupported muscle stretch feedback')
        if end_support not in ('axial-reference-surface-spring', 'full-reference-surface-spring-approximation'):
            raise ValueError('unsupported finite end support')
        self.stretch_feedback, self.end_support = stretch_feedback, end_support
        if (not isfinite(young_modulus_dyn_cm2) or young_modulus_dyn_cm2 <= 0 or
                not isfinite(poisson_ratio) or not 0 <= poisson_ratio < .5):
            raise ValueError('positive E and 0 <= nu < 0.5 required')
        if (not isfinite(end_spring_dyn_cm3) or end_spring_dyn_cm3 < 0 or
                not isfinite(tension_scale) or not 0 <= tension_scale <= 1 or
                not isfinite(end_taper_cm) or not 0 < end_taper_cm < geometry.length_cm/2 or
                not isfinite(dt_s) or dt_s <= 0 or not 0 < minimum_radius_ratio < 1):
            raise ValueError('invalid tube end spring, active load, time step or no-contact limit')
        self.volumetric_stabilization = validate_volumetric_stabilization(volumetric_stabilization)
        self.field_transfer = None
        self.collar_support = collar_support
        inactive = 0.
        if collar_support is not None:
            if (set(collar_support) != {'width_cm', 'stiffness_dyn_cm4'} or
                    not all(isfinite(v) and v > 0 for v in collar_support.values()) or
                    collar_support['width_cm']+end_taper_cm >= geometry.length_cm/2):
                raise ValueError('invalid finite elastic collar support')
            inactive = collar_support['width_cm']
        self.mu = young_modulus_dyn_cm2/(2*(1+poisson_ratio))
        self.lam = young_modulus_dyn_cm2*poisson_ratio/((1+poisson_ratio)*(1-2*poisson_ratio))
        self.dt_s, self.scale = dt_s, tension_scale
        self.end_nodes = torch.unique(mesh.end_faces)
        self.tube_geometry = geometry
        g = prepare_p1(mesh.X, mesh.cells, degree=5)
        self.previous_F = mesh.X.new_zeros((len(mesh.cells), 3, 3))
        self.previous_F.copy_(torch.eye(3, device=mesh.X.device, dtype=mesh.X.dtype))
        self.land_fields = mesh.X.new_zeros((*g.weights.shape, 4))
        self.reference_tension = torch.zeros_like(g.weights)
        z = torch.einsum('qa,ea->eq', g.values, mesh.X[mesh.cells, 2])-geometry.axial_start_cm
        taper = (torch.minimum(z, geometry.length_cm-z)-inactive)/end_taper_cm
        self.taper = torch.sin(.5*torch.pi*taper.clamp(0, 1)).square()
        self.average_weights = g.weights/g.weights.sum(1, keepdim=True)
        fields = (mesh.fiber, self.previous_F, self.land_fields, self.reference_tension,
                  self.taper, self.average_weights)

        def stress(F, values, time):
            fiber, previous, land, reference, taper_q, weights = values
            applied = self._tension(F, fiber, previous, land, reference)*taper_q*self.scale
            # Affine P1 F is constant per cell, but Land's nodal states and the
            # reference taper are evaluated at the existing 15 mechanics points.
            original = neo_hookean_pk1(F, self.mu, self.lam)
            if self.mechanism == 'circumferential-land-pk1':
                original = original+active_pk1(F, fiber, (weights*applied).sum())
            if self.volumetric_stabilization is None:
                return original
            return deviatoric_logj_pk1(F, original, self.volumetric_stabilization['bulk_modulus_dyn_cm2'])

        boundaries = ()
        if end_spring_dyn_cm3:
            reference = mesh.X[mesh.end_faces].clone()
            area = .5*torch.linalg.vector_norm(torch.linalg.cross(
                reference[:, 1]-reference[:, 0], reference[:, 2]-reference[:, 0]), dim=-1)
            def axial_spring(nodes, values, time):
                ref, face_area = values
                if self.end_support == 'full-reference-surface-spring-approximation':
                    displacement = nodes-ref
                    return -end_spring_dyn_cm3*face_area/12*(displacement+displacement.sum(0))
                dz = nodes[:, 2]-ref[:, 2]
                # Exact reference-triangle consistent mass integral. Radial
                # motion is free; this is a finite axial spring, not hard fixing.
                fz = -end_spring_dyn_cm3*face_area/12*(dz+dz.sum())
                zero = torch.zeros_like(fz)
                return torch.stack((zero, zero, fz), -1)
            boundaries = (BoundaryForce(mesh.end_faces, axial_spring, (reference, area)),)
        self.collar_nodes = mesh.cells.new_empty((0,))
        if collar_support is not None:
            local_z = mesh.X[mesh.cells, 2]-geometry.axial_start_cm
            selected = ((local_z.max(1).values <= inactive+1e-12) |
                        (local_z.min(1).values >= geometry.length_cm-inactive-1e-12))
            connectivity = mesh.cells[selected]
            if not len(connectivity):
                raise ValueError('elastic collar is unresolved by the solid mesh')
            self.collar_nodes = torch.unique(connectivity)
            reference = mesh.X[connectivity].clone()
            volume = g.weights.sum(1)[selected]
            stiffness = collar_support['stiffness_dyn_cm4']
            def collar_anchor(nodes, values, time):
                ref, vol = values
                displacement = nodes-ref
                # Exact consistent P1 tetrahedral mass integral of -k*(x-X).
                return -stiffness*vol/20*(displacement+displacement.sum(0))
            boundaries += (BoundaryForce(connectivity, collar_anchor, (reference, volume)),)
        if mechanism == 'paired-transverse-land-traction':
            from .paired_tube import PairedTransverseTraction
            self.paired_load = PairedTransverseTraction(mesh, geometry, scale=tension_scale,
                taper_cm=end_taper_cm, inactive_cm=inactive, dt_s=dt_s)
            boundaries += (self.paired_load.boundary,)
        reference_areas = section_areas(mesh.X, mesh)
        def above_no_contact_limit(x):
            return section_areas(x, mesh) > minimum_radius_ratio**2*reference_areas
        checks = (above_no_contact_limit,)
        if self.paired_load is not None:
            checks += (lambda x: self.paired_load.valid_gap(x, minimum_radius_ratio),)
        super().__init__(mesh, stress, cell_fields=fields, boundaries=boundaries,
                         degree=5, validity_checks=checks)

    def _tension(self, F, fiber, previous_F, fields, reference):
        if self.stretch_feedback == 'isometric-driver':
            xs, xw, _, _ = fields.unbind(-1)
            one, zero = torch.full_like(xs, self.land_reference_ratio), torch.zeros_like(xs)
            return (land_tension(one, one, xs, xw, zero, zero, self.dt_s)-reference).clamp_min(0)
        stretch = self.land_reference_ratio*torch.linalg.vector_norm(F@fiber).clamp_min(1e-8)
        previous = self.land_reference_ratio*torch.linalg.vector_norm(previous_F@fiber).clamp_min(1e-8)
        ta = land_tension(stretch, previous, *fields.unbind(-1), self.dt_s)
        return (ta-reference).clamp_min(0)

    @torch.no_grad()
    def set_reference(self, nodal_tension):
        if self.field_transfer is not None:
            nodal_tension = self.field_transfer.prolong(nodal_tension)
        self.reference_tension.copy_(torch.einsum('qa,ea->eq', self.geometry.values,
                                                 nodal_tension[self.mesh.cells]))
        if self.paired_load is not None:
            self.paired_load.set_reference(nodal_tension)

    @torch.no_grad()
    def stage(self, previous_x, tissue, zs, zw):
        if self.paired_load is not None:
            self.paired_load.stage(tissue)
        self.previous_F.copy_(self.element_gradient(previous_x))
        nodal = torch.stack((tissue.land[:, 0], tissue.land[:, 1], zs, zw), -1)
        if self.field_transfer is not None:
            nodal = self.field_transfer.prolong(nodal)
        self.land_fields.copy_(torch.einsum('qa,eak->eqk', self.geometry.values, nodal[self.mesh.cells]))

    def cell_stretch(self, x):
        Ff = torch.einsum('eij,ej->ei', self.element_gradient(x), self.mesh.fiber)
        return torch.linalg.vector_norm(Ff, dim=-1)

    def nodal_stretch(self, ep, x):
        projected = ep.project_cell_field(self.cell_stretch(x))
        if not torch.isfinite(projected).all() or (projected <= 0).any():
            raise ValueError('nonpositive consistent-projected circumferential stretch')
        return projected

    def activation_stretch(self, geometric_stretch):
        """Land input only; never change the mechanical deformation gradient."""
        geometric = torch.ones_like(geometric_stretch) if self.stretch_feedback == 'isometric-driver' else geometric_stretch
        return self.land_reference_ratio*geometric

    def applied_tension(self, x):
        ta = torch.vmap(self._tension)(self.element_gradient(x), self.mesh.fiber,
            self.previous_F, self.land_fields, self.reference_tension)
        return self.scale*self.taper*ta

    def diagnostics(self, x):
        result = super().diagnostics(x)
        if self.collar_support is not None:
            result['collar_support'] = self.collar_support
            result['maximum_collar_displacement_cm'] = float(torch.linalg.vector_norm(
                x[self.collar_nodes]-self.mesh.X[self.collar_nodes], dim=-1).max())
        areas = section_areas(x, self.mesh)
        reference = section_areas(self.mesh.X, self.mesh)
        result.update(lumen_volume_ml=float(lumen_volume(x, self.mesh)),
            minimum_section_area_cm2=float(areas.min()),
            mid_section_area_cm2=float(areas[len(areas)//2]),
            minimum_radius_ratio=float((areas/reference).min().sqrt()),
            maximum_applied_tension_dyn_cm2=float(self.applied_tension(x).max()))
        result['minimum_geometric_hoop_stretch'] = float(self.cell_stretch(x).min())
        result['maximum_end_displacement_cm'] = float((x[self.end_nodes]-self.mesh.X[self.end_nodes]).norm(dim=-1).max())
        result['muscle_stretch_feedback'] = self.stretch_feedback
        result['land_reference_ratio'] = self.land_reference_ratio
        result['volumetric_stabilization'] = self.volumetric_stabilization
        result['active_mechanism'] = self.mechanism
        if self.paired_load is not None:
            result.update(self.paired_load.diagnostics(x))
        return result
