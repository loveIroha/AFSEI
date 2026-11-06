"""Explicit 2D hoop-muscle reduction driven by the existing GPB/Land states.

Paired lumen edges represent omitted circumferential fibers. No displacement
wave is prescribed. Electrophysiological fiber direction remains axial.
The load is a reduced model, not the 3D myocardial active-PK1 law.
"""
import numpy as np
import torch
from math import isfinite
from .land import land_tension, distortion_step
from ..mac.execution import tensor_kernel


def land_resting_equilibrium(calcium_um, stretch):
    """Exact fixed point of the existing Land ODE at fixed Ca/stretch and Z=0.

    Used only to initialize this tube benchmark; no original Land rate changes.
    """
    ca50=(.535-2.4*(stretch.clamp_max(1.2)-1)).clamp_min(.1)
    ratio=(calcium_um/ca50).square()
    ct=ratio/(1+ratio)
    kb=.04*.35**2.4/(1-.25-.5*(1-.25))
    tm_over_xu=(kb/.04)*ct.clamp_min(1e-12).pow(-2.4)
    xu=1/(1+1+2/3+tm_over_xu)
    return torch.stack(((2/3)*xu,xu,ct,tm_over_xu*xu),-1)


class PairedHoopLoad:
    def __init__(self, mesh, config, *, tension_scale=1., end_taper_cm=.1,
                 rest_reference='land-equilibrium-increment',
                 stretch_model='lumen-gap', end_inactive_cm=0., fused=True):
        if not 0 <= tension_scale <= 1 or not 0 < end_taper_cm < config.length_cm/2:
            raise ValueError('tension scale in [0,1] and a finite end taper inside the tube required')
        self.mesh, self.config = mesh, config
        if rest_reference != 'land-equilibrium-increment':
            raise ValueError('this demo requires its explicit resting-tension reference')
        if stretch_model not in ('lumen-gap', 'plane-strain-isometric'):
            raise ValueError('stretch_model must be lumen-gap or plane-strain-isometric')
        # The omitted circumferential direction has F33=1 in plane strain.
        # Isometric tension drives this reduced actuator; circumferential
        # electromechanical feedback is not resolved. Keep the old gap proxy
        # as default so saved configurations reproduce their original model.
        self.stretch_model = stretch_model
        if not isfinite(end_inactive_cm) or not 0 <= end_inactive_cm < config.length_cm/2-end_taper_cm:
            raise ValueError('end_inactive_cm must leave a nonempty active region and taper')
        self.end_inactive_cm = end_inactive_cm
        self.scale, self.taper_length = tension_scale, end_taper_cm
        self.nx, self.ny = config.axial_elements, config.thickness_elements
        n = (self.nx+1)*(self.ny+1)
        self.lower = torch.arange(self.nx+1, device=mesh.X.device)*(self.ny+1)+self.ny
        self.upper = self.lower+n
        # Locate actual P2 boundary midpoints by coordinates once during setup.
        edges = {}
        for row in mesh.cells.detach().cpu().tolist():
            for a, b, mid in ((row[0], row[1], row[3]), (row[0], row[2], row[4]), (row[1], row[2], row[5])):
                edges[tuple(sorted((a, b)))] = mid
        def edge_ids(nodes):
            ids = nodes.detach().cpu().tolist()
            return torch.tensor([[a,b,edges[tuple(sorted((a,b)))]] for a,b in zip(ids[:-1],ids[1:])],
                                device=mesh.X.device)
        self.lower_edges, self.upper_edges = edge_ids(self.lower), edge_ids(self.upper)
        z,w = np.polynomial.legendre.leggauss(3)
        s = mesh.X.new_tensor((z+1)/2)
        self.N = torch.stack(((1-s)*(1-2*s),s*(2*s-1),4*s*(1-s)), -1)
        self.L = torch.stack((1-s,s), -1)
        self.weights = mesh.X.new_tensor(w/2)*(config.length_cm/self.nx)
        xq = torch.einsum('qa,eai->eqi', self.N, mesh.X[self.lower_edges])[..., 0]-config.inlet_buffer_cm
        a = (torch.minimum(xq, config.length_cm-xq)-end_inactive_cm)/end_taper_cm
        self.taper = torch.sin(.5*torch.pi*a.clamp(0,1)).square()
        self.fields = mesh.X.new_zeros((self.nx+1, 4))
        self.previous_stretch = mesh.X.new_ones(self.nx+1)
        self.reference_tension = mesh.X.new_zeros(self.nx+1)
        self.dt = 5e-5
        self._force = tensor_kernel(self._force_impl, mesh.X.device) if fused else self._force_impl

    def stretch_sections(self, x):
        return torch.linalg.vector_norm(x[self.upper]-x[self.lower], dim=-1)/self.config.lumen_height_cm

    def nodal_stretch(self, x):
        sections = self.stretch_sections(x)
        if self.stretch_model == 'plane-strain-isometric':
            sections = torch.ones_like(sections)
        wall = sections[:, None].expand(-1,self.ny+1).reshape(-1)
        return torch.cat((wall,wall))

    def set_history(self, tissue, previous_stretch, zs, zw, dt):
        fields = torch.stack((tissue.land[:,0],tissue.land[:,1],zs,zw), -1)
        self.fields.copy_(.5*(fields[self.lower]+fields[self.upper]))
        self.previous_stretch.copy_(.5*(previous_stretch[self.lower]+previous_stretch[self.upper]))
        self.dt = dt

    def set_reference(self, tension):
        self.reference_tension.copy_(.5*(tension[self.lower]+tension[self.upper]))

    def _traction_impl(self, x):
        lower = torch.einsum('qa,eai->eqi',self.N,x[self.lower_edges])
        upper = torch.einsum('qa,eai->eqi',self.N,x[self.upper_edges])
        delta = upper-lower
        gap = torch.linalg.vector_norm(delta,dim=-1)
        stretch = gap/self.config.lumen_height_cm
        if self.stretch_model == 'plane-strain-isometric':
            stretch = torch.ones_like(stretch)
        pairs = torch.stack((self.fields[:-1],self.fields[1:]),1)
        fields = torch.einsum('qa,eak->eqk',self.L,pairs)
        previous = torch.einsum('qa,ea->eq',self.L,torch.stack((self.previous_stretch[:-1],self.previous_stretch[1:]),1))
        ta = land_tension(stretch,previous,*fields.unbind(-1),self.dt)
        baseline = torch.einsum('qa,ea->eq',self.L,
            torch.stack((self.reference_tension[:-1],self.reference_tension[1:]),1))
        ta = (ta-baseline).clamp_min(0)
        # Hoop balance p_a = Ta h/R0; reference line measure and reference radius.
        traction = self.scale*(2*self.config.wall_thickness_cm/self.config.lumen_height_cm)*ta*self.taper
        return delta, gap, stretch, ta, traction

    def _force_impl(self, x):
        delta, gap, _, _, traction = self._traction_impl(x)
        qforce = traction[...,None]*delta/gap[...,None]
        local = torch.einsum('q,qa,eqi->eai',self.weights,self.N,qforce)
        result = torch.zeros_like(x).index_add(0,self.lower_edges.reshape(-1),local.reshape(-1,2))
        return result.index_add(0,self.upper_edges.reshape(-1),-local.reshape(-1,2))

    def force(self,x):
        return self._force(x)

    def load_fields(self, x):
        """Sample-only fields on the actual paired-edge quadrature."""
        delta, _, stretch, ta, traction = self._traction_impl(x)
        lower = torch.einsum('qa,eai->eqi',self.N,x[self.lower_edges])
        return dict(points=lower+.5*delta, muscle_stretch=stretch,
                    tension_increment=ta, traction=traction)

    def diagnostics(self, x):
        fields = self.load_fields(x)
        stretch, ta, traction = (fields[k] for k in ('muscle_stretch','tension_increment','traction'))
        weights = self.weights.expand_as(traction)
        values = torch.stack((traction.max(),
            (traction*weights).sum()/self.config.length_cm,
            ta.max(),(ta*weights).sum()/self.config.length_cm,
            stretch.min(),stretch.max(),
            ((ta == 0)*weights).sum()/self.config.length_cm)).tolist()
        return dict(zip(('active_traction_max_dyn_cm2','active_traction_mean_dyn_cm2',
            'active_edge_tension_increment_max_dyn_cm2','active_edge_tension_increment_mean_dyn_cm2',
            'active_muscle_stretch_min','active_muscle_stretch_max',
            'active_zero_increment_length_fraction'),values))


class ActiveTubeSolid:
    """Adapter retaining ValveStepper and TriangleTransfer without changes."""
    def __init__(self, passive, active):
        self.passive,self.active = passive,active
        self.mesh,self.geometry = passive.mesh,passive.geometry

    def validate(self,x):
        self.passive.validate(x)
        if (self.active.stretch_sections(x) < .2).any():
            raise ValueError('tube reached the no-contact demo limit (gap < 20% of reference)')

    def force_and_det(self,x):
        force,J = self.passive.force_and_det(x)
        return force+self.active.force(x),J

    def force(self,x):
        return self.force_and_det(x)[0]
