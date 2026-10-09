"""Consistent P1 monodomain FE assembly and PCG in PyTorch, on the tensor device.

Reference coordinates: cm; voltage: V; time: s; currents: microA/cm^2.
Numerical conductivities are microS/cm. Defaults retain Active_land.ufl's
NUMERICAL coefficients (0.117, 0.039, 0.039). Its mS/cm annotation is not
dimensionally consistent with V/s/microA; converting literal mS to microS
would change the source PDE by 1000 and is intentionally not done here.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from ..p1 import prepare_p1
from ..fluid.solvers import SolverOptions, pcg


@dataclass(frozen=True)
class MonodomainParameters:
    capacitance_uf_per_cm2: float = 1.0
    surface_to_volume_per_cm: float = 0.1
    conductivity_us_per_cm: tuple = (0.117, 0.039, 0.039)

    def __post_init__(self):
        object.__setattr__(self, 'conductivity_us_per_cm', tuple(self.conductivity_us_per_cm))
        values = (self.capacitance_uf_per_cm2, self.surface_to_volume_per_cm,
                  *self.conductivity_us_per_cm)
        if len(self.conductivity_us_per_cm) != 3 or any(not isfinite(v) or v <= 0 for v in values):
            raise ValueError('positive finite Cm, Sv and three conductivities required')


def _assemble(cells, local, size):
    rows = cells[:, :, None].expand(-1, 4, 4)
    cols = cells[:, None, :].expand(-1, 4, 4)
    return torch.sparse_coo_tensor(torch.stack((rows.reshape(-1), cols.reshape(-1))),
        local.reshape(-1), (size, size), device=local.device, dtype=local.dtype,
        check_invariants=True).coalesce().to_sparse_csr()


class Monodomain:
    """Fixed reference-mesh, insulating-boundary monodomain operator.

    directions='nodal' reproduces the source PDE's P1 direction fields;
    directions='cell' explicitly uses the existing AFSI DG0 inputs instead.
    Directions are never silently normalized, averaged or converted to P1.
    """
    def __init__(self, X, cells, fiber, sheet, *, directions='nodal', parameters=None,
                 solver_options=None):
        if directions not in ('nodal', 'cell'):
            raise ValueError('directions must be nodal or cell')
        self.geometry = prepare_p1(X, cells, degree=5)
        self.X = X.detach().clone()
        self.cells = self.geometry.cells
        self.parameters = parameters or MonodomainParameters()
        self.solver_options = solver_options or SolverOptions(rtol=1e-10, atol=1e-13, check_every=5)
        count = len(X) if directions == 'nodal' else len(cells)
        for value in (fiber, sheet):
            if (value.shape != (count, 3) or value.dtype != X.dtype or value.device != X.device
                    or not torch.isfinite(value).all()):
                raise ValueError('directions must match mesh shape, dtype and device')
        N, weights = self.geometry.values, self.geometry.weights
        if directions == 'nodal':
            f = torch.einsum('qa,eai->eqi', N, fiber[self.cells])
            s = torch.einsum('qa,eai->eqi', N, sheet[self.cells])
        else:
            f, s = fiber[:, None], sheet[:, None]
        n = torch.linalg.cross(f, s)
        if (torch.linalg.vector_norm(n, dim=-1) <= 100*torch.finfo(X.dtype).eps).any():
            raise ValueError('fiber and sheet must be nonzero and linearly independent')
        gf, gs, gn = self.parameters.conductivity_us_per_cm
        outer = lambda u: u[..., :, None]*u[..., None, :]
        D = gf*outer(f)+gs*outer(s)+gn*outer(n)
        gradients = self.geometry.gradients[:, 0]
        local_mass = torch.einsum('eq,qa,qb->eab', weights, N, N)
        local_stiffness = torch.einsum('eq,eaI,eqIJ,ebJ->eab', weights, gradients, D, gradients)
        self.mass = _assemble(self.cells, local_mass, len(X))
        self.stiffness = _assemble(self.cells, local_stiffness, len(X))
        scatter = lambda local: X.new_zeros(len(X)).index_add(0, self.cells.reshape(-1), local.reshape(-1))
        self.mass_diagonal = scatter(local_mass.diagonal(dim1=-2, dim2=-1))
        self.stiffness_diagonal = scatter(local_stiffness.diagonal(dim1=-2, dim2=-1))
        if (self.mass_diagonal <= 0).any():
            raise ValueError('every EP node must belong to a tetrahedron')

    def _field(self, value, name):
        if (value.shape != (len(self.X),) or value.dtype != self.X.dtype
                or value.device != self.X.device or not torch.isfinite(value).all()):
            raise ValueError(f'{name} must be a finite P1 scalar on the mesh device')

    @staticmethod
    def _mv(matrix, value):
        return torch.sparse.mm(matrix, value[:, None]).flatten()

    @torch.no_grad()
    def step(self, voltage_v, ionic_ua_per_cm2, stimulus_ua_per_cm2, dt_s):
        if not isfinite(dt_s) or dt_s <= 0:
            raise ValueError('EP dt must be positive seconds')
        for value, name in ((voltage_v, 'voltage'), (ionic_ua_per_cm2, 'ionic current'),
                            (stimulus_ua_per_cm2, 'stimulus')):
            self._field(value, name)
        p = self.parameters
        alpha = p.surface_to_volume_per_cm*p.capacitance_uf_per_cm2/dt_s
        rhs = self._mv(self.mass, alpha*voltage_v+
                       p.surface_to_volume_per_cm*(stimulus_ua_per_cm2-ionic_ua_per_cm2))
        action = lambda v: alpha*self._mv(self.mass, v)+self._mv(self.stiffness, v)
        return pcg(action, rhs, alpha*self.mass_diagonal+self.stiffness_diagonal,
                   initial=voltage_v, options=self.solver_options)

    @torch.no_grad()
    def project_cell_field(self, cell_value):
        """Consistent L2 projection, for ODE feedback; never a stress average."""
        if (cell_value.shape != (len(self.cells),) or cell_value.device != self.X.device
                or cell_value.dtype != self.X.dtype or not torch.isfinite(cell_value).all()):
            raise ValueError('finite scalar value per cell required')
        local = cell_value[:, None]*torch.einsum('eq,qa->ea', self.geometry.weights, self.geometry.values)
        rhs = self.X.new_zeros(len(self.X)).index_add(0, self.cells.reshape(-1), local.reshape(-1))
        return pcg(lambda v: self._mv(self.mass, v), rhs, self.mass_diagonal,
                   options=self.solver_options)[0]
