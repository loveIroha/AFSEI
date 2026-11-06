"""P1 monodomain with vertex quadrature and optional AmgX PCG-AMG.

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
from .numerics import MonodomainNumerics


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
    vertices = cells.shape[1]
    rows = cells[:, :, None].expand(-1, vertices, vertices)
    cols = cells[:, None, :].expand(-1, vertices, vertices)
    return torch.sparse_coo_tensor(torch.stack((rows.reshape(-1), cols.reshape(-1))),
        local.reshape(-1), (size, size), device=local.device, dtype=local.dtype,
        check_invariants=True).coalesce().to_sparse_csr()


class Monodomain:
    """Fixed reference-mesh, insulating-boundary monodomain operator.

    directions='nodal' reproduces the source PDE's P1 direction fields;
    directions='cell' explicitly uses the existing AFSI DG0 inputs instead.
    Directions are never silently normalized, averaged or converted to P1.
    Default diffusion uses Crank-Nicolson; the supplied ionic/stimulus field is
    the existing frozen or substep-averaged source, not an implicit reaction solve.
    """
    def __init__(self, X, cells, fiber, sheet, *, directions='nodal', parameters=None,
                 solver_options=None, numerics=None):
        if directions not in ('nodal', 'cell'):
            raise ValueError('directions must be nodal or cell')
        self.numerics = numerics or MonodomainNumerics()
        diffusion = (self.numerics.discretization if self.numerics.diffusion_quadrature=='match_mass'
                     else self.numerics.diffusion_quadrature)
        self.geometry = prepare_p1(X, cells,
            # Affine simplex volume/gradient use a low-order geometry rule.
            # The nodal PDE evaluates tensor and mass at every simplex vertex.
            degree=5 if 'consistent_p1' in (self.numerics.discretization,diffusion) else 1)
        self.X = X.detach().clone()
        self.cells = self.geometry.cells
        self.parameters = parameters or MonodomainParameters()
        self.solver_options = solver_options or self.numerics.solver
        self._systems = {}
        dimension, vertices = X.shape[1], self.cells.shape[1]
        count = len(X) if directions == 'nodal' else len(cells)
        for value in (fiber, sheet):
            if (value.shape != (count, dimension) or value.dtype != X.dtype or value.device != X.device
                    or not torch.isfinite(value).all()):
                raise ValueError('directions must match mesh shape, dtype and device')
        volume = self.geometry.weights.sum(dim=1)
        self.reference_volume = volume.clone()
        self._deformation_input = None
        self._deformation_version = None
        self.deformation_gradient = torch.eye(dimension,device=X.device,dtype=X.dtype).expand(len(cells),-1,-1).clone()
        self.volume_ratio = torch.ones_like(volume)
        if self.numerics.discretization == 'nodal_p1':
            N = torch.eye(vertices, dtype=X.dtype, device=X.device)
            weights = (volume[:, None]/vertices).expand(-1, vertices)
        else:
            N, weights = self.geometry.values, self.geometry.weights
        local_mass = torch.einsum('eq,qa,qb->eab', weights, N, N)
        if diffusion!=self.numerics.discretization:
            if diffusion=='nodal_p1':
                N=torch.eye(vertices,dtype=X.dtype,device=X.device)
                weights=(volume[:,None]/vertices).expand(-1,vertices)
            else:
                N,weights=self.geometry.values,self.geometry.weights
        if directions == 'nodal':
            f = torch.einsum('qa,eai->eqi', N, fiber[self.cells])
            s = torch.einsum('qa,eai->eqi', N, sheet[self.cells])
        else:
            f, s = fiber[:, None], sheet[:, None]
        n = torch.linalg.cross(f, s) if dimension == 3 else f[..., 0]*s[..., 1]-f[..., 1]*s[..., 0]
        cross_norm = torch.linalg.vector_norm(n, dim=-1) if dimension == 3 else n.abs()
        if (cross_norm <= 100*torch.finfo(X.dtype).eps).any():
            raise ValueError('fiber and sheet must be nonzero and linearly independent')
        gf, gs, gn = self.parameters.conductivity_us_per_cm
        outer = lambda u: u[..., :, None]*u[..., None, :]
        D = gf*outer(f)+gs*outer(s)
        if dimension == 3:
            D = D+gn*outer(n)
        self.reference_conductivity = D
        gradients = self.geometry.gradients[:, 0]
        local_stiffness = torch.einsum('eq,eaI,eqIJ,ebJ->eab', weights, gradients, D, gradients)
        scatter = lambda local: X.new_zeros(len(X)).index_add(0, self.cells.reshape(-1), local.reshape(-1))
        self.mass_diagonal = scatter(local_mass.diagonal(dim1=-2, dim2=-1))
        self.stiffness_diagonal = scatter(local_stiffness.diagonal(dim1=-2, dim2=-1))
        # Release quadrature work before the large global COO/coalesce peak.
        del f,s,n,D,N,weights
        if self.numerics.discretization=='nodal_p1':
            del local_mass
        self.stiffness = _assemble(self.cells, local_stiffness, len(X))
        del local_stiffness
        if self.numerics.discretization == 'nodal_p1':
            indices = torch.arange(len(X), device=X.device)
            self.mass = torch.sparse_coo_tensor(torch.stack((indices, indices)),
                self.mass_diagonal, (len(X), len(X)),
                check_invariants=True).coalesce().to_sparse_csr()
        else:
            self.mass = _assemble(self.cells, local_mass, len(X))
        if (self.mass_diagonal <= 0).any():
            raise ValueError('every EP node must belong to a simplex')
        # Land feedback retains the ORIGINAL consistent L2 projection even
        # when the PDE uses vertex quadrature. Affine P1 integrals are analytic.
        self.projection_local_weights = (volume[:, None]/vertices).expand(-1, vertices)
        if self.numerics.discretization == 'consistent_p1':
            self.projection_mass = self.mass
            self.projection_diagonal = self.mass_diagonal
        else:
            consistent = volume[:, None, None]/(vertices*(vertices+1))*(X.new_ones((vertices, vertices))+torch.eye(
                vertices, dtype=X.dtype, device=X.device))
            self.projection_mass = _assemble(self.cells, consistent, len(X))
            self.projection_diagonal = scatter(consistent.diagonal(dim1=-2, dim2=-1))

    @torch.no_grad()
    def update_deformation(self, coordinates):
        """Accepted material-node coordinates; exact Eq.14 pullback on P1 cells.

        K stays invariant. M and reaction/stimulus integration use current cell
        volumes. Reference Land projection and voltage DOF identities stay fixed.
        Frozen geometry is used throughout the next mechanics interval (staggered).
        """
        if coordinates.shape!=self.X.shape or coordinates.device!=self.X.device or coordinates.dtype!=self.X.dtype:
            raise ValueError('deformed EP coordinates must match reference nodes')
        version=None if torch.is_inference(coordinates) else coordinates._version
        if coordinates is self._deformation_input and version is not None and version==self._deformation_version:
            return False
        F=torch.einsum('eai,eaJ->eiJ',coordinates[self.cells],self.geometry.gradients[:,0])
        J=torch.linalg.det(F)
        if not torch.isfinite(coordinates).all() or not torch.isfinite(J).all() or (J<=0).any():
            raise ValueError('invalid/inverted EP deformation; previous operator retained')
        vertices=self.cells.shape[1];volume=self.reference_volume*J
        scatter=lambda values:self.X.new_zeros(len(self.X)).index_add(0,self.cells.flatten(),values.flatten())
        if self.numerics.discretization=='nodal_p1':
            diagonal=scatter((volume[:,None]/vertices).expand(-1,vertices))
            index=torch.arange(len(self.X),device=self.X.device)
            mass=torch.sparse_coo_tensor(torch.stack((index,index)),diagonal,(len(self.X),len(self.X))).coalesce().to_sparse_csr()
        else:
            local=volume[:,None,None]/(vertices*(vertices+1))*(self.X.new_ones((vertices,vertices))+
                torch.eye(vertices,device=self.X.device,dtype=self.X.dtype))
            diagonal=scatter(local.diagonal(dim1=-2,dim2=-1))
            mass=_assemble(self.cells,local,len(self.X))
        # Invalidate both linear solver and fused RHS graphs before replacing M.
        self.close()
        self.mass,self.mass_diagonal=mass,diagonal
        self.deformation_gradient,self.volume_ratio=F,J
        self._deformation_input,self._deformation_version=coordinates,version
        return True

    def spatial_conductivity(self):
        """Current Gd at the original diffusion quadrature sites, for diagnostics."""
        from .deformation import push_forward_conductivity
        return push_forward_conductivity(self.deformation_gradient,self.reference_conductivity)

    def _system(self, dt_s):
        if dt_s not in self._systems:
            p = self.parameters
            alpha = p.surface_to_volume_per_cm*p.capacitance_uf_per_cm2/dt_s
            matrix = (self.numerics.theta*self.stiffness.to_sparse_coo()+
                      alpha*self.mass.to_sparse_coo()).coalesce().to_sparse_csr()
            backend = None
            if self.numerics.backend == 'amgx_pcg_amg':
                from .amgx import AmgXPCG
                backend = AmgXPCG(matrix, self.solver_options,profile=self.numerics.amg_profile)
            elif self.numerics.backend == 'torch_pcg_jacobi_light':
                from .jacobi import JacobiPCG
                diagonal=alpha*self.mass_diagonal+self.numerics.theta*self.stiffness_diagonal
                backend=JacobiPCG(matrix,diagonal,self.solver_options)
            self._systems[dt_s] = (alpha, matrix, backend)
        return self._systems[dt_s]

    def prepare(self, dt_s):
        """Build a fixed matrix and AMG hierarchy outside the first EP step."""
        if not isfinite(dt_s) or dt_s <= 0:
            raise ValueError('EP dt must be positive seconds')
        self._system(dt_s)

    def close(self):
        for _, _, backend in self._systems.values():
            if backend is not None:
                backend.close()
        self._systems.clear()
        if hasattr(self,'_rhs_buffers'):self._rhs_buffers.clear()

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
        fields=(voltage_v,ionic_ua_per_cm2,stimulus_ua_per_cm2)
        for value in fields:
            if value.shape!=(len(self.X),) or value.dtype!=self.X.dtype or value.device!=self.X.device:
                raise ValueError('EP fields must match mesh shape, dtype and device')
        if not (torch.isfinite(fields[0]).all() & torch.isfinite(fields[1]).all() & torch.isfinite(fields[2]).all()):
            raise ValueError('EP fields must be finite')
        p = self.parameters
        alpha, matrix, backend = self._system(dt_s)
        if self.X.is_cuda:
            from .gpu_kernels import BufferedRHS
            if not hasattr(self,'_rhs_buffers'):self._rhs_buffers={}
            if dt_s not in self._rhs_buffers:
                self._rhs_buffers[dt_s]=BufferedRHS(self.mass,self.mass_diagonal,self.stiffness,
                    alpha,p.surface_to_volume_per_cm,self.numerics.theta,self.numerics.discretization=='nodal_p1')
            rhs=self._rhs_buffers[dt_s](*fields)
        else:
            source = alpha*voltage_v+p.surface_to_volume_per_cm*(stimulus_ua_per_cm2-ionic_ua_per_cm2)
            rhs = self.mass_diagonal*source if self.numerics.discretization == 'nodal_p1' else self._mv(self.mass, source)
            if self.numerics.theta!=1.:
                rhs=rhs-(1-self.numerics.theta)*self._mv(self.stiffness,voltage_v)
        if backend is not None:
            if self.X.is_cuda and self.numerics.backend=='torch_pcg_jacobi_light':
                return backend.solve(rhs,initial=voltage_v,
                    initial_residual=self._rhs_buffers[dt_s].initial_residual)
            return backend.solve(rhs, initial=voltage_v)
        action = lambda v: self._mv(matrix, v)
        return pcg(action, rhs, alpha*self.mass_diagonal+self.numerics.theta*self.stiffness_diagonal,
                   initial=voltage_v, options=self.solver_options)

    @torch.no_grad()
    def project_cell_field(self, cell_value):
        """Consistent L2 projection, for ODE feedback; never a stress average."""
        if (cell_value.shape != (len(self.cells),) or cell_value.device != self.X.device
                or cell_value.dtype != self.X.dtype or not torch.isfinite(cell_value).all()):
            raise ValueError('finite scalar value per cell required')
        local = cell_value[:, None]*self.projection_local_weights
        rhs = self.X.new_zeros(len(self.X)).index_add(0, self.cells.reshape(-1), local.reshape(-1))
        return pcg(lambda v: self._mv(self.projection_mass, v), rhs, self.projection_diagonal,
                   options=self.solver_options)[0]
