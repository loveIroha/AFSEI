"""Two straight P2 Neo-Hookean walls for the existing 2D AFSI interface.

An immersed tube is inside the Cartesian fluid box. Exterior background fluid
is retained by the fictitious-domain formulation, and is not counted as lumen
flow. End faces are elastically tethered; no moving-wall displacement is imposed.
"""
from dataclasses import dataclass
from math import isfinite
import numpy as np
import torch
from ..afsi340 import ValveMesh, prepare_triangle, determinant
from ..mac.execution import tensor_kernel


@dataclass(frozen=True)
class TubeGeometry:
    length_cm: float = 2.4
    lumen_height_cm: float = .6
    wall_thickness_cm: float = .1
    inlet_buffer_cm: float = .4
    center_y_cm: float = .8
    axial_elements: int = 48
    thickness_elements: int = 2

    def __post_init__(self):
        if any(not isfinite(v) or v <= 0 for v in (self.length_cm, self.lumen_height_cm,
                self.wall_thickness_cm, self.inlet_buffer_cm, self.center_y_cm)):
            raise ValueError('positive finite tube dimensions required')
        if any(type(v) is not int or v < 1 for v in (self.axial_elements, self.thickness_elements)):
            raise ValueError('positive integer wall subdivisions required')
        if self.center_y_cm <= self.lumen_height_cm/2+self.wall_thickness_cm:
            raise ValueError('lower wall must be inside the fluid box')


def generate_tube(config=None, *, device='cpu'):
    c = config or TubeGeometry()
    nx, ny = c.axial_elements, c.thickness_elements
    x = np.linspace(c.inlet_buffer_cm, c.inlet_buffer_cm+c.length_cm, nx+1)
    y = np.linspace(c.center_y_cm-c.lumen_height_cm/2-c.wall_thickness_cm,
                    c.center_y_cm-c.lumen_height_cm/2, ny+1)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    bottom = np.stack((xx.ravel(), yy.ravel()), -1)
    top = bottom.copy(); top[:, 1] = 2*c.center_y_cm-top[:, 1]
    n = len(bottom)
    tri = []
    for i in range(nx):
        for j in range(ny):
            a = i*(ny+1)+j
            b, d = a+ny+1, a+1
            tri.extend(((a, b, b+1), (a, b+1, d)))
    lower = np.asarray(tri, dtype=np.int64)
    triangles = np.concatenate((lower, lower[:, [0, 2, 1]]+n))
    vertices = np.concatenate((bottom, top))
    pairs = np.sort(triangles[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2), axis=1)
    unique, inverse, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    nv = len(vertices)
    X = np.concatenate((vertices, vertices[unique].mean(1)))
    cells = np.concatenate((triangles, nv+inverse.reshape(-1, 3)), 1)
    end = ((np.abs(vertices[unique, 0]-x[0]) < 1e-12).all(1)
           | (np.abs(vertices[unique, 0]-x[-1]) < 1e-12).all(1)) & (counts == 1)
    roots = np.column_stack((unique[end], nv+np.flatnonzero(end)))
    to = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=device)
    return ValveMesh(to(X, torch.float64), to(cells, torch.int64),
        to(np.repeat([1, 11], len(lower)), torch.int64), to(roots, torch.int64),
        to(np.ones(len(roots)), torch.int64), nv, 'structured/no-gmsh')


def neo_hookean_plane_strain_pk1(F, shear, lame):
    """Plane-strain embedding F33=1, returning the in-plane PK1 stress.

    W=mu/2*(F:F-2)-mu log J+lambda/2*(log J)^2.
    The matching 3D law has F33=1, and therefore exactly this in-plane PK1.
    """
    J = determinant(F)
    cof = torch.stack((F[..., 1, 1], -F[..., 1, 0], -F[..., 0, 1], F[..., 0, 0]), -1).reshape_as(F)
    invT = cof/J[..., None, None]
    return shear*(F-invT)+lame*J.log()[..., None, None]*invT


class NeoHookeanTube:
    def __init__(self, mesh, *, shear_modulus=1e5, lame_lambda=2e5, end_spring=1e6, fused=True):
        if any(not isfinite(v) or v <= 0 for v in (shear_modulus, lame_lambda, end_spring)):
            raise ValueError('positive finite material and tether coefficients required')
        self.mesh, self.geometry = mesh, prepare_triangle(mesh)
        self.shear, self.lame, self.end_spring = shear_modulus, lame_lambda, end_spring
        self.external_force = torch.zeros_like(mesh.X)
        z, w = np.polynomial.legendre.leggauss(3)
        s = mesh.X.new_tensor((z+1)/2)
        self.root_values = torch.stack(((1-s)*(1-2*s), s*(2*s-1), 4*s*(1-s)), -1)
        length = torch.linalg.vector_norm(mesh.X[mesh.roots[:, 1]]-mesh.X[mesh.roots[:, 0]], dim=-1)
        self.root_weights = length[:, None]*mesh.X.new_tensor(w/2)
        self._kernel = tensor_kernel(self._force_and_det, mesh.X.device) if fused else self._force_and_det

    def _force_and_det(self, x):
        g = self.geometry
        F = torch.einsum('eai,eqaJ->eqiJ', x[g.cells], g.gradients)
        P = neo_hookean_plane_strain_pk1(F, self.shear, self.lame)
        local = -torch.einsum('eq,eqiJ,eqaJ->eai', g.weights, P, g.gradients)
        force = torch.zeros_like(x).index_add(0, g.cells.reshape(-1), local.reshape(-1, 2))
        displacement = x[self.mesh.roots]-self.mesh.X[self.mesh.roots]
        u = torch.einsum('qa,bai->bqi', self.root_values, displacement)
        spring = -self.end_spring*torch.einsum('bq,qa,bqi->bai', self.root_weights, self.root_values, u)
        force = force.index_add(0, self.mesh.roots.reshape(-1), spring.reshape(-1, 2))
        return force+self.external_force, determinant(F)

    def force_and_det(self, x):
        return self._kernel(x)

    def force(self, x):
        return self._kernel(x)[0]

    def validate(self, x):
        if x.shape != self.mesh.X.shape or x.dtype != self.mesh.X.dtype or x.device != self.mesh.X.device:
            raise ValueError('wall coordinates must match the reference mesh')
        g = self.geometry
        J = determinant(torch.einsum('eai,eqaJ->eqiJ', x[g.cells], g.gradients))
        if not torch.isfinite(J).all() or (J <= 0).any():
            raise ValueError('nonpositive or nonfinite wall det(F)')
