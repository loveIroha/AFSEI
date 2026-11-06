"""Generated full circular tube: conforming, periodic P1 tetrahedral wall.

All dimensions are cm. The tube axis is z. There is no fluid tetrahedral mesh:
this mesh is used only by solid mechanics and electrophysiology.
"""
from dataclasses import dataclass
from itertools import permutations
from math import isfinite
import numpy as np
import torch


@dataclass(frozen=True)
class CircularTubeGeometry:
    length_cm: float = 4.
    inner_radius_cm: float = 1.
    wall_thickness_cm: float = .2
    center_xy_cm: tuple = (2., 2.)
    axial_start_cm: float = .5
    axial_elements: int = 32
    circumferential_elements: int = 64
    thickness_elements: int = 2
    radial_interfaces_cm: tuple = ()
    radial_elements_per_layer: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'center_xy_cm', tuple(self.center_xy_cm))
        object.__setattr__(self, 'radial_interfaces_cm', tuple(self.radial_interfaces_cm))
        object.__setattr__(self, 'radial_elements_per_layer', tuple(self.radial_elements_per_layer))
        for value in (self.length_cm, self.inner_radius_cm, self.wall_thickness_cm):
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError('positive finite tube dimensions in cm required')
        if (len(self.center_xy_cm) != 2 or
                any(not isfinite(v) for v in (*self.center_xy_cm, self.axial_start_cm))):
            raise ValueError('finite tube center and axial start required')
        for name, minimum in (('axial_elements', 2), ('circumferential_elements', 8),
                              ('thickness_elements', 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.radial_interfaces_cm or self.radial_elements_per_layer:
            radii = (self.inner_radius_cm,*self.radial_interfaces_cm,self.inner_radius_cm+self.wall_thickness_cm)
            if (len(self.radial_elements_per_layer) != len(radii)-1 or
                    any(not isfinite(v) for v in radii) or
                    any(b <= a for a,b in zip(radii[:-1],radii[1:])) or
                    any(type(n) is not int or n < 1 for n in self.radial_elements_per_layer) or
                    sum(self.radial_elements_per_layer) != self.thickness_elements):
                raise ValueError('explicit radial layers must be ordered, resolved, and sum to thickness_elements')


@dataclass(frozen=True)
class CircularTubeMesh:
    X: torch.Tensor
    cells: torch.Tensor
    fiber: torch.Tensor             # DG0 circumferential mechanical fiber
    sheet: torch.Tensor             # DG0 radial direction (visualization)
    inner_faces: torch.Tensor       # oriented OUT of the lumen
    end_faces: torch.Tensor         # annular wall end faces
    inner_rings: torch.Tensor       # (axial section, circumferential node)
    cell_layer: torch.Tensor | None = None  # ascending radius; zero for a single layer


def generate_circular_tube(config, *, device='cpu', dtype=torch.float64):
    """Split every logical annular hex into six matching Freudenthal tets.

    The angular seam shares node IDs; there are no duplicate seam nodes.
    Surface orientation is obtained from the positive tetrahedral orientation.
    CPU construction is setup work only; the resulting FE arrays live on device.
    """
    c = config
    na, nr, nt = c.axial_elements, c.thickness_elements, c.circumferential_elements
    ids = np.arange((na+1)*(nr+1)*nt).reshape(na+1, nr+1, nt)
    if c.radial_elements_per_layer:
        bounds=(c.inner_radius_cm,*c.radial_interfaces_cm,c.inner_radius_cm+c.wall_thickness_cm)
        radii=np.concatenate([np.linspace(a,b,n+1)[:-1] for a,b,n in
            zip(bounds[:-1],bounds[1:],c.radial_elements_per_layer)]+[np.array([bounds[-1]])])
    else:
        radii=np.linspace(c.inner_radius_cm,c.inner_radius_cm+c.wall_thickness_cm,nr+1)
    z, r, theta = np.meshgrid(
        np.linspace(c.axial_start_cm, c.axial_start_cm+c.length_cm, na+1),
        radii,
        2*np.pi*np.arange(nt)/nt, indexing='ij')
    X = np.stack((c.center_xy_cm[0]+r*np.cos(theta),
                  c.center_xy_cm[1]+r*np.sin(theta), z), -1).reshape(-1, 3)
    a, b, k = np.meshgrid(np.arange(na), np.arange(nr), np.arange(nt), indexing='ij')
    cells = []
    for order in permutations(range(3)):
        corners = [np.zeros(3, dtype=int)]
        for axis in order:
            corner = corners[-1].copy(); corner[axis] += 1
            corners.append(corner)
        cells.append(np.stack([ids[a+da, b+db, (k+dk) % nt]
                               for da, db, dk in corners], -1).reshape(-1, 4))
    cells = np.concatenate(cells)
    signed = np.linalg.det((X[cells[:, 1:]]-X[cells[:, :1]]).transpose(0, 2, 1))
    if np.any(np.abs(signed) < np.finfo(float).eps*c.length_cm*c.inner_radius_cm*c.wall_thickness_cm):
        raise ValueError('generated circular wall contains a degenerate tetrahedron')
    negative = signed < 0
    cells[negative, :2] = cells[negative, 1::-1]
    oriented = cells[:, [[1,2,3], [0,3,2], [0,1,3], [0,2,1]]].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(oriented, axis=1), axis=0,
                                 return_index=True, return_counts=True)
    if (counts > 2).any():
        raise ValueError('nonmanifold generated circular wall')
    surface = oriented[first[counts == 1]]
    radial_index = (surface // nt) % (nr+1)
    axial_index = surface // ((nr+1)*nt)
    inner = surface[(radial_index == 0).all(1)][:, [0,2,1]]
    ends = surface[((axial_index == 0).all(1)) | ((axial_index == na).all(1))]
    outer_count = (radial_index == nr).all(1).sum()
    if len(inner) != 2*na*nt or len(ends) != 4*nr*nt or len(surface) != len(inner)+len(ends)+outer_count:
        raise ValueError('unexpected boundary at periodic tube seam')
    centers = X[cells].mean(1)
    radial = centers[:, :2]-np.asarray(c.center_xy_cm)
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    sheet = np.column_stack((radial, np.zeros(len(radial))))
    fiber = np.column_stack((-radial[:, 1], radial[:, 0], np.zeros(len(radial))))
    floating = lambda v: torch.as_tensor(v, dtype=dtype, device=device)
    integer = lambda v: torch.as_tensor(v.copy(), dtype=torch.int64, device=device)
    layer=np.searchsorted(np.cumsum(c.radial_elements_per_layer or (nr,)),
        ((cells//nt)%(nr+1)).min(1),side='right')
    return CircularTubeMesh(floating(X), integer(cells), floating(fiber), floating(sheet),
                            integer(inner), integer(ends), integer(ids[:, 0]),integer(layer))


def section_areas(x, mesh):
    """Signed x-y projected lumen polygon areas; a diagnostic, not a constraint."""
    p = x[mesh.inner_rings, :2]
    # Center first to avoid cancellation when the tube translates.
    p = p-p.mean(1, keepdim=True)
    q = p.roll(-1, dims=1)
    return .5*(p[..., 0]*q[..., 1]-p[..., 1]*q[..., 0]).sum(1)


def lumen_volume(x, mesh):
    """Volume enclosed by the deformed inner surface and diagnostic end caps.

    The caps close the volume integral only; they exert no force on the fluid.
    """
    # Use an interior origin to reduce cancellation in the surface formula.
    origin = x[mesh.inner_rings].mean((0, 1))
    triangles = x[mesh.inner_faces]-origin
    volume6 = (triangles[:, 0]*torch.linalg.cross(triangles[:, 1], triangles[:, 2])).sum()
    for ring, sign in ((mesh.inner_rings[0], -1.), (mesh.inner_rings[-1], 1.)):
        p = x[ring]-origin
        center = p.mean(0)
        volume6 = volume6+sign*(center*torch.linalg.cross(p, p.roll(-1, dims=0))).sum()
    return volume6/6
