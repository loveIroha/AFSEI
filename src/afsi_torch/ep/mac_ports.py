"""Material basal cap and work-conjugate port operators on the original MAC box.

These operators do not seal the cap or impose valve/RCR conditions. They are
prepared from a geometry snapshot without advancing or modifying any state.
The cap is a boundary integration surface, never a fitted blood-volume mesh.
Region order: non-port cap, idealized MV, idealized AV. Flux is outward.
"""
from dataclasses import dataclass
from math import isfinite, pi, prod
import torch
from ..ib import peskin4
from ..p1 import cavity_rim
from ..circulation.geometry import PortGeometry, _check_disc


def _stencil(grid, points):
    """Same unclipped four-point Peskin kernel as the original MAC FE transfer."""
    h = points.new_tensor(grid.spacing)
    scaled_box = (points-points.new_tensor(grid.origin))/h
    if (not torch.isfinite(points).all() or (scaled_box < 2).any()
            or (scaled_box >= points.new_tensor(grid.shape)-2).any()):
        raise ValueError('MAC surface support reaches the box boundary; enlarge/refine the box')
    offsets = torch.arange(4, device=points.device)
    result = []
    for c in range(3):
        scaled = (points-points.new_tensor(grid.face_origin(c)))/h
        nodes = torch.floor(scaled-1).to(torch.int64)[:, :, None]+offsets
        shape = grid.face_shape(c)
        phi = peskin4(scaled[:, :, None]-nodes.to(points.dtype))
        ids = ((nodes[:, 0, :, None, None]*shape[1]+nodes[:, 1, None, :, None])
               *shape[2]+nodes[:, 2, None, None, :]).reshape(-1, 64)
        weights = ((phi[:, 0, :, None, None]*phi[:, 1, None, :, None])
                   *phi[:, 2, None, None, :]).reshape(-1, 64)
        result.append((ids, weights))
    return result


@dataclass(frozen=True)
class MACFluxRows:
    grid: object
    components: tuple
    like: torch.Tensor

    def flow(self, velocity):
        """Three oriented absolute surface fluxes, cm³/s = ml/s."""
        self.grid.check_velocity(velocity)
        if any(u.dtype != self.like.dtype or u.device != self.like.device for u in velocity):
            raise ValueError('surface and MAC velocity must share dtype/device')
        result = self.like.new_zeros(3)
        for u, (region, index, coefficient) in zip(velocity, self.components):
            result = result.index_add(0, region, coefficient*u.reshape(-1)[index])
        return result

    def check_pressure(self, pressure_cgs):
        if (pressure_cgs.shape != (3,) or pressure_cgs.dtype != self.like.dtype
                or pressure_cgs.device != self.like.device or not torch.isfinite(pressure_cgs).all()):
            raise ValueError('three finite surface pressures with matching dtype/device required')

    def pressure_density(self, pressure_cgs):
        """Fluid force density -B^T p / h³; no second surface weight."""
        self.check_pressure(pressure_cgs)
        result = []
        for c, (region, index, coefficient) in enumerate(self.components):
            out = self.like.new_zeros(self.grid.face_shape(c))
            out.reshape(-1).index_add_(0, index, -coefficient*pressure_cgs[region]/self.grid.volume)
            result.append(out)
        return tuple(result)


@torch.no_grad()
def prepare_flux_rows(grid, points, area_vectors, labels):
    """Integrate and coalesce the point transfer into three compact flux rows."""
    if (points.ndim != 2 or points.shape[1] != 3 or area_vectors.shape != points.shape
            or labels.shape != (len(points),) or labels.dtype != torch.int64
            or labels.device != points.device or area_vectors.device != points.device
            or area_vectors.dtype != points.dtype or points.dtype != torch.float64
            or not torch.isfinite(area_vectors).all() or (labels < 0).any() or (labels > 2).any()):
        raise ValueError('finite float64 surface points/areas and labels 0..2 required')
    rows = labels[:, None].expand(-1, 64).reshape(-1)
    components = []
    for c, (ids, weights) in enumerate(_stencil(grid, points)):
        values = (weights*area_vectors[:, c, None]).reshape(-1)
        # Surface quadrature can be finer than MAC resolution; coalesce it once.
        sparse = torch.sparse_coo_tensor(torch.stack((rows, ids.reshape(-1))), values,
            (3, prod(grid.face_shape(c))), device=points.device, check_invariants=False).coalesce()
        indices, coefficients = sparse.indices(), sparse.values()
        nonzero = coefficients != 0
        components.append((indices[0, nonzero], indices[1, nonzero], coefficients[nonzero]))
    return MACFluxRows(grid, tuple(components), points)


def _subdivision(level, like):
    triangles = torch.eye(3, device=like.device, dtype=like.dtype)[None]
    for _ in range(level):
        a, b, c = triangles.unbind(1)
        ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
        triangles = torch.stack((torch.stack((a, ab, ca), 1), torch.stack((ab, b, bc), 1),
            torch.stack((ca, bc, c), 1), torch.stack((ab, bc, ca), 1)), 1).reshape(-1, 3, 3)
    return triangles


class MACBasalPorts:
    """Reference-labelled cap transported by current rim nodes and their mean.

    A planar, star-shaped reference rim is required. Current caps may warp;
    no node is flattened or snapped to the Cartesian grid. Circle boundaries
    are approximated by subtriangle labels, with explicit area error control.
    """
    @torch.no_grad()
    def __init__(self, reference_x, endocardial_faces, grid, *, ports=None,
                 subdivision_level=4, area_rtol=.03, with_ports=True):
        if (reference_x.ndim != 2 or reference_x.shape[1] != 3 or reference_x.dtype != torch.float64
                or not torch.isfinite(reference_x).all() or endocardial_faces.ndim != 2
                or endocardial_faces.shape[1] != 3 or endocardial_faces.dtype != torch.int64
                or endocardial_faces.device != reference_x.device
                or type(subdivision_level) is not int or not 0 <= subdivision_level <= 6
                or not isfinite(area_rtol) or not 0 < area_rtol <= .10
                or type(with_ports) is not bool):
            raise ValueError('float64 P1 surface, subdivision level 0..6 and area tolerance in (0,.1] required')
        self.X = reference_x.detach().clone()
        self.endo = endocardial_faces.detach().clone()
        self.grid = grid
        self.ports = ports or PortGeometry()
        self.with_ports = with_ports
        self.rim = cavity_rim(self.endo)
        rim_x = self.X[self.rim[:, 0]]
        self.center = rim_x.mean(0)
        _, _, basis = torch.linalg.svd(rim_x-self.center)
        normal = basis[-1]
        if (normal*(self.center-self.X[torch.unique(self.endo)].mean(0))).sum() < 0:
            normal = -normal
        axis = self.X.new_tensor([1., 0., 0.])
        if torch.linalg.vector_norm(axis-(axis*normal).sum()*normal) < 1e-6:
            axis = self.X.new_tensor([0., 1., 0.])
        first = axis-(axis*normal).sum()*normal
        first /= torch.linalg.vector_norm(first)
        self.frame = torch.stack((first, torch.linalg.cross(normal, first), normal))
        local_rim = (rim_x-self.center)@self.frame.T
        plane_error = local_rim[:, 2].abs().max().item()
        if plane_error > 1e-5:
            raise ValueError('reference basal rim must be planar; no silent flattening')
        following = dict(self.rim.cpu().tolist()); current = int(self.rim[0, 0]); ordered = []
        for _ in range(len(self.rim)):
            ordered.append(current); current = following[current]
        polygon = (self.X[torch.tensor(ordered, device=self.X.device)]-self.center)@self.frame.T
        for center, radius in (() if not with_ports else ((self.ports.mv_center_cm, self.ports.mv_radius_cm),
                               (self.ports.av_center_cm, self.ports.av_radius_cm))):
            _check_disc(polygon[:, :2], center, radius)
        self.subtriangles = _subdivision(subdivision_level, self.X)
        q = len(self.subtriangles)
        self.parent = torch.arange(len(self.rim), device=self.X.device).repeat_interleave(q)
        self.barycentric = self.subtriangles.mean(1).repeat(len(self.rim), 1)
        points, areas = self.surface(self.X)
        if ((areas@normal) <= 0).any():
            raise ValueError('reference basal fan must be star-shaped with outward orientation')
        local = (points-self.center)@self.frame.T
        self.labels = torch.zeros(len(points), device=self.X.device, dtype=torch.int64)
        for label, center, radius in (() if not with_ports else ((1, self.ports.mv_center_cm, self.ports.mv_radius_cm),
                                     (2, self.ports.av_center_cm, self.ports.av_radius_cm))):
            disc = (local[:, :2]-local.new_tensor(center)).square().sum(-1) <= radius**2
            self.labels[disc] = label
        region_area = self.region_sum(torch.linalg.vector_norm(areas, dim=-1))
        nominal = (self.X.new_tensor([pi*self.ports.mv_radius_cm**2, pi*self.ports.av_radius_cm**2])
                   if with_ports else self.X.new_zeros(2))
        error = (region_area[1:]-nominal)/nominal if with_ports else self.X.new_zeros(2)
        if with_ports and ((error.abs() > area_rtol).any() or (region_area <= 0).any()):
            raise ValueError('basal port area is underresolved; increase surface subdivision level')
        self.metadata = dict(anatomical_valves_present=False,
            geometry='material mean-rim triangular cap on the original Cartesian MAC box',
            reference_planar_rim_error_cm=plane_error, rim_nodes=len(self.rim),
            cap_triangles=len(self.rim), subdivision_level=subdivision_level,
            surface_quadrature_points=len(points), reference_region_area_cm2=region_area.cpu().tolist(),
            nominal_reference_port_area_cm2=nominal.cpu().tolist(),
            relative_reference_port_area_error=error.cpu().tolist(),
            idealized_port_discs_present=with_ports,
            label_order=['non_port_cap', 'mv', 'av'],
            ports_imposed=False, cap_sealed=False, windkessel_ports_connected=False)

    def region_sum(self, value):
        return value.new_zeros((3, *value.shape[1:])).index_add(0, self.labels, value)

    def material_points(self, nodal_value):
        if (nodal_value.shape != self.X.shape or nodal_value.device != self.X.device
                or nodal_value.dtype != self.X.dtype):
            raise ValueError('material nodal field must match the cap reference mesh')
        endpoints = nodal_value[self.rim[self.parent]]
        center = nodal_value[self.rim[:, 0]].mean(0)
        b = self.barycentric
        return b[:, :1]*endpoints[:, 0]+b[:, 1:2]*endpoints[:, 1]+b[:, 2:]*center

    def surface(self, x):
        endpoints = x[self.rim]
        center = x[self.rim[:, 0]].mean(0)
        a, b = endpoints[:, 0], endpoints[:, 1]
        area = .5*torch.linalg.cross(b-a, center-a)
        return self.material_points(x), area[self.parent]/len(self.subtriangles)

    @torch.no_grad()
    def prepare(self, x):
        points, areas = self.surface(x)
        normal = areas.sum(0)
        if (not torch.isfinite(areas).all() or (areas@normal <= 0).any()):
            raise ValueError('current basal cap is folded or degenerate')
        return PreparedBasalPorts(self, points, areas, prepare_flux_rows(self.grid, points, areas, self.labels))


@dataclass(frozen=True)
class PreparedBasalPorts:
    cap: MACBasalPorts
    points: torch.Tensor
    area_vectors: torch.Tensor
    rows: MACFluxRows

    def material_flow(self, nodal_velocity):
        return self.cap.region_sum((self.cap.material_points(nodal_velocity)*self.area_vectors).sum(-1))

    def relative_flow(self, velocity, nodal_velocity):
        return self.rows.flow(velocity)-self.material_flow(nodal_velocity)

    def material_reaction(self, pressure_cgs):
        """C^T p: algebraic moving-cap reaction paired with relative port flux.

        This optional virtual-work operator is NOT an imposed anatomical valve
        load. Its use in the future boundary model must be explicit.
        """
        self.rows.check_pressure(pressure_cgs)
        cap = self.cap
        force = self.area_vectors*pressure_cgs[cap.labels, None]
        b = cap.barycentric
        out = cap.X.new_zeros(cap.X.shape)
        out.index_add_(0, cap.rim[cap.parent, 0], force*b[:, :1])
        out.index_add_(0, cap.rim[cap.parent, 1], force*b[:, 1:2])
        central = (force*b[:, 2:]).sum(0)/len(cap.rim)
        out.index_add_(0, cap.rim[:, 0], central.expand(len(cap.rim), -1))
        return out
