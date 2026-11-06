"""Read-only geometric/flow budgets for sealed MAC tube demonstrations.

The fixed-end control surface consists of the actual FE inner wall, virtual
connectors to the reference end rings, and fixed caps in the septum planes.
Connectors are integration surfaces only: they impose no force or constraint.
They need not be material/impermeable surfaces. Consequently a fixed-control
volume/port imbalance is NOT automatically an IB wall leak.

Each accepted step accumulates end-face flux on device using backward Euler,
as does the RC update. Surface-velocity diagnostics run only at output samples.
The RT0 reconstruction used there is independent of regularized-delta IB;
two spatial quadrature refinements estimate its integration sensitivity.
"""
import numpy as np
import torch


def rt0_velocity(grid, velocity, points):
    """Lowest-order rectangular H(div) reconstruction of MAC normal fluxes.

    Within each MAC cell, component c is linear only in coordinate c. Its
    divergence equals the MAC cell divergence. Tangential jumps mean surface
    quadrature is not exact unless intersections with cell faces are resolved.
    All diagnostic surfaces must remain strictly inside the MAC box.
    """
    dim = points.shape[-1]
    origin = points.new_tensor(getattr(grid, 'origin', (0.,)*dim))
    q = (points-origin)/points.new_tensor(grid.spacing)
    ij = q.floor().long()
    # Guard only when constructing geometry/tests; runtime acceptance already
    # keeps tube support inside the box. No device-to-host read here.
    ij = torch.stack([ij[:, c].clamp(0, grid.shape[c]-1) for c in range(dim)], -1)
    frac = q-ij
    values = []
    for c in range(dim):
        left = tuple(ij[:, a] for a in range(dim))
        right = tuple(ij[:, a]+int(a == c) for a in range(dim))
        values.append((1-frac[:, c])*velocity[c][left]+frac[:, c]*velocity[c][right])
    return torch.stack(values, -1)


class TriangleSurface:
    def __init__(self, indices, like):
        self.indices = indices
        self.origin = like.mean(0)
        self.weights = like.new_tensor(((2/3,1/6,1/6),(1/6,2/3,1/6),(1/6,1/6,2/3)))

    def area_vectors(self, x):
        p = x[self.indices]
        return .5*torch.linalg.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])

    def volume(self, x):
        p = x[self.indices]-self.origin
        return (p.mean(1)*self.area_vectors(x)).sum()/3

    def swept(self, old, new):
        displacement = (new[self.indices]-old[self.indices]).mean(1)
        area = (self.area_vectors(old)+4*self.area_vectors((old+new)/2)+self.area_vectors(new))/6
        return (displacement*area).sum()

    def flux(self, x, velocity, grid, level):
        # Barycentric subdivision avoids rebuilding mesh connectivity on GPU.
        triangles = x[self.indices]
        for _ in range(level):
            a,b,c = triangles.unbind(1)
            ab,bc,ca = (a+b)/2,(b+c)/2,(c+a)/2
            triangles = torch.cat([torch.stack(v,1) for v in
                ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))])
        points = torch.einsum('qa,eai->eqi', self.weights, triangles)
        u = rt0_velocity(grid, velocity, points.reshape(-1,3)).reshape_as(points)
        area = .5*torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        return (u.mean(1)*area).sum()


class CurveSurface:
    def __init__(self, indices, like):
        # Node order is start, midpoint, end.
        self.indices = indices
        self.origin = like.mean(0)
        self.rules = {level:self.rule(like,level) for level in (0,1,2)}

    @staticmethod
    def rule(like, level):
        g,w = np.polynomial.legendre.leggauss(3)
        n = 2**level
        s = like.new_tensor(((np.arange(n)[:,None]+(g+1)/2)/n).reshape(-1))
        w = like.new_tensor(np.tile(w/(2*n),n))
        N = torch.stack(((1-s)*(1-2*s),4*s*(1-s),s*(2*s-1)), -1)
        dN = torch.stack((4*s-3,4-8*s,4*s-1), -1)
        return N,dN,w

    def geometry(self, x, level=0):
        N,dN,w = self.rules[level]
        p = x[self.indices]-self.origin
        point = torch.einsum('qa,eai->eqi',N,p)+self.origin
        tangent = torch.einsum('qa,eai->eqi',dN,p)
        normal = torch.stack((tangent[...,1],-tangent[...,0]),-1)*w[None,:,None]
        return point,normal

    def volume(self, x):
        point,normal = self.geometry(x)
        return ((point-self.origin)*normal).sum()/2

    def swept(self, old, new):
        a,an = self.geometry(old)
        b,bn = self.geometry(new)
        return ((b-a)*(an+bn)/2).sum()

    def flux(self, x, velocity, grid, level):
        point,normal = self.geometry(x,level)
        u = rt0_velocity(grid,velocity,point.reshape(-1,2)).reshape_as(point)
        return (u*normal).sum()


class TubeBudget:
    """No mutation of solver arrays; no host transfers in accepted()."""
    def __init__(self, mesh, domain, *, lower=None, upper=None, initial_x=None):
        self.mesh,self.domain,self.grid = mesh,domain,domain.grid
        self.depth = domain.depth
        X = mesh.X
        self.dimension = X.shape[1]
        if self.dimension == 3:
            lo,hi = mesh.inner_rings[0],mesh.inner_rings[-1]
            nt,n = len(lo),len(X)
            fixed_lo = torch.arange(n,n+nt,device=X.device)
            fixed_hi = fixed_lo+nt
            self.fixed = torch.cat((X[lo],X[hi],X[lo].mean(0,keepdim=True),X[hi].mean(0,keepdim=True)))
            def quad(a,b):
                return torch.cat((torch.stack((a,a.roll(-1),b.roll(-1)),1),
                                  torch.stack((a,b.roll(-1),b),1)))
            connectors = torch.cat((quad(fixed_lo,lo),quad(hi,fixed_hi)))
            def cap(ring,center,positive):
                c = torch.full_like(ring,center)
                return torch.stack((c,ring,ring.roll(-1)),1) if positive else torch.stack((c,ring.roll(-1),ring),1)
            fixed_caps = torch.cat((cap(fixed_lo,n+2*nt,False),cap(fixed_hi,n+2*nt+1,True)))
            # Moving diagnostic cap centers follow the ring mean linearly.
            self.moving_cap_rings = (lo,hi)
            moving_caps = torch.cat((cap(lo,n+2*nt+2,False),cap(hi,n+2*nt+3,True)))
            maker = TriangleSurface
            wall = mesh.inner_faces
        else:
            if lower is None or upper is None:
                raise ValueError('P2 planar budget requires ordered lower/upper inner vertices')
            lo,hi = lower.cpu().tolist(),upper.cpu().tolist()
            # Setup-only CPU connectivity lookup, respecting the actual P2 ordering.
            edge = {}
            for c in mesh.cells.cpu().tolist():
                for (i,j),m in zip(((0,1),(0,2),(1,2)),c[3:]):
                    edge[tuple(sorted((c[i],c[j])))]=m
            wall = []
            for chain in (lo,hi[::-1]):
                wall.extend((a,edge[tuple(sorted((a,b)))],b) for a,b in zip(chain[:-1],chain[1:]))
            n = len(X)
            ids = (lo[0],lo[-1],hi[0],hi[-1])
            self.ends = ids
            # Fixed reference end nodes, then connector midpoints, cap midpoints.
            self.fixed = X[list(ids)].clone()
            connectors = [(lo[-1],n+4,n+1),(n+3,n+5,hi[-1]),
                          (hi[0],n+6,n+2),(n,n+7,lo[0])]
            fixed_caps = [(n+1,n+8,n+3),(n+2,n+9,n)]
            moving_caps = [(lo[-1],n+10,hi[-1]),(hi[0],n+11,lo[0])]
            maker = CurveSurface
        integer = lambda v: torch.as_tensor(v,device=X.device,dtype=torch.long)
        self.wall = maker(integer(wall),X)
        self.connectors = maker(integer(connectors),X)
        self.fixed_caps = maker(integer(fixed_caps),X)
        self.moving_caps = maker(integer(moving_caps),X)
        # A checkpoint window can start at a deformed configuration while the
        # fixed septum rings remain at their original reference positions.
        self.reference = self.extend(X if initial_x is None else initial_x)
        self.initial_moving_volume = self.moving_volume(self.reference)
        self.initial_fixed_volume = self.fixed_volume(self.reference)
        self.cumulative = X.new_zeros(5)  # Qin, Qout, moving-cap sweep, wall sweep, connector sweep
        self.last = None
        self.steps = 0

    def extend(self, x):
        if self.dimension == 3:
            lo,hi = self.moving_cap_rings
            return torch.cat((x,self.fixed,x[lo].mean(0,keepdim=True),x[hi].mean(0,keepdim=True)))
        a,b,c,d = self.ends
        p = self.fixed
        mid = torch.stack(((x[b]+p[1])/2,(p[3]+x[d])/2,(x[c]+p[2])/2,(p[0]+x[a])/2,
                           (p[1]+p[3])/2,(p[2]+p[0])/2,(x[b]+x[d])/2,(x[c]+x[a])/2))
        return torch.cat((x,p,mid))

    def moving_volume(self,x):
        return (self.wall.volume(x)+self.moving_caps.volume(x))*self.depth

    def fixed_volume(self,x):
        return (self.wall.volume(x)+self.connectors.volume(x)+self.fixed_caps.volume(x))*self.depth

    @torch.no_grad()
    def accepted(self, old_x, new_x, velocity, dt):
        old,new = self.extend(old_x),self.extend(new_x)
        qi = self.domain.flux(velocity,self.domain.planes[0])
        qo = self.domain.flux(velocity,self.domain.planes[1])
        sweeps = [s.swept(old,new)*self.depth for s in (self.moving_caps,self.wall,self.connectors)]
        self.cumulative.add_(torch.stack((dt*qi,dt*qo,*sweeps)))
        self.last = (old,new,torch.stack((qi,qo,*[s/dt for s in sweeps])),dt)
        self.steps += 1

    def _values(self):
        old,new,rates,dt = self.last if self.last is not None else (self.reference,self.reference,None,None)
        vm,vf = self.moving_volume(new),self.fixed_volume(new)
        qi,qo,cap,wall,conn = self.cumulative.unbind()
        result = dict(budget_moving_lumen_volume_ml=vm,budget_fixed_end_volume_ml=vf,
            budget_initial_moving_volume_ml=self.initial_moving_volume,
            budget_integrated_inlet_ml=qi,budget_integrated_outlet_ml=qo,
            budget_integrated_net_inflow_ml=qi-qo,budget_moving_cap_sweep_ml=cap,
            budget_wall_sweep_ml=wall,budget_connector_sweep_ml=conn,
            budget_fixed_control_imbalance_ml=vf-self.initial_fixed_volume-(qi-qo),
            budget_geometric_identity_error_ml=vm-self.initial_moving_volume-wall-cap,
            budget_fixed_geometric_identity_error_ml=vf-self.initial_fixed_volume-wall-conn)
        return result

    @torch.no_grad()
    def diagnostics(self,velocity=None):
        result = self._values()
        if velocity is not None and self.last is not None:
            old,new,rates,dt = self.last
            # Both existing steppers freeze IB interpolation geometry at x_n.
            fluxes = [[s.flux(old,velocity,self.grid,level)*self.depth
                       for s in (self.wall,self.connectors)] for level in (1,2)]
            wall,conn = fluxes[1]
            result.update(budget_wall_rt0_flux_ml_s=wall,budget_connector_rt0_flux_ml_s=conn,
                budget_surface_quadrature_difference_ml_s=sum((a-b).abs() for a,b in zip(*fluxes)),
                budget_surface_divergence_defect_ml_s=wall+conn+rates[1]-rates[0],
                budget_wall_motion_minus_rt0_ml_s=rates[3]-wall,
                budget_connector_motion_minus_rt0_ml_s=rates[4]-conn,
                budget_fixed_control_increment_imbalance_ml_s=(self.fixed_volume(new)-self.fixed_volume(old))/dt-rates[0]+rates[1])
        keys=list(result)
        values=torch.stack([result[k] for k in keys]).cpu().tolist()
        return dict(zip(keys,values),budget_accepted_steps=self.steps)

    def checkpoint(self):
        return dict(tube_budget_cumulative_ml=self.cumulative.cpu().numpy(),tube_budget_steps=self.steps)

    def summary(self):
        return dict(enabled=True,accepted_steps=self.steps,
            time_quadrature='accepted-step backward Euler; same as RC',
            geometry='exact P2 curve or P1 triangle volume and swept-surface formulas',
            fixed_control='actual inner wall + virtual connectors + reference caps at fixed septa',
            virtual_connectors_apply_loads=False,
            surface_diagnostic='RT0 MAC reconstruction on old geometry; two quadrature refinements; sampled only',
            imbalance_is_proven_wall_leak=False)
