"""Stable tensor buffers for compiled active force; buffers are trial scratch.

Accepted physical history belongs to ActiveLoadState, never these buffers.
Publishing a new trial copies tensor values without replacing captured tensor
objects or specializing the compiled force on each ramp factor/time step.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from .land import active_pk1, LandQuadratureLoad


@dataclass(frozen=True)
class ActiveLoadState:
    previous_x: torch.Tensor
    xs: torch.Tensor
    xw: torch.Tensor
    zs_old: torch.Tensor
    zw_old: torch.Tensor
    dt_s: float
    use_land: bool = True
    anchor: torch.Tensor | None = None
    ramp_factor: float = 0.0

    def materialize(self, mesh):
        # Zero mode uses a zero frozen anchor. This is only for checkpoint
        # validation/reference comparisons; the hot loop uses buffered tables.
        anchor = None if self.use_land else self.anchor
        if not self.use_land and anchor is None:
            from ..p1 import prepare_p1
            anchor = torch.zeros_like(prepare_p1(mesh.X,mesh.cells,5).weights)
        return LandQuadratureLoad.prepare(mesh.X,mesh.cells,mesh.fiber,self.previous_x,
            self.xs,self.xw,self.zs_old,self.zw_old,self.dt_s,
            anchor=anchor,ramp_factor=self.ramp_factor)


class BufferedLandLoad:
    def __init__(self, geometry, fiber):
        self.geometry = geometry
        self.fiber = fiber.detach().clone()
        self.previous_F = fiber.new_zeros((len(geometry.cells),3,3))
        self.fields = fiber.new_zeros((*geometry.weights.shape,4))
        self.anchor = torch.zeros_like(geometry.weights)
        self.dt = fiber.new_tensor(1e-4)
        self.use_land = fiber.new_tensor(0.)
        self.ramp = fiber.new_tensor(0.)

    @torch.no_grad()
    def stage(self, state):
        g = self.geometry
        if (not isfinite(state.dt_s) or state.dt_s <= 0 or not isfinite(state.ramp_factor)
                or not 0 <= state.ramp_factor <= 1 or type(state.use_land) is not bool):
            raise ValueError('invalid active trial step or mode')
        for value,shape in ((state.previous_x,(g.node_count,3)),
                            *((v,(g.node_count,)) for v in (state.xs,state.xw,state.zs_old,state.zw_old))):
            if (value.shape != shape or value.dtype != self.fiber.dtype or value.device != self.fiber.device
                    or not torch.isfinite(value).all()):
                raise ValueError('active trial fields must match the solid mesh')
        if state.anchor is not None and (state.anchor.shape != self.anchor.shape
                or state.anchor.dtype != self.fiber.dtype or state.anchor.device != self.fiber.device
                or not torch.isfinite(state.anchor).all() or (state.anchor < 0).any()):
            raise ValueError('nonnegative closure quadrature anchor required')
        if not state.use_land and state.ramp_factor > 0 and state.anchor is None:
            raise ValueError('positive closure ramp requires an accepted anchor')
        nodal = torch.stack((state.xs,state.xw,state.zs_old,state.zw_old),-1)
        self.fields.copy_(torch.einsum('qa,eak->eqk',g.values,nodal[g.cells]))
        self.previous_F.copy_(torch.einsum('eai,eaJ->eiJ',state.previous_x[g.cells],g.gradients[:,0]))
        if state.anchor is None: self.anchor.zero_()
        else: self.anchor.copy_(state.anchor)
        self.dt.fill_(state.dt_s)
        self.use_land.fill_(float(state.use_land))
        self.ramp.fill_(state.ramp_factor)

    def _tension(self, F, fiber, previous_F, fields, anchor):
        # Leading axes may be (cell,) or absent under vmap of element kernels.
        ff = torch.einsum('...ij,...j->...i',F,fiber)
        oldff = torch.einsum('...ij,...j->...i',previous_F,fiber)
        stretch = ff.square().sum(-1).clamp_min(1e-16).sqrt()[...,None]
        previous = oldff.square().sum(-1).clamp_min(1e-16).sqrt()[...,None]
        lc = stretch.clamp_max(1.2)
        h = (1+2.3*(lc+lc.clamp_max(.87)-1.87)).clamp_min(0)
        xs,xw,zs,zw = fields.unbind(-1)
        def update(z,rate):
            a = rate*self.dt
            return z*torch.exp(-a)+10*(stretch-previous)*(-torch.expm1(-a)/a)
        ta = (h*4.8e6*(xs*(update(zs,40.14)+1)+xw*update(zw,405.86))).clamp_min(0)
        return torch.where(self.use_land > .5,ta,anchor*self.ramp)

    def _element_force(self,nodes,gradients,weights,fiber,previous_F,fields,anchor):
        F = nodes.T@gradients
        ta = self._tension(F,fiber,previous_F,fields,anchor)
        P = active_pk1(F[None],fiber[None],ta)
        return -torch.einsum('q,qiJ,aJ->ai',weights,P,gradients)

    def _args(self,x):
        g = self.geometry
        return x[g.cells],g.gradients[:,0],g.weights,self.fiber,self.previous_F,self.fields,self.anchor

    def tension(self,x):
        g = self.geometry
        F = torch.einsum('eai,eaJ->eiJ',x[g.cells],g.gradients[:,0])
        return self._tension(F,self.fiber,self.previous_F,self.fields,self.anchor)

    def force(self,x):
        g = self.geometry
        F = torch.einsum('eai,eaJ->eiJ',x[g.cells],g.gradients[:,0])
        ta = self._tension(F,self.fiber,self.previous_F,self.fields,self.anchor)
        P = active_pk1(F[:,None],self.fiber[:,None],ta)
        local = -torch.einsum('eq,eqiJ,eaJ->eai',g.weights,P,g.gradients[:,0])
        return torch.zeros_like(x).index_add(0,g.cells.reshape(-1),local.reshape(-1,3))

    def force_jvp(self,x,direction):
        return torch.func.jvp(self.force,(x,),(direction,))[1]

    def closure_snapshot(self,x):
        return self.tension(x).detach().clone()
