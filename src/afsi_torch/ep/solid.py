"""EP-driven active load adapter for the existing real-LV BE/IB solid API.

No prescribed waveform is read. Fluid pressure acts via the fluid/solid
interface; the adapter applies zero additional endocardial pressure traction.
A complete driver must still define physical ports, preload and accept/retry
state publication. Existing demo configuration/checkpoint formats do not
serialize these extra EP/Land fields yet.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from ..paper_lv import PaperLVSolid, PaperHOTangent


@dataclass(frozen=True)
class InterfaceLoads:
    def at(self, time):
        if not isfinite(time) or time < 0:
            raise ValueError('nonnegative finite force time required')
        return 0., 0.


class EPDrivenPaperSolid(PaperLVSolid):
    """Prepared per-trial solid with fixed EP states and differentiable Land Ta."""
    def __init__(self, mesh, config, active_load):
        super().__init__(mesh, config)
        g = active_load.geometry
        if (g.node_count != len(mesh.X) or g.cells.device != mesh.X.device
                or not torch.equal(g.cells, mesh.cells)
                or not torch.equal(g.gradients[:,0], self.gradients)
                or not torch.equal(active_load.fiber, mesh.fiber)):
            raise ValueError('active load must use the identical solid mesh, gradients and fibers')
        if active_load.geometry.weights.shape != self.geometry.weights.shape:
            raise ValueError('active and passive quadrature rules must match')
        self.active_load = active_load
        self.loads = InterfaceLoads()

    def force_from_geometry(self, x, F, endo_area, loads):
        # Never add the legacy pressure/tension even if an execution caller
        # supplies a cached old pair. The current fluid stress carries pressure.
        passive = super().force_from_geometry(x,F,endo_area,(0.,0.))
        return passive+self.active_load.force(x)

    def tangent_factory(self, chunk_size=2048):
        return EPDrivenTangent(self,chunk_size)

    def diagnostics(self, x):
        result = super().diagnostics(x)
        F = self.element_gradient(x)
        stretch = torch.linalg.vector_norm(torch.einsum('eij,ej->ei',F,self.mesh.fiber),dim=-1)
        multiplier = 1+4.9*(stretch.clamp_min(1)-1)
        ta = self.active_load.tension(x)
        result.update(minimum_active_stretch_multiplier=multiplier.min().item(),
                      negative_active_multiplier_cell_count=0,
                      minimum_land_tension_cgs=ta.min().item(),maximum_land_tension_cgs=ta.max().item(),
                      additional_endocardial_pressure_cgs=0.)
        return result


class EPDrivenTangent(PaperHOTangent):
    @torch.no_grad()
    def assemble(self, x, time):
        # Existing sparse structure and passive/basal tangent are retained.
        base = super().assemble(x,time)
        values = base.values().clone()
        load = self.model.active_load
        args = load._args(x)
        kernel = torch.vmap(torch.func.jacrev(load._element_force,argnums=0))
        for start in range(0,len(load.geometry.cells),self.chunk_size):
            local = kernel(*(v[start:start+self.chunk_size] for v in args))
            begin = start*12*12
            values.index_add_(0,self.inverse[begin:begin+local.numel()],local.reshape(-1))
        if not torch.isfinite(values).all():
            raise FloatingPointError('nonfinite EP-driven solid tangent')
        return torch.sparse_csr_tensor(self.crow,self.col,values,size=(self.size,self.size),
            dtype=x.dtype,device=x.device,check_invariants=True)
