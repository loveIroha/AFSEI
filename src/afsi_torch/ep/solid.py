"""EP-driven active load adapter for the existing real-LV BE/IB solid API.

No prescribed active waveform is read. Fluid pressure acts via the fluid/solid
interface. An optional fixed endocardial traction retains the original AFSI
preloading mechanism; it is not a physical cavity/RCR pressure boundary.
The AFSEI driver publishes accepted EP/Land state and its own checkpoints.
Physical circulation ports and preload remain separate pending components.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from ..paper_lv import PaperLVSolid, PaperHOTangent


@dataclass(frozen=True)
class InterfaceLoads:
    pressure_mmhg: float = 0.

    def at(self, time):
        if not isfinite(time) or time < 0:
            raise ValueError('nonnegative finite force time required')
        from .windkessel import MMHG_TO_DYN_CM2
        return self.pressure_mmhg*MMHG_TO_DYN_CM2, 0.


class EPDrivenPaperSolid(PaperLVSolid):
    """Prepared per-trial solid with fixed EP states and differentiable Land Ta."""
    def __init__(self, mesh, config, active_load,*,source_quadrature=False,
                 prescribed_pressure_mmhg=0.):
        super().__init__(mesh, config)
        if source_quadrature:
            from .source_mechanics import source_basal_surface
            self.base = source_basal_surface(self.base,mesh.X)
            R = self.base.reference_positions[...,:2]-mesh.X.new_tensor(config.basal_center_cm)
            if (R.square().sum(-1) <= torch.finfo(mesh.X.dtype).eps).any():
                raise ValueError('source radial basal constraint undefined at its center')
            self.radial = R/R.square().sum(-1,keepdim=True).sqrt()
        g = active_load.geometry
        if (g.node_count != len(mesh.X) or g.cells.device != mesh.X.device
                or not torch.equal(g.cells, mesh.cells)
                or not torch.equal(g.gradients[:,0], self.gradients)
                or not torch.equal(active_load.fiber, mesh.fiber)):
            raise ValueError('active load must use the identical solid mesh, gradients and fibers')
        if active_load.geometry.weights.shape != self.geometry.weights.shape:
            raise ValueError('active and passive quadrature rules must match')
        self.active_load = active_load
        if not isfinite(prescribed_pressure_mmhg) or prescribed_pressure_mmhg < 0:
            raise ValueError('nonnegative finite prescribed endocardial pressure required')
        self.loads = InterfaceLoads(prescribed_pressure_mmhg)
        self.prescribed_pressure_cgs = self.loads.at(0.)[0]

    def force_from_geometry(self, x, F, endo_area, loads):
        # A stale legacy waveform must not override the explicit fixed traction.
        passive = super().force_from_geometry(x,F,endo_area,(self.prescribed_pressure_cgs,0.))
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
                      additional_endocardial_pressure_cgs=self.prescribed_pressure_cgs)
        return result


class EPDrivenTangent(PaperHOTangent):
    def __init__(self, model, chunk_size=2048):
        super().__init__(model, chunk_size)
        from .active_tangent import active_local_jacobian
        from ..mac.execution import tensor_kernel
        self._active_kernel = tensor_kernel(active_local_jacobian, model.mesh.X.device)

    @torch.no_grad()
    def assemble(self, x, time):
        # Existing sparse structure and passive/basal tangent are retained.
        base = super().assemble(x,time)
        values = base.values().clone()
        load = self.model.active_load
        args = load._args(x)
        # Both production buffered trials and immutable reference loads work.
        if hasattr(load, 'dt'):
            dt, use_land, ramp = load.dt, load.use_land, load.ramp
        else:
            dt = x.new_tensor(load.dt_s)
            use_land = x.new_tensor(float(load.mode == 'land'))
            ramp = x.new_tensor(load.ramp_factor)
        for start in range(0,len(load.geometry.cells),self.chunk_size):
            local = self._active_kernel(*(v[start:start+self.chunk_size] for v in args),
                                        dt,use_land,ramp)
            begin = start*12*12
            values.index_add_(0,self.inverse[begin:begin+local.numel()],local.reshape(-1))
        if not torch.isfinite(values).all():
            raise FloatingPointError('nonfinite EP-driven solid tangent')
        return torch.sparse_csr_tensor(self.crow,self.col,values,size=(self.size,self.size),
            dtype=x.dtype,device=x.device,check_invariants=True)
