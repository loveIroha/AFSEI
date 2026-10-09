"""Transactional EP split: frozen left-endpoint Vm -> GPB/Land -> PyTorch PDE.

Inputs are never overwritten. A caller publishes returned state only after
the full fluid/solid step converges, or discards it and retries from old state.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from .initial import GPB_INITIAL, RESTING_VOLTAGE_V
from . import cuda_tissue


@dataclass(frozen=True)
class TissueState:
    voltage_v: torch.Tensor
    gpb: torch.Tensor
    land: torch.Tensor  # XS, XW, CaTrpn, TmB


def initial_state(like):
    if like.ndim != 1 or like.dtype != torch.float64 or not like.numel():
        raise ValueError('nonempty float64 nodal scalar template required')
    return TissueState(torch.full_like(like, RESTING_VOLTAGE_V),
        like.new_tensor(GPB_INITIAL).expand(len(like), -1).clone(),
        like.new_tensor((0., 0., 0., 1.)).expand(len(like), -1).clone())


class GPBLand:
    def __init__(self, dt_ep_s=5e-5, dt_ode_ms=0.002):
        if any(not isfinite(v) or v <= 0 for v in (dt_ep_s, dt_ode_ms)):
            raise ValueError('positive EP seconds and ODE milliseconds required')
        ratio = dt_ep_s*1000/dt_ode_ms
        substeps = round(ratio)
        if substeps < 1 or abs(ratio-substeps) > 1e-9:
            raise ValueError('EP step must contain an integer number of ODE substeps')
        self.dt_ep_s, self.dt_ode_ms, self.substeps = dt_ep_s, dt_ode_ms, substeps

    def _validate(self, state, stretch, zs, zw, time_s):
        v = state.voltage_v
        if (v.ndim != 1 or not v.is_cuda or v.dtype != torch.float64 or not v.numel()
                or not isfinite(time_s) or time_s < 0):
            raise ValueError('nonempty CUDA float64 voltage and nonnegative time in seconds required')
        for value, shape in ((state.gpb, (len(v), 57)), (state.land, (len(v), 4)),
                             *((a, v.shape) for a in (v, stretch, zs, zw))):
            if (value.shape != shape or value.device != v.device or value.dtype != v.dtype
                    or value.requires_grad or not torch.isfinite(value).all()):
                raise ValueError('finite, gradient-free GPB/Land fields with matching shape/device/dtype required')
        if (stretch <= 0).any() or (state.gpb[:, 30:38] <= 0).any():
            raise ValueError('positive stretch and GPB concentrations required')

    @torch.no_grad()
    def advance_ode(self, state, stretch, zs, zw, time_s, *, advance_land=True):
        self._validate(state, stretch, zs, zw, time_s)
        gpb, land, current, guards = cuda_tissue.advance(state.gpb.contiguous(), state.land.contiguous(),
            (state.voltage_v*1000).contiguous(), stretch.contiguous(), zs.contiguous(), zw.contiguous(),
            time_s*1000, self.dt_ode_ms, self.substeps, advance_land)
        # The source silently replaces nonfinite values with zero. Report any
        # such intervention so an FSI caller can reject/restore this trial.
        if guards.any() or not torch.isfinite(land).all() or not torch.isfinite(current).all():
            raise FloatingPointError('GPB/Land needed nonfinite-value protection; reject this physical trial')
        if (gpb[:, 30:38] <= 0).any():
            raise FloatingPointError('GPB produced nonpositive concentration; reject this physical trial')
        return TissueState(state.voltage_v, gpb, land), current

    @torch.no_grad()
    def step(self, ep, state, stretch, zs, zw, time_s, stimulus, *, advance_land=True):
        ode, current = self.advance_ode(state, stretch, zs, zw, time_s, advance_land=advance_land)
        voltage, info = ep.step(state.voltage_v, current, stimulus, self.dt_ep_s)
        return TissueState(voltage, ode.gpb, ode.land), current, info
