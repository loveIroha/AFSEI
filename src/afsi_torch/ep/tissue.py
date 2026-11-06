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


@dataclass(frozen=True)
class ODERecoveryConfig:
    """Opt-in, failed-node-only substepping of the unchanged SDIRK kernel.

    Two successive finite refinements must agree in both the final state and
    the interval-averaged ionic current. The latter resolves fast gate
    transients that a finite endpoint alone does not detect. Old checkpoints
    retain rejection-only behavior unless explicitly migrated.
    """
    enabled: bool = False
    max_refinement: int = 1024
    rtol: float = 0.01
    state_atol: float = 1e-9
    current_atol: float = 1e-3  # uA/cm^2

    def __post_init__(self):
        n = self.max_refinement
        if (type(self.enabled) is not bool or type(n) is not int or n < 4
                or n & (n-1) or n > 4096
                or any(not isfinite(v) or v <= 0 for v in
                       (self.rtol, self.state_atol, self.current_atol))
                or self.rtol > 0.05):
            raise ValueError('invalid failed-node ODE recovery settings')


def initial_state(like):
    if like.ndim != 1 or like.dtype != torch.float64 or not like.numel():
        raise ValueError('nonempty float64 nodal scalar template required')
    return TissueState(torch.full_like(like, RESTING_VOLTAGE_V),
        like.new_tensor(GPB_INITIAL).expand(len(like), -1).clone(),
        like.new_tensor((0., 0., 0., 1.)).expand(len(like), -1).clone())


class GPBLand:
    def __init__(self, dt_ep_s=5e-5, dt_ode_ms=0.002, *, recovery=None, method='sdirk'):
        if any(not isfinite(v) or v <= 0 for v in (dt_ep_s, dt_ode_ms)):
            raise ValueError('positive EP seconds and ODE milliseconds required')
        ratio = dt_ep_s*1000/dt_ode_ms
        substeps = round(ratio)
        if substeps < 1 or abs(ratio-substeps) > 1e-9:
            raise ValueError('EP step must contain an integer number of ODE substeps')
        self.dt_ep_s, self.dt_ode_ms, self.substeps = dt_ep_s, dt_ode_ms, substeps
        self.recovery = recovery if recovery is not None else ODERecoveryConfig()
        if not isinstance(self.recovery, ODERecoveryConfig):
            raise ValueError('ODERecoveryConfig required')
        if method not in ('sdirk','explicit_euler'):
            raise ValueError('GPB ODE method must be sdirk or explicit_euler')
        if method=='explicit_euler' and self.recovery.enabled:
            raise ValueError('SDIRK failed-node recovery is not applicable to Euler')
        self.method=method
        self.last_recovery = None

    @staticmethod
    def _invalid(gpb, land, current, guards):
        # GPB nonfinite entries are counted by the existing CUDA kernel.
        return ((guards != 0) | ~torch.isfinite(land).all(1)
                | ~torch.isfinite(current) | (gpb[:, 30:38] <= 0).any(1))

    def _recover(self, state, stretch, zs, zw, time_s, advance_land, outputs, bad):
        settings = self.recovery
        indices = bad.nonzero().flatten()
        count = len(indices)
        voltages = state.voltage_v[indices]*1000
        event = dict(time_ms=time_s*1000, recovered_nodes=count,
            first_node_indices=indices[:16].cpu().tolist(),
            voltage_range_mv=[float(voltages.min()), float(voltages.max())],
            extra_node_substeps=0, refinements=[], max_refinement=0,
            max_accepted_error_ratio=0.)
        inputs = tuple(a[indices].contiguous() for a in
            (state.gpb, state.land, state.voltage_v*1000, stretch, zs, zw))
        previous = None
        previous_valid = None
        factor = 2
        while factor <= settings.max_refinement:
            trial = cuda_tissue.advance(*inputs, time_s*1000,
                self.dt_ode_ms/factor, self.substeps*factor, advance_land)
            gpb, land, current, guards = trial
            valid = ~self._invalid(*trial)
            # Bad candidates never provide a reference for step doubling.
            error = torch.full_like(current, float('inf'))
            if previous is not None:
                errors = [((new-old).abs() /
                    (atol+settings.rtol*torch.maximum(new.abs(),old.abs())))
                    for new,old,atol in zip(trial[:3],previous,
                        (settings.state_atol,settings.state_atol,settings.current_atol))]
                error = torch.maximum(torch.maximum(errors[0].amax(1),errors[1].amax(1)),errors[2])
                error = torch.where(valid & previous_valid, error,
                                    torch.full_like(error, float('inf')))
            accepted = valid & (error <= 1.)
            event['extra_node_substeps'] += len(indices)*self.substeps*factor
            event['refinements'].append(dict(factor=factor, nodes=len(indices),
                                             accepted_nodes=int(accepted.sum())))
            if accepted.any():
                for destination,value in zip(outputs,trial[:3]):
                    destination[indices[accepted]] = value[accepted]
                event['max_refinement'] = factor
                event['max_accepted_error_ratio'] = max(event['max_accepted_error_ratio'],
                                                        float(error[accepted].max()))
            pending = ~accepted
            if not pending.any():
                self.last_recovery = event
                return
            indices = indices[pending]
            inputs = tuple(a[pending].contiguous() for a in inputs)
            previous = tuple(a[pending] for a in trial[:3])
            previous_valid = valid[pending]
            factor *= 2
        raise FloatingPointError('GPB/Land failed-node ODE recovery did not converge: '
            f'{len(indices)} nodes, time={time_s*1000:.6f} ms, '
            f'max_refinement={settings.max_refinement}; reject this physical trial')

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
        self.last_recovery = None
        self._validate(state, stretch, zs, zw, time_s)
        if self.method=='explicit_euler':
            from . import cuda_euler
            advance=cuda_euler.advance
        else:advance=cuda_tissue.advance
        gpb, land, current, guards = advance(state.gpb.contiguous(), state.land.contiguous(),
            (state.voltage_v*1000).contiguous(), stretch.contiguous(), zs.contiguous(), zw.contiguous(),
            time_s*1000, self.dt_ode_ms, self.substeps, advance_land)
        bad = self._invalid(gpb, land, current, guards)
        if bad.any():
            if not self.recovery.enabled:
                raise FloatingPointError('GPB/Land produced invalid values; reject this physical trial')
            # Retry from the ORIGINAL state, never from the protected/zeroed
            # output. Successful nodes retain the original arithmetic exactly.
            self._recover(state,stretch,zs,zw,time_s,advance_land,(gpb,land,current),bad)
        return TissueState(state.voltage_v, gpb, land), current

    @torch.no_grad()
    def step(self, ep, state, stretch, zs, zw, time_s, stimulus, *, advance_land=True):
        ode, current = self.advance_ode(state, stretch, zs, zw, time_s, advance_land=advance_land)
        voltage, info = ep.step(state.voltage_v, current, stimulus, self.dt_ep_s)
        return TissueState(voltage, ode.gpb, ode.land), current, info
