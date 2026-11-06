"""Sharifi (2021), Eq. (2): implicit two-element RC pressure outlet.

CGS internally; paper R [mmHg s/mm^3], C [mm^3/mmHg]. No porous region.
The adapters load a WHOLE OUTER MAC BOX FACE, not an immersed tube section.
They are a boundary-validation stage, not a reproduction of the paper geometry.
The existing zero-pressure fluid solve and its Jacobian remain untouched.
One cached unit-pressure projection gives an exact scalar Schur complement
for this split projection operator, including the RC derivative in FSI Newton.
Only commit() publishes capacitor history, after acceptance of the FSI step.
"""
from dataclasses import dataclass
from math import isfinite, prod
import torch

MMHG_TO_DYN_CM2 = 1333.22387415


@dataclass(frozen=True)
class RCParameters:
    resistance_mmhg_s_per_mm3: float = 60.
    compliance_mm3_per_mmhg: float = .26
    initial_pressure_mmhg: float = 0.

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError('RC parameters must be finite numbers')
            if name != 'initial_pressure_mmhg' and value <= 0:
                raise ValueError('RC resistance and compliance must be positive')

    @property
    def resistance(self):
        return self.resistance_mmhg_s_per_mm3*MMHG_TO_DYN_CM2*1000

    @property
    def compliance(self):
        return self.compliance_mm3_per_mmhg/1000/MMHG_TO_DYN_CM2


def parse_rc(config, *, dimension):
    expected = {'model', 'resistance_mmhg_s_per_mm3', 'compliance_mm3_per_mmhg',
                'initial_pressure_mmhg', 'outlet_scope'}
    if dimension == 2:
        expected.add('extrusion_depth_cm')
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError('circulation requires exactly: '+', '.join(sorted(expected)))
    scope = 'x-high-full-box-face' if dimension == 2 else 'z-high-full-box-face'
    if config['model'] != 'sharifi-2021-rc' or config['outlet_scope'] != scope:
        raise ValueError('RC boundary stage requires '+scope+' and sharifi-2021-rc')
    if dimension == 2:
        d = config['extrusion_depth_cm']
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not isfinite(d) or d <= 0:
            raise ValueError('2D RC flow requires an explicit positive extrusion_depth_cm')
    return RCParameters(**{k:config[k] for k in RCParameters.__dataclass_fields__})


class _RCOutlet:
    def __getattr__(self, name):
        return getattr(self.base, name)

    def _configure(self, base, parameters, axis, depth, unit_p, unit_velocity):
        self.base, self.parameters, self.axis = base, parameters, axis
        self.unit_pressure = unit_p
        # Solvers may own/reuse workspace storage. The lift must not alias it.
        self.unit_velocity = tuple(v.clone() for v in unit_velocity)
        self.face_area = prod(h for a,h in enumerate(base.grid.spacing) if a != axis)*depth
        self.unit_flux = float(self.flux(self.unit_velocity))
        if not isfinite(self.unit_flux) or self.unit_flux >= 0:
            raise RuntimeError('outlet unit-pressure response must oppose outward flow')
        self.C, self.R = parameters.compliance, parameters.resistance
        self.D = self.C+base.dt/self.R
        self.schur = self.D-base.dt*self.unit_flux
        self.pressure = unit_p.new_tensor(parameters.initial_pressure_mmhg*MMHG_TO_DYN_CM2)
        self.committed_steps = 0
        self.max_balance = torch.zeros_like(self.pressure)
        self.max_energy_error = torch.zeros_like(self.pressure)
        self.integrated_outlet_flow = torch.zeros_like(self.pressure)
        self.last_accepted = dict(pressure_dyn_cm2=self.pressure.clone(),
            flow_ml_s=self.pressure.new_zeros(()), balance_ml_s=self.pressure.new_zeros(()),
            energy_error_erg_s=self.pressure.new_zeros(()),
            capacitor_energy_erg=.5*self.C*self.pressure.square())

    def flux(self, velocity):
        return velocity[self.axis].select(self.axis, -1).sum()*self.face_area

    def _couple(self, velocity, pressure, info):
        q0 = self.flux(velocity)
        p = (self.C*self.pressure+self.dt*q0)/self.schur
        u = tuple(v+p*w for v,w in zip(velocity, self.unit_velocity))
        return u, pressure+p*self.unit_pressure, dict(info,
            rc_model='implicit BE RC / exact scalar Schur complement of original projection',
            rc_outlet_scope='whole outer box face; not anatomical tube mouth')

    def commit(self, velocity, *, step):
        if type(step) is not int or step != self.committed_steps+1:
            raise ValueError('RC history must be committed exactly once per accepted FSI step')
        q = self.flux(velocity)
        old = self.pressure
        p = (self.C*old+self.dt*q)/self.D
        balance = self.C*(p-old)/self.dt-q+p/self.R
        energy = .5*self.C*p.square()
        storage_rate = (energy-.5*self.C*old.square())/self.dt
        be_loss = .5*self.C*(p-old).square()/self.dt
        self.last_accepted = dict(pressure_dyn_cm2=p.clone(), flow_ml_s=q.clone(),
            balance_ml_s=balance, capacitor_energy_erg=energy,
            resistance_loss_erg_s=p.square()/self.R, be_loss_erg_s=be_loss,
            boundary_output_power_erg_s=p*q,
            energy_error_erg_s=p*q-storage_rate-p.square()/self.R-be_loss)
        self.max_balance = torch.maximum(self.max_balance, balance.abs())
        self.max_energy_error = torch.maximum(self.max_energy_error,
                                             self.last_accepted['energy_error_erg_s'].abs())
        self.integrated_outlet_flow = self.integrated_outlet_flow+self.dt*q
        if hasattr(self.base, 'domain'):
            self.base.domain.accepted(velocity)
            self.last_accepted['port_power_error_erg_s'] = p*(q-self.base.domain.last['outlet_aperture_ml_s'])
        self.pressure = p.clone()
        self.committed_steps = step

    def diagnostics(self):
        return dict({'rc_'+k:float(v) for k,v in self.last_accepted.items()},
                    rc_pressure_mmhg=float(self.pressure)/MMHG_TO_DYN_CM2,
                    rc_max_balance_ml_s=float(self.max_balance),
                    rc_max_energy_error_erg_s=float(self.max_energy_error),
                    rc_integrated_outlet_ml=float(self.integrated_outlet_flow),
                    rc_committed_steps=self.committed_steps)

    def checkpoint(self):
        return dict(rc_pressure_dyn_cm2=self.pressure.detach().cpu().numpy(),
                    rc_committed_steps=self.committed_steps,
                    rc_integrated_outlet_ml=self.integrated_outlet_flow.detach().cpu().numpy())


class RCChannelFlow(_RCOutlet):
    """2D end-pressure MAC: x-low p=0, x-high RC, outer y walls no-slip.

    The planar area flux is multiplied by an explicit extrusion depth before
    entering the 3D-volume-flow RC equation. All existing transport is retained.
    """
    def __init__(self, base, parameters, like, *, extrusion_depth_cm=1.):
        from ..mac2d.pressure_channel import pressure_gradient
        if base.inlet_pressure != 0 or base.outlet_pressure != 0:
            raise ValueError('RC adapter requires a zero-pressure base channel')
        if not isfinite(extrusion_depth_cm) or extrusion_depth_cm <= 0:
            raise ValueError('positive extrusion depth required')
        rhs = like.new_zeros(base.grid.shape)
        rhs[-1] = 2/base.grid.spacing[0]**2
        p, _ = base.pressure_solver.solve(rhs)
        if hasattr(base, 'domain'):
            from ..mac.sealed_pressure import gradient
            grad = list(gradient(p, base.grid.spacing, base.domain.masks))
            grad[0][-1] += 2/base.grid.spacing[0]
        else:
            grad = pressure_gradient(p, base.grid.spacing, 0., 1.)
        velocity = tuple(-base.dt/base.rho*g for g in grad)
        self._configure(base, parameters, 0, extrusion_depth_cm, p.clone(), velocity)

    def step(self, velocity, density, *, time, pressure_initial=None):
        from ..mac2d.flow import ChannelResult
        result = self.base.step(velocity, density, time=time, pressure_initial=pressure_initial)
        u,p,info = self._couple(result.velocity, result.pressure, result.diagnostics)
        return ChannelResult(u,p,info)


class RCBoxFlow(_RCOutlet):
    """3D original open-box BE projection with RC on its entire high-z face.

    Other five box faces retain p=0 / velocity normal derivative zero. The
    existing tube and its exterior fluid remain inside the box at this stage.
    The load is *not* attached to the approximate lumen flux diagnostic.
    """
    def __init__(self, base, parameters, like):
        from ..mac.open_boundary import gradient
        rhs = like.new_zeros(base.grid.shape)
        rhs[:, :, -1] = 2/base.grid.spacing[2]**2
        p, _ = base.pressure_solver.solve(rhs)
        grad = list(gradient(p, base.grid.spacing))
        grad[2] = grad[2].clone()
        grad[2][:, :, -1] += 2/base.grid.spacing[2]
        unit = tuple(-base.dt/base.rho*g for g in grad)
        if hasattr(base, 'domain'):
            unit = base.domain.mask(unit)
        self._configure(base, parameters, 2, 1., p.clone(), unit)

    def advance(self, departure_velocity, density, pressure_initial=None):
        return self._couple(*self.base.advance(departure_velocity, density, pressure_initial))

    def response(self, density):
        velocity = self.base.response(density)
        dp = self.dt*self.flux(velocity)/self.schur
        return tuple(v+dp*w for v,w in zip(velocity, self.unit_velocity))
