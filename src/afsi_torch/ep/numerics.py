"""Independent EP quadrature and linear-solver settings."""
from dataclasses import dataclass, field
from ..fluid.solvers import SolverOptions


@dataclass(frozen=True)
class MonodomainNumerics:
    discretization: str = 'nodal_p1'
    backend: str = 'amgx_pcg_amg'
    solver: SolverOptions = field(default_factory=lambda: SolverOptions(
        rtol=1e-10, atol=1e-13, check_every=5))
    time_scheme: str = 'crank_nicolson'
    amg_profile: str = 'size2'

    def __post_init__(self):
        if self.discretization not in ('nodal_p1', 'consistent_p1'):
            raise ValueError('EP discretization must be nodal_p1 or consistent_p1')
        if self.backend not in ('amgx_pcg_amg', 'torch_pcg_jacobi', 'torch_pcg_jacobi_light'):
            raise ValueError('unsupported EP solver backend')
        if not isinstance(self.solver, SolverOptions):
            raise ValueError('EP solver settings must be SolverOptions')
        if self.time_scheme not in ('backward_euler','crank_nicolson'):
            raise ValueError('EP time_scheme must be backward_euler or crank_nicolson')
        if self.amg_profile not in ('size2','size4','size8','size8_two_level'):
            raise ValueError('unsupported EP amg_profile')

    @property
    def theta(self):
        return .5 if self.time_scheme=='crank_nicolson' else 1.

    @classmethod
    def legacy(cls):
        return cls(discretization='consistent_p1', backend='torch_pcg_jacobi',time_scheme='backward_euler')
