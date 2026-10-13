"""Independent EP quadrature and linear-solver settings."""
from dataclasses import dataclass, field
from ..fluid.solvers import SolverOptions


@dataclass(frozen=True)
class MonodomainNumerics:
    discretization: str = 'nodal_p1'
    backend: str = 'amgx_pcg_amg'
    solver: SolverOptions = field(default_factory=lambda: SolverOptions(
        rtol=1e-10, atol=1e-13, check_every=5))

    def __post_init__(self):
        if self.discretization not in ('nodal_p1', 'consistent_p1'):
            raise ValueError('EP discretization must be nodal_p1 or consistent_p1')
        if self.backend not in ('amgx_pcg_amg', 'torch_pcg_jacobi'):
            raise ValueError('EP backend must be amgx_pcg_amg or torch_pcg_jacobi')
        if not isinstance(self.solver, SolverOptions):
            raise ValueError('EP solver settings must be SolverOptions')

    @classmethod
    def legacy(cls):
        return cls(discretization='consistent_p1', backend='torch_pcg_jacobi')
