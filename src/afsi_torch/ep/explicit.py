"""Lumped P1 forward-Euler diffusion with a conservative spectral CFL bound."""
from math import ceil, isfinite
from types import SimpleNamespace
import torch


class ExplicitNodal:
    def __init__(self, operator, dt_s, safety=.9):
        if operator.numerics.discretization != 'nodal_p1':
            raise ValueError('explicit diffusion requires lumped nodal mass')
        if not isfinite(dt_s) or dt_s<=0 or not 0<safety<1:
            raise ValueError('positive dt and CFL safety in (0,1) required')
        self.operator=operator;self.dt=dt_s
        p=operator.parameters
        self.inverse_capacity=1/(p.surface_to_volume_per_cm*p.capacitance_uf_per_cm2*operator.mass_diagonal)
        K=operator.stiffness.to_sparse_coo().coalesce()
        rows=K.indices()[0]
        absolute_rows=operator.X.new_zeros(len(operator.X)).index_add_(0,rows,K.values().abs())
        # M^-1 K is similar to symmetric PSD M^-1/2 K M^-1/2.
        # Its largest eigenvalue <= infinity norm, so dt <= 2/bound is sufficient.
        bound=(absolute_rows*self.inverse_capacity).max().item()
        if not isfinite(bound) or bound<=0:raise ValueError('invalid diffusion CFL bound')
        self.dt_bound=2/bound
        self.substeps=max(1,ceil(dt_s/(safety*self.dt_bound)))
        self.sub_dt=dt_s/self.substeps

    @torch.no_grad()
    def step(self,V,I,S,dt_s):
        if dt_s!=self.dt:raise ValueError('explicit macro dt changed')
        ep=self.operator
        for v,name in [(V,'voltage'),(I,'ionic current'),(S,'stimulus')]:ep._field(v,name)
        forcing=(S-I)/ep.parameters.capacitance_uf_per_cm2
        value=V
        for _ in range(self.substeps):
            diffusion=torch.sparse.mm(ep.stiffness,value[:,None]).flatten()*self.inverse_capacity
            value=value+self.sub_dt*(forcing-diffusion)
        return value,SimpleNamespace(iterations=0,substeps=self.substeps)
