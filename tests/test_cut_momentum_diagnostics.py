"""Independent uniform-channel balances and read-only observer checks."""
import pytest
import torch
from test_cut_plane import channel
from afsi_torch.ep.cut_diagnostics import fixed_tract_momentum
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C


@pytest.mark.parametrize('dt',[1e-4,5e-5])
def test_transient_pressure_transport_and_force_balance(dt):
    flow=channel(dt)
    old=list(flow.grid.zeros());old[2].fill_(.15)
    departure=list(flow.grid.zeros());departure[2].fill_(.1)
    density=list(flow.grid.zeros());density[2].fill_(4.)
    flow.prescribe([8.,7.]);u,p,_=flow.advance(tuple(departure),tuple(density))
    saved=[v.clone() for v in (*old,*departure,*density,*u,p)]
    bottom=3;length=2.-bottom*flow.grid.spacing[2]
    b=fixed_tract_momentum(flow,old,departure,u,p,density,axis=2,bottom=bottom,
        aperture=flow.planes[-1].aperture)
    expected_u=.1+dt*4.+dt*C/2.
    assert abs(b['measured_drop_mmhg']-length/2.)<1e-8
    assert abs(b['terms_mmhg']['unsteady']-(expected_u-.15)/dt*length/C)<1e-8
    assert abs(b['terms_mmhg']['transport']-.05/dt*length/C)<1e-8
    assert abs(b['terms_mmhg']['ib_and_applied_force']+4*length/C)<1e-10
    assert abs(b['terms_mmhg']['viscous_predictor'])<1e-8
    assert abs(b['closure_error_mmhg'])<1e-8
    assert abs(b['pressure_integral_error_mmhg'])<1e-10
    for actual,original in zip((*old,*departure,*density,*u,p),saved):assert torch.equal(actual,original)


def test_uniform_hydrostatic_case_has_no_spurious_momentum_terms():
    flow=channel();zero=flow.grid.zeros();flow.prescribe([8.,8.])
    u,p,_=flow.advance(zero,zero)
    b=fixed_tract_momentum(flow,zero,zero,u,p,zero,axis=2,bottom=3,aperture=flow.planes[-1].aperture)
    assert max(abs(v) for v in b['terms_mmhg'].values())<1e-7
    assert abs(b['closure_error_mmhg'])<1e-8


def test_nonuniform_predictor_retains_projection_diffusion_contribution():
    flow=channel();zero=flow.grid.zeros();density=list(flow.grid.zeros())
    z=flow.grid.coordinates(2)[...,2]
    density[2].copy_(1000*torch.cos(3*z))
    flow.prescribe([8.,7.]);u,p,_=flow.advance(zero,tuple(density))
    b=fixed_tract_momentum(flow,zero,zero,u,p,density,axis=2,bottom=3,aperture=flow.planes[-1].aperture)
    assert abs(b['terms_mmhg']['viscous_predictor'])>1e-6
    assert abs(b['closure_error_mmhg'])<1e-7
