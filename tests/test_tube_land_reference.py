from dataclasses import replace
import torch
import pytest
from afsi_torch.solids.circular_tube import CircularTubeGeometry,generate_circular_tube
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.ep.tissue import initial_state,GPBLand
from afsi_torch.ep.planar_tube import land_resting_equilibrium
from afsi_torch.ep.land import land_tension,land_ode_step

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_reference_ratio_rest_feedback_and_consistent_tangent(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    g=CircularTubeGeometry(axial_elements=3,circumferential_elements=8,thickness_elements=1)
    m=generate_circular_tube(g,device=device)
    solid=CircularTubeSolid(m,g,young_modulus_dyn_cm2=3e5,poisson_ratio=.3,
        end_spring_dyn_cm3=0.,tension_scale=1.,end_taper_cm=.2,dt_s=5e-5,land_reference_ratio=1.1)
    state=initial_state(m.X[:,0]);one=torch.ones_like(state.voltage_v);zero=torch.zeros_like(one)
    lam=solid.activation_stretch(one);assert float(lam.min())==1.1
    state=replace(state,land=land_resting_equilibrium(state.gpb[:,37]*1000,lam))
    solid.set_reference(land_tension(lam,lam,state.land[:,0],state.land[:,1],zero,zero,5e-5))
    solid.stage(m.X,state,zero,zero)
    assert float(solid.force(m.X,0).abs().max())<1e-7
    x=m.X.clone();x[:,:2]*=.9
    torch.testing.assert_close(solid.activation_stretch(solid.cell_stretch(x)),one.new_full((len(m.cells),),.99),rtol=1e-12,atol=1e-12)
    land=state.land.clone();land[:,0]=.03;land[:,1]=.05;state=replace(state,land=land)
    solid.stage(x,state,zero,zero)
    direction=torch.sin(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x);direction/=direction.norm()
    matrix=solid.tangent_factory().assemble(x,.1)
    tangent=torch.mv(matrix,direction.flatten()).reshape_as(x);eps=2e-7
    fd=(solid.force(x+eps*direction,.1)-solid.force(x-eps*direction,.1))/(2*eps)
    torch.testing.assert_close(tangent,fd,rtol=3e-5,atol=.02)
    if device=='cuda':
        ode=GPBLand(1e-6,.001,method='explicit_euler');nodal=one*.99
        actual,_=ode.advance_ode(state,nodal,zero,zero,0.)
        expected=land_ode_step(state.land,actual.gpb[:,37]*1000,nodal,zero,zero,.001)
        torch.testing.assert_close(actual.land,expected,rtol=2e-13,atol=2e-14)
