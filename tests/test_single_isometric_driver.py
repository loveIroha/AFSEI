from dataclasses import replace
import pytest
import torch
from test_circular_tube import example
from afsi_torch.ep.circular_tube import CircularTubeSolid


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_isometric_driver_removes_length_history_but_keeps_geometric_tangent(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    c,mesh,_,state,zero=example()
    from afsi_torch.solids.circular_tube import generate_circular_tube
    from afsi_torch.ep.tissue import TissueState
    mesh=generate_circular_tube(c,device=device)
    state=TissueState(*(v.to(device) for v in (state.voltage_v,state.gpb,state.land)))
    zero=zero.to(device)
    model=CircularTubeSolid(mesh,c,young_modulus_dyn_cm2=3e5,poisson_ratio=.3,
        end_spring_dyn_cm3=0.,tension_scale=1.,end_taper_cm=.2,dt_s=5e-5,
        stretch_feedback='isometric-driver')
    land=state.land.clone();land[:,0]=.03
    model.stage(mesh.X,replace(state,land=land),zero,zero)
    x=mesh.X.clone();center=x.new_tensor(c.center_xy_cm);x[:,:2]=center+.8*(x[:,:2]-center)
    torch.testing.assert_close(model.applied_tension(x),model.applied_tension(mesh.X))
    g=torch.Generator(device=device).manual_seed(17)
    direction=torch.randn(x.shape,generator=g,device=device,dtype=x.dtype);direction/=direction.norm()
    J=model.tangent_factory().assemble(x,.1)
    actual=torch.mv(J,direction.flatten()).reshape_as(x)
    eps=2e-7
    fd=(model.force(x+eps*direction,.1)-model.force(x-eps*direction,.1))/(2*eps)
    torch.testing.assert_close(actual,fd,rtol=2e-5,atol=.01)
    assert actual.norm()>0
    torch.testing.assert_close(model.activation_stretch(x[:,0]*0+.8),torch.ones_like(x[:,0]))


def test_full_end_spring_has_radial_restoring_reaction_and_exact_triangle_energy():
    c,mesh,_,_,_=example()
    args=dict(young_modulus_dyn_cm2=3e5,poisson_ratio=.3,end_spring_dyn_cm3=1e6,
        tension_scale=0.,end_taper_cm=.2,dt_s=5e-5)
    full=CircularTubeSolid(mesh,c,**args,end_support='full-reference-surface-spring-approximation')
    axial=CircularTubeSolid(mesh,c,**args)
    shift=mesh.X.new_tensor([.01,0.,0.]);x=mesh.X+shift
    assert abs(float(axial.force(x,0.).sum(0)[0]))<1e-6
    force=full.force(x,0.).sum(0)
    faces=mesh.X[mesh.end_faces]
    area=.5*torch.linalg.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]).norm(dim=-1).sum()
    torch.testing.assert_close(force,-1e6*area*shift,rtol=1e-10,atol=1e-6)
