import math
import pytest
import torch
from afsi_torch.ep.land import LandQuadratureLoad, distortion_step, land_tension, active_pk1, land_ode_step


@pytest.fixture(params=['cpu','cuda'])
def setup(request):
    if request.param == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    X = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=torch.float64,device=request.param)
    cells = torch.tensor([[0,1,2,3]],device=request.param)
    fiber = X.new_tensor([[1.,0.,0.]])
    xs = X.new_tensor([.03,.06,.09,.15])
    xw = X.new_tensor([.06,.08,.13,.16])
    zs = X.new_tensor([.01,.02,-.03,-.02])
    zw = X.new_tensor([-.03,.04,.02,-.01])
    prev = X.clone(); prev[:,0] *= 1.03
    load = LandQuadratureLoad.prepare(X,cells,fiber,prev,xs,xw,zs,zw,.001)
    x = X.clone(); x[:,0] *= 1.09; x[:,1] *= .96
    return X,cells,fiber,prev,xs,xw,zs,zw,load,x


def test_spatial_quadrature_and_weak_force_balance(setup):
    X,cells,fiber,prev,xs,xw,zs,zw,load,x = setup
    N = load.geometry.values
    # Scalar source formula evaluated independently at the actual quadrature points.
    expected = []
    for shape in N.cpu().tolist():
        interp = lambda v: sum(a*b for a,b in zip(shape,v.cpu().tolist()))
        update = lambda old,c: old*math.exp(-c*.001)+10*.06*(-math.expm1(-c*.001))/(c*.001)
        h = 1+2.3*(1.09+.87-1.87)
        expected.append(max(0,h*4.8e6*(interp(xs)*(update(interp(zs),40.14)+1)
                         +interp(xw)*update(interp(zw),405.86))))
    torch.testing.assert_close(load.tension(x).cpu().flatten(),torch.tensor(expected,dtype=X.dtype),rtol=1e-13,atol=1e-8)
    assert load.tension(x).max() > load.tension(x).min()*1.1
    force = load.force(x)
    torch.testing.assert_close(force.sum(0),X.new_zeros(3),rtol=0,atol=1e-9)
    # Virtual work checks the minus weak-form sign and complete integrated PK1.
    direction = X.new_tensor([[.1,.02,-.03],[.01,-.05,.03],[-.02,.04,.06],[.03,.05,.02]])
    F = x.T@load.geometry.gradients[0,0]
    deltaF = direction.T@load.geometry.gradients[0,0]
    P = active_pk1(F[None],fiber[0][None],load.tension(x)[0])
    work = -(load.geometry.weights[0]*(P*deltaF).sum((-1,-2))).sum()
    torch.testing.assert_close((force*direction).sum(),work,rtol=1e-12,atol=1e-8)


def test_consistent_tangent_and_trial_history(setup):
    X,cells,fiber,prev,xs,xw,zs,zw,load,x = setup
    before = [load.fields.clone(),load.previous_F.clone()]
    direction = X.new_tensor([[.1,.02,-.03],[.01,-.05,.03],[-.02,.04,.06],[.03,.05,.02]])
    jvp = load.force_jvp(x,direction)
    eps = 1e-6
    fd = (load.force(x+eps*direction)-load.force(x-eps*direction))/(2*eps)
    torch.testing.assert_close(jvp,fd,rtol=2e-8,atol=1e-3)
    local = load.local_jacobian(x,1)
    torch.testing.assert_close(torch.einsum('eaibk,ebk->eai',local,direction[cells])[0],jvp,rtol=1e-12,atol=1e-8)
    # Rejecting a trial must not change histories or the result on retry.
    load.force(x*1.04)
    torch.testing.assert_close(load.fields,before[0],rtol=0,atol=0)
    torch.testing.assert_close(load.previous_F,before[1],rtol=0,atol=0)
    # Freezing Ta loses a material part of the tangent during contraction.
    anchor = load.closure_snapshot(x)
    frozen = LandQuadratureLoad.prepare(X,cells,fiber,prev,xs,xw,zs,zw,.001,anchor=anchor,ramp_factor=1)
    assert torch.linalg.vector_norm(jvp-frozen.force_jvp(x,direction)) > .1*torch.linalg.vector_norm(jvp)


def test_closure_anchor_ramp_and_clipped_geometric_factor(setup):
    X,cells,fiber,prev,xs,xw,zs,zw,load,x = setup
    anchor = load.closure_snapshot(x)
    ramp = LandQuadratureLoad.prepare(X,cells,fiber,prev,xs,xw,zs,zw,.001,anchor=anchor,ramp_factor=.5)
    torch.testing.assert_close(ramp.force(x),.5*load.force(x),rtol=1e-13,atol=1e-9)
    deformed = x.clone(); deformed[:,0] *= .8
    torch.testing.assert_close(ramp.tension(deformed),anchor*.5,rtol=0,atol=0)
    identity = torch.eye(3,dtype=X.dtype,device=X.device); identity[0,0]=.7
    P = active_pk1(identity,fiber[0],X.new_tensor(10.))
    torch.testing.assert_close(P[0,0],X.new_tensor(7.),rtol=0,atol=1e-14)
    anchor.mul_(0)  # prepared closure load owns its frozen copy
    assert ramp.tension(x).max() > 0


def test_small_step_distortion_and_euler_constraints(setup):
    X = setup[0]
    z = X.new_tensor(0.3); dl = X.new_tensor(.01)
    torch.testing.assert_close(distortion_step(z,dl,1e-14,40.14),z+10*dl,rtol=1e-12,atol=0)
    state = X.new_tensor([[0.,0.,0.,1.],[.1,.2,.3,.4]])
    c = X.new_tensor([.1,.8]); stretch = X.new_tensor([1.,1.1]); zs = X.new_tensor([0.,.2]); zw = X.new_tensor([0.,-.1])
    result = land_ode_step(state,c,stretch,zs,zw,.002)
    assert result[0,2] > 0 and result[0,0] == 0
    assert (result >= 0).all() and (result[:,3] <= 1).all()
    with pytest.raises(ValueError): distortion_step(z,dl,0,40.14)
