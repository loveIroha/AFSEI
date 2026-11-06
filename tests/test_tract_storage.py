"""Moving P1 cap / fixed cut volume, finite-step work and Newton derivative."""
import os
from types import SimpleNamespace
import torch
import pytest
from afsi_torch.ep.tract_storage import TractStorage


def setup():
    device=os.environ.get('AFSEI_TEST_DEVICE','cpu')
    x=torch.tensor([[0.,0.,0.],[2.,0.,0.],[2.,1.,0.],[0.,1.,0.],[.5,.5,-1.]],
                   device=device,dtype=torch.float64)
    cap=SimpleNamespace(rim=torch.tensor([[0,1],[1,2],[2,3],[3,0]],device=device))
    domain=SimpleNamespace(axis=2,bounds=[[(.25,.75),(.25,.75)],[(1.25,1.75),(.25,.75)]],
                           metadata={'cut_coordinate_cm':1.})
    return x,TractStorage(cap,domain)


def test_uniform_cap_motion_has_exact_tract_storage():
    x,s=setup()
    y=x.clone();y[:4,2]=.03
    g=s.discrete(x,y)
    assert abs(s.volume(x)-.5)<1e-12
    assert abs(s.volume(y)-.485)<1e-12
    assert abs(float((g*(y-x)).sum())+.015)<1e-12
    assert g[4].abs().max().item()==0.


def test_discrete_gradient_and_full_geometry_jacobian():
    x,s=setup()
    dy=torch.sin(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x)*.005
    y=x+dy
    g=s.discrete(x,y)
    assert abs(float((g*dy).sum())-(s.volume(y)-s.volume(x)))<1e-12
    action=s.derivative(x,y)
    v=torch.cos(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x)
    jv=action(v)
    eps=1e-5
    plus=s.discrete(x,y+eps*v).clone()
    minus=s.discrete(x,y-eps*v).clone()
    fd=(plus-minus)/(2*eps)
    assert ((jv-fd).norm()/jv.norm()).item()<2e-4


def test_detached_cap_cannot_silently_lose_tract_volume():
    x,s=setup()
    x[:4,0]+=10.
    with pytest.raises(ValueError,match='no longer covers'):
        s.volume(x)


def test_static_volume_gradient_hessian_supports_block_vectors():
    x,s=setup()
    v=torch.cos(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x)
    block=torch.stack((v,-.3*v),-1)
    action=s.derivative(x,x)
    eps=1e-5
    fd=(s.discrete(x+eps*v,x+eps*v)-s.discrete(x-eps*v,x-eps*v))/(2*eps)
    h=2*action(v)
    torch.testing.assert_close(2*action(block),torch.stack((h,-.3*h),-1),rtol=1e-11,atol=1e-12)
    torch.testing.assert_close(h,fd,rtol=2e-5,atol=1e-8)


def test_tiny_motion_does_not_amplify_volume_roundoff():
    x,s=setup()
    x[:4,2]+=10.
    s.domain.metadata['cut_coordinate_cm']=13.
    for scale in (1e-7,1e-9,1e-11):
        y=x+scale*torch.cos(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x)
        g=s.discrete(x,y)
        assert abs(float((g*(y-x)).sum())-(s.volume(y)-s.volume(x)))<2e-12
        assert torch.isfinite(s.derivative(x,y)(torch.ones_like(x))).all()
