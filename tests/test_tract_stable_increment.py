"""Cancellation-free evaluation of the existing clipped volume difference."""
from types import SimpleNamespace
import pytest
import torch
from afsi_torch.ep.tract_storage_gpu import GPUTractStorage


def geometry():
    if not torch.cuda.is_available():pytest.skip('requires actual CUDA')
    x=torch.tensor([[0.,0.,10.],[2.,0.,10.],[2.,1.,10.],[0.,1.,10.]],device='cuda',dtype=torch.float64)
    cap=SimpleNamespace(rim=torch.tensor([[0,1],[1,2],[2,3],[3,0]],device='cuda'))
    domain=SimpleNamespace(axis=2,bounds=[[(.25,.75),(.25,.75)],[(1.25,1.75),(.25,.75)]],metadata={'cut_coordinate_cm':13.})
    return x,GPUTractStorage(cap,domain)


@pytest.mark.parametrize('exponent',[16,30,40])
def test_uniform_translation_retains_tiny_exact_increment(exponent):
    x,s=geometry();y=x.clone();y[:,2]+=2.**(-exponent)
    s.discrete(x,y)
    inc=s.stable_increment(s.rim_value(x),s.rim_value(y))
    expected=-.5*(y[0,2]-x[0,2])
    torch.testing.assert_close(inc,expected,rtol=2e-14,atol=1e-28)


def test_deforming_increment_agrees_with_endpoint_volumes():
    x,s=geometry()
    y=x+.003*torch.sin(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x)
    s.discrete(x,y);inc=s.stable_increment(s.rim_value(x),s.rim_value(y))
    assert inc is not None
    assert abs(float(inc)-(s.volume(y)-s.volume(x)))<3e-14


def test_topology_change_explicitly_requests_general_fallback():
    x,s=geometry();y=x.clone();y[1,0]+=.7;y[2,0]+=.7
    s.discrete(x,y)
    assert s.stable_increment(s.rim_value(x),s.rim_value(y)) is None
