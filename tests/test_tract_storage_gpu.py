"""Exact CPU/GPU geometry agreement, including moving clip topology."""
import pytest
import torch
from types import SimpleNamespace
from afsi_torch.ep.tract_storage import TractStorage
from afsi_torch.ep.tract_storage_gpu import GPUTractStorage


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires actual CUDA')
@pytest.mark.parametrize('scale',[0.,1e-9,.005,.15])
def test_cpu_gpu_exact_geometry(scale):
    x=torch.tensor([[0.,0.,0.],[2.,0.,0.],[2.,1.,0.],[0.,1.,0.],[.5,.5,-1.]],device='cuda',dtype=torch.float64)
    cap=SimpleNamespace(rim=torch.tensor([[0,1],[1,2],[2,3],[3,0]],device='cuda'))
    domain=SimpleNamespace(axis=2,bounds=[[(.25,.75),(.25,.75)],[(1.25,1.75),(.25,.75)]],metadata={'cut_coordinate_cm':1.})
    cpu=TractStorage(cap,domain);gpu=GPUTractStorage(cap,domain)
    v=torch.cos(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x)
    for offset in (0.,.08,-.04):
        y=x+scale*v;y[:,0]+=offset
        assert abs(cpu.volume(y)-gpu.volume(y))<2e-13
        torch.testing.assert_close(gpu.discrete(x,y),cpu.discrete(x,y),rtol=1e-8,atol=3e-11)
        block=torch.stack((v,-.3*v),-1)
        torch.testing.assert_close(gpu.derivative(x,y)(block),cpu.derivative(x,y)(block),rtol=5e-7,atol=2e-9)
        assert abs(float((gpu.discrete(x,y)*(y-x)).sum())-(gpu.volume(y)-gpu.volume(x)))<2e-12
    assert gpu.fallback_count==0
    y=x.clone();y[:,0]+=10.
    with pytest.raises(ValueError,match='no longer covers'):gpu.volume(y)
