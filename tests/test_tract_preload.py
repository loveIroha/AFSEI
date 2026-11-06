"""Static pressure work and its batched tangent, independently differenced."""
import os
from types import SimpleNamespace
import torch
from afsi_torch.ep.tract_storage import TractStorage
from afsi_torch.ep.tract_preload import TractPressureLoad
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C


def case():
    device=os.environ.get('AFSEI_TEST_DEVICE','cpu')
    x=torch.tensor([[0.,0.,0.],[2.,0.,0.],[2.,1.,0.],[0.,1.,0.],[.5,.5,-1.]],device=device,dtype=torch.float64)
    rim=torch.tensor([[0,1],[1,2],[2,3],[3,0]],device=device)
    model=SimpleNamespace(rim=rim,endo_faces=torch.tensor([[0,1,4],[1,2,4],[2,3,4],[3,0,4]],device=device))
    domain=SimpleNamespace(axis=2,bounds=[[(.25,.75),(.25,.75)],[(1.25,1.75),(.25,.75)]],
                           metadata={'cut_coordinate_cm':1.})
    storage=TractStorage(SimpleNamespace(rim=rim),domain)
    return x,TractPressureLoad(model,storage)


def test_static_force_is_derivative_of_total_blood_volume():
    x,load=case()
    v=torch.cos(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x)
    eps=1e-6;p=8.
    volume=lambda y:float(load.closed.volume(y))+load.storage.volume(y)
    fd=C*p*(volume(x+eps*v)-volume(x-eps*v))/(2*eps)
    work=float((load.force(x,p)*v).sum())
    assert abs(fd-work)<1e-5*max(abs(work),1.)


def test_static_pressure_tangent_matches_finite_difference_for_blocks():
    x,load=case()
    x=x+.005*torch.sin(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x)
    v=torch.cos(torch.arange(x.numel(),device=x.device,dtype=x.dtype)).reshape_as(x)
    block=torch.stack((v,-.3*v),-1);eps=1e-6;p=8.
    actual=load.linearization(x,p)(block)
    for c in range(2):
        fd=(load.force(x+eps*block[...,c],p)-load.force(x-eps*block[...,c],p))/(2*eps)
        torch.testing.assert_close(actual[...,c],fd,rtol=3e-5,atol=1e-5)
