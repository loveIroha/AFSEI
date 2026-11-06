"""Fixed upper outlets / trimmed basal collar, including exact moving storage."""
import os
from types import SimpleNamespace
import pytest
import torch
from afsi_torch.ep.gao_tracts import LVTractDomain
from afsi_torch.ep.tract_storage import TractStorage
from afsi_torch.mac.grid import MACGrid


def case():
    device=os.environ.get('AFSEI_TEST_DEVICE','cpu')
    x=torch.tensor([[2.18,1.3,2.],[2.45,1.6,2.],[1.4,1.3,2.],[1.82,1.6,2.]],
                   dtype=torch.float64,device=device)
    cap=SimpleNamespace(frame=torch.eye(3,device=device,dtype=x.dtype),
        labels=torch.tensor([1,1,2,2],device=device),surface=lambda y:(y,torch.ones_like(y)))
    grid=MACGrid((16,16,16),(4.,4.,4.))
    old=LVTractDomain(grid,cap,x,mode='rcr',shared_collar_cells=3)
    new=LVTractDomain(grid,cap,x,mode='rcr',shared_collar_cells=3,neck_corner_trim_cells=1)
    return x,cap,old,new


def test_trim_keeps_upper_cut_and_seals_ledge_but_opens_lower_neck():
    x,cap,old,new=case()
    assert old.bounds==new.bounds
    assert old.metadata['cut_areas_cm2']==new.metadata['cut_areas_cm2']
    assert all(torch.equal(a,b) for a,b in zip(old.apertures,new.apertures))
    assert sum(int(a.sum()) for a in old.neck_apertures)-sum(int(a.sum()) for a in new.neck_apertures)==4
    # One removed lower corner is an upper duct cell with a sealed floor.
    assert new.label[5,4,8]==0 and new.label[5,4,12]==2
    assert new.masks[2][5,4,new.shared_top_index]==0
    assert new.masks[2][6,5,new.bottom_index]==1
    for a,b in zip(old.masks,new.masks):
        assert torch.equal(a[:,:,13:],b[:,:,13:])
    new.validate_wall(cap,x)
    y=x.clone();y[2,:2]=y.new_tensor([1.375,1.125])
    with pytest.raises(ValueError,match='shared collar'):new.validate_wall(cap,y)
    # A seam in the rectangular integration partition is an interior point.
    y=x.clone();y[2,:2]=y.new_tensor([1.5,1.375]);new.validate_wall(cap,y)
    y[2,:2]=y.new_tensor([1.5,1.125])
    with pytest.raises(ValueError,match='shared collar'):new.validate_wall(cap,y)


def test_lower_flux_budget_excludes_removed_corner_cells():
    x,cap,old,new=case();g=new.grid
    u=list(g.zeros(device=x.device));u[2][5,4,new.bottom_index]=3.
    cap.material_points=lambda value:value
    prepared=SimpleNamespace(points=x,area_vectors=torch.zeros_like(x),cap=cap)
    assert new.budget(tuple(u),prepared,torch.zeros_like(x))['fixed_tract_mass_error_ml_s']==0
    assert old.budget(tuple(u),prepared,torch.zeros_like(x))['fixed_tract_mass_error_ml_s'].abs()>0


@pytest.mark.parametrize('gpu',[False,True])
def test_trimmed_neck_volume_work_and_geometry_derivative(gpu):
    if gpu and not torch.cuda.is_available():pytest.skip('requires actual CUDA')
    x,_,_,domain=case()
    # Broad planar cap, moving below the fixed upper-to-lower transition.
    y=x.new_tensor([[0.,0.,2.],[3.,0.,2.],[3.,3.,2.],[0.,3.,2.]])
    if gpu:y=y.cuda()
    cap=SimpleNamespace(rim=torch.tensor([[0,1],[1,2],[2,3],[3,0]],device=y.device))
    if gpu:
        from afsi_torch.ep.tract_storage_gpu import GPUTractStorage
        s=GPUTractStorage(cap,domain)
    else:s=TractStorage(cap,domain)
    cpu=TractStorage(cap,domain)
    top_area=sum(domain.metadata['cut_areas_cm2']);neck_area=sum(domain.metadata['neck_areas_cm2'])
    height=domain.storage_top_coordinate_cm
    expected=top_area*(4-height)+neck_area*(height-2)
    assert abs(s.volume(y)-expected)<2e-13
    z=y.clone();z[:,2]+=.03
    assert abs(s.volume(z)-expected+.03*neck_area)<2e-13
    torch.testing.assert_close(s.discrete(y,z),cpu.discrete(y,z),rtol=1e-9,atol=2e-12)
    assert abs(float((s.discrete(y,z)*(z-y)).sum())+.03*neck_area)<2e-13
    direction=torch.cos(torch.arange(y.numel(),device=y.device,dtype=y.dtype)).reshape_as(y)
    z=y+.005*direction;action=s.derivative(y,z);eps=1e-5
    fd=(s.discrete(y,z+eps*direction)-s.discrete(y,z-eps*direction))/(2*eps)
    torch.testing.assert_close(action(direction),fd,rtol=2e-4,atol=2e-9)
    assert abs(float((s.discrete(y,z)*(z-y)).sum())-(s.volume(z)-s.volume(y)))<2e-12
    # A cap truly detached from a neck still raises; no tolerance was relaxed.
    z[:,0]+=10.
    with pytest.raises(ValueError,match='no longer covers'):s.volume(z)
