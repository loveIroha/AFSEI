import pytest
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.ep.mac_snapshot_diagnostics import CellSampler,normal_viscous_stress


def test_staggered_affine_pressure_and_rotated_normal_stress():
    grid=MACGrid((10,12,14),(2.,3.,4.),(-1.,-2.,-3.))
    points=torch.tensor([[.13,-.23,-.36],[-.21,.14,.17]],dtype=torch.float64)
    sampler=CellSampler(grid,points)
    cells=grid.coordinates()
    slope=torch.tensor([1.7,-2.3,.9],dtype=torch.float64)
    field=7.+cells@slope
    torch.testing.assert_close(sampler(field),7.+points@slope,rtol=1e-12,atol=1e-12)
    A=torch.tensor([[.2,1.1,-.3],[-.7,.4,.8],[.6,-.2,-.6]],dtype=torch.float64)
    shift=torch.tensor([3.,-5.,2.],dtype=torch.float64)
    velocity=tuple((grid.coordinates(c)@A[c])+shift[c] for c in range(3))
    n=torch.tensor([[1.,2.,3.],[-2.,1.,.5]],dtype=torch.float64)
    n/=n.norm(dim=-1,keepdim=True)
    mu=.04
    expected=2*mu*torch.einsum('ni,ij,nj->n',n,A,n)
    torch.testing.assert_close(normal_viscous_stress(grid,velocity,n,sampler,mu),expected,
                               rtol=1e-11,atol=1e-12)


def test_layer_sampling_rejects_silent_boundary_clipping():
    grid=MACGrid((8,8,8),(1.,1.,1.))
    with pytest.raises(ValueError,match='interior cell support'):
        CellSampler(grid,torch.tensor([[.01,.5,.5]],dtype=torch.float64))
