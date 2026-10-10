"""Independent matrix, eigenmodes and original projection equivalence."""
import pytest
import torch
from afsi_torch.mac.grid import MACGrid,divergence
from afsi_torch.mac.open_boundary import negative_laplacian
from afsi_torch.mac.spectral_dirichlet import SpectralDirichletPressure
from afsi_torch.mac.dirichlet_mg import DirichletMultigrid
from afsi_torch.mac.backward_euler import BackwardEulerFlow

DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required'))]

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('shape',[(4,5,6),(8,8,8)])
def test_exact_original_pressure_matrix(device,shape):
    grid=MACGrid(shape,(2.,3.,4.))
    solver=SpectralDirichletPressure(grid,device=device)
    n=torch.tensor(shape).prod().item()
    eye=torch.eye(n,device=device,dtype=torch.float64)
    matrix=torch.vmap(lambda p:negative_laplacian(p.reshape(shape),grid.spacing).flatten())(eye).T
    rhs=torch.cos(torch.arange(n,device=device,dtype=eye.dtype)).reshape(shape)
    actual,info=solver.solve(rhs)
    expected=torch.linalg.solve(matrix,rhs.flatten()).reshape(shape)
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-12)
    assert info['residual_norm'] <= info['tolerance']
    zero,_=solver.solve(torch.zeros_like(rhs))
    assert torch.count_nonzero(zero)==0
    with pytest.raises(ValueError): solver.solve(rhs.float())
    with pytest.raises(ValueError): solver.solve(rhs*float('nan'))


@pytest.mark.parametrize('device',DEVICES)
def test_low_and_high_cell_centred_sine_modes(device):
    grid=MACGrid((16,12,8),(13.,9.,7.))
    solver=SpectralDirichletPressure(grid,device=device)
    for modes in ((1,1,1),grid.shape,(3,5,7)):
        p=solver.rhs.new_ones(grid.shape); eigenvalue=0.
        for axis,(n,h,k) in enumerate(zip(grid.shape,grid.spacing,modes)):
            j=torch.arange(n,device=device,dtype=p.dtype)
            s=[1,1,1]; s[axis]=n
            p*=torch.sin(torch.pi*k*(j+.5)/n).reshape(s)
            eigenvalue+=4*torch.sin(p.new_tensor(torch.pi*k/(2*n)))**2/h**2
        torch.testing.assert_close(negative_laplacian(p,grid.spacing),eigenvalue*p,rtol=2e-12,atol=2e-12)
        actual,_=solver.solve(eigenvalue*p)
        torch.testing.assert_close(actual,p,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('device',DEVICES)
def test_be_pressure_projection_matches_multigrid(device):
    grid=MACGrid((16,16,16),(3.,4.,5.))
    flow=BackwardEulerFlow(grid,dt=.0001,device=device)
    rhs=tuple(torch.sin(torch.arange(u.numel(),device=device,dtype=u.dtype)).reshape_as(u)*.01
              for u in grid.zeros(device=device))
    a,p,_=flow.solve_rhs(rhs)
    flow.pressure_solver=SpectralDirichletPressure(grid,device=device,options=flow.pressure_solver.options)
    b,q,info=flow.solve_rhs(rhs)
    for u,v in zip(a,b): torch.testing.assert_close(u,v,rtol=1e-8,atol=2e-11)
    torch.testing.assert_close(p,q,rtol=2e-7,atol=2e-7)
    assert divergence(b,grid.spacing).norm()<1e-11
    assert info['pressure']['residual_norm'] <= info['pressure']['tolerance']
