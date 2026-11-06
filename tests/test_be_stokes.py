"""Coupled BE pressure/velocity checked against an independent dense system."""
import pytest
import torch
from afsi_torch.mac.be_stokes import BEStokesFlow
from afsi_torch.mac.grid import MACGrid,divergence
from afsi_torch.mac.open_boundary import gradient,velocity_laplacian
from afsi_torch.mac.backward_euler import BEFlowOptions
from afsi_torch.mac.multigrid import MGOptions


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required'))])
def test_be_stokes_matches_dense_open_boundary_system(device):
    grid=MACGrid((4,4,4),(1.,1.,1.))
    flow=BEStokesFlow(grid,dt=.03,mu=.7,device=device,
        options=BEFlowOptions(convection=False,helmholtz_rtol=1e-13,helmholtz_atol=1e-14),
        pressure_options=MGOptions(rtol=1e-10,atol=1e-12))
    like=torch.zeros((),device=device,dtype=torch.float64)
    shapes=[grid.face_shape(j) for j in range(3)]+[grid.shape]
    sizes=[torch.Size(s).numel() for s in shapes]; n=sum(sizes)
    def unpack(v): return [a.reshape(s) for a,s in zip(v.split(sizes),shapes)]
    def pack(parts): return torch.cat([a.flatten() for a in parts])
    def operator(v):
        *u,p=unpack(v); g=gradient(p,grid.spacing)
        return pack([v-flow.alpha*velocity_laplacian(v,j,grid.spacing)+flow.dt*g[j] for j,v in enumerate(u)]+[divergence(u,grid.spacing)])
    eye=torch.eye(n,device=device,dtype=like.dtype)
    matrix=torch.stack([operator(col) for col in eye],dim=1)
    b=tuple(torch.sin(torch.arange(k,device=device,dtype=like.dtype)+j).reshape(s) for j,(k,s) in enumerate(zip(sizes[:3],shapes[:3])))
    exact=torch.linalg.solve(matrix,pack([*b,like.new_zeros(grid.shape)]))
    u,p,info=flow.solve_rhs(b)
    torch.testing.assert_close(pack([*u,p]),exact,rtol=2e-8,atol=2e-9)
    assert info['stokes']['momentum_residual']<=info['stokes']['momentum_tolerance']
    assert info['stokes']['divergence_norm']<=info['stokes']['divergence_tolerance']
    # A pressure gradient must not create a spurious velocity, including the
    # physical box faces. No artificial pressure-mean subtraction is valid.
    hydro=(2+torch.sin(torch.arange(64,device=device,dtype=like.dtype))).reshape(grid.shape)
    u,p,info=flow.solve_rhs(tuple(flow.dt*g for g in gradient(hydro,grid.spacing)))
    assert max(v.abs().max().item() for v in u)<1e-9
    torch.testing.assert_close(p,hydro,rtol=1e-8,atol=1e-8)
