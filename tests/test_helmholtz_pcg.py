import json
from pathlib import Path
import pytest
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.open_boundary import velocity_laplacian,velocity_diagonal
from afsi_torch.mac.helmholtz_pcg import BEViscousPCG


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    return request.param


def example(device,alpha=49.):
    grid=MACGrid((8,8,8),(.008,.008,.008))
    like=torch.empty((),device=device,dtype=torch.float64)
    masks=list(torch.ones(grid.face_shape(c),device=device,dtype=like.dtype) for c in range(3))
    for c,m in enumerate(masks):m.select(c,3).fill_(0.)
    masks=tuple(masks)
    dt=5e-5;rho=1.;mu=alpha*grid.spacing[0]**2/dt
    diagonals=tuple(1+dt*mu*velocity_diagonal(grid.face_shape(c),c,grid.spacing,like) for c in range(3))
    solver=BEViscousPCG(grid,dt,rho,mu,masks,diagonals,max_iterations=2000)
    return grid,like,masks,solver


def test_weighted_operator_is_symmetric_on_free_faces(device):
    _,like,masks,s=example(device)
    torch.manual_seed(9)
    a=torch.randn_like(s.diagonal).masked_fill(s.fixed,0)
    b=torch.randn_like(a).masked_fill(s.fixed,0)
    torch.testing.assert_close((a*s.action(b)).sum(),(b*s.action(a)).sum(),atol=2e-10,rtol=2e-12)
    assert float((a*s.action(a)).sum())>0


def test_manufactured_velocity_preserves_original_component_targets(device):
    grid,like,masks,s=example(device)
    torch.manual_seed(10)
    exact=tuple(m*torch.randn_like(m)*scale for m,scale in zip(masks,(1e-2,1e-5,1e-8)))
    rhs=tuple(m*(x-s.alpha*velocity_laplacian(x,c,grid.spacing)) for c,(x,m) in enumerate(zip(exact,masks)))
    targets=[max(1e-15,1e-11*float(b.norm())) for b in rhs]
    got,info=s.solve(rhs,targets)
    for c,(x,y,b,m,tol) in enumerate(zip(got,exact,rhs,masks,targets)):
        residual=m*(b-x+s.alpha*velocity_laplacian(x,c,grid.spacing))
        assert float(residual.norm())<=tol
        # H >= I in the half-face weighted norm. A residual target controls
        # absolute L2 error, not relative pointwise error near zeros.
        assert float((x-y).norm())<=2*tol
        assert not x[m==0].count_nonzero()
    assert info.iterations<1000


def test_zero_rhs_and_constant_open_face_solution(device):
    grid,like,_,_=example(device)
    masks=tuple(torch.ones(grid.face_shape(c),device=device,dtype=like.dtype) for c in range(3))
    dt=5e-5;mu=.98
    diagonal=tuple(1+dt*mu*velocity_diagonal(grid.face_shape(c),c,grid.spacing,like) for c in range(3))
    s=BEViscousPCG(grid,dt,1.,mu,masks,diagonal)
    rhs=tuple(torch.full_like(m,value) for m,value in zip(masks,(.1,0.,-.001)))
    u,_=s.solve(rhs,[1e-10,1e-15,1e-12])
    for x,b in zip(u,rhs):torch.testing.assert_close(x,b,rtol=1e-10,atol=1e-12)


def test_viscous_solver_config_is_explicit():
    from afsi_torch.simulation.circular_tube_config import resolve_config,config_summary
    path=Path(__file__).resolve().parents[1]/'demo/active_tube_3d/configs/tube_sharifi_micron_layered_gpu.json'
    c=json.loads(path.read_text());c['fsi']['flow']['helmholtz_solver']='pcg'
    resolve_config(c)
    assert config_summary(c)['viscous_linear_solver']=='pcg'
    c['fsi']['flow']['helmholtz_solver']='invalid'
    with pytest.raises(ValueError,match='helmholtz_solver'):resolve_config(c)
    c['fsi']['flow']['helmholtz_solver']='pcg'
    c.pop('hydraulic_domain');c.pop('circulation')
    c['fluid']['boundary']='zero-pressure-neumann-velocity-all-box-faces'
    with pytest.raises(ValueError,match='sealed tube'):resolve_config(c)
