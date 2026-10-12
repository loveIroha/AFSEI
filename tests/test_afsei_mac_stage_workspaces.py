"""Independent transforms/solves, fused rotation oracle and scratch ownership."""
import pytest
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.spectral_dirichlet import SpectralDirichletPressure
from afsi_torch.mac.mass_graph import GraphMassSolver
from afsi_torch.mac.mass_solver import MassSolver
from afsi_torch.fluid.solvers import SolverOptions
from afsi_torch.krylov_workspace import KrylovWorkspace
from afsi_torch.nonlinear import gmres,GMRESOptions

DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='actual CUDA required'))]


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('shape',[(4,7,9),(16,12,8)])
def test_reused_dst_matches_original_transforms_and_keeps_owned_pressures(device,shape):
    grid=MACGrid(shape,(2.,3.,4.))
    old=SpectralDirichletPressure(grid,device=device,workspace=False)
    new=SpectralDirichletPressure(grid,device=device)
    base=torch.arange(old.rhs.numel(),device=device,dtype=torch.float64).reshape(shape)
    owned=[]
    pointers=[v.data_ptr() for v in (new.workspace.real,new.workspace.frequency,*new.workspace.values)]
    for rhs in (torch.sin(base),torch.cos(3*base),torch.zeros_like(base)):
        expected,info=old.solve(rhs)
        actual,report=new.solve(rhs)
        torch.testing.assert_close(actual,expected,rtol=2e-11,atol=2e-12)
        assert report['residual_norm']<=report['tolerance']
        for previous,saved in owned: torch.testing.assert_close(previous,saved,rtol=0,atol=0)
        owned.append((actual,actual.clone()))
    assert pointers==[v.data_ptr() for v in (new.workspace.real,new.workspace.frequency,*new.workspace.values)]
    direct=new.inverse_operator(torch.sin(base)); saved=direct.clone()
    new.inverse_operator(torch.cos(base))
    torch.testing.assert_close(direct,saved,rtol=0,atol=0)


@pytest.mark.parametrize('device',DEVICES)
def test_dst_accepts_strided_rhs_and_rejects_incompatible_storage(device):
    grid=MACGrid((5,6,7),(2.,3.,4.))
    old=SpectralDirichletPressure(grid,device=device,workspace=False)
    new=SpectralDirichletPressure(grid,device=device)
    rhs=torch.arange(210,device=device,dtype=torch.float64).reshape(7,6,5).permute(2,1,0).sin()
    assert not rhs.is_contiguous()
    torch.testing.assert_close(new.inverse_operator(rhs),old.inverse_operator(rhs),rtol=2e-11,atol=2e-12)
    for bad in (rhs[:1],rhs.float()):
        with pytest.raises(ValueError,match='shape, dtype and device'):
            new.inverse_operator(bad)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_captured_mass_stages_match_original_schedule_and_recover_bad_inputs(device,dtype):
    n=17
    mass=(3*torch.eye(n,device=device,dtype=dtype)
          -torch.diag(torch.ones(n-1,device=device,dtype=dtype),1)
          -torch.diag(torch.ones(n-1,device=device,dtype=dtype),-1)).to_sparse_csr()
    diagonal=torch.full((n,3),3.,device=device,dtype=dtype)
    options=SolverOptions(rtol=1e-5 if dtype==torch.float32 else 1e-11,
                          atol=1e-7 if dtype==torch.float32 else 1e-13,
                          check_every=4,recompute_every=7,max_iterations=80)
    original=MassSolver(mass,diagonal,options); trial=GraphMassSolver(mass,diagonal,options)
    assert set(trial.stage_graphs)==({'zero','guess','true'} if device=='cuda' else set())
    base=torch.arange(n*3,device=device,dtype=dtype).reshape(n,3)
    rhs=torch.sin(base)
    for initial in (None,.1*torch.cos(base)):
        expected,old=original.solve(rhs,initial)
        actual,new=trial.solve(rhs,initial)
        torch.testing.assert_close(actual,expected,rtol=3e-5 if dtype==torch.float32 else 1e-9,
                                   atol=3e-6 if dtype==torch.float32 else 1e-11)
        assert new.iterations==old.iterations and new.residual_norm<=new.tolerance
    with pytest.raises(ValueError,match='finite'):
        trial.solve(torch.full_like(rhs,float('nan')))
    with pytest.raises(ValueError,match='finite'):
        trial.solve(rhs,torch.full_like(rhs,float('inf')))
    actual,info=trial.solve(rhs)
    assert info.residual_norm<=info.tolerance


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_single_launch_givens_matches_sequential_torch_oracle(dtype):
    reference=KrylovWorkspace(fused=False); trial=KrylovWorkspace()
    sample=torch.empty(64,device='cuda',dtype=dtype)
    reference.ensure(sample,50); trial.ensure(sample,50)
    base=torch.arange(51*50,device='cuda',dtype=dtype).reshape(51,50)
    reference.H.copy_(torch.sin(base*.017)); trial.H.copy_(reference.H)
    beta=sample.new_tensor(2.)
    reference.reset(50,beta); trial.reset(50,beta)
    reference.H.copy_(torch.sin(base*.017)); trial.H.copy_(reference.H)
    for j in range(50):
        expected=reference.rotate(reference.H,reference.rotated,reference.cosines,reference.sines,reference.g,j)
        actual=trial.rotate(trial.H,trial.rotated,trial.cosines,trial.sines,trial.g,j)
        tol=2e-5 if dtype==torch.float32 else 3e-13
        torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
        torch.testing.assert_close(trial.rotated[:j+2,j],reference.rotated[:j+2,j],rtol=tol,atol=tol)
    # Degenerate and very large/small columns exercise robust hypot/clamping.
    for value in (0.,1e20 if dtype==torch.float32 else 1e250,1e-20 if dtype==torch.float32 else 1e-250):
        for workspace in (reference,trial):
            workspace.reset(50,beta); workspace.H[:2,0].fill_(value)
            workspace.rotate(workspace.H,workspace.rotated,workspace.cosines,workspace.sines,workspace.g,0)
        torch.testing.assert_close(trial.estimate,reference.estimate,rtol=tol,atol=tol)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('policy',['periodic','estimated'])
def test_reused_krylov_scratch_handles_restarts_and_never_leaks_old_columns(device,policy):
    n=29
    A=torch.diag(torch.linspace(1.,3.,n,device=device,dtype=torch.float64))
    A+=torch.diag(torch.full((n-1,),.35,device=device,dtype=A.dtype),1)
    rhs=torch.sin(torch.arange(n,device=device,dtype=A.dtype))
    options=GMRESOptions(restart=5,rtol=1e-10,atol=1e-13,
                         check_policy=policy,check_every=2,true_check_interval=7)
    workspace=KrylovWorkspace()
    first,info=gmres(lambda v:A@v,rhs,options=options,workspace=workspace)
    saved=first.clone(); addresses=[workspace.V.data_ptr(),workspace.Z.data_ptr(),workspace.work.data_ptr()]
    assert info['iterations']>5 and info['residual_norm']<=info['tolerance']
    workspace.V.fill_(float('nan')); workspace.Z.fill_(float('nan'))
    second,report=gmres(lambda v:A@v,-rhs,options=options,workspace=workspace)
    torch.testing.assert_close(second,torch.linalg.solve(A,-rhs),rtol=2e-9,atol=2e-10)
    torch.testing.assert_close(first,saved,rtol=0,atol=0)
    assert addresses==[workspace.V.data_ptr(),workspace.Z.data_ptr(),workspace.work.data_ptr()]
    assert report['residual_norm']<=report['tolerance']
    zero,info=gmres(lambda v:A@v,torch.zeros_like(rhs),options=options,workspace=workspace)
    assert zero.count_nonzero()==0 and info['iterations']==0
