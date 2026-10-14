"""Actual multilevel AMG profile checks; preserve the same CN PDE equation."""
import pytest
import torch
from afsi_torch.ep import Monodomain,MonodomainNumerics
from afsi_torch.fluid.solvers import SolverOptions
from afsi_torch.geometry.tissue_slab import structured_tissue_slab


@pytest.mark.parametrize('profile',['size2','size4','size8','size8_two_level'])
def test_multilevel_amg_profiles_match_dense_cn(profile):
    if not torch.cuda.is_available():pytest.skip('actual CUDA required')
    mesh=structured_tissue_slab((.3,.2,.2),.05,device='cuda')
    X=mesh.X;f=X.new_tensor([1.,0.,0.]).expand(len(X),-1)
    s=X.new_tensor([0.,1.,0.]).expand(len(X),-1)
    ep=Monodomain(X,mesh.cells,f,s,numerics=MonodomainNumerics(amg_profile=profile,
        solver=SolverOptions(rtol=1e-11,atol=1e-14),time_scheme='crank_nicolson'))
    V=-.08+.04*torch.sin(X[:,0]*10)+.02*X[:,1]
    I=X[:,2]+.1;S=V.new_zeros(len(V));S[X[:,0]<.1]=25.
    dt=5e-5;M=ep.mass.to_dense();K=ep.stiffness.to_dense()
    rhs=M@(2000*V+.1*(S-I))-.5*K@V
    expected=torch.linalg.solve(2000*M+.5*K,rhs)
    result,info=ep.step(V,I,S,dt)
    torch.testing.assert_close(result,expected,rtol=2e-9,atol=2e-12)
    assert info.residual_norm<=info.tolerance
    assert ep._systems[dt][2].setup_count==1
    ep.close()
