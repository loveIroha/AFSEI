import pytest
import torch
from afsi_torch.ep import Monodomain, MonodomainParameters
from afsi_torch.fluid.solvers import SolverOptions


@pytest.fixture(params=['cpu', 'cuda'])
def mesh(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    X = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]],
                     dtype=torch.float64, device=request.param)
    cells = torch.tensor([[0,1,2,3],[4,1,2,3]],device=request.param)
    f = X.new_tensor([1.,0.,0.]).expand(len(X),-1)
    s = X.new_tensor([0.,1.,0.]).expand(len(X),-1)
    return X,cells,f,s


def test_exact_mass_stiffness_and_dense_independent_solution(mesh):
    X,cells,f,s = mesh
    p = Monodomain(X,cells,f,s,solver_options=SolverOptions(rtol=1e-13,atol=1e-14))
    # Independent affine-tetrahedron reference, without production FE tables.
    M = torch.zeros((5,5),dtype=X.dtype)
    K = torch.zeros_like(M)
    D = torch.diag(torch.tensor([.117,.039,.039],dtype=X.dtype))
    for cell in cells.cpu():
        nodes = X.cpu()[cell]
        affine = torch.cat((torch.ones((4,1),dtype=X.dtype),nodes),1)
        grads = torch.linalg.inv(affine)[1:].T
        vol = torch.linalg.det((nodes[1:]-nodes[:1]).T).abs()/6
        mass = vol/20*(torch.ones((4,4),dtype=X.dtype)+torch.eye(4,dtype=X.dtype))
        stiffness = vol*grads@D@grads.T
        M[cell[:,None],cell[None,:]] += mass
        K[cell[:,None],cell[None,:]] += stiffness
    torch.testing.assert_close(p.mass.to_dense().cpu(),M,rtol=1e-13,atol=1e-15)
    torch.testing.assert_close(p.stiffness.to_dense().cpu(),K,rtol=1e-13,atol=1e-15)
    voltage = X[:,0]*.05-.08
    ionic = X[:,1]*3+.2
    stimulus = X[:,2]*25
    dt = 5e-5
    expected = torch.linalg.solve(.1/dt*M+K,M@(.1/dt*voltage.cpu()+.1*(stimulus-ionic).cpu()))
    result,info = p.step(voltage,ionic,stimulus,dt)
    torch.testing.assert_close(result.cpu(),expected,rtol=1e-10,atol=1e-12)
    assert info.residual_norm <= info.tolerance


def test_insulating_conservation_and_uniform_current_units(mesh):
    X,cells,f,s = mesh
    p = Monodomain(X,cells,f,s,solver_options=SolverOptions(rtol=1e-13,atol=1e-14))
    v = X.new_full((5,),-.081)
    result,_ = p.step(v,v.new_full(v.shape,2),v.new_full(v.shape,25),5e-5)
    torch.testing.assert_close(result,v+5e-5*(25-2),rtol=1e-11,atol=1e-13)
    z = torch.zeros_like(v)
    nonuniform = v+X[:,0]*.05
    result,_ = p.step(nonuniform,z,z,.02)
    mass = p.mass
    total = lambda value: torch.sparse.mm(mass,value[:,None]).sum()
    torch.testing.assert_close(total(result),total(nonuniform),rtol=1e-10,atol=1e-13)
    assert torch.linalg.vector_norm(result-result.mean()) < torch.linalg.vector_norm(nonuniform-nonuniform.mean())
    projected = p.project_cell_field(X.new_full((2,),1.13))
    torch.testing.assert_close(projected,X.new_full((5,),1.13),rtol=1e-10,atol=1e-12)


def test_nodal_directions_remain_spatially_variable(mesh):
    X,cells,f,s = mesh
    f = f.clone(); f[:,0] = 1+X[:,1]*.2
    p = Monodomain(X,cells,f,s)
    # f=(1+.2*y,0,0), n=(0,0,1+.2*y): integrate exact y/y^2 moments.
    gradients = p.geometry.gradients[:,0].cpu()
    ref = torch.zeros((5,5),dtype=X.dtype)
    for e,cell in enumerate(cells.cpu()):
        yy = X.cpu()[cell,1]
        mean_y = yy.sum()/4
        mean_y2 = (yy.square().sum()+yy.sum().square())/20
        factor = 1+.4*mean_y+.04*mean_y2
        D = torch.diag(torch.tensor([.117*factor,.039,.039*factor],dtype=X.dtype))
        v = p.geometry.weights[e].sum().cpu()
        ref[cell[:,None],cell[None,:]] += v*gradients[e]@D@gradients[e].T
    torch.testing.assert_close(p.stiffness.to_dense().cpu(),ref,rtol=1e-12,atol=1e-14)


def test_invalid_setup_and_nonconvergence(mesh):
    X,cells,f,s = mesh
    with pytest.raises(ValueError): Monodomain(X,cells,f,s,directions='ambiguous')
    with pytest.raises(ValueError): Monodomain(X,cells,f,f)
    with pytest.raises(ValueError): MonodomainParameters(conductivity_us_per_cm=(0.,.1,.1))
    p = Monodomain(X,cells,f,s,solver_options=SolverOptions(rtol=1e-15,atol=1e-18,max_iterations=1))
    z = X.new_zeros(5)
    with pytest.raises(RuntimeError,match='did not converge'):
        p.step(X[:,0]-.08,z,X[:,2],.1)
