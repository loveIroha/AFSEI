from types import SimpleNamespace as NS
import pytest
import torch
from afsi_torch.mac.basal_support import BasalSupport,SupportedBEProblem
from afsi_torch.nonlinear import newton,NewtonOptions,GMRESOptions


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_support_virtual_work_radial_freedom_and_physical_reactions(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA required')
    X=torch.tensor([[1.,1.,1.],[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]],dtype=torch.float64,device=device)
    cells=torch.tensor([[0,1,2,3]],device=device);faces=cells[:,:3]
    b=BasalSupport(X,faces,(0.,0.))
    radial=X.clone();radial[:,2]=0
    assert b.apply(radial).abs().max()<1e-14
    u=torch.sin(X);lam=torch.arange(6,dtype=X.dtype,device=device).reshape(-1,2)
    assert float((b.transpose(lam)*u).sum())==pytest.approx(float((lam*b.apply(u)).sum()),abs=1e-12)
    assert b.transpose(lam)[3].abs().max()==0  # apex unsupported
    # Non-diagonal fluid mobility, independent direct saddle-point reference.
    n=X.numel();dt=.01;k=10.
    A=torch.sin(torch.arange(n*n,dtype=X.dtype,device=device)).reshape(n,n)
    M=A@A.T/n+torch.eye(n,dtype=X.dtype,device=device)
    forcing=torch.cos(X)*3.;departure=torch.sin(X)*.1
    flow=NS(dt=dt,rho=1.)
    flow.advance=lambda dep,f,p:(dep+dt*(M@f.flatten()).reshape_as(X),X.new_zeros(1),{})
    flow.response=lambda f:dt*(M@f.flatten()).reshape_as(X)
    transfer=NS(reset_warm_start=lambda:None,spread=lambda f,s:(f,None),interpolate=lambda u,s:(u,None))
    tangent=(-k*torch.eye(n,dtype=X.dtype,device=device)).to_sparse_csr()
    model=NS(beta=0.,mesh=NS(X=X,cells=cells),tangent_factory=lambda:NS(assemble=lambda x,t:tangent))
    solid=NS(validate=lambda x:None,force=lambda x,t:forcing-k*(x-X))
    d=NS(flow=flow,transfer=transfer,model=model,solid=solid,assembler=None,basal_support=b,
         config=NS(nonlinear_solver='newton'),support_tolerance_cm=1e-10)
    state=NS(x=X.clone(),step=0,pressure=None)
    problem=SupportedBEProblem(d,state,None,departure)
    z=problem.initial(torch.zeros_like(X))
    result=newton(problem.residual,z,validate=problem.validate,linearization_factory=problem.linearization,
        options=NewtonOptions(atol=1e-12,rtol=1e-10,max_iterations=5,
            linear=GMRESOptions(atol=1e-13,rtol=1e-9,max_iterations=100,restart=30)))
    residual,x=problem.accepted(result.x)
    B=torch.stack([b.apply(torch.eye(n,dtype=X.dtype,device=device)[i].reshape_as(X)).flatten() for i in range(n)],1)
    K=X.new_zeros((n+6,n+6));K[:n,:n]=torch.eye(n,device=device,dtype=X.dtype)+dt**2*k*M
    K[:n,n:]=-dt**2*M@B.T;K[n:,:n]=B
    rhs=torch.cat((dt*departure.flatten()+dt**2*M@forcing.flatten(),X.new_zeros(6)))
    reference=torch.linalg.solve(K,rhs)
    torch.testing.assert_close((x-X).flatten(),reference[:n],atol=1e-10,rtol=1e-8)
    torch.testing.assert_close(problem.last_reaction.flatten(),B.T@reference[n:],atol=1e-8,rtol=1e-8)
    assert float(b.error(x).abs().max())<1e-10
    assert abs(d.last_support_report['support_power_erg_s'])<1e-8
    assert float((x-X)[b.nodes,:2].norm())>1e-6
    # Directional derivative of the complete displacement/reaction residual.
    v=torch.sin(torch.arange(z.numel(),device=device,dtype=X.dtype));base=result.x
    action=problem.linearization(base)(v);eps=1e-6
    fd=(problem.residual(base+eps*v)-problem.residual(base-eps*v))/(2*eps)
    torch.testing.assert_close(action,fd,atol=2e-9,rtol=2e-8)
    # Old drifting states must not be silently projected into a new support.
    state.x=state.x+X.new_tensor([0.,0.,.001])
    with pytest.raises(ValueError,match='fresh preload'):SupportedBEProblem(d,state,None,departure)
