from dataclasses import replace
import pytest
import torch
from afsi_torch.ep.monodomain import Monodomain,_assemble
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.p1 import prepare_p1


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('kind',['nodal_p1','consistent_p1'])
@pytest.mark.parametrize('dim',[2,3])
def test_current_assembly_and_cn_equivalence(device,kind,dim):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    X=torch.cat((torch.zeros(1,dim),torch.eye(dim))).to(device=device,dtype=torch.float64)
    cells=torch.arange(dim+1,device=device)[None]
    f=torch.zeros_like(X);s=f.clone();f[:,0]=1;s[:,1]=1
    ep=Monodomain(X,cells,f,s,numerics=MonodomainNumerics(discretization=kind))
    K0=ep.stiffness.to_dense().clone();M0=ep.mass.to_dense().clone()
    F=torch.eye(dim,device=device,dtype=X.dtype);F[0,0]=1.2;F[1,1]=.9;F[0,1]=.25
    x=X@F.T+.3;ep.prepare(5e-5);ep.update_deformation(x)
    geo=prepare_p1(x,cells,degree=1);g=geo.gradients[:,0];vol=geo.weights.sum(1)
    Dd=ep.spatial_conductivity().mean(1)
    literal=_assemble(cells,torch.einsum('e,eai,eij,ebj->eab',vol,g,Dd,g),len(X)).to_dense()
    torch.testing.assert_close(literal,K0,rtol=2e-12,atol=2e-14)
    torch.testing.assert_close(ep.mass.to_dense(),torch.linalg.det(F)*M0,rtol=2e-12,atol=2e-14)
    assert not ep._systems
    v=torch.linspace(-.08,.02,len(X),device=device,dtype=X.dtype)
    zero=torch.zeros_like(v);dt=5e-5
    actual,info=ep.step(v,zero,zero,dt)
    alpha=ep.parameters.capacitance_uf_per_cm2*ep.parameters.surface_to_volume_per_cm/dt
    M=ep.mass.to_dense()
    expected=torch.linalg.solve(alpha*M+.5*literal,(alpha*M-.5*literal)@v)
    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-11)
    assert not ep.update_deformation(x)
    before=ep.mass.to_dense().clone()
    bad=x.clone();bad[:,0]*=-1
    with pytest.raises(ValueError,match='inverted'):ep.update_deformation(bad)
    torch.testing.assert_close(ep.mass.to_dense(),before,rtol=0,atol=0)
    # Rigid rotation and volume-preserving shear must not alter material-node PDE.
    R=torch.eye(dim,device=device,dtype=X.dtype);R[0,0]=0;R[0,1]=-1;R[1,0]=1;R[1,1]=0
    ep.update_deformation(X@R.T)
    torch.testing.assert_close(ep.mass.to_dense(),M0,rtol=2e-12,atol=2e-14)
    ep.close()


from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_isovolumic import precise_driver
from test_afsei_driver import assert_same
from afsi_torch.ep.mac_isovolumic import IsovolumicAFSEIStepper
from afsi_torch.ep.checkpoint import save,load


def test_accepted_geometry_driver_restart(mac_case,tmp_path):
    base=precise_driver(mac_case)
    driver=IsovolumicAFSEIStepper(base.mesh,replace(base.config,ep_deformation_feedback='santiago2018'))
    old=driver.lock(driver.initialize())
    new,info=driver.step(old)
    assert info['ep']['steps']>0
    F=torch.einsum('eai,eaJ->eiJ',old.fsi.x[driver.ep.cells],driver.ep.geometry.gradients[:,0])
    torch.testing.assert_close(driver.ep.deformation_gradient,F)
    path=tmp_path/'feedback.npz';save(path,driver,new)
    restored,state,_=load(path)
    assert restored.config.ep_deformation_feedback=='santiago2018'
    a,_=driver.step(new);b,_=restored.step(state)
    assert_same(a,b)
