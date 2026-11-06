from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from afsi_torch.ep.refined_mesh import nested_mesh,NestedP1Transfer,RefinedMonodomain
from afsi_torch.ep.monodomain import Monodomain
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.ep.buffered_load import BufferedLandLoad,ActiveLoadState
from afsi_torch.fluid.solvers import SolverOptions
from afsi_torch.p1 import prepare_p1


@pytest.fixture(params=['cpu','cuda'])
def refined(request):
    if request.param=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    X=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]],
        dtype=torch.float64,device=request.param)
    cells=torch.tensor([[0,1,2,3],[4,2,1,3]],device=request.param)
    data=nested_mesh(X.cpu().numpy(),cells.cpu().numpy(),3)
    g=prepare_p1(X,cells,5)
    transfer=NestedP1Transfer(data,g)
    return X,cells,data,g,transfer


def test_conforming_factor_three_and_exact_P1_transfer(refined):
    X,cells,data,g,transfer=refined
    assert len(data['cells'])==27*len(cells)
    points=data['X'][data['cells']]
    volume=np.abs(np.linalg.det(points[:,1:]-points[:,:1]))/6
    expected=g.weights.sum(1).cpu().numpy()
    np.testing.assert_allclose(volume.reshape(-1,27),np.broadcast_to(expected[:,None]/27,(len(cells),27)),rtol=2e-14)
    faces=np.concatenate([data['cells'][:,idx] for idx in ([0,1,2],[0,1,3],[0,2,3],[1,2,3])])
    _,counts=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True)
    assert np.all((counts==1)|(counts==2))
    assert np.count_nonzero(counts==1)==6*9  # two tetrahedra share one face
    affine=2+X[:,0]*.3-X[:,1]*.7+X[:,2]*.2
    q=transfer.prolong(affine)
    fineX=torch.as_tensor(data['X'],device=X.device)
    torch.testing.assert_close(q,2+fineX[:,0]*.3-fineX[:,1]*.7+fineX[:,2]*.2,rtol=1e-14,atol=1e-14)
    coarse=torch.einsum('qa,ea->eq',g.values,affine[cells])
    torch.testing.assert_close(transfer.quadrature(q),coarse,rtol=1e-14,atol=1e-14)
    # A fine-only hat contributes at structural quadrature even though it is
    # zero at every original structural node: no coarse-node downsampling.
    hat=torch.zeros_like(q);hat[len(X):]=1
    assert transfer.quadrature(hat).max()>.5


def test_refined_active_force_and_tangent_preserve_affine_fields(refined):
    X,cells,data,g,transfer=refined
    fiber=X.new_tensor([1.,0.,0.]).expand(len(cells),-1)
    old=BufferedLandLoad(g,fiber);new=BufferedLandLoad(g,fiber);new.field_transfer=transfer
    fields=[X[:,0]*.001+.002,X[:,1]*.002+.003,X[:,2]*.01,X[:,0]*.02]
    load=ActiveLoadState(X.clone(),*fields,1e-4)
    old.stage(load)
    newload=replace(load,**{key:transfer.prolong(value) for key,value in
        zip(('xs','xw','zs_old','zw_old'),fields)})
    new.stage(newload)
    x=X*1.03;direction=X*.013+.001
    torch.testing.assert_close(new.fields,old.fields,rtol=1e-13,atol=1e-14)
    torch.testing.assert_close(new.force(x),old.force(x),rtol=1e-12,atol=1e-8)
    torch.testing.assert_close(new.force_jvp(x,direction),old.force_jvp(x,direction),rtol=1e-11,atol=1e-7)
    mesh=SimpleNamespace(X=X,cells=cells,fiber=fiber)
    immutable=newload.materialize(mesh,transfer)
    torch.testing.assert_close(immutable.force(x),new.force(x),rtol=1e-12,atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA/AmgX required')
def test_shared_diffusion_mass_choice_and_fine_operator(tmp_path):
    X=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]],dtype=torch.float64,device='cuda')
    cells=torch.tensor([[0,1,2,3],[4,2,1,3]],device='cuda')
    f=torch.stack((1+.2*X[:,1],X[:,0]*.03,X[:,2]*0),1)
    s=X.new_tensor([0.,1.,0.]).expand(len(X),-1)
    stim=X[:,0]==0
    mesh=SimpleNamespace(X=X,cells=cells)
    opt=SolverOptions(rtol=1e-11,atol=1e-14,max_iterations=1000)
    numerical=MonodomainNumerics(discretization='consistent_p1',diffusion_quadrature='nodal_p1',solver=opt)
    cfg=replace(AFSEIConfig(),ep_refinement_factor=3,ep_refinement_cache=str(tmp_path),ep_numerics=numerical)
    g=prepare_p1(X,cells,5)
    a=RefinedMonodomain(mesh,cfg,f,s,stim,g)
    b=RefinedMonodomain(mesh,replace(cfg,ep_numerics=replace(numerical,discretization='nodal_p1')),f,s,stim,g)
    assert a.cache==b.cache
    torch.testing.assert_close(a.stiffness.to_dense(),b.stiffness.to_dense(),rtol=0,atol=0)
    torch.testing.assert_close(a.mass.to_dense().sum(1),b.mass_diagonal,rtol=1e-13,atol=1e-14)
    ff=a.transfer.prolong(f);ss=a.transfer.prolong(s)
    reference=Monodomain(a.X,a.cells,ff,ss,numerics=numerical)
    torch.testing.assert_close(a.mass.to_dense(),reference.mass.to_dense(),rtol=1e-13,atol=1e-14)
    torch.testing.assert_close(a.stiffness.to_dense(),reference.stiffness.to_dense(),rtol=1e-13,atol=1e-14)
    zero=a.X[:,0]*0;voltage=zero-.08+.03*a.X[:,0];dt=5e-5
    for p in (a,b):
        result,info=p.step(voltage,zero,p.stimulus*25,dt)
        M=p.mass.to_dense();K=p.stiffness.to_dense();alpha=.1/dt
        dense=torch.linalg.solve(alpha*M+.5*K,M@(alpha*voltage+.1*p.stimulus*25)-.5*K@voltage)
        torch.testing.assert_close(result,dense,rtol=1e-10,atol=1e-12)
        assert info.residual_norm<=info.tolerance
    a.close();b.close();reference.close()


# Imported fixtures exercise the actual coupled preload and checkpoint path.
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_balanced_dynamic import balanced_path,dynamic_driver


@pytest.mark.parametrize('mass',['consistent_p1','nodal_p1'])
def test_refined_cardiac_preload_restart_and_next_step(mac_case,balanced_path,tmp_path,mass):
    from afsi_torch.ep.mac_cycle import CardiacStepper
    from afsi_torch.ep import mac_balanced_checkpoint as restart
    from test_afsei_driver import assert_same
    base=dynamic_driver(mac_case)
    cfg=replace(base.config,ep_refinement_factor=3,ep_refinement_cache=str(tmp_path/'fine_cache'),
        dt_ode_ms=.001,ep_numerics=replace(base.config.ep_numerics,
            discretization=mass,diffusion_quadrature='nodal_p1'))
    d=CardiacStepper(base.mesh,cfg,level=2)
    d.port_geometry=base.port_geometry;d.pressure_guard_cells=1
    path,h=balanced_path
    old=d.initialize_preload(path)
    assert len(old.tissue.voltage_v)>len(d.mesh.X)
    assert len(old.fsi.x)==len(base.mesh.X)
    assert d.config.fsi.fluid.shape==base.config.fsi.fluid.shape
    torch.testing.assert_close(old.strain.stretch,d.prolong_ep(h.stretch),rtol=1e-11,atol=1e-11)
    state,info=d.step(old)
    assert info['ep']['steps']==2 and info['ep']['ode_substeps']==100
    assert abs(info['balanced']['mass_error_ml_s'])<1e-5
    saved=tmp_path/f'refined_{mass}.npz'
    restart.save(saved,d,state,dict(ep_steps=2))
    restored,copy,_=restart.load(saved,cardiac=True)
    assert restored.config.ep_refinement_factor==3
    assert_same(copy,state,exact=True)
    a,_=d.step(state);b,_=restored.step(copy)
    assert_same(a,b)
