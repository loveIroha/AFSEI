import pytest
import torch
from dataclasses import replace
from afsi_torch.moving_cavity.geometry import from_solid,swept_geometry
from afsi_torch.moving_cavity import ALEStepper,ALEConfig
from afsi_torch.moving_cavity.operators import ALEOperators
from afsi_torch.circulation.geometry import PortGeometry


class CubeSolid:
    def __init__(self,device='cpu'):
        # Open top solid endocardium, normals pointing INTO blood.
        self.X=torch.tensor([[x,y,z] for z in (0.,1.) for y in (0.,1.) for x in (0.,1.)],dtype=torch.float64,device=device)
        self.faces=torch.tensor([[0,3,2],[0,1,3],[0,5,1],[0,4,5],
            [2,7,6],[2,3,7],[0,6,4],[0,2,6],[1,7,3],[1,5,7]],device=device)
    def surface(self,tag): return self.faces


def cube(device='cpu'):
    source=CubeSolid(device)
    ports=PortGeometry(mv_center_cm=(.2,0),av_center_cm=(-.2,0),mv_radius_cm=.15,av_radius_cm=.15)
    return from_solid(source,ports,cap_max_edge_cm=.10)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_swept_gcl_and_interface_work(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    mesh=cube(device); X=mesh.reference_solid
    F=X.new_tensor([[1.02,.03,0],[0,.98,.01],[.02,0,1.01]])
    new=X@F.T+X.new_tensor([.2,-.1,.05]); g=swept_geometry(mesh,X,new,.01)
    assert g.gcl_max_ml_s<1e-11
    assert abs(g.volumes.sum().item()-torch.linalg.det(F).item())<1e-12
    force=torch.randn(len(mesh.faces),3,device=device,dtype=torch.float64)
    solid=mesh.transfer_reaction(force)
    w=mesh.coordinates(new-X)[mesh.faces].mean(1)/.01
    torch.testing.assert_close((solid*(new-X)/.01).sum(),(force*w).sum(),rtol=1e-12,atol=1e-10)
    op=ALEOperators(mesh,g,(True,True)); p=torch.randn(len(mesh.cells),device=device,dtype=torch.float64)
    u=torch.randn_like(force).masked_fill(op.fixed[:,None],0.)
    torch.testing.assert_close((p*op.divergence(u)).sum(),(op.transpose(p)*u).sum(),rtol=1e-12,atol=1e-12)


def test_closed_uniform_translation_free_stream():
    mesh=cube(); X=mesh.reference_solid; dt=.001; velocity=X.new_tensor([.4,-.2,.1])
    g=swept_geometry(mesh,X,X+dt*velocity,dt); op=ALEOperators(mesh,g,(False,False))
    old=velocity.expand(len(mesh.faces),3).clone()
    c=ALEConfig(dt_s=dt,convection=True)
    star,rhs,info=op.viscous(old,dt,c.density,c.viscosity,c.solver,True)
    u,p,info=op.project(star,X.new_zeros(2),dt,c.density,c.solver)
    torch.testing.assert_close(u,old,atol=1e-10,rtol=1e-10)
    assert op.divergence(u).abs().max()<1e-10
    assert g.gcl_max_ml_s<1e-11


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_independent_valves_volume_balance_and_closure(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    mesh=cube(device); driver=ALEStepper(mesh,ALEConfig(initial_pc_mmhg=85.,convection=False))
    old=driver.initial_state(); X=mesh.reference_solid; center=X.mean(0)
    for scale,expected in ((.999999,(False,True)),(1.000001,(True,False))):
        new_x=center+scale*(X-center); new,diag,force=driver.step(old,new_x)
        assert new.opened==expected
        assert abs(diag['mass_error_ml_s'])<1e-8
        assert diag['wall_relative_flux_max_ml_s']==0
        assert diag['max_gcl_error_ml_s']<1e-10
        assert abs(diag['interface_work_error_erg_s'])<1e-8
        if expected==(True,False):
            assert diag['accepted_outlet_closure']
            assert diag['q_in_ml_s']>0 and diag['q_out_ml_s']==0
        else: assert diag['q_out_ml_s']>0 and diag['q_in_ml_s']==0
        assert torch.equal(old.solid_x,X) if old.step==0 else True
        old=new


def test_incompatible_closed_motion_is_rejected_without_leak():
    mesh=cube(); X=mesh.reference_solid; g=swept_geometry(mesh,X,X*1.001,.001)
    op=ALEOperators(mesh,g,(False,False)); c=ALEConfig()
    star,rhs,info=op.viscous(torch.zeros(len(mesh.faces),3,dtype=X.dtype),.001,1.,.04,c.solver,False)
    with pytest.raises(ValueError,match='changes volume'): op.project(star,X.new_zeros(2),.001,1.,c.solver)


def test_moving_checkpoint_resume_checksum_and_cfl(tmp_path):
    from afsi_torch.moving_cavity import checkpoint
    import numpy as np
    mesh=cube(); driver=ALEStepper(mesh,ALEConfig(initial_pc_mmhg=85.,convection=False)); old=driver.initial_state()
    X=mesh.reference_solid; center=X.mean(0)
    state,_,_=driver.step(old,center+.999999*(X-center))
    path=tmp_path/'ale.npz'; checkpoint.save(path,driver,state,{'mode':'test'})
    restored,restart,progress=checkpoint.load(path,'cpu')
    next_x=center+1.000001*(X-center)
    direct,diag,_=driver.step(state,next_x); resumed,rdiag,_=restored.step(restart,next_x)
    assert diag==rdiag and progress=={'mode':'test'}
    torch.testing.assert_close(direct.velocity,resumed.velocity,rtol=0,atol=0)
    with np.load(path,allow_pickle=False) as f: data={n:f[n] for n in f.files}
    data['pc_mmhg']=data['pc_mmhg']+1; np.savez_compressed(path,**data)
    with pytest.raises(ValueError,match='checksum'): checkpoint.load(path,'cpu')
    strict=ALEStepper(mesh,replace(driver.config,convection=True,cfl_limit=1e-12))
    with pytest.raises(ValueError,match='CFL'): strict.step(state,next_x)


def test_custom_spd_inverse_keeps_boundary_and_true_residual():
    from afsi_torch.fluid.solvers import pcg,SolverOptions
    A=torch.tensor([[3.,-1.,0.,0.],[-1.,3.,-1.,0.],[0.,-1.,3.,-1.],[0.,0.,-1.,3.]],dtype=torch.float64)
    expected=torch.tensor([2.,3.,-4.,7.],dtype=torch.float64); rhs=A@expected
    fixed=torch.tensor([True,False,False,False]); values=torch.zeros_like(rhs); values[0]=2.
    result,info=pcg(lambda v:A@v,rhs,A.diagonal(),fixed=fixed,values=values,
        preconditioner=lambda r:torch.linalg.solve(A,r),options=SolverOptions(rtol=1e-12,atol=1e-12))
    torch.testing.assert_close(result,expected,atol=1e-11,rtol=1e-12)
    assert result[0]==2. and (rhs-A@result)[~fixed].norm()<=info.tolerance
    with pytest.raises(ValueError,match='preconditioner'):
        pcg(lambda v:A@v,rhs,A.diagonal(),preconditioner=lambda r:r[:,None])


def test_positive_endpoints_do_not_hide_path_inversion():
    mesh=cube(); X=mesh.reference_solid
    F=X.new_tensor([[-2.,0.,0.],[0.,-.5,0.],[0.,0.,1.]])
    assert torch.linalg.det(F)==1.
    with pytest.raises(ValueError,match='between its endpoints'): swept_geometry(mesh,X,X@F.T,.001)


def test_passive_two_port_throughflow_is_not_a_systolic_ejection():
    mesh=cube(); driver=ALEStepper(mesh,ALEConfig(convection=False,initial_pc_mmhg=8.,atrium_mmhg=8.5))
    new,diag,_=driver.step(driver.initial_state(),mesh.reference_solid)
    assert new.opened==(True,True) and diag['q_out_ml_s']>0
    assert not new.has_ejected and not diag['accepted_outlet_closure']
