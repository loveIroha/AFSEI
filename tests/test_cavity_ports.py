import pytest
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.circulation import CavityDomain,CavityFlowConfig,CavityFlowStepper
from afsi_torch.circulation.operators import CavityOperators
from afsi_torch.circulation.geometry import vertical_inside,PortGeometry


def cube(device='cpu'):
    grid=MACGrid((6,6,6),(1.2,1.2,1.2))
    fluid=torch.ones(grid.shape,device=device,dtype=torch.bool)
    tags=tuple(torch.zeros(grid.face_shape(c),device=device,dtype=torch.int64) for c in range(3))
    tags[0][0]=1; tags[0][-1]=2
    return CavityDomain(grid,fluid,tags,torch.zeros(3,device=device,dtype=torch.float64),torch.eye(3,device=device,dtype=torch.float64))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_pressure_adjoint_and_symmetry(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    domain=cube(device); domain.validate(); op=CavityOperators(domain,True)
    torch.manual_seed(102); p=torch.randn(domain.grid.shape,device=device,dtype=torch.float64)
    u=tuple(torch.randn_like(m)*a for m,a in zip(op.mass,op.active)); b=p.new_tensor([3.,7.])
    g=op.gradient(p,b)
    lhs=sum((m*v*w).sum()*op.grid.volume for m,v,w in zip(op.mass,u,g))
    rhs=-(p*op.divergence(u)).sum()*op.grid.volume+(b*op.flux(u)).sum()
    torch.testing.assert_close(lhs,rhs,rtol=1e-13,atol=1e-13)
    A=lambda x:op.apply(op.pressure_matrix,x)
    torch.testing.assert_close(A(p),-op.divergence(op.gradient(p,torch.zeros_like(b))),rtol=1e-13,atol=1e-13)
    r=torch.randn_like(p)
    torch.testing.assert_close((p*A(r)).sum(),(r*A(p)).sum(),rtol=1e-13,atol=1e-12)
    assert (p*A(p)).sum()>0


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('atrium,pc,one_way,sign',[(8.5,8.,True,1),(8.,80.,True,0),(8.,80.,False,-1)])
def test_flux_rcr_energy_and_immutability(device,atrium,pc,one_way,sign):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    stepper=CavityFlowStepper(cube(device),CavityFlowConfig(atrium_mmhg=atrium,initial_pc_mmhg=pc,one_way=one_way))
    old=stepper.initial_state(); original=[v.clone() for v in old.velocity]; cfg=stepper.config
    for i in range(3):
        new,report=stepper.step(old)
        assert new.step==i+1
        assert report['max_divergence_s_inv']<1e-8
        assert abs(report['mass_error_ml_s'])<1e-9
        assert report['wall_leak_cm_s']==0
        assert report['total_energy_relative_error']<1e-8
        assert abs(report['rcr_balance_ml_s'])<1e-9
        assert new.ports_open==(sign!=0)
        assert new.q_out_ml_s[1]*sign>0 if sign else new.q_out_ml_s.abs().max()==0
        if sign:
            wk=cfg.windkessel
            expected=new.pc_mmhg+(wk.rc_mmhg_s_per_ml+cfg.av_resistance)*new.q_out_ml_s[1]
            torch.testing.assert_close(new.port_pressure_cgs[1]/1333.22368421,expected,rtol=1e-10,atol=1e-10)
        for v,w in zip(old.velocity,original): torch.testing.assert_close(v,w,rtol=0,atol=0)
        old=new; original=[v.clone() for v in old.velocity]


def test_shared_triangle_diagonal_crossings():
    vertices=torch.tensor([[x,y,z] for z in (0.,1.) for y in (0.,1.) for x in (0.,1.)],dtype=torch.float64)
    faces=torch.tensor([[0,2,3],[0,3,1],[4,5,7],[4,7,6],
        [0,1,5],[0,5,4],[2,6,7],[2,7,3],[0,4,6],[0,6,2],[1,3,7],[1,7,5]])
    xy=torch.tensor([[.5,.5],[.1,.3],[1.5,.5]],dtype=torch.float64); z=torch.tensor([-.1,.1,.5,.9,1.1],dtype=torch.float64)
    actual=vertical_inside(vertices,faces,xy,z)
    assert actual.tolist()==[[False,True,True,True,False],[False,True,True,True,False],[False]*5]


def test_port_geometry_rejects_overlap():
    with pytest.raises(ValueError,match='disjoint'): PortGeometry(av_center_cm=(.65,0.))


def test_checkpoint_resume_and_checksum(tmp_path):
    import json,numpy as np
    from afsi_torch.circulation import checkpoint
    driver=CavityFlowStepper(cube()); state=driver.initial_state()
    state,_=driver.step(state)
    path=tmp_path/'rigid.npz'; checkpoint.save(path,driver,state,{'a':1})
    restored,loaded,progress=checkpoint.load(path,'cpu')
    assert progress=={'a':1}
    direct,diag=driver.step(state); resumed,rdiag=restored.step(loaded)
    for v,w in zip(direct.velocity,resumed.velocity): torch.testing.assert_close(v,w,rtol=0,atol=0)
    torch.testing.assert_close(direct.pressure,resumed.pressure,rtol=0,atol=0)
    assert diag==rdiag
    with np.load(path,allow_pickle=False) as archive: data={n:archive[n] for n in archive.files}
    data['pc_mmhg']=data['pc_mmhg']+1
    np.savez_compressed(path,**data)
    with pytest.raises(ValueError,match='checksum'): checkpoint.load(path,'cpu')
    meta=json.loads(str(data['metadata'])); meta['producer']='afsei-ep-fsi'; data['metadata']=json.dumps(meta)
    np.savez_compressed(path,**data)
    with pytest.raises(ValueError,match='producer'): checkpoint.load(path,'cpu')


def test_joint_closure_enforces_velocity_not_only_scalar_flux():
    from dataclasses import replace
    driver=CavityFlowStepper(cube()); old,_=driver.step(driver.initial_state())
    # Force a subsequent adverse reservoir gradient while retaining inertia.
    adversarial=replace(old,pc_mmhg=old.pc_mmhg.new_tensor(80.))
    new,report=driver.step(adversarial)
    assert not new.ports_open
    for u,t in zip(new.velocity,driver.domain.tags): assert (u[t>0]==0).all()
    assert report['q_out_ml_s']==0
    assert report['total_energy_relative_error']<1e-8


def test_uniform_pressure_mode_changes_no_velocity():
    from dataclasses import replace
    from afsi_torch.ep.windkessel import WindkesselParameters
    cfg=CavityFlowConfig()
    a=CavityFlowStepper(cube(),cfg)
    b=CavityFlowStepper(cube(),replace(cfg,atrium_mmhg=cfg.atrium_mmhg+100.,initial_pc_mmhg=cfg.initial_pc_mmhg+100.,
        windkessel=WindkesselParameters(venous_mmhg=108.)))
    sa,_=a.step(a.initial_state()); sb,_=b.step(b.initial_state())
    for u,v in zip(sa.velocity,sb.velocity): torch.testing.assert_close(u,v,atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(sb.pressure-sa.pressure,torch.full_like(sa.pressure,100*1333.22368421),rtol=1e-12,atol=1e-9)
