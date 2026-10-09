import pytest
import torch
from types import SimpleNamespace
from afsi_torch.two_way import CoupledBlood,TwoWayMechanics,MechanicalConfig,SolidState
from afsi_torch.moving_cavity import ALEConfig
from afsi_torch.moving_cavity.geometry import swept_geometry
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from test_moving_cavity import cube


class SpringSolid:
    """Independent linear solid oracle, including inward prescribed loading."""
    def __init__(self,mesh,load_pressure=0.,stiffness=1e4):
        X=mesh.reference_solid
        cells=torch.tensor([[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]],device=X.device)
        self.mesh=SimpleNamespace(X=X,cells=cells)
        self.volumes=X.new_full((6,),1/6); self.stiffness=stiffness
        g=swept_geometry(mesh,X,X,.0001)
        self.pg=mesh.transfer_reaction(g.areas.masked_fill((mesh.tags<0)[:,None],0.))
        self.load=-load_pressure*C*self.pg
    def force(self,x,t): return self.load-self.stiffness*(x-self.mesh.X)
    def validate(self,x):
        if not torch.isfinite(x).all(): raise ValueError('invalid coordinates')
    def tangent_factory(self):
        model=self
        class Tangent:
            def assemble(self,x,t):
                index=torch.arange(x.numel(),device=x.device)
                return torch.sparse_coo_tensor(torch.stack((index,index)),x.new_full((x.numel(),),-model.stiffness),(x.numel(),)*2).coalesce().to_sparse_csr()
        return Tangent()


def setup(device,pressure):
    mesh=cube(device); f=CoupledBlood(mesh,ALEConfig(initial_pc_mmhg=85.,convection=False))
    s=SpringSolid(mesh,pressure); x=mesh.reference_solid
    old=SolidState(0,0.,x.clone(),torch.zeros_like(x),s.force(x,0.),0.,0.)
    return s,f,old,f.initial_state()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_closed_pressure_is_determined_by_solid_and_volume(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s,f,old,blood=setup(device,50.)
    new,b,diag,force=TwoWayMechanics(s,f).step(old,blood,.0001,.0001)
    assert b.opened==(False,False)
    assert diag['pressure_mean_mmhg']==pytest.approx(50.,abs=1e-8)
    assert abs(diag['volume_rate_ml_s'])<=1e-8
    assert diag['solid_momentum_residual_norm_dyn']<=diag['solid_momentum_tolerance_dyn']
    torch.testing.assert_close(new.x,old.x,atol=1e-12,rtol=0)
    torch.testing.assert_close(force,-s.load,atol=1e-4,rtol=1e-7)
    assert not diag['original_box_fluid_used']


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_blood_feedback_inflates_unloaded_solid_and_conserves_mass(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    s,f,old,blood=setup(device,0.)
    new,b,diag,force=TwoWayMechanics(s,f).step(old,blood,.0001,.0001)
    assert b.opened==(True,False)
    assert diag['volume_rate_ml_s']>0 and diag['q_in_ml_s']>0
    assert (new.x-old.x).norm()>0 and force.norm()>0
    assert abs(diag['mass_error_ml_s'])<1e-7
    assert abs(diag['interface_work_error_erg_s'])<1e-8
    assert diag['solid_momentum_residual_norm_dyn']<=diag['solid_momentum_tolerance_dyn']
    assert torch.equal(old.x,s.mesh.X) and blood.step==0


def test_failed_solve_retains_all_accepted_states():
    s,f,old,blood=setup('cpu',0.)
    with pytest.raises(RuntimeError,match='converged'):
        TwoWayMechanics(s,f,MechanicalConfig(max_iterations=1)).step(old,blood,.0001,.0001)
    assert old.step==blood.step==0
    assert torch.equal(old.x,s.mesh.X) and torch.equal(blood.solid_x,s.mesh.X)


def test_closed_incompatible_trial_cannot_be_committed():
    s,f,old,blood=setup('cpu',50.)
    tr=f.trial(blood,old.x*1.000001,.0001,(False,False),50.)
    with pytest.raises(RuntimeError,match='mass|volume'):
        f.commit(blood,old.x*1.000001,.0001,tr)


@pytest.mark.parametrize('pressure',[60.,30.,10.])
def test_closed_unloading_changes_pressure_without_changing_volume(pressure):
    s,f,old,blood=setup('cpu',70.)
    solver=TwoWayMechanics(s,f)
    old,blood,_,_=solver.step(old,blood,.0001,.0001)
    s.load=-pressure*C*s.pg
    new,b,diag,force=solver.step(old,blood,.0001,.0002)
    assert b.opened==(False,False)
    assert diag['pressure_mean_mmhg']==pytest.approx(pressure,abs=1e-7)
    assert abs(diag['volume_rate_ml_s'])<=1e-8
    assert diag['q_in_ml_s']==diag['q_out_ml_s']==0.


def test_numerical_fluid_jvp_refines_nonuniform_closed_coupling():
    s,f,old,blood=setup('cpu',50.)
    # A nonsymmetric nodal perturbation exercises nonconstant fluid pressure,
    # wall acceleration, and the augmented pressure/volume block together.
    s.load[0,0]+=2.
    solver=TwoWayMechanics(s,f,MechanicalConfig(newton_after=1))
    new,b,diag,force=solver.step(old,blood,.0001,.0001)
    assert b.opened==(False,False) and (new.x-old.x).norm()>0
    assert abs(diag['volume_rate_ml_s'])<=1e-8
    assert diag['solid_momentum_residual_norm_dyn']<=diag['solid_momentum_tolerance_dyn']
