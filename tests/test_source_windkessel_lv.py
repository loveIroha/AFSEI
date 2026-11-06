"""Pressure-driven source equations and accepted phase/force histories."""
from dataclasses import dataclass,replace
from types import SimpleNamespace as NS
import math
import pytest
import torch
from afsi_torch.ep.source_windkessel import pressure_driven_trial
from afsi_torch.ep.windkessel import WindkesselParameters
from afsi_torch.ep.aortic_valve import AorticValveParameters
from afsi_torch.ep.mac_source_cycle import (SourceLVStepper,SourceLVState,SourceCycleParameters,
    SourceCycleProblem,filling_pressure)
from afsi_torch.ep.mac_isovolumic import IsovolumicProblem
from afsi_torch.ep.driver import AFSEIStepper,CoupledState,StrainHistory
from afsi_torch.ep.tissue import TissueState
from afsi_torch.ep.protocol import EPProtocol,CycleState


@pytest.mark.parametrize('p',[0.,84.,85.,85.001,85.1,86.,100.,250.])
@pytest.mark.parametrize('dt',[5e-5,1e-4,.001])
def test_pressure_law_balance_and_derivative(p,dt):
    pars=WindkesselParameters();valve=AorticValveParameters()
    trial=pressure_driven_trial(85.,p,dt,pars,valve)
    assert abs(pars.compliance_ml_per_mmhg*(trial.pc_mmhg-85.)/dt-
        trial.flow_ml_s+(trial.pc_mmhg-pars.venous_mmhg)/pars.rp_mmhg_s_per_ml)<1e-8
    assert abs(trial.arterial_mmhg-trial.pc_mmhg-pars.rc_mmhg_s_per_ml*trial.flow_ml_s)<1e-12
    assert trial.flow_ml_s == pytest.approx(valve.flow_and_derivative(p-trial.arterial_mmhg)[0],abs=2e-6)
    eps=1e-4
    derivative=(pressure_driven_trial(85.,p+eps,dt).flow_ml_s-
                pressure_driven_trial(85.,p-eps,dt).flow_ml_s)/(2*eps)
    assert trial.dq_dp_ml_s_mmhg == pytest.approx(derivative,rel=5e-5,abs=2e-6)


def test_closed_decay_and_no_trial_commit():
    pc=85.;a=pressure_driven_trial(pc,150.,1e-4);b=pressure_driven_trial(pc,0.,1e-4)
    assert pc==85. and a.pc_mmhg>b.pc_mmhg
    assert b.flow_ml_s==0 and b.dq_dp_ml_s_mmhg==0
    assert pressure_driven_trial(pc,0.,1e-4)==b


def test_constraint_jacobian_and_schur(monkeypatch):
    import afsi_torch.ep.mac_source_cycle as module
    x=torch.tensor([[1.,2.,3.]],dtype=torch.float64)
    grad=x.new_tensor([[2.,3.,4.]])
    monkeypatch.setattr(module,'closed_volume_gradient',lambda *args:grad)
    obj=SourceCycleProblem.__new__(SourceCycleProblem)
    obj.control=NS(target_volume_ml=100.,arterial_pc_mmhg=85.,windkessel=WindkesselParameters())
    obj.driver=NS(owner=NS(circulation=SourceCycleParameters()))
    obj.load=NS(faces=None,rim=None);obj.time=1e-4;obj.volume_scale=3.;obj.pressure_scale=5.
    obj.filling=False;obj._wk_key=obj._wk=None
    c,d=obj.constraint_derivatives(x,100.)
    h=1e-4;direction=x.new_tensor([[.1,-.3,.2]])
    finite=(obj.constraint(100.+h*(grad*direction).sum(),100.+h*5)-
            obj.constraint(100.-h*(grad*direction).sum(),100.-h*5))/(2*h)
    assert finite==pytest.approx(float(c@direction.flatten()+d),abs=1e-9)
    obj.c=c;obj.b=x.new_tensor([.1,.2,.3]);obj.pressure_diagonal=d;obj.preconditioner_applications=0
    A=torch.eye(4,dtype=x.dtype);A[:3,3]=obj.b;A[3,:3]=c;A[3,3]=d
    rhs=x.new_tensor([.3,.4,-.2,.1])
    assert torch.allclose(IsovolumicProblem.preconditioner(obj,None)(rhs),torch.linalg.solve(A,rhs),atol=1e-12)
    obj.filling=True
    c,d=obj.constraint_derivatives(x,0.)
    assert torch.equal(c,torch.zeros(3,dtype=x.dtype)) and d==1


def test_filling_clock():
    p=EPProtocol(pulse_starts_s=(.5,));cfg=SourceCycleParameters();state=NS(filling_start_tick=1000)
    for delay,value in [(0,0.),(2000,4.),(4000,8.),(10000,8.)]:
        assert filling_pressure(state,1000+delay,p,cfg)==pytest.approx(value)


@dataclass(frozen=True)
class DummyConfig:
    protocol: EPProtocol


def phase_fixture(monkeypatch,phase,old_tick,new_tick,p,mode='land',closure_tick=-1):
    driver=SourceLVStepper.__new__(SourceLVStepper)
    driver.circulation=SourceCycleParameters();driver.protocol=EPProtocol(pulse_starts_s=(.5,))
    driver.config=DummyConfig(driver.protocol)
    driver.validate_control=lambda state:None
    driver._reset_scratch=lambda state:None
    driver.buffer=NS(closure_snapshot=lambda x:torch.ones(1))
    z=torch.zeros(1,dtype=torch.float64);ones=torch.ones(1,dtype=torch.float64)
    cycle=CycleState(mode=mode,closure_tick=closure_tick,resting_land=ones.clone())
    ep=CoupledState(old_tick,NS(x=z),TissueState(z,z,2*ones),StrainHistory(ones,z,z),cycle,None,z)
    old=SourceLVState(ep,100.,p,phase=phase,beat=1,initial_volume_ml=100.)
    accepted=replace(ep,tick=new_tick)
    monkeypatch.setattr(AFSEIStepper,'step',lambda *args,**kwargs:(accepted,{
        'accepted_dt_s':(new_tick-old_tick)*driver.protocol.dt_ep_s,
        'isovolumic':{'cavity_pressure_mmhg':p,'volume_ml':100.}}))
    return driver,old


def test_closure_freezes_tension_and_resets_land(monkeypatch):
    driver,old=phase_fixture(monkeypatch,'EJECTION',15000,15002,40.)
    new,info=driver.step(old)
    assert new.phase=='IVR' and new.cycle.closure_tick==15002
    assert new.cycle.mode=='relaxation' and new.cycle.factor(16602,driver.protocol)==0
    assert torch.equal(new.tissue.land,old.cycle.resting_land)
    assert torch.equal(old.tissue.land,torch.full((1,),2.,dtype=torch.float64))
    assert info['circulation']['valve_closed_event']


def test_tiny_forward_flow_is_open_as_in_source(monkeypatch):
    driver,old=phase_fixture(monkeypatch,'IVC',15000,15002,84.995)
    new,info=driver.step(old)
    assert 0<new.outlet_flow_ml_s<driver.circulation.flow_tolerance_ml_s
    assert new.phase=='EJECTION' and info['circulation']['valve_open']


def test_filling_waits_for_unloading_and_schedules_next_pace(monkeypatch):
    driver,old=phase_fixture(monkeypatch,'IVR',16000,16002,5.,mode='relaxation',closure_tick=15000)
    new,_=driver.step(old)
    assert new.phase=='IVR' and driver.protocol.pulse_starts_s==(.5,)
    driver,old=phase_fixture(monkeypatch,'IVR',16600,16602,5.,mode='zero',closure_tick=15000)
    new,_=driver.step(old)
    assert new.phase=='FILLING' and new.filling_start_tick==16602
    assert driver.protocol.pulse_ticks== (10000,26602)
    assert old.phase=='IVR' and old.filling_start_tick==0


def test_rejected_step_does_not_schedule_stimulus(monkeypatch):
    driver,old=phase_fixture(monkeypatch,'IVR',16600,16602,5.,mode='zero',closure_tick=15000)
    def reject(state):
        if state.tick!=old.tick: raise ValueError('rejected ledger')
    driver.validate_control=reject
    with pytest.raises(ValueError,match='ledger'): driver.step(old)
    assert driver.protocol.pulse_starts_s==(.5,) and driver._iso_context is None
