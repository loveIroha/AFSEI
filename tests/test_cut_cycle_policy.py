"""Controller-only accepted-state events; physical GPU stepping is separate."""
from dataclasses import dataclass,replace
from types import SimpleNamespace
import torch
from afsi_torch.ep.cut_cycle import CutCardiacStepper,CutCardiacState
from afsi_torch.ep.mac_balanced_dynamic import BalancedAFSEIStepper
from afsi_torch.ep.protocol import EPProtocol,CycleState
from afsi_torch.ep.tissue import TissueState
from afsi_torch.ep.driver import StrainHistory


@dataclass(frozen=True)
class MiniEP:
    tick:int
    cycle:object
    tissue:object
    strain:object
    force_load:object
    fsi:object


def case():
    d=CutCardiacStepper.__new__(CutCardiacStepper)
    d.protocol=EPProtocol(pulse_starts_s=(.5,1.4));d.period_ticks=18000
    d.cycles=2;d.end_tick=46000;d.filling_hold_ticks=2000
    d.fill_target_mmhg=8.;d.filling_model='gao2017';d.inlet_one_way=True
    from afsi_torch.ep.gao_filling import GaoFillingParameters
    d.gao_filling=GaoFillingParameters(inlet_ramp_fraction=.1)
    d.validate_control=lambda s:None
    d.cut_pressure=lambda s,label:80.
    d.inlet_pressure=lambda s:0.
    z=torch.zeros(2,dtype=torch.float64);land=torch.ones(2,6,dtype=torch.float64)
    tissue=TissueState(z,z[:,None],2*land)
    ep=MiniEP(12000,CycleState(mode='land',resting_land=land),tissue,
        StrainHistory(torch.ones_like(z),z,z),object(),SimpleNamespace(x=torch.zeros(2,3),time=.6))
    s=CutCardiacState(ep,100.,torch.zeros(2),torch.zeros(2,3),mode='closed',
        initial_volume_ml=100.,initial_tract_volume_ml=10.,target_total_volume_ml=110.,edv_ml=100.)
    d.buffer=SimpleNamespace(stage=lambda load:None,closure_snapshot=lambda x:torch.full((2,3),7.))
    d._reset_scratch=lambda state:None
    return d,s


def diagnostic(v=100.,tract=10.,qin=0.,qout=0.):
    return dict(accepted_dt_s=.0001,balanced=dict(volume_ml=v,tract_volume_ml=tract,
        inlet_flow_ml_s=qin,physical_outlet_flow_ml_s=qout,measured_relative_q_ml_s=999.,
        supply_pressure_mmhg=8.))


def test_closed_volume_uses_lv_plus_tract_not_lv_alone():
    d,s=case()
    new,info=d.publish_cut_phase(s,s,diagnostic(99.5,10.5))
    assert new.phase=='ISOVOLUMIC_CONTRACTION'
    assert info['cardiac']['cumulative_mass_error_ml']==0.
    assert info['balanced']['boundary_pressure_mmhg']==80.
    assert info['balanced']['prescribed_cut_pressure_mmhg'] is None
    assert info['balanced']['boundary_pressure_kind']=='closed-upstream-fluid-trace'


def test_cumulative_output_uses_actual_cut_flux():
    d,s=case();s=replace(s,phase='EJECTION',mode='rcr')
    new,info=d.publish_cut_phase(s,s,diagnostic(99.9998,10.,qout=2.))
    assert new.cumulative_out_ml==.0002
    assert info['balanced']['measured_relative_q_ml_s']==2.
    assert abs(info['cardiac']['cumulative_mass_error_ml'])<1e-12


def test_outlet_closure_keeps_source_tension_snapshot_and_80ms_unload():
    d,s=case();s=replace(s,phase='EJECTION',mode='rcr')
    new,info=d.publish_cut_phase(s,replace(s,mode='closed'),diagnostic(),valve_closed=True)
    assert new.phase=='ISOVOLUMIC_RELAXATION' and new.mode=='closed'
    assert new.cycle.closure_tick==s.tick
    assert torch.equal(new.cycle.anchor,torch.full((2,3),7.))
    assert new.cycle.factor(s.tick,d.protocol)==1.
    assert new.cycle.factor(s.tick+1600,d.protocol)==0.
    assert torch.equal(new.tissue.land,s.cycle.resting_land)
    assert info['cardiac']['events']==['OUTLET_CLOSURE_TA_SNAPSHOT']
    assert info['balanced']['boundary_pressure_kind']=='closed-upstream-fluid-trace'
    assert info['cardiac']['solved_mode']=='closed'


def test_relaxation_filling_hold_and_next_pacing():
    d,s=case()
    zero=replace(s.cycle,mode='zero',closure_tick=16236)
    ep=replace(s.ep,tick=17836,cycle=zero)
    off=replace(s,ep=ep,phase='ISOVOLUMIC_RELAXATION',mode='closed')
    fill,info=d.publish_cut_phase(off,off,diagnostic())
    assert fill.phase=='FILLING' and fill.fill_end_tick==26000
    d.inlet_pressure=lambda s:-1.
    fill,info=d.publish_cut_phase(fill,fill,diagnostic())
    assert fill.mode=='filling'
    paced=replace(fill,ep=replace(fill.ep,tick=28000,cycle=replace(zero,mode='land')))
    new,info=d.publish_cut_phase(fill,paced,diagnostic())
    assert new.phase=='ISOVOLUMIC_CONTRACTION' and new.mode=='closed' and new.beat==2
    assert new.target_total_volume_ml==110. and new.edv_ml==100.


def test_opening_compares_closed_cut_pressure_to_arterial_state():
    d,s=case();d.cut_pressure=lambda s,label:86.
    new,info=d.publish_cut_phase(s,s,diagnostic())
    assert new.phase=='EJECTION' and new.mode=='rcr'
    assert info['cardiac']['events']==['PHYSICAL_AV_OPENING']


def test_reverse_outlet_trial_is_discarded_before_event_publication(monkeypatch):
    d,s=case();s=replace(s,phase='EJECTION',mode='rcr')
    calls=[]
    def trial(driver,old,**kwargs):
        calls.append(old.mode)
        ep=replace(old.ep,tick=old.tick+2)
        info=diagnostic(qout=-1. if old.mode=='rcr' else 0.)
        return replace(old,ep=ep),info
    monkeypatch.setattr(BalancedAFSEIStepper,'step',trial)
    new,info=d.step(s,target_tick=s.tick+2)
    assert calls==['rcr','closed']
    assert info['cut_valve_trial_discarded']
    assert new.cumulative_out_ml==0.
    assert new.phase=='ISOVOLUMIC_RELAXATION'
    assert s.phase=='EJECTION' and s.tick==12000
