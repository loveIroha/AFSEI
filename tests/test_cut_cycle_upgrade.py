from dataclasses import dataclass,replace
from types import SimpleNamespace
import os
import pytest
import torch
from afsi_torch.ep.cycle_policy import FillingClock,PeriodicMonitor,frozen_flow_initial_pc
from afsi_torch.ep.gao_filling import GaoFillingParameters
from afsi_torch.ep.windkessel import WindkesselParameters
from afsi_torch.ep.cut_cycle import CutCardiacStepper
from afsi_torch.ep.protocol import EPProtocol
from afsi_torch.ep.gao_tracts import LVTractDomain
from afsi_torch.mac.grid import MACGrid


def test_shared_collar_retains_cut_area_and_storage_footprint_but_opens_lower_septum():
    device=os.environ.get('AFSEI_TEST_DEVICE','cpu')
    x=torch.tensor([[2.18,1.3,2.],[2.45,1.6,2.],[1.4,1.3,2.],[1.82,1.6,2.]],
                   dtype=torch.float64,device=device)
    cap=SimpleNamespace(frame=torch.eye(3,device=device,dtype=x.dtype),
        labels=torch.tensor([1,1,2,2],device=device),surface=lambda y:(y,torch.ones_like(y)))
    grid=MACGrid((16,16,16),(4.,4.,4.))
    old=LVTractDomain(grid,cap,x,mode='rcr')
    new=LVTractDomain(grid,cap,x,mode='rcr',shared_collar_cells=3)
    assert old.bounds==new.bounds and old.bottom_index==new.bottom_index
    assert old.metadata['cut_areas_cm2']==new.metadata['cut_areas_cm2']
    assert all(torch.equal(a,b) for a,b in zip(old.apertures,new.apertures))
    # At x=2,y=1.375, the septum is opened only in the basal collar.
    assert old.masks[0][8,5,8]==0 and new.masks[0][8,5,8]==1
    assert new.masks[0][8,5,13]==0
    # Divergence-free exchange within the collar: the AV neck alone has flow,
    # but the union neck and closed terminal cuts have zero net flow.
    u=list(grid.zeros(device=device));u[2][7,5,5:9]=1.;u[2][8,5,5:9]=-1.;u[0][8,5,8]=1.
    cap.material_points=lambda value:value
    areas=torch.zeros_like(x);areas[:,2]=.1
    prepared=SimpleNamespace(points=x,area_vectors=areas,cap=cap)
    budget=new.budget(tuple(u),prepared,torch.zeros_like(x))
    assert budget['fixed_tract_mass_error_ml_s'].abs()<1e-12
    assert budget['branch_neck_outward_ml_s'].abs()>0
    assert budget['control_volume']=='shared-collar-union'
    y=x.clone();y[0,0]=1.99999
    with pytest.raises(ValueError,match='detached'):old.validate_wall(cap,y)
    new.validate_wall(cap,y)
    y[0,0]=.5
    with pytest.raises(ValueError,match='shared collar'):new.validate_wall(cap,y)
    y=x.clone();y[:,2]=2.3
    with pytest.raises(ValueError,match='support margin'):new.validate_wall(cap,y)


@dataclass(frozen=True)
class Config:
    protocol:object


def timing_driver():
    d=CutCardiacStepper.__new__(CutCardiacStepper)
    d.protocol=EPProtocol(pulse_starts_s=(.5,1.4));d.config=Config(d.protocol)
    d.period_ticks=18000;d.end_tick=46000;d.cycles=2;d.filling_clock=FillingClock()
    d.gao_filling=GaoFillingParameters();d.fill_target_mmhg=8.
    return d


def test_full_filling_clock_delays_future_pacing_and_final_end_without_stretching_waveform():
    d=timing_driver();state=SimpleNamespace(tick=17836,beat=1,fill_start_tick=17836)
    end,delayed=d.reserve_filling_window(state)
    assert delayed and end==33836
    assert d.protocol.pulse_ticks==(10000,35836)
    assert d.config.protocol==d.protocol
    assert d.filling_loads(state,25836)==(4.,12.)
    assert d.filling_loads(state,33836)==(8.,0.)
    assert d.filling_loads(state,35836)==(8.,0.)
    last=SimpleNamespace(tick=43666,beat=2)
    end,delayed=d.reserve_filling_window(last)
    assert not delayed and end==59666 and d.end_tick==61666


def state(t,beat,pc=85.,calcium=.0001):
    fsi=SimpleNamespace(time=t,x=torch.ones(3,3,dtype=torch.float64),
        velocity=tuple(torch.zeros(3,dtype=torch.float64) for _ in range(3)),
        pressure=torch.ones(3,dtype=torch.float64))
    return SimpleNamespace(tick=round(t/5e-5),fsi=fsi,beat=beat,edv_ml=138.,
        arterial_pc_mmhg=pc,tissue=SimpleNamespace(voltage_v=torch.full((3,),-.08,dtype=torch.float64),
        gpb=torch.full((3,57),calcium,dtype=torch.float64),land=torch.zeros(3,4,dtype=torch.float64)),
        strain=SimpleNamespace(stretch=torch.ones(3,dtype=torch.float64)))


def test_periodicity_requires_electrical_states_and_repeated_same_rr_not_just_volume():
    m=PeriodicMonitor()
    for i in range(4):m.capture(state(.5+1.3*i,i+1))
    assert m.summary()['periodic'] and m.summary()['completed_periodic_cycles']==0
    clone=PeriodicMonitor();clone.restore(m.metadata(),{k:v.clone() for k,v in m.reference.items()})
    clone.capture(state(5.7,5))
    assert clone.summary()['completed_periodic_cycles']==1
    clone.capture(state(7.,6,calcium=.0002))
    assert not clone.summary()['periodic']
    assert clone.comparisons[-1]['metrics']['calcium_relative_rms']>.5


def test_fluid_transient_prevents_periodicity_despite_identical_edv_and_ionic_states():
    m=PeriodicMonitor()
    for i in range(4):m.capture(state(.5+1.3*i,i+1))
    changed=state(5.7,5)
    changed.fsi.velocity=(torch.ones(3,dtype=torch.float64),)*3
    m.capture(changed)
    assert not m.summary()['periodic']
    assert m.comparisons[-1]['metrics']['velocity_relative_rms']==1.


def test_frozen_flow_seed_is_exact_for_constant_outflow_and_never_claims_coupled_periodicity():
    p=WindkesselParameters(rc_mmhg_s_per_ml=.0345,rp_mmhg_s_per_ml=.7245)
    seed=frozen_flow_initial_pc([.001]*900,[50.]*900,p)
    assert abs(seed['pc_mmhg']-(8+.7245*50))<1e-10
    assert not seed['coupled_periodic_state']
    with pytest.raises(ValueError):frozen_flow_initial_pc([.001],[-1.],p)


def test_late_ejection_cannot_receive_next_pulse_before_relaxation_and_full_filling():
    d=timing_driver();d.maximum_ticks=2
    late=SimpleNamespace(tick=27998,beat=1,phase='EJECTION')
    assert d.protect_unready_pacing(late)
    assert d.protocol.pulse_ticks[1]==47600
    assert d.protocol.pulse_ticks[0]==10000
