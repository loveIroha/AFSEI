"""Real CUDA regression for fast sodium gates at a CN voltage undershoot."""
from dataclasses import asdict
import pytest
import torch
from afsi_torch.ep import cuda_tissue
from afsi_torch.ep.tissue import GPBLand, TissueState, ODERecoveryConfig, initial_state
from afsi_torch.ep.config import decode_checkpoint_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='actual CUDA kernel required')


def fields():
    state = initial_state(torch.zeros(3,device='cuda',dtype=torch.float64))
    voltage=state.voltage_v.clone()
    voltage[0]=-.1072351926792726
    state.gpb[0,0]=.7749204887725445
    state=TissueState(voltage,state.gpb,state.land)
    one=torch.ones_like(voltage)
    zero=torch.zeros_like(voltage)
    return state,one,zero,zero


def raw(state,stretch,zs,zw,*,factor=1,advance_land=True):
    return cuda_tissue.advance(state.gpb,state.land,state.voltage_v*1000,
        stretch,zs,zw,510.95,.001/factor,50*factor,advance_land)


@pytest.mark.parametrize('advance_land',[True,False])
def test_local_retry_preserves_successful_nodes_and_resolves_current(advance_land):
    state,s,zs,zw=fields()
    saved=[v.clone() for v in (state.voltage_v,state.gpb,state.land,s,zs,zw)]
    baseline=raw(state,s,zs,zw,advance_land=advance_land)
    assert baseline[3][0]>0
    assert torch.equal(baseline[3][1:],torch.zeros_like(baseline[3][1:]))
    solver=GPBLand(5e-5,.001,recovery=ODERecoveryConfig(enabled=True))
    new,current=solver.advance_ode(state,s,zs,zw,.51095,advance_land=advance_land)
    for actual,original in zip((new.gpb,new.land,current),baseline[:3]):
        assert torch.equal(actual[1:],original[1:])
    assert all(torch.equal(a,b) for a,b in zip(saved,
        (state.voltage_v,state.gpb,state.land,s,zs,zw)))
    assert torch.equal(new.voltage_v,state.voltage_v)
    reference=raw(TissueState(*(a[:1].contiguous() for a in
        (state.voltage_v,state.gpb,state.land))),s[:1],zs[:1],zw[:1],factor=2048,
        advance_land=advance_land)
    torch.testing.assert_close(new.gpb[:1],reference[0],rtol=.01,atol=1e-9)
    torch.testing.assert_close(new.land[:1],reference[1],rtol=.01,atol=1e-9)
    torch.testing.assert_close(current[:1],reference[2],rtol=.01,atol=1e-3)
    event=solver.last_recovery
    assert event['recovered_nodes']==1 and event['first_node_indices']==[0]
    assert event['max_accepted_error_ratio']<=1
    assert event['extra_node_substeps']>0
    if not advance_land:
        assert torch.equal(new.land,state.land)
    # A subsequent successful step must not retain the previous event.
    good=TissueState(*(a[1:].contiguous() for a in (state.voltage_v,state.gpb,state.land)))
    solver.advance_ode(good,s[1:],zs[1:],zw[1:],.511)
    assert solver.last_recovery is None


def test_failed_recovery_is_bounded_and_does_not_mutate_inputs():
    state,s,zs,zw=fields()
    before=state.gpb.clone()
    with pytest.raises(FloatingPointError,match='reject'):
        GPBLand(5e-5,.001).advance_ode(state,s,zs,zw,.51095)
    solver=GPBLand(5e-5,.001,recovery=ODERecoveryConfig(enabled=True,max_refinement=4))
    with pytest.raises(FloatingPointError,match='did not converge'):
        solver.advance_ode(state,s,zs,zw,.51095)
    assert torch.equal(state.gpb,before)
    assert solver.last_recovery is None


def test_recovery_config_round_trip_and_legacy_default():
    old=decode_checkpoint_config({})
    assert not old.ode_recovery.enabled
    settings=asdict(old)
    settings['ode_recovery']=asdict(ODERecoveryConfig(enabled=True))
    assert decode_checkpoint_config(settings).ode_recovery.enabled
    for kwargs in ({'enabled':1},{'max_refinement':3},{'max_refinement':1},
                   {'rtol':0.},{'rtol':float('nan')},{'rtol':.1}):
        with pytest.raises(ValueError):ODERecoveryConfig(**kwargs)
