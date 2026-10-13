"""Validate original ORd equations and SDIRK against independent CPU C code."""
from dataclasses import replace
from pathlib import Path
import os
import numpy as np
import pytest
import torch
from afsi_torch.ep.ord import ORdCell
from afsi_torch.ep.experiment import EPTissueConfig,TissueInitial


@pytest.fixture
def reference():
    if not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    source=Path(__file__).resolve().parents[1]/'.deps/ord_reference/ord_generated.c'
    if not source.exists():
        if os.environ.get('AFSEI_REQUIRE_ORD')=='1': raise RuntimeError('download pinned independent ORd reference')
        pytest.skip('independent original reference not downloaded')
    from validation.ord_reference import Reference
    return Reference(1)


def test_original_rhs_and_native_initialization_match_cpu_reference(reference):
    cell=ORdCell(); state=cell.initial(torch.zeros(7,dtype=torch.float64,device='cuda'))
    v=np.array([-87.,-70.,-50.,-30.,-10.,20.,50.])
    # Nonzero channel gates exercise all transmembrane currents, not just rest.
    y=np.tile(reference.initial[1:],(7,1)); y[:,9:38]=np.linspace(.05,.8,29)
    actual,current=torch.ops.afsei_ord_cuda.rhs(torch.as_tensor(y,device='cuda'),
        torch.as_tensor(v,device='cuda'),cell.constants)
    expected=[reference.rhs(a,b) for a,b in zip(y,v)]
    np.testing.assert_allclose(actual.cpu().numpy(),np.stack([a for a,b in expected]),rtol=2e-12,atol=2e-10)
    np.testing.assert_allclose(current.cpu().numpy(),[b for a,b in expected],rtol=2e-12,atol=2e-10)
    np.testing.assert_allclose(state.ionic[0].cpu().numpy(),reference.initial[1:],rtol=0,atol=0)
    assert cell.constants[12].item()==0. and cell.constants[42].item()==75.
    assert cell.provenance()['original_sodium_current']


def test_ghk_zero_voltage_is_finite_and_matches_analytic_limit(reference):
    cell=ORdCell(); state=cell.initial(torch.zeros(1,dtype=torch.float64,device='cuda'))
    y=state.ionic.clone(); y[:,9:38]=.3
    actual,current=torch.ops.afsei_ord_cuda.rhs(y,torch.zeros_like(state.voltage_v),cell.constants)
    assert torch.isfinite(actual).all() and torch.isfinite(current).all()
    positive=reference.rhs(y[0].cpu().numpy(),1e-4); negative=reference.rhs(y[0].cpu().numpy(),-1e-4)
    np.testing.assert_allclose(actual[0].cpu().numpy(),(positive[0]+negative[0])/2,rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(current.cpu().numpy(),[(positive[1]+negative[1])/2],rtol=1e-8,atol=1e-8)


def test_sdirk_substeps_current_mean_and_input_ownership_match_reference(reference):
    cell=ORdCell(); state=cell.initial(torch.zeros(3,dtype=torch.float64,device='cuda'))
    voltages=(-80.,-40.,20.)
    state=replace(state,voltage_v=state.voltage_v.new_tensor(voltages)/1000)
    original=state.ionic.clone()
    result,current=cell.advance_ode(state,0.)
    expected=[reference.advance(reference.initial[1:],v) for v in voltages]
    np.testing.assert_allclose(result.ionic.cpu().numpy(),np.stack([a for a,b in expected]),rtol=2e-11,atol=2e-12)
    np.testing.assert_allclose(current.cpu().numpy(),[b for a,b in expected],rtol=2e-11,atol=2e-12)
    torch.testing.assert_close(state.ionic,original,rtol=0,atol=0)


def test_native_initial_policy_is_required_for_ord():
    with pytest.raises(ValueError,match='own native'):
        replace(EPTissueConfig(),cell_model='ord2011_epi')
    cfg=replace(EPTissueConfig(),cell_model='ord2011_epi',
                initial=TissueInitial(state='model_native_rest',voltage_v=None,conditioning_s=.2))
    assert cfg.cell_model=='ord2011_epi'
