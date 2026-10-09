import pytest
import torch
from afsi_torch.ep.windkessel import WindkesselParameters, flow_driven_trial, oriented_triangle_flux


@pytest.mark.parametrize('flow',[0.,200.,-2.])
def test_flow_driven_balance_and_one_proximal_resistance(flow):
    p = WindkesselParameters()
    old = 85.; dt = 1e-4
    trial = flow_driven_trial(old,flow,dt,p)
    assert p.compliance_ml_per_mmhg*(trial.pc_mmhg-old)/dt == pytest.approx(
        flow-(trial.pc_mmhg-p.venous_mmhg)/p.rp_mmhg_s_per_ml,abs=2e-10)
    assert trial.outlet_mmhg-trial.pc_mmhg == pytest.approx(p.rc_mmhg_s_per_ml*flow)
    # Nonlinear trials always read the SAME accepted pressure.
    flow_driven_trial(old,flow+30,dt,p)
    assert flow_driven_trial(old,flow,dt,p) == trial


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_outlet_impedance_and_relative_flux(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    q = torch.tensor(180.,dtype=torch.float64,device=device,requires_grad=True)
    trial = flow_driven_trial(85.,q,.001)
    derivative, = torch.autograd.grad(trial.outlet_mmhg,q)
    assert derivative.item() == pytest.approx(trial.impedance_mmhg_s_per_ml,rel=1e-13)
    nodes = q.new_tensor([[[0.,0.,0.],[2.,0.,0.],[0.,3.,0.]]])
    velocity = torch.zeros_like(nodes); velocity[0,:,2] = q.new_tensor([1.,2.,3.])
    moving = torch.zeros_like(nodes); moving[:,:,2] = .5
    assert oriented_triangle_flux(nodes,velocity).item() == pytest.approx(6.)
    assert oriented_triangle_flux(nodes,velocity,moving).item() == pytest.approx(4.5)
    assert oriented_triangle_flux(nodes[:,[0,2,1]],velocity[:,[0,2,1]],moving[:,[0,2,1]]).item() == pytest.approx(-4.5)
    assert oriented_triangle_flux(nodes,velocity,velocity).item() == 0


def test_invalid_time():
    with pytest.raises(ValueError): flow_driven_trial(85.,0.,0.)
