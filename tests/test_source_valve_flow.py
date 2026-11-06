import math
import pytest
import torch
from afsi_torch.ep.aortic_valve import (AorticValveParameters,ValveClosureRequired,
    solve_fluid_valve,flow_driven_valved_trial)
from afsi_torch.ep.windkessel import WindkesselParameters
from afsi_torch.ep.single_port import SinglePortBoundary
from afsi_torch.mac.grid import MACGrid,divergence
from afsi_torch.mac.backward_euler import BackwardEulerFlow,BEFlowOptions


@pytest.mark.parametrize('dp',[-10.,0.,.001,.1,.5,1.,50.])
def test_source_valve_law_and_differential(dp):
    v=AorticValveParameters();q,dq=v.flow_and_derivative(dp)
    if dp<=0:assert q==dq==0.;return
    # Independent original tanh form (source uses equivalent stable exp).
    r=.5*(v.rmax-v.rmin)*(1-math.tanh(dp/v.epsilon_mmhg))+v.rmin
    assert q==pytest.approx(dp/r,rel=1e-9)
    h=1e-7*max(1.,dp)
    fd=(v.flow_and_derivative(dp+h)[0]-v.flow_and_derivative(dp-h)[0])/(2*h)
    assert dq==pytest.approx(fd,rel=2e-7)


@pytest.mark.parametrize('dt',[1e-4,5e-5])
def test_actual_fluid_rcr_valve_balance_and_dissipation(dt):
    v=AorticValveParameters();r=WindkesselParameters()
    s=solve_fluid_valve(300.,-2.,85.,dt,r,v)
    q=s['flow_ml_s'];pc=s['pc_mmhg'];p=s['upstream_mmhg']
    assert q==pytest.approx(300.-2*p,abs=1e-9)
    assert r.compliance_ml_per_mmhg*(pc-85.)/dt==pytest.approx(q-(pc-8.)/r.rp_mmhg_s_per_ml,abs=3e-10)
    assert s['arterial_mmhg']==pytest.approx(pc+r.rc_mmhg_s_per_ml*q)
    assert s['valve_dissipation_mmhg_ml_s']>0
    _,inv=flow_driven_valved_trial(85.,q,dt,r,v)
    assert inv==pytest.approx(p,abs=1e-10)
    with pytest.raises(ValveClosureRequired):solve_fluid_valve(0.,-2.,85.,dt,r,v)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_mac_valve_nonlinear_response_and_actual_flux(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    grid=MACGrid((8,8,8),(1.,1.,1.))
    base=BackwardEulerFlow(grid,dt=1e-4,device=device,
        options=BEFlowOptions(convection=False,helmholtz_backend='workspace'))
    a=torch.zeros((8,8),dtype=torch.bool,device=device);a[2:6,2:6]=True
    f=SinglePortBoundary(base,a,valve=AorticValveParameters());zero=grid.zeros(device=device)
    r=WindkesselParameters()
    # Manufactured negative downstream pressure drives forward flow from the
    # exterior p=0 reservoir. This is a boundary test, not a physiological LV.
    f.configure('ejection',old_pc_mmhg=-10.,parameters=r)
    u,p,info=f.advance(zero,zero)
    b=info['cut_boundary']
    assert b['outward_flux_ml_s'][0]>0
    assert abs(float(b['valve_flow_residual_ml_s']))<1e-8
    assert abs(float(b['rcr_gap_mmhg']))<1e-9
    assert float(b['valve_dissipation_erg_s'])>0
    assert float(divergence(u,grid.spacing).abs().max())<1e-7
    density=tuple(1000*torch.sin(torch.arange(x.numel(),device=device,dtype=x.dtype)).reshape_as(x) for x in zero)
    tangent=f.response(density)
    eps=1e-4
    plus=f.advance(zero,tuple(eps*x for x in density))[0]
    minus=f.advance(zero,tuple(-eps*x for x in density))[0]
    err=sum(((a-b)/(2*eps)-t).square().sum() for a,b,t in zip(plus,minus,tangent)).sqrt()
    norm=sum(t.square().sum() for t in tangent).sqrt()
    assert float(err/norm)<2e-5
    f.configure('ejection',old_pc_mmhg=85.,parameters=r)
    with pytest.raises(ValveClosureRequired):f.advance(zero,zero)
    f.configure('closed',old_pc_mmhg=85.,parameters=r)
    u,p,info=f.advance(zero,zero)
    assert info['single_port']['outward_flux_ml_s']==0
