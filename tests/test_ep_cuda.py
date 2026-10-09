from pathlib import Path
import os
import subprocess
import numpy as np
import pytest
import torch
from afsi_torch.ep import GPBLand, TissueState, initial_state, Monodomain
from afsi_torch.ep import cuda_tissue
from afsi_torch.ep.land import land_ode_step

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA required')


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    folder = ROOT/'validation/ep_reference'
    binary = tmp_path_factory.mktemp('gpb-oracle')/'oracle'
    # Original CPU code compiled separately from the allocation-free device RHS.
    subprocess.run([os.environ.get('CXX','g++'),'-O2','-ffp-contract=off','-std=c++17',
                    str(folder/'GPB_cell_model.cpp'),str(folder/'oracle.cpp'),'-o',str(binary)],check=True)
    return binary


def data():
    voltage_mv = torch.tensor([-90.,-81.5455936324844,-60.,-47.1301,-47.1299,-40.,
                               -5.0001,-4.9999,-.001,20.,40.],dtype=torch.float64,device='cuda')
    state = initial_state(voltage_mv)
    y = state.gpb.clone()
    y[:,35:38] *= torch.linspace(.8,1.2,len(y),dtype=y.dtype,device=y.device)[:,None]
    land = y.new_tensor([.02,.07,.15,.8]).expand(len(y),-1).clone()
    stretch = torch.linspace(.85,1.25,len(y),dtype=y.dtype,device=y.device)
    zs = torch.linspace(-1.3,.3,len(y),dtype=y.dtype,device=y.device)
    zw = torch.linspace(-.1,.1,len(y),dtype=y.dtype,device=y.device)
    return y,land,voltage_mv,stretch,zs,zw


def cpu_oracle(binary,args,*,steps=25,live=True,rhs=False):
    y,land,v,stretch,zs,zw = (a.cpu().numpy() for a in args)
    rows = np.column_stack((v,stretch,zs,zw,land,y))
    text = f'{len(y)} {steps} {int(live)} 100.0 0.002\n'
    text += '\n'.join(' '.join(format(x,'.17g') for x in row) for row in rows)+'\n'
    result = subprocess.run([str(binary)]+(['rhs'] if rhs else []),input=text,text=True,capture_output=True,check=True)
    return torch.tensor(np.loadtxt(result.stdout.splitlines()),dtype=torch.float64,device='cuda')


def test_all_57_rhs_states_against_original_cpp(oracle):
    args = data()
    y,land,v,stretch,zs,zw = args
    expected = cpu_oracle(oracle,args,rhs=True)
    dy,current = cuda_tissue.rhs(y,v,100.)
    torch.testing.assert_close(dy,expected[:,:57],rtol=2e-11,atol=1e-12)
    torch.testing.assert_close(current,expected[:,57],rtol=2e-11,atol=1e-12)


@pytest.mark.parametrize('steps',[1,25])
@pytest.mark.parametrize('live',[True,False])
def test_sdirk_subcycles_land_and_final_current_against_cpp(oracle,steps,live):
    args = data()
    snapshots = [v.clone() for v in args]
    y,land,v,stretch,zs,zw = args
    expected = cpu_oracle(oracle,args,steps=steps,live=live)
    out_y,out_land,current,guards = cuda_tissue.advance(y,land,v,stretch,zs,zw,100.,.002,steps,live)
    assert not guards.any()
    torch.testing.assert_close(out_y,expected[:,:57],rtol=3e-11,atol=1e-12)
    torch.testing.assert_close(out_land,expected[:,57:61],rtol=3e-11,atol=1e-12)
    torch.testing.assert_close(current,expected[:,61],rtol=3e-11,atol=1e-12)
    for original,snapshot in zip(args,snapshots):
        torch.testing.assert_close(original,snapshot,rtol=0,atol=0)
    if not live:
        torch.testing.assert_close(out_land,land,rtol=0,atol=0)
    elif steps == 1:
        ref_land = land_ode_step(land,out_y[:,37]*1000,stretch,zs,zw,.002)
        torch.testing.assert_close(out_land,ref_land,rtol=2e-12,atol=1e-14)


def test_current_stream_and_graph_replay():
    cuda_tissue.build()
    args = data()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = cuda_tissue.advance(*args,100.,.002,25,True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            captured = cuda_tissue.advance(*args,100.,.002,25,True)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    for a,b in zip(expected,captured): torch.testing.assert_close(a,b,rtol=0,atol=0)
    args[2].add_(.1)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
        changed = cuda_tissue.advance(*args,100.,.002,25,True)
    torch.cuda.current_stream().wait_stream(stream)
    for a,b in zip(changed,captured): torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_transactional_split_and_land_pause():
    X = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=torch.float64,device='cuda')
    cells = torch.tensor([[0,1,2,3]],device='cuda')
    ep = Monodomain(X,cells,X.new_tensor([1.,0.,0.]).expand(4,-1),X.new_tensor([0.,1.,0.]).expand(4,-1))
    state = initial_state(X[:,0])
    original = [state.voltage_v.clone(),state.gpb.clone(),state.land.clone()]
    stretch = X.new_ones(4); z = X.new_zeros(4); stim = X.new_full((4,),25.)
    model = GPBLand()
    assert model.substeps == 25
    first,current,_ = model.step(ep,state,stretch,z,z,.5,stim)
    retry,again,_ = model.step(ep,state,stretch,z,z,.5,stim)
    torch.testing.assert_close(first.voltage_v,retry.voltage_v,rtol=0,atol=0)
    torch.testing.assert_close(first.gpb,retry.gpb,rtol=0,atol=0)
    torch.testing.assert_close(first.voltage_v,state.voltage_v+5e-5*(25-current),rtol=1e-10,atol=1e-12)
    paused,_,_ = model.step(ep,state,stretch,z,z,.5,stim,advance_land=False)
    torch.testing.assert_close(paused.land,state.land,rtol=0,atol=0)
    assert not torch.equal(paused.gpb,state.gpb)
    for value,copy in zip((state.voltage_v,state.gpb,state.land),original):
        torch.testing.assert_close(value,copy,rtol=0,atol=0)


def test_invalid_inputs_and_source_nonfinite_protection():
    args = data()
    with pytest.raises(RuntimeError,match='float64'):
        cuda_tissue.rhs(args[0].float(),args[2],0.)
    with pytest.raises(ValueError,match='integer'):
        GPBLand(dt_ep_s=5.1e-5)
    state = initial_state(args[2])
    state = TissueState(torch.zeros_like(state.voltage_v),state.gpb,state.land)
    # Original GPB has removable singularities exactly at Vm=0, -5, -47.13mV.
    # Preserve its arithmetic in this port and SURFACE source protection rather
    # than silently accepting an invalid zeroed ionic-current trial.
    with pytest.raises(FloatingPointError,match='protection'):
        GPBLand().advance_ode(state,args[3],args[4],args[5],0.)
