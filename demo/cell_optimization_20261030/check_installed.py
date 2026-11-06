from pathlib import Path
import sys,json,torch
ROOT=Path('/mnt/large2/gjh/AFSEI');sys.path.insert(0,str(ROOT/'src'))
from afsi_torch.ep.tissue import GPBLand,initial_state
from fused import production
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
n=257;like=torch.zeros(n,device='cuda');state=initial_state(like);copy=state.gpb.clone()
stretch=torch.linspace(.9,1.18,n,device='cuda');zs=torch.linspace(-.1,.1,n,device='cuda');zw=zs*.3
model=GPBLand(5e-5,.002);out,current=model.advance_ode(state,stretch,zs,zw,.5)
expected=production().advance(state.gpb,state.land,state.voltage_v*1000,stretch,zs,zw,500.,.002,25,True)
assert torch.equal(out.gpb,expected[0]) and torch.equal(out.land,expected[1]) and torch.equal(current,expected[2])
assert torch.equal(state.gpb,copy) and out.gpb.data_ptr()!=state.gpb.data_ptr()
report=dict(passed=True,installed_default='node_fused_cached_sdirk',numerical_method='Original SDIRK',
    gpb_max_difference=float((out.gpb-expected[0]).abs().max()),Land_max_difference=float((out.land-expected[1]).abs().max()),
    current_max_difference=float((current-expected[2]).abs().max()),source_state_unchanged=True,
    substeps=25,nodes=n,explicit_default_enabled=False)
(Path(__file__).parent/'installed_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
