"""Nonzero independent perturbations exercise every diagnostic budget term."""
from pathlib import Path
import runpy
import pytest
import torch
from afsi_torch.p1 import cavity_rim, cavity_volume


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_budget_with_motion_and_cap_defects(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    audit=runpy.run_path(str(Path(__file__).parents[1]/'demo/lv_passive_validation/budget_audit.py'))
    old=torch.tensor([[0.,0.,-2.],[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]],
                     device=device,dtype=torch.float64)
    faces=torch.tensor([[0,1,2],[0,2,3],[0,3,4],[0,4,1]],device=device,dtype=torch.int64)
    rim=cavity_rim(faces)
    x=old*1.03+.04*torch.sin(old+1.)
    dt=.01
    nodal=(x-old)/dt+.02*torch.cos(old+2.)
    cap=torch.sin(old+.7)*.1
    class Rows:
        area=old.new_ones(1)
        def reaction(self, p): return p[0]*cap
        def flow(self, velocity): return velocity
        def material_flow(self, v): return (cap*v).sum().reshape(1)
    flux=old.new_tensor([.43])
    copies=[t.clone() for t in (old,x,nodal,flux)]
    report=audit['decompose_budget'](old,x,faces,rim,dt,nodal,flux,Rows())
    expected=((cavity_volume(x,faces,rim)-cavity_volume(old,faces,rim))/dt
              +flux.sum()-(cap*(x-old)/dt).sum()).item()
    assert report['total_budget_ml_s']==pytest.approx(expected,abs=1e-11)
    keys=('geometry_time_ml_s','motion_residual_ml_s','cap_quadrature_ml_s','spatial_wall_port_ml_s')
    assert all(abs(report[k])>1e-5 for k in keys)
    assert sum(report[k] for k in keys)==pytest.approx(expected,abs=1e-11)
    for value,copy in zip((old,x,nodal,flux),copies):
        assert torch.equal(value,copy)
