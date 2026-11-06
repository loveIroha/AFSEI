from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
import math
import pytest
import torch
from afsi_torch.solids.circular_tube import CircularTubeGeometry,generate_circular_tube
from afsi_torch.ep.layered_circular_tube import LayeredCircularTubeSolid
from afsi_torch.simulation.circular_tube_config import resolve_config


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    return request.param


def case(device):
    path=Path(__file__).resolve().parents[1]/'demo/active_tube_3d/configs/tube_sharifi_micron_layered_check_gpu.json'
    c=json.loads(path.read_text());g,_,_,_=resolve_config(c)
    g=replace(g,axial_elements=3,circumferential_elements=12)
    m=generate_circular_tube(g,device=device)
    s=LayeredCircularTubeSolid(m,g,layers=c['solid']['layers'],end_spring_dyn_cm3=c['solid']['end_spring_dyn_cm3'],
        **{k:v for k,v in c['active'].items() if k not in ('mechanism','rest_reference')},dt_s=c['time']['fsi_dt_s'])
    n=len(s.ep_nodes);land=m.X.new_zeros((n,4));land[:,0]=.02
    z=m.X.new_zeros(n);t=SimpleNamespace(land=land)
    s.set_reference(z);s.stage(m.X,t,z,z)
    return c,g,m,s


def test_layer_volumes_interface_and_ep_domain(device):
    _,g,m,s=case(device)
    polygon=.5*g.circumferential_elements*math.sin(2*math.pi/g.circumferential_elements)
    radii=[g.inner_radius_cm,*g.radial_interfaces_cm,g.inner_radius_cm+g.wall_thickness_cm]
    for i in range(2):
        actual=s.volumes[m.cell_layer==i].sum()
        assert float(actual)==pytest.approx(polygon*(radii[i+1]**2-radii[i]**2)*g.length_cm,rel=2e-12)
    inner=torch.unique(m.cells[m.cell_layer==0]);outer=torch.unique(m.cells[m.cell_layer==1])
    interface=inner[torch.isin(inner,outer)]
    r=(m.X[interface,:2]-m.X.new_tensor(g.center_xy_cm)).norm(dim=-1)
    torch.testing.assert_close(r,torch.full_like(r,g.radial_interfaces_cm[0]),atol=1e-16,rtol=1e-12)
    assert len(interface)==(g.axial_elements+1)*g.circumferential_elements
    assert torch.equal(s.ep_nodes,outer)
    assert not torch.isin(s.ep_nodes,inner[~torch.isin(inner,interface)]).any()


def test_jelly_passive_and_muscle_generates_contraction(device):
    _,g,m,s=case(device)
    F=s.element_gradient(m.X)
    P=torch.vmap(lambda f,*v:s.stress(f,v,torch.tensor(0.,device=f.device)))(F,*s.cell_fields)
    assert float(P[m.cell_layer==0].abs().max())<1e-9
    assert float(s.applied_tension(m.X)[m.cell_layer==0].abs().max())==0.
    assert float(s.applied_tension(m.X)[m.cell_layer==1].max())>0
    force=s.force(m.X,0.)
    expansion=m.X.clone();expansion[:,:2]-=m.X.new_tensor(g.center_xy_cm);expansion[:,2]=0
    assert float((force*expansion).sum())<0
    assert float(force.sum(0).abs().max())<1e-12


def test_material_and_full_end_spring_tangent(device):
    _,g,m,s=case(device)
    # Test full force/tangent, including material contrast and all spring axes.
    x=m.X.clone()
    center=x.new_tensor(g.center_xy_cm)
    x[:,:2]=center+.99*(x[:,:2]-center)
    x[:,0]+=.000001*torch.sin(x[:,2]*100)
    torch.manual_seed(12)
    v=torch.randn_like(x);v/=v.norm()
    A=s.tangent_factory().assemble(x,.001)
    actual=torch.mv(A,v.flatten()).reshape_as(v)
    eps=1e-9
    fd=(s.force(x+eps*v,.001)-s.force(x-eps*v,.001))/(2*eps)
    torch.testing.assert_close(actual,fd,atol=2e-5,rtol=2e-5)
    assert s.diagnostics(x)['end_condition_is_exact_dirichlet'] is False


def test_paper_units_and_distinct_config():
    c,g,_,s=case('cpu')
    assert g.length_cm==.018 and g.inner_radius_cm==.0025
    assert c['fluid']['viscosity_dyn_s_per_cm2']==.08
    assert c['fluid']['density_g_per_cm3']==1.035
    assert c['solid']['layers'][0]['shear_modulus_dyn_cm2']==40.
    assert c['solid']['layers'][1]['shear_modulus_dyn_cm2']==100000.
    assert c['solid']['layers'][1]['poisson_ratio']==.49
    from afsi_torch.circulation.rc_outlet import parse_rc
    p=parse_rc(c['circulation'],dimension=3)
    assert p is not None
    bad=json.loads(json.dumps(c));bad['solid']['layers'][0]['active']=True
    with pytest.raises(ValueError,match='jelly is passive'):resolve_config(bad)
