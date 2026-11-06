"""Omitted-direction stretch and paired-traction physical invariants."""
from types import SimpleNamespace
import pytest
import torch
from afsi_torch.solids.planar_tube import TubeGeometry, generate_tube
from afsi_torch.ep.planar_tube import PairedHoopLoad


@pytest.fixture(params=['cpu','cuda'])
def specimen(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    g = TubeGeometry(axial_elements=12,thickness_elements=1)
    mesh = generate_tube(g,device=request.param)
    return g,mesh


def prepared(g,mesh,model):
    load = PairedHoopLoad(mesh,g,stretch_model=model,fused=False)
    n = mesh.vertex_count
    land = mesh.X.new_zeros((n,4));land[:,0] = .03;land[:,1] = .08
    stretch = load.nodal_stretch(mesh.X)
    load.set_history(SimpleNamespace(land=land),stretch,torch.zeros_like(stretch),torch.zeros_like(stretch),5e-5)
    return load


def test_omitted_plane_strain_direction_is_isometric_under_gap_closure(specimen):
    g,mesh = specimen
    iso = prepared(g,mesh,'plane-strain-isometric')
    old = prepared(g,mesh,'lumen-gap')
    x = mesh.X.clone();x[:,1] = g.center_y_cm+.5*(x[:,1]-g.center_y_cm)
    torch.testing.assert_close(iso.stretch_sections(x),mesh.X.new_full((g.axial_elements+1,),.5))
    torch.testing.assert_close(iso.nodal_stretch(x),torch.ones_like(iso.nodal_stretch(x)))
    torch.testing.assert_close(old.nodal_stretch(x),torch.full_like(old.nodal_stretch(x),.5))
    # A geometric gap reduction cannot extinguish out-of-plane isometric
    # muscle activity; the explicitly selected legacy proxy retains that law.
    torch.testing.assert_close(iso.force(x),iso.force(mesh.X),rtol=1e-11,atol=1e-9)
    assert float(iso.force(x).norm()) > 0
    torch.testing.assert_close(old.force(x),torch.zeros_like(x),atol=1e-12,rtol=0)


def test_paired_force_balance_rigid_rotation_and_actual_traction(specimen):
    g,mesh = specimen;load = prepared(g,mesh,'plane-strain-isometric')
    x = mesh.X.clone();force = load.force(x)
    theta = x.new_tensor(.37);co,si = theta.cos(),theta.sin()
    rotation = torch.stack((torch.stack((co,-si)),torch.stack((si,co))))
    rotated = x@rotation.T+x.new_tensor((.2,-.15))
    f = load.force(rotated)
    torch.testing.assert_close(f,force@rotation.T,atol=1e-9,rtol=1e-11)
    torch.testing.assert_close(f.sum(0),x.new_zeros(2),atol=1e-9,rtol=0)
    torque = (rotated[:,0]*f[:,1]-rotated[:,1]*f[:,0]).sum()
    assert abs(float(torque)) < 1e-8
    data = load.diagnostics(x)
    lower = mesh.X[:,1] < g.center_y_cm
    torch.testing.assert_close(force[lower,1].sum()/g.length_cm,
        x.new_tensor(data['active_traction_mean_dyn_cm2']),atol=1e-9,rtol=1e-11)
    assert data['active_traction_max_dyn_cm2'] >= data['active_traction_mean_dyn_cm2'] > 0


def test_legacy_default_and_invalid_stretch_model(specimen):
    g,mesh = specimen
    assert PairedHoopLoad(mesh,g,fused=False).stretch_model == 'lumen-gap'
    with pytest.raises(ValueError,match='stretch_model'):
        PairedHoopLoad(mesh,g,stretch_model='unknown',fused=False)


def test_fixed_connector_region_is_not_actively_loaded(specimen):
    g,mesh = specimen
    load = PairedHoopLoad(mesh,g,stretch_model='plane-strain-isometric',end_inactive_cm=.2,fused=False)
    n=mesh.vertex_count;land=mesh.X.new_zeros((n,4));land[:,0]=.03
    stretch=load.nodal_stretch(mesh.X);z=torch.zeros_like(stretch)
    load.set_history(SimpleNamespace(land=land),stretch,z,z,5e-5)
    fields=load.load_fields(mesh.X)
    distance=torch.minimum(fields['points'][...,0]-g.inlet_buffer_cm,
        g.inlet_buffer_cm+g.length_cm-fields['points'][...,0])
    assert torch.all(fields['traction'][distance < .2] == 0)
    assert torch.all(fields['traction'][distance > .3] > 0)
