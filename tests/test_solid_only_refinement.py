import json
from pathlib import Path
import torch
from afsi_torch.solids.circular_tube import CircularTubeGeometry,generate_circular_tube,section_areas,lumen_volume
from afsi_torch.solids.refined_circular_tube import refine_circular_tube
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.ep.tissue import TissueState


def pair():
    g=CircularTubeGeometry(length_cm=2,inner_radius_cm=.3,wall_thickness_cm=.1,
        center_xy_cm=(.7,.7),axial_start_cm=.4,axial_elements=4,circumferential_elements=8,thickness_elements=1)
    m=generate_circular_tube(g);fine,tr=refine_circular_tube(m,g)
    return g,m,fine,tr


def model(m,g):
    return CircularTubeSolid(m,g,young_modulus_dyn_cm2=3e5,poisson_ratio=.3,
        end_spring_dyn_cm3=1e6,tension_scale=0.,end_taper_cm=.15,dt_s=5e-5,
        stretch_feedback='isometric-driver',end_support='full-reference-surface-spring-approximation')


def test_exact_geometry_and_parent_material():
    g,m,f,tr=pair()
    assert len(f.cells)==8*len(m.cells)
    assert torch.equal(f.X[:len(m.X)],m.X)
    assert torch.equal(f.fiber.reshape(len(m.cells),8,3)[:,0],m.fiber)
    assert torch.allclose(section_areas(f.X,f)[::2],section_areas(m.X,m),atol=1e-14,rtol=1e-13)
    assert torch.allclose(lumen_volume(f.X,f),lumen_volume(m.X,m),atol=1e-14,rtol=1e-13)
    c=model(m,g);s=model(f,g)
    assert torch.allclose(s.volumes.reshape(-1,8).sum(1),c.volumes,atol=1e-14,rtol=1e-12)
    assert torch.all(s.element_gradient(f.X).det()>0)


def test_exact_coarse_p1_fields_and_virtual_work():
    g,m,f,tr=pair();torch.manual_seed(710)
    values=torch.randn((len(m.X),4),dtype=m.X.dtype)
    affine=torch.column_stack((m.X,m.X[:,0]*2-m.X[:,1]+3))
    expected=torch.column_stack((f.X,f.X[:,0]*2-f.X[:,1]+3))
    assert torch.allclose(tr.prolong(affine),expected,atol=1e-14,rtol=1e-14)
    assert torch.equal(tr.prolong(values)[:len(m.X)],values)
    coarse_velocity=torch.randn_like(m.X);force=torch.randn_like(f.X)
    assert torch.allclose((force*tr.prolong(coarse_velocity)).sum(),
                          (tr.pullback_force(force)*coarse_velocity).sum(),rtol=1e-12,atol=1e-12)


def test_passive_force_pullback_preserves_coarse_weak_form():
    g,m,f,tr=pair();coarse=model(m,g);fine=model(f,g)
    X=m.X;u=.01*torch.stack((torch.sin(X[:,2]),torch.cos(X[:,2]),X[:,0]-.7),1)
    x=X+u;y=tr.prolong(x)
    fc=coarse.force(x,0.);ff=fine.force(y,0.)
    assert torch.linalg.vector_norm(tr.pullback_force(ff)-fc)/torch.linalg.vector_norm(fc)<1e-11


def test_original_ep_fields_reach_fine_quadrature_without_resampling_odes():
    from types import SimpleNamespace
    g,m,f,tr=pair();solid=model(f,g);solid.field_transfer=tr
    rest=m.X[:,2]*.2+1
    land=torch.stack((.01+.002*m.X[:,0],.03+.001*m.X[:,2],torch.zeros(len(m.X)),torch.zeros(len(m.X))),1)
    zero=m.X.new_zeros(len(m.X))
    solid.set_reference(rest);solid.stage(f.X,SimpleNamespace(land=land),zero,zero)
    q=torch.einsum('qa,eai->eqi',solid.geometry.values,f.X[f.cells])
    assert torch.allclose(solid.land_fields[...,0],.01+.002*q[...,0],atol=1e-14,rtol=1e-13)
    assert torch.allclose(solid.land_fields[...,1],.03+.001*q[...,2],atol=1e-14,rtol=1e-13)
    assert torch.allclose(solid.reference_tension,1+.2*q[...,2],atol=1e-14,rtol=1e-13)


def test_refinement_config_excludes_unimplemented_reverse_feedback():
    import pytest
    from afsi_torch.simulation.circular_tube_config import resolve_config
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'demo/active_tube_3d/configs/tube_single_isometric_full_gpu.json').read_text())
    c['solid_refinement_factor']=2
    resolve_config(c)
    c['active']['stretch_feedback']='circumferential'
    with pytest.raises(ValueError,match='isometric-driver'):resolve_config(c)
