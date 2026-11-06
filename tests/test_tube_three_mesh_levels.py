"""Check physical-domain and transfer invariance for the new spatial levels."""
from itertools import combinations
import numpy as np
import pytest
import torch
from afsi_torch.solids.circular_tube import CircularTubeGeometry,generate_circular_tube,section_areas,lumen_volume
from afsi_torch.solids.refined_circular_tube import refine_circular_tube
from afsi_torch.ep.circular_tube import CircularTubeSolid


def wall(mesh,g):
    return CircularTubeSolid(mesh,g,young_modulus_dyn_cm2=3e5,poisson_ratio=.3,
        end_spring_dyn_cm3=1e6,tension_scale=0.,end_taper_cm=.15,dt_s=5e-5,
        stretch_feedback='isometric-driver',end_support='full-reference-surface-spring-approximation',
        volumetric_stabilization=dict(formulation='deviatoric-logj',bulk_modulus_dyn_cm2=538461.5384615385))


@pytest.mark.parametrize('factor',[2,3,4])
def test_same_polyhedron_conforming_surface_and_virtual_work(factor):
    g=CircularTubeGeometry(axial_elements=3,circumferential_elements=8,thickness_elements=1)
    coarse=generate_circular_tube(g);fine,tr=refine_circular_tube(coarse,g,factor)
    a,b=wall(coarse,g),wall(fine,g)
    assert len(fine.cells)==len(coarse.cells)*factor**3
    assert torch.equal(fine.X[:len(coarse.X)],coarse.X)
    torch.testing.assert_close(b.volumes.reshape(-1,factor**3).sum(1),a.volumes,rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(section_areas(fine.X,fine)[::factor],section_areas(coarse.X,coarse),rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(lumen_volume(fine.X,fine),lumen_volume(coarse.X,coarse),rtol=1e-12,atol=1e-14)
    # Every stored child boundary facet must actually belong to exactly one tet.
    cells=fine.cells.numpy();faces=np.concatenate([np.sort(cells[:,q],axis=1) for q in combinations(range(4),3)])
    unique,count=np.unique(faces,axis=0,return_counts=True)
    assert count.max()==2
    boundary={tuple(v) for v in unique[count==1]}
    for parent,child in ((coarse.inner_faces,fine.inner_faces),(coarse.end_faces,fine.end_faces)):
        assert len(child)==len(parent)*factor**2
        assert all(tuple(v) in boundary for v in np.sort(child.numpy(),axis=1))
    generator=torch.Generator().manual_seed(20261025)
    velocity=torch.randn(coarse.X.shape,generator=generator,dtype=coarse.X.dtype)
    force=torch.randn(fine.X.shape,generator=generator,dtype=fine.X.dtype)
    torch.testing.assert_close((tr.prolong(velocity)*force).sum(),(velocity*tr.pullback_force(force)).sum(),rtol=1e-12,atol=1e-12)
    u=.005*torch.stack((torch.sin(coarse.X[:,2]),torch.cos(coarse.X[:,2]),coarse.X[:,0]),1)
    x=coarse.X+u
    # Exact coarse weak-form pullback for this piecewise affine passive state.
    fc=a.force(x,0.);ff=tr.pullback_force(b.force(tr.prolong(x),0.))
    torch.testing.assert_close(ff,fc,rtol=2e-10,atol=1e-9)


def test_default_factor_two_is_bitwise_unchanged():
    import importlib.util,os
    from pathlib import Path
    old=Path(os.environ['AFSEI_REFINEMENT_BEFORE'])
    spec=importlib.util.spec_from_file_location('afsi_torch.solids._refinement_before',old)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    g=CircularTubeGeometry(axial_elements=3,circumferential_elements=8,thickness_elements=1)
    m=generate_circular_tube(g);old_mesh,old_tr=module.refine_circular_tube(m,g);new_mesh,new_tr=refine_circular_tube(m,g)
    for key in ('X','cells','fiber','sheet','inner_faces','end_faces','inner_rings'):
        assert torch.equal(getattr(old_mesh,key),getattr(new_mesh,key)),key
    assert torch.equal(old_tr.ids,new_tr.ids) and torch.equal(old_tr.weights,new_tr.weights)
