"""Exact tetrahedral observation and model-only comparison contracts."""
import numpy as np
import pytest
from dataclasses import asdict,replace
from afsi_torch.ep.experiment import EPTissueConfig,TissueInitial
from afsi_torch.simulation.ep_tissue_analysis import sample_slab_p1,_cell_signature
from afsi_torch.geometry.tissue_slab import structured_tissue_slab
import torch
from afsi_torch.ep import Monodomain,MonodomainNumerics


def test_diagonal_interpolation_reproduces_affine_p1_field_and_node_values():
    mesh=structured_tissue_slab((.2,.1,.1),.05)
    X=mesh.X.numpy()
    values=X@np.array([2.,-3.,4.])+5
    points=np.linspace(0,1,101)[:,None]*np.array([.2,.1,.1])
    np.testing.assert_allclose(sample_slab_p1(X,mesh.shape,points,values),points@np.array([2.,-3.,4.])+5,
                               rtol=0,atol=8e-15)
    values[1:]=np.nan
    assert sample_slab_p1(X,mesh.shape,X[:1],values)[0]==5.
    with pytest.raises(ValueError,match='inside'):
        sample_slab_p1(X,mesh.shape,np.array([[.3,0.,0.]]),values)


def test_external_signature_allows_only_cell_model_and_output_changes():
    gpb=replace(EPTissueConfig(),initial=TissueInitial(state='model_native_rest',voltage_v=None,conditioning_s=.2))
    ord=replace(gpb,cell_model='ord2011_epi')
    assert _cell_signature(asdict(gpb))==_cell_signature(asdict(ord))
    legacy=asdict(gpb);legacy['domain'].pop('axis_spacing_cm')
    assert _cell_signature(legacy)==_cell_signature(asdict(ord))
    legacy['ep_numerics'].pop('time_scheme')
    be=replace(gpb,ep_numerics=replace(gpb.ep_numerics,time_scheme='backward_euler'))
    assert _cell_signature(legacy)==_cell_signature(asdict(be))
    assert _cell_signature(legacy)!=_cell_signature(asdict(ord))
    for changed in (replace(ord,domain=replace(ord.domain,spacing_cm=.025)),
                    replace(ord,time=replace(ord.time,dt_ep_s=1e-5,dt_ode_ms=.001)),
                    replace(ord,parameters=replace(ord.parameters,surface_to_volume_per_cm=1.)),
                    replace(ord,initial=replace(ord.initial,conditioning_s=.1))):
        assert _cell_signature(asdict(gpb))!=_cell_signature(asdict(changed))


def test_embedded_planar_cable_interior_operator_is_exact_longitudinal_laplacian():
    mesh=structured_tissue_slab((.2,.1,.1),(.0125,.1,.1))
    f=mesh.X.new_tensor([1.,0.,0.]).expand(len(mesh.X),-1)
    s=mesh.X.new_tensor([0.,1.,0.]).expand(len(mesh.X),-1)
    ep=Monodomain(mesh.X,mesh.cells,f,s,numerics=MonodomainNumerics(backend='torch_pcg_jacobi'))
    x=mesh.X[:,0]
    interior=(x>1e-12)&(x<.2-1e-12)
    action=torch.sparse.mm(ep.stiffness,(x*x)[:,None]).flatten()/ep.mass_diagonal
    torch.testing.assert_close(action[interior],torch.full_like(action[interior],-2*.117),rtol=1e-12,atol=1e-13)
