"""Constitutive identities, consistent tangent, and actual CUDA execution."""
from dataclasses import replace
from pathlib import Path
import json, math
import pytest
import torch
from afsi_torch.solids.volumetric_stabilization import (
    validate_volumetric_stabilization, logarithmic_volume_energy,
    logarithmic_volume_pk1, deviatoric_logj_pk1)
from afsi_torch.solids.circular_tube import CircularTubeGeometry, generate_circular_tube
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.ep.tissue import initial_state
from afsi_torch.simulation.circular_tube_config import resolve_config


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_cauchy_deviatoric_stress_and_volume_energy_gradient(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA required')
    F=torch.tensor([[[.82,.13,0.],[.03,1.12,.04],[0.,.06,1.03]],
                    [[1.12,0.,.09],[0.,.83,.05],[.02,0.,1.2]]],dtype=torch.float64,device=device)
    P=torch.tensor([[[3.,.4,0.],[.1,5.,.2],[0.,.3,2.]],
                    [[4.,.1,.2],[.4,2.,.3],[.1,.2,5.]]],dtype=F.dtype,device=device)*1e5
    bulk=5.384615384615385e5;J=torch.linalg.det(F)
    Q=deviatoric_logj_pk1(F,P,bulk)
    eye=torch.eye(3,dtype=F.dtype,device=device)
    sigma=P@F.transpose(-1,-2)/J[:,None,None]
    new=Q@F.transpose(-1,-2)/J[:,None,None]
    trace=lambda a:a.diagonal(dim1=-2,dim2=-1).sum(-1)
    dev=lambda a:a-trace(a)[:,None,None]*eye/3
    torch.testing.assert_close(dev(new),dev(sigma),rtol=1e-12,atol=2e-10)
    torch.testing.assert_close(trace(new),3*bulk*torch.log(J)/J,rtol=1e-12,atol=2e-10)
    energy_gradient=torch.vmap(torch.func.grad(lambda f:logarithmic_volume_energy(f,bulk)))(F)
    torch.testing.assert_close(energy_gradient,logarithmic_volume_pk1(F,bulk),rtol=1e-12,atol=1e-9)
    assert float(logarithmic_volume_pk1(eye,bulk).abs().max())==0.


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_full_solid_force_tangent_and_default_identity(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA required')
    g=CircularTubeGeometry(axial_elements=3,circumferential_elements=12,thickness_elements=1)
    m=generate_circular_tube(g,device=device)
    kw=dict(young_modulus_dyn_cm2=3e5,poisson_ratio=.3,end_spring_dyn_cm3=0.,
        tension_scale=1.,end_taper_cm=.15,dt_s=5e-5,stretch_feedback='isometric-driver')
    old=CircularTubeSolid(m,g,**kw)
    explicit=CircularTubeSolid(m,g,**kw,volumetric_stabilization=None)
    new=CircularTubeSolid(m,g,**kw,volumetric_stabilization=dict(
        formulation='deviatoric-logj',bulk_modulus_dyn_cm2=538461.5384615385))
    tissue=initial_state(m.X[:,0]);land=tissue.land.clone();land[:,0]=.03
    tissue=replace(tissue,land=land);zero=torch.zeros(len(m.X),dtype=m.X.dtype,device=device)
    for s in (old,explicit,new):s.stage(m.X,tissue,zero,zero)
    x=m.X.clone();x[:,:2]=m.X.new_tensor(g.center_xy_cm)+.9*(x[:,:2]-m.X.new_tensor(g.center_xy_cm))
    x[:,2]=g.axial_start_cm+1.04*(x[:,2]-g.axial_start_cm)
    assert torch.equal(old.force(x,.06),explicit.force(x,.06))
    torch.testing.assert_close(old.applied_tension(x),new.applied_tension(x),rtol=0,atol=0)
    force=new.force(x,.06)
    assert float(force.sum(0).abs().max())<1e-7
    assert float(torch.linalg.cross(x,force).sum(0).abs().max())<1e-7
    angle=.29;R=x.new_tensor([[math.cos(angle),-math.sin(angle),0.],[math.sin(angle),math.cos(angle),0.],[0.,0.,1.]])
    torch.testing.assert_close(new.force(x@R.T+x.new_tensor([.2,-.1,.3]),.06),force@R.T,rtol=1e-10,atol=1e-7)
    gen=torch.Generator(device=device).manual_seed(20261025)
    d=torch.randn(x.shape,generator=gen,dtype=x.dtype,device=device);d/=d.norm()
    matrix=new.tangent_factory().assemble(x,.06)
    actual=torch.mv(matrix,d.flatten()).reshape_as(x)
    eps=1e-7;fd=(new.force(x+eps*d,.06)-new.force(x-eps*d,.06))/(2*eps)
    torch.testing.assert_close(actual,fd,rtol=3e-5,atol=.02)
    if device=='cuda':
        execution=new.execution_factory()
        torch.testing.assert_close(execution.force(x,.06),force,rtol=1e-11,atol=1e-7)


def test_config_validation_and_explicit_optional_branch():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'demo/active_tube_3d/configs/tube_single_isometric_full_gpu.json').read_text())
    resolve_config(c)
    c['solid']['volumetric_stabilization']=dict(formulation='deviatoric-logj',bulk_modulus_dyn_cm2=538461.5384615385)
    resolve_config(c)
    for value in (-1,float('nan'),True):
        c['solid']['volumetric_stabilization']['bulk_modulus_dyn_cm2']=value
        with pytest.raises(ValueError,match='bulk modulus'):resolve_config(c)
    with pytest.raises(ValueError):validate_volumetric_stabilization(dict(formulation='typo',bulk_modulus_dyn_cm2=1.))
