import pytest
import torch
from afsi_torch.solids.planar_tube import TubeGeometry,generate_tube,NeoHookeanTube,neo_hookean_plane_strain_pk1
from afsi_torch.ep.planar_tube import PairedHoopLoad,land_resting_equilibrium
from afsi_torch.ep.land import land_ode_step,land_tension
from afsi_torch.ep.tissue import initial_state
from afsi_torch.mac2d.grid import ChannelGrid
from afsi_torch.mac2d.transfer import TriangleTransfer


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_passive_energy_active_balance_and_adjoint_transfer(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    cfg=TubeGeometry(axial_elements=12,thickness_elements=1)
    mesh=generate_tube(cfg,device=device)
    solid=NeoHookeanTube(mesh,fused=False)
    x=(mesh.X+.0001*torch.sin(mesh.X*3)).requires_grad_()
    g=solid.geometry
    F=torch.einsum('eai,eqaJ->eqiJ',x[g.cells],g.gradients)
    logj=torch.linalg.det(F).log()
    energy=(g.weights*(.5*solid.shear*(F.square().sum((-1,-2))-2)-solid.shear*logj+.5*solid.lame*logj.square())).sum()
    u=torch.einsum('qa,bai->bqi',solid.root_values,x[mesh.roots]-mesh.X[mesh.roots])
    energy=energy+.5*solid.end_spring*(solid.root_weights*u.square().sum(-1)).sum()
    expected=-torch.autograd.grad(energy,x)[0]
    torch.testing.assert_close(solid.force(x.detach()),expected,atol=2e-8,rtol=1e-10)
    torch.testing.assert_close(solid.force(mesh.X),torch.zeros_like(mesh.X),atol=2e-8,rtol=0)
    load=PairedHoopLoad(mesh,cfg,fused=False)
    tissue=initial_state(mesh.X[:mesh.vertex_count,0])
    tissue.land[:,0]=.01
    stretch=load.nodal_stretch(mesh.X)
    zeros=torch.zeros_like(stretch)
    load.set_history(tissue,stretch,zeros,zeros,5e-5)
    force=load.force(x.detach())
    torch.testing.assert_close(force.sum(0),torch.zeros(2,dtype=x.dtype,device=device),atol=1e-10,rtol=0)
    torque=(x.detach()[:,0]*force[:,1]-x.detach()[:,1]*force[:,0]).sum()
    assert abs(float(torque)) < 1e-9
    assert force[load.lower_edges.reshape(-1),1].sum()>0
    assert force[load.upper_edges.reshape(-1),1].sum()<0
    # The unchanged quadrature transfer must still preserve virtual work.
    grid=ChannelGrid((64,32),(3.2,1.6))
    transfer=TriangleTransfer(grid,g,mass_backend='pcg',fused=False,warm_start=False)
    stencil=transfer.prepare(mesh.X)
    velocity=tuple(torch.randn_like(a) for a in grid.zeros(device=device))
    U,_=transfer.interpolate(velocity,stencil)
    density,_=transfer.spread(force,stencil)
    torch.testing.assert_close((U*force).sum(),grid.volume*sum((a*b).sum() for a,b in zip(velocity,density)),atol=1e-7,rtol=1e-9)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_land_rest_fixed_point_and_zero_reference_force(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    cfg=TubeGeometry(axial_elements=8,thickness_elements=1)
    mesh=generate_tube(cfg,device=device)
    load=PairedHoopLoad(mesh,cfg,fused=False)
    tissue=initial_state(mesh.X[:mesh.vertex_count,0])
    ca=tissue.gpb[:,37]*1000
    stretch=torch.ones_like(ca); z=torch.zeros_like(ca)
    state=land_resting_equilibrium(ca,stretch)
    torch.testing.assert_close(land_ode_step(state,ca,stretch,z,z,.002),state,atol=1e-15,rtol=1e-13)
    tissue.land.copy_(state)
    tension=land_tension(stretch,stretch,state[:,0],state[:,1],z,z,5e-5)
    load.set_reference(tension)
    load.set_history(tissue,stretch,z,z,5e-5)
    assert load.force(mesh.X).abs().max() < 1e-10
