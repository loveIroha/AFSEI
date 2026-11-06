"""Fixed FE endpoints retain the original quadrature IB virtual work pair."""
import pytest
import torch
from afsi_torch.solids.planar_tube import TubeGeometry,generate_tube,NeoHookeanTube
from afsi_torch.mac2d.grid import ChannelGrid
from afsi_torch.mac2d.fixed_transfer import FixedEndTriangleTransfer
from afsi_torch.fluid.solvers import SolverOptions


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('support',['faces','sealing-collars'])
def test_fixed_end_projection_and_virtual_work(device,support):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.manual_seed(17)
    g=TubeGeometry(axial_elements=12,thickness_elements=1);m=generate_tube(g,device=device)
    solid=NeoHookeanTube(m,fused=False);grid=ChannelGrid((64,32),(3.2,1.6))
    fixed=torch.unique(m.roots)
    if support=='sealing-collars':
        z=m.X[:,0]-g.inlet_buffer_cm
        fixed=torch.where((z<=.1+1e-12)|(z>=g.length_cm-.1-1e-12))[0]
    transfer=FixedEndTriangleTransfer(grid,solid.geometry,fixed_nodes=fixed,
        mass_backend='graph' if device=='cuda' else 'pcg',fused=False,warm_start=False,
        options=SolverOptions(rtol=1e-12,atol=1e-13,max_iterations=500,check_every=4))
    stencil=transfer.prepare(m.X)
    u=tuple(torch.randn_like(v) for v in grid.zeros(device=device))
    nodal,_=transfer.interpolate(u,stencil)
    torch.testing.assert_close(nodal[fixed],torch.zeros_like(nodal[fixed]),atol=0,rtol=0)
    force=torch.randn_like(m.X)
    spread,_=transfer.spread(force,stencil)
    fluid=grid.volume*sum((a*b).sum() for a,b in zip(u,spread))
    torch.testing.assert_close((nodal*force).sum(),fluid,rtol=1e-10,atol=1e-10)
    # Pure support reactions must not drive the free fluid/solid space.
    reaction=torch.zeros_like(force);reaction[fixed]=force[fixed]
    out,_=transfer.spread(reaction,stencil)
    for v in out:torch.testing.assert_close(v,torch.zeros_like(v),atol=0,rtol=0)
