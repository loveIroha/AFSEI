"""Gaussian cap integration preserves material regions, constraints and work."""
import pytest
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.open_boundary import face_weights
from afsi_torch.ep.mac_ports import MACBasalPorts
from afsi_torch.ep.mac_seal import prepare_local_cap
from afsi_torch.ep.mac_compact_cap import CompactCapRows


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('rule',['gauss3','gauss7','gauss12'])
def test_gaussian_partition_and_both_work_adjoint_pairs(device,rule):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA required')
    X=torch.tensor([[0.,0.,-2.],[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]],device=device,dtype=torch.float64)
    faces=torch.tensor([[0,1,2],[0,2,3],[0,3,4],[0,4,1]],device=device)
    grid=MACGrid((16,)*3,(8.,)*3,(-4.,)*3)
    base=MACBasalPorts(X,faces,grid,subdivision_level=2,with_ports=False)
    improved=MACBasalPorts(X,faces,grid,subdivision_level=2,with_ports=False,integration_rule=rule)
    x=X.clone(); x[:,2]+=.08*x[:,0]*x[:,1]
    p0=base.prepare(x); p=improved.prepare(x)
    r0=prepare_local_cap(p0,patch_width_cells=2.,regions=(0,1,2))
    r=prepare_local_cap(p,patch_width_cells=2.,regions=(0,1,2))
    compact=CompactCapRows(r)
    assert r.count==r0.count
    assert torch.equal(r.patches,r0.patches[improved.integration_parent])
    torch.testing.assert_close(r.area,r0.area,rtol=2e-13,atol=2e-13)
    w=.03*torch.cos(x)
    pressure=torch.arange(r.count,device=device,dtype=X.dtype)+.4
    torch.testing.assert_close(r.material_flow(w),r0.material_flow(w),rtol=1e-12,atol=1e-13)
    torch.testing.assert_close(r.reaction(pressure),r0.reaction(pressure),rtol=1e-12,atol=1e-13)
    u=tuple(torch.sin(grid.coordinates(c,device=device,dtype=X.dtype)[...,c]*1.7+.3) for c in range(3))
    for rows in (r,compact):
        fluid=sum((a*b*face_weights(grid,c,X)).sum() for c,(a,b) in enumerate(zip(u,rows.lift(pressure))))
        torch.testing.assert_close(fluid,(pressure*rows.flow(u)).sum(),rtol=2e-12,atol=1e-12)
        torch.testing.assert_close((rows.reaction(pressure)*w).sum(),(pressure*rows.material_flow(w)).sum(),rtol=2e-12,atol=1e-12)
    # The new rules integrate polynomials over the planar [-1,1]^2 cap.
    planar=improved.prepare(X)
    value=(planar.points[:,0]**2*planar.area_vectors[:,2]).sum()
    torch.testing.assert_close(value,X.new_tensor(4/3),rtol=2e-13,atol=2e-13)
    if rule in ('gauss7','gauss12'):
        value=(planar.points[:,0]**4*planar.area_vectors[:,2]).sum()
        torch.testing.assert_close(value,X.new_tensor(4/5),rtol=2e-13,atol=2e-13)
    if rule=='gauss12':
        value=(planar.points[:,0]**6*planar.area_vectors[:,2]).sum()
        torch.testing.assert_close(value,X.new_tensor(4/7),rtol=2e-13,atol=2e-13)
