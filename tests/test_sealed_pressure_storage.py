"""Exact geometry storage preserves the pressure operator and V-cycle solve."""
import os
import pytest
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.sealed_pressure import SealedPressure
from afsi_torch.mac.multigrid import MGOptions


@pytest.mark.parametrize('nonbinary',[False,True])
def test_lossless_coefficients_and_owned_double_pressure(nonbinary):
    device=os.environ.get('AFSEI_TEST_DEVICE','cpu')
    grid=MACGrid((16,16,16),(1.3,1.3,1.3))
    masks=[torch.ones(grid.face_shape(c),device=device,dtype=torch.float64) for c in range(3)]
    masks[0][8,2:14,4:16]=0.
    if nonbinary:masks[1][4,5,6]=.1
    kwargs=dict(options=MGOptions(),fused=True,graphs=True,max_iterations=600)
    reference=SealedPressure(grid,tuple(masks),(),**kwargs)
    compact=SealedPressure(grid,tuple(masks),(),compact_geometry=True,**kwargs)
    for canonical,packed in zip(compact.masks,compact.execution_masks):
        for a,b in zip(canonical,packed):assert torch.equal(a,b.to(a.dtype))
    if nonbinary:assert compact.execution_masks[0][1].dtype==torch.float64
    else:assert all(m.dtype==torch.bool for m in compact.execution_masks[0])
    exact=torch.sin(torch.arange(16**3,device=device,dtype=torch.float64)).reshape(grid.shape)
    rhs=reference.actions[0](exact)
    torch.testing.assert_close(compact.actions[0](exact),rhs,rtol=0,atol=0)
    torch.testing.assert_close(reference.precondition(rhs),compact.precondition(rhs),rtol=0,atol=0)
    x0,i0=reference.solve(rhs);x1,i1=compact.solve(rhs)
    torch.testing.assert_close(x0,x1,rtol=0,atol=0)
    assert x1.dtype==torch.float64 and i0['iterations']==i1['iterations']
    saved=x1.clone();compact.solve(rhs*.5)
    assert torch.equal(x1,saved)
