import pytest
import torch
from afsi_torch.ep.mapping import l2_project_cell_fields, orthonormalize_directions


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_projection_reference_and_cell_order(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    X=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]],
                   dtype=torch.float64,device=device)
    cells=torch.tensor([[0,1,2,3],[4,1,2,3]],device=device)
    fields=X.new_tensor([[1.,.1,0.,0.,1.,.1],[.9,.3,0.,.1,1.,.2]])
    M=X.new_zeros((5,5)); b=X.new_zeros((5,6))
    for c,g in zip(cells,fields):
        v=torch.linalg.det((X[c[1:]]-X[c[0]]).T).abs()/6
        M[c[:,None],c[None,:]] += v/20*(X.new_ones((4,4))+torch.eye(4,device=device,dtype=X.dtype))
        b[c] += v/4*g
    out=l2_project_cell_fields(X,cells,fields)
    torch.testing.assert_close(out,torch.linalg.solve(M,b),rtol=1e-11,atol=1e-13)
    torch.testing.assert_close(l2_project_cell_fields(X,cells.flip(0),fields.flip(0)),out)
    constant=fields[:1].expand(2,-1)
    torch.testing.assert_close(l2_project_cell_fields(X,cells,constant),fields[:1].expand(5,-1))
    f,s=out[:,:3],out[:,3:]
    n=torch.linalg.cross(f,s)
    torch.testing.assert_close((n*f).sum(1),X.new_zeros(5),atol=1e-14,rtol=0)
    torch.testing.assert_close((n*s).sum(1),X.new_zeros(5),atol=1e-14,rtol=0)
    f,s,n=orthonormalize_directions(f,s)
    frame=torch.stack((f,s,n),dim=-1)
    torch.testing.assert_close(frame.transpose(-1,-2)@frame,
                             torch.eye(3,dtype=X.dtype,device=device).expand(5,-1,-1),atol=1e-14,rtol=0)
    torch.testing.assert_close(torch.linalg.det(frame),X.new_ones(5))
    with pytest.raises(ValueError,match='degenerate'):
        orthonormalize_directions(f,f)
