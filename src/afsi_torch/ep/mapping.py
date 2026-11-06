"""Serial DOLFIN 2019 CG1 interpolation of cell-dependent expressions.

FunctionSpace::interpolate_from_any visits cells in index order and inserts
each cell's DOFs. Shared vertices therefore retain the LAST incident cell.
Do not replace this operation with averaging or a union of endocardial nodes.
Parallel DOLFIN ownership may differ and is outside this serial mapping.
"""
import torch


def orthonormalize_directions(fiber, sheet):
    """Preserve fiber orientation, Gram-Schmidt sheet, right-handed normal."""
    def unit(v):
        norm = torch.linalg.vector_norm(v,dim=-1,keepdim=True)
        if not torch.isfinite(v).all() or (norm <= 1e-10).any():
            raise ValueError('degenerate projected fiber/sheet frame')
        return v/norm
    f = unit(fiber)
    s = unit(sheet)
    s = unit(s-(s*f).sum(-1,keepdim=True)*f)
    n = unit(torch.linalg.cross(f,s))
    return f,s,n


@torch.no_grad()
def l2_project_cell_fields(X, cells, fields):
    """Consistent L2 projection DG0 -> CG1, componentwise, without normalization.

    Uses the consistent mass even when EP itself uses lumped mass. Fiber signs
    must already be coherent; normalization/orthogonalization is not L2 projection.
    """
    from ..p1 import prepare_p1
    from .monodomain import _assemble
    from ..fluid.solvers import pcg, SolverOptions
    if (fields.ndim != 2 or fields.shape[0] != len(cells)
            or fields.device != X.device or fields.dtype != X.dtype
            or not torch.isfinite(fields).all()):
        raise ValueError('finite cell-by-component fields matching mesh required')
    geometry = prepare_p1(X, cells, degree=1)
    volume = geometry.weights.sum(1)
    nv = cells.shape[1]
    local = volume[:, None, None]/(nv*(nv+1))*(
        X.new_ones((nv,nv))+torch.eye(nv,device=X.device,dtype=X.dtype))
    mass = _assemble(cells,local,len(X))
    diagonal = X.new_zeros(len(X)).index_add_(0,cells.flatten(),
        local.diagonal(dim1=-2,dim2=-1).reshape(-1))
    if (diagonal <= 0).any():
        raise ValueError('all projection nodes must belong to a cell')
    rhs = X.new_zeros((len(X),fields.shape[1])).index_add_(0,cells.flatten(),
        (volume[:,None,None]/nv*fields[:,None,:]).expand(-1,nv,-1).reshape(-1,fields.shape[1]))
    result = []
    options = SolverOptions(rtol=1e-12,atol=1e-14)
    for column in rhs.unbind(1):
        value, info = pcg(lambda v: torch.sparse.mm(mass,v[:,None]).flatten(),
                          column,diagonal,options=options)
        if not torch.isfinite(value).all() or info.residual_norm > info.tolerance:
            raise RuntimeError('DG0-to-CG1 L2 projection did not converge')
        result.append(value)
    return torch.stack(result,dim=1)


def source_p1_fields(mesh):
    cells = mesh.cells
    owner = cells.new_full((len(mesh.X),), -1)
    ids = torch.arange(len(cells),device=cells.device).repeat_interleave(4)
    owner.scatter_reduce_(0,cells.reshape(-1),ids,reduce='amax',include_self=True)
    if (owner < 0).any():
        raise ValueError('all EP nodes must have an incident cell')
    endo_cell = torch.zeros(len(cells),dtype=torch.bool,device=cells.device)
    endo_cell[mesh.boundary_cells[mesh.facet_tags == 1]] = True
    return mesh.fiber[owner].clone(),mesh.sheet[owner].clone(),endo_cell[owner].clone(),owner
