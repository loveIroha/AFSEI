"""Serial DOLFIN 2019 CG1 interpolation of cell-dependent expressions.

FunctionSpace::interpolate_from_any visits cells in index order and inserts
each cell's DOFs. Shared vertices therefore retain the LAST incident cell.
Do not replace this operation with averaging or a union of endocardial nodes.
Parallel DOLFIN ownership may differ and is outside this serial mapping.
"""
import torch


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
