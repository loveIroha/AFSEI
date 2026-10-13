"""Conforming, structured P1 tetrahedra for fixed tissue experiments (cm)."""
from dataclasses import dataclass
from itertools import permutations
from math import isfinite
import torch


@dataclass(frozen=True)
class TissueSlabMesh:
    X: torch.Tensor
    cells: torch.Tensor
    shape: tuple
    spacing_cm: tuple


def structured_tissue_slab(lengths_cm, spacing_cm, origin_cm=(0., 0., 0.), *, device='cpu'):
    """Six Freudenthal tetrahedra per brick; shared faces have matching diagonals.

    spacing_cm denotes scalar or three axis spacings, not tetrahedron edge length.
    All lengths must be integral multiples of it; no silent geometry rounding.
    Nodes have ijk ordering with z varying fastest, and cells positive orientation.
    """
    lengths, origin = tuple(lengths_cm), tuple(origin_cm)
    spacings=tuple(spacing_cm) if isinstance(spacing_cm,(tuple,list)) else (spacing_cm,)*3
    if (len(lengths) != 3 or len(origin) != 3 or
            any(not isfinite(v) or v <= 0 for v in lengths) or
            any(not isfinite(v) for v in origin) or
            len(spacings)!=3 or any(not isfinite(h) or h<=0 for h in spacings)):
        raise ValueError('three positive lengths, finite origin and positive spacing required')
    shape = tuple(round(v/h) for v,h in zip(lengths,spacings))
    if any(n < 1 or abs(n*h-v) > 1e-12 for n,v,h in zip(shape,lengths,spacings)):
        raise ValueError('slab lengths must be integer multiples of mesh spacing')
    axes = [torch.linspace(o, o+v, n+1, dtype=torch.float64, device=device)
            for o,v,n in zip(origin, lengths, shape)]
    X = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    stride = ((shape[1]+1)*(shape[2]+1), shape[2]+1, 1)
    indices = torch.meshgrid(*(torch.arange(n, device=device) for n in shape), indexing='ij')
    base = sum(a*s for a,s in zip(indices, stride)).flatten()
    offsets = []
    for order in permutations(range(3)):
        cell = [0, stride[order[0]], stride[order[0]]+stride[order[1]], sum(stride)]
        inversions = sum(order[i] > order[j] for i in range(3) for j in range(i+1,3))
        if inversions % 2:
            cell[1], cell[2] = cell[2], cell[1]
        offsets.append(cell)
    cells = (base[:, None, None]+torch.tensor(offsets, device=device)).reshape(-1, 4)
    return TissueSlabMesh(X, cells, shape, tuple(v/n for v,n in zip(lengths, shape)))
