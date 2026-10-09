"""Mechanical coefficients from the pinned EP project's Active_land.ufl.

The source binds cell-dependent FiberDirections Expressions directly to H,
J and the stretch projection. Their restriction is constant within each cell,
unlike the globally interpolated CG1 Functions used only by the EP operator.
The existing AFSI DG0 weak-form direction treatment is therefore appropriate.
"""
from dataclasses import replace
from itertools import permutations
import torch
from ..paper_lv import PaperHOParameters

SOURCE_COMMIT = '5bdde73c6be10cefadb005a0fcec5fdb6521f336'
SOURCE_BASAL_CENTER_CM = (7.30042735,7.6448494)
SOURCE_BASAL_BETA = 1e6


def source_material():
    return PaperHOParameters(a=2400.,b=5.08,a_f=14600.,b_f=4.15,a_s=8700.,b_s=1.6,
                             a_fs=3000.,b_fs=1.3,kappa=5e6,active_stretch_slope=4.9)


def source_fsi_config(config):
    """Explicitly select source coefficients without changing AFSI defaults."""
    return replace(config,material=source_material(),beta=SOURCE_BASAL_BETA,
                   basal_center_cm=SOURCE_BASAL_CENTER_CM)


def source_parameters_match(config):
    return (config.material == source_material() and config.beta == SOURCE_BASAL_BETA
            and config.basal_center_cm == SOURCE_BASAL_CENTER_CM)


def source_basal_surface(surface,X):
    """FFC degree-5 symmetric 7-point triangle rule for the source spring.

    Its radial projection is rational in reference coordinates. Replacing
    this rule with another degree-5 rule need not give identical force/J.
    Preserve the old AFSI Duffy rule for inherited checkpoints.
    """
    barycentric=[]; weights=[]
    orbits=[((1/3,)*3,.1125),
            ((.7974269853530873,.10128650732345633,.10128650732345633),.06296959027241358),
            ((.05971587178976981,.47014206410511505,.47014206410511505),.06619707639425308)]
    for orbit,weight in orbits:
        for values in sorted(set(permutations(orbit))):
            barycentric.append(values); weights.append(weight)
    N=X.new_tensor(barycentric); w=X.new_tensor(weights)
    dN=X.new_tensor([[-1.,-1.],[1.,0.],[0.,1.]]).expand(len(w),-1,-1)
    area=torch.linalg.vector_norm(surface.reference_area_vectors,dim=-1)
    return replace(surface,values=N,derivatives=dN,quadrature_weights=w,
                   reference_weights=area[:,None]*w,
                   reference_positions=torch.einsum('qa,bai->bqi',N,X[surface.faces]))
