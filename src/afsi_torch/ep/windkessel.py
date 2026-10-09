"""Flow-driven RCR port relation for FSI, separate from LV wall traction.

This module does not create or seal AV/MV ports in the current MAC geometry.
Q must come from the actual oriented fluid port, not -d(cavity volume)/dt.
"""
from dataclasses import dataclass
from math import isfinite
import torch

MMHG_TO_DYN_CM2 = 1333.22368421  # retain the source Windkessel conversion


@dataclass(frozen=True)
class WindkesselParameters:
    rc_mmhg_s_per_ml: float = 0.05
    rp_mmhg_s_per_ml: float = 1.05
    compliance_ml_per_mmhg: float = 1.30
    venous_mmhg: float = 8.0

    def __post_init__(self):
        positive = (self.rc_mmhg_s_per_ml, self.rp_mmhg_s_per_ml, self.compliance_ml_per_mmhg)
        if any(not isfinite(v) or v <= 0 for v in positive) or not isfinite(self.venous_mmhg):
            raise ValueError('positive finite RCR parameters and finite venous pressure required')


@dataclass(frozen=True)
class WindkesselTrial:
    pc_mmhg: object
    outlet_mmhg: object
    flow_ml_per_s: object
    impedance_mmhg_s_per_ml: float

    @property
    def outlet_cgs(self):
        return self.outlet_mmhg*MMHG_TO_DYN_CM2


def flow_driven_trial(old_pc_mmhg, flow_ml_per_s, dt_s, parameters=None):
    """Pure BE trial; caller retains old_pc during every nonlinear iterate.

    Publish trial.pc_mmhg only after the full FSI step is accepted. Proximal
    resistance appears exactly once. This pressure belongs at the arterial
    port; do not apply it again to the ventricular endocardial wall.
    """
    p = parameters or WindkesselParameters()
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError('positive finite time step in seconds required')
    for value in (old_pc_mmhg, flow_ml_per_s):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError('scalar finite Windkessel pressure/flow required')
        elif not isfinite(value):
            raise ValueError('finite Windkessel pressure/flow required')
    denom = p.compliance_ml_per_mmhg+dt_s/p.rp_mmhg_s_per_ml
    pc = (p.compliance_ml_per_mmhg*old_pc_mmhg+
          dt_s*(flow_ml_per_s+p.venous_mmhg/p.rp_mmhg_s_per_ml))/denom
    return WindkesselTrial(pc, pc+p.rc_mmhg_s_per_ml*flow_ml_per_s,
                          flow_ml_per_s, p.rc_mmhg_s_per_ml+dt_s/denom)


def oriented_triangle_flux(triangles_cm, velocity_cm_per_s, grid_velocity_cm_per_s=None):
    """Exact P1 flux across CURRENT triangles oriented outward from the fluid.

    Shapes are (faces,3,3). The optional port/grid velocity gives ALE relative
    flux. MAC interpolation to these nodes is a separate, pending operation.
    """
    if (triangles_cm.ndim != 3 or triangles_cm.shape[1:] != (3, 3)
            or not triangles_cm.is_floating_point()):
        raise ValueError('port triangles must have shape (faces,3,3)')
    for value in (velocity_cm_per_s, grid_velocity_cm_per_s):
        if value is not None and (value.shape != triangles_cm.shape or value.dtype != triangles_cm.dtype
                or value.device != triangles_cm.device or not torch.isfinite(value).all()):
            raise ValueError('finite port velocity with triangle shape, dtype and device required')
    if not torch.isfinite(triangles_cm).all():
        raise ValueError('finite port coordinates required')
    relative = velocity_cm_per_s if grid_velocity_cm_per_s is None else velocity_cm_per_s-grid_velocity_cm_per_s
    edges = triangles_cm[:, 1:]-triangles_cm[:, :1]
    area_vector = 0.5*torch.linalg.cross(edges[:, 0], edges[:, 1])
    if (torch.linalg.vector_norm(area_vector, dim=-1) <= 0).any():
        raise ValueError('nondegenerate port triangles required')
    return (relative.mean(1)*area_vector).sum()
