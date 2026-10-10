"""Volume-consistent transport covectors for the unchanged P1/MAC IB method.

The cavity is measured with the existing mean-rim cap. Wall and cap rates
use the SAME projected nodal velocity J u as the original motion equation.
They are geometric boundary transport, never measured MV/AV outlet flows.
No surface constraints, pressure laws or physical loads are changed here.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from ..p1 import cavity_rim,cavity_volume
from ..mac.open_boundary import face_weights


def closed_volume_covectors(x,faces,rim):
    """Exact current P1 wall/cap normal integrals, shape (2, node_count, 3).

    The rim-centre velocity is its nodal mean, including its transpose
    contribution to every rim node. Source solid-outward faces are reversed
    for the blood-outward wall; cap follows the source oriented rim edges.
    This small hot-path primitive assumes validated model geometry.
    """
    nodes=x[faces]
    wall_area=-.5*torch.linalg.cross(nodes[:,1]-nodes[:,0],nodes[:,2]-nodes[:,0])
    wall=x.new_zeros(x.shape).index_add(0,faces.flatten(),
        (wall_area[:,None].expand(-1,3,-1)/3).reshape(-1,3))
    edge=x[rim]; center=x[rim[:,0]].mean(0)
    cap_area=.5*torch.linalg.cross(edge[:,1]-edge[:,0],center-edge[:,0])
    cap=x.new_zeros(x.shape).index_add(0,rim.flatten(),
        (cap_area[:,None].expand(-1,2,-1)/3).reshape(-1,3))
    cap=cap.index_add(0,rim[:,0],(cap_area.sum(0)/(3*len(rim))).expand(len(rim),-1))
    return torch.stack((wall,cap))


def closed_volume_gradient(x,faces,rim):
    """Analytic gradient of the existing cavity_volume, including the cap."""
    return closed_volume_covectors(x,faces,rim).sum(0)


@dataclass(frozen=True)
class ClosedVolumeTransport:
    previous_x: torch.Tensor
    x: torch.Tensor
    faces: torch.Tensor
    rim: torch.Tensor
    dt: float
    covectors: torch.Tensor
    current_covectors: torch.Tensor

    @classmethod
    @torch.no_grad()
    def prepare(cls,previous_x,x,faces,dt,*,rim=None):
        if (not isfinite(dt) or dt<=0 or previous_x.ndim!=2 or previous_x.shape[1]!=3
                or previous_x.dtype!=torch.float64 or x.shape!=previous_x.shape
                or x.dtype!=previous_x.dtype or x.device!=previous_x.device
                or not torch.isfinite(previous_x).all() or not torch.isfinite(x).all()
                or faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype!=torch.int64
                or faces.device!=x.device or len(faces)==0
                or (faces<0).any() or (faces>=len(x)).any()):
            raise ValueError('positive dt, matching finite float64 geometry and oriented P1 faces required')
        rim=cavity_rim(faces) if rim is None else rim
        if (rim.ndim!=2 or rim.shape[1]!=2 or len(rim)<3 or rim.dtype!=torch.int64
                or rim.device!=x.device or (rim<0).any() or (rim>=len(x)).any()):
            raise ValueError('valid matching oriented rim required')
        if cavity_volume(previous_x,faces,rim)<=0 or cavity_volume(x,faces,rim)<=0:
            raise ValueError('positive closed endpoint volumes required')
        old=closed_volume_covectors(previous_x,faces,rim)
        middle=closed_volume_covectors((previous_x+x)/2,faces,rim)
        current=closed_volume_covectors(x,faces,rim)
        # Every area covector is quadratic along the straight nodal motion.
        # Simpson is exact, including a moving/warped mean-rim cap.
        return cls(previous_x,x,faces,rim,dt,(old+4*middle+current)/6,current)

    def _nodal(self,value):
        if (value.shape!=self.x.shape or value.dtype!=self.x.dtype or value.device!=self.x.device
                or not torch.isfinite(value).all()):
            raise ValueError('matching finite float64 nodal field required')

    def rates(self,nodal_velocity):
        """Wall/cap outward geometric transport rates (ml/s) of J u."""
        self._nodal(nodal_velocity)
        return torch.einsum('rni,ni->r',self.covectors,nodal_velocity)

    def pressure_covector(self,pressure_cgs):
        """Transpose of rates, for adjoint checks only; not a source load."""
        if (pressure_cgs.shape!=(2,) or pressure_cgs.dtype!=self.x.dtype
                or pressure_cgs.device!=self.x.device or not torch.isfinite(pressure_cgs).all()):
            raise ValueError('two matching finite wall/cap pressures required')
        return torch.einsum('r,rni->ni',pressure_cgs,self.covectors)

    def negative_adjoint_density(self,transfer,stencil,pressure_cgs):
        """-J* gbar p under the original fluid/solid work pairing.

        Uses the existing paired spread and mass solve. This constructs an
        audit field only; it is never added to simulation force or velocity.
        Caller owns transfer scratch/warm-start policy, as in original IB.
        """
        return transfer.spread(-self.pressure_covector(pressure_cgs),stencil)

    @torch.no_grad()
    def audit(self,nodal_velocity):
        self._nodal(nodal_velocity)
        material=(self.x-self.previous_x)/self.dt
        projected=self.rates(nodal_velocity); geometric=self.rates(material)
        secant=(cavity_volume(self.x,self.faces,self.rim)
                -cavity_volume(self.previous_x,self.faces,self.rim))/self.dt
        mismatch=nodal_velocity-material
        predicted=torch.einsum('ni,ni->',self.covectors.sum(0),mismatch)
        current=torch.einsum('rni,ni->r',self.current_covectors,nodal_velocity)
        return dict(scope='time-integrated wall/cap transport of original IB projected velocity; not outlet flow',
            dt_s=self.dt,projected_wall_transport_ml_s=projected[0].item(),
            projected_cap_transport_ml_s=projected[1].item(),
            projected_closed_transport_ml_s=projected.sum().item(),
            material_wall_transport_ml_s=geometric[0].item(),
            material_cap_transport_ml_s=geometric[1].item(),
            secant_volume_rate_ml_s=secant.item(),
            swept_geometry_identity_error_ml_s=abs((geometric.sum()-secant).item()),
            projected_volume_balance_error_ml_s=(projected.sum()-secant).item(),
            predicted_projection_defect_ml_s=predicted.item(),
            transport_defect_identity_error_ml_s=abs((projected.sum()-geometric.sum()-predicted).item()),
            projected_motion_residual_l2_cm=(self.dt*mismatch.norm()).item(),
            current_wall_transport_ml_s=current[0].item(),current_cap_transport_ml_s=current[1].item(),
            current_closed_transport_ml_s=current.sum().item(),
            endpoint_geometry_rate_difference_ml_s=(current.sum()-projected.sum()).item(),
            source_endocardial_load_changed=False,source_state_modified=False,
            pointwise_impermeability_established=False,
            windkessel_actual_port_flow_connected=False,three_field_integration_implemented=False)


@torch.no_grad()
def audit_projected_transport(transport,transfer,stencil,velocity,*,nodal_velocity=None):
    """Offline original-IB transport/adjoint check; clears transfer scratch.

    No simulation field is overwritten. The two probe pressures only test
    the algebraic pairing; neither is a valve or physical cavity pressure.
    An already computed J u avoids a second interpolation in surface audits.
    """
    try:
        transfer.reset_warm_start()
        if nodal_velocity is None:
            nodal_velocity,_=transfer.interpolate(velocity,stencil)
        report=transport.audit(nodal_velocity)
        probe=transport.x.new_tensor([8000.,13000.])
        transfer.reset_warm_start()
        density,_=transport.negative_adjoint_density(transfer,stencil,probe)
        fluid_power=sum((u*f*face_weights(transfer.grid,c,transport.x)).sum()
            for c,(u,f) in enumerate(zip(velocity,density)))
        expected=-(probe*transport.rates(nodal_velocity)).sum()
        error=abs((fluid_power-expected).item())
        relative=error/max(abs(fluid_power.item()),abs(expected.item()),1.)
        report.update(adjoint_probe_pressure_dyn_cm2=probe.cpu().tolist(),
            adjoint_fluid_power_erg_s=fluid_power.item(),adjoint_expected_power_erg_s=expected.item(),
            adjoint_power_error_erg_s=error,adjoint_power_relative_error=relative,
            probe_density_applied_to_simulation=False,
            volume_transport_audit_passed=(report['swept_geometry_identity_error_ml_s']<=1e-7
                and abs(report['projected_volume_balance_error_ml_s'])<=1e-5
                and report['transport_defect_identity_error_ml_s']<=1e-7 and relative<=1e-7),
            targets_scope='offline geometric/adjoint engineering checks; not step acceptance or wall sealing')
        return report
    finally:
        transfer.reset_warm_start()
