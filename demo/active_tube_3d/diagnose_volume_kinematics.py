"""Exact one-step determinant budget from saved accepted states (NumPy only)."""
import argparse,json
from pathlib import Path
import numpy as np


def analyze(path,spacing,dt):
    with np.load(path) as a:
        X=a['reference'];cells=a['cells'];x=a['x'];old=a['previous_x']
        velocity=[a[n] for n in ['u','v','w']];clock=float(a['time_s'])
    edge=lambda v:np.swapaxes(v[cells[:,1:]]-v[cells[:,:1]],1,2)
    D=edge(X);inverse=np.linalg.inv(D);weights=np.linalg.det(D)/6
    F0=edge(old)@inverse;F1=edge(x)@inverse
    G=(edge(x)-edge(old))@inverse/dt
    A=G@np.linalg.inv(F0);trace=np.trace(A,axis1=1,axis2=2)
    J0=np.linalg.det(F0);J1=np.linalg.det(F1)
    physical_weights=weights*J0
    exact=float((weights*(J1-J0)).sum())
    linear=float(dt*(physical_weights*trace).sum())
    # det(I+dt*A) is an exact degree-three polynomial. The difference from
    # the linear term is the geometric time-update contribution for this step.
    remainder=exact-linear
    check=float(np.max(np.abs(J0*np.linalg.det(np.eye(3)+dt*A)-J1)))
    div=sum(np.diff(v,axis=i)/spacing[i] for i,v in enumerate(velocity))
    return dict(checkpoint=str(path),time_ms=clock*1000,dt_s=dt,
        wall_volume_change_this_step_cm3=exact,
        first_order_FE_velocity_divergence_contribution_cm3=linear,
        geometric_higher_order_time_contribution_cm3=remainder,
        higher_order_fraction_of_net_step_change=abs(remainder/exact) if abs(exact)>1e-20 else None,
        determinant_identity_max_error=check,
        FE_velocity_divergence_volume_mean_s_inv=float((physical_weights*trace).sum()/physical_weights.sum()),
        FE_velocity_divergence_volume_rms_s_inv=float(np.sqrt((physical_weights*trace**2).sum()/physical_weights.sum())),
        FE_velocity_divergence_max_abs_s_inv=float(np.abs(trace).max()),
        MAC_divergence_max_abs_s_inv=float(np.abs(div).max()),
        scope='One accepted step. FE velocity inferred from accepted nodal displacement/dt. This does not integrate a full-history error or isolate every spatial transfer suboperator.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    config=json.loads((a.root/'active_120ms/config.resolved.json').read_text());spacing=np.array(config['fluid']['lengths_cm'])/config['fluid']['shape'];dt=config['time']['fsi_dt_s']
    result=[analyze(a.root/n/'final_state.npz',spacing,dt) for n in ['active_120ms','active_continued_800ms']]
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'volume_kinematics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
