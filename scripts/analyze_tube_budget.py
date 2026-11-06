"""Analyze saved tube budgets; optionally inspect one saved 3D accepted step."""
import argparse
import json
from pathlib import Path
import numpy as np


def saved_3d_step(folder):
    import torch
    from afsi_torch.simulation.circular_tube_config import resolve_config
    from afsi_torch.solids.circular_tube import generate_circular_tube
    from afsi_torch.mac.grid import MACGrid
    from afsi_torch.mac.sealed_ports import SealedDomain
    from afsi_torch.simulation.tube_budget import TubeBudget
    c=json.loads((folder/'config.resolved.json').read_text())
    g,_,_,_=resolve_config(c)
    mesh=generate_circular_tube(g)
    grid=MACGrid(tuple(c['fluid']['shape']),tuple(c['fluid']['lengths_cm']),tuple(c['fluid']['origin_cm']))
    domain=SealedDomain(grid,g,c['hydraulic_domain'],mesh.X,dimension=3)
    with np.load(folder/'final_state.npz') as a:
        get=lambda key:torch.from_numpy(a[key].copy())
        old,new=get('previous_x'),get('x')
        velocity=tuple(get(k) for k in ('u','v','w'))
        if not np.array_equal(a['reference'],mesh.X.numpy()):
            raise ValueError('saved geometry does not match regenerated mesh')
        b=TubeBudget(mesh,domain,initial_x=old)
        b.accepted(old,new,velocity,c['time']['fsi_dt_s'])
        r=b.diagnostics(velocity)
        r.update(scope='only final accepted step, not full-run cumulative budget',
            window_dt_s=c['time']['fsi_dt_s'],source=str(folder/'final_state.npz'))
        return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-2d',type=Path,required=True)
    p.add_argument('--reference-2d',type=Path)
    p.add_argument('--saved-3d',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    h=json.loads((a.case_2d/'history.json').read_text())
    get=lambda name:np.array([r[name] for r in h])
    t=get('time_s');moving=get('budget_moving_lumen_volume_ml');fixed=get('budget_fixed_end_volume_ml')
    net=get('budget_integrated_net_inflow_ml');cap=get('budget_moving_cap_sweep_ml')
    error=get('budget_fixed_control_imbalance_ml')
    q=np.array([r.get('inlet_aperture_ml_s',0)-r.get('outlet_aperture_ml_s',0) for r in h])
    sparse=np.r_[0,np.cumsum(np.diff(t)*(q[1:]+q[:-1])/2)]
    summary=dict(scope='accepted-step budgets; virtual connector boundaries are not material walls',
        case_2d=str(a.case_2d),accepted_steps=h[-1]['budget_accepted_steps'],
        moving_volume_change_ml=float(moving[-1]-moving[0]),
        fixed_volume_change_ml=float(fixed[-1]-fixed[0]),
        accepted_step_net_inflow_ml=float(net[-1]),
        sparse_trapezoid_net_inflow_ml=float(sparse[-1]),
        moving_cap_sweep_ml=float(cap[-1]),
        fixed_control_imbalance_ml=float(error[-1]),
        fixed_control_imbalance_fraction_of_volume_change=float(abs(error[-1]/(fixed[-1]-fixed[0]))),
        maximum_abs_fixed_control_imbalance_ml=float(np.max(np.abs(error))),
        maximum_imbalance_fraction_of_final_volume_change=float(np.max(np.abs(error))/abs(fixed[-1]-fixed[0])),
        maximum_geometric_identity_error_ml=max(abs(r['budget_geometric_identity_error_ml']) for r in h),
        maximum_fixed_geometric_identity_error_ml=max(abs(r['budget_fixed_geometric_identity_error_ml']) for r in h),
        outlet_budget_minus_rc_integral_ml=h[-1]['budget_integrated_outlet_ml']-h[-1]['rc_integrated_outlet_ml'],
        simulation_report=json.loads((a.case_2d/'report.json').read_text()))
    if a.reference_2d:
        with np.load(a.case_2d/'snapshots.npz') as x,np.load(a.reference_2d/'snapshots.npz') as y:
            if not np.array_equal(x['times'],y['times']):raise ValueError('comparison times differ')
            summary['unchanged_solution_max_abs_difference']={k:float(np.max(np.abs(x[k]-y[k])))
                for k in ('x','u','v','p','voltage')}
    if a.saved_3d:summary['saved_3d_last_step']=saved_3d_step(a.saved_3d)
    (a.output/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    ax[0].plot(t*1000,moving[0]-moving,label='Moving-end lumen volume decrease')
    ax[0].plot(t*1000,fixed[0]-fixed,label='Fixed-end control volume decrease')
    ax[0].plot(t*1000,-net,'--',label='Accepted-step net expelled volume')
    ax[0].set(xlabel='Time (ms)',ylabel='Volume (mL)',title='Compare the same control surfaces')
    ax[1].plot(t*1000,(moving-moving[0]-sparse)*1000,label='Moving volume vs sparse flux')
    ax[1].plot(t*1000,(moving-moving[0]-net)*1000,label='Moving volume vs stepwise flux')
    ax[1].plot(t*1000,error*1000,label='Fixed-end volume vs stepwise flux')
    ax[1].set(xlabel='Time (ms)',ylabel='Signed difference (µL)',title='End motion and sampling explain most of the gap')
    for axis in ax:
        axis.axhline(0,color='gray',lw=.6);axis.grid(alpha=.2);axis.legend(fontsize=8)
    for ext in ('png','svg'):fig.savefig(a.output/f'volume_flux_budget.{ext}',dpi=180)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('simulation_report','saved_3d_last_step')},indent=2))


if __name__=='__main__':main()
