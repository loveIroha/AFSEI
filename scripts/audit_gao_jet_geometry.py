"""CPU-only aperture and velocity-profile audit of the same saved endpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from afsi_torch.ep.mac_ports import MACBasalPorts
from afsi_torch.circulation.geometry import PortGeometry
from afsi_torch.mac.grid import MACGrid
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path);parser.add_argument('audit_directory',type=Path)
    a=parser.parse_args();torch.set_num_threads(4)
    frozen=json.loads((a.audit_directory/'audit.json').read_text())
    last=json.loads((a.checkpoint.parent/'report.json').read_text())['records'][-1]
    with np.load(a.checkpoint,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));X=torch.from_numpy(z['X']);old=torch.from_numpy(z['cap_geometry'])
        faces=torch.from_numpy(z['faces']);tags=torch.from_numpy(z['facet_tags'])
        pressure=z['pressure'];velocity=z['velocity_2']
    cfg=meta['config']['fsi'];g=cfg['fluid'];grid=MACGrid(tuple(g['shape']),tuple(g['lengths']),tuple(g['origin']))
    geometry=last['balanced']['tract_geometry'];axis=geometry['axis']
    if axis!=2:raise ValueError('this case has a z-aligned basal tract')
    port_config=meta['balanced_execution']['port_geometry']
    ports=PortGeometry(**port_config) if port_config else PortGeometry()
    # ImportedSolidMesh stores the source->model tag map: RealLVSolid uses
    # surface(1) for endocardium; cfg.endo_tag is the PRE-import source label.
    cap=MACBasalPorts(X,faces[tags==1],grid,ports=ports,integration_rule='centroid')
    points,area=cap.surface(old)
    av=cap.labels==2
    projected=float(area[av,axis].sum());material=float(area[av].norm(dim=1).sum())
    bounds=geometry['tract_bounds_cm'][1];h=grid.spacing[axis]
    xx,yy=np.meshgrid(*(grid.origin[j]+(np.arange(grid.shape[j])+.5)*grid.spacing[j] for j in (0,1)),indexing='ij')
    aperture=(xx>=bounds[0][0])&(xx<bounds[0][1])&(yy>=bounds[1][0])&(yy<bounds[1][1])
    count=aperture.sum();rect_area=count*grid.volume/h
    selected=torch.ones(len(points),dtype=torch.bool)
    for j,(low,high) in enumerate(bounds):selected&=(points[:,j]>=low)&(points[:,j]<high)
    closed_in_rectangle=float(area[selected&(cap.labels==0),axis].sum())
    bottom=int(round((geometry['bottom_coordinate_cm']-grid.origin[axis])/h))
    split=int(round((frozen['geometry']['diagnostic_split_cm']-grid.origin[axis])/h))
    profile=[]
    for k in range(bottom,grid.shape[axis]+1):
        u=velocity[:,:,k][aperture]
        if k==grid.shape[axis]:p=np.full(len(u),last['balanced']['prescribed_cut_pressure_mmhg'])
        else:p=.5*(pressure[:,:,k-1][aperture]+pressure[:,:,k][aperture])/C
        profile.append(dict(z_cm=grid.origin[axis]+k*h,mean_cm_s=float(u.mean()),
            mean_square_cm2_s2=float(np.mean(u*u)),maximum_cm_s=float(u.max()),minimum_cm_s=float(u.min()),
            negative_area_fraction=float(np.mean(u<0)),q_ml_s=float(u.sum()*grid.volume/h),
            pressure_mean_mmhg=float(p.mean()),pressure_std_mmhg=float(p.std())))
    convective={}
    for label,start,stop in (('basal_support',bottom,split),('remote_tract',split,grid.shape[axis])):
        estimate=g['rho']*(profile[stop-bottom]['mean_square_cm2_s2']-profile[start-bottom]['mean_square_cm2_s2'])/C
        convective[label]=dict(axial_momentum_flux_estimate_mmhg=estimate,
            original_semilagrangian_transport_plus_residual_mmhg=frozen['segments'][label]['inferred_transport_plus_predictor_residual_mmhg'])
    result=dict(time_ms=meta['time']*1000,physical_time_advanced=False,pressure_reference_unchanged=True,
        ports=port_config,reference_nominal_areas_cm2=cap.metadata['nominal_reference_port_area_cm2'],
        av_material_area_cm2=material,av_projected_area_cm2=projected,fixed_tract_area_cm2=rect_area,
        av_to_tract_projected_area_ratio=projected/rect_area,
        centroid_estimated_closed_cap_area_inside_tract_cm2=closed_in_rectangle,
        q_actual_ml_s=last['balanced']['physical_outlet_flow_ml_s'],
        aperture_mean_velocity_from_q_cm_s=last['balanced']['physical_outlet_flow_ml_s']/projected,
        maximum_profile_velocity_cm_s=max(r['maximum_cm_s'] for r in profile),
        momentum_flux_comparison=convective,profile=profile,
        caveats='Area uses the same reference-labelled material subtriangles and old frozen geometry. Closed-area selection uses centroids. Momentum-flux estimate is not identical to the original semi-Lagrangian transport discretization; both are reported without substitution.')
    (a.audit_directory/'jet_geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    z=[p['z_cm'] for p in profile]
    fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
    for key,label in (('mean_cm_s','Cross-section mean'),('maximum_cm_s','Maximum'),('minimum_cm_s','Minimum')):
        ax[0].plot(z,[p[key] for p in profile],label=label)
    ax[0].set(ylabel='Axial velocity [cm/s]',title='Fixed AV-tract velocity profile');ax[0].legend()
    ax[1].plot(z,[p['pressure_mean_mmhg'] for p in profile],label='Cross-section mean pressure')
    ax[1].set(xlabel='Axial coordinate [cm]',ylabel='Pressure [mmHg]');ax[1].legend()
    for x in ax:
        for key in ('basal_cap_min_cm','basal_cap_max_cm','diagnostic_split_cm'):
            x.axvline(frozen['geometry'][key],ls=':',color='gray')
        x.grid(alpha=.2)
    fig.suptitle(f'Frozen aperture {projected:.3f} cm2 / tract {rect_area:.3f} cm2; no physical advance')
    fig.savefig(a.audit_directory/'jet_geometry.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ('profile',)},indent=2))


if __name__=='__main__':main()
