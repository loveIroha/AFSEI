"""Read saved MAC fields on the CPU; no interpolation into a new solver."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory
    cfg=json.loads((root/'configuration.json').read_text())['config']['fsi']['fluid']
    records=json.loads((root/'report.json').read_text())['records']
    row=records[-1]
    geometry=row['balanced']['tract_geometry'];axis=geometry['axis']
    shape=np.asarray(cfg['shape']);origin=np.asarray(cfg['origin'])
    spacing=np.asarray(cfg['lengths'])/shape
    coordinates=[origin[a]+(np.arange(shape[a])+.5)*spacing[a] for a in range(3)]
    other=[a for a in range(3) if a!=axis]
    mesh=np.meshgrid(*(coordinates[a] for a in other),indexing='ij')
    bottom=int(round((geometry['bottom_coordinate_cm']-origin[axis])/spacing[axis]))
    with np.load(root/'checkpoint.npz',allow_pickle=False) as archive:
        pressure=archive['pressure'];velocity=archive[f'velocity_{axis}']
        metadata=json.loads(str(archive['metadata']))
    time_ms=metadata['time']*1000
    if abs(time_ms-row['time_ms'])>1e-7:raise ValueError('checkpoint is not the report endpoint')
    C=1333.22368421;face_area=float(np.prod(spacing)/spacing[axis])
    profiles={}
    for label,bounds in zip(('mv','av'),geometry['tract_bounds_cm']):
        aperture=np.ones(mesh[0].shape,dtype=bool)
        for values,(low,high) in zip(mesh,bounds):aperture&=(values>=low)&(values<high)
        data=[]
        for k in range(bottom,int(shape[axis])):
            p=np.take(pressure,k,axis=axis)[aperture]/C
            q0=float(np.take(velocity,k,axis=axis)[aperture].sum()*face_area)
            q1=float(np.take(velocity,k+1,axis=axis)[aperture].sum()*face_area)
            data.append(dict(z_center_cm=float(coordinates[axis][k]),mean_mmhg=float(p.mean()),
                std_mmhg=float(p.std()),q_low_face_ml_s=q0,q_high_face_ml_s=q1))
        neck=.5*(np.take(pressure,bottom-1,axis=axis)[aperture].mean()
                  +np.take(pressure,bottom,axis=axis)[aperture].mean())/C
        profiles[label]=dict(area_cm2=float(aperture.sum()*face_area),
            fixed_collar_neck_pressure_mmhg=float(neck),rows=data)
    core=row['balanced']['pressure_core']['mean_mmhg']
    cut=row['balanced']['prescribed_cut_pressure_mmhg']
    neck=profiles['av']['fixed_collar_neck_pressure_mmhg']
    previous=records[-2]
    q=row['balanced']['physical_outlet_flow_ml_s']
    old_q=previous['balanced']['physical_outlet_flow_ml_s']
    acceleration=(q-old_q)/row['accepted_dt_s']
    length=geometry['cut_coordinate_cm']-geometry['bottom_coordinate_cm']
    inertance=cfg['rho']*length/profiles['av']['area_cm2']/C
    result=dict(time_ms=time_ms,axis=axis,profiles=profiles,
        diagnostic_partition=dict(core_mmhg=core,av_fixed_neck_mmhg=neck,prescribed_cut_mmhg=cut,
            core_minus_fixed_neck_mmhg=core-neck,fixed_neck_minus_cut_mmhg=neck-cut,
            core_minus_cut_mmhg=core-cut),
        uniform_tract_inertia_estimate=dict(length_cm=length,area_cm2=profiles['av']['area_cm2'],
            q_old_ml_s=old_q,q_new_ml_s=q,dt_s=row['accepted_dt_s'],q_rate_ml_s2=acceleration,
            inertance_mmhg_s2_per_ml=inertance,estimated_pressure_mmhg=inertance*acceleration,
            measured_neck_minus_cut_mmhg=neck-cut),
        interpretation='Neck pressure is a MAC cell-average interpolation at the fixed collar bottom, not an anatomical valve pressure. The rho*length/area*dQ/dt estimate is a uniform fixed-tract diagnostic; convection, viscosity, IB forcing and discretization terms are not separately audited.')
    (root/'cut_pressure_profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(time_ms=time_ms,diagnostic_partition=result['diagnostic_partition'],
        av_cut_flow_ml_s=profiles['av']['rows'][-1]['q_high_face_ml_s'],
        av_fixed_neck_flow_ml_s=profiles['av']['rows'][0]['q_low_face_ml_s'],
        uniform_tract_inertia_estimate=result['uniform_tract_inertia_estimate']),indent=2))


if __name__=='__main__':main()
