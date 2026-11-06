"""Read native IBAMR snapshots and compare the shared constant-drive FSI IVP.

Differences between methods are not continuum errors. A short pilot cannot
establish full-trajectory matched accuracy or an end-to-end EP/ODE speedup.
"""
from pathlib import Path
import argparse,json,struct
import numpy as np

def read_frame(path):
    with path.open('rb') as f:
        version,n,nn,step=struct.unpack('<4Q',f.read(32));time=struct.unpack('<d',f.read(8))[0]
        if version!=1:raise ValueError('unknown native snapshot version')
        x=np.fromfile(f,dtype='<f8',count=3*nn).reshape(nn,3)
        p=np.fromfile(f,dtype='<f8',count=n**3).reshape(n,n,n)
        u=np.fromfile(f,dtype='<f8',count=3*n**3).reshape(n,n,n,3)
        if f.read(1):raise ValueError('trailing native snapshot data')
    return dict(n=n,step=step,time_s=time,x=x,pressure=p,velocity=u)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    seed=np.load(root/'ibamr_inputs/constant_60ms_active.npz');X=seed['reference'];cells=seed['cells']
    D=(X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1);volume=abs(np.linalg.det(D))/6
    inv=np.linalg.inv(D);candidates={};reference=None
    aroot=root/'afsei_constant_active_pilot'
    if (aroot/'report.json').exists() and (aroot/'final_state.npz').exists():
        a=np.load(aroot/'final_state.npz');reference=dict(x=a['x'],pressure=a['pressure'],velocity=np.stack([
            .5*(np.take(a[k],np.arange(a[k].shape[j]-1),axis=j)+np.take(a[k],np.arange(1,a[k].shape[j]),axis=j)) for j,k in enumerate(('u','v','w'))],axis=-1))
        for key in ('previous_pressure','pressure_time_s','previous_pressure_time_s'):
            if key in a:reference[key]=a[key]
    for group in ('ibamr_pilots','ibamr_pressure_pilots','ibamr_release_traction_pilots_early','ibamr_release_pilots'):
        folder=root/group
        if not folder.exists():continue
        for case in folder.iterdir():
            if not case.is_dir():continue
            frames=sorted(case.glob('frame_*.bin'))
            if not frames:continue
            initial=read_frame(frames[0]);last=read_frame(frames[-1]);J=np.linalg.det((last['x'][cells[:,1:]]-last['x'][cells[:,:1]]).transpose(0,2,1))/np.linalg.det(D)
            row=dict(last_time_ms=1000*last['time_s'],saved_frames=len(frames),
                reference_geometry_max_difference_cm=float(abs(initial['x']-X).max()),
                max_displacement_mm=float(np.linalg.norm(last['x']-X,axis=1).max()*10),
                wall_volume_change_percent=float(100*(np.sum(volume*J)/np.sum(volume)-1)),
                local_J_RMS=float(np.sqrt(np.sum(volume*(J-1)**2)/volume.sum())),min_J=float(J.min()),max_J=float(J.max()),
                data_finite=all(np.isfinite(last[k]).all() for k in ('x','pressure','velocity')),
                release_dependencies=group in ('ibamr_release_pilots','ibamr_release_traction_pilots_early'),formal_performance_baseline=False)
            cp=case/'native_contract.json';contract=json.loads(cp.read_text()) if cp.exists() else {}
            row['boundary_coefficients_aligned']=str(contract.get('open_boundary','')).startswith('homogeneous velocity Neumann and pressure Dirichlet')
            if reference is not None and abs(last['time_s']-.002)<1e-10:
                u=last['x']-X;v=reference['x']-X
                def solid_norm(q):
                    z=q[cells];return np.sqrt(np.sum(volume/20*(np.sum(z.sum(1)**2,axis=1)+np.sum(z*z,axis=(1,2)))))
                row['displacement_relative_L2_difference']=float(solid_norm(u-v)/max(solid_norm(v),1e-30))
                for k in ('velocity','pressure'):
                    target=reference[k]
                    if k=='pressure':
                        # Native CN Stokes pressure is at n+1/2; AFSEI BE is at n+1.
                        tp=last['time_s']-.5*last['time_s']/last['step']
                        row['native_pressure_time_s']=tp;row['pressure_time_aligned']=False
                        if all(key in reference for key in ('previous_pressure','pressure_time_s','previous_pressure_time_s')):
                            t0=float(reference['previous_pressure_time_s']);t1=float(reference['pressure_time_s']);alpha=(tp-t0)/(t1-t0)
                            if -1e-10<=alpha<=1+1e-10:
                                target=(1-alpha)*reference['previous_pressure']+alpha*reference['pressure']
                                row['pressure_time_aligned']=True;row['AFSEI_pressure_interpolation_weight']=alpha
                    row[k+'_relative_L2_difference']=float(np.linalg.norm(last[k]-target)/max(np.linalg.norm(target),1e-30))
                row['pilot_field_agreement_passed']=row['boundary_coefficients_aligned'] and row.get('pressure_time_aligned',False) and all(row[k+'_relative_L2_difference']<=.05 for k in ('velocity','pressure')) and row['displacement_relative_L2_difference']<=.05
            rp=case/'report.json'
            if rp.exists():row['native_report']=json.loads(rp.read_text())
            candidates[group+'/'+case.name]=row
    report=dict(cases=candidates,AFSEI_pilot_available=reference is not None,
        formal_matched_accuracy_speedup_established=False,
        required_remaining_controls=['open-boundary response under independent native versus split discretizations',
            'spatial and temporal error control on the actual benchmark window',
            'repeated isolated timing after accuracy controls pass'],
        pressure_comparison='Native midpoint pressure compared with linear interpolation between neighboring AFSEI endpoint pressure states; interpolation is a postprocessing operation, not a solver change.',
        scope='2ms new constant EP-derived-force FSI initial value problem; no EP or ODE advancement in this pilot')
    (root/'native_pilot_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
