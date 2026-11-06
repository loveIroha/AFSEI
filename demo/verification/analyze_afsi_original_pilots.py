"""Physical-point field comparison, original Q2/Q1 AFSI versus MAC AFSEI.

Intermethod differences are not continuum errors or an accuracy certificate.
The declared box translation is undone; no fitted alignment or pressure shift.
"""
from pathlib import Path
import argparse,json
import numpy as np

def q2_sample(values,points,length=2.8):
    n=(values.shape[0]-1)//2
    s=points/(length/n);base=np.clip(np.floor(s).astype(int),0,n-1);t=s-base
    w=np.stack(((2*t-1)*(t-1),4*t*(1-t),t*(2*t-1)),axis=-1)
    result=np.zeros((len(points),)+values.shape[3:],dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                v=values[2*base[:,0]+i,2*base[:,1]+j,2*base[:,2]+k]
                weight=w[:,0,i]*w[:,1,j]*w[:,2,k]
                result+=v*weight.reshape((-1,)+(1,)*(v.ndim-1))
    return result

def q1_sample(values,points,length=2.8):
    n=values.shape[0]-1;s=points/(length/n);base=np.clip(np.floor(s).astype(int),0,n-1);t=s-base
    result=np.zeros(len(points))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                w=(t[:,0] if i else 1-t[:,0])*(t[:,1] if j else 1-t[:,1])*(t[:,2] if k else 1-t[:,2])
                result+=values[base[:,0]+i,base[:,1]+j,base[:,2]+k]*w
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;folder=root/'afsi_original';reference=np.load(root/'afsei_constant_active_pilot/final_state.npz')
    X=reference['reference'];cells=reference['cells'];D=(X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1)
    volumes=abs(np.linalg.det(D))/6
    def solid_norm(q):
        z=q[cells];return np.sqrt(np.sum(volumes/20*(np.sum(z.sum(1)**2,axis=1)+np.sum(z*z,axis=(1,2)))))
    n=reference['pressure'].shape[0]
    points=np.stack(np.meshgrid(*[((np.arange(n)+.5)*2.8/n)]*3,indexing='ij'),axis=-1).reshape(-1,3)
    target_u=np.stack([.5*(np.take(reference[k],np.arange(reference[k].shape[j]-1),axis=j)+
        np.take(reference[k],np.arange(1,reference[k].shape[j]),axis=j)) for j,k in enumerate(('u','v','w'))],axis=-1).reshape(-1,3)
    target_p=reference['pressure'].ravel();rows={}
    for path in sorted(folder.glob('n*_dt*_mpi*')):
        if not path.is_dir() or not (path/'report.json').exists():continue
        report=json.loads((path/'report.json').read_text());row=dict(native_report=report,
            formal_matched_accuracy_speedup_established=False)
        rows[path.name]=row
        if report.get('status')!='completed':continue
        frames=json.loads((path/'frames.json').read_text());last=frames[-1]
        if abs(last['time_s']-.002)>1e-12:continue
        state=np.load(path/last['file']);u=state['x']-X;target=reference['x']-X
        row['displacement_relative_L2_difference']=float(solid_norm(u-target)/max(solid_norm(target),1e-30))
        row['velocity_relative_L2_difference']=float(np.linalg.norm(q2_sample(state['velocity_q2'],points)-target_u)/max(np.linalg.norm(target_u),1e-30))
        row['pressure_relative_L2_difference']=float(np.linalg.norm(q1_sample(state['pressure_q1'],points)-target_p)/max(np.linalg.norm(target_p),1e-30))
        J=np.linalg.det((state['x'][cells[:,1:]]-state['x'][cells[:,:1]]).transpose(0,2,1))/np.linalg.det(D)
        row.update(min_J=float(J.min()),max_J=float(J.max()),
            max_displacement_mm=float(np.linalg.norm(u,axis=1).max()*10),
            wall_volume_change_percent=float(100*(np.sum(volumes*J)/volumes.sum()-1)),
            endpoint_pressure_comparison=True,pilot_field_agreement_passed=all(row[k+'_relative_L2_difference']<=.05 for k in ('displacement','velocity','pressure')))
    result=dict(cases=rows,formal_matched_accuracy_speedup_established=False,
        scope='2ms fixed EP-derived load FSI IVP; zero initial fluid/reference wall; no EP/ODE advancement',
        interpolation='Exact tensor Q2 velocity and Q1 pressure evaluated at common MAC cell centers, with a fixed declared box translation.',
        interpretation='Intermethod differences against a finite-grid AFSEI pilot, not continuum errors. Full-window spatial/time controls and isolated repeated CPU/MPI/GPU timing are still required.')
    (folder/'pilot_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
