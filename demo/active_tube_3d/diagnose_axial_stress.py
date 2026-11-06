"""Saved-state stress and trial-mode material-curvature diagnostic, no stepping."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import meshio
from afsi_torch.solids.circular_tube import generate_circular_tube
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.p1 import prepare_p1
from afsi_torch.materials import neo_hookean_pk1
from afsi_torch.ep.land import active_pk1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--case',type=Path);p.add_argument('--times',default='0,120,180,200,220,250,280,300,320,350,400');a=p.parse_args()
    case=a.case or a.root/'active_120ms'
    c=json.loads((case/'config.resolved.json').read_text());g,_,_,_=resolve_config(c)
    m=generate_circular_tube(g,device='cuda');X=m.X;device=X.device;geo=prepare_p1(X,m.cells,degree=5)
    grad=geo.gradients[:,0];vol=geo.weights.sum(1)
    E=c['solid']['young_modulus_dyn_cm2'];nu=c['solid']['poisson_ratio'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
    src=case/'vtk/solid.pvd' if a.case else a.root/'analysis/solid.pvd';entries=ET.parse(src).findall('.//DataSet')
    saved={round(float(e.attrib['timestep'])*1000):src.parent/e.attrib['file'] for e in entries}
    results=[]
    for ms in map(int,a.times.split(',')):
        x=torch.as_tensor(meshio.read(saved[ms]).points,device=device,dtype=X.dtype)
        ep=meshio.read(a.root/'ep_waveform_800ms/vtk'/f'ep_{ms*20:06d}.vtu')
        ta=torch.as_tensor(ep.cell_data['applied_tension_dyn_cm2'][0],device=device,dtype=X.dtype)
        F=torch.einsum('eai,eaJ->eiJ',x[m.cells],grad)
        def stress(v):return neo_hookean_pk1(v,mu,lam)+active_pk1(v,m.fiber,ta)
        P=stress(F);J=torch.linalg.det(F);sigma=P@F.transpose(-1,-2)/J[:,None,None]
        zref=X[m.cells,2].mean(1)-g.axial_start_cm
        central=(zref>=.2*g.length_cm)&(zref<=.8*g.length_cm)
        rows=dict(time_ms=ms,central_mean_axial_cauchy_stress_kpa=float((sigma[central,2,2]*vol[central]).sum()/vol[central].sum()/1e4),
            central_axially_compressed_reference_volume_fraction=float(((sigma[central,2,2]<0)*vol[central]).sum()/vol[central].sum()),
            central_min_axial_cauchy_stress_kpa=float(sigma[central,2,2].min()/1e4),
            central_max_axial_cauchy_stress_kpa=float(sigma[central,2,2].max()/1e4),material_trial_mode_curvatures=[])
        # C1 compact transverse bending perturbations, supported in the middle
        # 60% of the tube. End-support energy is zero for these modes.
        s=(X[:,2]-g.axial_start_cm)/g.length_cm
        u=((s-.2)/.6).clamp(0,1);inside=(s>.2)&(s<.8)
        xring=x.reshape(g.axial_elements+1,g.thickness_elements+1,g.circumferential_elements,3).mean((1,2))
        zprime=torch.gradient(xring[:,2],spacing=(g.length_cm/g.axial_elements,))[0]
        ids=torch.arange(len(X),device=device)//((g.thickness_elements+1)*g.circumferential_elements)
        local_center=xring[ids]
        for axis in [0,1]:
            for n in [1,2]:
                f=torch.sin(n*torch.pi*u).square()*inside
                df=n*torch.pi*torch.sin(2*n*torch.pi*u)/(.6*g.length_cm)*inside
                q=torch.zeros_like(X);q[:,axis]=f
                q[:,2]=-df*(x[:,axis]-local_center[:,axis])/zprime[ids].clamp_min(.1)
                dF=torch.einsum('eai,eaJ->eiJ',q[m.cells],grad)
                _,dP=torch.func.jvp(stress,(F,),(dF,))
                curvature=float((vol[:,None,None]*dF*dP).sum())
                qnorm=float((vol*(q[m.cells].square().sum(-1).mean(-1))).sum())
                rows['material_trial_mode_curvatures'].append(dict(axis=axis,waves=n,energy_second_derivative=curvature,
                    normalized_curvature=curvature/qnorm))
        # Integrate the existing full-vector end spring reaction exactly.
        ref=X[m.end_faces];uend=x[m.end_faces]-ref
        area=.5*torch.linalg.cross(ref[:,1]-ref[:,0],ref[:,2]-ref[:,0]).norm(dim=-1)
        reaction=-c['solid']['end_spring_dyn_cm3']*area[:,None,None]/12*(uend+uend.sum(1,keepdim=True))
        side=ref[:,:,2].mean(1)<g.axial_start_cm+g.length_cm/2
        rows['lower_end_spring_reaction_dyn']=reaction[side].sum((0,1)).cpu().tolist()
        rows['upper_end_spring_reaction_dyn']=reaction[~side].sum((0,1)).cpu().tolist()
        results.append(rows);print(json.dumps(rows),flush=True)
    a.output.mkdir(exist_ok=True,parents=True)
    (a.output/'axial_stress.json').write_text(json.dumps(dict(results=results,
        scope='Frozen active amplitude and saved geometry; material tangent only. No fluid response, no dynamic eigenvalue or full-FSI stability claim.'),indent=2)+'\n')


if __name__=='__main__':main()
