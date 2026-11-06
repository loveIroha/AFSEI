"""Read-only geometric audit of the saved single-tube full experiment."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from afsi_torch.simulation.circular_tube_config import resolve_config
from afsi_torch.solids.circular_tube import generate_circular_tube


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();out=a.root/'analysis'
    c=json.loads((a.root/'active_120ms/config.resolved.json').read_text());g,_,_,_=resolve_config(c)
    m=generate_circular_tube(g);X=m.X.numpy();rings=m.inner_rings.numpy()
    radial=X[:,:2]-np.array(g.center_xy_cm);radial/=np.linalg.norm(radial,axis=1,keepdims=True)
    data=[];states={};times=[0.,.12,.21,.32,.5,.8]
    pvd=out/'solid.pvd'
    for entry in ET.parse(pvd).findall('.//DataSet'):
        t=float(entry.attrib['timestep']);x=meshio.read(pvd.parent/entry.attrib['file']).points
        if x.shape!=X.shape:raise ValueError('unexpected saved FE node ordering/shape')
        d=x-X;inward=-(d[:,:2]*radial).sum(1)
        q=x[rings,:2];opposite=np.linalg.norm(q-np.roll(q,len(rings[0])//2,axis=1),axis=2)
        q3=x[rings];opposite3=np.linalg.norm(q3-np.roll(q3,len(rings[0])//2,axis=1),axis=2)
        center=q3.mean(1)
        local_radius=np.linalg.norm(q3-center[:,None,:],axis=2)
        transverse=np.linalg.norm(center[:,:2]-np.array(g.center_xy_cm),axis=1)
        rows=dict(time_ms=t*1000,maximum_reference_inward_projection_mm=float(inward.max()*10),
            maximum_inner_radius_decrease_about_current_centroid_mm=float((g.inner_radius_cm-local_radius).max()*10),
            maximum_total_displacement_mm=float(np.linalg.norm(d,axis=1).max()*10),
            maximum_axial_displacement_mm=float(abs(d[:,2]).max()*10),
            maximum_cross_axis_displacement_mm=float(np.linalg.norm(d[:,:2],axis=1).max()*10),
            minimum_projected_opposed_wall_chord_mm=float(opposite.min()*10),
            minimum_3d_opposed_wall_chord_mm=float(opposite3.min()*10),
            maximum_transverse_centerline_shift_mm=float(transverse.max()*10),
            minimum_material_section_spacing_cm=float(np.diff(center[:,2]).min()))
        data.append(rows)
        for target in times:
            if abs(t-target)<1e-10:states[target]=x
    h=json.loads((out/'history.json').read_text());ref=h[0]
    result=dict(saved_frame_metrics=data,
        sampled_maximum_local_inner_radius_decrease=max(data,key=lambda r:r['maximum_inner_radius_decrease_about_current_centroid_mm']),
        sampled_maximum_total=max(data,key=lambda r:r['maximum_total_displacement_mm']),
        sampled_minimum_opposed_chord=min(data,key=lambda r:r['minimum_projected_opposed_wall_chord_mm']),
        sampled_maximum_centerline_shift=max(data,key=lambda r:r['maximum_transverse_centerline_shift_mm']),
        final=data[-1],
        maximum_wall_volume_loss_percent=100*(1-min(r['wall_volume_cm3'] for r in h)/ref['wall_volume_cm3']),
        final_wall_volume_change_percent=100*(h[-1]['wall_volume_cm3']/ref['wall_volume_cm3']-1),
        min_sampled_detF=min(r['minimum_detF'] for r in h),max_sampled_detF=max(r['maximum_detF'] for r in h),
        final_max_fluid_velocity_cm_s=h[-1]['max_velocity_cm_s'],
        max_sampled_end_displacement_mm=max(r['maximum_end_displacement_cm'] for r in h)*10,
        max_sampled_nonlinear_residual_ratio=max(r['nonlinear_residual']/max(r['nonlinear_tolerance'],1e-30) for r in h),
        diagnostic_scope='Saved frames every 10 ms; opposed reference-angle chord is not a closest-surface contact gap. Wall volume may change in the chosen compressible solid model.')
    (out/'geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True)
    # Reference material meridians, not a silhouette or an axisymmetry assumption.
    for ax,t in zip(axs.flat,times):
        x=states[t];points=x.reshape(g.axial_elements+1,g.thickness_elements+1,g.circumferential_elements,3)
        for radial_id in [0,g.thickness_elements]:
            for angle in [0,g.circumferential_elements//2]:
                v=points[:,radial_id,angle];ax.plot((v[:,2]-g.axial_start_cm)*10,(v[:,0]-g.center_xy_cm[0])*10,
                    color='tab:blue' if radial_id==0 else 'tab:orange',lw=1.5)
        ax.set(title=f'{t*1000:.0f} ms',xlabel='Axial coordinate z (mm)',ylabel='x - reference center (mm)',xlim=(-1,21),ylim=(-6.5,6.5))
        ax.grid(alpha=.25)
    fig.suptitle('Saved material meridians: inner blue, outer orange; actual deformed coordinates')
    fig.tight_layout();fig.savefig(out/'tube_meridians.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='saved_frame_metrics'},indent=2))


if __name__=='__main__':main()
