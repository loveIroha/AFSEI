"""Snapshot existing tube outputs: local J, bending and 3D plane sections.

No simulation, constitutive change or GPU use. Measures intersections of the
actual inner-wall triangles with centerline-normal planes. Curves explicitly
end at the last saved frame; an unfinished run is never extrapolated.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,xml.etree.ElementTree as ET
import numpy as np,meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/large2/gjh/AFSEI/results')
RUN=ROOT/'volumetric_stabilization_20261025/from_rest_300ms'
OUT=ROOT/'volumetric_stabilization_20261025/results_analysis'


def area2(p):
    return abs(.5*np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))


def section(points,faces,center,normal):
    """Closed intersection component enclosing the local centerline point."""
    normal=normal/np.linalg.norm(normal);origin=center+1e-8*normal
    tri=points[faces];dist=(tri-origin)@normal
    edge=np.array([[0,1],[1,2],[2,0]])
    crosses=(dist[:,edge[:,0]]<0)!=(dist[:,edge[:,1]]<0)
    active=crosses.sum(1)==2
    if not active.any():return dict(valid=False,reason='no-crossing')
    selected=np.nonzero(active)[0];r,e=np.nonzero(crosses[active]);edges=edge[e]
    di=dist[selected[r],edges[:,0]];dj=dist[selected[r],edges[:,1]]
    f=di/(di-dj)
    p=tri[selected[r],edges[:,0]]+f[:,None]*(tri[selected[r],edges[:,1]]-tri[selected[r],edges[:,0]])
    # Merge only numerical duplicates on shared edges, with 1e-8 cm tolerance.
    keys,first,inv=np.unique(np.round(p/1e-8).astype(np.int64),axis=0,return_index=True,return_inverse=True)
    xyz=p[first];segments=np.unique(np.sort(inv.reshape(-1,2),axis=1),axis=0)
    segments=segments[segments[:,0]!=segments[:,1]]
    adj={}
    for a,b in segments:
        adj.setdefault(int(a),set()).add(int(b));adj.setdefault(int(b),set()).add(int(a))
    axis=np.eye(3)[np.argmin(abs(normal))];u=np.cross(normal,axis);u/=np.linalg.norm(u);v=np.cross(normal,u)
    xy=np.column_stack(((xyz-origin)@u,(xyz-origin)@v));seen=set();loops=[];invalid_components=0
    for start in adj:
        if start in seen:continue
        stack=[start];component=set()
        while stack:
            i=stack.pop()
            if i in component:continue
            component.add(i);stack.extend(adj[i]-component)
        seen.update(component)
        if any(len(adj[i])!=2 for i in component):invalid_components+=1;continue
        chain=[start];previous=None;current=start
        for _ in range(len(component)+1):
            nxt=next(j for j in sorted(adj[current]) if j!=previous)
            if nxt==start:break
            chain.append(nxt);previous,current=current,nxt
        if len(chain)!=len(component):invalid_components+=1;continue
        poly=xy[chain];a=poly;b=np.roll(poly,-1,axis=0)
        y0=a[:,1];y1=b[:,1];den=np.where(abs(y1-y0)>1e-30,y1-y0,1.)
        intercept=a[:,0]-y0*(b[:,0]-a[:,0])/den
        inside=bool(np.count_nonzero(((y0>0)!=(y1>0))&(intercept>0))%2)
        loops.append(dict(area_cm2=float(area2(poly)),contains_center=inside,
            centroid_distance_cm=float(np.linalg.norm(poly.mean(0))),vertices=len(poly)))
    candidates=[a for a in loops if a['contains_center']]
    if not candidates:return dict(valid=False,reason='no-closed-loop-enclosing-center',loops=loops,invalid_components=invalid_components)
    chosen=min(candidates,key=lambda a:a['centroid_distance_cm'])
    return dict(valid=True,area_cm2=chosen['area_cm2'],diameter_mm=float(20*np.sqrt(chosen['area_cm2']/np.pi)),
                closed_loops=len(loops),enclosing_loops=len(candidates),invalid_components=invalid_components)


def geometry_data(folder):
    a=np.load(folder/'prefix_60ms/final_state.npz');X=a['reference'].copy();cells=a['cells'].copy()
    D=(X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1)
    return dict(X=X,cells=cells,vol=np.abs(np.linalg.det(D))/6,inner_faces=a['inner_faces'].copy(),rings=a['inner_rings'].copy())


def load(folder):
    history={};frames={}
    for phase in ('prefix_60ms','continued_300ms'):
        p=folder/phase
        if not p.exists():continue
        for h in json.loads((p/'history.json').read_text()):history[round(h['time_s'],9)]=h
        for e in ET.parse(p/'vtk/solid.pvd').findall('.//DataSet'):
            frames[round(float(e.attrib['timestep']),9)]=p/'vtk'/e.attrib['file']
    return history,frames


def measure(points,J,geo,factor):
    vol=geo['vol'];rings=geo['rings'];center=points[rings].mean(1)
    chord=center[-1]-center[0];chord/=np.linalg.norm(chord)
    delta=center-center[0];perp=delta-(delta@chord)[:,None]*chord
    indices=np.arange(6,35)*factor  # Same material positions 0.15...0.85 on both spaces.
    normal=[];xy=[];details=[]
    for i in indices:
        cut=section(points,geo['inner_faces'],center[i],center[i+factor]-center[i-factor])
        details.append(dict(material_fraction=float(i/(len(rings)-1)),**cut))
        if cut['valid']:normal.append(cut['diameter_mm'])
        polygon=points[rings[i],:2];xy.append(20*np.sqrt(area2(polygon-polygon.mean(0))/np.pi))
    jmean=float(np.sum(vol*(J-1))/vol.sum());jrms=float(np.sqrt(np.sum(vol*(J-1)**2)/vol.sum()))
    # RMS^2 = signed_mean^2 + variance; large variance can hide in a small mean.
    return dict(wall_volume_change_percent=100*jmean,J_rms_error=jrms,J_std=float(np.sqrt(max(0,jrms*jrms-jmean*jmean))),
        reference_fraction_abs_Jerror_gt10pct=float(vol[abs(J-1)>.1].sum()/vol.sum()),
        reference_fraction_abs_Jerror_gt20pct=float(vol[abs(J-1)>.2].sum()/vol.sum()),
        minimum_J=float(J.min()),maximum_J=float(J.max()),
        max_centerline_transverse_mm=float(np.linalg.norm(center[:,:2]-.7,axis=1).max()*10),
        max_bending_from_end_line_mm=float(np.linalg.norm(perp,axis=1).max()*10),
        sampled_min_normal_diameter_mm=float(min(normal)) if len(normal)==len(indices) else None,
        sampled_min_xy_diameter_mm=float(min(xy)),valid_sections=len(normal),expected_sections=len(indices),sections=details)


def validate(geo):
    X=geo['X'];rings=geo['rings'];i=len(rings)//2;center=X[rings].mean(1)
    expected=area2(X[rings[i],:2]-center[i,:2])
    base=section(X,geo['inner_faces'],center[i],center[i+1]-center[i-1]);assert base['valid']
    angle=np.pi/3;R=np.array([[1.,0.,0.],[0.,np.cos(angle),-np.sin(angle)],[0.,np.sin(angle),np.cos(angle)]])
    rotated=X@R.T+np.array([.4,-.2,.3]);rc=rotated[rings].mean(1)
    rigid=section(rotated,geo['inner_faces'],rc[i],rc[i+1]-rc[i-1]);assert rigid['valid']
    deformed=X.copy();deformed[:,:2]=.7+.6*(deformed[:,:2]-.7);deformed[:,2]*=1.2;dc=deformed[rings].mean(1)
    scaled=section(deformed,geo['inner_faces'],dc[i],dc[i+1]-dc[i-1]);assert scaled['valid']
    errors=dict(straight_relative=abs(base['area_cm2']/expected-1),rigid_rotation_relative=abs(rigid['area_cm2']/expected-1),
                radial_scaling_relative=abs(scaled['area_cm2']/(expected*.6**2)-1))
    assert max(errors.values())<1e-9
    return dict(passed=True,errors=errors,rotation_degrees=60,radial_scale=.6,axial_scale=1.2)


def main():
    OUT.mkdir(exist_ok=True)
    folders={'baseline_solid1':ROOT/'domain_space_campaign_20261025/coarse_box2',
             'stabilized_solid1':RUN/'solid1','baseline_solid2':ROOT/'domain_space_campaign_20261025/fine_box2',
             'stabilized_solid2':RUN/'solid2'}
    result={};curves={};histories={};checks={};run_reports={}
    for label,folder in folders.items():
        factor=int(label[-1]);geo=geometry_data(folder);checks[label]=validate(geo)
        hs,frames=load(folder);histories[label]=[hs[t] for t in sorted(hs)]
        rows=[]
        for t,path in sorted(frames.items()):
            m=meshio.read(path);J=m.cell_data_dict['J_min']['tetra']
            row=dict(time_ms=t*1000,**measure(m.points,J,geo,factor));rows.append(row)
        curves[label]=rows
        state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {'stage':'completed'}
        reports=[]
        for phase in ('prefix_60ms','continued_300ms'):
            p=folder/phase/'report.json'
            if p.exists():reports.append(json.loads(p.read_text()))
        run_reports[label]=reports
        result[label]=dict(status=state['stage'],last_history_ms=max(hs)*1000,last_field_ms=rows[-1]['time_ms'],
            final_saved_geometry={k:v for k,v in rows[-1].items() if k!='sections'},
            max_sampled_absolute_volume_error_pct=max(abs(r['wall_volume_change_percent']) for r in rows),
            max_sampled_bending_mm=max(r['max_bending_from_end_line_mm'] for r in rows),
            max_sampled_bending_time_ms=max(rows,key=lambda r:r['max_bending_from_end_line_mm'])['time_ms'],
            all_plane_sections_valid=all(r['valid_sections']==r['expected_sections'] for r in rows))
        if state['stage']=='completed' and len(reports)==2:
            seconds=sum(r['elapsed_seconds'] for r in reports)
            result[label]['observed_main_seconds_including_output']=seconds
            result[label]['observed_seconds_per_step']=seconds/sum(r['accepted_steps_this_run'] for r in reports)
        print(label,json.dumps(result[label]),flush=True)
    common_end=min(v['last_field_ms'] for v in result.values())
    common={}
    for label,rows in curves.items():
        r=max((r for r in rows if r['time_ms']<=common_end+1e-7),key=lambda r:r['time_ms'])
        common[label]={k:v for k,v in r.items() if k!='sections'}
    report=dict(snapshot_time_utc=datetime.now(timezone.utc).isoformat(),results=result,common_end_ms=common_end,common=common,
        section_checks=checks,section_definition='29 planes normal to centered centerline secants, material positions .15 to .85; actual P1 inner-wall intersections; choose closed component enclosing center; plane shift 1e-8 cm',
        limitations=['Plane normal depends on measured centerline; not a unique anatomical section definition.',
                     'Sampled central 70% minimum differs from global minimum over all old x-y rings.',
                     'Unfinished fine trajectory is not extrapolated; volume and J are FE discrete metrics, not true-solution error estimates.',
                     'Existing GPU timing includes output; GPU0 shared occupancy prevents formal speedup claims.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2));(OUT/'frame_metrics.json').write_text(json.dumps(curves,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    specs=[('wall_volume_change_percent','Signed wall-volume change (%)'),('J_rms_error','RMS local J - 1'),
           ('reference_fraction_abs_Jerror_gt10pct','Reference wall fraction |J-1| > 10%'),
           ('max_bending_from_end_line_mm','Bending from current end line (mm)'),
           ('sampled_min_normal_diameter_mm','Central 70% minimum normal-section diameter (mm)'),
           ('sampled_min_xy_diameter_mm','Central 70% minimum x-y projected diameter (mm)')]
    for label,rows in curves.items():
        color='#176d9c' if label.endswith('1') else '#be602b';ls='--' if label.startswith('baseline') else '-'
        title=label.replace('_',' ')
        for ax,(key,ylabel) in zip(axes.flat,specs):
            yy=[float('nan') if r[key] is None else r[key] for r in rows]
            ax.plot([r['time_ms'] for r in rows],yy,ls=ls,color=color,lw=1.6,label=title)
            ax.set(xlabel='Time from rest (ms)',ylabel=ylabel);ax.grid(alpha=.2)
            ax.axvline(common_end,color='#999',lw=.7,ls=':')
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'From-rest stabilization results: common saved window 0-{common_end:g} ms; available coarse trajectory to 300 ms',fontsize=13)
    fig.supxlabel('Curves stop at their last saved frame. Section metrics use actual wall/plane intersections, not an imposed constraint.',fontsize=9)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    # End-state centerlines of the completed coarse pair, no centering imposed.
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for label in ('baseline_solid1','stabilized_solid1'):
        folder=folders[label];a=np.load(folder/'continued_300ms/final_state.npz');cent=a['x'][a['inner_rings']].mean(1)
        ax.plot(cent[:,2]*10,cent[:,0]*10,label=label.replace('_',' '))
    ax.set(xlabel='Current z (mm)',ylabel='Current x (mm)',aspect='equal',title='Coarse solid centerline at 300 ms: x-z projection')
    ax.legend(fontsize=9);ax.grid(alpha=.2);fig.savefig(OUT/'centerlines_300ms.png',dpi=180);plt.close(fig)
    print('Common saved window:',common_end,'ms; no simulations advanced.')


if __name__=='__main__':main()
