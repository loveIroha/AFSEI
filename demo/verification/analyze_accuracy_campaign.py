"""Same-material field and work/accuracy assessment of existing tube outputs.

Missing/incomplete trajectories remain pending. No simulation, extrapolation,
rigid registration, pressure offset, or passing an invalid Richardson fit.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,base64,json,math,xml.etree.ElementTree as ET,zlib
import numpy as np,meshio
from afsi_torch.ep.refined_mesh import nested_mesh,quadrature_map
from analyze_available_volume_results import measure

ROOT=Path('/mnt/large2/gjh/AFSEI/results')
CAMPAIGN=ROOT/'accuracy_performance_20261025'

def sample_trilinear(value,index):
    """Trilinear interpolation with nearest boundary extension, NumPy only."""
    p=np.stack([np.clip(index[j],0,value.shape[j]-1) for j in range(3)])
    lo=np.floor(p).astype(np.int64);fraction=p-lo;result=np.zeros(p.shape[1:])
    from itertools import product
    for corner in product((0,1),repeat=3):
        weight=np.ones(p.shape[1:]);indices=[]
        for j,c in enumerate(corner):
            indices.append(np.minimum(lo[j]+c,value.shape[j]-1))
            weight*=fraction[j] if c else 1-fraction[j]
        result+=weight*value[tuple(indices)]
    return result

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def sources():
    old=ROOT/'volumetric_stabilization_20261025/from_rest_300ms'
    return {**{f'solid{f}_mac64':old/f'solid{f}' for f in (1,2)},
        **{f'solid3_mac{n}':CAMPAIGN/f'solid3_mac{n}' for n in (64,96,128)}}

def collect(folder,kind):
    result={}
    for phase in ('prefix_60ms','prefix_1ms','continued_300ms'):
        path=folder/phase/'vtk'/(kind+'.pvd')
        if not path.exists():continue
        for node in ET.parse(path).findall('.//DataSet'):
            time=round(float(node.attrib['timestep']),9)
            if abs(time/.01-round(time/.01))<1e-6:
                result[time]=path.parent/node.attrib['file']
    return result

def read_vti(path):
    root=ET.parse(path);image=root.find('ImageData')
    extent=list(map(int,image.attrib['WholeExtent'].split()))
    shape=tuple(extent[2*j+1]-extent[2*j] for j in range(3));result={}
    for node in root.findall('.//CellData/DataArray'):
        s=node.text.strip();first=np.frombuffer(base64.b64decode(s[:32]),dtype='<u8')
        blocks=int(first[0]);header_chars=4*math.ceil(8*(3+blocks)/3)
        h=np.frombuffer(base64.b64decode(s[:header_chars]),dtype='<u8')
        payload=base64.b64decode(s[header_chars:]);parts=[];offset=0
        for size in h[3:]:
            n=int(size);parts.append(zlib.decompress(payload[offset:offset+n]));offset+=n
        a=np.frombuffer(b''.join(parts),dtype='<f8');components=int(node.attrib.get('NumberOfComponents','1'))
        if components==1:a=a.reshape(shape[::-1]).transpose(2,1,0)
        else:a=a.reshape((*shape[::-1],components)).transpose(2,1,0,3)
        result[node.attrib['Name']]=a
    return result,np.array(image.attrib['Origin'].split(),float),np.array(image.attrib['Spacing'].split(),float)

def setup(folder,name):
    cfgpath=folder/'config.json'
    if not cfgpath.exists():cfgpath=folder/'prefix_60ms/config.resolved.json'
    c=json.loads(cfgpath.read_text());factor=c.get('solid_refinement_factor',1)
    state=next((folder/p/'final_state.npz' for p in ('prefix_1ms','prefix_60ms','continued_300ms') if (folder/p/'final_state.npz').exists()),None)
    if state is None:return None
    with np.load(state) as z:
        X=z['reference'].copy();cells=z['cells'].copy();rings=z['inner_rings'].copy();faces=z['inner_faces'].copy()
    D=(X[cells[:,1:]]-X[cells[:,:1]]).transpose(0,2,1)
    geo=dict(X=X,cells=cells,vol=abs(np.linalg.det(D))/6,rings=rings,inner_faces=faces)
    # Four symmetric degree-two points in each ORIGINAL EP/solid tetrahedron.
    a=.5854101966249685;b=.1381966011250105;values=np.full((4,4),b);np.fill_diagonal(values,a)
    # Child 0 need not carry all original tetra vertices: recover original topology.
    from afsi_torch.solids.circular_tube import CircularTubeGeometry,generate_circular_tube
    base=generate_circular_tube(CircularTubeGeometry(**c['geometry']))
    coarse_cells=base.cells.numpy();coarse_X=base.X.numpy()
    if not np.array_equal(coarse_X,X[:len(coarse_X)]):raise ValueError('original material coordinates changed')
    V=abs(np.linalg.det((coarse_X[coarse_cells[:,1:]]-coarse_X[coarse_cells[:,:1]]).transpose(0,2,1)))/6
    if factor==1:
        ids=np.broadcast_to(coarse_cells[:,None,:],(len(coarse_cells),4,4));weights=np.broadcast_to(values,(len(coarse_cells),4,4))
    else:
        data=nested_mesh(coarse_X,coarse_cells,factor)
        if not np.array_equal(data['X'],X):raise ValueError('material interpolation mesh mismatch')
        ids,weights=quadrature_map(data,values)
    reference=np.sum(X[ids]*weights[...,None],axis=2)
    return dict(config=c,geo=geo,factor=factor,ids=ids,weights=weights,reference=reference,qweight=np.repeat(V[:,None]/4,4,axis=1))

def scalar_summary(rows):
    return dict(last_time_ms=rows[-1]['time_ms'],
        minimum_normal_diameter_mm=min(r['sampled_min_normal_diameter_mm'] for r in rows if r['sampled_min_normal_diameter_mm'] is not None),
        max_wall_volume_error_fraction=max(abs(r['wall_volume_change_percent'])/100 for r in rows),
        final_local_J_RMS=rows[-1]['J_rms_error'],
        final_fraction_abs_J_error_over_10pct=rows[-1]['reference_fraction_abs_Jerror_gt10pct'],
        all_sections_valid=all(r['valid_sections']==r['expected_sections'] for r in rows),
        min_sampled_J=min(r['minimum_J'] for r in rows))

def main():
    import torch
    torch.set_num_threads(2)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=CAMPAIGN/'analysis')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    folders=sources();setups={};frames={};curves={};status={}
    for name,folder in folders.items():
        if not folder.exists():status[name]=dict(stage='not-started');continue
        sp=folder/'status.json';status[name]=json.loads(sp.read_text()) if sp.exists() else dict(stage='unknown')
        q=setup(folder,name)
        if q is None:continue
        setups[name]=q;solid=collect(folder,'solid');fluid=collect(folder,'fluid');frames[name]=dict(solid=solid,fluid=fluid)
        rows=[];hs=[]
        for phase in ('prefix_60ms','prefix_1ms','continued_300ms'):
            hp=folder/phase/'history.json'
            if hp.exists():hs.extend(json.loads(hp.read_text()))
        for t,path in sorted(solid.items()):
            m=meshio.read(path);J=m.cell_data_dict['J_min']['tetra'];r=measure(m.points,J,q['geo'],q['factor'])
            near=[h for h in hs if abs(h['time_s']-t)<1e-8]
            if near:r['lumen_volume_ml']=near[-1]['lumen_volume_ml']
            rows.append(dict(time_ms=t*1000,**r))
        curves[name]=rows
        print(name,'frames',len(rows),'last_ms',rows[-1]['time_ms'] if rows else None,flush=True)
    pairs={};origin=np.array([-.7,-.7,0.]);h=2.8/64
    available_ends=[max(v['solid']) for v in frames.values() if v['solid'] and v['fluid']]
    comparison_end=min(available_ends) if available_ends else 0.
    # The completed solid study must not be truncated by a still-running fluid
    # study. Both differences in EACH three-level trend use the same window.
    families={'solid_at_mac64':('solid1_mac64','solid2_mac64','solid3_mac64'),
              'fluid_at_solid3':('solid3_mac64','solid3_mac96','solid3_mac128')}
    windows={family:min(min(max(frames[name][kind]) for kind in ('solid','fluid'))
        for name in names) if all(name in frames and all(frames[name][kind] for kind in ('solid','fluid'))
        for name in names) else 0. for family,names in families.items()}
    probe=np.stack(np.meshgrid(*[origin[j]+h*(np.arange(64)+.5) for j in range(3)],indexing='ij'),axis=-1)
    for left,right in [('solid1_mac64','solid2_mac64'),('solid2_mac64','solid3_mac64'),('solid3_mac64','solid3_mac96'),('solid3_mac96','solid3_mac128')]:
        if left not in frames or right not in frames:continue
        family='solid_at_mac64' if left.startswith(('solid1_','solid2_')) else 'fluid_at_solid3'
        pair_end=windows[family]
        common=sorted(t for t in set(frames[left]['solid'])&set(frames[right]['solid'])&set(frames[left]['fluid'])&set(frames[right]['fluid']) if t<=pair_end+1e-10)
        if not common:continue
        qa,qb=setups[left],setups[right];ra={round(r['time_ms']/1000,9):r for r in curves[left]};rb={round(r['time_ms']/1000,9):r for r in curves[right]}
        rows=[]
        for t in common:
            ma=meshio.read(frames[left]['solid'][t]);mb=meshio.read(frames[right]['solid'][t])
            ua=np.sum(ma.points[qa['ids']]*qa['weights'][...,None],axis=2)-qa['reference']
            ub=np.sum(mb.points[qb['ids']]*qb['weights'][...,None],axis=2)-qb['reference']
            w=qb['qweight'];norm=lambda x:float(np.sqrt(np.sum(w*np.sum(x*x,axis=-1))))
            row=dict(time_ms=t*1000,displacement_error_L2=norm(ua-ub),displacement_reference_L2=norm(ub),
                diameter_relative_error=abs(ra[t]['sampled_min_normal_diameter_mm']-rb[t]['sampled_min_normal_diameter_mm'])/rb[t]['sampled_min_normal_diameter_mm'],
                lumen_volume_relative_error=abs(ra[t]['lumen_volume_ml']-rb[t]['lumen_volume_ml'])/rb[t]['lumen_volume_ml'])
            fields=[]
            for name in (left,right):
                data,o,spacing=read_vti(frames[name]['fluid'][t]);index=np.moveaxis((probe-o)/spacing-.5,-1,0)
                pressure=sample_trilinear(data['pressure_dyn_per_cm2'],index)
                velocity=np.stack([sample_trilinear(data['velocity_cm_per_s'][...,k],index) for k in range(3)],axis=-1)
                fields.append((pressure,velocity))
            for k,label in ((0,'pressure'),(1,'velocity')):
                a,b=fields[0][k],fields[1][k]
                row[label+'_error_L2']=float(np.sqrt(np.sum((a-b)**2)*h**3))
                row[label+'_reference_L2']=float(np.sqrt(np.sum(b*b)*h**3))
            rows.append(row)
        pair=dict(left=left,right=right,family=family,last_common_ms=common[-1]*1000,curve=rows)
        for label in ('displacement','velocity','pressure'):
            pair[label+'_relative_Linf_time_L2_space']=max(r[label+'_error_L2'] for r in rows)/max(max(r[label+'_reference_L2'] for r in rows),1e-30)
        pair['max_diameter_relative_error']=max(r['diameter_relative_error'] for r in rows)
        pair['max_lumen_volume_relative_error']=max(r['lumen_volume_relative_error'] for r in rows)
        pairs[left+'__'+right]=pair;print('pair',left,right,pair['displacement_relative_Linf_time_L2_space'],flush=True)
    summaries={k:scalar_summary(v) for k,v in curves.items() if v}
    complete=all(status.get(k,{}).get('stage')=='completed' for k in folders) and all(s.get('last_time_ms',0)>=300-1e-7 for s in summaries.values()) and len(summaries)==5
    checks={}
    for key in ('solid2_mac64__solid3_mac64','solid3_mac96__solid3_mac128'):
        r=pairs.get(key)
        if r is None:continue
        checks[key]=dict(diameter=r['max_diameter_relative_error']<=.02,lumen_volume=r['max_lumen_volume_relative_error']<=.02,
            **{label:r[label+'_relative_Linf_time_L2_space']<=.05 for label in ('displacement','velocity','pressure')})
    for key in ('solid3_mac64','solid3_mac128'):
        if key not in summaries:continue
        s=summaries[key];checks[key]=dict(global_volume=s['max_wall_volume_error_fraction']<=.01,
            local_J_RMS=s['final_local_J_RMS']<=.02,local_J_tail=s['final_fraction_abs_J_error_over_10pct']<=.01,
            positive_J=s['min_sampled_J']>0,valid_sections=s['all_sections_valid'])
    for coarse,middle in [('solid1_mac64__solid2_mac64','solid2_mac64__solid3_mac64'),('solid3_mac64__solid3_mac96','solid3_mac96__solid3_mac128')]:
        if coarse in pairs and middle in pairs:
            keys=['displacement_relative_Linf_time_L2_space','velocity_relative_Linf_time_L2_space','pressure_relative_Linf_time_L2_space','max_diameter_relative_error','max_lumen_volume_relative_error']
            checks['decreasing_'+middle]={k:pairs[middle][k]<pairs[coarse][k] for k in keys}
    passed=complete and len(checks)==6 and all(all(v.values()) for v in checks.values())
    report=dict(updated_utc=datetime.now(timezone.utc).isoformat(),complete=complete,
        fixed_dt_spatial_acceptance_passed=passed,full_space_time_convergence_established=False,comparison_end_ms=comparison_end*1000,
        comparison_windows_ms={family:t*1000 for family,t in windows.items()},
        cross_software_performance_established=False,status=status,summaries=summaries,checks=checks,
        pairs=pairs,normalization='field differences: max time L2 error / max time reference L2 norm; scalar relative errors at each common time',
        fluid_fields='cell-centered visualized MAC velocity, sampled trilinearly on common physical 64^3 centers; raw staggered final fields retained separately',
        pressure='same zero boundary reference; Eulerian multiplier, no fitted pressure shift',
        limitations=['Reference is a finer discrete solution, not an exact continuum truth.','Only fixed coarse EP drive spatial FSI study.','Finite window 0-300 ms.','Full space-time convergence needs time refinement.'])
    save(out/'report.json',report);save(out/'frame_metrics.json',curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name,rows in curves.items():
        for ax,key,label in zip(axes.flat,('sampled_min_normal_diameter_mm','lumen_volume_ml','wall_volume_change_percent','J_rms_error'),('Minimum normal-section diameter (mm)','Lumen volume (mL)','Wall-volume change (%)','Local J RMS error')):
            ax.plot([r['time_ms'] for r in rows],[r.get(key,np.nan) for r in rows],label=name.replace('_',' / '))
            ax.set(xlabel='Time (ms)',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Fixed-model spatial study: curves end at actual saved times')
    for ext in ('png','pdf','svg'):fig.savefig(out/f'comparison.{ext}',dpi=180)
    print(json.dumps(dict(complete=complete,fixed_dt_spatial_acceptance_passed=passed,checks=checks),indent=2))

if __name__=='__main__':main()
