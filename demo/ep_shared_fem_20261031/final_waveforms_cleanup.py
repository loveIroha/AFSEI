from pathlib import Path
import argparse,json,tarfile,hashlib,time,os,shutil
import numpy as np
ROOT=Path('/mnt/large2/gjh/AFSEI');RESULTS=ROOT/'results';PAIR=RESULTS/'ep_shared_fem_20261031'
TARGETS=['ep_afsei_nodal_euler1us_20261031','ep_afsei_nodal_ode_optimized_2ms_20261030',
 'ep_afsei_nodal_opencarp_matched_20261030','ep_afsei_nodal_opencarp_matched_20261030_smoke',
 'ep_euler_fem_pair_20261031','ep_pcg_optimization_20261031','ep_pcg_optimization_20261031_final',
 'ep_opencarp_sigma_nativeinit_20261030']
def inventory():
    data=[]
    for name in TARGETS:
        p=RESULTS/name
        if not p.exists():continue
        large=[];reports=[];total=0
        for f in p.rglob('*'):
            if not f.is_file() or f.is_symlink():continue
            size=f.stat().st_size;total+=size
            if size>=20_000_000:large.append(dict(path=str(f.relative_to(p)),bytes=size))
            if f.name=='report.json':
                try:
                    r=json.loads(f.read_text());reports.append(dict(path=str(f.relative_to(p)),status=r.get('status'),duration_ms=r.get('duration_ms'),steps=r.get('steps')))
                except Exception:pass
        data.append(dict(directory=name,total_bytes=total,reports=reports,large_files=large))
    print(json.dumps(data,indent=2),flush=True)
def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    oc=RESULTS/'opencarp_gpb_h0p0625_20261030';cache=RESULTS/'ep_opencarp_same_mesh_20261030'
    X=np.load(cache/'coordinates_cm.npy')*10;N=len(X)
    ref=np.memmap(oc/'simulation_v2/vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N))
    traces={m:np.load(PAIR/('h0p0625_'+m)/'traces.npz') for m in ['consistent_afsei','nodal_afsei']}
    probes=traces['nodal_afsei']['probe_ids'];out=PAIR/'comparison'
    labels={'consistent_afsei':'AFSEI consistent CG FEM + Euler','nodal_afsei':'AFSEI nodal FEM + Euler'}
    colors={'consistent_afsei':'#267bb7','nodal_afsei':'#e68620'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for j,ax in enumerate(axes.flat):
        for m in traces:ax.plot(traces[m]['time_ms'],traces[m]['voltage_mv'][:,j],lw=1.65,color=colors[m],label=labels[m])
        ax.plot(np.arange(51),ref[:,probes[j]],color='#303943',ls='--',marker='o',ms=2.3,lw=1.3,label='openCARP reference (reused)')
        ax.set(title='('+', '.join(f'{v:g}' for v in X[probes[j]])+') mm',xlabel='Time [ms]',ylabel='Vm [mV]',xlim=(0,50));ax.grid(alpha=.2)
        if j==0:ax.legend(fontsize=8)
    fig.suptitle('Final 3D benchmark | h = 0.0625 mm | shared optimized AFSEI\nCN PDE 20 us | identical GPB Euler ODE 1 us | original openCARP reference',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(out/('final_h0p0625_voltage_waveforms.'+ext),dpi=180)
    plt.close(fig)
    # Save the plotted data, on common 1 ms times; original fine EP traces remain.
    columns=[np.arange(51)]
    headers=['time_ms']
    for j in range(4):
        for m in traces:columns.append(np.interp(np.arange(51),traces[m]['time_ms'],traces[m]['voltage_mv'][:,j]));headers.append(f'probe{j}_{m}_mv')
        columns.append(ref[:,probes[j]]);headers.append(f'probe{j}_openCARP_mv')
    np.savetxt(out/'final_h0p0625_voltage_waveforms_1ms.csv',np.column_stack(columns),delimiter=',',header=','.join(headers),comments='')
    print('PLOT',str(out/'final_h0p0625_voltage_waveforms.png'),flush=True)
def clean(dry_run=False):
    """Only explicit obsolete EP directories; never traverse symlinks."""
    results=RESULTS.resolve(strict=True);pair=PAIR.resolve(strict=True)
    protected_roots=[pair,(RESULTS/'opencarp_gpb_h0p0625_20261030').resolve(strict=True),
        (RESULTS/'ep_opencarp_same_mesh_20261030').resolve(strict=True)]
    allowed_roots={(RESULTS/name).resolve(strict=True) for name in TARGETS if (RESULTS/name).exists()}
    # Pre-delete exclusions, not just an after-the-fact sentinel check.
    for base in allowed_roots:
        assert base.parent==results and base not in protected_roots
        assert all(not base.is_relative_to(p) and not p.is_relative_to(base) for p in protected_roots)
    protected=[RESULTS/'opencarp_gpb_h0p0625_20261030/simulation_v2/vm.igb',
        RESULTS/'ep_opencarp_same_mesh_20261030/operators.npz']
    for name in ('h0p125_consistent_afsei','h0p125_nodal_afsei','h0p0625_consistent_afsei','h0p0625_nodal_afsei'):
        base=PAIR/name;protected.extend([base/'report.json',base/'vm.igb',base/'visualization/ep.pvd'])
        protected.extend(base/'visualization'/f'ep_{k:03d}.vti' for k in range(51))
    signature=lambda f:(f.stat().st_size,f.stat().st_mtime_ns)
    initial={str(p):signature(p) for p in protected}
    plan=[];failed=[]
    for name in TARGETS:
        base=RESULTS/name
        if not base.exists():continue
        assert not base.is_symlink() and base.resolve().parent==results
        assert base.resolve()!=pair
        for f in base.rglob('*'):
            if not f.is_file() or f.is_symlink():continue
            resolved=f.resolve(strict=True)
            assert resolved.is_relative_to(base.resolve()) and resolved.is_relative_to(results)
            assert not any(resolved.is_relative_to(p) for p in protected_roots)
            if f.name=='report.json':
                try:
                    r=json.loads(f.read_text())
                    if r.get('status')=='failed':failed.append(str(f))
                except (ValueError,OSError):pass
            remove=(f.name in ('vm.igb','final_gpb_state.npy','fields.npz') or
                f.suffix.lower() in ('.vti','.vtu','.pvd','.pvtu'))
            if remove:plan.append(dict(path=str(resolved),bytes=f.stat().st_size))
    plan.sort(key=lambda x:x['path'])
    plan_digest=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    manifest=dict(scope='User-authorized obsolete EP benchmark bulk-output cleanup; keep final shared AFSEI runs, openCARP, meshes and all FSI results',
        target_directories=TARGETS,files=plan,planned_bytes=sum(x['bytes'] for x in plan),failed_reports=failed,status='planned')
    manifest.update(protected_roots=[str(p) for p in protected_roots],plan_sha256=plan_digest)
    reviewed=PAIR/'cleanup_plan_reviewed.json'
    if dry_run:
        reviewed.write_text(json.dumps(manifest,indent=2))
        summary={name:dict(files=sum(Path(x['path']).is_relative_to(RESULTS/name) for x in plan),bytes=sum(x['bytes'] for x in plan if Path(x['path']).is_relative_to(RESULTS/name))) for name in TARGETS}
        print(json.dumps(dict(dry_run=True,plan_sha256=plan_digest,protected_roots=manifest['protected_roots'],summary=summary,planned_bytes=manifest['planned_bytes'],sample=plan[:4]),indent=2));return
    assert reviewed.exists() and json.loads(reviewed.read_text())['plan_sha256']==plan_digest,'Read-only review must match exact deletion plan'
    dest=PAIR/'cleanup_manifest_20261031.json'
    assert not dest.exists(),'Cleanup already recorded'
    dest.write_text(json.dumps(manifest,indent=2))
    freed=0
    for item in plan:
        f=Path(item['path']);assert f.resolve().is_relative_to(results) and not f.is_symlink()
        resolved=f.resolve(strict=True)
        assert any(resolved.is_relative_to(p) for p in allowed_roots)
        assert not any(resolved.is_relative_to(p) for p in protected_roots),'Protected output explicitly excluded before unlink'
        assert f.stat().st_size==item['bytes'],'Output changed after inventory'
        f.unlink();freed+=item['bytes']
    for name in TARGETS:
        base=RESULTS/name
        if base.exists():
            (base/'RESULTS_PRUNED_20261031.md').write_text('旧EP结果已按用户要求精简。保留配置、报告、日志、探针及激活记录；重复三维电位/VTK与大型最终细胞状态已删除。此目录不再是完整模拟输出。\n最终完整AFSEI对照：../ep_shared_fem_20261031/；openCARP参考：../opencarp_gpb_h0p0625_20261030/。\n删除清单：../ep_shared_fem_20261031/cleanup_manifest_20261031.json。\n')
    assert initial=={str(p):signature(p) for p in protected},'Protected data changed'
    manifest.update(status='completed',removed_files=len(plan),removed_bytes=freed,removed_gib=freed/1024**3,protected_files_verified=len(protected))
    dest.write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inventory','plot','plan','clean']);a=ap.parse_args()
    if a.action=='inventory':inventory()
    elif a.action=='plot':plot()
    elif a.action=='plan':clean(dry_run=True)
    else:clean()
