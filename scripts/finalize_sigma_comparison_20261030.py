from pathlib import Path
import json,os,shutil,subprocess,tarfile,xml.etree.ElementTree as ET
root=Path('/mnt/large2/gjh/AFSEI');out=root/'results/ep_opencarp_sigma_nativeinit_20261030'
report=json.loads((out/'report.json').read_text())
if report['status']!='completed':raise RuntimeError('Both trajectories must be complete')
audit=json.loads((out/'initial_frame_audit.json').read_text())
series={}
for method in ('consistent_tc','nodal_tc'):
    if not audit[method]['all_nodes_have_native_initial_voltage']:raise RuntimeError('Initial-value audit failed')
    for name in ('view_voltage_fixed_range.pvsm','README_sigma_nativeinit_zh.md'):
        shutil.copy2(out/name,out/'3d'/method/name)
    folder=out/'3d'/method
    datasets=list(ET.parse(folder/'ep.pvd').iter('DataSet'))
    if len(datasets)!=13 or not all((folder/p.attrib['file']).is_file() for p in datasets):raise RuntimeError('Incomplete PVD collection')
    series[method]=dict(frames=len(datasets),first_time_s=float(datasets[0].attrib['timestep']),last_time_s=float(datasets[-1].attrib['timestep']))
(out/'series_audit.json').write_text(json.dumps(series,indent=2))
env=os.environ.copy();env['MPLCONFIGDIR']='/mnt/large2/gjh/opencarp-runtime/v20.0/matplotlib-cache'
subprocess.run(['/mnt/large2/gjh/venvs/opencarp/bin/python',str(root/'scripts/compare_completed_sigma_20261030.py')],cwd=root,env=env,check=True)
dest=out/'comparison'
for name in ('initial_frame_audit.json','series_audit.json','alignment.json','initial_fixed_range.png','README_sigma_nativeinit_zh.md','view_voltage_fixed_range.pvsm'):
    shutil.copy2(out/name,dest/name)
with tarfile.open(dest/'comparison_bundle.tar.gz','w:gz') as tar:
    for p in sorted(dest.iterdir()):
        if p.suffix in ('.png','.pdf','.json','.csv','.md','.pvsm'):tar.add(p,arcname=p.name)
print('COMPLETE',dest,flush=True)
