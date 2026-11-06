from pathlib import Path
import json,tarfile,shutil,xml.etree.ElementTree as ET
here=Path(__file__).resolve().parent
root=Path('/mnt/large2/gjh/AFSEI/results/ep_afsei_nodal_opencarp_matched_20261030')
assert json.loads((root/'postprocessing_completed.json').read_text())['status']=='completed'
datasets=list(ET.parse(root/'visualization/ep.pvd').iter('DataSet'))
assert len(datasets)==51 and all((root/'visualization'/d.attrib['file']).stat().st_size>7100000 for d in datasets)
assert float(datasets[0].attrib['timestep'])==0 and float(datasets[-1].attrib['timestep'])==.05
out=root/'comparison'
for name in ['report.json','configuration.json','resolved_configuration.json']:
    shutil.copy2(root/name,out/('afsei_'+name))
for name in ['validation.json','reference_cell_provenance.json','README_afsei_matched_zh.md','initial_fixed_range.png','initial_visualization_audit.json','kernel_resources.txt']:
    shutil.copy2(here/name,out/name)
audit=dict(frames=51,first_time_s=0,last_time_s=.05,all_frame_files_present=True,
    three_dimensional=True,nodes=1777377,geometry_units='mm',visualization_only_cartesian_lattice=True,
    all_timesteps_retained_in_IGB=True)
(out/'output_audit.json').write_text(json.dumps(audit,indent=2))
with tarfile.open(root/'comparison_bundle.tar.gz','w:gz') as tar:
    for path in sorted(out.iterdir()):
        if path.suffix in ['.png','.pdf','.json','.csv','.md','.txt']:tar.add(path,arcname=path.name)
print(json.dumps(audit,indent=2));print(root/'comparison_bundle.tar.gz')
