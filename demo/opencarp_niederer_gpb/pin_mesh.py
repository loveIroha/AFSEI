from pathlib import Path
import json
demo=Path('/mnt/large2/gjh/AFSEI/demo/opencarp_niederer_gpb')
out=Path('/mnt/large2/gjh/AFSEI/results/opencarp_gpb_h0p0625_20261030')
report=json.loads((out/'mesh_validation.json').read_text())
p=demo/'run.py';src=p.read_text()
old='    meshname = mesh.generate(geom)'
if old not in src:raise RuntimeError('Expected original mesh generation line')
src=src.replace(old,'    # Reuse the native mesh already generated and independently validated.\n    meshname = '+repr(report['mesh_path']))
p.write_text(src)
manifest=json.loads((demo/'experiment.json').read_text())
manifest.update(carputils_git='49bc2a4eabc5efc019aab174142ddf89fda18812',
    mesh_path=report['mesh_path'],execution_note='Exact carputils commit pinned by openCARP v20.0; reuse existing native mesh',
    simulation_directory=str(out/'simulation_v2'))
(demo/'experiment.json').write_text(json.dumps(manifest,indent=2))
print('Reusing',report['mesh_path'])
