"""Independent UFL stress/derivative check; no AFSEI/Torch imports."""
from pathlib import Path
import json,numpy as np
from reference_solid_stabilized import ReferenceSolid

root=Path('/work');a=np.load(root/'input.npz');c=json.loads((root/'model.json').read_text())
solid=ReferenceSolid(a['X'],a['cells'],a['rest'],c);solid.stage(a['land'])
force=solid.force(a['x']).reshape(3,-1).T
jvp=(solid.jacobian(a['x'])@a['direction'].T.ravel()).reshape(3,-1).T
relative=lambda x,y:float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))
report=dict(scope='new stabilized stress and full FE tangent only; independent DOLFINx/UFL versus CPU PyTorch assembly; CUDA equivalence tested separately',
    nodes=len(a['X']),cells=len(a['cells']),force_relative=relative(a['force'],force),jvp_relative=relative(a['jvp'],jvp),
    tolerances=dict(force_relative=1e-10,jvp_relative=1e-9))
report['passed']=all(report[k]<=v for k,v in report['tolerances'].items())
np.savez_compressed(root/'reference.npz',force=force,jvp=jvp)
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if not report['passed']:raise AssertionError('independent stabilized stress mismatch')
