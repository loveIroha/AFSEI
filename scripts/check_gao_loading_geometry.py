"""Read-only coverage check over reference and saved preloaded cap geometry."""
import json
from pathlib import Path
import numpy as np
import torch
from afsi_torch.ep.mac_ports import MACBasalPorts
from afsi_torch.circulation.geometry import PortGeometry
from afsi_torch.mac.grid import MACGrid
from afsi_torch.ep.gao_tracts import LVTractDomain
from afsi_torch.ep.tract_storage import TractStorage

torch.set_num_threads(4)
source='results/afsei_peak665_av060_20261019/diagnostics/tick_00010002/checkpoint.npz'
with np.load(source,allow_pickle=False) as a:
    meta=json.loads(str(a['metadata']))
    X=torch.as_tensor(a['X'],device='cuda');x=torch.as_tensor(a['x'],device='cuda')
    faces=torch.as_tensor(a['faces'][a['facet_tags']==1],device='cuda')
grid=MACGrid((128,)*3,(13.,)*3)
ports=meta['balanced_execution']['port_geometry']
cap=MACBasalPorts(X,faces,grid,ports=PortGeometry(**ports) if ports else None,
    integration_rule=meta['balanced_execution'].get('cap_integration_rule','centroid'))
out=[]
for name,anchor in [('preloaded',x),('reference',X)]:
    try:
        domain=LVTractDomain(grid,cap,anchor,mode='filling');storage=TractStorage(cap,domain)
    except ValueError as exc:
        out.append(dict(anchor=name,valid=False,reason=str(exc)))
        continue
    entry=dict(anchor=name,geometry=domain.metadata,states=[])
    for fraction in (0.,.25,.5,.75,1.):
        y=X+fraction*(x-X)
        try:
            value=storage.volume(y);domain.validate_wall(cap,y)
            entry['states'].append(dict(fraction=fraction,valid=True,tract_volume_ml=value))
        except ValueError as exc:
            entry['states'].append(dict(fraction=fraction,valid=False,reason=str(exc)))
    out.append(entry)
path=Path('results/gao_loading_geometry_20261026.json')
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
