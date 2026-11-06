import json, sys
import numpy as np
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.ep.mac_ports import MACBasalPorts
from afsi_torch.circulation.geometry import PortGeometry

torch.set_num_threads(4)
with np.load(sys.argv[1], allow_pickle=False) as a:
    get=lambda k:torch.as_tensor(a[k])
    X,x=get('X'),get('x')
    f=get('faces')[get('facet_tags')==1]
    cap=MACBasalPorts(X,f,MACGrid((128,)*3,(13.,)*3),ports=PortGeometry(av_radius_cm=.6))
    p,av=cap.surface(x)
    out={'frame':cap.frame.tolist(),'center':cap.center.tolist(),'rim_z':x[cap.rim[:,0],2].aminmax().__str__(),'ports':[]}
    for label in (1,2):
        pts=p[cap.labels==label]
        lo,hi=pts.amin(0),pts.amax(0)
        out['ports'].append({'center':((lo+hi)/2).tolist(),'lower':lo.tolist(),'upper':hi.tolist()})
    print(json.dumps(out,indent=2))
