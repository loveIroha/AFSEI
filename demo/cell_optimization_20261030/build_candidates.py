from pathlib import Path
import re,json
ROOT=Path('/mnt/large2/gjh/AFSEI');out=ROOT/'demo/cell_optimization_20261030'
native=ROOT/'demo/ep_afsei_nodal_opencarp_matched';prod=ROOT/'src/afsi_torch/ep/csrc'
cu=(native/'reference_sdirk.cu').read_text()
cu=cu.replace('#include "reference_grandi_rhs.h"','#include "cached_grandi_rhs.h"')
cu=cu.replace('#include "reference_grandi_rl_check.h"',f'#include "{native}/reference_grandi_rl_check.h"')
cu=cu.replace('reference_grandi_sdirk::rhs','cached_grandi_sdirk::rhs')
cu=cu.replace('double k1[NS],k2[NS],temp[NS],previous[NS];double sum=0.;',
    'auto cache=cached_grandi_sdirk::make_voltage_cache(V);\n  double k1[NS],k2[NS],temp[NS],previous[NS];double sum=0.;')
cu=cu.replace('double s[NS],d[NS];','double s[NS],d[NS];auto cache=cached_grandi_sdirk::make_voltage_cache(V[id]);',1)
cu=re.sub(r'cached_grandi_sdirk::rhs\(([^;]+?)\)',lambda m:'cached_grandi_sdirk::rhs('+m.group(1)+',cache)',cu)
# CPU rhs wrapper has an independent per-node scope.
cu=cu.replace('double s[NS],d[NS];for(int j=0;',
    'double s[NS],d[NS];auto cache=cached_grandi_sdirk::make_voltage_cache(v[id]);for(int j=0;')
(out/'cached_reference.cu').write_text(cu)
(out/'cached_reference.cpp').write_text((native/'reference_sdirk.cpp').read_text())

cu=(prod/'tissue.cu').read_text().replace('#include "gpb_device.h"','#include "cached_gpb_rhs.h"')
cu=cu.replace('#include "land_device.h"',f'#include "{prod}/land_device.h"')
cu=cu.replace('double time, double* derivative) {','double time, double* derivative,const GPBVoltageCache& cache) {')
cu=cu.replace('gpb_full_rhs(time,full,voltage,0.0,dfull)','gpb_cached_rhs(time,full,voltage,0.0,dfull,cache)')
cu=cu.replace('int& guarded) {','int& guarded,const GPBVoltageCache& cache) {')
cu=cu.replace('if (i>=n) return;', 'if (i>=n) return;\n    auto cache=make_gpb_voltage_cache(voltage[i]);')
cu=cu.replace('derivative+57*i);','derivative+57*i,cache);')
cu=cu.replace('time,k1);','time,k1,cache);').replace('time+gamma*dt,k1);','time+gamma*dt,k1,cache);')
cu=cu.replace('time+(1.0-gamma)*dt,k2);','time+(1.0-gamma)*dt,k2,cache);')
cu=cu.replace('time+dt,temp);','time+dt,temp,cache);')
cu=cu.replace('dt,guarded);','dt,guarded,cache);')
(out/'cached_tissue.cu').write_text(cu)
(out/'cached_tissue.cpp').write_text((prod/'tissue.cpp').read_text().replace('afsei_ep_cuda','afsei_ep_cuda_cached'))
(out/'candidates.py').write_text('''from pathlib import Path
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
def native():
 return load(name='afsei_cached_native_gpb',sources=[str(ROOT/'cached_reference.cpp'),str(ROOT/'cached_reference.cu')],extra_cflags=['-O3','-ffp-contract=off'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
def production():
 load(name='afsei_ep_cuda_cached',sources=[str(ROOT/'cached_tissue.cpp'),str(ROOT/'cached_tissue.cu')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],is_python_module=False,verbose=False)
 import torch
 return torch.ops.afsei_ep_cuda_cached
''')
print('Cached native and production candidates generated; baseline sources untouched.')
