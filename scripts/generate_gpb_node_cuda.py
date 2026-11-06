"""Regenerate the optimized production GPB kernel from unchanged sources.

No GPB equation/parameter, FP64 precision, SDIRK stage iteration, Land update,
or current quadrature is changed. Run from any directory; inputs resolve
relative to the AFSEI repository containing this script.
"""
from pathlib import Path
import re,json,hashlib
from generate_cached_rhs import generate
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'src/afsi_torch/ep/csrc'
original=(folder/'gpb_device.h').read_text();tissue=(folder/'tissue.cu').read_text()
assert len(re.findall(r'\bt\b',re.sub(r'//[^\n]*','',original)))==1,'Derivative reuse requires an autonomous GPB RHS'
target=folder/'gpb_node_device.h'
meta=generate(folder/'gpb_device.h',target)
h=target.read_text().replace('gpb_cached_rhs','gpb_nodal_rhs').replace('double* y, double Vm','const double* y, double Vm')
h=h.replace('const GPBVoltageCache& cache) {','const GPBVoltageCache& cache,double* relative_error=nullptr) {')
h=re.sub(r'for \(int i=0; i<58; \+\+i\) ydot\[i\]=0\.0;','',h)
h=re.sub(r'^\s*y\[38\] = Vm;','',h,flags=re.M)
h=re.sub(r'^\s*ydot\[38\] = 0\.0;[^\n]*','',h,flags=re.M)
h=re.sub(r'for \(int i = 43; i < 58; i\+\+\) \{\s*ydot\[i\] = 0\.0;\s*\}',
    '\n'.join(f'ydot[{i}] = 0.0;' for i in range(43,58)),h)
h=re.sub(r'\by\[(\d+)\]',lambda m:'Vm' if int(m[1])==38 else 'y['+str(int(m[1])-(int(m[1])>38))+']',h)
h=re.sub(r'\bydot\[(\d+)\]',lambda m:'ydot['+str(int(m[1])-(int(m[1])>38))+']',h)
h=re.sub(r'//[^\n]*','',h)
def error_write(m):
    index='ydot['+m[1]+']'
    return '{double next=('+m[2]+');if(relative_error) *relative_error=fmax(*relative_error,fabs(next-'+index+')/fmax(fabs(next),1.));'+index+'=next;}'
h=re.sub(r'ydot\[(\d+)\]\s*=\s*([^;]+);',error_write,h)
target.write_text('// Generated from GPB_cell_model.cpp revision 5bdde73c6be10cefadb005a0fcec5fdb6521f336 via generate_gpb_node_cuda.py.\n'+h)
cu=tissue.replace('#include "gpb_device.h"','#include "gpb_node_device.h"')
start=cu.index('__device__ inline double tissue_rhs(');end=cu.index('\n__global__ void rhs_kernel',start)
cu=cu[:start]+'''__device__ inline double tissue_rhs(const double* state,double voltage,double time,double* derivative,
    const GPBVoltageCache& cache,double* relative_error=nullptr) {
 return gpb_nodal_rhs(time,state,voltage,0.,derivative,cache,relative_error);
}
'''+cu[end:]
cu=cu.replace('if (i>=n) return;', 'if (i>=n) return;\n    auto cache=make_gpb_voltage_cache(voltage[i]);')
cu=cu.replace('derivative+57*i);','derivative+57*i,cache);')
cu=cu.replace('int& guarded) {','int& guarded,double* k1,const GPBVoltageCache& cache) {')
cu=cu.replace('double k1[57],k2[57],temp[57],previous[57];','double k2[57],temp[57];')
cu=cu.replace('    tissue_rhs(state,voltage,time,k1);','')
cu=cu.replace('previous[j]=k1[j];','').replace('previous[j]=k2[j];','')
cu=cu.replace('tissue_rhs(temp,voltage,time+gamma*dt,k1);',
    'double error=0.;tissue_rhs(temp,voltage,time+gamma*dt,k1,cache,iter>0?&error:nullptr);')
cu=cu.replace('tissue_rhs(temp,voltage,time+(1.0-gamma)*dt,k2);',
    'double error=0.;tissue_rhs(temp,voltage,time+(1.0-gamma)*dt,k2,cache,iter>0?&error:nullptr);')
cu=re.sub(r'if \(iter>0\) \{\s*double error=0\.0;\s*for \(int j=0;j<57;\+\+j\) error=fmax\(error,fabs\(k[12]\[j\]-previous\[j\]\)/fmax\(fabs\(k[12]\[j\]\),1\.0\)\);\s*if \(error<1e-6\) break;\s*\}',
    'if (iter>0 && error<1e-6) break;',cu)
cu=cu.replace('time+dt,temp);','time+dt,k1,cache);')
cu=cu.replace('double sum=0.0;','double sum=0.0;\n    double k1[57];tissue_rhs(state,voltage[i],time,k1,cache);')
cu=cu.replace('dt,guarded);','dt,guarded,k1,cache);')
assert 'previous[' not in cu and 'full[58]' not in cu
(folder/'tissue_node.cu').write_text(cu)
meta.update(source_files_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['gpb_device.h','tissue.cu']},
    generated_files=['gpb_node_device.h','tissue_node.cu'],SDIRK_changed=False,Land_changed=False,
    endpoint_derivative_reuse='Autonomous GPB only; verified no t use beyond original function signature',
    performance_validation='demo/cell_optimization_20261030/validation_fused.json')
(folder/'gpb_node_provenance.json').write_text(json.dumps(meta,indent=2))
print('Generated production nodal GPB execution: unchanged model, FP64 and SDIRK.')
