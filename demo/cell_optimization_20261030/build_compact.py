"""Remove redundant full-state remapping and previous-derivative arrays.

Every derivative formula, iteration cap and convergence test is retained.
The error max is accumulated as derivatives are published instead of first
copying an entire previous vector. Source GPB equations are never hand edited.
"""
from pathlib import Path
import re,json,hashlib
root=Path('/mnt/large2/gjh/AFSEI');out=root/'demo/cell_optimization_20261030'

def write_error(expr,index):
    return ('{double next=('+expr+');if(relative_error) *relative_error=fmax(*relative_error,'
        'fabs(next-'+index+')/fmax(fabs(next),1.));'+index+'=next;}')

h=(out/'cached_grandi_rhs.h').read_text()
h=h.replace('namespace cached_grandi_sdirk','namespace compact_grandi_sdirk')
h=h.replace('const GrandiVoltageCache& cache) {','const GrandiVoltageCache& cache,double* relative_error=nullptr) {')
h=re.sub(r'dy\[(\d+)\]\s*=\s*([^;]+);',lambda m:write_error(m[2],'dy['+m[1]+']'),h)
(out/'compact_grandi_rhs.h').write_text(h)
cu=(out/'cached_reference.cu').read_text().replace('cached_grandi_rhs.h','compact_grandi_rhs.h')
cu=cu.replace('cached_grandi_sdirk','compact_grandi_sdirk')
cu=cu.replace('temp[NS],previous[NS]','temp[NS]')
cu=cu.replace('previous[j]=k1[j];','').replace('previous[j]=k2[j];','')
cu=cu.replace('compact_grandi_sdirk::rhs(temp,V,k1,cache);',
    'double error=0.;compact_grandi_sdirk::rhs(temp,V,k1,cache,iter>0?&error:nullptr);')
cu=cu.replace('compact_grandi_sdirk::rhs(temp,V,k2,cache);',
    'double error=0.;compact_grandi_sdirk::rhs(temp,V,k2,cache,iter>0?&error:nullptr);')
cu=re.sub(r'if\(iter>0\)\{double e=0\.;for\(int j=0;j<NS;\+\+j\)e=fmax\(e,fabs\(k[12]\[j\]-previous\[j\]\)/fmax\(fabs\(k[12]\[j\]\),1\.\)\);if\(e<1e-6\)break;\}',
    'if(iter>0 && error<1e-6)break;',cu)
assert 'previous[' not in cu
(out/'compact_reference.cu').write_text(cu)
(out/'compact_reference.cpp').write_text((out/'cached_reference.cpp').read_text())

h=(out/'cached_gpb_rhs.h').read_text().replace('gpb_cached_rhs','gpb_nodal_rhs')
h=h.replace('double* y, double Vm','const double* y, double Vm')
h=h.replace('const GPBVoltageCache& cache) {','const GPBVoltageCache& cache,double* relative_error=nullptr) {')
h=re.sub(r'for \(int i=0; i<58; \+\+i\) ydot\[i\]=0\.0;','',h)
h=re.sub(r'^\s*y\[38\] = Vm;','',h,flags=re.M)
h=re.sub(r'^\s*ydot\[38\] = 0\.0;[^\n]*','',h,flags=re.M)
h=re.sub(r'for \(int i = 43; i < 58; i\+\+\) \{\s*ydot\[i\] = 0\.0;\s*\}',
    '\n'.join(f'ydot[{i}] = 0.0;' for i in range(43,58)),h)
h=re.sub(r'\by\[(\d+)\]',lambda m:'Vm' if int(m[1])==38 else 'y['+str(int(m[1])-(int(m[1])>38))+']',h)
h=re.sub(r'\bydot\[(\d+)\]',lambda m:'ydot['+str(int(m[1])-(int(m[1])>38))+']',h)
# Remove commented equations before changing derivative publications.
h=re.sub(r'//[^\n]*','',h)
h=re.sub(r'ydot\[(\d+)\]\s*=\s*([^;]+);',lambda m:write_error(m[2],'ydot['+m[1]+']'),h)
assert not re.search(r'ydot\[i\]|y\[38\]\s*=',h)
(out/'compact_gpb_rhs.h').write_text('// Generated from unchanged gpb_device.h via generate_cached_rhs.py + build_compact.py.\n'+h)
cu=(out/'cached_tissue.cu').read_text().replace('cached_gpb_rhs.h','compact_gpb_rhs.h')
start=cu.index('__device__ inline double tissue_rhs(');end=cu.index('\n__global__ void rhs_kernel',start)
cu=cu[:start]+'''__device__ inline double tissue_rhs(const double* state,double voltage,double time,double* derivative,
    const GPBVoltageCache& cache,double* relative_error=nullptr) {
 return gpb_nodal_rhs(time,state,voltage,0.,derivative,cache,relative_error);
}
'''+cu[end:]
cu=cu.replace('temp[57],previous[57]','temp[57]')
cu=cu.replace('previous[j]=k1[j];','').replace('previous[j]=k2[j];','')
cu=cu.replace('tissue_rhs(temp,voltage,time+gamma*dt,k1,cache);',
    'double error=0.;tissue_rhs(temp,voltage,time+gamma*dt,k1,cache,iter>0?&error:nullptr);')
cu=cu.replace('tissue_rhs(temp,voltage,time+(1.0-gamma)*dt,k2,cache);',
    'double error=0.;tissue_rhs(temp,voltage,time+(1.0-gamma)*dt,k2,cache,iter>0?&error:nullptr);')
cu=re.sub(r'if \(iter>0\) \{\s*double error=0\.0;\s*for \(int j=0;j<57;\+\+j\) error=fmax\(error,fabs\(k[12]\[j\]-previous\[j\]\)/fmax\(fabs\(k[12]\[j\]\),1\.0\)\);\s*if \(error<1e-6\) break;\s*\}',
    'if (iter>0 && error<1e-6) break;',cu)
assert 'previous[' not in cu
(out/'compact_tissue.cu').write_text(cu)
(out/'compact_tissue.cpp').write_text((out/'cached_tissue.cpp').read_text().replace('afsei_ep_cuda_cached','afsei_ep_cuda_compact'))
(out/'compact.py').write_text('''from pathlib import Path
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
def native():
 return load(name='afsei_compact_native_gpb',sources=[str(ROOT/'compact_reference.cpp'),str(ROOT/'compact_reference.cu')],extra_cflags=['-O3','-ffp-contract=off'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
def production():
 load(name='afsei_ep_cuda_compact',sources=[str(ROOT/'compact_tissue.cpp'),str(ROOT/'compact_tissue.cu')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],is_python_module=False,verbose=False)
 import torch
 return torch.ops.afsei_ep_cuda_compact
''')
print('Compact candidates generated: unchanged equations, FP64 and SDIRK.')
