from pathlib import Path
import re,json
root=Path('/mnt/large2/gjh/AFSEI');out=root/'demo/cell_optimization_20261030'
source=(root/'src/afsi_torch/ep/csrc/gpb_device.h').read_text()
assert len(re.findall(r'\bt\b',re.sub(r'//[^\n]*','',source)))==1,'Endpoint reuse requires an autonomous GPB RHS'
cu=(out/'compact_reference.cu').read_text()
old='for(int sub=0;sub<steps;++sub) {\n    compact_grandi_sdirk::rhs(state,V,k1,cache);'
assert old in cu
cu=cu.replace(old,'compact_grandi_sdirk::rhs(state,V,k1,cache);\n  for(int sub=0;sub<steps;++sub) {')
cu=cu.replace('rhs(state,V,temp,cache)','rhs(state,V,k1,cache)')
cu+='''
__global__ void euler_kernel(const double* y,const double* V,double* z,double* I,int64_t* guards,int64_t n,double dt,int steps) {
 auto id=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=n)return;
 double s[NS],d[NS];for(int j=0;j<NS;++j)s[j]=y[j*n+id];
 auto cache=compact_grandi_sdirk::make_voltage_cache(V[id]);
 compact_grandi_sdirk::rhs(s,V[id],d,cache);double sum=0.;int bad=0;
 for(int sub=0;sub<steps;++sub) {
   for(int j=0;j<NS;++j){s[j]+=dt*d[j];if(!isfinite(s[j])){s[j]=0.;++bad;}}
   double cur=compact_grandi_sdirk::rhs(s,V[id],d,cache);
   if(!isfinite(cur)){cur=0.;++bad;}sum+=cur;
 }
 POSITIVE_CHECKS
 for(int j=0;j<NS;++j)z[j*n+id]=s[j];I[id]=sum/steps;guards[id]=bad;
}
void experimental_euler(Tensor y,Tensor V,Tensor z,Tensor I,Tensor guard,double dt,int64_t steps) {
 const c10::cuda::CUDAGuard g(V.device());auto n=V.numel();
 TORCH_CHECK(y.is_cuda() && z.is_cuda() && I.is_cuda() && guard.is_cuda() && y.device()==V.device() && z.device()==V.device() && I.device()==V.device() && guard.device()==V.device(),"Euler fields must share a CUDA device");
 TORCH_CHECK(y.scalar_type()==at::kDouble && V.scalar_type()==at::kDouble && z.scalar_type()==at::kDouble && I.scalar_type()==at::kDouble && guard.scalar_type()==at::kLong,"Euler requires FP64 state/voltage/current and int64 guards");
 TORCH_CHECK(y.dim()==2 && y.size(0)==NS && y.size(1)==n && z.sizes()==y.sizes() && I.numel()==n && guard.numel()==n && n>0 && dt>0 && std::isfinite(dt) && steps>0,"invalid Euler shape or step");
 TORCH_CHECK(y.is_contiguous() && V.is_contiguous() && z.is_contiguous() && I.is_contiguous() && guard.is_contiguous(),"Euler requires contiguous buffers");
 euler_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(y.data_ptr<double>(),V.data_ptr<double>(),z.data_ptr<double>(),I.data_ptr<double>(),guard.data_ptr<int64_t>(),n,dt,steps);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
names=json.loads((root/'demo/ep_afsei_nodal_opencarp_matched/reference_cell_provenance.json').read_text())['states']
positive=[names.index(name) for name in ['Ca_sr','Caj','Casl','Cai','Naj','Nasl','Nai','Ki']]
cu=cu.replace('POSITIVE_CHECKS','\n'.join(f'if(s[{j}]<=0.)++bad;' for j in positive))
(out/'fused_reference.cu').write_text(cu)
cpp=(out/'compact_reference.cpp').read_text().replace('PYBIND11_MODULE',
    'void experimental_euler(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,int64_t);\nPYBIND11_MODULE')
cpp=cpp.replace('m.def("rhs",','m.def("euler",&experimental_euler);m.def("rhs",')
(out/'fused_reference.cpp').write_text(cpp)
cu=(out/'compact_tissue.cu').read_text()
cu=cu.replace('int& guarded,const GPBVoltageCache& cache)', 'int& guarded,double* k1,const GPBVoltageCache& cache)')
cu=cu.replace('double k1[57],k2[57],temp[57];','double k2[57],temp[57];')
cu=cu.replace('    tissue_rhs(state,voltage,time,k1,cache);','')
cu=cu.replace('time+dt,temp,cache);','time+dt,k1,cache);')
cu=cu.replace('double sum=0.0;','double sum=0.0;\n    double k1[57];tissue_rhs(state,voltage[i],time,k1,cache);')
cu=cu.replace('dt,guarded,cache);','dt,guarded,k1,cache);')
(out/'fused_tissue.cu').write_text(cu)
(out/'fused_tissue.cpp').write_text((out/'compact_tissue.cpp').read_text().replace('afsei_ep_cuda_compact','afsei_ep_cuda_fused'))
(out/'fused.py').write_text('''from pathlib import Path
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
def native():
 return load(name='afsei_fused_native_gpb',sources=[str(ROOT/'fused_reference.cpp'),str(ROOT/'fused_reference.cu')],extra_cflags=['-O3','-ffp-contract=off'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
def production():
 load(name='afsei_ep_cuda_fused',sources=[str(ROOT/'fused_tissue.cpp'),str(ROOT/'fused_tissue.cu')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],is_python_module=False,verbose=False)
 import torch
 return torch.ops.afsei_ep_cuda_fused
''')
print('Fused SDIRK endpoint reuse generated; Euler is an experimental diagnostic entry only.')
