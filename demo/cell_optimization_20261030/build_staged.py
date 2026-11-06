"""GPU-resident staged execution of the same per-node SDIRK algorithm."""
from pathlib import Path
import re,json
root=Path('/mnt/large2/gjh/AFSEI');out=root/'demo/cell_optimization_20261030'
specs=[]
for native,label,source,space,cache,make,fn in [
 (True,'Native','compact_grandi_rhs.h','staged_grandi_sdirk::','GrandiVoltageCache','make_voltage_cache','rhs'),
 (False,'Production','compact_gpb_rhs.h','','GPBVoltageCache','make_gpb_voltage_cache','gpb_nodal_rhs')]:
    h=(out/source).read_text().replace('compact_grandi_sdirk','staged_grandi_sdirk')
    h=h.replace('double* relative_error=nullptr)', 'int64_t stride,double* relative_error=nullptr)')
    # All ionic input/output arrays are SoA. No local per-thread state arrays.
    h=re.sub(r'\b(y|dy|ydot)\[(\d+)\]',lambda m:f'{m[1]}[{m[2]}*stride]',h)
    # Aliasing never occurs: stage input, state and derivative workspaces differ.
    h=h.replace('const double* y,','const double* __restrict__ y,')
    h=h.replace('double* dy,','double* __restrict__ dy,').replace('double* ydot,','double* __restrict__ ydot,')
    filename='staged_'+source
    (out/filename).write_text(h)
    meta=json.loads((out/('cached_grandi_rhs.json' if native else 'cached_gpb_rhs.json')).read_text())
    fields=meta['cache_fields'];NS=54 if native else 57
    arguments='y,V,d,c,n,e' if native else 'time,y,V,0.,d,c,n,e'
    model=f'''struct {label} {{
 static constexpr int states={NS}, coefficients={len(fields)};
 __device__ static void prepare(double V,double* cache,int64_t n) {{
 auto c={space}{make}(V);
'''+''.join(f' cache[{j}*n]=c.{name};\n' for j,name in enumerate(fields))+'''
 }
 __device__ static double rhs(const double* y,double V,double time,double* d,const double* cache,int64_t n,double* e) {
'''+f'{space}{cache} c;\n'+''.join(f' c.{name}=cache[{j}*n];\n' for j,name in enumerate(fields))+f' return {space}{fn}({arguments});\n }}\n}};\n'
    specs.append(model)
cu='''#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include "staged_compact_grandi_rhs.h"
#include "staged_compact_gpb_rhs.h"
'''+f'#include "{root}/src/afsi_torch/ep/csrc/land_device.h"\n'+''.join(specs)+'''
using at::Tensor;
template<class M> __global__ void prepare_cache(const double* V,double* c,int64_t n) {
 auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)M::prepare(V[i],c+i,n);
}
template<class M> __global__ void prepare_stage(const double* state,double* k1,double* k2,double* temp,
 bool* active,int64_t n,double dt,int stage,int iter) {
 auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 if(iter==0)active[i]=true;else if(!active[i])return;
 constexpr double gamma=.788675134594812882254;constexpr double a21=1.-2.*gamma;
 for(int j=0;j<M::states;++j) {
   auto p=j*n+i;
   if(stage==0)temp[p]=state[p]+dt*gamma*k1[p];
   else {if(iter==0)k2[p]=k1[p];temp[p]=state[p]+dt*(a21*k1[p]+gamma*k2[p]);}
 }
}
template<class M> __global__ void stage_rhs(const double* temp,const double* V,double time,double* k,
 const double* cache,bool* active,int64_t n,int iter) {
 auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n||!active[i])return;
 double error=0.;M::rhs(temp+i,V[i],time,k+i,cache+i,n,iter>0?&error:nullptr);
 if(iter>0)active[i]=!(error<1e-6);
}
template<class M> __global__ void update_state(double* state,const double* k1,const double* k2,
 double* land,const double* stretch,const double* zs,const double* zw,int64_t* guards,
 int64_t n,double dt,bool use_land) {
 auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 int bad=0;
 for(int j=0;j<M::states;++j) {auto p=j*n+i;
   state[p]+=dt*(.5*k1[p]+.5*k2[p]);if(!isfinite(state[p])){state[p]=0.;++bad;}
 }
 if(use_land) {LandState l;l.XS=land[4*i];l.XW=land[4*i+1];l.CaTrpn=land[4*i+2];l.TmB=land[4*i+3];
   update_land_ode(l,state[37*n+i]*1000.,stretch[i],zs[i],zw[i],dt);
   land[4*i]=l.XS;land[4*i+1]=l.XW;land[4*i+2]=l.CaTrpn;land[4*i+3]=l.TmB;
 }
 guards[i]+=bad;
}
template<class M> __global__ void endpoint_rhs(const double* state,const double* V,double time,
 double* k1,const double* cache,double* sum,int64_t* guards,int64_t n,bool add_current) {
 auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 double I=M::rhs(state+i,V[i],time,k1+i,cache+i,n,nullptr);
 if(add_current){if(!isfinite(I)){I=0.;++guards[i];}sum[i]+=I;}
}
template<class M> void execute(Tensor state,Tensor V,Tensor cache,Tensor k1,Tensor k2,Tensor temp,
 Tensor active,Tensor land,Tensor stretch,Tensor zs,Tensor zw,Tensor sum,Tensor guards,
 double time,double dt,int64_t steps,bool use_land) {
 const c10::cuda::CUDAGuard guard(V.device());auto stream=at::cuda::getCurrentCUDAStream();
 auto n=V.numel();constexpr int block=128;int grid=(n+block-1)/block;
 auto s=state.data_ptr<double>();auto v=V.data_ptr<double>();auto c=cache.data_ptr<double>();
 auto a=k1.data_ptr<double>();auto b=k2.data_ptr<double>();auto t=temp.data_ptr<double>();
 auto flags=active.data_ptr<bool>();auto total=sum.data_ptr<double>();auto g=guards.data_ptr<int64_t>();
 prepare_cache<M><<<grid,block,0,stream>>>(v,c,n);
 endpoint_rhs<M><<<grid,block,0,stream>>>(s,v,time,a,c,total,g,n,false);
 constexpr double gamma=.788675134594812882254;
 for(int64_t sub=0;sub<steps;++sub) {
   for(int stage=0;stage<2;++stage)for(int iter=0;iter<3;++iter) {
     prepare_stage<M><<<grid,block,0,stream>>>(s,a,b,t,flags,n,dt,stage,iter);
     stage_rhs<M><<<grid,block,0,stream>>>(t,v,time+sub*dt+(stage==0?gamma:1.-gamma)*dt,
       stage==0?a:b,c,flags,n,iter);
   }
   update_state<M><<<grid,block,0,stream>>>(s,a,b,land.data_ptr<double>(),stretch.data_ptr<double>(),
     zs.data_ptr<double>(),zw.data_ptr<double>(),g,n,dt,use_land);
   // The GPB RHS is autonomous. Its final derivative is exactly the next
   // substep's initial derivative, so reuse it instead of reevaluating.
   endpoint_rhs<M><<<grid,block,0,stream>>>(s,v,time+(sub+1)*dt,a,c,total,g,n,true);
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void staged_advance(Tensor state,Tensor V,Tensor cache,Tensor k1,Tensor k2,Tensor temp,
 Tensor active,Tensor land,Tensor stretch,Tensor zs,Tensor zw,Tensor sum,Tensor guards,
 double time,double dt,int64_t steps,bool use_land,bool native) {
 auto n=V.numel();int ns=native?Native::states:Production::states;
 TORCH_CHECK(n>0 && V.is_cuda() && V.scalar_type()==at::kDouble && state.size(0)==ns && state.size(1)==n,"invalid staged state");
 TORCH_CHECK(steps>0 && dt>0,"positive dt and substeps required");
 if(native)execute<Native>(state,V,cache,k1,k2,temp,active,land,stretch,zs,zw,sum,guards,time,dt,steps,false);
 else execute<Production>(state,V,cache,k1,k2,temp,active,land,stretch,zs,zw,sum,guards,time,dt,steps,use_land);
}
'''
(out/'staged.cu').write_text(cu)
(out/'staged.cpp').write_text('''#include <torch/extension.h>
void staged_advance(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,double,int64_t,bool,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("advance",&staged_advance);}
''')
(out/'staged_metadata.json').write_text(json.dumps(dict(native_cache_fields=json.loads((out/'cached_grandi_rhs.json').read_text())['cache_count'],
    production_cache_fields=json.loads((out/'cached_gpb_rhs.json').read_text())['cache_count'],
    autonomous_rhs_reuse=True,per_node_independent_iteration=True,method_changed=False),indent=2))
print('Staged kernels generated; all stages and current averages retain original SDIRK arithmetic.')
