#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include "compact_grandi_rhs.h"
#include "/mnt/large2/gjh/AFSEI/demo/ep_afsei_nodal_opencarp_matched/reference_grandi_rl_check.h"
using at::Tensor;
constexpr int NS=54;
__host__ __device__ inline double advance_one(double* state,double V,double dt,int steps,int& bad) {
  constexpr double gamma=0.788675134594812882254;constexpr double a21=1.-2.*gamma;
  auto cache=compact_grandi_sdirk::make_voltage_cache(V);
  double k1[NS],k2[NS],temp[NS];double sum=0.;
  compact_grandi_sdirk::rhs(state,V,k1,cache);
  for(int sub=0;sub<steps;++sub) {
    for(int iter=0;iter<3;++iter) {
      for(int j=0;j<NS;++j){temp[j]=state[j]+dt*gamma*k1[j];}
      double error=0.;compact_grandi_sdirk::rhs(temp,V,k1,cache,iter>0?&error:nullptr);
      if(iter>0 && error<1e-6)break;
    }
    for(int j=0;j<NS;++j)k2[j]=k1[j];
    for(int iter=0;iter<3;++iter) {
      for(int j=0;j<NS;++j){temp[j]=state[j]+dt*(a21*k1[j]+gamma*k2[j]);}
      double error=0.;compact_grandi_sdirk::rhs(temp,V,k2,cache,iter>0?&error:nullptr);
      if(iter>0 && error<1e-6)break;
    }
    for(int j=0;j<NS;++j){state[j]+=dt*(.5*k1[j]+.5*k2[j]);if(!isfinite(state[j])){state[j]=0.;++bad;}}
    double I=compact_grandi_sdirk::rhs(state,V,k1,cache);if(!isfinite(I)){I=0.;++bad;}sum+=I;
  }
  if(state[16]<=0.)++bad;
  if(state[18]<=0.)++bad;
  if(state[19]<=0.)++bad;
  if(state[17]<=0.)++bad;
  if(state[27]<=0.)++bad;
  if(state[28]<=0.)++bad;
  if(state[26]<=0.)++bad;
  if(state[21]<=0.)++bad;

  return sum/steps;
}
__global__ void rhs_kernel(const double* y,const double* V,double* z,double* I,int64_t n) {
 auto id=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=n)return;double s[NS],d[NS];auto cache=compact_grandi_sdirk::make_voltage_cache(V[id]);
 for(int j=0;j<NS;++j)s[j]=y[j*n+id];I[id]=compact_grandi_sdirk::rhs(s,V[id],d,cache);
 for(int j=0;j<NS;++j)z[j*n+id]=d[j];
}
__global__ void advance_kernel(const double* y,const double* V,double* z,double* I,int64_t* guards,int64_t n,double dt,int steps) {
 auto id=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=n)return;double s[NS];
 for(int j=0;j<NS;++j)s[j]=y[j*n+id];int bad=0;I[id]=advance_one(s,V[id],dt,steps,bad);guards[id]=bad;
 for(int j=0;j<NS;++j)z[j*n+id]=s[j];
}
void reference_rhs(Tensor y,Tensor V,Tensor z,Tensor I) {
 auto n=V.numel();TORCH_CHECK(y.numel()==n*NS && z.numel()==n*NS,"state shape mismatch");
 if(V.is_cuda()){const c10::cuda::CUDAGuard g(V.device());rhs_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(y.data_ptr<double>(),V.data_ptr<double>(),z.data_ptr<double>(),I.data_ptr<double>(),n);C10_CUDA_KERNEL_LAUNCH_CHECK();}
 else {auto* a=y.data_ptr<double>();auto* v=V.data_ptr<double>();auto* b=z.data_ptr<double>();auto* c=I.data_ptr<double>();
 for(int64_t id=0;id<n;++id){double s[NS],d[NS];auto cache=compact_grandi_sdirk::make_voltage_cache(v[id]);for(int j=0;j<NS;++j)s[j]=a[j*n+id];c[id]=compact_grandi_sdirk::rhs(s,v[id],d,cache);for(int j=0;j<NS;++j)b[j*n+id]=d[j];}}
}
void reference_advance(Tensor y,Tensor V,Tensor z,Tensor I,Tensor guard,double dt,int64_t steps) {
 auto n=V.numel();TORCH_CHECK(y.numel()==n*NS && z.numel()==n*NS && steps>0,"state shape or steps mismatch");
 if(V.is_cuda()){const c10::cuda::CUDAGuard g(V.device());advance_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(y.data_ptr<double>(),V.data_ptr<double>(),z.data_ptr<double>(),I.data_ptr<double>(),guard.data_ptr<int64_t>(),n,dt,steps);C10_CUDA_KERNEL_LAUNCH_CHECK();}
 else {auto* a=y.data_ptr<double>();auto* v=V.data_ptr<double>();auto* b=z.data_ptr<double>();auto* c=I.data_ptr<double>();auto* g=guard.data_ptr<int64_t>();
 for(int64_t id=0;id<n;++id){double s[NS];for(int j=0;j<NS;++j)s[j]=a[j*n+id];int bad=0;c[id]=advance_one(s,v[id],dt,steps,bad);g[id]=bad;for(int j=0;j<NS;++j)b[j*n+id]=s[j];}}
}
void reference_rl_check(Tensor y,Tensor V,Tensor z,Tensor I,double dt) {
 TORCH_CHECK(!V.is_cuda(),"Independent RL derivative check is CPU-only");
 auto n=V.numel();auto* a=y.data_ptr<double>();auto* v=V.data_ptr<double>();auto* b=z.data_ptr<double>();auto* c=I.data_ptr<double>();
 for(int64_t id=0;id<n;++id){reference_grandi_double::Grandi_state s{};
 s.C1=a[0*n+id];
 s.C10=a[1*n+id];
 s.C11=a[2*n+id];
 s.C12=a[3*n+id];
 s.C13=a[4*n+id];
 s.C14=a[5*n+id];
 s.C15=a[6*n+id];
 s.C2=a[7*n+id];
 s.C3=a[8*n+id];
 s.C4=a[9*n+id];
 s.C5=a[10*n+id];
 s.C6=a[11*n+id];
 s.C7=a[12*n+id];
 s.C8=a[13*n+id];
 s.C9=a[14*n+id];
 s.CaM=a[15*n+id];
 s.Ca_sr=a[16*n+id];
 s.Cai=a[17*n+id]*1000.;
 s.Caj=a[18*n+id];
 s.Casl=a[19*n+id];
 s.Csqnb=a[20*n+id];
 s.Ki=a[21*n+id];
 s.Myoc=a[22*n+id];
 s.Myom=a[23*n+id];
 s.NaBj=a[24*n+id];
 s.NaBsl=a[25*n+id];
 s.Nai=a[26*n+id];
 s.Naj=a[27*n+id];
 s.Nasl=a[28*n+id];
 s.O1=a[29*n+id];
 s.RyRi=a[30*n+id];
 s.RyRo=a[31*n+id];
 s.RyRr=a[32*n+id];
 s.SLHj=a[33*n+id];
 s.SLHsl=a[34*n+id];
 s.SLLj=a[35*n+id];
 s.SLLsl=a[36*n+id];
 s.SRB=a[37*n+id];
 s.TnCHc=a[38*n+id];
 s.TnCHm=a[39*n+id];
 s.TnCL=a[40*n+id];
 s.d=a[41*n+id];
 s.f=a[42*n+id];
 s.fcaBj=a[43*n+id];
 s.fcaBsl=a[44*n+id];
 s.h=a[45*n+id];
 s.j=a[46*n+id];
 s.m=a[47*n+id];
 s.xkr=a[48*n+id];
 s.xks=a[49*n+id];
 s.xtof=a[50*n+id];
 s.xtos=a[51*n+id];
 s.ytof=a[52*n+id];
 s.ytos=a[53*n+id];
 int g=0;c[id]=reference_grandi_double::step(&s,v[id],dt,g);
 b[0*n+id]=s.C1;
 b[1*n+id]=s.C10;
 b[2*n+id]=s.C11;
 b[3*n+id]=s.C12;
 b[4*n+id]=s.C13;
 b[5*n+id]=s.C14;
 b[6*n+id]=s.C15;
 b[7*n+id]=s.C2;
 b[8*n+id]=s.C3;
 b[9*n+id]=s.C4;
 b[10*n+id]=s.C5;
 b[11*n+id]=s.C6;
 b[12*n+id]=s.C7;
 b[13*n+id]=s.C8;
 b[14*n+id]=s.C9;
 b[15*n+id]=s.CaM;
 b[16*n+id]=s.Ca_sr;
 b[17*n+id]=s.Cai/1000.;
 b[18*n+id]=s.Caj;
 b[19*n+id]=s.Casl;
 b[20*n+id]=s.Csqnb;
 b[21*n+id]=s.Ki;
 b[22*n+id]=s.Myoc;
 b[23*n+id]=s.Myom;
 b[24*n+id]=s.NaBj;
 b[25*n+id]=s.NaBsl;
 b[26*n+id]=s.Nai;
 b[27*n+id]=s.Naj;
 b[28*n+id]=s.Nasl;
 b[29*n+id]=s.O1;
 b[30*n+id]=s.RyRi;
 b[31*n+id]=s.RyRo;
 b[32*n+id]=s.RyRr;
 b[33*n+id]=s.SLHj;
 b[34*n+id]=s.SLHsl;
 b[35*n+id]=s.SLLj;
 b[36*n+id]=s.SLLsl;
 b[37*n+id]=s.SRB;
 b[38*n+id]=s.TnCHc;
 b[39*n+id]=s.TnCHm;
 b[40*n+id]=s.TnCL;
 b[41*n+id]=s.d;
 b[42*n+id]=s.f;
 b[43*n+id]=s.fcaBj;
 b[44*n+id]=s.fcaBsl;
 b[45*n+id]=s.h;
 b[46*n+id]=s.j;
 b[47*n+id]=s.m;
 b[48*n+id]=s.xkr;
 b[49*n+id]=s.xks;
 b[50*n+id]=s.xtof;
 b[51*n+id]=s.xtos;
 b[52*n+id]=s.ytof;
 b[53*n+id]=s.ytos;
 }
}

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
 if(s[16]<=0.)++bad;
if(s[18]<=0.)++bad;
if(s[19]<=0.)++bad;
if(s[17]<=0.)++bad;
if(s[27]<=0.)++bad;
if(s[28]<=0.)++bad;
if(s[26]<=0.)++bad;
if(s[21]<=0.)++bad;
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
