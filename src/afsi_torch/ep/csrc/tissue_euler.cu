#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <tuple>
#include "gpb_euler_device.h"
#include "land_device.h"
using at::Tensor;
__global__ void euler_kernel(const double* y,const double* land,const double* v,const double* stretch,
 const double* zs,const double* zw,double time,double dt,int64_t steps,bool use_land,
 double* oy,double* ol,double* oi,int64_t* guards,int64_t n) {
 auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 auto cache=make_gpb_voltage_cache(v[i]);double state[57],dy[57];
 for(int j=0;j<57;++j)state[j]=y[57*i+j];
 LandState l;l.XS=land[4*i];l.XW=land[4*i+1];l.CaTrpn=land[4*i+2];l.TmB=land[4*i+3];
 int bad=0;double sum=0.;
 gpb_nodal_rhs(time,state,v[i],0.,dy,cache);
 for(int64_t k=0;k<steps;++k) {
  for(int j=0;j<57;++j) {state[j]+=dt*dy[j];if(!isfinite(state[j]))++bad;}
  for(int j=30;j<38;++j)if(!(state[j]>0))++bad;
  double current=gpb_nodal_rhs(time+(k+1)*dt,state,v[i],0.,dy,cache);
  if(!isfinite(current))++bad;sum+=current;
  if(use_land)update_land_ode(l,state[37]*1000.,stretch[i],zs[i],zw[i],dt);
 }
 for(int j=0;j<57;++j)oy[57*i+j]=state[j];
 ol[4*i]=l.XS;ol[4*i+1]=l.XW;ol[4*i+2]=l.CaTrpn;ol[4*i+3]=l.TmB;
 oi[i]=sum/steps;guards[i]=bad;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> gpb_land_euler_cuda(const Tensor& y,const Tensor& land,const Tensor& v,
 const Tensor& stretch,const Tensor& zs,const Tensor& zw,double time,double dt,int64_t steps,bool use_land) {
 const c10::cuda::CUDAGuard guard(y.device());auto oy=at::empty_like(y),ol=at::empty_like(land),oi=at::empty_like(v);
 auto guards=at::empty(v.sizes(),v.options().dtype(at::kLong));auto n=y.size(0);
 if(n)euler_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(y.data_ptr<double>(),land.data_ptr<double>(),v.data_ptr<double>(),stretch.data_ptr<double>(),zs.data_ptr<double>(),zw.data_ptr<double>(),time,dt,steps,use_land,oy.data_ptr<double>(),ol.data_ptr<double>(),oi.data_ptr<double>(),guards.data_ptr<int64_t>(),n);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {oy,ol,oi,guards};
}
