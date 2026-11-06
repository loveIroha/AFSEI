#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <tuple>
using at::Tensor;
std::tuple<Tensor,Tensor,Tensor,Tensor> gpb_land_euler_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,double,double,int64_t,bool);
static std::tuple<Tensor,Tensor,Tensor,Tensor> advance(const Tensor& y,const Tensor& land,const Tensor& voltage,
 const Tensor& stretch,const Tensor& zs,const Tensor& zw,double t,double dt,int64_t steps,bool use_land) {
 for (const auto& a:{y,land,voltage,stretch,zs,zw}) {
  TORCH_CHECK(a.is_cuda() && a.device()==y.device() && a.scalar_type()==at::kDouble && a.is_contiguous() && !a.requires_grad(),"Euler GPB/Land requires matching contiguous gradient-free CUDA FP64 inputs");
 }
 TORCH_CHECK(y.dim()==2 && y.size(1)==57 && voltage.dim()==1 && voltage.size(0)==y.size(0),"GPB57 state/voltage shape mismatch");
 TORCH_CHECK(land.dim()==2 && land.size(0)==y.size(0) && land.size(1)==4,"Land requires (N,4)");
 for(const auto& a:{stretch,zs,zw})TORCH_CHECK(a.sizes()==voltage.sizes(),"Euler feedback shape mismatch");
 TORCH_CHECK(std::isfinite(t)&&t>=0&&std::isfinite(dt)&&dt>0&&steps>0,"Positive finite Euler step and nonnegative time required");
 return gpb_land_euler_cuda(y,land,voltage,stretch,zs,zw,t,dt,steps,use_land);
}
TORCH_LIBRARY(afsei_ep_euler,m) {
 m.def("advance(Tensor state, Tensor land, Tensor voltage_mv, Tensor stretch, Tensor zs, Tensor zw, float time_ms, float dt_ms, int substeps, bool advance_land) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(afsei_ep_euler,CUDA,m) {m.impl("advance",&advance);}
