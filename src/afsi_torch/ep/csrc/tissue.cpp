#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <tuple>

using at::Tensor;
std::tuple<Tensor,Tensor> gpb_rhs_cuda(const Tensor&,const Tensor&,double);
std::tuple<Tensor,Tensor,Tensor,Tensor> gpb_land_cuda(const Tensor&,const Tensor&,
    const Tensor&,const Tensor&,const Tensor&,const Tensor&,double,double,int64_t,bool);

static void check(const Tensor& t, const Tensor& like) {
    TORCH_CHECK(t.is_cuda() && t.device()==like.device(), "GPB/Land inputs must share a CUDA device");
    TORCH_CHECK(t.scalar_type()==at::kDouble && t.is_contiguous(), "GPB/Land requires contiguous float64 tensors");
    TORCH_CHECK(!t.requires_grad(), "GPB/Land ODE gradients are not registered");
}
static void check_state(const Tensor& y,const Tensor& voltage,double t) {
    check(y,y); check(voltage,y);
    TORCH_CHECK(y.dim()==2 && y.size(1)==57, "GPB state must have shape (N,57), excluding voltage");
    TORCH_CHECK(voltage.dim()==1 && voltage.size(0)==y.size(0), "voltage must have shape (N,)");
    TORCH_CHECK(std::isfinite(t) && t>=0, "time must be finite nonnegative milliseconds");
}
static std::tuple<Tensor,Tensor> rhs(const Tensor& y,const Tensor& voltage,double t) {
    check_state(y,voltage,t);
    return gpb_rhs_cuda(y,voltage,t);
}
static std::tuple<Tensor,Tensor,Tensor,Tensor> advance(const Tensor& y,const Tensor& land,
    const Tensor& voltage,const Tensor& stretch,const Tensor& zs,const Tensor& zw,
    double t,double dt,int64_t steps,bool advance_land) {
    check_state(y,voltage,t);
    check(land,y);
    TORCH_CHECK(land.dim()==2 && land.size(0)==y.size(0) && land.size(1)==4, "Land state must have shape (N,4)");
    for (const auto& value : {stretch,zs,zw}) {
        check(value,y);
        TORCH_CHECK(value.sizes()==voltage.sizes(), "feedback must have shape (N,)");
    }
    TORCH_CHECK(std::isfinite(dt) && dt>0 && steps>0, "positive ODE milliseconds and substep count required");
    return gpb_land_cuda(y,land,voltage,stretch,zs,zw,t,dt,steps,advance_land);
}
TORCH_LIBRARY(afsei_ep_cuda,m) {
    m.def("rhs(Tensor state, Tensor voltage_mv, float time_ms) -> (Tensor, Tensor)");
    m.def("advance(Tensor state, Tensor land, Tensor voltage_mv, Tensor stretch, Tensor zs, Tensor zw, float time_ms, float dt_ms, int substeps, bool advance_land) -> (Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(afsei_ep_cuda,CUDA,m) {
    m.impl("rhs",&rhs);
    m.impl("advance",&advance);
}
