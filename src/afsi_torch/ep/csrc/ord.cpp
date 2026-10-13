#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include <tuple>
#include "ord_device.h"
using at::Tensor;
std::tuple<Tensor,Tensor> ord_rhs_cuda(const Tensor&,const Tensor&,const Tensor&);
std::tuple<Tensor,Tensor,Tensor> ord_advance_cuda(const Tensor&,const Tensor&,const Tensor&,double,int64_t);

static std::tuple<Tensor,Tensor> initialize(int64_t celltype) {
    TORCH_CHECK(celltype==0 || celltype==1,"ORd supports endocardial=0 or epicardial=1");
    auto options=at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
    auto constants=at::empty({139},options), state=at::empty({41},options);
    ord_init(constants.data_ptr<double>(),state.data_ptr<double>(),celltype);
    return {constants,state};
}
static void check(const Tensor& y,const Tensor& v,const Tensor& c) {
    TORCH_CHECK(y.is_cuda() && v.device()==y.device() && c.device()==y.device(),"ORd requires one CUDA device");
    for (const auto& t:{y,v,c}) {
        TORCH_CHECK(t.scalar_type()==at::kDouble && t.is_contiguous() && !t.requires_grad(),
                    "ORd requires contiguous gradient-free FP64 tensors");
    }
    TORCH_CHECK(y.dim()==2 && y.size(1)==40 && y.size(0)>0,"ORd ionic state must have shape (N,40)");
    TORCH_CHECK(v.dim()==1 && v.size(0)==y.size(0),"ORd voltage must have shape (N)");
    TORCH_CHECK(c.dim()==1 && c.size(0)==139,"ORd constants must have shape (139)");
}
static std::tuple<Tensor,Tensor> rhs(const Tensor& y,const Tensor& v,const Tensor& c) {
    check(y,v,c); return ord_rhs_cuda(y,v,c);
}
static std::tuple<Tensor,Tensor,Tensor> advance(const Tensor& y,const Tensor& v,const Tensor& c,double dt,int64_t steps) {
    check(y,v,c);
    TORCH_CHECK(std::isfinite(dt) && dt>0 && steps>0,"positive ODE dt and substeps required");
    return ord_advance_cuda(y,v,c,dt,steps);
}
TORCH_LIBRARY(afsei_ord_cuda,m) {
    m.def("initialize(int celltype) -> (Tensor, Tensor)",&initialize);
    m.def("rhs(Tensor state, Tensor voltage_mv, Tensor constants) -> (Tensor, Tensor)");
    m.def("advance(Tensor state, Tensor voltage_mv, Tensor constants, float dt_ms, int substeps) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(afsei_ord_cuda,CUDA,m) { m.impl("rhs",&rhs); m.impl("advance",&advance); }
