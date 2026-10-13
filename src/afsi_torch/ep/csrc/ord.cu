#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <tuple>
#include "ord_device.h"
using at::Tensor;

// Same two-stage SDIRK coefficients, three fixed-point iterations, stopping
// criterion and mean-of-substep-final-currents as the existing GPB kernel.
__device__ inline double step(double* y,double v,const double* c,double dt,int64_t& guards) {
    constexpr double gamma=.788675134594812882254,a21=1-2*gamma;
    double k1[40],k2[40],temp[40],previous[40];
    ord_ionic_rhs(y,v,c,k1);
    for (int iter=0;iter<3;++iter) {
        for (int j=0;j<40;++j) { previous[j]=k1[j];temp[j]=y[j]+dt*gamma*k1[j]; }
        ord_ionic_rhs(temp,v,c,k1);
        if (iter>0) {
            double error=0;
            for (int j=0;j<40;++j) error=fmax(error,fabs(k1[j]-previous[j])/fmax(fabs(k1[j]),1.));
            if (error<1e-6) break;
        }
    }
    for (int j=0;j<40;++j) k2[j]=k1[j];
    for (int iter=0;iter<3;++iter) {
        for (int j=0;j<40;++j) { previous[j]=k2[j];temp[j]=y[j]+dt*(a21*k1[j]+gamma*k2[j]); }
        ord_ionic_rhs(temp,v,c,k2);
        if (iter>0) {
            double error=0;
            for (int j=0;j<40;++j) error=fmax(error,fabs(k2[j]-previous[j])/fmax(fabs(k2[j]),1.));
            if (error<1e-6) break;
        }
    }
    for (int j=0;j<40;++j) { y[j]+=dt*.5*(k1[j]+k2[j]); if (!isfinite(y[j])) ++guards; }
    double current=ord_ionic_rhs(y,v,c,temp);
    if (!isfinite(current)) ++guards;
    return current;
}
__global__ void rhs_kernel(const double* y,const double* v,const double* c,double* dy,double* current,int64_t n) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<n) current[i]=ord_ionic_rhs(y+40*i,v[i],c,dy+40*i);
}
__global__ void advance_kernel(const double* y,const double* v,const double* c,double dt,int64_t steps,
                               double* output,double* current,int64_t* guards,int64_t n) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=n) return;
    double state[40]; for (int j=0;j<40;++j) state[j]=y[40*i+j];
    double sum=0; int64_t guarded=0;
    for (int64_t k=0;k<steps;++k) sum+=step(state,v[i],c,dt,guarded);
    for (int j=0;j<40;++j) output[40*i+j]=state[j];
    current[i]=sum/steps; guards[i]=guarded;
}
std::tuple<Tensor,Tensor> ord_rhs_cuda(const Tensor& y,const Tensor& v,const Tensor& c) {
    const c10::cuda::CUDAGuard guard(y.device());
    auto dy=at::empty_like(y),current=at::empty_like(v); auto n=y.size(0);
    rhs_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<double>(),v.data_ptr<double>(),c.data_ptr<double>(),dy.data_ptr<double>(),current.data_ptr<double>(),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return {dy,current};
}
std::tuple<Tensor,Tensor,Tensor> ord_advance_cuda(const Tensor& y,const Tensor& v,const Tensor& c,double dt,int64_t steps) {
    const c10::cuda::CUDAGuard guard(y.device());
    auto output=at::empty_like(y),current=at::empty_like(v),guards=at::empty(v.sizes(),v.options().dtype(at::kLong));
    auto n=y.size(0);
    advance_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<double>(),v.data_ptr<double>(),c.data_ptr<double>(),dt,steps,
        output.data_ptr<double>(),current.data_ptr<double>(),guards.data_ptr<int64_t>(),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return {output,current,guards};
}
