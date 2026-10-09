#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <tuple>
#include "gpb_device.h"
#include "land_device.h"

using at::Tensor;

__device__ inline double tissue_rhs(const double* state, double voltage, double time, double* derivative) {
    double full[58], dfull[58];
    for (int j=0;j<38;++j) full[j]=state[j];
    full[38]=voltage;
    for (int j=38;j<57;++j) full[j+1]=state[j];
    double current=gpb_full_rhs(time,full,voltage,0.0,dfull);
    for (int j=0;j<38;++j) derivative[j]=dfull[j];
    for (int j=38;j<57;++j) derivative[j]=dfull[j+1];
    return current;
}

__global__ void rhs_kernel(const double* y,const double* voltage,double time,double* derivative,double* current,int64_t n) {
    auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=n) return;
    current[i]=tissue_rhs(y+57*i,voltage[i],time,derivative+57*i);
}

// Match the SOURCE'S actual two-stage fixed-point SDIRK arithmetic, iteration
// ceiling, relative error test and FINAL-state current (not a stage average).
__device__ inline double sdirk_step(double* state,double voltage,double time,double dt,int& guarded) {
    constexpr double gamma=0.788675134594812882254;
    constexpr double a21=1.0-2.0*gamma;
    double k1[57],k2[57],temp[57],previous[57];
    tissue_rhs(state,voltage,time,k1);
    for (int iter=0;iter<3;++iter) {
        for (int j=0;j<57;++j) { previous[j]=k1[j]; temp[j]=state[j]+dt*gamma*k1[j]; }
        tissue_rhs(temp,voltage,time+gamma*dt,k1);
        if (iter>0) {
            double error=0.0;
            for (int j=0;j<57;++j) error=fmax(error,fabs(k1[j]-previous[j])/fmax(fabs(k1[j]),1.0));
            if (error<1e-6) break;
        }
    }
    for (int j=0;j<57;++j) k2[j]=k1[j];
    for (int iter=0;iter<3;++iter) {
        for (int j=0;j<57;++j) { previous[j]=k2[j]; temp[j]=state[j]+dt*(a21*k1[j]+gamma*k2[j]); }
        tissue_rhs(temp,voltage,time+(1.0-gamma)*dt,k2);
        if (iter>0) {
            double error=0.0;
            for (int j=0;j<57;++j) error=fmax(error,fabs(k2[j]-previous[j])/fmax(fabs(k2[j]),1.0));
            if (error<1e-6) break;
        }
    }
    for (int j=0;j<57;++j) {
        state[j]+=dt*(0.5*k1[j]+0.5*k2[j]);
        if (!isfinite(state[j])) { state[j]=0.0; ++guarded; }
    }
    double current=tissue_rhs(state,voltage,time+dt,temp);
    if (!isfinite(current)) { current=0.0; ++guarded; }
    return current;
}

__global__ void advance_kernel(const double* y,const double* land,const double* voltage,
    const double* stretch,const double* zs,const double* zw,double time,double dt,int64_t steps,
    bool advance_land,double* out_y,double* out_land,double* current,int64_t* guards,int64_t n) {
    auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=n) return;
    double state[57];
    for (int j=0;j<57;++j) state[j]=y[57*i+j];
    LandState l;
    l.XS=land[4*i]; l.XW=land[4*i+1]; l.CaTrpn=land[4*i+2]; l.TmB=land[4*i+3];
    int guarded=0;
    double sum=0.0;
    for (int64_t k=0;k<steps;++k) {
        sum+=sdirk_step(state,voltage[i],time+k*dt,dt,guarded);
        if (advance_land) update_land_ode(l,state[37]*1000.0,stretch[i],zs[i],zw[i],dt);
    }
    for (int j=0;j<57;++j) out_y[57*i+j]=state[j];
    out_land[4*i]=l.XS; out_land[4*i+1]=l.XW; out_land[4*i+2]=l.CaTrpn; out_land[4*i+3]=l.TmB;
    current[i]=sum/steps;
    guards[i]=guarded;
}

std::tuple<Tensor,Tensor> gpb_rhs_cuda(const Tensor& y,const Tensor& voltage,double time) {
    const c10::cuda::CUDAGuard guard(y.device());
    auto dy=at::empty_like(y),current=at::empty_like(voltage);
    auto n=y.size(0);
    if (n) rhs_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<double>(),voltage.data_ptr<double>(),time,dy.data_ptr<double>(),current.data_ptr<double>(),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dy,current};
}

std::tuple<Tensor,Tensor,Tensor,Tensor> gpb_land_cuda(const Tensor& y,const Tensor& land,
    const Tensor& voltage,const Tensor& stretch,const Tensor& zs,const Tensor& zw,
    double time,double dt,int64_t steps,bool advance_land) {
    const c10::cuda::CUDAGuard guard(y.device());
    auto out_y=at::empty_like(y),out_land=at::empty_like(land),current=at::empty_like(voltage);
    auto guards=at::empty(voltage.sizes(),voltage.options().dtype(at::kLong));
    auto n=y.size(0);
    if (n) advance_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
        y.data_ptr<double>(),land.data_ptr<double>(),voltage.data_ptr<double>(),stretch.data_ptr<double>(),
        zs.data_ptr<double>(),zw.data_ptr<double>(),time,dt,steps,advance_land,out_y.data_ptr<double>(),
        out_land.data_ptr<double>(),current.data_ptr<double>(),guards.data_ptr<int64_t>(),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_y,out_land,current,guards};
}
