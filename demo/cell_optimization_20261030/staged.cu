#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include "staged_compact_grandi_rhs.h"
#include "staged_compact_gpb_rhs.h"
#include "/mnt/large2/gjh/AFSEI/src/afsi_torch/ep/csrc/land_device.h"
struct Native {
 static constexpr int states=54, coefficients=30;
 __device__ static void prepare(double V,double* cache,int64_t n) {
 auto c=staged_grandi_sdirk::make_voltage_cache(V);
 cache[0*n]=c.rkr;
 cache[1*n]=c.kp_kp;
 cache[2*n]=c.d_infinity;
 cache[3*n]=c.f_infinity;
 cache[4*n]=c.tau_f;
 cache[5*n]=c.h_infinity;
 cache[6*n]=c.tau_h;
 cache[7*n]=c.j_infinity;
 cache[8*n]=c.tau_j;
 cache[9*n]=c.m_infinity;
 cache[10*n]=c.tau_m;
 cache[11*n]=c.xkr_infinity;
 cache[12*n]=c.tau_xkr;
 cache[13*n]=c.xks_infinity;
 cache[14*n]=c.tau_xks;
 cache[15*n]=c.xtos_infinity;
 cache[16*n]=c.xtof_infinity;
 cache[17*n]=c.tau_xtof;
 cache[18*n]=c.tau_xtos;
 cache[19*n]=c.ytos_infinity;
 cache[20*n]=c.ytof_infinity;
 cache[21*n]=c.tau_ytof;
 cache[22*n]=c.tau_ytos;
 cache[23*n]=c.math_0;
 cache[24*n]=c.math_1;
 cache[25*n]=c.math_2;
 cache[26*n]=c.math_3;
 cache[27*n]=c.math_4;
 cache[28*n]=c.math_5;
 cache[29*n]=c.math_6;

 }
 __device__ static double rhs(const double* y,double V,double time,double* d,const double* cache,int64_t n,double* e) {
staged_grandi_sdirk::GrandiVoltageCache c;
 c.rkr=cache[0*n];
 c.kp_kp=cache[1*n];
 c.d_infinity=cache[2*n];
 c.f_infinity=cache[3*n];
 c.tau_f=cache[4*n];
 c.h_infinity=cache[5*n];
 c.tau_h=cache[6*n];
 c.j_infinity=cache[7*n];
 c.tau_j=cache[8*n];
 c.m_infinity=cache[9*n];
 c.tau_m=cache[10*n];
 c.xkr_infinity=cache[11*n];
 c.tau_xkr=cache[12*n];
 c.xks_infinity=cache[13*n];
 c.tau_xks=cache[14*n];
 c.xtos_infinity=cache[15*n];
 c.xtof_infinity=cache[16*n];
 c.tau_xtof=cache[17*n];
 c.tau_xtos=cache[18*n];
 c.ytos_infinity=cache[19*n];
 c.ytof_infinity=cache[20*n];
 c.tau_ytof=cache[21*n];
 c.tau_ytos=cache[22*n];
 c.math_0=cache[23*n];
 c.math_1=cache[24*n];
 c.math_2=cache[25*n];
 c.math_3=cache[26*n];
 c.math_4=cache[27*n];
 c.math_5=cache[28*n];
 c.math_6=cache[29*n];
 return staged_grandi_sdirk::rhs(y,V,d,c,n,e);
 }
};
struct Production {
 static constexpr int states=57, coefficients=31;
 __device__ static void prepare(double V,double* cache,int64_t n) {
 auto c=make_gpb_voltage_cache(V);
 cache[0*n]=c.mss;
 cache[1*n]=c.taum;
 cache[2*n]=c.tauh;
 cache[3*n]=c.hss;
 cache[4*n]=c.tauj;
 cache[5*n]=c.jss;
 cache[6*n]=c.a_na_L;
 cache[7*n]=c.b_na_L;
 cache[8*n]=c.h_na_L;
 cache[9*n]=c.fnak;
 cache[10*n]=c.xrss;
 cache[11*n]=c.tauxr;
 cache[12*n]=c.rkr;
 cache[13*n]=c.xsss;
 cache[14*n]=c.tauxs;
 cache[15*n]=c.kp_kp;
 cache[16*n]=c.xtoss;
 cache[17*n]=c.ytoss;
 cache[18*n]=c.tauxtos;
 cache[19*n]=c.tauytos;
 cache[20*n]=c.tauxtof;
 cache[21*n]=c.tauytof;
 cache[22*n]=c.I_Clbk;
 cache[23*n]=c.dss;
 cache[24*n]=c.taud;
 cache[25*n]=c.fss;
 cache[26*n]=c.tauf;
 cache[27*n]=c.math_0;
 cache[28*n]=c.math_1;
 cache[29*n]=c.math_2;
 cache[30*n]=c.math_3;

 }
 __device__ static double rhs(const double* y,double V,double time,double* d,const double* cache,int64_t n,double* e) {
GPBVoltageCache c;
 c.mss=cache[0*n];
 c.taum=cache[1*n];
 c.tauh=cache[2*n];
 c.hss=cache[3*n];
 c.tauj=cache[4*n];
 c.jss=cache[5*n];
 c.a_na_L=cache[6*n];
 c.b_na_L=cache[7*n];
 c.h_na_L=cache[8*n];
 c.fnak=cache[9*n];
 c.xrss=cache[10*n];
 c.tauxr=cache[11*n];
 c.rkr=cache[12*n];
 c.xsss=cache[13*n];
 c.tauxs=cache[14*n];
 c.kp_kp=cache[15*n];
 c.xtoss=cache[16*n];
 c.ytoss=cache[17*n];
 c.tauxtos=cache[18*n];
 c.tauytos=cache[19*n];
 c.tauxtof=cache[20*n];
 c.tauytof=cache[21*n];
 c.I_Clbk=cache[22*n];
 c.dss=cache[23*n];
 c.taud=cache[24*n];
 c.fss=cache[25*n];
 c.tauf=cache[26*n];
 c.math_0=cache[27*n];
 c.math_1=cache[28*n];
 c.math_2=cache[29*n];
 c.math_3=cache[30*n];
 return gpb_nodal_rhs(time,y,V,0.,d,c,n,e);
 }
};

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
