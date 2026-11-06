"""Official Grandi RHS, integrated with the existing AFSEI SDIRK arithmetic.

This is a benchmark model adapter. It does not replace the production GPB/Land.
No Rush-Larsen update is used in its simulated SDIRK trajectory.
"""
from pathlib import Path
import re,json,hashlib
old=Path('/mnt/large2/gjh/AFSEI/demo/ep_opencarp_comparison')
root=Path('/mnt/large2/gjh/AFSEI/demo/ep_afsei_nodal_opencarp_matched');root.mkdir(exist_ok=True)
cc=(old/'generated/Grandi.cc').read_text();hh=(old/'generated/Grandi.h').read_text()
macros=cc[cc.index('#define Bmax_CaM'):cc.index('void GrandiIonType::initialize_params')]
macro_expr=dict(re.findall(r'^#define\s+(\w+)\s+(.+)$',macros,re.M))
defs={}
# Use only the compute routine, not initialization expressions with the same names.
body=cc[cc.index('void compute_Grandi_cpu'):cc.index('bool GrandiIonType::has_trace')]
for name,expr in re.findall(r'^\s*(?:GlobalData_t|double)\s+(\w+)\s*=\s*(.*);\s*$',body,re.M):
    if name not in defs:defs[name]=expr
defs.pop('V', None)  # Voltage is the external scalar argument of the extracted RHS.
ion=[v for v in re.findall(r'^\s*Iion\s*=\s*(.*);\s*$',body,re.M) if v.strip()!='Iion']
if len(ion)!=1:raise RuntimeError('Ambiguous official ionic current assignment')
defs['Iion']=ion[0]
struct=re.search(r'struct Grandi_state \{.*?\n\};',hh,re.S).group()
fields=[name for typ,name in re.findall(r'(GlobalData_t|Gatetype)\s+(\w+)\s*;',struct) if name!='__Cai_mM_local']
targets={}
available=set(defs)|set(macro_expr)
for name in fields:
    direct='diff_'+name
    if direct in available:targets[name]=direct
    elif name+'_infinity' in available and 'tau_'+name in available:
        targets[name]=f'({name}_infinity-sv->{name})/tau_{name}'
    elif 'set_'+name+'_tozero_in_diff_'+name in available and 'partial_diff_'+name+'_del_'+name in available:
        targets[name]=f'set_{name}_tozero_in_diff_{name}+partial_diff_{name}_del_{name}*sv->{name}'
    else:raise RuntimeError('No official RHS expression for '+name)
order=[];done=set();active=set()
def dependencies(expr):
    clean=re.sub(r'(?:sv|p)->\w+','',expr)
    return [n for n in re.findall(r'\b[A-Za-z_]\w*\b',clean) if n in available]
def visit(name):
    if name in done:return
    if name in active:raise RuntimeError('Cyclic RHS dependency: '+name)
    active.add(name)
    for n in dependencies(defs[name] if name in defs else macro_expr[name]):visit(n)
    active.remove(name);done.add(name)
    if name in defs:order.append(name)
for expr in [*targets.values(),'Iion']:
    for name in dependencies(expr):visit(name)
if any('rush_larsen' in name or name.endswith('_new') for name in order):raise RuntimeError('Time-integration expression leaked into RHS')
params=re.search(r'struct Grandi_Params \{.*?\n\};',hh,re.S).group()
default_body=cc[cc.index('void GrandiIonType::initialize_params'):cc.index('void GrandiIonType::construct_tables')]
defaults={}
for name,expr in re.findall(r'p->(\w+)\s*=\s*([^;]+);',default_body):defaults.setdefault(name,expr)
struct=struct.replace('Gatetype','GlobalData_t')
load='\n'.join(f'  value.{name}=y[{i}]'+('*1000.;' if name=='Cai' else ';') for i,name in enumerate(fields))
store='\n'.join(f'  dy[{i}]=({expr})'+('/1000.;' if name=='Cai' else ';') for i,(name,expr) in enumerate(targets.items()))
undef='\n'.join('#undef '+name for name in macro_expr)
header='''// SPDX-FileCopyrightText: Copyright (c) NumeriCor GmbH
// SPDX-License-Identifier: LicenseRef-APL-1.1
// Grandi, Pasqualini & Bers 2010, doi:10.1016/j.yjmcc.2009.09.019.
// Mechanically extracted RHS; original expressions and physical constants retained.
#pragma once
#include <cmath>
#ifdef __CUDACC__
#define AFSEI_CELL_HD __host__ __device__
#else
#define AFSEI_CELL_HD
#endif
namespace reference_grandi_sdirk {
using GlobalData_t=double;
'''+params+'\n'+struct+'\n'+macros+'''
AFSEI_CELL_HD inline double rhs(const double* y,double V,double* dy) {
  Grandi_Params ps{};auto* p=&ps;
'''+ '\n'.join(f'  p->{name}={expr};' for name,expr in defaults.items())+'''
  Grandi_state value{};auto* sv=&value;
'''+load+'\n'+'\n'.join(f'  double {name}={defs[name]};' for name in order)+'\n'+store+'''
  return Iion;
}
'''+undef+'''
}
#undef AFSEI_CELL_HD
'''
(root/'reference_grandi_rhs.h').write_text(header)
# Double-precision version of the original generated RL update is retained
# ONLY as an independent infinitesimal-step derivative check, never as the solver.
ref=(old/'shared_grandi.h').read_text().replace('namespace shared_grandi','namespace reference_grandi_double').replace('using Gatetype = float;','using Gatetype = double;')
(root/'reference_grandi_rl_check.h').write_text(ref)
cpp='''#include <torch/extension.h>
void reference_rhs(at::Tensor,at::Tensor,at::Tensor,at::Tensor);
void reference_advance(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double,int64_t);
void reference_rl_check(at::Tensor,at::Tensor,at::Tensor,at::Tensor,double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("rhs",&reference_rhs);m.def("advance",&reference_advance);m.def("rl_check",&reference_rl_check);}
'''
(root/'reference_sdirk.cpp').write_text(cpp)
ns=len(fields)
positive=[fields.index(n) for n in ['Ca_sr','Caj','Casl','Cai','Naj','Nasl','Nai','Ki']]
cu='''#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include "reference_grandi_rhs.h"
#include "reference_grandi_rl_check.h"
using at::Tensor;
constexpr int NS='''+str(ns)+''';
__host__ __device__ inline double advance_one(double* state,double V,double dt,int steps,int& bad) {
  constexpr double gamma=0.788675134594812882254;constexpr double a21=1.-2.*gamma;
  double k1[NS],k2[NS],temp[NS],previous[NS];double sum=0.;
  for(int sub=0;sub<steps;++sub) {
    reference_grandi_sdirk::rhs(state,V,k1);
    for(int iter=0;iter<3;++iter) {
      for(int j=0;j<NS;++j){previous[j]=k1[j];temp[j]=state[j]+dt*gamma*k1[j];}
      reference_grandi_sdirk::rhs(temp,V,k1);
      if(iter>0){double e=0.;for(int j=0;j<NS;++j)e=fmax(e,fabs(k1[j]-previous[j])/fmax(fabs(k1[j]),1.));if(e<1e-6)break;}
    }
    for(int j=0;j<NS;++j)k2[j]=k1[j];
    for(int iter=0;iter<3;++iter) {
      for(int j=0;j<NS;++j){previous[j]=k2[j];temp[j]=state[j]+dt*(a21*k1[j]+gamma*k2[j]);}
      reference_grandi_sdirk::rhs(temp,V,k2);
      if(iter>0){double e=0.;for(int j=0;j<NS;++j)e=fmax(e,fabs(k2[j]-previous[j])/fmax(fabs(k2[j]),1.));if(e<1e-6)break;}
    }
    for(int j=0;j<NS;++j){state[j]+=dt*(.5*k1[j]+.5*k2[j]);if(!isfinite(state[j])){state[j]=0.;++bad;}}
    double I=reference_grandi_sdirk::rhs(state,V,temp);if(!isfinite(I)){I=0.;++bad;}sum+=I;
  }
'''+''.join(f'  if(state[{i}]<=0.)++bad;\n' for i in positive)+'''
  return sum/steps;
}
__global__ void rhs_kernel(const double* y,const double* V,double* z,double* I,int64_t n) {
 auto id=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=n)return;double s[NS],d[NS];
 for(int j=0;j<NS;++j)s[j]=y[j*n+id];I[id]=reference_grandi_sdirk::rhs(s,V[id],d);
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
 for(int64_t id=0;id<n;++id){double s[NS],d[NS];for(int j=0;j<NS;++j)s[j]=a[j*n+id];c[id]=reference_grandi_sdirk::rhs(s,v[id],d);for(int j=0;j<NS;++j)b[j*n+id]=d[j];}}
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
'''+ '\n'.join(f' s.{name}=a[{i}*n+id]'+('*1000.;' if name=='Cai' else ';') for i,name in enumerate(fields))+'''
 int g=0;c[id]=reference_grandi_double::step(&s,v[id],dt,g);
'''+ '\n'.join(f' b[{i}*n+id]=s.{name}'+('/1000.;' if name=='Cai' else ';') for i,name in enumerate(fields))+'''
 }
}
'''
(root/'reference_sdirk.cu').write_text(cu)
(root/'reference_cell.py').write_text('''from pathlib import Path
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
def module():
 return load(name='afsei_reference_grandi_sdirk_v20',sources=[str(ROOT/'reference_sdirk.cpp'),str(ROOT/'reference_sdirk.cu')],extra_cflags=['-O3','-ffp-contract=off'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
''')
manifest=dict(model='Official Grandi v20 RHS',states=fields,state_count=ns,rhs_intermediate_count=len(order),
    derivative_expressions=targets,source_sha256={n:hashlib.sha256((old/'generated'/n).read_bytes()).hexdigest() for n in ['Grandi.cc','Grandi.h']},
    ode_method='Same two-stage fixed-point SDIRK arithmetic as AFSEI; gamma=.7886751345948129, 3 stage iterations, relative criterion 1e-6',
    ode_current='Average of final-state currents across ODE substeps, same as production AFSEI',
    state_precision='float64, same as production AFSEI',physical_constants_changed=False,production_cell_modified=False,
    state_unit_mapping={'Cai':'mM externally, matching AFSEI SDIRK error scaling; exact x1000 mapping to native Grandi uM inside RHS'},
    rl_check_note='Official generated RL update retained only for independent derivative validation; not the simulation method')
(root/'reference_cell_provenance.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(states=ns,intermediates=len(order),root=str(root)),indent=2))
