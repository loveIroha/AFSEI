"""Mechanical adapter of official generated Grandi code, benchmark use only.

Generated expressions, Rush-Larsen updates, float gates and clamps are retained.
The source license and hashes are recorded; production GPB/Land is untouched.
"""
from pathlib import Path
import re,json,hashlib
root=Path('/mnt/large2/gjh/AFSEI/demo/ep_opencarp_comparison')
cc=(root/'generated/Grandi.cc').read_text();hh=(root/'generated/Grandi.h').read_text()
def braces(text,start):
    a=text.index('{',start);depth=1;i=a+1
    while depth:
        depth+=(text[i]=='{')-(text[i]=='}');i+=1
    return text[a+1:i-1]
def function(name):return braces(cc,cc.index(name))
def loop(src):return braces(src,src.index('for (')) if 'for (' in src else braces(src,src.index('for('))
macro=cc[cc.index('#define Bmax_CaM'):cc.index('void GrandiIonType::initialize_params')]
params=re.search(r'struct Grandi_Params \{.*?\n\};',hh,re.S).group()
state=re.search(r'struct Grandi_state \{.*?\n\};',hh,re.S).group()
fields=re.findall(r'(GlobalData_t|Gatetype)\s+(\w+)\s*;',state)
parambody=function('void GrandiIonType::initialize_params')
defaults={}
for name,value in re.findall(r'p->(\w+)\s*=\s*([^;]+);',parambody):defaults.setdefault(name,value)
paramcode='\n'.join('  p->'+name+' = '+value+';' for name,value in defaults.items())
def scalar_code(name):
    src=function(name)
    pre=src[src.index('// Define the constants that depend on the parameters.'):src.index('//Prepare all the public arrays.')]
    body=loop(src)
    body=body[body.index('//Initialize the external vars'):]
    body=body[:body.index('//Save all external vars')]
    body=body.replace('GlobalData_t Iion = Iion_ext[__i];','GlobalData_t Iion = 0.;')
    body=body.replace('GlobalData_t V = V_ext[__i];','GlobalData_t V = voltage;')
    body=re.sub(r'IIF_warn\(__i, "[^"]*"\);','++guards;',body)
    return pre+'\n'+body
init=scalar_code('void GrandiIonType::initialize_sv')
compute=scalar_code('void compute_Grandi_cpu')
definitions=re.findall(r'^#define\s+(\w+)',macro,re.M)
undef='\n'.join('#undef '+name for name in definitions)
header='''// SPDX-FileCopyrightText: Copyright (c) NumeriCor GmbH
// SPDX-License-Identifier: LicenseRef-APL-1.1
// Adapter generated mechanically from the official openCARP v20.0 Grandi code.
// Grandi, Pasqualini & Bers 2010, doi:10.1016/j.yjmcc.2009.09.019.
#pragma once
#include <cmath>
#ifdef __CUDACC__
#define SHARED_CELL_HD __host__ __device__
#else
#define SHARED_CELL_HD
#endif
namespace shared_grandi {
using GlobalData_t = double;
using Gatetype = float;
'''+params+'\n'+state+'\n'+macro+'''
SHARED_CELL_HD inline void defaults(Grandi_Params* p) {
'''+paramcode+'''
}
SHARED_CELL_HD inline double initialize(Grandi_state* sv, double dt) {
  Grandi_Params storage{};defaults(&storage);auto* p=&storage;
  double voltage=0.; int guards=0;
'''+init+'''
  return V;
}
SHARED_CELL_HD inline double step(Grandi_state* sv,double voltage,double dt,int& guards) {
  Grandi_Params storage{};defaults(&storage);auto* p=&storage;
'''+compute+'''
  return Iion;
}
'''+undef+'\n}\n#undef SHARED_CELL_HD\n'
(root/'shared_grandi.h').write_text(header)
loads='\n'.join(f'  s.{name}=y[{i}*n+id];' for i,(_,name) in enumerate(fields))
stores='\n'.join(f'  z[{i}*n+id]=s.{name};' for i,(_,name) in enumerate(fields))
kernel='''#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include "shared_grandi.h"
using at::Tensor;
constexpr int NS='''+str(len(fields))+''';
__global__ void init_kernel(double* z,double* v,int64_t n,double dt) {
  auto id=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=n)return;
  shared_grandi::Grandi_state s{};v[id]=shared_grandi::initialize(&s,dt);
'''+stores+'''
}
__global__ void step_kernel(const double* y,const double* v,double* z,double* I,int64_t* guard,int64_t n,double dt) {
  auto id=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=n)return;
  shared_grandi::Grandi_state s{};
'''+loads+'''
  int guards=0;I[id]=shared_grandi::step(&s,v[id],dt,guards);guard[id]=guards;
'''+stores+'''
}
void shared_init(Tensor z,Tensor v,double dt) {
  TORCH_CHECK(z.scalar_type()==at::kDouble && v.scalar_type()==at::kDouble,"float64 required");
  auto n=v.numel();TORCH_CHECK(z.numel()==NS*n,"state shape mismatch");
  if(v.is_cuda()) {
    const c10::cuda::CUDAGuard g(v.device());
    init_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(z.data_ptr<double>(),v.data_ptr<double>(),n,dt);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto* zz=z.data_ptr<double>();auto* vv=v.data_ptr<double>();
    for(int64_t id=0;id<n;++id) {shared_grandi::Grandi_state s{};vv[id]=shared_grandi::initialize(&s,dt);
'''+stores.replace('z[','zz[')+'''
    }
  }
}
void shared_step(Tensor y,Tensor v,Tensor z,Tensor I,Tensor guard,double dt) {
  auto n=v.numel();TORCH_CHECK(y.numel()==NS*n && z.numel()==NS*n,"state shape mismatch");
  if(v.is_cuda()) {
    const c10::cuda::CUDAGuard g(v.device());
    step_kernel<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(y.data_ptr<double>(),v.data_ptr<double>(),z.data_ptr<double>(),I.data_ptr<double>(),guard.data_ptr<int64_t>(),n,dt);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto* yy=y.data_ptr<double>();auto* zz=z.data_ptr<double>();auto* vv=v.data_ptr<double>();
    auto* ii=I.data_ptr<double>();auto* gg=guard.data_ptr<int64_t>();
    for(int64_t id=0;id<n;++id) {shared_grandi::Grandi_state s{};
'''+loads.replace('y[','yy[')+'''
      int guards=0;ii[id]=shared_grandi::step(&s,vv[id],dt,guards);gg[id]=guards;
'''+stores.replace('z[','zz[')+'''
    }
  }
}
'''
(root/'shared_grandi.cu').write_text(kernel)
(root/'shared_grandi.cpp').write_text('''#include <torch/extension.h>
void shared_init(at::Tensor,at::Tensor,double);
void shared_step(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("initialize",&shared_init);m.def("step",&shared_step);}
''')
(root/'shared_cell.py').write_text('''from pathlib import Path
import torch
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
def module():
    return load(name='afsei_benchmark_shared_grandi_v20',sources=[str(ROOT/'shared_grandi.cpp'),str(ROOT/'shared_grandi.cu')],
        extra_cflags=['-O3','-ffp-contract=off'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
''')
manifest=dict(source='openCARP v20.0 generated Grandi.cc and Grandi.h',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'generated').glob('Grandi.*')},
    state_fields=fields,gate_storage='float32 as in official native code',other_storage='float64',state_count=len(fields),
    method='Official generated Rush-Larsen update retained; execution wrapper only',production_gpb_modified=False)
(root/'shared_cell_provenance.json').write_text(json.dumps(manifest,indent=2))
print('GENERATED',len(fields),'states')
