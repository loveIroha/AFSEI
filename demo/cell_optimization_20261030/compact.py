from pathlib import Path
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
def native():
 return load(name='afsei_compact_native_gpb',sources=[str(ROOT/'compact_reference.cpp'),str(ROOT/'compact_reference.cu')],extra_cflags=['-O3','-ffp-contract=off'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
def production():
 load(name='afsei_ep_cuda_compact',sources=[str(ROOT/'compact_tissue.cpp'),str(ROOT/'compact_tissue.cu')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],is_python_module=False,verbose=False)
 import torch
 return torch.ops.afsei_ep_cuda_compact
