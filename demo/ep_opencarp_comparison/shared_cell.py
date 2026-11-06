from pathlib import Path
import torch
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
def module():
    return load(name='afsei_benchmark_shared_grandi_v20',sources=[str(ROOT/'shared_grandi.cpp'),str(ROOT/'shared_grandi.cu')],
        extra_cflags=['-O3','-ffp-contract=off'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
