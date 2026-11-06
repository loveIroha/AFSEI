from pathlib import Path
from torch.utils.cpp_extension import load
ROOT=Path(__file__).resolve().parent
def module():
 return load(name='afsei_reference_grandi_sdirk_v20',sources=[str(ROOT/'reference_sdirk.cpp'),str(ROOT/'reference_sdirk.cu')],extra_cflags=['-O3','-ffp-contract=off'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
