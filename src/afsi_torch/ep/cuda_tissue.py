"""Lazy GPB/Land CUDA extension; imports and monodomain PDE do not compile it."""
from pathlib import Path
import threading
import torch

_lock = threading.Lock()
_loaded = False


def build(verbose=False):
    global _loaded
    if _loaded:
        return
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError('GPB/Land requires CUDA-enabled PyTorch and an available GPU')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('build GPB/Land CUDA operators before graph capture')
    with _lock:
        if _loaded:
            return
        from torch.utils.cpp_extension import CUDA_HOME, load
        from ..mac.cuda_ib import _validate_toolkit
        if CUDA_HOME is None:
            raise RuntimeError('set CUDA_HOME to a toolkit compatible with PyTorch')
        _validate_toolkit(CUDA_HOME, torch.version.cuda)
        folder = Path(__file__).parent/'csrc'
        load(name='afsei_ep_cuda', sources=[str(folder/f) for f in ('tissue.cpp', 'tissue.cu')],
             extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--fmad=false', '-lineinfo'],
             is_python_module=False, verbose=verbose)
        _loaded = True


def rhs(state, voltage_mv, time_ms=0.0):
    build()
    return torch.ops.afsei_ep_cuda.rhs(state, voltage_mv, float(time_ms))


def advance(state, land, voltage_mv, stretch, zs, zw, time_ms, dt_ms, substeps, advance_land=True):
    build()
    return torch.ops.afsei_ep_cuda.advance(state, land, voltage_mv, stretch, zs, zw,
        float(time_ms), float(dt_ms), substeps, advance_land)
