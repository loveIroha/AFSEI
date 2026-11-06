"""Explicit Euler GPB57/Land, sharing the production equations and FP64 cache."""
from pathlib import Path
import torch
import threading
import json,hashlib
_loaded=False
_lock=threading.Lock()
def advance(*args):
    global _loaded
    if not _loaded:
        with _lock:
            if not _loaded:
                # Verify generated GPB source provenance with the common loader.
                from .cuda_tissue import build
                build()
                from torch.utils.cpp_extension import load
                p=Path(__file__).parent/'csrc'
                audit=json.loads((p/'gpb_euler_provenance.json').read_text())
                if (hashlib.sha256((p/'gpb_node_device.h').read_bytes()).hexdigest()!=audit['source_sha256'] or
                    hashlib.sha256((p/'gpb_euler_device.h').read_bytes()).hexdigest()!=audit['generated_sha256']):
                    raise RuntimeError('GPB Euler header provenance changed; regenerate stable header')
                load(name='afsei_ep_euler_v1',sources=[str(p/'tissue_euler.cpp'),str(p/'tissue_euler.cu')],
                    extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],is_python_module=False)
                _loaded=True
    return torch.ops.afsei_ep_euler.advance(*args)
