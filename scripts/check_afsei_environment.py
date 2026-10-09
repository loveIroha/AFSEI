"""Check package provenance, CUDA tools and actual GPU tensor operations."""
from pathlib import Path
import argparse
import importlib.metadata as metadata
import json
import os
import re
import shutil
import subprocess
import sys
import torch
import afsi_torch

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
module = Path(afsi_torch.__file__).resolve()
if root not in module.parents:
    raise SystemExit(f'Solver import points outside AFSEI: {module}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable; this check requires a real GPU.')
cuda_home = Path(os.environ.get('CUDA_HOME', sys.prefix))
nvcc = cuda_home/'bin/nvcc'
if not nvcc.is_file():
    raise SystemExit(f'CUDA compiler missing: {nvcc}')
nvcc_version = subprocess.check_output([str(nvcc), '--version'], text=True)
release = re.search(r'release (\d+\.\d+)', nvcc_version)
if not release or release.group(1).split('.')[0] != torch.version.cuda.split('.')[0]:
    raise SystemExit('CUDA compiler and PyTorch CUDA major versions differ.')
if not shutil.which('c++'):
    raise SystemExit('C++ compiler missing.')
packages = {}
for name in ('afsei', 'torch', 'numpy', 'triton', 'ninja', 'h5py', 'meshio', 'fenics-basix', 'gmsh', 'matplotlib', 'pytest'):
    packages[name] = metadata.version(name)
from torch.utils.cpp_extension import CUDA_HOME as torch_cuda_home
if not torch_cuda_home or Path(torch_cuda_home).resolve() != cuda_home.resolve():
    raise SystemExit('PyTorch extension loader selects a different CUDA toolkit.')
matrix = torch.tensor([[4., 1.], [1., 3.]], device='cuda', dtype=torch.float64)
rhs = torch.tensor([1., 2.], device='cuda', dtype=torch.float64)
answer = torch.linalg.solve(matrix, rhs)
torch.testing.assert_close(matrix @ answer, rhs, rtol=1e-12, atol=1e-12)
csr = matrix.to_sparse_csr()
torch.testing.assert_close(torch.sparse.mm(csr, answer[:, None])[:, 0], rhs, rtol=1e-12, atol=1e-12)
torch.cuda.synchronize()
report = dict(passed=True, python=sys.version.split()[0], python_executable=sys.executable,
    packages=packages, torch_cuda=torch.version.cuda, nvcc_release=release.group(1),
    gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
    physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), cuda_arch_list=os.environ.get('TORCH_CUDA_ARCH_LIST'), solver_module=str(module),
    dense_and_sparse_gpu_checks=True, three_field_integration_implemented=False)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
