# AFSEI environment baseline

The independent AFSEI environment is cloned from the server's verified afsi-torch environment.
Environment packages reside on the data disk at `/mnt/large2/gjh/conda-envs/afsei`.

Expected core versions: Python 3.12.14, PyTorch 2.14.0+cu130, CUDA Toolkit 13.0,
NumPy 2.5.3 and Triton 3.8.0; GPU architecture 8.9 (RTX 4090).

`conda-explicit.txt` records the Linux Conda package builds.
`pip-freeze.txt` is an inventory of Python packages, not a portable replacement for the Conda toolchain.
`server-validation.json` records the actual environment check.
`cuda-validation.txt` records the inherited CUDA tests run for this initialization.

The afsei editable installation must point to AFSEI/src, not AFSI_GPU/src.
After activating the environment, installation is `python -m pip install --no-deps --no-build-isolation -e .`.
Custom GPB/Land kernels are not implemented yet. Existing AFSI CUDA kernels are verified separately.
FEniCS/PETSc/MUMPS are not required for the planned PyTorch GPU solver; an independent reference environment
can be used later for comparisons with the original C++ electromechanics project.


To reconstruct on the same Linux platform:

```bash
conda create --yes --prefix /mnt/large2/gjh/conda-envs/afsei --file environment/conda-explicit.txt
source scripts/activate_afsei.sh
python -m pip install -r environment/pip-freeze.txt
python scripts/check_afsei_environment.py
```

The explicit file captures Conda builds and channel URLs; PyPI inventory captures the pip-installed CUDA
PyTorch distribution as well. Review availability on a new machine instead of upgrading versions implicitly.
Only architecture 8.9 is compiled by default; use AFSEI_CUDA_ARCH_LIST for another GPU architecture.
