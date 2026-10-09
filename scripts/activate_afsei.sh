# Source this file from bash; execution in a child shell does not activate the caller.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Use: source scripts/activate_afsei.sh" >&2
  exit 1
fi
AFSEI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AFSEI_ENV_PREFIX="${AFSEI_ENV_PREFIX:-/mnt/large2/gjh/conda-envs/afsei}"
if ! declare -F conda >/dev/null; then
  source "${AFSEI_CONDA_ROOT:-/home/user/miniconda3}/etc/profile.d/conda.sh" || return 1
fi
conda activate "$AFSEI_ENV_PREFIX" || return 1
export CUDA_VISIBLE_DEVICES="${AFSEI_GPU:-1}"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${AFSEI_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${AFSEI_BUILD_JOBS:-2}"
AFSEI_CACHE_ROOT="${AFSEI_CACHE_ROOT:-$AFSEI_ROOT/.cache}"
export TORCH_EXTENSIONS_DIR="$AFSEI_CACHE_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$AFSEI_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$AFSEI_CACHE_ROOT/torchinductor"
export PYTHONPATH="$AFSEI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" || return 1
echo "AFSEI environment: $CONDA_PREFIX; physical GPU: $CUDA_VISIBLE_DEVICES"
