#!/usr/bin/env bash
# Independent, pinned-source UFL assembly at supplied converged positions.
set -eo pipefail
AFSEI_TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
AFSEI_TASK_SOURCE=${AFSEI_EP_SOURCE_REPO:-/mnt/large2/gjh/test_ele_soild_heart-test}
AFSEI_TASK_COMMIT=5bdde73c6be10cefadb005a0fcec5fdb6521f336
AFSEI_TASK_REF="$AFSEI_TASK_ROOT/.cache/source-preload-ref"
AFSEI_TASK_MESH=${1:-/mnt/large2/gjh/realistic_left_ventricle}
AFSEI_TASK_OUTPUT=${2:?supply directory with source/closed input CSV}
if [[ ${AFSEI_FENICS_ISOLATED:-0} != 1 ]]; then
  # A child of the active AFSEI environment otherwise inherits Conda CC/CXX,
  # sysroot and library search paths incompatible with the existing FEniCS.
  exec env -i HOME="$HOME" USER="${USER:-}" LOGNAME="${LOGNAME:-}" PATH=/usr/bin:/bin \
    AFSEI_FENICS_ISOLATED=1 AFSEI_EP_SOURCE_REPO="$AFSEI_TASK_SOURCE" \
    AFSEI_FENICS_SETUP="${AFSEI_FENICS_SETUP:-/home/fenics/spack/share/spack/setup-env.sh}" \
    DOLFIN_DIR="${DOLFIN_DIR:-/home/fenics/spack/opt/spack/linux-ubuntu24.04-icelake/gcc-11.4.0/fenics-2019.1.0.post0-nvb7ww7rb4ee65zs7d7bf5b54n6zzvbi/share/dolfin/cmake}" \
    bash "$0" "$AFSEI_TASK_MESH" "$AFSEI_TASK_OUTPUT"
fi
source "${AFSEI_FENICS_SETUP:-/home/fenics/spack/share/spack/setup-env.sh}"
spack load /nvb7ww7
mkdir -p "$AFSEI_TASK_REF"
for AFSEI_TASK_NAME in Active_land.ufl cavity_volume.h cavity_surface.h; do
  git -C "$AFSEI_TASK_SOURCE" show "$AFSEI_TASK_COMMIT:$AFSEI_TASK_NAME" > "$AFSEI_TASK_REF/$AFSEI_TASK_NAME"
done
(
  cd "$AFSEI_TASK_REF"
  ffc -l dolfin Active_land.ufl
)
cmake -S "$AFSEI_TASK_ROOT/validation/source_preload" -B "$AFSEI_TASK_ROOT/.cache/source-preload-build-system" \
  -DDOLFIN_DIR="${DOLFIN_DIR:-/home/fenics/spack/opt/spack/linux-ubuntu24.04-icelake/gcc-11.4.0/fenics-2019.1.0.post0-nvb7ww7rb4ee65zs7d7bf5b54n6zzvbi/share/dolfin/cmake}" \
  -DAFSEI_REFERENCE_INCLUDE="$AFSEI_TASK_REF" -DBoost_NO_SYSTEM_PATHS=ON -DBoost_NO_BOOST_CMAKE=ON
cmake --build "$AFSEI_TASK_ROOT/.cache/source-preload-build-system" -j 2
for AFSEI_TASK_MODE in source closed; do
  "$AFSEI_TASK_ROOT/.cache/source-preload-build-system/afsei_source_preload" "$AFSEI_TASK_MESH" \
    "$AFSEI_TASK_OUTPUT/${AFSEI_TASK_MODE}_input.csv" "$AFSEI_TASK_OUTPUT/${AFSEI_TASK_MODE}_oracle.csv" 8
done
