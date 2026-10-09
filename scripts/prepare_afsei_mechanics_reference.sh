#!/usr/bin/env bash
# Regenerate the pinned UFL in an isolated shell; never use the dirty header.
set -eo pipefail
AFSEI_TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
AFSEI_TASK_SOURCE=${AFSEI_EP_SOURCE_REPO:-/mnt/large2/gjh/test_ele_soild_heart-test}
AFSEI_TASK_MESH=${AFSEI_REFERENCE_MESH:-/mnt/large2/gjh/realistic_left_ventricle}
AFSEI_TASK_REF="$AFSEI_TASK_ROOT/.cache/source-mechanics-ref"
AFSEI_TASK_COMMIT=5bdde73c6be10cefadb005a0fcec5fdb6521f336
source "${AFSEI_FENICS_SETUP:-/home/fenics/spack/share/spack/setup-env.sh}"
spack load /nvb7ww7
mkdir -p "$AFSEI_TASK_REF"
git -C "$AFSEI_TASK_SOURCE" show "$AFSEI_TASK_COMMIT:Active_land.ufl" > "$AFSEI_TASK_REF/Active_land.ufl"
(
  cd "$AFSEI_TASK_REF"
  ffc -l dolfin Active_land.ufl
)
cmake -S "$AFSEI_TASK_ROOT/validation/source_mechanics" -B "$AFSEI_TASK_ROOT/.cache/source-mechanics-build" \
  -DDOLFIN_DIR="${DOLFIN_DIR:-/home/fenics/spack/opt/spack/linux-ubuntu24.04-icelake/gcc-11.4.0/fenics-2019.1.0.post0-nvb7ww7rb4ee65zs7d7bf5b54n6zzvbi/share/dolfin/cmake}" \
  -DAFSEI_REFERENCE_INCLUDE="$AFSEI_TASK_REF" \
  -DBoost_NO_SYSTEM_PATHS=ON -DBoost_NO_BOOST_CMAKE=ON
cmake --build "$AFSEI_TASK_ROOT/.cache/source-mechanics-build" -j 2
for AFSEI_TASK_MODE in passive land frozen; do
  "$AFSEI_TASK_ROOT/.cache/source-mechanics-build/afsei_source_mechanics" \
    "$AFSEI_TASK_MESH" "$AFSEI_TASK_REF/$AFSEI_TASK_MODE.csv" "$AFSEI_TASK_MODE" expression
done
"$AFSEI_TASK_ROOT/.cache/source-mechanics-build/afsei_source_mechanics" \
  "$AFSEI_TASK_MESH" "$AFSEI_TASK_REF/cg1_land.csv" land cg1
echo "Source mechanical oracle written to $AFSEI_TASK_REF"
