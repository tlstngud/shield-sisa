#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build every component into ./build/<component>.
#   ./build.sh                     audio, prep, cpu, cuda (if nvcc is found), service + tools
#   SHIELD_NO_CUDA=1 ./build.sh    skip the CUDA engine (see KNOWN_ISSUES.md: the service currently links it)
set -euo pipefail
cd "$(dirname "$0")"
GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)
JOBS=${JOBS:-$(nproc)}
b() {  # b <source dir> <build dir> [cmake args...]
  local src=$1 out=$2; shift 2
  cmake -S "$src" -B "build/$out" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release "$@"
  cmake --build "build/$out" -j "$JOBS"
}
b engine/audio audio
b engine/preprocess prep
b engine/model cpu
if [ -z "${SHIELD_NO_CUDA:-}" ] && { command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ]; }; then
  export PATH=/usr/local/cuda/bin:$PATH
  b engine/cuda cuda -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-86}"   # sm_80+ required (A100=80, A6000/A10G=86, L4=89, H100=90)
else
  echo "nvcc not found: skipping engine/cuda" >&2
fi
b service service
echo "built: $(ls build/*/lib*.so | tr '\n' ' ')"
