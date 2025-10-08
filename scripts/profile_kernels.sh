#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build/bin/gemm_benchmark"
PROFILE_DIR="${ROOT_DIR}"

mkdir -p "${PROFILE_DIR}"

if [ ! -f "${BIN}" ]; then
    echo "❌ Binary not found. Run ./scripts/build.sh first."
    exit 1
fi

# Choose sizes to profile
SIZES=(512 1024 2048 4096)

echo "🔍 Running Nsight Compute profiling..."

for size in "${SIZES[@]}"; do
    OUT_FILE="${PROFILE_DIR}/profile_gemm_${size}.ncu-rep"
    echo "→ Profiling gemm with size=${size}"
    ncu --set full \
        --target-processes all \
        --export "${OUT_FILE}" \
        "${BIN}" --mode profile --size "${size}"
done


echo "✅ Nsight Compute profiles stored in ${PROFILE_DIR}/"