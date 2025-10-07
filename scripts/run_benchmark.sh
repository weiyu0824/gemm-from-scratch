#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/results
echo "🚀 Running benchmarks..."
./build/bin/gemm_benchmark | tee data/results/gemm_naive.log
echo "✅ Results saved to data/results/gemm_naive.log"
