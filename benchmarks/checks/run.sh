#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
for entry in convolution scalar_convolution array_api array_api_zero_workspace; do
    "$CC" -O3 -march=native -ffp-contract=off -std=c11 "out/build/integration/$entry.c" -lpthread -lm -o "out/build/integration/$entry"
    "$py" benchmarks/checks/check_runtime.py --entry "$entry"
done
"$CC" -O3 -march=native -ffp-contract=off -std=c11 out/build/integration/tensor_flow.c -lpthread -lm -o out/build/integration/tensor_flow
"$py" -B benchmarks/checks/tensor_flow_cases.py --run
