#!/bin/sh
# Builds the compiled model and runs it, saving the block outputs.
set -eu
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
threads=${1:-1}
result=native
if [ "$threads" -ne 1 ]; then result="native_${threads}t"; fi
"$CC" -std=c11 -O3 -march=native -ffp-contract=off out/models/vit_b16.c -lpthread -lm -o out/models/vit_b16
export VIT_OUTPUT="out/results/vit_b16/$result"
export VIT_LAYERS=1
export VIT_EXTRA="${VIT_EXTRA:-5}"
mkdir -p "$VIT_OUTPUT"
out/models/vit_b16 +RTS --threads "$threads" --gpu off -RTS | tee "$VIT_OUTPUT/forward.txt"
