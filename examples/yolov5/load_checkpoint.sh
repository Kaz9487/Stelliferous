#!/bin/sh
# Builds the compiled checkpoint loader and runs it on a checkpoint, the
# official one by default, saving every tensor under
# out/results/checkpoint/<case>/bend, and with the batch normalizations
# folded under out/results/checkpoint/<case>/folded.
set -eu
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
checkpoint="${1:-data/weights/yolov5n-v7.0.pt}"
output="out/results/checkpoint/${2:-official}"
mkdir -p "$output/bend" "$output/folded"
"$CC" -std=c11 -O3 -march=native -ffp-contract=off out/models/load_checkpoint.c -lpthread -lm -o out/models/load_checkpoint
CHECKPOINT="$checkpoint" CHECKPOINT_OUTPUT="$output/bend" CHECKPOINT_FOLDED="$output/folded" \
  out/models/load_checkpoint --threads 1 --gpu off | tee "$output/load.txt"
