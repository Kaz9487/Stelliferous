#!/bin/sh
# Builds epoch.bend and times one epoch in Bend and in PyTorch with the given
# number of threads. The lines go to out/results/training/epoch_THREADS.txt.
set -eu
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
threads="${1:-4}"
mkdir -p out/results/training
"$CC" -O3 -march=native -ffp-contract=off -std=c11 out/build/training/epoch.c -lpthread -lm -o out/build/training/epoch
{
  out/build/training/epoch --threads "$threads" --gpu off
  "$py" -B benchmarks/training/epoch_torch.py "$threads"
} | tee "out/results/training/epoch_$threads.txt"
