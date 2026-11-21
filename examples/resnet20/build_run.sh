#!/bin/sh
# Builds resnet20.bend and trains it, then runs resnet20_torch.py when its
# output folder is given. Both continue after the last epoch their folder holds
# and add their lines to train.txt there.
# sh build_run.sh THREADS EPOCHS STEPS FILES TESTS BEND_OUTPUT [TORCH_OUTPUT]
set -eu
cd "$(dirname "$0")/../.."
. scripts/linux_environment.sh
threads=$1

# The command's lines go to the screen and to train.txt; its status is kept.
logged() {
  folder=$1
  shift
  mkdir -p "$folder"
  { "$@"; echo $? > "$folder/status"; } | tee -a "$folder/train.txt"
  return "$(cat "$folder/status")"
}

if [ ! out/models/resnet20 -nt out/models/resnet20.c ]; then
  "$CC" -std=c11 -O3 -march=native -ffp-contract=off out/models/resnet20.c -lpthread -lm -o out/models/resnet20
fi
mkdir -p "$6/checkpoint"
start=0
while [ -f "$6/losses_$start.bin" ]; do start=$((start + 1)); done
if [ "$start" -lt "$2" ]; then
  RESNET_EPOCHS=$2 RESNET_START=$start RESNET_STEPS=$3 RESNET_FILES=$4 RESNET_TESTS=$5 RESNET_OUTPUT="$6/" \
    logged "$6" out/models/resnet20 +RTS --threads "$threads" --gpu off -RTS
fi
if [ $# -ge 7 ]; then
  logged "$7" "$py" -B examples/resnet20/resnet20_torch.py "$threads" "$7" --epochs "$2" --steps "$3" --train-files "$4" --test-images "$5"
fi
