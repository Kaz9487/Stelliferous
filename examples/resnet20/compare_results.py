"""Compare the losses and test results that resnet20.bend and resnet20_torch.py wrote.

python compare_results.py EPOCHS BEND [BEND ...] --torch TORCH [--tolerance T]
Every Bend folder must hold the bits of the first. With --tolerance, the
first loss, a forward pass from the same weights, must be within T of
PyTorch's, relative to it. Later losses are listed: two FP32 trainings drift
apart after the first step, as PyTorch's own FP32 and float64 gradients do.
The comparison goes to BEND/comparison.json of the first folder.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read(folder, name, epoch):
    return np.fromfile(folder / f'{name}_{epoch}.bin', dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('epochs', type=int)
    parser.add_argument('bend', type=Path, nargs='+')
    parser.add_argument('--torch', type=Path, required=True)
    parser.add_argument('--tolerance', type=float)
    options = parser.parse_args()
    first, failures, epochs = options.bend[0], [], []
    for epoch in range(options.epochs):
        losses, correct = read(first, 'losses', epoch), read(first, 'correct', epoch)
        for other in options.bend[1:]:
            for name, values in (('losses', losses), ('correct', correct)):
                if read(other, name, epoch).tobytes() != values.tobytes():
                    failures.append(f'{other} {name}_{epoch}.bin differs from {first}')
        torch_losses, torch_correct = read(options.torch, 'losses', epoch), read(options.torch, 'correct', epoch)
        difference = float(np.max(np.abs(losses - torch_losses) / np.abs(torch_losses)))
        if epoch == 0 and options.tolerance is not None:
            first_difference = float(abs(losses[0] - torch_losses[0]) / abs(torch_losses[0]))
            if first_difference > options.tolerance:
                failures.append(f'the first loss is {first_difference:.2e} from PyTorch, above {options.tolerance:.0e}')
        epochs.append({'epoch': epoch, 'loss': float(losses.mean()), 'torch_loss': float(torch_losses.mean()),
                       'largest_relative_difference': difference, 'correct': int(correct[0]), 'torch_correct': int(torch_correct[0])})
        print(f'epoch {epoch}: loss {losses.mean():.4f} (PyTorch {torch_losses.mean():.4f}, largest difference {difference:.1e}), '
              f'test {int(correct[0])} (PyTorch {int(torch_correct[0])})')
    report = {'bend': [str(folder) for folder in options.bend], 'torch': str(options.torch), 'epochs': epochs, 'failures': failures}
    (first / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    for failure in failures:
        print('FAIL:', failure)
    if failures:
        sys.exit(1)
    print(f'PASS: {first / "comparison.json"}')


if __name__ == '__main__':
    main()
