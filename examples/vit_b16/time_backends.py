"""Time the compiled ViT-B/16 and the official torchvision model on the same
input, alternately: one PyTorch warmup, then three rounds of ten forwards each.

python time_backends.py --threads N    (run.py vit --timing runs it)
"""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models import vit_b_16

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'out/results/vit_b16'
ROUNDS, SAMPLES = 3, 10


def stelliferous(threads):
    # Ten forwards in one process: VIT_EXTRA runs nine before the saved one.
    output = FOLDER / 'timing'
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, VIT_EXTRA=str(SAMPLES - 1), VIT_OUTPUT=str(output))
    environment.pop('VIT_LAYERS', None)
    run = subprocess.run([str(ROOT / 'out/models/vit_b16'), '+RTS', '--threads', str(threads), '--gpu', 'off', '-RTS'],
                         cwd=ROOT, env=environment, check=True, capture_output=True, text=True)
    times = [float(line.split()[1]) for line in run.stdout.splitlines() if line.startswith('forward ')]
    assert len(times) == SAMPLES, run.stdout
    return times


def pytorch(model, image):
    times = []
    with torch.inference_mode():
        for _ in range(SAMPLES):
            start = time.perf_counter()
            model(image)
            times.append((time.perf_counter() - start) * 1000.0)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=1)
    threads = parser.parse_args().threads
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model = vit_b_16()
    weights = ROOT / 'out/weights/vit_b_16/vit_b_16-c867db91.pth'
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.eval()
    image = torch.from_numpy(np.fromfile(FOLDER / 'input.bin', dtype='<f4').reshape(1, 3, 224, 224))
    with torch.inference_mode():
        model(image)
    samples = {'Stelliferous': [], 'PyTorch': []}
    for round_number in range(1, ROUNDS + 1):
        for name, measure in [('Stelliferous', lambda: stelliferous(threads)), ('PyTorch', lambda: pytorch(model, image))]:
            times = measure()
            samples[name] += times
            print(f'round {round_number} {name}: ' + ' '.join(f'{t:.1f}' for t in sorted(times)), flush=True)
    print(f'\nViT-B/16, {threads} thread(s), {ROUNDS} x {SAMPLES} forwards, median (min-max) in ms')
    for name, times in samples.items():
        print(f'{name:13} {statistics.median(times):6.1f} ({min(times):.1f}-{max(times):.1f})')
    (FOLDER / f'timing_{threads}t.json').write_text(json.dumps(dict(threads=threads, torch=torch.__version__, samples=samples), indent=2) + '\n')


if __name__ == '__main__':
    main()
