"""Run the official torchvision ViT-B/16 on one image and save its input, block outputs and logits."""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / 'out/weights/vit_b_16/vit_b_16-c867db91.pth'
OUTPUT = ROOT / 'out/results/vit_b16'


def timed(model, image, threads, samples=11):
    torch.set_num_threads(threads)
    times = []
    with torch.inference_mode():
        for _ in range(samples):
            start = time.perf_counter()
            model(image)
            times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='bus')
    options = parser.parse_args()
    if not WEIGHTS.is_file():
        raise SystemExit(f'Missing {WEIGHTS}: see examples/vit_b16/README.md')
    model = vit_b_16()
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    model.eval()
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    image = weights.transforms()(Image.open(ROOT / f'data/images/{options.image}.jpg').convert('RGB')).unsqueeze(0)
    folder = OUTPUT / 'torch'
    folder.mkdir(parents=True, exist_ok=True)
    image.numpy().astype('<f4').tofile(OUTPUT / 'input.bin')

    outputs = {}
    def hook(name):
        def store(module, inputs, output):
            outputs[name] = output.detach().numpy().astype('<f4')
        return store
    model.encoder.ln.register_forward_hook(hook('encoder'))
    for index, layer in enumerate(model.encoder.layers):
        layer.register_forward_hook(hook(f'block_{index}'))
    torch.set_num_threads(1)
    with torch.inference_mode():
        logits = model(image)
        patches = model._process_input(image)
    outputs['logits'] = logits.numpy().astype('<f4')
    outputs['patches'] = patches.numpy().astype('<f4')
    for name, value in outputs.items():
        np.save(folder / f'{name}.npy', value)
    for layer_hooks in [model.encoder.ln, *model.encoder.layers]:
        layer_hooks._forward_hooks.clear()
    top = torch.topk(logits[0], 5)
    reference = dict(image=options.image, shape=list(image.shape), top5=top.indices.tolist(),
                     labels=[weights.meta['categories'][i] for i in top.indices.tolist()],
                     outputs={name: list(value.shape) for name, value in outputs.items()},
                     torch=torch.__version__,
                     forward_ms={str(threads): timed(model, image, threads) for threads in (1, 4)})
    (OUTPUT / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    print(json.dumps(reference, indent=2))


if __name__ == '__main__':
    main()
