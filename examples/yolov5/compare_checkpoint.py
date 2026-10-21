"""Compare the checkpoint loaded in Bend with torch.load: names, shapes and every bit.

The tensors folded in Bend are compared the same way with fuse_conv_and_bn
computed in NumPy, whose square root is rounded to nearest. The report also
gives their distance in units in the last place from model.fuse(), whose
torch.sqrt is not always rounded to nearest.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / 'bend.cmd').is_file()
)
sys.path.insert(0, str(ROOT / '.tools/yolov5-official'))
import torch


def saved_model(weights):
    saved = torch.load(weights, map_location='cpu', weights_only=False)
    return saved['model'] if isinstance(saved, dict) and 'model' in saved else saved


def state_dict(weights):
    """The state dict of the saved model, or the saved state dict itself."""
    model = saved_model(weights)
    return model.state_dict() if hasattr(model, 'state_dict') else model


def folded_state(model):
    """The state dict with each batch normalization folded, in the order of fuse_conv_and_bn."""
    state = {name: tensor.detach().float().contiguous().numpy() for name, tensor in model.state_dict().items()}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.BatchNorm2d):
            continue
        prefix = name.removesuffix('.bn')
        gamma, beta, mean, variance = (state.pop(f'{name}.{field}') for field in ('weight', 'bias', 'running_mean', 'running_var'))
        state.pop(f'{name}.num_batches_tracked')
        root = np.sqrt(np.add(variance, np.float32(module.eps), dtype=np.float32))
        weights = state[f'{prefix}.conv.weight']
        state[f'{prefix}.conv.weight'] = np.multiply(weights, np.divide(gamma, root)[:, None, None, None], dtype=np.float32)
        state[f'{prefix}.conv.bias'] = np.subtract(beta, np.divide(np.multiply(gamma, mean), root), dtype=np.float32)
    return state


def units(expected, actual):
    """The largest distance of two FP32 tensors in units in the last place."""
    words = [np.asarray(value, '<f4').view(np.int32).astype(np.int64) for value in (expected, actual)]
    ordered = [np.where(word < 0, -2 ** 31 - word, word) for word in words]
    return int(np.abs(ordered[0] - ordered[1]).max(initial=0))


def compared(expected, loaded, ordered):
    """The mismatches of the tensors saved in the folder against the expected ones."""
    names = (loaded / 'names.txt').read_text(encoding='utf-8').splitlines()
    mismatches = []
    if (names if ordered else sorted(names)) != (list(expected) if ordered else sorted(expected)):
        mismatches.append(f'names differ: {len(names)} loaded, {len(expected)} expected')
    for name in names:
        if name not in expected:
            mismatches.append(f'{name}: not in the state dict')
            continue
        actual = np.load(loaded / f'{name}.npy', allow_pickle=False)
        if actual.dtype != np.dtype('<f4'):
            mismatches.append(f'{name}: dtype {actual.dtype}')
        elif actual.shape != expected[name].shape:
            mismatches.append(f'{name}: shape {list(actual.shape)} differs from {list(expected[name].shape)}')
        elif not same_bits(expected[name], actual):
            mismatches.append(f'{name}: values differ')
    return names, mismatches


def same_bits(expected, actual):
    """Equal bit for bit; a NaN position only needs a NaN."""
    if expected.shape != actual.shape:
        return False
    expected_words = expected.view(np.uint32).ravel()
    actual_words = actual.view(np.uint32).ravel()
    nan = np.isnan(expected.ravel())
    return bool(np.array_equal(expected_words[~nan], actual_words[~nan]) and np.isnan(actual.ravel()[nan]).all())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='data/weights/yolov5n-v7.0.pt')
    parser.add_argument('--loaded', default='out/results/checkpoint/official/bend')
    options = parser.parse_args()
    state = state_dict(ROOT / options.weights)
    # each tensor widened to FP32 as torch does, in row-major order
    expected = {name: tensor.detach().float().contiguous().numpy() for name, tensor in state.items()}
    loaded = ROOT / options.loaded
    names, mismatches = compared(expected, loaded, True)
    model = saved_model(ROOT / options.weights)
    folded, fuse_units = [], {}
    if hasattr(model, 'fuse'):
        model = model.float().eval()
        reference = folded_state(model)
        # the folded tensors are kept in the store in another order
        folded, folded_mismatches = compared(reference, loaded.parent / 'folded', False)
        mismatches += [f'folded {line}' for line in folded_mismatches]
        fused = {name: tensor.detach().float().contiguous().numpy() for name, tensor in model.fuse().state_dict().items()}
        fuse_units = {name: units(fused[name], reference[name]) for name in fused if units(fused[name], reference[name])}
    load_text = (loaded.parent / 'load.txt').read_text(encoding='utf-8')
    milliseconds = re.search(r'^load (\d+) ms$', load_text, re.M)
    report = dict(
        entries=len(expected),
        compared=len(names),
        folded=len(folded),
        units_from_model_fuse=fuse_units,
        elements=int(sum(int(np.prod(a.shape, dtype=np.int64)) for a in expected.values())),
        dtypes=sorted({str(tensor.dtype) for tensor in state.values()}),
        mismatches=mismatches,
        load_milliseconds=int(milliseconds[1]) if milliseconds else None,
        torch=torch.__version__,
    )
    (loaded.parent / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for line in mismatches[:20]:
        print(line)
    if mismatches:
        raise SystemExit(f'{len(mismatches)} mismatches; see {loaded.parent / "comparison.json"}')
    print(f'PASS: {len(names)} tensors equal bit for bit, {len(folded)} folded tensors equal bit for bit, load {report["load_milliseconds"]} ms')


if __name__ == '__main__':
    main()
