"""Write a checkpoint of the storage formats and layouts the official one lacks."""

import sys
from pathlib import Path

import numpy as np

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / 'bend.cmd').is_file()
)
sys.path.insert(0, str(ROOT / '.tools/yolov5-official'))
import torch


def doubles():
    """FP64 values across the FP32 range, its edges and its rounding ties."""
    tiny = 2.0 ** -149
    big = float(np.finfo(np.float32).max)
    edges = [
        0.0, -0.0, float('inf'), -float('inf'), float('nan'), 5e-324, 1e-300, -1e-300,
        tiny, -tiny, tiny / 2, tiny / 2 * (1 + 2 ** -40), tiny * 1.5, tiny * 2.5,
        2.0 ** -126, 2.0 ** -126 * (1 - 2 ** -30), 1.0, 1 + 2 ** -24, 1 + 3 * 2 ** -24,
        1 + 2 ** -24 + 2 ** -50, big, -big, big * (1 + 2 ** -25), big * (1 + 2 ** -24), 1e39, -1e300,
    ]
    rng = np.random.default_rng(7)
    words = rng.integers(0, 2 ** 52, 20000, dtype=np.uint64)
    exponents = rng.integers(860, 1160, 20000, dtype=np.uint64)
    signs = rng.integers(0, 2, 20000, dtype=np.uint64)
    random = ((signs << np.uint64(63)) | (exponents << np.uint64(52)) | words).view(np.float64)
    return torch.tensor(np.concatenate([np.array(edges), random]), dtype=torch.float64)


def cases():
    """Entries of each storage class and of strided, offset and broadcast views."""
    return {
        'double.values': doubles(),
        'double.transposed': torch.arange(15, dtype=torch.float64).reshape(3, 5).t(),
        'float.transposed': torch.arange(24, dtype=torch.float32).reshape(4, 6).t(),
        'float.expanded': torch.arange(3, dtype=torch.float32).reshape(3, 1).expand(3, 4),
        'half.sliced': torch.arange(60, dtype=torch.float16).reshape(5, 12)[1:4, ::3],
        'float.scalar': torch.tensor(2.5),
        'float.empty': torch.zeros(0, 3),
    }


if __name__ == '__main__':
    target = ROOT / sys.argv[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cases(), target)
    print(f'wrote {target.relative_to(ROOT)}')
