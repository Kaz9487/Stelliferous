"""The FP32 sum of the library in NumPy: eight lanes of TwoSum and a fixed merge tree.

Value i of the summed sequence goes to lane i % 8; lanes past the end are left
as they are. Every addition and subtraction rounds to FP32 on its own.
"""
import numpy as np

F32 = np.float32


def two_sum(a, b):
    """The rounded sum and the error it left out."""
    hi = np.add(a, b, dtype=F32)
    z = np.subtract(hi, a, dtype=F32)
    return hi, np.add(np.subtract(a, np.subtract(hi, z, dtype=F32), dtype=F32), np.subtract(b, z, dtype=F32), dtype=F32)


def merge(ah, al, bh, bl):
    hi, error = two_sum(ah, bh)
    return two_sum(hi, np.add(np.add(al, bl, dtype=F32), error, dtype=F32))


def reduced_last(values, axes):
    """The values with the axes in increasing index moved last and joined into one."""
    values = np.asarray(values, dtype=F32)
    axes = sorted(axes)
    kept = [extent for index, extent in enumerate(values.shape) if index not in axes]
    moved = np.moveaxis(values, axes, range(values.ndim - len(axes), values.ndim))
    return moved.reshape(tuple(kept) + (-1,))


def sum8(values, axes=None):
    """The sum of every value, or over axes, as F.sum and F.sum_axes compute it."""
    values = np.asarray(values, dtype=F32)
    source = values.ravel() if axes is None else reduced_last(values, axes)
    count = source.shape[-1]
    hi = np.zeros(source.shape[:-1] + (8,), dtype=F32)
    lo = np.zeros_like(hi)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        for offset in range(0, count, 8):
            width = min(8, count - offset)
            hi[..., :width], error = two_sum(hi[..., :width], source[..., offset:offset + width])
            lo[..., :width] = np.add(lo[..., :width], error, dtype=F32)
        # pairs (0,1),(2,3),(4,5),(6,7), then pairs of those
        for width in (8, 4, 2):
            hi, lo = merge(hi[..., :width:2], lo[..., :width:2], hi[..., 1:width:2], lo[..., 1:width:2])
        return np.add(F32(0), np.add(hi[..., 0], lo[..., 0], dtype=F32), dtype=F32)
