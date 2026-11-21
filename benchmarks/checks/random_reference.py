"""The random values of lib/math_random.bend: the first word of Threefry-2x32 with 20 rounds for the key (seed, stream) and the counter (index, 0)."""

import numpy as np

ROTATIONS = [13, 15, 26, 6, 17, 29, 16, 24]


def threefry(seed, stream, index):
    keys = [seed, stream, seed ^ stream ^ 0x1BD11BDA]
    x, y = index, 0
    for r in range(20):
        if r % 4 == 0:
            x = (x + keys[(r // 4) % 3]) & 0xffffffff
            y = (y + keys[(r // 4 + 1) % 3] + r // 4) & 0xffffffff
        x = (x + y) & 0xffffffff
        left = ROTATIONS[r % 8]
        y = (((y << left) | (y >> (32 - left))) & 0xffffffff) ^ x
    return (x + keys[2]) & 0xffffffff


def words(count, seed, stream):
    """The words of the counters 0 to count - 1, vectorised over the counters."""
    keys = [np.uint32(seed), np.uint32(stream), np.uint32(seed ^ stream ^ 0x1BD11BDA)]
    x = np.arange(count, dtype=np.uint32)
    y = np.zeros(count, dtype=np.uint32)
    with np.errstate(over='ignore'):
        for r in range(20):
            if r % 4 == 0:
                x = x + keys[(r // 4) % 3]
                y = y + keys[(r // 4 + 1) % 3] + np.uint32(r // 4)
            x = x + y
            left = np.uint32(ROTATIONS[r % 8])
            y = ((y << left) | (y >> (np.uint32(32) - left))) ^ x
        return x + keys[2]


def uniform(count, seed, stream):
    """F.random_uniform([count], seed, stream): the top 24 bits of each word over 2^24."""
    return (words(count, seed, stream) >> np.uint32(8)).astype(np.float32) * np.float32(2.0 ** -24)


def below(count, bound, seed, stream):
    """F.random_below([count], bound, seed, stream): the top word of each word times the bound."""
    return ((words(count, seed, stream).astype(np.uint64) * np.uint64(bound)) >> np.uint64(32)).astype(np.float32)
