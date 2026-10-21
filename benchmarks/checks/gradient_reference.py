"""Reference for the convolution gradients, in the fixed FP32 order of the training contract.

Every product is rounded to FP32, then every addition; nothing is fused.

Weight gradient of weight (o,c,r,t): the terms D[n,o,u,v] * X0[n,c,s*u+r-p,s*v+t-p]
of one image, X0 being X extended by zeros, in row-major order of (u,v).
Consecutive whole output rows form a group of at most `group` terms; a longer
row is cut every `group` terms. A group is folded from +0. The groups of an
image are added by a tree of adjacent pairs, an odd last one moving up
unchanged, and the images by the same tree. The bias gradient adds
D[n,o,u,v] the same way.

Input gradient of input (n,c,i,j): folded from +0 over the output channel o
increasing, the kernel row r decreasing and the kernel column t decreasing,
of the terms D0[n,o,(i+p-r)/s,(j+p-t)/s] * W[o,c,r,t] whose two quotients are
whole, D0 being D extended by zeros, and +0 added last. At stride 1 it is the forward
convolution of D with the weights transposed in (o,c) and flipped in (r,t),
under padding k-1-p and a zero bias.

    python benchmarks/checks/gradient_reference.py
"""
import numpy as np

F32 = np.float32
GROUP = 256


def output_size(size, kernel, stride, padding):
    return (size + 2 * padding - kernel) // stride + 1


def fold(terms):
    """The last axis added in order from +0."""
    start = np.zeros(terms.shape[:-1] + (1,), F32)
    return np.add.accumulate(np.concatenate([start, terms], axis=-1), axis=-1, dtype=F32)[..., -1]


def pair_tree(values):
    values = list(values)
    while len(values) > 1:
        paired = [np.add(values[i], values[i + 1], dtype=F32) for i in range(0, len(values) - 1, 2)]
        values = paired + values[len(values) & ~1:]
    return values[0]


def group_ranges(height, width, group):
    """The position ranges of the groups of a height x width output, in order."""
    if width > group:
        return [(row * width + start, row * width + min(start + group, width))
                for row in range(height) for start in range(0, width, group)]
    rows = group // width
    return [(row * width, min(row + rows, height) * width) for row in range(0, height, rows)]


def grouped(terms, height, width, group):
    """Terms [..., height * width] of one image, folded by group and joined by the tree."""
    return pair_tree(fold(terms[..., first:last]) for first, last in group_ranges(height, width, group))


def patches(image, kernel, stride, padding, height, width):
    """[C * k * k, height * width]: the input sample of every weight at every output position."""
    padded = np.pad(image, ((0, 0), (padding, padding), (padding, padding)))
    return np.stack([padded[c, r:r + stride * height:stride, t:t + stride * width:stride].reshape(-1)
                     for c in range(image.shape[0]) for r in range(kernel) for t in range(kernel)])


def weight_gradient(x, d, kernel, stride, padding, group=GROUP):
    """x [N,C,H,W] and d [N,O,Ho,Wo] give [O,C,k,k]."""
    x, d = np.asarray(x, F32), np.asarray(d, F32)
    images, outputs, height, width = d.shape
    per_image = []
    for n in range(images):
        samples = patches(x[n], kernel, stride, padding, height, width)
        per_image.append(np.stack([
            grouped(np.multiply(d[n, o].reshape(1, -1), samples, dtype=F32), height, width, group)
            for o in range(outputs)]))
    return pair_tree(per_image).reshape(outputs, x.shape[1], kernel, kernel)


def bias_gradient(d, group=GROUP):
    """d [N,O,Ho,Wo] gives [O]."""
    d = np.asarray(d, F32)
    images, outputs, height, width = d.shape
    return pair_tree(grouped(d[n].reshape(outputs, -1), height, width, group) for n in range(images))


def input_gradient(d, weights, height, width, stride, padding):
    """d [N,O,Ho,Wo] and weights [O,C,k,k] give [N,C,height,width]."""
    d, weights = np.asarray(d, F32), np.asarray(weights, F32)
    images, outputs, output_height, output_width = d.shape
    kernel = weights.shape[2]

    def axis(size, output, tap):
        """The inputs fed by this tap, their output positions, and which of those exist."""
        inputs = np.arange((tap - padding) % stride, size, stride)
        positions = (inputs + padding - tap) // stride
        return inputs, positions.clip(0, output - 1), (positions >= 0) & (positions < output)

    gradient = np.zeros((images, weights.shape[1], height, width), F32)
    for o in range(outputs):
        for r in reversed(range(kernel)):
            rows, u, row_inside = axis(height, output_height, r)
            for t in reversed(range(kernel)):
                columns, v, column_inside = axis(width, output_width, t)
                if not (len(rows) and len(columns)):
                    continue
                extended = np.where(row_inside[:, None] & column_inside[None, :], d[:, o][:, u][:, :, v], F32(0))
                terms = np.multiply(extended[:, None], weights[o, :, r, t][None, :, None, None], dtype=F32)
                target = gradient[:, :, rows[0]:height:stride, columns[0]:width:stride]
                np.add(target, terms, out=target)
    return np.add(gradient, F32(0), dtype=F32)


# The same definitions one term at a time, to check the array forms above.

def scalar_tree(values):
    values = list(values)
    while len(values) > 1:
        values = [F32(values[i] + values[i + 1]) for i in range(0, len(values) - 1, 2)] + values[len(values) & ~1:]
    return values[0]


def scalar_grouped(term, height, width, group):
    folded = []
    for first, last in group_ranges(height, width, group):
        total = F32(0)
        for position in range(first, last):
            total = F32(total + term(position // width, position % width))
        folded.append(total)
    return scalar_tree(folded)


def scalar_weight_gradient(x, d, kernel, stride, padding, group):
    images, outputs, height, width = d.shape
    channels, input_height, input_width = x.shape[1:]

    def sample(n, c, row, column):
        return x[n, c, row, column] if 0 <= row < input_height and 0 <= column < input_width else F32(0)

    return np.array([[[[scalar_tree(
        scalar_grouped(lambda u, v: F32(d[n, o, u, v] * sample(n, c, stride * u + r - padding, stride * v + t - padding)),
                       height, width, group) for n in range(images))
        for t in range(kernel)] for r in range(kernel)] for c in range(channels)] for o in range(outputs)], F32)


def scalar_bias_gradient(d, group):
    images, outputs, height, width = d.shape
    return np.array([scalar_tree(scalar_grouped(lambda u, v: d[n, o, u, v], height, width, group)
                                 for n in range(images)) for o in range(outputs)], F32)


def scalar_input_gradient(d, weights, height, width, stride, padding):
    images, outputs, output_height, output_width = d.shape
    channels, kernel = weights.shape[1], weights.shape[2]
    gradient = np.zeros((images, channels, height, width), F32)
    for n, c, i, j in np.ndindex(*gradient.shape):
        total = F32(0)
        for o in range(outputs):
            for r in reversed(range(kernel)):
                for t in reversed(range(kernel)):
                    if (i + padding - r) % stride or (j + padding - t) % stride:
                        continue
                    u, v = (i + padding - r) // stride, (j + padding - t) // stride
                    inside = 0 <= u < output_height and 0 <= v < output_width
                    total = F32(total + F32((d[n, o, u, v] if inside else F32(0)) * weights[o, c, r, t]))
        gradient[n, c, i, j] = F32(total + F32(0))
    return gradient


def same_bits(left, right):
    return np.array_equal(np.asarray(left, F32).view(np.uint32), np.asarray(right, F32).view(np.uint32))


def check():
    rng = np.random.default_rng(20261011)
    # images, channels, outputs, height, width, kernel, stride, padding, group
    shapes = [(2, 3, 5, 11, 10, 3, 2, 1, 256), (3, 2, 3, 7, 9, 3, 1, 1, 256), (2, 5, 7, 6, 5, 1, 1, 0, 256),
              (1, 2, 2, 9, 12, 3, 1, 1, 8), (3, 1, 2, 8, 13, 3, 2, 1, 4), (2, 1, 3, 2, 3, 3, 2, 4, 256),
              (1, 2, 3, 5, 5, 2, 2, 0, 256), (2, 2, 2, 5, 4, 3, 1, 2, 256), (5, 1, 1, 6, 6, 6, 2, 2, 5)]
    for images, channels, outputs, height, width, kernel, stride, padding, group in shapes:
        output_height, output_width = (output_size(size, kernel, stride, padding) for size in (height, width))
        x = rng.normal(size=(images, channels, height, width)).astype(F32)
        weights = rng.normal(size=(outputs, channels, kernel, kernel)).astype(F32)
        d = rng.normal(size=(images, outputs, output_height, output_width)).astype(F32)
        assert same_bits(weight_gradient(x, d, kernel, stride, padding, group),
                         scalar_weight_gradient(x, d, kernel, stride, padding, group))
        assert same_bits(bias_gradient(d, group), scalar_bias_gradient(d, group))
        assert same_bits(input_gradient(d, weights, height, width, stride, padding),
                         scalar_input_gradient(d, weights, height, width, stride, padding))
        # Small whole numbers are exact in FP32, so both gradients must be transposes
        # of the forward map: <Y,D> = <X,dX> = <W,dW>.
        x, weights, d = (rng.integers(-4, 5, size=array.shape).astype(F32) for array in (x, weights, d))
        whole = lambda array: array.astype(np.int64)
        samples = np.stack([patches(image, kernel, stride, padding, output_height, output_width) for image in x])
        y = np.einsum('ok,nkp->nop', whole(weights.reshape(outputs, -1)), whole(samples))
        paired = int((y * whole(d.reshape(images, outputs, -1))).sum())
        assert paired == int((whole(x) * whole(input_gradient(d, weights, height, width, stride, padding))).sum())
        assert paired == int((whole(weights) * whole(weight_gradient(x, d, kernel, stride, padding, group))).sum())
    print(f'{len(shapes)} shapes: the array forms equal the scalar definitions bit for bit, '
          'and both gradients transpose the forward map.')


if __name__ == '__main__':
    check()
