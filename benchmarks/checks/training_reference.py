"""Reference for training steps of convolution and SiLU layers, in the fixed FP32 order.

Every operation is rounded to FP32; nothing is fused.

Forward: an output element folds from +0 over the input channel, the kernel
row and the kernel column increasing, of X0 * W, then adds its bias. sigmoid
and silu are the operation sequence of lib/math_fp32.bend.

SiLU derivative at z, with v = max(-|z|, -110) and t the sigmoid of v before
its final scaling: silu_derivative_root(v) where -2 <= v <= -0.5, the scaled
t * (1 + (v - v * t)) elsewhere; that value where z is negative, one minus it
elsewhere.

Loss: the mean of (y - target)^2, the squares added in row-major order from
+0. Its gradient is ((y - target) * 2) / count.

Update: w - rate * g.

    python benchmarks/checks/training_reference.py
"""
import numpy as np
from gradient_reference import F32, bias_gradient, input_gradient, output_size, same_bits, weight_gradient
from reduction_reference import sum8

U32 = np.uint32


def add(left, right):
    return np.add(left, right, dtype=F32)


def subtract(left, right):
    return np.subtract(left, right, dtype=F32)


def multiply(left, right):
    return np.multiply(left, right, dtype=F32)


def divide(left, right):
    return np.divide(left, right, dtype=F32)


def convolution(x, weights, bias, stride, padding):
    """x [N,C,H,W], weights [O,C,k,k] and bias [O] give [N,O,Ho,Wo]."""
    x, weights, bias = np.asarray(x, F32), np.asarray(weights, F32), np.asarray(bias, F32)
    kernel = weights.shape[2]
    height, width = (output_size(size, kernel, stride, padding) for size in x.shape[2:])
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    output = np.zeros((x.shape[0], weights.shape[0], height, width), F32)
    for c in range(x.shape[1]):
        for r in range(kernel):
            for t in range(kernel):
                samples = padded[:, c, r:r + stride * height:stride, t:t + stride * width:stride]
                np.add(output, multiply(samples[:, None], weights[:, c, r, t][None, :, None, None]), out=output)
    return add(output, bias[None, :, None, None])


def logistic(x):
    """The clamped input, the scaled power of two and the denominator of sigmoid."""
    x = np.asarray(x, F32)
    value = np.maximum(np.minimum(F32(18), x), F32(-110))
    biased = add(multiply(value, F32(-1.44269504088896341)), F32(12582912))
    n = subtract(biased, F32(12582912))
    r = add(subtract(multiply(n, F32(-0.693359375)), value), multiply(n, F32(0.00021219444005469058)))
    power = np.subtract(U32(1602224128), np.left_shift(biased.view(U32), U32(23)), dtype=U32)
    q = np.subtract(np.maximum(power, U32(545259520)), U32(536870912), dtype=U32).view(F32)
    z = multiply(r, r)
    high = add(add(F32(0.041666556149721146), multiply(F32(0.008363173343241215), r)), multiply(z, F32(0.0013926175888627768)))
    remainder = multiply(z, add(add(F32(0.5), multiply(F32(0.16666577756404877), r)), multiply(z, high)))
    base = add(F32(1), q)
    return value, power.view(F32), add(base, add(r, add(remainder, add(subtract(F32(1), base), q))))


def sigmoid(x):
    value, power, denominator = logistic(x)
    return multiply(divide(power, denominator), F32(2.0 ** -64))


def silu(x):
    x = np.asarray(x, F32)
    value, power, denominator = logistic(x)
    return np.where(x > F32(18), x, multiply(divide(multiply(value, power), denominator), F32(2.0 ** -64)))


def silu_derivative_root(value):
    """The derivative on [-2,-0.5]: its distance from its zero times a polynomial."""
    d = add(value, F32(1.2784645557403564))
    z = multiply(d, d)
    p0 = add(F32(0.2178117036819458), multiply(F32(0.14664886891841888), d))
    p1 = add(F32(0.01887473464012146), multiply(F32(-0.01522465143352747), d))
    p2 = add(F32(-0.006606101058423519), multiply(F32(0.0001401008339598775), d))
    p3 = add(F32(0.0007981295930221677), multiply(F32(0.00015176329179666936), d))
    tail = add(p3, multiply(z, F32(-4.2798692447831854e-05)))
    for p in (p2, p1, p0):
        tail = add(p, multiply(z, tail))
    return multiply(subtract(d, F32(1.2979282537628478e-08)), tail)


def silu_derivative(z):
    z = np.asarray(z, F32)
    value, power, denominator = logistic(-np.abs(z))
    scaled = divide(power, denominator)
    s = multiply(scaled, F32(2.0 ** -64))
    tail = multiply(multiply(scaled, add(F32(1), subtract(value, multiply(value, s)))), F32(2.0 ** -64))
    tail = np.where((value >= F32(-2)) & (value <= F32(-0.5)), silu_derivative_root(value), tail)
    return np.where(np.isnan(z), z, np.where(np.signbit(z), tail, subtract(F32(1), tail))).astype(F32)


def squared_error(y, target, mean):
    """(the loss, its gradient with respect to y): the sum of the squares, or their mean."""
    difference = subtract(y, target)
    squares = multiply(difference, difference).reshape(-1)
    total = F32(sum8(squares))
    doubled = multiply(difference, F32(2))
    if not mean:
        return total, doubled
    count = F32(difference.size)
    return divide(total, count), divide(doubled, count)


def mean_squared_error(y, target):
    return squared_error(y, target, True)


def spread(values):
    """A per-channel [C] seen as [1,C,1,1]."""
    return np.asarray(values, F32).reshape(1, -1, 1, 1)


def batch_norm_forward(x, gamma, beta, running_mean, running_variance, epsilon, momentum):
    """x [N,C,H,W]: (output, xhat, deviation, running mean, running variance) as BatchNorm.forward computes them."""
    x = np.asarray(x, F32)
    count = x.shape[0] * x.shape[2] * x.shape[3]
    mean = divide(sum8(x, (0, 2, 3)), F32(count))
    difference = subtract(x, spread(mean))
    squares = sum8(multiply(difference, difference), (0, 2, 3))
    deviation = divide(np.ones(x.shape[1], F32), np.sqrt(add(divide(squares, F32(count)), F32(epsilon))))
    xhat = multiply(difference, spread(deviation))
    keep = subtract(F32(1), F32(momentum))
    moved = lambda running, statistic: add(multiply(running, keep), multiply(statistic, F32(momentum)))
    return (add(multiply(xhat, spread(gamma)), spread(beta)), xhat, deviation,
            moved(running_mean, mean), moved(running_variance, divide(squares, F32(count - 1))))


def batch_norm_backward(dy, xhat, deviation, gamma):
    """(dx, dgamma, dbeta) as BatchNorm.backward computes them."""
    dy = np.asarray(dy, F32)
    count = F32(dy.shape[0] * dy.shape[2] * dy.shape[3])
    dbeta = sum8(dy, (0, 2, 3))
    dgamma = sum8(multiply(dy, xhat), (0, 2, 3))
    inner = subtract(subtract(multiply(dy, count), spread(dbeta)), multiply(xhat, spread(dgamma)))
    return multiply(inner, spread(divide(multiply(gamma, deviation), count))), dgamma, dbeta


def batch_norm_evaluate(x, gamma, beta, running_mean, running_variance, epsilon):
    deviation = divide(np.ones(len(gamma), F32), np.sqrt(add(running_variance, F32(epsilon))))
    xhat = multiply(subtract(x, spread(running_mean)), spread(deviation))
    return add(multiply(xhat, spread(gamma)), spread(beta))


def updated(parameter, gradient, rate):
    return subtract(parameter, multiply(F32(rate), gradient))


def step(x, layers, target, rate):
    """layers: (weights, bias, stride, padding) of each convolution followed by SiLU.

    Gives every intermediate value by name; layer i owns z{i}, y{i}, dy{i}, dz{i},
    dweights{i}, dbias{i}, weights{i} and bias{i} after the update.
    """
    values, inputs = {}, [np.asarray(x, F32)]
    for index, (weights, bias, stride, padding) in enumerate(layers, 1):
        values[f'z{index}'] = convolution(inputs[-1], weights, bias, stride, padding)
        values[f'y{index}'] = silu(values[f'z{index}'])
        inputs.append(values[f'y{index}'])
    values['loss'], gradient = mean_squared_error(inputs[-1], target)
    for index in range(len(layers), 0, -1):
        weights, bias, stride, padding = layers[index - 1]
        source = inputs[index - 1]
        values[f'dy{index}'] = gradient
        d = values[f'dz{index}'] = multiply(gradient, silu_derivative(values[f'z{index}']))
        values[f'dweights{index}'] = weight_gradient(source, d, weights.shape[2], stride, padding)
        values[f'dbias{index}'] = bias_gradient(d)
        gradient = input_gradient(d, weights, source.shape[2], source.shape[3], stride, padding)
        values[f'weights{index}'] = updated(weights, values[f'dweights{index}'], rate)
        values[f'bias{index}'] = updated(bias, values[f'dbias{index}'], rate)
    values['dx'] = gradient
    return values


def two_layer_case():
    """Conv(3->5,k3,s2,p1)-SiLU-Conv(5->7,k3,s1,p1)-SiLU on two 11x10 images."""
    rng = np.random.default_rng(20261011)
    x = rng.normal(0, 0.5, (2, 3, 11, 10)).astype(F32)
    layers = [(rng.normal(0, 0.12, (5, 3, 3, 3)).astype(F32), rng.normal(0, 0.02, 5).astype(F32), 2, 1),
              (rng.normal(0, 0.12, (7, 5, 3, 3)).astype(F32), rng.normal(0, 0.02, 7).astype(F32), 1, 1)]
    return x, layers, rng.normal(0, 0.3, (2, 7, 6, 5)).astype(F32), 0.01


def forward(x, layers):
    for weights, bias, stride, padding in layers:
        x = silu(convolution(x, weights, bias, stride, padding))
    return x


def learning_case():
    """Two Conv(32->32,k3,s1,p1)-BatchNorm-SiLU layers on two 40x40 images learning the output of a teacher of Conv-SiLU layers.

    The layers are large enough that the convolutions, their gradients and the
    channel sums run in parts on several workers. Each layer is (weights,
    bias, stride, padding, (gamma, beta, running mean, running variance)).
    """
    rng = np.random.default_rng(20261021)
    x = rng.normal(0, 0.5, (2, 32, 40, 40)).astype(F32)
    shapes = [(32, 32, 3, 3), (32, 32, 3, 3)]
    teacher = [(rng.normal(0, 0.1, shape).astype(F32), rng.normal(0, 0.1, shape[0]).astype(F32), 1, 1) for shape in shapes]
    student = [(rng.normal(0, 0.04, shape).astype(F32), rng.normal(0, 0.02, shape[0]).astype(F32), 1, 1,
                (rng.normal(1, 0.1, shape[0]).astype(F32), rng.normal(0, 0.1, shape[0]).astype(F32),
                 np.zeros(shape[0], F32), np.ones(shape[0], F32))) for shape in shapes]
    return x, student, forward(x, teacher), 0.5


EPSILON = 1e-5
MOMENTUM = 0.1


def normalized_step(x, layers, target, rate):
    """One step of Conv-BatchNorm-SiLU layers: (the loss, the layers after the update)."""
    inputs, kept = [np.asarray(x, F32)], []
    for weights, bias, stride, padding, (gamma, beta, mean, variance) in layers:
        z = convolution(inputs[-1], weights, bias, stride, padding)
        u, xhat, deviation, mean, variance = batch_norm_forward(z, gamma, beta, mean, variance, EPSILON, MOMENTUM)
        kept.append((u, xhat, deviation, mean, variance))
        inputs.append(silu(u))
    loss, gradient = mean_squared_error(inputs[-1], target)
    after = [None] * len(layers)
    for index in range(len(layers) - 1, -1, -1):
        weights, bias, stride, padding, (gamma, beta, _, _) = layers[index]
        u, xhat, deviation, mean, variance = kept[index]
        source = inputs[index]
        dz, dgamma, dbeta = batch_norm_backward(multiply(gradient, silu_derivative(u)), xhat, deviation, gamma)
        dweights = weight_gradient(source, dz, weights.shape[2], stride, padding)
        dbias = bias_gradient(dz)
        gradient = input_gradient(dz, weights, source.shape[2], source.shape[3], stride, padding)
        after[index] = (updated(weights, dweights, rate), updated(bias, dbias, rate), stride, padding,
                        (updated(gamma, dgamma, rate), updated(beta, dbeta, rate), mean, variance))
    return loss, after


def train(x, layers, target, rate, count):
    """The loss of every step and the layers after count steps."""
    losses = []
    for _ in range(count):
        loss, layers = normalized_step(x, layers, target, rate)
        losses.append(loss)
    return np.array(losses, F32), layers


def scalar_convolution(x, weights, bias, stride, padding):
    images, channels, height, width = x.shape
    outputs, kernel = weights.shape[0], weights.shape[2]
    rows, columns = output_size(height, kernel, stride, padding), output_size(width, kernel, stride, padding)
    result = np.zeros((images, outputs, rows, columns), F32)
    for n, o, u, v in np.ndindex(*result.shape):
        total = F32(0)
        for c, r, t in np.ndindex(channels, kernel, kernel):
            i, j = stride * u + r - padding, stride * v + t - padding
            sample = x[n, c, i, j] if 0 <= i < height and 0 <= j < width else F32(0)
            total = F32(total + F32(sample * weights[o, c, r, t]))
        result[n, o, u, v] = F32(total + bias[o])
    return result


def units(value, exact):
    """The distance of FP32 values from float64 values, in units of the last place of the exact value."""
    exact = np.asarray(exact, np.float64)
    return np.abs(np.asarray(value, np.float64) - exact) / np.spacing(np.abs(exact).astype(F32)).astype(np.float64)


def check():
    rng = np.random.default_rng(20261011)
    for images, channels, outputs, height, width, kernel, stride, padding in [
            (2, 3, 5, 7, 6, 3, 2, 1), (1, 5, 2, 6, 5, 1, 1, 0), (2, 2, 3, 5, 8, 3, 1, 2), (1, 1, 4, 9, 7, 5, 2, 2)]:
        x = rng.normal(size=(images, channels, height, width)).astype(F32)
        weights = rng.normal(size=(outputs, channels, kernel, kernel)).astype(F32)
        bias = rng.normal(size=outputs).astype(F32)
        assert same_bits(convolution(x, weights, bias, stride, padding), scalar_convolution(x, weights, bias, stride, padding))

    points = np.concatenate([np.linspace(-30, 30, 600001), rng.normal(0, 3, 200000)]).astype(F32)
    exact = points.astype(np.float64)
    logistic_exact = 1 / (1 + np.exp(-exact))
    assert units(sigmoid(points), logistic_exact).max() <= 2
    assert units(silu(points), exact * logistic_exact).max() <= 2
    derivative_exact = logistic_exact * (1 + exact * (1 - logistic_exact))
    derivative = silu_derivative(points)
    # The derivative crosses zero near -1.278, so that neighbourhood is held to an absolute bound.
    crossing = (points >= -2) & (points < -0.5)
    assert units(derivative[~crossing], derivative_exact[~crossing]).max() <= 4
    assert np.abs(derivative[crossing] - derivative_exact[crossing]).max() <= 1e-7
    edges = silu_derivative(np.array([np.inf, -np.inf, 0.0, -0.0], F32))
    assert same_bits(edges, np.array([1, -0.0, 0.5, 0.5], F32))

    x, layers, target, rate = two_layer_case()
    values = step(x, layers, target, rate)
    after = step(x, [(values[f'weights{i}'], values[f'bias{i}'], layers[i - 1][2], layers[i - 1][3]) for i in (1, 2)], target, rate)
    assert np.isfinite(values['loss']) and after['loss'] < values['loss']

    # The loss gradient against central differences of the same network in float64.
    def loss64(layers64):
        y = x.astype(np.float64)
        for weights, bias, stride, padding in layers64:
            kernel = weights.shape[2]
            rows, columns = (output_size(size, kernel, stride, padding) for size in y.shape[2:])
            padded = np.pad(y, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
            z = np.zeros((y.shape[0], weights.shape[0], rows, columns))
            for r in range(kernel):
                for t in range(kernel):
                    z += np.einsum('nchw,oc->nohw', padded[:, :, r:r + stride * rows:stride, t:t + stride * columns:stride], weights[:, :, r, t])
            z += bias[None, :, None, None]
            y = z / (1 + np.exp(-z))
        return np.mean((y - target.astype(np.float64)) ** 2)

    base = [(weights.astype(np.float64), bias.astype(np.float64), stride, padding) for weights, bias, stride, padding in layers]
    for index in (0, 1):
        for position in [(0, 0, 0, 0), (1, 2, 1, 2), (4, 1, 2, 0)]:
            moved = []
            for sign in (1, -1):
                trial = [(weights.copy(), bias, stride, padding) for weights, bias, stride, padding in base]
                trial[index][0][position] += sign * 1e-4
                moved.append(loss64(trial))
            slope = (moved[0] - moved[1]) / 2e-4
            assert abs(values[f'dweights{index + 1}'][position] - slope) <= 1e-5 * max(1.0, abs(slope))
    print('The training reference agrees with its scalar definitions, the ideal derivative and central differences.')


if __name__ == '__main__':
    check()
