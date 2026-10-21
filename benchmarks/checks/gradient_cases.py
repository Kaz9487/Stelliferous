"""Run the convolution gradients natively and compare them bit for bit with gradient_reference.py.

Every case runs with 1, 2, 4 and 8 workers. Every source tensor returned
by a gradient must retain the bits it received.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import numpy as np
from gradient_reference import F32, input_gradient, weight_gradient, bias_gradient, output_size, scalar_weight_gradient

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'out/checks/gradients.bend'
BUILD = ROOT / 'out/build/gradients'
REPORT = ROOT / 'out/results/gradients'
WORKERS = [1, 2, 4, 8]

# name, images (0: an input without the batch axis), channels, outputs, height, width, kernel, stride, padding
INPUT = [
    ('tails', 2, 5, 7, 6, 5, 3, 1, 1),
    ('pointwise', 2, 5, 7, 6, 5, 1, 1, 0),
    ('unpadded', 3, 2, 3, 7, 9, 3, 1, 0),
    ('full_padding', 1, 3, 2, 5, 4, 3, 1, 2),
    ('wide_kernel', 2, 1, 4, 9, 8, 5, 1, 2),
    ('single_image', 0, 4, 9, 8, 11, 3, 1, 1),
    ('layer_sized', 2, 32, 64, 40, 40, 3, 1, 1),
    ('stride_two', 2, 5, 7, 15, 14, 3, 2, 1),
    ('kernel_six', 1, 3, 8, 12, 16, 6, 2, 2),
    ('strided_pointwise', 2, 5, 7, 6, 5, 1, 2, 0),
    ('uncovered_tail', 1, 3, 5, 5, 7, 2, 2, 0),
    ('stride_three', 2, 3, 5, 10, 13, 3, 3, 1),
    ('unbatched_strided', 0, 5, 9, 8, 11, 3, 2, 1),
    ('empty_phases', 1, 1, 2, 8, 11, 1, 3, 0),
    ('large_padding', 2, 1, 3, 2, 3, 3, 2, 4),
    ('signed_zero', 1, 1, 2, 5, 7, 1, 2, 0),
    ('phase_tile_tail', 2, 9, 5, 19, 35, 5, 2, 2),
    ('phase_one_empty_axis', 1, 3, 9, 11, 17, 2, 5, 0),
]


# name, images (0: no batch axis), C, O, H, W, k, s, p, group.
PARAMETERS = [
    ('parameter_tails', 3, 5, 7, 6, 5, 3, 1, 1, 256),
    ('parameter_pointwise', 1, 5, 7, 6, 5, 1, 1, 0, 256),
    ('parameter_unpadded', 3, 2, 3, 7, 9, 3, 1, 0, 256),
    ('parameter_full_padding', 1, 3, 2, 5, 4, 3, 1, 2, 256),
    ('parameter_stride_two', 3, 2, 3, 11, 10, 3, 2, 1, 256),
    ('parameter_wide', 1, 2, 3, 3, 517, 1, 1, 0, 256),
    ('parameter_whole_rows', 3, 1, 2, 131, 5, 3, 1, 1, 256),
    ('parameter_no_batch', 0, 4, 9, 8, 11, 3, 2, 1, 256),
    ('parameter_group_one', 3, 2, 3, 4, 5, 3, 1, 1, 1),
    ('parameter_group_three', 3, 2, 3, 4, 5, 3, 1, 1, 3),
    ('parameter_group_128', 1, 1, 2, 7, 259, 1, 1, 0, 128),
    ('parameter_group_512', 3, 1, 2, 131, 5, 1, 1, 0, 512),
    ('parameter_group_1024', 3, 1, 2, 13, 79, 1, 1, 0, 1024),
    ('parameter_padding_nonfinite', 1, 1, 1, 3, 3, 3, 1, 2, 256),
    ('parameter_kernel_tiles_tail', 2, 1, 5, 14, 20, 3, 1, 1, 256),
    ('parameter_single_group_tail', 1, 1, 5, 2, 3, 3, 1, 1, 256),
    ('parameter_large_padding', 1, 1, 1, 2, 2, 3, 4294967295, 4294967295, 256),
]


def write(name, array):
    np.asarray(array, '<f4').tofile(REPORT / name)


def read(name, shape):
    return np.fromfile(REPORT / name, '<f4').reshape(shape)


def generate():
    rng = np.random.default_rng(20261011)
    steps = []
    for name, images, channels, outputs, height, width, kernel, stride, padding in INPUT:
        batch = max(images, 1)
        d_shape = [batch, outputs] + [output_size(size, kernel, stride, padding) for size in (height, width)]
        weights_shape = [outputs, channels, kernel, kernel]
        d = rng.normal(size=d_shape).astype(F32)
        weights = rng.normal(size=weights_shape).astype(F32)
        if name == 'signed_zero':
            d[...] = F32(-0.0)
            weights[0] = F32(-0.0)
            weights[1] = F32(0.0)
        if name == 'empty_phases':
            weights[0] = F32(np.inf)
            weights[1] = F32(-np.inf)
        write(f'{name}_d.bin', d)
        write(f'{name}_weights.bin', weights)
        loaded = d_shape if images else d_shape[1:]
        steps += [f'    {name}_{workers} : F.Tensor() <- input_case("{name}","{workers}",{loaded},{weights_shape},{stride},{padding},{height},{width},{workers})\n'
                  for workers in WORKERS]
    for name, images, channels, outputs, height, width, kernel, stride, padding, group in PARAMETERS:
        batch = max(images, 1)
        xs = [batch, channels, height, width]
        ds = [batch, outputs] + [output_size(size, kernel, stride, padding) for size in (height, width)]
        x = rng.normal(size=xs).astype(F32)
        d = rng.normal(size=ds).astype(F32)
        # Include signed zeros, tiny terms and cancellation without changing
        # the contract's finite-input arithmetic or hiding mismatched bits.
        x.ravel()[::17] = F32(-0.0)
        d.ravel()[::19] = F32(2 ** -140)
        if name == 'parameter_padding_nonfinite':
            d.ravel()[0] = F32(np.inf)
        write(f'{name}_x.bin', x)
        write(f'{name}_d.bin', d)
        loaded_x, loaded_d = (xs, ds) if images else (xs[1:], ds[1:])
        steps += [f'    {name}_{workers} : F.Tensor() <- parameter_case("{name}","{workers}",{loaded_x},{loaded_d},{kernel},{stride},{padding},{group},{workers})\n'
                  for workers in WORKERS]
    SOURCE.write_text('''import Base
import ../../lib/tensor_f32.bend as F
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/tensor_parameter_gradient.bend as Grouped

def folder() -> String:
  "out/results/gradients/"

def saved(+name: String,+tag: String,pair: F.Tensor() & F.Tensor()) -> IO(F.Tensor()):
  (gradient,weights) = pair
  do IO<F.Tensor()>:
    gradient : F.Tensor() <- F.save_raw(folder() ++ name ++ "_input_" ++ tag ++ ".bin",gradient)
    gradient : F.Tensor() <- F.expect(gradient)
    weights : F.Tensor() <- F.save_raw(folder() ++ name ++ "_weights_" ++ tag ++ ".bin",weights)
    F.expect(weights)

def input_case(+name: String,+tag: String,+d: List<&2,U32>,+weights: List<&2,U32>,+stride: U32,+padding: U32,+height: U32,+width: U32,+workers: U32) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    d : F.Tensor() <- F.load_raw(folder() ++ name ++ "_d.bin",d)
    weights : F.Tensor() <- F.load_raw(folder() ++ name ++ "_weights.bin",weights)
    saved(name,tag,Gradient.input(F.with_workers(d,workers),weights,stride,padding,height,width))

def bias_saved(+name: String,+tag: String,pair: F.Tensor() & F.Tensor()) -> IO(F.Tensor()):
  (gradient,d) = pair
  do IO<F.Tensor()>:
    gradient : F.Tensor() <- F.save_raw(folder() ++ name ++ "_bias_default_" ++ tag ++ ".bin",gradient)
    gradient : F.Tensor() <- F.expect(gradient)
    d : F.Tensor() <- F.save_raw(folder() ++ name ++ "_bias_d_" ++ tag ++ ".bin",d)
    F.expect(d)

def bias_group_saved(+name: String,+tag: String,pair: F.Tensor() & F.Tensor()) -> IO(F.Tensor()):
  (gradient,d) = pair
  do IO<F.Tensor()>:
    gradient : F.Tensor() <- F.save_raw(folder() ++ name ++ "_bias_" ++ tag ++ ".bin",gradient)
    gradient : F.Tensor() <- F.expect(gradient)
    F.expect(d)

def parameter_saved(+name: String,+tag: String,pair: F.Tensor() & (F.Tensor() & F.Tensor())) -> IO(F.Tensor()):
  (gradient,(x,d)) = pair
  do IO<F.Tensor()>:
    gradient : F.Tensor() <- F.save_raw(folder() ++ name ++ "_weight_" ++ tag ++ ".bin",gradient)
    gradient : F.Tensor() <- F.expect(gradient)
    x : F.Tensor() <- F.save_raw(folder() ++ name ++ "_returned_x_" ++ tag ++ ".bin",x)
    x : F.Tensor() <- F.expect(x)
    d : F.Tensor() <- F.save_raw(folder() ++ name ++ "_weight_d_" ++ tag ++ ".bin",d)
    d : F.Tensor() <- F.expect(d)
    bias_saved(name,tag,Gradient.bias(d))

def parameter_case(+name: String,+tag: String,+x: List<&2,U32>,+d: List<&2,U32>,k: U32,s: U32,p: U32,+group: U32,+workers: U32) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    x : F.Tensor() <- F.load_raw(folder() ++ name ++ "_x.bin",x)
    d : F.Tensor() <- F.load_raw(folder() ++ name ++ "_d.bin",d)
    d : F.Tensor() <- parameter_saved(name,tag,Grouped.weight(F.with_workers(x,workers),F.with_workers(d,workers),k,s,p,group))
    bias_group_saved(name,tag,Grouped.bias(d,group))

def main() -> IO(Unit):
  do IO<Unit>:
''' + ''.join(steps) + '    IO.print("done")\n')


def compare():
    failures = []
    for name, images, channels, outputs, height, width, kernel, stride, padding in INPUT:
        batch = max(images, 1)
        d = read(f'{name}_d.bin', [batch, outputs] + [output_size(size, kernel, stride, padding) for size in (height, width)])
        weights = read(f'{name}_weights.bin', [outputs, channels, kernel, kernel])
        with np.errstate(invalid='ignore'):
            expected = input_gradient(d, weights, height, width, stride, padding).view(np.uint32)
        for workers in WORKERS:
            gradient = np.fromfile(REPORT / f'{name}_input_{workers}.bin', np.uint32)
            returned = np.fromfile(REPORT / f'{name}_weights_{workers}.bin', np.uint32)
            if not (np.array_equal(gradient, expected.ravel()) and np.array_equal(returned, weights.view(np.uint32).ravel())):
                failures.append(f'{name} with {workers} workers')
    for name, images, channels, outputs, height, width, kernel, stride, padding, group in PARAMETERS:
        batch = max(images, 1)
        x = read(f'{name}_x.bin', [batch, channels, height, width])
        d = read(f'{name}_d.bin', [batch, outputs] + [output_size(size, kernel, stride, padding) for size in (height, width)])
        # The array reference pads the whole input; a padding too large for that uses the scalar form of the same contract.
        reference = scalar_weight_gradient if padding > max(height, width) * 1024 else weight_gradient
        expected = {'weight': reference(x, d, kernel, stride, padding, group),
                    'bias': bias_gradient(d, group), 'bias_default': bias_gradient(d),
                    'returned_x': x, 'weight_d': d, 'bias_d': d}
        for workers in WORKERS:
            for kind, values in expected.items():
                actual = np.fromfile(REPORT / f'{name}_{kind}_{workers}.bin', np.uint32)
                bits = values.view(np.uint32).ravel()
                if not np.array_equal(actual, bits):
                    mismatches = np.flatnonzero(actual != bits) if actual.size == bits.size else []
                    evidence = [(int(i), hex(int(bits[i])), hex(int(actual[i]))) for i in mismatches[:4]]
                    failures.append(f'{name} {kind} group={group} workers={workers}: {evidence}')
    return failures


def main():
    for folder in (SOURCE.parent, BUILD, REPORT):
        folder.mkdir(parents=True, exist_ok=True)
    generate()
    generated = BUILD / 'gradients.c'
    environment = dict(os.environ, BEND_NO_TELEMETRY='1')
    subprocess.run(['node', 'scripts/bend_launcher.mjs', str(SOURCE), '-o', str(generated)], cwd=ROOT, env=environment, check=True)
    cc = os.environ.get('CC') or shutil.which('clang-19') or shutil.which('clang')
    if not cc:
        raise SystemExit('The gradient cases need clang on PATH or CC.')
    subprocess.run([cc, '-O3', '-ffp-contract=off', '-std=c11', str(generated), '-lpthread', '-lm', '-o', str(BUILD / 'gradients')], check=True)
    subprocess.run([str(BUILD / 'gradients'), '--threads', str(max(WORKERS)), '--gpu', 'off'], cwd=ROOT, check=True, capture_output=True)
    failures = compare()
    (REPORT / 'status.json').write_text(json.dumps(dict(input_cases=len(INPUT), parameter_cases=len(PARAMETERS), workers=WORKERS, failures=failures), indent=2))
    assert not failures, failures
    print(f'{len(INPUT)} input and {len(PARAMETERS)} weight/bias cases equal the reference and retain their sources bit for bit with {WORKERS} workers.')


if __name__ == '__main__':
    main()
