"""Public Tensor integration across producers of different layouts.

A convolution lays its lanes on the channels, apply_indexed on the axis its rank
gives. The producers use exact identity kernels (one stride-2) to keep the
workload cheap to describe. The later convolution reuses the ordered FP32 oracle.
"""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from convolution_cases import oracle

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'out/checks/tensor_flow.bend'
REPORT = ROOT / 'out/results/integration/tensor_flow.json'

BEND_SOURCE = '''import Base
import ../../lib/tensor.bend as T
import ../../lib/tensor_f32.bend as F
import ../../lib/math_activation.bend as Act
import ../../lib/storage_buffer.bend as Storage

def input_value(index: U32,previous: F32) -> F32:
  (U32.to_f32((index % 561 % 31 : U32)) / 4.0 : F32)

# input_value + 0.25 in one pass, so the operand keeps the whole layout.
def shifted_value(index: U32,previous: F32) -> F32:
  (input_value(index,previous) + 0.25 : F32)

# The same values at every even row and column of a 66x34 image, so a
# stride-2 identity convolution reproduces input_value's 33x17 image.
def strided_value(index: U32,previous: F32) -> F32:
  +position = (index % 2244 : U32)
  (U32.to_f32(((position / 34 / 2 * 17 + position % 34 / 2) % 31 : U32)) / 4.0 : F32)

def identity_three(index: U32,previous: F32) -> F32:
  +index = index
  Bool.pick(F32,U32.is_eq((index % 9 : U32),4) && U32.is_eq((index / 9 % 64 : U32),(index / 576 : U32)),1.0,0.0)

def identity_five(index: U32,previous: F32) -> F32:
  +index = index
  Bool.pick(F32,U32.is_eq((index % 25 : U32),12) && U32.is_eq((index / 25 % 64 : U32),(index / 1600 : U32)),1.0,0.0)

def indexed_value(index: U32,previous: F32) -> F32:
  (previous + U32.to_f32((index % 13 : U32)) / 8.0 : F32)

def producer(large: Bool) -> F.Tensor():
  match large:
    case False{}:
      F.conv2d(F.with_workers(F.apply_indexed(~strided_value,9.0,F.zeros([2,64,66,34])),4),
        F.apply_indexed(~identity_three,8.0,F.zeros([64,64,3,3])),F.full([64],0.25),2,1,Act.identity())
    case True{}:
      F.conv2d(F.with_workers(F.apply_indexed(~input_value,4.0,F.zeros([2,64,33,17])),4),
        F.apply_indexed(~identity_five,8.0,F.zeros([64,64,5,5])),F.full([64],0.5),1,2,Act.identity())

def bits(values: List<&2,F32>) -> List<&2,U32>:
  match values:
    case []: []
    case F32{value} <> rest: U32{value} <> bits(rest)

def values(result: Result<&2,&2,T.Problem,List<&2,F32>>) -> List<&2,U32>:
  match result:
    case Done{result}: bits(result)
    case Fail{problem}: [4294967295]

def positive(value: F32) -> F32:
  (value * 0.5 : F32)

def finish(left: F.Tensor(),right: F.Tensor()) -> F.Tensor():
  Storage.bind_pair(~F.Tensor(),~F.Tensor(),~F.Tensor(),F.copy(left),left_a => left_b =>
    Storage.bind_pair(~F.Tensor(),~F.Tensor(),~F.Tensor(),F.copy(right),right_a => right_b =>
      F.upsample_nearest(F.conv2d(
        F.max_pool2d_with(F.concat(F.cons(F.apply_indexed(~indexed_value,4.0,F.apply(~positive,1.0,F.add(left_a,right_a))),
          F.single(F.sub_scalar(F.mul(left_b,right_b),0.5))),1),5,2,2),
        F.full([3,128,3,3],0.03125),F.full([3],F32.neg(1.0)),2,1,Act.identity()),2)))

def first_operand(whole: Bool) -> F.Tensor():
  match whole:
    case False{}: producer(False{})
    case True{}: F.with_workers(F.apply_indexed(~shifted_value,5.0,F.zeros([2,64,33,17])),4)

def flow(whole: Bool) -> List<&2,U32>:
  values(F.to_list(finish(first_operand(whole),producer(True{}))))

def padding_flow(whole: Bool,padding: U32) -> List<&2,U32>:
  values(F.to_list(F.conv2d(first_operand(whole),F.full([1,64,1,1],0.03125),F.full([1],1.0),8,padding,Act.identity())))

def empty_batch() -> List<&2,U32>:
  values(F.to_list(F.conv2d(F.with_workers(F.zeros([0,64,33,17]),4),F.full([1,64,3,3],0.03125),F.full([1],0.5),1,1,Act.identity())))

def main() -> List<&2,List<&2,U32>>:
  [flow(False{}),flow(True{}),padding_flow(False{},1),padding_flow(True{},1),padding_flow(False{},2),padding_flow(True{},2),
    padding_flow(False{},40),padding_flow(True{},40),empty_batch()]
'''


def expected():
    shape = (2, 64, 33, 17)
    x = (np.arange(np.prod(shape), dtype=np.uint32) % 561 % 31).astype(np.float32).reshape(shape) / 4
    left, right = x + np.float32(.25), x + np.float32(.5)
    indexed = ((np.arange(x.size, dtype=np.uint32) % 13).astype(np.float32) / 8).reshape(shape)
    first = (left + right) * np.float32(.5) + indexed
    second = left * right - np.float32(.5)
    joined = np.concatenate((first, second), axis=1)
    padded = np.pad(joined, ((0, 0), (0, 0), (2, 2), (2, 2)), constant_values=-np.inf)
    pooled = np.empty((2, 128, 17, 9), np.float32)
    for row in range(17):
        for col in range(9):
            pooled[:, :, row, col] = padded[:, :, row * 2:row * 2 + 5, col * 2:col * 2 + 5].max(axis=(2, 3))
    images = []
    for image in pooled:
        result = oracle(dict(shape=[128, 3, 17, 9, 3, 2, 1], x=image.ravel().tolist(),
                             w=[.03125] * (3 * 128 * 9), b=[-1.] * 3))
        output = np.asarray(result['bits'], np.uint32).view(np.float32).reshape(3, 9, 5)
        images.append(output.repeat(2, axis=1).repeat(2, axis=2))
    return np.asarray(images).view(np.uint32).ravel().tolist()


def padding_expected(padding):
    shape = (2, 64, 33, 17)
    x = (np.arange(np.prod(shape), dtype=np.uint32) % 561 % 31).astype(np.float32).reshape(shape) / 4 + np.float32(.25)
    return [bit for image in x for bit in oracle(dict(shape=[64, 1, 33, 17, 1, 8, padding],
        x=image.ravel().tolist(), w=[.03125] * 64, b=[1.]))['bits']]


def run():
    paddings = [1, 1, 2, 2, 40, 40]
    references = [expected()] * 2 + [padding_expected(padding) for padding in paddings] + [[]]
    results = []
    for threads in (1, 2, 4, 8):
        process = subprocess.run([str(ROOT / 'out/build/integration/tensor_flow'), '+RTS', '--threads', str(threads), '--gpu', 'off', '-RTS'],
                                 check=True, capture_output=True, text=True)
        cases = json.loads(process.stdout)
        if len(cases) != len(references):
            raise AssertionError((threads, len(cases), len(references)))
        for index, (actual, reference) in enumerate(zip(cases, references)):
            if actual != reference:
                differences = [(i, a, b) for i, (a, b) in enumerate(zip(actual, reference)) if a != b]
                raise AssertionError((threads, index, len(actual), len(reference), differences[:10]))
            case = 'flow' if index < 2 else 'empty_batch' if index == len(references) - 1 else 'padding_ge_kernel'
            results.append(dict(threads=threads, case=case, first_from_convolution=index % 2 == 0, bitwise_equal=True))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(dict(cases=results, flow=['convolution_and_indexed_producers', 'binary',
        'apply', 'global_apply_indexed', 'scalar', 'channel_concat', 'halo_pool', 'stride_two_conv', 'upsample', 'gather',
        'padding_ge_kernel', 'empty_batch']), indent=2))
    print('PASS: complete tensor flow from different producers, threads 1/2/4/8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run()
    else:
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        SOURCE.write_text(BEND_SOURCE, encoding='utf-8')
        print(SOURCE.relative_to(ROOT))
