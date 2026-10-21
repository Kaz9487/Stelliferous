"""Convolution gradients and same-shape forward, using the existing phase clock.

Pinned compiler, ordinary native flags, existing I/O clock effects only.
The report records every sample (milliseconds), median and forward multiple.
"""
from pathlib import Path
import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import numpy as np
from gradient_reference import F32, output_size, weight_gradient, bias_gradient, input_gradient

ROOT = Path(__file__).resolve().parents[2]
GROUPS = [128, 256, 512, 1024]
# name, N, C, O, H, W, k, s, p
SHAPES = [('image320', 1, 3, 4, 320, 320, 3, 1, 1),
          ('pointwise40', 1, 8, 16, 40, 40, 1, 1, 0),
          ('batch_tails', 3, 5, 7, 17, 13, 3, 2, 1)]
REPORT = ROOT / 'out/results/parameter_gradients'
SOURCE = ROOT / 'out/checks/parameter_gradient_benchmark.bend'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='store_true', help='Measure input gradients and same-shape forward')
    parser.add_argument('--workers', type=int, nargs='+', default=[1], help='Tensor and runtime worker counts')
    parser.add_argument('--shape', type=lambda value: tuple(map(int, value.split(','))), action='append', help='N,C,O,H,W,k,s,p; repeat for more shapes')
    options = parser.parse_args()
    shapes = [('stride_two', 2, 5, 7, 31, 30, 3, 2, 1),
              ('kernel_six', 1, 3, 8, 48, 64, 6, 2, 2),
              ('stride_three', 2, 3, 5, 10, 13, 3, 3, 1)] if options.input else SHAPES
    if options.shape:
        if any(len(shape) != 8 or min(shape[:7]) < 1 or shape[7] < 0 for shape in options.shape):
            parser.error('--shape requires eight dimensions, positive except padding')
        shapes = [(f'shape_{index}', *shape) for index, shape in enumerate(options.shape)]
    if min(options.workers) < 1:
        parser.error('--workers must be positive')
    REPORT.mkdir(parents=True, exist_ok=True)
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    SOURCE.write_text('''import Base
import ../../lib/tensor_f32.bend as F
import ../../lib/tensor.bend as T
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor_parameter_gradient.bend as G
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/kernel_input_gradient.bend as InputKernel
import ../../lib/math_activation.bend as Act

# Reuse the existing file/phase-clock effects; all tensor arithmetic is Bend.
def Lab.param(i: U32) -> IO(U32):
  import "../../benchmarks/convolution/io.c"
def Lab.reps() -> IO(U32):
  import "../../benchmarks/convolution/io.c"
def Lab.load(i: U32,count: U32) -> IO(Array<F32>):
  import "../../benchmarks/convolution/io.c"
def Lab.start() -> IO(Unit):
  import "../../benchmarks/convolution/io.c"
def Lab.reordered(a: Array<U32>) -> IO(Array<U32>):
  import "../../benchmarks/convolution/io.c"
def Lab.computed(a: Array<U32>) -> IO(Array<U32>):
  import "../../benchmarks/convolution/io.c"
def Lab.finish(a: Array<F32>) -> IO(Array<F32>):
  import "../../benchmarks/convolution/io.c"
def Lab.save(a: Array<F32>,count: U32) -> IO(Unit):
  import "../../benchmarks/convolution/io.c"

def weight_result(pair: F.Tensor() & (F.Tensor() & F.Tensor())) -> F.Tensor():
  (gradient,(x,d)) = pair
  gradient

def bias_result(pair: F.Tensor() & F.Tensor()) -> F.Tensor():
  (gradient,d) = pair
  gradient

def stored(result: Result<&2,&1,T.Problem,Storage.Buffer<F32>>) -> IO(Unit):
  match result:
    case Fail{problem}: IO.print("failed")
    case Done{Storage.Buffer{storage,+length,certificate}}:
      do IO<Unit>:
        storage : Array<F32> <- Lab.finish(storage)
        Lab.save(storage,U32.from_nat(length))

def measured(result: F.Tensor()) -> IO(Unit):
  do IO<Unit>:
    result : F.Tensor() <- F.expect(result)
    stored(T.into_storage(~F32,result))

type Case is Data:
  Case{kind: U32,n: U32,c: U32,o: U32,h: U32,w: U32,k: U32,s: U32,p: U32,oh: U32,ow: U32,g: U32,workers: U32}

def selected(kind: U32,x: F.Tensor(),d: F.Tensor(),weights: F.Tensor(),bias: F.Tensor(),+k: U32,+s: U32,+p: U32,+g: U32,+h: U32,+w: U32,+n: U32,+c: U32,+o: U32,+oh: U32,+ow: U32) -> IO(Unit):
  match kind:
    case 0: measured(weight_result(G.weight(x,d,k,s,p,g)))
    case 1: measured(bias_result(G.bias(d,g)))
    case 3: measured(bias_result(Gradient.input(d,weights,s,p,h,w)))
    case kind: measured(F.conv2d(x,weights,bias,s,p,Act.identity()))

def run(context: Case) -> IO(Unit):
  Case{kind,+n,+c,+o,+h,+w,+k,s,p,+oh,+ow,g,+workers} = context
  do IO<Unit>:
    x : F.Tensor() <- F.load_raw("x.bin",[n,c,h,w])
    d : F.Tensor() <- F.load_raw("d.bin",[n,o,oh,ow])
    weights : F.Tensor() <- F.load_raw("w.bin",[o,c,k,k])
    bias : F.Tensor() <- F.load_raw("b.bin",[o])
    Lab.start()
    selected(kind,F.with_workers(x,workers),F.with_workers(d,workers),weights,bias,k,s,p,g,h,w,n,c,o,oh,ow)

def repeat(remaining: Nat,+context: Case) -> IO(Unit):
  match remaining:
    case 0n: IO.print("done")
    case 1n+rest:
      do IO<Unit>:
        run(context)
        repeat(rest,context)

def main() -> IO(Unit):
  do IO<Unit>:
    kind : U32 <- Lab.param(0)
    n : U32 <- Lab.param(1)
    c : U32 <- Lab.param(2)
    o : U32 <- Lab.param(3)
    h : U32 <- Lab.param(4)
    w : U32 <- Lab.param(5)
    k : U32 <- Lab.param(6)
    s : U32 <- Lab.param(7)
    p : U32 <- Lab.param(8)
    oh : U32 <- Lab.param(9)
    ow : U32 <- Lab.param(10)
    g : U32 <- Lab.param(11)
    workers : U32 <- Lab.param(12)
    reps : U32 <- Lab.reps()
    loaded : Array<F32> <- Lab.load(0,1)
    dummy : Array<U32> <- Lab.reordered(Array.new(U32,0n,0))
    dummy : Array<U32> <- Lab.computed(dummy)
    repeat(U32.to_nat(reps),Case{kind,n,c,o,h,w,k,s,p,oh,ow,g,workers})
''')
    np.ones(1,F32).tofile(REPORT/'input.bin')
    generated = SOURCE.with_suffix('.c')
    subprocess.run(['node', 'scripts/bend_launcher.mjs', str(SOURCE), '-o', str(generated)], cwd=ROOT, check=True)
    cc = os.environ.get('CC') or shutil.which('clang-19') or shutil.which('clang')
    binary = REPORT / 'benchmark'
    flags = ['-O3', '-march=native', '-ffp-contract=off', '-std=c11']
    subprocess.run([cc, *flags, str(generated), '-lpthread', '-lm', '-o', str(binary)], check=True)
    rng = np.random.default_rng(20261011)
    rows = []
    for name, n, ci, co, h, w, k, s, p in shapes:
        oh, ow = output_size(h,k,s,p), output_size(w,k,s,p)
        x = rng.normal(size=[n,ci,h,w]).astype(F32)
        d = rng.normal(size=[n,co,oh,ow]).astype(F32)
        for kind, values in [('x',x),('d',d),('w',rng.normal(size=[co,ci,k,k]).astype(F32)),('b',np.zeros(co,F32))]:
            values.tofile(REPORT / f'{kind}.bin')
        kinds = [('forward',[0]),('input',[0])] if options.input else [('forward',[0]),('weight',GROUPS),('bias',GROUPS)]
        for workers in options.workers:
            samples = {}
            for kind, groups in kinds:
                for group in groups:
                    native_kind = {'weight':0,'bias':1,'forward':2,'input':3}[kind]
                    np.array([native_kind,n,ci,co,h,w,k,s,p,oh,ow,group,workers],'<u4').tofile(REPORT/'config.bin')
                    (REPORT/'bend_samples.jsonl').unlink(missing_ok=True)
                    subprocess.run([str(binary),'--threads',str(workers),'--gpu','off'],cwd=REPORT,env=dict(os.environ,CONV_REPS='16' if options.input else '4'),check=True,capture_output=True)
                    samples[(kind,group)] = [json.loads(line)['total_ms'] for line in (REPORT/'bend_samples.jsonl').read_text().splitlines()][1:]
                    if kind != 'forward':
                        expected = input_gradient(d, np.fromfile(REPORT/'w.bin',F32).reshape(co,ci,k,k),h,w,s,p) if kind == 'input' else weight_gradient(x,d,k,s,p,group) if kind == 'weight' else bias_gradient(d,group)
                        actual = np.fromfile(REPORT/'bend.bin',np.uint32)
                        assert np.array_equal(actual,expected.view(np.uint32).ravel()), (name,kind,group,workers)
                    print(name,kind,group,workers,samples[(kind,group)],flush=True)
            forward = statistics.median(samples[('forward',0)])
            if options.input:
                timing = statistics.median(samples[('input',0)])
                row = dict(name=name,shape=[n,ci,co,h,w,k,s,p],workers=workers,input_ms=timing,forward_ms=forward,
                           input_forward_multiple=timing/forward,input_samples=samples[('input',0)],forward_samples=samples[('forward',0)])
                rows.append(row)
                continue
            for group in GROUPS:
                weight, bias = (statistics.median(samples[(kind,group)]) for kind in ('weight','bias'))
                rows.append(dict(name=name,shape=[n,ci,co,h,w,k,s,p],workers=workers,group=group,weight_ms=weight,bias_ms=bias,forward_ms=forward,
                                 weight_forward_multiple=weight/forward,bias_forward_multiple=bias/forward,
                                 samples={kind:samples[(kind,group)] for kind in ('weight','bias')},forward_samples=samples[('forward',0)]))
    report = dict(machine=platform.platform(),compiler=cc,flags=flags,workers=options.workers,
                  clock=f'Existing convolution/io.c phase clock, nanoseconds; source loading and raw output excluded; one warmup and {15 if options.input else 3} measured samples',rows=rows)
    (REPORT/('input_summary.json' if options.input else 'summary.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
