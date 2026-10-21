"""Time one elementwise function of the public API in two checkouts, with the project's compilation flags.

The function is applied in a chain to one tensor. GELU and SiLU inputs stay at
20.0 and sigmoid converges to a normal fixed point, so no value becomes subnormal.
Times include process startup, allocation, one final read and teardown.

    python -B benchmarks/checks/function_timing.py --baseline OTHER_CHECKOUT --function gelu_tanh
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS = ['identity', 'relu', 'relu6', 'leaky_relu', 'hard_sigmoid', 'hard_swish',
             'sigmoid', 'silu', 'tanh', 'gelu_tanh']
FUNCTIONS += [name + '_derivative' for name in FUNCTIONS]


def call(function):
    activation = function.removesuffix('_derivative')
    if activation in ['identity', 'relu', 'relu6', 'leaky_relu', 'hard_sigmoid', 'hard_swish']:
        kind = 'activate_derivative' if function.endswith('_derivative') else 'activate'
        argument = 'Act.leaky_relu(0.125)' if activation == 'leaky_relu' else f'Act.{activation}()'
        return f'F.{kind}(tensor,{argument})'
    return f'F.{function}(tensor)'



def program(count, rounds, function):
    return f'''import Base
import ../../lib/tensor_f32.bend as F
import ../../lib/storage_buffer.bend as Storage
import ../../lib/math_activation.bend as Act

def chain(rounds: Nat,tensor: F.Tensor()) -> F.Tensor():
  match rounds:
    case 0n: tensor
    case 1n+rest: chain(rest,{call(function)})

def value(read: Array<F32> & F32) -> F32:
  (array,value) = read
  value

def finished(result: Result<&2,&1,F.Problem(),F.Storage()>) -> F32:
  match result:
    case Fail{{problem}}: 0.0
    case Done{{Storage.Buffer{{array,length,certificate}}}}: value(Array.get(F32,array,0))

def main() -> F32:
  finished(F.into_storage(chain({rounds}n,F.full([{count}],20.0))))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT)
    parser.add_argument('--function', choices=FUNCTIONS, default='gelu_tanh')
    parser.add_argument('--baseline-function', choices=FUNCTIONS, help='function to compare, defaults to the requested function')
    parser.add_argument('--count', default=65536, type=int)
    parser.add_argument('--rounds', default=1024, type=int)
    parser.add_argument('--samples', default=7, type=int)
    args = parser.parse_args()
    if args.count < 8 or args.rounds < 1 or args.samples < 3:
        parser.error('Use count >= 8, positive rounds and at least three samples')
    cc = os.environ.get('CC') or shutil.which('clang-19') or shutil.which('clang')
    if not cc:
        parser.error('Set CC to Clang')
    flags = ['-O3', '-march=native', '-ffp-contract=off', '-std=c11']
    report = ROOT / 'out/results/function_timing'
    report.mkdir(parents=True, exist_ok=True)
    binaries = {}
    names = {"baseline": args.baseline_function or args.function, "current": args.function}
    for name, root in [('baseline', args.baseline.resolve()), ('current', ROOT)]:
        source = root / f'out/checks/function_timing_{name}.bend'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(program(args.count, args.rounds, names[name]))
        build = root / 'out/build/function_timing' / name
        build.mkdir(parents=True, exist_ok=True)
        c = build / 'timing.c'
        subprocess.run(['node', 'scripts/bend_launcher.mjs', str(source), '-o', str(c)], cwd=root, check=True)
        binary = build / 'timing'
        subprocess.run([cc, *flags, str(c), '-lpthread', '-lm', '-o', str(binary)], cwd=root, check=True)
        subprocess.run([cc, *flags, '-S', str(c), '-o', str(build / 'timing.s')], cwd=root, check=True)
        binaries[name] = binary
    samples = {name: [] for name in binaries}
    for iteration in range(2 + args.samples):
        order = list(binaries) if iteration % 2 == 0 else list(reversed(binaries))
        for name in order:
            start = time.perf_counter_ns()
            result = subprocess.run([str(binaries[name]), '--threads', '1', '--gpu', 'off'],
                                    cwd=ROOT, capture_output=True, text=True, check=True)
            elapsed = time.perf_counter_ns() - start
            value = float(result.stdout.strip())
            function = names[name]
            if (function in ['gelu_tanh', 'silu'] and value != 20.0) or (function == 'sigmoid' and not 0.0 < value < 1.0) or not -100.0 < value < 100.0:
                raise SystemExit(f'{name} returned {result.stdout!r}, invalid {args.function} chain output')
            if iteration >= 2:
                samples[name].append(elapsed / (args.count * args.rounds))
    result = dict(function=args.function, baseline_function=names["baseline"], count=args.count, rounds=args.rounds, input=20.0, flags=flags,
                  protocol=f'Two warmups and {args.samples} alternating process samples, one worker, no chained subnormals',
                  samples_ns_per_element=samples,
                  median_ns_per_element={name: statistics.median(values) for name, values in samples.items()})
    (report / f'{args.function}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
