"""Measure the unit prices of the task planner on one thread and compare them
with lib/default_kernel_cost.bend.

Run on Linux from the repository root: python3 -B scripts/toolchain_rules/planner_prices.py
One probe program times four chains, each against the same chain without the
priced work:
  row   1x1 convolutions of 256 to 256 channels over 40 by 40 positions with
        zero weights; one row is one output cell's multiply-add of eight lanes
        over one input channel, so a call is positions * 32 * 256 rows
  step  a scalar loop of four linked chains, eight additions and
        multiplications an iteration, as addressing counts them
  byte  copies of a 64 channel 80 by 80 tensor, its 1.6 MB of tiles copied once
  move  transposes of the two spatial axes of that tensor, each of its 51200
        cells moved once through a view
wake_rounds.py checks the wake round (rule H11).
"""
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BUILD = ROOT / 'out/toolchain_rules'
FLAGS = ['-O3', '-march=native', '-ffp-contract=off', '-std=c11']
REPEATS = 5
CHANNELS, SIDE, CALLS = 256, 40, 60
COPY_CHANNELS, COPY_SIDE, COPIES = 64, 80, 600
MOVES = 200
ITERATIONS, STEPS_PER_ITERATION = 400000000, 8
NAMES = ('row', 'step', 'byte', 'move')

SOURCE = '''import Base
import ../../lib/tensor_f32.bend as F
import ../../lib/math_activation.bend as Act

def convolved(count: Nat, +channels: U32, priced: Bool, x: F.Tensor()) -> F.Tensor():
  match count priced:
    case 1n+rest True{}: convolved(rest, channels, True{}, F.conv2d(x, F.zeros([channels, channels, 1, 1]), F.zeros([channels]), 1, 0, Act.Identity{}))
    case 1n+rest False{}: convolved(rest, channels, False{}, x)
    case count priced: x

def kept(pair: F.Tensor() & F.Tensor()) -> F.Tensor():
  (first, second) = pair
  first

def copied(count: Nat, priced: Bool, x: F.Tensor()) -> F.Tensor():
  match count priced:
    case 1n+rest True{}: copied(rest, True{}, kept(F.copy(x)))
    case 1n+rest False{}: copied(rest, False{}, x)
    case count priced: x

def transposed(count: Nat, priced: Bool, x: F.Tensor()) -> F.Tensor():
  match count priced:
    case 1n+rest True{}: transposed(rest, True{}, F.transpose(x, [0, 1, 3, 2]))
    case 1n+rest False{}: transposed(rest, False{}, x)
    case count priced: x

def stepped(n: Nat, +a: U32, +b: U32, +c: U32, +d: U32) -> U32:
  match n:
    case 0n: (a + b + c + d : U32)
    case 1n+rest: stepped(rest, (a * 3 + 1 : U32), (b + a + 1 : U32), (c * b + 2 : U32), (d + c * 7 : U32))

def total(x: F.Tensor()) -> String:
  F32.show(F.item_or(F.sum(x), 0.0))

def timed(label: String, value: Unit -> String, next: IO(Unit)) -> IO(Unit):
  IO.bind(Nat, Unit, IO.now(), t0 =>
    IO.bind(Unit, Unit, IO.print(value(Unit{})), u =>
      IO.bind(Nat, Unit, IO.now(), t1 =>
        IO.bind(Unit, Unit, IO.print(label ++ " " ++ Nat.show(Nat.sub(t1, t0))), v => next))))

def main() -> IO(Unit):
  @BODY@
'''


def program():
    """Each chain with and without its priced work, REPEATS times."""
    chains = [
        ('row', f'u => total(convolved(U32.to_nat({CALLS}), {CHANNELS}, @P, F.zeros([1, {CHANNELS}, {SIDE}, {SIDE}])))'),
        ('byte', f'u => total(copied(U32.to_nat({COPIES}), @P, F.zeros([1, {COPY_CHANNELS}, {COPY_SIDE}, {COPY_SIDE}])))'),
        ('step', f'u => U32.show(stepped(U32.to_nat(@N), 1, 2, 3, 4))'),
        ('move', f'u => total(transposed(U32.to_nat({MOVES}), @P, F.zeros([1, {COPY_CHANNELS}, {COPY_SIDE}, {COPY_SIDE}])))'),
    ]
    body = 'IO.print("done")'
    for _ in range(REPEATS):
        for name, value in reversed(chains):
            for priced in (True, False):
                text = value.replace('@P', 'True{}' if priced else 'False{}').replace('@N', str(ITERATIONS if priced else 0))
                body = f'timed("{name} {int(priced)}", {text}, {body})'
    return SOURCE.replace('@BODY@', body)


def build():
    """Compile the probe; return the executable."""
    BUILD.mkdir(parents=True, exist_ok=True)
    bend = BUILD / 'planner_prices.bend'
    bend.write_text(program(), encoding='utf-8', newline='\n')
    generated = BUILD / 'planner_prices.c'
    generated.unlink(missing_ok=True)
    subprocess.run(['node', 'scripts/bend_launcher.mjs', bend.relative_to(ROOT).as_posix(), '-o',
                    generated.relative_to(ROOT).as_posix()], cwd=ROOT, check=True, capture_output=True)
    executable = BUILD / 'planner_prices'
    subprocess.run([os.environ.get('CC', 'clang'), *FLAGS, str(generated), '-lpthread', '-lm', '-o', str(executable)], check=True)
    return executable


def charged():
    """The prices of the default cost configuration, in ns."""
    text = (ROOT / 'lib/default_kernel_cost.bend').read_text(encoding='utf-8')
    return {name: float(value) for name, value in re.findall(r'^def (row|byte|step|move)_ns\(\) -> F32:\n  ([0-9.]+)', text, re.M)}


def main():
    executable = build()
    lines = subprocess.run([str(executable), '+RTS', '--threads', '1', '-RTS'], check=True, capture_output=True,
                           text=True).stdout.splitlines()
    times = {}
    for line in lines:
        words = line.split()
        if len(words) == 3 and words[0] in NAMES:
            times.setdefault((words[0], words[1]), []).append(int(words[2]))
    counts = dict(row=CALLS * SIDE * SIDE * CHANNELS // 8 * CHANNELS,
                  byte=COPIES * COPY_SIDE * COPY_SIDE * COPY_CHANNELS * 4,
                  step=ITERATIONS * STEPS_PER_ITERATION,
                  move=MOVES * COPY_SIDE * COPY_SIDE * COPY_CHANNELS // 8)
    prices = charged()
    # IO.now counts milliseconds.
    print('price  measured ns  charged ns  priced ms  unpriced ms  samples')
    for name in NAMES:
        priced = statistics.median(times[(name, '1')])
        unpriced = statistics.median(times[(name, '0')])
        measured = (priced - unpriced) * 1e6 / counts[name]
        print(f'{name:<6} {measured:>11.4f} {prices.get(name, float("nan")):>11.4f} {priced:>10} {unpriced:>12}  {len(times[(name, "1")])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
