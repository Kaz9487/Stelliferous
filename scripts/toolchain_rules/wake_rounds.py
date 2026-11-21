"""Measure what a parallel region costs the runtime (rule H11) and compare it with
the wake round the planner charges.

Run on Linux from the repository root: python3 -B scripts/toolchain_rules/wake_rounds.py
It compiles one probe that runs rounds of serial work followed by a region of
four fork-free leaves, or the same work without a fork, and prints the time
each config took. The cost of a region is the difference between the forked
and the serial runs at four threads, divided by the rounds. The planner
charges two wake rounds a region (lib/default_kernel_cost.bend), so a round
must cover half of the largest cost measured.
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
THREADS = 4
REPEATS = 3
# Rounds, serial work before each region (generator steps), leaf work, label.
GAPS = ((4000, 0, 'back to back'), (500, 2000000, 'short gap'), (200, 20000000, 'medium gap'), (60, 80000000, 'long gap'))
LEAF = 50000

SOURCE = '''import Base

def heavy(n: Nat, acc: U32) -> U32:
  match n:
    case 0n: acc
    case 1n+p: heavy(p, (acc * 1664525 + 1013904223 : U32))

def leaf(n: Nat, seed: U32) -> U32:
  heavy(n, seed)

def tree(depth: Nat, +n: Nat, +seed: U32) -> U32:
  match depth:
    case 0n: heavy(n, seed)
    case 1n:
      left right = leaf(n, (seed * 2 : U32)) leaf(n, (seed * 2 + 1 : U32))
      (left + right : U32)
    case 2n+ +rest:
      left right = tree(1n+rest, n, (seed * 2 : U32)) tree(1n+rest, n, (seed * 2 + 1 : U32))
      (left + right : U32)

def regions(count: Nat, +depth: Nat, +n: Nat, +g: Nat, acc: U32) -> U32:
  match count:
    case 0n: acc
    case 1n+rest: regions(rest, depth, n, g, tree(depth, n, heavy(g, acc)))

def timed(count: Nat, depth: Nat, n: Nat, g: Nat, next: IO(Unit)) -> IO(Unit):
  IO.bind(Nat, Unit, IO.now(), t0 =>
    IO.bind(Unit, Unit, IO.print(U32.show(regions(count, depth, n, g, 1))), u =>
      IO.bind(Nat, Unit, IO.now(), t1 =>
        IO.bind(Unit, Unit, IO.print(Nat.show(Nat.sub(t1, t0))), v => next))))

def main() -> IO(Unit):
'''


def program():
    """The probe source: each gap without a fork, then with one."""
    body = 'IO.print("done")'
    for count, gap, _ in reversed(GAPS):
        for depth in ('2n', '0n'):
            body = f'timed(U32.to_nat({count}), {depth}, U32.to_nat({LEAF}), U32.to_nat({gap}), {body})'
    return SOURCE + '  ' + body + '\n'


def build():
    """Compile the probe; return the executable."""
    BUILD.mkdir(parents=True, exist_ok=True)
    bend = BUILD / 'wake_rounds.bend'
    bend.write_text(program(), encoding='utf-8', newline='\n')
    generated = BUILD / 'wake_rounds.c'
    generated.unlink(missing_ok=True)
    subprocess.run(['node', 'scripts/bend_launcher.mjs', bend.relative_to(ROOT).as_posix(), '-o',
                    generated.relative_to(ROOT).as_posix()], cwd=ROOT, check=True, capture_output=True)
    executable = BUILD / 'wake_rounds'
    subprocess.run([os.environ.get('CC', 'clang'), *FLAGS, str(generated), '-lpthread', '-lm', '-o', str(executable)], check=True)
    return executable


def observe(executable, threads):
    """Run once; return the milliseconds of each config, serial then forked for each gap."""
    lines = subprocess.run([str(executable), '+RTS', '--threads', str(threads), '-RTS'], check=True, capture_output=True,
                           text=True).stdout.split()
    return [int(word) for word in lines[1:-1:2]]


def charged_round_us():
    """The wake round of the default cost configuration, in microseconds."""
    text = (ROOT / 'lib/default_kernel_cost.bend').read_text(encoding='utf-8')
    return float(re.search(r'^def wake_round_ns\(\) -> F32:\n  ([0-9.]+)', text, re.M).group(1)) / 1000


def main():
    executable = build()
    runs = [observe(executable, THREADS) for _ in range(REPEATS)]
    times = [statistics.median(run[i] for run in runs) for i in range(len(runs[0]))]
    print(f'gap            rounds  serial ms  forked ms  region us at {THREADS} threads')
    largest = 0.0
    for index, (count, _, label) in enumerate(GAPS):
        serial, forked = times[2 * index], times[2 * index + 1]
        region = (forked - serial) * 1000 / count
        largest = max(largest, region)
        print(f'{label:<14} {count:>6} {serial:>10} {forked:>10} {region:>10.1f}')
    charged = charged_round_us()
    print(f'two rounds charged a region: {2 * charged:.1f} us, largest region measured: {largest:.1f} us')
    if largest > 2 * charged:
        print('DIFFERENT: a region costs more than two wake rounds; raise the wake round in lib/default_kernel_cost.bend')
        return 1
    print('RULE H11 HOLDS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
