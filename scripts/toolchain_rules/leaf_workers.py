"""Run the two leaf probes of rule H4 and count where their leaves run.

Run on Linux from the repository root: python3 -B scripts/toolchain_rules/leaf_workers.py
It compiles `leaf_task` and `leaf_inline` from probes.py with 4, 8 and 16
leaves, records every task the runtime runs with its thread, its turn and its
time, and runs each program at 2 and 4 threads. A leaf is a task that lasts at
least a third of the longest one. The rule holds when the fork-free form runs
no leaf in a grow turn and the inlined form runs two. Wall time and the
largest number of leaves that run at once are printed for comparison.
"""
import os
import statistics
import subprocess
import sys
from pathlib import Path

from probes import CODEGEN

ROOT = Path(__file__).resolve().parent.parent.parent
BUILD = ROOT / 'out/toolchain_rules'
LEAVES = (4, 8, 16)
THREADS = (2, 4)
REPEATS = 5
GROW_LEAVES = {'leaf_task': 0, 'leaf_inline': 2}
FLAGS = ['-O3', '-march=native', '-ffp-contract=off', '-std=c11']

RECORDER = r'''
typedef struct { uint64_t t0, t1; uint32_t grow; uint64_t thread; } LeafSpan;
static LeafSpan leaf_spans[65536];
static uint32_t leaf_count;
static uint64_t leaf_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void leaf_add(uint64_t t0, uint32_t grow) {
  uint32_t i = __atomic_fetch_add(&leaf_count, 1, __ATOMIC_RELAXED);
  if (i < 65536) leaf_spans[i] = (LeafSpan){ t0, leaf_now(), grow, (uint64_t)pthread_self() };
}
static void leaf_dump(void) {
  FILE* f = fopen(getenv("LEAF_SPANS"), "w");
  for (uint32_t i = 0; i < leaf_count && i < 65536; i += 1)
    fprintf(f, "%llu %llu %u %llu\n", (unsigned long long)leaf_spans[i].t0, (unsigned long long)leaf_spans[i].t1,
            leaf_spans[i].grow, (unsigned long long)leaf_spans[i].thread);
  fclose(f);
}
'''
EDITS = (
    ('// Monk\n// ====', RECORDER + '\n// Monk\n// ===='),
    ('  bool     seq = stride == 0;', '  bool     seq = stride == 0;\n  uint32_t leaf_grow = stride != 0;'),
    ('    Term r = work_loop(e, stk, t, seq);',
     '    uint64_t leaf_t0 = leaf_now();\n    Term r = work_loop(e, stk, t, seq);\n    leaf_add(leaf_t0, leaf_grow);'),
    ('int main(int argc, char** argv) {', 'int main(int argc, char** argv) {\n  atexit(leaf_dump);'),
)


def build(name, leaves):
    """Compile one probe with the given number of leaves and the recorder; return the executable."""
    source = CODEGEN[name][3]
    fixed = 'Nat.mod(now, 1n)'
    assert source.count(fixed) == 1
    stem = f'{name}_{leaves}'
    bend = BUILD / f'{stem}.bend'
    bend.write_text('import Base\n' + source.replace(fixed, f'Nat.add(Nat.mod(now, 1n), {leaves}n)'),
                    encoding='utf-8', newline='\n')
    plain = BUILD / f'{stem}.c'
    plain.unlink(missing_ok=True)
    subprocess.run(['node', 'scripts/bend_launcher.mjs', bend.relative_to(ROOT).as_posix(), '-o',
                    plain.relative_to(ROOT).as_posix()], cwd=ROOT, check=True, capture_output=True)
    text = plain.read_text(encoding='utf-8')
    for needle, replacement in EDITS:
        assert text.count(needle) == 1, f'the runtime no longer contains {needle!r}'
        text = text.replace(needle, replacement)
    recorded = BUILD / f'{stem}_recorded.c'
    recorded.write_text(text, encoding='utf-8', newline='\n')
    executable = BUILD / f'{stem}_recorded'
    subprocess.run([os.environ.get('CC', 'clang'), *FLAGS, str(recorded), '-lpthread', '-lm', '-o', str(executable)],
                   check=True)
    return executable


def observe(executable, threads):
    """Run once; return wall milliseconds, leaves in all, leaves in a grow turn, the largest overlap and the threads used."""
    spans = BUILD / 'leaf_spans.txt'
    subprocess.run([str(executable), '--threads', str(threads)], check=True, capture_output=True,
                   env=dict(os.environ, LEAF_SPANS=str(spans)))
    rows = [tuple(int(word) for word in line.split()) for line in spans.read_text().splitlines()]
    longest = max(stop - start for start, stop, _, _ in rows)
    leaves = [row for row in rows if (row[1] - row[0]) * 3 >= longest]
    edges = sorted([(start, 1) for start, _, _, _ in leaves] + [(stop, -1) for _, stop, _, _ in leaves])
    running = most = 0
    for _, change in edges:
        running += change
        most = max(most, running)
    wall = (max(row[1] for row in leaves) - min(row[0] for row in leaves)) / 1e6
    return wall, len(leaves), sum(1 for row in leaves if row[2]), most, len({row[3] for row in leaves})


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    failures = []
    print('form         leaves threads  wall ms  leaves run  in grow turn  at once  threads used')
    for leaves in LEAVES:
        for name in GROW_LEAVES:
            executable = build(name, leaves)
            for threads in THREADS:
                runs = [observe(executable, threads) for _ in range(REPEATS)]
                wall = statistics.median(run[0] for run in runs)
                counts = sorted({run[1:] for run in runs})
                for ran, grow, most, used in counts:
                    print(f'{name:<12} {leaves:>6} {threads:>7} {wall:>8.1f} {ran:>11} {grow:>13} {most:>8} {used:>13}')
                    if ran != leaves:
                        failures.append(f'{name}, {leaves} leaves, {threads} threads: {ran} leaves were recorded')
                    if grow != GROW_LEAVES[name]:
                        failures.append(f'{name}, {leaves} leaves, {threads} threads: {grow} leaves ran in a grow turn, '
                                        f'expected {GROW_LEAVES[name]}')
    for failure in failures:
        print('DIFFERENT:', failure)
    print('RULE H4 HOLDS' if not failures else f'{len(failures)} differences: reread the sources and update rule H4')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
