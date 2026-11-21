"""Rule F13: find wide float vectors whose lanes clang's SLP vectorizer permuted, and the PHI groups that cause it.

Run from the repository root, with the compiler the build uses:
  CC=clang-19 python -B scripts/toolchain_rules/slp_lanes.py GENERATED.c [NAME ...]
NAME keeps only functions whose name contains it. Each irregular shuffle of a
float vector into 16 or more lanes is printed. Regular layouts are one source
vector repeated, or each run of eight lanes one source lane. Then, in the IR
built without SLP, each block holding 16 or more float or i32 PHIs of one type
whose use counts differ is printed, with the count of every PHI in block order.
"""
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from verify import BUILD, FLAGS

SHUFFLE = re.compile(r'shufflevector <(\d+) x float> %\S+, <\d+ x float> \S+, <(\d+) x i32> <([^>]*)>')
PHI = re.compile(r'^\s+(%[\w.]+) = phi (float|i32) ')


def functions(text, names):
    for match in re.finditer(r'\ndefine [^@]*@([\w.$]+)\(.*?\n}\n', text, re.S):
        if not names or any(name in match[1] for name in names):
            yield match[1], match[0]


def regular(width, lanes, mask):
    # A poison lane (None) is free to take any source lane.
    repeated = all(lane is None or lane == index % width for index, lane in enumerate(mask))
    runs = [mask[start:start + 8] for start in range(0, lanes, 8)]
    return repeated or all(len({lane for lane in run if lane is not None}) <= 1 for run in runs)


def irregular_shuffles(body):
    for match in SHUFFLE.finditer(body):
        width, lanes = int(match[1]), int(match[2])
        entries = [entry.split()[-1] for entry in match[3].split(',')]
        mask = [None if entry in ('poison', 'undef') else int(entry) for entry in entries]
        if lanes >= 16 and not regular(width, lanes, mask):
            yield f'<{width} x float> to {lanes} lanes: ' + ' '.join('_' if lane is None else str(lane) for lane in mask)


def uneven_phis(body):
    uses = Counter(re.findall(r'%[\w.]+', body))
    block, group = 'entry', []
    for line in body.splitlines() + ['end:']:
        label = re.match(r'^([\w.]+):', line)
        found = PHI.match(line)
        if found:
            group.append((found[1], found[2]))
            continue
        if label or line.strip():
            for kind in {kind for _, kind in group}:
                counts = [uses[name] - 1 for name, other in group if other == kind]
                if len(counts) >= 16 and len(set(counts)) > 1:
                    yield f'block {block}: {len(counts)} {kind} PHIs, uses ' + ' '.join(map(str, counts))
            group = []
        if label:
            block = label[1]


def emit(source, extra, suffix):
    target = BUILD / (source.stem + suffix)
    compiler = os.environ.get('CC', 'clang-19')
    subprocess.run([compiler, *FLAGS, *extra, '-S', '-emit-llvm', str(source), '-o', str(target)], check=True)
    return target.read_text(errors='replace')


def main():
    source, names = Path(sys.argv[1]), sys.argv[2:]
    BUILD.mkdir(parents=True, exist_ok=True)
    for name, body in functions(emit(source, [], '.ll'), names):
        for line in irregular_shuffles(body):
            print(f'{name}: {line}')
    for name, body in functions(emit(source, ['-fno-slp-vectorize'], '.no_slp.ll'), names):
        for line in uneven_phis(body):
            print(f'{name}: {line}')


if __name__ == '__main__':
    main()
