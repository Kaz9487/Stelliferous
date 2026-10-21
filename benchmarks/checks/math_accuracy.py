"""Measure the elementwise mathematical functions against MPFR on every FP32 input.

The public functions run in a compiled Bend program over blocks of consecutive
32-bit words; math_accuracy.c compares each result with MPFR. Linux, with
libmpfr-dev installed. Blocks run side by side, one per job; a finished block
is kept, so an interrupted run resumes.
"""

import argparse
import bisect
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import random
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'out/results/math_accuracy'
BUILD = ROOT / 'out/build/math_accuracy'
SOURCE = ROOT / 'out/checks/math_accuracy.bend'
ACTIVATIONS = ['identity', 'relu', 'relu6', 'leaky_relu', 'hard_sigmoid', 'hard_swish', 'sigmoid', 'tanh', 'silu', 'gelu_tanh']
FUNCTIONS = ['sigmoid', 'silu', 'tanh', 'gelu_tanh', 'exp', 'log'] + [name + '_derivative' for name in ACTIVATIONS]


def call(name, tensor):
    if name.endswith('_derivative'):
        activation = name.removesuffix('_derivative')
        argument = 'Act.leaky_relu(0.125)' if activation == 'leaky_relu' else f'Act.{activation}()'
        return f'F.activate_derivative({tensor},{argument})'
    return f'F.{name}({tensor})'


def program(count, names):
    steps = ''.join(
        f'    {name}_input : F.Tensor() <- F.load_raw("input.bin",[{count}])\n'
        f'    {name}_saved : F.Tensor() <- F.save_raw("{name}.bin",{call(name, name + "_input")})\n'
        f'    {name}_checked : F.Tensor() <- F.expect({name}_saved)\n'
        for name in names
    )
    return (
        'import Base\nimport ../../lib/tensor_f32.bend as F\nimport ../../lib/math_activation.bend as Act\n\n'
        'def main() -> IO(Unit):\n  do IO<Unit>:\n' + steps + '    IO.print("done")\n'
    )


def build(cc, count, names):
    for folder in (OUT / 'blocks', BUILD, SOURCE.parent):
        folder.mkdir(parents=True, exist_ok=True)
    SOURCE.write_text(program(count, names))
    subprocess.run([sys.executable, '-B', 'scripts/toolchain_rules/lint.py', str(SOURCE)], cwd=ROOT, check=True)
    generated = BUILD / 'math_accuracy.c'
    subprocess.run(['node', 'scripts/bend_launcher.mjs', str(SOURCE), '-o', str(generated)], cwd=ROOT, check=True)
    subprocess.run(
        [cc, '-O3', '-march=native', '-ffp-contract=off', '-std=c11', str(generated), '-lpthread', '-lm',
         '-o', str(BUILD / 'functions')],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        [cc, '-O2', '-std=c11', 'benchmarks/checks/math_accuracy.c', '-lmpfr', '-lgmp', '-lm',
         '-o', str(BUILD / 'compare')],
        cwd=ROOT, check=True,
    )


def measure(block, bits, names):
    base, count = block << bits, 1 << bits
    compare = str(BUILD / 'compare')
    work = OUT / 'work' / str(block)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    subprocess.run([compare, 'input', str(base), str(count), 'input.bin'], cwd=work, check=True)
    subprocess.run([str(BUILD / 'functions'), '--threads', '1', '--gpu', 'off'], cwd=work, check=True,
                   capture_output=True)
    path = OUT / 'blocks' / f'{bits}_{block:05d}.json'
    kept = json.loads(path.read_text()) if path.is_file() else []
    rows = [row for row in kept if row['function'] not in names]
    for name in names:
        result = subprocess.run(
            [compare, 'check', name, str(base), str(count), 'input.bin', f'{name}.bin'],
            cwd=work, check=True, capture_output=True, text=True,
        )
        rows.append(json.loads(result.stdout))
    shutil.rmtree(work)
    rows.sort(key=lambda row: FUNCTIONS.index(row['function']))
    path.write_text(json.dumps(rows) + '\n')
    print(f'block {block}: ' + ', '.join(f'{row["function"]} {row["largest_ulp"]:.3f}' for row in rows), flush=True)


def summarize(bits, first=0, last=None):
    blocks = [path for path in sorted((OUT / 'blocks').glob(f'{bits}_*.json'))
              if first <= int(path.stem.split('_')[1]) < (last if last is not None else 1 << (32 - bits))]
    totals = {}
    for path in blocks:
        for row in json.loads(path.read_text()):
            total = totals.setdefault(row['function'], dict(
                finite=0, largest_ulp=0.0, largest_word=0, steps=[0] * 5, undecided=0, nan_wrong=0,
                infinity_wrong=0, zero_sign_wrong=0, plus_infinity=None, minus_infinity=None))
            for key in ('finite', 'undecided', 'nan_wrong', 'infinity_wrong', 'zero_sign_wrong'):
                total[key] += row[key]
            total['steps'] = [a + b for a, b in zip(total['steps'], row['steps'])]
            if row['largest_ulp'] > total['largest_ulp']:
                total['largest_ulp'], total['largest_word'] = row['largest_ulp'], row['largest_word']
            first, last = row['base'], row['base'] + row['count'] - 1
            if first <= 0x7f800000 <= last:
                total['plus_infinity'] = row['plus_infinity']
            if first <= 0xff800000 <= last:
                total['minus_infinity'] = row['minus_infinity']
    inputs = len(blocks) << bits
    summary = dict(inputs=inputs, complete=inputs == 1 << 32, functions=totals)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = [
        f'{inputs} of {1 << 32} input words.', '',
        '| Function | Largest error (ULP) | At input | Correctly rounded | 1 step | 2 steps | 3 steps | 4 or more |',
        '|---|---:|---|---:|---:|---:|---:|---:|',
    ]
    for name, total in totals.items():
        value = struct.unpack('<f', struct.pack('<I', total['largest_word']))[0]
        share = total['steps'][0] / max(1, total['finite'])
        lines.append(
            f'| `{name}` | {total["largest_ulp"]:.3f} | `{value!r}` | {share:.4%} | '
            + ' | '.join(str(step) for step in total['steps'][1:]) + ' |')
    problems = {
        name: {key: total[key] for key in ('undecided', 'nan_wrong', 'infinity_wrong', 'zero_sign_wrong') if total[key]}
        for name, total in totals.items()
    }
    lines += ['', 'Other findings: ' + json.dumps({name: found for name, found in problems.items() if found})]
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def sample(count, seed, names, region, magnitude_bounds=(), sample_blocks=(), block_bits=24):
    # Uniform words, plus adjacent words at zero, the clamps and infinities.
    rng = random.Random(seed)
    anchors = [0, 0x00800000, 0x3f000000, 0x40000000, 0x40c00000, 0x41800000, 0x41900000,
               0x7f7fffff, 0x7f800000, 0x7fc00000]
    edges = [(sign | ((word + offset) & 0x7fffffff))
             for sign in (0, 0x80000000) for word in anchors for offset in (-1, 0, 1)]
    words = edges + [(rng.randrange(0xbf000000, 0xc1800000) if region == 'negative'
                      else rng.getrandbits(32)) for _ in range(count - len(edges))]
    if sample_blocks:
        rng = random.Random(seed)
        words = edges + [(sample_blocks[i % len(sample_blocks)] << block_bits) + rng.getrandbits(block_bits)
                         for i in range(count - len(edges))]
    work = OUT / 'sample_work'
    work.mkdir(parents=True, exist_ok=True)
    (work / 'input.bin').write_bytes(struct.pack(f'<{count}I', *words))
    subprocess.run([str(BUILD / 'functions'), '--threads', '1', '--gpu', 'off'],
                   cwd=work, check=True, capture_output=True)
    rows = []
    for name in names:
        result = subprocess.run([str(BUILD / 'compare'), 'check', name, 'sample', str(count),
                                 'input.bin', f'{name}.bin'], cwd=work, check=True,
                                capture_output=True, text=True)
        row = json.loads(result.stdout)
        row['largest_input'] = struct.unpack('<f', struct.pack('<I', row['largest_word']))[0]
        rows.append(row)
    report = dict(inputs=count, complete=False, seed=seed, region=region, functions=rows)
    if magnitude_bounds or sample_blocks:
        inputs = (work / 'input.bin').read_bytes()
        outputs = {name: (work / f'{name}.bin').read_bytes() for name in names}
        groups = []
        if magnitude_bounds:
            bins = [[] for _ in range(len(magnitude_bounds) + 1)]
            for index, (value,) in enumerate(struct.iter_unpack('<f', inputs)):
                if not math.isnan(value):
                    bins[bisect.bisect_right(magnitude_bounds, abs(value))].append(index)
            report['magnitude_regions'] = []
            groups.extend(('magnitude_regions', dict(
                lower=magnitude_bounds[group - 1] if group else 0,
                upper=magnitude_bounds[group] if group < len(magnitude_bounds) else None), indices)
                for group, indices in enumerate(bins))
        if sample_blocks:
            bins = {block: [] for block in sample_blocks}
            for index, (word,) in enumerate(struct.iter_unpack('<I', inputs)):
                block = word >> block_bits
                if block in bins:
                    bins[block].append(index)
            report.update(sample_blocks=list(sample_blocks), block_bits=block_bits, block_regions=[])
            groups.extend(('block_regions', dict(block=block), indices) for block, indices in bins.items())
        for destination, metadata, indices in groups:
            if not indices:
                continue
            (work / 'region_input.bin').write_bytes(b''.join(inputs[i * 4:i * 4 + 4] for i in indices))
            measured = []
            for name in names:
                output = outputs[name]
                (work / 'region_output.bin').write_bytes(b''.join(output[i * 4:i * 4 + 4] for i in indices))
                result = subprocess.run(
                    [str(BUILD / 'compare'), 'check', name, 'sample', str(len(indices)),
                     'region_input.bin', 'region_output.bin'],
                    cwd=work, check=True, capture_output=True, text=True,
                )
                measured.append(json.loads(result.stdout))
            report[destination].append(dict(metadata, functions=measured))
    (OUT / 'sample.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    shutil.rmtree(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-bits', type=int, default=24, help='log2 of the inputs in one block')
    parser.add_argument('--first', type=int, default=0, help='first block')
    parser.add_argument('--last', type=int, help='one past the last block; every block by default')
    parser.add_argument('--jobs', type=int, default=os.cpu_count(), help='blocks measured at the same time')
    parser.add_argument('--functions', nargs='+', choices=FUNCTIONS,
                        help='measure these again in every block and keep the other results')
    parser.add_argument('--sample-region', choices=['words', 'negative'], default='words')
    parser.add_argument('--samples', type=int, help='sample this many words instead of consecutive blocks')
    parser.add_argument('--seed', type=int, default=20261011, help='seed for word sampling')
    parser.add_argument('--sample-blocks', type=int, nargs='+', default=[],
                        help='sample equally from these block IDs and report each block separately')
    parser.add_argument('--report-magnitude-bounds', type=float, nargs='+', default=[],
                        help='also report samples by absolute input, in half-open intervals at these bounds')
    args = parser.parse_args()
    if args.samples is not None and args.samples < 60:
        parser.error('--samples must be at least 60')
    if not 1 <= args.block_bits <= 32:
        parser.error('--block-bits must be between 1 and 32')
    if args.sample_blocks and (args.samples is None or args.sample_region != 'words'
                              or len(set(args.sample_blocks)) != len(args.sample_blocks)
                              or any(b < 0 or b >= 1 << (32 - args.block_bits) for b in args.sample_blocks)):
        parser.error('--sample-blocks requires samples, the words region and unique valid block IDs')
    bounds = args.report_magnitude_bounds
    if bounds and (args.samples is None or any(not math.isfinite(x) or x <= 0 for x in bounds)
                   or any(a >= b for a, b in zip(bounds, bounds[1:]))):
        parser.error('--report-magnitude-bounds requires samples and strictly increasing positive finite bounds')
    cc = os.environ.get('CC') or shutil.which('clang-19') or shutil.which('clang')
    if not cc:
        raise SystemExit('The accuracy measurement needs clang on PATH or CC.')
    last = args.last if args.last is not None else 1 << (32 - args.block_bits)
    names = args.functions or FUNCTIONS
    build(cc, args.samples or (1 << args.block_bits), names)
    if args.samples is not None:
        sample(args.samples, args.seed, names, args.sample_region, bounds, args.sample_blocks, args.block_bits)
        return
    blocks = [
        block for block in range(args.first, last)
        if args.functions or not (OUT / 'blocks' / f'{args.block_bits}_{block:05d}.json').is_file()
    ]
    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(lambda block: measure(block, args.block_bits, names), blocks))
    summarize(args.block_bits, args.first, last)


if __name__ == '__main__':
    main()
