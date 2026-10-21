"""Re-verify docs/toolchain_rules.md against the pinned official Bend.

Run from the repository root: python -B scripts/toolchain_rules/verify.py
It checks that the source passages the rules cite still exist, that the
snapshot of checker diagnostic fragments is unchanged, and that every probe in
probes.py still behaves as recorded. After reviewing a new Bend version and
updating the rules, run it with --snapshot to record that version.
"""
import argparse
from collections import Counter
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from fork_flags import task_flags
from lint import lint
from probes import CHECKER, CODEGEN, TASKS, LINT, EXECUTION, SEGMENTS, CONSTRUCTORS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCES = ROOT / '.tools/bend/bend2'
BUILD = ROOT / 'out/toolchain_rules'
CALL = re.compile(r'(?:parse_fail|Err)\(')
TEXT = re.compile(r'"((?:[^"\\]|\\.){6,})"')
FLAGS = ['-O3', '-march=native', '-ffp-contract=off', '-std=c11']


def assembly(target):
    """Keep assembly and optimization remarks, with static counts per function.

    Counts include all emitted instructions, not executions or elapsed cost.
    C identifiers and assembly locations let callers inspect the hot region.
    """
    compiler = os.environ.get('CC', 'clang-19')
    asm = target.with_suffix('.s')
    command = [compiler, *FLAGS, '-S', '-Rpass=loop-vectorize',
               '-Rpass-missed=loop-vectorize', '-Rpass=slp-vectorizer',
               '-Rpass-missed=inline', str(target), '-o', str(asm)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    target.with_suffix('.remarks.txt').write_text(result.stderr, encoding='utf-8')
    result.check_returncode()
    functions = {}
    current = None
    for number, line in enumerate(asm.read_text().splitlines(), 1):
        label = re.match(r'^([A-Za-z_][\w.$]*):', line)
        if label:
            current = label[1]
            functions[current] = dict(line=number, mnemonics=Counter(), calls=[])
        instruction = re.match(r'^\s+([a-z][a-z0-9.]*)\s*(.*?)\s*(?:#.*)?$', line)
        if current and instruction:
            op, operands = instruction.groups()
            functions[current]['mnemonics'][op] += 1
            if op.startswith('call') or op in ('bl', 'blr'):
                functions[current]['calls'].append(dict(line=number, target=operands))
    for value in functions.values():
        value['instructions'] = sum(value['mnemonics'].values())
    text = target.read_text()
    report = dict(compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                  flags=FLAGS, machine=os.uname().machine,
                  static_instructions=sum(v['instructions'] for v in functions.values()),
                  functions=functions, tasks=task_flags(text),
                  c_marker_note='Lexical occurrences including helper definitions; not dynamic calls.',
                  c_markers={key: len(re.findall(r'\b' + key + r'\s*\(', text))
                             for key in ('ctr_take', 'ctr_new', 'blk_copy', 'term_clo', 'task_node')})
    path = target.with_suffix('.assembly.json')
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'{target.name}: {report["static_instructions"]} static instructions, {len(functions)} functions')
    return report


def diagnostics():
    """Literal fragments on Err/parse_fail lines; not all toolchain diagnostics."""
    found = set()
    for line in (SOURCES / 'bend.ts').read_text(encoding='utf-8').splitlines():
        if CALL.search(line):
            found.update(TEXT.findall(line))
    return sorted(found)


def launch(name, source, *arguments):
    path = BUILD / f'{name}.bend'
    if isinstance(source, Path):
        path = source.resolve()
    elif isinstance(source, dict):
        path = BUILD / name / 'main.bend'
        for relative, content in source['files'].items():
            fixture = path.parent / relative
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_text(content, encoding='utf-8', newline='\n')
        source = source['main']
    if not isinstance(source, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('import Base\n' + source, encoding='utf-8', newline='\n')
    result = subprocess.run(['node', 'scripts/bend_launcher.mjs', path.relative_to(ROOT).as_posix(), *arguments],
                            cwd=ROOT, capture_output=True, text=True, timeout=300)
    return result


def check_probe(name, expected, source):
    result = launch(name, source, '--check-only')
    output = result.stdout + result.stderr
    errors = []
    if name in LINT:
        findings = {rule for _, rule, _ in lint(BUILD / f'{name}.bend')}
        if findings != LINT[name]:
            errors.append(f'lint {name}: expected {sorted(LINT[name])}, got {sorted(findings)}')
    passed = result.returncode == 0 and 'ALL PROOFS CHECK' in output
    if (passed and expected == 'pass') or (result.returncode != 0 and expected != 'pass' and expected in output):
        return errors
    first = next((line for line in output.splitlines() if line.strip()), 'no output')
    return errors + [f'probe {name}: expected {expected!r}, got {first!r}']


def check_codegen(name, present, absent, shared, source):
    """Compile one probe to C and compare the markers the rules rely on."""
    target = BUILD / f'{name}.c'
    target.unlink(missing_ok=True)
    result = launch(name, source, '-o', target.relative_to(ROOT).as_posix())
    output = result.stdout + result.stderr
    if result.returncode != 0 or not target.exists():
        return [f'codegen {name}: no C was produced ({output.strip()[:160]})']
    text = target.read_text(encoding='utf-8', errors='replace')
    found = [f'codegen {name}: expected {needle!r} in the C' for needle in present if needle not in text]
    found += [f'codegen {name}: did not expect {needle!r} in the C' for needle in absent if needle in text]
    table = re.search(r'CID_T\[\]\[2\] = \{(.*)\};', text)
    flags = [int(flag) for flag in re.findall(r'\{\s*\d+\s*,\s*(\d+)\s*\}', table.group(1))] if table else []
    actual = 'none' if sum(flags) == 0 else 'all' if sum(flags) == len(flags) else 'some'
    if actual != shared:
        found.append(f'codegen {name}: expected {shared} constructors shared, got {sum(flags)} of {len(flags)}')
    ids = {key: int(value) for key, value in re.findall(r'^#define (CID_\w+) (\d+)$', text, re.M)}
    for key, expected in CONSTRUCTORS.get(name, {}).items():
        index = ids.get(key, len(flags))
        actual_flag = flags[index] if index < len(flags) else None
        if actual_flag != expected:
            found.append(f'codegen {name}: {key} expected heat {expected}, got {actual_flag}')
    for key, (need, reject) in SEGMENTS.get(name, {}).items():
        pattern = (r'^(?:INLINE|DFAR) Term ' + re.escape(key) + r'\([^\n]*\) \{(.*?)^}'
                   if key.startswith('spin_') else
                   r'WL_CASE\(' + re.escape(key) + r'\)\s*\{(.*?)\n\s*\}\}')
        match = re.search(pattern, text, re.S | re.M)
        if match is None:
            found.append(f'codegen {name}: missing segment {key}')
            continue
        body = match[1]
        found += [f'codegen {name}/{key}: missing {s!r}' for s in need if s not in body]
        found += [f'codegen {name}/{key}: unexpected {s!r}' for s in reject if s in body]
    tasks = task_flags(text)
    found += [f'codegen {name}: expected task {task} to be {flag}, got {tasks.get(task)}'
              for task, flag in TASKS.get(name, {}).items() if tasks.get(task) != flag]
    return found


def execution(name, source, modes, exact=False):
    """Exercise the evaluator, generated JS and native C as separate routes."""
    failures = []
    for mode, (success, expected) in modes.items():
        if mode in ('check', 'eval'):
            result = launch(name, source, *(['--check-only'] if mode == 'check' else []))
        elif mode.startswith('emit-'):
            target = BUILD / f'{name}.{mode[5:]}'
            target.unlink(missing_ok=True)
            result = launch(name, source, '-o', target.relative_to(ROOT).as_posix())
        else:
            target = BUILD / f'{name}.{mode}'
            target.unlink(missing_ok=True)
            result = launch(name, source, '-o', target.relative_to(ROOT).as_posix())
            if result.returncode != 0 or not target.exists():
                failures.append(f'{name}/{mode}: generation failed: {result.stderr[:200]}')
                continue
            command = ['node', str(target)]
            if mode == 'c':
                binary = target.with_suffix('.exe')
                build = subprocess.run([os.environ.get('CC', 'clang-19'), *FLAGS,
                    str(target), '-lpthread', '-lm', '-o', str(binary)],
                    capture_output=True, text=True, timeout=300)
                if build.returncode != 0:
                    failures.append(f'{name}/c: build failed: {build.stderr[:200]}')
                    continue
                command = [str(binary), '-t', '1']
            result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                    text=True, timeout=60)
        output = result.stdout + result.stderr
        matches = output.strip() == expected if exact or success else expected in output
        if (result.returncode == 0) != success or not matches:
            failures.append(f'{name}/{mode}: exit {result.returncode}, expected '
                            f'{success}, {expected!r}; got {output[:200]!r}')
        print(f'{name}/{mode}: exit {result.returncode}: {output.strip()[:120]}')
    return failures


def official(paths):
    """Run selected official tests with their imports and compare their #| output.

    Failing and main-less tests run in the plain CLI; tests with a main also run as JS and C.
    """
    failures = []
    for path in paths:
        source = path.read_text(encoding='utf-8')
        expected = '\n'.join(line[2:] for line in source.splitlines() if line.startswith('#|')).strip()
        if not expected:
            failures.append(f'official {path}: no #| expectation')
            continue
        failed = expected.endswith('\nexit 1')
        if failed:
            expected = expected.removesuffix('\nexit 1')
        modes = {'eval': (not failed, expected)}
        if not failed and re.search(r'^(def|law) main(?:\(|:)', source, re.M):
            modes.update({mode: (True, expected) for mode in ('js', 'c')})
        name = 'official_' + path.parent.name + '_' + path.stem
        failures += execution(name, path, modes, exact=True)
    for failure in failures:
        print('DIFFERENT:', failure)
    print(f'{len(paths)} official files: ' + ('PASS' if not failures else f'{len(failures)} differences'))
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--probe', action='append', help='run only this named probe, repeatable')
    parser.add_argument('--assembly', action='store_true', help='emit assembly and static counts for codegen probes')
    parser.add_argument('--runtime', action='store_true', help='also compare evaluator, generated JS and native C probes')
    parser.add_argument('--official', type=Path, nargs='+', help='run selected official #| tests (eval, JS and C)')
    parser.add_argument('--inspect', type=Path, nargs='+', help='emit assembly and counts for existing generated C')
    options = parser.parse_args()
    if options.official:
        BUILD.mkdir(parents=True, exist_ok=True)
        return official(options.official)
    if options.inspect:
        for target in options.inspect:
            assembly(target)
        return 0
    snapshot = options.snapshot
    available = CHECKER.keys() | CODEGEN.keys() | (EXECUTION.keys() if options.runtime else set())
    selected = set(options.probe or available)
    unknown = selected - available
    if unknown:
        parser.error('unknown probes: ' + ', '.join(sorted(unknown)))
    BUILD.mkdir(parents=True, exist_ok=True)
    failures = []
    pin = json.loads((ROOT / 'scripts/toolchain.json').read_text(encoding='utf-8'))
    current = dict(tag=pin['tag'], commit=pin['commit'])
    reviewed_path = HERE / 'reviewed.json'
    reviewed = json.loads(reviewed_path.read_text(encoding='utf-8'))
    if reviewed != current and not snapshot:
        failures.append(f"the rules were reviewed for {reviewed['tag']}; the pin is now {current['tag']}")
    anchors = json.loads((HERE / 'anchors.json').read_text(encoding='utf-8'))
    for name, entries in anchors.items():
        text = (SOURCES / name).read_text(encoding='utf-8')
        failures += [f'anchor missing in {name}: {rule}' for rule, needle in entries.items() if needle not in text]
    messages = diagnostics()
    listing = HERE / 'diagnostics.txt'
    if not snapshot:
        before = listing.read_text(encoding='utf-8').splitlines()
        failures += [f'new checker message: {line}' for line in sorted(set(messages) - set(before))]
        failures += [f'removed checker message: {line}' for line in sorted(set(before) - set(messages))]
    for name, (expected, source) in CHECKER.items():
        if name in selected:
            failures += check_probe(name, expected, source)
    for name, (present, absent, shared, source) in CODEGEN.items():
        if name in selected:
            errors = check_codegen(name, present, absent, shared, source)
            failures += errors
            if options.assembly and not errors:
                assembly(BUILD / f'{name}.c')
    for name, (source, modes) in EXECUTION.items():
        if options.runtime and name in selected:
            failures += execution(name, source, modes)
    if snapshot:
        reviewed_path.write_text(json.dumps(current, indent=2) + '\n', encoding='utf-8', newline='\n')
        listing.write_text('\n'.join(messages) + '\n', encoding='utf-8', newline='\n')
        print(f"snapshot written for {current['tag']}: {len(messages)} checker messages")
    count = sum(len(entries) for entries in anchors.values())
    print(f"{current['tag']}: {count} anchors, {len(messages)} checker messages, "
          f'{len(selected & CHECKER.keys())} checker probes, {len(selected & CODEGEN.keys())} codegen probes, '
          f'{len(selected & EXECUTION.keys()) if options.runtime else 0} execution probes')
    for failure in failures:
        print('DIFFERENT:', failure)
    print('RULES HOLD' if not failures else f'{len(failures)} differences: reread the sources and update docs/toolchain_rules.md')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
