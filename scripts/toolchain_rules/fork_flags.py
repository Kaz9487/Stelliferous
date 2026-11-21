"""List which tasks in a generated C file the runtime treats as fork-free.

Run: python -B scripts/toolchain_rules/fork_flags.py GENERATED.c NAME_PART [...]
It prints every task whose FID name contains one of the name parts. The grow
turn skips a fork-free task and leaves it for the work turn (rule H4 in
docs/toolchain_rules.md); a task that may fork runs in the grow turn.
Continuation, closure and join segments are left out.
"""
import re
import sys
from pathlib import Path


def normalized(text):
    """Task and constructor names in upper case, as the probes spell them:
    the generated C keeps the case of the source (FID_run, CID_True)."""
    return re.sub(r'\b(?:WL_)?[FC]ID_\w+', lambda name: name[0].upper(), text)


def constructor_flags(text):
    """The hot flag of every constructor id. Each row is { arity, hot, spine }."""
    table = re.search(r'CID_T\[\] = \{(.*)\};', text)
    if table is None:
        return None
    return [int(flag) for flag in re.findall(r'\{\s*\d+\s*,\s*(\d+)\s*,\s*\d+\s*\}', table.group(1))]


def task_flags(text):
    """Map every task name of a generated C file to 'fork-free' or 'may fork'.
    Each row is { arity, resw, bangs, nofk }."""
    text = normalized(text)
    table = re.search(r'FID_T\[\] = \{(.*?)\};', text, re.S).group(1)
    free = [int(row) for row in re.findall(r'\{\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*\}', table)]
    return {name: 'fork-free' if free[int(number)] else 'may fork'
            for name, number in re.findall(r'#define (FID_\w+) +(\d+)', text)
            if int(number) < len(free) and not re.search(r'_[KCJ]\d+$', name)}


def main():
    if len(sys.argv) < 3:
        sys.exit('Usage: fork_flags.py GENERATED.c NAME_PART [...]')
    flags = task_flags(Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace'))
    for name, flag in flags.items():
        if any(part.upper() in name for part in sys.argv[2:]):
            print(name, flag)


if __name__ == '__main__':
    main()
