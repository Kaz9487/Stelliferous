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


def task_flags(text):
    """Map every task name of a generated C file to 'fork-free' or 'may fork'."""
    table = re.search(r'FID_T\[\]\[3\] = \{(.*?)\};', text, re.S).group(1)
    rows = re.findall(r'\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}', table)
    return {name: 'fork-free' if int(rows[int(number)][2]) & 2 else 'may fork'
            for name, number in re.findall(r'#define (FID_\w+) +(\d+)', text)
            if int(number) < len(rows) and not re.search(r'_[KCJ]\d+$', name)}


def main():
    if len(sys.argv) < 3:
        sys.exit('Usage: fork_flags.py GENERATED.c NAME_PART [...]')
    flags = task_flags(Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace'))
    for name, flag in flags.items():
        if any(part in name for part in sys.argv[2:]):
            print(name, flag)


if __name__ == '__main__':
    main()
