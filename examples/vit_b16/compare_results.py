"""Compare the native ViT-B/16 outputs with PyTorch, and bit for bit with the run on one thread."""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'out/results/vit_b16'
RELATIVE, ABSOLUTE = 0.0003, 0.0003


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=1)
    options = parser.parse_args()
    reference = json.loads((FOLDER / 'reference.json').read_text())
    native = FOLDER / ('native' if options.threads == 1 else f'native_{options.threads}t')
    rows, passed = [], True
    for name, shape in reference['outputs'].items():
        mine = np.load(native / f'{name}.npy')
        theirs = np.load(FOLDER / 'torch' / f'{name}.npy')
        same = list(mine.shape) == shape and bool(np.isfinite(mine).all() and np.allclose(mine, theirs, rtol=RELATIVE, atol=ABSOLUTE))
        largest = float(np.abs(mine - theirs).max())
        passed &= same
        rows.append(dict(name=name, max_abs=largest, passed=same))
        print(name, 'max_abs', largest, 'PASS' if same else 'FAIL')
    logits = np.load(native / 'logits.npy')[0]
    top = np.argsort(-logits)[:5].tolist()
    passed &= top == reference['top5']
    print('top 5', top, [reference['labels'][reference['top5'].index(i)] if i in reference['top5'] else i for i in top], 'PASS' if top == reference['top5'] else 'FAIL')
    if options.threads != 1 and (FOLDER / 'native/logits.npy').is_file():
        for name in reference['outputs']:
            same = np.load(native / f'{name}.npy').tobytes() == np.load(FOLDER / 'native' / f'{name}.npy').tobytes()
            passed &= same
            if not same:
                print(name, 'differs from one thread')
        print('bit for bit with one thread')
    (FOLDER / f'comparison_{options.threads}t.json').write_text(json.dumps(dict(passed=passed, rows=rows, top5=top), indent=2) + '\n')
    if not passed:
        raise SystemExit('The ViT-B/16 comparison did not pass')


if __name__ == '__main__':
    main()
