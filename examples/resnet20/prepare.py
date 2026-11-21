"""Unpack the CIFAR-10 binary archive and write the initial weights that resnet20.bend and resnet20_torch.py both read."""

import hashlib
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.resnet20.resnet20_torch import DATA, RESULTS, convolutions, initial_model

ARCHIVE = ROOT / 'out/data/cifar10_download/cifar-10-binary.tar.gz'
ARCHIVE_MD5 = 'c32a1d4ab5d03f1284b67883e8d87530'
FILES = [f'data_batch_{number}.bin' for number in range(1, 6)] + ['test_batch.bin']


def unpacked():
    if all((DATA / name).is_file() for name in FILES):
        return
    if not ARCHIVE.is_file():
        raise SystemExit(f'Missing {ARCHIVE}: see examples/resnet20/README.md')
    digest = hashlib.md5(ARCHIVE.read_bytes()).hexdigest()
    if digest != ARCHIVE_MD5:
        raise SystemExit(f'{ARCHIVE} has MD5 {digest}, not the published {ARCHIVE_MD5}')
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(DATA.parent, filter='data')


def main():
    unpacked()
    folder = RESULTS / 'initial'
    folder.mkdir(parents=True, exist_ok=True)
    for name, convolution in convolutions(initial_model()):
        convolution.weight.detach().numpy().tofile(folder / f'{name}_weights.bin')
        convolution.bias.detach().numpy().tofile(folder / f'{name}_bias.bin')
    print(f'CIFAR-10 in {DATA}, initial weights in {folder}')


if __name__ == '__main__':
    main()
