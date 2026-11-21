"""ResNet-20 on CIFAR-10 in PyTorch, FP32 on the CPU, as resnet20.bend trains it.

The same initial weights, the same order of images, crops and flips from the
counter-based generator of lib/math_random.bend, and the same steps:
python resnet20_torch.py THREADS OUTPUT [--epochs E] [--steps S] [--train-files F] [--test-images T].
Each epoch writes OUTPUT/checkpoint.pt, then losses_E.bin and correct_E.bin;
a later run continues after the last epoch written.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.checks.random_reference import below, uniform

DATA = ROOT / 'out/data/cifar10/cifar-10-batches-bin'
RESULTS = ROOT / 'out/results/resnet20'
SEED = 20
BATCH = 128
TEST_BATCH = 100
CLASSES = 10
RECORD = 3073
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
DEVIATION = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001


def rate(epoch):
    """0.1, then 0.01 from epoch 80 and 0.001 from epoch 120, as He et al. train it for 160 epochs."""
    return 0.1 if epoch < 80 else 0.01 if epoch < 120 else 0.001


class Block(torch.nn.Module):
    """Two 3x3 convolutions with batch normalization, the shortcut added before the last ReLU."""

    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.first = torch.nn.Conv2d(inputs, outputs, 3, stride, 1)
        self.first_norm = torch.nn.BatchNorm2d(outputs)
        self.second = torch.nn.Conv2d(outputs, outputs, 3, 1, 1)
        self.second_norm = torch.nn.BatchNorm2d(outputs)
        self.shortcut = None
        if stride != 1 or inputs != outputs:
            self.shortcut = torch.nn.Sequential(torch.nn.Conv2d(inputs, outputs, 1, stride, 0), torch.nn.BatchNorm2d(outputs))

    def forward(self, x):
        y = self.second_norm(self.second(torch.relu(self.first_norm(self.first(x)))))
        return torch.relu(y + (x if self.shortcut is None else self.shortcut(x)))


class ResNet20(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.stem_norm = torch.nn.BatchNorm2d(16)
        blocks = []
        for inputs, outputs, stride in [(16, 16, 1), (16, 32, 2), (32, 64, 2)]:
            blocks += [Block(inputs, outputs, stride), Block(outputs, outputs, 1), Block(outputs, outputs, 1)]
        self.blocks = torch.nn.Sequential(*blocks)
        self.head = torch.nn.Conv2d(64, CLASSES, 1)

    def forward(self, x):
        y = self.blocks(torch.relu(self.stem_norm(self.stem(x))))
        return self.head(y.mean(dim=(2, 3), keepdim=True)).flatten(1)


def convolutions(model):
    """(name, convolution) in the order resnet20.bend reads them."""
    named = [('stem', model.stem)]
    for index, block in enumerate(model.blocks):
        stage, number = index // 3 + 1, index % 3 + 1
        named += [(f'stage{stage}_block{number}_first', block.first), (f'stage{stage}_block{number}_second', block.second)]
        if block.shortcut is not None:
            named.append((f'stage{stage}_block{number}_shortcut', block.shortcut[0]))
    return named + [('head', model.head)]


def initial_model():
    """He initialisation of the convolutions before batch normalization, zero biases, the default initialisation of the head."""
    torch.manual_seed(SEED)
    model = ResNet20()
    for name, convolution in convolutions(model):
        if name != 'head':
            torch.nn.init.kaiming_normal_(convolution.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.zeros_(convolution.bias)
    return model


def load_initial(model, folder):
    for name, convolution in convolutions(model):
        for part, tensor in (('weights', convolution.weight), ('bias', convolution.bias)):
            values = np.fromfile(folder / f'{name}_{part}.bin', dtype=np.float32).reshape(tensor.shape)
            with torch.no_grad():
                tensor.copy_(torch.from_numpy(values))


def records(names):
    return np.concatenate([np.fromfile(DATA / name, dtype=np.uint8).reshape(-1, RECORD) for name in names])


def normalized(images):
    x = torch.from_numpy(images) / 255.0
    return (x - torch.from_numpy(MEAN)) / torch.from_numpy(DEVIATION)


def batch(data, order, rows, columns, flips, first):
    """The images at positions first to first + BATCH of the epoch: each padded by 4, cropped at its offsets, flipped when chosen."""
    images = []
    for position in range(first, first + BATCH):
        image = data[order[position], 1:].reshape(3, 32, 32).astype(np.float32)
        row, column = int(rows[position]), int(columns[position])
        cropped = np.pad(image, ((0, 0), (4, 4), (4, 4)))[:, row:row + 32, column:column + 32]
        images.append(cropped[:, :, ::-1] if flips[position] else cropped)
    labels = data[order[first:first + BATCH], 0].astype(np.int64)
    return normalized(np.ascontiguousarray(np.stack(images))), torch.from_numpy(labels)


def accuracy(model, test, count):
    model.eval()
    correct = 0
    with torch.no_grad():
        for first in range(0, count, TEST_BATCH):
            images = test[first:first + TEST_BATCH, 1:].reshape(-1, 3, 32, 32).astype(np.float32)
            predicted = model(normalized(np.ascontiguousarray(images))).argmax(dim=1)
            correct += int((predicted == torch.from_numpy(test[first:first + TEST_BATCH, 0].astype(np.int64))).sum())
    return correct


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('threads', type=int)
    parser.add_argument('output', type=Path)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=390)
    parser.add_argument('--train-files', type=int, default=5)
    parser.add_argument('--test-images', type=int, default=10000)
    options = parser.parse_args()
    torch.set_num_threads(options.threads)
    torch.set_num_interop_threads(1)
    train = records([f'data_batch_{number}.bin' for number in range(1, options.train_files + 1)])
    test = records(['test_batch.bin'])
    model = ResNet20()
    load_initial(model, RESULTS / 'initial')
    optimizer = torch.optim.SGD(model.parameters(), lr=rate(0), momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    output = options.output
    output.mkdir(parents=True, exist_ok=True)
    start = 0
    while (output / f'losses_{start}.bin').is_file():
        start += 1
    if start:
        state = torch.load(output / 'checkpoint.pt')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
    count = len(train)
    for epoch in range(start, options.epochs):
        for group in optimizer.param_groups:
            group['lr'] = rate(epoch)
        order = np.argsort(uniform(count, SEED, epoch), kind='stable')
        rows, columns, flips = below(count, 9, SEED, 1000 + epoch), below(count, 9, SEED, 2000 + epoch), below(count, 2, SEED, 3000 + epoch)
        model.train()
        losses = []
        began = time.perf_counter()
        for step in range(options.steps):
            x, labels = batch(train, order, rows, columns, flips, step * BATCH)
            loss = torch.nn.functional.cross_entropy(model(x), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        correct = accuracy(model, test, options.test_images)
        seconds = time.perf_counter() - began
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, output / 'checkpoint.pt')
        np.asarray([correct], dtype=np.float32).tofile(output / f'correct_{epoch}.bin')
        np.asarray(losses, dtype=np.float32).tofile(output / f'losses_{epoch}.bin')
        print(f'PyTorch epoch {epoch}, threads {options.threads}: {seconds:.1f} s, loss {np.mean(losses):.4f}, test {correct} of {options.test_images}', flush=True)


if __name__ == '__main__':
    main()
