"""The network, images, initial values and steps of epoch.bend in PyTorch, FP32 on the CPU: python epoch_torch.py THREADS."""

import sys
import time

import torch

STEPS, BATCH, CLASSES = 390, 128, 10


def blocks():
    """The (inputs, outputs, stride) of the 19 convolutions, as in epoch.bend."""
    layers = [(3, 16, 1)]
    for inputs, outputs, stride in [(16, 16, 1), (16, 32, 2), (32, 64, 2)]:
        layers += [(inputs, outputs, stride)] + [(outputs, outputs, 1)] * 5
    return layers


def network():
    layers = []
    for inputs, outputs, stride in blocks():
        convolution = torch.nn.Conv2d(inputs, outputs, 3, stride, 1)
        torch.nn.init.constant_(convolution.weight, 0.01)
        torch.nn.init.zeros_(convolution.bias)
        layers += [convolution, torch.nn.BatchNorm2d(outputs, eps=1e-5, momentum=0.1), torch.nn.ReLU()]
    head = torch.nn.Conv2d(64, CLASSES, 1)
    torch.nn.init.constant_(head.weight, 0.01)
    torch.nn.init.zeros_(head.bias)
    return torch.nn.Sequential(*layers, torch.nn.AdaptiveAvgPool2d(1), head, torch.nn.Flatten())


def main():
    threads = int(sys.argv[1])
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model = network().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = (torch.arange(0, BATCH * 3 * 32 * 32, dtype=torch.float32) * 0.00001).reshape(BATCH, 3, 32, 32)
    labels = torch.zeros(BATCH, dtype=torch.long)
    losses = []
    start = time.perf_counter()
    for _ in range(STEPS):
        loss = torch.nn.functional.cross_entropy(model(x), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    elapsed = round((time.perf_counter() - start) * 1000)
    print(f'PyTorch epoch of {STEPS} steps, batch {BATCH}, threads {threads}: {elapsed} ms, loss {losses[0]:.7g} ... {losses[-1]:.7g}')


if __name__ == '__main__':
    main()
