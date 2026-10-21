# Training epoch

[English](README.md) | [繁體中文](README.zh-TW.md)

Times one epoch of training in Stelliferous and in PyTorch, on the same machine and with the same number of threads.

```sh
python run.py training --threads 4
```

Both lines go to `out/results/training/epoch_4.txt`. Published numbers are in [performance](../../docs/performance.md#training).

## Method

- The network has the shape of ResNet-20 for CIFAR-10 without its residual additions: 19 blocks of a 3x3 convolution, batch normalization and ReLU (16, 32 and 64 channels, stride 2 at the start of the second and third stage), the average pool and a 1x1 convolution to 10 classes.
- 390 steps of batch 128, so 49,920 images of 3 × 32 × 32. One synthetic batch is reused, so reading data is not timed.
- Each step is forward, cross entropy, backward and SGD with rate 0.1. Every weight starts at 0.01 and every bias at 0.
- The timer covers the whole epoch. Both programs print the first and last loss.

## Files

| File | Role |
|---|---|
| [epoch.bend](epoch.bend) | The epoch through the public `Train` API |
| [epoch_torch.py](epoch_torch.py) | The same network, data and steps in PyTorch |
| [run_epoch.sh](run_epoch.sh) | Builds the Bend program and runs both |
