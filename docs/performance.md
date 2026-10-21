# Performance

[English](performance.md) | [繁體中文](performance.zh-TW.md)

YOLOv5n v7.0 forward, the bus image at 640 × 640, FP32, batch one. Two warmups and seven samples per setting, median in milliseconds.

| Backend | 1 thread | 4 threads |
|---|---:|---:|
| Stelliferous 0.0.3 | 117 | 47 |
| Stelliferous 0.0.2 | 115 | 44 |
| Stelliferous 0.0.1 | 153.7 | 54.8 |
| NumPy / OpenBLAS | 133.7 | 116.3 |
| PyTorch | 51.1 | 15.2 |

Measured on 2026-10-21 on a Google Cloud `c4d-standard-8` virtual machine. CPU affinity and frequency are not pinned. Every setting passed the layer, prediction and detection comparison with the official reference. Stelliferous reports whole milliseconds. The comparison ran four times, alternating the released 0.0.2 tree with its own Bend v2.0.36 and 0.0.3, so each Stelliferous row has 14 samples and NumPy and PyTorch have 28. Stelliferous 0.0.1 is the released tree with its own Bend v2.0.35, measured in the same way on 2026-10-10 on the same machine type.

## Reproduce

After the [YOLO example setup](../examples/yolov5/README.md):

```sh
python run.py backends      # the table above
python run.py benchmark     # single convolution shapes
```

Samples and the environment go to `out/results/backends/`, and the report is `out/results/backend_comparison.md`. The Benchmark workflow in the Actions tab runs the same comparison on a GitHub-hosted runner. [Benchmarks](../benchmarks/README.md) lists the other workflows.

## Environment

- Google Cloud `c4d-standard-8`, Ubuntu 24.04: AMD EPYC 9B45, 8 virtual CPUs (4 cores, 2 threads each).
- Bend v2.0.37 (official, unmodified), Clang 19.1.1 with `-O3 -march=native -ffp-contract=off -std=c11`.
- NumPy 2.4.6 with its bundled OpenBLAS, PyTorch 2.14.0 (CPU, eager, FP32), one inter-op thread.

## What is timed

The forward computation: allocation, ownership copies, packing, kernels, gather and detection decode. Loading, compilation, preprocessing, diagnostic output and non-maximum suppression are outside the timer.

## Samples

| Setting | Median (ms) | Min–max (ms) |
|---|---:|---:|
| Stelliferous 0.0.3, 1 thread | 117 | 115–121 |
| Stelliferous 0.0.3, 4 threads | 47 | 46–50 |
| Stelliferous 0.0.2, 1 thread | 115 | 112–116 |
| Stelliferous 0.0.2, 4 threads | 44 | 44–47 |
| Stelliferous 0.0.1, 1 thread | 153.7 | 153.4–154.4 |
| Stelliferous 0.0.1, 4 threads | 54.8 | 54.6–56.5 |
| NumPy, 1 thread | 133.7 | 129.9–141.4 |
| NumPy, 4 threads | 116.3 | 113.9–122.1 |
| PyTorch, 1 thread | 51.1 | 49.0–52.4 |
| PyTorch, 4 threads | 15.2 | 14.7–16.5 |

## Training

One epoch of a ResNet-20-shaped network: 390 steps of batch 128, each a forward pass, cross entropy, a backward pass and SGD. Measured once on 2026-10-21 on the same machine type, four threads.

| Library | Epoch | Loss after the last step |
|---|---:|---:|
| Stelliferous 0.0.3 | 767 s | 0.026418747 |
| PyTorch 2.14.0 | 31.1 s | 0.02641915 |

```sh
python run.py training --threads 4
```

The network, data and timer are described in the [training benchmark](../benchmarks/training/README.md).

## Other convolution shapes

Some shapes gain little from more workers: many weights with few output positions, batches of more than one image, and outputs narrower than six positions. See [design](design.md#known-limits).
