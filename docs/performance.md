# Performance

[English](performance.md) | [繁體中文](performance.zh-TW.md)

YOLOv5n v7.0 forward, the bus image at 640 × 640, FP32, batch one. Median in milliseconds.

| Backend | 1 thread | 4 threads |
|---|---:|---:|
| Stelliferous 0.0.4 | 80 | 35 |
| Stelliferous 0.0.3 | 117 | 48 |
| NumPy / OpenBLAS | 132.5 | 120.8 |
| PyTorch | 48.5 | 14.8 |

Measured on 2026-11-20 on a Google Cloud `c4d-standard-8` virtual machine. CPU affinity and frequency are not pinned. Stelliferous reports whole milliseconds. The released 0.0.3 tree with its own Bend v2.0.37 and 0.0.4 ran alternately four times, eleven forwards per run, so each Stelliferous row has 44 samples, and both versions give the same predictions bit for bit. NumPy and PyTorch come from one `run.py backends` run with two warmups and seven samples, where every setting passed the layer, prediction and detection comparison with the official reference.

## Reproduce

After the [YOLO example setup](../examples/yolov5/README.md):

```sh
python run.py backends      # Stelliferous, NumPy and PyTorch on this tree
python run.py benchmark     # single convolution shapes
```

Samples and the environment go to `out/results/backends/`, and the report is `out/results/backend_comparison.md`. The Benchmark workflow in the Actions tab runs the same comparison on a GitHub-hosted runner. [Benchmarks](../benchmarks/README.md) lists the other workflows.

## The 0.0.3 rows

The 0.0.3 rows come from the tag `v0.0.3` with its own Bend v2.0.37, set up as its getting started says. Each version is measured in its own checkout:

```sh
git clone --branch v0.0.3 https://github.com/Kaz9487/Stelliferous.git stelliferous-0.0.3
cd stelliferous-0.0.3
python run.py model                    # builds out/models/yolov5n, needed below
python run.py backends                 # its YOLOv5n row
python run.py training --threads 4     # its 767 s epoch
```

For the YOLOv5n table the two compiled models ran alternately in one session, to cancel drift between runs. The loop runs in the folder that holds both checkouts, where `stelliferous` is this version after `python run.py model`. 0.0.3 takes its options without `+RTS`:

```sh
# 11 forwards of each version, four rounds, 1 and 4 threads
for round in 1 2 3 4
do
  for threads in 1 4
  do
    (cd stelliferous-0.0.3 && YOLO_EXTRA=10 YOLO_SIZE=640 \
      out/models/yolov5n --threads $threads --gpu off | grep '^forward')
    (cd stelliferous && YOLO_EXTRA=10 YOLO_SIZE=640 \
      out/models/yolov5n +RTS --threads $threads --gpu off -RTS | grep '^forward')
  done
done
```

## Environment

- Google Cloud `c4d-standard-8`, Ubuntu 24.04: AMD EPYC 9B45, 8 virtual CPUs (4 cores, 2 threads each).
- Bend v2.0.40 (official, unmodified), Clang 19.1.1 with `-O3 -march=native -ffp-contract=off -std=c11`.
- NumPy 2.4.6 with its bundled OpenBLAS, PyTorch 2.14.0 and torchvision 0.29.0 (CPU, eager, FP32), one inter-op thread.

## What is timed

The forward computation: allocation, ownership copies, packing, kernels, gather and detection decode. Loading, compilation, preprocessing, diagnostic output and non-maximum suppression are outside the timer.

## Samples

| Setting | Median (ms) | Min–max (ms) |
|---|---:|---:|
| Stelliferous 0.0.4, 1 thread | 80 | 76–86 |
| Stelliferous 0.0.4, 4 threads | 35 | 33–41 |
| Stelliferous 0.0.3, 1 thread | 117 | 113–125 |
| Stelliferous 0.0.3, 4 threads | 48 | 46–57 |
| NumPy, 1 thread | 132.5 | 128.9–136.7 |
| NumPy, 4 threads | 120.8 | 118.5–122.3 |
| PyTorch, 1 thread | 48.5 | 47.6–49.6 |
| PyTorch, 4 threads | 14.8 | 14.3–15.1 |

## ViT-B/16

The official torchvision ViT-B/16 with its ImageNet weights, one 224 × 224 image, FP32. Stelliferous 0.0.4 and PyTorch ran alternately three times, ten forwards per run after one PyTorch warmup, on the same machine on the same day.

| Backend | 1 thread | 4 threads |
|---|---:|---:|
| Stelliferous 0.0.4 | 360 (349–380) | 176 (171–184) |
| PyTorch | 353 (350–356) | 121 (120–124) |

```sh
python run.py vit --threads 4 --timing
```

The setup and the comparison with PyTorch are in the [ViT-B/16 example](../examples/vit_b16/README.md).

## Training

ResNet-20 on CIFAR-10 for 160 epochs from the same initial weights and batches. An epoch is 390 steps of batch 128 and the test of 10,000 images. Measured once on 2026-11-20 on the same machine type with Stelliferous 0.0.4, four threads.

| Library | Epoch, mean | Test accuracy, last epoch | Best test accuracy |
|---|---:|---:|---:|
| Stelliferous 0.0.4 | 137.5 s | 91.40% | 91.57% |
| PyTorch 2.14.0 | 34.5 s | 91.34% | 91.54% |

0.0.3 cannot train ResNet-20. At its release on 2026-10-21 it took 767 s for one epoch of 390 steps without the test, on the ResNet-20 shape without residual additions or projections, with plain SGD and one synthetic batch reused. [The 0.0.3 rows](#the-003-rows) shows how to measure it.

Without the test, one epoch of 0.0.4 takes 238 s on one thread and 112 s on four, measured on 2026-11-20 on the same machine type. These are the times in the changelog, and like the 767 s of 0.0.3 they leave the test out. After `python run.py training` has prepared the data and compiled the model:

```sh
# threads, epochs, steps, training files, test images (none), output folder
sh examples/resnet20/build_run.sh 1 1 390 5 0 out/results/resnet20/epoch_1t
sh examples/resnet20/build_run.sh 4 1 390 5 0 out/results/resnet20/epoch_4t
```

```sh
python run.py training --threads 4 --epochs 160
```

The network, data and schedule are described in the [ResNet-20 example](../examples/resnet20/README.md).

## Other convolution shapes

Two shapes carry extra costs: many weights with few output positions, and outputs narrower than eight positions. See [design](design.md#costs-that-remain).
