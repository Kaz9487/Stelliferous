# Changelog

[English](CHANGELOG.md) | [繁體中文](CHANGELOG.zh-TW.md)

This file records the notable changes in each version of Stelliferous. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Before 1.0.0, the public API may still change between minor versions.

## [0.0.4] - 2026-11-21

Stelliferous 0.0.4 trains a full ResNet-20, runs a Vision Transformer, and is much faster. The main changes are

- Networks with residual blocks train end to end. ResNet-20 trains on CIFAR-10 to the same accuracy as PyTorch from the same weights and batches.
- Matrix products, layer normalization and the exact GELU run the official ViT-B/16 from its torchvision weights, in 176 ms at four threads.
- YOLOv5n inference takes 80 ms at one thread and 35 ms at four, against 117 and 48 ms in 0.0.3.
- A training epoch is about seven times faster than in 0.0.3, and the gap to PyTorch narrows from about 25 times to about 4.

This version moves to Bend v2.0.40, and imports again on current Bend. The published 0.0.3 does not import on Bend v2.0.38 or later.

### Upgrade notes

- **Runtime options go between `+RTS` and `-RTS`**

  Since Bend v2.0.38 a compiled program reads its runtime options only inside `+RTS ... -RTS`. Previously `./program --threads 4` set four threads. Now it passes `--threads 4` to the program itself, and the program uses every CPU. Write `./program +RTS --threads 4 --gpu off -RTS` instead.

- **Batch normalization rejects an axis of length zero**

  The forward pass, the backward pass and `evaluate` of a batch normalization layer now return a failed result with `InvalidShape` when any axis of the input has length zero.

- **`max_pool2d` rejects tensors of rank 0 and 1**

  0.0.3 accepted them. They now return a failed result.

- **The internal storage of tensors has changed**

  `F.Tensor()` now holds its values in cells of eight. The public functions behave as before. Programs that imported the internal modules `tensor_band*.bend` must move to the public `F` functions, because those modules are gone.

- **`python run.py training` trains ResNet-20**

  It used to time one epoch of a network without residual additions on one synthetic batch. It now trains ResNet-20 on CIFAR-10 and compares it bit for bit across thread counts and with PyTorch, and `--epochs E` trains E full epochs. The old `benchmarks/training/` folder is removed.

### Added

- **Networks and residual blocks**

  `Train.Network` chains layers and residual blocks. `forward_network(x, network)` returns the output and a tape, `backward_network(dy, tape)` returns the input gradient and the network holding every layer's gradients, and `sgd_network(network, rate)` applies them. `evaluate_network` runs the network for inference. A `Train.Residual` block adds its branch and its shortcut, and its backward pass gives the input the sum of both gradients.

  `save_network(folder, network)` writes the parameters, running statistics and velocities as raw FP32 files, and `restore_network(folder, network)` reads them back, so a training can continue after any epoch.

- **SGD with momentum and weight decay**

  `sgd_step_with(layer, rate, momentum, weight_decay)` and `sgd_network_with(network, rate, momentum, weight_decay)` follow PyTorch's SGD without dampening. The layer keeps its velocities for the next step.

- **Layer normalization**

  `layer_norm_layer(features)`, `layer_norm_layer_with(features, epsilon)` and `layer_norm_layer_from(gamma, beta, epsilon)` normalize over the last axis, with forward and backward passes.

- **Matrix products**

  `matmul(left, right)` multiplies matrices with the same leading axes, or by one shared right matrix. `linear(x, weight, bias)` reads the weight in PyTorch's `[outputs, inputs]` order. Each output adds its products in order of the inner index from +0.0, the same on any number of workers.

- **More tensor functions**

  `gelu` is the exact GELU, PyTorch's `approximate='none'`, with `gelu_derivative`. Over all 2³² FP32 inputs its largest error is 2.76 ULP, and 3.79 ULP for the derivative. `argmax(tensor, axis)` gives the place of the first largest element. `depthwise_conv2d` convolves each channel with its own weights. `Act.gelu()` lets convolutions and `activation_layer` use the exact GELU.

- **Random values and byte files**

  `random_uniform(shape, seed, stream)` and `random_below(shape, bound, seed, stream)` give the same values on any machine and any number of workers. `load_u8(path, shape)` reads one byte per value, and `read_storage(storage, index)` reads one value of a storage.

- **ResNet-20 and ViT-B/16 examples**

  [ResNet-20](examples/resnet20/README.md) trains on CIFAR-10 beside a PyTorch twin that uses the same initial weights and batches. `python run.py training` checks two short epochs bit for bit on 1, 2, 4 and 8 threads and after a resume.

  [ViT-B/16](examples/vit_b16/README.md) runs the official torchvision weights and compares each block, the logits and the top five classes with PyTorch. `python run.py vit` builds, runs and compares it.

### Fixed

- **`load_checkpoint` reads saved state dicts**

  The file that `torch.save(model.state_dict())` writes keeps attributes such as `_metadata` beside the entries. 0.0.3 stopped on it with `BUILD on a value that is not an object`. 0.0.4 reads the entries and leaves the attributes.

### Developer tools

- `python run.py dependencies` lists where each Bend declaration and file is used, the unused imports and the declarations nothing reaches.
- The operation timeline of `benchmarks/model/profile_timeline.py` gives the time and the entries of each library module.
- `benchmarks/checks/math_accuracy.py` also measures `gelu`.
- The planner report of the YOLOv5n graph, `benchmarks/model/report_plans.py` and `plan_observation.bend`, is removed. `benchmarks/convolution/report_plans.py` still reports the plans of single convolution shapes.

### Performance

YOLOv5n single inference on a 640 × 640 input image, on the same test machine.

| Version | 1 thread | 4 threads |
| ----- | -------: | ------: |
| 0.0.3 | 117 ms | 48 ms |
| 0.0.4 | 80 ms | 35 ms |

The two versions were measured alternately on the same machine.

ViT-B/16 single inference on a 224 × 224 input image, on the same machine:

| Library | 1 thread | 4 threads |
| ----- | -------: | ------: |
| Stelliferous 0.0.4 | 360 ms | 176 ms |
| PyTorch 2.14 | 353 ms | 121 ms |

ResNet-20 on CIFAR-10 for 160 epochs from the same initial weights and batches, at four threads on the same machine. An epoch is 390 steps of batch 128 and a test of 10,000 images.

| Library | Epoch, mean | Test accuracy, last epoch |
| ----- | -------: | ------: |
| Stelliferous 0.0.4 | 137.5 s | 91.40% |
| PyTorch 2.14 | 34.5 s | 91.34% |

0.0.3 cannot train ResNet-20. At its release it took 767 s for 390 steps of the same shape without residual additions, on one synthetic batch. 0.0.4 takes 112 s for 390 steps of the real ResNet-20, without the test.

For the full results, see the [performance document](docs/performance.md).

- **A new GEMM kernel**

  The GEMM of 0.0.3 computed 4 output channels by 8 positions at a time and served the forward pass and the weight gradient of convolutions. The GEMM of 0.0.4 computes blocks of 8 by 8. The forward pass, the input gradient and the weight gradient of convolutions, `matmul` and `linear` all run on it, and it writes its results straight into the storage of tensors, in cells of eight.

  A 1 × 1 convolution of stride 1 reads its input in place and computes 16 output channels at a time, without rearranging the input first. When the input channels are not a multiple of 16, the rest are added on their own, with no padding and no extra products.

  When the weights do not fit in the cache block, the GEMM changes its order so that the weights come from memory once, instead of once for every eight positions. The linear layers of ViT-B/16 gain the most.

- **A new weight gradient**

  0.0.3 computed the weight gradient with the kernel positions as rows, as [position][output channel], and transposed it to [output channel][position] at the end. 0.0.4 reads the cells of eight of the tensors directly, computes 8 input channels by 8 output channels at a time, and writes the final layout without a transpose.

  The work is split over the output channels as before, and now also over the images, so layers with few output channels use every worker too. The order of the sums does not change, so the result is the same on any number of workers. The bias gradient sums eight output channels at a time.

- **No conversions between layers**

  Tensors keep the same storage from input to output. Element-wise functions, activations, pooling and reductions work on it directly, so layers no longer rearrange values between them.

- **Fewer passes in batch normalization**

  The statistics come from two reductions, and the normalization runs in one parallel pass.

- **Batches on four workers**

  In 0.0.3 a batch of small images with many channels ran slower on four workers than on one. A ResNet-20 epoch of batches of 128 now takes 238 s on one thread and 112 s on four.

- **Even shares of work**

  The work of an operation is cut into even shares that keep every worker busy in each round, so the last round no longer waits on a single worker.

- **Fewer copies in convolutions and matrix products**

  Matrix products read their inputs and weights in place, without a copy for each share of the work. Convolutions other than 1 × 1 transpose each input value once and build their packed tiles by shifting lanes, instead of gathering the value again for every kernel position that reads it.

### Formal proofs

- **New formal proofs**

  `matmul`, `linear`, `depthwise_conv2d`, layer normalization, `argmax`, the random values, `one_hot`, SGD with momentum and weight decay, and the forward, backward and update of whole networks with residual blocks are proved. Every output element equals the value the specification defines. For details, see the [proofs document](docs/proofs.md).

  Existing functions without a value theorem in 0.0.3 have one now: `apply`, `apply_indexed`, `arange`, `clamp`, `conv2d_custom`, `from_list`, `from_storage`, `zeros`, `ones`, `max_pool2d_with` and `pad_edge`.

- **The proofs follow the faster code**

  The proofs cover the new storage of tensors, the splitting of work and the new convolution kernels. The proof that the native inference equals the matrix specification keeps its statement and conditions. `python run.py proofs` checks 569 modules and 20,051 declarations with the official checker.

### Known limitations

- Training provides SGD only, and there is no automatic differentiation. A network is a fixed chain of layers and residual blocks.
- The numerical errors come from measurement. There is no formal proof of an error bound.
- The proofs are verified by the official checker with `python run.py proofs`. `bend --verdict` is not used.
- Only the FP32 (F32) data type is supported.
- Convolutions with many weights and few output positions, and outputs narrower than eight positions, usually gain little from more workers. See the [design document](docs/design.md#costs-that-remain).
- Inference is still slower than PyTorch, and training about 4 times slower. See the [performance document](docs/performance.md).

## [0.0.3] - 2026-10-21

Stelliferous 0.0.3 adds training, and reads PyTorch checkpoints directly in Bend. The main changes are

- Convolution, batch normalization, activation and average pool layers have forward and backward passes, with the mean squared error, cross entropy and SGD. Training gives bit-for-bit the same results with any number of workers.
- PyTorch `.pt` checkpoints can be read directly in Bend. The YOLOv5n example now runs from the official `yolov5n.pt`.
- `softmax`, `select`, batch normalization, the losses and the training layers have formal proofs.

This version moves to Bend v2.0.37.

### Upgrade notes

- **`sum_axis`, `mean_axis`, `max_axis` and `min_axis` become `sum_axes`, `mean_axes`, `max_axes` and `min_axes`**

  The new functions take a list of axes. Rewrite `sum_axis(t, a)` as `sum_axes(t, [a])`, and the other three in the same way. Listing an axis twice, or an axis beyond the rank of the tensor, returns a failed result.

- **The numerical results of sums have changed**

  `sum`, `mean`, `sum_axes`, `mean_axes`, the denominator of `softmax` and the mean squared error now add in eight lanes with error compensation (TwoSum) and merge them in a fixed order. The results are more accurate, and still bit-for-bit identical on different machines and with any number of workers.

  The new outputs may differ from 0.0.2 in the last bits. If you used 0.0.2 outputs as a test baseline, regenerate your reference results.

- **The numerical results of `gelu_tanh` have changed**

  `gelu_tanh` is rewritten for accuracy. Over all 2³² FP32 inputs the largest error is now 1.99 ULP, where 0.0.2 reached 15 ULP for inputs between -2 and -0.5. If you used 0.0.2 outputs of `gelu_tanh` as a test baseline, regenerate your reference results.

- **`max_pool2d` returns a failed result for an input of height or width 0**

  0.0.2 returned a result of -∞ in that case. An output with no elements still succeeds.

- **`upsample_nearest` returns a failed input as it is**

  0.0.2 returned the failure with the enlarged shape. It now keeps the shape of the input. The reason of the failure is unchanged.

- **How `load_weights` reports a missing `names.txt`**

  0.0.2 returned empty weights, and `take` then only reported a missing name. Now every call to `take` reports that `names.txt` could not be read.

### Added

- **Training layers**

  The new module `tensor_training.bend`, imported as `Train`, represents each layer as a value. `conv2d_layer`, `batch_norm_layer`, `activation_layer` and `average_pool_layer` create the layers.

  `forward(x, layer)` returns the output, what the backward pass needs, and the layer. `backward(dy, saved, layer)` returns the input gradient and the layer holding its parameter gradients, and `sgd_step(layer, rate)` applies them. `evaluate(x, layer)` runs a layer in inference mode.

  Batch normalization computes as PyTorch's `BatchNorm2d` in training mode, and updates its running statistics with the unbiased variance.

  `examples/training/learning.bend` trains two blocks of convolution, batch normalization and SiLU for 40 steps. For usage, see the [API reference](lib/README.md#training) and [Getting started](docs/getting_started.md).

- **Losses**

  `Train.mse(prediction, target)` and `Train.cross_entropy(logits, target)` return the loss, its gradient and the target. `Train.one_hot(labels, classes)` turns class numbers into the form `cross_entropy` takes.

- **Activation gradients**

  `activate_forward`, `activate_backward` and `activate_derivative`, and `sigmoid_derivative`, `silu_derivative`, `tanh_derivative` and `gelu_tanh_derivative`.

- **Reading PyTorch checkpoints**

  `load_checkpoint(path)` reads a `.pt` file written by `torch.save`. Each state-dict entry becomes one tensor under its PyTorch name. FP16 and BFloat16 values convert to FP32 exactly, FP64 values and integers round to the nearest FP32.

  The 349 tensors of the official `yolov5n-v7.0.pt` read bit-for-bit equal to `torch.load`. Compressed entries and ZIP64 files are not supported yet.

  `weight_names()` lists the names in a set of weights. The YOLOv5n example folds the batch normalizations of the official checkpoint into the convolution weights in Bend.

- **Tensor operations**

  `sqrt(tensor)` and `flip(tensor, axes)`.

### Performance

YOLOv5n single inference on a 640 × 640 input image, on the same test machine.

| Version | 1 thread | 4 threads |
| ----- | -------: | ------: |
| 0.0.2 | 115 ms | 44 ms |
| 0.0.3 | 117 ms | 47 ms |

The two versions were measured alternately on the same machine. Inference is slightly slower than in 0.0.2, by about 2% at one thread and 7% at four threads.

One training epoch of a ResNet-20-shaped network, 390 steps of batch 128 on 32 × 32 images, on the same machine at four threads:

| Library | Time |
| ----- | -------: |
| Stelliferous 0.0.3 | 767 s |
| PyTorch 2.14 | 31.1 s |

Both start from the same weights and reach the same loss. `python run.py training` runs the comparison.

For the full results, see the [performance document](docs/performance.md).

- **Training uses all workers**

  A batch of several images can be split across workers, and convolution weight gradients are split across workers by output channel.

- **Layout operations work on each worker's data**

  `flip`, `transpose`, `slice`, `broadcast_to`, `concat` along any axis and the reductions run on the parts of a tensor that each worker holds, without first gathering the whole tensor. The results are unchanged.

### Formal proofs and numerical accuracy

- **New formal proofs**

  `softmax`, `select`, the activation derivatives, batch normalization, `mse`, `cross_entropy`, the average pool, the forward and backward passes of the convolution layer and `sgd` are proved.

  These proofs establish that every output element equals the value the specification defines. Reading a tensor back is also proved to give the values its contract describes. For details, see the [proofs document](docs/proofs.md).

- **Reorganized proofs**

  Every operation now has one execution path and one proof. The proof of walking over the parts of a tensor is shared by all operations, each operation proves only its own step, and duplicate lemmas were removed. `upsample_nearest` now relies entirely on the shared proof of reading through a view, and the proof files of their own for `upsample_nearest`, `max_pool2d` and `concat` went from 1.6 MB to 0.7 MB.

- **Numerical error test**

  `gelu_tanh` was evaluated on all 2³² FP32 bit patterns, with the correctly rounded result computed by MPFR as the reference. The largest error is 1.99 ULP.

### Known limitations

- Training provides SGD only. The layers of a network are chained by hand, and there is no automatic differentiation yet.
- `one_hot` has only been tested against PyTorch and has no formal proof yet.
- The numerical errors come from measurement. There is no formal proof of an error bound.
- The proofs are verified by the official checker with `python run.py proofs`. `bend --verdict` is not used.
- Only the FP32 (F32) data type is supported.
- For convolutions with many weights and few output positions, and for an output width below 6, more workers usually bring no clear performance gain. With a batch size above 1, small images with many channels run slower on four workers. See the [design document](docs/design.md).
- Inference is still slower than PyTorch, and training about 25 times slower. See the [performance document](docs/performance.md).

### Thanks

Thanks to [@nuxyel](https://github.com/nuxyel) of [bend-ml](https://github.com/nuxyel/bend-ml) for reports on the behaviour of Bend's compiler, now part of the [toolchain rules](docs/toolchain_rules.md).

## [0.0.2] - 2026-10-10

Stelliferous 0.0.2 adds file reading and writing and model weight loading, and improves YOLOv5n inference performance. The main changes are

- NumPy `.npy` files can be read and written directly in Bend.
- A model can be defined completely as a Bend program that takes each layer's weights by name.
- YOLOv5n inference is faster. Single-thread latency drops from 153.7 ms to 110 ms, and four-thread latency from 54.8 ms to 43 ms.

From this version on, the project is renamed from **Bend-Conv** to **Stelliferous**, and moves to Bend v2.0.36.

### Upgrade notes

- **`conv2d_with` is renamed `conv2d_custom`**

  The call and its arguments are unchanged. Only the function name needs updating.

- **`activate` no longer supports custom functions**

  Passing a custom activation function to `activate` now returns a failed result. To use a custom activation, call `apply` or `conv2d_custom` instead.

- **The numerical results of some mathematical functions have changed**

  `sigmoid`, `silu`, `tanh` and `gelu_tanh` use a new algorithm that produces bit-for-bit identical results on different machines.

  The new outputs may differ from 0.0.1 in the last bits. If you used 0.0.1 outputs as a test baseline, regenerate your reference results.

### Added

- **File reading and writing**

  `load_npy(path)` and `save_npy(path, tensor)` read and write the NumPy `.npy` format.

  `load_raw(path, shape)` and `save_raw(path, tensor)` read and write raw FP32 data without a header.

- **Model weight management**

  `load_weights(folder)` loads every weight in a folder at once. While the model runs, `take(name)` takes the weights it needs by name, and `run(model, weights)` runs the whole model.

  For the other related functions, see the [API reference](lib/README.md#models).

- **Mathematical functions**

  `exp`, `log` and `softmax(tensor, axis)`.

- **Comparison and conditional selection**

  Six elementwise comparisons are added: `less`, `less_equal`, `greater`, `greater_equal`, `equal` and `not_equal`.

  The result is a tensor of 1.0 and 0.0, meaning the condition holds or does not hold at that position. All six have a `_scalar` form that compares with a single number.

  `select(mask, yes, no)` picks each element from one of two tensors according to a mask, and `clamp(tensor, low, high)` limits the elements of a tensor to a given range.

- **A complete YOLOv5n example**

  `examples/yolov5/yolov5n.bend` implements the whole YOLOv5n network as a single Bend program on the public API, for input images whose side is a multiple of 32.

  `examples/save_and_load.bend` shows how to save and load tensors.

### Performance

YOLOv5n single inference on a 640 × 640 input image, on the same test machine.

| Version | 1 thread | 4 threads |
| ----- | -------: | ------: |
| 0.0.1 | 153.7 ms | 54.8 ms |
| 0.0.2 |   110 ms |   43 ms |

For the full results, see the [performance document](docs/performance.md).

The improvement comes from the following changes.

- **Convolution runs in blocks**

  Inside a convolution, the input arrangement and the multiply-add work are cut into blocks of a fixed size that reuse one piece of working memory, instead of allocating and initializing one large temporary area for the whole layer. One thread and several threads use the same procedure.

- **Activations process eight elements at a time**

  `sigmoid`, `silu`, `tanh` and `gelu_tanh` are computed eight elements at a time, which compiles to SIMD instructions. This applies to an activation that follows a convolution and to one called on its own.

- **No calls to the system math library**

  In 0.0.1, `sigmoid`, `silu`, `tanh` and `gelu_tanh` went through the system's `exp` and `tanh`, converting to double precision and back to FP32. The new versions use only FP32 addition, subtraction, multiplication and division and 32-bit integer operations. They are faster, and their results no longer change with the version of the system library.

- **YOLOv5n detection decoding is tensor arithmetic**

  The decoding step uses the library's tensor operations and `sigmoid`, sharing one implementation with the rest of the network.

- **Updated cost estimates in the work planner**

  The planner estimates memory use from the blocked procedure, and counts the cost of each activation from its actual instructions, so that the division of work across threads follows the real cost more closely.

- **Shorter compile time for the model program**

  The YOLOv5n program generated from the official weights is now a list of operations run by a single loop. Compiling the 640 × 640 program takes about 14 seconds instead of about 6 minutes, and needs under 1 GB of memory instead of more than 12 GB.

### Formal proofs and numerical accuracy

- **New formal proofs**

  Elementwise semantics are proved for the six comparisons and for `sigmoid`, `silu`, `tanh`, `gelu_tanh`, `exp` and `log`.

  These proofs establish that every output element is the result of applying the given function to the corresponding input element. For details, see the [proofs document](docs/proofs.md).

- **Proofs for blocked convolution**

  The new blocked procedure is proved, for a single task and for parallel tasks. The unconditional theorem that native convolution matches the matrix specification for every input still holds.

- **Proofs for file reading and writing**

  Writing a value as its bits and reading it back is proved to give exactly the same bits, NaN payloads included.

- **Numerical error test**

  `sigmoid`, `silu`, `tanh`, `exp` and `log` were evaluated on all 2³² FP32 bit patterns, with the correctly rounded result computed by MPFR as the reference.

  The largest measured errors, in ULP (units in the last place), are below.

  | `sigmoid` | `silu` | `tanh` | `exp` | `log` |
  | --------: | -----: | -----: | ----: | ----: |
  |      1.81 |   1.80 |   1.06 |  0.80 |  0.52 |

  `benchmarks/checks/math_accuracy.py` reproduces the test.

### Known limitations

- **`gelu_tanh` is still inaccurate for negative inputs.** For inputs between -2 and -0.5 the largest error reaches 15 ULP. Below -2 the result is close to zero, and its reliability cannot be guaranteed yet.
- `select`, `softmax` and the other activation functions have only been tested against NumPy and have no formal proof yet.
- The numerical errors above come from measurement. There is no formal proof of an error bound.
- **`bend --verdict` cannot be used directly on this library at present.** Bend v2.0.36 stops before verification starts. The existing proofs are still verified by the official checker, with `python run.py proofs`.
- Only inference is supported at present. Model training and automatic differentiation are not implemented yet.
- Only the FP32 (F32) data type is supported.
- For convolutions with many weights and few output positions, and for a batch size above 1 or an output width below 6, more workers usually bring no clear performance gain. See the [design document](docs/design.md).
- Inference is still slower than PyTorch. For a detailed comparison, see the [performance document](docs/performance.md).

## [0.0.1] - 2026-10-05

The first public version of Stelliferous (then Bend-Conv), built on the official Bend v2.0.35.

### Added

- **FP32 tensor API**

  `lib/tensor_f32.bend` provides the basic tensor operations. Tensor shapes are checked at run time, and errors are carried by the returned result.

  The main features are

  - **Construction** — `zeros`, `ones`, `full`, `arange`, `from_list`, `from_storage`.
  - **Elementwise operations** — `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, with NumPy broadcasting rules and matching `_scalar` forms.
  - **Activation functions** — `relu`, `relu6`, `leaky_relu`, `sigmoid`, `silu`, `tanh`, `hard_sigmoid`, `hard_swish`, `gelu_tanh`, `activate`, and `apply` and `apply_indexed` for custom functions.
  - **Reductions** — `sum`, `mean`, `max`, `min`, and along a given axis `sum_axis`, `mean_axis`, `max_axis`, `min_axis`.
  - **Shape and layout operations** — `reshape`, `transpose`, `slice`, `set_slice`, `concat`, `pad`, `pad_edge`, `broadcast_to`.
  - **Convolution and window operations** — `conv2d`, `conv2d_with`, `max_pool2d`, `max_pool2d_with`, `upsample_nearest`.
  - **Reading values and checking state** — `to_list`, `item`, `to_list_or`, `item_or`, `check`, `describe`, `into_storage`.

- **FP32 convolution with deterministic results**

  The FP32 operations and the accumulation order are fixed, without FMA (fused multiply-add) or reassociation, so results are bit-for-bit identical for every worker count.

- **A work planner implemented in Bend**

  It decides how each convolution is divided into tasks from the tensor shapes and the worker count.

- **Row-band data layout**

  Row-band storage lets several workers share the work on a tensor without each copying the whole data.

  The layout is kept across convolution, elementwise operations, activation, channel concatenation (concat), upsampling and pooling, which avoids unnecessary copies.

- **Formal proofs**

  The theorems are verified by the unmodified official Bend checker. They include the unconditional theorem that native convolution matches the matrix specification for every input, and proofs of the numerical properties of the tensor operations.

  For details, see the [proofs document](docs/proofs.md).

- **YOLOv5n inference example**

  A Bend program generated from the official pretrained YOLOv5n network, compared numerically with PyTorch layer by layer.

- **Correctness tests**

  79 test programs on the public API compare their results bit for bit with NumPy.

  The convolution tests cover 1, 2, 4 and 8 threads and check native convolution for each worker count.

- **Bilingual documentation**

  Documentation in English and Traditional Chinese.

### Known limitations

- Only model inference is supported. Training and automatic differentiation are not implemented yet.
- Only the FP32 (F32) data type is supported.
- The library has no general model weight loading yet. The YOLO example uses its own file reading program.
- For convolutions with many weights and few output positions, and for a batch size above 1 or an output width below 6, more workers usually bring no clear performance gain. See the [design document](docs/design.md).
- Inference is still slower than PyTorch. For a detailed comparison, see the [performance document](docs/performance.md).
- Because Bend does not yet support default arguments, named arguments or function overloading, some APIs come in both a short and a full form.
