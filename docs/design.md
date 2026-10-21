# Design

[English](design.md) | [繁體中文](design.zh-TW.md)

## Layers

| Layer | What it owns | Main files |
|---|---|---|
| Public F32 API | The functions a program calls | [tensor_f32.bend](../lib/tensor_f32.bend), [math_activation.bend](../lib/math_activation.bend), [tensor_training.bend](../lib/tensor_training.bend) |
| Training | Layers, losses, SGD and the gradients of convolution, batch normalization and activations | [tensor_layer.bend](../lib/tensor_layer.bend), [tensor_batch_norm.bend](../lib/tensor_batch_norm.bend), [tensor_loss.bend](../lib/tensor_loss.bend), [tensor_optimizer.bend](../lib/tensor_optimizer.bend), `tensor_*_gradient.bend` |
| Generic tensor | Shape, status, worker count, band storage, broadcasting, layout, windows, FP32 files, named weights, PyTorch checkpoints | [tensor.bend](../lib/tensor.bend), [tensor_layout.bend](../lib/tensor_layout.bend), [tensor_window.bend](../lib/tensor_window.bend), [tensor_convolution.bend](../lib/tensor_convolution.bend), [tensor_file.bend](../lib/tensor_file.bend), [tensor_model.bend](../lib/tensor_model.bend), [tensor_checkpoint.bend](../lib/tensor_checkpoint.bend), `checkpoint_*.bend`, `tensor_band*.bend` |
| Checked operations | `copy`, `combine`, `apply`, `reduce`, `scatter`, `sweep` over views, with guards | [tensor_operations.bend](../lib/tensor_operations.bend), [tensor_view.bend](../lib/tensor_view.bend), [tensor_cursor.bend](../lib/tensor_cursor.bend), [tensor_access.bend](../lib/tensor_access.bend) |
| Convolution kernel | Weight reorder, packing, 4x8 GEMM, gather, planning, input and weight gradients | `kernel_*.bend` |
| Storage and traversal | Owned buffers with certificates, and the loops everything else uses | `storage_*.bend`, `traversal_*.bend` |
| Convolution package | The frozen specification, the reference, the proved Array engine | [convolution/](../convolution/README.md) |

`lib/` does not depend on `convolution/`, the examples or the benchmarks.

## Tensors

A tensor owns its storage, its extents, a fill value, a status and a worker count.

- **Operations consume their arguments.** Binary arithmetic reuses the left operand's storage. `F.copy` keeps a tensor.
- **Failures travel with the value.** After a failure, later operations skip their work and pass the first problem on. The program sees it where it reads the result.
- **Shapes are checked at run time.** Extents are values, not types.
- **Views are internal.** `transpose`, `slice`, `broadcast_to` and `flip` copy band by band into new storage. `reshape` only relabels.

### Band storage

A tensor's storage is a set of bands: the NCHW tensor cut into ranges of rows. Each band is its own buffer, in `[channels][rows][width]` order. Row-major storage is one band of one row that holds every element.

A worker can take one band without cloning the whole tensor. Convolution produces bands. Elementwise arithmetic, activation, concat, nearest upsampling, max pooling, the views, reductions and batch normalization work band by band. Readback, saving, convolution weights and a broadcast right operand read a tensor once as its one whole band. The layout never changes a value.

### Strided access

A cursor walks a view without division. A strided copy runs the innermost axis as a plain loop and steps the cursor once per run. `set_slice`, the concat of each band and constant `pad` are one checked scatter: a contiguous source written through a strided target view.

### Functions in hot loops

A function used per element is passed as a template argument (`~F32.add`), never stored in a record. The compiler then specializes the loop. A function held in a record is called through a closure for every element, which measured about six times slower.

## Convolution

Each output starts from +0.0, adds the products in input-channel, kernel-row, kernel-column order, then adds the bias. Padding samples take part as zeros. There is no fused multiply-add and no reassociation, so the result has the same bits for every worker count and every plan.

1. **Reorder weights** once into tiles of eight output channels.
2. **Pack** the input positions of a leaf into tiles of eight positions.
3. **GEMM** over four output channels by eight positions per step, with partial sums in local accumulators.
4. **Store or gather** the leaf outputs into the result.

With several workers the output positions are split into leaves. A split clones the reordered weights and the bias, and gives the first half only the input rows it reads.

### Planning

The planner chooses the number of leaves inside the call, from the shapes and the tensor's worker count. Its estimate has five parameters, in [default_kernel_cost.bend](../lib/default_kernel_cost.bend):

| Parameter | Value | Meaning |
|---|---:|---|
| `mac_ns` | 0.050 | one multiply-add: four cycles per 32 at a fixed 0.4 ns reference cycle |
| `operation_ns` | 0.4 | one counted scalar operation |
| `copy_byte_ns` | 0.100 | one copied or initialized byte |
| `grain_ns` | 250000 | least predicted work of a leaf worth a task |
| `wake_round_ns` | 180000 | one round of waking workers |

- At most five tasks per worker, and every leaf keeps at least one block of eight positions.
- Candidates double the leaf count. A split is kept only when the whole estimate (work, copies, wake rounds) is strictly smaller.
- One worker never splits.

The numbers are a fixed reference scale, not a calibration to one machine. A wrong estimate can only cost time: the result is proved equal for every split.

### Known limits

Measured on shapes from other networks, one machine, one run each:

- **Many weights, few output positions** (for example 512 to 512 channels, 3x3, on 7x7): little or no gain from more workers. Every call reorders the weights and every split clones them.
- **Batch greater than one:** the images are split over the workers one at a time, each part with its own copy of the weights. Small images with many channels (64 channels on 8x8) run slower on four workers than on one.
- **Outputs narrower than six positions** use a slower per-entry packing path.

## What Bend does not provide

- No default arguments, named arguments or overloading. Short and full forms have two names (`max_pool2d` / `max_pool2d_with`).
- A `match` applies to a parameter, not to a call result. The library provides `to_list_or` and `item_or` for reading a `Result`.
- No fused multiply-add and no SIMD type. The kernel is scalar Bend that the C compiler vectorizes.
- `Array.new` initializes every element.
- Shared read-only storage needs `@unsafe`, which this library does not use. A split clones what both halves read.
- A duplicated `List<Tensor>` makes the compiler mark every constructor as shared, which slows unrelated code. `concat` therefore takes `Parts` (`F.cons`, `F.single`).

The [toolchain rules](toolchain_rules.md) list the checker, compiler and runtime rules in detail.
