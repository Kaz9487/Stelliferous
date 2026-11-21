# Design

[English](design.md) | [繁體中文](design.zh-TW.md)

## Layers

| Layer | What it owns | Main files |
|---|---|---|
| Public F32 API | The functions a program calls | [tensor_f32.bend](../lib/tensor_f32.bend), [math_activation.bend](../lib/math_activation.bend), [tensor_training.bend](../lib/tensor_training.bend) |
| Training | Layers, losses, SGD and the gradients of convolution, batch normalization and activations | [tensor_layer.bend](../lib/tensor_layer.bend), [tensor_batch_norm.bend](../lib/tensor_batch_norm.bend), [tensor_loss.bend](../lib/tensor_loss.bend), [tensor_optimizer.bend](../lib/tensor_optimizer.bend), `tensor_*_gradient.bend` |
| Tiled cells | The form of every `F` tensor and the operations on its cells | [tensor_tiled.bend](../lib/tensor_tiled.bend), `tensor_tiled_*.bend`, [tensor_select.bend](../lib/tensor_select.bend), [tensor_matrix_products.bend](../lib/tensor_matrix_products.bend), [tensor_batch_norm_cells.bend](../lib/tensor_batch_norm_cells.bend) |
| Generic tensor | Extents, status, worker count and canonical pieces of any element type, the tasks of an operation, moves, views, windows, files, named weights, PyTorch checkpoints | [tensor.bend](../lib/tensor.bend), [tensor_pieces.bend](../lib/tensor_pieces.bend), [tensor_grid.bend](../lib/tensor_grid.bend), [tensor_tasks.bend](../lib/tensor_tasks.bend), [tensor_move.bend](../lib/tensor_move.bend), [tensor_halo.bend](../lib/tensor_halo.bend), [tensor_layout.bend](../lib/tensor_layout.bend), [tensor_window.bend](../lib/tensor_window.bend), [tensor_convolution.bend](../lib/tensor_convolution.bend), [tensor_file.bend](../lib/tensor_file.bend), [tensor_model.bend](../lib/tensor_model.bend), [tensor_checkpoint.bend](../lib/tensor_checkpoint.bend), `checkpoint_*.bend` |
| Checked operations | Copies, combinations, sweeps, pools and reductions over views, with guards | [tensor_operations.bend](../lib/tensor_operations.bend), [tensor_view.bend](../lib/tensor_view.bend), [tensor_cursor.bend](../lib/tensor_cursor.bend), [tensor_access.bend](../lib/tensor_access.bend) |
| Kernels | Weight tiles, packing, the GEMM block, gather, planning, depthwise taps, input and weight gradients | `kernel_*.bend` |
| Storage and traversal | Owned buffers with certificates, and the loops everything else uses | `storage_*.bend`, `traversal_*.bend` |
| Convolution package | The frozen specification, the reference, the proved Array engine | [convolution/](../convolution/README.md) |

`lib/` does not depend on `convolution/`, the examples or the benchmarks.

## Tensors

A tensor owns its cells, its extents, a status and a worker count.

- **Operations consume their arguments.** `F.copy` keeps a tensor.
- **Failures travel with the value.** After a failure, later operations skip their work and pass the first problem on. The program sees it where it reads the result.
- **Shapes are checked at run time.** Extents are values, not types.

### Cells

An `F` tensor blocks one axis, its lane axis, by eight: the eight values of a group of that axis are the lanes of one `Tile8`, and lanes past the true extent hold +0. The other axes stay row-major. The lane axis of a new tensor is the channel axis of `[N,]C,H,W`, otherwise the last axis. A tensor of rank 0 is one lane of one cell.

The cells lie in one canonical order: the other axes row by row, the groups of the lane axis innermost (outermost for weights). They are stored in pieces, ranges of that order of about 4096 cells whose boundaries only the shape decides, so two tensors of one shape have the same pieces.

Values enter and leave as F32 only at the boundary: `from_list`, `from_storage` and the file loads tile them, and `to_list`, `item`, `into_storage` and the saves read them back. Every operation in between runs on the cells.

- An operation runs on the lanes where they are. One that needs them on another axis first moves them: each result cell reads its eight lanes from the eight cells that hold them.
- A view that keeps the lane axis gathers whole cells. A view that keeps no lane axis moves values.
- `reshape` relabels the cells when the lane axis and its extent stay, and otherwise moves values.

### Strided access

A cursor walks a view without division. A strided copy runs the innermost axis as a plain loop and steps the cursor once per run. `set_slice` and `pad` write through a view of the result, on the cells when the view keeps the lanes.

### Functions in hot loops

A function used per element is passed as a template argument (`~F32.add`), never stored in a record. The compiler then specializes the loop. A function held in a record is called through a closure for every element, which measured about six times slower.

## Convolution

Each output starts from +0.0, adds the products in input-channel, kernel-row, kernel-column order, then adds the bias. Padding samples take part as zeros. There is no fused multiply-add and no reassociation, so the result has the same bits for every worker count and every plan.

1. **Weight tiles:** each coordinate of the reduction holds the weights of eight output channels in one tile.
2. **Pack** the input cells a task reads, whose lanes are the channels, into tiles of eight output positions in reduction order.
3. **GEMM** blocks of up to eight output channels by eight positions, each folding the reduction from +0.
4. **Store** the task's output cells, eight channels a tile, then apply the activation in one pass while they are in cache.

A task holds adjacent output pieces. It reads the input pieces of its rows and receives the rows above and below them from a walk over the tasks. Every task copies the weight tiles and the bias.

### Tasks

Every operation groups its pieces into tasks with one decision ([tensor_tasks.bend](../lib/tensor_tasks.bend)). The serial plan is one task. The parallel plan joins adjacent pieces up to one worker's share, or up to a smaller power of two of it when such tasks end the last wave sooner, and runs when its estimate is smaller than the serial work: two wake rounds, the waves of its largest task, the dispatch of every task and what each task after the first allocates or copies. A cell counts rows, vector instructions of eight lanes with its tile loads and stores among them, and steps, scalar instructions. A target cell of a moving operation also pays one move through a view, and a block copy pays its bytes. The prices are in [default_kernel_cost.bend](../lib/default_kernel_cost.bend):

| Parameter | Value | Meaning |
|---|---:|---|
| `row_ns` | 0.113 | one row: a multiply-add of eight lanes in the GEMM block |
| `step_ns` | 0.122 | one scalar step of a loop |
| `byte_ns` | 0.036 | one byte a block copy writes |
| `move_ns` | 5.9 | one cell moved through a view |
| `wake_round_ns` | 20000 | one round of waking workers |
| `dispatch_ns` | 120 | dispatching one more task |
| `grain_ns` | 250000 | least predicted work of a leaf of the Array engine |

One worker never runs in parallel. [planner_prices.py](../scripts/toolchain_rules/planner_prices.py) measures the first four prices on one thread of the reference machine, and [wake_rounds.py](../scripts/toolchain_rules/wake_rounds.py) checks the wake round. A wrong estimate can only cost time: the result is proved equal for every grouping.

### Costs that remain

- Every task of a convolution copies the weight tiles. That cost weighs most on a layer with many weights and few output positions.
- A tile of an output narrower than eight positions spans several output rows, so the packer reads, shifts and merges it once for every row it spans.

## What Bend does not provide

- No default arguments, named arguments or overloading. Short and full forms have two names (`max_pool2d` / `max_pool2d_with`).
- A `match` applies to a parameter, not to a call result. The library provides `to_list_or` and `item_or` for reading a `Result`.
- No fused multiply-add and no SIMD type. A tile is a record of eight F32 values that the C compiler vectorizes.
- `Array.new` initializes every element.
- Shared read-only storage needs `@unsafe`, which this library does not use. A split copies what both halves read.
- A duplicated `List<Tensor>` makes the compiler mark every constructor as shared, which slows unrelated code. `concat` therefore takes `Parts` (`F.cons`, `F.single`).

The [toolchain rules](toolchain_rules.md) list the checker, compiler and runtime rules in detail.
