# Proofs

[English](proofs.md) | [繁體中文](proofs.zh-TW.md)

Every proof is a Bend term checked by the unmodified official Bend checker.

```sh
python run.py proofs
```

## The root theorem

[native_inference_law.bend](../convolution/proofs/native_inference_law.bend):

```text
law native_inference_matches_matrix:
  for +problem: ConvSpec.Problem
  {NativeConv.infer(problem) == MatrixSpec.infer(problem) : ConvSpec.ConvOutcome}
```

A problem holds a shape, an input, weights and a bias. The law puts no condition on it: it covers rejected input, the reference fallback and native execution, for every split depth (`inference_with_any_split_depth_matches_matrix`) and both strategies (`every_strategy_matches_matrix`).

**The numerical contract.** Each output starts at FP32 +0.0, adds the products in input-channel, kernel-row, kernel-column order (multiply, then add), and adds the bias last. Padding samples take part in that order. The theorem is about this sequence of FP32 operations, not about real numbers, and it does not allow fused multiply-add or reassociation.

**Frozen files.** [specification.bend](../convolution/specification.bend), [specification_law.bend](../convolution/specification_law.bend), [ordered_matrix.bend](../lib/ordered_matrix.bend), [matrix_specification.bend](../convolution/matrix_specification.bend) and [matrix_convolution_law.bend](../convolution/proofs/matrix_convolution_law.bend) must keep the hashes in [frozen.json](../data/contracts/frozen.json). Every proof run checks them.

## What is trusted

- The official checker and Base library.
- The compiler, the runtime and its scheduler, the C compiler and the hardware. The theorem is about the Bend source.
- Bend's file I/O under `load` and `save`. Library and convolution proofs contain no `@unsafe` definition.

Numerical checks cover what the proofs do not: generated code, model wiring, activation, I/O and detection decoding are compared with NumPy and PyTorch, and the mathematical functions with MPFR over every FP32 input ([benchmarks](../benchmarks/README.md)).

## Following the convolution proof

| Obligation | Source |
|---|---|
| Direct (scalar) execution | [scalar_inference_refinement.bend](../convolution/proofs/scalar_inference_refinement.bend) |
| Packed execution | [packed_inference_refinement.bend](../convolution/proofs/packed_inference_refinement.bend) |
| Array boundaries, cells partition and GEMM, cells gather, output untile | [array_cell_execution.bend](../convolution/proofs/array_cell_execution.bend), [microkernel_storage_execution.bend](../convolution/proofs/microkernel_storage_execution.bend) |
| Array boundary packing into eight-lane cells | [kernel_weights_refinement.bend](../lib/proofs/kernel_weights_refinement.bend), [window_cell_values.bend](../convolution/proofs/window_cell_values.bend) |
| The transpose and pack of a leaf fill its tiles with the problem values | [kernel_spatial_packing_pixels.bend](../lib/proofs/kernel_spatial_packing_pixels.bend), [window_leaf_readback.bend](../convolution/proofs/window_leaf_readback.bend), [array_leaf_cells.bend](../convolution/proofs/array_leaf_cells.bend) |
| GEMM storage as the writes of each job's view, with bounds, values and coverage | [kernel_gemm_writes.bend](../lib/proofs/kernel_gemm_writes.bend), [kernel_gemm_jobs.bend](../lib/proofs/kernel_gemm_jobs.bend), [kernel_gemm_readback.bend](../lib/proofs/kernel_gemm_readback.bend), [view_writes.bend](../lib/proofs/view_writes.bend) |
| The accumulator equals the matrix dot | [gemm_dot_matches_matrix.bend](../convolution/proofs/gemm_dot_matches_matrix.bend) |
| Every leaf output equals the matrix value | [microkernel_leaf_readback.bend](../convolution/proofs/microkernel_leaf_readback.bend) |
| Input row windows of every split | [window_partition.bend](../convolution/proofs/window_partition.bend), [window_leaf_readback.bend](../convolution/proofs/window_leaf_readback.bend) |
| Cell write bounds, coverage and lane readback | [cell_gather_leaf.bend](../convolution/proofs/cell_gather_leaf.bend), [gather_tree_readback.bend](../convolution/proofs/gather_tree_readback.bend), [write_program_observation.bend](../lib/proofs/write_program_observation.bend) |
| Arbitrary caller array capacities | [array_api_validation.bend](../convolution/proofs/array_api_validation.bend) |
| Shared storage contracts | [storage_refinement.bend](../lib/proofs/storage_refinement.bend) |

The pattern is the same throughout: describe what a loop stores as a list of writes, prove every write is in bounds, prove the value written, and prove that later writes do not disturb it.

## The tensor contract

A public tensor holds `TiledPublic.Contract` of a value reader ([tensor_tiled_public.bend](../lib/proofs/tensor_tiled_public.bend)): its cells are the canonical pieces of its shape, laid out along the lane axis, Usable, and every lane is the value the reader gives at its index. Lanes past the length of the lane axis hold +0. A value reader is a context and a function from an index to a value, and each operand has its own.

Each theorem below takes the contract of every operand and gives the contract of the result. Most also take these premises:

- no extent is zero (an empty tensor takes another route),
- the rank is at most the machine capacity, and the limits have the default capacity,
- the checks the operation makes itself, such as the view it builds, `count_of` of the result and `Views.fits`,
- for an operation that adds along lines, `2 * elements + 4 <= cap <= 2^31` for some `cap`.

## Convolution

| Function | What the result holds | Source |
|---|---|---|
| `conv2d`, `conv2d_custom` | for an input `[C, H, W]` or `[N, C, H, W]`, weights `[O, C, K, K]` and bias `[O]`: the activation of the ordered convolution value at every index. Premises: the problem is valid, the machine supports it and there is an image | [convolution_call.bend](../convolution/proofs/convolution_call.bend) (`conv2d_contract`), [convolution_public.bend](../convolution/proofs/convolution_public.bend) |
| `Gradient.input` | the gradient of the input of [input_gradient_specification.bend](../convolution/input_gradient_specification.bend), and the weights back. Premises: `Call.checks` and an image | [input_gradient_public.bend](../convolution/proofs/input_gradient_public.bend) (`tiled_input`) |
| `Gradient.weight`, `Gradient.bias` | the gradients of [parameter_gradient_specification.bend](../convolution/parameter_gradient_specification.bend), and x and the output gradient back. Premises: `Parameter.valid` and `Parameter.positive`, the checks the runtime makes | [parameter_gradient_public.bend](../convolution/proofs/parameter_gradient_public.bend) (`tiled_weight`, `tiled_bias`) |

The input gradient adds, for each input element, the products from +0.0 over the output channel increasing and the kernel row and column decreasing, keeping the taps congruent to its padded position modulo the stride, then adds +0.0. A weight or bias gradient adds the terms of consecutive output rows in groups from +0.0, then the groups and the images by a tree of adjacent pairs.

Each task of a convolution or a gradient reads a window of the input rows and writes its output pieces ([window_windowed.bend](../convolution/proofs/window_windowed.bend), [input_gradient_windowed.bend](../convolution/proofs/input_gradient_windowed.bend), [tensor_parameter_gradient_cells.bend](../lib/proofs/tensor_parameter_gradient_cells.bend)).

## Training

| Function | What the result holds | Source |
|---|---|---|
| `Layer.forward`, `Layer.backward`, `Layer.backward_parameters` | the activation of the convolution, the saved input and convolution value, and the input, weight and bias gradients of `dz = dy * activation'(z)`. Premises: those of `conv2d` or of the gradients, and an activation other than `Custom` | [layer_public.bend](../convolution/proofs/layer_public.bend) |
| `Train.forward`, `Train.evaluate`, `Train.backward`, `Train.sgd_step_with` of a convolution | what `conv2d`, `Gradient.*` and `Optimizer.sgd_with` give, with the layer back and its parameters in their slots. The step takes nonempty weights and bias | [training_convolution_public.bend](../convolution/proofs/training_convolution_public.bend) |
| the same of an activation, a batch normalization, a layer normalization and the average pool | what the operation they call gives, with the layer back | [tensor_training_layer_values.bend](../lib/proofs/tensor_training_layer_values.bend) |
| `forward_network`, `evaluate_network`, `backward_network`, `sgd_network_with` | see below | [training_network_public.bend](../convolution/proofs/training_network_public.bend) |

A network theorem takes a description of the network: a value for each parameter of each layer, and the facts each layer needs of the extents of its input. A block also takes that its branch and its shortcut give the same extents, nonempty and within the machine. Then:

- `forward` gives an output that holds the values of the layers applied in order, a block the sum of its branch and its shortcut, and a tape that holds what each layer saved.
- `evaluate` gives the same in inference mode, with the running statistics, and the network back as described.
- `backward` of a tape that holds such a record gives the gradient of the input, and the network with the gradients of every parameter.
- `sgd` moves every parameter that has a gradient by `Optimizer.sgd_with` and clears the gradients.

## Tensor operations

| Operation | What the result holds | Source |
|---|---|---|
| `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, with broadcasting | `operation(left, right)` at every index, in that argument order | [tensor_binary_values.bend](../lib/proofs/tensor_binary_values.bend), [tensor_tiled_map.bend](../lib/proofs/tensor_tiled_map.bend), [tensor_tiled_broadcast.bend](../lib/proofs/tensor_tiled_broadcast.bend) |
| `add_scalar`, `mul_scalar`, ... | `operation(x, scalar)` | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `less`, `less_equal`, `greater`, `greater_equal`, `equal`, `not_equal`, their `_scalar` forms, `clamp` | 1.0 or 0.0 by the comparison at every index, and for `clamp` the value clamped to the bounds | [tensor_comparison_values.bend](../lib/proofs/tensor_comparison_values.bend) |
| `sigmoid`, `silu`, `tanh`, `gelu`, `gelu_tanh`, `exp`, `log`, `sqrt`, `apply` | the scalar function of [math_fp32.bend](../lib/math_fp32.bend) or [math_activation.bend](../lib/math_activation.bend) at every index | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `activate`, `activate_derivative`, `activate_forward`, `activate_backward` | the activation or its derivative at every index, and for backward the gradient times the derivative at the input. The activation is not `Custom` | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `Optimizer.sgd`, `Optimizer.sgd_with` | `parameter - applied * rate` at every index, with `direction = gradient + parameter * decay` and `applied = velocity * momentum + direction` (`direction` on the first step). The velocity left holds `applied`, and remains exactly when the momentum is not zero | [tensor_optimizer_values.bend](../lib/proofs/tensor_optimizer_values.bend) |
| `BatchNorm.forward`, `BatchNorm.backward`, `BatchNorm.evaluate` | for x in `[N, C, H, W]` and per-channel tensors in `[C]`: `xhat * gamma + beta` with `xhat = (x - mean) * deviation`, `deviation = 1 / sqrt(s / M + epsilon)`, the sums along `[0, 2, 3]` with rows of H × W planes, an image and a channel each, merged over the images (a sum can differ from `F.sum` along `[0, 2, 3]` in the last bit), and the running statistics moved towards the mean and `s / (M - 1)`. Backward gives `dbeta`, `dgamma` and `dx` in the runtime's FP32 order | [tensor_batch_norm_public.bend](../lib/proofs/tensor_batch_norm_public.bend) |
| `LayerNorm.forward`, `LayerNorm.backward`, `LayerNorm.evaluate` | for x in `[..., D]` and gamma, beta in `[D]`: `xhat * gamma + beta` with the means along the last axis. Backward gives `dx = (dz - mean(dz) - xhat * mean(dz * xhat)) * deviation` of `dz = dy * gamma`, and the sums of `dy * xhat` and `dy` over the other axes | [tensor_layer_norm_values.bend](../lib/proofs/tensor_layer_norm_values.bend) |
| `Loss.mse`, `Loss.cross_entropy` | for a prediction and a target of one shape: the loss, its gradient and the target. `cross_entropy` takes logits `[N, K]` | [tensor_loss_values.bend](../lib/proofs/tensor_loss_values.bend), [tensor_cross_entropy_values.bend](../lib/proofs/tensor_cross_entropy_values.bend) |
| `Train.one_hot` | at row `i` and column `j` of `[N, K]`, the `equal` of label `i` and `j` | [tensor_one_hot_values.bend](../lib/proofs/tensor_one_hot_values.bend) |
| `copy`, `reshape` | the values of the input, reshape in their row-major order | [tensor_copy_contract.bend](../lib/proofs/tensor_copy_contract.bend), [tensor_tiled_reshape.bend](../lib/proofs/tensor_tiled_reshape.bend) |
| `softmax` | FP32 `exp(x[i] - m) / s`, with `m` the fold of the axis in increasing index and `s` the sum of [math_sum.bend](../lib/math_sum.bend) along the axis | [tensor_softmax_values.bend](../lib/proofs/tensor_softmax_values.bend) |
| `matmul`, `linear` | at row `i` and column `j` of each matrix, the products added in increasing `k` from +0.0, then +0.0 or the bias of `j` | [matmul_call.bend](../convolution/proofs/matmul_call.bend), [linear_call.bend](../convolution/proofs/linear_call.bend), [product_values.bend](../convolution/proofs/product_values.bend) |
| `apply_indexed`, `random_uniform`, `random_below`, `arange` | the function of the row-major index and the value there, the generator of [math_random.bend](../lib/math_random.bend) at the seed, the stream and the index, and the index plus the start | [tensor_tiled_indexed_values.bend](../lib/proofs/tensor_tiled_indexed_values.bend), [tensor_random_values.bend](../lib/proofs/tensor_random_values.bend) |
| `select` | the mask's magnitude bits choose yes or no and keep all 32 bits of the chosen word | [tensor_select_public.bend](../lib/proofs/tensor_select_public.bend), [math_select_values.bend](../lib/proofs/math_select_values.bend) |
| `transpose`, `slice`, `flip`, `broadcast_to` | the input at the address the view gives to that index | [tensor_tiled_view.bend](../lib/proofs/tensor_tiled_view.bend) |
| `upsample_nearest` | the input at the nearest index of the last two axes | [tensor_tiled_reshape.bend](../lib/proofs/tensor_tiled_reshape.bend) |
| `set_slice` | the part from its start along the axis, the rest unchanged | [tensor_tiled_set_slice.bend](../lib/proofs/tensor_tiled_set_slice.bend) |
| `pad`, `pad_edge` | the input after the padding before it, the constant or the nearest element elsewhere | [tensor_tiled_pad.bend](../lib/proofs/tensor_tiled_pad.bend) |
| `concat` | each part at its place along the axis | [tensor_tiled_concat.bend](../lib/proofs/tensor_tiled_concat.bend) |
| `sum_axes`, `mean_axes`, `max_axes`, `min_axes`, `argmax`, `sum`, `mean`, `max`, `min` | at each index of the kept axes, the fold of the listed axes in row-major order, the sums by [math_sum.bend](../lib/math_sum.bend). The axes are distinct and inside the shape | [tensor_tiled_reduce.bend](../lib/proofs/tensor_tiled_reduce.bend), [tensor_reduce_values.bend](../lib/proofs/tensor_reduce_values.bend) |
| `max_pool2d`, `max_pool2d_with` | the fold of each window, rows then columns, in increasing index, from -inf. The window is valid and the input has two axes or more | [tensor_tiled_pool.bend](../lib/proofs/tensor_tiled_pool.bend) |
| `depthwise_conv2d` | at each image and channel `c`, the bias of `c` plus the taps of channel `c` times its weights, rows then columns. The cells of the result are usable | [tensor_tiled_depthwise.bend](../lib/proofs/tensor_tiled_depthwise.bend) |

The `softmax` value is that FP32 sequence, with the library `exp`. Nothing is claimed about the real-number normalization.

Underneath:

| Subject | What is shown | Source |
|---|---|---|
| Pieces and tasks | The pieces of a shape and their ranges, the tasks a decision groups them into, and that the tasks write every piece once | [tensor_grid_pieces.bend](../lib/proofs/tensor_grid_pieces.bend), [tensor_grid_ranges.bend](../lib/proofs/tensor_grid_ranges.bend), [tensor_task_runs.bend](../lib/proofs/tensor_task_runs.bend), [tensor_pieces_grouped_with.bend](../lib/proofs/tensor_pieces_grouped_with.bend), [tensor_placed.bend](../lib/proofs/tensor_placed.bend) |
| Maps | A map or a combine of pieces holds the function of the values | [tensor_pieces_map.bend](../lib/proofs/tensor_pieces_map.bend), [tensor_pieces_combine.bend](../lib/proofs/tensor_pieces_combine.bend) |
| Moves | Every component of a move takes its source pieces once and writes its target pieces, and a run of any step reads the source it names | [tensor_moved.bend](../lib/proofs/tensor_moved.bend), [tensor_components.bend](../lib/proofs/tensor_components.bend), [tensor_move_readback.bend](../lib/proofs/tensor_move_readback.bend) |
| Views, placing and relane | The gather law of a view, a placing and a move of the lane axis | [tensor_view_gather.bend](../lib/proofs/tensor_view_gather.bend), [tensor_placing_law.bend](../lib/proofs/tensor_placing_law.bend), [tensor_relane_gather.bend](../lib/proofs/tensor_relane_gather.bend), [tensor_tiled_relane.bend](../lib/proofs/tensor_tiled_relane.bend) |
| Split of the lane axis | The gather law of a reshape that splits the lane axis in two, and the contract of its result | [tensor_split_law.bend](../lib/proofs/tensor_split_law.bend), [tensor_tiled_split.bend](../lib/proofs/tensor_tiled_split.bend) |
| Windows | Each task of a stencil holds the input rows it reads | [tensor_halo_windows.bend](../lib/proofs/tensor_halo_windows.bend), [tensor_window_windowed.bend](../lib/proofs/tensor_window_windowed.bend) |
| Lines | A reduction folds each line in one order, whether the line lies within a piece or across pieces | [tensor_partials_values.bend](../lib/proofs/tensor_partials_values.bend) |
| Bounds | From the guards the code checks: every read and write is in range | [tensor_operation_bounds.bend](../lib/proofs/tensor_operation_bounds.bend) |

## Making a tensor

[tensor_entry_values.bend](../lib/proofs/tensor_entry_values.bend) makes a tensor of the contract of the values it is given.

| Function | What the result holds |
|---|---|
| `from_list` | the values of any reader that agrees with the list at every index of the shape |
| `from_storage` | for storage that holds a sized model, its values at their flat index, in the shape |
| `full`, `zeros`, `ones` | the value at every index ([tensor_full_values.bend](../lib/proofs/tensor_full_values.bend)) |

`read_storage` of such storage gives the storage back with its value at the index, or 0.0 past its end.

## Reading a tensor

[tensor_exit_values.bend](../lib/proofs/tensor_exit_values.bend) reads a tensor of the contract as the reader's values in flat index order.

| Function | What it gives |
|---|---|
| `into_storage` | Storage whose element `i` is the reader's value at `i` |
| `to_list` | The list of the reader's values at `0` to `n - 1` |
| `item` | For a tensor of one element, the reader's value at `0` |
| `save_raw`, `save_npy` | They write that storage, after the `.npy` header for `save_npy`, and return a tensor of the same contract |

## A store of named tensors

[tensor_model_values.bend](../lib/proofs/tensor_model_values.bend) describes a store as named tensors in order, each holding its values.

| Function | What it gives |
|---|---|
| `take` | the first tensor of the name, holding its values, and the store of the other entries |
| `keep` | the tensor back and the store with a copy in front, both holding its values |

## What is not proved

- The cost model and the parallel decisions. They choose how the pieces are grouped, never a result.
- Real-number accuracy. [math_accuracy.py](../benchmarks/checks/math_accuracy.py) measures it.
- The YOLO example's graph, I/O and detection decoding.
- Compiler and runtime behaviour.
- The text of `describe`, and `shape`, `check` and `workers`, which read metadata.

## Inspecting

`python run.py proofs` also writes the list of declarations with their premises under `out/results/proof-audit/`, and `python run.py dependencies` the references of every declaration. They are for navigation. The verdict is the checker's.
