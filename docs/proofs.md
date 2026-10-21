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
| The actual pipeline: reorder, partition, pack and GEMM, gather | [microkernel_inference_refinement.bend](../convolution/proofs/microkernel_inference_refinement.bend), [microkernel_storage_execution.bend](../convolution/proofs/microkernel_storage_execution.bend) |
| Weight reorder into eight-row tiles | [weight_reorder_readback.bend](../convolution/proofs/weight_reorder_readback.bend) |
| Packing, and the input it leaves unchanged | [packing_storage_refinement.bend](../convolution/proofs/packing_storage_refinement.bend), [packing_contents.bend](../convolution/proofs/packing_contents.bend) |
| GEMM storage as the writes of each job's view, with bounds, values and coverage | [kernel_gemm_writes.bend](../lib/proofs/kernel_gemm_writes.bend), [kernel_gemm_readback.bend](../lib/proofs/kernel_gemm_readback.bend), [view_writes.bend](../lib/proofs/view_writes.bend) |
| The accumulator equals the matrix dot | [gemm_dot_matches_matrix.bend](../convolution/proofs/gemm_dot_matches_matrix.bend) |
| Every leaf output equals the matrix value | [microkernel_leaf_readback.bend](../convolution/proofs/microkernel_leaf_readback.bend) |
| Input row windows of every split | [window_partition.bend](../convolution/proofs/window_partition.bend), [window_leaf_readback.bend](../convolution/proofs/window_leaf_readback.bend) |
| Gather writes and shared clipping bounds | [gather_write_program.bend](../convolution/proofs/gather_write_program.bend), [kernel_gather_width.bend](../lib/proofs/kernel_gather_width.bend) |
| Arbitrary caller array capacities | [array_api_validation.bend](../convolution/proofs/array_api_validation.bend) |
| Shared storage contracts | [storage_refinement.bend](../lib/proofs/storage_refinement.bend) |

The pattern is the same throughout: describe what a loop stores as a list of writes, prove every write is in bounds, prove the value written, and prove that later writes do not disturb it.

## The shared stencil pipeline

The common pipeline accepts rectangular taps and independent row and column origins. Its square wrappers keep the forward convolution interfaces. The following modules describe the shared packing and GEMM operations:

| Obligation | Source |
|---|---|
| Stencil samples and native packed-tile readback | [kernel_spatial_packing_samples.bend](../lib/proofs/kernel_spatial_packing_samples.bend), [kernel_packing_window_coordinates.bend](../lib/proofs/kernel_packing_window_coordinates.bend), [kernel_packing_window_readback.bend](../lib/proofs/kernel_packing_window_readback.bend) |
| Ordered stencil dot and bounded block readback | [kernel_pipeline_window_values.bend](../lib/proofs/kernel_pipeline_window_values.bend), [kernel_pipeline_window_readback.bend](../lib/proofs/kernel_pipeline_window_readback.bend) |

## The public convolution

`F.conv2d` runs the same kernel through buffers that carry storage certificates ([kernel_pipeline.bend](../lib/kernel_pipeline.bend)). The root theorem is stated for the Array engine. The public path has its own theorems:

- **Band path:** every call runs it. The input arrives in the layout the planner chooses: again from its own plan when it holds the input geometry, otherwise the first split. For the plan the code selects, the output bands cover every row and stay inside their allocations ([band_pipeline_well_formed.bend](../convolution/proofs/band_pipeline_well_formed.bend)), and every stored value is the ordered matrix value of its output point ([convolution_leaf_contract.bend](../convolution/proofs/convolution_leaf_contract.bend), [band_pipeline_execution.bend](../convolution/proofs/band_pipeline_execution.bend)).
- **The call:** for a Usable input of any layout, and weights and bias of any layout, each holding the values of its own value reader, the tensor `F.conv2d` returns is Ready in the output shape of the call, well formed and holds at every element the ordered convolution value of those values. A failed input or parameter stays failed, and an invalid geometry, parameters that do not fit it or a shape the machine guards reject give a Failed result ([convolution_public.bend](../convolution/proofs/convolution_public.bend)).

These are separate from the unconditional root. Their premises are a Usable input in a well-formed layout with finite values, matching shapes, and the machine guards the code checks.

## The input gradient

[input_gradient_specification.bend](../convolution/input_gradient_specification.bend) defines the gradient of the input of a convolution of any stride: each input element adds its products from +0.0 over the output channel increasing and the kernel row and column decreasing, keeping the taps congruent to its padded position modulo the stride, then adds +0.0. A congruent tap outside the output gradient counts as zero.

`input` in [input_gradient_public.bend](../convolution/proofs/input_gradient_public.bend) proves it of the tensor that `Gradient.input` returns, for an output gradient and weights of any layout, each holding the values of its own value reader: a Ready result is Normal and every element is the specification value of the readers' values at its image, and the weights come back as a tensor of their shape that holds their values. It covers batches with or without the batch axis, including a batch of no image.

| File | Statement |
|------|-----------|
| [tensor_full_values.bend](../lib/proofs/tensor_full_values.bend) | A Ready `full` tensor holds its value at every element. |
| [tensor_row_major_coordinates.bend](../lib/proofs/tensor_row_major_coordinates.bend) | The address a copy or a flip reads at a row-major index, from the digits of that index. |
| [input_gradient_call.bend](../convolution/proofs/input_gradient_call.bend) | When `Gradient.input` accepts its arguments, every shape and machine bound it checks holds. |
| [input_gradient_tiled_run.bend](../convolution/proofs/input_gradient_tiled_run.bend) | The kernel run gives the weights back, and its bands are the bands of the batch. |
| [input_gradient_tiled_images.bend](../convolution/proofs/input_gradient_tiled_images.bend) | The bands of the batch are well formed and every cell holds the specification value of the kernel's buffers. |
| [input_gradient_tiled_leaves.bend](../convolution/proofs/input_gradient_tiled_leaves.bend) | Every leaf of the kernel holds the specification value at every cell of its band. |

The leaf proof rests on these obligations.

| Obligation | Source |
|---|---|
| Actual tile, offset-table and weight packing readback | [input_gradient_tiled_chunk_reads.bend](../lib/proofs/input_gradient_tiled_chunk_reads.bend), [input_gradient_tiled_weight_reads.bend](../lib/proofs/input_gradient_tiled_weight_reads.bend), [input_gradient_tiled_pack_values.bend](../lib/proofs/input_gradient_tiled_pack_values.bend) |
| Source rows and actual MAC operands | [input_gradient_tiled_source_rows.bend](../lib/proofs/input_gradient_tiled_source_rows.bend), [input_gradient_tiled_product_reads.bend](../lib/proofs/input_gradient_tiled_product_reads.bend), [input_gradient_tiled_products.bend](../convolution/proofs/input_gradient_tiled_products.bend) |
| Ordered MAC value of every write, and the requested point is written | [input_gradient_tiled_dot.bend](../convolution/proofs/input_gradient_tiled_dot.bend), [input_gradient_tiled_written.bend](../convolution/proofs/input_gradient_tiled_written.bend), [input_gradient_tiled_coverage.bend](../lib/proofs/input_gradient_tiled_coverage.bend) |
| Chunk geometry, capacity and source owner preservation | [input_gradient_tiled_chunk_geometry.bend](../lib/proofs/input_gradient_tiled_chunk_geometry.bend), [input_gradient_tiled_capacity.bend](../lib/proofs/input_gradient_tiled_capacity.bend), [input_gradient_tiled_leaf_execution.bend](../lib/proofs/input_gradient_tiled_leaf_execution.bend) |

## Weight and bias gradients

[parameter_gradient_specification.bend](../convolution/parameter_gradient_specification.bend) defines the gradient of each weight and each bias: per image, the terms of consecutive output rows form groups, each group is added in order from +0.0, and the groups, then the images, are added by a tree of adjacent pairs.

[parameter_gradient_ready.bend](../convolution/proofs/parameter_gradient_ready.bend) proves that a Ready result of `Gradient.weight` or `Gradient.bias` holds that value at every index for the values of the input and output gradient value readers, for operands of any layout and any group length. The operands come back as tensors of their shapes that hold their values.

| File | Statement |
|------|-----------|
| [grouped_reduction_refinement.bend](../lib/proofs/grouped_reduction_refinement.bend) | The grouped traversal is the ordered fold of each group and the tree of the groups. |
| [list_storage_readback.bend](../lib/proofs/list_storage_readback.bend), [tensor_list_values.bend](../lib/proofs/tensor_list_values.bend) | A Ready tensor made from a list holds the items of the list, and the values of any value reader that agrees with the list inside its shape. |
| [parameter_gradient_bridge.bend](../convolution/proofs/parameter_gradient_bridge.bend) | Packed lanes and ordered group/image reductions equal the specification. |

The weight kernel packs gradient channels and input taps, multiplies them with the shared GEMM kernel without bias, and adds groups and images with the library's adjacent-pair tree. The bias gradient uses the scalar kernel.

| File | Statement |
|---|---|
| [kernel_weight_gradient_packing.bend](../lib/proofs/kernel_weight_gradient_packing.bend) | Native masked readers and both packing traversals refine the canonical storage model. |
| [kernel_weight_gradient_group.bend](../lib/proofs/kernel_weight_gradient_group.bend) | Bounded blocks, partial tiles and ordered continuation execute the shared GEMM model with the no-bias specialization. |
| [kernel_weight_gradient_refinement.bend](../lib/proofs/kernel_weight_gradient_refinement.bend) | Complete group and image pair trees and transposed readback refine the native weight kernel. |

## The training layer

The layer theorems compose the public convolution, activation, product and gradient theorems. Every operand may be in any layout and holds the values of its own value reader. Forward has the premises of the public convolution.

| File | Statement |
|---|---|
| [layer_public.bend](../convolution/proofs/layer_public.bend) | With any activation but `Custom`, `Layer.forward` returns an output that is Ready in the output shape of the call and holds the activation of the ordered convolution value at every index. The saved pair holds the input and that convolution value, Ready in the same shape, and the weights and bias come back. Ready gradients of `Layer.backward` and `Layer.backward_parameters` hold the input, weight and bias gradient specifications of x, dz = dy * activation'(z) and the weights, and the weights come back. |

`Train.forward`, `Train.backward` and `Train.evaluate` pass a convolution to `Layer.forward` and `Layer.backward` with the identity activation and to `conv2d`, a batch normalization to `BatchNorm`, and an activation to `activate_forward`, `activate_backward` and `activate`. `Train.sgd_step` applies `Optimizer.sgd`.

## Tensor operations

Each theorem has one premise about the call: it returned a Usable result. Operands may be in any layout. Every operand and the result hold the contract of [tensor_global_shape_count.bend](../lib/proofs/tensor_global_shape_count.bend): the tensor is Normal, and every element of its bands is the value a value reader gives at its index. A value reader is a context and a function from an index to a value. Each operand has its own.

| Operation | What a Usable result holds | Source |
|---|---|---|
| `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, with broadcasting | `operation(left, right)` at every index, in that argument order | [tensor_combine_public.bend](../lib/proofs/tensor_combine_public.bend) for inputs of one shape, [tensor_combine_broadcast.bend](../lib/proofs/tensor_combine_broadcast.bend) for broadcasting |
| `add_scalar`, `mul_scalar`, ... | `operation(x, scalar)` | [tensor_band_normal_unary.bend](../lib/proofs/tensor_band_normal_unary.bend) |
| `less`, `less_equal`, `greater`, `greater_equal`, `equal`, `not_equal` | 1.0 or 0.0 by the comparison of `left` and `right` at every index | [tensor_comparison_values.bend](../lib/proofs/tensor_comparison_values.bend) |
| `sigmoid`, `silu`, `tanh`, `gelu_tanh`, `exp`, `log`, `sqrt` | the ordered map of the scalar function in [math_fp32.bend](../lib/math_fp32.bend) | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend), [tensor_band_normal_unary.bend](../lib/proofs/tensor_band_normal_unary.bend) |
| `activate`, `activate_derivative` | the activation of [math_activation.bend](../lib/math_activation.bend), or its derivative, of the value at every index | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `activate_forward`, `activate_backward` | with any activation but `Custom`, the activation at every index beside the input with its values, both Ready in its shape. For a gradient and an input of one shape, the gradient times the derivative at the input | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `Optimizer.sgd` | `parameter - gradient * rate` at every index, for a parameter and a gradient of one shape | [tensor_optimizer_values.bend](../lib/proofs/tensor_optimizer_values.bend) |
| `BatchNorm.forward`, `BatchNorm.backward`, `BatchNorm.evaluate` | for x in `[N, C, H, W]` and per-channel tensors in `[C]`: forward gives `xhat * gamma + beta` with `xhat = (x - mean) * deviation`, `mean = sum_axes(x, [0, 2, 3]) / M`, `deviation = 1 / sqrt(s / M + epsilon)` of `s = sum_axes(d * d, [0, 2, 3])`, and the running statistics moved towards `mean` and `s / (M - 1)` by `momentum`. Backward gives `dbeta`, `dgamma` and `dx` from the saved `xhat` and `deviation` in the same steps. Evaluate gives the affine output of the running statistics | [tensor_batch_norm_values.bend](../lib/proofs/tensor_batch_norm_values.bend) |
| `Loss.mse`, `Loss.cross_entropy` | for a prediction or logits and a target of one shape: the loss, its gradient and the target. With `d = prediction - target`, `mse` gives `sum(d * d)` and `d * 2`, both divided by the count for `mean`. `cross_entropy` takes logits `[N, K]`: `p = c - log(sum(exp(c)))` along each row of `c = x - max`, the loss `-sum(target * p) / N` and the gradient `(exp(p) - target) / N`. Sums use [math_sum.bend](../lib/math_sum.bend) | [tensor_loss_values.bend](../lib/proofs/tensor_loss_values.bend), [tensor_cross_entropy_values.bend](../lib/proofs/tensor_cross_entropy_values.bend) |
| `Train.forward`, `Train.evaluate`, `Train.backward` of the average pool | forward and evaluate: at each image and channel of `[N, C, H, W]` the sum of its positions divided by `H * W`, in `[N, C, 1, 1]`. backward: `dy / (H * W)` at the address the view of `broadcast_to` gives to each index of `[N, C, H, W]` | [tensor_training_pool_values.bend](../lib/proofs/tensor_training_pool_values.bend) |
| `copy`, `reshape` | Both copy results preserve the original values, reshape preserves their order | [tensor_copy_contract.bend](../lib/proofs/tensor_copy_contract.bend), [tensor_reshape_values.bend](../lib/proofs/tensor_reshape_values.bend) |
| `softmax` | FP32 `exp(x[i] - m) / s`, with `m` the fold of the axis in increasing index and `s` the sum of [math_sum.bend](../lib/math_sum.bend) along the axis | [tensor_softmax_values.bend](../lib/proofs/tensor_softmax_values.bend) |
| `select` | The mask's magnitude bits choose yes or no and preserve all 32 bits of the chosen word | [tensor_select_public.bend](../lib/proofs/tensor_select_public.bend) for inputs of one shape, [tensor_select_broadcast.bend](../lib/proofs/tensor_select_broadcast.bend) for broadcasting |
| `transpose`, `slice`, `broadcast_to` | the input at the address the view gives to that index | [tensor_gather_public.bend](../lib/proofs/tensor_gather_public.bend), [tensor_band_view_values.bend](../lib/proofs/tensor_band_view_values.bend) |
| `flip` | the input at the address its view gives to the digits of that index, with the digit `d` of every listed axis replaced by `extent - 1 - d` | [tensor_gather_public.bend](../lib/proofs/tensor_gather_public.bend), [tensor_band_view_values.bend](../lib/proofs/tensor_band_view_values.bend) |
| `set_slice`, `pad` | the part written through its view, the rest unchanged | [tensor_set_slice_values.bend](../lib/proofs/tensor_set_slice_values.bend), [tensor_band_view_writes.bend](../lib/proofs/tensor_band_view_writes.bend), [view_writes.bend](../lib/proofs/view_writes.bend) |
| `concat` | the part that covers the place of that index along the axis, read at that place of `[outer, extent, inner]` | [tensor_concat_public.bend](../lib/proofs/tensor_concat_public.bend), [tensor_band_concat_semantics.bend](../lib/proofs/tensor_band_concat_semantics.bend) |
| `max_axes`, `min_axes`, `max`, `min` | at each index of the kept axes, the fold of the values of the listed axes in row-major order, from the initial value. `max` and `min` list every axis and give a rank-0 result | [tensor_reduce_values.bend](../lib/proofs/tensor_reduce_values.bend) |
| `sum_axes`, `mean_axes`, `sum`, `mean` | the same values added by [math_sum.bend](../lib/math_sum.bend). A mean divides the sum by the number of values | [tensor_reduce_values.bend](../lib/proofs/tensor_reduce_values.bend) |
| `upsample_nearest` | the input at the nearest index of the last two axes | [tensor_gather_public.bend](../lib/proofs/tensor_gather_public.bend) |
| `max_pool2d` | the fold of each window, rows then columns, in increasing index, from -inf | [tensor_band_pool_public.bend](../lib/proofs/tensor_band_pool_public.bend) |

The `softmax` value is that FP32 sequence, with the library `exp`. Nothing is claimed about the real-number normalization. [math_select_values.bend](../lib/proofs/math_select_values.bend) states the word `select` returns: `yes` when the mask has a nonzero magnitude bit, `no` otherwise.

A band input gives the same values. Functions, operations with a scalar and indexed maps run on each band and keep the band layout well formed ([tensor_band_map.bend](../lib/proofs/tensor_band_map.bend), [tensor_band_indexed.bend](../lib/proofs/tensor_band_indexed.bend), [tensor_band_normal_unary.bend](../lib/proofs/tensor_band_normal_unary.bend)). Max pooling runs on the planes of the last two axes: bands that split the image rows stay, any other input arrives as one band of its planes, and each band pools the rows its outputs read. Transpose, slice, nearest upsampling, `set_slice` and `pad` go through a view: bands whose images and rows the view keeps in place run it on their own rows, other inputs first arrive as one band ([tensor_band_view_values.bend](../lib/proofs/tensor_band_view_values.bend), [tensor_band_view_writes.bend](../lib/proofs/tensor_band_view_writes.bend)). Two operands of one shape, and the parts of a `concat` along any axis, meet in the plan one of them offers, arrive in it and run band by band ([tensor_band_combine.bend](../lib/proofs/tensor_band_combine.bend), [tensor_concat_public.bend](../lib/proofs/tensor_concat_public.bend)). The pooling and `concat` statements are for a nonempty result. A reduction arrives as lines of `[kept, listed]`, through the view that moves the listed axes last unless they are last already, and folds the lines band by band ([tensor_reduce_values.bend](../lib/proofs/tensor_reduce_values.bend)). The other layout operations run on their operands arrived as one whole band. Each statement gives that band's buffer the operand's values and reads the result, one whole band, as the contract ([tensor_whole_leaf.bend](../lib/proofs/tensor_whole_leaf.bend)).

Underneath:

| Subject | What is shown | Source |
|---|---|---|
| Bounds and non-overlap | From the guards the code checks: every read and write is in range, and source and destination do not overlap | [tensor_operation_bounds.bend](../lib/proofs/tensor_operation_bounds.bend) |
| Cursor | The native cursor visits coordinates in row-major order. Copying in runs of the innermost axis equals visiting each element | [tensor_cursor_refinement.bend](../lib/proofs/tensor_cursor_refinement.bend), [tensor_cursor_order.bend](../lib/proofs/tensor_cursor_order.bend), [tensor_access_refinement.bend](../lib/proofs/tensor_access_refinement.bend) |
| Storage certificates | A buffer carries a proof that its array has the shape the model says. Certificate work is constant per operation | [storage_buffer_refinement.bend](../lib/proofs/storage_buffer_refinement.bend) |
| Failure | Which checked call each public function makes. A failed input stays failed | [tensor_public_refinement.bend](../lib/proofs/tensor_public_refinement.bend) |
| Plan recursion | `Bands.each_part` runs every band operation and the parts of the array convolution and input gradient over one plan, and is proved once. Each operation proves only its leaf and its split | [tensor_band_walk.bend](../lib/proofs/tensor_band_walk.bend), [tensor_band_each.bend](../lib/proofs/tensor_band_each.bend) |
| Band layout | When a band tensor is well formed, and its values by channel, row and column | [tensor_band_shape_refinement.bend](../lib/proofs/tensor_band_shape_refinement.bend), [tensor_band_values.bend](../lib/proofs/tensor_band_values.bend) |
| Arrival | A tensor brought to the layout an operation runs in keeps its values: bands of the same geometry are aligned, others are read in row-major order and cut | [tensor_band_arrive.bend](../lib/proofs/tensor_band_arrive.bend), [tensor_band_target.bend](../lib/proofs/tensor_band_target.bend) |
| Indexed maps | Running an indexed map in blocks equals running it element by element | [tensor_indexed_refinement.bend](../lib/proofs/tensor_indexed_refinement.bend), [tensor_operations_refinement.bend](../lib/proofs/tensor_operations_refinement.bend) |
| Axis order | `permute` with order `[1, 0, 2, 3]` exchanges the first two axes of a four-axis kernel view and their strides | [tensor_adjoint_index.bend](../lib/proofs/tensor_adjoint_index.bend) |

## Reading a tensor

[tensor_exit_values.bend](../lib/proofs/tensor_exit_values.bend) reads a tensor of the contract of a value reader, in any layout, as the reader's values in flat index order.

| Function | What it gives |
|---|---|
| `into_storage` | Storage whose element `i` is the reader's value at `i` |
| `to_list` | The list of the reader's values at `0` to `n - 1` |
| `item` | For a tensor of one element, the reader's value at `0` |
| `save_raw`, `save_npy` | They write that storage, after the `.npy` header for `save_npy`, and return a tensor of the same contract |

## What is not proved

- The planner's cost estimates, latency and workspace accounting. They cannot affect a result.
- Real-number accuracy. [math_accuracy.py](../benchmarks/checks/math_accuracy.py) measures it.
- A closed form of which indices a slice view reaches. The statements follow the cursor's coordinates.
- The YOLO example's graph, I/O and detection decoding.
- `Train.one_hot`. The classifier case of `python run.py gradients` compares it with PyTorch.
- Compiler and runtime behaviour.

## Inspecting

`python run.py proofs` also writes the list of declarations with their premises, and a reference graph from the root, under `out/results/proof-audit/`. They are for navigation. The verdict is the checker's.
