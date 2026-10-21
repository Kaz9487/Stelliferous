# 證明

[English](proofs.md) | [繁體中文](proofs.zh-TW.md)

每個證明都是一個 Bend term，由未修改的官方 Bend 檢查器檢查。

```sh
python run.py proofs
```

## 根定理

[native_inference_law.bend](../convolution/proofs/native_inference_law.bend)：

```text
law native_inference_matches_matrix:
  for +problem: ConvSpec.Problem
  {NativeConv.infer(problem) == MatrixSpec.infer(problem) : ConvSpec.ConvOutcome}
```

一個 problem 包含形狀、輸入、權重與 bias。這條定律對它不設任何條件：涵蓋被拒絕的輸入、參考實作的 fallback 與原生執行，對每個分割深度（`inference_with_any_split_depth_matches_matrix`）與兩種策略（`every_strategy_matches_matrix`）都成立。

**數值契約。**每個輸出從 FP32 的 +0.0 開始，依輸入通道、kernel 列、kernel 行的順序加上各個乘積（先乘後加），最後加上 bias。Padding 的取樣也依這個順序參與。定理談的是這一串 FP32 運算，不是實數，也不允許 FMA 或重新結合。

**凍結檔。**[specification.bend](../convolution/specification.bend)、[specification_law.bend](../convolution/specification_law.bend)、[ordered_matrix.bend](../lib/ordered_matrix.bend)、[matrix_specification.bend](../convolution/matrix_specification.bend) 與 [matrix_convolution_law.bend](../convolution/proofs/matrix_convolution_law.bend) 的雜湊必須與 [frozen.json](../data/contracts/frozen.json) 相同。每次檢查證明時都會核對。

## 信任什麼

- 官方檢查器與 Base 函式庫。
- 編譯器、runtime 與其排程器、C 編譯器與硬體。定理談的是 Bend 原始碼。
- `load` 與 `save` 底下的 Bend 檔案 I/O。函式庫與卷積的證明不含任何 `@unsafe` 定義。

證明沒有涵蓋的部分由數值檢查負責：產生的程式碼、模型接線、activation、I/O 與偵測解碼，都與 NumPy 及 PyTorch 比對，數學函式則在每一個 FP32 輸入上與 MPFR 比對（[benchmarks](../benchmarks/README.zh-TW.md)）。

## 卷積證明的路線

| 義務 | 來源 |
|---|---|
| Direct（純量）執行 | [scalar_inference_refinement.bend](../convolution/proofs/scalar_inference_refinement.bend) |
| Packed 執行 | [packed_inference_refinement.bend](../convolution/proofs/packed_inference_refinement.bend) |
| 實際的流程：重排、分割、pack 與 GEMM、gather | [microkernel_inference_refinement.bend](../convolution/proofs/microkernel_inference_refinement.bend)、[microkernel_storage_execution.bend](../convolution/proofs/microkernel_storage_execution.bend) |
| 權重重排成八列一組的 tile | [weight_reorder_readback.bend](../convolution/proofs/weight_reorder_readback.bend) |
| Packing，以及它不會改動的輸入 | [packing_storage_refinement.bend](../convolution/proofs/packing_storage_refinement.bend)、[packing_contents.bend](../convolution/proofs/packing_contents.bend) |
| 把 GEMM 的儲存寫成每個 job 的 view 寫入，以及邊界、數值與涵蓋範圍 | [kernel_gemm_writes.bend](../lib/proofs/kernel_gemm_writes.bend)、[kernel_gemm_readback.bend](../lib/proofs/kernel_gemm_readback.bend)、[view_writes.bend](../lib/proofs/view_writes.bend) |
| 累加器等於矩陣內積 | [gemm_dot_matches_matrix.bend](../convolution/proofs/gemm_dot_matches_matrix.bend) |
| 每個 leaf 的輸出等於矩陣的值 | [microkernel_leaf_readback.bend](../convolution/proofs/microkernel_leaf_readback.bend) |
| 每次分割的輸入列視窗 | [window_partition.bend](../convolution/proofs/window_partition.bend)、[window_leaf_readback.bend](../convolution/proofs/window_leaf_readback.bend) |
| Gather 寫入與共用裁切界 | [gather_write_program.bend](../convolution/proofs/gather_write_program.bend)、[kernel_gather_width.bend](../lib/proofs/kernel_gather_width.bend) |
| 呼叫端 array 的任意容量 | [array_api_validation.bend](../convolution/proofs/array_api_validation.bend) |
| 共用的儲存契約 | [storage_refinement.bend](../lib/proofs/storage_refinement.bend) |

做法從頭到尾相同：把迴圈存入的內容寫成一份寫入清單，證明每次寫入都在界內，證明寫入的值，再證明後面的寫入不會動到它。

## 共用 stencil pipeline

共用 pipeline 接受長方形核位置，以及彼此獨立的列、行起點。方形 wrapper 保留前向卷積的介面。以下模組描述共用的 packing 與 GEMM 操作：

| 義務 | 來源 |
|---|---|
| Stencil 取樣與 native packed tile 讀回 | [kernel_spatial_packing_samples.bend](../lib/proofs/kernel_spatial_packing_samples.bend)、[kernel_packing_window_coordinates.bend](../lib/proofs/kernel_packing_window_coordinates.bend)、[kernel_packing_window_readback.bend](../lib/proofs/kernel_packing_window_readback.bend) |
| 有序 stencil dot 與有界 block 讀回 | [kernel_pipeline_window_values.bend](../lib/proofs/kernel_pipeline_window_values.bend)、[kernel_pipeline_window_readback.bend](../lib/proofs/kernel_pipeline_window_readback.bend) |

## 公開的卷積

`F.conv2d` 透過帶有儲存憑證的 buffer 執行同一個 kernel（[kernel_pipeline.bend](../lib/kernel_pipeline.bend)）。根定理是針對 Array 引擎敘述的。公開路徑有自己的定理：

- **Band 路徑：**每次呼叫都走這條路徑。輸入會到位成規劃器選擇的 layout：輸入已是輸入幾何時，從它自己的 plan 重新選擇，否則用第一次分割。對程式實際選擇的規劃，輸出的 band 涵蓋每一列且不超出其配置（[band_pipeline_well_formed.bend](../convolution/proofs/band_pipeline_well_formed.bend)），而且每個存入的值都是該輸出點的有序矩陣值（[convolution_leaf_contract.bend](../convolution/proofs/convolution_leaf_contract.bend)、[band_pipeline_execution.bend](../convolution/proofs/band_pipeline_execution.bend)）。
- **呼叫本身：**輸入為 Usable 時，不論輸入、權重與 bias 是哪種 layout，各自存放自己 value reader 的值，`F.conv2d` 回傳的張量都是 Ready、形狀為這次呼叫的輸出形狀、良構的，且每個元素都是這些值的有序卷積值。失敗的輸入或參數維持失敗。不合法的幾何、與幾何不符的參數，或超出機器 guard 的形狀，會得到 Failed 結果（[convolution_public.bend](../convolution/proofs/convolution_public.bend)）。

這些與無條件的根定理是分開的。它們的前提是：輸入為 Usable、layout 良構、數值有限、形狀相符，以及程式會檢查的機器 guard。

## 輸入梯度

[input_gradient_specification.bend](../convolution/input_gradient_specification.bend) 定義任意步幅的卷積對輸入的梯度：每個輸入元素依輸出通道遞增、核的列與行遞減，只取與它補零後位置對步幅同餘的核位置，把乘積從 +0.0 加總，最後加 +0.0。落在輸出梯度之外的同餘核位置當作零。

[input_gradient_public.bend](../convolution/proofs/input_gradient_public.bend) 的 `input` 證明 `Gradient.input` 回傳的張量就是這個值：輸出梯度與權重可以是任何 layout，各自存放自己 value reader 的值，Ready 結果是 Normal，每個元素都是所屬影像對這些值的規格值，權重也以自己形狀的張量帶著自己的值傳回。有或沒有批次軸都成立，也包括沒有影像的批次。

| 檔案 | 敘述 |
|------|------|
| [tensor_full_values.bend](../lib/proofs/tensor_full_values.bend) | Ready 的 `full` 張量每個元素都是指定的值。 |
| [tensor_row_major_coordinates.bend](../lib/proofs/tensor_row_major_coordinates.bend) | 複製或翻轉在某個列優先索引讀取的位址，由該索引的各位數給出。 |
| [input_gradient_call.bend](../convolution/proofs/input_gradient_call.bend) | `Gradient.input` 接受參數時，它檢查的每個形狀與機器界限都成立。 |
| [input_gradient_tiled_run.bend](../convolution/proofs/input_gradient_tiled_run.bend) | 內核執行後交回權重，它的 band 就是整批的 band。 |
| [input_gradient_tiled_images.bend](../convolution/proofs/input_gradient_tiled_images.bend) | 整批的 band 是良構的，每格都是內核緩衝區的規格值。 |
| [input_gradient_tiled_leaves.bend](../convolution/proofs/input_gradient_tiled_leaves.bend) | 內核的每個葉在它的 band 的每一格都是規格值。 |

葉的證明建立在以下義務上。

| 義務 | 來源 |
|---|---|
| 實際瓦片、位移表與權重的打包讀回 | [input_gradient_tiled_chunk_reads.bend](../lib/proofs/input_gradient_tiled_chunk_reads.bend)、[input_gradient_tiled_weight_reads.bend](../lib/proofs/input_gradient_tiled_weight_reads.bend)、[input_gradient_tiled_pack_values.bend](../lib/proofs/input_gradient_tiled_pack_values.bend) |
| 保留的來源列與實際 MAC 操作數 | [input_gradient_tiled_source_rows.bend](../lib/proofs/input_gradient_tiled_source_rows.bend)、[input_gradient_tiled_product_reads.bend](../lib/proofs/input_gradient_tiled_product_reads.bend)、[input_gradient_tiled_products.bend](../convolution/proofs/input_gradient_tiled_products.bend) |
| 每筆寫入的有序 MAC 值，以及查詢點必被寫入 | [input_gradient_tiled_dot.bend](../convolution/proofs/input_gradient_tiled_dot.bend)、[input_gradient_tiled_written.bend](../convolution/proofs/input_gradient_tiled_written.bend)、[input_gradient_tiled_coverage.bend](../lib/proofs/input_gradient_tiled_coverage.bend) |
| 分塊幾何、容量與來源 owner 保持 | [input_gradient_tiled_chunk_geometry.bend](../lib/proofs/input_gradient_tiled_chunk_geometry.bend)、[input_gradient_tiled_capacity.bend](../lib/proofs/input_gradient_tiled_capacity.bend)、[input_gradient_tiled_leaf_execution.bend](../lib/proofs/input_gradient_tiled_leaf_execution.bend) |

## 權重與偏差梯度

[parameter_gradient_specification.bend](../convolution/parameter_gradient_specification.bend) 定義每個權重與每個偏差的梯度：每張影像裡，連續的輸出列的項組成一組，每組從 +0.0 依序相加，各組再用相鄰配對樹相加，各張影像也用同一種樹。

[parameter_gradient_ready.bend](../convolution/proofs/parameter_gradient_ready.bend) 證明 `Gradient.weight` 與 `Gradient.bias` 的 Ready 結果在每個索引上，對輸入與輸出梯度的 value reader 的值，都是這個值，運算元可以是任何 layout，對任意組長成立。運算元也以自己形狀的張量帶著自己的值傳回。

| 檔案 | 敘述 |
|------|------|
| [grouped_reduction_refinement.bend](../lib/proofs/grouped_reduction_refinement.bend) | 分組走訪就是每組的依序累加與各組的配對樹。 |
| [list_storage_readback.bend](../lib/proofs/list_storage_readback.bend)、[tensor_list_values.bend](../lib/proofs/tensor_list_values.bend) | 由串列建立的 Ready 張量保有串列的每個項目，也保有在形狀內與串列一致的任何 value reader 的值。 |
| [parameter_gradient_bridge.bend](../convolution/proofs/parameter_gradient_bridge.bend) | 排列的 lane 與有序組／影像歸約等於規格。 |

權重內核把梯度通道與輸入位置排列好，用共用的乘加內核（不加偏差）相乘，各組與各張影像用函式庫的相鄰配對樹相加。偏差梯度使用純量內核。

| 檔案 | 敘述 |
|---|---|
| [kernel_weight_gradient_packing.bend](../lib/proofs/kernel_weight_gradient_packing.bend) | 原生遮罩讀取與兩個排列遍歷等於 canonical 儲存模型。 |
| [kernel_weight_gradient_group.bend](../lib/proofs/kernel_weight_gradient_group.bend) | 有界區塊、尾端 tile 與依序接續都執行共享 GEMM 模型的不加偏差特化。 |
| [kernel_weight_gradient_refinement.bend](../lib/proofs/kernel_weight_gradient_refinement.bend) | 完整的組與影像配對樹、轉置讀回等於原生權重內核的儲存模型。 |

## 訓練用的層

層的定理組合公開的卷積、activation、乘法與梯度定理。每個運算元可以是任何 layout，各自存放自己 value reader 的值。forward 的前提與公開卷積相同。

| 檔案 | 敘述 |
|---|---|
| [layer_public.bend](../convolution/proofs/layer_public.bend) | activation 不是 `Custom` 時，`Layer.forward` 的輸出是 Ready，形狀是這次呼叫的輸出形狀，每個索引上是依序卷積值的 activation。保存的一對是輸入與這個卷積值，卷積值也是同一形狀的 Ready，權重與偏差也傳回。`Layer.backward` 與 `Layer.backward_parameters` 的 Ready 梯度是 x、dz = dy * activation'(z) 與權重的輸入、權重、偏差梯度規格值，權重也傳回。 |

`Train.forward`、`Train.backward` 與 `Train.evaluate` 把卷積交給以 identity activation 呼叫的 `Layer.forward`、`Layer.backward` 與 `conv2d`，batch normalization 交給 `BatchNorm`，activation 交給 `activate_forward`、`activate_backward` 與 `activate`。`Train.sgd_step` 套用 `Optimizer.sgd`。

## 張量運算

每個定理對呼叫只有一個前提：它回傳了 Usable 的結果。運算元可以是任何 layout。每個運算元與結果都滿足 [tensor_global_shape_count.bend](../lib/proofs/tensor_global_shape_count.bend) 的 contract：張量是 Normal，且其 band 的每個元素都是 value reader 在它所在索引給出的值。Value reader 是一個 context 加上從索引到值的函式，每個運算元各有自己的 value reader。

| 運算 | Usable 的結果裡是什麼 | 來源 |
|---|---|---|
| `add`、`sub`、`mul`、`div`、`maximum`、`minimum`，含廣播 | 每個索引上是 `operation(left, right)`，參數順序如此 | 形狀相同的輸入：[tensor_combine_public.bend](../lib/proofs/tensor_combine_public.bend)，需要廣播：[tensor_combine_broadcast.bend](../lib/proofs/tensor_combine_broadcast.bend) |
| `add_scalar`、`mul_scalar` 等 | `operation(x, scalar)` | [tensor_band_normal_unary.bend](../lib/proofs/tensor_band_normal_unary.bend) |
| `less`、`less_equal`、`greater`、`greater_equal`、`equal`、`not_equal` | 每個索引上依 `left` 與 `right` 的比較為 1.0 或 0.0 | [tensor_comparison_values.bend](../lib/proofs/tensor_comparison_values.bend) |
| `sigmoid`、`silu`、`tanh`、`gelu_tanh`、`exp`、`log`、`sqrt` | [math_fp32.bend](../lib/math_fp32.bend) 中純量函式的依序 map | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend)、[tensor_band_normal_unary.bend](../lib/proofs/tensor_band_normal_unary.bend) |
| `activate`、`activate_derivative` | 每個索引上的值經 [math_activation.bend](../lib/math_activation.bend) 的 activation，或它的導數 | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `activate_forward`、`activate_backward` | `Custom` 以外的 activation：每個索引上的 activation，旁邊是帶著原值的輸入，兩者都在它的形狀中 Ready。同形狀的梯度與輸入：每個索引上是梯度乘以輸入處的導數 | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `Optimizer.sgd` | 同形狀的參數與梯度，每個索引上是 `parameter - gradient * rate` | [tensor_optimizer_values.bend](../lib/proofs/tensor_optimizer_values.bend) |
| `BatchNorm.forward`、`BatchNorm.backward`、`BatchNorm.evaluate` | x 為 `[N, C, H, W]`，各通道張量為 `[C]`：forward 給出 `xhat * gamma + beta`，其中 `xhat = (x - mean) * deviation`、`mean = sum_axes(x, [0, 2, 3]) / M`、`deviation = 1 / sqrt(s / M + epsilon)`、`s = sum_axes(d * d, [0, 2, 3])`，執行統計量以 `momentum` 移向 `mean` 與 `s / (M - 1)`。backward 以保存的 `xhat` 與 `deviation` 用同樣的步驟給出 `dbeta`、`dgamma` 與 `dx`。evaluate 給出執行統計量的仿射輸出 | [tensor_batch_norm_values.bend](../lib/proofs/tensor_batch_norm_values.bend) |
| `Loss.mse`、`Loss.cross_entropy` | 預測或 logits 與同形狀的 target：損失、它的梯度與 target。令 `d = prediction - target`，`mse` 給出 `sum(d * d)` 與 `d * 2`，`mean` 時兩者都除以元素數。`cross_entropy` 的 logits 為 `[N, K]`：對 `c = x - max` 的每一列取 `p = c - log(sum(exp(c)))`，損失為 `-sum(target * p) / N`，梯度為 `(exp(p) - target) / N`。加總使用 [math_sum.bend](../lib/math_sum.bend) | [tensor_loss_values.bend](../lib/proofs/tensor_loss_values.bend)、[tensor_cross_entropy_values.bend](../lib/proofs/tensor_cross_entropy_values.bend) |
| 平均池化的 `Train.forward`、`Train.evaluate`、`Train.backward` | forward 與 evaluate：`[N, C, H, W]` 每張影像每個通道的位置總和除以 `H * W`，形狀 `[N, C, 1, 1]`。backward：`[N, C, H, W]` 每個索引在 `broadcast_to` 的 view 給出的位址上的 `dy / (H * W)` | [tensor_training_pool_values.bend](../lib/proofs/tensor_training_pool_values.bend) |
| `copy`、`reshape` | copy 的兩個回傳張量保留原值，reshape 保留依序值 | [tensor_copy_contract.bend](../lib/proofs/tensor_copy_contract.bend)、[tensor_reshape_values.bend](../lib/proofs/tensor_reshape_values.bend) |
| `softmax` | FP32 的 `exp(x[i] - m) / s`，`m` 是沿軸依索引遞增的 fold，`s` 是沿軸以 [math_sum.bend](../lib/math_sum.bend) 計算的總和 | [tensor_softmax_values.bend](../lib/proofs/tensor_softmax_values.bend) |
| `select` | 遮罩的 magnitude 位元選擇 yes 或 no，被選中的字保留全部 32 位元 | 形狀相同的輸入：[tensor_select_public.bend](../lib/proofs/tensor_select_public.bend)，需要廣播：[tensor_select_broadcast.bend](../lib/proofs/tensor_select_broadcast.bend) |
| `transpose`、`slice`、`broadcast_to` | view 給該索引的位址上的輸入值 | [tensor_gather_public.bend](../lib/proofs/tensor_gather_public.bend)、[tensor_band_view_values.bend](../lib/proofs/tensor_band_view_values.bend) |
| `flip` | view 給該索引各位數的位址上的輸入值，其中每個列出的軸的位數 `d` 換成 `extent - 1 - d` | [tensor_gather_public.bend](../lib/proofs/tensor_gather_public.bend)、[tensor_band_view_values.bend](../lib/proofs/tensor_band_view_values.bend) |
| `set_slice`、`pad` | 透過 view 寫入的部分，其餘不變 | [tensor_set_slice_values.bend](../lib/proofs/tensor_set_slice_values.bend)、[tensor_band_view_writes.bend](../lib/proofs/tensor_band_view_writes.bend)、[view_writes.bend](../lib/proofs/view_writes.bend) |
| `concat` | 沿該軸涵蓋該索引位置的那個部分，在 `[outer, extent, inner]` 的同一位置讀出的值 | [tensor_concat_public.bend](../lib/proofs/tensor_concat_public.bend)、[tensor_band_concat_semantics.bend](../lib/proofs/tensor_band_concat_semantics.bend) |
| `max_axes`、`min_axes`、`max`、`min` | 在保留軸的每個索引上，列出各軸的值依列優先順序、從初始值開始的 fold。`max` 與 `min` 列出所有軸，結果為零維 | [tensor_reduce_values.bend](../lib/proofs/tensor_reduce_values.bend) |
| `sum_axes`、`mean_axes`、`sum`、`mean` | 同一組值以 [math_sum.bend](../lib/math_sum.bend) 計算的總和。mean 是總和除以值的個數 | [tensor_reduce_values.bend](../lib/proofs/tensor_reduce_values.bend) |
| `upsample_nearest` | 最後兩軸取最近索引處的輸入值 | [tensor_gather_public.bend](../lib/proofs/tensor_gather_public.bend) |
| `max_pool2d` | 每個視窗先列後行、依索引遞增、從 -inf 開始的 fold | [tensor_band_pool_public.bend](../lib/proofs/tensor_band_pool_public.bend) |

`softmax` 的值就是這串 FP32 運算，`exp` 是函式庫的 `exp`，不涉及實數上的正規化。[math_select_values.bend](../lib/proofs/math_select_values.bend) 說明 `select` 回傳哪個字：遮罩的 magnitude 位元非零時是 `yes`，否則是 `no`。

band 輸入得到相同的值。函式、與純量的運算及依索引的 map 在每個 band 上執行，並保持 band layout 良構（[tensor_band_map.bend](../lib/proofs/tensor_band_map.bend)、[tensor_band_indexed.bend](../lib/proofs/tensor_band_indexed.bend)、[tensor_band_normal_unary.bend](../lib/proofs/tensor_band_normal_unary.bend)）。Max pooling 在最後兩軸的平面上執行：分帶已切在影像列上就保留，其他輸入到位成一個由平面組成的 band，每個 band 對其輸出所需的列做 pooling。Transpose、slice、nearest upsample、`set_slice` 與 `pad` 經由一個 view 執行：view 讓影像與列留在原位的 band 在自己的列上執行，其他輸入先到位成一個 band（[tensor_band_view_values.bend](../lib/proofs/tensor_band_view_values.bend)、[tensor_band_view_writes.bend](../lib/proofs/tensor_band_view_writes.bend)）。形狀相同的兩個運算元，以及沿任一軸 `concat` 的各部分，會在其中一個提出的 plan 會合、到位成該 plan，再逐 band 執行（[tensor_band_combine.bend](../lib/proofs/tensor_band_combine.bend)、[tensor_concat_public.bend](../lib/proofs/tensor_concat_public.bend)）。Pooling 與 `concat` 的敘述針對非空的結果。Reduction 以 `[保留軸, 列出軸]` 的列到位，列出軸不在最後時經由把它們移到最後的 view，再逐 band fold 各列（[tensor_reduce_values.bend](../lib/proofs/tensor_reduce_values.bend)）。其他 layout 運算在到位成一整個 band 的運算元上執行。每個敘述把運算元的值交給該 band 的緩衝區，再把結果（一整個 band）讀成 contract（[tensor_whole_leaf.bend](../lib/proofs/tensor_whole_leaf.bend)）。

底層：

| 主題 | 證明的內容 | 來源 |
|---|---|---|
| 邊界與不重疊 | 從程式檢查的 guard 推出：每次讀寫都在範圍內，來源與目的不重疊 | [tensor_operation_bounds.bend](../lib/proofs/tensor_operation_bounds.bend) |
| Cursor | 原生 cursor 依 row-major 順序走訪座標。以最內層軸為單位成段複製，等於逐元素走訪 | [tensor_cursor_refinement.bend](../lib/proofs/tensor_cursor_refinement.bend)、[tensor_cursor_order.bend](../lib/proofs/tensor_cursor_order.bend)、[tensor_access_refinement.bend](../lib/proofs/tensor_access_refinement.bend) |
| 儲存憑證 | Buffer 帶著一個證明：它的 array 具有模型所說的形狀。憑證的工作量每個運算是常數 | [storage_buffer_refinement.bend](../lib/proofs/storage_buffer_refinement.bend) |
| 失敗 | 每個公開函式呼叫哪個檢查過的運算。失敗的輸入維持失敗 | [tensor_public_refinement.bend](../lib/proofs/tensor_public_refinement.bend) |
| Plan 遞迴 | `Bands.each_part` 依同一個 plan 執行所有 band 運算，以及 array 卷積與輸入梯度的各部分，只證明一次。各運算只證明自己的 leaf 與切分 | [tensor_band_walk.bend](../lib/proofs/tensor_band_walk.bend)、[tensor_band_each.bend](../lib/proofs/tensor_band_each.bend) |
| Band layout | Band 張量何時良構，以及以通道、列、行表示的數值 | [tensor_band_shape_refinement.bend](../lib/proofs/tensor_band_shape_refinement.bend)、[tensor_band_values.bend](../lib/proofs/tensor_band_values.bend) |
| Arrival | 張量到位成運算所用的切法後值不變：同幾何的分帶直接對齊，其他的依列優先順序讀出再切開 | [tensor_band_arrive.bend](../lib/proofs/tensor_band_arrive.bend)、[tensor_band_target.bend](../lib/proofs/tensor_band_target.bend) |
| 依索引的 map | 分塊執行依索引的 map，等於逐元素執行 | [tensor_indexed_refinement.bend](../lib/proofs/tensor_indexed_refinement.bend)、[tensor_operations_refinement.bend](../lib/proofs/tensor_operations_refinement.bend) |
| 軸的順序 | 以順序 `[1, 0, 2, 3]` 做 `permute`，會交換四軸 kernel view 的前兩軸及其步長 | [tensor_adjoint_index.bend](../lib/proofs/tensor_adjoint_index.bend) |

## 讀出張量

[tensor_exit_values.bend](../lib/proofs/tensor_exit_values.bend)：滿足某個讀值函式的 contract 的張量，不論存法，讀出來就是讀值函式依平坦索引排列的值。

| 函式 | 給出什麼 |
|---|---|
| `into_storage` | 第 `i` 個元素是讀值函式在 `i` 的值的儲存區 |
| `to_list` | 讀值函式在 `0` 到 `n - 1` 的值組成的串列 |
| `item` | 只有一個元素的張量：讀值函式在 `0` 的值 |
| `save_raw`、`save_npy` | 寫出那個儲存區（`save_npy` 先寫 `.npy` 的檔頭），並回傳滿足同一個 contract 的張量 |

## 沒有證明的部分

- 規劃器的成本估計、延遲與 workspace 計算。它們不會影響結果。
- 實數上的精確度。由 [math_accuracy.py](../benchmarks/checks/math_accuracy.py) 量測。
- Slice view 會到達哪些索引的封閉形式。敘述是跟著 cursor 的座標走的。
- YOLO 範例的運算圖、I/O 與偵測解碼。
- `Train.one_hot`。`python run.py gradients` 的分類案例拿它與 PyTorch 比對。
- 編譯器與 runtime 的行為。

## 查看

`python run.py proofs` 也會把所有宣告及其前提的清單，以及從根定理出發的參照圖，寫到 `out/results/proof-audit/`。它們用來導覽，判定以檢查器為準。
