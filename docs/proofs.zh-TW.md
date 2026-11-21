# 證明

[English](proofs.md) | [繁體中文](proofs.zh-TW.md)

每個證明都是一個 Bend term，由未修改的官方 Bend checker 檢查。

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

一個問題包含形狀、輸入、權重與 bias。這條定律對它不加任何條件：涵蓋被拒絕的輸入、參考實作的退路與原生執行，對每一種分割深度（`inference_with_any_split_depth_matches_matrix`）與兩種策略（`every_strategy_matches_matrix`）都成立。

**數值契約。** 每個輸出從 FP32 +0.0 開始，依輸入通道、kernel 列、kernel 行的順序加上乘積（先乘再加），最後加上 bias。padding 的樣本也照這個順序參與。定理說的是這串 FP32 運算，不是實數，也不允許 fused multiply-add 或重新結合。

**凍結檔。** [specification.bend](../convolution/specification.bend)、[specification_law.bend](../convolution/specification_law.bend)、[ordered_matrix.bend](../lib/ordered_matrix.bend)、[matrix_specification.bend](../convolution/matrix_specification.bend) 與 [matrix_convolution_law.bend](../convolution/proofs/matrix_convolution_law.bend) 必須保持 [frozen.json](../data/contracts/frozen.json) 裡的雜湊。每次證明執行都會檢查。

## 信任的部分

- 官方 checker 與 Base 函式庫。
- 編譯器、執行期與它的排程器、C 編譯器與硬體。定理是關於 Bend 原始碼。
- `load` 與 `save` 下的 Bend 檔案 I/O。函式庫與卷積的證明沒有 `@unsafe` 定義。

數值檢查涵蓋證明沒涵蓋的部分：生成的程式碼、模型接線、activation、I/O 與偵測解碼和 NumPy、PyTorch 比對，數學函式以 MPFR 對每個 FP32 輸入比對（[benchmarks](../benchmarks/README.zh-TW.md)）。

## 跟著卷積的證明走

| 義務 | 來源 |
|---|---|
| 直接（純量）執行 | [scalar_inference_refinement.bend](../convolution/proofs/scalar_inference_refinement.bend) |
| 打包執行 | [packed_inference_refinement.bend](../convolution/proofs/packed_inference_refinement.bend) |
| Array 邊界、cells 分割與 GEMM、cells gather、輸出 untile | [array_cell_execution.bend](../convolution/proofs/array_cell_execution.bend)、[microkernel_storage_execution.bend](../convolution/proofs/microkernel_storage_execution.bend) |
| Array 邊界打包成八個 lane 的 cells | [kernel_weights_refinement.bend](../lib/proofs/kernel_weights_refinement.bend)、[window_cell_values.bend](../convolution/proofs/window_cell_values.bend) |
| leaf 的轉置與打包把問題的值填進 tiles | [kernel_spatial_packing_pixels.bend](../lib/proofs/kernel_spatial_packing_pixels.bend)、[window_leaf_readback.bend](../convolution/proofs/window_leaf_readback.bend)、[array_leaf_cells.bend](../convolution/proofs/array_leaf_cells.bend) |
| GEMM 的儲存是每個 job 的 view 的寫入，含界限、值與涵蓋 | [kernel_gemm_writes.bend](../lib/proofs/kernel_gemm_writes.bend)、[kernel_gemm_jobs.bend](../lib/proofs/kernel_gemm_jobs.bend)、[kernel_gemm_readback.bend](../lib/proofs/kernel_gemm_readback.bend)、[view_writes.bend](../lib/proofs/view_writes.bend) |
| 累加器等於矩陣的內積 | [gemm_dot_matches_matrix.bend](../convolution/proofs/gemm_dot_matches_matrix.bend) |
| 每個 leaf 的輸出等於矩陣值 | [microkernel_leaf_readback.bend](../convolution/proofs/microkernel_leaf_readback.bend) |
| 每種分割的輸入列視窗 | [window_partition.bend](../convolution/proofs/window_partition.bend)、[window_leaf_readback.bend](../convolution/proofs/window_leaf_readback.bend) |
| cell 寫入的界限、涵蓋與 lane 讀回 | [cell_gather_leaf.bend](../convolution/proofs/cell_gather_leaf.bend)、[gather_tree_readback.bend](../convolution/proofs/gather_tree_readback.bend)、[write_program_observation.bend](../lib/proofs/write_program_observation.bend) |
| 呼叫端任意的 array 容量 | [array_api_validation.bend](../convolution/proofs/array_api_validation.bend) |
| 共用的儲存契約 | [storage_refinement.bend](../lib/proofs/storage_refinement.bend) |

模式都一樣：把迴圈存的東西寫成一串寫入，證明每次寫入都在範圍內，證明寫入的值，再證明之後的寫入不會動到它。

## 張量契約

公開的張量持有一個值讀取器的 `TiledPublic.Contract`（[tensor_tiled_public.bend](../lib/proofs/tensor_tiled_public.bend)）：它的 cells 是形狀的標準分塊，沿 lane 軸排列、Usable，每個 lane 是讀取器在該索引給的值。超出 lane 軸長度的 lane 是 +0。值讀取器是一個 context 和一個從索引到值的函式，每個運算元各有一個。

下面每個定理都取每個運算元的契約，給出結果的契約。多數定理還取這些前提：

- 沒有長度為零的軸（空張量走另一條路線），
- 軸數不超過機器容量，limits 的容量等於預設容量，
- 運算本身做的檢查，例如它建出的 view、結果的 `count_of` 與 `Views.fits`，
- 沿線加總的運算要有某個 `cap` 使 `2 * elements + 4 <= cap <= 2^31`。

## 卷積

| 函式 | 結果持有的值 | 來源 |
|---|---|---|
| `conv2d`、`conv2d_custom` | 輸入 `[C, H, W]` 或 `[N, C, H, W]`、權重 `[O, C, K, K]`、bias `[O]`：每個索引是有序卷積值的 activation。前提：問題有效、機器支援、至少一張影像 | [convolution_call.bend](../convolution/proofs/convolution_call.bend)（`conv2d_contract`）、[convolution_public.bend](../convolution/proofs/convolution_public.bend) |
| `Gradient.input` | [input_gradient_specification.bend](../convolution/input_gradient_specification.bend) 的輸入梯度，並交回權重。前提：`Call.checks` 與至少一張影像 | [input_gradient_public.bend](../convolution/proofs/input_gradient_public.bend)（`tiled_input`） |
| `Gradient.weight`、`Gradient.bias` | [parameter_gradient_specification.bend](../convolution/parameter_gradient_specification.bend) 的梯度，並交回 x 與輸出梯度。前提：`Parameter.valid` 與 `Parameter.positive`，也就是 runtime 做的檢查 | [parameter_gradient_public.bend](../convolution/proofs/parameter_gradient_public.bend)（`tiled_weight`、`tiled_bias`） |

輸入梯度對每個輸入元素，從 +0.0 開始依輸出通道遞增、kernel 列與行遞減加上乘積，只取和它 padding 後的位置對 stride 同餘的 tap，最後加 +0.0。權重或 bias 的梯度把連續輸出列的項從 +0.0 分組相加，再以相鄰配對的樹加總各組與各影像。

卷積或梯度的每個 task 讀一段輸入列的視窗，寫出它的輸出 pieces（[window_windowed.bend](../convolution/proofs/window_windowed.bend)、[input_gradient_windowed.bend](../convolution/proofs/input_gradient_windowed.bend)、[tensor_parameter_gradient_cells.bend](../lib/proofs/tensor_parameter_gradient_cells.bend)）。

## 訓練

| 函式 | 結果持有的值 | 來源 |
|---|---|---|
| `Layer.forward`、`Layer.backward`、`Layer.backward_parameters` | 卷積的 activation、存下的輸入與卷積值，以及 `dz = dy * activation'(z)` 的輸入、權重與 bias 梯度。前提：`conv2d` 或梯度的前提，activation 不是 `Custom` | [layer_public.bend](../convolution/proofs/layer_public.bend) |
| 卷積層的 `Train.forward`、`Train.evaluate`、`Train.backward`、`Train.sgd_step_with` | `conv2d`、`Gradient.*` 與 `Optimizer.sgd_with` 給的值，層照原樣交回、參數在各自的槽。step 需要非空的權重與 bias | [training_convolution_public.bend](../convolution/proofs/training_convolution_public.bend) |
| activation、batch normalization、layer normalization 與平均池化的同樣函式 | 它們呼叫的運算給的值，層照原樣交回 | [tensor_training_layer_values.bend](../lib/proofs/tensor_training_layer_values.bend) |
| `forward_network`、`evaluate_network`、`backward_network`、`sgd_network_with` | 見下 | [training_network_public.bend](../convolution/proofs/training_network_public.bend) |

網路的定理取網路的描述：每層每個參數的值，以及每層對其輸入形狀需要的事實。block 另外要求分支與捷徑給出相同的形狀，非空且放得進機器。於是：

- `forward` 的輸出持有依序套用各層的值，block 是分支與捷徑的和，tape 持有每層存下的東西。
- `evaluate` 在推論模式下給出同樣的值，使用 running 統計量，網路照描述交回。
- 對持有這種紀錄的 tape，`backward` 給出輸入的梯度，網路的每個參數帶有它的梯度。
- `sgd` 以 `Optimizer.sgd_with` 移動每個有梯度的參數，並清空梯度。

## 張量運算

| 運算 | 結果持有的值 | 來源 |
|---|---|---|
| `add`、`sub`、`mul`、`div`、`maximum`、`minimum`，含 broadcasting | 每個索引是 `operation(left, right)`，照這個參數順序 | [tensor_binary_values.bend](../lib/proofs/tensor_binary_values.bend)、[tensor_tiled_map.bend](../lib/proofs/tensor_tiled_map.bend)、[tensor_tiled_broadcast.bend](../lib/proofs/tensor_tiled_broadcast.bend) |
| `add_scalar`、`mul_scalar`、... | `operation(x, scalar)` | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `less`、`less_equal`、`greater`、`greater_equal`、`equal`、`not_equal`、它們的 `_scalar` 形式、`clamp` | 每個索引依比較結果是 1.0 或 0.0，`clamp` 是夾在界限內的值 | [tensor_comparison_values.bend](../lib/proofs/tensor_comparison_values.bend) |
| `sigmoid`、`silu`、`tanh`、`gelu`、`gelu_tanh`、`exp`、`log`、`sqrt`、`apply` | 每個索引是 [math_fp32.bend](../lib/math_fp32.bend) 或 [math_activation.bend](../lib/math_activation.bend) 的純量函式值 | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `activate`、`activate_derivative`、`activate_forward`、`activate_backward` | 每個索引是 activation 或它的導數，backward 是梯度乘上輸入處的導數。activation 不是 `Custom` | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `Optimizer.sgd`、`Optimizer.sgd_with` | 每個索引是 `parameter - applied * rate`，其中 `direction = gradient + parameter * decay`，`applied = velocity * momentum + direction`（第一步是 `direction`）。留下的 velocity 持有 `applied`，只有 momentum 不為零時才保留 | [tensor_optimizer_values.bend](../lib/proofs/tensor_optimizer_values.bend) |
| `BatchNorm.forward`、`BatchNorm.backward`、`BatchNorm.evaluate` | x 在 `[N, C, H, W]`、每通道張量在 `[C]`：`xhat * gamma + beta`，`xhat = (x - mean) * deviation`，`deviation = 1 / sqrt(s / M + epsilon)`，加總沿 `[0, 2, 3]`，以每張影像每個通道的 H × W 平面為列，再合併各影像（和 `F.sum` 沿 `[0, 2, 3]` 的結果可能差最後一位），running 統計量往平均與 `s / (M - 1)` 移動。backward 依 runtime 的 FP32 順序給出 `dbeta`、`dgamma` 與 `dx` | [tensor_batch_norm_public.bend](../lib/proofs/tensor_batch_norm_public.bend) |
| `LayerNorm.forward`、`LayerNorm.backward`、`LayerNorm.evaluate` | x 在 `[..., D]`、gamma 與 beta 在 `[D]`：`xhat * gamma + beta`，平均沿最後一軸。backward 給出 `dx = (dz - mean(dz) - xhat * mean(dz * xhat)) * deviation`，`dz = dy * gamma`，以及 `dy * xhat` 與 `dy` 沿其他軸的和 | [tensor_layer_norm_values.bend](../lib/proofs/tensor_layer_norm_values.bend) |
| `Loss.mse`、`Loss.cross_entropy` | 同形狀的預測與目標：loss、它的梯度與目標。`cross_entropy` 取 logits `[N, K]` | [tensor_loss_values.bend](../lib/proofs/tensor_loss_values.bend)、[tensor_cross_entropy_values.bend](../lib/proofs/tensor_cross_entropy_values.bend) |
| `Train.one_hot` | `[N, K]` 的第 `i` 列第 `j` 行是標籤 `i` 與 `j` 的 `equal` | [tensor_one_hot_values.bend](../lib/proofs/tensor_one_hot_values.bend) |
| `copy`、`reshape` | 輸入的值，reshape 照 row-major 順序 | [tensor_copy_contract.bend](../lib/proofs/tensor_copy_contract.bend)、[tensor_tiled_reshape.bend](../lib/proofs/tensor_tiled_reshape.bend) |
| `softmax` | FP32 的 `exp(x[i] - m) / s`，`m` 是沿軸依索引遞增的 fold，`s` 是沿軸以 [math_sum.bend](../lib/math_sum.bend) 加總 | [tensor_softmax_values.bend](../lib/proofs/tensor_softmax_values.bend) |
| `matmul`、`linear` | 每個矩陣的第 `i` 列第 `j` 行，乘積從 +0.0 依 `k` 遞增相加，再加 +0.0 或 `j` 的 bias | [matmul_call.bend](../convolution/proofs/matmul_call.bend)、[linear_call.bend](../convolution/proofs/linear_call.bend)、[product_values.bend](../convolution/proofs/product_values.bend) |
| `apply_indexed`、`random_uniform`、`random_below`、`arange` | row-major 索引與該處值的函式、[math_random.bend](../lib/math_random.bend) 在 seed、stream 與索引的產生器，以及索引加上起點 | [tensor_tiled_indexed_values.bend](../lib/proofs/tensor_tiled_indexed_values.bend)、[tensor_random_values.bend](../lib/proofs/tensor_random_values.bend) |
| `select` | mask 的大小位元決定 yes 或 no，保留被選字的全部 32 位元 | [tensor_select_public.bend](../lib/proofs/tensor_select_public.bend)、[math_select_values.bend](../lib/proofs/math_select_values.bend) |
| `transpose`、`slice`、`flip`、`broadcast_to` | 輸入在 view 給該索引的位址的值 | [tensor_tiled_view.bend](../lib/proofs/tensor_tiled_view.bend) |
| `upsample_nearest` | 輸入在最後兩軸最近索引的值 | [tensor_tiled_reshape.bend](../lib/proofs/tensor_tiled_reshape.bend) |
| `set_slice` | part 從它沿軸的起點寫入，其餘不變 | [tensor_tiled_set_slice.bend](../lib/proofs/tensor_tiled_set_slice.bend) |
| `pad`、`pad_edge` | 輸入放在前方 padding 之後，其他位置是常數或最近的元素 | [tensor_tiled_pad.bend](../lib/proofs/tensor_tiled_pad.bend) |
| `concat` | 每個 part 在它沿軸的位置 | [tensor_tiled_concat.bend](../lib/proofs/tensor_tiled_concat.bend) |
| `sum_axes`、`mean_axes`、`max_axes`、`min_axes`、`argmax`、`sum`、`mean`、`max`、`min` | 每個保留軸的索引，是列出的軸依 row-major 順序的 fold，加總用 [math_sum.bend](../lib/math_sum.bend)。軸不重複且在形狀內 | [tensor_tiled_reduce.bend](../lib/proofs/tensor_tiled_reduce.bend)、[tensor_reduce_values.bend](../lib/proofs/tensor_reduce_values.bend) |
| `max_pool2d`、`max_pool2d_with` | 每個視窗從 -inf 開始，先列後行依索引遞增的 fold。視窗有效，輸入至少兩軸 | [tensor_tiled_pool.bend](../lib/proofs/tensor_tiled_pool.bend) |
| `depthwise_conv2d` | 每張影像的每個通道 `c`，是 `c` 的 bias 加上通道 `c` 的 tap 乘它的權重，先列後行。結果的 cells 是 usable | [tensor_tiled_depthwise.bend](../lib/proofs/tensor_tiled_depthwise.bend) |

`softmax` 的值就是那串 FP32 運算，使用函式庫的 `exp`。不宣稱任何實數上的正規化性質。

底層：

| 主題 | 證明的內容 | 來源 |
|---|---|---|
| Pieces 與 tasks | 形狀的 pieces 與它們的範圍、決策把它們分成的 tasks，以及每個 piece 恰好寫一次 | [tensor_grid_pieces.bend](../lib/proofs/tensor_grid_pieces.bend)、[tensor_grid_ranges.bend](../lib/proofs/tensor_grid_ranges.bend)、[tensor_task_runs.bend](../lib/proofs/tensor_task_runs.bend)、[tensor_pieces_grouped_with.bend](../lib/proofs/tensor_pieces_grouped_with.bend)、[tensor_placed.bend](../lib/proofs/tensor_placed.bend) |
| Map | pieces 的 map 或 combine 持有函式的值 | [tensor_pieces_map.bend](../lib/proofs/tensor_pieces_map.bend)、[tensor_pieces_combine.bend](../lib/proofs/tensor_pieces_combine.bend) |
| 搬移 | 搬移的每個分量恰好取一次來源 pieces 並寫出目標 pieces，任意步長的 run 讀它指定的來源 | [tensor_moved.bend](../lib/proofs/tensor_moved.bend)、[tensor_components.bend](../lib/proofs/tensor_components.bend)、[tensor_move_readback.bend](../lib/proofs/tensor_move_readback.bend) |
| View、placing 與 relane | view、placing 與 lane 軸搬移的 gather 律 | [tensor_view_gather.bend](../lib/proofs/tensor_view_gather.bend)、[tensor_placing_law.bend](../lib/proofs/tensor_placing_law.bend)、[tensor_relane_gather.bend](../lib/proofs/tensor_relane_gather.bend)、[tensor_tiled_relane.bend](../lib/proofs/tensor_tiled_relane.bend) |
| lane 軸的拆分 | 把 lane 軸拆成兩軸的 reshape 的 gather 律，以及結果的 contract | [tensor_split_law.bend](../lib/proofs/tensor_split_law.bend)、[tensor_tiled_split.bend](../lib/proofs/tensor_tiled_split.bend) |
| 視窗 | stencil 的每個 task 持有它讀的輸入列 | [tensor_halo_windows.bend](../lib/proofs/tensor_halo_windows.bend)、[tensor_window_windowed.bend](../lib/proofs/tensor_window_windowed.bend) |
| 線 | 歸約依同一個順序 fold 每條線，不論線在一個 piece 內或跨 pieces | [tensor_partials_values.bend](../lib/proofs/tensor_partials_values.bend) |
| 界限 | 由程式檢查的 guard 推出：每次讀寫都在範圍內 | [tensor_operation_bounds.bend](../lib/proofs/tensor_operation_bounds.bend) |

## 建立張量

[tensor_entry_values.bend](../lib/proofs/tensor_entry_values.bend) 建出持有給定值的契約張量。

| 函式 | 結果持有的值 |
|---|---|
| `from_list` | 任何在形狀的每個索引都和清單一致的讀取器的值 |
| `from_storage` | 持有一個有大小的模型的 storage，依 flat 索引的值，形狀照給定 |
| `full`、`zeros`、`ones` | 每個索引都是該值（[tensor_full_values.bend](../lib/proofs/tensor_full_values.bend)） |

對這種 storage，`read_storage` 交回 storage 以及該索引的值，超出結尾時是 0.0。

## 讀出張量

[tensor_exit_values.bend](../lib/proofs/tensor_exit_values.bend) 依 flat 索引順序讀出契約張量的讀取器值。

| 函式 | 給出 |
|---|---|
| `into_storage` | 第 `i` 個元素是讀取器在 `i` 的值的 storage |
| `to_list` | 讀取器在 `0` 到 `n - 1` 的值的清單 |
| `item` | 只有一個元素的張量，讀取器在 `0` 的值 |
| `save_raw`、`save_npy` | 寫出那個 storage（`save_npy` 先寫 `.npy` 檔頭），並回傳符合同一個契約的張量 |

## 具名張量的 store

[tensor_model_values.bend](../lib/proofs/tensor_model_values.bend) 把 store 描述成依序的具名張量，每個持有它的值。

| 函式 | 給出 |
|---|---|
| `take` | 該名稱的第一個張量（持有它的值），以及其他項目的 store |
| `keep` | 交回張量，以及前面多一份複本的 store，兩者都持有它的值 |

## 沒有證明的部分

- 成本模型與平行決策。它們只決定 pieces 怎麼分組，不影響結果。
- 實數上的精度。[math_accuracy.py](../benchmarks/checks/math_accuracy.py) 量測它。
- YOLO 範例的運算圖、I/O 與偵測解碼。
- 編譯器與執行期的行為。
- `describe` 的文字，以及讀 metadata 的 `shape`、`check` 與 `workers`。

## 檢視

`python run.py proofs` 也會把宣告與它們的前提列在 `out/results/proof-audit/`，`python run.py dependencies` 列出每個宣告的引用。它們用來瀏覽，判定以 checker 為準。
