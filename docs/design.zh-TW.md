# 設計

[English](design.md) | [繁體中文](design.zh-TW.md)

## 分層

| 層 | 負責 | 主要檔案 |
|---|---|---|
| 公開 F32 API | 程式呼叫的函式 | [tensor_f32.bend](../lib/tensor_f32.bend)、[math_activation.bend](../lib/math_activation.bend)、[tensor_training.bend](../lib/tensor_training.bend) |
| 訓練 | 層、loss、SGD，以及卷積、batch normalization 與 activation 的梯度 | [tensor_layer.bend](../lib/tensor_layer.bend)、[tensor_batch_norm.bend](../lib/tensor_batch_norm.bend)、[tensor_loss.bend](../lib/tensor_loss.bend)、[tensor_optimizer.bend](../lib/tensor_optimizer.bend)、`tensor_*_gradient.bend` |
| 分塊 cells | 每個 `F` 張量的形式，以及 cells 上的運算 | [tensor_tiled.bend](../lib/tensor_tiled.bend)、`tensor_tiled_*.bend`、[tensor_select.bend](../lib/tensor_select.bend)、[tensor_matrix_products.bend](../lib/tensor_matrix_products.bend)、[tensor_batch_norm_cells.bend](../lib/tensor_batch_norm_cells.bend) |
| 泛用張量 | 任意元素型別的 extents、狀態、worker 數與標準分塊，運算的 task、搬移、view、window、檔案、具名權重、PyTorch checkpoint | [tensor.bend](../lib/tensor.bend)、[tensor_pieces.bend](../lib/tensor_pieces.bend)、[tensor_grid.bend](../lib/tensor_grid.bend)、[tensor_tasks.bend](../lib/tensor_tasks.bend)、[tensor_move.bend](../lib/tensor_move.bend)、[tensor_halo.bend](../lib/tensor_halo.bend)、[tensor_layout.bend](../lib/tensor_layout.bend)、[tensor_window.bend](../lib/tensor_window.bend)、[tensor_convolution.bend](../lib/tensor_convolution.bend)、[tensor_file.bend](../lib/tensor_file.bend)、[tensor_model.bend](../lib/tensor_model.bend)、[tensor_checkpoint.bend](../lib/tensor_checkpoint.bend)、`checkpoint_*.bend` |
| 檢查過的運算 | 在 view 上的複製、合併、sweep、pool 與歸約，附 guard | [tensor_operations.bend](../lib/tensor_operations.bend)、[tensor_view.bend](../lib/tensor_view.bend)、[tensor_cursor.bend](../lib/tensor_cursor.bend)、[tensor_access.bend](../lib/tensor_access.bend) |
| Kernel | 權重 tile、packing、GEMM 區塊、gather、規劃、depthwise tap、輸入與權重梯度 | `kernel_*.bend` |
| 儲存與遍歷 | 帶 certificate 的 owned buffer，以及其他部分共用的迴圈 | `storage_*.bend`、`traversal_*.bend` |
| 卷積套件 | 凍結規格、參考實作、已證明的 Array 引擎 | [convolution/](../convolution/README.zh-TW.md) |

`lib/` 不依賴 `convolution/`、範例或 benchmark。

## 張量

張量擁有自己的 cells、extents、狀態與 worker 數。

- **運算會消耗參數。** `F.copy` 可保留張量。
- **失敗跟著值走。** 失敗之後，後續運算跳過工作，把第一個問題傳下去。程式在讀取結果時看到它。
- **形狀在執行期檢查。** Extents 是值，不是型別。

### Cells

`F` 張量把一個軸（lane 軸）以八個為一組分塊：該軸一組的八個值是一個 `Tile8` 的 lane，超出真實長度的 lane 是 +0。其他軸維持 row-major。新張量的 lane 軸在 `[N,]C,H,W` 是通道軸，其餘是最後一軸。秩 0 的張量是一個 cell 的一個 lane。

Cells 依一個標準順序排列：其他軸逐列，lane 軸的組在最內層（權重在最外層）。它們存成 pieces，也就是這個順序裡約 4096 個 cells 的範圍，邊界只由形狀決定，所以同形狀的兩個張量 pieces 相同。

值只在邊界以 F32 進出：`from_list`、`from_storage` 與檔案讀入把值分塊，`to_list`、`item`、`into_storage` 與存檔再讀回 F32。中間的每個運算都在 cells 上執行。

- 運算在 lane 所在的軸上執行。需要 lane 在別的軸時先搬移：每個結果 cell 從持有它那八個值的八個 cell 讀出 lane。
- 保留 lane 軸的 view 整個 cell 地 gather。沒有保留 lane 軸的 view 搬移值。
- lane 軸與它的長度不變時，`reshape` 只改 cells 的標示，否則搬移值。

### Strided 存取

Cursor 走訪 view 時不做除法。Strided 複製把最內層的軸當成一般迴圈，每一段只推進 cursor 一次。`set_slice` 與 `pad` 經由結果的 view 寫入，view 保留 lane 時在 cells 上寫。

### 熱迴圈裡的函式

逐元素使用的函式以 template 參數傳入（`~F32.add`），不存進 record。編譯器因此能特化迴圈。存在 record 裡的函式每個元素都要經過 closure 呼叫，實測約慢六倍。

## 卷積

每個輸出從 +0.0 開始，依輸入通道、kernel 列、kernel 欄的順序加上乘積，最後加偏差。Padding 的取樣以零參與。沒有 fused multiply-add，也不重新結合，所以結果在任何 worker 數與任何規劃下位元都相同。

1. **權重 tile：** 歸約的每個座標把八個輸出通道的權重放在一個 tile。
2. **Pack：** 把一個 task 讀的輸入 cells（lane 是通道）依歸約順序打包成每 tile 八個輸出位置。
3. **GEMM：** 每個區塊最多八個輸出通道乘八個位置，各自從 +0 依序累加。
4. **寫回：** task 的輸出 cells 每 tile 八個通道，接著趁它們還在快取時一次套用 activation。

一個 task 持有相鄰的輸出 pieces。它讀自己那些列的輸入 pieces，上下鄰列由一次走訪各 task 帶給它。每個 task 複製權重 tile 與偏差。

### Task

每個運算以同一個決策把 pieces 分成 task（[tensor_tasks.bend](../lib/tensor_tasks.bend)）。序列計畫是一個 task。平行計畫把相鄰的 pieces 合到一個 worker 的份量，若合到其中較小的 2 的次方能讓最後一個 wave 早點結束就合到那裡，估計比序列工作小時才執行：兩輪喚醒、最大 task 的 waves、每個 task 的分派，以及第一個之後每個 task 配置或複製的量。一個 cell 計算兩種數量：row 是八個 lane 的向量指令，tile 的讀寫也算在內，step 是純量指令。搬移運算的每個目標 cell 另外付一次經過 view 的搬移，區塊複製則按位元組計價。單價在 [default_kernel_cost.bend](../lib/default_kernel_cost.bend)：

| 參數 | 值 | 意義 |
|---|---:|---|
| `row_ns` | 0.113 | 一個 row：GEMM 區塊裡八個 lane 的一次乘加 |
| `step_ns` | 0.122 | 迴圈的一個純量 step |
| `byte_ns` | 0.036 | 區塊複製寫入的一個位元組 |
| `move_ns` | 5.9 | 經過 view 搬移一個 cell |
| `wake_round_ns` | 20000 | 喚醒 worker 的一輪 |
| `dispatch_ns` | 120 | 多分派一個 task |
| `grain_ns` | 250000 | Array 引擎一個 leaf 最少的預估工作量 |

一個 worker 時從不平行。[planner_prices.py](../scripts/toolchain_rules/planner_prices.py) 在參考機器的單一執行緒上量前四個單價，[wake_rounds.py](../scripts/toolchain_rules/wake_rounds.py) 檢查喚醒的一輪。估計錯了只會花時間：結果在每種分組下都已證明相同。

### 仍有的成本

- 卷積的每個 task 都複製權重 tile。權重多而輸出位置少的層，這個成本最明顯。
- 輸出寬度小於八個位置時，一個 tile 跨好幾個輸出列，所以 packer 對它跨過的每一列各讀取、位移並合併一次。

## Bend 沒有提供的

- 沒有預設參數、具名參數或 overloading。短版與完整版用兩個名字（`max_pool2d` / `max_pool2d_with`）。
- `match` 只能作用在參數上，不能作用在呼叫結果上。函式庫提供 `to_list_or` 與 `item_or` 來讀取 `Result`。
- 沒有 fused multiply-add，也沒有 SIMD 型別。一個 tile 是八個 F32 值的 record，由 C 編譯器向量化。
- `Array.new` 會初始化每個元素。
- 共享唯讀儲存需要 `@unsafe`，本函式庫不使用。分割時複製兩半都要讀的資料。
- 重複使用 `List<Tensor>` 會讓編譯器把每個 constructor 標成共享，拖慢無關的程式碼。所以 `concat` 接受 `Parts`（`F.cons`、`F.single`）。

[工具鏈規則](toolchain_rules.zh-TW.md) 詳列 checker、編譯器與執行期的規則。
