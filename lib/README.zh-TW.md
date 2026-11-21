# API 參考

[English](README.md) | [繁體中文](README.zh-TW.md)

```python
import stelliferous@0.0.4.0/tensor_f32.bend as F
import stelliferous@0.0.4.0/math_activation.bend as Act
```

`F` 是張量運算，`Train` 是訓練用的層（[訓練](#訓練)）。形狀是 `List<&2,U32>`。軸從零算起。張量是 `F.Tensor()`，只有 F32。值以八個一組儲存，只在進出檔案、list 與 storage 時轉換。完整的程式見[開始使用](../docs/getting_started.zh-TW.md)。

每個函式都適用三條規則：

- **引數會被用掉。**運算用掉它的張量並回傳結果。二元運算重用左邊的儲存。要保留張量請用 `copy`。
- **失敗留在結果上。**之後的運算跳過計算，把第一個問題往下傳。讀值的時候才檢查。
- **順序固定。**每個加總與每次卷積都依明訂的順序相加，所以結果與 worker 數無關。

## 建立

| 函式 | 結果 |
|---|---|
| `zeros(shape)`、`ones(shape)`、`full(shape, value)` | 填滿同一個值的張量 |
| `arange(start, count)` | 一維：`start`、`start + 1`……共 `count` 個 |
| `random_uniform(shape, seed, stream)` | `[0, 1)` 的值，取自 Threefry-2x32 字組的最高 24 位元。元素 `i` 只取決於 seed、stream 與 `i` |
| `random_below(shape, bound, seed, stream)` | 取自同樣字組、小於 `bound` 的整數，每個都是浮點數。`bound` 不超過 2^24 時數值精確 |
| `from_list(shape, values)` | Row-major 的數值。清單長度必須正好等於形狀的元素數，否則失敗 |
| `from_storage(shape, storage)` | 包裝一個來自 `into_storage` 或檔案 I/O 的 buffer |

## 檔案

| 函式 | 結果 |
|---|---|
| `load_raw(path, shape)`、`save_raw(path, tensor)` | Little-endian 的 FP32 數值，row-major 順序。檔案讀不到或與形狀不符時得到失敗的張量 |
| `load_npy(path)`、`save_npy(path, tensor)` | FP32 的 NumPy `.npy` 檔。形狀記在檔案裡 |
| `load_u8(path, shape)` | 每個值一個位元組，row-major 順序，每個都是 0 到 255 的 FP32。檔案讀不到或與形狀不符時，得到失敗的張量 |

## 模型

| 函式 | 結果 |
|---|---|
| `load_weights(folder)`、`save_weights(folder, weights)` | `folder/names.txt` 每行一個名稱。`folder/<name>.npy` 是該名稱的張量。`names.txt` 讀不到時，每次 `take` 都回報原因 |
| `load_checkpoint(path)` | `torch.save` 寫出的 PyTorch `.pt` checkpoint：state dict 的每一項各成一個張量，用它的 PyTorch 名稱。FP16 與 BFloat16 值精確放寬為 FP32，FP64 值與整數捨入為最接近的 FP32，有 stride 的儲存區透過它的 view 讀取。不讀壓縮的項目與 ZIP64 檔案。讀不了的檔案給出的權重，每次 `take` 都回報原因 |
| `take(name)` | `do F.Model<T>:` 區塊中的一步：從權重取出一個張量 |
| `keep(name, tensor)` | 把一份複本放進權重，供之後的 `take` 取用 |
| `weight_names()` | 一步：依順序給出權重裡的名稱，權重不變 |
| `run(model, weights)` | 模型的結果 |
| `run_keeping(model, weights)` | 結果與剩下的權重 |
| `copy_weights(weights)` | 兩份權重 |

## 讀值

| 函式 | 結果 |
|---|---|
| `to_list_or(tensor, fallback)` | Row-major 順序的數值。張量失敗時回傳 `fallback` |
| `item_or(tensor, fallback)` | 單一元素張量的那個值，或 `fallback` |
| `to_list(tensor)`、`item(tensor)` | `Result`：`Done{values}` 或 `Fail{problem}` |
| `check(tensor)` | `Done{tensor}`，或 `Fail{(tensor, problem)}`（狀態已清除） |
| `describe(problem)` | 文字，例如 `add: shapes [2] and [3] do not match` |
| `shape(tensor)` | `(tensor, extents)`。搭配 `then_shape` 使用 |
| `into_storage(tensor)`、`expect(tensor)`、`expect_storage(tensor)` | 以 `Result` 取出 buffer，或在 `IO` 裡取出張量或 buffer，失敗時回報 |
| `read_storage(storage, index)` | `(storage, value)`：row-major 順序第 `index` 個值，超過結尾時是 0.0 |

Problem 會寫出失敗的 `F` 函式名稱，種類有三：形狀不相符、形狀不合法或超過機器容量、對該形狀而言引數不合法。

## 保留與組合

| 函式 | 結果 |
|---|---|
| `copy(tensor)` | `(原本的, 複本)` |
| `then(pair, body)` | 把一對張量交給 `body`：`F.then(F.copy(t), kept => copied => F.add(kept, F.relu(copied)))` |
| `then_shape(pair, body)` | 用於 `shape`：`F.then_shape(F.shape(t), tensor => extents => F.reshape(tensor, extents))` |
| `with_workers(tensor, count)`、`workers(tensor)` | 設定或讀取 worker 數。用 `IO.thread_count()` 讀一次 worker 數。結果保留運算元中最大的 |

Bend 只能解構參數與欄位，不能解構呼叫結果，所以成對的值要經過 `then`。

## 逐元素

| 函式 | 結果 |
|---|---|
| `add`、`sub`、`mul`、`div`、`maximum`、`minimum` `(left, right)` | 每個元素 `operation(left, right)`，支援 NumPy 廣播 |
| `add_scalar`、`sub_scalar`、`mul_scalar`、`div_scalar` `(tensor, value)` | `operation(element, value)` |
| `less`、`less_equal`、`greater`、`greater_equal` `(left, right)` | 逐元素大小比較，回傳 1 或 0。任一方 NaN 為 0，沿用二元廣播 |
| `equal`、`not_equal` `(left, right)` | 逐元素相等／不等比較，正負零相等，NaN 不等於任何值 |
| `select(mask, yes, no)` | 遮罩不為零（含 NaN）時取 yes，否則取 no，保留選中值的位元並支援廣播 |
| `less_scalar(tensor, value)` | 元素小於 value 時為 1，否則為 0。NaN 為 0 |
| `less_equal_scalar(tensor, value)` | 元素小於等於 value 時為 1，否則為 0。NaN 為 0 |
| `greater_scalar(tensor, value)` | 元素大於 value 時為 1，否則為 0。NaN 為 0 |
| `greater_equal_scalar(tensor, value)` | 元素大於等於 value 時為 1，否則為 0。NaN 為 0 |
| `equal_scalar(tensor, value)` | 元素等於 value 時為 1，否則為 0。NaN 為 0，正負零相等 |
| `not_equal_scalar(tensor, value)` | 元素不等於 value 或任一方為 NaN 時為 1，否則為 0 |
| `clamp(tensor, low, high)` | 先與 low 取 maximum，再與 high 取 minimum，沿用其 NaN 與正負零行為 |
| `exp(tensor)` | 逐元素自然指數，下溢逐漸進入次正規數 |
| `log(tensor)` | 逐元素自然對數。零回傳 −∞，負數回傳 NaN |
| `sqrt(tensor)` | 逐元素平方根，捨入到最近值。負數回傳 NaN |
| `broadcast_to(tensor, shape)` | 把張量重複成 `shape` |

廣播從最後一軸對齊。缺少的軸視為 1。每一軸的長度要相等，或其中一個是 1。`maximum` 與 `minimum` 對 NaN 與正負零依循 Base，順序為左、右。除數為零得到無限大或 NaN，不算失敗。

## Activation

| 函式 | 說明 |
|---|---|
| `relu`、`relu6`、`sigmoid`、`silu`、`tanh`、`hard_sigmoid`、`hard_swish`、`gelu`、`gelu_tanh` `(tensor)` | 與 PyTorch 相同。`gelu` 是 `approximate='none'`，`gelu_tanh` 是 `approximate='tanh'` |
| `leaky_relu(tensor, slope)` | |
| `activate(tensor, activation)` | 使用 `Act` 的值：`Act.identity()`、`Act.relu()`、`Act.leaky_relu(slope)` 等 |
| `activate_derivative(tensor, activation)` | 激活函式在每個元素的導數。`Custom` 會失敗 |
| `activate_forward(tensor, activation)` | `(activate(tensor, activation), tensor)`：輸出旁邊放著 `activate_backward` 要讀的輸入 |
| `activate_backward(gradient, input, activation)` | `gradient * activate_derivative(input, activation)`。兩個 tensor 形狀相同，否則呼叫失敗 |
| `sigmoid_derivative`、`silu_derivative`、`tanh_derivative`、`gelu_derivative`、`gelu_tanh_derivative` `(tensor)` | |
| `apply(~operation, operations, tensor)` | 自訂的 `F32 -> F32` 函式 |
| `apply_indexed(~operation, operations, tensor)` | 自訂的 `U32 -> F32 -> F32` 函式，另外收到 row-major 索引 |

自訂函式是一個具名的 `def`（或 lambda），以 `~` 當模板傳入，並附上每個元素的純量運算數：每個算術、比較或選擇算一個，`F32.exp` 約 37，`F32.tanh` 約 82。不確定時用 `Act.unknown_operations()`。這個數字只幫助規劃器，結果不受它影響。

```python
def clipped(value: F32) -> F32:
  F32.min(Act.relu_value(value),6.0)

# F.apply(~clipped,4.0,tensor)
# F.conv2d_custom(~clipped,4.0,input,weights,bias,1,1)
```

## Reduction

| 函式 | 結果 |
|---|---|
| `sum`、`mean`、`max`、`min` `(tensor)` | 對所有元素依 row-major 順序計算，得到 rank-0 張量 |
| `sum_axes`、`mean_axes`、`max_axes`、`min_axes` `(tensor, axes)` | 列出的軸被移除 |
| `softmax(tensor, axis)` | 沿指定軸減去最大值後，以指數總和正規化 |
| `argmax(tensor, axis)` | 沿軸第一個最大元素的位置，以浮點數表示。NaN 算最大。該軸被移除 |

每個結果依列優先順序取列出各軸的值，與軸的列出順序無關。同一軸列出兩次或超出範圍時結果失敗。`max` 與 `min` 依索引遞增取元素，從 `-inf` 或 `+inf` 開始。加總把第 `i` 個元素放進八條 TwoSum lane 的第 `i % 8` 條，再依 [math_sum.bend](math_sum.bend) 的固定樹合併各 lane。`softmax` 也這樣加總指數。mean 是總和除以個數。總和的最後幾個位元可能與 NumPy 的 `sum` 不同。空張量的 `mean` 是 NaN。

## 佈局

| 函式 | 結果 |
|---|---|
| `reshape(tensor, shape)` | 元素相同、形狀不同。不複製 |
| `transpose(tensor, order)` | 依指定順序排列各軸 |
| `flip(tensor, axes)` | 沿列出的每個軸反轉順序 |
| `slice(tensor, axis, start, count, step)` | 沿 `axis` 從 `start` 起每隔 `step` 取 `count` 個。`step` 為正 |
| `set_slice(target, axis, start, part)` | 把 `part` 沿 `axis` 從 `start` 寫進 `target` |
| `concat(parts, axis)` | 沿 `axis` 接起來：`F.concat(F.cons(a, F.cons(b, F.single(c))), 1)` |
| `pad(tensor, before, after, value)` | 第 `i` 軸前面加 `before[i]` 個、後面加 `after[i]` 個，值為 `value` |
| `pad_edge(tensor, before, after)` | 同上，但重複最近的元素 |

## 矩陣

| 函式 | 結果 |
|---|---|
| `matmul(left, right)` | `left` 是 `[..., rows, inner]`，`right` 是前面軸相同的 `[..., inner, outputs]`，或 `[inner, outputs]`。每個輸出從 +0.0 開始，依 `k` 遞增加上 `left[i, k] * right[k, j]`，與卷積的加法相同。有長度為零的軸時得到失敗的結果 |
| `linear(x, weight, bias)` | `x` 是 `[..., rows, inputs]`，`weight` 是 `[outputs, inputs]`，`bias` 是 `[outputs]`，和 PyTorch 的存放方式相同。每個輸出從 +0.0 開始，依 `k` 遞增加上 `x[i, k] * weight[j, k]`，再加偏置。有長度為零的軸時得到失敗的結果。 |

## 視窗

張量是 `[通道, 高, 寬]` 或 `[張數, 通道, 高, 寬]`。

| 函式 | 結果 |
|---|---|
| `conv2d(input, weights, bias, stride, padding, activation)` | 權重是 `[輸出通道, 輸入通道, 核高, 核寬]`，bias 是 `[輸出通道]`。兩個方向用同樣的 stride 與 padding。activation 套用在輸出上 |
| `conv2d_custom(~activation, operations, input, weights, bias, stride, padding)` | 使用自訂的 activation 函式 |
| `depthwise_conv2d(input, weights, bias, stride, padding, activation)` | 權重是 `[通道, 1, 核, 核]`，bias 是 `[通道]`。每個通道與自己的權重和 bias 做卷積。activation 不能是 `Custom` |
| `max_pool2d(tensor, kernel)` | `kernel` 乘 `kernel` 的不重疊視窗（stride 等於 `kernel`，padding 為 0） |
| `max_pool2d_with(tensor, kernel, stride, padding)` | 完整形式。padding 視為 `-inf` |
| `upsample_nearest(tensor, scale)` | 每一列與每一行重複 `scale` 次 |

卷積的每個輸出從 +0.0 開始，依輸入通道、核的列、核的行的順序加上乘積，最後加 bias。它怎麼執行與規劃見[設計](../docs/design.zh-TW.md)。證明了什麼見[證明](../docs/proofs.zh-TW.md)。

## 訓練

```python
import stelliferous@0.0.4.0/tensor_training.bend as Train
```

一層是 `Train.Layer` 的值。每個動詞都收下這一層，再把它還回來。backward 把梯度存在層裡，`sgd_step` 再套用。`Train.Network` 依序把層與殘差塊跑過同樣的動詞。[learning.bend](../examples/training/learning.bend) 訓練兩個 Conv-BatchNorm-SiLU 區塊，[ResNet-20](../examples/resnet20/README.zh-TW.md) 在 CIFAR-10 上訓練。

```python
# A residual block: branch and shortcut both read the input, their sum goes on to rest.
# An empty shortcut (Train.End{}) passes the input itself.
block = {Train.Residual{branch,Train.End{},Train.Then{Train.activation_layer(Act.relu()),rest}} : Train.Network}
```

| 函式 | 結果 |
|---|---|
| `conv2d_layer(weights, bias, stride, padding)` | 一層卷積，等同不帶 activation 的 `F.conv2d` |
| `batch_norm_layer(channels)` | 對 `[N,C,H,W]` 做 batch normalization：gamma 1、beta 0、running mean 0、running variance 1、epsilon 1e-5、momentum 0.1。形狀有長度為零的軸時結果為失敗 |
| `batch_norm_layer_with(channels, epsilon, momentum)` | 同上，換成其他 epsilon 與 momentum |
| `batch_norm_layer_from(gamma, beta, running_mean, running_variance, epsilon, momentum)` | 同上，使用給定的參數與 running statistics |
| `layer_norm_layer(features)` | 對最後一軸做 layer normalization：gamma 1、beta 0，都是 `[features]`，epsilon 1e-5 |
| `layer_norm_layer_with(features, epsilon)` | 同上，換一個 epsilon |
| `layer_norm_layer_from(gamma, beta, epsilon)` | 同上，使用給定的參數 |
| `activation_layer(activation)` | `Act` 的一個 activation |
| `average_pool_layer()` | 每個 channel 的平均：`[N,C,H,W]` 變成 `[N,C,1,1]` |
| `forward(x, layer)` | `(y, (saved, layer))`。batch normalization 使用這批資料的統計量，並更新 running statistics |
| `backward(dy, saved, layer)` | `(dx, layer)`，層裡帶著參數梯度。`saved` 來自別種層時，`dx` 為失敗 |
| `evaluate(x, layer)` | 推論用的 `(y, layer)`。batch normalization 使用 running statistics |
| `sgd_step(layer, rate)` | 有梯度的參數變成 `parameter - gradient * rate`，梯度清空 |
| `sgd_step_with(layer, rate, momentum, weight_decay)` | 與 PyTorch 不含 dampening 的 SGD 相同。層保留 velocity 供下一步使用 |
| `forward_network(x, network)` | `(y, tape)`，tape 是 `backward_network` 讀的內容 |
| `backward_network(dy, tape)` | `(dx, network)`，每一層帶著參數梯度。殘差塊的輸入梯度是分支與 shortcut 梯度的和 |
| `evaluate_network(x, network)` | 推論用的 `(y, network)` |
| `sgd_network(network, rate)`、`sgd_network_with(network, rate, momentum, weight_decay)` | 依序對每一層做一步 |
| `save_network(folder, network)` | 每一層的參數、running statistics 與 velocity，以 raw FP32 檔寫進 `folder`。不存梯度 |
| `restore_network(folder, network)` | 讀回 `save_network` 寫的內容，形狀與 workers 照 `network`，兩者的層相同 |
| `mse(prediction, target)` | `(loss, (gradient, target))`：均方誤差與它對 `prediction` 的梯度 |
| `cross_entropy(logits, target)` | `(loss, (gradient, target))`，logits 為 `[N,K]`，target 為 `[N,K]`，每一列是各類別的機率：平均 cross entropy 與它對 `logits` 的梯度 |
| `one_hot(labels, classes)` | 浮點數的類別編號 `[N]` 變成 `[N,classes]` 的列，該類別為 1.0，其餘為 0.0 |

## 函式庫內部

應用程式不需要這些。列出來是給讀原始碼的人。

| 模組 | 角色 |
|---|---|
| [tensor_tiled.bend](tensor_tiled.bend)、`tensor_tiled_*.bend`、[tensor_select.bend](tensor_select.bend)、[tensor_matrix_products.bend](tensor_matrix_products.bend) | `F` 張量的形式：一個軸以八個為一組放進 `Tile8` cells 的 lane，以及 cells 上的運算 |
| [tensor.bend](tensor.bend)、[tensor_layout.bend](tensor_layout.bend)、[tensor_window.bend](tensor_window.bend)、[tensor_convolution.bend](tensor_convolution.bend) | 存放 cells 的泛型張量：任意元素型別、以標準分塊儲存 |
| [tensor_pieces.bend](tensor_pieces.bend)、[tensor_grid.bend](tensor_grid.bend)、[tensor_tasks.bend](tensor_tasks.bend)、[tensor_move.bend](tensor_move.bend)、[tensor_halo.bend](tensor_halo.bend) | 形狀的 pieces、運算的 task、版面之間的搬移，以及 task 上下鄰列 |
| [tensor_operations.bend](tensor_operations.bend) | 在 view 上有檢查的複製、合併、sweep、pool 與歸約。`Accepted` 與 `Rejected` 都交回所有 owner |
| [tensor_view.bend](tensor_view.bend)、[tensor_shape.bend](tensor_shape.bend)、[tensor_cursor.bend](tensor_cursor.bend)、[tensor_access.bend](tensor_access.bend) | View（軸長度、stride、offset）、有檢查的元素數、strided 游標 |
| `kernel_*.bend` | 卷積核心：權重、packing、GEMM、gather、規劃、成本、梯度 |
| [storage_buffer.bend](storage_buffer.bend)、[storage_allocation.bend](storage_allocation.bend)、`storage_*.bend` | `Buffer`：owned array，帶邏輯長度與儲存證書 |
| [storage_bytes.bend](storage_bytes.bend)、[storage_element_format.bend](storage_element_format.bend) | 從檔案讀出的位元組，以及每種元素儲存格式的 FP32 值 |
| `checkpoint_*.bend`、[tensor_checkpoint.bend](tensor_checkpoint.bend) | `.pt` 檔：它的 ZIP 項目、pickle 程式，以及它們描述的張量 |
| [tensor_file.bend](tensor_file.bend)、[tensor_model.bend](tensor_model.bend) | 張量檔案，以及模型取用的具名權重 |
| [tensor_layer.bend](tensor_layer.bend)、[tensor_batch_norm.bend](tensor_batch_norm.bend)、[tensor_batch_norm_cells.bend](tensor_batch_norm_cells.bend)、[tensor_layer_norm.bend](tensor_layer_norm.bend)、[tensor_loss.bend](tensor_loss.bend)、[tensor_optimizer.bend](tensor_optimizer.bend)、`tensor_*_gradient.bend` | `Train` 背後的層、loss、更新與梯度 |
| `traversal_*.bend` | 迴圈：copy、update、fold、indexed output、切分樹 |
| [math_activation.bend](math_activation.bend)、[math_choice.bend](math_choice.bend)、[math_fp32.bend](math_fp32.bend)、[math_sequence.bend](math_sequence.bend)、[math_sum.bend](math_sum.bend)、[math_largest.bend](math_largest.bend)、[math_random.bend](math_random.bend)、[ordered_matrix.bend](ordered_matrix.bend) | Activation 的值與純量函式。依旗標二選一，逐型別編譯。八條 lane 的加總。證明所用的有序序列與矩陣乘積 |
| [machine_limits.bend](machine_limits.bend)、[kernel_cost.bend](kernel_cost.bend) 與它們的 `default_*` 檔 | 證明使用的機器界限。規劃器的成本參數 |
| [proofs/](proofs/) | 函式庫的證明 |
