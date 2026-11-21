# 更新紀錄

[English](CHANGELOG.md) | [繁體中文](CHANGELOG.zh-TW.md)

本文件記錄 Stelliferous 各版本的重要變更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，版本編號採用[語意化版本](https://semver.org/lang/zh-TW/)規範。

在正式發布 1.0.0 之前，公開 API 仍可能隨次版本更新而調整。

## [0.0.4] - 2026-11-21

Stelliferous 0.0.4 可以完整訓練 ResNet-20、執行 Vision Transformer，速度也大幅提升。本次更新的主要內容包括

- 含殘差區塊的網路可以完整訓練。ResNet-20 在 CIFAR-10 上，從相同的權重與批次出發，達到與 PyTorch 相同的準確率。
- 新增矩陣乘積、layer normalization 與精確的 GELU，可以從 torchvision 的權重執行官方的 ViT-B/16，四個執行緒一次推論 176 毫秒。
- YOLOv5n 推論在一個執行緒為 80 毫秒、四個執行緒為 35 毫秒，0.0.3 為 117 與 48 毫秒。
- 一個訓練 epoch 比 0.0.3 快約七倍，與 PyTorch 的差距從約 25 倍縮小到約 4 倍。

本版本升級至 Bend v2.0.40，在目前的 Bend 上可以正常匯入。已發布的 0.0.3 在 Bend v2.0.38 之後無法匯入。

### 升級注意事項

- **執行期選項要放在 `+RTS` 與 `-RTS` 之間**

  自 Bend v2.0.38 起，編譯後的程式只在 `+RTS ... -RTS` 之間讀取執行期選項。以前 `./program --threads 4` 會使用四個執行緒，現在 `--threads 4` 會交給程式本身，程式則使用所有 CPU。請改寫為 `./program +RTS --threads 4 --gpu off -RTS`。

- **Batch normalization 拒絕長度為零的軸**

  輸入的任何一個軸長度為零時，batch normalization 層的前向傳播、反向傳播與 `evaluate` 都會回傳 `InvalidShape` 的失敗結果。

- **`max_pool2d` 拒絕秩為 0 與 1 的張量**

  0.0.3 會接受這類張量，現在會回傳失敗結果。

- **張量的內部儲存方式已改變**

  `F.Tensor()` 現在以八個一組的方式儲存數值，公開函式的行為不變。先前匯入內部模組 `tensor_band*.bend` 的程式請改用公開的 `F` 函式，這些模組已經移除。

- **`python run.py training` 改為訓練 ResNet-20**

  以前它用一個沒有殘差相加的網路與一個合成 batch 量一個 epoch 的時間。現在它在 CIFAR-10 上訓練 ResNet-20，在不同執行緒數之間逐位元比對，並與 PyTorch 對照，`--epochs E` 訓練 E 個完整的 epoch。原本的 `benchmarks/training/` 已經移除。

### 新增功能

- **網路與殘差區塊**

  `Train.Network` 依序串接層與殘差區塊。`forward_network(x, network)` 回傳輸出與一份紀錄，`backward_network(dy, tape)` 回傳輸入梯度，以及存有每一層梯度的網路，再由 `sgd_network(network, rate)` 套用。`evaluate_network` 以推論模式執行網路。`Train.Residual` 區塊把分支與捷徑的輸出相加，反向傳播時輸入得到兩者梯度的和。

  `save_network(folder, network)` 把參數、執行統計量與速度寫成原始 FP32 檔案，`restore_network(folder, network)` 再讀回來，所以訓練可以從任何一個 epoch 之後繼續。

- **含 momentum 與 weight decay 的 SGD**

  `sgd_step_with(layer, rate, momentum, weight_decay)` 與 `sgd_network_with(network, rate, momentum, weight_decay)` 與 PyTorch 不含 dampening 的 SGD 相同。層會保留速度供下一步使用。

- **Layer normalization**

  `layer_norm_layer(features)`、`layer_norm_layer_with(features, epsilon)` 與 `layer_norm_layer_from(gamma, beta, epsilon)` 沿最後一個軸正規化，並提供前向與反向傳播。

- **矩陣乘積**

  `matmul(left, right)` 計算前置軸相同的矩陣乘積，也可以乘上一個共用的右矩陣。`linear(x, weight, bias)` 以 PyTorch 的 `[outputs, inputs]` 順序讀取權重。每個輸出都從 +0.0 開始，依內側索引的順序相加，結果與 worker 數無關。

- **更多張量函式**

  `gelu` 是精確的 GELU，即 PyTorch 的 `approximate='none'`，另有 `gelu_derivative`。對全部 2³² 種 FP32 輸入，最大誤差為 2.76 ULP，導數為 3.79 ULP。`argmax(tensor, axis)` 回傳第一個最大元素的位置。`depthwise_conv2d` 以每個通道自己的權重做卷積。`Act.gelu()` 讓卷積與 `activation_layer` 也能使用精確的 GELU。

- **亂數與位元組檔案**

  `random_uniform(shape, seed, stream)` 與 `random_below(shape, bound, seed, stream)` 在任何機器、任何 worker 數下都產生相同的值。`load_u8(path, shape)` 以一個位元組讀取一個值，`read_storage(storage, index)` 讀取 storage 中的一個值。

- **ResNet-20 與 ViT-B/16 範例**

  [ResNet-20](examples/resnet20/README.zh-TW.md) 在 CIFAR-10 上訓練，並與使用相同初始權重與批次的 PyTorch 版本對照。`python run.py training` 以 1、2、4、8 個執行緒及中斷後繼續的情況，逐位元檢查兩個短 epoch。

  [ViT-B/16](examples/vit_b16/README.zh-TW.md) 使用官方的 torchvision 權重，並與 PyTorch 比較每個區塊、logits 與前五名的類別。`python run.py vit` 會編譯、執行並比對。

### 修正

- **`load_checkpoint` 可以讀取儲存的 state dict**

  `torch.save(model.state_dict())` 寫出的檔案在項目之外還帶有 `_metadata` 之類的屬性。0.0.3 讀到這種檔案會停在 `BUILD on a value that is not an object`。0.0.4 會讀取所有項目並略過這些屬性。

### 開發工具

- `python run.py dependencies` 列出每個 Bend 宣告與檔案被哪裡使用、沒有用到的 import，以及沒有任何地方用到的宣告。
- `benchmarks/model/profile_timeline.py` 的運算時間軸會列出每個函式庫模組的時間與進入次數。
- `benchmarks/checks/math_accuracy.py` 也會量測 `gelu`。
- YOLOv5n 圖的規劃器報告 `benchmarks/model/report_plans.py` 與 `plan_observation.bend` 已經移除。單一卷積形狀的規劃仍可用 `benchmarks/convolution/report_plans.py` 查看。

### 效能

YOLOv5n 在 640 × 640 輸入影像上的單次推論，於同一台測試機量測。

| 版本 | 1 執行緒 | 4 執行緒 |
| ----- | -------: | ------: |
| 0.0.3 | 117 ms | 48 ms |
| 0.0.4 | 80 ms | 35 ms |

兩個版本在同一台機器上交替量測。

ViT-B/16 在 224 × 224 輸入影像上的單次推論，於同一台機器量測：

| 函式庫 | 1 執行緒 | 4 執行緒 |
| ----- | -------: | ------: |
| Stelliferous 0.0.4 | 360 ms | 176 ms |
| PyTorch 2.14 | 353 ms | 121 ms |

ResNet-20 在 CIFAR-10 上從相同的初始權重與批次訓練 160 個 epoch，於同一台機器以四個執行緒量測。每個 epoch 為 390 步、batch 128，加上 10,000 張影像的測試。

| 函式庫 | 每個 epoch 平均 | 最後一個 epoch 的測試準確率 |
| ----- | -------: | ------: |
| Stelliferous 0.0.4 | 137.5 s | 91.40% |
| PyTorch 2.14 | 34.5 s | 91.34% |

0.0.3 不能訓練 ResNet-20。它發布時在相同形狀但沒有殘差相加的網路上，用一個合成 batch 跑 390 步需要 767 秒。0.0.4 在真正的 ResNet-20 上跑 390 步，不含測試需要 112 秒。

完整結果請見[效能文件](docs/performance.zh-TW.md)。

- **新的 GEMM 核心**

  0.0.3 的 GEMM 每次計算 4 個輸出通道乘 8 個位置，用在卷積的前向傳播與權重梯度。0.0.4 的 GEMM 每次計算 8 × 8 的區塊，卷積的前向傳播、輸入梯度與權重梯度、`matmul` 和 `linear` 都使用它，結果直接以八個一組的方式寫進張量的儲存。

  Stride 1 的 1 × 1 卷積直接讀取輸入，每次計算 16 個輸出通道，不必先重新排列輸入。輸入通道數不是 16 的倍數時，剩下的通道另外加總，不補零，也不多算乘積。

  權重放不進快取區塊時，GEMM 會改變計算順序，讓權重只從記憶體讀一次，而不是每八個位置讀一次。ViT-B/16 的 linear 層受益最多。

- **新的權重梯度計算**

  0.0.3 以 kernel 的每個位置為列計算權重梯度，結果排成 [位置][輸出通道]，最後再轉置成 [輸出通道][位置]。0.0.4 直接讀取張量八個一組的儲存，每次計算 8 個輸入通道乘 8 個輸出通道，結果直接寫成最後的排列，不再轉置。

  工作除了照舊依輸出通道切分，現在也可以依影像切分，所以輸出通道少的層也能用上所有 worker。加總的順序不變，結果與 worker 數無關。Bias 的梯度一次加總八個輸出通道。

- **層與層之間不再轉換格式**

  張量從輸入到輸出都維持同一種儲存方式。逐元素函式、activation、池化與歸約直接在上面計算，層與層之間不再重新排列數值。

- **Batch normalization 少走幾趟資料**

  統計量以兩次歸約求得，正規化在一次平行處理中完成。

- **Batch 在四個 worker 上變快**

  0.0.3 處理通道多的小影像 batch 時，四個 worker 反而比一個慢。現在 batch 128 的 ResNet-20 一個 epoch，一個執行緒 238 秒，四個執行緒 112 秒。

- **工作平均分配**

  一個運算的工作切成平均的份量，每個 worker 每一輪各拿一份，最後一輪不再只等一個 worker。

- **卷積與矩陣乘積減少複製**

  矩陣乘積直接讀取輸入與權重，不必為每一份工作各複製一份。1 × 1 以外的卷積把每個輸入值只轉置一次，再以移動 lane 的方式組出打包的 tile，不再為讀到它的每個 kernel 位置各取一次。

### 形式化證明

- **新增的形式化證明**

  `matmul`、`linear`、`depthwise_conv2d`、layer normalization、`argmax`、亂數、`one_hot`、含 momentum 與 weight decay 的 SGD，以及含殘差區塊的整個網路的前向、反向與更新，都已完成證明。每個輸出元素都等於規格定義的值。詳情請見[證明文件](docs/proofs.zh-TW.md)。

  0.0.3 沒有值定理的既有函式，現在也有了：`apply`、`apply_indexed`、`arange`、`clamp`、`conv2d_custom`、`from_list`、`from_storage`、`zeros`、`ones`、`max_pool2d_with` 與 `pad_edge`。

- **證明跟著更快的程式碼**

  證明涵蓋新的張量儲存方式、工作切分與新的卷積核心。原生推論等於矩陣規格的定理，敘述與條件都不變。`python run.py proofs` 以官方檢查器檢查 569 個模組、20,051 個宣告。

### 已知限制

- 訓練只提供 SGD，也沒有自動微分。網路是固定的層與殘差區塊串接。
- 數值誤差來自量測，尚未有誤差上界的形式化證明。
- 證明由官方檢查器以 `python run.py proofs` 驗證，未使用 `bend --verdict`。
- 只支援 FP32（F32）資料型別。
- 權重多而輸出位置少的卷積，以及寬度小於八個位置的輸出，增加 worker 通常沒有明顯效果。請見[設計文件](docs/design.zh-TW.md#仍有的成本)。
- 推論仍比 PyTorch 慢，訓練約慢 4 倍。請見[效能文件](docs/performance.zh-TW.md)。

## [0.0.3] - 2026-10-21

Stelliferous 0.0.3 新增訓練功能，並支援在 Bend 中直接讀取 PyTorch checkpoint。本次更新的主要內容包括

- 支援卷積、batch normalization、activation 與平均池化層的前向與反向傳播，並提供均方誤差、cross entropy 與 SGD。無論使用多少個 worker，訓練結果皆逐位元一致。
- 支援直接在 Bend 中讀取 PyTorch 的 `.pt` checkpoint。YOLOv5n 範例改為直接從官方的 `yolov5n.pt` 執行。
- `softmax`、`select`、batch normalization、損失函式與訓練層皆已完成形式化證明。

本版本升級至 Bend v2.0.37。

### 升級注意事項

- **`sum_axis`、`mean_axis`、`max_axis` 與 `min_axis` 改為 `sum_axes`、`mean_axes`、`max_axes` 與 `min_axes`**

  新函式改為接受一串軸。請將 `sum_axis(t, a)` 改寫為 `sum_axes(t, [a])`，其他三個函式依此類推。同一個軸列出兩次，或軸超出張量的維度時，會回傳失敗結果。

- **加總的數值結果有所調整**

  `sum`、`mean`、`sum_axes`、`mean_axes`、`softmax` 的分母與均方誤差，已改為以八條通道搭配誤差補償（TwoSum）加總，再依固定順序合併。結果更加精確，在不同機器與不同 worker 數量下仍逐位元一致。

  新版輸出與 0.0.2 可能存在末端位元差異。若先前使用 0.0.2 的輸出作為測試基準，請重新產生參考結果。

- **`gelu_tanh` 的數值結果有所調整**

  `gelu_tanh` 為提升精度而重新改寫。對全部 2³² 種 FP32 輸入，最大誤差降至 1.99 ULP，0.0.2 在輸入介於 -2 至 -0.5 時最大可達 15 ULP。若先前使用 0.0.2 的 `gelu_tanh` 輸出作為測試基準，請重新產生參考結果。

- **`max_pool2d` 在輸入高度或寬度為 0 時回傳失敗結果**

  0.0.2 在這種情況下會回傳全為 -∞ 的結果。輸出本身沒有元素的情況仍維持成功。

- **`upsample_nearest` 收到失敗的輸入時原樣傳回**

  0.0.2 回傳的失敗結果帶著放大後的形狀，現在保留輸入的形狀，失敗原因不變。

- **`load_weights` 讀不到 `names.txt` 時的回報方式**

  0.0.2 會回傳空的權重，之後 `take` 只回報找不到名稱。現在每次呼叫 `take` 都會回報無法讀取 `names.txt`。

### 新增功能

- **訓練層**

  新增 `tensor_training.bend` 模組（以 `Train` 匯入），將每一層表示為一個值。可透過 `conv2d_layer`、`batch_norm_layer`、`activation_layer` 與 `average_pool_layer` 建立各種層。

  `forward(x, layer)` 回傳輸出、反向傳播需要的資料與這一層。`backward(dy, saved, layer)` 回傳輸入梯度，以及帶著參數梯度的這一層，再由 `sgd_step(layer, rate)` 套用梯度。`evaluate(x, layer)` 以推論模式執行一層。

  Batch normalization 的計算方式與 PyTorch `BatchNorm2d` 的訓練模式相同，執行統計量以無偏變異數更新。

  `examples/training/learning.bend` 示範以兩個「卷積、batch normalization、SiLU」區塊訓練 40 步。使用方式請參閱 [API 參考文件](lib/README.zh-TW.md#訓練) 與 [入門指南](docs/getting_started.zh-TW.md)。

- **損失函式**

  新增 `Train.mse(prediction, target)` 與 `Train.cross_entropy(logits, target)`，回傳損失、損失的梯度與 target。另提供 `Train.one_hot(labels, classes)`，將類別編號轉換為 `cross_entropy` 所需的格式。

- **Activation 的梯度**

  新增 `activate_forward`、`activate_backward` 與 `activate_derivative`，以及 `sigmoid_derivative`、`silu_derivative`、`tanh_derivative` 與 `gelu_tanh_derivative`。

- **讀取 PyTorch checkpoint**

  新增 `load_checkpoint(path)`，可讀取 `torch.save` 寫出的 `.pt` 檔案，state dict 的每一項各成為一個張量，名稱與 PyTorch 相同。FP16 與 BFloat16 會精確轉為 FP32，FP64 與整數則捨入為最接近的 FP32。

  官方 `yolov5n-v7.0.pt` 的 349 個張量讀取結果與 `torch.load` 逐位元一致。目前不支援壓縮的項目與 ZIP64 檔案。

  另提供 `weight_names()`，可列出一組權重的所有名稱。YOLOv5n 範例在 Bend 中將官方 checkpoint 的 batch normalization 併入卷積權重。

- **張量運算**

  新增 `sqrt(tensor)` 與 `flip(tensor, axes)`。

### 效能

以下為 YOLOv5n 在相同測試機器上，使用 640 × 640 輸入影像進行單次推論的效能比較。

| 版本    |  1 執行緒 |  4 執行緒 |
| ----- | -----: | -----: |
| 0.0.2 | 115 毫秒 | 44 毫秒 |
| 0.0.3 | 117 毫秒 | 47 毫秒 |

兩個版本在同一台機器上交替量測。推論比 0.0.2 略慢，單執行緒約 2%，四執行緒約 7%。

在同一台機器上以四執行緒訓練一個 ResNet-20 形狀網路的 epoch（390 步、batch 128、32 × 32 影像）：

| 函式庫 | 時間 |
| ----- | -------: |
| Stelliferous 0.0.3 | 767 秒 |
| PyTorch 2.14 | 31.1 秒 |

兩者從相同的權重開始，得到相同的 loss。`python run.py training` 可重現這項比較。

完整測試結果請參閱 [效能測試文件](docs/performance.zh-TW.md)。

- **訓練使用所有 worker**

  多張影像的 batch 可以分給多個 worker 處理，卷積的權重梯度則依輸出通道切分給各 worker。

- **版面運算直接在各 worker 的資料上執行**

  `flip`、`transpose`、`slice`、`broadcast_to`、任意軸的 `concat` 與各種歸約運算，皆直接在各 worker 持有的張量區段上執行，不需先將整個張量收集起來。計算結果不變。

### 形式化證明與數值精度

- **新增形式化證明**

  新增 `softmax`、`select`、activation 的導數、batch normalization、`mse`、`cross_entropy`、平均池化、卷積層的前向與反向傳播以及 `sgd` 的證明。

  這些證明確認每個輸出元素皆等於規格所定義的值。另已證明讀回張量時，取得的值即為其契約所描述的值。詳細內容請參閱 [形式化證明文件](docs/proofs.zh-TW.md)。

- **證明的整理**

  每個運算現在只有一條執行路徑，也只有一份證明。走訪張量各部分的證明由所有運算共用，各運算只證明自己的那一步，重複的引理已刪除。`upsample_nearest` 完全改用共用的 view 讀取證明，`upsample_nearest`、`max_pool2d` 與 `concat` 各自的證明檔由 1.6 MB 縮小至 0.7 MB。

- **數值誤差測試**

  `gelu_tanh` 已完整遍歷所有 2³² 種 FP32 位元組合，並以 MPFR 計算的正確捨入結果作為參考，最大誤差為 1.99 ULP。

### 已知限制

- 訓練目前僅提供 SGD。網路的各層需要自行串接，尚未實作自動微分。
- `one_hot` 目前僅完成 PyTorch 對照測試，尚未提供形式化證明。
- 數值誤差來自實際量測，尚無對應的形式化誤差界限證明。
- 證明由官方檢查器驗證，可使用 `python run.py proofs` 執行，不使用 `bend --verdict`。
- 僅支援 FP32 (F32) 資料型別。
- 對於權重數量較多、輸出位置較少的卷積運算，以及輸出寬度小於 6 的情況，增加 worker 數量通常無法帶來明顯的效能改善。batch size 大於 1 時，通道多的小影像用四個 worker 反而較慢。詳見 [設計文件](docs/design.zh-TW.md)。
- 目前推論效能仍落後於 PyTorch，訓練約慢 25 倍。詳細比較請參閱 [效能測試文件](docs/performance.zh-TW.md)。

### 致謝

感謝 [bend-ml](https://github.com/nuxyel/bend-ml) 的 [@nuxyel](https://github.com/nuxyel) 回報 Bend 編譯器的行為。這些內容已整理進 [工具鏈規則](docs/toolchain_rules.zh-TW.md)。

## [0.0.2] - 2026-10-10

Stelliferous 0.0.2 新增檔案讀寫與模型權重載入功能，同時改善 YOLOv5n 的推論效能。本次更新的主要內容包括

- 支援直接在 Bend 中讀寫 NumPy `.npy` 檔案。
- 支援以 Bend 程式完整定義模型架構，透過名稱取得各層權重。
- 改善 YOLOv5n 推論效能。單執行緒延遲由 153.7 毫秒降至 110 毫秒，四執行緒則由 54.8 毫秒降至 43 毫秒。

自本版本起，專案正式由 **Bend-Conv** 更名為 **Stelliferous**，並升級至 Bend v2.0.36。

### 升級注意事項

- **`conv2d_with` 更名為 `conv2d_custom`**

  原有的呼叫方式與引數均維持不變，只需更新函式名稱。

- **`activate` 不再支援自訂函式**

  將自訂的 activation 函式傳入 `activate`，現在會回傳失敗結果。如需使用自訂 activation，請改用 `apply` 或 `conv2d_custom`。

- **部分數學函式的數值結果有所調整**

  `sigmoid`、`silu`、`tanh` 與 `gelu_tanh` 已改用新的計算演算法，確保在不同機器上產生逐位元一致的結果。

  新版輸出與 0.0.1 可能存在末端位元差異。若先前使用 0.0.1 的輸出作為測試基準，請重新產生參考結果。

### 新增功能

- **檔案讀寫**

  新增 `load_npy(path)` 與 `save_npy(path, tensor)`，支援讀寫 NumPy `.npy` 格式。

  另提供 `load_raw(path, shape)` 與 `save_raw(path, tensor)`，可讀寫不含檔頭的原始 FP32 資料。

- **模型權重管理**

  新增 `load_weights(folder)`，可一次載入指定資料夾中的所有權重。模型執行時，可透過 `take(name)` 依名稱取得所需權重，再使用 `run(model, weights)` 執行完整模型。

  其他相關函式請參閱 [API 參考文件](lib/README.zh-TW.md#模型)。

- **數學函式**

  新增 `exp`、`log` 與 `softmax(tensor, axis)`。

- **比較與條件選擇**

  新增六種逐元素比較運算，包括 `less`、`less_equal`、`greater`、`greater_equal`、`equal` 與 `not_equal`。

  比較結果為由 1.0 與 0.0 組成的張量，分別表示對應位置的條件成立或不成立。六種運算皆提供 `_scalar` 版本，可直接與單一數值比較。

  新增 `select(mask, yes, no)`，依據遮罩從兩個張量中選取對應元素，以及 `clamp(tensor, low, high)`，將張量元素限制在指定範圍內。

- **完整的 YOLOv5n 範例**

  `examples/yolov5/yolov5n.bend` 使用公開 API，以單一 Bend 程式實作完整的 YOLOv5n 網路，支援邊長為 32 倍數的輸入影像。

  另新增 `examples/save_and_load.bend`，示範張量的儲存與載入方式。

### 效能改善

以下為 YOLOv5n 在相同測試機器上，使用 640 × 640 輸入影像進行單次推論的效能比較。

| 版本    |    1 執行緒 |   4 執行緒 |
| ----- | -------: | ------: |
| 0.0.1 | 153.7 毫秒 | 54.8 毫秒 |
| 0.0.2 |   110 毫秒 |   43 毫秒 |

完整測試結果請參閱 [效能測試文件](docs/performance.zh-TW.md)。

本次效能提升來自以下幾項調整。

- **卷積改為分塊執行**

  卷積內部將輸入整理與乘加運算切成固定大小的區塊，各區塊重複使用同一塊工作記憶體，不再為整層配置並初始化一大塊暫存空間。單執行緒與多執行緒採用同一套流程。

- **Activation 改為一次處理八個元素**

  `sigmoid`、`silu`、`tanh` 與 `gelu_tanh` 改以每次八個元素的方式計算，編譯後可使用 SIMD 指令。卷積後接的 activation 與單獨呼叫的 activation 皆適用。

- **不再呼叫系統數學函式庫**

  0.0.1 的 `sigmoid`、`silu`、`tanh` 與 `gelu_tanh` 透過系統提供的 `exp` 與 `tanh` 計算，過程中需轉為雙精度再轉回 FP32。新版只使用 FP32 的加減乘除與 32 位元整數運算，速度更快，結果也不再隨系統函式庫版本而改變。

- **YOLOv5n 的偵測解碼改為張量運算**

  解碼步驟改用函式庫的張量運算與 `sigmoid`，與網路其餘部分共用同一套實作。

- **工作規劃器的成本估計更新**

  規劃器改依分塊後的實際流程估算記憶體用量，並以實際指令數計算各 activation 的成本，使多執行緒的分工決策更貼近實際開銷。

- **模型程式的編譯時間縮短**

  由官方權重產生的 YOLOv5n 程式改為「運算清單加上單一迴圈」的形式。640 × 640 的程式編譯時間由約 6 分鐘縮短至約 14 秒，編譯所需記憶體由 12 GB 以上降至 1 GB 以下。

### 形式化證明與數值精度

- **新增形式化證明**

  新增六種比較運算，以及 `sigmoid`、`silu`、`tanh`、`gelu_tanh`、`exp` 與 `log` 的逐元素語意證明。

  這些證明確認每個輸出元素皆符合對應輸入元素套用指定函式後的結果。詳細內容請參閱 [形式化證明文件](docs/proofs.zh-TW.md)。

- **分塊卷積的證明**

  新的分塊執行流程已完成證明，涵蓋單一工作與平行工作兩種情況。「原生卷積對所有輸入皆符合矩陣規格」的無條件定理維持成立。

- **檔案讀寫的證明**

  已證明數值寫成位元再讀回會得到完全相同的位元，包含 NaN 的內容位元。

- **數值誤差測試**

  針對 `sigmoid`、`silu`、`tanh`、`exp` 與 `log`，已完整遍歷所有 2³² 種 FP32 位元組合，並以 MPFR 計算的正確捨入結果作為參考，評估各函式的數值誤差。

  以下為測得的最大誤差，以 ULP (Unit in the Last Place) 為單位。

  | `sigmoid` | `silu` | `tanh` | `exp` | `log` |
  | --------: | -----: | -----: | ----: | ----: |
  |      1.81 |   1.80 |   1.06 |  0.80 |  0.52 |

  可透過 `benchmarks/checks/math_accuracy.py` 重現上述測試。

### 已知限制

- **`gelu_tanh` 的負數精度仍有問題。** 輸入介於 -2 至 -0.5 時，最大誤差可達 15 ULP。當輸入低於 -2 時，計算結果接近零，目前尚無法保證其可靠性。
- `select`、`softmax` 與其他 activation 函式目前僅完成 NumPy 對照測試，尚未提供形式化證明。
- 上述數值誤差來自實際量測，尚無對應的形式化誤差界限證明。
- **目前無法直接使用 `bend --verdict` 驗證本函式庫。** Bend v2.0.36 會在正式開始驗證前中止執行。現有證明仍由官方檢查器完成驗證，可使用 `python run.py proofs` 執行。
- 目前僅支援推論，尚未實作模型訓練與自動微分。
- 僅支援 FP32 (F32) 資料型別。
- 對於權重數量較多、輸出位置較少的卷積運算，以及 batch size 大於 1 或輸出寬度小於 6 的情況，增加 worker 數量通常無法帶來明顯的效能改善。詳見 [設計文件](docs/design.zh-TW.md)。
- 目前推論效能仍落後於 PyTorch。詳細比較請參閱 [效能測試文件](docs/performance.zh-TW.md)。

## [0.0.1] - 2026-10-05

Stelliferous (原 Bend-Conv) 的首個公開版本，基於官方 Bend v2.0.35 開發。

### 新增功能

- **FP32 張量 API**

  新增 `lib/tensor_f32.bend`，提供基本張量運算與操作介面。張量形狀會在執行期間檢查，錯誤資訊透過回傳結果傳遞。

  主要功能包括

  - **張量建立** — `zeros`、`ones`、`full`、`arange`、`from_list`、`from_storage`。
  - **逐元素運算** — `add`、`sub`、`mul`、`div`、`maximum`、`minimum`，支援 NumPy 廣播規則，另提供對應的 `_scalar` 版本。
  - **Activation 函式** — `relu`、`relu6`、`leaky_relu`、`sigmoid`、`silu`、`tanh`、`hard_sigmoid`、`hard_swish`、`gelu_tanh`、`activate`，以及用於自訂函式的 `apply` 與 `apply_indexed`。
  - **歸約運算 (Reduction)** — `sum`、`mean`、`max`、`min`，以及沿指定軸運算的 `sum_axis`、`mean_axis`、`max_axis`、`min_axis`。
  - **張量形狀與佈局操作** — `reshape`、`transpose`、`slice`、`set_slice`、`concat`、`pad`、`pad_edge`、`broadcast_to`。
  - **卷積與視窗運算** — `conv2d`、`conv2d_with`、`max_pool2d`、`max_pool2d_with`、`upsample_nearest`。
  - **資料存取與狀態檢查** — `to_list`、`item`、`to_list_or`、`item_or`、`check`、`describe`、`into_storage`。

- **具備確定性結果的 FP32 卷積**

  固定 FP32 運算與累加順序，不使用 FMA (Fused Multiply-Add) 或重新結合運算，確保不同 worker 數量下的計算結果逐位元一致。

- **以 Bend 實作的工作規劃器**

  根據張量形狀與 worker 數量，決定每次卷積運算的 task 分配方式。

- **Row band 資料佈局**

  採用 Row band 儲存方式，讓多個 worker 能夠分工處理張量，無須各自複製完整資料。

  此佈局可在卷積、逐元素運算、activation、通道串接 (concat)、上採樣 (upsample) 與池化 (pooling) 等操作之間保留，減少不必要的資料複製。

- **形式化證明**

  使用未經修改的官方 Bend 檢查器驗證相關定理，包括原生卷積對所有輸入皆符合矩陣規格的無條件定理，以及張量運算的數值性質證明。

  詳細內容請參閱 [形式化證明文件](docs/proofs.zh-TW.md)。

- **YOLOv5n 推論範例**

  提供由官方預訓練 YOLOv5n 網路產生的 Bend 程式，並與 PyTorch 進行逐層數值比對。

- **正確性測試**

  建立 79 支使用公開 API 的測試程式，將運算結果與 NumPy 進行逐位元比對。

  卷積測試涵蓋 1、2、4、8 執行緒配置，檢查原生卷積在不同 worker 數量下的正確性。

- **雙語文件**

  提供英文與繁體中文文件。

### 已知限制

- 僅支援模型推論，尚未實作訓練與自動微分。
- 僅支援 FP32 (F32) 資料型別。
- 函式庫尚未提供通用的模型權重載入功能，YOLO 範例使用獨立實作的檔案讀取程式。
- 對於權重數量較多、輸出位置較少的卷積運算，以及 batch size 大於 1 或輸出寬度小於 6 的情況，增加 worker 數量通常無法帶來明顯的效能改善。詳見 [設計文件](docs/design.zh-TW.md)。
- 目前推論效能仍落後於 PyTorch。詳細比較請參閱 [效能測試文件](docs/performance.zh-TW.md)。
- 由於 Bend 尚未支援預設引數、具名引數與函式多載，部分 API 同時提供簡化與完整兩種呼叫形式。
