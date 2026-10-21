# 訓練一個 epoch

[English](README.md) | [繁體中文](README.zh-TW.md)

在同一台機器、同樣的執行緒數下，量 Stelliferous 與 PyTorch 各訓練一個 epoch 的時間。

```sh
python run.py training --threads 4
```

兩行結果寫到 `out/results/training/epoch_4.txt`。公布的數字在[效能](../../docs/performance.zh-TW.md#訓練)。

## 方法

- 網路是 CIFAR-10 用的 ResNet-20 形狀，不含殘差相加：19 個 3x3 卷積、batch normalization 與 ReLU 的區塊（16、32、64 通道，第二與第三階段開頭 stride 2），接平均池化與到 10 類的 1x1 卷積。
- 390 步、batch 128，共 49,920 張 3 × 32 × 32 的影像。重複使用同一個合成 batch，所以不計讀資料的時間。
- 每一步是 forward、cross entropy、backward 與 rate 0.1 的 SGD。權重初值都是 0.01，偏差都是 0。
- 計時涵蓋整個 epoch。兩個程式都印出第一步與最後一步的 loss。

## 檔案

| 檔案 | 用途 |
|---|---|
| [epoch.bend](epoch.bend) | 透過公開的 `Train` API 跑一個 epoch |
| [epoch_torch.py](epoch_torch.py) | 同樣的網路、資料與步驟，用 PyTorch |
| [run_epoch.sh](run_epoch.sh) | 編譯 Bend 程式並執行兩者 |
