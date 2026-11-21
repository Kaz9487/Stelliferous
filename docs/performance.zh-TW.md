# 效能

[English](performance.md) | [繁體中文](performance.zh-TW.md)

YOLOv5n v7.0 forward，bus 影像 640 × 640，FP32，batch 一。取中位數（毫秒）。

| 後端 | 1 執行緒 | 4 執行緒 |
|---|---:|---:|
| Stelliferous 0.0.4 | 80 | 35 |
| Stelliferous 0.0.3 | 117 | 48 |
| NumPy / OpenBLAS | 132.5 | 120.8 |
| PyTorch | 48.5 | 14.8 |

2026-11-20 在 Google Cloud 的 `c4d-standard-8` 虛擬機上量測，沒有固定 CPU affinity 與頻率。Stelliferous 的時間以整數毫秒回報。已發布的 0.0.3（搭配它自己的 Bend v2.0.37）與 0.0.4 交替執行四次，每次 11 個 forward，所以 Stelliferous 每列有 44 個樣本，兩版的預測逐位元相同。NumPy 與 PyTorch 來自一次 `run.py backends`，兩次暖機、七個樣本，每個設定都通過與官方參考的逐層、預測與偵測比對。

## 重現

先完成 [YOLO 範例的設定](../examples/yolov5/README.zh-TW.md)：

```sh
python run.py backends      # 這個版本的 Stelliferous、NumPy 與 PyTorch
python run.py benchmark     # 單一卷積形狀
```

樣本與環境寫在 `out/results/backends/`，報告是 `out/results/backend_comparison.md`。Actions 分頁的 Benchmark workflow 會在 GitHub 提供的 runner 上執行同一項比較。其他流程見 [benchmarks](../benchmarks/README.zh-TW.md)。

## 環境

- Google Cloud `c4d-standard-8`，Ubuntu 24.04：AMD EPYC 9B45，8 個虛擬 CPU（4 核心，各 2 執行緒）。
- Bend v2.0.40（官方、未修改），Clang 19.1.1，`-O3 -march=native -ffp-contract=off -std=c11`。
- NumPy 2.4.6 與其內附的 OpenBLAS，PyTorch 2.14.0 與 torchvision 0.29.0（CPU、eager、FP32），inter-op 一個執行緒。

## 計時範圍

Forward 計算：配置、所有權複製、packing、kernel、gather 與偵測解碼。載入、編譯、前處理、診斷輸出與 NMS 不計時。

## 樣本

| 設定 | 中位數（毫秒） | 最小–最大（毫秒） |
|---|---:|---:|
| Stelliferous 0.0.4，1 執行緒 | 80 | 76–86 |
| Stelliferous 0.0.4，4 執行緒 | 35 | 33–41 |
| Stelliferous 0.0.3，1 執行緒 | 117 | 113–125 |
| Stelliferous 0.0.3，4 執行緒 | 48 | 46–57 |
| NumPy，1 執行緒 | 132.5 | 128.9–136.7 |
| NumPy，4 執行緒 | 120.8 | 118.5–122.3 |
| PyTorch，1 執行緒 | 48.5 | 47.6–49.6 |
| PyTorch，4 執行緒 | 14.8 | 14.3–15.1 |

## ViT-B/16

官方 torchvision 的 ViT-B/16 與其 ImageNet 權重，一張 224 × 224 影像，FP32。Stelliferous 0.0.4 與 PyTorch 在同一天、同一台機器上交替執行三次，每次 10 個 forward，PyTorch 先暖機一次。

| 後端 | 1 執行緒 | 4 執行緒 |
|---|---:|---:|
| Stelliferous 0.0.4 | 360（349–380） | 176（171–184） |
| PyTorch | 353（350–356） | 121（120–124） |

```sh
python run.py vit --threads 4
```

設定與和 PyTorch 的比對見 [ViT-B/16 範例](../examples/vit_b16/README.zh-TW.md)。

## 訓練

ResNet-20 在 CIFAR-10 上訓練 160 個 epoch，兩邊從相同的初始權重與 batch 開始。一個 epoch 是 390 步、batch 128，加上 10,000 張測試影像。2026-11-20 在同型機器上以 Stelliferous 0.0.4、四執行緒量測一次。

| 函式庫 | Epoch 平均 | 最後一個 epoch 的測試準確率 | 最佳測試準確率 |
|---|---:|---:|---:|
| Stelliferous 0.0.4 | 137.5 秒 | 91.40% | 91.57% |
| PyTorch 2.14.0 | 34.5 秒 | 91.34% | 91.54% |

0.0.3 不能訓練 ResNet-20。它在 2026-10-21 發布時量的是一個 epoch 的 390 步，不含測試，花了 767 秒。網路是 ResNet-20 的形狀但沒有殘差相加與 projection，用單純的 SGD，重複使用同一個合成 batch。

```sh
python run.py training --threads 4 --epochs 160
```

網路、資料與排程見 [ResNet-20 範例](../examples/resnet20/README.zh-TW.md)。

## 其他卷積形狀

兩種形狀有額外成本：權重多而輸出位置少，以及寬度小於八個位置的輸出。見[設計](design.zh-TW.md#仍有的成本)。
