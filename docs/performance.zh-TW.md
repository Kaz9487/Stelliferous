# 效能

[English](performance.md) | [繁體中文](performance.zh-TW.md)

YOLOv5n v7.0 forward，bus 影像 640 × 640，FP32，batch 一。每個設定兩次暖機、七個樣本，取中位數（毫秒）。

| 後端 | 1 執行緒 | 4 執行緒 |
|---|---:|---:|
| Stelliferous 0.0.3 | 117 | 47 |
| Stelliferous 0.0.2 | 115 | 44 |
| Stelliferous 0.0.1 | 153.7 | 54.8 |
| NumPy / OpenBLAS | 133.7 | 116.3 |
| PyTorch | 51.1 | 15.2 |

2026-10-21 在 Google Cloud 的 `c4d-standard-8` 虛擬機上量測，沒有固定 CPU affinity 與頻率。每個設定都通過與官方參考的逐層、預測與偵測比對。Stelliferous 的時間以整數毫秒回報。比較共跑四次，已發布的 0.0.2（搭配它自己的 Bend v2.0.36）與 0.0.3 交替執行，所以 Stelliferous 每列有 14 個樣本，NumPy 與 PyTorch 各有 28 個。Stelliferous 0.0.1 是已發布的那一版搭配它自己的 Bend v2.0.35，2026-10-10 在同型機器上以相同方式量測。

## 重現

先完成 [YOLO 範例的設定](../examples/yolov5/README.zh-TW.md)：

```sh
python run.py backends      # 上面的表
python run.py benchmark     # 單一卷積形狀
```

樣本與環境寫在 `out/results/backends/`，報告是 `out/results/backend_comparison.md`。Actions 分頁的 Benchmark workflow 會在 GitHub 提供的 runner 上執行同一項比較。其他流程見 [benchmarks](../benchmarks/README.zh-TW.md)。

## 環境

- Google Cloud `c4d-standard-8`，Ubuntu 24.04：AMD EPYC 9B45，8 個虛擬 CPU（4 核心，各 2 執行緒）。
- Bend v2.0.37（官方、未修改），Clang 19.1.1，`-O3 -march=native -ffp-contract=off -std=c11`。
- NumPy 2.4.6 與其內附的 OpenBLAS，PyTorch 2.14.0（CPU、eager、FP32），inter-op 一個執行緒。

## 計時範圍

Forward 計算：配置、所有權複製、packing、kernel、gather 與偵測解碼。載入、編譯、前處理、診斷輸出與 NMS 不計時。

## 樣本

| 設定 | 中位數（毫秒） | 最小–最大（毫秒） |
|---|---:|---:|
| Stelliferous 0.0.3，1 執行緒 | 117 | 115–121 |
| Stelliferous 0.0.3，4 執行緒 | 47 | 46–50 |
| Stelliferous 0.0.2，1 執行緒 | 115 | 112–116 |
| Stelliferous 0.0.2，4 執行緒 | 44 | 44–47 |
| Stelliferous 0.0.1，1 執行緒 | 153.7 | 153.4–154.4 |
| Stelliferous 0.0.1，4 執行緒 | 54.8 | 54.6–56.5 |
| NumPy，1 執行緒 | 133.7 | 129.9–141.4 |
| NumPy，4 執行緒 | 116.3 | 113.9–122.1 |
| PyTorch，1 執行緒 | 51.1 | 49.0–52.4 |
| PyTorch，4 執行緒 | 15.2 | 14.7–16.5 |

## 訓練

ResNet-20 形狀的網路訓練一個 epoch：390 步、batch 128，每一步是 forward、cross entropy、backward 與 SGD。2026-10-21 在同型機器上以四執行緒量測一次。

| 函式庫 | Epoch | 最後一步之後的 loss |
|---|---:|---:|
| Stelliferous 0.0.3 | 767 秒 | 0.026418747 |
| PyTorch 2.14.0 | 31.1 秒 | 0.02641915 |

```sh
python run.py training --threads 4
```

網路、資料與計時方式見[訓練 benchmark](../benchmarks/training/README.zh-TW.md)。

## 其他卷積形狀

部分形狀增加 worker 的幫助不大：權重多而輸出位置少、多於一張圖的 batch，以及寬度小於六的輸出。見[設計](design.zh-TW.md#已知限制)。
