# Benchmarks

[English](README.md) | [繁體中文](README.zh-TW.md)

完成[開始使用](../docs/getting_started.zh-TW.md)後，在儲存庫根目錄執行。每個流程都寫到 `out/results/` 底下。

| 指令 | 作用 | 說明 |
|---|---|---|
| `python run.py integration` | 原生卷積案例，逐位元比對 | [convolution_cases.py](checks/convolution_cases.py) |
| `python run.py api` | 使用公開 API 的程式，與 NumPy 比對 | [public_api_cases.py](checks/public_api_cases.py) |
| `python run.py gradients` | 梯度、loss 與訓練步驟，與參考程式逐位元比對，也與 PyTorch 比對到捨入誤差 | [gradient_cases.py](checks/gradient_cases.py) |
| `python run.py benchmark` | 單一卷積形狀的時間 | [convolution](convolution/README.zh-TW.md) |
| `python run.py training` | 訓練一個 epoch 的時間，Stelliferous 與 PyTorch | [training](training/README.zh-TW.md) |
| `python run.py backends` | YOLOv5n 在 NumPy、Stelliferous 與 PyTorch 上的 forward 時間 | [backends](backends/README.zh-TW.md) |
| `python -B benchmarks/model/...` | 模型各運算的時間、配置與 worker 時間軸 | [model](model/README.zh-TW.md) |
| `python -B benchmarks/capture_baseline.py LABEL` | 把目前 `benchmark` 與 `backends` 的結果收集成 `out/results/baselines/LABEL/evidence.json` | [capture_baseline.py](capture_baseline.py) |
| `python -B benchmarks/checks/math_accuracy.py` | `sigmoid`、`silu`、`tanh`、`gelu_tanh`、`exp`、`log` 對 MPFR 的最大誤差，涵蓋每一個 FP32 輸入。Linux，需安裝 MPFR | [math_accuracy.py](checks/math_accuracy.py) |
| `python -B benchmarks/checks/function_timing.py --baseline OTHER_CHECKOUT --function gelu_tanh` | 一個逐元素函數每個元素的時間，這個 checkout 與另一個的對照 | [function_timing.py](checks/function_timing.py) |
| `python -B benchmarks/checks/measure_parameter_gradients.py` | 權重與偏差梯度在幾種組長下的時間，與同形狀前向卷積對照 | [measure_parameter_gradients.py](checks/measure_parameter_gradients.py) |
| `python -B benchmarks/checks/measure_arithmetic_dispatch.py` | 在同一個迴圈裡，比較存在 record 裡的函式與以 template 傳入的函式。Linux，需設定 `CC` | [設計](../docs/design.zh-TW.md#熱迴圈裡的函式) |

公布的數字在[效能](../docs/performance.zh-TW.md)。
