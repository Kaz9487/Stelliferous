# YOLOv5n 範例

[English](README.md) | [繁體中文](README.zh-TW.md)

以本函式庫執行官方預訓練的 FP32 YOLOv5n，並與 PyTorch 比對。內附的權重與影像有各自的[出處與授權](../../data/README.zh-TW.md)。

## 設定

先完成[函式庫的設定](../../docs/getting_started.zh-TW.md)：

```sh
git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git .tools/yolov5-official
python -m venv .venv
. .venv/bin/activate                 # Windows：.venv\Scripts\activate
python -m pip install -r requirements-lock.txt
```

在 Linux 上，OpenCV 需要 libGL（Debian 或 Ubuntu：`apt install libgl1`）。

## 執行

```sh
python run.py model --case bus_640 --threads 4
```

它會在缺少時準備 PyTorch 參考結果，編譯模型、執行，並比對每一層、預測與偵測結果。結果寫在 `out/results/<case>/`。同一個編譯好的程式可用於每個案例與任何 `--threads` 值。

## 模型

[yolov5n.bend](yolov5n.bend) 就是這個網路，依官方模型一個區塊一個區塊寫成。它讀官方的 checkpoint，先把每個批次正規化依 PyTorch `fuse_conv_and_bn` 的運算順序合併進它前面的卷積（[batch_norm.bend](batch_norm.bend)）。區塊用 PyTorch 的名稱取用權重：

```python
# Convolution with its folded batch normalization, then SiLU.
def conv(x: F.Tensor(),+name: String,stride: U32,padding: U32) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    weights : F.Tensor() <- F.take(name ++ ".conv.weight")
    bias : F.Tensor() <- F.take(name ++ ".conv.bias")
    F.Model.pure(F.Tensor(),F.conv2d(x,weights,bias,stride,padding,Act.silu()))
```

```sh
node scripts/bend_launcher.mjs examples/yolov5/yolov5n.bend -o out/models/yolov5n
YOLO_OUTPUT=out ./out/models/yolov5n --threads 4      # forward ... ms，以及 out/pred.npy
```

| 變數 | 意義 | 預設 |
|---|---|---|
| `YOLO_SIZE` | 正方形影像的邊長，32 的倍數 | 640 |
| `YOLO_INPUT` | 原始 FP32 影像 `[1,3,size,size]` | `out/results/bus_640/input.bin` |
| `YOLO_OUTPUT` | 存放 `pred.npy` 的資料夾 | `out/results/bus_640/native` |
| `YOLO_LAYERS` | 有設定時，每一層的輸出也會存下來 | 未設定 |
| `YOLO_EXTRA` | 存檔那一次之前先跑幾次 forward | 0 |

## Checkpoint

[load_checkpoint.bend](load_checkpoint.bend) 用 `F.load_checkpoint` 在 Bend 裡讀官方的 `.pt` 檔，並把每個張量用它的 PyTorch 名稱寫出。比對程式會對 `torch.load` 檢查名稱、形狀與每一個位元。第二個 checkpoint 放的是官方檔裡沒有的儲存格式與排列：涵蓋 FP32 範圍與其捨入平手點的 FP64 值，以及轉置、切片與廣播的 view：

```sh
python run.py checkpoint         # PASS: 349 tensors equal bit for bit, 121 folded tensors equal bit for bit, then the 7 cases
```

合併後的張量以同樣方式和用 NumPy 算的 `fuse_conv_and_bn` 比對，NumPy 的平方根捨入到最近值。`comparison.json` 另外記下它們和 `model.fuse()` 相差幾個最後一位單位：`torch.sqrt` 不一定捨入到最近值。

## 檔案

| 檔案 | 角色 |
|---|---|
| [yolov5n.bend](yolov5n.bend) | 模型與它的程式 |
| [batch_norm.bend](batch_norm.bend) | 把 checkpoint 的批次正規化合併進卷積 |
| [prepare_reference.py](prepare_reference.py) | 匯出 PyTorch 參考結果與融合後的權重，存成 `out/weights/named/<name>.npy` |
| [numpy_yolo.py](numpy_yolo.py)、[golden_model.py](golden_model.py)、[image_pipeline.py](image_pipeline.py) | NumPy 模型與影像處理 |
| [compare_results.py](compare_results.py) | 比對與其門檻 |
| [build_run.sh](build_run.sh) | 建置編譯好的模型，並在一個案例上執行 |
| [load_checkpoint.bend](load_checkpoint.bend)、[load_checkpoint.sh](load_checkpoint.sh) | 在 Bend 裡載入一個 checkpoint，並把每個張量存成 `out/results/checkpoint/<case>/bend/<name>.npy`，合併後的存在 `.../folded` |
| [compare_checkpoint.py](compare_checkpoint.py) | 逐位元比對載入的張量與 `torch.load`，以及合併後的張量與 NumPy 算的 `fuse_conv_and_bn` |
| [checkpoint_cases.py](checkpoint_cases.py) | 寫出其他儲存格式與排列的 checkpoint |

偵測解碼屬於這個範例，不屬於函式庫。重複計時請用[後端 benchmark](../../benchmarks/backends/README.zh-TW.md)。
