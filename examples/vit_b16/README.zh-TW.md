# ViT-B/16 範例

[English](README.md) | [繁體中文](README.zh-TW.md)

在這個函式庫上跑官方 torchvision ViT-B/16（ImageNet 權重，224 × 224），並和 PyTorch 比對。

## 準備

完成[函式庫的準備](../../docs/getting_started.zh-TW.md)後，下載權重（330 MB）：

```sh
mkdir -p out/weights/vit_b_16
curl -L -o out/weights/vit_b_16/vit_b_16-c867db91.pth https://download.pytorch.org/models/vit_b_16-c867db91.pth
```

## 執行

```sh
python run.py vit --threads 4
```

沒有 PyTorch 參考值時會先產生，接著編譯模型、執行，比對每個 block、logits 與前五名類別。多執行緒的結果也會和單執行緒逐位元比對。結果放在 `out/results/vit_b16/`。

```sh
python run.py vit --threads 4 --timing   # then time it and PyTorch alternately
```

計時跑三輪，每輪是編譯後的模型十次前向，接著 PyTorch 十次（PyTorch 先暖機一次），最後印出兩者的中位數與範圍。樣本寫在 `out/results/vit_b16/timing_4t.json`。

## 模型

[vit_b16.bend](vit_b16.bend) 是網路本身，和 torchvision 一樣逐 block 寫。

```python
# A linear layer: F.linear reads the weight [outputs,inputs] as PyTorch stores it.
def linear(x: F.Tensor(),+weight_name: String,+bias_name: String) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    weight : F.Tensor() <- F.take(weight_name)
    bias : F.Tensor() <- F.take(bias_name)
    F.Model.pure(F.Tensor(),F.linear(x,weight,bias))
```

| 檔案 | 用途 |
|---|---|
| [vit_b16.bend](vit_b16.bend) | 模型與計時的前向 |
| [prepare_reference.py](prepare_reference.py) | PyTorch 的輸入、block 輸出與 logits |
| [build_run.sh](build_run.sh) | 編譯，並用指定的執行緒數執行 |
| [compare_results.py](compare_results.py) | 和 PyTorch、和單執行緒比對 |
| [time_backends.py](time_backends.py) | 編譯後的模型與 PyTorch 交替計時 |
