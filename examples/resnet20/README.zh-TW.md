# ResNet-20 訓練範例

[English](README.md) | [繁體中文](README.zh-TW.md)

用本函式庫在 CIFAR-10 上訓練 ResNet-20，並以相同的初始權重與相同的 batch 用 PyTorch 訓練一次。

## 準備

完成[函式庫安裝](../../docs/getting_started.zh-TW.md)後，下載 CIFAR-10（163 MB）：

```sh
mkdir -p out/data/cifar10_download
curl -L -o out/data/cifar10_download/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
```

## 執行

```sh
python run.py training                         # 兩個短 epoch，逐位元比較
python run.py training --threads 4 --epochs 160  # 完整訓練
```

兩者都會在缺少時解開壓縮檔並寫出初始權重，再編譯模型。短檢查在 1、2、4、8 個執行緒上各訓練兩個 epoch、每個 3 步，並從第一個 epoch 的 checkpoint 再跑一次第二個 epoch。每次執行的位元必須相同，第一個 loss（從相同權重算的一次 forward）與 PyTorch 的相差不超過 1e-5。之後的 loss 和 PyTorch 的並列：兩個 FP32 訓練從第一步起就會逐漸分開，PyTorch 自己的 FP32 與 float64 梯度也是如此。完整訓練每個 epoch 後寫出 checkpoint，並從最後寫出的 epoch 之後繼續。結果在 `out/results/resnet20/`。

## 訓練方式

[resnet20.bend](resnet20.bend) 依照 He et al.（2016）：三個 stage、各三個殘差塊，形狀改變處用 1×1 投影加 batch normalization，SGD 的 momentum 0.9、weight decay 1e-4，學習率 0.1，第 80 個 epoch 起 0.01，第 120 個起 0.001，每個 batch 128 張影像，各補 4 格邊後隨機裁切與翻轉。

```python
# The layers are read from the end of the network: the second convolution first.
second : Train.Network <- convolution(name ++ "_second",outputs,outputs,3,1,workers,batch_normalized(outputs,Train.End{}))
branch : Train.Network <- convolution(name ++ "_first",outputs,inputs,3,stride,workers,batch_normalized(outputs,relu(second)))
# A 1x1 convolution with batch normalization where the shape changes, otherwise Train.End{}.
projection : Train.Network <- shortcut(U32.is_ne(stride,1) || U32.is_ne(inputs,outputs),name,inputs,outputs,stride,workers)
# The shortcut is added to the branch before the last ReLU.
IO.pure(Train.Network,Train.Residual{branch,projection,relu(rest)})
```

影像順序、裁切與翻轉來自 `F.random_uniform` 與 `F.random_below`，所以 PyTorch 對照組訓練的是同樣的 batch。每個卷積都有 bias，輸出層是平均池化後的 1×1 卷積。

| 檔案 | 用途 |
|---|---|
| [resnet20.bend](resnet20.bend) | 資料管線、網路、epoch、測試與 checkpoint |
| [resnet20_torch.py](resnet20_torch.py) | 同樣的訓練，用 PyTorch |
| [prepare.py](prepare.py) | 解開 CIFAR-10 並寫出初始權重 |
| [build_run.sh](build_run.sh) | 編譯並以指定執行緒數訓練，從最後寫出的 epoch 之後繼續 |
| [compare_results.py](compare_results.py) | 不同執行緒數之間與和 PyTorch 的比較 |
