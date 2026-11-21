# ResNet-20 training example

[English](README.md) | [繁體中文](README.zh-TW.md)

Trains ResNet-20 on CIFAR-10 with the library and, from the same initial weights and the same batches, with PyTorch.

## Set up

After the [library setup](../../docs/getting_started.md), download CIFAR-10 (163 MB):

```sh
mkdir -p out/data/cifar10_download
curl -L -o out/data/cifar10_download/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
```

## Run

```sh
python run.py training                         # two short epochs, compared bit for bit
python run.py training --threads 4 --epochs 160  # the full training
```

Both unpack the archive and write the initial weights if missing, then compile the model. The short check trains 3 steps of each of two epochs on 1, 2, 4 and 8 threads, and the second epoch again from the checkpoint of the first. Every run must hold the same bits, and the first loss, a forward pass from the same weights, must be within 1e-5 of PyTorch's. The later losses are listed beside PyTorch's: two FP32 trainings drift apart from the first step on, as PyTorch's own FP32 and float64 gradients do. The full training writes a checkpoint after every epoch and continues after the last one written. Results go to `out/results/resnet20/`.

## The training

[resnet20.bend](resnet20.bend) follows He et al. (2016): three stages of three residual blocks, a 1×1 projection with batch normalization where the shape changes, SGD with momentum 0.9 and weight decay 1e-4, the rate 0.1, then 0.01 from epoch 80 and 0.001 from epoch 120, batches of 128 images padded by 4, cropped and flipped at random.

```python
# The layers are read from the end of the network: the second convolution first.
second : Train.Network <- convolution(name ++ "_second",outputs,outputs,3,1,workers,batch_normalized(outputs,Train.End{}))
branch : Train.Network <- convolution(name ++ "_first",outputs,inputs,3,stride,workers,batch_normalized(outputs,relu(second)))
# A 1x1 convolution with batch normalization where the shape changes, otherwise Train.End{}.
projection : Train.Network <- shortcut(U32.is_ne(stride,1) || U32.is_ne(inputs,outputs),name,inputs,outputs,stride,workers)
# The shortcut is added to the branch before the last ReLU.
IO.pure(Train.Network,Train.Residual{branch,projection,relu(rest)})
```

The order of the images, the crops and the flips come from `F.random_uniform` and `F.random_below`, so the PyTorch twin trains on the same batches. Every convolution has a bias, and the head is a 1×1 convolution after the average pool.

| File | Role |
|---|---|
| [resnet20.bend](resnet20.bend) | The data pipeline, the network, the epochs, the test and the checkpoints |
| [resnet20_torch.py](resnet20_torch.py) | The same training in PyTorch |
| [prepare.py](prepare.py) | Unpacks CIFAR-10 and writes the initial weights |
| [build_run.sh](build_run.sh) | Compiles and trains on a thread count, continuing after the last epoch written |
| [compare_results.py](compare_results.py) | The comparison between thread counts and with PyTorch |
