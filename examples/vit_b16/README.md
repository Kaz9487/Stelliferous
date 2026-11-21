# ViT-B/16 example

[English](README.md) | [繁體中文](README.zh-TW.md)

Runs the official torchvision ViT-B/16 (ImageNet weights, 224 by 224) on the library and compares it with PyTorch.

## Set up

After the [library setup](../../docs/getting_started.md), download the weights (330 MB):

```sh
mkdir -p out/weights/vit_b_16
curl -L -o out/weights/vit_b_16/vit_b_16-c867db91.pth https://download.pytorch.org/models/vit_b_16-c867db91.pth
```

## Run

```sh
python run.py vit --threads 4
```

It prepares the PyTorch reference if missing, compiles the model, runs it, and compares each block, the logits and the top five classes. A run on more threads is also compared bit for bit with the run on one. Results go to `out/results/vit_b16/`.

## The model

[vit_b16.bend](vit_b16.bend) is the network, block by block as in torchvision.

```python
# A linear layer: F.linear reads the weight [outputs,inputs] as PyTorch stores it.
def linear(x: F.Tensor(),+weight_name: String,+bias_name: String) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    weight : F.Tensor() <- F.take(weight_name)
    bias : F.Tensor() <- F.take(bias_name)
    F.Model.pure(F.Tensor(),F.linear(x,weight,bias))
```

| File | Role |
|---|---|
| [vit_b16.bend](vit_b16.bend) | The model and its timed forward pass |
| [prepare_reference.py](prepare_reference.py) | The input, the block outputs and the logits of PyTorch |
| [build_run.sh](build_run.sh) | Compiles and runs on a thread count |
| [compare_results.py](compare_results.py) | The comparison with PyTorch and with one thread |
