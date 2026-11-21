# ViT-B/16 on the public tensor API, block by block as in torchvision, for images
# [count,3,224,224]. Weights carry their PyTorch names.
import Base
import ../../lib/tensor_f32.bend as F
import ../../lib/math_activation.bend as Act
import ../../lib/tensor_training.bend as Train

# Blocks

def linear(x: F.Tensor(),+weight_name: String,+bias_name: String) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    weight : F.Tensor() <- F.take(weight_name)
    bias : F.Tensor() <- F.take(bias_name)
    F.Model.pure(F.Tensor(),F.linear(x,weight,bias))

def normalized(result: F.Tensor() & Train.Layer) -> F.Tensor():
  (y,layer) = result
  y

def norm(x: F.Tensor(),+name: String) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    gamma : F.Tensor() <- F.take(name ++ ".weight")
    beta : F.Tensor() <- F.take(name ++ ".bias")
    F.Model.pure(F.Tensor(),normalized(Train.evaluate(x,Train.layer_norm_layer_from(gamma,beta,0.000001))))

# [count,length,768] as twelve heads [count,12,length,64].
def heads(x: F.Tensor(),+count: U32,+length: U32) -> F.Tensor():
  F.transpose(F.reshape(x,[count,length,12,64]),[0,2,1,3])

# softmax(q k^T / sqrt(64)) v, with the heads joined again.
def attended(q: F.Tensor(),k: F.Tensor(),v: F.Tensor(),+count: U32,+length: U32) -> F.Tensor():
  F.reshape(F.transpose(F.matmul(F.softmax(F.mul_scalar(F.matmul(heads(q,count,length),F.transpose(heads(k,count,length),[0,1,3,2])),0.125),3),heads(v,count,length)),[0,2,1,3]),[count,length,768])

def last_two(q: F.Tensor(),pair: F.Tensor() & F.Tensor(),+count: U32,+length: U32) -> F.Tensor():
  (k,v) = pair
  attended(q,F.slice(k,2,768,768,1),F.slice(v,2,1536,768,1),count,length)

# The projection of 2304 features holds q, k and v in turn.
def split(pair: F.Tensor() & F.Tensor(),+count: U32,+length: U32) -> F.Tensor():
  (q,rest) = pair
  last_two(F.slice(q,2,0,768,1),F.copy(rest),count,length)

def attention(x: F.Tensor(),+name: String,+count: U32,+length: U32) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    projected : F.Tensor() <- linear(x,name ++ ".in_proj_weight",name ++ ".in_proj_bias")
    linear(split(F.copy(projected),count,length),name ++ ".out_proj.weight",name ++ ".out_proj.bias")

def feed_forward(pair: F.Tensor() & F.Tensor(),+name: String) -> F.Model(F.Tensor()):
  (x,kept) = pair
  do F.Model<F.Tensor()>:
    y : F.Tensor() <- norm(x,name ++ ".ln_2")
    y : F.Tensor() <- linear(y,name ++ ".mlp.linear_1.weight",name ++ ".mlp.linear_1.bias")
    y : F.Tensor() <- linear(F.gelu(y),name ++ ".mlp.linear_2.weight",name ++ ".mlp.linear_2.bias")
    F.Model.pure(F.Tensor(),F.add(kept,y))

def block(pair: F.Tensor() & F.Tensor(),+name: String,+count: U32,+length: U32) -> F.Model(F.Tensor()):
  (x,kept) = pair
  do F.Model<F.Tensor()>:
    y : F.Tensor() <- norm(x,name ++ ".ln_1")
    y : F.Tensor() <- attention(y,name ++ ".self_attention",count,length)
    feed_forward(F.copy(F.add(y,kept)),name)

def saved(+trace: Bool,+name: String,x: F.Tensor()) -> F.Model(F.Tensor()):
  match trace:
    case True{}: F.keep(name,x)
    case False{}: F.Model.pure(F.Tensor(),x)

def blocks(remaining: Nat,+index: U32,x: F.Tensor(),+count: U32,+length: U32,+trace: Bool) -> F.Model(F.Tensor()):
  match remaining:
    case 0n: F.Model.pure(F.Tensor(),x)
    case 1n+rest:
      do F.Model<F.Tensor()>:
        y : F.Tensor() <- block(F.copy(x),"encoder.layers.encoder_layer_" ++ U32.show(index),count,length)
        y : F.Tensor() <- saved(trace,"block_" ++ U32.show(index),y)
        blocks(rest,(index + 1 : U32),y,count,length,trace)

# The network

def network(image: F.Tensor(),+count: U32,+trace: Bool) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    weights : F.Tensor() <- F.take("conv_proj.weight")
    bias : F.Tensor() <- F.take("conv_proj.bias")
    token : F.Tensor() <- F.take("class_token")
    position : F.Tensor() <- F.take("encoder.pos_embedding")
    patches : F.Tensor() <- saved(trace,"patches",F.transpose(F.reshape(F.conv2d(image,weights,bias,16,0,Act.identity()),[count,768,196]),[0,2,1]))
    x : F.Tensor() <- blocks(12n,0,F.add(F.concat(F.cons(F.broadcast_to(token,[count,1,768]),F.single(patches)),1),position),count,197,trace)
    x : F.Tensor() <- norm(x,"encoder.ln")
    x : F.Tensor() <- saved(trace,"encoder",x)
    linear(F.reshape(F.slice(x,1,0,1,1),[count,768]),"heads.head.weight","heads.head.bias")

# Running

# The forward pass has run when its result is known to be ready or failed.
def checked(pair: F.Tensor() & F.Weights()) -> IO(F.Tensor() & F.Weights()):
  (logits,kept) = pair
  do IO<F.Tensor() & F.Weights()>:
    logits : F.Tensor() <- F.expect(logits)
    IO.pure(F.Tensor() & F.Weights(),(logits,kept))

def text(result: Result<&1,&1,U32 & String,String>,fallback: String) -> String:
  match result:
    case Done{value}: value
    case Fail{error}: fallback

def set(result: Result<&1,&1,U32 & String,String>) -> Bool:
  match result:
    case Done{value}: True{}
    case Fail{error}: False{}

def variable(+name: String,fallback: String) -> IO(String):
  do IO<String>:
    result : Result<&1,&1,U32 & String,String> <- IO.get_env(name)
    IO.pure(String,text(result,fallback))

def flag(+name: String) -> IO(Bool):
  do IO<Bool>:
    result : Result<&1,&1,U32 & String,String> <- IO.get_env(name)
    IO.pure(Bool,set(result))

def number(text: Maybe<&2,U32>,fallback: U32) -> U32:
  match text:
    case Some{value}: value
    case None{}: fallback

def counted(+name: String,fallback: U32) -> IO(U32):
  do IO<U32>:
    result : Result<&1,&1,U32 & String,String> <- IO.get_env(name)
    IO.pure(U32,number(U32.read(text(result,"none")),fallback))

type Inputs is Type:
  Inputs{image: F.Tensor(),weights: F.Weights()}

# One timed forward pass.
def forward(image: F.Tensor(),weights: F.Weights(),+count: U32,+workers: U32,+trace: Bool) -> IO(F.Tensor() & F.Weights()):
  do IO<F.Tensor() & F.Weights()>:
    start : Nat <- IO.now()
    result : F.Tensor() & F.Weights() <- checked(F.run_keeping(F.Tensor(),network(F.with_workers(image,workers),count,trace),weights))
    end : Nat <- IO.now()
    IO.print("forward " ++ Nat.show(Nat.sub(end,start)) ++ " ms")
    IO.pure(F.Tensor() & F.Weights(),result)

# A forward pass on copies; the image and the weights stay for the next one.
def practice(images: F.Tensor() & F.Tensor(),stores: F.Weights() & F.Weights(),+count: U32,+workers: U32) -> IO(Inputs):
  (image,next_image) = images
  (weights,next_weights) = stores
  do IO<Inputs>:
    result : F.Tensor() & F.Weights() <- forward(image,weights,count,workers,False{})
    IO.pure(Inputs,Inputs{next_image,next_weights})

def repeated(extra: Nat,ready: Inputs,+count: U32,+workers: U32,+trace: Bool) -> IO(F.Tensor() & F.Weights()):
  match extra ready:
    case 0n Inputs{image,weights}: forward(image,weights,count,workers,trace)
    case 1n+rest Inputs{image,weights}:
      do IO<F.Tensor() & F.Weights()>:
        next : Inputs <- practice(F.copy(image),F.copy_weights(weights),count,workers)
        repeated(rest,next,count,workers,trace)

def finish(pair: F.Tensor() & F.Weights(),+folder: String) -> IO(Unit):
  (logits,kept) = pair
  do IO<Unit>:
    logits : F.Tensor() <- F.save_npy(folder ++ "/logits.npy",logits)
    layers : F.Weights() <- F.save_weights(folder,kept)
    IO.pure(Unit,Unit{})

def begin(+path: String,+input: String,+output: String,+trace: Bool,+extra: U32,+count: U32) -> IO(Unit):
  do IO<Unit>:
    workers : U32 <- IO.thread_count()
    weights : F.Weights() <- F.load_checkpoint(path)
    image : F.Tensor() <- F.load_raw(input,[count,3,224,224])
    result : F.Tensor() & F.Weights() <- repeated(U32.to_nat(extra),Inputs{image,weights},count,workers,trace)
    finish(result,output)

# VIT_WEIGHTS: the official .pth. VIT_INPUT: a raw [VIT_COUNT,3,224,224] tensor.
# VIT_OUTPUT: the folder for logits.npy. VIT_LAYERS: when set, the outputs of the
# blocks are saved there too. VIT_EXTRA: forward passes to run before the saved one.
def main() -> IO(Unit):
  do IO<Unit>:
    path : String <- variable("VIT_WEIGHTS","out/weights/vit_b_16/vit_b_16-c867db91.pth")
    input : String <- variable("VIT_INPUT","out/results/vit_b16/input.bin")
    output : String <- variable("VIT_OUTPUT","out/results/vit_b16/native")
    trace : Bool <- flag("VIT_LAYERS")
    extra : U32 <- counted("VIT_EXTRA",0)
    count : U32 <- counted("VIT_COUNT",1)
    begin(path,input,output,trace,extra,count)
