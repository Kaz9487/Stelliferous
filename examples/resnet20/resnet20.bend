# ResNet-20 (He et al. 2016) trained on CIFAR-10 through the public training
# API: three stages of three residual blocks, a 1x1 convolution with batch
# normalization as the shortcut where the shape changes, the global average
# pool and a 1x1 convolution to the 10 classes. SGD with momentum 0.9 and
# weight decay 1e-4, the rate 0.1, 0.01 from epoch 80 and 0.001 from epoch 120,
# batches of 128 images, each padded by 4, cropped and flipped at random. The
# order, the crops and the flips come from the counter-based generator, so
# resnet20_torch.py trains on the same batches. README.md lists the settings.
import Base
import ../../lib/math_activation.bend as Act
import ../../lib/tensor_f32.bend as F
import ../../lib/tensor_training.bend as Train

def seed() -> U32:
  20

def batch() -> U32:
  128

def test_batch() -> U32:
  100

# A label byte and 3 x 32 x 32 pixel bytes.
def record() -> U32:
  3073

def data() -> String:
  "out/data/cifar10/cifar-10-batches-bin/"

def initial() -> String:
  "out/results/resnet20/initial/"

# 0.1, then 0.01 from epoch 80 and 0.001 from epoch 120.
def rate(+epoch: U32) -> F32:
  Bool.pick(F32,U32.is_lt(epoch,80),0.1,Bool.pick(F32,U32.is_lt(epoch,120),0.01,0.001))

# ---- Settings ----

def text(result: Result<&1,&1,U32 & String,String>,fallback: String) -> String:
  match result:
    case Done{value}: value
    case Fail{error}: fallback

def number(text: Maybe<&2,U32>,fallback: U32) -> U32:
  match text:
    case Some{value}: value
    case None{}: fallback

def variable(+name: String,fallback: String) -> IO(String):
  do IO<String>:
    result : Result<&1,&1,U32 & String,String> <- IO.get_env(name)
    IO.pure(String,text(result,fallback))

def counted(+name: String,fallback: U32) -> IO(U32):
  do IO<U32>:
    result : Result<&1,&1,U32 & String,String> <- IO.get_env(name)
    IO.pure(U32,number(U32.read(text(result,"none")),fallback))

# epochs: the epochs to reach; start: the first epoch, which reads the
# checkpoint of the one before when it is not 0; steps: the batches of each
# epoch; files: the training files read; tests: the test images evaluated.
type Settings is Data:
  Settings{epochs: U32,start: U32,steps: U32,files: U32,tests: U32,workers: U32,output: String}

# ---- Images ----

# A random crop of one record: the top-left corner of the 32 x 32 window in
# the record padded by 4, and whether the window is flipped left to right.
type Crop is Data:
  Crop{base: U32,row: U32,column: U32,flipped: Bool}

def channel_mean() -> List<&2,F32>:
  [0.4914,0.4822,0.4465]

def channel_deviation() -> List<&2,F32>:
  [0.2470,0.2435,0.2616]

# The value read where the window is inside the image, otherwise zero, in
# front of tail.
def read_into(+inside: Bool,result: F.Storage() & F32,tail: List<&2,F32>) -> F.Storage() & List<&2,F32>:
  (storage,+value) = result
  (storage,Bool.pick(F32,inside,value,0.0) <> tail)

# The remaining values of the window up to index in front of the list of
# state, in row-major order: zero where the window covers the padding.
def cropped(remaining: Nat,state: F.Storage() & List<&2,F32>,+index: U32,+base: U32,+row: U32,+column: U32,+flipped: Bool) -> F.Storage() & List<&2,F32>:
  match remaining:
    case 0n: state
    case 1n+rest:
      (storage,tail) = state
      +x = (index % 32 : U32)
      +padded_row = ((index / 32 : U32) % 32 + row : U32)
      +padded_column = (Bool.pick(U32,flipped,(31 - x : U32),x) + column : U32)
      +inside = U32.is_le(4,padded_row) && U32.is_lt(padded_row,36) && U32.is_le(4,padded_column) && U32.is_lt(padded_column,36)
      +address = (base + 1 + (index / 1024 : U32) * 1024 + (padded_row - 4 : U32) * 32 + (padded_column - 4 : U32) : U32)
      cropped(rest,read_into(inside,F.read_storage(storage,Bool.pick(U32,inside,address,base)),tail),(index - 1 : U32),base,row,column,flipped)

# Images and labels of a batch, in order.
type Gathered is Type:
  Gathered{storage: F.Storage(),values: List<&2,F32>,labels: List<&2,F32>}

def label_read(values: List<&2,F32>,+labels: List<&2,F32>,result: F.Storage() & F32) -> Gathered:
  (storage,+label) = result
  Gathered{storage,values,label <> labels}

def labelled(+base: U32,+labels: List<&2,F32>,result: F.Storage() & List<&2,F32>) -> Gathered:
  (storage,values) = result
  label_read(values,labels,F.read_storage(storage,base))

def windowed(+crop: Crop,result: Gathered) -> Gathered:
  Crop{+base,+row,+column,+flipped} = crop
  Gathered{storage,values,+labels} = result
  labelled(base,labels,cropped(3072n,(storage,values),3071,base,row,column,flipped))

# The crops are in reverse order: each one goes in front of the ones after it.
def gathered(+reversed: List<&2,Crop>,result: Gathered) -> Gathered:
  match reversed:
    case Nil{}: result
    case Con{+crop,rest}: gathered(rest,windowed(crop,result))

def normalized(values: F.Tensor()) -> F.Tensor():
  F.div(F.sub(F.div_scalar(values,255.0),F.from_list([1,3,1,1],channel_mean())),F.from_list([1,3,1,1],channel_deviation()))

# Images [count,3,32,32] normalized as torchvision does, labels as one-hot rows.
type Batch is Type:
  Batch{storage: F.Storage(),images: F.Tensor(),target: F.Tensor()}

def batched(+count: U32,+workers: U32,result: Gathered) -> Batch:
  Gathered{storage,+values,+labels} = result
  Batch{storage,F.with_workers(normalized(F.from_list([count,3,32,32],values)),workers),
    Train.one_hot(F.with_workers(F.from_list([count],labels),workers),10)}

# ---- The order of an epoch ----

type Keyed is Data:
  Keyed{key: F32,index: U32}

def before(+left: Keyed,+right: Keyed) -> Bool:
  Keyed{+left_key,+left_index} = left
  Keyed{+right_key,+right_index} = right
  F32.is_lt(left_key,right_key) || (F32.is_eq(left_key,right_key) && U32.is_lt(left_index,right_index))

def picked(second_first: Bool,+first: Keyed,+rest: List<&2,Keyed>,+second: Keyed,+others: List<&2,Keyed>,
  next: List<&2,Keyed> -> List<&2,Keyed> -> List<&2,Keyed>) -> List<&2,Keyed>:
  match second_first:
    case True{}: second <> next(first <> rest,others)
    case False{}: first <> next(rest,second <> others)

# fuel is at least the length of both lists.
def merged(fuel: Nat,+left: List<&2,Keyed>,+right: List<&2,Keyed>) -> List<&2,Keyed>:
  match fuel left right:
    case 0n left right: left
    case 1n+more Nil{} right: right
    case 1n+more Con{+first,rest} Nil{}: first <> rest
    case 1n+more Con{+first,rest} Con{+second,others}:
      picked(before(second,first),first,rest,second,others,left => right => merged(more,left,right))

# The values alternately in front of mine and theirs.
def dealt(+values: List<&2,Keyed>,+mine: List<&2,Keyed>,+theirs: List<&2,Keyed>) -> List<&2,Keyed> & List<&2,Keyed>:
  match values:
    case Nil{}: (mine,theirs)
    case Con{+first,rest}: dealt(rest,theirs,first <> mine)

def halves(pair: List<&2,Keyed> & List<&2,Keyed>,next: List<&2,Keyed> -> List<&2,Keyed> -> List<&2,Keyed>) -> List<&2,Keyed>:
  (left,right) = pair
  next(left,right)

# A merge sort of at most count values. The keys of the epoch, with ties on
# the smaller index first, are in one order whichever way the values are dealt.
def sorted(fuel: Nat,+count: U32,+values: List<&2,Keyed>) -> List<&2,Keyed>:
  match fuel values:
    case 0n values: values
    case 1n+more Nil{}: []
    case 1n+more Con{+first,Nil{}}: [first]
    case 1n+more values:
      +depth = more
      halves(dealt(values,[],[]),left => right => merged(U32.to_nat(count),sorted(depth,count,left),sorted(depth,count,right)))

def keyed(+keys: List<&2,F32>,+index: U32) -> List<&2,Keyed>:
  match keys:
    case Nil{}: []
    case Con{+key,rest}: Keyed{key,index} <> keyed(rest,(index + 1 : U32))

# The crop of each position of the epoch: the image in the sorted order, the
# corner and the flip drawn for the position.
def zipped(+order: List<&2,Keyed>,+rows: List<&2,F32>,+columns: List<&2,F32>,+flips: List<&2,F32>) -> List<&2,Crop>:
  match order rows columns flips:
    case Con{Keyed{+key,+index},order} Con{+row,rows} Con{+column,columns} Con{+flip,flips}:
      Crop{(index * record() : U32),F32.to_u32(row),F32.to_u32(column),F32.is_eq(flip,1.0)} <> zipped(order,rows,columns,flips)
    case order rows columns flips: []

def random_list(tensor: F.Tensor()) -> List<&2,F32>:
  F.to_list_or(tensor,[])

# The order sorts the images by a uniform key of the epoch.
def plan(+epoch: U32,+count: U32) -> List<&2,Crop>:
  zipped(sorted(32n,count,keyed(random_list(F.random_uniform([count],seed(),epoch)),0)),random_list(F.random_below([count],9,seed(),(1000 + epoch : U32))),
    random_list(F.random_below([count],9,seed(),(2000 + epoch : U32))),random_list(F.random_below([count],2,seed(),(3000 + epoch : U32))))

# The next count crops in front of chosen, the last one first, and the rest.
def taken(count: Nat,+crops: List<&2,Crop>,+chosen: List<&2,Crop>) -> List<&2,Crop> & List<&2,Crop>:
  match count crops:
    case 0n crops: (chosen,crops)
    case 1n+more Nil{}: (chosen,[])
    case 1n+more Con{+crop,rest}: taken(more,rest,crop <> chosen)

# ---- The network ----

def load(+name: String,+shape: List<&2,U32>,+workers: U32) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    tensor : F.Tensor() <- F.load_raw(initial() ++ name ++ ".bin",shape)
    IO.pure(F.Tensor(),F.with_workers(tensor,workers))

# The network is built from its end, so only networks wait across a load.
def convolution(+name: String,+outputs: U32,+inputs: U32,+size: U32,+stride: U32,+workers: U32,rest: Train.Network) -> IO(Train.Network):
  do IO<Train.Network>:
    weights : F.Tensor() <- load(name ++ "_weights",[outputs,inputs,size,size],workers)
    bias : F.Tensor() <- load(name ++ "_bias",[outputs],workers)
    IO.pure(Train.Network,Train.Then{Train.conv2d_layer(weights,bias,stride,(size / 2 : U32)),rest})

def batch_normalized(+channels: U32,rest: Train.Network) -> Train.Network:
  Train.Then{Train.batch_norm_layer(channels),rest}

def relu(rest: Train.Network) -> Train.Network:
  Train.Then{Train.activation_layer(Act.relu()),rest}

# A 1x1 convolution with batch normalization where the stride or the channels
# change; otherwise the input itself.
def shortcut(projected: Bool,+name: String,+inputs: U32,+outputs: U32,+stride: U32,+workers: U32) -> IO(Train.Network):
  match projected:
    case False{}: IO.pure(Train.Network,Train.End{})
    case True{}: convolution(name ++ "_shortcut",outputs,inputs,1,stride,workers,batch_normalized(outputs,Train.End{}))

# Two 3x3 convolutions with batch normalization; the shortcut is added before
# the last ReLU.
def block(+name: String,+inputs: U32,+outputs: U32,+stride: U32,+workers: U32,rest: Train.Network) -> IO(Train.Network):
  do IO<Train.Network>:
    second : Train.Network <- convolution(name ++ "_second",outputs,outputs,3,1,workers,batch_normalized(outputs,Train.End{}))
    branch : Train.Network <- convolution(name ++ "_first",outputs,inputs,3,stride,workers,batch_normalized(outputs,relu(second)))
    projection : Train.Network <- shortcut(U32.is_ne(stride,1) || U32.is_ne(inputs,outputs),name,inputs,outputs,stride,workers)
    IO.pure(Train.Network,Train.Residual{branch,projection,relu(rest)})

def stage(+number: String,+inputs: U32,+outputs: U32,+stride: U32,+workers: U32,rest: Train.Network) -> IO(Train.Network):
  do IO<Train.Network>:
    third : Train.Network <- block("stage" ++ number ++ "_block3",outputs,outputs,1,workers,rest)
    second : Train.Network <- block("stage" ++ number ++ "_block2",outputs,outputs,1,workers,third)
    block("stage" ++ number ++ "_block1",inputs,outputs,stride,workers,second)

def network(+workers: U32) -> IO(Train.Network):
  do IO<Train.Network>:
    head : Train.Network <- convolution("head",10,64,1,1,workers,Train.End{})
    third : Train.Network <- stage("3",32,64,2,workers,Train.Then{Train.average_pool_layer(),head})
    second : Train.Network <- stage("2",16,32,2,workers,third)
    first : Train.Network <- stage("1",16,16,1,workers,second)
    convolution("stem",16,3,3,1,workers,batch_normalized(16,relu(first)))
# ---- Training ----

def descended(+rate: F32,result: F.Tensor() & Train.Network) -> Train.Network:
  (dx,network) = result
  Train.sgd_network_with(network,rate,0.9,0.0001)

def lost(+rate: F32,tape: Train.Tape,result: F.Tensor() & (F.Tensor() & F.Tensor())) -> F32 & Train.Network:
  (loss,(dlogits,target)) = result
  (F.item_or(loss,0.0),descended(rate,Train.backward_network(F.reshape(dlogits,[batch(),10,1,1]),tape)))

def predicted(+rate: F32,target: F.Tensor(),result: F.Tensor() & Train.Tape) -> F32 & Train.Network:
  (logits,tape) = result
  lost(rate,tape,Train.cross_entropy(F.reshape(logits,[batch(),10]),target))

# One step on a batch: the loss and the network after it.
def step(+rate: F32,batch: Batch,network: Train.Network) -> F.Storage() & (F32 & Train.Network):
  Batch{storage,images,target} = batch
  (storage,predicted(rate,target,Train.forward_network(images,network)))

type Trained is Type:
  Trained{storage: F.Storage(),losses: List<&2,F32>,network: Train.Network}

def continued(+loss: F32,result: Trained) -> Trained:
  Trained{storage,+losses,network} = result
  Trained{storage,loss <> losses,network}

def stepped(result: F.Storage() & (F32 & Train.Network),next: F.Storage() -> F32 -> Train.Network -> Trained) -> Trained:
  (storage,(loss,network)) = result
  next(storage,loss,network)

def crops_then(result: List<&2,Crop> & List<&2,Crop>,next: List<&2,Crop> -> List<&2,Crop> -> Trained) -> Trained:
  (+chosen,+crops) = result
  next(chosen,crops)

# The loss of every step of the epoch, in order.
def trained(remaining: Nat,+crops: List<&2,Crop>,+rate: F32,+workers: U32,storage: F.Storage(),network: Train.Network) -> Trained:
  match remaining:
    case 0n: Trained{storage,[],network}
    case 1n+rest:
      crops_then(taken(U32.to_nat(batch()),crops,[]),chosen => crops =>
        stepped(step(rate,batched(batch(),workers,gathered(chosen,Gathered{storage,[],[]})),network),storage => loss => network =>
          continued(loss,trained(rest,crops,rate,workers,storage,network))))

# ---- Testing ----

def test_crops(remaining: Nat,+first: U32,+crops: List<&2,Crop>) -> List<&2,Crop>:
  match remaining:
    case 0n: crops
    case 1n+rest: test_crops(rest,(first + 1 : U32),Crop{(first * record() : U32),4,4,False{}} <> crops)

def correct(predicted: F.Tensor(),target: F.Tensor()) -> F32:
  F.item_or(F.sum(F.equal(predicted,target)),0.0)

def judged(+count: U32,labels: F.Tensor(),result: F.Tensor() & Train.Network) -> F32 & Train.Network:
  (logits,network) = result
  (correct(F.argmax(F.reshape(logits,[count,10]),1),labels),network)

def evaluated(+count: U32,batch: Batch,network: Train.Network) -> F.Storage() & (F32 & Train.Network):
  Batch{storage,images,target} = batch
  (storage,judged(count,F.argmax(target,1),Train.evaluate_network(images,network)))

type Tested is Type:
  Tested{storage: F.Storage(),correct: F32,network: Train.Network}

def counted_in(+right: F32,result: Tested) -> Tested:
  Tested{storage,+correct,network} = result
  Tested{storage,(correct + right : F32),network}

def judged_then(result: F.Storage() & (F32 & Train.Network),next: F.Storage() -> F32 -> Train.Network -> Tested) -> Tested:
  (storage,(right,network)) = result
  next(storage,right,network)

# The test images classified correctly, a batch of test_batch() at a time.
def tested(remaining: Nat,+first: U32,+workers: U32,storage: F.Storage(),network: Train.Network) -> Tested:
  match remaining:
    case 0n: Tested{storage,0.0,network}
    case 1n+rest:
      judged_then(evaluated(test_batch(),batched(test_batch(),workers,gathered(test_crops(U32.to_nat(test_batch()),first,[]),Gathered{storage,[],[]})),network),
        storage => right => network => counted_in(right,tested(rest,(first + test_batch() : U32),workers,storage,network)))

# ---- Checkpoints ----

# The initial network at epoch 0, otherwise the one after the epoch before.
def resumed(fresh: Bool,+folder: String,network: Train.Network) -> IO(Train.Network):
  match fresh:
    case True{}: IO.pure(Train.Network,network)
    case False{}: Train.restore_network(folder,network)

# ---- Epochs ----

def save(+path: String,tensor: F.Tensor()) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    tensor : F.Tensor() <- F.save_raw(path,tensor)
    F.expect(tensor)

def total(+values: List<&2,F32>,+sum: F32) -> F32:
  match values:
    case Nil{}: sum
    case Con{+value,rest}: total(rest,(sum + value : F32))

type Run is Type:
  Run{training: F.Storage(),test: F.Storage(),network: Train.Network}

def finished(+settings: Settings,+epoch: U32,+begin: Nat,+losses: List<&2,F32>,result: Tested,training: F.Storage()) -> IO(Run):
  Settings{+epochs,+start,+steps,+files,+tests,+workers,+output} = settings
  Tested{test,+right,network} = result
  do IO<Run>:
    end : Nat <- IO.now()
    network : Train.Network <- Train.save_network(output ++ "checkpoint/",network)
    correct : F.Tensor() <- save(output ++ "correct_" ++ U32.show(epoch) ++ ".bin",F.from_list([1],[right]))
    lost : F.Tensor() <- save(output ++ "losses_" ++ U32.show(epoch) ++ ".bin",F.from_list([steps],losses))
    IO.print("Stelliferous epoch " ++ U32.show(epoch) ++ ", workers " ++ U32.show(workers) ++ ": " ++ Nat.show(Nat.sub(end,begin)) ++ " ms, loss "
      ++ F32.show((total(losses,0.0) / U32.to_f32(steps) : F32)) ++ ", test " ++ U32.show(F32.to_u32(right)) ++ " of " ++ U32.show(tests))
    IO.pure(Run,Run{training,test,network})

def epoch_tested(+settings: Settings,+epoch: U32,+begin: Nat,test: F.Storage(),result: Trained) -> IO(Run):
  Settings{+epochs,+start,+steps,+files,+tests,+workers,+output} = settings
  Trained{training,+losses,network} = result
  finished(settings,epoch,begin,losses,tested(U32.to_nat((tests / test_batch() : U32)),0,workers,test,network),training)

# The losses of each step are written before the line of the epoch, after the
# checkpoint it resumes from.
def epoch_run(+settings: Settings,+epoch: U32,run: Run) -> IO(Run):
  Settings{+epochs,+start,+steps,+files,+tests,+workers,+output} = settings
  Run{training,test,network} = run
  do IO<Run>:
    begin : Nat <- IO.now()
    epoch_tested(settings,epoch,begin,test,trained(U32.to_nat(steps),plan(epoch,(files * 10000 : U32)),rate(epoch),workers,training,network))

def epochs_from(remaining: Nat,+settings: Settings,+epoch: U32,run: Run) -> IO(Run):
  match remaining:
    case 0n: IO.pure(Run,run)
    case 1n+rest:
      do IO<Run>:
        run : Run <- epoch_run(settings,epoch,run)
        epochs_from(rest,settings,(epoch + 1 : U32),run)

# ---- Main ----

def records(+name: String,+workers: U32) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    tensor : F.Tensor() <- F.load_u8(data() ++ name,[10000,record()])
    F.expect(F.with_workers(tensor,workers))

def training_file(+number: U32) -> String:
  "data_batch_" ++ U32.show(number) ++ ".bin"

# first and the remaining files from data_batch_number.bin, one record after
# another.
def followed(remaining: Nat,+number: U32,+workers: U32,first: F.Tensor()) -> IO(F.Parts()):
  match remaining:
    case 0n: IO.pure(F.Parts(),F.single(first))
    case 1n+more:
      do IO<F.Parts()>:
        next : F.Tensor() <- records(training_file(number),workers)
        rest : F.Parts() <- followed(more,(number + 1 : U32),workers,next)
        IO.pure(F.Parts(),F.cons(first,rest))

def trained_from(+settings: Settings) -> IO(Unit):
  Settings{+epochs,+start,+steps,+files,+tests,+workers,+output} = settings
  do IO<Unit>:
    first : F.Tensor() <- records(training_file(1),workers)
    parts : F.Parts() <- followed(U32.to_nat((files - 1 : U32)),2,workers,first)
    training : F.Storage() <- F.expect_storage(F.concat(parts,0))
    test : F.Tensor() <- records("test_batch.bin",workers)
    test : F.Storage() <- F.expect_storage(test)
    built : Train.Network <- network(workers)
    network : Train.Network <- resumed(U32.is_eq(start,0),output ++ "checkpoint/",built)
    run : Run <- epochs_from(U32.to_nat((epochs - start : U32)),settings,start,Run{training,test,network})
    IO.pure(Unit,Unit{})

def main() -> IO(Unit):
  do IO<Unit>:
    epochs : U32 <- counted("RESNET_EPOCHS",1)
    start : U32 <- counted("RESNET_START",0)
    steps : U32 <- counted("RESNET_STEPS",390)
    files : U32 <- counted("RESNET_FILES",5)
    tests : U32 <- counted("RESNET_TESTS",10000)
    output : String <- variable("RESNET_OUTPUT","out/results/resnet20/bend/")
    workers : U32 <- IO.thread_count()
    trained_from(Settings{epochs,start,steps,files,tests,workers,output})
