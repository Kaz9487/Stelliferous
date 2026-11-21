# Batch normalization of [N,C,H,W] over the images and positions of each
# channel, as a training layer: forward normalizes with the statistics of the
# batch and moves the running statistics, backward gives the gradients,
# evaluate normalizes with the running statistics. The cells take their lanes
# on the channels with the groups innermost. The sums of a channel are those
# of sum_axes, in the same order, read straight from the cells; every pass
# then reads the cells of a group with the statistics of the group.
import Base
import ./tensor.bend as Tensors
import ./tensor_f32.bend as F
import ./kernel_tile.bend as Tile
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_batch_norm_cells.bend as Cells
import ./kernel_batch_norm.bend as Kernel
import ./tensor_shape.bend as Shapes
import ./tensor_tiled.bend as Tiled
import ./tensor_tiled_map.bend as TiledMap
import ./tensor_tiled_reduce.bend as TiledReduce

def Tensor() -> Type:
  F.Tensor()

type Settings is Data:
  Settings{epsilon: F32,momentum: F32}

def settings(epsilon: F32,momentum: F32) -> Settings:
  Settings{epsilon,momentum}

# gamma and beta, each [C].
def Parameters() -> Type:
  Tensor() & Tensor()

# The running mean and the running variance, each [C].
def Running() -> Type:
  Tensor() & Tensor()

# The normalized input and 1 / sqrt(variance + epsilon). One backward call
# uses it.
def Saved() -> Type:
  Tensor() & Tensor()

def channels(shape: List<&2,U32>) -> U32:
  match shape:
    case [images,channels,height,width]: channels
    case shape: 0

# The number of values of each channel, N * H * W.
def count(shape: List<&2,U32>) -> U32:
  match shape:
    case [images,channels,height,width]: (images * (height * width : U32) : U32)
    case shape: 0

# 1 / sqrt(variance + epsilon).
def inverse_deviation(variance: Tensor(),+channels: U32,+epsilon: F32) -> Tensor():
  F.div(F.ones([channels]),F.sqrt(F.add_scalar(variance,epsilon)))

# running * (1 - momentum) + statistic * momentum.
def moved(running: Tensor(),statistic: Tensor(),+momentum: F32) -> Tensor():
  F.add(F.mul_scalar(running,(1.0 - momentum : F32)),F.mul_scalar(statistic,momentum))

# ---- The cells of the statistics ----

def Listed() -> Type:
  Result<&2,&2,Tensors.Problem,List<&2,Tile.Tile8>>

def read_kept(pair: Tensor() & Tensor()) -> Tensor() & Listed():
  (tensor,copy) = pair
  (tensor,Tiled.listed(copy))

def parameter_kept(+channels: U32,pair: Tensor() & Tensor()) -> Tensor() & Listed():
  (tensor,copy) = pair
  (tensor,Tiled.listed(F.reshape(copy,[channels])))

# The lanes on the channels, the groups innermost.
def innermost(tensor: Tensor()) -> Tensor():
  Tiled.arranged(tensor,1n,Grid.Innermost{})

def tensor_pieces(tensor: Tensors.Tensor<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  Tensors.Tensor{pieces,shape,fill,status,workers} = tensor
  pieces

def pieces_of(tensor: Tensor()) -> Pieces.Pieces(~Tile.Tile8):
  tensor_pieces(Tiled.cells_of(tensor))

# The extents of a layer.
type Layer is Data:
  Layer{images: U32,channels: U32,height: U32,width: U32,workers: U32}

def layer_values(+layer: Layer) -> U32:
  Layer{+images,channels,+height,+width,workers} = layer
  (images * (height * width : U32) : U32)

# The cells of the layer, [N,H,W,G].
def cell_shape(+layer: Layer) -> List<&2,U32>:
  Layer{+images,+channels,+height,+width,workers} = layer
  [images,height,width,Tiled.groups(channels)]

def layer_groups(+layer: Layer) -> U32:
  Layer{images,+channels,height,width,workers} = layer
  Tiled.groups(channels)

def layer_shape(+layer: Layer) -> List<&2,U32>:
  Layer{+images,+channels,+height,+width,workers} = layer
  [images,channels,height,width]

# What zeroes the lanes of the last group past the channels, as for a map
# of the cells.
def layer_padding(+layer: Layer) -> TiledMap.Padding:
  Layer{images,+channels,height,width,workers} = layer
  TiledMap.padding_of(cell_shape(layer),1n,channels,Grid.Innermost{})

def layer_workers(+layer: Layer) -> U32:
  Layer{images,channels,height,width,+workers} = layer
  workers

# Pieces of the cells of the layer as a tensor [N,C,H,W].
def layer_tensor(+layer: Layer,pieces: Pieces.Pieces(~Tile.Tile8)) -> Tensor():
  Layer{+images,+channels,+height,+width,+workers} = layer
  Tiled.Tiled{Tensors.Tensor{pieces,[images,height,width,Tiled.groups(channels)],Tiled.zero_tile(),Tensors.Usable{},workers},1n,channels,Grid.Innermost{}}

# The sums of the groups as a tensor [C], as the sums of axes [0, 2, 3] of
# [N,C,H,W] on the lanes of C hand them back.
def channel_sums(+layer: Layer,sums: List<&2,Tile.Tile8>) -> Tensor():
  Layer{images,+channels,height,width,+workers} = layer
  TiledReduce.kept_cells([0n,2n,3n],4n,1n,channels,Grid.Outermost{},
    Tensors.with_workers(~Tile.Tile8,Tensors.from_list(~Tile.Tile8,[Tiled.groups(channels)],sums,Tiled.zero_tile()),workers))

def normal_groups(means: List<&2,Tile.Tile8>,deviations: List<&2,Tile.Tile8>,gammas: List<&2,Tile.Tile8>,betas: List<&2,Tile.Tile8>) -> List<&2,Kernel.Normal>:
  match means deviations gammas betas:
    case Con{m,ms} Con{d,ds} Con{g,gs} Con{b,bs}: Con{Kernel.Normal{m,d,g,b},normal_groups(ms,ds,gs,bs)}
    case means deviations gammas betas: []

def normal_tiles(means: Listed(),deviations: Listed(),gammas: Listed(),betas: Listed()) -> Result<&2,&2,Tensors.Problem,List<&2,Kernel.Normal>>:
  match means deviations gammas betas:
    case Done{m} Done{d} Done{g} Done{b}: Done{normal_groups(m,d,g,b)}
    case Fail{problem} deviations gammas betas: Fail{problem}
    case means Fail{problem} gammas betas: Fail{problem}
    case means deviations Fail{problem} betas: Fail{problem}
    case means deviations gammas Fail{problem}: Fail{problem}

# ---- Training ----

# (the output, (what backward reads, (parameters, running statistics))).
def Forward() -> Type:
  Tensor() & (Saved() & (Parameters() & Running()))

# A failed forward fails every result it computes.
def forward_failed(+problem: Tensors.Problem,+shape: List<&2,U32>,+workers: U32,parameters: Parameters()) -> Forward():
  (Tiled.failed_as(problem,shape,workers),((Tiled.failed_as(problem,shape,workers),Tiled.failed_as(problem,[channels(shape)],workers)),
    (parameters,(Tiled.failed_as(problem,[channels(shape)],workers),Tiled.failed_as(problem,[channels(shape)],workers)))))

def layer_failed(+problem: Tensors.Problem,+layer: Layer,parameters: Parameters()) -> Forward():
  Layer{+images,+channels,+height,+width,+workers} = layer
  forward_failed(problem,[images,channels,height,width],workers,parameters)

def forward_passed(+layer: Layer,+momentum: F32,deviation: Tensor(),mean: Tensor(),unbiased: Tensor(),parameters: Parameters(),running: Running(),
  passes: Pieces.Pieces(~Tile.Tile8) & Pieces.Pieces(~Tile.Tile8)) -> Forward():
  (running_mean,running_variance) = running
  (xhat,output) = passes
  (layer_tensor(layer,output),((layer_tensor(layer,xhat),deviation),(parameters,(moved(running_mean,mean,momentum),moved(running_variance,unbiased,momentum)))))

def forward_grouped(+layer: Layer,+momentum: F32,pieces: Pieces.Pieces(~Tile.Tile8),deviation: Tensor(),mean: Tensor(),unbiased: Tensor(),parameters: Parameters(),running: Running(),
  groups: Result<&2,&2,Tensors.Problem,List<&2,Kernel.Normal>>) -> Forward():
  match groups:
    case Done{groups}:
      forward_passed(layer,momentum,deviation,mean,unbiased,parameters,running,Cells.forward(groups,layer_groups(layer),layer_padding(layer),cell_shape(layer),layer_workers(layer),pieces))
    case Fail{+problem}: layer_failed(problem,layer,parameters)

def forward_read(+layer: Layer,+momentum: F32,pieces: Pieces.Pieces(~Tile.Tile8),mean: Tensor() & Listed(),deviation: Tensor() & Listed(),unbiased: Tensor(),
  gamma: Tensor() & Listed(),beta: Tensor() & Listed(),running: Running()) -> Forward():
  (mean,means) = mean
  (deviation,deviations) = deviation
  (gamma,gammas) = gamma
  (beta,betas) = beta
  forward_grouped(layer,momentum,pieces,deviation,mean,unbiased,(gamma,beta),running,normal_tiles(means,deviations,gammas,betas))

def forward_squared(+layer: Layer,+epsilon: F32,+momentum: F32,mean: Tensor(),parameters: Parameters(),running: Running(),
  squares: Tensor() & Tensor(),pieces: Pieces.Pieces(~Tile.Tile8)) -> Forward():
  Layer{images,+channels,height,width,workers} = layer
  (gamma,beta) = parameters
  (squares,squares_kept) = squares
  +values = layer_values(layer)
  forward_read(layer,momentum,pieces,read_kept(F.copy(mean)),read_kept(F.copy(inverse_deviation(F.div_scalar(squares,U32.to_f32(values)),channels,epsilon))),
    F.div_scalar(squares_kept,U32.to_f32((values - 1 : U32))),parameter_kept(channels,F.copy(gamma)),parameter_kept(channels,F.copy(beta)),running)

def forward_squares(+layer: Layer,+epsilon: F32,+momentum: F32,mean: Tensor(),parameters: Parameters(),running: Running(),
  result: Pieces.Pieces(~Tile.Tile8) & List<&2,Tile.Tile8>) -> Forward():
  (pieces,squares) = result
  forward_squared(layer,epsilon,momentum,mean,parameters,running,F.copy(channel_sums(layer,squares)),pieces)

def forward_mean(+layer: Layer,+epsilon: F32,+momentum: F32,pieces: Pieces.Pieces(~Tile.Tile8),parameters: Parameters(),running: Running(),means: Tensor() & Listed()) -> Forward():
  (mean,means) = means
  match means:
    case Done{means}: forward_squares(layer,epsilon,momentum,mean,parameters,running,Cells.centered_sums(means,cell_shape(layer),layer_workers(layer),pieces))
    case Fail{+problem}: layer_failed(problem,layer,parameters)

def forward_sum(+layer: Layer,+epsilon: F32,+momentum: F32,parameters: Parameters(),running: Running(),result: Pieces.Pieces(~Tile.Tile8) & List<&2,Tile.Tile8>) -> Forward():
  (pieces,sums) = result
  forward_mean(layer,epsilon,momentum,pieces,parameters,running,read_kept(F.copy(F.div_scalar(channel_sums(layer,sums),U32.to_f32(layer_values(layer))))))

def forward_cells(+layer: Layer,+epsilon: F32,+momentum: F32,parameters: Parameters(),running: Running(),x: Tensor()) -> Forward():
  forward_sum(layer,epsilon,momentum,parameters,running,Cells.sums(cell_shape(layer),layer_workers(layer),pieces_of(x)))

# A layer with a zero extent has no statistics.
def forward_sized(empty: Bool,+images: U32,+channels: U32,+height: U32,+width: U32,+workers: U32,+epsilon: F32,+momentum: F32,x: Tensor(),parameters: Parameters(),running: Running()) -> Forward():
  match empty:
    case False{}: forward_cells(Layer{images,channels,height,width,workers},epsilon,momentum,parameters,running,innermost(x))
    case True{}: forward_failed(Tensors.Problem{"batch_norm",Tensors.InvalidShape{[images,channels,height,width]}},[images,channels,height,width],workers,parameters)

def forward_shaped(+settings: Settings,+shape: List<&2,U32>,+workers: U32,x: Tensor(),parameters: Parameters(),running: Running()) -> Forward():
  Settings{+epsilon,+momentum} = settings
  match shape:
    case [+images,+channels,+height,+width]:
      forward_sized(Shapes.has_zero([images,channels,height,width]),images,channels,height,width,workers,epsilon,momentum,x,parameters,running)
    case shape: forward_failed(Tensors.Problem{"batch_norm",Tensors.InvalidShape{shape}},shape,workers,parameters)

def forward_status(+settings: Settings,+status: Tensors.Status,+shape: List<&2,U32>,+workers: U32,x: Tensor(),parameters: Parameters(),running: Running()) -> Forward():
  match status:
    case Tensors.Usable{}: forward_shaped(settings,shape,workers,x,parameters,running)
    case Tensors.Failed{+problem}: forward_failed(problem,shape,workers,parameters)

# x [N,C,H,W], with M = N * H * W, mean = sum_axes(x,[0,2,3]) / M, d = x - mean
# and s = sum_axes(d * d,[0,2,3]): xhat = d * (1 / sqrt(s / M + epsilon)), the
# output xhat * gamma + beta, the running mean moved towards mean and the
# running variance towards s / (M - 1).
def forward(x: Tensor(),parameters: Parameters(),running: Running(),+settings: Settings) -> Forward():
  Tiled.with_metadata(~Forward(),Tiled.along(x,1n),x => shape => status => workers => forward_status(settings,status,shape,workers,x,parameters,running))

# (dx, (dgamma, (dbeta, gamma))).
def Backward() -> Type:
  Tensor() & (Tensor() & (Tensor() & Tensor()))

def backward_failed(+problem: Tensors.Problem,+shape: List<&2,U32>,+workers: U32,gamma: Tensor()) -> Backward():
  (Tiled.failed_as(problem,shape,workers),(Tiled.failed_as(problem,[channels(shape)],workers),(Tiled.failed_as(problem,[channels(shape)],workers),gamma)))

def back_groups(+total: F32,betas: List<&2,Tile.Tile8>,gammas: List<&2,Tile.Tile8>,scales: List<&2,Tile.Tile8>) -> List<&2,Kernel.Back>:
  match betas gammas scales:
    case Con{b,bs} Con{g,gs} Con{s,ss}: Con{Kernel.Back{b,g,s,total},back_groups(total,bs,gs,ss)}
    case betas gammas scales: []

def back_tiles(+total: F32,betas: Listed(),gammas: Listed(),scales: Listed()) -> Result<&2,&2,Tensors.Problem,List<&2,Kernel.Back>>:
  match betas gammas scales:
    case Done{b} Done{g} Done{s}: Done{back_groups(total,b,g,s)}
    case Fail{problem} gammas scales: Fail{problem}
    case betas Fail{problem} scales: Fail{problem}
    case betas gammas Fail{problem}: Fail{problem}

def back_grouped(+layer: Layer,beta: Tensor(),gamma_sum: Tensor(),gamma: Tensor(),pairs: Cells.Pairs(),groups: Result<&2,&2,Tensors.Problem,List<&2,Kernel.Back>>) -> Backward():
  match groups:
    case Done{groups}: (layer_tensor(layer,Cells.backward(groups,layer_groups(layer),layer_padding(layer),cell_shape(layer),layer_workers(layer),pairs)),(gamma_sum,(beta,gamma)))
    case Fail{+problem}: backward_failed(problem,layer_shape(layer),layer_workers(layer),gamma)

def back_read(+layer: Layer,beta: Tensor() & Listed(),gamma_sum: Tensor() & Listed(),scales: Listed(),gamma: Tensor(),pairs: Cells.Pairs()) -> Backward():
  (beta,betas) = beta
  (gamma_sum,gammas) = gamma_sum
  back_grouped(layer,beta,gamma_sum,gamma,pairs,back_tiles(U32.to_f32(layer_values(layer)),betas,gammas,scales))

def back_first(+layer: Layer,deviation: Tensor(),gammas: Tensor() & Tensor(),sums: List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,pairs: Cells.Pairs()) -> Backward():
  (gamma,gamma_kept) = gammas
  (shifts,scales) = sums
  back_read(layer,read_kept(F.copy(channel_sums(layer,shifts))),read_kept(F.copy(channel_sums(layer,scales))),
    Tiled.listed(F.div_scalar(F.mul(gamma,deviation),U32.to_f32(layer_values(layer)))),gamma_kept,pairs)

def back_summed(+layer: Layer,deviation: Tensor(),gamma: Tensor(),result: Cells.Pairs() & (List<&2,Tile.Tile8> & List<&2,Tile.Tile8>)) -> Backward():
  (pairs,sums) = result
  back_first(layer,deviation,F.copy(gamma),sums,pairs)

def back_cells(+layer: Layer,deviation: Tensor(),gamma: Tensor(),dy: Tensor(),xhat: Tensor()) -> Backward():
  back_summed(layer,deviation,gamma,Cells.gradient_sums(cell_shape(layer),layer_workers(layer),pieces_of(dy),pieces_of(xhat)))

def back_sized(empty: Bool,+images: U32,+channels: U32,+height: U32,+width: U32,+workers: U32,dy: Tensor(),xhat: Tensor(),deviation: Tensor(),gamma: Tensor()) -> Backward():
  match empty:
    case False{}: back_cells(Layer{images,channels,height,width,workers},deviation,gamma,innermost(dy),innermost(xhat))
    case True{}: backward_failed(Tensors.Problem{"batch_norm",Tensors.InvalidShape{[images,channels,height,width]}},[images,channels,height,width],workers,gamma)

def back_shaped(+shape: List<&2,U32>,+workers: U32,dy: Tensor(),xhat: Tensor(),deviation: Tensor(),gamma: Tensor()) -> Backward():
  match shape:
    case [+images,+channels,+height,+width]:
      back_sized(Shapes.has_zero([images,channels,height,width]),images,channels,height,width,workers,dy,xhat,deviation,gamma)
    case shape: backward_failed(Tensors.Problem{"batch_norm",Tensors.InvalidShape{shape}},shape,workers,gamma)

def back_same(same: Bool,+shape: List<&2,U32>,+other: List<&2,U32>,+workers: U32,dy: Tensor(),xhat: Tensor(),deviation: Tensor(),gamma: Tensor()) -> Backward():
  match same:
    case True{}: back_shaped(shape,workers,dy,xhat,deviation,gamma)
    case False{}: backward_failed(Tensors.Problem{"batch_norm",Tensors.ShapeMismatch{shape,other}},shape,workers,gamma)

def back_saved(+status: Tensors.Status,+shape: List<&2,U32>,+other: List<&2,U32>,+workers: U32,dy: Tensor(),xhat: Tensor(),deviation: Tensor(),gamma: Tensor()) -> Backward():
  match status:
    case Tensors.Usable{}: back_same(Shapes.same_extents(shape,other),shape,other,workers,dy,xhat,deviation,gamma)
    case Tensors.Failed{+problem}: backward_failed(problem,shape,workers,gamma)

def back_status(+status: Tensors.Status,+shape: List<&2,U32>,+workers: U32,dy: Tensor(),xhat: Tensor(),deviation: Tensor(),gamma: Tensor()) -> Backward():
  match status:
    case Tensors.Usable{}:
      Tiled.with_metadata(~Backward(),Tiled.along(xhat,1n),xhat => other => other_status => other_workers => back_saved(other_status,shape,other,workers,dy,xhat,deviation,gamma))
    case Tensors.Failed{+problem}: backward_failed(problem,shape,workers,gamma)

# dy is the gradient of the loss with respect to the output. With
# dbeta = sum_axes(dy,[0,2,3]) and dgamma = sum_axes(dy * xhat,[0,2,3]):
# dx = ((dy * M - dbeta) - xhat * dgamma) * ((gamma * deviation) / M), where
# deviation is the saved 1 / sqrt(variance + epsilon).
def backward(dy: Tensor(),saved: Saved(),gamma: Tensor()) -> Backward():
  (xhat,deviation) = saved
  Tiled.with_metadata(~Backward(),Tiled.along(dy,1n),dy => shape => status => workers => back_status(status,shape,workers,dy,xhat,deviation,gamma))

# ---- Evaluation ----

# (the output, (parameters, running statistics)).
def Evaluated() -> Type:
  Tensor() & (Parameters() & Running())

def evaluate_failed(+problem: Tensors.Problem,+shape: List<&2,U32>,+workers: U32,parameters: Parameters(),running: Running()) -> Evaluated():
  (Tiled.failed_as(problem,shape,workers),(parameters,running))

def evaluate_grouped(+layer: Layer,x: Pieces.Pieces(~Tile.Tile8),parameters: Parameters(),running: Running(),groups: Result<&2,&2,Tensors.Problem,List<&2,Kernel.Normal>>) -> Evaluated():
  match groups:
    case Done{groups}: (layer_tensor(layer,Cells.evaluated(groups,layer_groups(layer),layer_padding(layer),cell_shape(layer),layer_workers(layer),x)),(parameters,running))
    case Fail{+problem}: evaluate_failed(problem,layer_shape(layer),layer_workers(layer),parameters,running)

def evaluate_read(+layer: Layer,+epsilon: F32,x: Pieces.Pieces(~Tile.Tile8),gamma: Tensor() & Listed(),beta: Tensor() & Listed(),mean: Tensor() & Listed(),variance: Tensor() & Tensor()) ->
  Evaluated():
  Layer{images,+channels,height,width,workers} = layer
  (gamma,gammas) = gamma
  (beta,betas) = beta
  (mean,means) = mean
  (variance,variance_kept) = variance
  evaluate_grouped(layer,x,(gamma,beta),(mean,variance_kept),normal_tiles(means,Tiled.listed(inverse_deviation(variance,channels,epsilon)),gammas,betas))

def evaluate_cells(+layer: Layer,+epsilon: F32,parameters: Parameters(),running: Running(),x: Tensor()) -> Evaluated():
  Layer{images,+channels,height,width,workers} = layer
  (gamma,beta) = parameters
  (running_mean,running_variance) = running
  evaluate_read(layer,epsilon,pieces_of(x),parameter_kept(channels,F.copy(gamma)),parameter_kept(channels,F.copy(beta)),read_kept(F.copy(running_mean)),F.copy(running_variance))

def evaluate_sized(empty: Bool,+images: U32,+channels: U32,+height: U32,+width: U32,+workers: U32,+epsilon: F32,x: Tensor(),parameters: Parameters(),running: Running()) -> Evaluated():
  match empty:
    case False{}: evaluate_cells(Layer{images,channels,height,width,workers},epsilon,parameters,running,innermost(x))
    case True{}: evaluate_failed(Tensors.Problem{"batch_norm",Tensors.InvalidShape{[images,channels,height,width]}},[images,channels,height,width],workers,parameters,running)

def evaluate_shaped(+settings: Settings,+shape: List<&2,U32>,+workers: U32,x: Tensor(),parameters: Parameters(),running: Running()) -> Evaluated():
  Settings{+epsilon,momentum} = settings
  match shape:
    case [+images,+channels,+height,+width]:
      evaluate_sized(Shapes.has_zero([images,channels,height,width]),images,channels,height,width,workers,epsilon,x,parameters,running)
    case shape: evaluate_failed(Tensors.Problem{"batch_norm",Tensors.InvalidShape{shape}},shape,workers,parameters,running)

def evaluate_status(+settings: Settings,+status: Tensors.Status,+shape: List<&2,U32>,+workers: U32,x: Tensor(),parameters: Parameters(),running: Running()) -> Evaluated():
  match status:
    case Tensors.Usable{}: evaluate_shaped(settings,shape,workers,x,parameters,running)
    case Tensors.Failed{+problem}: evaluate_failed(problem,shape,workers,parameters,running)

# The output with the running statistics:
# (x - running mean) * (1 / sqrt(running variance + epsilon)) * gamma + beta.
def evaluate(x: Tensor(),parameters: Parameters(),running: Running(),+settings: Settings) -> Evaluated():
  Tiled.with_metadata(~Evaluated(),Tiled.along(x,1n),x => shape => status => workers => evaluate_status(settings,status,shape,workers,x,parameters,running))
