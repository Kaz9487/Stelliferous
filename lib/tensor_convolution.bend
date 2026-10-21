# CHW / NCHW entry. Geometry and allocation checks finish before planning.
# Every convolution runs on row bands: the input arrives in the layout the
# planner chooses for it and the band pipeline runs there.
import Base
import ./tensor.bend as Tensors
import ./storage_buffer.bend as Storage
import ./kernel_shape.bend as KernelShape
import ./kernel_gemm_buffer.bend as GemmBuffer
import ./kernel_tile.bend as Tile
import ./kernel_planning.bend as KernelPlanning
import ./default_kernel_cost.bend as DefaultCost
import ./default_machine_limits.bend as Default
import ./machine_limits.bend as Machine
import ./proofs/storage_certificate.bend as Certificate
import ./tensor_band.bend as Bands
import ./tensor_band_window.bend as BandWindow
import ./kernel_band_pipeline.bend as BandPipeline
import ./kernel_band_planning.bend as BandPlanning
import ./tensor_band_schedule.bend as BandSchedule
import ./math_activation.bend as Activations

type InputShape is Data:
  InputShape{batch: U32,batched: Bool,channels: U32,height: U32,width: U32}

type Geometry is Data:
  Geometry{batch: U32,batched: Bool,dimensions: KernelShape.Dimensions}

def input_shape(shape: List<&2,U32>) -> Maybe<&2,InputShape>:
  match shape:
    case [channels,height,width]: Some{InputShape{1,False{},channels,height,width}}
    case [batch,channels,height,width]: Some{InputShape{batch,True{},channels,height,width}}
    case shape: None{}

def matched(matches: Bool,geometry: Geometry) -> Maybe<&2,Geometry>:
  match matches:
    case True{}: Some{geometry}
    case False{}: None{}

def geometry(input: Maybe<&2,InputShape>,weights: List<&2,U32>,bias: List<&2,U32>,stride: U32,padding: U32) -> Maybe<&2,Geometry>:
  match input weights bias:
    case Some{InputShape{batch,batched,+channels,height,width}} Con{+rows,Con{weight_channels,Con{+kernel,Con{kernel_width,Nil{}}}}} Con{bias_channels,Nil{}}:
      matched(U32.is_eq(channels,weight_channels) && U32.is_eq(kernel,kernel_width) && U32.is_eq(rows,bias_channels),
        Geometry{batch,batched,KernelShape.Dimensions{U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)}})
    case input weights bias: None{}

def output_shape(batched: Bool,batch: U32,rows: U32,height: U32,width: U32) -> List<&2,U32>:
  match batched:
    case False{}: [rows,height,width]
    case True{}: [batch,rows,height,width]

def band_context(geometry: Geometry,parallel: Bool,+activation: Activations.Activation) -> BandPipeline.Context:
  Geometry{count,batched,KernelShape.Dimensions{+ci,rows,+height,+width,+kernel,+stride,+padding}} = geometry
  BandPipeline.Context{BandWindow.Geometry{Bands.Geometry{count,U32.from_nat(ci),U32.from_nat(height),U32.from_nat(width)},U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
    U32.from_nat(KernelShape.axis_size(height,kernel,stride,padding)),U32.from_nat(KernelShape.axis_size(width,kernel,stride,padding))},U32.from_nat(rows),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),parallel,activation}

def band_work(context: BandPipeline.Context,bias_capacity: U32) -> KernelPlanning.Work:
  BandPipeline.Context{BandWindow.Geometry{Bands.Geometry{batch,+channels,+height,+width},kernel,stride,padding,oh,+ow},rows,inner,parallel,activation} = context
  KernelPlanning.Work{(oh * ow : U32),inner,rows,KernelPlanning.Owners{U32.from_nat(KernelPlanning.capacity((channels * height * width : U32))),bias_capacity},
    KernelPlanning.Geometry{channels,height,width,kernel,stride,padding,ow},activation}

def context_window(context: BandPipeline.Context) -> BandWindow.Geometry:
  BandPipeline.Context{window,rows,inner,parallel,activation} = context
  window

def band_output(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: Geometry,parallel: Bool,input: Bands.Bands(~F32),weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,workers: U32) -> Tensors.Tensor<F32>:
  Geometry{+count,batched,KernelShape.Dimensions{ci,+rows,height,width,+kernel,+stride,+padding}} = geometry
  +oh = U32.from_nat(KernelShape.axis_size(height,kernel,stride,padding))
  +ow = U32.from_nat(KernelShape.axis_size(width,kernel,stride,padding))
  Tensors.Tensor{BandPipeline.execute(~custom,band_context(geometry,parallel,activation),input,weights,bias),
    Bands.Geometry{count,U32.from_nat(rows),oh,ow},output_shape(batched,count,U32.from_nat(rows),oh,ow),0.0,Tensors.Usable{},workers}

# The weights in the order the kernel reads them: once, after the checks.
def reordered(dimensions: KernelShape.Dimensions,weights: Storage.Buffer<F32>) -> Storage.Buffer<Tile.Tile8>:
  KernelShape.Dimensions{ci,rows,h,w,+k,s,p} = dimensions
  GemmBuffer.reorder(U32.from_nat(rows),U32.from_nat(Nat.mul(ci,Nat.mul(k,k))),weights)

def packed(geometry: Geometry,weights: Storage.Buffer<F32>) -> Storage.Buffer<Tile.Tile8>:
  Geometry{count,batched,dimensions} = geometry
  reordered(dimensions,weights)

def machine_support(geometry: Geometry,+limits: Machine.Limits) -> Bool:
  Geometry{+count,batched,+dimensions} = geometry
  KernelShape.Dimensions{+ci,+rows,+h,+w,+k,+s,+p} = dimensions
  KernelShape.supports_packed(limits,dimensions,Nat.mul(ci,Nat.mul(h,w)),Nat.mul(rows,Nat.mul(ci,Nat.mul(k,k)))) &&
    Nat.is_le(Nat.mul(U32.to_nat(count),Nat.mul(rows,Nat.mul(KernelShape.axis_size(h,k,s,p),KernelShape.axis_size(w,k,s,p)))),U32.to_nat(Machine.capacity(limits)))

def input_geometry(geometry: Geometry) -> Bands.Geometry:
  Geometry{count,batched,KernelShape.Dimensions{ci,rows,h,w,k,s,p}} = geometry
  Bands.Geometry{count,U32.from_nat(ci),U32.from_nat(h),U32.from_nat(w)}

# Arrived bands cover every row of the images of the input.
def covered(+dimensions: KernelShape.Dimensions,+count: U32,inspection: Bands.Inspection) -> Bool:
  KernelShape.Dimensions{ci,rows,+h,w,k,s,p} = dimensions
  Bands.Inspection{valid,first,end,leaves} = inspection
  valid && U32.is_zero(first) && Nat.is_eq(U32.to_nat(end),Nat.mul(U32.to_nat(count),h))

def arrived_covered(~custom: F32 -> F32,+activation: Activations.Activation,covered: Bool,+geometry: Geometry,parallel: Bool,input: Bands.Bands(~F32),
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,workers: U32) -> Tensors.Tensor<F32>:
  match covered:
    case True{}: band_output(~custom,activation,geometry,parallel,input,weights,bias,workers)
    case False{}: Tensors.Tensor{input,input_geometry(geometry),shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidArgument{shape}}},workers}

def arrived(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: Geometry,parallel: Bool,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  +shape: List<&2,U32>,fill: F32,workers: U32,inspected: Bands.Bands(~F32) & Bands.Inspection) -> Tensors.Tensor<F32>:
  Geometry{+count,batched,+dimensions} = geometry
  (input,inspection) = inspected
  arrived_covered(~custom,activation,covered(dimensions,count,inspection),Geometry{count,batched,dimensions},parallel,input,weights,bias,shape,fill,workers)

def selected(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: Geometry,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  +shape: List<&2,U32>,fill: F32,workers: U32,tensor: Tensors.Tensor<F32>,selection: BandPlanning.LayoutPlan) -> Tensors.Tensor<F32>:
  BandPlanning.LayoutPlan{layout,parallel,time,repartition} = selection
  +input = input_geometry(geometry)
  arrived(~custom,activation,geometry,parallel,weights,bias,shape,fill,workers,Bands.inspect(~F32,input,Tensors.arrive(~F32,input,layout,tensor)))

# The planner chooses the layout again from the input's plan when the input
# holds the input geometry, otherwise it chooses the first split.
def offered(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: Geometry,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  +shape: List<&2,U32>,fill: F32,+workers: U32,owners: Tensors.Tensor<F32> & BandSchedule.Target) -> Tensors.Tensor<F32>:
  Storage.Buffer{bias,length,Certificate.Certificate{+depth,equation}} = bias
  (tensor,target) = owners
  BandSchedule.Target{other,+previous,split} = target
  +context = band_context(geometry,True{},activation)
  selected(~custom,activation,geometry,weights,Storage.Buffer{bias,length,Certificate.Certificate{depth,equation}},shape,fill,workers,tensor,
    BandPlanning.layout_for(Bands.same_geometry(input_geometry(geometry),other),previous,band_work(context,U32.shln(1,depth)),context_window(context),workers,
      DefaultCost.configuration()))

# A shape the machine cannot hold.
def unsupported(tensor: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  Tensors.Tensor{bands,geometry,+shape,fill,status,workers} = tensor
  Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidShape{shape}}},workers}

def band_machine(~custom: F32 -> F32,+activation: Activations.Activation,supported: Bool,+geometry: Geometry,tensor: Tensors.Tensor<F32>,
  weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,workers: U32) -> Tensors.Tensor<F32>:
  match supported:
    case True{}: offered(~custom,activation,geometry,packed(geometry,weights),bias,shape,fill,workers,Tensors.offer(~F32,tensor))
    case False{}: unsupported(tensor)

def band_supported(~custom: F32 -> F32,+activation: Activations.Activation,valid: Bool,+geometry: Geometry,tensor: Tensors.Tensor<F32>,
  weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,workers: U32) -> Tensors.Tensor<F32>:
  match valid:
    case False{}: Tensors.rejected(~F32,"conv2d",tensor)
    case True{}:
      band_machine(~custom,activation,machine_support(geometry,Default.limits()),geometry,tensor,weights,bias,shape,fill,workers)

# What a call needs of its parameters: lengths that fit the shape and the
# capacities that hold them.
def parameters_valid(+dimensions: KernelShape.Dimensions,+weight_length: Nat,+weight_depth: Nat,+bias_length: Nat,+bias_depth: Nat) -> Bool:
  KernelShape.Dimensions{+ci,rows,+h,+w,k,s,p} = dimensions
  KernelShape.valid(dimensions,Nat.mul(ci,Nat.mul(h,w)),weight_length,bias_length) &&
    Nat.is_le(weight_length,U32.to_nat(U32.shln(1,weight_depth))) && Nat.is_le(bias_length,U32.to_nat(U32.shln(1,bias_depth)))

def band_checked(~custom: F32 -> F32,+activation: Activations.Activation,geometry: Maybe<&2,Geometry>,tensor: Tensors.Tensor<F32>,
  weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,workers: U32) -> Tensors.Tensor<F32>:
  match geometry:
    case None{}: Tensors.rejected(~F32,"conv2d",tensor)
    case Some{+geometry}:
      Geometry{count,batched,+dimensions} = geometry
      Storage.Buffer{weights,+weight_length,Certificate.Certificate{+weight_depth,weight_equation}} = weights
      Storage.Buffer{bias,+bias_length,Certificate.Certificate{+bias_depth,bias_equation}} = bias
      band_supported(~custom,activation,parameters_valid(dimensions,weight_length,weight_depth,bias_length,bias_depth),geometry,tensor,
        Storage.Buffer{weights,weight_length,Certificate.Certificate{weight_depth,weight_equation}},
        Storage.Buffer{bias,bias_length,Certificate.Certificate{bias_depth,bias_equation}},shape,fill,workers)

def band_status(~custom: F32 -> F32,+activation: Activations.Activation,status: Tensors.Status,layout: Bands.Geometry,input: Bands.Bands(~F32),+shape: List<&2,U32>,+fill: F32,
  weights: Storage.Buffer<F32>,weight_shape: List<&2,U32>,bias: Storage.Buffer<F32>,bias_shape: List<&2,U32>,stride: U32,padding: U32,+workers: U32) -> Tensors.Tensor<F32>:
  match status:
    case Tensors.Failed{problem}: Tensors.Tensor{input,layout,shape,fill,Tensors.Failed{problem},workers}
    case Tensors.Usable{}:
      band_checked(~custom,activation,geometry(input_shape(shape),weight_shape,bias_shape,stride,padding),Tensors.Tensor{input,layout,shape,fill,Tensors.Usable{},workers},
        weights,bias,shape,fill,workers)

def input_layout(~custom: F32 -> F32,+activation: Activations.Activation,input: Tensors.Tensor<F32>,weights: Storage.Buffer<F32>,weight_shape: List<&2,U32>,bias: Storage.Buffer<F32>,bias_shape: List<&2,U32>,
  parameter_status: Tensors.Status,parameter_workers: U32,stride: U32,padding: U32) -> Tensors.Tensor<F32>:
  Tensors.Tensor{input,layout,+shape,fill,input_status,workers} = input
  band_status(~custom,activation,Tensors.first_failure(input_status,parameter_status),layout,input,shape,fill,weights,weight_shape,bias,bias_shape,stride,padding,
    U32.max(workers,parameter_workers))

# Weights and bias are read whole, so band parameters are gathered first.
def conv2d(~custom: F32 -> F32,+activation: Activations.Activation,input: Tensors.Tensor<F32>,weights: Tensors.Tensor<F32>,bias: Tensors.Tensor<F32>,stride: U32,padding: U32) -> Tensors.Tensor<F32>:
  Tensors.with_whole(~F32,~Tensors.Tensor<F32>,weights,weights => weight_shape => weight_fill => weight_status => weight_workers =>
    Tensors.with_whole(~F32,~Tensors.Tensor<F32>,bias,bias => bias_shape => bias_fill => bias_status => bias_workers =>
      input_layout(~custom,activation,input,weights,weight_shape,bias,bias_shape,Tensors.first_failure(weight_status,bias_status),U32.max(weight_workers,bias_workers),stride,padding)))
