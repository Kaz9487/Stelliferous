# Shape validation and affine tensor ownership for the shared reduction kernel.
import Base
import ./tensor.bend as Tensors
import ./storage_buffer.bend as Storage
import ./kernel_parameter_gradient.bend as Kernel
import ./kernel_parameter_gradient_buffer.bend as KernelBuffer
import ./kernel_weight_gradient.bend as WeightKernel
import ./kernel_shape.bend as Shape

type ImageGeometry is Data:
  ImageGeometry{images: U32,channels: U32,height: U32,width: U32}

def image(shape: List<&2,U32>) -> Maybe<&2,ImageGeometry>:
  match shape:
    case [n,c,h,w]: Some{ImageGeometry{n,c,h,w}}
    case [c,h,w]: Some{ImageGeometry{1,c,h,w}}
    case shape: None{}

def positive(image: ImageGeometry) -> Bool:
  ImageGeometry{n,c,h,w} = image
  U32.is_lt(0,n) && U32.is_lt(0,c) && U32.is_lt(0,h) && U32.is_lt(0,w)

def work(x: ImageGeometry,d: ImageGeometry,kernel: U32,stride: U32,padding: U32,group: U32) -> Kernel.Work:
  ImageGeometry{n,c,h,w} = x
  ImageGeometry{dn,o,rows,columns} = d
  Kernel.Work{U32.to_nat(n),U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(group)}

def fits(count: Maybe<&2,U32>) -> Bool:
  match count:
    case None{}: False{}
    case Some{count}: True{}

def valid(+x: ImageGeometry,+d: ImageGeometry,+kernel: U32,+stride: U32,+padding: U32,group: U32) -> Bool:
  ImageGeometry{+n,+c,+h,+w} = x
  ImageGeometry{dn,+o,rows,columns} = d
  positive(x) && positive(d) && U32.is_eq(n,dn) && U32.is_lt(0,kernel) && U32.is_lt(0,stride) && U32.is_lt(0,group) &&
    Nat.is_le(U32.to_nat(kernel),Nat.add(U32.to_nat(h),Nat.mul(2n,U32.to_nat(padding)))) &&
    Nat.is_le(U32.to_nat(kernel),Nat.add(U32.to_nat(w),Nat.mul(2n,U32.to_nat(padding)))) &&
    Nat.is_eq(U32.to_nat(rows),Shape.axis_size(U32.to_nat(h),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding))) &&
    Nat.is_eq(U32.to_nat(columns),Shape.axis_size(U32.to_nat(w),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding))) && fits(Tensors.count_of([o,c,kernel,kernel]))

def weight_finished(+xs: List<&2,U32>,xf: F32,xs_status: Tensors.Status,+xw: U32,+ds: List<&2,U32>,df: F32,ds_status: Tensors.Status,+dw: U32,
  shape: List<&2,U32>,result: (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>) -> Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>):
  ((x,d),values) = result
  (Tensors.whole(~F32,values,shape,0.0,Tensors.Usable{},U32.max(xw,dw)),
    (Tensors.whole(~F32,x,xs,xf,xs_status,xw),Tensors.whole(~F32,d,ds,df,ds_status,dw)))

def weight_selected(accepted: Bool,+xi: ImageGeometry,+di: ImageGeometry,+k: U32,s: U32,p: U32,g: U32,
  x: Storage.Buffer<F32>,+xs: List<&2,U32>,xf: F32,xs_status: Tensors.Status,+xw: U32,
  d: Storage.Buffer<F32>,ds: List<&2,U32>,df: F32,ds_status: Tensors.Status,+dw: U32) -> Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>):
  match accepted:
    case False{}:
      (Tensors.failed(~F32,"conv2d_weight_gradient",Tensors.InvalidArgument{xs},[],0.0,xw),
        (Tensors.whole(~F32,x,xs,xf,xs_status,xw),Tensors.whole(~F32,d,ds,df,ds_status,dw)))
    case True{}:
      ImageGeometry{n,c,h,w} = xi
      ImageGeometry{dn,o,rows,columns} = di
      weight_finished(xs,xf,xs_status,xw,ds,df,ds_status,dw,[o,c,k,k],WeightKernel.weight_parts(work(xi,di,k,s,p,g),U32.max(xw,dw),x,d))

# The statuses are matched, not tested and kept: a status used twice would
# make the constructors of the whole call shared.
def weight_shaped(xi: Maybe<&2,ImageGeometry>,di: Maybe<&2,ImageGeometry>,xs_status: Tensors.Status,ds_status: Tensors.Status,+k: U32,+s: U32,+p: U32,+g: U32,
  x: Storage.Buffer<F32>,+xs: List<&2,U32>,xf: F32,+xw: U32,
  d: Storage.Buffer<F32>,ds: List<&2,U32>,df: F32,dw: U32) -> Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>):
  match xi di xs_status ds_status:
    case Some{+xi} Some{+di} Tensors.Usable{} Tensors.Usable{}:
      weight_selected(valid(xi,di,k,s,p,g),xi,di,k,s,p,g,x,xs,xf,Tensors.Usable{},xw,d,ds,df,Tensors.Usable{},dw)
    case xi di xs_status ds_status:
      weight_selected(False{},ImageGeometry{0,0,0,0},ImageGeometry{0,0,0,0},k,s,p,g,x,xs,xf,xs_status,xw,d,ds,df,ds_status,dw)

def weight(x: Tensors.Tensor<F32>,d: Tensors.Tensor<F32>,+k: U32,+s: U32,+p: U32,+g: U32) -> Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>):
  Tensors.with_whole(~F32,~(Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>)),x,x => xs => xf => status => xw =>
    +xs = xs
    Tensors.with_whole(~F32,~(Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>)),d,d => ds => df => d_status => dw =>
      +ds = ds
      weight_shaped(image(xs),image(ds),status,d_status,k,s,p,g,x,xs,xf,xw,d,ds,df,dw)))

def bias_work(d: ImageGeometry,group: U32) -> Kernel.Work:
  ImageGeometry{n,o,rows,columns} = d
  Kernel.Work{U32.to_nat(n),0n,U32.to_nat(o),0n,0n,1n,1n,0n,U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(group)}

def bias_finished(+shape: List<&2,U32>,fill: F32,status: Tensors.Status,+workers: U32,+outputs: U32,result: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Tensors.Tensor<F32> & Tensors.Tensor<F32>:
  (d,values) = result
  (Tensors.whole(~F32,values,[outputs],0.0,Tensors.Usable{},workers),Tensors.whole(~F32,d,shape,fill,status,workers))

# One output per channel, written into storage of the count of [o].
def bias_counted(count: Maybe<&2,U32>,+di: ImageGeometry,+g: U32,d: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,status: Tensors.Status,+workers: U32) ->
  Tensors.Tensor<F32> & Tensors.Tensor<F32>:
  match count:
    case Some{+count}:
      ImageGeometry{n,+o,rows,columns} = di
      bias_finished(shape,fill,status,workers,o,KernelBuffer.bias(bias_work(ImageGeometry{n,o,rows,columns},g),d,Tensors.buffer(~F32,0.0,count)))
    case None{}:
      ImageGeometry{n,o,rows,columns} = di
      (Tensors.failed(~F32,"conv2d_bias_gradient",Tensors.InvalidShape{[o]},[o],0.0,workers),Tensors.whole(~F32,d,shape,fill,status,workers))

def bias_selected(accepted: Bool,+di: ImageGeometry,g: U32,d: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,status: Tensors.Status,+workers: U32) -> Tensors.Tensor<F32> & Tensors.Tensor<F32>:
  match accepted:
    case False{}: (Tensors.failed(~F32,"conv2d_bias_gradient",Tensors.InvalidArgument{shape},[],0.0,workers),Tensors.whole(~F32,d,shape,fill,status,workers))
    case True{}:
      ImageGeometry{n,+o,rows,columns} = di
      bias_counted(Tensors.count_of([o]),ImageGeometry{n,o,rows,columns},g,d,shape,fill,status,workers)

def bias_shaped(di: Maybe<&2,ImageGeometry>,+g: U32,d: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,+status: Tensors.Status,workers: U32) -> Tensors.Tensor<F32> & Tensors.Tensor<F32>:
  match di:
    case None{}: bias_selected(False{},ImageGeometry{0,0,0,0},g,d,shape,fill,status,workers)
    case Some{+di}: bias_selected(Tensors.usable(status) && positive(di) && U32.is_lt(0,g),di,g,d,shape,fill,status,workers)

def bias(d: Tensors.Tensor<F32>,+g: U32) -> Tensors.Tensor<F32> & Tensors.Tensor<F32>:
  Tensors.with_whole(~F32,~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),d,d => shape => fill => status => workers =>
    +shape = shape
    bias_shaped(image(shape),g,d,shape,fill,status,workers))
