import Base
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/proofs/kernel_spatial_packing_samples.bend as SpatialSamples
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/kernel_packing_samples.bend as PackingSamples
import ../../lib/kernel_gemm.bend as Gemm

# The stencil of a square kernel over values read from zero.
def window(height: U32,width: U32,+kernel: U32,stride: U32,+padding: U32,output_width: U32,+inner: U32) -> Packing.Stencil:
  Packing.square(height,width,kernel,stride,padding,output_width,inner,Gemm.blocks((inner / (kernel * kernel) : U32)))

def value(+x: Tree.Tree<F32>,+pos: U32,+q: U32,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32) -> F32:
  PackingSamples.masked(Tree.read(F32,x,
      ((q / (k * k)) * h * w + ((pos / output_width) * stride + (q / k) % k - pad : U32) * w + ((pos % output_width) * stride + q % k - pad : U32) : U32)),
    U32.is_lt(((pos / output_width) * stride + (q / k) % k - pad : U32),h) && U32.is_lt(((pos % output_width) * stride + q % k - pad : U32),w))

def pixel(+x: Tree.Tree<F32>,+pos: U32,+q: U32,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,
  +output_width: U32) -> {PackingSamples.pixel(Tree.pack(F32,x),pos,q,h,w,k,stride,pad,output_width) == (Tree.pack(F32,x),value(x,pos,q,h,w,k,stride,pad,
      output_width)) : Array<F32> & F32}:
  %Equal.sym(Array<F32> & F32,Array.get(F32,Tree.pack(F32,x),
      ((q / (k * k)) * h * w + ((pos / output_width) * stride + (q / k) % k - pad : U32) * w + ((pos % output_width) * stride + q % k - pad : U32) : U32)),
    (Tree.pack(F32,x),Tree.read(F32,x,
        ((q / (k * k)) * h * w + ((pos / output_width) * stride + (q / k) % k - pad : U32) * w + ((pos % output_width) * stride + q % k - pad : U32) : U32))),
    StorageLaws.read(F32,x,
      ((q / (k * k)) * h * w + ((pos / output_width) * stride + (q / k) % k - pad : U32) * w + ((pos % output_width) * stride + q % k - pad : U32) :
        U32))) : {PackingSamples.mask_pixel(U32.is_lt(((pos / output_width) * stride + (q / k) % k - pad : U32),h) && U32.is_lt(((pos % output_width) * stride + q % k - pad :
        U32),w),_) == (Tree.pack(F32,x),value(x,pos,q,h,w,k,stride,pad,output_width)) : Array<F32> & F32}
  {==}

def address(+row: U32,+column: U32,channel: U32,+height: U32,+width: U32) -> PackingSamples.PixelAddress:
  PackingSamples.PixelAddress{(channel + (row * width + column : U32) * 1 : U32),U32.is_lt(row,height) && U32.is_lt(column,width)}

# The pixel of a channel plane as the sample index of a value.
def plane_index(+channel: U32,+height: U32,+width: U32,+row: U32,+column: U32) ->
  {(channel * (height * width : U32) + (row * width + column : U32) * 1 : U32) == (channel * height * width + row * width + column : U32) : U32}:
  +offset = (row * width + column : U32)
  +wanted = (channel * height * width + row * width + column : U32)
  Equal.trans(U32,(channel * (height * width : U32) + offset * 1 : U32),(channel * (height * width : U32) + offset : U32),wanted,
    Equal.cong(U32,U32,tail => (channel * (height * width : U32) + tail : U32),(offset * 1 : U32),offset,Arithmetic.multiply_one(offset)),
    Equal.trans(U32,(channel * (height * width : U32) + offset : U32),(channel * height * width + offset : U32),wanted,
      Equal.cong(U32,U32,head => (head + offset : U32),(channel * (height * width : U32) : U32),(channel * height * width : U32),
        Equal.sym(U32,(channel * height * width : U32),(channel * (height * width : U32) : U32),Arithmetic.multiply_associative(channel,height,width))),
      Equal.sym(U32,wanted,(channel * height * width + offset : U32),Arithmetic.add_associative((channel * height * width : U32),(row * width : U32),column))))

def square_address(+row: U32,+column: U32,+channel: U32,+height: U32,+width: U32,+padding: U32) ->
  {address((row + (0 - padding : U32) : U32),(column + (0 - padding : U32) : U32),(0 + channel : U32),height,width) ==
    address((row - padding : U32),(column - padding : U32),channel,height,width) : PackingSamples.PixelAddress}:
  Equal.trans(PackingSamples.PixelAddress,address((row + (0 - padding : U32) : U32),(column + (0 - padding : U32) : U32),(0 + channel : U32),height,width),
    address((row - padding : U32),(column + (0 - padding : U32) : U32),(0 + channel : U32),height,width),address((row - padding : U32),(column - padding : U32),channel,height,width),
    Equal.cong(U32,PackingSamples.PixelAddress,r => address(r,(column + (0 - padding : U32) : U32),(0 + channel : U32),height,width),(row + (0 - padding : U32) : U32),(row - padding : U32),Arithmetic.add_zero_subtract(row,padding)),
    Equal.trans(PackingSamples.PixelAddress,address((row - padding : U32),(column + (0 - padding : U32) : U32),(0 + channel : U32),height,width),
      address((row - padding : U32),(column - padding : U32),(0 + channel : U32),height,width),address((row - padding : U32),(column - padding : U32),channel,height,width),
      Equal.cong(U32,PackingSamples.PixelAddress,c => address((row - padding : U32),c,(0 + channel : U32),height,width),(column + (0 - padding : U32) : U32),(column - padding : U32),Arithmetic.add_zero_subtract(column,padding)),
      Equal.cong(U32,PackingSamples.PixelAddress,c => address((row - padding : U32),(column - padding : U32),c,height,width),(0 + channel : U32),channel,Arithmetic.add_zero_left(channel))))

def square_sample(+input: Tree.Tree<F32>,+position: U32,+reduction: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_width: U32,+inner: U32,+index: Nat) ->
  {SpatialSamples.window_sample(~F32,~Packing.identity,input,SpatialSamples.window_context(~SpatialSamples.planes,~SpatialSamples.adjacent,window(height,width,kernel,stride,padding,output_width,inner),position,reduction),index) ==
    value(input,(position + U32.from_nat(index) : U32),reduction,height,width,kernel,stride,padding,output_width) : F32}:
  +sample_position = (position + U32.from_nat(index) : U32)
  +row = (sample_position / output_width * stride + reduction / kernel % kernel : U32)
  +column = (sample_position % output_width * stride + reduction % kernel : U32)
  +plane_channel = (reduction / (kernel * kernel) : U32)
  +channel = (plane_channel * (height * width : U32) : U32)
  +inside = (U32.is_lt((row - padding : U32),height) && U32.is_lt((column - padding : U32),width))
  Equal.trans(F32,SpatialSamples.address_sample(~F32,~Packing.identity,input,address((row + (0 - padding : U32) : U32),(column + (0 - padding : U32) : U32),(0 + channel : U32),height,width)),
    SpatialSamples.address_sample(~F32,~Packing.identity,input,address((row - padding : U32),(column - padding : U32),channel,height,width)),
    value(input,sample_position,reduction,height,width,kernel,stride,padding,output_width),
    Equal.cong(PackingSamples.PixelAddress,F32,pixel => SpatialSamples.address_sample(~F32,~Packing.identity,input,pixel),
      address((row + (0 - padding : U32) : U32),(column + (0 - padding : U32) : U32),(0 + channel : U32),height,width),
      address((row - padding : U32),(column - padding : U32),channel,height,width),square_address(row,column,channel,height,width,padding)),
    Equal.cong(U32,F32,index => SpatialSamples.address_sample(~F32,~Packing.identity,input,PackingSamples.PixelAddress{index,inside}),
      (channel + ((row - padding : U32) * width + (column - padding : U32) : U32) * 1 : U32),
      (plane_channel * height * width + (row - padding : U32) * width + (column - padding : U32) : U32),
      plane_index(plane_channel,height,width,(row - padding : U32),(column - padding : U32))))

def square_first(+input: Tree.Tree<F32>,+position: U32,+reduction: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_width: U32,+inner: U32) ->
  {SpatialSamples.window_value(~F32,~Packing.identity,~SpatialSamples.planes,~SpatialSamples.adjacent,input,window(height,width,kernel,stride,padding,output_width,inner),position,reduction) == value(input,position,reduction,height,width,kernel,stride,padding,output_width) : F32}:
  Equal.trans(F32,SpatialSamples.window_value(~F32,~Packing.identity,~SpatialSamples.planes,~SpatialSamples.adjacent,input,window(height,width,kernel,stride,padding,output_width,inner),position,reduction),value(input,(position + 0 : U32),reduction,height,width,kernel,stride,padding,output_width),value(input,position,reduction,height,width,kernel,stride,padding,output_width),
    square_sample(input,position,reduction,height,width,kernel,stride,padding,output_width,inner,0n),
    Equal.cong(U32,F32,pos => value(input,pos,reduction,height,width,kernel,stride,padding,output_width),(position + 0 : U32),position,Arithmetic.add_zero(position)))
