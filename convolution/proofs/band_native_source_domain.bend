# Public convolution dimensions are decoded native words. Preserve their
# actual count envelope without deriving unrelated leaf narrowing bounds.
import Base
import ../specification.bend as ConvSpec
import ../../lib/tensor_band.bend as Bands
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ./convolution_guard_facts.bend as Guards
import ./band_leaf_caller.bend as Leaf

type Positive<-channels: Nat,-height: Nat,-width: Nat> is Data:
  Positive{channels_positive: {Nat.is_lt(0n,channels) == True{} : Bool},
    height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
    width_positive: {Nat.is_lt(0n,width) == True{} : Bool}}

def positive(+problem: ConvSpec.Problem,facts: Guards.Valid<problem>) ->
  Positive<Guards.input_channel_count(problem),Guards.height(problem),Guards.width(problem)>:
  Guards.Valid{ci,co,h,w,k,s,kh,kw,input,weights,bias} = facts
  Positive{ci,h,w}

def dimensions(+channels: U32,+output_channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,padding: U32) -> KernelShape.Dimensions:
  KernelShape.Dimensions{U32.to_nat(channels),U32.to_nat(output_channels),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)}

def geometry_roundtrip(+images: U32,+channels: U32,+output_channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32) ->
  {Leaf.input_geometry(images,dimensions(channels,output_channels,height,width,kernel,stride,padding)) == Bands.Geometry{images,channels,height,width} : Bands.Geometry}:
  %Equal.sym(U32,U32.from_nat(U32.to_nat(channels)),channels,Arithmetic.native_roundtrip(channels)) :
    {Bands.Geometry{images,_,U32.from_nat(U32.to_nat(height)),U32.from_nat(U32.to_nat(width))} == Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(height)),height,Arithmetic.native_roundtrip(height)) :
    {Bands.Geometry{images,channels,_,U32.from_nat(U32.to_nat(width))} == Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(width)),width,Arithmetic.native_roundtrip(width)) :
    {Bands.Geometry{images,channels,height,_} == Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  {==}

def converted(+before: Bands.Geometry,+after: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,
  exact: {before == after : Bands.Geometry},whole: BandGather.Domain<before,limits,capacity>) -> BandGather.Domain<after,limits,capacity>:
  %exact : BandGather.Domain<_,limits,capacity>
  whole
