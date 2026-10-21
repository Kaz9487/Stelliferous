# One phase of a band: whether it reads a tap for the outputs [first, last),
# its lowest tile and one past its highest. A band fold step keeps the extreme.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../traversal_array.bend as Traversal
import ./traversal_array_refinement.bend as Arrays
import ./traversal_extrema.bend as Folds
import ./input_gradient_phase_geometry.bend as Geometry

type BandSample is Data:
  BandSample{used: Bool,low: U32,high: U32}
def band_sample_from(+context: Kernel.Axis,+kernel: U32,+stride: U32,+first: U32,+last: U32) -> BandSample:
  Kernel.Axis{+start,+count,tap,+taps,+source} = context
  +low = Bool.pick(U32,U32.is_lt(start,first),((first - start + stride - 1) / stride : U32),0)
  +high = U32.min(count,Bool.pick(U32,U32.is_lt(start,last),((last - start + stride - 1) / stride : U32),0))
  BandSample{U32.is_lt(0,taps) && U32.is_lt(low,high),(low + source + kernel : U32),(high - 1 + source + kernel + taps : U32)}

def band_sample(context: Kernel.Band,index: U32) -> BandSample:
  Kernel.Band{+size,+kernel,+stride,+padding,first,last,what} = context
  band_sample_from(Kernel.axis(index,size,kernel,stride,padding),kernel,stride,first,last)
def band_sample_used(sample: BandSample) -> Bool:
  BandSample{used,low,high} = sample
  used
def band_sample_low(sample: BandSample) -> U32:
  BandSample{used,low,high} = sample
  low
def band_sample_high(sample: BandSample) -> U32:
  BandSample{used,low,high} = sample
  high
def band_used(context: Kernel.Band,index: U32) -> Bool:
  band_sample_used(band_sample(context,index))
def band_low(context: Kernel.Band,index: U32) -> U32:
  band_sample_low(band_sample(context,index))
def band_high(context: Kernel.Band,index: U32) -> U32:
  band_sample_high(band_sample(context,index))
def band_count(context: Kernel.Band,index: U32) -> U32:
  Kernel.Band{+size,+kernel,+stride,+padding,first,last,what} = context
  Geometry.axis_count(Kernel.axis(index,size,kernel,stride,padding))

def band_with_what(context: Kernel.Band,what: U32) -> Kernel.Band:
  Kernel.Band{size,kernel,stride,padding,first,last,old} = context
  Kernel.Band{size,kernel,stride,padding,first,last,what}
def lower_band_context(query: Folds.Query<Kernel.Band>) -> Kernel.Band:
  band_with_what(Folds.context(~Kernel.Band,query),0)
def higher_band_context(query: Folds.Query<Kernel.Band>) -> Kernel.Band:
  band_with_what(Folds.context(~Kernel.Band,query),1)
def count_band_context(query: Folds.Query<Kernel.Band>) -> Kernel.Band:
  band_with_what(Folds.context(~Kernel.Band,query),2)

def lower_band_step(+index: U32,+query: Folds.Query<Kernel.Band>,+value: U32) ->
  {Folds.update(~False{},~Kernel.Band,~band_used,~band_low,index,query,value) == Kernel.band_phase(index,lower_band_context(query),value) : U32}:
  Folds.Query{Kernel.Band{size,kernel,stride,padding,first,last,what},count,selected} = query
  {==}
def higher_band_step(+index: U32,+query: Folds.Query<Kernel.Band>,+value: U32) ->
  {Folds.update(~True{},~Kernel.Band,~band_used,~band_high,index,query,value) == Kernel.band_phase(index,higher_band_context(query),value) : U32}:
  Folds.Query{Kernel.Band{size,kernel,stride,padding,first,last,what},count,selected} = query
  {==}
def count_band_step(+index: U32,+query: Folds.Query<Kernel.Band>,+value: U32) ->
  {Folds.update(~True{},~Kernel.Band,~band_used,~band_count,index,query,value) == Kernel.band_phase(index,count_band_context(query),value) : U32}:
  Folds.Query{Kernel.Band{size,kernel,stride,padding,first,last,what},count,selected} = query
  {==}

def lower_band_contains(+query: Folds.Query<Kernel.Band>,+value: U32,used: Folds.frame(~Kernel.Band,~band_used,query,value),inside: {Nat.is_lt(Folds.index(~Kernel.Band,query),Folds.count(~Kernel.Band,query)) == True{} : Bool}) ->
  {U32.is_le(Traversal.indexed_fold(~U32,~Kernel.Band,~Kernel.band_phase,Folds.count(~Kernel.Band,query),0,lower_band_context(query),value),Folds.wanted(~Kernel.Band,~band_low,query)) == True{} : Bool}:
  %Arrays.indexed_fold_change_context(~U32,~Folds.Query<Kernel.Band>,~Kernel.Band,~lower_band_context,~Folds.update(~False{},~Kernel.Band,~band_used,~band_low),~Kernel.band_phase,~lower_band_step,Folds.count(~Kernel.Band,query),0,query,value) :
    {U32.is_le(_,Folds.wanted(~Kernel.Band,~band_low,query)) == True{} : Bool}
  Folds.contains(~False{},~Kernel.Band,~band_used,~band_low,query,value,used,inside)

def higher_band_contains(+query: Folds.Query<Kernel.Band>,+value: U32,used: Folds.frame(~Kernel.Band,~band_used,query,value),inside: {Nat.is_lt(Folds.index(~Kernel.Band,query),Folds.count(~Kernel.Band,query)) == True{} : Bool}) ->
  {U32.is_le(Folds.wanted(~Kernel.Band,~band_high,query),Traversal.indexed_fold(~U32,~Kernel.Band,~Kernel.band_phase,Folds.count(~Kernel.Band,query),0,higher_band_context(query),value)) == True{} : Bool}:
  %Arrays.indexed_fold_change_context(~U32,~Folds.Query<Kernel.Band>,~Kernel.Band,~higher_band_context,~Folds.update(~True{},~Kernel.Band,~band_used,~band_high),~Kernel.band_phase,~higher_band_step,Folds.count(~Kernel.Band,query),0,query,value) :
    {U32.is_le(Folds.wanted(~Kernel.Band,~band_high,query),_) == True{} : Bool}
  Folds.contains(~True{},~Kernel.Band,~band_used,~band_high,query,value,used,inside)

def count_band_contains(+query: Folds.Query<Kernel.Band>,+value: U32,used: Folds.frame(~Kernel.Band,~band_used,query,value),inside: {Nat.is_lt(Folds.index(~Kernel.Band,query),Folds.count(~Kernel.Band,query)) == True{} : Bool}) ->
  {U32.is_le(Folds.wanted(~Kernel.Band,~band_count,query),Traversal.indexed_fold(~U32,~Kernel.Band,~Kernel.band_phase,Folds.count(~Kernel.Band,query),0,count_band_context(query),value)) == True{} : Bool}:
  %Arrays.indexed_fold_change_context(~U32,~Folds.Query<Kernel.Band>,~Kernel.Band,~count_band_context,~Folds.update(~True{},~Kernel.Band,~band_used,~band_count),~Kernel.band_phase,~count_band_step,Folds.count(~Kernel.Band,query),0,query,value) :
    {U32.is_le(Folds.wanted(~Kernel.Band,~band_count,query),_) == True{} : Bool}
  Folds.contains(~True{},~Kernel.Band,~band_used,~band_count,query,value,used,inside)
