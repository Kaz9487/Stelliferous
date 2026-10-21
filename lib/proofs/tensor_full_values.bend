# Public full tensors use the existing checked allocation and constant cells.
import Base
import ../tensor.bend as Tensors
import ./tensor_contract.bend as Contract
import ./tensor_global_shape_count.bend as Global
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_gather_values.bend as Gather
import ./tensor_count_refinement.bend as Count
import ./storage_buffer_model.bend as Buffer
import ./word_encoding.bend as Words
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./storage_readback.bend as StorageReadback
import ./storage_allocation_refinement.bend as AllocationLaws
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_count.bend as BandCount

# The starting output: every element is the initial value.
def constant(~Element: Data,initial: Element,position: U32) -> Element:
  initial

def fresh_cells(~Element: Data,remaining: Nat,+position: Nat,+depth: Nat,+initial: Element,+length: Nat) ->
  Members.All(Nat,{index => Contract.At(~Element,~Element,~constant(~Element),initial,Buffer.Content{Tree.fresh(Element,depth,initial),length},index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{StorageReadback.read_fresh(~Element,depth,initial,U32.from_nat(position)),fresh_cells(~Element,rest,1n+position,depth,initial,length)}

def initial_counted(~Element: Data,+unit: List<&2,U32>,+initial: Element,+count: U32,evidence: Count.Count<ShapeLaws.elements(unit),count>) ->
  Gather.Facts(~Element,~Element,~constant(~Element),initial,unit,Buffer.contents(~Element,Tensors.buffer(~Element,initial,count))):
  Count.Count{+exact,+bounded} = evidence
  +fresh = {Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),initial),U32.to_nat(count)} : Buffer.Content<Element>}
  +allocated = AllocationLaws.contents(~Element,Tensors.depth_for(count),initial,U32.to_nat(count))
  Shape.Both{
    Contract.WellFormed{
      Equal.trans(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Tensors.buffer(~Element,initial,count))),U32.to_nat(count),ShapeLaws.elements(unit),
        Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Tensors.buffer(~Element,initial,count)),fresh,allocated),
        Equal.sym(Nat,ShapeLaws.elements(unit),U32.to_nat(count),exact)),
      Buffer.native_content_ready(~Element,Tensors.buffer(~Element,initial,count),BandCount.allocated_ready(~Element,count,initial,Depth.bounded(count,bounded)))},
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,initial,count)),fresh,allocated) : Contract.Values(~Element,~Element,~constant(~Element),initial,_)
    fresh_cells(~Element,U32.to_nat(count),0n,Tensors.depth_for(count),initial,U32.to_nat(count))}

def allocated(~Element: Data,+shape: List<&2,U32>,+value: Element,+count: U32,
  facts: Gather.Facts(~Element,~Element,~constant(~Element),value,shape,Buffer.contents(~Element,Tensors.buffer(~Element,value,count)))) ->
  Global.Contract(~Element,~Element,~constant(~Element),value,Tensors.full_count(~Element,shape,Some{count},value)):
  WholeLeaf.stored(~Element,~Element,~constant(~Element),value,Tensors.buffer(~Element,value,count),shape,value,1,Buffer.observe(~Element,Tensors.buffer(~Element,value,count)),facts)

def selected(~Element: Data,+shape: List<&2,U32>,+value: Element,count: Maybe<&2,U32>,
  +classified: {Tensors.count_of(shape) == count : Maybe<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,Tensors.full_count(~Element,shape,count,value)) == True{} : Bool}) ->
  Global.Contract(~Element,~Element,~constant(~Element),value,Tensors.full_count(~Element,shape,count,value)):
  match count:
    case None{}:
      Empty.absurd(Global.Contract(~Element,~Element,~constant(~Element),value,Tensors.full_count(~Element,shape,None{},value)),Words.false_true(accepted))
    case Some{+count}:
      allocated(~Element,shape,value,count,initial_counted(~Element,shape,value,count,Count.count_of(shape,count,classified)))

def full(~Element: Data,+shape: List<&2,U32>,+value: Element,
  accepted: {Contract.tensor_ready(~Element,Tensors.full(~Element,shape,value)) == True{} : Bool}) ->
  Global.Contract(~Element,~Element,~constant(~Element),value,Tensors.full(~Element,shape,value)):
  selected(~Element,shape,value,Tensors.count_of(shape),{==},accepted)

# In particular, every logical cell of full([channels],+0.0) is +0.0.
def zeros(+channels: U32,accepted: {Contract.tensor_ready(~F32,Tensors.full(~F32,[channels],0.0)) == True{} : Bool}) ->
  Global.Contract(~F32,~F32,~constant(~F32),0.0,Tensors.full(~F32,[channels],0.0)):
  full(~F32,[channels],0.0,accepted)
