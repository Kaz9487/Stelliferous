# A physical storage certificate bounds the whole public shape count. The
# actual count_of branch is completed from this bound, without a count oracle.
import Base
import ../tensor.bend as Tensors
import ../storage_buffer.bend as Storage
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./machine_limit_bounds.bend as Limits
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_contract.bend as Contract
import ./tensor_count_refinement.bend as Count
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_global_shape_count.bend as Global
import ./tensor_gather_values.bend as GatherValues

type Capacity is Data:
  Capacity{capacity: U32,actual: {capacity == Machine.capacity(Default.limits()) : U32},
    allocatable: {capacity == 1073741824 : U32},fits: {U32.is_le(capacity,2147483648) == True{} : Bool}}

def default_capacity() -> Capacity:
  Capacity{Machine.capacity(Default.limits()),{==},{==},Limits.capacity_fits(Default.limits())}

def physical_capacity(~Element: Data,+content: Buffer.Content<Element>,ready: Buffer.ContentReady<Element,content>) ->
  {U32.is_le(Tree.capacity(Element,Buffer.content_tree(~Element,content)),1073741824) == True{} : Bool}:
  Buffer.ContentReady{+depth,balanced,depth_ok,room} = ready
  %Equal.sym(U32,Tree.capacity(Element,Buffer.content_tree(~Element,content)),U32.shln(1,depth),
    Shape.capacity(Element,depth,Buffer.content_tree(~Element,content),balanced)) : {U32.is_le(_,1073741824) == True{} : Bool}
  Power.native_allocatable(depth,depth_ok)

def content_room(~Element: Data,+content: Buffer.Content<Element>,+capacity: U32,+allocatable: {capacity == 1073741824 : U32},+ready: Buffer.ContentReady<Element,content>) ->
  {Nat.is_le(Buffer.content_length(~Element,content),U32.to_nat(capacity)) == True{} : Bool}:
  Buffer.ContentReady{depth,balanced,depth_ok,room} = ready
  Order.transitive(Buffer.content_length(~Element,content),U32.to_nat(Tree.capacity(Element,Buffer.content_tree(~Element,content))),
    U32.to_nat(capacity),room,
    Bounds.as_nat_le(Tree.capacity(Element,Buffer.content_tree(~Element,content)),capacity,
      %Equal.sym(U32,capacity,1073741824,allocatable) : {U32.is_le(Tree.capacity(Element,Buffer.content_tree(~Element,content)),_) == True{} : Bool}
      physical_capacity(~Element,content,ready)))

def shape_room(~Element: Data,+shape: List<&2,U32>,+content: Buffer.Content<Element>,+capacity: U32,+allocatable: {capacity == 1073741824 : U32},well_formed: Contract.WellFormed<Element,shape,content>) ->
  {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(capacity)) == True{} : Bool}:
  Contract.WellFormed{length_exact,ready} = well_formed
  %length_exact : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
  content_room(~Element,content,capacity,allocatable,ready)

def count_fact(~Element: Data,+shape: List<&2,U32>,+content: Buffer.Content<Element>,+capacity: U32,+actual: {capacity == Machine.capacity(Default.limits()) : U32},
  +allocatable: {capacity == 1073741824 : U32},+fits: {U32.is_le(capacity,2147483648) == True{} : Bool},well_formed: Contract.WellFormed<Element,shape,content>) ->
  Count.Count<ShapeLaws.elements(shape),U32.from_nat(ShapeLaws.elements(shape))>:
  +room = shape_room(~Element,shape,content,capacity,allocatable,well_formed)
  +exact = Bounds.roundtrip(ShapeLaws.elements(shape),capacity,fits,room)
  Count.Count{Equal.sym(Nat,U32.to_nat(U32.from_nat(ShapeLaws.elements(shape))),ShapeLaws.elements(shape),exact),
    %actual : {U32.is_le(U32.from_nat(ShapeLaws.elements(shape)),_) == True{} : Bool}
    %Equal.sym(Bool,U32.is_le(U32.from_nat(ShapeLaws.elements(shape)),capacity),Nat.is_le(U32.to_nat(U32.from_nat(ShapeLaws.elements(shape))),U32.to_nat(capacity)),
      Comparison.le(U32.from_nat(ShapeLaws.elements(shape)),capacity)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ShapeLaws.elements(shape))),ShapeLaws.elements(shape),exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room}

def counted_fact(+shape: List<&2,U32>,+count: U32,evidence: Count.Count<ShapeLaws.elements(shape),count>) -> Global.Counted<shape>:
  Count.Count{+exact,bounded} = evidence
  Global.Counted{count,CountCompletion.count_of(shape,count,exact,bounded)}

def counted_capacity(~Element: Data,+shape: List<&2,U32>,+content: Buffer.Content<Element>,well_formed: Contract.WellFormed<Element,shape,content>,certificate: Capacity) -> Global.Counted<shape>:
  Capacity{+capacity,+actual,+allocatable,+fits} = certificate
  counted_fact(shape,U32.from_nat(ShapeLaws.elements(shape)),count_fact(~Element,shape,content,capacity,actual,allocatable,fits,well_formed))

def counted(~Element: Data,+shape: List<&2,U32>,+content: Buffer.Content<Element>,well_formed: Contract.WellFormed<Element,shape,content>) -> Global.Counted<shape>:
  counted_capacity(~Element,shape,content,well_formed,default_capacity())

