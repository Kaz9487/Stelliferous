# A physical storage certificate bounds the whole public shape count. The
# actual count_of branch is completed from this bound, without a count oracle.
import Base
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

def counted_fact(+shape: List<&2,U32>,+count: U32,evidence: Count.Count<ShapeLaws.elements(shape),count>) -> Global.Counted<shape>:
  Count.Count{+exact,bounded} = evidence
  Global.Counted{count,CountCompletion.count_of(shape,count,exact,bounded)}

def counted_within(+limits: Machine.Limits,+shape: List<&2,U32>,+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) -> Global.Counted<shape>:
  +count = U32.from_nat(ShapeLaws.elements(shape))
  +exact = Bounds.roundtrip(ShapeLaws.elements(shape),Machine.capacity(limits),Limits.capacity_fits(limits),room)
  counted_fact(shape,count,Count.Count{Equal.sym(Nat,U32.to_nat(count),ShapeLaws.elements(shape),exact),
    %default_capacity : {U32.is_le(count,_) == True{} : Bool}
    %Equal.sym(Bool,U32.is_le(count,Machine.capacity(limits)),Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))),Comparison.le(count,Machine.capacity(limits))) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(count),ShapeLaws.elements(shape),exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    room})

def count_room(+limits: Machine.Limits,+expected: Nat,+count: U32,+evidence: Count.Count<expected,count>,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) -> {Nat.is_le(expected,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Count.Count{+exact,+bounded} = evidence
  %Equal.sym(Nat,expected,U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Bounds.as_nat_le(count,Machine.capacity(limits),%Equal.sym(U32,Machine.capacity(limits),Machine.capacity(Default.limits()),default_capacity) : {U32.is_le(count,_) == True{} : Bool} bounded)

def counted_room(+limits: Machine.Limits,+shape: List<&2,U32>,+counted: Global.Counted<shape>,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) -> {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  count_room(limits,ShapeLaws.elements(shape),Global.count(shape,counted),Global.evidence(shape,counted),default_capacity)
