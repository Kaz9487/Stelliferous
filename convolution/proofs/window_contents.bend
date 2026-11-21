# Window dimensions and allocations fit the convolution's machine bounds.
import Base
import ../specification.bend as ConvSpec
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/boolean_logic.bend as Boolean
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./window_band_values.bend as WindowBandValues
import ./window_problem.bend as WindowProblem
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../inference.bend as NativeConv
import ./supported_padded_coordinates.bend as Padded
import ../../lib/proofs/word_multiplication.bend as WordMultiplication

def valid_condition(+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+index: Nat) ->
  {Boolean.condition_at(Guards.valid_conditions(problem),index) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Boolean.every_condition_holds(Guards.valid_conditions(ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias}),index,valid)

type RootContext is Data:
  RootContext{problem: ConvSpec.Problem,storage_contract: Inputs.Contract<problem>,valid: {ConvSpec.valid(problem) == True{} : Bool}}

def root_tree(context: RootContext) -> Tree.Tree<F32>:
  RootContext{problem,storage_contract,valid} = context
  Inputs.input_tree(problem,storage_contract)

def root_band(context: RootContext) -> WindowBandValues.Band:
  RootContext{+problem,storage_contract,valid} = context
  WindowProblem.band(problem,0n,Guards.height(problem))

def root_limit(context: RootContext) -> Nat:
  RootContext{+problem,storage_contract,valid} = context
  WindowProblem.count(problem,Guards.height(problem))

def root_read(+context: RootContext,+index: Nat,+inside: {Nat.is_lt(index,root_limit(context)) == True{} : Bool}) ->
  {Tree.read(F32,root_tree(context),U32.from_nat(index)) == WindowBandValues.value(root_band(context),index) : F32}:
  RootContext{+problem,+storage_contract,+valid} = context
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence,+parameters} = storage_contract
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +height_positive = Boolean.every_condition_holds(Guards.valid_conditions(problem),8n,valid)
  +width_positive = Boolean.every_condition_holds(Guards.valid_conditions(problem),7n,valid)
  +input_length = Boolean.every_condition_holds(Guards.valid_conditions(problem),2n,valid)
  +plane = Nat.mul(height,width)
  +positive = Order.product_positive(height,width,height_positive,width_positive)
  %DivisionOrder.quotient_exact(index,plane,positive) :
    {Tree.read(F32,StorageLaws.sequence_storage(F32,ConvSpec.length(input),input,input_evidence),U32.from_nat(index)) == ConvSpec.at(_,input) : F32}
  Inputs.stored_value(input,input_evidence,index,
    %Equal.sym(Nat,ConvSpec.length(input),Nat.mul(ci,plane),Order.equal_of_is_eq(ConvSpec.length(input),Nat.mul(ci,plane),input_length)) :
      {Nat.is_lt(index,_) == True{} : Bool}
    inside)

def root_contents(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+valid: {ConvSpec.valid(problem) == True{} : Bool}) ->
  WindowBandValues.Contents(WindowProblem.count(problem,Guards.height(problem)),WindowProblem.band(problem,0n,Guards.height(problem)),
    Inputs.input_tree(problem,storage_contract)):
  +count = WindowProblem.count(problem,Guards.height(problem))
  WindowBandValues.listed(~RootContext,~root_tree,~root_band,~root_limit,~root_read,count,0n,RootContext{problem,storage_contract,valid},Order.reflexive(count))

def input_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +support: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Guards.input_channel_count(problem),Nat.mul(Guards.height(problem),Guards.width(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Guards.Supported{ci_bound,co_bound,height_bound,width_bound,kernel_bound,stride_bound,padding_bound,+input_bound,weight_bound,
    output_bound,vertical_bound,horizontal_bound} = support
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +input_length = Boolean.every_condition_holds(Guards.valid_conditions(problem),2n,valid)
  %Order.equal_of_is_eq(ConvSpec.length(input),Nat.mul(ci,Nat.mul(height,width)),input_length) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  input_bound

def area_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Guards.height(problem),Guards.width(problem)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +ci = Guards.input_channel_count(problem)
  +area = Nat.mul(Guards.height(problem),Guards.width(problem))
  Order.transitive(area,Nat.mul(ci,area),U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(area,ci) : {Nat.is_le(area,_) == True{} : Bool}
    RowMajor.product_covers(area,ci,valid_condition(problem,valid,10n)),
    input_room(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)),valid))

def count_le(+problem: ConvSpec.Problem,+parent_first: U32,+parent_rows: U32,+first: U32,+rows: U32,
  +within: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows))) == True{} : Bool},
  +parent_fits: {Nat.is_le(Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows)),Guards.height(problem)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(rows),Guards.height(problem)) == True{} : Bool}:
  Order.transitive(U32.to_nat(rows),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Guards.height(problem),Order.right_le(U32.to_nat(first),U32.to_nat(rows)),
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows)),Guards.height(problem),within,parent_fits))
