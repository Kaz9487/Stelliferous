# Shared ordered-reduction model for the four-row native accumulator. The
# native loop refinement is separate; this module never substitutes a model for it.
# Model indices: input at the absolute tile address, weights relative to base.
import Base
import ../ordered_matrix.bend as OrderedMatrix
import ./storage_shape.bend as Shape
import ./ordered_reduction_split.bend as Ordered
import ./ordered_ranges.bend as Ranges
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition
import ../kernel_gemm.bend as Gemm
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ../math_sequence.bend as Sequence
import ../math_fp32.bend as Fp32
import ../traversal_reduction.bend as Reduction
import ./storage_tree_model.bend as Tree
import ./kernel_accumulator.bend as Accumulator
import ./traversal_reduction_refinement.bend as ReductionLaws

def State() -> Data:
  Reduction.State<&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,U32,Rows.Rows<Tile.Tile8>>

def input_tile(tree: Tree.Tree<Tile.Tile8>,base: U32,index: U32) -> Tile.Tile8:
  Tree.read(Tile.Tile8,tree,index)

def weights_half(~upper: Bool,tree: Tree.Tree<Tile.Tile8>,base: U32,index: U32) -> Rows.Rows<F32>:
  Rows.half(upper,Tree.read(Tile.Tile8,tree,(base + index : U32)))

def update(base: U32,accumulators: Rows.Rows<Tile.Tile8>,input: Tile.Tile8,weights: Rows.Rows<F32>) -> Rows.Rows<Tile.Tile8>:
  Gemm.update(accumulators,input,weights)

type RowLane is Data:
  RowLane{row: Accumulator.Row,lane: Sequence.Index(8n)}

def input_at(key: RowLane,input: Tile.Tile8) -> F32:
  RowLane{row,lane} = key
  Tile.element_at(lane,input)

def weight_at(key: RowLane,weights: Rows.Rows<F32>) -> F32:
  RowLane{row,lane} = key
  Accumulator.row(~F32,row,weights)

def accumulator_at(key: RowLane,accumulator: Rows.Rows<Tile.Tile8>) -> F32:
  RowLane{row,lane} = key
  Tile.element_at(lane,Accumulator.row(~Tile.Tile8,row,accumulator))

def update_matches(+context: U32,+key: RowLane,+accumulator: Rows.Rows<Tile.Tile8>,+input: Tile.Tile8,+weights: Rows.Rows<F32>) ->
  {accumulator_at(key,update(context,accumulator,input,weights)) ==
    Fp32.multiply_add(accumulator_at(key,accumulator),input_at(key,input),weight_at(key,weights)) : F32}:
  RowLane{+row,+lane} = key
  %Equal.sym(Tile.Tile8,
    Accumulator.row(~Tile.Tile8,row,Gemm.update(accumulator,input,weights)),
    Tile.update(Accumulator.row(~Tile.Tile8,row,accumulator),input,Accumulator.row(~F32,row,weights)),
    Accumulator.row_update(row,accumulator,input,weights)) :
    {Tile.element_at(lane,_) == Fp32.multiply_add(Tile.element_at(lane,Accumulator.row(~Tile.Tile8,row,accumulator)),
      Tile.element_at(lane,input),Accumulator.row(~F32,row,weights)) : F32}
  Accumulator.update_at(lane,Accumulator.row(~Tile.Tile8,row,accumulator),input,Accumulator.row(~F32,row,weights))

def run(~upper: Bool,count: Nat,state: State()) -> State():
  Reduction.run(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,
    ~ReductionLaws.observe_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~U32,~input_tile),
    ~ReductionLaws.observe_read(~Tree.Tree<Tile.Tile8>,~Rows.Rows<F32>,~U32,~weights_half(~upper)),~update,count,state)

def ordered_dot(~upper: Bool,key: RowLane,count: Nat,state: State()) -> F32:
  ReductionLaws.ordered_dot(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,~RowLane,
    ~input_tile,~weights_half(~upper),~input_at,~weight_at,~accumulator_at,key,count,state)

def run_matches_ordered_dot(~upper: Bool,+key: RowLane,+count: Nat,+state: State()) ->
  {accumulator_at(key,Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,run(~upper,count,state))) ==
    ordered_dot(~upper,key,count,state) : F32}:
  ReductionLaws.model_loop_matches_ordered_dot(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,~RowLane,
    ~input_tile,~weights_half(~upper),~input_at,~weight_at,~accumulator_at,~update,~update_matches,key,count,state)

def initial(input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,input_index: U32,weight_index: U32,base: U32) -> State():
  Reduction.State{input,weights,input_index,weight_index,base,Rows.filled(~Tile.Tile8,Tile.filled(0.0))}

def selected(upper: Bool,count: Nat,state: State()) -> State():
  match upper:
    case False{}: run(~False{},count,state)
    case True{}: run(~True{},count,state)

def ordered_selected(upper: Bool,key: RowLane,count: Nat,state: State()) -> F32:
  match upper:
    case False{}: ordered_dot(~False{},key,count,state)
    case True{}: ordered_dot(~True{},key,count,state)

def selected_ordered(+upper: Bool,+key: RowLane,+count: Nat,+state: State()) ->
  {accumulator_at(key,Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,selected(upper,count,state))) ==
    ordered_selected(upper,key,count,state) : F32}:
  match upper:
    case False{}: run_matches_ordered_dot(~False{},key,count,state)
    case True{}: run_matches_ordered_dot(~True{},key,count,state)

# The same address tape, indexed by the natural reduction coordinate.
def natural_input() -> Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>> -> RowLane -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~ReductionLaws.AddressPair,~RowLane,~ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>,~ReductionLaws.AddressLayout,
    ~ReductionLaws.address_key,~ReductionLaws.input_element(~Tile.Tile8,~Tile.Tile8,~U32,~Tile.Tile8,~RowLane,~input_tile,~input_at))

def natural_weight(~upper: Bool) -> Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>> -> RowLane -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~ReductionLaws.AddressPair,~RowLane,~ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>,~ReductionLaws.AddressLayout,
    ~ReductionLaws.address_key,~ReductionLaws.weight_element(~Tile.Tile8,~Tile.Tile8,~U32,~Rows.Rows<F32>,~RowLane,~weights_half(~upper),~weight_at))

def natural_source(state: State()) -> Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>:
  Reduction.State{input,weights,input_index,weight_index,context,accumulator} = state
  Shape.Both{ReductionLaws.AddressLayout{U32.to_nat(input_index),U32.to_nat(weight_index)},ReductionLaws.SampleSource{input,weights,context}}

def natural_dot(~upper: Bool,+key: RowLane,count: Nat,+state: State()) -> F32:
  OrderedMatrix.dot(~RowLane,~Nat,~RowLane,~Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>,
    ~natural_input(),~natural_weight(~upper),Ranges.range(count,0n),key,key,natural_source(state),accumulator_at(key,Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,state)))

def encoded_dot(~upper: Bool,+key: RowLane,count: Nat,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,input_shift: Nat,weight_shift: Nat,context: U32,+accumulator: Rows.Rows<Tile.Tile8>) -> F32:
  OrderedMatrix.dot(~RowLane,~Nat,~RowLane,~Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>,
    ~natural_input(),~natural_weight(~upper),Ranges.range(count,0n),key,key,
    Shape.Both{ReductionLaws.AddressLayout{input_shift,weight_shift},ReductionLaws.SampleSource{input,weights,context}},accumulator_at(key,accumulator))

def encoded_addresses(+count: Nat,+input_shift: Nat,+weight_shift: Nat) ->
  {ReductionLaws.addresses(count,U32.from_nat(input_shift),U32.from_nat(weight_shift)) == ReductionLaws.address_taps(Ranges.range(count,0n),input_shift,weight_shift) : List<&2,ReductionLaws.AddressPair>}:
  %Addition.zero_right(input_shift) : {ReductionLaws.addresses(count,U32.from_nat(_),U32.from_nat(weight_shift)) == ReductionLaws.address_taps(Ranges.range(count,0n),input_shift,weight_shift) : List<&2,ReductionLaws.AddressPair>}
  %Addition.zero_right(weight_shift) : {ReductionLaws.addresses(count,U32.from_nat(Nat.add(input_shift,0n)),U32.from_nat(_)) == ReductionLaws.address_taps(Ranges.range(count,0n),input_shift,weight_shift) : List<&2,ReductionLaws.AddressPair>}
  ReductionLaws.consecutive_taps(count,0n,input_shift,weight_shift)

def encoded_as_natural(~upper: Bool,+key: RowLane,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+input_shift: Nat,+weight_shift: Nat,+context: U32,+accumulator: Rows.Rows<Tile.Tile8>) ->
  {ordered_dot(~upper,key,count,Reduction.State{input,weights,U32.from_nat(input_shift),U32.from_nat(weight_shift),context,accumulator}) == encoded_dot(~upper,key,count,input,weights,input_shift,weight_shift,context,accumulator) : F32}:
  %Equal.sym(List<&2,ReductionLaws.AddressPair>,ReductionLaws.addresses(count,U32.from_nat(input_shift),U32.from_nat(weight_shift)),ReductionLaws.address_taps(Ranges.range(count,0n),input_shift,weight_shift),encoded_addresses(count,input_shift,weight_shift)) :
    {OrderedMatrix.dot(~RowLane,~ReductionLaws.AddressPair,~RowLane,~ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>,
      ~ReductionLaws.input_element(~Tile.Tile8,~Tile.Tile8,~U32,~Tile.Tile8,~RowLane,~input_tile,~input_at),
      ~ReductionLaws.weight_element(~Tile.Tile8,~Tile.Tile8,~U32,~Rows.Rows<F32>,~RowLane,~weights_half(~upper),~weight_at),
      _,key,key,ReductionLaws.SampleSource{input,weights,context},accumulator_at(key,accumulator)) == encoded_dot(~upper,key,count,input,weights,input_shift,weight_shift,context,accumulator) : F32}
  Ordered.dot_reindexed(~Nat,~ReductionLaws.AddressPair,~RowLane,~RowLane,~ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>,~ReductionLaws.AddressLayout,
    ~ReductionLaws.address_key,
    ~ReductionLaws.input_element(~Tile.Tile8,~Tile.Tile8,~U32,~Tile.Tile8,~RowLane,~input_tile,~input_at),
    ~ReductionLaws.weight_element(~Tile.Tile8,~Tile.Tile8,~U32,~Rows.Rows<F32>,~RowLane,~weights_half(~upper),~weight_at),
    Ranges.range(count,0n),ReductionLaws.AddressLayout{input_shift,weight_shift},key,key,ReductionLaws.SampleSource{input,weights,context},accumulator_at(key,accumulator))

def ordered_as_natural(~upper: Bool,+key: RowLane,+count: Nat,+state: State()) ->
  {ordered_dot(~upper,key,count,state) == natural_dot(~upper,key,count,state) : F32}:
  Reduction.State{+input,+weights,+input_index,+weight_index,+context,+accumulator} = state
  %Arithmetic.native_roundtrip(input_index) : {ordered_dot(~upper,key,count,Reduction.State{input,weights,_,weight_index,context,accumulator}) == natural_dot(~upper,key,count,state) : F32}
  %Arithmetic.native_roundtrip(weight_index) : {ordered_dot(~upper,key,count,Reduction.State{input,weights,U32.from_nat(U32.to_nat(input_index)),_,context,accumulator}) == natural_dot(~upper,key,count,state) : F32}
  encoded_as_natural(~upper,key,count,input,weights,U32.to_nat(input_index),U32.to_nat(weight_index),context,accumulator)

def initial_zero(+key: RowLane) -> {accumulator_at(key,Rows.filled(~Tile.Tile8,Tile.filled(0.0))) == 0.0 : F32}:
  RowLane{+row,+lane} = key
  match row:
    case Accumulator.R0{}: Accumulator.initial_at(lane,0.0)
    case Accumulator.R1{}: Accumulator.initial_at(lane,0.0)
    case Accumulator.R2{}: Accumulator.initial_at(lane,0.0)
    case Accumulator.R3{}: Accumulator.initial_at(lane,0.0)
