# The actual table-indirected loop refines the shared ordered reduction.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../kernel_rows.bend as Rows
import ../traversal_loop.bend as Iteration
import ../traversal_reduction.bend as Reduction
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as Storage
import ./loop_projection.bend as Projection
import ./traversal_reduction_refinement.bend as Reductions
import ./kernel_reduction_model.bend as Accumulators
import ./storage_shape.bend as Shape
import ./ordered_reduction_split.bend as Ordered
import ./ordered_ranges.bend as Ranges
import ./bounded_u32_arithmetic.bend as Arithmetic
import ../ordered_matrix.bend as OrderedMatrix

type ReadContext is Data:
  ReadContext{native: Kernel.Context,offsets: Tree.Tree<U32>}

def input_tile(input: Tree.Tree<Tile.Tile8>,context: ReadContext,index: U32) -> Tile.Tile8:
  ReadContext{Kernel.Context{source,weight_base},offsets} = context
  Tree.read(Tile.Tile8,input,(source + Tree.read(U32,offsets,index) : U32))

def weights_half(~upper: Bool,weights: Tree.Tree<Tile.Tile8>,context: ReadContext,index: U32) -> Rows.Rows<F32>:
  ReadContext{Kernel.Context{source,weight_base},offsets} = context
  Rows.half(upper,Tree.read(Tile.Tile8,weights,(weight_base + index : U32)))

def update(context: ReadContext,accumulators: Rows.Rows<Tile.Tile8>,input: Tile.Tile8,weights: Rows.Rows<F32>) -> Rows.Rows<Tile.Tile8>:
  Rows.map(~Tile.Tile8,~Tile.Tile8,~F32,~Tile.update,accumulators,input,weights)

type Loop is Data:
  Loop{tiles: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,index: U32,context: Kernel.Context,accumulators: Rows.Rows<Tile.Tile8>}

def represent(state: Loop) -> Kernel.Loop:
  Loop{tiles,weights,offsets,index,context,accumulators} = state
  Kernel.Loop{Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),index,context,accumulators}

def step(~upper: Bool,state: Loop) -> Loop:
  Loop{+tiles,+weights,+offsets,+index,+context,accumulators} = state
  Loop{tiles,weights,offsets,(index + 1 : U32),context,
    update(ReadContext{context,offsets},accumulators,input_tile(tiles,ReadContext{context,offsets},index),weights_half(~upper,weights,ReadContext{context,offsets},index))}

def step_matches(~upper: Bool,+state: Loop) -> {Kernel.step(~upper,represent(state)) == represent(step(~upper,state)) : Kernel.Loop}:
  Loop{+tiles,+weights,+offsets,+index,Kernel.Context{+source,+weight_base},+accumulators} = state
  +weight = Tree.read(Tile.Tile8,weights,(weight_base + index : U32))
  +offset = Tree.read(U32,offsets,index)
  +input = Tree.read(Tile.Tile8,tiles,(source + offset : U32))
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,weights),(weight_base + index : U32)),
    (Tree.pack(Tile.Tile8,weights),weight),Storage.read(Tile.Tile8,weights,(weight_base + index : U32))) :
    {Kernel.with_weights(~upper,Tree.pack(Tile.Tile8,tiles),Tree.pack(U32,offsets),index,Kernel.Context{source,weight_base},source,accumulators,_) == represent(step(~upper,state)) : Kernel.Loop}
  %Equal.sym(Array<U32> & U32,Array.get(U32,Tree.pack(U32,offsets),index),(Tree.pack(U32,offsets),offset),Storage.read(U32,offsets,index)) :
    {Kernel.with_offset(Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),index,Kernel.Context{source,weight_base},source,accumulators,Rows.half(upper,weight),_) == represent(step(~upper,state)) : Kernel.Loop}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,tiles),(source + offset : U32)),(Tree.pack(Tile.Tile8,tiles),input),Storage.read(Tile.Tile8,tiles,(source + offset : U32))) :
    {Kernel.with_tile(Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),index,Kernel.Context{source,weight_base},accumulators,Rows.half(upper,weight),_) == represent(step(~upper,state)) : Kernel.Loop}
  {==}

def loop_matches(~upper: Bool,+count: Nat,+state: Loop) ->
  {Iteration.run(~Kernel.Loop,~Kernel.step(~upper),count,represent(state)) == represent(Iteration.run(~Loop,~step(~upper),count,state)) : Kernel.Loop}:
  Projection.iteration_refines_model(~Kernel.Loop,~Loop,~Kernel.step(~upper),~step(~upper),~represent,~step_matches(~upper),count,state)

def ReductionState() -> Data:
  Reduction.State<&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,ReadContext,Rows.Rows<Tile.Tile8>>

def as_reduction(state: Loop) -> ReductionState():
  Loop{tiles,weights,offsets,+index,context,accumulators} = state
  Reduction.State{tiles,weights,index,index,ReadContext{context,offsets},accumulators}

def reduction_step_matches(~upper: Bool,+state: Loop) ->
  {Reduction.step(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Rows.Rows<F32>,~ReadContext,~Rows.Rows<Tile.Tile8>,~Reductions.observe_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~ReadContext,~input_tile),~Reductions.observe_read(~Tree.Tree<Tile.Tile8>,~Rows.Rows<F32>,~ReadContext,~weights_half(~upper)),~update,as_reduction(state)) == as_reduction(step(~upper,state)) : ReductionState()}:
  Loop{tiles,weights,offsets,index,context,accumulators} = state
  {==}

def reduction_matches(~upper: Bool,+count: Nat,+state: Loop) ->
  {Iteration.run(~ReductionState(),~Reduction.step(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Rows.Rows<F32>,~ReadContext,~Rows.Rows<Tile.Tile8>,~Reductions.observe_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~ReadContext,~input_tile),~Reductions.observe_read(~Tree.Tree<Tile.Tile8>,~Rows.Rows<F32>,~ReadContext,~weights_half(~upper)),~update),count,as_reduction(state)) == as_reduction(Iteration.run(~Loop,~step(~upper),count,state)) : ReductionState()}:
  Projection.iteration_refines_model(~ReductionState(),~Loop,~Reduction.step(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Rows.Rows<F32>,~ReadContext,~Rows.Rows<Tile.Tile8>,~Reductions.observe_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~ReadContext,~input_tile),~Reductions.observe_read(~Tree.Tree<Tile.Tile8>,~Rows.Rows<F32>,~ReadContext,~weights_half(~upper)),~update),~step(~upper),~as_reduction,~reduction_step_matches(~upper),count,state)

def accumulators(state: Loop) -> Rows.Rows<Tile.Tile8>:
  Loop{tiles,weights,offsets,index,context,accumulators} = state
  accumulators

type Sources is Data:
  Sources{tiles: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,context: Kernel.Context}

def sources(state: Loop) -> Sources:
  Loop{tiles,weights,offsets,index,context,accumulators} = state
  Sources{tiles,weights,offsets,context}

def step_keeps_sources(~upper: Bool,+state: Loop) -> {sources(step(~upper,state)) == sources(state) : Sources}:
  Loop{tiles,weights,offsets,index,context,accumulators} = state
  {==}

def loop_keeps_sources(~upper: Bool,+count: Nat,+state: Loop) ->
  {sources(Iteration.run(~Loop,~step(~upper),count,state)) == sources(state) : Sources}:
  Projection.preserves_observation(~Loop,~Sources,~step(~upper),~sources,~step_keeps_sources(~upper),count,state)

def accumulators_match(+state: Loop) ->
  {Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~ReadContext,~Rows.Rows<Tile.Tile8>,as_reduction(state)) == accumulators(state) : Rows.Rows<Tile.Tile8>}:
  Loop{tiles,weights,offsets,index,context,accumulators} = state
  {==}

def update_at(+context: ReadContext,+key: Accumulators.RowLane,+values: Rows.Rows<Tile.Tile8>,+input: Tile.Tile8,+weights: Rows.Rows<F32>) ->
  {Accumulators.accumulator_at(key,update(context,values,input,weights)) ==
    (Accumulators.accumulator_at(key,values) + Accumulators.input_at(key,input) * Accumulators.weight_at(key,weights) : F32) : F32}:
  Accumulators.update_matches(0,key,values,input,weights)

def ordered_dot(~upper: Bool,key: Accumulators.RowLane,count: Nat,state: Loop) -> F32:
  Reductions.ordered_dot(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~ReadContext,~Rows.Rows<Tile.Tile8>,~Accumulators.RowLane,
    ~input_tile,~weights_half(~upper),~Accumulators.input_at,~Accumulators.weight_at,~Accumulators.accumulator_at,key,count,as_reduction(state))

def ordered_value(~upper: Bool,+key: Accumulators.RowLane,+count: Nat,+state: Loop) ->
  {Accumulators.accumulator_at(key,accumulators(Iteration.run(~Loop,~step(~upper),count,state))) == ordered_dot(~upper,key,count,state) : F32}:
  +result = Iteration.run(~Loop,~step(~upper),count,state)
  +reduced = Iteration.run(~ReductionState(),~Reduction.step(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Rows.Rows<F32>,~ReadContext,~Rows.Rows<Tile.Tile8>,~Reductions.observe_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~ReadContext,~input_tile),~Reductions.observe_read(~Tree.Tree<Tile.Tile8>,~Rows.Rows<F32>,~ReadContext,~weights_half(~upper)),~update),count,as_reduction(state))
  %accumulators_match(result) :
    {Accumulators.accumulator_at(key,_) == ordered_dot(~upper,key,count,state) : F32}
  %reduction_matches(~upper,count,state) :
    {Accumulators.accumulator_at(key,Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~ReadContext,~Rows.Rows<Tile.Tile8>,_)) == ordered_dot(~upper,key,count,state) : F32}
  Reductions.model_loop_matches_ordered_dot(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~ReadContext,~Rows.Rows<Tile.Tile8>,~Accumulators.RowLane,
    ~input_tile,~weights_half(~upper),~Accumulators.input_at,~Accumulators.weight_at,~Accumulators.accumulator_at,~update,~update_at,key,count,as_reduction(state))

def native_context(state: Loop) -> Kernel.Context:
  Loop{tiles,weights,offsets,index,context,accumulators} = state
  context

def loop_index(state: Loop) -> U32:
  Loop{tiles,weights,offsets,index,context,accumulators} = state
  index

# Reindex the actual indirect address tape without changing MAC order.
def natural_input() -> Shape.Both<Reductions.AddressLayout,Reductions.SampleSource<Tile.Tile8,Tile.Tile8,ReadContext>> -> Accumulators.RowLane -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~Reductions.AddressPair,~Accumulators.RowLane,~Reductions.SampleSource<Tile.Tile8,Tile.Tile8,ReadContext>,~Reductions.AddressLayout,
    ~Reductions.address_key,~Reductions.input_element(~Tile.Tile8,~Tile.Tile8,~ReadContext,~Tile.Tile8,~Accumulators.RowLane,~input_tile,~Accumulators.input_at))

def natural_weight(~upper: Bool) -> Shape.Both<Reductions.AddressLayout,Reductions.SampleSource<Tile.Tile8,Tile.Tile8,ReadContext>> -> Accumulators.RowLane -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~Reductions.AddressPair,~Accumulators.RowLane,~Reductions.SampleSource<Tile.Tile8,Tile.Tile8,ReadContext>,~Reductions.AddressLayout,
    ~Reductions.address_key,~Reductions.weight_element(~Tile.Tile8,~Tile.Tile8,~ReadContext,~Rows.Rows<F32>,~Accumulators.RowLane,~weights_half(~upper),~Accumulators.weight_at))

def natural_source(state: Loop) -> Shape.Both<Reductions.AddressLayout,Reductions.SampleSource<Tile.Tile8,Tile.Tile8,ReadContext>>:
  Loop{input,weights,offsets,+input_index,native,accumulator} = state
  +weight_index = input_index
  +context = {ReadContext{native,offsets} : ReadContext}
  Shape.Both{Reductions.AddressLayout{U32.to_nat(input_index),U32.to_nat(weight_index)},Reductions.SampleSource{input,weights,context}}

def natural_dot(~upper: Bool,+key: Accumulators.RowLane,count: Nat,+state: Loop) -> F32:
  OrderedMatrix.dot(~Accumulators.RowLane,~Nat,~Accumulators.RowLane,~Shape.Both<Reductions.AddressLayout,Reductions.SampleSource<Tile.Tile8,Tile.Tile8,ReadContext>>,
    ~natural_input(),~natural_weight(~upper),Ranges.range(count,0n),key,key,natural_source(state),Accumulators.accumulator_at(key,accumulators(state)))

def encoded_dot(~upper: Bool,+key: Accumulators.RowLane,count: Nat,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,input_shift: Nat,weight_shift: Nat,context: ReadContext,+accumulator: Rows.Rows<Tile.Tile8>) -> F32:
  OrderedMatrix.dot(~Accumulators.RowLane,~Nat,~Accumulators.RowLane,~Shape.Both<Reductions.AddressLayout,Reductions.SampleSource<Tile.Tile8,Tile.Tile8,ReadContext>>,
    ~natural_input(),~natural_weight(~upper),Ranges.range(count,0n),key,key,
    Shape.Both{Reductions.AddressLayout{input_shift,weight_shift},Reductions.SampleSource{input,weights,context}},Accumulators.accumulator_at(key,accumulator))

def encoded_as_natural(~upper: Bool,+key: Accumulators.RowLane,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+input_shift: Nat,+weight_shift: Nat,+context: ReadContext,+accumulator: Rows.Rows<Tile.Tile8>) ->
  {Reductions.ordered_dot(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~ReadContext,~Rows.Rows<Tile.Tile8>,~Accumulators.RowLane,~input_tile,~weights_half(~upper),~Accumulators.input_at,~Accumulators.weight_at,~Accumulators.accumulator_at,key,count,Reduction.State{input,weights,U32.from_nat(input_shift),U32.from_nat(weight_shift),context,accumulator}) == encoded_dot(~upper,key,count,input,weights,input_shift,weight_shift,context,accumulator) : F32}:
  %Equal.sym(List<&2,Reductions.AddressPair>,Reductions.addresses(count,U32.from_nat(input_shift),U32.from_nat(weight_shift)),Reductions.address_taps(Ranges.range(count,0n),input_shift,weight_shift),Accumulators.encoded_addresses(count,input_shift,weight_shift)) :
    {OrderedMatrix.dot(~Accumulators.RowLane,~Reductions.AddressPair,~Accumulators.RowLane,~Reductions.SampleSource<Tile.Tile8,Tile.Tile8,ReadContext>,
      ~Reductions.input_element(~Tile.Tile8,~Tile.Tile8,~ReadContext,~Tile.Tile8,~Accumulators.RowLane,~input_tile,~Accumulators.input_at),
      ~Reductions.weight_element(~Tile.Tile8,~Tile.Tile8,~ReadContext,~Rows.Rows<F32>,~Accumulators.RowLane,~weights_half(~upper),~Accumulators.weight_at),
      _,key,key,Reductions.SampleSource{input,weights,context},Accumulators.accumulator_at(key,accumulator)) == encoded_dot(~upper,key,count,input,weights,input_shift,weight_shift,context,accumulator) : F32}
  Ordered.dot_reindexed(~Nat,~Reductions.AddressPair,~Accumulators.RowLane,~Accumulators.RowLane,~Reductions.SampleSource<Tile.Tile8,Tile.Tile8,ReadContext>,~Reductions.AddressLayout,
    ~Reductions.address_key,
    ~Reductions.input_element(~Tile.Tile8,~Tile.Tile8,~ReadContext,~Tile.Tile8,~Accumulators.RowLane,~input_tile,~Accumulators.input_at),
    ~Reductions.weight_element(~Tile.Tile8,~Tile.Tile8,~ReadContext,~Rows.Rows<F32>,~Accumulators.RowLane,~weights_half(~upper),~Accumulators.weight_at),
    Ranges.range(count,0n),Reductions.AddressLayout{input_shift,weight_shift},key,key,Reductions.SampleSource{input,weights,context},Accumulators.accumulator_at(key,accumulator))

def ordered_as_natural(~upper: Bool,+key: Accumulators.RowLane,+count: Nat,+state: Loop) ->
  {ordered_dot(~upper,key,count,state) == natural_dot(~upper,key,count,state) : F32}:
  Loop{+input,+weights,+offsets,+index,+native,+accumulator} = state
  %Arithmetic.native_roundtrip(index) : {Reductions.ordered_dot(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~ReadContext,~Rows.Rows<Tile.Tile8>,~Accumulators.RowLane,~input_tile,~weights_half(~upper),~Accumulators.input_at,~Accumulators.weight_at,~Accumulators.accumulator_at,key,count,Reduction.State{input,weights,_,_,ReadContext{native,offsets},accumulator}) == natural_dot(~upper,key,count,state) : F32}
  encoded_as_natural(~upper,key,count,input,weights,U32.to_nat(index),U32.to_nat(index),ReadContext{native,offsets},accumulator)
