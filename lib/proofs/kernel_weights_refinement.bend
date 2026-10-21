# The actual block traversal refines a row-major read / packed write model.
# Coordinate coverage is separate from this unconditional execution equation.
import Base
import ../kernel_weights.bend as Weights
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../storage_allocation.bend as Allocation
import ../tensor.bend as Tensors
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_allocation_refinement.bend as AllocationLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./kernel_weights_readback.bend as WeightsReadback
import ./loop_projection.bend as Projection
import ./nat_order.bend as Order
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction
import ./traversal_map_readback.bend as MapReadback
import ./storage_shape.bend as Shape
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_interval.bend as Interval
import ./row_major_index.bend as RowMajor
import ./encoded_coordinates.bend as Encoded
import ./u32_shift.bend as Shift
import ./word_arithmetic.bend as WordArithmetic
import ./indexed_output_readback.bend as Indexed

type Owners is Data:
  Owners{input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>}

def represent(owners: Owners) -> Array<F32> & Array<Tile.Tile8>:
  Owners{input,output} = owners
  (Tree.pack(F32,input),Tree.pack(Tile.Tile8,output))

def output(owners: Owners) -> Tree.Tree<Tile.Tile8>:
  Owners{input,output} = owners
  output

def output_matches(+owners: Owners) ->
  {Weights.reordered(represent(owners)) == Tree.pack(Tile.Tile8,output(owners)) : Array<Tile.Tile8>}:
  Owners{input,output} = owners
  {==}

def allocated(+depth: Nat) ->
  {Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0)) == Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))) : Array<Tile.Tile8>}:
  Equal.trans(Array<Tile.Tile8>,Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0)),Array.new(Tile.Tile8,depth,Tile.filled(0.0)),
    Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))),AllocationLaws.array_same(Tile.Tile8,depth,Tile.filled(0.0)),
    StorageLaws.fresh(Tile.Tile8,depth,Tile.filled(0.0)))

def finish_mapped_rows(~Context: Data,state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>) -> Owners:
  Traversal.State{input,output,index,context} = state
  Owners{input,output}

def finished_mapped_rows(~Context: Data,+state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>) ->
  {Traversal.finish(~Array<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,
      ArrayLaws.represent(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~ArrayLaws.pack_tree(~F32),state)) ==
    represent(finish_mapped_rows(~Context,state)) : Array<F32> & Array<Tile.Tile8>}:
  Traversal.State{input,output,index,context} = state
  {==}

def block_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,block: U32,layout: Weights.MappedLayout<Context>,owners: Owners) -> Owners:
  Weights.MappedLayout{rows,+inner,row_stride,+context} = layout
  Owners{input,output} = owners
  +block = block
  finish_mapped_rows(~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>,
    ~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~WeightsReadback.sample_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context)),U32.to_nat(inner),
    Traversal.State{input,output,0,Weights.MappedRows{rows,(block * 8 : U32),(block * inner : U32),context,row_bases(context,row_stride,(block * 8 : U32))}}))

def block_mapped_matches_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+index: U32,+layout: Weights.MappedLayout<Context>,+owners: Owners) ->
  {Weights.copy_mapped_with(~Context,~offset,~row_bases,index,layout,represent(owners)) == represent(block_mapped_with(~Context,~offset,~row_bases,index,layout,owners)) : Array<F32> & Array<Tile.Tile8>}:
  Weights.MappedLayout{+rows,+inner,+row_stride,+context} = layout
  Owners{+input,+output} = owners
  +state = {Traversal.State{input,output,0,Weights.MappedRows{rows,(index * 8 : U32),(index * inner : U32),context,row_bases(context,row_stride,(index * 8 : U32))}} :
    Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>}
  +model = Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>,
    ~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~WeightsReadback.sample_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context)),U32.to_nat(inner),state)
  %Equal.sym(Traversal.State<&1,&1,Array<F32>,Array<Tile.Tile8>,Weights.MappedRows<Context>>,
    Traversal.run(~Array<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~Weights.read_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context),U32.to_nat(inner),
      ArrayLaws.represent(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~ArrayLaws.pack_tree(~F32),state)),
    ArrayLaws.represent(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~ArrayLaws.pack_tree(~F32),model),
    ArrayLaws.traversal_refines_model(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~ArrayLaws.pack_tree(~F32),
      ~Weights.read_mapped_rows(~Context,~offset),~WeightsReadback.sample_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context),~WeightsReadback.read_mapped_rows(~Context,~offset),U32.to_nat(inner),state)) :
    {Traversal.finish(~Array<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,_) == represent(finish_mapped_rows(~Context,model)) : Array<F32> & Array<Tile.Tile8>}
  finished_mapped_rows(~Context,model)

def run_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,row_stride: U32,context: Context,input: Tree.Tree<F32>) -> Owners:
  +blocks = ((rows + 7) / 8 : U32)
  Traversal.indexed_fold(~Owners,~Weights.MappedLayout<Context>,~block_mapped_with(~Context,~offset,~row_bases),U32.to_nat(blocks),0,Weights.MappedLayout{rows,inner,row_stride,context},
    Owners{input,Tree.fresh(Tile.Tile8,Tensors.depth_for((blocks * inner : U32)),Tile.filled(0.0))})

def run_mapped(~Context: Data,~offset: Context -> U32 -> U32,+rows: U32,+inner: U32,row_stride: U32,context: Context,input: Tree.Tree<F32>) -> Owners:
  run_mapped_with(~Context,~offset,~Weights.linear_bases(~Context),rows,inner,row_stride,context,input)

def run_mapped_matches_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,+input: Tree.Tree<F32>) ->
  {Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Tree.pack(F32,input)) == represent(run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input)) : Array<F32> & Array<Tile.Tile8>}:
  +blocks = ((rows + 7) / 8 : U32)
  +depth = Tensors.depth_for((blocks * inner : U32))
  +layout = {Weights.MappedLayout{rows,inner,row_stride,context} : Weights.MappedLayout<Context>}
  +owners = {Owners{input,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))} : Owners}
  %Equal.sym(Array<Tile.Tile8>,Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0)),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))),allocated(depth)) :
    {Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~Weights.MappedLayout<Context>,~Weights.copy_mapped_with(~Context,~offset,~row_bases),U32.to_nat(blocks),0,layout,
      (Tree.pack(F32,input),_)) == represent(run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input)) : Array<F32> & Array<Tile.Tile8>}
  ArrayLaws.indexed_fold_refines_model(~(Array<F32> & Array<Tile.Tile8>),~Owners,~Weights.MappedLayout<Context>,
    ~Weights.copy_mapped_with(~Context,~offset,~row_bases),~block_mapped_with(~Context,~offset,~row_bases),~represent,~block_mapped_matches_with(~Context,~offset,~row_bases),U32.to_nat(blocks),0,layout,owners)

def run_mapped_matches(~Context: Data,~offset: Context -> U32 -> U32,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,+input: Tree.Tree<F32>) ->
  {Weights.run_mapped(~Context,~offset,rows,inner,row_stride,context,Tree.pack(F32,input)) == represent(run_mapped(~Context,~offset,rows,inner,row_stride,context,input)) : Array<F32> & Array<Tile.Tile8>}:
  run_mapped_matches_with(~Context,~offset,~Weights.linear_bases(~Context),rows,inner,row_stride,context,input)

def input_tree(owners: Owners) -> Tree.Tree<F32>:
  Owners{input,output} = owners
  input

def mapped_inner(~Context: Data,layout: Weights.MappedLayout<Context>) -> U32:
  Weights.MappedLayout{rows,inner,row_stride,context} = layout
  inner

# A mapped block is the common contiguous map with a fixed source reader.
# Absolute destination positions are converted back to the block's inner index.
def block_context_with(~Context: Data,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+index: U32,layout: Weights.MappedLayout<Context>) -> Weights.MappedRows<Context>:
  Weights.MappedLayout{rows,+inner,row_stride,+context} = layout
  +row = (index * 8 : U32)
  Weights.MappedRows{rows,row,(index * inner : U32),context,row_bases(context,row_stride,row)}

type RowsMapContext<-Context: Data> is Data:
  RowsMapContext{input: Tree.Tree<F32>,context: Weights.MappedRows<Context>}

def finished_input_rows(~Context: Data,+state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>) ->
  {input_tree(finish_mapped_rows(~Context,state)) == ArrayLaws.model_reader(~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,state) : Tree.Tree<F32>}:
  Traversal.State{input,output,index,context} = state
  {==}

def block_mapped_input_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+index: U32,+layout: Weights.MappedLayout<Context>,+owners: Owners) ->
  {input_tree(block_mapped_with(~Context,~offset,~row_bases,index,layout,owners)) == input_tree(owners) : Tree.Tree<F32>}:
  Weights.MappedLayout{+rows,+inner,+row_stride,+context} = layout
  Owners{+input,+output} = owners
  +state = {Traversal.State{input,output,0,Weights.MappedRows{rows,(index * 8 : U32),(index * inner : U32),context,row_bases(context,row_stride,(index * 8 : U32))}} :
    Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>}
  +after = Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>,
    ~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~WeightsReadback.sample_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context)),U32.to_nat(inner),state)
  Equal.trans(Tree.Tree<F32>,input_tree(finish_mapped_rows(~Context,after)),ArrayLaws.model_reader(~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,after),input,
    finished_input_rows(~Context,after),ArrayLaws.model_reader_kept(~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~WeightsReadback.sample_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context),U32.to_nat(inner),state))

# The tiles Weights.run makes from contiguous rows.
def run(+rows: U32,+inner: U32,input: Tree.Tree<F32>) -> Tree.Tree<Tile.Tile8>:
  output(run_mapped(~Unit,~Weights.identity,rows,inner,inner,Unit{},input))

def run_matches(+rows: U32,+inner: U32,+input: Tree.Tree<F32>) ->
  {Weights.run(rows,inner,Tree.pack(F32,input)) == Tree.pack(Tile.Tile8,run(rows,inner,input)) : Array<Tile.Tile8>}:
  +result = run_mapped(~Unit,~Weights.identity,rows,inner,inner,Unit{},input)
  -actual = Weights.run_mapped(~Unit,~Weights.identity,rows,inner,inner,Unit{},Tree.pack(F32,input))
  Equal.trans(Array<Tile.Tile8>,Weights.reordered(actual),Weights.reordered(represent(result)),Tree.pack(Tile.Tile8,output(result)),
    Equal.cong(Array<F32> & Array<Tile.Tile8>,Array<Tile.Tile8>,Weights.reordered,actual,represent(result),run_mapped_matches(~Unit,~Weights.identity,rows,inner,inner,Unit{},input)),
    output_matches(result))



def map_value_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,context: RowsMapContext<Context>,position: U32,previous: Tile.Tile8) -> Tile.Tile8:
  RowsMapContext{input,Weights.MappedRows{rows,row,+destination,context,bases}} = context
  WeightsReadback.sample_mapped_rows(~Context,~offset,input,Weights.MappedRows{rows,row,destination,context,bases},(position - destination : U32))

def map_state_rows(~Context: Data,state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>) -> Traversal.State<&2,&2,Unit,Tree.Tree<Tile.Tile8>,RowsMapContext<Context>>:
  Traversal.State{input,output,position,Weights.MappedRows{rows,row,+destination,context,bases}} = state
  Traversal.State{Unit{},output,(destination + position : U32),RowsMapContext{input,Weights.MappedRows{rows,row,destination,context,bases}}}

def map_step_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>) ->
  {ArrayLaws.model_map_step(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),map_state_rows(~Context,state)) ==
    map_state_rows(~Context,ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~WeightsReadback.sample_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context),state)) : Traversal.State<&2,&2,Unit,Tree.Tree<Tile.Tile8>,RowsMapContext<Context>>}:
  Traversal.State{+input,+output,+position,Weights.MappedRows{+rows,+row,+destination,+context,+bases}} = state
  +mapped = {Weights.MappedRows{rows,row,destination,context,bases} : Weights.MappedRows<Context>}
  +sample = WeightsReadback.sample_mapped_rows(~Context,~offset,input,mapped,position)
  +same = Equal.trans(U32,((destination + position : U32) - destination : U32),((position + destination : U32) - destination : U32),position,
    Equal.cong(U32,U32,value => (value - destination : U32),(destination + position : U32),(position + destination : U32),Arithmetic.add_commutative(destination,position)),Subtraction.subtract_add(position,destination))
  %Equal.sym(U32,((destination + position : U32) - destination : U32),position,same) :
    {Traversal.State{Unit{},Tree.write(Tile.Tile8,output,(destination + position : U32),WeightsReadback.sample_mapped_rows(~Context,~offset,input,mapped,_)),((destination + position : U32) + 1 : U32),RowsMapContext{input,mapped}} == map_state_rows(~Context,ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~WeightsReadback.sample_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context),state)) : Traversal.State<&2,&2,Unit,Tree.Tree<Tile.Tile8>,RowsMapContext<Context>>}
  %Equal.sym(U32,((destination + position : U32) + 1 : U32),(destination + (position + 1 : U32) : U32),Arithmetic.add_associative(destination,position,1)) :
    {Traversal.State{Unit{},Tree.write(Tile.Tile8,output,(destination + position : U32),sample),_,RowsMapContext{input,mapped}} == map_state_rows(~Context,ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~WeightsReadback.sample_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context),state)) : Traversal.State<&2,&2,Unit,Tree.Tree<Tile.Tile8>,RowsMapContext<Context>>}
  {==}

def map_output_rows(~Context: Data,+state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>) ->
  {MapReadback.output(~Tile.Tile8,~RowsMapContext<Context>,map_state_rows(~Context,state)) == output(finish_mapped_rows(~Context,state)) : Tree.Tree<Tile.Tile8>}:
  Traversal.State{input,output,position,Weights.MappedRows{rows,row,destination,context,bases}} = state
  {==}

def block_as_map_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+index: U32,+layout: Weights.MappedLayout<Context>,+owners: Owners) ->
  {output(block_mapped_with(~Context,~offset,~row_bases,index,layout,owners)) ==
    MapReadback.run_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),U32.to_nat(mapped_inner(~Context,layout)),(index * mapped_inner(~Context,layout) : U32),RowsMapContext{input_tree(owners),block_context_with(~Context,~row_bases,index,layout)},output(owners)) : Tree.Tree<Tile.Tile8>}:
  Weights.MappedLayout{+rows,+inner,+row_stride,+context} = layout
  Owners{+input,+out} = owners
  +state = {Traversal.State{input,out,0,Weights.MappedRows{rows,(index * 8 : U32),(index * inner : U32),context,row_bases(context,row_stride,(index * 8 : U32))}} : Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>}
  +model = Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>,~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~WeightsReadback.sample_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context)),U32.to_nat(inner),state)
  +mapped = Loop.run(~Traversal.State<&2,&2,Unit,Tree.Tree<Tile.Tile8>,RowsMapContext<Context>>,~ArrayLaws.model_map_step(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases)),U32.to_nat(inner),map_state_rows(~Context,state))
  +same = Projection.iteration_refines_model(~Traversal.State<&2,&2,Unit,Tree.Tree<Tile.Tile8>,RowsMapContext<Context>>,~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Weights.MappedRows<Context>>,~ArrayLaws.model_map_step(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases)),~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~WeightsReadback.sample_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context)),~map_state_rows(~Context),~map_step_with(~Context,~offset,~row_bases),U32.to_nat(inner),state)
  %Arithmetic.add_zero((index * inner : U32)) :
    {output(block_mapped_with(~Context,~offset,~row_bases,index,layout,owners)) == MapReadback.run_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),U32.to_nat(inner),_,RowsMapContext{input,block_context_with(~Context,~row_bases,index,layout)},out) : Tree.Tree<Tile.Tile8>}
  Equal.sym(Tree.Tree<Tile.Tile8>,MapReadback.output(~Tile.Tile8,~RowsMapContext<Context>,mapped),output(finish_mapped_rows(~Context,model)),
    Equal.trans(Tree.Tree<Tile.Tile8>,MapReadback.output(~Tile.Tile8,~RowsMapContext<Context>,mapped),MapReadback.output(~Tile.Tile8,~RowsMapContext<Context>,map_state_rows(~Context,model)),output(finish_mapped_rows(~Context,model)),
      Equal.cong(Traversal.State<&2,&2,Unit,Tree.Tree<Tile.Tile8>,RowsMapContext<Context>>,Tree.Tree<Tile.Tile8>,MapReadback.output(~Tile.Tile8,~RowsMapContext<Context>),mapped,map_state_rows(~Context,model),same),map_output_rows(~Context,model)))

def mapped_rows(~Context: Data,layout: Weights.MappedLayout<Context>) -> U32:
  Weights.MappedLayout{rows,inner,row_stride,context} = layout
  rows

def mapped_blocks(~Context: Data,layout: Weights.MappedLayout<Context>) -> Nat:
  U32.to_nat(((mapped_rows(~Context,layout) + 7) / 8 : U32))

def mapped_depth(~Context: Data,+layout: Weights.MappedLayout<Context>) -> Nat:
  Tensors.depth_for((((mapped_rows(~Context,layout) + 7) / 8) * mapped_inner(~Context,layout) : U32))

# Only the logical allocation and selected coordinates constrain this query.
# The source mapping itself has no per-leaf or FP32 premise.
type Query<-Context: Data> is Data:
  Query{layout: Weights.MappedLayout<Context>,input: Tree.Tree<F32>,target: Nat,tap: Nat,
    depth_ok: {Nat.is_le(mapped_depth(~Context,layout),30n) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(mapped_blocks(~Context,layout),U32.to_nat(mapped_inner(~Context,layout))),U32.to_nat(U32.shln(1,mapped_depth(~Context,layout)))) == True{} : Bool},
    target_inside: {Nat.is_lt(target,mapped_blocks(~Context,layout)) == True{} : Bool},
    tap_inside: {Nat.is_lt(tap,U32.to_nat(mapped_inner(~Context,layout))) == True{} : Bool}}

def query_layout(~Context: Data,query: Query<Context>) -> Weights.MappedLayout<Context>:
  Query{layout,input,target,tap,depth_ok,room,target_inside,tap_inside} = query
  layout

def query_input(~Context: Data,query: Query<Context>) -> Tree.Tree<F32>:
  Query{layout,input,target,tap,depth_ok,room,target_inside,tap_inside} = query
  input

def query_target(~Context: Data,query: Query<Context>) -> Nat:
  Query{layout,input,target,tap,depth_ok,room,target_inside,tap_inside} = query
  target

def query_tap(~Context: Data,query: Query<Context>) -> Nat:
  Query{layout,input,target,tap,depth_ok,room,target_inside,tap_inside} = query
  tap

def query_limit(~Context: Data,query: Query<Context>) -> Nat:
  mapped_blocks(~Context,query_layout(~Context,query))

def query_width(~Context: Data,query: Query<Context>) -> Nat:
  U32.to_nat(mapped_inner(~Context,query_layout(~Context,query)))

def query_depth(~Context: Data,query: Query<Context>) -> Nat:
  mapped_depth(~Context,query_layout(~Context,query))

def query_capacity(~Context: Data,query: Query<Context>) -> U32:
  U32.shln(1,query_depth(~Context,query))

def query_position(~Context: Data,+query: Query<Context>) -> Nat:
  Nat.add(Nat.mul(query_target(~Context,query),query_width(~Context,query)),query_tap(~Context,query))

def query_address(~Context: Data,query: Query<Context>) -> U32:
  U32.from_nat(query_position(~Context,query))

def query_read(~Context: Data,query: Query<Context>,owners: Owners) -> Tile.Tile8:
  Tree.read(Tile.Tile8,output(owners),query_address(~Context,query))

def query_wanted_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,query: Query<Context>) -> Tile.Tile8:
  +query = query
  WeightsReadback.sample_mapped_rows(~Context,~offset,query_input(~Context,query),block_context_with(~Context,~row_bases,U32.from_nat(query_target(~Context,query)),query_layout(~Context,query)),U32.from_nat(query_tap(~Context,query)))

def query_wanted(~Context: Data,~offset: Context -> U32 -> U32,query: Query<Context>) -> Tile.Tile8:
  query_wanted_with(~Context,~offset,~Weights.linear_bases(~Context),query)

def query_update_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,index: U32,query: Query<Context>,owners: Owners) -> Owners:
  block_mapped_with(~Context,~offset,~row_bases,index,query_layout(~Context,query),owners)

def QueryFrame(~Context: Data,query: Query<Context>,owners: Owners) -> Data:
  +query = query
  +owners = owners
  Shape.Both<{input_tree(owners) == query_input(~Context,query) : Tree.Tree<F32>},Shape.Balanced(Tile.Tile8,query_depth(~Context,query),output(owners))>

def query_frame_step_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+query: Query<Context>,+index: U32,+owners: Owners,frame: QueryFrame(~Context,query,owners)) -> QueryFrame(~Context,query,query_update_with(~Context,~offset,~row_bases,index,query,owners)):
  Shape.Both{same,balanced} = frame
  +layout = query_layout(~Context,query)
  Shape.Both{Equal.trans(Tree.Tree<F32>,input_tree(block_mapped_with(~Context,~offset,~row_bases,index,layout,owners)),input_tree(owners),query_input(~Context,query),block_mapped_input_with(~Context,~offset,~row_bases,index,layout,owners),same),{
    %Equal.sym(Tree.Tree<Tile.Tile8>,output(block_mapped_with(~Context,~offset,~row_bases,index,layout,owners)),MapReadback.run_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),U32.to_nat(mapped_inner(~Context,layout)),(index * mapped_inner(~Context,layout) : U32),RowsMapContext{input_tree(owners),block_context_with(~Context,~row_bases,index,layout)},output(owners)),block_as_map_with(~Context,~offset,~row_bases,index,layout,owners)) : Shape.Balanced(Tile.Tile8,query_depth(~Context,query),_)
    MapReadback.balanced_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),U32.to_nat(mapped_inner(~Context,layout)),(index * mapped_inner(~Context,layout) : U32),RowsMapContext{input_tree(owners),block_context_with(~Context,~row_bases,index,layout)},output(owners),query_depth(~Context,query),balanced) : Shape.Balanced(Tile.Tile8,query_depth(~Context,query),output(block_mapped_with(~Context,~offset,~row_bases,index,layout,owners)))}}

def query_depth_ok(~Context: Data,+query: Query<Context>) -> {Nat.is_le(query_depth(~Context,query),30n) == True{} : Bool}:
  Query{layout,input,target,tap,depth_ok,room,target_inside,tap_inside} = query
  depth_ok

def query_room(~Context: Data,+query: Query<Context>) -> {Nat.is_le(Nat.mul(query_limit(~Context,query),query_width(~Context,query)),U32.to_nat(query_capacity(~Context,query))) == True{} : Bool}:
  Query{layout,input,target,tap,depth_ok,room,target_inside,tap_inside} = query
  room

def query_target_inside(~Context: Data,+query: Query<Context>) -> {Nat.is_lt(query_target(~Context,query),query_limit(~Context,query)) == True{} : Bool}:
  Query{layout,input,target,tap,depth_ok,room,target_inside,tap_inside} = query
  target_inside

def query_tap_inside(~Context: Data,+query: Query<Context>) -> {Nat.is_lt(query_tap(~Context,query),query_width(~Context,query)) == True{} : Bool}:
  Query{layout,input,target,tap,depth_ok,room,target_inside,tap_inside} = query
  tap_inside

def query_capacity_ok(~Context: Data,+query: Query<Context>) -> {U32.is_le(query_capacity(~Context,query),2147483648) == True{} : Bool}:
  Shift.power_fits(query_depth(~Context,query),query_depth_ok(~Context,query))

def query_index_room(~Context: Data,+query: Query<Context>,+index: Nat,inside: {Nat.is_lt(index,query_limit(~Context,query)) == True{} : Bool}) -> {Nat.is_le(index,U32.to_nat(query_capacity(~Context,query))) == True{} : Bool}:
  Order.transitive(index,query_limit(~Context,query),U32.to_nat(query_capacity(~Context,query)),Bounds.strict_le(index,query_limit(~Context,query),inside),
    Order.transitive(query_limit(~Context,query),Nat.mul(query_limit(~Context,query),query_width(~Context,query)),U32.to_nat(query_capacity(~Context,query)),
      RowMajor.product_covers(query_limit(~Context,query),query_width(~Context,query),Order.le_lt_trans(0n,query_tap(~Context,query),query_width(~Context,query),Order.zero_le(query_tap(~Context,query)),query_tap_inside(~Context,query))),query_room(~Context,query)))

def query_base_word(~Context: Data,+query: Query<Context>,+index: Nat,+inside: {Nat.is_lt(index,query_limit(~Context,query)) == True{} : Bool}) ->
  {(U32.from_nat(index) * mapped_inner(~Context,query_layout(~Context,query)) : U32) == U32.from_nat(Nat.mul(index,query_width(~Context,query))) : U32}:
  Encoded.multiply_known_values(U32.from_nat(index),mapped_inner(~Context,query_layout(~Context,query)),index,query_width(~Context,query),
    Bounds.roundtrip(index,query_capacity(~Context,query),query_capacity_ok(~Context,query),query_index_room(~Context,query,index,inside)),{==})

def query_block_room(~Context: Data,+query: Query<Context>,+index: Nat,inside: {Nat.is_lt(index,query_limit(~Context,query)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(index,query_width(~Context,query)),query_width(~Context,query)),U32.to_nat(query_capacity(~Context,query))) == True{} : Bool}:
  %Order.commutative(query_width(~Context,query),Nat.mul(index,query_width(~Context,query))) : {Nat.is_le(_,U32.to_nat(query_capacity(~Context,query))) == True{} : Bool}
  Order.transitive(Nat.mul(1n+index,query_width(~Context,query)),Nat.mul(query_limit(~Context,query),query_width(~Context,query)),U32.to_nat(query_capacity(~Context,query)),
    RowMajor.mul_le(1n+index,query_limit(~Context,query),query_width(~Context,query),Order.strict_follow(index,query_limit(~Context,query),inside)),query_room(~Context,query))

def query_position_inside(~Context: Data,+query: Query<Context>) -> {Nat.is_lt(query_position(~Context,query),U32.to_nat(query_capacity(~Context,query))) == True{} : Bool}:
  Bounds.nat_trans(query_position(~Context,query),Nat.mul(query_limit(~Context,query),query_width(~Context,query)),U32.to_nat(query_capacity(~Context,query)),
    RowMajor.radix(query_target(~Context,query),query_limit(~Context,query),query_tap(~Context,query),query_width(~Context,query),query_target_inside(~Context,query),query_tap_inside(~Context,query)),query_room(~Context,query))

def query_address_exact(~Context: Data,+query: Query<Context>) -> {U32.to_nat(query_address(~Context,query)) == query_position(~Context,query) : Nat}:
  Bounds.roundtrip(query_position(~Context,query),query_capacity(~Context,query),query_capacity_ok(~Context,query),Bounds.strict_le(query_position(~Context,query),U32.to_nat(query_capacity(~Context,query)),query_position_inside(~Context,query)))

def query_inner_word(~Context: Data,+query: Query<Context>) ->
  {(query_address(~Context,query) - (U32.from_nat(query_target(~Context,query)) * mapped_inner(~Context,query_layout(~Context,query)) : U32) : U32) == U32.from_nat(query_tap(~Context,query)) : U32}:
  +base = Nat.mul(query_target(~Context,query),query_width(~Context,query))
  +tap = query_tap(~Context,query)
  %Equal.sym(U32,(U32.from_nat(query_target(~Context,query)) * mapped_inner(~Context,query_layout(~Context,query)) : U32),U32.from_nat(base),query_base_word(~Context,query,query_target(~Context,query),query_target_inside(~Context,query))) : {(query_address(~Context,query) - _ : U32) == U32.from_nat(tap) : U32}
  %WordArithmetic.native_add(base,tap) : {(_ - U32.from_nat(base) : U32) == U32.from_nat(tap) : U32}
  %Arithmetic.add_commutative(U32.from_nat(tap),U32.from_nat(base)) : {(_ - U32.from_nat(base) : U32) == U32.from_nat(tap) : U32}
  Subtraction.subtract_add(U32.from_nat(tap),U32.from_nat(base))

def map_value_block_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+input: Tree.Tree<F32>,+layout: Weights.MappedLayout<Context>,+index: U32,+position: U32,+previous: Tile.Tile8) ->
  {map_value_with(~Context,~offset,~row_bases,RowsMapContext{input,block_context_with(~Context,~row_bases,index,layout)},position,previous) == WeightsReadback.sample_mapped_rows(~Context,~offset,input,block_context_with(~Context,~row_bases,index,layout),(position - (index * mapped_inner(~Context,layout) : U32) : U32)) : Tile.Tile8}:
  Weights.MappedLayout{rows,inner,row_stride,context} = layout
  {==}

def query_target_written_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+query: Query<Context>,+owners: Owners,frame: QueryFrame(~Context,query,owners)) ->
  {query_read(~Context,query,query_update_with(~Context,~offset,~row_bases,U32.from_nat(query_target(~Context,query)),query,owners)) == query_wanted_with(~Context,~offset,~row_bases,query) : Tile.Tile8}:
  Shape.Both{+same,+balanced} = frame
  +layout = query_layout(~Context,query)
  +index = U32.from_nat(query_target(~Context,query))
  +count = query_width(~Context,query)
  +base = Nat.mul(query_target(~Context,query),count)
  +native_base = (index * mapped_inner(~Context,layout) : U32)
  +map_context = {RowsMapContext{input_tree(owners),block_context_with(~Context,~row_bases,index,layout)} : RowsMapContext<Context>}
  +mapped = MapReadback.run_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),count,native_base,map_context,output(owners))
  +encoded = MapReadback.run_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),count,U32.from_nat(base),map_context,output(owners))
  +value = map_value_with(~Context,~offset,~row_bases,map_context,query_address(~Context,query),Tree.read(Tile.Tile8,output(owners),query_address(~Context,query)))
  Equal.trans(Tile.Tile8,query_read(~Context,query,query_update_with(~Context,~offset,~row_bases,index,query,owners)),Tree.read(Tile.Tile8,mapped,query_address(~Context,query)),query_wanted_with(~Context,~offset,~row_bases,query),
    Equal.cong(Tree.Tree<Tile.Tile8>,Tile.Tile8,tree => Tree.read(Tile.Tile8,tree,query_address(~Context,query)),output(block_mapped_with(~Context,~offset,~row_bases,index,layout,owners)),mapped,block_as_map_with(~Context,~offset,~row_bases,index,layout,owners)),
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,mapped,query_address(~Context,query)),Tree.read(Tile.Tile8,encoded,query_address(~Context,query)),query_wanted_with(~Context,~offset,~row_bases,query),
      Equal.cong(U32,Tile.Tile8,position => Tree.read(Tile.Tile8,MapReadback.run_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),count,position,map_context,output(owners)),query_address(~Context,query)),native_base,U32.from_nat(base),query_base_word(~Context,query,query_target(~Context,query),query_target_inside(~Context,query))),
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,encoded,query_address(~Context,query)),value,query_wanted_with(~Context,~offset,~row_bases,query),
        MapReadback.inside_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),count,base,query_tap(~Context,query),map_context,output(owners),query_depth(~Context,query),query_capacity(~Context,query),
          Shape.capacity(Tile.Tile8,query_depth(~Context,query),output(owners),balanced),balanced,Order.successor(query_depth(~Context,query),30n,query_depth_ok(~Context,query)),query_capacity_ok(~Context,query),query_block_room(~Context,query,query_target(~Context,query),query_target_inside(~Context,query)),query_tap_inside(~Context,query)),
        Equal.trans(Tile.Tile8,value,WeightsReadback.sample_mapped_rows(~Context,~offset,input_tree(owners),block_context_with(~Context,~row_bases,index,layout),U32.from_nat(query_tap(~Context,query))),query_wanted_with(~Context,~offset,~row_bases,query),
          Equal.trans(Tile.Tile8,value,WeightsReadback.sample_mapped_rows(~Context,~offset,input_tree(owners),block_context_with(~Context,~row_bases,index,layout),(query_address(~Context,query) - native_base : U32)),WeightsReadback.sample_mapped_rows(~Context,~offset,input_tree(owners),block_context_with(~Context,~row_bases,index,layout),U32.from_nat(query_tap(~Context,query))),
            map_value_block_with(~Context,~offset,~row_bases,input_tree(owners),layout,index,query_address(~Context,query),Tree.read(Tile.Tile8,output(owners),query_address(~Context,query))),
            Equal.cong(U32,Tile.Tile8,position => WeightsReadback.sample_mapped_rows(~Context,~offset,input_tree(owners),block_context_with(~Context,~row_bases,index,layout),position),(query_address(~Context,query) - native_base : U32),U32.from_nat(query_tap(~Context,query)),query_inner_word(~Context,query))),
          Equal.cong(Tree.Tree<F32>,Tile.Tile8,tree => WeightsReadback.sample_mapped_rows(~Context,~offset,tree,block_context_with(~Context,~row_bases,index,layout),U32.from_nat(query_tap(~Context,query))),input_tree(owners),query_input(~Context,query),same)))))

def query_later_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+query: Query<Context>,+index: Nat,+owners: Owners,frame: QueryFrame(~Context,query,owners),
  after: {Nat.is_lt(query_target(~Context,query),index) == True{} : Bool},inside: {Nat.is_lt(index,query_limit(~Context,query)) == True{} : Bool}) ->
  {query_read(~Context,query,query_update_with(~Context,~offset,~row_bases,U32.from_nat(index),query,owners)) == query_read(~Context,query,owners) : Tile.Tile8}:
  Shape.Both{same,+balanced} = frame
  +inside = inside
  +layout = query_layout(~Context,query)
  +native = U32.from_nat(index)
  +count = query_width(~Context,query)
  +base = Nat.mul(index,count)
  +native_base = (native * mapped_inner(~Context,layout) : U32)
  +map_context = {RowsMapContext{input_tree(owners),block_context_with(~Context,~row_bases,native,layout)} : RowsMapContext<Context>}
  +mapped = MapReadback.run_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),count,native_base,map_context,output(owners))
  +encoded = MapReadback.run_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),count,U32.from_nat(base),map_context,output(owners))
  +before = {
    %Equal.sym(Nat,U32.to_nat(query_address(~Context,query)),query_position(~Context,query),query_address_exact(~Context,query)) : {Nat.is_lt(_,base) == True{} : Bool}
    RowMajor.radix(query_target(~Context,query),index,query_tap(~Context,query),count,after,query_tap_inside(~Context,query)) : {Nat.is_lt(U32.to_nat(query_address(~Context,query)),base) == True{} : Bool}}
  Equal.trans(Tile.Tile8,query_read(~Context,query,query_update_with(~Context,~offset,~row_bases,native,query,owners)),Tree.read(Tile.Tile8,mapped,query_address(~Context,query)),query_read(~Context,query,owners),
    Equal.cong(Tree.Tree<Tile.Tile8>,Tile.Tile8,tree => Tree.read(Tile.Tile8,tree,query_address(~Context,query)),output(block_mapped_with(~Context,~offset,~row_bases,native,layout,owners)),mapped,block_as_map_with(~Context,~offset,~row_bases,native,layout,owners)),
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,mapped,query_address(~Context,query)),Tree.read(Tile.Tile8,encoded,query_address(~Context,query)),query_read(~Context,query,owners),
      Equal.cong(U32,Tile.Tile8,position => Tree.read(Tile.Tile8,MapReadback.run_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),count,position,map_context,output(owners)),query_address(~Context,query)),native_base,U32.from_nat(base),query_base_word(~Context,query,index,inside)),
      MapReadback.outside_indexed(~Tile.Tile8,~RowsMapContext<Context>,~map_value_with(~Context,~offset,~row_bases),count,base,query_address(~Context,query),map_context,output(owners),query_depth(~Context,query),query_capacity(~Context,query),
        Shape.capacity(Tile.Tile8,query_depth(~Context,query),output(owners),balanced),balanced,Order.successor(query_depth(~Context,query),30n,query_depth_ok(~Context,query)),query_capacity_ok(~Context,query),query_block_room(~Context,query,index,inside),
        MapReadback.address_inside(query_position(~Context,query),query_capacity(~Context,query),query_capacity_ok(~Context,query),query_position_inside(~Context,query)),Interval.Before{before})))

def query_update_matches_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+index: U32,+query: Query<Context>,+owners: Owners) ->
  {query_update_with(~Context,~offset,~row_bases,index,query,owners) == block_mapped_with(~Context,~offset,~row_bases,index,query_layout(~Context,query),owners) : Owners}:
  {==}

def mapped_stride(~Context: Data,layout: Weights.MappedLayout<Context>) -> U32:
  Weights.MappedLayout{rows,inner,row_stride,context} = layout
  row_stride

def mapped_context(~Context: Data,layout: Weights.MappedLayout<Context>) -> Context:
  Weights.MappedLayout{rows,inner,row_stride,context} = layout
  context

def mapped_lookup_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+query: Query<Context>) ->
  {query_read(~Context,query,run_mapped_with(~Context,~offset,~row_bases,mapped_rows(~Context,query_layout(~Context,query)),mapped_inner(~Context,query_layout(~Context,query)),mapped_stride(~Context,query_layout(~Context,query)),mapped_context(~Context,query_layout(~Context,query)),query_input(~Context,query))) == query_wanted_with(~Context,~offset,~row_bases,query) : Tile.Tile8}:
  Query{Weights.MappedLayout{+rows,+inner,+row_stride,+context},+input,+target,+tap,+depth_ok,+room,+target_inside,+tap_inside} = query
  +layout = {Weights.MappedLayout{rows,inner,row_stride,context} : Weights.MappedLayout<Context>}
  +query = {Query{layout,input,target,tap,depth_ok,room,target_inside,tap_inside} : Query<Context>}
  +initial = {Owners{input,Tree.fresh(Tile.Tile8,mapped_depth(~Context,layout),Tile.filled(0.0))} : Owners}
  +count = mapped_blocks(~Context,layout)
  +outer = Traversal.indexed_fold(~Owners,~Query<Context>,~query_update_with(~Context,~offset,~row_bases),count,0,query,initial)
  +inner_result = Traversal.indexed_fold(~Owners,~Weights.MappedLayout<Context>,~block_mapped_with(~Context,~offset,~row_bases),count,0,layout,initial)
  Equal.trans(Tile.Tile8,query_read(~Context,query,inner_result),query_read(~Context,query,outer),query_wanted_with(~Context,~offset,~row_bases,query),
    Equal.cong(Owners,Tile.Tile8,owners => query_read(~Context,query,owners),inner_result,outer,
      Equal.sym(Owners,outer,inner_result,ArrayLaws.indexed_fold_change_context(~Owners,~Query<Context>,~Weights.MappedLayout<Context>,~query_layout(~Context),~query_update_with(~Context,~offset,~row_bases),~block_mapped_with(~Context,~offset,~row_bases),~query_update_matches_with(~Context,~offset,~row_bases),count,0,query,initial))),
    Indexed.complete_readback(~Owners,~Query<Context>,~Tile.Tile8,~query_limit(~Context),~query_target(~Context),~query_wanted_with(~Context,~offset,~row_bases),~QueryFrame(~Context),~query_read(~Context),~query_update_with(~Context,~offset,~row_bases),~query_frame_step_with(~Context,~offset,~row_bases),~query_target_written_with(~Context,~offset,~row_bases),~query_later_with(~Context,~offset,~row_bases),query,initial,
      Shape.Both{{==},Shape.fresh(Tile.Tile8,mapped_depth(~Context,layout),Tile.filled(0.0))},target_inside))

def mapped_lookup(~Context: Data,~offset: Context -> U32 -> U32,+query: Query<Context>) ->
  {query_read(~Context,query,run_mapped(~Context,~offset,mapped_rows(~Context,query_layout(~Context,query)),mapped_inner(~Context,query_layout(~Context,query)),mapped_stride(~Context,query_layout(~Context,query)),mapped_context(~Context,query_layout(~Context,query)),query_input(~Context,query))) == query_wanted(~Context,~offset,query) : Tile.Tile8}:
  Query{Weights.MappedLayout{+rows,+inner,+row_stride,+context},+input,+target,+tap,+depth_ok,+room,+target_inside,+tap_inside} = query
  mapped_lookup_with(~Context,~offset,~Weights.linear_bases(~Context),query)
