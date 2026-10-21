# Packed group execution specializes the shared GEMM proof without bias.
import Base
import ../kernel_weight_gradient.bend as Kernel
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_grouped.bend as Grouped
import ./kernel_weight_gradient_packing.bend as Packing
import ./kernel_gemm_refinement.bend as Gemms
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ../traversal_loop.bend as Loop
import ../traversal_array.bend as Traversal
import ./loop_projection.bend as Projection
import ./traversal_array_refinement.bend as Traversals
import ../kernel_rows.bend as Rows
import ./kernel_gemm_writes.bend as GemmWrites
import ./write_program.bend as Writes
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./storage_shape.bend as Shape
import ./kernel_accumulator.bend as Accumulator
import ./kernel_tile_refinement.bend as TileLaws
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_interval.bend as Interval
import ./kernel_reduction_model.bend as ReductionModel
import ../traversal_reduction.bend as Reduction
import ../ordered_matrix.bend as OrderedMatrix
import ./ordered_reduction_split.bend as Ordered
import ./ordered_ranges.bend as Ranges
import ./traversal_reduction_refinement.bend as ReductionLaws
import ./sequence_membership.bend as Members
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ../machine_limits.bend as Machine
import ./nat_division_order.bend as DivisionOrder
import ./u32_extrema.bend as NativeExtrema
import ./nat_extrema.bend as Extrema
import ./u32_comparison.bend as Comparison
import ./u32_full_range.bend as FullRange
import ./word_arithmetic.bend as WordArithmetic
import ./u32_division.bend as NativeDivision
import ./nat_ceiling.bend as Ceiling
import ./storage_readback.bend as StorageReadback
import ./indexed_output_readback.bend as IndexedReadback
import ./write_program_readback.bend as ProgramRead
import ./boolean_logic.bend as Logic
import ./word_encoding.bend as Words
import ./binary_radix.bend as Radix
import ./row_major_index.bend as RowMajor
import ./word_subtraction.bend as Subtraction
import ./encoded_coordinates.bend as Encoded
import ./machine_limit_bounds.bend as Limits
import ./tensor_allocation_depth.bend as AllocationDepth
import ./grouped_reduction_refinement.bend as GroupLaws
import ./u32_shift.bend as Shift

type Group is Data:
  Group{input: Buffer.Model<F32>,gradient: Buffer.Model<F32>,values: Tree.Tree<F32>}

def represent(result: Group) -> (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>:
  Group{input,gradient,values} = result
  ((Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),Tree.pack(F32,values))

def window_gemm(continued: Bool,count: U32,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,output: Tree.Tree<F32>) -> Tree.Tree<F32>:
  match continued:
    case False{}: Gemms.run_at(~False{},8,input,weights,Tree.fresh(F32,Storage.depth_for(1),0.0),output,8,count,0,8)
    case True{}: Gemms.result(Gemms.continue_keeping(8,input,weights,Tree.fresh(F32,Storage.depth_for(1),0.0),output,8,count))

def window_gemm_matches(continued: Bool,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<F32>) ->
  {Kernel.window_gemm(continued,count,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,output)) ==
    Tree.pack(F32,window_gemm(continued,count,input,weights,output)) : Array<F32>}:
  match continued:
    case False{}:
      %Equal.sym(Array<F32>,Array.new(F32,Storage.depth_for(1),0.0),Tree.pack(F32,Tree.fresh(F32,Storage.depth_for(1),0.0)),StorageLaws.fresh(F32,Storage.depth_for(1),0.0)) :
        {Gemm.run_at(~False{},8,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),_,Tree.pack(F32,output),8,count,0,8) == Tree.pack(F32,window_gemm(False{},count,input,weights,output)) : Array<F32>}
      Gemms.run_at_matches(~False{},8,input,weights,Tree.fresh(F32,Storage.depth_for(1),0.0),output,8,count,0,8)
    case True{}:
      +bias = Tree.fresh(F32,Storage.depth_for(1),0.0)
      +state = Gemms.continue_keeping(8,input,weights,bias,output,8,count)
      %Equal.sym(Array<F32>,Array.new(F32,Storage.depth_for(1),0.0),Tree.pack(F32,bias),StorageLaws.fresh(F32,Storage.depth_for(1),0.0)) :
        {Gemm.result(Gemm.continue_keeping(8,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),_,Tree.pack(F32,output),8,count)) == Tree.pack(F32,window_gemm(True{},count,input,weights,output)) : Array<F32>}
      %Equal.sym(Gemm.Buffers,Gemm.continue_keeping(8,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),8,count),
        Gemms.represent(state),Gemms.continue_keeping_matches(8,input,weights,bias,output,8,count)) :
        {Gemm.result(_) == Tree.pack(F32,window_gemm(True{},count,input,weights,output)) : Array<F32>}
      Gemms.result_matches(state)

type ChunkModel is Data:
  ChunkModel{input: Buffer.Model<F32>,gradient: Buffer.Model<F32>,values: Tree.Tree<F32>,remaining: U32,window: Kernel.WeightWindow}

def represent_chunk(state: ChunkModel) -> Kernel.ChunkState:
  ChunkModel{input,gradient,values,remaining,window} = state
  Kernel.ChunkState{Buffer.pack(~F32,input),Buffer.pack(~F32,gradient),Tree.pack(F32,values),remaining,window}

def chunk_result(~continued: Bool,+count: U32,+remaining: U32,window: Kernel.WeightWindow,values: Tree.Tree<F32>,packed: Packing.Packing,read: Packing.Packed) -> ChunkModel:
  Kernel.WeightWindow{geometry,image,tap,output,old_row,old_column,first} = window
  Packing.Packing{input,weights,index,row,column,columns,patch,h,w,s,p} = packed
  Packing.Packed{gradient,tiles} = read
  ChunkModel{input,gradient,window_gemm(continued,count,tiles,weights,values),(remaining - count : U32),Kernel.WeightWindow{geometry,image,tap,output,row,column,(first + count : U32)}}

def chunk_result_matches(~continued: Bool,+count: U32,+remaining: U32,+window: Kernel.WeightWindow,+values: Tree.Tree<F32>,+packed: Packing.Packing,+read: Packing.Packed) ->
  {Kernel.chunk_result(~continued,count,remaining,window,Tree.pack(F32,values),Packing.represent(packed),Packing.owners(read)) ==
    represent_chunk(chunk_result(~continued,count,remaining,window,values,packed,read)) : Kernel.ChunkState}:
  Kernel.WeightWindow{geometry,image,tap,output,old_row,old_column,first} = window
  Packing.Packing{input,+weights,index,row,column,columns,patch,h,w,s,p} = packed
  Packing.Packed{gradient,+tiles} = read
  Equal.cong(Array<F32>,Kernel.ChunkState,values => Kernel.ChunkState{Buffer.pack(~F32,input),Buffer.pack(~F32,gradient),values,(remaining - count : U32),
    Kernel.WeightWindow{geometry,image,tap,output,row,column,(first + count : U32)}},
    Kernel.window_gemm(continued,count,Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,values)),
    Tree.pack(F32,window_gemm(continued,count,tiles,weights,values)),window_gemm_matches(continued,count,tiles,weights,values))

def chunk(~continued: Bool,state: ChunkModel) -> ChunkModel:
  ChunkModel{input,gradient,values,+remaining,+window} = state
  +count = U32.min(256,remaining)
  chunk_result(~continued,count,remaining,window,values,Packing.window_input(window,count,input),Packing.window_gradient(window,count,gradient))

def chunk_matches(~continued: Bool,+state: ChunkModel) ->
  {Kernel.chunk(~continued,represent_chunk(state)) == represent_chunk(chunk(~continued,state)) : Kernel.ChunkState}:
  ChunkModel{+input,+gradient,+values,+remaining,+window} = state
  +count = U32.min(256,remaining)
  +packed = Packing.window_input(window,count,input)
  +read = Packing.window_gradient(window,count,gradient)
  %Equal.sym(Kernel.Packing,Kernel.window_input(window,count,Buffer.pack(~F32,input)),Packing.represent(packed),Packing.window_input_matches(window,count,input)) :
    {Kernel.chunk_result(~continued,count,remaining,window,Tree.pack(F32,values),_,Kernel.window_gradient(window,count,Buffer.pack(~F32,gradient))) == represent_chunk(chunk(~continued,state)) : Kernel.ChunkState}
  %Equal.sym(Storage.Buffer<F32> & Array<Tile.Tile8>,Kernel.window_gradient(window,count,Buffer.pack(~F32,gradient)),Packing.owners(read),Packing.window_gradient_matches(window,count,gradient)) :
    {Kernel.chunk_result(~continued,count,remaining,window,Tree.pack(F32,values),Packing.represent(packed),_) == represent_chunk(chunk(~continued,state)) : Kernel.ChunkState}
  chunk_result_matches(~continued,count,remaining,window,values,packed,read)

def chunk_values(state: ChunkModel) -> Group:
  ChunkModel{input,gradient,values,remaining,window} = state
  Group{input,gradient,values}

def chunk_values_matches(+state: ChunkModel) ->
  {Kernel.chunk_values(represent_chunk(state)) == represent(chunk_values(state)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}:
  ChunkModel{input,gradient,values,remaining,window} = state
  {==}

def window_product(+count: U32,window: Kernel.WeightWindow,input: Buffer.Model<F32>,gradient: Buffer.Model<F32>) -> Group:
  +turns = (((count - 1) / 256) + 1 : U32)
  chunk_values(Loop.run(~ChunkModel,~chunk(~True{}),U32.to_nat((turns - 1 : U32)),
    chunk(~False{},ChunkModel{input,gradient,Tree.fresh(F32,Storage.depth_for(64),0.0),count,window})))

def window_product_matches(+count: U32,+window: Kernel.WeightWindow,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {Kernel.window_product(count,window,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)) == represent(window_product(count,window,input,gradient)) :
    (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}:
  +turns = (((count - 1) / 256) + 1 : U32)
  +steps = U32.to_nat((turns - 1 : U32))
  +initial = {ChunkModel{input,gradient,Tree.fresh(F32,Storage.depth_for(64),0.0),count,window} : ChunkModel}
  +first = chunk(~False{},initial)
  +state = Loop.run(~ChunkModel,~chunk(~True{}),steps,first)
  %Equal.sym(Array<F32>,Array.new(F32,Storage.depth_for(64),0.0),Tree.pack(F32,Tree.fresh(F32,Storage.depth_for(64),0.0)),StorageLaws.fresh(F32,Storage.depth_for(64),0.0)) :
    {Kernel.chunk_values(Loop.run(~Kernel.ChunkState,~Kernel.chunk(~True{}),steps,Kernel.chunk(~False{},Kernel.ChunkState{Buffer.pack(~F32,input),Buffer.pack(~F32,gradient),_,count,window}))) == represent(window_product(count,window,input,gradient)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}
  %Equal.sym(Kernel.ChunkState,Kernel.chunk(~False{},represent_chunk(initial)),represent_chunk(first),chunk_matches(~False{},initial)) :
    {Kernel.chunk_values(Loop.run(~Kernel.ChunkState,~Kernel.chunk(~True{}),steps,_)) == represent(window_product(count,window,input,gradient)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}
  %Equal.sym(Kernel.ChunkState,Loop.run(~Kernel.ChunkState,~Kernel.chunk(~True{}),steps,represent_chunk(first)),represent_chunk(state),
    Projection.iteration_refines_model(~Kernel.ChunkState,~ChunkModel,~Kernel.chunk(~True{}),~chunk(~True{}),~represent_chunk,~chunk_matches(~True{}),steps,first)) :
    {Kernel.chunk_values(_) == represent(window_product(count,window,input,gradient)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}
  chunk_values_matches(state)

type Sources is Data:
  Sources{input: Buffer.Model<F32>,gradient: Buffer.Model<F32>}

def chunk_sources(state: ChunkModel) -> Sources:
  ChunkModel{input,gradient,values,remaining,window} = state
  Sources{input,gradient}

def group_sources(state: Group) -> Sources:
  Group{input,gradient,values} = state
  Sources{input,gradient}

def chunk_result_sources(~continued: Bool,+count: U32,+remaining: U32,+window: Kernel.WeightWindow,+values: Tree.Tree<F32>,+packed: Packing.Packing,+read: Packing.Packed) ->
  {chunk_sources(chunk_result(~continued,count,remaining,window,values,packed,read)) == Sources{Packing.packing_reader(packed),Packing.packed_reader(read)} : Sources}:
  Kernel.WeightWindow{geometry,image,tap,output,old_row,old_column,first} = window
  Packing.Packing{input,weights,index,row,column,columns,patch,h,w,s,p} = packed
  Packing.Packed{gradient,tiles} = read
  {==}

def chunk_sources_step(~continued: Bool,+state: ChunkModel) -> {chunk_sources(chunk(~continued,state)) == chunk_sources(state) : Sources}:
  ChunkModel{+input,+gradient,+values,+remaining,+window} = state
  +count = U32.min(256,remaining)
  +packed = Packing.window_input(window,count,input)
  +read = Packing.window_gradient(window,count,gradient)
  %Equal.sym(Sources,chunk_sources(chunk_result(~continued,count,remaining,window,values,packed,read)),Sources{Packing.packing_reader(packed),Packing.packed_reader(read)},
    chunk_result_sources(~continued,count,remaining,window,values,packed,read)) : {_ == Sources{input,gradient} : Sources}
  %Equal.sym(Buffer.Model<F32>,Packing.packing_reader(packed),input,Packing.window_input_reader(window,count,input)) :
    {Sources{_,Packing.packed_reader(read)} == Sources{input,gradient} : Sources}
  %Equal.sym(Buffer.Model<F32>,Packing.packed_reader(read),gradient,Packing.window_gradient_kept(window,count,gradient)) :
    {Sources{input,_} == Sources{input,gradient} : Sources}
  {==}

def chunk_group_sources(+state: ChunkModel) -> {group_sources(chunk_values(state)) == chunk_sources(state) : Sources}:
  ChunkModel{input,gradient,values,remaining,window} = state
  {==}

def window_product_sources(+count: U32,+window: Kernel.WeightWindow,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {group_sources(window_product(count,window,input,gradient)) == Sources{input,gradient} : Sources}:
  +initial = {ChunkModel{input,gradient,Tree.fresh(F32,Storage.depth_for(64),0.0),count,window} : ChunkModel}
  +first = chunk(~False{},initial)
  +steps = U32.to_nat((((count - 1) / 256 + 1) - 1 : U32))
  +state = Loop.run(~ChunkModel,~chunk(~True{}),steps,first)
  %Equal.sym(Sources,group_sources(chunk_values(state)),chunk_sources(state),chunk_group_sources(state)) : {_ == Sources{input,gradient} : Sources}
  Equal.trans(Sources,chunk_sources(state),chunk_sources(first),Sources{input,gradient},
    Projection.preserves_observation(~ChunkModel,~Sources,~chunk(~True{}),~chunk_sources,~chunk_sources_step(~True{}),steps,first),chunk_sources_step(~False{},initial))

type CopyOwners is Data:
  CopyOwners{input: Tree.Tree<F32>,output: Tree.Tree<F32>}

def represent_copy(state: CopyOwners) -> Array<F32> & Array<F32>:
  CopyOwners{input,output} = state
  (Tree.pack(F32,input),Tree.pack(F32,output))

def copy_write(valid: Bool,address: U32,output: Tree.Tree<F32>,value: F32) -> Tree.Tree<F32>:
  match valid:
    case False{}: output
    case True{}: Tree.write(F32,output,address,value)

def copy_write_matches(valid: Bool,+address: U32,+output: Tree.Tree<F32>,+value: F32) ->
  {Gemm.store_lane(valid,Tree.pack(F32,output),address,value) == Tree.pack(F32,copy_write(valid,address,output,value)) : Array<F32>}:
  match valid:
    case False{}: {==}
    case True{}: StorageLaws.write(F32,output,address,value)

def copy_cell(index: U32,context: Kernel.BlockCopy,state: CopyOwners) -> CopyOwners:
  Kernel.BlockCopy{+tap,+output,taps,+outputs} = context
  CopyOwners{+input,destination} = state
  +index = index
  +row = (tap + index / 8 : U32)
  +column = (output + index % 8 : U32)
  CopyOwners{input,copy_write(U32.is_lt(row,taps) && U32.is_lt(column,outputs),(row * outputs + column : U32),destination,Tree.read(F32,input,index))}

def copy_cell_matches(+index: U32,+context: Kernel.BlockCopy,+state: CopyOwners) ->
  {Kernel.copy_cell(index,context,represent_copy(state)) == represent_copy(copy_cell(index,context,state)) : Array<F32> & Array<F32>}:
  Kernel.BlockCopy{+tap,+output,taps,+outputs} = context
  CopyOwners{+input,+destination} = state
  +row = (tap + index / 8 : U32)
  +column = (output + index % 8 : U32)
  +valid = U32.is_lt(row,taps) && U32.is_lt(column,outputs)
  +address = (row * outputs + column : U32)
  +value = Tree.read(F32,input,index)
  %Equal.sym(Array<F32> & F32,Array.get(F32,Tree.pack(F32,input),index),(Tree.pack(F32,input),value),StorageLaws.read(F32,input,index)) :
    {Kernel.copied(valid,address,Tree.pack(F32,destination),_) == represent_copy(copy_cell(index,context,state)) : Array<F32> & Array<F32>}
  Equal.cong(Array<F32>,Array<F32> & Array<F32>,output => (Tree.pack(F32,input),output),Gemm.store_lane(valid,Tree.pack(F32,destination),address,value),
    Tree.pack(F32,copy_write(valid,address,destination,value)),copy_write_matches(valid,address,destination,value))

def copy_output(state: CopyOwners) -> Tree.Tree<F32>:
  CopyOwners{input,output} = state
  output

def copy_output_matches(+state: CopyOwners) ->
  {Pair.snd(Array<F32>,Array<F32>,represent_copy(state)) == Tree.pack(F32,copy_output(state)) : Array<F32>}:
  CopyOwners{input,output} = state
  {==}

def copy_address(context: Kernel.BlockCopy,+index: U32) -> U32:
  Kernel.BlockCopy{tap,output,taps,outputs} = context
  ((tap + index / 8) * outputs + (output + index % 8) : U32)

def copy_valid(context: Kernel.BlockCopy,+index: U32) -> Bool:
  Kernel.BlockCopy{tap,output,taps,outputs} = context
  U32.is_lt((tap + index / 8 : U32),taps) && U32.is_lt((output + index % 8 : U32),outputs)

def copy_source(state: CopyOwners) -> Tree.Tree<F32>:
  CopyOwners{input,output} = state
  input

def copy_write_balanced(valid: Bool,+depth: Nat,+tree: Tree.Tree<F32>,+address: U32,+value: F32,
  balanced: Shape.Balanced(F32,depth,tree)) -> Shape.Balanced(F32,depth,copy_write(valid,address,tree,value)):
  match valid:
    case False{}: balanced
    case True{}: Shape.write(F32,depth,tree,address,value,balanced)

def copy_write_capacity(valid: Bool,+address: U32,+tree: Tree.Tree<F32>,+value: F32) ->
  {Tree.capacity(F32,copy_write(valid,address,tree,value)) == Tree.capacity(F32,tree) : U32}:
  match valid:
    case False{}: {==}
    case True{}: StorageReadback.write_capacity(~F32,tree,address,value)

def copy_selected(valid: Bool,+address: U32,+query: U32,+tree: Tree.Tree<F32>,+value: F32,+depth: Nat,
  balanced: Shape.Balanced(F32,depth,tree),limit: {Nat.is_le(depth,30n) == True{} : Bool},
  written_inside: Logic.When(valid,{U32.is_lt(address,Tree.capacity(F32,tree)) == True{} : Bool}),
  query_inside: {U32.is_lt(query,Tree.capacity(F32,tree)) == True{} : Bool}) ->
  {Tree.read(F32,copy_write(valid,address,tree,value),query) == Bool.pick(F32,valid && U32.is_eq(address,query),value,Tree.read(F32,tree,query)) : F32}:
  match valid:
    case False{}: {==}
    case True{}: ProgramRead.one(~F32,depth,tree,address,query,value,balanced,limit,written_inside,query_inside)

def pick_matching(enabled: Bool,+actual: F32,+wanted: F32,+old: F32,
  matching: Logic.When(enabled,{actual == wanted : F32}),previous: {old == wanted : F32}) ->
  {Bool.pick(F32,enabled,actual,old) == wanted : F32}:
  match enabled:
    case False{}: previous
    case True{}: matching

def copy_matching(+valid: Bool,+address: U32,+query: U32,+tree: Tree.Tree<F32>,+value: F32,+wanted: F32,+depth: Nat,
  balanced: Shape.Balanced(F32,depth,tree),limit: {Nat.is_le(depth,30n) == True{} : Bool},
  written_inside: Logic.When(valid,{U32.is_lt(address,Tree.capacity(F32,tree)) == True{} : Bool}),
  query_inside: {U32.is_lt(query,Tree.capacity(F32,tree)) == True{} : Bool},
  matching: Logic.When(valid && U32.is_eq(address,query),{value == wanted : F32}),previous: {Tree.read(F32,tree,query) == wanted : F32}) ->
  {Tree.read(F32,copy_write(valid,address,tree,value),query) == wanted : F32}:
  Equal.trans(F32,Tree.read(F32,copy_write(valid,address,tree,value),query),Bool.pick(F32,valid && U32.is_eq(address,query),value,Tree.read(F32,tree,query)),wanted,
    copy_selected(valid,address,query,tree,value,depth,balanced,limit,written_inside,query_inside),
    pick_matching(valid && U32.is_eq(address,query),value,wanted,Tree.read(F32,tree,query),matching,previous))

def copy_output_cell(+index: U32,+context: Kernel.BlockCopy,+state: CopyOwners) ->
  {copy_output(copy_cell(index,context,state)) ==
    copy_write(copy_valid(context,index),copy_address(context,index),copy_output(state),Tree.read(F32,copy_source(state),index)) : Tree.Tree<F32>}:
  Kernel.BlockCopy{+tap,+output,taps,+outputs} = context
  CopyOwners{input,destination} = state
  {==}

def copy_row(context: Kernel.BlockCopy,index: U32) -> U32:
  Kernel.BlockCopy{tap,output,taps,outputs} = context
  (tap + index / 8 : U32)

def copy_column(context: Kernel.BlockCopy,index: U32) -> U32:
  Kernel.BlockCopy{tap,output,taps,outputs} = context
  (output + index % 8 : U32)

def copy_width(context: Kernel.BlockCopy) -> U32:
  Kernel.BlockCopy{tap,output,taps,outputs} = context
  outputs

def CopyExtent(context: Kernel.BlockCopy,capacity: U32) -> Data:
  Kernel.BlockCopy{tap,output,taps,outputs} = context
  {Nat.is_le(Nat.mul(U32.to_nat(taps),U32.to_nat(outputs)),U32.to_nat(capacity)) == True{} : Bool}

def copy_linear(context: Kernel.BlockCopy,+index: U32) -> Nat:
  Kernel.BlockCopy{+tap,+output,taps,+outputs} = context
  Nat.add(Nat.mul(U32.to_nat((tap + index / 8 : U32)),U32.to_nat(outputs)),U32.to_nat((output + index % 8 : U32)))

def copy_linear_room(+context: Kernel.BlockCopy,+index: U32,+capacity: U32,
  +extent: CopyExtent(context,capacity),+valid: {copy_valid(context,index) == True{} : Bool}) ->
  {Nat.is_lt(copy_linear(context,index),U32.to_nat(capacity)) == True{} : Bool}:
  Kernel.BlockCopy{+tap,+output,+taps,+outputs} = context
  +row = (tap + index / 8 : U32)
  +column = (output + index % 8 : U32)
  Bounds.nat_trans(copy_linear(context,index),Nat.mul(U32.to_nat(taps),U32.to_nat(outputs)),U32.to_nat(capacity),
    RowMajor.radix(U32.to_nat(row),U32.to_nat(taps),U32.to_nat(column),U32.to_nat(outputs),
      %Comparison.lt(row,taps) : {_ == True{} : Bool} Logic.left(U32.is_lt(row,taps),U32.is_lt(column,outputs),valid),
      %Comparison.lt(column,outputs) : {_ == True{} : Bool} Logic.right(U32.is_lt(row,taps),U32.is_lt(column,outputs),valid)),extent)

def copy_address_encoded(+context: Kernel.BlockCopy,+index: U32) ->
  {copy_address(context,index) == U32.from_nat(copy_linear(context,index)) : U32}:
  Kernel.BlockCopy{+tap,+output,taps,+outputs} = context
  +row = (tap + index / 8 : U32)
  +column = (output + index % 8 : U32)
  Encoded.native_madd(row,outputs,column)

def copy_address_exact(+context: Kernel.BlockCopy,+index: U32,+capacity: U32,
  +extent: CopyExtent(context,capacity),+valid: {copy_valid(context,index) == True{} : Bool}) ->
  {U32.to_nat(copy_address(context,index)) == copy_linear(context,index) : Nat}:
  %Equal.sym(U32,copy_address(context,index),U32.from_nat(copy_linear(context,index)),copy_address_encoded(context,index)) :
    {U32.to_nat(_) == copy_linear(context,index) : Nat}
  FullRange.roundtrip(copy_linear(context,index),capacity,Bounds.strict_le(copy_linear(context,index),U32.to_nat(capacity),copy_linear_room(context,index,capacity,extent,valid)))

def copy_written_inside(+context: Kernel.BlockCopy,+index: U32,+capacity: U32,
  +extent: CopyExtent(context,capacity),+valid: {copy_valid(context,index) == True{} : Bool}) ->
  {U32.is_lt(copy_address(context,index),capacity) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt(copy_address(context,index),capacity),Nat.is_lt(U32.to_nat(copy_address(context,index)),U32.to_nat(capacity)),Comparison.lt(copy_address(context,index),capacity)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(copy_address(context,index)),copy_linear(context,index),copy_address_exact(context,index,capacity,extent,valid)) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
  copy_linear_room(context,index,capacity,extent,valid)

def add_cancel_left(+base: U32,+first: U32,+second: U32,+same: {(base + first : U32) == (base + second : U32) : U32}) ->
  {first == second : U32}:
  %Subtraction.subtract_add(first,base) : {_ == second : U32}
  %Subtraction.subtract_add(second,base) : {((first + base : U32) - base : U32) == _ : U32}
  %Equal.sym(U32,(first + base : U32),(base + first : U32),Arithmetic.add_commutative(first,base)) : {((_ : U32) - base : U32) == ((second + base : U32) - base : U32) : U32}
  %Equal.sym(U32,(second + base : U32),(base + second : U32),Arithmetic.add_commutative(second,base)) : {((base + first : U32) - base : U32) == ((_ : U32) - base : U32) : U32}
  Equal.cong(U32,U32,value => (value - base : U32),(base + first : U32),(base + second : U32),same)

def eight_parts_unique(+first: U32,+second: U32,+rows: {(first / 8 : U32) == (second / 8 : U32) : U32},
  +columns: {(first % 8 : U32) == (second % 8 : U32) : U32}) -> {first == second : U32}:
  +fq = NativeDivision.divide(first,8,2147483648,{==},{==},{==})
  +sq = NativeDivision.divide(second,8,2147483648,{==},{==},{==})
  +fr = NativeDivision.modulo(first,8,2147483648,{==},{==},{==})
  +sr = NativeDivision.modulo(second,8,2147483648,{==},{==},{==})
  Arithmetic.to_nat_injective(first,second,
    %Equal.sym(Nat,U32.to_nat(first),Nat.add(Nat.mul(Nat.div(U32.to_nat(first),8n),8n),Nat.mod(U32.to_nat(first),8n)),Radix.division_identity(first,8n,{==})) : {_ == U32.to_nat(second) : Nat}
    %Equal.sym(Nat,U32.to_nat(second),Nat.add(Nat.mul(Nat.div(U32.to_nat(second),8n),8n),Nat.mod(U32.to_nat(second),8n)),Radix.division_identity(second,8n,{==})) : {Nat.add(Nat.mul(Nat.div(U32.to_nat(first),8n),8n),Nat.mod(U32.to_nat(first),8n)) == _ : Nat}
    %fq : {Nat.add(Nat.mul(_,8n),Nat.mod(U32.to_nat(first),8n)) == Nat.add(Nat.mul(Nat.div(U32.to_nat(second),8n),8n),Nat.mod(U32.to_nat(second),8n)) : Nat}
    %sq : {Nat.add(Nat.mul(U32.to_nat((first / 8 : U32)),8n),Nat.mod(U32.to_nat(first),8n)) == Nat.add(Nat.mul(_,8n),Nat.mod(U32.to_nat(second),8n)) : Nat}
    %fr : {Nat.add(Nat.mul(U32.to_nat((first / 8 : U32)),8n),_) == Nat.add(Nat.mul(U32.to_nat((second / 8 : U32)),8n),Nat.mod(U32.to_nat(second),8n)) : Nat}
    %sr : {Nat.add(Nat.mul(U32.to_nat((first / 8 : U32)),8n),U32.to_nat((first % 8 : U32))) == Nat.add(Nat.mul(U32.to_nat((second / 8 : U32)),8n),_) : Nat}
    %Equal.sym(U32,(first % 8 : U32),(second % 8 : U32),columns) : {Nat.add(Nat.mul(U32.to_nat((first / 8 : U32)),8n),U32.to_nat(_)) == Nat.add(Nat.mul(U32.to_nat((second / 8 : U32)),8n),U32.to_nat((second % 8 : U32))) : Nat}
    %Equal.sym(U32,(first / 8 : U32),(second / 8 : U32),rows) : {Nat.add(Nat.mul(U32.to_nat(_),8n),U32.to_nat((second % 8 : U32))) == Nat.add(Nat.mul(U32.to_nat((second / 8 : U32)),8n),U32.to_nat((second % 8 : U32))) : Nat}
    {==})

def copy_index_unique(+context: Kernel.BlockCopy,+first: U32,+second: U32,+capacity: U32,
  +extent: CopyExtent(context,capacity),+positive: {Nat.is_lt(0n,U32.to_nat(copy_width(context))) == True{} : Bool},
  +first_valid: {copy_valid(context,first) == True{} : Bool},+second_valid: {copy_valid(context,second) == True{} : Bool},
  +same: {copy_address(context,first) == copy_address(context,second) : U32}) -> {first == second : U32}:
  Kernel.BlockCopy{+tap,+output,+taps,+outputs} = context
  +a = (tap + first / 8 : U32)
  +b = (tap + second / 8 : U32)
  +x = (output + first % 8 : U32)
  +y = (output + second % 8 : U32)
  +hx = {%Comparison.lt(x,outputs) : {_ == True{} : Bool} Logic.right(U32.is_lt(a,taps),U32.is_lt(x,outputs),first_valid) : {Nat.is_lt(U32.to_nat(x),U32.to_nat(outputs)) == True{} : Bool}}
  +hy = {%Comparison.lt(y,outputs) : {_ == True{} : Bool} Logic.right(U32.is_lt(b,taps),U32.is_lt(y,outputs),second_valid) : {Nat.is_lt(U32.to_nat(y),U32.to_nat(outputs)) == True{} : Bool}}
  +linear = {
    %copy_address_exact(context,first,capacity,extent,first_valid) : {_ == copy_linear(context,second) : Nat}
    %copy_address_exact(context,second,capacity,extent,second_valid) : {U32.to_nat(copy_address(context,first)) == _ : Nat}
    Equal.cong(U32,Nat,U32.to_nat,copy_address(context,first),copy_address(context,second),same) : {copy_linear(context,first) == copy_linear(context,second) : Nat}}
  eight_parts_unique(first,second,
    add_cancel_left(tap,(first / 8 : U32),(second / 8 : U32),Arithmetic.to_nat_injective(a,b,Radix.row_unique(U32.to_nat(a),U32.to_nat(b),U32.to_nat(outputs),U32.to_nat(x),U32.to_nat(y),positive,hx,hy,linear))),
    add_cancel_left(output,(first % 8 : U32),(second % 8 : U32),Arithmetic.to_nat_injective(x,y,Radix.col_unique(U32.to_nat(a),U32.to_nat(b),U32.to_nat(outputs),U32.to_nat(x),U32.to_nat(y),positive,hx,hy,linear))))

def copy_unique_nat(+copy: Kernel.BlockCopy,+index: Nat,+target: Nat,+capacity: U32,
  +extent: CopyExtent(copy,capacity),+positive: {Nat.is_lt(0n,U32.to_nat(copy_width(copy))) == True{} : Bool},
  +valid: {copy_valid(copy,U32.from_nat(index)) == True{} : Bool},+target_valid: {copy_valid(copy,U32.from_nat(target)) == True{} : Bool},
  +target_inside: {Nat.is_lt(target,64n) == True{} : Bool},+inside: {Nat.is_lt(index,64n) == True{} : Bool},
  +same: {copy_address(copy,U32.from_nat(index)) == copy_address(copy,U32.from_nat(target)) : U32}) -> {index == target : Nat}:
  %FullRange.roundtrip(index,64,Bounds.strict_le(index,64n,inside)) : {_ == target : Nat}
  %FullRange.roundtrip(target,64,Bounds.strict_le(target,64n,target_inside)) : {U32.to_nat(U32.from_nat(index)) == _ : Nat}
  Equal.cong(U32,Nat,U32.to_nat,U32.from_nat(index),U32.from_nat(target),copy_index_unique(copy,U32.from_nat(index),U32.from_nat(target),capacity,extent,positive,valid,target_valid,same))

def micro_index(upper: Bool,row: Accumulator.Row,lane: Nat) -> Nat:
  Nat.add(Nat.mul(Packing.patch_key(upper,row),8n),lane)

def micro_inside(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Nat.is_lt(micro_index(upper,row,lane),64n) == True{} : Bool}:
  RowMajor.radix(Packing.patch_key(upper,row),8n,lane,8n,Packing.patch_key_inside(upper,row),inside)

def micro_encoded(upper: Bool,row: Accumulator.Row,+lane: Nat,+inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {GemmWrites.half_address(upper,row,lane) == U32.from_nat(micro_index(upper,row,lane)) : U32}:
  match upper row:
    case False{} Accumulator.R0{}: GemmWrites.encoded_lane_address(0,lane,inside)
    case False{} Accumulator.R1{}: GemmWrites.encoded_lane_address(8,lane,inside)
    case False{} Accumulator.R2{}: GemmWrites.encoded_lane_address(16,lane,inside)
    case False{} Accumulator.R3{}: GemmWrites.encoded_lane_address(24,lane,inside)
    case True{} Accumulator.R0{}: GemmWrites.encoded_lane_address(32,lane,inside)
    case True{} Accumulator.R1{}: GemmWrites.encoded_lane_address(40,lane,inside)
    case True{} Accumulator.R2{}: GemmWrites.encoded_lane_address(48,lane,inside)
    case True{} Accumulator.R3{}: GemmWrites.encoded_lane_address(56,lane,inside)

def micro_exact(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(micro_index(upper,row,lane))) == micro_index(upper,row,lane) : Nat}:
  FullRange.roundtrip(micro_index(upper,row,lane),64,Bounds.strict_le(micro_index(upper,row,lane),64n,micro_inside(upper,row,lane,inside)))

def micro_row(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {(U32.from_nat(micro_index(upper,row,lane)) / 8 : U32) == U32.from_nat(Packing.patch_key(upper,row)) : U32}:
  Arithmetic.to_nat_injective((U32.from_nat(micro_index(upper,row,lane)) / 8 : U32),U32.from_nat(Packing.patch_key(upper,row)),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(Packing.patch_key(upper,row))),Packing.patch_key(upper,row),FullRange.roundtrip(Packing.patch_key(upper,row),8,Bounds.strict_le(Packing.patch_key(upper,row),8n,Packing.patch_key_inside(upper,row)))) :
      {U32.to_nat((U32.from_nat(micro_index(upper,row,lane)) / 8 : U32)) == _ : Nat}
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(micro_index(upper,row,lane)) / 8 : U32)),Nat.div(U32.to_nat(U32.from_nat(micro_index(upper,row,lane))),8n),NativeDivision.divide(U32.from_nat(micro_index(upper,row,lane)),8,2147483648,{==},{==},{==})) :
      {_ == Packing.patch_key(upper,row) : Nat}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(micro_index(upper,row,lane))),micro_index(upper,row,lane),micro_exact(upper,row,lane,inside)) : {Nat.div(_,8n) == Packing.patch_key(upper,row) : Nat}
    Radix.quotient(Packing.patch_key(upper,row),lane,8n,{==},inside))

def micro_column(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {(U32.from_nat(micro_index(upper,row,lane)) % 8 : U32) == U32.from_nat(lane) : U32}:
  Arithmetic.to_nat_injective((U32.from_nat(micro_index(upper,row,lane)) % 8 : U32),U32.from_nat(lane),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,FullRange.roundtrip(lane,8,Bounds.strict_le(lane,8n,inside))) :
      {U32.to_nat((U32.from_nat(micro_index(upper,row,lane)) % 8 : U32)) == _ : Nat}
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(micro_index(upper,row,lane)) % 8 : U32)),Nat.mod(U32.to_nat(U32.from_nat(micro_index(upper,row,lane))),8n),NativeDivision.modulo(U32.from_nat(micro_index(upper,row,lane)),8,2147483648,{==},{==},{==})) : {_ == lane : Nat}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(micro_index(upper,row,lane))),micro_index(upper,row,lane),micro_exact(upper,row,lane,inside)) : {Nat.mod(_,8n) == lane : Nat}
    Radix.remainder(Packing.patch_key(upper,row),lane,8n,{==},inside))

type CopyQuery is Data:
  CopyQuery{copy: Kernel.BlockCopy,target: Nat,query: U32,source: Tree.Tree<F32>,depth: Nat,capacity: U32,
    depth_limit: {Nat.is_le(depth,31n) == True{} : Bool},
    query_inside: {U32.is_lt(query,capacity) == True{} : Bool},
    target_inside: {Nat.is_lt(target,64n) == True{} : Bool},
    target_valid: {copy_valid(copy,U32.from_nat(target)) == True{} : Bool},
    target_address: {copy_address(copy,U32.from_nat(target)) == query : U32},
    written_inside: CopyExtent(copy,capacity),
    unique: {Nat.is_lt(0n,U32.to_nat(copy_width(copy))) == True{} : Bool}}

def copy_context(context: CopyQuery) -> Kernel.BlockCopy:
  CopyQuery{copy,target,query,source,depth,capacity,depth_limit,query_inside,target_inside,target_valid,target_address,written_inside,unique} = context
  copy

def copy_target(context: CopyQuery) -> Nat:
  CopyQuery{copy,target,query,source,depth,capacity,depth_limit,query_inside,target_inside,target_valid,target_address,written_inside,unique} = context
  target

def copy_limit(context: CopyQuery) -> Nat:
  64n

def copy_wanted(context: CopyQuery) -> F32:
  CopyQuery{copy,target,query,source,depth,capacity,depth_limit,query_inside,target_inside,target_valid,target_address,written_inside,unique} = context
  Tree.read(F32,source,U32.from_nat(target))

def copy_read(context: CopyQuery,state: CopyOwners) -> F32:
  CopyQuery{copy,target,query,source,depth,capacity,depth_limit,query_inside,target_inside,target_valid,target_address,written_inside,unique} = context
  Tree.read(F32,copy_output(state),query)

def CopyFrame(context: CopyQuery,state: CopyOwners) -> Data:
  CopyQuery{copy,target,query,source,depth,capacity,depth_limit,query_inside,target_inside,target_valid,target_address,written_inside,unique} = context
  CopyOwners{input,output} = state
  Shape.Both<Shape.Balanced(F32,depth,output),Shape.Both<{Tree.capacity(F32,output) == capacity : U32},{input == source : Tree.Tree<F32>}>>

def copy_update(index: U32,context: CopyQuery,state: CopyOwners) -> CopyOwners:
  copy_cell(index,copy_context(context),state)

def copy_frame_step(+context: CopyQuery,+index: U32,+state: CopyOwners,frame: CopyFrame(context,state)) ->
  CopyFrame(context,copy_update(index,context,state)):
  CopyQuery{+copy,target,query,source,+depth,+capacity,depth_limit,query_inside,target_inside,target_valid,target_address,written_inside,unique} = context
  Kernel.BlockCopy{tap,output,taps,outputs} = copy
  CopyOwners{+input,+destination} = state
  Shape.Both{+balanced,Shape.Both{+capacity_equal,source_equal}} = frame
  +valid = copy_valid(copy,index)
  +address = copy_address(copy,index)
  +value = Tree.read(F32,input,index)
  Shape.Both{copy_write_balanced(valid,depth,destination,address,value,balanced),Shape.Both{
    Equal.trans(U32,Tree.capacity(F32,copy_write(valid,address,destination,value)),Tree.capacity(F32,destination),capacity,
      copy_write_capacity(valid,address,destination,value),capacity_equal),source_equal}}

def copy_at_target(+context: CopyQuery,+state: CopyOwners,frame: CopyFrame(context,state)) ->
  {copy_read(context,copy_update(U32.from_nat(copy_target(context)),context,state)) == copy_wanted(context) : F32}:
  CopyQuery{+copy,+target,+query,+source,depth,capacity,depth_limit,query_inside,target_inside,+target_valid,+target_address,written_inside,unique} = context
  CopyOwners{+input,+output} = state
  Shape.Both{balanced,Shape.Both{capacity_equal,+source_equal}} = frame
  %Equal.sym(Tree.Tree<F32>,copy_output(copy_cell(U32.from_nat(target),copy,state)),
    copy_write(copy_valid(copy,U32.from_nat(target)),copy_address(copy,U32.from_nat(target)),output,Tree.read(F32,input,U32.from_nat(target))),
    copy_output_cell(U32.from_nat(target),copy,state)) : {Tree.read(F32,_,query) == Tree.read(F32,source,U32.from_nat(target)) : F32}
  %Equal.sym(Bool,copy_valid(copy,U32.from_nat(target)),True{},target_valid) :
    {Tree.read(F32,copy_write(_,copy_address(copy,U32.from_nat(target)),output,Tree.read(F32,input,U32.from_nat(target))),query) == Tree.read(F32,source,U32.from_nat(target)) : F32}
  %Equal.sym(U32,copy_address(copy,U32.from_nat(target)),query,target_address) :
    {Tree.read(F32,Tree.write(F32,output,_,Tree.read(F32,input,U32.from_nat(target))),query) == Tree.read(F32,source,U32.from_nat(target)) : F32}
  %Equal.sym(Tree.Tree<F32>,input,source,source_equal) :
    {Tree.read(F32,Tree.write(F32,output,query,Tree.read(F32,_,U32.from_nat(target))),query) == Tree.read(F32,source,U32.from_nat(target)) : F32}
  StorageReadback.read_written(~F32,output,query,Tree.read(F32,source,U32.from_nat(target)))

def copy_distinct(+target: Nat,+index: Nat,+address: U32,+query: U32,
  +after: {Nat.is_lt(target,index) == True{} : Bool},
  unique: {address == query : U32} -> {index == target : Nat},
  equal: Bool,+compared: {U32.is_eq(address,query) == equal : Bool}) -> {equal == False{} : Bool}:
  match equal:
    case False{}: {==}
    case True{}:
      +same = unique(Arithmetic.equal_of_is_eq(address,query,compared))
      Empty.absurd({True{} == False{} : Bool},Words.false_true(
        Equal.trans(Bool,False{},Nat.is_lt(target,target),True{},
          Equal.sym(Bool,Nat.is_lt(target,target),False{},Order.not_below(target,target,Order.reflexive(target))),
          %same : {Nat.is_lt(target,_) == True{} : Bool} after)))

def copy_later_valid(+context: CopyQuery,+index: Nat,+state: CopyOwners,+frame: CopyFrame(context,state),
  +after: {Nat.is_lt(copy_target(context),index) == True{} : Bool},+inside: {Nat.is_lt(index,64n) == True{} : Bool},
  enabled: Bool,+valid: {copy_valid(copy_context(context),U32.from_nat(index)) == enabled : Bool}) ->
  {copy_read(context,copy_update(U32.from_nat(index),context,state)) == copy_read(context,state) : F32}:
  CopyQuery{+copy,+target,+query,source,+depth,+capacity,+depth_limit,+query_inside,+target_inside,+target_valid,+target_address,+written_inside,+unique} = context
  CopyOwners{+input,+output} = state
  Shape.Both{+balanced,Shape.Both{+capacity_equal,source_equal}} = frame
  match enabled:
    case False{}:
      %Equal.sym(Tree.Tree<F32>,copy_output(copy_cell(U32.from_nat(index),copy,state)),
        copy_write(copy_valid(copy,U32.from_nat(index)),copy_address(copy,U32.from_nat(index)),output,Tree.read(F32,input,U32.from_nat(index))),
        copy_output_cell(U32.from_nat(index),copy,state)) : {Tree.read(F32,_,query) == Tree.read(F32,output,query) : F32}
      %Equal.sym(Bool,copy_valid(copy,U32.from_nat(index)),enabled,valid) :
        {Tree.read(F32,copy_write(_,copy_address(copy,U32.from_nat(index)),output,Tree.read(F32,input,U32.from_nat(index))),query) == Tree.read(F32,output,query) : F32}
      {==}
    case True{}:
      %Equal.sym(Tree.Tree<F32>,copy_output(copy_cell(U32.from_nat(index),copy,state)),
        copy_write(copy_valid(copy,U32.from_nat(index)),copy_address(copy,U32.from_nat(index)),output,Tree.read(F32,input,U32.from_nat(index))),
        copy_output_cell(U32.from_nat(index),copy,state)) : {Tree.read(F32,_,query) == Tree.read(F32,output,query) : F32}
      %Equal.sym(Bool,copy_valid(copy,U32.from_nat(index)),enabled,valid) :
        {Tree.read(F32,copy_write(_,copy_address(copy,U32.from_nat(index)),output,Tree.read(F32,input,U32.from_nat(index))),query) == Tree.read(F32,output,query) : F32}
      +address = copy_address(copy,U32.from_nat(index))
      StorageReadback.read_preserved_full(~F32,depth,output,address,query,Tree.read(F32,input,U32.from_nat(index)),balanced,depth_limit,
        %Equal.sym(U32,Tree.capacity(F32,output),capacity,capacity_equal) : {U32.is_lt(address,_) == True{} : Bool} copy_written_inside(copy,U32.from_nat(index),capacity,written_inside,valid),
        %Equal.sym(U32,Tree.capacity(F32,output),capacity,capacity_equal) : {U32.is_lt(query,_) == True{} : Bool} query_inside,
        copy_distinct(target,index,address,query,after,same => copy_unique_nat(copy,index,target,capacity,written_inside,unique,valid,target_valid,target_inside,inside,Equal.trans(U32,copy_address(copy,U32.from_nat(index)),query,copy_address(copy,U32.from_nat(target)),same,Equal.sym(U32,copy_address(copy,U32.from_nat(target)),query,target_address))),U32.is_eq(address,query),{==}))

def copy_later(+context: CopyQuery,+index: Nat,+state: CopyOwners,frame: CopyFrame(context,state),
  after: {Nat.is_lt(copy_target(context),index) == True{} : Bool},inside: {Nat.is_lt(index,copy_limit(context)) == True{} : Bool}) ->
  {copy_read(context,copy_update(U32.from_nat(index),context,state)) == copy_read(context,state) : F32}:
  copy_later_valid(context,index,state,frame,after,inside,copy_valid(copy_context(context),U32.from_nat(index)),{==})

def copy_readback(+context: CopyQuery,+state: CopyOwners,+frame: CopyFrame(context,state)) ->
  {copy_read(context,Traversal.indexed_fold(~CopyOwners,~CopyQuery,~copy_update,64n,0,context,state)) == copy_wanted(context) : F32}:
  CopyQuery{copy,target,query,source,depth,capacity,depth_limit,query_inside,+target_inside,target_valid,target_address,written_inside,unique} = context
  IndexedReadback.complete_readback(~CopyOwners,~CopyQuery,~F32,~copy_limit,~copy_target,~copy_wanted,~CopyFrame,~copy_read,~copy_update,
    ~copy_frame_step,~copy_at_target,~copy_later,context,state,frame,target_inside)

def copy_update_inner(+index: U32,+context: CopyQuery,+state: CopyOwners) ->
  {copy_update(index,context,state) == copy_cell(index,copy_context(context),state) : CopyOwners}:
  {==}

def copy_query_depth(context: CopyQuery) -> Nat:
  CopyQuery{copy,target,query,source,depth,capacity,depth_limit,query_inside,target_inside,target_valid,target_address,written_inside,unique} = context
  depth

def copy_query_capacity(context: CopyQuery) -> U32:
  CopyQuery{copy,target,query,source,depth,capacity,depth_limit,query_inside,target_inside,target_valid,target_address,written_inside,unique} = context
  capacity

def copy_query_source(context: CopyQuery) -> Tree.Tree<F32>:
  CopyQuery{copy,target,query,source,depth,capacity,depth_limit,query_inside,target_inside,target_valid,target_address,written_inside,unique} = context
  source

def copy_query_address(context: CopyQuery) -> U32:
  CopyQuery{copy,target,query,source,depth,capacity,depth_limit,query_inside,target_inside,target_valid,target_address,written_inside,unique} = context
  query

def copy_fold_readback(+context: CopyQuery,+output: Tree.Tree<F32>,
  balanced: Shape.Balanced(F32,copy_query_depth(context),output),
  capacity_equal: {Tree.capacity(F32,output) == copy_query_capacity(context) : U32}) ->
  {Tree.read(F32,copy_output(Traversal.indexed_fold(~CopyOwners,~Kernel.BlockCopy,~copy_cell,64n,0,copy_context(context),CopyOwners{copy_query_source(context),output})),copy_query_address(context)) == copy_wanted(context) : F32}:
  CopyQuery{+copy,+target,+query,+source,depth,capacity,depth_limit,query_inside,target_inside,target_valid,target_address,written_inside,unique} = context
  +initial = {CopyOwners{source,output} : CopyOwners}
  +outer = Traversal.indexed_fold(~CopyOwners,~CopyQuery,~copy_update,64n,0,context,initial)
  +inner = Traversal.indexed_fold(~CopyOwners,~Kernel.BlockCopy,~copy_cell,64n,0,copy,initial)
  %Traversals.indexed_fold_change_context(~CopyOwners,~CopyQuery,~Kernel.BlockCopy,~copy_context,~copy_update,~copy_cell,~copy_update_inner,64n,0,context,initial) :
    {Tree.read(F32,copy_output(_),query) == Tree.read(F32,source,U32.from_nat(target)) : F32}
  copy_readback(context,initial,Shape.Both{balanced,Shape.Both{capacity_equal,{==}}})

type CopyShape is Data:
  CopyShape{copy: Kernel.BlockCopy,depth: Nat,capacity: U32,source: Tree.Tree<F32>}

def copy_shape_context(context: CopyShape) -> Kernel.BlockCopy:
  CopyShape{copy,depth,capacity,source} = context
  copy

def CopyShapeFrame(context: CopyShape,state: CopyOwners) -> Data:
  CopyShape{copy,depth,capacity,source} = context
  CopyOwners{input,output} = state
  Shape.Both<Shape.Balanced(F32,depth,output),Shape.Both<{Tree.capacity(F32,output) == capacity : U32},{input == source : Tree.Tree<F32>}>>

def CopyShapeInvariant(context: CopyShape,index: Nat,state: CopyOwners) -> Data:
  CopyShapeFrame(context,state)

def copy_shape_limit(context: CopyShape) -> Nat:
  64n

def copy_shape_update(index: U32,context: CopyShape,state: CopyOwners) -> CopyOwners:
  copy_cell(index,copy_shape_context(context),state)

def copy_shape_update_inner(+index: U32,+context: CopyShape,+state: CopyOwners) ->
  {copy_shape_update(index,context,state) == copy_cell(index,copy_shape_context(context),state) : CopyOwners}:
  {==}

def copy_shape_frame_step(+context: CopyShape,+index: U32,+state: CopyOwners,frame: CopyShapeFrame(context,state)) ->
  CopyShapeFrame(context,copy_shape_update(index,context,state)):
  CopyShape{+copy,+depth,+capacity,source} = context
  Kernel.BlockCopy{tap,output,taps,outputs} = copy
  CopyOwners{+input,+destination} = state
  Shape.Both{+balanced,Shape.Both{+capacity_equal,source_equal}} = frame
  +position = index
  +valid = copy_valid(copy,position)
  +address = copy_address(copy,position)
  +value = Tree.read(F32,input,position)
  Shape.Both{copy_write_balanced(valid,depth,destination,address,value,balanced),Shape.Both{
    Equal.trans(U32,Tree.capacity(F32,copy_write(valid,address,destination,value)),Tree.capacity(F32,destination),capacity,
      copy_write_capacity(valid,address,destination,value),capacity_equal),source_equal}}

def copy_shape_step(+context: CopyShape,+index: Nat,+state: CopyOwners,
  inside: {Nat.is_lt(index,copy_shape_limit(context)) == True{} : Bool},frame: CopyShapeInvariant(context,index,state)) ->
  CopyShapeInvariant(context,1n+index,copy_shape_update(U32.from_nat(index),context,state)):
  copy_shape_frame_step(context,U32.from_nat(index),state,frame)

def copy_shape_fold(+context: CopyShape,+state: CopyOwners,frame: CopyShapeFrame(context,state)) ->
  CopyShapeFrame(context,Traversal.indexed_fold(~CopyOwners,~Kernel.BlockCopy,~copy_cell,64n,0,copy_shape_context(context),state)):
  %Traversals.indexed_fold_change_context(~CopyOwners,~CopyShape,~Kernel.BlockCopy,~copy_shape_context,~copy_shape_update,~copy_cell,~copy_shape_update_inner,64n,0,context,state) :
    CopyShapeFrame(context,_)
  Traversals.indexed_fold_preserves_invariant(~CopyOwners,~CopyShape,~copy_shape_limit,~CopyShapeInvariant,~copy_shape_update,~copy_shape_step,64n,0n,context,state,{==},frame)

def StorageFrame(depth: Nat,capacity: U32,tree: Tree.Tree<F32>) -> Data:
  Shape.Both<Shape.Balanced(F32,depth,tree),{Tree.capacity(F32,tree) == capacity : U32}>

def copy_shape_depth(context: CopyShape) -> Nat:
  CopyShape{copy,depth,capacity,source} = context
  depth

def copy_shape_capacity(context: CopyShape) -> U32:
  CopyShape{copy,depth,capacity,source} = context
  capacity

def copy_shape_storage(+context: CopyShape,+state: CopyOwners,frame: CopyShapeFrame(context,state)) ->
  StorageFrame(copy_shape_depth(context),copy_shape_capacity(context),copy_output(state)):
  CopyShape{copy,depth,capacity,source} = context
  CopyOwners{input,output} = state
  Shape.Both{balanced,Shape.Both{capacity_equal,source_equal}} = frame
  Shape.Both{balanced,capacity_equal}

def block_result(context: Kernel.BlockCopy,output: Tree.Tree<F32>,result: Group) -> Group:
  Group{input,gradient,values} = result
  Group{input,gradient,copy_output(Traversal.indexed_fold(~CopyOwners,~Kernel.BlockCopy,~copy_cell,64n,0,context,CopyOwners{values,output}))}

def block_result_matches(+context: Kernel.BlockCopy,+output: Tree.Tree<F32>,+result: Group) ->
  {Kernel.block_result(context,Tree.pack(F32,output),represent(result)) == represent(block_result(context,output,result)) :
    (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}:
  Group{+input,+gradient,+values} = result
  +initial = {CopyOwners{values,output} : CopyOwners}
  +state = Traversal.indexed_fold(~CopyOwners,~Kernel.BlockCopy,~copy_cell,64n,0,context,initial)
  %Equal.sym(Array<F32> & Array<F32>,Traversal.indexed_fold(~(Array<F32> & Array<F32>),~Kernel.BlockCopy,~Kernel.copy_cell,64n,0,context,represent_copy(initial)),represent_copy(state),
    Traversals.indexed_fold_refines_model(~(Array<F32> & Array<F32>),~CopyOwners,~Kernel.BlockCopy,~Kernel.copy_cell,~copy_cell,~represent_copy,~copy_cell_matches,64n,0,context,initial)) :
    {((Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),Pair.snd(Array<F32>,Array<F32>,_)) == represent(block_result(context,output,result)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}
  Equal.cong(Array<F32>,(Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>,output => ((Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),output),
    Pair.snd(Array<F32>,Array<F32>,represent_copy(state)),Tree.pack(F32,copy_output(state)),copy_output_matches(state))

def group_output(result: Group) -> Tree.Tree<F32>:
  Group{input,gradient,values} = result
  values

def block_result_storage(+copy: Kernel.BlockCopy,+output: Tree.Tree<F32>,+result: Group,+depth: Nat,+capacity: U32,
  frame: StorageFrame(depth,capacity,output)) -> StorageFrame(depth,capacity,group_output(block_result(copy,output,result))):
  Group{input,gradient,+values} = result
  Shape.Both{balanced,capacity_equal} = frame
  +context = {CopyShape{copy,depth,capacity,values} : CopyShape}
  +initial = {CopyOwners{values,output} : CopyOwners}
  +state = Traversal.indexed_fold(~CopyOwners,~Kernel.BlockCopy,~copy_cell,64n,0,copy,initial)
  copy_shape_storage(context,state,copy_shape_fold(context,initial,Shape.Both{balanced,Shape.Both{capacity_equal,{==}}}))

def block_result_sources(+copy: Kernel.BlockCopy,+output: Tree.Tree<F32>,+result: Group) ->
  {group_sources(block_result(copy,output,result)) == group_sources(result) : Sources}:
  Group{input,gradient,values} = result
  {==}

def block(index: U32,context: Kernel.GroupBlocks,state: Group) -> Group:
  Kernel.GroupBlocks{Kernel.WeightWindow{+geometry,+image,tap,output,+row,+column,+first},count,+output_blocks} = context
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  Group{input,gradient,values} = state
  +index = index
  +tap = (index / output_blocks * 8 : U32)
  +output = (index % output_blocks * 8 : U32)
  block_result(Kernel.BlockCopy{tap,output,taps,o},values,window_product(count,Kernel.WeightWindow{geometry,image,tap,output,row,column,first},input,gradient))

def block_matches(+index: U32,+context: Kernel.GroupBlocks,+state: Group) ->
  {Kernel.block(index,context,represent(state)) == represent(block(index,context,state)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}:
  Kernel.GroupBlocks{Kernel.WeightWindow{+geometry,+image,tap,output,+row,+column,+first},+count,+output_blocks} = context
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  Group{+input,+gradient,+values} = state
  +tap = (index / output_blocks * 8 : U32)
  +output = (index % output_blocks * 8 : U32)
  +window = {Kernel.WeightWindow{geometry,image,tap,output,row,column,first} : Kernel.WeightWindow}
  +result = window_product(count,window,input,gradient)
  %Equal.sym((Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>,Kernel.window_product(count,window,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),represent(result),window_product_matches(count,window,input,gradient)) :
    {Kernel.block_result(Kernel.BlockCopy{tap,output,taps,o},Tree.pack(F32,values),_) == represent(block(index,context,state)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}
  block_result_matches(Kernel.BlockCopy{tap,output,taps,o},values,result)

def block_sources(+index: U32,+context: Kernel.GroupBlocks,+state: Group) ->
  {group_sources(block(index,context,state)) == group_sources(state) : Sources}:
  Kernel.GroupBlocks{Kernel.WeightWindow{+geometry,+image,tap,output,+row,+column,+first},+count,+output_blocks} = context
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  Group{+input,+gradient,+values} = state
  +tap = (index / output_blocks * 8 : U32)
  +output = (index % output_blocks * 8 : U32)
  +copy = {Kernel.BlockCopy{tap,output,taps,o} : Kernel.BlockCopy}
  +window = {Kernel.WeightWindow{geometry,image,tap,output,row,column,first} : Kernel.WeightWindow}
  +result = window_product(count,window,input,gradient)
  Equal.trans(Sources,group_sources(block_result(copy,values,result)),group_sources(result),Sources{input,gradient},
    block_result_sources(copy,values,result),window_product_sources(count,window,input,gradient))

def block_storage(+index: U32,+context: Kernel.GroupBlocks,+state: Group,+depth: Nat,+capacity: U32,
  frame: StorageFrame(depth,capacity,group_output(state))) -> StorageFrame(depth,capacity,group_output(block(index,context,state))):
  Kernel.GroupBlocks{Kernel.WeightWindow{+geometry,+image,tap,output,+row,+column,+first},+count,+output_blocks} = context
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  Group{+input,+gradient,+values} = state
  +tap = (index / output_blocks * 8 : U32)
  +output = (index % output_blocks * 8 : U32)
  +copy = {Kernel.BlockCopy{tap,output,taps,o} : Kernel.BlockCopy}
  +window = {Kernel.WeightWindow{geometry,image,tap,output,row,column,first} : Kernel.WeightWindow}
  block_result_storage(copy,values,window_product(count,window,input,gradient),depth,capacity,frame)

type BlockShape is Data:
  BlockShape{blocks: Kernel.GroupBlocks,depth: Nat,capacity: U32,input: Buffer.Model<F32>,gradient: Buffer.Model<F32>,jobs: Nat}

def block_shape_context(context: BlockShape) -> Kernel.GroupBlocks:
  BlockShape{blocks,depth,capacity,input,gradient,jobs} = context
  blocks

def block_shape_limit(context: BlockShape) -> Nat:
  BlockShape{blocks,depth,capacity,input,gradient,jobs} = context
  jobs

def BlockFrame(context: BlockShape,state: Group) -> Data:
  BlockShape{blocks,depth,capacity,input,gradient,jobs} = context
  Shape.Both<StorageFrame(depth,capacity,group_output(state)),{group_sources(state) == Sources{input,gradient} : Sources}>

def BlockInvariant(context: BlockShape,index: Nat,state: Group) -> Data:
  BlockFrame(context,state)

def block_shape_update(index: U32,context: BlockShape,state: Group) -> Group:
  block(index,block_shape_context(context),state)

def block_shape_update_inner(+index: U32,+context: BlockShape,+state: Group) ->
  {block_shape_update(index,context,state) == block(index,block_shape_context(context),state) : Group}:
  {==}

def block_shape_frame_step(+context: BlockShape,+index: U32,+state: Group,frame: BlockFrame(context,state)) ->
  BlockFrame(context,block_shape_update(index,context,state)):
  BlockShape{+blocks,+depth,+capacity,+input,+gradient,jobs} = context
  Shape.Both{storage,sources} = frame
  Shape.Both{block_storage(index,blocks,state,depth,capacity,storage),
    Equal.trans(Sources,group_sources(block(index,blocks,state)),group_sources(state),Sources{input,gradient},
      block_sources(index,blocks,state),sources)}

def block_shape_step(+context: BlockShape,+index: Nat,+state: Group,
  inside: {Nat.is_lt(index,block_shape_limit(context)) == True{} : Bool},frame: BlockInvariant(context,index,state)) ->
  BlockInvariant(context,1n+index,block_shape_update(U32.from_nat(index),context,state)):
  block_shape_frame_step(context,U32.from_nat(index),state,frame)

def block_shape_fold(+context: BlockShape,+state: Group,frame: BlockFrame(context,state)) ->
  BlockFrame(context,Traversal.indexed_fold(~Group,~Kernel.GroupBlocks,~block,block_shape_limit(context),0,block_shape_context(context),state)):
  %Traversals.indexed_fold_change_context(~Group,~BlockShape,~Kernel.GroupBlocks,~block_shape_context,~block_shape_update,~block,~block_shape_update_inner,block_shape_limit(context),0,context,state) :
    BlockFrame(context,_)
  Traversals.indexed_fold_preserves_invariant(~Group,~BlockShape,~block_shape_limit,~BlockInvariant,~block_shape_update,~block_shape_step,block_shape_limit(context),0n,context,state,
    Order.reflexive(block_shape_limit(context)),frame)

def group(+geometry: Kernel.Geometry,+image: U32,range: Grouped.Range,input: Buffer.Model<F32>,gradient: Buffer.Model<F32>) -> Group:
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,+columns,+taps} = geometry
  Grouped.Range{+first,count} = range
  +output_blocks = ((o + 7) / 8 : U32)
  +context = {Kernel.GroupBlocks{Kernel.WeightWindow{geometry,image,0,0,U32.from_nat(Nat.div(first,U32.to_nat(columns))),U32.from_nat(Nat.mod(first,U32.to_nat(columns))),U32.from_nat(first)},U32.from_nat(count),output_blocks} : Kernel.GroupBlocks}
  Traversal.indexed_fold(~Group,~Kernel.GroupBlocks,~block,U32.to_nat((((taps + 7) / 8) * output_blocks : U32)),0,context,
    Group{input,gradient,Tree.fresh(F32,Storage.depth_for((((taps * o + 7) / 8) * 8 : U32)),0.0)})

def group_matches(+geometry: Kernel.Geometry,+image: U32,+range: Grouped.Range,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {Kernel.group(geometry,image,range,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)) == represent(group(geometry,image,range,input,gradient)) :
    (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}:
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,+columns,+taps} = geometry
  Grouped.Range{+first,count} = range
  +output_blocks = ((o + 7) / 8 : U32)
  +context = {Kernel.GroupBlocks{Kernel.WeightWindow{geometry,image,0,0,U32.from_nat(Nat.div(first,U32.to_nat(columns))),U32.from_nat(Nat.mod(first,U32.to_nat(columns))),U32.from_nat(first)},U32.from_nat(count),output_blocks} : Kernel.GroupBlocks}
  +depth = Storage.depth_for((((taps * o + 7) / 8) * 8 : U32))
  +initial = {Group{input,gradient,Tree.fresh(F32,depth,0.0)} : Group}
  %Equal.sym(Array<F32>,Array.new(F32,depth,0.0),Tree.pack(F32,Tree.fresh(F32,depth,0.0)),StorageLaws.fresh(F32,depth,0.0)) :
    {Traversal.indexed_fold(~((Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>),~Kernel.GroupBlocks,~Kernel.block,U32.to_nat((((taps + 7) / 8) * output_blocks : U32)),0,context,((Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),_)) == represent(group(geometry,image,range,input,gradient)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}
  Traversals.indexed_fold_refines_model(~((Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>),~Group,~Kernel.GroupBlocks,~Kernel.block,~block,~represent,~block_matches,
    U32.to_nat((((taps + 7) / 8) * output_blocks : U32)),0,context,initial)

# The sources the blocks of a group see.
def block_sources_seen(index: U32,context: Kernel.GroupBlocks,sources: Sources) -> Sources:
  sources

def block_sources_fixed(+index: U32,+context: Kernel.GroupBlocks,+sources: Sources) -> {block_sources_seen(index,context,sources) == sources : Sources}:
  {==}

# A group gives back the input and the gradient it read.
def group_sources_kept(+geometry: Kernel.Geometry,+image: U32,+range: Grouped.Range,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {group_sources(group(geometry,image,range,input,gradient)) == Sources{input,gradient} : Sources}:
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,+columns,+taps} = geometry
  Grouped.Range{+first,+count} = range
  +output_blocks = ((o + 7) / 8 : U32)
  +context = {Kernel.GroupBlocks{Kernel.WeightWindow{Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},image,0,0,U32.from_nat(Nat.div(first,U32.to_nat(columns))),U32.from_nat(Nat.mod(first,U32.to_nat(columns))),U32.from_nat(first)},
    U32.from_nat(count),output_blocks} : Kernel.GroupBlocks}
  +blocks = U32.to_nat((((taps + 7) / 8) * output_blocks : U32))
  +initial = {Group{input,gradient,Tree.fresh(F32,Storage.depth_for((((taps * o + 7) / 8) * 8 : U32)),0.0)} : Group}
  Equal.trans(Sources,group_sources(Traversal.indexed_fold(~Group,~Kernel.GroupBlocks,~block,blocks,0,context,initial)),
    Traversal.indexed_fold(~Sources,~Kernel.GroupBlocks,~block_sources_seen,blocks,0,context,Sources{input,gradient}),Sources{input,gradient},
    Traversals.indexed_fold_observes(~Group,~Sources,~Kernel.GroupBlocks,~block,~group_sources,~block_sources_seen,~block_sources,blocks,0,context,initial),
    Traversals.indexed_fold_fixed(~Sources,~Kernel.GroupBlocks,~block_sources_seen,~block_sources_fixed,blocks,0,context,Sources{input,gradient}))

def micro_job(upper: Bool,+count: U32) ->
  {Gemm.job(Gemm.Layout{8,8,count,0,8},Bool.pick(U32,upper,1,0)) == GemmWrites.half_job(upper,U32.to_nat(count)) : Gemm.Job}:
  match upper:
    case False{}:
      %Equal.sym(U32,(0 * count : U32),0,Arithmetic.multiply_zero_left(count)) :
        {Gemm.Job{Gemm.Context{0,0,8,8,_,_},False{},U32.to_nat(count)} == GemmWrites.half_job(False{},U32.to_nat(count)) : Gemm.Job}
      {==}
    case True{}:
      %Equal.sym(U32,(0 * count : U32),0,Arithmetic.multiply_zero_left(count)) :
        {Gemm.Job{Gemm.Context{4,0,8,8,_,_},True{},U32.to_nat(count)} == GemmWrites.half_job(True{},U32.to_nat(count)) : Gemm.Job}
      {==}

def unseeded_half(+upper: Bool,+count: Nat,+buffers: Gemms.Buffers) ->
  {Gemms.execute(~False{},GemmWrites.half_job(upper,count),buffers) == Gemms.execute_with(~False{},GemmWrites.half_job(upper,count),buffers,Rows.filled(~Tile.Tile8,Tile.filled(0.0))) : Gemms.Buffers}:
  Gemms.Buffers{input,weights,bias,output} = buffers
  {==}

def seeded_half(upper: Bool,+count: Nat,+buffers: Gemms.Buffers) ->
  {Gemms.execute_seeded(GemmWrites.half_job(upper,count),buffers) == Gemms.execute_with(~False{},GemmWrites.half_job(upper,count),buffers,Gemms.seed(Gemms.result(buffers),Bool.pick(U32,upper,32,0),8)) : Gemms.Buffers}:
  match upper buffers:
    case False{} Gemms.Buffers{input,weights,bias,output}: {==}
    case True{} Gemms.Buffers{input,weights,bias,output}: {==}

def initial_rows(continued: Bool,+upper: Bool,output: Tree.Tree<F32>) -> Rows.Rows<Tile.Tile8>:
  match continued:
    case False{}: Rows.filled(~Tile.Tile8,Tile.filled(0.0))
    case True{}: Gemms.seed(output,Bool.pick(U32,upper,32,0),8)

def window_first(+continued: Bool,+count: U32,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>) -> Gemms.Buffers:
  GemmWrites.half_execute(False{},U32.to_nat(count),input,weights,bias,Writes.Initial{output},initial_rows(continued,False{},output))

def window_execute(+continued: Bool,+count: U32,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Tree.Tree<F32>) -> Gemms.Buffers:
  +first = window_first(continued,count,input,weights,bias,output)
  Gemms.execute_with(~False{},GemmWrites.half_job(True{},U32.to_nat(count)),first,initial_rows(continued,True{},Gemms.result(first)))

def window_two_jobs(continued: Bool,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<F32>) ->
  {window_gemm(continued,count,input,weights,output) == Gemms.result(window_execute(continued,count,input,weights,Tree.fresh(F32,Storage.depth_for(1),0.0),output)) : Tree.Tree<F32>}:
  match continued:
    case False{}:
      +bias = Tree.fresh(F32,Storage.depth_for(1),0.0)
      +initial = {Gemms.Buffers{input,weights,bias,Writes.Initial{output}} : Gemms.Buffers}
      +first = window_first(False{},count,input,weights,bias,output)
      %Equal.sym(Gemm.Job,Gemm.job(Gemm.Layout{8,8,count,0,8},0),GemmWrites.half_job(False{},U32.to_nat(count)),micro_job(False{},count)) :
        {Gemms.result(Gemms.execute(~False{},Gemm.job(Gemm.Layout{8,8,count,0,8},1),Gemms.execute(~False{},_,initial))) == Gemms.result(window_execute(False{},count,input,weights,bias,output)) : Tree.Tree<F32>}
      %Equal.sym(Gemm.Job,Gemm.job(Gemm.Layout{8,8,count,0,8},1),GemmWrites.half_job(True{},U32.to_nat(count)),micro_job(True{},count)) :
        {Gemms.result(Gemms.execute(~False{},_,Gemms.execute(~False{},GemmWrites.half_job(False{},U32.to_nat(count)),initial))) == Gemms.result(window_execute(False{},count,input,weights,bias,output)) : Tree.Tree<F32>}
      %Equal.sym(Gemms.Buffers,Gemms.execute(~False{},GemmWrites.half_job(False{},U32.to_nat(count)),initial),first,unseeded_half(False{},U32.to_nat(count),initial)) :
        {Gemms.result(Gemms.execute(~False{},GemmWrites.half_job(True{},U32.to_nat(count)),_)) == Gemms.result(window_execute(False{},count,input,weights,bias,output)) : Tree.Tree<F32>}
      Equal.cong(Gemms.Buffers,Tree.Tree<F32>,Gemms.result,Gemms.execute(~False{},GemmWrites.half_job(True{},U32.to_nat(count)),first),window_execute(False{},count,input,weights,bias,output),unseeded_half(True{},U32.to_nat(count),first))
    case True{}:
      +bias = Tree.fresh(F32,Storage.depth_for(1),0.0)
      +initial = {Gemms.Buffers{input,weights,bias,Writes.Initial{output}} : Gemms.Buffers}
      +first = window_first(True{},count,input,weights,bias,output)
      %Equal.sym(Gemm.Job,Gemm.job(Gemm.Layout{8,8,count,0,8},0),GemmWrites.half_job(False{},U32.to_nat(count)),micro_job(False{},count)) :
        {Gemms.result(Gemms.execute_seeded(Gemm.job(Gemm.Layout{8,8,count,0,8},1),Gemms.execute_seeded(_,initial))) == Gemms.result(window_execute(True{},count,input,weights,bias,output)) : Tree.Tree<F32>}
      %Equal.sym(Gemm.Job,Gemm.job(Gemm.Layout{8,8,count,0,8},1),GemmWrites.half_job(True{},U32.to_nat(count)),micro_job(True{},count)) :
        {Gemms.result(Gemms.execute_seeded(_,Gemms.execute_seeded(GemmWrites.half_job(False{},U32.to_nat(count)),initial))) == Gemms.result(window_execute(True{},count,input,weights,bias,output)) : Tree.Tree<F32>}
      %Equal.sym(Gemms.Buffers,Gemms.execute_seeded(GemmWrites.half_job(False{},U32.to_nat(count)),initial),first,seeded_half(False{},U32.to_nat(count),initial)) :
        {Gemms.result(Gemms.execute_seeded(GemmWrites.half_job(True{},U32.to_nat(count)),_)) == Gemms.result(window_execute(True{},count,input,weights,bias,output)) : Tree.Tree<F32>}
      Equal.cong(Gemms.Buffers,Tree.Tree<F32>,Gemms.result,Gemms.execute_seeded(GemmWrites.half_job(True{},U32.to_nat(count)),first),window_execute(True{},count,input,weights,bias,output),seeded_half(True{},U32.to_nat(count),first))

def window_first_owned(+continued: Bool,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>) ->
  {window_first(continued,count,input,weights,bias,output) == Gemms.Buffers{input,weights,bias,GemmWrites.buffers_program(window_first(continued,count,input,weights,bias,output))} : Gemms.Buffers}:
  GemmWrites.executed_with(~False{},GemmWrites.half_job(False{},U32.to_nat(count)),input,weights,bias,Writes.Initial{output},initial_rows(continued,False{},output))

def window_second(+continued: Bool,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>) -> Gemms.Buffers:
  +first = window_first(continued,count,input,weights,bias,output)
  GemmWrites.half_execute(True{},U32.to_nat(count),input,weights,bias,GemmWrites.buffers_program(first),initial_rows(continued,True{},Gemms.result(first)))

def window_execute_owned(+continued: Bool,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>) ->
  {window_execute(continued,count,input,weights,bias,output) == window_second(continued,count,input,weights,bias,output) : Gemms.Buffers}:
  +first = window_first(continued,count,input,weights,bias,output)
  +initial = initial_rows(continued,True{},Gemms.result(first))
  Equal.cong(Gemms.Buffers,Gemms.Buffers,buffers => Gemms.execute_with(~False{},GemmWrites.half_job(True{},U32.to_nat(count)),buffers,initial),first,Gemms.Buffers{input,weights,bias,GemmWrites.buffers_program(first)},window_first_owned(continued,count,input,weights,bias,output))

def window_upper_seed(continued: Bool,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+balanced: Shape.Balanced(F32,6n,output)) ->
  {initial_rows(continued,True{},Gemms.result(window_first(continued,count,input,weights,bias,output))) == initial_rows(continued,True{},output) : Rows.Rows<Tile.Tile8>}:
  match continued:
    case False{}: {==}
    case True{}:
      +values = GemmWrites.half_accumulators(False{},U32.to_nat(count),input,weights,initial_rows(True{},False{},output))
      +first = window_first(True{},count,input,weights,bias,output)
      Equal.trans(Rows.Rows<Tile.Tile8>,Gemms.seed(Gemms.result(first),32,8),Gemms.seed(Writes.pure(F32,GemmWrites.half_writes(False{},values,bias),output),32,8),Gemms.seed(output,32,8),
        Equal.cong(Tree.Tree<F32>,Rows.Rows<Tile.Tile8>,tree => Gemms.seed(tree,32,8),Gemms.result(first),Writes.pure(F32,GemmWrites.half_writes(False{},values,bias),output),GemmWrites.half_execute_output(False{},U32.to_nat(count),input,weights,bias,Writes.Initial{output},initial_rows(True{},False{},output))),GemmWrites.upper_seed_preserved(values,bias,output,balanced))

def window_clean(+continued: Bool,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>) -> Gemms.Buffers:
  +first = window_first(continued,count,input,weights,bias,output)
  GemmWrites.half_execute(True{},U32.to_nat(count),input,weights,bias,GemmWrites.buffers_program(first),initial_rows(continued,True{},output))

def window_clean_matches(+continued: Bool,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+balanced: Shape.Balanced(F32,6n,output)) ->
  {window_execute(continued,count,input,weights,bias,output) == window_clean(continued,count,input,weights,bias,output) : Gemms.Buffers}:
  +first = window_first(continued,count,input,weights,bias,output)
  +old = initial_rows(continued,True{},Gemms.result(first))
  +initial = initial_rows(continued,True{},output)
  Equal.trans(Gemms.Buffers,window_execute(continued,count,input,weights,bias,output),window_second(continued,count,input,weights,bias,output),window_clean(continued,count,input,weights,bias,output),window_execute_owned(continued,count,input,weights,bias,output),
    Equal.cong(Rows.Rows<Tile.Tile8>,Gemms.Buffers,rows => GemmWrites.half_execute(True{},U32.to_nat(count),input,weights,bias,GemmWrites.buffers_program(first),rows),old,initial,window_upper_seed(continued,count,input,weights,bias,output,balanced)))

def window_first_balance(+continued: Bool,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,balanced: Shape.Balanced(F32,6n,output)) ->
  Shape.Balanced(F32,6n,Gemms.result(window_first(continued,count,input,weights,bias,output))):
  GemmWrites.half_execute_balance(False{},U32.to_nat(count),input,weights,bias,Writes.Initial{output},initial_rows(continued,False{},output),balanced)

def window_program_balance(+continued: Bool,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,balanced: Shape.Balanced(F32,6n,output)) ->
  Shape.Balanced(F32,6n,Writes.evaluate(F32,GemmWrites.buffers_program(window_first(continued,count,input,weights,bias,output)))):
  +first = window_first(continued,count,input,weights,bias,output)
  %GemmWrites.result_program(first) : Shape.Balanced(F32,6n,_)
  window_first_balance(continued,count,input,weights,bias,output,balanced)

def window_upper_readback(+continued: Bool,+row: Accumulator.Row,+lane: Nat,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<F32>,+balanced: Shape.Balanced(F32,6n,output),+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Tree.read(F32,window_gemm(continued,count,input,weights,output),GemmWrites.half_address(True{},row,lane)) ==
    TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,GemmWrites.half_accumulators(True{},U32.to_nat(count),input,weights,initial_rows(continued,True{},output)))) : F32}:
  +bias = Tree.fresh(F32,Storage.depth_for(1),0.0)
  +first = window_first(continued,count,input,weights,bias,output)
  +wanted = TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,GemmWrites.half_accumulators(True{},U32.to_nat(count),input,weights,initial_rows(continued,True{},output))))
  +query = GemmWrites.half_address(True{},row,lane)
  %Equal.sym(Tree.Tree<F32>,window_gemm(continued,count,input,weights,output),Gemms.result(window_execute(continued,count,input,weights,bias,output)),window_two_jobs(continued,count,input,weights,output)) : {Tree.read(F32,_,query) == wanted : F32}
  %Equal.sym(Gemms.Buffers,window_execute(continued,count,input,weights,bias,output),window_clean(continued,count,input,weights,bias,output),window_clean_matches(continued,count,input,weights,bias,output,balanced)) : {Tree.read(F32,Gemms.result(_),query) == wanted : F32}
  GemmWrites.half_execute_readback(True{},row,lane,U32.to_nat(count),input,weights,bias,GemmWrites.buffers_program(first),initial_rows(continued,True{},output),window_program_balance(continued,count,input,weights,bias,output,balanced),within)

def upper_keeps_lower(+values: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>,+tree: Tree.Tree<F32>,+row: Accumulator.Row,+lane: Nat,+balanced: Shape.Balanced(F32,6n,tree),+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Tree.read(F32,Writes.pure(F32,GemmWrites.half_writes(True{},values,bias),tree),GemmWrites.half_address(False{},row,lane)) == Tree.read(F32,tree,GemmWrites.half_address(False{},row,lane)) : F32}:
  +query = GemmWrites.half_address(False{},row,lane)
  %Arithmetic.native_roundtrip(query) : {Tree.read(F32,Writes.pure(F32,GemmWrites.half_writes(True{},values,bias),tree),_) == Tree.read(F32,tree,_) : F32}
  GemmWrites.half_preserved(True{},values,bias,tree,U32.to_nat(query),balanced,Bounds.as_nat_lt(query,64,GemmWrites.half_address_inside(False{},row,lane,within)),Interval.Before{Bounds.as_nat_lt(query,32,GemmWrites.lower_address_inside(row,lane,within))})

def window_lower_readback(+continued: Bool,+row: Accumulator.Row,+lane: Nat,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<F32>,+balanced: Shape.Balanced(F32,6n,output),+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Tree.read(F32,window_gemm(continued,count,input,weights,output),GemmWrites.half_address(False{},row,lane)) ==
    TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,GemmWrites.half_accumulators(False{},U32.to_nat(count),input,weights,initial_rows(continued,False{},output)))) : F32}:
  +bias = Tree.fresh(F32,Storage.depth_for(1),0.0)
  +first = window_first(continued,count,input,weights,bias,output)
  +program = GemmWrites.buffers_program(first)
  +tree = Writes.evaluate(F32,program)
  +balanced_first = window_program_balance(continued,count,input,weights,bias,output,balanced)
  +query = GemmWrites.half_address(False{},row,lane)
  +wanted = TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,GemmWrites.half_accumulators(False{},U32.to_nat(count),input,weights,initial_rows(continued,False{},output))))
  +values = GemmWrites.half_accumulators(True{},U32.to_nat(count),input,weights,initial_rows(continued,True{},output))
  %Equal.sym(Tree.Tree<F32>,window_gemm(continued,count,input,weights,output),Gemms.result(window_execute(continued,count,input,weights,bias,output)),window_two_jobs(continued,count,input,weights,output)) : {Tree.read(F32,_,query) == wanted : F32}
  %Equal.sym(Gemms.Buffers,window_execute(continued,count,input,weights,bias,output),window_clean(continued,count,input,weights,bias,output),window_clean_matches(continued,count,input,weights,bias,output,balanced)) : {Tree.read(F32,Gemms.result(_),query) == wanted : F32}
  %Equal.sym(Tree.Tree<F32>,Gemms.result(window_clean(continued,count,input,weights,bias,output)),Writes.pure(F32,GemmWrites.half_writes(True{},values,bias),tree),GemmWrites.half_execute_output(True{},U32.to_nat(count),input,weights,bias,program,initial_rows(continued,True{},output))) : {Tree.read(F32,_,query) == wanted : F32}
  Equal.trans(F32,Tree.read(F32,Writes.pure(F32,GemmWrites.half_writes(True{},values,bias),tree),query),Tree.read(F32,tree,query),wanted,upper_keeps_lower(values,bias,tree,row,lane,balanced_first,within),
    %GemmWrites.result_program(first) : {Tree.read(F32,_,query) == wanted : F32}
    GemmWrites.half_execute_readback(False{},row,lane,U32.to_nat(count),input,weights,bias,Writes.Initial{output},initial_rows(continued,False{},output),balanced,within))

def half_dot(upper: Bool,+row: Accumulator.Row,lane: Nat,count: Nat,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,initial: Rows.Rows<Tile.Tile8>) -> F32:
  match upper:
    case False{}: ReductionModel.ordered_dot(~False{},ReductionModel.RowLane{row,TileLaws.lane_of(lane)},count,Reduction.State{input,weights,0,0,0,initial})
    case True{}: ReductionModel.ordered_dot(~True{},ReductionModel.RowLane{row,TileLaws.lane_of(lane)},count,Reduction.State{input,weights,0,0,0,initial})

def half_dot_matches(upper: Bool,+row: Accumulator.Row,+lane: Nat,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+initial: Rows.Rows<Tile.Tile8>) ->
  {TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,GemmWrites.half_accumulators(upper,count,input,weights,initial))) == half_dot(upper,row,lane,count,input,weights,initial) : F32}:
  match upper:
    case False{}:
      +state = {Reduction.State{input,weights,0,0,0,initial} : ReductionModel.State()}
      +values = GemmWrites.half_accumulators(False{},count,input,weights,initial)
      +key = {ReductionModel.RowLane{row,TileLaws.lane_of(lane)} : ReductionModel.RowLane}
      Equal.trans(F32,TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)),ReductionModel.accumulator_at(key,values),half_dot(False{},row,lane,count,input,weights,initial),TileLaws.selected_lane(lane,Accumulator.row(~Tile.Tile8,row,values)),ReductionModel.run_matches_ordered_dot(~False{},key,count,state))
    case True{}:
      +state = {Reduction.State{input,weights,0,0,0,initial} : ReductionModel.State()}
      +values = GemmWrites.half_accumulators(True{},count,input,weights,initial)
      +key = {ReductionModel.RowLane{row,TileLaws.lane_of(lane)} : ReductionModel.RowLane}
      Equal.trans(F32,TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)),ReductionModel.accumulator_at(key,values),half_dot(True{},row,lane,count,input,weights,initial),TileLaws.selected_lane(lane,Accumulator.row(~Tile.Tile8,row,values)),ReductionModel.run_matches_ordered_dot(~True{},key,count,state))

def window_readback(+continued: Bool,upper: Bool,+row: Accumulator.Row,+lane: Nat,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<F32>,+balanced: Shape.Balanced(F32,6n,output),+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Tree.read(F32,window_gemm(continued,count,input,weights,output),GemmWrites.half_address(upper,row,lane)) == half_dot(upper,row,lane,U32.to_nat(count),input,weights,initial_rows(continued,upper,output)) : F32}:
  match upper:
    case False{}:
      +values = GemmWrites.half_accumulators(False{},U32.to_nat(count),input,weights,initial_rows(continued,False{},output))
      Equal.trans(F32,Tree.read(F32,window_gemm(continued,count,input,weights,output),GemmWrites.half_address(False{},row,lane)),TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)),half_dot(False{},row,lane,U32.to_nat(count),input,weights,initial_rows(continued,False{},output)),window_lower_readback(continued,row,lane,count,input,weights,output,balanced,within),half_dot_matches(False{},row,lane,U32.to_nat(count),input,weights,initial_rows(continued,False{},output)))
    case True{}:
      +values = GemmWrites.half_accumulators(True{},U32.to_nat(count),input,weights,initial_rows(continued,True{},output))
      Equal.trans(F32,Tree.read(F32,window_gemm(continued,count,input,weights,output),GemmWrites.half_address(True{},row,lane)),TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)),half_dot(True{},row,lane,U32.to_nat(count),input,weights,initial_rows(continued,True{},output)),window_upper_readback(continued,row,lane,count,input,weights,output,balanced,within),half_dot_matches(True{},row,lane,U32.to_nat(count),input,weights,initial_rows(continued,True{},output)))

# Reuse the forward matrix's consecutive-address bridge for position keys.
def local_input() -> Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>> -> ReductionModel.RowLane -> Nat -> F32:
  ReductionModel.natural_input()

def local_weight(~upper: Bool) -> Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>> -> ReductionModel.RowLane -> Nat -> F32:
  ReductionModel.natural_weight(~upper)

def local_dot(~upper: Bool,+key: ReductionModel.RowLane,count: Nat,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,initial: Rows.Rows<Tile.Tile8>) -> F32:
  OrderedMatrix.dot(~ReductionModel.RowLane,~Nat,~ReductionModel.RowLane,~Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>,
    ~local_input(),~local_weight(~upper),Ranges.range(count,0n),key,key,Shape.Both{ReductionLaws.AddressLayout{0n,0n},ReductionLaws.SampleSource{input,weights,0}},ReductionModel.accumulator_at(key,initial))

def ordered_dot_reindexed(~upper: Bool,+key: ReductionModel.RowLane,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+initial: Rows.Rows<Tile.Tile8>) ->
  {ReductionModel.ordered_dot(~upper,key,count,Reduction.State{input,weights,0,0,0,initial}) == local_dot(~upper,key,count,input,weights,initial) : F32}:
  ReductionModel.ordered_as_natural(~upper,key,count,Reduction.State{input,weights,0,0,0,initial})

def natural_dot(upper: Bool,+row: Accumulator.Row,lane: Nat,count: Nat,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,initial: Rows.Rows<Tile.Tile8>) -> F32:
  match upper:
    case False{}: local_dot(~False{},ReductionModel.RowLane{row,TileLaws.lane_of(lane)},count,input,weights,initial)
    case True{}: local_dot(~True{},ReductionModel.RowLane{row,TileLaws.lane_of(lane)},count,input,weights,initial)

def half_dot_as_natural(upper: Bool,+row: Accumulator.Row,+lane: Nat,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+initial: Rows.Rows<Tile.Tile8>) ->
  {half_dot(upper,row,lane,count,input,weights,initial) == natural_dot(upper,row,lane,count,input,weights,initial) : F32}:
  match upper:
    case False{}: ordered_dot_reindexed(~False{},ReductionModel.RowLane{row,TileLaws.lane_of(lane)},count,input,weights,initial)
    case True{}: ordered_dot_reindexed(~True{},ReductionModel.RowLane{row,TileLaws.lane_of(lane)},count,input,weights,initial)

def window_balance(+continued: Bool,+count: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<F32>,+balanced: Shape.Balanced(F32,6n,output)) -> Shape.Balanced(F32,6n,window_gemm(continued,count,input,weights,output)):
  +bias = Tree.fresh(F32,Storage.depth_for(1),0.0)
  +first = window_first(continued,count,input,weights,bias,output)
  %Equal.sym(Tree.Tree<F32>,window_gemm(continued,count,input,weights,output),Gemms.result(window_execute(continued,count,input,weights,bias,output)),window_two_jobs(continued,count,input,weights,output)) : Shape.Balanced(F32,6n,_)
  %Equal.sym(Gemms.Buffers,window_execute(continued,count,input,weights,bias,output),window_clean(continued,count,input,weights,bias,output),window_clean_matches(continued,count,input,weights,bias,output,balanced)) : Shape.Balanced(F32,6n,Gemms.result(_))
  GemmWrites.half_execute_balance(True{},U32.to_nat(count),input,weights,bias,GemmWrites.buffers_program(first),initial_rows(continued,True{},output),window_program_balance(continued,count,input,weights,bias,output,balanced))

# Both samples of each visited MAC come from the actual packed buffers.
def packed_input_at(+window: Kernel.WeightWindow,+count: U32,+gradient: Buffer.Model<F32>,+weights: Tree.Tree<Tile.Tile8>,+row: Accumulator.Row,+lane: Nat,+index: Nat,
  bounded: {U32.is_le(count,256) == True{} : Bool},+within: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {local_input()(Shape.Both{ReductionLaws.AddressLayout{0n,0n},ReductionLaws.SampleSource{Packing.packed_tree(Packing.window_gradient(window,count,gradient)),weights,0}},ReductionModel.RowLane{row,TileLaws.lane_of(lane)},index) == Packing.gradient_value(gradient,Packing.gradient_context(window),lane,U32.from_nat(index)) : F32}:
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,Packing.packed_tree(Packing.window_gradient(window,count,gradient)),U32.from_nat(index)),Packing.gradient_tile_value(gradient,Packing.gradient_context(window),U32.from_nat(index)),Packing.window_gradient_readback(window,count,gradient,index,bounded,within)) :
    {Tile.element_at(TileLaws.lane_of(lane),_) == Packing.gradient_value(gradient,Packing.gradient_context(window),lane,U32.from_nat(index)) : F32}
  Packing.gradient_tile_element(gradient,Packing.gradient_context(window),U32.from_nat(index),lane,lane_inside)

def packed_weight_at(~upper: Bool,+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>,+gradient: Tree.Tree<Tile.Tile8>,+row: Accumulator.Row,+lane: Nat,+position: Nat,+index: Nat,
  +positive: {Nat.is_lt(0n,U32.to_nat(Packing.window_columns(window))) == True{} : Bool},
  +row_match: {Packing.window_row(window) == U32.from_nat(Nat.div(position,U32.to_nat(Packing.window_columns(window)))) : U32},
  +column_match: {Packing.window_column(window) == U32.from_nat(Nat.mod(position,U32.to_nat(Packing.window_columns(window)))) : U32},
  bounded: {U32.is_le(count,256) == True{} : Bool},+within: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {local_weight(~upper)(Shape.Both{ReductionLaws.AddressLayout{0n,0n},ReductionLaws.SampleSource{gradient,Packing.packing_output(Packing.window_input(window,count,input)),0}},ReductionModel.RowLane{row,TileLaws.lane_of(lane)},index) == Packing.natural_patch_value(Packing.window_patch_context(window,input),upper,row,Nat.add(position,index)) : F32}:
  %Equal.sym(U32,(0 + U32.from_nat(index) : U32),U32.from_nat(index),Arithmetic.add_zero_left(U32.from_nat(index))) :
    {Accumulator.row(~F32,row,Rows.half(upper,Tree.read(Tile.Tile8,Packing.packing_output(Packing.window_input(window,count,input)),_))) == Packing.natural_patch_value(Packing.window_patch_context(window,input),upper,row,Nat.add(position,index)) : F32}
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,Packing.packing_output(Packing.window_input(window,count,input)),U32.from_nat(index)),Packing.natural_sample(Packing.window_patch_context(window,input),Nat.add(position,index)),Packing.window_input_readback(window,count,input,position,index,positive,row_match,column_match,bounded,within)) :
    {Accumulator.row(~F32,row,Rows.half(upper,_)) == Packing.natural_patch_value(Packing.window_patch_context(window,input),upper,row,Nat.add(position,index)) : F32}
  Packing.natural_patch_element(Packing.window_patch_context(window,input),upper,row,Nat.add(position,index))

type PackedProblem is Data:
  PackedProblem{window: Kernel.WeightWindow,count: U32,input: Buffer.Model<F32>,gradient: Buffer.Model<F32>,first: Nat,
    positive: {Nat.is_lt(0n,U32.to_nat(Packing.window_columns(window))) == True{} : Bool},
    row_match: {Packing.window_row(window) == U32.from_nat(Nat.div(first,U32.to_nat(Packing.window_columns(window)))) : U32},
    column_match: {Packing.window_column(window) == U32.from_nat(Nat.mod(first,U32.to_nat(Packing.window_columns(window)))) : U32},
    bounded: {U32.is_le(count,256) == True{} : Bool}}

type PackedKey is Data:
  PackedKey{row: Accumulator.Row,lane: Nat}

def key_lane(key: PackedKey) -> Nat:
  PackedKey{row,lane} = key
  lane

def left_key(key: PackedKey) -> ReductionModel.RowLane:
  PackedKey{row,lane} = key
  ReductionModel.RowLane{row,TileLaws.lane_of(lane)}

def packed_source(problem: PackedProblem) -> Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>:
  PackedProblem{+window,+count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  Shape.Both{ReductionLaws.AddressLayout{0n,0n},ReductionLaws.SampleSource{Packing.packed_tree(Packing.window_gradient(window,count,gradient)),Packing.packing_output(Packing.window_input(window,count,input)),0}}

def packed_count(problem: PackedProblem) -> Nat:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  U32.to_nat(count)

def target_input(problem: PackedProblem,key: PackedKey,index: Nat) -> F32:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  PackedKey{row,lane} = key
  Packing.gradient_value(gradient,Packing.gradient_context(window),lane,U32.from_nat(index))

def target_weight(~upper: Bool,problem: PackedProblem,key: PackedKey,index: Nat) -> F32:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  PackedKey{row,lane} = key
  Packing.natural_patch_value(Packing.window_patch_context(window,input),upper,row,Nat.add(first,index))

type DotAgreement is Data:
  DotAgreement{problem: PackedProblem,key: PackedKey,lane_inside: {Nat.is_lt(key_lane(key),8n) == True{} : Bool}}

def agreement_size(context: DotAgreement) -> Nat:
  DotAgreement{problem,key,lane_inside} = context
  packed_count(problem)

def Agreement(~upper: Bool,+context: DotAgreement,+index: Nat) -> Data:
  DotAgreement{+problem,+key,lane_inside} = context
  Shape.Both<
    {local_input()(packed_source(problem),left_key(key),index) == target_input(problem,key,index) : F32},
    {local_weight(~upper)(packed_source(problem),left_key(key),index) == target_weight(~upper,problem,key,index) : F32}>

def agreement_cell(~upper: Bool,+context: DotAgreement,+index: Nat,+within: {Nat.is_lt(index,agreement_size(context)) == True{} : Bool}) -> Agreement(~upper,context,index):
  DotAgreement{+problem,+key,+lane_inside} = context
  PackedProblem{+window,+count,+input,+gradient,+first,+positive,+row_match,+column_match,+bounded} = problem
  PackedKey{+row,+lane} = key
  Shape.Both{
    packed_input_at(window,count,gradient,Packing.packing_output(Packing.window_input(window,count,input)),row,lane,index,bounded,within,lane_inside),
    packed_weight_at(~upper,window,count,input,Packing.packed_tree(Packing.window_gradient(window,count,gradient)),row,lane,first,index,positive,row_match,column_match,bounded,within)}

def packed_dot(~upper: Bool,+problem: PackedProblem,+key: PackedKey,initial: F32) -> F32:
  OrderedMatrix.dot(~PackedKey,~Nat,~PackedKey,~PackedProblem,~target_input,~target_weight(~upper),Ranges.range(packed_count(problem),0n),key,key,problem,initial)

def packed_dot_samples(~upper: Bool,+problem: PackedProblem,+key: PackedKey,+initial: F32,+lane_inside: {Nat.is_lt(key_lane(key),8n) == True{} : Bool}) ->
  {OrderedMatrix.dot(~ReductionModel.RowLane,~Nat,~ReductionModel.RowLane,~Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>,~local_input(),~local_weight(~upper),Ranges.range(packed_count(problem),0n),left_key(key),left_key(key),packed_source(problem),initial) == packed_dot(~upper,problem,key,initial) : F32}:
  +context = {DotAgreement{problem,key,lane_inside} : DotAgreement}
  Ordered.dot_congruence(~Nat,~ReductionModel.RowLane,~ReductionModel.RowLane,~Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>,~PackedKey,~PackedKey,~PackedProblem,
    ~local_input(),~local_weight(~upper),~target_input,~target_weight(~upper),Ranges.range(packed_count(problem),0n),left_key(key),left_key(key),packed_source(problem),key,key,problem,initial,
    Members.range_all(~DotAgreement,~Agreement(~upper),~agreement_size,~agreement_cell(~upper),packed_count(problem),0n,context,Order.reflexive(packed_count(problem))))

def key_row(key: PackedKey) -> Accumulator.Row:
  PackedKey{row,lane} = key
  row

def problem_window(problem: PackedProblem) -> Kernel.WeightWindow:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  window

def problem_count(problem: PackedProblem) -> U32:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  count

def problem_input(problem: PackedProblem) -> Buffer.Model<F32>:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  input

def problem_gradient(problem: PackedProblem) -> Buffer.Model<F32>:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  gradient

def packed_dot_value(upper: Bool,+problem: PackedProblem,+key: PackedKey,initial: F32) -> F32:
  match upper:
    case False{}: packed_dot(~False{},problem,key,initial)
    case True{}: packed_dot(~True{},problem,key,initial)

def local_dot_packed(~upper: Bool,+problem: PackedProblem,+key: PackedKey,+initial: Rows.Rows<Tile.Tile8>,+lane_inside: {Nat.is_lt(key_lane(key),8n) == True{} : Bool}) ->
  {local_dot(~upper,ReductionModel.RowLane{key_row(key),TileLaws.lane_of(key_lane(key))},packed_count(problem),Packing.packed_tree(Packing.window_gradient(problem_window(problem),problem_count(problem),problem_gradient(problem))),Packing.packing_output(Packing.window_input(problem_window(problem),problem_count(problem),problem_input(problem))),initial) == packed_dot(~upper,problem,key,ReductionModel.accumulator_at(left_key(key),initial)) : F32}:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  PackedKey{row,lane} = key
  packed_dot_samples(~upper,problem,key,ReductionModel.accumulator_at(left_key(key),initial),lane_inside)

def natural_dot_packed(upper: Bool,+problem: PackedProblem,+key: PackedKey,+initial: Rows.Rows<Tile.Tile8>,+lane_inside: {Nat.is_lt(key_lane(key),8n) == True{} : Bool}) ->
  {natural_dot(upper,key_row(key),key_lane(key),packed_count(problem),Packing.packed_tree(Packing.window_gradient(problem_window(problem),problem_count(problem),problem_gradient(problem))),Packing.packing_output(Packing.window_input(problem_window(problem),problem_count(problem),problem_input(problem))),initial) == packed_dot_value(upper,problem,key,ReductionModel.accumulator_at(left_key(key),initial)) : F32}:
  match upper:
    case False{}: local_dot_packed(~False{},problem,key,initial,lane_inside)
    case True{}: local_dot_packed(~True{},problem,key,initial,lane_inside)

def initial_element(continued: Bool,+upper: Bool,+row: Accumulator.Row,+lane: Nat,+output: Tree.Tree<F32>,+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {ReductionModel.accumulator_at(ReductionModel.RowLane{row,TileLaws.lane_of(lane)},initial_rows(continued,upper,output)) == Bool.pick(F32,continued,Tree.read(F32,output,GemmWrites.half_address(upper,row,lane)),0.0) : F32}:
  match continued:
    case False{}:
      %Equal.sym(Tile.Tile8,Accumulator.row(~Tile.Tile8,row,Rows.filled(~Tile.Tile8,Tile.filled(0.0))),Tile.filled(0.0),Accumulator.row_filled(~Tile.Tile8,row,Tile.filled(0.0))) :
        {Tile.element_at(TileLaws.lane_of(lane),_) == 0.0 : F32}
      Accumulator.initial_at(TileLaws.lane_of(lane),0.0)
    case True{}:
      +values = Gemms.seed(output,Bool.pick(U32,upper,32,0),8)
      +tile = Accumulator.row(~Tile.Tile8,row,values)
      Equal.trans(F32,Tile.element_at(TileLaws.lane_of(lane),tile),TileLaws.select_lane(lane,tile),Tree.read(F32,output,GemmWrites.half_address(upper,row,lane)),
        Equal.sym(F32,TileLaws.select_lane(lane,tile),Tile.element_at(TileLaws.lane_of(lane),tile),TileLaws.selected_lane(lane,tile)),GemmWrites.half_seed_at(upper,row,lane,output,within))

def window_packed_readback(+continued: Bool,+upper: Bool,+problem: PackedProblem,+key: PackedKey,+output: Tree.Tree<F32>,+balanced: Shape.Balanced(F32,6n,output),+lane_inside: {Nat.is_lt(key_lane(key),8n) == True{} : Bool}) ->
  {Tree.read(F32,window_gemm(continued,problem_count(problem),Packing.packed_tree(Packing.window_gradient(problem_window(problem),problem_count(problem),problem_gradient(problem))),Packing.packing_output(Packing.window_input(problem_window(problem),problem_count(problem),problem_input(problem))),output),GemmWrites.half_address(upper,key_row(key),key_lane(key))) ==
    packed_dot_value(upper,problem,key,Bool.pick(F32,continued,Tree.read(F32,output,GemmWrites.half_address(upper,key_row(key),key_lane(key))),0.0)) : F32}:
  PackedProblem{+window,+count,+input,+gradient,first,positive,row_match,column_match,bounded} = problem
  PackedKey{+row,+lane} = key
  +packed_gradient = Packing.packed_tree(Packing.window_gradient(window,count,gradient))
  +packed_input = Packing.packing_output(Packing.window_input(window,count,input))
  +initial = initial_rows(continued,upper,output)
  +initial_value = ReductionModel.accumulator_at(left_key(key),initial)
  +incoming = Bool.pick(F32,continued,Tree.read(F32,output,GemmWrites.half_address(upper,row,lane)),0.0)
  +actual = Tree.read(F32,window_gemm(continued,count,packed_gradient,packed_input,output),GemmWrites.half_address(upper,row,lane))
  +half = half_dot(upper,row,lane,U32.to_nat(count),packed_gradient,packed_input,initial)
  +natural = natural_dot(upper,row,lane,U32.to_nat(count),packed_gradient,packed_input,initial)
  Equal.trans(F32,actual,half,packed_dot_value(upper,problem,key,incoming),window_readback(continued,upper,row,lane,count,packed_gradient,packed_input,output,balanced,lane_inside),
    Equal.trans(F32,half,natural,packed_dot_value(upper,problem,key,incoming),half_dot_as_natural(upper,row,lane,U32.to_nat(count),packed_gradient,packed_input,initial),
      Equal.trans(F32,natural,packed_dot_value(upper,problem,key,initial_value),packed_dot_value(upper,problem,key,incoming),natural_dot_packed(upper,problem,key,initial,lane_inside),
        Equal.cong(F32,F32,value => packed_dot_value(upper,problem,key,value),initial_value,incoming,initial_element(continued,upper,row,lane,output,lane_inside)))))

# The global position is the reduction key, independently of chunk boundaries.
type NumericSource is Data:
  NumericSource{input: Buffer.Model<F32>,gradient: Buffer.Model<F32>,geometry: Kernel.Geometry,image: U32}

def numeric_input(source: NumericSource,output: U32,position: Nat) -> F32:
  NumericSource{input,gradient,geometry,image} = source
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Tree.read(F32,Buffer.tree(~F32,gradient),U32.from_nat(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(o)),U32.to_nat(output)),U32.to_nat(rows)),U32.to_nat(columns)),position)))

def numeric_weight(source: NumericSource,tap: U32,position: Nat) -> F32:
  NumericSource{input,gradient,+geometry,image} = source
  Kernel.Geometry{c,o,h,w,k,s,p,rows,+columns,taps} = geometry
  +position = position
  Packing.natural_tap_value(input,geometry,image,tap,U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))))

def numeric_source(problem: PackedProblem) -> NumericSource:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap,output,row,column,start} = window
  NumericSource{input,gradient,geometry,image}

def problem_first(problem: PackedProblem) -> Nat:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  first

def problem_geometry(problem: PackedProblem) -> Kernel.Geometry:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap,output,row,column,start} = window
  geometry

def global_tap(+upper: Bool,problem: PackedProblem,key: PackedKey) -> U32:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap,output,row,column,start} = window
  PackedKey{row,lane} = key
  (tap + U32.from_nat(Packing.patch_key(upper,row)) : U32)

def global_output(problem: PackedProblem,key: PackedKey) -> U32:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap,output,row,column,start} = window
  PackedKey{row,lane} = key
  (output + U32.from_nat(lane) : U32)

def geometry_stride(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  s

def geometry_rows(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  rows

def geometry_columns(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  columns

def geometry_outputs(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  o

type NumericAgreement<-upper: Bool> is Data:
  NumericAgreement{limits: Machine.Limits,problem: PackedProblem,key: PackedKey,
    first: {Packing.window_first(problem_window(problem)) == U32.from_nat(problem_first(problem)) : U32},
    kernel: {Nat.is_lt(0n,U32.to_nat(Packing.geometry_kernel(problem_geometry(problem)))) == True{} : Bool},
    stride: {Nat.is_lt(0n,U32.to_nat(geometry_stride(problem_geometry(problem)))) == True{} : Bool},
    square: {Nat.is_le(Nat.mul(U32.to_nat(Packing.geometry_kernel(problem_geometry(problem))),U32.to_nat(Packing.geometry_kernel(problem_geometry(problem)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    tap: {U32.is_lt(global_tap(upper,problem,key),Packing.geometry_taps(problem_geometry(problem))) == True{} : Bool},
    output: {U32.is_lt(global_output(problem,key),geometry_outputs(problem_geometry(problem))) == True{} : Bool},
    room: {Nat.is_le(Nat.add(problem_first(problem),packed_count(problem)),Nat.mul(U32.to_nat(geometry_rows(problem_geometry(problem))),U32.to_nat(geometry_columns(problem_geometry(problem))))) == True{} : Bool}}

def numeric_size(~upper: Bool,context: NumericAgreement<upper>) -> Nat:
  NumericAgreement{limits,problem,key,first,kernel,stride,square,tap,output,room} = context
  packed_count(problem)

def shifted_input() -> Shape.Both<Nat,NumericSource> -> U32 -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~Nat,~U32,~NumericSource,~Nat,~Ordered.shifted_key,~numeric_input)

def shifted_weight() -> Shape.Both<Nat,NumericSource> -> U32 -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~Nat,~U32,~NumericSource,~Nat,~Ordered.shifted_key,~numeric_weight)

def NumericMatching(~upper: Bool,+context: NumericAgreement<upper>,+index: Nat) -> Data:
  NumericAgreement{limits,+problem,+key,first,kernel,stride,square,tap,output,room} = context
  Shape.Both<{target_input(problem,key,index) == shifted_input()(Shape.Both{problem_first(problem),numeric_source(problem)},global_output(problem,key),index) : F32},
    {target_weight(~upper,problem,key,index) == shifted_weight()(Shape.Both{problem_first(problem),numeric_source(problem)},global_tap(upper,problem,key),index) : F32}>

def numeric_cell(~upper: Bool,+context: NumericAgreement<upper>,+index: Nat,+within: {Nat.is_lt(index,numeric_size(~upper,context)) == True{} : Bool}) -> NumericMatching(~upper,context,index):
  NumericAgreement{+limits,+problem,+key,+first_match,+kernel_positive,+stride_positive,+square_bound,+tap_inside,+output_inside,+room} = context
  PackedProblem{+window,+count,+input,+gradient,+first,+positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,+image,+tap0,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,+rows,+columns,taps} = geometry
  PackedKey{+row,+lane} = key
  +position = Nat.add(first,index)
  +pixel = Packing.tap_numeric_pixel(input,geometry,image,global_tap(upper,problem,key),U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))))
  +inside = Bounds.nat_trans(position,Nat.add(first,U32.to_nat(count)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns)),Order.offset_lt(first,index,U32.to_nat(count),within),room)
  +ready = {Packing.PixelReady{stride_positive,Packing.position_row_within(position,rows,columns,positive,inside),Packing.position_column_within(position,columns,positive)} : Packing.PixelReady<pixel>}
  Shape.Both{
    Packing.window_gradient_natural(gradient,window,lane,first,index,first_match,output_inside),
    Equal.trans(F32,Packing.natural_patch_value(Packing.window_patch_context(window,input),upper,row,position),Packing.tap_pixel_at(window,input,Packing.patch_key(upper,row),position),numeric_weight(numeric_source(problem),global_tap(upper,problem,key),position),
      Packing.natural_patch_at(window,input,upper,row,position),
      Packing.tap_pixel_natural(limits,input,geometry,image,global_tap(upper,problem,key),U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))),kernel_positive,square_bound,tap_inside,ready))}

def numeric_dot(source: NumericSource,tap: U32,output: U32,count: Nat,first: Nat,initial: F32) -> F32:
  OrderedMatrix.dot(~U32,~Nat,~U32,~NumericSource,~numeric_input,~numeric_weight,Ranges.range(count,first),tap,output,source,initial)

def numeric_dot_shifted(+source: NumericSource,+tap: U32,+output: U32,+count: Nat,+first: Nat,+initial: F32) ->
  {numeric_dot(source,tap,output,count,first,initial) == OrderedMatrix.dot(~U32,~Nat,~U32,~Shape.Both<Nat,NumericSource>,~shifted_input(),~shifted_weight(),Ranges.range(count,0n),tap,output,Shape.Both{first,source},initial) : F32}:
  +mapped = Ordered.map_keys(~Nat,~Nat,~Nat,~Ordered.shifted_key,Ranges.range(count,0n),first)
  +exact = Equal.trans(List<&2,Nat>,mapped,Ranges.range(count,Nat.add(first,0n)),Ranges.range(count,first),Ordered.range_keys_shift(count,0n,first),Equal.cong(Nat,List<&2,Nat>,start => Ranges.range(count,start),Nat.add(first,0n),first,Addition.zero_right(first)))
  %exact :
    {OrderedMatrix.dot(~U32,~Nat,~U32,~NumericSource,~numeric_input,~numeric_weight,_,tap,output,source,initial) == OrderedMatrix.dot(~U32,~Nat,~U32,~Shape.Both<Nat,NumericSource>,~shifted_input(),~shifted_weight(),Ranges.range(count,0n),tap,output,Shape.Both{first,source},initial) : F32}
  Ordered.dot_reindexed(~Nat,~Nat,~U32,~U32,~NumericSource,~Nat,~Ordered.shifted_key,~numeric_input,~numeric_weight,Ranges.range(count,0n),first,tap,output,source,initial)

def numeric_problem(upper: Bool,context: NumericAgreement<upper>) -> PackedProblem:
  NumericAgreement{limits,problem,key,first,kernel,stride,square,tap,output,room} = context
  problem

def numeric_key(upper: Bool,context: NumericAgreement<upper>) -> PackedKey:
  NumericAgreement{limits,problem,key,first,kernel,stride,square,tap,output,room} = context
  key

def packed_dot_numeric(~upper: Bool,+context: NumericAgreement<upper>,+initial: F32) ->
  {packed_dot(~upper,numeric_problem(upper,context),numeric_key(upper,context),initial) == numeric_dot(numeric_source(numeric_problem(upper,context)),global_tap(upper,numeric_problem(upper,context),numeric_key(upper,context)),global_output(numeric_problem(upper,context),numeric_key(upper,context)),numeric_size(~upper,context),problem_first(numeric_problem(upper,context)),initial) : F32}:
  NumericAgreement{limits,+problem,+key,first,kernel,stride,square,tap,output,room} = context
  +source = numeric_source(problem)
  +first = problem_first(problem)
  +count = packed_count(problem)
  +tap = global_tap(upper,problem,key)
  +output = global_output(problem,key)
  +shifted = OrderedMatrix.dot(~U32,~Nat,~U32,~Shape.Both<Nat,NumericSource>,~shifted_input(),~shifted_weight(),Ranges.range(count,0n),tap,output,Shape.Both{first,source},initial)
  Equal.trans(F32,packed_dot(~upper,problem,key,initial),shifted,numeric_dot(source,tap,output,count,first,initial),
    Ordered.dot_congruence(~Nat,~PackedKey,~PackedKey,~PackedProblem,~U32,~U32,~Shape.Both<Nat,NumericSource>,~target_input,~target_weight(~upper),~shifted_input(),~shifted_weight(),Ranges.range(count,0n),key,key,problem,tap,output,Shape.Both{first,source},initial,
      Members.range_all(~NumericAgreement<upper>,~NumericMatching(~upper),~numeric_size(~upper),~numeric_cell(~upper),count,0n,context,Order.reflexive(count))),
    Equal.sym(F32,numeric_dot(source,tap,output,count,first,initial),shifted,numeric_dot_shifted(source,tap,output,count,first,initial)))

def packed_dot_value_numeric(upper: Bool,+context: NumericAgreement<upper>,+initial: F32) ->
  {packed_dot_value(upper,numeric_problem(upper,context),numeric_key(upper,context),initial) == numeric_dot(numeric_source(numeric_problem(upper,context)),global_tap(upper,numeric_problem(upper,context),numeric_key(upper,context)),global_output(numeric_problem(upper,context),numeric_key(upper,context)),numeric_size(~upper,context),problem_first(numeric_problem(upper,context)),initial) : F32}:
  match upper:
    case False{}: packed_dot_numeric(~False{},context,initial)
    case True{}: packed_dot_numeric(~True{},context,initial)

def window_numeric_readback(+continued: Bool,+upper: Bool,+context: NumericAgreement<upper>,+output: Tree.Tree<F32>,+balanced: Shape.Balanced(F32,6n,output),+lane_inside: {Nat.is_lt(key_lane(numeric_key(upper,context)),8n) == True{} : Bool}) ->
  {Tree.read(F32,window_gemm(continued,problem_count(numeric_problem(upper,context)),Packing.packed_tree(Packing.window_gradient(problem_window(numeric_problem(upper,context)),problem_count(numeric_problem(upper,context)),problem_gradient(numeric_problem(upper,context)))),Packing.packing_output(Packing.window_input(problem_window(numeric_problem(upper,context)),problem_count(numeric_problem(upper,context)),problem_input(numeric_problem(upper,context)))),output),GemmWrites.half_address(upper,key_row(numeric_key(upper,context)),key_lane(numeric_key(upper,context)))) ==
    numeric_dot(numeric_source(numeric_problem(upper,context)),global_tap(upper,numeric_problem(upper,context),numeric_key(upper,context)),global_output(numeric_problem(upper,context),numeric_key(upper,context)),numeric_size(~upper,context),problem_first(numeric_problem(upper,context)),Bool.pick(F32,continued,Tree.read(F32,output,GemmWrites.half_address(upper,key_row(numeric_key(upper,context)),key_lane(numeric_key(upper,context)))),0.0)) : F32}:
  NumericAgreement{limits,+problem,+key,first,kernel,stride,square,tap,selected_output,room} = context
  +incoming = Bool.pick(F32,continued,Tree.read(F32,output,GemmWrites.half_address(upper,key_row(key),key_lane(key))),0.0)
  +actual = Tree.read(F32,window_gemm(continued,problem_count(problem),Packing.packed_tree(Packing.window_gradient(problem_window(problem),problem_count(problem),problem_gradient(problem))),Packing.packing_output(Packing.window_input(problem_window(problem),problem_count(problem),problem_input(problem))),output),GemmWrites.half_address(upper,key_row(key),key_lane(key)))
  Equal.trans(F32,actual,packed_dot_value(upper,problem,key,incoming),numeric_dot(numeric_source(problem),global_tap(upper,problem,key),global_output(problem,key),packed_count(problem),problem_first(problem),incoming),
    window_packed_readback(continued,upper,problem,key,output,balanced,lane_inside),packed_dot_value_numeric(upper,context,incoming))

def clipped_count(+remaining: U32) -> {U32.to_nat(U32.min(256,remaining)) == Nat.min(U32.to_nat(remaining),256n) : Nat}:
  Equal.trans(Nat,U32.to_nat(U32.min(256,remaining)),Nat.min(256n,U32.to_nat(remaining)),Nat.min(U32.to_nat(remaining),256n),NativeExtrema.minimum(256,remaining),Extrema.minimum_commutative(256n,U32.to_nat(remaining)))

def clipped_bound(+remaining: U32) -> {U32.is_le(U32.min(256,remaining),256) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(U32.min(256,remaining),256),Nat.is_le(U32.to_nat(U32.min(256,remaining)),256n),Comparison.le(U32.min(256,remaining),256)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.min(256,remaining)),Nat.min(256n,U32.to_nat(remaining)),NativeExtrema.minimum(256,remaining)) : {Nat.is_le(_,256n) == True{} : Bool}
  Extrema.minimum_left(256n,U32.to_nat(remaining))

def clipped_inside(+remaining: U32) -> {Nat.is_le(U32.to_nat(U32.min(256,remaining)),U32.to_nat(remaining)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.min(256,remaining)),Nat.min(256n,U32.to_nat(remaining)),NativeExtrema.minimum(256,remaining)) : {Nat.is_le(_,U32.to_nat(remaining)) == True{} : Bool}
  Extrema.minimum_right(256n,U32.to_nat(remaining))

def clipped_remaining(+remaining: U32) ->
  {U32.to_nat((remaining - U32.min(256,remaining) : U32)) == Nat.sub(U32.to_nat(remaining),U32.to_nat(U32.min(256,remaining))) : Nat}:
  FullRange.subtract(remaining,U32.min(256,remaining),remaining,clipped_inside(remaining),Order.sub_le(U32.to_nat(remaining),U32.to_nat(U32.min(256,remaining))))

def clipped_room(+position: Nat,+remaining: U32,+rows: U32,+columns: U32,
  +room: {Nat.is_le(Nat.add(position,U32.to_nat(remaining)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(position,U32.to_nat(U32.min(256,remaining))),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}:
  Order.transitive(Nat.add(position,U32.to_nat(U32.min(256,remaining))),Nat.add(position,U32.to_nat(remaining)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns)),
    %Order.commutative(U32.to_nat(U32.min(256,remaining)),position) : {Nat.is_le(_,Nat.add(position,U32.to_nat(remaining))) == True{} : Bool}
    %Order.commutative(U32.to_nat(remaining),position) : {Nat.is_le(Nat.add(U32.to_nat(U32.min(256,remaining)),position),_) == True{} : Bool}
    Order.add_le(U32.to_nat(U32.min(256,remaining)),U32.to_nat(remaining),position,clipped_inside(remaining)),room)

def clipped_next_room(+position: Nat,+remaining: U32,+rows: U32,+columns: U32,
  +room: {Nat.is_le(Nat.add(position,U32.to_nat(remaining)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(position,U32.to_nat(U32.min(256,remaining))),U32.to_nat((remaining - U32.min(256,remaining) : U32))),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}:
  +count = U32.to_nat(U32.min(256,remaining))
  %Equal.sym(Nat,U32.to_nat((remaining - U32.min(256,remaining) : U32)),Nat.sub(U32.to_nat(remaining),count),clipped_remaining(remaining)) : {Nat.is_le(Nat.add(Nat.add(position,count),_),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.add(position,count),Nat.sub(U32.to_nat(remaining),count)),Nat.add(position,Nat.add(count,Nat.sub(U32.to_nat(remaining),count))),Addition.associative(position,count,Nat.sub(U32.to_nat(remaining),count))) : {Nat.is_le(_,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(count,Nat.sub(U32.to_nat(remaining),count)),U32.to_nat(remaining),Order.restore_shift(count,U32.to_nat(remaining),clipped_inside(remaining))) : {Nat.is_le(Nat.add(position,_),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}
  room

type ChunkProof is Data:
  ChunkProof{upper: Bool,agreement: NumericAgreement<upper>,values: Tree.Tree<F32>,remaining: U32,
    count: {problem_count(numeric_problem(upper,agreement)) == U32.min(256,remaining) : U32},
    balanced: Shape.Balanced(F32,6n,values),
    room: {Nat.is_le(Nat.add(problem_first(numeric_problem(upper,agreement)),U32.to_nat(remaining)),Nat.mul(U32.to_nat(geometry_rows(problem_geometry(numeric_problem(upper,agreement)))),U32.to_nat(geometry_columns(problem_geometry(numeric_problem(upper,agreement)))))) == True{} : Bool},
    lane: {Nat.is_lt(key_lane(numeric_key(upper,agreement)),8n) == True{} : Bool}}

def proof_chunk(proof: ChunkProof) -> ChunkModel:
  ChunkProof{upper,agreement,values,remaining,count,balanced,room,lane} = proof
  NumericAgreement{limits,problem,key,first,kernel,stride,square,tap,output,range} = agreement
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  ChunkModel{input,gradient,values,remaining,window}

def count_room(+position: Nat,+remaining: U32,+count: U32,+rows: U32,+columns: U32,+count_match: {count == U32.min(256,remaining) : U32},
  +room: {Nat.is_le(Nat.add(position,U32.to_nat(remaining)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(position,U32.to_nat(count)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}:
  %Equal.sym(U32,count,U32.min(256,remaining),count_match) : {Nat.is_le(Nat.add(position,U32.to_nat(_)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}
  clipped_room(position,remaining,rows,columns,room)

def count_next_room(+position: Nat,+remaining: U32,+count: U32,+rows: U32,+columns: U32,+count_match: {count == U32.min(256,remaining) : U32},
  +room: {Nat.is_le(Nat.add(position,U32.to_nat(remaining)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(position,U32.to_nat(count)),U32.to_nat((remaining - count : U32))),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}:
  %Equal.sym(U32,count,U32.min(256,remaining),count_match) : {Nat.is_le(Nat.add(Nat.add(position,U32.to_nat(_)),U32.to_nat((remaining - _ : U32))),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}
  clipped_next_room(position,remaining,rows,columns,room)

def first_advanced(+word: U32,+first: Nat,+count: U32,+exact: {word == U32.from_nat(first) : U32}) ->
  {(word + count : U32) == U32.from_nat(Nat.add(first,U32.to_nat(count))) : U32}:
  %Equal.sym(U32,word,U32.from_nat(first),exact) : {(_ + count : U32) == U32.from_nat(Nat.add(first,U32.to_nat(count))) : U32}
  %Arithmetic.native_roundtrip(count) : {(U32.from_nat(first) + _ : U32) == U32.from_nat(Nat.add(first,U32.to_nat(count))) : U32}
  WordArithmetic.native_add(first,U32.to_nat(count))

def proof_step(~continued: Bool,proof: ChunkProof) -> ChunkProof:
  ChunkProof{+upper,agreement,+values,+remaining,+count_match,+balanced,+whole_room,+lane_inside} = proof
  NumericAgreement{+limits,problem,+key,first_match,+kernel_positive,+stride_positive,+square_bound,+tap_inside,+output_inside,range} = agreement
  PackedProblem{+window,+count,+input,+gradient,+first,+positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,+image,tap0,output0,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,+rows,+columns,taps} = geometry
  PackedKey{row,lane} = key
  +position = Nat.add(first,U32.to_nat(count))
  +left = (remaining - count : U32)
  +next_count = U32.min(256,left)
  +next_window = {Kernel.WeightWindow{geometry,image,tap0,output0,U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))),U32.from_nat(position)} : Kernel.WeightWindow}
  +next_room = count_next_room(first,remaining,count,rows,columns,count_match,whole_room)
  +next_problem = {PackedProblem{next_window,next_count,input,gradient,position,positive,{==},{==},clipped_bound(left)} : PackedProblem}
  +next_agreement = {NumericAgreement{limits,next_problem,key,{==},kernel_positive,stride_positive,square_bound,tap_inside,output_inside,clipped_room(position,left,rows,columns,next_room)} : NumericAgreement<upper>}
  +packed_gradient = Packing.packed_tree(Packing.window_gradient(window,count,gradient))
  +packed_input = Packing.packing_output(Packing.window_input(window,count,input))
  +output = window_gemm(continued,count,packed_gradient,packed_input,values)
  ChunkProof{upper,next_agreement,output,left,{==},window_balance(continued,count,packed_gradient,packed_input,values,balanced),next_room,lane_inside}

def advanced_window(window: Kernel.WeightWindow,+position: Nat) -> Kernel.WeightWindow:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,+columns,taps} = geometry
  Kernel.WeightWindow{geometry,image,tap,output,U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))),U32.from_nat(position)}

def observed_window(window: Kernel.WeightWindow,count: U32,packed: Packing.Packing) -> Kernel.WeightWindow:
  Kernel.WeightWindow{geometry,image,tap,output,row,column,first} = window
  Packing.Packing{input,weights,index,row2,column2,columns,patch,h,w,s,p} = packed
  Kernel.WeightWindow{geometry,image,tap,output,row2,column2,(first + count : U32)}

def chunk_result_observed(~continued: Bool,+count: U32,+remaining: U32,+window: Kernel.WeightWindow,+values: Tree.Tree<F32>,+packed: Packing.Packing,+read: Packing.Packed) ->
  {chunk_result(~continued,count,remaining,window,values,packed,read) == ChunkModel{Packing.packing_reader(packed),Packing.packed_reader(read),window_gemm(continued,count,Packing.packed_tree(read),Packing.packing_output(packed),values),(remaining - count : U32),observed_window(window,count,packed)} : ChunkModel}:
  Kernel.WeightWindow{geometry,image,tap,output,row,column,first} = window
  Packing.Packing{input,weights,index,row2,column2,columns,patch,h,w,s,p} = packed
  Packing.Packed{gradient,tiles} = read
  {==}

def observed_window_projected(+window: Kernel.WeightWindow,+count: U32,packed: Packing.Packing,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>,+values: Tree.Tree<F32>,+remaining: U32,+position: Nat,
  +row_match: {Packing.packing_row(packed) == U32.from_nat(Nat.div(Nat.add(position,U32.to_nat(count)),U32.to_nat(Packing.window_columns(window)))) : U32},
  +column_match: {Packing.packing_column(packed) == U32.from_nat(Nat.mod(Nat.add(position,U32.to_nat(count)),U32.to_nat(Packing.window_columns(window)))) : U32},
  +first_match: {(Packing.window_first(window) + count : U32) == U32.from_nat(Nat.add(position,U32.to_nat(count))) : U32}) ->
  {ChunkModel{input,gradient,values,remaining,observed_window(window,count,packed)} == ChunkModel{input,gradient,values,remaining,advanced_window(window,Nat.add(position,U32.to_nat(count)))} : ChunkModel}:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,+first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,+columns,taps} = geometry
  Packing.Packing{reader,weights,index,+row2,+column2,columns2,patch,height,width,stride,padding} = packed
  +next = Nat.add(position,U32.to_nat(count))
  %Equal.sym(U32,row2,U32.from_nat(Nat.div(next,U32.to_nat(columns))),row_match) : {ChunkModel{input,gradient,values,remaining,Kernel.WeightWindow{geometry,image,tap,output,_,column2,(first + count : U32)}} == ChunkModel{input,gradient,values,remaining,advanced_window(window,next)} : ChunkModel}
  %Equal.sym(U32,column2,U32.from_nat(Nat.mod(next,U32.to_nat(columns))),column_match) : {ChunkModel{input,gradient,values,remaining,Kernel.WeightWindow{geometry,image,tap,output,U32.from_nat(Nat.div(next,U32.to_nat(columns))),_,(first + count : U32)}} == ChunkModel{input,gradient,values,remaining,advanced_window(window,next)} : ChunkModel}
  Equal.cong(U32,ChunkModel,start => ChunkModel{input,gradient,values,remaining,Kernel.WeightWindow{geometry,image,tap,output,U32.from_nat(Nat.div(next,U32.to_nat(columns))),U32.from_nat(Nat.mod(next,U32.to_nat(columns))),start}},(first + count : U32),U32.from_nat(next),first_match)
def observed_window_canonical(+window: Kernel.WeightWindow,+count: U32,+packed_input: Buffer.Model<F32>,+position: Nat,
  +positive: {Nat.is_lt(0n,U32.to_nat(Packing.window_columns(window))) == True{} : Bool},
  +row_match: {Packing.window_row(window) == U32.from_nat(Nat.div(position,U32.to_nat(Packing.window_columns(window)))) : U32},
  +column_match: {Packing.window_column(window) == U32.from_nat(Nat.mod(position,U32.to_nat(Packing.window_columns(window)))) : U32},
  +first_match: {Packing.window_first(window) == U32.from_nat(position) : U32},+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>,+values: Tree.Tree<F32>,+remaining: U32) ->
  {ChunkModel{input,gradient,values,remaining,observed_window(window,count,Packing.window_input(window,count,packed_input))} == ChunkModel{input,gradient,values,remaining,advanced_window(window,Nat.add(position,U32.to_nat(count)))} : ChunkModel}:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,+first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,+columns,taps} = geometry
  +packed = Packing.window_input(window,count,packed_input)
  +next = Nat.add(position,U32.to_nat(count))
  observed_window_projected(window,count,packed,input,gradient,values,remaining,position,
    Packing.window_input_row_end(window,count,packed_input,position,positive,row_match,column_match),Packing.window_input_column_end(window,count,packed_input,position,positive,row_match,column_match),first_advanced(first,position,count,first_match))

def chunk_result_canonical(~continued: Bool,+count: U32,+remaining: U32,+window: Kernel.WeightWindow,+values: Tree.Tree<F32>,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>,+position: Nat,
  +positive: {Nat.is_lt(0n,U32.to_nat(Packing.window_columns(window))) == True{} : Bool},
  +row_match: {Packing.window_row(window) == U32.from_nat(Nat.div(position,U32.to_nat(Packing.window_columns(window)))) : U32},
  +column_match: {Packing.window_column(window) == U32.from_nat(Nat.mod(position,U32.to_nat(Packing.window_columns(window)))) : U32},
  +first_match: {Packing.window_first(window) == U32.from_nat(position) : U32}) ->
  {chunk_result(~continued,count,remaining,window,values,Packing.window_input(window,count,input),Packing.window_gradient(window,count,gradient)) ==
    ChunkModel{input,gradient,window_gemm(continued,count,Packing.packed_tree(Packing.window_gradient(window,count,gradient)),Packing.packing_output(Packing.window_input(window,count,input)),values),(remaining - count : U32),advanced_window(window,Nat.add(position,U32.to_nat(count)))} : ChunkModel}:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,+start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,+columns,taps} = geometry
  +packed = Packing.window_input(window,count,input)
  +read = Packing.window_gradient(window,count,gradient)
  +values2 = window_gemm(continued,count,Packing.packed_tree(read),Packing.packing_output(packed),values)
  +next = Nat.add(position,U32.to_nat(count))
  +wanted = {ChunkModel{input,gradient,values2,(remaining - count : U32),advanced_window(window,next)} : ChunkModel}
  %Equal.sym(ChunkModel,chunk_result(~continued,count,remaining,window,values,packed,read),ChunkModel{Packing.packing_reader(packed),Packing.packed_reader(read),values2,(remaining - count : U32),observed_window(window,count,packed)},chunk_result_observed(~continued,count,remaining,window,values,packed,read)) : {_ == wanted : ChunkModel}
  %Equal.sym(Buffer.Model<F32>,Packing.packing_reader(packed),input,Packing.window_input_kept(window,count,input,position,positive,row_match,column_match)) : {ChunkModel{_,Packing.packed_reader(read),values2,(remaining - count : U32),observed_window(window,count,packed)} == wanted : ChunkModel}
  %Equal.sym(Buffer.Model<F32>,Packing.packed_reader(read),gradient,Packing.window_gradient_kept(window,count,gradient)) : {ChunkModel{input,_,values2,(remaining - count : U32),observed_window(window,count,packed)} == wanted : ChunkModel}
  observed_window_canonical(window,count,input,position,positive,row_match,column_match,first_match,input,gradient,values2,(remaining - count : U32))

def proof_step_matches(~continued: Bool,+proof: ChunkProof) ->
  {chunk(~continued,proof_chunk(proof)) == proof_chunk(proof_step(~continued,proof)) : ChunkModel}:
  ChunkProof{+upper,agreement,+values,+remaining,+count_match,balanced,whole_room,lane_inside} = proof
  NumericAgreement{limits,problem,key,+first_match,kernel_positive,stride_positive,square_bound,tap_inside,output_inside,range} = agreement
  PackedProblem{+window,+count,+input,+gradient,+first,+positive,+row_match,+column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap0,output0,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  PackedKey{row,lane} = key
  %count_match :
    {chunk_result(~continued,_,remaining,window,values,Packing.window_input(window,_,input),Packing.window_gradient(window,_,gradient)) == proof_chunk(proof_step(~continued,proof)) : ChunkModel}
  chunk_result_canonical(~continued,count,remaining,window,values,input,gradient,first,positive,row_match,column_match,first_match)

def proof_loop(+count: Nat,+proof: ChunkProof) ->
  {Loop.run(~ChunkModel,~chunk(~True{}),count,proof_chunk(proof)) == proof_chunk(Loop.run(~ChunkProof,~proof_step(~True{}),count,proof)) : ChunkModel}:
  Projection.iteration_refines_model(~ChunkModel,~ChunkProof,~chunk(~True{}),~proof_step(~True{}),~proof_chunk,~proof_step_matches(~True{}),count,proof)

type ScalarChunk is Data:
  ScalarChunk{source: NumericSource,tap: U32,output: U32,position: Nat,remaining: U32,total: F32}

def scalar_step(~continued: Bool,state: ScalarChunk) -> ScalarChunk:
  ScalarChunk{+source,+tap,+output,+position,+remaining,total} = state
  +count = U32.min(256,remaining)
  ScalarChunk{source,tap,output,Nat.add(position,U32.to_nat(count)),(remaining - count : U32),numeric_dot(source,tap,output,U32.to_nat(count),position,Bool.pick(F32,continued,total,0.0))}

def proof_scalar(proof: ChunkProof) -> ScalarChunk:
  ChunkProof{+upper,agreement,values,remaining,count,balanced,room,lane_inside} = proof
  NumericAgreement{limits,problem,+key,first_match,kernel_positive,stride_positive,square_bound,tap_inside,output_inside,range} = agreement
  PackedProblem{+window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap0,output0,row0,column0,start} = window
  PackedKey{+row,+lane} = key
  ScalarChunk{NumericSource{input,gradient,geometry,image},(tap0 + U32.from_nat(Packing.patch_key(upper,row)) : U32),(output0 + U32.from_nat(lane) : U32),first,remaining,Tree.read(F32,values,GemmWrites.half_address(upper,row,lane))}

def scalar_step_matches(~continued: Bool,+proof: ChunkProof) ->
  {scalar_step(~continued,proof_scalar(proof)) == proof_scalar(proof_step(~continued,proof)) : ScalarChunk}:
  ChunkProof{+upper,+agreement,+values,+remaining,+count_match,+balanced,whole_room,+lane_inside} = proof
  NumericAgreement{limits,+problem,+key,first_match,kernel_positive,stride_positive,square_bound,tap_inside,output_inside,range} = agreement
  PackedProblem{+window,+count,+input,+gradient,+first,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,+image,+tap0,+output0,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  PackedKey{+row,+lane} = key
  +source = {NumericSource{input,gradient,geometry,image} : NumericSource}
  +tap = (tap0 + U32.from_nat(Packing.patch_key(upper,row)) : U32)
  +output = (output0 + U32.from_nat(lane) : U32)
  +old = Tree.read(F32,values,GemmWrites.half_address(upper,row,lane))
  +incoming = Bool.pick(F32,continued,old,0.0)
  +packed_gradient = Packing.packed_tree(Packing.window_gradient(window,count,gradient))
  +packed_input = Packing.packing_output(Packing.window_input(window,count,input))
  +actual = Tree.read(F32,window_gemm(continued,count,packed_gradient,packed_input,values),GemmWrites.half_address(upper,row,lane))
  +wanted = numeric_dot(source,tap,output,U32.to_nat(count),first,incoming)
  %count_match :
    {ScalarChunk{source,tap,output,Nat.add(first,U32.to_nat(_)),(remaining - _ : U32),numeric_dot(source,tap,output,U32.to_nat(_),first,incoming)} == proof_scalar(proof_step(~continued,proof)) : ScalarChunk}
  Equal.cong(F32,ScalarChunk,total => ScalarChunk{source,tap,output,Nat.add(first,U32.to_nat(count)),(remaining - count : U32),total},wanted,actual,
    Equal.sym(F32,actual,wanted,window_numeric_readback(continued,upper,agreement,values,balanced,lane_inside)))

def scalar_loop(+count: Nat,+proof: ChunkProof) ->
  {Loop.run(~ScalarChunk,~scalar_step(~True{}),count,proof_scalar(proof)) == proof_scalar(Loop.run(~ChunkProof,~proof_step(~True{}),count,proof)) : ScalarChunk}:
  Projection.iteration_refines_model(~ScalarChunk,~ChunkProof,~scalar_step(~True{}),~proof_step(~True{}),~proof_scalar,~scalar_step_matches(~True{}),count,proof)

def numeric_split(+source: NumericSource,+tap: U32,+output: U32,+left: Nat,+right: Nat,+first: Nat,+initial: F32) ->
  {numeric_dot(source,tap,output,Nat.add(left,right),first,initial) == numeric_dot(source,tap,output,right,Nat.add(first,left),numeric_dot(source,tap,output,left,first,initial)) : F32}:
  %Equal.sym(List<&2,Nat>,Ranges.range(Nat.add(left,right),first),OrderedMatrix.cat(Nat,Ranges.range(left,first),Ranges.range(right,Nat.add(first,left))),Ranges.split(left,right,first)) :
    {OrderedMatrix.dot(~U32,~Nat,~U32,~NumericSource,~numeric_input,~numeric_weight,_,tap,output,source,initial) == numeric_dot(source,tap,output,right,Nat.add(first,left),numeric_dot(source,tap,output,left,first,initial)) : F32}
  Ordered.reduction_split(~U32,~Nat,~U32,~NumericSource,~numeric_input,~numeric_weight,Ranges.range(left,first),Ranges.range(right,Nat.add(first,left)),tap,output,source,initial)

def clipped_dot(+source: NumericSource,+tap: U32,+output: U32,+whole: Nat,+head_capacity: Nat,+tail_capacity: Nat,+first: Nat,+initial: F32) ->
  {numeric_dot(source,tap,output,Nat.min(Nat.sub(whole,Nat.min(whole,head_capacity)),tail_capacity),Nat.add(first,Nat.min(whole,head_capacity)),numeric_dot(source,tap,output,Nat.min(whole,head_capacity),first,initial)) == numeric_dot(source,tap,output,Nat.min(whole,Nat.add(head_capacity,tail_capacity)),first,initial) : F32}:
  +head = Nat.min(whole,head_capacity)
  +tail = Nat.min(Nat.sub(whole,head),tail_capacity)
  Equal.trans(F32,numeric_dot(source,tap,output,tail,Nat.add(first,head),numeric_dot(source,tap,output,head,first,initial)),numeric_dot(source,tap,output,Nat.add(head,tail),first,initial),numeric_dot(source,tap,output,Nat.min(whole,Nat.add(head_capacity,tail_capacity)),first,initial),
    Equal.sym(F32,numeric_dot(source,tap,output,Nat.add(head,tail),first,initial),numeric_dot(source,tap,output,tail,Nat.add(first,head),numeric_dot(source,tap,output,head,first,initial)),numeric_split(source,tap,output,head,tail,first,initial)),
    Equal.cong(Nat,F32,count => numeric_dot(source,tap,output,count,first,initial),Nat.add(head,tail),Nat.min(whole,Nat.add(head_capacity,tail_capacity)),Extrema.minimum_split(whole,head_capacity,tail_capacity)))

def scalar_total(context: Unit,state: ScalarChunk) -> F32:
  ScalarChunk{source,tap,output,position,remaining,total} = state
  total

def scalar_denote(context: Unit,turns: Nat,state: ScalarChunk) -> F32:
  ScalarChunk{source,tap,output,position,remaining,total} = state
  numeric_dot(source,tap,output,Nat.min(U32.to_nat(remaining),Nat.mul(turns,256n)),position,total)

def scalar_zero(+context: Unit,+state: ScalarChunk) -> {scalar_total(context,state) == scalar_denote(context,0n,state) : F32}:
  ScalarChunk{+source,+tap,+output,+position,+remaining,+total} = state
  %Equal.sym(Nat,Nat.min(U32.to_nat(remaining),0n),0n,Extrema.minimum_commutative(U32.to_nat(remaining),0n)) : {total == numeric_dot(source,tap,output,_,position,total) : F32}
  {==}

def scalar_after(+context: Unit,+turns: Nat,+state: ScalarChunk) -> {scalar_denote(context,turns,scalar_step(~True{},state)) == scalar_denote(context,1n+turns,state) : F32}:
  ScalarChunk{+source,+tap,+output,+position,+remaining,+total} = state
  +count = U32.to_nat(U32.min(256,remaining))
  +whole = U32.to_nat(remaining)
  +capacity = Nat.mul(turns,256n)
  %Equal.sym(Nat,U32.to_nat((remaining - U32.min(256,remaining) : U32)),Nat.sub(whole,count),clipped_remaining(remaining)) :
    {numeric_dot(source,tap,output,Nat.min(_,capacity),Nat.add(position,count),numeric_dot(source,tap,output,count,position,total)) == scalar_denote(context,1n+turns,state) : F32}
  %Equal.sym(Nat,count,Nat.min(whole,256n),clipped_count(remaining)) :
    {numeric_dot(source,tap,output,Nat.min(Nat.sub(whole,_),capacity),Nat.add(position,_),numeric_dot(source,tap,output,_,position,total)) == scalar_denote(context,1n+turns,state) : F32}
  clipped_dot(source,tap,output,whole,256n,capacity,position,total)

def scalar_fold(+turns: Nat,+state: ScalarChunk) ->
  {scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),turns,state)) == scalar_denote(Unit{},turns,state) : F32}:
  Projection.iteration_matches_denotation(~F32,~ScalarChunk,~Unit,~scalar_step(~True{}),~scalar_total,~scalar_denote,~scalar_zero,~scalar_after,turns,Unit{},state)

def turns_word(+count: U32) -> U32:
  (((count - 1) / 256) + 1 : U32)

def turns_formula(+count: Nat,+positive: {Nat.is_lt(0n,count) == True{} : Bool}) ->
  {Ceiling.ceiling(count,256n) == 1n+Nat.div(Nat.sub(count,1n),256n) : Nat}:
  %Order.positive_has_last(count,positive) : {Ceiling.ceiling(_,256n) == 1n+Nat.div(Nat.sub(count,1n),256n) : Nat}
  Ceiling.successor_formula(Nat.sub(count,1n),256n,{==})

def turns_quotient(+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) ->
  {U32.to_nat(((count - 1) / 256 : U32)) == Nat.div(Nat.sub(U32.to_nat(count),1n),256n) : Nat}:
  +before = (count - 1 : U32)
  +exact = FullRange.subtract(count,1,count,Order.strict_follow(0n,U32.to_nat(count),positive),Order.sub_le(U32.to_nat(count),1n))
  Equal.trans(Nat,U32.to_nat((before / 256 : U32)),Nat.div(U32.to_nat(before),256n),Nat.div(Nat.sub(U32.to_nat(count),1n),256n),
    NativeDivision.divide(before,256,2147483648,{==},{==},{==}),Equal.cong(Nat,Nat,value => Nat.div(value,256n),U32.to_nat(before),Nat.sub(U32.to_nat(count),1n),exact))

def turns_exact(+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) ->
  {U32.to_nat(turns_word(count)) == Ceiling.ceiling(U32.to_nat(count),256n) : Nat}:
  +quotient = ((count - 1) / 256 : U32)
  +natural = Nat.div(Nat.sub(U32.to_nat(count),1n),256n)
  +exact = turns_quotient(count,positive)
  +formula = turns_formula(U32.to_nat(count),positive)
  +room = {
    %Equal.sym(Nat,U32.to_nat(quotient),natural,exact) : {Nat.is_le(Nat.add(_,1n),U32.to_nat(count)) == True{} : Bool}
    %Order.commutative(1n,natural) : {Nat.is_le(_,U32.to_nat(count)) == True{} : Bool}
    %formula : {Nat.is_le(_,U32.to_nat(count)) == True{} : Bool}
    Ceiling.bounded(U32.to_nat(count),256n,{==}) : {Nat.is_le(Nat.add(U32.to_nat(quotient),1n),U32.to_nat(count)) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat(turns_word(count)),Nat.add(U32.to_nat(quotient),1n),Ceiling.ceiling(U32.to_nat(count),256n),FullRange.add(quotient,1,count,room),
    Equal.trans(Nat,Nat.add(U32.to_nat(quotient),1n),Nat.add(natural,1n),Ceiling.ceiling(U32.to_nat(count),256n),Equal.cong(Nat,Nat,value => Nat.add(value,1n),U32.to_nat(quotient),natural,exact),
      Equal.trans(Nat,Nat.add(natural,1n),1n+natural,Ceiling.ceiling(U32.to_nat(count),256n),Order.commutative(natural,1n),Equal.sym(Nat,Ceiling.ceiling(U32.to_nat(count),256n),1n+natural,formula))))

def turns_positive(+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(turns_word(count))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(turns_word(count)),Ceiling.ceiling(U32.to_nat(count),256n),turns_exact(count,positive)) : {Nat.is_lt(0n,_) == True{} : Bool}
  %Equal.sym(Nat,Ceiling.ceiling(U32.to_nat(count),256n),1n+Nat.div(Nat.sub(U32.to_nat(count),1n),256n),turns_formula(U32.to_nat(count),positive)) : {Nat.is_lt(0n,_) == True{} : Bool}
  {==}

def turns_remaining(+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) ->
  {1n+U32.to_nat((turns_word(count) - 1 : U32)) == Ceiling.ceiling(U32.to_nat(count),256n) : Nat}:
  +turns = turns_word(count)
  +nonempty = turns_positive(count,positive)
  +subtracted = FullRange.subtract(turns,1,turns,Order.strict_follow(0n,U32.to_nat(turns),nonempty),Order.sub_le(U32.to_nat(turns),1n))
  Equal.trans(Nat,1n+U32.to_nat((turns - 1 : U32)),1n+Nat.sub(U32.to_nat(turns),1n),Ceiling.ceiling(U32.to_nat(count),256n),
    Equal.cong(Nat,Nat,value => 1n+value,U32.to_nat((turns - 1 : U32)),Nat.sub(U32.to_nat(turns),1n),subtracted),
    Equal.trans(Nat,1n+Nat.sub(U32.to_nat(turns),1n),U32.to_nat(turns),Ceiling.ceiling(U32.to_nat(count),256n),Order.positive_has_last(U32.to_nat(turns),nonempty),turns_exact(count,positive)))

def scalar_full(+source: NumericSource,+tap: U32,+output: U32,+position: Nat,+remaining: U32,+total: F32) ->
  {scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),Ceiling.ceiling(U32.to_nat(remaining),256n),ScalarChunk{source,tap,output,position,remaining,total})) == numeric_dot(source,tap,output,U32.to_nat(remaining),position,total) : F32}:
  +state = {ScalarChunk{source,tap,output,position,remaining,total} : ScalarChunk}
  +turns = Ceiling.ceiling(U32.to_nat(remaining),256n)
  +count = Nat.min(U32.to_nat(remaining),Nat.mul(turns,256n))
  Equal.trans(F32,scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),turns,state)),scalar_denote(Unit{},turns,state),numeric_dot(source,tap,output,U32.to_nat(remaining),position,total),scalar_fold(turns,state),
    Equal.cong(Nat,F32,size => numeric_dot(source,tap,output,size,position,total),count,U32.to_nat(remaining),Extrema.minimum_when_ordered(U32.to_nat(remaining),Nat.mul(turns,256n),Ceiling.covers(U32.to_nat(remaining),256n,{==}))))

type Query is Data:
  Query{upper: Bool,key: PackedKey}

def proof_query(proof: ChunkProof) -> Query:
  ChunkProof{upper,agreement,values,remaining,count,balanced,room,lane_inside} = proof
  NumericAgreement{limits,problem,key,first_match,kernel_positive,stride_positive,square_bound,tap_inside,output_inside,range} = agreement
  Query{upper,key}

def proof_query_step(~continued: Bool,+proof: ChunkProof) -> {proof_query(proof_step(~continued,proof)) == proof_query(proof) : Query}:
  ChunkProof{upper,agreement,values,remaining,count,balanced,room,lane_inside} = proof
  NumericAgreement{limits,problem,key,first_match,kernel_positive,stride_positive,square_bound,tap_inside,output_inside,range} = agreement
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap,output,row,column,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  PackedKey{row,lane} = key
  {==}

def proof_query_kept(+count: Nat,+proof: ChunkProof) -> {proof_query(Loop.run(~ChunkProof,~proof_step(~True{}),count,proof)) == proof_query(proof) : Query}:
  Projection.preserves_observation(~ChunkProof,~Query,~proof_step(~True{}),~proof_query,~proof_query_step(~True{}),count,proof)

def query_read(query: Query,state: ChunkModel) -> F32:
  Query{upper,key} = query
  PackedKey{row,lane} = key
  ChunkModel{input,gradient,values,remaining,window} = state
  Tree.read(F32,values,GemmWrites.half_address(upper,row,lane))

def proof_read(+proof: ChunkProof) -> {query_read(proof_query(proof),proof_chunk(proof)) == scalar_total(Unit{},proof_scalar(proof)) : F32}:
  ChunkProof{upper,agreement,values,remaining,count,balanced,room,lane_inside} = proof
  NumericAgreement{limits,problem,key,first_match,kernel_positive,stride_positive,square_bound,tap_inside,output_inside,range} = agreement
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap,output,row,column,start} = window
  PackedKey{row,lane} = key
  {==}

def proof_loop_read(+count: Nat,+proof: ChunkProof) ->
  {query_read(proof_query(proof),Loop.run(~ChunkModel,~chunk(~True{}),count,proof_chunk(proof))) == scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),count,proof_scalar(proof))) : F32}:
  +result = Loop.run(~ChunkProof,~proof_step(~True{}),count,proof)
  %Equal.sym(ChunkModel,Loop.run(~ChunkModel,~chunk(~True{}),count,proof_chunk(proof)),proof_chunk(result),proof_loop(count,proof)) :
    {query_read(proof_query(proof),_) == scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),count,proof_scalar(proof))) : F32}
  %proof_query_kept(count,proof) : {query_read(_,proof_chunk(result)) == scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),count,proof_scalar(proof))) : F32}
  Equal.trans(F32,query_read(proof_query(result),proof_chunk(result)),scalar_total(Unit{},proof_scalar(result)),scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),count,proof_scalar(proof))),proof_read(result),
    Equal.cong(ScalarChunk,F32,scalar_total(Unit{}),proof_scalar(result),Loop.run(~ScalarChunk,~scalar_step(~True{}),count,proof_scalar(proof)),Equal.sym(ScalarChunk,Loop.run(~ScalarChunk,~scalar_step(~True{}),count,proof_scalar(proof)),proof_scalar(result),scalar_loop(count,proof))))

def initial_proof(+upper: Bool,+agreement: NumericAgreement<upper>,+remaining: U32,
  count: {problem_count(numeric_problem(upper,agreement)) == U32.min(256,remaining) : U32},
  room: {Nat.is_le(Nat.add(problem_first(numeric_problem(upper,agreement)),U32.to_nat(remaining)),Nat.mul(U32.to_nat(geometry_rows(problem_geometry(numeric_problem(upper,agreement)))),U32.to_nat(geometry_columns(problem_geometry(numeric_problem(upper,agreement)))))) == True{} : Bool},
  lane: {Nat.is_lt(key_lane(numeric_key(upper,agreement)),8n) == True{} : Bool}) -> ChunkProof:
  ChunkProof{upper,agreement,Tree.fresh(F32,6n,0.0),remaining,count,Shape.fresh(F32,6n,0.0),room,lane}

def initial_scalar(+upper: Bool,+agreement: NumericAgreement<upper>,+remaining: U32,
  +count: {problem_count(numeric_problem(upper,agreement)) == U32.min(256,remaining) : U32},
  +room: {Nat.is_le(Nat.add(problem_first(numeric_problem(upper,agreement)),U32.to_nat(remaining)),Nat.mul(U32.to_nat(geometry_rows(problem_geometry(numeric_problem(upper,agreement)))),U32.to_nat(geometry_columns(problem_geometry(numeric_problem(upper,agreement)))))) == True{} : Bool},
  +lane: {Nat.is_lt(key_lane(numeric_key(upper,agreement)),8n) == True{} : Bool}) ->
  {proof_scalar(initial_proof(upper,agreement,remaining,count,room,lane)) == ScalarChunk{numeric_source(numeric_problem(upper,agreement)),global_tap(upper,numeric_problem(upper,agreement),numeric_key(upper,agreement)),global_output(numeric_problem(upper,agreement),numeric_key(upper,agreement)),problem_first(numeric_problem(upper,agreement)),remaining,0.0} : ScalarChunk}:
  NumericAgreement{limits,+problem,+key,first_match,kernel_positive,stride_positive,square_bound,tap_inside,output_inside,range} = agreement
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap,output,row0,column0,start} = window
  PackedKey{+row,+index} = key
  +source = numeric_source(problem)
  +tap = global_tap(upper,problem,key)
  +output = global_output(problem,key)
  Equal.cong(F32,ScalarChunk,total => ScalarChunk{source,tap,output,first,remaining,total},Tree.read(F32,Tree.fresh(F32,6n,0.0),GemmWrites.half_address(upper,row,index)),0.0,StorageReadback.read_fresh(~F32,6n,0.0,GemmWrites.half_address(upper,row,index)))

def initial_scalar_step(+upper: Bool,+agreement: NumericAgreement<upper>,+remaining: U32,
  +count: {problem_count(numeric_problem(upper,agreement)) == U32.min(256,remaining) : U32},
  +room: {Nat.is_le(Nat.add(problem_first(numeric_problem(upper,agreement)),U32.to_nat(remaining)),Nat.mul(U32.to_nat(geometry_rows(problem_geometry(numeric_problem(upper,agreement)))),U32.to_nat(geometry_columns(problem_geometry(numeric_problem(upper,agreement)))))) == True{} : Bool},
  +lane: {Nat.is_lt(key_lane(numeric_key(upper,agreement)),8n) == True{} : Bool}) ->
  {proof_scalar(proof_step(~False{},initial_proof(upper,agreement,remaining,count,room,lane))) == scalar_step(~True{},ScalarChunk{numeric_source(numeric_problem(upper,agreement)),global_tap(upper,numeric_problem(upper,agreement),numeric_key(upper,agreement)),global_output(numeric_problem(upper,agreement),numeric_key(upper,agreement)),problem_first(numeric_problem(upper,agreement)),remaining,0.0}) : ScalarChunk}:
  +proof = initial_proof(upper,agreement,remaining,count,room,lane)
  +state = {ScalarChunk{numeric_source(numeric_problem(upper,agreement)),global_tap(upper,numeric_problem(upper,agreement),numeric_key(upper,agreement)),global_output(numeric_problem(upper,agreement),numeric_key(upper,agreement)),problem_first(numeric_problem(upper,agreement)),remaining,0.0} : ScalarChunk}
  Equal.trans(ScalarChunk,proof_scalar(proof_step(~False{},proof)),scalar_step(~False{},proof_scalar(proof)),scalar_step(~True{},state),
    Equal.sym(ScalarChunk,scalar_step(~False{},proof_scalar(proof)),proof_scalar(proof_step(~False{},proof)),scalar_step_matches(~False{},proof)),
    Equal.trans(ScalarChunk,scalar_step(~False{},proof_scalar(proof)),scalar_step(~False{},state),scalar_step(~True{},state),
      Equal.cong(ScalarChunk,ScalarChunk,scalar_step(~False{}),proof_scalar(proof),state,initial_scalar(upper,agreement,remaining,count,room,lane)),{==}))

def first_loop_read(+count: Nat,+proof: ChunkProof) ->
  {query_read(proof_query(proof),Loop.run(~ChunkModel,~chunk(~True{}),count,chunk(~False{},proof_chunk(proof)))) == scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),count,proof_scalar(proof_step(~False{},proof)))) : F32}:
  +first = proof_step(~False{},proof)
  %Equal.sym(ChunkModel,chunk(~False{},proof_chunk(proof)),proof_chunk(first),proof_step_matches(~False{},proof)) :
    {query_read(proof_query(proof),Loop.run(~ChunkModel,~chunk(~True{}),count,_)) == scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),count,proof_scalar(first))) : F32}
  %proof_query_step(~False{},proof) :
    {query_read(_,Loop.run(~ChunkModel,~chunk(~True{}),count,proof_chunk(first))) == scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),count,proof_scalar(first))) : F32}
  proof_loop_read(count,first)

def group_read(query: Query,result: Group) -> F32:
  Query{upper,key} = query
  PackedKey{row,lane} = key
  Group{input,gradient,values} = result
  Tree.read(F32,values,GemmWrites.half_address(upper,row,lane))

def chunk_read(+query: Query,+state: ChunkModel) -> {group_read(query,chunk_values(state)) == query_read(query,state) : F32}:
  Query{upper,key} = query
  PackedKey{row,lane} = key
  ChunkModel{input,gradient,values,remaining,window} = state
  {==}

def query_address(query: Query) -> U32:
  Query{upper,key} = query
  PackedKey{row,lane} = key
  GemmWrites.half_address(upper,row,lane)

def group_read_output(+query: Query,+result: Group) ->
  {group_read(query,result) == Tree.read(F32,group_output(result),query_address(query)) : F32}:
  Query{upper,key} = query
  PackedKey{row,lane} = key
  Group{input,gradient,values} = result
  {==}

def window_product_numeric(+upper: Bool,+agreement: NumericAgreement<upper>,+remaining: U32,
  +count_match: {problem_count(numeric_problem(upper,agreement)) == U32.min(256,remaining) : U32},
  +whole_room: {Nat.is_le(Nat.add(problem_first(numeric_problem(upper,agreement)),U32.to_nat(remaining)),Nat.mul(U32.to_nat(geometry_rows(problem_geometry(numeric_problem(upper,agreement)))),U32.to_nat(geometry_columns(problem_geometry(numeric_problem(upper,agreement)))))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(key_lane(numeric_key(upper,agreement)),8n) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,U32.to_nat(remaining)) == True{} : Bool}) ->
  {group_read(Query{upper,numeric_key(upper,agreement)},window_product(remaining,problem_window(numeric_problem(upper,agreement)),problem_input(numeric_problem(upper,agreement)),problem_gradient(numeric_problem(upper,agreement)))) == numeric_dot(numeric_source(numeric_problem(upper,agreement)),global_tap(upper,numeric_problem(upper,agreement),numeric_key(upper,agreement)),global_output(numeric_problem(upper,agreement),numeric_key(upper,agreement)),U32.to_nat(remaining),problem_first(numeric_problem(upper,agreement)),0.0) : F32}:
  NumericAgreement{limits,+problem,+key,first_match,kernel_positive,stride_positive,square_bound,tap_inside,output_inside,range} = agreement
  PackedProblem{+window,count,+input,+gradient,+first,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap0,output0,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  PackedKey{row,lane} = key
  +proof = initial_proof(upper,agreement,remaining,count_match,whole_room,lane_inside)
  +first_proof = proof_step(~False{},proof)
  +steps = U32.to_nat((turns_word(remaining) - 1 : U32))
  +source = numeric_source(problem)
  +tap = global_tap(upper,problem,key)
  +output = global_output(problem,key)
  +state = {ScalarChunk{source,tap,output,first,remaining,0.0} : ScalarChunk}
  +native_state = Loop.run(~ChunkModel,~chunk(~True{}),steps,chunk(~False{},proof_chunk(proof)))
  +actual = group_read(Query{upper,key},window_product(remaining,window,input,gradient))
  +first_scalar = scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),steps,proof_scalar(first_proof)))
  +whole_scalar = scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),1n+steps,state))
  +ceil_scalar = scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),Ceiling.ceiling(U32.to_nat(remaining),256n),state))
  +wanted = numeric_dot(source,tap,output,U32.to_nat(remaining),first,0.0)
  Equal.trans(F32,actual,first_scalar,wanted,
    Equal.trans(F32,actual,query_read(Query{upper,key},native_state),first_scalar,chunk_read(Query{upper,key},native_state),first_loop_read(steps,proof)),
    Equal.trans(F32,first_scalar,whole_scalar,wanted,
      Equal.cong(ScalarChunk,F32,initial => scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),steps,initial)),proof_scalar(first_proof),scalar_step(~True{},state),initial_scalar_step(upper,agreement,remaining,count_match,whole_room,lane_inside)),
      Equal.trans(F32,whole_scalar,ceil_scalar,wanted,Equal.cong(Nat,F32,turns => scalar_total(Unit{},Loop.run(~ScalarChunk,~scalar_step(~True{}),turns,state)),1n+steps,Ceiling.ceiling(U32.to_nat(remaining),256n),turns_remaining(remaining,nonempty)),scalar_full(source,tap,output,first,remaining,0.0))))

def problem_copy(problem: PackedProblem) -> Kernel.BlockCopy:
  PackedProblem{window,count,input,gradient,first,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap,output,row,column,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Kernel.BlockCopy{tap,output,taps,o}

type RowChoice<-index: Nat> is Data:
  RowChoice{upper: Bool,row: Accumulator.Row,exact: {Packing.patch_key(upper,row) == index : Nat}}

def row_choice(+index: Nat,inside: {Nat.is_lt(index,8n) == True{} : Bool}) -> RowChoice<index>:
  match index:
    case 0n: RowChoice{False{},Accumulator.R0{},{==}}
    case 1n: RowChoice{False{},Accumulator.R1{},{==}}
    case 2n: RowChoice{False{},Accumulator.R2{},{==}}
    case 3n: RowChoice{False{},Accumulator.R3{},{==}}
    case 4n: RowChoice{True{},Accumulator.R0{},{==}}
    case 5n: RowChoice{True{},Accumulator.R1{},{==}}
    case 6n: RowChoice{True{},Accumulator.R2{},{==}}
    case 7n: RowChoice{True{},Accumulator.R3{},{==}}
    case 8n+ +rest:
      Empty.absurd(RowChoice<8n+rest>,Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},
        Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Order.not_below(0n,rest,Order.zero_le(rest))),inside)))

type WindowReady is Data:
  WindowReady{limits: Machine.Limits,problem: PackedProblem,remaining: U32,
    count_match: {problem_count(problem) == U32.min(256,remaining) : U32},
    first: {Packing.window_first(problem_window(problem)) == U32.from_nat(problem_first(problem)) : U32},
    kernel: {Nat.is_lt(0n,U32.to_nat(Packing.geometry_kernel(problem_geometry(problem)))) == True{} : Bool},
    stride: {Nat.is_lt(0n,U32.to_nat(geometry_stride(problem_geometry(problem)))) == True{} : Bool},
    square: {Nat.is_le(Nat.mul(U32.to_nat(Packing.geometry_kernel(problem_geometry(problem))),U32.to_nat(Packing.geometry_kernel(problem_geometry(problem)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    room: {Nat.is_le(Nat.add(problem_first(problem),U32.to_nat(remaining)),Nat.mul(U32.to_nat(geometry_rows(problem_geometry(problem))),U32.to_nat(geometry_columns(problem_geometry(problem))))) == True{} : Bool},
    nonempty: {Nat.is_lt(0n,U32.to_nat(remaining)) == True{} : Bool}}

def ready_problem(ready: WindowReady) -> PackedProblem:
  WindowReady{limits,problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  problem

def ready_remaining(ready: WindowReady) -> U32:
  WindowReady{limits,problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  remaining

def ready_result(ready: WindowReady) -> Group:
  WindowReady{limits,problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  window_product(remaining,window,input,gradient)

def ready_agreement(+upper: Bool,+ready: WindowReady,+key: PackedKey,
  tap: {U32.is_lt(global_tap(upper,ready_problem(ready),key),Packing.geometry_taps(problem_geometry(ready_problem(ready)))) == True{} : Bool},
  output: {U32.is_lt(global_output(ready_problem(ready),key),geometry_outputs(problem_geometry(ready_problem(ready)))) == True{} : Bool}) -> NumericAgreement<upper>:
  WindowReady{+limits,+problem,+remaining,+count_match,first,kernel,stride,square,+room,nonempty} = ready
  PackedProblem{window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  NumericAgreement{limits,problem,key,first,kernel,stride,square,tap,output,
    count_room(problem_first(problem),remaining,problem_count(problem),geometry_rows(problem_geometry(problem)),geometry_columns(problem_geometry(problem)),count_match,room)}

def ready_value(+upper: Bool,+ready: WindowReady,+key: PackedKey,
  +tap: {U32.is_lt(global_tap(upper,ready_problem(ready),key),Packing.geometry_taps(problem_geometry(ready_problem(ready)))) == True{} : Bool},
  +output: {U32.is_lt(global_output(ready_problem(ready),key),geometry_outputs(problem_geometry(ready_problem(ready)))) == True{} : Bool},
  +lane: {Nat.is_lt(key_lane(key),8n) == True{} : Bool}) ->
  {group_read(Query{upper,key},ready_result(ready)) == numeric_dot(numeric_source(ready_problem(ready)),global_tap(upper,ready_problem(ready),key),global_output(ready_problem(ready),key),U32.to_nat(ready_remaining(ready)),problem_first(ready_problem(ready)),0.0) : F32}:
  WindowReady{limits,+problem,+remaining,+count_match,first,kernel,stride,square,+room,+nonempty} = ready
  PackedProblem{window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  +agreement = ready_agreement(upper,ready,key,tap,output)
  window_product_numeric(upper,agreement,remaining,count_match,room,lane,nonempty)

def ready_cell_by_row(+ready: WindowReady,+index: Nat,choice: RowChoice<Nat.div(index,8n)>,
  +inside: {Nat.is_lt(index,64n) == True{} : Bool},+lane_inside: {Nat.is_lt(Nat.mod(index,8n),8n) == True{} : Bool},
  +valid: {copy_valid(problem_copy(ready_problem(ready)),U32.from_nat(index)) == True{} : Bool}) ->
  {Tree.read(F32,group_output(ready_result(ready)),U32.from_nat(index)) ==
    numeric_dot(numeric_source(ready_problem(ready)),copy_row(problem_copy(ready_problem(ready)),U32.from_nat(index)),copy_column(problem_copy(ready_problem(ready)),U32.from_nat(index)),U32.to_nat(ready_remaining(ready)),problem_first(ready_problem(ready)),0.0) : F32}:
  WindowReady{limits,+problem,+remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,+tap0,+output0,row0,column0,start} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  RowChoice{+upper,+row,+exact} = choice
  +lane = Nat.mod(index,8n)
  +key = {PackedKey{row,lane} : PackedKey}
  +copy = {Kernel.BlockCopy{tap0,output0,taps,o} : Kernel.BlockCopy}
  +micro = micro_index(upper,row,lane)
  +same_index = {
    %Equal.sym(Nat,Packing.patch_key(upper,row),Nat.div(index,8n),exact) : {Nat.add(Nat.mul(_,8n),lane) == index : Nat}
    Equal.sym(Nat,index,Nat.add(Nat.mul(Nat.div(index,8n),8n),Nat.mod(index,8n)),DivisionOrder.quotient_exact(index,8n,{==})) : {micro == index : Nat}}
  +same_word = Equal.cong(Nat,U32,U32.from_nat,micro,index,same_index)
  +tap = global_tap(upper,problem,key)
  +output = global_output(problem,key)
  +copy_row = copy_row(copy,U32.from_nat(index))
  +copy_column = copy_column(copy,U32.from_nat(index))
  +row_equal = {
    %same_word : {tap == (tap0 + (_ / 8 : U32) : U32) : U32}
    %Equal.sym(U32,(U32.from_nat(micro) / 8 : U32),U32.from_nat(Packing.patch_key(upper,row)),micro_row(upper,row,lane,lane_inside)) : {tap == (tap0 + _ : U32) : U32}
    {==} : {tap == copy_row : U32}}
  +column_equal = {
    %same_word : {output == (output0 + (_ % 8 : U32) : U32) : U32}
    %Equal.sym(U32,(U32.from_nat(micro) % 8 : U32),U32.from_nat(lane),micro_column(upper,row,lane,lane_inside)) : {output == (output0 + _ : U32) : U32}
    {==} : {output == copy_column : U32}}
  +tap_valid = {
    %Equal.sym(U32,tap,copy_row,row_equal) : {U32.is_lt(_,taps) == True{} : Bool}
    Logic.left(U32.is_lt(copy_row,taps),U32.is_lt(copy_column,o),valid) : {U32.is_lt(tap,taps) == True{} : Bool}}
  +output_valid = {
    %Equal.sym(U32,output,copy_column,column_equal) : {U32.is_lt(_,o) == True{} : Bool}
    Logic.right(U32.is_lt(copy_row,taps),U32.is_lt(copy_column,o),valid) : {U32.is_lt(output,o) == True{} : Bool}}
  +source = group_output(ready_result(ready))
  +address_equal = Equal.trans(U32,GemmWrites.half_address(upper,row,lane),U32.from_nat(micro),U32.from_nat(index),micro_encoded(upper,row,lane,lane_inside),same_word)
  +queried = group_read(Query{upper,key},ready_result(ready))
  +wanted = numeric_dot(numeric_source(problem),tap,output,U32.to_nat(remaining),position,0.0)
  Equal.trans(F32,Tree.read(F32,source,U32.from_nat(index)),queried,numeric_dot(numeric_source(problem),copy_row,copy_column,U32.to_nat(remaining),position,0.0),
    Equal.trans(F32,Tree.read(F32,source,U32.from_nat(index)),Tree.read(F32,source,GemmWrites.half_address(upper,row,lane)),queried,
      Equal.cong(U32,F32,address => Tree.read(F32,source,address),U32.from_nat(index),GemmWrites.half_address(upper,row,lane),Equal.sym(U32,GemmWrites.half_address(upper,row,lane),U32.from_nat(index),address_equal)),
      Equal.sym(F32,queried,Tree.read(F32,source,GemmWrites.half_address(upper,row,lane)),group_read_output(Query{upper,key},ready_result(ready)))),
    Equal.trans(F32,queried,wanted,numeric_dot(numeric_source(problem),copy_row,copy_column,U32.to_nat(remaining),position,0.0),ready_value(upper,ready,key,tap_valid,output_valid,lane_inside),
      %row_equal : {wanted == numeric_dot(numeric_source(problem),_,copy_column,U32.to_nat(remaining),position,0.0) : F32}
      %column_equal : {wanted == numeric_dot(numeric_source(problem),tap,_,U32.to_nat(remaining),position,0.0) : F32}
      {==}))

def ready_cell_value(+ready: WindowReady,+index: Nat,+inside: {Nat.is_lt(index,64n) == True{} : Bool},
  +valid: {copy_valid(problem_copy(ready_problem(ready)),U32.from_nat(index)) == True{} : Bool}) ->
  {Tree.read(F32,group_output(ready_result(ready)),U32.from_nat(index)) ==
    numeric_dot(numeric_source(ready_problem(ready)),copy_row(problem_copy(ready_problem(ready)),U32.from_nat(index)),copy_column(problem_copy(ready_problem(ready)),U32.from_nat(index)),U32.to_nat(ready_remaining(ready)),problem_first(ready_problem(ready)),0.0) : F32}:
  ready_cell_by_row(ready,index,row_choice(Nat.div(index,8n),DivisionOrder.quotient_below(index,8n,8n,{==},inside)),inside,DivisionOrder.remainder_small(index,8n,{==}),valid)

def reference_copy(ready: WindowReady,row: U32,column: U32) -> Kernel.BlockCopy:
  WindowReady{limits,problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Kernel.BlockCopy{row,column,taps,o}

type TileRead is Data:
  TileRead{ready: WindowReady,row: U32,column: U32,depth: Nat,capacity: U32,
    depth_limit: {Nat.is_le(depth,30n) == True{} : Bool},extent: CopyExtent(problem_copy(ready_problem(ready)),capacity),
    target_valid: {copy_valid(reference_copy(ready,row,column),0) == True{} : Bool}}

def tile_ready(context: TileRead) -> WindowReady:
  TileRead{ready,row,column,depth,capacity,depth_limit,extent,target_valid} = context
  ready

def tile_copy(context: TileRead) -> Kernel.BlockCopy:
  TileRead{ready,row,column,depth,capacity,depth_limit,extent,target_valid} = context
  problem_copy(ready_problem(ready))

def tile_reference(context: TileRead) -> Kernel.BlockCopy:
  TileRead{ready,row,column,depth,capacity,depth_limit,extent,target_valid} = context
  reference_copy(ready,row,column)

def tile_query(context: TileRead) -> U32:
  copy_address(tile_reference(context),0)

def tile_wanted(+context: TileRead) -> F32:
  +ready = tile_ready(context)
  numeric_dot(numeric_source(ready_problem(ready)),copy_row(tile_reference(context),0),copy_column(tile_reference(context),0),U32.to_nat(ready_remaining(ready)),problem_first(ready_problem(ready)),0.0)

def tile_capacity(context: TileRead) -> U32:
  TileRead{ready,row,column,depth,capacity,depth_limit,extent,target_valid} = context
  capacity

def tile_reference_extent(+context: TileRead) -> CopyExtent(tile_reference(context),tile_capacity(context)):
  TileRead{+ready,row,column,depth,capacity,depth_limit,extent,target_valid} = context
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  extent

def tile_query_inside(+context: TileRead) -> {U32.is_lt(tile_query(context),tile_capacity(context)) == True{} : Bool}:
  TileRead{+ready,+row,+column,depth,+capacity,depth_limit,extent,+target_valid} = context
  copy_written_inside(reference_copy(ready,row,column),0,capacity,tile_reference_extent(context),target_valid)

def tile_cell_matches(+context: TileRead,+index: Nat,+inside: {Nat.is_lt(index,64n) == True{} : Bool},
  +valid: {copy_valid(tile_copy(context),U32.from_nat(index)) == True{} : Bool},
  +compared: {U32.is_eq(copy_address(tile_copy(context),U32.from_nat(index)),tile_query(context)) == True{} : Bool}) ->
  {Tree.read(F32,group_output(ready_result(tile_ready(context))),U32.from_nat(index)) == tile_wanted(context) : F32}:
  TileRead{+ready,+target_row,+target_column,depth,+capacity,depth_limit,+extent,+target_valid} = context
  WindowReady{limits,+problem,+remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,+position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,+tap0,+output0,row0,column0,start} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  +copy = {Kernel.BlockCopy{tap0,output0,taps,o} : Kernel.BlockCopy}
  +reference = {Kernel.BlockCopy{target_row,target_column,taps,o} : Kernel.BlockCopy}
  +word = U32.from_nat(index)
  +a = copy_row(copy,word)
  +b = copy_row(reference,0)
  +x = copy_column(copy,word)
  +y = copy_column(reference,0)
  +hx = {%Comparison.lt(x,o) : {_ == True{} : Bool} Logic.right(U32.is_lt(a,taps),U32.is_lt(x,o),valid) : {Nat.is_lt(U32.to_nat(x),U32.to_nat(o)) == True{} : Bool}}
  +hy = {%Comparison.lt(y,o) : {_ == True{} : Bool} Logic.right(U32.is_lt(b,taps),U32.is_lt(y,o),target_valid) : {Nat.is_lt(U32.to_nat(y),U32.to_nat(o)) == True{} : Bool}}
  +positive_width = Order.le_lt_trans(0n,U32.to_nat(y),U32.to_nat(o),Order.zero_le(U32.to_nat(y)),hy)
  +same = Arithmetic.equal_of_is_eq(copy_address(copy,word),copy_address(reference,0),compared)
  +linear = {
    %copy_address_exact(copy,word,capacity,extent,valid) : {_ == copy_linear(reference,0) : Nat}
    %copy_address_exact(reference,0,capacity,tile_reference_extent(context),target_valid) : {U32.to_nat(copy_address(copy,word)) == _ : Nat}
    Equal.cong(U32,Nat,U32.to_nat,copy_address(copy,word),copy_address(reference,0),same) : {copy_linear(copy,word) == copy_linear(reference,0) : Nat}}
  +row_equal = Arithmetic.to_nat_injective(a,b,Radix.row_unique(U32.to_nat(a),U32.to_nat(b),U32.to_nat(o),U32.to_nat(x),U32.to_nat(y),positive_width,hx,hy,linear))
  +column_equal = Arithmetic.to_nat_injective(x,y,Radix.col_unique(U32.to_nat(a),U32.to_nat(b),U32.to_nat(o),U32.to_nat(x),U32.to_nat(y),positive_width,hx,hy,linear))
  +sample = Tree.read(F32,group_output(ready_result(ready)),word)
  +wanted = numeric_dot(numeric_source(problem),a,x,U32.to_nat(remaining),position,0.0)
  Equal.trans(F32,sample,wanted,tile_wanted(context),ready_cell_value(ready,index,inside,valid),
    %row_equal : {wanted == numeric_dot(numeric_source(problem),_,y,U32.to_nat(remaining),position,0.0) : F32}
    %column_equal : {wanted == numeric_dot(numeric_source(problem),a,_,U32.to_nat(remaining),position,0.0) : F32}
    {==})

def tile_depth(context: TileRead) -> Nat:
  TileRead{ready,row,column,depth,capacity,depth_limit,extent,target_valid} = context
  depth

def tile_source(context: TileRead) -> Tree.Tree<F32>:
  group_output(ready_result(tile_ready(context)))

def tile_shape(+context: TileRead) -> CopyShape:
  CopyShape{tile_copy(context),tile_depth(context),tile_capacity(context),tile_source(context)}

def TileFrame(+context: TileRead,state: CopyOwners) -> Data:
  CopyShapeFrame(tile_shape(context),state)

def TilePost(+context: TileRead,+state: CopyOwners) -> Data:
  Shape.Both<TileFrame(context,state),{Tree.read(F32,copy_output(state),tile_query(context)) == tile_wanted(context) : F32}>

def tile_limit(context: TileRead) -> Nat:
  64n

def TileInvariant(context: TileRead,index: Nat,state: CopyOwners) -> Data:
  TilePost(context,state)

def tile_update(index: U32,context: TileRead,state: CopyOwners) -> CopyOwners:
  copy_cell(index,tile_copy(context),state)

def tile_frame_step(+context: TileRead,+index: U32,+state: CopyOwners,frame: TileFrame(context,state)) -> TileFrame(context,tile_update(index,context,state)):
  copy_shape_frame_step(tile_shape(context),index,state,frame)

def tile_written_inside(+context: TileRead,+index: Nat,+state: CopyOwners,+frame: TileFrame(context,state),
  enabled: Bool,+valid: {copy_valid(tile_copy(context),U32.from_nat(index)) == enabled : Bool}) ->
  Logic.When(enabled,{U32.is_lt(copy_address(tile_copy(context),U32.from_nat(index)),Tree.capacity(F32,copy_output(state))) == True{} : Bool}):
  TileRead{+ready,row,column,depth,+capacity,depth_limit,+extent,target_valid} = context
  CopyOwners{input,+output} = state
  Shape.Both{balanced,Shape.Both{+capacity_equal,source_equal}} = frame
  match enabled:
    case False{}: Unit{}
    case True{}:
      %Equal.sym(U32,Tree.capacity(F32,output),capacity,capacity_equal) :
        {U32.is_lt(copy_address(problem_copy(ready_problem(ready)),U32.from_nat(index)),_) == True{} : Bool}
      copy_written_inside(problem_copy(ready_problem(ready)),U32.from_nat(index),capacity,extent,valid)

def tile_matching_value(+context: TileRead,+index: Nat,+state: CopyOwners,+frame: TileFrame(context,state),
  +inside: {Nat.is_lt(index,64n) == True{} : Bool},enabled: Bool,equal: Bool,
  +valid: {copy_valid(tile_copy(context),U32.from_nat(index)) == enabled : Bool},
  +compared: {U32.is_eq(copy_address(tile_copy(context),U32.from_nat(index)),tile_query(context)) == equal : Bool}) ->
  Logic.When(enabled && equal,{Tree.read(F32,copy_source(state),U32.from_nat(index)) == tile_wanted(context) : F32}):
  CopyOwners{+input,output} = state
  Shape.Both{balanced,Shape.Both{capacity_equal,+source_equal}} = frame
  match enabled equal:
    case False{} equal: Unit{}
    case True{} False{}: Unit{}
    case True{} True{}:
      %Equal.sym(Tree.Tree<F32>,input,tile_source(context),source_equal) :
        {Tree.read(F32,_,U32.from_nat(index)) == tile_wanted(context) : F32}
      tile_cell_matches(context,index,inside,valid,compared)

def tile_post_step(+context: TileRead,+index: Nat,+state: CopyOwners,
  inside: {Nat.is_lt(index,tile_limit(context)) == True{} : Bool},post: TileInvariant(context,index,state)) ->
  TileInvariant(context,1n+index,tile_update(U32.from_nat(index),context,state)):
  TileRead{+ready,row,column,+depth,+capacity,+depth_limit,extent,target_valid} = context
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  CopyOwners{+source,+destination} = state
  Shape.Both{+frame,+previous} = post
  Shape.Both{+balanced,Shape.Both{+capacity_equal,source_equal}} = frame
  +word = U32.from_nat(index)
  +copy = tile_copy(context)
  +valid = copy_valid(copy,word)
  +address = copy_address(copy,word)
  +query = tile_query(context)
  +value = Tree.read(F32,source,word)
  +wanted = tile_wanted(context)
  +equal = U32.is_eq(address,query)
  Shape.Both{tile_frame_step(context,word,state,frame),
    %Equal.sym(Tree.Tree<F32>,copy_output(copy_cell(word,copy,state)),copy_write(valid,address,destination,value),copy_output_cell(word,copy,state)) :
      {Tree.read(F32,_,query) == wanted : F32}
    copy_matching(valid,address,query,destination,value,wanted,depth,balanced,depth_limit,
      tile_written_inside(context,index,state,frame,valid,{==}),
      %Equal.sym(U32,Tree.capacity(F32,destination),capacity,capacity_equal) : {U32.is_lt(query,_) == True{} : Bool} tile_query_inside(context),
      tile_matching_value(context,index,state,frame,inside,valid,equal,{==},{==}),previous)}

def tile_update_inner(+index: U32,+context: TileRead,+state: CopyOwners) ->
  {tile_update(index,context,state) == copy_cell(index,tile_copy(context),state) : CopyOwners}:
  {==}

def tile_preserves_value(+context: TileRead,+state: CopyOwners,post: TilePost(context,state)) ->
  TilePost(context,Traversal.indexed_fold(~CopyOwners,~Kernel.BlockCopy,~copy_cell,64n,0,tile_copy(context),state)):
  %Traversals.indexed_fold_change_context(~CopyOwners,~TileRead,~Kernel.BlockCopy,~tile_copy,~tile_update,~copy_cell,~tile_update_inner,64n,0,context,state) : TilePost(context,_)
  Traversals.indexed_fold_preserves_invariant(~CopyOwners,~TileRead,~tile_limit,~TileInvariant,~tile_update,~tile_post_step,64n,0n,context,state,{==},post)

def tile_initial_post(+context: TileRead,+output: Tree.Tree<F32>,
  frame: StorageFrame(tile_depth(context),tile_capacity(context),output),
  previous: {Tree.read(F32,output,tile_query(context)) == tile_wanted(context) : F32}) ->
  TilePost(context,CopyOwners{tile_source(context),output}):
  TileRead{ready,row,column,depth,capacity,depth_limit,extent,target_valid} = context
  Shape.Both{balanced,capacity_equal} = frame
  Shape.Both{Shape.Both{balanced,Shape.Both{capacity_equal,{==}}},previous}

def tile_post_read(+context: TileRead,+state: CopyOwners,post: TilePost(context,state)) ->
  {Tree.read(F32,copy_output(state),tile_query(context)) == tile_wanted(context) : F32}:
  Shape.Both{frame,readback} = post
  readback

def block_ready(+ready: WindowReady,+output_blocks: U32,+index: U32) -> WindowReady:
  WindowReady{+limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap,output,row,column,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  WindowReady{limits,PackedProblem{Kernel.WeightWindow{geometry,image,(index / output_blocks * 8 : U32),(index % output_blocks * 8 : U32),row,column,start},
    count,input,gradient,position,positive,row_match,column_match,bounded},remaining,count_match,first,kernel,stride,square,room,nonempty}

def ready_blocks(+ready: WindowReady,output_blocks: U32) -> Kernel.GroupBlocks:
  WindowReady{limits,problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.GroupBlocks{window,remaining,output_blocks}

def block_tile(+ready: WindowReady,+output_blocks: U32,+index: U32,+row: U32,+column: U32,+depth: Nat,+capacity: U32,
  depth_limit: {Nat.is_le(depth,30n) == True{} : Bool},extent: CopyExtent(problem_copy(ready_problem(ready)),capacity),
  target_valid: {copy_valid(reference_copy(ready,row,column),0) == True{} : Bool}) -> TileRead:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  TileRead{block_ready(ready,output_blocks,index),row,column,depth,capacity,depth_limit,extent,target_valid}

def block_query(+ready: WindowReady,row: U32,column: U32) -> U32:
  copy_address(reference_copy(ready,row,column),0)

def block_wanted(+ready: WindowReady,+row: U32,+column: U32) -> F32:
  +reference = reference_copy(ready,row,column)
  numeric_dot(numeric_source(ready_problem(ready)),copy_row(reference,0),copy_column(reference,0),U32.to_nat(ready_remaining(ready)),problem_first(ready_problem(ready)),0.0)

def block_tile_query(+ready: WindowReady,+output_blocks: U32,+index: U32,+row: U32,+column: U32,+depth: Nat,+capacity: U32,
  +depth_limit: {Nat.is_le(depth,30n) == True{} : Bool},+extent: CopyExtent(problem_copy(ready_problem(ready)),capacity),
  +target_valid: {copy_valid(reference_copy(ready,row,column),0) == True{} : Bool}) ->
  {tile_query(block_tile(ready,output_blocks,index,row,column,depth,capacity,depth_limit,extent,target_valid)) == block_query(ready,row,column) : U32}:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  {==}

def block_tile_wanted(+ready: WindowReady,+output_blocks: U32,+index: U32,+row: U32,+column: U32,+depth: Nat,+capacity: U32,
  +depth_limit: {Nat.is_le(depth,30n) == True{} : Bool},+extent: CopyExtent(problem_copy(ready_problem(ready)),capacity),
  +target_valid: {copy_valid(reference_copy(ready,row,column),0) == True{} : Bool}) ->
  {tile_wanted(block_tile(ready,output_blocks,index,row,column,depth,capacity,depth_limit,extent,target_valid)) == block_wanted(ready,row,column) : F32}:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  {==}

def sources_input(sources: Sources) -> Buffer.Model<F32>:
  Sources{input,gradient} = sources
  input

def sources_gradient(sources: Sources) -> Buffer.Model<F32>:
  Sources{input,gradient} = sources
  gradient

def group_sources_replace(+state: Group,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>,
  +same: {group_sources(state) == Sources{input,gradient} : Sources}) -> {state == Group{input,gradient,group_output(state)} : Group}:
  Group{+old_input,+old_gradient,+values} = state
  +input_equal = Equal.cong(Sources,Buffer.Model<F32>,sources_input,Sources{old_input,old_gradient},Sources{input,gradient},same)
  +gradient_equal = Equal.cong(Sources,Buffer.Model<F32>,sources_gradient,Sources{old_input,old_gradient},Sources{input,gradient},same)
  %Equal.sym(Buffer.Model<F32>,old_input,input,input_equal) : {Group{_,old_gradient,values} == Group{input,gradient,values} : Group}
  %Equal.sym(Buffer.Model<F32>,old_gradient,gradient,gradient_equal) : {Group{input,_,values} == Group{input,gradient,values} : Group}
  {==}

def block_relocated(+ready: WindowReady,+output_blocks: U32,+index: U32,+output: Tree.Tree<F32>) ->
  {block(index,ready_blocks(ready,output_blocks),Group{problem_input(ready_problem(ready)),problem_gradient(ready_problem(ready)),output}) ==
    block_result(problem_copy(ready_problem(block_ready(ready,output_blocks,index))),output,ready_result(block_ready(ready,output_blocks,index))) : Group}:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output0,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  {==}

def block_result_copy(+copy: Kernel.BlockCopy,+output: Tree.Tree<F32>,+result: Group) ->
  {group_output(block_result(copy,output,result)) ==
    copy_output(Traversal.indexed_fold(~CopyOwners,~Kernel.BlockCopy,~copy_cell,64n,0,copy,CopyOwners{group_output(result),output})) : Tree.Tree<F32>}:
  Group{input,gradient,values} = result
  {==}

def block_preserves_value(+ready: WindowReady,+output_blocks: U32,+index: U32,+row: U32,+column: U32,+output: Tree.Tree<F32>,+depth: Nat,+capacity: U32,
  +depth_limit: {Nat.is_le(depth,30n) == True{} : Bool},+extent: CopyExtent(problem_copy(ready_problem(ready)),capacity),
  +target_valid: {copy_valid(reference_copy(ready,row,column),0) == True{} : Bool},
  frame: StorageFrame(depth,capacity,output),previous: {Tree.read(F32,output,block_query(ready,row,column)) == block_wanted(ready,row,column) : F32}) ->
  {Tree.read(F32,group_output(block(index,ready_blocks(ready,output_blocks),Group{problem_input(ready_problem(ready)),problem_gradient(ready_problem(ready)),output})),block_query(ready,row,column)) == block_wanted(ready,row,column) : F32}:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output0,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  +shifted = block_ready(ready,output_blocks,index)
  +context = block_tile(ready,output_blocks,index,row,column,depth,capacity,depth_limit,extent,target_valid)
  +copy = problem_copy(ready_problem(shifted))
  +result = ready_result(shifted)
  +initial = {CopyOwners{group_output(result),output} : CopyOwners}
  +state = Traversal.indexed_fold(~CopyOwners,~Kernel.BlockCopy,~copy_cell,64n,0,copy,initial)
  +query = block_query(ready,row,column)
  +wanted = block_wanted(ready,row,column)
  +query_equal = block_tile_query(ready,output_blocks,index,row,column,depth,capacity,depth_limit,extent,target_valid)
  +wanted_equal = block_tile_wanted(ready,output_blocks,index,row,column,depth,capacity,depth_limit,extent,target_valid)
  +post = tile_initial_post(context,output,frame,
    %Equal.sym(U32,tile_query(context),query,query_equal) : {Tree.read(F32,output,_) == tile_wanted(context) : F32}
    %Equal.sym(F32,tile_wanted(context),wanted,wanted_equal) : {Tree.read(F32,output,query) == _ : F32}
    previous)
  %Equal.sym(Group,block(index,ready_blocks(ready,output_blocks),Group{input,gradient,output}),block_result(copy,output,result),block_relocated(ready,output_blocks,index,output)) :
    {Tree.read(F32,group_output(_),query) == wanted : F32}
  %Equal.sym(Tree.Tree<F32>,group_output(block_result(copy,output,result)),copy_output(state),block_result_copy(copy,output,result)) :
    {Tree.read(F32,_,query) == wanted : F32}
  %query_equal : {Tree.read(F32,copy_output(state),_) == wanted : F32}
  %wanted_equal : {Tree.read(F32,copy_output(state),tile_query(context)) == _ : F32}
  tile_post_read(context,state,tile_preserves_value(context,initial,post))

type BlockRead is Data:
  BlockRead{ready: WindowReady,output_blocks: U32,row: U32,column: U32,depth: Nat,capacity: U32,jobs: Nat,target: Nat,cell: Nat,
    depth_limit: {Nat.is_le(depth,30n) == True{} : Bool},extent: CopyExtent(problem_copy(ready_problem(ready)),capacity),
    query_valid: {copy_valid(reference_copy(ready,row,column),0) == True{} : Bool},
    target_inside: {Nat.is_lt(target,jobs) == True{} : Bool},cell_inside: {Nat.is_lt(cell,64n) == True{} : Bool},
    cell_valid: {copy_valid(problem_copy(ready_problem(block_ready(ready,output_blocks,U32.from_nat(target)))),U32.from_nat(cell)) == True{} : Bool},
    cell_address: {copy_address(problem_copy(ready_problem(block_ready(ready,output_blocks,U32.from_nat(target)))),U32.from_nat(cell)) == block_query(ready,row,column) : U32}}

def block_read_meta(context: BlockRead) -> BlockShape:
  BlockRead{+ready,output_blocks,row,column,depth,capacity,jobs,target,cell,depth_limit,extent,query_valid,target_inside,cell_inside,cell_valid,cell_address} = context
  BlockShape{ready_blocks(ready,output_blocks),depth,capacity,problem_input(ready_problem(ready)),problem_gradient(ready_problem(ready)),jobs}

def block_read_limit(context: BlockRead) -> Nat:
  BlockRead{ready,output_blocks,row,column,depth,capacity,jobs,target,cell,depth_limit,extent,query_valid,target_inside,cell_inside,cell_valid,cell_address} = context
  jobs

def block_read_target(context: BlockRead) -> Nat:
  BlockRead{ready,output_blocks,row,column,depth,capacity,jobs,target,cell,depth_limit,extent,query_valid,target_inside,cell_inside,cell_valid,cell_address} = context
  target

def block_read_context(context: BlockRead) -> Kernel.GroupBlocks:
  BlockRead{ready,output_blocks,row,column,depth,capacity,jobs,target,cell,depth_limit,extent,query_valid,target_inside,cell_inside,cell_valid,cell_address} = context
  ready_blocks(ready,output_blocks)

def block_read_query(context: BlockRead) -> U32:
  BlockRead{ready,output_blocks,row,column,depth,capacity,jobs,target,cell,depth_limit,extent,query_valid,target_inside,cell_inside,cell_valid,cell_address} = context
  block_query(ready,row,column)

def block_read_wanted(context: BlockRead) -> F32:
  BlockRead{ready,output_blocks,row,column,depth,capacity,jobs,target,cell,depth_limit,extent,query_valid,target_inside,cell_inside,cell_valid,cell_address} = context
  block_wanted(ready,row,column)

def BlockReadFrame(context: BlockRead,state: Group) -> Data:
  BlockFrame(block_read_meta(context),state)

def BlockReadPost(context: BlockRead,state: Group) -> Data:
  +context = context
  +state = state
  Shape.Both<BlockReadFrame(context,state),{Tree.read(F32,group_output(state),block_read_query(context)) == block_read_wanted(context) : F32}>

def BlockReadInvariant(context: BlockRead,index: Nat,state: Group) -> Data:
  BlockReadPost(context,state)

def block_read_update(index: U32,context: BlockRead,state: Group) -> Group:
  block(index,block_read_context(context),state)

def block_read_frame_step(+context: BlockRead,+index: U32,+state: Group,frame: BlockReadFrame(context,state)) ->
  BlockReadFrame(context,block_read_update(index,context,state)):
  BlockRead{ready,output_blocks,row,column,depth,capacity,jobs,target,cell,depth_limit,extent,query_valid,target_inside,cell_inside,cell_valid,cell_address} = context
  block_shape_frame_step(block_read_meta(context),index,state,frame)

def block_read_post_step(+context: BlockRead,+index: Nat,+state: Group,
  inside: {Nat.is_lt(index,block_read_limit(context)) == True{} : Bool},post: BlockReadInvariant(context,index,state)) ->
  BlockReadInvariant(context,1n+index,block_read_update(U32.from_nat(index),context,state)):
  BlockRead{+ready,+output_blocks,+row,+column,+depth,+capacity,jobs,target,cell,+depth_limit,+extent,+query_valid,target_inside,cell_inside,cell_valid,cell_address} = context
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,+input,+gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Group{old_input,old_gradient,+values} = state
  Shape.Both{Shape.Both{+storage,+sources},previous} = post
  +word = U32.from_nat(index)
  +next_read = {
    %Equal.sym(Group,state,Group{input,gradient,values},group_sources_replace(state,input,gradient,sources)) :
      {Tree.read(F32,group_output(block(word,ready_blocks(ready,output_blocks),_)),block_query(ready,row,column)) == block_wanted(ready,row,column) : F32}
    block_preserves_value(ready,output_blocks,word,row,column,values,depth,capacity,depth_limit,extent,query_valid,storage,previous) :
      {Tree.read(F32,group_output(block_read_update(word,context,state)),block_read_query(context)) == block_read_wanted(context) : F32}}
  Shape.Both{block_read_frame_step(context,word,state,Shape.Both{storage,sources}),next_read}

def block_target_copy_query(+context: BlockRead,+source: Tree.Tree<F32>) -> CopyQuery:
  BlockRead{+ready,+output_blocks,+row,+column,+depth,+capacity,jobs,+target,+cell,+depth_limit,+extent,+query_valid,target_inside,+cell_inside,+cell_valid,+cell_address} = context
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  +word = U32.from_nat(target)
  +tile = block_tile(ready,output_blocks,word,row,column,depth,capacity,depth_limit,extent,query_valid)
  +copy = problem_copy(ready_problem(block_ready(ready,output_blocks,word)))
  +query = block_query(ready,row,column)
  +reference = reference_copy(ready,row,column)
  +reference_column = copy_column(reference,0)
  +reference_row = copy_row(reference,0)
  +positive_width = Order.le_lt_trans(0n,U32.to_nat(reference_column),U32.to_nat(o),Order.zero_le(U32.to_nat(reference_column)),
    %Comparison.lt(reference_column,o) : {_ == True{} : Bool}
    Logic.right(U32.is_lt(reference_row,taps),U32.is_lt(reference_column,o),query_valid))
  CopyQuery{copy,cell,query,source,depth,capacity,Order.successor(depth,30n,depth_limit),
    %block_tile_query(ready,output_blocks,word,row,column,depth,capacity,depth_limit,extent,query_valid) : {U32.is_lt(_,capacity) == True{} : Bool} tile_query_inside(tile),
    cell_inside,cell_valid,cell_address,extent,positive_width}

def block_read_at_target(+context: BlockRead,+state: Group,frame: BlockReadFrame(context,state)) ->
  BlockReadPost(context,block_read_update(U32.from_nat(block_read_target(context)),context,state)):
  BlockRead{+ready,+output_blocks,+row,+column,+depth,+capacity,jobs,+target,+cell,+depth_limit,+extent,+query_valid,target_inside,+cell_inside,+cell_valid,+cell_address} = context
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,+input,+gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Group{old_input,old_gradient,+values} = state
  Shape.Both{Shape.Both{+balanced,+capacity_equal},+sources} = frame
  +word = U32.from_nat(target)
  +shifted = block_ready(ready,output_blocks,word)
  +copy = problem_copy(ready_problem(shifted))
  +result = ready_result(shifted)
  +source = group_output(result)
  +tile = block_tile(ready,output_blocks,word,row,column,depth,capacity,depth_limit,extent,query_valid)
  +copy_query = block_target_copy_query(context,source)
  +copied = copy_output(Traversal.indexed_fold(~CopyOwners,~Kernel.BlockCopy,~copy_cell,64n,0,copy,CopyOwners{source,values}))
  +query = block_query(ready,row,column)
  +wanted = block_wanted(ready,row,column)
  +micro = Tree.read(F32,source,U32.from_nat(cell))
  +value_matches = {
    %block_tile_wanted(ready,output_blocks,word,row,column,depth,capacity,depth_limit,extent,query_valid) : {micro == _ : F32}
    tile_cell_matches(tile,cell,cell_inside,cell_valid,
      %Equal.sym(U32,tile_query(tile),query,block_tile_query(ready,output_blocks,word,row,column,depth,capacity,depth_limit,extent,query_valid)) :
        {U32.is_eq(copy_address(copy,U32.from_nat(cell)),_) == True{} : Bool}
      Encoded.equal_words(copy_address(copy,U32.from_nat(cell)),query,cell_address)) : {micro == wanted : F32}}
  +readback = {
    %Equal.sym(Group,state,Group{input,gradient,values},group_sources_replace(state,input,gradient,sources)) :
      {Tree.read(F32,group_output(block(word,ready_blocks(ready,output_blocks),_)),query) == wanted : F32}
    %Equal.sym(Group,block(word,ready_blocks(ready,output_blocks),Group{input,gradient,values}),block_result(copy,values,result),block_relocated(ready,output_blocks,word,values)) :
      {Tree.read(F32,group_output(_),query) == wanted : F32}
    %Equal.sym(Tree.Tree<F32>,group_output(block_result(copy,values,result)),copied,block_result_copy(copy,values,result)) :
      {Tree.read(F32,_,query) == wanted : F32}
    Equal.trans(F32,Tree.read(F32,copied,query),micro,wanted,copy_fold_readback(copy_query,values,balanced,capacity_equal),value_matches) :
      {Tree.read(F32,group_output(block_read_update(word,context,state)),block_read_query(context)) == block_read_wanted(context) : F32}}
  Shape.Both{block_read_frame_step(context,word,state,Shape.Both{Shape.Both{balanced,capacity_equal},sources}),readback}

def block_read_all(+context: BlockRead,+state: Group,frame: BlockReadFrame(context,state)) ->
  BlockReadPost(context,Traversal.indexed_fold(~Group,~BlockRead,~block_read_update,block_read_limit(context),0,context,state)):
  BlockRead{ready,output_blocks,row,column,depth,capacity,jobs,target,cell,depth_limit,extent,query_valid,+target_inside,cell_inside,cell_valid,cell_address} = context
  IndexedReadback.complete_postcondition(~Group,~BlockRead,~block_read_limit,~block_read_target,~BlockReadFrame,~BlockReadPost,~block_read_update,
    ~block_read_frame_step,~block_read_at_target,~block_read_post_step,context,state,frame,target_inside)

def block_read_post_read(+context: BlockRead,+state: Group,post: BlockReadPost(context,state)) ->
  {Tree.read(F32,group_output(state),block_read_query(context)) == block_read_wanted(context) : F32}:
  Shape.Both{frame,readback} = post
  readback

def block_read_update_inner(+index: U32,+context: BlockRead,+state: Group) ->
  {block_read_update(index,context,state) == block(index,block_read_context(context),state) : Group}:
  {==}

def block_readback(+context: BlockRead,+state: Group,frame: BlockReadFrame(context,state)) ->
  {Tree.read(F32,group_output(Traversal.indexed_fold(~Group,~Kernel.GroupBlocks,~block,block_read_limit(context),0,block_read_context(context),state)),block_read_query(context)) == block_read_wanted(context) : F32}:
  %Traversals.indexed_fold_change_context(~Group,~BlockRead,~Kernel.GroupBlocks,~block_read_context,~block_read_update,~block,~block_read_update_inner,block_read_limit(context),0,context,state) :
    {Tree.read(F32,group_output(_),block_read_query(context)) == block_read_wanted(context) : F32}
  block_read_post_read(context,Traversal.indexed_fold(~Group,~BlockRead,~block_read_update,block_read_limit(context),0,context,state),block_read_all(context,state,frame))

type GroupPlan is Data:
  GroupPlan{limits: Machine.Limits,geometry: Kernel.Geometry,
    taps_positive: {Nat.is_lt(0n,U32.to_nat(Packing.geometry_taps(geometry))) == True{} : Bool},
    outputs_positive: {Nat.is_lt(0n,U32.to_nat(geometry_outputs(geometry))) == True{} : Bool},
    bound: {Nat.is_le(Nat.mul(U32.to_nat(Packing.geometry_taps(geometry)),U32.to_nat(geometry_outputs(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    aligned: {U32.is_zero((Machine.capacity(limits) % 8 : U32)) == True{} : Bool}}

def plan_taps(plan: GroupPlan) -> U32:
  GroupPlan{limits,geometry,taps_positive,outputs_positive,bound,aligned} = plan
  Packing.geometry_taps(geometry)

def plan_outputs(plan: GroupPlan) -> U32:
  GroupPlan{limits,geometry,taps_positive,outputs_positive,bound,aligned} = plan
  geometry_outputs(geometry)

def plan_limits(plan: GroupPlan) -> Machine.Limits:
  GroupPlan{limits,geometry,taps_positive,outputs_positive,bound,aligned} = plan
  limits

def plan_count(+plan: GroupPlan) -> U32:
  (plan_taps(plan) * plan_outputs(plan) : U32)

def plan_output_blocks(plan: GroupPlan) -> U32:
  ((plan_outputs(plan) + 7) / 8 : U32)

def plan_tap_blocks(plan: GroupPlan) -> U32:
  ((plan_taps(plan) + 7) / 8 : U32)

def plan_jobs(+plan: GroupPlan) -> U32:
  (plan_tap_blocks(plan) * plan_output_blocks(plan) : U32)

def plan_jobs_nat(+plan: GroupPlan) -> Nat:
  Nat.mul(Ceiling.ceiling(U32.to_nat(plan_taps(plan)),8n),Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n))

def plan_taps_room(+plan: GroupPlan) -> {Nat.is_le(U32.to_nat(plan_taps(plan)),U32.to_nat(Machine.capacity(plan_limits(plan)))) == True{} : Bool}:
  GroupPlan{+limits,+geometry,taps_positive,+outputs_positive,+bound,aligned} = plan
  +taps = U32.to_nat(Packing.geometry_taps(geometry))
  +outputs = U32.to_nat(geometry_outputs(geometry))
  Order.transitive(taps,Nat.mul(taps,outputs),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(taps,outputs,outputs_positive),bound)

def plan_outputs_room(+plan: GroupPlan) -> {Nat.is_le(U32.to_nat(plan_outputs(plan)),U32.to_nat(Machine.capacity(plan_limits(plan)))) == True{} : Bool}:
  GroupPlan{+limits,+geometry,+taps_positive,outputs_positive,+bound,aligned} = plan
  +taps = U32.to_nat(Packing.geometry_taps(geometry))
  +outputs = U32.to_nat(geometry_outputs(geometry))
  Order.transitive(outputs,Nat.mul(taps,outputs),U32.to_nat(Machine.capacity(limits)),
    %Equal.sym(Nat,Nat.mul(taps,outputs),Nat.mul(outputs,taps),RowMajor.multiply_commutative(taps,outputs)) : {Nat.is_le(outputs,_) == True{} : Bool} RowMajor.product_covers(outputs,taps,taps_positive),bound)

def plan_count_exact(+plan: GroupPlan) -> {U32.to_nat(plan_count(plan)) == Nat.mul(U32.to_nat(plan_taps(plan)),U32.to_nat(plan_outputs(plan))) : Nat}:
  GroupPlan{+limits,+geometry,taps_positive,outputs_positive,+bound,aligned} = plan
  Arithmetic.multiply(Packing.geometry_taps(geometry),geometry_outputs(geometry),Machine.capacity(limits),Limits.capacity_fits(limits),bound)

def plan_count_room(+plan: GroupPlan) -> {Nat.is_le(U32.to_nat(plan_count(plan)),U32.to_nat(Machine.capacity(plan_limits(plan)))) == True{} : Bool}:
  GroupPlan{limits,geometry,taps_positive,outputs_positive,bound,aligned} = plan
  %Equal.sym(Nat,U32.to_nat(plan_count(plan)),Nat.mul(U32.to_nat(plan_taps(plan)),U32.to_nat(plan_outputs(plan))),plan_count_exact(plan)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  bound

def plan_jobs_room(+plan: GroupPlan) -> {Nat.is_le(plan_jobs_nat(plan),Nat.mul(U32.to_nat(plan_taps(plan)),U32.to_nat(plan_outputs(plan)))) == True{} : Bool}:
  +taps = U32.to_nat(plan_taps(plan))
  +outputs = U32.to_nat(plan_outputs(plan))
  +tap_blocks = Ceiling.ceiling(taps,8n)
  +output_blocks = Ceiling.ceiling(outputs,8n)
  Order.transitive(Nat.mul(tap_blocks,output_blocks),Nat.mul(taps,output_blocks),Nat.mul(taps,outputs),
    RowMajor.mul_le(tap_blocks,taps,output_blocks,Ceiling.bounded(taps,8n,{==})),
    %RowMajor.multiply_commutative(output_blocks,taps) : {Nat.is_le(_,Nat.mul(taps,outputs)) == True{} : Bool}
    %RowMajor.multiply_commutative(outputs,taps) : {Nat.is_le(Nat.mul(output_blocks,taps),_) == True{} : Bool}
    RowMajor.mul_le(output_blocks,outputs,taps,Ceiling.bounded(outputs,8n,{==})))

def plan_jobs_exact(+plan: GroupPlan) -> {U32.to_nat(plan_jobs(plan)) == plan_jobs_nat(plan) : Nat}:
  GroupPlan{+limits,+geometry,taps_positive,outputs_positive,+bound,aligned} = plan
  +taps = Packing.geometry_taps(geometry)
  +outputs = geometry_outputs(geometry)
  +tap_blocks = ((taps + 7) / 8 : U32)
  +output_blocks = ((outputs + 7) / 8 : U32)
  +rows_exact = Encoded.eight_blocks_exact(limits,taps,plan_taps_room(plan))
  +columns_exact = Encoded.eight_blocks_exact(limits,outputs,plan_outputs_room(plan))
  +native_room = {
    %Equal.sym(Nat,U32.to_nat(tap_blocks),Ceiling.ceiling(U32.to_nat(taps),8n),rows_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(output_blocks)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(output_blocks),Ceiling.ceiling(U32.to_nat(outputs),8n),columns_exact) : {Nat.is_le(Nat.mul(Ceiling.ceiling(U32.to_nat(taps),8n),_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    Order.transitive(plan_jobs_nat(plan),Nat.mul(U32.to_nat(taps),U32.to_nat(outputs)),U32.to_nat(Machine.capacity(limits)),plan_jobs_room(plan),bound) : {Nat.is_le(Nat.mul(U32.to_nat(tap_blocks),U32.to_nat(output_blocks)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  %rows_exact : {U32.to_nat((tap_blocks * output_blocks : U32)) == Nat.mul(_,Ceiling.ceiling(U32.to_nat(outputs),8n)) : Nat}
  %columns_exact : {U32.to_nat((tap_blocks * output_blocks : U32)) == Nat.mul(U32.to_nat(tap_blocks),_) : Nat}
  Arithmetic.multiply(tap_blocks,output_blocks,Machine.capacity(limits),Limits.capacity_fits(limits),native_room)

def plan_padded(plan: GroupPlan) -> U32:
  (((plan_count(plan) + 7) / 8) * 8 : U32)

def plan_padded_room(+plan: GroupPlan) -> {Nat.is_le(U32.to_nat(plan_padded(plan)),U32.to_nat(Machine.capacity(plan_limits(plan)))) == True{} : Bool}:
  GroupPlan{+limits,geometry,taps_positive,outputs_positive,bound,+aligned} = plan
  Encoded.padded_capacity(limits,plan_count(plan),plan_count_room(plan),aligned)

def plan_allocation(+plan: GroupPlan) -> AllocationDepth.Sized<plan_padded(plan)>:
  GroupPlan{+limits,geometry,taps_positive,outputs_positive,bound,aligned} = plan
  AllocationDepth.bounded(plan_padded(plan),Encoded.native_le_transitive(plan_padded(plan),Machine.capacity(limits),1073741824,
    %Equal.sym(Bool,U32.is_le(plan_padded(plan),Machine.capacity(limits)),Nat.is_le(U32.to_nat(plan_padded(plan)),U32.to_nat(Machine.capacity(limits))),Comparison.le(plan_padded(plan),Machine.capacity(limits))) : {_ == True{} : Bool} plan_padded_room(plan),
    Limits.capacity_allocatable(limits)))

type QueryPlan is Data:
  QueryPlan{plan: GroupPlan,row: U32,column: U32,
    row_inside: {U32.is_lt(row,plan_taps(plan)) == True{} : Bool},column_inside: {U32.is_lt(column,plan_outputs(plan)) == True{} : Bool}}

def query_plan(context: QueryPlan) -> GroupPlan:
  QueryPlan{plan,row,column,row_inside,column_inside} = context
  plan

def query_row(context: QueryPlan) -> U32:
  QueryPlan{plan,row,column,row_inside,column_inside} = context
  row

def query_column(context: QueryPlan) -> U32:
  QueryPlan{plan,row,column,row_inside,column_inside} = context
  column

def query_job(+context: QueryPlan) -> Nat:
  Nat.add(Nat.mul(Nat.div(U32.to_nat(query_row(context)),8n),Ceiling.ceiling(U32.to_nat(plan_outputs(query_plan(context))),8n)),Nat.div(U32.to_nat(query_column(context)),8n))

def query_cell(+context: QueryPlan) -> Nat:
  Nat.add(Nat.mul(Nat.mod(U32.to_nat(query_row(context)),8n),8n),Nat.mod(U32.to_nat(query_column(context)),8n))

def query_row_block_inside(+context: QueryPlan) ->
  {Nat.is_lt(Nat.div(U32.to_nat(query_row(context)),8n),Ceiling.ceiling(U32.to_nat(plan_taps(query_plan(context))),8n)) == True{} : Bool}:
  QueryPlan{+plan,+row,column,+row_inside,column_inside} = context
  +taps = U32.to_nat(plan_taps(plan))
  DivisionOrder.quotient_below(U32.to_nat(row),Ceiling.ceiling(taps,8n),8n,{==},
    Bounds.nat_trans(U32.to_nat(row),taps,Nat.mul(Ceiling.ceiling(taps,8n),8n),
      %Comparison.lt(row,plan_taps(plan)) : {_ == True{} : Bool} row_inside,Ceiling.covers(taps,8n,{==})))

def query_column_block_inside(+context: QueryPlan) ->
  {Nat.is_lt(Nat.div(U32.to_nat(query_column(context)),8n),Ceiling.ceiling(U32.to_nat(plan_outputs(query_plan(context))),8n)) == True{} : Bool}:
  QueryPlan{+plan,row,+column,row_inside,+column_inside} = context
  +outputs = U32.to_nat(plan_outputs(plan))
  DivisionOrder.quotient_below(U32.to_nat(column),Ceiling.ceiling(outputs,8n),8n,{==},
    Bounds.nat_trans(U32.to_nat(column),outputs,Nat.mul(Ceiling.ceiling(outputs,8n),8n),
      %Comparison.lt(column,plan_outputs(plan)) : {_ == True{} : Bool} column_inside,Ceiling.covers(outputs,8n,{==})))

def query_job_inside(+context: QueryPlan) -> {Nat.is_lt(query_job(context),plan_jobs_nat(query_plan(context))) == True{} : Bool}:
  QueryPlan{+plan,+row,+column,row_inside,column_inside} = context
  RowMajor.radix(Nat.div(U32.to_nat(row),8n),Ceiling.ceiling(U32.to_nat(plan_taps(plan)),8n),Nat.div(U32.to_nat(column),8n),Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n),
    query_row_block_inside(context),query_column_block_inside(context))

def query_cell_inside(+context: QueryPlan) -> {Nat.is_lt(query_cell(context),64n) == True{} : Bool}:
  QueryPlan{plan,+row,+column,row_inside,column_inside} = context
  RowMajor.radix(Nat.mod(U32.to_nat(row),8n),8n,Nat.mod(U32.to_nat(column),8n),8n,
    DivisionOrder.remainder_small(U32.to_nat(row),8n,{==}),DivisionOrder.remainder_small(U32.to_nat(column),8n,{==}))

def query_job_exact(+context: QueryPlan) -> {U32.to_nat(U32.from_nat(query_job(context))) == query_job(context) : Nat}:
  QueryPlan{+plan,row,column,row_inside,column_inside} = context
  FullRange.roundtrip(query_job(context),plan_jobs(plan),
    %Equal.sym(Nat,U32.to_nat(plan_jobs(plan)),plan_jobs_nat(plan),plan_jobs_exact(plan)) : {Nat.is_le(query_job(context),_) == True{} : Bool}
    Bounds.strict_le(query_job(context),plan_jobs_nat(plan),query_job_inside(context)))

def query_cell_exact(+context: QueryPlan) -> {U32.to_nat(U32.from_nat(query_cell(context))) == query_cell(context) : Nat}:
  FullRange.roundtrip(query_cell(context),64,Bounds.strict_le(query_cell(context),64n,query_cell_inside(context)))

def plan_output_blocks_exact(+plan: GroupPlan) -> {U32.to_nat(plan_output_blocks(plan)) == Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n) : Nat}:
  Encoded.eight_blocks_exact(plan_limits(plan),plan_outputs(plan),plan_outputs_room(plan))

def plan_output_blocks_room(+plan: GroupPlan) -> {Nat.is_le(Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n),U32.to_nat(Machine.capacity(plan_limits(plan)))) == True{} : Bool}:
  Order.transitive(Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n),U32.to_nat(plan_outputs(plan)),U32.to_nat(Machine.capacity(plan_limits(plan))),Ceiling.bounded(U32.to_nat(plan_outputs(plan)),8n,{==}),plan_outputs_room(plan))

def plan_output_blocks_positive(+plan: GroupPlan) -> {Nat.is_lt(0n,Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n)) == True{} : Bool}:
  GroupPlan{limits,+geometry,taps_positive,+outputs_positive,bound,aligned} = plan
  +outputs = U32.to_nat(geometry_outputs(geometry))
  RowMajor.left_factor_positive(Ceiling.ceiling(outputs,8n),8n,
    Bounds.nat_trans(0n,outputs,Nat.mul(Ceiling.ceiling(outputs,8n),8n),outputs_positive,Ceiling.covers(outputs,8n,{==})))

def query_job_row(+context: QueryPlan) ->
  {U32.to_nat((U32.from_nat(query_job(context)) / plan_output_blocks(query_plan(context)) : U32)) == Nat.div(U32.to_nat(query_row(context)),8n) : Nat}:
  QueryPlan{+plan,+row,+column,row_inside,column_inside} = context
  Radix.quotient_word(U32.from_nat(query_job(context)),plan_output_blocks(plan),Nat.div(U32.to_nat(row),8n),Nat.div(U32.to_nat(column),8n),Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n),
    Machine.half_word(plan_limits(plan)),Limits.half_word_fits(plan_limits(plan)),query_job_exact(context),plan_output_blocks_exact(plan),plan_output_blocks_positive(plan),query_column_block_inside(context),
    Limits.double_bound(plan_limits(plan),Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n),plan_output_blocks_room(plan)))

def query_job_column(+context: QueryPlan) ->
  {U32.to_nat((U32.from_nat(query_job(context)) % plan_output_blocks(query_plan(context)) : U32)) == Nat.div(U32.to_nat(query_column(context)),8n) : Nat}:
  QueryPlan{+plan,+row,+column,row_inside,column_inside} = context
  Radix.remainder_word(U32.from_nat(query_job(context)),plan_output_blocks(plan),Nat.div(U32.to_nat(row),8n),Nat.div(U32.to_nat(column),8n),Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n),
    Machine.half_word(plan_limits(plan)),Limits.half_word_fits(plan_limits(plan)),query_job_exact(context),plan_output_blocks_exact(plan),plan_output_blocks_positive(plan),query_column_block_inside(context),
    Limits.double_bound(plan_limits(plan),Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n),plan_output_blocks_room(plan)))

def query_cell_row(+context: QueryPlan) -> {U32.to_nat((U32.from_nat(query_cell(context)) / 8 : U32)) == Nat.mod(U32.to_nat(query_row(context)),8n) : Nat}:
  QueryPlan{plan,+row,+column,row_inside,column_inside} = context
  Radix.quotient_word(U32.from_nat(query_cell(context)),8,Nat.mod(U32.to_nat(row),8n),Nat.mod(U32.to_nat(column),8n),8n,2147483648,{==},query_cell_exact(context),{==},{==},DivisionOrder.remainder_small(U32.to_nat(column),8n,{==}),{==})

def query_cell_column(+context: QueryPlan) -> {U32.to_nat((U32.from_nat(query_cell(context)) % 8 : U32)) == Nat.mod(U32.to_nat(query_column(context)),8n) : Nat}:
  QueryPlan{plan,+row,+column,row_inside,column_inside} = context
  Radix.remainder_word(U32.from_nat(query_cell(context)),8,Nat.mod(U32.to_nat(row),8n),Nat.mod(U32.to_nat(column),8n),8n,2147483648,{==},query_cell_exact(context),{==},{==},DivisionOrder.remainder_small(U32.to_nat(column),8n,{==}),{==})

def eight_join(+base: U32,+local: U32,+original: U32,+quotient: Nat,+remainder: Nat,
  +base_match: {U32.to_nat(base) == quotient : Nat},+local_match: {U32.to_nat(local) == remainder : Nat},
  +identity: {U32.to_nat(original) == Nat.add(Nat.mul(quotient,8n),remainder) : Nat}) -> {(base * 8 + local : U32) == original : U32}:
  %Arithmetic.native_roundtrip(original) : {(base * 8 + local : U32) == _ : U32}
  %Equal.sym(Nat,U32.to_nat(original),Nat.add(Nat.mul(quotient,8n),remainder),identity) : {(base * 8 + local : U32) == U32.from_nat(_) : U32}
  %Equal.sym(U32,base,U32.from_nat(quotient),Encoded.encoding(base,quotient,base_match)) : {(_ * 8 + local : U32) == U32.from_nat(Nat.add(Nat.mul(quotient,8n),remainder)) : U32}
  %Equal.sym(U32,local,U32.from_nat(remainder),Encoded.encoding(local,remainder,local_match)) : {(U32.from_nat(quotient) * 8 + _ : U32) == U32.from_nat(Nat.add(Nat.mul(quotient,8n),remainder)) : U32}
  Encoded.encoded_madd(quotient,8n,remainder)

def query_native_row(+context: QueryPlan) ->
  {((U32.from_nat(query_job(context)) / plan_output_blocks(query_plan(context))) * 8 + U32.from_nat(query_cell(context)) / 8 : U32) == query_row(context) : U32}:
  QueryPlan{+plan,+row,column,row_inside,column_inside} = context
  eight_join((U32.from_nat(query_job(context)) / plan_output_blocks(plan) : U32),(U32.from_nat(query_cell(context)) / 8 : U32),row,Nat.div(U32.to_nat(row),8n),Nat.mod(U32.to_nat(row),8n),query_job_row(context),query_cell_row(context),DivisionOrder.quotient_exact(U32.to_nat(row),8n,{==}))

def query_native_column(+context: QueryPlan) ->
  {((U32.from_nat(query_job(context)) % plan_output_blocks(query_plan(context))) * 8 + U32.from_nat(query_cell(context)) % 8 : U32) == query_column(context) : U32}:
  QueryPlan{+plan,row,+column,row_inside,column_inside} = context
  eight_join((U32.from_nat(query_job(context)) % plan_output_blocks(plan) : U32),(U32.from_nat(query_cell(context)) % 8 : U32),column,Nat.div(U32.to_nat(column),8n),Nat.mod(U32.to_nat(column),8n),query_job_column(context),query_cell_column(context),DivisionOrder.quotient_exact(U32.to_nat(column),8n,{==}))

def ready_limits(ready: WindowReady) -> Machine.Limits:
  WindowReady{limits,problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  limits

type GroupBounds<-ready: WindowReady> is Data:
  GroupBounds{taps_positive: {Nat.is_lt(0n,U32.to_nat(Packing.geometry_taps(problem_geometry(ready_problem(ready))))) == True{} : Bool},
    outputs_positive: {Nat.is_lt(0n,U32.to_nat(geometry_outputs(problem_geometry(ready_problem(ready))))) == True{} : Bool},
    bound: {Nat.is_le(Nat.mul(U32.to_nat(Packing.geometry_taps(problem_geometry(ready_problem(ready)))),U32.to_nat(geometry_outputs(problem_geometry(ready_problem(ready))))),U32.to_nat(Machine.capacity(ready_limits(ready)))) == True{} : Bool},
    aligned: {U32.is_zero((Machine.capacity(ready_limits(ready)) % 8 : U32)) == True{} : Bool}}

def ready_plan(+ready: WindowReady,bounds: GroupBounds<ready>) -> GroupPlan:
  WindowReady{+limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{geometry,image,tap,output,row,column,start} = window
  GroupBounds{taps_positive,outputs_positive,bound,aligned} = bounds
  GroupPlan{limits,geometry,taps_positive,outputs_positive,bound,aligned}

def ready_query_plan(+ready: WindowReady,+bounds: GroupBounds<ready>,+row: U32,+column: U32,
  row_inside: {U32.is_lt(row,Packing.geometry_taps(problem_geometry(ready_problem(ready)))) == True{} : Bool},
  column_inside: {U32.is_lt(column,geometry_outputs(problem_geometry(ready_problem(ready)))) == True{} : Bool}) -> QueryPlan:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  GroupBounds{taps_positive,outputs_positive,bound,aligned} = bounds
  QueryPlan{ready_plan(ready,bounds),row,column,row_inside,column_inside}

def reference_address(+ready: WindowReady,+row: U32,+column: U32) ->
  {block_query(ready,row,column) == (row * geometry_outputs(problem_geometry(ready_problem(ready))) + column : U32) : U32}:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,taps} = geometry
  %Equal.sym(U32,(row + 0 : U32),row,Arithmetic.add_zero(row)) : {(_ * o + (column + 0) : U32) == (row * o + column : U32) : U32}
  %Equal.sym(U32,(column + 0 : U32),column,Arithmetic.add_zero(column)) : {(row * o + _ : U32) == (row * o + column : U32) : U32}
  {==}

def reference_enabled(+ready: WindowReady,+row: U32,+column: U32,
  +row_inside: {U32.is_lt(row,Packing.geometry_taps(problem_geometry(ready_problem(ready)))) == True{} : Bool},
  +column_inside: {U32.is_lt(column,geometry_outputs(problem_geometry(ready_problem(ready)))) == True{} : Bool}) ->
  {copy_valid(reference_copy(ready,row,column),0) == True{} : Bool}:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  %Equal.sym(U32,(row + 0 : U32),row,Arithmetic.add_zero(row)) : {U32.is_lt(_,taps) && U32.is_lt((column + 0 : U32),o) == True{} : Bool}
  %Equal.sym(U32,(column + 0 : U32),column,Arithmetic.add_zero(column)) : {U32.is_lt(row,taps) && U32.is_lt(_,o) == True{} : Bool}
  %Equal.sym(Bool,U32.is_lt(row,taps),True{},row_inside) : {_ && U32.is_lt(column,o) == True{} : Bool}
  column_inside

def ready_query_copy_valid(+ready: WindowReady,+bounds: GroupBounds<ready>,+row: U32,+column: U32,
  +row_inside: {U32.is_lt(row,Packing.geometry_taps(problem_geometry(ready_problem(ready)))) == True{} : Bool},
  +column_inside: {U32.is_lt(column,geometry_outputs(problem_geometry(ready_problem(ready)))) == True{} : Bool}) ->
  {copy_valid(problem_copy(ready_problem(block_ready(ready,plan_output_blocks(ready_plan(ready,bounds)),U32.from_nat(query_job(ready_query_plan(ready,bounds,row,column,row_inside,column_inside)))))),
    U32.from_nat(query_cell(ready_query_plan(ready,bounds,row,column,row_inside,column_inside)))) == True{} : Bool}:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  GroupBounds{taps_positive,outputs_positive,bound,aligned} = bounds
  +plan = ready_plan(ready,bounds)
  +context = ready_query_plan(ready,bounds,row,column,row_inside,column_inside)
  +job = U32.from_nat(query_job(context))
  +cell = U32.from_nat(query_cell(context))
  +outputs = plan_output_blocks(plan)
  %Equal.sym(U32,(job / outputs * 8 + cell / 8 : U32),row,query_native_row(context)) :
    {U32.is_lt(_,taps) && U32.is_lt((job % outputs * 8 + cell % 8 : U32),o) == True{} : Bool}
  %Equal.sym(U32,(job % outputs * 8 + cell % 8 : U32),column,query_native_column(context)) :
    {U32.is_lt(row,taps) && U32.is_lt(_,o) == True{} : Bool}
  %Equal.sym(Bool,U32.is_lt(row,taps),True{},row_inside) : {_ && U32.is_lt(column,o) == True{} : Bool}
  column_inside

def ready_query_copy_address(+ready: WindowReady,+bounds: GroupBounds<ready>,+row: U32,+column: U32,
  +row_inside: {U32.is_lt(row,Packing.geometry_taps(problem_geometry(ready_problem(ready)))) == True{} : Bool},
  +column_inside: {U32.is_lt(column,geometry_outputs(problem_geometry(ready_problem(ready)))) == True{} : Bool}) ->
  {copy_address(problem_copy(ready_problem(block_ready(ready,plan_output_blocks(ready_plan(ready,bounds)),U32.from_nat(query_job(ready_query_plan(ready,bounds,row,column,row_inside,column_inside)))))),
    U32.from_nat(query_cell(ready_query_plan(ready,bounds,row,column,row_inside,column_inside)))) == block_query(ready,row,column) : U32}:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,taps} = geometry
  GroupBounds{taps_positive,outputs_positive,bound,aligned} = bounds
  +plan = ready_plan(ready,bounds)
  +context = ready_query_plan(ready,bounds,row,column,row_inside,column_inside)
  +job = U32.from_nat(query_job(context))
  +cell = U32.from_nat(query_cell(context))
  +outputs = plan_output_blocks(plan)
  %Equal.sym(U32,block_query(ready,row,column),(row * o + column : U32),reference_address(ready,row,column)) :
    {((job / outputs * 8 + cell / 8) * o + (job % outputs * 8 + cell % 8) : U32) == _ : U32}
  %Equal.sym(U32,(job / outputs * 8 + cell / 8 : U32),row,query_native_row(context)) :
    {(_ * o + (job % outputs * 8 + cell % 8) : U32) == (row * o + column : U32) : U32}
  %Equal.sym(U32,(job % outputs * 8 + cell % 8 : U32),column,query_native_column(context)) :
    {(row * o + _ : U32) == (row * o + column : U32) : U32}
  {==}

def ready_block_read(+ready: WindowReady,+bounds: GroupBounds<ready>,+row: U32,+column: U32,+depth: Nat,+capacity: U32,
  +row_inside: {U32.is_lt(row,Packing.geometry_taps(problem_geometry(ready_problem(ready)))) == True{} : Bool},
  +column_inside: {U32.is_lt(column,geometry_outputs(problem_geometry(ready_problem(ready)))) == True{} : Bool},
  depth_limit: {Nat.is_le(depth,30n) == True{} : Bool},extent: CopyExtent(problem_copy(ready_problem(ready)),capacity)) -> BlockRead:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  GroupBounds{taps_positive,outputs_positive,bound,aligned} = bounds
  +plan = ready_plan(ready,bounds)
  +context = ready_query_plan(ready,bounds,row,column,row_inside,column_inside)
  +target = query_job(context)
  BlockRead{ready,plan_output_blocks(plan),row,column,depth,capacity,U32.to_nat(plan_jobs(plan)),target,query_cell(context),depth_limit,extent,
    reference_enabled(ready,row,column,row_inside,column_inside),
    %Equal.sym(Nat,U32.to_nat(plan_jobs(plan)),plan_jobs_nat(plan),plan_jobs_exact(plan)) : {Nat.is_lt(target,_) == True{} : Bool} query_job_inside(context),query_cell_inside(context),
    ready_query_copy_valid(ready,bounds,row,column,row_inside,column_inside),ready_query_copy_address(ready,bounds,row,column,row_inside,column_inside)}

def plan_depth(plan: GroupPlan) -> Nat:
  Storage.depth_for(plan_padded(plan))

def plan_capacity(plan: GroupPlan) -> U32:
  U32.shln(1,plan_depth(plan))

def allocation_depth_limit(+count: U32,allocation: AllocationDepth.Sized<count>) -> {Nat.is_le(Storage.depth_for(count),30n) == True{} : Bool}:
  AllocationDepth.Sized{depth_ok,room} = allocation
  depth_ok

def plan_depth_limit(+plan: GroupPlan) -> {Nat.is_le(plan_depth(plan),30n) == True{} : Bool}:
  allocation_depth_limit(plan_padded(plan),plan_allocation(plan))

def plan_physical_extent(+plan: GroupPlan) ->
  {Nat.is_le(Nat.mul(U32.to_nat(plan_taps(plan)),U32.to_nat(plan_outputs(plan))),U32.to_nat(plan_capacity(plan))) == True{} : Bool}:
  GroupPlan{+limits,geometry,taps_positive,outputs_positive,bound,aligned} = plan
  +count = plan_count(plan)
  +padded = plan_padded(plan)
  %plan_count_exact(plan) : {Nat.is_le(_,U32.to_nat(plan_capacity(plan))) == True{} : Bool}
  Order.transitive(U32.to_nat(count),U32.to_nat(padded),U32.to_nat(plan_capacity(plan)),
    %Equal.sym(Nat,U32.to_nat(padded),Encoded.ceiling_eight(U32.to_nat(count)),Encoded.padded_exact(limits,count,plan_count_room(plan))) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool}
    Encoded.within_ceiling(U32.to_nat(count)),AllocationDepth.allocation_room(padded,plan_allocation(plan)))

def ready_physical_extent(+ready: WindowReady,+bounds: GroupBounds<ready>) ->
  CopyExtent(problem_copy(ready_problem(ready)),plan_capacity(ready_plan(ready,bounds))):
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  GroupBounds{taps_positive,outputs_positive,bound,aligned} = bounds
  plan_physical_extent(ready_plan(ready,bounds))

def ready_group(+ready: WindowReady,+bounds: GroupBounds<ready>) -> Group:
  +plan = ready_plan(ready,bounds)
  +problem = ready_problem(ready)
  Traversal.indexed_fold(~Group,~Kernel.GroupBlocks,~block,U32.to_nat(plan_jobs(plan)),0,
    ready_blocks(ready,plan_output_blocks(plan)),Group{problem_input(problem),problem_gradient(problem),Tree.fresh(F32,plan_depth(plan),0.0)})

def ready_group_read(+ready: WindowReady,+bounds: GroupBounds<ready>,+row: U32,+column: U32,
  +row_inside: {U32.is_lt(row,Packing.geometry_taps(problem_geometry(ready_problem(ready)))) == True{} : Bool},
  +column_inside: {U32.is_lt(column,geometry_outputs(problem_geometry(ready_problem(ready)))) == True{} : Bool}) ->
  {Tree.read(F32,group_output(ready_group(ready,bounds)),block_query(ready,row,column)) == block_wanted(ready,row,column) : F32}:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,+input,+gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  GroupBounds{taps_positive,outputs_positive,bound,aligned} = bounds
  +plan = ready_plan(ready,bounds)
  +depth = plan_depth(plan)
  +capacity = plan_capacity(plan)
  +values = Tree.fresh(F32,depth,0.0)
  +context = ready_block_read(ready,bounds,row,column,depth,capacity,row_inside,column_inside,plan_depth_limit(plan),ready_physical_extent(ready,bounds))
  block_readback(context,Group{input,gradient,values},Shape.Both{Shape.Both{Shape.fresh(F32,depth,0.0),Shape.capacity(F32,depth,values,Shape.fresh(F32,depth,0.0))},{==}})

def reference_value(+ready: WindowReady,+row: U32,+column: U32) ->
  {block_wanted(ready,row,column) == numeric_dot(numeric_source(ready_problem(ready)),row,column,U32.to_nat(ready_remaining(ready)),problem_first(ready_problem(ready)),0.0) : F32}:
  WindowReady{limits,+problem,remaining,count_match,first,kernel,stride,square,room,nonempty} = ready
  PackedProblem{+window,count,input,gradient,position,positive,row_match,column_match,bounded} = problem
  Kernel.WeightWindow{+geometry,image,tap,output,row0,column0,start} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  +source = numeric_source(problem)
  %Equal.sym(U32,(row + 0 : U32),row,Arithmetic.add_zero(row)) : {numeric_dot(source,_,(column + 0 : U32),U32.to_nat(remaining),position,0.0) == numeric_dot(source,row,column,U32.to_nat(remaining),position,0.0) : F32}
  %Equal.sym(U32,(column + 0 : U32),column,Arithmetic.add_zero(column)) : {numeric_dot(source,row,_,U32.to_nat(remaining),position,0.0) == numeric_dot(source,row,column,U32.to_nat(remaining),position,0.0) : F32}
  {==}

def ready_group_numeric(+ready: WindowReady,+bounds: GroupBounds<ready>,+row: U32,+column: U32,
  +row_inside: {U32.is_lt(row,Packing.geometry_taps(problem_geometry(ready_problem(ready)))) == True{} : Bool},
  +column_inside: {U32.is_lt(column,geometry_outputs(problem_geometry(ready_problem(ready)))) == True{} : Bool}) ->
  {Tree.read(F32,group_output(ready_group(ready,bounds)),(row * geometry_outputs(problem_geometry(ready_problem(ready))) + column : U32)) ==
    numeric_dot(numeric_source(ready_problem(ready)),row,column,U32.to_nat(ready_remaining(ready)),problem_first(ready_problem(ready)),0.0) : F32}:
  %reference_address(ready,row,column) : {Tree.read(F32,group_output(ready_group(ready,bounds)),_) == numeric_dot(numeric_source(ready_problem(ready)),row,column,U32.to_nat(ready_remaining(ready)),problem_first(ready_problem(ready)),0.0) : F32}
  %reference_value(ready,row,column) : {Tree.read(F32,group_output(ready_group(ready,bounds)),block_query(ready,row,column)) == _ : F32}
  ready_group_read(ready,bounds,row,column,row_inside,column_inside)

# Geometry evidence supplies the common packed and output allocation bounds.
type GeometryReady<-geometry: Kernel.Geometry> is Data:
  GeometryReady{limits: Machine.Limits,
    columns: {Nat.is_lt(0n,U32.to_nat(geometry_columns(geometry))) == True{} : Bool},
    kernel: {Nat.is_lt(0n,U32.to_nat(Packing.geometry_kernel(geometry))) == True{} : Bool},
    stride: {Nat.is_lt(0n,U32.to_nat(geometry_stride(geometry))) == True{} : Bool},
    square: {Nat.is_le(Nat.mul(U32.to_nat(Packing.geometry_kernel(geometry)),U32.to_nat(Packing.geometry_kernel(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    taps: {Nat.is_lt(0n,U32.to_nat(Packing.geometry_taps(geometry))) == True{} : Bool},
    outputs: {Nat.is_lt(0n,U32.to_nat(geometry_outputs(geometry))) == True{} : Bool},
    count: {Nat.is_le(Nat.mul(U32.to_nat(Packing.geometry_taps(geometry)),U32.to_nat(geometry_outputs(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    aligned: {U32.is_zero((Machine.capacity(limits) % 8 : U32)) == True{} : Bool}}

def range_ready(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>,+image: U32,+range: Grouped.Range,+group: U32,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>,
  evidence: GroupLaws.RangeBounded(U32.to_nat(geometry_rows(geometry)),U32.to_nat(geometry_columns(geometry)),U32.to_nat(group),range)) -> WindowReady:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  GeometryReady{+limits,+columns_positive,+kernel_positive,+stride_positive,+square,taps_positive,outputs_positive,count,aligned} = domain
  Grouped.Range{+first,+remaining} = range
  Shape.Both{+nonempty,Shape.Both{+bounded,+room}} = evidence
  +encoded = U32.from_nat(remaining)
  +exact = FullRange.roundtrip(remaining,group,bounded)
  +window = {Kernel.WeightWindow{geometry,image,0,0,U32.from_nat(Nat.div(first,U32.to_nat(columns))),U32.from_nat(Nat.mod(first,U32.to_nat(columns))),U32.from_nat(first)} : Kernel.WeightWindow}
  +problem = {PackedProblem{window,U32.min(256,encoded),input,gradient,first,columns_positive,{==},{==},clipped_bound(encoded)} : PackedProblem}
  WindowReady{limits,problem,encoded,{==},{==},kernel_positive,stride_positive,square,
    %Equal.sym(Nat,U32.to_nat(encoded),remaining,exact) : {Nat.is_le(Nat.add(first,_),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool} room,
    %Equal.sym(Nat,U32.to_nat(encoded),remaining,exact) : {Nat.is_lt(0n,_) == True{} : Bool} nonempty}

def range_group_bounds(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>,+image: U32,+range: Grouped.Range,+size: U32,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>,
  +evidence: GroupLaws.RangeBounded(U32.to_nat(geometry_rows(geometry)),U32.to_nat(geometry_columns(geometry)),U32.to_nat(size),range)) ->
  GroupBounds<range_ready(geometry,domain,image,range,size,input,gradient,evidence)>:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned} = domain
  Grouped.Range{first,remaining} = range
  Shape.Both{nonempty,Shape.Both{bounded,room}} = evidence
  GroupBounds{taps_positive,outputs_positive,count,aligned}

def range_group_matches(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>,+image: U32,+range: Grouped.Range,+size: U32,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>,
  +evidence: GroupLaws.RangeBounded(U32.to_nat(geometry_rows(geometry)),U32.to_nat(geometry_columns(geometry)),U32.to_nat(size),range)) ->
  {group(geometry,image,range,input,gradient) == ready_group(range_ready(geometry,domain,image,range,size,input,gradient,evidence),range_group_bounds(geometry,domain,image,range,size,input,gradient,evidence)) : Group}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned} = domain
  Grouped.Range{first,remaining} = range
  Shape.Both{nonempty,Shape.Both{bounded,room}} = evidence
  {==}

def range_group_numeric(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>,+image: U32,+range: Grouped.Range,+size: U32,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>,
  +evidence: GroupLaws.RangeBounded(U32.to_nat(geometry_rows(geometry)),U32.to_nat(geometry_columns(geometry)),U32.to_nat(size),range),+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(geometry)) == True{} : Bool},+output_inside: {U32.is_lt(output,geometry_outputs(geometry)) == True{} : Bool}) ->
  {Tree.read(F32,group_output(group(geometry,image,range,input,gradient)),(tap * geometry_outputs(geometry) + output : U32)) ==
    numeric_dot(numeric_source(ready_problem(range_ready(geometry,domain,image,range,size,input,gradient,evidence))),tap,output,
      U32.to_nat(ready_remaining(range_ready(geometry,domain,image,range,size,input,gradient,evidence))),problem_first(ready_problem(range_ready(geometry,domain,image,range,size,input,gradient,evidence))),0.0) : F32}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned} = domain
  Grouped.Range{first,remaining} = range
  Shape.Both{nonempty,Shape.Both{bounded,room}} = evidence
  +ready = range_ready(geometry,domain,image,range,size,input,gradient,evidence)
  +bounds = range_group_bounds(geometry,domain,image,range,size,input,gradient,evidence)
  %Equal.sym(Group,group(geometry,image,range,input,gradient),ready_group(ready,bounds),range_group_matches(geometry,domain,image,range,size,input,gradient,evidence)) :
    {Tree.read(F32,group_output(_),(tap * geometry_outputs(geometry) + output : U32)) == numeric_dot(numeric_source(ready_problem(ready)),tap,output,U32.to_nat(ready_remaining(ready)),problem_first(ready_problem(ready)),0.0) : F32}
  ready_group_numeric(ready,bounds,tap,output,tap_inside,output_inside)

def geometry_plan(+geometry: Kernel.Geometry,domain: GeometryReady<geometry>) -> GroupPlan:
  GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned} = domain
  GroupPlan{limits,geometry,taps,outputs,count,aligned}

def group_frame(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>,+image: U32,+range: Grouped.Range,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  Shape.Both<StorageFrame(plan_depth(geometry_plan(geometry,domain)),plan_capacity(geometry_plan(geometry,domain)),group_output(group(geometry,image,range,input,gradient))),
    {group_sources(group(geometry,image,range,input,gradient)) == Sources{input,gradient} : Sources}>:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned} = domain
  Grouped.Range{+first,+remaining} = range
  +plan = geometry_plan(geometry,domain)
  +depth = plan_depth(plan)
  +capacity = plan_capacity(plan)
  +output_blocks = plan_output_blocks(plan)
  +window = {Kernel.WeightWindow{geometry,image,0,0,U32.from_nat(Nat.div(first,U32.to_nat(columns))),U32.from_nat(Nat.mod(first,U32.to_nat(columns))),U32.from_nat(first)} : Kernel.WeightWindow}
  +blocks = {Kernel.GroupBlocks{window,U32.from_nat(remaining),output_blocks} : Kernel.GroupBlocks}
  +meta = {BlockShape{blocks,depth,capacity,input,gradient,U32.to_nat(plan_jobs(plan))} : BlockShape}
  +values = Tree.fresh(F32,depth,0.0)
  block_shape_fold(meta,Group{input,gradient,values},Shape.Both{Shape.Both{Shape.fresh(F32,depth,0.0),Shape.capacity(F32,depth,values,Shape.fresh(F32,depth,0.0))},{==}})

# The eight-lane extent is the same padded word used by the allocation.
def plan_round_extent(+plan: GroupPlan) ->
  {Nat.mul(U32.to_nat(((plan_count(plan) + 7) / 8 : U32)),8n) == U32.to_nat(plan_padded(plan)) : Nat}:
  +count = plan_count(plan)
  +limits = plan_limits(plan)
  +room = plan_count_room(plan)
  +rounds = Equal.trans(Nat,U32.to_nat(((count + 7) / 8 : U32)),Ceiling.ceiling(U32.to_nat(count),8n),Nat.div(Nat.add(U32.to_nat(count),7n),8n),
    Encoded.eight_blocks_exact(limits,count,room),Equal.sym(Nat,Nat.div(Nat.add(U32.to_nat(count),7n),8n),Ceiling.ceiling(U32.to_nat(count),8n),Ceiling.shifted_formula(U32.to_nat(count),8n,{==})))
  Equal.trans(Nat,Nat.mul(U32.to_nat(((count + 7) / 8 : U32)),8n),Encoded.ceiling_eight(U32.to_nat(count)),U32.to_nat(plan_padded(plan)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,8n),U32.to_nat(((count + 7) / 8 : U32)),Nat.div(Nat.add(U32.to_nat(count),7n),8n),rounds),
    Equal.sym(Nat,U32.to_nat(plan_padded(plan)),Encoded.ceiling_eight(U32.to_nat(count)),Encoded.padded_exact(limits,count,room)))

def plan_round_room(+plan: GroupPlan) ->
  {Nat.is_le(Nat.mul(U32.to_nat(((plan_count(plan) + 7) / 8 : U32)),8n),U32.to_nat(plan_capacity(plan))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(U32.to_nat(((plan_count(plan) + 7) / 8 : U32)),8n),U32.to_nat(plan_padded(plan)),plan_round_extent(plan)) : {Nat.is_le(_,U32.to_nat(plan_capacity(plan))) == True{} : Bool}
  AllocationDepth.allocation_room(plan_padded(plan),plan_allocation(plan))

def plan_round_selected(+plan: GroupPlan,+index: Nat,+selected: {Nat.is_lt(index,U32.to_nat(plan_count(plan))) == True{} : Bool}) ->
  {Nat.is_lt(index,Nat.mul(U32.to_nat(((plan_count(plan) + 7) / 8 : U32)),8n)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(U32.to_nat(((plan_count(plan) + 7) / 8 : U32)),8n),U32.to_nat(plan_padded(plan)),plan_round_extent(plan)) : {Nat.is_lt(index,_) == True{} : Bool}
  Bounds.nat_trans(index,U32.to_nat(plan_count(plan)),U32.to_nat(plan_padded(plan)),selected,
    %Equal.sym(Nat,U32.to_nat(plan_padded(plan)),Encoded.ceiling_eight(U32.to_nat(plan_count(plan))),Encoded.padded_exact(plan_limits(plan),plan_count(plan),plan_count_room(plan))) : {Nat.is_le(U32.to_nat(plan_count(plan)),_) == True{} : Bool}
    Encoded.within_ceiling(U32.to_nat(plan_count(plan))))

def plan_capacity_bound(+plan: GroupPlan) -> {U32.is_le(plan_capacity(plan),2147483648) == True{} : Bool}:
  Shift.power_fits(plan_depth(plan),plan_depth_limit(plan))

def plan_query_inside(+plan: GroupPlan,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,plan_taps(plan)) == True{} : Bool},+output_inside: {U32.is_lt(output,plan_outputs(plan)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(tap),U32.to_nat(plan_outputs(plan))),U32.to_nat(output)),U32.to_nat(plan_count(plan))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(plan_count(plan)),Nat.mul(U32.to_nat(plan_taps(plan)),U32.to_nat(plan_outputs(plan))),plan_count_exact(plan)) :
    {Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(tap),U32.to_nat(plan_outputs(plan))),U32.to_nat(output)),_) == True{} : Bool}
  RowMajor.radix(U32.to_nat(tap),U32.to_nat(plan_taps(plan)),U32.to_nat(output),U32.to_nat(plan_outputs(plan)),Bounds.as_nat_lt(tap,plan_taps(plan),tap_inside),Bounds.as_nat_lt(output,plan_outputs(plan),output_inside))
