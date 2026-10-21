# Canonical storage readers for gradient and patch tiles share the tile proof.
import Base
import ../kernel_weight_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../math_sequence.bend as Sequence
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./kernel_tile_refinement.bend as Tiles

import ../traversal_loop.bend as Loop
import ../traversal_array.bend as Traversal
import ./loop_projection.bend as Projection
import ./traversal_array_refinement.bend as Traversals
import ./storage_refinement.bend as StorageLaws
import ./traversal_indexed_values.bend as IndexedValues
import ./tensor_access_refinement.bend as AccessLaws
import ./storage_shape.bend as Shape
import ./tensor_allocation_depth.bend as AllocationDepth
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order
import ./nat_ceiling.bend as Ceiling
import ./nat_extrema.bend as Extrema
import ./nat_to_u32_bounds.bend as Bounds
import ./word_encoding.bend as Words
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./kernel_accumulator.bend as Accumulator
import ../kernel_rows.bend as Rows
import ./kernel_coordinate_refinement.bend as Coordinates
import ./encoded_coordinates.bend as Encoded
import ./boolean_logic.bend as Logic
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./u32_division.bend as NativeDivision
import ./row_major_index.bend as RowMajor
import ./list_storage_readback.bend as ListRead
import ./nat_division_order.bend as DivisionOrder
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./binary_radix.bend as Radix

def pixel_value(input: Buffer.Model<F32>,+row: U32,+column: U32,height: U32,+width: U32,+stride: U32,+padding: U32,lane: Kernel.Lane) -> F32:
  Kernel.Lane{base,kr,kc,valid,rf,re,cf,ce} = lane
  Bool.pick(F32,valid && U32.is_le(rf,row) && U32.is_lt(row,re) && U32.is_le(cf,column) && U32.is_lt(column,ce),
    Tree.read(F32,Buffer.tree(~F32,input),(base + (row * stride + kr - padding) * width + (column * stride + kc - padding) : U32)),0.0)

def pixel(+input: Buffer.Model<F32>,+row: U32,+column: U32,+height: U32,+width: U32,+stride: U32,+padding: U32,+lane: Kernel.Lane) ->
  {Kernel.pixel(Buffer.pack(~F32,input),row,column,height,width,stride,padding,lane) ==
    (Buffer.pack(~F32,input),pixel_value(input,row,column,height,width,stride,padding,lane)) : Storage.Buffer<F32> & F32}:
  Kernel.Lane{base,kr,kc,valid,rf,re,cf,ce} = lane
  Buffer.read_masked(~F32,0.0,valid && U32.is_le(rf,row) && U32.is_lt(row,re) && U32.is_le(cf,column) && U32.is_lt(column,ce),input,
    (base + (row * stride + kr - padding) * width + (column * stride + kc - padding) : U32))

def patch_sample(input: Buffer.Model<F32>,context: Kernel.Pixel,index: Nat) -> F32:
  Kernel.Pixel{patch,row,column,height,width,stride,padding} = context
  pixel_value(input,row,column,height,width,stride,padding,Kernel.selected(patch,index))

def patch_read(+input: Buffer.Model<F32>,+context: Kernel.Pixel,+index: Nat) ->
  {Kernel.read_pixel(Buffer.pack(~F32,input),context,index) == (Buffer.pack(~F32,input),patch_sample(input,context,index)) : Storage.Buffer<F32> & F32}:
  Kernel.Pixel{patch,row,column,height,width,stride,padding} = context
  pixel(input,row,column,height,width,stride,padding,Kernel.selected(patch,index))

def tile_value(~Context: Data,~sample: Buffer.Model<F32> -> Context -> Nat -> F32,+input: Buffer.Model<F32>,+context: Context) -> Tile.Tile8:
  Tile.from_sequence(Sequence.Elements{sample(input,context,0n),
    Sequence.tabulate(~F32,~Tiles.SampleContext<Buffer.Model<F32>,Context>,~Tiles.sample_context(~Buffer.Model<F32>,~Context,~sample),7n,1n,Tiles.SampleContext{input,context})})

def tile_read(~Context: Data,~read: Storage.Buffer<F32> -> Context -> Nat -> Storage.Buffer<F32> & F32,
  ~sample: Buffer.Model<F32> -> Context -> Nat -> F32,
  ~matches: @+input: Buffer.Model<F32> -> @+context: Context -> @+index: Nat ->
    {read(Buffer.pack(~F32,input),context,index) == (Buffer.pack(~F32,input),sample(input,context,index)) : Storage.Buffer<F32> & F32},
  +input: Buffer.Model<F32>,+context: Context) ->
  {Tile.read_tail(~Storage.Buffer<F32>,~Context,~read,read(Buffer.pack(~F32,input),context,0n),context) ==
    (Buffer.pack(~F32,input),tile_value(~Context,~sample,input,context)) : Storage.Buffer<F32> & Tile.Tile8}:
  %Equal.sym(Storage.Buffer<F32> & F32,read(Buffer.pack(~F32,input),context,0n),(Buffer.pack(~F32,input),sample(input,context,0n)),matches(input,context,0n)) :
    {Tile.read_tail(~Storage.Buffer<F32>,~Context,~read,_,context) == (Buffer.pack(~F32,input),tile_value(~Context,~sample,input,context)) : Storage.Buffer<F32> & Tile.Tile8}
  Tiles.tail_block_refines(~Storage.Buffer<F32>,~Buffer.Model<F32>,~Context,~Buffer.pack(~F32),~read,~sample,~matches,input,context,sample(input,context,0n))

def patch_tile(+input: Buffer.Model<F32>,+context: Kernel.Pixel) ->
  {Tile.read_tail(~Storage.Buffer<F32>,~Kernel.Pixel,~Kernel.read_pixel,Kernel.read_pixel(Buffer.pack(~F32,input),context,0n),context) ==
    (Buffer.pack(~F32,input),tile_value(~Kernel.Pixel,~patch_sample,input,context)) : Storage.Buffer<F32> & Tile.Tile8}:
  tile_read(~Kernel.Pixel,~Kernel.read_pixel,~patch_sample,~patch_read,input,context)

def gradient_value(input: Buffer.Model<F32>,context: Kernel.Gradient,+lane: Nat,+position: U32) -> F32:
  Kernel.Gradient{outputs,+spatial,+start,+row,destination} = context
  Bool.pick(F32,U32.is_lt((row + U32.from_nat(lane) : U32),outputs),
    Tree.read(F32,Buffer.tree(~F32,input),((row + U32.from_nat(lane)) * spatial + start + position : U32)),0.0)

def gradient_pixel(+input: Buffer.Model<F32>,+context: Kernel.Gradient,+lane: Nat,+position: U32) ->
  {Kernel.gradient_pixel(Buffer.pack(~F32,input),context,lane,position) == (Buffer.pack(~F32,input),gradient_value(input,context,lane,position)) : Storage.Buffer<F32> & F32}:
  Kernel.Gradient{outputs,spatial,start,row,destination} = context
  Buffer.read_masked(~F32,0.0,U32.is_lt((row + U32.from_nat(lane) : U32),outputs),input,((row + U32.from_nat(lane)) * spatial + start + position : U32))

def gradient_sample(input: Buffer.Model<F32>,context: Kernel.GradientRead,lane: Nat) -> F32:
  Kernel.GradientRead{context,position} = context
  gradient_value(input,context,lane,position)

def gradient_lane(+input: Buffer.Model<F32>,+context: Kernel.GradientRead,+lane: Nat) ->
  {Kernel.gradient_lane(Buffer.pack(~F32,input),context,lane) == (Buffer.pack(~F32,input),gradient_sample(input,context,lane)) : Storage.Buffer<F32> & F32}:
  Kernel.GradientRead{context,position} = context
  gradient_pixel(input,context,lane,position)

def gradient_tile(+input: Buffer.Model<F32>,+context: Kernel.Gradient,+position: U32) ->
  {Kernel.gradient_tile(Buffer.pack(~F32,input),context,position) ==
    (Buffer.pack(~F32,input),tile_value(~Kernel.GradientRead,~gradient_sample,input,Kernel.GradientRead{context,position})) : Storage.Buffer<F32> & Tile.Tile8}:
  tile_read(~Kernel.GradientRead,~Kernel.gradient_lane,~gradient_sample,~gradient_lane,input,Kernel.GradientRead{context,position})

type Packed is Data:
  Packed{input: Buffer.Model<F32>,output: Tree.Tree<Tile.Tile8>}

type Packing is Data:
  Packing{input: Buffer.Model<F32>,output: Tree.Tree<Tile.Tile8>,index: U32,row: U32,column: U32,columns: U32,patch: Kernel.Patch,height: U32,width: U32,stride: U32,padding: U32}

def represent(state: Packing) -> Kernel.Packing:
  Packing{input,output,index,row,column,columns,patch,height,width,stride,padding} = state
  Kernel.Packing{Buffer.pack(~F32,input),Tree.pack(Tile.Tile8,output),index,row,column,columns,patch,height,width,stride,padding}

def advanced(wrap: Bool,input: Buffer.Model<F32>,output: Tree.Tree<Tile.Tile8>,index: U32,+row: U32,column: U32,columns: U32,patch: Kernel.Patch,height: U32,width: U32,stride: U32,padding: U32) -> Packing:
  match wrap:
    case False{}: Packing{input,output,index,row,column,columns,patch,height,width,stride,padding}
    case True{}: Packing{input,output,index,(row + 1 : U32),0,columns,patch,height,width,stride,padding}

def advanced_matches(wrap: Bool,+input: Buffer.Model<F32>,+output: Tree.Tree<Tile.Tile8>,+index: U32,+row: U32,+column: U32,+columns: U32,+patch: Kernel.Patch,+height: U32,+width: U32,+stride: U32,+padding: U32) ->
  {Kernel.advanced(wrap,Buffer.pack(~F32,input),Tree.pack(Tile.Tile8,output),index,row,column,columns,patch,height,width,stride,padding) ==
    represent(advanced(wrap,input,output,index,row,column,columns,patch,height,width,stride,padding)) : Kernel.Packing}:
  match wrap:
    case False{}: {==}
    case True{}: {==}

def step(state: Packing) -> Packing:
  Packing{+input,output,+index,+row,+column,+columns,+patch,+height,+width,+stride,+padding} = state
  advanced(U32.is_eq((column + 1 : U32),columns),input,
    Tree.write(Tile.Tile8,output,index,tile_value(~Kernel.Pixel,~patch_sample,input,Kernel.Pixel{patch,row,column,height,width,stride,padding})),
    (index + 1 : U32),row,(column + 1 : U32),columns,patch,height,width,stride,padding)

def step_matches(+state: Packing) -> {Kernel.pack_step(represent(state)) == represent(step(state)) : Kernel.Packing}:
  Packing{+input,+output,+index,+row,+column,+columns,+patch,+height,+width,+stride,+padding} = state
  +context = {Kernel.Pixel{patch,row,column,height,width,stride,padding} : Kernel.Pixel}
  +tile = tile_value(~Kernel.Pixel,~patch_sample,input,context)
  %Equal.sym(Storage.Buffer<F32> & Tile.Tile8,
    Tile.read_tail(~Storage.Buffer<F32>,~Kernel.Pixel,~Kernel.read_pixel,Kernel.read_pixel(Buffer.pack(~F32,input),context,0n),context),
    (Buffer.pack(~F32,input),tile),patch_tile(input,context)) :
    {Kernel.packed(Tree.pack(Tile.Tile8,output),index,row,column,columns,patch,height,width,stride,padding,_) == represent(step(state)) : Kernel.Packing}
  %Equal.sym(Array<Tile.Tile8>,Array.set(Tile.Tile8,Tree.pack(Tile.Tile8,output),index,tile),Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,output,index,tile)),StorageLaws.write(Tile.Tile8,output,index,tile)) :
    {Kernel.advanced(U32.is_eq((column + 1 : U32),columns),Buffer.pack(~F32,input),_,(index + 1 : U32),row,(column + 1 : U32),columns,patch,height,width,stride,padding) == represent(step(state)) : Kernel.Packing}
  advanced_matches(U32.is_eq((column + 1 : U32),columns),input,Tree.write(Tile.Tile8,output,index,tile),(index + 1 : U32),row,(column + 1 : U32),columns,patch,height,width,stride,padding)

def packing_loop(+count: Nat,+state: Packing) ->
  {Loop.run(~Kernel.Packing,~Kernel.pack_step,count,represent(state)) == represent(Loop.run(~Packing,~step,count,state)) : Kernel.Packing}:
  Projection.iteration_refines_model(~Kernel.Packing,~Packing,~Kernel.pack_step,~step,~represent,~step_matches,count,state)

def owners(result: Packed) -> Storage.Buffer<F32> & Array<Tile.Tile8>:
  Packed{input,output} = result
  (Buffer.pack(~F32,input),Tree.pack(Tile.Tile8,output))

def gradient_tile_value(input: Buffer.Model<F32>,context: Kernel.Gradient,position: U32) -> Tile.Tile8:
  tile_value(~Kernel.GradientRead,~gradient_sample,input,Kernel.GradientRead{context,position})

def gradient_finished(state: Traversal.State<&2,&2,Buffer.Model<F32>,Tree.Tree<Tile.Tile8>,Kernel.Gradient>) -> Packed:
  Traversal.State{input,output,position,context} = state
  Packed{input,output}

def gradient_finished_matches(+state: Traversal.State<&2,&2,Buffer.Model<F32>,Tree.Tree<Tile.Tile8>,Kernel.Gradient>) ->
  {Traversal.finish(~Storage.Buffer<F32>,~Tile.Tile8,~Kernel.Gradient,
    Traversals.represent(~Storage.Buffer<F32>,~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~Buffer.pack(~F32),state)) ==
    owners(gradient_finished(state)) : Storage.Buffer<F32> & Array<Tile.Tile8>}:
  Traversal.State{input,output,position,context} = state
  {==}

def packed_tree(result: Packed) -> Tree.Tree<Tile.Tile8>:
  Packed{input,output} = result
  output

def gradient_model_output(+state: Traversal.State<&2,&2,Buffer.Model<F32>,Tree.Tree<Tile.Tile8>,Kernel.Gradient>) ->
  {packed_tree(gradient_finished(state)) == IndexedValues.output_tree(~Buffer.Model<F32>,~Tile.Tile8,AccessLaws.finish_model(~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,state)) : Tree.Tree<Tile.Tile8>}:
  Traversal.State{input,output,position,context} = state
  {==}

def gradient_place_identity(+context: Kernel.Gradient,+position: U32) -> {Kernel.gradient_destination(context,position) == position : U32}:
  {==}

def window_input(window: Kernel.WeightWindow,+count: U32,input: Buffer.Model<F32>) -> Packing:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Loop.run(~Packing,~step,U32.to_nat(count),
    Packing{input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0)),0,row,column,columns,Kernel.patch(tap,geometry,image),h,w,s,p})

def window_input_matches(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>) ->
  {Kernel.window_input(window,count,Buffer.pack(~F32,input)) == represent(window_input(window,count,input)) : Kernel.Packing}:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  +depth = Storage.depth_for(count)
  +initial = {Packing{input,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0)),0,row,column,columns,Kernel.patch(tap,geometry,image),h,w,s,p} : Packing}
  %Equal.sym(Array<Tile.Tile8>,Array.new(Tile.Tile8,depth,Tile.filled(0.0)),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))),StorageLaws.fresh(Tile.Tile8,depth,Tile.filled(0.0))) :
    {Loop.run(~Kernel.Packing,~Kernel.pack_step,U32.to_nat(count),Kernel.Packing{Buffer.pack(~F32,input),_,0,row,column,columns,Kernel.patch(tap,geometry,image),h,w,s,p}) == represent(window_input(window,count,input)) : Kernel.Packing}
  packing_loop(U32.to_nat(count),initial)

def window_gradient(window: Kernel.WeightWindow,+count: U32,input: Buffer.Model<F32>) -> Packed:
  Kernel.WeightWindow{geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,+rows,+columns,taps} = geometry
  gradient_finished(Loop.run(~Traversal.State<&2,&2,Buffer.Model<F32>,Tree.Tree<Tile.Tile8>,Kernel.Gradient>,
    ~Traversals.model_step(~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~gradient_tile_value,~Kernel.gradient_destination),U32.to_nat(count),
    Traversal.State{input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0)),0,
      Kernel.Gradient{o,(rows * columns : U32),(image * o * rows * columns + first : U32),output,0}}))

def gradient_context(window: Kernel.WeightWindow) -> Kernel.Gradient:
  Kernel.WeightWindow{geometry,+image,tap,output,row,column,first} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,+rows,+columns,taps} = geometry
  Kernel.Gradient{o,(rows * columns : U32),(image * o * rows * columns + first : U32),output,0}

def window_gradient_output(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>) ->
  {packed_tree(window_gradient(window,count,input)) == IndexedValues.run(~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~gradient_tile_value,~Kernel.gradient_destination,
    U32.to_nat(count),0,gradient_context(window),input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0))) : Tree.Tree<Tile.Tile8>}:
  Kernel.WeightWindow{geometry,+image,tap,output,row,column,first} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,+rows,+columns,taps} = geometry
  +context = {Kernel.Gradient{o,(rows * columns : U32),(image * o * rows * columns + first : U32),output,0} : Kernel.Gradient}
  +state = Loop.run(~Traversal.State<&2,&2,Buffer.Model<F32>,Tree.Tree<Tile.Tile8>,Kernel.Gradient>,~Traversals.model_step(~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~gradient_tile_value,~Kernel.gradient_destination),U32.to_nat(count),
    Traversal.State{input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0)),0,context})
  gradient_model_output(state)

def window_gradient_readback_sized(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>,+position: Nat,
  sized: AllocationDepth.Sized<count>,+within: {Nat.is_lt(position,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,packed_tree(window_gradient(window,count,input)),U32.from_nat(position)) == gradient_tile_value(input,gradient_context(window),U32.from_nat(position)) : Tile.Tile8}:
  AllocationDepth.Sized{+depth_ok,room} = sized
  +depth = Storage.depth_for(count)
  +tree = Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))
  +balanced = Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0))
  +capacity = U32.shln(1,depth)
  %Equal.sym(Tree.Tree<Tile.Tile8>,packed_tree(window_gradient(window,count,input)),IndexedValues.run(~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~gradient_tile_value,~Kernel.gradient_destination,U32.to_nat(count),0,gradient_context(window),input,tree),window_gradient_output(window,count,input)) :
    {Tree.read(Tile.Tile8,_,U32.from_nat(position)) == gradient_tile_value(input,gradient_context(window),U32.from_nat(position)) : Tile.Tile8}
  IndexedValues.stored(~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~gradient_tile_value,~Kernel.gradient_destination,~gradient_place_identity,U32.to_nat(count),gradient_context(window),input,tree,depth,capacity,position,Shape.capacity(Tile.Tile8,depth,tree,balanced),balanced,depth_ok,Power.native_power_bound(depth,Order.successor(depth,30n,depth_ok)),room,within)

def window_gradient_readback(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>,+position: Nat,
  bounded: {U32.is_le(count,256) == True{} : Bool},+within: {Nat.is_lt(position,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,packed_tree(window_gradient(window,count,input)),U32.from_nat(position)) == gradient_tile_value(input,gradient_context(window),U32.from_nat(position)) : Tile.Tile8}:
  window_gradient_readback_sized(window,count,input,position,AllocationDepth.bounded_with(count,256,{==},{==},{==},bounded),within)

def window_gradient_matches(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>) ->
  {Kernel.window_gradient(window,count,Buffer.pack(~F32,input)) == owners(window_gradient(window,count,input)) : Storage.Buffer<F32> & Array<Tile.Tile8>}:
  Kernel.WeightWindow{geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,+rows,+columns,taps} = geometry
  +depth = Storage.depth_for(count)
  +context = {Kernel.Gradient{o,(rows * columns : U32),(image * o * rows * columns + first : U32),output,0} : Kernel.Gradient}
  +initial = {Traversal.State{input,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0)),0,context} : Traversal.State<&2,&2,Buffer.Model<F32>,Tree.Tree<Tile.Tile8>,Kernel.Gradient>}
  +state = Loop.run(~Traversal.State<&2,&2,Buffer.Model<F32>,Tree.Tree<Tile.Tile8>,Kernel.Gradient>,
    ~Traversals.model_step(~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~gradient_tile_value,~Kernel.gradient_destination),U32.to_nat(count),initial)
  %Equal.sym(Array<Tile.Tile8>,Array.new(Tile.Tile8,depth,Tile.filled(0.0)),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))),StorageLaws.fresh(Tile.Tile8,depth,Tile.filled(0.0))) :
    {Traversal.finish(~Storage.Buffer<F32>,~Tile.Tile8,~Kernel.Gradient,
      Traversal.run(~Storage.Buffer<F32>,~Tile.Tile8,~Kernel.Gradient,~Kernel.gradient_tile,~Kernel.gradient_destination,U32.to_nat(count),Traversal.State{Buffer.pack(~F32,input),_,0,context})) == owners(window_gradient(window,count,input)) : Storage.Buffer<F32> & Array<Tile.Tile8>}
  %Equal.sym(Traversal.State<&1,&1,Storage.Buffer<F32>,Array<Tile.Tile8>,Kernel.Gradient>,
    Traversal.run(~Storage.Buffer<F32>,~Tile.Tile8,~Kernel.Gradient,~Kernel.gradient_tile,~Kernel.gradient_destination,U32.to_nat(count),
      Traversals.represent(~Storage.Buffer<F32>,~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~Buffer.pack(~F32),initial)),
    Traversals.represent(~Storage.Buffer<F32>,~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~Buffer.pack(~F32),state),
    Traversals.traversal_refines_model(~Storage.Buffer<F32>,~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~Buffer.pack(~F32),~Kernel.gradient_tile,~gradient_tile_value,~Kernel.gradient_destination,~gradient_tile,U32.to_nat(count),initial)) :
    {Traversal.finish(~Storage.Buffer<F32>,~Tile.Tile8,~Kernel.Gradient,_) == owners(window_gradient(window,count,input)) : Storage.Buffer<F32> & Array<Tile.Tile8>}
  gradient_finished_matches(state)

# Row-major cursor arithmetic stays exact across every column boundary.
def encoded_pair(pair: Nat & Nat) -> U32 & U32:
  (row,column) = pair
  (U32.from_nat(row),U32.from_nat(column))

def next_coordinates(wrap: Bool,+row: U32,column: U32) -> U32 & U32:
  match wrap:
    case False{}: (row,column)
    case True{}: ((row + 1 : U32),0)

def next_encoded(wrap: Bool,+row: Nat,+column: Nat) ->
  {next_coordinates(wrap,U32.from_nat(row),U32.from_nat(1n+column)) == encoded_pair(Bool.pick(Nat & Nat,wrap,(1n+row,0n),(row,1n+column))) : U32 & U32}:
  match wrap:
    case False{}: {==}
    case True{}:
      %Equal.sym(U32,(U32.from_nat(row) + 1 : U32),U32.from_nat(1n+row),Bounds.increment_encoded(row)) : {(_,0) == (U32.from_nat(1n+row),0) : U32 & U32}
      {==}

def wrap_exact(+remainder: Nat,+columns: U32,+bound: {Nat.is_le(1n+remainder,U32.to_nat(columns)) == True{} : Bool}) ->
  {U32.is_eq((U32.from_nat(remainder) + 1 : U32),columns) == Nat.is_eq(1n+remainder,U32.to_nat(columns)) : Bool}:
  %Equal.sym(U32,(U32.from_nat(remainder) + 1 : U32),U32.from_nat(1n+remainder),Bounds.increment_encoded(remainder)) : {U32.is_eq(_,columns) == Nat.is_eq(1n+remainder,U32.to_nat(columns)) : Bool}
  %Equal.sym(Bool,U32.is_eq(U32.from_nat(1n+remainder),columns),Nat.is_eq(U32.to_nat(U32.from_nat(1n+remainder)),U32.to_nat(columns)),Comparison.eq(U32.from_nat(1n+remainder),columns)) : {_ == Nat.is_eq(1n+remainder,U32.to_nat(columns)) : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(1n+remainder)),1n+remainder,FullRange.roundtrip(1n+remainder,columns,bound)) : {Nat.is_eq(_,U32.to_nat(columns)) == Nat.is_eq(1n+remainder,U32.to_nat(columns)) : Bool}
  {==}

def coordinate_increment(+position: Nat,+columns: U32,+positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool}) ->
  {next_coordinates(U32.is_eq((U32.from_nat(Nat.mod(position,U32.to_nat(columns))) + 1 : U32),columns),U32.from_nat(Nat.div(position,U32.to_nat(columns))),(U32.from_nat(Nat.mod(position,U32.to_nat(columns))) + 1 : U32)) == encoded_pair(Nat.divmod(1n+position,U32.to_nat(columns))) : U32 & U32}:
  +divisor = U32.to_nat(columns)
  +row = Nat.div(position,divisor)
  +column = Nat.mod(position,divisor)
  +small = DivisionOrder.remainder_small(position,divisor,positive)
  +bound = Order.strict_follow(column,divisor,small)
  %Equal.sym(Bool,U32.is_eq((U32.from_nat(column) + 1 : U32),columns),Nat.is_eq(1n+column,divisor),wrap_exact(column,columns,bound)) :
    {next_coordinates(_,U32.from_nat(row),(U32.from_nat(column) + 1 : U32)) == encoded_pair(Nat.divmod(1n+position,divisor)) : U32 & U32}
  %Equal.sym(U32,(U32.from_nat(column) + 1 : U32),U32.from_nat(1n+column),Bounds.increment_encoded(column)) :
    {next_coordinates(Nat.is_eq(1n+column,divisor),U32.from_nat(row),_) == encoded_pair(Nat.divmod(1n+position,divisor)) : U32 & U32}
  %Equal.sym(Nat & Nat,Nat.divmod(1n+position,divisor),Bool.pick(Nat & Nat,Nat.is_eq(1n+column,divisor),(1n+row,0n),(row,1n+column)),DivisionOrder.increment_case(position,divisor,row,column,DivisionOrder.quotient_exact(position,divisor,positive),small,positive,Order.upper_bound_cases(1n+column,divisor,bound))) :
    {next_coordinates(Nat.is_eq(1n+column,divisor),U32.from_nat(row),U32.from_nat(1n+column)) == encoded_pair(_) : U32 & U32}
  next_encoded(Nat.is_eq(1n+column,divisor),row,column)

type NaturalPacking is Data:
  NaturalPacking{input: Buffer.Model<F32>,output: Tree.Tree<Tile.Tile8>,index: Nat,position: Nat,columns: U32,patch: Kernel.Patch,height: U32,width: U32,stride: U32,padding: U32,positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool}}

type PatchContext is Data:
  PatchContext{input: Buffer.Model<F32>,columns: U32,patch: Kernel.Patch,height: U32,width: U32,stride: U32,padding: U32}

def natural_sample(context: PatchContext,+position: Nat) -> Tile.Tile8:
  PatchContext{input,+columns,patch,height,width,stride,padding} = context
  tile_value(~Kernel.Pixel,~patch_sample,input,Kernel.Pixel{patch,U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))),height,width,stride,padding})

def natural_value(context: PatchContext,position: Nat) -> Tile.Tile8:
  natural_sample(context,position)

def natural_represent(state: NaturalPacking) -> Packing:
  NaturalPacking{input,output,index,+position,+columns,patch,height,width,stride,padding,positive} = state
  Packing{input,output,U32.from_nat(index),U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))),columns,patch,height,width,stride,padding}

def natural_step(state: NaturalPacking) -> NaturalPacking:
  NaturalPacking{+input,output,+index,+position,+columns,+patch,+height,+width,+stride,+padding,positive} = state
  NaturalPacking{input,Tree.write(Tile.Tile8,output,U32.from_nat(index),natural_sample(PatchContext{input,columns,patch,height,width,stride,padding},position)),1n+index,1n+position,columns,patch,height,width,stride,padding,positive}

def packing_coordinates(input: Buffer.Model<F32>,output: Tree.Tree<Tile.Tile8>,index: U32,columns: U32,patch: Kernel.Patch,height: U32,width: U32,stride: U32,padding: U32,pair: U32 & U32) -> Packing:
  (row,column) = pair
  Packing{input,output,index,row,column,columns,patch,height,width,stride,padding}

def advanced_coordinates(wrap: Bool,+input: Buffer.Model<F32>,+output: Tree.Tree<Tile.Tile8>,+index: U32,+row: U32,+column: U32,+columns: U32,+patch: Kernel.Patch,+height: U32,+width: U32,+stride: U32,+padding: U32) ->
  {advanced(wrap,input,output,index,row,column,columns,patch,height,width,stride,padding) == packing_coordinates(input,output,index,columns,patch,height,width,stride,padding,next_coordinates(wrap,row,column)) : Packing}:
  match wrap:
    case False{}: {==}
    case True{}: {==}

def packing_encoded(+input: Buffer.Model<F32>,+output: Tree.Tree<Tile.Tile8>,+index: U32,+columns: U32,+patch: Kernel.Patch,+height: U32,+width: U32,+stride: U32,+padding: U32,pair: Nat & Nat) ->
  {packing_coordinates(input,output,index,columns,patch,height,width,stride,padding,encoded_pair(pair)) == Packing{input,output,index,U32.from_nat(Pair.fst(Nat,Nat,pair)),U32.from_nat(Pair.snd(Nat,Nat,pair)),columns,patch,height,width,stride,padding} : Packing}:
  (row,column) = pair
  {==}

def natural_step_matches(+state: NaturalPacking) -> {step(natural_represent(state)) == natural_represent(natural_step(state)) : Packing}:
  NaturalPacking{+input,+output,+index,+position,+columns,+patch,+height,+width,+stride,+padding,+positive} = state
  +row = U32.from_nat(Nat.div(position,U32.to_nat(columns)))
  +column = U32.from_nat(Nat.mod(position,U32.to_nat(columns)))
  +written = Tree.write(Tile.Tile8,output,U32.from_nat(index),natural_sample(PatchContext{input,columns,patch,height,width,stride,padding},position))
  %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
    {advanced(U32.is_eq((column + 1 : U32),columns),input,written,_,row,(column + 1 : U32),columns,patch,height,width,stride,padding) == natural_represent(natural_step(state)) : Packing}
  Equal.trans(Packing,
    advanced(U32.is_eq((column + 1 : U32),columns),input,written,U32.from_nat(1n+index),row,(column + 1 : U32),columns,patch,height,width,stride,padding),
    packing_coordinates(input,written,U32.from_nat(1n+index),columns,patch,height,width,stride,padding,next_coordinates(U32.is_eq((column + 1 : U32),columns),row,(column + 1 : U32))),
    natural_represent(natural_step(state)),
    advanced_coordinates(U32.is_eq((column + 1 : U32),columns),input,written,U32.from_nat(1n+index),row,(column + 1 : U32),columns,patch,height,width,stride,padding),
    Equal.trans(Packing,
      packing_coordinates(input,written,U32.from_nat(1n+index),columns,patch,height,width,stride,padding,next_coordinates(U32.is_eq((column + 1 : U32),columns),row,(column + 1 : U32))),
      packing_coordinates(input,written,U32.from_nat(1n+index),columns,patch,height,width,stride,padding,encoded_pair(Nat.divmod(1n+position,U32.to_nat(columns)))),
      natural_represent(natural_step(state)),
      Equal.cong(U32 & U32,Packing,pair => packing_coordinates(input,written,U32.from_nat(1n+index),columns,patch,height,width,stride,padding,pair),
        next_coordinates(U32.is_eq((column + 1 : U32),columns),row,(column + 1 : U32)),encoded_pair(Nat.divmod(1n+position,U32.to_nat(columns))),coordinate_increment(position,columns,positive)),
      packing_encoded(input,written,U32.from_nat(1n+index),columns,patch,height,width,stride,padding,Nat.divmod(1n+position,U32.to_nat(columns)))))

def natural_loop(+count: Nat,+state: NaturalPacking) ->
  {Loop.run(~Packing,~step,count,natural_represent(state)) == natural_represent(Loop.run(~NaturalPacking,~natural_step,count,state)) : Packing}:
  Projection.iteration_refines_model(~Packing,~NaturalPacking,~step,~natural_step,~natural_represent,~natural_step_matches,count,state)

def natural_output(index: Unit,state: NaturalPacking) -> Tree.Tree<Tile.Tile8>:
  NaturalPacking{input,output,offset,position,columns,patch,height,width,stride,padding,positive} = state
  output

def natural_denote(index: Unit,count: Nat,state: NaturalPacking) -> Tree.Tree<Tile.Tile8>:
  NaturalPacking{+input,output,offset,position,+columns,+patch,+height,+width,+stride,+padding,positive} = state
  Tree.write_sequence(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~PatchContext,~natural_value,count,position,PatchContext{input,columns,patch,height,width,stride,padding}),U32.from_nat(offset),output)

def natural_zero(+index: Unit,+state: NaturalPacking) -> {natural_output(index,state) == natural_denote(index,0n,state) : Tree.Tree<Tile.Tile8>}:
  NaturalPacking{input,output,offset,position,columns,patch,height,width,stride,padding,positive} = state
  {==}

def natural_after(+index: Unit,+count: Nat,+state: NaturalPacking) -> {natural_denote(index,count,natural_step(state)) == natural_denote(index,1n+count,state) : Tree.Tree<Tile.Tile8>}:
  NaturalPacking{+input,+output,+offset,+position,+columns,+patch,+height,+width,+stride,+padding,positive} = state
  %Equal.sym(U32,(U32.from_nat(offset) + 1 : U32),U32.from_nat(1n+offset),Bounds.increment_encoded(offset)) :
    {natural_denote(index,count,natural_step(state)) == Tree.write_sequence(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~PatchContext,~natural_value,count,1n+position,PatchContext{input,columns,patch,height,width,stride,padding}),_,Tree.write(Tile.Tile8,output,U32.from_nat(offset),natural_sample(PatchContext{input,columns,patch,height,width,stride,padding},position))) : Tree.Tree<Tile.Tile8>}
  {==}

def natural_written(+count: Nat,+state: NaturalPacking) -> {natural_output(Unit{},Loop.run(~NaturalPacking,~natural_step,count,state)) == natural_denote(Unit{},count,state) : Tree.Tree<Tile.Tile8>}:
  Projection.iteration_matches_denotation(~Tree.Tree<Tile.Tile8>,~NaturalPacking,~Unit,~natural_step,~natural_output,~natural_denote,~natural_zero,~natural_after,count,Unit{},state)

def packing_output(state: Packing) -> Tree.Tree<Tile.Tile8>:
  Packing{input,output,index,row,column,columns,patch,height,width,stride,padding} = state
  output

def represented_output(+state: NaturalPacking) -> {packing_output(natural_represent(state)) == natural_output(Unit{},state) : Tree.Tree<Tile.Tile8>}:
  NaturalPacking{input,output,index,position,columns,patch,height,width,stride,padding,positive} = state
  {==}

def natural_packing_written(+count: Nat,+state: NaturalPacking) ->
  {packing_output(Loop.run(~Packing,~step,count,natural_represent(state))) == natural_denote(Unit{},count,state) : Tree.Tree<Tile.Tile8>}:
  %Equal.sym(Packing,Loop.run(~Packing,~step,count,natural_represent(state)),natural_represent(Loop.run(~NaturalPacking,~natural_step,count,state)),natural_loop(count,state)) :
    {packing_output(_) == natural_denote(Unit{},count,state) : Tree.Tree<Tile.Tile8>}
  Equal.trans(Tree.Tree<Tile.Tile8>,packing_output(natural_represent(Loop.run(~NaturalPacking,~natural_step,count,state))),natural_output(Unit{},Loop.run(~NaturalPacking,~natural_step,count,state)),natural_denote(Unit{},count,state),represented_output(Loop.run(~NaturalPacking,~natural_step,count,state)),natural_written(count,state))

def window_patch_context(window: Kernel.WeightWindow,input: Buffer.Model<F32>) -> PatchContext:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  PatchContext{input,columns,Kernel.patch(tap,geometry,image),h,w,s,p}

def window_columns(window: Kernel.WeightWindow) -> U32:
  Kernel.WeightWindow{geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  columns

def window_row(window: Kernel.WeightWindow) -> U32:
  Kernel.WeightWindow{geometry,image,tap,output,row,column,first} = window
  row

def window_column(window: Kernel.WeightWindow) -> U32:
  Kernel.WeightWindow{geometry,image,tap,output,row,column,first} = window
  column

def window_input_output_natural(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>,+position: Nat,
  +positive: {Nat.is_lt(0n,U32.to_nat(window_columns(window))) == True{} : Bool},
  +row_match: {window_row(window) == U32.from_nat(Nat.div(position,U32.to_nat(window_columns(window)))) : U32},
  +column_match: {window_column(window) == U32.from_nat(Nat.mod(position,U32.to_nat(window_columns(window)))) : U32}) ->
  {packing_output(window_input(window,count,input)) == Tree.write_sequence(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~PatchContext,~natural_value,U32.to_nat(count),position,window_patch_context(window,input)),0,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0))) : Tree.Tree<Tile.Tile8>}:
  Kernel.WeightWindow{+geometry,image,tap,output,+row,+column,first} = window
  Kernel.Geometry{c,o,+h,+w,k,+s,+p,rows,+columns,taps} = geometry
  %Equal.sym(U32,row,U32.from_nat(Nat.div(position,U32.to_nat(columns))),row_match) :
    {packing_output(Loop.run(~Packing,~step,U32.to_nat(count),Packing{input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0)),0,_,column,columns,Kernel.patch(tap,geometry,image),h,w,s,p})) == Tree.write_sequence(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~PatchContext,~natural_value,U32.to_nat(count),position,window_patch_context(window,input)),0,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0))) : Tree.Tree<Tile.Tile8>}
  %Equal.sym(U32,column,U32.from_nat(Nat.mod(position,U32.to_nat(columns))),column_match) :
    {packing_output(Loop.run(~Packing,~step,U32.to_nat(count),Packing{input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0)),0,U32.from_nat(Nat.div(position,U32.to_nat(columns))),_,columns,Kernel.patch(tap,geometry,image),h,w,s,p})) == Tree.write_sequence(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~PatchContext,~natural_value,U32.to_nat(count),position,window_patch_context(window,input)),0,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0))) : Tree.Tree<Tile.Tile8>}
  natural_packing_written(U32.to_nat(count),NaturalPacking{input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0)),0n,position,columns,Kernel.patch(tap,geometry,image),h,w,s,p,positive})

def window_input_readback_sized(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>,+position: Nat,+index: Nat,
  +positive: {Nat.is_lt(0n,U32.to_nat(window_columns(window))) == True{} : Bool},
  +row_match: {window_row(window) == U32.from_nat(Nat.div(position,U32.to_nat(window_columns(window)))) : U32},
  +column_match: {window_column(window) == U32.from_nat(Nat.mod(position,U32.to_nat(window_columns(window)))) : U32},
  sized: AllocationDepth.Sized<count>,+within: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,packing_output(window_input(window,count,input)),U32.from_nat(index)) == natural_sample(window_patch_context(window,input),Nat.add(position,index)) : Tile.Tile8}:
  AllocationDepth.Sized{+depth_ok,+room} = sized
  +depth = Storage.depth_for(count)
  +tree = Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))
  +balanced = Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0))
  +context = window_patch_context(window,input)
  +values = ListRead.enumerate(~Tile.Tile8,~PatchContext,~natural_value,U32.to_nat(count),position,context)
  +length = ListRead.enumerate_length(~Tile.Tile8,~PatchContext,~natural_value,U32.to_nat(count),position,context)
  %Equal.sym(Tree.Tree<Tile.Tile8>,packing_output(window_input(window,count,input)),Tree.write_sequence(~Tile.Tile8,values,0,tree),window_input_output_natural(window,count,input,position,positive,row_match,column_match)) :
    {Tree.read(Tile.Tile8,_,U32.from_nat(index)) == natural_sample(context,Nat.add(position,index)) : Tile.Tile8}
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Tree.write_sequence(~Tile.Tile8,values,0,tree),U32.from_nat(index)),ListRead.at(~Tile.Tile8,Tile.filled(0.0),index,values),natural_sample(context,Nat.add(position,index)),
    ListRead.listed(~Tile.Tile8,values,index,0n,depth,tree,Tile.filled(0.0),balanced,depth_ok,
      %Equal.sym(U32,Tree.capacity(Tile.Tile8,tree),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,tree,balanced)) : {Nat.is_le(List.length(&2,Tile.Tile8,values),U32.to_nat(_)) == True{} : Bool}
      %Equal.sym(Nat,List.length(&2,Tile.Tile8,values),U32.to_nat(count),length) : {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool} room,
      %Equal.sym(Nat,List.length(&2,Tile.Tile8,values),U32.to_nat(count),length) : {Nat.is_lt(index,_) == True{} : Bool} within),
    ListRead.enumerate_at(~Tile.Tile8,~PatchContext,~natural_value,Tile.filled(0.0),U32.to_nat(count),index,position,context,within))

def window_input_readback(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>,+position: Nat,+index: Nat,
  +positive: {Nat.is_lt(0n,U32.to_nat(window_columns(window))) == True{} : Bool},
  +row_match: {window_row(window) == U32.from_nat(Nat.div(position,U32.to_nat(window_columns(window)))) : U32},
  +column_match: {window_column(window) == U32.from_nat(Nat.mod(position,U32.to_nat(window_columns(window)))) : U32},
  bounded: {U32.is_le(count,256) == True{} : Bool},+within: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,packing_output(window_input(window,count,input)),U32.from_nat(index)) == natural_sample(window_patch_context(window,input),Nat.add(position,index)) : Tile.Tile8}:
  window_input_readback_sized(window,count,input,position,index,positive,row_match,column_match,AllocationDepth.bounded_with(count,256,{==},{==},{==},bounded),within)

# Interval masks use natural cold planning; padding + size never wraps.
def axis_first_encoded(+offset: U32,+stride: U32,+padding: U32,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {Kernel.axis_first(offset,stride,padding) == U32.from_nat(Ceiling.ceiling(Nat.sub(U32.to_nat(padding),U32.to_nat(offset)),U32.to_nat(stride))) : U32}:
  Equal.cong(Nat,U32,U32.from_nat,Nat.div(Nat.add(Nat.sub(U32.to_nat(padding),U32.to_nat(offset)),Nat.sub(U32.to_nat(stride),1n)),U32.to_nat(stride)),Ceiling.ceiling(Nat.sub(U32.to_nat(padding),U32.to_nat(offset)),U32.to_nat(stride)),Ceiling.shifted_formula(Nat.sub(U32.to_nat(padding),U32.to_nat(offset)),U32.to_nat(stride),positive))

def axis_end_encoded(+offset: U32,+stride: U32,+padding: U32,+size: U32,+positions: U32,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {Kernel.axis_end(offset,stride,padding,size,positions) == U32.from_nat(Nat.min(U32.to_nat(positions),Ceiling.ceiling(Nat.sub(Nat.add(U32.to_nat(padding),U32.to_nat(size)),U32.to_nat(offset)),U32.to_nat(stride)))) : U32}:
  Equal.cong(Nat,U32,value => U32.from_nat(Nat.min(U32.to_nat(positions),value)),
    Nat.div(Nat.add(Nat.sub(Nat.add(U32.to_nat(padding),U32.to_nat(size)),U32.to_nat(offset)),Nat.sub(U32.to_nat(stride),1n)),U32.to_nat(stride)),
    Ceiling.ceiling(Nat.sub(Nat.add(U32.to_nat(padding),U32.to_nat(size)),U32.to_nat(offset)),U32.to_nat(stride)),
    Ceiling.shifted_formula(Nat.sub(Nat.add(U32.to_nat(padding),U32.to_nat(size)),U32.to_nat(offset)),U32.to_nat(stride),positive))

def axis_first_decoded(+offset: U32,+stride: U32,+padding: U32,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {U32.to_nat(Kernel.axis_first(offset,stride,padding)) == Ceiling.ceiling(Nat.sub(U32.to_nat(padding),U32.to_nat(offset)),U32.to_nat(stride)) : Nat}:
  +difference = Nat.sub(U32.to_nat(padding),U32.to_nat(offset))
  +first = Ceiling.ceiling(difference,U32.to_nat(stride))
  %Equal.sym(U32,Kernel.axis_first(offset,stride,padding),U32.from_nat(first),axis_first_encoded(offset,stride,padding,positive)) : {U32.to_nat(_) == first : Nat}
  FullRange.roundtrip(first,padding,Order.transitive(first,difference,U32.to_nat(padding),Ceiling.bounded(difference,U32.to_nat(stride),positive),Order.sub_le(U32.to_nat(padding),U32.to_nat(offset))))

def axis_end_decoded(+offset: U32,+stride: U32,+padding: U32,+size: U32,+positions: U32,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {U32.to_nat(Kernel.axis_end(offset,stride,padding,size,positions)) == Nat.min(U32.to_nat(positions),Ceiling.ceiling(Nat.sub(Nat.add(U32.to_nat(padding),U32.to_nat(size)),U32.to_nat(offset)),U32.to_nat(stride))) : Nat}:
  +ceiling = Ceiling.ceiling(Nat.sub(Nat.add(U32.to_nat(padding),U32.to_nat(size)),U32.to_nat(offset)),U32.to_nat(stride))
  +end = Nat.min(U32.to_nat(positions),ceiling)
  %Equal.sym(U32,Kernel.axis_end(offset,stride,padding,size,positions),U32.from_nat(end),axis_end_encoded(offset,stride,padding,size,positions,positive)) : {U32.to_nat(_) == end : Nat}
  FullRange.roundtrip(end,positions,Extrema.minimum_left(U32.to_nat(positions),ceiling))

def axis_mask(+offset: U32,+stride: U32,+padding: U32,+size: U32,+positions: U32,+position: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+within: {Nat.is_lt(U32.to_nat(position),U32.to_nat(positions)) == True{} : Bool}) ->
  {U32.is_le(Kernel.axis_first(offset,stride,padding),position) && U32.is_lt(position,Kernel.axis_end(offset,stride,padding,size,positions)) ==
    Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) &&
    Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size))) : Bool}:
  +first = Kernel.axis_first(offset,stride,padding)
  +end = Kernel.axis_end(offset,stride,padding,size,positions)
  +begin = Nat.sub(U32.to_nat(padding),U32.to_nat(offset))
  +limit = Nat.sub(Nat.add(U32.to_nat(padding),U32.to_nat(size)),U32.to_nat(offset))
  %Equal.sym(Bool,U32.is_le(first,position),Nat.is_le(U32.to_nat(first),U32.to_nat(position)),Comparison.le(first,position)) :
    {_ && U32.is_lt(position,end) == Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) && Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size))) : Bool}
  %Equal.sym(Bool,U32.is_lt(position,end),Nat.is_lt(U32.to_nat(position),U32.to_nat(end)),Comparison.lt(position,end)) :
    {Nat.is_le(U32.to_nat(first),U32.to_nat(position)) && _ == Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) && Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size))) : Bool}
  %Equal.sym(Nat,U32.to_nat(first),Ceiling.ceiling(begin,U32.to_nat(stride)),axis_first_decoded(offset,stride,padding,positive)) :
    {Nat.is_le(_,U32.to_nat(position)) && Nat.is_lt(U32.to_nat(position),U32.to_nat(end)) == Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) && Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size))) : Bool}
  %Equal.sym(Nat,U32.to_nat(end),Nat.min(U32.to_nat(positions),Ceiling.ceiling(limit,U32.to_nat(stride))),axis_end_decoded(offset,stride,padding,size,positions,positive)) :
    {Nat.is_le(Ceiling.ceiling(begin,U32.to_nat(stride)),U32.to_nat(position)) && Nat.is_lt(U32.to_nat(position),_) == Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) && Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size))) : Bool}
  %Equal.sym(Bool,Nat.is_lt(U32.to_nat(position),Nat.min(U32.to_nat(positions),Ceiling.ceiling(limit,U32.to_nat(stride)))),Nat.is_lt(U32.to_nat(position),U32.to_nat(positions)) && Nat.is_lt(U32.to_nat(position),Ceiling.ceiling(limit,U32.to_nat(stride))),Extrema.below_minimum(U32.to_nat(position),U32.to_nat(positions),Ceiling.ceiling(limit,U32.to_nat(stride)))) :
    {Nat.is_le(Ceiling.ceiling(begin,U32.to_nat(stride)),U32.to_nat(position)) && _ == Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) && Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size))) : Bool}
  %Equal.sym(Bool,Nat.is_lt(U32.to_nat(position),U32.to_nat(positions)),True{},within) :
    {Nat.is_le(Ceiling.ceiling(begin,U32.to_nat(stride)),U32.to_nat(position)) && (_ && Nat.is_lt(U32.to_nat(position),Ceiling.ceiling(limit,U32.to_nat(stride)))) == Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) && Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size))) : Bool}
  %Equal.sym(Bool,Nat.is_lt(U32.to_nat(position),Ceiling.ceiling(limit,U32.to_nat(stride))),Nat.is_lt(Nat.mul(U32.to_nat(position),U32.to_nat(stride)),limit),Ceiling.upper_comparison(limit,U32.to_nat(stride),U32.to_nat(position),positive)) :
    {Nat.is_le(Ceiling.ceiling(begin,U32.to_nat(stride)),U32.to_nat(position)) && _ == Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) && Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size))) : Bool}
  %Equal.sym(Bool,Nat.is_lt(Nat.mul(U32.to_nat(position),U32.to_nat(stride)),limit),Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size))),Order.subtraction_lt(Nat.add(U32.to_nat(padding),U32.to_nat(size)),U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) :
    {Nat.is_le(Ceiling.ceiling(begin,U32.to_nat(stride)),U32.to_nat(position)) && _ == Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) && Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size))) : Bool}
  Equal.cong(Bool,Bool,value => value && Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size))),
    Nat.is_le(Ceiling.ceiling(begin,U32.to_nat(stride)),U32.to_nat(position)),
    Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))),
    Equal.trans(Bool,Nat.is_le(Ceiling.ceiling(begin,U32.to_nat(stride)),U32.to_nat(position)),Nat.is_le(begin,Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))),Ceiling.lower_comparison(begin,U32.to_nat(stride),U32.to_nat(position),positive),Order.subtraction_le(U32.to_nat(padding),U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))))

# Finite hardware lanes project the existing eight-sample tile reader.
def tile_value_at(~Context: Data,~sample: Buffer.Model<F32> -> Context -> Nat -> F32,+input: Buffer.Model<F32>,+context: Context,index: Nat,
  within: {Nat.is_lt(index,8n) == True{} : Bool}) ->
  {Tiles.select_lane(index,tile_value(~Context,~sample,input,context)) == sample(input,context,index) : F32}:
  match index:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+ +rest: Empty.absurd({Tiles.select_lane(8n+rest,tile_value(~Context,~sample,input,context)) == sample(input,context,8n+rest) : F32},Words.false_true(%Order.not_below(0n,rest,Order.zero_le(rest)) : {_ == True{} : Bool} within))

def patch_lane_at(+tap: U32,+geometry: Kernel.Geometry,+image: U32,index: Nat,within: {Nat.is_lt(index,8n) == True{} : Bool}) ->
  {Kernel.selected(Kernel.patch(tap,geometry,image),index) == Kernel.lane((tap + U32.from_nat(index) : U32),geometry,image) : Kernel.Lane}:
  match index:
    case 0n:
      %Equal.sym(U32,(tap + 0 : U32),tap,Arithmetic.add_zero(tap)) : {Kernel.lane(tap,geometry,image) == Kernel.lane(_,geometry,image) : Kernel.Lane}
      {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+ +rest: Empty.absurd({Kernel.selected(Kernel.patch(tap,geometry,image),8n+rest) == Kernel.lane((tap + U32.from_nat(8n+rest) : U32),geometry,image) : Kernel.Lane},Words.false_true(%Order.not_below(0n,rest,Order.zero_le(rest)) : {_ == True{} : Bool} within))

def half_tile_value_at(upper: Bool,row: Accumulator.Row,+input: Buffer.Model<F32>,+context: Kernel.Pixel) ->
  {Accumulator.row(~F32,row,Rows.half(upper,tile_value(~Kernel.Pixel,~patch_sample,input,context))) == patch_sample(input,context,Nat.add(Bool.pick(Nat,upper,4n,0n),Coordinates.row_number(row))) : F32}:
  match upper row:
    case False{} Accumulator.R0{}: {==}
    case False{} Accumulator.R1{}: {==}
    case False{} Accumulator.R2{}: {==}
    case False{} Accumulator.R3{}: {==}
    case True{} Accumulator.R0{}: {==}
    case True{} Accumulator.R1{}: {==}
    case True{} Accumulator.R2{}: {==}
    case True{} Accumulator.R3{}: {==}

def gradient_tile_element(+input: Buffer.Model<F32>,+context: Kernel.Gradient,+position: U32,+lane: Nat,+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Tile.element_at(Tiles.lane_of(lane),gradient_tile_value(input,context,position)) == gradient_value(input,context,lane,position) : F32}:
  +tile = gradient_tile_value(input,context,position)
  Equal.trans(F32,Tile.element_at(Tiles.lane_of(lane),tile),Tiles.select_lane(lane,tile),gradient_value(input,context,lane,position),
    Equal.sym(F32,Tiles.select_lane(lane,tile),Tile.element_at(Tiles.lane_of(lane),tile),Tiles.selected_lane(lane,tile)),
    tile_value_at(~Kernel.GradientRead,~gradient_sample,input,Kernel.GradientRead{context,position},lane,within))

def natural_patch_value(context: PatchContext,upper: Bool,row: Accumulator.Row,+position: Nat) -> F32:
  PatchContext{input,+columns,patch,height,width,stride,padding} = context
  patch_sample(input,Kernel.Pixel{patch,U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))),height,width,stride,padding},Nat.add(Bool.pick(Nat,upper,4n,0n),Coordinates.row_number(row)))

def natural_patch_element(+context: PatchContext,+upper: Bool,+row: Accumulator.Row,+position: Nat) ->
  {Accumulator.row(~F32,row,Rows.half(upper,natural_sample(context,position))) == natural_patch_value(context,upper,row,position) : F32}:
  PatchContext{+input,+columns,+patch,+height,+width,+stride,+padding} = context
  half_tile_value_at(upper,row,input,Kernel.Pixel{patch,U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))),height,width,stride,padding})

# A chunk returns the same row-major cursor at its next unconsumed position.
def natural_position(state: NaturalPacking) -> Nat:
  NaturalPacking{input,output,index,position,columns,patch,height,width,stride,padding,positive} = state
  position

def natural_columns(state: NaturalPacking) -> U32:
  NaturalPacking{input,output,index,position,columns,patch,height,width,stride,padding,positive} = state
  columns

def natural_position_step(+state: NaturalPacking) -> {natural_position(natural_step(state)) == 1n+natural_position(state) : Nat}:
  NaturalPacking{input,output,index,position,columns,patch,height,width,stride,padding,positive} = state
  {==}

def natural_columns_step(+state: NaturalPacking) -> {natural_columns(natural_step(state)) == natural_columns(state) : U32}:
  NaturalPacking{input,output,index,position,columns,patch,height,width,stride,padding,positive} = state
  {==}

def natural_position_end(+count: Nat,+state: NaturalPacking) -> {natural_position(Loop.run(~NaturalPacking,~natural_step,count,state)) == Nat.add(natural_position(state),count) : Nat}:
  %Order.commutative(count,natural_position(state)) : {natural_position(Loop.run(~NaturalPacking,~natural_step,count,state)) == _ : Nat}
  Projection.natural_counter_progress(~NaturalPacking,~natural_step,~natural_position,~natural_position_step,count,state)

def natural_columns_end(+count: Nat,+state: NaturalPacking) -> {natural_columns(Loop.run(~NaturalPacking,~natural_step,count,state)) == natural_columns(state) : U32}:
  Projection.preserves_observation(~NaturalPacking,~U32,~natural_step,~natural_columns,~natural_columns_step,count,state)

def packing_row(state: Packing) -> U32:
  Packing{input,output,index,row,column,columns,patch,height,width,stride,padding} = state
  row

def packing_column(state: Packing) -> U32:
  Packing{input,output,index,row,column,columns,patch,height,width,stride,padding} = state
  column

def natural_row_represented(+state: NaturalPacking) -> {packing_row(natural_represent(state)) == U32.from_nat(Nat.div(natural_position(state),U32.to_nat(natural_columns(state)))) : U32}:
  NaturalPacking{input,output,index,position,columns,patch,height,width,stride,padding,positive} = state
  {==}

def natural_column_represented(+state: NaturalPacking) -> {packing_column(natural_represent(state)) == U32.from_nat(Nat.mod(natural_position(state),U32.to_nat(natural_columns(state)))) : U32}:
  NaturalPacking{input,output,index,position,columns,patch,height,width,stride,padding,positive} = state
  {==}

def packing_row_end(+count: Nat,+state: NaturalPacking) ->
  {packing_row(Loop.run(~Packing,~step,count,natural_represent(state))) == U32.from_nat(Nat.div(Nat.add(natural_position(state),count),U32.to_nat(natural_columns(state)))) : U32}:
  +result = Loop.run(~NaturalPacking,~natural_step,count,state)
  %Equal.sym(Packing,Loop.run(~Packing,~step,count,natural_represent(state)),natural_represent(result),natural_loop(count,state)) : {packing_row(_) == U32.from_nat(Nat.div(Nat.add(natural_position(state),count),U32.to_nat(natural_columns(state)))) : U32}
  %Equal.sym(U32,packing_row(natural_represent(result)),U32.from_nat(Nat.div(natural_position(result),U32.to_nat(natural_columns(result)))),natural_row_represented(result)) : {_ == U32.from_nat(Nat.div(Nat.add(natural_position(state),count),U32.to_nat(natural_columns(state)))) : U32}
  %Equal.sym(Nat,natural_position(result),Nat.add(natural_position(state),count),natural_position_end(count,state)) : {U32.from_nat(Nat.div(_,U32.to_nat(natural_columns(result)))) == U32.from_nat(Nat.div(Nat.add(natural_position(state),count),U32.to_nat(natural_columns(state)))) : U32}
  %Equal.sym(U32,natural_columns(result),natural_columns(state),natural_columns_end(count,state)) : {U32.from_nat(Nat.div(Nat.add(natural_position(state),count),U32.to_nat(_))) == U32.from_nat(Nat.div(Nat.add(natural_position(state),count),U32.to_nat(natural_columns(state)))) : U32}
  {==}

def packing_column_end(+count: Nat,+state: NaturalPacking) ->
  {packing_column(Loop.run(~Packing,~step,count,natural_represent(state))) == U32.from_nat(Nat.mod(Nat.add(natural_position(state),count),U32.to_nat(natural_columns(state)))) : U32}:
  +result = Loop.run(~NaturalPacking,~natural_step,count,state)
  %Equal.sym(Packing,Loop.run(~Packing,~step,count,natural_represent(state)),natural_represent(result),natural_loop(count,state)) : {packing_column(_) == U32.from_nat(Nat.mod(Nat.add(natural_position(state),count),U32.to_nat(natural_columns(state)))) : U32}
  %Equal.sym(U32,packing_column(natural_represent(result)),U32.from_nat(Nat.mod(natural_position(result),U32.to_nat(natural_columns(result)))),natural_column_represented(result)) : {_ == U32.from_nat(Nat.mod(Nat.add(natural_position(state),count),U32.to_nat(natural_columns(state)))) : U32}
  %Equal.sym(Nat,natural_position(result),Nat.add(natural_position(state),count),natural_position_end(count,state)) : {U32.from_nat(Nat.mod(_,U32.to_nat(natural_columns(result)))) == U32.from_nat(Nat.mod(Nat.add(natural_position(state),count),U32.to_nat(natural_columns(state)))) : U32}
  %Equal.sym(U32,natural_columns(result),natural_columns(state),natural_columns_end(count,state)) : {U32.from_nat(Nat.mod(Nat.add(natural_position(state),count),U32.to_nat(_))) == U32.from_nat(Nat.mod(Nat.add(natural_position(state),count),U32.to_nat(natural_columns(state)))) : U32}
  {==}

type NumericPixel is Data:
  NumericPixel{input: Buffer.Model<F32>,base: Nat,row: U32,column: U32,kernel_row: U32,kernel_column: U32,height: U32,width: U32,stride: U32,padding: U32,rows: U32,columns: U32}

def numeric_native_mask(pixel: NumericPixel) -> Bool:
  NumericPixel{input,base,+row,+column,+kr,+kc,+h,+w,+s,+p,rows,columns} = pixel
  U32.is_le(Kernel.axis_first(kr,s,p),row) && U32.is_lt(row,Kernel.axis_end(kr,s,p,h,rows)) && U32.is_le(Kernel.axis_first(kc,s,p),column) && U32.is_lt(column,Kernel.axis_end(kc,s,p,w,columns))

def axis_inside(+offset: U32,+stride: U32,+padding: U32,size: U32,+position: U32) -> Bool:
  Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) && Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size)))

def numeric_inside(pixel: NumericPixel) -> Bool:
  NumericPixel{input,base,row,column,kr,kc,h,w,+s,+p,rows,columns} = pixel
  axis_inside(kr,s,p,h,row) && axis_inside(kc,s,p,w,column)

def numeric_stride(pixel: NumericPixel) -> U32:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  s

def numeric_row_within(pixel: NumericPixel) -> Bool:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  Nat.is_lt(U32.to_nat(row),U32.to_nat(rows))

def numeric_column_within(pixel: NumericPixel) -> Bool:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  Nat.is_lt(U32.to_nat(column),U32.to_nat(columns))

type PixelReady<-pixel: NumericPixel> is Data:
  PixelReady{positive: {Nat.is_lt(0n,U32.to_nat(numeric_stride(pixel))) == True{} : Bool},row: {numeric_row_within(pixel) == True{} : Bool},column: {numeric_column_within(pixel) == True{} : Bool}}

def numeric_mask_matches(+pixel: NumericPixel,ready: PixelReady<pixel>) -> {numeric_native_mask(pixel) == numeric_inside(pixel) : Bool}:
  NumericPixel{input,base,+row,+column,+kr,+kc,+h,+w,+s,+p,+rows,+columns} = pixel
  PixelReady{+positive,row_within,column_within} = ready
  +rf = U32.is_le(Kernel.axis_first(kr,s,p),row)
  +re = U32.is_lt(row,Kernel.axis_end(kr,s,p,h,rows))
  +cf = U32.is_le(Kernel.axis_first(kc,s,p),column)
  +ce = U32.is_lt(column,Kernel.axis_end(kc,s,p,w,columns))
  %Equal.sym(Bool,((rf && re) && cf) && ce,(rf && re) && (cf && ce),Logic.and_associative((rf && re),cf,ce)) : {_ == numeric_inside(pixel) : Bool}
  %Equal.sym(Bool,rf && re,axis_inside(kr,s,p,h,row),axis_mask(kr,s,p,h,rows,row,positive,row_within)) : {_ && (cf && ce) == numeric_inside(pixel) : Bool}
  %Equal.sym(Bool,cf && ce,axis_inside(kc,s,p,w,column),axis_mask(kc,s,p,w,columns,column,positive,column_within)) : {axis_inside(kr,s,p,h,row) && _ == numeric_inside(pixel) : Bool}
  {==}

def numeric_native_index(pixel: NumericPixel) -> U32:
  NumericPixel{input,base,row,column,kr,kc,height,+width,+stride,+padding,rows,columns} = pixel
  (U32.from_nat(base) + (row * stride + kr - padding) * width + (column * stride + kc - padding) : U32)

def numeric_index(pixel: NumericPixel) -> Nat:
  NumericPixel{input,base,row,column,kr,kc,height,width,+stride,+padding,rows,columns} = pixel
  Nat.add(Nat.add(base,Nat.mul(Nat.sub(Nat.add(U32.to_nat(kr),Nat.mul(U32.to_nat(row),U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(width))),Nat.sub(Nat.add(U32.to_nat(kc),Nat.mul(U32.to_nat(column),U32.to_nat(stride))),U32.to_nat(padding)))

def numeric_tree(pixel: NumericPixel) -> Tree.Tree<F32>:
  NumericPixel{input,base,row,column,kr,kc,height,width,stride,padding,rows,columns} = pixel
  Buffer.tree(~F32,input)

def numeric_index_matches(+pixel: NumericPixel,+inside: {numeric_inside(pixel) == True{} : Bool}) -> {numeric_native_index(pixel) == U32.from_nat(numeric_index(pixel)) : U32}:
  NumericPixel{input,+base,+row,+column,+kr,+kc,+height,+width,+stride,+padding,rows,columns} = pixel
  +row_inside = Logic.left(axis_inside(kr,stride,padding,height,row),axis_inside(kc,stride,padding,width,column),inside)
  +column_inside = Logic.right(axis_inside(kr,stride,padding,height,row),axis_inside(kc,stride,padding,width,column),inside)
  Encoded.padded_pixel_address(base,row,column,kr,kc,width,stride,padding,
    Logic.left(Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(kr),Nat.mul(U32.to_nat(row),U32.to_nat(stride)))),Nat.is_lt(Nat.add(U32.to_nat(kr),Nat.mul(U32.to_nat(row),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(height))),row_inside),
    Logic.left(Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(kc),Nat.mul(U32.to_nat(column),U32.to_nat(stride)))),Nat.is_lt(Nat.add(U32.to_nat(kc),Nat.mul(U32.to_nat(column),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(width))),column_inside))

def numeric_selected(inside: Bool,+pixel: NumericPixel,equation: {numeric_inside(pixel) == inside : Bool}) ->
  {Bool.pick(F32,inside,Tree.read(F32,numeric_tree(pixel),numeric_native_index(pixel)),0.0) == Bool.pick(F32,inside,Tree.read(F32,numeric_tree(pixel),U32.from_nat(numeric_index(pixel))),0.0) : F32}:
  match inside:
    case False{}: {==}
    case True{}:
      Equal.cong(U32,F32,index => Tree.read(F32,numeric_tree(pixel),index),numeric_native_index(pixel),U32.from_nat(numeric_index(pixel)),numeric_index_matches(pixel,equation))

def numeric_native_value(pixel: NumericPixel) -> F32:
  NumericPixel{input,base,row,column,+kr,+kc,+h,+w,+s,+p,rows,columns} = pixel
  pixel_value(input,row,column,h,w,s,p,Kernel.Lane{U32.from_nat(base),kr,kc,True{},Kernel.axis_first(kr,s,p),Kernel.axis_end(kr,s,p,h,rows),Kernel.axis_first(kc,s,p),Kernel.axis_end(kc,s,p,w,columns)})

def numeric_value(pixel: NumericPixel) -> F32:
  +pixel = pixel
  Bool.pick(F32,numeric_inside(pixel),Tree.read(F32,numeric_tree(pixel),U32.from_nat(numeric_index(pixel))),0.0)

def numeric_native_expanded(+pixel: NumericPixel) -> {numeric_native_value(pixel) == Bool.pick(F32,numeric_native_mask(pixel),Tree.read(F32,numeric_tree(pixel),numeric_native_index(pixel)),0.0) : F32}:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  {==}

def numeric_pixel_matches(+pixel: NumericPixel,+ready: PixelReady<pixel>) -> {numeric_native_value(pixel) == numeric_value(pixel) : F32}:
  %Equal.sym(F32,numeric_native_value(pixel),Bool.pick(F32,numeric_native_mask(pixel),Tree.read(F32,numeric_tree(pixel),numeric_native_index(pixel)),0.0),numeric_native_expanded(pixel)) : {_ == numeric_value(pixel) : F32}
  %Equal.sym(Bool,numeric_native_mask(pixel),numeric_inside(pixel),numeric_mask_matches(pixel,ready)) : {Bool.pick(F32,_,Tree.read(F32,numeric_tree(pixel),numeric_native_index(pixel)),0.0) == numeric_value(pixel) : F32}
  numeric_selected(numeric_inside(pixel),pixel,{==})

# Cold tap decoding uses the existing radix and checked allocation bounds.
def kernel_bound(+limits: Machine.Limits,+kernel: U32,+positive: {Nat.is_lt(0n,U32.to_nat(kernel)) == True{} : Bool},+square_bound: {Nat.is_le(Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(kernel),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(U32.to_nat(kernel),Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(kernel),U32.to_nat(kernel),positive),square_bound)

def kernel_square_exact(+limits: Machine.Limits,+kernel: U32,+square_bound: {Nat.is_le(Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((kernel * kernel : U32)) == Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)) : Nat}:
  Arithmetic.multiply(kernel,kernel,Machine.capacity(limits),Limits.capacity_fits(limits),square_bound)

def kernel_square_positive(+limits: Machine.Limits,+kernel: U32,+positive: {Nat.is_lt(0n,U32.to_nat(kernel)) == True{} : Bool},+square_bound: {Nat.is_le(Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat((kernel * kernel : U32))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((kernel * kernel : U32)),Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)),kernel_square_exact(limits,kernel,square_bound)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Bounds.nat_trans(0n,U32.to_nat(kernel),Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)),positive,RowMajor.product_covers(U32.to_nat(kernel),U32.to_nat(kernel),positive))

def kernel_square_bound(+limits: Machine.Limits,+kernel: U32,+square_bound: {Nat.is_le(Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat((kernel * kernel : U32)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((kernel * kernel : U32)),Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)),kernel_square_exact(limits,kernel,square_bound)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  square_bound

def tap_row_decoded(+limits: Machine.Limits,+tap: U32,+kernel: U32,+positive: {Nat.is_lt(0n,U32.to_nat(kernel)) == True{} : Bool},+square_bound: {Nat.is_le(Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((tap / kernel % kernel : U32)) == Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(kernel)),U32.to_nat(kernel)) : Nat}:
  +bound = kernel_bound(limits,kernel,positive,square_bound)
  +doubled = Limits.double_bound(limits,U32.to_nat(kernel),bound)
  +q = (tap / kernel : U32)
  Equal.trans(Nat,U32.to_nat((q % kernel : U32)),Nat.mod(U32.to_nat(q),U32.to_nat(kernel)),Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(kernel)),U32.to_nat(kernel)),
    NativeDivision.modulo(q,kernel,Machine.half_word(limits),Limits.half_word_fits(limits),doubled,positive),
    Equal.cong(Nat,Nat,value => Nat.mod(value,U32.to_nat(kernel)),U32.to_nat(q),Nat.div(U32.to_nat(tap),U32.to_nat(kernel)),NativeDivision.divide(tap,kernel,Machine.half_word(limits),Limits.half_word_fits(limits),doubled,positive)))

def tap_column_decoded(+limits: Machine.Limits,+tap: U32,+kernel: U32,+positive: {Nat.is_lt(0n,U32.to_nat(kernel)) == True{} : Bool},+square_bound: {Nat.is_le(Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((tap % kernel : U32)) == Nat.mod(U32.to_nat(tap),U32.to_nat(kernel)) : Nat}:
  NativeDivision.modulo(tap,kernel,Machine.half_word(limits),Limits.half_word_fits(limits),Limits.double_bound(limits,U32.to_nat(kernel),kernel_bound(limits,kernel,positive,square_bound)),positive)

def tap_channel_decoded(+limits: Machine.Limits,+tap: U32,+kernel: U32,+positive: {Nat.is_lt(0n,U32.to_nat(kernel)) == True{} : Bool},+square_bound: {Nat.is_le(Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((tap / (kernel * kernel) : U32)) == Nat.div(U32.to_nat(tap),Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel))) : Nat}:
  %kernel_square_exact(limits,kernel,square_bound) : {U32.to_nat((tap / (kernel * kernel) : U32)) == Nat.div(U32.to_nat(tap),_) : Nat}
  NativeDivision.divide(tap,(kernel * kernel : U32),Machine.half_word(limits),Limits.half_word_fits(limits),Limits.double_bound(limits,U32.to_nat((kernel * kernel : U32)),kernel_square_bound(limits,kernel,square_bound)),kernel_square_positive(limits,kernel,positive,square_bound))

def geometry_kernel(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  k

def tap_base(geometry: Kernel.Geometry,image: U32,tap: U32) -> Nat:
  Kernel.Geometry{c,o,h,w,+k,s,p,rows,columns,taps} = geometry
  Nat.mul(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(c)),Nat.div(U32.to_nat(tap),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(h)),U32.to_nat(w))

def tap_native_base(geometry: Kernel.Geometry,image: U32,tap: U32) -> U32:
  Kernel.Geometry{c,o,h,w,+k,s,p,rows,columns,taps} = geometry
  ((image * c + tap / (k * k)) * h * w : U32)

def tap_base_matches(+limits: Machine.Limits,+geometry: Kernel.Geometry,+image: U32,+tap: U32,+positive: {Nat.is_lt(0n,U32.to_nat(geometry_kernel(geometry))) == True{} : Bool},+square_bound: {Nat.is_le(Nat.mul(U32.to_nat(geometry_kernel(geometry)),U32.to_nat(geometry_kernel(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {tap_native_base(geometry,image,tap) == U32.from_nat(tap_base(geometry,image,tap)) : U32}:
  Kernel.Geometry{+c,o,+h,+w,+k,s,p,rows,columns,taps} = geometry
  +channel = (tap / (k * k) : U32)
  +natural_channel = Nat.div(U32.to_nat(tap),Nat.mul(U32.to_nat(k),U32.to_nat(k)))
  Equal.trans(U32,((image * c + channel) * h * w : U32),U32.from_nat(Nat.mul(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(c)),U32.to_nat(channel)),U32.to_nat(h)),U32.to_nat(w))),U32.from_nat(tap_base(geometry,image,tap)),
    Encoded.channel_base_encoded(image,c,channel,h,w),
    Equal.cong(Nat,U32,value => U32.from_nat(Nat.mul(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(c)),value),U32.to_nat(h)),U32.to_nat(w))),U32.to_nat(channel),natural_channel,tap_channel_decoded(limits,tap,k,positive,square_bound)))

def tap_numeric_pixel(input: Buffer.Model<F32>,+geometry: Kernel.Geometry,+image: U32,+tap: U32,row: U32,column: U32) -> NumericPixel:
  Kernel.Geometry{c,o,h,w,+k,s,p,rows,columns,taps} = geometry
  NumericPixel{input,tap_base(geometry,image,tap),row,column,(tap / k % k : U32),(tap % k : U32),h,w,s,p,rows,columns}

def geometry_taps(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  taps

def tap_pixel_value(input: Buffer.Model<F32>,+geometry: Kernel.Geometry,image: U32,tap: U32,row: U32,column: U32) -> F32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  pixel_value(input,row,column,h,w,s,p,Kernel.lane(tap,geometry,image))

def tap_pixel_matches(+limits: Machine.Limits,+input: Buffer.Model<F32>,+geometry: Kernel.Geometry,+image: U32,+tap: U32,+row: U32,+column: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(geometry_kernel(geometry))) == True{} : Bool},
  +square_bound: {Nat.is_le(Nat.mul(U32.to_nat(geometry_kernel(geometry)),U32.to_nat(geometry_kernel(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +tap_inside: {U32.is_lt(tap,geometry_taps(geometry)) == True{} : Bool},
  +ready: PixelReady<tap_numeric_pixel(input,geometry,image,tap,row,column)>) ->
  {tap_pixel_value(input,geometry,image,tap,row,column) == numeric_value(tap_numeric_pixel(input,geometry,image,tap,row,column)) : F32}:
  Kernel.Geometry{+c,o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +kr = (tap / k % k : U32)
  +kc = (tap % k : U32)
  %Equal.sym(Bool,U32.is_lt(tap,taps),True{},tap_inside) :
    {pixel_value(input,row,column,h,w,s,p,Kernel.Lane{tap_native_base(geometry,image,tap),kr,kc,_,Kernel.axis_first(kr,s,p),Kernel.axis_end(kr,s,p,h,rows),Kernel.axis_first(kc,s,p),Kernel.axis_end(kc,s,p,w,columns)}) == numeric_value(tap_numeric_pixel(input,geometry,image,tap,row,column)) : F32}
  %Equal.sym(U32,tap_native_base(geometry,image,tap),U32.from_nat(tap_base(geometry,image,tap)),tap_base_matches(limits,geometry,image,tap,positive,square_bound)) :
    {pixel_value(input,row,column,h,w,s,p,Kernel.Lane{_,kr,kc,True{},Kernel.axis_first(kr,s,p),Kernel.axis_end(kr,s,p,h,rows),Kernel.axis_first(kc,s,p),Kernel.axis_end(kc,s,p,w,columns)}) == numeric_value(tap_numeric_pixel(input,geometry,image,tap,row,column)) : F32}
  numeric_pixel_matches(tap_numeric_pixel(input,geometry,image,tap,row,column),ready)

# The scalar address model is used only in proofs; integer products may wrap
# in the native reader while its final encoded address remains the same.
def sampled_pixel(input: Tree.Tree<F32>,base: Nat,row: Nat,column: Nat,kr: Nat,kc: Nat,height: Nat,width: Nat,+stride: Nat,padding: Nat) -> F32:
  +y = Nat.add(kr,Nat.mul(row,stride))
  +x = Nat.add(kc,Nat.mul(column,stride))
  +padding = padding
  +width = width
  Bool.pick(F32,(Nat.is_le(padding,y) && Nat.is_lt(y,Nat.add(padding,height))) && (Nat.is_le(padding,x) && Nat.is_lt(x,Nat.add(padding,width))),
    Tree.read(F32,input,U32.from_nat(Nat.add(Nat.add(base,Nat.mul(Nat.sub(y,padding),width)),Nat.sub(x,padding)))),0.0)

def numeric_base(pixel: NumericPixel) -> Nat:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  base

def numeric_row(pixel: NumericPixel) -> U32:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  row

def numeric_column(pixel: NumericPixel) -> U32:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  column

def numeric_kr(pixel: NumericPixel) -> U32:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  kr

def numeric_kc(pixel: NumericPixel) -> U32:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  kc

def numeric_height(pixel: NumericPixel) -> U32:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  h

def numeric_width(pixel: NumericPixel) -> U32:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  w

def numeric_padding(pixel: NumericPixel) -> U32:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  p

def numeric_sampled(+pixel: NumericPixel) ->
  {numeric_value(pixel) == sampled_pixel(numeric_tree(pixel),numeric_base(pixel),U32.to_nat(numeric_row(pixel)),U32.to_nat(numeric_column(pixel)),U32.to_nat(numeric_kr(pixel)),U32.to_nat(numeric_kc(pixel)),U32.to_nat(numeric_height(pixel)),U32.to_nat(numeric_width(pixel)),U32.to_nat(numeric_stride(pixel)),U32.to_nat(numeric_padding(pixel))) : F32}:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,rows,columns} = pixel
  {==}

def natural_tap_value(input: Buffer.Model<F32>,+geometry: Kernel.Geometry,+image: U32,+tap: U32,row: U32,column: U32) -> F32:
  Kernel.Geometry{c,o,h,w,+k,s,p,rows,columns,taps} = geometry
  sampled_pixel(Buffer.tree(~F32,input),tap_base(geometry,image,tap),U32.to_nat(row),U32.to_nat(column),Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(k)),Nat.mod(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(h),U32.to_nat(w),U32.to_nat(s),U32.to_nat(p))

def tap_pixel_natural(+limits: Machine.Limits,+input: Buffer.Model<F32>,+geometry: Kernel.Geometry,+image: U32,+tap: U32,+row: U32,+column: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(geometry_kernel(geometry))) == True{} : Bool},
  +square_bound: {Nat.is_le(Nat.mul(U32.to_nat(geometry_kernel(geometry)),U32.to_nat(geometry_kernel(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +tap_inside: {U32.is_lt(tap,geometry_taps(geometry)) == True{} : Bool},
  +ready: PixelReady<tap_numeric_pixel(input,geometry,image,tap,row,column)>) ->
  {tap_pixel_value(input,geometry,image,tap,row,column) == natural_tap_value(input,geometry,image,tap,row,column) : F32}:
  Kernel.Geometry{c,o,+h,+w,+k,+s,+p,rows,columns,taps} = geometry
  %Equal.sym(F32,tap_pixel_value(input,geometry,image,tap,row,column),numeric_value(tap_numeric_pixel(input,geometry,image,tap,row,column)),tap_pixel_matches(limits,input,geometry,image,tap,row,column,positive,square_bound,tap_inside,ready)) : {_ == natural_tap_value(input,geometry,image,tap,row,column) : F32}
  %Equal.sym(F32,numeric_value(tap_numeric_pixel(input,geometry,image,tap,row,column)),sampled_pixel(Buffer.tree(~F32,input),tap_base(geometry,image,tap),U32.to_nat(row),U32.to_nat(column),U32.to_nat((tap / k % k : U32)),U32.to_nat((tap % k : U32)),U32.to_nat(h),U32.to_nat(w),U32.to_nat(s),U32.to_nat(p)),numeric_sampled(tap_numeric_pixel(input,geometry,image,tap,row,column))) : {_ == natural_tap_value(input,geometry,image,tap,row,column) : F32}
  %Equal.sym(Nat,U32.to_nat((tap / k % k : U32)),Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(k)),tap_row_decoded(limits,tap,k,positive,square_bound)) : {sampled_pixel(Buffer.tree(~F32,input),tap_base(geometry,image,tap),U32.to_nat(row),U32.to_nat(column),_,U32.to_nat((tap % k : U32)),U32.to_nat(h),U32.to_nat(w),U32.to_nat(s),U32.to_nat(p)) == natural_tap_value(input,geometry,image,tap,row,column) : F32}
  Equal.cong(Nat,F32,value => sampled_pixel(Buffer.tree(~F32,input),tap_base(geometry,image,tap),U32.to_nat(row),U32.to_nat(column),Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(k)),value,U32.to_nat(h),U32.to_nat(w),U32.to_nat(s),U32.to_nat(p)),U32.to_nat((tap % k : U32)),Nat.mod(U32.to_nat(tap),U32.to_nat(k)),tap_column_decoded(limits,tap,k,positive,square_bound))

def weight_prefix(+output: Nat,+channels: Nat,+kernel: Nat) ->
  {Nat.mul(output,Nat.mul(channels,Nat.mul(kernel,kernel))) == Nat.mul(Nat.mul(Nat.mul(output,channels),kernel),kernel) : Nat}:
  Equal.trans(Nat,Nat.mul(output,Nat.mul(channels,Nat.mul(kernel,kernel))),Nat.mul(Nat.mul(output,channels),Nat.mul(kernel,kernel)),Nat.mul(Nat.mul(Nat.mul(output,channels),kernel),kernel),
    Equal.sym(Nat,Nat.mul(Nat.mul(output,channels),Nat.mul(kernel,kernel)),Nat.mul(output,Nat.mul(channels,Nat.mul(kernel,kernel))),RowMajor.assoc(output,channels,Nat.mul(kernel,kernel))),
    Equal.sym(Nat,Nat.mul(Nat.mul(Nat.mul(output,channels),kernel),kernel),Nat.mul(Nat.mul(output,channels),Nat.mul(kernel,kernel)),RowMajor.assoc(Nat.mul(output,channels),kernel,kernel)))

def weight_column_decode(+output: Nat,+channels: Nat,+kernel: Nat,+tap: Nat,+positive: {Nat.is_lt(0n,kernel) == True{} : Bool}) ->
  {Nat.mod(Nat.add(Nat.mul(output,Nat.mul(channels,Nat.mul(kernel,kernel))),tap),kernel) == Nat.mod(tap,kernel) : Nat}:
  %Equal.sym(Nat,Nat.mul(output,Nat.mul(channels,Nat.mul(kernel,kernel))),Nat.mul(Nat.mul(Nat.mul(output,channels),kernel),kernel),weight_prefix(output,channels,kernel)) :
    {Nat.mod(Nat.add(_,tap),kernel) == Nat.mod(tap,kernel) : Nat}
  DivisionOrder.remainder_shift(tap,kernel,Nat.mul(Nat.mul(output,channels),kernel),positive)

def weight_row_decode(+output: Nat,+channels: Nat,+kernel: Nat,+tap: Nat,+positive: {Nat.is_lt(0n,kernel) == True{} : Bool}) ->
  {Nat.mod(Nat.div(Nat.add(Nat.mul(output,Nat.mul(channels,Nat.mul(kernel,kernel))),tap),kernel),kernel) == Nat.mod(Nat.div(tap,kernel),kernel) : Nat}:
  %Equal.sym(Nat,Nat.mul(output,Nat.mul(channels,Nat.mul(kernel,kernel))),Nat.mul(Nat.mul(Nat.mul(output,channels),kernel),kernel),weight_prefix(output,channels,kernel)) :
    {Nat.mod(Nat.div(Nat.add(_,tap),kernel),kernel) == Nat.mod(Nat.div(tap,kernel),kernel) : Nat}
  Equal.trans(Nat,Nat.mod(Nat.div(Nat.add(Nat.mul(Nat.mul(Nat.mul(output,channels),kernel),kernel),tap),kernel),kernel),Nat.mod(Nat.add(Nat.mul(Nat.mul(output,channels),kernel),Nat.div(tap,kernel)),kernel),Nat.mod(Nat.div(tap,kernel),kernel),
    Equal.cong(Nat,Nat,value => Nat.mod(value,kernel),Nat.div(Nat.add(Nat.mul(Nat.mul(Nat.mul(output,channels),kernel),kernel),tap),kernel),Nat.add(Nat.mul(Nat.mul(output,channels),kernel),Nat.div(tap,kernel)),DivisionOrder.quotient_shift(tap,kernel,Nat.mul(Nat.mul(output,channels),kernel),positive)),
    DivisionOrder.remainder_shift(Nat.div(tap,kernel),kernel,Nat.mul(output,channels),positive))

def weight_channel_decode(+output: Nat,+channels: Nat,+kernel: Nat,+tap: Nat,
  +channels_positive: {Nat.is_lt(0n,channels) == True{} : Bool},+square_positive: {Nat.is_lt(0n,Nat.mul(kernel,kernel)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,Nat.mul(channels,Nat.mul(kernel,kernel))) == True{} : Bool}) ->
  {Nat.mod(Nat.div(Nat.add(Nat.mul(output,Nat.mul(channels,Nat.mul(kernel,kernel))),tap),Nat.mul(kernel,kernel)),channels) == Nat.div(tap,Nat.mul(kernel,kernel)) : Nat}:
  +area = Nat.mul(kernel,kernel)
  %RowMajor.assoc(output,channels,area) : {Nat.mod(Nat.div(Nat.add(_,tap),area),channels) == Nat.div(tap,area) : Nat}
  Equal.trans(Nat,Nat.mod(Nat.div(Nat.add(Nat.mul(Nat.mul(output,channels),area),tap),area),channels),Nat.mod(Nat.add(Nat.mul(output,channels),Nat.div(tap,area)),channels),Nat.div(tap,area),
    Equal.cong(Nat,Nat,value => Nat.mod(value,channels),Nat.div(Nat.add(Nat.mul(Nat.mul(output,channels),area),tap),area),Nat.add(Nat.mul(output,channels),Nat.div(tap,area)),DivisionOrder.quotient_shift(tap,area,Nat.mul(output,channels),square_positive)),
    Radix.remainder(output,Nat.div(tap,area),channels,channels_positive,DivisionOrder.quotient_below(tap,channels,area,square_positive,tap_inside)))

def weight_output_decode(+output: Nat,+channels: Nat,+kernel: Nat,+tap: Nat,
  +taps_positive: {Nat.is_lt(0n,Nat.mul(channels,Nat.mul(kernel,kernel))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,Nat.mul(channels,Nat.mul(kernel,kernel))) == True{} : Bool}) ->
  {Nat.div(Nat.add(Nat.mul(output,Nat.mul(channels,Nat.mul(kernel,kernel))),tap),Nat.mul(channels,Nat.mul(kernel,kernel))) == output : Nat}:
  Radix.quotient(output,tap,Nat.mul(channels,Nat.mul(kernel,kernel)),taps_positive,tap_inside)

def window_natural_state(window: Kernel.WeightWindow,+count: U32,input: Buffer.Model<F32>,position: Nat,positive: {Nat.is_lt(0n,U32.to_nat(window_columns(window))) == True{} : Bool}) -> NaturalPacking:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  NaturalPacking{input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0)),0n,position,columns,Kernel.patch(tap,geometry,image),h,w,s,p,positive}

def window_natural_initial(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>,+position: Nat,
  +positive: {Nat.is_lt(0n,U32.to_nat(window_columns(window))) == True{} : Bool},
  +row_match: {window_row(window) == U32.from_nat(Nat.div(position,U32.to_nat(window_columns(window)))) : U32},
  +column_match: {window_column(window) == U32.from_nat(Nat.mod(position,U32.to_nat(window_columns(window)))) : U32}) ->
  {window_input(window,count,input) == Loop.run(~Packing,~step,U32.to_nat(count),natural_represent(window_natural_state(window,count,input,position,positive))) : Packing}:
  Kernel.WeightWindow{+geometry,image,tap,output,+row,+column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,+columns,taps} = geometry
  %Equal.sym(U32,row,U32.from_nat(Nat.div(position,U32.to_nat(columns))),row_match) :
    {Loop.run(~Packing,~step,U32.to_nat(count),Packing{input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0)),0,_,column,columns,Kernel.patch(tap,geometry,image),h,w,s,p}) == Loop.run(~Packing,~step,U32.to_nat(count),natural_represent(window_natural_state(window,count,input,position,positive))) : Packing}
  %Equal.sym(U32,column,U32.from_nat(Nat.mod(position,U32.to_nat(columns))),column_match) :
    {Loop.run(~Packing,~step,U32.to_nat(count),Packing{input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0)),0,U32.from_nat(Nat.div(position,U32.to_nat(columns))),_,columns,Kernel.patch(tap,geometry,image),h,w,s,p}) == Loop.run(~Packing,~step,U32.to_nat(count),natural_represent(window_natural_state(window,count,input,position,positive))) : Packing}
  {==}

def window_input_row_end(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>,+position: Nat,
  +positive: {Nat.is_lt(0n,U32.to_nat(window_columns(window))) == True{} : Bool},
  +row_match: {window_row(window) == U32.from_nat(Nat.div(position,U32.to_nat(window_columns(window)))) : U32},
  +column_match: {window_column(window) == U32.from_nat(Nat.mod(position,U32.to_nat(window_columns(window)))) : U32}) ->
  {packing_row(window_input(window,count,input)) == U32.from_nat(Nat.div(Nat.add(position,U32.to_nat(count)),U32.to_nat(window_columns(window)))) : U32}:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  +initial = window_natural_state(window,count,input,position,positive)
  Equal.trans(U32,packing_row(window_input(window,count,input)),packing_row(Loop.run(~Packing,~step,U32.to_nat(count),natural_represent(initial))),U32.from_nat(Nat.div(Nat.add(position,U32.to_nat(count)),U32.to_nat(columns))),
    Equal.cong(Packing,U32,packing_row,window_input(window,count,input),Loop.run(~Packing,~step,U32.to_nat(count),natural_represent(initial)),window_natural_initial(window,count,input,position,positive,row_match,column_match)),
    packing_row_end(U32.to_nat(count),initial))

def window_input_column_end(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>,+position: Nat,
  +positive: {Nat.is_lt(0n,U32.to_nat(window_columns(window))) == True{} : Bool},
  +row_match: {window_row(window) == U32.from_nat(Nat.div(position,U32.to_nat(window_columns(window)))) : U32},
  +column_match: {window_column(window) == U32.from_nat(Nat.mod(position,U32.to_nat(window_columns(window)))) : U32}) ->
  {packing_column(window_input(window,count,input)) == U32.from_nat(Nat.mod(Nat.add(position,U32.to_nat(count)),U32.to_nat(window_columns(window)))) : U32}:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  +initial = window_natural_state(window,count,input,position,positive)
  Equal.trans(U32,packing_column(window_input(window,count,input)),packing_column(Loop.run(~Packing,~step,U32.to_nat(count),natural_represent(initial))),U32.from_nat(Nat.mod(Nat.add(position,U32.to_nat(count)),U32.to_nat(columns))),
    Equal.cong(Packing,U32,packing_column,window_input(window,count,input),Loop.run(~Packing,~step,U32.to_nat(count),natural_represent(initial)),window_natural_initial(window,count,input,position,positive,row_match,column_match)),
    packing_column_end(U32.to_nat(count),initial))

def window_first(window: Kernel.WeightWindow) -> U32:
  Kernel.WeightWindow{geometry,image,tap,output,row,column,first} = window
  first

def window_output(window: Kernel.WeightWindow) -> U32:
  Kernel.WeightWindow{geometry,image,tap,output,row,column,first} = window
  output

def window_outputs(window: Kernel.WeightWindow) -> U32:
  Kernel.WeightWindow{geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  o

def natural_gradient_value(input: Buffer.Model<F32>,window: Kernel.WeightWindow,lane: Nat,position: Nat) -> F32:
  Kernel.WeightWindow{geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Tree.read(F32,Buffer.tree(~F32,input),U32.from_nat(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(o)),U32.to_nat((output + U32.from_nat(lane) : U32))),U32.to_nat(rows)),U32.to_nat(columns)),position)))

def window_gradient_natural(+input: Buffer.Model<F32>,+window: Kernel.WeightWindow,+lane: Nat,+first: Nat,+index: Nat,
  +first_match: {window_first(window) == U32.from_nat(first) : U32},
  +output_inside: {U32.is_lt((window_output(window) + U32.from_nat(lane) : U32),window_outputs(window)) == True{} : Bool}) ->
  {gradient_value(input,gradient_context(window),lane,U32.from_nat(index)) == natural_gradient_value(input,window,lane,Nat.add(first,index)) : F32}:
  Kernel.WeightWindow{geometry,+image,tap,+output,row,column,+start} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,+rows,+columns,taps} = geometry
  %Equal.sym(Bool,U32.is_lt((output + U32.from_nat(lane) : U32),o),True{},output_inside) :
    {Bool.pick(F32,_,Tree.read(F32,Buffer.tree(~F32,input),((output + U32.from_nat(lane)) * (rows * columns) + (image * o * rows * columns + start) + U32.from_nat(index) : U32)),0.0) == natural_gradient_value(input,window,lane,Nat.add(first,index)) : F32}
  %Equal.sym(U32,start,U32.from_nat(first),first_match) :
    {Tree.read(F32,Buffer.tree(~F32,input),((output + U32.from_nat(lane)) * (rows * columns) + (image * o * rows * columns + _) + U32.from_nat(index) : U32)) == natural_gradient_value(input,window,lane,Nat.add(first,index)) : F32}
  Equal.cong(U32,F32,address => Tree.read(F32,Buffer.tree(~F32,input),address),
    ((output + U32.from_nat(lane)) * (rows * columns) + (image * o * rows * columns + U32.from_nat(first)) + U32.from_nat(index) : U32),
    U32.from_nat(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(o)),U32.to_nat((output + U32.from_nat(lane) : U32))),U32.to_nat(rows)),U32.to_nat(columns)),Nat.add(first,index))),
    Encoded.nchw_gradient_encoded(image,o,(output + U32.from_nat(lane) : U32),rows,columns,first,index))

def patch_key(+upper: Bool,row: Accumulator.Row) -> Nat:
  Nat.add(Bool.pick(Nat,upper,4n,0n),Coordinates.row_number(row))

def patch_key_inside(upper: Bool,row: Accumulator.Row) -> {Nat.is_lt(patch_key(upper,row),8n) == True{} : Bool}:
  match upper row:
    case False{} Accumulator.R0{}: {==}
    case False{} Accumulator.R1{}: {==}
    case False{} Accumulator.R2{}: {==}
    case False{} Accumulator.R3{}: {==}
    case True{} Accumulator.R0{}: {==}
    case True{} Accumulator.R1{}: {==}
    case True{} Accumulator.R2{}: {==}
    case True{} Accumulator.R3{}: {==}

def tap_pixel_at(window: Kernel.WeightWindow,input: Buffer.Model<F32>,lane: Nat,+position: Nat) -> F32:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  tap_pixel_value(input,geometry,image,(tap + U32.from_nat(lane) : U32),U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))))

def natural_patch_at(+window: Kernel.WeightWindow,+input: Buffer.Model<F32>,+upper: Bool,+row: Accumulator.Row,+position: Nat) ->
  {natural_patch_value(window_patch_context(window,input),upper,row,position) == tap_pixel_at(window,input,patch_key(upper,row),position) : F32}:
  Kernel.WeightWindow{+geometry,+image,+tap,output,row0,column,first} = window
  Kernel.Geometry{c,o,+h,+w,k,+s,+p,rows,+columns,taps} = geometry
  Equal.cong(Kernel.Lane,F32,lane => pixel_value(input,U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))),h,w,s,p,lane),
    Kernel.selected(Kernel.patch(tap,geometry,image),patch_key(upper,row)),Kernel.lane((tap + U32.from_nat(patch_key(upper,row)) : U32),geometry,image),
    patch_lane_at(tap,geometry,image,patch_key(upper,row),patch_key_inside(upper,row)))

def position_row_within(+position: Nat,+rows: U32,+columns: U32,+positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},
  +within: {Nat.is_lt(position,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(Nat.div(position,U32.to_nat(columns)))),U32.to_nat(rows)) == True{} : Bool}:
  +quotient = Nat.div(position,U32.to_nat(columns))
  +inside = DivisionOrder.quotient_below(position,U32.to_nat(rows),U32.to_nat(columns),positive,within)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(quotient)),quotient,FullRange.roundtrip(quotient,rows,Bounds.strict_le(quotient,U32.to_nat(rows),inside))) : {Nat.is_lt(_,U32.to_nat(rows)) == True{} : Bool}
  inside

def position_column_within(+position: Nat,+columns: U32,+positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(Nat.mod(position,U32.to_nat(columns)))),U32.to_nat(columns)) == True{} : Bool}:
  +remainder = Nat.mod(position,U32.to_nat(columns))
  +inside = DivisionOrder.remainder_small(position,U32.to_nat(columns),positive)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(remainder)),remainder,FullRange.roundtrip(remainder,columns,Bounds.strict_le(remainder,U32.to_nat(columns),inside))) : {Nat.is_lt(_,U32.to_nat(columns)) == True{} : Bool}
  inside

def packing_reader(state: Packing) -> Buffer.Model<F32>:
  Packing{input,output,index,row,column,columns,patch,h,w,s,p} = state
  input

def advanced_reader(wrap: Bool,+input: Buffer.Model<F32>,+output: Tree.Tree<Tile.Tile8>,+index: U32,+row: U32,+column: U32,+columns: U32,+patch: Kernel.Patch,+height: U32,+width: U32,+stride: U32,+padding: U32) ->
  {packing_reader(advanced(wrap,input,output,index,row,column,columns,patch,height,width,stride,padding)) == input : Buffer.Model<F32>}:
  match wrap:
    case False{}: {==}
    case True{}: {==}

def packing_reader_step(+state: Packing) -> {packing_reader(step(state)) == packing_reader(state) : Buffer.Model<F32>}:
  Packing{+input,+output,+index,+row,+column,+columns,+patch,+height,+width,+stride,+padding} = state
  advanced_reader(U32.is_eq((column + 1 : U32),columns),input,
    Tree.write(Tile.Tile8,output,index,tile_value(~Kernel.Pixel,~patch_sample,input,Kernel.Pixel{patch,row,column,height,width,stride,padding})),
    (index + 1 : U32),row,(column + 1 : U32),columns,patch,height,width,stride,padding)

def window_input_reader(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>) ->
  {packing_reader(window_input(window,count,input)) == input : Buffer.Model<F32>}:
  Kernel.WeightWindow{+geometry,+image,+tap,output,+row,+column,first} = window
  Kernel.Geometry{c,o,+h,+w,k,+s,+p,rows,+columns,taps} = geometry
  Projection.preserves_observation(~Packing,~Buffer.Model<F32>,~step,~packing_reader,~packing_reader_step,U32.to_nat(count),
    Packing{input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0)),0,row,column,columns,Kernel.patch(tap,geometry,image),h,w,s,p})

def natural_reader(state: NaturalPacking) -> Buffer.Model<F32>:
  NaturalPacking{input,output,index,position,columns,patch,h,w,s,p,positive} = state
  input

def natural_reader_step(+state: NaturalPacking) -> {natural_reader(natural_step(state)) == natural_reader(state) : Buffer.Model<F32>}:
  NaturalPacking{input,output,index,position,columns,patch,h,w,s,p,positive} = state
  {==}

def represented_reader(+state: NaturalPacking) -> {packing_reader(natural_represent(state)) == natural_reader(state) : Buffer.Model<F32>}:
  NaturalPacking{input,output,index,position,columns,patch,h,w,s,p,positive} = state
  {==}

def natural_reader_end(+count: Nat,+state: NaturalPacking) ->
  {packing_reader(Loop.run(~Packing,~step,count,natural_represent(state))) == natural_reader(state) : Buffer.Model<F32>}:
  +result = Loop.run(~NaturalPacking,~natural_step,count,state)
  Equal.trans(Buffer.Model<F32>,packing_reader(Loop.run(~Packing,~step,count,natural_represent(state))),packing_reader(natural_represent(result)),natural_reader(state),
    Equal.cong(Packing,Buffer.Model<F32>,packing_reader,Loop.run(~Packing,~step,count,natural_represent(state)),natural_represent(result),natural_loop(count,state)),
    Equal.trans(Buffer.Model<F32>,packing_reader(natural_represent(result)),natural_reader(result),natural_reader(state),represented_reader(result),
      Projection.preserves_observation(~NaturalPacking,~Buffer.Model<F32>,~natural_step,~natural_reader,~natural_reader_step,count,state)))

def window_input_kept(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>,+position: Nat,
  +positive: {Nat.is_lt(0n,U32.to_nat(window_columns(window))) == True{} : Bool},
  +row_match: {window_row(window) == U32.from_nat(Nat.div(position,U32.to_nat(window_columns(window)))) : U32},
  +column_match: {window_column(window) == U32.from_nat(Nat.mod(position,U32.to_nat(window_columns(window)))) : U32}) ->
  {packing_reader(window_input(window,count,input)) == input : Buffer.Model<F32>}:
  Kernel.WeightWindow{+geometry,image,tap,output,row,column,first} = window
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  +initial = window_natural_state(window,count,input,position,positive)
  Equal.trans(Buffer.Model<F32>,packing_reader(window_input(window,count,input)),packing_reader(Loop.run(~Packing,~step,U32.to_nat(count),natural_represent(initial))),input,
    Equal.cong(Packing,Buffer.Model<F32>,packing_reader,window_input(window,count,input),Loop.run(~Packing,~step,U32.to_nat(count),natural_represent(initial)),window_natural_initial(window,count,input,position,positive,row_match,column_match)),natural_reader_end(U32.to_nat(count),initial))

def packed_reader(result: Packed) -> Buffer.Model<F32>:
  Packed{input,output} = result
  input

def gradient_finished_reader(+state: Traversal.State<&2,&2,Buffer.Model<F32>,Tree.Tree<Tile.Tile8>,Kernel.Gradient>) ->
  {packed_reader(gradient_finished(state)) == Traversals.model_reader(~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,state) : Buffer.Model<F32>}:
  Traversal.State{input,output,position,context} = state
  {==}

def window_gradient_kept(+window: Kernel.WeightWindow,+count: U32,+input: Buffer.Model<F32>) ->
  {packed_reader(window_gradient(window,count,input)) == input : Buffer.Model<F32>}:
  Kernel.WeightWindow{geometry,+image,tap,output,row,column,first} = window
  Kernel.Geometry{c,+o,h,w,k,s,p,+rows,+columns,taps} = geometry
  +initial = {Traversal.State{input,Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0)),0,Kernel.Gradient{o,(rows * columns : U32),(image * o * rows * columns + first : U32),output,0}} : Traversal.State<&2,&2,Buffer.Model<F32>,Tree.Tree<Tile.Tile8>,Kernel.Gradient>}
  +result = Loop.run(~Traversal.State<&2,&2,Buffer.Model<F32>,Tree.Tree<Tile.Tile8>,Kernel.Gradient>,~Traversals.model_step(~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~gradient_tile_value,~Kernel.gradient_destination),U32.to_nat(count),initial)
  Equal.trans(Buffer.Model<F32>,packed_reader(gradient_finished(result)),Traversals.model_reader(~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,result),input,
    gradient_finished_reader(result),Traversals.model_reader_kept(~Buffer.Model<F32>,~Tile.Tile8,~Kernel.Gradient,~gradient_tile_value,~Kernel.gradient_destination,U32.to_nat(count),initial))
