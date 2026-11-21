# Weight-gradient tiles: the input cells of every phase plane of a band and
# the gradient cells of every output block, read in position order.
import Base
import ../kernel_weight_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../math_sequence.bend as Sequence
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./kernel_tile_refinement.bend as Tiles
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_values.bend as TiledValues

import ../traversal_loop.bend as Loop
import ../traversal_array.bend as Traversal
import ./traversal_array_refinement.bend as Traversals
import ./traversal_indexed_values.bend as IndexedValues
import ./tensor_access_refinement.bend as AccessLaws
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./nat_ceiling.bend as Ceiling
import ./nat_extrema.bend as Extrema
import ./nat_to_u32_bounds.bend as Bounds
import ./word_encoding.bend as Words
import ./nat_division.bend as Division
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./boolean_logic.bend as Logic
import ./row_major_index.bend as RowMajor
import ./list_storage_readback.bend as ListRead
import ./nat_division_order.bend as DivisionOrder
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./binary_radix.bend as Radix
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./natural_addition.bend as Addition
import ../traversal_blocks.bend as TraversalBlocks

type Packed is Data:
  Packed{input: Buffer.Model<Tile.Tile8>,output: Tree.Tree<Tile.Tile8>}

def packed_input(state: Packed) -> Buffer.Model<Tile.Tile8>:
  Packed{input,output} = state
  input

def packed_output(state: Packed) -> Tree.Tree<Tile.Tile8>:
  Packed{input,output} = state
  output

def owners(result: Packed) -> Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>:
  Packed{input,output} = result
  (Buffer.pack(~Tile.Tile8,input),Tree.pack(Tile.Tile8,output))

# No lane index is eight or more.
def beyond_eight(+rest: Nat,small: {Nat.is_lt(8n+rest,8n) == True{} : Bool}) -> Empty:
  Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),small))

def filled_lane(+index: Nat,+value: F32,small: {Nat.is_lt(index,8n) == True{} : Bool}) -> {Tiles.select_lane(index,Tile.filled(value)) == value : F32}:
  match index:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+rest: Empty.absurd({Tiles.select_lane(8n+rest,Tile.filled(value)) == value : F32},beyond_eight(rest,small))

# ---- Reading traversals ----

# The model states of a traversal that reads a buffer and stores at its
# position.
def trace(~Context: Data,~sample: Buffer.Model<Tile.Tile8> -> Context -> U32 -> Tile.Tile8,count: Nat,position: U32,context: Context,
  input: Buffer.Model<Tile.Tile8>,output: Tree.Tree<Tile.Tile8>) -> Traversal.State<&2,&2,Buffer.Model<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Context>:
  Loop.run(~Traversal.State<&2,&2,Buffer.Model<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Context>,
    ~Traversals.model_step(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,~sample,~TraversalBlocks.in_place(~Context)),count,Traversal.State{input,output,position,context})

def read_run(~Context: Data,~sample: Buffer.Model<Tile.Tile8> -> Context -> U32 -> Tile.Tile8,count: Nat,position: U32,context: Context,
  input: Buffer.Model<Tile.Tile8>,output: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  IndexedValues.run(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,~sample,~TraversalBlocks.in_place(~Context),count,position,context,input,output)

def result_parts(~Context: Data,+state: Traversal.State<&2,&2,Buffer.Model<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Context>) ->
  {AccessLaws.pack_result(~Storage.Buffer<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Buffer.pack(~Tile.Tile8),AccessLaws.finish_model(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,state)) ==
    (Buffer.pack(~Tile.Tile8,Traversals.model_reader(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,state)),
      Tree.pack(Tile.Tile8,IndexedValues.output_tree(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,AccessLaws.finish_model(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,state)))) :
    Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>}:
  Traversal.State{reader,output,position,context} = state
  {==}

# The buffer read and the model run it stores.
def owned(~Context: Data,~sample: Buffer.Model<Tile.Tile8> -> Context -> U32 -> Tile.Tile8,count: Nat,position: U32,context: Context,
  +input: Buffer.Model<Tile.Tile8>,output: Tree.Tree<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>:
  (Buffer.pack(~Tile.Tile8,input),Tree.pack(Tile.Tile8,read_run(~Context,~sample,count,position,context,input,output)))

# A traversal reading a buffer gives the buffer back and stores the model
# run.
def read_owners(~Context: Data,~read: Storage.Buffer<Tile.Tile8> -> Context -> U32 -> Storage.Buffer<Tile.Tile8> & Tile.Tile8,
  ~sample: Buffer.Model<Tile.Tile8> -> Context -> U32 -> Tile.Tile8,
  ~read_matches: @+reader: Buffer.Model<Tile.Tile8> -> @+context: Context -> @+position: U32 ->
    {read(Buffer.pack(~Tile.Tile8,reader),context,position) == (Buffer.pack(~Tile.Tile8,reader),sample(reader,context,position)) : Storage.Buffer<Tile.Tile8> & Tile.Tile8},
  +count: Nat,+position: U32,+context: Context,+input: Buffer.Model<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>) ->
  {Traversal.finish(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Context,Traversal.run(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Context,~read,~TraversalBlocks.in_place(~Context),count,
      Traversal.State{Buffer.pack(~Tile.Tile8,input),Tree.pack(Tile.Tile8,output),position,context})) ==
    owned(~Context,~sample,count,position,context,input,output) : Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>}:
  +start = {Traversal.State{input,output,position,context} : Traversal.State<&2,&2,Buffer.Model<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Context>}
  +model = trace(~Context,~sample,count,position,context,input,output)
  %Equal.sym(Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>,
      Traversal.finish(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Context,Traversal.run(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Context,~read,~TraversalBlocks.in_place(~Context),count,
        Traversal.State{Buffer.pack(~Tile.Tile8,input),Tree.pack(Tile.Tile8,output),position,context})),
      AccessLaws.pack_result(~Storage.Buffer<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Buffer.pack(~Tile.Tile8),AccessLaws.finish_model(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,model)),
      AccessLaws.finish_relation(~Storage.Buffer<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,~Context,~Buffer.pack(~Tile.Tile8),
        ~Traversals.represent(~Storage.Buffer<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,~Buffer.pack(~Tile.Tile8)),
        ~AccessLaws.indexed_finished(~Storage.Buffer<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,~Buffer.pack(~Tile.Tile8)),
        Traversal.run(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Context,~read,~TraversalBlocks.in_place(~Context),count,
          Traversal.State{Buffer.pack(~Tile.Tile8,input),Tree.pack(Tile.Tile8,output),position,context}),model,
        Traversals.traversal_refines_model(~Storage.Buffer<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,~Buffer.pack(~Tile.Tile8),~read,~sample,~TraversalBlocks.in_place(~Context),
          ~read_matches,count,start))) :
    {_ == owned(~Context,~sample,count,position,context,input,output) : Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>}
  %Equal.sym(Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>,
      AccessLaws.pack_result(~Storage.Buffer<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Buffer.pack(~Tile.Tile8),AccessLaws.finish_model(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,model)),
      (Buffer.pack(~Tile.Tile8,Traversals.model_reader(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,model)),
        Tree.pack(Tile.Tile8,IndexedValues.output_tree(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,AccessLaws.finish_model(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,model)))),
      result_parts(~Context,model)) :
    {_ == owned(~Context,~sample,count,position,context,input,output) : Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>}
  %Equal.sym(Buffer.Model<Tile.Tile8>,Traversals.model_reader(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,model),input,
      Traversals.model_reader_kept(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,~sample,~TraversalBlocks.in_place(~Context),count,start)) :
    {(Buffer.pack(~Tile.Tile8,_),Tree.pack(Tile.Tile8,IndexedValues.output_tree(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,AccessLaws.finish_model(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,model)))) ==
      owned(~Context,~sample,count,position,context,input,output) : Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>}
  {==}

def zero_base(~Context: Data,context: Context) -> U32:
  0

def zero_affine(~Context: Data,+context: Context,+position: U32) -> {TraversalBlocks.in_place(~Context,context,position) == (zero_base(~Context,context) + position : U32) : U32}:
  Equal.sym(U32,(0 + position : U32),position,Arithmetic.add_zero_left(position))

# A run from position first writes its samples in order from first.
def read_sequence(~Context: Data,~sample: Buffer.Model<Tile.Tile8> -> Context -> U32 -> Tile.Tile8,+count: Nat,+first: Nat,+context: Context,
  +input: Buffer.Model<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>) ->
  {read_run(~Context,~sample,count,U32.from_nat(first),context,input,output) ==
    Tree.write_sequence(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~IndexedValues.Sampled<Buffer.Model<Tile.Tile8>,Context>,
      ~IndexedValues.sampled(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,~sample),count,first,IndexedValues.Sampled{input,context}),U32.from_nat(first),output) : Tree.Tree<Tile.Tile8>}:
  +values = ListRead.enumerate(~Tile.Tile8,~IndexedValues.Sampled<Buffer.Model<Tile.Tile8>,Context>,
    ~IndexedValues.sampled(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,~sample),count,first,IndexedValues.Sampled{input,context})
  %Arithmetic.add_zero_left(U32.from_nat(first)) :
    {read_run(~Context,~sample,count,U32.from_nat(first),context,input,output) == Tree.write_sequence(~Tile.Tile8,values,_,output) : Tree.Tree<Tile.Tile8>}
  IndexedValues.sequence(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Context,~sample,~TraversalBlocks.in_place(~Context),~zero_base(~Context),~zero_affine(~Context),count,first,context,input,output)

# ---- Pixel planes ----

def plane_sample(input: Buffer.Model<Tile.Tile8>,plane: Kernel.Plane,position: U32) -> Tile.Tile8:
  Kernel.Plane{+base,+row,+column,+stride,+width,+groups,+row_first,+row_end,+column_first,+column_end,+first_row,+band_columns,+start} = plane
  +within = (position - start : U32)
  +tile_row = (first_row + within / band_columns : U32)
  +tile_column = (within % band_columns : U32)
  Bool.pick(Tile.Tile8,U32.is_le(row_first,tile_row) && U32.is_lt(tile_row,row_end) && U32.is_le(column_first,tile_column) && U32.is_lt(tile_column,column_end),
    Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),(base + ((tile_row * stride + row) * width + tile_column * stride + column) * groups : U32)),Tile.filled(0.0))

def plane_read_matches(+input: Buffer.Model<Tile.Tile8>,+plane: Kernel.Plane,+position: U32) ->
  {Kernel.plane_read(Buffer.pack(~Tile.Tile8,input),plane,position) == (Buffer.pack(~Tile.Tile8,input),plane_sample(input,plane,position)) : Storage.Buffer<Tile.Tile8> & Tile.Tile8}:
  Kernel.Plane{+base,+row,+column,+stride,+width,+groups,+row_first,+row_end,+column_first,+column_end,+first_row,+band_columns,+start} = plane
  +within = (position - start : U32)
  +tile_row = (first_row + within / band_columns : U32)
  +tile_column = (within % band_columns : U32)
  Buffer.read_masked(~Tile.Tile8,Tile.filled(0.0),U32.is_le(row_first,tile_row) && U32.is_lt(tile_row,row_end) && U32.is_le(column_first,tile_column) && U32.is_lt(tile_column,column_end),
    input,(base + ((tile_row * stride + row) * width + tile_column * stride + column) * groups : U32))

def pixel_plane(plane: U32,pixels: Kernel.Pixels,state: Packed) -> Packed:
  Kernel.Pixels{+geometry,+base,+phases,+first_row,+band_rows,+band_columns} = pixels
  Packed{+input,output} = state
  +plane = plane
  Packed{input,read_run(~Kernel.Plane,~plane_sample,U32.to_nat((band_rows * band_columns : U32)),(plane * band_rows * band_columns : U32),
    Kernel.plane_of(plane,Kernel.Pixels{geometry,base,phases,first_row,band_rows,band_columns}),input,output)}

def pixel_plane_matches(+plane: U32,+pixels: Kernel.Pixels,+state: Packed) ->
  {Kernel.pixel_plane(plane,pixels,owners(state)) == owners(pixel_plane(plane,pixels,state)) : Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>}:
  Kernel.Pixels{+geometry,+base,+phases,+first_row,+band_rows,+band_columns} = pixels
  Packed{+input,+output} = state
  read_owners(~Kernel.Plane,~Kernel.plane_read,~plane_sample,~plane_read_matches,U32.to_nat((band_rows * band_columns : U32)),(plane * band_rows * band_columns : U32),
    Kernel.plane_of(plane,Kernel.Pixels{geometry,base,phases,first_row,band_rows,band_columns}),input,output)

# ---- Blocks of planes ----

# A fold whose step index writes its row of width tiles at index * width
# writes the rows one after another.
def blocks_points(~Context: Data,~Ready: Context -> Data,~reader: Context -> Buffer.Model<Tile.Tile8>,~step: U32 -> Context -> Packed -> Packed,~row: Context -> Nat -> List<&2,Tile.Tile8>,
  ~width: Context -> Nat,~count: Context -> Nat,
  ~written: @+context: Context -> @+ready: Ready(context) -> @+index: Nat -> @+inside: {Nat.is_lt(index,count(context)) == True{} : Bool} ->
    @+tree: Tree.Tree<Tile.Tile8> ->
    {step(U32.from_nat(index),context,Packed{reader(context),tree}) == Packed{reader(context),Tree.write_sequence(~Tile.Tile8,row(context,index),U32.from_nat(Nat.mul(index,width(context))),tree)} : Packed},
  ~length: @+context: Context -> @+index: Nat -> {List.length(&2,Tile.Tile8,row(context,index)) == width(context) : Nat},
  n: Nat,+first: Nat,+context: Context,+ready: Ready(context),+tree: Tree.Tree<Tile.Tile8>,
  +room: {Nat.is_le(Nat.add(first,n),count(context)) == True{} : Bool}) ->
  {Traversal.indexed_fold(~Packed,~Context,~step,n,U32.from_nat(first),context,Packed{reader(context),tree}) ==
    Packed{reader(context),Tree.write_sequence(~Tile.Tile8,List.concat(&2,Tile.Tile8,ListRead.enumerate(~List<&2,Tile.Tile8>,~Context,~row,n,first,context)),
      U32.from_nat(Nat.mul(first,width(context))),tree)} : Packed}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +size = width(context)
      +here = row(context,first)
      +offset = U32.from_nat(Nat.mul(first,size))
      +later = List.concat(&2,Tile.Tile8,ListRead.enumerate(~List<&2,Tile.Tile8>,~Context,~row,rest,1n+first,context))
      +front = Tree.write_sequence(~Tile.Tile8,here,offset,tree)
      +input = reader(context)
      +wanted = {Packed{input,Tree.write_sequence(~Tile.Tile8,List.append(&2,Tile.Tile8,here,later),offset,tree)} : Packed}
      +ending = Equal.trans(U32,ListRead.end_position(~Tile.Tile8,here,offset),(offset + U32.from_nat(List.length(&2,Tile.Tile8,here)) : U32),U32.from_nat(Nat.mul(1n+first,size)),
        ListRead.end_position_length(~Tile.Tile8,here,offset),
        %Equal.sym(Nat,List.length(&2,Tile.Tile8,here),size,length(context,first)) : {(offset + U32.from_nat(_) : U32) == U32.from_nat(Nat.mul(1n+first,size)) : U32}
        %Order.commutative(Nat.mul(first,size),size) : {(offset + U32.from_nat(size) : U32) == U32.from_nat(_) : U32}
        WordArithmetic.native_add(Nat.mul(first,size),size))
      %Equal.sym(Packed,step(U32.from_nat(first),context,Packed{input,tree}),Packed{input,front},
        written(context,ready,first,Order.room_index(first,rest,count(context),room),tree)) :
        {Traversal.indexed_fold(~Packed,~Context,~step,rest,(U32.from_nat(first) + 1 : U32),context,_) == wanted : Packed}
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        {Traversal.indexed_fold(~Packed,~Context,~step,rest,_,context,Packed{input,front}) == wanted : Packed}
      %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.write_sequence(~Tile.Tile8,List.append(&2,Tile.Tile8,here,later),offset,tree),
        Tree.write_sequence(~Tile.Tile8,later,ListRead.end_position(~Tile.Tile8,here,offset),front),ListRead.concatenated(~Tile.Tile8,here,later,offset,tree)) :
        {Traversal.indexed_fold(~Packed,~Context,~step,rest,U32.from_nat(1n+first),context,Packed{input,front}) == Packed{input,_} : Packed}
      %Equal.sym(U32,ListRead.end_position(~Tile.Tile8,here,offset),U32.from_nat(Nat.mul(1n+first,size)),ending) :
        {Traversal.indexed_fold(~Packed,~Context,~step,rest,U32.from_nat(1n+first),context,Packed{input,front}) == Packed{input,Tree.write_sequence(~Tile.Tile8,later,_,front)} : Packed}
      blocks_points(~Context,~Ready,~reader,~step,~row,~width,~count,~written,~length,rest,1n+first,context,ready,front,Order.advance_room(first,rest,count(context),room))

# Tile i of row p of the blocks reads back at p * width + i.
def blocks_read(~Context: Data,~row: Context -> Nat -> List<&2,Tile.Tile8>,~width: Context -> Nat,
  ~length: @+context: Context -> @+index: Nat -> {List.length(&2,Tile.Tile8,row(context,index)) == width(context) : Nat},
  +rows: Nat,+context: Context,+index: Nat,+lane: Nat,+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tree),
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},+room: {Nat.is_le(Nat.mul(rows,width(context)),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},
  +inside: {Nat.is_lt(index,rows) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,width(context)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Tree.write_sequence(~Tile.Tile8,List.concat(&2,Tile.Tile8,ListRead.enumerate(~List<&2,Tile.Tile8>,~Context,~row,rows,0n,context)),U32.from_nat(0n),tree),
      U32.from_nat(Nat.add(Nat.mul(index,width(context)),lane))) ==
    ListRead.at(~Tile.Tile8,Tile.filled(0.0),lane,row(context,index)) : Tile.Tile8}:
  +values = List.concat(&2,Tile.Tile8,ListRead.enumerate(~List<&2,Tile.Tile8>,~Context,~row,rows,0n,context))
  +position = Nat.add(Nat.mul(index,width(context)),lane)
  +total = ListRead.concat_enumerate_length(~Tile.Tile8,~Context,~row,~width,~length,rows,0n,context)
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Tree.write_sequence(~Tile.Tile8,values,U32.from_nat(0n),tree),U32.from_nat(position)),
    ListRead.at(~Tile.Tile8,Tile.filled(0.0),position,values),ListRead.at(~Tile.Tile8,Tile.filled(0.0),lane,row(context,index)),
    ListRead.listed(~Tile.Tile8,values,position,0n,depth,tree,Tile.filled(0.0),balanced,limit,
      %Equal.sym(Nat,List.length(&2,Tile.Tile8,values),Nat.mul(rows,width(context)),total) : {Nat.is_le(_,U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool} room,
      %Equal.sym(Nat,List.length(&2,Tile.Tile8,values),Nat.mul(rows,width(context)),total) : {Nat.is_lt(position,_) == True{} : Bool} RowMajor.radix(index,rows,lane,width(context),inside,lane_inside)),
    ListRead.concat_enumerate_at(~Tile.Tile8,~Context,~row,~width,~length,Tile.filled(0.0),rows,0n,index,lane,context,inside,lane_inside))

# The planes of a pixel band: plane p holds area samples of its plane from
# p * area.
type Planes is Data:
  Planes{input: Buffer.Model<Tile.Tile8>,pixels: Kernel.Pixels,area: Nat,count: Nat}

def pixels_area(+pixels: Kernel.Pixels) -> U32:
  Kernel.Pixels{geometry,base,phases,first_row,+band_rows,+band_columns} = pixels
  (band_rows * band_columns : U32)

type PlanesReady<-pixels: Kernel.Pixels,-area: Nat> is Data:
  PlanesReady{area_exact: {pixels_area(pixels) == U32.from_nat(area) : U32},area_back: {U32.to_nat(U32.from_nat(area)) == area : Nat}}

def planes_ready(context: Planes) -> Data:
  Planes{input,+pixels,+area,count} = context
  PlanesReady<pixels,area>

def planes_reader(context: Planes) -> Buffer.Model<Tile.Tile8>:
  Planes{input,pixels,area,count} = context
  input

def planes_step(plane: U32,context: Planes,state: Packed) -> Packed:
  Planes{input,pixels,area,count} = context
  pixel_plane(plane,pixels,state)

def planes_project(context: Planes) -> Kernel.Pixels:
  Planes{input,pixels,area,count} = context
  pixels

def plane_samples(+input: Buffer.Model<Tile.Tile8>,+pixels: Kernel.Pixels,+plane: Nat) -> IndexedValues.Sampled<Buffer.Model<Tile.Tile8>,Kernel.Plane>:
  IndexedValues.Sampled{input,Kernel.plane_of(U32.from_nat(plane),pixels)}

def planes_row(context: Planes,plane: Nat) -> List<&2,Tile.Tile8>:
  Planes{+input,+pixels,+area,count} = context
  +plane = plane
  ListRead.enumerate(~Tile.Tile8,~IndexedValues.Sampled<Buffer.Model<Tile.Tile8>,Kernel.Plane>,~IndexedValues.sampled(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Kernel.Plane,~plane_sample),
    area,Nat.mul(plane,area),plane_samples(input,pixels,plane))

def planes_width(context: Planes) -> Nat:
  Planes{input,pixels,area,count} = context
  area

def planes_count(context: Planes) -> Nat:
  Planes{input,pixels,area,count} = context
  count

def planes_length(+context: Planes,+plane: Nat) -> {List.length(&2,Tile.Tile8,planes_row(context,plane)) == planes_width(context) : Nat}:
  Planes{+input,+pixels,+area,+count} = context
  ListRead.enumerate_length(~Tile.Tile8,~IndexedValues.Sampled<Buffer.Model<Tile.Tile8>,Kernel.Plane>,~IndexedValues.sampled(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Kernel.Plane,~plane_sample),
    area,Nat.mul(plane,area),plane_samples(input,pixels,plane))

def planes_same(+plane: U32,+context: Planes,+state: Packed) -> {planes_step(plane,context,state) == pixel_plane(plane,planes_project(context),state) : Packed}:
  Planes{+input,+pixels,+area,+count} = context
  {==}

def planes_written(+context: Planes,+ready: planes_ready(context),+plane: Nat,+inside: {Nat.is_lt(plane,planes_count(context)) == True{} : Bool},+tree: Tree.Tree<Tile.Tile8>) ->
  {planes_step(U32.from_nat(plane),context,Packed{planes_reader(context),tree}) ==
    Packed{planes_reader(context),Tree.write_sequence(~Tile.Tile8,planes_row(context,plane),U32.from_nat(Nat.mul(plane,planes_width(context))),tree)} : Packed}:
  Planes{+input,+pixels,+area,+count} = context
  Kernel.Pixels{+geometry,+base,+phases,+first_row,+band_rows,+band_columns} = pixels
  PlanesReady{+area_exact,+area_back} = ready
  +word = U32.from_nat(plane)
  +whole = {Kernel.Pixels{geometry,base,phases,first_row,band_rows,band_columns} : Kernel.Pixels}
  +context_plane = Kernel.plane_of(word,whole)
  +start = Nat.mul(plane,area)
  +values = ListRead.enumerate(~Tile.Tile8,~IndexedValues.Sampled<Buffer.Model<Tile.Tile8>,Kernel.Plane>,~IndexedValues.sampled(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Kernel.Plane,~plane_sample),
    area,start,IndexedValues.Sampled{input,context_plane})
  +wanted = {Packed{input,Tree.write_sequence(~Tile.Tile8,values,U32.from_nat(start),tree)} : Packed}
  %Equal.sym(U32,(word * band_rows * band_columns : U32),(word * (band_rows * band_columns : U32) : U32),Arithmetic.multiply_associative(word,band_rows,band_columns)) :
    {Packed{input,read_run(~Kernel.Plane,~plane_sample,U32.to_nat((band_rows * band_columns : U32)),_,context_plane,input,tree)} == wanted : Packed}
  %Equal.sym(U32,(band_rows * band_columns : U32),U32.from_nat(area),area_exact) :
    {Packed{input,read_run(~Kernel.Plane,~plane_sample,U32.to_nat(_),(word * _ : U32),context_plane,input,tree)} == wanted : Packed}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(area)),area,area_back) :
    {Packed{input,read_run(~Kernel.Plane,~plane_sample,_,(word * U32.from_nat(area) : U32),context_plane,input,tree)} == wanted : Packed}
  %Equal.sym(U32,(word * U32.from_nat(area) : U32),U32.from_nat(start),Multiplication.native_encoded_multiply(plane,area)) :
    {Packed{input,read_run(~Kernel.Plane,~plane_sample,area,_,context_plane,input,tree)} == wanted : Packed}
  Equal.cong(Tree.Tree<Tile.Tile8>,Packed,output => Packed{input,output},read_run(~Kernel.Plane,~plane_sample,area,U32.from_nat(start),context_plane,input,tree),
    Tree.write_sequence(~Tile.Tile8,values,U32.from_nat(start),tree),read_sequence(~Kernel.Plane,~plane_sample,area,start,context_plane,input,tree))

# The planes of a pixel band are the samples of each plane in plane order.
def planes_points(+context: Planes,+ready: planes_ready(context),+tree: Tree.Tree<Tile.Tile8>) ->
  {Traversal.indexed_fold(~Packed,~Kernel.Pixels,~pixel_plane,planes_count(context),U32.from_nat(0n),planes_project(context),Packed{planes_reader(context),tree}) ==
    Packed{planes_reader(context),Tree.write_sequence(~Tile.Tile8,List.concat(&2,Tile.Tile8,ListRead.enumerate(~List<&2,Tile.Tile8>,~Planes,~planes_row,planes_count(context),0n,context)),
      U32.from_nat(0n),tree)} : Packed}:
  +count = planes_count(context)
  %Traversals.indexed_fold_change_context(~Packed,~Planes,~Kernel.Pixels,~planes_project,~planes_step,~pixel_plane,~planes_same,count,U32.from_nat(0n),context,
      Packed{planes_reader(context),tree}) :
    {_ == Packed{planes_reader(context),Tree.write_sequence(~Tile.Tile8,List.concat(&2,Tile.Tile8,ListRead.enumerate(~List<&2,Tile.Tile8>,~Planes,~planes_row,count,0n,context)),
      U32.from_nat(0n),tree)} : Packed}
  blocks_points(~Planes,~planes_ready,~planes_reader,~planes_step,~planes_row,~planes_width,~planes_count,~planes_written,~planes_length,count,0n,context,ready,tree,
    Order.reflexive(count))

# ---- Gradient cells ----

def gradient_sample(gradient: Buffer.Model<Tile.Tile8>,context: Kernel.Positions,position: U32) -> Tile.Tile8:
  Kernel.Positions{+base,+groups,+count} = context
  +position = position
  Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,gradient),(base + position % count * groups + position / count : U32))

def gradient_read_matches(+gradient: Buffer.Model<Tile.Tile8>,+context: Kernel.Positions,+position: U32) ->
  {Kernel.gradient_read(Buffer.pack(~Tile.Tile8,gradient),context,position) == (Buffer.pack(~Tile.Tile8,gradient),gradient_sample(gradient,context,position)) :
    Storage.Buffer<Tile.Tile8> & Tile.Tile8}:
  Kernel.Positions{+base,+groups,+count} = context
  +position = position
  Buffer.read(~Tile.Tile8,gradient,(base + position % count * groups + position / count : U32))

# The gradient cells of every output block of a group, tile b * count + t
# the cell of block b at position t.
def gradients(+context: Kernel.Positions,+count: Nat,gradient: Buffer.Model<Tile.Tile8>,tree: Tree.Tree<Tile.Tile8>) -> Packed:
  +gradient = gradient
  Packed{gradient,read_run(~Kernel.Positions,~gradient_sample,count,0,context,gradient,tree)}

def gradients_matches(+context: Kernel.Positions,+count: Nat,+gradient: Buffer.Model<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) ->
  {Traversal.finish(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Kernel.Positions,
    Traversal.run(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Kernel.Positions,~Kernel.gradient_read,~TraversalBlocks.in_place(~Kernel.Positions),count,
      Traversal.State{Buffer.pack(~Tile.Tile8,gradient),Tree.pack(Tile.Tile8,tree),0,context})) == owners(gradients(context,count,gradient,tree)) :
    Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>}:
  read_owners(~Kernel.Positions,~Kernel.gradient_read,~gradient_sample,~gradient_read_matches,count,0,context,gradient,tree)

# Inside the count, gradient tile i holds its sample.
def gradients_stored(+context: Kernel.Positions,+count: Nat,+gradient: Buffer.Model<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,+position: Nat,
  capacity_equal: {Tree.capacity(Tile.Tile8,tree) == capacity : U32},balanced: Shape.Balanced(Tile.Tile8,depth,tree),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},room: {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool},
  +selected: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,packed_output(gradients(context,count,gradient,tree)),U32.from_nat(position)) == gradient_sample(gradient,context,U32.from_nat(position)) : Tile.Tile8}:
  IndexedValues.stored(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Kernel.Positions,~gradient_sample,~TraversalBlocks.in_place(~Kernel.Positions),~TiledValues.in_place_identity(~Kernel.Positions),
    count,context,gradient,tree,depth,capacity,position,capacity_equal,balanced,depth_ok,capacity_ok,room,selected)

# ---- Packing keeps its reader ----

def same_input(~Context: Data,index: U32,context: Context,input: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  input

def same_input_fixed(~Context: Data,+index: U32,+context: Context,+input: Buffer.Model<Tile.Tile8>) -> {same_input(~Context,index,context,input) == input : Buffer.Model<Tile.Tile8>}:
  {==}

# A fold whose every step keeps the reader keeps it.
def fold_kept(~Context: Data,~step: U32 -> Context -> Packed -> Packed,
  ~kept: @+index: U32 -> @+context: Context -> @+state: Packed -> {packed_input(step(index,context,state)) == packed_input(state) : Buffer.Model<Tile.Tile8>},
  +count: Nat,+first: U32,+context: Context,+state: Packed) ->
  {packed_input(Traversal.indexed_fold(~Packed,~Context,~step,count,first,context,state)) == packed_input(state) : Buffer.Model<Tile.Tile8>}:
  Equal.trans(Buffer.Model<Tile.Tile8>,packed_input(Traversal.indexed_fold(~Packed,~Context,~step,count,first,context,state)),
    Traversal.indexed_fold(~Buffer.Model<Tile.Tile8>,~Context,~same_input(~Context),count,first,context,packed_input(state)),packed_input(state),
    Traversals.indexed_fold_observes(~Packed,~Buffer.Model<Tile.Tile8>,~Context,~step,~packed_input,~same_input(~Context),~kept,count,first,context,state),
    Traversals.indexed_fold_fixed(~Buffer.Model<Tile.Tile8>,~Context,~same_input(~Context),~same_input_fixed(~Context),count,first,context,packed_input(state)))

def pixel_plane_kept(+plane: U32,+pixels: Kernel.Pixels,+state: Packed) -> {packed_input(pixel_plane(plane,pixels,state)) == packed_input(state) : Buffer.Model<Tile.Tile8>}:
  Kernel.Pixels{+geometry,+base,+phases,+first_row,+band_rows,+band_columns} = pixels
  Packed{+input,+output} = state
  {==}

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

def geometry_kernel(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  k

def geometry_taps(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  taps

# The scalar address model is used only in proofs; integer products may wrap
# in the native reader while its final encoded address remains the same.
# Channel channel at input row y and column x of the cells of a source from
# base, read cell by cell, the groups of a pixel side by side, zero outside
# the image.
def sampled_cell(~Source: Data,~cell: Source -> Nat -> Tile.Tile8,source: Source,base: Nat,+channel: Nat,+y: Nat,+x: Nat,height: Nat,+width: Nat,+padding: Nat,groups: Nat) -> F32:
  Bool.pick(F32,(Nat.is_le(padding,y) && Nat.is_lt(y,Nat.add(padding,height))) && (Nat.is_le(padding,x) && Nat.is_lt(x,Nat.add(padding,width))),
    Tiles.select_lane(Nat.mod(channel,8n),cell(source,Nat.add(Nat.add(base,Nat.div(channel,8n)),Nat.mul(Nat.add(Nat.mul(Nat.sub(y,padding),width),Nat.sub(x,padding)),groups)))),0.0)

def model_cell(input: Buffer.Model<Tile.Tile8>,index: Nat) -> Tile.Tile8:
  Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),U32.from_nat(index))

def sampled_point(input: Buffer.Model<Tile.Tile8>,base: Nat,+channel: Nat,+y: Nat,+x: Nat,height: Nat,+width: Nat,+padding: Nat,groups: Nat) -> F32:
  sampled_cell(~Buffer.Model<Tile.Tile8>,~model_cell,input,base,channel,y,x,height,width,padding,groups)

# Two sources read the same sample when they hold the same cell there inside
# the image.
def picked_agrees(inside: Bool,+left: F32,+right: F32,same: @inside_true: {inside == True{} : Bool} -> {left == right : F32}) ->
  {Bool.pick(F32,inside,left,0.0) == Bool.pick(F32,inside,right,0.0) : F32}:
  match inside:
    case True{}: same({==})
    case False{}: {==}

def sampled_agrees(~Left: Data,~left_cell: Left -> Nat -> Tile.Tile8,~Right: Data,~right_cell: Right -> Nat -> Tile.Tile8,
  +left: Left,+right: Right,+left_base: Nat,+right_base: Nat,+channel: Nat,+y: Nat,+x: Nat,+height: Nat,+width: Nat,+padding: Nat,+groups: Nat,
  same: @inside_true: {(Nat.is_le(padding,y) && Nat.is_lt(y,Nat.add(padding,height))) && (Nat.is_le(padding,x) && Nat.is_lt(x,Nat.add(padding,width))) == True{} : Bool} ->
    {left_cell(left,Nat.add(Nat.add(left_base,Nat.div(channel,8n)),Nat.mul(Nat.add(Nat.mul(Nat.sub(y,padding),width),Nat.sub(x,padding)),groups))) ==
      right_cell(right,Nat.add(Nat.add(right_base,Nat.div(channel,8n)),Nat.mul(Nat.add(Nat.mul(Nat.sub(y,padding),width),Nat.sub(x,padding)),groups))) : Tile.Tile8}) ->
  {sampled_cell(~Left,~left_cell,left,left_base,channel,y,x,height,width,padding,groups) == sampled_cell(~Right,~right_cell,right,right_base,channel,y,x,height,width,padding,groups) : F32}:
  +offset = Nat.mul(Nat.add(Nat.mul(Nat.sub(y,padding),width),Nat.sub(x,padding)),groups)
  +left_tile = left_cell(left,Nat.add(Nat.add(left_base,Nat.div(channel,8n)),offset))
  +right_tile = right_cell(right,Nat.add(Nat.add(right_base,Nat.div(channel,8n)),offset))
  picked_agrees((Nat.is_le(padding,y) && Nat.is_lt(y,Nat.add(padding,height))) && (Nat.is_le(padding,x) && Nat.is_lt(x,Nat.add(padding,width))),
    Tiles.select_lane(Nat.mod(channel,8n),left_tile),Tiles.select_lane(Nat.mod(channel,8n),right_tile),
    inside_true => Equal.cong(Tile.Tile8,F32,tile => Tiles.select_lane(Nat.mod(channel,8n),tile),left_tile,right_tile,same(inside_true)))

# The input value that tap tap reads at an output row and column, from the
# cells of an image of a source from base.
def tap_cell(~Source: Data,~cell: Source -> Nat -> Tile.Tile8,source: Source,+geometry: Kernel.Geometry,+base: Nat,+tap: U32,row: U32,column: U32) -> F32:
  Kernel.Geometry{c,o,h,w,+k,+s,p,rows,columns,taps} = geometry
  sampled_cell(~Source,~cell,source,base,Nat.div(U32.to_nat(tap),Nat.mul(U32.to_nat(k),U32.to_nat(k))),
    Nat.add(Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(k)),Nat.mul(U32.to_nat(row),U32.to_nat(s))),
    Nat.add(Nat.mod(U32.to_nat(tap),U32.to_nat(k)),Nat.mul(U32.to_nat(column),U32.to_nat(s))),U32.to_nat(h),U32.to_nat(w),U32.to_nat(p),U32.to_nat(((c + 7) / 8 : U32)))

def natural_tap_value(input: Buffer.Model<Tile.Tile8>,+geometry: Kernel.Geometry,+base: U32,+tap: U32,row: U32,column: U32) -> F32:
  tap_cell(~Buffer.Model<Tile.Tile8>,~model_cell,input,geometry,U32.to_nat(base),tap,row,column)

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

# ---- Pixel samples ----

# Input pixel (row * stride + kernel_row - padding, column * stride +
# kernel_column - padding) of the cells from base, groups cells a pixel, for a
# tile row below rows and a tile column below columns; zero outside the image.
type NumericPixel is Data:
  NumericPixel{input: Buffer.Model<Tile.Tile8>,base: Nat,row: U32,column: U32,kernel_row: U32,kernel_column: U32,height: U32,width: U32,stride: U32,padding: U32,
    groups: U32,rows: U32,columns: U32}

def numeric_native_mask(pixel: NumericPixel) -> Bool:
  NumericPixel{input,base,+row,+column,+kr,+kc,+h,+w,+s,+p,groups,rows,columns} = pixel
  U32.is_le(Kernel.axis_first(kr,s,p),row) && U32.is_lt(row,Kernel.axis_end(kr,s,p,h,rows)) && U32.is_le(Kernel.axis_first(kc,s,p),column) && U32.is_lt(column,Kernel.axis_end(kc,s,p,w,columns))

def axis_inside(+offset: U32,+stride: U32,+padding: U32,size: U32,+position: U32) -> Bool:
  Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) && Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(size)))

def numeric_inside(pixel: NumericPixel) -> Bool:
  NumericPixel{input,base,row,column,kr,kc,h,w,+s,+p,groups,rows,columns} = pixel
  axis_inside(kr,s,p,h,row) && axis_inside(kc,s,p,w,column)

def numeric_stride(pixel: NumericPixel) -> U32:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,groups,rows,columns} = pixel
  s

def numeric_row_within(pixel: NumericPixel) -> Bool:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,groups,rows,columns} = pixel
  Nat.is_lt(U32.to_nat(row),U32.to_nat(rows))

def numeric_column_within(pixel: NumericPixel) -> Bool:
  NumericPixel{input,base,row,column,kr,kc,h,w,s,p,groups,rows,columns} = pixel
  Nat.is_lt(U32.to_nat(column),U32.to_nat(columns))

type PixelReady<-pixel: NumericPixel> is Data:
  PixelReady{positive: {Nat.is_lt(0n,U32.to_nat(numeric_stride(pixel))) == True{} : Bool},row: {numeric_row_within(pixel) == True{} : Bool},column: {numeric_column_within(pixel) == True{} : Bool}}

def numeric_mask_matches(+pixel: NumericPixel,ready: PixelReady<pixel>) -> {numeric_native_mask(pixel) == numeric_inside(pixel) : Bool}:
  NumericPixel{input,base,+row,+column,+kr,+kc,+h,+w,+s,+p,groups,+rows,+columns} = pixel
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
  NumericPixel{input,base,row,column,kr,kc,height,+width,+stride,+padding,groups,rows,columns} = pixel
  (U32.from_nat(base) + ((row * stride + kr - padding) * width + (column * stride + kc - padding)) * groups : U32)

def numeric_index(pixel: NumericPixel) -> Nat:
  NumericPixel{input,base,row,column,kr,kc,height,width,+stride,+padding,groups,rows,columns} = pixel
  Nat.add(base,Nat.mul(Nat.add(Nat.mul(Nat.sub(Nat.add(U32.to_nat(kr),Nat.mul(U32.to_nat(row),U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(width)),
    Nat.sub(Nat.add(U32.to_nat(kc),Nat.mul(U32.to_nat(column),U32.to_nat(stride))),U32.to_nat(padding))),U32.to_nat(groups)))

def numeric_tree(pixel: NumericPixel) -> Tree.Tree<Tile.Tile8>:
  NumericPixel{input,base,row,column,kr,kc,height,width,stride,padding,groups,rows,columns} = pixel
  Buffer.tree(~Tile.Tile8,input)

# The offset of a pixel inside the image within its plane, as a word.
def offset_word(+row: U32,+column: U32,+kernel_row: U32,+kernel_column: U32,+width: U32,+stride: U32,+padding: U32,
  +row_lower: {Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(kernel_row),Nat.mul(U32.to_nat(row),U32.to_nat(stride)))) == True{} : Bool},
  +column_lower: {Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(kernel_column),Nat.mul(U32.to_nat(column),U32.to_nat(stride)))) == True{} : Bool}) ->
  {((row * stride + kernel_row - padding) * width + (column * stride + kernel_column - padding) : U32) ==
    U32.from_nat(Nat.add(Nat.mul(Nat.sub(Nat.add(U32.to_nat(kernel_row),Nat.mul(U32.to_nat(row),U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(width)),
      Nat.sub(Nat.add(U32.to_nat(kernel_column),Nat.mul(U32.to_nat(column),U32.to_nat(stride))),U32.to_nat(padding)))) : U32}:
  +offset = Nat.add(Nat.mul(Nat.sub(Nat.add(U32.to_nat(kernel_row),Nat.mul(U32.to_nat(row),U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(width)),
    Nat.sub(Nat.add(U32.to_nat(kernel_column),Nat.mul(U32.to_nat(column),U32.to_nat(stride))),U32.to_nat(padding)))
  %Arithmetic.add_zero_left(((row * stride + kernel_row - padding) * width : U32)) :
    {(_ + (column * stride + kernel_column - padding) : U32) == U32.from_nat(offset) : U32}
  Encoded.padded_pixel_address(0n,row,column,kernel_row,kernel_column,width,stride,padding,row_lower,column_lower)

def numeric_index_matches(+pixel: NumericPixel,+inside: {numeric_inside(pixel) == True{} : Bool}) -> {numeric_native_index(pixel) == U32.from_nat(numeric_index(pixel)) : U32}:
  NumericPixel{input,+base,+row,+column,+kr,+kc,+height,+width,+stride,+padding,+groups,rows,columns} = pixel
  +row_inside = Logic.left(axis_inside(kr,stride,padding,height,row),axis_inside(kc,stride,padding,width,column),inside)
  +column_inside = Logic.right(axis_inside(kr,stride,padding,height,row),axis_inside(kc,stride,padding,width,column),inside)
  +offset = Nat.add(Nat.mul(Nat.sub(Nat.add(U32.to_nat(kr),Nat.mul(U32.to_nat(row),U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(width)),
    Nat.sub(Nat.add(U32.to_nat(kc),Nat.mul(U32.to_nat(column),U32.to_nat(stride))),U32.to_nat(padding)))
  +wanted = U32.from_nat(Nat.add(base,Nat.mul(offset,U32.to_nat(groups))))
  %Equal.sym(U32,((row * stride + kr - padding) * width + (column * stride + kc - padding) : U32),U32.from_nat(offset),
      offset_word(row,column,kr,kc,width,stride,padding,
        Logic.left(Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(kr),Nat.mul(U32.to_nat(row),U32.to_nat(stride)))),Nat.is_lt(Nat.add(U32.to_nat(kr),Nat.mul(U32.to_nat(row),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(height))),row_inside),
        Logic.left(Nat.is_le(U32.to_nat(padding),Nat.add(U32.to_nat(kc),Nat.mul(U32.to_nat(column),U32.to_nat(stride)))),Nat.is_lt(Nat.add(U32.to_nat(kc),Nat.mul(U32.to_nat(column),U32.to_nat(stride))),Nat.add(U32.to_nat(padding),U32.to_nat(width))),column_inside))) :
    {(U32.from_nat(base) + _ * groups : U32) == wanted : U32}
  %Arithmetic.native_roundtrip(groups) : {(U32.from_nat(base) + U32.from_nat(offset) * _ : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(offset) * U32.from_nat(U32.to_nat(groups)) : U32),U32.from_nat(Nat.mul(offset,U32.to_nat(groups))),
      Multiplication.native_encoded_multiply(offset,U32.to_nat(groups))) :
    {(U32.from_nat(base) + _ : U32) == wanted : U32}
  WordArithmetic.native_add(base,Nat.mul(offset,U32.to_nat(groups)))

def numeric_selected(inside: Bool,+pixel: NumericPixel,equation: {numeric_inside(pixel) == inside : Bool}) ->
  {Bool.pick(Tile.Tile8,inside,Tree.read(Tile.Tile8,numeric_tree(pixel),numeric_native_index(pixel)),Tile.filled(0.0)) ==
    Bool.pick(Tile.Tile8,inside,Tree.read(Tile.Tile8,numeric_tree(pixel),U32.from_nat(numeric_index(pixel))),Tile.filled(0.0)) : Tile.Tile8}:
  match inside:
    case False{}: {==}
    case True{}:
      Equal.cong(U32,Tile.Tile8,index => Tree.read(Tile.Tile8,numeric_tree(pixel),index),numeric_native_index(pixel),U32.from_nat(numeric_index(pixel)),numeric_index_matches(pixel,equation))

def numeric_cell(pixel: NumericPixel) -> Tile.Tile8:
  +pixel = pixel
  Bool.pick(Tile.Tile8,numeric_inside(pixel),Tree.read(Tile.Tile8,numeric_tree(pixel),U32.from_nat(numeric_index(pixel))),Tile.filled(0.0))

# The cell of channel block block in phase (row_offset, column_offset) at a
# tile row and column of a pixel band, the cells of the image from base.
def cell_pixel(+input: Buffer.Model<Tile.Tile8>,+block: U32,+row_offset: U32,+column_offset: U32,+row_end: U32,+column_end: U32,+geometry: Kernel.Geometry,+base: U32,
  +row: U32,+column: U32) -> NumericPixel:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  NumericPixel{input,Nat.add(U32.to_nat(base),U32.to_nat(block)),row,column,row_offset,column_offset,h,w,s,p,((c + 7) / 8 : U32),row_end,column_end}

# The plane of a block and phase in a band from first_row of band_columns
# columns, its cells from start.
def block_plane(+block: U32,+row_offset: U32,+column_offset: U32,+row_end: U32,+column_end: U32,+geometry: Kernel.Geometry,+base: U32,
  +first_row: U32,+band_columns: U32,+start: U32) -> Kernel.Plane:
  Kernel.Geometry{c,o,h,+w,k,+s,+p,rows,columns,taps} = geometry
  Kernel.Plane{(base + block : U32),(row_offset - p : U32),(column_offset - p : U32),s,w,((c + 7) / 8 : U32),
    Kernel.axis_first(row_offset,s,p),Kernel.axis_end(row_offset,s,p,h,row_end),Kernel.axis_first(column_offset,s,p),Kernel.axis_end(column_offset,s,p,w,column_end),
    first_row,band_columns,start}

# A position of the plane at a tile row and column reads the masked native
# cell of its pixel.
def plane_native(+input: Buffer.Model<Tile.Tile8>,+block: U32,+row_offset: U32,+column_offset: U32,+row_end: U32,+column_end: U32,+geometry: Kernel.Geometry,+base: U32,
  +first_row: U32,+band_columns: U32,+start: U32,+position: U32,+row: U32,+column: U32,
  +row_exact: {(first_row + (position - start : U32) / band_columns : U32) == row : U32},+column_exact: {((position - start : U32) % band_columns : U32) == column : U32}) ->
  {plane_sample(input,block_plane(block,row_offset,column_offset,row_end,column_end,geometry,base,first_row,band_columns,start),position) ==
    Bool.pick(Tile.Tile8,numeric_native_mask(cell_pixel(input,block,row_offset,column_offset,row_end,column_end,geometry,base,row,column)),
      Tree.read(Tile.Tile8,numeric_tree(cell_pixel(input,block,row_offset,column_offset,row_end,column_end,geometry,base,row,column)),
        numeric_native_index(cell_pixel(input,block,row_offset,column_offset,row_end,column_end,geometry,base,row,column))),Tile.filled(0.0)) : Tile.Tile8}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +tree = Buffer.tree(~Tile.Tile8,input)
  +groups = ((c + 7) / 8 : U32)
  +first_cell = (base + block : U32)
  +number = Nat.add(U32.to_nat(base),U32.to_nat(block))
  +rf = Kernel.axis_first(row_offset,s,p)
  +re = Kernel.axis_end(row_offset,s,p,h,row_end)
  +cf = Kernel.axis_first(column_offset,s,p)
  +ce = Kernel.axis_end(column_offset,s,p,w,column_end)
  +tile_row = (first_row + (position - start : U32) / band_columns : U32)
  +tile_column = ((position - start : U32) % band_columns : U32)
  +down = (row * s : U32)
  +across = (column * s : U32)
  +wanted = Bool.pick(Tile.Tile8,U32.is_le(rf,row) && U32.is_lt(row,re) && U32.is_le(cf,column) && U32.is_lt(column,ce),
    Tree.read(Tile.Tile8,tree,(U32.from_nat(number) + ((down + row_offset - p) * w + (across + column_offset - p)) * groups : U32)),Tile.filled(0.0))
  %Equal.sym(U32,tile_row,row,row_exact) :
    {Bool.pick(Tile.Tile8,U32.is_le(rf,_) && U32.is_lt(_,re) && U32.is_le(cf,tile_column) && U32.is_lt(tile_column,ce),
      Tree.read(Tile.Tile8,tree,(first_cell + ((_ * s + (row_offset - p : U32)) * w + tile_column * s + (column_offset - p : U32)) * groups : U32)),Tile.filled(0.0)) == wanted : Tile.Tile8}
  %Equal.sym(U32,tile_column,column,column_exact) :
    {Bool.pick(Tile.Tile8,U32.is_le(rf,row) && U32.is_lt(row,re) && U32.is_le(cf,_) && U32.is_lt(_,ce),
      Tree.read(Tile.Tile8,tree,(first_cell + ((down + (row_offset - p : U32)) * w + _ * s + (column_offset - p : U32)) * groups : U32)),Tile.filled(0.0)) == wanted : Tile.Tile8}
  %Arithmetic.subtract_from_sum(down,row_offset,p) :
    {Bool.pick(Tile.Tile8,U32.is_le(rf,row) && U32.is_lt(row,re) && U32.is_le(cf,column) && U32.is_lt(column,ce),
      Tree.read(Tile.Tile8,tree,(first_cell + (_ * w + across + (column_offset - p : U32)) * groups : U32)),Tile.filled(0.0)) == wanted : Tile.Tile8}
  +head = ((down + row_offset - p : U32) * w : U32)
  %Equal.sym(U32,((head + across : U32) + (column_offset - p : U32) : U32),(head + (across + (column_offset - p : U32) : U32) : U32),
      Arithmetic.add_associative(head,across,(column_offset - p : U32))) :
    {Bool.pick(Tile.Tile8,U32.is_le(rf,row) && U32.is_lt(row,re) && U32.is_le(cf,column) && U32.is_lt(column,ce),
      Tree.read(Tile.Tile8,tree,(first_cell + _ * groups : U32)),Tile.filled(0.0)) == wanted : Tile.Tile8}
  %Arithmetic.subtract_from_sum(across,column_offset,p) :
    {Bool.pick(Tile.Tile8,U32.is_le(rf,row) && U32.is_lt(row,re) && U32.is_le(cf,column) && U32.is_lt(column,ce),
      Tree.read(Tile.Tile8,tree,(first_cell + (head + _) * groups : U32)),Tile.filled(0.0)) == wanted : Tile.Tile8}
  +first_exact = {%Arithmetic.native_roundtrip(base) : {(_ + block : U32) == U32.from_nat(number) : U32}
    %Arithmetic.native_roundtrip(block) : {(U32.from_nat(U32.to_nat(base)) + _ : U32) == U32.from_nat(number) : U32}
    WordArithmetic.native_add(U32.to_nat(base),U32.to_nat(block)) : {first_cell == U32.from_nat(number) : U32}}
  %Equal.sym(U32,first_cell,U32.from_nat(number),first_exact) :
    {Bool.pick(Tile.Tile8,U32.is_le(rf,row) && U32.is_lt(row,re) && U32.is_le(cf,column) && U32.is_lt(column,ce),
      Tree.read(Tile.Tile8,tree,(_ + ((down + row_offset - p : U32) * w + (across + column_offset - p : U32)) * groups : U32)),Tile.filled(0.0)) == wanted : Tile.Tile8}
  {==}

# A position of the plane at a tile row and column inside the band holds
# the cell of its pixel.
def plane_cell(+input: Buffer.Model<Tile.Tile8>,+block: U32,+row_offset: U32,+column_offset: U32,+row_end: U32,+column_end: U32,+geometry: Kernel.Geometry,+base: U32,
  +first_row: U32,+band_columns: U32,+start: U32,+position: U32,+row: U32,+column: U32,
  +row_exact: {(first_row + (position - start : U32) / band_columns : U32) == row : U32},+column_exact: {((position - start : U32) % band_columns : U32) == column : U32},
  +ready: PixelReady<cell_pixel(input,block,row_offset,column_offset,row_end,column_end,geometry,base,row,column)>) ->
  {plane_sample(input,block_plane(block,row_offset,column_offset,row_end,column_end,geometry,base,first_row,band_columns,start),position) ==
    numeric_cell(cell_pixel(input,block,row_offset,column_offset,row_end,column_end,geometry,base,row,column)) : Tile.Tile8}:
  +pixel = cell_pixel(input,block,row_offset,column_offset,row_end,column_end,geometry,base,row,column)
  +native = Tree.read(Tile.Tile8,numeric_tree(pixel),numeric_native_index(pixel))
  %Equal.sym(Tile.Tile8,plane_sample(input,block_plane(block,row_offset,column_offset,row_end,column_end,geometry,base,first_row,band_columns,start),position),
      Bool.pick(Tile.Tile8,numeric_native_mask(pixel),native,Tile.filled(0.0)),
      plane_native(input,block,row_offset,column_offset,row_end,column_end,geometry,base,first_row,band_columns,start,position,row,column,row_exact,column_exact)) :
    {_ == numeric_cell(pixel) : Tile.Tile8}
  %Equal.sym(Bool,numeric_native_mask(pixel),numeric_inside(pixel),numeric_mask_matches(pixel,ready)) : {Bool.pick(Tile.Tile8,_,native,Tile.filled(0.0)) == numeric_cell(pixel) : Tile.Tile8}
  numeric_selected(numeric_inside(pixel),pixel,{==})

# A lane of a picked cell is the picked value when the lane of the cell is
# that value.
def lane_picked(inside: Bool,+lane: Nat,+tile: Tile.Tile8,+value: F32,+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  same: @inside_true: {inside == True{} : Bool} -> {Tiles.select_lane(lane,tile) == value : F32}) ->
  {Tiles.select_lane(lane,Bool.pick(Tile.Tile8,inside,tile,Tile.filled(0.0))) == Bool.pick(F32,inside,value,0.0) : F32}:
  match inside:
    case True{}: same({==})
    case False{}: filled_lane(lane,0.0,small)

# Lane lane of the cell of a pixel of block block is channel 8 * block + lane
# of the cells from base there, zero outside the image.
def cell_pixel_lane(+input: Buffer.Model<Tile.Tile8>,+block: U32,+row_offset: U32,+column_offset: U32,+row_end: U32,+column_end: U32,
  +c: U32,+o: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+rows: U32,+columns: U32,+taps: U32,+base: U32,+row: U32,+column: U32,+lane: Nat,
  +small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Tiles.select_lane(lane,numeric_cell(cell_pixel(input,block,row_offset,column_offset,row_end,column_end,Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},base,row,column))) ==
    sampled_point(input,U32.to_nat(base),Nat.add(Nat.mul(U32.to_nat(block),8n),lane),
      Nat.add(U32.to_nat(row_offset),Nat.mul(U32.to_nat(row),U32.to_nat(s))),Nat.add(U32.to_nat(column_offset),Nat.mul(U32.to_nat(column),U32.to_nat(s))),
      U32.to_nat(h),U32.to_nat(w),U32.to_nat(p),U32.to_nat(((c + 7) / 8 : U32))) : F32}:
  +pixel = cell_pixel(input,block,row_offset,column_offset,row_end,column_end,Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},base,row,column)
  +tree = Buffer.tree(~Tile.Tile8,input)
  +b = U32.to_nat(block)
  +channel = Nat.add(Nat.mul(b,8n),lane)
  +offset = Nat.add(Nat.mul(Nat.sub(Nat.add(U32.to_nat(row_offset),Nat.mul(U32.to_nat(row),U32.to_nat(s))),U32.to_nat(p)),U32.to_nat(w)),
    Nat.sub(Nat.add(U32.to_nat(column_offset),Nat.mul(U32.to_nat(column),U32.to_nat(s))),U32.to_nat(p)))
  +g = U32.to_nat(((c + 7) / 8 : U32))
  +read = Tree.read(Tile.Tile8,tree,U32.from_nat(numeric_index(pixel)))
  lane_picked(numeric_inside(pixel),lane,read,Tiles.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.add(U32.to_nat(base),Nat.div(channel,8n)),Nat.mul(offset,g))))),
    small,
    inside_true =>
      %Equal.sym(Nat,Nat.mod(channel,8n),lane,Radix.remainder(b,lane,8n,{==},small)) :
        {Tiles.select_lane(lane,read) == Tiles.select_lane(_,Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.add(U32.to_nat(base),Nat.div(channel,8n)),Nat.mul(offset,g))))) : F32}
      %Equal.sym(Nat,Nat.div(channel,8n),b,Radix.quotient(b,lane,8n,{==},small)) :
        {Tiles.select_lane(lane,read) == Tiles.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.add(U32.to_nat(base),_),Nat.mul(offset,g))))) : F32}
      {==})

# A tap offset below the kernel is its phase plus phases * its tile rows, with
# phases = min(stride, kernel): the stride when it is below the kernel, else
# the kernel, which leaves the offset as its own phase.
def phase_case(choice: Bool,+offset: Nat,+row: Nat,+stride: Nat,+kernel: Nat,+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +below: {Nat.is_lt(offset,kernel) == True{} : Bool}) ->
  {Nat.add(Nat.mod(offset,Bool.pick(Nat,choice,stride,kernel)),Nat.mul(Nat.add(row,Nat.div(offset,Bool.pick(Nat,choice,stride,kernel))),stride)) ==
    Nat.add(offset,Nat.mul(row,stride)) : Nat}:
  match choice:
    case True{}:
      +quotient = Nat.div(offset,stride)
      +remainder = Nat.mod(offset,stride)
      +lower = Nat.mul(row,stride)
      +rows = Nat.mul(quotient,stride)
      +wanted = Nat.add(offset,lower)
      %Equal.sym(Nat,Nat.mul(Nat.add(row,quotient),stride),Nat.add(lower,rows),RowMajor.distrib(row,quotient,stride)) : {Nat.add(remainder,_) == wanted : Nat}
      %Equal.sym(Nat,Nat.add(remainder,Nat.add(lower,rows)),Nat.add(lower,Nat.add(remainder,rows)),RowMajor.add_swap(remainder,lower,rows)) : {_ == wanted : Nat}
      +whole = Equal.trans(Nat,Nat.add(remainder,rows),Nat.add(rows,remainder),offset,Order.commutative(remainder,rows),
        Equal.sym(Nat,offset,Nat.add(rows,remainder),DivisionOrder.quotient_exact(offset,stride,stride_positive)))
      %Equal.sym(Nat,Nat.add(remainder,rows),offset,whole) : {Nat.add(lower,_) == wanted : Nat}
      Order.commutative(lower,offset)
    case False{}:
      +parts = Division.small(offset,kernel,below)
      %Equal.sym(Nat,Nat.div(offset,kernel),0n,Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(offset,kernel),(0n,offset),parts)) :
        {Nat.add(Nat.mod(offset,kernel),Nat.mul(Nat.add(row,_),stride)) == Nat.add(offset,Nat.mul(row,stride)) : Nat}
      %Equal.sym(Nat,Nat.mod(offset,kernel),offset,Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(offset,kernel),(0n,offset),parts)) :
        {Nat.add(_,Nat.mul(Nat.add(row,0n),stride)) == Nat.add(offset,Nat.mul(row,stride)) : Nat}
      %Equal.sym(Nat,Nat.add(row,0n),row,Addition.zero_right(row)) : {Nat.add(offset,Nat.mul(_,stride)) == Nat.add(offset,Nat.mul(row,stride)) : Nat}
      {==}

def phase_offset(+offset: Nat,+row: Nat,+stride: Nat,+kernel: Nat,+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},+below: {Nat.is_lt(offset,kernel) == True{} : Bool}) ->
  {Nat.add(Nat.mod(offset,Nat.min(stride,kernel)),Nat.mul(Nat.add(row,Nat.div(offset,Nat.min(stride,kernel))),stride)) == Nat.add(offset,Nat.mul(row,stride)) : Nat}:
  %Equal.sym(Nat,Nat.min(stride,kernel),Bool.pick(Nat,Nat.is_lt(stride,kernel),stride,kernel),Extrema.minimum_choice(stride,kernel)) :
    {Nat.add(Nat.mod(offset,_),Nat.mul(Nat.add(row,Nat.div(offset,_)),stride)) == Nat.add(offset,Nat.mul(row,stride)) : Nat}
  phase_case(Nat.is_lt(stride,kernel),offset,row,stride,kernel,stride_positive,below)
