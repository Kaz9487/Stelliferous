# The window pipeline on trees. A call transposes its input cells into the
# planes of its positions, then every block packs its tiles from the planes
# and multiplies them. On values every block fills the same tiles from the
# values of the channels of the input cells.
import Base
import ./storage_buffer_model.bend as Buffer
import ./kernel_pipeline_buffer_shape.bend as BufferShape
import ./kernel_pipeline_projection.bend as Projection
import ./kernel_pipeline_bounds.bend as BlockBounds
import ../storage_buffer.bend as Storage
import ../kernel_pipeline.bend as Pipeline
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../kernel_spatial_packing.bend as Packing
import ../kernel_window.bend as Window
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_readback.bend as GemmReadback
import ./kernel_gemm_jobs.bend as GemmJobs
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./write_program.bend as Writes
import ./kernel_spatial_packing_refinement.bend as PackingModel
import ./kernel_spatial_packing_planes.bend as Planes
import ./kernel_spatial_packing_passes.bend as Passes
import ./kernel_spatial_packing_pixels.bend as Pixels
import ./kernel_spatial_packing_samples.bend as SpatialSamples
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_division.bend as WordDivision
import ./word_multiplication.bend as Multiplication
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_extrema.bend as Extrema
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./word_subtraction.bend as Subtraction
import ./kernel_spatial_packing_call.bend as Call

# ---- The blocks on cells ----

# Each block is a tile product of its rows and span into the whole output.
def block_matrix(+rows: U32,+end: U32,+inner: U32,+start: U32,+span: U32) -> Gemm.TileMatrix:
  Gemm.TileMatrix{rows,span,inner,start,end}

def block_tree(+rows: U32,+end: U32,+inner: U32,+start: U32,+span: U32,+tiles: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,
  +output: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  GemmReadback.cell_tree(block_matrix(rows,end,inner,start,span),tiles,weights,bias,output)

# The blocks of a call on cells: the planes, the packed tiles, the weights,
# the bias and the output.
type Cells is Data:
  Cells{pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<Tile.Tile8>,output: Tree.Tree<Tile.Tile8>}

# A call on cells: its input and its blocks.
type Call is Data:
  Call{input: Tree.Tree<Tile.Tile8>,blocks: Cells}

def represent_blocks(state: Cells) -> Projection.State:
  Cells{pairs,tiles,weights,bias,output} = state
  Projection.State{Tree.pack(Tile.Tile16,pairs),Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias),Tree.pack(Tile.Tile8,output)}

def represent(call: Call) -> Projection.Called:
  Call{input,blocks} = call
  Projection.Called{Tree.pack(Tile.Tile8,input),represent_blocks(blocks)}

# A block packs its tiles from the planes and stores their product.
def cells_step(work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+start: U32,+span: U32,state: Cells) -> Cells:
  Pipeline.WindowWork{+rows,+end,+window} = work
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner,groups} = window
  Cells{+pairs,tiles,+weights,+bias,output} = state
  +packed = PackingModel.packed_tiles(window,planes,U32.to_nat((span / 8 * inner : U32)),pack_start,pairs,tiles)
  Cells{pairs,packed,weights,bias,block_tree(rows,end,inner,start,span,packed,weights,bias,output)}

def kept_cells(pairs: Tree.Tree<Tile.Tile16>,operands: GemmLaws.CellOperands) -> Cells:
  GemmLaws.CellOperands{bias,GemmLaws.Operands{tiles,weights,unused_bias,output}} = operands
  Cells{pairs,tiles,weights,bias,Writes.evaluate(Tile.Tile8,output)}

def kept_cells_matches(+pairs: Tree.Tree<Tile.Tile16>,+operands: GemmLaws.CellOperands) ->
  {Projection.kept(Tree.pack(Tile.Tile16,pairs),GemmLaws.cell_represent(operands)) == represent_blocks(kept_cells(pairs,operands)) : Projection.State}:
  GemmLaws.CellOperands{bias,GemmLaws.Operands{tiles,weights,unused_bias,output}} = operands
  {==}

# A product keeps its operands and stores the product tree.
def product_result(+rows: U32,+end: U32,+inner: U32,+start: U32,+span: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,
  +bias: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>) ->
  {kept_cells(pairs,GemmReadback.cell_run(block_matrix(rows,end,inner,start,span),tiles,weights,bias,output)) ==
    Cells{pairs,tiles,weights,bias,block_tree(rows,end,inner,start,span,tiles,weights,bias,output)} : Cells}:
  +matrix = block_matrix(rows,end,inner,start,span)
  +model = GemmReadback.cell_run(matrix,tiles,weights,bias,output)
  %Equal.sym(GemmLaws.CellOperands,model,GemmJobs.cell_operands(GemmJobs.CellReaders{tiles,weights,bias,Tree.Leaf{0.0}},GemmJobs.cell_program(model)),
    GemmJobs.cell_run_keeps(U32.to_nat(Gemm.tile_jobs(matrix)),0,matrix,GemmJobs.CellReaders{tiles,weights,bias,Tree.Leaf{0.0}},Writes.Initial{output})) :
    {kept_cells(pairs,_) == Cells{pairs,tiles,weights,bias,block_tree(rows,end,inner,start,span,tiles,weights,bias,output)} : Cells}
  {==}

def step_matches(+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+start: U32,+span: U32,+state: Cells) ->
  {Projection.step_window(work,planes,pack_start,start,span,represent_blocks(state)) == represent_blocks(cells_step(work,planes,pack_start,start,span,state)) :
    Projection.State}:
  Pipeline.WindowWork{+rows,+end,Packing.Stencil{+base,+h,+w,+kr,+kc,+s,+ro,+co,+ow,+inner,+groups}} = work
  Cells{+pairs,+tiles,+weights,+bias,+output} = state
  +window = {Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner,groups} : Packing.Stencil}
  +count = U32.to_nat((span / 8 * inner : U32))
  +packed = PackingModel.packed_tiles(window,planes,count,pack_start,pairs,tiles)
  +matrix = block_matrix(rows,end,inner,start,span)
  +wanted = {Cells{pairs,packed,weights,bias,block_tree(rows,end,inner,start,span,packed,weights,bias,output)} : Cells}
  %Equal.sym(Array<Tile.Tile16> & Array<Tile.Tile8>,Packing.pack(window,planes,count,pack_start,Tree.pack(Tile.Tile16,pairs),Tree.pack(Tile.Tile8,tiles)),
    (Tree.pack(Tile.Tile16,pairs),Tree.pack(Tile.Tile8,packed)),PackingModel.pack_matches(window,planes,count,pack_start,pairs,tiles)) :
    {Projection.packed(rows,end,inner,start,span,Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias),Tree.pack(Tile.Tile8,output),_) == represent_blocks(wanted) :
      Projection.State}
  %Equal.sym(Gemm.CellOperands,Gemm.cell_run(U32.to_nat(Gemm.tile_jobs(matrix)),matrix,
      GemmLaws.cell_represent(GemmLaws.CellOperands{bias,GemmLaws.Operands{packed,weights,Tree.Leaf{0.0},Writes.Initial{output}}})),
    GemmLaws.cell_represent(GemmReadback.cell_run(matrix,packed,weights,bias,output)),GemmReadback.cell_native(matrix,packed,weights,bias,output)) :
    {Projection.kept(Tree.pack(Tile.Tile16,pairs),_) == represent_blocks(wanted) : Projection.State}
  %product_result(rows,end,inner,start,span,pairs,packed,weights,bias,output) :
    {Projection.kept(Tree.pack(Tile.Tile16,pairs),GemmLaws.cell_represent(GemmReadback.cell_run(matrix,packed,weights,bias,output))) == represent_blocks(_) :
      Projection.State}
  kept_cells_matches(pairs,GemmReadback.cell_run(matrix,packed,weights,bias,output))

def cells_blocks(remaining: Nat,+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+start: U32,+padded: U32,+span: U32,state: Cells) -> Cells:
  match remaining:
    case 0n: state
    case 1n+rest:
      cells_blocks(rest,work,planes,(pack_start + span : U32),(start + span : U32),padded,span,
        cells_step(work,planes,pack_start,start,U32.min(span,(padded - start : U32)),state))

def blocks_matches(+remaining: Nat,+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+start: U32,+padded: U32,+span: U32,+state: Cells) ->
  {Projection.blocks_window(remaining,work,planes,pack_start,start,padded,span,represent_blocks(state)) ==
    represent_blocks(cells_blocks(remaining,work,planes,pack_start,start,padded,span,state)) : Projection.State}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +clip = U32.min(span,(padded - start : U32))
      +next = cells_step(work,planes,pack_start,start,clip,state)
      %Equal.sym(Projection.State,Projection.step_window(work,planes,pack_start,start,clip,represent_blocks(state)),represent_blocks(next),
        step_matches(work,planes,pack_start,start,clip,state)) :
        {Projection.blocks_window(rest,work,planes,(pack_start + span : U32),(start + span : U32),padded,span,_) ==
          represent_blocks(cells_blocks(rest,work,planes,(pack_start + span : U32),(start + span : U32),padded,span,next)) : Projection.State}
      blocks_matches(rest,work,planes,(pack_start + span : U32),(start + span : U32),padded,span,next)

# The end of the positions of a call and the planes it transposes into.
def padded_end(+work: Pipeline.WindowWork,+offset: U32) -> U32:
  Pipeline.WindowWork{rows,+end,window} = work
  (offset + Window.padded((end - offset : U32)) : U32)

def call_planes(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32) -> Packing.Planes:
  Pipeline.WindowWork{rows,end,+window} = work
  Packing.planes_of(window,pack_start,(padded_end(work,offset) - offset : U32))

def call_span(+work: Pipeline.WindowWork) -> U32:
  Pipeline.WindowWork{rows,end,Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner,groups}} = work
  Pipeline.block_positions(inner)

def call_blocks(+work: Pipeline.WindowWork,+offset: U32) -> Nat:
  U32.to_nat(((padded_end(work,offset) - offset + call_span(work) - 1) / call_span(work) : U32))

def call_window(+work: Pipeline.WindowWork) -> Packing.Stencil:
  Pipeline.WindowWork{rows,end,+window} = work
  window

# A call: the transpose of its input into the planes of its positions, then
# its blocks from the packing start on.
def cells_into(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+input: Tree.Tree<Tile.Tile8>,state: Cells) -> Call:
  Cells{pairs,tiles,weights,bias,output} = state
  +planes = call_planes(work,pack_start,offset)
  Call{input,cells_blocks(call_blocks(work,offset),work,planes,pack_start,offset,padded_end(work,offset),call_span(work),
    Cells{PackingModel.transposed(call_window(work),planes,input,pairs),tiles,weights,bias,output})}

def into_matches(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,
  +weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>) ->
  {Projection.into(work,pack_start,offset,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias),Tree.pack(Tile.Tile16,pairs),
      Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,output)) ==
    represent(cells_into(work,pack_start,offset,input,Cells{pairs,tiles,weights,bias,output})) : Projection.Called}:
  Pipeline.WindowWork{+rows,+end,Packing.Stencil{+base,+h,+w,+kr,+kc,+s,+ro,+co,+ow,+inner,+groups}} = work
  +window = {Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner,groups} : Packing.Stencil}
  +work = {Pipeline.WindowWork{rows,end,window} : Pipeline.WindowWork}
  +padded = (offset + Window.padded((end - offset : U32)) : U32)
  +planes = Packing.planes_of(window,pack_start,(padded - offset : U32))
  +span = Pipeline.block_positions(inner)
  +remaining = U32.to_nat(((padded - offset + span - 1) / span : U32))
  +transposed = PackingModel.transposed(window,planes,input,pairs)
  %Equal.sym(Array<Tile.Tile8> & Array<Tile.Tile16>,Packing.transpose(window,planes,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile16,pairs)),
    (Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile16,transposed)),PackingModel.transpose_matches(window,planes,input,pairs)) :
    {Projection.blocks_of(work,planes,pack_start,offset,padded,Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias),Tree.pack(Tile.Tile8,output),_) ==
      Projection.Called{Tree.pack(Tile.Tile8,input),represent_blocks(cells_blocks(remaining,work,planes,pack_start,offset,padded,span,Cells{transposed,tiles,weights,bias,output}))} :
      Projection.Called}
  Equal.cong(Projection.State,Projection.Called,blocks => Projection.Called{Tree.pack(Tile.Tile8,input),blocks},
    Projection.blocks_window(remaining,work,planes,pack_start,offset,padded,span,represent_blocks(Cells{transposed,tiles,weights,bias,output})),
    represent_blocks(cells_blocks(remaining,work,planes,pack_start,offset,padded,span,Cells{transposed,tiles,weights,bias,output})),
    blocks_matches(remaining,work,planes,pack_start,offset,padded,span,Cells{transposed,tiles,weights,bias,output}))

# ---- What a call keeps ----

def cells_pairs(state: Cells) -> Tree.Tree<Tile.Tile16>:
  Cells{pairs,tiles,weights,bias,output} = state
  pairs

def cells_tiles(state: Cells) -> Tree.Tree<Tile.Tile8>:
  Cells{pairs,tiles,weights,bias,output} = state
  tiles

def cells_weights(state: Cells) -> Tree.Tree<Tile.Tile8>:
  Cells{pairs,tiles,weights,bias,output} = state
  weights

def cells_bias(state: Cells) -> Tree.Tree<Tile.Tile8>:
  Cells{pairs,tiles,weights,bias,output} = state
  bias

def cells_result(state: Cells) -> Tree.Tree<Tile.Tile8>:
  Cells{pairs,tiles,weights,bias,output} = state
  output

def call_input(call: Call) -> Tree.Tree<Tile.Tile8>:
  Call{input,blocks} = call
  input

def call_cells(call: Call) -> Cells:
  Call{input,blocks} = call
  blocks

# What the blocks read and do not write: the planes, the weights and the
# bias.
type Frame is Data:
  Frame{pairs: Tree.Tree<Tile.Tile16>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<Tile.Tile8>}

def frame(state: Cells) -> Frame:
  Cells{pairs,tiles,weights,bias,output} = state
  Frame{pairs,weights,bias}

def step_frame(+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+start: U32,+span: U32,+state: Cells) ->
  {frame(cells_step(work,planes,pack_start,start,span,state)) == frame(state) : Frame}:
  Pipeline.WindowWork{+rows,+end,Packing.Stencil{+base,+h,+w,+kr,+kc,+s,+ro,+co,+ow,+inner,+groups}} = work
  Cells{+pairs,+tiles,+weights,+bias,+output} = state
  {==}

def blocks_frame(+remaining: Nat,+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+start: U32,+padded: U32,+span: U32,+state: Cells) ->
  {frame(cells_blocks(remaining,work,planes,pack_start,start,padded,span,state)) == frame(state) : Frame}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +next = cells_step(work,planes,pack_start,start,U32.min(span,(padded - start : U32)),state)
      Equal.trans(Frame,frame(cells_blocks(rest,work,planes,(pack_start + span : U32),(start + span : U32),padded,span,next)),frame(next),frame(state),
        blocks_frame(rest,work,planes,(pack_start + span : U32),(start + span : U32),padded,span,next),
        step_frame(work,planes,pack_start,start,U32.min(span,(padded - start : U32)),state))

def framed(+state: Cells) -> {frame(state) == Frame{cells_pairs(state),cells_weights(state),cells_bias(state)} : Frame}:
  Cells{+pairs,+tiles,+weights,+bias,+output} = state
  {==}

# A call keeps its input, gives its blocks the transposed planes, and its
# blocks keep the planes, the weights and the bias.
def into_frame(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+input: Tree.Tree<Tile.Tile8>,+state: Cells) ->
  {frame(call_cells(cells_into(work,pack_start,offset,input,state))) ==
    Frame{PackingModel.transposed(call_window(work),call_planes(work,pack_start,offset),input,cells_pairs(state)),cells_weights(state),cells_bias(state)} : Frame}:
  Cells{+pairs,+tiles,+weights,+bias,+output} = state
  +planes = call_planes(work,pack_start,offset)
  blocks_frame(call_blocks(work,offset),work,planes,pack_start,offset,padded_end(work,offset),call_span(work),
    Cells{PackingModel.transposed(call_window(work),planes,input,pairs),tiles,weights,bias,output})

# ---- The blocks on values ----

# The blocks of a call on the values of the channels of its input cells.
type State is Data:
  State{values: Tree.Tree<F32>,tiles: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<Tile.Tile8>,output: Tree.Tree<Tile.Tile8>}

# A block fills its tiles with the value vectors of its entries, read from
# zero, and stores their product.
def step_window(work: Pipeline.WindowWork,+pack_start: U32,+start: U32,+span: U32,state: State) -> State:
  Pipeline.WindowWork{+rows,+end,+window} = work
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner,groups} = window
  State{+values,tiles,+weights,+bias,output} = state
  +filled = SpatialSamples.fill(U32.to_nat((span / 8 * inner : U32)),values,SpatialSamples.plain(window),pack_start,tiles)
  State{values,filled,weights,bias,block_tree(rows,end,inner,start,span,filled,weights,bias,output)}

def blocks_window(remaining: Nat,+work: Pipeline.WindowWork,+pack_start: U32,+start: U32,+padded: U32,+span: U32,state: State) -> State:
  match remaining:
    case 0n: state
    case 1n+rest:
      blocks_window(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,step_window(work,pack_start,start,U32.min(span,(padded - start : U32)),state))

def values_into(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,state: State) -> State:
  blocks_window(call_blocks(work,offset),work,pack_start,offset,padded_end(work,offset),call_span(work),state)

def result(state: State) -> Tree.Tree<Tile.Tile8>:
  State{values,tiles,weights,bias,output} = state
  output

def workspace(state: State) -> Tree.Tree<Tile.Tile8>:
  State{values,tiles,weights,bias,output} = state
  tiles

# The state on values of the blocks on cells.
def values_of(values: Tree.Tree<F32>,state: Cells) -> State:
  Cells{pairs,tiles,weights,bias,output} = state
  State{values,tiles,weights,bias,output}

def values_result(+values: Tree.Tree<F32>,+state: Cells) -> {result(values_of(values,state)) == cells_result(state) : Tree.Tree<Tile.Tile8>}:
  Cells{+pairs,+tiles,+weights,+bias,+output} = state
  {==}

def values_workspace(+values: Tree.Tree<F32>,+state: Cells) -> {workspace(values_of(values,state)) == cells_tiles(state) : Tree.Tree<Tile.Tile8>}:
  Cells{+pairs,+tiles,+weights,+bias,+output} = state
  {==}

# ---- Cells and values agree block by block ----

# What every block of a call needs to pack from the planes: the room of a
# whole span of units, the entries of a pass, the room of the values and a
# tile depth below 31.
type Packable<-window: Packing.Stencil,-room_units: Nat,-depth: Nat,-cap: U32> is Data:
  Packable{room: Passes.PackRoom<window,room_units,depth>,
    entries_exact: {PackingModel.pass_entries(window) == Nat.mul(Planes.channel_count(window),Passes.tap_rows_of(window)) : Nat},
    values_room: Pixels.ValuesRoom<window,cap>,limit: {Nat.is_le(depth,30n) == True{} : Bool}}

# Fewer units fit where more do.
def room_fewer(+window: Packing.Stencil,+units: Nat,+fewer: Nat,+depth: Nat,+room: Passes.PackRoom<window,units,depth>,
  +ordered: {Nat.is_le(fewer,units) == True{} : Bool}) -> Passes.PackRoom<window,fewer,depth>:
  Passes.PackRoom{+cap,+cap_ok,+limit,+capped,+taps_bound,+taps_positive,+columns_positive,+entries_positive,+inner_exact,+fits} = room
  +columns = Nat.mul(PackingModel.pass_entries(window),Passes.tap_columns_of(window))
  Passes.PackRoom{cap,cap_ok,limit,capped,taps_bound,taps_positive,columns_positive,entries_positive,inner_exact,
    Order.transitive(Nat.mul(fewer,columns),Nat.mul(units,columns),U32.to_nat(U32.shln(1,depth)),RowMajor.mul_le(fewer,units,columns,ordered),fits)}

def block_cap(+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+units: Nat,+depth: Nat,+cap: U32,+block: Pixels.Block<window,planes,start,units,depth,cap>) ->
  {U32.is_le(cap,2147483648) == True{} : Bool}:
  Pixels.Block{+geometry,sixteen,positive,width_bound,lanes_room,stride_positive,stride_bound,ordered,row_exact,width_fits,columns_room,values_room,below_first,covered,
    positions_room} = block
  Pixels.geometry_cap(window,planes,depth,cap,geometry)

def block_sixteen(+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+units: Nat,+depth: Nat,+cap: U32,+block: Pixels.Block<window,planes,start,units,depth,cap>) ->
  {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool}:
  Pixels.Block{geometry,+sixteen,positive,width_bound,lanes_room,stride_positive,stride_bound,ordered,row_exact,width_fits,columns_room,values_room,below_first,covered,
    positions_room} = block
  sixteen

# A block inside the block of a call keeps its facts.
def block_within(+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+units: Nat,+depth: Nat,+cap: U32,+block: Pixels.Block<window,planes,start,units,depth,cap>,
  +inner_start: U32,+inner_units: Nat,+after: {Nat.is_le(U32.to_nat(start),U32.to_nat(inner_start)) == True{} : Bool},
  +inside: {Nat.is_le(Pixels.block_end(inner_start,inner_units),Pixels.block_end(start,units)) == True{} : Bool}) -> Pixels.Block<window,planes,inner_start,inner_units,depth,cap>:
  Pixels.Block{+geometry,+sixteen,+positive,+width_bound,+lanes_room,+stride_positive,+stride_bound,+ordered,+row_exact,+width_fits,+columns_room,+values_room,+below_first,
    +covered,+positions_room} = block
  +width = Pixels.output_width_of(window)
  +first = Pixels.first_row_of(planes)
  +stride = Pixels.stride_of(window)
  +taps = Passes.tap_rows_of(window)
  +tiles = Planes.tiles_of(planes)
  +inner_row = Nat.div(Pixels.block_last(inner_start,inner_units),width)
  +outer_row = Nat.div(Pixels.block_last(start,units),width)
  +rows_le = DivisionOrder.quotient_le(Pixels.block_last(inner_start,inner_units),Pixels.block_last(start,units),width,positive,
    DivisionOrder.subtract_le(Pixels.block_end(inner_start,inner_units),Pixels.block_end(start,units),1n,inside))
  +spans_le = RowMajor.mul_le(Nat.sub(inner_row,first),Nat.sub(outer_row,first),stride,DivisionOrder.subtract_le(inner_row,outer_row,first,rows_le))
  +reach_le = Order.add_le(Nat.mul(Nat.sub(inner_row,first),stride),Nat.mul(Nat.sub(outer_row,first),stride),taps,spans_le)
  Pixels.Block{geometry,sixteen,positive,width_bound,lanes_room,stride_positive,stride_bound,ordered,row_exact,width_fits,columns_room,values_room,
    Order.transitive(first,Nat.div(U32.to_nat(start),width),Nat.div(U32.to_nat(inner_start),width),below_first,
      DivisionOrder.quotient_le(U32.to_nat(start),U32.to_nat(inner_start),width,positive,after)),
    Order.transitive(Nat.mul(Nat.add(Nat.mul(Nat.sub(inner_row,first),stride),taps),tiles),Nat.mul(Nat.add(Nat.mul(Nat.sub(outer_row,first),stride),taps),tiles),
      Planes.elements_of(planes),RowMajor.mul_le(Nat.add(Nat.mul(Nat.sub(inner_row,first),stride),taps),Nat.add(Nat.mul(Nat.sub(outer_row,first),stride),taps),tiles,reach_le),
      covered),
    Order.transitive(Pixels.block_end(inner_start,inner_units),Pixels.block_end(start,units),U32.to_nat(cap),inside,positions_room)}

# Where a block stands in the blocks of its call: the blocks left reach the
# padded end, the pack position lies at or after the call's, the positions
# left fit in the block of the call, and both ends fit the capacity.
type Track<-call_start: U32,-units: Nat,-pack_start: U32,-start: U32,-padded: U32,-span: U32,-remaining: Nat,-cap: U32> is Data:
  Track{last: BlockBounds.Last(remaining,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span)),
    after: {Nat.is_le(U32.to_nat(call_start),U32.to_nat(pack_start)) == True{} : Bool},
    inside: {Nat.is_le(Nat.add(U32.to_nat(pack_start),U32.to_nat(padded)),Nat.add(U32.to_nat(start),Pixels.block_end(call_start,units))) == True{} : Bool},
    padded_room: {Nat.is_le(U32.to_nat(padded),U32.to_nat(cap)) == True{} : Bool},
    ending: {Nat.is_le(Pixels.block_end(call_start,units),U32.to_nat(cap)) == True{} : Bool}}

def swap_last(+first: Nat,+second: Nat,+third: Nat) -> {Nat.add(Nat.add(first,second),third) == Nat.add(Nat.add(first,third),second) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.add(first,second),third),Nat.add(first,Nat.add(second,third)),Addition.associative(first,second,third)) :
    {_ == Nat.add(Nat.add(first,third),second) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(first,third),second),Nat.add(first,Nat.add(third,second)),Addition.associative(first,third,second)) :
    {Nat.add(first,Nat.add(second,third)) == _ : Nat}
  %Order.commutative(third,second) : {Nat.add(first,_) == Nat.add(first,Nat.add(third,second)) : Nat}
  {==}

# Moving the start of a sum past the pack position: b + (p + (e - b)) is p + e.
def moved_sum(+start: Nat,+pack: Nat,+end: Nat,+ordered: {Nat.is_le(start,end) == True{} : Bool}) ->
  {Nat.add(start,Nat.add(pack,Nat.sub(end,start))) == Nat.add(pack,end) : Nat}:
  +rest = Nat.sub(end,start)
  Equal.trans(Nat,Nat.add(start,Nat.add(pack,rest)),Nat.add(pack,Nat.add(start,rest)),Nat.add(pack,end),
    Equal.trans(Nat,Nat.add(start,Nat.add(pack,rest)),Nat.add(Nat.add(start,pack),rest),Nat.add(pack,Nat.add(start,rest)),
      Equal.sym(Nat,Nat.add(Nat.add(start,pack),rest),Nat.add(start,Nat.add(pack,rest)),Addition.associative(start,pack,rest)),{
        %Order.commutative(pack,start) : {Nat.add(_,rest) == Nat.add(pack,Nat.add(start,rest)) : Nat}
        Addition.associative(pack,start,rest) : {Nat.add(Nat.add(start,pack),rest) == Nat.add(pack,Nat.add(start,rest)) : Nat}}),
    Equal.cong(Nat,Nat,value => Nat.add(pack,value),Nat.add(start,rest),end,Order.restore_shift(start,end,ordered)))

# The positions left of a block fit in the block of the call.
def left_inside(+pack: Nat,+start: Nat,+end: Nat,+limit: Nat,+ordered: {Nat.is_le(start,end) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(pack,end),Nat.add(start,limit)) == True{} : Bool}) -> {Nat.is_le(Nat.add(pack,Nat.sub(end,start)),limit) == True{} : Bool}:
  Order.cancel_offset_le(start,Nat.add(pack,Nat.sub(end,start)),limit,{
    %Equal.sym(Nat,Nat.add(start,Nat.add(pack,Nat.sub(end,start))),Nat.add(pack,end),moved_sum(start,pack,end,ordered)) :
      {Nat.is_le(_,Nat.add(start,limit)) == True{} : Bool}
    inside : {Nat.is_le(Nat.add(start,Nat.add(pack,Nat.sub(end,start))),Nat.add(start,limit)) == True{} : Bool}})

# The next block of a call stands one span further.
def track_next(+call_start: U32,+units: Nat,+pack_start: U32,+start: U32,+padded: U32,+span: U32,+more: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+track: Track<call_start,units,pack_start,start,padded,span,1n+1n+more,cap>) ->
  Track<call_start,units,(pack_start + span : U32),(start + span : U32),padded,span,1n+more,cap>:
  Track{+last,+after,+inside,+padded_room,+ending} = track
  +p = U32.to_nat(pack_start)
  +s = U32.to_nat(start)
  +e = U32.to_nat(padded)
  +n = U32.to_nat(span)
  +limit = Pixels.block_end(call_start,units)
  +ahead = Order.le_lt_trans(Nat.add(s,n),Nat.add(s,Nat.add(n,Nat.mul(more,n))),e,Order.offset_le(s,n,Nat.add(n,Nat.mul(more,n)),Order.le_add(n,Nat.mul(more,n))),last)
  +reach = Bounds.strict_le(Nat.add(s,n),e,ahead)
  +ordered = Order.transitive(s,Nat.add(s,n),e,Order.le_add(s,n),reach)
  +room = Order.cancel_offset_le(s,n,Nat.sub(e,s),{
    %Equal.sym(Nat,Nat.add(s,Nat.sub(e,s)),e,Order.restore_shift(s,e,ordered)) : {Nat.is_le(Nat.add(s,n),_) == True{} : Bool}
    reach : {Nat.is_le(Nat.add(s,n),Nat.add(s,Nat.sub(e,s))) == True{} : Bool}})
  +start_exact = Arithmetic.add(start,span,cap,cap_ok,Order.transitive(Nat.add(s,n),e,U32.to_nat(cap),reach,padded_room))
  +left = left_inside(p,s,e,limit,ordered,inside)
  +pack_exact = Arithmetic.add(pack_start,span,cap,cap_ok,
    Order.transitive(Nat.add(p,n),Nat.add(p,Nat.sub(e,s)),U32.to_nat(cap),Order.offset_le(p,n,Nat.sub(e,s),room),
      Order.transitive(Nat.add(p,Nat.sub(e,s)),limit,U32.to_nat(cap),left,ending)))
  +next_last = {
    %Equal.sym(Nat,U32.to_nat((start + span : U32)),Nat.add(s,n),start_exact) : BlockBounds.Last(1n+more,_,e,n)
    BlockBounds.last_advanced(1n+more,s,e,n,last) : BlockBounds.Last(1n+more,U32.to_nat((start + span : U32)),e,n)}
  +next_after = {
    %Equal.sym(Nat,U32.to_nat((pack_start + span : U32)),Nat.add(p,n),pack_exact) : {Nat.is_le(U32.to_nat(call_start),_) == True{} : Bool}
    Order.transitive(U32.to_nat(call_start),p,Nat.add(p,n),after,Order.le_add(p,n)) : {Nat.is_le(U32.to_nat(call_start),U32.to_nat((pack_start + span : U32))) == True{} : Bool}}
  +next_inside = {
    %Equal.sym(Nat,U32.to_nat((pack_start + span : U32)),Nat.add(p,n),pack_exact) : {Nat.is_le(Nat.add(_,e),Nat.add(U32.to_nat((start + span : U32)),limit)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat((start + span : U32)),Nat.add(s,n),start_exact) : {Nat.is_le(Nat.add(Nat.add(p,n),e),Nat.add(_,limit)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.add(p,n),e),Nat.add(Nat.add(p,e),n),swap_last(p,n,e)) : {Nat.is_le(_,Nat.add(Nat.add(s,n),limit)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.add(s,n),limit),Nat.add(Nat.add(s,limit),n),swap_last(s,n,limit)) : {Nat.is_le(Nat.add(Nat.add(p,e),n),_) == True{} : Bool}
    Order.add_le(Nat.add(p,e),Nat.add(s,limit),n,inside) :
      {Nat.is_le(Nat.add(U32.to_nat((pack_start + span : U32)),e),Nat.add(U32.to_nat((start + span : U32)),limit)) == True{} : Bool}}
  Track{next_last,next_after,next_inside,padded_room,ending}

# The facts of one block: its units, their room, the units of its count, the
# count itself and the block inside the block of the call.
def clip_of(+start: U32,+padded: U32,+span: U32) -> U32:
  U32.min(span,(padded - start : U32))

def count_of(+window: Packing.Stencil,+span: U32) -> Nat:
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner,groups} = window
  U32.to_nat((span / 8 * inner : U32))

type Step<-window: Packing.Stencil,-planes: Packing.Planes,-pack_start: U32,-clip: U32,-depth: Nat,-pair_depth: Nat,-cap: U32> is Data:
  Step{units: Nat,room: Passes.PackRoom<window,units,depth>,units_exact: {PackingModel.unit_count(window,count_of(window,clip)) == units : Nat},
    count_exact: {Nat.mul(units,SpatialSamples.inner_count(window)) == count_of(window,clip) : Nat},block: Pixels.Block<window,planes,pack_start,units,pair_depth,cap>}

def step_facts(+window: Packing.Stencil,+planes: Packing.Planes,+pack_start: U32,+start: U32,+padded: U32,+span: U32,+rest: Nat,+depth: Nat,+pair_depth: Nat,
  +cap: U32,+call_start: U32,+units: Nat,+packable: Packable<window,Nat.min(Nat.div(U32.to_nat(span),8n),units),depth,cap>,+block: Pixels.Block<window,planes,call_start,units,pair_depth,cap>,
  +track: Track<call_start,units,pack_start,start,padded,span,1n+rest,cap>) -> Step<window,planes,pack_start,clip_of(start,padded,span),depth,pair_depth,cap>:
  Packing.Stencil{+base,+h,+w,+kr,+kc,+s,+ro,+co,+ow,+inner,+groups} = window
  Packable{+room,entries_exact,values_room,limit} = packable
  Passes.PackRoom{+room_cap,+room_cap_ok,room_limit,+capped,taps_bound,taps_positive,+columns_positive,+entries_positive,+inner_exact,+fits} = room
  Track{+last,+after,+inside,+padded_room,+ending} = track
  +window = {Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner,groups} : Packing.Stencil}
  +p = U32.to_nat(pack_start)
  +b = U32.to_nat(start)
  +e = U32.to_nat(padded)
  +n = U32.to_nat(span)
  +limit = Pixels.block_end(call_start,units)
  +cap_ok = block_cap(window,planes,call_start,units,pair_depth,cap,block)
  +below = Order.le_lt_trans(b,Nat.add(b,Nat.mul(rest,n)),e,Order.le_add(b,Nat.mul(rest,n)),last)
  +ordered = Bounds.strict_le(b,e,below)
  +difference = Arithmetic.subtract(padded,start,cap,cap_ok,ordered,Order.transitive(Nat.sub(e,b),e,U32.to_nat(cap),Order.sub_le(e,b),padded_room))
  +clip = clip_of(start,padded,span)
  +clip_exact = BlockBounds.clipped_exact(start,padded,span,difference)
  +c = Nat.min(n,Nat.sub(e,b))
  +count = Nat.div(c,8n)
  +columns = Nat.mul(PackingModel.pass_entries(window),Passes.tap_columns_of(window))
  +fewer = DivisionOrder.quotient_le(c,n,8n,{==},Extrema.minimum_left(n,Nat.sub(e,b)))
  +clip_bound = {
    %Equal.sym(Nat,U32.to_nat(clip),c,clip_exact) : {Nat.is_le(_,Nat.sub(e,b)) == True{} : Bool}
    Extrema.minimum_right(n,Nat.sub(e,b)) : {Nat.is_le(U32.to_nat(clip),Nat.sub(e,b)) == True{} : Bool}}
  +below_clip = {
    %Equal.sym(Nat,U32.to_nat(clip),c,clip_exact) : {Nat.is_le(Nat.mul(count,8n),_) == True{} : Bool}
    DivisionOrder.product_below(c,8n,{==}) : {Nat.is_le(Nat.mul(count,8n),U32.to_nat(clip)) == True{} : Bool}}
  +units_room = Order.transitive(Nat.mul(count,8n),U32.to_nat(clip),Nat.sub(e,b),below_clip,clip_bound)
  +left = left_inside(p,b,e,limit,ordered,inside)
  +inner_block = Order.transitive(Pixels.block_end(pack_start,count),Nat.add(p,Nat.sub(e,b)),limit,Order.offset_le(p,Nat.mul(count,8n),Nat.sub(e,b),units_room),left)
  +call_units = U32.to_nat(call_start)
  +within = RowMajor.cancel_multiply_le(count,units,8n,{==},
    Order.cancel_offset_le(call_units,Nat.mul(count,8n),Nat.mul(units,8n),
      Order.transitive(Nat.add(call_units,Nat.mul(count,8n)),Pixels.block_end(pack_start,count),limit,Order.add_le(call_units,p,Nat.mul(count,8n),after),inner_block)))
  +bounded = Extrema.minimum_lower(Nat.div(n,8n),units,count,fewer,within)
  +eighth = WordDivision.divide(clip,8,cap,cap_ok,block_sixteen(window,planes,call_start,units,pair_depth,cap,block),{==})
  +eighth_value = Equal.trans(Nat,U32.to_nat((clip / 8 : U32)),Nat.div(U32.to_nat(clip),8n),count,eighth,
    Equal.cong(Nat,Nat,value => Nat.div(value,8n),U32.to_nat(clip),c,clip_exact))
  +inner_room = {
    %Equal.sym(Nat,U32.to_nat(inner),columns,inner_exact) : {Nat.is_le(Nat.mul(count,_),Nat.mul(count,columns)) == True{} : Bool}
    Order.reflexive(Nat.mul(count,columns)) : {Nat.is_le(Nat.mul(count,U32.to_nat(inner)),Nat.mul(count,columns)) == True{} : Bool}}
  +product_room = Order.transitive(Nat.mul(count,U32.to_nat(inner)),Nat.mul(count,columns),U32.to_nat(room_cap),inner_room,
    Order.transitive(Nat.mul(count,columns),U32.to_nat(U32.shln(1,depth)),U32.to_nat(room_cap),
      Order.transitive(Nat.mul(count,columns),Nat.mul(Nat.min(Nat.div(n,8n),units),columns),U32.to_nat(U32.shln(1,depth)),RowMajor.mul_le(count,Nat.min(Nat.div(n,8n),units),columns,bounded),fits),capped))
  +eighth_room = {
    %Equal.sym(Nat,U32.to_nat((clip / 8 : U32)),count,eighth_value) : {Nat.is_le(Nat.mul(_,U32.to_nat(inner)),U32.to_nat(room_cap)) == True{} : Bool}
    product_room : {Nat.is_le(Nat.mul(U32.to_nat((clip / 8 : U32)),U32.to_nat(inner)),U32.to_nat(room_cap)) == True{} : Bool}}
  +product_exact = Equal.trans(Nat,U32.to_nat((clip / 8 * inner : U32)),Nat.mul(U32.to_nat((clip / 8 : U32)),U32.to_nat(inner)),Nat.mul(count,U32.to_nat(inner)),
    Multiplication.bounded_multiply((clip / 8 : U32),inner,room_cap,room_cap_ok,eighth_room),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(inner)),U32.to_nat((clip / 8 : U32)),count,eighth_value))
  +inner_positive = {
    %Equal.sym(Nat,U32.to_nat(inner),columns,inner_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(PackingModel.pass_entries(window),Passes.tap_columns_of(window),entries_positive,columns_positive) :
      {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool}}
  +count_exact = Equal.sym(Nat,U32.to_nat((clip / 8 * inner : U32)),Nat.mul(count,U32.to_nat(inner)),product_exact)
  +units_exact = {
    %count_exact : {Nat.div(_,U32.to_nat(inner)) == count : Nat}
    Division.exact_quotient(count,U32.to_nat(inner),inner_positive) : {Nat.div(U32.to_nat((clip / 8 * inner : U32)),U32.to_nat(inner)) == count : Nat}}
  Step{count,room_fewer(window,Nat.min(Nat.div(U32.to_nat(span),8n),units),count,depth,room,bounded),units_exact,count_exact,
    block_within(window,planes,call_start,units,pair_depth,cap,block,pack_start,count,after,inner_block)}

# The pack of a block keeps the tiles balanced.
def step_balanced(+window: Packing.Stencil,+planes: Packing.Planes,+pack_start: U32,+clip: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+depth: Nat,
  +pair_depth: Nat,+cap: U32,+facts: Step<window,planes,pack_start,clip,depth,pair_depth,cap>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles)) ->
  Shape.Balanced(Tile.Tile8,depth,PackingModel.packed_tiles(window,planes,count_of(window,clip),pack_start,pairs,tiles)):
  Step{+units,+room,+units_exact,count_exact,block} = facts
  Passes.pack_balanced(window,planes,count_of(window,clip),pack_start,pairs,tiles,units,depth,room,units_exact,balanced)

# A block on cells after the transpose is the block on the values of the
# input cells.
def step_values(+rows: U32,+end: U32,+window: Packing.Stencil,+planes: Packing.Planes,+pack_start: U32,+start: U32,+padded: U32,+span: U32,+input: Tree.Tree<Tile.Tile8>,
  +pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,
  +pair_depth: Nat,+cap: U32,+room_units: Nat,+packable: Packable<window,room_units,depth,cap>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),+facts: Step<window,planes,pack_start,clip_of(start,padded,span),depth,pair_depth,cap>) ->
  {values_of(Pixels.window_values(window,input),cells_step(Pipeline.WindowWork{rows,end,window},planes,pack_start,start,clip_of(start,padded,span),
      Cells{PackingModel.transposed(window,planes,input,pairs),tiles,weights,bias,output})) ==
    step_window(Pipeline.WindowWork{rows,end,window},pack_start,start,clip_of(start,padded,span),State{Pixels.window_values(window,input),tiles,weights,bias,output}) : State}:
  Packing.Stencil{+base,+h,+w,+kr,+kc,+s,+ro,+co,+ow,+inner,+groups} = window
  Packable{room,+entries_exact,+values_room,+limit} = packable
  Step{+units,+units_room,+units_exact,+count_exact,+block} = facts
  +window = {Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner,groups} : Packing.Stencil}
  +clip = clip_of(start,padded,span)
  +count = U32.to_nat((clip / 8 * inner : U32))
  +values = Pixels.window_values(window,input)
  %count_exact :
    {State{values,Pixels.packed_of(window,planes,count,pack_start,input,pairs,tiles),weights,bias,
        block_tree(rows,end,inner,start,clip,Pixels.packed_of(window,planes,count,pack_start,input,pairs,tiles),weights,bias,output)} ==
      State{values,SpatialSamples.fill(_,values,SpatialSamples.plain(window),pack_start,tiles),weights,bias,
        block_tree(rows,end,inner,start,clip,SpatialSamples.fill(_,values,SpatialSamples.plain(window),pack_start,tiles),weights,bias,output)} : State}
  +entries = Nat.mul(units,U32.to_nat(inner))
  Equal.cong(Tree.Tree<Tile.Tile8>,State,packed => State{values,packed,weights,bias,block_tree(rows,end,inner,start,clip,packed,weights,bias,output)},
    Pixels.packed_of(window,planes,count,pack_start,input,pairs,tiles),SpatialSamples.fill(entries,values,SpatialSamples.plain(window),pack_start,tiles),
    Equal.trans(Tree.Tree<Tile.Tile8>,Pixels.packed_of(window,planes,count,pack_start,input,pairs,tiles),Pixels.filled_of(window,units,pack_start,input,tiles),
      SpatialSamples.fill(entries,values,SpatialSamples.plain(window),pack_start,tiles),
      Pixels.pack_fill(input,pairs,tiles,window,planes,count,pack_start,units,depth,pair_depth,cap,units_room,units_exact,balanced,pairs_balanced,entries_exact,values_room,
        block,limit),
      SpatialSamples.fill_plain(entries,values,window,pack_start,tiles)))

# The blocks of a call on cells after the transpose are the blocks on the
# values of the input cells.
def blocks_values(remaining: Nat,+rows: U32,+end: U32,+window: Packing.Stencil,+planes: Packing.Planes,+pack_start: U32,+start: U32,+padded: U32,+span: U32,
  +input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,
  +output: Tree.Tree<Tile.Tile8>,+depth: Nat,+pair_depth: Nat,+cap: U32,+call_start: U32,+units: Nat,+packable: Packable<window,Nat.min(Nat.div(U32.to_nat(span),8n),units),depth,cap>,
  +block: Pixels.Block<window,planes,call_start,units,pair_depth,cap>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),+track: Track<call_start,units,pack_start,start,padded,span,remaining,cap>) ->
  {values_of(Pixels.window_values(window,input),cells_blocks(remaining,Pipeline.WindowWork{rows,end,window},planes,pack_start,start,padded,span,
      Cells{PackingModel.transposed(window,planes,input,pairs),tiles,weights,bias,output})) ==
    blocks_window(remaining,Pipeline.WindowWork{rows,end,window},pack_start,start,padded,span,State{Pixels.window_values(window,input),tiles,weights,bias,output}) : State}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      match rest:
        case 0n:
          step_values(rows,end,window,planes,pack_start,start,padded,span,input,pairs,tiles,weights,bias,output,depth,pair_depth,cap,Nat.min(Nat.div(U32.to_nat(span),8n),units),packable,balanced,pairs_balanced,
            step_facts(window,planes,pack_start,start,padded,span,0n,depth,pair_depth,cap,call_start,units,packable,block,track))
        case 1n+ +more:
          Packing.Stencil{+base,+h,+w,+kr,+kc,+s,+ro,+co,+ow,+inner,+groups} = window
          +window = {Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner,groups} : Packing.Stencil}
          +work = {Pipeline.WindowWork{rows,end,window} : Pipeline.WindowWork}
          +clip = clip_of(start,padded,span)
          +facts = step_facts(window,planes,pack_start,start,padded,span,1n+more,depth,pair_depth,cap,call_start,units,packable,block,track)
          +values = Pixels.window_values(window,input)
          +transposed = PackingModel.transposed(window,planes,input,pairs)
          +packed = PackingModel.packed_tiles(window,planes,U32.to_nat((clip / 8 * inner : U32)),pack_start,transposed,tiles)
          +next = {Cells{transposed,packed,weights,bias,block_tree(rows,end,inner,start,clip,packed,weights,bias,output)} : Cells}
          %step_values(rows,end,window,planes,pack_start,start,padded,span,input,pairs,tiles,weights,bias,output,depth,pair_depth,cap,Nat.min(Nat.div(U32.to_nat(span),8n),units),packable,balanced,pairs_balanced,facts) :
            {values_of(values,cells_blocks(1n+more,work,planes,(pack_start + span : U32),(start + span : U32),padded,span,next)) ==
              blocks_window(1n+more,work,(pack_start + span : U32),(start + span : U32),padded,span,_) : State}
          blocks_values(1n+more,rows,end,window,planes,(pack_start + span : U32),(start + span : U32),padded,span,input,pairs,packed,weights,bias,
            block_tree(rows,end,inner,start,clip,packed,weights,bias,output),depth,pair_depth,cap,call_start,units,packable,block,
            step_balanced(window,planes,pack_start,clip,transposed,tiles,depth,pair_depth,cap,facts,balanced),pairs_balanced,
            track_next(call_start,units,pack_start,start,padded,span,more,cap,block_cap(window,planes,call_start,units,pair_depth,cap,block),track))

# The blocks of a call on cells are the blocks on the values of its input
# cells.
def into_values(+rows: U32,+end: U32,+window: Packing.Stencil,+pack_start: U32,+offset: U32,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,
  +tiles: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,+pair_depth: Nat,+cap: U32,
  +units: Nat,+packable: Packable<window,Nat.min(Nat.div(U32.to_nat(call_span(Pipeline.WindowWork{rows,end,window})),8n),units),depth,cap>,
  +block: Pixels.Block<window,call_planes(Pipeline.WindowWork{rows,end,window},pack_start,offset),pack_start,units,pair_depth,cap>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),
  +track: Track<pack_start,units,pack_start,offset,padded_end(Pipeline.WindowWork{rows,end,window},offset),call_span(Pipeline.WindowWork{rows,end,window}),
    call_blocks(Pipeline.WindowWork{rows,end,window},offset),cap>) ->
  {values_of(Pixels.window_values(window,input),call_cells(cells_into(Pipeline.WindowWork{rows,end,window},pack_start,offset,input,Cells{pairs,tiles,weights,bias,output}))) ==
    values_into(Pipeline.WindowWork{rows,end,window},pack_start,offset,State{Pixels.window_values(window,input),tiles,weights,bias,output}) : State}:
  +work = {Pipeline.WindowWork{rows,end,window} : Pipeline.WindowWork}
  blocks_values(call_blocks(work,offset),rows,end,window,call_planes(work,pack_start,offset),pack_start,offset,padded_end(work,offset),call_span(work),input,pairs,tiles,
    weights,bias,output,depth,pair_depth,cap,pack_start,units,packable,block,balanced,pairs_balanced,track)

# ---- A call ready to run ----

# The blocks on values read the values from zero, whatever the base of the
# window.
def blocks_plain(remaining: Nat,+rows: U32,+end: U32,+window: Packing.Stencil,+pack_start: U32,+start: U32,+padded: U32,+span: U32,+state: State) ->
  {blocks_window(remaining,Pipeline.WindowWork{rows,end,window},pack_start,start,padded,span,state) ==
    blocks_window(remaining,Pipeline.WindowWork{rows,end,SpatialSamples.plain(window)},pack_start,start,padded,span,state) : State}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
      State{+values,+tiles,+weights,+bias,+output} = state
      blocks_plain(rest,rows,end,window,(pack_start + span : U32),(start + span : U32),padded,span,
        step_window(Pipeline.WindowWork{rows,end,window},pack_start,start,U32.min(span,(padded - start : U32)),state))

def values_plain(+rows: U32,+end: U32,+window: Packing.Stencil,+pack_start: U32,+offset: U32,+state: State) ->
  {values_into(Pipeline.WindowWork{rows,end,window},pack_start,offset,state) == values_into(Pipeline.WindowWork{rows,end,SpatialSamples.plain(window)},pack_start,offset,state) :
    State}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +work = {Pipeline.WindowWork{rows,end,window} : Pipeline.WindowWork}
  blocks_plain(call_blocks(work,offset),rows,end,window,pack_start,offset,padded_end(work,offset),call_span(work),state)

def add_cancel(+left: U32,+right: U32) -> {((left + right : U32) - left : U32) == right : U32}:
  %Arithmetic.add_commutative(right,left) : {(_ - left : U32) == right : U32}
  Subtraction.subtract_add(right,left)

# A call of count positions from offset covers their padded count.
def call_count(+offset: U32,+count: U32) -> {((offset + Window.padded(((offset + count : U32) - offset : U32)) : U32) - offset : U32) == Window.padded(count) : U32}:
  %Equal.sym(U32,((offset + count : U32) - offset : U32),count,add_cancel(offset,count)) :
    {((offset + Window.padded(_) : U32) - offset : U32) == Window.padded(count) : U32}
  add_cancel(offset,Window.padded(count))

# A schedule from zero, moved to start.
def last_shifted(remaining: Nat,+start: Nat,+end: Nat,+span: Nat,+last: BlockBounds.Last(remaining,0n,end,span)) -> BlockBounds.Last(remaining,start,Nat.add(start,end),span):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest: Order.offset_lt(start,Nat.mul(rest,span),end,last)

def exchange(+first: Nat,+second: Nat,+third: Nat) -> {Nat.add(first,Nat.add(second,third)) == Nat.add(second,Nat.add(first,third)) : Nat}:
  Equal.trans(Nat,Nat.add(first,Nat.add(second,third)),Nat.add(Nat.add(first,second),third),Nat.add(second,Nat.add(first,third)),
    Equal.sym(Nat,Nat.add(Nat.add(first,second),third),Nat.add(first,Nat.add(second,third)),Addition.associative(first,second,third)),
    %Order.commutative(second,first) : {Nat.add(_,third) == Nat.add(second,Nat.add(first,third)) : Nat}
    Addition.associative(second,first,third))

# What a call needs to agree with its blocks on values.
type CallReady<-work: Pipeline.WindowWork,-pack_start: U32,-offset: U32,-pairs: Tree.Tree<Tile.Tile16>,-tiles: Tree.Tree<Tile.Tile8>,-depth: Nat,-pair_depth: Nat,-units: Nat,
  -cap: U32> is Data:
  CallReady{packable: Packable<call_window(work),Nat.min(Nat.div(U32.to_nat(call_span(work)),8n),units),depth,cap>,
    block: Pixels.Block<call_window(work),call_planes(work,pack_start,offset),pack_start,units,pair_depth,cap>,
    balanced: Shape.Balanced(Tile.Tile8,depth,tiles),pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),
    track: Track<pack_start,units,pack_start,offset,padded_end(work,offset),call_span(work),call_blocks(work,offset),cap>}

# A call of count positions from offset is ready when its planes have room,
# its blocks are scheduled from zero and its workspaces are balanced.
def ready(+rows: U32,+count: U32,+window: Packing.Stencil,+pack_start: U32,+offset: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+depth: Nat,
  +pair_depth: Nat,+units: Nat,+cap: U32,+packable: Packable<window,Nat.min(Nat.div(U32.to_nat(Pipeline.block_positions(Passes.inner_of(window))),8n),units),depth,cap>,
  +room: Call.CallRoom<window,pack_start,Window.padded(count),pair_depth,cap>,+units_exact: {U32.to_nat(Window.padded(count)) == Nat.mul(units,8n) : Nat},
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),
  +last: BlockBounds.Last(U32.to_nat(((Window.padded(count) + Pipeline.block_positions(Passes.inner_of(window)) - 1) / Pipeline.block_positions(Passes.inner_of(window)) : U32)),0n,
    U32.to_nat(Window.padded(count)),U32.to_nat(Pipeline.block_positions(Passes.inner_of(window)))),
  +offset_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(Window.padded(count))),U32.to_nat(cap)) == True{} : Bool}) ->
  CallReady<Pipeline.WindowWork{rows,(offset + count : U32),window},pack_start,offset,pairs,tiles,depth,pair_depth,units,cap>:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Call.CallRoom{+cap_ok,+sixteen,+positive,+width_bound,+stride_positive,+stride_bound,+taps_positive,+columns_positive,+lanes_room,+columns_room,+tiles_room,+channels_positive,+divisor_bound,
    +planes_bound,+count_positive,+positions_room,+area_room,+limit,+capped,+fits} = room
  +work = {Pipeline.WindowWork{rows,(offset + count : U32),window} : Pipeline.WindowWork}
  +padded = Window.padded(count)
  +span = Pipeline.block_positions(inner)
  +o = U32.to_nat(offset)
  +p = U32.to_nat(pack_start)
  +n = U32.to_nat(padded)
  +counted = call_count(offset,count)
  +end_word = {
    %Equal.sym(U32,((offset + count : U32) - offset : U32),count,add_cancel(offset,count)) : {(offset + Window.padded(_) : U32) == (offset + padded : U32) : U32}
    {==} : {padded_end(work,offset) == (offset + padded : U32) : U32}}
  +ended = Equal.trans(Nat,U32.to_nat(padded_end(work,offset)),U32.to_nat((offset + padded : U32)),Nat.add(o,n),
    Equal.cong(U32,Nat,word => U32.to_nat(word),padded_end(work,offset),(offset + padded : U32),end_word),
    Arithmetic.sum_exact(cap,cap_ok,offset,padded,o,n,{==},{==},offset_room))
  +block = {
    %Equal.sym(U32,(padded_end(work,offset) - offset : U32),padded,counted) : Pixels.Block<window,Packing.planes_of(window,pack_start,_),pack_start,units,pair_depth,cap>
    Call.call_block(window,pack_start,padded,units,pair_depth,cap,room,units_exact) : Pixels.Block<window,call_planes(work,pack_start,offset),pack_start,units,pair_depth,cap>}
  +moved = {
    %Equal.sym(U32,(padded_end(work,offset) - offset : U32),padded,counted) :
      BlockBounds.Last(U32.to_nat(((_ + span - 1) / span : U32)),o,U32.to_nat(padded_end(work,offset)),U32.to_nat(span))
    %Equal.sym(Nat,U32.to_nat(padded_end(work,offset)),Nat.add(o,n),ended) : BlockBounds.Last(U32.to_nat(((padded + span - 1) / span : U32)),o,_,U32.to_nat(span))
    last_shifted(U32.to_nat(((padded + span - 1) / span : U32)),o,n,U32.to_nat(span),last) :
      BlockBounds.Last(call_blocks(work,offset),o,U32.to_nat(padded_end(work,offset)),U32.to_nat(span))}
  +inside = {
    %Equal.sym(Nat,U32.to_nat(padded_end(work,offset)),Nat.add(o,n),ended) : {Nat.is_le(Nat.add(p,_),Nat.add(o,Pixels.block_end(pack_start,units))) == True{} : Bool}
    %units_exact : {Nat.is_le(Nat.add(p,Nat.add(o,n)),Nat.add(o,Nat.add(p,_))) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(o,Nat.add(p,n)),Nat.add(p,Nat.add(o,n)),exchange(o,p,n)) : {Nat.is_le(Nat.add(p,Nat.add(o,n)),_) == True{} : Bool}
    Order.reflexive(Nat.add(p,Nat.add(o,n))) :
      {Nat.is_le(Nat.add(p,U32.to_nat(padded_end(work,offset))),Nat.add(o,Pixels.block_end(pack_start,units))) == True{} : Bool}}
  +padded_room = {
    %Equal.sym(Nat,U32.to_nat(padded_end(work,offset)),Nat.add(o,n),ended) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    offset_room : {Nat.is_le(U32.to_nat(padded_end(work,offset)),U32.to_nat(cap)) == True{} : Bool}}
  +ending = {
    %units_exact : {Nat.is_le(Nat.add(p,_),U32.to_nat(cap)) == True{} : Bool}
    positions_room : {Nat.is_le(Pixels.block_end(pack_start,units),U32.to_nat(cap)) == True{} : Bool}}
  CallReady{packable,block,balanced,pairs_balanced,Track{moved,Order.reflexive(p),inside,padded_room,ending}}

def into_ready(+rows: U32,+end: U32,+window: Packing.Stencil,+pack_start: U32,+offset: U32,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,
  +tiles: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,+pair_depth: Nat,+units: Nat,
  +cap: U32,+ready: CallReady<Pipeline.WindowWork{rows,end,window},pack_start,offset,pairs,tiles,depth,pair_depth,units,cap>) ->
  {values_of(Pixels.window_values(window,input),call_cells(cells_into(Pipeline.WindowWork{rows,end,window},pack_start,offset,input,Cells{pairs,tiles,weights,bias,output}))) ==
    values_into(Pipeline.WindowWork{rows,end,SpatialSamples.plain(window)},pack_start,offset,State{Pixels.window_values(window,input),tiles,weights,bias,output}) : State}:
  CallReady{+packable,+block,+balanced,+pairs_balanced,+track} = ready
  +state = {State{Pixels.window_values(window,input),tiles,weights,bias,output} : State}
  Equal.trans(State,values_of(Pixels.window_values(window,input),call_cells(cells_into(Pipeline.WindowWork{rows,end,window},pack_start,offset,input,
      Cells{pairs,tiles,weights,bias,output}))),
    values_into(Pipeline.WindowWork{rows,end,window},pack_start,offset,state),values_into(Pipeline.WindowWork{rows,end,SpatialSamples.plain(window)},pack_start,offset,state),
    into_values(rows,end,window,pack_start,offset,input,pairs,tiles,weights,bias,output,depth,pair_depth,cap,units,packable,block,balanced,pairs_balanced,track),
    values_plain(rows,end,window,pack_start,offset,state))

# The pairs a ready call leaves are a tree of the depth of its pairs.
def ready_pairs(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,
  +depth: Nat,+pair_depth: Nat,+units: Nat,+cap: U32,+ready: CallReady<work,pack_start,offset,pairs,tiles,depth,pair_depth,units,cap>) ->
  Shape.Balanced(Tile.Tile16,pair_depth,PackingModel.transposed(call_window(work),call_planes(work,pack_start,offset),input,pairs)):
  CallReady{+packable,+block,+balanced,+pairs_balanced,+track} = ready
  Pixels.Block{+geometry,sixteen,positive,width_bound,lanes_room,stride_positive,stride_bound,ordered,row_exact,width_fits,columns_room,values_room,below_first,covered,
    positions_room} = block
  Planes.transposed_balanced(input,pairs,call_window(work),call_planes(work,pack_start,offset),pair_depth,cap,geometry,pairs_balanced)

# ---- The buffers a call keeps ----

# The cells a buffer holds.
def cells_of(~Element: Data,buffer: Storage.Buffer<Element>) -> Tree.Tree<Element>:
  Buffer.content_tree(~Element,Buffer.contents(~Element,buffer))

def kept_call(kept: Pipeline.Kept) -> Call:
  Pipeline.Kept{input,weights,bias,planes,tiles,output} = kept
  Call{cells_of(~Tile.Tile8,input),Cells{cells_of(~Tile.Tile16,planes),cells_of(~Tile.Tile8,tiles),cells_of(~Tile.Tile8,weights),cells_of(~Tile.Tile8,bias),
    cells_of(~Tile.Tile8,output)}}

def reflected(called: Projection.Called) -> Call:
  Projection.Called{input,Projection.State{pairs,tiles,weights,bias,output}} = called
  Call{Tree.reflect(Tile.Tile8,input),Cells{Tree.reflect(Tile.Tile16,pairs),Tree.reflect(Tile.Tile8,tiles),Tree.reflect(Tile.Tile8,weights),Tree.reflect(Tile.Tile8,bias),
    Tree.reflect(Tile.Tile8,output)}}

def kept_reflected(kept: Pipeline.Kept) -> {kept_call(kept) == reflected(Projection.project_kept(kept)) : Call}:
  Pipeline.Kept{Storage.Buffer{input,il,ic},Storage.Buffer{weights,wl,wc},Storage.Buffer{bias,bl,bc},Storage.Buffer{planes,pl,pc},Storage.Buffer{tiles,tl,tc},
    Storage.Buffer{output,ol,oc}} = kept
  {==}

def represented_reflected(+call: Call) -> {reflected(represent(call)) == call : Call}:
  Call{+input,Cells{+pairs,+tiles,+weights,+bias,+output}} = call
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,input)),input,StorageLaws.reflect_packed(Tile.Tile8,input)) :
    {Call{_,Cells{Tree.reflect(Tile.Tile16,Tree.pack(Tile.Tile16,pairs)),Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,tiles)),Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),
      Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,bias)),Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output))}} == Call{input,Cells{pairs,tiles,weights,bias,output}} : Call}
  %Equal.sym(Tree.Tree<Tile.Tile16>,Tree.reflect(Tile.Tile16,Tree.pack(Tile.Tile16,pairs)),pairs,StorageLaws.reflect_packed(Tile.Tile16,pairs)) :
    {Call{input,Cells{_,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,tiles)),Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),
      Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,bias)),Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output))}} == Call{input,Cells{pairs,tiles,weights,bias,output}} : Call}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,tiles)),tiles,StorageLaws.reflect_packed(Tile.Tile8,tiles)) :
    {Call{input,Cells{pairs,_,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,bias)),
      Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output))}} == Call{input,Cells{pairs,tiles,weights,bias,output}} : Call}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),weights,StorageLaws.reflect_packed(Tile.Tile8,weights)) :
    {Call{input,Cells{pairs,tiles,_,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,bias)),Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output))}} ==
      Call{input,Cells{pairs,tiles,weights,bias,output}} : Call}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,bias)),bias,StorageLaws.reflect_packed(Tile.Tile8,bias)) :
    {Call{input,Cells{pairs,tiles,weights,_,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output))}} == Call{input,Cells{pairs,tiles,weights,bias,output}} : Call}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output,StorageLaws.reflect_packed(Tile.Tile8,output)) :
    {Call{input,Cells{pairs,tiles,weights,bias,_}} == Call{input,Cells{pairs,tiles,weights,bias,output}} : Call}
  {==}

# The arrays of a call on the buffers of models are those of the call on the
# trees of the models.
def into_called(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,
  +planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Projection.project_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),
      Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))) ==
    represent(cells_into(work,pack_start,offset,Buffer.tree(~Tile.Tile8,input),Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),
      Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,output)})) : Projection.Called}:
  Buffer.Model{+it,+il,+ic} = input
  Buffer.Model{+wt,+wl,+wc} = weights
  Buffer.Model{+bt,+bl,+bc} = bias
  Buffer.Model{+pt,+pl,+pc} = planes
  Buffer.Model{+tt,+tl,+tc} = tiles
  Buffer.Model{+ot,+ol,+oc} = output
  Equal.trans(Projection.Called,
    Projection.project_kept(Pipeline.blocked_into(work,pack_start,offset,Storage.Buffer{Tree.pack(Tile.Tile8,it),il,ic},Storage.Buffer{Tree.pack(Tile.Tile8,wt),wl,wc},
      Storage.Buffer{Tree.pack(Tile.Tile8,bt),bl,bc},Storage.Buffer{Tree.pack(Tile.Tile16,pt),pl,pc},Storage.Buffer{Tree.pack(Tile.Tile8,tt),tl,tc},
      Storage.Buffer{Tree.pack(Tile.Tile8,ot),ol,oc})),
    Projection.into(work,pack_start,offset,Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile8,wt),Tree.pack(Tile.Tile8,bt),Tree.pack(Tile.Tile16,pt),Tree.pack(Tile.Tile8,tt),
      Tree.pack(Tile.Tile8,ot)),
    represent(cells_into(work,pack_start,offset,it,Cells{pt,tt,wt,bt,ot})),
    Projection.into_projection(work,pack_start,offset,Buffer.Model{it,il,ic},Buffer.Model{wt,wl,wc},Buffer.Model{bt,bl,bc},Buffer.Model{pt,pl,pc},Buffer.Model{tt,tl,tc},
      Buffer.Model{ot,ol,oc}),
    into_matches(work,pack_start,offset,it,pt,tt,wt,bt,ot))

# The cells of the model a buffer was seen as.
def seen_tree(~Element: Data,-buffer: Storage.Buffer<Element>,seen: Buffer.Observation<Element,buffer>) -> Tree.Tree<Element>:
  Buffer.tree(~Element,Buffer.observed_model(~Element,buffer,seen))

def packed_cells_of(~Element: Data,+model: Buffer.Model<Element>) -> {cells_of(~Element,Buffer.pack(~Element,model)) == Buffer.tree(~Element,model) : Tree.Tree<Element>}:
  Buffer.Model{+tree,+length,+certificate} = model
  StorageLaws.reflect_packed(Element,tree)

def seen_cells(~Element: Data,-buffer: Storage.Buffer<Element>,+seen: Buffer.Observation<Element,buffer>) ->
  {seen_tree(~Element,buffer,seen) == cells_of(~Element,buffer) : Tree.Tree<Element>}:
  Buffer.Observation{+model,+equation} = seen
  %equation : {Buffer.tree(~Element,model) == cells_of(~Element,_) : Tree.Tree<Element>}
  Equal.sym(Tree.Tree<Element>,cells_of(~Element,Buffer.pack(~Element,model)),Buffer.tree(~Element,model),packed_cells_of(~Element,model))

# The buffers a call keeps hold the call on the cells of its buffers.
def into_call(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,-input: Storage.Buffer<Tile.Tile8>,-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<Tile.Tile8>,
  -planes: Storage.Buffer<Tile.Tile16>,-tiles: Storage.Buffer<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>,
  input_seen: Buffer.Observation<Tile.Tile8,input>,weight_seen: Buffer.Observation<Tile.Tile8,weights>,bias_seen: Buffer.Observation<Tile.Tile8,bias>,
  plane_seen: Buffer.Observation<Tile.Tile16,planes>,tile_seen: Buffer.Observation<Tile.Tile8,tiles>,output_seen: Buffer.Observation<Tile.Tile8,output>) ->
  {kept_call(Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output)) ==
    cells_into(work,pack_start,offset,seen_tree(~Tile.Tile8,input,input_seen),Cells{seen_tree(~Tile.Tile16,planes,plane_seen),seen_tree(~Tile.Tile8,tiles,tile_seen),
      seen_tree(~Tile.Tile8,weights,weight_seen),seen_tree(~Tile.Tile8,bias,bias_seen),seen_tree(~Tile.Tile8,output,output_seen)}) : Call}:
  Buffer.Observation{+input_model,+input_equation} = input_seen
  Buffer.Observation{+weight_model,+weight_equation} = weight_seen
  Buffer.Observation{+bias_model,+bias_equation} = bias_seen
  Buffer.Observation{+plane_model,+plane_equation} = plane_seen
  Buffer.Observation{+tile_model,+tile_equation} = tile_seen
  Buffer.Observation{+output_model,+output_equation} = output_seen
  +x = Buffer.tree(~Tile.Tile8,input_model)
  +state = {Cells{Buffer.tree(~Tile.Tile16,plane_model),Buffer.tree(~Tile.Tile8,tile_model),Buffer.tree(~Tile.Tile8,weight_model),Buffer.tree(~Tile.Tile8,bias_model),
    Buffer.tree(~Tile.Tile8,output_model)} : Cells}
  %input_equation : {kept_call(Pipeline.blocked_into(work,pack_start,offset,_,weights,bias,planes,tiles,output)) == cells_into(work,pack_start,offset,x,state) : Call}
  %weight_equation : {kept_call(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input_model),_,bias,planes,tiles,output)) ==
    cells_into(work,pack_start,offset,x,state) : Call}
  %bias_equation : {kept_call(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,weight_model),_,planes,tiles,output)) ==
    cells_into(work,pack_start,offset,x,state) : Call}
  %plane_equation : {kept_call(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,weight_model),
      Buffer.pack(~Tile.Tile8,bias_model),_,tiles,output)) == cells_into(work,pack_start,offset,x,state) : Call}
  %tile_equation : {kept_call(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,weight_model),
      Buffer.pack(~Tile.Tile8,bias_model),Buffer.pack(~Tile.Tile16,plane_model),_,output)) == cells_into(work,pack_start,offset,x,state) : Call}
  %output_equation : {kept_call(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,weight_model),
      Buffer.pack(~Tile.Tile8,bias_model),Buffer.pack(~Tile.Tile16,plane_model),Buffer.pack(~Tile.Tile8,tile_model),_)) == cells_into(work,pack_start,offset,x,state) : Call}
  Equal.trans(Call,kept_call(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~Tile.Tile8,bias_model),Buffer.pack(~Tile.Tile16,plane_model),Buffer.pack(~Tile.Tile8,tile_model),Buffer.pack(~Tile.Tile8,output_model))),reflected(Projection.project_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~Tile.Tile8,bias_model),Buffer.pack(~Tile.Tile16,plane_model),Buffer.pack(~Tile.Tile8,tile_model),Buffer.pack(~Tile.Tile8,output_model)))),cells_into(work,pack_start,offset,x,state),
    kept_reflected(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~Tile.Tile8,bias_model),Buffer.pack(~Tile.Tile16,plane_model),Buffer.pack(~Tile.Tile8,tile_model),Buffer.pack(~Tile.Tile8,output_model))),
    Equal.trans(Call,reflected(Projection.project_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~Tile.Tile8,bias_model),Buffer.pack(~Tile.Tile16,plane_model),Buffer.pack(~Tile.Tile8,tile_model),Buffer.pack(~Tile.Tile8,output_model)))),reflected(represent(cells_into(work,pack_start,offset,x,state))),cells_into(work,pack_start,offset,x,state),
      Equal.cong(Projection.Called,Call,called => reflected(called),Projection.project_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,weight_model),Buffer.pack(~Tile.Tile8,bias_model),Buffer.pack(~Tile.Tile16,plane_model),Buffer.pack(~Tile.Tile8,tile_model),Buffer.pack(~Tile.Tile8,output_model))),represent(cells_into(work,pack_start,offset,x,state)),
        into_called(work,pack_start,offset,input_model,weight_model,bias_model,plane_model,tile_model,output_model)),
      represented_reflected(cells_into(work,pack_start,offset,x,state))))

# The trees of the models of the buffers a call keeps.
def kept_trees(model: BufferShape.KeptModel) -> Call:
  BufferShape.KeptModel{input,weights,bias,planes,tiles,output} = model
  Call{Buffer.tree(~Tile.Tile8,input),Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),
    Buffer.tree(~Tile.Tile8,output)}}

def packed_kept_trees(+model: BufferShape.KeptModel) -> {kept_call(BufferShape.pack_kept(model)) == kept_trees(model) : Call}:
  BufferShape.KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  %Equal.sym(Tree.Tree<Tile.Tile8>,cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input)),Buffer.tree(~Tile.Tile8,input),packed_cells_of(~Tile.Tile8,input)) :
    {Call{_,Cells{cells_of(~Tile.Tile16,Buffer.pack(~Tile.Tile16,planes)),cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),
      cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias)),cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output))}} == kept_trees(BufferShape.KeptModel{input,weights,bias,planes,tiles,output}) :
      Call}
  %Equal.sym(Tree.Tree<Tile.Tile16>,cells_of(~Tile.Tile16,Buffer.pack(~Tile.Tile16,planes)),Buffer.tree(~Tile.Tile16,planes),packed_cells_of(~Tile.Tile16,planes)) :
    {Call{Buffer.tree(~Tile.Tile8,input),Cells{_,cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),
      cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias)),cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output))}} == kept_trees(BufferShape.KeptModel{input,weights,bias,planes,tiles,output}) :
      Call}
  %Equal.sym(Tree.Tree<Tile.Tile8>,cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),Buffer.tree(~Tile.Tile8,tiles),packed_cells_of(~Tile.Tile8,tiles)) :
    {Call{Buffer.tree(~Tile.Tile8,input),Cells{Buffer.tree(~Tile.Tile16,planes),_,cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),
      cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias)),cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output))}} == kept_trees(BufferShape.KeptModel{input,weights,bias,planes,tiles,output}) :
      Call}
  %Equal.sym(Tree.Tree<Tile.Tile8>,cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),Buffer.tree(~Tile.Tile8,weights),packed_cells_of(~Tile.Tile8,weights)) :
    {Call{Buffer.tree(~Tile.Tile8,input),Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),_,cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias)),
      cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output))}} == kept_trees(BufferShape.KeptModel{input,weights,bias,planes,tiles,output}) : Call}
  %Equal.sym(Tree.Tree<Tile.Tile8>,cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias)),Buffer.tree(~Tile.Tile8,bias),packed_cells_of(~Tile.Tile8,bias)) :
    {Call{Buffer.tree(~Tile.Tile8,input),Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),_,
      cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output))}} == kept_trees(BufferShape.KeptModel{input,weights,bias,planes,tiles,output}) : Call}
  %Equal.sym(Tree.Tree<Tile.Tile8>,cells_of(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output)),Buffer.tree(~Tile.Tile8,output),packed_cells_of(~Tile.Tile8,output)) :
    {Call{Buffer.tree(~Tile.Tile8,input),Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),_}} ==
      kept_trees(BufferShape.KeptModel{input,weights,bias,planes,tiles,output}) : Call}
  {==}

# The models of the buffers a kept state holds are its cells.
def observed_kept_trees(-kept: Pipeline.Kept,+observation: BufferShape.KeptObservation<kept>) ->
  {kept_trees(BufferShape.kept_model(kept,observation)) == kept_call(kept) : Call}:
  BufferShape.KeptObservation{+model,+equation} = observation
  %equation : {kept_trees(model) == kept_call(_) : Call}
  Equal.sym(Call,kept_call(BufferShape.pack_kept(model)),kept_trees(model),packed_kept_trees(model))

# The models of the buffers a call gives back hold the call on the cells of
# its buffers.
def into_kept_trees(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,-input: Storage.Buffer<Tile.Tile8>,-weights: Storage.Buffer<Tile.Tile8>,
  -bias: Storage.Buffer<Tile.Tile8>,-planes: Storage.Buffer<Tile.Tile16>,-tiles: Storage.Buffer<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>,
  input_seen: Buffer.Observation<Tile.Tile8,input>,weight_seen: Buffer.Observation<Tile.Tile8,weights>,bias_seen: Buffer.Observation<Tile.Tile8,bias>,
  plane_seen: Buffer.Observation<Tile.Tile16,planes>,tile_seen: Buffer.Observation<Tile.Tile8,tiles>,output_seen: Buffer.Observation<Tile.Tile8,output>,
  +observation: BufferShape.KeptObservation<Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output)>) ->
  {kept_trees(BufferShape.kept_model(Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output),observation)) ==
    cells_into(work,pack_start,offset,seen_tree(~Tile.Tile8,input,input_seen),Cells{seen_tree(~Tile.Tile16,planes,plane_seen),seen_tree(~Tile.Tile8,tiles,tile_seen),
      seen_tree(~Tile.Tile8,weights,weight_seen),seen_tree(~Tile.Tile8,bias,bias_seen),seen_tree(~Tile.Tile8,output,output_seen)}) : Call}:
  Equal.trans(Call,kept_trees(BufferShape.kept_model(Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output),observation)),
    kept_call(Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output)),
    cells_into(work,pack_start,offset,seen_tree(~Tile.Tile8,input,input_seen),Cells{seen_tree(~Tile.Tile16,planes,plane_seen),seen_tree(~Tile.Tile8,tiles,tile_seen),
      seen_tree(~Tile.Tile8,weights,weight_seen),seen_tree(~Tile.Tile8,bias,bias_seen),seen_tree(~Tile.Tile8,output,output_seen)}),
    observed_kept_trees(Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output),observation),
    into_call(work,pack_start,offset,input,weights,bias,planes,tiles,output,input_seen,weight_seen,bias_seen,plane_seen,tile_seen,output_seen))

def kept_input_tree(+model: BufferShape.KeptModel) -> {Buffer.tree(~Tile.Tile8,BufferShape.model_input(model)) == call_input(kept_trees(model)) : Tree.Tree<Tile.Tile8>}:
  BufferShape.KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  {==}

# The contents of a buffer are its cells and its length.
def contents_parts(-buffer: Storage.Buffer<Tile.Tile8>,seen: Buffer.Observation<Tile.Tile8,buffer>) ->
  {Buffer.Content{cells_of(~Tile.Tile8,buffer),BufferShape.buffer_length(~Tile.Tile8,buffer)} == Buffer.contents(~Tile.Tile8,buffer) : Buffer.Content<Tile.Tile8>}:
  Buffer.Observation{model,+equation} = seen
  Buffer.Model{+tree,+length,+certificate} = model
  %equation : {Buffer.Content{cells_of(~Tile.Tile8,_),BufferShape.buffer_length(~Tile.Tile8,_)} == Buffer.contents(~Tile.Tile8,_) : Buffer.Content<Tile.Tile8>}
  {==}

# A model of the cells and the length of a buffer has its contents.
def content_from(+model: Buffer.Model<Tile.Tile8>,-buffer: Storage.Buffer<Tile.Tile8>,seen: Buffer.Observation<Tile.Tile8,buffer>,
  +tree_exact: {Buffer.tree(~Tile.Tile8,model) == cells_of(~Tile.Tile8,buffer) : Tree.Tree<Tile.Tile8>},
  +length_exact: {Buffer.length(~Tile.Tile8,model) == BufferShape.buffer_length(~Tile.Tile8,buffer) : Nat}) ->
  {Buffer.content(~Tile.Tile8,model) == Buffer.contents(~Tile.Tile8,buffer) : Buffer.Content<Tile.Tile8>}:
  Buffer.Model{+tree,+length,+certificate} = model
  Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.Content{tree,length},Buffer.Content{cells_of(~Tile.Tile8,buffer),BufferShape.buffer_length(~Tile.Tile8,buffer)},
    Buffer.contents(~Tile.Tile8,buffer),{
      %tree_exact : {Buffer.Content{tree,length} == Buffer.Content{_,BufferShape.buffer_length(~Tile.Tile8,buffer)} : Buffer.Content<Tile.Tile8>}
      %length_exact : {Buffer.Content{tree,length} == Buffer.Content{tree,_} : Buffer.Content<Tile.Tile8>}
      {==} : {Buffer.Content{tree,length} == Buffer.Content{cells_of(~Tile.Tile8,buffer),BufferShape.buffer_length(~Tile.Tile8,buffer)} : Buffer.Content<Tile.Tile8>}},
    contents_parts(buffer,seen))

# The input a call gives back has the contents it was given.
def kept_input_content(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,-input: Storage.Buffer<Tile.Tile8>,-weights: Storage.Buffer<Tile.Tile8>,
  -bias: Storage.Buffer<Tile.Tile8>,-planes: Storage.Buffer<Tile.Tile16>,-tiles: Storage.Buffer<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>,
  +input_seen: Buffer.Observation<Tile.Tile8,input>,+weight_seen: Buffer.Observation<Tile.Tile8,weights>,+bias_seen: Buffer.Observation<Tile.Tile8,bias>,
  +plane_seen: Buffer.Observation<Tile.Tile16,planes>,+tile_seen: Buffer.Observation<Tile.Tile8,tiles>,+output_seen: Buffer.Observation<Tile.Tile8,output>,
  +observation: BufferShape.KeptObservation<Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output)>) ->
  {Buffer.content(~Tile.Tile8,BufferShape.model_input(BufferShape.kept_model(Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output),observation))) ==
    Buffer.contents(~Tile.Tile8,input) : Buffer.Content<Tile.Tile8>}:
  -kept = Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output)
  +model = BufferShape.kept_model(kept,observation)
  +state = {Cells{seen_tree(~Tile.Tile16,planes,plane_seen),seen_tree(~Tile.Tile8,tiles,tile_seen),seen_tree(~Tile.Tile8,weights,weight_seen),
    seen_tree(~Tile.Tile8,bias,bias_seen),seen_tree(~Tile.Tile8,output,output_seen)} : Cells}
  +call = cells_into(work,pack_start,offset,seen_tree(~Tile.Tile8,input,input_seen),state)
  +tree_exact = Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,BufferShape.model_input(model)),call_input(kept_trees(model)),cells_of(~Tile.Tile8,input),
    kept_input_tree(model),
    Equal.trans(Tree.Tree<Tile.Tile8>,call_input(kept_trees(model)),call_input(call),cells_of(~Tile.Tile8,input),
      Equal.cong(Call,Tree.Tree<Tile.Tile8>,found => call_input(found),kept_trees(model),call,
        into_kept_trees(work,pack_start,offset,input,weights,bias,planes,tiles,output,input_seen,weight_seen,bias_seen,plane_seen,tile_seen,output_seen,observation)),
      seen_cells(~Tile.Tile8,input,input_seen)))
  +length_exact = Equal.trans(Nat,Buffer.length(~Tile.Tile8,BufferShape.model_input(model)),BufferShape.input_length(BufferShape.model_lengths(model)),
    BufferShape.buffer_length(~Tile.Tile8,input),BufferShape.model_input_length(model),
    Equal.cong(BufferShape.Lengths,Nat,lengths => BufferShape.input_length(lengths),BufferShape.model_lengths(model),
      BufferShape.Lengths{BufferShape.buffer_length(~Tile.Tile8,input),BufferShape.buffer_length(~Tile.Tile16,planes),BufferShape.buffer_length(~Tile.Tile8,tiles),
        BufferShape.buffer_length(~Tile.Tile8,weights),BufferShape.buffer_length(~Tile.Tile8,bias),BufferShape.buffer_length(~Tile.Tile8,output)},
      BufferShape.into_kept_lengths(work,pack_start,offset,input,weights,bias,planes,tiles,output,input_seen,weight_seen,bias_seen,plane_seen,tile_seen,output_seen,observation)))
  content_from(BufferShape.model_input(model),input,input_seen,tree_exact,length_exact)
