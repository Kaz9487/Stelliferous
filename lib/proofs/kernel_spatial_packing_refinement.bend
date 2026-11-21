# The packer on storage trees: the transpose and the pack of
# kernel_spatial_packing, run on packed trees, give the packed trees of these
# models and hand back the tree they read.
import Base
import ../traversal_loop.bend as Loop
import ../kernel_spatial_packing.bend as Packing
import ../kernel_tile.bend as Tile
import ./loop_projection.bend as Projection
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./kernel_spatial_packing_samples.bend as Reads

# ---- The transpose ----

type Rowing is Data:
  Rowing{input: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>,previous: Tile.Tile8,column: U32,write: U32,row: U32,in_row: U32,row_tiles: U32,first_column: U32,
    channel_base: U32,height: U32,width: U32,stride: U32,groups: U32,channels: U32}

def pack_rowing(state: Rowing) -> Packing.Rowing:
  Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels} = state
  Packing.Rowing{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile16,pairs),previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,
    channels}

# Eight input columns stride apart, lane of the cells of channel_base.
def row_tile(~lane: Tile.Tile8 -> F32,+input: Tree.Tree<Tile.Tile8>,+row: U32,+column: U32,+channel_base: U32,+height: U32,+width: U32,+stride: U32,+groups: U32) -> Tile.Tile8:
  Reads.pure_vector(~Tile.Tile8,~lane,input,Packing.PixelBlock{row,column,channel_base,height,width,stride,groups})

def moved(input: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>,tile: Tile.Tile8,+column: U32,write: U32,+row: U32,+in_row: U32,+row_tiles: U32,+first_column: U32,
  channel_base: U32,height: U32,width: U32,+stride: U32,groups: U32,channels: U32) -> Rowing:
  +wrap = U32.is_eq((in_row + 1 : U32),row_tiles)
  Rowing{input,pairs,tile,Bool.pick(U32,wrap,first_column,(column + 8 * stride : U32)),write,Bool.pick(U32,wrap,(row + 1 : U32),row),Bool.pick(U32,wrap,0,(in_row + 1 : U32)),
    row_tiles,first_column,channel_base,height,width,stride,groups,channels}

def row_first(~lane: Tile.Tile8 -> F32,state: Rowing) -> Rowing:
  Rowing{+input,pairs,previous,+column,write,+row,in_row,row_tiles,first_column,+channel_base,+height,+width,+stride,+groups,channels} = state
  moved(input,pairs,row_tile(~lane,input,row,column,channel_base,height,width,stride,groups),column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,
    groups,channels)

def row_step(~lane: Tile.Tile8 -> F32,state: Rowing) -> Rowing:
  Rowing{+input,pairs,previous,+column,+write,+row,in_row,row_tiles,first_column,+channel_base,+height,+width,+stride,+groups,+channels} = state
  +tile = row_tile(~lane,input,row,column,channel_base,height,width,stride,groups)
  moved(input,Tree.write(Tile.Tile16,pairs,write,Tile.Tiles{previous,tile}),tile,column,(write + channels : U32),row,in_row,row_tiles,first_column,channel_base,height,width,
    stride,groups,channels)

def row_read(~lane: Tile.Tile8 -> F32,+input: Tree.Tree<Tile.Tile8>,+row: U32,+column: U32,+channel_base: U32,+height: U32,+width: U32,+stride: U32,+groups: U32) ->
  {Packing.block(~Tile.Tile8,~lane,Tree.pack(Tile.Tile8,input),row,column,channel_base,height,width,stride,0,0,0,0,0,groups) ==
    (Tree.pack(Tile.Tile8,input),row_tile(~lane,input,row,column,channel_base,height,width,stride,groups)) : Array<Tile.Tile8> & Tile.Tile8}:
  Reads.native_vector_matches_tabulation(~Tile.Tile8,~lane,input,Packing.PixelBlock{row,column,channel_base,height,width,stride,groups})

def row_first_matches(~lane: Tile.Tile8 -> F32,+state: Rowing) -> {Packing.row_first(~lane,pack_rowing(state)) == pack_rowing(row_first(~lane,state)) : Packing.Rowing}:
  Rowing{+input,+pairs,+previous,+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = state
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Packing.block(~Tile.Tile8,~lane,Tree.pack(Tile.Tile8,input),row,column,channel_base,height,width,stride,0,0,0,0,0,groups),
    (Tree.pack(Tile.Tile8,input),row_tile(~lane,input,row,column,channel_base,height,width,stride,groups)),
    row_read(~lane,input,row,column,channel_base,height,width,stride,groups)) :
    {Packing.row_first_read(Tree.pack(Tile.Tile16,pairs),previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels,_) ==
      pack_rowing(row_first(~lane,Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels})) : Packing.Rowing}
  {==}

def row_step_matches(~lane: Tile.Tile8 -> F32,+state: Rowing) -> {Packing.row_step(~lane,pack_rowing(state)) == pack_rowing(row_step(~lane,state)) : Packing.Rowing}:
  Rowing{+input,+pairs,+previous,+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = state
  +tile = row_tile(~lane,input,row,column,channel_base,height,width,stride,groups)
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Packing.block(~Tile.Tile8,~lane,Tree.pack(Tile.Tile8,input),row,column,channel_base,height,width,stride,0,0,0,0,0,groups),
    (Tree.pack(Tile.Tile8,input),tile),row_read(~lane,input,row,column,channel_base,height,width,stride,groups)) :
    {Packing.row_read(Tree.pack(Tile.Tile16,pairs),previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels,_) ==
      pack_rowing(row_step(~lane,Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels})) : Packing.Rowing}
  %Equal.sym(Array<Tile.Tile16>,Array.set(Tile.Tile16,Tree.pack(Tile.Tile16,pairs),write,Tile.Tiles{previous,tile}),
    Tree.pack(Tile.Tile16,Tree.write(Tile.Tile16,pairs,write,Tile.Tiles{previous,tile})),StorageLaws.write(Tile.Tile16,pairs,write,Tile.Tiles{previous,tile})) :
    {Packing.row_moved(Tree.pack(Tile.Tile8,input),_,tile,column,(write + channels : U32),row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels) ==
      pack_rowing(row_step(~lane,Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels})) : Packing.Rowing}
  {==}

def row_run(~lane: Tile.Tile8 -> F32,count: Nat,state: Rowing) -> Rowing:
  Loop.run(~Rowing,~row_step(~lane),count,row_first(~lane,state))

def row_run_matches(~lane: Tile.Tile8 -> F32,+count: Nat,+state: Rowing) ->
  {Packing.row_run(~lane,count,pack_rowing(state)) == pack_rowing(row_run(~lane,count,state)) : Packing.Rowing}:
  %Equal.sym(Packing.Rowing,Packing.row_first(~lane,pack_rowing(state)),pack_rowing(row_first(~lane,state)),row_first_matches(~lane,state)) :
    {Loop.run(~Packing.Rowing,~Packing.row_step(~lane),count,_) == pack_rowing(row_run(~lane,count,state)) : Packing.Rowing}
  Projection.iteration_refines_model(~Packing.Rowing,~Rowing,~Packing.row_step(~lane),~row_step(~lane),~pack_rowing,~row_step_matches(~lane),count,row_first(~lane,state))

def row_low(low: Bool,first: Bool,count: Nat,state: Rowing) -> Rowing:
  match low first:
    case True{} True{}: row_run(~Tile.lane0,count,state)
    case True{} False{}: row_run(~Tile.lane1,count,state)
    case False{} True{}: row_run(~Tile.lane2,count,state)
    case False{} False{}: row_run(~Tile.lane3,count,state)

def row_high(low: Bool,first: Bool,count: Nat,state: Rowing) -> Rowing:
  match low first:
    case True{} True{}: row_run(~Tile.lane4,count,state)
    case True{} False{}: row_run(~Tile.lane5,count,state)
    case False{} True{}: row_run(~Tile.lane6,count,state)
    case False{} False{}: row_run(~Tile.lane7,count,state)

def row_lane(half: Bool,low: Bool,first: Bool,count: Nat,state: Rowing) -> Rowing:
  match half:
    case True{}: row_low(low,first,count,state)
    case False{}: row_high(low,first,count,state)

def row_low_matches(+low: Bool,+first: Bool,+count: Nat,+state: Rowing) ->
  {Packing.row_low(low,first,count,pack_rowing(state)) == pack_rowing(row_low(low,first,count,state)) : Packing.Rowing}:
  match low first:
    case True{} True{}: row_run_matches(~Tile.lane0,count,state)
    case True{} False{}: row_run_matches(~Tile.lane1,count,state)
    case False{} True{}: row_run_matches(~Tile.lane2,count,state)
    case False{} False{}: row_run_matches(~Tile.lane3,count,state)

def row_high_matches(+low: Bool,+first: Bool,+count: Nat,+state: Rowing) ->
  {Packing.row_high(low,first,count,pack_rowing(state)) == pack_rowing(row_high(low,first,count,state)) : Packing.Rowing}:
  match low first:
    case True{} True{}: row_run_matches(~Tile.lane4,count,state)
    case True{} False{}: row_run_matches(~Tile.lane5,count,state)
    case False{} True{}: row_run_matches(~Tile.lane6,count,state)
    case False{} False{}: row_run_matches(~Tile.lane7,count,state)

def row_lane_matches(+half: Bool,+low: Bool,+first: Bool,+count: Nat,+state: Rowing) ->
  {Packing.row_lane(half,low,first,count,pack_rowing(state)) == pack_rowing(row_lane(half,low,first,count,state)) : Packing.Rowing}:
  match half:
    case True{}: row_low_matches(low,first,count,state)
    case False{}: row_high_matches(low,first,count,state)

type Transposing is Data:
  Transposing{input: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>,plane: U32,window: Packing.Stencil,planes: Packing.Planes}

def pack_transposing(state: Transposing) -> Packing.Transposing:
  Transposing{input,pairs,plane,window,planes} = state
  Packing.Transposing{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile16,pairs),plane,window,planes}

def plane_done(+plane: U32,+window: Packing.Stencil,+planes: Packing.Planes,done: Rowing) -> Transposing:
  Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels} = done
  Transposing{input,pairs,(plane + 1 : U32),window,planes}

# Plane phase * channels + channel starts at its phase column, its first
# element and the first plane row.
def plane_start(input: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>,+plane: U32,+window: Packing.Stencil,+planes: Packing.Planes) -> Rowing:
  Packing.Stencil{+base,+height,+width,tap_rows,tap_columns,+stride,+row_origin,+column_origin,output_width,inner,+groups} = window
  Packing.Planes{+first_output_row,rows,+row_width,+elements} = planes
  +channels = Packing.channels_of(window)
  +channel = (plane % channels : U32)
  +phase = (plane / channels : U32)
  Rowing{input,pairs,Tile.filled(0.0),(phase + column_origin : U32),(phase * elements * channels + channel : U32),(first_output_row * stride + row_origin : U32),0,
    (row_width / 8 : U32),(phase + column_origin : U32),(base + channel / 8 : U32),height,width,stride,groups,channels}

# The lane of a plane's channel in its cells.
def plane_lane(+plane: U32,+window: Packing.Stencil) -> U32:
  (plane % Packing.channels_of(window) % 8 : U32)

def plane_rows(+plane: U32,+window: Packing.Stencil,+planes: Packing.Planes,state: Rowing) -> Rowing:
  Packing.Planes{first_output_row,rows,row_width,+elements} = planes
  +lane = plane_lane(plane,window)
  row_lane(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),U32.to_nat(elements),state)

def plane_step(state: Transposing) -> Transposing:
  Transposing{input,pairs,+plane,+window,+planes} = state
  plane_done(plane,window,planes,plane_rows(plane,window,planes,plane_start(input,pairs,plane,window,planes)))

def plane_done_matches(+plane: U32,+window: Packing.Stencil,+planes: Packing.Planes,+done: Rowing) ->
  {Packing.plane_done(plane,window,planes,pack_rowing(done)) == pack_transposing(plane_done(plane,window,planes,done)) : Packing.Transposing}:
  Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels} = done
  {==}

def plane_step_matches(+state: Transposing) -> {Packing.plane_step(pack_transposing(state)) == pack_transposing(plane_step(state)) : Packing.Transposing}:
  Transposing{+input,+pairs,+plane,+window,+planes} = state
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +start = plane_start(input,pairs,plane,window,planes)
  +lane = plane_lane(plane,window)
  %Equal.sym(Packing.Rowing,Packing.row_lane(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),U32.to_nat(elements),pack_rowing(start)),
    pack_rowing(plane_rows(plane,window,planes,start)),
    row_lane_matches(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),U32.to_nat(elements),start)) :
    {Packing.plane_done(plane,window,planes,_) == pack_transposing(plane_step(Transposing{input,pairs,plane,window,planes})) : Packing.Transposing}
  plane_done_matches(plane,window,planes,plane_rows(plane,window,planes,start))

def transposing(+window: Packing.Stencil,+planes: Packing.Planes,input: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>) -> Transposing:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,output_width,inner,groups} = window
  Loop.run(~Transposing,~plane_step,U32.to_nat((stride * Packing.channels_of(window) : U32)),Transposing{input,pairs,0,window,planes})

def transposing_input(state: Transposing) -> Tree.Tree<Tile.Tile8>:
  Transposing{input,pairs,plane,window,planes} = state
  input

def transposing_pairs(state: Transposing) -> Tree.Tree<Tile.Tile16>:
  Transposing{input,pairs,plane,window,planes} = state
  pairs

# The pairs of the planes.
def transposed(+window: Packing.Stencil,+planes: Packing.Planes,input: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>) -> Tree.Tree<Tile.Tile16>:
  transposing_pairs(transposing(window,planes,input,pairs))

def transposed_done(+state: Transposing) ->
  {Packing.transposed(pack_transposing(state)) == (Tree.pack(Tile.Tile8,transposing_input(state)),Tree.pack(Tile.Tile16,transposing_pairs(state))) :
    Array<Tile.Tile8> & Array<Tile.Tile16>}:
  Transposing{input,pairs,plane,window,planes} = state
  {==}

def rowing_input(state: Rowing) -> Tree.Tree<Tile.Tile8>:
  Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels} = state
  input

def row_keeps_input(~lane: Tile.Tile8 -> F32,+state: Rowing) -> {rowing_input(row_step(~lane,state)) == rowing_input(state) : Tree.Tree<Tile.Tile8>}:
  Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels} = state
  {==}

def row_run_input(~lane: Tile.Tile8 -> F32,+count: Nat,+state: Rowing) -> {rowing_input(row_run(~lane,count,state)) == rowing_input(state) : Tree.Tree<Tile.Tile8>}:
  Rowing{+input,+pairs,+previous,+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = state
  Projection.preserves_observation(~Rowing,~Tree.Tree<Tile.Tile8>,~row_step(~lane),~rowing_input,~row_keeps_input(~lane),count,
    row_first(~lane,Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels}))

def row_low_input(+low: Bool,+first: Bool,+count: Nat,+state: Rowing) -> {rowing_input(row_low(low,first,count,state)) == rowing_input(state) : Tree.Tree<Tile.Tile8>}:
  match low first:
    case True{} True{}: row_run_input(~Tile.lane0,count,state)
    case True{} False{}: row_run_input(~Tile.lane1,count,state)
    case False{} True{}: row_run_input(~Tile.lane2,count,state)
    case False{} False{}: row_run_input(~Tile.lane3,count,state)

def row_high_input(+low: Bool,+first: Bool,+count: Nat,+state: Rowing) -> {rowing_input(row_high(low,first,count,state)) == rowing_input(state) : Tree.Tree<Tile.Tile8>}:
  match low first:
    case True{} True{}: row_run_input(~Tile.lane4,count,state)
    case True{} False{}: row_run_input(~Tile.lane5,count,state)
    case False{} True{}: row_run_input(~Tile.lane6,count,state)
    case False{} False{}: row_run_input(~Tile.lane7,count,state)

def row_lane_input(+half: Bool,+low: Bool,+first: Bool,+count: Nat,+state: Rowing) ->
  {rowing_input(row_lane(half,low,first,count,state)) == rowing_input(state) : Tree.Tree<Tile.Tile8>}:
  match half:
    case True{}: row_low_input(low,first,count,state)
    case False{}: row_high_input(low,first,count,state)

def plane_done_input(+plane: U32,+window: Packing.Stencil,+planes: Packing.Planes,+done: Rowing) ->
  {transposing_input(plane_done(plane,window,planes,done)) == rowing_input(done) : Tree.Tree<Tile.Tile8>}:
  Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels} = done
  {==}

def plane_keeps_input(+state: Transposing) -> {transposing_input(plane_step(state)) == transposing_input(state) : Tree.Tree<Tile.Tile8>}:
  Transposing{+input,+pairs,+plane,+window,+planes} = state
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +start = plane_start(input,pairs,plane,window,planes)
  +lane = plane_lane(plane,window)
  %Equal.sym(Tree.Tree<Tile.Tile8>,transposing_input(plane_done(plane,window,planes,plane_rows(plane,window,planes,start))),rowing_input(plane_rows(plane,window,planes,start)),
    plane_done_input(plane,window,planes,plane_rows(plane,window,planes,start))) : {_ == input : Tree.Tree<Tile.Tile8>}
  row_lane_input(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),U32.to_nat(elements),start)

def transposing_keeps_input(+window: Packing.Stencil,+planes: Packing.Planes,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) ->
  {transposing_input(transposing(window,planes,input,pairs)) == input : Tree.Tree<Tile.Tile8>}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  Projection.preserves_observation(~Transposing,~Tree.Tree<Tile.Tile8>,~plane_step,~transposing_input,~plane_keeps_input,U32.to_nat((stride * Packing.channels_of(window) : U32)),
    Transposing{input,pairs,0,window,planes})

# The transpose on packed trees hands back its input and the packed pairs of
# the model.
law transpose_matches:
  for +window: Packing.Stencil
  for +planes: Packing.Planes
  for +input: Tree.Tree<Tile.Tile8>
  for +pairs: Tree.Tree<Tile.Tile16>
  {Packing.transpose(window,planes,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile16,pairs)) ==
    (Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile16,transposed(window,planes,input,pairs))) : Array<Tile.Tile8> & Array<Tile.Tile16>}
def transpose_matches(window,planes,input,pairs):
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +count = U32.to_nat((stride * Packing.channels_of(window) : U32))
  +state = {Transposing{input,pairs,0,window,planes} : Transposing}
  %Equal.sym(Packing.Transposing,Loop.run(~Packing.Transposing,~Packing.plane_step,count,pack_transposing(state)),
    pack_transposing(Loop.run(~Transposing,~plane_step,count,state)),
    Projection.iteration_refines_model(~Packing.Transposing,~Transposing,~Packing.plane_step,~plane_step,~pack_transposing,~plane_step_matches,count,state)) :
    {Packing.transposed(_) == (Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile16,transposed(window,planes,input,pairs))) : Array<Tile.Tile8> & Array<Tile.Tile16>}
  %transposing_keeps_input(window,planes,input,pairs) :
    {Packing.transposed(pack_transposing(transposing(window,planes,input,pairs))) == (Tree.pack(Tile.Tile8,_),Tree.pack(Tile.Tile16,transposed(window,planes,input,pairs))) :
      Array<Tile.Tile8> & Array<Tile.Tile16>}
  transposed_done(transposing(window,planes,input,pairs))

# ---- The pack ----

type Passing is Data:
  Passing{pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>,start: U32,read: U32,kernel_row: U32,write: U32,lane: U32,tap_rows: U32,tap_columns: U32,row_step: U32}

def pack_passing(state: Passing) -> Packing.Passing:
  Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step} = state
  Packing.Passing{Tree.pack(Tile.Tile16,pairs),Tree.pack(Tile.Tile8,tiles),start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step}

def advanced(pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>,+start: U32,+read: U32,+kernel_row: U32,+write: U32,+lane: U32,+tap_rows: U32,+tap_columns: U32,
  +row_step: U32) -> Passing:
  +wrap = U32.is_eq((kernel_row + 1 : U32),tap_rows)
  Passing{pairs,tiles,Bool.pick(U32,wrap,(start + 1 : U32),start),Bool.pick(U32,wrap,(start + 1 : U32),(read + row_step : U32)),Bool.pick(U32,wrap,0,(kernel_row + 1 : U32)),
    Bool.pick(U32,wrap,(write - (tap_rows - 1) * tap_columns + tap_rows * tap_columns : U32),(write + tap_columns : U32)),lane,tap_rows,tap_columns,row_step}

def first_step(~shift: Tile.Tile16 -> Tile.Tile8,state: Passing) -> Passing:
  Passing{+pairs,tiles,+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = state
  advanced(pairs,Tree.write(Tile.Tile8,tiles,write,shift(Tree.read(Tile.Tile16,pairs,read))),start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step)

def merge_step(~shift: Tile.Tile16 -> Tile.Tile8,state: Passing) -> Passing:
  Passing{+pairs,+tiles,+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = state
  advanced(pairs,Tree.write(Tile.Tile8,tiles,write,Packing.merged(lane,Tree.read(Tile.Tile8,tiles,write),shift(Tree.read(Tile.Tile16,pairs,read)))),start,read,kernel_row,
    write,lane,tap_rows,tap_columns,row_step)

def first_step_matches(~shift: Tile.Tile16 -> Tile.Tile8,+state: Passing) ->
  {Packing.first_step(~shift,pack_passing(state)) == pack_passing(first_step(~shift,state)) : Packing.Passing}:
  Passing{+pairs,+tiles,+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = state
  +pair = Tree.read(Tile.Tile16,pairs,read)
  %Equal.sym(Array<Tile.Tile16> & Tile.Tile16,Array.get(Tile.Tile16,Tree.pack(Tile.Tile16,pairs),read),(Tree.pack(Tile.Tile16,pairs),pair),
    StorageLaws.read(Tile.Tile16,pairs,read)) :
    {Packing.first_stored(~shift,Tree.pack(Tile.Tile8,tiles),start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step,_) ==
      pack_passing(first_step(~shift,Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step})) : Packing.Passing}
  %Equal.sym(Array<Tile.Tile8>,Array.set(Tile.Tile8,Tree.pack(Tile.Tile8,tiles),write,shift(pair)),Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,tiles,write,shift(pair))),
    StorageLaws.write(Tile.Tile8,tiles,write,shift(pair))) :
    {Packing.advanced(Tree.pack(Tile.Tile16,pairs),_,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step) ==
      pack_passing(first_step(~shift,Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step})) : Packing.Passing}
  {==}

def merge_step_matches(~shift: Tile.Tile16 -> Tile.Tile8,+state: Passing) ->
  {Packing.merge_step(~shift,pack_passing(state)) == pack_passing(merge_step(~shift,state)) : Packing.Passing}:
  Passing{+pairs,+tiles,+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = state
  +pair = Tree.read(Tile.Tile16,pairs,read)
  +kept = Tree.read(Tile.Tile8,tiles,write)
  +merged = Packing.merged(lane,kept,shift(pair))
  %Equal.sym(Array<Tile.Tile16> & Tile.Tile16,Array.get(Tile.Tile16,Tree.pack(Tile.Tile16,pairs),read),(Tree.pack(Tile.Tile16,pairs),pair),
    StorageLaws.read(Tile.Tile16,pairs,read)) :
    {Packing.merge_read(~shift,Tree.pack(Tile.Tile8,tiles),start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step,_) ==
      pack_passing(merge_step(~shift,Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step})) : Packing.Passing}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,tiles),write),(Tree.pack(Tile.Tile8,tiles),kept),StorageLaws.read(Tile.Tile8,tiles,write)) :
    {Packing.merge_stored(Tree.pack(Tile.Tile16,pairs),shift(pair),start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step,_) ==
      pack_passing(merge_step(~shift,Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step})) : Packing.Passing}
  %Equal.sym(Array<Tile.Tile8>,Array.set(Tile.Tile8,Tree.pack(Tile.Tile8,tiles),write,merged),Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,tiles,write,merged)),
    StorageLaws.write(Tile.Tile8,tiles,write,merged)) :
    {Packing.advanced(Tree.pack(Tile.Tile16,pairs),_,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step) ==
      pack_passing(merge_step(~shift,Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step})) : Packing.Passing}
  {==}

def pass_run(~shift: Tile.Tile16 -> Tile.Tile8,first: Bool,count: Nat,state: Passing) -> Passing:
  match first:
    case True{}: Loop.run(~Passing,~first_step(~shift),count,state)
    case False{}: Loop.run(~Passing,~merge_step(~shift),count,state)

def pass_run_matches(~shift: Tile.Tile16 -> Tile.Tile8,+first: Bool,+count: Nat,+state: Passing) ->
  {Packing.pass_run(~shift,first,count,pack_passing(state)) == pack_passing(pass_run(~shift,first,count,state)) : Packing.Passing}:
  match first:
    case True{}:
      Projection.iteration_refines_model(~Packing.Passing,~Passing,~Packing.first_step(~shift),~first_step(~shift),~pack_passing,~first_step_matches(~shift),count,state)
    case False{}:
      Projection.iteration_refines_model(~Packing.Passing,~Passing,~Packing.merge_step(~shift),~merge_step(~shift),~pack_passing,~merge_step_matches(~shift),count,state)

def pass_low(low: Bool,odd: Bool,first: Bool,count: Nat,state: Passing) -> Passing:
  match low odd:
    case True{} False{}: pass_run(~Packing.shifted0,first,count,state)
    case True{} True{}: pass_run(~Packing.shifted1,first,count,state)
    case False{} False{}: pass_run(~Packing.shifted2,first,count,state)
    case False{} True{}: pass_run(~Packing.shifted3,first,count,state)

def pass_high(low: Bool,odd: Bool,first: Bool,count: Nat,state: Passing) -> Passing:
  match low odd:
    case True{} False{}: pass_run(~Packing.shifted4,first,count,state)
    case True{} True{}: pass_run(~Packing.shifted5,first,count,state)
    case False{} False{}: pass_run(~Packing.shifted6,first,count,state)
    case False{} True{}: pass_run(~Packing.shifted7,first,count,state)

def pass_shift(half: Bool,low: Bool,odd: Bool,first: Bool,count: Nat,state: Passing) -> Passing:
  match half:
    case True{}: pass_low(low,odd,first,count,state)
    case False{}: pass_high(low,odd,first,count,state)

def pass_low_matches(+low: Bool,+odd: Bool,+first: Bool,+count: Nat,+state: Passing) ->
  {Packing.pass_low(low,odd,first,count,pack_passing(state)) == pack_passing(pass_low(low,odd,first,count,state)) : Packing.Passing}:
  match low odd:
    case True{} False{}: pass_run_matches(~Packing.shifted0,first,count,state)
    case True{} True{}: pass_run_matches(~Packing.shifted1,first,count,state)
    case False{} False{}: pass_run_matches(~Packing.shifted2,first,count,state)
    case False{} True{}: pass_run_matches(~Packing.shifted3,first,count,state)

def pass_high_matches(+low: Bool,+odd: Bool,+first: Bool,+count: Nat,+state: Passing) ->
  {Packing.pass_high(low,odd,first,count,pack_passing(state)) == pack_passing(pass_high(low,odd,first,count,state)) : Packing.Passing}:
  match low odd:
    case True{} False{}: pass_run_matches(~Packing.shifted4,first,count,state)
    case True{} True{}: pass_run_matches(~Packing.shifted5,first,count,state)
    case False{} False{}: pass_run_matches(~Packing.shifted6,first,count,state)
    case False{} True{}: pass_run_matches(~Packing.shifted7,first,count,state)

def pass_shift_matches(+half: Bool,+low: Bool,+odd: Bool,+first: Bool,+count: Nat,+state: Passing) ->
  {Packing.pass_shift(half,low,odd,first,count,pack_passing(state)) == pack_passing(pass_shift(half,low,odd,first,count,state)) : Packing.Passing}:
  match half:
    case True{}: pass_low_matches(low,odd,first,count,state)
    case False{}: pass_high_matches(low,odd,first,count,state)

type Passes is Data:
  Passes{pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>,pass: U32,lane: U32,base: U32,second_lane: U32,kernel_column: U32,unit: U32,window: Packing.Stencil,
    planes: Packing.Planes}

def pack_passes(state: Passes) -> Packing.Passes:
  Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes} = state
  Packing.Passes{Packing.Arrays{Tree.pack(Tile.Tile16,pairs),Tree.pack(Tile.Tile8,tiles)},pass,lane,base,second_lane,kernel_column,unit,window,planes}

def pass_done(+pass: U32,+lane: U32,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes,done: Passing) -> Passes:
  Passing{pairs,tiles,start,read,kernel_row,write,passing_lane,tap_rows,tap_columns,row_step} = done
  Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes}

# The first entry of a pass at the pass's lane and plane offset.
def pass_start(pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>,+lane: U32,+base: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes) ->
  Passing:
  Packing.Stencil{base_cell,height,width,+tap_rows,+tap_columns,+stride,row_origin,column_origin,output_width,+inner,groups} = window
  Packing.Planes{first_output_row,rows,+row_width,+elements} = planes
  +channels = Packing.channels_of(window)
  +read = (((kernel_column % stride) * elements + base / 8) * channels : U32)
  Passing{pairs,tiles,read,read,0,(unit * inner + kernel_column : U32),lane,tap_rows,tap_columns,(row_width / 8 * channels : U32)}

def pass_entries(+window: Packing.Stencil) -> Nat:
  Packing.Stencil{base,height,width,+tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  U32.to_nat((Packing.channels_of(window) * tap_rows : U32))

def pass_step(state: Passes) -> Passes:
  Passes{pairs,tiles,+pass,+lane,+base,+second_lane,+kernel_column,+unit,+window,+planes} = state
  Packing.Stencil{base_cell,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  Packing.Planes{first_output_row,rows,+row_width,elements} = planes
  +shift = (base % 8 : U32)
  pass_done((pass + 1 : U32),Bool.pick(U32,U32.is_zero(pass),second_lane,(lane + output_width : U32)),(base + stride * row_width - output_width : U32),second_lane,kernel_column,
    unit,window,planes,
    pass_shift(U32.is_lt(shift,4),U32.is_lt((shift % 4 : U32),2),U32.is_eq((shift % 2 : U32),1),U32.is_zero(pass),pass_entries(window),
      pass_start(pairs,tiles,lane,base,kernel_column,unit,window,planes)))

def pass_done_matches(+pass: U32,+lane: U32,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes,+done: Passing) ->
  {Packing.Passes{Packing.passed(pack_passing(done)),pass,lane,base,second_lane,kernel_column,unit,window,planes} ==
    pack_passes(pass_done(pass,lane,base,second_lane,kernel_column,unit,window,planes,done)) : Packing.Passes}:
  Passing{pairs,tiles,start,read,kernel_row,write,passing_lane,tap_rows,tap_columns,row_step} = done
  {==}

def pass_step_matches(+state: Passes) -> {Packing.pass_step(pack_passes(state)) == pack_passes(pass_step(state)) : Packing.Passes}:
  Passes{+pairs,+tiles,+pass,+lane,+base,+second_lane,+kernel_column,+unit,+window,+planes} = state
  Packing.Stencil{+base_cell,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +shift = (base % 8 : U32)
  +start = pass_start(pairs,tiles,lane,base,kernel_column,unit,window,planes)
  +half = U32.is_lt(shift,4)
  +low = U32.is_lt((shift % 4 : U32),2)
  +odd = U32.is_eq((shift % 2 : U32),1)
  %Equal.sym(Packing.Passing,Packing.pass_shift(half,low,odd,U32.is_zero(pass),pass_entries(window),pack_passing(start)),
    pack_passing(pass_shift(half,low,odd,U32.is_zero(pass),pass_entries(window),start)),pass_shift_matches(half,low,odd,U32.is_zero(pass),pass_entries(window),start)) :
    {Packing.Passes{Packing.passed(_),(pass + 1 : U32),Bool.pick(U32,U32.is_zero(pass),second_lane,(lane + output_width : U32)),(base + stride * row_width - output_width : U32),
        second_lane,kernel_column,unit,window,planes} ==
      pack_passes(pass_step(Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes})) : Packing.Passes}
  pass_done_matches((pass + 1 : U32),Bool.pick(U32,U32.is_zero(pass),second_lane,(lane + output_width : U32)),(base + stride * row_width - output_width : U32),second_lane,
    kernel_column,unit,window,planes,pass_shift(half,low,odd,U32.is_zero(pass),pass_entries(window),start))

type Arrays is Data:
  Arrays{pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>}

def pack_arrays(arrays: Arrays) -> Packing.Arrays:
  Arrays{pairs,tiles} = arrays
  Packing.Arrays{Tree.pack(Tile.Tile16,pairs),Tree.pack(Tile.Tile8,tiles)}

def arrays_pairs(arrays: Arrays) -> Tree.Tree<Tile.Tile16>:
  Arrays{pairs,tiles} = arrays
  pairs

def arrays_tiles(arrays: Arrays) -> Tree.Tree<Tile.Tile8>:
  Arrays{pairs,tiles} = arrays
  tiles

def passes_done(state: Passes) -> Arrays:
  Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes} = state
  Arrays{pairs,tiles}

def passes_done_matches(+state: Passes) -> {Packing.passes_done(pack_passes(state)) == pack_arrays(passes_done(state)) : Packing.Arrays}:
  Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes} = state
  {==}

# The passes of one kernel column of the unit at position.
def pass_count(+window: Packing.Stencil,+position: U32) -> Nat:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,+output_width,inner,groups} = window
  +second_lane = (output_width - position % output_width : U32)
  U32.to_nat(Bool.pick(U32,U32.is_le(8,second_lane),1,(1 + (8 - second_lane + output_width - 1) / output_width : U32)))

def first_base(+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+kernel_column: U32) -> U32:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  Packing.Planes{+first_output_row,rows,+row_width,elements} = planes
  ((position / output_width - first_output_row) * stride * row_width + position % output_width + kernel_column / stride : U32)

def column_passes(pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32) -> Passes:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,+output_width,inner,groups} = window
  Passes{pairs,tiles,0,0,first_base(window,planes,position,kernel_column),(output_width - position % output_width : U32),kernel_column,unit,window,planes}

def kernel_column_step(+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,arrays: Arrays) -> Arrays:
  Arrays{pairs,tiles} = arrays
  passes_done(Loop.run(~Passes,~pass_step,pass_count(window,position),column_passes(pairs,tiles,kernel_column,window,planes,position,unit)))

def kernel_column_matches(+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,+arrays: Arrays) ->
  {Packing.kernel_column_step(kernel_column,window,planes,position,unit,pack_arrays(arrays)) ==
    pack_arrays(kernel_column_step(kernel_column,window,planes,position,unit,arrays)) : Packing.Arrays}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  Arrays{+pairs,+tiles} = arrays
  +count = pass_count(window,position)
  +state = column_passes(pairs,tiles,kernel_column,window,planes,position,unit)
  %Equal.sym(Packing.Passes,Loop.run(~Packing.Passes,~Packing.pass_step,count,pack_passes(state)),pack_passes(Loop.run(~Passes,~pass_step,count,state)),
    Projection.iteration_refines_model(~Packing.Passes,~Passes,~Packing.pass_step,~pass_step,~pack_passes,~pass_step_matches,count,state)) :
    {Packing.passes_done(_) == pack_arrays(kernel_column_step(kernel_column,window,planes,position,unit,Arrays{pairs,tiles})) : Packing.Arrays}
  passes_done_matches(Loop.run(~Passes,~pass_step,count,state))

def kernel_columns(remaining: Nat,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,arrays: Arrays) -> Arrays:
  match remaining:
    case 0n: arrays
    case 1n+rest: kernel_columns(rest,(kernel_column + 1 : U32),window,planes,position,unit,kernel_column_step(kernel_column,window,planes,position,unit,arrays))

def kernel_columns_match(+remaining: Nat,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,+arrays: Arrays) ->
  {Packing.kernel_columns(remaining,kernel_column,window,planes,position,unit,pack_arrays(arrays)) ==
    pack_arrays(kernel_columns(remaining,kernel_column,window,planes,position,unit,arrays)) : Packing.Arrays}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Packing.Arrays,Packing.kernel_column_step(kernel_column,window,planes,position,unit,pack_arrays(arrays)),
        pack_arrays(kernel_column_step(kernel_column,window,planes,position,unit,arrays)),kernel_column_matches(kernel_column,window,planes,position,unit,arrays)) :
        {Packing.kernel_columns(rest,(kernel_column + 1 : U32),window,planes,position,unit,_) ==
          pack_arrays(kernel_columns(rest,(kernel_column + 1 : U32),window,planes,position,unit,kernel_column_step(kernel_column,window,planes,position,unit,arrays))) :
          Packing.Arrays}
      kernel_columns_match(rest,(kernel_column + 1 : U32),window,planes,position,unit,kernel_column_step(kernel_column,window,planes,position,unit,arrays))

def column_count(+window: Packing.Stencil) -> Nat:
  Packing.Stencil{base,height,width,tap_rows,+tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  U32.to_nat(tap_columns)

def units(remaining: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,arrays: Arrays) -> Arrays:
  match remaining:
    case 0n: arrays
    case 1n+rest: units(rest,window,planes,(position + 8 : U32),(unit + 1 : U32),kernel_columns(column_count(window),0,window,planes,position,unit,arrays))

def units_match(+remaining: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,+arrays: Arrays) ->
  {Packing.units(remaining,window,planes,position,unit,pack_arrays(arrays)) == pack_arrays(units(remaining,window,planes,position,unit,arrays)) : Packing.Arrays}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
      +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
      +next = kernel_columns(U32.to_nat(tap_columns),0,window,planes,position,unit,arrays)
      %Equal.sym(Packing.Arrays,Packing.kernel_columns(U32.to_nat(tap_columns),0,window,planes,position,unit,pack_arrays(arrays)),pack_arrays(next),
        kernel_columns_match(U32.to_nat(tap_columns),0,window,planes,position,unit,arrays)) :
        {Packing.units(rest,window,planes,(position + 8 : U32),(unit + 1 : U32),_) == pack_arrays(units(rest,window,planes,(position + 8 : U32),(unit + 1 : U32),next)) :
          Packing.Arrays}
      units_match(rest,window,planes,(position + 8 : U32),(unit + 1 : U32),next)

def unit_count(+window: Packing.Stencil,+count: Nat) -> Nat:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,+inner,groups} = window
  Nat.div(count,U32.to_nat(inner))

# The tiles of the pack.
def packed_tiles(+window: Packing.Stencil,+planes: Packing.Planes,+count: Nat,+start: U32,pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  arrays_tiles(units(unit_count(window,count),window,planes,start,0,Arrays{pairs,tiles}))

def packed_done(+arrays: Arrays) ->
  {Packing.packed(pack_arrays(arrays)) == (Tree.pack(Tile.Tile16,arrays_pairs(arrays)),Tree.pack(Tile.Tile8,arrays_tiles(arrays))) : Array<Tile.Tile16> & Array<Tile.Tile8>}:
  Arrays{pairs,tiles} = arrays
  {==}

# ---- The pack hands its pairs back ----

def passing_pairs(state: Passing) -> Tree.Tree<Tile.Tile16>:
  Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step} = state
  pairs

def first_keeps_pairs(~shift: Tile.Tile16 -> Tile.Tile8,+state: Passing) -> {passing_pairs(first_step(~shift,state)) == passing_pairs(state) : Tree.Tree<Tile.Tile16>}:
  Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step} = state
  {==}

def merge_keeps_pairs(~shift: Tile.Tile16 -> Tile.Tile8,+state: Passing) -> {passing_pairs(merge_step(~shift,state)) == passing_pairs(state) : Tree.Tree<Tile.Tile16>}:
  Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step} = state
  {==}

def pass_run_pairs(~shift: Tile.Tile16 -> Tile.Tile8,+first: Bool,+count: Nat,+state: Passing) ->
  {passing_pairs(pass_run(~shift,first,count,state)) == passing_pairs(state) : Tree.Tree<Tile.Tile16>}:
  match first:
    case True{}: Projection.preserves_observation(~Passing,~Tree.Tree<Tile.Tile16>,~first_step(~shift),~passing_pairs,~first_keeps_pairs(~shift),count,state)
    case False{}: Projection.preserves_observation(~Passing,~Tree.Tree<Tile.Tile16>,~merge_step(~shift),~passing_pairs,~merge_keeps_pairs(~shift),count,state)

def pass_low_pairs(+low: Bool,+odd: Bool,+first: Bool,+count: Nat,+state: Passing) ->
  {passing_pairs(pass_low(low,odd,first,count,state)) == passing_pairs(state) : Tree.Tree<Tile.Tile16>}:
  match low odd:
    case True{} False{}: pass_run_pairs(~Packing.shifted0,first,count,state)
    case True{} True{}: pass_run_pairs(~Packing.shifted1,first,count,state)
    case False{} False{}: pass_run_pairs(~Packing.shifted2,first,count,state)
    case False{} True{}: pass_run_pairs(~Packing.shifted3,first,count,state)

def pass_high_pairs(+low: Bool,+odd: Bool,+first: Bool,+count: Nat,+state: Passing) ->
  {passing_pairs(pass_high(low,odd,first,count,state)) == passing_pairs(state) : Tree.Tree<Tile.Tile16>}:
  match low odd:
    case True{} False{}: pass_run_pairs(~Packing.shifted4,first,count,state)
    case True{} True{}: pass_run_pairs(~Packing.shifted5,first,count,state)
    case False{} False{}: pass_run_pairs(~Packing.shifted6,first,count,state)
    case False{} True{}: pass_run_pairs(~Packing.shifted7,first,count,state)

def pass_shift_pairs(+half: Bool,+low: Bool,+odd: Bool,+first: Bool,+count: Nat,+state: Passing) ->
  {passing_pairs(pass_shift(half,low,odd,first,count,state)) == passing_pairs(state) : Tree.Tree<Tile.Tile16>}:
  match half:
    case True{}: pass_low_pairs(low,odd,first,count,state)
    case False{}: pass_high_pairs(low,odd,first,count,state)

def passes_pairs(state: Passes) -> Tree.Tree<Tile.Tile16>:
  Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes} = state
  pairs

def pass_done_pairs(+pass: U32,+lane: U32,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes,+done: Passing) ->
  {passes_pairs(pass_done(pass,lane,base,second_lane,kernel_column,unit,window,planes,done)) == passing_pairs(done) : Tree.Tree<Tile.Tile16>}:
  Passing{pairs,tiles,start,read,kernel_row,write,passing_lane,tap_rows,tap_columns,row_step} = done
  {==}

def pass_keeps_pairs(+state: Passes) -> {passes_pairs(pass_step(state)) == passes_pairs(state) : Tree.Tree<Tile.Tile16>}:
  Passes{+pairs,+tiles,+pass,+lane,+base,+second_lane,+kernel_column,+unit,+window,+planes} = state
  Packing.Stencil{+base_cell,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +shift = (base % 8 : U32)
  +start = pass_start(pairs,tiles,lane,base,kernel_column,unit,window,planes)
  +done = pass_shift(U32.is_lt(shift,4),U32.is_lt((shift % 4 : U32),2),U32.is_eq((shift % 2 : U32),1),U32.is_zero(pass),pass_entries(window),start)
  +next_lane = Bool.pick(U32,U32.is_zero(pass),second_lane,(lane + output_width : U32))
  +next_base = (base + stride * row_width - output_width : U32)
  Equal.trans(Tree.Tree<Tile.Tile16>,passes_pairs(pass_done((pass + 1 : U32),next_lane,next_base,second_lane,kernel_column,unit,window,planes,done)),passing_pairs(done),pairs,
    pass_done_pairs((pass + 1 : U32),next_lane,next_base,second_lane,kernel_column,unit,window,planes,done),
    pass_shift_pairs(U32.is_lt(shift,4),U32.is_lt((shift % 4 : U32),2),U32.is_eq((shift % 2 : U32),1),U32.is_zero(pass),pass_entries(window),start))

def passes_done_pairs(+state: Passes) -> {arrays_pairs(passes_done(state)) == passes_pairs(state) : Tree.Tree<Tile.Tile16>}:
  Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes} = state
  {==}

def column_passes_pairs(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,
  +unit: U32) -> {passes_pairs(column_passes(pairs,tiles,kernel_column,window,planes,position,unit)) == pairs : Tree.Tree<Tile.Tile16>}:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  {==}

def kernel_column_pairs(+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,+arrays: Arrays) ->
  {arrays_pairs(kernel_column_step(kernel_column,window,planes,position,unit,arrays)) == arrays_pairs(arrays) : Tree.Tree<Tile.Tile16>}:
  Arrays{+pairs,+tiles} = arrays
  +state = column_passes(pairs,tiles,kernel_column,window,planes,position,unit)
  +last = Loop.run(~Passes,~pass_step,pass_count(window,position),state)
  Equal.trans(Tree.Tree<Tile.Tile16>,arrays_pairs(passes_done(last)),passes_pairs(last),pairs,passes_done_pairs(last),
    Equal.trans(Tree.Tree<Tile.Tile16>,passes_pairs(last),passes_pairs(state),pairs,
      Projection.preserves_observation(~Passes,~Tree.Tree<Tile.Tile16>,~pass_step,~passes_pairs,~pass_keeps_pairs,pass_count(window,position),state),
      column_passes_pairs(pairs,tiles,kernel_column,window,planes,position,unit)))

def kernel_columns_pairs(+remaining: Nat,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,+arrays: Arrays) ->
  {arrays_pairs(kernel_columns(remaining,kernel_column,window,planes,position,unit,arrays)) == arrays_pairs(arrays) : Tree.Tree<Tile.Tile16>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +next = kernel_column_step(kernel_column,window,planes,position,unit,arrays)
      Equal.trans(Tree.Tree<Tile.Tile16>,arrays_pairs(kernel_columns(rest,(kernel_column + 1 : U32),window,planes,position,unit,next)),arrays_pairs(next),arrays_pairs(arrays),
        kernel_columns_pairs(rest,(kernel_column + 1 : U32),window,planes,position,unit,next),kernel_column_pairs(kernel_column,window,planes,position,unit,arrays))

def units_pairs(+remaining: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,+arrays: Arrays) ->
  {arrays_pairs(units(remaining,window,planes,position,unit,arrays)) == arrays_pairs(arrays) : Tree.Tree<Tile.Tile16>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +next = kernel_columns(column_count(window),0,window,planes,position,unit,arrays)
      Equal.trans(Tree.Tree<Tile.Tile16>,arrays_pairs(units(rest,window,planes,(position + 8 : U32),(unit + 1 : U32),next)),arrays_pairs(next),arrays_pairs(arrays),
        units_pairs(rest,window,planes,(position + 8 : U32),(unit + 1 : U32),next),kernel_columns_pairs(column_count(window),0,window,planes,position,unit,arrays))

# The pack on packed trees hands back its pairs and the packed tiles of the
# model.
law pack_matches:
  for +window: Packing.Stencil
  for +planes: Packing.Planes
  for +count: Nat
  for +start: U32
  for +pairs: Tree.Tree<Tile.Tile16>
  for +tiles: Tree.Tree<Tile.Tile8>
  {Packing.pack(window,planes,count,start,Tree.pack(Tile.Tile16,pairs),Tree.pack(Tile.Tile8,tiles)) ==
    (Tree.pack(Tile.Tile16,pairs),Tree.pack(Tile.Tile8,packed_tiles(window,planes,count,start,pairs,tiles))) : Array<Tile.Tile16> & Array<Tile.Tile8>}
def pack_matches(window,planes,count,start,pairs,tiles):
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +arrays = {Arrays{pairs,tiles} : Arrays}
  +result = units(unit_count(window,count),window,planes,start,0,arrays)
  %Equal.sym(Packing.Arrays,Packing.units(unit_count(window,count),window,planes,start,0,pack_arrays(arrays)),pack_arrays(result),
    units_match(unit_count(window,count),window,planes,start,0,arrays)) :
    {Packing.packed(_) == (Tree.pack(Tile.Tile16,pairs),Tree.pack(Tile.Tile8,packed_tiles(window,planes,count,start,pairs,tiles))) : Array<Tile.Tile16> & Array<Tile.Tile8>}
  %units_pairs(unit_count(window,count),window,planes,start,0,arrays) :
    {Packing.packed(pack_arrays(result)) == (Tree.pack(Tile.Tile16,_),Tree.pack(Tile.Tile8,arrays_tiles(result))) : Array<Tile.Tile16> & Array<Tile.Tile8>}
  packed_done(result)
