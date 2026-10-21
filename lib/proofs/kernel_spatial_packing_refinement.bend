# Spatial packing on storage trees preserves the input and stores reader vectors.
import Base
import ../traversal_loop.bend as Loop
import ../kernel_spatial_packing.bend as Packing
import ../kernel_tile.bend as Tile
import ./loop_projection.bend as Projection
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./kernel_spatial_packing_samples.bend as Reads

def RowTiles() -> Data:
  Packing.RowState<&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>>

def Crossings() -> Data:
  Packing.CrossingState<&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>>

def Tiled() -> Data:
  Packing.TileState<&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>>

def pack_row(state: RowTiles()) -> Packing.RowTiles():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  Packing.RowTile{Tree.pack(F32,x),Tree.pack(Tile.Tile8,y),i,c,kr,kc,row,offset,window,tap_rows,tap_columns}

def pack_crossing(state: Crossings()) -> Packing.Crossings():
  Packing.Crossing{x,y,i,q,position,window} = state
  Packing.Crossing{Tree.pack(F32,x),Tree.pack(Tile.Tile8,y),i,q,position,window}

def pack_tiles(state: Tiled()) -> Packing.Tiled():
  Packing.Tiles{x,y,i,position,window,channels,tap_rows,tap_columns,inner} = state
  Packing.Tiles{Tree.pack(F32,x),Tree.pack(Tile.Tile8,y),i,position,window,channels,tap_rows,tap_columns,inner}

# The eight lanes of one reduction entry: of one output row, or of an output row and the next.
def block(+x: Tree.Tree<F32>,+row: U32,+offset: U32,+c: U32,+kr: U32,+kc: U32,+h: U32,+w: U32,+s: U32,+p: U32) -> Tile.Tile8:
  Reads.pure_vector(x,Packing.PixelBlock{(row * s + kr - p : U32),(offset * s + kc - p : U32),(c * h * w : U32),h,w,s})

def Lanes() -> Type:
  Tree.Tree<F32> -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> Tile.Tile8

def row_lanes(x: Tree.Tree<F32>,iy: U32,ix: U32,cb: U32,h: U32,w: U32,s: U32,next_row: U32,offset: U32,ow: U32,kc: U32,p: U32) -> Tile.Tile8:
  Reads.pure_vector(x,Packing.PixelBlock{iy,ix,cb,h,w,s})

def wrap_lanes(x: Tree.Tree<F32>,iy: U32,ix: U32,cb: U32,h: U32,w: U32,s: U32,next_row: U32,offset: U32,ow: U32,kc: U32,p: U32) -> Tile.Tile8:
  Reads.pure_wrap_vector(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)

def column_step(~lanes: Lanes(),state: RowTiles()) -> RowTiles():
  Packing.RowTile{+x,y,+i,+c,+kr,+kc,+row,+offset,+window,tap_rows,tap_columns} = state
  Packing.Stencil{+base,+h,+w,tap_height,tap_width,+s,+row_origin,+column_origin,+ow,inner_word} = window
  Packing.RowTile{x,Tree.write(Tile.Tile8,y,i,lanes(x,(row * s + kr + row_origin : U32),(offset * s + kc + column_origin : U32),(base + c * h * w : U32),h,w,s,((row + 1) * s + kr + row_origin : U32),offset,ow,kc,(0 - column_origin : U32))),(i + 1 : U32),c,kr,(kc + 1 : U32),row,offset,window,tap_rows,tap_columns}

def next_kernel_row(state: RowTiles()) -> RowTiles():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  Packing.RowTile{x,y,i,c,(kr + 1 : U32),0,row,offset,window,tap_rows,tap_columns}

def kernel_row_step(~lanes: Lanes(),state: RowTiles()) -> RowTiles():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,+tap_rows,+tap_columns} = state
  next_kernel_row(Loop.run(~RowTiles(),~column_step(~lanes),tap_columns,Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns}))

def next_channel(state: RowTiles()) -> RowTiles():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  Packing.RowTile{x,y,i,(c + 1 : U32),0,0,row,offset,window,tap_rows,tap_columns}

def channel_step(~lanes: Lanes(),state: RowTiles()) -> RowTiles():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,+tap_rows,+tap_columns} = state
  next_channel(Loop.run(~RowTiles(),~kernel_row_step(~lanes),tap_rows,Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns}))

def crossing_step(state: Crossings()) -> Crossings():
  Packing.Crossing{+x,y,+i,+q,+position,+window} = state
  Packing.Crossing{x,Tree.write(Tile.Tile8,y,i,Reads.window_vector(x,window,position,q)),(i + 1 : U32),(q + 1 : U32),position,window}

def after_row_tile(position: U32,channels: Nat,inner: Nat,state: RowTiles()) -> Tiled():
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  Packing.Tiles{x,y,i,(position + 8 : U32),window,channels,tap_rows,tap_columns,inner}

def after_crossing(channels: Nat,tap_rows: Nat,tap_columns: Nat,inner: Nat,state: Crossings()) -> Tiled():
  Packing.Crossing{x,y,i,q,position,window} = state
  Packing.Tiles{x,y,i,(position + 8 : U32),window,channels,tap_rows,tap_columns,inner}

def tile(within: Bool,wraps: Bool,x: Tree.Tree<F32>,y: Tree.Tree<Tile.Tile8>,i: U32,+position: U32,+window: Packing.Stencil,
  +channels: Nat,tap_rows: Nat,tap_columns: Nat,+inner: Nat) -> Tiled():
  match within wraps:
    case True{} wraps:
      Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
      after_row_tile(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),window,tap_rows,tap_columns}))
    case False{} True{}:
      Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
      after_row_tile(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),window,tap_rows,tap_columns}))
    case False{} False{}:
      Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
      after_crossing(channels,tap_rows,tap_columns,inner,Loop.run(~Crossings(),~crossing_step,inner,Packing.Crossing{x,y,i,0,position,window}))

def tile_step(state: Tiled()) -> Tiled():
  Packing.Tiles{x,y,i,+position,+window,channels,tap_rows,tap_columns,inner} = state
  Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
  tile(U32.is_le((position % ow + 8 : U32),ow),U32.is_le((position % ow + 8 : U32),(ow + ow : U32)),x,y,i,position,window,channels,tap_rows,tap_columns,inner)

# Each native reader on a packed tree returns the lanes of the model.
def row_agree(+x: Tree.Tree<F32>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32) ->
  {Packing.block(Tree.pack(F32,x),iy,ix,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),row_lanes(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8}:
  Reads.native_vector_matches_tabulation(x,Packing.PixelBlock{iy,ix,cb,h,w,s})

def wrap_agree(+x: Tree.Tree<F32>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32) ->
  {Packing.wrap_block(Tree.pack(F32,x),iy,ix,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),wrap_lanes(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8}:
  Reads.native_wrap_vector_matches_tabulation(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)

def column_refines(~block: Packing.Block(),~lanes: Lanes(),
  ~agree: @+x: Tree.Tree<F32> -> @+iy: U32 -> @+ix: U32 -> @+cb: U32 -> @+h: U32 -> @+w: U32 -> @+s: U32 -> @+next_row: U32 -> @+offset: U32 -> @+ow: U32 ->
    @+kc: U32 -> @+p: U32 -> {block(Tree.pack(F32,x),iy,ix,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),lanes(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8},
  +state: RowTiles()) -> {Packing.column_step(~block,pack_row(state)) == pack_row(column_step(~lanes,state)) : Packing.RowTiles()}:
  Packing.RowTile{+x,+y,+i,+c,+kr,+kc,+row,+offset,+window,+tap_rows,+tap_columns} = state
  Packing.Stencil{+base,+h,+w,tap_height,tap_width,+s,+row_origin,+column_origin,+ow,inner_word} = window
  +value = lanes(x,(row * s + kr + row_origin : U32),(offset * s + kc + column_origin : U32),(base + c * h * w : U32),h,w,s,((row + 1) * s + kr + row_origin : U32),offset,ow,kc,(0 - column_origin : U32))
  %Equal.sym(Array<F32> & Tile.Tile8,block(Tree.pack(F32,x),(row * s + kr + row_origin : U32),(offset * s + kc + column_origin : U32),(base + c * h * w : U32),h,w,s,((row + 1) * s + kr + row_origin : U32),offset,ow,kc,(0 - column_origin : U32)),(Tree.pack(F32,x),value),
    agree(x,(row * s + kr + row_origin : U32),(offset * s + kc + column_origin : U32),(base + c * h * w : U32),h,w,s,((row + 1) * s + kr + row_origin : U32),offset,ow,kc,(0 - column_origin : U32))) :
    {Packing.store_block(Tree.pack(Tile.Tile8,y),i,c,kr,kc,row,offset,window,tap_rows,tap_columns,_) ==
      pack_row(Packing.RowTile{x,Tree.write(Tile.Tile8,y,i,value),(i + 1 : U32),c,kr,(kc + 1 : U32),row,offset,window,tap_rows,tap_columns}) : Packing.RowTiles()}
  %Equal.sym(Array<Tile.Tile8>,Array.set(Tile.Tile8,Tree.pack(Tile.Tile8,y),i,value),Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,y,i,value)),
    StorageLaws.write(Tile.Tile8,y,i,value)) :
    {Packing.RowTile{Tree.pack(F32,x),_,(i + 1 : U32),c,kr,(kc + 1 : U32),row,offset,window,tap_rows,tap_columns} ==
      pack_row(Packing.RowTile{x,Tree.write(Tile.Tile8,y,i,value),(i + 1 : U32),c,kr,(kc + 1 : U32),row,offset,window,tap_rows,tap_columns}) : Packing.RowTiles()}
  {==}

def next_kernel_row_refines(+state: RowTiles()) -> {Packing.next_kernel_row(pack_row(state)) == pack_row(next_kernel_row(state)) : Packing.RowTiles()}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  {==}

def kernel_row_refines(~block: Packing.Block(),~lanes: Lanes(),
  ~agree: @+x: Tree.Tree<F32> -> @+iy: U32 -> @+ix: U32 -> @+cb: U32 -> @+h: U32 -> @+w: U32 -> @+s: U32 -> @+next_row: U32 -> @+offset: U32 -> @+ow: U32 ->
    @+kc: U32 -> @+p: U32 -> {block(Tree.pack(F32,x),iy,ix,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),lanes(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8},
  +state: RowTiles()) -> {Packing.kernel_row_step(~block,pack_row(state)) == pack_row(kernel_row_step(~lanes,state)) : Packing.RowTiles()}:
  Packing.RowTile{+x,+y,+i,+c,+kr,+kc,+row,+offset,+window,+tap_rows,+tap_columns} = state
  +start = {Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} : RowTiles()}
  %Equal.sym(Packing.RowTiles(),Loop.run(~Packing.RowTiles(),~Packing.column_step(~block),tap_columns,pack_row(start)),pack_row(Loop.run(~RowTiles(),~column_step(~lanes),tap_columns,start)),
    Projection.iteration_refines_model(~Packing.RowTiles(),~RowTiles(),~Packing.column_step(~block),~column_step(~lanes),~pack_row,~column_refines(~block,~lanes,~agree),tap_columns,start)) :
    {Packing.next_kernel_row(_) == pack_row(next_kernel_row(Loop.run(~RowTiles(),~column_step(~lanes),tap_columns,start))) : Packing.RowTiles()}
  next_kernel_row_refines(Loop.run(~RowTiles(),~column_step(~lanes),tap_columns,start))

def next_channel_refines(+state: RowTiles()) -> {Packing.next_channel(pack_row(state)) == pack_row(next_channel(state)) : Packing.RowTiles()}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  {==}

def channel_refines(~block: Packing.Block(),~lanes: Lanes(),
  ~agree: @+x: Tree.Tree<F32> -> @+iy: U32 -> @+ix: U32 -> @+cb: U32 -> @+h: U32 -> @+w: U32 -> @+s: U32 -> @+next_row: U32 -> @+offset: U32 -> @+ow: U32 ->
    @+kc: U32 -> @+p: U32 -> {block(Tree.pack(F32,x),iy,ix,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),lanes(x,iy,ix,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8},
  +state: RowTiles()) -> {Packing.channel_step(~block,pack_row(state)) == pack_row(channel_step(~lanes,state)) : Packing.RowTiles()}:
  Packing.RowTile{+x,+y,+i,+c,+kr,+kc,+row,+offset,+window,+tap_rows,+tap_columns} = state
  +start = {Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} : RowTiles()}
  %Equal.sym(Packing.RowTiles(),Loop.run(~Packing.RowTiles(),~Packing.kernel_row_step(~block),tap_rows,pack_row(start)),pack_row(Loop.run(~RowTiles(),~kernel_row_step(~lanes),tap_rows,start)),
    Projection.iteration_refines_model(~Packing.RowTiles(),~RowTiles(),~Packing.kernel_row_step(~block),~kernel_row_step(~lanes),~pack_row,~kernel_row_refines(~block,~lanes,~agree),tap_rows,start)) :
    {Packing.next_channel(_) == pack_row(next_channel(Loop.run(~RowTiles(),~kernel_row_step(~lanes),tap_rows,start))) : Packing.RowTiles()}
  next_channel_refines(Loop.run(~RowTiles(),~kernel_row_step(~lanes),tap_rows,start))

# The eight lanes of a crossing tile, read natively, are its tabulated vector.
def crossing_refines(+state: Crossings()) -> {Packing.crossing_step(pack_crossing(state)) == pack_crossing(crossing_step(state)) : Packing.Crossings()}:
  Packing.Crossing{+x,+y,+i,+q,+position,+window} = state
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner} = window
  +value = Reads.window_vector(x,window,position,q)
  %Equal.sym(Array<F32> & Tile.Tile8,Reads.native_window_vector(Tree.pack(F32,x),window,position,q),(Tree.pack(F32,x),value),Reads.window_vector_matches(x,window,position,q)) :
    {Packing.store_crossing(Tree.pack(Tile.Tile8,y),i,q,position,window,_) ==
      pack_crossing(Packing.Crossing{x,Tree.write(Tile.Tile8,y,i,value),(i + 1 : U32),(q + 1 : U32),position,window}) : Packing.Crossings()}
  %Equal.sym(Array<Tile.Tile8>,Array.set(Tile.Tile8,Tree.pack(Tile.Tile8,y),i,value),Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,y,i,value)),StorageLaws.write(Tile.Tile8,y,i,value)) :
    {Packing.Crossing{Tree.pack(F32,x),_,(i + 1 : U32),(q + 1 : U32),position,window} ==
      pack_crossing(Packing.Crossing{x,Tree.write(Tile.Tile8,y,i,value),(i + 1 : U32),(q + 1 : U32),position,window}) : Packing.Crossings()}
  {==}

def after_row_tile_refines(+position: U32,+channels: Nat,+inner: Nat,+state: RowTiles()) ->
  {Packing.after_row_tile(position,channels,inner,pack_row(state)) == pack_tiles(after_row_tile(position,channels,inner,state)) : Packing.Tiled()}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  {==}

def after_crossing_refines(+channels: Nat,+tap_rows: Nat,+tap_columns: Nat,+inner: Nat,+state: Crossings()) ->
  {Packing.after_crossing(channels,tap_rows,tap_columns,inner,pack_crossing(state)) == pack_tiles(after_crossing(channels,tap_rows,tap_columns,inner,state)) : Packing.Tiled()}:
  Packing.Crossing{x,y,i,q,position,window} = state
  {==}

def tile_refines(+within: Bool,+wraps: Bool,+x: Tree.Tree<F32>,+y: Tree.Tree<Tile.Tile8>,+i: U32,+position: U32,+window: Packing.Stencil,
  +channels: Nat,+tap_rows: Nat,+tap_columns: Nat,+inner: Nat) ->
  {Packing.tile(within,wraps,Tree.pack(F32,x),Tree.pack(Tile.Tile8,y),i,position,window,channels,tap_rows,tap_columns,inner) ==
    pack_tiles(tile(within,wraps,x,y,i,position,window,channels,tap_rows,tap_columns,inner)) : Packing.Tiled()}:
  match within wraps:
    case True{} wraps:
      Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
      +start = {Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),window,tap_rows,tap_columns} : RowTiles()}
      %Equal.sym(Packing.RowTiles(),Loop.run(~Packing.RowTiles(),~Packing.channel_step(~Packing.block),channels,pack_row(start)),pack_row(Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,start)),
        Projection.iteration_refines_model(~Packing.RowTiles(),~RowTiles(),~Packing.channel_step(~Packing.block),~channel_step(~row_lanes),~pack_row,~channel_refines(~Packing.block,~row_lanes,~row_agree),channels,start)) :
        {Packing.after_row_tile(position,channels,inner,_) == pack_tiles(after_row_tile(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,start))) : Packing.Tiled()}
      after_row_tile_refines(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,start))
    case False{} True{}:
      Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
      +start = {Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),window,tap_rows,tap_columns} : RowTiles()}
      %Equal.sym(Packing.RowTiles(),Loop.run(~Packing.RowTiles(),~Packing.channel_step(~Packing.wrap_block),channels,pack_row(start)),pack_row(Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,start)),
        Projection.iteration_refines_model(~Packing.RowTiles(),~RowTiles(),~Packing.channel_step(~Packing.wrap_block),~channel_step(~wrap_lanes),~pack_row,~channel_refines(~Packing.wrap_block,~wrap_lanes,~wrap_agree),channels,start)) :
        {Packing.after_row_tile(position,channels,inner,_) == pack_tiles(after_row_tile(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,start))) : Packing.Tiled()}
      after_row_tile_refines(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,start))
    case False{} False{}:
      Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
      +start = {Packing.Crossing{x,y,i,0,position,window} : Crossings()}
      %Equal.sym(Packing.Crossings(),Loop.run(~Packing.Crossings(),~Packing.crossing_step,inner,pack_crossing(start)),pack_crossing(Loop.run(~Crossings(),~crossing_step,inner,start)),
        Projection.iteration_refines_model(~Packing.Crossings(),~Crossings(),~Packing.crossing_step,~crossing_step,~pack_crossing,~crossing_refines,inner,start)) :
        {Packing.after_crossing(channels,tap_rows,tap_columns,inner,_) == pack_tiles(after_crossing(channels,tap_rows,tap_columns,inner,Loop.run(~Crossings(),~crossing_step,inner,start))) : Packing.Tiled()}
      after_crossing_refines(channels,tap_rows,tap_columns,inner,Loop.run(~Crossings(),~crossing_step,inner,start))

def tile_step_refines(+state: Tiled()) -> {Packing.tile_step(pack_tiles(state)) == pack_tiles(tile_step(state)) : Packing.Tiled()}:
  Packing.Tiles{+x,+y,+i,+position,+window,+channels,+tap_rows,+tap_columns,+inner} = state
  Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
  tile_refines(U32.is_le((position % ow + 8 : U32),ow),U32.is_le((position % ow + 8 : U32),(ow + ow : U32)),x,y,i,position,window,channels,tap_rows,tap_columns,inner)

def tiles_refine(+count: Nat,+state: Tiled()) ->
  {Loop.run(~Packing.Tiled(),~Packing.tile_step,count,pack_tiles(state)) == pack_tiles(Loop.run(~Tiled(),~tile_step,count,state)) : Packing.Tiled()}:
  Projection.iteration_refines_model(~Packing.Tiled(),~Tiled(),~Packing.tile_step,~tile_step,~pack_tiles,~tile_step_refines,count,state)

# Input observations through the packing loops.
def row_input(state: RowTiles()) -> Tree.Tree<F32>:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  x

def crossing_input(state: Crossings()) -> Tree.Tree<F32>:
  Packing.Crossing{x,y,i,q,position,window} = state
  x

def tiles_input(state: Tiled()) -> Tree.Tree<F32>:
  Packing.Tiles{x,y,i,position,window,channels,tap_rows,tap_columns,inner} = state
  x

def column_keeps_input(~lanes: Lanes(),+state: RowTiles()) -> {row_input(column_step(~lanes,state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,output_width,inner} = window
  {==}

def next_kernel_row_keeps_input(+state: RowTiles()) -> {row_input(next_kernel_row(state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  {==}

def kernel_row_keeps_input(~lanes: Lanes(),+state: RowTiles()) -> {row_input(kernel_row_step(~lanes,state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{+x,+y,+i,+c,+kr,+kc,+row,+offset,+window,+tap_rows,+tap_columns} = state
  +start = {Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} : RowTiles()}
  %Projection.preserves_observation(~RowTiles(),~Tree.Tree<F32>,~column_step(~lanes),~row_input,~column_keeps_input(~lanes),tap_columns,start) :
    {row_input(next_kernel_row(Loop.run(~RowTiles(),~column_step(~lanes),tap_columns,start))) == _ : Tree.Tree<F32>}
  next_kernel_row_keeps_input(Loop.run(~RowTiles(),~column_step(~lanes),tap_columns,start))

def next_channel_keeps_input(+state: RowTiles()) -> {row_input(next_channel(state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  {==}

def channel_keeps_input(~lanes: Lanes(),+state: RowTiles()) -> {row_input(channel_step(~lanes,state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{+x,+y,+i,+c,+kr,+kc,+row,+offset,+window,+tap_rows,+tap_columns} = state
  +start = {Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} : RowTiles()}
  %Projection.preserves_observation(~RowTiles(),~Tree.Tree<F32>,~kernel_row_step(~lanes),~row_input,~kernel_row_keeps_input(~lanes),tap_rows,start) :
    {row_input(next_channel(Loop.run(~RowTiles(),~kernel_row_step(~lanes),tap_rows,start))) == _ : Tree.Tree<F32>}
  next_channel_keeps_input(Loop.run(~RowTiles(),~kernel_row_step(~lanes),tap_rows,start))

def crossing_keeps_input(+state: Crossings()) -> {crossing_input(crossing_step(state)) == crossing_input(state) : Tree.Tree<F32>}:
  Packing.Crossing{x,y,i,q,position,window} = state
  {==}

def after_row_tile_input(+position: U32,+channels: Nat,+inner: Nat,+state: RowTiles()) ->
  {tiles_input(after_row_tile(position,channels,inner,state)) == row_input(state) : Tree.Tree<F32>}:
  Packing.RowTile{x,y,i,c,kr,kc,row,offset,window,tap_rows,tap_columns} = state
  {==}

def after_crossing_input(+channels: Nat,+tap_rows: Nat,+tap_columns: Nat,+inner: Nat,+state: Crossings()) ->
  {tiles_input(after_crossing(channels,tap_rows,tap_columns,inner,state)) == crossing_input(state) : Tree.Tree<F32>}:
  Packing.Crossing{x,y,i,q,position,window} = state
  {==}

def tile_keeps_input(+within: Bool,+wraps: Bool,+x: Tree.Tree<F32>,+y: Tree.Tree<Tile.Tile8>,+i: U32,+position: U32,+window: Packing.Stencil,
  +channels: Nat,+tap_rows: Nat,+tap_columns: Nat,+inner: Nat) -> {tiles_input(tile(within,wraps,x,y,i,position,window,channels,tap_rows,tap_columns,inner)) == x : Tree.Tree<F32>}:
  match within wraps:
    case True{} wraps:
      Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
      +start = {Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),window,tap_rows,tap_columns} : RowTiles()}
      %Projection.preserves_observation(~RowTiles(),~Tree.Tree<F32>,~channel_step(~row_lanes),~row_input,~channel_keeps_input(~row_lanes),channels,start) :
        {tiles_input(after_row_tile(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,start))) == _ : Tree.Tree<F32>}
      after_row_tile_input(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~row_lanes),channels,start))
    case False{} True{}:
      Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
      +start = {Packing.RowTile{x,y,i,0,0,0,(position / ow : U32),(position % ow : U32),window,tap_rows,tap_columns} : RowTiles()}
      %Projection.preserves_observation(~RowTiles(),~Tree.Tree<F32>,~channel_step(~wrap_lanes),~row_input,~channel_keeps_input(~wrap_lanes),channels,start) :
        {tiles_input(after_row_tile(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,start))) == _ : Tree.Tree<F32>}
      after_row_tile_input(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~wrap_lanes),channels,start))
    case False{} False{}:
      Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
      +start = {Packing.Crossing{x,y,i,0,position,window} : Crossings()}
      %Projection.preserves_observation(~Crossings(),~Tree.Tree<F32>,~crossing_step,~crossing_input,~crossing_keeps_input,inner,start) :
        {tiles_input(after_crossing(channels,tap_rows,tap_columns,inner,Loop.run(~Crossings(),~crossing_step,inner,start))) == _ : Tree.Tree<F32>}
      after_crossing_input(channels,tap_rows,tap_columns,inner,Loop.run(~Crossings(),~crossing_step,inner,start))

def tile_step_keeps_input(+state: Tiled()) -> {tiles_input(tile_step(state)) == tiles_input(state) : Tree.Tree<F32>}:
  Packing.Tiles{+x,+y,+i,+position,+window,+channels,+tap_rows,+tap_columns,+inner} = state
  Packing.Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+ow,inner_word} = window
  tile_keeps_input(U32.is_le((position % ow + 8 : U32),ow),U32.is_le((position % ow + 8 : U32),(ow + ow : U32)),x,y,i,position,window,channels,tap_rows,tap_columns,inner)

def tiles_keep_input(+count: Nat,+state: Tiled()) -> {tiles_input(Loop.run(~Tiled(),~tile_step,count,state)) == tiles_input(state) : Tree.Tree<F32>}:
  Projection.preserves_observation(~Tiled(),~Tree.Tree<F32>,~tile_step,~tiles_input,~tile_step_keeps_input,count,state)

type Packed is Data:
  Packed{input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>}

def pack_pair(packed: Packed) -> Array<F32> & Array<Tile.Tile8>:
  Packed{input,output} = packed
  (Tree.pack(F32,input),Tree.pack(Tile.Tile8,output))

def finish_tiles(state: Tiled()) -> Packed:
  Packing.Tiles{input,output,index,position,window,channels,tap_rows,tap_columns,inner} = state
  Packed{input,output}

def finish_matches(+state: Tiled()) -> {Packing.finish(pack_tiles(state)) == pack_pair(finish_tiles(state)) : Array<F32> & Array<Tile.Tile8>}:
  Packing.Tiles{input,output,index,position,window,channels,tap_rows,tap_columns,inner} = state
  {==}

def initial_window(+window: Packing.Stencil,start: U32,input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>) -> Tiled():
  Packing.Stencil{base,height,width,+tap_rows,+tap_columns,stride,row_origin,column_origin,output_width,+inner} = window
  Packing.Tiles{input,output,0,start,window,U32.to_nat((inner / (tap_rows * tap_columns) : U32)),U32.to_nat(tap_rows),U32.to_nat(tap_columns),U32.to_nat(inner)}

def packed_window(+window: Packing.Stencil,count: Nat,start: U32,input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>) -> Packed:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,+inner} = window
  finish_tiles(Loop.run(~Tiled(),~tile_step,Nat.div(count,U32.to_nat(inner)),initial_window(window,start,input,output)))

def packed_window_matches(+window: Packing.Stencil,+count: Nat,+start: U32,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>) ->
  {Packing.pack_window(window,count,start,Tree.pack(F32,input),Tree.pack(Tile.Tile8,output)) == pack_pair(packed_window(window,count,start,input,output)) : Array<F32> & Array<Tile.Tile8>}:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,+inner} = window
  +tiles = Nat.div(count,U32.to_nat(inner))
  +state = initial_window(window,start,input,output)
  %Equal.sym(Packing.Tiled(),Loop.run(~Packing.Tiled(),~Packing.tile_step,tiles,pack_tiles(state)),pack_tiles(Loop.run(~Tiled(),~tile_step,tiles,state)),tiles_refine(tiles,state)) :
    {Packing.finish(_) == pack_pair(finish_tiles(Loop.run(~Tiled(),~tile_step,tiles,state))) : Array<F32> & Array<Tile.Tile8>}
  finish_matches(Loop.run(~Tiled(),~tile_step,tiles,state))

def packed_input(packed: Packed) -> Tree.Tree<F32>:
  Packed{input,output} = packed
  input

def packed_output(packed: Packed) -> Tree.Tree<Tile.Tile8>:
  Packed{input,output} = packed
  output

def finish_input(+state: Tiled()) -> {packed_input(finish_tiles(state)) == tiles_input(state) : Tree.Tree<F32>}:
  Packing.Tiles{input,output,index,position,window,channels,tap_rows,tap_columns,inner} = state
  {==}

def packed_window_input(+window: Packing.Stencil,+count: Nat,+start: U32,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>) ->
  {packed_input(packed_window(window,count,start,input,output)) == input : Tree.Tree<F32>}:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,+inner} = window
  +tiles = Nat.div(count,U32.to_nat(inner))
  +state = initial_window(window,start,input,output)
  %tiles_keep_input(tiles,state) : {packed_input(finish_tiles(Loop.run(~Tiled(),~tile_step,tiles,state))) == _ : Tree.Tree<F32>}
  finish_input(Loop.run(~Tiled(),~tile_step,tiles,state))
