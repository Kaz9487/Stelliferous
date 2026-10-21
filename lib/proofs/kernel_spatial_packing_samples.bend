# Storage-tree samples for the spatial packing readers.
import Base
import ../math_sequence.bend as Sequence
import ../kernel_packing_samples.bend as PackingSamples
import ../kernel_tile.bend as Tile
import ../kernel_spatial_packing.bend as Packing
import ./kernel_tile_refinement.bend as TileLaws
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./bounded_u32_arithmetic.bend as Arithmetic

def value(+x: Tree.Tree<F32>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32) -> F32:
  PackingSamples.masked(Tree.read(F32,x,(cb + iy * w + ix : U32)),U32.is_lt(iy,h) && U32.is_lt(ix,w))

def pixel(+x: Tree.Tree<F32>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32) -> {Packing.pixel(Tree.pack(F32,x),iy,ix,cb,h,
    w) == (Tree.pack(F32,x),value(x,iy,ix,cb,h,w)) : Array<F32> & F32}:
  %Equal.sym(Array<F32> & F32,Array.get(F32,Tree.pack(F32,x),(cb + iy * w + ix : U32)),(Tree.pack(F32,x),Tree.read(F32,x,
        (cb + iy * w + ix : U32))),StorageLaws.read(F32,x,(cb + iy * w + ix : U32))) : {Packing.masked(U32.is_lt(iy,h) && U32.is_lt(ix,
        w),_) == (Tree.pack(F32,x),value(x,iy,ix,cb,h,w)) : Array<F32> & F32}
  {==}

def pack_input(input: Tree.Tree<F32>) -> Array<F32>:
  Tree.pack(F32,input)

def block_sample(input: Tree.Tree<F32>,context: Packing.PixelBlock,index: Nat) -> F32:
  Packing.PixelBlock{row,column,channel_base,height,width,+stride} = context
  value(input,row,(column + U32.from_nat(index) * stride : U32),channel_base,height,width)

def native_block_read(+input: Tree.Tree<F32>,+context: Packing.PixelBlock,+index: Nat) -> {Packing.read_block_pixel(pack_input(input),
    context,index) == (pack_input(input),block_sample(input,context,index)) : Array<F32> & F32}:
  Packing.PixelBlock{row,column,channel_base,height,width,+stride} = context
  pixel(input,row,(column + U32.from_nat(index) * stride : U32),channel_base,height,width)

def native_vector(input: Array<F32>,context: Packing.PixelBlock) -> Array<F32> & Tile.Tile8:
  Packing.PixelBlock{+row,+column,+channel_base,+height,+width,+stride} = context
  Tile.read_tail(~Array<F32>,~Packing.PixelBlock,~Packing.read_block_pixel,
    Packing.pixel(input,row,column,channel_base,height,width),context)

def pure_vector(+input: Tree.Tree<F32>,+context: Packing.PixelBlock) -> Tile.Tile8:
  Tile.from_sequence(Sequence.tabulate(~F32,~TileLaws.SampleContext<Tree.Tree<F32>,Packing.PixelBlock>,
    ~TileLaws.sample_context(~Tree.Tree<F32>,~Packing.PixelBlock,~block_sample),8n,0n,TileLaws.SampleContext{input,context}))

def zero_lane_column(+column: U32,+stride: U32) -> {(column + 0 * stride : U32) == column : U32}:
  %Equal.sym(U32,(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)) : {(column + _ : U32) == column : U32}
  Arithmetic.add_zero(column)

def native_vector_matches_tabulation(+input: Tree.Tree<F32>,+context: Packing.PixelBlock) ->
  {native_vector(Tree.pack(F32,input),context) == (Tree.pack(F32,input),pure_vector(input,context)) : Array<F32> & Tile.Tile8}:
  Packing.PixelBlock{+row,+column,+channel_base,+height,+width,+stride} = context
  %Equal.sym(Array<F32> & F32,Packing.pixel(Tree.pack(F32,input),row,column,channel_base,height,width),
    (Tree.pack(F32,input),value(input,row,column,channel_base,height,width)),pixel(input,row,column,channel_base,height,width)) :
    {Tile.read_tail(~Array<F32>,~Packing.PixelBlock,~Packing.read_block_pixel,_,context) == (Tree.pack(F32,input),pure_vector(input,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(U32,(column + 0 * stride : U32),column,zero_lane_column(column,stride)) :
    {Tile.read_tail(~Array<F32>,~Packing.PixelBlock,~Packing.read_block_pixel,(Tree.pack(F32,input),value(input,row,column,channel_base,height,width)),context) ==
      (Tree.pack(F32,input),Tile.from_sequence(Sequence.Elements{value(input,row,_,channel_base,height,width),
        Sequence.tabulate(~F32,~TileLaws.SampleContext<Tree.Tree<F32>,Packing.PixelBlock>,~TileLaws.sample_context(~Tree.Tree<F32>,~Packing.PixelBlock,~block_sample),7n,1n,TileLaws.SampleContext{input,context})})) : Array<F32> & Tile.Tile8}
  TileLaws.tail_block_refines(~Array<F32>,~Tree.Tree<F32>,~Packing.PixelBlock,~pack_input,~Packing.read_block_pixel,
    ~block_sample,~native_block_read,input,context,value(input,row,column,channel_base,height,width))

# A lane of a tile that may end in the next output row.
def wrap_value(same: Bool,+x: Tree.Tree<F32>,+row: U32,+column: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32,+lane: U32) -> F32:
  match same:
    case True{}: value(x,row,(column + lane * s : U32),cb,h,w)
    case False{}: value(x,next_row,((offset + lane - ow) * s + kc - p : U32),cb,h,w)

def wrap_pixel(+same: Bool,+x: Tree.Tree<F32>,+row: U32,+column: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32,+lane: U32) ->
  {Packing.wrap_pixel(same,Tree.pack(F32,x),row,column,cb,h,w,s,next_row,offset,ow,kc,p,lane) == (Tree.pack(F32,x),wrap_value(same,x,row,column,cb,h,w,s,next_row,offset,ow,kc,p,lane)) : Array<F32> & F32}:
  match same:
    case True{}: pixel(x,row,(column + lane * s : U32),cb,h,w)
    case False{}: pixel(x,next_row,((offset + lane - ow) * s + kc - p : U32),cb,h,w)

def wrap_lane(+x: Tree.Tree<F32>,+row: U32,+column: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32,+lane: U32) -> F32:
  wrap_value(U32.is_lt((offset + lane : U32),ow),x,row,column,cb,h,w,s,next_row,offset,ow,kc,p,lane)

def wrap_sample(input: Tree.Tree<F32>,context: Packing.WrapBlock,index: Nat) -> F32:
  Packing.WrapBlock{row,column,cb,h,w,s,next_row,offset,ow,kc,p} = context
  wrap_lane(input,row,column,cb,h,w,s,next_row,offset,ow,kc,p,U32.from_nat(index))

def native_wrap_read(+input: Tree.Tree<F32>,+context: Packing.WrapBlock,+index: Nat) -> {Packing.read_wrap_pixel(pack_input(input),
    context,index) == (pack_input(input),wrap_sample(input,context,index)) : Array<F32> & F32}:
  Packing.WrapBlock{+row,+column,+cb,+h,+w,+s,+next_row,+offset,+ow,+kc,+p} = context
  wrap_pixel(U32.is_lt((offset + U32.from_nat(index) : U32),ow),input,row,column,cb,h,w,s,next_row,offset,ow,kc,p,U32.from_nat(index))

# Lane 0 lies in the output row of the tile.
def pure_wrap_vector(+x: Tree.Tree<F32>,+row: U32,+column: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32) -> Tile.Tile8:
  Tile.from_sequence(Sequence.Elements{value(x,row,column,cb,h,w),
    Sequence.tabulate(~F32,~TileLaws.SampleContext<Tree.Tree<F32>,Packing.WrapBlock>,~TileLaws.sample_context(~Tree.Tree<F32>,~Packing.WrapBlock,~wrap_sample),7n,1n,
      TileLaws.SampleContext{x,Packing.WrapBlock{row,column,cb,h,w,s,next_row,offset,ow,kc,p}})})

def native_wrap_vector_matches_tabulation(+x: Tree.Tree<F32>,+row: U32,+column: U32,+cb: U32,+h: U32,+w: U32,+s: U32,+next_row: U32,+offset: U32,+ow: U32,+kc: U32,+p: U32) ->
  {Packing.wrap_block(Tree.pack(F32,x),row,column,cb,h,w,s,next_row,offset,ow,kc,p) == (Tree.pack(F32,x),pure_wrap_vector(x,row,column,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Packing.pixel(Tree.pack(F32,x),row,column,cb,h,w),(Tree.pack(F32,x),value(x,row,column,cb,h,w)),pixel(x,row,column,cb,h,w)) :
    {Tile.read_tail(~Array<F32>,~Packing.WrapBlock,~Packing.read_wrap_pixel,_,Packing.WrapBlock{row,column,cb,h,w,s,next_row,offset,ow,kc,p}) ==
      (Tree.pack(F32,x),pure_wrap_vector(x,row,column,cb,h,w,s,next_row,offset,ow,kc,p)) : Array<F32> & Tile.Tile8}
  TileLaws.tail_block_refines(~Array<F32>,~Tree.Tree<F32>,~Packing.WrapBlock,~pack_input,~Packing.read_wrap_pixel,
    ~wrap_sample,~native_wrap_read,x,Packing.WrapBlock{row,column,cb,h,w,s,next_row,offset,ow,kc,p},value(x,row,column,cb,h,w))

def address_sample(input: Tree.Tree<F32>,address: PackingSamples.PixelAddress) -> F32:
  PackingSamples.PixelAddress{index,inside} = address
  PackingSamples.masked(Tree.read(F32,input,index),inside)

def native_address_read(+input: Tree.Tree<F32>,+address: PackingSamples.PixelAddress) ->
  {PackingSamples.read_pixel_address(pack_input(input),address) ==
    (pack_input(input),address_sample(input,address)) : Array<F32> & F32}:
  PackingSamples.PixelAddress{+index,+inside} = address
  %Equal.sym(Array<F32> & F32,Array.get(F32,pack_input(input),index),
    (pack_input(input),Tree.read(F32,input,index)),StorageLaws.read(F32,input,index)) :
    {PackingSamples.mask_pixel(inside,_) == (pack_input(input),PackingSamples.masked(Tree.read(F32,input,index),inside)) : Array<F32> & F32}
  {==}

def window_sample(input: Tree.Tree<F32>,context: PackingSamples.WindowPixelBlock,index: Nat) -> F32:
  address_sample(input,PackingSamples.window_pixel_address(context,index))

def native_window_read(+input: Tree.Tree<F32>,+context: PackingSamples.WindowPixelBlock,+index: Nat) ->
  {PackingSamples.read_window_pixel(pack_input(input),context,index) ==
    (pack_input(input),window_sample(input,context,index)) : Array<F32> & F32}:
  native_address_read(input,PackingSamples.window_pixel_address(context,index))

def window_context(+window: Packing.Stencil,position: U32,reduction: U32) -> PackingSamples.WindowPixelBlock:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner} = window
  PackingSamples.window_pixel_block(position,reduction,base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width)

def window_value(input: Tree.Tree<F32>,window: Packing.Stencil,position: U32,reduction: U32) -> F32:
  window_sample(input,window_context(window,position,reduction),0n)

def window_vector(+input: Tree.Tree<F32>,+window: Packing.Stencil,+position: U32,+reduction: U32) -> Tile.Tile8:
  Tile.from_sequence(Sequence.Elements{window_value(input,window,position,reduction),
    Sequence.tabulate(~F32,~TileLaws.SampleContext<Tree.Tree<F32>,PackingSamples.WindowPixelBlock>,
      ~TileLaws.sample_context(~Tree.Tree<F32>,~PackingSamples.WindowPixelBlock,~window_sample),7n,1n,
      TileLaws.SampleContext{input,window_context(window,position,reduction)})})

def native_window_vector(input: Array<F32>,+window: Packing.Stencil,+position: U32,+reduction: U32) -> Array<F32> & Tile.Tile8:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner} = window
  Tile.read_tail(~Array<F32>,~PackingSamples.WindowPixelBlock,~PackingSamples.read_window_pixel,
    PackingSamples.window_pixel(input,position,reduction,base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width),
    window_context(window,position,reduction))

def window_vector_matches(+input: Tree.Tree<F32>,+window: Packing.Stencil,+position: U32,+reduction: U32) ->
  {native_window_vector(pack_input(input),window,position,reduction) == (pack_input(input),window_vector(input,window,position,reduction)) : Array<F32> & Tile.Tile8}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,inner} = window
  %Equal.sym(Array<F32> & F32,PackingSamples.window_pixel(pack_input(input),position,reduction,base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width),
    (pack_input(input),window_value(input,window,position,reduction)),native_window_read(input,window_context(window,position,reduction),0n)) :
    {Tile.read_tail(~Array<F32>,~PackingSamples.WindowPixelBlock,~PackingSamples.read_window_pixel,_,window_context(window,position,reduction)) ==
      (pack_input(input),window_vector(input,window,position,reduction)) : Array<F32> & Tile.Tile8}
  TileLaws.tail_block_refines(~Array<F32>,~Tree.Tree<F32>,~PackingSamples.WindowPixelBlock,~pack_input,~PackingSamples.read_window_pixel,
    ~window_sample,~native_window_read,input,window_context(window,position,reduction),window_value(input,window,position,reduction))

# Scalar projection of the canonical window block, with lane zero's position unchanged.
def window_scalar_context(input: Tree.Tree<F32>,context: PackingSamples.WindowPixelBlock) -> F32:
  PackingSamples.WindowPixelBlock{+position,kernel_row,kernel_column,channel_base,height,width,+stride,row_origin,column_origin,+output_width} = context
  value(input,(position / output_width * stride + kernel_row + row_origin : U32),
    (position % output_width * stride + kernel_column + column_origin : U32),channel_base,height,width)

def window_scalar(input: Tree.Tree<F32>,window: Packing.Stencil,position: U32,reduction: U32) -> F32:
  window_scalar_context(input,window_context(window,position,reduction))

def window_scalar_context_matches(+input: Tree.Tree<F32>,context: PackingSamples.WindowPixelBlock) ->
  {window_sample(input,context,0n) == window_scalar_context(input,context) : F32}:
  PackingSamples.WindowPixelBlock{+position,+kernel_row,+kernel_column,+channel_base,+height,+width,+stride,+row_origin,+column_origin,+output_width} = context
  Equal.cong(U32,F32,at => value(input,(at / output_width * stride + kernel_row + row_origin : U32),
    (at % output_width * stride + kernel_column + column_origin : U32),channel_base,height,width),
    (position + 0 : U32),position,Arithmetic.add_zero(position))

def window_scalar_matches(+input: Tree.Tree<F32>,+window: Packing.Stencil,+position: U32,+reduction: U32) ->
  {window_value(input,window,position,reduction) == window_scalar(input,window,position,reduction) : F32}:
  window_scalar_context_matches(input,window_context(window,position,reduction))
