# Storage-tree samples for the spatial packing readers, for planes of values
# and for the cells of every position in turn.
import Base
import ../math_sequence.bend as Sequence
import ../kernel_packing_samples.bend as PackingSamples
import ../kernel_tile.bend as Tile
import ../kernel_spatial_packing.bend as Packing
import ./kernel_tile_refinement.bend as TileLaws
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./storage_shape.bend as Shape
import ./list_storage_readback.bend as ListRead

def value(~Element: Data,~field: Element -> F32,+x: Tree.Tree<Element>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32,+step: U32) -> F32:
  PackingSamples.masked(field(Tree.read(Element,x,(cb + (iy * w + ix) * step : U32))),U32.is_lt(iy,h) && U32.is_lt(ix,w))

# A sample at step 1 reads the plane at row and column.
def unit_step_value(~Element: Data,~field: Element -> F32,+x: Tree.Tree<Element>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32) ->
  {value(~Element,~field,x,iy,ix,cb,h,w,1) == PackingSamples.masked(field(Tree.read(Element,x,(cb + iy * w + ix : U32))),U32.is_lt(iy,h) && U32.is_lt(ix,w)) : F32}:
  +offset = (iy * w + ix : U32)
  Equal.cong(U32,F32,index => PackingSamples.masked(field(Tree.read(Element,x,index)),U32.is_lt(iy,h) && U32.is_lt(ix,w)),(cb + offset * 1 : U32),(cb + iy * w + ix : U32),
    Equal.trans(U32,(cb + offset * 1 : U32),(cb + offset : U32),(cb + iy * w + ix : U32),
      Equal.cong(U32,U32,tail => (cb + tail : U32),(offset * 1 : U32),offset,Arithmetic.multiply_one(offset)),
      Equal.sym(U32,(cb + iy * w + ix : U32),(cb + offset : U32),Arithmetic.add_associative(cb,(iy * w : U32),ix))))

# A pixel at step 1 reads the plane at row and column.
def unit_step_pixel(~Element: Data,~field: Element -> F32,input: Array<Element>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32) ->
  {Packing.pixel(~Element,~field,input,iy,ix,cb,h,w,1) ==
    Packing.masked_field(~Element,U32.is_lt(iy,h) && U32.is_lt(ix,w),Packing.field_of(~Element,~field,Array.get(Element,input,(cb + iy * w + ix : U32)))) : Array<Element> & F32}:
  +offset = (iy * w + ix : U32)
  Equal.cong(U32,Array<Element> & F32,index => Packing.masked_field(~Element,U32.is_lt(iy,h) && U32.is_lt(ix,w),Packing.field_of(~Element,~field,Array.get(Element,input,index))),
    (cb + offset * 1 : U32),(cb + iy * w + ix : U32),
    Equal.trans(U32,(cb + offset * 1 : U32),(cb + offset : U32),(cb + iy * w + ix : U32),
      Equal.cong(U32,U32,tail => (cb + tail : U32),(offset * 1 : U32),offset,Arithmetic.multiply_one(offset)),
      Equal.sym(U32,(cb + iy * w + ix : U32),(cb + offset : U32),Arithmetic.add_associative(cb,(iy * w : U32),ix))))

def pixel(~Element: Data,~field: Element -> F32,+x: Tree.Tree<Element>,+iy: U32,+ix: U32,+cb: U32,+h: U32,+w: U32,+step: U32) ->
  {Packing.pixel(~Element,~field,Tree.pack(Element,x),iy,ix,cb,h,w,step) == (Tree.pack(Element,x),value(~Element,~field,x,iy,ix,cb,h,w,step)) : Array<Element> & F32}:
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,x),(cb + (iy * w + ix) * step : U32)),(Tree.pack(Element,x),Tree.read(Element,x,(cb + (iy * w + ix) * step : U32))),
    StorageLaws.read(Element,x,(cb + (iy * w + ix) * step : U32))) :
    {Packing.masked_field(~Element,U32.is_lt(iy,h) && U32.is_lt(ix,w),Packing.field_of(~Element,~field,_)) == (Tree.pack(Element,x),value(~Element,~field,x,iy,ix,cb,h,w,step)) :
      Array<Element> & F32}
  {==}

def pack_input(~Element: Data,input: Tree.Tree<Element>) -> Array<Element>:
  Tree.pack(Element,input)

def block_sample(~Element: Data,~field: Element -> F32,input: Tree.Tree<Element>,context: Packing.PixelBlock,index: Nat) -> F32:
  Packing.PixelBlock{row,column,channel_base,height,width,+stride,step} = context
  value(~Element,~field,input,row,(column + U32.from_nat(index) * stride : U32),channel_base,height,width,step)

def native_block_read(~Element: Data,~field: Element -> F32,+input: Tree.Tree<Element>,+context: Packing.PixelBlock,+index: Nat) ->
  {Packing.read_block_pixel(~Element,~field,pack_input(~Element,input),context,index) == (pack_input(~Element,input),block_sample(~Element,~field,input,context,index)) : Array<Element> & F32}:
  Packing.PixelBlock{row,column,channel_base,height,width,+stride,step} = context
  pixel(~Element,~field,input,row,(column + U32.from_nat(index) * stride : U32),channel_base,height,width,step)

def native_vector(~Element: Data,~field: Element -> F32,input: Array<Element>,context: Packing.PixelBlock) -> Array<Element> & Tile.Tile8:
  Packing.PixelBlock{+row,+column,+channel_base,+height,+width,+stride,+step} = context
  Tile.read_tail(~Array<Element>,~Packing.PixelBlock,~Packing.read_block_pixel(~Element,~field),
    Packing.pixel(~Element,~field,input,row,column,channel_base,height,width,step),context)

def pure_vector(~Element: Data,~field: Element -> F32,+input: Tree.Tree<Element>,+context: Packing.PixelBlock) -> Tile.Tile8:
  Tile.from_sequence(Sequence.tabulate(~F32,~TileLaws.SampleContext<Tree.Tree<Element>,Packing.PixelBlock>,
    ~TileLaws.sample_context(~Tree.Tree<Element>,~Packing.PixelBlock,~block_sample(~Element,~field)),8n,0n,TileLaws.SampleContext{input,context}))

def zero_lane_column(+column: U32,+stride: U32) -> {(column + 0 * stride : U32) == column : U32}:
  %Equal.sym(U32,(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)) : {(column + _ : U32) == column : U32}
  Arithmetic.add_zero(column)

def native_vector_matches_tabulation(~Element: Data,~field: Element -> F32,+input: Tree.Tree<Element>,+context: Packing.PixelBlock) ->
  {native_vector(~Element,~field,Tree.pack(Element,input),context) == (Tree.pack(Element,input),pure_vector(~Element,~field,input,context)) : Array<Element> & Tile.Tile8}:
  Packing.PixelBlock{+row,+column,+channel_base,+height,+width,+stride,+step} = context
  %Equal.sym(Array<Element> & F32,Packing.pixel(~Element,~field,Tree.pack(Element,input),row,column,channel_base,height,width,step),
    (Tree.pack(Element,input),value(~Element,~field,input,row,column,channel_base,height,width,step)),pixel(~Element,~field,input,row,column,channel_base,height,width,step)) :
    {Tile.read_tail(~Array<Element>,~Packing.PixelBlock,~Packing.read_block_pixel(~Element,~field),_,context) ==
      (Tree.pack(Element,input),pure_vector(~Element,~field,input,context)) : Array<Element> & Tile.Tile8}
  %Equal.sym(U32,(column + 0 * stride : U32),column,zero_lane_column(column,stride)) :
    {Tile.read_tail(~Array<Element>,~Packing.PixelBlock,~Packing.read_block_pixel(~Element,~field),
        (Tree.pack(Element,input),value(~Element,~field,input,row,column,channel_base,height,width,step)),context) ==
      (Tree.pack(Element,input),Tile.from_sequence(Sequence.Elements{value(~Element,~field,input,row,_,channel_base,height,width,step),
        Sequence.tabulate(~F32,~TileLaws.SampleContext<Tree.Tree<Element>,Packing.PixelBlock>,~TileLaws.sample_context(~Tree.Tree<Element>,~Packing.PixelBlock,~block_sample(~Element,~field)),
          7n,1n,TileLaws.SampleContext{input,context})})) : Array<Element> & Tile.Tile8}
  TileLaws.tail_block_refines(~Array<Element>,~Tree.Tree<Element>,~Packing.PixelBlock,~pack_input(~Element),~Packing.read_block_pixel(~Element,~field),
    ~block_sample(~Element,~field),~native_block_read(~Element,~field),input,context,value(~Element,~field,input,row,column,channel_base,height,width,step))

def address_sample(~Element: Data,~field: Element -> F32,input: Tree.Tree<Element>,address: PackingSamples.PixelAddress) -> F32:
  PackingSamples.PixelAddress{index,inside} = address
  PackingSamples.masked(field(Tree.read(Element,input,index)),inside)

def native_address_read(~Element: Data,~field: Element -> F32,+input: Tree.Tree<Element>,+address: PackingSamples.PixelAddress) ->
  {PackingSamples.read_pixel_address(~Element,~field,pack_input(~Element,input),address) ==
    (pack_input(~Element,input),address_sample(~Element,~field,input,address)) : Array<Element> & F32}:
  PackingSamples.PixelAddress{+index,+inside} = address
  %Equal.sym(Array<Element> & Element,Array.get(Element,pack_input(~Element,input),index),
    (pack_input(~Element,input),Tree.read(Element,input,index)),StorageLaws.read(Element,input,index)) :
    {PackingSamples.mask_field(~Element,~field,inside,_) == (pack_input(~Element,input),PackingSamples.masked(field(Tree.read(Element,input,index)),inside)) : Array<Element> & F32}
  {==}

def window_sample(~Element: Data,~field: Element -> F32,input: Tree.Tree<Element>,context: PackingSamples.WindowPixelBlock,index: Nat) -> F32:
  address_sample(~Element,~field,input,PackingSamples.window_pixel_address(context,index))

def native_window_read(~Element: Data,~field: Element -> F32,+input: Tree.Tree<Element>,+context: PackingSamples.WindowPixelBlock,+index: Nat) ->
  {PackingSamples.read_window_pixel(~Element,~field,pack_input(~Element,input),context,index) ==
    (pack_input(~Element,input),window_sample(~Element,~field,input,context,index)) : Array<Element> & F32}:
  native_address_read(~Element,~field,input,PackingSamples.window_pixel_address(context,index))

def window_context(~located: U32 -> U32 -> U32 -> U32,~spread: U32 -> U32,+window: Packing.Stencil,position: U32,reduction: U32) -> PackingSamples.WindowPixelBlock:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  PackingSamples.window_pixel_block(~located,~spread,position,reduction,base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,groups)

def window_value(~Element: Data,~field: Element -> F32,~located: U32 -> U32 -> U32 -> U32,~spread: U32 -> U32,input: Tree.Tree<Element>,window: Packing.Stencil,position: U32,reduction: U32) -> F32:
  window_sample(~Element,~field,input,window_context(~located,~spread,window,position,reduction),0n)

def window_vector(~Element: Data,~field: Element -> F32,~located: U32 -> U32 -> U32 -> U32,~spread: U32 -> U32,+input: Tree.Tree<Element>,+window: Packing.Stencil,+position: U32,+reduction: U32) -> Tile.Tile8:
  Tile.from_sequence(Sequence.Elements{window_value(~Element,~field,~located,~spread,input,window,position,reduction),
    Sequence.tabulate(~F32,~TileLaws.SampleContext<Tree.Tree<Element>,PackingSamples.WindowPixelBlock>,
      ~TileLaws.sample_context(~Tree.Tree<Element>,~PackingSamples.WindowPixelBlock,~window_sample(~Element,~field)),7n,1n,
      TileLaws.SampleContext{input,window_context(~located,~spread,window,position,reduction)})})

def native_window_vector(~Element: Data,~field: Element -> F32,~located: U32 -> U32 -> U32 -> U32,~spread: U32 -> U32,input: Array<Element>,+window: Packing.Stencil,+position: U32,+reduction: U32) ->
  Array<Element> & Tile.Tile8:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  Tile.read_tail(~Array<Element>,~PackingSamples.WindowPixelBlock,~PackingSamples.read_window_pixel(~Element,~field),
    PackingSamples.window_pixel(~Element,~field,~located,~spread,input,position,reduction,base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,groups),
    window_context(~located,~spread,window,position,reduction))

def window_vector_matches(~Element: Data,~field: Element -> F32,~located: U32 -> U32 -> U32 -> U32,~spread: U32 -> U32,+input: Tree.Tree<Element>,+window: Packing.Stencil,+position: U32,+reduction: U32) ->
  {native_window_vector(~Element,~field,~located,~spread,pack_input(~Element,input),window,position,reduction) ==
    (pack_input(~Element,input),window_vector(~Element,~field,~located,~spread,input,window,position,reduction)) : Array<Element> & Tile.Tile8}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,inner,+groups} = window
  %Equal.sym(Array<Element> & F32,
    PackingSamples.window_pixel(~Element,~field,~located,~spread,pack_input(~Element,input),position,reduction,base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,groups),
    (pack_input(~Element,input),window_value(~Element,~field,~located,~spread,input,window,position,reduction)),
    native_window_read(~Element,~field,input,window_context(~located,~spread,window,position,reduction),0n)) :
    {Tile.read_tail(~Array<Element>,~PackingSamples.WindowPixelBlock,~PackingSamples.read_window_pixel(~Element,~field),_,window_context(~located,~spread,window,position,reduction)) ==
      (pack_input(~Element,input),window_vector(~Element,~field,~located,~spread,input,window,position,reduction)) : Array<Element> & Tile.Tile8}
  TileLaws.tail_block_refines(~Array<Element>,~Tree.Tree<Element>,~PackingSamples.WindowPixelBlock,~pack_input(~Element),~PackingSamples.read_window_pixel(~Element,~field),
    ~window_sample(~Element,~field),~native_window_read(~Element,~field),input,window_context(~located,~spread,window,position,reduction),
    window_value(~Element,~field,~located,~spread,input,window,position,reduction))

# ---- The values of a plain window ----

# Channel c of values placed channel after channel, plane apart, pixels
# adjacent.
def planes(channel: U32,plane: U32,groups: U32) -> U32:
  (channel * plane : U32)

def adjacent(groups: U32) -> U32:
  1

# The same stencil over values read from zero.
def plain(+window: Packing.Stencil) -> Packing.Stencil:
  Packing.Stencil{base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Stencil{0,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups}

# The window vector of values at a position and a reduction index.
def value_vector(+values: Tree.Tree<F32>,+window: Packing.Stencil,+position: U32,+reduction: U32) -> Tile.Tile8:
  window_vector(~F32,~Packing.identity,~planes,~adjacent,values,plain(window),position,reduction)

# ---- The packed entries ----

# The inner extent of a window: the entries of the block of a position.
def inner_count(+window: Packing.Stencil) -> Nat:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,+inner,groups} = window
  U32.to_nat(inner)

# What every packed entry reads: the values, the window and the first
# position.
type Entries is Data:
  Entries{values: Tree.Tree<F32>,window: Packing.Stencil,start: U32}

# Entry index of a packing: the value vector at the block index / inner of
# eight positions from the start and the reduction index index % inner.
def packed_entry(context: Entries,index: Nat) -> Tile.Tile8:
  Entries{+values,+window,start} = context
  +inner = inner_count(window)
  +at = index
  value_vector(values,window,(start + U32.from_nat(Nat.div(at,inner)) * 8 : U32),U32.from_nat(Nat.mod(at,inner)))

# The first count packed entries written into tiles from zero on.
def fill(count: Nat,+values: Tree.Tree<F32>,+window: Packing.Stencil,+start: U32,tiles: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Tree.write_sequence(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~Entries,~packed_entry,count,0n,Entries{values,window,start}),U32.from_nat(0n),tiles)

def fill_balanced(+count: Nat,+values: Tree.Tree<F32>,+window: Packing.Stencil,+start: U32,+depth: Nat,+tiles: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles)) -> Shape.Balanced(Tile.Tile8,depth,fill(count,values,window,start,tiles)):
  ListRead.sequence_balanced(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~Entries,~packed_entry,count,0n,Entries{values,window,start}),U32.from_nat(0n),depth,tiles,balanced)

# A filled index holds its packed entry.
def fill_entry(+count: Nat,+values: Tree.Tree<F32>,+window: Packing.Stencil,+start: U32,+depth: Nat,+tiles: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Tile.Tile8,tiles))) == True{} : Bool},+query: Nat,+within: {Nat.is_lt(query,count) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,fill(count,values,window,start,tiles),U32.from_nat(query)) == packed_entry(Entries{values,window,start},query) : Tile.Tile8}:
  +context = {Entries{values,window,start} : Entries}
  +entries = ListRead.enumerate(~Tile.Tile8,~Entries,~packed_entry,count,0n,context)
  +length = ListRead.enumerate_length(~Tile.Tile8,~Entries,~packed_entry,count,0n,context)
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,fill(count,values,window,start,tiles),U32.from_nat(query)),ListRead.at(~Tile.Tile8,Tile.filled(0.0),query,entries),
    packed_entry(context,query),
    ListRead.listed(~Tile.Tile8,entries,query,0n,depth,tiles,Tile.filled(0.0),balanced,limit,
      %Equal.sym(Nat,List.length(&2,Tile.Tile8,entries),count,length) : {Nat.is_le(_,U32.to_nat(Tree.capacity(Tile.Tile8,tiles))) == True{} : Bool} room,
      %Equal.sym(Nat,List.length(&2,Tile.Tile8,entries),count,length) : {Nat.is_lt(query,_) == True{} : Bool} within),
    ListRead.enumerate_at(~Tile.Tile8,~Entries,~packed_entry,Tile.filled(0.0),count,query,0n,context,within))

# An index past the filled ones keeps its tile.
def fill_kept(+count: Nat,+values: Tree.Tree<F32>,+window: Packing.Stencil,+start: U32,+depth: Nat,+tiles: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Tile.Tile8,tiles))) == True{} : Bool},+query: Nat,
  +query_inside: {Nat.is_lt(query,U32.to_nat(Tree.capacity(Tile.Tile8,tiles))) == True{} : Bool},+after: {Nat.is_le(count,query) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,fill(count,values,window,start,tiles),U32.from_nat(query)) == Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}:
  +context = {Entries{values,window,start} : Entries}
  +entries = ListRead.enumerate(~Tile.Tile8,~Entries,~packed_entry,count,0n,context)
  +length = ListRead.enumerate_length(~Tile.Tile8,~Entries,~packed_entry,count,0n,context)
  ListRead.preserved_after(~Tile.Tile8,entries,0n,query,depth,tiles,balanced,limit,
    %Equal.sym(Nat,List.length(&2,Tile.Tile8,entries),count,length) : {Nat.is_le(_,U32.to_nat(Tree.capacity(Tile.Tile8,tiles))) == True{} : Bool} room,
    query_inside,%Equal.sym(Nat,List.length(&2,Tile.Tile8,entries),count,length) : {Nat.is_le(_,query) == True{} : Bool} after)

# The entries read the values from zero, so the base of the window does not
# change them.
def entries_plain(n: Nat,+first: Nat,+values: Tree.Tree<F32>,+window: Packing.Stencil,+start: U32) ->
  {ListRead.enumerate(~Tile.Tile8,~Entries,~packed_entry,n,first,Entries{values,window,start}) ==
    ListRead.enumerate(~Tile.Tile8,~Entries,~packed_entry,n,first,Entries{values,plain(window),start}) : List<&2,Tile.Tile8>}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
      +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
      Equal.cong(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,tail => packed_entry(Entries{values,window,start},first) <> tail,
        ListRead.enumerate(~Tile.Tile8,~Entries,~packed_entry,rest,1n+first,Entries{values,window,start}),
        ListRead.enumerate(~Tile.Tile8,~Entries,~packed_entry,rest,1n+first,Entries{values,plain(window),start}),
        entries_plain(rest,1n+first,values,window,start))

def fill_plain(+count: Nat,+values: Tree.Tree<F32>,+window: Packing.Stencil,+start: U32,+tiles: Tree.Tree<Tile.Tile8>) ->
  {fill(count,values,window,start,tiles) == fill(count,values,plain(window),start,tiles) : Tree.Tree<Tile.Tile8>}:
  Equal.cong(List<&2,Tile.Tile8>,Tree.Tree<Tile.Tile8>,entries => Tree.write_sequence(~Tile.Tile8,entries,U32.from_nat(0n),tiles),
    ListRead.enumerate(~Tile.Tile8,~Entries,~packed_entry,count,0n,Entries{values,window,start}),
    ListRead.enumerate(~Tile.Tile8,~Entries,~packed_entry,count,0n,Entries{values,plain(window),start}),
    entries_plain(count,0n,values,window,start))
