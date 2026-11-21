# Maps of the cells of a tiled tensor. A function of the tiles may leave
# values in the lanes past the true extent; every leaf puts +0 back there
# before it returns its piece, so the result is again the tiled form of its
# values.
import Base
import ./kernel_tile.bend as Tile
import ./storage_buffer.bend as Storage
import ./storage_allocation.bend as Allocation
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_operations.bend as Operations
import ./tensor_view.bend as Views
import ./tensor_tiled.bend as Tiled
import ./tensor_tiled_cells.bend as Cells
import ./tensor_tiled_indexed.bend as TiledIndexed
import ./traversal_array.bend as Traversal
import ./proofs/traversal_certificate.bend as TraversalCertificate
import ./proofs/tensor_tiled_indexed_certificate.bend as TiledIndexedCertificate

# ---- Padding ----

# The lanes of the lane axis, its groups, the cells of one group and where
# the groups lie.
type Padding is Data:
  Padding{lanes: U32,groups: U32,outer: U32,position: Grid.Position}

def padding_when(within: Bool,+cells: List<&2,U32>,+lanes: U32,+position: Grid.Position) -> Padding:
  match within:
    case True{}: Padding{lanes,Grid.groups(lanes),Grid.outer_count(cells,position),position}
    case False{}: Padding{1,1,Grid.product(cells),Grid.Innermost{}}

def padding_of(+cells: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position) -> Padding:
  padding_when(Nat.is_lt(axis,List.length(&2,U32,cells)),cells,lanes,position)

def padded_at(+start: U32,+count: U32,+tail: Cells.Tail,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{storage,length,certificate} = buffer
  Storage.Buffer{Traversal.map_in_place(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),start,tail,storage),length,
    TraversalCertificate.map_in_place(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),start,tail,storage,certificate)}

# The last cell of every run of groups cells, from the cell at position on.
def padded_rows(remaining: Nat,+position: U32,+groups: U32,+tail: Cells.Tail,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  match remaining:
    case 0n: buffer
    case 1n+rest: padded_rows(rest,(position + groups : U32),groups,tail,padded_at(position,1,tail,buffer))

# A piece may start inside a run of groups when one run outgrows a piece.
def padded_inner(single: Bool,+first: U32,+count: U32,+groups: U32,+tail: Cells.Tail,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  match single:
    case True{}: padded_at(0,count,tail,buffer)
    case False{}:
      +start = ((groups - 1) - first % groups : U32)
      padded_rows(U32.to_nat(Bool.pick(U32,U32.is_lt(start,count),((count - start - 1) / groups + 1 : U32),0)),start,groups,tail,buffer)

def padded_range(meets: Bool,+start: U32,+count: U32,+tail: Cells.Tail,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  match meets:
    case True{}: padded_at(start,count,tail,buffer)
    case False{}: buffer

# With the groups outermost the last group is one range of cells.
def padded_outer(+first: U32,+count: U32,+groups: U32,+outer: U32,+tail: Cells.Tail,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  +low = U32.max(((groups - 1) * outer : U32),first)
  +high = U32.min((groups * outer : U32),(first + count : U32))
  padded_range(U32.is_lt(low,high),(low - first : U32),(high - low : U32),tail,buffer)

def padded_placed(position: Grid.Position,+first: U32,+count: U32,+groups: U32,+outer: U32,+tail: Cells.Tail,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  match position:
    case Grid.Innermost{}: padded_inner(U32.is_le(groups,1),first,count,groups,tail,buffer)
    case Grid.Outermost{}: padded_outer(first,count,groups,outer,tail,buffer)

def padded_when(aligned: Bool,position: Grid.Position,+first: U32,+count: U32,+lanes: U32,+groups: U32,+outer: U32,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  match aligned:
    case True{}: buffer
    case False{}: padded_placed(position,first,count,groups,outer,Cells.Tail{((groups - 1) * 8 : U32),lanes},buffer)

# The piece of first cell first and count cells with +0 in every lane past the
# true extent: the last group holds those lanes when the count is not a
# multiple of eight.
def padded(+padding: Padding,+first: U32,+count: U32,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Padding{+lanes,+groups,+outer,position} = padding
  padded_when(U32.is_zero((lanes % 8 : U32)),position,first,count,lanes,groups,outer,buffer)

def padding_cells(tiled: Tiled.Tiled) -> Tiled.Tiled & Padding:
  Tiled.Tiled{Tensors.Tensor{pieces,+cells,fill,status,workers},+axis,+lanes,+position} = tiled
  (Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,status,workers},axis,lanes,position},padding_of(cells,axis,lanes,position))

def clear_tail(aligned: Bool,+padding: Padding,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match aligned:
    case True{}: tiled
    case False{}:
      Tiled.Tiled{cells,axis,lanes,position} = tiled
      Tiled.Tiled{Tensors.map_pieces(~Tile.Tile8,~Padding,~padded,padding,0.0,cells),axis,lanes,position}

def restore_padding(pair: Tiled.Tiled & Padding) -> Tiled.Tiled:
  (tiled,padding) = pair
  Padding{+lanes,groups,outer,position} = padding
  clear_tail(U32.is_zero((lanes % 8 : U32)),padding,tiled)

# ---- Maps ----

# The scalar function of a context at every lane of a tile.
def lanes(~Context: Data,~operation: Context -> F32 -> F32,context: Context,tile: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  +context = context
  Tile.Tile{operation(context,a),operation(context,b),operation(context,c),operation(context,d),
    operation(context,e),operation(context,f),operation(context,g),operation(context,h)}

# What every piece of a map reads: the context of the function and the
# padding of the cells.
type Mapping<-Context: Data> is Data:
  Mapping{context: Context,padding: Padding}

def map_leaf(~Context: Data,~transform: Context -> Tile.Tile8 -> Tile.Tile8,+mapping: Mapping<Context>,+first: U32,+count: U32,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Mapping{+context,+padding} = mapping
  padded(padding,first,count,Operations.mapped_context(~Tile.Tile8,~Context,~transform,U32.to_nat(count),0,context,buffer))

def mapped(~Context: Data,~transform: Context -> Tile.Tile8 -> Tile.Tile8,context: Context,operations: F32,pair: Tiled.Tiled & Padding) -> Tiled.Tiled:
  (Tiled.Tiled{cells,axis,lanes,position},padding) = pair
  Tiled.Tiled{Tensors.map_pieces(~Tile.Tile8,~Mapping<Context>,~map_leaf(~Context,~transform),Mapping{context,padding},operations,cells),axis,lanes,position}

# The function of a context at every tile. operations counts the scalar work
# of one value.
def map(~Context: Data,~transform: Context -> Tile.Tile8 -> Tile.Tile8,context: Context,operations: F32,tiled: Tiled.Tiled) -> Tiled.Tiled:
  mapped(~Context,~transform,context,operations,padding_cells(tiled))

# A tensor of one value: zeros with the value in every true lane.
def constant(value: F32,element: F32) -> F32:
  value

def full(+shape: List<&2,U32>,+value: F32) -> Tiled.Tiled:
  map(~F32,~lanes(~F32,~constant),value,1.0,Tiled.zeros(shape))

# The scalar operation at every pair of lanes of two tiles.
def pairs(~operation: F32 -> F32 -> F32,left: Tile.Tile8,right: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a,b,c,d,e,f,g,h} = left
  Tile.Tile{i,j,k,l,m,n,o,p} = right
  Tile.Tile{operation(a,i),operation(b,j),operation(c,k),operation(d,l),
    operation(e,m),operation(f,n),operation(g,o),operation(h,p)}

def combined(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,name: String,operations: F32,pair: Tiled.Tiled & Padding,right: Tiled.Tiled) -> Tiled.Tiled:
  (Tiled.Tiled{left,axis,lanes,position},padding) = pair
  Tiled.Tiled{right,right_axis,right_lanes,right_position} = right
  Tiled.Tiled{Tensors.combine(~Tile.Tile8,~Padding,~operation,~padded,padding,name,operations,left,right),axis,lanes,position}

# The function of two tiles at every cell of two tensors whose lanes lie on
# the same axis at the same position.
def combine(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,name: String,operations: F32,left: Tiled.Tiled,right: Tiled.Tiled) -> Tiled.Tiled:
  combined(~operation,name,operations,padding_cells(left),right)

def combined_viewed(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,operations: F32,pair: Tiled.Tiled & Padding,right: Tiled.Tiled,+axes: List<&2,Views.Axis>,
  +offset: U32) -> Tiled.Tiled:
  (Tiled.Tiled{left,axis,lanes,position},padding) = pair
  Tiled.Tiled{right,right_axis,right_lanes,right_position} = right
  Tiled.Tiled{Tensors.combine_viewed(~Tile.Tile8,~Padding,~operation,~padded,padding,operations,left,right,axes,offset),axis,lanes,position}

# The function of two tiles at every cell of a tensor and the cell of another
# that a view of its cells with the cell extents of the first reads.
def combine_viewed(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,operations: F32,left: Tiled.Tiled,right: Tiled.Tiled,+axes: List<&2,Views.Axis>,+offset: U32) ->
  Tiled.Tiled:
  combined_viewed(~operation,operations,padding_cells(left),right,axes,offset)

# ---- Indexed maps ----

def indexed_buffers(~operation: U32 -> F32 -> F32,+call: TiledIndexed.Call,input: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{output,output_length,output_certificate} = output
  Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Operations.pair_buffers(~Tile.Tile8,input_length,output_length,TiledIndexed.run(~operation,call,input,output),
    TiledIndexedCertificate.certificates(~operation,call,input,output,input_certificate,output_certificate)))

# What every piece of an indexed map reads: the logical axes around the lane
# axis and the padding of the cells.
type Indexed is Data:
  Indexed{after: U32,padding: Padding}

# A new piece of cells from the piece input, with +0 put back past the lanes.
def indexed_leaf(~operation: U32 -> F32 -> F32,+context: Indexed,+first: U32,+count: U32,input: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Indexed{+after,+padding} = context
  Padding{+lanes,+groups,+outer,+position} = padding
  padded(padding,first,count,
    indexed_buffers(~operation,TiledIndexed.Call{U32.to_nat(count),TiledIndexed.Indexing{outer,after,lanes,groups,position,first}},input,
      Allocation.allocate(~Tile.Tile8,Storage.allocation_depth(U32.max(count,1)),Tiled.zero_tile(),U32.to_nat(count))))

def indexed_around(~operation: U32 -> F32 -> F32,operations: F32,axes: Maybe<&2,Tiled.Around>,cells: Tensors.Tensor<Tile.Tile8>,axis: Nat,lanes: U32,position: Grid.Position,
  padding: Padding) -> Tiled.Tiled:
  match axes:
    case Some{Tiled.Around{before,extent,after}}:
      Tiled.Tiled{Tensors.map_pieces(~Tile.Tile8,~Indexed,~indexed_leaf(~operation),Indexed{Tiled.product(after),padding},operations,cells),axis,lanes,position}
    case None{}: Tiled.Tiled{cells,axis,lanes,position}

def indexed_padded(~operation: U32 -> F32 -> F32,operations: F32,pair: Tiled.Tiled & Padding,axes: Maybe<&2,Tiled.Around>) -> Tiled.Tiled:
  (Tiled.Tiled{cells,axis,lanes,position},padding) = pair
  indexed_around(~operation,operations,axes,cells,axis,lanes,position,padding)

def indexed_shaped(~operation: U32 -> F32 -> F32,operations: F32,shape: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.Tiled{cells,+axis,lanes,position} = tiled
  indexed_padded(~operation,operations,padding_cells(Tiled.Tiled{cells,axis,lanes,position}),Tiled.around(shape,axis,1))

# The function of the row-major index and the value at every lane.
def indexed(~operation: U32 -> F32 -> F32,operations: F32,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers => indexed_shaped(~operation,operations,shape,tiled))
