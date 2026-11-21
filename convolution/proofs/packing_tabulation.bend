# The flat tabulation of packed entries: entry i holds tile i / inner at reduction
# index i % inner, every lane read independently.
# The tabulation still uses native pixel addresses. Connecting its quotient /
# remainder coordinates to CONV.patch is a separate theorem, not assumed here.
import Base
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/proofs/kernel_spatial_packing_samples.bend as SpatialSamples
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/math_sequence.bend as Sequence
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./packing_lane_reads.bend as LaneReads
import ./tile_array_projection.bend as ArrayProjection
import ./packed_reduction_samples.bend as ReductionSamples
import ../specification.bend as ConvSpec
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/list_storage_readback.bend as ListRead
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/u32_division.bend as WordDivision
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic

def indexed_position(+position: U32,index: Nat) -> U32:
  match index:
    case 0n: position
    case 1n+rest: (position + U32.from_nat(1n+rest) : U32)

# An indexed position from an encoded position is the encoded sum.
def indexed_encoded(+position: Nat,index: Nat) -> {indexed_position(U32.from_nat(position),index) == U32.from_nat(Nat.add(position,index)) : U32}:
  match index:
    case 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) : {U32.from_nat(position) == U32.from_nat(_) : U32}
      {==}
    case 1n+rest:
      WordArithmetic.native_add(position,1n+rest)

type PixelContext is Data:
  PixelContext{input: Tree.Tree<F32>,position: U32,reduction: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,
    output_width: U32}

def sample_pixel(context: PixelContext,index: Nat) -> F32:
  PixelContext{input,position,reduction,height,width,kernel,stride,padding,output_width} = context
  LaneReads.value(input,indexed_position(position,index),reduction,height,width,kernel,stride,padding,output_width)

def vector(+input: Tree.Tree<F32>,+position: U32,+reduction: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_width: U32) -> Tile.Tile8:
  Tile.from_sequence(Sequence.tabulate(~F32,~PixelContext,~sample_pixel,8n,0n,
    PixelContext{input,position,reduction,height,width,kernel,stride,padding,output_width}))

def entry(+i: U32,+start: U32,+x: Tree.Tree<F32>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,+inner: U32) -> Tile.Tile8:
  vector(x,(start + (i / inner) * 8 : U32),(i % inner : U32),h,w,k,stride,pad,output_width)

def end(n: Nat,i: U32) -> U32:
  match n:
    case 0n:
      i
    case 1n+rest:
      end(rest,(i + 1 : U32))

def fill(n: Nat,+i: U32,+start: U32,+x: Tree.Tree<F32>,y: Tree.Tree<Tile.Tile8>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,
  +output_width: U32,+inner: U32) -> Tree.Tree<Tile.Tile8>:
  match n:
    case 0n:
      y
    case 1n+rest:
      fill(rest,(i + 1 : U32),start,x,Tree.write(Tile.Tile8,y,i,entry(i,start,x,h,w,k,stride,pad,output_width,inner)),h,w,k,stride,pad,output_width,inner)

def lane_values(n: Nat,+i: U32,+start: U32,+x: Tree.Tree<F32>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,+inner: U32,
  +lane: Sequence.Index(8n)) -> List<&2,F32>:
  match n:
    case 0n:
      []
    case 1n+rest:
      ReductionSamples.select(lane,entry(i,start,x,h,w,k,stride,pad,output_width,inner)) <> lane_values(rest,(i + 1 : U32),start,x,h,w,k,stride,pad,
        output_width,inner,lane)

def length(+n: Nat,+i: U32,+start: U32,+x: Tree.Tree<F32>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,+inner: U32,
  +lane: Sequence.Index(8n)) -> {ConvSpec.length(lane_values(n,i,start,x,h,w,k,stride,pad,output_width,inner,lane)) == n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %length(rest,(i + 1 : U32),start,x,h,w,k,stride,pad,output_width,inner,lane) : {1n+ConvSpec.length(lane_values(rest,(i + 1 : U32),start,x,h,w,k,
            stride,pad,output_width,inner,lane)) == 1n+_ : Nat}
      {==}

law projected:
  for +n: Nat
  for +i: U32
  for +start: U32
  for +x: Tree.Tree<F32>
  for +y: Tree.Tree<Tile.Tile8>
  for +h: U32
  for +w: U32
  for +k: U32
  for +stride: U32
  for +pad: U32
  for +output_width: U32
  for +inner: U32
  for +lane: Sequence.Index(8n)
  {ArrayProjection.project(fill(n,i,start,x,y,h,w,k,stride,pad,output_width,inner),lane) == Tree.write_sequence(~F32,lane_values(n,i,start,x,h,w,k,
        stride,pad,output_width,inner,lane),i,ArrayProjection.project(y,lane)) : Tree.Tree<F32>}
def projected(n,i,start,x,y,h,w,k,stride,pad,output_width,inner,lane):
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %ArrayProjection.write_projected(y,lane,i,entry(i,start,x,h,w,k,stride,pad,output_width,inner)) : {ArrayProjection.project(fill(1n+rest,i,start,x,y,h,w,k,
            stride,pad,output_width,inner),lane) == Tree.write_sequence(~F32,lane_values(rest,(i + 1 : U32),start,x,h,w,k,stride,pad,output_width,inner,lane),
          (i + 1 : U32),_) : Tree.Tree<F32>}
      projected(rest,(i + 1 : U32),start,x,Tree.write(Tile.Tile8,y,i,entry(i,start,x,h,w,k,stride,pad,output_width,inner)),h,w,k,stride,pad,output_width,inner,lane)

# Overwriting any prefix leaves the reusable tile workspace's depth unchanged.
def fill_balanced(+n: Nat,+i: U32,+start: U32,+x: Tree.Tree<F32>,+y: Tree.Tree<Tile.Tile8>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,
  +output_width: U32,+inner: U32,+depth: Nat,balanced: Shape.Balanced(Tile.Tile8,depth,y)) ->
  Shape.Balanced(Tile.Tile8,depth,fill(n,i,start,x,y,h,w,k,stride,pad,output_width,inner)):
  match n:
    case 0n: balanced
    case 1n+ +rest:
      fill_balanced(rest,(i + 1 : U32),start,x,Tree.write(Tile.Tile8,y,i,entry(i,start,x,h,w,k,stride,pad,output_width,inner)),h,w,k,stride,pad,output_width,inner,depth,
        Shape.write(Tile.Tile8,depth,y,i,entry(i,start,x,h,w,k,stride,pad,output_width,inner),balanced))

def square_vector(+input: Tree.Tree<F32>,+position: U32,+reduction: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_width: U32,+inner: U32) ->
  {SpatialSamples.window_vector(~F32,~Packing.identity,~SpatialSamples.planes,~SpatialSamples.adjacent,input,LaneReads.window(height,width,kernel,stride,padding,output_width,inner),position,reduction) == vector(input,position,reduction,height,width,kernel,stride,padding,output_width) : Tile.Tile8}:
  TileLaws.tiles_equal(SpatialSamples.window_vector(~F32,~Packing.identity,~SpatialSamples.planes,~SpatialSamples.adjacent,input,LaneReads.window(height,width,kernel,stride,padding,output_width,inner),position,reduction),vector(input,position,reduction,height,width,kernel,stride,padding,output_width),
    LaneReads.square_first(input,position,reduction,height,width,kernel,stride,padding,output_width,inner),
    LaneReads.square_sample(input,position,reduction,height,width,kernel,stride,padding,output_width,inner,1n),
    LaneReads.square_sample(input,position,reduction,height,width,kernel,stride,padding,output_width,inner,2n),
    LaneReads.square_sample(input,position,reduction,height,width,kernel,stride,padding,output_width,inner,3n),
    LaneReads.square_sample(input,position,reduction,height,width,kernel,stride,padding,output_width,inner,4n),
    LaneReads.square_sample(input,position,reduction,height,width,kernel,stride,padding,output_width,inner,5n),
    LaneReads.square_sample(input,position,reduction,height,width,kernel,stride,padding,output_width,inner,6n),
    LaneReads.square_sample(input,position,reduction,height,width,kernel,stride,padding,output_width,inner,7n))

# ---- The library packing fills the tabulated entries ----

# The value vector of an entry of the library packing is the tabulated entry.
def filled_entry(+x: Tree.Tree<F32>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,+inner: U32,+start: U32,+first: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+first_room: {Nat.is_le(first,U32.to_nat(cap)) == True{} : Bool},
  +divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool}) ->
  {SpatialSamples.packed_entry(SpatialSamples.Entries{x,LaneReads.window(h,w,k,stride,pad,output_width,inner),start},first) ==
    entry(U32.from_nat(first),start,x,h,w,k,stride,pad,output_width,inner) : Tile.Tile8}:
  +n = U32.to_nat(inner)
  +inner_room = Order.transitive(n,Nat.double(n),U32.to_nat(cap),Order.le_double(n),divisor_bound)
  +quotient = Nat.div(first,n)
  +remainder = Nat.mod(first,n)
  +divided = {
    %Arithmetic.native_roundtrip(inner) : {(U32.from_nat(first) / _ : U32) == U32.from_nat(quotient) : U32}
    WordDivision.encoded_divide(first,n,cap,cap_ok,first_room,inner_room,divisor_bound,positive) : {(U32.from_nat(first) / inner : U32) == U32.from_nat(quotient) : U32}}
  +reduced = {
    %Arithmetic.native_roundtrip(inner) : {(U32.from_nat(first) % _ : U32) == U32.from_nat(remainder) : U32}
    WordDivision.encoded_modulo(first,n,cap,cap_ok,first_room,inner_room,divisor_bound,positive) : {(U32.from_nat(first) % inner : U32) == U32.from_nat(remainder) : U32}}
  %Equal.sym(U32,(U32.from_nat(first) / inner : U32),U32.from_nat(quotient),divided) :
    {SpatialSamples.packed_entry(SpatialSamples.Entries{x,LaneReads.window(h,w,k,stride,pad,output_width,inner),start},first) ==
      vector(x,(start + _ * 8 : U32),(U32.from_nat(first) % inner : U32),h,w,k,stride,pad,output_width) : Tile.Tile8}
  %Equal.sym(U32,(U32.from_nat(first) % inner : U32),U32.from_nat(remainder),reduced) :
    {SpatialSamples.packed_entry(SpatialSamples.Entries{x,LaneReads.window(h,w,k,stride,pad,output_width,inner),start},first) ==
      vector(x,(start + U32.from_nat(quotient) * 8 : U32),_,h,w,k,stride,pad,output_width) : Tile.Tile8}
  square_vector(x,(start + U32.from_nat(quotient) * 8 : U32),U32.from_nat(remainder),h,w,k,stride,pad,output_width,inner)

# Writing the value vectors of the entries from first on writes the
# tabulated entries from first on.
def filled_from(n: Nat,+first: Nat,+start: U32,+x: Tree.Tree<F32>,+y: Tree.Tree<Tile.Tile8>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,
  +inner: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(first,n),U32.to_nat(cap)) == True{} : Bool},
  +divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool}) ->
  {Tree.write_sequence(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~SpatialSamples.Entries,~SpatialSamples.packed_entry,n,first,
      SpatialSamples.Entries{x,LaneReads.window(h,w,k,stride,pad,output_width,inner),start}),U32.from_nat(first),y) ==
    fill(n,U32.from_nat(first),start,x,y,h,w,k,stride,pad,output_width,inner) : Tree.Tree<Tile.Tile8>}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +context = {SpatialSamples.Entries{x,LaneReads.window(h,w,k,stride,pad,output_width,inner),start} : SpatialSamples.Entries}
      +value = entry(U32.from_nat(first),start,x,h,w,k,stride,pad,output_width,inner)
      +first_room = Order.transitive(first,Nat.add(first,1n+rest),U32.to_nat(cap),Order.le_add(first,1n+rest),room)
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.increment_encoded(first)) :
        {Tree.write_sequence(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~SpatialSamples.Entries,~SpatialSamples.packed_entry,rest,1n+first,context),_,
            Tree.write(Tile.Tile8,y,U32.from_nat(first),SpatialSamples.packed_entry(context,first))) ==
          fill(rest,_,start,x,Tree.write(Tile.Tile8,y,U32.from_nat(first),value),h,w,k,stride,pad,output_width,inner) : Tree.Tree<Tile.Tile8>}
      %Equal.sym(Tile.Tile8,SpatialSamples.packed_entry(context,first),value,
        filled_entry(x,h,w,k,stride,pad,output_width,inner,start,first,cap,cap_ok,first_room,divisor_bound,positive)) :
        {Tree.write_sequence(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~SpatialSamples.Entries,~SpatialSamples.packed_entry,rest,1n+first,context),U32.from_nat(1n+first),
            Tree.write(Tile.Tile8,y,U32.from_nat(first),_)) ==
          fill(rest,U32.from_nat(1n+first),start,x,Tree.write(Tile.Tile8,y,U32.from_nat(first),value),h,w,k,stride,pad,output_width,inner) : Tree.Tree<Tile.Tile8>}
      filled_from(rest,1n+first,start,x,Tree.write(Tile.Tile8,y,U32.from_nat(first),value),h,w,k,stride,pad,output_width,inner,cap,cap_ok,
        Order.advance_room(first,rest,U32.to_nat(cap),room),divisor_bound,positive)

# The library packing of a window of values is the tabulation of its entries.
def filled(+count: Nat,+start: U32,+x: Tree.Tree<F32>,+y: Tree.Tree<Tile.Tile8>,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,+inner: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(cap)) == True{} : Bool},
  +divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool}) ->
  {SpatialSamples.fill(count,x,LaneReads.window(h,w,k,stride,pad,output_width,inner),start,y) == fill(count,0,start,x,y,h,w,k,stride,pad,output_width,inner) :
    Tree.Tree<Tile.Tile8>}:
  filled_from(count,0n,start,x,y,h,w,k,stride,pad,output_width,inner,cap,cap_ok,room,divisor_bound,positive)
