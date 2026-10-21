# A rectangular stencil packs a contiguous prefix of canonical eight-lane
# entries. Geometry and counter bounds are independent of source values.
import Base
import ../kernel_spatial_packing.bend as Packing
import ../kernel_packing_samples.bend as PackingSamples
import ../kernel_tile.bend as Tile
import ../math_sequence.bend as Sequence
import ../traversal_loop.bend as Loop
import ./kernel_spatial_packing_refinement.bend as Model
import ./kernel_spatial_packing_samples.bend as Samples
import ./kernel_tile_refinement.bend as TileLaws
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./traversal_map_readback.bend as MapReadback
import ./traversal_batch_refinement.bend as Batch
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./row_major_index.bend as RowMajor
import ./binary_radix.bend as Radix
import ./nat_division.bend as Division
import ./indexed_sequence_laws.bend as SequenceLaws
import ./kernel_packing_window_coordinates.bend as Coordinates
import ./encoded_coordinates.bend as Encoded
import ./u32_division.bend as U32Division

type Domain is Data:
  Domain{geometry: Coordinates.Geometry,groups: Nat,start: U32,
    packed_room: {Nat.is_le(Nat.mul(groups,Coordinates.inner(geometry)),U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool},
    position_room: {Nat.is_le(Nat.add(U32.to_nat(start),Nat.mul(groups,8n)),U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool}}

def geometry(domain: Domain) -> Coordinates.Geometry:
  Domain{geometry,groups,start,packed_room,position_room} = domain
  geometry

def groups(domain: Domain) -> Nat:
  Domain{geometry,groups,start,packed_room,position_room} = domain
  groups

def start(domain: Domain) -> U32:
  Domain{geometry,groups,start,packed_room,position_room} = domain
  start

def count(+domain: Domain) -> Nat:
  Nat.mul(groups(domain),Coordinates.inner(geometry(domain)))

def reduction(+geometry: Coordinates.Geometry,+channel: Nat,+row: Nat,column: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(channel,Coordinates.rows(geometry)),row),Coordinates.columns(geometry)),column)

type Tabulation is Data:
  Tabulation{input: Tree.Tree<F32>,window: Packing.Stencil,start: U32}

def entry(context: Tabulation,index: U32) -> Tile.Tile8:
  Tabulation{input,+window,start} = context
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner} = window
  +index = index
  Samples.window_vector(input,window,
    U32.from_nat(Nat.add(U32.to_nat(start),Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(inner)),8n))),
    U32.from_nat(Nat.mod(U32.to_nat(index),U32.to_nat(inner))))

def entry_map(context: Tabulation,index: U32,previous: Tile.Tile8) -> Tile.Tile8:
  entry(context,index)

def native_entry(context: Tabulation,index: U32) -> Tile.Tile8:
  Tabulation{input,+window,start} = context
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner} = window
  +index = index
  Samples.window_vector(input,window,(start + index / inner * 8 : U32),(index % inner : U32))

def native_position(+start: U32,+group: U32) ->
  {(start + group * 8 : U32) == U32.from_nat(Nat.add(U32.to_nat(start),Nat.mul(U32.to_nat(group),8n))) : U32}:
  %WordArithmetic.native_add(U32.to_nat(start),Nat.mul(U32.to_nat(group),8n)) :
    {(start + group * 8 : U32) == _ : U32}
  %Encoded.scale_encoded(U32.to_nat(group),8) :
    {(start + group * 8 : U32) == (U32.from_nat(U32.to_nat(start)) + _ : U32) : U32}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(start)),start,Arithmetic.native_roundtrip(start)) :
    {(start + group * 8 : U32) == (_ + U32.from_nat(U32.to_nat(group)) * 8 : U32) : U32}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(group)),group,Arithmetic.native_roundtrip(group)) :
    {(start + group * 8 : U32) == (start + _ * 8 : U32) : U32}
  {==}

def entry_native(+g: Coordinates.Geometry,+start: U32,+input: Tree.Tree<F32>,+index: U32) ->
  {entry(Tabulation{input,Coordinates.window(g),start},index) == native_entry(Tabulation{input,Coordinates.window(g),start},index) : Tile.Tile8}:
  Coordinates.Geometry{window,channels,+cap,+ok,rp,cp,+ip,wp,ie,ce,rr,cr,ar,+ir,wr} = g
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner_word} = window
  %U32Division.divide(index,inner_word,cap,ok,ir,ip) :
    {Samples.window_vector(input,window,U32.from_nat(Nat.add(U32.to_nat(start),Nat.mul(_,8n))),U32.from_nat(Nat.mod(U32.to_nat(index),U32.to_nat(inner_word)))) ==
      native_entry(Tabulation{input,window,start},index) : Tile.Tile8}
  %U32Division.modulo(index,inner_word,cap,ok,ir,ip) :
    {Samples.window_vector(input,window,U32.from_nat(Nat.add(U32.to_nat(start),Nat.mul(U32.to_nat((index / inner_word : U32)),8n))),U32.from_nat(_)) ==
      native_entry(Tabulation{input,window,start},index) : Tile.Tile8}
  %native_position(start,(index / inner_word : U32)) :
    {Samples.window_vector(input,window,_,U32.from_nat(U32.to_nat((index % inner_word : U32)))) == native_entry(Tabulation{input,window,start},index) : Tile.Tile8}
  %Equal.sym(U32,U32.from_nat(U32.to_nat((index % inner_word : U32))),(index % inner_word : U32),Arithmetic.native_roundtrip((index % inner_word : U32))) :
    {Samples.window_vector(input,window,(start + index / inner_word * 8 : U32),_) == native_entry(Tabulation{input,window,start},index) : Tile.Tile8}
  {==}

def filled(count: Nat,index: U32,context: Tabulation,output: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  MapReadback.run_indexed(~Tile.Tile8,~Tabulation,~entry_map,count,index,context,output)

def filled_split(+first: Nat,+second: Nat,+index: Nat,+context: Tabulation,+output: Tree.Tree<Tile.Tile8>) ->
  {filled(Nat.add(first,second),U32.from_nat(index),context,output) ==
    filled(second,U32.from_nat(Nat.add(index,first)),context,filled(first,U32.from_nat(index),context,output)) : Tree.Tree<Tile.Tile8>}:
  Batch.map_split(~Tile.Tile8,~Tabulation,~entry_map,first,second,index,context,output)

def filled_next(+rest: Nat,+index: Nat,+context: Tabulation,+output: Tree.Tree<Tile.Tile8>) ->
  {filled(1n+rest,U32.from_nat(index),context,output) ==
    filled(rest,U32.from_nat(1n+index),context,Tree.write(Tile.Tile8,output,U32.from_nat(index),entry(context,U32.from_nat(index)))) : Tree.Tree<Tile.Tile8>}:
  %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
    {filled(rest,_,context,Tree.write(Tile.Tile8,output,U32.from_nat(index),entry(context,U32.from_nat(index)))) ==
      filled(rest,U32.from_nat(1n+index),context,Tree.write(Tile.Tile8,output,U32.from_nat(index),entry(context,U32.from_nat(index)))) : Tree.Tree<Tile.Tile8>}
  {==}

def filled_balanced(+count: Nat,+index: U32,+context: Tabulation,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,
  balanced: Shape.Balanced(Tile.Tile8,depth,output)) -> Shape.Balanced(Tile.Tile8,depth,filled(count,index,context,output)):
  MapReadback.balanced_indexed(~Tile.Tile8,~Tabulation,~entry_map,count,index,context,output,depth,balanced)

def filled_capacity(+count: Nat,+index: U32,+context: Tabulation,+output: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,filled(count,index,context,output)) == Tree.capacity(Tile.Tile8,output) : U32}:
  MapReadback.capacity_indexed(~Tile.Tile8,~Tabulation,~entry_map,count,index,context,output)

def entry_at(+geometry: Coordinates.Geometry,+start: U32,+input: Tree.Tree<F32>,+group: Nat,+q: Nat,+index: Nat,+position: Nat,
  indexed: {index == Nat.add(Nat.mul(group,Coordinates.inner(geometry)),q) : Nat},
  positioned: {position == Nat.add(U32.to_nat(start),Nat.mul(group,8n)) : Nat},
  index_room: {Nat.is_le(index,U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool},
  +q_inside: {Nat.is_lt(q,Coordinates.inner(geometry)) == True{} : Bool}) ->
  {entry(Tabulation{input,Coordinates.window(geometry),start},U32.from_nat(index)) ==
    Samples.window_vector(input,Coordinates.window(geometry),U32.from_nat(position),U32.from_nat(q)) : Tile.Tile8}:
  Coordinates.Geometry{window,channels,+cap,+cap_ok,rp,cp,+ip,wp,ie,ce,rr,cr,ar,ir,wr} = geometry
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner_word} = window
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,cap,cap_ok,index_room)) :
    {Samples.window_vector(input,window,U32.from_nat(Nat.add(U32.to_nat(start),Nat.mul(Nat.div(_,Coordinates.inner(geometry)),8n))),U32.from_nat(Nat.mod(_,Coordinates.inner(geometry)))) ==
      Samples.window_vector(input,window,U32.from_nat(position),U32.from_nat(q)) : Tile.Tile8}
  %Equal.sym(Nat,index,Nat.add(Nat.mul(group,Coordinates.inner(geometry)),q),indexed) :
    {Samples.window_vector(input,window,U32.from_nat(Nat.add(U32.to_nat(start),Nat.mul(Nat.div(_,Coordinates.inner(geometry)),8n))),U32.from_nat(Nat.mod(_,Coordinates.inner(geometry)))) ==
      Samples.window_vector(input,window,U32.from_nat(position),U32.from_nat(q)) : Tile.Tile8}
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(group,Coordinates.inner(geometry)),q),Coordinates.inner(geometry)),group,Radix.quotient(group,q,Coordinates.inner(geometry),ip,q_inside)) :
    {Samples.window_vector(input,window,U32.from_nat(Nat.add(U32.to_nat(start),Nat.mul(_,8n))),U32.from_nat(Nat.mod(Nat.add(Nat.mul(group,Coordinates.inner(geometry)),q),Coordinates.inner(geometry)))) ==
      Samples.window_vector(input,window,U32.from_nat(position),U32.from_nat(q)) : Tile.Tile8}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(group,Coordinates.inner(geometry)),q),Coordinates.inner(geometry)),q,Radix.remainder(group,q,Coordinates.inner(geometry),ip,q_inside)) :
    {Samples.window_vector(input,window,U32.from_nat(Nat.add(U32.to_nat(start),Nat.mul(group,8n))),U32.from_nat(_)) ==
      Samples.window_vector(input,window,U32.from_nat(position),U32.from_nat(q)) : Tile.Tile8}
  Equal.cong(Nat,Tile.Tile8,pos => Samples.window_vector(input,window,U32.from_nat(pos),U32.from_nat(q)),Nat.add(U32.to_nat(start),Nat.mul(group,8n)),position,Equal.sym(Nat,position,Nat.add(U32.to_nat(start),Nat.mul(group,8n)),positioned))

def reduction_inside(+geometry: Coordinates.Geometry,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,Coordinates.channels(geometry)) == True{} : Bool},
  row_inside: {Nat.is_lt(row,Coordinates.rows(geometry)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,Coordinates.columns(geometry)) == True{} : Bool}) ->
  {Nat.is_lt(reduction(geometry,channel,row,column),Coordinates.inner(geometry)) == True{} : Bool}:
  Coordinates.Geometry{window,+channels,cap,cap_ok,rp,cp,ip,wp,+ie,ce,rr,cr,ar,ir,wr} = geometry
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner_word} = window
  %Equal.sym(Nat,Coordinates.inner(geometry),Nat.mul(channels,Nat.mul(Coordinates.rows(geometry),Coordinates.columns(geometry))),ie) :
    {Nat.is_lt(reduction(geometry,channel,row,column),_) == True{} : Bool}
  %RowMajor.assoc(channels,Coordinates.rows(geometry),Coordinates.columns(geometry)) :
    {Nat.is_lt(reduction(geometry,channel,row,column),_) == True{} : Bool}
  RowMajor.radix(Nat.add(Nat.mul(channel,Coordinates.rows(geometry)),row),Nat.mul(channels,Coordinates.rows(geometry)),column,Coordinates.columns(geometry),
    RowMajor.radix(channel,channels,row,Coordinates.rows(geometry),channel_inside,row_inside),column_inside)

def row_state(+geometry: Coordinates.Geometry,input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>,index: Nat,channel: Nat,row: Nat,column: Nat,+position: Nat) -> Model.RowTiles():
  Packing.RowTile{input,output,U32.from_nat(index),U32.from_nat(channel),U32.from_nat(row),U32.from_nat(column),
    (U32.from_nat(position) / Coordinates.width_word(geometry) : U32),(U32.from_nat(position) % Coordinates.width_word(geometry) : U32),
    Coordinates.window(geometry),Coordinates.rows(geometry),Coordinates.columns(geometry)}

def selected_lanes(wraps: Bool) -> Model.Lanes():
  match wraps:
    case False{}: Model.row_lanes
    case True{}: Model.wrap_lanes

def lanes(~wraps: Bool) -> Model.Lanes():
  selected_lanes(wraps)

def span(wraps: Bool,+width: Nat) -> Nat:
  match wraps:
    case False{}: width
    case True{}: Nat.add(width,width)

type TileFacts<-domain: Domain,-group: Nat,-position: Nat> is Data:
  TileFacts{inside: {Nat.is_lt(group,groups(domain)) == True{} : Bool},
    located: {position == Nat.add(U32.to_nat(start(domain)),Nat.mul(group,8n)) : Nat}}

def tile_located(+domain: Domain,+group: Nat,+position: Nat,facts: TileFacts<domain,group,position>) ->
  {position == Nat.add(U32.to_nat(start(domain)),Nat.mul(group,8n)) : Nat}:
  TileFacts{inside,located} = facts
  located

def index_room(+domain: Domain,+group: Nat,+position: Nat,+q: Nat,+index: Nat,facts: TileFacts<domain,group,position>,
  located: {index == Nat.add(Nat.mul(group,Coordinates.inner(geometry(domain))),q) : Nat},
  inside: {Nat.is_lt(q,Coordinates.inner(geometry(domain))) == True{} : Bool}) ->
  {Nat.is_le(index,U32.to_nat(Coordinates.capacity(geometry(domain)))) == True{} : Bool}:
  Domain{+geometry,+groups,start,+room,position_room} = domain
  TileFacts{group_inside,positioned} = facts
  %Equal.sym(Nat,index,Nat.add(Nat.mul(group,Coordinates.inner(geometry)),q),located) :
    {Nat.is_le(_,U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool}
  Bounds.strict_le(Nat.add(Nat.mul(group,Coordinates.inner(geometry)),q),U32.to_nat(Coordinates.capacity(geometry)),
    Bounds.nat_trans(Nat.add(Nat.mul(group,Coordinates.inner(geometry)),q),Nat.mul(groups,Coordinates.inner(geometry)),U32.to_nat(Coordinates.capacity(geometry)),
      RowMajor.radix(group,groups,q,Coordinates.inner(geometry),group_inside,inside),room))

def tile_room(+domain: Domain,+group: Nat,+position: Nat,facts: TileFacts<domain,group,position>) ->
  {Nat.is_le(Nat.add(position,8n),U32.to_nat(Coordinates.capacity(geometry(domain)))) == True{} : Bool}:
  Domain{geometry,+groups,+start,packed_room,+position_room} = domain
  TileFacts{inside,located} = facts
  %Equal.sym(Nat,position,Nat.add(U32.to_nat(start),Nat.mul(group,8n)),located) :
    {Nat.is_le(Nat.add(_,8n),U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(start),Nat.mul(group,8n)),8n),Nat.add(U32.to_nat(start),Nat.add(Nat.mul(group,8n),8n)),Addition.associative(U32.to_nat(start),Nat.mul(group,8n),8n)) :
    {Nat.is_le(_,U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool}
  %Order.commutative(8n,Nat.mul(group,8n)) :
    {Nat.is_le(Nat.add(U32.to_nat(start),_),U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool}
  Order.transitive(Nat.add(U32.to_nat(start),Nat.mul(1n+group,8n)),Nat.add(U32.to_nat(start),Nat.mul(groups,8n)),U32.to_nat(Coordinates.capacity(geometry)),
    Order.offset_le(U32.to_nat(start),Nat.mul(1n+group,8n),Nat.mul(groups,8n),RowMajor.mul_le(1n+group,groups,8n,Order.strict_follow(group,groups,inside))),position_room)

def increment_relation(+prefix: Nat,+index: Nat,+q: Nat,located: {index == Nat.add(prefix,q) : Nat}) ->
  {1n+index == Nat.add(prefix,1n+q) : Nat}:
  %Equal.sym(Nat,Nat.add(prefix,1n+q),1n+Nat.add(prefix,q),Addition.successor_right(prefix,q)) : {1n+index == _ : Nat}
  Equal.cong(Nat,Nat,value => 1n+value,index,Nat.add(prefix,q),located)

def column_index(+geometry: Coordinates.Geometry,+group: Nat,+index: Nat,+channel: Nat,+row: Nat,+column: Nat,
  located: {index == Nat.add(Nat.mul(group,Coordinates.inner(geometry)),reduction(geometry,channel,row,column)) : Nat}) ->
  {1n+index == Nat.add(Nat.mul(group,Coordinates.inner(geometry)),reduction(geometry,channel,row,1n+column)) : Nat}:
  %Equal.sym(Nat,reduction(geometry,channel,row,1n+column),1n+reduction(geometry,channel,row,column),
    Addition.successor_right(Nat.mul(Nat.add(Nat.mul(channel,Coordinates.rows(geometry)),row),Coordinates.columns(geometry)),column)) :
    {1n+index == Nat.add(Nat.mul(group,Coordinates.inner(geometry)),_) : Nat}
  increment_relation(Nat.mul(group,Coordinates.inner(geometry)),index,reduction(geometry,channel,row,column),located)

def shifted_count(+prefix: Nat,+index: Nat,+q: Nat,+amount: Nat,located: {index == Nat.add(prefix,q) : Nat}) ->
  {Nat.add(index,amount) == Nat.add(prefix,Nat.add(q,amount)) : Nat}:
  %Addition.associative(prefix,q,amount) : {Nat.add(index,amount) == _ : Nat}
  Equal.cong(Nat,Nat,value => Nat.add(value,amount),index,Nat.add(prefix,q),located)

def row_reduction(+g: Coordinates.Geometry,+channel: Nat,+row: Nat) ->
  {Nat.add(reduction(g,channel,row,0n),Coordinates.columns(g)) == reduction(g,channel,1n+row,0n) : Nat}:
  +prefix = Nat.add(Nat.mul(channel,Coordinates.rows(g)),row)
  +next = Nat.add(Nat.mul(channel,Coordinates.rows(g)),1n+row)
  +columns = Coordinates.columns(g)
  %Equal.sym(Nat,Nat.add(Nat.mul(next,columns),0n),Nat.mul(next,columns),Addition.zero_right(Nat.mul(next,columns))) :
    {Nat.add(reduction(g,channel,row,0n),columns) == _ : Nat}
  %Equal.sym(Nat,next,1n+prefix,Addition.successor_right(Nat.mul(channel,Coordinates.rows(g)),row)) :
    {Nat.add(reduction(g,channel,row,0n),columns) == Nat.mul(_,columns) : Nat}
  %Order.commutative(Nat.mul(prefix,columns),columns) :
    {Nat.add(reduction(g,channel,row,0n),columns) == _ : Nat}
  %Equal.sym(Nat,Nat.add(Nat.mul(prefix,columns),0n),Nat.mul(prefix,columns),Addition.zero_right(Nat.mul(prefix,columns))) :
    {Nat.add(_,columns) == Nat.add(Nat.mul(prefix,columns),columns) : Nat}
  {==}

def row_index(+g: Coordinates.Geometry,+group: Nat,+index: Nat,+channel: Nat,+row: Nat,
  located: {index == Nat.add(Nat.mul(group,Coordinates.inner(g)),reduction(g,channel,row,0n)) : Nat}) ->
  {Nat.add(index,Coordinates.columns(g)) == Nat.add(Nat.mul(group,Coordinates.inner(g)),reduction(g,channel,1n+row,0n)) : Nat}:
  Equal.trans(Nat,Nat.add(index,Coordinates.columns(g)),Nat.add(Nat.mul(group,Coordinates.inner(g)),Nat.add(reduction(g,channel,row,0n),Coordinates.columns(g))),
    Nat.add(Nat.mul(group,Coordinates.inner(g)),reduction(g,channel,1n+row,0n)),
    shifted_count(Nat.mul(group,Coordinates.inner(g)),index,reduction(g,channel,row,0n),Coordinates.columns(g),located),
    Equal.cong(Nat,Nat,q => Nat.add(Nat.mul(group,Coordinates.inner(g)),q),Nat.add(reduction(g,channel,row,0n),Coordinates.columns(g)),reduction(g,channel,1n+row,0n),row_reduction(g,channel,row)))

def next_row_written(+g: Coordinates.Geometry,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+index: Nat,+channel: Nat,+row: Nat,+column: Nat,+position: Nat) ->
  {Model.next_kernel_row(row_state(g,input,output,index,channel,row,column,position)) == row_state(g,input,output,index,channel,1n+row,0n,position) : Model.RowTiles()}:
  Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = g
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner_word} = window
  %Equal.sym(U32,(U32.from_nat(row) + 1 : U32),U32.from_nat(1n+row),Bounds.increment_encoded(row)) :
    {Packing.RowTile{input,output,U32.from_nat(index),U32.from_nat(channel),_,0,(U32.from_nat(position) / ow : U32),(U32.from_nat(position) % ow : U32),window,U32.to_nat(kr),U32.to_nat(kc)} ==
      row_state(g,input,output,index,channel,1n+row,0n,position) : Model.RowTiles()}
  {==}

def plane_reduction(+g: Coordinates.Geometry,+channel: Nat) ->
  {reduction(g,channel,0n,0n) == Nat.mul(channel,Nat.mul(Coordinates.rows(g),Coordinates.columns(g))) : Nat}:
  %RowMajor.assoc(channel,Coordinates.rows(g),Coordinates.columns(g)) : {reduction(g,channel,0n,0n) == _ : Nat}
  %Equal.sym(Nat,Nat.add(Nat.mul(channel,Coordinates.rows(g)),0n),Nat.mul(channel,Coordinates.rows(g)),Addition.zero_right(Nat.mul(channel,Coordinates.rows(g)))) :
    {Nat.add(Nat.mul(_,Coordinates.columns(g)),0n) == Nat.mul(Nat.mul(channel,Coordinates.rows(g)),Coordinates.columns(g)) : Nat}
  Addition.zero_right(Nat.mul(Nat.mul(channel,Coordinates.rows(g)),Coordinates.columns(g)))

def channel_reduction(+g: Coordinates.Geometry,+channel: Nat) ->
  {Nat.add(reduction(g,channel,0n,0n),Nat.mul(Coordinates.rows(g),Coordinates.columns(g))) == reduction(g,1n+channel,0n,0n) : Nat}:
  +area = Nat.mul(Coordinates.rows(g),Coordinates.columns(g))
  %Equal.sym(Nat,reduction(g,1n+channel,0n,0n),Nat.mul(1n+channel,area),plane_reduction(g,1n+channel)) :
    {Nat.add(reduction(g,channel,0n,0n),area) == _ : Nat}
  %Order.commutative(Nat.mul(channel,area),area) :
    {Nat.add(reduction(g,channel,0n,0n),area) == _ : Nat}
  %Equal.sym(Nat,reduction(g,channel,0n,0n),Nat.mul(channel,area),plane_reduction(g,channel)) :
    {Nat.add(_,area) == Nat.add(Nat.mul(channel,area),area) : Nat}
  {==}

def channel_index(+g: Coordinates.Geometry,+group: Nat,+index: Nat,+channel: Nat,
  located: {index == Nat.add(Nat.mul(group,Coordinates.inner(g)),reduction(g,channel,0n,0n)) : Nat}) ->
  {Nat.add(index,Nat.mul(Coordinates.rows(g),Coordinates.columns(g))) == Nat.add(Nat.mul(group,Coordinates.inner(g)),reduction(g,1n+channel,0n,0n)) : Nat}:
  +area = Nat.mul(Coordinates.rows(g),Coordinates.columns(g))
  Equal.trans(Nat,Nat.add(index,area),Nat.add(Nat.mul(group,Coordinates.inner(g)),Nat.add(reduction(g,channel,0n,0n),area)),
    Nat.add(Nat.mul(group,Coordinates.inner(g)),reduction(g,1n+channel,0n,0n)),
    shifted_count(Nat.mul(group,Coordinates.inner(g)),index,reduction(g,channel,0n,0n),area,located),
    Equal.cong(Nat,Nat,q => Nat.add(Nat.mul(group,Coordinates.inner(g)),q),Nat.add(reduction(g,channel,0n,0n),area),reduction(g,1n+channel,0n,0n),channel_reduction(g,channel)))

def next_channel_written(+g: Coordinates.Geometry,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+index: Nat,+channel: Nat,+row: Nat,+column: Nat,+position: Nat) ->
  {Model.next_channel(row_state(g,input,output,index,channel,row,column,position)) == row_state(g,input,output,index,1n+channel,0n,0n,position) : Model.RowTiles()}:
  Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = g
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner_word} = window
  %Equal.sym(U32,(U32.from_nat(channel) + 1 : U32),U32.from_nat(1n+channel),Bounds.increment_encoded(channel)) :
    {Packing.RowTile{input,output,U32.from_nat(index),_,0,0,(U32.from_nat(position) / ow : U32),(U32.from_nat(position) % ow : U32),window,U32.to_nat(kr),U32.to_nat(kc)} ==
      row_state(g,input,output,index,1n+channel,0n,0n,position) : Model.RowTiles()}
  {==}

def column_value(wraps: Bool,+geometry: Coordinates.Geometry,input: Tree.Tree<F32>,channel: Nat,row: Nat,column: Nat,position: Nat) -> Tile.Tile8:
  Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = geometry
  Packing.Stencil{+base,+h,+w,kr,kc,+s,+ro,+co,+ow,inner_word} = window
  +position = U32.from_nat(position)
  +column = U32.from_nat(column)
  +row = U32.from_nat(row)
  selected_lanes(wraps)(input,(position / ow * s + row + ro : U32),(position % ow * s + column + co : U32),
    (base + U32.from_nat(channel) * h * w : U32),h,w,s,((position / ow + 1) * s + row + ro : U32),(position % ow : U32),ow,column,(0 - co : U32))

def column_written(~wraps: Bool,+geometry: Coordinates.Geometry,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+index: Nat,+channel: Nat,+row: Nat,+column: Nat,+position: Nat) ->
  {Model.column_step(~lanes(~wraps),row_state(geometry,input,output,index,channel,row,column,position)) ==
    row_state(geometry,input,Tree.write(Tile.Tile8,output,U32.from_nat(index),column_value(wraps,geometry,input,channel,row,column,position)),1n+index,channel,row,1n+column,position) : Model.RowTiles()}:
  Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = geometry
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner_word} = window
  %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
    {Packing.RowTile{input,Tree.write(Tile.Tile8,output,U32.from_nat(index),column_value(wraps,geometry,input,channel,row,column,position)),_,U32.from_nat(channel),U32.from_nat(row),(U32.from_nat(column) + 1 : U32),
      (U32.from_nat(position) / ow : U32),(U32.from_nat(position) % ow : U32),window,U32.to_nat(kr),U32.to_nat(kc)} ==
      row_state(geometry,input,Tree.write(Tile.Tile8,output,U32.from_nat(index),column_value(wraps,geometry,input,channel,row,column,position)),1n+index,channel,row,1n+column,position) : Model.RowTiles()}
  %Equal.sym(U32,(U32.from_nat(column) + 1 : U32),U32.from_nat(1n+column),Bounds.increment_encoded(column)) :
    {Packing.RowTile{input,Tree.write(Tile.Tile8,output,U32.from_nat(index),column_value(wraps,geometry,input,channel,row,column,position)),U32.from_nat(1n+index),U32.from_nat(channel),U32.from_nat(row),_,
      (U32.from_nat(position) / ow : U32),(U32.from_nat(position) % ow : U32),window,U32.to_nat(kr),U32.to_nat(kc)} ==
      row_state(geometry,input,Tree.write(Tile.Tile8,output,U32.from_nat(index),column_value(wraps,geometry,input,channel,row,column,position)),1n+index,channel,row,1n+column,position) : Model.RowTiles()}
  {==}

def position_encoded(+geometry: Coordinates.Geometry,+position: Nat,
  room: {Nat.is_le(Nat.add(position,8n),U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(position)) == position : Nat}:
  Coordinates.Geometry{window,channels,+cap,+ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = geometry
  Bounds.roundtrip(position,cap,ok,Order.transitive(position,Nat.add(position,8n),U32.to_nat(cap),Order.le_add(position,8n),room))

def column_canonical(wraps: Bool,+geometry: Coordinates.Geometry,+input: Tree.Tree<F32>,+channel: Nat,+row: Nat,+column: Nat,+position: Nat,
  within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(U32.from_nat(position)),Coordinates.width(geometry)),8n),span(wraps,Coordinates.width(geometry))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,8n),U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool},
  channel_inside: {Nat.is_lt(channel,Coordinates.channels(geometry)) == True{} : Bool},
  row_inside: {Nat.is_lt(row,Coordinates.rows(geometry)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,Coordinates.columns(geometry)) == True{} : Bool}) ->
  {column_value(wraps,geometry,input,channel,row,column,position) ==
    Samples.window_vector(input,Coordinates.window(geometry),U32.from_nat(position),U32.from_nat(reduction(geometry,channel,row,column))) : Tile.Tile8}:
  match wraps:
    case False{}:
      Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = geometry
      Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner_word} = window
      %Equal.sym(Nat,reduction(geometry,channel,row,column),Coordinates.reduction(geometry,channel,row,column),RowMajor.mixed(channel,Coordinates.rows(geometry),Coordinates.columns(geometry),row,column)) :
        {column_value(False{},geometry,input,channel,row,column,position) == Samples.window_vector(input,Coordinates.window(geometry),U32.from_nat(position),U32.from_nat(_)) : Tile.Tile8}
      Coordinates.row_entry(geometry,input,U32.from_nat(position),channel,row,column,within,
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,position_encoded(geometry,position,room)) :
          {Nat.is_le(Nat.add(_,8n),U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool} room,
        channel_inside,row_inside,column_inside)
    case True{}:
      Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = geometry
      Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner_word} = window
      %Equal.sym(Nat,reduction(geometry,channel,row,column),Coordinates.reduction(geometry,channel,row,column),RowMajor.mixed(channel,Coordinates.rows(geometry),Coordinates.columns(geometry),row,column)) :
        {column_value(True{},geometry,input,channel,row,column,position) == Samples.window_vector(input,Coordinates.window(geometry),U32.from_nat(position),U32.from_nat(_)) : Tile.Tile8}
      Coordinates.wrap_entry(geometry,input,U32.from_nat(position),channel,row,column,within,
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,position_encoded(geometry,position,room)) :
          {Nat.is_le(Nat.add(_,8n),U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool} room,
        channel_inside,row_inside,column_inside)

def columns_filled(~wraps: Bool,+domain: Domain,+n: Nat,+group: Nat,+channel: Nat,+row: Nat,+column: Nat,+index: Nat,+position: Nat,
  +input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+facts: TileFacts<domain,group,position>,
  +within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(U32.from_nat(position)),Coordinates.width(geometry(domain))),8n),span(wraps,Coordinates.width(geometry(domain)))) == True{} : Bool},
  +indexed: {index == Nat.add(Nat.mul(group,Coordinates.inner(geometry(domain))),reduction(geometry(domain),channel,row,column)) : Nat},
  +channel_inside: {Nat.is_lt(channel,Coordinates.channels(geometry(domain))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,Coordinates.rows(geometry(domain))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(column,n),Coordinates.columns(geometry(domain))) == True{} : Bool}) ->
  {Loop.run(~Model.RowTiles(),~Model.column_step(~lanes(~wraps)),n,row_state(geometry(domain),input,output,index,channel,row,column,position)) ==
    row_state(geometry(domain),input,filled(n,U32.from_nat(index),Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},output),Nat.add(index,n),channel,row,Nat.add(column,n),position) : Model.RowTiles()}:
  match n:
    case 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) :
        {row_state(geometry(domain),input,output,index,channel,row,column,position) == row_state(geometry(domain),input,output,_,channel,row,Nat.add(column,0n),position) : Model.RowTiles()}
      %Equal.sym(Nat,Nat.add(column,0n),column,Addition.zero_right(column)) :
        {row_state(geometry(domain),input,output,index,channel,row,column,position) == row_state(geometry(domain),input,output,index,channel,row,_,position) : Model.RowTiles()}
      {==}
    case 1n+ +rest:
      +g = geometry(domain)
      +context = {Tabulation{input,Coordinates.window(g),start(domain)} : Tabulation}
      +q = reduction(g,channel,row,column)
      +column_inside = Order.room_index(column,rest,Coordinates.columns(g),room)
      +q_inside = reduction_inside(g,channel,row,column,channel_inside,row_inside,column_inside)
      +value = entry(context,U32.from_nat(index))
      +next = Tree.write(Tile.Tile8,output,U32.from_nat(index),value)
      +canonical = Equal.trans(Tile.Tile8,column_value(wraps,g,input,channel,row,column,position),Samples.window_vector(input,Coordinates.window(g),U32.from_nat(position),U32.from_nat(q)),value,
        column_canonical(wraps,g,input,channel,row,column,position,within,tile_room(domain,group,position,facts),channel_inside,row_inside,column_inside),
        Equal.sym(Tile.Tile8,value,Samples.window_vector(input,Coordinates.window(g),U32.from_nat(position),U32.from_nat(q)),
          entry_at(g,start(domain),input,group,q,index,position,indexed,tile_located(domain,group,position,facts),index_room(domain,group,position,q,index,facts,indexed,q_inside),q_inside)))
      %Equal.sym(Model.RowTiles(),Model.column_step(~lanes(~wraps),row_state(g,input,output,index,channel,row,column,position)),
        row_state(g,input,Tree.write(Tile.Tile8,output,U32.from_nat(index),column_value(wraps,g,input,channel,row,column,position)),1n+index,channel,row,1n+column,position),
        column_written(~wraps,g,input,output,index,channel,row,column,position)) :
        {Loop.run(~Model.RowTiles(),~Model.column_step(~lanes(~wraps)),rest,_) == row_state(g,input,filled(1n+rest,U32.from_nat(index),context,output),Nat.add(index,1n+rest),channel,row,Nat.add(column,1n+rest),position) : Model.RowTiles()}
      %Equal.sym(Tile.Tile8,column_value(wraps,g,input,channel,row,column,position),value,canonical) :
        {Loop.run(~Model.RowTiles(),~Model.column_step(~lanes(~wraps)),rest,row_state(g,input,Tree.write(Tile.Tile8,output,U32.from_nat(index),_),1n+index,channel,row,1n+column,position)) ==
          row_state(g,input,filled(1n+rest,U32.from_nat(index),context,output),Nat.add(index,1n+rest),channel,row,Nat.add(column,1n+rest),position) : Model.RowTiles()}
      %Equal.sym(Tree.Tree<Tile.Tile8>,filled(1n+rest,U32.from_nat(index),context,output),filled(rest,U32.from_nat(1n+index),context,next),filled_next(rest,index,context,output)) :
        {Loop.run(~Model.RowTiles(),~Model.column_step(~lanes(~wraps)),rest,row_state(g,input,next,1n+index,channel,row,1n+column,position)) ==
          row_state(g,input,_,Nat.add(index,1n+rest),channel,row,Nat.add(column,1n+rest),position) : Model.RowTiles()}
      %Equal.sym(Nat,Nat.add(index,1n+rest),Nat.add(1n+index,rest),Addition.successor_right(index,rest)) :
        {Loop.run(~Model.RowTiles(),~Model.column_step(~lanes(~wraps)),rest,row_state(g,input,next,1n+index,channel,row,1n+column,position)) ==
          row_state(g,input,filled(rest,U32.from_nat(1n+index),context,next),_,channel,row,Nat.add(column,1n+rest),position) : Model.RowTiles()}
      %Equal.sym(Nat,Nat.add(column,1n+rest),Nat.add(1n+column,rest),Addition.successor_right(column,rest)) :
        {Loop.run(~Model.RowTiles(),~Model.column_step(~lanes(~wraps)),rest,row_state(g,input,next,1n+index,channel,row,1n+column,position)) ==
          row_state(g,input,filled(rest,U32.from_nat(1n+index),context,next),Nat.add(1n+index,rest),channel,row,_,position) : Model.RowTiles()}
      columns_filled(~wraps,domain,rest,group,channel,row,1n+column,1n+index,position,input,next,facts,within,
        column_index(g,group,index,channel,row,column,indexed),channel_inside,row_inside,Order.advance_room(column,rest,Coordinates.columns(g),room))

def rows_filled(~wraps: Bool,+domain: Domain,+n: Nat,+group: Nat,+channel: Nat,+row: Nat,+index: Nat,+position: Nat,
  +input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+facts: TileFacts<domain,group,position>,
  +within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(U32.from_nat(position)),Coordinates.width(geometry(domain))),8n),span(wraps,Coordinates.width(geometry(domain)))) == True{} : Bool},
  +indexed: {index == Nat.add(Nat.mul(group,Coordinates.inner(geometry(domain))),reduction(geometry(domain),channel,row,0n)) : Nat},
  +channel_inside: {Nat.is_lt(channel,Coordinates.channels(geometry(domain))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(row,n),Coordinates.rows(geometry(domain))) == True{} : Bool}) ->
  {Loop.run(~Model.RowTiles(),~Model.kernel_row_step(~lanes(~wraps)),n,row_state(geometry(domain),input,output,index,channel,row,0n,position)) ==
    row_state(geometry(domain),input,filled(Nat.mul(n,Coordinates.columns(geometry(domain))),U32.from_nat(index),Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},output),
      Nat.add(index,Nat.mul(n,Coordinates.columns(geometry(domain)))),channel,Nat.add(row,n),0n,position) : Model.RowTiles()}:
  match n:
    case 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) :
        {row_state(geometry(domain),input,output,index,channel,row,0n,position) == row_state(geometry(domain),input,output,_,channel,Nat.add(row,0n),0n,position) : Model.RowTiles()}
      %Equal.sym(Nat,Nat.add(row,0n),row,Addition.zero_right(row)) :
        {row_state(geometry(domain),input,output,index,channel,row,0n,position) == row_state(geometry(domain),input,output,index,channel,_,0n,position) : Model.RowTiles()}
      {==}
    case 1n+ +rest:
      +g = geometry(domain)
      +cols = Coordinates.columns(g)
      +remaining = Nat.mul(rest,cols)
      +context = {Tabulation{input,Coordinates.window(g),start(domain)} : Tabulation}
      +next = filled(cols,U32.from_nat(index),context,output)
      %Equal.sym(Model.RowTiles(),Loop.run(~Model.RowTiles(),~Model.column_step(~lanes(~wraps)),cols,row_state(g,input,output,index,channel,row,0n,position)),
        row_state(g,input,next,Nat.add(index,cols),channel,row,cols,position),
        columns_filled(~wraps,domain,cols,group,channel,row,0n,index,position,input,output,facts,within,indexed,channel_inside,
          Order.room_index(row,rest,Coordinates.rows(g),room),Order.reflexive(cols))) :
        {Loop.run(~Model.RowTiles(),~Model.kernel_row_step(~lanes(~wraps)),rest,Model.next_kernel_row(_)) ==
          row_state(g,input,filled(Nat.add(cols,remaining),U32.from_nat(index),context,output),Nat.add(index,Nat.add(cols,remaining)),channel,Nat.add(row,1n+rest),0n,position) : Model.RowTiles()}
      %Equal.sym(Model.RowTiles(),Model.next_kernel_row(row_state(g,input,next,Nat.add(index,cols),channel,row,cols,position)),
        row_state(g,input,next,Nat.add(index,cols),channel,1n+row,0n,position),next_row_written(g,input,next,Nat.add(index,cols),channel,row,cols,position)) :
        {Loop.run(~Model.RowTiles(),~Model.kernel_row_step(~lanes(~wraps)),rest,_) ==
          row_state(g,input,filled(Nat.add(cols,remaining),U32.from_nat(index),context,output),Nat.add(index,Nat.add(cols,remaining)),channel,Nat.add(row,1n+rest),0n,position) : Model.RowTiles()}
      %Equal.sym(Tree.Tree<Tile.Tile8>,filled(Nat.add(cols,remaining),U32.from_nat(index),context,output),filled(remaining,U32.from_nat(Nat.add(index,cols)),context,next),filled_split(cols,remaining,index,context,output)) :
        {Loop.run(~Model.RowTiles(),~Model.kernel_row_step(~lanes(~wraps)),rest,row_state(g,input,next,Nat.add(index,cols),channel,1n+row,0n,position)) ==
          row_state(g,input,_,Nat.add(index,Nat.add(cols,remaining)),channel,Nat.add(row,1n+rest),0n,position) : Model.RowTiles()}
      %Addition.associative(index,cols,remaining) :
        {Loop.run(~Model.RowTiles(),~Model.kernel_row_step(~lanes(~wraps)),rest,row_state(g,input,next,Nat.add(index,cols),channel,1n+row,0n,position)) ==
          row_state(g,input,filled(remaining,U32.from_nat(Nat.add(index,cols)),context,next),_,channel,Nat.add(row,1n+rest),0n,position) : Model.RowTiles()}
      %Equal.sym(Nat,Nat.add(row,1n+rest),Nat.add(1n+row,rest),Addition.successor_right(row,rest)) :
        {Loop.run(~Model.RowTiles(),~Model.kernel_row_step(~lanes(~wraps)),rest,row_state(g,input,next,Nat.add(index,cols),channel,1n+row,0n,position)) ==
          row_state(g,input,filled(remaining,U32.from_nat(Nat.add(index,cols)),context,next),Nat.add(Nat.add(index,cols),remaining),channel,_,0n,position) : Model.RowTiles()}
      rows_filled(~wraps,domain,rest,group,channel,1n+row,Nat.add(index,cols),position,input,next,facts,within,
        row_index(g,group,index,channel,row,indexed),channel_inside,Order.advance_room(row,rest,Coordinates.rows(g),room))

def channels_filled(~wraps: Bool,+domain: Domain,+n: Nat,+group: Nat,+channel: Nat,+index: Nat,+position: Nat,
  +input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+facts: TileFacts<domain,group,position>,
  +within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(U32.from_nat(position)),Coordinates.width(geometry(domain))),8n),span(wraps,Coordinates.width(geometry(domain)))) == True{} : Bool},
  +indexed: {index == Nat.add(Nat.mul(group,Coordinates.inner(geometry(domain))),reduction(geometry(domain),channel,0n,0n)) : Nat},
  +room: {Nat.is_le(Nat.add(channel,n),Coordinates.channels(geometry(domain))) == True{} : Bool}) ->
  {Loop.run(~Model.RowTiles(),~Model.channel_step(~lanes(~wraps)),n,row_state(geometry(domain),input,output,index,channel,0n,0n,position)) ==
    row_state(geometry(domain),input,filled(Nat.mul(n,Nat.mul(Coordinates.rows(geometry(domain)),Coordinates.columns(geometry(domain)))),U32.from_nat(index),Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},output),
      Nat.add(index,Nat.mul(n,Nat.mul(Coordinates.rows(geometry(domain)),Coordinates.columns(geometry(domain))))),Nat.add(channel,n),0n,0n,position) : Model.RowTiles()}:
  match n:
    case 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) :
        {row_state(geometry(domain),input,output,index,channel,0n,0n,position) == row_state(geometry(domain),input,output,_,Nat.add(channel,0n),0n,0n,position) : Model.RowTiles()}
      %Equal.sym(Nat,Nat.add(channel,0n),channel,Addition.zero_right(channel)) :
        {row_state(geometry(domain),input,output,index,channel,0n,0n,position) == row_state(geometry(domain),input,output,index,_,0n,0n,position) : Model.RowTiles()}
      {==}
    case 1n+ +rest:
      +g = geometry(domain)
      +rows = Coordinates.rows(g)
      +area = Nat.mul(rows,Coordinates.columns(g))
      +remaining = Nat.mul(rest,area)
      +context = {Tabulation{input,Coordinates.window(g),start(domain)} : Tabulation}
      +next = filled(area,U32.from_nat(index),context,output)
      %Equal.sym(Model.RowTiles(),Loop.run(~Model.RowTiles(),~Model.kernel_row_step(~lanes(~wraps)),rows,row_state(g,input,output,index,channel,0n,0n,position)),
        row_state(g,input,next,Nat.add(index,area),channel,rows,0n,position),
        rows_filled(~wraps,domain,rows,group,channel,0n,index,position,input,output,facts,within,indexed,
          Order.room_index(channel,rest,Coordinates.channels(g),room),Order.reflexive(rows))) :
        {Loop.run(~Model.RowTiles(),~Model.channel_step(~lanes(~wraps)),rest,Model.next_channel(_)) ==
          row_state(g,input,filled(Nat.add(area,remaining),U32.from_nat(index),context,output),Nat.add(index,Nat.add(area,remaining)),Nat.add(channel,1n+rest),0n,0n,position) : Model.RowTiles()}
      %Equal.sym(Model.RowTiles(),Model.next_channel(row_state(g,input,next,Nat.add(index,area),channel,rows,0n,position)),
        row_state(g,input,next,Nat.add(index,area),1n+channel,0n,0n,position),next_channel_written(g,input,next,Nat.add(index,area),channel,rows,0n,position)) :
        {Loop.run(~Model.RowTiles(),~Model.channel_step(~lanes(~wraps)),rest,_) ==
          row_state(g,input,filled(Nat.add(area,remaining),U32.from_nat(index),context,output),Nat.add(index,Nat.add(area,remaining)),Nat.add(channel,1n+rest),0n,0n,position) : Model.RowTiles()}
      %Equal.sym(Tree.Tree<Tile.Tile8>,filled(Nat.add(area,remaining),U32.from_nat(index),context,output),filled(remaining,U32.from_nat(Nat.add(index,area)),context,next),filled_split(area,remaining,index,context,output)) :
        {Loop.run(~Model.RowTiles(),~Model.channel_step(~lanes(~wraps)),rest,row_state(g,input,next,Nat.add(index,area),1n+channel,0n,0n,position)) ==
          row_state(g,input,_,Nat.add(index,Nat.add(area,remaining)),Nat.add(channel,1n+rest),0n,0n,position) : Model.RowTiles()}
      %Addition.associative(index,area,remaining) :
        {Loop.run(~Model.RowTiles(),~Model.channel_step(~lanes(~wraps)),rest,row_state(g,input,next,Nat.add(index,area),1n+channel,0n,0n,position)) ==
          row_state(g,input,filled(remaining,U32.from_nat(Nat.add(index,area)),context,next),_,Nat.add(channel,1n+rest),0n,0n,position) : Model.RowTiles()}
      %Equal.sym(Nat,Nat.add(channel,1n+rest),Nat.add(1n+channel,rest),Addition.successor_right(channel,rest)) :
        {Loop.run(~Model.RowTiles(),~Model.channel_step(~lanes(~wraps)),rest,row_state(g,input,next,Nat.add(index,area),1n+channel,0n,0n,position)) ==
          row_state(g,input,filled(remaining,U32.from_nat(Nat.add(index,area)),context,next),Nat.add(Nat.add(index,area),remaining),_,0n,0n,position) : Model.RowTiles()}
      channels_filled(~wraps,domain,rest,group,1n+channel,Nat.add(index,area),position,input,next,facts,within,
        channel_index(g,group,index,channel,indexed),Order.advance_room(channel,rest,Coordinates.channels(g),room))

def crossing_state(+g: Coordinates.Geometry,input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>,index: Nat,q: Nat,position: Nat) -> Model.Crossings():
  Packing.Crossing{input,output,U32.from_nat(index),U32.from_nat(q),U32.from_nat(position),Coordinates.window(g)}

def crossing_written(+g: Coordinates.Geometry,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+index: Nat,+q: Nat,+position: Nat) ->
  {Model.crossing_step(crossing_state(g,input,output,index,q,position)) ==
    crossing_state(g,input,Tree.write(Tile.Tile8,output,U32.from_nat(index),Samples.window_vector(input,Coordinates.window(g),U32.from_nat(position),U32.from_nat(q))),1n+index,1n+q,position) : Model.Crossings()}:
  %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
    {Packing.Crossing{input,Tree.write(Tile.Tile8,output,U32.from_nat(index),Samples.window_vector(input,Coordinates.window(g),U32.from_nat(position),U32.from_nat(q))),_,(U32.from_nat(q) + 1 : U32),U32.from_nat(position),Coordinates.window(g)} ==
      crossing_state(g,input,Tree.write(Tile.Tile8,output,U32.from_nat(index),Samples.window_vector(input,Coordinates.window(g),U32.from_nat(position),U32.from_nat(q))),1n+index,1n+q,position) : Model.Crossings()}
  %Equal.sym(U32,(U32.from_nat(q) + 1 : U32),U32.from_nat(1n+q),Bounds.increment_encoded(q)) :
    {Packing.Crossing{input,Tree.write(Tile.Tile8,output,U32.from_nat(index),Samples.window_vector(input,Coordinates.window(g),U32.from_nat(position),U32.from_nat(q))),U32.from_nat(1n+index),_,U32.from_nat(position),Coordinates.window(g)} ==
      crossing_state(g,input,Tree.write(Tile.Tile8,output,U32.from_nat(index),Samples.window_vector(input,Coordinates.window(g),U32.from_nat(position),U32.from_nat(q))),1n+index,1n+q,position) : Model.Crossings()}
  {==}

def crossings_filled(+domain: Domain,+n: Nat,+group: Nat,+q: Nat,+index: Nat,+position: Nat,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,
  +facts: TileFacts<domain,group,position>,+indexed: {index == Nat.add(Nat.mul(group,Coordinates.inner(geometry(domain))),q) : Nat},
  +room: {Nat.is_le(Nat.add(q,n),Coordinates.inner(geometry(domain))) == True{} : Bool}) ->
  {Loop.run(~Model.Crossings(),~Model.crossing_step,n,crossing_state(geometry(domain),input,output,index,q,position)) ==
    crossing_state(geometry(domain),input,filled(n,U32.from_nat(index),Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},output),Nat.add(index,n),Nat.add(q,n),position) : Model.Crossings()}:
  match n:
    case 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) :
        {crossing_state(geometry(domain),input,output,index,q,position) == crossing_state(geometry(domain),input,output,_,Nat.add(q,0n),position) : Model.Crossings()}
      %Equal.sym(Nat,Nat.add(q,0n),q,Addition.zero_right(q)) :
        {crossing_state(geometry(domain),input,output,index,q,position) == crossing_state(geometry(domain),input,output,index,_,position) : Model.Crossings()}
      {==}
    case 1n+ +rest:
      +g = geometry(domain)
      +context = {Tabulation{input,Coordinates.window(g),start(domain)} : Tabulation}
      +inside = Order.room_index(q,rest,Coordinates.inner(g),room)
      +value = entry(context,U32.from_nat(index))
      +next = Tree.write(Tile.Tile8,output,U32.from_nat(index),value)
      +canonical = entry_at(g,start(domain),input,group,q,index,position,indexed,tile_located(domain,group,position,facts),index_room(domain,group,position,q,index,facts,indexed,inside),inside)
      %Equal.sym(Model.Crossings(),Model.crossing_step(crossing_state(g,input,output,index,q,position)),
        crossing_state(g,input,Tree.write(Tile.Tile8,output,U32.from_nat(index),Samples.window_vector(input,Coordinates.window(g),U32.from_nat(position),U32.from_nat(q))),1n+index,1n+q,position),crossing_written(g,input,output,index,q,position)) :
        {Loop.run(~Model.Crossings(),~Model.crossing_step,rest,_) == crossing_state(g,input,filled(1n+rest,U32.from_nat(index),context,output),Nat.add(index,1n+rest),Nat.add(q,1n+rest),position) : Model.Crossings()}
      %canonical :
        {Loop.run(~Model.Crossings(),~Model.crossing_step,rest,crossing_state(g,input,Tree.write(Tile.Tile8,output,U32.from_nat(index),_),1n+index,1n+q,position)) ==
          crossing_state(g,input,filled(1n+rest,U32.from_nat(index),context,output),Nat.add(index,1n+rest),Nat.add(q,1n+rest),position) : Model.Crossings()}
      %Equal.sym(Tree.Tree<Tile.Tile8>,filled(1n+rest,U32.from_nat(index),context,output),filled(rest,U32.from_nat(1n+index),context,next),filled_next(rest,index,context,output)) :
        {Loop.run(~Model.Crossings(),~Model.crossing_step,rest,crossing_state(g,input,next,1n+index,1n+q,position)) == crossing_state(g,input,_,Nat.add(index,1n+rest),Nat.add(q,1n+rest),position) : Model.Crossings()}
      %Equal.sym(Nat,Nat.add(index,1n+rest),Nat.add(1n+index,rest),Addition.successor_right(index,rest)) :
        {Loop.run(~Model.Crossings(),~Model.crossing_step,rest,crossing_state(g,input,next,1n+index,1n+q,position)) == crossing_state(g,input,filled(rest,U32.from_nat(1n+index),context,next),_,Nat.add(q,1n+rest),position) : Model.Crossings()}
      %Equal.sym(Nat,Nat.add(q,1n+rest),Nat.add(1n+q,rest),Addition.successor_right(q,rest)) :
        {Loop.run(~Model.Crossings(),~Model.crossing_step,rest,crossing_state(g,input,next,1n+index,1n+q,position)) == crossing_state(g,input,filled(rest,U32.from_nat(1n+index),context,next),Nat.add(1n+index,rest),_,position) : Model.Crossings()}
      crossings_filled(domain,rest,group,1n+q,1n+index,position,input,next,facts,increment_relation(Nat.mul(group,Coordinates.inner(g)),index,q,indexed),Order.advance_room(q,rest,Coordinates.inner(g),room))

def tiled(+g: Coordinates.Geometry,input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>,index: Nat,position: Nat) -> Model.Tiled():
  Packing.Tiles{input,output,U32.from_nat(index),U32.from_nat(position),Coordinates.window(g),Coordinates.channels(g),Coordinates.rows(g),Coordinates.columns(g),Coordinates.inner(g)}

def inner_equation(+g: Coordinates.Geometry) -> {Coordinates.inner(g) == Nat.mul(Coordinates.channels(g),Nat.mul(Coordinates.rows(g),Coordinates.columns(g))) : Nat}:
  Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = g
  ie

def channels_complete(~wraps: Bool,+domain: Domain,+group: Nat,+index: Nat,+position: Nat,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,
  +facts: TileFacts<domain,group,position>,+indexed: {index == Nat.add(Nat.mul(group,Coordinates.inner(geometry(domain))),0n) : Nat},
  +within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(U32.from_nat(position)),Coordinates.width(geometry(domain))),8n),span(wraps,Coordinates.width(geometry(domain)))) == True{} : Bool}) ->
  {Loop.run(~Model.RowTiles(),~Model.channel_step(~lanes(~wraps)),Coordinates.channels(geometry(domain)),row_state(geometry(domain),input,output,index,0n,0n,0n,position)) ==
    row_state(geometry(domain),input,filled(Coordinates.inner(geometry(domain)),U32.from_nat(index),Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},output),Nat.add(index,Coordinates.inner(geometry(domain))),Coordinates.channels(geometry(domain)),0n,0n,position) : Model.RowTiles()}:
  +g = geometry(domain)
  %Equal.sym(Nat,Coordinates.inner(g),Nat.mul(Coordinates.channels(g),Nat.mul(Coordinates.rows(g),Coordinates.columns(g))),inner_equation(g)) :
    {Loop.run(~Model.RowTiles(),~Model.channel_step(~lanes(~wraps)),Coordinates.channels(g),row_state(g,input,output,index,0n,0n,0n,position)) ==
      row_state(g,input,filled(_,U32.from_nat(index),Tabulation{input,Coordinates.window(g),start(domain)},output),Nat.add(index,_),Coordinates.channels(g),0n,0n,position) : Model.RowTiles()}
  channels_filled(~wraps,domain,Coordinates.channels(g),group,0n,index,position,input,output,facts,within,indexed,Order.reflexive(Coordinates.channels(g)))

def after_row_written(+g: Coordinates.Geometry,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+index: Nat,+channel: Nat,+row: Nat,+column: Nat,+position: Nat) ->
  {Model.after_row_tile(U32.from_nat(position),Coordinates.channels(g),Coordinates.inner(g),row_state(g,input,output,index,channel,row,column,position)) == tiled(g,input,output,index,Nat.add(position,8n)) : Model.Tiled()}:
  %Equal.sym(U32,(U32.from_nat(position) + 8 : U32),U32.from_nat(Nat.add(position,8n)),WordArithmetic.native_add(position,8n)) :
    {Packing.Tiles{input,output,U32.from_nat(index),_,Coordinates.window(g),Coordinates.channels(g),Coordinates.rows(g),Coordinates.columns(g),Coordinates.inner(g)} == tiled(g,input,output,index,Nat.add(position,8n)) : Model.Tiled()}
  {==}

def after_crossing_written(+g: Coordinates.Geometry,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+index: Nat,+q: Nat,+position: Nat) ->
  {Model.after_crossing(Coordinates.channels(g),Coordinates.rows(g),Coordinates.columns(g),Coordinates.inner(g),crossing_state(g,input,output,index,q,position)) == tiled(g,input,output,index,Nat.add(position,8n)) : Model.Tiled()}:
  %Equal.sym(U32,(U32.from_nat(position) + 8 : U32),U32.from_nat(Nat.add(position,8n)),WordArithmetic.native_add(position,8n)) :
    {Packing.Tiles{input,output,U32.from_nat(index),_,Coordinates.window(g),Coordinates.channels(g),Coordinates.rows(g),Coordinates.columns(g),Coordinates.inner(g)} == tiled(g,input,output,index,Nat.add(position,8n)) : Model.Tiled()}
  {==}

def tile_native_room(+domain: Domain,+group: Nat,+position: Nat,+facts: TileFacts<domain,group,position>) ->
  {Nat.is_le(Nat.add(U32.to_nat(U32.from_nat(position)),8n),U32.to_nat(Coordinates.capacity(geometry(domain)))) == True{} : Bool}:
  +room = tile_room(domain,group,position,facts)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,position_encoded(geometry(domain),position,room)) :
    {Nat.is_le(Nat.add(_,8n),U32.to_nat(Coordinates.capacity(geometry(domain)))) == True{} : Bool}
  room

def tile_filled(within: Bool,wraps: Bool,+domain: Domain,+group: Nat,+index: Nat,+position: Nat,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,
  +facts: TileFacts<domain,group,position>,+indexed: {index == Nat.add(Nat.mul(group,Coordinates.inner(geometry(domain))),0n) : Nat},
  within_exact: {U32.is_le((U32.from_nat(position) % Coordinates.width_word(geometry(domain)) + 8 : U32),Coordinates.width_word(geometry(domain))) == within : Bool},
  wraps_exact: {U32.is_le((U32.from_nat(position) % Coordinates.width_word(geometry(domain)) + 8 : U32),(Coordinates.width_word(geometry(domain)) + Coordinates.width_word(geometry(domain)) : U32)) == wraps : Bool}) ->
  {Model.tile(within,wraps,input,output,U32.from_nat(index),U32.from_nat(position),Coordinates.window(geometry(domain)),Coordinates.channels(geometry(domain)),Coordinates.rows(geometry(domain)),Coordinates.columns(geometry(domain)),Coordinates.inner(geometry(domain))) ==
    tiled(geometry(domain),input,filled(Coordinates.inner(geometry(domain)),U32.from_nat(index),Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},output),Nat.add(index,Coordinates.inner(geometry(domain))),Nat.add(position,8n)) : Model.Tiled()}:
  match within wraps:
    case True{} wraps:
      Domain{+g,groups,start,packed_room,position_room} = domain
      Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = g
      Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner_word} = window
      +next = filled(Coordinates.inner(g),U32.from_nat(index),Tabulation{input,window,start},output)
      %Equal.sym(Model.RowTiles(),Loop.run(~Model.RowTiles(),~Model.channel_step(~Model.row_lanes),Coordinates.channels(g),row_state(g,input,output,index,0n,0n,0n,position)),
        row_state(g,input,next,Nat.add(index,Coordinates.inner(g)),Coordinates.channels(g),0n,0n,position),
        channels_complete(~False{},domain,group,index,position,input,output,facts,indexed,Coordinates.within_columns(g,U32.from_nat(position),tile_native_room(domain,group,position,facts),within_exact))) :
        {Model.after_row_tile(U32.from_nat(position),Coordinates.channels(g),Coordinates.inner(g),_) == tiled(g,input,next,Nat.add(index,Coordinates.inner(g)),Nat.add(position,8n)) : Model.Tiled()}
      after_row_written(g,input,next,Nat.add(index,Coordinates.inner(g)),Coordinates.channels(g),0n,0n,position)
    case False{} True{}:
      Domain{+g,groups,start,packed_room,position_room} = domain
      Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = g
      Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner_word} = window
      +next = filled(Coordinates.inner(g),U32.from_nat(index),Tabulation{input,window,start},output)
      %Equal.sym(Model.RowTiles(),Loop.run(~Model.RowTiles(),~Model.channel_step(~Model.wrap_lanes),Coordinates.channels(g),row_state(g,input,output,index,0n,0n,0n,position)),
        row_state(g,input,next,Nat.add(index,Coordinates.inner(g)),Coordinates.channels(g),0n,0n,position),
        channels_complete(~True{},domain,group,index,position,input,output,facts,indexed,Coordinates.wrap_columns(g,U32.from_nat(position),tile_native_room(domain,group,position,facts),wraps_exact))) :
        {Model.after_row_tile(U32.from_nat(position),Coordinates.channels(g),Coordinates.inner(g),_) == tiled(g,input,next,Nat.add(index,Coordinates.inner(g)),Nat.add(position,8n)) : Model.Tiled()}
      after_row_written(g,input,next,Nat.add(index,Coordinates.inner(g)),Coordinates.channels(g),0n,0n,position)
    case False{} False{}:
      Domain{+g,groups,start,packed_room,position_room} = domain
      Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = g
      Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner_word} = window
      +next = filled(Coordinates.inner(g),U32.from_nat(index),Tabulation{input,window,start},output)
      %Equal.sym(Model.Crossings(),Loop.run(~Model.Crossings(),~Model.crossing_step,Coordinates.inner(g),crossing_state(g,input,output,index,0n,position)),
        crossing_state(g,input,next,Nat.add(index,Coordinates.inner(g)),Coordinates.inner(g),position),
        crossings_filled(domain,Coordinates.inner(g),group,0n,index,position,input,output,facts,indexed,Order.reflexive(Coordinates.inner(g)))) :
        {Model.after_crossing(Coordinates.channels(g),Coordinates.rows(g),Coordinates.columns(g),Coordinates.inner(g),_) == tiled(g,input,next,Nat.add(index,Coordinates.inner(g)),Nat.add(position,8n)) : Model.Tiled()}
      after_crossing_written(g,input,next,Nat.add(index,Coordinates.inner(g)),Coordinates.inner(g),position)

def tile_step_filled(+domain: Domain,+group: Nat,+index: Nat,+position: Nat,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,
  +facts: TileFacts<domain,group,position>,+indexed: {index == Nat.add(Nat.mul(group,Coordinates.inner(geometry(domain))),0n) : Nat}) ->
  {Model.tile_step(tiled(geometry(domain),input,output,index,position)) ==
    tiled(geometry(domain),input,filled(Coordinates.inner(geometry(domain)),U32.from_nat(index),Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},output),Nat.add(index,Coordinates.inner(geometry(domain))),Nat.add(position,8n)) : Model.Tiled()}:
  Domain{+g,groups,start,packed_room,position_room} = domain
  Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = g
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,+ow,inner_word} = window
  tile_filled(U32.is_le((U32.from_nat(position) % ow + 8 : U32),ow),U32.is_le((U32.from_nat(position) % ow + 8 : U32),(ow + ow : U32)),domain,group,index,position,input,output,facts,indexed,{==},{==})

def next_group_index(+g: Coordinates.Geometry,+group: Nat,+index: Nat,indexed: {index == Nat.add(Nat.mul(group,Coordinates.inner(g)),0n) : Nat}) ->
  {Nat.add(index,Coordinates.inner(g)) == Nat.add(Nat.mul(1n+group,Coordinates.inner(g)),0n) : Nat}:
  +inner = Coordinates.inner(g)
  %Equal.sym(Nat,Nat.add(Nat.mul(1n+group,inner),0n),Nat.mul(1n+group,inner),Addition.zero_right(Nat.mul(1n+group,inner))) :
    {Nat.add(index,inner) == _ : Nat}
  %Equal.sym(Nat,Nat.add(inner,Nat.mul(group,inner)),Nat.add(Nat.mul(group,inner),inner),Order.commutative(inner,Nat.mul(group,inner))) :
    {Nat.add(index,inner) == _ : Nat}
  %Equal.sym(Nat,index,Nat.mul(group,inner),Equal.trans(Nat,index,Nat.add(Nat.mul(group,inner),0n),Nat.mul(group,inner),indexed,Addition.zero_right(Nat.mul(group,inner)))) :
    {Nat.add(_,inner) == Nat.add(Nat.mul(group,inner),inner) : Nat}
  {==}

def next_position(+start: U32,+group: Nat,+position: Nat,located: {position == Nat.add(U32.to_nat(start),Nat.mul(group,8n)) : Nat}) ->
  {Nat.add(position,8n) == Nat.add(U32.to_nat(start),Nat.mul(1n+group,8n)) : Nat}:
  %Equal.sym(Nat,Nat.add(8n,Nat.mul(group,8n)),Nat.add(Nat.mul(group,8n),8n),Order.commutative(8n,Nat.mul(group,8n))) :
    {Nat.add(position,8n) == Nat.add(U32.to_nat(start),_) : Nat}
  %Addition.associative(U32.to_nat(start),Nat.mul(group,8n),8n) : {Nat.add(position,8n) == _ : Nat}
  %Equal.sym(Nat,position,Nat.add(U32.to_nat(start),Nat.mul(group,8n)),located) : {Nat.add(_,8n) == Nat.add(Nat.add(U32.to_nat(start),Nat.mul(group,8n)),8n) : Nat}
  {==}

def tiles_filled(+domain: Domain,+n: Nat,+group: Nat,+index: Nat,+position: Nat,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,
  +indexed: {index == Nat.add(Nat.mul(group,Coordinates.inner(geometry(domain))),0n) : Nat},
  +located: {position == Nat.add(U32.to_nat(start(domain)),Nat.mul(group,8n)) : Nat},
  +room: {Nat.is_le(Nat.add(group,n),groups(domain)) == True{} : Bool}) ->
  {Model.finish_tiles(Loop.run(~Model.Tiled(),~Model.tile_step,n,tiled(geometry(domain),input,output,index,position))) ==
    Model.Packed{input,filled(Nat.mul(n,Coordinates.inner(geometry(domain))),U32.from_nat(index),Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},output)} : Model.Packed}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +g = geometry(domain)
      +inner = Coordinates.inner(g)
      +remaining = Nat.mul(rest,inner)
      +context = {Tabulation{input,Coordinates.window(g),start(domain)} : Tabulation}
      +next = filled(inner,U32.from_nat(index),context,output)
      +facts = {TileFacts{Order.room_index(group,rest,groups(domain),room),located} : TileFacts<domain,group,position>}
      %Equal.sym(Model.Tiled(),Model.tile_step(tiled(g,input,output,index,position)),tiled(g,input,next,Nat.add(index,inner),Nat.add(position,8n)),tile_step_filled(domain,group,index,position,input,output,facts,indexed)) :
        {Model.finish_tiles(Loop.run(~Model.Tiled(),~Model.tile_step,rest,_)) == Model.Packed{input,filled(Nat.add(inner,remaining),U32.from_nat(index),context,output)} : Model.Packed}
      %Equal.sym(Tree.Tree<Tile.Tile8>,filled(Nat.add(inner,remaining),U32.from_nat(index),context,output),filled(remaining,U32.from_nat(Nat.add(index,inner)),context,next),filled_split(inner,remaining,index,context,output)) :
        {Model.finish_tiles(Loop.run(~Model.Tiled(),~Model.tile_step,rest,tiled(g,input,next,Nat.add(index,inner),Nat.add(position,8n)))) == Model.Packed{input,_} : Model.Packed}
      tiles_filled(domain,rest,1n+group,Nat.add(index,inner),Nat.add(position,8n),input,next,
        next_group_index(g,group,index,indexed),next_position(start(domain),group,position,located),Order.advance_room(group,rest,groups(domain),room))

def initial_matches(+domain: Domain,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>) ->
  {Model.initial_window(Coordinates.window(geometry(domain)),start(domain),input,output) == tiled(geometry(domain),input,output,0n,U32.to_nat(start(domain))) : Model.Tiled()}:
  Domain{+g,groups,+start,packed_room,position_room} = domain
  Coordinates.Geometry{window,+channels,cap,ok,rp,cp,ip,wp,ie,+ce,rr,cr,ar,ir,wr} = g
  Packing.Stencil{base,h,w,+kr,+kc,s,ro,co,ow,+inner_word} = window
  %Equal.sym(Nat,U32.to_nat((inner_word / (kr * kc) : U32)),channels,ce) :
    {Packing.Tiles{input,output,0,start,window,_,U32.to_nat(kr),U32.to_nat(kc),U32.to_nat(inner_word)} == tiled(g,input,output,0n,U32.to_nat(start)) : Model.Tiled()}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(start)),start,Arithmetic.native_roundtrip(start)) :
    {Packing.Tiles{input,output,0,start,window,channels,U32.to_nat(kr),U32.to_nat(kc),U32.to_nat(inner_word)} == Packing.Tiles{input,output,0,_,window,channels,U32.to_nat(kr),U32.to_nat(kc),U32.to_nat(inner_word)} : Model.Tiled()}
  {==}

def packed_prefix(+domain: Domain,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>) ->
  {Model.packed_window(Coordinates.window(geometry(domain)),count(domain),start(domain),input,output) ==
    Model.Packed{input,filled(count(domain),0,Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},output)} : Model.Packed}:
  Domain{+g,+groups,+start,packed_room,position_room} = domain
  Coordinates.Geometry{window,channels,cap,ok,rp,cp,+ip,wp,ie,ce,rr,cr,ar,ir,wr} = g
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner_word} = window
  %Equal.sym(Nat,Nat.div(Nat.mul(groups,Coordinates.inner(g)),Coordinates.inner(g)),groups,Division.exact_quotient(groups,Coordinates.inner(g),ip)) :
    {Model.finish_tiles(Loop.run(~Model.Tiled(),~Model.tile_step,_,Model.initial_window(window,start,input,output))) == Model.Packed{input,filled(count(domain),0,Tabulation{input,window,start},output)} : Model.Packed}
  %Equal.sym(Model.Tiled(),Model.initial_window(window,start,input,output),tiled(g,input,output,0n,U32.to_nat(start)),initial_matches(domain,input,output)) :
    {Model.finish_tiles(Loop.run(~Model.Tiled(),~Model.tile_step,groups,_)) == Model.Packed{input,filled(count(domain),0,Tabulation{input,window,start},output)} : Model.Packed}
  tiles_filled(domain,groups,0n,0n,U32.to_nat(start),input,output,{==},Equal.sym(Nat,Nat.add(U32.to_nat(start),0n),U32.to_nat(start),Addition.zero_right(U32.to_nat(start))),Order.reflexive(groups))

def packed_readback(+domain: Domain,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,+index: Nat,
  capacity_equal: {Tree.capacity(Tile.Tile8,output) == capacity : U32},balanced: Shape.Balanced(Tile.Tile8,depth,output),
  depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(count(domain),U32.to_nat(capacity)) == True{} : Bool},selected: {Nat.is_lt(index,count(domain)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Model.packed_output(Model.packed_window(Coordinates.window(geometry(domain)),count(domain),start(domain),input,output)),U32.from_nat(index)) ==
    entry(Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},U32.from_nat(index)) : Tile.Tile8}:
  %Equal.sym(Model.Packed,Model.packed_window(Coordinates.window(geometry(domain)),count(domain),start(domain),input,output),
    Model.Packed{input,filled(count(domain),0,Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},output)},packed_prefix(domain,input,output)) :
    {Tree.read(Tile.Tile8,Model.packed_output(_),U32.from_nat(index)) == entry(Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},U32.from_nat(index)) : Tile.Tile8}
  MapReadback.inside_indexed(~Tile.Tile8,~Tabulation,~entry_map,count(domain),0n,index,Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},output,depth,capacity,
    capacity_equal,balanced,Order.successor(depth,30n,depth_ok),capacity_ok,room,selected)

def packed_native_readback(+domain: Domain,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,+index: Nat,
  capacity_equal: {Tree.capacity(Tile.Tile8,output) == capacity : U32},balanced: Shape.Balanced(Tile.Tile8,depth,output),
  depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(count(domain),U32.to_nat(capacity)) == True{} : Bool},selected: {Nat.is_lt(index,count(domain)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Model.packed_output(Model.packed_window(Coordinates.window(geometry(domain)),count(domain),start(domain),input,output)),U32.from_nat(index)) ==
    native_entry(Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},U32.from_nat(index)) : Tile.Tile8}:
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Model.packed_output(Model.packed_window(Coordinates.window(geometry(domain)),count(domain),start(domain),input,output)),U32.from_nat(index)),
    entry(Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},U32.from_nat(index)),native_entry(Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},U32.from_nat(index)),
    packed_readback(domain,input,output,depth,capacity,index,capacity_equal,balanced,depth_ok,capacity_ok,room,selected),entry_native(geometry(domain),start(domain),input,U32.from_nat(index)))

def native_sample(+domain: Domain,input: Tree.Tree<F32>,+index: U32,lane: Sequence.Index(8n)) -> F32:
  +g = geometry(domain)
  Samples.window_sample(input,Samples.window_context(Coordinates.window(g),(start(domain) + index / Coordinates.inner_word(g) * 8 : U32),(index % Coordinates.inner_word(g) : U32)),Sequence.index_value(8n,lane))

def window_lane(+input: Tree.Tree<F32>,+window: Packing.Stencil,+position: U32,+reduction: U32,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,Samples.window_vector(input,window,position,reduction)) ==
    Samples.window_sample(input,Samples.window_context(window,position,reduction),Sequence.index_value(8n,lane)) : F32}:
  match lane:
    case Sequence.First{}: {==}
    case Sequence.Next{+tail}:
      %Addition.successor_width(Sequence.index_value(7n,tail)) :
        {Tile.element_at(Sequence.Next{tail},Samples.window_vector(input,window,position,reduction)) ==
          Samples.window_sample(input,Samples.window_context(window,position,reduction),_) : F32}
      SequenceLaws.get_tabulated(~F32,~TileLaws.SampleContext<Tree.Tree<F32>,PackingSamples.WindowPixelBlock>,
        ~TileLaws.sample_context(~Tree.Tree<F32>,~PackingSamples.WindowPixelBlock,~Samples.window_sample),7n,tail,1n,
        TileLaws.SampleContext{input,Samples.window_context(window,position,reduction)})


def native_entry_lane(+domain: Domain,+input: Tree.Tree<F32>,+index: U32,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,native_entry(Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},index)) == native_sample(domain,input,index,lane) : F32}:
  Domain{+g,groups,+start,packed_room,position_room} = domain
  Coordinates.Geometry{window,channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = g
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner_word} = window
  window_lane(input,window,(start + index / inner_word * 8 : U32),(index % inner_word : U32),lane)

def packed_native_lane(+domain: Domain,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,+index: Nat,+lane: Sequence.Index(8n),
  capacity_equal: {Tree.capacity(Tile.Tile8,output) == capacity : U32},balanced: Shape.Balanced(Tile.Tile8,depth,output),
  depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(count(domain),U32.to_nat(capacity)) == True{} : Bool},selected: {Nat.is_lt(index,count(domain)) == True{} : Bool}) ->
  {Tile.element_at(lane,Tree.read(Tile.Tile8,Model.packed_output(Model.packed_window(Coordinates.window(geometry(domain)),count(domain),start(domain),input,output)),U32.from_nat(index))) ==
    native_sample(domain,input,U32.from_nat(index),lane) : F32}:
  Equal.trans(F32,Tile.element_at(lane,Tree.read(Tile.Tile8,Model.packed_output(Model.packed_window(Coordinates.window(geometry(domain)),count(domain),start(domain),input,output)),U32.from_nat(index))),
    Tile.element_at(lane,native_entry(Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},U32.from_nat(index))),native_sample(domain,input,U32.from_nat(index),lane),
    Equal.cong(Tile.Tile8,F32,tile => Tile.element_at(lane,tile),Tree.read(Tile.Tile8,Model.packed_output(Model.packed_window(Coordinates.window(geometry(domain)),count(domain),start(domain),input,output)),U32.from_nat(index)),
      native_entry(Tabulation{input,Coordinates.window(geometry(domain)),start(domain)},U32.from_nat(index)),packed_native_readback(domain,input,output,depth,capacity,index,capacity_equal,balanced,depth_ok,capacity_ok,room,selected)),
    native_entry_lane(domain,input,U32.from_nat(index),lane))
