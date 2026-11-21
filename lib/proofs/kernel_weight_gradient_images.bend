# The weight kernel over the images of a batch: each image lends the band of
# its input cells and of its gradient cells from the pieces the kernel holds,
# sums its groups over them and gives the bands back. A lane of the sums of
# an image is the adjacent tree of the products of its groups over the cells
# the holdings hold.
import Base
import ../kernel_weight_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../tensor_move.bend as Move
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../traversal_array.bend as Traversal
import ../traversal_grouped.bend as Grouped
import ../traversal_loop.bend as Loop
import ../machine_limits.bend as Machine
import ../kernel_gemm.bend as KernelGemm
import ./kernel_weight_gradient_group.bend as Groups
import ./kernel_weight_gradient_packing.bend as Packing
import ./kernel_weight_gradient_refinement.bend as Refinement
import ./kernel_weight_gradient_held.bend as Sources
import ./tensor_move_readback.bend as Held
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./kernel_tile_refinement.bend as TileLaws
import ./traversal_array_refinement.bend as Traversals
import ./grouped_reduction_refinement.bend as GroupLaws
import ./sequence_membership.bend as Members
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./word_arithmetic.bend as WordArithmetic

# ---- One image ----

def image_banded_of(+geometry: Kernel.Geometry,+index: U32,+ranges: List<&2,Grouped.Range>,+x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>) ->
  {Kernel.image(geometry,index,ranges,Held.pack(~Tile.Tile8,x),Held.pack(~Tile.Tile8,d)) ==
    Kernel.image_banded(geometry,ranges,Sources.input_band(x,geometry,index),Sources.gradient_band(d,geometry,index)) : Kernel.Held() & Array<Tile.Tile8>}:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  {==}

def image_eta(+geometry: Kernel.Geometry,+ranges: List<&2,Grouped.Range>,x: Move.Band<Tile.Tile8>,d: Move.Band<Tile.Tile8>) ->
  {Kernel.image_banded(geometry,ranges,x,d) ==
    Kernel.reduced(List.length(&2,Grouped.Range,ranges),Kernel.cells(geometry),Kernel.image_returned(Held.band_loan(~Tile.Tile8,x),Held.band_loan(~Tile.Tile8,d),
      Kernel.groups(ranges,geometry,Held.band_offset(~Tile.Tile8,x),Held.band_offset(~Tile.Tile8,d),Held.band_buffer(~Tile.Tile8,x),Held.band_buffer(~Tile.Tile8,d)))) :
    Kernel.Held() & Array<Tile.Tile8>}:
  Move.Band{x_lent,x_band,x_base} = x
  Move.Band{d_lent,d_band,d_base} = d
  {==}

# The group sums of an image over its bands, their adjacent tree, and the
# holdings once the bands are back.
def image_list(+geometry: Kernel.Geometry,+ranges: List<&2,Grouped.Range>,+x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>,+index: U32) -> List<&2,Tree.Tree<Tile.Tile8>>:
  Pair.snd(Groups.Sources,List<&2,Tree.Tree<Tile.Tile8>>,Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~Kernel.GroupContext,~Refinement.group_read,ranges,
    Groups.Sources{Sources.band_model(Sources.input_band(x,geometry,index)),Sources.band_model(Sources.gradient_band(d,geometry,index))},
    Kernel.GroupContext{geometry,Held.band_offset(~Tile.Tile8,Sources.input_band(x,geometry,index)),Held.band_offset(~Tile.Tile8,Sources.gradient_band(d,geometry,index))}))

def image_tree(+geometry: Kernel.Geometry,+ranges: List<&2,Grouped.Range>,+x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>,+index: U32) -> Tree.Tree<Tile.Tile8>:
  Grouped.levels(~&2,~Tree.Tree<Tile.Tile8>,~U32,~Refinement.zero,~{count => left => right => Refinement.sum(count,left,right) : U32 -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8>},
    List.length(&2,Grouped.Range,ranges),Kernel.cells(geometry),image_list(geometry,ranges,x,d,index))

def returned_input(+geometry: Kernel.Geometry,+x: Held.Holding<Tile.Tile8>,+index: U32) -> Move.Holding<Tile.Tile8>:
  Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Sources.input_band(x,geometry,index)),Buffer.pack(~Tile.Tile8,Sources.band_model(Sources.input_band(x,geometry,index))))

def returned_gradient(+geometry: Kernel.Geometry,+d: Held.Holding<Tile.Tile8>,+index: U32) -> Move.Holding<Tile.Tile8>:
  Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Sources.gradient_band(d,geometry,index)),Buffer.pack(~Tile.Tile8,Sources.band_model(Sources.gradient_band(d,geometry,index))))

# An image gives back its bands with the cells it was lent and the adjacent
# tree of its group sums.
def image_matches(+geometry: Kernel.Geometry,+index: U32,+ranges: List<&2,Grouped.Range>,+x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>) ->
  {Kernel.image(geometry,index,ranges,Held.pack(~Tile.Tile8,x),Held.pack(~Tile.Tile8,d)) ==
    ((returned_input(geometry,x,index),returned_gradient(geometry,d,index)),Tree.pack(Tile.Tile8,image_tree(geometry,ranges,x,d,index))) : Kernel.Held() & Array<Tile.Tile8>}:
  +x_offset = Held.band_offset(~Tile.Tile8,Sources.input_band(x,geometry,index))
  +d_offset = Held.band_offset(~Tile.Tile8,Sources.gradient_band(d,geometry,index))
  +x_model = Sources.band_model(Sources.input_band(x,geometry,index))
  +d_model = Sources.band_model(Sources.gradient_band(d,geometry,index))
  +list = image_list(geometry,ranges,x,d,index)
  +fuel = List.length(&2,Grouped.Range,ranges)
  +count = Kernel.cells(geometry)
  +grouped = Equal.trans((Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>,
    Kernel.groups(ranges,geometry,x_offset,d_offset,Held.band_buffer(~Tile.Tile8,Sources.input_band(x,geometry,index)),Held.band_buffer(~Tile.Tile8,Sources.gradient_band(d,geometry,index))),
    Kernel.groups(ranges,geometry,x_offset,d_offset,Buffer.pack(~Tile.Tile8,x_model),Buffer.pack(~Tile.Tile8,d_model)),
    Refinement.represent_list(Refinement.GroupList{x_model,d_model,list}),
    Equal.trans((Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>,
      Kernel.groups(ranges,geometry,x_offset,d_offset,Held.band_buffer(~Tile.Tile8,Sources.input_band(x,geometry,index)),Held.band_buffer(~Tile.Tile8,Sources.gradient_band(d,geometry,index))),
      Kernel.groups(ranges,geometry,x_offset,d_offset,Buffer.pack(~Tile.Tile8,x_model),Held.band_buffer(~Tile.Tile8,Sources.gradient_band(d,geometry,index))),
      Kernel.groups(ranges,geometry,x_offset,d_offset,Buffer.pack(~Tile.Tile8,x_model),Buffer.pack(~Tile.Tile8,d_model)),
      Equal.cong(Storage.Buffer<Tile.Tile8>,(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>,
        buffer => Kernel.groups(ranges,geometry,x_offset,d_offset,buffer,Held.band_buffer(~Tile.Tile8,Sources.gradient_band(d,geometry,index))),
        Held.band_buffer(~Tile.Tile8,Sources.input_band(x,geometry,index)),Buffer.pack(~Tile.Tile8,x_model),
        Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,x_model),Held.band_buffer(~Tile.Tile8,Sources.input_band(x,geometry,index)),
          Buffer.remodeled(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,Sources.input_band(x,geometry,index))))),
      Equal.cong(Storage.Buffer<Tile.Tile8>,(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>,
        buffer => Kernel.groups(ranges,geometry,x_offset,d_offset,Buffer.pack(~Tile.Tile8,x_model),buffer),
        Held.band_buffer(~Tile.Tile8,Sources.gradient_band(d,geometry,index)),Buffer.pack(~Tile.Tile8,d_model),
        Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,d_model),Held.band_buffer(~Tile.Tile8,Sources.gradient_band(d,geometry,index)),
          Buffer.remodeled(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,Sources.gradient_band(d,geometry,index)))))),
    Equal.trans((Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>,
      Kernel.groups(ranges,geometry,x_offset,d_offset,Buffer.pack(~Tile.Tile8,x_model),Buffer.pack(~Tile.Tile8,d_model)),
      Refinement.represent_list(Refinement.groups(ranges,geometry,x_offset,d_offset,x_model,d_model)),
      Refinement.represent_list(Refinement.GroupList{x_model,d_model,list}),
      Refinement.groups_matches(ranges,geometry,x_offset,d_offset,x_model,d_model),
      Equal.cong(Refinement.GroupList,(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>,Refinement.represent_list,
        Refinement.groups(ranges,geometry,x_offset,d_offset,x_model,d_model),Refinement.GroupList{x_model,d_model,list},
        Refinement.groups_kept(ranges,geometry,x_offset,d_offset,x_model,d_model))))
  +reduced = Equal.cong(Array<Tile.Tile8>,Kernel.Held() & Array<Tile.Tile8>,value => ((returned_input(geometry,x,index),returned_gradient(geometry,d,index)),value),
    Grouped.levels(~&1,~Array<Tile.Tile8>,~U32,~(count => Kernel.array(~Tile.Tile8,count,Tile.filled(0.0))),~(count => left => right => Kernel.sum(count,left,right)),fuel,count,
      GroupLaws.pack_values(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~(tree => Tree.pack(Tile.Tile8,tree)),list)),
    Tree.pack(Tile.Tile8,image_tree(geometry,ranges,x,d,index)),
    GroupLaws.levels_refines_model(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~(count => Kernel.array(~Tile.Tile8,count,Tile.filled(0.0))),~Refinement.zero,
      ~(count => left => right => Kernel.sum(count,left,right)),~{count => left => right => Refinement.sum(count,left,right) : U32 -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8>},
      ~(tree => Tree.pack(Tile.Tile8,tree)),~Refinement.zero_matches,~Refinement.sum_matches,fuel,count,list))
  Equal.trans(Kernel.Held() & Array<Tile.Tile8>,Kernel.image(geometry,index,ranges,Held.pack(~Tile.Tile8,x),Held.pack(~Tile.Tile8,d)),
    Kernel.image_banded(geometry,ranges,Sources.input_band(x,geometry,index),Sources.gradient_band(d,geometry,index)),
    ((returned_input(geometry,x,index),returned_gradient(geometry,d,index)),Tree.pack(Tile.Tile8,image_tree(geometry,ranges,x,d,index))),
    image_banded_of(geometry,index,ranges,x,d),
    Equal.trans(Kernel.Held() & Array<Tile.Tile8>,Kernel.image_banded(geometry,ranges,Sources.input_band(x,geometry,index),Sources.gradient_band(d,geometry,index)),
      Kernel.reduced(fuel,count,Kernel.image_returned(Held.band_loan(~Tile.Tile8,Sources.input_band(x,geometry,index)),Held.band_loan(~Tile.Tile8,Sources.gradient_band(d,geometry,index)),
        Kernel.groups(ranges,geometry,x_offset,d_offset,Held.band_buffer(~Tile.Tile8,Sources.input_band(x,geometry,index)),Held.band_buffer(~Tile.Tile8,Sources.gradient_band(d,geometry,index))))),
      ((returned_input(geometry,x,index),returned_gradient(geometry,d,index)),Tree.pack(Tile.Tile8,image_tree(geometry,ranges,x,d,index))),
      image_eta(geometry,ranges,Sources.input_band(x,geometry,index),Sources.gradient_band(d,geometry,index)),
      Equal.trans(Kernel.Held() & Array<Tile.Tile8>,
        Kernel.reduced(fuel,count,Kernel.image_returned(Held.band_loan(~Tile.Tile8,Sources.input_band(x,geometry,index)),Held.band_loan(~Tile.Tile8,Sources.gradient_band(d,geometry,index)),
          Kernel.groups(ranges,geometry,x_offset,d_offset,Held.band_buffer(~Tile.Tile8,Sources.input_band(x,geometry,index)),Held.band_buffer(~Tile.Tile8,Sources.gradient_band(d,geometry,index))))),
        Kernel.reduced(fuel,count,Kernel.image_returned(Held.band_loan(~Tile.Tile8,Sources.input_band(x,geometry,index)),Held.band_loan(~Tile.Tile8,Sources.gradient_band(d,geometry,index)),
          Refinement.represent_list(Refinement.GroupList{x_model,d_model,list}))),
        ((returned_input(geometry,x,index),returned_gradient(geometry,d,index)),Tree.pack(Tile.Tile8,image_tree(geometry,ranges,x,d,index))),
        Equal.cong((Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>,Kernel.Held() & Array<Tile.Tile8>,
          result => Kernel.reduced(fuel,count,Kernel.image_returned(Held.band_loan(~Tile.Tile8,Sources.input_band(x,geometry,index)),Held.band_loan(~Tile.Tile8,Sources.gradient_band(d,geometry,index)),result)),
          Kernel.groups(ranges,geometry,x_offset,d_offset,Held.band_buffer(~Tile.Tile8,Sources.input_band(x,geometry,index)),Held.band_buffer(~Tile.Tile8,Sources.gradient_band(d,geometry,index))),
          Refinement.represent_list(Refinement.GroupList{x_model,d_model,list}),grouped),
        reduced)))

# ---- The lane of an image ----

# One lane of one tap of the sums, as the sum query of the kernel's cells.
def lane_sum(+geometry: Kernel.Geometry,+domain: Groups.GeometryReady<geometry>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(geometry)) == True{} : Bool},+output_inside: {U32.is_lt(output,Groups.geometry_outputs(geometry)) == True{} : Bool}) -> Refinement.SumQuery:
  Groups.GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  +plan = Groups.geometry_plan(geometry,domain)
  Refinement.plan_sum_query(plan,Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),U32.to_nat(Groups.plan_taps(plan))),U32.to_nat(tap)),Nat.mod(U32.to_nat(output),8n),
    Groups.plan_query_inside(plan,tap,output,tap_inside,output_inside))

# The products of a lane of an image over the cells the holdings hold: one
# per group, and their adjacent tree.
type HeldLane<-X: Data,-D: Data> is Data:
  HeldLane{source: Sources.HeldSource<X,D>,tap: U32,output: U32}

def held_sample(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,lane: HeldLane<X,D>,range: Grouped.Range) -> F32:
  HeldLane{source,+tap,+output} = lane
  Grouped.Range{first,count} = range
  Sources.held_dot(~X,~xv,~D,~dv,source,tap,output,count,first,0.0)

def held_image(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,lane: HeldLane<X,D>,+ranges: List<&2,Grouped.Range>) -> F32:
  Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),List.length(&2,Grouped.Range,ranges),Unit{},
    Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~HeldLane<X,D>,~held_sample(~X,~xv,~D,~dv),ranges,lane))

# The group query, sum query, lane and tree of an image lent from two
# holdings.
def lent_query(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,lending: Sources.Lending<X,xv,D,dv>,+size: U32,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool}) -> Refinement.GroupQuery:
  Sources.Lending{input,gradient,+x,+d,+geometry,+domain,+image,x_high,x_tiled,x_values,x_room,x_bounded,d_high,d_tiled,d_values,d_room,d_bounded} = lending
  +index = U32.from_nat(image)
  Refinement.GroupQuery{geometry,domain,size,Held.band_offset(~Tile.Tile8,Sources.input_band(x,geometry,index)),Held.band_offset(~Tile.Tile8,Sources.gradient_band(d,geometry,index)),
    Sources.band_model(Sources.input_band(x,geometry,index)),Sources.band_model(Sources.gradient_band(d,geometry,index)),tap,output,tap_inside,output_inside}

def lent_sum(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,lending: Sources.Lending<X,xv,D,dv>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool}) -> Refinement.SumQuery:
  Sources.Lending{input,gradient,x,d,+geometry,+domain,image,x_high,x_tiled,x_values,x_room,x_bounded,d_high,d_tiled,d_values,d_room,d_bounded} = lending
  lane_sum(geometry,domain,tap,output,tap_inside,output_inside)

def lent_lane(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lending: Sources.Lending<X,xv,D,dv>,+tap: U32,+output: U32) -> HeldLane<X,D>:
  HeldLane{Sources.held_source(~X,~xv,~D,~dv,lending),tap,output}

def lent_tree(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,lending: Sources.Lending<X,xv,D,dv>,+ranges: List<&2,Grouped.Range>) -> Tree.Tree<Tile.Tile8>:
  Sources.Lending{input,gradient,+x,+d,+geometry,domain,+image,x_high,x_tiled,x_values,x_room,x_bounded,d_high,d_tiled,d_values,d_room,d_bounded} = lending
  image_tree(geometry,ranges,x,d,U32.from_nat(image))

def RangesBounded(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lending: Sources.Lending<X,xv,D,dv>,+size: U32,ranges: List<&2,Grouped.Range>) -> Data:
  Members.All(Grouped.Range,{range => GroupLaws.RangeBounded(Groups.row_count(Sources.lent_geometry(~X,~xv,~D,~dv,lending)),Groups.column_count(Sources.lent_geometry(~X,~xv,~D,~dv,lending)),
    U32.to_nat(size),range) : Grouped.Range -> Data},ranges)

# A group product over the bands is the product over the held cells.
def sample_held(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lending: Sources.Lending<X,xv,D,dv>,+size: U32,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +range: Grouped.Range,
  +bounded: GroupLaws.RangeBounded(Groups.row_count(Sources.lent_geometry(~X,~xv,~D,~dv,lending)),Groups.column_count(Sources.lent_geometry(~X,~xv,~D,~dv,lending)),U32.to_nat(size),range)) ->
  {Refinement.group_query_sample(lent_query(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside),range) ==
    held_sample(~X,~xv,~D,~dv,lent_lane(~X,~xv,~D,~dv,lending,tap,output),range) : F32}:
  Sources.Lending{+input,+gradient,+x,+d,+geometry,+domain,+image,+x_high,+x_tiled,+x_values,+x_room,+x_bounded,+d_high,+d_tiled,+d_values,+d_room,+d_bounded} = lending
  Grouped.Range{+first,+count} = range
  KernelGemm.Both{nonempty,KernelGemm.Both{small,+room}} = bounded
  Sources.lent_dot(~X,~xv,~D,~dv,lending,tap,output,Bounds.as_nat_lt(tap,Packing.geometry_taps(geometry),tap_inside),
    Bounds.as_nat_lt(output,Groups.geometry_outputs(geometry),output_inside),count,first,0.0,room)

def samples_held(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lending: Sources.Lending<X,xv,D,dv>,+size: U32,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  ranges: List<&2,Grouped.Range>,bounded: RangesBounded(~X,~xv,~D,~dv,lending,size,ranges)) ->
  {Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Refinement.GroupQuery,~Refinement.group_query_sample,ranges,lent_query(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside)) ==
    Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~HeldLane<X,D>,~held_sample(~X,~xv,~D,~dv),ranges,lent_lane(~X,~xv,~D,~dv,lending,tap,output)) : List<&2,F32>}:
  match ranges:
    case []: {==}
    case +range <> +rest:
      KernelGemm.Both{+head,+tail} = bounded
      +query = lent_query(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside)
      +lane = lent_lane(~X,~xv,~D,~dv,lending,tap,output)
      %Equal.sym(F32,Refinement.group_query_sample(query,range),held_sample(~X,~xv,~D,~dv,lane,range),
        sample_held(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside,range,head)) :
        {_ <> Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Refinement.GroupQuery,~Refinement.group_query_sample,rest,query) ==
          held_sample(~X,~xv,~D,~dv,lane,range) <> Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~HeldLane<X,D>,~held_sample(~X,~xv,~D,~dv),rest,lane) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,values => held_sample(~X,~xv,~D,~dv,lane,range) <> values,
        Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Refinement.GroupQuery,~Refinement.group_query_sample,rest,query),
        Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~HeldLane<X,D>,~held_sample(~X,~xv,~D,~dv),rest,lane),
        samples_held(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside,rest,tail))

# The collected groups of a query: their frames and their lanes.
def collected_parts(+query: Refinement.GroupQuery,+ranges: List<&2,Grouped.Range>,result: Groups.Sources & List<&2,Tree.Tree<Tile.Tile8>>,
  frame: Traversals.collected_list_frame(~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~Refinement.GroupQuery,~F32,
    ~Refinement.group_query_frame,~Refinement.group_query_value,~Refinement.group_query_observe,~Refinement.group_query_sample,query,ranges,result)) ->
  KernelGemm.Both<Members.All(Tree.Tree<Tile.Tile8>,{tree => Refinement.sum_query_frame(Refinement.group_query_sum(query),tree) : Tree.Tree<Tile.Tile8> -> Data},
      Pair.snd(Groups.Sources,List<&2,Tree.Tree<Tile.Tile8>>,result)),
    {GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~Refinement.GroupQuery,~Refinement.group_query_observe,Pair.snd(Groups.Sources,List<&2,Tree.Tree<Tile.Tile8>>,result),query) ==
      Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Refinement.GroupQuery,~Refinement.group_query_sample,ranges,query) : List<&2,F32>}>:
  (+source,+values) = result
  KernelGemm.Both{source_frame,KernelGemm.Both{+values_frame,+values_equal}} = frame
  KernelGemm.Both{values_frame,values_equal}

# The sum query of the groups of a lent image reads and frames the cells of
# the kernel as the sum query of its lane.
def query_count(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lending: Sources.Lending<X,xv,D,dv>,+size: U32,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool}) ->
  {Refinement.sum_query_count(Refinement.group_query_sum(lent_query(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside))) ==
    Kernel.cells(Sources.lent_geometry(~X,~xv,~D,~dv,lending)) : U32}:
  Sources.Lending{input,gradient,x,d,geometry,domain,image,x_high,x_tiled,x_values,x_room,x_bounded,d_high,d_tiled,d_values,d_room,d_bounded} = lending
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Groups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned,taps_exact,pixels,gradients} = domain
  {==}

def query_read(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lending: Sources.Lending<X,xv,D,dv>,+size: U32,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},+tree: Tree.Tree<Tile.Tile8>) ->
  {Refinement.sum_query_read(Refinement.group_query_sum(lent_query(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside)),tree) ==
    Refinement.sum_query_read(lent_sum(~X,~xv,~D,~dv,lending,tap,output,tap_inside,output_inside),tree) : F32}:
  Sources.Lending{input,gradient,x,d,geometry,domain,image,x_high,x_tiled,x_values,x_room,x_bounded,d_high,d_tiled,d_values,d_room,d_bounded} = lending
  Groups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned,taps_exact,pixels,gradients} = domain
  {==}

def query_frame(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lending: Sources.Lending<X,xv,D,dv>,+size: U32,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},+tree: Tree.Tree<Tile.Tile8>,
  frame: Refinement.sum_query_frame(Refinement.group_query_sum(lent_query(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside)),tree)) ->
  Refinement.sum_query_frame(lent_sum(~X,~xv,~D,~dv,lending,tap,output,tap_inside,output_inside),tree):
  Sources.Lending{input,gradient,x,d,geometry,domain,image,x_high,x_tiled,x_values,x_room,x_bounded,d_high,d_tiled,d_values,d_room,d_bounded} = lending
  Groups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned,taps_exact,pixels,gradients} = domain
  frame

# The adjacent tree of lanes, and of cell trees of a count of cells.
def lane_levels(+fuel: Nat,values: List<&2,F32>) -> F32:
  Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},values)

def tree_levels(+fuel: Nat,+list: List<&2,Tree.Tree<Tile.Tile8>>,count: U32) -> Tree.Tree<Tile.Tile8>:
  Grouped.levels(~&2,~Tree.Tree<Tile.Tile8>,~U32,~Refinement.zero,~{count => left => right => Refinement.sum(count,left,right) : U32 -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8>},
    fuel,count,list)

# A lane of the sums of an image is the adjacent tree of its group products
# over the cells the holdings hold.
def image_lane(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lending: Sources.Lending<X,xv,D,dv>,+size: U32,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Sources.lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +ranges: List<&2,Grouped.Range>,+bounded: RangesBounded(~X,~xv,~D,~dv,lending,size,ranges)) ->
  KernelGemm.Both<Refinement.sum_query_frame(lent_sum(~X,~xv,~D,~dv,lending,tap,output,tap_inside,output_inside),lent_tree(~X,~xv,~D,~dv,lending,ranges)),
    {Refinement.sum_query_read(lent_sum(~X,~xv,~D,~dv,lending,tap,output,tap_inside,output_inside),lent_tree(~X,~xv,~D,~dv,lending,ranges)) ==
      held_image(~X,~xv,~D,~dv,lent_lane(~X,~xv,~D,~dv,lending,tap,output),ranges) : F32}>:
  Sources.Lending{+input,+gradient,+x,+d,+geometry,+domain,+image,+x_high,+x_tiled,+x_values,+x_room,+x_bounded,+d_high,+d_tiled,+d_values,+d_room,+d_bounded} = lending
  +query = lent_query(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside)
  +sum = Refinement.group_query_sum(query)
  +lent = lent_sum(~X,~xv,~D,~dv,lending,tap,output,tap_inside,output_inside)
  +list = image_list(geometry,ranges,x,d,U32.from_nat(image))
  +fuel = List.length(&2,Grouped.Range,ranges)
  +lane = lent_lane(~X,~xv,~D,~dv,lending,tap,output)
  +parts = collected_parts(query,ranges,
    Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~Kernel.GroupContext,~Refinement.group_read,ranges,
      Refinement.group_query_source(query),Refinement.group_query_context(query)),
    Refinement.group_query_collect(query,ranges,bounded))
  +observed = GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~Refinement.SumQuery,~Refinement.sum_query_read,list,sum)
  +held = Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~HeldLane<X,D>,~held_sample(~X,~xv,~D,~dv),ranges,lane)
  +lanes = Equal.trans(List<&2,F32>,observed,GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~Refinement.GroupQuery,~Refinement.group_query_observe,list,query),held,
    Equal.sym(List<&2,F32>,GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~Refinement.GroupQuery,~Refinement.group_query_observe,list,query),observed,Refinement.observed_query(query,list)),
    Equal.trans(List<&2,F32>,GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~Refinement.GroupQuery,~Refinement.group_query_observe,list,query),
      Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Refinement.GroupQuery,~Refinement.group_query_sample,ranges,query),held,
      Shape.second(Members.All(Tree.Tree<Tile.Tile8>,{tree => Refinement.sum_query_frame(sum,tree) : Tree.Tree<Tile.Tile8> -> Data},list),{GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~Refinement.GroupQuery,~Refinement.group_query_observe,list,query) ==
    Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Refinement.GroupQuery,~Refinement.group_query_sample,ranges,query) : List<&2,F32>},parts),samples_held(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside,ranges,bounded)))
  +count = Refinement.sum_query_count(sum)
  +summed = Refinement.sum_tree(fuel,sum,list,Shape.first(Members.All(Tree.Tree<Tile.Tile8>,{tree => Refinement.sum_query_frame(sum,tree) : Tree.Tree<Tile.Tile8> -> Data},list),
    {GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~Refinement.GroupQuery,~Refinement.group_query_observe,list,query) ==
      Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Refinement.GroupQuery,~Refinement.group_query_sample,ranges,query) : List<&2,F32>},parts))
  +frame = query_frame(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside,tree_levels(fuel,list,count),
    Shape.first(Refinement.sum_query_frame(sum,tree_levels(fuel,list,count)),{Refinement.sum_query_read(sum,tree_levels(fuel,list,count)) == lane_levels(fuel,observed) : F32},summed))
  +read = Equal.trans(F32,Refinement.sum_query_read(lent,tree_levels(fuel,list,count)),Refinement.sum_query_read(sum,tree_levels(fuel,list,count)),held_image(~X,~xv,~D,~dv,lane,ranges),
    Equal.sym(F32,Refinement.sum_query_read(sum,tree_levels(fuel,list,count)),Refinement.sum_query_read(lent,tree_levels(fuel,list,count)),
      query_read(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside,tree_levels(fuel,list,count))),
    Equal.trans(F32,Refinement.sum_query_read(sum,tree_levels(fuel,list,count)),lane_levels(fuel,observed),held_image(~X,~xv,~D,~dv,lane,ranges),
      Shape.second(Refinement.sum_query_frame(sum,tree_levels(fuel,list,count)),{Refinement.sum_query_read(sum,tree_levels(fuel,list,count)) == lane_levels(fuel,observed) : F32},summed),
      Equal.cong(List<&2,F32>,F32,values => lane_levels(fuel,values),observed,held,lanes)))
  %query_count(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside) :
    KernelGemm.Both<Refinement.sum_query_frame(lent,tree_levels(fuel,list,_)),{Refinement.sum_query_read(lent,tree_levels(fuel,list,_)) == held_image(~X,~xv,~D,~dv,lane,ranges) : F32}>
  KernelGemm.Both{frame,read}

# ---- The images of a batch ----

# A batch of images whose input cells and gradient cells two holdings hold
# from cell 0 on, each with the values of its reader, and the ranges of the
# groups of an image.
type Batch<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8> is Data:
  Batch{input: X,gradient: D,geometry: Kernel.Geometry,domain: Groups.GeometryReady<geometry>,images: Nat,rows: Nat,columns: Nat,group: Nat,size: U32,
    bounded: Members.All(Grouped.Range,{range => GroupLaws.RangeBounded(Groups.row_count(geometry),Groups.column_count(geometry),U32.to_nat(size),range) : Grouped.Range -> Data},
      Grouped.ranges(rows,columns,group)),
    x_high: Nat,x_layout: Held.Spans,
    x_room: {Nat.is_le(Nat.mul(images,Sources.input_cells(geometry)),x_high) == True{} : Bool},
    x_bounded: {Nat.is_le(x_high,U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))) == True{} : Bool},
    d_high: Nat,d_layout: Held.Spans,
    d_room: {Nat.is_le(Nat.mul(images,Groups.gradient_extent(geometry)),d_high) == True{} : Bool},
    d_bounded: {Nat.is_le(d_high,U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))) == True{} : Bool}}

def batch_input(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> X:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  input

def batch_gradient(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> D:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  gradient

def batch_geometry(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> Kernel.Geometry:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  geometry

def batch_images(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> Nat:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  images

def batch_rows(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> Nat:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  rows

def batch_columns(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> Nat:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  columns

def batch_group(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> Nat:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  group

def batch_x_high(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> Nat:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  x_high

def batch_x_layout(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> Held.Spans:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  x_layout

def batch_d_high(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> Nat:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  d_high

def batch_d_layout(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> Held.Spans:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  d_layout

def batch_ranges(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>) -> List<&2,Grouped.Range>:
  Batch{input,gradient,geometry,domain,images,+rows,+columns,+group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  Grouped.ranges(rows,columns,group)

# The lane of an image of the batch over the cells the holdings hold, and the
# sum query of a lane.
def batch_lane(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Batch<X,xv,D,dv>,+image: Nat,+tap: U32,+output: U32) -> HeldLane<X,D>:
  Batch{input,gradient,+geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  HeldLane{Sources.HeldSource{input,gradient,geometry,Nat.mul(image,Sources.input_cells(geometry)),Nat.mul(image,Groups.gradient_extent(geometry))},tap,output}

# The lane of an image of a batch reads the batch's holders and geometry.
def batch_lane_parts(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Batch<X,xv,D,dv>,+image: Nat,+tap: U32,+output: U32) ->
  {batch_lane(~X,~xv,~D,~dv,batch,image,tap,output) ==
    HeldLane{Sources.HeldSource{batch_input(~X,~xv,~D,~dv,batch),batch_gradient(~X,~xv,~D,~dv,batch),batch_geometry(~X,~xv,~D,~dv,batch),
      Nat.mul(image,Sources.input_cells(batch_geometry(~X,~xv,~D,~dv,batch))),Nat.mul(image,Groups.gradient_extent(batch_geometry(~X,~xv,~D,~dv,batch)))},tap,output} : HeldLane<X,D>}:
  Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  {==}

def batch_sum(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Batch<X,xv,D,dv>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool}) -> Refinement.SumQuery:
  Batch{input,gradient,+geometry,+domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  lane_sum(geometry,domain,tap,output,tap_inside,output_inside)

# The lanes of images [image, image + remaining).
def held_images(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,remaining: Nat,+image: Nat,+batch: Batch<X,xv,D,dv>,+tap: U32,+output: U32) -> List<&2,F32>:
  match remaining:
    case 0n: []
    case 1n+rest:
      held_image(~X,~xv,~D,~dv,batch_lane(~X,~xv,~D,~dv,batch,image,tap,output),batch_ranges(~X,~xv,~D,~dv,batch)) <>
        held_images(~X,~xv,~D,~dv,rest,1n+image,batch,tap,output)

# The images the kernel collects from a state on.
def gathered_images(fuel: Nat,state: Traversal.Gathering<&1,&1,Kernel.Held(),Array<Tile.Tile8>,Kernel.ImageContext>) -> Kernel.Held() & List<&1,Array<Tile.Tile8>>:
  Traversal.gathering_result(~&1,~Kernel.Held(),~Array<Tile.Tile8>,~Kernel.ImageContext,
    Loop.run(~Traversal.Gathering<&1,&1,Kernel.Held(),Array<Tile.Tile8>,Kernel.ImageContext>,~Traversal.gather_step(~&1,~Kernel.Held(),~Array<Tile.Tile8>,~Kernel.ImageContext,~Kernel.image_read),
      fuel,state))

def pack_trees(values: List<&2,Tree.Tree<Tile.Tile8>>) -> List<&1,Array<Tile.Tile8>>:
  GroupLaws.pack_values(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~(tree => Tree.pack(Tile.Tile8,tree)),values)

def observed_trees(+sum: Refinement.SumQuery,values: List<&2,Tree.Tree<Tile.Tile8>>) -> List<&2,F32>:
  GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~Refinement.SumQuery,~Refinement.sum_query_read,values,sum)

# What the images from start give back after fuel more: the holdings, which
# still cover the cells with their values, and the sums of every image so
# far, in order, with their frames and lanes.
type ImagesDone<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-batch: Batch<X,xv,D,dv>,-tap: U32,-output: U32,-sum: Refinement.SumQuery,
  -before: List<&2,Tree.Tree<Tile.Tile8>>,-fuel: Nat,-start: Nat,-result: Kernel.Held() & List<&1,Array<Tile.Tile8>>> is Data:
  ImagesDone{trees: List<&2,Tree.Tree<Tile.Tile8>>,
    values: {result == (Pair.fst(Kernel.Held(),List<&1,Array<Tile.Tile8>>,result),pack_trees(trees)) : Kernel.Held() & List<&1,Array<Tile.Tile8>>},
    x_covered: Held.Covered<Tile.Tile8,X,xv,batch_input(~X,~xv,~D,~dv,batch),Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),List<&1,Array<Tile.Tile8>>,result)),
      0n,batch_x_high(~X,~xv,~D,~dv,batch),batch_x_layout(~X,~xv,~D,~dv,batch)>,
    d_covered: Held.Covered<Tile.Tile8,D,dv,batch_gradient(~X,~xv,~D,~dv,batch),Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),List<&1,Array<Tile.Tile8>>,result)),
      0n,batch_d_high(~X,~xv,~D,~dv,batch),batch_d_layout(~X,~xv,~D,~dv,batch)>,
    frames: Members.All(Tree.Tree<Tile.Tile8>,{tree => Refinement.sum_query_frame(sum,tree) : Tree.Tree<Tile.Tile8> -> Data},trees),
    observed: {observed_trees(sum,trees) == Traversal.reversed_onto(~&2,~F32,observed_trees(sum,before),held_images(~X,~xv,~D,~dv,fuel,start,batch,tap,output)) : List<&2,F32>}}

def all_reversed(-Element: Data,-Property: Element -> Data,+values: List<&2,Element>,+tail: List<&2,Element>,
  values_proof: Members.All(Element,Property,values),tail_proof: Members.All(Element,Property,tail)) ->
  Members.All(Element,Property,Traversal.reversed_onto(~&2,~Element,values,tail)):
  match values:
    case Nil{}: tail_proof
    case +head <> +rest:
      KernelGemm.Both{first,others} = values_proof
      all_reversed(Element,Property,rest,head <> tail,others,KernelGemm.Both{first,tail_proof})

def observed_reversed(+sum: Refinement.SumQuery,values: List<&2,Tree.Tree<Tile.Tile8>>,+tail: List<&2,Tree.Tree<Tile.Tile8>>) ->
  {observed_trees(sum,Traversal.reversed_onto(~&2,~Tree.Tree<Tile.Tile8>,values,tail)) ==
    Traversal.reversed_onto(~&2,~F32,observed_trees(sum,values),observed_trees(sum,tail)) : List<&2,F32>}:
  match values:
    case Nil{}: {==}
    case +head <> +rest: observed_reversed(sum,rest,head <> tail)

# The cells of image start fit below the cells of the batch.
def slot_room(+start: Nat,+rest: Nat,+images: Nat,+cells: Nat,+high: Nat,+room: {Nat.is_le(Nat.add(start,1n+rest),images) == True{} : Bool},
  +total: {Nat.is_le(Nat.mul(images,cells),high) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.mul(start,cells),cells),high) == True{} : Bool}:
  %Order.commutative(cells,Nat.mul(start,cells)) : {Nat.is_le(_,high) == True{} : Bool}
  Order.transitive(Nat.mul(1n+start,cells),Nat.mul(images,cells),high,
    RowMajor.mul_le(1n+start,images,cells,Order.strict_follow(start,images,Order.room_index(start,rest,images,room))),total)

def next_index(+start: Nat) -> {(U32.from_nat(start) + 1 : U32) == U32.from_nat(1n+start) : U32}:
  %Order.commutative(start,1n) : {(U32.from_nat(start) + 1 : U32) == U32.from_nat(_) : U32}
  WordArithmetic.native_add(start,1n)

# A step of the loop runs the image of its context and keeps its sums.
def image_step(+geometry: Kernel.Geometry,+position: U32,+rows: Nat,+columns: Nat,+group: Nat,+x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>,+before: List<&2,Tree.Tree<Tile.Tile8>>) ->
  {Traversal.gather_step(~&1,~Kernel.Held(),~Array<Tile.Tile8>,~Kernel.ImageContext,~Kernel.image_read,
      Traversal.Gathering{(Held.pack(~Tile.Tile8,x),Held.pack(~Tile.Tile8,d)),Kernel.ImageContext{geometry,position,rows,columns,group},pack_trees(before)}) ==
    Traversal.Gathering{(returned_input(geometry,x,position),returned_gradient(geometry,d,position)),Kernel.ImageContext{geometry,(position + 1 : U32),rows,columns,group},
      pack_trees(image_tree(geometry,Grouped.ranges(rows,columns,group),x,d,position) <> before)} : Traversal.Gathering<&1,&1,Kernel.Held(),Array<Tile.Tile8>,Kernel.ImageContext>}:
  Equal.cong(Kernel.Held() & Array<Tile.Tile8>,Traversal.Gathering<&1,&1,Kernel.Held(),Array<Tile.Tile8>,Kernel.ImageContext>,
    result => Traversal.gathered(~&1,~Kernel.Held(),~Array<Tile.Tile8>,~Kernel.ImageContext,
      Kernel.image_collected(Kernel.ImageContext{geometry,(position + 1 : U32),rows,columns,group},result),pack_trees(before)),
    Kernel.image(geometry,position,Grouped.ranges(rows,columns,group),Held.pack(~Tile.Tile8,x),Held.pack(~Tile.Tile8,d)),
    ((returned_input(geometry,x,position),returned_gradient(geometry,d,position)),Tree.pack(Tile.Tile8,image_tree(geometry,Grouped.ranges(rows,columns,group),x,d,position))),
    image_matches(geometry,position,Grouped.ranges(rows,columns,group),x,d))

# The images after a step, with the lane of the image of the step.
def done_step(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Batch<X,xv,D,dv>,+tap: U32,+output: U32,+sum: Refinement.SumQuery,
  +tree: Tree.Tree<Tile.Tile8>,+before: List<&2,Tree.Tree<Tile.Tile8>>,+rest: Nat,+start: Nat,-result: Kernel.Held() & List<&1,Array<Tile.Tile8>>,
  next: ImagesDone<X,xv,D,dv,batch,tap,output,sum,tree <> before,rest,1n+start,result>,
  +lane: {Refinement.sum_query_read(sum,tree) == held_image(~X,~xv,~D,~dv,batch_lane(~X,~xv,~D,~dv,batch,start,tap,output),batch_ranges(~X,~xv,~D,~dv,batch)) : F32}) ->
  ImagesDone<X,xv,D,dv,batch,tap,output,sum,before,1n+rest,start,result>:
  ImagesDone{+trees,+values,+x_covered,+d_covered,+frames,+observed} = next
  ImagesDone{trees,values,x_covered,d_covered,frames,
    Equal.trans(List<&2,F32>,observed_trees(sum,trees),
      Traversal.reversed_onto(~&2,~F32,observed_trees(sum,tree <> before),held_images(~X,~xv,~D,~dv,rest,1n+start,batch,tap,output)),
      Traversal.reversed_onto(~&2,~F32,observed_trees(sum,before),held_images(~X,~xv,~D,~dv,1n+rest,start,batch,tap,output)),observed,
      Equal.cong(F32,List<&2,F32>,value => Traversal.reversed_onto(~&2,~F32,observed_trees(sum,before),value <> held_images(~X,~xv,~D,~dv,rest,1n+start,batch,tap,output)),
        Refinement.sum_query_read(sum,tree),held_image(~X,~xv,~D,~dv,batch_lane(~X,~xv,~D,~dv,batch,start,tap,output),batch_ranges(~X,~xv,~D,~dv,batch)),lane))}

# The images from start on: every image lends its bands, sums its groups and
# gives the bands back.
def images_loop(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,fuel: Nat,+batch: Batch<X,xv,D,dv>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +start: Nat,+index: U32,+index_exact: {index == U32.from_nat(start) : U32},
  +room: {Nat.is_le(Nat.add(start,fuel),batch_images(~X,~xv,~D,~dv,batch)) == True{} : Bool},
  -x_held: Move.Holding<Tile.Tile8>,-d_held: Move.Holding<Tile.Tile8>,
  x_covered: Held.Covered<Tile.Tile8,X,xv,batch_input(~X,~xv,~D,~dv,batch),x_held,0n,batch_x_high(~X,~xv,~D,~dv,batch),batch_x_layout(~X,~xv,~D,~dv,batch)>,
  d_covered: Held.Covered<Tile.Tile8,D,dv,batch_gradient(~X,~xv,~D,~dv,batch),d_held,0n,batch_d_high(~X,~xv,~D,~dv,batch),batch_d_layout(~X,~xv,~D,~dv,batch)>,
  +before: List<&2,Tree.Tree<Tile.Tile8>>,
  frames: Members.All(Tree.Tree<Tile.Tile8>,{tree => Refinement.sum_query_frame(batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside),tree) : Tree.Tree<Tile.Tile8> -> Data},before)) ->
  ImagesDone<X,xv,D,dv,batch,tap,output,batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside),before,fuel,start,
    gathered_images(fuel,Traversal.Gathering{(x_held,d_held),
      Kernel.ImageContext{batch_geometry(~X,~xv,~D,~dv,batch),index,batch_rows(~X,~xv,~D,~dv,batch),batch_columns(~X,~xv,~D,~dv,batch),batch_group(~X,~xv,~D,~dv,batch)},pack_trees(before)})>:
  match fuel:
    case 0n:
      +sum = batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside)
      ImagesDone{Traversal.reversed_onto(~&2,~Tree.Tree<Tile.Tile8>,before,[]),
        Equal.cong(List<&1,Array<Tile.Tile8>>,Kernel.Held() & List<&1,Array<Tile.Tile8>>,list => ((x_held,d_held),list),
          Traversal.reversed_onto(~&1,~Array<Tile.Tile8>,pack_trees(before),[]),pack_trees(Traversal.reversed_onto(~&2,~Tree.Tree<Tile.Tile8>,before,[])),
          Equal.sym(List<&1,Array<Tile.Tile8>>,pack_trees(Traversal.reversed_onto(~&2,~Tree.Tree<Tile.Tile8>,before,[])),
            Traversal.reversed_onto(~&1,~Array<Tile.Tile8>,pack_trees(before),[]),
            Traversals.collect_values_reversed(~&1,~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~(tree => Tree.pack(Tile.Tile8,tree)),before,[]))),
        x_covered,d_covered,
        all_reversed(Tree.Tree<Tile.Tile8>,{tree => Refinement.sum_query_frame(sum,tree) : Tree.Tree<Tile.Tile8> -> Data},before,[],frames,Unit{}),
        observed_reversed(sum,before,[])}
    case 1n+ +rest:
      Batch{+x_holder,+d_holder,+geometry,+domain,+images,+rows,+columns,+group,+size,+bounded,+x_high,+x_layout,+x_room,+x_bounded,+d_high,+d_layout,+d_room,+d_bounded} = batch
      Held.Covered{+x,+x_packed,+x_tiled,+x_values,+x_ordered} = x_covered
      Held.Covered{+d,+d_packed,+d_tiled,+d_values,+d_ordered} = d_covered
      +sum = lane_sum(geometry,domain,tap,output,tap_inside,output_inside)
      +ranges = Grouped.ranges(rows,columns,group)
      +position = U32.from_nat(start)
      +lending = {Sources.Lending{x_holder,d_holder,x,d,geometry,domain,start,x_high,x_tiled,x_values,slot_room(start,rest,images,Sources.input_cells(geometry),x_high,room,x_room),x_bounded,
        d_high,d_tiled,d_values,slot_room(start,rest,images,Groups.gradient_extent(geometry),d_high,room,d_room),d_bounded} : Sources.Lending<X,xv,D,dv>}
      +tree = image_tree(geometry,ranges,x,d,position)
      +lane = image_lane(~X,~xv,~D,~dv,lending,size,tap,output,tap_inside,output_inside,ranges,bounded)
      +x_after = {%Equal.sym(Move.Spare<Tile.Tile8>,Move.spare(~Tile.Tile8,Tile.filled(0.0)),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))),Held.spare_packed(~Tile.Tile8,Tile.filled(0.0))) :
        Held.CoveredFact(~Tile.Tile8,~X,~xv,x_holder,Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,x),(position * Sources.input_word(geometry) : U32),Sources.input_word(geometry),Tile.filled(0.0),_)),Buffer.pack(~Tile.Tile8,Sources.band_model(Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,x),(position * Sources.input_word(geometry) : U32),Sources.input_word(geometry),Tile.filled(0.0),_)))),0n,x_high,x_layout)
        Held.band_returned(~Tile.Tile8,~X,~xv,x_holder,x,(position * Sources.input_word(geometry) : U32),Sources.input_word(geometry),Tile.filled(0.0),Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0)),0n,x_high,x_tiled,x_values,
          x_layout,x_ordered,Sources.band_model(Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,x),(position * Sources.input_word(geometry) : U32),Sources.input_word(geometry),Tile.filled(0.0),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))))),Buffer.remodel_content(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,x),(position * Sources.input_word(geometry) : U32),Sources.input_word(geometry),Tile.filled(0.0),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))))))) :
        Held.CoveredFact(~Tile.Tile8,~X,~xv,x_holder,returned_input(geometry,x,position),0n,x_high,x_layout)}
      +d_after = {%Equal.sym(Move.Spare<Tile.Tile8>,Move.spare(~Tile.Tile8,Tile.filled(0.0)),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))),Held.spare_packed(~Tile.Tile8,Tile.filled(0.0))) :
        Held.CoveredFact(~Tile.Tile8,~D,~dv,d_holder,Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,d),(position * Sources.gradient_word(geometry) : U32),Sources.gradient_word(geometry),Tile.filled(0.0),_)),Buffer.pack(~Tile.Tile8,Sources.band_model(Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,d),(position * Sources.gradient_word(geometry) : U32),Sources.gradient_word(geometry),Tile.filled(0.0),_)))),0n,d_high,d_layout)
        Held.band_returned(~Tile.Tile8,~D,~dv,d_holder,d,(position * Sources.gradient_word(geometry) : U32),Sources.gradient_word(geometry),Tile.filled(0.0),Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0)),0n,d_high,d_tiled,d_values,
          d_layout,d_ordered,Sources.band_model(Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,d),(position * Sources.gradient_word(geometry) : U32),Sources.gradient_word(geometry),Tile.filled(0.0),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))))),Buffer.remodel_content(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,d),(position * Sources.gradient_word(geometry) : U32),Sources.gradient_word(geometry),Tile.filled(0.0),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))))))) :
        Held.CoveredFact(~Tile.Tile8,~D,~dv,d_holder,returned_gradient(geometry,d,position),0n,d_high,d_layout)}
      +next = images_loop(~X,~xv,~D,~dv,rest,batch,tap,output,tap_inside,output_inside,1n+start,(position + 1 : U32),next_index(start),Order.advance_room(start,rest,images,room),
        returned_input(geometry,x,position),returned_gradient(geometry,d,position),x_after,d_after,tree <> before,
        KernelGemm.Both{Shape.first(Refinement.sum_query_frame(sum,tree),{Refinement.sum_query_read(sum,tree) == held_image(~X,~xv,~D,~dv,batch_lane(~X,~xv,~D,~dv,batch,start,tap,output),ranges) : F32},lane),
          frames})
      %Equal.sym(Move.Holding<Tile.Tile8>,x_held,Held.pack(~Tile.Tile8,x),x_packed) :
        ImagesDone<X,xv,D,dv,batch,tap,output,sum,before,1n+rest,start,gathered_images(1n+rest,Traversal.Gathering{(_,d_held),Kernel.ImageContext{geometry,index,rows,columns,group},pack_trees(before)})>
      %Equal.sym(Move.Holding<Tile.Tile8>,d_held,Held.pack(~Tile.Tile8,d),d_packed) :
        ImagesDone<X,xv,D,dv,batch,tap,output,sum,before,1n+rest,start,
          gathered_images(1n+rest,Traversal.Gathering{(Held.pack(~Tile.Tile8,x),_),Kernel.ImageContext{geometry,index,rows,columns,group},pack_trees(before)})>
      %Equal.sym(U32,index,position,index_exact) :
        ImagesDone<X,xv,D,dv,batch,tap,output,sum,before,1n+rest,start,
          gathered_images(1n+rest,Traversal.Gathering{(Held.pack(~Tile.Tile8,x),Held.pack(~Tile.Tile8,d)),Kernel.ImageContext{geometry,_,rows,columns,group},pack_trees(before)})>
      %Equal.sym(Traversal.Gathering<&1,&1,Kernel.Held(),Array<Tile.Tile8>,Kernel.ImageContext>,
        Traversal.gather_step(~&1,~Kernel.Held(),~Array<Tile.Tile8>,~Kernel.ImageContext,~Kernel.image_read,
          Traversal.Gathering{(Held.pack(~Tile.Tile8,x),Held.pack(~Tile.Tile8,d)),Kernel.ImageContext{geometry,position,rows,columns,group},pack_trees(before)}),
        Traversal.Gathering{(returned_input(geometry,x,position),returned_gradient(geometry,d,position)),Kernel.ImageContext{geometry,(position + 1 : U32),rows,columns,group},
          pack_trees(tree <> before)},
        image_step(geometry,position,rows,columns,group,x,d,before)) :
        ImagesDone<X,xv,D,dv,batch,tap,output,sum,before,1n+rest,start,gathered_images(rest,_)>
      done_step(~X,~xv,~D,~dv,batch,tap,output,sum,tree,before,rest,start,
        gathered_images(rest,Traversal.Gathering{(returned_input(geometry,x,position),returned_gradient(geometry,d,position)),
          Kernel.ImageContext{geometry,(position + 1 : U32),rows,columns,group},pack_trees(tree <> before)}),
        next,Shape.second(Refinement.sum_query_frame(sum,tree),
          {Refinement.sum_query_read(sum,tree) == held_image(~X,~xv,~D,~dv,batch_lane(~X,~xv,~D,~dv,batch,start,tap,output),ranges) : F32},lane))

# The images of a batch, from the first.
def images_done(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Batch<X,xv,D,dv>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  -x_held: Move.Holding<Tile.Tile8>,-d_held: Move.Holding<Tile.Tile8>,
  x_covered: Held.Covered<Tile.Tile8,X,xv,batch_input(~X,~xv,~D,~dv,batch),x_held,0n,batch_x_high(~X,~xv,~D,~dv,batch),batch_x_layout(~X,~xv,~D,~dv,batch)>,
  d_covered: Held.Covered<Tile.Tile8,D,dv,batch_gradient(~X,~xv,~D,~dv,batch),d_held,0n,batch_d_high(~X,~xv,~D,~dv,batch),batch_d_layout(~X,~xv,~D,~dv,batch)>) ->
  ImagesDone<X,xv,D,dv,batch,tap,output,batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside),[],batch_images(~X,~xv,~D,~dv,batch),0n,
    Kernel.images(batch_images(~X,~xv,~D,~dv,batch),0,batch_geometry(~X,~xv,~D,~dv,batch),batch_rows(~X,~xv,~D,~dv,batch),batch_columns(~X,~xv,~D,~dv,batch),batch_group(~X,~xv,~D,~dv,batch),
      x_held,d_held)>:
  images_loop(~X,~xv,~D,~dv,batch_images(~X,~xv,~D,~dv,batch),batch,tap,output,tap_inside,output_inside,0n,0,{==},Order.reflexive(batch_images(~X,~xv,~D,~dv,batch)),
    x_held,d_held,x_covered,d_covered,[],Unit{})

# ---- The weight of a batch ----

# The sum query of a lane counts the cells of the kernel and reads its lane
# of its cell.
def cells_count(+geometry: Kernel.Geometry,+domain: Groups.GeometryReady<geometry>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(geometry)) == True{} : Bool},+output_inside: {U32.is_lt(output,Groups.geometry_outputs(geometry)) == True{} : Bool}) ->
  {Refinement.sum_query_count(lane_sum(geometry,domain,tap,output,tap_inside,output_inside)) == Kernel.cells(geometry) : U32}:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Groups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned,taps_exact,pixels,gradients} = domain
  {==}

def lane_read(+geometry: Kernel.Geometry,+domain: Groups.GeometryReady<geometry>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(geometry)) == True{} : Bool},+output_inside: {U32.is_lt(output,Groups.geometry_outputs(geometry)) == True{} : Bool},
  +tree: Tree.Tree<Tile.Tile8>) ->
  {Refinement.sum_query_read(lane_sum(geometry,domain,tap,output,tap_inside,output_inside),tree) ==
    TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),Refinement.tile_at(tree,Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),U32.to_nat(Packing.geometry_taps(geometry))),U32.to_nat(tap)))) : F32}:
  Groups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned,taps_exact,pixels,gradients} = domain
  {==}

def batch_cells(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Batch<X,xv,D,dv>) -> U32:
  Kernel.cells(batch_geometry(~X,~xv,~D,~dv,batch))

# What the weight kernel gives back for a batch: the holdings, which still
# cover the cells with their values, and the cells of the sums of the images,
# a lane of which is the adjacent tree of the lanes of the images.
type WeightDone<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-batch: Batch<X,xv,D,dv>,-tap: U32,-output: U32,
  -result: Kernel.Held() & Storage.Buffer<Tile.Tile8>> is Data:
  WeightDone{sums: Tree.Tree<Tile.Tile8>,
    contents: {Buffer.contents(~Tile.Tile8,Pair.snd(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)) ==
      Buffer.Content{Tree.write_sequence(~Tile.Tile8,Refinement.tiles(U32.to_nat(batch_cells(~X,~xv,~D,~dv,batch)),0n,sums),0,
        Tree.fresh(Tile.Tile8,Tensors.depth_for(batch_cells(~X,~xv,~D,~dv,batch)),Tile.filled(0.0))),U32.to_nat(batch_cells(~X,~xv,~D,~dv,batch))} : Buffer.Content<Tile.Tile8>},
    x_covered: Held.Covered<Tile.Tile8,X,xv,batch_input(~X,~xv,~D,~dv,batch),Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)),
      0n,batch_x_high(~X,~xv,~D,~dv,batch),batch_x_layout(~X,~xv,~D,~dv,batch)>,
    d_covered: Held.Covered<Tile.Tile8,D,dv,batch_gradient(~X,~xv,~D,~dv,batch),Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)),
      0n,batch_d_high(~X,~xv,~D,~dv,batch),batch_d_layout(~X,~xv,~D,~dv,batch)>,
    lane: {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),
        Refinement.tile_at(sums,Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),U32.to_nat(Packing.geometry_taps(batch_geometry(~X,~xv,~D,~dv,batch)))),U32.to_nat(tap)))) ==
      lane_levels(batch_images(~X,~xv,~D,~dv,batch),held_images(~X,~xv,~D,~dv,batch_images(~X,~xv,~D,~dv,batch),0n,batch,tap,output)) : F32}}

# What the sums of the images give back: the holdings, which still cover the
# cells with their values, and the cells of the sums, a lane of which is the
# adjacent tree of the lanes of the images.
type ReducedDone<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-batch: Batch<X,xv,D,dv>,-tap: U32,-output: U32,
  -result: Kernel.Held() & Array<Tile.Tile8>> is Data:
  ReducedDone{sums: Tree.Tree<Tile.Tile8>,
    values: {result == (Pair.fst(Kernel.Held(),Array<Tile.Tile8>,result),Tree.pack(Tile.Tile8,sums)) : Kernel.Held() & Array<Tile.Tile8>},
    x_covered: Held.Covered<Tile.Tile8,X,xv,batch_input(~X,~xv,~D,~dv,batch),Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Array<Tile.Tile8>,result)),
      0n,batch_x_high(~X,~xv,~D,~dv,batch),batch_x_layout(~X,~xv,~D,~dv,batch)>,
    d_covered: Held.Covered<Tile.Tile8,D,dv,batch_gradient(~X,~xv,~D,~dv,batch),Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Array<Tile.Tile8>,result)),
      0n,batch_d_high(~X,~xv,~D,~dv,batch),batch_d_layout(~X,~xv,~D,~dv,batch)>,
    lane: {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),
        Refinement.tile_at(sums,Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),U32.to_nat(Packing.geometry_taps(batch_geometry(~X,~xv,~D,~dv,batch)))),U32.to_nat(tap)))) ==
      lane_levels(batch_images(~X,~xv,~D,~dv,batch),held_images(~X,~xv,~D,~dv,batch_images(~X,~xv,~D,~dv,batch),0n,batch,tap,output)) : F32}}

# The images, reduced into the cells of the sums.
def weight_reduced(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Batch<X,xv,D,dv>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  -images: Kernel.Held() & List<&1,Array<Tile.Tile8>>,
  done: ImagesDone<X,xv,D,dv,batch,tap,output,batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside),[],batch_images(~X,~xv,~D,~dv,batch),0n,images>) ->
  ReducedDone<X,xv,D,dv,batch,tap,output,Kernel.reduced(batch_images(~X,~xv,~D,~dv,batch),batch_cells(~X,~xv,~D,~dv,batch),images)>:
  Batch{+x_holder,+d_holder,+geometry,+domain,+n,+rows,+columns,+group,+size,+bounded,+x_high,+x_layout,+x_room,+x_bounded,+d_high,+d_layout,+d_room,+d_bounded} = batch
  ImagesDone{+trees,+values,+x_covered,+d_covered,+frames,+observed} = done
  +count = Kernel.cells(geometry)
  +sum = lane_sum(geometry,domain,tap,output,tap_inside,output_inside)
  +tree = tree_levels(n,trees,count)
  +summed = Refinement.sum_tree(n,sum,trees,frames)
  +reduced = GroupLaws.levels_refines_model(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~(count => Kernel.array(~Tile.Tile8,count,Tile.filled(0.0))),~Refinement.zero,
    ~(count => left => right => Kernel.sum(count,left,right)),~{count => left => right => Refinement.sum(count,left,right) : U32 -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8>},
    ~(tree => Tree.pack(Tile.Tile8,tree)),~Refinement.zero_matches,~Refinement.sum_matches,n,count,trees)
  +counted = cells_count(geometry,domain,tap,output,tap_inside,output_inside)
  +lanes = {%counted : {Refinement.sum_query_read(sum,tree_levels(n,trees,_)) == lane_levels(n,observed_trees(sum,trees)) : F32}
    Shape.second(Refinement.sum_query_frame(sum,tree_levels(n,trees,Refinement.sum_query_count(sum))),
      {Refinement.sum_query_read(sum,tree_levels(n,trees,Refinement.sum_query_count(sum))) == lane_levels(n,observed_trees(sum,trees)) : F32},summed) :
    {Refinement.sum_query_read(sum,tree) == lane_levels(n,observed_trees(sum,trees)) : F32}}
  %Equal.sym(Kernel.Held() & List<&1,Array<Tile.Tile8>>,images,(Pair.fst(Kernel.Held(),List<&1,Array<Tile.Tile8>>,images),pack_trees(trees)),values) :
    ReducedDone<X,xv,D,dv,batch,tap,output,Kernel.reduced(n,count,_)>
  %Equal.sym(Array<Tile.Tile8>,Grouped.levels(~&1,~Array<Tile.Tile8>,~U32,~(count => Kernel.array(~Tile.Tile8,count,Tile.filled(0.0))),~(count => left => right => Kernel.sum(count,left,right)),n,count,
      pack_trees(trees)),Tree.pack(Tile.Tile8,tree),reduced) :
    ReducedDone<X,xv,D,dv,batch,tap,output,(Pair.fst(Kernel.Held(),List<&1,Array<Tile.Tile8>>,images),_)>
  ReducedDone{tree,{==},x_covered,d_covered,
    Equal.trans(F32,TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),Refinement.tile_at(tree,Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),U32.to_nat(Packing.geometry_taps(geometry))),U32.to_nat(tap)))),
      Refinement.sum_query_read(sum,tree),lane_levels(n,held_images(~X,~xv,~D,~dv,n,0n,batch,tap,output)),
      Equal.sym(F32,Refinement.sum_query_read(sum,tree),
        TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),Refinement.tile_at(tree,Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),U32.to_nat(Packing.geometry_taps(geometry))),U32.to_nat(tap)))),
        lane_read(geometry,domain,tap,output,tap_inside,output_inside,tree)),
      Equal.trans(F32,Refinement.sum_query_read(sum,tree),lane_levels(n,observed_trees(sum,trees)),lane_levels(n,held_images(~X,~xv,~D,~dv,n,0n,batch,tap,output)),lanes,
        Equal.cong(List<&2,F32>,F32,values => lane_levels(n,values),observed_trees(sum,trees),held_images(~X,~xv,~D,~dv,n,0n,batch,tap,output),observed)))}

# The sums, written into the cells of the result.
def reduced_written(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Batch<X,xv,D,dv>,+tap: U32,+output: U32,
  -result: Kernel.Held() & Array<Tile.Tile8>,done: ReducedDone<X,xv,D,dv,batch,tap,output,result>) ->
  WeightDone<X,xv,D,dv,batch,tap,output,Kernel.written(batch_cells(~X,~xv,~D,~dv,batch),result)>:
  ReducedDone{+sums,+values,+x_covered,+d_covered,+lane} = done
  %Equal.sym(Kernel.Held() & Array<Tile.Tile8>,result,(Pair.fst(Kernel.Held(),Array<Tile.Tile8>,result),Tree.pack(Tile.Tile8,sums)),values) :
    WeightDone<X,xv,D,dv,batch,tap,output,Kernel.written(batch_cells(~X,~xv,~D,~dv,batch),_)>
  WeightDone{sums,Refinement.tiles_fresh(batch_cells(~X,~xv,~D,~dv,batch),sums),x_covered,d_covered,lane}

# The images, reduced and written into the cells of the result.
def weight_finished(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Batch<X,xv,D,dv>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  -images: Kernel.Held() & List<&1,Array<Tile.Tile8>>,
  done: ImagesDone<X,xv,D,dv,batch,tap,output,batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside),[],batch_images(~X,~xv,~D,~dv,batch),0n,images>) ->
  WeightDone<X,xv,D,dv,batch,tap,output,Kernel.written(batch_cells(~X,~xv,~D,~dv,batch),Kernel.reduced(batch_images(~X,~xv,~D,~dv,batch),batch_cells(~X,~xv,~D,~dv,batch),images))>:
  reduced_written(~X,~xv,~D,~dv,batch,tap,output,Kernel.reduced(batch_images(~X,~xv,~D,~dv,batch),batch_cells(~X,~xv,~D,~dv,batch),images),
    weight_reduced(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside,images,done))

# The weight kernel over a batch: a lane of a cell of the result is the
# adjacent tree of the lanes of the images.
def batch_weight(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Batch<X,xv,D,dv>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  -x_held: Move.Holding<Tile.Tile8>,-d_held: Move.Holding<Tile.Tile8>,
  x_covered: Held.Covered<Tile.Tile8,X,xv,batch_input(~X,~xv,~D,~dv,batch),x_held,0n,batch_x_high(~X,~xv,~D,~dv,batch),batch_x_layout(~X,~xv,~D,~dv,batch)>,
  d_covered: Held.Covered<Tile.Tile8,D,dv,batch_gradient(~X,~xv,~D,~dv,batch),d_held,0n,batch_d_high(~X,~xv,~D,~dv,batch),batch_d_layout(~X,~xv,~D,~dv,batch)>) ->
  WeightDone<X,xv,D,dv,batch,tap,output,Kernel.written(batch_cells(~X,~xv,~D,~dv,batch),Kernel.reduced(batch_images(~X,~xv,~D,~dv,batch),batch_cells(~X,~xv,~D,~dv,batch),
    Kernel.images(batch_images(~X,~xv,~D,~dv,batch),0,batch_geometry(~X,~xv,~D,~dv,batch),batch_rows(~X,~xv,~D,~dv,batch),batch_columns(~X,~xv,~D,~dv,batch),batch_group(~X,~xv,~D,~dv,batch),
      x_held,d_held)))>:
  weight_finished(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside,
    Kernel.images(batch_images(~X,~xv,~D,~dv,batch),0,batch_geometry(~X,~xv,~D,~dv,batch),batch_rows(~X,~xv,~D,~dv,batch),batch_columns(~X,~xv,~D,~dv,batch),batch_group(~X,~xv,~D,~dv,batch),
      x_held,d_held),
    images_done(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside,x_held,d_held,x_covered,d_covered))
