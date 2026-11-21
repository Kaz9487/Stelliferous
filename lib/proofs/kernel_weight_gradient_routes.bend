# The weight kernel on the workers: parts of the images or of the output
# blocks. A part holds copies of the cells it reads, runs the serial kernel on
# them and gives back cells whose lanes are the lanes of the whole call.
import Base
import ../kernel_weight_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../kernel_parameter_gradient.bend as Parameter
import ../tensor_move.bend as Move
import ../ordered_matrix.bend as OrderedMatrix
import ../traversal_grouped.bend as Grouped
import ../machine_limits.bend as Machine
import ../kernel_gemm.bend as KernelGemm
import ../storage_buffer.bend as Storage
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ./kernel_gemm_output_groups.bend as OutputGroups
import ./kernel_weight_gradient_group.bend as Groups
import ./kernel_weight_gradient_packing.bend as Packing
import ./kernel_weight_gradient_held.bend as Sources
import ./kernel_weight_gradient_images.bend as Images
import ./kernel_weight_gradient_refinement.bend as Refinement
import ./storage_tree_model.bend as Tree
import ./tensor_move_readback.bend as Held
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_each.bend as Each
import ./tensor_pieces_plan.bend as PiecesPlan
import ./storage_buffer_model.bend as Buffer
import ./machine_limit_bounds.bend as Limits
import ./u32_full_range.bend as FullRange
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./kernel_tile_refinement.bend as TileLaws
import ./traversal_array_refinement.bend as Traversals
import ./ordered_reduction_split.bend as Ordered
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./nat_ceiling.bend as Ceiling
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./word_arithmetic.bend as WordArithmetic
import ../tensor.bend as Tensors
import ./tensor_view_gather.bend as Gather
import ./tensor_move_part.bend as Part
import ./tensor_view_part.bend as ViewPart
import ./tensor_grid_box.bend as Box
import ./nat_division.bend as NatDivision
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./boolean_logic.bend as Logic
import ./grouped_reduction_refinement.bend as GroupLaws
import ./nat_extrema.bend as NatExtrema
import ./u32_extrema.bend as Extrema
import ./word_encoding.bend as WordEncoding
import ./storage_refinement.bend as StorageLaws
import ./u32_comparison.bend as Comparison
import ./list_storage_readback.bend as ListRead
import ./tensor_list_values.bend as Listed
import ./encoded_coordinates.bend as Encoded

# ---- Lanes that read the same samples ----

# Two lanes read the same input and weight at a position.
def SameSample(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,~Y: Data,~yv: Y -> U32 -> Tile.Tile8,~E: Data,~ev: E -> U32 -> Tile.Tile8,
  +left: Images.HeldLane<X,D>,+right: Images.HeldLane<Y,E>,+position: Nat) -> Data:
  Images.HeldLane{+left_source,+left_tap,+left_output} = left
  Images.HeldLane{+right_source,+right_tap,+right_output} = right
  KernelGemm.Both<{Sources.held_input(~X,~xv,~D,~dv,left_source,left_output,position) == Sources.held_input(~Y,~yv,~E,~ev,right_source,right_output,position) : F32},
    {Sources.held_weight(~X,~xv,~D,~dv,left_source,left_tap,position) == Sources.held_weight(~Y,~yv,~E,~ev,right_source,right_tap,position) : F32}>

# Two lanes read the same samples at every position of a group.
def RangeSamples(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,~Y: Data,~yv: Y -> U32 -> Tile.Tile8,~E: Data,~ev: E -> U32 -> Tile.Tile8,
  +left: Images.HeldLane<X,D>,+right: Images.HeldLane<Y,E>,range: Grouped.Range) -> Data:
  Grouped.Range{+first,+count} = range
  Members.All(Nat,{position => SameSample(~X,~xv,~D,~dv,~Y,~yv,~E,~ev,left,right,position) : Nat -> Data},Ranges.range(count,first))

# Lanes that read the same samples have the same group products and the same
# lane of an image.
def same_group(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,~Y: Data,~yv: Y -> U32 -> Tile.Tile8,~E: Data,~ev: E -> U32 -> Tile.Tile8,
  +left: Images.HeldLane<X,D>,+right: Images.HeldLane<Y,E>,+range: Grouped.Range,samples: RangeSamples(~X,~xv,~D,~dv,~Y,~yv,~E,~ev,left,right,range)) ->
  {Images.held_sample(~X,~xv,~D,~dv,left,range) == Images.held_sample(~Y,~yv,~E,~ev,right,range) : F32}:
  Images.HeldLane{+left_source,+left_tap,+left_output} = left
  Images.HeldLane{+right_source,+right_tap,+right_output} = right
  Grouped.Range{+first,+count} = range
  Ordered.dot_congruence(~Nat,~U32,~U32,~Sources.HeldSource<X,D>,~U32,~U32,~Sources.HeldSource<Y,E>,~Sources.held_input(~X,~xv,~D,~dv),~Sources.held_weight(~X,~xv,~D,~dv),
    ~Sources.held_input(~Y,~yv,~E,~ev),~Sources.held_weight(~Y,~yv,~E,~ev),Ranges.range(count,first),left_tap,left_output,left_source,right_tap,right_output,right_source,0.0,samples)

def same_groups(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,~Y: Data,~yv: Y -> U32 -> Tile.Tile8,~E: Data,~ev: E -> U32 -> Tile.Tile8,
  +left: Images.HeldLane<X,D>,+right: Images.HeldLane<Y,E>,ranges: List<&2,Grouped.Range>,
  samples: Members.All(Grouped.Range,{range => RangeSamples(~X,~xv,~D,~dv,~Y,~yv,~E,~ev,left,right,range) : Grouped.Range -> Data},ranges)) ->
  {Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Images.HeldLane<X,D>,~Images.held_sample(~X,~xv,~D,~dv),ranges,left) ==
    Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Images.HeldLane<Y,E>,~Images.held_sample(~Y,~yv,~E,~ev),ranges,right) : List<&2,F32>}:
  match ranges:
    case Nil{}: {==}
    case +range <> +rest:
      KernelGemm.Both{+head,+tail} = samples
      %Equal.sym(F32,Images.held_sample(~X,~xv,~D,~dv,left,range),Images.held_sample(~Y,~yv,~E,~ev,right,range),same_group(~X,~xv,~D,~dv,~Y,~yv,~E,~ev,left,right,range,head)) :
        {_ <> Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Images.HeldLane<X,D>,~Images.held_sample(~X,~xv,~D,~dv),rest,left) ==
          Images.held_sample(~Y,~yv,~E,~ev,right,range) <> Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Images.HeldLane<Y,E>,~Images.held_sample(~Y,~yv,~E,~ev),rest,right) :
          List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,values => Images.held_sample(~Y,~yv,~E,~ev,right,range) <> values,
        Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Images.HeldLane<X,D>,~Images.held_sample(~X,~xv,~D,~dv),rest,left),
        Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Images.HeldLane<Y,E>,~Images.held_sample(~Y,~yv,~E,~ev),rest,right),
        same_groups(~X,~xv,~D,~dv,~Y,~yv,~E,~ev,left,right,rest,tail))

def same_image(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,~Y: Data,~yv: Y -> U32 -> Tile.Tile8,~E: Data,~ev: E -> U32 -> Tile.Tile8,
  +left: Images.HeldLane<X,D>,+right: Images.HeldLane<Y,E>,+ranges: List<&2,Grouped.Range>,
  samples: Members.All(Grouped.Range,{range => RangeSamples(~X,~xv,~D,~dv,~Y,~yv,~E,~ev,left,right,range) : Grouped.Range -> Data},ranges)) ->
  {Images.held_image(~X,~xv,~D,~dv,left,ranges) == Images.held_image(~Y,~yv,~E,~ev,right,ranges) : F32}:
  Equal.cong(List<&2,F32>,F32,values => Images.lane_levels(List.length(&2,Grouped.Range,ranges),values),
    Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Images.HeldLane<X,D>,~Images.held_sample(~X,~xv,~D,~dv),ranges,left),
    Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Images.HeldLane<Y,E>,~Images.held_sample(~Y,~yv,~E,~ev),ranges,right),
    same_groups(~X,~xv,~D,~dv,~Y,~yv,~E,~ev,left,right,ranges,samples))

# ---- The images of a copy ----

def regrouped(+offset: Nat,+first: Nat,+middle: Nat,+last: Nat) ->
  {Nat.add(offset,Nat.add(Nat.add(first,middle),last)) == Nat.add(Nat.add(Nat.add(offset,first),middle),last) : Nat}:
  Equal.sym(Nat,Nat.add(Nat.add(Nat.add(offset,first),middle),last),Nat.add(offset,Nat.add(Nat.add(first,middle),last)),
    Equal.trans(Nat,Nat.add(Nat.add(Nat.add(offset,first),middle),last),Nat.add(Nat.add(offset,Nat.add(first,middle)),last),Nat.add(offset,Nat.add(Nat.add(first,middle),last)),
      Equal.cong(Nat,Nat,value => Nat.add(value,last),Nat.add(Nat.add(offset,first),middle),Nat.add(offset,Nat.add(first,middle)),Addition.associative(offset,first,middle)),
      Addition.associative(offset,Nat.add(first,middle),last)))

# A cell read past a base offset is the cell read at the offset added.
def base_read(~H: Data,~hv: H -> U32 -> Tile.Tile8,+holder: H,+base: U32,+offset: Nat,+exact: {base == U32.from_nat(offset) : U32},
  +first: Nat,+middle: Nat,+last: Nat) ->
  {hv(holder,(base + U32.from_nat(Nat.add(Nat.add(first,middle),last)) : U32)) == hv(holder,U32.from_nat(Nat.add(Nat.add(Nat.add(offset,first),middle),last))) : Tile.Tile8}:
  Equal.cong(U32,Tile.Tile8,index => hv(holder,index),(base + U32.from_nat(Nat.add(Nat.add(first,middle),last)) : U32),U32.from_nat(Nat.add(Nat.add(Nat.add(offset,first),middle),last)),
    %Equal.sym(U32,base,U32.from_nat(offset),exact) :
      {(_ + U32.from_nat(Nat.add(Nat.add(first,middle),last)) : U32) == U32.from_nat(Nat.add(Nat.add(Nat.add(offset,first),middle),last)) : U32}
    Equal.trans(U32,(U32.from_nat(offset) + U32.from_nat(Nat.add(Nat.add(first,middle),last)) : U32),U32.from_nat(Nat.add(offset,Nat.add(Nat.add(first,middle),last))),
      U32.from_nat(Nat.add(Nat.add(Nat.add(offset,first),middle),last)),
      WordArithmetic.native_add(offset,Nat.add(Nat.add(first,middle),last)),
      Equal.cong(Nat,U32,value => U32.from_nat(value),Nat.add(offset,Nat.add(Nat.add(first,middle),last)),Nat.add(Nat.add(Nat.add(offset,first),middle),last),
        regrouped(offset,first,middle,last))))

# The gradient a copy holds from a base on is the gradient of the whole call
# past the base.
def shifted_input(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+input: X,+gradient: D,+geometry: Kernel.Geometry,+x_base: U32,+d_base: U32,
  +d_offset: Nat,+d_exact: {d_base == U32.from_nat(d_offset) : U32},+x_first: Nat,+d_first: Nat,+whole_x: Nat,+output: U32,+position: Nat) ->
  {Sources.held_input(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),
      Sources.HeldSource{Held.Shifted{input,x_base},Held.Shifted{gradient,d_base},geometry,x_first,d_first},output,position) ==
    Sources.held_input(~X,~xv,~D,~dv,Sources.HeldSource{input,gradient,geometry,whole_x,Nat.add(d_offset,d_first)},output,position) : F32}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),tile),
    Held.shifted_value(~Tile.Tile8,~D,~dv,Held.Shifted{gradient,d_base},U32.from_nat(Nat.add(Nat.add(d_first,Nat.mul(position,Ceiling.ceiling(U32.to_nat(o),8n))),Nat.div(U32.to_nat(output),8n)))),
    dv(gradient,U32.from_nat(Nat.add(Nat.add(Nat.add(d_offset,d_first),Nat.mul(position,Ceiling.ceiling(U32.to_nat(o),8n))),Nat.div(U32.to_nat(output),8n)))),
    base_read(~D,~dv,gradient,d_base,d_offset,d_exact,d_first,Nat.mul(position,Ceiling.ceiling(U32.to_nat(o),8n)),Nat.div(U32.to_nat(output),8n)))

# The input a copy holds from a base on is the input of the whole call past
# the base.
def shifted_weight(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+input: X,+gradient: D,+geometry: Kernel.Geometry,+x_base: U32,+d_base: U32,
  +x_offset: Nat,+x_exact: {x_base == U32.from_nat(x_offset) : U32},+x_first: Nat,+d_first: Nat,+whole_d: Nat,+tap: U32,+position: Nat) ->
  {Sources.held_weight(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),
      Sources.HeldSource{Held.Shifted{input,x_base},Held.Shifted{gradient,d_base},geometry,x_first,d_first},tap,position) ==
    Sources.held_weight(~X,~xv,~D,~dv,Sources.HeldSource{input,gradient,geometry,Nat.add(x_offset,x_first),whole_d},tap,position) : F32}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +row = U32.from_nat(Nat.div(position,U32.to_nat(columns)))
  +column = U32.from_nat(Nat.mod(position,U32.to_nat(columns)))
  +channel = Nat.div(U32.to_nat(tap),Nat.mul(U32.to_nat(k),U32.to_nat(k)))
  +y = Nat.add(Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(k)),Nat.mul(U32.to_nat(row),U32.to_nat(s)))
  +x_at = Nat.add(Nat.mod(U32.to_nat(tap),U32.to_nat(k)),Nat.mul(U32.to_nat(column),U32.to_nat(s)))
  +groups = U32.to_nat(((c + 7) / 8 : U32))
  Packing.sampled_agrees(~Held.Shifted<X>,~Sources.held_cell(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv)),~X,~Sources.held_cell(~X,~xv),
    Held.Shifted{input,x_base},input,x_first,Nat.add(x_offset,x_first),channel,y,x_at,U32.to_nat(h),U32.to_nat(w),U32.to_nat(p),groups,
    inside_true => base_read(~X,~xv,input,x_base,x_offset,x_exact,x_first,Nat.div(channel,8n),
      Nat.mul(Nat.add(Nat.mul(Nat.sub(y,U32.to_nat(p)),U32.to_nat(w)),Nat.sub(x_at,U32.to_nat(p))),groups)))

# Where a copy is read from a base on: the holders, their bases and offsets,
# and the first cells of the image in the copy and in the whole call.
type CopyRead<-X: Data,-D: Data> is Data:
  CopyRead{input: X,gradient: D,geometry: Kernel.Geometry,x_base: U32,d_base: U32,x_offset: Nat,d_offset: Nat,
    x_exact: {x_base == U32.from_nat(x_offset) : U32},d_exact: {d_base == U32.from_nat(d_offset) : U32},x_first: Nat,d_first: Nat}

def copy_source(~X: Data,~D: Data,read: CopyRead<X,D>) -> Sources.HeldSource<Held.Shifted<X>,Held.Shifted<D>>:
  CopyRead{input,gradient,geometry,x_base,d_base,x_offset,d_offset,x_exact,d_exact,x_first,d_first} = read
  Sources.HeldSource{Held.Shifted{input,x_base},Held.Shifted{gradient,d_base},geometry,x_first,d_first}

def whole_source(~X: Data,~D: Data,read: CopyRead<X,D>) -> Sources.HeldSource<X,D>:
  CopyRead{input,gradient,geometry,x_base,d_base,+x_offset,+d_offset,x_exact,d_exact,+x_first,+d_first} = read
  Sources.HeldSource{input,gradient,geometry,Nat.add(x_offset,x_first),Nat.add(d_offset,d_first)}

# A copy read from a base on and the whole call read the same samples.
def copy_lane(~X: Data,~D: Data,read: CopyRead<X,D>,+tap: U32,+output: U32) -> Images.HeldLane<Held.Shifted<X>,Held.Shifted<D>>:
  Images.HeldLane{copy_source(~X,~D,read),tap,output}

def whole_lane(~X: Data,~D: Data,read: CopyRead<X,D>,+tap: U32,+output: U32) -> Images.HeldLane<X,D>:
  Images.HeldLane{whole_source(~X,~D,read),tap,output}

def copy_sample(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+read: CopyRead<X,D>,+tap: U32,+output: U32,+position: Nat) ->
  SameSample(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),~X,~xv,~D,~dv,
    copy_lane(~X,~D,read,tap,output),whole_lane(~X,~D,read,tap,output),position):
  CopyRead{+input,+gradient,+geometry,+x_base,+d_base,+x_offset,+d_offset,+x_exact,+d_exact,+x_first,+d_first} = read
  KernelGemm.Both{shifted_input(~X,~xv,~D,~dv,input,gradient,geometry,x_base,d_base,d_offset,d_exact,x_first,d_first,Nat.add(x_offset,x_first),output,position),
    shifted_weight(~X,~xv,~D,~dv,input,gradient,geometry,x_base,d_base,x_offset,x_exact,x_first,d_first,Nat.add(d_offset,d_first),tap,position)}

def copy_samples(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+read: CopyRead<X,D>,+tap: U32,+output: U32,keys: List<&2,Nat>) ->
  Members.All(Nat,{position => SameSample(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),~X,~xv,~D,~dv,
    copy_lane(~X,~D,read,tap,output),whole_lane(~X,~D,read,tap,output),position) : Nat -> Data},keys):
  match keys:
    case Nil{}: Unit{}
    case +position <> +rest: KernelGemm.Both{copy_sample(~X,~xv,~D,~dv,read,tap,output,position),copy_samples(~X,~xv,~D,~dv,read,tap,output,rest)}

def copy_range(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+read: CopyRead<X,D>,+tap: U32,+output: U32,+range: Grouped.Range) ->
  RangeSamples(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),~X,~xv,~D,~dv,
    copy_lane(~X,~D,read,tap,output),whole_lane(~X,~D,read,tap,output),range):
  Grouped.Range{+first,+count} = range
  copy_samples(~X,~xv,~D,~dv,read,tap,output,Ranges.range(count,first))

def copy_ranges(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+read: CopyRead<X,D>,+tap: U32,+output: U32,ranges: List<&2,Grouped.Range>) ->
  Members.All(Grouped.Range,{range => RangeSamples(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),~X,~xv,~D,~dv,
    copy_lane(~X,~D,read,tap,output),whole_lane(~X,~D,read,tap,output),range) : Grouped.Range -> Data},ranges):
  match ranges:
    case Nil{}: Unit{}
    case +range <> +rest: KernelGemm.Both{copy_range(~X,~xv,~D,~dv,read,tap,output,range),copy_ranges(~X,~xv,~D,~dv,read,tap,output,rest)}

# ---- The images of a copy of consecutive images ----

def batch_domain(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>) ->
  Groups.GeometryReady<Images.batch_geometry(~X,~xv,~D,~dv,batch)>:
  Images.Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  domain

# The images of a batch that copies of its holdings hold, from bases on, with
# the shape of the batch.
def copy_batch(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+whole: Images.Batch<X,xv,D,dv>,+x_base: U32,+d_base: U32,+count: Nat,
  +x_high: Nat,+x_layout: Held.Spans,
  +x_room: {Nat.is_le(Nat.mul(count,Sources.input_cells(Images.batch_geometry(~X,~xv,~D,~dv,whole))),x_high) == True{} : Bool},
  +x_bounded: {Nat.is_le(x_high,U32.to_nat(Machine.capacity(Groups.domain_limits(Images.batch_geometry(~X,~xv,~D,~dv,whole),batch_domain(~X,~xv,~D,~dv,whole))))) == True{} : Bool},
  +d_high: Nat,+d_layout: Held.Spans,
  +d_room: {Nat.is_le(Nat.mul(count,Groups.gradient_extent(Images.batch_geometry(~X,~xv,~D,~dv,whole))),d_high) == True{} : Bool},
  +d_bounded: {Nat.is_le(d_high,U32.to_nat(Machine.capacity(Groups.domain_limits(Images.batch_geometry(~X,~xv,~D,~dv,whole),batch_domain(~X,~xv,~D,~dv,whole))))) == True{} : Bool}) ->
  Images.Batch<Held.Shifted<X>,Held.shifted_value(~Tile.Tile8,~X,~xv),Held.Shifted<D>,Held.shifted_value(~Tile.Tile8,~D,~dv)>:
  Images.Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,whole_x_high,whole_x_layout,whole_x_room,whole_x_bounded,
    whole_d_high,whole_d_layout,whole_d_room,whole_d_bounded} = whole
  Images.Batch{Held.Shifted{input,x_base},Held.Shifted{gradient,d_base},geometry,domain,count,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,
    d_high,d_layout,d_room,d_bounded}

def moved_first(+image: Nat,+first: Nat,+cells: Nat) -> {Nat.add(Nat.mul(first,cells),Nat.mul(image,cells)) == Nat.mul(Nat.add(image,first),cells) : Nat}:
  Equal.sym(Nat,Nat.mul(Nat.add(image,first),cells),Nat.add(Nat.mul(first,cells),Nat.mul(image,cells)),
    Equal.trans(Nat,Nat.mul(Nat.add(image,first),cells),Nat.add(Nat.mul(image,cells),Nat.mul(first,cells)),Nat.add(Nat.mul(first,cells),Nat.mul(image,cells)),
      RowMajor.distrib(image,first,cells),Order.commutative(Nat.mul(image,cells),Nat.mul(first,cells))))

# The lanes of the images of a copy from image on are the lanes of the images
# of the batch from first + image on.
def copy_images(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,remaining: Nat,+image: Nat,+first: Nat,+whole: Images.Batch<X,xv,D,dv>,
  +x_base: U32,+d_base: U32,+count: Nat,+x_high: Nat,+x_layout: Held.Spans,
  +x_room: {Nat.is_le(Nat.mul(count,Sources.input_cells(Images.batch_geometry(~X,~xv,~D,~dv,whole))),x_high) == True{} : Bool},
  +x_bounded: {Nat.is_le(x_high,U32.to_nat(Machine.capacity(Groups.domain_limits(Images.batch_geometry(~X,~xv,~D,~dv,whole),batch_domain(~X,~xv,~D,~dv,whole))))) == True{} : Bool},
  +d_high: Nat,+d_layout: Held.Spans,
  +d_room: {Nat.is_le(Nat.mul(count,Groups.gradient_extent(Images.batch_geometry(~X,~xv,~D,~dv,whole))),d_high) == True{} : Bool},
  +d_bounded: {Nat.is_le(d_high,U32.to_nat(Machine.capacity(Groups.domain_limits(Images.batch_geometry(~X,~xv,~D,~dv,whole),batch_domain(~X,~xv,~D,~dv,whole))))) == True{} : Bool},
  +x_exact: {x_base == U32.from_nat(Nat.mul(first,Sources.input_cells(Images.batch_geometry(~X,~xv,~D,~dv,whole)))) : U32},
  +d_exact: {d_base == U32.from_nat(Nat.mul(first,Groups.gradient_extent(Images.batch_geometry(~X,~xv,~D,~dv,whole)))) : U32},+tap: U32,+output: U32) ->
  {Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),remaining,image,
      copy_batch(~X,~xv,~D,~dv,whole,x_base,d_base,count,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded),tap,output) ==
    Images.held_images(~X,~xv,~D,~dv,remaining,Nat.add(image,first),whole,tap,output) : List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Images.Batch{+input,+gradient,+geometry,+domain,+images,+rows,+columns,+group,+size,+bounded,+whole_x_high,+whole_x_layout,+whole_x_room,+whole_x_bounded,
        +whole_d_high,+whole_d_layout,+whole_d_room,+whole_d_bounded} = whole
      +cells = Sources.input_cells(geometry)
      +extent = Groups.gradient_extent(geometry)
      +ranges = Grouped.ranges(rows,columns,group)
      +read = {CopyRead{input,gradient,geometry,x_base,d_base,Nat.mul(first,cells),Nat.mul(first,extent),x_exact,d_exact,Nat.mul(image,cells),Nat.mul(image,extent)} : CopyRead<X,D>}
      +copy_head = Images.held_image(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),copy_lane(~X,~D,read,tap,output),ranges)
      +whole_head = Images.held_image(~X,~xv,~D,~dv,
        Images.HeldLane{Sources.HeldSource{input,gradient,geometry,Nat.mul(Nat.add(image,first),cells),Nat.mul(Nat.add(image,first),extent)},tap,output},ranges)
      +moved = {%moved_first(image,first,cells) :
          {Images.held_image(~X,~xv,~D,~dv,whole_lane(~X,~D,read,tap,output),ranges) ==
            Images.held_image(~X,~xv,~D,~dv,Images.HeldLane{Sources.HeldSource{input,gradient,geometry,_,Nat.mul(Nat.add(image,first),extent)},tap,output},ranges) : F32}
        %moved_first(image,first,extent) :
          {Images.held_image(~X,~xv,~D,~dv,whole_lane(~X,~D,read,tap,output),ranges) ==
            Images.held_image(~X,~xv,~D,~dv,Images.HeldLane{Sources.HeldSource{input,gradient,geometry,Nat.add(Nat.mul(first,cells),Nat.mul(image,cells)),_},tap,output},ranges) : F32}
        {==} : {Images.held_image(~X,~xv,~D,~dv,whole_lane(~X,~D,read,tap,output),ranges) == whole_head : F32}}
      +copy_tail = Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),rest,1n+image,
        copy_batch(~X,~xv,~D,~dv,Images.Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,whole_x_high,whole_x_layout,whole_x_room,whole_x_bounded,
          whole_d_high,whole_d_layout,whole_d_room,whole_d_bounded},x_base,d_base,count,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded),tap,output)
      +whole_tail = Images.held_images(~X,~xv,~D,~dv,rest,1n+Nat.add(image,first),
        Images.Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,whole_x_high,whole_x_layout,whole_x_room,whole_x_bounded,
          whole_d_high,whole_d_layout,whole_d_room,whole_d_bounded},tap,output)
      Equal.trans(List<&2,F32>,copy_head <> copy_tail,whole_head <> copy_tail,whole_head <> whole_tail,
        Equal.cong(F32,List<&2,F32>,head => head <> copy_tail,copy_head,whole_head,
          Equal.trans(F32,copy_head,Images.held_image(~X,~xv,~D,~dv,whole_lane(~X,~D,read,tap,output),ranges),whole_head,same_image(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),~X,~xv,~D,~dv,
            copy_lane(~X,~D,read,tap,output),whole_lane(~X,~D,read,tap,output),ranges,copy_ranges(~X,~xv,~D,~dv,read,tap,output,ranges)),moved)),
        Equal.cong(List<&2,F32>,List<&2,F32>,tail => whole_head <> tail,copy_tail,whole_tail,
          copy_images(~X,~xv,~D,~dv,rest,1n+image,first,
            Images.Batch{input,gradient,geometry,domain,images,rows,columns,group,size,bounded,whole_x_high,whole_x_layout,whole_x_room,whole_x_bounded,
              whole_d_high,whole_d_layout,whole_d_room,whole_d_bounded},x_base,d_base,count,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded,x_exact,d_exact,tap,output)))

# The lanes of before + after images are the lanes of the first before and
# then of the next after.
def images_joined(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,before: Nat,+after: Nat,+image: Nat,+batch: Images.Batch<X,xv,D,dv>,+tap: U32,+output: U32) ->
  {Images.held_images(~X,~xv,~D,~dv,Nat.add(before,after),image,batch,tap,output) ==
    OrderedMatrix.cat(F32,Images.held_images(~X,~xv,~D,~dv,before,image,batch,tap,output),Images.held_images(~X,~xv,~D,~dv,after,Nat.add(image,before),batch,tap,output)) : List<&2,F32>}:
  match before:
    case 0n:
      %Equal.sym(Nat,Nat.add(image,0n),image,Addition.zero_right(image)) :
        {Images.held_images(~X,~xv,~D,~dv,after,image,batch,tap,output) == Images.held_images(~X,~xv,~D,~dv,after,_,batch,tap,output) : List<&2,F32>}
      {==}
    case 1n+ +rest:
      +head = Images.held_image(~X,~xv,~D,~dv,Images.batch_lane(~X,~xv,~D,~dv,batch,image,tap,output),Images.batch_ranges(~X,~xv,~D,~dv,batch))
      %Equal.sym(Nat,Nat.add(image,1n+rest),1n+Nat.add(image,rest),Addition.successor_right(image,rest)) :
        {head <> Images.held_images(~X,~xv,~D,~dv,Nat.add(rest,after),1n+image,batch,tap,output) ==
          head <> OrderedMatrix.cat(F32,Images.held_images(~X,~xv,~D,~dv,rest,1n+image,batch,tap,output),Images.held_images(~X,~xv,~D,~dv,after,_,batch,tap,output)) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => head <> tail,Images.held_images(~X,~xv,~D,~dv,Nat.add(rest,after),1n+image,batch,tap,output),
        OrderedMatrix.cat(F32,Images.held_images(~X,~xv,~D,~dv,rest,1n+image,batch,tap,output),Images.held_images(~X,~xv,~D,~dv,after,1n+Nat.add(image,rest),batch,tap,output)),
        images_joined(~X,~xv,~D,~dv,rest,after,1n+image,batch,tap,output))


# ---- Parts made from copies ----

# The models of the two holdings the parts copy from, covering the cells of a
# batch with its values.
type HeldBoth<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-batch: Images.Batch<X,xv,D,dv>,-x: Held.Holding<Tile.Tile8>,-d: Held.Holding<Tile.Tile8>> is Data:
  HeldBoth{x_tiled: Held.Tiled(~Tile.Tile8,x,0n,Images.batch_x_high(~X,~xv,~D,~dv,batch)),x_values: Held.HoldingValues(~Tile.Tile8,~X,~xv,Images.batch_input(~X,~xv,~D,~dv,batch),x),
    x_ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,x)) == Images.batch_x_layout(~X,~xv,~D,~dv,batch) : Held.Spans},
    d_tiled: Held.Tiled(~Tile.Tile8,d,0n,Images.batch_d_high(~X,~xv,~D,~dv,batch)),d_values: Held.HoldingValues(~Tile.Tile8,~D,~dv,Images.batch_gradient(~X,~xv,~D,~dv,batch),d),
    d_ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,d)) == Images.batch_d_layout(~X,~xv,~D,~dv,batch) : Held.Spans}}

def Owned() -> Type:
  Kernel.Owners & Partition.Tree<&1,Kernel.Owners,Unit>

def holders(+x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>) -> Kernel.Owners:
  Kernel.Owners{Held.pack(~Tile.Tile8,x),Held.pack(~Tile.Tile8,d)}

# What the parts made along a plan give back: the holdings they copied from,
# which still cover the cells of the batch, and the parts with a fact of each.
type PartsDone<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-Part: Data,-pack: Part -> Kernel.Owners,-Context: Data,
  -Fact: @+scope: KernelGemm.Both<Images.Batch<X,xv,D,dv>,Context> -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  -batch: Images.Batch<X,xv,D,dv>,-context: Context,-layout: Pieces.Ranges,-result: Owned()> is Data:
  PartsDone{x: Held.Holding<Tile.Tile8>,d: Held.Holding<Tile.Tile8>,
    state: {Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,result) == holders(x,d) : Kernel.Owners},
    holds: HeldBoth<X,xv,D,dv,batch,x,d>,parts: PiecesModel.Shaped<Part>,
    packed: {PiecesModel.shaped_pack(~Part,~Kernel.Owners,~pack,parts) == Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,result) : Partition.Tree<&1,Kernel.Owners,Unit>},
    planned: {PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),parts) == layout : Pieces.Ranges},
    facts: PiecesModel.ShapedAll(~Part,~KernelGemm.Both<Images.Batch<X,xv,D,dv>,Context>,~Fact,KernelGemm.Both{batch,context},parts)}

# The leaf of a build of parts: on a range of the plan below the limit it
# copies from covering holdings and makes one part with its fact.
def PartRule(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,~Part: Data,~pack: Part -> Kernel.Owners,~Context: Data,
  ~Fact: @+scope: KernelGemm.Both<Images.Batch<X,xv,D,dv>,Context> -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  ~Ready: Images.Batch<X,xv,D,dv> -> Context -> Nat -> Data,~leaf: Context -> U32 -> U32 -> Kernel.Owners -> Owned()) -> Type:
  @+batch: Images.Batch<X,xv,D,dv> -> @+context: Context -> @+limit: Nat -> @+ready: Ready(batch,context,limit) -> @+first: U32 -> @+rows: U32 ->
    @+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),limit) == True{} : Bool} ->
    @+x: Held.Holding<Tile.Tile8> -> @+d: Held.Holding<Tile.Tile8> -> @+holds: HeldBoth<X,xv,D,dv,batch,x,d> ->
    PartsDone<X,xv,D,dv,Part,pack,Context,Fact,batch,context,Pieces.Range{first,rows},leaf(context,first,rows,holders(x,d))>

def RightParts(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,~Part: Data,~pack: Part -> Kernel.Owners,~Context: Data,
  ~Fact: @+scope: KernelGemm.Both<Images.Batch<X,xv,D,dv>,Context> -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  ~leaf: Context -> U32 -> U32 -> Kernel.Owners -> Owned(),+batch: Images.Batch<X,xv,D,dv>,+context: Context,+layout: Pieces.Ranges) -> Type:
  @+x: Held.Holding<Tile.Tile8> -> @+d: Held.Holding<Tile.Tile8> -> @+holds: HeldBoth<X,xv,D,dv,batch,x,d> ->
    PartsDone<X,xv,D,dv,Part,pack,Context,Fact,batch,context,layout,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,layout,context,holders(x,d))>

# Both halves of a split: the right half from the holdings the left gives back.
def parts_joined(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,~Part: Data,~pack: Part -> Kernel.Owners,~Context: Data,
  ~Fact: @+scope: KernelGemm.Both<Images.Batch<X,xv,D,dv>,Context> -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  +batch: Images.Batch<X,xv,D,dv>,+context: Context,+left_plan: Pieces.Ranges,+right_plan: Pieces.Ranges,-left_result: Owned(),-right_result: Owned(),
  +left_parts: PiecesModel.Shaped<Part>,
  +left_packed: {PiecesModel.shaped_pack(~Part,~Kernel.Owners,~pack,left_parts) == Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,left_result) : Partition.Tree<&1,Kernel.Owners,Unit>},
  +left_planned: {PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),left_parts) == left_plan : Pieces.Ranges},
  +left_facts: PiecesModel.ShapedAll(~Part,~KernelGemm.Both<Images.Batch<X,xv,D,dv>,Context>,~Fact,KernelGemm.Both{batch,context},left_parts),
  right: PartsDone<X,xv,D,dv,Part,pack,Context,Fact,batch,context,right_plan,right_result>) ->
  PartsDone<X,xv,D,dv,Part,pack,Context,Fact,batch,context,Pieces.Split{left_plan,right_plan},
    (Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,right_result),
      Partition.Fork{Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,left_result),Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,right_result)})>:
  PartsDone{+x,+d,+state,+holds,+right_parts,+right_packed,+right_planned,+right_facts} = right
  PartsDone{x,d,state,holds,PiecesModel.Pieces{left_parts,right_parts},
    %left_packed : {Partition.Fork{PiecesModel.shaped_pack(~Part,~Kernel.Owners,~pack,left_parts),PiecesModel.shaped_pack(~Part,~Kernel.Owners,~pack,right_parts)} ==
      Partition.Fork{_,Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,right_result)} : Partition.Tree<&1,Kernel.Owners,Unit>}
    %right_packed : {Partition.Fork{PiecesModel.shaped_pack(~Part,~Kernel.Owners,~pack,left_parts),PiecesModel.shaped_pack(~Part,~Kernel.Owners,~pack,right_parts)} ==
      Partition.Fork{PiecesModel.shaped_pack(~Part,~Kernel.Owners,~pack,left_parts),_} : Partition.Tree<&1,Kernel.Owners,Unit>}
    {==},
    %left_planned : {Pieces.Split{PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),left_parts),PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),right_parts)} ==
      Pieces.Split{_,right_plan} : Pieces.Ranges}
    %right_planned : {Pieces.Split{PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),left_parts),PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),right_parts)} ==
      Pieces.Split{PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),left_parts),_} : Pieces.Ranges}
    {==},
    KernelGemm.Both{left_facts,right_facts}}

def parts_split(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,~Part: Data,~pack: Part -> Kernel.Owners,~Context: Data,
  ~Fact: @+scope: KernelGemm.Both<Images.Batch<X,xv,D,dv>,Context> -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  ~leaf: Context -> U32 -> U32 -> Kernel.Owners -> Owned(),+batch: Images.Batch<X,xv,D,dv>,+context: Context,+left_plan: Pieces.Ranges,+right_plan: Pieces.Ranges,
  -left_result: Owned(),left: PartsDone<X,xv,D,dv,Part,pack,Context,Fact,batch,context,left_plan,left_result>,
  right: RightParts(~X,~xv,~D,~dv,~Part,~pack,~Context,~Fact,~leaf,batch,context,right_plan)) ->
  PartsDone<X,xv,D,dv,Part,pack,Context,Fact,batch,context,Pieces.Split{left_plan,right_plan},
    Storage.bind_pair(~Kernel.Owners,~Partition.Tree<&1,Kernel.Owners,Unit>,~Owned(),
      Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,right_plan,context,Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,left_result)),
      state => right => (state,Partition.Fork{Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,left_result),right}))>:
  PartsDone{+x,+d,+state,+holds,+left_parts,+left_packed,+left_planned,+left_facts} = left
  %Equal.sym(Kernel.Owners,Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,left_result),holders(x,d),state) :
    PartsDone<X,xv,D,dv,Part,pack,Context,Fact,batch,context,Pieces.Split{left_plan,right_plan},
      Storage.bind_pair(~Kernel.Owners,~Partition.Tree<&1,Kernel.Owners,Unit>,~Owned(),Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,right_plan,context,_),
        state => right => (state,Partition.Fork{Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,left_result),right}))>
  %Equal.sym(Owned(),Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,right_plan,context,holders(x,d)),
      (Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,right_plan,context,holders(x,d))),
        Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,right_plan,context,holders(x,d)))),
      Held.pair_eta(~Kernel.Owners,~Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,right_plan,context,holders(x,d)))) :
    PartsDone<X,xv,D,dv,Part,pack,Context,Fact,batch,context,Pieces.Split{left_plan,right_plan},
      Storage.bind_pair(~Kernel.Owners,~Partition.Tree<&1,Kernel.Owners,Unit>,~Owned(),_,
        state => right => (state,Partition.Fork{Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,left_result),right}))>
  parts_joined(~X,~xv,~D,~dv,~Part,~pack,~Context,~Fact,batch,context,left_plan,right_plan,left_result,
    Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,right_plan,context,holders(x,d)),left_parts,left_packed,left_planned,left_facts,right(x,d,holds))

# The parts made along a plan of ranges [start, start + count) below the
# limit, from holdings that cover the cells of the batch.
def parts_made(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,~Part: Data,~pack: Part -> Kernel.Owners,~Context: Data,
  ~Fact: @+scope: KernelGemm.Both<Images.Batch<X,xv,D,dv>,Context> -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  ~Ready: Images.Batch<X,xv,D,dv> -> Context -> Nat -> Data,~leaf: Context -> U32 -> U32 -> Kernel.Owners -> Owned(),
  ~rule: PartRule(~X,~xv,~D,~dv,~Part,~pack,~Context,~Fact,~Ready,~leaf),layout: Pieces.Ranges,+batch: Images.Batch<X,xv,D,dv>,+context: Context,
  +limit: Nat,+ready: Ready(batch,context,limit),+start: Nat,+count: Nat,+coverage: PiecesPlan.Coverage(layout,start,count),
  +room: {Nat.is_le(Nat.add(start,count),limit) == True{} : Bool},+x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>,+holds: HeldBoth<X,xv,D,dv,batch,x,d>) ->
  PartsDone<X,xv,D,dv,Part,pack,Context,Fact,batch,context,layout,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,layout,context,holders(x,d))>:
  match layout:
    case Pieces.Range{+first,+rows}:
      PiecesPlan.LeafCoverage{+first_equal,+rows_equal} = coverage
      rule(batch,context,limit,ready,first,rows,
        %Equal.sym(Nat,U32.to_nat(first),start,first_equal) : {Nat.is_le(Nat.add(_,U32.to_nat(rows)),limit) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(rows),count,rows_equal) : {Nat.is_le(Nat.add(start,_),limit) == True{} : Bool}
        room,x,d,holds)
    case Pieces.Split{+left_plan,+right_plan}:
      PiecesPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
      +whole = {%rows_equal : {Nat.is_le(Nat.add(start,_),limit) == True{} : Bool} room : {Nat.is_le(Nat.add(start,Nat.add(left_rows,right_rows)),limit) == True{} : Bool}}
      +left_room = Order.transitive(Nat.add(start,left_rows),Nat.add(start,Nat.add(left_rows,right_rows)),limit,
        Order.offset_le(start,left_rows,Nat.add(left_rows,right_rows),Order.le_add(left_rows,right_rows)),whole)
      +right_room = {%Equal.sym(Nat,Nat.add(Nat.add(start,left_rows),right_rows),Nat.add(start,Nat.add(left_rows,right_rows)),Addition.associative(start,left_rows,right_rows)) :
          {Nat.is_le(_,limit) == True{} : Bool}
        whole : {Nat.is_le(Nat.add(Nat.add(start,left_rows),right_rows),limit) == True{} : Bool}}
      %Equal.sym(Owned(),Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,left_plan,context,holders(x,d)),
          (Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,left_plan,context,holders(x,d))),
            Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,left_plan,context,holders(x,d)))),
          Held.pair_eta(~Kernel.Owners,~Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,left_plan,context,holders(x,d)))) :
        PartsDone<X,xv,D,dv,Part,pack,Context,Fact,batch,context,Pieces.Split{left_plan,right_plan},
          Storage.bind_pair(~Kernel.Owners,~Partition.Tree<&1,Kernel.Owners,Unit>,~Owned(),_,
            state => left => Storage.bind_pair(~Kernel.Owners,~Partition.Tree<&1,Kernel.Owners,Unit>,~Owned(),
              Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,right_plan,context,state),state => right => (state,Partition.Fork{left,right})))>
      parts_split(~X,~xv,~D,~dv,~Part,~pack,~Context,~Fact,~leaf,batch,context,left_plan,right_plan,
        Pieces.build(~Kernel.Owners,~Kernel.Owners,~Context,~leaf,left_plan,context,holders(x,d)),
        parts_made(~X,~xv,~D,~dv,~Part,~pack,~Context,~Fact,~Ready,~leaf,~rule,left_plan,batch,context,limit,ready,start,left_rows,left_coverage,left_room,x,d,holds),
        right_x => right_d => right_holds => parts_made(~X,~xv,~D,~dv,~Part,~pack,~Context,~Fact,~Ready,~leaf,~rule,right_plan,batch,context,limit,ready,
          Nat.add(start,left_rows),right_rows,right_coverage,right_room,right_x,right_d,right_holds))


# The owners of a part: copies of its input cells and gradient cells, each
# held as the only piece.
type CopiedPart is Data:
  CopiedPart{input: Buffer.Model<Tile.Tile8>,x_count: U32,gradient: Buffer.Model<Tile.Tile8>,d_count: U32}

def copied_owners(part: CopiedPart) -> Kernel.Owners:
  CopiedPart{input,+x_count,gradient,+d_count} = part
  Kernel.Owners{Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,input)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,gradient))}

def copy_model(copied: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,Pair.snd(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,copied))

# A copy is what it leaves of its source and the buffer of its model.
def copy_split(copied: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) ->
  {copied == (Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,copied),Buffer.pack(~Tile.Tile8,copy_model(copied))) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  (source,copy) = copied
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,copy)),copy,Buffer.remodeled(~Tile.Tile8,copy)) :
    {(source,copy) == (source,_) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  {==}

# The part a leaf makes from the copies of the holdings, and what it leaves of
# them.
def copies_done(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,~Context: Data,
  ~Fact: @+scope: KernelGemm.Both<Images.Batch<X,xv,D,dv>,Context> -> @+part: CopiedPart -> @+first: U32 -> @+rows: U32 -> Data,
  +batch: Images.Batch<X,xv,D,dv>,+context: Context,+first: U32,+rows: U32,-inputs: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,+x_count: U32,
  -gradients: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,+d_count: U32,
  +x_split: {inputs == (Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,inputs),Buffer.pack(~Tile.Tile8,copy_model(inputs))) :
    Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>},
  +d_split: {gradients == (Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,gradients),Buffer.pack(~Tile.Tile8,copy_model(gradients))) :
    Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>},
  x_covered: Held.Covered<Tile.Tile8,X,xv,Images.batch_input(~X,~xv,~D,~dv,batch),Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,inputs),
    0n,Images.batch_x_high(~X,~xv,~D,~dv,batch),Images.batch_x_layout(~X,~xv,~D,~dv,batch)>,
  d_covered: Held.Covered<Tile.Tile8,D,dv,Images.batch_gradient(~X,~xv,~D,~dv,batch),Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,gradients),
    0n,Images.batch_d_high(~X,~xv,~D,~dv,batch),Images.batch_d_layout(~X,~xv,~D,~dv,batch)>,
  +x_copy: Buffer.Model<Tile.Tile8>,+x_model: {copy_model(inputs) == x_copy : Buffer.Model<Tile.Tile8>},
  +d_copy: Buffer.Model<Tile.Tile8>,+d_model: {copy_model(gradients) == d_copy : Buffer.Model<Tile.Tile8>},
  +fact: Fact(KernelGemm.Both{batch,context},CopiedPart{x_copy,x_count,d_copy,d_count},first,rows)) ->
  PartsDone<X,xv,D,dv,CopiedPart,copied_owners,Context,Fact,batch,context,Pieces.Range{first,rows},Kernel.owned(first,rows,x_count,d_count,inputs,gradients)>:
  Held.Covered{+x,+x_packed,+x_tiled,+x_values,+x_ordered} = x_covered
  Held.Covered{+d,+d_packed,+d_tiled,+d_values,+d_ordered} = d_covered
  %Equal.sym(Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,inputs,(Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,inputs),Buffer.pack(~Tile.Tile8,copy_model(inputs))),x_split) :
    PartsDone<X,xv,D,dv,CopiedPart,copied_owners,Context,Fact,batch,context,Pieces.Range{first,rows},Kernel.owned(first,rows,x_count,d_count,_,gradients)>
  %Equal.sym(Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,gradients,(Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,gradients),Buffer.pack(~Tile.Tile8,copy_model(gradients))),d_split) :
    PartsDone<X,xv,D,dv,CopiedPart,copied_owners,Context,Fact,batch,context,Pieces.Range{first,rows},
      Kernel.owned(first,rows,x_count,d_count,(Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,inputs),Buffer.pack(~Tile.Tile8,copy_model(inputs))),_)>
  %Equal.sym(Buffer.Model<Tile.Tile8>,copy_model(inputs),x_copy,x_model) : PartsDone<X,xv,D,dv,CopiedPart,copied_owners,Context,Fact,batch,context,Pieces.Range{first,rows},
    Kernel.owned(first,rows,x_count,d_count,(Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,inputs),Buffer.pack(~Tile.Tile8,_)),(Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,gradients),Buffer.pack(~Tile.Tile8,copy_model(gradients))))>
  %Equal.sym(Buffer.Model<Tile.Tile8>,copy_model(gradients),d_copy,d_model) : PartsDone<X,xv,D,dv,CopiedPart,copied_owners,Context,Fact,batch,context,Pieces.Range{first,rows},
    Kernel.owned(first,rows,x_count,d_count,(Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,inputs),Buffer.pack(~Tile.Tile8,x_copy)),(Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,gradients),Buffer.pack(~Tile.Tile8,_)))>
  PartsDone{x,d,
    %Equal.sym(Move.Holding<Tile.Tile8>,Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,inputs),Held.pack(~Tile.Tile8,x),x_packed) : {Kernel.Owners{_,Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,gradients)} == holders(x,d) : Kernel.Owners}
    %Equal.sym(Move.Holding<Tile.Tile8>,Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,gradients),Held.pack(~Tile.Tile8,d),d_packed) : {Kernel.Owners{Held.pack(~Tile.Tile8,x),_} == holders(x,d) : Kernel.Owners}
    {==},
    HeldBoth{x_tiled,x_values,x_ordered,d_tiled,d_values,d_ordered},
    PiecesModel.Piece{CopiedPart{x_copy,x_count,d_copy,d_count},first,rows},
    {==},
    {==},
    fact}

# ---- The parts of the images ----

# Where the copy of rows images of cells each from image first lies, as
# numbers, below the cells of images images.
type CopyRoom<-word: U32,-cells: Nat,-first: U32,-rows: U32,-high: Nat,-capacity: U32> is Data:
  CopyRoom{base: {(word * first : U32) == U32.from_nat(Nat.mul(U32.to_nat(first),cells)) : U32},
    count: {U32.to_nat((word * rows : U32)) == Nat.mul(U32.to_nat(rows),cells) : Nat},
    fits: {Nat.is_le(Nat.add(U32.to_nat((word * first : U32)),U32.to_nat((word * rows : U32))),high) == True{} : Bool},
    allocatable: {U32.is_le((word * rows : U32),1073741824) == True{} : Bool},
    bounded: {Nat.is_le(U32.to_nat((word * rows : U32)),U32.to_nat(capacity)) == True{} : Bool}}

def scaled_word(+word: U32,+cells: Nat,+exact: {word == U32.from_nat(cells) : U32},+index: U32) ->
  {(word * index : U32) == U32.from_nat(Nat.mul(U32.to_nat(index),cells)) : U32}:
  %Equal.sym(U32,word,U32.from_nat(cells),exact) : {(_ * index : U32) == U32.from_nat(Nat.mul(U32.to_nat(index),cells)) : U32}
  Equal.trans(U32,(U32.from_nat(cells) * index : U32),U32.from_nat(Nat.mul(cells,U32.to_nat(index))),U32.from_nat(Nat.mul(U32.to_nat(index),cells)),
    Sources.product_word(cells,index),
    Equal.cong(Nat,U32,value => U32.from_nat(value),Nat.mul(cells,U32.to_nat(index)),Nat.mul(U32.to_nat(index),cells),RowMajor.multiply_commutative(cells,U32.to_nat(index))))

def scaled_nat(+word: U32,+cells: Nat,+exact: {word == U32.from_nat(cells) : U32},+index: U32,+capacity: U32,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(index),cells),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((word * index : U32)) == Nat.mul(U32.to_nat(index),cells) : Nat}:
  %Equal.sym(U32,(word * index : U32),U32.from_nat(Nat.mul(U32.to_nat(index),cells)),scaled_word(word,cells,exact,index)) :
    {U32.to_nat(_) == Nat.mul(U32.to_nat(index),cells) : Nat}
  FullRange.roundtrip(Nat.mul(U32.to_nat(index),cells),capacity,room)

def copy_room(+word: U32,+cells: Nat,+exact: {word == U32.from_nat(cells) : U32},+first: U32,+rows: U32,+images: Nat,+high: Nat,+capacity: U32,
  +allocatable: {U32.is_le(capacity,1073741824) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),images) == True{} : Bool},
  +total: {Nat.is_le(Nat.mul(images,cells),high) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool}) ->
  CopyRoom<word,cells,first,rows,high,capacity>:
  +start = U32.to_nat(first)
  +count = U32.to_nat(rows)
  +before = Nat.mul(start,cells)
  +taken = Nat.mul(count,cells)
  +span = {%RowMajor.distrib(start,count,cells) : {Nat.is_le(_,high) == True{} : Bool}
    Order.transitive(Nat.mul(Nat.add(start,count),cells),Nat.mul(images,cells),high,RowMajor.mul_le(Nat.add(start,count),images,cells,room),total) :
    {Nat.is_le(Nat.add(before,taken),high) == True{} : Bool}}
  +first_room = Order.transitive(before,high,U32.to_nat(capacity),Order.transitive(before,Nat.add(before,taken),high,Order.le_add(before,taken),span),bounded)
  +rows_room = Order.transitive(taken,high,U32.to_nat(capacity),Order.drop_left_le(before,taken,high,span),bounded)
  +first_nat = scaled_nat(word,cells,exact,first,capacity,first_room)
  +rows_nat = scaled_nat(word,cells,exact,rows,capacity,rows_room)
  CopyRoom{scaled_word(word,cells,exact,first),rows_nat,
    %Equal.sym(Nat,U32.to_nat((word * first : U32)),before,first_nat) : {Nat.is_le(Nat.add(_,U32.to_nat((word * rows : U32))),high) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat((word * rows : U32)),taken,rows_nat) : {Nat.is_le(Nat.add(before,_),high) == True{} : Bool}
    span,
    Arithmetic.count_within((word * rows : U32),taken,capacity,1073741824,rows_nat,rows_room,allocatable),
    %Equal.sym(Nat,U32.to_nat((word * rows : U32)),taken,rows_nat) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    rows_room}

# A part of the images [first, first + rows) of a batch: its copies cover
# their cells with the values of the batch from the first cell of image first
# on, and hold the cells of rows images.
type CopiedImages<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-batch: Images.Batch<X,xv,D,dv>,-work: Parameter.Work,-first: U32,-rows: U32,
  -input: Buffer.Model<Tile.Tile8>,-x_count: U32,-gradient: Buffer.Model<Tile.Tile8>,-d_count: U32> is Data:
  CopiedImages{
    x_covered: Held.Covered<Tile.Tile8,Held.Shifted<X>,Held.shifted_value(~Tile.Tile8,~X,~xv),Held.Shifted{Images.batch_input(~X,~xv,~D,~dv,batch),(Kernel.input_image(work) * first : U32)},
      Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,input)),0n,U32.to_nat(x_count),Held.layout_spans(Pieces.Range{0,x_count},Held.Ended{})>,
    d_covered: Held.Covered<Tile.Tile8,Held.Shifted<D>,Held.shifted_value(~Tile.Tile8,~D,~dv),Held.Shifted{Images.batch_gradient(~X,~xv,~D,~dv,batch),(Kernel.gradient_image(work) * first : U32)},
      Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,gradient)),0n,U32.to_nat(d_count),Held.layout_spans(Pieces.Range{0,d_count},Held.Ended{})>,
    x_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),Sources.input_cells(Images.batch_geometry(~X,~xv,~D,~dv,batch))),U32.to_nat(x_count)) == True{} : Bool},
    x_bounded: {Nat.is_le(U32.to_nat(x_count),U32.to_nat(Machine.capacity(Groups.domain_limits(Images.batch_geometry(~X,~xv,~D,~dv,batch),batch_domain(~X,~xv,~D,~dv,batch))))) == True{} : Bool},
    x_exact: {(Kernel.input_image(work) * first : U32) == U32.from_nat(Nat.mul(U32.to_nat(first),Sources.input_cells(Images.batch_geometry(~X,~xv,~D,~dv,batch)))) : U32},
    d_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),Groups.gradient_extent(Images.batch_geometry(~X,~xv,~D,~dv,batch))),U32.to_nat(d_count)) == True{} : Bool},
    d_bounded: {Nat.is_le(U32.to_nat(d_count),U32.to_nat(Machine.capacity(Groups.domain_limits(Images.batch_geometry(~X,~xv,~D,~dv,batch),batch_domain(~X,~xv,~D,~dv,batch))))) == True{} : Bool},
    d_exact: {(Kernel.gradient_image(work) * first : U32) == U32.from_nat(Nat.mul(U32.to_nat(first),Groups.gradient_extent(Images.batch_geometry(~X,~xv,~D,~dv,batch)))) : U32}}

def batch_fact(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+scope: KernelGemm.Both<Images.Batch<X,xv,D,dv>,Parameter.Work>,+part: CopiedPart,
  +first: U32,+rows: U32) -> Data:
  KernelGemm.Both{batch,work} = scope
  CopiedPart{input,x_count,gradient,d_count} = part
  CopiedImages<X,xv,D,dv,batch,work,first,rows,input,x_count,gradient,d_count>

# The words of the cells of an image of a work are those of its geometry, and
# the limit of the plan is the images of the batch.
type BatchReady<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-batch: Images.Batch<X,xv,D,dv>,-work: Parameter.Work,-limit: Nat> is Data:
  BatchReady{x_word: {Kernel.input_image(work) == U32.from_nat(Sources.input_cells(Images.batch_geometry(~X,~xv,~D,~dv,batch))) : U32},
    d_word: {Kernel.gradient_image(work) == U32.from_nat(Groups.gradient_extent(Images.batch_geometry(~X,~xv,~D,~dv,batch))) : U32},
    images: {limit == Images.batch_images(~X,~xv,~D,~dv,batch) : Nat}}

def batch_ready(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Images.Batch<X,xv,D,dv>,work: Parameter.Work,limit: Nat) -> Data:
  BatchReady<X,xv,D,dv,batch,work,limit>

def images_of(+work: Parameter.Work,+first: U32,+rows: U32,+x: Held.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Kernel.images_of(Kernel.input_image(work),first,rows,Held.pack(~Tile.Tile8,x))

def gradients_of(+work: Parameter.Work,+first: U32,+rows: U32,+d: Held.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Kernel.images_of(Kernel.gradient_image(work),first,rows,Held.pack(~Tile.Tile8,d))

# A leaf of a part of the images copies its images from covering holdings.
def batch_copies(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,+work: Parameter.Work,+first: U32,+rows: U32,
  +x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>,+holds: HeldBoth<X,xv,D,dv,batch,x,d>,
  +x_room: CopyRoom<Kernel.input_image(work),Sources.input_cells(Images.batch_geometry(~X,~xv,~D,~dv,batch)),first,rows,Images.batch_x_high(~X,~xv,~D,~dv,batch),
    Machine.capacity(Groups.domain_limits(Images.batch_geometry(~X,~xv,~D,~dv,batch),batch_domain(~X,~xv,~D,~dv,batch)))>,
  +d_room: CopyRoom<Kernel.gradient_image(work),Groups.gradient_extent(Images.batch_geometry(~X,~xv,~D,~dv,batch)),first,rows,Images.batch_d_high(~X,~xv,~D,~dv,batch),
    Machine.capacity(Groups.domain_limits(Images.batch_geometry(~X,~xv,~D,~dv,batch),batch_domain(~X,~xv,~D,~dv,batch)))>) ->
  PartsDone<X,xv,D,dv,CopiedPart,copied_owners,Parameter.Work,batch_fact(~X,~xv,~D,~dv),batch,work,Pieces.Range{first,rows},
    Kernel.owned(first,rows,(Kernel.input_image(work) * rows : U32),(Kernel.gradient_image(work) * rows : U32),images_of(work,first,rows,x),gradients_of(work,first,rows,d))>:
  Images.Batch{+input,+gradient,+geometry,+domain,+images,+height,+width,+group,+size,+bounded,+x_high,+x_layout,+x_whole,+x_bounded,+d_high,+d_layout,+d_whole,+d_bounded} = batch
  HeldBoth{+x_tiled,+x_values,+x_ordered,+d_tiled,+d_values,+d_ordered} = holds
  CopyRoom{+x_base,+x_count,+x_fits,+x_allocatable,+x_count_bounded} = x_room
  CopyRoom{+d_base,+d_count,+d_fits,+d_allocatable,+d_count_bounded} = d_room
  +limits = Groups.domain_limits(geometry,domain)
  +capacity = Machine.capacity(limits)
  +fill = Tile.filled(0.0)
  +x_start = (Kernel.input_image(work) * first : U32)
  +x_span = (Kernel.input_image(work) * rows : U32)
  +d_start = (Kernel.gradient_image(work) * first : U32)
  +d_span = (Kernel.gradient_image(work) * rows : U32)
  copies_done(~X,~xv,~D,~dv,~Parameter.Work,~batch_fact(~X,~xv,~D,~dv),
    Images.Batch{input,gradient,geometry,domain,images,height,width,group,size,bounded,x_high,x_layout,x_whole,x_bounded,d_high,d_layout,d_whole,d_bounded},work,first,rows,
    images_of(work,first,rows,x),x_span,gradients_of(work,first,rows,d),d_span,copy_split(images_of(work,first,rows,x)),copy_split(gradients_of(work,first,rows,d)),
    Held.copied_covered(~Tile.Tile8,~X,~xv,input,x,x_start,x_span,fill,0n,x_high,x_tiled,x_values,x_layout,x_ordered),
    Held.copied_covered(~Tile.Tile8,~D,~dv,gradient,d,d_start,d_span,fill,0n,d_high,d_tiled,d_values,d_layout,d_ordered),
    copy_model(images_of(work,first,rows,x)),{==},copy_model(gradients_of(work,first,rows,d)),{==},
    CopiedImages{
      Held.copied_alone(~Tile.Tile8,~X,~xv,input,x,x_start,x_span,fill,0n,x_high,capacity,Limits.capacity_fits(limits),x_bounded,x_tiled,x_values,
        Order.zero_le(U32.to_nat(x_start)),x_fits,x_allocatable),
      Held.copied_alone(~Tile.Tile8,~D,~dv,gradient,d,d_start,d_span,fill,0n,d_high,capacity,Limits.capacity_fits(limits),d_bounded,d_tiled,d_values,
        Order.zero_le(U32.to_nat(d_start)),d_fits,d_allocatable),
      %Equal.sym(Nat,U32.to_nat(x_span),Nat.mul(U32.to_nat(rows),Sources.input_cells(geometry)),x_count) : {Nat.is_le(Nat.mul(U32.to_nat(rows),Sources.input_cells(geometry)),_) == True{} : Bool}
      Order.reflexive(Nat.mul(U32.to_nat(rows),Sources.input_cells(geometry))),
      x_count_bounded,x_base,
      %Equal.sym(Nat,U32.to_nat(d_span),Nat.mul(U32.to_nat(rows),Groups.gradient_extent(geometry)),d_count) : {Nat.is_le(Nat.mul(U32.to_nat(rows),Groups.gradient_extent(geometry)),_) == True{} : Bool}
      Order.reflexive(Nat.mul(U32.to_nat(rows),Groups.gradient_extent(geometry))),
      d_count_bounded,d_base})

# A leaf of the build of the parts of the images.
def batch_leaf(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,+work: Parameter.Work,+limit: Nat,
  +ready: batch_ready(~X,~xv,~D,~dv,batch,work,limit),+first: U32,+rows: U32,+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),limit) == True{} : Bool},
  +x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>,+holds: HeldBoth<X,xv,D,dv,batch,x,d>) ->
  PartsDone<X,xv,D,dv,CopiedPart,copied_owners,Parameter.Work,batch_fact(~X,~xv,~D,~dv),batch,work,Pieces.Range{first,rows},Kernel.batch_own(work,first,rows,holders(x,d))>:
  Images.Batch{+input,+gradient,+geometry,+domain,+images,+height,+width,+group,+size,+bounded,+x_high,+x_layout,+x_whole,+x_bounded,+d_high,+d_layout,+d_whole,+d_bounded} = batch
  BatchReady{+x_word,+d_word,+limit_exact} = ready
  +limits = Groups.domain_limits(geometry,domain)
  +capacity = Machine.capacity(limits)
  +images_room = {%limit_exact : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool} room :
    {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),images) == True{} : Bool}}
  +shape = {Images.Batch{input,gradient,geometry,domain,images,height,width,group,size,bounded,x_high,x_layout,x_whole,x_bounded,d_high,d_layout,d_whole,d_bounded} : Images.Batch<X,xv,D,dv>}
  batch_copies(~X,~xv,~D,~dv,shape,work,first,rows,x,d,holds,
    copy_room(Kernel.input_image(work),Sources.input_cells(geometry),x_word,first,rows,images,x_high,capacity,Limits.capacity_allocatable(limits),images_room,x_whole,x_bounded),
    copy_room(Kernel.gradient_image(work),Groups.gradient_extent(geometry),d_word,first,rows,images,d_high,capacity,Limits.capacity_allocatable(limits),images_room,d_whole,d_bounded))


# ---- The lanes of the parts of the images ----

def arrays_appended(left: List<&2,Tree.Tree<Tile.Tile8>>,+right: List<&2,Tree.Tree<Tile.Tile8>>) ->
  {List.append(&1,Array<Tile.Tile8>,Images.pack_trees(left),Images.pack_trees(right)) == Images.pack_trees(OrderedMatrix.cat(Tree.Tree<Tile.Tile8>,left,right)) : List<&1,Array<Tile.Tile8>>}:
  match left:
    case Nil{}: {==}
    case +head <> +rest:
      Equal.cong(List<&1,Array<Tile.Tile8>>,List<&1,Array<Tile.Tile8>>,tail => Tree.pack(Tile.Tile8,head) <> tail,
        List.append(&1,Array<Tile.Tile8>,Images.pack_trees(rest),Images.pack_trees(right)),Images.pack_trees(OrderedMatrix.cat(Tree.Tree<Tile.Tile8>,rest,right)),
        arrays_appended(rest,right))

def observed_joined(+sum: Refinement.SumQuery,left: List<&2,Tree.Tree<Tile.Tile8>>,+right: List<&2,Tree.Tree<Tile.Tile8>>) ->
  {Images.observed_trees(sum,OrderedMatrix.cat(Tree.Tree<Tile.Tile8>,left,right)) == OrderedMatrix.cat(F32,Images.observed_trees(sum,left),Images.observed_trees(sum,right)) : List<&2,F32>}:
  match left:
    case Nil{}: {==}
    case +head <> +rest:
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => Refinement.sum_query_read(sum,head) <> tail,
        Images.observed_trees(sum,OrderedMatrix.cat(Tree.Tree<Tile.Tile8>,rest,right)),OrderedMatrix.cat(F32,Images.observed_trees(sum,rest),Images.observed_trees(sum,right)),
        observed_joined(sum,rest,right))

# The sums the parts of the images [first, first + count) give, in order: the
# arrays of their cell trees, with their frames and their lanes.
type PartsLanes<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-batch: Images.Batch<X,xv,D,dv>,-tap: U32,-output: U32,-sum: Refinement.SumQuery,
  -first: Nat,-count: Nat,-values: List<&1,Array<Tile.Tile8>>> is Data:
  PartsLanes{trees: List<&2,Tree.Tree<Tile.Tile8>>,
    values: {values == Images.pack_trees(trees) : List<&1,Array<Tile.Tile8>>},
    frames: Members.All(Tree.Tree<Tile.Tile8>,{tree => Refinement.sum_query_frame(sum,tree) : Tree.Tree<Tile.Tile8> -> Data},trees),
    observed: {Images.observed_trees(sum,trees) == Images.held_images(~X,~xv,~D,~dv,count,first,batch,tap,output) : List<&2,F32>}}

def lanes_joined(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,+tap: U32,+output: U32,+sum: Refinement.SumQuery,
  +first: Nat,+count: Nat,+left_rows: Nat,+right_rows: Nat,+rows_equal: {count == Nat.add(left_rows,right_rows) : Nat},
  -left_values: List<&1,Array<Tile.Tile8>>,-right_values: List<&1,Array<Tile.Tile8>>,
  left: PartsLanes<X,xv,D,dv,batch,tap,output,sum,first,left_rows,left_values>,right: PartsLanes<X,xv,D,dv,batch,tap,output,sum,Nat.add(first,left_rows),right_rows,right_values>) ->
  PartsLanes<X,xv,D,dv,batch,tap,output,sum,first,count,List.append(&1,Array<Tile.Tile8>,left_values,right_values)>:
  PartsLanes{+left_trees,+left_equal,+left_frames,+left_observed} = left
  PartsLanes{+right_trees,+right_equal,+right_frames,+right_observed} = right
  +held_left = Images.held_images(~X,~xv,~D,~dv,left_rows,first,batch,tap,output)
  +held_right = Images.held_images(~X,~xv,~D,~dv,right_rows,Nat.add(first,left_rows),batch,tap,output)
  +held_both = Images.held_images(~X,~xv,~D,~dv,Nat.add(left_rows,right_rows),first,batch,tap,output)
  PartsLanes{OrderedMatrix.cat(Tree.Tree<Tile.Tile8>,left_trees,right_trees),
    %Equal.sym(List<&1,Array<Tile.Tile8>>,left_values,Images.pack_trees(left_trees),left_equal) :
      {List.append(&1,Array<Tile.Tile8>,_,right_values) == Images.pack_trees(OrderedMatrix.cat(Tree.Tree<Tile.Tile8>,left_trees,right_trees)) : List<&1,Array<Tile.Tile8>>}
    %Equal.sym(List<&1,Array<Tile.Tile8>>,right_values,Images.pack_trees(right_trees),right_equal) :
      {List.append(&1,Array<Tile.Tile8>,Images.pack_trees(left_trees),_) == Images.pack_trees(OrderedMatrix.cat(Tree.Tree<Tile.Tile8>,left_trees,right_trees)) : List<&1,Array<Tile.Tile8>>}
    arrays_appended(left_trees,right_trees),
    Members.all_concatenated(Tree.Tree<Tile.Tile8>,{tree => Refinement.sum_query_frame(sum,tree) : Tree.Tree<Tile.Tile8> -> Data},left_trees,right_trees,left_frames,right_frames),
    %Equal.sym(Nat,count,Nat.add(left_rows,right_rows),rows_equal) :
      {Images.observed_trees(sum,OrderedMatrix.cat(Tree.Tree<Tile.Tile8>,left_trees,right_trees)) == Images.held_images(~X,~xv,~D,~dv,_,first,batch,tap,output) : List<&2,F32>}
    Equal.trans(List<&2,F32>,Images.observed_trees(sum,OrderedMatrix.cat(Tree.Tree<Tile.Tile8>,left_trees,right_trees)),
      OrderedMatrix.cat(F32,Images.observed_trees(sum,left_trees),Images.observed_trees(sum,right_trees)),held_both,
      observed_joined(sum,left_trees,right_trees),
      %Equal.sym(List<&2,F32>,Images.observed_trees(sum,left_trees),held_left,left_observed) :
        {OrderedMatrix.cat(F32,_,Images.observed_trees(sum,right_trees)) == held_both : List<&2,F32>}
      %Equal.sym(List<&2,F32>,Images.observed_trees(sum,right_trees),held_right,right_observed) :
        {OrderedMatrix.cat(F32,held_left,_) == held_both : List<&2,F32>}
      Equal.sym(List<&2,F32>,held_both,OrderedMatrix.cat(F32,held_left,held_right),images_joined(~X,~xv,~D,~dv,left_rows,right_rows,first,batch,tap,output)))}

# The sums of the images of one part, from what the serial kernel gives back
# over its copies.
def piece_lanes(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,
  +copy: Images.Batch<Held.Shifted<X>,Held.shifted_value(~Tile.Tile8,~X,~xv),Held.Shifted<D>,Held.shifted_value(~Tile.Tile8,~D,~dv)>,+tap: U32,+output: U32,+sum: Refinement.SumQuery,
  +first: Nat,+count: Nat,-images: Kernel.Held() & List<&1,Array<Tile.Tile8>>,
  done: Images.ImagesDone<Held.Shifted<X>,Held.shifted_value(~Tile.Tile8,~X,~xv),Held.Shifted<D>,Held.shifted_value(~Tile.Tile8,~D,~dv),copy,tap,output,sum,[],count,0n,images>,
  +moved: {Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),count,0n,copy,tap,output) ==
    Images.held_images(~X,~xv,~D,~dv,count,first,batch,tap,output) : List<&2,F32>}) ->
  PartsLanes<X,xv,D,dv,batch,tap,output,sum,first,count,Kernel.batch_values(images)>:
  Images.ImagesDone{+trees,+values,x_covered,d_covered,+frames,+observed} = done
  PartsLanes{trees,
    Equal.cong(Kernel.Held() & List<&1,Array<Tile.Tile8>>,List<&1,Array<Tile.Tile8>>,result => Kernel.batch_values(result),images,
      (Pair.fst(Kernel.Held(),List<&1,Array<Tile.Tile8>>,images),Images.pack_trees(trees)),values),
    frames,
    Equal.trans(List<&2,F32>,Images.observed_trees(sum,trees),
      Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),count,0n,copy,tap,output),
      Images.held_images(~X,~xv,~D,~dv,count,first,batch,tap,output),observed,moved)}

def batch_context_of(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>) -> Kernel.BatchContext:
  Images.Batch{input,gradient,+geometry,domain,images,+rows,+columns,+group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  Kernel.BatchContext{geometry,rows,columns,group}

def part_arrays(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,parts: PiecesModel.Shaped<CopiedPart>) -> List<&1,Array<Tile.Tile8>>:
  Kernel.batch_arrays(Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,
    batch_context_of(~X,~xv,~D,~dv,batch),parts))

# One part of the images runs the serial kernel over its copies.
def piece_part(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,+work: Parameter.Work,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +part: CopiedPart,+first: U32,+rows: U32,+fact: batch_fact(~X,~xv,~D,~dv,KernelGemm.Both{batch,work},part,first,rows)) ->
  PartsLanes<X,xv,D,dv,batch,tap,output,Images.batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside),U32.to_nat(first),U32.to_nat(rows),
    part_arrays(~X,~xv,~D,~dv,batch,PiecesModel.Piece{part,first,rows})>:
  Images.Batch{+input,+gradient,+geometry,+domain,+images,+height,+width,+group,+size,+bounded,+x_high,+x_layout,+x_whole,+x_bounded,+d_high,+d_layout,+d_whole,+d_bounded} = batch
  CopiedPart{+x_model,+x_count,+d_model,+d_count} = part
  CopiedImages{+x_covered,+d_covered,+x_room,+x_part_bounded,+x_exact,+d_room,+d_part_bounded,+d_exact} = fact
  +shape = {Images.Batch{input,gradient,geometry,domain,images,height,width,group,size,bounded,x_high,x_layout,x_whole,x_bounded,d_high,d_layout,d_whole,d_bounded} : Images.Batch<X,xv,D,dv>}
  +x_base = (Kernel.input_image(work) * first : U32)
  +d_base = (Kernel.gradient_image(work) * first : U32)
  +count = U32.to_nat(rows)
  +x_spans = Held.layout_spans(Pieces.Range{0,x_count},Held.Ended{})
  +d_spans = Held.layout_spans(Pieces.Range{0,d_count},Held.Ended{})
  +copy = copy_batch(~X,~xv,~D,~dv,shape,x_base,d_base,count,U32.to_nat(x_count),x_spans,x_room,x_part_bounded,U32.to_nat(d_count),d_spans,d_room,d_part_bounded)
  piece_lanes(~X,~xv,~D,~dv,shape,copy,tap,output,Images.lane_sum(geometry,domain,tap,output,tap_inside,output_inside),U32.to_nat(first),count,
    Kernel.images(count,0,geometry,height,width,group,Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))),
    Images.images_done(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Held.Shifted<D>,~Held.shifted_value(~Tile.Tile8,~D,~dv),copy,tap,output,tap_inside,output_inside,
      Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model)),x_covered,d_covered),
    copy_images(~X,~xv,~D,~dv,count,0n,U32.to_nat(first),shape,x_base,d_base,count,U32.to_nat(x_count),x_spans,x_room,x_part_bounded,U32.to_nat(d_count),d_spans,d_room,d_part_bounded,
      x_exact,d_exact,tap,output))

# The parts of the images [start, start + count) run the serial kernel over
# their copies; their sums, joined in order, are those of the images.
def parts_lanes(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,parts: PiecesModel.Shaped<CopiedPart>,+batch: Images.Batch<X,xv,D,dv>,+work: Parameter.Work,
  +tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +start: Nat,+count: Nat,+coverage: PiecesPlan.Coverage(PiecesModel.shaped_plan(~CopiedPart,~Each.piece_rows(~CopiedPart),parts),start,count),
  +facts: PiecesModel.ShapedAll(~CopiedPart,~KernelGemm.Both<Images.Batch<X,xv,D,dv>,Parameter.Work>,~batch_fact(~X,~xv,~D,~dv),KernelGemm.Both{batch,work},parts)) ->
  PartsLanes<X,xv,D,dv,batch,tap,output,Images.batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside),start,count,part_arrays(~X,~xv,~D,~dv,batch,parts)>:
  match parts:
    case PiecesModel.Piece{+part,+first,+rows}:
      PiecesPlan.LeafCoverage{+first_equal,+rows_equal} = coverage
      %first_equal :
        PartsLanes<X,xv,D,dv,batch,tap,output,Images.batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside),_,count,
          part_arrays(~X,~xv,~D,~dv,batch,PiecesModel.Piece{part,first,rows})>
      %rows_equal :
        PartsLanes<X,xv,D,dv,batch,tap,output,Images.batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside),U32.to_nat(first),_,
          part_arrays(~X,~xv,~D,~dv,batch,PiecesModel.Piece{part,first,rows})>
      piece_part(~X,~xv,~D,~dv,batch,work,tap,output,tap_inside,output_inside,part,first,rows,facts)
    case PiecesModel.Pieces{+left,+right}:
      PiecesPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
      KernelGemm.Both{+left_facts,+right_facts} = facts
      +context = batch_context_of(~X,~xv,~D,~dv,batch)
      +sum = Images.batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside)
      %Equal.sym(Partition.Tree<&1,Unit,Unit> & List<&1,Array<Tile.Tile8>>,
          Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,left)),
          (Pair.fst(Partition.Tree<&1,Unit,Unit>,List<&1,Array<Tile.Tile8>>,
              Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,left))),
            Pair.snd(Partition.Tree<&1,Unit,Unit>,List<&1,Array<Tile.Tile8>>,
              Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,left)))),
          Held.pair_eta(~Partition.Tree<&1,Unit,Unit>,~List<&1,Array<Tile.Tile8>>,
            Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,left)))) :
        PartsLanes<X,xv,D,dv,batch,tap,output,sum,start,count,
          Pair.snd(Partition.Tree<&1,Unit,Unit>,List<&1,Array<Tile.Tile8>>,
            Pieces.reduced(~Unit,~List<&1,Array<Tile.Tile8>>,~List.append(&1,Array<Tile.Tile8>),_,
              Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,right))))>
      %Equal.sym(Partition.Tree<&1,Unit,Unit> & List<&1,Array<Tile.Tile8>>,
          Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,right)),
          (Pair.fst(Partition.Tree<&1,Unit,Unit>,List<&1,Array<Tile.Tile8>>,
              Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,right))),
            Pair.snd(Partition.Tree<&1,Unit,Unit>,List<&1,Array<Tile.Tile8>>,
              Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,right)))),
          Held.pair_eta(~Partition.Tree<&1,Unit,Unit>,~List<&1,Array<Tile.Tile8>>,
            Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,right)))) :
        PartsLanes<X,xv,D,dv,batch,tap,output,sum,start,count,
          Pair.snd(Partition.Tree<&1,Unit,Unit>,List<&1,Array<Tile.Tile8>>,
            Pieces.reduced(~Unit,~List<&1,Array<Tile.Tile8>>,~List.append(&1,Array<Tile.Tile8>),
              (Pair.fst(Partition.Tree<&1,Unit,Unit>,List<&1,Array<Tile.Tile8>>,
                  Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,left))),
                Pair.snd(Partition.Tree<&1,Unit,Unit>,List<&1,Array<Tile.Tile8>>,
                  Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,left)))),_))>
      lanes_joined(~X,~xv,~D,~dv,batch,tap,output,sum,start,count,left_rows,right_rows,rows_equal,
        part_arrays(~X,~xv,~D,~dv,batch,left),part_arrays(~X,~xv,~D,~dv,batch,right),
        parts_lanes(~X,~xv,~D,~dv,left,batch,work,tap,output,tap_inside,output_inside,start,left_rows,left_coverage,left_facts),
        parts_lanes(~X,~xv,~D,~dv,right,batch,work,tap,output,tap_inside,output_inside,Nat.add(start,left_rows),right_rows,right_coverage,right_facts))


# ---- The cells of the weight kernel ----

# The cell of a lane: the output block of the output at the tap.
def cell_index(+geometry: Kernel.Geometry,+tap: U32,+output: U32) -> Nat:
  Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),Groups.tap_count(geometry)),U32.to_nat(tap))

# What the weight kernel gives back for a batch on any route: the holdings,
# which still cover the cells with their values, and the cells written in
# order into a fresh result, a lane of which is the adjacent tree of the lanes
# of the images.
type WeightCells<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-batch: Images.Batch<X,xv,D,dv>,-tap: U32,-output: U32,
  -result: Kernel.Held() & Storage.Buffer<Tile.Tile8>> is Data:
  WeightCells{list: List<&2,Tile.Tile8>,
    contents: {Buffer.contents(~Tile.Tile8,Pair.snd(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)) ==
      Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(Images.batch_cells(~X,~xv,~D,~dv,batch)),Tile.filled(0.0))),
        U32.to_nat(Images.batch_cells(~X,~xv,~D,~dv,batch))} : Buffer.Content<Tile.Tile8>},
    length: {List.length(&2,Tile.Tile8,list) ==
      Nat.mul(Groups.output_blocks(Images.batch_geometry(~X,~xv,~D,~dv,batch)),Groups.tap_count(Images.batch_geometry(~X,~xv,~D,~dv,batch))) : Nat},
    x_covered: Held.Covered<Tile.Tile8,X,xv,Images.batch_input(~X,~xv,~D,~dv,batch),
      Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)),
      0n,Images.batch_x_high(~X,~xv,~D,~dv,batch),Images.batch_x_layout(~X,~xv,~D,~dv,batch)>,
    d_covered: Held.Covered<Tile.Tile8,D,dv,Images.batch_gradient(~X,~xv,~D,~dv,batch),
      Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)),
      0n,Images.batch_d_high(~X,~xv,~D,~dv,batch),Images.batch_d_layout(~X,~xv,~D,~dv,batch)>,
    lane: {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),
        ListRead.at(~Tile.Tile8,Tile.filled(0.0),cell_index(Images.batch_geometry(~X,~xv,~D,~dv,batch),tap,output),list)) ==
      Images.lane_levels(Images.batch_images(~X,~xv,~D,~dv,batch),Images.held_images(~X,~xv,~D,~dv,Images.batch_images(~X,~xv,~D,~dv,batch),0n,batch,tap,output)) : F32}}

# The product of two words that fits a capacity is the product of their
# numbers.
def word_product(+left: U32,+right: U32,+capacity: U32,+room: {Nat.is_le(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((left * right : U32)) == Nat.mul(U32.to_nat(left),U32.to_nat(right)) : Nat}:
  %Arithmetic.native_roundtrip(left) : {U32.to_nat((_ * right : U32)) == Nat.mul(U32.to_nat(left),U32.to_nat(right)) : Nat}
  %Equal.sym(U32,(U32.from_nat(U32.to_nat(left)) * right : U32),U32.from_nat(Nat.mul(U32.to_nat(left),U32.to_nat(right))),Sources.product_word(U32.to_nat(left),right)) :
    {U32.to_nat(_) == Nat.mul(U32.to_nat(left),U32.to_nat(right)) : Nat}
  FullRange.roundtrip(Nat.mul(U32.to_nat(left),U32.to_nat(right)),capacity,room)

# The cells of a result are the output blocks at every tap.
def cells_count(+geometry: Kernel.Geometry,+domain: Groups.GeometryReady<geometry>) ->
  {U32.to_nat(Kernel.cells(geometry)) == Nat.mul(Groups.output_blocks(geometry),Groups.tap_count(geometry)) : Nat}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  Groups.GeometryReady{+limits,columns_ok,kernel,stride,square,+taps_ok,outputs_ok,+count,aligned,taps_exact,pixels,gradients} = domain
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +t = U32.to_nat(taps)
  +n = U32.to_nat(o)
  +blocks = Ceiling.ceiling(n,8n)
  +outputs_room = Order.transitive(n,Nat.mul(t,n),cap,%RowMajor.multiply_commutative(n,t) : {Nat.is_le(n,_) == True{} : Bool} RowMajor.product_covers(n,t,taps_ok),count)
  +blocks_exact = Encoded.eight_blocks_exact(limits,o,outputs_room)
  +room = {%Equal.sym(Nat,U32.to_nat(((o + 7) / 8 : U32)),blocks,blocks_exact) : {Nat.is_le(Nat.mul(_,t),cap) == True{} : Bool}
    Order.transitive(Nat.mul(blocks,t),Nat.mul(n,t),cap,RowMajor.mul_le(blocks,n,t,Ceiling.bounded(n,8n,{==})),
      %RowMajor.multiply_commutative(t,n) : {Nat.is_le(_,cap) == True{} : Bool} count) :
    {Nat.is_le(Nat.mul(U32.to_nat(((o + 7) / 8 : U32)),t),cap) == True{} : Bool}}
  %blocks_exact : {U32.to_nat(((o + 7) / 8 * taps : U32)) == Nat.mul(_,t) : Nat}
  word_product(((o + 7) / 8 : U32),taps,capacity,room)

# A lane of the cells of a batch lies in the cells.
def cell_inside(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool}) ->
  {Nat.is_lt(cell_index(Images.batch_geometry(~X,~xv,~D,~dv,batch),tap,output),
    Nat.mul(Groups.output_blocks(Images.batch_geometry(~X,~xv,~D,~dv,batch)),Groups.tap_count(Images.batch_geometry(~X,~xv,~D,~dv,batch)))) == True{} : Bool}:
  +geometry = Images.batch_geometry(~X,~xv,~D,~dv,batch)
  +total = Groups.output_count(geometry)
  +whole_output = U32.to_nat(output)
  +output_lt = {%Comparison.lt(output,Groups.geometry_outputs(geometry)) : {_ == True{} : Bool} output_inside : {Nat.is_lt(whole_output,total) == True{} : Bool}}
  RowMajor.radix(Nat.div(whole_output,8n),Groups.output_blocks(geometry),U32.to_nat(tap),Groups.tap_count(geometry),
    DivisionOrder.quotient_below(whole_output,Groups.output_blocks(geometry),8n,{==},
      Bounds.nat_trans(whole_output,total,Nat.mul(Groups.output_blocks(geometry),8n),output_lt,Ceiling.covers(total,8n,{==}))),
    %Comparison.lt(tap,Packing.geometry_taps(geometry)) : {_ == True{} : Bool} tap_inside)

# The cells of the sums of the images, as one list.
def weight_cells(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  -result: Kernel.Held() & Storage.Buffer<Tile.Tile8>,done: Images.WeightDone<X,xv,D,dv,batch,tap,output,result>) ->
  WeightCells<X,xv,D,dv,batch,tap,output,result>:
  Images.WeightDone{+sums,+contents,+x_covered,+d_covered,+lane} = done
  +geometry = Images.batch_geometry(~X,~xv,~D,~dv,batch)
  +count = U32.to_nat(Images.batch_cells(~X,~xv,~D,~dv,batch))
  +whole = Nat.mul(Groups.output_blocks(geometry),Groups.tap_count(geometry))
  +counted = cells_count(geometry,batch_domain(~X,~xv,~D,~dv,batch))
  +index = cell_index(geometry,tap,output)
  +inside = {%Equal.sym(Nat,count,whole,counted) : {Nat.is_lt(index,_) == True{} : Bool} cell_inside(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside) :
    {Nat.is_lt(index,count) == True{} : Bool}}
  WeightCells{Refinement.tiles(count,0n,sums),contents,
    Equal.trans(Nat,List.length(&2,Tile.Tile8,Refinement.tiles(count,0n,sums)),count,whole,
      ListRead.enumerate_length(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~Refinement.tile_at,count,0n,sums),counted),
    x_covered,d_covered,
    %Equal.sym(Tile.Tile8,ListRead.at(~Tile.Tile8,Tile.filled(0.0),index,Refinement.tiles(count,0n,sums)),Refinement.tile_at(sums,index),
        ListRead.enumerate_at(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~Refinement.tile_at,Tile.filled(0.0),count,index,0n,sums,inside)) :
      {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),_) ==
        Images.lane_levels(Images.batch_images(~X,~xv,~D,~dv,batch),Images.held_images(~X,~xv,~D,~dv,Images.batch_images(~X,~xv,~D,~dv,batch),0n,batch,tap,output)) : F32}
    lane}

# The cells of a batch from the holdings of the parts and the list they wrote.
def cells_done(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,+tap: U32,+output: U32,
  +x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>,+holds: HeldBoth<X,xv,D,dv,batch,x,d>,-buffer: Storage.Buffer<Tile.Tile8>,+list: List<&2,Tile.Tile8>,
  +contents: {Buffer.contents(~Tile.Tile8,buffer) ==
    Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(Images.batch_cells(~X,~xv,~D,~dv,batch)),Tile.filled(0.0))),
      U32.to_nat(Images.batch_cells(~X,~xv,~D,~dv,batch))} : Buffer.Content<Tile.Tile8>},
  +length: {List.length(&2,Tile.Tile8,list) ==
    Nat.mul(Groups.output_blocks(Images.batch_geometry(~X,~xv,~D,~dv,batch)),Groups.tap_count(Images.batch_geometry(~X,~xv,~D,~dv,batch))) : Nat},
  +lane: {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),
      ListRead.at(~Tile.Tile8,Tile.filled(0.0),cell_index(Images.batch_geometry(~X,~xv,~D,~dv,batch),tap,output),list)) ==
    Images.lane_levels(Images.batch_images(~X,~xv,~D,~dv,batch),Images.held_images(~X,~xv,~D,~dv,Images.batch_images(~X,~xv,~D,~dv,batch),0n,batch,tap,output)) : F32}) ->
  WeightCells<X,xv,D,dv,batch,tap,output,((Held.pack(~Tile.Tile8,x),Held.pack(~Tile.Tile8,d)),buffer)>:
  HeldBoth{+x_tiled,+x_values,+x_ordered,+d_tiled,+d_values,+d_ordered} = holds
  WeightCells{list,contents,length,Held.Covered{x,{==},x_tiled,x_values,x_ordered},Held.Covered{d,{==},d_tiled,d_values,d_ordered},lane}

# ---- The weight kernel in parts of the images ----

def work_images(work: Parameter.Work) -> Nat:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  n

# A batch of the images of a work: its geometry, images and groups, the words
# of the cells of an image, and its images below the capacity.
type WorkBatch<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-batch: Images.Batch<X,xv,D,dv>,-work: Parameter.Work> is Data:
  WorkBatch{context: {Kernel.batch_context(work) == batch_context_of(~X,~xv,~D,~dv,batch) : Kernel.BatchContext},
    images: {work_images(work) == Images.batch_images(~X,~xv,~D,~dv,batch) : Nat},
    cells: {Kernel.cells(Kernel.geometry(work)) == Images.batch_cells(~X,~xv,~D,~dv,batch) : U32},
    x_word: {Kernel.input_image(work) == U32.from_nat(Sources.input_cells(Images.batch_geometry(~X,~xv,~D,~dv,batch))) : U32},
    d_word: {Kernel.gradient_image(work) == U32.from_nat(Groups.gradient_extent(Images.batch_geometry(~X,~xv,~D,~dv,batch))) : U32},
    images_room: {Nat.is_le(Images.batch_images(~X,~xv,~D,~dv,batch),
      U32.to_nat(Machine.capacity(Groups.domain_limits(Images.batch_geometry(~X,~xv,~D,~dv,batch),batch_domain(~X,~xv,~D,~dv,batch))))) == True{} : Bool},
    positions: {U32.to_nat(work_positions(work)) == Nat.mul(Images.batch_images(~X,~xv,~D,~dv,batch),
      Nat.mul(Groups.row_count(Images.batch_geometry(~X,~xv,~D,~dv,batch)),Groups.column_count(Images.batch_geometry(~X,~xv,~D,~dv,batch)))) : Nat},
    groups: {U32.to_nat(work_groups(work)) == Groups.output_blocks(Images.batch_geometry(~X,~xv,~D,~dv,batch)) : Nat},
    input_cells: {U32.to_nat(work_cells(work)) == Nat.mul(Images.batch_images(~X,~xv,~D,~dv,batch),Sources.input_cells(Images.batch_geometry(~X,~xv,~D,~dv,batch))) : Nat},
    blocks: {U32.to_nat(Kernel.blocks(work)) == Groups.output_blocks(Images.batch_geometry(~X,~xv,~D,~dv,batch)) : Nat}}

def parts_result(+count: U32,+images: Nat,owners: Kernel.Owners,arrays: List<&1,Array<Tile.Tile8>>) -> Kernel.Held() & Storage.Buffer<Tile.Tile8>:
  Kernel.written(count,Kernel.reduced(images,count,(Kernel.owners_of(owners),arrays)))

def batch_each(+plan: Pieces.Ranges,+context: Kernel.BatchContext,tree: Partition.Tree<&1,Kernel.Owners,Unit>) -> List<&1,Array<Tile.Tile8>>:
  Kernel.batch_arrays(Pieces.each(~Kernel.Owners,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.BatchContext,~Pieces.unshared,~Kernel.batch_part,~Kernel.batch_unmatched,
    plan,True{},context,tree,Unit{}))

# The sums of the parts, reduced and written into the cells of the result.
def batch_finished(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>,+holds: HeldBoth<X,xv,D,dv,batch,x,d>,-arrays: List<&1,Array<Tile.Tile8>>,
  lanes: PartsLanes<X,xv,D,dv,batch,tap,output,Images.batch_sum(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside),0n,Images.batch_images(~X,~xv,~D,~dv,batch),arrays>) ->
  Images.WeightDone<X,xv,D,dv,batch,tap,output,parts_result(Images.batch_cells(~X,~xv,~D,~dv,batch),Images.batch_images(~X,~xv,~D,~dv,batch),holders(x,d),arrays)>:
  HeldBoth{+x_tiled,+x_values,+x_ordered,+d_tiled,+d_values,+d_ordered} = holds
  PartsLanes{+trees,+values,+frames,+observed} = lanes
  Images.weight_finished(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside,((Held.pack(~Tile.Tile8,x),Held.pack(~Tile.Tile8,d)),arrays),
    Images.ImagesDone{trees,
      Equal.cong(List<&1,Array<Tile.Tile8>>,Kernel.Held() & List<&1,Array<Tile.Tile8>>,list => ((Held.pack(~Tile.Tile8,x),Held.pack(~Tile.Tile8,d)),list),arrays,Images.pack_trees(trees),values),
      Held.Covered{x,{==},x_tiled,x_values,x_ordered},Held.Covered{d,{==},d_tiled,d_values,d_ordered},frames,observed})

# What the parts of a plan of the images give back is the weight kernel over
# the batch.
def batch_assembled(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+depth: Nat,+work: Parameter.Work,+batch: Images.Batch<X,xv,D,dv>,
  +same: WorkBatch<X,xv,D,dv,batch,work>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +coverage: PiecesPlan.Coverage(Kernel.batch_plan(depth,work),0n,work_images(work)),-result: Owned(),
  done: PartsDone<X,xv,D,dv,CopiedPart,copied_owners,Parameter.Work,batch_fact(~X,~xv,~D,~dv),batch,work,Kernel.batch_plan(depth,work),result>) ->
  Images.WeightDone<X,xv,D,dv,batch,tap,output,
    parts_result(Kernel.cells(Kernel.geometry(work)),work_images(work),Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,result),
      batch_each(Kernel.batch_plan(depth,work),Kernel.batch_context(work),Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,result)))>:
  WorkBatch{+context_same,+images_same,+cells_same,+x_word,+d_word,+images_room,+positions_exact,+groups_exact,+input_exact,+blocks_exact} = same
  PartsDone{+x,+d,+state,+holds,+parts,+packed,+planned,+facts} = done
  +plan = Kernel.batch_plan(depth,work)
  +count = Kernel.cells(Kernel.geometry(work))
  +images = work_images(work)
  +context = Kernel.batch_context(work)
  +own_plan = PiecesModel.shaped_plan(~CopiedPart,~Each.piece_rows(~CopiedPart),parts)
  +whole = Images.batch_images(~X,~xv,~D,~dv,batch)
  +cells = Images.batch_cells(~X,~xv,~D,~dv,batch)
  +covered = {%Equal.sym(Pieces.Ranges,own_plan,plan,planned) : PiecesPlan.Coverage(_,0n,whole)
    %images_same : PiecesPlan.Coverage(plan,0n,_)
    coverage : PiecesPlan.Coverage(own_plan,0n,whole)}
  %Equal.sym(Kernel.Owners,Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,result),holders(x,d),state) :
    Images.WeightDone<X,xv,D,dv,batch,tap,output,parts_result(count,images,_,batch_each(plan,context,Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,result)))>
  %packed : Images.WeightDone<X,xv,D,dv,batch,tap,output,parts_result(count,images,holders(x,d),batch_each(plan,context,_))>
  %planned : Images.WeightDone<X,xv,D,dv,batch,tap,output,
    parts_result(count,images,holders(x,d),batch_each(_,context,PiecesModel.shaped_pack(~CopiedPart,~Kernel.Owners,~copied_owners,parts)))>
  %Equal.sym(Partition.Tree<&1,List<&1,Array<Tile.Tile8>>,Unit>,
      Pieces.each(~Kernel.Owners,~List<&1,Array<Tile.Tile8>>,~Unit,~Kernel.BatchContext,~Pieces.unshared,~Kernel.batch_part,~Kernel.batch_unmatched,
        own_plan,True{},context,PiecesModel.shaped_pack(~CopiedPart,~Kernel.Owners,~copied_owners,parts),Unit{}),
      Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,context,parts),
      Each.shaped_planned(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,~Kernel.batch_unmatched,parts,True{},context)) :
    Images.WeightDone<X,xv,D,dv,batch,tap,output,parts_result(count,images,holders(x,d),Kernel.batch_arrays(_))>
  %Equal.sym(Kernel.BatchContext,context,batch_context_of(~X,~xv,~D,~dv,batch),context_same) :
    Images.WeightDone<X,xv,D,dv,batch,tap,output,parts_result(count,images,holders(x,d),
      Kernel.batch_arrays(Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~List<&1,Array<Tile.Tile8>>,~Kernel.BatchContext,~Kernel.batch_part,_,parts)))>
  %Equal.sym(Nat,images,whole,images_same) :
    Images.WeightDone<X,xv,D,dv,batch,tap,output,parts_result(count,_,holders(x,d),part_arrays(~X,~xv,~D,~dv,batch,parts))>
  %Equal.sym(U32,count,cells,cells_same) :
    Images.WeightDone<X,xv,D,dv,batch,tap,output,parts_result(_,whole,holders(x,d),part_arrays(~X,~xv,~D,~dv,batch,parts))>
  batch_finished(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside,x,d,holds,part_arrays(~X,~xv,~D,~dv,batch,parts),
    parts_lanes(~X,~xv,~D,~dv,parts,batch,work,tap,output,tap_inside,output_inside,0n,whole,covered,facts))

# The weight kernel in parts of the images: a lane of a cell of the result is
# the adjacent tree of the lanes of the images, as over the whole batch.
def batch_route(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+depth: Nat,+work: Parameter.Work,+batch: Images.Batch<X,xv,D,dv>,
  +same: WorkBatch<X,xv,D,dv,batch,work>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  -x_held: Move.Holding<Tile.Tile8>,-d_held: Move.Holding<Tile.Tile8>,
  x_covered: Held.Covered<Tile.Tile8,X,xv,Images.batch_input(~X,~xv,~D,~dv,batch),x_held,0n,Images.batch_x_high(~X,~xv,~D,~dv,batch),Images.batch_x_layout(~X,~xv,~D,~dv,batch)>,
  d_covered: Held.Covered<Tile.Tile8,D,dv,Images.batch_gradient(~X,~xv,~D,~dv,batch),d_held,0n,Images.batch_d_high(~X,~xv,~D,~dv,batch),Images.batch_d_layout(~X,~xv,~D,~dv,batch)>) ->
  WeightCells<X,xv,D,dv,batch,tap,output,Kernel.batch_parts(depth,work,x_held,d_held)>:
  Parameter.Work{+n,+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+g} = work
  WorkBatch{+context_same,+images_same,+cells_same,+x_word,+d_word,+images_room,+positions_exact,+groups_exact,+input_exact,+blocks_exact} = same
  Held.Covered{+x,+x_packed,+x_tiled,+x_values,+x_ordered} = x_covered
  Held.Covered{+d,+d_packed,+d_tiled,+d_values,+d_ordered} = d_covered
  +shape = {Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} : Parameter.Work}
  +limits = Groups.domain_limits(Images.batch_geometry(~X,~xv,~D,~dv,batch),batch_domain(~X,~xv,~D,~dv,batch))
  +capacity = Machine.capacity(limits)
  +plan = Kernel.batch_plan(depth,shape)
  +n_room = {%Equal.sym(Nat,n,Images.batch_images(~X,~xv,~D,~dv,batch),images_same) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    images_room : {Nat.is_le(n,U32.to_nat(capacity)) == True{} : Bool}}
  +roundtrip = FullRange.roundtrip(n,capacity,n_room)
  +coverage = {%roundtrip : PiecesPlan.Coverage(plan,0n,_)
    PiecesPlan.refinement_unit_ranges(depth,1,0,U32.from_nat(n),capacity,Limits.capacity_fits(limits),8,{==},{==},{==},{==},
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(n)),n,roundtrip) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} n_room) :
    PiecesPlan.Coverage(plan,0n,n)}
  +rebuilt = {WorkBatch{context_same,images_same,cells_same,x_word,d_word,images_room,positions_exact,groups_exact,input_exact,blocks_exact} :
    WorkBatch<X,xv,D,dv,batch,shape>}
  %Equal.sym(Move.Holding<Tile.Tile8>,x_held,Held.pack(~Tile.Tile8,x),x_packed) :
    WeightCells<X,xv,D,dv,batch,tap,output,Kernel.batch_parts(depth,shape,_,d_held)>
  %Equal.sym(Move.Holding<Tile.Tile8>,d_held,Held.pack(~Tile.Tile8,d),d_packed) :
    WeightCells<X,xv,D,dv,batch,tap,output,Kernel.batch_parts(depth,shape,Held.pack(~Tile.Tile8,x),_)>
  %Equal.sym(Owned(),Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.batch_own,plan,shape,holders(x,d)),
      (Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.batch_own,plan,shape,holders(x,d))),
        Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.batch_own,plan,shape,holders(x,d)))),
      Held.pair_eta(~Kernel.Owners,~Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.batch_own,plan,shape,holders(x,d)))) :
    WeightCells<X,xv,D,dv,batch,tap,output,Kernel.batch_run(depth,shape,_)>
  weight_cells(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside,
    Kernel.batch_run(depth,shape,(Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.batch_own,plan,shape,holders(x,d))),
      Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.batch_own,plan,shape,holders(x,d))))),
  batch_assembled(~X,~xv,~D,~dv,depth,shape,batch,rebuilt,tap,output,tap_inside,output_inside,coverage,
    Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.batch_own,plan,shape,holders(x,d)),
    parts_made(~X,~xv,~D,~dv,~CopiedPart,~copied_owners,~Parameter.Work,~batch_fact(~X,~xv,~D,~dv),~batch_ready(~X,~xv,~D,~dv),~Kernel.batch_own,~batch_leaf(~X,~xv,~D,~dv),
      plan,batch,shape,n,{BatchReady{x_word,d_word,images_same} : BatchReady<X,xv,D,dv,batch,shape,n>},0n,n,coverage,Order.reflexive(n),x,d,
      HeldBoth{x_tiled,x_values,x_ordered,d_tiled,d_values,d_ordered})))

# ---- The parts of the outputs ----

# The cells of blocks [first, first + count) at every position of a gradient
# of groups cells a position, as the axes of the part that copies them.
def blocks_axes(+positions: U32,+groups: U32,+count: U32) -> List<&2,Move.Axis>:
  [Move.Axis{positions,groups,False{},count},Move.Axis{count,1,False{},1}]

def divided_one(+value: Nat) -> {Nat.div(value,1n) == value : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(value,1n),(value,0n),
      NatDivision.unique(value,1n,value,0n,
        Equal.sym(Nat,Nat.add(Nat.mul(value,1n),0n),value,
          Equal.trans(Nat,Nat.add(Nat.mul(value,1n),0n),Nat.mul(value,1n),value,Addition.zero_right(Nat.mul(value,1n)),Order.times_one(value))),{==},{==})) :
    {Pair.fst(Nat,Nat,_) == value : Nat}
  {==}

# Index index of the part of blocks reads cell index % count of the blocks
# at position index / count.
def blocks_cell(+positions: U32,+groups: U32,+count: U32,+first: U32,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Part.total(blocks_axes(positions,groups,count))),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(index,Part.total(blocks_axes(positions,groups,count))) == True{} : Bool}) ->
  {Part.cell_source(blocks_axes(positions,groups,count),index,first,0) ==
    U32.from_nat(Nat.add(Nat.mul(Nat.div(index,U32.to_nat(count)),U32.to_nat(groups)),Nat.add(U32.to_nat(first),Nat.mod(index,U32.to_nat(count))))) : U32}:
  +c = U32.to_nat(count)
  +g = U32.to_nat(groups)
  +f = U32.to_nat(first)
  +q = Nat.div(index,c)
  +r = Nat.mod(index,c)
  Equal.trans(U32,Part.cell_source(blocks_axes(positions,groups,count),index,first,0),
    (first + Box.stepped_sum(ViewPart.move_extents(blocks_axes(positions,groups,count)),ViewPart.move_steps(blocks_axes(positions,groups,count)),index) : U32),
    U32.from_nat(Nat.add(Nat.mul(q,g),Nat.add(f,r))),
    ViewPart.cell_sum(blocks_axes(positions,groups,count),index,first,0,cap,cap_ok,room,inside),
    {%Equal.sym(Nat,Nat.mul(c,1n),c,Order.times_one(c)) :
        {(first + ((U32.from_nat(Nat.div(index,_)) * groups : U32) + ((U32.from_nat(Nat.div(Nat.mod(index,_),1n)) * 1 : U32) + 0 : U32) : U32) : U32) ==
          U32.from_nat(Nat.add(Nat.mul(q,g),Nat.add(f,r))) : U32}
      %Equal.sym(Nat,Nat.div(r,1n),r,divided_one(r)) :
        {(first + ((U32.from_nat(q) * groups : U32) + ((U32.from_nat(_) * 1 : U32) + 0 : U32) : U32) : U32) == U32.from_nat(Nat.add(Nat.mul(q,g),Nat.add(f,r))) : U32}
      %Equal.sym(U32,(U32.from_nat(r) * 1 : U32),U32.from_nat(r),Arithmetic.multiply_one(U32.from_nat(r))) :
        {(first + ((U32.from_nat(q) * groups : U32) + (_ + 0 : U32) : U32) : U32) == U32.from_nat(Nat.add(Nat.mul(q,g),Nat.add(f,r))) : U32}
      %Equal.sym(U32,(U32.from_nat(r) + 0 : U32),U32.from_nat(r),Arithmetic.add_zero(U32.from_nat(r))) :
        {(first + ((U32.from_nat(q) * groups : U32) + _ : U32) : U32) == U32.from_nat(Nat.add(Nat.mul(q,g),Nat.add(f,r))) : U32}
      %Equal.sym(U32,(U32.from_nat(q) * groups : U32),U32.from_nat(Nat.mul(q,g)),Sources.product_word(q,groups)) :
        {(first + (_ + U32.from_nat(r) : U32) : U32) == U32.from_nat(Nat.add(Nat.mul(q,g),Nat.add(f,r))) : U32}
      %Equal.sym(U32,(U32.from_nat(Nat.mul(q,g)) + U32.from_nat(r) : U32),U32.from_nat(Nat.add(Nat.mul(q,g),r)),WordArithmetic.native_add(Nat.mul(q,g),r)) :
        {(first + _ : U32) == U32.from_nat(Nat.add(Nat.mul(q,g),Nat.add(f,r))) : U32}
      %Arithmetic.native_roundtrip(first) : {(_ + U32.from_nat(Nat.add(Nat.mul(q,g),r)) : U32) == U32.from_nat(Nat.add(Nat.mul(q,g),Nat.add(f,r))) : U32}
      %Equal.sym(U32,(U32.from_nat(f) + U32.from_nat(Nat.add(Nat.mul(q,g),r)) : U32),U32.from_nat(Nat.add(f,Nat.add(Nat.mul(q,g),r))),
          WordArithmetic.native_add(f,Nat.add(Nat.mul(q,g),r))) :
        {_ == U32.from_nat(Nat.add(Nat.mul(q,g),Nat.add(f,r))) : U32}
      Equal.cong(Nat,U32,value => U32.from_nat(value),Nat.add(f,Nat.add(Nat.mul(q,g),r)),Nat.add(Nat.mul(q,g),Nat.add(f,r)),RowMajor.add_swap(f,Nat.mul(q,g),r)) :
      {(first + Box.stepped_sum(ViewPart.move_extents(blocks_axes(positions,groups,count)),ViewPart.move_steps(blocks_axes(positions,groups,count)),index) : U32) ==
        U32.from_nat(Nat.add(Nat.mul(q,g),Nat.add(f,r))) : U32}})

# The part of blocks inside the groups of every position reads cells below
# the cells of all positions.
def blocks_below(+positions: U32,+groups: U32,+count: U32,+first: U32,+high: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Part.total(blocks_axes(positions,groups,count))),U32.to_nat(cap)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)) == True{} : Bool},
  +whole: {Nat.is_le(Nat.mul(U32.to_nat(positions),U32.to_nat(groups)),high) == True{} : Bool},
  +capacity: U32,+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,Part.total(blocks_axes(positions,groups,count))) == True{} : Bool}) ->
  Gather.Between<0n,high,U32.to_nat(Part.cell_source(blocks_axes(positions,groups,count),index,first,0))>:
  +p = U32.to_nat(positions)
  +c = U32.to_nat(count)
  +g = U32.to_nat(groups)
  +f = U32.to_nat(first)
  +q = Nat.div(index,c)
  +r = Nat.mod(index,c)
  +n = Nat.add(Nat.mul(q,g),Nat.add(f,r))
  +product_positive = Order.le_lt_trans(0n,index,Nat.mul(p,Nat.mul(c,1n)),Order.zero_le(index),inside)
  +count_positive = {%Order.times_one(c) : {Nat.is_lt(0n,_) == True{} : Bool}
    RowMajor.left_factor_positive(Nat.mul(c,1n),p,%RowMajor.multiply_commutative(p,Nat.mul(c,1n)) : {Nat.is_lt(0n,_) == True{} : Bool} product_positive) :
    {Nat.is_lt(0n,c) == True{} : Bool}}
  +q_below = DivisionOrder.quotient_below(index,p,c,count_positive,%Order.times_one(c) : {Nat.is_lt(index,Nat.mul(p,_)) == True{} : Bool} inside)
  +part_below = Bounds.nat_trans(Nat.add(f,r),Nat.add(f,c),g,Order.offset_lt(f,r,c,DivisionOrder.remainder_small(index,c,count_positive)),fits)
  +next = {%Equal.sym(Nat,Nat.add(Nat.mul(q,g),g),Nat.add(g,Nat.mul(q,g)),Order.commutative(Nat.mul(q,g),g)) : {Nat.is_le(_,Nat.mul(p,g)) == True{} : Bool}
    RowMajor.mul_le(1n+q,p,g,Order.strict_follow(q,p,q_below)) : {Nat.is_le(Nat.add(Nat.mul(q,g),g),Nat.mul(p,g)) == True{} : Bool}}
  +below = Bounds.nat_trans(n,Nat.mul(p,g),high,Bounds.nat_trans(n,Nat.add(Nat.mul(q,g),g),Nat.mul(p,g),Order.offset_lt(Nat.mul(q,g),Nat.add(f,r),g,part_below),next),whole)
  Gather.Between{Order.zero_le(U32.to_nat(Part.cell_source(blocks_axes(positions,groups,count),index,first,0))),
    %Equal.sym(U32,Part.cell_source(blocks_axes(positions,groups,count),index,first,0),U32.from_nat(n),blocks_cell(positions,groups,count,first,index,cap,cap_ok,room,inside)) :
      {Nat.is_lt(U32.to_nat(_),high) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(n)),n,FullRange.roundtrip(n,capacity,Order.transitive(n,high,U32.to_nat(capacity),Bounds.strict_le(n,high,below),bounded))) :
      {Nat.is_lt(_,high) == True{} : Bool}
    below}

# Pieces that tile the cells of all positions hold every cell the part of
# blocks reads.
def blocks_held(count: Nat,+positions: U32,+groups: U32,+rows: U32,+first: U32,+high: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Part.total(blocks_axes(positions,groups,rows))),U32.to_nat(cap)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(groups)) == True{} : Bool},
  +whole: {Nat.is_le(Nat.mul(U32.to_nat(positions),U32.to_nat(groups)),high) == True{} : Bool},
  +capacity: U32,+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +stack: Held.Stack<Tile.Tile8>,+tiles: Held.Tiles(~Tile.Tile8,stack,0n,high),+index: Nat,
  +inside: {Nat.is_le(Nat.add(index,count),Part.total(blocks_axes(positions,groups,rows))) == True{} : Bool}) ->
  Part.CellsHeld(blocks_axes(positions,groups,rows),first,0,Held.spans(~Tile.Tile8,stack),index,count):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +total = Part.total(blocks_axes(positions,groups,rows))
      Held.Both{Gather.between_held(~Tile.Tile8,stack,0n,high,U32.to_nat(Part.cell_source(blocks_axes(positions,groups,rows),index,first,0)),tiles,
          blocks_below(positions,groups,rows,first,high,cap,cap_ok,room,fits,whole,capacity,bounded,index,Order.room_index(index,rest,total,inside))),
        blocks_held(rest,positions,groups,rows,first,high,cap,cap_ok,room,fits,whole,capacity,bounded,stack,tiles,1n+index,
          Order.advance_room(index,rest,total,inside))}

# The gradient cells of the blocks [first, first + rows) at every position lie
# below the capacity.
def blocks_count_room(+positions: U32,+groups: U32,+first: U32,+rows: U32,+high: Nat,+capacity: Nat,
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(groups)) == True{} : Bool},
  +whole: {Nat.is_le(Nat.mul(U32.to_nat(positions),U32.to_nat(groups)),high) == True{} : Bool},+bounded: {Nat.is_le(high,capacity) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(positions),U32.to_nat(rows)),capacity) == True{} : Bool}:
  Order.transitive(Nat.mul(U32.to_nat(positions),U32.to_nat(rows)),Nat.mul(U32.to_nat(positions),U32.to_nat(groups)),capacity,
    RowMajor.mul_le_left(U32.to_nat(positions),U32.to_nat(rows),U32.to_nat(groups),Order.drop_left_le(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(groups),fits)),
    Order.transitive(Nat.mul(U32.to_nat(positions),U32.to_nat(groups)),high,capacity,whole,bounded))

# Twice the cells of the part of blocks fit the half word.
def blocks_total_room(+positions: U32,+groups: U32,+rows: U32,+limits: Machine.Limits,
  +copied: {Nat.is_le(Nat.mul(U32.to_nat(positions),U32.to_nat(rows)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(Part.total(blocks_axes(positions,groups,rows))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +p = U32.to_nat(positions)
  +c = U32.to_nat(rows)
  Limits.double_bound(limits,Part.total(blocks_axes(positions,groups,rows)),
    %Equal.sym(Nat,Nat.mul(c,1n),c,Order.times_one(c)) : {Nat.is_le(Nat.mul(p,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} copied)

# The copy of the blocks of a covering gradient, held as the only piece,
# covers its cells with the cells of the blocks.
def blocks_alone(~D: Data,~dv: D -> U32 -> Tile.Tile8,+gradient: D,+model: Held.Holding<Tile.Tile8>,+positions: U32,+groups: U32,+first: U32,+rows: U32,
  +limits: Machine.Limits,+high: Nat,
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(groups)) == True{} : Bool},
  +groups_room: {Nat.is_le(U32.to_nat(groups),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +whole: {Nat.is_le(Nat.mul(U32.to_nat(positions),U32.to_nat(groups)),high) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,model,0n,high),+values: Held.HoldingValues(~Tile.Tile8,~D,~dv,gradient,model)) ->
  Held.Covered<Tile.Tile8,Gather.Parted<D>,Gather.parted_value(~Tile.Tile8,~D,~dv),Gather.Parted{gradient,blocks_axes(positions,groups,rows),first},
    Move.held(~Tile.Tile8,Partition.Leaf{Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Pair.snd(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
      Move.part(~Tile.Tile8,Move.Part{blocks_axes(positions,groups,rows),first,0},Held.pack(~Tile.Tile8,model),
        Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),(positions * rows : U32)))))),0,(positions * rows : U32),Unit{}}),
    0n,U32.to_nat((positions * rows : U32)),Held.layout_spans(Pieces.Range{0,(positions * rows : U32)},Held.Ended{})>:
  +p = U32.to_nat(positions)
  +c = U32.to_nat(rows)
  +g = U32.to_nat(groups)
  +capacity = Machine.capacity(limits)
  +half = Machine.half_word(limits)
  +copied = blocks_count_room(positions,groups,first,rows,high,U32.to_nat(capacity),fits,whole,bounded)
  +count_exact = word_product(positions,rows,capacity,copied)
  +total_exact = {%Equal.sym(Nat,Nat.mul(c,1n),c,Order.times_one(c)) : {Nat.mul(p,_) == U32.to_nat((positions * rows : U32)) : Nat}
    Equal.sym(Nat,U32.to_nat((positions * rows : U32)),Nat.mul(p,c),count_exact) :
    {Part.total(blocks_axes(positions,groups,rows)) == U32.to_nat((positions * rows : U32)) : Nat}}
  +room = blocks_total_room(positions,groups,rows,limits,copied)
  +one_room = Order.transitive(1n,7n,U32.to_nat(capacity),{==},Bounds.as_nat_le(7,capacity,Limits.seven_within_capacity(limits)))
  Gather.parted_alone(~Tile.Tile8,~D,~dv,gradient,model,blocks_axes(positions,groups,rows),first,Tile.filled(0.0),(positions * rows : U32),
    Arithmetic.count_within((positions * rows : U32),Nat.mul(p,c),capacity,1073741824,count_exact,copied,Limits.capacity_allocatable(limits)),
    total_exact,half,Limits.half_word_fits(limits),room,
    Held.Both{Equal.sym(U32,(rows * 1 : U32),rows,Arithmetic.multiply_one(rows)),Held.Both{{==},Unit{}}},
    half,Limits.half_word_fits(limits),
    Held.Both{Logic.when_held({Nat.is_le(Nat.double(g),U32.to_nat(half)) == True{} : Bool},U32.is_lt(1,positions),Limits.double_bound(limits,g,groups_room)),
      Held.Both{Logic.when_held({Nat.is_le(Nat.double(1n),U32.to_nat(half)) == True{} : Bool},U32.is_lt(1,rows),Limits.double_bound(limits,1n,one_room)),Unit{}}},
    0n,high,capacity,Limits.capacity_fits(limits),bounded,tiled,values,
    blocks_held(Part.total(blocks_axes(positions,groups,rows)),positions,groups,rows,first,high,half,Limits.half_word_fits(limits),room,fits,whole,
      capacity,bounded,Held.order(~Tile.Tile8,model),Held.order_tiles(~Tile.Tile8,model,0n,high,tiled),0n,Order.reflexive(Part.total(blocks_axes(positions,groups,rows)))))

# ---- The parts of the outputs, made from copies ----

# The words of the positions, the output blocks and the input cells of a work.
def work_positions(+work: Parameter.Work) -> U32:
  Parameter.Work{+n,c,o,h,w,k,s,p,+rows,+columns,g} = work
  U32.from_nat(Nat.mul(n,Nat.mul(rows,columns)))

def work_groups(+work: Parameter.Work) -> U32:
  Parameter.Work{n,c,+o,h,w,k,s,p,rows,columns,g} = work
  U32.from_nat(Nat.div(Nat.add(o,7n),8n))

def work_cells(+work: Parameter.Work) -> U32:
  Parameter.Work{+n,+c,o,+h,+w,k,s,p,rows,columns,g} = work
  U32.from_nat(Nat.mul(n,Nat.mul(Nat.div(Nat.add(c,7n),8n),Nat.mul(h,w))))

# A part of the output blocks [first, first + rows): its copies cover their
# cells with the input of the batch and with the gradient of its blocks at
# every position, and hold the input cells of all images and the gradient
# cells of its blocks.
type CopiedBlocks<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-batch: Images.Batch<X,xv,D,dv>,-work: Parameter.Work,-first: U32,-rows: U32,
  -input: Buffer.Model<Tile.Tile8>,-x_count: U32,-gradient: Buffer.Model<Tile.Tile8>,-d_count: U32> is Data:
  CopiedBlocks{
    x_covered: Held.Covered<Tile.Tile8,Held.Shifted<X>,Held.shifted_value(~Tile.Tile8,~X,~xv),Held.Shifted{Images.batch_input(~X,~xv,~D,~dv,batch),0},
      Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,input)),0n,U32.to_nat(x_count),Held.layout_spans(Pieces.Range{0,x_count},Held.Ended{})>,
    d_covered: Held.Covered<Tile.Tile8,Gather.Parted<D>,Gather.parted_value(~Tile.Tile8,~D,~dv),
      Gather.Parted{Images.batch_gradient(~X,~xv,~D,~dv,batch),blocks_axes(work_positions(work),work_groups(work),rows),first},
      Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,gradient)),0n,U32.to_nat(d_count),Held.layout_spans(Pieces.Range{0,d_count},Held.Ended{})>,
    x_exact: {U32.to_nat(x_count) == Nat.mul(Images.batch_images(~X,~xv,~D,~dv,batch),Sources.input_cells(Images.batch_geometry(~X,~xv,~D,~dv,batch))) : Nat},
    d_exact: {U32.to_nat(d_count) == Nat.mul(U32.to_nat(work_positions(work)),U32.to_nat(rows)) : Nat}}

def blocks_fact(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+scope: KernelGemm.Both<Images.Batch<X,xv,D,dv>,Parameter.Work>,+part: CopiedPart,
  +first: U32,+rows: U32) -> Data:
  KernelGemm.Both{batch,work} = scope
  CopiedPart{input,x_count,gradient,d_count} = part
  CopiedBlocks<X,xv,D,dv,batch,work,first,rows,input,x_count,gradient,d_count>

# The positions, the output blocks and the input cells of a work are those of
# the batch, and the limit of the plan is its output blocks.
type BlocksReady<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8,-batch: Images.Batch<X,xv,D,dv>,-work: Parameter.Work,-limit: Nat> is Data:
  BlocksReady{positions: {U32.to_nat(work_positions(work)) == Nat.mul(Images.batch_images(~X,~xv,~D,~dv,batch),
      Nat.mul(Groups.row_count(Images.batch_geometry(~X,~xv,~D,~dv,batch)),Groups.column_count(Images.batch_geometry(~X,~xv,~D,~dv,batch)))) : Nat},
    groups: {U32.to_nat(work_groups(work)) == Groups.output_blocks(Images.batch_geometry(~X,~xv,~D,~dv,batch)) : Nat},
    cells: {U32.to_nat(work_cells(work)) == Nat.mul(Images.batch_images(~X,~xv,~D,~dv,batch),Sources.input_cells(Images.batch_geometry(~X,~xv,~D,~dv,batch))) : Nat},
    blocks: {limit == Groups.output_blocks(Images.batch_geometry(~X,~xv,~D,~dv,batch)) : Nat}}

def blocks_ready(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,batch: Images.Batch<X,xv,D,dv>,work: Parameter.Work,limit: Nat) -> Data:
  BlocksReady<X,xv,D,dv,batch,work,limit>

# The gradient cells of the blocks at every position of all images.
def blocks_product(+images: Nat,+rows: Nat,+columns: Nat,+blocks: Nat) ->
  {Nat.mul(Nat.mul(images,Nat.mul(rows,columns)),blocks) == Nat.mul(images,Nat.mul(Nat.mul(blocks,rows),columns)) : Nat}:
  Equal.trans(Nat,Nat.mul(Nat.mul(images,Nat.mul(rows,columns)),blocks),Nat.mul(images,Nat.mul(Nat.mul(rows,columns),blocks)),
    Nat.mul(images,Nat.mul(Nat.mul(blocks,rows),columns)),RowMajor.assoc(images,Nat.mul(rows,columns),blocks),
    Equal.cong(Nat,Nat,value => Nat.mul(images,value),Nat.mul(Nat.mul(rows,columns),blocks),Nat.mul(Nat.mul(blocks,rows),columns),
      Equal.trans(Nat,Nat.mul(Nat.mul(rows,columns),blocks),Nat.mul(blocks,Nat.mul(rows,columns)),Nat.mul(Nat.mul(blocks,rows),columns),
        RowMajor.multiply_commutative(Nat.mul(rows,columns),blocks),
        Equal.sym(Nat,Nat.mul(Nat.mul(blocks,rows),columns),Nat.mul(blocks,Nat.mul(rows,columns)),RowMajor.assoc(blocks,rows,columns)))))

# The gradient cells of all positions hold those of the output blocks.
def positions_room(+images: Nat,+rows: Nat,+columns: Nat,+blocks: Nat,+positions: U32,+groups: U32,+high: Nat,
  +positions_exact: {U32.to_nat(positions) == Nat.mul(images,Nat.mul(rows,columns)) : Nat},+groups_exact: {U32.to_nat(groups) == blocks : Nat},
  +whole: {Nat.is_le(Nat.mul(images,Nat.mul(Nat.mul(blocks,rows),columns)),high) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(positions),U32.to_nat(groups)),high) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(positions),Nat.mul(images,Nat.mul(rows,columns)),positions_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(groups)),high) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(groups),blocks,groups_exact) : {Nat.is_le(Nat.mul(Nat.mul(images,Nat.mul(rows,columns)),_),high) == True{} : Bool}
  %Equal.sym(Nat,Nat.mul(Nat.mul(images,Nat.mul(rows,columns)),blocks),Nat.mul(images,Nat.mul(Nat.mul(blocks,rows),columns)),blocks_product(images,rows,columns,blocks)) :
    {Nat.is_le(_,high) == True{} : Bool}
  whole

# A leaf of the build of the parts of the outputs copies the input cells of
# all images and the gradient cells of its blocks from covering holdings.
def blocks_leaf(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,+work: Parameter.Work,+limit: Nat,
  +ready: blocks_ready(~X,~xv,~D,~dv,batch,work,limit),+first: U32,+rows: U32,+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),limit) == True{} : Bool},
  +x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>,+holds: HeldBoth<X,xv,D,dv,batch,x,d>) ->
  PartsDone<X,xv,D,dv,CopiedPart,copied_owners,Parameter.Work,blocks_fact(~X,~xv,~D,~dv),batch,work,Pieces.Range{first,rows},Kernel.own(work,first,rows,holders(x,d))>:
  Images.Batch{+input,+gradient,+geometry,+domain,+images,+height,+width,+group,+size,+bounded,+x_high,+x_layout,+x_whole,+x_bounded,+d_high,+d_layout,+d_whole,+d_bounded} = batch
  Parameter.Work{+n,+c,+o,+h,+w,+k,+s,+p,+out_rows,+out_columns,+g} = work
  BlocksReady{+positions_exact,+groups_exact,+cells_exact,+limit_exact} = ready
  HeldBoth{+x_tiled,+x_values,+x_ordered,+d_tiled,+d_values,+d_ordered} = holds
  +shape = {Images.Batch{input,gradient,geometry,domain,images,height,width,group,size,bounded,x_high,x_layout,x_whole,x_bounded,d_high,d_layout,d_whole,d_bounded} :
    Images.Batch<X,xv,D,dv>}
  +shape_work = {Parameter.Work{n,c,o,h,w,k,s,p,out_rows,out_columns,g} : Parameter.Work}
  +limits = Groups.domain_limits(geometry,domain)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +fill = Tile.filled(0.0)
  +positions = U32.from_nat(Nat.mul(n,Nat.mul(out_rows,out_columns)))
  +groups = U32.from_nat(Nat.div(Nat.add(o,7n),8n))
  +cells = U32.from_nat(Nat.mul(n,Nat.mul(Nat.div(Nat.add(c,7n),8n),Nat.mul(h,w))))
  +d_count = (positions * rows : U32)
  +blocks = Groups.output_blocks(geometry)
  +r = Groups.row_count(geometry)
  +q = Groups.column_count(geometry)
  +x_total = Nat.mul(images,Sources.input_cells(geometry))
  +x_room = Order.transitive(x_total,x_high,cap,x_whole,x_bounded)
  +fits = {%Equal.sym(Nat,U32.to_nat(groups),blocks,groups_exact) : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool}
    %limit_exact : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool}
    room : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(groups)) == True{} : Bool}}
  +groups_room = {%Equal.sym(Nat,U32.to_nat(groups),blocks,groups_exact) : {Nat.is_le(_,cap) == True{} : Bool}
    Groups.output_blocks_room(geometry,domain) : {Nat.is_le(U32.to_nat(groups),cap) == True{} : Bool}}
  +whole = positions_room(images,r,q,blocks,positions,groups,d_high,positions_exact,groups_exact,d_whole)
  +d_room = blocks_count_room(positions,groups,first,rows,d_high,cap,fits,whole,d_bounded)
  copies_done(~X,~xv,~D,~dv,~Parameter.Work,~blocks_fact(~X,~xv,~D,~dv),shape,shape_work,first,rows,
    Move.copied(~Tile.Tile8,Held.pack(~Tile.Tile8,x),0,cells,fill),cells,
    Move.part(~Tile.Tile8,Move.Part{blocks_axes(positions,groups,rows),first,0},Held.pack(~Tile.Tile8,d),Tensors.buffer(~Tile.Tile8,fill,d_count)),d_count,
    copy_split(Move.copied(~Tile.Tile8,Held.pack(~Tile.Tile8,x),0,cells,fill)),
    copy_split(Move.part(~Tile.Tile8,Move.Part{blocks_axes(positions,groups,rows),first,0},Held.pack(~Tile.Tile8,d),Tensors.buffer(~Tile.Tile8,fill,d_count))),
    Held.copied_covered(~Tile.Tile8,~X,~xv,input,x,0,cells,fill,0n,x_high,x_tiled,x_values,x_layout,x_ordered),
    Gather.parted_covered(~Tile.Tile8,~D,~dv,gradient,blocks_axes(positions,groups,rows),first,0,fill,d_count,Held.pack(~Tile.Tile8,d),0n,d_high,d_layout,
      Held.Covered{d,{==},d_tiled,d_values,d_ordered}),
    copy_model(Move.copied(~Tile.Tile8,Held.pack(~Tile.Tile8,x),0,cells,fill)),{==},
    copy_model(Move.part(~Tile.Tile8,Move.Part{blocks_axes(positions,groups,rows),first,0},Held.pack(~Tile.Tile8,d),Tensors.buffer(~Tile.Tile8,fill,d_count))),{==},
    CopiedBlocks{
      Held.copied_alone(~Tile.Tile8,~X,~xv,input,x,0,cells,fill,0n,x_high,capacity,Limits.capacity_fits(limits),x_bounded,x_tiled,x_values,Order.zero_le(0n),
        %Equal.sym(Nat,U32.to_nat(cells),x_total,cells_exact) : {Nat.is_le(_,x_high) == True{} : Bool} x_whole,
        Arithmetic.count_within(cells,x_total,capacity,1073741824,cells_exact,x_room,Limits.capacity_allocatable(limits))),
      blocks_alone(~D,~dv,gradient,d,positions,groups,first,rows,limits,d_high,fits,groups_room,whole,d_bounded,d_tiled,d_values),
      cells_exact,
      word_product(positions,rows,capacity,d_room)})

# ---- The lanes of the parts of the outputs ----

# The geometry of a part of the outputs.
def outputs_geometry(+geometry: Kernel.Geometry,+outputs: U32) -> Kernel.Geometry:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Kernel.Geometry{c,outputs,h,w,k,s,p,rows,columns,taps}

# A part of the outputs reads the input of the whole call from its copy.
def blocks_weight(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+input: X,+gradient: D,+axes: List<&2,Move.Axis>,+first: U32,
  +geometry: Kernel.Geometry,+outputs: U32,+image: Nat,+tap: U32,+position: Nat) ->
  {Sources.held_weight(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),
      Sources.HeldSource{Held.Shifted{input,0},Gather.Parted{gradient,axes,first},outputs_geometry(geometry,outputs),
        Nat.mul(image,Sources.input_cells(outputs_geometry(geometry,outputs))),Nat.mul(image,Groups.gradient_extent(outputs_geometry(geometry,outputs)))},tap,position) ==
    Sources.held_weight(~X,~xv,~D,~dv,Sources.HeldSource{input,gradient,geometry,Nat.mul(image,Sources.input_cells(geometry)),Nat.mul(image,Groups.gradient_extent(geometry))},
      tap,position) : F32}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +first_cell = Nat.mul(image,Nat.mul(Nat.mul(Ceiling.ceiling(U32.to_nat(c),8n),U32.to_nat(h)),U32.to_nat(w)))
  +row = U32.from_nat(Nat.div(position,U32.to_nat(columns)))
  +column = U32.from_nat(Nat.mod(position,U32.to_nat(columns)))
  +channel = Nat.div(U32.to_nat(tap),Nat.mul(U32.to_nat(k),U32.to_nat(k)))
  +y = Nat.add(Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(k)),Nat.mul(U32.to_nat(row),U32.to_nat(s)))
  +x_at = Nat.add(Nat.mod(U32.to_nat(tap),U32.to_nat(k)),Nat.mul(U32.to_nat(column),U32.to_nat(s)))
  +groups = U32.to_nat(((c + 7) / 8 : U32))
  Packing.sampled_agrees(~Held.Shifted<X>,~Sources.held_cell(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv)),~X,~Sources.held_cell(~X,~xv),
    Held.Shifted{input,0},input,first_cell,first_cell,channel,y,x_at,U32.to_nat(h),U32.to_nat(w),U32.to_nat(p),groups,
    inside_true => base_read(~X,~xv,input,0,0n,{==},first_cell,Nat.div(channel,8n),
      Nat.mul(Nat.add(Nat.mul(Nat.sub(y,U32.to_nat(p)),U32.to_nat(w)),Nat.sub(x_at,U32.to_nat(p))),groups)))

# The cell of block column at a position of an image of blocks blocks at
# every position of rows by columns.
def part_index(+image: Nat,+rows: Nat,+columns: Nat,+position: Nat,+blocks: Nat,+column: Nat) ->
  {Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,rows),columns)),Nat.mul(position,blocks)),column) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(rows,columns)),position),blocks),column) : Nat}:
  +area = Nat.mul(rows,columns)
  +extent = Equal.trans(Nat,Nat.mul(Nat.mul(blocks,rows),columns),Nat.mul(blocks,area),Nat.mul(area,blocks),RowMajor.assoc(blocks,rows,columns),
    RowMajor.multiply_commutative(blocks,area))
  +scaled = Equal.trans(Nat,Nat.mul(image,Nat.mul(Nat.mul(blocks,rows),columns)),Nat.mul(image,Nat.mul(area,blocks)),Nat.mul(Nat.mul(image,area),blocks),
    Equal.cong(Nat,Nat,value => Nat.mul(image,value),Nat.mul(Nat.mul(blocks,rows),columns),Nat.mul(area,blocks),extent),
    Equal.sym(Nat,Nat.mul(Nat.mul(image,area),blocks),Nat.mul(image,Nat.mul(area,blocks)),RowMajor.assoc(image,area,blocks)))
  Equal.cong(Nat,Nat,value => Nat.add(value,column),Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,rows),columns)),Nat.mul(position,blocks)),
    Nat.mul(Nat.add(Nat.mul(image,area),position),blocks),
    Equal.trans(Nat,Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,rows),columns)),Nat.mul(position,blocks)),Nat.add(Nat.mul(Nat.mul(image,area),blocks),Nat.mul(position,blocks)),
      Nat.mul(Nat.add(Nat.mul(image,area),position),blocks),
      Equal.cong(Nat,Nat,value => Nat.add(value,Nat.mul(position,blocks)),Nat.mul(image,Nat.mul(Nat.mul(blocks,rows),columns)),Nat.mul(Nat.mul(image,area),blocks),scaled),
      Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,area),position),blocks),Nat.add(Nat.mul(Nat.mul(image,area),blocks),Nat.mul(position,blocks)),
        RowMajor.distrib(Nat.mul(image,area),position,blocks))))

# Output first * 8 + output lies in block first + output / 8, at lane
# output % 8.
def eight_split(+first: Nat,+output: Nat) -> {Nat.divmod(Nat.add(Nat.mul(first,8n),output),8n) == (Nat.add(first,Nat.div(output,8n)),Nat.mod(output,8n)) : Nat & Nat}:
  +block = Nat.div(output,8n)
  +lane = Nat.mod(output,8n)
  NatDivision.unique(Nat.add(Nat.mul(first,8n),output),8n,Nat.add(first,block),lane,
    Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(first,block),8n),lane),Nat.add(Nat.mul(first,8n),output),
      Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(first,block),8n),lane),Nat.add(Nat.add(Nat.mul(first,8n),Nat.mul(block,8n)),lane),Nat.add(Nat.mul(first,8n),output),
        Equal.cong(Nat,Nat,value => Nat.add(value,lane),Nat.mul(Nat.add(first,block),8n),Nat.add(Nat.mul(first,8n),Nat.mul(block,8n)),RowMajor.distrib(first,block,8n)),
        Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(first,8n),Nat.mul(block,8n)),lane),Nat.add(Nat.mul(first,8n),Nat.add(Nat.mul(block,8n),lane)),Nat.add(Nat.mul(first,8n),output),
          Addition.associative(Nat.mul(first,8n),Nat.mul(block,8n),lane),
          Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(first,8n),value),Nat.add(Nat.mul(block,8n),lane),output,
            Equal.sym(Nat,output,Nat.add(Nat.mul(block,8n),lane),DivisionOrder.quotient_exact(output,8n,{==})))))),
    {==},DivisionOrder.remainder_small(output,8n,{==}))

# A part of the outputs reads the gradient of output first * 8 + output of the
# whole call from its copy of the blocks.
def blocks_input(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+input: X,+gradient: D,+geometry: Kernel.Geometry,+outputs: U32,
  +positions: U32,+groups: U32,+rows: U32,+first: U32,+images: Nat,+image: Nat,+part_output: U32,+whole_output: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(Part.total(blocks_axes(positions,groups,rows))),U32.to_nat(cap)) == True{} : Bool},
  +rows_exact: {Ceiling.ceiling(U32.to_nat(outputs),8n) == U32.to_nat(rows) : Nat},+groups_exact: {U32.to_nat(groups) == Groups.output_blocks(geometry) : Nat},
  +positions_exact: {U32.to_nat(positions) == Nat.mul(images,Nat.mul(Groups.row_count(geometry),Groups.column_count(geometry))) : Nat},
  +whole_exact: {U32.to_nat(whole_output) == Nat.add(Nat.mul(U32.to_nat(first),8n),U32.to_nat(part_output)) : Nat},
  +output_inside: {Nat.is_lt(U32.to_nat(part_output),U32.to_nat(outputs)) == True{} : Bool},+image_inside: {Nat.is_lt(image,images) == True{} : Bool},
  +position: Nat,+position_inside: {Nat.is_lt(position,Nat.mul(Groups.row_count(geometry),Groups.column_count(geometry))) == True{} : Bool}) ->
  {Sources.held_input(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),
      Sources.HeldSource{Held.Shifted{input,0},Gather.Parted{gradient,blocks_axes(positions,groups,rows),first},outputs_geometry(geometry,outputs),
        Nat.mul(image,Sources.input_cells(outputs_geometry(geometry,outputs))),Nat.mul(image,Groups.gradient_extent(outputs_geometry(geometry,outputs)))},part_output,position) ==
    Sources.held_input(~X,~xv,~D,~dv,Sources.HeldSource{input,gradient,geometry,Nat.mul(image,Sources.input_cells(geometry)),Nat.mul(image,Groups.gradient_extent(geometry))},
      whole_output,position) : F32}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+height,+width,+taps} = geometry
  +f = U32.to_nat(first)
  +count = U32.to_nat(rows)
  +g = U32.to_nat(groups)
  +op = U32.to_nat(part_output)
  +block = Nat.div(op,8n)
  +lane = Nat.mod(op,8n)
  +blocks = Ceiling.ceiling(U32.to_nat(o),8n)
  +tall = U32.to_nat(height)
  +wide = U32.to_nat(width)
  +area = Nat.mul(tall,wide)
  +q = Nat.add(Nat.mul(image,area),position)
  +index = Nat.add(Nat.mul(q,count),block)
  +axes = blocks_axes(positions,groups,rows)
  +block_below = DivisionOrder.quotient_below(op,count,8n,{==},
    Bounds.nat_trans(op,U32.to_nat(outputs),Nat.mul(count,8n),output_inside,
      %rows_exact : {Nat.is_le(U32.to_nat(outputs),Nat.mul(_,8n)) == True{} : Bool} Ceiling.covers(U32.to_nat(outputs),8n,{==})))
  +index_below = {%Equal.sym(Nat,Nat.mul(count,1n),count,Order.times_one(count)) : {Nat.is_lt(index,Nat.mul(U32.to_nat(positions),_)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(positions),Nat.mul(images,area),positions_exact) : {Nat.is_lt(index,Nat.mul(_,count)) == True{} : Bool}
    RowMajor.radix(q,Nat.mul(images,area),block,count,RowMajor.radix(image,images,position,area,image_inside,position_inside),block_below) :
    {Nat.is_lt(index,Part.total(blocks_axes(positions,groups,rows))) == True{} : Bool}}
  +whole_value = TileLaws.select_lane(Nat.mod(U32.to_nat(whole_output),8n),
    dv(gradient,U32.from_nat(Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,tall),wide)),Nat.mul(position,blocks)),Nat.div(U32.to_nat(whole_output),8n)))))
  %Equal.sym(Nat,Ceiling.ceiling(U32.to_nat(outputs),8n),count,rows_exact) :
    {TileLaws.select_lane(lane,dv(gradient,Part.cell_source(axes,Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(_,tall),wide)),Nat.mul(position,_)),block),first,0))) ==
      whole_value : F32}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(count,tall),wide)),Nat.mul(position,count)),block),index,part_index(image,tall,wide,position,count,block)) :
    {TileLaws.select_lane(lane,dv(gradient,Part.cell_source(axes,_,first,0))) == whole_value : F32}
  %Equal.sym(U32,Part.cell_source(axes,index,first,0),U32.from_nat(Nat.add(Nat.mul(Nat.div(index,count),g),Nat.add(f,Nat.mod(index,count)))),
      blocks_cell(positions,groups,rows,first,index,cap,cap_ok,room,index_below)) :
    {TileLaws.select_lane(lane,dv(gradient,_)) == whole_value : F32}
  %Equal.sym(Nat & Nat,Nat.divmod(index,count),(q,block),NatDivision.unique(index,count,q,block,{==},Order.positive_bound(block,count,block_below),block_below)) :
    {TileLaws.select_lane(lane,dv(gradient,U32.from_nat(Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),g),Nat.add(f,Pair.snd(Nat,Nat,_)))))) == whole_value : F32}
  %Equal.sym(Nat,g,Groups.output_blocks(geometry),groups_exact) :
    {TileLaws.select_lane(lane,dv(gradient,U32.from_nat(Nat.add(Nat.mul(q,_),Nat.add(f,block))))) == whole_value : F32}
  %Equal.sym(Nat,U32.to_nat(whole_output),Nat.add(Nat.mul(f,8n),op),whole_exact) :
    {TileLaws.select_lane(lane,dv(gradient,U32.from_nat(Nat.add(Nat.mul(q,blocks),Nat.add(f,block))))) ==
      TileLaws.select_lane(Nat.mod(_,8n),dv(gradient,U32.from_nat(Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,tall),wide)),Nat.mul(position,blocks)),Nat.div(_,8n))))) : F32}
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(f,8n),op),8n),(Nat.add(f,block),lane),eight_split(f,op)) :
    {TileLaws.select_lane(lane,dv(gradient,U32.from_nat(Nat.add(Nat.mul(q,blocks),Nat.add(f,block))))) ==
      TileLaws.select_lane(Pair.snd(Nat,Nat,_),dv(gradient,U32.from_nat(Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,tall),wide)),Nat.mul(position,blocks)),Pair.fst(Nat,Nat,_))))) : F32}
  Equal.cong(Nat,F32,value => TileLaws.select_lane(lane,dv(gradient,U32.from_nat(value))),Nat.add(Nat.mul(q,blocks),Nat.add(f,block)),
    Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,tall),wide)),Nat.mul(position,blocks)),Nat.add(f,block)),
    Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,tall),wide)),Nat.mul(position,blocks)),Nat.add(f,block)),Nat.add(Nat.mul(q,blocks),Nat.add(f,block)),
      part_index(image,tall,wide,position,blocks,Nat.add(f,block))))

# What a part of fewer outputs needs of the geometry of the whole call.
def outputs_ready(+geometry: Kernel.Geometry,+domain: Groups.GeometryReady<geometry>,+outputs: U32,+positive: {Nat.is_lt(0n,U32.to_nat(outputs)) == True{} : Bool},
  +fewer: {Nat.is_le(U32.to_nat(outputs),Groups.output_count(geometry)) == True{} : Bool},
  +blocks_fewer: {Nat.is_le(Ceiling.ceiling(U32.to_nat(outputs),8n),Groups.output_blocks(geometry)) == True{} : Bool}) ->
  Groups.GeometryReady<outputs_geometry(geometry,outputs)>:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  Groups.GeometryReady{+limits,+columns_ok,+kernel,+stride,+square,+taps_ok,+outputs_ok,+count,+aligned,+taps_exact,+pixels,+gradients} = domain
  +capacity = U32.to_nat(Machine.capacity(limits))
  +blocks = Ceiling.ceiling(U32.to_nat(o),8n)
  +fewer_blocks = Ceiling.ceiling(U32.to_nat(outputs),8n)
  Groups.GeometryReady{limits,columns_ok,kernel,stride,square,taps_ok,positive,
    Order.transitive(Nat.mul(U32.to_nat(taps),U32.to_nat(outputs)),Nat.mul(U32.to_nat(taps),U32.to_nat(o)),capacity,
      RowMajor.mul_le_left(U32.to_nat(taps),U32.to_nat(outputs),U32.to_nat(o),fewer),count),
    aligned,taps_exact,pixels,
    Order.transitive(Nat.mul(Nat.mul(fewer_blocks,U32.to_nat(rows)),U32.to_nat(columns)),Nat.mul(Nat.mul(blocks,U32.to_nat(rows)),U32.to_nat(columns)),capacity,
      RowMajor.mul_le(Nat.mul(fewer_blocks,U32.to_nat(rows)),Nat.mul(blocks,U32.to_nat(rows)),U32.to_nat(columns),RowMajor.mul_le(fewer_blocks,blocks,U32.to_nat(rows),blocks_fewer)),
      gradients)}

def outputs_bounded(+geometry: Kernel.Geometry,+outputs: U32,+rows: Nat,+columns: Nat,+group: Nat,+size: U32,
  +bounded: Members.All(Grouped.Range,{range => GroupLaws.RangeBounded(Groups.row_count(geometry),Groups.column_count(geometry),U32.to_nat(size),range) : Grouped.Range -> Data},
    Grouped.ranges(rows,columns,group))) ->
  Members.All(Grouped.Range,{range => GroupLaws.RangeBounded(Groups.row_count(outputs_geometry(geometry,outputs)),Groups.column_count(outputs_geometry(geometry,outputs)),
    U32.to_nat(size),range) : Grouped.Range -> Data},Grouped.ranges(rows,columns,group)):
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+height,+width,+taps} = geometry
  bounded

# The images of a batch whose copies hold its input from cell 0 on and the
# gradient of the blocks of a part, with the geometry of the outputs of the
# part.
def outputs_batch(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+whole: Images.Batch<X,xv,D,dv>,+outputs: U32,+axes: List<&2,Move.Axis>,
  +first: U32,+domain: Groups.GeometryReady<outputs_geometry(Images.batch_geometry(~X,~xv,~D,~dv,whole),outputs)>,+x_high: Nat,+x_layout: Held.Spans,
  +x_room: {Nat.is_le(Nat.mul(Images.batch_images(~X,~xv,~D,~dv,whole),Sources.input_cells(outputs_geometry(Images.batch_geometry(~X,~xv,~D,~dv,whole),outputs))),x_high) == True{} : Bool},
  +x_bounded: {Nat.is_le(x_high,U32.to_nat(Machine.capacity(Groups.domain_limits(outputs_geometry(Images.batch_geometry(~X,~xv,~D,~dv,whole),outputs),domain)))) == True{} : Bool},
  +d_high: Nat,+d_layout: Held.Spans,
  +d_room: {Nat.is_le(Nat.mul(Images.batch_images(~X,~xv,~D,~dv,whole),Groups.gradient_extent(outputs_geometry(Images.batch_geometry(~X,~xv,~D,~dv,whole),outputs))),d_high) == True{} : Bool},
  +d_bounded: {Nat.is_le(d_high,U32.to_nat(Machine.capacity(Groups.domain_limits(outputs_geometry(Images.batch_geometry(~X,~xv,~D,~dv,whole),outputs),domain)))) == True{} : Bool}) ->
  Images.Batch<Held.Shifted<X>,Held.shifted_value(~Tile.Tile8,~X,~xv),Gather.Parted<D>,Gather.parted_value(~Tile.Tile8,~D,~dv)>:
  Images.Batch{input,gradient,+geometry,whole_domain,images,+rows,+columns,+group,+size,bounded,whole_x_high,whole_x_layout,whole_x_room,whole_x_bounded,
    whole_d_high,whole_d_layout,whole_d_room,whole_d_bounded} = whole
  Images.Batch{Held.Shifted{input,0},Gather.Parted{gradient,axes,first},outputs_geometry(geometry,outputs),domain,images,rows,columns,group,size,
    outputs_bounded(geometry,outputs,rows,columns,group,size,bounded),x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded}

# Where a part of the outputs reads an image: the holders of the whole call,
# the outputs and blocks of the part, and one output of the part and of the
# whole call.
type BlocksRead<-X: Data,-D: Data> is Data:
  BlocksRead{input: X,gradient: D,geometry: Kernel.Geometry,outputs: U32,positions: U32,groups: U32,rows: U32,first: U32,images: Nat,image: Nat,tap: U32,
    part_output: U32,whole_output: U32,cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.double(Part.total(blocks_axes(positions,groups,rows))),U32.to_nat(cap)) == True{} : Bool},
    rows_exact: {Ceiling.ceiling(U32.to_nat(outputs),8n) == U32.to_nat(rows) : Nat},groups_exact: {U32.to_nat(groups) == Groups.output_blocks(geometry) : Nat},
    positions_exact: {U32.to_nat(positions) == Nat.mul(images,Nat.mul(Groups.row_count(geometry),Groups.column_count(geometry))) : Nat},
    whole_exact: {U32.to_nat(whole_output) == Nat.add(Nat.mul(U32.to_nat(first),8n),U32.to_nat(part_output)) : Nat},
    output_inside: {Nat.is_lt(U32.to_nat(part_output),U32.to_nat(outputs)) == True{} : Bool},image_inside: {Nat.is_lt(image,images) == True{} : Bool}}

def blocks_lane(~X: Data,~D: Data,read: BlocksRead<X,D>) -> Images.HeldLane<Held.Shifted<X>,Gather.Parted<D>>:
  BlocksRead{input,gradient,+geometry,+outputs,positions,groups,rows,first,images,+image,tap,part_output,whole_output,cap,cap_ok,room,rows_exact,groups_exact,positions_exact,
    whole_exact,output_inside,image_inside} = read
  Images.HeldLane{Sources.HeldSource{Held.Shifted{input,0},Gather.Parted{gradient,blocks_axes(positions,groups,rows),first},outputs_geometry(geometry,outputs),
    Nat.mul(image,Sources.input_cells(outputs_geometry(geometry,outputs))),Nat.mul(image,Groups.gradient_extent(outputs_geometry(geometry,outputs)))},tap,part_output}

def whole_blocks_lane(~X: Data,~D: Data,read: BlocksRead<X,D>) -> Images.HeldLane<X,D>:
  BlocksRead{input,gradient,+geometry,outputs,positions,groups,rows,first,images,+image,tap,part_output,whole_output,cap,cap_ok,room,rows_exact,groups_exact,positions_exact,
    whole_exact,output_inside,image_inside} = read
  Images.HeldLane{Sources.HeldSource{input,gradient,geometry,Nat.mul(image,Sources.input_cells(geometry)),Nat.mul(image,Groups.gradient_extent(geometry))},tap,whole_output}

def blocks_geometry(~X: Data,~D: Data,read: BlocksRead<X,D>) -> Kernel.Geometry:
  BlocksRead{input,gradient,geometry,outputs,positions,groups,rows,first,images,image,tap,part_output,whole_output,cap,cap_ok,room,rows_exact,groups_exact,positions_exact,
    whole_exact,output_inside,image_inside} = read
  geometry

def positions_of(+geometry: Kernel.Geometry) -> Nat:
  Nat.mul(Groups.row_count(geometry),Groups.column_count(geometry))

def blocks_size(~X: Data,~D: Data,read: BlocksRead<X,D>) -> Nat:
  positions_of(blocks_geometry(~X,~D,read))

def BlocksSample(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+read: BlocksRead<X,D>,+position: Nat) -> Data:
  SameSample(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),~X,~xv,~D,~dv,
    blocks_lane(~X,~D,read),whole_blocks_lane(~X,~D,read),position)

def blocks_sample(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+read: BlocksRead<X,D>,+position: Nat,
  +inside: {Nat.is_lt(position,blocks_size(~X,~D,read)) == True{} : Bool}) -> BlocksSample(~X,~xv,~D,~dv,read,position):
  BlocksRead{+input,+gradient,+geometry,+outputs,+positions,+groups,+rows,+first,+images,+image,+tap,+part_output,+whole_output,+cap,+cap_ok,+room,+rows_exact,+groups_exact,
    +positions_exact,+whole_exact,+output_inside,+image_inside} = read
  KernelGemm.Both{
    blocks_input(~X,~xv,~D,~dv,input,gradient,geometry,outputs,positions,groups,rows,first,images,image,part_output,whole_output,cap,cap_ok,room,rows_exact,groups_exact,
      positions_exact,whole_exact,output_inside,image_inside,position,inside),
    blocks_weight(~X,~xv,~D,~dv,input,gradient,blocks_axes(positions,groups,rows),first,geometry,outputs,image,tap,position)}

def blocks_range(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+read: BlocksRead<X,D>,+size: U32,+range: Grouped.Range,
  +bounded: GroupLaws.RangeBounded(Groups.row_count(blocks_geometry(~X,~D,read)),Groups.column_count(blocks_geometry(~X,~D,read)),U32.to_nat(size),range)) ->
  RangeSamples(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),~X,~xv,~D,~dv,
    blocks_lane(~X,~D,read),whole_blocks_lane(~X,~D,read),range):
  Grouped.Range{+first,+count} = range
  KernelGemm.Both{nonempty,KernelGemm.Both{small,+room}} = bounded
  Members.range_all(~BlocksRead<X,D>,~BlocksSample(~X,~xv,~D,~dv),~blocks_size(~X,~D),~blocks_sample(~X,~xv,~D,~dv),count,first,read,room)

def blocks_ranges(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+read: BlocksRead<X,D>,+size: U32,ranges: List<&2,Grouped.Range>,
  bounded: Members.All(Grouped.Range,{range => GroupLaws.RangeBounded(Groups.row_count(blocks_geometry(~X,~D,read)),Groups.column_count(blocks_geometry(~X,~D,read)),
    U32.to_nat(size),range) : Grouped.Range -> Data},ranges)) ->
  Members.All(Grouped.Range,{range => RangeSamples(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),~X,~xv,~D,~dv,
    blocks_lane(~X,~D,read),whole_blocks_lane(~X,~D,read),range) : Grouped.Range -> Data},ranges):
  match ranges:
    case Nil{}: Unit{}
    case +range <> +rest:
      KernelGemm.Both{+head,+tail} = bounded
      KernelGemm.Both{blocks_range(~X,~xv,~D,~dv,read,size,range,head),blocks_ranges(~X,~xv,~D,~dv,read,size,rest,tail)}

# The lanes of the images of a part of the outputs from image on are the
# lanes of the images of the batch at the matching output.
def blocks_images(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,remaining: Nat,+image: Nat,+whole: Images.Batch<X,xv,D,dv>,+outputs: U32,
  +positions: U32,+groups: U32,+rows: U32,+first: U32,+domain: Groups.GeometryReady<outputs_geometry(Images.batch_geometry(~X,~xv,~D,~dv,whole),outputs)>,+x_high: Nat,+x_layout: Held.Spans,
  +x_room: {Nat.is_le(Nat.mul(Images.batch_images(~X,~xv,~D,~dv,whole),Sources.input_cells(outputs_geometry(Images.batch_geometry(~X,~xv,~D,~dv,whole),outputs))),x_high) == True{} : Bool},
  +x_bounded: {Nat.is_le(x_high,U32.to_nat(Machine.capacity(Groups.domain_limits(outputs_geometry(Images.batch_geometry(~X,~xv,~D,~dv,whole),outputs),domain)))) == True{} : Bool},
  +d_high: Nat,+d_layout: Held.Spans,
  +d_room: {Nat.is_le(Nat.mul(Images.batch_images(~X,~xv,~D,~dv,whole),Groups.gradient_extent(outputs_geometry(Images.batch_geometry(~X,~xv,~D,~dv,whole),outputs))),d_high) == True{} : Bool},
  +d_bounded: {Nat.is_le(d_high,U32.to_nat(Machine.capacity(Groups.domain_limits(outputs_geometry(Images.batch_geometry(~X,~xv,~D,~dv,whole),outputs),domain)))) == True{} : Bool},
  +tap: U32,+part_output: U32,+whole_output: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Part.total(blocks_axes(positions,groups,rows))),U32.to_nat(cap)) == True{} : Bool},
  +rows_exact: {Ceiling.ceiling(U32.to_nat(outputs),8n) == U32.to_nat(rows) : Nat},
  +groups_exact: {U32.to_nat(groups) == Groups.output_blocks(Images.batch_geometry(~X,~xv,~D,~dv,whole)) : Nat},
  +positions_exact: {U32.to_nat(positions) == Nat.mul(Images.batch_images(~X,~xv,~D,~dv,whole),Nat.mul(Groups.row_count(Images.batch_geometry(~X,~xv,~D,~dv,whole)),
    Groups.column_count(Images.batch_geometry(~X,~xv,~D,~dv,whole)))) : Nat},
  +whole_exact: {U32.to_nat(whole_output) == Nat.add(Nat.mul(U32.to_nat(first),8n),U32.to_nat(part_output)) : Nat},
  +output_inside: {Nat.is_lt(U32.to_nat(part_output),U32.to_nat(outputs)) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(image,remaining),Images.batch_images(~X,~xv,~D,~dv,whole)) == True{} : Bool}) ->
  {Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),remaining,image,
      outputs_batch(~X,~xv,~D,~dv,whole,outputs,blocks_axes(positions,groups,rows),first,domain,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded),tap,part_output) ==
    Images.held_images(~X,~xv,~D,~dv,remaining,image,whole,tap,whole_output) : List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Images.Batch{+input,+gradient,+geometry,+whole_domain,+images,+height,+width,+group,+size,+bounded,+whole_x_high,+whole_x_layout,+whole_x_room,+whole_x_bounded,
        +whole_d_high,+whole_d_layout,+whole_d_room,+whole_d_bounded} = whole
      +shape = {Images.Batch{input,gradient,geometry,whole_domain,images,height,width,group,size,bounded,whole_x_high,whole_x_layout,whole_x_room,whole_x_bounded,
        whole_d_high,whole_d_layout,whole_d_room,whole_d_bounded} : Images.Batch<X,xv,D,dv>}
      +ranges = Grouped.ranges(height,width,group)
      +read = {BlocksRead{input,gradient,geometry,outputs,positions,groups,rows,first,images,image,tap,part_output,whole_output,cap,cap_ok,room,rows_exact,groups_exact,
        positions_exact,whole_exact,output_inside,Order.room_index(image,rest,images,inside)} : BlocksRead<X,D>}
      +part_head = Images.held_image(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),blocks_lane(~X,~D,read),ranges)
      +whole_head = Images.held_image(~X,~xv,~D,~dv,whole_blocks_lane(~X,~D,read),ranges)
      +part_tail = Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),rest,1n+image,
        outputs_batch(~X,~xv,~D,~dv,shape,outputs,blocks_axes(positions,groups,rows),first,domain,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded),tap,part_output)
      +whole_tail = Images.held_images(~X,~xv,~D,~dv,rest,1n+image,shape,tap,whole_output)
      Equal.trans(List<&2,F32>,part_head <> part_tail,whole_head <> part_tail,whole_head <> whole_tail,
        Equal.cong(F32,List<&2,F32>,head => head <> part_tail,part_head,whole_head,
          same_image(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),~X,~xv,~D,~dv,
            blocks_lane(~X,~D,read),whole_blocks_lane(~X,~D,read),ranges,blocks_ranges(~X,~xv,~D,~dv,read,size,ranges,bounded))),
        Equal.cong(List<&2,F32>,List<&2,F32>,tail => whole_head <> tail,part_tail,whole_tail,
          blocks_images(~X,~xv,~D,~dv,rest,1n+image,shape,outputs,positions,groups,rows,first,domain,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded,
            tap,part_output,whole_output,cap,cap_ok,room,rows_exact,groups_exact,positions_exact,whole_exact,output_inside,Order.advance_room(image,rest,images,inside))))

# ---- The outputs of a part ----

# ---- The lane of a part of the outputs ----

# A part of the outputs keeps the taps, the input cells, the positions and the
# limits of the whole call.
def outputs_taps(+geometry: Kernel.Geometry,+outputs: U32) -> {Packing.geometry_taps(outputs_geometry(geometry,outputs)) == Packing.geometry_taps(geometry) : U32}:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  {==}

def outputs_count(+geometry: Kernel.Geometry,+outputs: U32) -> {Groups.geometry_outputs(outputs_geometry(geometry,outputs)) == outputs : U32}:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  {==}

def outputs_input(+geometry: Kernel.Geometry,+outputs: U32) -> {Sources.input_cells(outputs_geometry(geometry,outputs)) == Sources.input_cells(geometry) : Nat}:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  {==}

def outputs_extent(+geometry: Kernel.Geometry,+outputs: U32) ->
  {Groups.gradient_extent(outputs_geometry(geometry,outputs)) ==
    Nat.mul(Nat.mul(Ceiling.ceiling(U32.to_nat(outputs),8n),Groups.row_count(geometry)),Groups.column_count(geometry)) : Nat}:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  {==}

def outputs_limits(+geometry: Kernel.Geometry,+domain: Groups.GeometryReady<geometry>,+outputs: U32,+positive: {Nat.is_lt(0n,U32.to_nat(outputs)) == True{} : Bool},
  +fewer: {Nat.is_le(U32.to_nat(outputs),Groups.output_count(geometry)) == True{} : Bool},
  +blocks_fewer: {Nat.is_le(Ceiling.ceiling(U32.to_nat(outputs),8n),Groups.output_blocks(geometry)) == True{} : Bool}) ->
  {Groups.domain_limits(outputs_geometry(geometry,outputs),outputs_ready(geometry,domain,outputs,positive,fewer,blocks_fewer)) == Groups.domain_limits(geometry,domain) :
    Machine.Limits}:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Groups.GeometryReady{limits,columns_ok,kernel,stride,square,taps_ok,outputs_ok,count,aligned,taps_exact,pixels,gradients} = domain
  {==}

# The geometry of a part of the outputs of a work.
def part_geometry(+work: Parameter.Work,+outputs: U32) -> {Kernel.geometry(Kernel.part_work(work,outputs)) == outputs_geometry(Kernel.geometry(work),outputs) : Kernel.Geometry}:
  Parameter.Work{n,+c,o,+h,+w,+k,+s,+p,+rows,+columns,g} = work
  Equal.cong(U32,Kernel.Geometry,value => Kernel.Geometry{U32.from_nat(c),value,U32.from_nat(h),U32.from_nat(w),U32.from_nat(k),U32.from_nat(s),U32.from_nat(p),
      U32.from_nat(rows),U32.from_nat(columns),U32.from_nat(Nat.mul(c,Nat.mul(k,k)))},
    U32.from_nat(U32.to_nat(outputs)),outputs,Arithmetic.native_roundtrip(outputs))

def context_geometry(+context: Kernel.BatchContext) -> Kernel.Geometry:
  Kernel.BatchContext{+geometry,rows,columns,group} = context
  geometry

# The sums of the images of a context over a part of its outputs.
def context_values(+outputs: U32,+images: Nat,+context: Kernel.BatchContext,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) ->
  Kernel.Held() & Array<Tile.Tile8>:
  Kernel.BatchContext{+geometry,+rows,+columns,+group} = context
  Kernel.reduced(images,Kernel.cells(outputs_geometry(geometry,outputs)),Kernel.images(images,0,outputs_geometry(geometry,outputs),rows,columns,group,input,gradient))

# The serial kernel over a part of the outputs of a work sums the images of its
# context over those outputs.
def part_values_context(+work: Parameter.Work,+outputs: U32,-input: Move.Holding<Tile.Tile8>,-gradient: Move.Holding<Tile.Tile8>) ->
  {Kernel.weight_values(Kernel.part_work(work,outputs),input,gradient) == context_values(outputs,work_images(work),Kernel.batch_context(work),input,gradient) :
    Kernel.Held() & Array<Tile.Tile8>}:
  Parameter.Work{+n,+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+g} = work
  +shape = {Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} : Parameter.Work}
  %Equal.sym(Kernel.Geometry,Kernel.geometry(Kernel.part_work(shape,outputs)),outputs_geometry(Kernel.geometry(shape),outputs),part_geometry(shape,outputs)) :
    {Kernel.reduced(n,Kernel.cells(_),Kernel.images(n,0,_,rows,columns,g,input,gradient)) == context_values(outputs,n,Kernel.batch_context(shape),input,gradient) :
      Kernel.Held() & Array<Tile.Tile8>}
  {==}

# The word of a position of rows of width elements from row first.
def position_word(+first: U32,+width: U32) -> {U32.from_nat(Nat.mul(U32.to_nat(first),U32.to_nat(width))) == (first * width : U32) : U32}:
  Equal.trans(U32,U32.from_nat(Nat.mul(U32.to_nat(first),U32.to_nat(width))),(U32.from_nat(U32.to_nat(first)) * width : U32),(first * width : U32),
    Equal.sym(U32,(U32.from_nat(U32.to_nat(first)) * width : U32),U32.from_nat(Nat.mul(U32.to_nat(first),U32.to_nat(width))),Sources.product_word(U32.to_nat(first),width)),
    Equal.cong(U32,U32,value => (value * width : U32),U32.from_nat(U32.to_nat(first)),first,Arithmetic.native_roundtrip(first)))

def split_row(+first: Nat,+width: Nat,+local: Nat,+block: Nat,+tap: Nat,+rows: Nat,+split: DivisionOrder.RowSplit<first,width,local,block,tap,rows>) -> Nat:
  DivisionOrder.RowSplit{+row,at,block_exact,below} = split
  row

def split_at(+first: Nat,+width: Nat,+local: Nat,+block: Nat,+tap: Nat,+rows: Nat,+split: DivisionOrder.RowSplit<first,width,local,block,tap,rows>) ->
  {local == Nat.add(Nat.mul(split_row(first,width,local,block,tap,rows,split),width),tap) : Nat}:
  DivisionOrder.RowSplit{row,+at,block_exact,below} = split
  at

def split_block(+first: Nat,+width: Nat,+local: Nat,+block: Nat,+tap: Nat,+rows: Nat,+split: DivisionOrder.RowSplit<first,width,local,block,tap,rows>) ->
  {block == Nat.add(first,split_row(first,width,local,block,tap,rows,split)) : Nat}:
  DivisionOrder.RowSplit{row,at,+block_exact,below} = split
  block_exact

def split_below(+first: Nat,+width: Nat,+local: Nat,+block: Nat,+tap: Nat,+rows: Nat,+split: DivisionOrder.RowSplit<first,width,local,block,tap,rows>) ->
  {Nat.is_lt(split_row(first,width,local,block,tap,rows,split),rows) == True{} : Bool}:
  DivisionOrder.RowSplit{row,at,block_exact,+below} = split
  below

def outputs_positive(+first: Nat,+rows: Nat,+total: Nat,+outputs: Nat,+facts: OutputGroups.PartOutputs<first,rows,total,outputs>) -> {Nat.is_lt(0n,outputs) == True{} : Bool}:
  OutputGroups.PartOutputs{blocks,+positive,end,start,last} = facts
  positive

# The lane a part of the outputs gives back is the lane its batch reads, read
# out of the sums of its images.
def finished_lane(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+whole: Images.Batch<X,xv,D,dv>,
  +part: Images.Batch<Held.Shifted<X>,Held.shifted_value(~Tile.Tile8,~X,~xv),Gather.Parted<D>,Gather.parted_value(~Tile.Tile8,~D,~dv)>,+tap: U32,+part_output: U32,+output: U32,
  -result: Kernel.Held() & Array<Tile.Tile8>,
  done: Images.ReducedDone<Held.Shifted<X>,Held.shifted_value(~Tile.Tile8,~X,~xv),Gather.Parted<D>,Gather.parted_value(~Tile.Tile8,~D,~dv),part,tap,part_output,result>,
  +element: Tile.Tile8,
  +element_exact: {element == Refinement.tile_at(Tree.reflect(Tile.Tile8,Kernel.part_values(result)),
    Nat.add(Nat.mul(Nat.div(U32.to_nat(part_output),8n),U32.to_nat(Packing.geometry_taps(Images.batch_geometry(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),
      ~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),part)))),U32.to_nat(tap))) : Tile.Tile8},
  +lane_exact: {Nat.mod(U32.to_nat(part_output),8n) == Nat.mod(U32.to_nat(output),8n) : Nat},
  +images_exact: {Images.batch_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),part) ==
    Images.batch_images(~X,~xv,~D,~dv,whole) : Nat},
  +held_exact: {Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),
      Images.batch_images(~X,~xv,~D,~dv,whole),0n,part,tap,part_output) ==
    Images.held_images(~X,~xv,~D,~dv,Images.batch_images(~X,~xv,~D,~dv,whole),0n,whole,tap,output) : List<&2,F32>}) ->
  {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),element) ==
    Images.lane_levels(Images.batch_images(~X,~xv,~D,~dv,whole),Images.held_images(~X,~xv,~D,~dv,Images.batch_images(~X,~xv,~D,~dv,whole),0n,whole,tap,output)) : F32}:
  Images.ReducedDone{+sums,+values,+x_covered,+d_covered,+lane} = done
  +images = Images.batch_images(~X,~xv,~D,~dv,whole)
  +part_images = Images.batch_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),part)
  +index = Nat.add(Nat.mul(Nat.div(U32.to_nat(part_output),8n),U32.to_nat(Packing.geometry_taps(Images.batch_geometry(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),
    ~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),part)))),U32.to_nat(tap))
  %held_exact : {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),element) == Images.lane_levels(images,_) : F32}
  %images_exact : {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),element) ==
    Images.lane_levels(_,Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),_,0n,part,tap,part_output)) : F32}
  %lane_exact : {TileLaws.select_lane(_,element) ==
    Images.lane_levels(part_images,Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),part_images,0n,part,tap,
      part_output)) : F32}
  %Equal.sym(Tile.Tile8,element,Refinement.tile_at(Tree.reflect(Tile.Tile8,Kernel.part_values(result)),index),element_exact) :
    {TileLaws.select_lane(Nat.mod(U32.to_nat(part_output),8n),_) ==
      Images.lane_levels(part_images,Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),part_images,0n,part,
        tap,part_output)) : F32}
  %Equal.sym(Kernel.Held() & Array<Tile.Tile8>,result,(Pair.fst(Kernel.Held(),Array<Tile.Tile8>,result),Tree.pack(Tile.Tile8,sums)),values) :
    {TileLaws.select_lane(Nat.mod(U32.to_nat(part_output),8n),Refinement.tile_at(Tree.reflect(Tile.Tile8,Kernel.part_values(_)),index)) ==
      Images.lane_levels(part_images,Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),part_images,0n,part,
        tap,part_output)) : F32}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,sums)),sums,StorageLaws.reflect_packed(Tile.Tile8,sums)) :
    {TileLaws.select_lane(Nat.mod(U32.to_nat(part_output),8n),Refinement.tile_at(_,index)) ==
      Images.lane_levels(part_images,Images.held_images(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),part_images,0n,part,
        tap,part_output)) : F32}
  lane

# ---- The lanes of the weight kernel in parts of the outputs ----

# What a lane of the parts of the outputs reads: the batch and the work, what
# they share, and the lane.
type OutputsScope<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8> is Data:
  OutputsScope{batch: Images.Batch<X,xv,D,dv>,work: Parameter.Work,same: WorkBatch<X,xv,D,dv,batch,work>,tap: U32,output: U32,
    tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
    output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool}}

def scope_parts(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+scope: OutputsScope<X,xv,D,dv>) ->
  KernelGemm.Both<Images.Batch<X,xv,D,dv>,Parameter.Work>:
  OutputsScope{+batch,+work,same,tap,output,tap_inside,output_inside} = scope
  KernelGemm.Both{batch,work}

def scope_work(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+scope: OutputsScope<X,xv,D,dv>) -> Parameter.Work:
  OutputsScope{batch,+work,same,tap,output,tap_inside,output_inside} = scope
  work

def scope_limit(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+scope: OutputsScope<X,xv,D,dv>) -> Nat:
  U32.to_nat(Kernel.blocks(scope_work(~X,~xv,~D,~dv,scope)))

def scope_fill(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+scope: OutputsScope<X,xv,D,dv>) -> Tile.Tile8:
  Tile.filled(0.0)

def work_taps(+work: Parameter.Work) -> Nat:
  U32.to_nat(Kernel.taps(work))

# The cell and lane of the result a scope reads.
def lane_target(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+scope: OutputsScope<X,xv,D,dv>) -> Nat:
  OutputsScope{batch,+work,same,+tap,+output,tap_inside,output_inside} = scope
  Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),work_taps(work)),U32.to_nat(tap))

# The lane of a cell is the adjacent tree of the lanes of the images.
def LaneHolds(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+scope: OutputsScope<X,xv,D,dv>,+element: Tile.Tile8) -> Data:
  OutputsScope{+batch,work,same,+tap,+output,tap_inside,output_inside} = scope
  {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),element) ==
    Images.lane_levels(Images.batch_images(~X,~xv,~D,~dv,batch),Images.held_images(~X,~xv,~D,~dv,Images.batch_images(~X,~xv,~D,~dv,batch),0n,batch,tap,output)) : F32}

def lane_holds(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+scope: OutputsScope<X,xv,D,dv>,+index: Nat,+element: Tile.Tile8) -> Data:
  Logic.When(Nat.is_eq(index,lane_target(~X,~xv,~D,~dv,scope)),LaneHolds(~X,~xv,~D,~dv,scope,element))

# The cells a part of the outputs gives back, as a tree, and as the tiles of
# its rows.
def part_value(work: Parameter.Work,+part: CopiedPart,+first: U32,+rows: U32) -> Tree.Tree<Tile.Tile8>:
  CopiedPart{+input,+x_count,+gradient,+d_count} = part
  +work = work
  Tree.reflect(Tile.Tile8,Kernel.part_values(Kernel.weight_values(Kernel.part_work(work,Kernel.part_outputs(work,first,rows)),
    Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,input)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,gradient)))))

def value_tiles(taps: U32,+value: Tree.Tree<Tile.Tile8>,+first: U32,+rows: U32) -> List<&2,Tile.Tile8>:
  Refinement.tiles(Nat.mul(U32.to_nat(rows),U32.to_nat(taps)),0n,value)

def part_tiles(work: Parameter.Work,+part: CopiedPart,+first: U32,+rows: U32) -> List<&2,Tile.Tile8>:
  +work = work
  value_tiles(Kernel.taps(work),part_value(work,part,first,rows),first,rows)

# A lane of a part of the outputs at the cell of the scope.
def matched_lane(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,+work: Parameter.Work,
  +same: WorkBatch<X,xv,D,dv,batch,work>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +x_model: Buffer.Model<Tile.Tile8>,+x_count: U32,+d_model: Buffer.Model<Tile.Tile8>,+d_count: U32,+first: U32,+rows: U32,
  +fact: CopiedBlocks<X,xv,D,dv,batch,work,first,rows,x_model,x_count,d_model,d_count>,
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(Kernel.blocks(work))) == True{} : Bool},
  +local: Nat,+inside: {Nat.is_lt(local,Nat.mul(U32.to_nat(rows),work_taps(work))) == True{} : Bool},
  +same_index: {Nat.add(Nat.mul(U32.to_nat(first),work_taps(work)),local) == Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),work_taps(work)),U32.to_nat(tap)) : Nat}) ->
  LaneHolds(~X,~xv,~D,~dv,OutputsScope{batch,work,same,tap,output,tap_inside,output_inside},
    ListRead.at(~Tile.Tile8,Tile.filled(0.0),local,part_tiles(work,CopiedPart{x_model,x_count,d_model,d_count},first,rows))):
  Images.Batch{+input,+gradient,+geometry,+domain,+images,+height,+width,+group,+size,+bounded,+x_high,+x_layout,+x_whole,+x_bounded,+d_high,+d_layout,+d_whole,+d_bounded} = batch
  Parameter.Work{+n,+c,+o,+h,+w,+k,+s,+p,+out_rows,+out_columns,+g} = work
  WorkBatch{+context_same,+images_same,+cells_same,+x_word,+d_word,+images_room,+positions_exact,+groups_exact,+input_exact,+blocks_exact} = same
  CopiedBlocks{+x_covered,+d_covered,+x_exact,+d_exact} = fact
  +shape = {Images.Batch{input,gradient,geometry,domain,images,height,width,group,size,bounded,x_high,x_layout,x_whole,x_bounded,d_high,d_layout,d_whole,d_bounded} :
    Images.Batch<X,xv,D,dv>}
  +shape_work = {Parameter.Work{n,c,o,h,w,k,s,p,out_rows,out_columns,g} : Parameter.Work}
  +limits = Groups.domain_limits(geometry,domain)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +f = U32.to_nat(first)
  +r = U32.to_nat(rows)
  +t = U32.to_nat(tap)
  +whole_output = U32.to_nat(output)
  +taps = work_taps(shape_work)
  +total = Groups.output_count(geometry)
  +r_g = Groups.row_count(geometry)
  +c_g = Groups.column_count(geometry)
  +geometry_exact = Equal.cong(Kernel.BatchContext,Kernel.Geometry,context => context_geometry(context),Kernel.batch_context(shape_work),
    Kernel.BatchContext{geometry,height,width,group},context_same)
  +taps_exact = {%geometry_exact : {U32.to_nat(Packing.geometry_taps(Kernel.geometry(shape_work))) == U32.to_nat(Packing.geometry_taps(_)) : Nat} {==} :
    {taps == U32.to_nat(Packing.geometry_taps(geometry)) : Nat}}
  +total_exact = {%geometry_exact : {U32.to_nat(Groups.geometry_outputs(Kernel.geometry(shape_work))) == U32.to_nat(Groups.geometry_outputs(_)) : Nat} {==} :
    {U32.to_nat(Kernel.outputs(shape_work)) == total : Nat}}
  +tap_lt = {%Equal.sym(Nat,taps,U32.to_nat(Packing.geometry_taps(geometry)),taps_exact) : {Nat.is_lt(t,_) == True{} : Bool}
    %Comparison.lt(tap,Packing.geometry_taps(geometry)) : {_ == True{} : Bool} tap_inside : {Nat.is_lt(t,taps) == True{} : Bool}}
  +output_lt = {%Comparison.lt(output,Groups.geometry_outputs(geometry)) : {_ == True{} : Bool} output_inside : {Nat.is_lt(whole_output,total) == True{} : Bool}}
  +o8 = Nat.div(whole_output,8n)
  +o_mod = Nat.mod(whole_output,8n)
  +split = DivisionOrder.row_split(f,taps,local,o8,t,r,tap_lt,inside,same_index)
  +j = split_row(f,taps,local,o8,t,r,split)
  +local_at = split_at(f,taps,local,o8,t,r,split)
  +block_at = split_block(f,taps,local,o8,t,r,split)
  +j_below = split_below(f,taps,local,o8,t,r,split)
  +rows_positive = Order.le_lt_trans(0n,j,r,Order.zero_le(j),j_below)
  +within = {%blocks_exact : {Nat.is_le(Nat.add(f,r),_) == True{} : Bool} room : {Nat.is_le(Nat.add(f,r),Ceiling.ceiling(total,8n)) == True{} : Bool}}
  +facts = OutputGroups.outputs_split(f,r,total,rows_positive,within)
  +parts = Kernel.part_outputs(shape_work,first,rows)
  +f8 = Nat.mul(f,8n)
  +parts_nat = Nat.min(Nat.mul(r,8n),Nat.sub(total,f8))
  +total_room = Groups.outputs_room(geometry,domain)
  +parts_exact = OutputGroups.part_outputs_exact(Kernel.outputs(shape_work),first,rows,limits,total,total_exact,total_room,rows_positive,facts)
  +whole_split = DivisionOrder.quotient_exact(whole_output,8n,{==})
  +below_whole = {%Equal.sym(Nat,whole_output,Nat.add(Nat.mul(o8,8n),o_mod),whole_split) : {Nat.is_le(Nat.mul(o8,8n),_) == True{} : Bool}
    Order.le_add(Nat.mul(o8,8n),o_mod) : {Nat.is_le(Nat.mul(o8,8n),whole_output) == True{} : Bool}}
  +ordered = Order.transitive(f8,Nat.mul(o8,8n),whole_output,
    {%Equal.sym(Nat,o8,Nat.add(f,j),block_at) : {Nat.is_le(f8,Nat.mul(_,8n)) == True{} : Bool} RowMajor.mul_le(f,Nat.add(f,j),8n,Order.le_add(f,j)) :
      {Nat.is_le(f8,Nat.mul(o8,8n)) == True{} : Bool}},
    below_whole)
  +output_room = Order.transitive(whole_output,total,cap,Bounds.strict_le(whole_output,total,output_lt),total_room)
  +first8_word = word_product(first,8,capacity,Order.transitive(f8,whole_output,cap,ordered,output_room))
  +part_output = (output - first * 8 : U32)
  +part_word = Arithmetic.subtract(output,(first * 8 : U32),capacity,Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat((first * 8 : U32)),f8,first8_word) : {Nat.is_le(_,whole_output) == True{} : Bool} ordered,
    Order.transitive(Nat.sub(whole_output,U32.to_nat((first * 8 : U32))),whole_output,cap,Order.sub_le(whole_output,U32.to_nat((first * 8 : U32))),output_room))
  +part_nat = {%first8_word : {U32.to_nat(part_output) == Nat.sub(whole_output,_) : Nat} part_word : {U32.to_nat(part_output) == Nat.sub(whole_output,f8) : Nat}}
  +q = U32.to_nat(part_output)
  +whole_exact = Equal.sym(Nat,Nat.add(f8,q),whole_output,
    %Equal.sym(Nat,q,Nat.sub(whole_output,f8),part_nat) : {Nat.add(f8,_) == whole_output : Nat} Order.restore_shift(f8,whole_output,ordered))
  +q_value = Nat.add(Nat.mul(j,8n),o_mod)
  +whole_parts = Equal.trans(Nat,whole_output,Nat.add(Nat.mul(Nat.add(f,j),8n),o_mod),Nat.add(f8,q_value),
    %block_at : {whole_output == Nat.add(Nat.mul(_,8n),o_mod) : Nat} whole_split,
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(f,j),8n),o_mod),Nat.add(Nat.add(f8,Nat.mul(j,8n)),o_mod),Nat.add(f8,q_value),
      Equal.cong(Nat,Nat,value => Nat.add(value,o_mod),Nat.mul(Nat.add(f,j),8n),Nat.add(f8,Nat.mul(j,8n)),RowMajor.distrib(f,j,8n)),
      Addition.associative(f8,Nat.mul(j,8n),o_mod)))
  +q_exact = Order.cancel_sum(f8,q,q_value,Equal.trans(Nat,Nat.add(f8,q),whole_output,Nat.add(f8,q_value),Equal.sym(Nat,whole_output,Nat.add(f8,q),whole_exact),whole_parts))
  +o_mod_small = DivisionOrder.remainder_small(whole_output,8n,{==})
  +q_divmod = NatDivision.unique(q,8n,j,o_mod,q_exact,{==},o_mod_small)
  +q_block = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(q,8n),(j,o_mod),q_divmod)
  +q_lane = Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(q,8n),(j,o_mod),q_divmod)
  +q_rows = {%Equal.sym(Nat,q,q_value,q_exact) : {Nat.is_lt(_,Nat.mul(r,8n)) == True{} : Bool}
    Bounds.nat_trans(q_value,Nat.add(Nat.mul(j,8n),8n),Nat.mul(r,8n),Order.offset_lt(Nat.mul(j,8n),o_mod,8n,o_mod_small),
      %Order.commutative(8n,Nat.mul(j,8n)) : {Nat.is_le(_,Nat.mul(r,8n)) == True{} : Bool} RowMajor.mul_le(1n+j,r,8n,Order.strict_follow(j,r,j_below))) :
    {Nat.is_lt(q,Nat.mul(r,8n)) == True{} : Bool}}
  +start = Order.transitive(f8,whole_output,total,ordered,Bounds.strict_le(whole_output,total,output_lt))
  +q_rest = Order.cancel_offset_lt(f8,q,Nat.sub(total,f8),
    %Equal.sym(Nat,Nat.add(f8,Nat.sub(total,f8)),total,Order.restore_shift(f8,total,start)) : {Nat.is_lt(Nat.add(f8,q),_) == True{} : Bool}
    %whole_exact : {Nat.is_lt(_,total) == True{} : Bool} output_lt)
  +q_inside = {%Equal.sym(Nat,U32.to_nat(parts),parts_nat,parts_exact) : {Nat.is_lt(q,_) == True{} : Bool}
    NatExtrema.minimum_strict_lower(Nat.mul(r,8n),Nat.sub(total,f8),q,q_rows,q_rest) : {Nat.is_lt(q,U32.to_nat(parts)) == True{} : Bool}}
  +parts_positive = {%Equal.sym(Nat,U32.to_nat(parts),parts_nat,parts_exact) : {Nat.is_lt(0n,_) == True{} : Bool} outputs_positive(f,r,total,parts_nat,facts) :
    {Nat.is_lt(0n,U32.to_nat(parts)) == True{} : Bool}}
  +parts_fewer = {%Equal.sym(Nat,U32.to_nat(parts),parts_nat,parts_exact) : {Nat.is_le(_,total) == True{} : Bool}
    Order.transitive(parts_nat,Nat.sub(total,f8),total,NatExtrema.minimum_right(Nat.mul(r,8n),Nat.sub(total,f8)),Order.sub_le(total,f8)) :
    {Nat.is_le(U32.to_nat(parts),total) == True{} : Bool}}
  +parts_blocks = {%Equal.sym(Nat,U32.to_nat(parts),parts_nat,parts_exact) : {Ceiling.ceiling(_,8n) == r : Nat} OutputGroups.clipped_groups(f,r,total,parts_nat,facts) :
    {Ceiling.ceiling(U32.to_nat(parts),8n) == r : Nat}}
  +blocks_fewer = {%Equal.sym(Nat,Ceiling.ceiling(U32.to_nat(parts),8n),r,parts_blocks) : {Nat.is_le(_,Ceiling.ceiling(total,8n)) == True{} : Bool}
    Order.drop_left_le(f,r,Ceiling.ceiling(total,8n),within) : {Nat.is_le(Ceiling.ceiling(U32.to_nat(parts),8n),Groups.output_blocks(geometry)) == True{} : Bool}}
  +part_domain = outputs_ready(geometry,domain,parts,parts_positive,parts_fewer,blocks_fewer)
  +limits_same = outputs_limits(geometry,domain,parts,parts_positive,parts_fewer,blocks_fewer)
  +positions = work_positions(shape_work)
  +groups = work_groups(shape_work)
  +positions_whole = positions_room(images,r_g,c_g,Groups.output_blocks(geometry),positions,groups,d_high,positions_exact,groups_exact,d_whole)
  +fits = {%Equal.sym(Nat,U32.to_nat(groups),Groups.output_blocks(geometry),groups_exact) : {Nat.is_le(Nat.add(f,r),_) == True{} : Bool}
    within : {Nat.is_le(Nat.add(f,r),U32.to_nat(groups)) == True{} : Bool}}
  +d_room_cap = blocks_count_room(positions,groups,first,rows,d_high,cap,fits,positions_whole,d_bounded)
  +blocks_room = blocks_total_room(positions,groups,rows,limits,d_room_cap)
  +x_room = {%Equal.sym(Nat,Sources.input_cells(outputs_geometry(geometry,parts)),Sources.input_cells(geometry),outputs_input(geometry,parts)) :
      {Nat.is_le(Nat.mul(images,_),U32.to_nat(x_count)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(x_count),Nat.mul(images,Sources.input_cells(geometry)),x_exact) : {Nat.is_le(Nat.mul(images,Sources.input_cells(geometry)),_) == True{} : Bool}
    Order.reflexive(Nat.mul(images,Sources.input_cells(geometry))) :
    {Nat.is_le(Nat.mul(images,Sources.input_cells(outputs_geometry(geometry,parts))),U32.to_nat(x_count)) == True{} : Bool}}
  +x_bounded_part = {%Equal.sym(Machine.Limits,Groups.domain_limits(outputs_geometry(geometry,parts),part_domain),limits,limits_same) :
      {Nat.is_le(U32.to_nat(x_count),U32.to_nat(Machine.capacity(_))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(x_count),Nat.mul(images,Sources.input_cells(geometry)),x_exact) : {Nat.is_le(_,cap) == True{} : Bool}
    Order.transitive(Nat.mul(images,Sources.input_cells(geometry)),x_high,cap,x_whole,x_bounded) :
    {Nat.is_le(U32.to_nat(x_count),U32.to_nat(Machine.capacity(Groups.domain_limits(outputs_geometry(geometry,parts),part_domain)))) == True{} : Bool}}
  +d_room_part = {%Equal.sym(Nat,Groups.gradient_extent(outputs_geometry(geometry,parts)),Nat.mul(Nat.mul(Ceiling.ceiling(U32.to_nat(parts),8n),r_g),c_g),
        outputs_extent(geometry,parts)) :
      {Nat.is_le(Nat.mul(images,_),U32.to_nat(d_count)) == True{} : Bool}
    %Equal.sym(Nat,Ceiling.ceiling(U32.to_nat(parts),8n),r,parts_blocks) : {Nat.is_le(Nat.mul(images,Nat.mul(Nat.mul(_,r_g),c_g)),U32.to_nat(d_count)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(d_count),Nat.mul(U32.to_nat(positions),r),d_exact) : {Nat.is_le(Nat.mul(images,Nat.mul(Nat.mul(r,r_g),c_g)),_) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(positions),Nat.mul(images,Nat.mul(r_g,c_g)),positions_exact) : {Nat.is_le(Nat.mul(images,Nat.mul(Nat.mul(r,r_g),c_g)),Nat.mul(_,r)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.mul(images,Nat.mul(r_g,c_g)),r),Nat.mul(images,Nat.mul(Nat.mul(r,r_g),c_g)),blocks_product(images,r_g,c_g,r)) :
      {Nat.is_le(Nat.mul(images,Nat.mul(Nat.mul(r,r_g),c_g)),_) == True{} : Bool}
    Order.reflexive(Nat.mul(images,Nat.mul(Nat.mul(r,r_g),c_g))) :
    {Nat.is_le(Nat.mul(images,Groups.gradient_extent(outputs_geometry(geometry,parts))),U32.to_nat(d_count)) == True{} : Bool}}
  +d_bounded_part = {%Equal.sym(Machine.Limits,Groups.domain_limits(outputs_geometry(geometry,parts),part_domain),limits,limits_same) :
      {Nat.is_le(U32.to_nat(d_count),U32.to_nat(Machine.capacity(_))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(d_count),Nat.mul(U32.to_nat(positions),r),d_exact) : {Nat.is_le(_,cap) == True{} : Bool}
    d_room_cap : {Nat.is_le(U32.to_nat(d_count),U32.to_nat(Machine.capacity(Groups.domain_limits(outputs_geometry(geometry,parts),part_domain)))) == True{} : Bool}}
  +x_layout_part = Held.layout_spans(Pieces.Range{0,x_count},Held.Ended{})
  +d_layout_part = Held.layout_spans(Pieces.Range{0,d_count},Held.Ended{})
  +part_batch = outputs_batch(~X,~xv,~D,~dv,shape,parts,blocks_axes(positions,groups,rows),first,part_domain,U32.to_nat(x_count),x_layout_part,x_room,x_bounded_part,
    U32.to_nat(d_count),d_layout_part,d_room_part,d_bounded_part)
  +part_tap = {%Equal.sym(U32,Packing.geometry_taps(outputs_geometry(geometry,parts)),Packing.geometry_taps(geometry),outputs_taps(geometry,parts)) :
      {U32.is_lt(tap,_) == True{} : Bool}
    tap_inside : {U32.is_lt(tap,Packing.geometry_taps(outputs_geometry(geometry,parts))) == True{} : Bool}}
  +part_inside = {%Equal.sym(U32,Groups.geometry_outputs(outputs_geometry(geometry,parts)),parts,outputs_count(geometry,parts)) : {U32.is_lt(part_output,_) == True{} : Bool}
    %Equal.sym(Bool,U32.is_lt(part_output,parts),Nat.is_lt(q,U32.to_nat(parts)),Comparison.lt(part_output,parts)) : {_ == True{} : Bool}
    q_inside : {U32.is_lt(part_output,Groups.geometry_outputs(outputs_geometry(geometry,parts))) == True{} : Bool}}
  +held_exact = blocks_images(~X,~xv,~D,~dv,images,0n,shape,parts,positions,groups,rows,first,part_domain,U32.to_nat(x_count),x_layout_part,x_room,x_bounded_part,
    U32.to_nat(d_count),d_layout_part,d_room_part,d_bounded_part,tap,part_output,output,Machine.half_word(limits),Limits.half_word_fits(limits),blocks_room,parts_blocks,
    groups_exact,positions_exact,whole_exact,q_inside,Order.reflexive(images))
  +value = part_value(shape_work,CopiedPart{x_model,x_count,d_model,d_count},first,rows)
  +value_exact = Equal.trans(Kernel.Held() & Array<Tile.Tile8>,
    Kernel.weight_values(Kernel.part_work(shape_work,parts),Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))),
    context_values(parts,n,Kernel.batch_context(shape_work),Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))),
    context_values(parts,images,Kernel.BatchContext{geometry,height,width,group},Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),
      Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))),
    part_values_context(shape_work,parts,Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))),
    %images_same : {context_values(parts,n,Kernel.batch_context(shape_work),Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))) ==
      context_values(parts,_,Kernel.BatchContext{geometry,height,width,group},Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))) :
      Kernel.Held() & Array<Tile.Tile8>}
    %context_same : {context_values(parts,n,Kernel.batch_context(shape_work),Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))) ==
      context_values(parts,n,_,Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))) : Kernel.Held() & Array<Tile.Tile8>}
    {==})
  +index = Nat.add(Nat.mul(Nat.div(q,8n),U32.to_nat(Packing.geometry_taps(outputs_geometry(geometry,parts)))),t)
  +index_exact = {%Equal.sym(U32,Packing.geometry_taps(outputs_geometry(geometry,parts)),Packing.geometry_taps(geometry),outputs_taps(geometry,parts)) :
      {local == Nat.add(Nat.mul(Nat.div(q,8n),U32.to_nat(_)),t) : Nat}
    %taps_exact : {local == Nat.add(Nat.mul(Nat.div(q,8n),_),t) : Nat}
    %Equal.sym(Nat,Nat.div(q,8n),j,q_block) : {local == Nat.add(Nat.mul(_,taps),t) : Nat}
    local_at : {local == index : Nat}}
  +element = ListRead.at(~Tile.Tile8,Tile.filled(0.0),local,part_tiles(shape_work,CopiedPart{x_model,x_count,d_model,d_count},first,rows))
  +element_exact = Equal.trans(Tile.Tile8,element,Refinement.tile_at(value,local),
    Refinement.tile_at(Tree.reflect(Tile.Tile8,Kernel.part_values(context_values(parts,images,Kernel.BatchContext{geometry,height,width,group},
      Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))))),index),
    ListRead.enumerate_at(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~Refinement.tile_at,Tile.filled(0.0),Nat.mul(r,taps),local,0n,value,inside),
    %value_exact : {Refinement.tile_at(value,local) == Refinement.tile_at(Tree.reflect(Tile.Tile8,Kernel.part_values(_)),index) : Tile.Tile8}
    %index_exact : {Refinement.tile_at(value,local) == Refinement.tile_at(value,_) : Tile.Tile8}
    {==})
  finished_lane(~X,~xv,~D,~dv,shape,part_batch,tap,part_output,output,
    context_values(parts,images,Kernel.BatchContext{geometry,height,width,group},Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),
      Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))),
    Images.weight_reduced(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),part_batch,tap,part_output,part_tap,part_inside,
      Kernel.images(images,0,outputs_geometry(geometry,parts),height,width,group,Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model))),
      Images.images_done(~Held.Shifted<X>,~Held.shifted_value(~Tile.Tile8,~X,~xv),~Gather.Parted<D>,~Gather.parted_value(~Tile.Tile8,~D,~dv),part_batch,tap,part_output,part_tap,part_inside,
        Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,x_model)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,d_model)),x_covered,d_covered)),
    element,element_exact,q_lane,{==},held_exact)

# A lane of a part of the outputs holds at the cell of the scope.
def part_lane(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,+work: Parameter.Work,
  +same: WorkBatch<X,xv,D,dv,batch,work>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +x_model: Buffer.Model<Tile.Tile8>,+x_count: U32,+d_model: Buffer.Model<Tile.Tile8>,+d_count: U32,+first: U32,+rows: U32,
  +fact: CopiedBlocks<X,xv,D,dv,batch,work,first,rows,x_model,x_count,d_model,d_count>,
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(Kernel.blocks(work))) == True{} : Bool},
  +local: Nat,+inside: {Nat.is_lt(local,Nat.mul(U32.to_nat(rows),work_taps(work))) == True{} : Bool}) ->
  lane_holds(~X,~xv,~D,~dv,OutputsScope{batch,work,same,tap,output,tap_inside,output_inside},Nat.add(Nat.mul(U32.to_nat(first),work_taps(work)),local),
    ListRead.at(~Tile.Tile8,Tile.filled(0.0),local,part_tiles(work,CopiedPart{x_model,x_count,d_model,d_count},first,rows))):
  +scope = {OutputsScope{batch,work,same,tap,output,tap_inside,output_inside} : OutputsScope<X,xv,D,dv>}
  +index = Nat.add(Nat.mul(U32.to_nat(first),work_taps(work)),local)
  Logic.when_enabled(LaneHolds(~X,~xv,~D,~dv,scope,ListRead.at(~Tile.Tile8,Tile.filled(0.0),local,part_tiles(work,CopiedPart{x_model,x_count,d_model,d_count},first,rows))),
    Nat.is_eq(index,lane_target(~X,~xv,~D,~dv,scope)),
    equal => matched_lane(~X,~xv,~D,~dv,batch,work,same,tap,output,tap_inside,output_inside,x_model,x_count,d_model,d_count,first,rows,fact,room,local,inside,
      Order.equal_of_is_eq(index,lane_target(~X,~xv,~D,~dv,scope),equal)))

# Every lane of a part of the outputs holds at the cell of its scope.
def lane_rule(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+scope: OutputsScope<X,xv,D,dv>,+part: CopiedPart,+first: U32,+rows: U32,
  +fact: PiecesModel.viewed(~CopiedPart,~KernelGemm.Both<Images.Batch<X,xv,D,dv>,Parameter.Work>,~blocks_fact(~X,~xv,~D,~dv),~OutputsScope<X,xv,D,dv>,~scope_parts(~X,~xv,~D,~dv),
    scope,part,first,rows),
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),scope_limit(~X,~xv,~D,~dv,scope)) == True{} : Bool},
  +local: Nat,+inside: {Nat.is_lt(local,Nat.mul(U32.to_nat(rows),work_taps(scope_work(~X,~xv,~D,~dv,scope)))) == True{} : Bool}) ->
  lane_holds(~X,~xv,~D,~dv,scope,Nat.add(Nat.mul(U32.to_nat(first),work_taps(scope_work(~X,~xv,~D,~dv,scope))),local),
    ListRead.at(~Tile.Tile8,scope_fill(~X,~xv,~D,~dv,scope),local,part_tiles(scope_work(~X,~xv,~D,~dv,scope),part,first,rows))):
  OutputsScope{+batch,+work,+same,+tap,+output,+tap_inside,+output_inside} = scope
  CopiedPart{+x_model,+x_count,+d_model,+d_count} = part
  part_lane(~X,~xv,~D,~dv,batch,work,same,tap,output,tap_inside,output_inside,x_model,x_count,d_model,d_count,first,rows,fact,room,local,inside)

# ---- The weight kernel in parts of the outputs ----

def tree_array(tree: Tree.Tree<Tile.Tile8>) -> Array<Tile.Tile8>:
  Tree.pack(Tile.Tile8,tree)

# A part of the outputs gives back the pack of the tree of its cells.
def part_valued(+work: Parameter.Work,+part: CopiedPart,+first: U32,+rows: U32) ->
  {Kernel.part(work,first,rows,copied_owners(part),Unit{}) == Partition.Leaf{tree_array(part_value(work,part,first,rows)),first,rows,Unit{}} :
    Partition.Tree<&1,Array<Tile.Tile8>,Unit>}:
  CopiedPart{+input,+x_count,+gradient,+d_count} = part
  %Equal.sym(Array<Tile.Tile8>,
      Tree.pack(Tile.Tile8,Tree.reflect(Tile.Tile8,Kernel.part_values(Kernel.weight_values(Kernel.part_work(work,Kernel.part_outputs(work,first,rows)),
        Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,input)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,gradient)))))),
      Kernel.part_values(Kernel.weight_values(Kernel.part_work(work,Kernel.part_outputs(work,first,rows)),
        Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,input)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,gradient)))),
      StorageLaws.pack_reflected(Tile.Tile8,Kernel.part_values(Kernel.weight_values(Kernel.part_work(work,Kernel.part_outputs(work,first,rows)),
        Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,input)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,gradient)))))) :
    {Partition.Leaf{Kernel.part_values(Kernel.weight_values(Kernel.part_work(work,Kernel.part_outputs(work,first,rows)),
        Kernel.alone(x_count,Buffer.pack(~Tile.Tile8,input)),Kernel.alone(d_count,Buffer.pack(~Tile.Tile8,gradient)))),first,rows,Unit{}} ==
      Partition.Leaf{_,first,rows,Unit{}} : Partition.Tree<&1,Array<Tile.Tile8>,Unit>}
  {==}

def taps_width(taps: U32) -> Nat:
  U32.to_nat(taps)

def work_width(work: Parameter.Work) -> Nat:
  U32.to_nat(Kernel.taps(work))

def work_word(+work: Parameter.Work) -> U32:
  Kernel.taps(work)

# The step of a part of the outputs writes the tiles of its rows from its
# first row.
def blocks_step(taps: U32,+first: U32,+rows: U32,value: Tree.Tree<Tile.Tile8>,content: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  Buffer.Content{tree,length} = content
  +taps = taps
  +value = value
  Buffer.Content{Tree.write_sequence(~Tile.Tile8,value_tiles(taps,value,first,rows),U32.from_nat(Nat.mul(U32.to_nat(first),taps_width(taps))),tree),length}

def blocks_write(+taps: U32,+first: U32,+rows: U32,+value: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+length: Nat) ->
  {blocks_step(taps,first,rows,value,Buffer.Content{tree,length}) ==
    Buffer.Content{Tree.write_sequence(~Tile.Tile8,value_tiles(taps,value,first,rows),U32.from_nat(Nat.mul(U32.to_nat(first),taps_width(taps))),tree),length} :
    Buffer.Content<Tile.Tile8>}:
  {==}

def value_length(+taps: U32,+value: Tree.Tree<Tile.Tile8>,+first: U32,+rows: U32) ->
  {List.length(&2,Tile.Tile8,value_tiles(taps,value,first,rows)) == Nat.mul(U32.to_nat(rows),taps_width(taps)) : Nat}:
  ListRead.enumerate_length(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~Refinement.tile_at,Nat.mul(U32.to_nat(rows),U32.to_nat(taps)),0n,value)

def part_length(+work: Parameter.Work,+part: CopiedPart,+first: U32,+rows: U32) ->
  {List.length(&2,Tile.Tile8,part_tiles(work,part,first,rows)) == Nat.mul(U32.to_nat(rows),work_width(work)) : Nat}:
  value_length(Kernel.taps(work),part_value(work,part,first,rows),first,rows)

def tiles_listed(+work: Parameter.Work,+part: CopiedPart,+first: U32,+rows: U32) ->
  {value_tiles(work_word(work),part_value(work,part,first,rows),first,rows) == part_tiles(work,part,first,rows) : List<&2,Tile.Tile8>}:
  {==}

# A part of the outputs writes the tiles of its rows into the result.
def blocks_folded(+taps: U32,+first: U32,+rows: U32,+value: Tree.Tree<Tile.Tile8>,+kept: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,Kernel.part_cells(taps,first,rows,tree_array(value),Buffer.pack(~Tile.Tile8,kept))) ==
    blocks_step(taps,first,rows,value,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,kept))) : Buffer.Content<Tile.Tile8>}:
  Buffer.Model{+tree,+length,+certificate} = kept
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,kept)),Buffer.content(~Tile.Tile8,kept),Buffer.packed_contents(~Tile.Tile8,kept)) :
    {Buffer.contents(~Tile.Tile8,Kernel.part_cells(taps,first,rows,tree_array(value),Buffer.pack(~Tile.Tile8,kept))) == blocks_step(taps,first,rows,value,_) :
      Buffer.Content<Tile.Tile8>}
  %Equal.sym(U32,U32.from_nat(Nat.mul(U32.to_nat(first),U32.to_nat(taps))),(first * taps : U32),position_word(first,taps)) :
    {Buffer.contents(~Tile.Tile8,Kernel.part_cells(taps,first,rows,tree_array(value),Buffer.pack(~Tile.Tile8,kept))) ==
      Buffer.Content{Tree.write_sequence(~Tile.Tile8,value_tiles(taps,value,first,rows),_,tree),length} : Buffer.Content<Tile.Tile8>}
  Refinement.tiles_written(Nat.mul(U32.to_nat(rows),U32.to_nat(taps)),(first * taps : U32),value,kept)

# The geometry of a work is that of the batch it shares a context with.
def work_geometry(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+work: Parameter.Work,+batch: Images.Batch<X,xv,D,dv>,
  +context_same: {Kernel.batch_context(work) == batch_context_of(~X,~xv,~D,~dv,batch) : Kernel.BatchContext}) ->
  {Kernel.geometry(work) == Images.batch_geometry(~X,~xv,~D,~dv,batch) : Kernel.Geometry}:
  Parameter.Work{+n,+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+g} = work
  Images.Batch{input,gradient,+geometry,domain,images,+height,+width,+group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  Equal.cong(Kernel.BatchContext,Kernel.Geometry,context => context_geometry(context),Kernel.batch_context(Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g}),
    Kernel.BatchContext{geometry,height,width,group},context_same)

def work_taps_exact(+work: Parameter.Work,+geometry: Kernel.Geometry,+exact: {Kernel.geometry(work) == geometry : Kernel.Geometry}) ->
  {work_taps(work) == Groups.tap_count(geometry) : Nat}:
  Parameter.Work{+n,+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+g} = work
  %exact : {work_taps(Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g}) == Groups.tap_count(_) : Nat}
  {==}

# What the parts of a plan of the outputs give back is the weight kernel over
# the batch.
def outputs_assembled(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+depth: Nat,+work: Parameter.Work,+batch: Images.Batch<X,xv,D,dv>,
  +same: WorkBatch<X,xv,D,dv,batch,work>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +coverage: PiecesPlan.Coverage(Pieces.unit_ranges(depth,1,0,Kernel.blocks(work)),0n,U32.to_nat(Kernel.blocks(work))),-result: Owned(),
  done: PartsDone<X,xv,D,dv,CopiedPart,copied_owners,Parameter.Work,blocks_fact(~X,~xv,~D,~dv),batch,work,Pieces.unit_ranges(depth,1,0,Kernel.blocks(work)),result>) ->
  WeightCells<X,xv,D,dv,batch,tap,output,Kernel.parts_run(depth,work,(Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,result),
    Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,result)))>:
  WorkBatch{+context_same,+images_same,+cells_same,+x_word,+d_word,+images_room,+positions_exact,+groups_exact,+input_exact,+blocks_exact} = same
  PartsDone{+x,+d,+state,+holds,+parts,+packed,+planned,+facts} = done
  +plan = Pieces.unit_ranges(depth,1,0,Kernel.blocks(work))
  +limit = U32.to_nat(Kernel.blocks(work))
  +taps = Kernel.taps(work)
  +count = Kernel.cells(Kernel.geometry(work))
  +cells = Images.batch_cells(~X,~xv,~D,~dv,batch)
  +geometry = Images.batch_geometry(~X,~xv,~D,~dv,batch)
  +fill = Tile.filled(0.0)
  +own_plan = PiecesModel.shaped_plan(~CopiedPart,~Each.piece_rows(~CopiedPart),parts)
  +valued = PiecesModel.shaped_valued(~CopiedPart,~Tree.Tree<Tile.Tile8>,~Parameter.Work,~part_value,work,parts)
  +list = Each.shaped_list(~CopiedPart,~Tile.Tile8,~Parameter.Work,~part_tiles,work,parts)
  +covered = {%Equal.sym(Pieces.Ranges,own_plan,plan,planned) : PiecesPlan.Coverage(_,0n,limit) coverage : PiecesPlan.Coverage(own_plan,0n,limit)}
  +valued_covered = {%Equal.sym(Pieces.Ranges,PiecesModel.shaped_plan(~Tree.Tree<Tile.Tile8>,~Each.piece_rows(~Tree.Tree<Tile.Tile8>),valued),own_plan,
        Each.shaped_valued_plan(~CopiedPart,~Tree.Tree<Tile.Tile8>,~Parameter.Work,~part_value,work,parts)) :
      PiecesPlan.Coverage(_,0n,limit)
    covered : PiecesPlan.Coverage(PiecesModel.shaped_plan(~Tree.Tree<Tile.Tile8>,~Each.piece_rows(~Tree.Tree<Tile.Tile8>),valued),0n,limit)}
  +geometry_exact = work_geometry(~X,~xv,~D,~dv,work,batch,context_same)
  +taps_exact = work_taps_exact(work,geometry,geometry_exact)
  +start = {Buffer.Content{Tree.fresh(Tile.Tile8,Tensors.depth_for(count),fill),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}
  +start_equal = Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)))),
    Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))),start,
    Buffer.packed_contents(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))),
    Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))),
      Buffer.contents(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)),start,
      Buffer.remodel_content(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)),Listed.allocated(~Tile.Tile8,fill,count)))
  +folded = Equal.trans(Buffer.Content<Tile.Tile8>,
    Buffer.contents(~Tile.Tile8,Pieces.fold(~Array<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~U32,~Kernel.part_cells,taps,
      PiecesModel.shaped_pack(~Tree.Tree<Tile.Tile8>,~Array<Tile.Tile8>,~tree_array,valued),Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))))),
    PiecesModel.shaped_thread(~Tree.Tree<Tile.Tile8>,~Buffer.Content<Tile.Tile8>,~U32,~blocks_step,taps,valued,
      Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))))),
    Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),fill)),U32.to_nat(count)},
    Each.shaped_folded(~Tree.Tree<Tile.Tile8>,~Array<Tile.Tile8>,~tree_array,~Storage.Buffer<Tile.Tile8>,~U32,~Buffer.Model<Tile.Tile8>,~Buffer.pack(~Tile.Tile8),
      ~Each.buffer_held(~Tile.Tile8),~Buffer.Content<Tile.Tile8>,~Buffer.contents(~Tile.Tile8),~Kernel.part_cells,~blocks_step,~blocks_folded,valued,taps,
      Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))),
    %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)))),start,start_equal) :
      {PiecesModel.shaped_thread(~Tree.Tree<Tile.Tile8>,~Buffer.Content<Tile.Tile8>,~U32,~blocks_step,taps,valued,_) ==
        Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),fill)),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}
    %Each.shaped_list_valued(~CopiedPart,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~U32,~Parameter.Work,~part_value,~value_tiles,~work_word,~part_tiles,~tiles_listed,work,parts) :
      {PiecesModel.shaped_thread(~Tree.Tree<Tile.Tile8>,~Buffer.Content<Tile.Tile8>,~U32,~blocks_step,taps,valued,start) ==
        Buffer.Content{Tree.write_sequence(~Tile.Tile8,_,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),fill)),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}
    Each.shaped_written(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~U32,~blocks_step,~value_tiles,~taps_width,~blocks_write,~value_length,
      taps,valued,0n,limit,valued_covered,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),fill),U32.to_nat(count)))
  +contents = {%cells_same :
      {Buffer.contents(~Tile.Tile8,Pieces.fold(~Array<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~U32,~Kernel.part_cells,taps,
        PiecesModel.shaped_pack(~Tree.Tree<Tile.Tile8>,~Array<Tile.Tile8>,~tree_array,valued),Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))))) ==
        Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(_),fill)),U32.to_nat(_)} : Buffer.Content<Tile.Tile8>}
    folded :
    {Buffer.contents(~Tile.Tile8,Pieces.fold(~Array<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~U32,~Kernel.part_cells,taps,
      PiecesModel.shaped_pack(~Tree.Tree<Tile.Tile8>,~Array<Tile.Tile8>,~tree_array,valued),Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))))) ==
      Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(cells),fill)),U32.to_nat(cells)} : Buffer.Content<Tile.Tile8>}}
  +length = {%blocks_exact : {List.length(&2,Tile.Tile8,list) == Nat.mul(_,Groups.tap_count(geometry)) : Nat}
    %taps_exact : {List.length(&2,Tile.Tile8,list) == Nat.mul(limit,_) : Nat}
    Each.shaped_list_length(~CopiedPart,~Tile.Tile8,~Parameter.Work,~part_tiles,~work_width,~part_length,work,parts,0n,limit,covered) :
    {List.length(&2,Tile.Tile8,list) == Nat.mul(Groups.output_blocks(geometry),Groups.tap_count(geometry)) : Nat}}
  +scope = {OutputsScope{batch,work,same,tap,output,tap_inside,output_inside} : OutputsScope<X,xv,D,dv>}
  +target = lane_target(~X,~xv,~D,~dv,scope)
  +o_lt = {%Comparison.lt(output,Groups.geometry_outputs(geometry)) : {_ == True{} : Bool} output_inside :
    {Nat.is_lt(U32.to_nat(output),Groups.output_count(geometry)) == True{} : Bool}}
  +block_lt = {%Equal.sym(Nat,limit,Groups.output_blocks(geometry),blocks_exact) : {Nat.is_lt(Nat.div(U32.to_nat(output),8n),_) == True{} : Bool}
    DivisionOrder.quotient_below(U32.to_nat(output),Groups.output_blocks(geometry),8n,{==},
      Bounds.nat_trans(U32.to_nat(output),Groups.output_count(geometry),Nat.mul(Groups.output_blocks(geometry),8n),o_lt,Ceiling.covers(Groups.output_count(geometry),8n,{==}))) :
    {Nat.is_lt(Nat.div(U32.to_nat(output),8n),limit) == True{} : Bool}}
  +tap_lt = {%Equal.sym(Nat,work_taps(work),Groups.tap_count(geometry),taps_exact) : {Nat.is_lt(U32.to_nat(tap),_) == True{} : Bool}
    %Comparison.lt(tap,Packing.geometry_taps(geometry)) : {_ == True{} : Bool} tap_inside : {Nat.is_lt(U32.to_nat(tap),work_taps(work)) == True{} : Bool}}
  +at_target = Each.shaped_list_at(~CopiedPart,~Tile.Tile8,~Parameter.Work,~part_tiles,~work_width,~part_length,~OutputsScope<X,xv,D,dv>,~scope_work(~X,~xv,~D,~dv),
    ~scope_fill(~X,~xv,~D,~dv),~scope_limit(~X,~xv,~D,~dv),
    ~PiecesModel.viewed(~CopiedPart,~KernelGemm.Both<Images.Batch<X,xv,D,dv>,Parameter.Work>,~blocks_fact(~X,~xv,~D,~dv),~OutputsScope<X,xv,D,dv>,~scope_parts(~X,~xv,~D,~dv)),
    ~lane_holds(~X,~xv,~D,~dv),~lane_rule(~X,~xv,~D,~dv),scope,parts,0n,limit,covered,
    PiecesModel.shaped_all_view(~CopiedPart,~KernelGemm.Both<Images.Batch<X,xv,D,dv>,Parameter.Work>,~blocks_fact(~X,~xv,~D,~dv),~OutputsScope<X,xv,D,dv>,~scope_parts(~X,~xv,~D,~dv),
      scope,parts,facts),
    Order.reflexive(limit),target,RowMajor.radix(Nat.div(U32.to_nat(output),8n),limit,U32.to_nat(tap),work_taps(work),block_lt,tap_lt))
  +lane = {%taps_exact : {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),ListRead.at(~Tile.Tile8,fill,Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),_),U32.to_nat(tap)),list)) ==
      Images.lane_levels(Images.batch_images(~X,~xv,~D,~dv,batch),Images.held_images(~X,~xv,~D,~dv,Images.batch_images(~X,~xv,~D,~dv,batch),0n,batch,tap,output)) : F32}
    Logic.required(LaneHolds(~X,~xv,~D,~dv,scope,ListRead.at(~Tile.Tile8,fill,target,list)),Nat.is_eq(target,target),at_target,Order.equal_reflexive(target)) :
    {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),ListRead.at(~Tile.Tile8,fill,cell_index(geometry,tap,output),list)) ==
      Images.lane_levels(Images.batch_images(~X,~xv,~D,~dv,batch),Images.held_images(~X,~xv,~D,~dv,Images.batch_images(~X,~xv,~D,~dv,batch),0n,batch,tap,output)) : F32}}
  %Equal.sym(Kernel.Owners,Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,result),holders(x,d),state) :
    WeightCells<X,xv,D,dv,batch,tap,output,(Kernel.owners_of(_),Pieces.fold(~Array<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~U32,~Kernel.part_cells,taps,
      Pieces.each(~Kernel.Owners,~Array<Tile.Tile8>,~Unit,~Parameter.Work,~Pieces.unshared,~Kernel.part,~Kernel.unmatched,plan,True{},work,
        Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,result),Unit{}),Tensors.buffer(~Tile.Tile8,fill,count)))>
  %packed : WeightCells<X,xv,D,dv,batch,tap,output,(Kernel.owners_of(holders(x,d)),Pieces.fold(~Array<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~U32,~Kernel.part_cells,taps,
    Pieces.each(~Kernel.Owners,~Array<Tile.Tile8>,~Unit,~Parameter.Work,~Pieces.unshared,~Kernel.part,~Kernel.unmatched,plan,True{},work,_,Unit{}),
    Tensors.buffer(~Tile.Tile8,fill,count)))>
  %planned : WeightCells<X,xv,D,dv,batch,tap,output,(Kernel.owners_of(holders(x,d)),Pieces.fold(~Array<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~U32,~Kernel.part_cells,taps,
    Pieces.each(~Kernel.Owners,~Array<Tile.Tile8>,~Unit,~Parameter.Work,~Pieces.unshared,~Kernel.part,~Kernel.unmatched,_,True{},work,
      PiecesModel.shaped_pack(~CopiedPart,~Kernel.Owners,~copied_owners,parts),Unit{}),Tensors.buffer(~Tile.Tile8,fill,count)))>
  %Equal.sym(Partition.Tree<&1,Array<Tile.Tile8>,Unit>,
      Pieces.each(~Kernel.Owners,~Array<Tile.Tile8>,~Unit,~Parameter.Work,~Pieces.unshared,~Kernel.part,~Kernel.unmatched,own_plan,True{},work,
        PiecesModel.shaped_pack(~CopiedPart,~Kernel.Owners,~copied_owners,parts),Unit{}),
      Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~Array<Tile.Tile8>,~Parameter.Work,~Kernel.part,work,parts),
      Each.shaped_planned(~CopiedPart,~Kernel.Owners,~copied_owners,~Array<Tile.Tile8>,~Parameter.Work,~Kernel.part,~Kernel.unmatched,parts,True{},work)) :
    WeightCells<X,xv,D,dv,batch,tap,output,(Kernel.owners_of(holders(x,d)),Pieces.fold(~Array<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~U32,~Kernel.part_cells,taps,_,
      Tensors.buffer(~Tile.Tile8,fill,count)))>
  %Equal.sym(Partition.Tree<&1,Array<Tile.Tile8>,Unit>,Each.shaped_leaf_map(~CopiedPart,~Kernel.Owners,~copied_owners,~Array<Tile.Tile8>,~Parameter.Work,~Kernel.part,work,parts),
      PiecesModel.shaped_pack(~Tree.Tree<Tile.Tile8>,~Array<Tile.Tile8>,~tree_array,valued),
      Each.shaped_leaf_valued(~CopiedPart,~Kernel.Owners,~copied_owners,~Tree.Tree<Tile.Tile8>,~Array<Tile.Tile8>,~tree_array,~Parameter.Work,~Kernel.part,~part_value,~part_valued,
        work,parts)) :
    WeightCells<X,xv,D,dv,batch,tap,output,(Kernel.owners_of(holders(x,d)),Pieces.fold(~Array<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~U32,~Kernel.part_cells,taps,_,
      Tensors.buffer(~Tile.Tile8,fill,count)))>
  %Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)) :
    WeightCells<X,xv,D,dv,batch,tap,output,(Kernel.owners_of(holders(x,d)),Pieces.fold(~Array<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~U32,~Kernel.part_cells,taps,
      PiecesModel.shaped_pack(~Tree.Tree<Tile.Tile8>,~Array<Tile.Tile8>,~tree_array,valued),_))>
  cells_done(~X,~xv,~D,~dv,batch,tap,output,x,d,holds,
    Pieces.fold(~Array<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~U32,~Kernel.part_cells,taps,PiecesModel.shaped_pack(~Tree.Tree<Tile.Tile8>,~Array<Tile.Tile8>,~tree_array,valued),
      Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)))),
    list,contents,length,lane)

# The weight kernel in parts of the output blocks: a lane of a cell of the
# result is the adjacent tree of the lanes of the images, as over the whole
# batch.
def outputs_route(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+depth: Nat,+work: Parameter.Work,+batch: Images.Batch<X,xv,D,dv>,
  +same: WorkBatch<X,xv,D,dv,batch,work>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  -x_held: Move.Holding<Tile.Tile8>,-d_held: Move.Holding<Tile.Tile8>,
  x_covered: Held.Covered<Tile.Tile8,X,xv,Images.batch_input(~X,~xv,~D,~dv,batch),x_held,0n,Images.batch_x_high(~X,~xv,~D,~dv,batch),Images.batch_x_layout(~X,~xv,~D,~dv,batch)>,
  d_covered: Held.Covered<Tile.Tile8,D,dv,Images.batch_gradient(~X,~xv,~D,~dv,batch),d_held,0n,Images.batch_d_high(~X,~xv,~D,~dv,batch),Images.batch_d_layout(~X,~xv,~D,~dv,batch)>) ->
  WeightCells<X,xv,D,dv,batch,tap,output,Kernel.parts(depth,work,x_held,d_held)>:
  WorkBatch{+context_same,+images_same,+cells_same,+x_word,+d_word,+images_room,+positions_exact,+groups_exact,+input_exact,+blocks_exact} = same
  Held.Covered{+x,+x_packed,+x_tiled,+x_values,+x_ordered} = x_covered
  Held.Covered{+d,+d_packed,+d_tiled,+d_values,+d_ordered} = d_covered
  +geometry = Images.batch_geometry(~X,~xv,~D,~dv,batch)
  +limits = Groups.domain_limits(geometry,batch_domain(~X,~xv,~D,~dv,batch))
  +capacity = Machine.capacity(limits)
  +plan = Pieces.unit_ranges(depth,1,0,Kernel.blocks(work))
  +limit = U32.to_nat(Kernel.blocks(work))
  +limit_room = {%Equal.sym(Nat,limit,Groups.output_blocks(geometry),blocks_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Groups.output_blocks_room(geometry,batch_domain(~X,~xv,~D,~dv,batch)) : {Nat.is_le(limit,U32.to_nat(capacity)) == True{} : Bool}}
  +coverage = PiecesPlan.refinement_unit_ranges(depth,1,0,Kernel.blocks(work),capacity,Limits.capacity_fits(limits),8,{==},{==},{==},{==},limit_room)
  +rebuilt = {WorkBatch{context_same,images_same,cells_same,x_word,d_word,images_room,positions_exact,groups_exact,input_exact,blocks_exact} :
    WorkBatch<X,xv,D,dv,batch,work>}
  %Equal.sym(Move.Holding<Tile.Tile8>,x_held,Held.pack(~Tile.Tile8,x),x_packed) : WeightCells<X,xv,D,dv,batch,tap,output,Kernel.parts(depth,work,_,d_held)>
  %Equal.sym(Move.Holding<Tile.Tile8>,d_held,Held.pack(~Tile.Tile8,d),d_packed) : WeightCells<X,xv,D,dv,batch,tap,output,Kernel.parts(depth,work,Held.pack(~Tile.Tile8,x),_)>
  %Equal.sym(Owned(),Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.own,plan,work,holders(x,d)),
      (Pair.fst(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.own,plan,work,holders(x,d))),
        Pair.snd(Kernel.Owners,Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.own,plan,work,holders(x,d)))),
      Held.pair_eta(~Kernel.Owners,~Partition.Tree<&1,Kernel.Owners,Unit>,Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.own,plan,work,holders(x,d)))) :
    WeightCells<X,xv,D,dv,batch,tap,output,Kernel.parts_run(depth,work,_)>
  outputs_assembled(~X,~xv,~D,~dv,depth,work,batch,rebuilt,tap,output,tap_inside,output_inside,coverage,
    Pieces.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.own,plan,work,holders(x,d)),
    parts_made(~X,~xv,~D,~dv,~CopiedPart,~copied_owners,~Parameter.Work,~blocks_fact(~X,~xv,~D,~dv),~blocks_ready(~X,~xv,~D,~dv),~Kernel.own,~blocks_leaf(~X,~xv,~D,~dv),
      plan,batch,work,limit,{BlocksReady{positions_exact,groups_exact,input_exact,blocks_exact} : BlocksReady<X,xv,D,dv,batch,work,limit>},0n,limit,coverage,Order.reflexive(limit),
      x,d,HeldBoth{x_tiled,x_values,x_ordered,d_tiled,d_values,d_ordered}))

# ---- The weight kernel on its route ----

# The serial kernel over the images of a context.
def context_weight(+images: Nat,+context: Kernel.BatchContext,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) ->
  Kernel.Held() & Storage.Buffer<Tile.Tile8>:
  Kernel.BatchContext{+geometry,+rows,+columns,+group} = context
  Kernel.written(Kernel.cells(geometry),Kernel.reduced(images,Kernel.cells(geometry),Kernel.images(images,0,geometry,rows,columns,group,input,gradient)))

def work_weight(+work: Parameter.Work,-input: Move.Holding<Tile.Tile8>,-gradient: Move.Holding<Tile.Tile8>) ->
  {Kernel.weight(work,input,gradient) == context_weight(work_images(work),Kernel.batch_context(work),input,gradient) : Kernel.Held() & Storage.Buffer<Tile.Tile8>}:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  {==}

def batch_weight_context(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+batch: Images.Batch<X,xv,D,dv>,
  -input: Move.Holding<Tile.Tile8>,-gradient: Move.Holding<Tile.Tile8>) ->
  {context_weight(Images.batch_images(~X,~xv,~D,~dv,batch),batch_context_of(~X,~xv,~D,~dv,batch),input,gradient) ==
    Kernel.written(Images.batch_cells(~X,~xv,~D,~dv,batch),Kernel.reduced(Images.batch_images(~X,~xv,~D,~dv,batch),Images.batch_cells(~X,~xv,~D,~dv,batch),
      Kernel.images(Images.batch_images(~X,~xv,~D,~dv,batch),0,Images.batch_geometry(~X,~xv,~D,~dv,batch),Images.batch_rows(~X,~xv,~D,~dv,batch),
        Images.batch_columns(~X,~xv,~D,~dv,batch),Images.batch_group(~X,~xv,~D,~dv,batch),input,gradient))) : Kernel.Held() & Storage.Buffer<Tile.Tile8>}:
  Images.Batch{input_values,gradient_values,geometry,domain,images,rows,columns,group,size,bounded,x_high,x_layout,x_room,x_bounded,d_high,d_layout,d_room,d_bounded} = batch
  {==}

# The serial weight kernel over a work that shares its batch.
def serial_route(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+work: Parameter.Work,+batch: Images.Batch<X,xv,D,dv>,
  +same: WorkBatch<X,xv,D,dv,batch,work>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  -x_held: Move.Holding<Tile.Tile8>,-d_held: Move.Holding<Tile.Tile8>,
  x_covered: Held.Covered<Tile.Tile8,X,xv,Images.batch_input(~X,~xv,~D,~dv,batch),x_held,0n,Images.batch_x_high(~X,~xv,~D,~dv,batch),Images.batch_x_layout(~X,~xv,~D,~dv,batch)>,
  d_covered: Held.Covered<Tile.Tile8,D,dv,Images.batch_gradient(~X,~xv,~D,~dv,batch),d_held,0n,Images.batch_d_high(~X,~xv,~D,~dv,batch),Images.batch_d_layout(~X,~xv,~D,~dv,batch)>) ->
  WeightCells<X,xv,D,dv,batch,tap,output,Kernel.weight(work,x_held,d_held)>:
  WorkBatch{+context_same,+images_same,+cells_same,+x_word,+d_word,+images_room,+positions_exact,+groups_exact,+input_exact,+blocks_exact} = same
  +images = Images.batch_images(~X,~xv,~D,~dv,batch)
  %Equal.sym(Kernel.Held() & Storage.Buffer<Tile.Tile8>,Kernel.weight(work,x_held,d_held),context_weight(work_images(work),Kernel.batch_context(work),x_held,d_held),
      work_weight(work,x_held,d_held)) :
    WeightCells<X,xv,D,dv,batch,tap,output,_>
  %Equal.sym(Nat,work_images(work),images,images_same) :
    WeightCells<X,xv,D,dv,batch,tap,output,context_weight(_,Kernel.batch_context(work),x_held,d_held)>
  %Equal.sym(Kernel.BatchContext,Kernel.batch_context(work),batch_context_of(~X,~xv,~D,~dv,batch),context_same) :
    WeightCells<X,xv,D,dv,batch,tap,output,context_weight(images,_,x_held,d_held)>
  %Equal.sym(Kernel.Held() & Storage.Buffer<Tile.Tile8>,context_weight(images,batch_context_of(~X,~xv,~D,~dv,batch),x_held,d_held),
      Kernel.written(Images.batch_cells(~X,~xv,~D,~dv,batch),Kernel.reduced(images,Images.batch_cells(~X,~xv,~D,~dv,batch),
      Kernel.images(images,0,Images.batch_geometry(~X,~xv,~D,~dv,batch),Images.batch_rows(~X,~xv,~D,~dv,batch),Images.batch_columns(~X,~xv,~D,~dv,batch),
        Images.batch_group(~X,~xv,~D,~dv,batch),x_held,d_held))),
      batch_weight_context(~X,~xv,~D,~dv,batch,x_held,d_held)) :
    WeightCells<X,xv,D,dv,batch,tap,output,_>
  weight_cells(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside,
    Kernel.written(Images.batch_cells(~X,~xv,~D,~dv,batch),Kernel.reduced(images,Images.batch_cells(~X,~xv,~D,~dv,batch),
      Kernel.images(images,0,Images.batch_geometry(~X,~xv,~D,~dv,batch),Images.batch_rows(~X,~xv,~D,~dv,batch),Images.batch_columns(~X,~xv,~D,~dv,batch),
        Images.batch_group(~X,~xv,~D,~dv,batch),x_held,d_held))),
    Images.batch_weight(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside,x_held,d_held,x_covered,d_covered))

# The weight kernel on the output blocks halved depth times.
def routed_cells(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+depth: Nat,+work: Parameter.Work,+batch: Images.Batch<X,xv,D,dv>,
  +same: WorkBatch<X,xv,D,dv,batch,work>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  -x_held: Move.Holding<Tile.Tile8>,-d_held: Move.Holding<Tile.Tile8>,
  x_covered: Held.Covered<Tile.Tile8,X,xv,Images.batch_input(~X,~xv,~D,~dv,batch),x_held,0n,Images.batch_x_high(~X,~xv,~D,~dv,batch),Images.batch_x_layout(~X,~xv,~D,~dv,batch)>,
  d_covered: Held.Covered<Tile.Tile8,D,dv,Images.batch_gradient(~X,~xv,~D,~dv,batch),d_held,0n,Images.batch_d_high(~X,~xv,~D,~dv,batch),Images.batch_d_layout(~X,~xv,~D,~dv,batch)>) ->
  WeightCells<X,xv,D,dv,batch,tap,output,Kernel.routed(depth,work,x_held,d_held)>:
  match depth:
    case 0n: serial_route(~X,~xv,~D,~dv,work,batch,same,tap,output,tap_inside,output_inside,x_held,d_held,x_covered,d_covered)
    case 1n+ +rest: outputs_route(~X,~xv,~D,~dv,1n+rest,work,batch,same,tap,output,tap_inside,output_inside,x_held,d_held,x_covered,d_covered)

# The weight kernel on any route: a lane of a cell of the result is the
# adjacent tree of the lanes of the images.
def route_cells(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+route: Kernel.Route,+work: Parameter.Work,+batch: Images.Batch<X,xv,D,dv>,
  +same: WorkBatch<X,xv,D,dv,batch,work>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  -x_held: Move.Holding<Tile.Tile8>,-d_held: Move.Holding<Tile.Tile8>,
  x_covered: Held.Covered<Tile.Tile8,X,xv,Images.batch_input(~X,~xv,~D,~dv,batch),x_held,0n,Images.batch_x_high(~X,~xv,~D,~dv,batch),Images.batch_x_layout(~X,~xv,~D,~dv,batch)>,
  d_covered: Held.Covered<Tile.Tile8,D,dv,Images.batch_gradient(~X,~xv,~D,~dv,batch),d_held,0n,Images.batch_d_high(~X,~xv,~D,~dv,batch),Images.batch_d_layout(~X,~xv,~D,~dv,batch)>) ->
  WeightCells<X,xv,D,dv,batch,tap,output,Kernel.route_run(route,work,x_held,d_held)>:
  match route:
    case Kernel.OutputParts{+depth}:
      routed_cells(~X,~xv,~D,~dv,depth,work,batch,same,tap,output,tap_inside,output_inside,x_held,d_held,x_covered,d_covered)
    case Kernel.BatchParts{+depth}:
      batch_route(~X,~xv,~D,~dv,depth,work,batch,same,tap,output,tap_inside,output_inside,x_held,d_held,x_covered,d_covered)

# The weight gradient on the workers of the call.
def weight_parts(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+work: Parameter.Work,+workers: U32,+batch: Images.Batch<X,xv,D,dv>,
  +same: WorkBatch<X,xv,D,dv,batch,work>,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch))) == True{} : Bool},
  -x_held: Move.Holding<Tile.Tile8>,-d_held: Move.Holding<Tile.Tile8>,
  x_covered: Held.Covered<Tile.Tile8,X,xv,Images.batch_input(~X,~xv,~D,~dv,batch),x_held,0n,Images.batch_x_high(~X,~xv,~D,~dv,batch),Images.batch_x_layout(~X,~xv,~D,~dv,batch)>,
  d_covered: Held.Covered<Tile.Tile8,D,dv,Images.batch_gradient(~X,~xv,~D,~dv,batch),d_held,0n,Images.batch_d_high(~X,~xv,~D,~dv,batch),Images.batch_d_layout(~X,~xv,~D,~dv,batch)>) ->
  WeightCells<X,xv,D,dv,batch,tap,output,Kernel.weight_parts(work,workers,x_held,d_held)>:
  route_cells(~X,~xv,~D,~dv,Kernel.route(work,workers),work,batch,same,tap,output,tap_inside,output_inside,x_held,d_held,x_covered,d_covered)
