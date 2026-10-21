# Every leaf of a band plan reads a source that holds the gradient rows its
# band copies: the left part of a split reads the copy the split made, the
# right part keeps the source of the split.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../tensor.bend as Tensors
import ../tensor_band.bend as Plans
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_geometry.bend as Geometry
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_band_order.bend as BandOrder
import ./input_gradient_tiled_capacity.bend as Capacity
import ./input_gradient_tiled_copy_geometry.bend as Copy
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_partition.bend as Partition
import ./input_gradient_tiled_source_contents.bend as Contents
import ./input_gradient_tiled_source_coverage.bend as Coverage
import ./input_gradient_tiled_source_rows.bend as Rows
import ./input_gradient_tiled_source_witness.bend as Witnesses
import ./traversal_partition_refinement.bend as PartitionLaws
import ./tensor_band_walk.bend as BandWalk
import ./nat_order.bend as Order
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_buffer_model.bend as Buffer
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./tensor_band_copy.bend as BandCopy
import ./tensor_band_plan.bend as TensorBandPlan
import ./tensor_allocation_depth.bend as Depth
import ./word_power_of_two.bend as Power
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./machine_limit_bounds.bend as Limits
import ./encoded_coordinates.bend as Encoded
import ./word_arithmetic.bend as WordArithmetic
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./boolean_logic.bend as Boolean
import ./math_select_values.bend as Select

# The image a plan runs on: the validated call, its envelope, the gradient
# buffer and the base and plane of this image inside it.
type Setting is Data:
  Setting{call: Domain.Call,envelope: Geometry.WordEnvelope,original: Buffer.Model<F32>,base: U32,plane: U32}
def setting_call(setting: Setting) -> Domain.Call:
  Setting{call,envelope,original,base,plane} = setting
  call
def setting_envelope(setting: Setting) -> Geometry.WordEnvelope:
  Setting{call,envelope,original,base,plane} = setting
  envelope
def setting_original(setting: Setting) -> Buffer.Model<F32>:
  Setting{call,envelope,original,base,plane} = setting
  original
def setting_base(setting: Setting) -> U32:
  Setting{call,envelope,original,base,plane} = setting
  base
def setting_plane(setting: Setting) -> U32:
  Setting{call,envelope,original,base,plane} = setting
  plane
def setting_outputs(setting: Setting) -> U32:
  Capacity.outputs_word(Domain.geometry(setting_call(setting)))
def task(+setting: Setting,+layout: Kernel.Tiling) -> Kernel.Task:
  Kernel.Task{Domain.geometry(setting_call(setting)),layout}

# A leaf band holds a source that covers the gradient rows it copies.
type Held<-setting: Setting,-first: U32,-rows: U32,-source: Execution.Source> is Data:
  Held{room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Domain.height(Domain.geometry(setting_call(setting)))) == True{} : Bool},
    witness: Witnesses.Witness<Bands.caller_query(setting_call(setting),setting_envelope(setting),first,rows,room),source,setting_original(setting),setting_base(setting),setting_plane(setting),setting_outputs(setting)>}
def full_held(+setting: Setting,+first: U32,+rows: U32,
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Domain.height(Domain.geometry(setting_call(setting)))) == True{} : Bool}) ->
  Held<setting,first,rows,Rows.full_source(Bands.caller_query(setting_call(setting),setting_envelope(setting),first,rows,room),setting_original(setting),setting_base(setting),setting_plane(setting))>:
  Held{room,Witnesses.Full{{==}}}

# Word identities of the copy addresses: the copied row of a gradient row and
# the source address a copied cell was read from.
def row_shift(+start: U32,+row: Nat) -> {U32.from_nat(Nat.add(U32.to_nat(start),row)) == (start + U32.from_nat(row) : U32) : U32}:
  Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(start),row)),(U32.from_nat(U32.to_nat(start)) + U32.from_nat(row) : U32),(start + U32.from_nat(row) : U32),
    Equal.sym(U32,(U32.from_nat(U32.to_nat(start)) + U32.from_nat(row) : U32),U32.from_nat(Nat.add(U32.to_nat(start),row)),WordArithmetic.native_add(U32.to_nat(start),row)),
    Equal.cong(U32,U32,left => (left + U32.from_nat(row) : U32),U32.from_nat(U32.to_nat(start)),start,Arithmetic.native_roundtrip(start)))
def copy_address(+span: U32,+width: U32,+start: U32,+output: Nat,+row: Nat,+column: Nat) ->
  {(((0 + U32.from_nat(output) * span : U32) + ((U32.from_nat(Nat.add(U32.to_nat(start),row)) - start : U32) * width : U32) : U32) + U32.from_nat(column) : U32) ==
    ((0 + U32.from_nat(output) * span : U32) + U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)) : U32) : U32}:
  +prefix = (0 + U32.from_nat(output) * span : U32)
  +shifted = U32.from_nat(row)
  +scaled = U32.from_nat(Nat.mul(row,U32.to_nat(width)))
  +col = U32.from_nat(column)
  %Equal.sym(U32,U32.from_nat(Nat.add(U32.to_nat(start),row)),(start + shifted : U32),row_shift(start,row)) :
    {((prefix + ((_ - start : U32) * width : U32) : U32) + col : U32) == (prefix + U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)) : U32) : U32}
  %Equal.sym(U32,((start + shifted : U32) - start : U32),shifted,Arithmetic.subtract_first(start,shifted)) :
    {((prefix + (_ * width : U32) : U32) + col : U32) == (prefix + U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)) : U32) : U32}
  %Equal.sym(U32,(shifted * width : U32),scaled,Encoded.scale_encoded(row,width)) :
    {((prefix + _ : U32) + col : U32) == (prefix + U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)) : U32) : U32}
  Equal.trans(U32,((prefix + scaled : U32) + col : U32),(prefix + (scaled + col : U32) : U32),(prefix + U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)) : U32),
    Arithmetic.add_associative(prefix,scaled,col),
    Equal.cong(U32,U32,rest => (prefix + rest : U32),(scaled + col : U32),U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)),
      WordArithmetic.native_add(Nat.mul(row,U32.to_nat(width)),column)))
def regroup(+b: U32,+d: U32,+o: U32,+r: U32,+c: U32) ->
  {(((b + d : U32) + o : U32) + (r + c : U32) : U32) == (((b + o : U32) + (d + r : U32) : U32) + c : U32) : U32}:
  +head = Equal.trans(U32,((b + d : U32) + o : U32),(b + (d + o : U32) : U32),((b + o : U32) + d : U32),
    Arithmetic.add_associative(b,d,o),
    Equal.trans(U32,(b + (d + o : U32) : U32),(b + (o + d : U32) : U32),((b + o : U32) + d : U32),
      Equal.cong(U32,U32,sum => (b + sum : U32),(d + o : U32),(o + d : U32),Arithmetic.add_commutative(d,o)),
      Equal.sym(U32,((b + o : U32) + d : U32),(b + (o + d : U32) : U32),Arithmetic.add_associative(b,o,d))))
  Equal.trans(U32,(((b + d : U32) + o : U32) + (r + c : U32) : U32),(((b + o : U32) + d : U32) + (r + c : U32) : U32),(((b + o : U32) + (d + r : U32) : U32) + c : U32),
    Equal.cong(U32,U32,left => (left + (r + c : U32) : U32),((b + d : U32) + o : U32),((b + o : U32) + d : U32),head),
    Equal.trans(U32,(((b + o : U32) + d : U32) + (r + c : U32) : U32),((((b + o : U32) + d : U32) + r : U32) + c : U32),(((b + o : U32) + (d + r : U32) : U32) + c : U32),
      Equal.sym(U32,((((b + o : U32) + d : U32) + r : U32) + c : U32),(((b + o : U32) + d : U32) + (r + c : U32) : U32),Arithmetic.add_associative(((b + o : U32) + d : U32),r,c)),
      Equal.cong(U32,U32,left => (left + c : U32),(((b + o : U32) + d : U32) + r : U32),((b + o : U32) + (d + r : U32) : U32),Arithmetic.add_associative((b + o : U32),d,r))))
def source_address(+base: U32,+plane: U32,+width: U32,+first: U32,+start: U32,+output: Nat,+row: Nat,+column: Nat) ->
  {(((base + (start - first : U32) * width : U32) + U32.from_nat(output) * plane : U32) + U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)) : U32) ==
    (((base + U32.from_nat(output) * plane : U32) + ((U32.from_nat(Nat.add(U32.to_nat(start),row)) - first : U32) * width : U32) : U32) + U32.from_nat(column) : U32) : U32}:
  +out = U32.from_nat(output)
  +shifted = U32.from_nat(row)
  +scaled = U32.from_nat(Nat.mul(row,U32.to_nat(width)))
  +col = U32.from_nat(column)
  +origin = (base + (start - first : U32) * width : U32)
  %Equal.sym(U32,U32.from_nat(Nat.add(U32.to_nat(start),row)),(start + shifted : U32),row_shift(start,row)) :
    {((origin + out * plane : U32) + U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)) : U32) ==
      (((base + out * plane : U32) + ((_ - first : U32) * width : U32) : U32) + col : U32) : U32}
  %Arithmetic.add_after_subtract(start,first,shifted) :
    {((origin + out * plane : U32) + U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)) : U32) ==
      (((base + out * plane : U32) + (_ * width : U32) : U32) + col : U32) : U32}
  %Equal.sym(U32,(((start - first : U32) + shifted : U32) * width : U32),((start - first : U32) * width + shifted * width : U32),Arithmetic.multiply_additive((start - first : U32),shifted,width)) :
    {((origin + out * plane : U32) + U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)) : U32) ==
      (((base + out * plane : U32) + _ : U32) + col : U32) : U32}
  %Equal.sym(U32,(shifted * width : U32),scaled,Encoded.scale_encoded(row,width)) :
    {((origin + out * plane : U32) + U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)) : U32) ==
      (((base + out * plane : U32) + ((start - first : U32) * width + _ : U32) : U32) + col : U32) : U32}
  Equal.trans(U32,((origin + out * plane : U32) + U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)) : U32),((origin + out * plane : U32) + (scaled + col : U32) : U32),
    (((base + out * plane : U32) + ((start - first : U32) * width + scaled : U32) : U32) + col : U32),
    Equal.cong(U32,U32,rest => ((origin + out * plane : U32) + rest : U32),U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)),(scaled + col : U32),
      Equal.sym(U32,(scaled + col : U32),U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(width)),column)),WordArithmetic.native_add(Nat.mul(row,U32.to_nat(width)),column))),
    regroup(base,((start - first : U32) * width : U32),(out * plane : U32),scaled,col))

# A word below a bound stays below a larger one.
def word_le_lt(+a: U32,+b: U32,+c: U32,ab: {U32.is_le(a,b) == True{} : Bool},bc: {U32.is_lt(b,c) == True{} : Bool}) -> {U32.is_lt(a,c) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt(a,c),Nat.is_lt(U32.to_nat(a),U32.to_nat(c)),Comparison.lt(a,c)) : {_ == True{} : Bool}
  Order.le_lt_trans(U32.to_nat(a),U32.to_nat(b),U32.to_nat(c),Bounds.as_nat_le(a,b,ab),Bounds.as_nat_lt(b,c,bc))

# The depth guard of a copy buffer.
def guard_limit(+count: U32,guard: Depth.Guarded<count>) -> {Nat.is_le(Tensors.depth_for((count + 1 : U32)),31n) == True{} : Bool}:
  Depth.Guarded{depth_ok,room} = guard
  depth_ok
def guard_room(+count: U32,guard: Depth.Guarded<count>) -> {U32.is_lt(count,U32.shln(1,Tensors.depth_for((count + 1 : U32)))) == True{} : Bool}:
  Depth.Guarded{depth_ok,room} = guard
  room
def cover_at(+query: Bands.Query,+first: U32,+rows: U32,covers: Rows.Covers<query,first,rows>,+row: U32,
  +lower: {Nat.is_le(U32.to_nat(Copy.begin(query)),U32.to_nat(row)) == True{} : Bool},
  +upper: {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(Copy.begin(query)),U32.to_nat(Copy.row_count(query)))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat(row)) == True{} : Bool},{Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}>:
  Rows.Covers{at} = covers
  at(row,lower,upper)

# A copied band, the band it was copied from and the witness of that band.
type Copied is Data:
  Copied{nested: BandOrder.Nested,source: Execution.Source,original: Buffer.Model<F32>,image_base: U32,image_plane: U32,
    witness: Witnesses.Witness<BandOrder.parent(nested),source,original,image_base,image_plane,Capacity.outputs_word(Bands.geometry(BandOrder.parent(nested)))>}
def copied_slice(copied: Copied) -> Rows.Slice:
  Copied{+nested,source,original,image_base,image_plane,witness} = copied
  Rows.query_slice(BandOrder.child(nested),
    Partition.left_source(Bands.geometry(BandOrder.child(nested)),Copy.begin(BandOrder.child(nested)),Copy.row_count(BandOrder.child(nested)),source),original,image_base,image_plane)
def copied_outputs_word(copied: Copied) -> U32:
  Copied{nested,source,original,image_base,image_plane,witness} = copied
  Capacity.outputs_word(Bands.geometry(BandOrder.child(nested)))
def copied_outputs(copied: Copied) -> Nat:
  U32.to_nat(copied_outputs_word(copied))
def plane_size(copied: Copied) -> Nat:
  U32.to_nat(Execution.source_rows(Rows.source(copied_slice(copied))))
def line_size(copied: Copied) -> Nat:
  U32.to_nat(Rows.width(copied_slice(copied)))
def Planes(+copied: Copied,+output: Nat) -> Data:
  Members.All(Nat,{row => Members.All(Nat,{column => Contents.Cell(copied_slice(copied),output,row,column) : Nat -> Data},Ranges.range(line_size(copied),0n)) : Nat -> Data},
    Ranges.range(plane_size(copied),0n))
type Plane is Data:
  Plane{copied: Copied,output: Nat,output_inside: {Nat.is_lt(output,copied_outputs(copied)) == True{} : Bool}}
def plane_copied(plane: Plane) -> Copied:
  Plane{copied,output,output_inside} = plane
  copied
def plane_output(plane: Plane) -> Nat:
  Plane{copied,output,output_inside} = plane
  output
def plane_rows(plane: Plane) -> Nat:
  plane_size(plane_copied(plane))
def Lines(+plane: Plane,+row: Nat) -> Data:
  Members.All(Nat,{column => Contents.Cell(copied_slice(plane_copied(plane)),plane_output(plane),row,column) : Nat -> Data},Ranges.range(line_size(plane_copied(plane)),0n))
type Line is Data:
  Line{plane: Plane,row: Nat,row_inside: {Nat.is_lt(row,plane_rows(plane)) == True{} : Bool}}
def line_plane(line: Line) -> Plane:
  Line{plane,row,row_inside} = line
  plane
def line_row(line: Line) -> Nat:
  Line{plane,row,row_inside} = line
  row
def line_columns(line: Line) -> Nat:
  line_size(plane_copied(line_plane(line)))
def Point(+line: Line,+column: Nat) -> Data:
  Contents.Cell(copied_slice(plane_copied(line_plane(line))),plane_output(line_plane(line)),line_row(line),column)

# One cell of the copy holds the gradient value: the copy readback reaches the
# source of the band the copy was made from, which holds the image value.
def point(+line: Line,+column: Nat,+inside: {Nat.is_lt(column,line_columns(line)) == True{} : Bool}) -> Point(line,column):
  Line{+plane,+row,+row_inside} = line
  Plane{+copied,+output,+output_inside} = plane
  Copied{+nested,+source,+original,+image_base,+image_plane,+witness} = copied
  BandOrder.Nested{+call,+envelope,+parent_first,+parent_rows,+parent_room,+child_first,+child_rows,+child_room,+lower,+upper} = nested
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Execution.Source{+data,+source_base,+source_plane,+source_first,+source_rows} = source
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits,facts} : Domain.Call}
  +pair = {BandOrder.Nested{whole,envelope,parent_first,parent_rows,parent_room,child_first,child_rows,child_room,lower,upper} : BandOrder.Nested}
  +parent = {Bands.caller_query(whole,envelope,parent_first,parent_rows,parent_room) : Bands.Query}
  +child = {Bands.caller_query(whole,envelope,child_first,child_rows,child_room) : Bands.Query}
  +origin = {Execution.Source{data,source_base,source_plane,source_first,source_rows} : Execution.Source}
  +begin = Copy.begin(child)
  +count = Copy.row_count(child)
  +begin_n = U32.to_nat(begin)
  +count_n = U32.to_nat(count)
  +width_n = U32.to_nat(gw)
  +outputs_n = U32.to_nat(o)
  +height_n = U32.to_nat(gh)
  +span = (count * gw : U32)
  +span_n = U32.to_nat(span)
  +from = (source_base + (begin - source_first : U32) * gw : U32)
  +out = U32.from_nat(output)
  +row_n = Nat.add(begin_n,row)
  +row_word = U32.from_nat(row_n)
  +col = U32.from_nat(column)
  +index = Nat.add(Nat.mul(row,width_n),column)
  +depth = Partition.copy_depth(o,count,gw)
  +fresh = Tree.fresh(F32,depth,0.0)
  +writes = Partition.copy_writes(o,count,gw,source_plane,from,data)
  +query = ((0 + out * span : U32) + U32.from_nat(index) : U32)
  +sample = ((from + out * source_plane : U32) + U32.from_nat(index) : U32)
  +capacity_word = Machine.capacity(limits)
  +capacity_n = U32.to_nat(capacity_word)
  +coordinate_n = U32.to_nat(Machine.coordinate(limits))
  +stop_n = U32.to_nat(Copy.stop(child))
  +positive_rows = Order.le_lt_trans(0n,row,count_n,Order.zero_le(row),row_inside)
  +count_bound = Order.transitive(count_n,stop_n,height_n,
    %Equal.sym(Nat,count_n,Nat.sub(stop_n,begin_n),Copy.count_exact(child)) : {Nat.is_le(_,stop_n) == True{} : Bool}
    Order.sub_le(stop_n,begin_n),Copy.stop_bounded(child))
  +area_bound = Order.transitive(Nat.mul(count_n,width_n),Nat.mul(height_n,width_n),capacity_n,RowMajor.mul_le(count_n,height_n,width_n,count_bound),Domain.gradient_area_room(whole))
  +span_exact = Arithmetic.multiply(count,gw,capacity_word,Limits.capacity_fits(limits),area_bound)
  +index_inside = {%Equal.sym(Nat,span_n,Nat.mul(count_n,width_n),span_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(row,count_n,column,width_n,row_inside,inside) : {Nat.is_lt(index,span_n) == True{} : Bool}}
  +positive_width = Order.le_lt_trans(0n,column,width_n,Order.zero_le(column),inside)
  +positive_span = Order.le_lt_trans(0n,index,span_n,Order.zero_le(index),index_inside)
  +volume_bound = {%Equal.sym(Nat,Nat.mul(Nat.mul(outputs_n,count_n),width_n),Nat.mul(outputs_n,Nat.mul(count_n,width_n)),RowMajor.assoc(outputs_n,count_n,width_n)) :
      {Nat.is_le(_,capacity_n) == True{} : Bool}
    Order.transitive(Nat.mul(outputs_n,Nat.mul(count_n,width_n)),Nat.mul(outputs_n,Nat.mul(height_n,width_n)),capacity_n,
      DivisionOrder.multiply_left_le(outputs_n,Nat.mul(count_n,width_n),Nat.mul(height_n,width_n),RowMajor.mul_le(count_n,height_n,width_n,count_bound)),
      Domain.gradient_image_room(whole)) : {Nat.is_le(Nat.mul(Nat.mul(outputs_n,count_n),width_n),capacity_n) == True{} : Bool}}
  +volume_word = (o * count * gw : U32)
  +volume_exact = Encoded.multiply_three(o,count,gw,capacity_word,Limits.capacity_fits(limits),positive_width,volume_bound)
  +volume_fits = {%Equal.sym(Bool,U32.is_le(volume_word,capacity_word),Nat.is_le(U32.to_nat(volume_word),capacity_n),Comparison.le(volume_word,capacity_word)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(volume_word),Nat.mul(Nat.mul(outputs_n,count_n),width_n),volume_exact) : {Nat.is_le(_,capacity_n) == True{} : Bool}
    volume_bound : {U32.is_le(volume_word,capacity_word) == True{} : Bool}}
  +volume_below = word_le_lt(volume_word,capacity_word,2147483648,volume_fits,Limits.capacity_below_half(limits))
  +guard = Depth.guarded(volume_word,volume_below)
  +depth_ok = guard_limit(volume_word,guard)
  +depth_room = guard_room(volume_word,guard)
  +balanced = Shape.fresh(F32,depth,0.0)
  +tree_capacity = Tree.capacity(F32,fresh)
  +capacity_equal = Shape.capacity(F32,depth,fresh,balanced)
  +capacity_ok = {%Equal.sym(U32,tree_capacity,U32.shln(1,depth),capacity_equal) : {U32.is_le(_,2147483648) == True{} : Bool}
    Power.native_power_bound(depth,depth_ok) : {U32.is_le(tree_capacity,2147483648) == True{} : Bool}}
  +room_tree = {%Equal.sym(Nat,span_n,Nat.mul(count_n,width_n),span_exact) : {Nat.is_le(Nat.mul(outputs_n,_),U32.to_nat(tree_capacity)) == True{} : Bool}
    %RowMajor.assoc(outputs_n,count_n,width_n) : {Nat.is_le(_,U32.to_nat(tree_capacity)) == True{} : Bool}
    %volume_exact : {Nat.is_le(_,U32.to_nat(tree_capacity)) == True{} : Bool}
    %Equal.sym(U32,tree_capacity,U32.shln(1,depth),capacity_equal) : {Nat.is_le(U32.to_nat(volume_word),U32.to_nat(_)) == True{} : Bool}
    Bounds.strict_le(U32.to_nat(volume_word),U32.to_nat(U32.shln(1,depth)),Bounds.as_nat_lt(volume_word,U32.shln(1,depth),depth_room)) :
    {Nat.is_le(Nat.mul(outputs_n,span_n),U32.to_nat(tree_capacity)) == True{} : Bool}}
  +region = {BandCopy.whole_region(outputs_n,span,tree_capacity) : BandCopy.Region}
  +valid = BandCopy.whole_valid(outputs_n,span,tree_capacity,capacity_ok,positive_span,room_tree)
  +band_list = BandCopy.writes(~F32,outputs_n,0,span,source_plane,span,from,0,data)
  +bridge = Partition.band_writes(outputs_n,0,span,source_plane,from,span,data)
  +query_inside = BandCopy.query_word_inside(region,output,index,valid,output_inside,index_inside)
  +covered = BandCopy.writes_inside(~F32,outputs_n,0n,region,source_plane,from,data,valid,Order.reflexive(outputs_n))
  +program = Readback.program_full(~F32,writes,depth,fresh,query,balanced,depth_ok,query_inside,
    %Equal.sym(List<&2,Writes.Assignment<F32>>,writes,band_list,bridge) : Readback.Inside(~F32,tree_capacity,_) covered)
  +selected = {%Equal.sym(List<&2,Writes.Assignment<F32>>,writes,band_list,bridge) : {Readback.selected(~F32,_,query,Tree.read(F32,fresh,query)) == Tree.read(F32,data,sample) : F32}
    BandCopy.selected_inside(~F32,outputs_n,0n,output,index,region,source_plane,from,data,valid,Order.reflexive(outputs_n),output_inside,index_inside,Tree.read(F32,fresh,query)) :
    {Readback.selected(~F32,writes,query,Tree.read(F32,fresh,query)) == Tree.read(F32,data,sample) : F32}}
  +copy_read = Equal.trans(F32,Tree.read(F32,Writes.pure(F32,writes,fresh),query),Readback.selected(~F32,writes,query,Tree.read(F32,fresh,query)),Tree.read(F32,data,sample),program,selected)
  +child_address = copy_address(span,gw,begin,output,row,column)
  +parent_address = source_address(source_base,source_plane,gw,source_first,begin,output,row,column)
  +row_below = Order.offset_lt(begin_n,row,count_n,row_inside)
  +row_in_image = Bounds.nat_trans(row_n,Nat.add(begin_n,count_n),height_n,row_below,BandOrder.end_inside(child,positive_rows))
  +row_exact = FullRange.roundtrip(row_n,Machine.coordinate(limits),Order.transitive(row_n,height_n,coordinate_n,Bounds.strict_le(row_n,height_n,row_in_image),Rows.query_height_room(child)))
  +output_exact = FullRange.roundtrip(output,Machine.dimension(limits),
    Order.transitive(output,outputs_n,U32.to_nat(Machine.dimension(limits)),Bounds.strict_le(output,outputs_n,output_inside),Domain.outputs_room(whole)))
  +width_coordinate = Order.transitive(width_n,Nat.mul(width_n,U32.to_nat(s)),coordinate_n,RowMajor.product_covers(width_n,U32.to_nat(s),Domain.stride_positive(whole)),
    Order.transitive(Nat.mul(width_n,U32.to_nat(s)),Nat.add(Nat.mul(width_n,U32.to_nat(s)),U32.to_nat(k)),coordinate_n,Order.le_add(Nat.mul(width_n,U32.to_nat(s)),U32.to_nat(k)),Domain.column_source_room(whole)))
  +column_exact = FullRange.roundtrip(column,Machine.coordinate(limits),Order.transitive(column,width_n,coordinate_n,Bounds.strict_le(column,width_n,inside),width_coordinate))
  +output_word_inside = Bounds.native_bound(output,o,output_exact,output_inside)
  +row_word_inside = Bounds.native_bound(row_n,gh,row_exact,row_in_image)
  +column_word_inside = Bounds.native_bound(column,gw,column_exact,inside)
  cover = Witnesses.cover(parent,origin,original,image_base,image_plane,o,witness)
  +interval = cover_at(parent,source_first,source_rows,cover,row_word,
    %Equal.sym(Nat,U32.to_nat(row_word),row_n,row_exact) : {Nat.is_le(U32.to_nat(Copy.begin(parent)),_) == True{} : Bool}
    Order.transitive(U32.to_nat(Copy.begin(parent)),begin_n,row_n,BandOrder.begin_monotone(pair),Order.le_add(begin_n,row)),
    %Equal.sym(Nat,U32.to_nat(row_word),row_n,row_exact) : {Nat.is_lt(_,Nat.add(U32.to_nat(Copy.begin(parent)),U32.to_nat(Copy.row_count(parent)))) == True{} : Bool}
    Bounds.nat_trans(row_n,Nat.add(begin_n,count_n),Nat.add(U32.to_nat(Copy.begin(parent)),U32.to_nat(Copy.row_count(parent))),row_below,
      BandOrder.end_bound(pair,BandOrder.positive_count(child,positive_rows))))
  stored = Witnesses.stored(parent,origin,original,image_base,image_plane,o,witness)
  +parent_read = Rows.stored_read(Rows.query_slice(parent,origin,original,image_base,image_plane),o,out,row_word,col,stored,output_word_inside,
    Boolean.and_true(U32.is_lt(row_word,gh),U32.is_lt(col,gw),row_word_inside,column_word_inside),interval)
  Equal.trans(F32,Tree.read(F32,Writes.pure(F32,writes,fresh),(((0 + out * span : U32) + ((row_word - begin : U32) * gw : U32) : U32) + col : U32)),Tree.read(F32,data,sample),
    Tree.read(F32,Buffer.tree(~F32,original),(((image_base + out * image_plane : U32) + row_word * gw : U32) + col : U32)),
    Equal.trans(F32,Tree.read(F32,Writes.pure(F32,writes,fresh),(((0 + out * span : U32) + ((row_word - begin : U32) * gw : U32) : U32) + col : U32)),Tree.read(F32,Writes.pure(F32,writes,fresh),query),Tree.read(F32,data,sample),
      Equal.cong(U32,F32,address => Tree.read(F32,Writes.pure(F32,writes,fresh),address),(((0 + out * span : U32) + ((row_word - begin : U32) * gw : U32) : U32) + col : U32),query,child_address),
      copy_read),
    Equal.trans(F32,Tree.read(F32,data,sample),Tree.read(F32,data,(((source_base + out * source_plane : U32) + ((row_word - source_first : U32) * gw : U32) : U32) + col : U32)),
      Tree.read(F32,Buffer.tree(~F32,original),(((image_base + out * image_plane : U32) + row_word * gw : U32) + col : U32)),
      Equal.cong(U32,F32,address => Tree.read(F32,data,address),sample,(((source_base + out * source_plane : U32) + ((row_word - source_first : U32) * gw : U32) : U32) + col : U32),parent_address),
      parent_read))
def points(+plane: Plane,+row: Nat,+inside: {Nat.is_lt(row,plane_rows(plane)) == True{} : Bool}) -> Lines(plane,row):
  Members.range_all(~Line,~Point,~line_columns,~point,line_size(plane_copied(plane)),0n,Line{plane,row,inside},Order.reflexive(line_size(plane_copied(plane))))
def lines(+copied: Copied,+output: Nat,+inside: {Nat.is_lt(output,copied_outputs(copied)) == True{} : Bool}) -> Planes(copied,output):
  Members.range_all(~Plane,~Lines,~plane_rows,~points,plane_size(copied),0n,Plane{copied,output,inside},Order.reflexive(plane_size(copied)))
def planes(+copied: Copied) -> Contents.Content(copied_slice(copied),copied_outputs_word(copied)):
  Members.range_all(~Copied,~Planes,~copied_outputs,~lines,copied_outputs(copied),0n,copied,Order.reflexive(copied_outputs(copied)))

# The left part of a split reads the copy of the rows its band copies.
def copied_held(+setting: Setting,+first: U32,+rows: U32,+half: U32,+source: Execution.Source,
  +half_room: {Nat.is_le(U32.to_nat(half),U32.to_nat(rows)) == True{} : Bool},+held: Held<setting,first,rows,source>) ->
  Held<setting,first,half,Partition.left_source(Domain.geometry(setting_call(setting)),Kernel.copy_start(Domain.geometry(setting_call(setting)),first,half),
    Kernel.copy_count(Domain.geometry(setting_call(setting)),first,half),source)>:
  Setting{+call,+envelope,+original,+base,+plane} = setting
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Held{+room,+witness} = held
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits,facts} : Domain.Call}
  +site = {Setting{whole,envelope,original,base,plane} : Setting}
  +first_n = U32.to_nat(first)
  +room_left = Order.transitive(Nat.add(first_n,U32.to_nat(half)),Nat.add(first_n,U32.to_nat(rows)),U32.to_nat(h),Order.offset_le(first_n,U32.to_nat(half),U32.to_nat(rows),half_room),room)
  +band = {Bands.caller_query(whole,envelope,first,half,room_left) : Bands.Query}
  +pair = {BandOrder.Nested{whole,envelope,first,rows,room,first,half,room_left,Order.reflexive(first_n),Order.offset_le(first_n,U32.to_nat(half),U32.to_nat(rows),half_room)} : BandOrder.Nested}
  +rows_equal = Bands.caller_count_matches(whole,envelope,first,half,room_left)
  +start_equal = Equal.trans(U32,Kernel.copy_start(shape,first,half),Kernel.copy_start(shape,first,Bands.rows_word(band)),Copy.begin(band),
    Equal.cong(U32,U32,count => Kernel.copy_start(shape,first,count),half,Bands.rows_word(band),Equal.sym(U32,Bands.rows_word(band),half,rows_equal)),Copy.begin_matches(band))
  +count_equal = Equal.trans(U32,Kernel.copy_count(shape,first,half),Kernel.copy_count(shape,first,Bands.rows_word(band)),Copy.row_count(band),
    Equal.cong(U32,U32,count => Kernel.copy_count(shape,first,count),half,Bands.rows_word(band),Equal.sym(U32,Bands.rows_word(band),half,rows_equal)),Copy.count_matches(band))
  %Equal.sym(U32,Kernel.copy_start(shape,first,half),Copy.begin(band),start_equal) :
    Held<site,first,half,Partition.left_source(shape,_,Kernel.copy_count(shape,first,half),source)>
  %Equal.sym(U32,Kernel.copy_count(shape,first,half),Copy.row_count(band),count_equal) :
    Held<site,first,half,Partition.left_source(shape,Copy.begin(band),_,source)>
  Held{room_left,Witnesses.Retained{BandOrder.copy_range(band),Coverage.Exact{{==},{==}},planes(Copied{pair,source,original,base,plane,witness})}}

# The right part of a split keeps the source; its band lies inside the band of the split.
def narrowed(+pair: BandOrder.Nested,+source: Execution.Source,+original: Buffer.Model<F32>,+base: U32,+plane: U32,+outputs: U32,
  witness: Witnesses.Witness<BandOrder.parent(pair),source,original,base,plane,outputs>) ->
  Witnesses.Witness<BandOrder.child(pair),source,original,base,plane,outputs>:
  BandOrder.Nested{+call,+envelope,+parent_first,+parent_rows,+parent_room,+child_first,+child_rows,+child_room,+lower,+upper} = pair
  match witness:
    case Witnesses.Full{same}: Witnesses.Full{same}
    case Witnesses.Retained{range,coverage,contents}:
      Witnesses.Retained{range,
        BandOrder.restrict(BandOrder.Nested{call,envelope,parent_first,parent_rows,parent_room,child_first,child_rows,child_room,lower,upper},
          Execution.source_first(source),Execution.source_rows(source),coverage),
        contents}
def narrow(+setting: Setting,+first: U32,+rows: U32,+half: U32,+source: Execution.Source,
  +half_room: {Nat.is_le(U32.to_nat(half),U32.to_nat(rows)) == True{} : Bool},held: Held<setting,first,rows,source>) ->
  Held<setting,(first + half : U32),(rows - half : U32),source>:
  Setting{+call,+envelope,+original,+base,+plane} = setting
  Held{+room,witness} = held
  +first_n = U32.to_nat(first)
  +rows_n = U32.to_nat(rows)
  +half_n = U32.to_nat(half)
  +rest_n = Nat.sub(rows_n,half_n)
  +height = Domain.height(Domain.geometry(call))
  +whole = Nat.add(first_n,rows_n)
  +mid_exact = BandOrder.last_exact(call,first,half,Order.transitive(Nat.add(first_n,half_n),whole,height,Order.offset_le(first_n,half_n,rows_n,half_room),room))
  +rest_exact = FullRange.subtract(rows,half,rows,half_room,Order.sub_le(rows_n,half_n))
  +end_equal = Equal.trans(Nat,Nat.add(U32.to_nat((first + half : U32)),U32.to_nat((rows - half : U32))),Nat.add(Nat.add(first_n,half_n),U32.to_nat((rows - half : U32))),whole,
    Equal.cong(Nat,Nat,head => Nat.add(head,U32.to_nat((rows - half : U32))),U32.to_nat((first + half : U32)),Nat.add(first_n,half_n),mid_exact),
    Equal.trans(Nat,Nat.add(Nat.add(first_n,half_n),U32.to_nat((rows - half : U32))),Nat.add(Nat.add(first_n,half_n),rest_n),whole,
      Equal.cong(Nat,Nat,tail => Nat.add(Nat.add(first_n,half_n),tail),U32.to_nat((rows - half : U32)),rest_n,rest_exact),
      Equal.trans(Nat,Nat.add(Nat.add(first_n,half_n),rest_n),Nat.add(first_n,Nat.add(half_n,rest_n)),whole,Addition.associative(first_n,half_n,rest_n),
        Equal.cong(Nat,Nat,tail => Nat.add(first_n,tail),Nat.add(half_n,rest_n),rows_n,Order.restore_shift(half_n,rows_n,half_room)))))
  +room_right = {%Equal.sym(Nat,Nat.add(U32.to_nat((first + half : U32)),U32.to_nat((rows - half : U32))),whole,end_equal) : {Nat.is_le(_,height) == True{} : Bool}
    room : {Nat.is_le(Nat.add(U32.to_nat((first + half : U32)),U32.to_nat((rows - half : U32))),height) == True{} : Bool}}
  +pair = {BandOrder.Nested{call,envelope,first,rows,room,(first + half : U32),(rows - half : U32),room_right,
    %Equal.sym(Nat,U32.to_nat((first + half : U32)),Nat.add(first_n,half_n),mid_exact) : {Nat.is_le(first_n,_) == True{} : Bool} Order.le_add(first_n,half_n),
    %Equal.sym(Nat,Nat.add(U32.to_nat((first + half : U32)),U32.to_nat((rows - half : U32))),whole,end_equal) : {Nat.is_le(_,whole) == True{} : Bool} Order.reflexive(whole)} : BandOrder.Nested}
  Held{room_right,narrowed(pair,source,original,base,plane,Capacity.outputs_word(Domain.geometry(call)),witness)}

# Every band of a covering plan holds its source when the whole band does:
# the fact about every band of the model walk, under a scope that names the
# setting.
def scoped_held(~Scope: Data,~setting: Scope -> Setting,+scope: Scope,+task: Kernel.Task,+first: U32,+rows: U32,+state: Partition.Owners) -> Data:
  Held<setting(scope),first,rows,Partition.owner_source(state)>

def held_leaves(~Scope: Data,~setting: Scope -> Setting,plan: Plans.Plan,+scope: Scope,+layout: Kernel.Tiling,+first: U32,+rows: U32,+source: Execution.Source,
  +weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,coverage: TensorBandPlan.Coverage(plan,U32.to_nat(first),U32.to_nat(rows)),+held: Held<setting(scope),first,rows,source>) ->
  BandWalk.StateAll(~Partition.Owners,~Kernel.Task,~Scope,~Partition.split,~scoped_held(~Scope,~setting),scope,plan,task(setting(scope),layout),Partition.Owners{source,weights,offsets}):
  match plan:
    case Plans.Rows{+leaf_first,+count}:
      TensorBandPlan.LeafCoverage{+first_equal,+rows_equal} = coverage
      %Equal.sym(U32,leaf_first,first,Arithmetic.to_nat_injective(leaf_first,first,first_equal)) : Held<setting(scope),_,count,source>
      %Equal.sym(U32,count,rows,Arithmetic.to_nat_injective(count,rows,rows_equal)) : Held<setting(scope),first,_,source>
      held
    case Plans.Split{+left,+right}:
      TensorBandPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
      Held{+room,+witness} = held
      +geometry = Domain.geometry(setting_call(setting(scope)))
      +height = Domain.height(geometry)
      +first_n = U32.to_nat(first)
      +rows_n = U32.to_nat(rows)
      +half = (Plans.first(right) - first : U32)
      +mid_n = TensorBandPlan.coverage_first(right,Nat.add(first_n,left_rows),right_rows,right_coverage)
      +half_exact = Equal.trans(Nat,U32.to_nat(half),Nat.sub(U32.to_nat(Plans.first(right)),first_n),left_rows,
        FullRange.subtract(Plans.first(right),first,rows,
          %Equal.sym(Nat,U32.to_nat(Plans.first(right)),Nat.add(first_n,left_rows),mid_n) : {Nat.is_le(first_n,_) == True{} : Bool} Order.le_add(first_n,left_rows),
          %Equal.sym(Nat,U32.to_nat(Plans.first(right)),Nat.add(first_n,left_rows),mid_n) : {Nat.is_le(Nat.sub(_,first_n),rows_n) == True{} : Bool}
          %Equal.sym(Nat,Nat.sub(Nat.add(first_n,left_rows),first_n),left_rows,Order.subtract_shift(first_n,left_rows)) : {Nat.is_le(_,rows_n) == True{} : Bool}
          %Equal.sym(Nat,rows_n,Nat.add(left_rows,right_rows),rows_equal) : {Nat.is_le(left_rows,_) == True{} : Bool}
          Order.le_add(left_rows,right_rows)),
        %Equal.sym(Nat,U32.to_nat(Plans.first(right)),Nat.add(first_n,left_rows),mid_n) : {Nat.sub(_,first_n) == left_rows : Nat}
        Order.subtract_shift(first_n,left_rows))
      +half_room = {%Equal.sym(Nat,U32.to_nat(half),left_rows,half_exact) : {Nat.is_le(_,rows_n) == True{} : Bool}
        %Equal.sym(Nat,rows_n,Nat.add(left_rows,right_rows),rows_equal) : {Nat.is_le(left_rows,_) == True{} : Bool}
        Order.le_add(left_rows,right_rows) : {Nat.is_le(U32.to_nat(half),rows_n) == True{} : Bool}}
      +mid_exact = BandOrder.last_exact(setting_call(setting(scope)),first,half,
        Order.transitive(Nat.add(first_n,U32.to_nat(half)),Nat.add(first_n,rows_n),height,Order.offset_le(first_n,U32.to_nat(half),rows_n,half_room),room))
      +rest_exact = FullRange.subtract(rows,half,rows,half_room,Order.sub_le(rows_n,U32.to_nat(half)))
      +rest_rows = Equal.trans(Nat,U32.to_nat((rows - half : U32)),Nat.sub(rows_n,U32.to_nat(half)),right_rows,rest_exact,
        %Equal.sym(Nat,U32.to_nat(half),left_rows,half_exact) : {Nat.sub(rows_n,_) == right_rows : Nat}
        %Equal.sym(Nat,rows_n,Nat.add(left_rows,right_rows),rows_equal) : {Nat.sub(_,left_rows) == right_rows : Nat}
        Order.subtract_shift(left_rows,right_rows))
      %Equal.sym(U32,Plans.first(left),first,Arithmetic.to_nat_injective(Plans.first(left),first,TensorBandPlan.coverage_first(left,first_n,left_rows,left_coverage))) :
        Shape.Both<BandWalk.StateAll(~Partition.Owners,~Kernel.Task,~Scope,~Partition.split,~scoped_held(~Scope,~setting),scope,left,task(setting(scope),layout),
            Partition.Owners{Partition.left_source(geometry,Kernel.copy_start(geometry,_,(Plans.first(right) - _ : U32)),Kernel.copy_count(geometry,_,(Plans.first(right) - _ : U32)),source),weights,offsets}),
          BandWalk.StateAll(~Partition.Owners,~Kernel.Task,~Scope,~Partition.split,~scoped_held(~Scope,~setting),scope,right,task(setting(scope),layout),Partition.Owners{source,weights,offsets})>
      Shape.Both{held_leaves(~Scope,~setting,left,scope,layout,first,half,Partition.left_source(geometry,Kernel.copy_start(geometry,first,half),Kernel.copy_count(geometry,first,half),source),weights,offsets,
          %Equal.sym(Nat,U32.to_nat(half),left_rows,half_exact) : TensorBandPlan.Coverage(left,first_n,_) left_coverage,
          copied_held(setting(scope),first,rows,half,source,half_room,Held{room,witness})),
        held_leaves(~Scope,~setting,right,scope,layout,(first + half : U32),(rows - half : U32),source,weights,offsets,
          %Equal.sym(Nat,U32.to_nat((first + half : U32)),Nat.add(first_n,left_rows),
            Equal.trans(Nat,U32.to_nat((first + half : U32)),Nat.add(first_n,U32.to_nat(half)),Nat.add(first_n,left_rows),mid_exact,Equal.cong(Nat,Nat,tail => Nat.add(first_n,tail),U32.to_nat(half),left_rows,half_exact))) :
            TensorBandPlan.Coverage(right,_,U32.to_nat((rows - half : U32)))
          %Equal.sym(Nat,U32.to_nat((rows - half : U32)),right_rows,rest_rows) : TensorBandPlan.Coverage(right,Nat.add(first_n,left_rows),_)
          right_coverage,
          narrow(setting(scope),first,rows,half,source,half_room,Held{room,witness}))}
