# The input-gradient specification reads the gradient and the weights as the
# values of their cells, with the reader the runtime untiles them with: the
# gradient [N][O][gh][gw] with the groups of a pixel side by side, the weights
# [O][C][k][k] with their groups outermost.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/tensor_tiled_cells.bend as Cells
import ../../lib/tensor_grid.bend as Grid
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/list_storage_readback.bend as ListRead
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/tensor_adjoint_index.bend as WeightIndex
import ../../lib/proofs/input_gradient_phase_packing.bend as Packing
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_division.bend as NativeDivision
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/input_gradient_phase_bounds.bend as PhaseBounds
import ../../lib/proofs/input_gradient_phase_geometry.bend as Geometry
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/input_gradient_phase_weights.bend as Weights
import ../specification.bend as ConvSpec
import ../input_gradient_specification.bend as Spec
import ./input_gradient_sources.bend as Sources
import ./input_storage.bend as Inputs
import ./input_gradient_order.bend as StridedOrder

# ---- The reader ----

# Cells and the geometry of their values.
type Values is Data:
  Values{tree: Tree.Tree<Tile.Tile8>,cells: Cells.Cells}

def value_at(source: Values,index: Nat) -> F32:
  Values{tree,cells} = source
  TiledValues.value_value(tree,cells,U32.from_nat(index))

type Reader is Data:
  Reader{gradient: Values,weights: Values}

def gradient_cells(+geometry: Kernel.Geometry) -> Cells.Cells:
  Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
  Cells.Cells{0,(gh * gw : U32),o,Tiled.groups(o),Grid.Innermost{}}

def weight_cells(+geometry: Kernel.Geometry) -> Cells.Cells:
  Kernel.Geometry{+c,+o,h,w,+k,s,p,gh,gw} = geometry
  Cells.Cells{(o * (k * k : U32) : U32),(k * k : U32),c,Tiled.groups(c),Grid.Outermost{}}

def models_reader(+geometry: Kernel.Geometry,gradient: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>) -> Reader:
  Reader{Values{Buffer.tree(~Tile.Tile8,gradient),gradient_cells(geometry)},Values{Buffer.tree(~Tile.Tile8,weights),weight_cells(geometry)}}

def problem(+geometry: Kernel.Geometry,reader: Reader,+image: Nat) -> Spec.Problem:
  Reader{gradient,weights} = reader
  Spec.Problem{Sources.shape(geometry),ListRead.enumerate(~F32,~Values,~value_at,Domain.gradient_size(geometry),Nat.mul(image,Domain.gradient_size(geometry)),gradient),
    ListRead.enumerate(~F32,~Values,~value_at,Domain.weight_size(geometry),0n,weights)}

def weight_source(reader: Reader) -> Values:
  Reader{gradient,weights} = reader
  weights

def gradient_source(reader: Reader) -> Values:
  Reader{gradient,weights} = reader
  gradient

def enumerated_at(+count: Nat,+index: Nat,+first: Nat,+source: Values,inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {ConvSpec.at(index,ListRead.enumerate(~F32,~Values,~value_at,count,first,source)) == value_at(source,Nat.add(first,index)) : F32}:
  +values = ListRead.enumerate(~F32,~Values,~value_at,count,first,source)
  %Equal.sym(F32,ConvSpec.at(index,values),ListRead.at(~F32,0.0,index,values),Inputs.lookup_matches_readback(index,values)) : {_ == value_at(source,Nat.add(first,index)) : F32}
  ListRead.enumerate_at(~F32,~Values,~value_at,0.0,count,index,first,source,inside)

def weight(+geometry: Kernel.Geometry,+reader: Reader,+image: Nat,+output: Nat,+channel: Nat,+row: Nat,+column: Nat,
  output_inside: {Nat.is_lt(output,Domain.outputs(geometry)) == True{} : Bool},channel_inside: {Nat.is_lt(channel,Domain.channels(geometry)) == True{} : Bool},
  row_inside: {Nat.is_lt(row,Domain.kernel(geometry)) == True{} : Bool},column_inside: {Nat.is_lt(column,Domain.kernel(geometry)) == True{} : Bool}) ->
  {Spec.weight(problem(geometry,reader,image),output,channel,row,column) ==
    value_at(weight_source(reader),WeightIndex.index(Domain.channel_word(geometry),Domain.kernel_word(geometry),output,channel,row,column)) : F32}:
  Kernel.Geometry{+c,+o,h,w,+k,s,p,gh,gw} = geometry
  Reader{gradient,+weights} = reader
  enumerated_at(Domain.weight_size(geometry),WeightIndex.index(c,k,output,channel,row,column),0n,weights,
    %WeightIndex.elements_four(o,c,k) : {Nat.is_lt(WeightIndex.index(c,k,output,channel,row,column),_) == True{} : Bool}
    WeightIndex.inside(o,c,k,output,channel,row,column,output_inside,channel_inside,row_inside,column_inside))

def gradient(+geometry: Kernel.Geometry,+reader: Reader,+image: Nat,+output: Nat,+row: Nat,+column: Nat,
  output_inside: {Nat.is_lt(output,Domain.outputs(geometry)) == True{} : Bool},row_inside: {Nat.is_lt(row,Domain.gradient_height(geometry)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,Domain.gradient_width(geometry)) == True{} : Bool}) ->
  {ConvSpec.at(RowMajor.index(output,Domain.gradient_height(geometry),Domain.gradient_width(geometry),row,column),Spec.output_gradient(problem(geometry,reader,image))) ==
    value_at(gradient_source(reader),Nat.add(Nat.mul(image,Domain.gradient_size(geometry)),RowMajor.index(output,Domain.gradient_height(geometry),Domain.gradient_width(geometry),row,column))) : F32}:
  Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
  Reader{+gradient,weights} = reader
  enumerated_at(Domain.gradient_size(geometry),RowMajor.index(output,U32.to_nat(gh),U32.to_nat(gw),row,column),Nat.mul(image,Domain.gradient_size(geometry)),gradient,
    RowMajor.in_bounds(output,U32.to_nat(o),U32.to_nat(gh),U32.to_nat(gw),row,column,output_inside,row_inside,column_inside))

# The value index of channel c of image n at an offset of a plane.
def image_index(+image: Nat,+channels: Nat,+area: Nat,+channel: Nat,+offset: Nat) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(image,channels),channel),area),offset) == Nat.add(Nat.mul(image,Nat.mul(channels,area)),Nat.add(Nat.mul(channel,area),offset)) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,channels),channel),area),Nat.add(Nat.mul(Nat.mul(image,channels),area),Nat.mul(channel,area)),RowMajor.distrib(Nat.mul(image,channels),channel,area)) :
    {Nat.add(_,offset) == Nat.add(Nat.mul(image,Nat.mul(channels,area)),Nat.add(Nat.mul(channel,area),offset)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(image,channels),area),Nat.mul(image,Nat.mul(channels,area)),RowMajor.assoc(image,channels,area)) :
    {Nat.add(Nat.add(_,Nat.mul(channel,area)),offset) == Nat.add(Nat.mul(image,Nat.mul(channels,area)),Nat.add(Nat.mul(channel,area),offset)) : Nat}
  Addition.associative(Nat.mul(image,Nat.mul(channels,area)),Nat.mul(channel,area),offset)

# ---- The gradient ----

# The cell of a pixel of channel output of an image.
def pixel_word(+geometry: Kernel.Geometry,+image: U32,+output: Nat,+row: U32,+column: U32) -> U32:
  Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
  ((image * (gh * gw) * Tiled.groups(o) + U32.from_nat(output) / 8 : U32) + (row * gw + column) * Tiled.groups(o) : U32)

# Lane output % 8 of the cell of a pixel, or +0 off the gradient.
def pixel(+geometry: Kernel.Geometry,input: Buffer.Model<Tile.Tile8>,+image: U32,+output: Nat,+row: U32,+column: U32) -> F32:
  Bool.pick(F32,U32.is_lt(row,Domain.gradient_rows(geometry)) && U32.is_lt(column,Domain.gradient_columns(geometry)),
    Cells.lane_of(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),pixel_word(geometry,image,output,row,column)),(U32.from_nat(output) % 8 : U32)),0.0)

# Inside the gradient the lane of the cell of a pixel is the gradient value of
# the problem there.
def pixel_value(+call: Domain.Call,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+image: U32,+output: Nat,+row: U32,+column: U32,
  +image_inside: {Nat.is_lt(U32.to_nat(image),Domain.batch(call)) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Domain.geometry(call))) == True{} : Bool},
  +row_inside: {Nat.is_lt(U32.to_nat(row),Domain.gradient_height(Domain.geometry(call))) == True{} : Bool},
  +column_inside: {Nat.is_lt(U32.to_nat(column),Domain.gradient_width(Domain.geometry(call))) == True{} : Bool}) ->
  {Cells.lane_of(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),pixel_word(Domain.geometry(call),image,output,row,column)),(U32.from_nat(output) % 8 : U32)) ==
    ConvSpec.at(RowMajor.index(output,Domain.gradient_height(Domain.geometry(call)),Domain.gradient_width(Domain.geometry(call)),U32.to_nat(row),U32.to_nat(column)),
      Spec.output_gradient(problem(Domain.geometry(call),models_reader(Domain.geometry(call),input,weights),U32.to_nat(image)))) : F32}:
  Domain.Call{+geometry,+batch,+limits,facts} = call
  Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
  +tree = Buffer.tree(~Tile.Tile8,input)
  +reader = models_reader(geometry,input,weights)
  +n = U32.to_nat(image)
  +outputs = U32.to_nat(o)
  +rows = U32.to_nat(gh)
  +columns = U32.to_nat(gw)
  +area = Nat.mul(rows,columns)
  +area_word = (gh * gw : U32)
  +groups = Tiled.groups(o)
  +offset = Nat.add(Nat.mul(U32.to_nat(row),columns),U32.to_nat(column))
  +capacity = Machine.capacity(limits)
  +lane = (U32.from_nat(output) % 8 : U32)
  +wanted = ConvSpec.at(RowMajor.index(output,rows,columns,U32.to_nat(row),U32.to_nat(column)),Spec.output_gradient(problem(geometry,reader,n)))
  +area_exact = Arithmetic.product_exact(capacity,Limits.capacity_fits(limits),gh,gw,rows,columns,{==},{==},Domain.gradient_area_room(call))
  +offset_inside = {%Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.is_lt(offset,_) == True{} : Bool}
    RowMajor.radix(U32.to_nat(row),rows,U32.to_nat(column),columns,row_inside,column_inside) : {Nat.is_lt(offset,U32.to_nat(area_word)) == True{} : Bool}}
  +room = {%Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.is_le(Nat.mul(Nat.mul(batch,outputs),_),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.mul(batch,outputs),area),Nat.mul(batch,Nat.mul(outputs,area)),RowMajor.assoc(batch,outputs,area)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Domain.gradient_room(call) : {Nat.is_le(Nat.mul(Nat.mul(batch,outputs),U32.to_nat(area_word)),U32.to_nat(capacity)) == True{} : Bool}}
  +local = Nat.add(Nat.mul(Nat.add(Nat.mul(n,outputs),output),U32.to_nat(area_word)),offset)
  +global = Nat.add(Nat.mul(n,Domain.gradient_size(geometry)),RowMajor.index(output,rows,columns,U32.to_nat(row),U32.to_nat(column)))
  +cells_address = TiledValues.cells_address(Grid.Innermost{},0,area_word,groups,n,output,offset)
  +address_same = {
    %Equal.sym(U32,image,U32.from_nat(n),Encoded.encoding(image,n,{==})) :
      {((_ * area_word * groups + U32.from_nat(output) / 8 : U32) + (row * gw + column : U32) * groups : U32) == cells_address : U32}
    %Equal.sym(U32,(row * gw + column : U32),U32.from_nat(offset),Encoded.native_madd(row,gw,column)) :
      {((U32.from_nat(n) * area_word * groups + U32.from_nat(output) / 8 : U32) + _ * groups : U32) == cells_address : U32}
    {==} : {pixel_word(geometry,image,output,row,column) == cells_address : U32}}
  +index_same = {%Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.add(Nat.mul(Nat.add(Nat.mul(n,outputs),output),_),offset) == global : Nat}
    image_index(n,outputs,area,output,offset) : {local == global : Nat}}
  %Equal.sym(U32,pixel_word(geometry,image,output,row,column),cells_address,address_same) : {Cells.lane_of(Tree.read(Tile.Tile8,tree,_),lane) == wanted : F32}
  %TiledValues.cells_lane_within(limits,tree,0,area_word,o,groups,Grid.Innermost{},batch,n,output,offset,room,image_inside,output_inside,offset_inside) : {_ == wanted : F32}
  %Equal.sym(Nat,local,global,index_same) : {TiledValues.value_value(tree,gradient_cells(geometry),U32.from_nat(_)) == wanted : F32}
  Equal.sym(F32,wanted,value_at(gradient_source(reader),global),gradient(geometry,reader,n,output,U32.to_nat(row),U32.to_nat(column),output_inside,row_inside,column_inside))

def pixel_chosen(enabled: Bool,+call: Domain.Call,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+image: U32,+output: Nat,+row: U32,+column: U32,
  +image_inside: {Nat.is_lt(U32.to_nat(image),Domain.batch(call)) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Domain.geometry(call))) == True{} : Bool},
  +same: {U32.is_lt(row,Domain.gradient_rows(Domain.geometry(call))) && U32.is_lt(column,Domain.gradient_columns(Domain.geometry(call))) == enabled : Bool}) ->
  {pixel(Domain.geometry(call),input,image,output,row,column) ==
    ConvSpec.sample_if(enabled,RowMajor.index(output,Domain.gradient_height(Domain.geometry(call)),Domain.gradient_width(Domain.geometry(call)),U32.to_nat(row),U32.to_nat(column)),
      Spec.output_gradient(problem(Domain.geometry(call),models_reader(Domain.geometry(call),input,weights),U32.to_nat(image)))) : F32}:
  match enabled:
    case False{}:
      Domain.Call{+geometry,batch,limits,facts} = call
      Kernel.Geometry{c,o,h,w,k,s,p,+gh,+gw} = geometry
      +value = Cells.lane_of(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),pixel_word(geometry,image,output,row,column)),(U32.from_nat(output) % 8 : U32))
      %Equal.sym(Bool,U32.is_lt(row,gh) && U32.is_lt(column,gw),False{},same) : {Bool.pick(F32,_,value,0.0) == 0.0 : F32} {==}
    case True{}:
      Domain.Call{+geometry,batch,limits,facts} = call
      Kernel.Geometry{c,o,h,w,k,s,p,+gh,+gw} = geometry
      +value = Cells.lane_of(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),pixel_word(geometry,image,output,row,column)),(U32.from_nat(output) % 8 : U32))
      +wanted = ConvSpec.at(RowMajor.index(output,U32.to_nat(gh),U32.to_nat(gw),U32.to_nat(row),U32.to_nat(column)),Spec.output_gradient(problem(geometry,models_reader(geometry,input,weights),U32.to_nat(image))))
      +row_inside = {%Comparison.lt(row,gh) : {_ == True{} : Bool} Boolean.left(U32.is_lt(row,gh),U32.is_lt(column,gw),same) : {Nat.is_lt(U32.to_nat(row),U32.to_nat(gh)) == True{} : Bool}}
      +column_inside = {%Comparison.lt(column,gw) : {_ == True{} : Bool} Boolean.right(U32.is_lt(row,gh),U32.is_lt(column,gw),same) : {Nat.is_lt(U32.to_nat(column),U32.to_nat(gw)) == True{} : Bool}}
      %Equal.sym(Bool,U32.is_lt(row,gh) && U32.is_lt(column,gw),True{},same) : {Bool.pick(F32,_,value,0.0) == wanted : F32}
      pixel_value(call,input,weights,image,output,row,column,image_inside,output_inside,row_inside,column_inside)

def pixel_sample(+call: Domain.Call,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+image: U32,+output: Nat,+row: U32,+column: U32,
  +image_inside: {Nat.is_lt(U32.to_nat(image),Domain.batch(call)) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Domain.geometry(call))) == True{} : Bool}) ->
  {pixel(Domain.geometry(call),input,image,output,row,column) ==
    ConvSpec.sample_if(U32.is_lt(row,Domain.gradient_rows(Domain.geometry(call))) && U32.is_lt(column,Domain.gradient_columns(Domain.geometry(call))),
      RowMajor.index(output,Domain.gradient_height(Domain.geometry(call)),Domain.gradient_width(Domain.geometry(call)),U32.to_nat(row),U32.to_nat(column)),
      Spec.output_gradient(problem(Domain.geometry(call),models_reader(Domain.geometry(call),input,weights),U32.to_nat(image)))) : F32}:
  pixel_chosen(U32.is_lt(row,Domain.gradient_rows(Domain.geometry(call))) && U32.is_lt(column,Domain.gradient_columns(Domain.geometry(call))),call,input,weights,image,output,row,column,
    image_inside,output_inside,{==})

def congruent_pixel(+call: Domain.Call,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+image: U32,+output: Nat,+y: Nat,+x: Nat,+row: Nat,+column: Nat,
  +yq: U32,+xq: U32,+rq: U32,+cq: U32,+ry: Nat,+rx: Nat,
  +rows_room: {Nat.is_le(Nat.add(Domain.gradient_height(Domain.geometry(call)),U32.to_nat(rq)),U32.to_nat(Machine.coordinate(Domain.limits(call)))) == True{} : Bool},
  +columns_room: {Nat.is_le(Nat.add(Domain.gradient_width(Domain.geometry(call)),U32.to_nat(cq)),U32.to_nat(Machine.coordinate(Domain.limits(call)))) == True{} : Bool},
  +image_inside: {Nat.is_lt(U32.to_nat(image),Domain.batch(call)) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Domain.geometry(call))) == True{} : Bool},
  +y_exact: {Nat.add(y,Domain.padding(Domain.geometry(call))) == Nat.add(Nat.mul(U32.to_nat(yq),Domain.stride(Domain.geometry(call))),ry) : Nat},
  +x_exact: {Nat.add(x,Domain.padding(Domain.geometry(call))) == Nat.add(Nat.mul(U32.to_nat(xq),Domain.stride(Domain.geometry(call))),rx) : Nat},
  +row_exact: {row == Nat.add(Nat.mul(U32.to_nat(rq),Domain.stride(Domain.geometry(call))),ry) : Nat},
  +column_exact: {column == Nat.add(Nat.mul(U32.to_nat(cq),Domain.stride(Domain.geometry(call))),rx) : Nat}) ->
  {pixel(Domain.geometry(call),input,image,output,(yq - rq : U32),(xq - cq : U32)) ==
    Spec.sample(problem(Domain.geometry(call),models_reader(Domain.geometry(call),input,weights),U32.to_nat(image)),output,y,x,row,column) : F32}:
  Domain.Call{+geometry,batch,+limits,facts} = call
  Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
  +formal = problem(geometry,models_reader(geometry,input,weights),U32.to_nat(image))
  +values = Spec.output_gradient(formal)
  +read = pixel(geometry,input,image,output,(yq - rq : U32),(xq - cq : U32))
  +native = ConvSpec.sample_if(U32.is_lt((yq - rq : U32),gh) && U32.is_lt((xq - cq : U32),gw),
    RowMajor.index(output,U32.to_nat(gh),U32.to_nat(gw),U32.to_nat((yq - rq : U32)),U32.to_nat((xq - cq : U32))),values)
  +interval = Sources.interval_sample(U32.to_nat(gh),U32.to_nat(gw),output,U32.to_nat(yq),U32.to_nat(xq),U32.to_nat(rq),U32.to_nat(cq),values)
  +wanted = Spec.sample(formal,output,y,x,row,column)
  +same = {
    %Equal.sym(Nat,U32.to_nat(gh),Spec.out_height(formal),Domain.rows_exact(call)) :
      {wanted == Sources.interval_sample(_,U32.to_nat(gw),output,U32.to_nat(yq),U32.to_nat(xq),U32.to_nat(rq),U32.to_nat(cq),values) : F32}
    %Equal.sym(Nat,U32.to_nat(gw),Spec.out_width(formal),Domain.columns_exact(call)) :
      {wanted == Sources.interval_sample(Spec.out_height(formal),_,output,U32.to_nat(yq),U32.to_nat(xq),U32.to_nat(rq),U32.to_nat(cq),values) : F32}
    Sources.radix_problem(formal,output,y,x,row,column,U32.to_nat(yq),U32.to_nat(xq),U32.to_nat(rq),U32.to_nat(cq),ry,rx,Domain.stride_positive(call),y_exact,x_exact,row_exact,column_exact) : {wanted == interval : F32}}
  Equal.trans(F32,read,native,wanted,
    pixel_sample(call,input,weights,image,output,(yq - rq : U32),(xq - cq : U32),image_inside,output_inside),
    Equal.trans(F32,native,interval,wanted,Sources.native_wrapped(values,output,gh,gw,yq,xq,rq,cq,Machine.coordinate(limits),Limits.coordinate_fits(limits),rows_room,columns_room),Equal.sym(F32,wanted,interval,same)))

def phase_input_value(+call: Domain.Call,+first_y: U32,+first_x: U32,+image: U32,+local_row: U32,+local_column: U32,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,
  +output: Nat,+tap_row: Nat,+tap_column: Nat,+lane: Nat,+lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +image_inside: {Nat.is_lt(U32.to_nat(image),Domain.batch(call)) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Domain.geometry(call))) == True{} : Bool},
  +yfacts: PhaseBounds.AxisFacts<first_y,Domain.height_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),local_row,tap_row,Domain.gradient_rows(Domain.geometry(call)),Machine.coordinate(Domain.limits(call))>,
  +xfacts: PhaseBounds.AxisFacts<first_x,Domain.width_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),(local_column + U32.from_nat(lane) : U32),tap_column,Domain.gradient_columns(Domain.geometry(call)),Machine.coordinate(Domain.limits(call))>) ->
  {Tile.element_at(TileLaws.lane_of(lane),Packing.cells_input_tile(Kernel.phase(Domain.geometry(call),first_y,first_x),image,local_row,local_column,input,output,tap_row,tap_column)) ==
    Spec.sample(problem(Domain.geometry(call),models_reader(Domain.geometry(call),input,weights),U32.to_nat(image)),output,
      PhaseBounds.pixel(first_y,Domain.stride_word(Domain.geometry(call)),local_row),PhaseBounds.pixel(first_x,Domain.stride_word(Domain.geometry(call)),(local_column + U32.from_nat(lane) : U32)),
      U32.to_nat(PhaseBounds.tap(first_y,Domain.height_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),tap_row)),
      U32.to_nat(PhaseBounds.tap(first_x,Domain.width_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),tap_column))) : F32}:
  Domain.Call{+geometry,batch,+limits,facts} = call
  Kernel.Geometry{c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  PhaseBounds.AxisFacts{yrin,ypin,+yrad,ytin,+ytrad,+ycounter,+ysroom} = yfacts
  PhaseBounds.AxisFacts{xrin,xpin,+xrad,xtin,+xtrad,+xcounter,+xsroom} = xfacts
  +context = Kernel.phase(geometry,first_y,first_x)
  +x_position = (local_column + U32.from_nat(lane) : U32)
  +y = PhaseBounds.pixel(first_y,s,local_row)
  +x = PhaseBounds.pixel(first_x,s,x_position)
  +ytap = U32.to_nat(PhaseBounds.tap(first_y,h,k,s,p,tap_row))
  +xtap = U32.to_nat(PhaseBounds.tap(first_x,w,k,s,p,tap_column))
  +yq = PhaseBounds.pixel_quotient(first_y,s,p,local_row)
  +xq = PhaseBounds.pixel_quotient(first_x,s,p,x_position)
  +rq = PhaseBounds.tap_quotient(first_y,h,k,s,p,tap_row)
  +cq = PhaseBounds.tap_quotient(first_x,w,k,s,p,tap_column)
  +sample = {Packing.CellsSample{(Packing.cells_base(context,image) + U32.from_nat(output) / 8 : U32),
    (Packing.input_row(context,local_row) + U32.from_nat(tap_row) : U32),(Packing.input_column(context,local_column) + U32.from_nat(tap_column) : U32),
    Packing.gradient_rows(context),Packing.gradient_columns(context),U32.from_nat(output),Packing.gradient_groups(context)} : Packing.CellsSample}
  +formal = problem(geometry,models_reader(geometry,input,weights),U32.to_nat(image))
  +wanted = Spec.sample(formal,output,y,x,ytap,xtap)
  +native = Tile.element_at(TileLaws.lane_of(lane),Packing.cells_input_tile(context,image,local_row,local_column,input,output,tap_row,tap_column))
  Equal.trans(F32,native,Packing.cells_value(input,sample,lane),wanted,
    %TileLaws.lane_value_of(lane,lane_inside) : {native == Packing.cells_value(input,sample,_) : F32}
    Packing.cells_tile_index(input,sample,TileLaws.lane_of(lane)),
    %Equal.sym(U32,(Packing.input_row(context,local_row) + U32.from_nat(tap_row) : U32),(yq - rq : U32),ycounter) :
      {pixel(geometry,input,image,output,_,(Packing.input_column(context,local_column) + U32.from_nat(tap_column) + U32.from_nat(lane) : U32)) == wanted : F32}
    %Equal.sym(U32,(Packing.input_column(context,local_column) + U32.from_nat(tap_column) + U32.from_nat(lane) : U32),(xq - cq : U32),PhaseBounds.column_lane_counter(geometry,first_y,first_x,local_column,lane,tap_column,xcounter)) :
      {pixel(geometry,input,image,output,(yq - rq : U32),_) == wanted : F32}
    congruent_pixel(call,input,weights,image,output,y,x,ytap,xtap,yq,xq,rq,cq,U32.to_nat(Geometry.residue(first_y,s,p)),U32.to_nat(Geometry.residue(first_x,s,p)),ysroom,xsroom,
      image_inside,output_inside,yrad,xrad,ytrad,xtrad))

# ---- The weights ----

def phase_weight_value(+call: Domain.Call,+first_y: U32,+first_x: U32,+image: Nat,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,
  +block: Nat,+output: Nat,+tap_row: Nat,+tap_column: Nat,+lane: Nat,+lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Domain.geometry(call))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),Domain.channels(Domain.geometry(call))) == True{} : Bool},
  +row_inside: {Nat.is_lt(U32.to_nat(PhaseBounds.tap(first_y,Domain.height_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),tap_row)),Domain.kernel(Domain.geometry(call))) == True{} : Bool},
  +column_inside: {Nat.is_lt(U32.to_nat(PhaseBounds.tap(first_x,Domain.width_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),tap_column)),Domain.kernel(Domain.geometry(call))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Weights.cell_tile(weights,Kernel.phase(Domain.geometry(call),first_y,first_x),block,output,tap_row,tap_column)) ==
    Spec.weight(problem(Domain.geometry(call),models_reader(Domain.geometry(call),input,weights),image),output,Nat.add(Nat.mul(block,8n),lane),
      U32.to_nat(PhaseBounds.tap(first_y,Domain.height_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),tap_row)),
      U32.to_nat(PhaseBounds.tap(first_x,Domain.width_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),tap_column))) : F32}:
  Domain.Call{+geometry,batch,+limits,facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,gh,gw} = geometry
  +context = Kernel.phase(geometry,first_y,first_x)
  +reader = models_reader(geometry,input,weights)
  +formal = problem(geometry,reader,image)
  +tree = Buffer.tree(~Tile.Tile8,weights)
  +channel = Nat.add(Nat.mul(block,8n),lane)
  +capacity = Machine.capacity(limits)
  +cap_ok = Limits.capacity_fits(limits)
  +outputs = U32.to_nat(o)
  +channels = U32.to_nat(c)
  +area = Nat.mul(U32.to_nat(k),U32.to_nat(k))
  +area_word = (k * k : U32)
  +outer = (o * area_word : U32)
  +groups = Tiled.groups(c)
  +kr = PhaseBounds.tap(first_y,h,k,s,p,tap_row)
  +kc = PhaseBounds.tap(first_x,w,k,s,p,tap_column)
  +offset = Nat.add(Nat.mul(U32.to_nat(kr),U32.to_nat(k)),U32.to_nat(kc))
  +block_word = U32.from_nat(block)
  +channel_word = U32.from_nat(channel)
  +front = (U32.from_nat(output) * area_word : U32)
  +address = Weights.cell_address(context,block,output,tap_row,tap_column)
  +cells_address = TiledValues.cells_address(Grid.Outermost{},outer,area_word,groups,output,channel,offset)
  +weight_index = WeightIndex.index(c,k,output,channel,U32.to_nat(kr),U32.to_nat(kc))
  +prefix = Nat.add(Nat.mul(output,channels),channel)
  +local = Nat.add(Nat.mul(prefix,U32.to_nat(area_word)),offset)
  +wanted = Spec.weight(formal,output,channel,U32.to_nat(kr),U32.to_nat(kc))
  +area_exact = Arithmetic.product_exact(capacity,cap_ok,k,k,U32.to_nat(k),U32.to_nat(k),{==},{==},Domain.kernel_area_room(call))
  +offset_inside = {%Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.is_lt(offset,_) == True{} : Bool}
    RowMajor.radix(U32.to_nat(kr),U32.to_nat(k),U32.to_nat(kc),U32.to_nat(k),row_inside,column_inside) : {Nat.is_lt(offset,U32.to_nat(area_word)) == True{} : Bool}}
  +room = {%Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.is_le(Nat.mul(Nat.mul(outputs,channels),_),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.mul(outputs,channels),area),Nat.mul(outputs,Nat.mul(channels,area)),RowMajor.assoc(outputs,channels,area)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Domain.weight_room(call) : {Nat.is_le(Nat.mul(Nat.mul(outputs,channels),U32.to_nat(area_word)),U32.to_nat(capacity)) == True{} : Bool}}
  +channel_room = Order.transitive(channel,channels,U32.to_nat(capacity),Bounds.strict_le(channel,channels,channel_inside),Domain.channels_capacity_room(call))
  +channel_exact = Bounds.roundtrip(channel,capacity,cap_ok,channel_room)
  +group_nat = Equal.trans(Nat,U32.to_nat((channel_word / 8 : U32)),Nat.div(U32.to_nat(channel_word),8n),block,
    NativeDivision.divide(channel_word,8,16,{==},{==},{==}),
    %Equal.sym(Nat,U32.to_nat(channel_word),channel,channel_exact) : {Nat.div(_,8n) == block : Nat} Radix.quotient(block,lane,8n,{==},lane_inside))
  +lane_nat = Equal.trans(Nat,U32.to_nat((channel_word % 8 : U32)),Nat.mod(U32.to_nat(channel_word),8n),lane,
    NativeDivision.modulo(channel_word,8,16,{==},{==},{==}),
    %Equal.sym(Nat,U32.to_nat(channel_word),channel,channel_exact) : {Nat.mod(_,8n) == lane : Nat} Radix.remainder(block,lane,8n,{==},lane_inside))
  +group_word = Arithmetic.from_nat_exact((channel_word / 8 : U32),block,group_nat)
  +lane_word = Equal.sym(U32,U32.from_nat(lane),(channel_word % 8 : U32),Arithmetic.from_nat_exact((channel_word % 8 : U32),lane,lane_nat))
  +address_same = {
    %group_word : {address == ((_ * outer + front : U32) + U32.from_nat(offset) * 1 : U32) : U32}
    %Arithmetic.multiply_associative(o,k,k) : {address == ((block_word * _ + front : U32) + U32.from_nat(offset) * 1 : U32) : U32}
    %Equal.sym(U32,(U32.from_nat(offset) * 1 : U32),U32.from_nat(offset),Arithmetic.multiply_one(U32.from_nat(offset))) :
      {address == ((block_word * (o * k * k : U32) + front : U32) + _ : U32) : U32}
    %Encoded.native_madd(kr,k,kc) : {address == ((block_word * (o * k * k : U32) + front : U32) + _ : U32) : U32}
    %Arithmetic.add_commutative(front,(block_word * (o * k * k : U32) : U32)) : {address == (_ + (kr * k + kc : U32) : U32) : U32}
    Arithmetic.add_associative((front + block_word * (o * k * k : U32) : U32),(kr * k : U32),kc) : {address == cells_address : U32}}
  +index_same = {%Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.add(Nat.mul(prefix,_),offset) == weight_index : Nat}
    Equal.sym(Nat,weight_index,Nat.add(Nat.mul(prefix,area),offset),RowMajor.mixed(prefix,U32.to_nat(k),U32.to_nat(k),U32.to_nat(kr),U32.to_nat(kc))) : {local == weight_index : Nat}}
  %Equal.sym(U32,address,cells_address,address_same) : {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,_)) == wanted : F32}
  %TiledValues.lane_selected(Tree.read(Tile.Tile8,tree,cells_address),lane,lane_inside) : {_ == wanted : F32}
  %lane_word : {Cells.lane_of(Tree.read(Tile.Tile8,tree,cells_address),_) == wanted : F32}
  %TiledValues.cells_lane_within(limits,tree,outer,area_word,c,groups,Grid.Outermost{},outputs,output,channel,offset,room,output_inside,channel_inside,offset_inside) : {_ == wanted : F32}
  %Equal.sym(Nat,local,weight_index,index_same) : {TiledValues.value_value(tree,weight_cells(geometry),U32.from_nat(_)) == wanted : F32}
  Equal.sym(F32,wanted,value_at(weight_source(reader),weight_index),weight(geometry,reader,image,output,channel,U32.to_nat(kr),U32.to_nat(kc),output_inside,channel_inside,row_inside,column_inside))

# ---- The products of a phase ----

def phase_flat(+geometry: Kernel.Geometry,+first_y: U32,+first_x: U32,+position_y: U32,+position_x: U32,+image: U32,channel: Nat,input: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>) -> StridedOrder.Flat:
  StridedOrder.Flat{problem(geometry,models_reader(geometry,input,weights),U32.to_nat(image)),channel,PhaseBounds.pixel(first_y,Domain.stride_word(geometry),position_y),PhaseBounds.pixel(first_x,Domain.stride_word(geometry),position_x)}

def phase_spec_count(+call: Domain.Call,+first_y: U32,+first_x: U32,+position_y: U32,+position_x: U32,+image: U32,+channel: Nat,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,
  +first_y_inside: {Nat.is_lt(U32.to_nat(first_y),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +first_x_inside: {Nat.is_lt(U32.to_nat(first_x),Domain.width(Domain.geometry(call))) == True{} : Bool}) ->
  {U32.to_nat(Kernel.phase_inner(Kernel.phase(Domain.geometry(call),first_y,first_x))) == StridedOrder.flat_count(phase_flat(Domain.geometry(call),first_y,first_x,position_y,position_x,image,channel,input,weights)) : Nat}:
  Domain.Call{+geometry,batch,+limits,facts} = call
  Kernel.Geometry{c,+o,+h,+w,+k,+s,+p,gh,gw} = geometry
  +context = Kernel.phase(geometry,first_y,first_x)
  +flat = phase_flat(geometry,first_y,first_x,position_y,position_x,image,channel,input,weights)
  +cap = Machine.coordinate(limits)
  +cap_ok = Limits.coordinate_fits(limits)
  +first_y_padded = Order.transitive(Nat.add(U32.to_nat(first_y),U32.to_nat(p)),Nat.add(U32.to_nat(h),U32.to_nat(p)),U32.to_nat(cap),Order.add_le(U32.to_nat(first_y),U32.to_nat(h),U32.to_nat(p),Bounds.strict_le(U32.to_nat(first_y),U32.to_nat(h),first_y_inside)),PhaseBounds.padded_height_room(call))
  +first_x_padded = Order.transitive(Nat.add(U32.to_nat(first_x),U32.to_nat(p)),Nat.add(U32.to_nat(w),U32.to_nat(p)),U32.to_nat(cap),Order.add_le(U32.to_nat(first_x),U32.to_nat(w),U32.to_nat(p),Bounds.strict_le(U32.to_nat(first_x),U32.to_nat(w),first_x_inside)),PhaseBounds.padded_width_room(call))
  +rows = Sources.axis_spec_count(first_y,h,k,s,p,U32.to_nat(position_y),cap,cap_ok,first_y_padded,PhaseBounds.kernel_coordinate_room(call),PhaseBounds.stride_double_room(call),Domain.stride_positive(call))
  +columns = Sources.axis_spec_count(first_x,w,k,s,p,U32.to_nat(position_x),cap,cap_ok,first_x_padded,PhaseBounds.kernel_coordinate_room(call),PhaseBounds.stride_double_room(call),Domain.stride_positive(call))
  Equal.trans(Nat,U32.to_nat(Kernel.phase_inner(context)),Nat.mul(U32.to_nat(o),Packing.phase_area(context)),StridedOrder.flat_count(flat),PhaseBounds.phase_inner_exact(call,first_y,first_x),
    Equal.trans(Nat,Nat.mul(U32.to_nat(o),Packing.phase_area(context)),Nat.mul(U32.to_nat(o),Nat.mul(StridedOrder.flat_rows(flat),U32.to_nat(Packing.phase_columns(context)))),StridedOrder.flat_count(flat),
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(o),Nat.mul(value,U32.to_nat(Packing.phase_columns(context)))),U32.to_nat(Packing.phase_rows(context)),StridedOrder.flat_rows(flat),rows),
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(o),Nat.mul(StridedOrder.flat_rows(flat),value)),U32.to_nat(Packing.phase_columns(context)),StridedOrder.flat_columns(flat),columns)))

def phase_product(+call: Domain.Call,+first_y: U32,+first_x: U32,+image: U32,+block: Nat,+local_row: U32,+local_column: U32,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,
  +output: Nat,+tap_row: Nat,+tap_column: Nat,+row: Nat,+lane: Nat,
  +image_inside: {Nat.is_lt(U32.to_nat(image),Domain.batch(call)) == True{} : Bool},
  +row_small: {Nat.is_lt(row,8n) == True{} : Bool},+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Domain.geometry(call))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(Nat.mul(block,8n),row),Domain.channels(Domain.geometry(call))) == True{} : Bool},
  +first_y_inside: {Nat.is_lt(U32.to_nat(first_y),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +first_x_inside: {Nat.is_lt(U32.to_nat(first_x),Domain.width(Domain.geometry(call))) == True{} : Bool},
  +row_inside: {Nat.is_lt(tap_row,U32.to_nat(Packing.phase_rows(Kernel.phase(Domain.geometry(call),first_y,first_x)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(tap_column,U32.to_nat(Packing.phase_columns(Kernel.phase(Domain.geometry(call),first_y,first_x)))) == True{} : Bool},
  +yfacts: PhaseBounds.AxisFacts<first_y,Domain.height_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),local_row,tap_row,Domain.gradient_rows(Domain.geometry(call)),Machine.coordinate(Domain.limits(call))>,
  +xfacts: PhaseBounds.AxisFacts<first_x,Domain.width_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),(local_column + U32.from_nat(lane) : U32),tap_column,Domain.gradient_columns(Domain.geometry(call)),Machine.coordinate(Domain.limits(call))>) ->
  {ConvSpec.Product{Tile.element_at(TileLaws.lane_of(lane),Packing.cells_input_tile(Kernel.phase(Domain.geometry(call),first_y,first_x),image,local_row,local_column,input,output,tap_row,tap_column)),
      TileLaws.select_lane(row,Weights.cell_tile(weights,Kernel.phase(Domain.geometry(call),first_y,first_x),block,output,tap_row,tap_column))} ==
    StridedOrder.flat_product(phase_flat(Domain.geometry(call),first_y,first_x,local_row,(local_column + U32.from_nat(lane) : U32),image,Nat.add(Nat.mul(block,8n),row),input,weights),
      Packing.phase_index(Kernel.phase(Domain.geometry(call),first_y,first_x),output,tap_row,tap_column)) : ConvSpec.Product}:
  Domain.Call{+geometry,batch,+limits,facts} = call
  Kernel.Geometry{c,o,+h,+w,+k,+s,+p,gh,gw} = geometry
  PhaseBounds.AxisFacts{yrin,ypin,yrad,+ytin,ytrad,ycounter,ysroom} = yfacts
  PhaseBounds.AxisFacts{xrin,xpin,xrad,+xtin,xtrad,xcounter,xsroom} = xfacts
  +context = Kernel.phase(geometry,first_y,first_x)
  +x_position = (local_column + U32.from_nat(lane) : U32)
  +channel = Nat.add(Nat.mul(block,8n),row)
  +flat = phase_flat(geometry,first_y,first_x,local_row,x_position,image,channel,input,weights)
  +formal = problem(geometry,models_reader(geometry,input,weights),U32.to_nat(image))
  +y = PhaseBounds.pixel(first_y,s,local_row)
  +x = PhaseBounds.pixel(first_x,s,x_position)
  +ytap = U32.to_nat(PhaseBounds.tap(first_y,h,k,s,p,tap_row))
  +xtap = U32.to_nat(PhaseBounds.tap(first_x,w,k,s,p,tap_column))
  +cap = Machine.coordinate(limits)
  +cap_ok = Limits.coordinate_fits(limits)
  +first_y_padded = Order.transitive(Nat.add(U32.to_nat(first_y),U32.to_nat(p)),Nat.add(U32.to_nat(h),U32.to_nat(p)),U32.to_nat(cap),Order.add_le(U32.to_nat(first_y),U32.to_nat(h),U32.to_nat(p),Bounds.strict_le(U32.to_nat(first_y),U32.to_nat(h),first_y_inside)),PhaseBounds.padded_height_room(call))
  +first_x_padded = Order.transitive(Nat.add(U32.to_nat(first_x),U32.to_nat(p)),Nat.add(U32.to_nat(w),U32.to_nat(p)),U32.to_nat(cap),Order.add_le(U32.to_nat(first_x),U32.to_nat(w),U32.to_nat(p),Bounds.strict_le(U32.to_nat(first_x),U32.to_nat(w),first_x_inside)),PhaseBounds.padded_width_room(call))
  +rows = Sources.axis_spec_count(first_y,h,k,s,p,U32.to_nat(local_row),cap,cap_ok,first_y_padded,PhaseBounds.kernel_coordinate_room(call),PhaseBounds.stride_double_room(call),Domain.stride_positive(call))
  +columns = Sources.axis_spec_count(first_x,w,k,s,p,U32.to_nat(x_position),cap,cap_ok,first_x_padded,PhaseBounds.kernel_coordinate_room(call),PhaseBounds.stride_double_room(call),Domain.stride_positive(call))
  +row_tap = Sources.axis_spec_tap(first_y,h,k,s,p,U32.to_nat(local_row),tap_row,cap,cap_ok,first_y_padded,PhaseBounds.kernel_coordinate_room(call),PhaseBounds.stride_double_room(call),Domain.stride_positive(call),row_inside)
  +column_tap = Sources.axis_spec_tap(first_x,w,k,s,p,U32.to_nat(x_position),tap_column,cap,cap_ok,first_x_padded,PhaseBounds.kernel_coordinate_room(call),PhaseBounds.stride_double_room(call),Domain.stride_positive(call),column_inside)
  +native_input = Tile.element_at(TileLaws.lane_of(lane),Packing.cells_input_tile(context,image,local_row,local_column,input,output,tap_row,tap_column))
  +native_weight = TileLaws.select_lane(row,Weights.cell_tile(weights,context,block,output,tap_row,tap_column))
  +input_value = Spec.sample(formal,output,y,x,ytap,xtap)
  +weight_value = Spec.weight(formal,output,channel,ytap,xtap)
  +selected = {ConvSpec.Product{input_value,weight_value} : ConvSpec.Product}
  +specified = {ConvSpec.Product{Spec.sample(formal,output,y,x,StridedOrder.flat_row_tap(flat,tap_row),StridedOrder.flat_column_tap(flat,tap_column)),
    Spec.weight(formal,output,channel,StridedOrder.flat_row_tap(flat,tap_row),StridedOrder.flat_column_tap(flat,tap_column))} : ConvSpec.Product}
  +index = Packing.phase_index(context,output,tap_row,tap_column)
  +index_exact = Equal.trans(Nat,index,RowMajor.index(output,StridedOrder.flat_rows(flat),U32.to_nat(Packing.phase_columns(context)),tap_row,tap_column),StridedOrder.flat_index(flat,output,tap_row,tap_column),
    Equal.cong(Nat,Nat,value => RowMajor.index(output,value,U32.to_nat(Packing.phase_columns(context)),tap_row,tap_column),U32.to_nat(Packing.phase_rows(context)),StridedOrder.flat_rows(flat),rows),
    Equal.cong(Nat,Nat,value => RowMajor.index(output,StridedOrder.flat_rows(flat),value,tap_row,tap_column),U32.to_nat(Packing.phase_columns(context)),StridedOrder.flat_columns(flat),columns))
  Equal.trans(ConvSpec.Product,ConvSpec.Product{native_input,native_weight},selected,StridedOrder.flat_product(flat,index),
    Equal.trans(ConvSpec.Product,ConvSpec.Product{native_input,native_weight},ConvSpec.Product{input_value,native_weight},selected,
      Equal.cong(F32,ConvSpec.Product,value => ConvSpec.Product{value,native_weight},native_input,input_value,phase_input_value(call,first_y,first_x,image,local_row,local_column,input,weights,output,tap_row,tap_column,lane,lane_small,image_inside,output_inside,yfacts,xfacts)),
      Equal.cong(F32,ConvSpec.Product,value => ConvSpec.Product{input_value,value},native_weight,weight_value,phase_weight_value(call,first_y,first_x,U32.to_nat(image),input,weights,block,output,tap_row,tap_column,row,row_small,output_inside,channel_inside,ytin,xtin))),
    Equal.trans(ConvSpec.Product,selected,specified,StridedOrder.flat_product(flat,index),
      %Equal.sym(Nat,ytap,StridedOrder.flat_row_tap(flat,tap_row),row_tap) : {ConvSpec.Product{Spec.sample(formal,output,y,x,_,xtap),Spec.weight(formal,output,channel,_,xtap)} == specified : ConvSpec.Product}
      %Equal.sym(Nat,xtap,StridedOrder.flat_column_tap(flat,tap_column),column_tap) : {ConvSpec.Product{Spec.sample(formal,output,y,x,StridedOrder.flat_row_tap(flat,tap_row),_),Spec.weight(formal,output,channel,StridedOrder.flat_row_tap(flat,tap_row),_)} == specified : ConvSpec.Product} {==},
      %Equal.sym(Nat,index,StridedOrder.flat_index(flat,output,tap_row,tap_column),index_exact) : {specified == StridedOrder.flat_product(flat,_) : ConvSpec.Product}
      Equal.sym(ConvSpec.Product,StridedOrder.flat_product(flat,StridedOrder.flat_index(flat,output,tap_row,tap_column)),specified,
        StridedOrder.flat_product_at(flat,output,tap_row,tap_column,%rows : {Nat.is_lt(tap_row,_) == True{} : Bool} row_inside,%columns : {Nat.is_lt(tap_column,_) == True{} : Bool} column_inside))))
