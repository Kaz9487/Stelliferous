# The bias gradient of a tensor of cells: the run holds the pieces of the
# output gradient while it reads them, gives them back as a tensor of its
# contract, and every block of the result holds the cell of the bias model.

import Base
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_move.bend as Move
import ../tensor_tiled.bend as Tiled
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../storage_buffer.bend as Storage
import ../default_machine_limits.bend as Default
import ../tensor_parameter_gradient.bend as ParameterGradient
import ../kernel_parameter_gradient.bend as Kernel
import ./tensor_parameter_gradient_cells.bend as Cells
import ./kernel_bias_gradient.bend as Bias
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_move_readback.bend as Held
import ./tensor_halo_windows.bend as Windows
import ./tensor_released.bend as Released
import ./tensor_storage_count.bend as StorageCount
import ./storage_buffer_model.bend as Buffer
import ./encoded_coordinates.bend as Encoded
import ./nat_ceiling.bend as Ceiling
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./boolean_logic.bend as Logic
import ./u32_comparison.bend as Comparison

# ---- A call on cells ----

# A bias call on cells d that name its images, its lanes within limits whose
# capacity is the default capacity, accepted by the runtime.
type BiasCall is Data:
  BiasCall{limits: Machine.Limits,default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
    d_cells: List<&2,U32>,d_lanes: U32,di: ParameterGradient.ImageGeometry,
    d_named: {ParameterGradient.cell_image(d_cells,d_lanes) == Some{di} : Maybe<&2,ParameterGradient.ImageGeometry>},d_counted: Global.Counted<d_cells>,
    lanes_room: {Nat.is_le(U32.to_nat(d_lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},g: U32,
    accepted: {ParameterGradient.positive(di) && U32.is_lt(0,g) == True{} : Bool}}

def limits_of(call: BiasCall) -> Machine.Limits:
  BiasCall{limits,default_capacity,d_cells,d_lanes,di,d_named,d_counted,lanes_room,g,accepted} = call
  limits

def default_capacity_of(+call: BiasCall) -> {Machine.capacity(limits_of(call)) == Machine.capacity(Default.limits()) : U32}:
  BiasCall{limits,+default_capacity,d_cells,d_lanes,di,d_named,d_counted,lanes_room,g,accepted} = call
  default_capacity

def d_cells_of(call: BiasCall) -> List<&2,U32>:
  BiasCall{limits,default_capacity,d_cells,d_lanes,di,d_named,d_counted,lanes_room,g,accepted} = call
  d_cells

def d_lanes_of(call: BiasCall) -> U32:
  BiasCall{limits,default_capacity,d_cells,d_lanes,di,d_named,d_counted,lanes_room,g,accepted} = call
  d_lanes

def image_of(call: BiasCall) -> ParameterGradient.ImageGeometry:
  BiasCall{limits,default_capacity,d_cells,d_lanes,di,d_named,d_counted,lanes_room,g,accepted} = call
  di

def d_counted_of(+call: BiasCall) -> Global.Counted<d_cells_of(call)>:
  BiasCall{limits,default_capacity,d_cells,d_lanes,di,d_named,+d_counted,lanes_room,g,accepted} = call
  d_counted

def group_of(call: BiasCall) -> U32:
  BiasCall{limits,default_capacity,d_cells,d_lanes,di,d_named,d_counted,lanes_room,g,accepted} = call
  g

def outputs_of(+call: BiasCall) -> U32:
  Cells.image_channels(image_of(call))

def groups_of(+call: BiasCall) -> U32:
  Tiled.groups(outputs_of(call))

def work_of(+call: BiasCall) -> Kernel.Work:
  ParameterGradient.bias_work(image_of(call),group_of(call))

def layout_of(+call: BiasCall) -> Held.Spans:
  Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(d_cells_of(call))),Held.Ended{})

# The cells the bias writes into a fresh buffer of the groups of its outputs.
def written_of(~D: Data,~dv: D -> U32 -> Tile.Tile8,+d_context: D,+call: BiasCall) -> Buffer.Model<Tile.Tile8>:
  Bias.bias_written(~D,~dv,work_of(call),d_context,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call))))

# ---- What the kernel needs of a call ----

# The cells of the images of a call hold the groups of every output at every
# position of every image.
def bias_cells(+n: U32,+o: U32,+rows: U32,+columns: U32,+shape: List<&2,U32>,+lanes: U32,+limits: Machine.Limits,
  +cells: Cells.ImageCells<shape,lanes,ParameterGradient.ImageGeometry{n,o,rows,columns}>,
  +lanes_room: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Kernel.groups(U32.to_nat(o))),ShapeLaws.elements(shape)) == True{} : Bool}:
  Cells.ImageCells{+channels,+elements} = cells
  +nn = U32.to_nat(n)
  +rn = U32.to_nat(rows)
  +cn = U32.to_nat(columns)
  +groups = U32.to_nat(Grid.groups(lanes))
  +positions = Nat.mul(nn,Nat.mul(rn,cn))
  +blocks_exact = Equal.trans(Nat,Kernel.groups(U32.to_nat(o)),Ceiling.ceiling(U32.to_nat(o),8n),groups,
    Ceiling.shifted_formula(U32.to_nat(o),8n,{==}),
    Equal.trans(Nat,Ceiling.ceiling(U32.to_nat(o),8n),Ceiling.ceiling(U32.to_nat(lanes),8n),groups,
      Equal.cong(U32,Nat,value => Ceiling.ceiling(U32.to_nat(value),8n),o,lanes,channels),
      Equal.sym(Nat,groups,Ceiling.ceiling(U32.to_nat(lanes),8n),Encoded.eight_blocks_exact(limits,lanes,lanes_room))))
  +same = Equal.trans(Nat,Nat.mul(positions,Kernel.groups(U32.to_nat(o))),Nat.mul(positions,groups),Nat.mul(nn,Nat.mul(rn,Nat.mul(cn,groups))),
    Equal.cong(Nat,Nat,value => Nat.mul(positions,value),Kernel.groups(U32.to_nat(o)),groups,blocks_exact),
    Equal.trans(Nat,Nat.mul(positions,groups),Nat.mul(nn,Nat.mul(Nat.mul(rn,cn),groups)),Nat.mul(nn,Nat.mul(rn,Nat.mul(cn,groups))),
      RowMajor.assoc(nn,Nat.mul(rn,cn),groups),
      Equal.cong(Nat,Nat,value => Nat.mul(nn,value),Nat.mul(Nat.mul(rn,cn),groups),Nat.mul(rn,Nat.mul(cn,groups)),RowMajor.assoc(rn,cn,groups))))
  +exact = Equal.trans(Nat,Nat.mul(positions,Kernel.groups(U32.to_nat(o))),Nat.mul(nn,Nat.mul(rn,Nat.mul(cn,groups))),ShapeLaws.elements(shape),
    same,Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(nn,Nat.mul(rn,Nat.mul(cn,groups))),elements))
  %Equal.sym(Nat,Nat.mul(positions,Kernel.groups(U32.to_nat(o))),ShapeLaws.elements(shape),exact) : {Nat.is_le(_,ShapeLaws.elements(shape)) == True{} : Bool}
  Order.reflexive(ShapeLaws.elements(shape))

# The holding of the cells of d of a call is ready for the bias kernel.
def gradient_of(+call: BiasCall) -> Bias.Gradient<work_of(call),ShapeLaws.elements(d_cells_of(call))>:
  BiasCall{+limits,+default_capacity,+d_cells,+d_lanes,+di,+d_named,+d_counted,+lanes_room,+g,+accepted} = call
  ParameterGradient.ImageGeometry{+n,+o,+rows,+columns} = di
  +image = {ParameterGradient.ImageGeometry{n,o,rows,columns} : ParameterGradient.ImageGeometry}
  +sizes = Cells.image_sizes(image,Logic.left(ParameterGradient.positive(image),U32.is_lt(0,g),accepted))
  Bias.Gradient{limits,StorageCount.counted_room(limits,d_cells,d_counted,default_capacity),
    bias_cells(n,o,rows,columns,d_cells,d_lanes,limits,Cells.image_cells(d_cells,d_lanes,image,d_named),lanes_room),
    Cells.sizes_width(image,sizes),
    %Comparison.lt(0,g) : {_ == True{} : Bool} Logic.right(ParameterGradient.positive(image),U32.is_lt(0,g),accepted)}

# The cells of d of a call are not empty.
def nonempty_of(+call: BiasCall) -> {Nat.is_lt(0n,ShapeLaws.elements(d_cells_of(call))) == True{} : Bool}:
  BiasCall{+limits,+default_capacity,+d_cells,+d_lanes,+di,+d_named,+d_counted,+lanes_room,+g,+accepted} = call
  Cells.cells_nonempty(d_cells,d_lanes,di,Cells.image_cells(d_cells,d_lanes,di,d_named),
    Cells.image_sizes(di,Logic.left(ParameterGradient.positive(di),U32.is_lt(0,g),accepted)),limits,lanes_room)

# The outputs of a call fit the capacity.
def outputs_room(+call: BiasCall) -> {Nat.is_le(U32.to_nat(outputs_of(call)),U32.to_nat(Machine.capacity(limits_of(call)))) == True{} : Bool}:
  BiasCall{+limits,+default_capacity,+d_cells,+d_lanes,+di,+d_named,+d_counted,+lanes_room,+g,+accepted} = call
  %Equal.sym(U32,Cells.image_channels(di),d_lanes,Cells.cells_channels(d_cells,d_lanes,di,Cells.image_cells(d_cells,d_lanes,di,d_named))) :
    {Nat.is_le(U32.to_nat(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  lanes_room

# The groups of outputs within the capacity are the groups the kernel writes.
def groups_from(+di: ParameterGradient.ImageGeometry,+g: U32,+limits: Machine.Limits,
  +room: {Nat.is_le(U32.to_nat(Cells.image_channels(di)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Bias.work_groups(ParameterGradient.bias_work(di,g)) == U32.to_nat(Tiled.groups(Cells.image_channels(di))) : Nat}:
  ParameterGradient.ImageGeometry{+n,+o,+rows,+columns} = di
  Equal.trans(Nat,Kernel.groups(U32.to_nat(o)),Ceiling.ceiling(U32.to_nat(o),8n),U32.to_nat(Tiled.groups(o)),
    Ceiling.shifted_formula(U32.to_nat(o),8n,{==}),
    Equal.sym(Nat,U32.to_nat(Tiled.groups(o)),Ceiling.ceiling(U32.to_nat(o),8n),Encoded.eight_blocks_exact(limits,o,room)))

def groups_exact(+call: BiasCall) -> {Bias.work_groups(work_of(call)) == U32.to_nat(groups_of(call)) : Nat}:
  groups_from(image_of(call),group_of(call),limits_of(call),outputs_room(call))

# The groups of the outputs of a call fit the capacity.
def groups_room(+call: BiasCall) -> {Nat.is_le(U32.to_nat(groups_of(call)),U32.to_nat(Machine.capacity(limits_of(call)))) == True{} : Bool}:
  +outputs = U32.to_nat(outputs_of(call))
  +cap = U32.to_nat(Machine.capacity(limits_of(call)))
  %Equal.sym(Nat,U32.to_nat(groups_of(call)),Ceiling.ceiling(outputs,8n),Encoded.eight_blocks_exact(limits_of(call),outputs_of(call),outputs_room(call))) :
    {Nat.is_le(_,cap) == True{} : Bool}
  Order.transitive(Ceiling.ceiling(outputs,8n),outputs,cap,Ceiling.bounded(outputs,8n,{==}),outputs_room(call))

def groups_within(+call: BiasCall) -> {U32.is_le(groups_of(call),Machine.capacity(Default.limits())) == True{} : Bool}:
  %default_capacity_of(call) : {U32.is_le(groups_of(call),_) == True{} : Bool}
  %Equal.sym(Bool,U32.is_le(groups_of(call),Machine.capacity(limits_of(call))),Nat.is_le(U32.to_nat(groups_of(call)),U32.to_nat(Machine.capacity(limits_of(call)))),
      Comparison.le(groups_of(call),Machine.capacity(limits_of(call)))) : {_ == True{} : Bool}
  groups_room(call)

# ---- The run of a call ----

# The pieces of d of a call cover its cells.
def covered_of(~D: Data,~dv: D -> U32 -> Tile.Tile8,+d_context: D,+call: BiasCall,-dp: Pieces.Pieces(~Tile.Tile8),+df: Tile.Tile8,+dw: U32,
  +shaped: Global.Shaped(~Tile.Tile8,~D,~dv,d_context,d_cells_of(call),Tensors.Tensor{dp,d_cells_of(call),df,Tensors.Usable{},dw})) ->
  Held.Covered<Tile.Tile8,D,dv,d_context,Move.held(~Tile.Tile8,dp),0n,ShapeLaws.elements(d_cells_of(call)),layout_of(call)>:
  Windows.shaped_covered(~Tile.Tile8,~D,~dv,d_context,d_cells_of(call),dp,df,Tensors.Usable{},dw,shaped)

# The bias kernel over the pieces of d writes the cells of the bias model and
# gives back a holding that still covers d.
def kept_of(~D: Data,~dv: D -> U32 -> Tile.Tile8,+d_context: D,+call: BiasCall,-dp: Pieces.Pieces(~Tile.Tile8),+df: Tile.Tile8,+dw: U32,
  +shaped: Global.Shaped(~Tile.Tile8,~D,~dv,d_context,d_cells_of(call),Tensors.Tensor{dp,d_cells_of(call),df,Tensors.Usable{},dw})) ->
  Bias.Kept<D,dv,d_context,ShapeLaws.elements(d_cells_of(call)),layout_of(call),Storage.Buffer<Tile.Tile8>,
    Kernel.bias(work_of(call),Move.held(~Tile.Tile8,dp),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call))),
    Buffer.pack(~Tile.Tile8,written_of(~D,~dv,d_context,call))>:
  %Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call))) :
    Bias.Kept<D,dv,d_context,ShapeLaws.elements(d_cells_of(call)),layout_of(call),Storage.Buffer<Tile.Tile8>,
      Kernel.bias(work_of(call),Move.held(~Tile.Tile8,dp),_),Buffer.pack(~Tile.Tile8,written_of(~D,~dv,d_context,call))>
  Bias.bias_done(~D,~dv,d_context,work_of(call),ShapeLaws.elements(d_cells_of(call)),layout_of(call),gradient_of(call),Move.held(~Tile.Tile8,dp),
    covered_of(~D,~dv,d_context,call,dp,df,dw,shaped),Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call))))

# The cells a run gives are the cells of the bias model.
def values_exact(~D: Data,~dv: D -> U32 -> Tile.Tile8,+d_context: D,+call: BiasCall,-dp: Pieces.Pieces(~Tile.Tile8),+df: Tile.Tile8,+dw: U32,
  +shaped: Global.Shaped(~Tile.Tile8,~D,~dv,d_context,d_cells_of(call),Tensors.Tensor{dp,d_cells_of(call),df,Tensors.Usable{},dw})) ->
  {Pair.snd(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Kernel.bias(work_of(call),Move.held(~Tile.Tile8,dp),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call)))) ==
    Buffer.pack(~Tile.Tile8,written_of(~D,~dv,d_context,call)) : Storage.Buffer<Tile.Tile8>}:
  Equal.cong(Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,pair => Pair.snd(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,pair),
    Kernel.bias(work_of(call),Move.held(~Tile.Tile8,dp),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call))),(Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Kernel.bias(work_of(call),Move.held(~Tile.Tile8,dp),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call)))),Buffer.pack(~Tile.Tile8,written_of(~D,~dv,d_context,call))),
    Bias.kept_split(~D,~dv,~Storage.Buffer<Tile.Tile8>,d_context,ShapeLaws.elements(d_cells_of(call)),layout_of(call),Kernel.bias(work_of(call),Move.held(~Tile.Tile8,dp),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call))),
      Buffer.pack(~Tile.Tile8,written_of(~D,~dv,d_context,call)),kept_of(~D,~dv,d_context,call,dp,df,dw,shaped)))

# The output gradient a run gives back, placed in its grid, is a tensor of the
# contract of d.
def d_back(~D: Data,~dv: D -> U32 -> Tile.Tile8,+d_context: D,+call: BiasCall,-dp: Pieces.Pieces(~Tile.Tile8),+df: Tile.Tile8,+dw: U32,
  +shaped: Global.Shaped(~Tile.Tile8,~D,~dv,d_context,d_cells_of(call),Tensors.Tensor{dp,d_cells_of(call),df,Tensors.Usable{},dw})) ->
  Global.Contract(~Tile.Tile8,~D,~dv,d_context,ParameterGradient.back(ParameterGradient.Read{d_cells_of(call),df,Tensors.Usable{},dw},
    Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
      Kernel.bias(work_of(call),Move.held(~Tile.Tile8,dp),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call)))))):
  Released.released_counted(~Tile.Tile8,~D,~dv,d_context,limits_of(call),d_cells_of(call),df,dw,Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Kernel.bias(work_of(call),Move.held(~Tile.Tile8,dp),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call)))),
    d_counted_of(call),StorageCount.counted_room(limits_of(call),d_cells_of(call),d_counted_of(call),default_capacity_of(call)),nonempty_of(call),
    Bias.kept_covered(~D,~dv,~Storage.Buffer<Tile.Tile8>,d_context,ShapeLaws.elements(d_cells_of(call)),layout_of(call),Kernel.bias(work_of(call),Move.held(~Tile.Tile8,dp),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call))),
      Buffer.pack(~Tile.Tile8,written_of(~D,~dv,d_context,call)),kept_of(~D,~dv,d_context,call,dp,df,dw,shaped)))

# ---- The cells a run gives ----

def values_ready(~D: Data,~dv: D -> U32 -> Tile.Tile8,+d_context: D,+call: BiasCall) ->
  Buffer.ContentReady<Tile.Tile8,Buffer.content(~Tile.Tile8,written_of(~D,~dv,d_context,call))>:
  Bias.bias_ready(~D,~dv,work_of(call),d_context,Tiled.zero_tile(),groups_of(call),groups_within(call))

def values_length(~D: Data,~dv: D -> U32 -> Tile.Tile8,+d_context: D,+call: BiasCall) ->
  {Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,written_of(~D,~dv,d_context,call))) == U32.to_nat(groups_of(call)) : Nat}:
  Bias.bias_length(~D,~dv,work_of(call),d_context,Tiled.zero_tile(),groups_of(call))

# A block of the cells a run gives holds the cell of the bias model.
def value_at(~D: Data,~dv: D -> U32 -> Tile.Tile8,+d_context: D,+call: BiasCall,+block: Nat,+inside: {Nat.is_lt(block,U32.to_nat(groups_of(call))) == True{} : Bool}) ->
  {Buffer.value(~Tile.Tile8,U32.from_nat(block),Buffer.content(~Tile.Tile8,written_of(~D,~dv,d_context,call))) ==
    Bias.cell_values(~D,~Bias.held_sample(~D,~dv),d_context,Kernel.Block{work_of(call),block,Bias.work_ranges(work_of(call))}) : Tile.Tile8}:
  Bias.bias_value(~D,~dv,work_of(call),d_context,Tiled.zero_tile(),groups_of(call),groups_within(call),groups_exact(call),block,inside)

# The cells of the result fit the capacity and are counted.
def result_room(+call: BiasCall) -> {Nat.is_le(ShapeLaws.elements([groups_of(call)]),U32.to_nat(Machine.capacity(limits_of(call)))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(U32.to_nat(groups_of(call)),1n),U32.to_nat(groups_of(call)),Order.times_one(U32.to_nat(groups_of(call)))) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits_of(call)))) == True{} : Bool}
  groups_room(call)

def result_counted(+call: BiasCall) -> Global.Counted<[groups_of(call)]>:
  StorageCount.counted_within(limits_of(call),[groups_of(call)],result_room(call),default_capacity_of(call))

# ---- The runtime of a call ----

# An accepted call finishes the bias kernel over its holding.
def selected_exact(+di: ParameterGradient.ImageGeometry,+g: U32,dr: ParameterGradient.Read,-d: Move.Holding<Tile.Tile8>,+workers: U32,+shape: List<&2,U32>) ->
  {ParameterGradient.bias_selected(True{},di,g,dr,d,workers,shape) ==
    ParameterGradient.bias_finished(dr,workers,Cells.image_channels(di),
      Kernel.bias(ParameterGradient.bias_work(di,g),d,Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(Cells.image_channels(di))))) :
    Tiled.Tiled & Tensors.Tensor<Tile.Tile8>}:
  ParameterGradient.ImageGeometry{+n,+o,+rows,+columns} = di
  {==}

# The bias gradient of d of an accepted call finishes the bias kernel over the
# pieces of d.
def run_exact(+call: BiasCall,-dp: Pieces.Pieces(~Tile.Tile8),+df: Tile.Tile8,+dw: U32) ->
  {ParameterGradient.bias(Tensors.Tensor{dp,d_cells_of(call),df,Tensors.Usable{},dw},d_lanes_of(call),group_of(call)) ==
    ParameterGradient.bias_finished(ParameterGradient.Read{d_cells_of(call),df,Tensors.Usable{},dw},dw,outputs_of(call),
      Kernel.bias(work_of(call),Move.held(~Tile.Tile8,dp),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),groups_of(call)))) : Tiled.Tiled & Tensors.Tensor<Tile.Tile8>}:
  BiasCall{+limits,+default_capacity,+d_cells,+d_lanes,+di,+d_named,+d_counted,+lanes_room,+g,+accepted} = call
  %Equal.sym(Maybe<&2,ParameterGradient.ImageGeometry>,ParameterGradient.cell_image(d_cells,d_lanes),Some{di},d_named) :
    {ParameterGradient.bias_shaped(_,True{},g,ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},Move.held(~Tile.Tile8,dp),dw,d_cells) ==
      ParameterGradient.bias_finished(ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},dw,Cells.image_channels(di),
        Kernel.bias(ParameterGradient.bias_work(di,g),Move.held(~Tile.Tile8,dp),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(Cells.image_channels(di))))) :
      Tiled.Tiled & Tensors.Tensor<Tile.Tile8>}
  %Equal.sym(Bool,ParameterGradient.positive(di) && U32.is_lt(0,g),True{},accepted) :
    {ParameterGradient.bias_selected(_,di,g,ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},Move.held(~Tile.Tile8,dp),dw,d_cells) ==
      ParameterGradient.bias_finished(ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},dw,Cells.image_channels(di),
        Kernel.bias(ParameterGradient.bias_work(di,g),Move.held(~Tile.Tile8,dp),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(Cells.image_channels(di))))) :
      Tiled.Tiled & Tensors.Tensor<Tile.Tile8>}
  selected_exact(di,g,ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},Move.held(~Tile.Tile8,dp),dw,d_cells)

# A finished run is the whole tensor of its cells and the tensor it gives back.
def finished_exact(dr: ParameterGradient.Read,+workers: U32,+outputs: U32,result: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) ->
  {ParameterGradient.bias_finished(dr,workers,outputs,result) ==
    (Tiled.Tiled{Tensors.whole(~Tile.Tile8,Pair.snd(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,result),[Tiled.groups(outputs)],Tiled.zero_tile(),Tensors.Usable{},workers),
      0n,outputs,Grid.Innermost{}},ParameterGradient.back(dr,Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,result))) :
    Tiled.Tiled & Tensors.Tensor<Tile.Tile8>}:
  (held,values) = result
  {==}
