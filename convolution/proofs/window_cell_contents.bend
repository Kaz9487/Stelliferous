# Every cell of a copied window comes from its image row: cell index of the
# window from row first is image cell first * width * groups + index.
import Base
import ../specification.bend as ConvSpec
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/tensor.bend as Tensors
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/contiguous_copy_readback.bend as CopyRead
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/traversal_indexed_values.bend as IndexedValues
import ../../lib/tensor_tiled_cells.bend as TiledCells
import ./window_contents.bend as WindowBounds
import ./window_positions.bend as Positions
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as Coordinates
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ./convolution_guard_facts.bend as Guards
import ./input_storage.bend as Inputs
import ./window_problem.bend as WindowProblem
import ./window_band_values.bend as BandValues
import ../../lib/proofs/word_multiplication.bend as WordMultiplication

# The cells of the channels of pixels pixels of an image, eight channels a
# cell.
def image_cells(+channels: U32,+pixels: U32,input: Tree.Tree<F32>) -> Tree.Tree<Tile.Tile8>:
  IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,TiledValues.tiled_blocks(pixels,TiledCells.Tiling{pixels,channels,Gemm.blocks(channels),0},input,
    Tree.fresh(Tile.Tile8,Tensors.depth_for((pixels * Gemm.blocks(channels) : U32)),Tile.filled(0.0))))

# The cells of rows left_rows from row left_first, copied from a tree of the
# cells from row first on.
def copied_tree(+groups: U32,+width: U32,+first: U32,+left_first: U32,+left_rows: U32,+tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Writes.pure(Tile.Tile8,Contiguous.writes(~Tile.Tile8,U32.to_nat((left_rows * width * groups : U32)),tree,0,((left_first - first) * width * groups : U32),0),
    Tree.fresh(Tile.Tile8,Tensors.depth_for((left_rows * width * groups : U32)),Tile.filled(0.0)))

# The groups of eight input channels.
def group_count(+problem: ConvSpec.Problem) -> Nat:
  TiledValues.groups_of(Guards.input_channel_count(problem))

# The cells of rows rows of the image.
def count(+problem: ConvSpec.Problem,rows: Nat) -> Nat:
  Nat.mul(Nat.mul(rows,Guards.width(problem)),group_count(problem))

def input_cells(+problem: ConvSpec.Problem,storage: Inputs.Contract<problem>) -> Tree.Tree<Tile.Tile8>:
  image_cells(U32.from_nat(Guards.input_channel_count(problem)),(U32.from_nat(Guards.height(problem)) * U32.from_nat(Guards.width(problem)) : U32),Inputs.input_tree(problem,storage))

def source(+problem: ConvSpec.Problem,+first: Nat,index: Nat) -> Nat:
  Nat.add(count(problem,first),index)

def At(+problem: ConvSpec.Problem,+root: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+base: Nat,+first: Nat,index: Nat) -> Data:
  {Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(base,index))) == Tree.read(Tile.Tile8,root,U32.from_nat(source(problem,first,index))) : Tile.Tile8}

def Contents(+problem: ConvSpec.Problem,+root: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+base: Nat,+first: Nat,+rows: Nat) -> Data:
  Members.All(Nat,{index => At(problem,root,tree,base,first,index) : Nat -> Data},Ranges.range(count(problem,rows),0n))

def read(+problem: ConvSpec.Problem,+root: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+base: Nat,+first: Nat,+rows: Nat,
  contents: Contents(problem,root,tree,base,first,rows),+index: Nat,+inside: {Nat.is_lt(index,count(problem,rows)) == True{} : Bool}) ->
  At(problem,root,tree,base,first,index):
  Members.all_member(Nat,{position => At(problem,root,tree,base,first,position) : Nat -> Data},index,Ranges.range(count(problem,rows),0n),contents,
    Members.range_member(count(problem,rows),0n,index,inside))

type Root is Data:
  Root{problem: ConvSpec.Problem,tree: Tree.Tree<Tile.Tile8>}

def root_count(context: Root) -> Nat:
  Root{+problem,tree} = context
  count(problem,Guards.height(problem))

def RootAt(+context: Root,+index: Nat) -> Data:
  Root{+problem,+tree} = context
  At(problem,tree,tree,0n,0n,index)

def root_read(+context: Root,+index: Nat,+inside: {Nat.is_lt(index,root_count(context)) == True{} : Bool}) -> RootAt(context,index):
  Root{+problem,+tree} = context
  {==}

# The image is the window of all its rows from row zero.
def root(+problem: ConvSpec.Problem,+tree: Tree.Tree<Tile.Tile8>) -> Contents(problem,tree,tree,0n,0n,Guards.height(problem)):
  Members.range_all(~Root,~RootAt,~root_count,~root_read,count(problem,Guards.height(problem)),0n,Root{problem,tree},Order.reflexive(count(problem,Guards.height(problem))))

# Cell index of the window from row first, read from a window from row
# parent_first, lies past the cells of the rows between them.
def source_composed(+problem: ConvSpec.Problem,+parent_first: Nat,+first: Nat,+index: Nat,+after: {Nat.is_le(parent_first,first) == True{} : Bool}) ->
  {source(problem,parent_first,Nat.add(count(problem,Nat.sub(first,parent_first)),index)) == source(problem,first,index) : Nat}:
  +width = Guards.width(problem)
  +groups = group_count(problem)
  +shift = Nat.sub(first,parent_first)
  +target = source(problem,first,index)
  %Addition.associative(Nat.mul(Nat.mul(parent_first,width),groups),Nat.mul(Nat.mul(shift,width),groups),index) : {_ == target : Nat}
  %RowMajor.distrib(Nat.mul(parent_first,width),Nat.mul(shift,width),groups) : {Nat.add(_,index) == target : Nat}
  %RowMajor.distrib(parent_first,shift,width) : {Nat.add(Nat.mul(_,groups),index) == target : Nat}
  %Equal.sym(Nat,Nat.add(parent_first,shift),first,Order.restore_shift(parent_first,first,after)) : {Nat.add(Nat.mul(Nat.mul(_,width),groups),index) == target : Nat}
  {==}

# Cell offset * groups + group of the window from row first is the cell of
# group at pixel first * width + offset of the image.
def source_pixel(+problem: ConvSpec.Problem,+first: Nat,+offset: Nat,+group: Nat) ->
  {source(problem,first,Nat.add(Nat.mul(offset,group_count(problem)),group)) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(first,Guards.width(problem)),offset),group_count(problem)),group) : Nat}:
  +width = Guards.width(problem)
  +groups = group_count(problem)
  +target = Nat.add(Nat.mul(Nat.add(Nat.mul(first,width),offset),groups),group)
  %Addition.associative(Nat.mul(Nat.mul(first,width),groups),Nat.mul(offset,groups),group) : {_ == target : Nat}
  %RowMajor.distrib(Nat.mul(first,width),offset,groups) : {Nat.add(_,group) == target : Nat}
  {==}

# The value of channel at column of the window from row first lies at pixel
# first * width + column of its image plane.
def value_source(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+channel: Nat,+column: Nat,
  +inside: {Nat.is_lt(column,Nat.mul(rows,Guards.width(problem))) == True{} : Bool}) ->
  {BandValues.source(WindowProblem.band(problem,first,rows),Nat.add(Nat.mul(channel,Nat.mul(rows,Guards.width(problem))),column)) ==
    Nat.add(Nat.mul(channel,Nat.mul(Guards.height(problem),Guards.width(problem))),Nat.add(Nat.mul(first,Guards.width(problem)),column)) : Nat}:
  ConvSpec.Problem{ConvSpec.Shape{ci,co,h,w,k,s,p},input,weights,bias} = problem
  +plane = Nat.mul(rows,w)
  +positive = Order.le_lt_trans(0n,column,plane,Order.zero_le(column),inside)
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(channel,plane),column),plane),channel,Radix.quotient(channel,column,plane,positive,inside)) :
    {Nat.add(Nat.mul(_,Nat.mul(h,w)),Nat.add(Nat.mul(first,w),Nat.mod(Nat.add(Nat.mul(channel,plane),column),plane))) ==
      Nat.add(Nat.mul(channel,Nat.mul(h,w)),Nat.add(Nat.mul(first,w),column)) : Nat}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(channel,plane),column),plane),column,Radix.remainder(channel,column,plane,positive,inside)) :
    {Nat.add(Nat.mul(channel,Nat.mul(h,w)),Nat.add(Nat.mul(first,w),_)) == Nat.add(Nat.mul(channel,Nat.mul(h,w)),Nat.add(Nat.mul(first,w),column)) : Nat}
  {==}

# The cells of rows rows fit the machine.
def cells_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+rows: Nat,
  +rows_le: {Nat.is_le(rows,Guards.height(problem)) == True{} : Bool}) ->
  {Nat.is_le(count(problem,rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +ci = Guards.input_channel_count(problem)
  +width = Guards.width(problem)
  +area = Nat.mul(Guards.height(problem),width)
  +groups = group_count(problem)
  +cap = U32.to_nat(Machine.capacity(limits))
  +scalar = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  Order.transitive(count(problem,rows),Nat.mul(area,groups),cap,
    RowMajor.mul_le(Nat.mul(rows,width),area,groups,RowMajor.mul_le(rows,Guards.height(problem),width,rows_le)),
    Order.transitive(Nat.mul(area,groups),Nat.mul(area,ci),cap,
      DivisionOrder.multiply_left_le(area,groups,ci,Ceiling.bounded(ci,8n,{==})),
      %RowMajor.multiply_commutative(ci,area) : {Nat.is_le(_,cap) == True{} : Bool}
      WindowBounds.input_room(limits,problem,scalar,valid)))

# The cells of rows rows as a word.
def cells_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+rows: U32,+width: U32,+groups: U32,
  +width_exact: {U32.to_nat(width) == Guards.width(problem) : Nat},+groups_exact: {U32.to_nat(groups) == group_count(problem) : Nat},
  +rows_le: {Nat.is_le(U32.to_nat(rows),Guards.height(problem)) == True{} : Bool}) ->
  {U32.to_nat((rows * width * groups : U32)) == count(problem,U32.to_nat(rows)) : Nat}:
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +fits = Limits.capacity_fits(limits)
  +row_count = U32.to_nat(rows)
  +image_width = Guards.width(problem)
  +cells = group_count(problem)
  +plane_small = Order.transitive(Nat.mul(row_count,image_width),Nat.mul(Guards.height(problem),image_width),cap,
    RowMajor.mul_le(row_count,Guards.height(problem),image_width,rows_le),WindowBounds.area_room(limits,problem,valid,supported))
  +width_back = Equal.sym(Nat,U32.to_nat(width),image_width,width_exact)
  +plane_exact = {%width_exact : {U32.to_nat((rows * width : U32)) == Nat.mul(row_count,_) : Nat}
    WordMultiplication.bounded_multiply(rows,width,capacity,fits,%width_back : {Nat.is_le(Nat.mul(row_count,_),cap) == True{} : Bool} plane_small) :
      {U32.to_nat((rows * width : U32)) == Nat.mul(row_count,image_width) : Nat}}
  %plane_exact : {U32.to_nat((rows * width * groups : U32)) == Nat.mul(_,cells) : Nat}
  %groups_exact : {U32.to_nat((rows * width * groups : U32)) == Nat.mul(U32.to_nat((rows * width : U32)),_) : Nat}
  WordMultiplication.bounded_multiply((rows * width : U32),groups,capacity,fits,
    %Equal.sym(Nat,U32.to_nat((rows * width : U32)),Nat.mul(row_count,image_width),plane_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(groups)),cap) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(groups),cells,groups_exact) : {Nat.is_le(Nat.mul(Nat.mul(row_count,image_width),_),cap) == True{} : Bool}
    cells_room(limits,problem,valid,supported,row_count,rows_le))

# A window of rows rows from row first copied from a window from row
# parent_first that holds them.
type Copy is Data:
  Copy{problem: ConvSpec.Problem,root: Tree.Tree<Tile.Tile8>,parent: Tree.Tree<Tile.Tile8>,parent_first: U32,parent_rows: U32,first: U32,rows: U32,groups: U32,width: U32,
    parent_contents: Contents(problem,root,parent,0n,U32.to_nat(parent_first),U32.to_nat(parent_rows)),
    after: {Nat.is_le(U32.to_nat(parent_first),U32.to_nat(first)) == True{} : Bool},
    rows_room: {Nat.is_le(Nat.add(Nat.sub(U32.to_nat(first),U32.to_nat(parent_first)),U32.to_nat(rows)),U32.to_nat(parent_rows)) == True{} : Bool},
    shift_exact: {U32.to_nat(((first - parent_first) * width * groups : U32)) == count(problem,Nat.sub(U32.to_nat(first),U32.to_nat(parent_first))) : Nat},
    total_exact: {U32.to_nat((rows * width * groups : U32)) == count(problem,U32.to_nat(rows)) : Nat},
    ready: Depth.Sized<(rows * width * groups : U32)>}

def copy_tree(context: Copy) -> Tree.Tree<Tile.Tile8>:
  Copy{problem,root,parent,parent_first,parent_rows,first,rows,groups,width,contents,after,room,shift,total,ready} = context
  copied_tree(groups,width,parent_first,first,rows,parent)

def copy_count(context: Copy) -> Nat:
  Copy{+problem,root,parent,parent_first,parent_rows,first,+rows,groups,width,contents,after,room,shift,total,ready} = context
  count(problem,U32.to_nat(rows))

def CopyAt(+context: Copy,+index: Nat) -> Data:
  Copy{+problem,+root,parent,parent_first,parent_rows,+first,rows,groups,width,contents,after,room,shift,total,ready} = context
  At(problem,root,copy_tree(context),0n,U32.to_nat(first),index)

def copy_read(+context: Copy,+index: Nat,+inside: {Nat.is_lt(index,copy_count(context)) == True{} : Bool}) -> CopyAt(context,index):
  Copy{+problem,+root,+parent,+parent_first,+parent_rows,+first,+rows,+groups,+width,+parent_contents,+after,+rows_room,+shift_exact,+total_exact,
    Depth.Sized{+depth_ok,+allocation_room}} = context
  +start = U32.to_nat(first)
  +parent_start = U32.to_nat(parent_first)
  +row_count = U32.to_nat(rows)
  +shift = Nat.sub(start,parent_start)
  +shift_cells = count(problem,shift)
  +image_width = Guards.width(problem)
  +cells = group_count(problem)
  +total = (rows * width * groups : U32)
  +source_word = ((first - parent_first) * width * groups : U32)
  +depth = Tensors.depth_for(total)
  +fresh = Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))
  +model = Buffer.fresh(~Tile.Tile8,depth,Tile.filled(0.0),U32.to_nat(total))
  +capacity = Tree.capacity(Tile.Tile8,fresh)
  +capacity_ok = Buffer.capacity_bound(~Tile.Tile8,model,depth_ok)
  +room = {%Equal.sym(U32,capacity,U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,fresh,Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)))) :
      {Nat.is_le(U32.to_nat(total),U32.to_nat(_)) == True{} : Bool}
    allocation_room : {Nat.is_le(U32.to_nat(total),U32.to_nat(capacity)) == True{} : Bool}}
  +index_inside = {%Equal.sym(Nat,U32.to_nat(total),count(problem,row_count),total_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    inside : {Nat.is_lt(index,U32.to_nat(total)) == True{} : Bool}}
  +copied = copied_tree(groups,width,parent_first,first,rows,parent)
  +copied_value = CopyRead.fresh_read(~Tile.Tile8,U32.to_nat(total),index,parent,source_word,depth,Tile.filled(0.0),depth_ok,capacity_ok,room,index_inside)
  +word_exact = Equal.trans(U32,(source_word + U32.from_nat(index) : U32),(U32.from_nat(shift_cells) + U32.from_nat(index) : U32),U32.from_nat(Nat.add(shift_cells,index)),
    Equal.cong(U32,U32,base => (base + U32.from_nat(index) : U32),source_word,U32.from_nat(shift_cells),
      Equal.sym(U32,U32.from_nat(shift_cells),source_word,Arithmetic.from_nat_exact(source_word,shift_cells,shift_exact))),
    WordArithmetic.native_add(shift_cells,index))
  +combined = Equal.trans(Nat,Nat.add(shift_cells,count(problem,row_count)),Nat.mul(Nat.add(Nat.mul(shift,image_width),Nat.mul(row_count,image_width)),cells),count(problem,Nat.add(shift,row_count)),
    Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(shift,image_width),Nat.mul(row_count,image_width)),cells),Nat.add(shift_cells,count(problem,row_count)),
      RowMajor.distrib(Nat.mul(shift,image_width),Nat.mul(row_count,image_width),cells)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,cells),Nat.add(Nat.mul(shift,image_width),Nat.mul(row_count,image_width)),Nat.mul(Nat.add(shift,row_count),image_width),
      Equal.sym(Nat,Nat.mul(Nat.add(shift,row_count),image_width),Nat.add(Nat.mul(shift,image_width),Nat.mul(row_count,image_width)),RowMajor.distrib(shift,row_count,image_width))))
  +parent_index = Nat.add(shift_cells,index)
  +parent_inside = Bounds.nat_trans(parent_index,Nat.add(shift_cells,count(problem,row_count)),count(problem,U32.to_nat(parent_rows)),
    Order.offset_lt(shift_cells,index,count(problem,row_count),inside),
    %Equal.sym(Nat,Nat.add(shift_cells,count(problem,row_count)),count(problem,Nat.add(shift,row_count)),combined) :
      {Nat.is_le(_,count(problem,U32.to_nat(parent_rows))) == True{} : Bool}
    RowMajor.mul_le(Nat.mul(Nat.add(shift,row_count),image_width),Nat.mul(U32.to_nat(parent_rows),image_width),cells,
      RowMajor.mul_le(Nat.add(shift,row_count),U32.to_nat(parent_rows),image_width,rows_room)))
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,copied,U32.from_nat(index)),Tree.read(Tile.Tile8,parent,(source_word + U32.from_nat(index) : U32)),
    Tree.read(Tile.Tile8,root,U32.from_nat(source(problem,start,index))),copied_value,
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,parent,(source_word + U32.from_nat(index) : U32)),Tree.read(Tile.Tile8,parent,U32.from_nat(parent_index)),
      Tree.read(Tile.Tile8,root,U32.from_nat(source(problem,start,index))),
      Equal.cong(U32,Tile.Tile8,position => Tree.read(Tile.Tile8,parent,position),(source_word + U32.from_nat(index) : U32),U32.from_nat(parent_index),word_exact),
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,parent,U32.from_nat(parent_index)),Tree.read(Tile.Tile8,root,U32.from_nat(source(problem,parent_start,parent_index))),
        Tree.read(Tile.Tile8,root,U32.from_nat(source(problem,start,index))),
        read(problem,root,parent,0n,parent_start,U32.to_nat(parent_rows),parent_contents,parent_index,parent_inside),
        Equal.cong(Nat,Tile.Tile8,position => Tree.read(Tile.Tile8,root,U32.from_nat(position)),source(problem,parent_start,parent_index),source(problem,start,index),
          source_composed(problem,parent_start,start,index,after)))))

def copied(+context: Copy) -> Members.All(Nat,{index => CopyAt(context,index) : Nat -> Data},Ranges.range(copy_count(context),0n)):
  Members.range_all(~Copy,~CopyAt,~copy_count,~copy_read,copy_count(context),0n,context,Order.reflexive(copy_count(context)))

# The window a split copies holds the cells of its rows.
def copied_contents(+limits: Machine.Limits,+problem: ConvSpec.Problem,+root: Tree.Tree<Tile.Tile8>,+parent: Tree.Tree<Tile.Tile8>,
  +parent_first: U32,+parent_rows: U32,+first: U32,+rows: U32,+groups: U32,+width: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +groups_exact: {U32.to_nat(groups) == group_count(problem) : Nat},+width_exact: {U32.to_nat(width) == Guards.width(problem) : Nat},
  +parent_contents: Contents(problem,root,parent,0n,U32.to_nat(parent_first),U32.to_nat(parent_rows)),
  +after: {Nat.is_le(U32.to_nat(parent_first),U32.to_nat(first)) == True{} : Bool},
  +within: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows))) == True{} : Bool},
  +parent_fits: {Nat.is_le(Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows)),Guards.height(problem)) == True{} : Bool}) ->
  Contents(problem,root,copied_tree(groups,width,parent_first,first,rows,parent),0n,U32.to_nat(first),U32.to_nat(rows)):
  +height = Guards.height(problem)
  +parent_start = U32.to_nat(parent_first)
  +parent_count = U32.to_nat(parent_rows)
  +start = U32.to_nat(first)
  +row_count = U32.to_nat(rows)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +fits = Limits.capacity_fits(limits)
  +width_positive = WindowBounds.valid_condition(problem,valid,7n)
  +rows_le = WindowBounds.count_le(problem,parent_first,parent_rows,first,rows,within,parent_fits)
  +shift_le = Order.transitive(Nat.sub(start,parent_start),start,height,Order.sub_le(start,parent_start),
    Order.transitive(start,Nat.add(start,row_count),height,Order.le_add(start,row_count),Order.transitive(Nat.add(start,row_count),Nat.add(parent_start,parent_count),height,within,parent_fits)))
  +height_small = Order.transitive(height,Nat.mul(height,Guards.width(problem)),cap,RowMajor.product_covers(height,Guards.width(problem),width_positive),
    WindowBounds.area_room(limits,problem,valid,supported))
  +difference_exact = Arithmetic.subtract(first,parent_first,capacity,fits,after,Order.transitive(Nat.sub(start,parent_start),height,cap,shift_le,height_small))
  +shift_exact = {%difference_exact : {U32.to_nat(((first - parent_first) * width * groups : U32)) == count(problem,_) : Nat}
    cells_exact(limits,problem,valid,supported,(first - parent_first : U32),width,groups,width_exact,groups_exact,
      %Equal.sym(Nat,U32.to_nat((first - parent_first : U32)),Nat.sub(start,parent_start),difference_exact) : {Nat.is_le(_,height) == True{} : Bool} shift_le) :
      {U32.to_nat(((first - parent_first) * width * groups : U32)) == count(problem,Nat.sub(start,parent_start)) : Nat}}
  +total = (rows * width * groups : U32)
  +total_exact = cells_exact(limits,problem,valid,supported,rows,width,groups,width_exact,groups_exact,rows_le)
  +ready = Depth.bounded_with(total,capacity,fits,Limits.capacity_allocatable(limits),
    %Equal.sym(Bool,U32.is_le(1,capacity),Nat.is_le(1n,cap),Comparison.le(1,capacity)) : {_ == True{} : Bool}
      Order.transitive(1n,7n,cap,{==},Bounds.as_nat_le(7,capacity,Limits.seven_within_capacity(limits))),
    %Equal.sym(Bool,U32.is_le(total,capacity),Nat.is_le(U32.to_nat(total),cap),Comparison.le(total,capacity)) : {_ == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(total),count(problem,row_count),total_exact) : {Nat.is_le(_,cap) == True{} : Bool}
      cells_room(limits,problem,valid,supported,row_count,rows_le))
  +rows_room = {%Positions.sub_after(start,parent_start,row_count,after) : {Nat.is_le(_,parent_count) == True{} : Bool}
    %Order.subtract_shift(parent_start,parent_count) : {Nat.is_le(Nat.sub(Nat.add(start,row_count),parent_start),_) == True{} : Bool}
    DivisionOrder.subtract_le(Nat.add(start,row_count),Nat.add(parent_start,parent_count),parent_start,within) :
      {Nat.is_le(Nat.add(Nat.sub(start,parent_start),row_count),parent_count) == True{} : Bool}}
  copied(Copy{problem,root,parent,parent_first,parent_rows,first,rows,groups,width,parent_contents,after,rows_room,shift_exact,total_exact,ready})

# Lane channel % 8 of the input cell of pixel and group channel / 8 is the
# input value of channel at pixel.
def input_lane(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+channel: Nat,+pixel: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.input_channel_count(problem)) == True{} : Bool},
  +pixel_inside: {Nat.is_lt(pixel,Nat.mul(Guards.height(problem),Guards.width(problem))) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,input_cells(problem,storage),U32.from_nat(Nat.add(Nat.mul(pixel,group_count(problem)),Nat.div(channel,8n))))) ==
    ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(Guards.height(problem),Guards.width(problem))),pixel),Guards.input(problem)) : F32}:
  +ci = Guards.input_channel_count(problem)
  +height = Guards.height(problem)
  +width = Guards.width(problem)
  +area = Nat.mul(height,width)
  +native_channels = U32.from_nat(ci)
  +native_height = U32.from_nat(height)
  +native_width = U32.from_nat(width)
  +plane = (native_height * native_width : U32)
  +native_groups = Gemm.blocks(native_channels)
  +cells = (plane * native_groups : U32)
  +groups = group_count(problem)
  +scalar = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +ci_exact = Coordinates.ci_exact(limits,problem,scalar)
  +height_exact = Coordinates.height_exact(limits,problem,scalar)
  +width_exact = Coordinates.width_exact(limits,problem,scalar)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +fits = Limits.capacity_fits(limits)
  +ci_bound = Limits.dimension_capacity(limits,ci,Coordinates.ci_bound(limits,problem,scalar))
  +area_bound = WindowBounds.area_room(limits,problem,valid,supported)
  +plane_exact = {%height_exact : {U32.to_nat(plane) == Nat.mul(_,width) : Nat}
    %width_exact : {U32.to_nat(plane) == Nat.mul(U32.to_nat(native_height),_) : Nat}
    WordMultiplication.bounded_multiply(native_height,native_width,capacity,fits,
      %Equal.sym(Nat,U32.to_nat(native_height),height,height_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(native_width)),cap) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(native_width),width,width_exact) : {Nat.is_le(Nat.mul(height,_),cap) == True{} : Bool} area_bound) :
      {U32.to_nat(plane) == area : Nat}}
  +groups_exact = {%ci_exact : {U32.to_nat(native_groups) == TiledValues.groups_of(_) : Nat}
    Encoded.eight_blocks_exact(limits,native_channels,%Equal.sym(Nat,U32.to_nat(native_channels),ci,ci_exact) : {Nat.is_le(_,cap) == True{} : Bool} ci_bound) :
      {U32.to_nat(native_groups) == groups : Nat}}
  +cells_bound = {%Equal.sym(Nat,U32.to_nat(plane),area,plane_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(native_groups)),cap) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(native_groups),groups,groups_exact) : {Nat.is_le(Nat.mul(area,_),cap) == True{} : Bool}
    cells_room(limits,problem,valid,supported,height,Order.reflexive(height)) :
      {Nat.is_le(Nat.mul(U32.to_nat(plane),U32.to_nat(native_groups)),cap) == True{} : Bool}}
  +cells_word = WordMultiplication.bounded_multiply(plane,native_groups,capacity,fits,cells_bound)
  +cells_bounded = {%Equal.sym(Bool,U32.is_le(cells,capacity),Nat.is_le(U32.to_nat(cells),cap),Comparison.le(cells,capacity)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(cells),Nat.mul(U32.to_nat(plane),U32.to_nat(native_groups)),cells_word) : {Nat.is_le(_,cap) == True{} : Bool} cells_bound : {U32.is_le(cells,capacity) == True{} : Bool}}
  +lanes_bounded = {%Equal.sym(Nat,U32.to_nat(native_channels),ci,ci_exact) : {Nat.is_le(_,cap) == True{} : Bool} ci_bound : {Nat.is_le(U32.to_nat(native_channels),cap) == True{} : Bool}}
  +index = Nat.add(Nat.mul(channel,area),pixel)
  +length = Order.equal_of_is_eq(ConvSpec.length(Guards.input(problem)),Nat.mul(ci,area),WindowBounds.valid_condition(problem,valid,2n))
  Equal.trans(F32,TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,input_cells(problem,storage),U32.from_nat(Nat.add(Nat.mul(pixel,groups),Nat.div(channel,8n))))),
    Tree.read(F32,Inputs.input_tree(problem,storage),U32.from_nat(index)),ConvSpec.at(index,Guards.input(problem)),
    %plane_exact :
      {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,input_cells(problem,storage),U32.from_nat(Nat.add(Nat.mul(pixel,groups),Nat.div(channel,8n))))) ==
        Tree.read(F32,Inputs.input_tree(problem,storage),U32.from_nat(Nat.add(Nat.mul(channel,_),pixel))) : F32}
    %groups_exact :
      {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,input_cells(problem,storage),U32.from_nat(Nat.add(Nat.mul(pixel,_),Nat.div(channel,8n))))) ==
        Tree.read(F32,Inputs.input_tree(problem,storage),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(plane)),pixel))) : F32}
    TiledValues.tiled_lane(limits,cells,plane,native_channels,Tile.filled(0.0),Inputs.input_tree(problem,storage),channel,pixel,
      Equal.sym(Nat,U32.to_nat(cells),Nat.mul(U32.to_nat(plane),U32.to_nat(native_groups)),cells_word),cells_bounded,lanes_bounded,
      %Equal.sym(Nat,U32.to_nat(native_channels),ci,ci_exact) : {Nat.is_lt(channel,_) == True{} : Bool} channel_inside,
      %Equal.sym(Nat,U32.to_nat(plane),area,plane_exact) : {Nat.is_lt(pixel,_) == True{} : Bool} pixel_inside),
    Inputs.stored_value(Guards.input(problem),Inputs.input_sequence(problem,storage),index,
      %Equal.sym(Nat,ConvSpec.length(Guards.input(problem)),Nat.mul(ci,area),length) : {Nat.is_lt(index,_) == True{} : Bool}
      RowMajor.radix(channel,ci,pixel,area,channel_inside,pixel_inside)))

# Lane channel % 8 of the window cell of offset and group channel / 8 is the
# value of channel at offset of the window.
def window_lane(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+tree: Tree.Tree<Tile.Tile8>,+base: Nat,+first: Nat,+rows: Nat,+channel: Nat,+offset: Nat,
  +contents: Contents(problem,input_cells(problem,storage),tree,base,first,rows),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +rows_fit: {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.input_channel_count(problem)) == True{} : Bool},+offset_inside: {Nat.is_lt(offset,Nat.mul(rows,Guards.width(problem))) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(base,Nat.add(Nat.mul(offset,group_count(problem)),Nat.div(channel,8n)))))) ==
    BandValues.value(WindowProblem.band(problem,first,rows),Nat.add(Nat.mul(channel,Nat.mul(rows,Guards.width(problem))),offset)) : F32}:
  +ci = Guards.input_channel_count(problem)
  +width = Guards.width(problem)
  +area = Nat.mul(Guards.height(problem),width)
  +plane = Nat.mul(rows,width)
  +groups = group_count(problem)
  +group = Nat.div(channel,8n)
  +index = Nat.add(Nat.mul(offset,groups),group)
  +pixel = Nat.add(Nat.mul(first,width),offset)
  +group_inside = {%Equal.sym(Bool,Nat.is_lt(group,groups),Nat.is_lt(Nat.mul(group,8n),ci),Ceiling.upper_comparison(ci,8n,group,{==})) : {_ == True{} : Bool}
    Order.le_lt_trans(Nat.mul(group,8n),channel,ci,DivisionOrder.product_below(channel,8n,{==}),channel_inside) : {Nat.is_lt(group,groups) == True{} : Bool}}
  +pixel_inside = Bounds.nat_trans(pixel,Nat.add(Nat.mul(first,width),plane),area,Order.offset_lt(Nat.mul(first,width),offset,plane,offset_inside),
    %RowMajor.distrib(first,rows,width) : {Nat.is_le(_,area) == True{} : Bool} RowMajor.mul_le(Nat.add(first,rows),Guards.height(problem),width,rows_fit))
  +cell = Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(base,index)))
  +root_cell = Tree.read(Tile.Tile8,input_cells(problem,storage),U32.from_nat(Nat.add(Nat.mul(pixel,groups),group)))
  +cell_equal = Equal.trans(Tile.Tile8,cell,Tree.read(Tile.Tile8,input_cells(problem,storage),U32.from_nat(source(problem,first,index))),root_cell,
    read(problem,input_cells(problem,storage),tree,base,first,rows,contents,index,RowMajor.radix(offset,plane,group,groups,offset_inside,group_inside)),
    Equal.cong(Nat,Tile.Tile8,position => Tree.read(Tile.Tile8,input_cells(problem,storage),U32.from_nat(position)),source(problem,first,index),Nat.add(Nat.mul(pixel,groups),group),
      source_pixel(problem,first,offset,group)))
  %Equal.sym(Nat,BandValues.source(WindowProblem.band(problem,first,rows),Nat.add(Nat.mul(channel,plane),offset)),Nat.add(Nat.mul(channel,area),pixel),
    value_source(problem,first,rows,channel,offset,offset_inside)) :
    {TileLaws.select_lane(Nat.mod(channel,8n),cell) == ConvSpec.at(_,Guards.input(problem)) : F32}
  Equal.trans(F32,TileLaws.select_lane(Nat.mod(channel,8n),cell),TileLaws.select_lane(Nat.mod(channel,8n),root_cell),ConvSpec.at(Nat.add(Nat.mul(channel,area),pixel),Guards.input(problem)),
    Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(Nat.mod(channel,8n),tile),cell,root_cell,cell_equal),
    input_lane(limits,problem,storage,channel,pixel,valid,supported,channel_inside,pixel_inside))
