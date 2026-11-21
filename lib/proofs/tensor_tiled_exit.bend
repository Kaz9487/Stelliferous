# The tiled form read back as row-major values: every piece writes the
# valid lanes of its cells at the indices of their values.
import Base
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_refinement.bend as StorageLaws
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_each.bend as Each
import ./natural_addition.bend as Addition
import ./nat_division.bend as Division
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_cell_room.bend as Rooms
import ./tensor_global_shape_count.bend as Global
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_count_refinement.bend as Count
import ../default_machine_limits.bend as Default
import ./u32_shift.bend as Shift
import ./tensor_tiled_values.bend as TiledValues
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_tiled_entry.bend as Entry
import ./word_multiplication.bend as WordMultiplication
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_public.bend as Public
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./tensor_contract.bend as Contract
import ./tensor_tiled_indexed_values.bend as TiledIndexedValues
import ../tensor_tiled_indexed.bend as TiledIndexed
import ./tensor_tiled_representation.bend as Representation
import ./tensor_allocation_depth.bend as Depth
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./machine_limit_bounds.bend as Limits
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./nat_ceiling.bend as Ceiling
import ./encoded_coordinates.bend as Encoded
import ../machine_limits.bend as Machine
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding
import ../kernel_gemm.bend as KernelGemm
import ../traversal_loop.bend as Loop

# ---- What a piece writes ----

# A value written into contents.
def write_model(+index: U32,+value: F32,content: Buffer.Content<F32>) -> Buffer.Content<F32>:
  Buffer.Content{tree,length} = content
  Buffer.Content{Tree.write(F32,tree,index,value),length}

type WriteAt is Data:
  WriteAt{index: U32,value: F32}

def Written(at: WriteAt,-buffer: Storage.Buffer<F32>) -> Type:
  WriteAt{+index,+value} = at
  {Buffer.contents(~F32,Storage.write(~F32,index,value,buffer)) == write_model(index,value,Buffer.contents(~F32,buffer)) : Buffer.Content<F32>}

def written_packed(+index: U32,+value: F32,+model: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Storage.write(~F32,index,value,Buffer.pack(~F32,model))) == write_model(index,value,Buffer.contents(~F32,Buffer.pack(~F32,model))) : Buffer.Content<F32>}:
  Buffer.Model{+tree,+length,+certificate} = model
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,tree)),tree,StorageLaws.reflect_packed(F32,tree)) :
    {Buffer.contents(~F32,Storage.write(~F32,index,value,Buffer.pack(~F32,Buffer.Model{tree,length,certificate}))) == write_model(index,value,Buffer.Content{_,length}) :
      Buffer.Content<F32>}
  Buffer.write_value_packed(~F32,index,value,Buffer.Model{tree,length,certificate})

# A write into a buffer writes into its contents.
def write_contents(+index: U32,+value: F32,buffer: Storage.Buffer<F32>) ->
  {Buffer.contents(~F32,Storage.write(~F32,index,value,buffer)) == write_model(index,value,Buffer.contents(~F32,buffer)) : Buffer.Content<F32>}:
  Buffer.lift(~F32,~WriteAt,~Written,WriteAt{index,value},buffer,Buffer.observe(~F32,buffer),model => written_packed(index,value,model))

# A lane written when it holds a value.
def lane_model(wanted: Bool,+index: U32,+value: F32,content: Buffer.Content<F32>) -> Buffer.Content<F32>:
  match wanted:
    case True{}: write_model(index,value,content)
    case False{}: content

def lane_contents(+wanted: Bool,+index: U32,+value: F32,output: Storage.Buffer<F32>) ->
  {Buffer.contents(~F32,Tiled.lane_written(wanted,index,value,output)) == lane_model(wanted,index,value,Buffer.contents(~F32,output)) : Buffer.Content<F32>}:
  match wanted:
    case True{}: write_contents(index,value,output)
    case False{}: {==}

def lane_step(+wanted: Bool,+index: U32,+value: F32,output: Storage.Buffer<F32>,-before: Buffer.Content<F32>,+held: {Buffer.contents(~F32,output) == before : Buffer.Content<F32>}) ->
  {Buffer.contents(~F32,Tiled.lane_written(wanted,index,value,output)) == lane_model(wanted,index,value,before) : Buffer.Content<F32>}:
  Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,Tiled.lane_written(wanted,index,value,output)),lane_model(wanted,index,value,Buffer.contents(~F32,output)),
    lane_model(wanted,index,value,before),lane_contents(wanted,index,value,output),
    Equal.cong(Buffer.Content<F32>,Buffer.Content<F32>,content => lane_model(wanted,index,value,content),Buffer.contents(~F32,output),before,held))

# The valid lanes of a cell written from base on, step apart.
def cell_model(+base: U32,+step: U32,+valid: U32,tile: Tile.Tile8,content: Buffer.Content<F32>) -> Buffer.Content<F32>:
  Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h} = tile
  lane_model(U32.is_lt(7,valid),(base + 7 * step : U32),h,lane_model(U32.is_lt(6,valid),(base + 6 * step : U32),g,
    lane_model(U32.is_lt(5,valid),(base + 5 * step : U32),f,lane_model(U32.is_lt(4,valid),(base + 4 * step : U32),e,
      lane_model(U32.is_lt(3,valid),(base + 3 * step : U32),d,lane_model(U32.is_lt(2,valid),(base + 2 * step : U32),c,
        lane_model(U32.is_lt(1,valid),(base + step : U32),b,lane_model(U32.is_lt(0,valid),base,a,content))))))))

def lane0(+base: U32,+step: U32,+valid: U32,+value: F32,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Tiled.lane_written(U32.is_lt(0,valid),base,value,output)

def lane1(+base: U32,+step: U32,+valid: U32,+value: F32,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Tiled.lane_written(U32.is_lt(1,valid),(base + step : U32),value,output)

def lane2(+base: U32,+step: U32,+valid: U32,+value: F32,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Tiled.lane_written(U32.is_lt(2,valid),(base + 2 * step : U32),value,output)

def lane3(+base: U32,+step: U32,+valid: U32,+value: F32,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Tiled.lane_written(U32.is_lt(3,valid),(base + 3 * step : U32),value,output)

def lane4(+base: U32,+step: U32,+valid: U32,+value: F32,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Tiled.lane_written(U32.is_lt(4,valid),(base + 4 * step : U32),value,output)

def lane5(+base: U32,+step: U32,+valid: U32,+value: F32,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Tiled.lane_written(U32.is_lt(5,valid),(base + 5 * step : U32),value,output)

def lane6(+base: U32,+step: U32,+valid: U32,+value: F32,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Tiled.lane_written(U32.is_lt(6,valid),(base + 6 * step : U32),value,output)

# A cell written into packed contents writes its valid lanes into them.
def cell_contents(+base: U32,+step: U32,+valid: U32,+tile: Tile.Tile8,+model: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Tiled.cell_untiled(base,step,valid,tile,Buffer.pack(~F32,model))) == cell_model(base,step,valid,tile,Buffer.contents(~F32,Buffer.pack(~F32,model))) :
    Buffer.Content<F32>}:
  Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h} = tile
  +start = Buffer.contents(~F32,Buffer.pack(~F32,model))
  +m0 = lane_model(U32.is_lt(0,valid),base,a,start)
  +m1 = lane_model(U32.is_lt(1,valid),(base + step : U32),b,m0)
  +m2 = lane_model(U32.is_lt(2,valid),(base + 2 * step : U32),c,m1)
  +m3 = lane_model(U32.is_lt(3,valid),(base + 3 * step : U32),d,m2)
  +m4 = lane_model(U32.is_lt(4,valid),(base + 4 * step : U32),e,m3)
  +m5 = lane_model(U32.is_lt(5,valid),(base + 5 * step : U32),f,m4)
  +m6 = lane_model(U32.is_lt(6,valid),(base + 6 * step : U32),g,m5)
  lane_step(U32.is_lt(7,valid),(base + 7 * step : U32),h,
    lane6(base,step,valid,g,lane5(base,step,valid,f,lane4(base,step,valid,e,lane3(base,step,valid,d,lane2(base,step,valid,c,lane1(base,step,valid,b,
      lane0(base,step,valid,a,Buffer.pack(~F32,model)))))))),m6,
  lane_step(U32.is_lt(6,valid),(base + 6 * step : U32),g,
    lane5(base,step,valid,f,lane4(base,step,valid,e,lane3(base,step,valid,d,lane2(base,step,valid,c,lane1(base,step,valid,b,lane0(base,step,valid,a,Buffer.pack(~F32,model))))))),m5,
  lane_step(U32.is_lt(5,valid),(base + 5 * step : U32),f,
    lane4(base,step,valid,e,lane3(base,step,valid,d,lane2(base,step,valid,c,lane1(base,step,valid,b,lane0(base,step,valid,a,Buffer.pack(~F32,model)))))),m4,
  lane_step(U32.is_lt(4,valid),(base + 4 * step : U32),e,
    lane3(base,step,valid,d,lane2(base,step,valid,c,lane1(base,step,valid,b,lane0(base,step,valid,a,Buffer.pack(~F32,model))))),m3,
  lane_step(U32.is_lt(3,valid),(base + 3 * step : U32),d,lane2(base,step,valid,c,lane1(base,step,valid,b,lane0(base,step,valid,a,Buffer.pack(~F32,model)))),m2,
  lane_step(U32.is_lt(2,valid),(base + 2 * step : U32),c,lane1(base,step,valid,b,lane0(base,step,valid,a,Buffer.pack(~F32,model))),m1,
  lane_step(U32.is_lt(1,valid),(base + step : U32),b,lane0(base,step,valid,a,Buffer.pack(~F32,model)),m0,
  lane_step(U32.is_lt(0,valid),base,a,Buffer.pack(~F32,model),start,{==}))))))))

# The cell at a cell index written into contents, as the runtime places it.
def written_model(+context: Cells.Cells,+cell: U32,tile: Tile.Tile8,content: Buffer.Content<F32>) -> Buffer.Content<F32>:
  Cells.Cells{+outer,+after,+lanes,+groups,+position} = context
  +group = Bool.pick(U32,Grid.is_outermost(position),(cell / U32.max(outer,1) : U32),(cell % U32.max(groups,1) : U32))
  +index = Bool.pick(U32,Grid.is_outermost(position),(cell % U32.max(outer,1) : U32),(cell / U32.max(groups,1) : U32))
  +inner = U32.max(after,1)
  cell_model(((index / inner * lanes + group * 8) * inner + index % inner : U32),inner,(lanes - group * 8 : U32),tile,content)

def written_contents(+context: Cells.Cells,+cell: U32,+tile: Tile.Tile8,+model: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Tiled.written_cell(context,cell,tile,Buffer.pack(~F32,model))) == written_model(context,cell,tile,Buffer.contents(~F32,Buffer.pack(~F32,model))) :
    Buffer.Content<F32>}:
  Cells.Cells{+outer,+after,+lanes,+groups,+position} = context
  +group = Bool.pick(U32,Grid.is_outermost(position),(cell / U32.max(outer,1) : U32),(cell % U32.max(groups,1) : U32))
  +index = Bool.pick(U32,Grid.is_outermost(position),(cell % U32.max(outer,1) : U32),(cell / U32.max(groups,1) : U32))
  +inner = U32.max(after,1)
  cell_contents(((index / inner * lanes + group * 8) * inner + index % inner : U32),inner,(lanes - group * 8 : U32),tile,model)

# The cells of a piece from index on, written into contents.
def piece_model(fuel: Nat,+index: U32,+first: U32,+context: Cells.Cells,+cells: Tree.Tree<Tile.Tile8>,content: Buffer.Content<F32>) -> Buffer.Content<F32>:
  match fuel:
    case 0n: content
    case 1n+rest: piece_model(rest,(index + 1 : U32),first,context,cells,written_model(context,(first + index : U32),Tree.read(Tile.Tile8,cells,index),content))

def piece_contents(fuel: Nat,+index: U32,+first: U32,+context: Cells.Cells,+part: Buffer.Model<Tile.Tile8>,+model: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Tiled.untiled_output(Loop.run(~Tiled.Untiling,~Tiled.untiling_step,fuel,Tiled.Untiling{Buffer.pack(~Tile.Tile8,part),Buffer.pack(~F32,model),index,first,context}))) ==
    piece_model(fuel,index,first,context,Buffer.tree(~Tile.Tile8,part),Buffer.contents(~F32,Buffer.pack(~F32,model))) : Buffer.Content<F32>}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +cells = Buffer.tree(~Tile.Tile8,part)
      +tile = Tree.read(Tile.Tile8,cells,index)
      +cell = (first + index : U32)
      +next = (index + 1 : U32)
      +observation = Buffer.observe(~F32,Tiled.written_cell(context,cell,tile,Buffer.pack(~F32,model)))
      +written = Buffer.observed_model(~F32,Tiled.written_cell(context,cell,tile,Buffer.pack(~F32,model)),observation)
      +target = piece_model(rest,next,first,context,cells,written_model(context,cell,tile,Buffer.contents(~F32,Buffer.pack(~F32,model))))
      %Equal.sym(Storage.Buffer<Tile.Tile8> & Tile.Tile8,Storage.read(~Tile.Tile8,Buffer.pack(~Tile.Tile8,part),index),(Buffer.pack(~Tile.Tile8,part),tile),
          Buffer.read(~Tile.Tile8,part,index)) :
        {Buffer.contents(~F32,Tiled.untiled_output(Loop.run(~Tiled.Untiling,~Tiled.untiling_step,rest,Tiled.untiling_read(index,first,context,_,Buffer.pack(~F32,model))))) ==
          target : Buffer.Content<F32>}
      %Buffer.observed_equation(~F32,Tiled.written_cell(context,cell,tile,Buffer.pack(~F32,model)),observation) :
        {Buffer.contents(~F32,Tiled.untiled_output(Loop.run(~Tiled.Untiling,~Tiled.untiling_step,rest,Tiled.Untiling{Buffer.pack(~Tile.Tile8,part),_,next,first,context}))) == target : Buffer.Content<F32>}
      Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,Tiled.untiled_output(Loop.run(~Tiled.Untiling,~Tiled.untiling_step,rest,Tiled.Untiling{Buffer.pack(~Tile.Tile8,part),Buffer.pack(~F32,written),next,first,context}))),
        piece_model(rest,next,first,context,cells,Buffer.contents(~F32,Buffer.pack(~F32,written))),target,piece_contents(rest,next,first,context,part,written),
        Equal.cong(Buffer.Content<F32>,Buffer.Content<F32>,content => piece_model(rest,next,first,context,cells,content),Buffer.contents(~F32,Buffer.pack(~F32,written)),
          written_model(context,cell,tile,Buffer.contents(~F32,Buffer.pack(~F32,model))),
          Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,written)),Buffer.contents(~F32,Tiled.written_cell(context,cell,tile,Buffer.pack(~F32,model))),
            written_model(context,cell,tile,Buffer.contents(~F32,Buffer.pack(~F32,model))),
            Equal.cong(Storage.Buffer<F32>,Buffer.Content<F32>,buffer => Buffer.contents(~F32,buffer),Buffer.pack(~F32,written),
              Tiled.written_cell(context,cell,tile,Buffer.pack(~F32,model)),Buffer.observed_equation(~F32,Tiled.written_cell(context,cell,tile,Buffer.pack(~F32,model)),observation)),
            written_contents(context,cell,tile,model))))

# What a piece writes into the values.
def untile_step(context: Cells.Cells,+first: U32,+rows: U32,part: Buffer.Model<Tile.Tile8>,content: Buffer.Content<F32>) -> Buffer.Content<F32>:
  piece_model(U32.to_nat(rows),0,first,context,Buffer.tree(~Tile.Tile8,part),content)

def untile_rule(+context: Cells.Cells,+first: U32,+rows: U32,+part: Buffer.Model<Tile.Tile8>,+kept: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Tiled.untiled_piece(context,first,rows,Buffer.pack(~Tile.Tile8,part),Buffer.pack(~F32,kept))) ==
    untile_step(context,first,rows,part,Buffer.contents(~F32,Buffer.pack(~F32,kept))) : Buffer.Content<F32>}:
  piece_contents(U32.to_nat(rows),0,first,context,part,kept)

# The pieces of a model folded into values thread what every piece writes.
def untiled_fold(+context: Cells.Cells,+model: PiecesModel.Model<Tile.Tile8>,+kept: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pieces.fold(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Cells.Cells,~Tiled.untiled_piece,context,PiecesModel.pack(~Tile.Tile8,model),
      Buffer.pack(~F32,kept))) ==
    PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,context,PiecesModel.buffer_parts(~Tile.Tile8,model),
      Buffer.contents(~F32,Buffer.pack(~F32,kept))) : Buffer.Content<F32>}:
  %PiecesModel.buffer_parts_packed(~Tile.Tile8,model) :
    {Buffer.contents(~F32,Pieces.fold(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Cells.Cells,~Tiled.untiled_piece,context,_,Buffer.pack(~F32,kept))) ==
      PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,context,PiecesModel.buffer_parts(~Tile.Tile8,model),
        Buffer.contents(~F32,Buffer.pack(~F32,kept))) : Buffer.Content<F32>}
  Each.shaped_folded(~Buffer.Model<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~Buffer.pack(~Tile.Tile8),~Storage.Buffer<F32>,~Cells.Cells,~Buffer.Model<F32>,~Buffer.pack(~F32),
    ~Each.buffer_held(~F32),~Buffer.Content<F32>,~Buffer.contents(~F32),~Tiled.untiled_piece,~untile_step,~untile_rule,PiecesModel.buffer_parts(~Tile.Tile8,model),context,kept)

# ---- Writes of the values at their indices ----

# The value at an index of contents is the original there.
def Good(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+content: Buffer.Content<F32>,+query: Nat) -> Data:
  {Tree.read(F32,Buffer.content_tree(~F32,content),U32.from_nat(query)) == original(logical,U32.from_nat(query)) : F32}

# The shape written contents keep.
type Kept<-depth: Nat,-capacity: U32,-size: Nat,-content: Buffer.Content<F32>> is Data:
  Kept{balanced: Shape.Balanced(F32,depth,Buffer.content_tree(~F32,content)),sized: {Tree.capacity(F32,Buffer.content_tree(~F32,content)) == capacity : U32},
    long: {Buffer.content_length(~F32,content) == size : Nat}}

# A lane of a cell: whether it holds a value, its index and its value.
type Lane is Data:
  Lane{wanted: Bool,index: U32,value: F32}

def lanes_model(lanes: List<&2,Lane>,content: Buffer.Content<F32>) -> Buffer.Content<F32>:
  match lanes:
    case Nil{}: content
    case Con{Lane{+wanted,+index,+value},rest}: lanes_model(rest,lane_model(wanted,index,value,content))

# A write of the original at an index within the capacity.
def Right(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+capacity: U32,wanted: Bool,+index: U32,+value: F32) -> Data:
  match wanted:
    case True{}: KernelGemm.Both<{value == original(logical,index) : F32},{U32.is_lt(index,capacity) == True{} : Bool}>
    case False{}: Unit

def AllRight(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+capacity: U32,lanes: List<&2,Lane>) -> Data:
  match lanes:
    case Nil{}: Unit
    case Con{Lane{+wanted,+index,+value},rest}:
      KernelGemm.Both<Right(~Logical,~original,logical,capacity,wanted,index,value),AllRight(~Logical,~original,logical,capacity,rest)>

# The lane at a position writes at an index.
def HitsAt(position: Nat,lanes: List<&2,Lane>,+query: Nat) -> Data:
  match position lanes:
    case position Nil{}: Empty
    case 0n Con{Lane{+wanted,+index,+value},rest}: KernelGemm.Both<{wanted == True{} : Bool},{index == U32.from_nat(query) : U32}>
    case 1n+next Con{lane,rest}: HitsAt(next,rest,query)

def lane_kept(wanted: Bool,+depth: Nat,+capacity: U32,+size: Nat,+index: U32,+value: F32,+content: Buffer.Content<F32>,+kept: Kept<depth,capacity,size,content>) ->
  Kept<depth,capacity,size,lane_model(wanted,index,value,content)>:
  match wanted:
    case True{}:
      Buffer.Content{+tree,+length} = content
      Kept{+balanced,+sized,+long} = kept
      Kept{Shape.write(F32,depth,tree,index,value,balanced),
        Equal.trans(U32,Tree.capacity(F32,Tree.write(F32,tree,index,value)),Tree.capacity(F32,tree),capacity,StorageReadback.write_capacity(~F32,tree,index,value),sized),long}
    case False{}: kept

def lanes_kept(lanes: List<&2,Lane>,+depth: Nat,+capacity: U32,+size: Nat,+content: Buffer.Content<F32>,+kept: Kept<depth,capacity,size,content>) ->
  Kept<depth,capacity,size,lanes_model(lanes,content)>:
  match lanes:
    case Nil{}: kept
    case Con{Lane{+wanted,+index,+value},rest}:
      lanes_kept(rest,depth,capacity,size,lane_model(wanted,index,value,content),lane_kept(wanted,depth,capacity,size,index,value,content,kept))

# A read after a write of the original at an index holds the original when it
# did before or reads that index.
def write_cases(~Logical: Data,~original: Logical -> U32 -> F32,same: Bool,+logical: Logical,+depth: Nat,+tree: Tree.Tree<F32>,+index: U32,+value: F32,+query: Nat,
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},+balanced: Shape.Balanced(F32,depth,tree),
  +index_inside: {U32.is_lt(index,Tree.capacity(F32,tree)) == True{} : Bool},+query_inside: {U32.is_lt(U32.from_nat(query),Tree.capacity(F32,tree)) == True{} : Bool},
  +exact: {value == original(logical,index) : F32},+classified: {U32.is_eq(index,U32.from_nat(query)) == same : Bool},
  good: {same == False{} : Bool} -> {Tree.read(F32,tree,U32.from_nat(query)) == original(logical,U32.from_nat(query)) : F32}) ->
  {Tree.read(F32,Tree.write(F32,tree,index,value),U32.from_nat(query)) == original(logical,U32.from_nat(query)) : F32}:
  match same:
    case True{}:
      %Arithmetic.equal_of_is_eq(index,U32.from_nat(query),classified) : {Tree.read(F32,Tree.write(F32,tree,index,value),_) == original(logical,_) : F32}
      Equal.trans(F32,Tree.read(F32,Tree.write(F32,tree,index,value),index),value,original(logical,index),StorageReadback.read_written(~F32,tree,index,value),exact)
    case False{}:
      Equal.trans(F32,Tree.read(F32,Tree.write(F32,tree,index,value),U32.from_nat(query)),Tree.read(F32,tree,U32.from_nat(query)),original(logical,U32.from_nat(query)),
        StorageReadback.read_preserved(~F32,depth,tree,index,U32.from_nat(query),value,balanced,limit,index_inside,query_inside,classified),good({==}))

def inside_tree(+index: U32,+capacity: U32,+tree: Tree.Tree<F32>,+sized: {Tree.capacity(F32,tree) == capacity : U32},+inside: {U32.is_lt(index,capacity) == True{} : Bool}) ->
  {U32.is_lt(index,Tree.capacity(F32,tree)) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(F32,tree),capacity,sized) : {U32.is_lt(index,_) == True{} : Bool}
  inside

# A right lane keeps the original where contents held it.
def lane_good(~Logical: Data,~original: Logical -> U32 -> F32,wanted: Bool,+logical: Logical,+depth: Nat,+capacity: U32,+size: Nat,+index: U32,+value: F32,
  +content: Buffer.Content<F32>,+query: Nat,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+kept: Kept<depth,capacity,size,content>,
  +right: Right(~Logical,~original,logical,capacity,wanted,index,value),+query_inside: {U32.is_lt(U32.from_nat(query),capacity) == True{} : Bool},
  +good: Good(~Logical,~original,logical,content,query)) -> Good(~Logical,~original,logical,lane_model(wanted,index,value,content),query):
  match wanted:
    case True{}:
      Buffer.Content{+tree,+length} = content
      Kept{+balanced,+sized,+long} = kept
      KernelGemm.Both{+exact,+inside} = right
      write_cases(~Logical,~original,U32.is_eq(index,U32.from_nat(query)),logical,depth,tree,index,value,query,limit,balanced,inside_tree(index,capacity,tree,sized,inside),
        inside_tree(U32.from_nat(query),capacity,tree,sized,query_inside),exact,{==},different => good)
    case False{}: good

# A right lane that writes at the index holds the original there.
def lane_hit(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+depth: Nat,+capacity: U32,+size: Nat,+index: U32,+value: F32,
  +content: Buffer.Content<F32>,+query: Nat,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+kept: Kept<depth,capacity,size,content>,
  +right: Right(~Logical,~original,logical,capacity,True{},index,value),+query_inside: {U32.is_lt(U32.from_nat(query),capacity) == True{} : Bool},
  +hit: {index == U32.from_nat(query) : U32}) -> Good(~Logical,~original,logical,lane_model(True{},index,value,content),query):
  Buffer.Content{+tree,+length} = content
  Kept{+balanced,+sized,+long} = kept
  KernelGemm.Both{+exact,+inside} = right
  +same = {%hit : {U32.is_eq(index,_) == True{} : Bool} Arithmetic.equal_reflexive(index) : {U32.is_eq(index,U32.from_nat(query)) == True{} : Bool}}
  write_cases(~Logical,~original,True{},logical,depth,tree,index,value,query,limit,balanced,inside_tree(index,capacity,tree,sized,inside),
    inside_tree(U32.from_nat(query),capacity,tree,sized,query_inside),exact,same,
    different => Empty.absurd({Tree.read(F32,tree,U32.from_nat(query)) == original(logical,U32.from_nat(query)) : F32},WordEncoding.false_true(Equal.sym(Bool,True{},False{},different))))

# Right lanes keep the original where contents held it.
def lanes_good(~Logical: Data,~original: Logical -> U32 -> F32,lanes: List<&2,Lane>,+logical: Logical,+depth: Nat,+capacity: U32,+size: Nat,+content: Buffer.Content<F32>,
  +query: Nat,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+kept: Kept<depth,capacity,size,content>,+rights: AllRight(~Logical,~original,logical,capacity,lanes),
  +query_inside: {U32.is_lt(U32.from_nat(query),capacity) == True{} : Bool},+good: Good(~Logical,~original,logical,content,query)) ->
  Good(~Logical,~original,logical,lanes_model(lanes,content),query):
  match lanes:
    case Nil{}: good
    case Con{Lane{+wanted,+index,+value},rest}:
      KernelGemm.Both{+right,+later} = rights
      lanes_good(~Logical,~original,rest,logical,depth,capacity,size,lane_model(wanted,index,value,content),query,limit,lane_kept(wanted,depth,capacity,size,index,value,content,kept),
        later,query_inside,lane_good(~Logical,~original,wanted,logical,depth,capacity,size,index,value,content,query,limit,kept,right,query_inside,good))

# Right lanes one of which writes at the index hold the original there.
def lanes_hit(~Logical: Data,~original: Logical -> U32 -> F32,position: Nat,lanes: List<&2,Lane>,+logical: Logical,+depth: Nat,+capacity: U32,+size: Nat,
  +content: Buffer.Content<F32>,+query: Nat,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+kept: Kept<depth,capacity,size,content>,
  +rights: AllRight(~Logical,~original,logical,capacity,lanes),+query_inside: {U32.is_lt(U32.from_nat(query),capacity) == True{} : Bool},
  +hits: HitsAt(position,lanes,query)) -> Good(~Logical,~original,logical,lanes_model(lanes,content),query):
  match position lanes:
    case position Nil{}: Empty.absurd(Good(~Logical,~original,logical,content,query),hits)
    case 0n Con{Lane{+wanted,+index,+value},rest}:
      KernelGemm.Both{+right,+later} = rights
      KernelGemm.Both{+wanted_true,+hit} = hits
      %Equal.sym(Bool,wanted,True{},wanted_true) : Good(~Logical,~original,logical,lanes_model(rest,lane_model(_,index,value,content)),query)
      lanes_good(~Logical,~original,rest,logical,depth,capacity,size,lane_model(True{},index,value,content),query,limit,lane_kept(True{},depth,capacity,size,index,value,content,kept),
        later,query_inside,
        lane_hit(~Logical,~original,logical,depth,capacity,size,index,value,content,query,limit,kept,
          %wanted_true : Right(~Logical,~original,logical,capacity,_,index,value) right,query_inside,hit))
    case 1n+ +next Con{Lane{+wanted,+index,+value},rest}:
      KernelGemm.Both{+right,+later} = rights
      lanes_hit(~Logical,~original,next,rest,logical,depth,capacity,size,lane_model(wanted,index,value,content),query,limit,
        lane_kept(wanted,depth,capacity,size,index,value,content,kept),later,query_inside,hits)

# ---- The lanes of a cell ----

def cell_lanes(+base: U32,+step: U32,+valid: U32,tile: Tile.Tile8) -> List<&2,Lane>:
  Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h} = tile
  [Lane{U32.is_lt(0,valid),base,a},Lane{U32.is_lt(1,valid),(base + step : U32),b},Lane{U32.is_lt(2,valid),(base + 2 * step : U32),c},
    Lane{U32.is_lt(3,valid),(base + 3 * step : U32),d},Lane{U32.is_lt(4,valid),(base + 4 * step : U32),e},Lane{U32.is_lt(5,valid),(base + 5 * step : U32),f},
    Lane{U32.is_lt(6,valid),(base + 6 * step : U32),g},Lane{U32.is_lt(7,valid),(base + 7 * step : U32),h}]

def cell_listed(+base: U32,+step: U32,+valid: U32,+tile: Tile.Tile8,+content: Buffer.Content<F32>) ->
  {cell_model(base,step,valid,tile,content) == lanes_model(cell_lanes(base,step,valid,tile),content) : Buffer.Content<F32>}:
  Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h} = tile
  {==}

# The lanes of the cell at a cell index, as the runtime places them.
def written_lanes(+context: Cells.Cells,+cell: U32,tile: Tile.Tile8) -> List<&2,Lane>:
  Cells.Cells{+outer,+after,+lanes,+groups,+position} = context
  +group = Bool.pick(U32,Grid.is_outermost(position),(cell / U32.max(outer,1) : U32),(cell % U32.max(groups,1) : U32))
  +index = Bool.pick(U32,Grid.is_outermost(position),(cell % U32.max(outer,1) : U32),(cell / U32.max(groups,1) : U32))
  +inner = U32.max(after,1)
  cell_lanes(((index / inner * lanes + group * 8) * inner + index % inner : U32),inner,(lanes - group * 8 : U32),tile)

def written_listed(+context: Cells.Cells,+cell: U32,+tile: Tile.Tile8,+content: Buffer.Content<F32>) ->
  {written_model(context,cell,tile,content) == lanes_model(written_lanes(context,cell,tile),content) : Buffer.Content<F32>}:
  Cells.Cells{+outer,+after,+lanes,+groups,+position} = context
  +group = Bool.pick(U32,Grid.is_outermost(position),(cell / U32.max(outer,1) : U32),(cell % U32.max(groups,1) : U32))
  +index = Bool.pick(U32,Grid.is_outermost(position),(cell % U32.max(outer,1) : U32),(cell / U32.max(groups,1) : U32))
  +inner = U32.max(after,1)
  cell_listed(((index / inner * lanes + group * 8) * inner + index % inner : U32),inner,(lanes - group * 8 : U32),tile,content)

# A lane holds a value exactly when its channel lies below the lanes.
def lane_wanted(+limits: Machine.Limits,+lanes: U32,+group_word: U32,+group: Nat,+lane: Nat,+group_exact: {U32.to_nat(group_word) == group : Nat},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(lanes))) == True{} : Bool},
  +bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(lane),(lanes - group_word * 8 : U32)) == Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(lanes)) : Bool}:
  +count = U32.to_nat(lanes)
  +offset = Nat.mul(group,8n)
  +capacity = Machine.capacity(limits)
  +fits = Limits.capacity_fits(limits)
  +left = (lanes - group_word * 8 : U32)
  +rest = Nat.sub(count,offset)
  +inside = {%Encoded.eight_blocks_exact(limits,lanes,bound) : {Nat.is_lt(group,_) == True{} : Bool} group_inside : {Nat.is_lt(group,Ceiling.ceiling(count,8n)) == True{} : Bool}}
  +past = {%Ceiling.upper_comparison(count,8n,group,{==}) : {_ == True{} : Bool} inside : {Nat.is_lt(offset,count) == True{} : Bool}}
  +covered = Bounds.strict_le(offset,count,past)
  +product = Arithmetic.product_exact(capacity,fits,group_word,8,group,8n,group_exact,{==},Order.transitive(offset,count,U32.to_nat(capacity),covered,bound))
  +difference = Arithmetic.difference_exact(capacity,fits,lanes,(group_word * 8 : U32),count,offset,{==},product,covered,bound)
  +room = Order.transitive(lane,7n,U32.to_nat(capacity),{Order.strict_follow(lane,8n,small) : {Nat.is_le(lane,7n) == True{} : Bool}},
    Bounds.as_nat_le(7,capacity,Limits.seven_within_capacity(limits)))
  %Order.commutative(offset,lane) : {U32.is_lt(U32.from_nat(lane),left) == Nat.is_lt(_,count) : Bool}
  %Order.restore_shift(offset,count,covered) : {U32.is_lt(U32.from_nat(lane),left) == Nat.is_lt(Nat.add(offset,lane),_) : Bool}
  %Equal.sym(Bool,Nat.is_lt(Nat.add(offset,lane),Nat.add(offset,rest)),Nat.is_lt(lane,rest),Order.lt_offset(offset,lane,rest)) : {U32.is_lt(U32.from_nat(lane),left) == _ : Bool}
  %difference : {U32.is_lt(U32.from_nat(lane),left) == Nat.is_lt(lane,_) : Bool}
  %Bounds.roundtrip(lane,capacity,fits,room) : {U32.is_lt(U32.from_nat(lane),left) == Nat.is_lt(_,U32.to_nat(left)) : Bool}
  Comparison.lt(U32.from_nat(lane),left)

# ---- Every lane of a cell writes the original at its index ----

# The lanes of the cell at a cell word, with the outer index and the group the
# runtime splits the word into.
def written_words(position: Grid.Position,+outer: U32,+after: U32,+lanes: U32,+word: U32,+tile: Tile.Tile8) ->
  {written_lanes(Cells.Cells{outer,after,lanes,Grid.groups(lanes),position},word,tile) ==
    cell_lanes((((TiledIndexedValues.outer_word(position,word,outer,Grid.groups(lanes)) / U32.max(after,1)) * lanes +
        TiledIndexedValues.group_word(position,word,outer,Grid.groups(lanes)) * 8) * U32.max(after,1) +
        TiledIndexedValues.outer_word(position,word,outer,Grid.groups(lanes)) % U32.max(after,1) : U32),
      U32.max(after,1),(lanes - TiledIndexedValues.group_word(position,word,outer,Grid.groups(lanes)) * 8 : U32),tile) : List<&2,Lane>}:
  match position:
    case Grid.Innermost{}: {==}
    case Grid.Outermost{}: {==}

def decoded_group(+outer_word: U32,+group_word: U32,+geometry: Cells.Cells,+number: Nat,
  +decoded: TiledIndexedValues.Decoded<outer_word,group_word,geometry,number>) -> {U32.to_nat(group_word) == Representation.cell_group(geometry,number) : Nat}:
  TiledIndexedValues.Decoded{+outer_exact,+group_exact} = decoded
  group_exact

# The values of a geometry fill whole blocks of their outer indices within the
# capacity.
type Room<-geometry: Cells.Cells,-capacity: U32> is Data:
  Room{blocks: Nat,outer_exact: {Representation.outer_count(geometry) == Nat.mul(blocks,Representation.after_count(geometry)) : Nat},
    total: {Nat.is_le(Nat.mul(Nat.mul(blocks,Representation.lane_count(geometry)),Representation.after_count(geometry)),U32.to_nat(capacity)) == True{} : Bool},
    fits: {U32.is_le(capacity,2147483648) == True{} : Bool}}

# A lane that holds a value exactly below the lanes, at the index of its
# channel, writes the original there.
def right_cases(~Logical: Data,~original: Logical -> U32 -> F32,valid: Bool,+logical: Logical,+capacity: U32,+geometry: Cells.Cells,+at: Nat,+group: Nat,+lane: Nat,
  +index: U32,+classified: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Representation.lane_count(geometry)) == valid : Bool},
  +index_exact: {index == U32.from_nat(Representation.logical_index(geometry,at,Nat.add(lane,Nat.mul(group,8n)))) : U32},
  inside: {valid == True{} : Bool} -> {U32.is_lt(index,capacity) == True{} : Bool}) ->
  Right(~Logical,~original,logical,capacity,valid,index,Representation.lane(~Logical,~original,logical,geometry,at,group,lane)):
  match valid:
    case True{}:
      +target = U32.from_nat(Representation.logical_index(geometry,at,Nat.add(lane,Nat.mul(group,8n))))
      KernelGemm.Both{
        %Equal.sym(Bool,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Representation.lane_count(geometry)),True{},classified) :
          {Bool.pick(F32,_,original(logical,target),0.0) == original(logical,index) : F32}
        Equal.cong(U32,F32,word => original(logical,word),target,index,Equal.sym(U32,index,target,index_exact)),
        inside({==})}
    case False{}: Unit{}

def lane_right(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+capacity: U32,+geometry: Cells.Cells,+at: Nat,+group: Nat,+lane: Nat,
  +wanted: Bool,+index: U32,+wanted_exact: {wanted == Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Representation.lane_count(geometry)) : Bool},
  +index_exact: {index == U32.from_nat(Representation.logical_index(geometry,at,Nat.add(lane,Nat.mul(group,8n)))) : U32},
  inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Representation.lane_count(geometry)) == True{} : Bool} -> {U32.is_lt(index,capacity) == True{} : Bool}) ->
  Right(~Logical,~original,logical,capacity,wanted,index,Representation.lane(~Logical,~original,logical,geometry,at,group,lane)):
  %Equal.sym(Bool,wanted,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Representation.lane_count(geometry)),wanted_exact) :
    Right(~Logical,~original,logical,capacity,_,index,Representation.lane(~Logical,~original,logical,geometry,at,group,lane))
  right_cases(~Logical,~original,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Representation.lane_count(geometry)),logical,capacity,geometry,at,group,lane,index,{==},index_exact,inside)

# A natural below a bound within a capacity is a word below it.
def word_below(+value: Nat,+bound: Nat,+capacity: U32,+fits: {U32.is_le(capacity,2147483648) == True{} : Bool},+inside: {Nat.is_lt(value,bound) == True{} : Bool},
  +room: {Nat.is_le(bound,U32.to_nat(capacity)) == True{} : Bool}) -> {U32.is_lt(U32.from_nat(value),capacity) == True{} : Bool}:
  +strict = Bounds.nat_trans(value,bound,U32.to_nat(capacity),inside,room)
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(value),capacity),Nat.is_lt(U32.to_nat(U32.from_nat(value)),U32.to_nat(capacity)),Comparison.lt(U32.from_nat(value),capacity)) :
    {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(value)),value,Bounds.roundtrip(value,capacity,fits,Bounds.strict_le(value,U32.to_nat(capacity),strict))) :
    {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
  strict

def lane_room(+geometry: Cells.Cells,+capacity: U32,+at: Nat,+channel: Nat,+index: U32,+room: Room<geometry,capacity>,
  +at_inside: {Nat.is_lt(at,Representation.outer_count(geometry)) == True{} : Bool},
  +index_exact: {index == U32.from_nat(Representation.logical_index(geometry,at,channel)) : U32},
  +valid: {Nat.is_lt(channel,Representation.lane_count(geometry)) == True{} : Bool}) -> {U32.is_lt(index,capacity) == True{} : Bool}:
  Room{+blocks,+outer_exact,+total,+fits} = room
  +value = Representation.logical_index(geometry,at,channel)
  +bound = Nat.mul(Nat.mul(blocks,Representation.lane_count(geometry)),Representation.after_count(geometry))
  +below = Representation.logical_below(geometry,at,channel,blocks,%outer_exact :
    {Nat.is_lt(at,_) == True{} : Bool} at_inside,valid)
  %Equal.sym(U32,index,U32.from_nat(value),index_exact) : {U32.is_lt(_,capacity) == True{} : Bool}
  word_below(value,bound,capacity,fits,below,total)

# What a lane of a cell writes: whether its channel lies below the lanes, the
# index of its channel and the lane of the cell.
def LanePlaced(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+wanted: Bool,+index: U32,+value: F32,+geometry: Cells.Cells,+at: Nat,+group: Nat,
  +lane: Nat) -> Data:
  KernelGemm.Both<{wanted == Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Representation.lane_count(geometry)) : Bool},
    KernelGemm.Both<{index == U32.from_nat(Representation.logical_index(geometry,at,Nat.add(lane,Nat.mul(group,8n)))) : U32},
      {value == Representation.lane(~Logical,~original,logical,geometry,at,group,lane) : F32}>>

def Placed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,lanes: List<&2,Lane>,+geometry: Cells.Cells,+at: Nat,+group: Nat,+lane: Nat) -> Data:
  match lanes:
    case Nil{}: Unit
    case Con{Lane{+wanted,+index,+value},rest}:
      KernelGemm.Both<LanePlaced(~Logical,~original,logical,wanted,index,value,geometry,at,group,lane),Placed(~Logical,~original,logical,rest,geometry,at,group,1n+lane)>

# Every lane of the cell at a word below the cells holds a value exactly below
# the lanes, at the index of its channel.
def cell_placed(~Logical: Data,~original: Logical -> U32 -> F32,+position: Grid.Position,+logical: Logical,+limits: Machine.Limits,+outer: U32,+after: U32,+lanes: U32,
  +word: U32,+divides: TiledIndexedValues.Divides<limits,outer,lanes>,+after_spreads: TiledIndexedValues.Spreads<after>,
  +bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(word),Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))) == True{} : Bool}) ->
  Placed(~Logical,~original,logical,written_lanes(Cells.Cells{outer,after,lanes,Grid.groups(lanes),position},word,
    Representation.cell(~Logical,~original,logical,Cells.Cells{outer,after,lanes,Grid.groups(lanes),position},
      Representation.cell_outer(Cells.Cells{outer,after,lanes,Grid.groups(lanes),position},U32.to_nat(word)),
      Representation.cell_group(Cells.Cells{outer,after,lanes,Grid.groups(lanes),position},U32.to_nat(word)))),
    Cells.Cells{outer,after,lanes,Grid.groups(lanes),position},
    Representation.cell_outer(Cells.Cells{outer,after,lanes,Grid.groups(lanes),position},U32.to_nat(word)),
    Representation.cell_group(Cells.Cells{outer,after,lanes,Grid.groups(lanes),position},U32.to_nat(word)),0n):
  TiledIndexedValues.Divides{+outer_positive,+groups_positive,+outer_bound,+groups_bound} = divides
  TiledIndexedValues.Spreads{+cap,+cap_ok,+after_positive,+after_bounded} = after_spreads
  +groups = Grid.groups(lanes)
  +geometry = {Cells.Cells{outer,after,lanes,groups,position} : Cells.Cells}
  +number = U32.to_nat(word)
  +at = Representation.cell_outer(geometry,number)
  +group = Representation.cell_group(geometry,number)
  +tile = Representation.cell(~Logical,~original,logical,geometry,at,group)
  +outer_word = TiledIndexedValues.outer_word(position,word,outer,groups)
  +group_word = TiledIndexedValues.group_word(position,word,outer,groups)
  +inner = U32.max(after,1)
  +valid = (lanes - group_word * 8 : U32)
  +base = (((outer_word / inner) * lanes + group_word * 8) * inner + outer_word % inner : U32)
  +spreads = {TiledIndexedValues.Spreads{cap,cap_ok,after_positive,after_bounded} : TiledIndexedValues.Spreads<after>}
  +decoded = TiledIndexedValues.decode_along(position,Machine.half_word(limits),Limits.half_word_fits(limits),word,outer,after,lanes,groups,outer_positive,groups_positive,
    outer_bound,groups_bound)
  +count = Representation.after_count(geometry)
  +lane_total = Representation.lane_count(geometry)
  +quotient = Nat.div(at,count)
  +rest = Nat.mod(at,count)
  +start = TiledIndexedValues.index_natural(quotient,lane_total,group,count,rest)
  +base_exact = Equal.trans(U32,base,TiledIndexed.index_of(outer_word,group_word,after,lanes),U32.from_nat(start),
    Equal.cong(U32,U32,step => (((outer_word / inner) * lanes + group_word * 8) * step + outer_word % inner : U32),inner,after,Depth.positive_max(after,after_positive)),
    Equal.trans(U32,TiledIndexed.index_of(outer_word,group_word,after,lanes),TiledIndexed.index_at(position,word,outer,after,lanes,groups),U32.from_nat(start),
      Equal.sym(U32,TiledIndexed.index_at(position,word,outer,after,lanes,groups),TiledIndexed.index_of(outer_word,group_word,after,lanes),
        TiledIndexedValues.index_split(position,word,outer,after,lanes,groups)),
      TiledIndexedValues.index_exact(position,word,outer,groups,outer,after,lanes,position,spreads,decoded)))
  +step_exact = Equal.trans(U32,inner,after,U32.from_nat(count),Depth.positive_max(after,after_positive),TiledIndexedValues.step_exact(after,spreads))
  +group_exact = decoded_group(outer_word,group_word,geometry,number,decoded)
  +group_in = Representation.group_inside(position,U32.to_nat(outer),U32.to_nat(groups),number,inside,outer_positive,groups_positive)
  +index1 = Representation.logical_index(geometry,at,Nat.add(1n,Nat.mul(group,8n)))
  +index2 = Representation.logical_index(geometry,at,Nat.add(2n,Nat.mul(group,8n)))
  +index3 = Representation.logical_index(geometry,at,Nat.add(3n,Nat.mul(group,8n)))
  +index4 = Representation.logical_index(geometry,at,Nat.add(4n,Nat.mul(group,8n)))
  +index5 = Representation.logical_index(geometry,at,Nat.add(5n,Nat.mul(group,8n)))
  +index6 = Representation.logical_index(geometry,at,Nat.add(6n,Nat.mul(group,8n)))
  +index7 = Representation.logical_index(geometry,at,Nat.add(7n,Nat.mul(group,8n)))
  +exact1 = TiledIndexedValues.lane_index(base,inner,start,Nat.mul(1n,count),index1,base_exact,TiledIndexedValues.step_one(inner,count,step_exact),
    TiledIndexedValues.lane_shift(quotient,lane_total,group,count,rest,1n))
  +exact2 = TiledIndexedValues.lane_index(base,(2 * inner : U32),start,Nat.mul(2n,count),index2,base_exact,TiledIndexedValues.step_times(2n,inner,count,step_exact),
    TiledIndexedValues.lane_shift(quotient,lane_total,group,count,rest,2n))
  +exact3 = TiledIndexedValues.lane_index(base,(3 * inner : U32),start,Nat.mul(3n,count),index3,base_exact,TiledIndexedValues.step_times(3n,inner,count,step_exact),
    TiledIndexedValues.lane_shift(quotient,lane_total,group,count,rest,3n))
  +exact4 = TiledIndexedValues.lane_index(base,(4 * inner : U32),start,Nat.mul(4n,count),index4,base_exact,TiledIndexedValues.step_times(4n,inner,count,step_exact),
    TiledIndexedValues.lane_shift(quotient,lane_total,group,count,rest,4n))
  +exact5 = TiledIndexedValues.lane_index(base,(5 * inner : U32),start,Nat.mul(5n,count),index5,base_exact,TiledIndexedValues.step_times(5n,inner,count,step_exact),
    TiledIndexedValues.lane_shift(quotient,lane_total,group,count,rest,5n))
  +exact6 = TiledIndexedValues.lane_index(base,(6 * inner : U32),start,Nat.mul(6n,count),index6,base_exact,TiledIndexedValues.step_times(6n,inner,count,step_exact),
    TiledIndexedValues.lane_shift(quotient,lane_total,group,count,rest,6n))
  +exact7 = TiledIndexedValues.lane_index(base,(7 * inner : U32),start,Nat.mul(7n,count),index7,base_exact,TiledIndexedValues.step_times(7n,inner,count,step_exact),
    TiledIndexedValues.lane_shift(quotient,lane_total,group,count,rest,7n))
  %Equal.sym(List<&2,Lane>,written_lanes(geometry,word,tile),cell_lanes(base,inner,valid,tile),written_words(position,outer,after,lanes,word,tile)) :
    Placed(~Logical,~original,logical,_,geometry,at,group,0n)
  KernelGemm.Both{
    KernelGemm.Both{lane_wanted(limits,lanes,group_word,group,0n,group_exact,group_in,bound,{==}),KernelGemm.Both{base_exact,{==}}},
  KernelGemm.Both{
    KernelGemm.Both{lane_wanted(limits,lanes,group_word,group,1n,group_exact,group_in,bound,{==}),KernelGemm.Both{exact1,{==}}},
  KernelGemm.Both{
    KernelGemm.Both{lane_wanted(limits,lanes,group_word,group,2n,group_exact,group_in,bound,{==}),KernelGemm.Both{exact2,{==}}},
  KernelGemm.Both{
    KernelGemm.Both{lane_wanted(limits,lanes,group_word,group,3n,group_exact,group_in,bound,{==}),KernelGemm.Both{exact3,{==}}},
  KernelGemm.Both{
    KernelGemm.Both{lane_wanted(limits,lanes,group_word,group,4n,group_exact,group_in,bound,{==}),KernelGemm.Both{exact4,{==}}},
  KernelGemm.Both{
    KernelGemm.Both{lane_wanted(limits,lanes,group_word,group,5n,group_exact,group_in,bound,{==}),KernelGemm.Both{exact5,{==}}},
  KernelGemm.Both{
    KernelGemm.Both{lane_wanted(limits,lanes,group_word,group,6n,group_exact,group_in,bound,{==}),KernelGemm.Both{exact6,{==}}},
  KernelGemm.Both{
    KernelGemm.Both{lane_wanted(limits,lanes,group_word,group,7n,group_exact,group_in,bound,{==}),KernelGemm.Both{exact7,{==}}},
    Unit{}}}}}}}}}

# Placed lanes of an outer index below the outer count write the original.
def placed_rights(~Logical: Data,~original: Logical -> U32 -> F32,lanes: List<&2,Lane>,+logical: Logical,+capacity: U32,+geometry: Cells.Cells,+at: Nat,+group: Nat,
  +lane: Nat,+placed: Placed(~Logical,~original,logical,lanes,geometry,at,group,lane),+room: Room<geometry,capacity>,
  +at_inside: {Nat.is_lt(at,Representation.outer_count(geometry)) == True{} : Bool}) -> AllRight(~Logical,~original,logical,capacity,lanes):
  match lanes:
    case Nil{}: Unit{}
    case Con{Lane{+wanted,+index,+value},rest}:
      KernelGemm.Both{KernelGemm.Both{+wanted_exact,KernelGemm.Both{+index_exact,+value_exact}},+later} = placed
      KernelGemm.Both{
        %Equal.sym(F32,value,Representation.lane(~Logical,~original,logical,geometry,at,group,lane),value_exact) :
          Right(~Logical,~original,logical,capacity,wanted,index,_)
        lane_right(~Logical,~original,logical,capacity,geometry,at,group,lane,wanted,index,wanted_exact,index_exact,
          ok => lane_room(geometry,capacity,at,Nat.add(lane,Nat.mul(group,8n)),index,room,at_inside,index_exact,ok)),
        placed_rights(~Logical,~original,rest,logical,capacity,geometry,at,group,1n+lane,later,room,at_inside)}

# The placed lane of a channel below the lanes whose index is the query writes
# there.
def placed_hits(~Logical: Data,~original: Logical -> U32 -> F32,position: Nat,lanes: List<&2,Lane>,+logical: Logical,+geometry: Cells.Cells,+at: Nat,+group: Nat,
  +lane: Nat,+target: Nat,+query: Nat,+placed: Placed(~Logical,~original,logical,lanes,geometry,at,group,lane),
  +inside: {Nat.is_lt(position,List.length(&2,Lane,lanes)) == True{} : Bool},+exact: {Nat.add(position,lane) == target : Nat},
  +valid: {Nat.is_lt(Nat.add(target,Nat.mul(group,8n)),Representation.lane_count(geometry)) == True{} : Bool},
  +hit: {Representation.logical_index(geometry,at,Nat.add(target,Nat.mul(group,8n))) == query : Nat}) -> HitsAt(position,lanes,query):
  match position lanes:
    case position Nil{}: Empty.absurd(HitsAt(position,Nil{},query),WordEncoding.false_true(inside))
    case 0n Con{Lane{+wanted,+index,+value},rest}:
      KernelGemm.Both{KernelGemm.Both{+wanted_exact,KernelGemm.Both{+index_exact,+value_exact}},+later} = placed
      +channel = Nat.add(lane,Nat.mul(group,8n))
      KernelGemm.Both{
        Equal.trans(Bool,wanted,Nat.is_lt(channel,Representation.lane_count(geometry)),True{},wanted_exact,
          %Equal.sym(Nat,lane,target,exact) : {Nat.is_lt(Nat.add(_,Nat.mul(group,8n)),Representation.lane_count(geometry)) == True{} : Bool} valid),
        Equal.trans(U32,index,U32.from_nat(Representation.logical_index(geometry,at,channel)),U32.from_nat(query),index_exact,
          Equal.cong(Nat,U32,U32.from_nat,Representation.logical_index(geometry,at,channel),query,
            %Equal.sym(Nat,lane,target,exact) : {Representation.logical_index(geometry,at,Nat.add(_,Nat.mul(group,8n))) == query : Nat} hit))}
    case 1n+ +next Con{Lane{+wanted,+index,+value},rest}:
      KernelGemm.Both{+here,+later} = placed
      placed_hits(~Logical,~original,next,rest,logical,geometry,at,group,1n+lane,target,query,later,inside,
        Equal.trans(Nat,Nat.add(next,1n+lane),1n+Nat.add(next,lane),target,Addition.successor_right(next,lane),exact),valid,hit)

# ---- The lane that writes a logical index ----

def decoded_outer(+geometry: Cells.Cells,+query: Nat,+positive: {Nat.is_lt(0n,Representation.lane_count(geometry)) == True{} : Bool}) ->
  {Representation.value_outer(geometry,query) ==
    Nat.add(Nat.mul(Nat.div(Nat.div(query,Representation.after_count(geometry)),Representation.lane_count(geometry)),Representation.after_count(geometry)),
      Nat.mod(query,Representation.after_count(geometry))) : Nat}:
  +after = Representation.after_count(geometry)
  +count = Representation.lane_count(geometry)
  %Equal.sym(Nat,Representation.least_one(count),count,Representation.least_positive(count,positive)) :
    {Nat.add(Nat.mul(Nat.div(Nat.div(query,after),_),after),Nat.mod(query,after)) == Nat.add(Nat.mul(Nat.div(Nat.div(query,after),count),after),Nat.mod(query,after)) : Nat}
  {==}

def decoded_channel(+geometry: Cells.Cells,+query: Nat,+positive: {Nat.is_lt(0n,Representation.lane_count(geometry)) == True{} : Bool}) ->
  {Representation.value_channel(geometry,query) == Nat.mod(Nat.div(query,Representation.after_count(geometry)),Representation.lane_count(geometry)) : Nat}:
  +after = Representation.after_count(geometry)
  +count = Representation.lane_count(geometry)
  %Equal.sym(Nat,Representation.least_one(count),count,Representation.least_positive(count,positive)) :
    {Nat.mod(Nat.div(query,after),_) == Nat.mod(Nat.div(query,after),count) : Nat}
  {==}

# The outer index and the channel of a logical index below whole blocks.
type Decoded<-geometry: Cells.Cells,-query: Nat> is Data:
  Decoded{at_inside: {Nat.is_lt(Representation.value_outer(geometry,query),Representation.outer_count(geometry)) == True{} : Bool},
    channel_inside: {Nat.is_lt(Representation.value_channel(geometry,query),Representation.lane_count(geometry)) == True{} : Bool},
    exact: {Representation.logical_index(geometry,Representation.value_outer(geometry,query),Representation.value_channel(geometry,query)) == query : Nat}}

def decode_query(+geometry: Cells.Cells,+blocks: Nat,+query: Nat,
  +outer_exact: {Representation.outer_count(geometry) == Nat.mul(blocks,Representation.after_count(geometry)) : Nat},
  +inside: {Nat.is_lt(query,Nat.mul(Nat.mul(blocks,Representation.lane_count(geometry)),Representation.after_count(geometry))) == True{} : Bool}) ->
  Decoded<geometry,query>:
  +after = Representation.after_count(geometry)
  +count = Representation.lane_count(geometry)
  +positive = Representation.below_positive(query,Nat.mul(Nat.mul(blocks,count),after),inside)
  +after_positive = DivisionOrder.right_factor_positive(Nat.mul(blocks,count),after,positive)
  +lane_positive = DivisionOrder.right_factor_positive(blocks,count,RowMajor.left_factor_positive(Nat.mul(blocks,count),after,positive))
  +quotient = Nat.div(query,after)
  +remainder = Nat.mod(query,after)
  +block = Nat.div(quotient,count)
  +channel = Nat.mod(quotient,count)
  +at = Nat.add(Nat.mul(block,after),remainder)
  +quotient_inside = DivisionOrder.quotient_below(query,Nat.mul(blocks,count),after,after_positive,inside)
  +block_inside = DivisionOrder.quotient_below(quotient,blocks,count,lane_positive,quotient_inside)
  +remainder_small = DivisionOrder.remainder_small(query,after,after_positive)
  +split = Division.unique(at,after,block,remainder,{==},after_positive,remainder_small)
  +at_quotient = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(at,after),(block,remainder),split)
  +at_remainder = Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(at,after),(block,remainder),split)
  +outer_value = decoded_outer(geometry,query,lane_positive)
  +channel_value = decoded_channel(geometry,query,lane_positive)
  Decoded{
    %Equal.sym(Nat,Representation.value_outer(geometry,query),at,outer_value) : {Nat.is_lt(_,Representation.outer_count(geometry)) == True{} : Bool}
    %Equal.sym(Nat,Representation.outer_count(geometry),Nat.mul(blocks,after),outer_exact) : {Nat.is_lt(at,_) == True{} : Bool}
    RowMajor.radix(block,blocks,remainder,after,block_inside,remainder_small),
    %Equal.sym(Nat,Representation.value_channel(geometry,query),channel,channel_value) : {Nat.is_lt(_,count) == True{} : Bool}
    DivisionOrder.remainder_small(quotient,count,lane_positive),
    %Equal.sym(Nat,Representation.value_outer(geometry,query),at,outer_value) :
      {Representation.logical_index(geometry,_,Representation.value_channel(geometry,query)) == query : Nat}
    %Equal.sym(Nat,Representation.value_channel(geometry,query),channel,channel_value) : {Representation.logical_index(geometry,at,_) == query : Nat}
    %Equal.sym(Nat,Nat.div(at,after),block,at_quotient) : {Nat.add(Nat.mul(Nat.add(Nat.mul(_,count),channel),after),Nat.mod(at,after)) == query : Nat}
    %Equal.sym(Nat,Nat.mod(at,after),remainder,at_remainder) : {Nat.add(Nat.mul(Nat.add(Nat.mul(block,count),channel),after),_) == query : Nat}
    %DivisionOrder.quotient_exact(quotient,count,lane_positive) : {Nat.add(Nat.mul(_,after),remainder) == query : Nat}
    Equal.sym(Nat,query,Nat.add(Nat.mul(quotient,after),remainder),DivisionOrder.quotient_exact(query,after,after_positive))}

# The group of a channel below the lanes lies below the groups.
def channel_group(+limits: Machine.Limits,+lanes: U32,+channel: Nat,+bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(channel,U32.to_nat(lanes)) == True{} : Bool}) -> {Nat.is_lt(Nat.div(channel,8n),U32.to_nat(Grid.groups(lanes))) == True{} : Bool}:
  +count = U32.to_nat(lanes)
  +group = Nat.div(channel,8n)
  %Equal.sym(Nat,U32.to_nat(Grid.groups(lanes)),Ceiling.ceiling(count,8n),Encoded.eight_blocks_exact(limits,lanes,bound)) : {Nat.is_lt(group,_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(group,Ceiling.ceiling(count,8n)),Nat.is_lt(Nat.mul(group,8n),count),Ceiling.upper_comparison(count,8n,group,{==})) : {_ == True{} : Bool}
  Order.le_lt_trans(Nat.mul(group,8n),channel,count,DivisionOrder.product_below(channel,8n,{==}),inside)

# A channel is its lane past eight lanes per group.
def channel_split(+channel: Nat) -> {Nat.add(Nat.mod(channel,8n),Nat.mul(Nat.div(channel,8n),8n)) == channel : Nat}:
  %Order.commutative(Nat.mul(Nat.div(channel,8n),8n),Nat.mod(channel,8n)) : {_ == channel : Nat}
  Equal.sym(Nat,channel,Nat.add(Nat.mul(Nat.div(channel,8n),8n),Nat.mod(channel,8n)),DivisionOrder.quotient_exact(channel,8n,{==}))

# ---- Cells written in order keep the originals and write the query ----

def cells_of(+position: Grid.Position,+outer: U32,+after: U32,+lanes: U32) -> Cells.Cells:
  Cells.Cells{outer,after,lanes,Grid.groups(lanes),position}

# The lanes the cell at a cell word writes.
def cell_written(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+word: U32) -> List<&2,Lane>:
  written_lanes(geometry,word,Representation.cell(~Logical,~original,logical,geometry,Representation.cell_outer(geometry,U32.to_nat(word)),
    Representation.cell_group(geometry,U32.to_nat(word))))

# The cell and the lane that write a logical index.
def target_cell(+geometry: Cells.Cells,+query: Nat) -> Nat:
  Representation.cell_index(geometry,Representation.value_outer(geometry,query),Nat.div(Representation.value_channel(geometry,query),8n))

def target_lane(+geometry: Cells.Cells,+query: Nat) -> Nat:
  Nat.mod(Representation.value_channel(geometry,query),8n)

# Whether the cell that writes a query has been written: the original is
# there once it has.
def Seen(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+content: Buffer.Content<F32>,+query: Nat,reached: Bool) -> Data:
  match reached:
    case True{}: Good(~Logical,~original,logical,content,query)
    case False{}: Unit

# The cell of a query writes it from its target lane.
def target_hits(~Logical: Data,~original: Logical -> U32 -> F32,+position: Grid.Position,+logical: Logical,+limits: Machine.Limits,+outer: U32,+after: U32,+lanes: U32,
  +word: U32,+query: Nat,
  +placed: Placed(~Logical,~original,logical,cell_written(~Logical,~original,logical,cells_of(position,outer,after,lanes),word),cells_of(position,outer,after,lanes),
    Representation.cell_outer(cells_of(position,outer,after,lanes),U32.to_nat(word)),Representation.cell_group(cells_of(position,outer,after,lanes),U32.to_nat(word)),0n),
  +bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+decoded: Decoded<cells_of(position,outer,after,lanes),query>,
  +same: {target_cell(cells_of(position,outer,after,lanes),query) == U32.to_nat(word) : Nat}) ->
  HitsAt(target_lane(cells_of(position,outer,after,lanes),query),cell_written(~Logical,~original,logical,cells_of(position,outer,after,lanes),word),query):
  Decoded{+at_inside,+channel_inside,+exact} = decoded
  +groups = Grid.groups(lanes)
  +geometry = cells_of(position,outer,after,lanes)
  +at = Representation.value_outer(geometry,query)
  +channel = Representation.value_channel(geometry,query)
  +group = Nat.div(channel,8n)
  +lane = Nat.mod(channel,8n)
  +number = U32.to_nat(word)
  +target = target_cell(geometry,query)
  +written = cell_written(~Logical,~original,logical,geometry,word)
  +group_in = channel_group(limits,lanes,channel,bound,channel_inside)
  +at_exact = Representation.placed_outer(position,U32.to_nat(outer),U32.to_nat(groups),at,group,at_inside,group_in)
  +group_exact = Representation.placed_group(position,U32.to_nat(outer),U32.to_nat(groups),at,group,at_inside,group_in)
  +split = channel_split(channel)
  placed_hits(~Logical,~original,lane,written,logical,geometry,at,group,0n,lane,query,
    %group_exact : Placed(~Logical,~original,logical,written,geometry,at,_,0n)
    %at_exact : Placed(~Logical,~original,logical,written,geometry,_,Representation.cell_group(geometry,target),0n)
    %Equal.sym(Nat,target,number,same) : Placed(~Logical,~original,logical,written,geometry,Representation.cell_outer(geometry,_),Representation.cell_group(geometry,_),0n)
    placed,
    DivisionOrder.remainder_small(channel,8n,{==}),Addition.zero_right(lane),
    %Equal.sym(Nat,Nat.add(lane,Nat.mul(group,8n)),channel,split) : {Nat.is_lt(_,Representation.lane_count(geometry)) == True{} : Bool} channel_inside,
    %Equal.sym(Nat,Nat.add(lane,Nat.mul(group,8n)),channel,split) : {Representation.logical_index(geometry,at,_) == query : Nat} exact)

# Lanes that write the original keep it once the query has been written, and
# write it when they are the cell of the query.
def seen_cases(~Logical: Data,~original: Logical -> U32 -> F32,after: Bool,before: Bool,+lane: Nat,lanes: List<&2,Lane>,+logical: Logical,+depth: Nat,+capacity: U32,+size: Nat,
  +content: Buffer.Content<F32>,+query: Nat,+target: Nat,+number: Nat,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+kept: Kept<depth,capacity,size,content>,
  +rights: AllRight(~Logical,~original,logical,capacity,lanes),+query_inside: {U32.is_lt(U32.from_nat(query),capacity) == True{} : Bool},
  +after_exact: {Nat.is_lt(target,1n+number) == after : Bool},+before_exact: {Nat.is_lt(target,number) == before : Bool},
  +seen: Seen(~Logical,~original,logical,content,query,before),hits: {target == number : Nat} -> HitsAt(lane,lanes,query)) ->
  Seen(~Logical,~original,logical,lanes_model(lanes,content),query,after):
  match after:
    case True{}:
      match before:
        case True{}: lanes_good(~Logical,~original,lanes,logical,depth,capacity,size,content,query,limit,kept,rights,query_inside,seen)
        case False{}:
          +equal = Order.antisymmetric(target,number,Division.lt_succ_le(target,number,after_exact),Order.not_less(target,number,before_exact))
          lanes_hit(~Logical,~original,lane,lanes,logical,depth,capacity,size,content,query,limit,kept,rights,query_inside,hits(equal))
    case False{}: Unit{}

# Writing the cell at a word below the cells keeps the original once the query
# has been written, and writes it at the cell of the query.
def cell_seen(~Logical: Data,~original: Logical -> U32 -> F32,+position: Grid.Position,+logical: Logical,+limits: Machine.Limits,+outer: U32,+after: U32,+lanes: U32,
  +word: U32,+capacity: U32,+depth: Nat,+size: Nat,+content: Buffer.Content<F32>,+query: Nat,+divides: TiledIndexedValues.Divides<limits,outer,lanes>,
  +after_spreads: TiledIndexedValues.Spreads<after>,+bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(word),Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))) == True{} : Bool},
  +room: Room<cells_of(position,outer,after,lanes),capacity>,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+decoded: Decoded<cells_of(position,outer,after,lanes),query>,
  +query_inside: {U32.is_lt(U32.from_nat(query),capacity) == True{} : Bool},+kept: Kept<depth,capacity,size,content>,
  +seen: Seen(~Logical,~original,logical,content,query,Nat.is_lt(target_cell(cells_of(position,outer,after,lanes),query),U32.to_nat(word)))) ->
  Seen(~Logical,~original,logical,lanes_model(cell_written(~Logical,~original,logical,cells_of(position,outer,after,lanes),word),content),query,
    Nat.is_lt(target_cell(cells_of(position,outer,after,lanes),query),1n+U32.to_nat(word))):
  TiledIndexedValues.Divides{+outer_positive,+groups_positive,+outer_bound,+groups_bound} = divides
  +groups = Grid.groups(lanes)
  +geometry = cells_of(position,outer,after,lanes)
  +number = U32.to_nat(word)
  +target = target_cell(geometry,query)
  +written = cell_written(~Logical,~original,logical,geometry,word)
  +at = Representation.cell_outer(geometry,number)
  +group = Representation.cell_group(geometry,number)
  +placed = cell_placed(~Logical,~original,position,logical,limits,outer,after,lanes,word,
    {TiledIndexedValues.Divides{outer_positive,groups_positive,outer_bound,groups_bound} : TiledIndexedValues.Divides<limits,outer,lanes>},after_spreads,bound,inside)
  +at_inside = Representation.outer_inside(position,U32.to_nat(outer),U32.to_nat(groups),number,inside,outer_positive,groups_positive)
  +rights = placed_rights(~Logical,~original,written,logical,capacity,geometry,at,group,0n,placed,room,at_inside)
  seen_cases(~Logical,~original,Nat.is_lt(target,1n+number),Nat.is_lt(target,number),target_lane(geometry,query),written,logical,depth,capacity,size,content,query,
    target,number,limit,kept,rights,query_inside,{==},{==},seen,
    same => target_hits(~Logical,~original,position,logical,limits,outer,after,lanes,word,query,placed,bound,decoded,same))

# ---- The cells of a piece ----

# Writing the cells of a piece keeps the shape of the values.
def piece_kept(fuel: Nat,+index: U32,+first: U32,+context: Cells.Cells,+cells: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,+size: Nat,+content: Buffer.Content<F32>,
  +kept: Kept<depth,capacity,size,content>) -> Kept<depth,capacity,size,piece_model(fuel,index,first,context,cells,content)>:
  match fuel:
    case 0n: kept
    case 1n+ +rest:
      +word = (first + index : U32)
      +tile = Tree.read(Tile.Tile8,cells,index)
      +next = written_model(context,word,tile,content)
      piece_kept(rest,(index + 1 : U32),first,context,cells,depth,capacity,size,next,
        %Equal.sym(Buffer.Content<F32>,next,lanes_model(written_lanes(context,word,tile),content),written_listed(context,word,tile,content)) : Kept<depth,capacity,size,_>
        lanes_kept(written_lanes(context,word,tile),depth,capacity,size,content,kept))

# The cell a piece holds at an index.
def at_tile(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+first: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,+step: Nat,
  +at: Content.At(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,geometry},first,tree,length,step)) ->
  {Tree.read(Tile.Tile8,tree,U32.from_nat(step)) ==
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},U32.from_nat(Nat.add(U32.to_nat(first),step))) : Tile.Tile8}:
  KernelGemm.Both{+within,+value} = at
  value

def piece_tile(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+first: U32,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +index: U32,+step: Nat,+word: U32,+index_exact: {index == U32.from_nat(step) : U32},+word_exact: {word == U32.from_nat(Nat.add(U32.to_nat(first),step)) : U32},
  +leaf: Content.Leaf(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,geometry},first,count,tree,length),
  +within: {Nat.is_lt(step,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,tree,index) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word) : Tile.Tile8}:
  %Equal.sym(U32,index,U32.from_nat(step),index_exact) :
    {Tree.read(Tile.Tile8,tree,_) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word) : Tile.Tile8}
  %Equal.sym(U32,word,U32.from_nat(Nat.add(U32.to_nat(first),step)),word_exact) :
    {Tree.read(Tile.Tile8,tree,U32.from_nat(step)) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},_) : Tile.Tile8}
  at_tile(~Logical,~original,logical,geometry,first,tree,length,step,
    Content.leaf_cell(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,geometry},first,count,tree,length,
      step,leaf,within))

# Writing the cells of a piece from a step on, below the cells, keeps the
# original once the query has been written, and writes it at its cell.
def piece_seen(~Logical: Data,~original: Logical -> U32 -> F32,fuel: Nat,+position: Grid.Position,+logical: Logical,+limits: Machine.Limits,+outer: U32,+after: U32,
  +lanes: U32,+index: U32,+step: Nat,+first: U32,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,+capacity: U32,+depth: Nat,+size: Nat,+content: Buffer.Content<F32>,
  +query: Nat,+divides: TiledIndexedValues.Divides<limits,outer,lanes>,+after_spreads: TiledIndexedValues.Spreads<after>,
  +bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: Room<cells_of(position,outer,after,lanes),capacity>,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+decoded: Decoded<cells_of(position,outer,after,lanes),query>,
  +query_inside: {U32.is_lt(U32.from_nat(query),capacity) == True{} : Bool},
  +leaf: Content.Leaf(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,cells_of(position,outer,after,lanes)},
    first,count,tree,length),
  +index_exact: {index == U32.from_nat(step) : U32},+span: {Nat.add(step,fuel) == U32.to_nat(count) : Nat},
  +cells_inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +kept: Kept<depth,capacity,size,content>,
  +seen: Seen(~Logical,~original,logical,content,query,Nat.is_lt(target_cell(cells_of(position,outer,after,lanes),query),Nat.add(U32.to_nat(first),step)))) ->
  Seen(~Logical,~original,logical,piece_model(fuel,index,first,cells_of(position,outer,after,lanes),tree,content),query,
    Nat.is_lt(target_cell(cells_of(position,outer,after,lanes),query),Nat.add(U32.to_nat(first),U32.to_nat(count)))):
  match fuel:
    case 0n:
      +done = Equal.trans(Nat,step,Nat.add(step,0n),U32.to_nat(count),Equal.sym(Nat,Nat.add(step,0n),step,Addition.zero_right(step)),span)
      %done : Seen(~Logical,~original,logical,content,query,Nat.is_lt(target_cell(cells_of(position,outer,after,lanes),query),Nat.add(U32.to_nat(first),_)))
      seen
    case 1n+ +rest:
      +geometry = cells_of(position,outer,after,lanes)
      +target = target_cell(geometry,query)
      +word = (first + index : U32)
      +number = Nat.add(U32.to_nat(first),step)
      +tile = Tree.read(Tile.Tile8,tree,index)
      +cells = Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))
      +end = Nat.add(U32.to_nat(first),U32.to_nat(count))
      +within = Order.below_complement(step,rest,U32.to_nat(count),span)
      +below_end = Order.offset_lt(U32.to_nat(first),step,U32.to_nat(count),within)
      +below_cells = Bounds.nat_trans(number,end,cells,below_end,cells_inside)
      +number_room = Bounds.strict_le(number,U32.to_nat(Machine.capacity(limits)),Bounds.nat_trans(number,cells,U32.to_nat(Machine.capacity(limits)),below_cells,cells_room))
      +word_exact = {%Equal.sym(U32,index,U32.from_nat(step),index_exact) : {(first + _ : U32) == U32.from_nat(number) : U32}
        TiledIndexedValues.position_word(first,step) : {word == U32.from_nat(number) : U32}}
      +number_exact = Equal.trans(Nat,U32.to_nat(word),U32.to_nat(U32.from_nat(number)),number,Equal.cong(U32,Nat,U32.to_nat,word,U32.from_nat(number),word_exact),
        Bounds.roundtrip(number,Machine.capacity(limits),Limits.capacity_fits(limits),number_room))
      +tile_exact = piece_tile(~Logical,~original,logical,geometry,first,count,tree,length,index,step,word,index_exact,word_exact,leaf,within)
      +next = written_model(geometry,word,tile,content)
      +written = cell_written(~Logical,~original,logical,geometry,word)
      +listed = Equal.trans(Buffer.Content<F32>,next,lanes_model(written_lanes(geometry,word,tile),content),lanes_model(written,content),
        written_listed(geometry,word,tile,content),
        Equal.cong(Tile.Tile8,Buffer.Content<F32>,cell => lanes_model(written_lanes(geometry,word,cell),content),tile,
          Representation.cell(~Logical,~original,logical,geometry,Representation.cell_outer(geometry,U32.to_nat(word)),Representation.cell_group(geometry,U32.to_nat(word))),
          tile_exact))
      +inside = {%Equal.sym(Nat,U32.to_nat(word),number,number_exact) : {Nat.is_lt(_,cells) == True{} : Bool} below_cells : {Nat.is_lt(U32.to_nat(word),cells) == True{} : Bool}}
      +advanced = Equal.trans(Nat,1n+U32.to_nat(word),1n+number,Nat.add(U32.to_nat(first),1n+step),Equal.cong(Nat,Nat,value => 1n+value,U32.to_nat(word),number,number_exact),
        Equal.sym(Nat,Nat.add(U32.to_nat(first),1n+step),1n+number,Addition.successor_right(U32.to_nat(first),step)))
      piece_seen(~Logical,~original,rest,position,logical,limits,outer,after,lanes,(index + 1 : U32),1n+step,first,count,tree,length,capacity,depth,size,next,query,
        divides,after_spreads,bound,room,limit,decoded,query_inside,leaf,
        %Equal.sym(U32,index,U32.from_nat(step),index_exact) : {(_ + 1 : U32) == U32.from_nat(1n+step) : U32} Bounds.increment_encoded(step),
        Equal.trans(Nat,1n+Nat.add(step,rest),Nat.add(step,1n+rest),U32.to_nat(count),Equal.sym(Nat,Nat.add(step,1n+rest),1n+Nat.add(step,rest),Addition.successor_right(step,rest)),
          span),
        cells_inside,cells_room,
        %Equal.sym(Buffer.Content<F32>,next,lanes_model(written,content),listed) : Kept<depth,capacity,size,_>
        lanes_kept(written,depth,capacity,size,content,kept),
        %Equal.sym(Buffer.Content<F32>,next,lanes_model(written,content),listed) : Seen(~Logical,~original,logical,_,query,Nat.is_lt(target,Nat.add(U32.to_nat(first),1n+step)))
        %advanced : Seen(~Logical,~original,logical,lanes_model(written,content),query,Nat.is_lt(target,_))
        cell_seen(~Logical,~original,position,logical,limits,outer,after,lanes,word,capacity,depth,size,content,query,divides,after_spreads,bound,inside,room,limit,decoded,
          query_inside,kept,
          %Equal.sym(Nat,U32.to_nat(word),number,number_exact) : Seen(~Logical,~original,logical,content,query,Nat.is_lt(target,_)) seen))

# ---- Every piece in order ----

# Writing every piece keeps the shape of the values.
def thread_kept(+model: PiecesModel.Model<Tile.Tile8>,+context: Cells.Cells,+depth: Nat,+capacity: U32,+size: Nat,+content: Buffer.Content<F32>,+kept: Kept<depth,capacity,size,content>) ->
  Kept<depth,capacity,size,PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,context,PiecesModel.buffer_parts(~Tile.Tile8,model),content)>:
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}: piece_kept(U32.to_nat(rows),0,first,context,Buffer.tree(~Tile.Tile8,buffer),depth,capacity,size,content,kept)
    case PiecesModel.Fork{+left,+right}:
      thread_kept(right,context,depth,capacity,size,
        PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,context,PiecesModel.buffer_parts(~Tile.Tile8,left),content),
        thread_kept(left,context,depth,capacity,size,content,kept))

# Writing the pieces that cover cells from a start on, below the cells, keeps
# the original once the query has been written, and writes it at its cell.
def thread_seen(~Logical: Data,~original: Logical -> U32 -> F32,+model: PiecesModel.Model<Tile.Tile8>,+position: Grid.Position,+logical: Logical,+limits: Machine.Limits,
  +outer: U32,+after: U32,+lanes: U32,+capacity: U32,+depth: Nat,+size: Nat,+content: Buffer.Content<F32>,+query: Nat,+divides: TiledIndexedValues.Divides<limits,outer,lanes>,
  +after_spreads: TiledIndexedValues.Spreads<after>,+bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: Room<cells_of(position,outer,after,lanes),capacity>,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+decoded: Decoded<cells_of(position,outer,after,lanes),query>,
  +query_inside: {U32.is_lt(U32.from_nat(query),capacity) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +start: Nat,+rows: Nat,+coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,model)),start,rows),
  +inside: {Nat.is_le(Nat.add(start,rows),Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))) == True{} : Bool},
  +values: Content.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,cells_of(position,outer,after,lanes)},
    PiecesModel.content(~Tile.Tile8,model)),
  +kept: Kept<depth,capacity,size,content>,
  +seen: Seen(~Logical,~original,logical,content,query,Nat.is_lt(target_cell(cells_of(position,outer,after,lanes),query),start))) ->
  Seen(~Logical,~original,logical,
    PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,cells_of(position,outer,after,lanes),
      PiecesModel.buffer_parts(~Tile.Tile8,model),content),query,Nat.is_lt(target_cell(cells_of(position,outer,after,lanes),query),Nat.add(start,rows))):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+count}:
      Buffer.Model{+tree,+length,certificate} = buffer
      PiecesPlan.LeafCoverage{+first_equal,+rows_equal} = coverage
      +geometry = cells_of(position,outer,after,lanes)
      +target = target_cell(geometry,query)
      +opened = Equal.trans(Nat,Nat.add(U32.to_nat(first),0n),U32.to_nat(first),start,Addition.zero_right(U32.to_nat(first)),first_equal)
      %first_equal : Seen(~Logical,~original,logical,piece_model(U32.to_nat(count),0,first,geometry,tree,content),query,Nat.is_lt(target,Nat.add(_,rows)))
      %rows_equal : Seen(~Logical,~original,logical,piece_model(U32.to_nat(count),0,first,geometry,tree,content),query,Nat.is_lt(target,Nat.add(U32.to_nat(first),_)))
      piece_seen(~Logical,~original,U32.to_nat(count),position,logical,limits,outer,after,lanes,0,0n,first,count,tree,length,capacity,depth,size,content,query,
        divides,after_spreads,bound,room,limit,decoded,query_inside,values,{==},{==},
        %Equal.sym(Nat,U32.to_nat(first),start,first_equal) : {Nat.is_le(Nat.add(_,U32.to_nat(count)),Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(count),rows,rows_equal) : {Nat.is_le(Nat.add(start,_),Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))) == True{} : Bool}
        inside,
        cells_room,kept,
        %Equal.sym(Nat,Nat.add(U32.to_nat(first),0n),start,opened) : Seen(~Logical,~original,logical,content,query,Nat.is_lt(target,_)) seen)
    case PiecesModel.Fork{+left,+right}:
      PiecesPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
      KernelGemm.Both{+left_values,+right_values} = values
      +geometry = cells_of(position,outer,after,lanes)
      +target = target_cell(geometry,query)
      +cells = Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))
      +middle = Nat.add(start,left_rows)
      +joined = Equal.trans(Nat,Nat.add(middle,right_rows),Nat.add(start,Nat.add(left_rows,right_rows)),Nat.add(start,rows),
        Addition.associative(start,left_rows,right_rows),Equal.cong(Nat,Nat,value => Nat.add(start,value),Nat.add(left_rows,right_rows),rows,Equal.sym(Nat,rows,Nat.add(left_rows,right_rows),rows_equal)))
      +right_inside = {%Equal.sym(Nat,Nat.add(middle,right_rows),Nat.add(start,rows),joined) : {Nat.is_le(_,cells) == True{} : Bool} inside :
        {Nat.is_le(Nat.add(middle,right_rows),cells) == True{} : Bool}}
      +left_inside = Order.transitive(middle,Nat.add(middle,right_rows),cells,Order.le_add(middle,right_rows),right_inside)
      +before = PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,geometry,PiecesModel.buffer_parts(~Tile.Tile8,left),content)
      %joined : Seen(~Logical,~original,logical,
        PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,geometry,PiecesModel.buffer_parts(~Tile.Tile8,right),before),query,
        Nat.is_lt(target,_))
      thread_seen(~Logical,~original,right,position,logical,limits,outer,after,lanes,capacity,depth,size,before,query,divides,after_spreads,bound,room,limit,decoded,query_inside,
        cells_room,middle,right_rows,right_coverage,right_inside,right_values,thread_kept(left,geometry,depth,capacity,size,content,kept),
        thread_seen(~Logical,~original,left,position,logical,limits,outer,after,lanes,capacity,depth,size,content,query,divides,after_spreads,bound,room,limit,decoded,query_inside,
          cells_room,start,left_rows,left_coverage,left_inside,left_values,kept,seen))

# The cell of a query lies below the cells.
def target_inside(+position: Grid.Position,+limits: Machine.Limits,+outer: U32,+after: U32,+lanes: U32,+query: Nat,
  +bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+decoded: Decoded<cells_of(position,outer,after,lanes),query>) ->
  {Nat.is_lt(target_cell(cells_of(position,outer,after,lanes),query),Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))) == True{} : Bool}:
  Decoded{+at_inside,+channel_inside,+exact} = decoded
  +geometry = cells_of(position,outer,after,lanes)
  +channel = Representation.value_channel(geometry,query)
  Representation.index_inside(position,U32.to_nat(outer),U32.to_nat(Grid.groups(lanes)),Representation.value_outer(geometry,query),Nat.div(channel,8n),at_inside,
    channel_group(limits,lanes,channel,bound,channel_inside))

# Writing every piece that covers the cells writes the original at a query
# below whole blocks.
def thread_good(~Logical: Data,~original: Logical -> U32 -> F32,+model: PiecesModel.Model<Tile.Tile8>,+position: Grid.Position,+logical: Logical,+limits: Machine.Limits,
  +outer: U32,+after: U32,+lanes: U32,+capacity: U32,+depth: Nat,+size: Nat,+content: Buffer.Content<F32>,+query: Nat,+divides: TiledIndexedValues.Divides<limits,outer,lanes>,
  +after_spreads: TiledIndexedValues.Spreads<after>,+bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: Room<cells_of(position,outer,after,lanes),capacity>,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+decoded: Decoded<cells_of(position,outer,after,lanes),query>,
  +query_inside: {U32.is_lt(U32.from_nat(query),capacity) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,model)),0n,Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))),
  +values: Content.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,cells_of(position,outer,after,lanes)},
    PiecesModel.content(~Tile.Tile8,model)),
  +kept: Kept<depth,capacity,size,content>) ->
  Good(~Logical,~original,logical,
    PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,cells_of(position,outer,after,lanes),
      PiecesModel.buffer_parts(~Tile.Tile8,model),content),query):
  +geometry = cells_of(position,outer,after,lanes)
  +target = target_cell(geometry,query)
  +cells = Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))
  %target_inside(position,limits,outer,after,lanes,query,bound,decoded) :
    Seen(~Logical,~original,logical,
      PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,geometry,PiecesModel.buffer_parts(~Tile.Tile8,model),content),query,_)
  thread_seen(~Logical,~original,model,position,logical,limits,outer,after,lanes,capacity,depth,size,content,query,divides,after_spreads,bound,room,limit,decoded,query_inside,
    cells_room,0n,cells,coverage,Order.reflexive(cells),values,kept,
    %Equal.sym(Bool,Nat.is_lt(target,0n),False{},Division.lt_zero(target)) : Seen(~Logical,~original,logical,content,query,_) Unit{})

# ---- The values the cells give back ----

# The value of contents at an index is the value its tree holds there.
def content_value(+index: U32,+content: Buffer.Content<F32>) -> {Buffer.value(~F32,index,content) == Tree.read(F32,Buffer.content_tree(~F32,content),index) : F32}:
  Buffer.Content{+tree,+length} = content
  {==}

def kept_length(+depth: Nat,+capacity: U32,+size: Nat,+content: Buffer.Content<F32>,+kept: Kept<depth,capacity,size,content>) ->
  {Buffer.content_length(~F32,content) == size : Nat}:
  Kept{+balanced,+sized,+long} = kept
  long

def kept_ready(+depth: Nat,+capacity: U32,+size: Nat,+content: Buffer.Content<F32>,+kept: Kept<depth,capacity,size,content>,+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(size,U32.to_nat(capacity)) == True{} : Bool}) -> Buffer.ContentReady<F32,content>:
  Kept{+balanced,+sized,+long} = kept
  Buffer.ContentReady{depth,balanced,limit,
    %Equal.sym(Nat,Buffer.content_length(~F32,content),size,long) : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,Buffer.content_tree(~F32,content)))) == True{} : Bool}
    %Equal.sym(U32,Tree.capacity(F32,Buffer.content_tree(~F32,content)),capacity,sized) : {Nat.is_le(size,U32.to_nat(_)) == True{} : Bool} room}

# Writing every piece that covers the cells writes the original at every
# index from a position on below whole blocks.
def all_good(~Logical: Data,~original: Logical -> U32 -> F32,remaining: Nat,+index: Nat,+model: PiecesModel.Model<Tile.Tile8>,+position: Grid.Position,+logical: Logical,
  +limits: Machine.Limits,+outer: U32,+after: U32,+lanes: U32,+capacity: U32,+depth: Nat,+size: Nat,+content: Buffer.Content<F32>,
  +divides: TiledIndexedValues.Divides<limits,outer,lanes>,+after_spreads: TiledIndexedValues.Spreads<after>,
  +bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+blocks: Nat,
  +outer_exact: {Representation.outer_count(cells_of(position,outer,after,lanes)) == Nat.mul(blocks,Representation.after_count(cells_of(position,outer,after,lanes))) : Nat},
  +total: {Nat.is_le(size,U32.to_nat(capacity)) == True{} : Bool},+fits: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +size_exact: {size == Nat.mul(Nat.mul(blocks,U32.to_nat(lanes)),Representation.after_count(cells_of(position,outer,after,lanes))) : Nat},
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,model)),0n,Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))),
  +values: Content.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,cells_of(position,outer,after,lanes)},
    PiecesModel.content(~Tile.Tile8,model)),
  +kept: Kept<depth,capacity,size,content>,+room: {Nat.is_le(Nat.add(index,remaining),size) == True{} : Bool}) ->
  Members.All(Nat,{index => Contract.At(~F32,~Logical,~original,logical,
      PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,cells_of(position,outer,after,lanes),
        PiecesModel.buffer_parts(~Tile.Tile8,model),content),index) : Nat -> Data},Ranges.range(remaining,index)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +geometry = cells_of(position,outer,after,lanes)
      +final = PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,geometry,PiecesModel.buffer_parts(~Tile.Tile8,model),content)
      +inside = Order.room_index(index,rest,size,room)
      +blocks_inside = {%size_exact : {Nat.is_lt(index,_) == True{} : Bool} inside :
        {Nat.is_lt(index,Nat.mul(Nat.mul(blocks,U32.to_nat(lanes)),Representation.after_count(geometry))) == True{} : Bool}}
      +ample = {%size_exact : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} total :
        {Nat.is_le(Nat.mul(Nat.mul(blocks,U32.to_nat(lanes)),Representation.after_count(geometry)),U32.to_nat(capacity)) == True{} : Bool}}
      KernelGemm.Both{
        %Equal.sym(F32,Buffer.value(~F32,U32.from_nat(index),final),Tree.read(F32,Buffer.content_tree(~F32,final),U32.from_nat(index)),content_value(U32.from_nat(index),final)) :
          {_ == original(logical,U32.from_nat(index)) : F32}
        thread_good(~Logical,~original,model,position,logical,limits,outer,after,lanes,capacity,depth,size,content,index,divides,after_spreads,bound,
          {Room{blocks,outer_exact,ample,fits} : Room<geometry,capacity>},limit,decode_query(geometry,blocks,index,outer_exact,blocks_inside),
          word_below(index,size,capacity,fits,inside,total),cells_room,coverage,values,kept),
        all_good(~Logical,~original,rest,1n+index,model,position,logical,limits,outer,after,lanes,capacity,depth,size,content,divides,after_spreads,bound,blocks,
          outer_exact,total,fits,size_exact,limit,cells_room,coverage,values,kept,Order.advance_room(index,rest,size,room))}

# ---- The untiled values ----

# The values around the lanes of values along an axis, as the untiled form
# writes them.
def untiled_after(+values: List<&2,U32>,+axis: Nat) -> U32:
  Tiled.product(Public.lane_after(values,axis))

def untiled_outer(+values: List<&2,U32>,+axis: Nat) -> U32:
  (Public.lane_prefix(values,axis,1) * untiled_after(values,axis) : U32)

def untiled_count(+values: List<&2,U32>,+axis: Nat,+lanes: U32) -> U32:
  (Public.lane_prefix(values,axis,1) * lanes * untiled_after(values,axis) : U32)

def untiled_geometry(+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position) -> Cells.Cells:
  cells_of(position,untiled_outer(values,axis),untiled_after(values,axis),lanes)

# The buffer the pieces of cells write their values into.
def untiled_out(+model: PiecesModel.Model<Tile.Tile8>,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position) -> Storage.Buffer<F32>:
  Pieces.fold(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Cells.Cells,~Tiled.untiled_piece,untiled_geometry(values,axis,lanes,position),PiecesModel.pack(~Tile.Tile8,model),
    Tensors.buffer(~F32,0.0,untiled_count(values,axis,lanes)))

def untiled_model(+model: PiecesModel.Model<Tile.Tile8>,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position) -> Buffer.Model<F32>:
  Buffer.remodel(~F32,untiled_out(model,values,axis,lanes,position))

# What the pieces write into a fresh buffer of the values.
def untiled_final(+model: PiecesModel.Model<Tile.Tile8>,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position) -> Buffer.Content<F32>:
  +count = untiled_count(values,axis,lanes)
  PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,untiled_geometry(values,axis,lanes,position),
    PiecesModel.buffer_parts(~Tile.Tile8,model),Buffer.Content{Tree.fresh(F32,Tensors.depth_for(count),0.0),U32.to_nat(count)})

# The untiled form of the packed cells of values is the values of the buffer
# their pieces write.
def untiled_equation(+model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +values: List<&2,U32>,+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
  +restored: {Tiled.logical_shape(shape,axis,lanes,position) == values : List<&2,U32>}) ->
  {Tiled.untiled(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}) ==
    Tensors.values(~F32,Buffer.pack(~F32,untiled_model(model,values,axis,lanes,position)),values,0.0,Tensors.Usable{},workers) : Tensors.Tensor<F32>}:
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(shape,axis,lanes,position),values,restored) :
    {Tensors.values(~F32,Tiled.values_status(Tensors.Usable{},_,axis,lanes,position,PiecesModel.pack(~Tile.Tile8,model)),_,0.0,Tensors.Usable{},workers) ==
      Tensors.values(~F32,Buffer.pack(~F32,untiled_model(model,values,axis,lanes,position)),values,0.0,Tensors.Usable{},workers) : Tensors.Tensor<F32>}
  %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(values,axis,1),Some{Tiled.Around{Public.lane_prefix(values,axis,1),Grid.lanes_at(values,axis),Public.lane_after(values,axis)}},
      Public.located(values,axis,1,inside)) :
    {Tensors.values(~F32,Tiled.values_around(_,lanes,position,PiecesModel.pack(~Tile.Tile8,model)),values,0.0,Tensors.Usable{},workers) ==
      Tensors.values(~F32,Buffer.pack(~F32,untiled_model(model,values,axis,lanes,position)),values,0.0,Tensors.Usable{},workers) : Tensors.Tensor<F32>}
  %Equal.sym(Storage.Buffer<F32>,Buffer.pack(~F32,untiled_model(model,values,axis,lanes,position)),untiled_out(model,values,axis,lanes,position),
      Buffer.remodeled(~F32,untiled_out(model,values,axis,lanes,position))) :
    {Tensors.values(~F32,untiled_out(model,values,axis,lanes,position),values,0.0,Tensors.Usable{},workers) == Tensors.values(~F32,_,values,0.0,Tensors.Usable{},workers) :
      Tensors.Tensor<F32>}
  {==}

# The contents of the untiled buffer are what the pieces write into a fresh
# buffer of the values.
def untiled_content(+model: PiecesModel.Model<Tile.Tile8>,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position) ->
  {Buffer.content(~F32,untiled_model(model,values,axis,lanes,position)) == untiled_final(model,values,axis,lanes,position) : Buffer.Content<F32>}:
  +count = untiled_count(values,axis,lanes)
  +geometry = untiled_geometry(values,axis,lanes,position)
  +initial = {Buffer.Content{Tree.fresh(F32,Tensors.depth_for(count),0.0),U32.to_nat(count)} : Buffer.Content<F32>}
  +kept = Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,count))
  +start = Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,kept)),Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count)),initial,
    Equal.cong(Storage.Buffer<F32>,Buffer.Content<F32>,buffer => Buffer.contents(~F32,buffer),Buffer.pack(~F32,kept),Tensors.buffer(~F32,0.0,count),
      Buffer.remodeled(~F32,Tensors.buffer(~F32,0.0,count))),
    AllocationLaws.contents(~F32,Tensors.depth_for(count),0.0,U32.to_nat(count)))
  Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,untiled_model(model,values,axis,lanes,position)),Buffer.contents(~F32,untiled_out(model,values,axis,lanes,position)),
    untiled_final(model,values,axis,lanes,position),Buffer.remodel_content(~F32,untiled_out(model,values,axis,lanes,position)),
    %Buffer.remodeled(~F32,Tensors.buffer(~F32,0.0,count)) :
      {Buffer.contents(~F32,Pieces.fold(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Cells.Cells,~Tiled.untiled_piece,geometry,PiecesModel.pack(~Tile.Tile8,model),_)) ==
        untiled_final(model,values,axis,lanes,position) : Buffer.Content<F32>}
    Equal.trans(Buffer.Content<F32>,
      Buffer.contents(~F32,Pieces.fold(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Cells.Cells,~Tiled.untiled_piece,geometry,PiecesModel.pack(~Tile.Tile8,model),
        Buffer.pack(~F32,kept))),
      PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,geometry,PiecesModel.buffer_parts(~Tile.Tile8,model),
        Buffer.contents(~F32,Buffer.pack(~F32,kept))),
      untiled_final(model,values,axis,lanes,position),untiled_fold(geometry,model,kept),
      Equal.cong(Buffer.Content<F32>,Buffer.Content<F32>,
        state => PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,geometry,PiecesModel.buffer_parts(~Tile.Tile8,model),state),
        Buffer.contents(~F32,Buffer.pack(~F32,kept)),initial,start)))

# Every value of nonempty values is the original, with the facts of their
# cells.
def entry_all(~Logical: Data,~original: Logical -> U32 -> F32,+model: PiecesModel.Model<Tile.Tile8>,+position: Grid.Position,+logical: Logical,+limits: Machine.Limits,
  +outer: U32,+after: U32,+lanes: U32,+capacity: U32,+depth: Nat,+size: Nat,+content: Buffer.Content<F32>,+total_cells: Nat,
  +entry: Entry.Entry<limits,outer,after,lanes,total_cells>,+bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+blocks: Nat,
  +outer_exact: {Representation.outer_count(cells_of(position,outer,after,lanes)) == Nat.mul(blocks,Representation.after_count(cells_of(position,outer,after,lanes))) : Nat},
  +total: {Nat.is_le(size,U32.to_nat(capacity)) == True{} : Bool},+fits: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +size_exact: {size == Nat.mul(Nat.mul(blocks,U32.to_nat(lanes)),Representation.after_count(cells_of(position,outer,after,lanes))) : Nat},
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,model)),0n,total_cells),
  +values: Content.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,cells_of(position,outer,after,lanes)},
    PiecesModel.content(~Tile.Tile8,model)),
  +kept: Kept<depth,capacity,size,content>) ->
  Members.All(Nat,{index => Contract.At(~F32,~Logical,~original,logical,
      PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,cells_of(position,outer,after,lanes),
        PiecesModel.buffer_parts(~Tile.Tile8,model),content),index) : Nat -> Data},Ranges.range(size,0n)):
  Entry.Entry{+cells,+room,+divides,+after_spreads,+group_facts} = entry
  +cell_count = Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))
  all_good(~Logical,~original,size,0n,model,position,logical,limits,outer,after,lanes,capacity,depth,size,content,divides,after_spreads,bound,blocks,outer_exact,total,fits,
    size_exact,limit,
    %Equal.sym(Nat,cell_count,total_cells,cells) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room,
    %Equal.sym(Nat,cell_count,total_cells,cells) : PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,model)),0n,_) coverage,
    values,kept,Order.reflexive(size))

# Every value of nonempty counted values laid out in cells is the original.
def nonempty_all(~Logical: Data,~original: Logical -> U32 -> F32,+model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+count: U32,+accepted: {Tensors.count_of(values) == Some{count} : Maybe<&2,U32>},
  +exact: {ShapeLaws.elements(values) == U32.to_nat(count) : Nat},+bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+shaped: {Tiled.cell_shape(values,axis,position) == shape : List<&2,U32>},
  +restored: {Tiled.logical_shape(shape,axis,lanes,position) == values : List<&2,U32>},+pinned: {Grid.lanes_at(values,axis) == lanes : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+split: Public.Split<limits,values,axis,lanes>,+capacity: U32,+depth: Nat,+size: Nat,+content: Buffer.Content<F32>,
  +total: {Nat.is_le(size,U32.to_nat(capacity)) == True{} : Bool},+fits: {U32.is_le(capacity,2147483648) == True{} : Bool},+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +size_exact: {size == ShapeLaws.elements(values) : Nat},
  +coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,model)),0n,ShapeLaws.elements(shape)),
  +cell_values: Content.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,untiled_geometry(values,axis,lanes,position)},PiecesModel.content(~Tile.Tile8,model)),
  +kept: Kept<depth,capacity,size,content>) ->
  Members.All(Nat,{index => Contract.At(~F32,~Logical,~original,logical,
      PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,untiled_geometry(values,axis,lanes,position),
        PiecesModel.buffer_parts(~Tile.Tile8,model),content),index) : Nat -> Data},Ranges.range(size,0n)):
  Public.Split{+blocks,+outer_exact,+elements_exact,+bound,+after_positive} = split
  +elements = ShapeLaws.elements(values)
  +inner = untiled_after(values,axis)
  +outer = untiled_outer(values,axis)
  +values_room = Entry.values_room(limits,values,count,exact,bounded)
  +blocks_exact = Equal.trans(Nat,size,elements,Nat.mul(Nat.mul(blocks,U32.to_nat(lanes)),Representation.least_one(U32.to_nat(inner))),size_exact,elements_exact)
  +positive_cells = Rooms.cells_positive(limits,values,axis,position,nonempty,values_room)
  +laid = {Public.Laid{inside,shaped,restored,Global.Counted{count,accepted},pinned} : Public.Laid<shape,axis,lanes,position,values>}
  +shape_positive = {%shaped : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} positive_cells : {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}}
  +after_spreads = TiledIndexedValues.spread_of(ShapeLaws.elements(shape),inner,0n,shape_positive,TiledIndexedValues.after_facts(values,axis,lanes,position,shape,laid))
  +entry = Entry.entry_cases(Nat.is_lt(axis,List.length(&2,U32,values)),limits,values,axis,position,count,{==},exact,bounded,positive_cells,after_spreads)
  entry_all(~Logical,~original,model,position,logical,limits,outer,inner,lanes,capacity,depth,size,content,ShapeLaws.elements(shape),
    %shaped : Entry.Entry<limits,outer,inner,lanes,ShapeLaws.elements(_)>
    %pinned : Entry.Entry<limits,outer,inner,_,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))> entry,
    bound,blocks,outer_exact,total,fits,blocks_exact,limit,coverage,cell_values,kept)

# Every value of counted values laid out in cells is the original: none when
# they are empty.
def values_cases(~Logical: Data,~original: Logical -> U32 -> F32,nonempty: Bool,+model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,+axis: Nat,+lanes: U32,
  +position: Grid.Position,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+count: U32,+accepted: {Tensors.count_of(values) == Some{count} : Maybe<&2,U32>},
  +exact: {ShapeLaws.elements(values) == U32.to_nat(count) : Nat},+bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+shaped: {Tiled.cell_shape(values,axis,position) == shape : List<&2,U32>},
  +restored: {Tiled.logical_shape(shape,axis,lanes,position) == values : List<&2,U32>},+pinned: {Grid.lanes_at(values,axis) == lanes : U32},
  +classified: {Nat.is_lt(0n,ShapeLaws.elements(values)) == nonempty : Bool},+capacity: U32,+depth: Nat,+size: Nat,+content: Buffer.Content<F32>,
  +total: {Nat.is_le(size,U32.to_nat(capacity)) == True{} : Bool},+fits: {U32.is_le(capacity,2147483648) == True{} : Bool},+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +size_exact: {size == ShapeLaws.elements(values) : Nat},
  +coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,model)),0n,ShapeLaws.elements(shape)),
  +cell_values: Content.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,untiled_geometry(values,axis,lanes,position)},PiecesModel.content(~Tile.Tile8,model)),
  +kept: Kept<depth,capacity,size,content>) ->
  Members.All(Nat,{index => Contract.At(~F32,~Logical,~original,logical,
      PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,untiled_geometry(values,axis,lanes,position),
        PiecesModel.buffer_parts(~Tile.Tile8,model),content),index) : Nat -> Data},Ranges.range(size,0n)):
  match nonempty:
    case True{}:
      nonempty_all(~Logical,~original,model,shape,axis,lanes,position,logical,limits,values,count,accepted,exact,bounded,inside,shaped,restored,pinned,classified,
        Public.values_split(limits,values,axis,lanes,pinned,Entry.values_room(limits,values,count,exact,bounded),classified),capacity,depth,size,content,
        total,fits,limit,size_exact,coverage,cell_values,kept)
    case False{}:
      +empty = Equal.trans(Nat,size,ShapeLaws.elements(values),0n,size_exact,
        Order.zero_below(ShapeLaws.elements(values),Order.not_less(0n,ShapeLaws.elements(values),classified)))
      %Equal.sym(Nat,size,0n,empty) :
        Members.All(Nat,{index => Contract.At(~F32,~Logical,~original,logical,
            PiecesModel.shaped_thread(~Buffer.Model<Tile.Tile8>,~Buffer.Content<F32>,~Cells.Cells,~untile_step,untiled_geometry(values,axis,lanes,position),
              PiecesModel.buffer_parts(~Tile.Tile8,model),content),index) : Nat -> Data},Ranges.range(_,0n))
      Unit{}

# The values the untiled form of a tensor stores, which hold the original
# values of its shape.
type Untiled<-Logical: Data,-original: Logical -> U32 -> F32,-logical: Logical,-values: List<&2,U32>,-tensor: Tensors.Tensor<F32>> is Data:
  Untiled{model: Buffer.Model<F32>,workers: U32,
    equation: {tensor == Tensors.values(~F32,Buffer.pack(~F32,model),values,0.0,Tensors.Usable{},workers) : Tensors.Tensor<F32>},
    facts: KernelGemm.Both<Contract.WellFormed<F32,values,Buffer.content(~F32,model)>,Contract.Values(~F32,~Logical,~original,logical,Buffer.content(~F32,model))>}

def UntiledOf(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled) -> Data:
  Untiled<Logical,original,logical,values,Tiled.untiled(tiled)>

# The untiled form of the packed cells of counted values laid out in cells
# that hold their lifted values stores the original values.
def limited_untiled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,
  +fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,+limits: Machine.Limits,+count: U32,
  +accepted: {Tensors.count_of(values) == Some{count} : Maybe<&2,U32>},+exact: {ShapeLaws.elements(values) == U32.to_nat(count) : Nat},
  +bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
  +shaped: {Tiled.cell_shape(values,axis,position) == shape : List<&2,U32>},+restored: {Tiled.logical_shape(shape,axis,lanes,position) == values : List<&2,U32>},
  +pinned: {Grid.lanes_at(values,axis) == lanes : U32},
  +coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.contents(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,model))),0n,ShapeLaws.elements(shape)),
  +cell_values: Content.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,Public.geometry_of(values,axis,position)},PiecesModel.contents(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,model)))) ->
  UntiledOf(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}):
  +elements = ShapeLaws.elements(values)
  +values_room = Entry.values_room(limits,values,count,exact,bounded)
  +total_word = untiled_count(values,axis,lanes)
  +product_word = {%pinned : {((Public.lane_prefix(values,axis,1) * _ : U32) * untiled_after(values,axis) : U32) == (1 * Grid.product(values) : U32) : U32}
    Public.around_product(values,axis,1) : {total_word == (1 * Grid.product(values) : U32) : U32}}
  +size = U32.to_nat(total_word)
  +size_exact = Equal.trans(Nat,size,U32.to_nat(Grid.product(values)),elements,
    Equal.cong(U32,Nat,U32.to_nat,total_word,Grid.product(values),
      Equal.trans(U32,total_word,(1 * Grid.product(values) : U32),Grid.product(values),product_word,Public.one_times(Grid.product(values)))),
    Public.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),values,values_room))
  +within_word = {%Equal.sym(Bool,U32.is_le(total_word,Machine.capacity(limits)),Nat.is_le(size,U32.to_nat(Machine.capacity(limits))),
        Comparison.le(total_word,Machine.capacity(limits))) : {_ == True{} : Bool}
      %Equal.sym(Nat,size,elements,size_exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} values_room :
    {U32.is_le(total_word,Machine.capacity(limits)) == True{} : Bool}}
  +depth = Tensors.depth_for(total_word)
  +fresh = Tree.fresh(F32,depth,0.0)
  +capacity = Tree.capacity(F32,fresh)
  +sized = Depth.bounded(total_word,TiledValues.allocatable(limits,total_word,within_word))
  +limit = Depth.allocation_depth(total_word,sized)
  +capacity_exact = Shape.capacity(F32,depth,fresh,Shape.fresh(F32,depth,0.0))
  +total = {%Equal.sym(U32,capacity,U32.shln(1,depth),capacity_exact) : {Nat.is_le(size,U32.to_nat(_)) == True{} : Bool} Depth.allocation_room(total_word,sized) :
    {Nat.is_le(size,U32.to_nat(capacity)) == True{} : Bool}}
  +fits = {%Equal.sym(U32,capacity,U32.shln(1,depth),capacity_exact) : {U32.is_le(_,2147483648) == True{} : Bool} Shift.power_fits(depth,limit) :
    {U32.is_le(capacity,2147483648) == True{} : Bool}}
  +initial = {Buffer.Content{fresh,size} : Buffer.Content<F32>}
  +geometry = untiled_geometry(values,axis,lanes,position)
  +start = {Kept{Shape.fresh(F32,depth,0.0),{==},{==}} : Kept<depth,capacity,size,initial>}
  +kept = thread_kept(model,geometry,depth,capacity,size,initial,start)
  +final = untiled_final(model,values,axis,lanes,position)
  +out = untiled_model(model,values,axis,lanes,position)
  +content_exact = untiled_content(model,values,axis,lanes,position)
  +content_back = Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,out),final,content_exact)
  +length_exact = kept_length(depth,capacity,size,final,kept)
  +geometry_exact = Equal.trans(Cells.Cells,Public.geometry_of(values,axis,position),
    Cells.Cells{untiled_outer(values,axis),untiled_after(values,axis),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),position},geometry,
    Public.geometry_exact(values,axis,position,inside),
    Equal.cong(U32,Cells.Cells,word => Cells.Cells{untiled_outer(values,axis),untiled_after(values,axis),word,Tiled.groups(word),position},Grid.lanes_at(values,axis),lanes,pinned))
  +content_model = PiecesModel.packed_contents(~Tile.Tile8,model)
  Untiled{out,workers,untiled_equation(model,shape,fill,workers,axis,lanes,position,values,inside,restored),
    KernelGemm.Both{
      Contract.WellFormed{
        %content_back : {Buffer.content_length(~F32,_) == elements : Nat}
        Equal.trans(Nat,Buffer.content_length(~F32,final),size,elements,length_exact,size_exact),
        %content_back : Buffer.ContentReady<F32,_> kept_ready(depth,capacity,size,final,kept,limit,total)},
      %content_back : Contract.Values(~F32,~Logical,~original,logical,_)
      %Equal.sym(Nat,Buffer.content_length(~F32,final),size,length_exact) :
        Members.All(Nat,{index => Contract.At(~F32,~Logical,~original,logical,final,index) : Nat -> Data},Ranges.range(_,0n))
      values_cases(~Logical,~original,Nat.is_lt(0n,elements),model,shape,axis,lanes,position,logical,limits,values,count,accepted,exact,bounded,inside,shaped,restored,
        pinned,{==},
        capacity,depth,size,initial,total,fits,limit,size_exact,
        %content_model : PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,_),0n,ShapeLaws.elements(shape)) coverage,
        %content_model : Content.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,geometry},_)
        %geometry_exact : Content.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,_},
          PiecesModel.contents(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,model)))
        cell_values,
        start)}}

# The untiled form of the packed cells of a tensor of the contract stores the
# original values of its shape.
def packed_untiled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,
  +fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,values,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position})) ->
  UntiledOf(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}):
  KernelGemm.Both{KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+fields,+ready,+coverage},KernelGemm.Both{+cells_counted,+canonical}},+cell_values},
    Public.Laid{+inside,+shaped,+restored,Global.Counted{+count,+accepted},+pinned}} = contract
  +elements = ShapeLaws.elements(values)
  +counted_values = Count.count_of(values,count,accepted)
  limited_untiled(~Logical,~original,logical,values,model,shape,fill,workers,axis,lanes,position,Default.limits(),count,accepted,
    Count.count_exact(elements,count,counted_values),PaddedCells.count_bounded(elements,count,counted_values),inside,shaped,restored,pinned,coverage,cell_values)

# The untiled form of a tensor of the contract stores the original values of
# its shape.
def untiled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) -> UntiledOf(~Logical,~original,logical,values,tiled):
  Public.by_packed(~Logical,~original,(that => UntiledOf(~Logical,~original,logical,values,that)),logical,values,tiled,contract,
    model => shape => fill => workers => axis => lanes => position => packed =>
      packed_untiled(~Logical,~original,logical,values,model,shape,fill,workers,axis,lanes,position,packed))
