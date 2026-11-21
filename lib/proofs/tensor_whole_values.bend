# The whole tensor of a buffer: the pieces of the grid of its shape, each a
# copy of its range of the buffer, hold the values the buffer holds.
import Base
import ../storage_buffer.bend as Storage
import ../storage_copy.bend as Copy
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ../tensor_grid.bend as Grid
import ../traversal_partition.bend as Partition
import ../machine_limits.bend as Machine
import ../kernel_gemm.bend as KernelGemm
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_certificate.bend as Certificate
import ./storage_copy_refinement.bend as CopyLaws
import ./contiguous_copy.bend as Contiguous
import ./contiguous_copy_readback.bend as CopyRead
import ./write_program.bend as Writes
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_shape.bend as Shape
import ./tensor_allocation_depth.bend as Depth
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_each.bend as PiecesEach
import ./tensor_pieces_content.bend as Content
import ./tensor_global_shape_count.bend as Global
import ./tensor_contract.bend as Contract
import ./tensor_placed.bend as Placed
import ./tensor_task_runs.bend as Runs
import ./tensor_move_readback.bend as Held
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_grid_pieces.bend as GridPieces
import ./row_major_index.bend as RowMajor
import ./tensor_shape_refinement.bend as ShapeLaws
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./machine_limit_bounds.bend as Limits

# ---- A piece copied from the buffer ----

# The buffer a piece of count cells is copied into.
def target(~Element: Data,+fill: Element,+count: U32) -> Buffer.Model<Element>:
  Buffer.remodel(~Element,Tensors.buffer(~Element,fill,count))

# The cells of a piece: count cells of the values from first, copied into its
# buffer.
def piece(~Element: Data,fill: Element,values: Buffer.Content<Element>,+first: U32,+count: U32) -> Buffer.Content<Element>:
  +model = target(~Element,fill,count)
  Buffer.Content{Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),Buffer.content_tree(~Element,values),0,first,0),Buffer.tree(~Element,model)),
    Buffer.length(~Element,model)}

def model_tree(~Element: Data,+model: Buffer.Model<Element>) -> {Buffer.content_tree(~Element,Buffer.content(~Element,model)) == Buffer.tree(~Element,model) : Tree.Tree<Element>}:
  Buffer.Model{+tree,+length,+certificate} = model
  {==}

def model_length(~Element: Data,+model: Buffer.Model<Element>) -> {Buffer.content_length(~Element,Buffer.content(~Element,model)) == Buffer.length(~Element,model) : Nat}:
  Buffer.Model{+tree,+length,+certificate} = model
  {==}

# What a copy leaves: the contents of the values and of the piece.
def owned(~Element: Data,owners: CopyLaws.Owners<Element>,+first: U32,+count: U32) -> Buffer.Content<Element> & PiecesModel.Content<Element>:
  CopyLaws.Owners{source,source_length,output,output_length} = owners
  (Buffer.Content{source,source_length},PiecesModel.PieceContent{Buffer.Content{output,output_length},first,count})

def copied_seen(~Element: Data,+first: U32,+count: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {PiecesEach.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Tensors.cut_copied(~Element,first,count,owners)) ==
    owned(~Element,CopyLaws.observe(~Element,owners),first,count) : Buffer.Content<Element> & PiecesModel.Content<Element>}:
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate}) = owners
  {==}

# A cut leaf keeps the values and copies its piece of them.
def cut_seen(~Element: Data,+fill: Element,+first: U32,+count: U32,+tree: Tree.Tree<Element>,+length: Nat,+certificate: Certificate.Certificate<Element,Tree.pack(Element,tree)>,
  +model: Buffer.Model<Element>) ->
  {PiecesEach.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),
      Tensors.cut_copied(~Element,first,count,Copy.range(~Element,count,first,0,Storage.Buffer{Tree.pack(Element,tree),length,certificate},Buffer.pack(~Element,model)))) ==
    (Buffer.Content{tree,length},PiecesModel.PieceContent{Buffer.Content{Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),tree,0,first,0),Buffer.tree(~Element,model)),
      Buffer.length(~Element,model)},first,count}) : Buffer.Content<Element> & PiecesModel.Content<Element>}:
  Buffer.Model{+output,+output_length,+output_certificate} = model
  Equal.trans(Buffer.Content<Element> & PiecesModel.Content<Element>,
    PiecesEach.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),
      Tensors.cut_copied(~Element,first,count,Copy.range(~Element,count,first,0,Storage.Buffer{Tree.pack(Element,tree),length,certificate},
        Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate}))),
    owned(~Element,CopyLaws.observe(~Element,Copy.range(~Element,count,first,0,Storage.Buffer{Tree.pack(Element,tree),length,certificate},
      Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate})),first,count),
    (Buffer.Content{tree,length},PiecesModel.PieceContent{Buffer.Content{Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),tree,0,first,0),output),output_length},first,count}),
    copied_seen(~Element,first,count,Copy.range(~Element,count,first,0,Storage.Buffer{Tree.pack(Element,tree),length,certificate},
      Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate})),
    Equal.cong(CopyLaws.Owners<Element>,Buffer.Content<Element> & PiecesModel.Content<Element>,owners => owned(~Element,owners,first,count),
      CopyLaws.observe(~Element,Copy.range(~Element,count,first,0,Storage.Buffer{Tree.pack(Element,tree),length,certificate},
        Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate})),
      CopyLaws.model(~Element,count,first,0,CopyLaws.Owners{tree,length,output,output_length}),
      CopyLaws.packed(~Element,CopyLaws.Call{count,first,0,length,output_length},[],tree,output,certificate,output_certificate)))

def cut_rule(~Element: Data,+fill: Element,+first: U32,+count: U32,+kept: Buffer.Model<Element>) ->
  {PiecesEach.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Tensors.cut_piece(~Element,fill,first,count,Buffer.pack(~Element,kept))) ==
    (Buffer.contents(~Element,Buffer.pack(~Element,kept)),
      PiecesEach.row_part(~Element,~Element,~Buffer.Content<Element>,~piece(~Element),fill,Buffer.contents(~Element,Buffer.pack(~Element,kept)),first,count)) :
    Buffer.Content<Element> & PiecesModel.Content<Element>}:
  Buffer.Model{+tree,+length,+certificate} = kept
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,tree)),tree,StorageLaws.reflect_packed(Element,tree)) :
    {PiecesEach.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),
        Tensors.cut_piece(~Element,fill,first,count,Storage.Buffer{Tree.pack(Element,tree),length,certificate})) ==
      (Buffer.Content{_,length},PiecesEach.row_part(~Element,~Element,~Buffer.Content<Element>,~piece(~Element),fill,Buffer.Content{_,length},first,count)) :
      Buffer.Content<Element> & PiecesModel.Content<Element>}
  %Buffer.remodeled(~Element,Tensors.buffer(~Element,fill,count)) :
    {PiecesEach.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),
        Tensors.cut_copied(~Element,first,count,Copy.range(~Element,count,first,0,Storage.Buffer{Tree.pack(Element,tree),length,certificate},_))) ==
      (Buffer.Content{tree,length},PiecesEach.row_part(~Element,~Element,~Buffer.Content<Element>,~piece(~Element),fill,Buffer.Content{tree,length},first,count)) :
      Buffer.Content<Element> & PiecesModel.Content<Element>}
  cut_seen(~Element,fill,first,count,tree,length,certificate,target(~Element,fill,count))

# The pieces cut from a buffer keep it and hold the copy of every range.
def cut_built(~Element: Data,+fill: Element,+layout: Pieces.Ranges,+source: Buffer.Model<Element>) ->
  {PiecesEach.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),
      Pieces.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Element,~Tensors.cut_piece(~Element),layout,fill,Buffer.pack(~Element,source))) ==
    (Buffer.contents(~Element,Buffer.pack(~Element,source)),PiecesModel.build(~Element,~piece(~Element,fill,Buffer.contents(~Element,Buffer.pack(~Element,source))),layout)) :
    Buffer.Content<Element> & PiecesModel.Content<Element>}:
  PiecesEach.built(~Element,~Storage.Buffer<Element>,~Element,~Buffer.Model<Element>,~Buffer.pack(~Element),~PiecesEach.buffer_held(~Element),~Buffer.Content<Element>,
    ~Buffer.contents(~Element),~Tensors.cut_piece(~Element),~piece(~Element),~cut_rule(~Element),layout,fill,source)

def built_contents(~Element: Data,made: Storage.Buffer<Element> & Pieces.Pieces(~Element)) ->
  {PiecesModel.contents(~Element,Pair.snd(Storage.Buffer<Element>,Pieces.Pieces(~Element),made)) ==
    Pair.snd(Buffer.Content<Element>,PiecesModel.Content<Element>,PiecesEach.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),made)) :
    PiecesModel.Content<Element>}:
  (state,pieces) = made
  {==}

def cut_layout(~Element: Data,+fill: Element,+values: Buffer.Content<Element>,layout: Pieces.Ranges) ->
  {PiecesModel.content_layout(~Element,PiecesModel.build(~Element,~piece(~Element,fill,values),layout)) == layout : Pieces.Ranges}:
  match layout:
    case Pieces.Range{first,count}: {==}
    case Pieces.Split{+left,+right}:
      %Equal.sym(Pieces.Ranges,PiecesModel.content_layout(~Element,PiecesModel.build(~Element,~piece(~Element,fill,values),left)),left,cut_layout(~Element,fill,values,left)) :
        {Pieces.Split{_,PiecesModel.content_layout(~Element,PiecesModel.build(~Element,~piece(~Element,fill,values),right))} == Pieces.Split{left,right} : Pieces.Ranges}
      %Equal.sym(Pieces.Ranges,PiecesModel.content_layout(~Element,PiecesModel.build(~Element,~piece(~Element,fill,values),right)),right,cut_layout(~Element,fill,values,right)) :
        {Pieces.Split{left,_} == Pieces.Split{left,right} : Pieces.Ranges}
      {==}

# ---- The values of the pieces ----

# What the cells of a piece copied from a buffer of a reader's values read.
type Copied<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  Copied{logical: Logical,tree: Tree.Tree<Element>,length: Nat,first: U32,count: U32,fill: Element,
    values: Contract.Values(~Element,~Logical,~original,logical,Buffer.Content{tree,length}),
    inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),length) == True{} : Bool},sized: Depth.Sized<count>}

def copied_count(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,copied: Copied<Element,Logical,original>) -> Nat:
  Copied{logical,tree,length,first,count,fill,values,inside,sized} = copied
  U32.to_nat(count)

def CopiedAt(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+copied: Copied<Element,Logical,original>,+index: Nat) -> Data:
  Copied{+logical,+tree,+length,+first,+count,+fill,+values,+inside,+sized} = copied
  Content.At(~Element,~Logical,~original,logical,first,
    Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),tree,0,first,0),Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count),index)

def copied_at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+copied: Copied<Element,Logical,original>,+index: Nat,
  +inside: {Nat.is_lt(index,copied_count(~Element,~Logical,~original,copied)) == True{} : Bool}) -> CopiedAt(~Element,~Logical,~original,copied,index):
  Copied{+logical,+tree,+length,+first,+count,+fill,+values,+within,Depth.Sized{+depth_ok,+room}} = copied
  +depth = Tensors.depth_for(count)
  +fresh = Tree.fresh(Element,depth,fill)
  +capacity = Tree.capacity(Element,fresh)
  +capacity_ok = Buffer.capacity_bound(~Element,Buffer.fresh(~Element,depth,fill,U32.to_nat(count)),depth_ok)
  +fits = {%Equal.sym(U32,capacity,U32.shln(1,depth),Shape.capacity(Element,depth,fresh,Shape.fresh(Element,depth,fill))) :
      {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    room : {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool}}
  +position = Nat.add(U32.to_nat(first),index)
  +read = CopyRead.fresh_read(~Element,U32.to_nat(count),index,tree,first,depth,fill,depth_ok,capacity_ok,fits,inside)
  +word = Equal.trans(U32,(first + U32.from_nat(index) : U32),U32.from_nat(Nat.add(U32.to_nat(first),U32.to_nat(U32.from_nat(index)))),U32.from_nat(position),
    Arithmetic.native_add(first,U32.from_nat(index)),
    Equal.cong(Nat,U32,number => U32.from_nat(Nat.add(U32.to_nat(first),number)),U32.to_nat(U32.from_nat(index)),index,
      FullRange.roundtrip(index,count,Bounds.strict_le(index,U32.to_nat(count),inside))))
  +below = Bounds.nat_trans(position,Nat.add(U32.to_nat(first),U32.to_nat(count)),length,Order.offset_lt(U32.to_nat(first),index,U32.to_nat(count),inside),within)
  KernelGemm.Both{inside,
    Equal.trans(Element,Tree.read(Element,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),tree,0,first,0),fresh),U32.from_nat(index)),
      Tree.read(Element,tree,(first + U32.from_nat(index) : U32)),original(logical,U32.from_nat(position)),read,
      Equal.trans(Element,Tree.read(Element,tree,(first + U32.from_nat(index) : U32)),Tree.read(Element,tree,U32.from_nat(position)),original(logical,U32.from_nat(position)),
        Equal.cong(U32,Element,word_index => Tree.read(Element,tree,word_index),(first + U32.from_nat(index) : U32),U32.from_nat(position),word),
        Contract.read(~Element,~Logical,~original,logical,Buffer.Content{tree,length},position,values,below)))}

# A copied piece holds the values of the buffer from its first cell, as many
# cells as its rows, ready.
def copied_holds(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+fill: Element,+values: Buffer.Content<Element>,
  +source: Contract.Values(~Element,~Logical,~original,logical,values),+first: U32,+count: U32,
  +within: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Buffer.content_length(~Element,values)) == True{} : Bool},
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Placed.PieceHolds(~Element,~Logical,~original,logical,first,count,piece(~Element,fill,values,first,count)):
  Buffer.Content{+tree,+length} = values
  +depth = Tensors.depth_for(count)
  +fresh = Tree.fresh(Element,depth,fill)
  +rows = U32.to_nat(count)
  +sized = Depth.bounded(count,allocatable)
  +model = target(~Element,fill,count)
  +writes = Contiguous.writes(~Element,rows,tree,0,first,0)
  +same = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,model),Buffer.contents(~Element,Tensors.buffer(~Element,fill,count)),Buffer.Content{fresh,rows},
    Buffer.remodel_content(~Element,Tensors.buffer(~Element,fill,count)),AllocationLaws.contents(~Element,depth,fill,rows))
  +tree_same = Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,model),Buffer.content_tree(~Element,Buffer.content(~Element,model)),fresh,
    Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,model)),Buffer.tree(~Element,model),model_tree(~Element,model)),
    Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,content => Buffer.content_tree(~Element,content),Buffer.content(~Element,model),Buffer.Content{fresh,rows},same))
  +length_same = Equal.trans(Nat,Buffer.length(~Element,model),Buffer.content_length(~Element,Buffer.content(~Element,model)),rows,
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,model)),Buffer.length(~Element,model),model_length(~Element,model)),
    Equal.cong(Buffer.Content<Element>,Nat,content => Buffer.content_length(~Element,content),Buffer.content(~Element,model),Buffer.Content{fresh,rows},same))
  +room = {%Equal.sym(U32,Tree.capacity(Element,fresh),U32.shln(1,depth),Shape.capacity(Element,depth,fresh,Shape.fresh(Element,depth,fill))) :
      {Nat.is_le(rows,U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,sized) : {Nat.is_le(rows,U32.to_nat(Tree.capacity(Element,fresh))) == True{} : Bool}}
  %Equal.sym(Tree.Tree<Element>,Buffer.tree(~Element,model),fresh,tree_same) :
    Placed.PieceHolds(~Element,~Logical,~original,logical,first,count,Buffer.Content{Writes.pure(Element,writes,_),Buffer.length(~Element,model)})
  %Equal.sym(Nat,Buffer.length(~Element,model),rows,length_same) :
    Placed.PieceHolds(~Element,~Logical,~original,logical,first,count,Buffer.Content{Writes.pure(Element,writes,fresh),_})
  KernelGemm.Both{
    Members.range_all(~Copied<Element,Logical,original>,~CopiedAt(~Element,~Logical,~original),~copied_count(~Element,~Logical,~original),~copied_at(~Element,~Logical,~original),
      rows,0n,Copied{logical,tree,length,first,count,fill,source,within,sized},Order.reflexive(rows)),
    KernelGemm.Both{{==},Writes.content_ready(~Element,writes,fresh,rows,Buffer.ContentReady{depth,Shape.fresh(Element,depth,fill),Depth.allocation_depth(count,sized),room})}}

# Every piece of a run of a layout inside the buffer, cut from it, holds its
# values.
def copied_all(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+fill: Element,+values: Buffer.Content<Element>,
  +source: Contract.Values(~Element,~Logical,~original,logical,values),layout: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound),
  +bounded: {Nat.is_le(bound,Buffer.content_length(~Element,values)) == True{} : Bool},+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),PiecesModel.build(~Element,~piece(~Element,fill,values),layout)):
  match layout:
    case Pieces.Range{+first,+count}:
      +rows = U32.to_nat(count)
      +span = Nat.add(U32.to_nat(first),rows)
      +capacity = Machine.capacity(limits)
      +inside = Runs.fits_inside(unit,U32.to_nat(first),span,bound,run)
      +count_room = Order.transitive(rows,bound,U32.to_nat(capacity),Order.transitive(rows,span,bound,Order.right_le(U32.to_nat(first),rows),inside),room)
      +allocatable = Order.word_le(count,capacity,1073741824,
        %Equal.sym(Bool,U32.is_le(count,capacity),Nat.is_le(rows,U32.to_nat(capacity)),Comparison.le(count,capacity)) : {_ == True{} : Bool} count_room,
        Limits.capacity_allocatable(limits))
      copied_holds(~Element,~Logical,~original,logical,fill,values,source,first,count,Order.transitive(span,bound,Buffer.content_length(~Element,values),inside,bounded),allocatable)
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      KernelGemm.Both{copied_all(~Element,~Logical,~original,logical,limits,fill,values,source,left,unit,bound,left_run,bounded,room),
        copied_all(~Element,~Logical,~original,logical,limits,fill,values,source,right,unit,bound,right_run,bounded,room)}

# ---- The contract ----

# The pieces cut from a buffer of the values of a reader in the grid of a
# counted shape are a tensor of the contract of the reader.
def cut_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +source: Buffer.Model<Element>,+counted: Global.Counted<shape>,
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +facts: Contract.Facts(~Element,~Logical,~original,logical,shape,Buffer.contents(~Element,Buffer.pack(~Element,source)))) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Pair.snd(Storage.Buffer<Element>,Pieces.Pieces(~Element),
    Pieces.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Element,~Tensors.cut_piece(~Element),Pieces.ranges_of_grid(Tensors.grid_of(shape)),fill,Buffer.pack(~Element,source))),
    shape,fill,Tensors.Usable{},workers}):
  KernelGemm.Both{Contract.WellFormed{+length_exact,+ready},+values} = facts
  +layout = Pieces.ranges_of_grid(Tensors.grid_of(shape))
  +total = ShapeLaws.elements(shape)
  +contents = Buffer.contents(~Element,Buffer.pack(~Element,source))
  +divides = {GridRanges.Multiple{GridRanges.last_extent(shape),Equal.sym(Nat,Nat.mul(GridRanges.last_extent(shape),1n),GridRanges.last_extent(shape),
      Order.times_one(GridRanges.last_extent(shape)))} : GridRanges.Multiple<GridRanges.last_extent(shape),1n>}
  +grid = GridRanges.grid_layout(limits,shape,Grid.piece_target(),room,nonempty,1n,divides)
  +built = PiecesModel.build(~Element,~piece(~Element,fill,contents),layout)
  +contents_exact = Equal.trans(PiecesModel.Content<Element>,
    PiecesModel.contents(~Element,Pair.snd(Storage.Buffer<Element>,Pieces.Pieces(~Element),
      Pieces.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Element,~Tensors.cut_piece(~Element),layout,fill,Buffer.pack(~Element,source)))),
    Pair.snd(Buffer.Content<Element>,PiecesModel.Content<Element>,PiecesEach.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),
      Pieces.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Element,~Tensors.cut_piece(~Element),layout,fill,Buffer.pack(~Element,source)))),
    built,
    built_contents(~Element,Pieces.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Element,~Tensors.cut_piece(~Element),layout,fill,Buffer.pack(~Element,source))),
    Equal.cong(Buffer.Content<Element> & PiecesModel.Content<Element>,PiecesModel.Content<Element>,pair => Pair.snd(Buffer.Content<Element>,PiecesModel.Content<Element>,pair),
      PiecesEach.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),
        Pieces.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Element,~Tensors.cut_piece(~Element),layout,fill,Buffer.pack(~Element,source))),
      (contents,built),cut_built(~Element,fill,layout,source)))
  +bounded = Held.nat_le_right_moved(total,Buffer.content_length(~Element,contents),total,length_exact,Order.reflexive(total))
  GridRanges.content_layout_contract(~Element,~Logical,~original,logical,shape,fill,workers,
    Pair.snd(Storage.Buffer<Element>,Pieces.Pieces(~Element),
      Pieces.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Element,~Tensors.cut_piece(~Element),layout,fill,Buffer.pack(~Element,source))),
    1n,counted,grid,built,contents_exact,cut_layout(~Element,fill,contents,layout),
    copied_all(~Element,~Logical,~original,logical,limits,fill,contents,values,layout,1n,total,GridRanges.layout_run(shape,Grid.piece_target(),1n,grid),bounded,room))

# ---- A grid of one piece ----

def shape_outer(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool}:
  GridPieces.Shape{+outer_positive,+unit_positive,+block_positive,+block_below,+per_exact,+room} = shape
  outer_positive

# A grid of at most one piece lays its cells in one range.
def single_layout(+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+single: {U32.is_le((outer * per : U32),1) == True{} : Bool}) ->
  {Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}) == Pieces.Range{0,count} : Pieces.Ranges}:
  +outer_positive = shape_outer(limits,outer,extent,unit_cells,block,per,shape)
  +per_positive = GridPieces.per_positive(limits,outer,extent,unit_cells,block,per,shape)
  +pieces_word = GridPieces.pieces_word(limits,outer,extent,unit_cells,block,per,shape)
  +at_most = {%pieces_word : {Nat.is_le(_,1n) == True{} : Bool}
    {%Comparison.le((outer * per : U32),1) : {_ == True{} : Bool} single : {Nat.is_le(U32.to_nat((outer * per : U32)),1n) == True{} : Bool}} :
    {Nat.is_le(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),1n) == True{} : Bool}}
  +outer_one = Order.antisymmetric(U32.to_nat(outer),1n,Order.transitive(U32.to_nat(outer),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),1n,RowMajor.product_covers(U32.to_nat(outer),U32.to_nat(per),per_positive),at_most),Order.strict_follow(0n,U32.to_nat(outer),outer_positive))
  +per_one = Order.antisymmetric(U32.to_nat(per),1n,
    Order.transitive(U32.to_nat(per),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),1n,Held.nat_le_right_moved(U32.to_nat(per),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.mul(U32.to_nat(per),U32.to_nat(outer)),RowMajor.multiply_commutative(U32.to_nat(outer),U32.to_nat(per)),RowMajor.product_covers(U32.to_nat(per),U32.to_nat(outer),outer_positive)),at_most),
    Order.strict_follow(0n,U32.to_nat(per),per_positive))
  +pieces_one = Order.antisymmetric(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),1n,at_most,Order.strict_follow(0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Order.product_positive(U32.to_nat(outer),U32.to_nat(per),outer_positive,per_positive)))
  +outer_word = Arithmetic.to_nat_injective(outer,1,outer_one)
  +per_word = Arithmetic.to_nat_injective(per,1,per_one)
  +pieces_last = Arithmetic.to_nat_injective((outer * per : U32),1,Equal.trans(Nat,U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),1n,pieces_word,pieces_one))
  +start_zero = GridPieces.start_place(limits,count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32),0,0n,0n,per_positive,
    GridPieces.per_room(limits,outer,extent,unit_cells,block,per,shape),GridPieces.shape_unit(limits,outer,extent,unit_cells,block,per,shape),{==},per_positive,
    Order.zero_le(U32.to_nat(Machine.capacity(limits))))
  +span = GridRanges.piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,0n,Order.product_positive(U32.to_nat(outer),U32.to_nat(per),outer_positive,per_positive))
  +end_word = {%pieces_last : {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_)) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}
    GridRanges.end_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape) : {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},1)) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}}
  +cells_word = {%start_zero : {Nat.add(_,U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0,1))) == U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},1)) : Nat}
    span : {Nat.add(0n,U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0,1))) == U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},1)) : Nat}}
  +first_same = Arithmetic.to_nat_injective(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0),0,start_zero)
  +count_same = Arithmetic.to_nat_injective(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0,1),count,
    Equal.trans(Nat,U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0,1)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},1)),U32.to_nat(count),cells_word,
      Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},1)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(count),end_word,Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),cells))))
  +range_same = Equal.trans(Pieces.Ranges,Pieces.Range{Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0),Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0,1)},Pieces.Range{0,Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0,1)},Pieces.Range{0,count},
    Equal.cong(U32,Pieces.Ranges,first => Pieces.Range{first,Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0,1)},Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0),0,first_same),
    Equal.cong(U32,Pieces.Ranges,rows => Pieces.Range{0,rows},Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0,1),count,count_same))
  %Equal.sym(U32,outer,1,outer_word) :
    {Pieces.outer_ranges(32n,U32.is_le(U32.max((_ / 1 : U32),1),1),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.max(per,1) * 1 : U32),0,U32.max((_ / 1 : U32),1)) == Pieces.Range{0,count} : Pieces.Ranges}
  %Equal.sym(U32,per,1,per_word) :
    {Pieces.inner_ranges(32n,U32.is_le((U32.max(_,1) * 1 : U32),1),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(0 * (U32.max(_,1) * 1 : U32) : U32),(U32.max(_,1) * 1 : U32)) == Pieces.Range{0,count} : Pieces.Ranges}
  range_same

def grid_single(+limits: Machine.Limits,+grid: Grid.Grid,+facts: GridPieces.GridFacts<limits,grid>,+single: {U32.is_le(Grid.pieces_of(grid),1) == True{} : Bool}) ->
  {Pieces.ranges_of_grid(grid) == Pieces.Range{0,Grid.count_of(grid)} : Pieces.Ranges}:
  GridPieces.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  %Equal.sym(Grid.Grid,grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : {Pieces.ranges_of_grid(_) == Pieces.Range{0,Grid.count_of(_)} : Pieces.Ranges}
  single_layout(limits,count,prefix,outer,extent,unit_cells,block,per,shape,cells,
    {%exact : {U32.is_le(Grid.pieces_of(_),1) == True{} : Bool} single : {U32.is_le((outer * per : U32),1) == True{} : Bool}})

# What the cells of a buffer kept as the one piece read.
type Kept<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  Kept{logical: Logical,tree: Tree.Tree<Element>,length: Nat,rows: Nat,exact: {rows == length : Nat},
    values: Contract.Values(~Element,~Logical,~original,logical,Buffer.Content{tree,length})}

def kept_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,kept: Kept<Element,Logical,original>) -> Nat:
  Kept{logical,tree,length,rows,exact,values} = kept
  rows

def KeptAt(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+kept: Kept<Element,Logical,original>,+index: Nat) -> Data:
  Kept{+logical,+tree,+length,+rows,+exact,+values} = kept
  Content.At(~Element,~Logical,~original,logical,0,tree,length,index)

def kept_at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+kept: Kept<Element,Logical,original>,+index: Nat,+inside: {Nat.is_lt(index,kept_rows(~Element,~Logical,~original,kept)) == True{} : Bool}) -> KeptAt(~Element,~Logical,~original,kept,index):
  Kept{+logical,+tree,+length,+rows,+exact,+values} = kept
  +below = {%exact : {Nat.is_lt(index,_) == True{} : Bool} inside : {Nat.is_lt(index,length) == True{} : Bool}}
  KernelGemm.Both{below,Contract.read(~Element,~Logical,~original,logical,Buffer.Content{tree,length},index,values,below)}

# A buffer of the reader's values kept whole holds them from cell 0.
def kept_holds(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+values: Buffer.Content<Element>,+source: Contract.Values(~Element,~Logical,~original,logical,values),+count: U32,
  +exact: {U32.to_nat(count) == Buffer.content_length(~Element,values) : Nat},+ready: Buffer.ContentReady<Element,values>) ->
  Placed.PieceHolds(~Element,~Logical,~original,logical,0,count,values):
  Buffer.Content{+tree,+length} = values
  KernelGemm.Both{
    Members.range_all(~Kept<Element,Logical,original>,~KeptAt(~Element,~Logical,~original),~kept_rows(~Element,~Logical,~original),~kept_at(~Element,~Logical,~original),U32.to_nat(count),0n,
      Kept{logical,tree,length,U32.to_nat(count),exact,source},Order.reflexive(U32.to_nat(count))),
    KernelGemm.Both{Equal.sym(Nat,U32.to_nat(count),length,exact),ready}}

# A grid of one piece keeps the buffer of the reader's values as its piece.
def single_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+fill: Element,+workers: U32,+source: Buffer.Model<Element>,+counted: Global.Counted<shape>,
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +facts: Contract.Facts(~Element,~Logical,~original,logical,shape,Buffer.contents(~Element,Buffer.pack(~Element,source))),
  +single: {U32.is_le(Grid.pieces_of(Tensors.grid_of(shape)),1) == True{} : Bool}) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Partition.Leaf{Buffer.pack(~Element,source),0,Grid.count_of(Tensors.grid_of(shape)),Unit{}},shape,fill,Tensors.Usable{},workers}):
  KernelGemm.Both{Contract.WellFormed{+length_exact,+ready},+values} = facts
  +divides = {GridRanges.Multiple{GridRanges.last_extent(shape),Equal.sym(Nat,Nat.mul(GridRanges.last_extent(shape),1n),GridRanges.last_extent(shape),
      Order.times_one(GridRanges.last_extent(shape)))} : GridRanges.Multiple<GridRanges.last_extent(shape),1n>}
  +grid = GridRanges.grid_layout(limits,shape,Grid.piece_target(),room,nonempty,1n,divides)
  +one = grid_single(limits,Tensors.grid_of(shape),GridRanges.grid_facts(limits,shape,Grid.piece_target(),room,nonempty),single)
  +count_exact = Equal.trans(Nat,U32.to_nat(Grid.count_of(Tensors.grid_of(shape))),ShapeLaws.elements(shape),Buffer.content_length(~Element,Buffer.contents(~Element,Buffer.pack(~Element,source))),GridRanges.grid_count(limits,shape,Grid.piece_target(),room,nonempty),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Buffer.pack(~Element,source))),ShapeLaws.elements(shape),length_exact))
  GridRanges.content_layout_contract(~Element,~Logical,~original,logical,shape,fill,workers,Partition.Leaf{Buffer.pack(~Element,source),0,Grid.count_of(Tensors.grid_of(shape)),Unit{}},1n,counted,grid,
    PiecesModel.PieceContent{Buffer.contents(~Element,Buffer.pack(~Element,source)),0,Grid.count_of(Tensors.grid_of(shape))},{==},Equal.sym(Pieces.Ranges,Pieces.ranges_of_grid(Tensors.grid_of(shape)),Pieces.Range{0,Grid.count_of(Tensors.grid_of(shape))},one),
    kept_holds(~Element,~Logical,~original,logical,Buffer.contents(~Element,Buffer.pack(~Element,source)),values,Grid.count_of(Tensors.grid_of(shape)),count_exact,ready))

def cut_when_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,single: Bool,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+fill: Element,+workers: U32,+source: Buffer.Model<Element>,+counted: Global.Counted<shape>,
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +facts: Contract.Facts(~Element,~Logical,~original,logical,shape,Buffer.contents(~Element,Buffer.pack(~Element,source))),
  +exact: {U32.is_le(Grid.pieces_of(Tensors.grid_of(shape)),1) == single : Bool}) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Tensors.cut_when(~Element,single,Tensors.grid_of(shape),fill,Buffer.pack(~Element,source)),shape,fill,Tensors.Usable{},workers}):
  match single:
    case True{}: single_contract(~Element,~Logical,~original,logical,limits,shape,fill,workers,source,counted,room,nonempty,facts,exact)
    case False{}: cut_contract(~Element,~Logical,~original,logical,limits,shape,fill,workers,source,counted,room,nonempty,facts)

# A buffer of the values of a reader for a counted shape inside the machine,
# whole, is a tensor of the contract of the reader.
def whole_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+fill: Element,+workers: U32,+source: Buffer.Model<Element>,+counted: Global.Counted<shape>,
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +facts: Contract.Facts(~Element,~Logical,~original,logical,shape,Buffer.contents(~Element,Buffer.pack(~Element,source)))) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.whole(~Element,Buffer.pack(~Element,source),shape,fill,Tensors.Usable{},workers)):
  cut_when_contract(~Element,~Logical,~original,U32.is_le(Grid.pieces_of(Tensors.grid_of(shape)),1),logical,limits,shape,fill,workers,source,counted,room,nonempty,facts,{==})
