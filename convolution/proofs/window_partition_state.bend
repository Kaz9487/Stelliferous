# Every window a partition holds covers its span and holds its band, so each
# leaf stores the whole-input leaf output. Aligned rows copy bands; other rows
# keep the caller input, whose leaves pack it directly.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/traversal_partition.bend as Partition
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./input_storage.bend as Inputs
import ../../lib/proofs/kernel_partition_bounds.bend as TilePartitionBounds
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./microkernel_storage_execution.bend as Execution
import ./array_leaf_cells.bend as ArrayLeaf
import ./window_positions.bend as Positions
import ./window_covers.bend as Covers
import ./window_cell_contents.bend as CellContents
import ./window_cell_values.bend as CellValues
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ./window_leaf_readback.bend as LeafReadback
import ../scalar_kernel.bend as ScalarConv
import ../../lib/proofs/binary_radix.bend as Radix

# A partition window: its rows cover the span and hold their input band.
type State<-problem: ConvSpec.Problem,-storage_contract: Inputs.Contract<problem>,-window: Execution.InputWindow,-offset: U32,-span: U32> is Data:
  State{tree: Tree.Tree<Tile.Tile8>,first: U32,rows: U32,
    same: {window == Execution.InputWindow{tree,first,rows} : Execution.InputWindow},
    cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
    contents: CellContents.Contents(problem,CellContents.input_cells(problem,storage_contract),tree,0n,U32.to_nat(first),U32.to_nat(rows))}

def cover_fits(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Positions.Cover<problem,first,rows,offset,span>) ->
  {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool}:
  Positions.Cover{block,first_exact,+fits,rows_positive,kernel_fits,top,bottom} = cover
  fits

def cover_rows(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Positions.Cover<problem,first,rows,offset,span>) ->
  {Nat.is_lt(0n,rows) == True{} : Bool}:
  Positions.Cover{block,first_exact,fits,+rows_positive,kernel_fits,top,bottom} = cover
  rows_positive

def leaf_window(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+reference_storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset: U32,+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+offset_alignment: TilePartitionBounds.BlockAligned<offset>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +tree: Tree.Tree<Tile.Tile8>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: CellContents.Contents(problem,CellContents.input_cells(problem,storage_contract),tree,0n,U32.to_nat(first),U32.to_nat(rows))) ->
  {Execution.leaf(LeafReadback.work(problem),offset,span,Execution.InputWindow{tree,first,rows},Inputs.weight_cells(problem,reference_storage),
      Inputs.bias_cells(problem,reference_storage)) ==
    ArrayLeaf.layout_cells(problem,reference_storage,Partition.Leaf{Unit{},offset,span,Unit{}}) : Execution.CellResult()}:
  +kernel = U32.from_nat(Guards.kernel(problem))
  +channels = (Coordinates.native_inner(problem) / (kernel * kernel) : U32)
  +image_width = U32.from_nat(Guards.width(problem))
  +context = {CellValues.Context{limits,problem,storage_contract,tree,0,first,rows,channels,image_width,valid,supported,
    Coordinates.inner_channels(limits,problem,Guards.valid(problem,valid),Padded.scalar(limits,problem,Support.fast(limits,problem,supported))),SupportedCoordinateBounds.width_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported))),
    cover_fits(problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span),cover),
    cover_rows(problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span),cover),contents} : CellValues.Context}
  LeafReadback.band_leaf(limits,problem,reference_storage,valid,supported,offset,span,blocks,span_exact,span_bound,offset_alignment,range,width_blocks,width,
    CellValues.values(context),first,rows,cover,CellValues.listed(context),CellValues.room(context),tree,{==})

# A partition leaf over aligned rows stores the whole-input leaf output.
def leaf_state(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+reference_storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset: U32,+span: U32,+span_alignment: TilePartitionBounds.BlockAligned<span>,
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+offset_alignment: TilePartitionBounds.BlockAligned<offset>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +window: Execution.InputWindow,+state: State<problem,storage_contract,window,offset,span>) ->
  {Execution.leaf(LeafReadback.work(problem),offset,span,window,Inputs.weight_cells(problem,reference_storage),Inputs.bias_cells(problem,reference_storage)) ==
    ArrayLeaf.layout_cells(problem,reference_storage,Partition.Leaf{Unit{},offset,span,Unit{}}) : Execution.CellResult()}:
  TilePartitionBounds.BlockAligned{+blocks,+span_exact} = span_alignment
  State{+tree,+first,+rows,+same,+cover,+contents} = state
  %Equal.sym(Execution.InputWindow,window,Execution.InputWindow{tree,first,rows},same) :
    {Execution.leaf(LeafReadback.work(problem),offset,span,_,Inputs.weight_cells(problem,reference_storage),Inputs.bias_cells(problem,reference_storage)) ==
      ArrayLeaf.layout_cells(problem,reference_storage,Partition.Leaf{Unit{},offset,span,Unit{}}) : Execution.CellResult()}
  leaf_window(limits,problem,storage_contract,reference_storage,valid,supported,offset,span,blocks,span_exact,span_bound,offset_alignment,range,width_blocks,width,
    tree,first,rows,cover,contents)

# The two windows a split hands to the halves of a span.
type Children<-problem: ConvSpec.Problem,-storage_contract: Inputs.Contract<problem>,-pair: PartitionLaws.Split<Execution.InputWindow>,-offset: U32,-span: U32> is Data:
  Children{left: State<problem,storage_contract,PartitionLaws.first(~Execution.InputWindow,pair),offset,TilePartitionBounds.half(span)>,
    right: State<problem,storage_contract,PartitionLaws.second(~Execution.InputWindow,pair),(offset + TilePartitionBounds.half(span) : U32),(span - TilePartitionBounds.half(span) : U32)>}

def left_state(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+pair: PartitionLaws.Split<Execution.InputWindow>,+offset: U32,+span: U32,+children: Children<problem,storage_contract,pair,offset,span>) ->
  State<problem,storage_contract,PartitionLaws.first(~Execution.InputWindow,pair),offset,TilePartitionBounds.half(span)>:
  Children{+left,right} = children
  left

def right_state(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+pair: PartitionLaws.Split<Execution.InputWindow>,+offset: U32,+span: U32,+children: Children<problem,storage_contract,pair,offset,span>) ->
  State<problem,storage_contract,PartitionLaws.second(~Execution.InputWindow,pair),(offset + TilePartitionBounds.half(span) : U32),(span - TilePartitionBounds.half(span) : U32)>:
  Children{left,+right} = children
  right

# A window that covers a span covers its second half.
def kept_right(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+offset: U32,+span: U32,+tree: Tree.Tree<Tile.Tile8>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: CellContents.Contents(problem,CellContents.input_cells(problem,storage_contract),tree,0n,U32.to_nat(first),U32.to_nat(rows)),
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  State<problem,storage_contract,Execution.InputWindow{tree,first,rows},(offset + TilePartitionBounds.half(span) : U32),(span - TilePartitionBounds.half(span) : U32)>:
  +capacity = Machine.capacity(limits)
  +fits = Limits.capacity_fits(limits)
  +start = U32.to_nat(offset)
  +half = U32.to_nat(TilePartitionBounds.half(span))
  +end = Nat.add(start,U32.to_nat(span))
  State{tree,first,rows,{==},
    Positions.cover_within(problem,U32.to_nat(first),U32.to_nat(rows),start,U32.to_nat(span),U32.to_nat((offset + TilePartitionBounds.half(span) : U32)),
      U32.to_nat((span - TilePartitionBounds.half(span) : U32)),cover,Coordinates.width_positive(problem),
      %Equal.sym(Nat,U32.to_nat((offset + TilePartitionBounds.half(span) : U32)),Nat.add(start,half),TilePartitionBounds.offset_exact(offset,span,capacity,fits,capacity_room)) :
        {Nat.is_le(start,_) == True{} : Bool}
      Order.le_add(start,half),
      TilePartitionBounds.right_interval_bound(offset,span,capacity,end,fits,capacity_room,Order.reflexive(end))),
    contents}

# Without a copy both halves keep the parent window.
def kept(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+offset: U32,+span: U32,+tree: Tree.Tree<Tile.Tile8>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: CellContents.Contents(problem,CellContents.input_cells(problem,storage_contract),tree,0n,U32.to_nat(first),U32.to_nat(rows)),
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Children<problem,storage_contract,PartitionLaws.Split{Execution.InputWindow{tree,first,rows},Execution.InputWindow{tree,first,rows}},offset,span>:
  +start = U32.to_nat(offset)
  +end = Nat.add(start,U32.to_nat(span))
  Children{
    State{tree,first,rows,{==},
      Positions.cover_within(problem,U32.to_nat(first),U32.to_nat(rows),start,U32.to_nat(span),start,U32.to_nat(TilePartitionBounds.half(span)),cover,
        Coordinates.width_positive(problem),Order.reflexive(start),
        TilePartitionBounds.left_interval_bound(offset,span,Machine.capacity(limits),end,Limits.capacity_fits(limits),capacity_room,Order.reflexive(end))),
      contents},
    kept_right(limits,problem,storage_contract,offset,span,tree,first,rows,cover,contents,capacity_room)}

# A copied band holds its rows and covers the first half; the second keeps the parent.
def copied_with(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,
  +tree: Tree.Tree<Tile.Tile8>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: CellContents.Contents(problem,CellContents.input_cells(problem,storage_contract),tree,0n,U32.to_nat(first),U32.to_nat(rows)),
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups: U32,+image_width: U32,+groups_exact: {U32.to_nat(groups) == CellContents.group_count(problem) : Nat},
  +width_exact: {U32.to_nat(image_width) == Guards.width(problem) : Nat},+left_first: U32,+left_rows: U32,
  +facts: Covers.CopyFacts<problem,first,rows,left_first,left_rows,offset,TilePartitionBounds.half(span)>) ->
  Children<problem,storage_contract,PartitionLaws.Split{Execution.InputWindow{Execution.copied_tree(groups,image_width,first,left_first,left_rows,tree),left_first,left_rows},
    Execution.InputWindow{tree,first,rows}},offset,span>:
  Covers.CopyFacts{+left_cover,+after,+within} = facts
  +copied = CellContents.copied_contents(limits,problem,CellContents.input_cells(problem,storage_contract),tree,first,rows,left_first,left_rows,groups,image_width,
    valid,supported,groups_exact,width_exact,contents,after,within,
    cover_fits(problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span),cover))
  Children{
    State{Execution.copied_tree(groups,image_width,first,left_first,left_rows,tree),left_first,left_rows,{==},left_cover,copied},
    kept_right(limits,problem,storage_contract,offset,span,tree,first,rows,cover,contents,capacity_room)}

def children_copy(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,
  +tree: Tree.Tree<Tile.Tile8>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: CellContents.Contents(problem,CellContents.input_cells(problem,storage_contract),tree,0n,U32.to_nat(first),U32.to_nat(rows)),
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +groups: U32,+image_width: U32,+groups_exact: {U32.to_nat(groups) == CellContents.group_count(problem) : Nat},
  +width_exact: {U32.to_nat(image_width) == Guards.width(problem) : Nat},+band: KernelWindow.Band,
  +checked: {KernelWindow.covers(first,rows,U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),
    U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),offset,TilePartitionBounds.half(span),band) == True{} : Bool}) ->
  Children<problem,storage_contract,Execution.copied(groups,image_width,first,rows,band,tree),offset,span>:
  KernelWindow.Band{+left_first,+left_rows} = band
  copied_with(limits,problem,storage_contract,valid,supported,offset,span,tree,first,rows,cover,contents,capacity_room,groups,image_width,groups_exact,width_exact,
    left_first,left_rows,
    Covers.copy_facts(limits,problem,valid,supported,offset,TilePartitionBounds.half(span),first,rows,left_first,left_rows,width_blocks,width,
      TilePartitionBounds.left_interval_bound(offset,span,Machine.capacity(limits),Support.padded(problem),Limits.capacity_fits(limits),capacity_room,range),
      cover_fits(problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span),cover),checked))

# An eight-wide output row is a whole number of blocks of eight positions.
def blocks_per_row(problem: ConvSpec.Problem) -> Nat:
  U32.to_nat((U32.from_nat(ConvSpec.out_width(problem)) / 8 : U32))

def width_factor(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool},
  aligned: {U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)) == True{} : Bool}) ->
  {Nat.mul(blocks_per_row(problem),8n) == ConvSpec.out_width(problem) : Nat}:
  Equal.trans(Nat,Nat.mul(blocks_per_row(problem),8n),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ConvSpec.out_width(problem),
    Radix.divide_eight_exact_multiple(U32.from_nat(ConvSpec.out_width(problem)),aligned),Coordinates.width_exact(limits,problem,Guards.valid(problem,valid),Support.scalar(limits,problem,supported)))

def children_when(copy: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,
  +tree: Tree.Tree<Tile.Tile8>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: CellContents.Contents(problem,CellContents.input_cells(problem,storage_contract),tree,0n,U32.to_nat(first),U32.to_nat(rows)),
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +groups: U32,+image_width: U32,+groups_exact: {U32.to_nat(groups) == CellContents.group_count(problem) : Nat},
  +width_exact: {U32.to_nat(image_width) == Guards.width(problem) : Nat},+band: KernelWindow.Band,
  +checked: {U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)) && KernelWindow.covers(first,rows,U32.from_nat(Guards.height(problem)),
    U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),
    offset,TilePartitionBounds.half(span),band) == copy : Bool}) ->
  Children<problem,storage_contract,Execution.split_when(copy,groups,image_width,first,rows,band,tree),offset,span>:
  match copy:
    case False{}: kept(limits,problem,storage_contract,offset,span,tree,first,rows,cover,contents,capacity_room)
    case True{}:
      children_copy(limits,problem,storage_contract,valid,supported,offset,span,tree,first,rows,cover,contents,capacity_room,range,blocks_per_row(problem),
        width_factor(limits,problem,valid,Support.scalar_guard(limits,problem,supported),Boolean.left(U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)),KernelWindow.covers(first,rows,U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),offset,TilePartitionBounds.half(span),band),checked)),
        groups,image_width,groups_exact,width_exact,band,
        Boolean.right(U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)),KernelWindow.covers(first,rows,U32.from_nat(Guards.height(problem)),
          U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),
          offset,TilePartitionBounds.half(span),band),checked))

# Both windows of a split cover their halves and hold their bands.
def children(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +window: Execution.InputWindow,+state: State<problem,storage_contract,window,offset,span>) ->
  Children<problem,storage_contract,Execution.split(LeafReadback.work(problem),offset,TilePartitionBounds.half(span),window),offset,span>:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  State{+tree,+first,+rows,+same,+cover,+contents} = state
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +half = TilePartitionBounds.half(span)
  +out_width = ConvSpec.out_width(problem)
  +channels = (Coordinates.native_inner(problem) / (U32.from_nat(kernel) * U32.from_nat(kernel)) : U32)
  +band = KernelWindow.band(first,rows,U32.from_nat(height),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),U32.from_nat(out_width),offset,half)
  %Equal.sym(Execution.InputWindow,window,Execution.InputWindow{tree,first,rows},same) :
    Children<problem,storage_contract,Execution.split(LeafReadback.work(problem),offset,TilePartitionBounds.half(span),_),offset,span>
  children_when(U32.is_zero((U32.from_nat(out_width) % 8 : U32)) && KernelWindow.covers(first,rows,U32.from_nat(height),U32.from_nat(kernel),U32.from_nat(stride),
      U32.from_nat(padding),U32.from_nat(out_width),offset,half,band),
    limits,problem,storage_contract,valid,supported,offset,span,tree,first,rows,cover,contents,capacity_room,range,
    Gemm.blocks(channels),U32.from_nat(image_width),
    CellValues.group_count_exact(limits,problem,channels,supported,Coordinates.inner_channels(limits,problem,Guards.valid(problem,valid),support)),
    SupportedCoordinateBounds.width_exact(limits,problem,support),band,{==})
