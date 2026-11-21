# Every window a partition holds covers its span and holds its band, so each
# leaf stores the whole-input leaf output. Aligned rows copy bands; other rows
# keep the caller input, whose leaves pack it directly.
import Base
import ../../lib/kernel_gemm.bend as Gemm
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
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
import ./tile_layout.bend as TileLayout
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./microkernel_storage_execution.bend as Execution
import ./array_leaf_cells.bend as ArrayLeaf
import ./window_positions.bend as Positions
import ./window_contents.bend as Contents
import ./window_cell_contents.bend as CellContents
import ./window_leaf_readback.bend as LeafReadback
import ./gather_copy_coordinates.bend as GatherCopyCoordinates

import ./window_partition_state.bend as WindowState

# Over aligned rows every node's window covers its span and holds its band.
def partition_aligned(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+reference_storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  depth: Nat,+offset: U32,+span: U32,+offset_alignment: TilePartitionBounds.BlockAligned<offset>,+span_alignment: TilePartitionBounds.BlockAligned<span>,
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +window: Execution.InputWindow,+state: WindowState.State<problem,storage_contract,window,offset,span>) ->
  {Execution.partition(depth,LeafReadback.work(problem),offset,span,window,Inputs.weight_cells(problem,reference_storage),Inputs.bias_cells(problem,reference_storage)) ==
    ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(depth,offset,span)) : Execution.CellResult()}:
  match depth:
    case 0n:
      WindowState.leaf_state(limits,problem,storage_contract,reference_storage,valid,supported,offset,span,span_alignment,span_bound,offset_alignment,range,width_blocks,width,window,state)
    case 1n+ +rest:
      +capacity = Machine.capacity(limits)
      +fits = Limits.capacity_fits(limits)
      +padded = Support.padded(problem)
      +work = LeafReadback.work(problem)
      +reordered = Inputs.weight_cells(problem,reference_storage)
      +bias = Inputs.bias_cells(problem,reference_storage)
      +half = TilePartitionBounds.half(span)
      +pair = Execution.split(work,offset,half,window)
      +kids = WindowState.children(limits,problem,storage_contract,valid,supported,offset,span,capacity_room,range,window,state)
      +bounded = Order.transitive(U32.to_nat(span),padded,U32.to_nat(capacity),span_bound,
        Padded.padded_bound(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported)))
      +left_bound = Order.transitive(U32.to_nat(half),U32.to_nat(span),padded,TilePartitionBounds.half_le(span,capacity,fits,bounded),span_bound)
      +right_bound = Order.transitive(U32.to_nat((span - half : U32)),U32.to_nat(span),padded,TilePartitionBounds.remainder_le(span,capacity,fits,bounded),span_bound)
      +left_window = PartitionLaws.first(~Execution.InputWindow,pair)
      +right_window = PartitionLaws.second(~Execution.InputWindow,pair)
      %Equal.sym(Execution.CellResult(),Execution.partition(1n+rest,work,offset,span,window,reordered,bias),
        Partition.Fork{Execution.partition(rest,work,offset,half,left_window,reordered,bias),
          Execution.partition(rest,work,(offset + half : U32),(span - half : U32),right_window,reordered,bias)},
        Execution.partition_step(rest,work,offset,span,window,reordered,bias)) :
        {_ == ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(1n+rest,offset,span)) : Execution.CellResult()}
      %Equal.sym(Execution.CellResult(),Execution.partition(rest,work,offset,half,left_window,reordered,bias),
        ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(rest,offset,half)),
        partition_aligned(limits,problem,storage_contract,reference_storage,valid,supported,width_blocks,width,rest,offset,half,offset_alignment,
          TilePartitionBounds.half_alignment(span,capacity,fits,bounded),TilePartitionBounds.left_room(offset,span,capacity,fits,capacity_room),left_bound,
          TilePartitionBounds.left_interval_bound(offset,span,capacity,padded,fits,capacity_room,range),left_window,WindowState.left_state(problem,storage_contract,pair,offset,span,kids))) :
        {Partition.Fork{_,Execution.partition(rest,work,(offset + half : U32),(span - half : U32),right_window,reordered,bias)} ==
          ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(1n+rest,offset,span)) : Execution.CellResult()}
      %Equal.sym(Execution.CellResult(),Execution.partition(rest,work,(offset + half : U32),(span - half : U32),right_window,reordered,bias),
        ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(rest,(offset + half : U32),(span - half : U32))),
        partition_aligned(limits,problem,storage_contract,reference_storage,valid,supported,width_blocks,width,rest,(offset + half : U32),(span - half : U32),
          TilePartitionBounds.right_alignment(offset,span,capacity,fits,capacity_room,offset_alignment),TilePartitionBounds.remainder_alignment(span,capacity,fits,bounded,span_alignment),
          TilePartitionBounds.right_room(offset,span,capacity,fits,capacity_room),right_bound,
          TilePartitionBounds.right_interval_bound(offset,span,capacity,padded,fits,capacity_room,range),right_window,WindowState.right_state(problem,storage_contract,pair,offset,span,kids))) :
        {Partition.Fork{ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(rest,offset,half)),_} ==
          ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(1n+rest,offset,span)) : Execution.CellResult()}
      {==}

# The caller input as the first window of the partition.
def root_window(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>) -> Execution.InputWindow:
  Execution.InputWindow{CellContents.input_cells(problem,storage_contract),0,U32.from_nat(Guards.height(problem))}

# Unaligned rows never copy: both halves keep the caller input.
def root_split(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+offset: U32,+half: U32,
  +unaligned: {U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)) == False{} : Bool}) ->
  {Execution.split(LeafReadback.work(problem),offset,half,root_window(problem,storage_contract)) ==
    PartitionLaws.Split{root_window(problem,storage_contract),root_window(problem,storage_contract)} : PartitionLaws.Split<Execution.InputWindow>}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +tree = CellContents.input_cells(problem,storage_contract)
  +rows = U32.from_nat(height)
  +out_width = U32.from_nat(ConvSpec.out_width(problem))
  +band = KernelWindow.band(0,rows,rows,U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),out_width,offset,half)
  %Equal.sym(Bool,U32.is_zero((out_width % 8 : U32)),False{},unaligned) :
    {Execution.split_when(_ && KernelWindow.covers(0,rows,rows,U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),out_width,offset,half,band),
      Gemm.blocks((Coordinates.native_inner(problem) / (U32.from_nat(kernel) * U32.from_nat(kernel)) : U32)),U32.from_nat(image_width),0,rows,band,tree) ==
      PartitionLaws.Split{Execution.InputWindow{tree,0,rows},Execution.InputWindow{tree,0,rows}} : PartitionLaws.Split<Execution.InputWindow>}
  {==}

def partition_unaligned(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+reference_storage: Inputs.Contract<problem>,+values: {LeafReadback.input_values(problem,CellContents.input_cells(problem,storage_contract)) == Inputs.input_tree(problem,reference_storage) : Tree.Tree<F32>},+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},depth: Nat,+offset: U32,+span: U32,
  +offset_alignment: TilePartitionBounds.BlockAligned<offset>,+span_alignment: TilePartitionBounds.BlockAligned<span>,
  +capacity_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +unaligned: {U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)) == False{} : Bool}) ->
  {Execution.partition(depth,LeafReadback.work(problem),offset,span,root_window(problem,storage_contract),Inputs.weight_cells(problem,reference_storage),
      Inputs.bias_cells(problem,reference_storage)) ==
    ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(depth,offset,span)) : Execution.CellResult()}:
  match depth:
    case 0n: LeafReadback.root_leaf(limits,problem,reference_storage,offset,span,CellContents.input_cells(problem,storage_contract),valid,supported,values,span_alignment,range)
    case 1n+ +rest:
      +work = LeafReadback.work(problem)
      +reordered = Inputs.weight_cells(problem,reference_storage)
      +bias = Inputs.bias_cells(problem,reference_storage)
      +half = TilePartitionBounds.half(span)
      +root = root_window(problem,storage_contract)
      +capacity = Machine.capacity(limits)
      +fits = Limits.capacity_fits(limits)
      +padded = Support.padded(problem)
      +bounded = Order.transitive(U32.to_nat(span),padded,U32.to_nat(capacity),span_bound,
        Padded.padded_bound(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported)))
      +left_bound = Order.transitive(U32.to_nat(half),U32.to_nat(span),padded,TilePartitionBounds.half_le(span,capacity,fits,bounded),span_bound)
      +right_bound = Order.transitive(U32.to_nat((span - half : U32)),U32.to_nat(span),padded,TilePartitionBounds.remainder_le(span,capacity,fits,bounded),span_bound)
      %Equal.sym(Execution.CellResult(),Execution.partition(1n+rest,work,offset,span,root,reordered,bias),
        Partition.Fork{Execution.partition(rest,work,offset,half,PartitionLaws.first(~Execution.InputWindow,Execution.split(work,offset,half,root)),reordered,bias),
          Execution.partition(rest,work,(offset + half : U32),(span - half : U32),PartitionLaws.second(~Execution.InputWindow,Execution.split(work,offset,half,root)),reordered,bias)},
        Execution.partition_step(rest,work,offset,span,root,reordered,bias)) :
        {_ == ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(1n+rest,offset,span)) : Execution.CellResult()}
      %Equal.sym(PartitionLaws.Split<Execution.InputWindow>,Execution.split(work,offset,half,root),PartitionLaws.Split{root,root},
        root_split(problem,storage_contract,offset,half,unaligned)) :
        {Partition.Fork{Execution.partition(rest,work,offset,half,PartitionLaws.first(~Execution.InputWindow,_),reordered,bias),
          Execution.partition(rest,work,(offset + half : U32),(span - half : U32),PartitionLaws.second(~Execution.InputWindow,_),reordered,bias)} ==
          ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(1n+rest,offset,span)) : Execution.CellResult()}
      %Equal.sym(Execution.CellResult(),Execution.partition(rest,work,offset,half,root,reordered,bias),
        ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(rest,offset,half)),partition_unaligned(limits,problem,storage_contract,reference_storage,values,valid,supported,rest,offset,half,
          offset_alignment,TilePartitionBounds.half_alignment(span,capacity,fits,bounded),TilePartitionBounds.left_room(offset,span,capacity,fits,capacity_room),left_bound,
          TilePartitionBounds.left_interval_bound(offset,span,capacity,padded,fits,capacity_room,range),unaligned)) :
        {Partition.Fork{_,Execution.partition(rest,work,(offset + half : U32),(span - half : U32),root,reordered,bias)} ==
          ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(1n+rest,offset,span)) : Execution.CellResult()}
      %Equal.sym(Execution.CellResult(),Execution.partition(rest,work,(offset + half : U32),(span - half : U32),root,reordered,bias),
        ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(rest,(offset + half : U32),(span - half : U32))),
        partition_unaligned(limits,problem,storage_contract,reference_storage,values,valid,supported,rest,(offset + half : U32),(span - half : U32),
          TilePartitionBounds.right_alignment(offset,span,capacity,fits,capacity_room,offset_alignment),TilePartitionBounds.remainder_alignment(span,capacity,fits,bounded,span_alignment),
          TilePartitionBounds.right_room(offset,span,capacity,fits,capacity_room),right_bound,
          TilePartitionBounds.right_interval_bound(offset,span,capacity,padded,fits,capacity_room,range),unaligned)) :
        {Partition.Fork{ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(rest,offset,half)),_} ==
          ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(1n+rest,offset,span)) : Execution.CellResult()}
      {==}

# The caller input covers the whole padded span: its rows are the image.
def root_cover(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  Positions.Cover<problem,U32.to_nat(0),U32.to_nat(U32.from_nat(Guards.height(problem))),U32.to_nat(0),U32.to_nat(Padded.padded(problem))>:
  +height = Guards.height(problem)
  +height_exact = SupportedCoordinateBounds.height_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(height)),height,height_exact) : Positions.Cover<problem,0n,_,0n,U32.to_nat(Padded.padded(problem))>
  Positions.Cover{0n,{==},Order.reflexive(height),Contents.valid_condition(problem,valid,8n),Contents.valid_condition(problem,valid,4n),Positions.Start{{==}},Inl{{==}}}

def root_state(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  WindowState.State<problem,storage_contract,root_window(problem,storage_contract),0,Padded.padded(problem)>:
  +height = Guards.height(problem)
  +rows = U32.from_nat(height)
  +height_exact = SupportedCoordinateBounds.height_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))
  +tree = CellContents.input_cells(problem,storage_contract)
  WindowState.State{tree,0,rows,{==},root_cover(limits,problem,valid,supported),
    %Equal.sym(Nat,U32.to_nat(rows),height,height_exact) : CellContents.Contents(problem,tree,tree,0n,0n,_)
    CellContents.root(problem,tree)}

def root_when(aligned: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+reference_storage: Inputs.Contract<problem>,
  +values: {LeafReadback.input_values(problem,CellContents.input_cells(problem,storage_contract)) == Inputs.input_tree(problem,reference_storage) : Tree.Tree<F32>},
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+depth: Nat,
  +checked: {U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)) == aligned : Bool}) ->
  {Execution.partition(depth,LeafReadback.work(problem),0,Padded.padded(problem),root_window(problem,storage_contract),Inputs.weight_cells(problem,reference_storage),
      Inputs.bias_cells(problem,reference_storage)) ==
    ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(depth,0,Padded.padded(problem))) : Execution.CellResult()}:
  match aligned:
    case True{}:
      +facts = Guards.valid(problem,valid)
      +fast = Support.fast(limits,problem,supported)
      +span = Padded.padded(problem)
      +span_exact = Padded.padded_exact(limits,problem,facts,fast)
      +span_bound = {%Equal.sym(Nat,U32.to_nat(span),Support.padded(problem),span_exact) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
        Order.reflexive(Support.padded(problem)) : {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}}
      +span_alignment = {TilePartitionBounds.BlockAligned{Nat.div(Nat.add(GatherCopyCoordinates.spatial(problem),7n),8n),span_exact} : TilePartitionBounds.BlockAligned<span>}
      partition_aligned(limits,problem,storage_contract,reference_storage,valid,supported,WindowState.blocks_per_row(problem),
        WindowState.width_factor(limits,problem,valid,Support.scalar_guard(limits,problem,supported),checked),depth,0,span,TilePartitionBounds.BlockAligned{0n,{==}},
        span_alignment,Padded.root_room(limits,problem,facts,fast),span_bound,span_bound,
        root_window(problem,storage_contract),root_state(limits,problem,storage_contract,valid,supported))
    case False{}:
      +facts = Guards.valid(problem,valid)
      +fast = Support.fast(limits,problem,supported)
      +span = Padded.padded(problem)
      +span_exact = Padded.padded_exact(limits,problem,facts,fast)
      +span_bound = {%Equal.sym(Nat,U32.to_nat(span),Support.padded(problem),span_exact) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
        Order.reflexive(Support.padded(problem)) : {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}}
      +span_alignment = {TilePartitionBounds.BlockAligned{Nat.div(Nat.add(GatherCopyCoordinates.spatial(problem),7n),8n),span_exact} : TilePartitionBounds.BlockAligned<span>}
      partition_unaligned(limits,problem,storage_contract,reference_storage,values,valid,supported,depth,0,span,TilePartitionBounds.BlockAligned{0n,{==}},span_alignment,
        Padded.root_room(limits,problem,facts,fast),span_bound,span_bound,checked)

# The actual partition from the caller input equals the model leaves.
def partition_root(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+reference_storage: Inputs.Contract<problem>,
  +values: {LeafReadback.input_values(problem,CellContents.input_cells(problem,storage_contract)) == Inputs.input_tree(problem,reference_storage) : Tree.Tree<F32>},
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+depth: Nat) ->
  {Execution.partition(depth,LeafReadback.work(problem),0,Padded.padded(problem),root_window(problem,storage_contract),Inputs.weight_cells(problem,reference_storage),
      Inputs.bias_cells(problem,reference_storage)) ==
    ArrayLeaf.layout_cells(problem,reference_storage,TileLayout.divide(depth,0,Padded.padded(problem))) : Execution.CellResult()}:
  root_when(U32.is_zero((U32.from_nat(ConvSpec.out_width(problem)) % 8 : U32)),limits,problem,storage_contract,reference_storage,values,valid,supported,depth,{==})
