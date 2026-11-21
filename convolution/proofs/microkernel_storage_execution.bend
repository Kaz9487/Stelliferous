# The complete pipelined Array execution refines a storage model. Its leaf
# obligations are discharged with packing, ordered GEMM and allocation proofs.
import Base
import ../../lib/kernel_array_pipeline.bend as ArrayPipeline
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_weights.bend as Weights
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_gather.bend as Gather
import ../../lib/kernel_tile.bend as Tile
import ../../lib/storage_allocation.bend as Allocation
import ../../lib/tensor.bend as Tensors
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_allocation_refinement.bend as AllocationLaws
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/kernel_gemm_readback.bend as GemmReadback
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/kernel_spatial_packing_refinement.bend as LibPacking
import ../../lib/proofs/kernel_gather_blocks.bend as Blocks
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/traversal_array.bend as Traversal
import ../../lib/tensor_grid.bend as Grid
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/tensor_pieces.bend as Pieces
import ../../lib/proofs/tensor_pieces_walk.bend as Walk
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/tensor_tiled_cells.bend as TiledCells
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/traversal_indexed_values.bend as IndexedValues
import ../../lib/proofs/tensor_access_refinement.bend as AccessLaws
import ../../lib/proofs/traversal_array_refinement.bend as ArrayLaws

def allocated(+rows: U32,+span: U32) -> Tree.Tree<F32>:
  Tree.fresh(F32,Tensors.depth_for((rows * span : U32)),0.0)

def allocated_matches(+rows: U32,+span: U32) ->
  {ArrayPipeline.allocated(rows,span) == Tree.pack(F32,allocated(rows,span)) : Array<F32>}:
  AllocationLaws.fresh(~F32,Tensors.depth_for((rows * span : U32)),0.0)

# The output cells of a leaf of rows channels over span positions.
def allocated_tiles(+rows: U32,+span: U32) -> Tree.Tree<Tile.Tile8>:
  Tree.fresh(Tile.Tile8,Tensors.depth_for(Pipeline.output_tiles(rows,span)),Tile.filled(0.0))

def allocated_tiles_matches(+rows: U32,+span: U32) ->
  {ArrayPipeline.allocated_tiles(rows,span) == Tree.pack(Tile.Tile8,allocated_tiles(rows,span)) : Array<Tile.Tile8>}:
  AllocationLaws.fresh(~Tile.Tile8,Tensors.depth_for(Pipeline.output_tiles(rows,span)),Tile.filled(0.0))

# A fresh tree of count elements has room for them.
def allocated_room(~Element: Data,+count: U32,+fill: Element,+size: Nat,+exact: {U32.to_nat(count) == size : Nat},+ready: Depth.Sized<count>) ->
  {Nat.is_le(size,U32.to_nat(Tree.capacity(Element,Tree.fresh(Element,Tensors.depth_for(count),fill)))) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(Element,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.shln(1,Tensors.depth_for(count)),
    Shape.capacity(Element,Tensors.depth_for(count),Tree.fresh(Element,Tensors.depth_for(count),fill),Shape.fresh(Element,Tensors.depth_for(count),fill))) :
    {Nat.is_le(size,U32.to_nat(_)) == True{} : Bool}
  %exact : {Nat.is_le(_,U32.to_nat(U32.shln(1,Tensors.depth_for(count)))) == True{} : Bool}
  Depth.allocation_room(count,ready)

# The leaf product: span positions in rows over the channels in lanes, with
# bias.
def leaf_matrix(+rows: U32,+span: U32,+inner: U32) -> Gemm.TileMatrix:
  Gemm.TileMatrix{rows,span,inner,0,span}

def computed(+rows: U32,+span: U32,+inner: U32,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,+tiles: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  GemmReadback.cell_tree(leaf_matrix(rows,span,inner),tiles,weights,bias,allocated_tiles(rows,span))

def computed_matches(+rows: U32,+span: U32,+inner: U32,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>) ->
  {ArrayPipeline.computed(rows,span,inner,Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias),(Tree.pack(Tile.Tile16,pairs),Tree.pack(Tile.Tile8,tiles))) ==
    Tree.pack(Tile.Tile8,computed(rows,span,inner,weights,bias,tiles)) : Array<Tile.Tile8>}:
  +matrix = leaf_matrix(rows,span,inner)
  +fresh = allocated_tiles(rows,span)
  %Equal.sym(Array<Tile.Tile8>,ArrayPipeline.allocated_tiles(rows,span),Tree.pack(Tile.Tile8,fresh),allocated_tiles_matches(rows,span)) :
    {Gemm.cell_output(Gemm.cell_run(U32.to_nat(Gemm.tile_jobs(matrix)),matrix,
      Gemm.CellOperands{Tree.pack(Tile.Tile8,bias),Gemm.Operands{Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Unit{},_}})) ==
      Tree.pack(Tile.Tile8,GemmReadback.cell_tree(matrix,tiles,weights,bias,fresh)) : Array<Tile.Tile8>}
  %Equal.sym(Gemm.CellOperands,Gemm.cell_run(U32.to_nat(Gemm.tile_jobs(matrix)),matrix,
      GemmLaws.cell_represent(GemmLaws.CellOperands{bias,GemmLaws.Operands{tiles,weights,Tree.fresh(F32,0n,0.0),Writes.Initial{fresh}}})),
    GemmLaws.cell_represent(GemmReadback.cell_run(matrix,tiles,weights,bias,fresh)),GemmReadback.cell_native(matrix,tiles,weights,bias,fresh)) :
    {Gemm.cell_output(_) == Tree.pack(Tile.Tile8,GemmReadback.cell_tree(matrix,tiles,weights,bias,fresh)) : Array<Tile.Tile8>}
  GemmReadback.cell_output_matches(GemmReadback.cell_run(matrix,tiles,weights,bias,fresh))

def CellResult() -> Data:
  Partition.Tree<&2,Tree.Tree<Tile.Tile8>,U32>

def cell_block_writes(rows: U32,+offset: U32,+positions: U32,+span: U32,input: Tree.Tree<Tile.Tile8>) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  +groups = Gemm.blocks(rows)
  Blocks.block_writes(~Tile.Tile8,1,(offset * groups : U32),(positions * groups : U32),(span * groups : U32),input)

def cell_block(+rows: U32,+offset: U32,+positions: U32,+span: U32,+input: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>) ->
  {ArrayPipeline.gathered(rows,offset,positions,span,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output)) ==
    Tree.pack(Tile.Tile8,Writes.pure(Tile.Tile8,cell_block_writes(rows,offset,positions,span,input),output)) : Array<Tile.Tile8>}:
  +groups = Gemm.blocks(rows)
  Blocks.block(~Tile.Tile8,1,(offset * groups : U32),(positions * groups : U32),(span * groups : U32),input,output)

def gather(partitions: CellResult(),output: Tree.Tree<Tile.Tile8>,+positions: U32) -> Tree.Tree<Tile.Tile8>:
  Partition.gather(~&2,~Tree.Tree<Tile.Tile8>,~PartitionLaws.write_copy(~Tile.Tile8,~cell_block_writes),partitions,output,positions)

# Rows [first, first + rows) of the image, as a proof-only storage model.
type InputWindow is Data:
  InputWindow{tree: Tree.Tree<Tile.Tile8>,first: U32,rows: U32}

def represent(window: InputWindow) -> ArrayPipeline.InputWindow:
  InputWindow{tree,first,rows} = window
  ArrayPipeline.InputWindow{Tree.pack(Tile.Tile8,tree),first,rows}

# The rows of a band copied from a window: one run of cells.
def copied_tree(+groups: U32,+width: U32,+first: U32,+left_first: U32,+left_rows: U32,+tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  +count = (left_rows * width * groups : U32)
  Writes.pure(Tile.Tile8,Contiguous.writes(~Tile.Tile8,U32.to_nat(count),tree,0,((left_first - first) * width * groups : U32),0),
    Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))

def copied(+groups: U32,+width: U32,+first: U32,+rows: U32,band: KernelWindow.Band,+tree: Tree.Tree<Tile.Tile8>) -> PartitionLaws.Split<InputWindow>:
  KernelWindow.Band{+left_first,+left_rows} = band
  PartitionLaws.Split{InputWindow{copied_tree(groups,width,first,left_first,left_rows,tree),left_first,left_rows},InputWindow{tree,first,rows}}

def split_when(copy: Bool,+groups: U32,+width: U32,+first: U32,+rows: U32,+band: KernelWindow.Band,+tree: Tree.Tree<Tile.Tile8>) -> PartitionLaws.Split<InputWindow>:
  match copy:
    case False{}: PartitionLaws.Split{InputWindow{tree,first,rows},InputWindow{tree,first,rows}}
    case True{}: copied(groups,width,first,rows,band,tree)

def split(work: Pipeline.Work,offset: U32,half: U32,window: InputWindow) -> PartitionLaws.Split<InputWindow>:
  Pipeline.Work{channels_out,+height,+width,+kernel,+stride,+padding,+output_width,positions,+inner} = work
  InputWindow{tree,+first,+rows} = window
  +offset = offset
  +half = half
  +band = KernelWindow.band(first,rows,height,kernel,stride,padding,output_width,offset,half)
  split_when(KernelWindow.copies_band(first,rows,height,kernel,stride,padding,output_width,offset,half,band),
    Gemm.blocks((inner / (kernel * kernel) : U32)),width,first,rows,band,tree)

def copied_matches(+groups: U32,+width: U32,+first: U32,+rows: U32,+band: KernelWindow.Band,+tree: Tree.Tree<Tile.Tile8>) ->
  {ArrayPipeline.copied(groups,width,first,rows,band,Tree.pack(Tile.Tile8,tree)) ==
    PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,copied(groups,width,first,rows,band,tree)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  KernelWindow.Band{+left_first,+left_rows} = band
  +count = (left_rows * width * groups : U32)
  +source = ((left_first - first) * width * groups : U32)
  +fresh = Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))
  %Equal.sym(Array<Tile.Tile8>,Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)),Tree.pack(Tile.Tile8,fresh),AllocationLaws.fresh(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))) :
    {ArrayPipeline.windows(first,rows,left_first,left_rows,Traversal.copy(~Tile.Tile8,U32.to_nat(count),source,0,Tree.pack(Tile.Tile8,tree),_)) ==
      PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,copied(groups,width,first,rows,KernelWindow.Band{left_first,left_rows},tree)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}
  %Equal.sym(Array<Tile.Tile8> & Array<Tile.Tile8>,Traversal.copy(~Tile.Tile8,U32.to_nat(count),source,0,Tree.pack(Tile.Tile8,tree),Tree.pack(Tile.Tile8,fresh)),
    (Tree.pack(Tile.Tile8,tree),Tree.pack(Tile.Tile8,copied_tree(groups,width,first,left_first,left_rows,tree))),
    Contiguous.copy(~Tile.Tile8,U32.to_nat(count),tree,fresh,source,0)) :
    {ArrayPipeline.windows(first,rows,left_first,left_rows,_) ==
      PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,copied(groups,width,first,rows,KernelWindow.Band{left_first,left_rows},tree)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}
  {==}

def split_when_matches(+copy: Bool,+groups: U32,+width: U32,+first: U32,+rows: U32,+band: KernelWindow.Band,+tree: Tree.Tree<Tile.Tile8>) ->
  {ArrayPipeline.split_when(copy,groups,width,first,rows,band,Tree.pack(Tile.Tile8,tree)) ==
    PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,split_when(copy,groups,width,first,rows,band,tree)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  match copy:
    case False{}:
      %Equal.sym(Array<Tile.Tile8> & Array<Tile.Tile8>,Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,tree)),(Tree.pack(Tile.Tile8,tree),Tree.pack(Tile.Tile8,tree)),StorageLaws.clone(Tile.Tile8,tree)) :
        {ArrayPipeline.cloned(first,rows,_) == (ArrayPipeline.InputWindow{Tree.pack(Tile.Tile8,tree),first,rows},ArrayPipeline.InputWindow{Tree.pack(Tile.Tile8,tree),first,rows}) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}
      {==}
    case True{}:
      copied_matches(groups,width,first,rows,band,tree)

def split_matches(+work: Pipeline.Work,+offset: U32,+half: U32,+window: InputWindow) ->
  {ArrayPipeline.split(work,offset,half,represent(window)) == PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,split(work,offset,half,window)) : ArrayPipeline.InputWindow & ArrayPipeline.InputWindow}:
  Pipeline.Work{channels_out,+height,+width,+kernel,+stride,+padding,+output_width,positions,+inner} = work
  InputWindow{+tree,+first,+rows} = window
  +band = KernelWindow.band(first,rows,height,kernel,stride,padding,output_width,offset,half)
  split_when_matches(KernelWindow.copies_band(first,rows,height,kernel,stride,padding,output_width,offset,half,band),
    Gemm.blocks((inner / (kernel * kernel) : U32)),width,first,rows,band,tree)

def leaf(work: Pipeline.Work,offset: U32,span: U32,window: InputWindow,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<Tile.Tile8>) -> CellResult():
  Pipeline.Work{+rows,h,+w,+k,+s,+p,+ow,positions,+inner} = work
  InputWindow{input,+first,+height} = window
  +offset = offset
  +span = span
  +count = (span / 8 * inner : U32)
  +stencil = Packing.square(height,w,k,s,p,ow,inner,Gemm.blocks((inner / (k * k) : U32)))
  +start = KernelWindow.start(offset,s,ow,first)
  +planes = Packing.call_planes(stencil,start,KernelWindow.padded(span))
  Partition.Leaf{computed(rows,span,inner,weights,bias,
    LibPacking.packed_tiles(stencil,planes,U32.to_nat(count),start,
      LibPacking.transposed(stencil,planes,input,Tree.fresh(Tile.Tile16,Tensors.depth_for(Packing.plane_pairs(stencil,planes)),Tile.filled_pair(0.0))),
      Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)))),offset,span,rows}

def leaf_matches(+work: Pipeline.Work,+offset: U32,+span: U32,+window: InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>) ->
  {ArrayPipeline.leaf(work,offset,span,represent(window),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)) ==
    PartitionLaws.represent(~Tile.Tile8,leaf(work,offset,span,window,weights,bias)) : Partition.Tree<&1,Array<Tile.Tile8>,U32>}:
  Pipeline.Work{+rows,+h,+w,+k,+s,+p,+ow,positions,+inner} = work
  InputWindow{+input,+first,+height} = window
  +count = (span / 8 * inner : U32)
  +stencil = Packing.square(height,w,k,s,p,ow,inner,Gemm.blocks((inner / (k * k) : U32)))
  +start = KernelWindow.start(offset,s,ow,first)
  +planes = Packing.call_planes(stencil,start,KernelWindow.padded(span))
  +pair_depth = Tensors.depth_for(Packing.plane_pairs(stencil,planes))
  +pairs = Tree.fresh(Tile.Tile16,pair_depth,Tile.filled_pair(0.0))
  +tiles = Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))
  +transposed = LibPacking.transposed(stencil,planes,input,pairs)
  +packed = LibPacking.packed_tiles(stencil,planes,U32.to_nat(count),start,transposed,tiles)
  %Equal.sym(Array<Tile.Tile16>,Allocation.array(~Tile.Tile16,pair_depth,Tile.filled_pair(0.0)),Tree.pack(Tile.Tile16,pairs),
    AllocationLaws.fresh(~Tile.Tile16,pair_depth,Tile.filled_pair(0.0))) :
    {Partition.Leaf{ArrayPipeline.computed(rows,span,inner,Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias),ArrayPipeline.planes_packed(stencil,planes,U32.to_nat(count),start,
        Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)),Packing.transpose(stencil,planes,Tree.pack(Tile.Tile8,input),_))),offset,span,rows} ==
      PartitionLaws.represent(~Tile.Tile8,Partition.Leaf{computed(rows,span,inner,weights,bias,packed),offset,span,rows}) : Partition.Tree<&1,Array<Tile.Tile8>,U32>}
  %Equal.sym(Array<Tile.Tile8>,Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)),Tree.pack(Tile.Tile8,tiles),
    AllocationLaws.fresh(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))) :
    {Partition.Leaf{ArrayPipeline.computed(rows,span,inner,Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias),ArrayPipeline.planes_packed(stencil,planes,U32.to_nat(count),start,
        _,Packing.transpose(stencil,planes,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile16,pairs)))),offset,span,rows} ==
      PartitionLaws.represent(~Tile.Tile8,Partition.Leaf{computed(rows,span,inner,weights,bias,packed),offset,span,rows}) : Partition.Tree<&1,Array<Tile.Tile8>,U32>}
  %Equal.sym(Array<Tile.Tile8> & Array<Tile.Tile16>,Packing.transpose(stencil,planes,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile16,pairs)),
    (Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile16,transposed)),LibPacking.transpose_matches(stencil,planes,input,pairs)) :
    {Partition.Leaf{ArrayPipeline.computed(rows,span,inner,Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias),ArrayPipeline.planes_packed(stencil,planes,U32.to_nat(count),start,Tree.pack(Tile.Tile8,tiles),_)),offset,span,rows} ==
      PartitionLaws.represent(~Tile.Tile8,Partition.Leaf{computed(rows,span,inner,weights,bias,packed),offset,span,rows}) : Partition.Tree<&1,Array<Tile.Tile8>,U32>}
  %Equal.sym(Array<Tile.Tile16> & Array<Tile.Tile8>,Packing.pack(stencil,planes,U32.to_nat(count),start,Tree.pack(Tile.Tile16,transposed),Tree.pack(Tile.Tile8,tiles)),
    (Tree.pack(Tile.Tile16,transposed),Tree.pack(Tile.Tile8,packed)),LibPacking.pack_matches(stencil,planes,U32.to_nat(count),start,transposed,tiles)) :
    {Partition.Leaf{ArrayPipeline.computed(rows,span,inner,Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias),_),offset,span,rows} == PartitionLaws.represent(~Tile.Tile8,Partition.Leaf{computed(rows,span,inner,weights,bias,packed),offset,span,rows}) : Partition.Tree<&1,Array<Tile.Tile8>,U32>}
  %Equal.sym(Array<Tile.Tile8>,ArrayPipeline.computed(rows,span,inner,Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias),(Tree.pack(Tile.Tile16,transposed),Tree.pack(Tile.Tile8,packed))),
    Tree.pack(Tile.Tile8,computed(rows,span,inner,weights,bias,packed)),computed_matches(rows,span,inner,weights,bias,transposed,packed)) :
    {Partition.Leaf{_,offset,span,rows} == PartitionLaws.represent(~Tile.Tile8,Partition.Leaf{computed(rows,span,inner,weights,bias,packed),offset,span,rows}) : Partition.Tree<&1,Array<Tile.Tile8>,U32>}
  {==}

# What one part owns: its input window, its weights and its bias.
type Owners is Data:
  Owners{window: InputWindow,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<Tile.Tile8>}

def represent_owners(owners: Owners) -> ArrayPipeline.Owners:
  Owners{window,weights,bias} = owners
  ArrayPipeline.Owners{represent(window),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)}

def halves(+pair: PartitionLaws.Split<InputWindow>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>) -> PartitionLaws.Split<Owners>:
  PartitionLaws.Split{Owners{PartitionLaws.first(~InputWindow,pair),weights,bias},Owners{PartitionLaws.second(~InputWindow,pair),weights,bias}}

# The left half reads the positions of its plan; both halves keep the weights and bias.
def divide(work: Pipeline.Work,left: Pieces.Ranges,right: Pieces.Ranges,owners: Owners) -> PartitionLaws.Split<Owners>:
  Owners{window,+weights,+bias} = owners
  +offset = Pieces.first(left)
  halves(split(work,offset,(Pieces.first(right) - offset : U32),window),weights,bias)

def divide_matches(+work: Pipeline.Work,+left: Pieces.Ranges,+right: Pieces.Ranges,+owners: Owners) ->
  {ArrayPipeline.divide(work,left,right,represent_owners(owners)) ==
    Pieces.Divided{represent_owners(PartitionLaws.first(~Owners,divide(work,left,right,owners))),represent_owners(PartitionLaws.second(~Owners,divide(work,left,right,owners)))} :
    Pieces.Division<ArrayPipeline.Owners>}:
  Owners{+window,+weights,+bias} = owners
  +offset = Pieces.first(left)
  +half = (Pieces.first(right) - offset : U32)
  +pair = split(work,offset,half,window)
  %Equal.sym(ArrayPipeline.InputWindow & ArrayPipeline.InputWindow,ArrayPipeline.split(work,offset,half,represent(window)),
    PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,pair),split_matches(work,offset,half,window)) :
    {ArrayPipeline.divided(_,Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,bias))) ==
      Pieces.Divided{ArrayPipeline.Owners{represent(PartitionLaws.first(~InputWindow,pair)),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)},
        ArrayPipeline.Owners{represent(PartitionLaws.second(~InputWindow,pair)),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)}} : Pieces.Division<ArrayPipeline.Owners>}
  %Equal.sym(PartitionLaws.Split<InputWindow>,pair,PartitionLaws.Split{PartitionLaws.first(~InputWindow,pair),PartitionLaws.second(~InputWindow,pair)},
    PartitionLaws.split_eta(~InputWindow,pair)) :
    {ArrayPipeline.divided(PartitionLaws.represent_pair(~ArrayPipeline.InputWindow,~InputWindow,~represent,_),
        Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,bias))) ==
      Pieces.Divided{ArrayPipeline.Owners{represent(PartitionLaws.first(~InputWindow,pair)),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)},
        ArrayPipeline.Owners{represent(PartitionLaws.second(~InputWindow,pair)),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)}} : Pieces.Division<ArrayPipeline.Owners>}
  %Equal.sym(Array<Tile.Tile8> & Array<Tile.Tile8>,Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),(Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,weights)),
    StorageLaws.clone(Tile.Tile8,weights)) :
    {ArrayPipeline.divided((represent(PartitionLaws.first(~InputWindow,pair)),represent(PartitionLaws.second(~InputWindow,pair))),_,Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,bias))) ==
      Pieces.Divided{ArrayPipeline.Owners{represent(PartitionLaws.first(~InputWindow,pair)),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)},
        ArrayPipeline.Owners{represent(PartitionLaws.second(~InputWindow,pair)),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)}} : Pieces.Division<ArrayPipeline.Owners>}
  %Equal.sym(Array<Tile.Tile8> & Array<Tile.Tile8>,Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,bias)),(Tree.pack(Tile.Tile8,bias),Tree.pack(Tile.Tile8,bias)),StorageLaws.clone(Tile.Tile8,bias)) :
    {ArrayPipeline.divided((represent(PartitionLaws.first(~InputWindow,pair)),represent(PartitionLaws.second(~InputWindow,pair))),
        (Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,weights)),_) ==
      Pieces.Divided{ArrayPipeline.Owners{represent(PartitionLaws.first(~InputWindow,pair)),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)},
        ArrayPipeline.Owners{represent(PartitionLaws.second(~InputWindow,pair)),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)}} : Pieces.Division<ArrayPipeline.Owners>}
  {==}

def part(work: Pipeline.Work,offset: U32,span: U32,owners: Owners) -> CellResult():
  Owners{window,weights,bias} = owners
  leaf(work,offset,span,window,weights,bias)

def part_matches(+work: Pipeline.Work,+offset: U32,+span: U32,+owners: Owners) ->
  {ArrayPipeline.part(work,offset,span,represent_owners(owners)) == PartitionLaws.represent(~Tile.Tile8,part(work,offset,span,owners)) : Partition.Tree<&1,Array<Tile.Tile8>,U32>}:
  Owners{+window,+weights,+bias} = owners
  leaf_matches(work,offset,span,window,weights,bias)

# Positions [offset, offset + span) halved depth times at whole blocks of eight.
def partition(depth: Nat,work: Pipeline.Work,offset: U32,span: U32,input: InputWindow,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<Tile.Tile8>) -> CellResult():
  Walk.model(~Owners,~Tree.Tree<Tile.Tile8>,~U32,~Pipeline.Work,~divide,~part,Pieces.unit_ranges(depth,8,offset,span),work,Owners{input,weights,bias})

def partition_matches(+depth: Nat,+work: Pipeline.Work,+offset: U32,+span: U32,+input: InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>) ->
  {ArrayPipeline.partition(depth,work,offset,span,represent(input),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)) ==
    PartitionLaws.represent(~Tile.Tile8,partition(depth,work,offset,span,input,weights,bias)) : Partition.Tree<&1,Array<Tile.Tile8>,U32>}:
  Walk.walk_matches(~ArrayPipeline.Owners,~Owners,~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Pipeline.Work,~Unit,~Pieces.always,~ArrayPipeline.divide,~ArrayPipeline.part,~ArrayPipeline.whole,
    ~represent_owners,~PartitionLaws.pack_values(~Tile.Tile8),~divide,~part,~divide_matches,~part_matches,
    Pieces.unit_ranges(depth,8,offset,span),True{},Unit{},work,Owners{input,weights,bias})

# The halves of a halved plan start at its offset and after its first half.
def divide_halved(+rest: Nat,+work: Pipeline.Work,+offset: U32,+span: U32,+window: InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>) ->
  {divide(work,Pieces.unit_ranges(rest,8,offset,(span / 16 * 8 : U32)),Pieces.unit_ranges(rest,8,(offset + span / 16 * 8 : U32),(span - span / 16 * 8 : U32)),Owners{window,weights,bias}) ==
    halves(split(work,offset,(span / 16 * 8 : U32),window),weights,bias) : PartitionLaws.Split<Owners>}:
  +half = (span / 16 * 8 : U32)
  +next = (offset + half : U32)
  +left = Pieces.unit_ranges(rest,8,offset,half)
  +right = Pieces.unit_ranges(rest,8,next,(span - half : U32))
  %Equal.sym(U32,Pieces.first(left),offset,Walk.unit_ranges_first(rest,8,offset,half)) :
    {halves(split(work,_,(Pieces.first(right) - _ : U32),window),weights,bias) == halves(split(work,offset,half,window),weights,bias) : PartitionLaws.Split<Owners>}
  %Equal.sym(U32,Pieces.first(right),next,Walk.unit_ranges_first(rest,8,next,(span - half : U32))) :
    {halves(split(work,offset,(_ - offset : U32),window),weights,bias) == halves(split(work,offset,half,window),weights,bias) : PartitionLaws.Split<Owners>}
  %Equal.sym(U32,(next - offset : U32),half,Arithmetic.subtract_first(offset,half)) :
    {halves(split(work,offset,_,window),weights,bias) == halves(split(work,offset,half,window),weights,bias) : PartitionLaws.Split<Owners>}
  {==}

# One split of the halved plan: the left half copies its band, the right
# half keeps the window.
def partition_step(+rest: Nat,+work: Pipeline.Work,+offset: U32,+span: U32,+window: InputWindow,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>) ->
  {partition(1n+rest,work,offset,span,window,weights,bias) ==
    Partition.Fork{partition(rest,work,offset,(span / 16 * 8 : U32),PartitionLaws.first(~InputWindow,split(work,offset,(span / 16 * 8 : U32),window)),weights,bias),
      partition(rest,work,(offset + span / 16 * 8 : U32),(span - span / 16 * 8 : U32),PartitionLaws.second(~InputWindow,split(work,offset,(span / 16 * 8 : U32),window)),weights,bias)} :
    CellResult()}:
  +half = (span / 16 * 8 : U32)
  +left = Pieces.unit_ranges(rest,8,offset,half)
  +right = Pieces.unit_ranges(rest,8,(offset + half : U32),(span - half : U32))
  %Equal.sym(PartitionLaws.Split<Owners>,divide(work,left,right,Owners{window,weights,bias}),halves(split(work,offset,half,window),weights,bias),
    divide_halved(rest,work,offset,span,window,weights,bias)) :
    {Partition.Fork{Walk.model(~Owners,~Tree.Tree<Tile.Tile8>,~U32,~Pipeline.Work,~divide,~part,left,work,PartitionLaws.first(~Owners,_)),
        Walk.model(~Owners,~Tree.Tree<Tile.Tile8>,~U32,~Pipeline.Work,~divide,~part,right,work,PartitionLaws.second(~Owners,_))} ==
      Partition.Fork{partition(rest,work,offset,half,PartitionLaws.first(~InputWindow,split(work,offset,half,window)),weights,bias),
        partition(rest,work,(offset + half : U32),(span - half : U32),PartitionLaws.second(~InputWindow,split(work,offset,half,window)),weights,bias)} : CellResult()}
  {==}

def native_gather(partitions: Partition.Tree<&1,Array<Tile.Tile8>,U32>,output: Array<Tile.Tile8>,positions: U32) -> Array<Tile.Tile8>:
  Partition.gather(~&1,~Array<Tile.Tile8>,~ArrayPipeline.gathered,partitions,output,positions)

def gather_matches(+partitions: CellResult(),+output: Tree.Tree<Tile.Tile8>,+positions: U32) ->
  {native_gather(PartitionLaws.represent(~Tile.Tile8,partitions),Tree.pack(Tile.Tile8,output),positions) == Tree.pack(Tile.Tile8,gather(partitions,output,positions)) : Array<Tile.Tile8>}:
  PartitionLaws.gathered(~&1,~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~PartitionLaws.pack_values(~Tile.Tile8),~ArrayPipeline.gathered,
    ~PartitionLaws.write_copy(~Tile.Tile8,~cell_block_writes),~cell_block,partitions,output,positions)

def finish_leaf(identity: Bool,input: Tree.Tree<Tile.Tile8>,+rows: U32,offset: U32,span: U32,+positions: U32) -> Tree.Tree<Tile.Tile8>:
  match identity:
    case True{}: input
    case False{}: gather(Partition.Leaf{input,offset,span,rows},allocated_tiles(rows,positions),positions)

def finish_leaf_matches(+identity: Bool,+input: Tree.Tree<Tile.Tile8>,+rows: U32,+offset: U32,+span: U32,+positions: U32) ->
  {ArrayPipeline.finish_leaf(identity,Tree.pack(Tile.Tile8,input),rows,offset,span,positions) == Tree.pack(Tile.Tile8,finish_leaf(identity,input,rows,offset,span,positions)) : Array<Tile.Tile8>}:
  match identity:
    case True{}: {==}
    case False{}:
      %Equal.sym(Array<Tile.Tile8>,ArrayPipeline.allocated_tiles(rows,positions),Tree.pack(Tile.Tile8,allocated_tiles(rows,positions)),allocated_tiles_matches(rows,positions)) :
        {ArrayPipeline.gathered(rows,offset,positions,span,Tree.pack(Tile.Tile8,input),_) == Tree.pack(Tile.Tile8,finish_leaf(False{},input,rows,offset,span,positions)) : Array<Tile.Tile8>}
      cell_block(rows,offset,positions,span,input,allocated_tiles(rows,positions))

def finish(partitions: CellResult(),+rows: U32,+positions: U32) -> Tree.Tree<Tile.Tile8>:
  match partitions:
    case Partition.Leaf{input,+offset,+span,channels}: finish_leaf(U32.is_zero(offset) && U32.is_eq(span,positions),input,channels,offset,span,positions)
    case Partition.Fork{left,right}: gather(Partition.Fork{left,right},allocated_tiles(rows,positions),positions)

def finish_matches(+partitions: CellResult(),+rows: U32,+positions: U32) ->
  {ArrayPipeline.finish(PartitionLaws.represent(~Tile.Tile8,partitions),rows,positions) == Tree.pack(Tile.Tile8,finish(partitions,rows,positions)) : Array<Tile.Tile8>}:
  match partitions:
    case Partition.Leaf{input,+offset,+span,channels}: finish_leaf_matches(U32.is_zero(offset) && U32.is_eq(span,positions),input,channels,offset,span,positions)
    case Partition.Fork{+left,+right}:
      %Equal.sym(Array<Tile.Tile8>,ArrayPipeline.allocated_tiles(rows,positions),Tree.pack(Tile.Tile8,allocated_tiles(rows,positions)),allocated_tiles_matches(rows,positions)) :
        {native_gather(Partition.Fork{PartitionLaws.represent(~Tile.Tile8,left),PartitionLaws.represent(~Tile.Tile8,right)},_,positions) ==
          Tree.pack(Tile.Tile8,finish(Partition.Fork{left,right},rows,positions)) : Array<Tile.Tile8>}
      gather_matches(Partition.Fork{left,right},allocated_tiles(rows,positions),positions)

def run_cells(depth: U32,work: Pipeline.Work,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Pipeline.Work{+rows,+h,w,k,s,p,ow,+positions,inner} = work
  +padded = KernelWindow.padded(positions)
  +limit = Pipeline.depth_limit(padded)
  finish(partition(U32.to_nat(U32.min(depth,limit)),work,0,padded,InputWindow{input,0,h},weights,bias),rows,positions)

def run_cells_matches(+depth: U32,+work: Pipeline.Work,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>) ->
  {ArrayPipeline.run_cells(depth,work,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)) ==
    Tree.pack(Tile.Tile8,run_cells(depth,work,input,weights,bias)) : Array<Tile.Tile8>}:
  Pipeline.Work{+rows,+h,w,k,s,p,ow,+positions,inner} = work
  +padded = KernelWindow.padded(positions)
  +split = U32.to_nat(U32.min(depth,Pipeline.depth_limit(padded)))
  +parts = partition(split,work,0,padded,InputWindow{input,0,h},weights,bias)
  %Equal.sym(Partition.Tree<&1,Array<Tile.Tile8>,U32>,
    ArrayPipeline.partition(split,work,0,padded,ArrayPipeline.InputWindow{Tree.pack(Tile.Tile8,input),0,h},Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,bias)),
    PartitionLaws.represent(~Tile.Tile8,parts),partition_matches(split,work,0,padded,InputWindow{input,0,h},weights,bias)) :
    {ArrayPipeline.finish(_,rows,positions) == Tree.pack(Tile.Tile8,finish(parts,rows,positions)) : Array<Tile.Tile8>}
  finish_matches(parts,rows,positions)

# ---- Values and cells ----

# The cells of the channels of every pixel, the groups of a pixel side by side.
def input_cells(+channels: U32,+pixels: U32,input: Tree.Tree<F32>) -> Tree.Tree<Tile.Tile8>:
  +groups = Gemm.blocks(channels)
  IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,TiledValues.tiled_blocks(pixels,TiledCells.Tiling{pixels,channels,groups,0},input,
    Tree.fresh(Tile.Tile8,Tensors.depth_for((pixels * groups : U32)),Tile.filled(0.0))))

def cells_of(+result: AccessLaws.ModelResult<Tree.Tree<F32>,Tile.Tile8>) ->
  {ArrayPipeline.cells(AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),result)) ==
    Tree.pack(Tile.Tile8,IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,result)) : Array<Tile.Tile8>}:
  AccessLaws.ModelResult{reader,output} = result
  {==}

def input_cells_matches(+channels: U32,+pixels: U32,+input: Tree.Tree<F32>) ->
  {ArrayPipeline.input_cells(channels,pixels,Tree.pack(F32,input)) == Tree.pack(Tile.Tile8,input_cells(channels,pixels,input)) : Array<Tile.Tile8>}:
  +groups = Gemm.blocks(channels)
  +context = {TiledCells.Tiling{pixels,channels,groups,0} : TiledCells.Tiling}
  +count = (pixels * groups : U32)
  +output = Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))
  %Equal.sym(Array<Tile.Tile8>,Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)),Tree.pack(Tile.Tile8,output),AllocationLaws.fresh(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))) :
    {ArrayPipeline.cells(TiledCells.tiled(pixels,context,Tree.pack(F32,input),_)) == Tree.pack(Tile.Tile8,input_cells(channels,pixels,input)) : Array<Tile.Tile8>}
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,TiledCells.tiled(pixels,context,Tree.pack(F32,input),Tree.pack(Tile.Tile8,output)),
    AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),TiledValues.tiled_blocks(pixels,context,input,output)),
    TiledValues.tiled_run(pixels,context,input,output)) :
    {ArrayPipeline.cells(_) == Tree.pack(Tile.Tile8,input_cells(channels,pixels,input)) : Array<Tile.Tile8>}
  cells_of(TiledValues.tiled_blocks(pixels,context,input,output))

def leaf_values(+result: AccessLaws.ModelResult<Tree.Tree<Tile.Tile8>,F32>) ->
  {ArrayPipeline.values(AccessLaws.pack_result(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~F32,~ArrayLaws.pack_tree(~Tile.Tile8),result)) ==
    Tree.pack(F32,IndexedValues.output_tree(~Tree.Tree<Tile.Tile8>,~F32,result)) : Array<F32>}:
  AccessLaws.ModelResult{reader,output} = result
  {==}

def untiled_matches(+rows: U32,+positions: U32,+input: Tree.Tree<Tile.Tile8>) ->
  {ArrayPipeline.values(TiledCells.untiled(rows,TiledCells.Cells{positions,positions,rows,Gemm.blocks(rows),Grid.Innermost{}},Tree.pack(Tile.Tile8,input),ArrayPipeline.allocated(rows,positions))) ==
    Tree.pack(F32,TiledValues.image_values(rows,positions,input)) : Array<F32>}:
  +context = {TiledCells.Cells{positions,positions,rows,Gemm.blocks(rows),Grid.Innermost{}} : TiledCells.Cells}
  +output = allocated(rows,positions)
  %Equal.sym(Array<F32>,ArrayPipeline.allocated(rows,positions),Tree.pack(F32,output),allocated_matches(rows,positions)) :
    {ArrayPipeline.values(TiledCells.untiled(rows,context,Tree.pack(Tile.Tile8,input),_)) == Tree.pack(F32,TiledValues.image_values(rows,positions,input)) : Array<F32>}
  %Equal.sym(Array<Tile.Tile8> & Array<F32>,TiledCells.untiled(rows,context,Tree.pack(Tile.Tile8,input),Tree.pack(F32,output)),
    AccessLaws.pack_result(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~F32,~ArrayLaws.pack_tree(~Tile.Tile8),TiledValues.untiled_blocks(rows,context,input,output)),
    TiledValues.untiled_run(rows,context,input,output)) :
    {ArrayPipeline.values(_) == Tree.pack(F32,TiledValues.image_values(rows,positions,input)) : Array<F32>}
  leaf_values(TiledValues.untiled_blocks(rows,context,input,output))

def run(depth: U32,work: Pipeline.Work,input: Tree.Tree<F32>,weights: Tree.Tree<F32>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  Pipeline.Work{+rows,+h,+w,+k,s,p,ow,+positions,+inner} = work
  TiledValues.image_values(rows,positions,run_cells(depth,work,input_cells((inner / (k * k) : U32),(h * w : U32),input),WeightsLaws.run(rows,inner,weights),WeightsLaws.run(rows,1,bias)))

def run_matches(+depth: U32,+work: Pipeline.Work,+input: Tree.Tree<F32>,+weights: Tree.Tree<F32>,+bias: Tree.Tree<F32>) ->
  {ArrayPipeline.run(depth,work,Tree.pack(F32,input),Tree.pack(F32,weights),Tree.pack(F32,bias)) ==
    Tree.pack(F32,run(depth,work,input,weights,bias)) : Array<F32>}:
  Pipeline.Work{+rows,+h,+w,+k,s,p,ow,+positions,+inner} = work
  +channels = (inner / (k * k) : U32)
  +plane = (h * w : U32)
  +image_cells = input_cells(channels,plane,input)
  +weight_cells = WeightsLaws.run(rows,inner,weights)
  +bias_cells = WeightsLaws.run(rows,1,bias)
  +output = run_cells(depth,work,image_cells,weight_cells,bias_cells)
  +context = {TiledCells.Cells{positions,positions,rows,Gemm.blocks(rows),Grid.Innermost{}} : TiledCells.Cells}
  %Equal.sym(Array<Tile.Tile8>,ArrayPipeline.input_cells(channels,plane,Tree.pack(F32,input)),Tree.pack(Tile.Tile8,image_cells),input_cells_matches(channels,plane,input)) :
    {ArrayPipeline.values(TiledCells.untiled(rows,context,ArrayPipeline.run_cells(depth,work,_,Weights.run(rows,inner,Tree.pack(F32,weights)),Weights.run(rows,1,Tree.pack(F32,bias))),ArrayPipeline.allocated(rows,positions))) ==
      Tree.pack(F32,TiledValues.image_values(rows,positions,output)) : Array<F32>}
  %Equal.sym(Array<Tile.Tile8>,Weights.run(rows,inner,Tree.pack(F32,weights)),Tree.pack(Tile.Tile8,weight_cells),WeightsLaws.run_matches(rows,inner,weights)) :
    {ArrayPipeline.values(TiledCells.untiled(rows,context,ArrayPipeline.run_cells(depth,work,Tree.pack(Tile.Tile8,image_cells),_,Weights.run(rows,1,Tree.pack(F32,bias))),ArrayPipeline.allocated(rows,positions))) ==
      Tree.pack(F32,TiledValues.image_values(rows,positions,output)) : Array<F32>}
  %Equal.sym(Array<Tile.Tile8>,Weights.run(rows,1,Tree.pack(F32,bias)),Tree.pack(Tile.Tile8,bias_cells),WeightsLaws.run_matches(rows,1,bias)) :
    {ArrayPipeline.values(TiledCells.untiled(rows,context,ArrayPipeline.run_cells(depth,work,Tree.pack(Tile.Tile8,image_cells),Tree.pack(Tile.Tile8,weight_cells),_),ArrayPipeline.allocated(rows,positions))) ==
      Tree.pack(F32,TiledValues.image_values(rows,positions,output)) : Array<F32>}
  %Equal.sym(Array<Tile.Tile8>,ArrayPipeline.run_cells(depth,work,Tree.pack(Tile.Tile8,image_cells),Tree.pack(Tile.Tile8,weight_cells),Tree.pack(Tile.Tile8,bias_cells)),
    Tree.pack(Tile.Tile8,output),run_cells_matches(depth,work,image_cells,weight_cells,bias_cells)) :
    {ArrayPipeline.values(TiledCells.untiled(rows,context,_,ArrayPipeline.allocated(rows,positions))) == Tree.pack(F32,TiledValues.image_values(rows,positions,output)) : Array<F32>}
  untiled_matches(rows,positions,output)
