# The band partition of one image on storage-tree owners. A split copies the
# gradient rows the left band reads into a fresh tree and keeps the source for
# the right band, a leaf runs the leaf band on a model output, and the
# parallel walk of the kernel is the represented model walk on every plan.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_window.bend as KernelWindow
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../traversal_partition.bend as Partition
import ../storage_allocation.bend as NativeAllocation
import ../storage_buffer.bend as Storage
import ./input_gradient_phase_domain.bend as Domain
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_allocation_refinement.bend as Allocation
import ./storage_refinement.bend as StorageLaws
import ./write_program.bend as Writes
import ./kernel_window_copy.bend as WindowCopy
import ./contiguous_copy.bend as Contiguous
import ./tensor_band_copy.bend as BandCopy
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./traversal_partition_refinement.bend as PartitionLaws
import ./tensor_band_walk.bend as BandWalk
import ./tensor_band_model.bend as Model

# What one part owns: its source rows, the packed weights and the table.
type Owners is Data:
  Owners{source: Execution.Source,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>}
def represent(state: Owners) -> Kernel.Owners:
  Owners{source,weights,offsets} = state
  Kernel.Owners{Execution.represent_source(source),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets)}
def owner_source(state: Owners) -> Execution.Source:
  Owners{source,weights,offsets} = state
  source
def owner_weights(state: Owners) -> Tree.Tree<Tile.Tile8>:
  Owners{source,weights,offsets} = state
  weights
def owner_offsets(state: Owners) -> Tree.Tree<U32>:
  Owners{source,weights,offsets} = state
  offsets

# The copy of the gradient rows [start, start + count) of every output
# channel: one contiguous plane copy per channel into a fresh tree.
def copy_depth(+outputs: U32,+count: U32,+width: U32) -> Nat:
  Tensors.depth_for((outputs * count * width + 1 : U32))
def copy_writes(+outputs: U32,+count: U32,+width: U32,+plane: U32,+from: U32,+data: Tree.Tree<F32>) -> List<&2,Writes.Assignment<F32>>:
  WindowCopy.writes(U32.to_nat(outputs),0,(count * width : U32),plane,from,(count * width : U32),data)
def copied(+geometry: Kernel.Geometry,+start: U32,+count: U32,+source: Execution.Source) -> Tree.Tree<F32>:
  Kernel.Geometry{c,+o,h,w,k,s,p,gh,+gw} = geometry
  Execution.Source{+data,+base,+plane,+first,rows} = source
  Writes.pure(F32,copy_writes(o,count,gw,plane,(base + (start - first) * gw : U32),data),Tree.fresh(F32,copy_depth(o,count,gw),0.0))
def left_source(+geometry: Kernel.Geometry,+start: U32,+count: U32,+source: Execution.Source) -> Execution.Source:
  Execution.Source{copied(geometry,start,count,source),0,(count * Domain.gradient_columns(geometry) : U32),start,count}

# The plane copies are the band copy of the tensor library from offset zero.
def band_writes(remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+target_stride: U32,+input: Tree.Tree<F32>) ->
  {WindowCopy.writes(remaining,channel,count,source_stride,source_start,target_stride,input) ==
    BandCopy.writes(~F32,remaining,channel,count,source_stride,target_stride,source_start,0,input) : List<&2,Writes.Assignment<F32>>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +plane = Contiguous.writes(~F32,U32.to_nat(count),input,0,(channel * source_stride + source_start : U32),(channel * target_stride : U32))
      %Arithmetic.add_commutative((channel * source_stride : U32),source_start) :
        {WindowCopy.writes(1n+rest,channel,count,source_stride,source_start,target_stride,input) ==
          List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(count),input,0,_,(0 + channel * target_stride : U32)),
            BandCopy.writes(~F32,rest,(channel + 1 : U32),count,source_stride,target_stride,source_start,0,input)) : List<&2,Writes.Assignment<F32>>}
      %Equal.sym(U32,(0 + channel * target_stride : U32),(channel * target_stride : U32),Arithmetic.add_zero_left((channel * target_stride : U32))) :
        {WindowCopy.writes(1n+rest,channel,count,source_stride,source_start,target_stride,input) ==
          List.append(&2,Writes.Assignment<F32>,Contiguous.writes(~F32,U32.to_nat(count),input,0,(channel * source_stride + source_start : U32),_),
            BandCopy.writes(~F32,rest,(channel + 1 : U32),count,source_stride,target_stride,source_start,0,input)) : List<&2,Writes.Assignment<F32>>}
      Equal.cong(List<&2,Writes.Assignment<F32>>,List<&2,Writes.Assignment<F32>>,tail => List.append(&2,Writes.Assignment<F32>,plane,tail),
        WindowCopy.writes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input),
        BandCopy.writes(~F32,rest,(channel + 1 : U32),count,source_stride,target_stride,source_start,0,input),
        band_writes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input))

# The model split: the left band reads a copy of its rows, the right band
# keeps the source; both keep the weights and the table.
def split(task: Kernel.Task,left: Bands.Plan,right: Bands.Plan,state: Owners) -> PartitionLaws.Split<Owners>:
  Kernel.Task{+geometry,layout} = task
  Owners{+source,+weights,+offsets} = state
  +row = Bands.first(left)
  +half = (Bands.first(right) - row : U32)
  +start = Kernel.copy_start(geometry,row,half)
  +count = Kernel.copy_count(geometry,row,half)
  PartitionLaws.Split{Owners{left_source(geometry,start,count,source),weights,offsets},Owners{source,weights,offsets}}

def split_matches(+task: Kernel.Task,+left: Bands.Plan,+right: Bands.Plan,+state: Owners) ->
  {Kernel.split(task,Bands.first(left),(Bands.first(right) - Bands.first(left) : U32),Execution.represent_source(owner_source(state))) ==
    (Execution.represent_source(owner_source(PartitionLaws.first(~Owners,split(task,left,right,state)))),
      Execution.represent_source(owner_source(PartitionLaws.second(~Owners,split(task,left,right,state))))) : Kernel.Source & Kernel.Source}:
  Kernel.Task{+geometry,+layout} = task
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Owners{+source,+weights,+offsets} = state
  Execution.Source{+data,+base,+plane,+first,+rows} = source
  +row = Bands.first(left)
  +half = (Bands.first(right) - row : U32)
  +start = Kernel.copy_start(geometry,row,half)
  +count = Kernel.copy_count(geometry,row,half)
  +depth = copy_depth(o,count,gw)
  +from = (base + (start - first) * gw : U32)
  +span = (count * gw : U32)
  +fresh = Tree.fresh(F32,depth,0.0)
  %Equal.sym(Array<F32>,NativeAllocation.array(~F32,depth,0.0),Tree.pack(F32,fresh),Allocation.fresh(~F32,depth,0.0)) :
    {Kernel.copied_pair(start,count,gw,base,plane,first,rows,KernelWindow.copy_planes(U32.to_nat(o),0,span,plane,from,span,(Tree.pack(F32,data),_))) ==
      (Execution.represent_source(owner_source(PartitionLaws.first(~Owners,split(task,left,right,state)))),
        Execution.represent_source(owner_source(PartitionLaws.second(~Owners,split(task,left,right,state))))) : Kernel.Source & Kernel.Source}
  %Equal.sym(Array<F32> & Array<F32>,KernelWindow.copy_planes(U32.to_nat(o),0,span,plane,from,span,(Tree.pack(F32,data),Tree.pack(F32,fresh))),
    (Tree.pack(F32,data),Tree.pack(F32,Writes.pure(F32,WindowCopy.writes(U32.to_nat(o),0,span,plane,from,span,data),fresh))),
    WindowCopy.planes(U32.to_nat(o),0,span,plane,from,span,data,fresh)) :
    {Kernel.copied_pair(start,count,gw,base,plane,first,rows,_) ==
      (Execution.represent_source(owner_source(PartitionLaws.first(~Owners,split(task,left,right,state)))),
        Execution.represent_source(owner_source(PartitionLaws.second(~Owners,split(task,left,right,state))))) : Kernel.Source & Kernel.Source}
  {==}

def divide_matches(+task: Kernel.Task,+left: Bands.Plan,+right: Bands.Plan,+state: Owners) ->
  {Kernel.divide(task,left,right,represent(state)) ==
    Bands.Divided{represent(PartitionLaws.first(~Owners,split(task,left,right,state))),represent(PartitionLaws.second(~Owners,split(task,left,right,state)))} : Bands.Division<Kernel.Owners>}:
  Kernel.Task{+geometry,+layout} = task
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Owners{+source,+weights,+offsets} = state
  +row = Bands.first(left)
  +half = (Bands.first(right) - row : U32)
  +start = Kernel.copy_start(geometry,row,half)
  +count = Kernel.copy_count(geometry,row,half)
  %Equal.sym(Kernel.Source & Kernel.Source,Kernel.split(task,row,half,Execution.represent_source(source)),
    (Execution.represent_source(left_source(geometry,start,count,source)),Execution.represent_source(source)),split_matches(task,left,right,state)) :
    {Kernel.divided(_,Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),Array.clone(U32,Tree.pack(U32,offsets))) ==
      Bands.Divided{represent(PartitionLaws.first(~Owners,split(task,left,right,state))),represent(PartitionLaws.second(~Owners,split(task,left,right,state)))} : Bands.Division<Kernel.Owners>}
  %Equal.sym(Array<Tile.Tile8> & Array<Tile.Tile8>,Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),(Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,weights)),StorageLaws.clone(Tile.Tile8,weights)) :
    {Kernel.divided((Execution.represent_source(left_source(geometry,start,count,source)),Execution.represent_source(source)),_,Array.clone(U32,Tree.pack(U32,offsets))) ==
      Bands.Divided{represent(PartitionLaws.first(~Owners,split(task,left,right,state))),represent(PartitionLaws.second(~Owners,split(task,left,right,state)))} : Bands.Division<Kernel.Owners>}
  %Equal.sym(Array<U32> & Array<U32>,Array.clone(U32,Tree.pack(U32,offsets)),(Tree.pack(U32,offsets),Tree.pack(U32,offsets)),StorageLaws.clone(U32,offsets)) :
    {Kernel.divided((Execution.represent_source(left_source(geometry,start,count,source)),Execution.represent_source(source)),(Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,weights)),_) ==
      Bands.Divided{represent(PartitionLaws.first(~Owners,split(task,left,right,state))),represent(PartitionLaws.second(~Owners,split(task,left,right,state)))} : Bands.Division<Kernel.Owners>}
  {==}

# A leaf runs the leaf band of its rows on a fresh local output of
# [channels][rows][width] values.
def local_model(+count: U32) -> Buffer.Model<F32>:
  Buffer.remodel(~F32,Tensors.buffer(~F32,(0.0 + 0.0 : F32),count))
def local_count(+geometry: Kernel.Geometry,+rows: U32) -> U32:
  Kernel.Geometry{+c,o,h,+w,k,s,p,gh,gw} = geometry
  (c * rows * w : U32)
def context(+task: Kernel.Task,+first: U32,+rows: U32) -> Kernel.LeafContext:
  Kernel.Task{+geometry,+layout} = task
  Kernel.Geometry{c,o,h,+w,k,s,p,gh,gw} = geometry
  Kernel.LeafContext{Kernel.Task{geometry,layout},first,rows,0,(rows * w : U32)}
def model_leaf(task: Kernel.Task,first: U32,rows: U32,state: Owners) -> Partition.Tree<&2,Execution.Part,Unit>:
  Kernel.Task{+geometry,+layout} = task
  Kernel.Geometry{+c,o,h,+w,k,s,p,gh,gw} = geometry
  Owners{source,weights,offsets} = state
  +first = first
  +rows = rows
  Partition.Leaf{Execution.leaf_band(Kernel.LeafContext{Kernel.Task{geometry,layout},first,rows,0,(rows * w : U32)},source,weights,offsets,local_model((c * rows * w : U32))),first,rows,Unit{}}

def represented(tree: Partition.Tree<&2,Execution.Part,Unit>) -> Partition.Tree<&1,Kernel.Part,Unit>:
  PartitionLaws.carry(~&2,~&1,~Execution.Part,~Kernel.Part,~Unit,~Execution.represent_part,tree)

def leaf_matches(+task: Kernel.Task,+first: U32,+rows: U32,+state: Owners) ->
  {Kernel.leaf(task,first,rows,represent(state)) == represented(model_leaf(task,first,rows,state)) : Partition.Tree<&1,Kernel.Part,Unit>}:
  Kernel.Task{+geometry,+layout} = task
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Owners{+source,+weights,+offsets} = state
  +count = (c * rows * w : U32)
  +context = {Kernel.LeafContext{Kernel.Task{geometry,layout},first,rows,0,(rows * w : U32)} : Kernel.LeafContext}
  %Buffer.remodeled(~F32,Tensors.buffer(~F32,(0.0 + 0.0 : F32),count)) :
    {Partition.Leaf{Kernel.leaf_band(context,Execution.represent_source(source),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),_),first,rows,Unit{}} ==
      represented(model_leaf(task,first,rows,state)) : Partition.Tree<&1,Kernel.Part,Unit>}
  %Equal.sym(Kernel.Part,Kernel.leaf_band(context,Execution.represent_source(source),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,local_model(count))),
    Execution.represent_part(Execution.leaf_band(context,source,weights,offsets,local_model(count))),Execution.leaf_matches(context,source,weights,offsets,local_model(count))) :
    {Partition.Leaf{_,first,rows,Unit{}} == represented(model_leaf(task,first,rows,state)) : Partition.Tree<&1,Kernel.Part,Unit>}
  {==}

# The parallel walk of the kernel over a plan is the represented model walk.
def model(layout: Bands.Plan,+task: Kernel.Task,+state: Owners) -> Partition.Tree<&2,Execution.Part,Unit>:
  BandWalk.model(~Owners,~Execution.Part,~Unit,~Kernel.Task,~split,~model_leaf,layout,task,state)
def walk(+layout: Bands.Plan,+mode: Bool,+task: Kernel.Task,+state: Owners) ->
  {Bands.each_part(~Kernel.Owners,~Kernel.Part,~Unit,~Kernel.Task,~Unit,~Bands.always,~Kernel.divide,~Kernel.leaf,~Kernel.whole,layout,mode,Unit{},task,represent(state)) ==
    represented(model(layout,task,state)) : Partition.Tree<&1,Kernel.Part,Unit>}:
  BandWalk.walk_matches(~Kernel.Owners,~Owners,~Kernel.Part,~Execution.Part,~Unit,~Kernel.Task,~Unit,~Bands.always,~Kernel.divide,~Kernel.leaf,~Kernel.whole,
    ~represent,~Execution.represent_part,~split,~model_leaf,~divide_matches,~leaf_matches,layout,mode,Unit{},task,state)

# The gradient of the last piece of a model tree is the source of the walk.
def last(tree: Partition.Tree<&2,Execution.Part,Unit>) -> Tree.Tree<F32>:
  Model.shaped_last(~Execution.Part,~Tree.Tree<F32>,~Execution.gradient,Model.shaped_parts(~Execution.Part,tree))
def model_last(layout: Bands.Plan,+task: Kernel.Task,+state: Owners) ->
  {last(model(layout,task,state)) == Execution.source_data(owner_source(state)) : Tree.Tree<F32>}:
  match layout:
    case Bands.Rows{+first,+rows}:
      Kernel.Task{+geometry,+tiling} = task
      Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
      Owners{+source,+weights,+offsets} = state
      Execution.leaf_gradient(Kernel.LeafContext{Kernel.Task{geometry,tiling},first,rows,0,(rows * w : U32)},source,weights,offsets,local_model((c * rows * w : U32)))
    case Bands.Split{+left,+right}:
      Kernel.Task{+geometry,+tiling} = task
      Owners{+source,+weights,+offsets} = state
      model_last(right,task,PartitionLaws.second(~Owners,split(task,left,right,state)))
