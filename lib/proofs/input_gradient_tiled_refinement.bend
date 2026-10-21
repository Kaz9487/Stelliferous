# The actual tile fill, table writes and tile copies on storage-tree models.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../kernel_spatial_packing.bend as Spatial
import ../traversal_loop.bend as Loop
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as Storage
import ./kernel_spatial_packing_samples.bend as Samples
import ./loop_projection.bend as Projection
import ./traversal_array_refinement.bend as Arrays

type Fill is Data:
  Fill{input: Tree.Tree<F32>,tiles: Tree.Tree<Tile.Tile8>,index: U32,row: U32,column: U32,first: U32,base: U32,plane: U32,height: U32,width: U32,shifts: Nat}

def represent(state: Fill) -> Kernel.Fill:
  Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts} = state
  Kernel.Fill{Tree.pack(F32,input),Tree.pack(Tile.Tile8,tiles),index,row,column,first,base,plane,height,width,shifts}

def fill_shift(state: Fill) -> Fill:
  Fill{+input,tiles,+index,+row,+column,first,+base,plane,+height,+width,shifts} = state
  Fill{input,Tree.write(Tile.Tile8,tiles,index,Samples.pure_vector(input,Spatial.PixelBlock{row,column,base,height,width,1})),
    (index + 1 : U32),row,(column + 1 : U32),first,base,plane,height,width,shifts}

def shift_matches(+state: Fill) -> {Kernel.fill_shift(represent(state)) == represent(fill_shift(state)) : Kernel.Fill}:
  Fill{+input,+tiles,+index,+row,+column,+first,+base,+plane,+height,+width,+shifts} = state
  +context = {Spatial.PixelBlock{row,column,base,height,width,1} : Spatial.PixelBlock}
  %Equal.sym(Array<F32> & Tile.Tile8,Spatial.block(Tree.pack(F32,input),row,column,base,height,width,1,0,0,0,0,0),
    (Tree.pack(F32,input),Samples.pure_vector(input,context)),Samples.native_vector_matches_tabulation(input,context)) :
    {Kernel.filled(Tree.pack(Tile.Tile8,tiles),index,row,column,first,base,plane,height,width,shifts,_) == represent(fill_shift(state)) : Kernel.Fill}
  Equal.cong(Array<Tile.Tile8>,Kernel.Fill,
    tiles => Kernel.Fill{Tree.pack(F32,input),tiles,(index + 1 : U32),row,(column + 1 : U32),first,base,plane,height,width,shifts},
    Array.set(Tile.Tile8,Tree.pack(Tile.Tile8,tiles),index,Samples.pure_vector(input,context)),
    Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,tiles,index,Samples.pure_vector(input,context))),
    Storage.write(Tile.Tile8,tiles,index,Samples.pure_vector(input,context)))

def next_channel(state: Fill) -> Fill:
  Fill{input,tiles,index,row,column,+first,+base,+plane,height,width,shifts} = state
  Fill{input,tiles,index,row,first,first,(base + plane : U32),plane,height,width,shifts}

def next_matches(+state: Fill) -> {Kernel.next_channel(represent(state)) == represent(next_channel(state)) : Kernel.Fill}:
  Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts} = state
  {==}

def fill_channel(state: Fill) -> Fill:
  Fill{input,tiles,index,row,column,first,base,plane,height,width,+shifts} = state
  next_channel(Loop.run(~Fill,~fill_shift,shifts,Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts}))

def channel_matches(+state: Fill) -> {Kernel.fill_channel(represent(state)) == represent(fill_channel(state)) : Kernel.Fill}:
  Fill{+input,+tiles,+index,+row,+column,+first,+base,+plane,+height,+width,+shifts} = state
  %Equal.sym(Kernel.Fill,Loop.run(~Kernel.Fill,~Kernel.fill_shift,shifts,represent(state)),
    represent(Loop.run(~Fill,~fill_shift,shifts,state)),
    Projection.iteration_refines_model(~Kernel.Fill,~Fill,~Kernel.fill_shift,~fill_shift,~represent,~shift_matches,shifts,state)) :
    {Kernel.next_channel(_) == represent(fill_channel(state)) : Kernel.Fill}
  next_matches(Loop.run(~Fill,~fill_shift,shifts,state))

type Owners is Data:
  Owners{input: Tree.Tree<F32>,tiles: Tree.Tree<Tile.Tile8>}

def owners(state: Fill) -> Owners:
  Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts} = state
  Owners{input,tiles}

def pack_owners(state: Owners) -> Array<F32> & Array<Tile.Tile8>:
  Owners{input,tiles} = state
  (Tree.pack(F32,input),Tree.pack(Tile.Tile8,tiles))

def finished_matches(+state: Fill) -> {Kernel.fill_finished(represent(state)) == pack_owners(owners(state)) : Array<F32> & Array<Tile.Tile8>}:
  Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts} = state
  {==}

def fill_group(index: U32,context: Kernel.Filling,state: Owners) -> Owners:
  Kernel.Filling{Kernel.Tiling{groups,+shifts,+outputs,+column_low,+kernel,height,+width,+chunk_groups,chunk_rows},+low,base,plane,+first,+source_rows,+group,+span} = context
  Owners{input,tiles} = state
  +index = index
  +column = ((index % span + group) * 8 + column_low - kernel : U32)
  owners(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),
    Fill{input,tiles,((index / span * chunk_groups + index % span) * (outputs * shifts + 1) : U32),(index / span + low - kernel - first : U32),column,column,
      base,plane,source_rows,width,U32.to_nat(shifts)}))

def group_matches(+index: U32,+context: Kernel.Filling,+state: Owners) ->
  {Kernel.fill_group(index,context,pack_owners(state)) == pack_owners(fill_group(index,context,state)) : Array<F32> & Array<Tile.Tile8>}:
  Kernel.Filling{Kernel.Tiling{groups,+shifts,+outputs,+column_low,+kernel,height,+width,+chunk_groups,chunk_rows},+low,+base,+plane,+first,+source_rows,+group,+span} = context
  Owners{+input,+tiles} = state
  +column = ((index % span + group) * 8 + column_low - kernel : U32)
  +initial = {Fill{input,tiles,((index / span * chunk_groups + index % span) * (outputs * shifts + 1) : U32),(index / span + low - kernel - first : U32),column,column,
    base,plane,source_rows,width,U32.to_nat(shifts)} : Fill}
  %Equal.sym(Kernel.Fill,Loop.run(~Kernel.Fill,~Kernel.fill_channel,U32.to_nat(outputs),represent(initial)),
    represent(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),initial)),
    Projection.iteration_refines_model(~Kernel.Fill,~Fill,~Kernel.fill_channel,~fill_channel,~represent,~channel_matches,U32.to_nat(outputs),initial)) :
    {Kernel.fill_finished(_) == pack_owners(fill_group(index,context,state)) : Array<F32> & Array<Tile.Tile8>}
  finished_matches(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),initial))

def groups_match(+count: Nat,+first: U32,+context: Kernel.Filling,+state: Owners) ->
  {Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~Kernel.Filling,~Kernel.fill_group,count,first,context,pack_owners(state)) ==
    pack_owners(Traversal.indexed_fold(~Owners,~Kernel.Filling,~fill_group,count,first,context,state)) : Array<F32> & Array<Tile.Tile8>}:
  Arrays.indexed_fold_refines_model(~(Array<F32> & Array<Tile.Tile8>),~Owners,~Kernel.Filling,~Kernel.fill_group,~fill_group,~pack_owners,~group_matches,count,first,context,state)

def offset(index: U32,context: Kernel.Table,offsets: Tree.Tree<U32>) -> Tree.Tree<U32>:
  Kernel.Table{+tap_rows,+tap_columns,+row_step,+shifts,+first_shift,+base} = context
  +index = index
  Tree.write(U32,offsets,(base + index : U32),(index / tap_columns % tap_rows * row_step + index / (tap_rows * tap_columns) * shifts + index % tap_columns + first_shift : U32))

def pack_offsets(offsets: Tree.Tree<U32>) -> Array<U32>:
  Tree.pack(U32,offsets)

def offset_matches(+index: U32,+context: Kernel.Table,+offsets: Tree.Tree<U32>) ->
  {Kernel.offset(index,context,pack_offsets(offsets)) == pack_offsets(offset(index,context,offsets)) : Array<U32>}:
  Kernel.Table{+tap_rows,+tap_columns,+row_step,+shifts,+first_shift,+base} = context
  Storage.write(U32,offsets,(base + index : U32),(index / tap_columns % tap_rows * row_step + index / (tap_rows * tap_columns) * shifts + index % tap_columns + first_shift : U32))

def offsets_match(+count: Nat,+first: U32,+context: Kernel.Table,+offsets: Tree.Tree<U32>) ->
  {Traversal.indexed_fold(~Array<U32>,~Kernel.Table,~Kernel.offset,count,first,context,pack_offsets(offsets)) ==
    pack_offsets(Traversal.indexed_fold(~Tree.Tree<U32>,~Kernel.Table,~offset,count,first,context,offsets)) : Array<U32>}:
  Arrays.indexed_fold_refines_model(~Array<U32>,~Tree.Tree<U32>,~Kernel.Table,~Kernel.offset,~offset,~pack_offsets,~offset_matches,count,first,context,offsets)

def input(state: Fill) -> Tree.Tree<F32>:
  Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts} = state
  input

def shift_keeps_input(+state: Fill) -> {input(fill_shift(state)) == input(state) : Tree.Tree<F32>}:
  Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts} = state
  {==}

def next_keeps_input(+state: Fill) -> {input(next_channel(state)) == input(state) : Tree.Tree<F32>}:
  Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts} = state
  {==}

def channel_keeps_input(+state: Fill) -> {input(fill_channel(state)) == input(state) : Tree.Tree<F32>}:
  Fill{+source,+tiles,+index,+row,+column,+first,+base,+plane,+height,+width,+shifts} = state
  Equal.trans(Tree.Tree<F32>,input(next_channel(Loop.run(~Fill,~fill_shift,shifts,state))),input(Loop.run(~Fill,~fill_shift,shifts,state)),source,
    next_keeps_input(Loop.run(~Fill,~fill_shift,shifts,state)),
    Projection.preserves_observation(~Fill,~Tree.Tree<F32>,~fill_shift,~input,~shift_keeps_input,shifts,state))

def source(state: Owners) -> Tree.Tree<F32>:
  Owners{input,tiles} = state
  input

def owners_source(+state: Fill) -> {source(owners(state)) == input(state) : Tree.Tree<F32>}:
  Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts} = state
  {==}

def group_keeps_source(+index: U32,+context: Kernel.Filling,+state: Owners) ->
  {source(fill_group(index,context,state)) == source(state) : Tree.Tree<F32>}:
  Kernel.Filling{Kernel.Tiling{groups,+shifts,+outputs,+column_low,+kernel,height,+width,+chunk_groups,chunk_rows},+low,+base,+plane,+first,+source_rows,+group,+span} = context
  Owners{+gradient,+tiles} = state
  +column = ((index % span + group) * 8 + column_low - kernel : U32)
  +initial = {Fill{gradient,tiles,((index / span * chunk_groups + index % span) * (outputs * shifts + 1) : U32),
    (index / span + low - kernel - first : U32),column,column,base,plane,source_rows,width,U32.to_nat(shifts)} : Fill}
  Equal.trans(Tree.Tree<F32>,source(owners(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),initial))),input(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),initial)),gradient,
    owners_source(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),initial)),
    Projection.preserves_observation(~Fill,~Tree.Tree<F32>,~fill_channel,~input,~channel_keeps_input,U32.to_nat(outputs),initial))

def kept_source(index: U32,context: Kernel.Filling,input: Tree.Tree<F32>) -> Tree.Tree<F32>:
  input

def kept_source_fixed(+index: U32,+context: Kernel.Filling,+input: Tree.Tree<F32>) -> {kept_source(index,context,input) == input : Tree.Tree<F32>}:
  {==}

def groups_keep_source(+count: Nat,+first: U32,+context: Kernel.Filling,+state: Owners) ->
  {source(Traversal.indexed_fold(~Owners,~Kernel.Filling,~fill_group,count,first,context,state)) == source(state) : Tree.Tree<F32>}:
  Equal.trans(Tree.Tree<F32>,source(Traversal.indexed_fold(~Owners,~Kernel.Filling,~fill_group,count,first,context,state)),
    Traversal.indexed_fold(~Tree.Tree<F32>,~Kernel.Filling,~kept_source,count,first,context,source(state)),source(state),
    Arrays.indexed_fold_observes(~Owners,~Tree.Tree<F32>,~Kernel.Filling,~fill_group,~source,~kept_source,~group_keeps_source,count,first,context,state),
    Arrays.indexed_fold_fixed(~Tree.Tree<F32>,~Kernel.Filling,~kept_source,~kept_source_fixed,count,first,context,source(state)))

type TileCopy is Data:
  TileCopy{source: Tree.Tree<Tile.Tile8>,target: Tree.Tree<Tile.Tile8>,from: U32,to: U32}

def pack_copy(state: TileCopy) -> Kernel.TileCopy:
  TileCopy{source,target,from,to} = state
  Kernel.TileCopy{Tree.pack(Tile.Tile8,source),Tree.pack(Tile.Tile8,target),from,to}

def tile_step(state: TileCopy) -> TileCopy:
  TileCopy{+source,target,+from,+to} = state
  TileCopy{source,Tree.write(Tile.Tile8,target,to,Tree.read(Tile.Tile8,source,from)),(from + 1 : U32),(to + 1 : U32)}

def tile_step_matches(+state: TileCopy) -> {Kernel.tile_step(pack_copy(state)) == pack_copy(tile_step(state)) : Kernel.TileCopy}:
  TileCopy{+source,+target,+from,+to} = state
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,source),from),
    (Tree.pack(Tile.Tile8,source),Tree.read(Tile.Tile8,source,from)),Storage.read(Tile.Tile8,source,from)) :
    {Kernel.tile_copied(Tree.pack(Tile.Tile8,target),from,to,_) == pack_copy(tile_step(state)) : Kernel.TileCopy}
  Equal.cong(Array<Tile.Tile8>,Kernel.TileCopy,
    target => Kernel.TileCopy{Tree.pack(Tile.Tile8,source),target,(from + 1 : U32),(to + 1 : U32)},
    Array.set(Tile.Tile8,Tree.pack(Tile.Tile8,target),to,Tree.read(Tile.Tile8,source,from)),
    Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,target,to,Tree.read(Tile.Tile8,source,from))),
    Storage.write(Tile.Tile8,target,to,Tree.read(Tile.Tile8,source,from)))

def tile_copy_matches(+count: Nat,+state: TileCopy) ->
  {Loop.run(~Kernel.TileCopy,~Kernel.tile_step,count,pack_copy(state)) == pack_copy(Loop.run(~TileCopy,~tile_step,count,state)) : Kernel.TileCopy}:
  Projection.iteration_refines_model(~Kernel.TileCopy,~TileCopy,~Kernel.tile_step,~tile_step,~pack_copy,~tile_step_matches,count,state)

def tile_target(state: TileCopy) -> Tree.Tree<Tile.Tile8>:
  TileCopy{source,target,from,to} = state
  target

def tile_target_matches(+state: TileCopy) -> {Kernel.tile_target(pack_copy(state)) == Tree.pack(Tile.Tile8,tile_target(state)) : Array<Tile.Tile8>}:
  TileCopy{source,target,from,to} = state
  {==}
