# The tile fill on storage-tree models: every tile it writes, and the source
# it gives back.
import Base
import ../kernel_input_gradient_fill.bend as Kernel
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
  Fill{input: Tree.Tree<Tile.Tile8>,tiles: Tree.Tree<Tile.Tile8>,index: U32,row: U32,column: U32,first: U32,base: U32,groups: U32,height: U32,width: U32,shifts: Nat,channel: U32}

def represent(state: Fill) -> Kernel.Fill:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} = state
  Kernel.Fill{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,tiles),index,row,column,first,base,groups,height,width,shifts,channel}

def fill_shift(~field: Tile.Tile8 -> F32,state: Fill) -> Fill:
  Fill{+input,tiles,+index,+row,+column,first,+base,+groups,+height,+width,shifts,+channel} = state
  Fill{input,Tree.write(Tile.Tile8,tiles,index,Samples.pure_vector(~Tile.Tile8,~field,input,Spatial.PixelBlock{row,column,(base + channel / 8 : U32),height,width,1,groups})),
    (index + 1 : U32),row,(column + 1 : U32),first,base,groups,height,width,shifts,channel}

def shift_matches(~field: Tile.Tile8 -> F32,+state: Fill) -> {Kernel.fill_shift(~field,represent(state)) == represent(fill_shift(~field,state)) : Kernel.Fill}:
  Fill{+input,+tiles,+index,+row,+column,+first,+base,+groups,+height,+width,+shifts,+channel} = state
  +context = {Spatial.PixelBlock{row,column,(base + channel / 8 : U32),height,width,1,groups} : Spatial.PixelBlock}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Spatial.block(~Tile.Tile8,~field,Tree.pack(Tile.Tile8,input),row,column,(base + channel / 8 : U32),height,width,1,0,0,0,0,0,groups),
    (Tree.pack(Tile.Tile8,input),Samples.pure_vector(~Tile.Tile8,~field,input,context)),Samples.native_vector_matches_tabulation(~Tile.Tile8,~field,input,context)) :
    {Kernel.filled(Tree.pack(Tile.Tile8,tiles),index,row,column,first,base,groups,height,width,shifts,channel,_) == represent(fill_shift(~field,state)) : Kernel.Fill}
  Equal.cong(Array<Tile.Tile8>,Kernel.Fill,
    tiles => Kernel.Fill{Tree.pack(Tile.Tile8,input),tiles,(index + 1 : U32),row,(column + 1 : U32),first,base,groups,height,width,shifts,channel},
    Array.set(Tile.Tile8,Tree.pack(Tile.Tile8,tiles),index,Samples.pure_vector(~Tile.Tile8,~field,input,context)),
    Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,tiles,index,Samples.pure_vector(~Tile.Tile8,~field,input,context))),
    Storage.write(Tile.Tile8,tiles,index,Samples.pure_vector(~Tile.Tile8,~field,input,context)))

def next_channel(state: Fill) -> Fill:
  Fill{input,tiles,index,row,column,+first,+base,+groups,height,width,shifts,channel} = state
  Fill{input,tiles,index,row,first,first,base,groups,height,width,shifts,(channel + 1 : U32)}

def next_matches(+state: Fill) -> {Kernel.next_channel(represent(state)) == represent(next_channel(state)) : Kernel.Fill}:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} = state
  {==}

def fill_lane(~field: Tile.Tile8 -> F32,state: Fill) -> Fill:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,+shifts,+channel} = state
  next_channel(Loop.run(~Fill,~fill_shift(~field),shifts,Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel}))

def lane_matches(~field: Tile.Tile8 -> F32,+state: Fill) -> {Kernel.fill_lane(~field,represent(state)) == represent(fill_lane(~field,state)) : Kernel.Fill}:
  Fill{+input,+tiles,+index,+row,+column,+first,+base,+groups,+height,+width,+shifts,+channel} = state
  %Equal.sym(Kernel.Fill,Loop.run(~Kernel.Fill,~Kernel.fill_shift(~field),shifts,represent(state)),
    represent(Loop.run(~Fill,~fill_shift(~field),shifts,state)),
    Projection.iteration_refines_model(~Kernel.Fill,~Fill,~Kernel.fill_shift(~field),~fill_shift(~field),~represent,~shift_matches(~field),shifts,state)) :
    {Kernel.next_channel(_) == represent(fill_lane(~field,state)) : Kernel.Fill}
  next_matches(Loop.run(~Fill,~fill_shift(~field),shifts,state))

def fill_selected(half: Bool,low: Bool,first: Bool,state: Fill) -> Fill:
  match half low first:
    case True{} True{} True{}: fill_lane(~Tile.lane0,state)
    case True{} True{} False{}: fill_lane(~Tile.lane1,state)
    case True{} False{} True{}: fill_lane(~Tile.lane2,state)
    case True{} False{} False{}: fill_lane(~Tile.lane3,state)
    case False{} True{} True{}: fill_lane(~Tile.lane4,state)
    case False{} True{} False{}: fill_lane(~Tile.lane5,state)
    case False{} False{} True{}: fill_lane(~Tile.lane6,state)
    case False{} False{} False{}: fill_lane(~Tile.lane7,state)

def selected_matches(half: Bool,low: Bool,first: Bool,+state: Fill) -> {Kernel.fill_selected(half,low,first,represent(state)) == represent(fill_selected(half,low,first,state)) : Kernel.Fill}:
  match half low first:
    case True{} True{} True{}: lane_matches(~Tile.lane0,state)
    case True{} True{} False{}: lane_matches(~Tile.lane1,state)
    case True{} False{} True{}: lane_matches(~Tile.lane2,state)
    case True{} False{} False{}: lane_matches(~Tile.lane3,state)
    case False{} True{} True{}: lane_matches(~Tile.lane4,state)
    case False{} True{} False{}: lane_matches(~Tile.lane5,state)
    case False{} False{} True{}: lane_matches(~Tile.lane6,state)
    case False{} False{} False{}: lane_matches(~Tile.lane7,state)

def fill_channel(state: Fill) -> Fill:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,+channel} = state
  +lane = (channel % 8 : U32)
  fill_selected(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel})

def channel_matches(+state: Fill) -> {Kernel.fill_channel(represent(state)) == represent(fill_channel(state)) : Kernel.Fill}:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,+channel} = state
  selected_matches(U32.is_lt((channel % 8 : U32),4),U32.is_lt((channel % 8 % 4 : U32),2),U32.is_zero((channel % 8 % 2 : U32)),state)

type Owners is Data:
  Owners{input: Tree.Tree<Tile.Tile8>,tiles: Tree.Tree<Tile.Tile8>}

def owners(state: Fill) -> Owners:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} = state
  Owners{input,tiles}

def pack_owners(state: Owners) -> Array<Tile.Tile8> & Array<Tile.Tile8>:
  Owners{input,tiles} = state
  (Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,tiles))

def finished_matches(+state: Fill)
 -> {Kernel.fill_finished(represent(state)) == pack_owners(owners(state)) : Array<Tile.Tile8> & Array<Tile.Tile8>}:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} = state
  {==}

def fill_group(index: U32,context: Kernel.Filling,state: Owners) -> Owners:
  Kernel.Filling{+shifts,+outputs,+column_low,+kernel,+width,+chunk_groups,+low,base,groups,+first,+source_rows,+group,+span} = context
  Owners{input,tiles} = state
  +index = index
  +column = ((index % span + group) * 8 + column_low - kernel : U32)
  owners(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),
    Fill{input,tiles,((index / span * chunk_groups + index % span) * (outputs * shifts + 1) : U32),(index / span + low - kernel - first : U32),column,column,
      base,groups,source_rows,width,U32.to_nat(shifts),0}))

def group_matches(+index: U32,+context: Kernel.Filling,+state: Owners) ->
  {Kernel.fill_group(index,context,pack_owners(state)) == pack_owners(fill_group(index,context,state)) : Array<Tile.Tile8> & Array<Tile.Tile8>}:
  Kernel.Filling{+shifts,+outputs,+column_low,+kernel,+width,+chunk_groups,+low,+base,+groups,+first,+source_rows,+group,+span} = context
  Owners{+input,+tiles} = state
  +column = ((index % span + group) * 8 + column_low - kernel : U32)
  +initial = {Fill{input,tiles,((index / span * chunk_groups + index % span) * (outputs * shifts + 1) : U32),(index / span + low - kernel - first : U32),column,column,
    base,groups,source_rows,width,U32.to_nat(shifts),0} : Fill}
  %Equal.sym(Kernel.Fill,Loop.run(~Kernel.Fill,~Kernel.fill_channel,U32.to_nat(outputs),represent(initial)),
    represent(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),initial)),
    Projection.iteration_refines_model(~Kernel.Fill,~Fill,~Kernel.fill_channel,~fill_channel,~represent,~channel_matches,U32.to_nat(outputs),initial)) :
    {Kernel.fill_finished(_) == pack_owners(fill_group(index,context,state)) : Array<Tile.Tile8> & Array<Tile.Tile8>}
  finished_matches(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),initial))

def groups_match(+count: Nat,+first: U32,+context: Kernel.Filling,+state: Owners) ->
  {Traversal.indexed_fold(~(Array<Tile.Tile8> & Array<Tile.Tile8>),~Kernel.Filling,~Kernel.fill_group,count,first,context,pack_owners(state)) ==
    pack_owners(Traversal.indexed_fold(~Owners,~Kernel.Filling,~fill_group,count,first,context,state)) : Array<Tile.Tile8> & Array<Tile.Tile8>}:
  Arrays.indexed_fold_refines_model(~(Array<Tile.Tile8> & Array<Tile.Tile8>),~Owners,~Kernel.Filling,~Kernel.fill_group,~fill_group,~pack_owners,~group_matches,count,first,context,state)

def input(state: Fill) -> Tree.Tree<Tile.Tile8>:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} = state
  input

def shift_keeps_input(~field: Tile.Tile8 -> F32,+state: Fill) -> {input(fill_shift(~field,state)) == input(state) : Tree.Tree<Tile.Tile8>}:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} = state
  {==}

def next_keeps_input(+state: Fill) -> {input(next_channel(state)) == input(state) : Tree.Tree<Tile.Tile8>}:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} = state
  {==}

def lane_keeps_input(~field: Tile.Tile8 -> F32,+state: Fill) -> {input(fill_lane(~field,state)) == input(state) : Tree.Tree<Tile.Tile8>}:
  Fill{+source,+tiles,+index,+row,+column,+first,+base,+groups,+height,+width,+shifts,+channel} = state
  Equal.trans(Tree.Tree<Tile.Tile8>,input(next_channel(Loop.run(~Fill,~fill_shift(~field),shifts,state))),input(Loop.run(~Fill,~fill_shift(~field),shifts,state)),source,
    next_keeps_input(Loop.run(~Fill,~fill_shift(~field),shifts,state)),
    Projection.preserves_observation(~Fill,~Tree.Tree<Tile.Tile8>,~fill_shift(~field),~input,~shift_keeps_input(~field),shifts,state))

def selected_keeps_input(half: Bool,low: Bool,first: Bool,+state: Fill) -> {input(fill_selected(half,low,first,state)) == input(state) : Tree.Tree<Tile.Tile8>}:
  match half low first:
    case True{} True{} True{}: lane_keeps_input(~Tile.lane0,state)
    case True{} True{} False{}: lane_keeps_input(~Tile.lane1,state)
    case True{} False{} True{}: lane_keeps_input(~Tile.lane2,state)
    case True{} False{} False{}: lane_keeps_input(~Tile.lane3,state)
    case False{} True{} True{}: lane_keeps_input(~Tile.lane4,state)
    case False{} True{} False{}: lane_keeps_input(~Tile.lane5,state)
    case False{} False{} True{}: lane_keeps_input(~Tile.lane6,state)
    case False{} False{} False{}: lane_keeps_input(~Tile.lane7,state)

def channel_keeps_input(+state: Fill) -> {input(fill_channel(state)) == input(state) : Tree.Tree<Tile.Tile8>}:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,+channel} = state
  selected_keeps_input(U32.is_lt((channel % 8 : U32),4),U32.is_lt((channel % 8 % 4 : U32),2),U32.is_zero((channel % 8 % 2 : U32)),state)

def source(state: Owners) -> Tree.Tree<Tile.Tile8>:
  Owners{input,tiles} = state
  input

def tiles(state: Owners) -> Tree.Tree<Tile.Tile8>:
  Owners{input,tiles} = state
  tiles

def owners_source(+state: Fill) -> {source(owners(state)) == input(state) : Tree.Tree<Tile.Tile8>}:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} = state
  {==}

def group_keeps_source(+index: U32,+context: Kernel.Filling,+state: Owners) ->
  {source(fill_group(index,context,state)) == source(state) : Tree.Tree<Tile.Tile8>}:
  Kernel.Filling{+shifts,+outputs,+column_low,+kernel,+width,+chunk_groups,+low,+base,+groups,+first,+source_rows,+group,+span} = context
  Owners{+gradient,+tiles} = state
  +column = ((index % span + group) * 8 + column_low - kernel : U32)
  +initial = {Fill{gradient,tiles,((index / span * chunk_groups + index % span) * (outputs * shifts + 1) : U32),
    (index / span + low - kernel - first : U32),column,column,base,groups,source_rows,width,U32.to_nat(shifts),0} : Fill}
  Equal.trans(Tree.Tree<Tile.Tile8>,source(owners(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),initial))),input(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),initial)),gradient,
    owners_source(Loop.run(~Fill,~fill_channel,U32.to_nat(outputs),initial)),
    Projection.preserves_observation(~Fill,~Tree.Tree<Tile.Tile8>,~fill_channel,~input,~channel_keeps_input,U32.to_nat(outputs),initial))

def kept_source(index: U32,context: Kernel.Filling,input: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  input

def kept_source_fixed(+index: U32,+context: Kernel.Filling,+input: Tree.Tree<Tile.Tile8>) -> {kept_source(index,context,input) == input : Tree.Tree<Tile.Tile8>}:
  {==}

def groups_keep_source(+count: Nat,+first: U32,+context: Kernel.Filling,+state: Owners) ->
  {source(Traversal.indexed_fold(~Owners,~Kernel.Filling,~fill_group,count,first,context,state)) == source(state) : Tree.Tree<Tile.Tile8>}:
  Equal.trans(Tree.Tree<Tile.Tile8>,source(Traversal.indexed_fold(~Owners,~Kernel.Filling,~fill_group,count,first,context,state)),
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Kernel.Filling,~kept_source,count,first,context,source(state)),source(state),
    Arrays.indexed_fold_observes(~Owners,~Tree.Tree<Tile.Tile8>,~Kernel.Filling,~fill_group,~source,~kept_source,~group_keeps_source,count,first,context,state),
    Arrays.indexed_fold_fixed(~Tree.Tree<Tile.Tile8>,~Kernel.Filling,~kept_source,~kept_source_fixed,count,first,context,source(state)))

# A run fills its groups in order from group 0 and gives the source back.
def run(call: Kernel.Call,state: Owners) -> Owners:
  Kernel.Call{count,filling} = call
  Traversal.indexed_fold(~Owners,~Kernel.Filling,~fill_group,count,0,filling,state)

def run_matches(+call: Kernel.Call,+state: Owners) -> {Kernel.run(call,Tree.pack(Tile.Tile8,source(state)),Tree.pack(Tile.Tile8,tiles(state))) == pack_owners(run(call,state)) : Array<Tile.Tile8> & Array<Tile.Tile8>}:
  Kernel.Call{+count,+filling} = call
  Owners{input,tiles} = state
  groups_match(count,0,filling,state)

def run_keeps_source(+call: Kernel.Call,+state: Owners) -> {source(run(call,state)) == source(state) : Tree.Tree<Tile.Tile8>}:
  Kernel.Call{+count,+filling} = call
  groups_keep_source(count,0,filling,state)

def packed_source(+state: Owners) -> {Pair.fst(Array<Tile.Tile8>,Array<Tile.Tile8>,pack_owners(state)) == Tree.pack(Tile.Tile8,source(state)) : Array<Tile.Tile8>}:
  Owners{input,tiles} = state
  {==}

def packed_tiles(+state: Owners) -> {Pair.snd(Array<Tile.Tile8>,Array<Tile.Tile8>,pack_owners(state)) == Tree.pack(Tile.Tile8,tiles(state)) : Array<Tile.Tile8>}:
  Owners{input,tiles} = state
  {==}
