# Placing the leaf outputs of one image: the walk of the kernel makes every
# piece of the partition the band of its rows over the batch, so its bands
# are the band model of the pieces and its gradient is the last piece's.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../tensor_band.bend as Bands
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_refinement.bend as StorageLaws
import ./tensor_band_model.bend as Model
import ./tensor_band_each.bend as Each
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_sweep_selected.bend as Selected
import ./input_gradient_tiled_partition.bend as Parts

# The gradient the walk carries, held as its tree.
def hold(gradient: Tree.Tree<F32>) -> Array<F32>:
  Tree.pack(F32,gradient)
def held_tree(-storage: Array<F32>,observation: StorageLaws.Observation<F32,storage>) -> Each.Held<Array<F32>,Tree.Tree<F32>,hold,storage>:
  StorageLaws.Observation{tree,packed,reflected} = observation
  Each.Held{tree,packed}
def held(storage: Array<F32>) -> Each.Held<Array<F32>,Tree.Tree<F32>,hold,storage>:
  held_tree(storage,StorageLaws.observe(F32,storage))
def see(storage: Array<F32>) -> Tree.Tree<F32>:
  Tree.reflect(F32,storage)
def start(start: U32) -> U32:
  start

# The bands of the pieces of a walk over the rows of one image starting at start.
def bands(+start: U32,tree: Partition.Tree<&2,Execution.Part,Unit>) -> Model.Content<F32>:
  Each.shaped_bands(~Execution.Part,~F32,~Selected.part_output,start,Model.shaped_parts(~Execution.Part,tree))

# One piece: its output is the band of its rows, its gradient replaces the one carried.
def rule(+context: U32,+first: U32,+rows: U32,+part: Execution.Part,+kept: Tree.Tree<F32>) ->
  {Each.walk_seen(~F32,~Array<F32>,~Tree.Tree<F32>,~see,Kernel.placed(context,first,rows,Execution.represent_part(part),hold(kept))) ==
    (Model.RowBand{Buffer.content(~F32,Selected.part_output(part)),(start(context) + first : U32),rows},
      Model.replace(~Execution.Part,~Tree.Tree<F32>,~Execution.gradient,~U32,context,first,rows,part,see(hold(kept)))) : Model.Content<F32> & Tree.Tree<F32>}:
  Execution.Part{+gradient,+output} = part
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,output)),Buffer.content(~F32,output),Buffer.packed_contents(~F32,output)) :
    {(Model.RowBand{_,(context + first : U32),rows},see(Tree.pack(F32,gradient))) == (Model.RowBand{Buffer.content(~F32,output),(context + first : U32),rows},gradient) : Model.Content<F32> & Tree.Tree<F32>}
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,gradient)),gradient,StorageLaws.reflect_packed(F32,gradient)) :
    {(Model.RowBand{Buffer.content(~F32,output),(context + first : U32),rows},_) == (Model.RowBand{Buffer.content(~F32,output),(context + first : U32),rows},gradient) : Model.Content<F32> & Tree.Tree<F32>}
  {==}

# The walk of the represented pieces: the bands of the pieces, and the
# gradient of the last piece.
def placed(+tree: Partition.Tree<&2,Execution.Part,Unit>,+first: U32,+gradient: Tree.Tree<F32>) ->
  {Each.walk_seen(~F32,~Array<F32>,~Tree.Tree<F32>,~see,Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(tree),hold(gradient))) ==
    (bands(first,tree),Parts.last(tree)) : Model.Content<F32> & Tree.Tree<F32>}:
  %Equal.sym(Partition.Tree<&1,Kernel.Part,Unit>,Parts.represented(tree),Model.shaped_pack(~Execution.Part,~Kernel.Part,~Execution.represent_part,Model.shaped_parts(~Execution.Part,tree)),
    Model.shaped_carried(~Execution.Part,~Kernel.Part,~Execution.represent_part,tree)) :
    {Each.walk_seen(~F32,~Array<F32>,~Tree.Tree<F32>,~see,Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,_,hold(gradient))) ==
      (bands(first,tree),Parts.last(tree)) : Model.Content<F32> & Tree.Tree<F32>}
  %Model.replaced(~Execution.Part,~Tree.Tree<F32>,~Execution.gradient,~U32,first,Model.shaped_parts(~Execution.Part,tree),see(hold(gradient))) :
    {Each.walk_seen(~F32,~Array<F32>,~Tree.Tree<F32>,~see,Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,
      Model.shaped_pack(~Execution.Part,~Kernel.Part,~Execution.represent_part,Model.shaped_parts(~Execution.Part,tree)),hold(gradient))) ==
      (bands(first,tree),_) : Model.Content<F32> & Tree.Tree<F32>}
  Each.shaped_walked(~Execution.Part,~Kernel.Part,~Execution.represent_part,~F32,~Array<F32>,~U32,~Tree.Tree<F32>,~hold,~held,~Tree.Tree<F32>,~see,
    ~Kernel.placed,~start,~Selected.part_output,~Model.replace(~Execution.Part,~Tree.Tree<F32>,~Execution.gradient,~U32),~rule,
    Model.shaped_parts(~Execution.Part,tree),first,gradient)
