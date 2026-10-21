# The block loop on storage trees, connected to the native array loop.
import Base
import ./storage_certificate.bend as Certificate
import ./storage_buffer_model.bend as Buffer
import ./kernel_pipeline_buffer_shape.bend as BufferShape
import ./kernel_pipeline_activation.bend as ActivationProofs
import ../storage_buffer.bend as Storage
import ./kernel_pipeline_projection.bend as Projection
import ./kernel_spatial_packing_refinement.bend as PackingStorage
import ../kernel_pipeline.bend as Pipeline
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../kernel_spatial_packing.bend as Packing
import ../kernel_window.bend as Window
import ../tensor.bend as Tensors
import ../math_activation.bend as Activations
import ../traversal_array.bend as Traversal
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_writes.bend as GemmWrites
import ./kernel_gemm_readback.bend as GemmReadback
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./write_program.bend as Writes

import ../ordered_matrix.bend as OrderedMatrix
import ./ordered_ranges.bend as Ranges
import ./kernel_spatial_packing_samples.bend as Samples
import ./kernel_packing_window_coordinates.bend as Coordinates
import ./kernel_tile_refinement.bend as TileLaws

type WindowContext is Data:
  WindowContext{stencil: Packing.Stencil,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>}

def context_stencil(context: WindowContext) -> Packing.Stencil:
  WindowContext{stencil,input,weights} = context
  stencil

def window_inner(stencil: Packing.Stencil) -> U32:
  Coordinates.stencil_inner(stencil)

def window_sample(context: WindowContext,position: U32,q: Nat) -> F32:
  WindowContext{stencil,input,weights} = context
  Samples.window_value(input,stencil,position,U32.from_nat(q))

def window_weight(context: WindowContext,channel: Nat,q: Nat) -> F32:
  WindowContext{stencil,input,weights} = context
  +channel = channel
  Tile.element_at(TileLaws.lane_of(Nat.mod(channel,8n)),Tree.read(Tile.Tile8,weights,U32.from_nat(Nat.add(Nat.mul(Nat.div(channel,8n),U32.to_nat(window_inner(stencil))),q))))

def window_dot(+context: WindowContext,position: U32,channel: Nat) -> F32:
  (OrderedMatrix.dot(~Nat,~Nat,~U32,~WindowContext,~window_sample,~window_weight,Ranges.range(U32.to_nat(window_inner(context_stencil(context))),0n),channel,position,context,0.0) + 0.0 : F32)

type State is Data:
  State{input: Tree.Tree<F32>,tiles: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Tree.Tree<F32>}

def represent(state: State) -> Projection.State:
  State{input,tiles,weights,bias,output} = state
  Projection.State{Tree.pack(F32,input),Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output)}

def kept(input: Tree.Tree<F32>,state: GemmLaws.Buffers) -> State:
  GemmLaws.Buffers{tiles,weights,bias,output} = state
  State{input,tiles,weights,bias,Writes.evaluate(F32,output)}

def kept_matches(+input: Tree.Tree<F32>,+state: GemmLaws.Buffers) ->
  {Projection.kept(Tree.pack(F32,input),GemmLaws.represent(state)) == represent(kept(input,state)) : Projection.State}:
  GemmLaws.Buffers{tiles,weights,bias,output} = state
  {==}

def packed(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Tree.Tree<F32>,pair: PackingStorage.Packed) -> State:
  PackingStorage.Packed{input,tiles} = pair
  kept(input,Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step(~True{}),U32.to_nat(Gemm.job_count(rows,span)),0,
    Gemm.Layout{rows,span,inner,start,positions},GemmLaws.Buffers{tiles,weights,bias,Writes.Initial{output}}))

def packed_matches(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+pair: PackingStorage.Packed) ->
  {Projection.packed(rows,positions,inner,start,span,Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),PackingStorage.pack_pair(pair)) == represent(packed(rows,positions,inner,start,span,weights,bias,output,pair)) : Projection.State}:
  PackingStorage.Packed{+input,+tiles} = pair
  +state = Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step(~True{}),U32.to_nat(Gemm.job_count(rows,span)),0,
    Gemm.Layout{rows,span,inner,start,positions},GemmLaws.Buffers{tiles,weights,bias,Writes.Initial{output}})
  %Equal.sym(Gemm.Buffers,Gemm.run_keeping(~True{},rows,Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),span,inner,start,positions),GemmLaws.represent(state),
    GemmLaws.keeping_matches(~True{},rows,tiles,weights,bias,output,span,inner,start,positions)) :
    {Projection.kept(Tree.pack(F32,input),_) == represent(kept(input,state)) : Projection.State}
  kept_matches(input,state)

def kept_visible(input: Tree.Tree<F32>,state: GemmWrites.Visible) -> State:
  GemmWrites.Visible{tiles,weights,bias,output} = state
  State{input,tiles,weights,bias,output}

def kept_observation(+input: Tree.Tree<F32>,+state: GemmLaws.Buffers) ->
  {kept(input,state) == kept_visible(input,GemmWrites.visible(state)) : State}:
  GemmLaws.Buffers{tiles,weights,bias,output} = state
  {==}

# Each packed block returns its tile, weight and bias owners and applies its
# ordered GEMM writes directly to the full output tree.
def packed_writes(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,
  +weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+input: Tree.Tree<F32>,+tiles: Tree.Tree<Tile.Tile8>) ->
  {packed(rows,positions,inner,start,span,weights,bias,output,PackingStorage.Packed{input,tiles}) ==
    State{input,tiles,weights,bias,Writes.pure(F32,GemmWrites.all_writes(~True{},GemmWrites.job_count(rows,span),0,Gemm.Layout{rows,span,inner,start,positions},tiles,weights,bias),output)} : State}:
  +layout = {Gemm.Layout{rows,span,inner,start,positions} : Gemm.Layout}
  +count = GemmWrites.job_count(rows,span)
  +state = Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step(~True{}),count,0,layout,GemmLaws.Buffers{tiles,weights,bias,Writes.Initial{output}})
  %Equal.sym(State,kept(input,state),kept_visible(input,GemmWrites.visible(state)),kept_observation(input,state)) :
    {_ == State{input,tiles,weights,bias,Writes.pure(F32,GemmWrites.all_writes(~True{},count,0,layout,tiles,weights,bias),output)} : State}
  Equal.cong(GemmWrites.Visible,State,state => kept_visible(input,state),GemmWrites.visible(state),
    GemmWrites.Visible{tiles,weights,bias,Writes.pure(F32,GemmWrites.all_writes(~True{},count,0,layout,tiles,weights,bias),output)},
    GemmWrites.fold_visible(~True{},count,0,layout,tiles,weights,bias,Writes.Initial{output}))

def packed_result(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,
  +weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+input: Tree.Tree<F32>,+tiles: Tree.Tree<Tile.Tile8>) ->
  {packed(rows,positions,inner,start,span,weights,bias,output,PackingStorage.Packed{input,tiles}) ==
    State{input,tiles,weights,bias,GemmLaws.run_at(~True{},rows,tiles,weights,bias,output,span,inner,start,positions)} : State}:
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run_at(~True{},rows,tiles,weights,bias,output,span,inner,start,positions),
    Writes.pure(F32,GemmWrites.all_writes(~True{},GemmWrites.job_count(rows,span),0,Gemm.Layout{rows,span,inner,start,positions},tiles,weights,bias),output),
    GemmWrites.run_at_writes(~True{},rows,tiles,weights,bias,output,span,inner,start,positions)) :
    {packed(rows,positions,inner,start,span,weights,bias,output,PackingStorage.Packed{input,tiles}) == State{input,tiles,weights,bias,_} : State}
  packed_writes(rows,positions,inner,start,span,weights,bias,output,input,tiles)

def step_window(work: Pipeline.WindowWork,+pack_start: U32,+start: U32,+span: U32,state: State) -> State:
  Pipeline.WindowWork{rows,positions,+window} = work
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner} = window
  State{input,tiles,weights,bias,output} = state
  packed(rows,positions,inner,start,span,weights,bias,output,PackingStorage.packed_window(window,U32.to_nat((span / 8 * inner : U32)),pack_start,input,tiles))

def step(work: Pipeline.Work,pack_start: U32,start: U32,span: U32,state: State) -> State:
  step_window(Pipeline.square_work(work),pack_start,start,span,state)

def window_step_matches(+work: Pipeline.WindowWork,+pack_start: U32,+start: U32,+span: U32,+state: State) ->
  {Projection.step_window(work,pack_start,start,span,represent(state)) == represent(step_window(work,pack_start,start,span,state)) : Projection.State}:
  Pipeline.WindowWork{+rows,+positions,+window} = work
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner} = window
  State{+input,+tiles,+weights,+bias,+output} = state
  +count = U32.to_nat((span / 8 * inner : U32))
  +pair = PackingStorage.packed_window(window,count,pack_start,input,tiles)
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,Packing.pack_window(window,count,pack_start,Tree.pack(F32,input),Tree.pack(Tile.Tile8,tiles)),PackingStorage.pack_pair(pair),PackingStorage.packed_window_matches(window,count,pack_start,input,tiles)) :
    {Projection.packed(rows,positions,inner,start,span,Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),_) == represent(packed(rows,positions,inner,start,span,weights,bias,output,pair)) : Projection.State}
  packed_matches(rows,positions,inner,start,span,weights,bias,output,pair)

def blocks_window(remaining: Nat,+work: Pipeline.WindowWork,+pack_start: U32,+start: U32,+padded: U32,+span: U32,state: State) -> State:
  match remaining:
    case 0n: state
    case 1n+rest:
      blocks_window(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,step_window(work,pack_start,start,U32.min(span,(padded - start : U32)),state))

def blocks(remaining: Nat,work: Pipeline.Work,pack_start: U32,start: U32,padded: U32,span: U32,state: State) -> State:
  blocks_window(remaining,Pipeline.square_work(work),pack_start,start,padded,span,state)

def window_blocks_match(+remaining: Nat,+work: Pipeline.WindowWork,+pack_start: U32,+start: U32,+padded: U32,+span: U32,+state: State) ->
  {Projection.blocks_window(remaining,work,pack_start,start,padded,span,represent(state)) == represent(blocks_window(remaining,work,pack_start,start,padded,span,state)) : Projection.State}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +next = step_window(work,pack_start,start,U32.min(span,(padded - start : U32)),state)
      %Equal.sym(Projection.State,Projection.step_window(work,pack_start,start,U32.min(span,(padded - start : U32)),represent(state)),represent(next),window_step_matches(work,pack_start,start,U32.min(span,(padded - start : U32)),state)) :
        {Projection.blocks_window(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,_) == represent(blocks_window(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,next)) : Projection.State}
      window_blocks_match(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,next)

def finish(~custom: F32 -> F32,activation: Activations.Activation,count: U32,state: State) -> Tree.Tree<F32>:
  State{input,tiles,weights,bias,output} = state
  ActivationProofs.activation_tree(~custom,activation,count,output)

def finish_matches(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,+state: State) ->
  {Projection.finish(~custom,activation,count,represent(state)) == Tree.pack(F32,finish(~custom,activation,count,state)) : Array<F32>}:
  State{input,tiles,weights,bias,+output} = state
  ActivationProofs.activation_model(~custom,activation,count,output)

def allocated(~Element: Data,fill: Element,count: U32) -> Tree.Tree<Element>:
  Tree.fresh(Element,Tensors.depth_for(count),fill)

def allocation_matches(~Element: Data,+fill: Element,+count: U32) ->
  {Projection.allocated(~Element,fill,count) == Tree.pack(Element,allocated(~Element,fill,count)) : Array<Element>}:
  %Equal.sym(Storage.Buffer<Element>,Tensors.buffer(~Element,fill,count),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),ActivationProofs.allocated(~Element,fill,count)) :
    {ActivationProofs.storage(~Element,_) == Tree.pack(Element,allocated(~Element,fill,count)) : Array<Element>}
  StorageLaws.fresh(Element,Tensors.depth_for(count),fill)

def initial_window_with_target(+target: U32,work: Pipeline.WindowWork,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> State:
  Pipeline.WindowWork{rows,+positions,Packing.Stencil{base,h,w,kr,kc,stride,ro,co,ow,inner}} = work
  State{input,allocated(~Tile.Tile8,Tile.filled(0.0),Pipeline.workspace_tiles(target,inner,positions)),weights,bias,allocated(~F32,0.0,(rows * positions : U32))}

def run_window_with_budget(+target: U32,+work: Pipeline.WindowWork,pack_start: U32,state: State) -> State:
  Pipeline.WindowWork{rows,+positions,Packing.Stencil{base,h,w,kr,kc,stride,ro,co,ow,+inner}} = work
  +padded = Window.padded(positions)
  +span = (8 * U32.max(1,(target / inner : U32)) : U32)
  blocks_window(U32.to_nat(((padded + span - 1) / span : U32)),work,pack_start,0,padded,span,state)

def run_window_with(+work: Pipeline.WindowWork,pack_start: U32,state: State) -> State:
  run_window_with_budget(Pipeline.block_tiles(),work,pack_start,state)

def run_window_with_target(~custom: F32 -> F32,+target: U32,+activation: Activations.Activation,+work: Pipeline.WindowWork,+pack_start: U32,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  Pipeline.WindowWork{+rows,+positions,Packing.Stencil{base,h,w,kr,kc,stride,ro,co,ow,+inner}} = work
  finish(~custom,activation,(rows * positions : U32),run_window_with_budget(target,work,pack_start,initial_window_with_target(target,work,input,weights,bias)))

def run_with_target(~custom: F32 -> F32,+target: U32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  run_window_with_target(~custom,target,activation,Pipeline.square_work(work),pack_start,input,weights,bias)

def run_window(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.WindowWork,+pack_start: U32,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  run_window_with_target(~custom,Pipeline.block_tiles(),activation,work,pack_start,input,weights,bias)

def run(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>) -> Tree.Tree<F32>:
  run_window(~custom,activation,Pipeline.square_work(work),pack_start,input,weights,bias)

def window_run_matches(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.WindowWork,+pack_start: U32,+input: Tree.Tree<F32>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {Projection.run_window(~custom,activation,work,pack_start,Tree.pack(F32,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)) == Tree.pack(F32,run_window(~custom,activation,work,pack_start,input,weights,bias)) : Array<F32>}:
  Pipeline.WindowWork{+rows,+positions,Packing.Stencil{base,h,w,kr,kc,stride,ro,co,ow,+inner}} = work
  +padded = Window.padded(positions)
  +span = Pipeline.block_positions(inner)
  +remaining = U32.to_nat(((padded + span - 1) / span : U32))
  +state = initial_window_with_target(Pipeline.block_tiles(),work,input,weights,bias)
  %Equal.sym(Array<Tile.Tile8>,Projection.allocated(~Tile.Tile8,Tile.filled(0.0),Pipeline.workspace_tiles(Pipeline.block_tiles(),inner,positions)),Tree.pack(Tile.Tile8,allocated(~Tile.Tile8,Tile.filled(0.0),Pipeline.workspace_tiles(Pipeline.block_tiles(),inner,positions))),allocation_matches(~Tile.Tile8,Tile.filled(0.0),Pipeline.workspace_tiles(Pipeline.block_tiles(),inner,positions))) :
    {Projection.finish(~custom,activation,(rows * positions : U32),Projection.blocks_window(remaining,work,pack_start,0,padded,span,
      Projection.State{Tree.pack(F32,input),_,Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Projection.allocated(~F32,0.0,(rows * positions : U32))})) == Tree.pack(F32,run_window(~custom,activation,work,pack_start,input,weights,bias)) : Array<F32>}
  %Equal.sym(Array<F32>,Projection.allocated(~F32,0.0,(rows * positions : U32)),Tree.pack(F32,allocated(~F32,0.0,(rows * positions : U32))),allocation_matches(~F32,0.0,(rows * positions : U32))) :
    {Projection.finish(~custom,activation,(rows * positions : U32),Projection.blocks_window(remaining,work,pack_start,0,padded,span,
      Projection.State{Tree.pack(F32,input),Tree.pack(Tile.Tile8,allocated(~Tile.Tile8,Tile.filled(0.0),Pipeline.workspace_tiles(Pipeline.block_tiles(),inner,positions))),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),_})) == Tree.pack(F32,run_window(~custom,activation,work,pack_start,input,weights,bias)) : Array<F32>}
  %Equal.sym(Projection.State,Projection.blocks_window(remaining,work,pack_start,0,padded,span,represent(state)),represent(blocks_window(remaining,work,pack_start,0,padded,span,state)),window_blocks_match(remaining,work,pack_start,0,padded,span,state)) :
    {Projection.finish(~custom,activation,(rows * positions : U32),_) == Tree.pack(F32,run_window(~custom,activation,work,pack_start,input,weights,bias)) : Array<F32>}
  finish_matches(~custom,activation,(rows * positions : U32),blocks_window(remaining,work,pack_start,0,padded,span,state))

def run_matches(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,+input: Tree.Tree<F32>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {Projection.run(~custom,activation,work,pack_start,Tree.pack(F32,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias)) == Tree.pack(F32,run(~custom,activation,work,pack_start,input,weights,bias)) : Array<F32>}:
  window_run_matches(~custom,activation,Pipeline.square_work(work),pack_start,input,weights,bias)

# Array contents and the owner footprint jointly determine Buffer contents.
def contents_of_projection(buffer: Storage.Buffer<F32>,+tree: Tree.Tree<F32>,+length: Nat,
  +values: {ActivationProofs.storage(~F32,buffer) == Tree.pack(F32,tree) : Array<F32>},
  +size: {BufferShape.length(BufferShape.footprint(~F32,buffer)) == length : Nat}) ->
  {Buffer.contents(~F32,buffer) == Buffer.Content{tree,length} : Buffer.Content<F32>}:
  Storage.Buffer{array,+stored_length,Certificate.Certificate{depth,equation}} = buffer
  %Equal.sym(Array<F32>,array,Tree.pack(F32,tree),values) : {Buffer.Content{Tree.reflect(F32,_),stored_length} == Buffer.Content{tree,length} : Buffer.Content<F32>}
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,tree)),tree,StorageLaws.reflect_packed(F32,tree)) : {Buffer.Content{_,stored_length} == Buffer.Content{tree,length} : Buffer.Content<F32>}
  %Equal.sym(Nat,stored_length,length,size) : {Buffer.Content{tree,_} == Buffer.Content{tree,length} : Buffer.Content<F32>}
  {==}

def run_contents(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,
  +input: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Pipeline.blocked(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) ==
    Buffer.Content{run(~custom,activation,work,pack_start,Buffer.tree(~F32,input),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias)),U32.to_nat(BufferShape.output_size(work))} : Buffer.Content<F32>}:
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+weight_tree,+weight_length,+weight_certificate} = weights
  Buffer.Model{+bias_tree,+bias_length,+bias_certificate} = bias
  +input = {Buffer.Model{input_tree,input_length,input_certificate} : Buffer.Model<F32>}
  +weights = {Buffer.Model{weight_tree,weight_length,weight_certificate} : Buffer.Model<Tile.Tile8>}
  +bias = {Buffer.Model{bias_tree,bias_length,bias_certificate} : Buffer.Model<F32>}
  contents_of_projection(Pipeline.blocked(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)),
    run(~custom,activation,work,pack_start,input_tree,weight_tree,bias_tree),U32.to_nat(BufferShape.output_size(work)),
    Equal.trans(Array<F32>,ActivationProofs.storage(~F32,Pipeline.blocked(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))),
      Projection.run(~custom,activation,work,pack_start,Tree.pack(F32,input_tree),Tree.pack(Tile.Tile8,weight_tree),Tree.pack(F32,bias_tree)),
      Tree.pack(F32,run(~custom,activation,work,pack_start,input_tree,weight_tree,bias_tree)),
      Projection.run_projection(~custom,activation,work,pack_start,input,weights,bias),run_matches(~custom,activation,work,pack_start,input_tree,weight_tree,bias_tree)),
    Equal.cong(BufferShape.Footprint,Nat,value => BufferShape.length(value),BufferShape.footprint(~F32,Pipeline.blocked(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))),
      BufferShape.Footprint{Tensors.depth_for(BufferShape.output_size(work)),U32.to_nat(BufferShape.output_size(work))},BufferShape.run_footprint(~custom,activation,work,pack_start,input,weights,bias)))

def result(state: State) -> Tree.Tree<F32>:
  State{input,tiles,weights,bias,output} = state
  output

def packed_result_tree(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+pair: PackingStorage.Packed) ->
  {result(packed(rows,positions,inner,start,span,weights,bias,output,pair)) == GemmLaws.run_at(~True{},rows,PackingStorage.packed_output(pair),weights,bias,output,span,inner,start,positions) : Tree.Tree<F32>}:
  PackingStorage.Packed{+input,+tiles} = pair
  Equal.cong(State,Tree.Tree<F32>,state => result(state),packed(rows,positions,inner,start,span,weights,bias,output,pair),State{input,tiles,weights,bias,GemmLaws.run_at(~True{},rows,tiles,weights,bias,output,span,inner,start,positions)},packed_result(rows,positions,inner,start,span,weights,bias,output,input,tiles))

def step_gemm_query(+rows: U32,+span: U32,+window: Packing.Stencil,+pack_start: U32,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,channel: Nat,column: Nat,+groups: Nat,+limit: U32,
  limit_fits: {U32.is_le(limit,2147483648) == True{} : Bool},columns_exact: {U32.to_nat(span) == Nat.mul(groups,8n) : Nat},rows_room: {Nat.is_le(Nat.add(U32.to_nat(rows),3n),U32.to_nat(limit)) == True{} : Bool},area_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(span)),U32.to_nat(limit)) == True{} : Bool},
  channel_inside: {Nat.is_lt(channel,U32.to_nat(rows)) == True{} : Bool},column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool}) -> GemmReadback.Query:
  Packing.Stencil{base,h,w,kr,kc,stride,ro,co,ow,+inner} = window
  GemmReadback.Query{rows,span,inner,groups,limit,PackingStorage.packed_output(PackingStorage.packed_window(window,U32.to_nat((span / 8 * inner : U32)),pack_start,input,workspace)),weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside}

def window_step_readback(+rows: U32,+positions: U32,+window: Packing.Stencil,+pack_start: U32,+start: U32,+span: U32,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+channel: Nat,+column: Nat,+groups: Nat,+limit: U32,
  +limit_fits: {U32.is_le(limit,2147483648) == True{} : Bool},+columns_exact: {U32.to_nat(span) == Nat.mul(groups,8n) : Nat},+rows_room: {Nat.is_le(Nat.add(U32.to_nat(rows),3n),U32.to_nat(limit)) == True{} : Bool},+area_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(span)),U32.to_nat(limit)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(rows)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},+depth: Nat,balanced: Shape.Balanced(F32,depth,output),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(positions)),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},positions_room: {Nat.is_le(U32.to_nat(positions),U32.to_nat(limit)) == True{} : Bool},point_inside: {Nat.is_lt(Nat.add(U32.to_nat(start),column),U32.to_nat(positions)) == True{} : Bool}) ->
  {Tree.read(F32,result(step_window(Pipeline.WindowWork{rows,positions,window},pack_start,start,span,State{input,workspace,weights,bias,output})),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),Nat.add(U32.to_nat(start),column)))) ==
    GemmReadback.wanted(~True{},step_gemm_query(rows,span,window,pack_start,input,workspace,weights,bias,channel,column,groups,limit,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside)) : F32}:
  Packing.Stencil{base,h,w,kr,kc,stride,ro,co,ow,+inner} = window
  +pair = PackingStorage.packed_window(window,U32.to_nat((span / 8 * inner : U32)),pack_start,input,workspace)
  +query = step_gemm_query(rows,span,window,pack_start,input,workspace,weights,bias,channel,column,groups,limit,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside)
  %Equal.sym(Tree.Tree<F32>,result(packed(rows,positions,inner,start,span,weights,bias,output,pair)),GemmLaws.run_at(~True{},rows,PackingStorage.packed_output(pair),weights,bias,output,span,inner,start,positions),packed_result_tree(rows,positions,inner,start,span,weights,bias,output,pair)) :
    {Tree.read(F32,_,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),Nat.add(U32.to_nat(start),column)))) == GemmReadback.wanted(~True{},query) : F32}
  GemmReadback.run_at_readback(~True{},query,start,positions,output,depth,balanced,depth_ok,output_room,end_room,positions_room,point_inside)

def window_step_preserves_before(+rows: U32,+positions: U32,+window: Packing.Stencil,+pack_start: U32,+start: U32,+span: U32,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+channel: Nat,+column: Nat,+observed_column: Nat,+groups: Nat,+limit: U32,
  +limit_fits: {U32.is_le(limit,2147483648) == True{} : Bool},+columns_exact: {U32.to_nat(span) == Nat.mul(groups,8n) : Nat},+rows_room: {Nat.is_le(Nat.add(U32.to_nat(rows),3n),U32.to_nat(limit)) == True{} : Bool},+area_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(span)),U32.to_nat(limit)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(rows)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},+depth: Nat,balanced: Shape.Balanced(F32,depth,output),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(positions)),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},positions_room: {Nat.is_le(U32.to_nat(positions),U32.to_nat(limit)) == True{} : Bool},point_inside: {Nat.is_lt(observed_column,U32.to_nat(positions)) == True{} : Bool},before: {Nat.is_lt(observed_column,U32.to_nat(start)) == True{} : Bool}) ->
  {Tree.read(F32,result(step_window(Pipeline.WindowWork{rows,positions,window},pack_start,start,span,State{input,workspace,weights,bias,output})),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),observed_column))) ==
    Tree.read(F32,output,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),observed_column))) : F32}:
  Packing.Stencil{base,h,w,kr,kc,stride,ro,co,ow,+inner} = window
  +pair = PackingStorage.packed_window(window,U32.to_nat((span / 8 * inner : U32)),pack_start,input,workspace)
  +query = step_gemm_query(rows,span,window,pack_start,input,workspace,weights,bias,channel,column,groups,limit,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside)
  %Equal.sym(Tree.Tree<F32>,result(packed(rows,positions,inner,start,span,weights,bias,output,pair)),GemmLaws.run_at(~True{},rows,PackingStorage.packed_output(pair),weights,bias,output,span,inner,start,positions),packed_result_tree(rows,positions,inner,start,span,weights,bias,output,pair)) :
    {Tree.read(F32,_,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),observed_column))) == Tree.read(F32,output,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),observed_column))) : F32}
  GemmReadback.run_at_preserves_before(~True{},query,start,positions,output,depth,observed_column,balanced,depth_ok,output_room,end_room,positions_room,point_inside,before)

def window_step_balance(+rows: U32,+positions: U32,+window: Packing.Stencil,+pack_start: U32,+start: U32,+span: U32,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+depth: Nat,balanced: Shape.Balanced(F32,depth,output)) ->
  Shape.Balanced(F32,depth,result(step_window(Pipeline.WindowWork{rows,positions,window},pack_start,start,span,State{input,workspace,weights,bias,output}))):
  Packing.Stencil{base,h,w,kr,kc,stride,ro,co,ow,+inner} = window
  +pair = PackingStorage.packed_window(window,U32.to_nat((span / 8 * inner : U32)),pack_start,input,workspace)
  +tiles = PackingStorage.packed_output(pair)
  +writes = GemmWrites.all_writes(~True{},GemmWrites.job_count(rows,span),0,Gemm.Layout{rows,span,inner,start,positions},tiles,weights,bias)
  %Equal.sym(Tree.Tree<F32>,result(packed(rows,positions,inner,start,span,weights,bias,output,pair)),GemmLaws.run_at(~True{},rows,tiles,weights,bias,output,span,inner,start,positions),packed_result_tree(rows,positions,inner,start,span,weights,bias,output,pair)) : Shape.Balanced(F32,depth,_)
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run_at(~True{},rows,tiles,weights,bias,output,span,inner,start,positions),Writes.pure(F32,writes,output),GemmWrites.run_at_writes(~True{},rows,tiles,weights,bias,output,span,inner,start,positions)) : Shape.Balanced(F32,depth,_)
  Writes.writes_preserve_balance(F32,writes,output,depth,balanced)
