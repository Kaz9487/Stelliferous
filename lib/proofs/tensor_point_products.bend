# Buffer contents and cell order of a fragment emitted by a pointwise product.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../kernel_tile.bend as Tile
import ../kernel_window_pipeline.bend as Pipeline
import ../storage_buffer.bend as Storage
import ../storage_clone.bend as Clone
import ../storage_copy.bend as Copy
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_move.bend as Move
import ../tensor_pieces.bend as Pieces
import ../tensor_point_products.bend as Products
import ../traversal_partition.bend as Partition
import ./kernel_gemm_point_buffer.bend as PointBuffer
import ./kernel_gemm_point_jobs.bend as PointJobs
import ./kernel_gemm_read_into.bend as ReadInto
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_extrema.bend as NatExtrema
import ./storage_allocation_refinement.bend as Allocation
import ./storage_buffer_model.bend as Buffer
import ./storage_clone_refinement.bend as CloneProofs
import ./storage_tree_model.bend as Tree
import ./tensor_clipped_pieces.bend as Clipped
import ./tensor_move_readback.bend as MoveReadback
import ./tensor_pieces_model.bend as Model
import ./u32_extrema.bend as Extrema
import ./write_program.bend as Writes

def Stage() -> Type:
  Storage.Buffer<Tile.Tile16>

type LeftContents is Data:
  Empty{}
  Pending{piece: Model.Content<Tile.Tile8>,rest: LeftContents}

type Emitted is Data:
  Emitted{piece: Model.Content<Tile.Tile8>,lefts: LeftContents,stage: Buffer.Content<Tile.Tile16>}

def left_contents(lefts: Pieces.Lefts<Products.Cells()>) -> LeftContents:
  match lefts:
    case Pieces.NoLefts{}: Empty{}
    case Pieces.Left{tree,rest}: Pending{Model.contents(~Tile.Tile8,tree),left_contents(rest)}

def contents(fragment: Pieces.Walked<Products.Cells(),Stage()>) -> Emitted:
  Pieces.Walked{tree,lefts,stage} = fragment
  Emitted{Model.contents(~Tile.Tile8,tree),left_contents(lefts),Buffer.contents(~Tile.Tile16,stage)}

def with_range(+first: U32,+count: U32,lefts: LeftContents,buffers: Buffer.Content<Tile.Tile16> & Buffer.Content<Tile.Tile8>) -> Emitted:
  (stage,output) = buffers
  Emitted{Model.PieceContent{output,first,count},lefts,stage}

def emitted(+first: U32,+count: U32,-lefts: Pieces.Lefts<Products.Cells()>,buffers: Stage() & Products.Cells()) ->
  {contents(Products.emitted_piece(lefts,first,count,buffers)) == with_range(first,count,left_contents(lefts),ReadInto.buffers_contents(~Tile.Tile16,~Tile.Tile8,buffers)) : Emitted}:
  (stage,output) = buffers
  {==}

def slice_contents(+slice: Gemm.StagedSlice,+count: U32,+staged: Buffer.Content<Tile.Tile16>) -> Buffer.Content<Tile.Tile16> & Buffer.Content<Tile.Tile8>:
  (staged,Buffer.Content{PointBuffer.slice_loop(slice,count,Buffer.content_tree(~Tile.Tile16,staged),
    Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Products.zero())),U32.to_nat(count)})

def fragment(+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,
  +lefts: LeftContents,+staged: Buffer.Content<Tile.Tile16>) -> Emitted:
  +low = U32.max(first,Grid.start(grid,piece))
  +count = (U32.min(end,(Grid.start(grid,piece) + Grid.span(grid,piece) : U32)) - low : U32)
  with_range(low,count,lefts,slice_contents(Gemm.StagedSlice{groups,positions,(low - first : U32),group_major},count,staged))

def piece_contents(+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,
  -lefts: Pieces.Lefts<Products.Cells()>,+staged: Buffer.Model<Tile.Tile16>) ->
  {contents(Products.emit_piece(grid,piece,first,end,groups,positions,group_major,Pieces.Walking{lefts,Buffer.pack(~Tile.Tile16,staged)})) ==
    fragment(grid,piece,first,end,groups,positions,group_major,left_contents(lefts),Buffer.content(~Tile.Tile16,staged)) : Emitted}:
  Buffer.Model{+tree,+length,+certificate} = staged
  +low = U32.max(first,Grid.start(grid,piece))
  +count = (U32.min(end,(Grid.start(grid,piece) + Grid.span(grid,piece) : U32)) - low : U32)
  +slice = {Gemm.StagedSlice{groups,positions,(low - first : U32),group_major} : Gemm.StagedSlice}
  +output = Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Products.zero(),count))
  -stage = Buffer.pack(~Tile.Tile16,staged)
  -in_stack = left_contents(lefts)
  -expected = slice_contents(slice,count,Buffer.Content{tree,length})
  -buffers = GemmBuffer.staged_slice(slice,count,stage,Tensors.buffer(~Tile.Tile8,Products.zero(),count))
  %Equal.sym(Emitted,contents(Products.emitted_piece(lefts,low,count,buffers)),with_range(low,count,in_stack,ReadInto.buffers_contents(~Tile.Tile16,~Tile.Tile8,buffers)),
    emitted(low,count,lefts,GemmBuffer.staged_slice(slice,count,Buffer.pack(~Tile.Tile16,staged),Tensors.buffer(~Tile.Tile8,Products.zero(),count)))) :
    {_ == with_range(low,count,in_stack,expected) : Emitted}
  %Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Products.zero(),count)) :
    {with_range(low,count,in_stack,ReadInto.buffers_contents(~Tile.Tile16,~Tile.Tile8,GemmBuffer.staged_slice(slice,count,stage,_))) == with_range(low,count,in_stack,expected) : Emitted}
  %Equal.sym(Buffer.Content<Tile.Tile16> & Buffer.Content<Tile.Tile8>,
    ReadInto.buffers_contents(~Tile.Tile16,~Tile.Tile8,GemmBuffer.staged_slice(slice,count,stage,Buffer.pack(~Tile.Tile8,output))),
    (Buffer.Content{tree,length},Buffer.Content{PointBuffer.slice_loop(slice,count,tree,Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)}),
    PointBuffer.slice_contents(slice,count,staged,output)) :
    {with_range(low,count,in_stack,_) == with_range(low,count,in_stack,expected) : Emitted}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,output),Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Products.zero()),Allocation.buffer_tree(~Tile.Tile8,Products.zero(),count)) :
    {with_range(low,count,in_stack,(Buffer.Content{tree,length},Buffer.Content{PointBuffer.slice_loop(slice,count,tree,_),Buffer.length(~Tile.Tile8,output)})) ==
      with_range(low,count,in_stack,expected) : Emitted}
  Equal.cong(Nat,Emitted,counted => with_range(low,count,in_stack,(Buffer.Content{tree,length},
    Buffer.Content{PointBuffer.slice_loop(slice,count,tree,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Products.zero())),counted})),
    Buffer.length(~Tile.Tile8,output),U32.to_nat(count),Allocation.buffer_length(~Tile.Tile8,Products.zero(),count))

type EmitInput is Data:
  EmitInput{lefts: LeftContents,stage: Buffer.Content<Tile.Tile16>}

def ready_contents(ready: Pieces.Walking<Products.Cells(),Stage()>) -> EmitInput:
  Pieces.Walking{lefts,stage} = ready
  EmitInput{left_contents(lefts),Buffer.contents(~Tile.Tile16,stage)}

def pushed_model(emitted: Emitted) -> EmitInput:
  Emitted{piece,lefts,stage} = emitted
  EmitInput{Pending{piece,lefts},stage}

def joined_model(emitted: Emitted) -> Emitted:
  match emitted:
    case Emitted{piece,Empty{},stage}: Emitted{piece,Empty{},stage}
    case Emitted{right,Pending{left,rest},stage}: Emitted{Model.Joined{left,right},rest,stage}

def pushed_contents(walked: Pieces.Walked<Products.Cells(),Stage()>) ->
  {ready_contents(Pieces.pushed(~Products.Cells(),~Stage(),walked)) == pushed_model(contents(walked)) : EmitInput}:
  Pieces.Walked{tree,lefts,stage} = walked
  {==}

def joined_contents(walked: Pieces.Walked<Products.Cells(),Stage()>) ->
  {contents(Pieces.joined(~Products.Cells(),~Stage(),walked)) == joined_model(contents(walked)) : Emitted}:
  match walked:
    case Pieces.Walked{tree,Pieces.NoLefts{},stage}: {==}
    case Pieces.Walked{right,Pieces.Left{left,rest},stage}: {==}

def fragment_of(+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,ready: EmitInput) -> Emitted:
  EmitInput{lefts,stage} = ready
  fragment(grid,piece,first,end,groups,positions,group_major,lefts,stage)

def fragment_contents(+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,
  ready: Pieces.Walking<Products.Cells(),Stage()>) ->
  {contents(Products.emit_piece(grid,piece,first,end,groups,positions,group_major,ready)) ==
    fragment_of(grid,piece,first,end,groups,positions,group_major,ready_contents(ready)) : Emitted}:
  Pieces.Walking{lefts,stage} = ready
  +observed = Buffer.observe(~Tile.Tile16,stage)
  +model = Buffer.observed_model(~Tile.Tile16,stage,observed)
  %Buffer.observed_equation(~Tile.Tile16,stage,observed) :
    {contents(Products.emit_piece(grid,piece,first,end,groups,positions,group_major,Pieces.Walking{lefts,_})) ==
      fragment(grid,piece,first,end,groups,positions,group_major,left_contents(lefts),Buffer.contents(~Tile.Tile16,_)) : Emitted}
  %Equal.sym(Buffer.Content<Tile.Tile16>,Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,model)),Buffer.content(~Tile.Tile16,model),
    Buffer.packed_contents(~Tile.Tile16,model)) :
    {contents(Products.emit_piece(grid,piece,first,end,groups,positions,group_major,Pieces.Walking{lefts,Buffer.pack(~Tile.Tile16,model)})) ==
      fragment(grid,piece,first,end,groups,positions,group_major,left_contents(lefts),_) : Emitted}
  piece_contents(grid,piece,first,end,groups,positions,group_major,lefts,model)

def emit_model(remaining: Nat,+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,
  ready: EmitInput) -> Emitted:
  match remaining:
    case 0n: fragment_of(grid,piece,first,end,groups,positions,group_major,ready)
    case 1n+rest:
      joined_model(emit_model(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,
        pushed_model(fragment_of(grid,piece,first,end,groups,positions,group_major,ready))))

type LeftModels is Data:
  NoModels{}
  PendingModel{piece: Model.Model<Tile.Tile8>,rest: LeftModels}

def pack_lefts(lefts: LeftModels) -> Pieces.Lefts<Products.Cells()>:
  match lefts:
    case NoModels{}: Pieces.NoLefts{}
    case PendingModel{piece,rest}: Pieces.Left{Model.pack(~Tile.Tile8,piece),pack_lefts(rest)}

type LeftObservation<-lefts: Pieces.Lefts<Products.Cells()>> is Data:
  ObservedLefts{model: LeftModels,packed: {pack_lefts(model) == lefts : Pieces.Lefts<Products.Cells()>}}

def left_observation(-tree: Pieces.Pieces(~Tile.Tile8),-rest: Pieces.Lefts<Products.Cells()>,
  observed_piece: Model.Observation<Tile.Tile8,tree>,observed_rest: LeftObservation<rest>) -> LeftObservation<Pieces.Left{tree,rest}>:
  Model.Observation{+piece,tree_exact} = observed_piece
  ObservedLefts{+tail,rest_exact} = observed_rest
  ObservedLefts{PendingModel{piece,tail},
    %rest_exact : {Pieces.Left{Model.pack(~Tile.Tile8,piece),pack_lefts(tail)} == Pieces.Left{tree,_} : Pieces.Lefts<Products.Cells()>}
    %tree_exact : {Pieces.Left{Model.pack(~Tile.Tile8,piece),pack_lefts(tail)} == Pieces.Left{_,pack_lefts(tail)} : Pieces.Lefts<Products.Cells()>}
    {==}}

def observe_lefts(lefts: Pieces.Lefts<Products.Cells()>) -> LeftObservation<lefts>:
  match lefts:
    case Pieces.NoLefts{}: ObservedLefts{NoModels{},{==}}
    case Pieces.Left{tree,rest}: left_observation(tree,rest,Model.observe(~Tile.Tile8,tree),observe_lefts(rest))

type ReadyModel is Data:
  ReadyModel{lefts: LeftModels,stage: Buffer.Model<Tile.Tile16>}

def pack_ready(model: ReadyModel) -> Pieces.Walking<Products.Cells(),Stage()>:
  ReadyModel{lefts,stage} = model
  Pieces.Walking{pack_lefts(lefts),Buffer.pack(~Tile.Tile16,stage)}

type ReadyObservation<-ready: Pieces.Walking<Products.Cells(),Stage()>> is Data:
  ObservedReady{model: ReadyModel,packed: {pack_ready(model) == ready : Pieces.Walking<Products.Cells(),Stage()>}}

def ready_observation(-lefts: Pieces.Lefts<Products.Cells()>,-stage: Stage(),observed_lefts: LeftObservation<lefts>,
  observed_stage: Buffer.Observation<Tile.Tile16,stage>) -> ReadyObservation<Pieces.Walking{lefts,stage}>:
  ObservedLefts{+models,lefts_exact} = observed_lefts
  Buffer.Observation{+model,stage_exact} = observed_stage
  ObservedReady{ReadyModel{models,model},
    %lefts_exact : {Pieces.Walking{pack_lefts(models),Buffer.pack(~Tile.Tile16,model)} == Pieces.Walking{_,stage} : Pieces.Walking<Products.Cells(),Stage()>}
    %stage_exact : {Pieces.Walking{pack_lefts(models),Buffer.pack(~Tile.Tile16,model)} == Pieces.Walking{pack_lefts(models),_} : Pieces.Walking<Products.Cells(),Stage()>}
    {==}}

def observe_ready(ready: Pieces.Walking<Products.Cells(),Stage()>) -> ReadyObservation<ready>:
  Pieces.Walking{lefts,stage} = ready
  ready_observation(lefts,stage,observe_lefts(lefts),Buffer.observe(~Tile.Tile16,stage))

def observed_ready(-ready: Pieces.Walking<Products.Cells(),Stage()>,observation: ReadyObservation<ready>) -> ReadyModel:
  ObservedReady{model,packed} = observation
  model

def ready_packed(-ready: Pieces.Walking<Products.Cells(),Stage()>,+observation: ReadyObservation<ready>) ->
  {pack_ready(observed_ready(ready,observation)) == ready : Pieces.Walking<Products.Cells(),Stage()>}:
  ObservedReady{model,packed} = observation
  packed

# Every fragment in the actual writeback, including a range spanning pieces.
def emit_contents(remaining: Nat,+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,
  +input: ReadyModel) ->
  {contents(Products.emit(remaining,grid,piece,first,end,groups,positions,group_major,pack_ready(input))) ==
    emit_model(remaining,grid,piece,first,end,groups,positions,group_major,ready_contents(pack_ready(input))) : Emitted}:
  match remaining:
    case 0n: fragment_contents(grid,piece,first,end,groups,positions,group_major,pack_ready(input))
    case 1n+ +rest:
      -ready = pack_ready(input)
      -first_piece = Products.emit_piece(grid,piece,first,end,groups,positions,group_major,ready)
      -pushed = Pieces.pushed(~Products.Cells(),~Stage(),first_piece)
      -later = Products.emit(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,pushed)
      +observed = observe_ready(Pieces.pushed(~Products.Cells(),~Stage(),Products.emit_piece(grid,piece,first,end,groups,positions,group_major,pack_ready(input))))
      %Equal.sym(Emitted,contents(Pieces.joined(~Products.Cells(),~Stage(),later)),joined_model(contents(later)),
        joined_contents(Products.emit(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,
          Pieces.pushed(~Products.Cells(),~Stage(),Products.emit_piece(grid,piece,first,end,groups,positions,group_major,pack_ready(input)))))) :
        {_ == emit_model(1n+rest,grid,piece,first,end,groups,positions,group_major,ready_contents(ready)) : Emitted}
      %Equal.sym(Emitted,contents(later),emit_model(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,ready_contents(pushed)),
        {%ready_packed(pushed,observed) :
          {contents(Products.emit(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,_)) ==
            emit_model(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,ready_contents(_)) : Emitted}
          emit_contents(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,observed_ready(pushed,observed)) :
          {contents(later) == emit_model(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,ready_contents(pushed)) : Emitted}}) :
        {joined_model(_) == emit_model(1n+rest,grid,piece,first,end,groups,positions,group_major,ready_contents(ready)) : Emitted}
      %Equal.sym(EmitInput,ready_contents(pushed),pushed_model(contents(first_piece)),
        pushed_contents(Products.emit_piece(grid,piece,first,end,groups,positions,group_major,pack_ready(input)))) :
        {joined_model(emit_model(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,_)) ==
          emit_model(1n+rest,grid,piece,first,end,groups,positions,group_major,ready_contents(ready)) : Emitted}
      %Equal.sym(Emitted,contents(first_piece),fragment_of(grid,piece,first,end,groups,positions,group_major,ready_contents(ready)),
        fragment_contents(grid,piece,first,end,groups,positions,group_major,pack_ready(input))) :
        {joined_model(emit_model(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,pushed_model(_))) ==
          emit_model(1n+rest,grid,piece,first,end,groups,positions,group_major,ready_contents(ready)) : Emitted}
      {==}

def output_content(emitted: Emitted) -> Model.Content<Tile.Tile8>:
  Emitted{piece,lefts,stage} = emitted
  piece

def emitted_tree(remaining: Nat,+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,
  +stage: Buffer.Content<Tile.Tile16>) -> Model.Content<Tile.Tile8>:
  match remaining:
    case 0n: output_content(fragment(grid,piece,first,end,groups,positions,group_major,Empty{},stage))
    case 1n+rest:
      Model.Joined{output_content(fragment(grid,piece,first,end,groups,positions,group_major,Empty{},stage)),
        emitted_tree(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,stage)}

def emission_tree(remaining: Nat,+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,
  +lefts: LeftContents,+stage: Buffer.Content<Tile.Tile16>) ->
  {emit_model(remaining,grid,piece,first,end,groups,positions,group_major,EmitInput{lefts,stage}) ==
    Emitted{emitted_tree(remaining,grid,piece,first,end,groups,positions,group_major,stage),lefts,stage} : Emitted}:
  match remaining:
    case 0n: {==}
    case 1n+rest:
      +current = output_content(fragment(grid,piece,first,end,groups,positions,group_major,Empty{},stage))
      +next_lefts = {Pending{current,lefts} : LeftContents}
      +next = (piece + 1 : U32)
      -walked = emit_model(rest,grid,next,first,end,groups,positions,group_major,EmitInput{next_lefts,stage})
      -want = {Emitted{emitted_tree(rest,grid,next,first,end,groups,positions,group_major,stage),next_lefts,stage} : Emitted}
      %Equal.sym(Emitted,walked,want,emission_tree(rest,grid,next,first,end,groups,positions,group_major,next_lefts,stage)) :
        {joined_model(_) == Emitted{emitted_tree(1n+rest,grid,piece,first,end,groups,positions,group_major,stage),lefts,stage} : Emitted}
      {==}

def range_first(+grid: Grid.Grid,+piece: U32,+first: U32) -> {U32.max(first,Grid.start(grid,piece)) == Pipeline.clipped_first(grid,piece,first) : U32}:
  +start = Grid.start(grid,piece)
  Arithmetic.to_nat_injective(U32.max(first,start),U32.max(start,first),
    Equal.trans(Nat,U32.to_nat(U32.max(first,start)),Nat.max(U32.to_nat(first),U32.to_nat(start)),U32.to_nat(U32.max(start,first)),
      Extrema.maximum(first,start),
      Equal.trans(Nat,Nat.max(U32.to_nat(first),U32.to_nat(start)),Nat.max(U32.to_nat(start),U32.to_nat(first)),U32.to_nat(U32.max(start,first)),
        NatExtrema.maximum_commutative(U32.to_nat(first),U32.to_nat(start)),
        Equal.sym(Nat,U32.to_nat(U32.max(start,first)),Nat.max(U32.to_nat(start),U32.to_nat(first)),Extrema.maximum(start,first)))))

def range_count(+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32) ->
  {(U32.min(end,(Grid.start(grid,piece) + Grid.span(grid,piece) : U32)) - U32.max(first,Grid.start(grid,piece)) : U32) ==
    Pipeline.clipped_count(grid,piece,first,end) : U32}:
  +high = (Grid.start(grid,piece) + Grid.span(grid,piece) : U32)
  +minimum = Arithmetic.to_nat_injective(U32.min(end,high),U32.min(high,end),
    Equal.trans(Nat,U32.to_nat(U32.min(end,high)),Nat.min(U32.to_nat(end),U32.to_nat(high)),U32.to_nat(U32.min(high,end)),
      Extrema.minimum(end,high),
      Equal.trans(Nat,Nat.min(U32.to_nat(end),U32.to_nat(high)),Nat.min(U32.to_nat(high),U32.to_nat(end)),U32.to_nat(U32.min(high,end)),
        NatExtrema.minimum_commutative(U32.to_nat(end),U32.to_nat(high)),
        Equal.sym(Nat,U32.to_nat(U32.min(high,end)),Nat.min(U32.to_nat(high),U32.to_nat(end)),Extrema.minimum(high,end)))))
  %range_first(grid,piece,first) :
    {(U32.min(end,high) - U32.max(first,Grid.start(grid,piece)) : U32) == (U32.min(high,end) - _ : U32) : U32}
  %minimum :
    {(U32.min(end,high) - U32.max(first,Grid.start(grid,piece)) : U32) == (_ - U32.max(first,Grid.start(grid,piece)) : U32) : U32}
  {==}

def fragment_layout(+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,+stage: Buffer.Content<Tile.Tile16>) ->
  {Model.content_layout(~Tile.Tile8,output_content(fragment(grid,piece,first,end,groups,positions,group_major,Empty{},stage))) ==
    Pieces.Range{Pipeline.clipped_first(grid,piece,first),Pipeline.clipped_count(grid,piece,first,end)} : Pieces.Ranges}:
  %range_first(grid,piece,first) :
    {Model.content_layout(~Tile.Tile8,output_content(fragment(grid,piece,first,end,groups,positions,group_major,Empty{},stage))) ==
      Pieces.Range{_,Pipeline.clipped_count(grid,piece,first,end)} : Pieces.Ranges}
  %range_count(grid,piece,first,end) :
    {Model.content_layout(~Tile.Tile8,output_content(fragment(grid,piece,first,end,groups,positions,group_major,Empty{},stage))) ==
      Pieces.Range{U32.max(first,Grid.start(grid,piece)),_} : Pieces.Ranges}
  {==}

def emitted_tree_layout(remaining: Nat,+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,
  +stage: Buffer.Content<Tile.Tile16>) ->
  {Model.content_layout(~Tile.Tile8,emitted_tree(remaining,grid,piece,first,end,groups,positions,group_major,stage)) ==
    Clipped.clipped_layout(remaining,grid,piece,first,end,Pipeline.clipped_first(grid,piece,first),Pipeline.clipped_count(grid,piece,first,end)) : Pieces.Ranges}:
  match remaining:
    case 0n: fragment_layout(grid,piece,first,end,groups,positions,group_major,stage)
    case 1n+ +rest:
      +next = (piece + 1 : U32)
      -tail = Clipped.clipped_layout(rest,grid,next,first,end,Pipeline.clipped_first(grid,next,first),Pipeline.clipped_count(grid,next,first,end))
      %fragment_layout(grid,piece,first,end,groups,positions,group_major,stage) :
        {Model.content_layout(~Tile.Tile8,emitted_tree(1n+rest,grid,piece,first,end,groups,positions,group_major,stage)) == Pieces.Split{_,tail} : Pieces.Ranges}
      %emitted_tree_layout(rest,grid,next,first,end,groups,positions,group_major,stage) :
        {Model.content_layout(~Tile.Tile8,emitted_tree(1n+rest,grid,piece,first,end,groups,positions,group_major,stage)) ==
          Pieces.Split{Model.content_layout(~Tile.Tile8,output_content(fragment(grid,piece,first,end,groups,positions,group_major,Empty{},stage))),_} : Pieces.Ranges}
      {==}

def completed_content(completed: Products.Completed) -> Model.Content<Tile.Tile8>:
  Products.Completed{pieces,input,bias} = completed
  Model.contents(~Tile.Tile8,pieces)

def walk_output(walked: Pieces.Walked<Products.Cells(),Stage()>) ->
  {Model.contents(~Tile.Tile8,Pair.fst(Pieces.Pieces(~Tile.Tile8),Stage(),Pieces.walk_result(~Products.Cells(),~Stage(),walked))) ==
    output_content(contents(walked)) : Model.Content<Tile.Tile8>}:
  Pieces.Walked{tree,lefts,stage} = walked
  {==}

def rectangle_model(+rectangle: Products.Rectangle,+staged: Buffer.Content<Tile.Tile16>) -> Model.Content<Tile.Tile8>:
  Products.Rectangle{+first,+positions,inner,+channels,+group_major,+grid} = rectangle
  +groups = Gemm.blocks(channels)
  +count = (positions * groups : U32)
  output_content(emit_model(Grid.later_pieces(grid,first,count),grid,Grid.piece_of(grid,first),first,(first + count : U32),
    groups,positions,group_major,EmitInput{Empty{},staged}))

def clipped_rectangle(+rectangle: Products.Rectangle) -> Pieces.Ranges:
  Products.Rectangle{+first,+positions,inner,+channels,group_major,+grid} = rectangle
  +count = (positions * Gemm.blocks(channels) : U32)
  +piece = Grid.piece_of(grid,first)
  +end = (first + count : U32)
  Clipped.clipped_layout(Grid.later_pieces(grid,first,count),grid,piece,first,end,Pipeline.clipped_first(grid,piece,first),Pipeline.clipped_count(grid,piece,first,end))

def rectangle_layout(+rectangle: Products.Rectangle,+stage: Buffer.Content<Tile.Tile16>) ->
  {Model.content_layout(~Tile.Tile8,rectangle_model(rectangle,stage)) == clipped_rectangle(rectangle) : Pieces.Ranges}:
  Products.Rectangle{+first,+positions,inner,+channels,+group_major,+grid} = rectangle
  +groups = Gemm.blocks(channels)
  +count = (positions * groups : U32)
  +remaining = Grid.later_pieces(grid,first,count)
  +piece = Grid.piece_of(grid,first)
  +end = (first + count : U32)
  -walked = emit_model(remaining,grid,piece,first,end,groups,positions,group_major,EmitInput{Empty{},stage})
  -tree = emitted_tree(remaining,grid,piece,first,end,groups,positions,group_major,stage)
  %Equal.sym(Emitted,walked,Emitted{tree,Empty{},stage},emission_tree(remaining,grid,piece,first,end,groups,positions,group_major,Empty{},stage)) :
    {Model.content_layout(~Tile.Tile8,output_content(_)) == clipped_rectangle(rectangle) : Pieces.Ranges}
  emitted_tree_layout(remaining,grid,piece,first,end,groups,positions,group_major,stage)

# The rectangle emits the full walk from an empty stack of pending pieces.
def rectangle_contents(+rectangle: Products.Rectangle,-bias: Products.Cells(),-input: Products.Cells(),-weights: Stage(),
  +staged: Buffer.Model<Tile.Tile16>) ->
  {completed_content(Products.emitted(rectangle,GemmBuffer.PointKept{bias,input,weights,Buffer.pack(~Tile.Tile16,staged)})) ==
    rectangle_model(rectangle,Buffer.content(~Tile.Tile16,staged)) : Model.Content<Tile.Tile8>}:
  Products.Rectangle{+first,+positions,inner,+channels,+group_major,+grid} = rectangle
  +groups = Gemm.blocks(channels)
  +count = (positions * groups : U32)
  +remaining = Grid.later_pieces(grid,first,count)
  +piece = Grid.piece_of(grid,first)
  +end = (first + count : U32)
  +model = {ReadyModel{NoModels{},staged} : ReadyModel}
  -walked = Products.emit(remaining,grid,piece,first,end,groups,positions,group_major,pack_ready(model))
  -initial = {EmitInput{Empty{},Buffer.content(~Tile.Tile16,staged)} : EmitInput}
  %Equal.sym(Model.Content<Tile.Tile8>,Model.contents(~Tile.Tile8,Pair.fst(Pieces.Pieces(~Tile.Tile8),Stage(),Pieces.walk_result(~Products.Cells(),~Stage(),walked))),
    output_content(contents(walked)),walk_output(Products.emit(remaining,grid,piece,first,end,groups,positions,group_major,pack_ready(model)))) :
    {_ == output_content(emit_model(remaining,grid,piece,first,end,groups,positions,group_major,initial)) : Model.Content<Tile.Tile8>}
  %Buffer.packed_contents(~Tile.Tile16,staged) :
    {output_content(contents(walked)) == output_content(emit_model(remaining,grid,piece,first,end,groups,positions,group_major,EmitInput{Empty{},_})) :
      Model.Content<Tile.Tile8>}
  Equal.cong(Emitted,Model.Content<Tile.Tile8>,output_content,contents(walked),
    emit_model(remaining,grid,piece,first,end,groups,positions,group_major,ready_contents(pack_ready(model))),
    emit_contents(remaining,grid,piece,first,end,groups,positions,group_major,model))

def Run() -> Type:
  Nat -> Gemm.PointMatrix -> PointJobs.Operands -> PointJobs.Operands

def block_step(~run: Run(),+matrix: Gemm.PointMatrix,state: PointBuffer.PointContents) -> PointBuffer.PointContents:
  PointBuffer.PointContents{Buffer.Content{bias,bias_length},Buffer.Content{input,input_length},Buffer.Content{pairs,pair_length},Buffer.Content{staged,staged_length}} = state
  PointBuffer.operand_contents(run(U32.to_nat(Gemm.point_count(matrix)),matrix,PointJobs.Operands{bias,input,pairs,Writes.Initial{staged}}),
    PointBuffer.PointLengths{bias_length,input_length,pair_length,staged_length})

def block_step_matches(~run: Run(),+matrix: Gemm.PointMatrix,+input: Buffer.Model<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>,
  +bias: Buffer.Model<Tile.Tile8>,+staged: Buffer.Model<Tile.Tile16>) ->
  {PointBuffer.operand_contents(run(U32.to_nat(Gemm.point_count(matrix)),matrix,PointBuffer.model_operands(input,pairs,bias,staged)),
    PointBuffer.model_lengths(input,pairs,bias,staged)) == block_step(~run,matrix,PointBuffer.model_contents(PointBuffer.PointKeptModel{bias,input,pairs,staged})) : PointBuffer.PointContents}:
  Buffer.Model{+input_tree,+input_length,input_certificate} = input
  Buffer.Model{+pair_tree,+pair_length,pair_certificate} = pairs
  Buffer.Model{+bias_tree,+bias_length,bias_certificate} = bias
  Buffer.Model{+staged_tree,+staged_length,staged_certificate} = staged
  {==}

def kernel_call(~kernel: Products.Kernel(),+matrix: Gemm.PointMatrix,state: PointBuffer.PointKeptModel) -> GemmBuffer.PointKept:
  PointBuffer.PointKeptModel{bias,input,pairs,staged} = state
  kernel(matrix,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged))

def BlockAgreement(~kernel: Products.Kernel(),~run: Run()) -> Type:
  @+matrix: Gemm.PointMatrix -> @+state: PointBuffer.PointKeptModel ->
    {PointBuffer.kept_contents(kernel_call(~kernel,matrix,state)) == block_step(~run,matrix,PointBuffer.model_contents(state)) : PointBuffer.PointContents}

def block_whole(+matrix: Gemm.PointMatrix,+state: PointBuffer.PointKeptModel) ->
  {PointBuffer.kept_contents(kernel_call(~GemmBuffer.point_keeping,matrix,state)) ==
    block_step(~PointJobs.run,matrix,PointBuffer.model_contents(state)) : PointBuffer.PointContents}:
  PointBuffer.PointKeptModel{+bias,+input,+pairs,+staged} = state
  %block_step_matches(~PointJobs.run,matrix,input,pairs,bias,staged) :
    {PointBuffer.kept_contents(kernel_call(~GemmBuffer.point_keeping,matrix,state)) == _ : PointBuffer.PointContents}
  PointBuffer.keeping_contents(matrix,input,pairs,bias,staged)

def block_tail(+matrix: Gemm.PointMatrix,+state: PointBuffer.PointKeptModel) ->
  {PointBuffer.kept_contents(kernel_call(~GemmBuffer.point_keeping_tail,matrix,state)) ==
    block_step(~PointJobs.run_tail,matrix,PointBuffer.model_contents(state)) : PointBuffer.PointContents}:
  PointBuffer.PointKeptModel{+bias,+input,+pairs,+staged} = state
  %block_step_matches(~PointJobs.run_tail,matrix,input,pairs,bias,staged) :
    {PointBuffer.kept_contents(kernel_call(~GemmBuffer.point_keeping_tail,matrix,state)) == _ : PointBuffer.PointContents}
  PointBuffer.keeping_tail_contents(matrix,input,pairs,bias,staged)

def blocks_model(~run: Run(),remaining: Nat,+rectangle: Products.Rectangle,+input_base: U32,+first: U32,+span: U32,
  state: PointBuffer.PointContents) -> PointBuffer.PointContents:
  match remaining:
    case 0n: state
    case 1n+rest:
      Products.Rectangle{output_first,+positions,+inner,+channels,group_major,grid} = rectangle
      +rows = U32.min(span,(positions - first : U32))
      +groups = Gemm.blocks(inner)
      +end = (first + rows : U32)
      blocks_model(~run,rest,rectangle,input_base,end,span,
        block_step(~run,Gemm.PointMatrix{Gemm.TileMatrix{channels,rows,inner,first,end},(input_base + first * groups : U32),groups},state))

# The native position blocks use the modeled GEMM stores at each step.
def blocks_contents(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),
  remaining: Nat,+rectangle: Products.Rectangle,+input_base: U32,+first: U32,+span: U32,+state: PointBuffer.PointKeptModel) ->
  {PointBuffer.kept_contents(Products.rect_blocks(~kernel,remaining,rectangle,input_base,first,span,PointBuffer.pack_point_kept(state))) ==
    blocks_model(~run,remaining,rectangle,input_base,first,span,PointBuffer.model_contents(state)) : PointBuffer.PointContents}:
  match remaining:
    case 0n:
      PointBuffer.PointKeptModel{+bias,+input,+pairs,+staged} = state
      %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias)),Buffer.content(~Tile.Tile8,bias),Buffer.packed_contents(~Tile.Tile8,bias)) :
        {PointBuffer.PointContents{_,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input)),Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,pairs)),
          Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,staged))} == PointBuffer.model_contents(state) : PointBuffer.PointContents}
      %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input)),Buffer.content(~Tile.Tile8,input),Buffer.packed_contents(~Tile.Tile8,input)) :
        {PointBuffer.PointContents{Buffer.content(~Tile.Tile8,bias),_,Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,pairs)),
          Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,staged))} == PointBuffer.model_contents(state) : PointBuffer.PointContents}
      %Equal.sym(Buffer.Content<Tile.Tile16>,Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,pairs)),Buffer.content(~Tile.Tile16,pairs),Buffer.packed_contents(~Tile.Tile16,pairs)) :
        {PointBuffer.PointContents{Buffer.content(~Tile.Tile8,bias),Buffer.content(~Tile.Tile8,input),_,
          Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,staged))} == PointBuffer.model_contents(state) : PointBuffer.PointContents}
      %Equal.sym(Buffer.Content<Tile.Tile16>,Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,staged)),Buffer.content(~Tile.Tile16,staged),Buffer.packed_contents(~Tile.Tile16,staged)) :
        {PointBuffer.PointContents{Buffer.content(~Tile.Tile8,bias),Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile16,pairs),_} ==
          PointBuffer.model_contents(state) : PointBuffer.PointContents}
      {==}
    case 1n+ +rest:
      Products.Rectangle{output_first,+positions,+inner,+channels,group_major,grid} = rectangle
      PointBuffer.PointKeptModel{+bias,+input,+pairs,+staged} = state
      +rows = U32.min(span,(positions - first : U32))
      +groups = Gemm.blocks(inner)
      +end = (first + rows : U32)
      +matrix = {Gemm.PointMatrix{Gemm.TileMatrix{channels,rows,inner,first,end},(input_base + first * groups : U32),groups} : Gemm.PointMatrix}
      -done = kernel_call(~kernel,matrix,state)
      +observed = PointBuffer.remodel_point_kept(kernel_call(~kernel,matrix,state))
      %agrees(matrix,state) :
        {PointBuffer.kept_contents(Products.rect_blocks(~kernel,rest,rectangle,input_base,end,span,done)) ==
          blocks_model(~run,rest,rectangle,input_base,end,span,_) : PointBuffer.PointContents}
      %PointBuffer.remodeled_contents(kernel_call(~kernel,matrix,state)) :
        {PointBuffer.kept_contents(Products.rect_blocks(~kernel,rest,rectangle,input_base,end,span,done)) ==
          blocks_model(~run,rest,rectangle,input_base,end,span,_) : PointBuffer.PointContents}
      %PointBuffer.point_kept_remodeled(kernel_call(~kernel,matrix,state)) :
        {PointBuffer.kept_contents(Products.rect_blocks(~kernel,rest,rectangle,input_base,end,span,_)) ==
          blocks_model(~run,rest,rectangle,input_base,end,span,PointBuffer.model_contents(observed)) : PointBuffer.PointContents}
      blocks_contents(~kernel,~run,~agrees,rest,rectangle,input_base,end,span,observed)

def staged_content(state: PointBuffer.PointContents) -> Buffer.Content<Tile.Tile16>:
  PointBuffer.PointContents{bias,input,pairs,staged} = state
  staged

def emitted_contents(+rectangle: Products.Rectangle,kept: GemmBuffer.PointKept) ->
  {completed_content(Products.emitted(rectangle,kept)) ==
    rectangle_model(rectangle,staged_content(PointBuffer.kept_contents(kept))) : Model.Content<Tile.Tile8>}:
  GemmBuffer.PointKept{bias,input,weights,staged} = kept
  +observation = Buffer.observe(~Tile.Tile16,staged)
  +model = Buffer.observed_model(~Tile.Tile16,staged,observation)
  %Buffer.observed_equation(~Tile.Tile16,staged,observation) :
    {completed_content(Products.emitted(rectangle,GemmBuffer.PointKept{bias,input,weights,_})) ==
      rectangle_model(rectangle,Buffer.contents(~Tile.Tile16,_)) : Model.Content<Tile.Tile8>}
  %Equal.sym(Buffer.Content<Tile.Tile16>,Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,model)),Buffer.content(~Tile.Tile16,model),Buffer.packed_contents(~Tile.Tile16,model)) :
    {completed_content(Products.emitted(rectangle,GemmBuffer.PointKept{bias,input,weights,Buffer.pack(~Tile.Tile16,model)})) ==
      rectangle_model(rectangle,_) : Model.Content<Tile.Tile8>}
  rectangle_contents(rectangle,bias,input,weights,model)

def computed_staged(~run: Run(),+rectangle: Products.Rectangle,+input_base: U32,input: Buffer.Content<Tile.Tile8>,
  pairs: Buffer.Content<Tile.Tile16>,bias: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile16>:
  Products.Rectangle{first,+positions,+inner,+channels,group_major,grid} = rectangle
  +span = Products.block_positions(inner)
  +count = (Gemm.pairs_of(channels) * positions : U32)
  staged_content(blocks_model(~run,U32.to_nat(Grid.divided_up(positions,span)),rectangle,input_base,0,span,
    PointBuffer.PointContents{bias,input,pairs,Buffer.Content{Tree.fresh(Tile.Tile16,Tensors.depth_for(count),Tile.filled_pair(0.0)),U32.to_nat(count)}}))

def computed_model(~run: Run(),+rectangle: Products.Rectangle,+input_base: U32,input: Buffer.Content<Tile.Tile8>,
  pairs: Buffer.Content<Tile.Tile16>,bias: Buffer.Content<Tile.Tile8>) -> Model.Content<Tile.Tile8>:
  rectangle_model(rectangle,computed_staged(~run,rectangle,input_base,input,pairs,bias))

def computed_model_layout(~run: Run(),+rectangle: Products.Rectangle,+input_base: U32,+input: Buffer.Content<Tile.Tile8>,
  +pairs: Buffer.Content<Tile.Tile16>,+bias: Buffer.Content<Tile.Tile8>) ->
  {Model.content_layout(~Tile.Tile8,computed_model(~run,rectangle,input_base,input,pairs,bias)) == clipped_rectangle(rectangle) : Pieces.Ranges}:
  rectangle_layout(rectangle,computed_staged(~run,rectangle,input_base,input,pairs,bias))

# Both jobs use the staged allocation, position blocks and writeback of the same rectangle.
def computed_contents(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),
  +rectangle: Products.Rectangle,+input_base: U32,+input: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>,-weights: Products.Cells()) ->
  {completed_content(Products.computed(~kernel,rectangle,input_base,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,bias),(weights,Buffer.pack(~Tile.Tile16,pairs)))) ==
    computed_model(~run,rectangle,input_base,Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile16,pairs),Buffer.content(~Tile.Tile8,bias)) : Model.Content<Tile.Tile8>}:
  Products.Rectangle{first,+positions,+inner,+channels,group_major,grid} = rectangle
  +span = Products.block_positions(inner)
  +count = (Gemm.pairs_of(channels) * positions : U32)
  +remaining = U32.to_nat(Grid.divided_up(positions,span))
  +fill = Tile.filled_pair(0.0)
  +staged = Buffer.remodel(~Tile.Tile16,Tensors.buffer(~Tile.Tile16,fill,count))
  +initial = {PointBuffer.PointKeptModel{bias,input,pairs,staged} : PointBuffer.PointKeptModel}
  -run_result = Products.rect_blocks(~kernel,remaining,rectangle,input_base,0,span,PointBuffer.pack_point_kept(initial))
  -expected = computed_model(~run,rectangle,input_base,Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile16,pairs),Buffer.content(~Tile.Tile8,bias))
  %Buffer.remodeled(~Tile.Tile16,Tensors.buffer(~Tile.Tile16,fill,count)) :
    {completed_content(Products.emitted(rectangle,Products.rect_blocks(~kernel,remaining,rectangle,input_base,0,span,
      GemmBuffer.PointKept{Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile16,pairs),_}))) == expected : Model.Content<Tile.Tile8>}
  %Equal.sym(Model.Content<Tile.Tile8>,completed_content(Products.emitted(rectangle,run_result)),
    rectangle_model(rectangle,staged_content(PointBuffer.kept_contents(run_result))),
    emitted_contents(rectangle,Products.rect_blocks(~kernel,remaining,rectangle,input_base,0,span,PointBuffer.pack_point_kept(initial)))) :
    {_ == expected : Model.Content<Tile.Tile8>}
  %Allocation.buffer_content(~Tile.Tile16,fill,count) :
    {rectangle_model(rectangle,staged_content(PointBuffer.kept_contents(run_result))) ==
      rectangle_model(rectangle,staged_content(blocks_model(~run,remaining,rectangle,input_base,0,span,
        PointBuffer.PointContents{Buffer.content(~Tile.Tile8,bias),Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile16,pairs),_}))) : Model.Content<Tile.Tile8>}
  Equal.cong(PointBuffer.PointContents,Model.Content<Tile.Tile8>,point => rectangle_model(rectangle,staged_content(point)),
    PointBuffer.kept_contents(run_result),blocks_model(~run,remaining,rectangle,input_base,0,span,PointBuffer.model_contents(initial)),
    blocks_contents(~kernel,~run,~agrees,remaining,rectangle,input_base,0,span,initial))

def pair_content(buffers: Buffer.Content<Tile.Tile8> & Buffer.Content<Tile.Tile16>) -> Buffer.Content<Tile.Tile16>:
  (weights,pairs) = buffers
  pairs

def computed_pair(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),
  +rectangle: Products.Rectangle,+input_base: U32,+input: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,buffers: Products.Cells() & Stage()) ->
  {completed_content(Products.computed(~kernel,rectangle,input_base,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,bias),buffers)) ==
    computed_model(~run,rectangle,input_base,Buffer.content(~Tile.Tile8,input),pair_content(ReadInto.buffers_contents(~Tile.Tile8,~Tile.Tile16,buffers)),
      Buffer.content(~Tile.Tile8,bias)) : Model.Content<Tile.Tile8>}:
  (weights,pairs) = buffers
  +observed = Buffer.observe(~Tile.Tile16,pairs)
  +model = Buffer.observed_model(~Tile.Tile16,pairs,observed)
  %Buffer.observed_equation(~Tile.Tile16,pairs,observed) :
    {completed_content(Products.computed(~kernel,rectangle,input_base,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,bias),(weights,_))) ==
      computed_model(~run,rectangle,input_base,Buffer.content(~Tile.Tile8,input),Buffer.contents(~Tile.Tile16,_),Buffer.content(~Tile.Tile8,bias)) : Model.Content<Tile.Tile8>}
  %Equal.sym(Buffer.Content<Tile.Tile16>,Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,model)),Buffer.content(~Tile.Tile16,model),Buffer.packed_contents(~Tile.Tile16,model)) :
    {completed_content(Products.computed(~kernel,rectangle,input_base,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,bias),(weights,Buffer.pack(~Tile.Tile16,model)))) ==
      computed_model(~run,rectangle,input_base,Buffer.content(~Tile.Tile8,input),_,Buffer.content(~Tile.Tile8,bias)) : Model.Content<Tile.Tile8>}
  computed_contents(~kernel,~run,~agrees,rectangle,input_base,input,bias,model,weights)

def fresh_pairs(+channels: U32,+inner: U32) -> Buffer.Model<Tile.Tile16>:
  Buffer.remodel(~Tile.Tile16,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),(Gemm.pairs_of(channels) * inner : U32)))

def shared_pair_model(+rectangle: Products.Rectangle,+weights: Buffer.Model<Tile.Tile8>) -> Buffer.Content<Tile.Tile16>:
  Products.Rectangle{first,positions,+inner,+channels,group_major,grid} = rectangle
  +fresh = fresh_pairs(channels,inner)
  Buffer.Content{PointBuffer.pair_loop(channels,inner,Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile16,fresh)),Buffer.length(~Tile.Tile16,fresh)}

def panel_pair_model(+rectangle: Products.Rectangle,+base: U32,+weights: Buffer.Model<Tile.Tile8>) -> Buffer.Content<Tile.Tile16>:
  Products.Rectangle{first,positions,+inner,+channels,group_major,grid} = rectangle
  +fresh = fresh_pairs(channels,inner)
  Buffer.Content{PointBuffer.panel_loop(channels,inner,base,Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile16,fresh)),Buffer.length(~Tile.Tile16,fresh)}

def panel_band_contents(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),+rectangle: Products.Rectangle,+input_base: U32,+weight_base: U32,
  -left: Move.Lent<Tile.Tile8>,-right: Move.Lent<Tile.Tile8>,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  {completed_content(Products.panel_bands(~kernel,rectangle,Move.Band{left,Buffer.pack(~Tile.Tile8,input),input_base},Move.Band{right,Buffer.pack(~Tile.Tile8,weights),weight_base},Buffer.pack(~Tile.Tile8,bias))) ==
    computed_model(~run,rectangle,input_base,Buffer.content(~Tile.Tile8,input),panel_pair_model(rectangle,weight_base,weights),Buffer.content(~Tile.Tile8,bias)) : Model.Content<Tile.Tile8>}:
  Products.Rectangle{first,positions,+inner,+channels,group_major,grid} = rectangle
  +count = (Gemm.pairs_of(channels) * inner : U32)
  +fresh = fresh_pairs(channels,inner)
  -buffers = GemmBuffer.panel_pairs(channels,inner,weight_base,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile16,fresh))
  -expected = computed_model(~run,rectangle,input_base,Buffer.content(~Tile.Tile8,input),panel_pair_model(rectangle,weight_base,weights),Buffer.content(~Tile.Tile8,bias))
  %Buffer.remodeled(~Tile.Tile16,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),count)) :
    {completed_content(Products.computed(~kernel,rectangle,input_base,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,bias),
      GemmBuffer.panel_pairs(channels,inner,weight_base,Buffer.pack(~Tile.Tile8,weights),_))) == expected : Model.Content<Tile.Tile8>}
  %PointBuffer.panel_contents(channels,inner,weight_base,weights,fresh) :
    {completed_content(Products.computed(~kernel,rectangle,input_base,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,bias),buffers)) ==
      computed_model(~run,rectangle,input_base,Buffer.content(~Tile.Tile8,input),pair_content(_),Buffer.content(~Tile.Tile8,bias)) : Model.Content<Tile.Tile8>}
  computed_pair(~kernel,~run,~agrees,rectangle,input_base,input,bias,
    GemmBuffer.panel_pairs(channels,inner,weight_base,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile16,fresh)))

def copied_weight_model(weights: Move.Holding<Tile.Tile8> & Products.Cells()) -> Buffer.Model<Tile.Tile8>:
  (holding,buffer) = weights
  Buffer.remodel(~Tile.Tile8,buffer)

def copied_weight_projection(weights: Move.Holding<Tile.Tile8> & Products.Cells()) ->
  {copied_weight_model(weights) == Buffer.remodel(~Tile.Tile8,Pair.snd(Move.Holding<Tile.Tile8>,Products.Cells(),weights)) : Buffer.Model<Tile.Tile8>}:
  (holding,buffer) = weights
  {==}

# The pairs the held pairs of a rectangle leave.
def held_pairs_model(+rectangle: Products.Rectangle,+base: U32,weights: Move.Holding<Tile.Tile8>) -> Buffer.Content<Tile.Tile16>:
  Products.Rectangle{first,positions,+inner,+channels,group_major,grid} = rectangle
  +pairs = Gemm.pairs_of(channels)
  Buffer.contents(~Tile.Tile16,Products.held_pairs(U32.to_nat(pairs),0,Gemm.blocks(channels),inner,base,((weights,Move.spare(~Tile.Tile8,Products.zero())),
    Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),(pairs * inner : U32)))))

def shared_held_contents(+rectangle: Products.Rectangle,+input_base: U32,+weight_base: U32,+input: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,
  weights: Move.Holding<Tile.Tile8>) ->
  {completed_content(Products.compute_rectangle(~GemmBuffer.point_keeping,rectangle,input_base,weight_base,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,bias),weights)) ==
    computed_model(~PointJobs.run,rectangle,input_base,Buffer.content(~Tile.Tile8,input),held_pairs_model(rectangle,weight_base,weights),Buffer.content(~Tile.Tile8,bias)) :
    Model.Content<Tile.Tile8>}:
  Products.Rectangle{first,positions,+inner,+channels,group_major,grid} = rectangle
  +pairs = Gemm.pairs_of(channels)
  computed_pair(~GemmBuffer.point_keeping,~PointJobs.run,~block_whole,rectangle,input_base,input,bias,
    (Tensors.buffer(~Tile.Tile8,Products.zero(),0),Products.held_pairs(U32.to_nat(pairs),0,Gemm.blocks(channels),inner,weight_base,
      ((weights,Move.spare(~Tile.Tile8,Products.zero())),Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),(pairs * inner : U32))))))

def shared_held_layout(+rectangle: Products.Rectangle,+input_base: U32,+weight_base: U32,+input: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,
  +weights: MoveReadback.Holding<Tile.Tile8>) ->
  {Model.content_layout(~Tile.Tile8,completed_content(Products.compute_rectangle(~GemmBuffer.point_keeping,rectangle,input_base,weight_base,
    Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,bias),MoveReadback.pack(~Tile.Tile8,weights)))) == clipped_rectangle(rectangle) : Pieces.Ranges}:
  +pairs = held_pairs_model(rectangle,weight_base,MoveReadback.pack(~Tile.Tile8,weights))
  -native = completed_content(Products.compute_rectangle(~GemmBuffer.point_keeping,rectangle,input_base,weight_base,
    Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,bias),MoveReadback.pack(~Tile.Tile8,weights)))
  -model = computed_model(~PointJobs.run,rectangle,input_base,Buffer.content(~Tile.Tile8,input),pairs,Buffer.content(~Tile.Tile8,bias))
  %Equal.sym(Model.Content<Tile.Tile8>,native,model,shared_held_contents(rectangle,input_base,weight_base,input,bias,MoveReadback.pack(~Tile.Tile8,weights))) :
    {Model.content_layout(~Tile.Tile8,_) == clipped_rectangle(rectangle) : Pieces.Ranges}
  computed_model_layout(~PointJobs.run,rectangle,input_base,Buffer.content(~Tile.Tile8,input),pairs,Buffer.content(~Tile.Tile8,bias))

def band_stage(~run: Run(),+rectangle: Products.Rectangle,left: Move.Band<Tile.Tile8>,right: Move.Band<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) -> Buffer.Content<Tile.Tile16>:
  Move.Band{left_loan,left_buffer,+input_base} = left
  Move.Band{right_loan,right_buffer,+weight_base} = right
  computed_staged(~run,rectangle,input_base,Buffer.contents(~Tile.Tile8,left_buffer),panel_pair_model(rectangle,weight_base,Buffer.remodel(~Tile.Tile8,right_buffer)),
    Buffer.content(~Tile.Tile8,bias))

def band_model(~run: Run(),+rectangle: Products.Rectangle,left: Move.Band<Tile.Tile8>,right: Move.Band<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) -> Model.Content<Tile.Tile8>:
  rectangle_model(rectangle,band_stage(~run,rectangle,left,right,bias))

def band_model_layout(~run: Run(),+rectangle: Products.Rectangle,left: Move.Band<Tile.Tile8>,right: Move.Band<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  {Model.content_layout(~Tile.Tile8,band_model(~run,rectangle,left,right,bias)) == clipped_rectangle(rectangle) : Pieces.Ranges}:
  Move.Band{left_loan,left_buffer,+input_base} = left
  Move.Band{right_loan,right_buffer,+weight_base} = right
  computed_model_layout(~run,rectangle,input_base,Buffer.contents(~Tile.Tile8,left_buffer),panel_pair_model(rectangle,weight_base,Buffer.remodel(~Tile.Tile8,right_buffer)),Buffer.content(~Tile.Tile8,bias))

def panel_native_bands(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),+rectangle: Products.Rectangle,
  left: Move.Band<Tile.Tile8>,right: Move.Band<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  {completed_content(Products.panel_bands(~kernel,rectangle,left,right,Buffer.pack(~Tile.Tile8,bias))) == band_model(~run,rectangle,left,right,bias) : Model.Content<Tile.Tile8>}:
  Move.Band{left_loan,left_buffer,+input_base} = left
  Move.Band{right_loan,right_buffer,+weight_base} = right
  +observed_left = Buffer.observe(~Tile.Tile8,left_buffer)
  +observed_right = Buffer.observe(~Tile.Tile8,right_buffer)
  +input = Buffer.observed_model(~Tile.Tile8,left_buffer,observed_left)
  +weights = Buffer.observed_model(~Tile.Tile8,right_buffer,observed_right)
  -packed = Buffer.pack(~Tile.Tile8,input)
  -expected = panel_pair_model(rectangle,weight_base,weights)
  %Buffer.observed_equation(~Tile.Tile8,left_buffer,observed_left) :
    {completed_content(Products.panel_bands(~kernel,rectangle,Move.Band{left_loan,_,input_base},right,Buffer.pack(~Tile.Tile8,bias))) ==
      computed_model(~run,rectangle,input_base,Buffer.contents(~Tile.Tile8,_),expected,Buffer.content(~Tile.Tile8,bias)) : Model.Content<Tile.Tile8>}
  %Buffer.observed_equation(~Tile.Tile8,right_buffer,observed_right) :
    {completed_content(Products.panel_bands(~kernel,rectangle,Move.Band{left_loan,packed,input_base},Move.Band{right_loan,_,weight_base},Buffer.pack(~Tile.Tile8,bias))) ==
      computed_model(~run,rectangle,input_base,Buffer.contents(~Tile.Tile8,packed),expected,Buffer.content(~Tile.Tile8,bias)) : Model.Content<Tile.Tile8>}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,packed),Buffer.content(~Tile.Tile8,input),Buffer.packed_contents(~Tile.Tile8,input)) :
    {completed_content(Products.panel_bands(~kernel,rectangle,Move.Band{left_loan,packed,input_base},
      Move.Band{right_loan,Buffer.pack(~Tile.Tile8,weights),weight_base},Buffer.pack(~Tile.Tile8,bias))) ==
      computed_model(~run,rectangle,input_base,_,expected,Buffer.content(~Tile.Tile8,bias)) : Model.Content<Tile.Tile8>}
  panel_band_contents(~kernel,~run,~agrees,rectangle,input_base,weight_base,left_loan,right_loan,input,weights,bias)

def panel_geometry(+shape: Products.Shape,+first: U32) -> Products.Rectangle:
  Products.Shape{+positions,+inner,+channels,+grid} = shape
  Products.Rectangle{(first * positions * Gemm.blocks(channels) : U32),positions,inner,channels,False{},grid}

def panel_stage_model(~run: Run(),+shape: Products.Shape,+first: U32,input: Move.Holding<Tile.Tile8>,weights: Move.Holding<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  Buffer.Content<Tile.Tile16>:
  Products.Shape{+positions,+inner,+channels,+grid} = shape
  +in_cells = (positions * Gemm.blocks(inner) : U32)
  +weight_cells = (inner * Gemm.blocks(channels) : U32)
  band_stage(~run,panel_geometry(shape,first),
    Move.band(~Tile.Tile8,input,(first * in_cells : U32),in_cells,Products.zero(),Move.spare(~Tile.Tile8,Products.zero())),
    Move.band(~Tile.Tile8,weights,(first * weight_cells : U32),weight_cells,Products.zero(),Move.spare(~Tile.Tile8,Products.zero())),bias)

def panel_work_model(~run: Run(),+shape: Products.Shape,+first: U32,input: Move.Holding<Tile.Tile8>,weights: Move.Holding<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  Model.Content<Tile.Tile8>:
  rectangle_model(panel_geometry(shape,first),panel_stage_model(~run,shape,first,input,weights,bias))

def panel_held_contents(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),+shape: Products.Shape,+first: U32,
  input: Move.Holding<Tile.Tile8>,weights: Move.Holding<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  {completed_content(Products.panel_rectangle(~kernel,shape,first,input,weights,Buffer.pack(~Tile.Tile8,bias))) ==
    panel_work_model(~run,shape,first,input,weights,bias) : Model.Content<Tile.Tile8>}:
  Products.Shape{+positions,+inner,+channels,+grid} = shape
  +in_cells = (positions * Gemm.blocks(inner) : U32)
  +weight_cells = (inner * Gemm.blocks(channels) : U32)
  panel_native_bands(~kernel,~run,~agrees,Products.Rectangle{(first * positions * Gemm.blocks(channels) : U32),positions,inner,channels,False{},grid},
    Move.band(~Tile.Tile8,input,(first * in_cells : U32),in_cells,Products.zero(),Move.spare(~Tile.Tile8,Products.zero())),
    Move.band(~Tile.Tile8,weights,(first * weight_cells : U32),weight_cells,Products.zero(),Move.spare(~Tile.Tile8,Products.zero())),bias)

def panel_held_layout(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),+shape: Products.Shape,+first: U32,
  +input: MoveReadback.Holding<Tile.Tile8>,+weights: MoveReadback.Holding<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  {Model.content_layout(~Tile.Tile8,completed_content(Products.panel_rectangle(~kernel,shape,first,MoveReadback.pack(~Tile.Tile8,input),
    MoveReadback.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias)))) == clipped_rectangle(panel_geometry(shape,first)) : Pieces.Ranges}:
  Products.Shape{+positions,+inner,+channels,+grid} = shape
  +in_cells = (positions * Gemm.blocks(inner) : U32)
  +weight_cells = (inner * Gemm.blocks(channels) : U32)
  +rectangle = panel_geometry(shape,first)
  -native = completed_content(Products.panel_rectangle(~kernel,shape,first,MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias)))
  -model = panel_work_model(~run,shape,first,MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),bias)
  %Equal.sym(Model.Content<Tile.Tile8>,native,model,panel_held_contents(~kernel,~run,~agrees,shape,first,MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),bias)) :
    {Model.content_layout(~Tile.Tile8,_) == clipped_rectangle(rectangle) : Pieces.Ranges}
  band_model_layout(~run,rectangle,
    Move.band(~Tile.Tile8,MoveReadback.pack(~Tile.Tile8,input),(first * in_cells : U32),in_cells,Products.zero(),Move.spare(~Tile.Tile8,Products.zero())),
    Move.band(~Tile.Tile8,MoveReadback.pack(~Tile.Tile8,weights),(first * weight_cells : U32),weight_cells,Products.zero(),Move.spare(~Tile.Tile8,Products.zero())),bias)

type SharedResource is Data:
  SharedResource{input: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<Tile.Tile8>}

def Resources() -> Type:
  Products.Cells() & Products.Cells()

def SplitResources() -> Type:
  Resources() & Resources()

def pack_resource(resource: SharedResource) -> Resources():
  SharedResource{input,bias} = resource
  (Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,bias))

type ResourceContent is Data:
  ResourceContent{input: Buffer.Content<Tile.Tile8>,bias: Buffer.Content<Tile.Tile8>}

def resource_content(resource: SharedResource) -> ResourceContent:
  SharedResource{input,bias} = resource
  ResourceContent{Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile8,bias)}

type SharedCopies<-source: SharedResource> is Data:
  SharedCopies{left: SharedResource,right: SharedResource,
    exact: {(pack_resource(left),pack_resource(right)) == Products.shared_resources(pack_resource(source)) : SplitResources()},
    left_content: {resource_content(left) == resource_content(source) : ResourceContent},
    right_content: {resource_content(right) == resource_content(source) : ResourceContent}}

def observed_resources(+input: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,
  input_copies: CloneProofs.Observation<Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input))>,
  bias_copies: CloneProofs.Observation<Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias))>) -> SharedCopies<SharedResource{input,bias}>:
  CloneProofs.Observation{+left_input,+right_input,+input_equal} = input_copies
  CloneProofs.Observation{+left_bias,+right_bias,+bias_equal} = bias_copies
  +left = {SharedResource{left_input,left_bias} : SharedResource}
  +right = {SharedResource{right_input,right_bias} : SharedResource}
  +source = {SharedResource{input,bias} : SharedResource}
  -expected = {(pack_resource(left),pack_resource(right)) : SplitResources()}
  -copy_bias = Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias))
  +same = {%input_equal : {expected == Storage.bind_pair(~Products.Cells(),~Products.Cells(),~SplitResources(),_,left => right =>
      Storage.bind_pair(~Products.Cells(),~Products.Cells(),~SplitResources(),copy_bias,lower => upper => ((left,lower),(right,upper)))) : SplitResources()}
    %bias_equal : {expected == Storage.bind_pair(~Products.Cells(),~Products.Cells(),~SplitResources(),_,
      lower => upper => ((Buffer.pack(~Tile.Tile8,left_input),lower),(Buffer.pack(~Tile.Tile8,right_input),upper))) : SplitResources()}
    {==} : {expected == Products.shared_resources(pack_resource(source)) : SplitResources()}}
  SharedCopies{left,right,same,
    %CloneProofs.left_contents(~Tile.Tile8,input,left_input,right_input,input_equal) :
      {resource_content(left) == ResourceContent{_,Buffer.content(~Tile.Tile8,bias)} : ResourceContent}
    %CloneProofs.left_contents(~Tile.Tile8,bias,left_bias,right_bias,bias_equal) :
      {resource_content(left) == ResourceContent{Buffer.content(~Tile.Tile8,left_input),_} : ResourceContent}
    {==},
    %CloneProofs.right_contents(~Tile.Tile8,input,left_input,right_input,input_equal) :
      {resource_content(right) == ResourceContent{_,Buffer.content(~Tile.Tile8,bias)} : ResourceContent}
    %CloneProofs.right_contents(~Tile.Tile8,bias,left_bias,right_bias,bias_equal) :
      {resource_content(right) == ResourceContent{Buffer.content(~Tile.Tile8,right_input),_} : ResourceContent}
    {==}}

def shared_copies(+source: SharedResource) -> SharedCopies<source>:
  SharedResource{+input,+bias} = source
  observed_resources(input,bias,CloneProofs.observe_pair(~Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input))),
    CloneProofs.observe_pair(~Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias))))

def TaskOutput() -> Type:
  Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit>

def task_regions(task: TaskOutput()) -> Pieces.Ranges:
  match task:
    case Partition.Leaf{pieces,first,count,details}: Model.content_layout(~Tile.Tile8,Model.contents(~Tile.Tile8,pieces))
    case Partition.Fork{left,right}: Pieces.Split{task_regions(left),task_regions(right)}

def result_regions(~Resource: Type,result: TaskOutput() & Resource) -> Pieces.Ranges:
  (task,resource) = result
  task_regions(task)

def task_contents(task: TaskOutput()) -> Model.Content<Tile.Tile8>:
  match task:
    case Partition.Leaf{pieces,first,count,detail}: Model.contents(~Tile.Tile8,pieces)
    case Partition.Fork{left,right}: Model.Joined{task_contents(left),task_contents(right)}

def result_contents(~Resource: Type,result: TaskOutput() & Resource) -> Model.Content<Tile.Tile8>:
  (task,resource) = result
  task_contents(task)

def shared_done_contents(+first: U32,+count: U32,-bias: Products.Cells(),kept: Products.Completed) ->
  {result_contents(~Resources(),Products.shared_done(first,count,bias,kept)) == completed_content(kept) : Model.Content<Tile.Tile8>}:
  Products.Completed{pieces,input,local_bias} = kept
  {==}

def panel_done_contents(+first: U32,+count: U32,kept: Products.Completed) ->
  {result_contents(~Products.Cells(),Products.panel_done(first,count,kept)) == completed_content(kept) : Model.Content<Tile.Tile8>}:
  Products.Completed{pieces,input,bias} = kept
  {==}

def shared_done_layout(+first: U32,+count: U32,-bias: Products.Cells(),kept: Products.Completed) ->
  {result_regions(~Resources(),Products.shared_done(first,count,bias,kept)) == Model.content_layout(~Tile.Tile8,completed_content(kept)) : Pieces.Ranges}:
  Products.Completed{pieces,input,local_bias} = kept
  {==}

def shared_bias_layout(+rectangle: Products.Rectangle,+count: U32,+group: U32,+input: Buffer.Model<Tile.Tile8>,+weights: MoveReadback.Holding<Tile.Tile8>,
  biases: Resources()) ->
  {result_regions(~Resources(),Products.shared_with_bias(rectangle,count,group,Buffer.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),biases)) ==
    clipped_rectangle(rectangle) : Pieces.Ranges}:
  Products.Rectangle{+first,positions,+inner,channels,group_major,grid} = rectangle
  (bias,local_bias) = biases
  +observation = Buffer.observe(~Tile.Tile8,local_bias)
  +local = Buffer.observed_model(~Tile.Tile8,local_bias,observation)
  +weight_base = (group * inner : U32)
  %Buffer.observed_equation(~Tile.Tile8,local_bias,observation) :
    {result_regions(~Resources(),Products.shared_done(first,count,bias,Products.compute_rectangle(~GemmBuffer.point_keeping,rectangle,0,weight_base,
      Buffer.pack(~Tile.Tile8,input),_,MoveReadback.pack(~Tile.Tile8,weights)))) == clipped_rectangle(rectangle) : Pieces.Ranges}
  -computed = Products.compute_rectangle(~GemmBuffer.point_keeping,rectangle,0,weight_base,Buffer.pack(~Tile.Tile8,input),
    Buffer.pack(~Tile.Tile8,local),MoveReadback.pack(~Tile.Tile8,weights))
  %Equal.sym(Pieces.Ranges,result_regions(~Resources(),Products.shared_done(first,count,bias,computed)),
    Model.content_layout(~Tile.Tile8,completed_content(computed)),shared_done_layout(first,count,bias,
      Products.compute_rectangle(~GemmBuffer.point_keeping,rectangle,0,weight_base,Buffer.pack(~Tile.Tile8,input),
        Buffer.pack(~Tile.Tile8,local),MoveReadback.pack(~Tile.Tile8,weights)))) :
    {_ == clipped_rectangle(rectangle) : Pieces.Ranges}
  shared_held_layout(rectangle,0,weight_base,input,local,weights)

def shared_geometry(+work: Products.Shape,+first: U32,+count: U32) -> Products.Rectangle:
  Products.Shape{+positions,+inner,+channels,+grid} = work
  +group = (first / positions : U32)
  +groups = (count / positions : U32)
  Products.Rectangle{first,positions,inner,U32.min((groups * 8 : U32),(channels - group * 8 : U32)),True{},grid}

def shared_leaf_layout(+work: Products.Shape,+first: U32,+count: U32,+weights: MoveReadback.Holding<Tile.Tile8>,+resources: SharedResource) ->
  {result_regions(~Resources(),Products.shared_leaf(work,first,count,MoveReadback.pack(~Tile.Tile8,weights),pack_resource(resources))) ==
    clipped_rectangle(shared_geometry(work,first,count)) : Pieces.Ranges}:
  Products.Shape{+positions,+inner,+channels,+grid} = work
  SharedResource{+input,+bias} = resources
  +group = (first / positions : U32)
  +groups = (count / positions : U32)
  shared_bias_layout(shared_geometry(work,first,count),count,group,input,weights,
    Copy.range(~Tile.Tile8,groups,group,0,Buffer.pack(~Tile.Tile8,bias),Tensors.buffer(~Tile.Tile8,Products.zero(),groups)))

def panel_done_layout(+first: U32,+count: U32,kept: Products.Completed) ->
  {result_regions(~Products.Cells(),Products.panel_done(first,count,kept)) == Model.content_layout(~Tile.Tile8,completed_content(kept)) : Pieces.Ranges}:
  Products.Completed{pieces,input,bias} = kept
  {==}

def panel_leaf_layout(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),+shape: Products.Shape,+first: U32,+count: U32,
  +input: MoveReadback.Holding<Tile.Tile8>,+weights: MoveReadback.Holding<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  {result_regions(~Products.Cells(),Products.panel_leaf(~kernel,shape,first,count,Pieces.Both{MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights)},
    Buffer.pack(~Tile.Tile8,bias))) == clipped_rectangle(panel_geometry(shape,first)) : Pieces.Ranges}:
  -computed = Products.panel_rectangle(~kernel,shape,first,MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias))
  %Equal.sym(Pieces.Ranges,result_regions(~Products.Cells(),Products.panel_done(first,count,computed)),Model.content_layout(~Tile.Tile8,completed_content(computed)),
    panel_done_layout(first,count,Products.panel_rectangle(~kernel,shape,first,MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias)))) :
    {_ == clipped_rectangle(panel_geometry(shape,first)) : Pieces.Ranges}
  panel_held_layout(~kernel,~run,~agrees,shape,first,input,weights,bias)

def shared_bias_buffer(biases: Resources()) -> Buffer.Model<Tile.Tile8>:
  (whole,local) = biases
  Buffer.remodel(~Tile.Tile8,local)

def shared_bias_projection(biases: Resources()) ->
  {shared_bias_buffer(biases) == Buffer.remodel(~Tile.Tile8,Pair.snd(Products.Cells(),Products.Cells(),biases)) : Buffer.Model<Tile.Tile8>}:
  (whole,local) = biases
  {==}

def shared_bias_at(+resources: SharedResource,+positions: U32,+first: U32,+count: U32) -> Buffer.Model<Tile.Tile8>:
  SharedResource{input,+bias} = resources
  +group = (first / positions : U32)
  +groups = (count / positions : U32)
  shared_bias_buffer(Copy.range(~Tile.Tile8,groups,group,0,Buffer.pack(~Tile.Tile8,bias),Tensors.buffer(~Tile.Tile8,Products.zero(),groups)))

def shared_weight_base(+rectangle: Products.Rectangle,+group: U32) -> U32:
  Products.Rectangle{first,positions,+inner,channels,major,grid} = rectangle
  (group * inner : U32)

def shared_held_pairs(+rectangle: Products.Rectangle,+base: U32,+weights: MoveReadback.Holding<Tile.Tile8>) -> Buffer.Content<Tile.Tile16>:
  held_pairs_model(rectangle,base,MoveReadback.pack(~Tile.Tile8,weights))

def shared_leaf_model(+work: Products.Shape,+first: U32,+count: U32,+weights: MoveReadback.Holding<Tile.Tile8>,+resources: SharedResource) -> Model.Content<Tile.Tile8>:
  Products.Shape{+positions,+inner,+channels,+grid} = work
  SharedResource{+input,+bias} = resources
  +rectangle = shared_geometry(work,first,count)
  +base = (first / positions * inner : U32)
  computed_model(~PointJobs.run,rectangle,0,Buffer.content(~Tile.Tile8,input),
    shared_held_pairs(rectangle,base,weights),Buffer.content(~Tile.Tile8,shared_bias_at(resources,positions,first,count)))

def shared_bias_contents(+rectangle: Products.Rectangle,+count: U32,+group: U32,+input: Buffer.Model<Tile.Tile8>,+weights: MoveReadback.Holding<Tile.Tile8>,
  biases: Resources()) ->
  {result_contents(~Resources(),Products.shared_with_bias(rectangle,count,group,Buffer.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),biases)) ==
    computed_model(~PointJobs.run,rectangle,0,Buffer.content(~Tile.Tile8,input),
      shared_held_pairs(rectangle,shared_weight_base(rectangle,group),weights),Buffer.content(~Tile.Tile8,shared_bias_buffer(biases))) : Model.Content<Tile.Tile8>}:
  Products.Rectangle{+first,+positions,+inner,+channels,+group_major,+grid} = rectangle
  (whole,local) = biases
  +observation = Buffer.observe(~Tile.Tile8,local)
  +model = Buffer.observed_model(~Tile.Tile8,local,observation)
  +base = (group * inner : U32)
  +wanted = computed_model(~PointJobs.run,rectangle,0,Buffer.content(~Tile.Tile8,input),
    shared_held_pairs(rectangle,base,weights),Buffer.content(~Tile.Tile8,model))
  %Buffer.observed_equation(~Tile.Tile8,local,observation) :
    {result_contents(~Resources(),Products.shared_with_bias(rectangle,count,group,Buffer.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),(whole,_))) == wanted : Model.Content<Tile.Tile8>}
  -computed = Products.compute_rectangle(~GemmBuffer.point_keeping,rectangle,0,base,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,model),MoveReadback.pack(~Tile.Tile8,weights))
  Equal.trans(Model.Content<Tile.Tile8>,result_contents(~Resources(),Products.shared_done(first,count,whole,computed)),completed_content(computed),wanted,
    shared_done_contents(first,count,whole,Products.compute_rectangle(~GemmBuffer.point_keeping,rectangle,0,base,Buffer.pack(~Tile.Tile8,input),
      Buffer.pack(~Tile.Tile8,model),MoveReadback.pack(~Tile.Tile8,weights))),
    shared_held_contents(rectangle,0,base,input,model,MoveReadback.pack(~Tile.Tile8,weights)))

def shared_leaf_contents(+work: Products.Shape,+first: U32,+count: U32,+weights: MoveReadback.Holding<Tile.Tile8>,+resources: SharedResource) ->
  {result_contents(~Resources(),Products.shared_leaf(work,first,count,MoveReadback.pack(~Tile.Tile8,weights),pack_resource(resources))) ==
    shared_leaf_model(work,first,count,weights,resources) : Model.Content<Tile.Tile8>}:
  Products.Shape{+positions,+inner,+channels,+grid} = work
  SharedResource{+input,+bias} = resources
  +group = (first / positions : U32)
  +groups = (count / positions : U32)
  shared_bias_contents(shared_geometry(work,first,count),count,group,input,weights,
    Copy.range(~Tile.Tile8,groups,group,0,Buffer.pack(~Tile.Tile8,bias),Tensors.buffer(~Tile.Tile8,Products.zero(),groups)))

def panel_leaf_contents(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),+shape: Products.Shape,+first: U32,+count: U32,
  +input: MoveReadback.Holding<Tile.Tile8>,+weights: MoveReadback.Holding<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  {result_contents(~Products.Cells(),Products.panel_leaf(~kernel,shape,first,count,Pieces.Both{MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights)},
    Buffer.pack(~Tile.Tile8,bias))) == panel_work_model(~run,shape,first,MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),bias) : Model.Content<Tile.Tile8>}:
  -computed = Products.panel_rectangle(~kernel,shape,first,MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias))
  Equal.trans(Model.Content<Tile.Tile8>,result_contents(~Products.Cells(),Products.panel_done(first,count,computed)),completed_content(computed),
    panel_work_model(~run,shape,first,MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),bias),
    panel_done_contents(first,count,Products.panel_rectangle(~kernel,shape,first,MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias))),
    panel_held_contents(~kernel,~run,~agrees,shape,first,MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights),bias))

def stored_bias(state: PointBuffer.PointContents) -> Buffer.Content<Tile.Tile8>:
  PointBuffer.PointContents{bias,input,pairs,staged} = state
  bias

def BiasStep(~run: Run()) -> Type:
  @+matrix: Gemm.PointMatrix -> @+state: PointBuffer.PointContents ->
    {stored_bias(block_step(~run,matrix,state)) == stored_bias(state) : Buffer.Content<Tile.Tile8>}

def blocks_bias(~run: Run(),~keeps: BiasStep(~run),remaining: Nat,+rectangle: Products.Rectangle,+input_base: U32,+first: U32,+span: U32,
  +state: PointBuffer.PointContents) ->
  {stored_bias(blocks_model(~run,remaining,rectangle,input_base,first,span,state)) == stored_bias(state) : Buffer.Content<Tile.Tile8>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Products.Rectangle{output_first,+positions,+inner,+channels,group_major,grid} = rectangle
      +rows = U32.min(span,(positions - first : U32))
      +groups = Gemm.blocks(inner)
      +end = (first + rows : U32)
      +matrix = {Gemm.PointMatrix{Gemm.TileMatrix{channels,rows,inner,first,end},(input_base + first * groups : U32),groups} : Gemm.PointMatrix}
      +next = block_step(~run,matrix,state)
      Equal.trans(Buffer.Content<Tile.Tile8>,stored_bias(blocks_model(~run,rest,rectangle,input_base,end,span,next)),stored_bias(next),stored_bias(state),
        blocks_bias(~run,~keeps,rest,rectangle,input_base,end,span,next),keeps(matrix,state))

def completed_bias(finished: Products.Completed) -> Buffer.Content<Tile.Tile8>:
  Products.Completed{pieces,input,bias} = finished
  Buffer.contents(~Tile.Tile8,bias)

def emitted_bias(+rectangle: Products.Rectangle,kept: GemmBuffer.PointKept) ->
  {completed_bias(Products.emitted(rectangle,kept)) == stored_bias(PointBuffer.kept_contents(kept)) : Buffer.Content<Tile.Tile8>}:
  Products.Rectangle{first,positions,inner,channels,group_major,grid} = rectangle
  GemmBuffer.PointKept{bias,input,pairs,staged} = kept
  {==}

def computed_bias(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),~keeps: BiasStep(~run),
  +rectangle: Products.Rectangle,+input_base: U32,input: Products.Cells(),+bias: Buffer.Model<Tile.Tile8>,buffers: Products.Cells() & Stage()) ->
  {completed_bias(Products.computed(~kernel,rectangle,input_base,input,Buffer.pack(~Tile.Tile8,bias),buffers)) == Buffer.content(~Tile.Tile8,bias) : Buffer.Content<Tile.Tile8>}:
  Products.Rectangle{first,+positions,+inner,+channels,group_major,grid} = rectangle
  (weights,pairs) = buffers
  +span = Products.block_positions(inner)
  +count = (Gemm.pairs_of(channels) * positions : U32)
  +remaining = U32.to_nat(Grid.divided_up(positions,span))
  +fresh = Buffer.remodel(~Tile.Tile16,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),count))
  +input_observed = Buffer.observe(~Tile.Tile8,input)
  +pair_observed = Buffer.observe(~Tile.Tile16,pairs)
  +input_model = Buffer.observed_model(~Tile.Tile8,input,input_observed)
  +pair_model = Buffer.observed_model(~Tile.Tile16,pairs,pair_observed)
  +initial = {PointBuffer.PointKeptModel{bias,input_model,pair_model,fresh} : PointBuffer.PointKeptModel}
  -kept = Products.rect_blocks(~kernel,remaining,rectangle,input_base,0,span,PointBuffer.pack_point_kept(initial))
  +final_state = blocks_model(~run,remaining,rectangle,input_base,0,span,PointBuffer.model_contents(initial))
  +target = Buffer.content(~Tile.Tile8,bias)
  %Buffer.observed_equation(~Tile.Tile8,input,input_observed) :
    {completed_bias(Products.computed(~kernel,rectangle,input_base,_,Buffer.pack(~Tile.Tile8,bias),(weights,pairs))) == target : Buffer.Content<Tile.Tile8>}
  %Buffer.observed_equation(~Tile.Tile16,pairs,pair_observed) :
    {completed_bias(Products.computed(~kernel,rectangle,input_base,Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,bias),(weights,_))) == target : Buffer.Content<Tile.Tile8>}
  %Buffer.remodeled(~Tile.Tile16,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),count)) :
    {completed_bias(Products.emitted(rectangle,Products.rect_blocks(~kernel,remaining,rectangle,input_base,0,span,
      GemmBuffer.PointKept{Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile16,pair_model),_}))) == target : Buffer.Content<Tile.Tile8>}
  %Equal.sym(Buffer.Content<Tile.Tile8>,completed_bias(Products.emitted(rectangle,kept)),stored_bias(PointBuffer.kept_contents(kept)),
    emitted_bias(rectangle,Products.rect_blocks(~kernel,remaining,rectangle,input_base,0,span,PointBuffer.pack_point_kept(initial)))) :
    {_ == target : Buffer.Content<Tile.Tile8>}
  %Equal.sym(PointBuffer.PointContents,PointBuffer.kept_contents(kept),final_state,blocks_contents(~kernel,~run,~agrees,remaining,rectangle,input_base,0,span,initial)) :
    {stored_bias(_) == target : Buffer.Content<Tile.Tile8>}
  blocks_bias(~run,~keeps,remaining,rectangle,input_base,0,span,PointBuffer.model_contents(initial))

def panel_band_bias(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),~keeps: BiasStep(~run),
  +rectangle: Products.Rectangle,input: Move.Band<Tile.Tile8>,right: Move.Band<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  {completed_bias(Products.panel_bands(~kernel,rectangle,input,right,Buffer.pack(~Tile.Tile8,bias))) == Buffer.content(~Tile.Tile8,bias) : Buffer.Content<Tile.Tile8>}:
  Products.Rectangle{first,positions,+inner,+channels,group_major,grid} = rectangle
  Move.Band{input_loan,cells,+input_base} = input
  Move.Band{weight_loan,weights,+weight_base} = right
  computed_bias(~kernel,~run,~agrees,~keeps,rectangle,input_base,cells,bias,
    GemmBuffer.panel_pairs(channels,inner,weight_base,weights,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),(Gemm.pairs_of(channels) * inner : U32))))

def returned_bias(result: TaskOutput() & Products.Cells()) -> Buffer.Content<Tile.Tile8>:
  (pieces,bias) = result
  Buffer.contents(~Tile.Tile8,bias)

def done_bias(+first: U32,+count: U32,output: Products.Completed) ->
  {returned_bias(Products.panel_done(first,count,output)) == completed_bias(output) : Buffer.Content<Tile.Tile8>}:
  Products.Completed{pieces,input,bias} = output
  {==}

def panel_leaf_bias(~kernel: Products.Kernel(),~run: Run(),~agrees: BlockAgreement(~kernel,~run),~keeps: BiasStep(~run),
  +work: Products.Shape,+first: U32,+count: U32,+input: MoveReadback.Holding<Tile.Tile8>,+weights: MoveReadback.Holding<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  {returned_bias(Products.panel_leaf(~kernel,work,first,count,Pieces.Both{MoveReadback.pack(~Tile.Tile8,input),MoveReadback.pack(~Tile.Tile8,weights)},Buffer.pack(~Tile.Tile8,bias))) ==
    Buffer.content(~Tile.Tile8,bias) : Buffer.Content<Tile.Tile8>}:
  Products.Shape{+positions,+inner,+channels,+grid} = work
  +in_width = (positions * Gemm.blocks(inner) : U32)
  +out_width = (inner * Gemm.blocks(channels) : U32)
  +rect = panel_geometry(work,first)
  -left = Move.band(~Tile.Tile8,MoveReadback.pack(~Tile.Tile8,input),(first * in_width : U32),in_width,Products.zero(),Move.spare(~Tile.Tile8,Products.zero()))
  -right = Move.band(~Tile.Tile8,MoveReadback.pack(~Tile.Tile8,weights),(first * out_width : U32),out_width,Products.zero(),Move.spare(~Tile.Tile8,Products.zero()))
  -output = Products.panel_bands(~kernel,rect,left,right,Buffer.pack(~Tile.Tile8,bias))
  Equal.trans(Buffer.Content<Tile.Tile8>,returned_bias(Products.panel_done(first,count,output)),completed_bias(output),Buffer.content(~Tile.Tile8,bias),
    done_bias(first,count,Products.panel_bands(~kernel,rect,
      Move.band(~Tile.Tile8,MoveReadback.pack(~Tile.Tile8,input),(first * in_width : U32),in_width,Products.zero(),Move.spare(~Tile.Tile8,Products.zero())),
      Move.band(~Tile.Tile8,MoveReadback.pack(~Tile.Tile8,weights),(first * out_width : U32),out_width,Products.zero(),Move.spare(~Tile.Tile8,Products.zero())),Buffer.pack(~Tile.Tile8,bias))),
    panel_band_bias(~kernel,~run,~agrees,~keeps,rect,
      Move.band(~Tile.Tile8,MoveReadback.pack(~Tile.Tile8,input),(first * in_width : U32),in_width,Products.zero(),Move.spare(~Tile.Tile8,Products.zero())),
      Move.band(~Tile.Tile8,MoveReadback.pack(~Tile.Tile8,weights),(first * out_width : U32),out_width,Products.zero(),Move.spare(~Tile.Tile8,Products.zero())),bias))
