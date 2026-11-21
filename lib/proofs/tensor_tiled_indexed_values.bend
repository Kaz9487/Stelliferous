# The function of the row-major index and the value at every lane of a tiled
# tensor. A leaf writes a new piece from its input, lane l of the cell of
# group g at outer index o holding the function of the index of channel
# 8 * g + l at o, and puts +0 back past the lanes.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../machine_limits.bend as Machine
import ../storage_buffer.bend as Storage
import ../storage_allocation.bend as Allocation
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_operations.bend as Operations
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../tensor_tiled_indexed.bend as TiledIndexed
import ../tensor_tiled_map.bend as TiledMap
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../traversal_blocks.bend as TraversalBlocks
import ./storage_certificate.bend as Certificate
import ./traversal_certificate.bend as TraversalCertificate
import ./tensor_tiled_indexed_certificate.bend as TiledIndexedCertificate
import ./storage_refinement.bend as StorageLaws
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_buffer_model.bend as Buffer
import ./storage_copy_refinement.bend as StorageCopyLaws
import ./storage_tree_model.bend as Tree
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_access_refinement.bend as AccessLaws
import ./traversal_indexed_writes.bend as TraversalIndexedWrites
import ./write_program.bend as Writes
import ./tensor_tiled_padding.bend as Padding
import ../default_machine_limits.bend as Default
import ../tensor_tasks.bend as Tasks
import ../tensor_pieces.bend as Pieces
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_model.bend as Model
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as WordMultiplication
import ./u32_division.bend as U32Division
import ./word_encoding.bend as WordEncoding
import ./tensor_allocation_depth.bend as Depth
import ./storage_shape.bend as Shape
import ./word_power_of_two.bend as Power
import ./tensor_move_readback.bend as MoveReadback

# ---- The traversal of a leaf ----

# The tile a leaf writes at a position of its piece.
def sample(~operation: U32 -> F32 -> F32,input: Tree.Tree<Tile.Tile8>,context: TiledIndexed.Indexing,position: U32) -> Tile.Tile8:
  +context = context
  +position = position
  TiledIndexed.lanes(~operation,TiledIndexed.lane_index(context,position),TiledIndexed.lane_step(context),Tree.read(Tile.Tile8,input,position))

def sampled(~operation: U32 -> F32 -> F32,+input: Tree.Tree<Tile.Tile8>,+context: TiledIndexed.Indexing,+position: U32) ->
  {TiledIndexed.read(~operation,Tree.pack(Tile.Tile8,input),context,position) == (Tree.pack(Tile.Tile8,input),sample(~operation,input,context,position)) : Array<Tile.Tile8> & Tile.Tile8}:
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,input),position),
    (Tree.pack(Tile.Tile8,input),Tree.read(Tile.Tile8,input,position)),StorageLaws.read(Tile.Tile8,input,position)) :
    {TiledIndexed.cell(~operation,context,position,_) == (Tree.pack(Tile.Tile8,input),sample(~operation,input,context,position)) : Array<Tile.Tile8> & Tile.Tile8}
  {==}

def state(~operation: U32 -> F32 -> F32,count: Nat,context: TiledIndexed.Indexing,input: Tree.Tree<Tile.Tile8>,output: Tree.Tree<Tile.Tile8>) ->
  Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,TiledIndexed.Indexing>:
  Loop.run(~Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,TiledIndexed.Indexing>,
    ~ArrayLaws.model_step(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~sample(~operation),~TraversalBlocks.in_place(~TiledIndexed.Indexing)),
    count,Traversal.State{input,output,0,context})

def run(~operation: U32 -> F32 -> F32,+count: Nat,+context: TiledIndexed.Indexing,+input: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>) ->
  {TiledIndexed.run(~operation,TiledIndexed.Call{count,context},Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output)) ==
    AccessLaws.pack_result(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~ArrayLaws.pack_tree(~Tile.Tile8),
      AccessLaws.finish_model(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,state(~operation,count,context,input,output))) : Array<Tile.Tile8> & Array<Tile.Tile8>}:
  AccessLaws.finish_relation(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~TiledIndexed.Indexing,~ArrayLaws.pack_tree(~Tile.Tile8),
    ~ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~ArrayLaws.pack_tree(~Tile.Tile8)),
    ~AccessLaws.indexed_finished(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~ArrayLaws.pack_tree(~Tile.Tile8)),
    Traversal.run(~Array<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~TiledIndexed.read(~operation),~TraversalBlocks.in_place(~TiledIndexed.Indexing),count,
      Traversal.State{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output),0,context}),
    state(~operation,count,context,input,output),
    ArrayLaws.traversal_refines_model(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~ArrayLaws.pack_tree(~Tile.Tile8),
      ~TiledIndexed.read(~operation),~sample(~operation),~TraversalBlocks.in_place(~TiledIndexed.Indexing),
      ~sampled(~operation),count,Traversal.State{input,output,0,context}))

def writes(~operation: U32 -> F32 -> F32,count: Nat,context: TiledIndexed.Indexing,input: Tree.Tree<Tile.Tile8>) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  TraversalIndexedWrites.writes(~Tile.Tile8,~TraversalIndexedWrites.Input<Tree.Tree<Tile.Tile8>,TiledIndexed.Indexing>,
    ~TraversalIndexedWrites.value(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~sample(~operation)),count,0,TraversalIndexedWrites.Input{input,context})

def run_writes(~operation: U32 -> F32 -> F32,+count: Nat,+context: TiledIndexed.Indexing,+input: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>) ->
  {TiledIndexed.run(~operation,TiledIndexed.Call{count,context},Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output)) ==
    (Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,Writes.pure(Tile.Tile8,writes(~operation,count,context,input),output))) : Array<Tile.Tile8> & Array<Tile.Tile8>}:
  Equal.trans(Array<Tile.Tile8> & Array<Tile.Tile8>,TiledIndexed.run(~operation,TiledIndexed.Call{count,context},Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output)),
    AccessLaws.pack_result(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~ArrayLaws.pack_tree(~Tile.Tile8),
      AccessLaws.finish_model(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,state(~operation,count,context,input,output))),
    (Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,Writes.pure(Tile.Tile8,writes(~operation,count,context,input),output))),run(~operation,count,context,input,output),
    Equal.cong(AccessLaws.ModelResult<Tree.Tree<Tile.Tile8>,Tile.Tile8>,Array<Tile.Tile8> & Array<Tile.Tile8>,
      result => AccessLaws.pack_result(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~ArrayLaws.pack_tree(~Tile.Tile8),result),
      AccessLaws.finish_model(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,state(~operation,count,context,input,output)),
      AccessLaws.ModelResult{input,Writes.pure(Tile.Tile8,writes(~operation,count,context,input),output)},
      TraversalIndexedWrites.loop(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~sample(~operation),count,0,input,context,output)))

# ---- The buffers of a leaf ----

# The written output of a leaf.
def buffer_model(~operation: U32 -> F32 -> F32,+count: Nat,+context: TiledIndexed.Indexing,input: Buffer.Content<Tile.Tile8>,output: Buffer.Content<Tile.Tile8>) ->
  Buffer.Content<Tile.Tile8>:
  Buffer.Content{input,input_length} = input
  Buffer.Content{output,output_length} = output
  Buffer.Content{Writes.pure(Tile.Tile8,writes(~operation,count,context,input),output),output_length}

def owners_output(owners: StorageCopyLaws.Owners<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  StorageCopyLaws.Owners{source,source_length,output,output_length} = owners
  Buffer.Content{output,output_length}

def pair_output(+input_length: Nat,+output_length: Nat,result: Array<Tile.Tile8> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile8,Certificate.family(~Tile.Tile8),result>) ->
  {Buffer.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Operations.pair_buffers(~Tile.Tile8,input_length,output_length,result,certificates))) ==
    owners_output(StorageCopyLaws.values(~Tile.Tile8,input_length,output_length,result)) : Buffer.Content<Tile.Tile8>}:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  {==}

def buffers(~operation: U32 -> F32 -> F32,+count: Nat,+context: TiledIndexed.Indexing,+input: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.indexed_buffers(~operation,TiledIndexed.Call{count,context},Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,output))) ==
    buffer_model(~operation,count,context,Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile8,output)) : Buffer.Content<Tile.Tile8>}:
  Buffer.Model{+source,+input_length,+input_certificate} = input
  Buffer.Model{+target,+output_length,+output_certificate} = output
  +program = writes(~operation,count,context,source)
  %Equal.sym(Buffer.Content<Tile.Tile8>,
    Buffer.contents(~Tile.Tile8,TiledMap.indexed_buffers(~operation,TiledIndexed.Call{count,context},Buffer.pack(~Tile.Tile8,Buffer.Model{source,input_length,input_certificate}),
      Buffer.pack(~Tile.Tile8,Buffer.Model{target,output_length,output_certificate}))),
    owners_output(StorageCopyLaws.values(~Tile.Tile8,input_length,output_length,TiledIndexed.run(~operation,TiledIndexed.Call{count,context},Tree.pack(Tile.Tile8,source),Tree.pack(Tile.Tile8,target)))),
    pair_output(input_length,output_length,TiledIndexed.run(~operation,TiledIndexed.Call{count,context},Tree.pack(Tile.Tile8,source),Tree.pack(Tile.Tile8,target)),
      TiledIndexedCertificate.certificates(~operation,TiledIndexed.Call{count,context},Tree.pack(Tile.Tile8,source),Tree.pack(Tile.Tile8,target),input_certificate,output_certificate))) :
    {_ == buffer_model(~operation,count,context,Buffer.Content{source,input_length},Buffer.Content{target,output_length}) : Buffer.Content<Tile.Tile8>}
  %Equal.sym(Array<Tile.Tile8> & Array<Tile.Tile8>,TiledIndexed.run(~operation,TiledIndexed.Call{count,context},Tree.pack(Tile.Tile8,source),Tree.pack(Tile.Tile8,target)),(Tree.pack(Tile.Tile8,source),Tree.pack(Tile.Tile8,Writes.pure(Tile.Tile8,program,target))),
    run_writes(~operation,count,context,source,target)) :
    {owners_output(StorageCopyLaws.values(~Tile.Tile8,input_length,output_length,_)) ==
      buffer_model(~operation,count,context,Buffer.Content{source,input_length},Buffer.Content{target,output_length}) : Buffer.Content<Tile.Tile8>}
  Equal.cong(StorageCopyLaws.Owners<Tile.Tile8>,Buffer.Content<Tile.Tile8>,owners => owners_output(owners),
    StorageCopyLaws.values(~Tile.Tile8,input_length,output_length,(Tree.pack(Tile.Tile8,source),Tree.pack(Tile.Tile8,Writes.pure(Tile.Tile8,program,target)))),
    StorageCopyLaws.Owners{source,input_length,Writes.pure(Tile.Tile8,program,target),output_length},
    StorageCopyLaws.represented_values(~Tile.Tile8,source,input_length,Writes.pure(Tile.Tile8,program,target),output_length))

type Leaf is Data:
  Leaf{count: Nat,context: TiledIndexed.Indexing,input: Buffer.Model<Tile.Tile8>}

def LeafBuffers(~operation: U32 -> F32 -> F32,leaf: Leaf,-output: Storage.Buffer<Tile.Tile8>) -> Type:
  Leaf{+count,+context,+input} = leaf
  {Buffer.contents(~Tile.Tile8,TiledMap.indexed_buffers(~operation,TiledIndexed.Call{count,context},Buffer.pack(~Tile.Tile8,input),output)) ==
    buffer_model(~operation,count,context,Buffer.content(~Tile.Tile8,input),Buffer.contents(~Tile.Tile8,output)) : Buffer.Content<Tile.Tile8>}

# The output of a leaf, whatever buffer it writes into.
def buffers_native(~operation: U32 -> F32 -> F32,+count: Nat,+context: TiledIndexed.Indexing,+input: Buffer.Model<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.indexed_buffers(~operation,TiledIndexed.Call{count,context},Buffer.pack(~Tile.Tile8,input),output)) ==
    buffer_model(~operation,count,context,Buffer.content(~Tile.Tile8,input),Buffer.contents(~Tile.Tile8,output)) : Buffer.Content<Tile.Tile8>}:
  Buffer.lift(~Tile.Tile8,~Leaf,~LeafBuffers(~operation),Leaf{count,context,input},output,Buffer.observe(~Tile.Tile8,output),output =>
    +output = output
    %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output)),Buffer.content(~Tile.Tile8,output),Buffer.packed_contents(~Tile.Tile8,output)) :
      {Buffer.contents(~Tile.Tile8,TiledMap.indexed_buffers(~operation,TiledIndexed.Call{count,context},Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,output))) ==
        buffer_model(~operation,count,context,Buffer.content(~Tile.Tile8,input),_) : Buffer.Content<Tile.Tile8>}
    buffers(~operation,count,context,input,output))

# ---- The leaf of an indexed map ----

# What a leaf keeps of its piece: the written cells of the piece, then the
# padding.
def indexing_of(+context: TiledMap.Indexed,+first: U32) -> TiledIndexed.Indexing:
  TiledMap.Indexed{+after,+padding} = context
  TiledMap.Padding{+lanes,+groups,+outer,+position} = padding
  TiledIndexed.Indexing{outer,after,lanes,groups,position,first}

def padding_of(+context: TiledMap.Indexed) -> TiledMap.Padding:
  TiledMap.Indexed{+after,+padding} = context
  padding

def fresh(+count: U32) -> Buffer.Content<Tile.Tile8>:
  Buffer.Content{Tree.fresh(Tile.Tile8,Storage.allocation_depth(U32.max(count,1)),Tiled.zero_tile()),U32.to_nat(count)}

def written(~operation: U32 -> F32 -> F32,+context: TiledMap.Indexed,+first: U32,+count: U32,input: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  buffer_model(~operation,U32.to_nat(count),indexing_of(context,first),input,fresh(count))

def step(~operation: U32 -> F32 -> F32,+context: TiledMap.Indexed,+first: U32,+count: U32,input: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  Padding.step(padding_of(context),first,count,written(~operation,context,first,count,input))

def rule_model(~operation: U32 -> F32 -> F32,+after: U32,+lanes: U32,+groups: U32,+outer: U32,+position: Grid.Position,+first: U32,+count: U32,
  +input: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.indexed_leaf(~operation,TiledMap.Indexed{after,TiledMap.Padding{lanes,groups,outer,position}},first,count,Buffer.pack(~Tile.Tile8,input))) ==
    step(~operation,TiledMap.Indexed{after,TiledMap.Padding{lanes,groups,outer,position}},first,count,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input))) :
    Buffer.Content<Tile.Tile8>}:
  +padding = {TiledMap.Padding{lanes,groups,outer,position} : TiledMap.Padding}
  +context = {TiledIndexed.Indexing{outer,after,lanes,groups,position,first} : TiledIndexed.Indexing}
  +depth = Storage.allocation_depth(U32.max(count,1))
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input)),Buffer.content(~Tile.Tile8,input),Buffer.packed_contents(~Tile.Tile8,input)) :
    {Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,count,TiledMap.indexed_buffers(~operation,TiledIndexed.Call{U32.to_nat(count),context},Buffer.pack(~Tile.Tile8,input),Allocation.allocate(~Tile.Tile8,depth,Tiled.zero_tile(),U32.to_nat(count))))) == Padding.step(padding,first,count,buffer_model(~operation,U32.to_nat(count),context,_,fresh(count))) :
      Buffer.Content<Tile.Tile8>}
  Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,count,TiledMap.indexed_buffers(~operation,TiledIndexed.Call{U32.to_nat(count),context},Buffer.pack(~Tile.Tile8,input),Allocation.allocate(~Tile.Tile8,depth,Tiled.zero_tile(),U32.to_nat(count))))),Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,TiledMap.indexed_buffers(~operation,TiledIndexed.Call{U32.to_nat(count),context},Buffer.pack(~Tile.Tile8,input),Allocation.allocate(~Tile.Tile8,depth,Tiled.zero_tile(),U32.to_nat(count))))),
    Padding.step(padding,first,count,buffer_model(~operation,U32.to_nat(count),context,Buffer.content(~Tile.Tile8,input),fresh(count))),
    Padding.rule(padding,first,count,TiledMap.indexed_buffers(~operation,TiledIndexed.Call{U32.to_nat(count),context},Buffer.pack(~Tile.Tile8,input),Allocation.allocate(~Tile.Tile8,depth,Tiled.zero_tile(),U32.to_nat(count)))),
    Equal.cong(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile8>,content => Padding.step(padding,first,count,content),
      Buffer.contents(~Tile.Tile8,TiledMap.indexed_buffers(~operation,TiledIndexed.Call{U32.to_nat(count),context},Buffer.pack(~Tile.Tile8,input),Allocation.allocate(~Tile.Tile8,depth,Tiled.zero_tile(),U32.to_nat(count)))),buffer_model(~operation,U32.to_nat(count),context,Buffer.content(~Tile.Tile8,input),fresh(count)),
      Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,TiledMap.indexed_buffers(~operation,TiledIndexed.Call{U32.to_nat(count),context},Buffer.pack(~Tile.Tile8,input),Allocation.allocate(~Tile.Tile8,depth,Tiled.zero_tile(),U32.to_nat(count)))),
        buffer_model(~operation,U32.to_nat(count),context,Buffer.content(~Tile.Tile8,input),Buffer.contents(~Tile.Tile8,Allocation.allocate(~Tile.Tile8,depth,Tiled.zero_tile(),U32.to_nat(count)))),
        buffer_model(~operation,U32.to_nat(count),context,Buffer.content(~Tile.Tile8,input),fresh(count)),
        buffers_native(~operation,U32.to_nat(count),context,input,Allocation.allocate(~Tile.Tile8,depth,Tiled.zero_tile(),U32.to_nat(count))),
        Equal.cong(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile8>,content => buffer_model(~operation,U32.to_nat(count),context,Buffer.content(~Tile.Tile8,input),content),
          Buffer.contents(~Tile.Tile8,Allocation.allocate(~Tile.Tile8,depth,Tiled.zero_tile(),U32.to_nat(count))),fresh(count),AllocationLaws.contents(~Tile.Tile8,depth,Tiled.zero_tile(),U32.to_nat(count))))))

def rule_observed(~operation: U32 -> F32 -> F32,+after: U32,+lanes: U32,+groups: U32,+outer: U32,+position: Grid.Position,+first: U32,+count: U32,
  -buffer: Storage.Buffer<Tile.Tile8>,observation: Buffer.Observation<Tile.Tile8,buffer>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.indexed_leaf(~operation,TiledMap.Indexed{after,TiledMap.Padding{lanes,groups,outer,position}},first,count,buffer)) ==
    step(~operation,TiledMap.Indexed{after,TiledMap.Padding{lanes,groups,outer,position}},first,count,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}:
  Buffer.Observation{+model,+equation} = observation
  %equation : {Buffer.contents(~Tile.Tile8,TiledMap.indexed_leaf(~operation,TiledMap.Indexed{after,TiledMap.Padding{lanes,groups,outer,position}},first,count,_)) ==
    step(~operation,TiledMap.Indexed{after,TiledMap.Padding{lanes,groups,outer,position}},first,count,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}
  %equation : {Buffer.contents(~Tile.Tile8,TiledMap.indexed_leaf(~operation,TiledMap.Indexed{after,TiledMap.Padding{lanes,groups,outer,position}},first,count,Buffer.pack(~Tile.Tile8,model))) ==
    step(~operation,TiledMap.Indexed{after,TiledMap.Padding{lanes,groups,outer,position}},first,count,Buffer.contents(~Tile.Tile8,_)) : Buffer.Content<Tile.Tile8>}
  rule_model(~operation,after,lanes,groups,outer,position,first,count,model)

# The leaf of an indexed map writes the step of its piece.
def rule(~operation: U32 -> F32 -> F32,+context: TiledMap.Indexed,+first: U32,+count: U32,buffer: Storage.Buffer<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.indexed_leaf(~operation,context,first,count,buffer)) == step(~operation,context,first,count,Buffer.contents(~Tile.Tile8,buffer)) :
    Buffer.Content<Tile.Tile8>}:
  TiledMap.Indexed{+after,+padding} = context
  TiledMap.Padding{+lanes,+groups,+outer,+position} = padding
  rule_observed(~operation,after,lanes,groups,outer,position,first,count,buffer,Buffer.observe(~Tile.Tile8,buffer))

# ---- The index of a lane ----

# The function of the index and the value at an index.
def indexed_value(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,logical: Logical,index: U32) -> F32:
  +index = index
  operation(index,original(logical,index))

# The index of lane 0 of a cell from the quotient of its outer index, the
# lanes, its group, the after count and the rest of its outer index.
def index_form(+quotient: U32,+lanes: U32,+group: U32,+after: U32,+rest: U32) -> U32:
  ((quotient * lanes + group * 8) * after + rest : U32)

def index_natural(+quotient: Nat,+lanes: Nat,+group: Nat,+after: Nat,+rest: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,lanes),Nat.mul(group,8n)),after),rest)

def index_encoded(+quotient: Nat,+lanes: Nat,+group: Nat,+after: Nat,+rest: Nat) ->
  {index_form(U32.from_nat(quotient),U32.from_nat(lanes),U32.from_nat(group),U32.from_nat(after),U32.from_nat(rest)) ==
    U32.from_nat(index_natural(quotient,lanes,group,after,rest)) : U32}:
  +product = Nat.mul(quotient,lanes)
  +channel = Nat.mul(group,8n)
  +first = Nat.add(product,channel)
  +scaled = Nat.mul(first,after)
  +target = U32.from_nat(Nat.add(scaled,rest))
  %Equal.sym(U32,(U32.from_nat(quotient) * U32.from_nat(lanes) : U32),U32.from_nat(product),WordMultiplication.native_encoded_multiply(quotient,lanes)) :
    {((_ + U32.from_nat(group) * 8) * U32.from_nat(after) + U32.from_nat(rest) : U32) == target : U32}
  %Equal.sym(U32,(U32.from_nat(group) * 8 : U32),U32.from_nat(channel),WordMultiplication.native_encoded_multiply(group,8n)) :
    {((U32.from_nat(product) + _) * U32.from_nat(after) + U32.from_nat(rest) : U32) == target : U32}
  %Equal.sym(U32,(U32.from_nat(product) + U32.from_nat(channel) : U32),U32.from_nat(first),WordArithmetic.native_add(product,channel)) :
    {(_ * U32.from_nat(after) + U32.from_nat(rest) : U32) == target : U32}
  %Equal.sym(U32,(U32.from_nat(first) * U32.from_nat(after) : U32),U32.from_nat(scaled),WordMultiplication.native_encoded_multiply(first,after)) :
    {(_ + U32.from_nat(rest) : U32) == target : U32}
  WordArithmetic.native_add(scaled,rest)

def index_words(+quotient: U32,+lanes: U32,+group: U32,+after: U32,+rest: U32) ->
  {index_form(quotient,lanes,group,after,rest) ==
    U32.from_nat(index_natural(U32.to_nat(quotient),U32.to_nat(lanes),U32.to_nat(group),U32.to_nat(after),U32.to_nat(rest))) : U32}:
  +target = U32.from_nat(index_natural(U32.to_nat(quotient),U32.to_nat(lanes),U32.to_nat(group),U32.to_nat(after),U32.to_nat(rest)))
  %Arithmetic.native_roundtrip(quotient) : {index_form(_,lanes,group,after,rest) == target : U32}
  %Arithmetic.native_roundtrip(lanes) : {index_form(U32.from_nat(U32.to_nat(quotient)),_,group,after,rest) == target : U32}
  %Arithmetic.native_roundtrip(group) : {index_form(U32.from_nat(U32.to_nat(quotient)),U32.from_nat(U32.to_nat(lanes)),_,after,rest) == target : U32}
  %Arithmetic.native_roundtrip(after) :
    {index_form(U32.from_nat(U32.to_nat(quotient)),U32.from_nat(U32.to_nat(lanes)),U32.from_nat(U32.to_nat(group)),_,rest) == target : U32}
  %Arithmetic.native_roundtrip(rest) :
    {index_form(U32.from_nat(U32.to_nat(quotient)),U32.from_nat(U32.to_nat(lanes)),U32.from_nat(U32.to_nat(group)),U32.from_nat(U32.to_nat(after)),_) == target : U32}
  index_encoded(U32.to_nat(quotient),U32.to_nat(lanes),U32.to_nat(group),U32.to_nat(after),U32.to_nat(rest))

def index_decoded(+quotient: U32,+lanes: U32,+group: U32,+after: U32,+rest: U32,+quotient_value: Nat,+group_value: Nat,+rest_value: Nat,
  +quotient_exact: {U32.to_nat(quotient) == quotient_value : Nat},+group_exact: {U32.to_nat(group) == group_value : Nat},
  +rest_exact: {U32.to_nat(rest) == rest_value : Nat}) ->
  {index_form(quotient,lanes,group,after,rest) == U32.from_nat(index_natural(quotient_value,U32.to_nat(lanes),group_value,U32.to_nat(after),rest_value)) : U32}:
  %quotient_exact :
    {index_form(quotient,lanes,group,after,rest) == U32.from_nat(index_natural(_,U32.to_nat(lanes),group_value,U32.to_nat(after),rest_value)) : U32}
  %group_exact :
    {index_form(quotient,lanes,group,after,rest) == U32.from_nat(index_natural(U32.to_nat(quotient),U32.to_nat(lanes),_,U32.to_nat(after),rest_value)) : U32}
  %rest_exact :
    {index_form(quotient,lanes,group,after,rest) == U32.from_nat(index_natural(U32.to_nat(quotient),U32.to_nat(lanes),U32.to_nat(group),U32.to_nat(after),_)) : U32}
  index_words(quotient,lanes,group,after,rest)

# Lane k of a cell lies k after counts past lane 0.
def lane_shift(+quotient: Nat,+lanes: Nat,+group: Nat,+after: Nat,+rest: Nat,+lane: Nat) ->
  {Nat.add(index_natural(quotient,lanes,group,after,rest),Nat.mul(lane,after)) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,lanes),Nat.add(lane,Nat.mul(group,8n))),after),rest) : Nat}:
  +first = Nat.mul(quotient,lanes)
  +channel = Nat.mul(group,8n)
  +base = Nat.mul(Nat.add(first,channel),after)
  +shift = Nat.mul(lane,after)
  +target = Nat.add(Nat.mul(Nat.add(first,Nat.add(lane,channel)),after),rest)
  %Equal.sym(Nat,Nat.add(Nat.add(base,rest),shift),Nat.add(base,Nat.add(rest,shift)),Addition.associative(base,rest,shift)) : {_ == target : Nat}
  %Order.commutative(shift,rest) : {Nat.add(base,_) == target : Nat}
  %Addition.associative(base,shift,rest) : {_ == target : Nat}
  %RowMajor.distrib(Nat.add(first,channel),lane,after) : {Nat.add(_,rest) == target : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(first,channel),lane),Nat.add(first,Nat.add(channel,lane)),Addition.associative(first,channel,lane)) :
    {Nat.add(Nat.mul(_,after),rest) == target : Nat}
  %Order.commutative(lane,channel) : {Nat.add(Nat.mul(Nat.add(first,_),after),rest) == target : Nat}
  {==}

# The index word of a later lane: the index of lane 0 and a step.
def lane_index(+index: U32,+step: U32,+start: Nat,+shift: Nat,+target: Nat,+index_exact: {index == U32.from_nat(start) : U32},
  +step_exact: {step == U32.from_nat(shift) : U32},+shifted: {Nat.add(start,shift) == target : Nat}) -> {(index + step : U32) == U32.from_nat(target) : U32}:
  %shifted : {(index + step : U32) == U32.from_nat(_) : U32}
  %Equal.sym(U32,index,U32.from_nat(start),index_exact) : {(_ + step : U32) == U32.from_nat(Nat.add(start,shift)) : U32}
  %Equal.sym(U32,step,U32.from_nat(shift),step_exact) : {(U32.from_nat(start) + _ : U32) == U32.from_nat(Nat.add(start,shift)) : U32}
  WordArithmetic.native_add(start,shift)

def step_one(+step: U32,+after: Nat,+step_exact: {step == U32.from_nat(after) : U32}) -> {step == U32.from_nat(Nat.mul(1n,after)) : U32}:
  %Equal.sym(Nat,Nat.add(after,0n),after,Addition.zero_right(after)) : {step == U32.from_nat(_) : U32}
  step_exact

def step_times(+lane: Nat,+step: U32,+after: Nat,+step_exact: {step == U32.from_nat(after) : U32}) ->
  {(U32.from_nat(lane) * step : U32) == U32.from_nat(Nat.mul(lane,after)) : U32}:
  %Equal.sym(U32,step,U32.from_nat(after),step_exact) : {(U32.from_nat(lane) * _ : U32) == U32.from_nat(Nat.mul(lane,after)) : U32}
  WordMultiplication.native_encoded_multiply(lane,after)

# The function at an index word that names the logical index agrees with the
# function at that index below the lane count, and both are +0 past it.
def lane_valid(~operation: U32 -> F32 -> F32,+value: F32,valid: Bool,+index: U32,+word: U32,+same: {index == word : U32}) ->
  {Bool.pick(F32,valid,operation(index,Bool.pick(F32,valid,value,0.0)),0.0) == Bool.pick(F32,valid,operation(word,value),0.0) : F32}:
  match valid:
    case True{}:
      %Equal.sym(U32,index,word,same) : {operation(_,value) == operation(word,value) : F32}
      {==}
    case False{}: {==}

# A cell written at the index of its lane 0, stepping by the after count,
# holds the function of the index and the value at every lane.
def written_valid(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,
  +index: U32,+step: U32,
  +index_exact: {index == U32.from_nat(index_natural(Nat.div(outer,Representation.after_count(geometry)),Representation.lane_count(geometry),group,
    Representation.after_count(geometry),Nat.mod(outer,Representation.after_count(geometry)))) : U32},
  +step_exact: {step == U32.from_nat(Representation.after_count(geometry)) : U32}) ->
  PaddedCells.Valid<Representation.lane_count(geometry),group,TiledIndexed.lanes(~operation,index,step,Representation.cell(~Logical,~original,logical,geometry,outer,group)),
    Representation.cell(~Logical,~indexed_value(~operation,~Logical,~original),logical,geometry,outer,group)>:
  +after = Representation.after_count(geometry)
  +lanes = Representation.lane_count(geometry)
  +quotient = Nat.div(outer,after)
  +rest = Nat.mod(outer,after)
  +start = index_natural(quotient,lanes,group,after,rest)
  +index0 = Representation.logical_index(geometry,outer,Nat.add(0n,Nat.mul(group,8n)))
  +index1 = Representation.logical_index(geometry,outer,Nat.add(1n,Nat.mul(group,8n)))
  +index2 = Representation.logical_index(geometry,outer,Nat.add(2n,Nat.mul(group,8n)))
  +index3 = Representation.logical_index(geometry,outer,Nat.add(3n,Nat.mul(group,8n)))
  +index4 = Representation.logical_index(geometry,outer,Nat.add(4n,Nat.mul(group,8n)))
  +index5 = Representation.logical_index(geometry,outer,Nat.add(5n,Nat.mul(group,8n)))
  +index6 = Representation.logical_index(geometry,outer,Nat.add(6n,Nat.mul(group,8n)))
  +index7 = Representation.logical_index(geometry,outer,Nat.add(7n,Nat.mul(group,8n)))
  PaddedCells.Valid{
    lane_valid(~operation,original(logical,U32.from_nat(index0)),Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),lanes),index,U32.from_nat(index0),index_exact),
    lane_valid(~operation,original(logical,U32.from_nat(index1)),Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),lanes),(index + step : U32),U32.from_nat(index1),
      lane_index(index,step,start,Nat.mul(1n,after),index1,index_exact,step_one(step,after,step_exact),lane_shift(quotient,lanes,group,after,rest,1n))),
    lane_valid(~operation,original(logical,U32.from_nat(index2)),Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),lanes),(index + 2 * step : U32),U32.from_nat(index2),
      lane_index(index,(2 * step : U32),start,Nat.mul(2n,after),index2,index_exact,step_times(2n,step,after,step_exact),lane_shift(quotient,lanes,group,after,rest,2n))),
    lane_valid(~operation,original(logical,U32.from_nat(index3)),Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),lanes),(index + 3 * step : U32),U32.from_nat(index3),
      lane_index(index,(3 * step : U32),start,Nat.mul(3n,after),index3,index_exact,step_times(3n,step,after,step_exact),lane_shift(quotient,lanes,group,after,rest,3n))),
    lane_valid(~operation,original(logical,U32.from_nat(index4)),Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),lanes),(index + 4 * step : U32),U32.from_nat(index4),
      lane_index(index,(4 * step : U32),start,Nat.mul(4n,after),index4,index_exact,step_times(4n,step,after,step_exact),lane_shift(quotient,lanes,group,after,rest,4n))),
    lane_valid(~operation,original(logical,U32.from_nat(index5)),Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),lanes),(index + 5 * step : U32),U32.from_nat(index5),
      lane_index(index,(5 * step : U32),start,Nat.mul(5n,after),index5,index_exact,step_times(5n,step,after,step_exact),lane_shift(quotient,lanes,group,after,rest,5n))),
    lane_valid(~operation,original(logical,U32.from_nat(index6)),Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),lanes),(index + 6 * step : U32),U32.from_nat(index6),
      lane_index(index,(6 * step : U32),start,Nat.mul(6n,after),index6,index_exact,step_times(6n,step,after,step_exact),lane_shift(quotient,lanes,group,after,rest,6n))),
    lane_valid(~operation,original(logical,U32.from_nat(index7)),Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),lanes),(index + 7 * step : U32),U32.from_nat(index7),
      lane_index(index,(7 * step : U32),start,Nat.mul(7n,after),index7,index_exact,step_times(7n,step,after,step_exact),lane_shift(quotient,lanes,group,after,rest,7n)))}

# ---- The cell of a word ----

# The outer index and the group of a cell word as the runtime splits it.
def outer_word(position: Grid.Position,+word: U32,+outer: U32,+groups: U32) -> U32:
  match position:
    case Grid.Innermost{}: (word / U32.max(groups,1) : U32)
    case Grid.Outermost{}: (word % U32.max(outer,1) : U32)

def group_word(position: Grid.Position,+word: U32,+outer: U32,+groups: U32) -> U32:
  match position:
    case Grid.Innermost{}: (word % U32.max(groups,1) : U32)
    case Grid.Outermost{}: (word / U32.max(outer,1) : U32)

def index_split(position: Grid.Position,+word: U32,+outer: U32,+after: U32,+lanes: U32,+groups: U32) ->
  {TiledIndexed.index_at(position,word,outer,after,lanes,groups) ==
    TiledIndexed.index_of(outer_word(position,word,outer,groups),group_word(position,word,outer,groups),after,lanes) : U32}:
  match position:
    case Grid.Innermost{}: {==}
    case Grid.Outermost{}: {==}

# The outer index and the group a geometry names for a cell word.
type Decoded<-outer_word: U32,-group_word: U32,-geometry: Cells.Cells,-number: Nat> is Data:
  Decoded{outer: {U32.to_nat(outer_word) == Representation.cell_outer(geometry,number) : Nat},
    group: {U32.to_nat(group_word) == Representation.cell_group(geometry,number) : Nat}}

def divided_one(+value: Nat) -> {Nat.div(value,1n) == value : Nat}:
  %Order.times_one(value) : {Nat.div(_,1n) == value : Nat}
  Division.exact_quotient(value,1n,{==})

# Along the lane axis the runtime splits a word as the geometry does.
def decode_along(position: Grid.Position,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+word: U32,+outer: U32,+after: U32,+lanes: U32,+groups: U32,
  +outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},
  +outer_bound: {Nat.is_le(Nat.double(U32.to_nat(outer)),U32.to_nat(cap)) == True{} : Bool},
  +groups_bound: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(cap)) == True{} : Bool}) ->
  Decoded<outer_word(position,word,outer,groups),group_word(position,word,outer,groups),Cells.Cells{outer,after,lanes,groups,position},U32.to_nat(word)>:
  match position:
    case Grid.Innermost{}:
      +number = U32.to_nat(word)
      +count = U32.to_nat(groups)
      Decoded{
        %Equal.sym(U32,U32.max(groups,1),groups,Depth.positive_max(groups,groups_positive)) :
          {U32.to_nat((word / _ : U32)) == Nat.div(number,Representation.least_one(count)) : Nat}
        %Equal.sym(Nat,Representation.least_one(count),count,Representation.least_positive(count,groups_positive)) : {U32.to_nat((word / groups : U32)) == Nat.div(number,_) : Nat}
        U32Division.divide(word,groups,cap,cap_ok,groups_bound,groups_positive),
        %Equal.sym(U32,U32.max(groups,1),groups,Depth.positive_max(groups,groups_positive)) :
          {U32.to_nat((word % _ : U32)) == Nat.mod(number,Representation.least_one(count)) : Nat}
        %Equal.sym(Nat,Representation.least_one(count),count,Representation.least_positive(count,groups_positive)) : {U32.to_nat((word % groups : U32)) == Nat.mod(number,_) : Nat}
        U32Division.modulo(word,groups,cap,cap_ok,groups_bound,groups_positive)}
    case Grid.Outermost{}:
      +number = U32.to_nat(word)
      +count = U32.to_nat(outer)
      Decoded{
        %Equal.sym(U32,U32.max(outer,1),outer,Depth.positive_max(outer,outer_positive)) :
          {U32.to_nat((word % _ : U32)) == Nat.mod(number,Representation.least_one(count)) : Nat}
        %Equal.sym(Nat,Representation.least_one(count),count,Representation.least_positive(count,outer_positive)) : {U32.to_nat((word % outer : U32)) == Nat.mod(number,_) : Nat}
        U32Division.modulo(word,outer,cap,cap_ok,outer_bound,outer_positive),
        %Equal.sym(U32,U32.max(outer,1),outer,Depth.positive_max(outer,outer_positive)) :
          {U32.to_nat((word / _ : U32)) == Nat.div(number,Representation.least_one(count)) : Nat}
        %Equal.sym(Nat,Representation.least_one(count),count,Representation.least_positive(count,outer_positive)) : {U32.to_nat((word / outer : U32)) == Nat.div(number,_) : Nat}
        U32Division.divide(word,outer,cap,cap_ok,outer_bound,outer_positive)}

# With one lane every cell is its own outer index in group 0.
def decode_single(position: Grid.Position,+outer: U32,+after: U32,+word: U32,+small: {Nat.is_lt(U32.to_nat(word),U32.to_nat(outer)) == True{} : Bool}) ->
  Decoded<(word / 1 : U32),(word % 1 : U32),Cells.Cells{outer,after,1,Tiled.groups(1),position},U32.to_nat(word)>:
  match position:
    case Grid.Innermost{}:
      +number = U32.to_nat(word)
      Decoded{Equal.trans(Nat,U32.to_nat((word / 1 : U32)),number,Nat.div(number,1n),U32Division.divided_by_one(word),Equal.sym(Nat,Nat.div(number,1n),number,divided_one(number))),
        U32Division.modulo(word,1,2,{==},{==},{==})}
    case Grid.Outermost{}:
      +number = U32.to_nat(word)
      +count = U32.to_nat(outer)
      +positive = Order.positive_bound(number,count,small)
      +split = Division.small(number,count,small)
      +quotient = {Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(number,count),(0n,number),split) : {Nat.div(number,count) == 0n : Nat}}
      +remainder = {Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(number,count),(0n,number),split) : {Nat.mod(number,count) == number : Nat}}
      Decoded{
        %Equal.sym(Nat,Representation.least_one(count),count,Representation.least_positive(count,positive)) : {U32.to_nat((word / 1 : U32)) == Nat.mod(number,_) : Nat}
        Equal.trans(Nat,U32.to_nat((word / 1 : U32)),number,Nat.mod(number,count),U32Division.divided_by_one(word),Equal.sym(Nat,Nat.mod(number,count),number,remainder)),
        %Equal.sym(Nat,Representation.least_one(count),count,Representation.least_positive(count,positive)) : {U32.to_nat((word % 1 : U32)) == Nat.div(number,_) : Nat}
        %Equal.sym(Nat,Nat.div(number,count),0n,quotient) : {U32.to_nat((word % 1 : U32)) == _ : Nat}
        Equal.trans(Nat,U32.to_nat((word % 1 : U32)),Nat.mod(number,1n),0n,U32Division.modulo(word,1,2,{==},{==},{==}),PaddedCells.mod_one(number))}

# The runtime splits every cell word below the cells as the geometry of the
# values does.
def decode(+limits: Machine.Limits,+lanes: U32,+padded_outer: U32,+padded_position: Grid.Position,+outer: U32,+after: U32,+position: Grid.Position,+total: Nat,
  +word: U32,placement: Public.Placement<lanes,padded_outer,padded_position,outer,position>,
  +cells: {Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer)) == total : Nat},+inside: {Nat.is_lt(U32.to_nat(word),total) == True{} : Bool},
  +outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(Grid.groups(lanes))) == True{} : Bool},
  +outer_bound: {Nat.is_le(Nat.double(U32.to_nat(outer)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +groups_bound: {Nat.is_le(Nat.double(U32.to_nat(Grid.groups(lanes))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  Decoded<outer_word(padded_position,word,padded_outer,Grid.groups(lanes)),group_word(padded_position,word,padded_outer,Grid.groups(lanes)),
    Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position},U32.to_nat(word)>:
  match placement:
    case Public.Along{+outer_same,+position_same}:
      +groups = Grid.groups(lanes)
      +number = U32.to_nat(word)
      %Equal.sym(U32,padded_outer,outer,outer_same) :
        Decoded<outer_word(padded_position,word,_,groups),group_word(padded_position,word,_,groups),Cells.Cells{outer,after,lanes,groups,position},number>
      %Equal.sym(Grid.Position,padded_position,position,position_same) :
        Decoded<outer_word(_,word,outer,groups),group_word(_,word,outer,groups),Cells.Cells{outer,after,lanes,groups,position},number>
      decode_along(position,Machine.half_word(limits),Limits.half_word_fits(limits),word,outer,after,lanes,groups,outer_positive,groups_positive,outer_bound,groups_bound)
    case Public.Single{+one,+inner}:
      +number = U32.to_nat(word)
      +count = U32.to_nat(outer)
      +below = {%Equal.sym(Nat,Nat.mul(U32.to_nat(Grid.groups(lanes)),count),total,cells) : {Nat.is_lt(number,_) == True{} : Bool} inside :
        {Nat.is_lt(number,Nat.mul(U32.to_nat(Grid.groups(lanes)),count)) == True{} : Bool}}
      +single = {%one : {Nat.is_lt(number,Nat.mul(U32.to_nat(Grid.groups(_)),count)) == True{} : Bool} below : {Nat.is_lt(number,Nat.mul(1n,count)) == True{} : Bool}}
      +small = {%Addition.zero_right(count) : {Nat.is_lt(number,_) == True{} : Bool} single : {Nat.is_lt(number,count) == True{} : Bool}}
      %Equal.sym(Grid.Position,padded_position,Grid.Innermost{},inner) :
        Decoded<outer_word(_,word,padded_outer,Grid.groups(lanes)),group_word(_,word,padded_outer,Grid.groups(lanes)),
          Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position},number>
      %Equal.sym(U32,lanes,1,one) :
        Decoded<outer_word(Grid.Innermost{},word,padded_outer,Grid.groups(_)),group_word(Grid.Innermost{},word,padded_outer,Grid.groups(_)),
          Cells.Cells{outer,after,_,Tiled.groups(_),position},number>
      decode_single(position,outer,after,word,small)

# ---- The scope of the pieces ----

# The positive divisors of a geometry doubled within the half word: what a
# cell word needs to split along the lane axis.
type Divides<-limits: Machine.Limits,-outer: U32,-lanes: U32> is Data:
  Divides{outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},groups_positive: {Nat.is_lt(0n,U32.to_nat(Grid.groups(lanes))) == True{} : Bool},
    outer_bound: {Nat.is_le(Nat.double(U32.to_nat(outer)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
    groups_bound: {Nat.is_le(Nat.double(U32.to_nat(Grid.groups(lanes))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}

# What the after count gives a cell word: a positive divisor within the half
# word, or no cells at all.
type Spreads<-after: U32> is Data:
  Spreads{cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},positive: {Nat.is_lt(0n,U32.to_nat(after)) == True{} : Bool},
    bounded: {Nat.is_le(Nat.double(U32.to_nat(after)),U32.to_nat(cap)) == True{} : Bool}}

type AfterFacts<-total: Nat,-after: U32> is Data:
  NoCells{empty: {Nat.is_lt(0n,total) == False{} : Bool}}
  Spread{spreads: Spreads<after>}

def spread_of(+total: Nat,+after: U32,+number: Nat,+inside: {Nat.is_lt(number,total) == True{} : Bool},+facts: AfterFacts<total,after>) -> Spreads<after>:
  match facts:
    case NoCells{+empty}:
      Empty.absurd(Spreads<after>,WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(0n,total),True{},Equal.sym(Bool,Nat.is_lt(0n,total),False{},empty),
        Order.positive_bound(number,total,inside))))
    case Spread{+spreads}: spreads

def layout_after(+layout: PaddedCells.Layout) -> U32:
  PaddedCells.Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover} = layout
  after

def cells_after(+cells: Cells.Cells) -> U32:
  Cells.Cells{+outer,+after,+lanes,+groups,+position} = cells
  after

# What every piece of an indexed map reads: the logical reader, the layout of
# the cells and what the after count gives.
type Scope<-Logical: Data> is Data:
  Scope{logical: Logical,layout: PaddedCells.Layout,facts: AfterFacts<PaddedCells.layout_total(layout),layout_after(layout)>}

def itself(~Logical: Data,+scope: Scope<Logical>) -> Scope<Logical>:
  scope

def layout_of(~Logical: Data,+scope: Scope<Logical>) -> PaddedCells.Layout:
  Scope{+logical,+layout,+facts} = scope
  layout

def lifted_of(~Logical: Data,+scope: Scope<Logical>) -> Representation.Lifted<Logical>:
  Scope{+logical,+layout,+facts} = scope
  Representation.Lifted{logical,PaddedCells.layout_geometry(layout)}

def context_of(~Logical: Data,+scope: Scope<Logical>) -> TiledMap.Indexed:
  Scope{+logical,+layout,+facts} = scope
  TiledMap.Indexed{layout_after(layout),PaddedCells.layout_padding(layout)}

def total_of(~Logical: Data,+scope: Scope<Logical>) -> Nat:
  Scope{+logical,+layout,+facts} = scope
  PaddedCells.layout_total(layout)

# The cell a leaf writes at a cell word: the function of the index of every
# lane and the value of the cell there.
def written_cell(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,scope: Scope<Logical>,word: U32) -> Tile.Tile8:
  Scope{+logical,+layout,+facts} = scope
  PaddedCells.Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover} = layout
  +word = word
  +geometry = {Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position} : Cells.Cells}
  TiledIndexed.lanes(~operation,TiledIndexed.index_at(padded_position,word,padded_outer,after,lanes,Grid.groups(lanes)),after,
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word))

# ---- A written cell against the indexed reader ----

def index_exact(+padded_position: Grid.Position,+word: U32,+padded_outer: U32,+groups: U32,+outer: U32,+after: U32,+lanes: U32,+position: Grid.Position,
  +spreads: Spreads<after>,
  +decoded: Decoded<outer_word(padded_position,word,padded_outer,groups),group_word(padded_position,word,padded_outer,groups),
    Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position},U32.to_nat(word)>) ->
  {TiledIndexed.index_at(padded_position,word,padded_outer,after,lanes,groups) ==
    U32.from_nat(index_natural(Nat.div(Representation.cell_outer(Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position},U32.to_nat(word)),
      Representation.least_one(U32.to_nat(after))),U32.to_nat(lanes),Representation.cell_group(Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position},U32.to_nat(word)),
      Representation.least_one(U32.to_nat(after)),Nat.mod(Representation.cell_outer(Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position},U32.to_nat(word)),
      Representation.least_one(U32.to_nat(after))))) : U32}:
  Spreads{+cap,+cap_ok,+positive,+bounded} = spreads
  Decoded{+outer_exact,+group_exact} = decoded
  +geometry = {Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position} : Cells.Cells}
  +number = U32.to_nat(word)
  +cell = Representation.cell_outer(geometry,number)
  +group = Representation.cell_group(geometry,number)
  +count = U32.to_nat(after)
  +outer_index = outer_word(padded_position,word,padded_outer,groups)
  +group_index = group_word(padded_position,word,padded_outer,groups)
  +quotient_exact = {Equal.trans(Nat,U32.to_nat((outer_index / after : U32)),Nat.div(U32.to_nat(outer_index),count),Nat.div(cell,count),
      U32Division.divide(outer_index,after,cap,cap_ok,bounded,positive),
      Equal.cong(Nat,Nat,value => Nat.div(value,count),U32.to_nat(outer_index),cell,outer_exact)) : {U32.to_nat((outer_index / after : U32)) == Nat.div(cell,count) : Nat}}
  +rest_exact = {Equal.trans(Nat,U32.to_nat((outer_index % after : U32)),Nat.mod(U32.to_nat(outer_index),count),Nat.mod(cell,count),
      U32Division.modulo(outer_index,after,cap,cap_ok,bounded,positive),
      Equal.cong(Nat,Nat,value => Nat.mod(value,count),U32.to_nat(outer_index),cell,outer_exact)) : {U32.to_nat((outer_index % after : U32)) == Nat.mod(cell,count) : Nat}}
  +target = U32.from_nat(index_natural(Nat.div(cell,count),U32.to_nat(lanes),group,count,Nat.mod(cell,count)))
  %Equal.sym(Nat,Representation.least_one(count),count,Representation.least_positive(count,positive)) :
    {TiledIndexed.index_at(padded_position,word,padded_outer,after,lanes,groups) == U32.from_nat(index_natural(Nat.div(cell,_),U32.to_nat(lanes),group,_,Nat.mod(cell,_))) : U32}
  %Equal.sym(U32,TiledIndexed.index_at(padded_position,word,padded_outer,after,lanes,groups),TiledIndexed.index_of(outer_index,group_index,after,lanes),
    index_split(padded_position,word,padded_outer,after,lanes,groups)) : {_ == target : U32}
  %Equal.sym(U32,U32.max(after,1),after,Depth.positive_max(after,positive)) : {index_form((outer_index / _ : U32),lanes,group_index,after,(outer_index % _ : U32)) == target : U32}
  index_decoded((outer_index / after : U32),lanes,group_index,after,(outer_index % after : U32),Nat.div(cell,count),group,Nat.mod(cell,count),quotient_exact,group_exact,rest_exact)

def step_exact(+after: U32,+spreads: Spreads<after>) -> {after == U32.from_nat(Representation.least_one(U32.to_nat(after))) : U32}:
  Spreads{+cap,+cap_ok,+positive,+bounded} = spreads
  %Equal.sym(Nat,Representation.least_one(U32.to_nat(after)),U32.to_nat(after),Representation.least_positive(U32.to_nat(after),positive)) : {after == U32.from_nat(_) : U32}
  Equal.sym(U32,U32.from_nat(U32.to_nat(after)),after,Arithmetic.native_roundtrip(after))

def spread_agree(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+lanes: U32,+padded_outer: U32,
  +padded_position: Grid.Position,+outer: U32,+after: U32,+position: Grid.Position,+total: Nat,
  +cover: Public.Cover<limits,lanes,padded_outer,padded_position,outer,position,total>,+word: U32,+inside: {Nat.is_lt(U32.to_nat(word),total) == True{} : Bool},
  +spreads: Spreads<after>) ->
  PaddedCells.Valid<U32.to_nat(lanes),Representation.cell_group(Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position},U32.to_nat(word)),
    TiledIndexed.lanes(~operation,TiledIndexed.index_at(padded_position,word,padded_outer,after,lanes,Grid.groups(lanes)),after,
      Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position}},word)),
    Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original),Representation.Lifted{logical,Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position}},word)>:
  Public.Cover{+counted,+cells,+padded_cells,+placement} = cover
  +geometry = {Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position} : Cells.Cells}
  +number = U32.to_nat(word)
  +groups = U32.to_nat(Grid.groups(lanes))
  +outer_count = U32.to_nat(outer)
  +lane_count = U32.to_nat(lanes)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +product_positive = {%Equal.sym(Nat,Nat.mul(groups,outer_count),total,cells) : {Nat.is_lt(0n,_) == True{} : Bool} Order.positive_bound(number,total,inside) :
      {Nat.is_lt(0n,Nat.mul(groups,outer_count)) == True{} : Bool}}
  +outer_positive = DivisionOrder.right_factor_positive(groups,outer_count,product_positive)
  +groups_positive = RowMajor.left_factor_positive(groups,outer_count,product_positive)
  +lanes_bound = Order.transitive(lane_count,Nat.mul(lane_count,outer_count),capacity,RowMajor.product_covers(lane_count,outer_count,outer_positive),counted)
  +grouped = TiledValues.grouped(limits,lanes,lanes_bound)
  +total_bound = {%cells : {Nat.is_le(_,capacity) == True{} : Bool}
      Order.transitive(Nat.mul(groups,outer_count),Nat.mul(lane_count,outer_count),capacity,RowMajor.mul_le(groups,lane_count,outer_count,TiledValues.fewer_of(limits,lanes,grouped)),
        counted) :
      {Nat.is_le(total,capacity) == True{} : Bool}}
  +groups_room = Order.transitive(groups,total,capacity,%cells : {Nat.is_le(groups,_) == True{} : Bool} RowMajor.product_covers(groups,outer_count,outer_positive),total_bound)
  +outer_room = Order.transitive(outer_count,total,capacity,%cells : {Nat.is_le(outer_count,_) == True{} : Bool} DivisionOrder.le_multiple(groups,outer_count,groups_positive),
    total_bound)
  +decoded = decode(limits,lanes,padded_outer,padded_position,outer,after,position,total,word,placement,cells,inside,outer_positive,groups_positive,
    Limits.double_bound(limits,outer_count,outer_room),Limits.double_bound(limits,groups,groups_room))
  written_valid(~operation,~Logical,~original,logical,geometry,Representation.cell_outer(geometry,number),Representation.cell_group(geometry,number),
    TiledIndexed.index_at(padded_position,word,padded_outer,after,lanes,Grid.groups(lanes)),after,
    index_exact(padded_position,word,padded_outer,Grid.groups(lanes),outer,after,lanes,position,spreads,decoded),step_exact(after,spreads))

# Every written cell of the cells agrees with the cell of the indexed reader.
def written_agree(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+scope: Scope<Logical>,+word: U32,
  +inside: {Nat.is_lt(U32.to_nat(word),PaddedCells.layout_total(layout_of(~Logical,scope))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(PaddedCells.layout_geometry(layout_of(~Logical,scope))),
    Representation.cell_group(PaddedCells.layout_geometry(layout_of(~Logical,scope)),U32.to_nat(word)),
    written_cell(~operation,~Logical,~original,itself(~Logical,scope),word),
    Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original),lifted_of(~Logical,scope),word)>:
  Scope{+logical,+layout,+facts} = scope
  PaddedCells.Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover} = layout
  spread_agree(~operation,~Logical,~original,logical,limits,lanes,padded_outer,padded_position,outer,after,position,total,cover,word,inside,
    spread_of(total,after,U32.to_nat(word),inside,facts))

# ---- The written content of a piece ----

def fresh_of(+count: U32) -> Tree.Tree<Tile.Tile8>:
  Tree.fresh(Tile.Tile8,Storage.allocation_depth(U32.max(count,1)),Tiled.zero_tile())

def fresh_depth(+count: U32,+sized: Depth.Sized<count>) -> {Nat.is_le(Storage.allocation_depth(U32.max(count,1)),30n) == True{} : Bool}:
  %Depth.depth(count) : {Nat.is_le(_,30n) == True{} : Bool}
  Depth.allocation_depth(count,sized)

def fresh_room(+count: U32,+sized: Depth.Sized<count>) -> {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,fresh_of(count)))) == True{} : Bool}:
  +depth = Storage.allocation_depth(U32.max(count,1))
  %Equal.sym(U32,Tree.capacity(Tile.Tile8,fresh_of(count)),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,fresh_of(count),Shape.fresh(Tile.Tile8,depth,Tiled.zero_tile()))) :
    {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
  %Depth.depth(count) : {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,_))) == True{} : Bool}
  Depth.allocation_room(count,sized)

def fresh_capacity(+count: U32,+sized: Depth.Sized<count>) -> {U32.is_le(Tree.capacity(Tile.Tile8,fresh_of(count)),2147483648) == True{} : Bool}:
  +depth = Storage.allocation_depth(U32.max(count,1))
  %Equal.sym(U32,Tree.capacity(Tile.Tile8,fresh_of(count)),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,fresh_of(count),Shape.fresh(Tile.Tile8,depth,Tiled.zero_tile()))) :
    {U32.is_le(_,2147483648) == True{} : Bool}
  Power.native_power_bound(depth,Order.successor(depth,30n,fresh_depth(count,sized)))

# A ready piece holds at most 2^30 cells.
def count_allocatable(+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,+ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,
  +length_exact: {length == U32.to_nat(count) : Nat}) -> {U32.is_le(count,1073741824) == True{} : Bool}:
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  +power = U32.shln(1,depth)
  +bounded = {%Shape.capacity(Tile.Tile8,depth,tree,balanced) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool} room : {Nat.is_le(length,U32.to_nat(power)) == True{} : Bool}}
  +count_bound = {%length_exact : {Nat.is_le(_,U32.to_nat(power)) == True{} : Bool} bounded : {Nat.is_le(U32.to_nat(count),U32.to_nat(power)) == True{} : Bool}}
  Order.word_le(count,power,1073741824,
    %Equal.sym(Bool,U32.is_le(count,power),Nat.is_le(U32.to_nat(count),U32.to_nat(power)),Comparison.le(count,power)) : {_ == True{} : Bool} count_bound,
    Power.native_allocatable(depth,depth_ok))

def written_ready(~operation: U32 -> F32 -> F32,~Logical: Data,+scope: Scope<Logical>,+first: U32,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +sized: Depth.Sized<count>) -> Buffer.ContentReady<Tile.Tile8,written(~operation,context_of(~Logical,scope),first,count,Buffer.Content{tree,length})>:
  +depth = Storage.allocation_depth(U32.max(count,1))
  +program = writes(~operation,U32.to_nat(count),indexing_of(context_of(~Logical,scope),first),tree)
  Buffer.ContentReady{depth,MoveReadback.pure_balanced(~Tile.Tile8,program,depth,fresh_of(count),Shape.fresh(Tile.Tile8,depth,Tiled.zero_tile())),fresh_depth(count,sized),
    %Equal.sym(U32,Tree.capacity(Tile.Tile8,Writes.pure(Tile.Tile8,program,fresh_of(count))),Tree.capacity(Tile.Tile8,fresh_of(count)),
      MoveReadback.pure_capacity(~Tile.Tile8,program,fresh_of(count))) :
      {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    fresh_room(count,sized)}

def position_word(+first: U32,+index: Nat) -> {(first + U32.from_nat(index) : U32) == U32.from_nat(Nat.add(U32.to_nat(first),index)) : U32}:
  %Arithmetic.native_roundtrip(first) : {(_ + U32.from_nat(index) : U32) == U32.from_nat(Nat.add(U32.to_nat(first),index)) : U32}
  WordArithmetic.native_add(U32.to_nat(first),index)

# What one piece supplies to every one of its written cells.
type Piece<-Logical: Data> is Data:
  Piece{scope: Scope<Logical>,first: U32,count: U32,tree: Tree.Tree<Tile.Tile8>,length: Nat,length_exact: {length == U32.to_nat(count) : Nat},sized: Depth.Sized<count>}

def Before(~Logical: Data,~original: Logical -> U32 -> F32,+piece: Piece<Logical>,+index: Nat) -> Data:
  Piece{+scope,+first,count,+tree,+length,length_exact,sized} = piece
  Content.At(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_of(~Logical,scope),first,tree,length,index)

def After(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+piece: Piece<Logical>,+index: Nat) -> Data:
  Piece{+scope,+first,+count,+tree,+length,length_exact,sized} = piece
  +output = written(~operation,context_of(~Logical,scope),first,count,Buffer.Content{tree,length})
  Content.At(~Tile.Tile8,~Scope<Logical>,~written_cell(~operation,~Logical,~original),scope,first,Buffer.content_tree(~Tile.Tile8,output),
    Buffer.content_length(~Tile.Tile8,output),index)

# A written cell holds the function at the indices of the cell of the input.
def written_at(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+piece: Piece<Logical>,+index: Nat,
  +before: Before(~Logical,~original,piece,index)) -> After(~operation,~Logical,~original,piece,index):
  Piece{+scope,+first,+count,+tree,+length,+length_exact,+sized} = piece
  Scope{+logical,+layout,+facts} = scope
  PaddedCells.Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover} = layout
  KernelGemm.Both{+below,+equal} = before
  +depth = Storage.allocation_depth(U32.max(count,1))
  +fresh = fresh_of(count)
  +indexing = {TiledIndexed.Indexing{padded_outer,after,lanes,Grid.groups(lanes),padded_position,first} : TiledIndexed.Indexing}
  +word = U32.from_nat(Nat.add(U32.to_nat(first),index))
  +geometry = {Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position} : Cells.Cells}
  +selected = {%length_exact : {Nat.is_lt(index,_) == True{} : Bool} below : {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}}
  +read = TraversalIndexedWrites.read(~Tile.Tile8,~TraversalIndexedWrites.Input<Tree.Tree<Tile.Tile8>,TiledIndexed.Indexing>,
    ~TraversalIndexedWrites.value(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~sample(~operation)),U32.to_nat(count),TraversalIndexedWrites.Input{tree,indexing},
    fresh,depth,index,Shape.fresh(Tile.Tile8,depth,Tiled.zero_tile()),fresh_depth(count,sized),fresh_capacity(count,sized),fresh_room(count,sized),selected)
  KernelGemm.Both{selected,
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Writes.pure(Tile.Tile8,writes(~operation,U32.to_nat(count),indexing,tree),fresh),U32.from_nat(index)),
      sample(~operation,tree,indexing,U32.from_nat(index)),
      TiledIndexed.lanes(~operation,TiledIndexed.index_at(padded_position,word,padded_outer,after,lanes,Grid.groups(lanes)),after,
        Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word)),
      read,
      %Equal.sym(U32,(first + U32.from_nat(index) : U32),word,position_word(first,index)) :
        {TiledIndexed.lanes(~operation,TiledIndexed.index_at(padded_position,_,padded_outer,after,lanes,Grid.groups(lanes)),after,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) ==
          TiledIndexed.lanes(~operation,TiledIndexed.index_at(padded_position,word,padded_outer,after,lanes,Grid.groups(lanes)),after,
            Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word)) : Tile.Tile8}
      Equal.cong(Tile.Tile8,Tile.Tile8,tile => TiledIndexed.lanes(~operation,TiledIndexed.index_at(padded_position,word,padded_outer,after,lanes,Grid.groups(lanes)),after,tile),
        Tree.read(Tile.Tile8,tree,U32.from_nat(index)),Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word),equal))}

# A step whose written content agrees in length with its input.
def relength(~After: Data,~after_value: After -> U32 -> Tile.Tile8,+after: After,+first: U32,+count: U32,+input: Buffer.Content<Tile.Tile8>,
  +written: Buffer.Content<Tile.Tile8>,+output: Buffer.Content<Tile.Tile8>,
  +same_length: {Buffer.content_length(~Tile.Tile8,written) == Buffer.content_length(~Tile.Tile8,input) : Nat},
  stepped: PiecesMap.Stepped(~Tile.Tile8,~After,~after_value,after,first,count,written,output)) ->
  PiecesMap.Stepped(~Tile.Tile8,~After,~after_value,after,first,count,input,output):
  KernelGemm.Both{+ready,KernelGemm.Both{+same,+values}} = stepped
  KernelGemm.Both{ready,KernelGemm.Both{Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,output),Buffer.content_length(~Tile.Tile8,written),
    Buffer.content_length(~Tile.Tile8,input),same,same_length),values}}

# What the step of an indexed map keeps of a piece of the lifted values: the
# function at the indices of its cells, then the padding.
def piece_rule(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+scope: Scope<Logical>,+first: U32,+count: U32,
  +input: Buffer.Content<Tile.Tile8>,+ready: Buffer.ContentReady<Tile.Tile8,input>,+length_exact: {Buffer.content_length(~Tile.Tile8,input) == U32.to_nat(count) : Nat},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),total_of(~Logical,scope)) == True{} : Bool},
  +values: Content.piece_values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_of(~Logical,scope),first,count,input)) ->
  PiecesMap.Stepped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original)),lifted_of(~Logical,scope),
    first,count,input,step(~operation,context_of(~Logical,scope),first,count,input)):
  Scope{+logical,+layout,+facts} = scope
  Buffer.Content{+tree,+length} = input
  +rebuilt = {Scope{logical,layout,facts} : Scope<Logical>}
  +sized = Depth.bounded(count,count_allocatable(count,tree,length,ready,length_exact))
  +content = {Buffer.Content{tree,length} : Buffer.Content<Tile.Tile8>}
  +written_content = written(~operation,context_of(~Logical,rebuilt),first,count,content)
  relength(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original)),lifted_of(~Logical,rebuilt),first,count,
    content,written_content,Padding.step(PaddedCells.layout_padding(layout),first,count,written_content),Equal.sym(Nat,length,U32.to_nat(count),length_exact),
    PaddedCells.padded_piece(~Scope<Logical>,~Scope<Logical>,~written_cell(~operation,~Logical,~original),~Representation.Lifted<Logical>,
      ~Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original)),~layout_of(~Logical),~itself(~Logical),~lifted_of(~Logical),
      ~written_agree(~operation,~Logical,~original),rebuilt,first,count,written_content,written_ready(~operation,~Logical,rebuilt,first,count,tree,length,sized),{==},inside,
      Members.transform_all(~Nat,~Piece<Logical>,~Before(~Logical,~original),~After(~operation,~Logical,~original),~written_at(~operation,~Logical,~original),
        Ranges.range(U32.to_nat(count),0n),Piece{rebuilt,first,count,tree,length,length_exact,sized},values)))

# ---- The indexed map of a tiled tensor ----

# The indexed map of the pieces of a scope keeps the contract with the
# indexed reader.
def scoped(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+scope: Scope<Logical>,+parallel: Bool,+tasks: Pieces.Ranges,
  -pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +total_exact: {total_of(~Logical,scope) == ShapeLaws.elements(shape) : Nat},observation: Model.Observation<Tile.Tile8,pieces>,
  contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_of(~Logical,scope),
    Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original)),lifted_of(~Logical,scope),
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Indexed,~TiledMap.indexed_leaf(~operation),context_of(~Logical,scope),Tasks.Decision{parallel,tasks},pieces,shape,fill,workers)):
  Model.Observation{+model,+equation} = observation
  %equation : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original)),lifted_of(~Logical,scope),
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Indexed,~TiledMap.indexed_leaf(~operation),context_of(~Logical,scope),Tasks.Decision{parallel,tasks},_,shape,fill,workers))
  PiecesMap.decided(~Tile.Tile8,~TiledMap.Indexed,~TiledMap.indexed_leaf(~operation),~step(~operation),~rule(~operation),~Representation.Lifted<Logical>,
    ~Representation.lifted(~Logical,~original),~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original)),
    ~Scope<Logical>,~context_of(~Logical),~lifted_of(~Logical),~lifted_of(~Logical),~total_of(~Logical),~piece_rule(~operation,~Logical,~original),
    scope,Tasks.Decision{parallel,tasks},model,shape,fill,workers,total_exact,
    %Equal.sym(Pieces.Pieces(~Tile.Tile8),Model.pack(~Tile.Tile8,model),pieces,equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_of(~Logical,scope),
        Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers}) contract)

# The indexed map of the cells of values in their layout.
def placed_indexed(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+within: Bool,+logical: Logical,+values: List<&2,U32>,+axis: Nat,
  +lanes: U32,+position: Grid.Position,+parallel: Bool,+tasks: Pieces.Ranges,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,Public.geometry_of(values,axis,position)},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}),
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+placed: PaddedCells.Placed<shape,values,axis,lanes,position,within>,
  +facts: AfterFacts<ShapeLaws.elements(shape),Tiled.product(Public.lane_after(values,axis))>) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original)),
    Representation.Lifted{logical,Public.geometry_of(values,axis,position)},
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Indexed,~TiledMap.indexed_leaf(~operation),
      TiledMap.Indexed{Tiled.product(Public.lane_after(values,axis)),TiledMap.padding_when(within,shape,lanes,position)},Tasks.Decision{parallel,tasks},pieces,shape,fill,workers)):
  PaddedCells.Placed{+layout,+padding_exact,+geometry_exact,+total_exact} = placed
  PaddedCells.Layout{+limits,+cell_lanes,+padded_outer,+padded_position,+outer,+after,+cell_position,+total,+cover} = layout
  +rebuilt = {PaddedCells.Layout{limits,cell_lanes,padded_outer,padded_position,outer,after,cell_position,total,cover} : PaddedCells.Layout}
  +geometry = {Cells.Cells{outer,after,cell_lanes,Tiled.groups(cell_lanes),cell_position} : Cells.Cells}
  +padding = {TiledMap.Padding{cell_lanes,Grid.groups(cell_lanes),padded_outer,padded_position} : TiledMap.Padding}
  +others = Public.lane_after(values,axis)
  +value_lanes = Grid.lanes_at(values,axis)
  +after_exact = {Equal.cong(Cells.Cells,U32,cells => cells_after(cells),geometry,
      Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(others) : U32),Tiled.product(others),value_lanes,Tiled.groups(value_lanes),position},
      Equal.trans(Cells.Cells,geometry,Public.geometry_of(values,axis,position),
        Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(others) : U32),Tiled.product(others),value_lanes,Tiled.groups(value_lanes),position},
        geometry_exact,Public.geometry_exact(values,axis,position,inside))) : {after == Tiled.product(others) : U32}}
  +scope_facts = {%Equal.sym(U32,after,Tiled.product(others),after_exact) : AfterFacts<total,_>
      %Equal.sym(Nat,total,ShapeLaws.elements(shape),total_exact) : AfterFacts<_,Tiled.product(others)> facts : AfterFacts<total,after>}
  %padding_exact : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original)),
    Representation.Lifted{logical,Public.geometry_of(values,axis,position)},
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Indexed,~TiledMap.indexed_leaf(~operation),TiledMap.Indexed{Tiled.product(others),_},Tasks.Decision{parallel,tasks},
      pieces,shape,fill,workers))
  %geometry_exact : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original)),
    Representation.Lifted{logical,_},
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Indexed,~TiledMap.indexed_leaf(~operation),TiledMap.Indexed{Tiled.product(others),padding},Tasks.Decision{parallel,tasks},
      pieces,shape,fill,workers))
  %after_exact : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original)),
    Representation.Lifted{logical,geometry},
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Indexed,~TiledMap.indexed_leaf(~operation),TiledMap.Indexed{_,padding},Tasks.Decision{parallel,tasks},
      pieces,shape,fill,workers))
  scoped(~operation,~Logical,~original,Scope{logical,rebuilt,scope_facts},parallel,tasks,pieces,shape,fill,workers,total_exact,Model.observe(~Tile.Tile8,pieces),
    %Equal.sym(Cells.Cells,geometry,Public.geometry_of(values,axis,position),geometry_exact) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,_},
        Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}) global)

# ---- The after count of laid values ----

# The axes after the lane axis of values with values hold values, no more
# than the values.
def after_positive(+values: List<&2,U32>,+axis: Nat,+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(Public.lane_after(values,axis))) == True{} : Bool}:
  match values axis:
    case Nil{} axis: {==}
    case Con{+extent,+rest} 0n: DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive)
    case Con{+extent,+rest} 1n+ +later: after_positive(rest,later,DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive))

def after_bound(+values: List<&2,U32>,+axis: Nat,+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(Public.lane_after(values,axis)),ShapeLaws.elements(values)) == True{} : Bool}:
  match values axis:
    case Nil{} axis: {==}
    case Con{+extent,+rest} 0n:
      DivisionOrder.le_multiple(U32.to_nat(extent),ShapeLaws.elements(rest),RowMajor.left_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive))
    case Con{+extent,+rest} 1n+ +later:
      Order.transitive(ShapeLaws.elements(Public.lane_after(rest,later)),ShapeLaws.elements(rest),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)),
        after_bound(rest,later,DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive)),
        DivisionOrder.le_multiple(U32.to_nat(extent),ShapeLaws.elements(rest),RowMajor.left_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive)))

# Lanes with groups are lanes.
def lanes_cases(zero: Bool,+lanes: U32,+classified: {U32.is_zero(lanes) == zero : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(Grid.groups(lanes))) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool}:
  match zero:
    case True{}:
      Empty.absurd({Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool},
        WordEncoding.false_true(%Arithmetic.word_zero(lanes,classified) : {Nat.is_lt(0n,U32.to_nat(Grid.groups(_))) == True{} : Bool} groups_positive))
    case False{}: Arithmetic.word_nonzero_positive(lanes,classified)

# Values whose cells have cells have values.
def values_positive(within: Bool,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},
  +cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(within,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}:
  match within:
    case True{}:
      +others = Grid.removed(values,axis)
      +lanes = Grid.lanes_at(values,axis)
      +groups = Grid.groups(lanes)
      +product = {%Public.placed_cells(position,others,groups) : {Nat.is_lt(0n,_) == True{} : Bool} cells_positive :
        {Nat.is_lt(0n,Nat.mul(U32.to_nat(groups),ShapeLaws.elements(others))) == True{} : Bool}}
      %Equal.sym(Nat,ShapeLaws.elements(values),Nat.mul(U32.to_nat(lanes),ShapeLaws.elements(others)),Public.elements_split(values,axis,classified)) :
        {Nat.is_lt(0n,_) == True{} : Bool}
      Order.product_positive(U32.to_nat(lanes),ShapeLaws.elements(others),
        lanes_cases(U32.is_zero(lanes),lanes,{==},RowMajor.left_factor_positive(U32.to_nat(groups),ShapeLaws.elements(others),product)),
        DivisionOrder.right_factor_positive(U32.to_nat(groups),ShapeLaws.elements(others),product))
    case False{}:
      %Public.removed_past(values,axis,classified) : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool}
      cells_positive

# Values without values have no cells.
def cells_cases(positive: Bool,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+shape: List<&2,U32>,
  +shaped: {Tiled.cell_shape(values,axis,position) == shape : List<&2,U32>},+empty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == False{} : Bool},
  +classified: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == positive : Bool}) -> {Nat.is_lt(0n,ShapeLaws.elements(shape)) == False{} : Bool}:
  match positive:
    case True{}:
      +cells = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shaped) : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} classified :
        {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}}
      Empty.absurd({Nat.is_lt(0n,ShapeLaws.elements(shape)) == False{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(0n,ShapeLaws.elements(values)),True{},Equal.sym(Bool,Nat.is_lt(0n,ShapeLaws.elements(values)),False{},empty),
          values_positive(Nat.is_lt(axis,List.length(&2,U32,values)),values,axis,position,{==},cells))))
    case False{}: classified

def facts_cases(positive: Bool,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+shape: List<&2,U32>,+count: U32,
  +shaped: {Tiled.cell_shape(values,axis,position) == shape : List<&2,U32>},+accepted: {Tensors.count_of(values) == Some{count} : Maybe<&2,U32>},
  +classified: {Nat.is_lt(0n,ShapeLaws.elements(values)) == positive : Bool}) ->
  AfterFacts<ShapeLaws.elements(shape),Tiled.product(Public.lane_after(values,axis))>:
  match positive:
    case True{}:
      +limits = Default.limits()
      +capacity = Machine.capacity(limits)
      +exact = Count.count_exact(ShapeLaws.elements(values),count,Count.count_of(values,count,accepted))
      +bounded = PaddedCells.count_bounded(ShapeLaws.elements(values),count,Count.count_of(values,count,accepted))
      +values_room = {%Equal.sym(Nat,ShapeLaws.elements(values),U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
          Bounds.as_nat_le(count,capacity,bounded) : {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(capacity)) == True{} : Bool}}
      +others = Public.lane_after(values,axis)
      +room = Order.transitive(ShapeLaws.elements(others),ShapeLaws.elements(values),U32.to_nat(capacity),after_bound(values,axis,classified),values_room)
      +product = Public.product_exact(capacity,Limits.capacity_fits(limits),others,room)
      +word = U32.to_nat(Tiled.product(others))
      Spread{Spreads{Machine.half_word(limits),Limits.half_word_fits(limits),
        %Equal.sym(Nat,word,ShapeLaws.elements(others),product) : {Nat.is_lt(0n,_) == True{} : Bool} after_positive(values,axis,classified),
        Limits.double_bound(limits,word,%Equal.sym(Nat,word,ShapeLaws.elements(others),product) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} room)}}
    case False{}: NoCells{cells_cases(Nat.is_lt(0n,ShapeLaws.elements(shape)),values,axis,position,shape,shaped,classified,{==})}

def after_facts(+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,+shape: List<&2,U32>,+laid: Public.Laid<shape,axis,lanes,position,values>) ->
  AfterFacts<ShapeLaws.elements(shape),Tiled.product(Public.lane_after(values,axis))>:
  Public.Laid{+inside,+shaped,+restored,+counted,+pinned} = laid
  Global.Counted{+count,+accepted} = counted
  facts_cases(Nat.is_lt(0n,ShapeLaws.elements(values)),values,axis,position,shape,count,shaped,accepted,{==})

def laid_inside(+shape: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,+values: List<&2,U32>,+laid: Public.Laid<shape,axis,lanes,position,values>) ->
  {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool}:
  Public.Laid{+inside,+shaped,+restored,+counted,+pinned} = laid
  inside

def decided_indexed(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+axis: Nat,+lanes: U32,
  +position: Grid.Position,decision: Tasks.Decision,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,Public.geometry_of(values,axis,position)},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}),
  +laid: Public.Laid<shape,axis,lanes,position,values>) ->
  Public.Within(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~indexed_value(~operation,~Logical,~original)),~Logical,~Public.lifted_at(~Logical),
    logical,values,axis,lanes,position,
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Indexed,~TiledMap.indexed_leaf(~operation),
      TiledMap.Indexed{Tiled.product(Public.lane_after(values,axis)),TiledMap.padding_of(shape,axis,lanes,position)},decision,pieces,shape,fill,workers)):
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      +within = Nat.is_lt(axis,List.length(&2,U32,shape))
      KernelGemm.Both{placed_indexed(~operation,~Logical,~original,within,logical,values,axis,lanes,position,parallel,tasks,pieces,shape,fill,workers,global,
        laid_inside(shape,axis,lanes,position,values,laid),PaddedCells.placed(within,values,axis,lanes,position,shape,{==},laid),after_facts(values,axis,lanes,position,shape,laid)),laid}

# The function of the row-major index and the value at every lane keeps the
# contract of a tiled tensor with the indexed reader.
def indexed(~operation: U32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+operations: F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,
  contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~Logical,~indexed_value(~operation,~Logical,~original),logical,values,TiledMap.indexed(~operation,operations,tiled)):
  match tiled:
    case Tiled.Tiled{cells,+axis,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          KernelGemm.Both{global,+laid} = contract
          Public.Laid{+inside,+shaped,+restored,+counted,+pinned} = laid
          %Equal.sym(List<&2,U32>,Tiled.logical_shape(shape,axis,lanes,position),values,restored) :
            Public.Contract(~Logical,~indexed_value(~operation,~Logical,~original),logical,values,
              TiledMap.indexed_around(~operation,operations,Tiled.around(_,axis,1),Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position,
                TiledMap.padding_of(shape,axis,lanes,position)))
          %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(values,axis,1),
            Some{Tiled.Around{Public.lane_prefix(values,axis,1),Grid.lanes_at(values,axis),Public.lane_after(values,axis)}},Public.located(values,axis,1,inside)) :
            Public.Contract(~Logical,~indexed_value(~operation,~Logical,~original),logical,values,
              TiledMap.indexed_around(~operation,operations,_,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position,
                TiledMap.padding_of(shape,axis,lanes,position)))
          decided_indexed(~operation,~Logical,~original,logical,values,axis,lanes,position,
            Tasks.decide(Tensors.grid_of(shape),workers,Tensors.pass_rows(1.0,operations),Tensors.pass_steps(),Tasks.no_extra()),pieces,shape,fill,workers,global,Public.Laid{inside,shaped,restored,counted,pinned})
        case Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}:
          Empty.absurd(Public.Contract(~Logical,~indexed_value(~operation,~Logical,~original),logical,values,
            TiledMap.indexed(~operation,operations,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},axis,lanes,position})),contract)
