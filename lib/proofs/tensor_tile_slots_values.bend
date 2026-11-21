# The slots of a row as values. A fold of cells into the slots in place hands
# its buffer back and leaves slots that a view of their parts, phase and extra
# describes: each cell steps the slot of its phase.
import Base
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../math_sum.bend as Sum
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../tensor_line_chain.bend as LineChain
import ../traversal_partition.bend as Partition
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_refinement.bend as StorageLaws
import ./storage_copy_refinement.bend as CopyLaws
import ./traversal_certificate.bend as TraversalCertificate
import ./traversal_witness.bend as TraversalWitness
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_update_readback.bend as Updates
import ./tensor_pieces_regrouped.bend as Regrouped
import ./tensor_line_chain_values.bend as LineChainValues
import ./tensor_reduce_values.bend as ReduceValues
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction

# ---- The slots as values ----

type SlotsValue<-Extra: Data> is Data:
  SlotsValue{parts: Buffer.Model<Sum.TileParts>,phase: U32,extra: Extra}

def shown(~Extra: Data,value: SlotsValue<Extra>) -> Operations.TileSlots<Extra>:
  SlotsValue{parts,phase,extra} = value
  Operations.TileSlots{Buffer.pack(~Sum.TileParts,parts),phase,extra}

def valued(~Extra: Data,slots: Operations.TileSlots<Extra>) -> SlotsValue<Extra>:
  Operations.TileSlots{parts,phase,extra} = slots
  SlotsValue{Buffer.remodel(~Sum.TileParts,parts),phase,extra}

def valued_shown(~Extra: Data,slots: Operations.TileSlots<Extra>) -> {shown(~Extra,valued(~Extra,slots)) == slots : Operations.TileSlots<Extra>}:
  Operations.TileSlots{parts,+phase,+extra} = slots
  Equal.cong(Storage.Buffer<Sum.TileParts>,Operations.TileSlots<Extra>,parts => Operations.TileSlots{parts,phase,extra},
    Buffer.pack(~Sum.TileParts,Buffer.remodel(~Sum.TileParts,parts)),parts,Buffer.remodeled(~Sum.TileParts,parts))

# The slots of a fold of cells.
def fold_value(~Extra: Data,~read: Array<Tile.Tile8> -> Traversal.Spaced<Extra> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,
  ~read_preserves: @-reader: Array<Tile.Tile8> -> @context: Traversal.Spaced<Extra> -> @index: U32 -> TraversalWitness.array_evidence(~Tile.Tile8,reader) ->
    TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,reader,read(reader,context,index)>,
  count: Nat,position: U32,step: U32,value: SlotsValue<Extra>,model: Buffer.Model<Tile.Tile8>) -> SlotsValue<Extra>:
  valued(~Extra,Pair.snd(Storage.Buffer<Tile.Tile8>,Operations.TileSlots<Extra>,
    Operations.cells_slots(~Extra,~read,~read_preserves,count,position,step,shown(~Extra,value),Buffer.pack(~Tile.Tile8,model))))

def begin_value(~Extra: Data,extra: Extra) -> SlotsValue<Extra>:
  valued(~Extra,Operations.slots_begun(~Extra,extra))

def turn_value(~Extra: Data,value: SlotsValue<Extra>) -> Sum.TileParts:
  Operations.slots_closed(~Extra,shown(~Extra,value))

# ---- The buffer handed back ----

# The buffer of the slots is the input with its own certificate, carried back
# by the equation the run keeps its reader by.
def buffers_kept(~Extra: Data,+length: Nat,+slots_length: Nat,+phase: U32,+extra: Extra,-input: Array<Tile.Tile8>,result: Array<Tile.Tile8> & Array<Sum.TileParts>,
  +certificate: Certificate.Certificate<Tile.Tile8,input>,
  output_certificate: Certificate.Certificate<Sum.TileParts,TraversalWitness.second(~Array<Tile.Tile8>,~Array<Sum.TileParts>,result)>,
  +same: {TraversalWitness.first(~Array<Tile.Tile8>,~Array<Sum.TileParts>,result) == input : Array<Tile.Tile8>}) ->
  {Operations.slots_buffers(~Extra,length,slots_length,phase,extra,result,
      TraversalCertificate.Certificates{CopyLaws.returned_certificate(~Tile.Tile8,input,TraversalWitness.first(~Array<Tile.Tile8>,~Array<Sum.TileParts>,result),certificate,same),
        output_certificate}) ==
    (Storage.Buffer{input,length,certificate},Pair.snd(Storage.Buffer<Tile.Tile8>,Operations.TileSlots<Extra>,Operations.slots_buffers(~Extra,length,slots_length,phase,extra,result,
      TraversalCertificate.Certificates{CopyLaws.returned_certificate(~Tile.Tile8,input,TraversalWitness.first(~Array<Tile.Tile8>,~Array<Sum.TileParts>,result),certificate,same),
        output_certificate}))) : Storage.Buffer<Tile.Tile8> & Operations.TileSlots<Extra>}:
  (returned,output) = result
  Equal.cong(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8> & Operations.TileSlots<Extra>,
    buffer => (buffer,Operations.TileSlots{Storage.Buffer{output,slots_length,output_certificate},phase,extra}),
    Storage.Buffer{returned,length,CopyLaws.returned_certificate(~Tile.Tile8,input,returned,certificate,same)},Storage.Buffer{input,length,certificate},
    CopyLaws.source_certificate(~Tile.Tile8,length,input,returned,same,certificate))

# A fold of cells into the slots hands the buffer back.
def slots_kept(~Extra: Data,~read: Array<Tile.Tile8> -> Traversal.Spaced<Extra> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,
  ~read_preserves: @-reader: Array<Tile.Tile8> -> @context: Traversal.Spaced<Extra> -> @index: U32 -> TraversalWitness.array_evidence(~Tile.Tile8,reader) ->
    TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,reader,read(reader,context,index)>,
  +count: Nat,+position: U32,+step: U32,+value: SlotsValue<Extra>,+model: Buffer.Model<Tile.Tile8>) ->
  {Operations.cells_slots(~Extra,~read,~read_preserves,count,position,step,shown(~Extra,value),Buffer.pack(~Tile.Tile8,model)) ==
    (Buffer.pack(~Tile.Tile8,model),Pair.snd(Storage.Buffer<Tile.Tile8>,Operations.TileSlots<Extra>,
      Operations.cells_slots(~Extra,~read,~read_preserves,count,position,step,shown(~Extra,value),Buffer.pack(~Tile.Tile8,model)))) :
    Storage.Buffer<Tile.Tile8> & Operations.TileSlots<Extra>}:
  SlotsValue{parts,+phase,+extra} = value
  Buffer.Model{+slots,+slots_length,Certificate.Certificate{+slots_depth,+slots_equation}} = parts
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  +run = {Traversal.Steps{count,phase,Traversal.Spaced{(position - phase * step : U32),step,extra}} : Traversal.Steps<Traversal.Spaced<Extra>>}
  +certificate = {Certificate.Certificate{depth,equation} : Certificate.Certificate<Tile.Tile8,Tree.pack(Tile.Tile8,tree)>}
  result = Traversal.updated(~Array<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~read,~Sum.tile_slot_step,~Operations.slot_of(~Extra),run,
    Tree.pack(Tile.Tile8,tree),Tree.pack(Sum.TileParts,slots))
  buffers_kept(~Extra,length,slots_length,(phase + U32.from_nat(count) : U32),extra,Tree.pack(Tile.Tile8,tree),result,certificate,
    Certificate.Certificate{slots_depth,TraversalCertificate.operation_equation(~Array<Tile.Tile8>,~Sum.TileParts,~Traversal.Steps<Traversal.Spaced<Extra>>,~Certificate.family(~Tile.Tile8),
      ~Traversal.updated(~Array<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~read,~Sum.tile_slot_step,~Operations.slot_of(~Extra)),
      ~TraversalCertificate.update_preserves(~Array<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~TraversalWitness.array_evidence(~Tile.Tile8),~Certificate.family(~Tile.Tile8),
        ~read,~Sum.tile_slot_step,~Operations.slot_of(~Extra),~read_preserves,~TraversalCertificate.array_observed(~Tile.Tile8)),
      run,Tree.pack(Tile.Tile8,tree),Tree.pack(Sum.TileParts,slots),certificate,slots_depth,slots_equation)},
    TraversalWitness.source_unchanged(~Array<Tile.Tile8>,~Sum.TileParts,Tree.pack(Tile.Tile8,tree),slots_depth,result,
      TraversalCertificate.update_preserves(~Array<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~TraversalWitness.array_evidence(~Tile.Tile8),~Certificate.family(~Tile.Tile8),
        ~read,~Sum.tile_slot_step,~Operations.slot_of(~Extra),~read_preserves,~TraversalCertificate.array_observed(~Tile.Tile8),
        slots_depth,run,Tree.pack(Tile.Tile8,tree),Tree.pack(Sum.TileParts,slots),certificate,slots_equation)))

# ---- The line chain laws of the slots ----

def slots_same(~Extra: Data,~read: Array<Tile.Tile8> -> Traversal.Spaced<Extra> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,
  ~read_preserves: @-reader: Array<Tile.Tile8> -> @context: Traversal.Spaced<Extra> -> @index: U32 -> TraversalWitness.array_evidence(~Tile.Tile8,reader) ->
    TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,reader,read(reader,context,index)>,
  +count: Nat,+position: U32,+step: U32,+initial: SlotsValue<Extra>,+model: Buffer.Model<Tile.Tile8>) ->
  {Operations.cells_slots(~Extra,~read,~read_preserves,count,position,step,shown(~Extra,initial),Buffer.pack(~Tile.Tile8,model)) ==
    (Buffer.pack(~Tile.Tile8,model),shown(~Extra,fold_value(~Extra,~read,~read_preserves,count,position,step,initial,model))) :
    Storage.Buffer<Tile.Tile8> & Operations.TileSlots<Extra>}:
  Equal.trans(Storage.Buffer<Tile.Tile8> & Operations.TileSlots<Extra>,
    Operations.cells_slots(~Extra,~read,~read_preserves,count,position,step,shown(~Extra,initial),Buffer.pack(~Tile.Tile8,model)),
    (Buffer.pack(~Tile.Tile8,model),Pair.snd(Storage.Buffer<Tile.Tile8>,Operations.TileSlots<Extra>,
      Operations.cells_slots(~Extra,~read,~read_preserves,count,position,step,shown(~Extra,initial),Buffer.pack(~Tile.Tile8,model)))),
    (Buffer.pack(~Tile.Tile8,model),shown(~Extra,fold_value(~Extra,~read,~read_preserves,count,position,step,initial,model))),
    slots_kept(~Extra,~read,~read_preserves,count,position,step,initial,model),
    Equal.cong(Operations.TileSlots<Extra>,Storage.Buffer<Tile.Tile8> & Operations.TileSlots<Extra>,slots => (Buffer.pack(~Tile.Tile8,model),slots),
      Pair.snd(Storage.Buffer<Tile.Tile8>,Operations.TileSlots<Extra>,
        Operations.cells_slots(~Extra,~read,~read_preserves,count,position,step,shown(~Extra,initial),Buffer.pack(~Tile.Tile8,model))),
      shown(~Extra,fold_value(~Extra,~read,~read_preserves,count,position,step,initial,model)),
      Equal.sym(Operations.TileSlots<Extra>,shown(~Extra,fold_value(~Extra,~read,~read_preserves,count,position,step,initial,model)),
        Pair.snd(Storage.Buffer<Tile.Tile8>,Operations.TileSlots<Extra>,
          Operations.cells_slots(~Extra,~read,~read_preserves,count,position,step,shown(~Extra,initial),Buffer.pack(~Tile.Tile8,model))),
        valued_shown(~Extra,Pair.snd(Storage.Buffer<Tile.Tile8>,Operations.TileSlots<Extra>,
          Operations.cells_slots(~Extra,~read,~read_preserves,count,position,step,shown(~Extra,initial),Buffer.pack(~Tile.Tile8,model)))))))

def begun_same(~Extra: Data,+extra: Extra) -> {Operations.slots_begun(~Extra,extra) == shown(~Extra,begin_value(~Extra,extra)) : Operations.TileSlots<Extra>}:
  Equal.sym(Operations.TileSlots<Extra>,shown(~Extra,begin_value(~Extra,extra)),Operations.slots_begun(~Extra,extra),valued_shown(~Extra,Operations.slots_begun(~Extra,extra)))

def closed_same(~Extra: Data,+value: SlotsValue<Extra>) -> {Operations.slots_closed(~Extra,shown(~Extra,value)) == turn_value(~Extra,value) : Sum.TileParts}:
  {==}

# ---- A view of the slots ----

type SlotsView<-Extra: Data> is Data:
  SlotsView{slots: Tree.Tree<Sum.TileParts>,phase: U32,extra: Extra}

def slots_view(~Extra: Data,value: SlotsValue<Extra>) -> SlotsView<Extra>:
  SlotsValue{parts,phase,extra} = value
  SlotsView{Buffer.content_tree(~Sum.TileParts,Buffer.content(~Sum.TileParts,parts)),phase,extra}

# A cell steps the slot of the phase.
def cell_view(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,view: SlotsView<Extra>,cell: Tile.Tile8) -> SlotsView<Extra>:
  SlotsView{+slots,+phase,+extra} = view
  SlotsView{Tree.write(Sum.TileParts,slots,U32.and(phase,7),Sum.tile_slot_step(Tree.read(Sum.TileParts,slots,U32.and(phase,7)),point(extra,cell))),(phase + 1 : U32),extra}

def begin_view(~Extra: Data,extra: Extra) -> SlotsView<Extra>:
  slots_view(~Extra,begin_value(~Extra,extra))

def turn_view(~Extra: Data,view: SlotsView<Extra>) -> Sum.TileParts:
  SlotsView{slots,phase,extra} = view
  Operations.eight_rest(Operations.quad_at(0,Tree.pack(Sum.TileParts,slots)))

# The cell a spaced read samples, as its point takes it.
def spaced_sample(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,tree: Tree.Tree<Tile.Tile8>,context: Traversal.Spaced<Extra>,position: U32) -> Tile.Tile8:
  Traversal.Spaced{base,stride,extra} = context
  point(extra,Tree.read(Tile.Tile8,tree,(base + position * stride : U32)))

def held(~Extra: Data,extra: Extra,cell: Tile.Tile8) -> Tile.Tile8:
  cell

def spaced_matches(~Extra: Data,+tree: Tree.Tree<Tile.Tile8>,+context: Traversal.Spaced<Extra>,+position: U32) ->
  {Traversal.spaced(~Tile.Tile8,~Extra,ArrayLaws.pack_tree(~Tile.Tile8,tree),context,position) ==
    (ArrayLaws.pack_tree(~Tile.Tile8,tree),spaced_sample(~Extra,~held(~Extra),tree,context,position)) : Array<Tile.Tile8> & Tile.Tile8}:
  Traversal.Spaced{+base,+stride,extra} = context
  StorageLaws.read(Tile.Tile8,tree,(base + position * stride : U32))

# The slots of the parts handed back are the tree of their array.
def buffers_view(~Extra: Data,+length: Nat,+slots_length: Nat,+phase: U32,+extra: Extra,result: Array<Tile.Tile8> & Array<Sum.TileParts>,
  certificates: TraversalCertificate.Certificates<Array<Tile.Tile8>,Sum.TileParts,Certificate.family(~Tile.Tile8),result>) ->
  {slots_view(~Extra,valued(~Extra,Pair.snd(Storage.Buffer<Tile.Tile8>,Operations.TileSlots<Extra>,Operations.slots_buffers(~Extra,length,slots_length,phase,extra,result,certificates)))) ==
    SlotsView{Tree.reflect(Sum.TileParts,TraversalWitness.second(~Array<Tile.Tile8>,~Array<Sum.TileParts>,result)),phase,extra} : SlotsView<Extra>}:
  (input,slots) = result
  TraversalCertificate.Certificates{certificate,slots_certificate} = certificates
  Equal.cong(Buffer.Content<Sum.TileParts>,SlotsView<Extra>,content => SlotsView{Buffer.content_tree(~Sum.TileParts,content),phase,extra},
    Buffer.content(~Sum.TileParts,Buffer.remodel(~Sum.TileParts,Storage.Buffer{slots,slots_length,slots_certificate})),Buffer.Content{Tree.reflect(Sum.TileParts,slots),slots_length},
    Buffer.remodel_content(~Sum.TileParts,Storage.Buffer{slots,slots_length,slots_certificate}))

# The parts of a finished model run are the tree of its output.
def finished_tree(~Extra: Data,+state: Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Extra>>) ->
  {Tree.reflect(Sum.TileParts,TraversalWitness.second(~Array<Tile.Tile8>,~Array<Sum.TileParts>,Traversal.finish(~Array<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,
      ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,~ArrayLaws.pack_tree(~Tile.Tile8),state)))) ==
    Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,state) : Tree.Tree<Sum.TileParts>}:
  Traversal.State{reader,+output,position,context} = state
  StorageLaws.reflect_packed(Sum.TileParts,output)

# count cells from the phase on, at base + phase * step, step the slots as the
# view steps the cells of the model from the position they are at.
def updates_strided(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,+count: Nat,+phase: U32,+base: U32,+step: U32,+extra: Extra,+slots: Tree.Tree<Sum.TileParts>,
  +position: U32,+model: Buffer.Model<Tile.Tile8>,+at: {(base + phase * step : U32) == position : U32}) ->
  {SlotsView{Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,
        Loop.run(~Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Extra>>,
          ~ArrayLaws.model_update_step(~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~spaced_sample(~Extra,~point),~Sum.tile_slot_step,~Operations.slot_of(~Extra)),
          count,Traversal.State{Buffer.tree(~Tile.Tile8,model),slots,phase,Traversal.Spaced{base,step,extra}})),(phase + U32.from_nat(count) : U32),extra} ==
    LineChainValues.strided(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~SlotsView<Extra>,~ReduceValues.cell(~Tile.Tile8),~cell_view(~Extra,~point),count,position,step,
      SlotsView{slots,phase,extra},model) : SlotsView<Extra>}:
  match count:
    case 0n:
      Equal.cong(U32,SlotsView<Extra>,word => SlotsView{slots,word,extra},(phase + U32.from_nat(0n) : U32),phase,Arithmetic.add_zero(phase))
    case 1n+rest:
      +start = (base + phase * step : U32)
      +stepped = Tree.write(Sum.TileParts,slots,U32.and(phase,7),
        Sum.tile_slot_step(Tree.read(Sum.TileParts,slots,U32.and(phase,7)),point(extra,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,model),start))))
      %at :
        {SlotsView{Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,
              Loop.run(~Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Extra>>,
                ~ArrayLaws.model_update_step(~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~spaced_sample(~Extra,~point),~Sum.tile_slot_step,~Operations.slot_of(~Extra)),
                1n+rest,Traversal.State{Buffer.tree(~Tile.Tile8,model),slots,phase,Traversal.Spaced{base,step,extra}})),(phase + U32.from_nat(1n+rest) : U32),extra} ==
          LineChainValues.strided(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~SlotsView<Extra>,~ReduceValues.cell(~Tile.Tile8),~cell_view(~Extra,~point),1n+rest,_,step,
            SlotsView{slots,phase,extra},model) : SlotsView<Extra>}
      %Equal.sym(U32,(phase + U32.from_nat(1n+rest) : U32),((phase + 1 : U32) + U32.from_nat(rest) : U32),LineChainValues.next_position(phase,rest)) :
        {SlotsView{Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,
              Loop.run(~Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Extra>>,
                ~ArrayLaws.model_update_step(~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~spaced_sample(~Extra,~point),~Sum.tile_slot_step,~Operations.slot_of(~Extra)),
                rest,Traversal.State{Buffer.tree(~Tile.Tile8,model),stepped,(phase + 1 : U32),Traversal.Spaced{base,step,extra}})),_,extra} ==
          LineChainValues.strided(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~SlotsView<Extra>,~ReduceValues.cell(~Tile.Tile8),~cell_view(~Extra,~point),rest,(start + step : U32),step,
            SlotsView{stepped,(phase + 1 : U32),extra},model) : SlotsView<Extra>}
      updates_strided(~Extra,~point,rest,(phase + 1 : U32),base,step,extra,stepped,(start + step : U32),model,
        Equal.trans(U32,(base + (phase + 1 : U32) * step : U32),(base + (phase * step + step : U32) : U32),(start + step : U32),
          Equal.cong(U32,U32,word => (base + word : U32),((phase + 1 : U32) * step : U32),(phase * step + step : U32),Arithmetic.multiply_successor(phase,step)),
          Equal.sym(U32,(start + step : U32),(base + (phase * step + step : U32) : U32),Arithmetic.add_associative(base,(phase * step : U32),step))))

# The view of a fold of cells into the slots steps the view by each cell of
# the model, step apart from position.
def fold_viewed(~Extra: Data,~read: Array<Tile.Tile8> -> Traversal.Spaced<Extra> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,
  ~read_preserves: @-reader: Array<Tile.Tile8> -> @context: Traversal.Spaced<Extra> -> @index: U32 -> TraversalWitness.array_evidence(~Tile.Tile8,reader) ->
    TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,reader,read(reader,context,index)>,
  ~point: Extra -> Tile.Tile8 -> Tile.Tile8,
  ~read_matches: @+reader: Tree.Tree<Tile.Tile8> -> @+context: Traversal.Spaced<Extra> -> @+position: U32 ->
    {read(ArrayLaws.pack_tree(~Tile.Tile8,reader),context,position) == (ArrayLaws.pack_tree(~Tile.Tile8,reader),spaced_sample(~Extra,~point,reader,context,position)) : Array<Tile.Tile8> & Tile.Tile8},
  +count: Nat,+position: U32,+step: U32,+value: SlotsValue<Extra>,+model: Buffer.Model<Tile.Tile8>) ->
  {slots_view(~Extra,fold_value(~Extra,~read,~read_preserves,count,position,step,value,model)) ==
    LineChainValues.strided_model(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~SlotsView<Extra>,~ReduceValues.cell(~Tile.Tile8),~cell_view(~Extra,~point),count,position,step,
      slots_view(~Extra,value),model) : SlotsView<Extra>}:
  SlotsValue{parts,+phase,+extra} = value
  Buffer.Model{+slots,+slots_length,Certificate.Certificate{+slots_depth,+slots_equation}} = parts
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  +context = {Traversal.Spaced{(position - phase * step : U32),step,extra} : Traversal.Spaced<Extra>}
  +run = {Traversal.Steps{count,phase,context} : Traversal.Steps<Traversal.Spaced<Extra>>}
  +start = {Traversal.State{tree,slots,phase,context} : Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Extra>>}
  +walked = Loop.run(~Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Extra>>,
    ~ArrayLaws.model_update_step(~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~spaced_sample(~Extra,~point),~Sum.tile_slot_step,~Operations.slot_of(~Extra)),count,start)
  +certificate = {Certificate.Certificate{depth,equation} : Certificate.Certificate<Tile.Tile8,Tree.pack(Tile.Tile8,tree)>}
  result = Traversal.updated(~Array<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~read,~Sum.tile_slot_step,~Operations.slot_of(~Extra),run,
    Tree.pack(Tile.Tile8,tree),Tree.pack(Sum.TileParts,slots))
  +after = (phase + U32.from_nat(count) : U32)
  -target = LineChainValues.strided_model(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~SlotsView<Extra>,~ReduceValues.cell(~Tile.Tile8),~cell_view(~Extra,~point),count,position,step,
    SlotsView{slots,phase,extra},Buffer.Model{tree,length,certificate})
  Equal.trans(SlotsView<Extra>,slots_view(~Extra,fold_value(~Extra,~read,~read_preserves,count,position,step,SlotsValue{Buffer.Model{slots,slots_length,Certificate.Certificate{slots_depth,slots_equation}},phase,extra},
      Buffer.Model{tree,length,certificate})),
    SlotsView{Tree.reflect(Sum.TileParts,TraversalWitness.second(~Array<Tile.Tile8>,~Array<Sum.TileParts>,result)),after,extra},target,
    buffers_view(~Extra,length,slots_length,after,extra,result,
      TraversalCertificate.updated_certificates(~Array<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~TraversalWitness.array_evidence(~Tile.Tile8),~Certificate.family(~Tile.Tile8),
        ~read,~Sum.tile_slot_step,~Operations.slot_of(~Extra),~read_preserves,~TraversalCertificate.array_observed(~Tile.Tile8),run,Tree.pack(Tile.Tile8,tree),Tree.pack(Sum.TileParts,slots),
        certificate,certificate,certificate,Certificate.Certificate{slots_depth,slots_equation})),
    %Equal.sym(Traversal.State<&1,&1,Array<Tile.Tile8>,Array<Sum.TileParts>,Traversal.Spaced<Extra>>,
        Traversal.update(~Array<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~read,~Sum.tile_slot_step,~Operations.slot_of(~Extra),count,
          ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,~ArrayLaws.pack_tree(~Tile.Tile8),start)),
        ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,~ArrayLaws.pack_tree(~Tile.Tile8),walked),
      ArrayLaws.update_refines_model(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~ArrayLaws.pack_tree(~Tile.Tile8),~read,
        ~spaced_sample(~Extra,~point),~Sum.tile_slot_step,~Operations.slot_of(~Extra),~read_matches,count,start)) :
      {SlotsView{Tree.reflect(Sum.TileParts,TraversalWitness.second(~Array<Tile.Tile8>,~Array<Sum.TileParts>,Traversal.finish(~Array<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,_))),after,extra} ==
        target : SlotsView<Extra>}
    %Equal.sym(Tree.Tree<Sum.TileParts>,
        Tree.reflect(Sum.TileParts,TraversalWitness.second(~Array<Tile.Tile8>,~Array<Sum.TileParts>,Traversal.finish(~Array<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,
          ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,~ArrayLaws.pack_tree(~Tile.Tile8),walked)))),
        Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Sum.TileParts,~Traversal.Spaced<Extra>,walked),finished_tree(~Extra,walked)) :
      {SlotsView{_,after,extra} == target : SlotsView<Extra>}
    updates_strided(~Extra,~point,count,phase,(position - phase * step : U32),step,extra,slots,position,Buffer.Model{tree,length,certificate},
      Subtraction.add_subtract(position,(phase * step : U32))))

def begin_viewed(~Extra: Data,+extra: Extra) -> {slots_view(~Extra,begin_value(~Extra,extra)) == begin_view(~Extra,extra) : SlotsView<Extra>}:
  {==}

def turn_viewed(~Extra: Data,+value: SlotsValue<Extra>) -> {turn_value(~Extra,value) == turn_view(~Extra,slots_view(~Extra,value)) : Sum.TileParts}:
  SlotsValue{parts,+phase,+extra} = value
  Buffer.Model{+slots,+slots_length,slots_certificate} = parts
  {==}

# ---- Folds of lines into slots ----

# A fold of lines into slots has the blocks and finals of the folds of the
# model by the view of the slots.
def slots_chain(~Extra: Data,~read: Array<Tile.Tile8> -> Traversal.Spaced<Extra> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,
  ~read_preserves: @-reader: Array<Tile.Tile8> -> @context: Traversal.Spaced<Extra> -> @index: U32 -> TraversalWitness.array_evidence(~Tile.Tile8,reader) ->
    TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,reader,read(reader,context,index)>,
  ~point: Extra -> Tile.Tile8 -> Tile.Tile8,
  ~read_matches: @+reader: Tree.Tree<Tile.Tile8> -> @+context: Traversal.Spaced<Extra> -> @+position: U32 ->
    {read(ArrayLaws.pack_tree(~Tile.Tile8,reader),context,position) == (ArrayLaws.pack_tree(~Tile.Tile8,reader),spaced_sample(~Extra,~point,reader,context,position)) : Array<Tile.Tile8> & Tile.Tile8},
  +contexts: List<&2,Extra>,+line: LineChain.Line,+group: U32,+bytes: F32,+operations: F32,+grid: Grid.Grid,+fill: Tile.Tile8,+workers: U32,+tree: Partition.Tree<&2,Buffer.Model<Tile.Tile8>,Unit>) ->
  {LineChain.folds(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),
      ~Operations.cells_slots(~Extra,~read,~read_preserves),~Operations.slots_closed(~Extra),~Pieces.vacant_buffer(~Tile.Tile8),contexts,line,group,bytes,operations,grid,fill,workers,
      Regrouped.tree_of(~Storage.Buffer<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Buffer.pack(~Tile.Tile8),tree)) ==
    LineChainValues.unzipped_pair(~Storage.Buffer<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Buffer.pack(~Tile.Tile8),~Sum.TileParts,
      LineChainValues.folds_model(~Buffer.Model<Tile.Tile8>,~Extra,~SlotsView<Extra>,~Sum.TileParts,~begin_view(~Extra),
        ~LineChainValues.strided_model(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~SlotsView<Extra>,~ReduceValues.cell(~Tile.Tile8),~cell_view(~Extra,~point)),~turn_view(~Extra),
        ~Tile.Tile8,~ReduceValues.vacant_model(~Tile.Tile8),contexts,line,group,bytes,operations,grid,fill,workers,tree)) :
    Partition.Tree<&1,Storage.Buffer<Tile.Tile8>,Unit> & List<&2,Sum.TileParts>}:
  LineChainValues.chain_viewed(~Buffer.Model<Tile.Tile8>,~Extra,~SlotsValue<Extra>,~SlotsView<Extra>,~Sum.TileParts,~slots_view(~Extra),~begin_value(~Extra),~begin_view(~Extra),
    ~fold_value(~Extra,~read,~read_preserves),~LineChainValues.strided_model(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~SlotsView<Extra>,~ReduceValues.cell(~Tile.Tile8),~cell_view(~Extra,~point)),
    ~turn_value(~Extra),~turn_view(~Extra),~begin_viewed(~Extra),~fold_viewed(~Extra,~read,~read_preserves,~point,~read_matches),~turn_viewed(~Extra),
    ~Storage.Buffer<Tile.Tile8>,~Buffer.pack(~Tile.Tile8),~Operations.TileSlots<Extra>,~Operations.slots_begun(~Extra),~Operations.cells_slots(~Extra,~read,~read_preserves),
    ~Operations.slots_closed(~Extra),~Tile.Tile8,~Pieces.vacant_buffer(~Tile.Tile8),~ReduceValues.vacant_model(~Tile.Tile8),
    ~LineChainValues.folds_same(~Storage.Buffer<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Buffer.pack(~Tile.Tile8),~Extra,~Operations.TileSlots<Extra>,~SlotsValue<Extra>,~shown(~Extra),
      ~Sum.TileParts,~Operations.slots_begun(~Extra),~begin_value(~Extra),~Operations.cells_slots(~Extra,~read,~read_preserves),~fold_value(~Extra,~read,~read_preserves),
      ~Operations.slots_closed(~Extra),~turn_value(~Extra),~slots_same(~Extra,~read,~read_preserves),~begun_same(~Extra),~closed_same(~Extra),
      ~Tile.Tile8,~Pieces.vacant_buffer(~Tile.Tile8),~ReduceValues.vacant_model(~Tile.Tile8),~ReduceValues.vacant_same(~Tile.Tile8)),
    contexts,line,group,bytes,operations,grid,fill,workers,tree)
