# Two tensors of cells of one layout read together: the products of their
# cells fold into the slots of a row, hand both buffers back and stream the
# products of the values of two readers.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_batch_norm.bend as Kernel
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../storage_buffer.bend as Storage
import ../tensor_pieces.bend as Pieces
import ../tensor_operations.bend as Operations
import ../math_sum.bend as Sum
import ../tensor_line_chain.bend as LineChain
import ../traversal_partition.bend as Partition
import ../tensor_grid.bend as Grid
import ../tensor_batch_norm_cells.bend as BNCells
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./storage_buffer_model.bend as Buffer
import ./storage_certificate.bend as Certificate
import ./traversal_certificate.bend as TraversalCertificate
import ./traversal_witness.bend as TraversalWitness
import ./traversal_update_readback.bend as Updates
import ./kernel_batch_norm_certificate.bend as KernelCertificate
import ./tensor_reduce_values.bend as ReduceValues
import ./tensor_line_chain_values.bend as LineChainValues
import ./tensor_tile_slots_values.bend as SlotsValues
import ./tensor_pieces_each.bend as Each
import ./tensor_pieces_regrouped.bend as Regrouped
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction
import ./tensor_pieces_combine_leaf.bend as CombineLeaf
import ../kernel_gemm.bend as KernelGemm
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_pieces_model.bend as PiecesModel

# ---- Two trees ----

# The trees of two arrays.
type Trees is Data:
  Trees{left: Tree.Tree<Tile.Tile8>,right: Tree.Tree<Tile.Tile8>}

def packed(trees: Trees) -> Kernel.Leaves():
  Trees{left,right} = trees
  (Tree.pack(Tile.Tile8,left),Tree.pack(Tile.Tile8,right))

# The product of the cells of both trees at base + position * stride.
def product_sample(trees: Trees,context: Traversal.Spaced<Unit>,position: U32) -> Tile.Tile8:
  Trees{left,right} = trees
  Traversal.Spaced{+base,+stride,extra} = context
  +index = (base + position * stride : U32)
  Tile.times(Tree.read(Tile.Tile8,left,index),Tree.read(Tile.Tile8,right,index))

# Reading the products of two packed trees hands them back with the product.
def product_matches(+trees: Trees,+context: Traversal.Spaced<Unit>,+position: U32) ->
  {Kernel.read_product(packed(trees),context,position) == (packed(trees),product_sample(trees,context,position)) : Kernel.Leaves() & Tile.Tile8}:
  Trees{+left,+right} = trees
  Traversal.Spaced{+base,+stride,extra} = context
  +index = (base + position * stride : U32)
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,left),index),(Tree.pack(Tile.Tile8,left),Tree.read(Tile.Tile8,left,index)),StorageLaws.read(Tile.Tile8,left,index)) :
    {Kernel.product_left(Tree.pack(Tile.Tile8,right),index,_) == ((Tree.pack(Tile.Tile8,left),Tree.pack(Tile.Tile8,right)),
      Tile.times(Tree.read(Tile.Tile8,left,index),Tree.read(Tile.Tile8,right,index))) : Kernel.Leaves() & Tile.Tile8}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,right),index),(Tree.pack(Tile.Tile8,right),Tree.read(Tile.Tile8,right,index)),StorageLaws.read(Tile.Tile8,right,index)) :
    {Kernel.product_right(Tree.pack(Tile.Tile8,left),Tree.read(Tile.Tile8,left,index),_) == ((Tree.pack(Tile.Tile8,left),Tree.pack(Tile.Tile8,right)),
      Tile.times(Tree.read(Tile.Tile8,left,index),Tree.read(Tile.Tile8,right,index))) : Kernel.Leaves() & Tile.Tile8}
  {==}

# ---- Two buffer models ----

def trees(model: Each.PiecePair<Tile.Tile8>) -> Trees:
  Each.PiecePair{left,right} = model
  Trees{Buffer.tree(~Tile.Tile8,left),Buffer.tree(~Tile.Tile8,right)}

# The product of the cells of both models at a position.
def product_cell(model: Each.PiecePair<Tile.Tile8>,position: U32) -> Tile.Tile8:
  Each.PiecePair{left,right} = model
  +at = position
  Tile.times(ReduceValues.cell(~Tile.Tile8,left,at),ReduceValues.cell(~Tile.Tile8,right,at))

# The pair of buffers a vacant piece holds, as models.
def vacant_twin(fill: Tile.Tile8) -> Each.PiecePair<Tile.Tile8>:
  +tile = fill
  Each.PiecePair{ReduceValues.vacant_model(~Tile.Tile8,tile),ReduceValues.vacant_model(~Tile.Tile8,tile)}

def vacant_same(+fill: Tile.Tile8) -> {BNCells.vacant_pair(fill) == Each.pair_pack(~Tile.Tile8,vacant_twin(fill)) : BNCells.CellPair()}:
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Pieces.vacant_buffer(~Tile.Tile8,fill),Buffer.pack(~Tile.Tile8,ReduceValues.vacant_model(~Tile.Tile8,fill)),ReduceValues.vacant_same(~Tile.Tile8,fill)) :
    {Pieces.Both{_,Pieces.vacant_buffer(~Tile.Tile8,fill)} == Each.pair_pack(~Tile.Tile8,vacant_twin(fill)) : BNCells.CellPair()}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Pieces.vacant_buffer(~Tile.Tile8,fill),Buffer.pack(~Tile.Tile8,ReduceValues.vacant_model(~Tile.Tile8,fill)),ReduceValues.vacant_same(~Tile.Tile8,fill)) :
    {Pieces.Both{Buffer.pack(~Tile.Tile8,ReduceValues.vacant_model(~Tile.Tile8,fill)),_} == Each.pair_pack(~Tile.Tile8,vacant_twin(fill)) : BNCells.CellPair()}
  {==}

# ---- The buffers handed back ----

# The certificates of two arrays carried to the arrays handed back.
def owned_returned(-reader: Kernel.Leaves(),-returned: Kernel.Leaves(),owned: KernelCertificate.LeavesOwned<reader>,same: {returned == reader : Kernel.Leaves()}) ->
  KernelCertificate.LeavesOwned<returned>:
  %Equal.sym(Kernel.Leaves(),returned,reader,same) : KernelCertificate.LeavesOwned<_>
  owned

# The buffers of two arrays handed back with carried certificates are the
# buffers of the arrays.
def pair_returned(+left_length: Nat,+right_length: Nat,-left: Array<Tile.Tile8>,-right: Array<Tile.Tile8>,-returned: Kernel.Leaves(),+same: {returned == (left,right) : Kernel.Leaves()}) ->
  @owned: KernelCertificate.LeavesOwned<(left,right)> ->
  {BNCells.both_of(BNCells.leaves_buffers(left_length,right_length,returned,KernelCertificate.owned_certificates(returned,owned_returned((left,right),returned,owned,same)))) ==
    BNCells.both_of(BNCells.leaves_buffers(left_length,right_length,(left,right),KernelCertificate.owned_certificates((left,right),owned))) : BNCells.CellPair()}:
  %proof@same : @owned: KernelCertificate.LeavesOwned<_> ->
    {BNCells.both_of(BNCells.leaves_buffers(left_length,right_length,returned,KernelCertificate.owned_certificates(returned,owned_returned(_,returned,owned,proof)))) ==
      BNCells.both_of(BNCells.leaves_buffers(left_length,right_length,_,KernelCertificate.owned_certificates(_,owned))) : BNCells.CellPair()}
  owned => {==}

# The pair of a run of products into the slots is the pair of buffers it read.
def pair_slotted_kept(+left_length: Nat,+right_length: Nat,+slots_length: Nat,+phase: U32,-left: Array<Tile.Tile8>,-right: Array<Tile.Tile8>,result: Kernel.Leaves() & Array<Sum.TileParts>,
  +owned: KernelCertificate.LeavesOwned<(left,right)>,
  output_certificate: Certificate.Certificate<Sum.TileParts,TraversalWitness.second(~Kernel.Leaves(),~Array<Sum.TileParts>,result)>,
  +same: {TraversalWitness.first(~Kernel.Leaves(),~Array<Sum.TileParts>,result) == (left,right) : Kernel.Leaves()}) ->
  {BNCells.product_slotted(left_length,right_length,slots_length,phase,result,
      TraversalCertificate.Certificates{owned_returned((left,right),TraversalWitness.first(~Kernel.Leaves(),~Array<Sum.TileParts>,result),owned,same),output_certificate}) ==
    (BNCells.both_of(BNCells.leaves_buffers(left_length,right_length,(left,right),KernelCertificate.owned_certificates((left,right),owned))),
      Pair.snd(BNCells.CellPair(),Operations.TileSlots<Unit>,BNCells.product_slotted(left_length,right_length,slots_length,phase,result,
        TraversalCertificate.Certificates{owned_returned((left,right),TraversalWitness.first(~Kernel.Leaves(),~Array<Sum.TileParts>,result),owned,same),output_certificate}))) :
    BNCells.CellPair() & Operations.TileSlots<Unit>}:
  (returned,output) = result
  Equal.cong(BNCells.CellPair(),BNCells.CellPair() & Operations.TileSlots<Unit>,pair => (pair,Operations.TileSlots{Storage.Buffer{output,slots_length,output_certificate},phase,Unit{}}),
    BNCells.both_of(BNCells.leaves_buffers(left_length,right_length,returned,KernelCertificate.owned_certificates(returned,owned_returned((left,right),returned,owned,same)))),
    BNCells.both_of(BNCells.leaves_buffers(left_length,right_length,(left,right),KernelCertificate.owned_certificates((left,right),owned))),
    pair_returned(left_length,right_length,left,right,returned,same,owned))

# A fold of products of two buffers into the slots hands both buffers back.
def pair_kept(+count: Nat,+position: U32,+step: U32,+value: SlotsValues.SlotsValue<Unit>,+model: Each.PiecePair<Tile.Tile8>) ->
  {BNCells.product_slots(count,position,step,SlotsValues.shown(~Unit,value),Each.pair_pack(~Tile.Tile8,model)) ==
    (Each.pair_pack(~Tile.Tile8,model),Pair.snd(BNCells.CellPair(),Operations.TileSlots<Unit>,
      BNCells.product_slots(count,position,step,SlotsValues.shown(~Unit,value),Each.pair_pack(~Tile.Tile8,model)))) : BNCells.CellPair() & Operations.TileSlots<Unit>}:
  SlotsValues.SlotsValue{parts,+phase,extra} = value
  Buffer.Model{+slots,+slots_length,Certificate.Certificate{+slots_depth,+slots_equation}} = parts
  Each.PiecePair{left,right} = model
  Buffer.Model{+left_tree,+left_length,Certificate.Certificate{+left_depth,+left_equation}} = left
  Buffer.Model{+right_tree,+right_length,Certificate.Certificate{+right_depth,+right_equation}} = right
  +run = {Traversal.Steps{count,phase,Traversal.Spaced{(position - phase * step : U32),step,Unit{}}} : Traversal.Steps<Traversal.Spaced<Unit>>}
  +owned = KernelCertificate.owned_leaves(Tree.pack(Tile.Tile8,left_tree),Tree.pack(Tile.Tile8,right_tree),left_depth,left_equation,right_depth,right_equation)
  result = Traversal.updated(~Kernel.Leaves(),~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~Kernel.read_product,~Sum.tile_slot_step,~Operations.slot_of(~Unit),run,
    (Tree.pack(Tile.Tile8,left_tree),Tree.pack(Tile.Tile8,right_tree)),Tree.pack(Sum.TileParts,slots))
  pair_slotted_kept(left_length,right_length,slots_length,(phase + U32.from_nat(count) : U32),Tree.pack(Tile.Tile8,left_tree),Tree.pack(Tile.Tile8,right_tree),result,owned,
    Certificate.Certificate{slots_depth,TraversalCertificate.operation_equation(~Kernel.Leaves(),~Sum.TileParts,~Traversal.Steps<Traversal.Spaced<Unit>>,~KernelCertificate.leaves_owned,
      ~Traversal.updated(~Kernel.Leaves(),~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~Kernel.read_product,~Sum.tile_slot_step,~Operations.slot_of(~Unit)),
      ~TraversalCertificate.update_preserves(~Kernel.Leaves(),~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~KernelCertificate.leaves_evidence,~KernelCertificate.leaves_owned,
        ~Kernel.read_product,~Sum.tile_slot_step,~Operations.slot_of(~Unit),~KernelCertificate.read_product,~KernelCertificate.observe_leaves),
      run,(Tree.pack(Tile.Tile8,left_tree),Tree.pack(Tile.Tile8,right_tree)),Tree.pack(Sum.TileParts,slots),owned,slots_depth,slots_equation)},
    TraversalWitness.source_unchanged(~Kernel.Leaves(),~Sum.TileParts,(Tree.pack(Tile.Tile8,left_tree),Tree.pack(Tile.Tile8,right_tree)),slots_depth,result,
      TraversalCertificate.update_preserves(~Kernel.Leaves(),~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~KernelCertificate.leaves_evidence,~KernelCertificate.leaves_owned,
        ~Kernel.read_product,~Sum.tile_slot_step,~Operations.slot_of(~Unit),~KernelCertificate.read_product,~KernelCertificate.observe_leaves,
        slots_depth,run,(Tree.pack(Tile.Tile8,left_tree),Tree.pack(Tile.Tile8,right_tree)),Tree.pack(Sum.TileParts,slots),owned,slots_equation)))

# ---- The line chain laws of the products ----

# The slots of a fold of products.
def pair_fold_value(count: Nat,position: U32,step: U32,value: SlotsValues.SlotsValue<Unit>,model: Each.PiecePair<Tile.Tile8>) -> SlotsValues.SlotsValue<Unit>:
  SlotsValues.valued(~Unit,Pair.snd(BNCells.CellPair(),Operations.TileSlots<Unit>,
    BNCells.product_slots(count,position,step,SlotsValues.shown(~Unit,value),Each.pair_pack(~Tile.Tile8,model))))

def pair_same(+count: Nat,+position: U32,+step: U32,+initial: SlotsValues.SlotsValue<Unit>,+model: Each.PiecePair<Tile.Tile8>) ->
  {BNCells.product_slots(count,position,step,SlotsValues.shown(~Unit,initial),Each.pair_pack(~Tile.Tile8,model)) ==
    (Each.pair_pack(~Tile.Tile8,model),SlotsValues.shown(~Unit,pair_fold_value(count,position,step,initial,model))) : BNCells.CellPair() & Operations.TileSlots<Unit>}:
  Equal.trans(BNCells.CellPair() & Operations.TileSlots<Unit>,
    BNCells.product_slots(count,position,step,SlotsValues.shown(~Unit,initial),Each.pair_pack(~Tile.Tile8,model)),
    (Each.pair_pack(~Tile.Tile8,model),Pair.snd(BNCells.CellPair(),Operations.TileSlots<Unit>,
      BNCells.product_slots(count,position,step,SlotsValues.shown(~Unit,initial),Each.pair_pack(~Tile.Tile8,model)))),
    (Each.pair_pack(~Tile.Tile8,model),SlotsValues.shown(~Unit,pair_fold_value(count,position,step,initial,model))),
    pair_kept(count,position,step,initial,model),
    Equal.cong(Operations.TileSlots<Unit>,BNCells.CellPair() & Operations.TileSlots<Unit>,slots => (Each.pair_pack(~Tile.Tile8,model),slots),
      Pair.snd(BNCells.CellPair(),Operations.TileSlots<Unit>,BNCells.product_slots(count,position,step,SlotsValues.shown(~Unit,initial),Each.pair_pack(~Tile.Tile8,model))),
      SlotsValues.shown(~Unit,pair_fold_value(count,position,step,initial,model)),
      Equal.sym(Operations.TileSlots<Unit>,SlotsValues.shown(~Unit,pair_fold_value(count,position,step,initial,model)),
        Pair.snd(BNCells.CellPair(),Operations.TileSlots<Unit>,BNCells.product_slots(count,position,step,SlotsValues.shown(~Unit,initial),Each.pair_pack(~Tile.Tile8,model))),
        SlotsValues.valued_shown(~Unit,Pair.snd(BNCells.CellPair(),Operations.TileSlots<Unit>,
          BNCells.product_slots(count,position,step,SlotsValues.shown(~Unit,initial),Each.pair_pack(~Tile.Tile8,model)))))))

# The slots of the parts handed back are the tree of their array.
def pair_view(+left_length: Nat,+right_length: Nat,+slots_length: Nat,+phase: U32,result: Kernel.Leaves() & Array<Sum.TileParts>,
  certificates: TraversalCertificate.Certificates<Kernel.Leaves(),Sum.TileParts,KernelCertificate.leaves_owned,result>) ->
  {SlotsValues.slots_view(~Unit,SlotsValues.valued(~Unit,Pair.snd(BNCells.CellPair(),Operations.TileSlots<Unit>,
      BNCells.product_slotted(left_length,right_length,slots_length,phase,result,certificates)))) ==
    SlotsValues.SlotsView{Tree.reflect(Sum.TileParts,TraversalWitness.second(~Kernel.Leaves(),~Array<Sum.TileParts>,result)),phase,Unit{}} : SlotsValues.SlotsView<Unit>}:
  (leaves,slots) = result
  TraversalCertificate.Certificates{owned,slots_certificate} = certificates
  Equal.cong(Buffer.Content<Sum.TileParts>,SlotsValues.SlotsView<Unit>,content => SlotsValues.SlotsView{Buffer.content_tree(~Sum.TileParts,content),phase,Unit{}},
    Buffer.content(~Sum.TileParts,Buffer.remodel(~Sum.TileParts,Storage.Buffer{slots,slots_length,slots_certificate})),Buffer.Content{Tree.reflect(Sum.TileParts,slots),slots_length},
    Buffer.remodel_content(~Sum.TileParts,Storage.Buffer{slots,slots_length,slots_certificate}))

# The parts of a finished model run of products are the tree of its output.
def pair_finished_tree(+state: Traversal.State<&2,&2,Trees,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Unit>>) ->
  {Tree.reflect(Sum.TileParts,TraversalWitness.second(~Kernel.Leaves(),~Array<Sum.TileParts>,Traversal.finish(~Kernel.Leaves(),~Sum.TileParts,~Traversal.Spaced<Unit>,
      ArrayLaws.represent(~Kernel.Leaves(),~Trees,~Sum.TileParts,~Traversal.Spaced<Unit>,~packed,state)))) ==
    Updates.state_tree(~Trees,~Sum.TileParts,~Traversal.Spaced<Unit>,state) : Tree.Tree<Sum.TileParts>}:
  Traversal.State{reader,+output,position,context} = state
  StorageLaws.reflect_packed(Sum.TileParts,output)

# count products from the phase on, at base + phase * step, step the slots as
# the view steps the products of the models from the position they are at.
def product_strided(+model: Each.PiecePair<Tile.Tile8>,+count: Nat,+phase: U32,+base: U32,+step: U32,+slots: Tree.Tree<Sum.TileParts>,+position: U32,
  +at: {(base + phase * step : U32) == position : U32}) ->
  {SlotsValues.SlotsView{Updates.state_tree(~Trees,~Sum.TileParts,~Traversal.Spaced<Unit>,
        Loop.run(~Traversal.State<&2,&2,Trees,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Unit>>,
          ~ArrayLaws.model_update_step(~Trees,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~product_sample,~Sum.tile_slot_step,~Operations.slot_of(~Unit)),
          count,Traversal.State{trees(model),slots,phase,Traversal.Spaced{base,step,Unit{}}})),(phase + U32.from_nat(count) : U32),Unit{}} ==
    LineChainValues.strided(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~SlotsValues.SlotsView<Unit>,~product_cell,~SlotsValues.cell_view(~Unit,~SlotsValues.held(~Unit)),count,position,step,
      SlotsValues.SlotsView{slots,phase,Unit{}},model) : SlotsValues.SlotsView<Unit>}:
  Each.PiecePair{+left,+right} = model
  match count:
    case 0n:
      Equal.cong(U32,SlotsValues.SlotsView<Unit>,word => SlotsValues.SlotsView{slots,word,Unit{}},(phase + U32.from_nat(0n) : U32),phase,Arithmetic.add_zero(phase))
    case 1n+rest:
      +pair = {Each.PiecePair{left,right} : Each.PiecePair<Tile.Tile8>}
      +start = (base + phase * step : U32)
      +stepped = Tree.write(Sum.TileParts,slots,U32.and(phase,7),
        Sum.tile_slot_step(Tree.read(Sum.TileParts,slots,U32.and(phase,7)),Tile.times(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,left),start),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,right),start))))
      %at :
        {SlotsValues.SlotsView{Updates.state_tree(~Trees,~Sum.TileParts,~Traversal.Spaced<Unit>,
              Loop.run(~Traversal.State<&2,&2,Trees,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Unit>>,
                ~ArrayLaws.model_update_step(~Trees,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~product_sample,~Sum.tile_slot_step,~Operations.slot_of(~Unit)),
                1n+rest,Traversal.State{trees(pair),slots,phase,Traversal.Spaced{base,step,Unit{}}})),(phase + U32.from_nat(1n+rest) : U32),Unit{}} ==
          LineChainValues.strided(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~SlotsValues.SlotsView<Unit>,~product_cell,~SlotsValues.cell_view(~Unit,~SlotsValues.held(~Unit)),1n+rest,_,step,
            SlotsValues.SlotsView{slots,phase,Unit{}},pair) : SlotsValues.SlotsView<Unit>}
      %Equal.sym(U32,(phase + U32.from_nat(1n+rest) : U32),((phase + 1 : U32) + U32.from_nat(rest) : U32),LineChainValues.next_position(phase,rest)) :
        {SlotsValues.SlotsView{Updates.state_tree(~Trees,~Sum.TileParts,~Traversal.Spaced<Unit>,
              Loop.run(~Traversal.State<&2,&2,Trees,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Unit>>,
                ~ArrayLaws.model_update_step(~Trees,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~product_sample,~Sum.tile_slot_step,~Operations.slot_of(~Unit)),
                rest,Traversal.State{trees(pair),stepped,(phase + 1 : U32),Traversal.Spaced{base,step,Unit{}}})),_,Unit{}} ==
          LineChainValues.strided(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~SlotsValues.SlotsView<Unit>,~product_cell,~SlotsValues.cell_view(~Unit,~SlotsValues.held(~Unit)),rest,(start + step : U32),step,
            SlotsValues.SlotsView{stepped,(phase + 1 : U32),Unit{}},pair) : SlotsValues.SlotsView<Unit>}
      product_strided(pair,rest,(phase + 1 : U32),base,step,stepped,(start + step : U32),
        Equal.trans(U32,(base + (phase + 1 : U32) * step : U32),(base + (phase * step + step : U32) : U32),(start + step : U32),
          Equal.cong(U32,U32,word => (base + word : U32),((phase + 1 : U32) * step : U32),(phase * step + step : U32),Arithmetic.multiply_successor(phase,step)),
          Equal.sym(U32,(start + step : U32),(base + (phase * step + step : U32) : U32),Arithmetic.add_associative(base,(phase * step : U32),step))))

# The view of a fold of products into the slots steps the view by each
# product of the models, step apart from position.
def pair_fold_viewed(+count: Nat,+position: U32,+step: U32,+value: SlotsValues.SlotsValue<Unit>,+model: Each.PiecePair<Tile.Tile8>) ->
  {SlotsValues.slots_view(~Unit,pair_fold_value(count,position,step,value,model)) ==
    LineChainValues.strided_model(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~SlotsValues.SlotsView<Unit>,~product_cell,~SlotsValues.cell_view(~Unit,~SlotsValues.held(~Unit)),count,position,step,
      SlotsValues.slots_view(~Unit,value),model) : SlotsValues.SlotsView<Unit>}:
  SlotsValues.SlotsValue{Buffer.Model{+slots,+slots_length,Certificate.Certificate{+slots_depth,+slots_equation}},+phase,Unit{}} = value
  Each.PiecePair{left,right} = model
  Buffer.Model{+left_tree,+left_length,Certificate.Certificate{+left_depth,+left_equation}} = left
  Buffer.Model{+right_tree,+right_length,Certificate.Certificate{+right_depth,+right_equation}} = right
  +context = {Traversal.Spaced{(position - phase * step : U32),step,Unit{}} : Traversal.Spaced<Unit>}
  +run = {Traversal.Steps{count,phase,context} : Traversal.Steps<Traversal.Spaced<Unit>>}
  +pair = {Each.PiecePair{Buffer.Model{left_tree,left_length,Certificate.Certificate{left_depth,left_equation}},Buffer.Model{right_tree,right_length,Certificate.Certificate{right_depth,right_equation}}} :
    Each.PiecePair<Tile.Tile8>}
  +start = {Traversal.State{Trees{left_tree,right_tree},slots,phase,context} : Traversal.State<&2,&2,Trees,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Unit>>}
  +walked = Loop.run(~Traversal.State<&2,&2,Trees,Tree.Tree<Sum.TileParts>,Traversal.Spaced<Unit>>,
    ~ArrayLaws.model_update_step(~Trees,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~product_sample,~Sum.tile_slot_step,~Operations.slot_of(~Unit)),count,start)
  result = Traversal.updated(~Kernel.Leaves(),~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~Kernel.read_product,~Sum.tile_slot_step,~Operations.slot_of(~Unit),run,
    (Tree.pack(Tile.Tile8,left_tree),Tree.pack(Tile.Tile8,right_tree)),Tree.pack(Sum.TileParts,slots))
  +after = (phase + U32.from_nat(count) : U32)
  -target = LineChainValues.strided_model(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~SlotsValues.SlotsView<Unit>,~product_cell,~SlotsValues.cell_view(~Unit,~SlotsValues.held(~Unit)),count,position,step,
    SlotsValues.SlotsView{slots,phase,Unit{}},pair)
  Equal.trans(SlotsValues.SlotsView<Unit>,
    SlotsValues.slots_view(~Unit,pair_fold_value(count,position,step,SlotsValues.SlotsValue{Buffer.Model{slots,slots_length,Certificate.Certificate{slots_depth,slots_equation}},phase,Unit{}},pair)),
    SlotsValues.SlotsView{Tree.reflect(Sum.TileParts,TraversalWitness.second(~Kernel.Leaves(),~Array<Sum.TileParts>,result)),after,Unit{}},target,
    pair_view(left_length,right_length,slots_length,after,result,
      KernelCertificate.product_updates(run,Tree.pack(Tile.Tile8,left_tree),Tree.pack(Tile.Tile8,right_tree),Tree.pack(Sum.TileParts,slots),
        Certificate.Certificate{left_depth,left_equation},Certificate.Certificate{right_depth,right_equation},Certificate.Certificate{slots_depth,slots_equation})),
    %Equal.sym(Traversal.State<&1,&1,Kernel.Leaves(),Array<Sum.TileParts>,Traversal.Spaced<Unit>>,
        Traversal.update(~Kernel.Leaves(),~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~Kernel.read_product,~Sum.tile_slot_step,~Operations.slot_of(~Unit),count,
          ArrayLaws.represent(~Kernel.Leaves(),~Trees,~Sum.TileParts,~Traversal.Spaced<Unit>,~packed,start)),
        ArrayLaws.represent(~Kernel.Leaves(),~Trees,~Sum.TileParts,~Traversal.Spaced<Unit>,~packed,walked),
      ArrayLaws.update_refines_model(~Kernel.Leaves(),~Trees,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~packed,~Kernel.read_product,
        ~product_sample,~Sum.tile_slot_step,~Operations.slot_of(~Unit),~product_matches,count,start)) :
      {SlotsValues.SlotsView{Tree.reflect(Sum.TileParts,TraversalWitness.second(~Kernel.Leaves(),~Array<Sum.TileParts>,Traversal.finish(~Kernel.Leaves(),~Sum.TileParts,~Traversal.Spaced<Unit>,_))),after,Unit{}} ==
        target : SlotsValues.SlotsView<Unit>}
    %Equal.sym(Tree.Tree<Sum.TileParts>,
        Tree.reflect(Sum.TileParts,TraversalWitness.second(~Kernel.Leaves(),~Array<Sum.TileParts>,Traversal.finish(~Kernel.Leaves(),~Sum.TileParts,~Traversal.Spaced<Unit>,
          ArrayLaws.represent(~Kernel.Leaves(),~Trees,~Sum.TileParts,~Traversal.Spaced<Unit>,~packed,walked)))),
        Updates.state_tree(~Trees,~Sum.TileParts,~Traversal.Spaced<Unit>,walked),pair_finished_tree(walked)) :
      {SlotsValues.SlotsView{_,after,Unit{}} == target : SlotsValues.SlotsView<Unit>}
    product_strided(pair,count,phase,(position - phase * step : U32),step,slots,position,Subtraction.add_subtract(position,(phase * step : U32))))

# ---- Folds of lines of products into slots ----

# A fold of lines of products into slots has the pairs and finals of the
# folds of the models by the view of the slots.
def pair_chain(+contexts: List<&2,Unit>,+line: LineChain.Line,+group: U32,+bytes: F32,+operations: F32,+grid: Grid.Grid,+fill: Tile.Tile8,+workers: U32,
  +tree: Partition.Tree<&2,Each.PiecePair<Tile.Tile8>,Unit>) ->
  {LineChain.folds(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),
      ~BNCells.vacant_pair,contexts,line,group,bytes,operations,grid,fill,workers,Regrouped.tree_of(~BNCells.CellPair(),~Each.PiecePair<Tile.Tile8>,~Each.pair_pack(~Tile.Tile8),tree)) ==
    LineChainValues.unzipped_pair(~BNCells.CellPair(),~Each.PiecePair<Tile.Tile8>,~Each.pair_pack(~Tile.Tile8),~Sum.TileParts,
      LineChainValues.folds_model(~Each.PiecePair<Tile.Tile8>,~Unit,~SlotsValues.SlotsView<Unit>,~Sum.TileParts,~SlotsValues.begin_view(~Unit),
        ~LineChainValues.strided_model(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~SlotsValues.SlotsView<Unit>,~product_cell,~SlotsValues.cell_view(~Unit,~SlotsValues.held(~Unit))),~SlotsValues.turn_view(~Unit),
        ~Tile.Tile8,~vacant_twin,contexts,line,group,bytes,operations,grid,fill,workers,tree)) : Partition.Tree<&1,BNCells.CellPair(),Unit> & List<&2,Sum.TileParts>}:
  LineChainValues.chain_viewed(~Each.PiecePair<Tile.Tile8>,~Unit,~SlotsValues.SlotsValue<Unit>,~SlotsValues.SlotsView<Unit>,~Sum.TileParts,~SlotsValues.slots_view(~Unit),~SlotsValues.begin_value(~Unit),
    ~SlotsValues.begin_view(~Unit),~pair_fold_value,
    ~LineChainValues.strided_model(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~SlotsValues.SlotsView<Unit>,~product_cell,~SlotsValues.cell_view(~Unit,~SlotsValues.held(~Unit))),
    ~SlotsValues.turn_value(~Unit),~SlotsValues.turn_view(~Unit),~SlotsValues.begin_viewed(~Unit),~pair_fold_viewed,~SlotsValues.turn_viewed(~Unit),
    ~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),~Operations.TileSlots<Unit>,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),
    ~Tile.Tile8,~BNCells.vacant_pair,~vacant_twin,
    ~LineChainValues.folds_same(~BNCells.CellPair(),~Each.PiecePair<Tile.Tile8>,~Each.pair_pack(~Tile.Tile8),~Unit,~Operations.TileSlots<Unit>,~SlotsValues.SlotsValue<Unit>,~SlotsValues.shown(~Unit),
      ~Sum.TileParts,~Operations.slots_begun(~Unit),~SlotsValues.begin_value(~Unit),~BNCells.product_slots,~pair_fold_value,
      ~Operations.slots_closed(~Unit),~SlotsValues.turn_value(~Unit),~pair_same,~SlotsValues.begun_same(~Unit),~SlotsValues.closed_same(~Unit),
      ~Tile.Tile8,~BNCells.vacant_pair,~vacant_twin,~vacant_same),
    contexts,line,group,bytes,operations,grid,fill,workers,tree)

# ---- Two tensors of one layout ----

# Two lists of cells, multiplied in order.
def products(left: List<&2,Tile.Tile8>,right: List<&2,Tile.Tile8>) -> List<&2,Tile.Tile8>:
  match left right:
    case Con{cell,rest} Con{other,more}: Tile.times(cell,other) <> products(rest,more)
    case left right: []

# The products of the cells of a pair of models are the products of their cells.
def product_cells(count: Nat,+offset: U32,+left: Buffer.Model<Tile.Tile8>,+right: Buffer.Model<Tile.Tile8>) ->
  {LineChainValues.cells(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~product_cell,count,offset,Each.PiecePair{left,right}) == products(LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),count,offset,left),LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),count,offset,right)) : List<&2,Tile.Tile8>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,tail => product_cell(Each.PiecePair{left,right},offset) <> tail,
        LineChainValues.cells(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~product_cell,rest,(offset + 1 : U32),Each.PiecePair{left,right}),
        products(LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),rest,(offset + 1 : U32),left),LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),rest,(offset + 1 : U32),right)),product_cells(rest,(offset + 1 : U32),left,right))

# The products of the values of two readers are the products of their values.
def product_values(~Left: Data,~left_value: Left -> U32 -> Tile.Tile8,~Right: Data,~right_value: Right -> U32 -> Tile.Tile8,+left_context: Left,+right_context: Right,count: Nat,+start: Nat) ->
  {ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},count,start) ==
    products(ReduceValues.values_from(~Tile.Tile8,~Left,~left_value,left_context,count,start),ReduceValues.values_from(~Tile.Tile8,~Right,~right_value,right_context,count,start)) : List<&2,Tile.Tile8>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,tail => CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value,CombineLeaf.Context{left_context,right_context},U32.from_nat(start)) <> tail,
        ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},rest,1n+start),
        products(ReduceValues.values_from(~Tile.Tile8,~Left,~left_value,left_context,rest,1n+start),ReduceValues.values_from(~Tile.Tile8,~Right,~right_value,right_context,rest,1n+start)),
        product_values(~Left,~left_value,~Right,~right_value,left_context,right_context,rest,1n+start))

# The pairs of pieces of two trees of one layout that cover rows cells from
# first and hold the values of two readers stream the products of those values.
def product_stream(~Left: Data,~left_value: Left -> U32 -> Tile.Tile8,~Right: Data,~right_value: Right -> U32 -> Tile.Tile8,+left_context: Left,+right_context: Right,left: PiecesModel.Model<Tile.Tile8>,right: PiecesModel.Model<Tile.Tile8>,+first: Nat,+rows: Nat,
  +common: Each.Common(~Tile.Tile8,left,right),+coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,left)),first,rows),
  +left_values: Content.Values(~Tile.Tile8,~Left,~left_value,left_context,PiecesModel.content(~Tile.Tile8,left)),
  +right_values: Content.Values(~Tile.Tile8,~Right,~right_value,right_context,PiecesModel.content(~Tile.Tile8,right))) ->
  {LineChainValues.stream(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~product_cell,ReduceValues.shaped_tree(~Each.PiecePair<Tile.Tile8>,Each.paired_pieces(~Tile.Tile8,left,right))) ==
    ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},rows,first) : List<&2,Tile.Tile8>}:
  match left right:
    case PiecesModel.Leaf{Buffer.Model{+left_tree,+left_length,+left_certificate},+piece_first,+piece_rows} PiecesModel.Leaf{Buffer.Model{+right_tree,+right_length,+right_certificate},+other_first,+other_rows}:
      KernelGemm.Both{+same_first,+same_rows} = common
      PiecesPlan.LeafCoverage{+first_equal,+rows_equal} = coverage
      +left_model = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Tile.Tile8>}
      +right_model = {Buffer.Model{right_tree,right_length,right_certificate} : Buffer.Model<Tile.Tile8>}
      +left_cells = ReduceValues.leaf_stream(~Tile.Tile8,~Left,~left_value,left_context,piece_first,piece_rows,left_model,left_values)
      +right_cells = {%Equal.sym(U32,piece_first,other_first,same_first) :
          {LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,right_model) == ReduceValues.values_from(~Tile.Tile8,~Right,~right_value,right_context,U32.to_nat(piece_rows),U32.to_nat(_)) : List<&2,Tile.Tile8>}
        {%Equal.sym(U32,piece_rows,other_rows,same_rows) :
            {LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(_),0,right_model) == ReduceValues.values_from(~Tile.Tile8,~Right,~right_value,right_context,U32.to_nat(_),U32.to_nat(other_first)) : List<&2,Tile.Tile8>}
          ReduceValues.leaf_stream(~Tile.Tile8,~Right,~right_value,right_context,other_first,other_rows,right_model,right_values) :
          {LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,right_model) == ReduceValues.values_from(~Tile.Tile8,~Right,~right_value,right_context,U32.to_nat(piece_rows),U32.to_nat(other_first)) : List<&2,Tile.Tile8>}} :
        {LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,right_model) == ReduceValues.values_from(~Tile.Tile8,~Right,~right_value,right_context,U32.to_nat(piece_rows),U32.to_nat(piece_first)) : List<&2,Tile.Tile8>}}
      +left_list = ReduceValues.values_from(~Tile.Tile8,~Left,~left_value,left_context,U32.to_nat(piece_rows),U32.to_nat(piece_first))
      +right_list = ReduceValues.values_from(~Tile.Tile8,~Right,~right_value,right_context,U32.to_nat(piece_rows),U32.to_nat(piece_first))
      %rows_equal :
        {LineChainValues.cells(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~product_cell,U32.to_nat(piece_rows),0,Each.PiecePair{left_model,right_model}) == ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},_,first) : List<&2,Tile.Tile8>}
      %first_equal :
        {LineChainValues.cells(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~product_cell,U32.to_nat(piece_rows),0,Each.PiecePair{left_model,right_model}) ==
          ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},U32.to_nat(piece_rows),_) : List<&2,Tile.Tile8>}
      Equal.trans(List<&2,Tile.Tile8>,LineChainValues.cells(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~product_cell,U32.to_nat(piece_rows),0,Each.PiecePair{left_model,right_model}),
        products(LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,left_model),LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,right_model)),
        ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},U32.to_nat(piece_rows),U32.to_nat(piece_first)),
        product_cells(U32.to_nat(piece_rows),0,left_model,right_model),
        Equal.trans(List<&2,Tile.Tile8>,products(LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,left_model),LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,right_model)),products(left_list,right_list),
          ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},U32.to_nat(piece_rows),U32.to_nat(piece_first)),
          Equal.trans(List<&2,Tile.Tile8>,products(LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,left_model),LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,right_model)),
            products(left_list,LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,right_model)),products(left_list,right_list),
            Equal.cong(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,cells => products(cells,LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,right_model)),LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,left_model),left_list,left_cells),
            Equal.cong(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,cells => products(left_list,cells),LineChainValues.cells(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~ReduceValues.cell(~Tile.Tile8),U32.to_nat(piece_rows),0,right_model),right_list,right_cells)),
          Equal.sym(List<&2,Tile.Tile8>,ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},U32.to_nat(piece_rows),U32.to_nat(piece_first)),products(left_list,right_list),
            product_values(~Left,~left_value,~Right,~right_value,left_context,right_context,U32.to_nat(piece_rows),U32.to_nat(piece_first)))))
    case PiecesModel.Fork{+left_first,+left_second} PiecesModel.Fork{+right_first,+right_second}:
      KernelGemm.Both{+common_first,+common_second} = common
      PiecesPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
      KernelGemm.Both{+left_values_first,+left_values_second} = left_values
      KernelGemm.Both{+right_values_first,+right_values_second} = right_values
      +low = LineChainValues.stream(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~product_cell,ReduceValues.shaped_tree(~Each.PiecePair<Tile.Tile8>,Each.paired_pieces(~Tile.Tile8,left_first,right_first)))
      +high = LineChainValues.stream(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~product_cell,ReduceValues.shaped_tree(~Each.PiecePair<Tile.Tile8>,Each.paired_pieces(~Tile.Tile8,left_second,right_second)))
      +low_values = ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},left_rows,first)
      %Equal.sym(Nat,rows,Nat.add(left_rows,right_rows),rows_equal) :
        {List.append(&2,Tile.Tile8,low,high) == ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},_,first) : List<&2,Tile.Tile8>}
      Equal.trans(List<&2,Tile.Tile8>,List.append(&2,Tile.Tile8,low,high),List.append(&2,Tile.Tile8,low_values,high),ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},Nat.add(left_rows,right_rows),first),
        Equal.cong(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,cells => List.append(&2,Tile.Tile8,cells,high),low,low_values,
          product_stream(~Left,~left_value,~Right,~right_value,left_context,right_context,left_first,right_first,first,left_rows,common_first,left_coverage,left_values_first,right_values_first)),
        Equal.trans(List<&2,Tile.Tile8>,List.append(&2,Tile.Tile8,low_values,high),List.append(&2,Tile.Tile8,low_values,ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},right_rows,Nat.add(first,left_rows))),
          ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},Nat.add(left_rows,right_rows),first),
          Equal.cong(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,cells => List.append(&2,Tile.Tile8,low_values,cells),high,ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},right_rows,Nat.add(first,left_rows)),
            product_stream(~Left,~left_value,~Right,~right_value,left_context,right_context,left_second,right_second,Nat.add(first,left_rows),right_rows,common_second,right_coverage,
              left_values_second,right_values_second)),
          ReduceValues.values_append(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},left_rows,right_rows,first)))
    case PiecesModel.Leaf{+left_buffer,+piece_first,+piece_rows} PiecesModel.Fork{+right_first,+right_second}:
      Empty.absurd({LineChainValues.stream(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~product_cell,ReduceValues.shaped_tree(~Each.PiecePair<Tile.Tile8>,Each.paired_pieces(~Tile.Tile8,PiecesModel.Leaf{left_buffer,piece_first,piece_rows},
          PiecesModel.Fork{right_first,right_second}))) == ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},rows,first) : List<&2,Tile.Tile8>},common)
    case PiecesModel.Fork{+left_first,+left_second} PiecesModel.Leaf{+right_buffer,+other_first,+other_rows}:
      Empty.absurd({LineChainValues.stream(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~product_cell,ReduceValues.shaped_tree(~Each.PiecePair<Tile.Tile8>,Each.paired_pieces(~Tile.Tile8,PiecesModel.Fork{left_first,left_second},
          PiecesModel.Leaf{right_buffer,other_first,other_rows}))) == ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},rows,first) : List<&2,Tile.Tile8>},common)

# Two models of one layout laid out as layout that cover total cells and hold
# the values of two readers: their zipped pieces stream the products of values.
def product_streamed(~Left: Data,~left_value: Left -> U32 -> Tile.Tile8,~Right: Data,~right_value: Right -> U32 -> Tile.Tile8,+left_context: Left,+right_context: Right,+left: PiecesModel.Model<Tile.Tile8>,+right: PiecesModel.Model<Tile.Tile8>,+layout: Pieces.Ranges,+total: Nat,
  +left_canonical: {PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,left)) == layout : Pieces.Ranges},
  +right_canonical: {PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,right)) == layout : Pieces.Ranges},
  +coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,left)),0n,total),
  +left_values: Content.Values(~Tile.Tile8,~Left,~left_value,left_context,PiecesModel.content(~Tile.Tile8,left)),
  +right_values: Content.Values(~Tile.Tile8,~Right,~right_value,right_context,PiecesModel.content(~Tile.Tile8,right))) ->
  ReduceValues.Streamed<BNCells.CellPair(),Each.PiecePair<Tile.Tile8>,Tile.Tile8,Each.pair_pack(~Tile.Tile8),product_cell,
    Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,left),PiecesModel.pack(~Tile.Tile8,right)),layout,ReduceValues.values_from(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~Tile.times,~left_value,~right_value),CombineLeaf.Context{left_context,right_context},total,0n)>:
  +left_layout = Equal.trans(Pieces.Ranges,PiecesModel.layout(~Tile.Tile8,left),PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,left)),layout,
    PiecesModel.layout_matches_content(~Tile.Tile8,left),left_canonical)
  +right_layout = Equal.trans(Pieces.Ranges,PiecesModel.layout(~Tile.Tile8,right),PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,right)),layout,
    PiecesModel.layout_matches_content(~Tile.Tile8,right),right_canonical)
  +common = Each.matched(~Tile.Tile8,left,right,Equal.trans(Pieces.Ranges,PiecesModel.layout(~Tile.Tile8,left),layout,PiecesModel.layout(~Tile.Tile8,right),left_layout,
    Equal.sym(Pieces.Ranges,PiecesModel.layout(~Tile.Tile8,right),layout,right_layout)))
  +paired = Each.paired_pieces(~Tile.Tile8,left,right)
  ReduceValues.Streamed{ReduceValues.shaped_tree(~Each.PiecePair<Tile.Tile8>,paired),
    Equal.trans(Partition.Tree<&1,BNCells.CellPair(),Unit>,Regrouped.tree_of(~BNCells.CellPair(),~Each.PiecePair<Tile.Tile8>,~Each.pair_pack(~Tile.Tile8),ReduceValues.shaped_tree(~Each.PiecePair<Tile.Tile8>,paired)),
      PiecesModel.shaped_pack(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),paired),
      Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,left),PiecesModel.pack(~Tile.Tile8,right)),
      ReduceValues.shaped_tree_packed(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),paired),
      Equal.sym(Partition.Tree<&1,BNCells.CellPair(),Unit>,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,left),PiecesModel.pack(~Tile.Tile8,right)),
        PiecesModel.shaped_pack(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),paired),Each.paired_zip(~Tile.Tile8,left,right,common))),
    Equal.trans(Pieces.Ranges,Regrouped.tree_layout(~Each.PiecePair<Tile.Tile8>,ReduceValues.shaped_tree(~Each.PiecePair<Tile.Tile8>,paired)),PiecesModel.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),paired),layout,
      ReduceValues.shaped_tree_layout(~Each.PiecePair<Tile.Tile8>,paired),
      Equal.trans(Pieces.Ranges,PiecesModel.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),paired),PiecesModel.layout(~Tile.Tile8,left),layout,
        Each.paired_plan(~Tile.Tile8,left,right,common),left_layout)),
    product_stream(~Left,~left_value,~Right,~right_value,left_context,right_context,left,right,0n,total,common,coverage,left_values,right_values)}
