# The input-gradient kernel from the packing to the last image: the outcome
# of Kernel.run gives the weights back and its bands are the bands of the
# batch.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor_band.bend as Bands
import ../../lib/traversal_array.bend as Traversal
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_tiled_geometry.bend as Geometry
import ../../lib/proofs/input_gradient_tiled_packing_bounds.bend as PackingBounds
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_tiled_weights.bend as WeightPacking
import ../../lib/proofs/traversal_array_refinement.bend as Arrays
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_each.bend as Each
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ./input_gradient_tiled_leaves.bend as Leaves
import ./input_gradient_tiled_images.bend as Images

# A call of the kernel: the setup, the fit, the worker count and the
# machine limits the batch fits.
type Run is Data:
  Run{setup: Images.Setup,envelope: Geometry.WordEnvelope,
    fits: {Kernel.fits(Images.setup_geometry(setup),Machine.capacity(Domain.limits(Images.setup_call(setup)))) == True{} : Bool},
    workers: U32,limits: Machine.Limits,
    machine_room: {Nat.is_le(Nat.mul(Images.setup_images(setup),Images.size(setup)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
def run_setup(+run: Run) -> Images.Setup:
  Run{setup,envelope,fits,workers,limits,machine_room} = run
  setup
def run_workers(+run: Run) -> U32:
  Run{setup,envelope,fits,workers,limits,machine_room} = run
  workers
def run_images(+run: Run) -> Nat:
  Images.setup_images(run_setup(run))
def run_count(+run: Run) -> Nat:
  Nat.mul(Images.setup_images(run_setup(run)),Images.size(run_setup(run)))

# The split depth the kernel chooses, abstract in the planner's choice.
def split_depth(+geometry: Kernel.Geometry,+workers: U32) -> Nat:
  Kernel.select(geometry,Kernel.tiling(geometry),workers,
    Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.weight_total,Kernel.window_jobs(geometry),0,geometry,0),
    Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.inner_total,Kernel.window_jobs(geometry),0,geometry,0))

# The packed weights and the table, as the packing model builds them.
def packing_query(+setup: Images.Setup) -> Packed.Query:
  Packed.allocated_query(PackingBounds.Query{False{},Images.setup_call(setup)},Kernel.tiling(Images.setup_geometry(setup)),Images.setup_weights(setup),0n,0n,0n)
def packed(+setup: Images.Setup) -> WeightPacking.Packed:
  Packed.complete(packing_query(setup),Packed.initial(packing_query(setup)))
def weights_same(+setup: Images.Setup) -> {Tree.pack(Tile.Tile8,Packed.weight_tree(packed(setup))) == Leaves.native_weights(Images.packing(setup)) : Array<Tile.Tile8>}:
  Images.Setup{+call,+gradient,+weights} = setup
  +state = packed(Images.Setup{call,gradient,weights})
  +matches = Packed.native_complete_matches(PackingBounds.Query{False{},call},Kernel.tiling(Domain.geometry(call)),weights,0n,0n,0n)
  Equal.sym(Array<Tile.Tile8>,Leaves.native_weights(Images.packing(Images.Setup{call,gradient,weights})),Tree.pack(Tile.Tile8,Packed.weight_tree(state)),
    Equal.trans(Array<Tile.Tile8>,Leaves.native_weights(Images.packing(Images.Setup{call,gradient,weights})),Leaves.native_weights(WeightPacking.represent(state)),Tree.pack(Tile.Tile8,Packed.weight_tree(state)),
      Equal.cong(Kernel.Packed,Array<Tile.Tile8>,native => Leaves.native_weights(native),Images.packing(Images.Setup{call,gradient,weights}),WeightPacking.represent(state),matches),
      Leaves.represent_weights(state)))
def offsets_same(+setup: Images.Setup) -> {Tree.pack(U32,Packed.offset_tree(packed(setup))) == Leaves.native_offsets(Images.packing(setup)) : Array<U32>}:
  Images.Setup{+call,+gradient,+weights} = setup
  +state = packed(Images.Setup{call,gradient,weights})
  +matches = Packed.native_complete_matches(PackingBounds.Query{False{},call},Kernel.tiling(Domain.geometry(call)),weights,0n,0n,0n)
  Equal.sym(Array<U32>,Leaves.native_offsets(Images.packing(Images.Setup{call,gradient,weights})),Tree.pack(U32,Packed.offset_tree(state)),
    Equal.trans(Array<U32>,Leaves.native_offsets(Images.packing(Images.Setup{call,gradient,weights})),Leaves.native_offsets(WeightPacking.represent(state)),Tree.pack(U32,Packed.offset_tree(state)),
      Equal.cong(Kernel.Packed,Array<U32>,native => Leaves.native_offsets(native),Images.packing(Images.Setup{call,gradient,weights}),WeightPacking.represent(state),matches),
      Leaves.represent_offsets(state)))

# The batch of the call.
def run_batch(+run: Run) -> Images.Batch:
  Run{+setup,+envelope,+fits,+workers,+limits,+machine_room} = run
  Images.Batch{setup,envelope,fits,split_depth(Images.setup_geometry(setup),workers),Machine.capacity(limits),Limits.capacity_fits(limits),machine_room,
    limits,machine_room,Packed.weight_tree(packed(setup)),Packed.offset_tree(packed(setup)),weights_same(setup),offsets_same(setup)}

# The packing keeps its source, and a packed state is its parts.
def keep_source(index: U32,context: Kernel.PackContext,source: Buffer.Model<F32>) -> Buffer.Model<F32>:
  source
def keep_fixed(+index: U32,+context: Kernel.PackContext,+source: Buffer.Model<F32>) -> {keep_source(index,context,source) == source : Buffer.Model<F32>}:
  {==}
def source_kept(+query: Packed.Query) -> {WeightPacking.source(Packed.complete(query,Packed.initial(query))) == Packed.source(query) : Buffer.Model<F32>}:
  Equal.trans(Buffer.Model<F32>,WeightPacking.source(Packed.complete(query,Packed.initial(query))),
    Traversal.indexed_fold(~Buffer.Model<F32>,~Kernel.PackContext,~keep_source,Packed.count(query),0,Packed.context(query),WeightPacking.source(Packed.initial(query))),
    Packed.source(query),
    Arrays.indexed_fold_observes(~WeightPacking.Packed,~Buffer.Model<F32>,~Kernel.PackContext,~WeightPacking.pack_phase,~WeightPacking.source,~keep_source,~WeightPacking.phase_keeps_source,
      Packed.count(query),0,Packed.context(query),Packed.initial(query)),
    Arrays.indexed_fold_fixed(~Buffer.Model<F32>,~Kernel.PackContext,~keep_source,~keep_fixed,Packed.count(query),0,Packed.context(query),WeightPacking.source(Packed.initial(query))))
def packed_parts(state: WeightPacking.Packed) ->
  {WeightPacking.represent(state) == Kernel.Packed{Buffer.pack(~F32,WeightPacking.source(state)),Tree.pack(Tile.Tile8,Packed.weight_tree(state)),Tree.pack(U32,Packed.offset_tree(state)),
    WeightPacking.weight_base(state),WeightPacking.offset_base(state)} : Kernel.Packed}:
  WeightPacking.Packed{source,weights,offsets,weight_base,offset_base} = state
  {==}

# What an outcome gives: the weights and the contents of its bands.
def outcome_weights(outcome: Kernel.Outcome) -> Storage.Buffer<F32>:
  Kernel.Outcome{weights,output} = outcome
  weights
def outcome_contents(outcome: Kernel.Outcome) -> Model.Content<F32>:
  Kernel.Outcome{weights,output} = outcome
  Model.contents(~F32,output)
def Outcome(-weights: Storage.Buffer<F32>,+contents: Model.Content<F32>,-outcome: Kernel.Outcome) -> Data:
  Shape.Both<{outcome_weights(outcome) == weights : Storage.Buffer<F32>},{outcome_contents(outcome) == contents : Model.Content<F32>}>
def finished_weights(-source: Storage.Buffer<F32>,made: Kernel.ImageState & Bands.Bands(~F32)) -> {outcome_weights(Kernel.finished(source,made)) == source : Storage.Buffer<F32>}:
  (state,bands) = made
  {==}
def finished_contents(-source: Storage.Buffer<F32>,made: Kernel.ImageState & Bands.Bands(~F32)) ->
  {outcome_contents(Kernel.finished(source,made)) == Pair.snd(Images.State,Model.Content<F32>,Each.seen(~F32,~Kernel.ImageState,~Images.State,~Images.see,made)) : Model.Content<F32>}:
  (state,bands) = made
  {==}

# The kernel runs the images of a batch of at least one image: the outcome
# gives the weights back and its bands are the bands of the batch.
def run_matches(+run: Run,+rest: Nat,+count: {run_images(run) == 1n+rest : Nat}) ->
  Outcome(Buffer.pack(~F32,Images.setup_weights(run_setup(run))),Model.planned(~F32,~Images.batch_part(run_batch(run)),Images.batch_plan(run_batch(run))),
    Kernel.run(Images.setup_geometry(run_setup(run)),run_count(run),run_workers(run),Buffer.pack(~F32,Images.setup_gradient(run_setup(run))),Buffer.pack(~F32,Images.setup_weights(run_setup(run))))):
  Run{Images.Setup{Domain.Call{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+batch,+limits_call,+evidence},Buffer.Model{+gradient_tree,+gradient_length,+gradient_certificate},Buffer.Model{+weights_tree,+weights_length,+weights_certificate}},+envelope,+fits,+workers,+limits,+machine_room} = run
  +gradient_model = {Buffer.Model{gradient_tree,gradient_length,gradient_certificate} : Buffer.Model<F32>}
  +weights_model = {Buffer.Model{weights_tree,weights_length,weights_certificate} : Buffer.Model<F32>}
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits_call,evidence} : Domain.Call}
  +reader = {Images.Setup{whole,gradient_model,weights_model} : Images.Setup}
  +frame = {Run{reader,envelope,fits,workers,limits,machine_room} : Run}
  +images = run_batch(frame)
  +query = packing_query(reader)
  +state = Packed.complete(query,Packed.initial(query))
  +built = Images.built(images)
  +count_n = Nat.mul(batch,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w))))
  +nonzero = Equal.cong(Nat,Bool,value => Nat.is_eq(value,0n),batch,1n+rest,count)
  +contents = Equal.trans(Model.Content<F32>,
    outcome_contents(Kernel.finished(Buffer.pack(~F32,weights_model),Bands.build(~Storage.Buffer<F32>,~Kernel.ImageState,~Kernel.Images,~Kernel.image,
      Images.batch_plan(images),Images.batch_context(images),Images.represent(Images.batch_state(images))))),
    Pair.snd(Images.State,Model.Content<F32>,Each.seen(~F32,~Kernel.ImageState,~Images.State,~Images.see,Bands.build(~Storage.Buffer<F32>,~Kernel.ImageState,~Kernel.Images,~Kernel.image,
      Images.batch_plan(images),Images.batch_context(images),Images.represent(Images.batch_state(images))))),
    Model.planned(~F32,~Images.batch_part(images),Images.batch_plan(images)),
    finished_contents(Buffer.pack(~F32,weights_model),Bands.build(~Storage.Buffer<F32>,~Kernel.ImageState,~Kernel.Images,~Kernel.image,
      Images.batch_plan(images),Images.batch_context(images),Images.represent(Images.batch_state(images)))),
    Equal.cong(Images.State & Model.Content<F32>,Model.Content<F32>,made => Pair.snd(Images.State,Model.Content<F32>,made),
      Each.seen(~F32,~Kernel.ImageState,~Images.State,~Images.see,Bands.build(~Storage.Buffer<F32>,~Kernel.ImageState,~Kernel.Images,~Kernel.image,
        Images.batch_plan(images),Images.batch_context(images),Images.represent(Images.batch_state(images)))),
      (Images.batch_state(images),Model.planned(~F32,~Images.batch_part(images),Images.batch_plan(images))),built))
  %Equal.sym(Nat,Nat.div(count_n,U32.to_nat((c * h * w : U32))),batch,Domain.run_images(whole,count_n,{==})) :
    Outcome(Buffer.pack(~F32,weights_model),Model.planned(~F32,~Images.batch_part(images),Images.batch_plan(images)),
      Kernel.run_selected(Nat.is_eq(_,0n),shape,_,workers,Buffer.pack(~F32,gradient_model),Buffer.pack(~F32,weights_model)))
  %Equal.sym(Bool,Nat.is_eq(batch,0n),False{},nonzero) :
    Outcome(Buffer.pack(~F32,weights_model),Model.planned(~F32,~Images.batch_part(images),Images.batch_plan(images)),
      Kernel.run_selected(_,shape,batch,workers,Buffer.pack(~F32,gradient_model),Buffer.pack(~F32,weights_model)))
  %Equal.sym(Kernel.Packed,Packed.native_complete(PackingBounds.Query{False{},whole},Kernel.tiling(shape),weights_model),WeightPacking.represent(state),
      Packed.native_complete_matches(PackingBounds.Query{False{},whole},Kernel.tiling(shape),weights_model,0n,0n,0n)) :
    Outcome(Buffer.pack(~F32,weights_model),Model.planned(~F32,~Images.batch_part(images),Images.batch_plan(images)),
      Kernel.packed(shape,Kernel.tiling(shape),batch,split_depth(shape,workers),Tree.pack(F32,gradient_tree),_))
  %Equal.sym(Kernel.Packed,WeightPacking.represent(state),Kernel.Packed{Buffer.pack(~F32,WeightPacking.source(state)),Tree.pack(Tile.Tile8,Packed.weight_tree(state)),
      Tree.pack(U32,Packed.offset_tree(state)),WeightPacking.weight_base(state),WeightPacking.offset_base(state)},packed_parts(state)) :
    Outcome(Buffer.pack(~F32,weights_model),Model.planned(~F32,~Images.batch_part(images),Images.batch_plan(images)),
      Kernel.packed(shape,Kernel.tiling(shape),batch,split_depth(shape,workers),Tree.pack(F32,gradient_tree),_))
  %Equal.sym(Buffer.Model<F32>,WeightPacking.source(state),weights_model,source_kept(query)) :
    Outcome(Buffer.pack(~F32,weights_model),Model.planned(~F32,~Images.batch_part(images),Images.batch_plan(images)),
      Kernel.finished(Buffer.pack(~F32,_),Bands.build(~Storage.Buffer<F32>,~Kernel.ImageState,~Kernel.Images,~Kernel.image,
        Images.batch_plan(images),Images.batch_context(images),Images.represent(Images.batch_state(images)))))
  Shape.Both{finished_weights(Buffer.pack(~F32,weights_model),Bands.build(~Storage.Buffer<F32>,~Kernel.ImageState,~Kernel.Images,~Kernel.image,
      Images.batch_plan(images),Images.batch_context(images),Images.represent(Images.batch_state(images)))),contents}
