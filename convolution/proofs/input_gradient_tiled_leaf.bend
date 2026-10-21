# Actual leaf values from the leaf's write list: the requested point holds
# its specified value, an address outside the leaf region keeps the output,
# and the gradient source is kept.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_readback.bend as StorageReadback
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/write_program_every.bend as Every
import ../../lib/proofs/write_program_readback.bend as Readback
import ../../lib/proofs/input_gradient_tiled_producer.bend as Producer
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_output_layout.bend as Output
import ../../lib/proofs/input_gradient_tiled_output_coordinates.bend as Coordinates
import ../../lib/proofs/input_gradient_tiled_output_outside.bend as Outside
import ../../lib/proofs/input_gradient_tiled_sweep_selected.bend as Selected
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_leaf_writes.bend as LeafWrites
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_written.bend as Phases
import ./input_gradient_tiled_leaf_written.bend as LeafWritten
import ./input_gradient_tiled_empty.bend as EmptyValue
import ../../lib/proofs/input_gradient_tiled_coverage.bend as Coverage
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ../input_gradient_specification.bend as Specification

# A leaf, the point asked for, and what the output buffer holds before the
# leaf runs.
type Context is Data:
  Context{problem: Producer.Problem,image: U32,gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>,output: Buffer.Model<F32>,
    source: Phases.ChunkSource(Producer.query(problem),Producer.leaf_contained(problem),image,gradient),
    fits: Phases.Fits(Producer.query(problem)),
    layout: Output.Layout<Producer.query(problem)>,
    capacity: {Tree.capacity(F32,Buffer.tree(~F32,output)) == Output.capacity(Producer.query(problem),layout) : U32},
    depth: {Nat.is_le(Buffer.depth(~F32,output),30n) == True{} : Bool},
    initial_zero: {Tree.read(F32,Buffer.tree(~F32,output),Coverage.address(problem)) == (0.0 + 0.0 : F32) : F32}}
def problem(context: Context) -> Producer.Problem:
  Context{problem,image,gradient,weights,output,source,fits,layout,capacity,depth,initial_zero} = context
  problem
def image(context: Context) -> U32:
  Context{problem,image,gradient,weights,output,source,fits,layout,capacity,depth,initial_zero} = context
  image
def gradient(context: Context) -> Buffer.Model<F32>:
  Context{problem,image,gradient,weights,output,source,fits,layout,capacity,depth,initial_zero} = context
  gradient
def weights(context: Context) -> Buffer.Model<F32>:
  Context{problem,image,gradient,weights,output,source,fits,layout,capacity,depth,initial_zero} = context
  weights
def output(context: Context) -> Buffer.Model<F32>:
  Context{problem,image,gradient,weights,output,source,fits,layout,capacity,depth,initial_zero} = context
  output
def source(context: Context) -> Phases.ChunkSource(Producer.query(problem(context)),Producer.leaf_contained(problem(context)),image(context),gradient(context)):
  Context{problem,image,gradient,weights,output,source,fits,layout,capacity,depth,initial_zero} = context
  source
def fits(context: Context) -> Phases.Fits(Producer.query(problem(context))):
  Context{problem,image,gradient,weights,output,source,fits,layout,capacity,depth,initial_zero} = context
  fits
def layout(context: Context) -> Output.Layout<Producer.query(problem(context))>:
  Context{problem,image,gradient,weights,output,source,fits,layout,capacity,depth,initial_zero} = context
  layout
def capacity(context: Context) -> {Tree.capacity(F32,Buffer.tree(~F32,output(context))) == Output.capacity(Producer.query(problem(context)),layout(context)) : U32}:
  Context{problem,image,gradient,weights,output,source,fits,layout,capacity,depth,initial_zero} = context
  capacity
def depth(context: Context) -> {Nat.is_le(Buffer.depth(~F32,output(context)),30n) == True{} : Bool}:
  Context{problem,image,gradient,weights,output,source,fits,layout,capacity,depth,initial_zero} = context
  depth
def initial_zero(context: Context) -> {Tree.read(F32,Buffer.tree(~F32,output(context)),Coverage.address(problem(context))) == (0.0 + 0.0 : F32) : F32}:
  Context{problem,image,gradient,weights,output,source,fits,layout,capacity,depth,initial_zero} = context
  initial_zero

def query(+context: Context) -> Chunks.Query:
  Producer.query(problem(context))
def leaf(+context: Context) -> LeafWritten.Leaf:
  LeafWritten.Leaf{query(context),Producer.leaf_contained(problem(context)),fits(context),image(context),gradient(context),weights(context),source(context)}
def target(+context: Context) -> Phases.Target:
  LeafWritten.target(leaf(context))
def packed(+context: Context) -> Selected.Packed:
  +state = LeafWritten.packed(leaf(context))
  Selected.Packed{Packed.weight_tree(state),Packed.offset_tree(state)}
def actual(+context: Context) -> Kernel.Part:
  Selected.native_run(query(context),packed(context),output(context))
def address(+context: Context) -> U32:
  Coverage.address(problem(context))
def wanted(+context: Context) -> F32:
  +point = Producer.point(problem(context))
  Specification.value(Phases.target_problem(target(context)),Domain.channel(point),Domain.row(point),Domain.column(point))
def writes(+context: Context) -> List<&2,Writes.Assignment<F32>>:
  +leaf = leaf(context)
  LeafWrites.leaf_writes(LeafWritten.native(leaf),Chunks.source(query(context)),Packed.weight_tree(LeafWritten.packed(leaf)),Packed.offset_tree(LeafWritten.packed(leaf)))
def initial(+context: Context) -> Tree.Tree<F32>:
  Buffer.tree(~F32,output(context))
def tree_capacity(+context: Context) -> U32:
  Tree.capacity(F32,initial(context))
def region(+context: Context) -> Chunks.Query:
  Output.region(query(context))
def region_layout(+context: Context) -> Output.Layout<region(context)>:
  Output.region_layout(query(context),layout(context))

# The actual output is the initial output after the leaf's writes.
def run_tree(+part: Execution.Part) -> {Buffer.tree(~F32,Selected.part_output(part)) == LeafWrites.part_tree(part) : Tree.Tree<F32>}:
  Execution.Part{gradient,output} = part
  {==}
def actual_read(+context: Context,+index: U32) ->
  {Selected.native_value(actual(context),index) == Tree.read(F32,Writes.pure(F32,writes(context),initial(context)),index) : F32}:
  +q = query(context)
  +packed = packed(context)
  +output = output(context)
  +run = Selected.run(q,packed,output)
  +state = LeafWritten.packed(leaf(context))
  Equal.trans(F32,Selected.native_value(actual(context),index),Selected.native_value(Execution.represent_part(run),index),Tree.read(F32,Writes.pure(F32,writes(context),initial(context)),index),
    Equal.cong(Kernel.Part,F32,part => Selected.native_value(part,index),actual(context),Execution.represent_part(run),Selected.run_matches(q,packed,output)),
    Equal.trans(F32,Selected.native_value(Execution.represent_part(run),index),Tree.read(F32,Buffer.tree(~F32,Selected.part_output(run)),index),Tree.read(F32,Writes.pure(F32,writes(context),initial(context)),index),
      Selected.represented_value(run,index),
      Equal.cong(Tree.Tree<F32>,F32,tree => Tree.read(F32,tree,index),Buffer.tree(~F32,Selected.part_output(run)),Writes.pure(F32,writes(context),initial(context)),
        Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,Selected.part_output(run)),LeafWrites.part_tree(run),Writes.pure(F32,writes(context),initial(context)),run_tree(run),
          LeafWrites.leaf_tree_writes(Selected.native_context(q),Chunks.source(q),Packed.weight_tree(state),Packed.offset_tree(state),output)))))

# Every write lands inside the output buffer.
def native_inside(+query: Chunks.Query,+layout: Output.Layout<query>,+point: Output.Point<query>) ->
  {U32.is_lt(Output.native_address(query,point),Output.capacity(query,layout)) == True{} : Bool}:
  +native = Output.native_address(query,point)
  Equal.trans(Bool,U32.is_lt(native,Output.capacity(query,layout)),Nat.is_lt(U32.to_nat(native),U32.to_nat(Output.capacity(query,layout))),True{},
    Comparison.lt(native,Output.capacity(query,layout)),
    %Equal.sym(Nat,U32.to_nat(native),Output.address(query,point),Output.native_address_exact(query,layout,point)) :
      {Nat.is_lt(_,U32.to_nat(Output.capacity(query,layout))) == True{} : Bool}
    Output.address_inside(query,layout,point))
def point_inside(+context: Context,+point: Output.Point<region(context)>) ->
  {U32.is_lt(Output.native_address(region(context),point),tree_capacity(context)) == True{} : Bool}:
  %Equal.sym(U32,tree_capacity(context),Output.capacity(query(context),layout(context)),capacity(context)) :
    {U32.is_lt(Output.native_address(region(context),point),_) == True{} : Bool}
  native_inside(region(context),region_layout(context),point)
def written_inside(+context: Context,+address: U32,+value: F32,holds: Phases.Written(target(context),address,value)) ->
  {U32.is_lt(address,tree_capacity(context)) == True{} : Bool}:
  Phases.Witness{+point,+at,correct} = holds
  %at : {U32.is_lt(_,tree_capacity(context)) == True{} : Bool}
  point_inside(context,point)
def writes_inside(+context: Context) -> Readback.Inside(~F32,tree_capacity(context),writes(context)):
  Every.inside(~F32,~Phases.Target,~Phases.Written,~Context,~target,~tree_capacity,~written_inside,context,writes(context),LeafWritten.leaf_written(leaf(context)))

# A write at the requested address stores the requested point's value.
def selected_matches(+context: Context,+destination: U32,+value: F32,equal: Bool,+compared: {U32.is_eq(destination,address(context)) == equal : Bool},
  holds: Phases.Written(target(context),destination,value)) -> Boolean.When(equal,{value == wanted(context) : F32}):
  match equal:
    case False{}: Unit{}
    case True{}:
      Phases.Witness{+point,+at,+correct} = holds
      +area = region(context)
      +arrangement = region_layout(context)
      +wanted_point = Coverage.requested(problem(context))
      +same = Equal.trans(U32,Output.native_address(area,point),destination,Output.native_address(area,wanted_point),at,Arithmetic.equal_of_is_eq(destination,address(context),compared))
      +specified = Phases.target_problem(target(context))
      %Coordinates.channel_equal(area,arrangement,point,wanted_point,same) :
        {value == Specification.value(specified,_,Output.row(area,wanted_point),Output.column(area,wanted_point)) : F32}
      %Coordinates.row_equal(area,arrangement,point,wanted_point,same) :
        {value == Specification.value(specified,Output.channel(area,point),_,Output.column(area,wanted_point)) : F32}
      %Coordinates.column_equal(area,arrangement,point,wanted_point,same) :
        {value == Specification.value(specified,Output.channel(area,point),Output.row(area,point),_) : F32}
      correct
def written_matches(+context: Context,+destination: U32,+value: F32,holds: Phases.Written(target(context),destination,value)) ->
  Boolean.When(U32.is_eq(destination,address(context)),{value == wanted(context) : F32}):
  selected_matches(context,destination,value,U32.is_eq(destination,address(context)),{==},holds)
def writes_matching(+context: Context) -> Readback.Matching(~F32,address(context),wanted(context),writes(context)):
  Every.matching(~F32,~Phases.Target,~Phases.Written,~Context,~target,~address,~wanted,~written_matches,context,writes(context),LeafWritten.leaf_written(leaf(context)))

# The value at the requested address: the wanted value if a write is there,
# the initial value otherwise.
def read_requested(+context: Context) ->
  {Selected.native_value(actual(context),address(context)) == Bool.pick(F32,Readback.contains(~F32,address(context),writes(context)),wanted(context),Tree.read(F32,initial(context),address(context))) : F32}:
  +output = output(context)
  Equal.trans(F32,Selected.native_value(actual(context),address(context)),Tree.read(F32,Writes.pure(F32,writes(context),initial(context)),address(context)),
    Bool.pick(F32,Readback.contains(~F32,address(context),writes(context)),wanted(context),Tree.read(F32,initial(context),address(context))),
    actual_read(context,address(context)),
    Equal.trans(F32,Tree.read(F32,Writes.pure(F32,writes(context),initial(context)),address(context)),
      Readback.selected(~F32,writes(context),address(context),Tree.read(F32,initial(context),address(context))),
      Bool.pick(F32,Readback.contains(~F32,address(context),writes(context)),wanted(context),Tree.read(F32,initial(context),address(context))),
      Readback.program(~F32,writes(context),Buffer.depth(~F32,output),initial(context),address(context),Buffer.balanced(~F32,output),depth(context),
        point_inside(context,Coverage.requested(problem(context))),writes_inside(context)),
      Readback.selected_matching(~F32,writes(context),address(context),wanted(context),Tree.read(F32,initial(context),address(context)),writes_matching(context))))

# A point of an empty phase is never written: its wanted value is the
# initial +0.0 + 0.0.
def empty_value(+context: Context,+wanted_zero: {wanted(context) == (0.0 + 0.0 : F32) : F32}) ->
  {Selected.native_value(actual(context),address(context)) == wanted(context) : F32}:
  Equal.trans(F32,Selected.native_value(actual(context),address(context)),
    Bool.pick(F32,Readback.contains(~F32,address(context),writes(context)),wanted(context),Tree.read(F32,initial(context),address(context))),wanted(context),
    read_requested(context),
    StorageReadback.pick_same(~F32,Readback.contains(~F32,address(context),writes(context)),wanted(context),Tree.read(F32,initial(context),address(context)),wanted(context),
      {==},Equal.trans(F32,Tree.read(F32,initial(context),address(context)),(0.0 + 0.0 : F32),wanted(context),initial_zero(context),Equal.sym(F32,wanted(context),(0.0 + 0.0 : F32),wanted_zero))))

# A point of an enabled phase is written, so it holds its wanted value.
def active_value(+context: Context,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem(context)))),0) == False{} : Bool}) ->
  {Selected.native_value(actual(context),address(context)) == wanted(context) : F32}:
  +state = LeafWritten.packed(leaf(context))
  +active = {Coverage.Active{problem(context),enabled,fits(context)} : Coverage.Active}
  +covered = Coverage.leaf_covered(active,Chunks.source(query(context)),Packed.weight_tree(state),Packed.offset_tree(state))
  +before = Tree.read(F32,initial(context),address(context))
  Equal.trans(F32,Selected.native_value(actual(context),address(context)),Bool.pick(F32,Readback.contains(~F32,address(context),writes(context)),wanted(context),before),wanted(context),
    read_requested(context),
    %Equal.sym(Bool,Readback.contains(~F32,address(context),writes(context)),True{},covered) : {Bool.pick(F32,_,wanted(context),before) == wanted(context) : F32}
    {==})

# The requested point holds its wanted value.
def selected_value(disabled: Bool,+context: Context,+compared: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem(context)))),0) == disabled : Bool}) ->
  {Selected.native_value(actual(context),address(context)) == wanted(context) : F32}:
  match disabled:
    case True{}: empty_value(context,EmptyValue.value(problem(context),image(context),gradient(context),weights(context),compared))
    case False{}: active_value(context,compared)
def value(+context: Context) -> {Selected.native_value(actual(context),address(context)) == wanted(context) : F32}:
  selected_value(U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem(context)))),0),context,{==})

# An address outside the leaf region keeps its initial value.
type Probe is Data:
  Probe{context: Context,index: U32,outside: Outside.Outside<Producer.query(problem(context)),index>}
def probe_context(probe: Probe) -> Context:
  Probe{context,index,outside} = probe
  context
def probe_index(+probe: Probe) -> U32:
  Probe{context,index,outside} = probe
  index
def probe_outside(probe: Probe) -> Outside.Outside<Producer.query(problem(probe_context(probe))),probe_index(probe)>:
  Probe{context,index,outside} = probe
  outside
def probe_target(+probe: Probe) -> Phases.Target:
  target(probe_context(probe))
def written_apart(+probe: Probe,+address: U32,+value: F32,holds: Phases.Written(probe_target(probe),address,value)) ->
  {U32.is_eq(address,probe_index(probe)) == False{} : Bool}:
  Phases.Witness{+point,+at,correct} = holds
  +context = probe_context(probe)
  %at : {U32.is_eq(_,probe_index(probe)) == False{} : Bool}
  Outside.disjoint(region(context),region_layout(context),point,probe_index(probe),Outside.region_outside(query(context),probe_index(probe),probe_outside(probe)))
def outside(+context: Context,+index: U32,+excluded: Outside.Outside<query(context),index>,+inside: {U32.is_lt(index,Output.capacity(query(context),layout(context))) == True{} : Bool}) ->
  {Selected.native_value(actual(context),index) == Tree.read(F32,initial(context),index) : F32}:
  +output = output(context)
  +probe = {Probe{context,index,excluded} : Probe}
  Equal.trans(F32,Selected.native_value(actual(context),index),Tree.read(F32,Writes.pure(F32,writes(context),initial(context)),index),Tree.read(F32,initial(context),index),
    actual_read(context,index),
    Readback.excluded_readback(writes(context),initial(context),Buffer.depth(~F32,output),index,Buffer.balanced(~F32,output),depth(context),writes_inside(context),
      %Equal.sym(U32,tree_capacity(context),Output.capacity(query(context),layout(context)),capacity(context)) : {U32.is_lt(index,_) == True{} : Bool} inside,
      Every.excludes(~Phases.Target,~Phases.Written,~Probe,~probe_target,~probe_index,~written_apart,probe,writes(context),LeafWritten.leaf_written(leaf(context)))))

