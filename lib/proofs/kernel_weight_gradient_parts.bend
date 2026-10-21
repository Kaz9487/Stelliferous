# The weight gradient in parts of the outputs. Every part owns a copy of the
# input and the rows of the output gradient of its outputs; the parts run the
# serial kernel and write their rows into one result buffer.
import Base
import ../kernel_weight_gradient.bend as Kernel
import ../kernel_parameter_gradient.bend as Parameter
import ../storage_buffer.bend as Storage
import ../storage_clone.bend as Clone
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../traversal_partition.bend as Partition
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_clone_refinement.bend as CloneLaws
import ./tensor_band_model.bend as Model
import ./tensor_band_each.bend as Each
import ./tensor_band_from_whole.bend as FromWhole
import ./tensor_band_plan.bend as BandPlan
import ./tensor_list_values.bend as Listed
import ./kernel_weight_gradient_group.bend as Groups
import ./kernel_weight_gradient_refinement.bend as WeightModel
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_multiplication.bend as Multiplication
import ./nat_order.bend as Order
import ./storage_shape.bend as Shape

# ---- The owners of a part ----

# What is seen of the owners of a part: the contents of its input and of its
# output gradient.
type OwnersSeen is Data:
  OwnersSeen{input: Buffer.Content<F32>,gradient: Buffer.Content<F32>}

type OwnersModel is Data:
  OwnersModel{input: Buffer.Model<F32>,gradient: Buffer.Model<F32>}

def hold(owners: OwnersModel) -> Kernel.Owners:
  OwnersModel{input,gradient} = owners
  Kernel.Owners{Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)}

def see(owners: Kernel.Owners) -> OwnersSeen:
  Kernel.Owners{input,gradient} = owners
  OwnersSeen{Buffer.contents(~F32,input),Buffer.contents(~F32,gradient)}

def seen_model(owners: OwnersModel) -> OwnersSeen:
  OwnersModel{input,gradient} = owners
  OwnersSeen{Buffer.content(~F32,input),Buffer.content(~F32,gradient)}

def seen_pair(seen: OwnersSeen) -> Buffer.Content<F32> & Buffer.Content<F32>:
  OwnersSeen{input,gradient} = seen
  (input,gradient)

def owners_kept(-input: Storage.Buffer<F32>,-gradient: Storage.Buffer<F32>,input_observation: Buffer.Observation<F32,input>,gradient_observation: Buffer.Observation<F32,gradient>) ->
  Each.Held<Kernel.Owners,OwnersModel,hold,Kernel.Owners{input,gradient}>:
  Buffer.Observation{+input_model,input_equal} = input_observation
  Buffer.Observation{+gradient_model,gradient_equal} = gradient_observation
  Each.Held{OwnersModel{input_model,gradient_model},
    %input_equal : {hold(OwnersModel{input_model,gradient_model}) == Kernel.Owners{_,gradient} : Kernel.Owners}
    %gradient_equal : {hold(OwnersModel{input_model,gradient_model}) == Kernel.Owners{Buffer.pack(~F32,input_model),_} : Kernel.Owners}
    {==}}

def owners_held(owners: Kernel.Owners) -> Each.Held<Kernel.Owners,OwnersModel,hold,owners>:
  Kernel.Owners{input,gradient} = owners
  owners_kept(input,gradient,Buffer.observe(~F32,input),Buffer.observe(~F32,gradient))

def owners_observed(-owners: Kernel.Owners,held: Each.Held<Kernel.Owners,OwnersModel,hold,owners>) -> Model.Packed<Kernel.Owners,OwnersModel,hold,owners>:
  Each.Held{kept,equation} = held
  Model.Packed{kept,equation}

def owners_packed(owners: Kernel.Owners) -> Model.Packed<Kernel.Owners,OwnersModel,hold,owners>:
  owners_observed(owners,owners_held(owners))

def held_seen(+owners: OwnersModel) -> {see(hold(owners)) == seen_model(owners) : OwnersSeen}:
  OwnersModel{+input,+gradient} = owners
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,input)),Buffer.content(~F32,input),Buffer.packed_contents(~F32,input)) :
    {OwnersSeen{_,Buffer.contents(~F32,Buffer.pack(~F32,gradient))} == seen_model(owners) : OwnersSeen}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,gradient)),Buffer.content(~F32,gradient),Buffer.packed_contents(~F32,gradient)) :
    {OwnersSeen{Buffer.content(~F32,input),_} == seen_model(owners) : OwnersSeen}
  {==}

# ---- The owners made along the plan ----

def plan(+depth: Nat,+work: Parameter.Work) -> Bands.Plan:
  Bands.unit_plan(depth,8,0,Kernel.outputs(work))

# The output gradient seen as images of output rows.
def gradient_geometry(work: Parameter.Work) -> Bands.Geometry:
  Parameter.Work{n,c,o,h,w,k,s,p,height,width,g} = work
  Bands.Geometry{1,U32.from_nat(n),U32.from_nat(o),U32.from_nat(Nat.mul(height,width))}

# A part sees the whole input and the gradient rows of its outputs.
def piece(work: Parameter.Work,+before: OwnersSeen,+first: U32,+rows: U32) -> OwnersSeen:
  OwnersSeen{input,gradient} = before
  OwnersSeen{input,FromWhole.band(~F32,gradient_geometry(work),0.0,gradient,first,rows)}

def owned_view(+first: U32,+rows: U32,inputs: Buffer.Content<F32> & Buffer.Content<F32>,gradients: Buffer.Content<F32> & Buffer.Content<F32>) -> OwnersSeen & Model.Shaped<OwnersSeen>:
  (input,copy) = inputs
  (gradient,channels) = gradients
  (OwnersSeen{input,gradient},Model.Piece{OwnersSeen{copy,channels},first,rows})

def owned_seen(+first: U32,+rows: U32,inputs: Storage.Buffer<F32> & Storage.Buffer<F32>,gradients: Storage.Buffer<F32> & Storage.Buffer<F32>) ->
  {Each.built_seen(~Kernel.Owners,~Kernel.Owners,~OwnersSeen,~see,~OwnersSeen,~see,Kernel.owned(first,rows,inputs,gradients)) ==
    owned_view(first,rows,CloneLaws.contents(~F32,inputs),CloneLaws.contents(~F32,gradients)) : OwnersSeen & Model.Shaped<OwnersSeen>}:
  (input,copy) = inputs
  (gradient,channels) = gradients
  {==}

# The leaf keeps what is seen of the input and the gradient, and gives the
# part the whole input and the gradient rows of its outputs.
def own_seen(+work: Parameter.Work,+first: U32,+rows: U32,+kept: OwnersModel) ->
  {Each.built_seen(~Kernel.Owners,~Kernel.Owners,~OwnersSeen,~see,~OwnersSeen,~see,Kernel.own(work,first,rows,hold(kept))) ==
    (see(hold(kept)),Model.Piece{piece(work,see(hold(kept)),first,rows),first,rows}) : OwnersSeen & Model.Shaped<OwnersSeen>}:
  Parameter.Work{+n,+c,+o,+h,+w,+k,+s,+p,+height,+width,+g} = work
  OwnersModel{+input,+gradient} = kept
  +geometry = {Bands.Geometry{1,U32.from_nat(n),U32.from_nat(o),U32.from_nat(Nat.mul(height,width))} : Bands.Geometry}
  +input_content = Buffer.content(~F32,input)
  +gradient_content = Buffer.content(~F32,gradient)
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,input)),input_content,Buffer.packed_contents(~F32,input)) :
    {Each.built_seen(~Kernel.Owners,~Kernel.Owners,~OwnersSeen,~see,~OwnersSeen,~see,Kernel.own(work,first,rows,hold(kept))) ==
      (OwnersSeen{_,Buffer.contents(~F32,Buffer.pack(~F32,gradient))},
        Model.Piece{OwnersSeen{_,FromWhole.band(~F32,geometry,0.0,Buffer.contents(~F32,Buffer.pack(~F32,gradient)),first,rows)},first,rows}) : OwnersSeen & Model.Shaped<OwnersSeen>}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,gradient)),gradient_content,Buffer.packed_contents(~F32,gradient)) :
    {Each.built_seen(~Kernel.Owners,~Kernel.Owners,~OwnersSeen,~see,~OwnersSeen,~see,Kernel.own(work,first,rows,hold(kept))) ==
      (OwnersSeen{input_content,_},Model.Piece{OwnersSeen{input_content,FromWhole.band(~F32,geometry,0.0,_,first,rows)},first,rows}) : OwnersSeen & Model.Shaped<OwnersSeen>}
  Equal.trans(OwnersSeen & Model.Shaped<OwnersSeen>,Each.built_seen(~Kernel.Owners,~Kernel.Owners,~OwnersSeen,~see,~OwnersSeen,~see,Kernel.own(work,first,rows,hold(kept))),
    owned_view(first,rows,CloneLaws.contents(~F32,Clone.clone(~F32,Buffer.pack(~F32,input))),
      CloneLaws.contents(~F32,Bands.whole_rows(~F32,geometry,0.0,first,rows,Buffer.pack(~F32,gradient)))),
    owned_view(first,rows,(input_content,input_content),(gradient_content,FromWhole.band(~F32,geometry,0.0,gradient_content,first,rows))),
    owned_seen(first,rows,Clone.clone(~F32,Buffer.pack(~F32,input)),Bands.whole_rows(~F32,geometry,0.0,first,rows,Buffer.pack(~F32,gradient))),
    %Equal.sym(Buffer.Content<F32> & Buffer.Content<F32>,CloneLaws.contents(~F32,Clone.clone(~F32,Buffer.pack(~F32,input))),(input_content,input_content),
        CloneLaws.packed(~F32,input)) :
      {owned_view(first,rows,_,CloneLaws.contents(~F32,Bands.whole_rows(~F32,geometry,0.0,first,rows,Buffer.pack(~F32,gradient)))) ==
        owned_view(first,rows,(input_content,input_content),(gradient_content,FromWhole.band(~F32,geometry,0.0,gradient_content,first,rows))) : OwnersSeen & Model.Shaped<OwnersSeen>}
    %Equal.sym(Buffer.Content<F32> & Buffer.Content<F32>,CloneLaws.contents(~F32,Bands.whole_rows(~F32,geometry,0.0,first,rows,Buffer.pack(~F32,gradient))),
        (gradient_content,FromWhole.band(~F32,geometry,0.0,gradient_content,first,rows)),FromWhole.rows_contents(~F32,geometry,0.0,first,rows,gradient)) :
      {owned_view(first,rows,(input_content,input_content),_) ==
        owned_view(first,rows,(input_content,input_content),(gradient_content,FromWhole.band(~F32,geometry,0.0,gradient_content,first,rows))) : OwnersSeen & Model.Shaped<OwnersSeen>}
    {==})

# ---- The values of the parts ----

def pack_values(values: Tree.Tree<F32>) -> Array<F32>:
  Tree.pack(F32,values)

# The values of the outputs of a part, [tap][output], from its owners.
def part_value(work: Parameter.Work,+owners: OwnersModel,+first: U32,+rows: U32) -> Tree.Tree<F32>:
  OwnersModel{input,gradient} = owners
  +work = work
  Groups.group_output(WeightModel.values(Kernel.part_work(work,rows),input,gradient))

def represented_values(+result: Groups.Group) -> {Kernel.part_values(Groups.represent(result)) == pack_values(Groups.group_output(result)) : Array<F32>}:
  Groups.Group{input,gradient,values} = result
  {==}

def part_valued(+work: Parameter.Work,+owners: OwnersModel,+first: U32,+rows: U32) ->
  {Kernel.part(work,first,rows,hold(owners),Unit{}) == Partition.Leaf{pack_values(part_value(work,owners,first,rows)),first,rows,Unit{}} : Partition.Tree<&1,Array<F32>,Unit>}:
  OwnersModel{+input,+gradient} = owners
  +part = Kernel.part_work(work,rows)
  %Equal.sym((Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>,Kernel.weight_values(part,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),
      Groups.represent(WeightModel.values(part,input,gradient)),WeightModel.values_matches(part,input,gradient)) :
    {Partition.Leaf{Kernel.part_values(_),first,rows,Unit{}} == Partition.Leaf{pack_values(part_value(work,owners,first,rows)),first,rows,Unit{}} : Partition.Tree<&1,Array<F32>,Unit>}
  Equal.cong(Array<F32>,Partition.Tree<&1,Array<F32>,Unit>,values => Partition.Leaf{values,first,rows,Unit{}},
    Kernel.part_values(Groups.represent(WeightModel.values(part,input,gradient))),pack_values(Groups.group_output(WeightModel.values(part,input,gradient))),
    represented_values(WeightModel.values(part,input,gradient)))

def valued(+work: Parameter.Work,owners: Model.Shaped<OwnersModel>) -> Model.Shaped<Tree.Tree<F32>>:
  Model.shaped_valued(~OwnersModel,~Tree.Tree<F32>,~Parameter.Work,~part_value,work,owners)

# ---- The rows of the parts written into the result ----

def taps_width(taps: U32) -> Nat:
  U32.to_nat(taps)

# The values of a part, [output][tap].
def part_list(taps: U32,+values: Tree.Tree<F32>,+first: U32,+rows: U32) -> List<&2,F32>:
  WeightModel.transpose(U32.to_nat(rows),0,rows,U32.to_nat(taps),values)

def work_taps(+work: Parameter.Work) -> U32:
  Kernel.taps(work)

# The values of the outputs of a part, [output][tap], from its owners.
def owner_list(work: Parameter.Work,+owners: OwnersModel,+first: U32,+rows: U32) -> List<&2,F32>:
  +work = work
  part_list(Kernel.taps(work),part_value(work,owners,first,rows),first,rows)

def owner_listed(+work: Parameter.Work,+owners: OwnersModel,+first: U32,+rows: U32) ->
  {part_list(work_taps(work),part_value(work,owners,first,rows),first,rows) == owner_list(work,owners,first,rows) : List<&2,F32>}:
  {==}

def part_step(taps: U32,+first: U32,+rows: U32,values: Tree.Tree<F32>,content: Buffer.Content<F32>) -> Buffer.Content<F32>:
  Buffer.Content{tree,length} = content
  +taps = taps
  +values = values
  Buffer.Content{Tree.write_sequence(~F32,part_list(taps,values,first,rows),U32.from_nat(Nat.mul(U32.to_nat(first),taps_width(taps))),tree),length}

def part_written(+taps: U32,+first: U32,+rows: U32,+values: Tree.Tree<F32>,+tree: Tree.Tree<F32>,+length: Nat) ->
  {part_step(taps,first,rows,values,Buffer.Content{tree,length}) ==
    Buffer.Content{Tree.write_sequence(~F32,part_list(taps,values,first,rows),U32.from_nat(Nat.mul(U32.to_nat(first),taps_width(taps))),tree),length} : Buffer.Content<F32>}:
  {==}

def part_length(+taps: U32,+values: Tree.Tree<F32>,+first: U32,+rows: U32) ->
  {List.length(&2,F32,part_list(taps,values,first,rows)) == Nat.mul(U32.to_nat(rows),taps_width(taps)) : Nat}:
  WeightModel.transpose_length(U32.to_nat(rows),0,rows,U32.to_nat(taps),values)

def work_width(work: Parameter.Work) -> Nat:
  U32.to_nat(Kernel.taps(work))

def owner_length(+work: Parameter.Work,+owners: OwnersModel,+first: U32,+rows: U32) ->
  {List.length(&2,F32,owner_list(work,owners,first,rows)) == Nat.mul(U32.to_nat(rows),work_width(work)) : Nat}:
  part_length(Kernel.taps(work),part_value(work,owners,first,rows),first,rows)

def part_rows_step(+taps: U32,+first: U32,+rows: U32,+values: Tree.Tree<F32>,+kept: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Kernel.part_rows(taps,first,rows,pack_values(values),Buffer.pack(~F32,kept))) ==
    part_step(taps,first,rows,values,Buffer.contents(~F32,Buffer.pack(~F32,kept))) : Buffer.Content<F32>}:
  Buffer.Model{+tree,+length,+certificate} = kept
  +offset = Equal.trans(U32,((first + 0 : U32) * taps : U32),(first * taps : U32),U32.from_nat(Nat.mul(U32.to_nat(first),U32.to_nat(taps))),
    Equal.cong(U32,U32,start => (start * taps : U32),(first + 0 : U32),first,Arithmetic.add_zero(first)),Multiplication.native_multiply(first,taps))
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,kept)),Buffer.content(~F32,kept),Buffer.packed_contents(~F32,kept)) :
    {Buffer.contents(~F32,Kernel.part_rows(taps,first,rows,pack_values(values),Buffer.pack(~F32,kept))) == part_step(taps,first,rows,values,_) : Buffer.Content<F32>}
  %offset :
    {Buffer.contents(~F32,Kernel.part_rows(taps,first,rows,pack_values(values),Buffer.pack(~F32,kept))) ==
      Buffer.Content{Tree.write_sequence(~F32,WeightModel.transpose(U32.to_nat(rows),0,rows,U32.to_nat(taps),values),_,tree),length} : Buffer.Content<F32>}
  WeightModel.rows_written(rows,taps,first,values,kept)

def count(+work: Parameter.Work) -> U32:
  (Kernel.outputs(work) * Kernel.taps(work) : U32)

def fresh(+work: Parameter.Work) -> Tree.Tree<F32>:
  Tree.fresh(F32,Tensors.depth_for(count(work)),0.0)

# The values of the parts one after the other, [output][tap].
def parts_list(+work: Parameter.Work,owners: Model.Shaped<OwnersModel>) -> List<&2,F32>:
  Each.shaped_list(~OwnersModel,~F32,~Parameter.Work,~owner_list,work,owners)

def parts_content(+work: Parameter.Work,owners: Model.Shaped<OwnersModel>) -> Buffer.Content<F32>:
  Buffer.Content{Tree.write_sequence(~F32,parts_list(work,owners),0,fresh(work)),U32.to_nat(count(work))}

# The owners lie on the plan they were made along.
def owners_plan(+depth: Nat,+work: Parameter.Work,+before: OwnersSeen,+owners: Model.Shaped<OwnersModel>,
  +mapped: {Model.shaped_map(~OwnersModel,~OwnersSeen,~seen_model,owners) == Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,plan(depth,work)) : Model.Shaped<OwnersSeen>}) ->
  {Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners) == plan(depth,work) : Bands.Plan}:
  +mapped_seen = Model.shaped_map(~OwnersModel,~OwnersSeen,~seen_model,owners)
  Equal.trans(Bands.Plan,Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners),
    Model.shaped_plan(~OwnersSeen,~Each.piece_rows(~OwnersSeen),mapped_seen),plan(depth,work),
    Equal.sym(Bands.Plan,Model.shaped_plan(~OwnersSeen,~Each.piece_rows(~OwnersSeen),mapped_seen),Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners),
      Each.shaped_map_plan(~OwnersModel,~OwnersSeen,~seen_model,owners)),
    Equal.trans(Bands.Plan,Model.shaped_plan(~OwnersSeen,~Each.piece_rows(~OwnersSeen),mapped_seen),Model.shaped_plan(~OwnersSeen,~Each.piece_rows(~OwnersSeen),Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,plan(depth,work))),plan(depth,work),
      Equal.cong(Model.Shaped<OwnersSeen>,Bands.Plan,tree => Model.shaped_plan(~OwnersSeen,~Each.piece_rows(~OwnersSeen),tree),mapped_seen,Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,plan(depth,work)),mapped),
      Each.shaped_build_plan(~OwnersSeen,~OwnersSeen,~Parameter.Work,~piece,work,before,plan(depth,work))))

# The owners cover the outputs [0, outputs).
def owners_coverage(+depth: Nat,+work: Parameter.Work,+owners: Model.Shaped<OwnersModel>,+capacity_ok: {U32.is_le(Kernel.outputs(work),2147483648) == True{} : Bool},
  +plan_equal: {Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners) == plan(depth,work) : Bands.Plan}) ->
  BandPlan.Coverage(Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners),0n,U32.to_nat(Kernel.outputs(work))):
  %Equal.sym(Bands.Plan,Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners),plan(depth,work),plan_equal) :
    BandPlan.Coverage(_,0n,U32.to_nat(Kernel.outputs(work)))
  BandPlan.refinement_unit_plan(depth,8,0,Kernel.outputs(work),Kernel.outputs(work),capacity_ok,32,{==},{==},{==},{==},Order.reflexive(U32.to_nat(Kernel.outputs(work))))

def parts_coverage(+depth: Nat,+work: Parameter.Work,+owners: Model.Shaped<OwnersModel>,+capacity_ok: {U32.is_le(Kernel.outputs(work),2147483648) == True{} : Bool},
  +plan_equal: {Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners) == plan(depth,work) : Bands.Plan}) ->
  BandPlan.Coverage(Model.shaped_plan(~Tree.Tree<F32>,~Each.piece_rows(~Tree.Tree<F32>),valued(work,owners)),0n,U32.to_nat(Kernel.outputs(work))):
  %Equal.sym(Bands.Plan,Model.shaped_plan(~Tree.Tree<F32>,~Each.piece_rows(~Tree.Tree<F32>),valued(work,owners)),Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners),
      Each.shaped_valued_plan(~OwnersModel,~Tree.Tree<F32>,~Parameter.Work,~part_value,work,owners)) :
    BandPlan.Coverage(_,0n,U32.to_nat(Kernel.outputs(work)))
  owners_coverage(depth,work,owners,capacity_ok,plan_equal)

# The parts list a row of taps for every output.
def parts_length(+depth: Nat,+work: Parameter.Work,+owners: Model.Shaped<OwnersModel>,+capacity_ok: {U32.is_le(Kernel.outputs(work),2147483648) == True{} : Bool},
  +plan_equal: {Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners) == plan(depth,work) : Bands.Plan}) ->
  {List.length(&2,F32,parts_list(work,owners)) == Nat.mul(U32.to_nat(Kernel.outputs(work)),work_width(work)) : Nat}:
  Each.shaped_list_length(~OwnersModel,~F32,~Parameter.Work,~owner_list,~work_width,~owner_length,work,owners,0n,U32.to_nat(Kernel.outputs(work)),
    owners_coverage(depth,work,owners,capacity_ok,plan_equal))

# The result the parts write, from a fresh buffer.
def folded(+depth: Nat,+work: Parameter.Work,owners: Model.Shaped<OwnersModel>) -> Buffer.Content<F32>:
  Buffer.contents(~F32,Bands.fold(~Array<F32>,~Storage.Buffer<F32>,~U32,~Kernel.part_rows,Kernel.taps(work),
    Bands.each(~Kernel.Owners,~Array<F32>,~Unit,~Parameter.Work,~Bands.unshared,~Kernel.part,~Kernel.unmatched,plan(depth,work),True{},
      work,Model.shaped_pack(~OwnersModel,~Kernel.Owners,~hold,owners),Unit{}),
    Tensors.buffer(~F32,0.0,count(work))))

def parts_folded(+depth: Nat,+work: Parameter.Work,+owners: Model.Shaped<OwnersModel>,+capacity_ok: {U32.is_le(Kernel.outputs(work),2147483648) == True{} : Bool},
  +plan_equal: {Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners) == plan(depth,work) : Bands.Plan}) ->
  {folded(depth,work,owners) == parts_content(work,owners) : Buffer.Content<F32>}:
  +taps = Kernel.taps(work)
  +outputs = Kernel.outputs(work)
  +start = {Buffer.Content{fresh(work),U32.to_nat(count(work))} : Buffer.Content<F32>}
  +each_equal = Equal.trans(Partition.Tree<&1,Array<F32>,Unit>,
    Bands.each(~Kernel.Owners,~Array<F32>,~Unit,~Parameter.Work,~Bands.unshared,~Kernel.part,~Kernel.unmatched,Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners),True{},
      work,Model.shaped_pack(~OwnersModel,~Kernel.Owners,~hold,owners),Unit{}),
    Each.shaped_leaf_map(~OwnersModel,~Kernel.Owners,~hold,~Array<F32>,~Parameter.Work,~Kernel.part,work,owners),
    Model.shaped_pack(~Tree.Tree<F32>,~Array<F32>,~pack_values,valued(work,owners)),
    Each.shaped_planned(~OwnersModel,~Kernel.Owners,~hold,~Array<F32>,~Parameter.Work,~Kernel.part,~Kernel.unmatched,owners,True{},work),
    Each.shaped_leaf_valued(~OwnersModel,~Kernel.Owners,~hold,~Tree.Tree<F32>,~Array<F32>,~pack_values,~Parameter.Work,~Kernel.part,~part_value,~part_valued,work,owners))
  +start_equal = Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,count(work))))),
    Buffer.content(~F32,Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,count(work)))),start,
    Buffer.packed_contents(~F32,Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,count(work)))),
    Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,count(work)))),Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count(work))),start,
      Buffer.remodel_content(~F32,Tensors.buffer(~F32,0.0,count(work))),Listed.allocated(~F32,0.0,count(work))))
  %plan_equal :
    {Buffer.contents(~F32,Bands.fold(~Array<F32>,~Storage.Buffer<F32>,~U32,~Kernel.part_rows,taps,
      Bands.each(~Kernel.Owners,~Array<F32>,~Unit,~Parameter.Work,~Bands.unshared,~Kernel.part,~Kernel.unmatched,_,True{},
        work,Model.shaped_pack(~OwnersModel,~Kernel.Owners,~hold,owners),Unit{}),
      Tensors.buffer(~F32,0.0,count(work)))) == parts_content(work,owners) : Buffer.Content<F32>}
  %Equal.sym(Partition.Tree<&1,Array<F32>,Unit>,
      Bands.each(~Kernel.Owners,~Array<F32>,~Unit,~Parameter.Work,~Bands.unshared,~Kernel.part,~Kernel.unmatched,Model.shaped_plan(~OwnersModel,~Each.piece_rows(~OwnersModel),owners),True{},
        work,Model.shaped_pack(~OwnersModel,~Kernel.Owners,~hold,owners),Unit{}),
      Model.shaped_pack(~Tree.Tree<F32>,~Array<F32>,~pack_values,valued(work,owners)),each_equal) :
    {Buffer.contents(~F32,Bands.fold(~Array<F32>,~Storage.Buffer<F32>,~U32,~Kernel.part_rows,taps,_,Tensors.buffer(~F32,0.0,count(work)))) == parts_content(work,owners) : Buffer.Content<F32>}
  %Buffer.remodeled(~F32,Tensors.buffer(~F32,0.0,count(work))) :
    {Buffer.contents(~F32,Bands.fold(~Array<F32>,~Storage.Buffer<F32>,~U32,~Kernel.part_rows,taps,Model.shaped_pack(~Tree.Tree<F32>,~Array<F32>,~pack_values,valued(work,owners)),_)) ==
      parts_content(work,owners) : Buffer.Content<F32>}
  Equal.trans(Buffer.Content<F32>,
    Buffer.contents(~F32,Bands.fold(~Array<F32>,~Storage.Buffer<F32>,~U32,~Kernel.part_rows,taps,Model.shaped_pack(~Tree.Tree<F32>,~Array<F32>,~pack_values,valued(work,owners)),
      Buffer.pack(~F32,Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,count(work)))))),
    Model.shaped_thread(~Tree.Tree<F32>,~Buffer.Content<F32>,~U32,~part_step,taps,valued(work,owners),Buffer.contents(~F32,Buffer.pack(~F32,Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,count(work)))))),
    parts_content(work,owners),
    Each.shaped_folded(~Tree.Tree<F32>,~Array<F32>,~pack_values,~Storage.Buffer<F32>,~U32,~Buffer.Model<F32>,~Buffer.pack(~F32),~Each.buffer_held(~F32),~Buffer.Content<F32>,~Buffer.contents(~F32),
      ~Kernel.part_rows,~part_step,~part_rows_step,valued(work,owners),taps,Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,count(work)))),
    %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,count(work))))),start,start_equal) :
      {Model.shaped_thread(~Tree.Tree<F32>,~Buffer.Content<F32>,~U32,~part_step,taps,valued(work,owners),_) == parts_content(work,owners) : Buffer.Content<F32>}
    %Each.shaped_list_valued(~OwnersModel,~Tree.Tree<F32>,~F32,~U32,~Parameter.Work,~part_value,~part_list,~work_taps,~owner_list,~owner_listed,work,owners) :
      {Model.shaped_thread(~Tree.Tree<F32>,~Buffer.Content<F32>,~U32,~part_step,taps,valued(work,owners),start) ==
        Buffer.Content{Tree.write_sequence(~F32,_,0,fresh(work)),U32.to_nat(count(work))} : Buffer.Content<F32>}
    Each.shaped_written(~Tree.Tree<F32>,~F32,~U32,~part_step,~part_list,~taps_width,~part_written,~part_length,
      taps,valued(work,owners),0n,U32.to_nat(outputs),parts_coverage(depth,work,owners,capacity_ok,plan_equal),fresh(work),U32.to_nat(count(work))))

# ---- The parts ----

def returned(result: (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>) -> (Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>:
  (sources,weight) = result
  (CloneLaws.contents(~F32,sources),Buffer.contents(~F32,weight))

# The parts give back the contents of the input and the gradient, and write
# the values of their owners into a fresh result. Every part owns the whole
# input and the gradient rows of its outputs.
type Parts<-work: Parameter.Work,-layout: Bands.Plan,-before: OwnersSeen,-result: (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>> is Data:
  Parts{owners: Model.Shaped<OwnersModel>,
    seen: {Model.shaped_map(~OwnersModel,~OwnersSeen,~seen_model,owners) == Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,layout) : Model.Shaped<OwnersSeen>},
    contents: {returned(result) == (seen_pair(before),parts_content(work,owners)) : (Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>}}

def parts_packed(+depth: Nat,+work: Parameter.Work,+before: OwnersSeen,+capacity_ok: {U32.is_le(Kernel.outputs(work),2147483648) == True{} : Bool},
  +kept: OwnersModel,+owners: Model.Shaped<OwnersModel>,
  +seen_equal: {Each.built_seen(~Kernel.Owners,~Kernel.Owners,~OwnersSeen,~see,~OwnersSeen,~see,(hold(kept),Model.shaped_pack(~OwnersModel,~Kernel.Owners,~hold,owners))) ==
    (before,Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,plan(depth,work))) : OwnersSeen & Model.Shaped<OwnersSeen>}) ->
  Parts<work,plan(depth,work),before,Kernel.parts_run(depth,work,(hold(kept),Model.shaped_pack(~OwnersModel,~Kernel.Owners,~hold,owners)))>:
  OwnersModel{+input,+gradient} = kept
  +packed_seen = Model.shaped_seen(~Kernel.Owners,~OwnersSeen,~see,Model.shaped_pack(~OwnersModel,~Kernel.Owners,~hold,owners))
  +mapped_seen = Model.shaped_map(~OwnersModel,~OwnersSeen,~seen_model,owners)
  +sources_seen = Equal.cong(OwnersSeen & Model.Shaped<OwnersSeen>,OwnersSeen,pair => Pair.fst(OwnersSeen,Model.Shaped<OwnersSeen>,pair),
    (see(hold(kept)),packed_seen),(before,Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,plan(depth,work))),seen_equal)
  +shapes_seen = Equal.cong(OwnersSeen & Model.Shaped<OwnersSeen>,Model.Shaped<OwnersSeen>,pair => Pair.snd(OwnersSeen,Model.Shaped<OwnersSeen>,pair),
    (see(hold(kept)),packed_seen),(before,Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,plan(depth,work))),seen_equal)
  +mapped = Equal.trans(Model.Shaped<OwnersSeen>,mapped_seen,packed_seen,Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,plan(depth,work)),
    Equal.sym(Model.Shaped<OwnersSeen>,packed_seen,mapped_seen,Model.shaped_packed(~OwnersModel,~OwnersSeen,~Kernel.Owners,~hold,~see,~seen_model,~held_seen,owners)),
    shapes_seen)
  +plan_equal = owners_plan(depth,work,before,owners,mapped)
  Parts{owners,mapped,
    %sources_seen :
      {returned(Kernel.parts_run(depth,work,(hold(kept),Model.shaped_pack(~OwnersModel,~Kernel.Owners,~hold,owners)))) == (seen_pair(_),parts_content(work,owners)) :
        (Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>}
    Equal.cong(Buffer.Content<F32>,(Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>,
      weight => ((Buffer.contents(~F32,Buffer.pack(~F32,input)),Buffer.contents(~F32,Buffer.pack(~F32,gradient))),weight),
      folded(depth,work,owners),parts_content(work,owners),parts_folded(depth,work,owners,capacity_ok,plan_equal))}

def parts_split(+depth: Nat,+work: Parameter.Work,+before: OwnersSeen,+capacity_ok: {U32.is_le(Kernel.outputs(work),2147483648) == True{} : Bool},
  -sources: Kernel.Owners,held: Each.Held<Kernel.Owners,OwnersModel,hold,sources>,
  -tree: Partition.Tree<&1,Kernel.Owners,Unit>,observation: Model.ShapedObservation<OwnersModel,Kernel.Owners,hold,tree>,
  +seen_equal: {Each.built_seen(~Kernel.Owners,~Kernel.Owners,~OwnersSeen,~see,~OwnersSeen,~see,(sources,tree)) ==
    (before,Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,plan(depth,work))) : OwnersSeen & Model.Shaped<OwnersSeen>}) ->
  Parts<work,plan(depth,work),before,Kernel.parts_run(depth,work,(sources,tree))>:
  Each.Held{+kept,+kept_equal} = held
  Model.ShapedObservation{+owners,+tree_equal} = observation
  %kept_equal : Parts<work,plan(depth,work),before,Kernel.parts_run(depth,work,(_,tree))>
  %tree_equal : Parts<work,plan(depth,work),before,Kernel.parts_run(depth,work,(hold(kept),_))>
  parts_packed(depth,work,before,capacity_ok,kept,owners,
    %Equal.sym(Partition.Tree<&1,Kernel.Owners,Unit>,Model.shaped_pack(~OwnersModel,~Kernel.Owners,~hold,owners),tree,tree_equal) :
      {Each.built_seen(~Kernel.Owners,~Kernel.Owners,~OwnersSeen,~see,~OwnersSeen,~see,(hold(kept),_)) ==
        (before,Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,plan(depth,work))) : OwnersSeen & Model.Shaped<OwnersSeen>}
    %Equal.sym(Kernel.Owners,hold(kept),sources,kept_equal) :
      {Each.built_seen(~Kernel.Owners,~Kernel.Owners,~OwnersSeen,~see,~OwnersSeen,~see,(_,tree)) ==
        (before,Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,plan(depth,work))) : OwnersSeen & Model.Shaped<OwnersSeen>}
    seen_equal)

def parts_made(+depth: Nat,+work: Parameter.Work,+before: OwnersSeen,+capacity_ok: {U32.is_le(Kernel.outputs(work),2147483648) == True{} : Bool},
  made: Kernel.Owners & Partition.Tree<&1,Kernel.Owners,Unit>,
  +seen_equal: {Each.built_seen(~Kernel.Owners,~Kernel.Owners,~OwnersSeen,~see,~OwnersSeen,~see,made) ==
    (before,Model.shaped_build(~OwnersSeen,~OwnersSeen,~piece(work),before,plan(depth,work))) : OwnersSeen & Model.Shaped<OwnersSeen>}) ->
  Parts<work,plan(depth,work),before,Kernel.parts_run(depth,work,made)>:
  (sources,tree) = made
  parts_split(depth,work,before,capacity_ok,sources,owners_held(sources),tree,Model.shaped_observe(~OwnersModel,~Kernel.Owners,~hold,~owners_packed,tree),seen_equal)

# The parts of the outputs [0, outputs) of the plan of depth halvings.
def parts(+depth: Nat,+work: Parameter.Work,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>,+capacity_ok: {U32.is_le(Kernel.outputs(work),2147483648) == True{} : Bool}) ->
  Parts<work,plan(depth,work),OwnersSeen{Buffer.content(~F32,input),Buffer.content(~F32,gradient)},Kernel.parts(depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient))>:
  +kept = {OwnersModel{input,gradient} : OwnersModel}
  %Buffer.packed_contents(~F32,input) :
    Parts<work,plan(depth,work),OwnersSeen{_,Buffer.content(~F32,gradient)},Kernel.parts(depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient))>
  %Buffer.packed_contents(~F32,gradient) :
    Parts<work,plan(depth,work),OwnersSeen{Buffer.contents(~F32,Buffer.pack(~F32,input)),_},Kernel.parts(depth,work,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient))>
  parts_made(depth,work,see(hold(kept)),capacity_ok,Bands.build(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~Kernel.own,plan(depth,work),work,hold(kept)),
    Each.shaped_built(~Kernel.Owners,~Kernel.Owners,~Parameter.Work,~OwnersModel,~hold,~owners_held,~OwnersSeen,~see,~OwnersSeen,~see,~Kernel.own,~piece,~own_seen,
      plan(depth,work),work,kept))

# ---- One part ----

# The values of every output, [output][tap].
def serial_list(+work: Parameter.Work,input: Buffer.Model<F32>,gradient: Buffer.Model<F32>) -> List<&2,F32>:
  WeightModel.transpose(U32.to_nat(Kernel.outputs(work)),0,Kernel.outputs(work),U32.to_nat(Kernel.taps(work)),Groups.group_output(WeightModel.values(work,input,gradient)))

# The serial kernel gives back the contents of the input and the gradient,
# and writes the values of every output into a fresh result.
def serial(+work: Parameter.Work,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {returned(Kernel.weight(work,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient))) ==
    ((Buffer.content(~F32,input),Buffer.content(~F32,gradient)),Buffer.Content{Tree.write_sequence(~F32,serial_list(work,input,gradient),0,fresh(work)),U32.to_nat(count(work))}) :
    (Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>}:
  +outputs = Kernel.outputs(work)
  +taps = Kernel.taps(work)
  +result = WeightModel.values(work,input,gradient)
  +values = Groups.group_output(result)
  +weight = {Buffer.Content{Tree.write_sequence(~F32,serial_list(work,input,gradient),0,fresh(work)),U32.to_nat(count(work))} : Buffer.Content<F32>}
  %Equal.sym((Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>,Kernel.weight(work,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),WeightModel.written(work,result),
      WeightModel.weight_matches(work,input,gradient)) :
    {returned(_) == ((Buffer.content(~F32,input),Buffer.content(~F32,gradient)),weight) : (Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>}
  %Equal.sym(Groups.Group,result,Groups.Group{input,gradient,values},WeightModel.values_kept(work,input,gradient)) :
    {returned(WeightModel.written(work,_)) == ((Buffer.content(~F32,input),Buffer.content(~F32,gradient)),weight) : (Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,input)),Buffer.content(~F32,input),Buffer.packed_contents(~F32,input)) :
    {((_,Buffer.contents(~F32,Buffer.pack(~F32,gradient))),Buffer.contents(~F32,Kernel.rows_written(outputs,taps,0,Tree.pack(F32,values),Tensors.buffer(~F32,0.0,(outputs * taps : U32))))) == ((Buffer.content(~F32,input),Buffer.content(~F32,gradient)),weight) :
      (Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,gradient)),Buffer.content(~F32,gradient),Buffer.packed_contents(~F32,gradient)) :
    {((Buffer.content(~F32,input),_),Buffer.contents(~F32,Kernel.rows_written(outputs,taps,0,Tree.pack(F32,values),Tensors.buffer(~F32,0.0,(outputs * taps : U32))))) == ((Buffer.content(~F32,input),Buffer.content(~F32,gradient)),weight) :
      (Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>}
  Equal.cong(Buffer.Content<F32>,(Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>,
    content => ((Buffer.content(~F32,input),Buffer.content(~F32,gradient)),content),Buffer.contents(~F32,Kernel.rows_written(outputs,taps,0,Tree.pack(F32,values),Tensors.buffer(~F32,0.0,(outputs * taps : U32)))),weight,WeightModel.rows_fresh(outputs,taps,values))
