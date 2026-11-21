# The placing operations of the library: set_slice writes a part into a
# target, concat places parts one after another along an axis and pad places
# a tensor in a larger target. Each gives a tensor of the contract that holds
# at every cell what its sources give the cell.

import Base
import ../tensor.bend as Tensors
import ../tensor_tasks.bend as Tasks
import ../tensor_layout.bend as Layout
import ../tensor_shape.bend as Shapes
import ../kernel_gemm.bend as KernelGemm
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./machine_limit_bounds.bend as LimitBounds
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_global_shape_count.bend as Global
import ./tensor_storage_count.bend as StorageCount
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_components.bend as Components
import ./tensor_moved.bend as Moved
import ./tensor_moved_sources.bend as Sources
import ./tensor_move_readback.bend as Held
import ./tensor_placing_gather.bend as Gathered
import ./tensor_placing_facts.bend as Facts
import ./tensor_placing_moved.bend as PlacingMoved
import ./tensor_shape_refinement.bend as ShapeLaws
import ./boolean_logic.bend as Logic
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_full_range.bend as FullRange
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./word_encoding.bend as WordEncoding
import ./tensor_placing_law.bend as Law
import ./tensor_placing_parts.bend as Parts

# ---- The values of two kinds of source ----

# A source of a placing that reads a base or a part.
type Written<-Base: Data,-Part: Data> is Data:
  Before{base: Base}
  Over{part: Part}

def written_value(~Element: Data,~Base: Data,~base_value: Base -> U32 -> Element,~Part: Data,~part_value: Part -> U32 -> Element,written: Written<Base,Part>,index: U32) -> Element:
  match written:
    case Before{base}: base_value(base,index)
    case Over{part}: part_value(part,index)

# A tensor of the contract of a base holds the values of the base as a
# written source.
def as_before(~Element: Data,~Base: Data,~base_value: Base -> U32 -> Element,~Part: Data,~part_value: Part -> U32 -> Element,+base: Base,tensor: Tensors.Tensor<Element>,contract: Global.Contract(~Element,~Base,~base_value,base,tensor)) ->
  Global.Contract(~Element,~Written<Base,Part>,~written_value(~Element,~Base,~base_value,~Part,~part_value),Before{base},tensor):
  Global.relabeled(~Element,~Base,~Written<Base,Part>,~base_value,~written_value(~Element,~Base,~base_value,~Part,~part_value),
    ~(left => right => Global.Fixed(~Base,~Written<Base,Part>,base,Before{base},left,right)),
    ~(left => right => related => index => Global.fixed_agrees(~Element,~Base,~Written<Base,Part>,~base_value,~written_value(~Element,~Base,~base_value,~Part,~part_value),base,Before{base},index => {==},left,right,related,index)),
    base,Before{base},KernelGemm.Both{{==},{==}},tensor,contract)

# A tensor of the contract of a part holds the values of the part as a
# written source.
def as_over(~Element: Data,~Base: Data,~base_value: Base -> U32 -> Element,~Part: Data,~part_value: Part -> U32 -> Element,+part: Part,tensor: Tensors.Tensor<Element>,contract: Global.Contract(~Element,~Part,~part_value,part,tensor)) ->
  Global.Contract(~Element,~Written<Base,Part>,~written_value(~Element,~Base,~base_value,~Part,~part_value),Over{part},tensor):
  Global.relabeled(~Element,~Part,~Written<Base,Part>,~part_value,~written_value(~Element,~Base,~base_value,~Part,~part_value),
    ~(left => right => Global.Fixed(~Part,~Written<Base,Part>,part,Over{part},left,right)),
    ~(left => right => related => index => Global.fixed_agrees(~Element,~Part,~Written<Base,Part>,~part_value,~written_value(~Element,~Base,~base_value,~Part,~part_value),part,Over{part},index => {==},left,right,related,index)),
    part,Over{part},KernelGemm.Both{{==},{==}},tensor,contract)

# ---- Writing a part into a target ----

# The same value either way.
def same_pick(choice: Bool,+value: U32) -> {Bool.pick(U32,choice,value,value) == value : U32}:
  match choice:
    case True{}: {==}
    case False{}: {==}

# The target placed in itself from zero on every axis.
def whole_in(shape: List<&2,U32>,+index: U32) -> Facts.PlacedIn(shape,shape,Layout.shifts_at(List.length(&2,U32,shape),0,0,index)):
  match shape:
    case Nil{}: Unit{}
    case Con{+extent,+rest}:
      Held.Both{{%Equal.sym(U32,Bool.pick(U32,U32.is_eq(index,0),0,0),0,same_pick(U32.is_eq(index,0),0)) :
            {Nat.is_le(Nat.add(U32.to_nat(_),U32.to_nat(extent)),U32.to_nat(extent)) == True{} : Bool}
          Order.reflexive(U32.to_nat(extent)) : {Nat.is_le(Nat.add(U32.to_nat(Bool.pick(U32,U32.is_eq(index,0),0,0)),U32.to_nat(extent)),U32.to_nat(extent)) == True{} : Bool}},
        whole_in(rest,(index + 1 : U32))}

# An axis of a part that fits from start, or has the extent of the target.
def slice_axis(hit: Bool,+extent: U32,+width: U32,+start: U32,
  +fits: {Bool.pick(Bool,hit,U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),U32.is_eq(extent,width)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Bool.pick(U32,hit,start,0)),U32.to_nat(width)),U32.to_nat(extent)) == True{} : Bool}:
  match hit:
    case True{}:
      +s = U32.to_nat(start)
      +e = U32.to_nat(extent)
      +ordered = Bounds.as_nat_le(start,extent,Logic.left(U32.is_le(start,extent),U32.is_le(width,(extent - start : U32)),fits))
      +narrow = Bounds.as_nat_le(width,(extent - start : U32),Logic.right(U32.is_le(start,extent),U32.is_le(width,(extent - start : U32)),fits))
      +exact = FullRange.subtract(extent,start,extent,ordered,Order.sub_le(e,s))
      +inside = {%exact : {Nat.is_le(U32.to_nat(width),_) == True{} : Bool} narrow : {Nat.is_le(U32.to_nat(width),Nat.sub(e,s)) == True{} : Bool}}
      %Order.restore_shift(s,e,ordered) : {Nat.is_le(Nat.add(s,U32.to_nat(width)),_) == True{} : Bool}
      Equal.trans(Bool,Nat.is_le(Nat.add(s,U32.to_nat(width)),Nat.add(s,Nat.sub(e,s))),Nat.is_le(U32.to_nat(width),Nat.sub(e,s)),True{},
        Order.le_offset(s,U32.to_nat(width),Nat.sub(e,s)),inside)
    case False{}:
      %Equal.sym(U32,extent,width,Arithmetic.equal_of_is_eq(extent,width,fits)) : {Nat.is_le(U32.to_nat(width),U32.to_nat(_)) == True{} : Bool}
      Order.reflexive(U32.to_nat(width))

# A part that fits a target from start along axis is placed in the target.
def slice_in(shape: List<&2,U32>,part_shape: List<&2,U32>,+axis: U32,+start: U32,+index: U32,
  +fits: {Layout.slice_fits(shape,part_shape,axis,start,index) == True{} : Bool}) ->
  Facts.PlacedIn(shape,part_shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,index)):
  match shape part_shape:
    case Con{+extent,+rest} Con{+width,+part_rest}:
      +own = Bool.pick(Bool,U32.is_eq(index,axis),U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),U32.is_eq(extent,width))
      +later = Layout.slice_fits(rest,part_rest,axis,start,(index + 1 : U32))
      Held.Both{slice_axis(U32.is_eq(index,axis),extent,width,start,Logic.left(own,later,fits)),
        slice_in(rest,part_rest,axis,start,(index + 1 : U32),Logic.right(own,later,fits))}
    case Nil{} Nil{}:
      Unit{}
    case Con{+extent,+rest} Nil{}:
      Empty.absurd(Facts.PlacedIn(Con{extent,rest},Nil{},Layout.shifts_at(List.length(&2,U32,Con{extent,rest}),axis,start,index)),WordEncoding.false_true(fits))
    case Nil{} Con{+width,+part_rest}:
      Empty.absurd(Facts.PlacedIn(Nil{},Con{width,part_rest},Layout.shifts_at(0n,axis,start,index)),WordEncoding.false_true(fits))

# Two is no more sources than half a word.
def two_room(+limits: Machine.Limits) -> {Nat.is_le(2n,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(2n,16n,U32.to_nat(Machine.half_word(limits)),{==},Bounds.as_nat_le(16,Machine.half_word(limits),LimitBounds.sixteen_within_half(limits)))

def two_placed(~Element: Data,~Base: Data,~base_value: Base -> U32 -> Element,~Part: Data,~part_value: Part -> U32 -> Element,+base: Base,+part: Part,+base_model: PiecesModel.Model<Element>,+part_model: PiecesModel.Model<Element>,
  +shape: List<&2,U32>,+part_shape: List<&2,U32>,
  +base_held: Sources.SourceHeld(~Element,~Written<Base,Part>,~written_value(~Element,~Base,~base_value,~Part,~part_value),Before{base},base_model,shape),
  +part_held: Sources.SourceHeld(~Element,~Written<Base,Part>,~written_value(~Element,~Base,~base_value,~Part,~part_value),Over{part},part_model,part_shape)) ->
  PlacingMoved.SourcesPlaced(~Element,~Written<Base,Part>,~written_value(~Element,~Base,~base_value,~Part,~part_value),[Before{base},Over{part}],[shape,part_shape],Moved.Models{base_model,Moved.Models{part_model,Moved.NoModels{}}}):
  Held.Both{+laid,+holds} = base_held
  Held.Both{+part_laid,+part_holds} = part_held
  Held.Both{laid,Held.Both{holds,Held.Both{part_laid,Held.Both{part_holds,Unit{}}}}}

# A part of cells that fits a target along axis from start, both tensors of
# the contract, is written into the target: the result is a tensor of the
# contract that holds the part where it lies and the target elsewhere.
def set_slice(~Element: Data,~Base: Data,~base_value: Base -> U32 -> Element,~Part: Data,~part_value: Part -> U32 -> Element,+base: Base,+part: Part,+limits: Machine.Limits,
  +base_model: PiecesModel.Model<Element>,+part_model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+part_shape: List<&2,U32>,+axis: U32,+start: U32,
  +fill: Element,+part_fill: Element,+base_workers: U32,+part_workers: U32,
  +base_contract: Global.Contract(~Element,~Base,~base_value,base,Tensors.Tensor{PiecesModel.pack(~Element,base_model),shape,fill,Tensors.Usable{},base_workers}),
  +part_contract: Global.Contract(~Element,~Part,~part_value,part,Tensors.Tensor{PiecesModel.pack(~Element,part_model),part_shape,part_fill,Tensors.Usable{},part_workers}),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Layout.slice_fits(shape,part_shape,axis,start,0) == True{} : Bool},+nonempty: {Shapes.has_zero(part_shape) == False{} : Bool}) ->
  Global.Contract(~Element,~Gathered.Placings<Written<Base,Part>,Element>,~Gathered.placing_value(~Element,~Written<Base,Part>,~written_value(~Element,~Base,~base_value,~Part,~part_value)),Gathered.Placings{[Before{base},Over{part}],[Layout.placed_segments(shape,Layout.shifts_at(List.length(&2,U32,shape),0,0,0)),Layout.placed_segments(part_shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0))],[shape,part_shape],shape,fill},Layout.set_slice(~Element,Tensors.Tensor{PiecesModel.pack(~Element,base_model),shape,fill,Tensors.Usable{},base_workers},axis,start,Tensors.Tensor{PiecesModel.pack(~Element,part_model),part_shape,part_fill,Tensors.Usable{},part_workers})):
  +whole = Layout.shifts_at(List.length(&2,U32,shape),0,0,0)
  +shifts = Layout.shifts_at(List.length(&2,U32,shape),axis,start,0)
  +workers = U32.max(base_workers,part_workers)
  +segments = {[Layout.placed_segments(shape,Layout.shifts_at(List.length(&2,U32,shape),0,0,0)),Layout.placed_segments(part_shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0))] : List<&2,List<&2,List<&2,Tasks.Segment>>>}
  +sources = {[shape,part_shape] : List<&2,List<&2,U32>>}
  +shift_lists = {[whole,shifts] : List<&2,List<&2,U32>>}
  +clamps = {[False{},False{}] : List<&2,Bool>}
  +counted = Global.contract_counted(~Element,~Base,~base_value,base,PiecesModel.pack(~Element,base_model),shape,fill,base_workers,base_contract)
  +part_counted = Global.contract_counted(~Element,~Part,~part_value,part,PiecesModel.pack(~Element,part_model),part_shape,part_fill,part_workers,part_contract)
  +room = StorageCount.counted_room(limits,shape,counted,default_capacity)
  +part_room = StorageCount.counted_room(limits,part_shape,part_counted,default_capacity)
  +part_positive = CountCompletion.positive_elements(part_shape,nonempty)
  +within = slice_in(shape,part_shape,axis,start,0,fits)
  +positive = Facts.placed_positive(shape,part_shape,shifts,within,part_positive)
  +part_rank = {%Equal.sym(Nat,List.length(&2,U32,part_shape),List.length(&2,U32,shape),Facts.placed_length(shape,part_shape,shifts,within)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    rank_room : {Nat.is_le(List.length(&2,U32,part_shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +target_ok = {Held.Both{room,Held.Both{positive,rank_room}} : Components.ShapeOk(limits,shape)}
  +part_ok = {Held.Both{part_room,Held.Both{part_positive,part_rank}} : Components.ShapeOk(limits,part_shape)}
  +shaped = {Held.Both{target_ok,Held.Both{Facts.placed_axes(shape,shape,whole,limits,whole_in(shape,0),room,positive),
    Held.Both{part_ok,Held.Both{Facts.placed_axes(shape,part_shape,shifts,limits,within,room,positive),Unit{}}}}} :
    Facts.SourcesShaped(limits,shape,segments,sources,shift_lists,clamps)}
  +facts = Facts.placing_facts(limits,shape,segments,sources,shift_lists,clamps,target_ok,shaped,{==},two_room(limits))
  +placed = two_placed(~Element,~Base,~base_value,~Part,~part_value,base,part,base_model,part_model,shape,part_shape,
    Sources.source_held(~Element,~Written<Base,Part>,~written_value(~Element,~Base,~base_value,~Part,~part_value),Before{base},base_model,shape,fill,base_workers,
      as_before(~Element,~Base,~base_value,~Part,~part_value,base,Tensors.Tensor{PiecesModel.pack(~Element,base_model),shape,fill,Tensors.Usable{},base_workers},base_contract)),
    Sources.source_held(~Element,~Written<Base,Part>,~written_value(~Element,~Base,~base_value,~Part,~part_value),Over{part},part_model,part_shape,part_fill,part_workers,
      as_over(~Element,~Base,~base_value,~Part,~part_value,part,Tensors.Tensor{PiecesModel.pack(~Element,part_model),part_shape,part_fill,Tensors.Usable{},part_workers},part_contract)))
  %Equal.sym(Bool,Layout.slice_fits(shape,part_shape,axis,start,0),True{},fits) :
    Global.Contract(~Element,~Gathered.Placings<Written<Base,Part>,Element>,~Gathered.placing_value(~Element,~Written<Base,Part>,~written_value(~Element,~Base,~base_value,~Part,~part_value)),Gathered.Placings{[Before{base},Over{part}],[Layout.placed_segments(shape,Layout.shifts_at(List.length(&2,U32,shape),0,0,0)),Layout.placed_segments(part_shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0))],[shape,part_shape],shape,fill},
      Layout.slice_status(~Element,Tensors.Usable{},Tensors.Usable{},_,PiecesModel.pack(~Element,base_model),PiecesModel.pack(~Element,part_model),shape,part_shape,axis,start,fill,workers))
  %Equal.sym(Bool,Shapes.has_zero(part_shape),False{},nonempty) :
    Global.Contract(~Element,~Gathered.Placings<Written<Base,Part>,Element>,~Gathered.placing_value(~Element,~Written<Base,Part>,~written_value(~Element,~Base,~base_value,~Part,~part_value)),Gathered.Placings{[Before{base},Over{part}],[Layout.placed_segments(shape,Layout.shifts_at(List.length(&2,U32,shape),0,0,0)),Layout.placed_segments(part_shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0))],[shape,part_shape],shape,fill},
      Layout.slice_placed(~Element,_,PiecesModel.pack(~Element,base_model),PiecesModel.pack(~Element,part_model),shape,part_shape,axis,start,fill,workers))
  PlacingMoved.placed_contract(~Element,~Written<Base,Part>,~written_value(~Element,~Base,~base_value,~Part,~part_value),[Before{base},Over{part}],limits,shape,segments,sources,shift_lists,clamps,fill,workers,Moved.Models{base_model,Moved.Models{part_model,Moved.NoModels{}}},{==},facts,placed,
    Global.count(shape,counted),Global.accepted(shape,counted))

# ---- Placing parts one after another ----

# The parts of a concatenation: the logical, pieces model, shape, fill and
# workers of every part, in order.
type Inputs<-Element: Data,-Part: Data> is Data:
  Last{part: Part,model: PiecesModel.Model<Element>,shape: List<&2,U32>,fill: Element,workers: U32}
  More{part: Part,model: PiecesModel.Model<Element>,shape: List<&2,U32>,fill: Element,workers: U32,rest: Inputs<Element,Part>}

def parts_of(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) -> Tensors.Parts<Element>:
  match inputs:
    case Last{part,+model,+shape,+fill,+workers}: Tensors.Single{Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}}
    case More{part,+model,+shape,+fill,+workers,rest}: Tensors.Cons{Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},parts_of(~Element,~Part,rest)}

def first_shape(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) -> List<&2,U32>:
  match inputs:
    case Last{part,model,+shape,fill,workers}: shape
    case More{part,model,+shape,fill,workers,rest}: shape

def first_fill(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) -> Element:
  match inputs:
    case Last{part,model,shape,+fill,workers}: fill
    case More{part,model,shape,+fill,workers,rest}: fill

def logicals_of(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) -> List<&2,Part>:
  match inputs:
    case Last{+part,model,shape,fill,workers}: [part]
    case More{+part,model,shape,fill,workers,rest}: part <> logicals_of(~Element,~Part,rest)

def shapes_of(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) -> List<&2,List<&2,U32>>:
  match inputs:
    case Last{part,model,+shape,fill,workers}: [shape]
    case More{part,model,+shape,fill,workers,rest}: shape <> shapes_of(~Element,~Part,rest)

def models_of(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) -> Moved.Models<Element>:
  match inputs:
    case Last{part,+model,shape,fill,workers}: Moved.Models{model,Moved.NoModels{}}
    case More{part,+model,shape,fill,workers,rest}: Moved.Models{model,models_of(~Element,~Part,rest)}

def shifts_of(shapes: List<&2,List<&2,U32>>,+axis: U32,+start: U32) -> List<&2,List<&2,U32>>:
  match shapes:
    case Con{+shape,shapes}: Layout.shifts_at(List.length(&2,U32,shape),axis,start,0) <> shifts_of(shapes,axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32))
    case Nil{}: []

def segments_of(shapes: List<&2,List<&2,U32>>,+axis: U32,+start: U32) -> List<&2,List<&2,List<&2,Tasks.Segment>>>:
  match shapes:
    case Con{+shape,shapes}:
      Layout.placed_segments(shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0)) <> segments_of(shapes,axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32))
    case Nil{}: []

def clamps_of(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) -> List<&2,Bool>:
  match inputs:
    case Last{part,model,shape,fill,workers}: [False{}]
    case More{part,model,shape,fill,workers,rest}: False{} <> clamps_of(~Element,~Part,rest)

def inputs_count(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) -> Nat:
  match inputs:
    case Last{part,model,shape,fill,workers}: 1n
    case More{part,model,shape,fill,workers,rest}: 1n+inputs_count(~Element,~Part,rest)

# What the runtime survey of the parts finds: the extents fit a word, their
# total along the axis and the most workers.
def survey_valid(shapes: List<&2,List<&2,U32>>,+axis: Nat,valid: Bool,+total: U32) -> Bool:
  match shapes:
    case Con{+shape,shapes}:
      survey_valid(shapes,axis,valid && U32.is_le(Layout.extent_at(shape,axis),(4294967295 - total : U32)),(total + Layout.extent_at(shape,axis) : U32))
    case Nil{}: valid

def survey_total(shapes: List<&2,List<&2,U32>>,+axis: Nat,+total: U32) -> U32:
  match shapes:
    case Con{+shape,shapes}: survey_total(shapes,axis,(total + Layout.extent_at(shape,axis) : U32))
    case Nil{}: total

def survey_workers(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>,+most: U32) -> U32:
  match inputs:
    case Last{part,model,shape,fill,+workers}: U32.max(most,workers)
    case More{part,model,shape,fill,+workers,rest}: survey_workers(~Element,~Part,rest,U32.max(most,workers))

# The extents of the joined tensor.
def joined_shape(~Element: Data,~Part: Data,+inputs: Inputs<Element,Part>,+axis: U32) -> List<&2,U32>:
  Layout.with_extent(first_shape(~Element,~Part,inputs),U32.to_nat(axis),survey_total(shapes_of(~Element,~Part,inputs),U32.to_nat(axis),0))

def fits_of(shapes: List<&2,List<&2,U32>>,+target: List<&2,U32>,+axis: U32,+start: U32,fits: Bool) -> Bool:
  match shapes:
    case Con{+shape,shapes}: fits_of(shapes,target,axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32),fits && Layout.slice_fits(target,shape,axis,start,0))
    case Nil{}: fits

# Every part is a tensor of the contract of its logical.
def InputsHeld(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,inputs: Inputs<Element,Part>) -> Data:
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}: Global.Contract(~Element,~Part,~part_value,part,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers})
    case More{+part,+model,+shape,+fill,+workers,rest}: Held.Both<Global.Contract(~Element,~Part,~part_value,part,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),InputsHeld(~Element,~Part,~part_value,rest)>

# Every part has cells and fits the target from where the parts before it
# end.
def ShapesFit(shapes: List<&2,List<&2,U32>>,+target: List<&2,U32>,+axis: U32,+start: U32) -> Data:
  match shapes:
    case Con{+shape,shapes}:
      Held.Both<{Layout.slice_fits(target,shape,axis,start,0) == True{} : Bool},Held.Both<{Shapes.has_zero(shape) == False{} : Bool},
        ShapesFit(shapes,target,axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32))>>
    case Nil{}: Unit

# ---- What the runtime makes of the parts ----

def lead_parts(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) ->
  {Layout.lead(~Element,parts_of(~Element,~Part,inputs)) == (parts_of(~Element,~Part,inputs),(first_shape(~Element,~Part,inputs),first_fill(~Element,~Part,inputs))) :
    Tensors.Parts<Element> & (List<&2,U32> & Element)}:
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}: {==}
    case More{+part,+model,+shape,+fill,+workers,+rest}: {==}

def survey_parts(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>,+axis: Nat,+valid: Bool,+total: U32,+most: U32) ->
  {Layout.survey(~Element,parts_of(~Element,~Part,inputs),axis,Layout.Survey{valid,total,Tensors.Usable{},most}) ==
    (parts_of(~Element,~Part,inputs),Layout.Survey{survey_valid(shapes_of(~Element,~Part,inputs),axis,valid,total),survey_total(shapes_of(~Element,~Part,inputs),axis,total),Tensors.Usable{},
      survey_workers(~Element,~Part,inputs,most)}) : Tensors.Parts<Element> & Layout.Survey}:
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}: {==}
    case More{+part,+model,+shape,+fill,+workers,+rest}:
      +next_valid = valid && U32.is_le(Layout.extent_at(shape,axis),(4294967295 - total : U32))
      +next_total = (total + Layout.extent_at(shape,axis) : U32)
      +next_most = U32.max(most,workers)
      Equal.cong(Tensors.Parts<Element> & Layout.Survey,Tensors.Parts<Element> & Layout.Survey,result => Layout.prepend_survey(~Element,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},result),
        Layout.survey(~Element,parts_of(~Element,~Part,rest),axis,Layout.Survey{next_valid,next_total,Tensors.Usable{},next_most}),
        (parts_of(~Element,~Part,rest),Layout.Survey{survey_valid(shapes_of(~Element,~Part,rest),axis,next_valid,next_total),survey_total(shapes_of(~Element,~Part,rest),axis,next_total),Tensors.Usable{},
          survey_workers(~Element,~Part,rest,next_most)}),
        survey_parts(~Element,~Part,rest,axis,next_valid,next_total,next_most))

def fitting_parts(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>,+target: List<&2,U32>,+axis: U32,+start: U32,+fits: Bool) ->
  {Layout.fitting(~Element,parts_of(~Element,~Part,inputs),target,axis,start,fits) == (parts_of(~Element,~Part,inputs),fits_of(shapes_of(~Element,~Part,inputs),target,axis,start,fits)) : Tensors.Parts<Element> & Bool}:
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}: {==}
    case More{+part,+model,+shape,+fill,+workers,+rest}:
      +next_fits = fits && Layout.slice_fits(target,shape,axis,start,0)
      Equal.cong(Tensors.Parts<Element> & Bool,Tensors.Parts<Element> & Bool,result => Layout.prepend_fit(~Element,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},result),
        Layout.fitting(~Element,parts_of(~Element,~Part,rest),target,axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32),next_fits),
        (parts_of(~Element,~Part,rest),fits_of(shapes_of(~Element,~Part,rest),target,axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32),next_fits)),
        fitting_parts(~Element,~Part,rest,target,axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32),next_fits))

# Parts that fit pass the check of every part.
def fits_true(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>,+target: List<&2,U32>,+axis: U32,+start: U32,+fit: ShapesFit(shapes_of(~Element,~Part,inputs),target,axis,start)) ->
  {fits_of(shapes_of(~Element,~Part,inputs),target,axis,start,True{}) == True{} : Bool}:
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}:
      Held.Both{+fits,Held.Both{+nonempty,+later}} = fit
      fits
    case More{+part,+model,+shape,+fill,+workers,+rest}:
      Held.Both{+fits,Held.Both{+nonempty,+later}} = fit
      %Equal.sym(Bool,Layout.slice_fits(target,shape,axis,start,0),True{},fits) : {fits_of(shapes_of(~Element,~Part,rest),target,axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32),_) == True{} : Bool}
      fits_true(~Element,~Part,rest,target,axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32),later)

def sources_planned(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>,+axis: U32,+start: U32,+target: List<&2,U32>) ->
  {Layout.concat_sources(~Element,parts_of(~Element,~Part,inputs),axis,start,target,Tensors.grid_of(target)) ==
    (Moved.pack_sources(~Element,models_of(~Element,~Part,inputs)),Layout.Planned{Law.placing_origins(shapes_of(~Element,~Part,inputs)),
      Law.placing_rows(target,shapes_of(~Element,~Part,inputs),shifts_of(shapes_of(~Element,~Part,inputs),axis,start),clamps_of(~Element,~Part,inputs)),
      segments_of(shapes_of(~Element,~Part,inputs),axis,start)}) : Tensors.Sources<Element> & Layout.Planned}:
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}: {==}
    case More{+part,+model,+shape,+fill,+workers,+rest}:
      Equal.cong(Tensors.Sources<Element> & Layout.Planned,Tensors.Sources<Element> & Layout.Planned,later => Layout.planned_part(~Element,PiecesModel.pack(~Element,model),shape,axis,start,target,Tensors.grid_of(target),later),
        Layout.concat_sources(~Element,parts_of(~Element,~Part,rest),axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32),target,Tensors.grid_of(target)),
        (Moved.pack_sources(~Element,models_of(~Element,~Part,rest)),Layout.Planned{Law.placing_origins(shapes_of(~Element,~Part,rest)),
          Law.placing_rows(target,shapes_of(~Element,~Part,rest),shifts_of(shapes_of(~Element,~Part,rest),axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32)),clamps_of(~Element,~Part,rest)),
          segments_of(shapes_of(~Element,~Part,rest),axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32))}),
        sources_planned(~Element,~Part,rest,axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32),target))

# ---- What the parts give the placing ----

def shapes_count(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) -> {List.length(&2,List<&2,U32>,shapes_of(~Element,~Part,inputs)) == inputs_count(~Element,~Part,inputs) : Nat}:
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}: {==}
    case More{+part,+model,+shape,+fill,+workers,+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,List<&2,U32>,shapes_of(~Element,~Part,rest)),inputs_count(~Element,~Part,rest),shapes_count(~Element,~Part,rest))

def logicals_count(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) -> {List.length(&2,Part,logicals_of(~Element,~Part,inputs)) == inputs_count(~Element,~Part,inputs) : Nat}:
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}: {==}
    case More{+part,+model,+shape,+fill,+workers,+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Part,logicals_of(~Element,~Part,rest)),inputs_count(~Element,~Part,rest),logicals_count(~Element,~Part,rest))

def shapes_nonempty(~Element: Data,~Part: Data,inputs: Inputs<Element,Part>) -> {Nat.is_lt(0n,List.length(&2,List<&2,U32>,shapes_of(~Element,~Part,inputs))) == True{} : Bool}:
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}: {==}
    case More{+part,+model,+shape,+fill,+workers,+rest}: {==}

# A part of cells that fits the target has the shape facts and the axes a
# placing needs.
def part_shaped(+limits: Machine.Limits,+target: List<&2,U32>,+shape: List<&2,U32>,+axis: U32,+start: U32,+count: U32,+accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Layout.slice_fits(target,shape,axis,start,0) == True{} : Bool},+nonempty: {Shapes.has_zero(shape) == False{} : Bool}) ->
  Held.Both<Components.ShapeOk(limits,shape),Parts.SourceAxes(Layout.placed_segments(shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0)),target,shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0),False{},Machine.half_word(limits))>:
  +within = slice_in(target,shape,axis,start,0,fits)
  +same = Facts.placed_length(target,shape,Layout.shifts_at(List.length(&2,U32,target),axis,start,0),within)
  +back = Equal.sym(Nat,List.length(&2,U32,shape),List.length(&2,U32,target),same)
  Held.Both{Held.Both{StorageCount.counted_room(limits,shape,Global.Counted{count,accepted},default_capacity),Held.Both{CountCompletion.positive_elements(shape,nonempty),
      {%back : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room : {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}}},
    {%back : Parts.SourceAxes(Layout.placed_segments(shape,Layout.shifts_at(_,axis,start,0)),target,shape,Layout.shifts_at(_,axis,start,0),False{},Machine.half_word(limits))
      Facts.placed_axes(target,shape,Layout.shifts_at(List.length(&2,U32,target),axis,start,0),limits,within,room,positive) :
      Parts.SourceAxes(Layout.placed_segments(shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0)),target,shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0),False{},Machine.half_word(limits))}}

def shaped_cons(+limits: Machine.Limits,+target: List<&2,U32>,+shape: List<&2,U32>,+own: List<&2,List<&2,Tasks.Segment>>,+shift: List<&2,U32>,
  +segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,
  +part: Held.Both<Components.ShapeOk(limits,shape),Parts.SourceAxes(own,target,shape,shift,False{},Machine.half_word(limits))>,
  +later: Facts.SourcesShaped(limits,target,segments,sources,shifts,clamps)) ->
  Facts.SourcesShaped(limits,target,Con{own,segments},Con{shape,sources},Con{shift,shifts},Con{False{},clamps}):
  Held.Both{+shape_ok,+axes} = part
  Held.Both{shape_ok,Held.Both{axes,later}}

# Parts of the contract that fit the target give every source of the placing
# its facts.
def inputs_shaped(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,inputs: Inputs<Element,Part>,+limits: Machine.Limits,+target: List<&2,U32>,+axis: U32,+start: U32,
  +held: InputsHeld(~Element,~Part,~part_value,inputs),+fit: ShapesFit(shapes_of(~Element,~Part,inputs),target,axis,start),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Facts.SourcesShaped(limits,target,segments_of(shapes_of(~Element,~Part,inputs),axis,start),shapes_of(~Element,~Part,inputs),shifts_of(shapes_of(~Element,~Part,inputs),axis,start),clamps_of(~Element,~Part,inputs)):
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}:
      Held.Both{+fits,Held.Both{+nonempty,+later}} = fit
      +counted = Global.contract_counted(~Element,~Part,~part_value,part,PiecesModel.pack(~Element,model),shape,fill,workers,held)
      shaped_cons(limits,target,shape,Layout.placed_segments(shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0)),Layout.shifts_at(List.length(&2,U32,shape),axis,start,0),[],[],[],[],
        part_shaped(limits,target,shape,axis,start,Global.count(shape,counted),Global.accepted(shape,counted),default_capacity,room,positive,rank_room,fits,nonempty),Unit{})
    case More{+part,+model,+shape,+fill,+workers,+rest}:
      Held.Both{+contract,+held_rest} = held
      Held.Both{+fits,Held.Both{+nonempty,+fit_rest}} = fit
      +counted = Global.contract_counted(~Element,~Part,~part_value,part,PiecesModel.pack(~Element,model),shape,fill,workers,contract)
      shaped_cons(limits,target,shape,Layout.placed_segments(shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0)),Layout.shifts_at(List.length(&2,U32,shape),axis,start,0),segments_of(shapes_of(~Element,~Part,rest),axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32)),shapes_of(~Element,~Part,rest),
        shifts_of(shapes_of(~Element,~Part,rest),axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32)),clamps_of(~Element,~Part,rest),
        part_shaped(limits,target,shape,axis,start,Global.count(shape,counted),Global.accepted(shape,counted),default_capacity,room,positive,rank_room,fits,nonempty),
        inputs_shaped(~Element,~Part,~part_value,rest,limits,target,axis,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32),held_rest,fit_rest,default_capacity,room,positive,rank_room))

def placed_cons(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+part: Part,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+logicals: List<&2,Part>,+sources: List<&2,List<&2,U32>>,+models: Moved.Models<Element>,
  +own: Sources.SourceHeld(~Element,~Part,~part_value,part,model,shape),+later: PlacingMoved.SourcesPlaced(~Element,~Part,~part_value,logicals,sources,models)) ->
  PlacingMoved.SourcesPlaced(~Element,~Part,~part_value,Con{part,logicals},Con{shape,sources},Moved.Models{model,models}):
  Held.Both{+laid,+holds} = own
  Held.Both{laid,Held.Both{holds,later}}

# Parts of the contract are laid as the canonical pieces of their shapes.
def inputs_placed(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,inputs: Inputs<Element,Part>,+held: InputsHeld(~Element,~Part,~part_value,inputs)) ->
  PlacingMoved.SourcesPlaced(~Element,~Part,~part_value,logicals_of(~Element,~Part,inputs),shapes_of(~Element,~Part,inputs),models_of(~Element,~Part,inputs)):
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}:
      placed_cons(~Element,~Part,~part_value,part,model,shape,[],[],Moved.NoModels{},Sources.source_held(~Element,~Part,~part_value,part,model,shape,fill,workers,held),Unit{})
    case More{+part,+model,+shape,+fill,+workers,+rest}:
      Held.Both{+contract,+held_rest} = held
      placed_cons(~Element,~Part,~part_value,part,model,shape,logicals_of(~Element,~Part,rest),shapes_of(~Element,~Part,rest),models_of(~Element,~Part,rest),
        Sources.source_held(~Element,~Part,~part_value,part,model,shape,fill,workers,contract),inputs_placed(~Element,~Part,~part_value,rest,held_rest))

# The first part fits the target, so the target has its rank.
def first_rank(~Element: Data,~Part: Data,+inputs: Inputs<Element,Part>,+target: List<&2,U32>,+axis: U32,+fit: ShapesFit(shapes_of(~Element,~Part,inputs),target,axis,0)) ->
  {List.length(&2,U32,first_shape(~Element,~Part,inputs)) == List.length(&2,U32,target) : Nat}:
  match inputs:
    case Last{+part,+model,+shape,+fill,+workers}:
      Held.Both{+fits,Held.Both{+nonempty,+later}} = fit
      Facts.placed_length(target,shape,Layout.shifts_at(List.length(&2,U32,target),axis,0,0),slice_in(target,shape,axis,0,0,fits))
    case More{+part,+model,+shape,+fill,+workers,+rest}:
      Held.Both{+fits,Held.Both{+nonempty,+later}} = fit
      Facts.placed_length(target,shape,Layout.shifts_at(List.length(&2,U32,target),axis,0,0),slice_in(target,shape,axis,0,0,fits))

# Parts of the contract, each with cells, that the runtime accepts along an
# axis and whose joined extents count a nonzero number of cells make a
# ready tensor of the joined extents that holds every part in its place.
def concat(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+limits: Machine.Limits,+inputs: Inputs<Element,Part>,+axis: U32,+held: InputsHeld(~Element,~Part,~part_value,inputs),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,first_shape(~Element,~Part,inputs)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +parts_room: {Nat.is_le(inputs_count(~Element,~Part,inputs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +valid: {survey_valid(shapes_of(~Element,~Part,inputs),U32.to_nat(axis),Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first_shape(~Element,~Part,inputs))),0) == True{} : Bool},
  +fit: ShapesFit(shapes_of(~Element,~Part,inputs),joined_shape(~Element,~Part,inputs,axis),axis,0),
  +nonempty: {Shapes.has_zero(joined_shape(~Element,~Part,inputs,axis)) == False{} : Bool},
  +count: U32,+counted: {Tensors.count_of(joined_shape(~Element,~Part,inputs,axis)) == Some{count} : Maybe<&2,U32>}) ->
  Global.Shaped(~Element,~Gathered.Placings<Part,Element>,~Gathered.placing_value(~Element,~Part,~part_value),
    Gathered.Placings{logicals_of(~Element,~Part,inputs),segments_of(shapes_of(~Element,~Part,inputs),axis,0),shapes_of(~Element,~Part,inputs),joined_shape(~Element,~Part,inputs,axis),
      first_fill(~Element,~Part,inputs)},joined_shape(~Element,~Part,inputs,axis),
    Layout.concat(~Element,parts_of(~Element,~Part,inputs),axis)):
  +first = first_shape(~Element,~Part,inputs)
  +fill = first_fill(~Element,~Part,inputs)
  +ranked = Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first))
  +total = survey_total(shapes_of(~Element,~Part,inputs),U32.to_nat(axis),0)
  +workers = survey_workers(~Element,~Part,inputs,1)
  +target = joined_shape(~Element,~Part,inputs,axis)
  +logicals = logicals_of(~Element,~Part,inputs)
  +segments = segments_of(shapes_of(~Element,~Part,inputs),axis,0)
  +sources = shapes_of(~Element,~Part,inputs)
  +shifts = shifts_of(shapes_of(~Element,~Part,inputs),axis,0)
  +clamps = clamps_of(~Element,~Part,inputs)
  +placings = {Gathered.Placings{logicals,segments,sources,target,fill} : Gathered.Placings<Part,Element>}
  +accepted = {Global.Counted{count,counted} : Global.Counted<target>}
  +room = StorageCount.counted_room(limits,target,accepted,default_capacity)
  +positive = CountCompletion.positive_elements(target,nonempty)
  +target_rank = {%first_rank(~Element,~Part,inputs,target,axis,fit) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room : {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +target_ok = {Held.Both{room,Held.Both{positive,target_rank}} : Components.ShapeOk(limits,target)}
  +count_back = Equal.sym(Nat,List.length(&2,List<&2,U32>,sources),inputs_count(~Element,~Part,inputs),shapes_count(~Element,~Part,inputs))
  +count_room = {%count_back : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    LimitBounds.within_half(limits,inputs_count(~Element,~Part,inputs),parts_room) : {Nat.is_le(List.length(&2,List<&2,U32>,sources),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  +facts = Facts.placing_facts(limits,target,segments,sources,shifts,clamps,target_ok,
    inputs_shaped(~Element,~Part,~part_value,inputs,limits,target,axis,0,held,fit,default_capacity,room,positive,target_rank),shapes_nonempty(~Element,~Part,inputs),count_room)
  +named = Equal.trans(Nat,List.length(&2,Part,logicals),inputs_count(~Element,~Part,inputs),List.length(&2,List<&2,U32>,sources),logicals_count(~Element,~Part,inputs),
    Equal.sym(Nat,List.length(&2,List<&2,U32>,sources),inputs_count(~Element,~Part,inputs),shapes_count(~Element,~Part,inputs)))
  %Equal.sym(Tensors.Parts<Element> & (List<&2,U32> & Element),Layout.lead(~Element,parts_of(~Element,~Part,inputs)),(parts_of(~Element,~Part,inputs),(first,fill)),lead_parts(~Element,~Part,inputs)) :
    Global.Shaped(~Element,~Gathered.Placings<Part,Element>,~Gathered.placing_value(~Element,~Part,~part_value),placings,target,Layout.concat_led(~Element,axis,_))
  %Equal.sym(Tensors.Parts<Element> & Layout.Survey,Layout.survey(~Element,parts_of(~Element,~Part,inputs),U32.to_nat(axis),Layout.Survey{ranked,0,Tensors.Usable{},1}),
      (parts_of(~Element,~Part,inputs),Layout.Survey{survey_valid(shapes_of(~Element,~Part,inputs),U32.to_nat(axis),ranked,0),total,Tensors.Usable{},workers}),survey_parts(~Element,~Part,inputs,U32.to_nat(axis),ranked,0,1)) :
    Global.Shaped(~Element,~Gathered.Placings<Part,Element>,~Gathered.placing_value(~Element,~Part,~part_value),placings,target,Layout.joined(~Element,first,axis,fill,_))
  %Equal.sym(Bool,survey_valid(shapes_of(~Element,~Part,inputs),U32.to_nat(axis),ranked,0),True{},valid) :
    Global.Shaped(~Element,~Gathered.Placings<Part,Element>,~Gathered.placing_value(~Element,~Part,~part_value),placings,target,
      Layout.fitted_status(~Element,Tensors.Usable{},first,target,axis,fill,workers,Layout.fitting(~Element,parts_of(~Element,~Part,inputs),target,axis,0,_)))
  %Equal.sym(Tensors.Parts<Element> & Bool,Layout.fitting(~Element,parts_of(~Element,~Part,inputs),target,axis,0,True{}),(parts_of(~Element,~Part,inputs),fits_of(shapes_of(~Element,~Part,inputs),target,axis,0,True{})),
      fitting_parts(~Element,~Part,inputs,target,axis,0,True{})) :
    Global.Shaped(~Element,~Gathered.Placings<Part,Element>,~Gathered.placing_value(~Element,~Part,~part_value),placings,target,
      Layout.fitted_status(~Element,Tensors.Usable{},first,target,axis,fill,workers,_))
  %Equal.sym(Bool,fits_of(shapes_of(~Element,~Part,inputs),target,axis,0,True{}),True{},fits_true(~Element,~Part,inputs,target,axis,0,fit)) :
    Global.Shaped(~Element,~Gathered.Placings<Part,Element>,~Gathered.placing_value(~Element,~Part,~part_value),placings,target,
      Layout.join_status(~Element,Tensors.Usable{},_,first,target,axis,fill,workers,parts_of(~Element,~Part,inputs)))
  %Equal.sym(Bool,Shapes.has_zero(target),False{},nonempty) :
    Global.Shaped(~Element,~Gathered.Placings<Part,Element>,~Gathered.placing_value(~Element,~Part,~part_value),placings,target,
      Layout.join_shaped(~Element,_,target,axis,fill,workers,parts_of(~Element,~Part,inputs)))
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(target),Some{count},counted) :
    Global.Shaped(~Element,~Gathered.Placings<Part,Element>,~Gathered.placing_value(~Element,~Part,~part_value),placings,target,
      Layout.join_counted(~Element,_,target,axis,fill,workers,parts_of(~Element,~Part,inputs)))
  %Equal.sym(Tensors.Sources<Element> & Layout.Planned,Layout.concat_sources(~Element,parts_of(~Element,~Part,inputs),axis,0,target,Tensors.grid_of(target)),
      (Moved.pack_sources(~Element,models_of(~Element,~Part,inputs)),Layout.Planned{Law.placing_origins(sources),Law.placing_rows(target,sources,shifts,clamps),segments}),
      sources_planned(~Element,~Part,inputs,axis,0,target)) :
    Global.Shaped(~Element,~Gathered.Placings<Part,Element>,~Gathered.placing_value(~Element,~Part,~part_value),placings,target,
      Layout.joined_parts(~Element,target,fill,workers,_))
  Global.shaped(~Element,~Gathered.Placings<Part,Element>,~Gathered.placing_value(~Element,~Part,~part_value),placings,target,
    Tensors.Tensor{Tensors.placed_from(~Element,Law.placing_rows(target,sources,shifts,clamps),segments,Moved.pack_sources(~Element,models_of(~Element,~Part,inputs)),Law.placing_origins(sources),target,fill,workers),target,fill,Tensors.Usable{},workers},
    {==},{==},
    PlacingMoved.placed_contract(~Element,~Part,~part_value,logicals,limits,target,segments,sources,shifts,clamps,fill,workers,models_of(~Element,~Part,inputs),named,facts,
      inputs_placed(~Element,~Part,~part_value,inputs,held),count,counted))
