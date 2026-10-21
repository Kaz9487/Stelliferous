# The public concat caller: the survey of the parts and the band route it
# selects.
import Base
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ../tensor_band.bend as Bands
import ../tensor_band_concat.bend as BandConcat
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ../tensor.bend as Tensors
import ./nat_order.bend as Order
import ./u32_full_range.bend as FullRange
import ../tensor_layout.bend as Layout
import ../tensor_shape.bend as Shapes
import ./boolean_logic.bend as Boolean
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./tensor_shape_refinement.bend as ShapeLaws

# The public survey visits Parts in the same order as band concatenation.
# Its successful overflow and shape guards produce unbounded prefix totals;
# the modular concat cursor is not used as a substitute for those facts.
def survey_valid(summary: Layout.Survey) -> Bool:
  Layout.Survey{survey_valid,total,status,workers} = summary
  survey_valid

def survey_total(summary: Layout.Survey) -> U32:
  Layout.Survey{survey_valid,total,status,workers} = summary
  total

def tensor_status(~Element: Data,input: BandConcatExecution.TensorContent<Element>) -> Tensors.Status:
  BandConcatExecution.TensorContent{bands,geometry,shape,fill,status,workers} = input
  status

def tensor_workers(~Element: Data,input: BandConcatExecution.TensorContent<Element>) -> U32:
  BandConcatExecution.TensorContent{bands,geometry,shape,fill,status,workers} = input
  workers

def step(~Element: Data,+first: List<&2,U32>,+axis: Nat,+input: BandConcatExecution.TensorContent<Element>,summary: Layout.Survey) -> Layout.Survey:
  Layout.surveyed(summary,first,BandConcatExecution.tensor_shape(~Element,input),axis,tensor_status(~Element,input),tensor_workers(~Element,input))

def step_total(~Element: Data,+first: List<&2,U32>,+axis: Nat,input: BandConcatExecution.TensorContent<Element>,summary: Layout.Survey) ->
  {survey_total(step(~Element,first,axis,input,summary)) == (survey_total(summary) + Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) : U32) : U32}:
  BandConcatExecution.TensorContent{bands,geometry,shape,fill,status,workers} = input
  Layout.Survey{survey_valid,total,previous_status,previous_workers} = summary
  {==}

def survey_model(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,summary: Layout.Survey) -> Layout.Survey:
  match parts:
    case BandConcatExecution.SingleContent{input}: step(~Element,first,axis,input,summary)
    case BandConcatExecution.ConsContent{input,rest}: survey_model(~Element,rest,first,axis,step(~Element,first,axis,input,summary))

def extent_total(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+axis: Nat) -> Nat:
  match parts:
    case BandConcatExecution.SingleContent{input}: U32.to_nat(Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis))
    case BandConcatExecution.ConsContent{input,rest}: Nat.add(U32.to_nat(Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis)),extent_total(~Element,rest,axis))

def conditions(+first: List<&2,U32>,+shape: List<&2,U32>,+axis: Nat,summary: Layout.Survey) -> List<&2,Bool>:
  Layout.Survey{previous,+total,status,workers} = summary
  [U32.is_le(Layout.extent_at(shape,axis),(4294967295 - total : U32)),
    Shapes.same_extents(List.drop(&2,U32,first,1n+axis),List.drop(&2,U32,shape,1n+axis)),
    Shapes.same_extents(List.take(&2,U32,first,axis),List.take(&2,U32,shape,axis)),
    Nat.is_eq(List.length(&2,U32,first),List.length(&2,U32,shape)),previous]

def step_conditions(~Element: Data,+first: List<&2,U32>,+axis: Nat,input: BandConcatExecution.TensorContent<Element>,summary: Layout.Survey) ->
  {survey_valid(step(~Element,first,axis,input,summary)) == Boolean.all(conditions(first,BandConcatExecution.tensor_shape(~Element,input),axis,summary)) : Bool}:
  BandConcatExecution.TensorContent{bands,geometry,shape,fill,status,workers} = input
  Layout.Survey{previous,total,old_status,old_workers} = summary
  {==}

def step_previous(~Element: Data,+first: List<&2,U32>,+axis: Nat,+input: BandConcatExecution.TensorContent<Element>,+summary: Layout.Survey,
  accepted: {survey_valid(step(~Element,first,axis,input,summary)) == True{} : Bool}) -> {survey_valid(summary) == True{} : Bool}:
  Layout.Survey{+previous,+total,+status,+workers} = summary
  Boolean.every_condition_holds(conditions(first,BandConcatExecution.tensor_shape(~Element,input),axis,summary),4n,
    %step_conditions(~Element,first,axis,input,summary) : {_ == True{} : Bool} accepted)

def previous_valid(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,
  accepted: {survey_valid(survey_model(~Element,parts,first,axis,summary)) == True{} : Bool}) -> {survey_valid(summary) == True{} : Bool}:
  match parts:
    case BandConcatExecution.SingleContent{input}: step_previous(~Element,first,axis,input,summary,accepted)
    case BandConcatExecution.ConsContent{+input,+rest}:
      step_previous(~Element,first,axis,input,summary,previous_valid(~Element,rest,first,axis,step(~Element,first,axis,input,summary),accepted))

# Abstract limit keeps the general addition proof independent of a concrete
# enormous unary Nat. The actual public guard is transported by maximum.
def sum_room(+previous: U32,+extent: U32,+limit: U32,
  previous_room: {U32.is_le(previous,limit) == True{} : Bool},
  accepted: {U32.is_le(extent,(limit - previous : U32)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(extent)),U32.to_nat(limit)) == True{} : Bool}:
  +ordered = Bounds.as_nat_le(previous,limit,previous_room)
  +difference = FullRange.subtract(limit,previous,limit,ordered,Order.sub_le(U32.to_nat(limit),U32.to_nat(previous)))
  +extent_room = {
    %difference : {Nat.is_le(U32.to_nat(extent),_) == True{} : Bool}
    Bounds.as_nat_le(extent,(limit - previous : U32),accepted) : {Nat.is_le(U32.to_nat(extent),Nat.sub(U32.to_nat(limit),U32.to_nat(previous))) == True{} : Bool}}
  %Order.restore_shift(U32.to_nat(previous),U32.to_nat(limit),ordered) : {Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(extent)),_) == True{} : Bool}
  Order.offset_le(U32.to_nat(previous),U32.to_nat(extent),Nat.sub(U32.to_nat(limit),U32.to_nat(previous)),extent_room)

type StepFields<-first: List<&2,U32>,-shape: List<&2,U32>,-axis: Nat,-previous: U32,-limit: U32> is Data:
  StepFields{rank: {Nat.is_eq(List.length(&2,U32,first),List.length(&2,U32,shape)) == True{} : Bool},
    prefix: {Shapes.same_extents(List.take(&2,U32,first,axis),List.take(&2,U32,shape,axis)) == True{} : Bool},
    suffix: {Shapes.same_extents(List.drop(&2,U32,first,1n+axis),List.drop(&2,U32,shape,1n+axis)) == True{} : Bool},
    sum_room: {Nat.is_le(Nat.add(U32.to_nat(previous),U32.to_nat(Layout.extent_at(shape,axis))),U32.to_nat(limit)) == True{} : Bool},
    exact: {U32.to_nat((previous + Layout.extent_at(shape,axis) : U32)) == Nat.add(U32.to_nat(previous),U32.to_nat(Layout.extent_at(shape,axis))) : Nat}}

def step_fields(~Element: Data,+first: List<&2,U32>,+axis: Nat,+input: BandConcatExecution.TensorContent<Element>,+summary: Layout.Survey,+limit: U32,
  +maximum: {limit == 4294967295 : U32},accepted: {survey_valid(step(~Element,first,axis,input,summary)) == True{} : Bool}) ->
  StepFields<first,BandConcatExecution.tensor_shape(~Element,input),axis,survey_total(summary),limit>:
  Layout.Survey{+previous,+total,+status,+workers} = summary
  +shape = BandConcatExecution.tensor_shape(~Element,input)
  +holds = {
    %step_conditions(~Element,first,axis,input,summary) : {_ == True{} : Bool}
    accepted : {Boolean.all(conditions(first,shape,axis,summary)) == True{} : Bool}}
  +guard = Boolean.every_condition_holds(conditions(first,shape,axis,summary),0n,holds)
  +room = sum_room(total,Layout.extent_at(shape,axis),limit,
    %Equal.sym(U32,limit,4294967295,maximum) : {U32.is_le(total,_) == True{} : Bool} FullRange.maximum(total),
    %Equal.sym(U32,limit,4294967295,maximum) : {U32.is_le(Layout.extent_at(shape,axis),(_ - total : U32)) == True{} : Bool} guard)
  StepFields{Boolean.every_condition_holds(conditions(first,shape,axis,summary),3n,holds),Boolean.every_condition_holds(conditions(first,shape,axis,summary),2n,holds),
    Boolean.every_condition_holds(conditions(first,shape,axis,summary),1n,holds),room,FullRange.add(total,Layout.extent_at(shape,axis),limit,room)}

def Prefixes(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32) -> Data:
  match parts:
    case BandConcatExecution.SingleContent{input}: StepFields<first,BandConcatExecution.tensor_shape(~Element,input),axis,survey_total(summary),limit>
    case BandConcatExecution.ConsContent{+input,rest}:
      Shape.Both<StepFields<first,BandConcatExecution.tensor_shape(~Element,input),axis,survey_total(summary),limit>,Prefixes(~Element,rest,first,axis,step(~Element,first,axis,input,summary),limit)>

def prefixes(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32,+maximum: {limit == 4294967295 : U32},
  +accepted: {survey_valid(survey_model(~Element,parts,first,axis,summary)) == True{} : Bool}) -> Prefixes(~Element,parts,first,axis,summary,limit):
  match parts:
    case BandConcatExecution.SingleContent{input}: step_fields(~Element,first,axis,input,summary,limit,maximum,accepted)
    case BandConcatExecution.ConsContent{+input,+rest}:
      Shape.Both{step_fields(~Element,first,axis,input,summary,limit,maximum,previous_valid(~Element,rest,first,axis,step(~Element,first,axis,input,summary),accepted)),
        prefixes(~Element,rest,first,axis,step(~Element,first,axis,input,summary),limit,maximum,accepted)}

type TotalFields<-value: U32,-previous: U32,-extent: Nat,-limit: U32> is Data:
  TotalFields{exact: {U32.to_nat(value) == Nat.add(U32.to_nat(previous),extent) : Nat},
    room: {Nat.is_le(Nat.add(U32.to_nat(previous),extent),U32.to_nat(limit)) == True{} : Bool}}

def total_cons(~Element: Data,+input: BandConcatExecution.TensorContent<Element>,+rest: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32,
  head: StepFields<first,BandConcatExecution.tensor_shape(~Element,input),axis,survey_total(summary),limit>,
  remainder: TotalFields<survey_total(survey_model(~Element,rest,first,axis,step(~Element,first,axis,input,summary))),survey_total(step(~Element,first,axis,input,summary)),extent_total(~Element,rest,axis),limit>) ->
  TotalFields<survey_total(survey_model(~Element,BandConcatExecution.ConsContent{input,rest},first,axis,summary)),survey_total(summary),extent_total(~Element,BandConcatExecution.ConsContent{input,rest},axis),limit>:
  Layout.Survey{+previous,+total,+status,+workers} = summary
  StepFields{rank,prefix,suffix,head_room,+head_exact} = head
  TotalFields{+tail_exact,+tail_room} = remainder
  +extent = U32.to_nat(Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis))
  +next_exact = Equal.trans(Nat,U32.to_nat(survey_total(step(~Element,first,axis,input,summary))),U32.to_nat((total + Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) : U32)),Nat.add(U32.to_nat(total),extent),
    Equal.cong(U32,Nat,value => U32.to_nat(value),survey_total(step(~Element,first,axis,input,summary)),(total + Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) : U32),step_total(~Element,first,axis,input,summary)),head_exact)
  TotalFields{
    Equal.trans(Nat,U32.to_nat(survey_total(survey_model(~Element,rest,first,axis,step(~Element,first,axis,input,summary)))),
      Nat.add(U32.to_nat(survey_total(step(~Element,first,axis,input,summary))),extent_total(~Element,rest,axis)),Nat.add(U32.to_nat(total),Nat.add(extent,extent_total(~Element,rest,axis))),tail_exact,
      Equal.trans(Nat,Nat.add(U32.to_nat(survey_total(step(~Element,first,axis,input,summary))),extent_total(~Element,rest,axis)),Nat.add(Nat.add(U32.to_nat(total),extent),extent_total(~Element,rest,axis)),Nat.add(U32.to_nat(total),Nat.add(extent,extent_total(~Element,rest,axis))),
        Equal.cong(Nat,Nat,value => Nat.add(value,extent_total(~Element,rest,axis)),U32.to_nat(survey_total(step(~Element,first,axis,input,summary))),Nat.add(U32.to_nat(total),extent),next_exact),Addition.associative(U32.to_nat(total),extent,extent_total(~Element,rest,axis)))),
    %Addition.associative(U32.to_nat(total),extent,extent_total(~Element,rest,axis)) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
    %next_exact : {Nat.is_le(Nat.add(_,extent_total(~Element,rest,axis)),U32.to_nat(limit)) == True{} : Bool}
    tail_room}

def total_fields(~Element: Data,parts: BandConcatExecution.PartContents<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,+limit: U32,
  fields: Prefixes(~Element,parts,first,axis,summary,limit)) -> TotalFields<survey_total(survey_model(~Element,parts,first,axis,summary)),survey_total(summary),extent_total(~Element,parts,axis),limit>:
  match parts:
    case BandConcatExecution.SingleContent{+input}:
      Layout.Survey{+previous,+total,+status,+workers} = summary
      StepFields{rank,prefix,suffix,room,exact} = fields
      TotalFields{
        Equal.trans(Nat,U32.to_nat(survey_total(step(~Element,first,axis,input,summary))),U32.to_nat((total + Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) : U32)),
          Nat.add(U32.to_nat(total),U32.to_nat(Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis))),
          Equal.cong(U32,Nat,value => U32.to_nat(value),survey_total(step(~Element,first,axis,input,summary)),(total + Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),axis) : U32),step_total(~Element,first,axis,input,summary)),exact),room}
    case BandConcatExecution.ConsContent{+input,+rest}:
      Shape.Both{head,tail} = fields
      total_cons(~Element,input,rest,first,axis,summary,limit,head,total_fields(~Element,rest,first,axis,step(~Element,first,axis,input,summary),limit,tail))

def survey_contents(~Element: Data,owners: Tensors.Parts<Element> & Layout.Survey) -> BandConcatExecution.PartContents<Element> & Layout.Survey:
  (parts,summary) = owners
  (BandConcatExecution.parts_contents(~Element,parts),summary)

def observed_summary(~Element: Data,owners: Tensors.Parts<Element> & Layout.Survey) -> Layout.Survey:
  (parts,summary) = owners
  summary

def summary_projection(~Element: Data,owners: BandConcatExecution.PartContents<Element> & Layout.Survey) -> Layout.Survey:
  (parts,summary) = owners
  summary

def summary_contents(~Element: Data,owners: Tensors.Parts<Element> & Layout.Survey) ->
  {observed_summary(~Element,owners) == summary_projection(~Element,survey_contents(~Element,owners)) : Layout.Survey}:
  (parts,summary) = owners
  {==}

def prepend_model(~Element: Data,+input: BandConcatExecution.TensorContent<Element>,owners: BandConcatExecution.PartContents<Element> & Layout.Survey) -> BandConcatExecution.PartContents<Element> & Layout.Survey:
  (rest,summary) = owners
  (BandConcatExecution.ConsContent{input,rest},summary)

def prepend_projection(~Element: Data,input: Tensors.Tensor<Element>,owners: Tensors.Parts<Element> & Layout.Survey) ->
  {survey_contents(~Element,Layout.prepend_survey(~Element,input,owners)) == prepend_model(~Element,BandConcatExecution.tensor_contents(~Element,input),survey_contents(~Element,owners)) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  (rest,summary) = owners
  {==}

def single_survey(~Element: Data,input: Tensors.Tensor<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {survey_contents(~Element,Layout.survey(~Element,Tensors.Single{input},first,axis,summary)) ==
    (BandConcatExecution.SingleContent{BandConcatExecution.tensor_contents(~Element,input)},step(~Element,first,axis,BandConcatExecution.tensor_contents(~Element,input),summary)) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  Tensors.Tensor{bands,geometry,shape,fill,status,workers} = input
  {==}

def continued_survey(~Element: Data,+input: BandConcatExecution.TensorOwnerModel<Element>,+rest: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,
  rule: @+next: Layout.Survey -> {survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,next)) ==
    (BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,next)) : BandConcatExecution.PartContents<Element> & Layout.Survey}) ->
  {survey_contents(~Element,Layout.prepend_survey(~Element,BandConcatExecution.tensor_owner_pack(~Element,input),Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,summary))) ==
    (BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))},survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,summary)) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  Equal.trans(BandConcatExecution.PartContents<Element> & Layout.Survey,
    survey_contents(~Element,Layout.prepend_survey(~Element,BandConcatExecution.tensor_owner_pack(~Element,input),Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,summary))),
    prepend_model(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,summary))),
    prepend_model(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,summary))),
    prepend_projection(~Element,BandConcatExecution.tensor_owner_pack(~Element,input),Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,summary)),
    Equal.cong(BandConcatExecution.PartContents<Element> & Layout.Survey,BandConcatExecution.PartContents<Element> & Layout.Survey,owners => prepend_model(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),owners),
      survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,summary)),(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,summary)),rule(summary)))

def cons_survey(~Element: Data,input: BandConcatExecution.TensorOwnerModel<Element>,+rest: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,
  rule: @+next: Layout.Survey -> {survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,next)) ==
    (BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,next)) : BandConcatExecution.PartContents<Element> & Layout.Survey}) ->
  {survey_contents(~Element,Layout.survey(~Element,Tensors.Cons{BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.parts_owner_pack(~Element,rest)},first,axis,summary)) ==
    (BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest))},survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),first,axis,step(~Element,first,axis,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,input)),summary))) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  BandConcatExecution.TensorOwner{bands,geometry,+shape,fill,+status,+workers} = input
  continued_survey(~Element,BandConcatExecution.TensorOwner{bands,geometry,shape,fill,status,workers},rest,first,axis,Layout.surveyed(summary,first,shape,axis,status,workers),rule)

def native_survey_packed(~Element: Data,parts: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary)) ==
    (BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary)) : BandConcatExecution.PartContents<Element> & Layout.Survey}:
  match parts:
    case BandConcatExecution.SingleOwner{input}: single_survey(~Element,BandConcatExecution.tensor_owner_pack(~Element,input),first,axis,summary)
    case BandConcatExecution.ConsOwner{input,+rest}: cons_survey(~Element,input,rest,first,axis,summary,next => native_survey_packed(~Element,rest,first,axis,next))

def native_summary_packed(~Element: Data,+parts: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary)) == survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary) : Layout.Survey}:
  Equal.trans(Layout.Survey,observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary)),summary_projection(~Element,survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary))),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary),
    summary_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary)),
    Equal.cong(BandConcatExecution.PartContents<Element> & Layout.Survey,Layout.Survey,owners => summary_projection(~Element,owners),
      survey_contents(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,parts),first,axis,summary)),(BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,parts)),first,axis,summary)),native_survey_packed(~Element,parts,first,axis,summary)))

def initial_summary(+first: List<&2,U32>,axis: Nat) -> Layout.Survey:
  Layout.Survey{Nat.is_lt(axis,List.length(&2,U32,first)),0,Tensors.Usable{},1}

def status_of(summary: Layout.Survey) -> Tensors.Status:
  Layout.Survey{valid,total,status,workers} = summary
  status

def target_shape(+first: List<&2,U32>,+axis: U32,summary: Layout.Survey) -> List<&2,U32>:
  Layout.with_extent(first,U32.to_nat(axis),survey_total(summary))

type BandSurveyFields<-first: List<&2,U32>,-axis: U32,-summary: Layout.Survey,-count: U32> is Data:
  BandSurveyFields{valid: {survey_valid(summary) == True{} : Bool},status: {status_of(summary) == Tensors.Usable{} : Tensors.Status},
    counted: {Tensors.count_of(target_shape(first,axis,summary)) == Some{count} : Maybe<&2,U32>},
    nonempty: {Shapes.has_zero(target_shape(first,axis,summary)) == False{} : Bool}}

def tensor_fill(~Element: Data,input: BandConcatExecution.TensorContent<Element>) -> Element:
  BandConcatExecution.TensorContent{bands,geometry,shape,fill,status,workers} = input
  fill

# ---- The band route of the concat, on the tensors themselves ----

def survey_workers(summary: Layout.Survey) -> U32:
  Layout.Survey{survey_valid,total,status,workers} = summary
  workers

def summary_split(summary: Layout.Survey) ->
  {summary == Layout.Survey{survey_valid(summary),survey_total(summary),status_of(summary),survey_workers(summary)} : Layout.Survey}:
  Layout.Survey{valid,total,status,workers} = summary
  {==}

def surveyed_tensors(~Element: Data,owners: Tensors.Parts<Element> & Layout.Survey) -> Tensors.Parts<Element>:
  (parts,summary) = owners
  parts

def survey_split(~Element: Data,owners: Tensors.Parts<Element> & Layout.Survey) ->
  {owners == (surveyed_tensors(~Element,owners),observed_summary(~Element,owners)) : Tensors.Parts<Element> & Layout.Survey}:
  (parts,summary) = owners
  {==}

# The survey returns the parts it was given.
def survey_cons(~Element: Data,input: BandConcatExecution.TensorOwnerModel<Element>,+rest: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey,
  rule: @+next: Layout.Survey -> {surveyed_tensors(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,next)) ==
    BandConcatExecution.parts_owner_pack(~Element,rest) : Tensors.Parts<Element>}) ->
  {surveyed_tensors(~Element,Layout.survey(~Element,Tensors.Cons{BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.parts_owner_pack(~Element,rest)},first,axis,summary)) ==
    Tensors.Cons{BandConcatExecution.tensor_owner_pack(~Element,input),BandConcatExecution.parts_owner_pack(~Element,rest)} : Tensors.Parts<Element>}:
  BandConcatExecution.TensorOwner{+bands,+geometry,+shape,+fill,+status,+workers} = input
  +next = Layout.surveyed(summary,first,shape,axis,status,workers)
  %Equal.sym(Tensors.Parts<Element> & Layout.Survey,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,next),
    (surveyed_tensors(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,next)),
      observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,next))),
    survey_split(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,next))) :
    {surveyed_tensors(~Element,Layout.prepend_survey(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,status,workers},_)) ==
      Tensors.Cons{Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,status,workers},BandConcatExecution.parts_owner_pack(~Element,rest)} : Tensors.Parts<Element>}
  Equal.cong(Tensors.Parts<Element>,Tensors.Parts<Element>,others => Tensors.Cons{Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,status,workers},others},
    surveyed_tensors(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,rest),first,axis,next)),BandConcatExecution.parts_owner_pack(~Element,rest),rule(next))

def survey_owned(~Element: Data,sources: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: Nat,+summary: Layout.Survey) ->
  {surveyed_tensors(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),first,axis,summary)) == BandConcatExecution.parts_owner_pack(~Element,sources) : Tensors.Parts<Element>}:
  match sources:
    case BandConcatExecution.SingleOwner{BandConcatExecution.TensorOwner{bands,geometry,shape,fill,status,workers}}: {==}
    case BandConcatExecution.ConsOwner{input,+rest}:
      survey_cons(~Element,input,rest,first,axis,summary,next => survey_owned(~Element,rest,first,axis,next))

# A nonempty target with a count meets the parts in one plan of its joining.
def joined_route(~Element: Data,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,+count: U32,parts: Tensors.Parts<Element>,
  nonempty: {Shapes.has_zero(shape) == False{} : Bool},count_selected: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}) ->
  {Layout.join_shaped(~Element,Shapes.has_zero(shape),shape,axis,fill,workers,parts) ==
    BandConcat.met(~Element,~Layout.band_joined(~Element),BandConcat.joining(shape,axis),shape,fill,workers,BandConcat.offers(~Element,BandConcat.joining(shape,axis),parts)) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Shapes.has_zero(shape),False{},nonempty) :
    {Layout.join_shaped(~Element,_,shape,axis,fill,workers,parts) ==
      BandConcat.met(~Element,~Layout.band_joined(~Element),BandConcat.joining(shape,axis),shape,fill,workers,BandConcat.offers(~Element,BandConcat.joining(shape,axis),parts)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},count_selected) :
    {Layout.join_counted(~Element,_,shape,axis,fill,workers,parts) ==
      BandConcat.met(~Element,~Layout.band_joined(~Element),BandConcat.joining(shape,axis),shape,fill,workers,BandConcat.offers(~Element,BandConcat.joining(shape,axis),parts)) : Tensors.Tensor<Element>}
  {==}

def band_met(~Element: Data,+joining: BandConcat.Joining,+sources: BandConcatExecution.PartsOwnerModel<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  Tensors.Tensor<Element>:
  BandConcat.met(~Element,~Layout.band_joined(~Element),joining,shape,fill,workers,BandConcat.offers(~Element,joining,BandConcatExecution.parts_owner_pack(~Element,sources)))

def surveyed_summary(~Element: Data,sources: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: U32) -> Layout.Survey:
  survey_model(~Element,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),first,U32.to_nat(axis),initial_summary(first,U32.to_nat(axis)))

# The concat of the parts joins them by the status and the validity of their
# survey.
def surveyed_route(~Element: Data,+sources: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: U32,+fill: Element) ->
  {Layout.concat_first(~Element,axis,BandConcatExecution.parts_owner_pack(~Element,sources),first,fill) ==
    Layout.join_status(~Element,status_of(surveyed_summary(~Element,sources,first,axis)),survey_valid(surveyed_summary(~Element,sources,first,axis)),first,
      target_shape(first,axis,surveyed_summary(~Element,sources,first,axis)),axis,fill,survey_workers(surveyed_summary(~Element,sources,first,axis)),
      BandConcatExecution.parts_owner_pack(~Element,sources)) : Tensors.Tensor<Element>}:
  +initial = initial_summary(first,U32.to_nat(axis))
  +summary = surveyed_summary(~Element,sources,first,axis)
  +shape = target_shape(first,axis,summary)
  +workers = survey_workers(summary)
  %Equal.sym(Tensors.Parts<Element> & Layout.Survey,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),first,U32.to_nat(axis),initial),(surveyed_tensors(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),first,U32.to_nat(axis),initial)),observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),first,U32.to_nat(axis),initial))),survey_split(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),first,U32.to_nat(axis),initial))) :
    {Layout.joined(~Element,first,axis,fill,_) == Layout.join_status(~Element,status_of(summary),survey_valid(summary),first,shape,axis,fill,workers,BandConcatExecution.parts_owner_pack(~Element,sources)) : Tensors.Tensor<Element>}
  %Equal.sym(Tensors.Parts<Element>,surveyed_tensors(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),first,U32.to_nat(axis),initial)),BandConcatExecution.parts_owner_pack(~Element,sources),survey_owned(~Element,sources,first,U32.to_nat(axis),initial)) :
    {Layout.joined(~Element,first,axis,fill,(_,observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),first,U32.to_nat(axis),initial)))) == Layout.join_status(~Element,status_of(summary),survey_valid(summary),first,shape,axis,fill,workers,BandConcatExecution.parts_owner_pack(~Element,sources)) : Tensors.Tensor<Element>}
  %Equal.sym(Layout.Survey,observed_summary(~Element,Layout.survey(~Element,BandConcatExecution.parts_owner_pack(~Element,sources),first,U32.to_nat(axis),initial)),summary,native_summary_packed(~Element,sources,first,U32.to_nat(axis),initial)) :
    {Layout.joined(~Element,first,axis,fill,(BandConcatExecution.parts_owner_pack(~Element,sources),_)) == Layout.join_status(~Element,status_of(summary),survey_valid(summary),first,shape,axis,fill,workers,BandConcatExecution.parts_owner_pack(~Element,sources)) : Tensors.Tensor<Element>}
  %Equal.sym(Layout.Survey,summary,Layout.Survey{survey_valid(summary),survey_total(summary),status_of(summary),workers},summary_split(summary)) :
    {Layout.joined(~Element,first,axis,fill,(BandConcatExecution.parts_owner_pack(~Element,sources),_)) == Layout.join_status(~Element,status_of(summary),survey_valid(summary),first,shape,axis,fill,workers,BandConcatExecution.parts_owner_pack(~Element,sources)) : Tensors.Tensor<Element>}
  {==}

# A survey of the parts with a nonempty target and a count meets the parts in
# one plan of the joining of the target.
def concat_band_route(~Element: Data,+sources: BandConcatExecution.PartsOwnerModel<Element>,+first: List<&2,U32>,+axis: U32,+fill: Element,+count: U32,
  evidence: BandSurveyFields<first,axis,surveyed_summary(~Element,sources,first,axis),count>) ->
  {Layout.concat_first(~Element,axis,BandConcatExecution.parts_owner_pack(~Element,sources),first,fill) ==
    band_met(~Element,BandConcat.joining(target_shape(first,axis,surveyed_summary(~Element,sources,first,axis)),axis),sources,
      target_shape(first,axis,surveyed_summary(~Element,sources,first,axis)),fill,survey_workers(surveyed_summary(~Element,sources,first,axis))) : Tensors.Tensor<Element>}:
  BandSurveyFields{+valid,+status,+count_selected,+nonempty} = evidence
  +summary = surveyed_summary(~Element,sources,first,axis)
  +shape = target_shape(first,axis,summary)
  +workers = survey_workers(summary)
  +joining = BandConcat.joining(shape,axis)
  Equal.trans(Tensors.Tensor<Element>,Layout.concat_first(~Element,axis,BandConcatExecution.parts_owner_pack(~Element,sources),first,fill),
    Layout.join_status(~Element,status_of(summary),survey_valid(summary),first,shape,axis,fill,workers,BandConcatExecution.parts_owner_pack(~Element,sources)),
    band_met(~Element,joining,sources,shape,fill,workers),surveyed_route(~Element,sources,first,axis,fill),
    %Equal.sym(Tensors.Status,status_of(summary),Tensors.Usable{},status) :
      {Layout.join_status(~Element,_,survey_valid(summary),first,shape,axis,fill,workers,BandConcatExecution.parts_owner_pack(~Element,sources)) ==
        band_met(~Element,joining,sources,shape,fill,workers) : Tensors.Tensor<Element>}
    %Equal.sym(Bool,survey_valid(summary),True{},valid) :
      {Layout.join_status(~Element,Tensors.Usable{},_,first,shape,axis,fill,workers,BandConcatExecution.parts_owner_pack(~Element,sources)) ==
        band_met(~Element,joining,sources,shape,fill,workers) : Tensors.Tensor<Element>}
    joined_route(~Element,shape,axis,fill,workers,count,BandConcatExecution.parts_owner_pack(~Element,sources),nonempty,count_selected))
