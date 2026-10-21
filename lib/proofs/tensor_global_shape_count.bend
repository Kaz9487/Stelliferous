# Normal usable tensors retain their actual public shape count. Per-band
# metadata alone does not imply this whole-tensor machine envelope. A failed
# tensor carries no layout guarantee.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_window.bend as Window
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as Metadata
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_count.bend as BandCount
import ./tensor_band_each.bend as Each
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_map.bend as BandMap
import ./tensor_band_values.bend as BandValues
import ./tensor_contract.bend as Contract
import ./machine_limit_bounds.bend as Limits
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./natural_addition.bend as Addition
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition

type Counted<-shape: List<&2,U32>> is Data:
  Counted{count: U32,accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}}

# The bands hold as many elements as the shape names; how the geometry folds
# the axes of the shape is up to the operation that made it.
def Coherent(shape: List<&2,U32>,geometry: Bands.Geometry) -> Data:
  {ShapeLaws.elements(shape) == BandGather.elements(geometry) : Nat}

def Normal(~Element: Data,tensor: Tensors.Tensor<Element>) -> Data:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}:
      Shape.Both<Metadata.WellFormed<Element,geometry,Model.contents(~Element,bands)>,Shape.Both<Counted<shape>,Coherent(shape,geometry)>>
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}: Unit

# A Normal tensor whose cells hold the reader's values at their flat index.
def Contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,-tensor: Tensors.Tensor<Element>) -> Data:
  Shape.Both<Normal(~Element,tensor),BandMap.LogicalValues(~Element,~Logical,~original,logical,tensor)>

# Well formed bands of the geometry that hold the reader's values, under a
# shape of as many elements, make a tensor of the contract.
def contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  -bands: Bands.Bands(~Element),counted: Counted<shape>,coherent: Coherent(shape,geometry),
  evidence: Shape.Both<BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.contents(~Element,bands)),Metadata.WellFormed<Element,geometry,Model.contents(~Element,bands)>>) ->
  Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}):
  Shape.Both{values,well_formed} = evidence
  Shape.Both{Shape.Both{well_formed,Shape.Both{counted,coherent}},values}

# A tensor of the contract holds the values of another reader at a related
# context when the two readers agree at related contexts.
def relabeled(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  ~agreement: BandValues.Agreement(~Element,~Before,~After,~before_value,~after_value,~Related),+before: Before,+after: After,+related: Related(before,after),
  tensor: Tensors.Tensor<Element>,contract: Contract(~Element,~Before,~before_value,before,tensor)) ->
  Contract(~Element,~After,~after_value,after,tensor):
  match tensor:
    case Tensors.Tensor{bands,+geometry,shape,fill,status,workers}:
      Shape.Both{normal,values} = contract
      Shape.Both{normal,
        BandValues.values(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,Model.contents(~Element,bands),values)}

# A worker count leaves the contract of a tensor as it is.
def with_workers(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,tensor: Tensors.Tensor<Element>,+count: U32,
  accepted: {Contract.tensor_ready(~Element,Tensors.with_workers(~Element,tensor,count)) == True{} : Bool},
  contract: @ready: {Contract.tensor_ready(~Element,tensor) == True{} : Bool} -> Contract(~Element,~Logical,~original,logical,tensor)) ->
  Contract(~Element,~Logical,~original,logical,Tensors.with_workers(~Element,tensor,count)):
  match tensor:
    case Tensors.Tensor{bands,+geometry,+shape,+fill,status,+workers}:
      match status:
        case Tensors.Usable{}: contract(accepted)
        case Tensors.Failed{+problem}: Empty.absurd(Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},U32.max(1,count)}),WordEncoding.false_true(accepted))

# A ready tensor is a usable tensor of the bands of a model.
type Usable<-Element: Data,-shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>> is Data:
  Usable{bands: Model.Model<Element>,geometry: Bands.Geometry,fill: Element,workers: U32,
    equation: {Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers} == tensor : Tensors.Tensor<Element>}}

def usable_status(~Element: Data,+model: Model.Model<Element>,-bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,status: Tensors.Status,+workers: U32,
  equation: {Model.pack(~Element,model) == bands : Bands.Bands(~Element)},
  ready: {Contract.tensor_ready(~Element,Tensors.Tensor{bands,geometry,shape,fill,status,workers}) == True{} : Bool}) ->
  Usable<Element,shape,Tensors.Tensor{bands,geometry,shape,fill,status,workers}>:
  match status:
    case Tensors.Usable{}:
      Usable{model,geometry,fill,workers,
        Equal.cong(Bands.Bands(~Element),Tensors.Tensor<Element>,owner => Tensors.Tensor{owner,geometry,shape,fill,Tensors.Usable{},workers},Model.pack(~Element,model),bands,equation)}
    case Tensors.Failed{problem}:
      Empty.absurd(Usable<Element,shape,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}>,WordEncoding.false_true(ready))

def usable_observed(~Element: Data,-bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  observation: Model.Observation<Element,bands>,ready: {Contract.tensor_ready(~Element,Tensors.Tensor{bands,geometry,shape,fill,status,workers}) == True{} : Bool}) ->
  Usable<Element,shape,Tensors.Tensor{bands,geometry,shape,fill,status,workers}>:
  Model.Observation{+model,equation} = observation
  usable_status(~Element,model,bands,geometry,shape,fill,status,workers,equation,ready)

def usable(~Element: Data,tensor: Tensors.Tensor<Element>,ready: {Contract.tensor_ready(~Element,tensor) == True{} : Bool}) -> Usable<Element,Contract.extents(~Element,tensor),tensor>:
  Tensors.Tensor{bands,+geometry,+shape,+fill,+status,+workers} = tensor
  usable_observed(~Element,bands,geometry,shape,fill,status,workers,Model.observe(~Element,bands),ready)

# A Ready tensor of the bands of a model in a shape, of the contract of a
# reader.
def Shaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>) -> Data:
  Shape.Both<Usable<Element,shape,tensor>,Contract(~Element,~Logical,~original,logical,tensor)>

def shaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,tensor: Tensors.Tensor<Element>,
  ready: {Contract.tensor_ready(~Element,tensor) == True{} : Bool},extents: {Contract.extents(~Element,tensor) == shape : List<&2,U32>},
  contract: Contract(~Element,~Logical,~original,logical,tensor)) -> Shaped(~Element,~Logical,~original,logical,shape,tensor):
  Shape.Both{%extents : Usable<Element,_,tensor> usable(~Element,tensor,ready),contract}

# Two contexts, each the given one.
def Fixed(~Before: Data,~After: Data,+before: Before,+after: After,left: Before,right: After) -> Data:
  Shape.Both<{left == before : Before},{right == after : After}>

def fixed_agrees(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,+before: Before,+after: After,
  agrees: @+index: U32 -> {before_value(before,index) == after_value(after,index) : Element},
  +left: Before,+right: After,+related: Fixed(~Before,~After,before,after,left,right),+index: U32) ->
  {before_value(left,index) == after_value(right,index) : Element}:
  Shape.Both{+left_equation,+right_equation} = related
  %Equal.sym(Before,left,before,left_equation) : {before_value(_,index) == after_value(right,index) : Element}
  %Equal.sym(After,right,after,right_equation) : {before_value(before,index) == after_value(_,index) : Element}
  agrees(index)

# A Shaped tensor of a reader at a context is a Shaped tensor of another
# reader at a context that agrees with it at every index.
def restated(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,+before: Before,+after: After,
  agrees: @+index: U32 -> {before_value(before,index) == after_value(after,index) : Element},+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>,
  shaped: Shaped(~Element,~Before,~before_value,before,shape,tensor)) -> Shaped(~Element,~After,~after_value,after,shape,tensor):
  Shape.Both{usable,contract} = shaped
  Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  %equation : Shaped(~Element,~After,~after_value,after,shape,_)
  Shape.Both{Usable{bands,geometry,fill,workers,{==}},
    relabeled(~Element,~Before,~After,~before_value,~after_value,~(left => right => Fixed(~Before,~After,before,after,left,right)),
      ~(left => right => related => index => fixed_agrees(~Element,~Before,~After,~before_value,~after_value,before,after,agrees,left,right,related,index)),
      before,after,Shape.Both{{==},{==}},Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},
      %Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},tensor,equation) :
        Contract(~Element,~Before,~before_value,before,_)
      contract)}

# A whole tensor is one band of one row of one channel, as wide as its
# element count.
def whole_width(+shape: List<&2,U32>,+count: U32,accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}) -> {Tensors.count_or_zero(shape) == count : U32}:
  Equal.cong(Maybe<&2,U32>,U32,Tensors.known_count,Tensors.count_of(shape),Some{count},accepted)

def row_cells(+count: U32) -> {Nat.mul(1n,Nat.mul(1n,U32.to_nat(count))) == U32.to_nat(count) : Nat}:
  Equal.trans(Nat,Nat.mul(1n,Nat.mul(1n,U32.to_nat(count))),Nat.mul(1n,U32.to_nat(count)),U32.to_nat(count),
    Addition.zero_right(Nat.mul(1n,U32.to_nat(count))),Addition.zero_right(U32.to_nat(count)))

def whole_counted(+shape: List<&2,U32>,+count: U32,accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}) ->
  {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat}:
  Count.count_exact(ShapeLaws.elements(shape),count,Count.count_of(shape,count,accepted))

def whole_coherent(+shape: List<&2,U32>,+count: U32,accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}) ->
  Coherent(shape,Bands.Geometry{1,1,1,count}):
  +row = Nat.mul(1n,Nat.mul(1n,U32.to_nat(count)))
  Equal.trans(Nat,ShapeLaws.elements(shape),U32.to_nat(count),Nat.mul(1n,row),whole_counted(shape,count,accepted),
    Equal.sym(Nat,Nat.mul(1n,row),U32.to_nat(count),Equal.trans(Nat,Nat.mul(1n,row),row,U32.to_nat(count),Addition.zero_right(row),row_cells(count))))

def whole_well_formed(~Element: Data,-storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+count: U32,
  accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},formed: Contract.WellFormed<Element,shape,Buffer.contents(~Element,storage)>) ->
  Metadata.WellFormed<Element,Bands.Geometry{1,1,1,count},Model.contents(~Element,Partition.Leaf{storage,0,1,Unit{}})>:
  Contract.WellFormed{length_exact,ready} = formed
  +row = Nat.mul(1n,Nat.mul(1n,U32.to_nat(count)))
  Metadata.WellFormed{
    Metadata.Fields{{==},{==},
      Equal.trans(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,storage)),ShapeLaws.elements(shape),row,length_exact,
        Equal.trans(Nat,ShapeLaws.elements(shape),U32.to_nat(count),row,whole_counted(shape,count,accepted),Equal.sym(Nat,row,U32.to_nat(count),row_cells(count))))},
    ready,BandPlan.LeafCoverage{{==},{==}}}

# A usable whole tensor whose shape the machine counts is Normal.
def whole_normal(~Element: Data,-storage: Storage.Buffer<Element>,+shape: List<&2,U32>,fill: Element,workers: U32,
  counted: Counted<shape>,formed: Contract.WellFormed<Element,shape,Buffer.contents(~Element,storage)>) ->
  Normal(~Element,Tensors.whole(~Element,storage,shape,fill,Tensors.Usable{},workers)):
  Counted{+count,+accepted} = counted
  %Equal.sym(U32,Tensors.count_or_zero(shape),count,whole_width(shape,count,accepted)) :
    Normal(~Element,Tensors.Tensor{Partition.Leaf{storage,0,1,Unit{}},Bands.Geometry{1,1,1,_},shape,fill,Tensors.Usable{},workers})
  Shape.Both{whole_well_formed(~Element,storage,shape,count,accepted,formed),Shape.Both{Counted{count,accepted},whole_coherent(shape,count,accepted)}}

def count(+shape: List<&2,U32>,evidence: Counted<shape>) -> U32:
  Counted{count,accepted} = evidence
  count

def evidence(+shape: List<&2,U32>,+counted: Counted<shape>) -> Count.Count<ShapeLaws.elements(shape),count(shape,counted)>:
  Counted{+count,accepted} = counted
  Count.count_of(shape,count,accepted)

def geometry_evidence(+shape: List<&2,U32>,+geometry: Bands.Geometry,+counted: Counted<shape>,
  coherent: Coherent(shape,geometry)) -> Count.Count<BandGather.elements(geometry),count(shape,counted)>:
  %coherent : Count.Count<_,count(shape,counted)>
  evidence(shape,counted)

def domain(+shape: List<&2,U32>,+geometry: Bands.Geometry,+counted: Counted<shape>,
  coherent: Coherent(shape,geometry),
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool}) ->
  BandGather.Domain<geometry,Default.limits(),Machine.capacity(Default.limits())>:
  BandCount.evidence(geometry,count(shape,counted),channels_positive,height_positive,width_positive,
    %Equal.sym(Nat,ShapeLaws.elements(BandCount.extents(geometry)),BandGather.elements(geometry),BandCount.domain_elements(geometry)) : Count.Count<_,count(shape,counted)>
    geometry_evidence(shape,geometry,counted,coherent))

# Bands made by checked calls (Tensors.transformed) are the result when every
# call was usable and the bands cover the geometry. The result is then Normal
# and what every call gave holds of every band; an operation states, as
# Result, what such a banded result gives it.
def Banded(~Element: Data,~Scope: Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Result: @+scope: Scope -> @-tensor: Tensors.Tensor<Element> -> Data,+scope: Scope,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Type:
  @+model: Model.Model<Element> -> @normal: Normal(~Element,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers}) ->
    @facts: Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Model.pack(~Element,model))) ->
    Result(scope,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})

# What one usable call gives: a ready band, and what the operation states of it.
def Made(~Element: Data,~Scope: Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+scope: Scope) ->
  @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Model.Both(~Element,~Scope,~Model.always(~Element,~Scope,~Model.ready_band(~Element)),~After,scope)

def transformed_accepted(~Element: Data,~Scope: Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Result: @+scope: Scope -> @-tensor: Tensors.Tensor<Element> -> Data,+scope: Scope,+operation: String,+reason: Tensors.Reason,
  +geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+limits: Machine.Limits,
  +rows: {Nat.is_le(Inspection.total_rows(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +model: Model.Model<Element>,+usable: Bool,ok: Bool,+classified: {usable && Bands.covers(geometry,Inspection.model(~Element,geometry,model)) == ok : Bool},
  every: {usable == True{} : Bool} -> Model.ContentAll(~Element,~Made(~Element,~Scope,~After,scope),Model.contents(~Element,Model.pack(~Element,model))),
  counted: Counted<shape>,coherent: Coherent(shape,geometry),banded: Banded(~Element,~Scope,~After,~Result,scope,geometry,shape,fill,workers),
  accepted: {Contract.tensor_ready(~Element,Tensors.transformed_accepted(~Element,ok,operation,reason,Model.pack(~Element,model),geometry,shape,fill,workers)) == True{} : Bool}) ->
  Result(scope,Tensors.transformed_accepted(~Element,ok,operation,reason,Model.pack(~Element,model),geometry,shape,fill,workers)):
  match ok:
    case True{}:
      +inspection = Inspection.model(~Element,geometry,model)
      +covers = Bands.covers(geometry,inspection)
      +checked = Inspection.covered(geometry,inspection,Boolean.right(usable,covers,classified))
      +packed = Model.packed_contents(~Element,model)
      +facts = {%packed : Model.ContentAll(~Element,~Made(~Element,~Scope,~After,scope),_)
        every(Boolean.left(usable,covers,classified)) : Model.ContentAll(~Element,~Made(~Element,~Scope,~After,scope),Model.content(~Element,model))}
      +unpacked = Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,model)),Model.content(~Element,model),packed)
      banded(model,
        Shape.Both{{%unpacked : Metadata.WellFormed<Element,geometry,_>
            Metadata.WellFormed{
              Metadata.inspected(~Element,geometry,model,Boolean.left(Inspection.valid(inspection),U32.is_zero(Inspection.first(inspection)),
                Boolean.left(Inspection.valid(inspection) && U32.is_zero(Inspection.first(inspection)),Nat.is_eq(U32.to_nat(Inspection.end(inspection)),Inspection.total_rows(geometry)),checked))),
              Model.content_first(~Element,~Scope,~Model.always(~Element,~Scope,~Model.ready_band(~Element)),~After,scope,Model.content(~Element,model),facts),
              %Model.layout_matches_content(~Element,model) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
              Inspection.checked(~Element,geometry,model,Machine.capacity(limits),Limits.capacity_fits(limits),rows,checked)} :
            Metadata.WellFormed<Element,geometry,Model.contents(~Element,Model.pack(~Element,model))>},
          Shape.Both{counted,coherent}},
        {%unpacked : Model.ContentAll(~Element,~After(scope),_)
          Model.content_second(~Element,~Scope,~Model.always(~Element,~Scope,~Model.ready_band(~Element)),~After,scope,Model.content(~Element,model),facts) :
          Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Model.pack(~Element,model)))})
    case False{}:
      Empty.absurd(Result(scope,Tensors.transformed_accepted(~Element,False{},operation,reason,Model.pack(~Element,model),geometry,shape,fill,workers)),WordEncoding.false_true(accepted))

# The bands named by their model.
def transformed_inspected(~Element: Data,~Scope: Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Result: @+scope: Scope -> @-tensor: Tensors.Tensor<Element> -> Data,+scope: Scope,+operation: String,+reason: Tensors.Reason,
  +geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+limits: Machine.Limits,
  +rows: {Nat.is_le(Inspection.total_rows(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  -made: Bands.Bands(~Element),+usable: Bool,observation: Model.Observation<Element,made>,
  every: {usable == True{} : Bool} -> Model.ContentAll(~Element,~Made(~Element,~Scope,~After,scope),Model.contents(~Element,made)),
  counted: Counted<shape>,coherent: Coherent(shape,geometry),banded: Banded(~Element,~Scope,~After,~Result,scope,geometry,shape,fill,workers),
  accepted: {Contract.tensor_ready(~Element,Tensors.transformed_inspected(~Element,usable,operation,reason,geometry,shape,fill,workers,Bands.inspect(~Element,geometry,made))) == True{} : Bool}) ->
  Result(scope,Tensors.transformed_inspected(~Element,usable,operation,reason,geometry,shape,fill,workers,Bands.inspect(~Element,geometry,made))):
  Model.Observation{+model,+equation} = observation
  +inspected = Inspection.inspect(~Element,geometry,model)
  %equation : Result(scope,Tensors.transformed_inspected(~Element,usable,operation,reason,geometry,shape,fill,workers,Bands.inspect(~Element,geometry,_)))
  %Equal.sym(Bands.Bands(~Element) & Bands.Inspection,Bands.inspect(~Element,geometry,Model.pack(~Element,model)),(Model.pack(~Element,model),Inspection.model(~Element,geometry,model)),inspected) :
    Result(scope,Tensors.transformed_inspected(~Element,usable,operation,reason,geometry,shape,fill,workers,_))
  transformed_accepted(~Element,~Scope,~After,~Result,scope,operation,reason,geometry,shape,fill,workers,limits,rows,model,usable,
    usable && Bands.covers(geometry,Inspection.model(~Element,geometry,model)),{==},
    usable_true => %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),made,equation) :
      Model.ContentAll(~Element,~Made(~Element,~Scope,~After,scope),Model.contents(~Element,_)) every(usable_true),
    counted,coherent,banded,
    %inspected : {Contract.tensor_ready(~Element,Tensors.transformed_inspected(~Element,usable,operation,reason,geometry,shape,fill,workers,_)) == True{} : Bool}
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),made,equation) :
      {Contract.tensor_ready(~Element,Tensors.transformed_inspected(~Element,usable,operation,reason,geometry,shape,fill,workers,Bands.inspect(~Element,geometry,_))) == True{} : Bool}
    accepted)

# The checked calls give the bands and whether every call was usable.
def transformed(~Element: Data,~Scope: Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Result: @+scope: Scope -> @-tensor: Tensors.Tensor<Element> -> Data,+scope: Scope,+operation: String,+reason: Tensors.Reason,
  +geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+limits: Machine.Limits,
  +rows: {Nat.is_le(Inspection.total_rows(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},result: Bands.Bands(~Element) & Bool,
  every: {Each.usable(~Element,result) == True{} : Bool} -> Model.ContentAll(~Element,~Made(~Element,~Scope,~After,scope),Model.contents(~Element,Each.made(~Element,result))),
  counted: Counted<shape>,coherent: Coherent(shape,geometry),banded: Banded(~Element,~Scope,~After,~Result,scope,geometry,shape,fill,workers),
  accepted: {Contract.tensor_ready(~Element,Tensors.transformed(~Element,operation,reason,geometry,shape,fill,workers,result)) == True{} : Bool}) ->
  Result(scope,Tensors.transformed(~Element,operation,reason,geometry,shape,fill,workers,result)):
  (made,+usable) = result
  transformed_inspected(~Element,~Scope,~After,~Result,scope,operation,reason,geometry,shape,fill,workers,limits,rows,made,usable,Model.observe(~Element,made),
    every,counted,coherent,banded,accepted)
