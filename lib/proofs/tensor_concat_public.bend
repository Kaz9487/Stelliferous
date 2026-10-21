# Concatenating Ready tensors along any axis gives a Normal tensor that holds,
# at every flat index, the part that covers its place along the axis, read at
# that place of [outer, extent, inner], whatever storage each part has: the
# parts meet in the plan of one of them, each arrives in it in the geometry of
# its joining, and one band kernel runs.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_band_concat.bend as BandConcat
import ../tensor_layout.bend as Layout
import ../tensor_shape.bend as Shapes
import ../tensor_band_schedule.bend as Schedule
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_band_model.bend as Model
import ./tensor_band_values.bend as BandValues
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_plan.bend as Coverage
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_count.bend as BandCount
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_target.bend as Target
import ./tensor_band_arrive.bend as Arrive
import ./tensor_band_read.bend as Read
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_band_concat_caller.bend as BandConcatCaller
import ./tensor_band_concat_semantics.bend as Semantics
import ./tensor_contract.bend as Contract
import ./tensor_concat_leaf_values.bend as ConcatLeaf
import ./tensor_count_refinement.bend as Count
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_shape_refinement.bend as ShapeLaws
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./machine_limit_bounds.bend as Limits
import ./word_encoding.bend as WordEncoding

# A tensor that holds the concat of the readers of the parts.
def Concatenation(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+context: Semantics.Concatenated<Part>,-tensor: Tensors.Tensor<Element>) -> Data:
  Global.Contract(~Element,~Semantics.Concatenated<Part>,~Semantics.value(~Element,~Part,~part_value),context,tensor)

# ---- Ready parts ----

# A part: a Ready tensor of bands that holds a reader's values.
type Input<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element> is Data:
  Input{part: Part,bands: Model.Model<Element>,geometry: Bands.Geometry,shape: List<&2,U32>,fill: Element,workers: U32,
    contract: Global.Contract(~Element,~Part,~part_value,part,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})}

type Inputs<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element> is Data:
  Only{input: Input<Element,Part,part_value>}
  Also{input: Input<Element,Part,part_value>,rest: Inputs<Element,Part,part_value>}

def ready_tensor(~Element: Data,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}

def input_shape(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,input: Input<Element,Part,part_value>) -> List<&2,U32>:
  match input:
    case Input{part,bands,geometry,shape,fill,workers,contract}: shape

def input_part(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,input: Input<Element,Part,part_value>) -> Part:
  match input:
    case Input{part,bands,geometry,shape,fill,workers,contract}: part

def owner(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,input: Input<Element,Part,part_value>) -> BandConcatExecution.TensorOwnerModel<Element>:
  match input:
    case Input{part,bands,geometry,shape,fill,workers,contract}: BandConcatExecution.TensorOwner{bands,geometry,shape,fill,Tensors.Usable{},workers}

def owners(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,inputs: Inputs<Element,Part,part_value>) -> BandConcatExecution.PartsOwnerModel<Element>:
  match inputs:
    case Only{input}: BandConcatExecution.SingleOwner{owner(~Element,~Part,~part_value,input)}
    case Also{input,rest}: BandConcatExecution.ConsOwner{owner(~Element,~Part,~part_value,input),owners(~Element,~Part,~part_value,rest)}

def native(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,input: Input<Element,Part,part_value>) -> Tensors.Tensor<Element>:
  BandConcatExecution.tensor_owner_pack(~Element,owner(~Element,~Part,~part_value,input))

def natives(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,inputs: Inputs<Element,Part,part_value>) -> Tensors.Parts<Element>:
  BandConcatExecution.parts_owner_pack(~Element,owners(~Element,~Part,~part_value,inputs))

# The readers of the parts beside their extents along the axis.
def extents_of(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,inputs: Inputs<Element,Part,part_value>,+axis: U32) -> Semantics.Extents<Part>:
  match inputs:
    case Only{Input{part,bands,geometry,+shape,fill,workers,contract}}: Semantics.Last{part,Layout.extent_at(shape,U32.to_nat(axis))}
    case Also{Input{part,bands,geometry,+shape,fill,workers,contract},rest}:
      Semantics.Next{part,Layout.extent_at(shape,U32.to_nat(axis)),extents_of(~Element,~Part,~part_value,rest,axis)}

# ---- A Ready concat took the band route ----

def ready_usable(~Element: Data,status: Tensors.Status,valid: Bool,+first: List<&2,U32>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,-parts: Tensors.Parts<Element>,
  accepted: {Contract.tensor_ready(~Element,Layout.join_status(~Element,status,valid,first,shape,axis,fill,workers,parts)) == True{} : Bool}) -> {status == Tensors.Usable{} : Tensors.Status}:
  match status:
    case Tensors.Usable{}: {==}
    case Tensors.Failed{problem}: Empty.absurd({Tensors.Failed{problem} == Tensors.Usable{} : Tensors.Status},WordEncoding.false_true(accepted))

def ready_valid(~Element: Data,status: Tensors.Status,valid: Bool,+first: List<&2,U32>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,-parts: Tensors.Parts<Element>,
  accepted: {Contract.tensor_ready(~Element,Layout.join_status(~Element,status,valid,first,shape,axis,fill,workers,parts)) == True{} : Bool}) -> {valid == True{} : Bool}:
  match status valid:
    case Tensors.Usable{} True{}: {==}
    case Tensors.Usable{} False{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(accepted))
    case Tensors.Failed{problem} valid: Empty.absurd({valid == True{} : Bool},WordEncoding.false_true(accepted))

# A target without a count leaves the output Failed.
def ready_count(~Element: Data,count: Maybe<&2,U32>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,-parts: Tensors.Parts<Element>,
  +counts: {Tensors.count_of(shape) == count : Maybe<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,Layout.join_counted(~Element,count,shape,axis,fill,workers,parts)) == True{} : Bool}) -> Global.Counted<shape>:
  match count:
    case Some{+count}: Global.Counted{count,counts}
    case None{}: Empty.absurd(Global.Counted<shape>,WordEncoding.false_true(accepted))

# ---- Shapes split at the axis ----

# A shape is its extents before the axis, its extent at the axis and its
# extents after.
def split_elements(+shape: List<&2,U32>,+axis: Nat,inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {ShapeLaws.elements(shape) == Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,axis)),
    Nat.mul(U32.to_nat(Layout.extent_at(shape,axis)),ShapeLaws.elements(List.drop(&2,U32,shape,1n+axis)))) : Nat}:
  match shape:
    case []: Empty.absurd({1n == Nat.mul(1n,Nat.mul(U32.to_nat(Layout.extent_at([],axis)),1n)) : Nat},Order.below_zero(axis,inside))
    case +head <> +tail:
      match axis:
        case 0n:
          +cells = Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail))
          %Equal.sym(List<&2,U32>,List.drop(&2,U32,tail,0n),tail,ShapeLaws.drop_zero(tail)) : {cells == Nat.mul(1n,Nat.mul(U32.to_nat(head),ShapeLaws.elements(_))) : Nat}
          Equal.sym(Nat,Nat.add(cells,0n),cells,Addition.zero_right(cells))
        case 1n+ +previous:
          +before = ShapeLaws.elements(List.take(&2,U32,tail,previous))
          +rest = Nat.mul(U32.to_nat(Layout.extent_at(tail,previous)),ShapeLaws.elements(List.drop(&2,U32,tail,1n+previous)))
          Equal.trans(Nat,Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail)),Nat.mul(U32.to_nat(head),Nat.mul(before,rest)),Nat.mul(Nat.mul(U32.to_nat(head),before),rest),
            Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(head),value),ShapeLaws.elements(tail),Nat.mul(before,rest),split_elements(tail,previous,inside)),
            Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(head),before),rest),Nat.mul(U32.to_nat(head),Nat.mul(before,rest)),RowMajor.assoc(U32.to_nat(head),before,rest)))

# Setting the extent at the axis keeps the extents before and after it.
def set_take(+shape: List<&2,U32>,+axis: Nat,+value: U32) ->
  {List.take(&2,U32,List.set(&2,U32,shape,axis,value),axis) == List.take(&2,U32,shape,axis) : List<&2,U32>}:
  match shape:
    case []: {==}
    case +head <> +tail:
      match axis:
        case 0n: {==}
        case 1n+ +previous:
          Equal.cong(List<&2,U32>,List<&2,U32>,rest => head <> rest,List.take(&2,U32,List.set(&2,U32,tail,previous,value),previous),List.take(&2,U32,tail,previous),
            set_take(tail,previous,value))

def set_drop(+shape: List<&2,U32>,+axis: Nat,+value: U32) ->
  {List.drop(&2,U32,List.set(&2,U32,shape,axis,value),1n+axis) == List.drop(&2,U32,shape,1n+axis) : List<&2,U32>}:
  match shape:
    case []: {==}
    case +head <> +tail:
      match axis:
        case 0n: {==}
        case 1n+ +previous: set_drop(tail,previous,value)

def set_extent(+shape: List<&2,U32>,+axis: Nat,+value: U32,inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {Layout.extent_at(List.set(&2,U32,shape,axis,value),axis) == value : U32}:
  match shape:
    case []: Empty.absurd({Layout.extent_at([],axis) == value : U32},Order.below_zero(axis,inside))
    case +head <> +tail:
      match axis:
        case 0n: {==}
        case 1n+ +previous: set_extent(tail,previous,value,inside)

def set_length(+shape: List<&2,U32>,+axis: Nat,+value: U32) ->
  {List.length(&2,U32,List.set(&2,U32,shape,axis,value)) == List.length(&2,U32,shape) : Nat}:
  match shape:
    case []: {==}
    case +head <> +tail:
      match axis:
        case 0n: {==}
        case 1n+ +previous:
          Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,List.set(&2,U32,tail,previous,value)),List.length(&2,U32,tail),set_length(tail,previous,value))

# ---- The joining of a shape along an axis ----

# The outer cells of a tensor of the joining.
def joining_outer(joining: BandConcat.Joining) -> Nat:
  match joining:
    case BandConcat.Channels{axis,batch,height,width}: U32.to_nat(batch)
    case BandConcat.Lines{axis,lines,inner}: U32.to_nat(lines)

# The joining keeps the axis.
def imaged_axis(image: Bool,+channels: BandConcat.Joining,+shape: List<&2,U32>,+axis: U32,same: {BandConcat.axis(channels) == axis : U32}) ->
  {BandConcat.axis(BandConcat.imaged(image,channels,shape,axis)) == axis : U32}:
  match image:
    case True{}: same
    case False{}: {==}

def joining_axis(+shape: List<&2,U32>,+axis: U32) -> {BandConcat.axis(BandConcat.joining(shape,axis)) == axis : U32}:
  match shape:
    case []: {==}
    case +a <> +first_rest:
      match first_rest:
        case []: {==}
        case +b <> +second_rest:
          match second_rest:
            case []: {==}
            case +c <> +third_rest:
              match third_rest:
                case []: imaged_axis(U32.is_eq(axis,0),BandConcat.Channels{axis,1,b,c},[a,b,c],axis,{==})
                case +d <> +fourth_rest:
                  match fourth_rest:
                    case []: imaged_axis(U32.is_eq(axis,1),BandConcat.Channels{axis,a,c,d},[a,b,c,d],axis,{==})
                    case +e <> +fifth_rest: {==}

# The extent of a part along the axis of the joining is its extent at the
# axis.
def extent_on_axis(+joining: BandConcat.Joining,+axis: U32,+same: {BandConcat.axis(joining) == axis : U32},+shape: List<&2,U32>) ->
  {BandConcat.extent(joining,shape) == Layout.extent_at(shape,U32.to_nat(axis)) : U32}:
  Equal.cong(U32,U32,word => Layout.extent_at(shape,U32.to_nat(word)),BandConcat.axis(joining),axis,same)

# The cells of a joining before and after the axis of its shape.
type Framing<-joining: BandConcat.Joining,-shape: List<&2,U32>,-axis: U32> is Data:
  Framing{outer: {joining_outer(joining) == ShapeLaws.elements(List.take(&2,U32,shape,U32.to_nat(axis))) : Nat},
    inner: {Semantics.joining_inner(joining) == ShapeLaws.elements(List.drop(&2,U32,shape,1n+U32.to_nat(axis))) : Nat}}

# The cells of a counted part fit the capacity.
def count_room(+expected: Nat,+count: U32,+capacity: U32,+bounded_by: {Machine.capacity(Default.limits()) == capacity : U32},facts: Count.Count<expected,count>) ->
  {Nat.is_le(expected,U32.to_nat(capacity)) == True{} : Bool}:
  Count.Count{exact,bounded} = facts
  BandCount.natural_room(expected,count,capacity,exact,%bounded_by : {U32.is_le(count,_) == True{} : Bool} bounded)

def framing_outer(+joining: BandConcat.Joining,+shape: List<&2,U32>,+axis: U32,frame: Framing<joining,shape,axis>) ->
  {joining_outer(joining) == ShapeLaws.elements(List.take(&2,U32,shape,U32.to_nat(axis))) : Nat}:
  Framing{outer,inner} = frame
  outer

def framing_inner(+joining: BandConcat.Joining,+shape: List<&2,U32>,+axis: U32,frame: Framing<joining,shape,axis>) ->
  {Semantics.joining_inner(joining) == ShapeLaws.elements(List.drop(&2,U32,shape,1n+U32.to_nat(axis))) : Nat}:
  Framing{outer,inner} = frame
  inner

# A counted sub-shape counts its cells.
def counted_part(+shape: List<&2,U32>,+limit: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded_by: {Machine.capacity(Default.limits()) == capacity : U32},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(Tensors.count_or_zero(shape)) == ShapeLaws.elements(shape) : Nat}:
  +cells = ShapeLaws.elements(shape)
  +exact = Bounds.roundtrip(cells,capacity,capacity_ok,room)
  +within = {
    %Equal.sym(U32,Machine.capacity(Default.limits()),capacity,bounded_by) : {U32.is_le(U32.from_nat(cells),_) == True{} : Bool}
    %Equal.sym(Bool,U32.is_le(U32.from_nat(cells),capacity),Nat.is_le(U32.to_nat(U32.from_nat(cells)),U32.to_nat(capacity)),Comparison.le(U32.from_nat(cells),capacity)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(cells)),cells,exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room
    : {U32.is_le(U32.from_nat(cells),Machine.capacity(Default.limits())) == True{} : Bool}}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{U32.from_nat(cells)},
    CountCompletion.count_of(shape,U32.from_nat(cells),Equal.sym(Nat,U32.to_nat(U32.from_nat(cells)),cells,exact),within)) : {U32.to_nat(Tensors.known_count(_)) == cells : Nat}
  exact

# A lined joining counts the cells of a counted shape before and after the
# axis.
def lined_framing(+shape: List<&2,U32>,+axis: U32,+count: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded_by: {Machine.capacity(Default.limits()) == capacity : U32},+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> Framing<BandConcat.lined(shape,axis),shape,axis>:
  +room = count_room(ShapeLaws.elements(shape),count,capacity,bounded_by,Count.count_of(shape,count,counts))
  Framing{counted_part(List.take(&2,U32,shape,U32.to_nat(axis)),U32.to_nat(capacity),capacity,capacity_ok,bounded_by,
      CountCompletion.prefix_room(shape,U32.to_nat(axis),U32.to_nat(capacity),positive,room)),
    counted_part(List.drop(&2,U32,shape,1n+U32.to_nat(axis)),U32.to_nat(capacity),capacity,capacity_ok,bounded_by,
      CountCompletion.suffix_room(shape,1n+U32.to_nat(axis),U32.to_nat(capacity),positive,room))}

# An image shape along its channel axis joins its channels; any other axis
# joins lines.
def imaged_three(image: Bool,+channels: U32,+height: U32,+width: U32,+axis: U32,+count: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded_by: {Machine.capacity(Default.limits()) == capacity : U32},+counts: {Tensors.count_of([channels,height,width]) == Some{count} : Maybe<&2,U32>},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements([channels,height,width])) == True{} : Bool},+classified: {U32.is_eq(axis,0) == image : Bool}) ->
  Framing<BandConcat.imaged(image,BandConcat.Channels{axis,1,height,width},[channels,height,width],axis),[channels,height,width],axis>:
  match image:
    case True{}:
      %Equal.sym(U32,axis,0,Arithmetic.equal_of_is_eq(axis,0,classified)) : Framing<BandConcat.Channels{_,1,height,width},[channels,height,width],_>
      Framing{{==},Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(height),value),U32.to_nat(width),Nat.mul(U32.to_nat(width),1n),
        Equal.sym(Nat,Nat.mul(U32.to_nat(width),1n),U32.to_nat(width),Order.times_one(U32.to_nat(width))))}
    case False{}: lined_framing([channels,height,width],axis,count,capacity,capacity_ok,bounded_by,counts,positive)

def imaged_four(image: Bool,+batch: U32,+channels: U32,+height: U32,+width: U32,+axis: U32,+count: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded_by: {Machine.capacity(Default.limits()) == capacity : U32},+counts: {Tensors.count_of([batch,channels,height,width]) == Some{count} : Maybe<&2,U32>},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements([batch,channels,height,width])) == True{} : Bool},+classified: {U32.is_eq(axis,1) == image : Bool}) ->
  Framing<BandConcat.imaged(image,BandConcat.Channels{axis,batch,height,width},[batch,channels,height,width],axis),[batch,channels,height,width],axis>:
  match image:
    case True{}:
      %Equal.sym(U32,axis,1,Arithmetic.equal_of_is_eq(axis,1,classified)) : Framing<BandConcat.Channels{_,batch,height,width},[batch,channels,height,width],_>
      Framing{Equal.sym(Nat,Nat.mul(U32.to_nat(batch),1n),U32.to_nat(batch),Order.times_one(U32.to_nat(batch))),
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(height),value),U32.to_nat(width),Nat.mul(U32.to_nat(width),1n),
          Equal.sym(Nat,Nat.mul(U32.to_nat(width),1n),U32.to_nat(width),Order.times_one(U32.to_nat(width))))}
    case False{}: lined_framing([batch,channels,height,width],axis,count,capacity,capacity_ok,bounded_by,counts,positive)

def framing(+shape: List<&2,U32>,+axis: U32,+count: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded_by: {Machine.capacity(Default.limits()) == capacity : U32},+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> Framing<BandConcat.joining(shape,axis),shape,axis>:
  match shape:
    case []: lined_framing([],axis,count,capacity,capacity_ok,bounded_by,counts,positive)
    case +a <> +first_rest:
      match first_rest:
        case []: lined_framing([a],axis,count,capacity,capacity_ok,bounded_by,counts,positive)
        case +b <> +second_rest:
          match second_rest:
            case []: lined_framing([a,b],axis,count,capacity,capacity_ok,bounded_by,counts,positive)
            case +c <> +third_rest:
              match third_rest:
                case []: imaged_three(U32.is_eq(axis,0),a,b,c,axis,count,capacity,capacity_ok,bounded_by,counts,positive,{==})
                case +d <> +fourth_rest:
                  match fourth_rest:
                    case []: imaged_four(U32.is_eq(axis,1),a,b,c,d,axis,count,capacity,capacity_ok,bounded_by,counts,positive,{==})
                    case +e <> +fifth_rest: lined_framing(a <> b <> c <> d <> e <> fifth_rest,axis,count,capacity,capacity_ok,bounded_by,counts,positive)

# ---- The geometry of a joining ----

# A tensor of the joining has its outer cells, its extent and its inner cells.
def geometry_elements(+joining: BandConcat.Joining,+extent: U32,+capacity: U32,+fitted: Semantics.Fitted<joining,extent,capacity>) ->
  {BandGather.elements(BandConcat.geometry(joining,extent)) == Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(extent),Semantics.joining_inner(joining))) : Nat}:
  match joining:
    case BandConcat.Channels{axis,batch,height,width}: {==}
    case BandConcat.Lines{+axis,+lines,+inner}:
      Semantics.Fitted{+capacity_ok,+roomy} = fitted
      +cells = Nat.mul(U32.to_nat(lines),U32.to_nat(U32.mul(extent,inner)))
      Equal.trans(Nat,Nat.add(Nat.add(cells,0n),0n),Nat.add(cells,0n),Nat.mul(U32.to_nat(lines),Nat.mul(U32.to_nat(extent),U32.to_nat(inner))),
        Addition.zero_right(Nat.add(cells,0n)),
        Equal.trans(Nat,Nat.add(cells,0n),cells,Nat.mul(U32.to_nat(lines),Nat.mul(U32.to_nat(extent),U32.to_nat(inner))),Addition.zero_right(cells),
          Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(lines),value),U32.to_nat(U32.mul(extent,inner)),Nat.mul(U32.to_nat(extent),U32.to_nat(inner)),
            Arithmetic.multiply(extent,inner,capacity,capacity_ok,roomy))))

# A factor below its product with a positive one, on the left.
def covered(+left: Nat,+right: Nat,positive: {Nat.is_lt(0n,left) == True{} : Bool}) -> {Nat.is_le(right,Nat.mul(left,right)) == True{} : Bool}:
  %RowMajor.multiply_commutative(right,left) : {Nat.is_le(right,_) == True{} : Bool}
  RowMajor.product_covers(right,left,positive)

# The words of the output fit when its cells fit.
def fitted_of(+joining: BandConcat.Joining,+extent: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(extent),Semantics.joining_inner(joining))),U32.to_nat(capacity)) == True{} : Bool},
  +outer_positive: {Nat.is_lt(0n,joining_outer(joining)) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}) ->
  Semantics.Fitted<joining,extent,capacity>:
  match joining:
    case BandConcat.Channels{axis,+batch,+height,+width}:
      +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
      +image = Nat.mul(U32.to_nat(extent),spatial)
      Semantics.Fitted{capacity_ok,
        %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(batch),U32.to_nat(height)),U32.to_nat(width)),Nat.mul(U32.to_nat(batch),spatial),
          RowMajor.assoc(U32.to_nat(batch),U32.to_nat(height),U32.to_nat(width))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        Order.transitive(Nat.mul(U32.to_nat(batch),spatial),Nat.mul(U32.to_nat(batch),image),U32.to_nat(capacity),
          %RowMajor.multiply_commutative(spatial,U32.to_nat(batch)) : {Nat.is_le(_,Nat.mul(U32.to_nat(batch),image)) == True{} : Bool}
          %RowMajor.multiply_commutative(image,U32.to_nat(batch)) : {Nat.is_le(Nat.mul(spatial,U32.to_nat(batch)),_) == True{} : Bool}
          RowMajor.mul_le(spatial,image,U32.to_nat(batch),covered(U32.to_nat(extent),spatial,extent_positive)),room)}
    case BandConcat.Lines{axis,+lines,+inner}:
      Semantics.Fitted{capacity_ok,
        Order.transitive(Nat.mul(U32.to_nat(extent),U32.to_nat(inner)),Nat.mul(U32.to_nat(lines),Nat.mul(U32.to_nat(extent),U32.to_nat(inner))),U32.to_nat(capacity),
          covered(U32.to_nat(lines),Nat.mul(U32.to_nat(extent),U32.to_nat(inner)),outer_positive),room)}

# A part fits where the output fits and its cells fit.
def fitted_moved(+joining: BandConcat.Joining,+extent: U32,+total: U32,+capacity: U32,+fitted: Semantics.Fitted<joining,total,capacity>,
  +room: {Nat.is_le(Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(extent),Semantics.joining_inner(joining))),U32.to_nat(capacity)) == True{} : Bool},
  +outer_positive: {Nat.is_lt(0n,joining_outer(joining)) == True{} : Bool}) -> Semantics.Fitted<joining,extent,capacity>:
  match joining:
    case BandConcat.Channels{axis,batch,height,width}:
      Semantics.Fitted{capacity_ok,roomy} = fitted
      Semantics.Fitted{capacity_ok,roomy}
    case BandConcat.Lines{axis,+lines,+inner}:
      Semantics.Fitted{capacity_ok,roomy} = fitted
      Semantics.Fitted{capacity_ok,
        Order.transitive(Nat.mul(U32.to_nat(extent),U32.to_nat(inner)),Nat.mul(U32.to_nat(lines),Nat.mul(U32.to_nat(extent),U32.to_nat(inner))),U32.to_nat(capacity),
          covered(U32.to_nat(lines),Nat.mul(U32.to_nat(extent),U32.to_nat(inner)),outer_positive),room)}

type Positives<-geometry: Bands.Geometry> is Data:
  Positives{channels: {BandGather.channels_positive(geometry) == True{} : Bool},height: {BandGather.height_positive(geometry) == True{} : Bool},
    width: {BandGather.width_positive(geometry) == True{} : Bool}}

# The output geometry has channels, rows and columns when its cells do.
def positives(+joining: BandConcat.Joining,+extent: U32,+capacity: U32,+fitted: Semantics.Fitted<joining,extent,capacity>,
  +outer_positive: {Nat.is_lt(0n,joining_outer(joining)) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +inner_positive: {Nat.is_lt(0n,Semantics.joining_inner(joining)) == True{} : Bool}) -> Positives<BandConcat.geometry(joining,extent)>:
  match joining:
    case BandConcat.Channels{axis,batch,+height,+width}:
      Positives{extent_positive,RowMajor.left_factor_positive(U32.to_nat(height),U32.to_nat(width),inner_positive),
        DivisionOrder.right_factor_positive(U32.to_nat(height),U32.to_nat(width),inner_positive)}
    case BandConcat.Lines{+axis,+lines,+inner}:
      Semantics.Fitted{+capacity_ok,+roomy} = fitted
      Positives{{==},outer_positive,
        %Equal.sym(Nat,U32.to_nat(U32.mul(extent,inner)),Nat.mul(U32.to_nat(extent),U32.to_nat(inner)),Arithmetic.multiply(extent,inner,capacity,capacity_ok,roomy)) :
          {Nat.is_lt(0n,_) == True{} : Bool}
        Order.product_positive(U32.to_nat(extent),U32.to_nat(inner),extent_positive,inner_positive)}

# A target counted in a geometry with channels, rows and columns is a domain.
def output_domain(+shape: List<&2,U32>,+geometry: Bands.Geometry,+counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,geometry),signs: Positives<geometry>) ->
  BandGather.Domain<geometry,Default.limits(),Machine.capacity(Default.limits())>:
  Positives{channels,height,width} = signs
  Global.domain(shape,geometry,counted,coherent,channels,height,width)

# A plan of the rows of one tensor of the joining is a plan of the rows of
# any other.
def planned_moved(+joining: BandConcat.Joining,+extent: U32,+other: U32,+layout: Bands.Plan,planned: Target.Planned(BandConcat.geometry(joining,extent),layout)) ->
  Target.Planned(BandConcat.geometry(joining,other),layout):
  match joining:
    case BandConcat.Channels{axis,batch,height,width}: planned
    case BandConcat.Lines{axis,lines,inner}: planned

# One band of every image of a part is a plan of the rows of the output.
def unsplit_planned(+joining: BandConcat.Joining,+extent: U32,+other: U32,+capacity: U32,
  +domain: BandGather.Domain<BandConcat.geometry(joining,other),Default.limits(),capacity>) ->
  Target.Planned(BandConcat.geometry(joining,other),Bands.unsplit(BandConcat.geometry(joining,extent))):
  match joining:
    case BandConcat.Channels{+axis,+batch,+height,+width}: Target.split_planned(BandConcat.geometry(BandConcat.Channels{axis,batch,height,width},other),Default.limits(),capacity,0n,domain)
    case BandConcat.Lines{+axis,+lines,+inner}: Target.split_planned(BandConcat.geometry(BandConcat.Lines{axis,lines,inner},other),Default.limits(),capacity,0n,domain)

# ---- The parts in the joining of the output ----

# A part counted in the geometry of its extent, with its words in range.
type Framed<-shape: List<&2,U32>,-joining: BandConcat.Joining,-capacity: U32> is Data:
  Framed{coherent: Global.Coherent(shape,BandConcat.geometry(joining,BandConcat.extent(joining,shape))),
    fitted: Semantics.Fitted<joining,BandConcat.extent(joining,shape),capacity>}

# The output of the survey: the target of the parts and its joining.
def target_of(+first: List<&2,U32>,+axis: U32,+total: U32) -> List<&2,U32>:
  Layout.with_extent(first,U32.to_nat(axis),total)

def joining_of(+first: List<&2,U32>,+axis: U32,+total: U32) -> BandConcat.Joining:
  BandConcat.joining(target_of(first,axis,total),axis)

# A part the survey accepted has the extents of the target before and after
# the axis, so its cells are the outer cells, its extent and the inner cells.
def framed_part(+shape: List<&2,U32>,+first: List<&2,U32>,+axis: U32,+total: U32,+previous: U32,+limit: U32,+capacity: U32,
  +bounded_by: {Machine.capacity(Default.limits()) == capacity : U32},
  fields: BandConcatCaller.StepFields<first,shape,U32.to_nat(axis),previous,limit>,
  +frame: Framing<joining_of(first,axis,total),target_of(first,axis,total),axis>,
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first)) == True{} : Bool},
  +counted: Global.Counted<shape>,+fitted: Semantics.Fitted<joining_of(first,axis,total),total,capacity>,
  +outer_positive: {Nat.is_lt(0n,joining_outer(joining_of(first,axis,total))) == True{} : Bool}) -> Framed<shape,joining_of(first,axis,total),capacity>:
  BandConcatCaller.StepFields{+rank,+prefix,+suffix,sum_room,exact} = fields
  Framing{+outer_same,+inner_same} = frame
  Global.Counted{+count,+counts} = counted
  +index = U32.to_nat(axis)
  +target = target_of(first,axis,total)
  +joining = joining_of(first,axis,total)
  +extent = BandConcat.extent(joining,shape)
  +before = ShapeLaws.elements(List.take(&2,U32,shape,index))
  +after = ShapeLaws.elements(List.drop(&2,U32,shape,1n+index))
  +before_same = {
    %ShapeLaws.same_extents(List.take(&2,U32,first,index),List.take(&2,U32,shape,index),prefix) : {ShapeLaws.elements(_) == joining_outer(joining) : Nat}
    %set_take(first,index,total) : {ShapeLaws.elements(_) == joining_outer(joining) : Nat}
    Equal.sym(Nat,joining_outer(joining),ShapeLaws.elements(List.take(&2,U32,target,index)),outer_same)
    : {before == joining_outer(joining) : Nat}}
  +after_same = {
    %ShapeLaws.same_extents(List.drop(&2,U32,first,1n+index),List.drop(&2,U32,shape,1n+index),suffix) : {ShapeLaws.elements(_) == Semantics.joining_inner(joining) : Nat}
    %set_drop(first,index,total) : {ShapeLaws.elements(_) == Semantics.joining_inner(joining) : Nat}
    Equal.sym(Nat,Semantics.joining_inner(joining),ShapeLaws.elements(List.drop(&2,U32,target,1n+index)),inner_same)
    : {after == Semantics.joining_inner(joining) : Nat}}
  +shape_inside = {
    %Order.equal_of_is_eq(List.length(&2,U32,first),List.length(&2,U32,shape),rank) : {Nat.is_lt(index,_) == True{} : Bool}
    inside
    : {Nat.is_lt(index,List.length(&2,U32,shape)) == True{} : Bool}}
  +cells = {
    %before_same : {ShapeLaws.elements(shape) == Nat.mul(_,Nat.mul(U32.to_nat(extent),Semantics.joining_inner(joining))) : Nat}
    %after_same : {ShapeLaws.elements(shape) == Nat.mul(before,Nat.mul(U32.to_nat(extent),_)) : Nat}
    %Equal.sym(U32,extent,Layout.extent_at(shape,index),extent_on_axis(joining,axis,joining_axis(target,axis),shape)) :
      {ShapeLaws.elements(shape) == Nat.mul(before,Nat.mul(U32.to_nat(_),after)) : Nat}
    split_elements(shape,index,shape_inside)
    : {ShapeLaws.elements(shape) == Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(extent),Semantics.joining_inner(joining))) : Nat}}
  +room = {
    %cells : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    count_room(ShapeLaws.elements(shape),count,capacity,bounded_by,Count.count_of(shape,count,counts))
    : {Nat.is_le(Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(extent),Semantics.joining_inner(joining))),U32.to_nat(capacity)) == True{} : Bool}}
  +part_fitted = fitted_moved(joining,extent,total,capacity,fitted,room,outer_positive)
  Framed{Equal.trans(Nat,ShapeLaws.elements(shape),Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(extent),Semantics.joining_inner(joining))),
      BandGather.elements(BandConcat.geometry(joining,extent)),cells,
      Equal.sym(Nat,BandGather.elements(BandConcat.geometry(joining,extent)),Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(extent),Semantics.joining_inner(joining))),
        geometry_elements(joining,extent,capacity,part_fitted))),
    part_fitted}

def Frames(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,inputs: Inputs<Element,Part,part_value>,+joining: BandConcat.Joining,+capacity: U32) -> Data:
  match inputs:
    case Only{input}: Framed<input_shape(~Element,~Part,~part_value,input),joining,capacity>
    case Also{input,rest}: Shape.Both<Framed<input_shape(~Element,~Part,~part_value,input),joining,capacity>,Frames(~Element,~Part,~part_value,rest,joining,capacity)>

def input_framed(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,input: Input<Element,Part,part_value>,+first: List<&2,U32>,+axis: U32,+total: U32,+previous: U32,+limit: U32,
  +capacity: U32,+bounded_by: {Machine.capacity(Default.limits()) == capacity : U32},
  fields: BandConcatCaller.StepFields<first,BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,native(~Element,~Part,~part_value,input))),U32.to_nat(axis),previous,limit>,
  +frame: Framing<joining_of(first,axis,total),target_of(first,axis,total),axis>,
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first)) == True{} : Bool},+fitted: Semantics.Fitted<joining_of(first,axis,total),total,capacity>,
  +outer_positive: {Nat.is_lt(0n,joining_outer(joining_of(first,axis,total))) == True{} : Bool}) -> Framed<input_shape(~Element,~Part,~part_value,input),joining_of(first,axis,total),capacity>:
  match input:
    case Input{part,bands,geometry,+shape,fill,workers,contract}:
      Shape.Both{Shape.Both{well_formed,Shape.Both{counted,coherent}},values} = contract
      framed_part(shape,first,axis,total,previous,limit,capacity,bounded_by,fields,frame,inside,counted,fitted,outer_positive)

# Every part the survey accepted is counted in the geometry of its extent.
def frames(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,inputs: Inputs<Element,Part,part_value>,+first: List<&2,U32>,+axis: U32,+total: U32,
  +summary: Layout.Survey,+limit: U32,+capacity: U32,+bounded_by: {Machine.capacity(Default.limits()) == capacity : U32},
  fields: BandConcatCaller.Prefixes(~Element,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs)),first,U32.to_nat(axis),summary,limit),
  +frame: Framing<joining_of(first,axis,total),target_of(first,axis,total),axis>,
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,first)) == True{} : Bool},+fitted: Semantics.Fitted<joining_of(first,axis,total),total,capacity>,
  +outer_positive: {Nat.is_lt(0n,joining_outer(joining_of(first,axis,total))) == True{} : Bool}) -> Frames(~Element,~Part,~part_value,inputs,joining_of(first,axis,total),capacity):
  match inputs:
    case Only{+input}:
      input_framed(~Element,~Part,~part_value,input,first,axis,total,BandConcatCaller.survey_total(summary),limit,capacity,bounded_by,fields,frame,inside,fitted,outer_positive)
    case Also{+input,+rest}:
      Shape.Both{head,tail} = fields
      Shape.Both{input_framed(~Element,~Part,~part_value,input,first,axis,total,BandConcatCaller.survey_total(summary),limit,capacity,bounded_by,head,frame,inside,fitted,outer_positive),
        frames(~Element,~Part,~part_value,rest,first,axis,total,
          BandConcatCaller.step(~Element,first,U32.to_nat(axis),BandConcatExecution.tensor_contents(~Element,native(~Element,~Part,~part_value,input)),summary),
          limit,capacity,bounded_by,tail,frame,inside,fitted,outer_positive)}

# The total of a survey is the total before it and the extents it added.
def total_exact(+value: U32,+previous: U32,+extent: Nat,+limit: U32,fields: BandConcatCaller.TotalFields<value,previous,extent,limit>) ->
  {U32.to_nat(value) == Nat.add(U32.to_nat(previous),extent) : Nat}:
  BandConcatCaller.TotalFields{exact,room} = fields
  exact

# The extents of the survey are the extents along the axis of the joining.
def totals_agree(~Element: Data,+joining: BandConcat.Joining,+axis: U32,+same: {BandConcat.axis(joining) == axis : U32},parts: BandConcatExecution.PartContents<Element>) ->
  {BandConcatCaller.extent_total(~Element,parts,U32.to_nat(axis)) == Semantics.parts_total(~Element,joining,parts) : Nat}:
  match parts:
    case BandConcatExecution.SingleContent{+input}:
      Equal.cong(U32,Nat,word => U32.to_nat(word),Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),U32.to_nat(axis)),
        BandConcat.extent(joining,BandConcatExecution.tensor_shape(~Element,input)),
        Equal.sym(U32,BandConcat.extent(joining,BandConcatExecution.tensor_shape(~Element,input)),Layout.extent_at(BandConcatExecution.tensor_shape(~Element,input),U32.to_nat(axis)),
          extent_on_axis(joining,axis,same,BandConcatExecution.tensor_shape(~Element,input))))
    case BandConcatExecution.ConsContent{+input,+rest}:
      %extent_on_axis(joining,axis,same,BandConcatExecution.tensor_shape(~Element,input)) :
        {Nat.add(U32.to_nat(_),BandConcatCaller.extent_total(~Element,rest,U32.to_nat(axis))) ==
          Nat.add(U32.to_nat(BandConcat.extent(joining,BandConcatExecution.tensor_shape(~Element,input))),Semantics.parts_total(~Element,joining,rest)) : Nat}
      Equal.cong(Nat,Nat,total => Nat.add(U32.to_nat(BandConcat.extent(joining,BandConcatExecution.tensor_shape(~Element,input))),total),
        BandConcatCaller.extent_total(~Element,rest,U32.to_nat(axis)),Semantics.parts_total(~Element,joining,rest),totals_agree(~Element,joining,axis,same,rest))

# ---- The parts meet in one plan ----

def part_target(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+joining: BandConcat.Joining,input: Input<Element,Part,part_value>) -> Schedule.Target:
  match input:
    case Input{part,bands,+geometry,+shape,fill,workers,contract}:
      BandConcat.part_target(Bands.same_geometry(geometry,BandConcat.geometry(joining,BandConcat.extent(joining,shape))),BandConcat.geometry(joining,BandConcat.extent(joining,shape)),
        Model.layout(~Element,bands))

def targets(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+joining: BandConcat.Joining,inputs: Inputs<Element,Part,part_value>) -> Schedule.Target:
  match inputs:
    case Only{input}: part_target(~Element,~Part,~part_value,joining,input)
    case Also{input,rest}: Schedule.meet(part_target(~Element,~Part,~part_value,joining,input),targets(~Element,~Part,~part_value,joining,rest))

def target_layout(target: Schedule.Target) -> Bands.Plan:
  Schedule.Target{geometry,layout,split} = target
  layout

def target_geometry(target: Schedule.Target) -> Bands.Geometry:
  Schedule.Target{geometry,layout,split} = target
  geometry

def target_split(target: Schedule.Target) -> Bool:
  Schedule.Target{geometry,layout,split} = target
  split

def target_fields(+target: Schedule.Target) -> {target == Schedule.Target{target_geometry(target),target_layout(target),target_split(target)} : Schedule.Target}:
  Schedule.Target{geometry,layout,split} = target
  {==}

# The parts offer their plans in order, and the offers meet in the targets.
def offered(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+joining: BandConcat.Joining,+inputs: Inputs<Element,Part,part_value>) ->
  {BandConcat.offers(~Element,joining,natives(~Element,~Part,~part_value,inputs)) == (natives(~Element,~Part,~part_value,inputs),targets(~Element,~Part,~part_value,joining,inputs)) :
    Tensors.Parts<Element> & Schedule.Target}:
  match inputs:
    case Only{Input{part,+bands,+geometry,+shape,+fill,+workers,contract}}:
      +target = BandConcat.geometry(joining,BandConcat.extent(joining,shape))
      %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,bands)),(Model.pack(~Element,bands),Model.layout(~Element,bands)),Model.plan(~Element,bands)) :
        {BandConcat.offered_single(~Element,BandConcat.part_offered(~Element,target,geometry,shape,fill,Tensors.Usable{},workers,_)) ==
          (Tensors.Single{Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}},
            BandConcat.part_target(Bands.same_geometry(geometry,target),target,Model.layout(~Element,bands))) : Tensors.Parts<Element> & Schedule.Target}
      {==}
    case Also{Input{part,+bands,+geometry,+shape,+fill,+workers,contract},+rest}:
      +target = BandConcat.geometry(joining,BandConcat.extent(joining,shape))
      +offer = BandConcat.part_target(Bands.same_geometry(geometry,target),target,Model.layout(~Element,bands))
      %Equal.sym(Tensors.Parts<Element> & Schedule.Target,BandConcat.offers(~Element,joining,natives(~Element,~Part,~part_value,rest)),
        (natives(~Element,~Part,~part_value,rest),targets(~Element,~Part,~part_value,joining,rest)),offered(~Element,~Part,~part_value,joining,rest)) :
        {BandConcat.offered_cons(~Element,BandConcat.part_offer(~Element,joining,ready_tensor(~Element,bands,geometry,shape,fill,workers)),_) ==
          (Tensors.Cons{ready_tensor(~Element,bands,geometry,shape,fill,workers),natives(~Element,~Part,~part_value,rest)},Schedule.meet(offer,targets(~Element,~Part,~part_value,joining,rest))) :
          Tensors.Parts<Element> & Schedule.Target}
      %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,bands)),(Model.pack(~Element,bands),Model.layout(~Element,bands)),Model.plan(~Element,bands)) :
        {BandConcat.offered_cons(~Element,BandConcat.part_offered(~Element,target,geometry,shape,fill,Tensors.Usable{},workers,_),
          (natives(~Element,~Part,~part_value,rest),targets(~Element,~Part,~part_value,joining,rest))) ==
          (Tensors.Cons{ready_tensor(~Element,bands,geometry,shape,fill,workers),natives(~Element,~Part,~part_value,rest)},Schedule.meet(offer,targets(~Element,~Part,~part_value,joining,rest))) :
          Tensors.Parts<Element> & Schedule.Target}
      {==}

# A part offers a plan of the rows of the output: the plan of its bands when
# they are in the geometry of its extent, one band of every image otherwise.
def offered_rows(~Element: Data,rows: Bool,+shape: List<&2,U32>,+geometry: Bands.Geometry,+joining: BandConcat.Joining,+total: U32,+bands: Model.Model<Element>,+capacity: U32,
  +compared: {Bands.same_geometry(geometry,BandConcat.geometry(joining,BandConcat.extent(joining,shape))) == rows : Bool},
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),
  coverage: Coverage.Coverage(Model.content_layout(~Element,Model.content(~Element,bands)),0n,Inspection.total_rows(geometry)),
  +domain: BandGather.Domain<BandConcat.geometry(joining,total),Default.limits(),capacity>) ->
  Target.Holds(~Target.planned_in(BandConcat.geometry(joining,total)),
    BandConcat.part_target(rows,BandConcat.geometry(joining,BandConcat.extent(joining,shape)),Model.layout(~Element,bands))):
  match rows:
    case True{}:
      +target = BandConcat.geometry(joining,BandConcat.extent(joining,shape))
      +same = BandCount.same_geometry_equal(geometry,target,compared)
      planned_moved(joining,BandConcat.extent(joining,shape),total,Model.layout(~Element,bands),
        Shape.Both{%same : BandShape.Spans(_,Model.layout(~Element,bands)) BandShape.spans(~Element,geometry,bands,metadata),
          %same : Coverage.Coverage(Model.layout(~Element,bands),0n,Inspection.total_rows(_))
          %Equal.sym(Bands.Plan,Model.layout(~Element,bands),Model.content_layout(~Element,Model.content(~Element,bands)),Model.layout_matches_content(~Element,bands)) :
            Coverage.Coverage(_,0n,Inspection.total_rows(geometry))
          coverage})
    case False{}: unsplit_planned(joining,BandConcat.extent(joining,shape),total,capacity,domain)

def offer_planned(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+joining: BandConcat.Joining,input: Input<Element,Part,part_value>,+total: U32,+capacity: U32,
  +domain: BandGather.Domain<BandConcat.geometry(joining,total),Default.limits(),capacity>) ->
  Target.Holds(~Target.planned_in(BandConcat.geometry(joining,total)),part_target(~Element,~Part,~part_value,joining,input)):
  match input:
    case Input{part,+bands,+geometry,+shape,fill,workers,contract}:
      Shape.Both{Shape.Both{BandShape.WellFormed{metadata,ready,coverage},counted_coherent},values} = contract
      offered_rows(~Element,Bands.same_geometry(geometry,BandConcat.geometry(joining,BandConcat.extent(joining,shape))),shape,geometry,joining,total,bands,capacity,{==},
        %Model.packed_contents(~Element,bands) : BandShape.Contents(~Element,geometry,_) metadata,
        %Model.packed_contents(~Element,bands) : Coverage.Coverage(Model.content_layout(~Element,_),0n,Inspection.total_rows(geometry)) coverage,domain)

# The plan the parts meet in is a plan of the rows of the output.
def planned_targets(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+joining: BandConcat.Joining,+inputs: Inputs<Element,Part,part_value>,+total: U32,+capacity: U32,
  +domain: BandGather.Domain<BandConcat.geometry(joining,total),Default.limits(),capacity>) ->
  Target.Holds(~Target.planned_in(BandConcat.geometry(joining,total)),targets(~Element,~Part,~part_value,joining,inputs)):
  match inputs:
    case Only{input}: offer_planned(~Element,~Part,~part_value,joining,input,total,capacity,domain)
    case Also{+input,+rest}:
      Target.meet(~Target.planned_in(BandConcat.geometry(joining,total)),part_target(~Element,~Part,~part_value,joining,input),targets(~Element,~Part,~part_value,joining,rest),
        offer_planned(~Element,~Part,~part_value,joining,input,total,capacity,domain),
        planned_targets(~Element,~Part,~part_value,joining,rest,total,capacity,domain))

# ---- Bands that hold a part's values hold what read reads ----

# What read needs of a part's bands: their geometry, their coverage of its
# rows and the part's values.
def Held(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+part: Part,+input: BandConcatExecution.TensorContent<Element>,+geometry: Bands.Geometry) -> Data:
  BandConcatExecution.TensorContent{+bands,held_geometry,shape,fill,status,workers} = input
  Shape.Both<{held_geometry == geometry : Bands.Geometry},Shape.Both<Coverage.Coverage(Model.content_layout(~Element,bands),0n,Inspection.total_rows(geometry)),
    BandValues.Values(~Element,~Part,~part_value,part,geometry,bands)>>

def held_agrees(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+part: Part,+input: BandConcatExecution.TensorContent<Element>,+geometry: Bands.Geometry,
  +held: Held(~Element,~Part,~part_value,part,input,geometry),+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),BandGather.elements(geometry)) == True{} : Bool}) ->
  {part_value(part,index) == Semantics.tensor_original(~Element,~Read.read(~Element),input,index) : Element}:
  BandConcatExecution.TensorContent{+bands,+held_geometry,shape,fill,status,workers} = input
  Shape.Both{same,Shape.Both{coverage,values}} = held
  %Equal.sym(Bands.Geometry,held_geometry,geometry,same) : {part_value(part,index) == Read.read(~Element,bands,_,index) : Element}
  Equal.sym(Element,Read.read(~Element,bands,geometry,index),part_value(part,index),Read.read_values(~Element,~Part,~part_value,part,geometry,bands,coverage,values,index,inside))

# A part's bands in the joining: what read needs, the room of their cells and
# their cells as [outer, extent, inner].
def PairedPart(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+input: BandConcatExecution.TensorContent<Element>,+part: Part,+joining: BandConcat.Joining,+capacity: U32) -> Data:
  Shape.Both<Held(~Element,~Part,~part_value,part,input,Semantics.part_geometry(~Element,joining,input)),
    Shape.Both<{Nat.is_le(BandGather.elements(Semantics.part_geometry(~Element,joining,input)),U32.to_nat(capacity)) == True{} : Bool},
      {BandGather.elements(Semantics.part_geometry(~Element,joining,input)) ==
        Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(Semantics.part_extent(~Element,joining,input)),Semantics.joining_inner(joining))) : Nat}>>

def Paired(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,sources: BandConcatExecution.PartContents<Element>,parts: Semantics.Extents<Part>,
  +joining: BandConcat.Joining,+capacity: U32) -> Data:
  match sources parts:
    case BandConcatExecution.SingleContent{+input} Semantics.Last{+part,+extent}:
      Shape.Both<{Semantics.part_extent(~Element,joining,input) == extent : U32},PairedPart(~Element,~Part,~part_value,input,part,joining,capacity)>
    case BandConcatExecution.ConsContent{+input,rest} Semantics.Next{+part,+extent,others}:
      Shape.Both<Shape.Both<{Semantics.part_extent(~Element,joining,input) == extent : U32},PairedPart(~Element,~Part,~part_value,input,part,joining,capacity)>,
        Paired(~Element,~Part,~part_value,rest,others,joining,capacity)>
    case sources parts: Empty

def paired_total(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,sources: BandConcatExecution.PartContents<Element>,parts: Semantics.Extents<Part>,
  +joining: BandConcat.Joining,+capacity: U32,paired: Paired(~Element,~Part,~part_value,sources,parts,joining,capacity)) ->
  {Semantics.extent_total(~Part,parts) == Semantics.parts_total(~Element,joining,sources) : Nat}:
  match sources parts:
    case BandConcatExecution.SingleContent{+input} Semantics.Last{part,+extent}:
      Shape.Both{same,rest} = paired
      Equal.cong(U32,Nat,word => U32.to_nat(word),extent,Semantics.part_extent(~Element,joining,input),Equal.sym(U32,Semantics.part_extent(~Element,joining,input),extent,same))
    case BandConcatExecution.ConsContent{+input,+rest} Semantics.Next{part,+extent,+others}:
      Shape.Both{Shape.Both{same,held},tail} = paired
      %same :
        {Nat.add(U32.to_nat(_),Semantics.extent_total(~Part,others)) == Nat.add(U32.to_nat(Semantics.part_extent(~Element,joining,input)),Semantics.parts_total(~Element,joining,rest)) : Nat}
      Equal.cong(Nat,Nat,total => Nat.add(U32.to_nat(Semantics.part_extent(~Element,joining,input)),total),Semantics.extent_total(~Part,others),Semantics.parts_total(~Element,joining,rest),
        paired_total(~Element,~Part,~part_value,rest,others,joining,capacity,tail))
    case BandConcatExecution.SingleContent{input} Semantics.Next{part,extent,others}:
      Empty.absurd({Semantics.extent_total(~Part,Semantics.Next{part,extent,others}) == Semantics.parts_total(~Element,joining,BandConcatExecution.SingleContent{input}) : Nat},paired)
    case BandConcatExecution.ConsContent{input,rest} Semantics.Last{part,extent}:
      Empty.absurd({Semantics.extent_total(~Part,Semantics.Last{part,extent}) == Semantics.parts_total(~Element,joining,BandConcatExecution.ConsContent{input,rest}) : Nat},paired)

# ---- The parts arrive in the plan ----

# A part after it arrived: the kernel reads its bands, and they hold the
# part's values where read reads.
type Landed<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element,-tensor: Tensors.Tensor<Element>,-part: Part,-shape: List<&2,U32>,
  -joining: BandConcat.Joining,-layout: Bands.Plan,-capacity: U32> is Data:
  Landed{owner: BandConcatExecution.TensorOwnerModel<Element>,
    packed: {BandConcatExecution.tensor_owner_pack(~Element,owner) == tensor : Tensors.Tensor<Element>},
    shaped: {BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))) == shape : List<&2,U32>},
    contract: Semantics.SourceContract(~Element,~Read.read(~Element),joining,capacity,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))),
    layout_same: {Semantics.owner_layout(~Element,owner) == layout : Bands.Plan},
    paired: PairedPart(~Element,~Part,~part_value,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),part,joining,capacity)}

def landed_model(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+part: Part,+shape: List<&2,U32>,+fill: Element,+workers: U32,+joining: BandConcat.Joining,
  +layout: Bands.Plan,+capacity: U32,+model: Model.Model<Element>,
  facts: Arrive.Arrived(~Element,~Part,~part_value,part,BandConcat.geometry(joining,BandConcat.extent(joining,shape)),layout,Model.pack(~Element,model)),
  +room: {Nat.is_le(BandGather.elements(BandConcat.geometry(joining,BandConcat.extent(joining,shape))),U32.to_nat(capacity)) == True{} : Bool},
  +fitted: Semantics.Fitted<joining,BandConcat.extent(joining,shape),capacity>) ->
  Landed<Element,Part,part_value,Tensors.Tensor{Model.pack(~Element,model),BandConcat.geometry(joining,BandConcat.extent(joining,shape)),shape,fill,Tensors.Usable{},workers},part,shape,joining,layout,capacity>:
  Shape.Both{+layout_equal,Shape.Both{BandShape.WellFormed{+metadata,+ready,+coverage},+values}} = facts
  Semantics.Fitted{+capacity_ok,+roomy} = fitted
  +expected = BandConcat.geometry(joining,BandConcat.extent(joining,shape))
  +content = Model.contents(~Element,Model.pack(~Element,model))
  +source = {BandConcatExecution.TensorContent{content,expected,shape,fill,Tensors.Usable{},workers} : BandConcatExecution.TensorContent<Element>}
  +held = {Shape.Both{{==},Shape.Both{coverage,values}} : Held(~Element,~Part,~part_value,part,source,expected)}
  +layout_same = Equal.trans(Bands.Plan,Model.layout(~Element,model),Model.content_layout(~Element,content),layout,
    %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,model)),Model.content(~Element,model),Model.packed_contents(~Element,model)) :
      {Model.layout(~Element,model) == Model.content_layout(~Element,_) : Bands.Plan}
    Model.layout_matches_content(~Element,model),layout_equal)
  Landed{BandConcatExecution.TensorOwner{model,expected,shape,fill,Tensors.Usable{},workers},{==},{==},
    Shape.Both{BandShape.WellFormed{metadata,ready,coverage},
      Shape.Both{Read.agreed(~Element,~Part,~BandConcatExecution.TensorContent<Element>,~part_value,~Semantics.tensor_original(~Element,~Read.read(~Element)),~Held(~Element,~Part,~part_value),
          ~held_agrees(~Element,~Part,~part_value),part,source,expected,held,content,metadata,values,capacity,capacity_ok,room),
        Semantics.Fitted{capacity_ok,roomy}}},
    layout_same,Shape.Both{held,Shape.Both{room,geometry_elements(joining,BandConcat.extent(joining,shape),capacity,Semantics.Fitted{capacity_ok,roomy})}}}

def landed(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+part: Part,+shape: List<&2,U32>,+fill: Element,+workers: U32,+joining: BandConcat.Joining,
  +layout: Bands.Plan,+capacity: U32,-arrived: Bands.Bands(~Element),observation: Model.Observation<Element,arrived>,
  facts: Arrive.Arrived(~Element,~Part,~part_value,part,BandConcat.geometry(joining,BandConcat.extent(joining,shape)),layout,arrived),
  +room: {Nat.is_le(BandGather.elements(BandConcat.geometry(joining,BandConcat.extent(joining,shape))),U32.to_nat(capacity)) == True{} : Bool},
  +fitted: Semantics.Fitted<joining,BandConcat.extent(joining,shape),capacity>) ->
  Landed<Element,Part,part_value,Tensors.Tensor{arrived,BandConcat.geometry(joining,BandConcat.extent(joining,shape)),shape,fill,Tensors.Usable{},workers},part,shape,joining,layout,capacity>:
  Model.Observation{+model,+equation} = observation
  %equation : Landed<Element,Part,part_value,Tensors.Tensor{_,BandConcat.geometry(joining,BandConcat.extent(joining,shape)),shape,fill,Tensors.Usable{},workers},part,shape,joining,layout,capacity>
  landed_model(~Element,~Part,~part_value,part,shape,fill,workers,joining,layout,capacity,model,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),arrived,equation) : Arrive.Arrived(~Element,~Part,~part_value,part,BandConcat.geometry(joining,BandConcat.extent(joining,shape)),layout,_) facts,
    room,fitted)

# A part arrives in the plan the parts met in, in the geometry of its extent.
def arrived_input(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+input: Input<Element,Part,part_value>,+joining: BandConcat.Joining,+layout: Bands.Plan,+total: U32,
  +capacity: U32,+bounded_by: {Machine.capacity(Default.limits()) == capacity : U32},
  +framed: Framed<input_shape(~Element,~Part,~part_value,input),joining,capacity>,+planned: Target.Planned(BandConcat.geometry(joining,total),layout)) ->
  Landed<Element,Part,part_value,BandConcat.arrived(~Element,joining,layout,native(~Element,~Part,~part_value,input)),input_part(~Element,~Part,~part_value,input),
    input_shape(~Element,~Part,~part_value,input),joining,layout,capacity>:
  match input:
    case Input{+part,+bands,+geometry,+shape,+fill,+workers,contract}:
      Shape.Both{Shape.Both{well_formed,Shape.Both{+counted,coherent}},values} = contract
      Framed{+part_coherent,+fitted} = framed
      +expected = BandConcat.geometry(joining,BandConcat.extent(joining,shape))
      +arrival = {Target.Arrival{counted,part_coherent,planned_moved(joining,total,BandConcat.extent(joining,shape),layout,planned)} : Target.Arrival<shape,expected,layout>}
      landed(~Element,~Part,~part_value,part,shape,fill,workers,joining,layout,capacity,
        Tensors.arrive(~Element,expected,layout,ready_tensor(~Element,bands,geometry,shape,fill,workers)),
        Model.observe(~Element,Tensors.arrive(~Element,expected,layout,ready_tensor(~Element,bands,geometry,shape,fill,workers))),
        Arrive.arrived(~Element,~Part,~part_value,part,shape,expected,layout,geometry,fill,Tensors.Usable{},workers,bands,arrival,
          %Model.packed_contents(~Element,bands) : BandShape.WellFormed<Element,geometry,_> well_formed,coherent,
          %Model.packed_contents(~Element,bands) : BandValues.Values(~Element,~Part,~part_value,part,geometry,_) values),
        count_room(BandGather.elements(expected),Global.count(shape,counted),capacity,bounded_by,Global.geometry_evidence(shape,expected,counted,part_coherent)),fitted)

# The parts after they arrived, as the kernel reads them and as the readers
# of the parts read them.
type Arrivals<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element,-inputs: Inputs<Element,Part,part_value>,-joining: BandConcat.Joining,-axis: U32,-layout: Bands.Plan,
  -capacity: U32> is Data:
  Arrivals{owners: BandConcatExecution.PartsOwnerModel<Element>,
    packed: {BandConcatExecution.parts_owner_pack(~Element,owners) == BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,inputs)) : Tensors.Parts<Element>},
    contracts: Semantics.SourceContracts(~Element,~Read.read(~Element),joining,capacity,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,owners))),
    layouts: Semantics.Layouts(~Element,owners,layout),
    paired: Paired(~Element,~Part,~part_value,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,owners)),extents_of(~Element,~Part,~part_value,inputs,axis),
      joining,capacity),
    total: {Semantics.parts_total(~Element,joining,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,owners))) ==
      Semantics.parts_total(~Element,joining,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))) : Nat}}

def single_arrivals(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+input: Input<Element,Part,part_value>,+joining: BandConcat.Joining,+axis: U32,
  +same: {BandConcat.axis(joining) == axis : U32},+layout: Bands.Plan,+capacity: U32,
  landed: Landed<Element,Part,part_value,BandConcat.arrived(~Element,joining,layout,native(~Element,~Part,~part_value,input)),input_part(~Element,~Part,~part_value,input),
    input_shape(~Element,~Part,~part_value,input),joining,layout,capacity>) ->
  Arrivals<Element,Part,part_value,Only{input},joining,axis,layout,capacity>:
  match input:
    case Input{part,bands,geometry,+shape,fill,workers,contract}:
      Landed{+owner,+packed,+shaped,source_contract,layout_same,paired} = landed
      +arrived_shape = BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)))
      Arrivals{BandConcatExecution.SingleOwner{owner},
        Equal.cong(Tensors.Tensor<Element>,Tensors.Parts<Element>,tensor => Tensors.Single{tensor},BandConcatExecution.tensor_owner_pack(~Element,owner),
          BandConcat.arrived(~Element,joining,layout,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),packed),
        source_contract,layout_same,
        Shape.Both{
          %Equal.sym(List<&2,U32>,arrived_shape,shape,shaped) : {BandConcat.extent(joining,_) == Layout.extent_at(shape,U32.to_nat(axis)) : U32}
          extent_on_axis(joining,axis,same,shape),
          paired},
        Equal.cong(List<&2,U32>,Nat,extents => U32.to_nat(BandConcat.extent(joining,extents)),arrived_shape,shape,shaped)}

def cons_arrivals(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+input: Input<Element,Part,part_value>,+rest: Inputs<Element,Part,part_value>,+joining: BandConcat.Joining,
  +axis: U32,+same: {BandConcat.axis(joining) == axis : U32},+layout: Bands.Plan,+capacity: U32,
  landed: Landed<Element,Part,part_value,BandConcat.arrived(~Element,joining,layout,native(~Element,~Part,~part_value,input)),input_part(~Element,~Part,~part_value,input),
    input_shape(~Element,~Part,~part_value,input),joining,layout,capacity>,
  others: Arrivals<Element,Part,part_value,rest,joining,axis,layout,capacity>) ->
  Arrivals<Element,Part,part_value,Also{input,rest},joining,axis,layout,capacity>:
  match input:
    case Input{part,+bands,+geometry,+shape,+fill,+workers,contract}:
      Landed{+owner,+packed,+shaped,source_contract,layout_same,paired} = landed
      Arrivals{+arrived,+arrived_packed,contracts,layouts,others_paired,+others_total} = others
      +arrived_shape = BandConcatExecution.tensor_shape(~Element,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)))
      +rest_total = Semantics.parts_total(~Element,joining,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,arrived)))
      +natives_total = Semantics.parts_total(~Element,joining,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,rest)))
      Arrivals{BandConcatExecution.ConsOwner{owner,arrived},
        %Equal.sym(Tensors.Tensor<Element>,BandConcatExecution.tensor_owner_pack(~Element,owner),BandConcat.arrived(~Element,joining,layout,ready_tensor(~Element,bands,geometry,shape,fill,workers)),packed) :
          {Tensors.Cons{_,BandConcatExecution.parts_owner_pack(~Element,arrived)} ==
            Tensors.Cons{BandConcat.arrived(~Element,joining,layout,ready_tensor(~Element,bands,geometry,shape,fill,workers)),
              BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,rest))} : Tensors.Parts<Element>}
        %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,arrived),BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,rest)),arrived_packed) :
          {Tensors.Cons{BandConcat.arrived(~Element,joining,layout,ready_tensor(~Element,bands,geometry,shape,fill,workers)),_} ==
            Tensors.Cons{BandConcat.arrived(~Element,joining,layout,ready_tensor(~Element,bands,geometry,shape,fill,workers)),
              BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,rest))} : Tensors.Parts<Element>}
        {==},
        Shape.Both{source_contract,contracts},Shape.Both{layout_same,layouts},
        Shape.Both{Shape.Both{
            %Equal.sym(List<&2,U32>,arrived_shape,shape,shaped) : {BandConcat.extent(joining,_) == Layout.extent_at(shape,U32.to_nat(axis)) : U32}
            extent_on_axis(joining,axis,same,shape),
            paired},others_paired},
        %Equal.sym(List<&2,U32>,arrived_shape,shape,shaped) :
          {Nat.add(U32.to_nat(BandConcat.extent(joining,_)),rest_total) == Nat.add(U32.to_nat(BandConcat.extent(joining,shape)),natives_total) : Nat}
        %Equal.sym(Nat,rest_total,natives_total,others_total) : {Nat.add(U32.to_nat(BandConcat.extent(joining,shape)),_) == Nat.add(U32.to_nat(BandConcat.extent(joining,shape)),natives_total) : Nat}
        {==}}

def arrivals(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+inputs: Inputs<Element,Part,part_value>,+joining: BandConcat.Joining,+axis: U32,
  +same: {BandConcat.axis(joining) == axis : U32},+layout: Bands.Plan,+total: U32,+capacity: U32,+bounded_by: {Machine.capacity(Default.limits()) == capacity : U32},
  +framed: Frames(~Element,~Part,~part_value,inputs,joining,capacity),+planned: Target.Planned(BandConcat.geometry(joining,total),layout)) ->
  Arrivals<Element,Part,part_value,inputs,joining,axis,layout,capacity>:
  match inputs:
    case Only{+input}:
      single_arrivals(~Element,~Part,~part_value,input,joining,axis,same,layout,capacity,
        arrived_input(~Element,~Part,~part_value,input,joining,layout,total,capacity,bounded_by,framed,planned))
    case Also{+input,+rest}:
      Shape.Both{head,tail} = framed
      cons_arrivals(~Element,~Part,~part_value,input,rest,joining,axis,same,layout,capacity,
        arrived_input(~Element,~Part,~part_value,input,joining,layout,total,capacity,bounded_by,head,planned),
        arrivals(~Element,~Part,~part_value,rest,joining,axis,same,layout,total,capacity,bounded_by,tail,planned))

# ---- The output holds the concat of the readers ----

# A cell of a part below its outer cells lies below the cells of the part.
def framed_below(+outer: Nat,+extent: Nat,+inner: Nat,+line: Nat,+middle: Nat,+offset: Nat,
  line_inside: {Nat.is_lt(line,outer) == True{} : Bool},middle_inside: {Nat.is_lt(middle,extent) == True{} : Bool},offset_inside: {Nat.is_lt(offset,inner) == True{} : Bool}) ->
  {Nat.is_lt(Semantics.framed(extent,inner,line,middle,offset),Nat.mul(outer,Nat.mul(extent,inner))) == True{} : Bool}:
  RowMajor.radix(line,outer,Nat.add(Nat.mul(middle,inner),offset),Nat.mul(extent,inner),line_inside,RowMajor.radix(middle,extent,offset,inner,middle_inside,offset_inside))

# The part that covers a place along the axis holds its value there where
# read reads.
def head_sample(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+input: BandConcatExecution.TensorContent<Element>,+part: Part,+joining: BandConcat.Joining,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+paired: PairedPart(~Element,~Part,~part_value,input,part,joining,capacity),
  +line: Nat,+middle: Nat,+offset: Nat,
  line_inside: {Nat.is_lt(line,joining_outer(joining)) == True{} : Bool},middle_inside: {Nat.is_lt(middle,U32.to_nat(Semantics.part_extent(~Element,joining,input))) == True{} : Bool},
  offset_inside: {Nat.is_lt(offset,Semantics.joining_inner(joining)) == True{} : Bool}) ->
  {Semantics.tensor_original(~Element,~Read.read(~Element),input,U32.from_nat(Semantics.framed(U32.to_nat(Semantics.part_extent(~Element,joining,input)),Semantics.joining_inner(joining),line,middle,offset))) ==
    part_value(part,U32.from_nat(Semantics.framed(U32.to_nat(Semantics.part_extent(~Element,joining,input)),Semantics.joining_inner(joining),line,middle,offset))) : Element}:
  Shape.Both{+held,Shape.Both{+room,+cells}} = paired
  +geometry = Semantics.part_geometry(~Element,joining,input)
  +point = Semantics.framed(U32.to_nat(Semantics.part_extent(~Element,joining,input)),Semantics.joining_inner(joining),line,middle,offset)
  +below = {
    %Equal.sym(Nat,BandGather.elements(geometry),Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(Semantics.part_extent(~Element,joining,input)),Semantics.joining_inner(joining))),cells) :
      {Nat.is_lt(point,_) == True{} : Bool}
    framed_below(joining_outer(joining),U32.to_nat(Semantics.part_extent(~Element,joining,input)),Semantics.joining_inner(joining),line,middle,offset,line_inside,middle_inside,offset_inside)
    : {Nat.is_lt(point,BandGather.elements(geometry)) == True{} : Bool}}
  +exact = Bounds.roundtrip(point,capacity,capacity_ok,
    Order.transitive(point,BandGather.elements(geometry),U32.to_nat(capacity),Bounds.strict_le(point,BandGather.elements(geometry),below),room))
  +inside = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(point)),point,exact) : {Nat.is_lt(_,BandGather.elements(geometry)) == True{} : Bool} below :
    {Nat.is_lt(U32.to_nat(U32.from_nat(point)),BandGather.elements(geometry)) == True{} : Bool}}
  Equal.sym(Element,part_value(part,U32.from_nat(point)),Semantics.tensor_original(~Element,~Read.read(~Element),input,U32.from_nat(point)),
    held_agrees(~Element,~Part,~part_value,part,input,geometry,held,U32.from_nat(point),inside))

# The choice of a part by its extent picks the same part on both sides.
def picked(~Element: Data,choose: Bool,+left: Element,+right: Element,+other_left: Element,+other_right: Element,
  when_true: {choose == True{} : Bool} -> {left == other_left : Element},when_false: {choose == False{} : Bool} -> {right == other_right : Element}) ->
  {Bool.pick(Element,choose,left,right) == Bool.pick(Element,choose,other_left,other_right) : Element}:
  match choose:
    case True{}: when_true({==})
    case False{}: when_false({==})

# The bands of the parts hold, at every cell of the output, the part that
# covers its place, read by its reader.
def sampled(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,sources: BandConcatExecution.PartContents<Element>,parts: Semantics.Extents<Part>,
  +joining: BandConcat.Joining,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  paired: Paired(~Element,~Part,~part_value,sources,parts,joining,capacity),+line: Nat,+middle: Nat,+offset: Nat,
  +line_inside: {Nat.is_lt(line,joining_outer(joining)) == True{} : Bool},+middle_inside: {Nat.is_lt(middle,Semantics.parts_total(~Element,joining,sources)) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,Semantics.joining_inner(joining)) == True{} : Bool}) ->
  {Semantics.sample(~Element,~BandConcatExecution.TensorContent<Element>,~Semantics.tensor_original(~Element,~Read.read(~Element)),Semantics.extents_of(~Element,joining,sources),
      Semantics.joining_inner(joining),line,middle,offset) ==
    Semantics.sample(~Element,~Part,~part_value,parts,Semantics.joining_inner(joining),line,middle,offset) : Element}:
  match sources parts:
    case BandConcatExecution.SingleContent{+input} Semantics.Last{+part,+extent}:
      Shape.Both{+same,+held} = paired
      %same : {Semantics.sample(~Element,~BandConcatExecution.TensorContent<Element>,~Semantics.tensor_original(~Element,~Read.read(~Element)),Semantics.extents_of(~Element,joining,BandConcatExecution.SingleContent{input}),
          Semantics.joining_inner(joining),line,middle,offset) ==
        Semantics.sample(~Element,~Part,~part_value,Semantics.Last{part,_},Semantics.joining_inner(joining),line,middle,offset) : Element}
      head_sample(~Element,~Part,~part_value,input,part,joining,capacity,capacity_ok,held,line,middle,offset,line_inside,middle_inside,offset_inside)
    case BandConcatExecution.ConsContent{+input,+rest} Semantics.Next{+part,+extent,+others}:
      Shape.Both{Shape.Both{+same,+held},+tail} = paired
      +own = Semantics.part_extent(~Element,joining,input)
      +choose = Nat.is_lt(middle,U32.to_nat(own))
      +inner = Semantics.joining_inner(joining)
      %same : {Semantics.sample(~Element,~BandConcatExecution.TensorContent<Element>,~Semantics.tensor_original(~Element,~Read.read(~Element)),Semantics.extents_of(~Element,joining,BandConcatExecution.ConsContent{input,rest}),
          inner,line,middle,offset) ==
        Semantics.sample(~Element,~Part,~part_value,Semantics.Next{part,_,others},inner,line,middle,offset) : Element}
      picked(~Element,choose,
        Semantics.tensor_original(~Element,~Read.read(~Element),input,U32.from_nat(Semantics.framed(U32.to_nat(own),inner,line,middle,offset))),
        Semantics.sample(~Element,~BandConcatExecution.TensorContent<Element>,~Semantics.tensor_original(~Element,~Read.read(~Element)),Semantics.extents_of(~Element,joining,rest),
          inner,line,Nat.sub(middle,U32.to_nat(own)),offset),
        part_value(part,U32.from_nat(Semantics.framed(U32.to_nat(own),inner,line,middle,offset))),
        Semantics.sample(~Element,~Part,~part_value,others,inner,line,Nat.sub(middle,U32.to_nat(own)),offset),
        inside => head_sample(~Element,~Part,~part_value,input,part,joining,capacity,capacity_ok,held,line,middle,offset,line_inside,inside,offset_inside),
        later => sampled(~Element,~Part,~part_value,rest,others,joining,capacity,capacity_ok,tail,line,Nat.sub(middle,U32.to_nat(own)),offset,line_inside,
          Order.sub_lt(middle,U32.to_nat(own),Semantics.parts_total(~Element,joining,rest),
            Order.ge_true(middle,U32.to_nat(own),Order.comparison_not_less(Nat.cmp(middle,U32.to_nat(own)),later)),middle_inside),
          offset_inside))
    case BandConcatExecution.SingleContent{input} Semantics.Next{part,extent,others}:
      Empty.absurd({Semantics.sample(~Element,~BandConcatExecution.TensorContent<Element>,~Semantics.tensor_original(~Element,~Read.read(~Element)),
          Semantics.extents_of(~Element,joining,BandConcatExecution.SingleContent{input}),Semantics.joining_inner(joining),line,middle,offset) ==
        Semantics.sample(~Element,~Part,~part_value,Semantics.Next{part,extent,others},Semantics.joining_inner(joining),line,middle,offset) : Element},paired)
    case BandConcatExecution.ConsContent{input,rest} Semantics.Last{part,extent}:
      Empty.absurd({Semantics.sample(~Element,~BandConcatExecution.TensorContent<Element>,~Semantics.tensor_original(~Element,~Read.read(~Element)),
          Semantics.extents_of(~Element,joining,BandConcatExecution.ConsContent{input,rest}),Semantics.joining_inner(joining),line,middle,offset) ==
        Semantics.sample(~Element,~Part,~part_value,Semantics.Last{part,extent},Semantics.joining_inner(joining),line,middle,offset) : Element},paired)

# At every index of the output, the concat the band concat reads is the
# concat of the readers.
def value_agrees(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+sources: BandConcatExecution.PartContents<Element>,+parts: Semantics.Extents<Part>,
  +joining: BandConcat.Joining,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +paired: Paired(~Element,~Part,~part_value,sources,parts,joining,capacity),+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),Nat.mul(joining_outer(joining),Nat.mul(Semantics.parts_total(~Element,joining,sources),Semantics.joining_inner(joining)))) == True{} : Bool}) ->
  {Semantics.concat_value(~Element,~Read.read(~Element),Semantics.logical(~Element,joining,sources),index) ==
    Semantics.value(~Element,~Part,~part_value,Semantics.Concatenated{parts,Semantics.joining_inner(joining)},index) : Element}:
  +inner = Semantics.joining_inner(joining)
  +extent = Semantics.parts_total(~Element,joining,sources)
  +position = U32.to_nat(index)
  +frame = Nat.mul(extent,inner)
  +positive = Order.le_lt_trans(0n,position,Nat.mul(joining_outer(joining),frame),Order.zero_le(position),inside)
  +frame_positive = DivisionOrder.right_factor_positive(joining_outer(joining),frame,positive)
  +extent_positive = RowMajor.left_factor_positive(extent,inner,frame_positive)
  +inner_positive = DivisionOrder.right_factor_positive(extent,inner,frame_positive)
  %Equal.sym(Nat,Semantics.extent_total(~Part,parts),extent,paired_total(~Element,~Part,~part_value,sources,parts,joining,capacity,paired)) :
    {Semantics.concat_value(~Element,~Read.read(~Element),Semantics.logical(~Element,joining,sources),index) ==
      Semantics.sample(~Element,~Part,~part_value,parts,inner,Nat.div(position,Nat.mul(_,inner)),Nat.mod(Nat.div(position,inner),_),Nat.mod(position,inner)) : Element}
  sampled(~Element,~Part,~part_value,sources,parts,joining,capacity,capacity_ok,paired,Nat.div(position,frame),Nat.mod(Nat.div(position,inner),extent),Nat.mod(position,inner),
    DivisionOrder.quotient_below(position,joining_outer(joining),frame,frame_positive,inside),
    DivisionOrder.remainder_small(Nat.div(position,inner),extent,extent_positive),
    DivisionOrder.remainder_small(position,inner,inner_positive))

# What relates the concat the kernel reads to the concat of the readers.
type Joined<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element,-capacity: U32,-before: Semantics.Concatenated<BandConcatExecution.TensorContent<Element>>,
  -after: Semantics.Concatenated<Part>,-geometry: Bands.Geometry> is Data:
  Joined{sources: BandConcatExecution.PartContents<Element>,parts: Semantics.Extents<Part>,joining: BandConcat.Joining,extent: U32,
    before_equal: {before == Semantics.logical(~Element,joining,sources) : Semantics.Concatenated<BandConcatExecution.TensorContent<Element>>},
    after_equal: {after == Semantics.Concatenated{parts,Semantics.joining_inner(joining)} : Semantics.Concatenated<Part>},
    geometry_equal: {geometry == BandConcat.geometry(joining,extent) : Bands.Geometry},
    capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    total: {U32.to_nat(extent) == Semantics.parts_total(~Element,joining,sources) : Nat},
    fitted: Semantics.Fitted<joining,extent,capacity>,
    paired: Paired(~Element,~Part,~part_value,sources,parts,joining,capacity)}

def Joining(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+capacity: U32,+before: Semantics.Concatenated<BandConcatExecution.TensorContent<Element>>,
  +after: Semantics.Concatenated<Part>,+geometry: Bands.Geometry) -> Data:
  Joined<Element,Part,part_value,capacity,before,after,geometry>

def joined_agrees(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+capacity: U32,+before: Semantics.Concatenated<BandConcatExecution.TensorContent<Element>>,
  +after: Semantics.Concatenated<Part>,+geometry: Bands.Geometry,
  +related: Joining(~Element,~Part,~part_value,capacity,before,after,geometry),+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),BandGather.elements(geometry)) == True{} : Bool}) ->
  {Semantics.concat_value(~Element,~Read.read(~Element),before,index) == Semantics.value(~Element,~Part,~part_value,after,index) : Element}:
  Joined{+sources,+parts,+joining,+extent,+before_equal,+after_equal,+geometry_equal,+capacity_ok,+total,+fitted,+paired} = related
  %Equal.sym(Semantics.Concatenated<BandConcatExecution.TensorContent<Element>>,before,Semantics.logical(~Element,joining,sources),before_equal) :
    {Semantics.concat_value(~Element,~Read.read(~Element),_,index) == Semantics.value(~Element,~Part,~part_value,after,index) : Element}
  %Equal.sym(Semantics.Concatenated<Part>,after,Semantics.Concatenated{parts,Semantics.joining_inner(joining)},after_equal) :
    {Semantics.concat_value(~Element,~Read.read(~Element),Semantics.logical(~Element,joining,sources),index) == Semantics.value(~Element,~Part,~part_value,_,index) : Element}
  value_agrees(~Element,~Part,~part_value,sources,parts,joining,capacity,capacity_ok,paired,index,
    %total : {Nat.is_lt(U32.to_nat(index),Nat.mul(joining_outer(joining),Nat.mul(_,Semantics.joining_inner(joining)))) == True{} : Bool}
    %geometry_elements(joining,extent,capacity,fitted) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool}
    %geometry_equal : {Nat.is_lt(U32.to_nat(index),BandGather.elements(_)) == True{} : Bool}
    inside)

# The bands of the output hold the concat of the readers of the parts.
def contracted(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+sources: BandConcatExecution.PartContents<Element>,+parts: Semantics.Extents<Part>,
  +joining: BandConcat.Joining,+extent: U32,+shape: List<&2,U32>,+fill: Element,+workers: U32,+capacity: U32,
  +total: {U32.to_nat(extent) == Semantics.parts_total(~Element,joining,sources) : Nat},+fitted: Semantics.Fitted<joining,extent,capacity>,
  +paired: Paired(~Element,~Part,~part_value,sources,parts,joining,capacity),
  +room: {Nat.is_le(BandGather.elements(BandConcat.geometry(joining,extent)),U32.to_nat(capacity)) == True{} : Bool},
  +model: Model.Model<Element>,+normal: Global.Normal(~Element,Tensors.Tensor{Model.pack(~Element,model),BandConcat.geometry(joining,extent),shape,fill,Tensors.Usable{},workers}),
  values: BandValues.Values(~Element,~Semantics.Concatenated<BandConcatExecution.TensorContent<Element>>,~Semantics.concat_value(~Element,~Read.read(~Element)),
    Semantics.logical(~Element,joining,sources),BandConcat.geometry(joining,extent),Model.contents(~Element,Model.pack(~Element,model)))) ->
  Concatenation(~Element,~Part,~part_value,Semantics.Concatenated{parts,Semantics.joining_inner(joining)},
    Tensors.Tensor{Model.pack(~Element,model),BandConcat.geometry(joining,extent),shape,fill,Tensors.Usable{},workers}):
  Semantics.Fitted{+capacity_ok,+roomy} = fitted
  Shape.Both{BandShape.WellFormed{metadata,ready,coverage},counted_coherent} = normal
  +geometry = BandConcat.geometry(joining,extent)
  Shape.Both{normal,
    Read.agreed(~Element,~Semantics.Concatenated<BandConcatExecution.TensorContent<Element>>,~Semantics.Concatenated<Part>,~Semantics.concat_value(~Element,~Read.read(~Element)),
      ~Semantics.value(~Element,~Part,~part_value),~Joining(~Element,~Part,~part_value,capacity),~joined_agrees(~Element,~Part,~part_value,capacity),
      Semantics.logical(~Element,joining,sources),Semantics.Concatenated{parts,Semantics.joining_inner(joining)},geometry,
      Joined{sources,parts,joining,extent,{==},{==},{==},capacity_ok,total,Semantics.Fitted{capacity_ok,roomy},paired},
      Model.contents(~Element,Model.pack(~Element,model)),metadata,values,capacity,capacity_ok,room)}

# ---- The public concat on the band route ----

# The parts arrived in the plan they met in, and the band concat joins them.
def landed_concat(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+inputs: Inputs<Element,Part,part_value>,+joining: BandConcat.Joining,+axis: U32,+extent: U32,
  +layout: Bands.Plan,+shape: List<&2,U32>,+fill: Element,+workers: U32,arrivals: Arrivals<Element,Part,part_value,inputs,joining,axis,layout,Machine.capacity(Default.limits())>,
  +total: {U32.to_nat(extent) == Semantics.parts_total(~Element,joining,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))) : Nat},
  +domain: BandGather.Domain<BandConcat.geometry(joining,extent),Default.limits(),Machine.capacity(Default.limits())>,
  +fitted: Semantics.Fitted<joining,extent,Machine.capacity(Default.limits())>,
  counted: Global.Counted<shape>,coherent: Global.Coherent(shape,BandConcat.geometry(joining,extent)),
  accepted: {Contract.tensor_ready(~Element,BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,extent,
    BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,inputs)),shape,fill,workers)) == True{} : Bool}) ->
  Concatenation(~Element,~Part,~part_value,Semantics.Concatenated{extents_of(~Element,~Part,~part_value,inputs,axis),Semantics.joining_inner(joining)},
    BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,extent,BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,inputs)),shape,fill,workers)):
  Arrivals{+arrived,+packed,contracts,layouts,+paired,+arrived_total} = arrivals
  +geometry = BandConcat.geometry(joining,extent)
  +capacity = Machine.capacity(Default.limits())
  +sources = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,arrived))
  +kernel_total = Equal.trans(Nat,U32.to_nat(extent),Semantics.parts_total(~Element,joining,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),
    Semantics.parts_total(~Element,joining,sources),total,
    Equal.sym(Nat,Semantics.parts_total(~Element,joining,sources),Semantics.parts_total(~Element,joining,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),arrived_total))
  +room = BandCount.room(geometry,Default.limits(),capacity,domain)
  %packed : Concatenation(~Element,~Part,~part_value,Semantics.Concatenated{extents_of(~Element,~Part,~part_value,inputs,axis),Semantics.joining_inner(joining)},
    BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,extent,_,shape,fill,workers))
  Semantics.joined(~Element,~Read.read(~Element),~Concatenation(~Element,~Part,~part_value,Semantics.Concatenated{extents_of(~Element,~Part,~part_value,inputs,axis),Semantics.joining_inner(joining)}),
    arrived,joining,extent,layout,shape,fill,workers,kernel_total,domain,fitted,counted,coherent,contracts,layouts,
    model => normal => values => contracted(~Element,~Part,~part_value,sources,extents_of(~Element,~Part,~part_value,inputs,axis),joining,extent,shape,fill,workers,capacity,kernel_total,fitted,paired,room,
      model,normal,values),
    %Equal.sym(Tensors.Parts<Element>,BandConcatExecution.parts_owner_pack(~Element,arrived),BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,inputs)),packed) :
      {Contract.tensor_ready(~Element,BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,extent,_,shape,fill,workers)) == True{} : Bool}
    accepted)

# A Ready concat with a count: the parts meet in one plan of the joining of
# the target, arrive in it and the band concat joins them.
def counted_concat(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+inputs: Inputs<Element,Part,part_value>,+axis: U32,
  +usable: {BandConcatCaller.status_of(BandConcatCaller.surveyed_summary(~Element,owners(~Element,~Part,~part_value,inputs),
    Semantics.first_shape(~Element,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),axis)) == Tensors.Usable{} : Tensors.Status},
  +valid: {BandConcatCaller.survey_valid(BandConcatCaller.surveyed_summary(~Element,owners(~Element,~Part,~part_value,inputs),
    Semantics.first_shape(~Element,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),axis)) == True{} : Bool},
  counted: Global.Counted<BandConcatCaller.target_shape(Semantics.first_shape(~Element,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),axis,
    BandConcatCaller.surveyed_summary(~Element,owners(~Element,~Part,~part_value,inputs),Semantics.first_shape(~Element,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),axis))>,
  +nonempty: {Shapes.has_zero(BandConcatCaller.target_shape(Semantics.first_shape(~Element,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),axis,
    BandConcatCaller.surveyed_summary(~Element,owners(~Element,~Part,~part_value,inputs),Semantics.first_shape(~Element,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),axis))) == False{} : Bool},
  accepted: {Contract.tensor_ready(~Element,Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis)) == True{} : Bool}) ->
  Concatenation(~Element,~Part,~part_value,
    Semantics.Concatenated{extents_of(~Element,~Part,~part_value,inputs,axis),
      ShapeLaws.elements(List.drop(&2,U32,Semantics.first_shape(~Element,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),1n+U32.to_nat(axis)))},
    Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis)):
  Global.Counted{+count,+count_equal} = counted
  +sources = owners(~Element,~Part,~part_value,inputs)
  +contents = BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))
  +first = Semantics.first_shape(~Element,contents)
  +index = U32.to_nat(axis)
  +initial = BandConcatCaller.initial_summary(first,index)
  +summary = BandConcatCaller.surveyed_summary(~Element,sources,first,axis)
  +target = BandConcatCaller.target_shape(first,axis,summary)
  +total = BandConcatCaller.survey_total(summary)
  +workers = BandConcatCaller.survey_workers(summary)
  +fill = Semantics.first_fill(~Element,contents)
  +joining = BandConcat.joining(target,axis)
  +geometry = BandConcat.geometry(joining,total)
  +capacity = Machine.capacity(Default.limits())
  +capacity_ok = Limits.capacity_fits(Default.limits())
  +same = joining_axis(target,axis)
  +inside = BandConcatCaller.previous_valid(~Element,contents,first,index,initial,valid)
  +target_inside = {
    %Equal.sym(Nat,List.length(&2,U32,target),List.length(&2,U32,first),set_length(first,index,total)) : {Nat.is_lt(index,_) == True{} : Bool}
    inside
    : {Nat.is_lt(index,List.length(&2,U32,target)) == True{} : Bool}}
  +positive = CountCompletion.positive_elements(target,nonempty)
  +frame = framing(target,axis,count,capacity,capacity_ok,{==},count_equal,positive)
  +outer_same = framing_outer(joining,target,axis,frame)
  +inner_same = framing_inner(joining,target,axis,frame)
  +fields = BandConcatCaller.prefixes(~Element,contents,first,index,initial,4294967295,{==},valid)
  +totals = Equal.trans(Nat,U32.to_nat(total),BandConcatCaller.extent_total(~Element,contents,index),Semantics.parts_total(~Element,joining,contents),
    total_exact(total,0,BandConcatCaller.extent_total(~Element,contents,index),4294967295,BandConcatCaller.total_fields(~Element,contents,first,index,initial,4294967295,fields)),
    totals_agree(~Element,joining,axis,same,contents))
  +cells = {
    %Equal.sym(Nat,joining_outer(joining),ShapeLaws.elements(List.take(&2,U32,target,index)),outer_same) :
      {ShapeLaws.elements(target) == Nat.mul(_,Nat.mul(U32.to_nat(total),Semantics.joining_inner(joining))) : Nat}
    %Equal.sym(Nat,Semantics.joining_inner(joining),ShapeLaws.elements(List.drop(&2,U32,target,1n+index)),inner_same) :
      {ShapeLaws.elements(target) == Nat.mul(ShapeLaws.elements(List.take(&2,U32,target,index)),Nat.mul(U32.to_nat(total),_)) : Nat}
    %set_extent(first,index,total,inside) :
      {ShapeLaws.elements(target) == Nat.mul(ShapeLaws.elements(List.take(&2,U32,target,index)),Nat.mul(U32.to_nat(_),ShapeLaws.elements(List.drop(&2,U32,target,1n+index)))) : Nat}
    split_elements(target,index,target_inside)
    : {ShapeLaws.elements(target) == Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(total),Semantics.joining_inner(joining))) : Nat}}
  +cells_positive = {%cells : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(total),Semantics.joining_inner(joining)))) == True{} : Bool}}
  +outer_positive = RowMajor.left_factor_positive(joining_outer(joining),Nat.mul(U32.to_nat(total),Semantics.joining_inner(joining)),cells_positive)
  +frame_positive = DivisionOrder.right_factor_positive(joining_outer(joining),Nat.mul(U32.to_nat(total),Semantics.joining_inner(joining)),cells_positive)
  +extent_positive = RowMajor.left_factor_positive(U32.to_nat(total),Semantics.joining_inner(joining),frame_positive)
  +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(total),Semantics.joining_inner(joining),frame_positive)
  +room = {
    %cells : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    count_room(ShapeLaws.elements(target),count,capacity,{==},Count.count_of(target,count,count_equal))
    : {Nat.is_le(Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(total),Semantics.joining_inner(joining))),U32.to_nat(capacity)) == True{} : Bool}}
  +fitted = fitted_of(joining,total,capacity,capacity_ok,room,outer_positive,extent_positive)
  +coherent = Equal.trans(Nat,ShapeLaws.elements(target),Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(total),Semantics.joining_inner(joining))),BandGather.elements(geometry),
    cells,Equal.sym(Nat,BandGather.elements(geometry),Nat.mul(joining_outer(joining),Nat.mul(U32.to_nat(total),Semantics.joining_inner(joining))),geometry_elements(joining,total,capacity,fitted)))
  +domain = output_domain(target,geometry,Global.Counted{count,count_equal},coherent,positives(joining,total,capacity,fitted,outer_positive,extent_positive,inner_positive))
  +targets = targets(~Element,~Part,~part_value,joining,inputs)
  +layout = target_layout(targets)
  +led = {%Equal.sym(List<&2,U32>,first,first,{==}) : {Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis) == Layout.concat_first(~Element,axis,natives(~Element,~Part,~part_value,inputs),_,fill) : Tensors.Tensor<Element>}
    Semantics.concat_lead(~Element,axis,natives(~Element,~Part,~part_value,inputs)) :
    {Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis) == Layout.concat_first(~Element,axis,natives(~Element,~Part,~part_value,inputs),first,fill) : Tensors.Tensor<Element>}}
  +met = Equal.trans(Tensors.Tensor<Element>,Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis),
    Layout.concat_first(~Element,axis,natives(~Element,~Part,~part_value,inputs),first,fill),BandConcatCaller.band_met(~Element,joining,sources,target,fill,workers),led,
    BandConcatCaller.concat_band_route(~Element,sources,first,axis,fill,count,BandConcatCaller.BandSurveyFields{valid,usable,count_equal,nonempty}))
  +joined = {
    %Equal.sym(Tensors.Parts<Element> & Schedule.Target,BandConcat.offers(~Element,joining,natives(~Element,~Part,~part_value,inputs)),(natives(~Element,~Part,~part_value,inputs),targets),
      offered(~Element,~Part,~part_value,joining,inputs)) :
      {BandConcat.met(~Element,~Layout.band_joined(~Element),joining,target,fill,workers,_) ==
        BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,total,BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,inputs)),target,fill,workers) : Tensors.Tensor<Element>}
    %Equal.sym(Schedule.Target,targets,Schedule.Target{target_geometry(targets),layout,target_split(targets)},target_fields(targets)) :
      {BandConcat.met(~Element,~Layout.band_joined(~Element),joining,target,fill,workers,(natives(~Element,~Part,~part_value,inputs),_)) ==
        BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,total,BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,inputs)),target,fill,workers) : Tensors.Tensor<Element>}
    %Equal.sym(U32,BandConcat.extent(joining,target),total,
      Equal.trans(U32,BandConcat.extent(joining,target),Layout.extent_at(target,index),total,extent_on_axis(joining,axis,same,target),set_extent(first,index,total,inside))) :
      {BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,_,BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,inputs)),target,fill,workers) ==
        BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,total,BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,inputs)),target,fill,workers) : Tensors.Tensor<Element>}
    {==} : {BandConcatCaller.band_met(~Element,joining,sources,target,fill,workers) ==
      BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,total,BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,inputs)),target,fill,workers) : Tensors.Tensor<Element>}}
  +routed = Equal.trans(Tensors.Tensor<Element>,Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis),BandConcatCaller.band_met(~Element,joining,sources,target,fill,workers),
    BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,total,BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,inputs)),target,fill,workers),met,joined)
  +framed = frames(~Element,~Part,~part_value,inputs,first,axis,total,initial,4294967295,capacity,{==},fields,frame,inside,fitted,outer_positive)
  +planned = {
    %target_fields(targets) : Target.Holds(~Target.planned_in(geometry),_)
    planned_targets(~Element,~Part,~part_value,joining,inputs,total,capacity,domain) :
    Target.Planned(geometry,layout)}
  %set_drop(first,index,total) :
    Concatenation(~Element,~Part,~part_value,Semantics.Concatenated{extents_of(~Element,~Part,~part_value,inputs,axis),ShapeLaws.elements(_)},Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis))
  %inner_same :
    Concatenation(~Element,~Part,~part_value,Semantics.Concatenated{extents_of(~Element,~Part,~part_value,inputs,axis),_},Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis))
  %Equal.sym(Tensors.Tensor<Element>,Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis),
    BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,total,BandConcat.arrived_parts(~Element,joining,layout,natives(~Element,~Part,~part_value,inputs)),target,fill,workers),routed) :
    Concatenation(~Element,~Part,~part_value,Semantics.Concatenated{extents_of(~Element,~Part,~part_value,inputs,axis),Semantics.joining_inner(joining)},_)
  landed_concat(~Element,~Part,~part_value,inputs,joining,axis,total,layout,target,fill,workers,
    arrivals(~Element,~Part,~part_value,inputs,joining,axis,same,layout,total,capacity,{==},framed,planned),
    totals,domain,fitted,Global.Counted{count,count_equal},coherent,
    %routed : {Contract.tensor_ready(~Element,_) == True{} : Bool} accepted)

# Ready parts that agree on every extent but the axis: a Ready concat along
# the axis with a nonempty result is Normal and holds, at every flat index,
# the part that covers its place along the axis, read at that place of
# [outer, extent, inner].
def concat(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+inputs: Inputs<Element,Part,part_value>,+axis: U32,
  +nonempty: {Shapes.has_zero(BandConcatCaller.target_shape(Semantics.first_shape(~Element,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),axis,
    BandConcatCaller.surveyed_summary(~Element,owners(~Element,~Part,~part_value,inputs),Semantics.first_shape(~Element,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),axis))) == False{} : Bool},
  +accepted: {Contract.tensor_ready(~Element,Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis)) == True{} : Bool}) ->
  Concatenation(~Element,~Part,~part_value,
    Semantics.Concatenated{extents_of(~Element,~Part,~part_value,inputs,axis),
      ShapeLaws.elements(List.drop(&2,U32,Semantics.first_shape(~Element,BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))),1n+U32.to_nat(axis)))},
    Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis)):
  +sources = owners(~Element,~Part,~part_value,inputs)
  +contents = BandConcatExecution.parts_contents(~Element,natives(~Element,~Part,~part_value,inputs))
  +first = Semantics.first_shape(~Element,contents)
  +summary = BandConcatCaller.surveyed_summary(~Element,sources,first,axis)
  +target = BandConcatCaller.target_shape(first,axis,summary)
  +workers = BandConcatCaller.survey_workers(summary)
  +fill = Semantics.first_fill(~Element,contents)
  +status = BandConcatCaller.status_of(summary)
  +survey_valid = BandConcatCaller.survey_valid(summary)
  +led = Semantics.concat_lead(~Element,axis,natives(~Element,~Part,~part_value,inputs))
  +surveyed = Equal.trans(Tensors.Tensor<Element>,Layout.concat(~Element,natives(~Element,~Part,~part_value,inputs),axis),Layout.concat_first(~Element,axis,natives(~Element,~Part,~part_value,inputs),first,fill),
    Layout.join_status(~Element,status,survey_valid,first,target,axis,fill,workers,natives(~Element,~Part,~part_value,inputs)),led,
    BandConcatCaller.surveyed_route(~Element,sources,first,axis,fill))
  +ready = {%surveyed : {Contract.tensor_ready(~Element,_) == True{} : Bool} accepted :
    {Contract.tensor_ready(~Element,Layout.join_status(~Element,status,survey_valid,first,target,axis,fill,workers,natives(~Element,~Part,~part_value,inputs))) == True{} : Bool}}
  +usable = ready_usable(~Element,status,survey_valid,first,target,axis,fill,workers,natives(~Element,~Part,~part_value,inputs),ready)
  +valid = ready_valid(~Element,status,survey_valid,first,target,axis,fill,workers,natives(~Element,~Part,~part_value,inputs),ready)
  +selected = {
    %nonempty : {Contract.tensor_ready(~Element,Layout.join_shaped(~Element,_,target,axis,fill,workers,natives(~Element,~Part,~part_value,inputs))) == True{} : Bool}
    %valid : {Contract.tensor_ready(~Element,Layout.join_status(~Element,Tensors.Usable{},_,first,target,axis,fill,workers,natives(~Element,~Part,~part_value,inputs))) == True{} : Bool}
    %usable : {Contract.tensor_ready(~Element,Layout.join_status(~Element,_,survey_valid,first,target,axis,fill,workers,natives(~Element,~Part,~part_value,inputs))) == True{} : Bool}
    ready :
    {Contract.tensor_ready(~Element,Layout.join_counted(~Element,Tensors.count_of(target),target,axis,fill,workers,natives(~Element,~Part,~part_value,inputs))) == True{} : Bool}}
  counted_concat(~Element,~Part,~part_value,inputs,axis,usable,valid,
    ready_count(~Element,Tensors.count_of(target),target,axis,fill,workers,natives(~Element,~Part,~part_value,inputs),{==},selected),nonempty,accepted)
