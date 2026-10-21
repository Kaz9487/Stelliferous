# The values of the leaf concat. The parts are written one after another
# by set_slice into a tensor that starts at the fill value; appended is that
# fold in the order of the loop, each part written through its slice of the
# output at the running start along the axis.
import Base
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_contract.bend as Contract
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_global_shape_count.bend as Global
import ./tensor_set_slice_values.bend as Slices
import ../tensor_band_concat.bend as BandConcat
import ../tensor_band.bend as Bands
import ./tensor_full_values.bend as Full
import ./tensor_count_refinement.bend as Count
import ./word_encoding.bend as WordEncoding
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_division_order.bend as DivisionOrder

# The logical parts in order, each with its extents.
type Sources<-Part: Data> is Data:
  Final{part: Part,shape: List<&2,U32>}
  Then{part: Part,shape: List<&2,U32>,rest: Sources<Part>}

def appended(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,sources: Sources<Part>,+axis: U32,+target_shape: List<&2,U32>,+start: U32,accumulator: Element,+position: U32) -> Element:
  match sources:
    case Final{part,shape}: Slices.slice_element(~Element,~Part,~part_value,Slices.slice_of(axis,start,target_shape,shape),part,accumulator,position)
    case Then{part,+shape,rest}:
      appended(~Element,~Part,~part_value,rest,axis,target_shape,(start + Layout.extent_at(shape,U32.to_nat(axis)) : U32),
        Slices.slice_element(~Element,~Part,~part_value,Slices.slice_of(axis,start,target_shape,shape),part,accumulator,position),position)

type Appended<-Before: Data,-Part: Data> is Data:
  Appended{before: Before,sources: Sources<Part>,axis: U32,target_shape: List<&2,U32>,start: U32}

def appended_value(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,context: Appended<Before,Part>,position: U32) -> Element:
  Appended{before,sources,axis,target_shape,start} = context
  +position = position
  appended(~Element,~Part,~part_value,sources,axis,target_shape,start,before_value(before,position),position)

# The parts: Ready tensors of bands that hold the values of their readers.
type Pieces<-Element: Data,-Part: Data,-part_value: Part -> U32 -> Element> is Data:
  Last{part: Part,shape: List<&2,U32>,bands: Model.Model<Element>,geometry: Bands.Geometry,fill: Element,workers: U32,
    contract: Global.Contract(~Element,~Part,~part_value,part,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})}
  Next{part: Part,shape: List<&2,U32>,bands: Model.Model<Element>,geometry: Bands.Geometry,fill: Element,workers: U32,
    contract: Global.Contract(~Element,~Part,~part_value,part,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),rest: Pieces<Element,Part,part_value>}

def packed(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>) -> Tensors.Parts<Element>:
  match pieces:
    case Last{part,shape,bands,geometry,fill,workers,contract}: Tensors.Single{Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}}
    case Next{part,shape,bands,geometry,fill,workers,contract,rest}:
      Tensors.Cons{Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},packed(~Element,~Part,~part_value,rest)}

def sources_of(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>) -> Sources<Part>:
  match pieces:
    case Last{part,shape,bands,geometry,fill,workers,contract}: Final{part,shape}
    case Next{part,shape,bands,geometry,fill,workers,contract,rest}: Then{part,shape,sources_of(~Element,~Part,~part_value,rest)}

# A Failed output stays Failed through the write of every later part.
def stuck(~Element: Data,parts: Tensors.Parts<Element>,+axis: U32,tensor: Tensors.Tensor<Element>,+start: U32,
  failed: {Contract.tensor_ready(~Element,tensor) == False{} : Bool}) ->
  {Contract.tensor_ready(~Element,Layout.append_parts(~Element,parts,axis,(tensor,start))) == False{} : Bool}:
  match parts:
    case Tensors.Single{Tensors.Tensor{part_bands,part_geometry,part_shape,part_fill,part_status,part_workers}}:
      match tensor:
        case Tensors.Tensor{bands,geometry,shape,fill,status,workers}:
          match status:
            case Tensors.Usable{}:
              Empty.absurd({Contract.tensor_ready(~Element,Layout.append_parts(~Element,Tensors.Single{Tensors.Tensor{part_bands,part_geometry,part_shape,part_fill,part_status,part_workers}},axis,
                  (Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},start))) == False{} : Bool},
                WordEncoding.false_true(Equal.sym(Bool,True{},False{},failed)))
            case Tensors.Failed{problem}: {==}
    case Tensors.Cons{Tensors.Tensor{part_bands,part_geometry,+part_shape,part_fill,part_status,part_workers},rest}:
      match tensor:
        case Tensors.Tensor{bands,geometry,shape,fill,status,workers}:
          match status:
            case Tensors.Usable{}:
              Empty.absurd({Contract.tensor_ready(~Element,Layout.append_parts(~Element,Tensors.Cons{Tensors.Tensor{part_bands,part_geometry,part_shape,part_fill,part_status,part_workers},rest},axis,
                  (Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},start))) == False{} : Bool},
                WordEncoding.false_true(Equal.sym(Bool,True{},False{},failed)))
            case Tensors.Failed{problem}:
              stuck(~Element,rest,axis,
                Layout.set_slice(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers},axis,start,Tensors.Tensor{part_bands,part_geometry,part_shape,part_fill,part_status,part_workers}),
                (start + Layout.extent_at(part_shape,U32.to_nat(axis)) : U32),{==})

# A Ready append means the output it started from was Ready.
def continued(~Element: Data,parts: Tensors.Parts<Element>,+axis: U32,tensor: Tensors.Tensor<Element>,+start: U32,
  accepted: {Contract.tensor_ready(~Element,Layout.append_parts(~Element,parts,axis,(tensor,start))) == True{} : Bool}) ->
  {Contract.tensor_ready(~Element,tensor) == True{} : Bool}:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,status,workers}:
      match status:
        case Tensors.Usable{}: {==}
        case Tensors.Failed{problem}:
          Empty.absurd({Contract.tensor_ready(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}) == True{} : Bool},WordEncoding.false_true(
            %stuck(~Element,parts,axis,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers},start,{==}) : {_ == True{} : Bool}
            accepted))

# A Ready write through a slice is a tensor of the target shape of the
# reader of that write.
def placed_shaped(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,
  slice: Maybe<&2,Views.View>,+shape: List<&2,U32>,tensor: Tensors.Tensor<Element>,ready: {Contract.tensor_ready(~Element,tensor) == True{} : Bool},
  placing: Slices.Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,slice,shape,tensor)) ->
  Global.Shaped(~Element,~Slices.Placed<Before,Part>,~Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),Slices.Placed{before,part,slice},shape,tensor):
  Shape.Both{contract,extents} = placing
  Global.shaped(~Element,~Slices.Placed<Before,Part>,~Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),Slices.Placed{before,part,slice},shape,tensor,ready,extents,contract)

# The append of the parts onto a tensor of the shape: a Ready result is a
# tensor of the same shape whose reader is appended over the output.
def appended_pieces(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,pieces: Pieces<Element,Part,part_value>,-output: Tensors.Tensor<Element>,+axis: U32,+start: U32,+before: Before,+shape: List<&2,U32>,
  output_shaped: Global.Shaped(~Element,~Before,~before_value,before,shape,output),
  accepted: {Contract.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(output,start))) == True{} : Bool}) ->
  Global.Shaped(~Element,~Appended<Before,Part>,~appended_value(~Element,~Before,~Part,~before_value,~part_value),Appended{before,sources_of(~Element,~Part,~part_value,pieces),axis,shape,start},shape,
    Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(output,start))):
  match pieces:
    case Last{+part,+part_shape,+part_bands,+part_geometry,+part_fill,+part_workers,part_contract}:
      Shape.Both{usable,contract} = output_shaped
      Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
      +back = Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},output,equation)
      +ready = {
        %back : {Contract.tensor_ready(~Element,Layout.set_slice(~Element,_,axis,start,Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers})) == True{} : Bool}
        accepted
        : {Contract.tensor_ready(~Element,Layout.set_slice(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},axis,start,
            Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers})) == True{} : Bool}}
      %equation : Global.Shaped(~Element,~Appended<Before,Part>,~appended_value(~Element,~Before,~Part,~before_value,~part_value),Appended{before,Final{part,part_shape},axis,shape,start},shape,
        Layout.set_slice(~Element,_,axis,start,Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers}))
      Global.restated(~Element,~Slices.Placed<Before,Part>,~Appended<Before,Part>,~Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),~appended_value(~Element,~Before,~Part,~before_value,~part_value),
        Slices.Placed{before,part,Slices.slice_of(axis,start,shape,part_shape)},Appended{before,Final{part,part_shape},axis,shape,start},(index => {==}),shape,
        Layout.set_slice(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},axis,start,Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers}),
        placed_shaped(~Element,~Before,~Part,~before_value,~part_value,before,part,Slices.slice_of(axis,start,shape,part_shape),shape,
          Layout.set_slice(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},axis,start,
            Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers}),ready,
          Slices.set_slice(~Element,~Before,~Part,~before_value,~part_value,before,part,axis,start,bands,geometry,shape,fill,workers,part_bands,part_geometry,part_shape,part_fill,part_workers,
            %back : Global.Contract(~Element,~Before,~before_value,before,_) contract,part_contract,ready)))
    case Next{+part,+part_shape,+part_bands,+part_geometry,+part_fill,+part_workers,part_contract,+rest}:
      Shape.Both{usable,contract} = output_shaped
      Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
      +back = Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},output,equation)
      +next = (start + Layout.extent_at(part_shape,U32.to_nat(axis)) : U32)
      +proceeded = {
        %back : {Contract.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,Next{part,part_shape,part_bands,part_geometry,part_fill,part_workers,part_contract,rest}),axis,(_,start))) == True{} : Bool}
        accepted
        : {Contract.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,rest),axis,
            (Layout.set_slice(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},axis,start,
              Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers}),next))) == True{} : Bool}}
      +ready = continued(~Element,packed(~Element,~Part,~part_value,rest),axis,
        Layout.set_slice(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},axis,start,
          Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers}),next,proceeded)
      %equation : Global.Shaped(~Element,~Appended<Before,Part>,~appended_value(~Element,~Before,~Part,~before_value,~part_value),
        Appended{before,Then{part,part_shape,sources_of(~Element,~Part,~part_value,rest)},axis,shape,start},shape,
        Layout.append_parts(~Element,packed(~Element,~Part,~part_value,rest),axis,
          (Layout.set_slice(~Element,_,axis,start,Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers}),next)))
      Global.restated(~Element,~Appended<Slices.Placed<Before,Part>,Part>,~Appended<Before,Part>,~appended_value(~Element,~Slices.Placed<Before,Part>,~Part,~Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),~part_value),~appended_value(~Element,~Before,~Part,~before_value,~part_value),
        Appended{Slices.placement(~Before,~Part,before,part,axis,start,shape,part_shape),sources_of(~Element,~Part,~part_value,rest),axis,shape,next},
        Appended{before,Then{part,part_shape,sources_of(~Element,~Part,~part_value,rest)},axis,shape,start},(index => {==}),shape,
        Layout.append_parts(~Element,packed(~Element,~Part,~part_value,rest),axis,(Layout.set_slice(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},axis,start,Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers}),next)),
        appended_pieces(~Element,~Slices.Placed<Before,Part>,~Part,~Slices.placed_value(~Element,~Before,~Part,~before_value,~part_value),~part_value,rest,
          Layout.set_slice(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},axis,start,
            Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers}),
          axis,next,Slices.placement(~Before,~Part,before,part,axis,start,shape,part_shape),shape,
          placed_shaped(~Element,~Before,~Part,~before_value,~part_value,before,part,Slices.slice_of(axis,start,shape,part_shape),shape,
            Layout.set_slice(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},axis,start,
              Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers}),ready,
            Slices.set_slice(~Element,~Before,~Part,~before_value,~part_value,before,part,axis,start,bands,geometry,shape,fill,workers,part_bands,part_geometry,part_shape,part_fill,part_workers,
              %back : Global.Contract(~Element,~Before,~before_value,before,_) contract,part_contract,ready)),
          proceeded))

# A target shape without a count leaves the output Failed from the start.
def uncounted(~Element: Data,parts: Tensors.Parts<Element>,+axis: U32,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  accepted: {Contract.tensor_ready(~Element,Layout.append_parts(~Element,parts,axis,(Tensors.with_workers(~Element,Tensors.full_count(~Element,shape,None{},fill),workers),0))) == True{} : Bool}) -> Empty:
  WordEncoding.false_true(
    %stuck(~Element,parts,axis,Tensors.with_workers(~Element,Tensors.full_count(~Element,shape,None{},fill),workers),0,{==}) : {_ == True{} : Bool}
    accepted)

# A Ready append into the full tensor of the target shape is a tensor of that
# shape whose every index holds appended over the fill value.
def counted_shaped(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,count: Maybe<&2,U32>,+pieces: Pieces<Element,Part,part_value>,+axis: U32,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +counts: {Tensors.count_of(shape) == count : Maybe<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(Tensors.with_workers(~Element,Tensors.full_count(~Element,shape,count,fill),workers),0))) == True{} : Bool}) ->
  Global.Shaped(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~Full.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,shape,0},shape,
    Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(Tensors.with_workers(~Element,Tensors.full_count(~Element,shape,count,fill),workers),0))):
  match count:
    case Some{+count}:
      appended_pieces(~Element,~Element,~Part,~Full.constant(~Element),~part_value,pieces,Tensors.whole(~Element,Tensors.buffer(~Element,fill,count),shape,fill,Tensors.Usable{},U32.max(1,workers)),axis,0,fill,shape,
        Global.shaped(~Element,~Element,~Full.constant(~Element),fill,shape,Tensors.whole(~Element,Tensors.buffer(~Element,fill,count),shape,fill,Tensors.Usable{},U32.max(1,workers)),{==},{==},
          WholeLeaf.stored(~Element,~Element,~Full.constant(~Element),fill,Tensors.buffer(~Element,fill,count),shape,fill,U32.max(1,workers),
            Buffer.observe(~Element,Tensors.buffer(~Element,fill,count)),Full.initial_counted(~Element,shape,fill,count,Count.count_of(shape,count,counts)))),
        accepted)
    case None{}:
      Empty.absurd(Global.Shaped(~Element,~Appended<Element,Part>,~appended_value(~Element,~Element,~Part,~Full.constant(~Element),~part_value),Appended{fill,sources_of(~Element,~Part,~part_value,pieces),axis,shape,0},shape,
        Layout.append_parts(~Element,packed(~Element,~Part,~part_value,pieces),axis,(Tensors.with_workers(~Element,Tensors.full_count(~Element,shape,None{},fill),workers),0))),
        uncounted(~Element,packed(~Element,~Part,~part_value,pieces),axis,shape,fill,workers,accepted))

# ---- One cell of a concat along the middle axis ----

# Parts of extents [outer,extent,inner] appended along the middle axis, each
# with its count of cells.
type Layers<-Part: Data> is Data:
  Alone{part: Part,extent: U32,count: U32}
  Before{part: Part,extent: U32,count: U32,rest: Layers<Part>}

def lined(~Part: Data,layers: Layers<Part>,+outer: U32,+inner: U32) -> Sources<Part>:
  match layers:
    case Alone{part,extent,count}: Final{part,[outer,extent,inner]}
    case Before{part,extent,count,rest}: Then{part,[outer,extent,inner],lined(~Part,rest,outer,inner)}

def layers_total(~Part: Data,layers: Layers<Part>) -> Nat:
  match layers:
    case Alone{part,extent,count}: U32.to_nat(extent)
    case Before{part,extent,count,rest}: Nat.add(U32.to_nat(extent),layers_total(~Part,rest))

def Counted(~Part: Data,layers: Layers<Part>,+outer: U32,+inner: U32) -> Data:
  match layers:
    case Alone{part,+extent,+count}: {Nat.mul(U32.to_nat(outer),Slices.cells(extent,inner)) == U32.to_nat(count) : Nat}
    case Before{part,+extent,+count,rest}: Shape.Both<{Nat.mul(U32.to_nat(outer),Slices.cells(extent,inner)) == U32.to_nat(count) : Nat},Counted(~Part,rest,outer,inner)>

def head_extent(~Part: Data,layers: Layers<Part>) -> U32:
  match layers:
    case Alone{part,extent,count}: extent
    case Before{part,extent,count,rest}: extent

# The cell of a line at a channel and an offset inside it.
def cell_at(+extent: U32,+inner: U32,+line: Nat,+channel: Nat,+offset: Nat) -> Nat:
  Nat.add(Nat.mul(line,Slices.cells(extent,inner)),Nat.add(Nat.mul(channel,U32.to_nat(inner)),offset))

# The cell at a channel of a line: the part whose extent covers the channel,
# read at that channel of its line.
def along(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,layers: Layers<Part>,+inner: U32,+line: Nat,+channel: Nat,+offset: Nat) -> Element:
  match layers:
    case Alone{+part,+extent,count}: part_value(part,U32.from_nat(cell_at(extent,inner,line,channel,offset)))
    case Before{+part,+extent,count,rest}:
      Bool.pick(Element,Nat.is_lt(channel,U32.to_nat(extent)),part_value(part,U32.from_nat(cell_at(extent,inner,line,channel,offset))),
        along(~Element,~Part,~part_value,rest,inner,line,Nat.sub(channel,U32.to_nat(extent)),offset))

# The target [outer,extent,inner] of the parts: its cells fit the capacity.
type Target<-outer: U32,-extent: U32,-inner: U32,-count: U32,-capacity: U32> is Data:
  Target{exact: {Nat.mul(U32.to_nat(outer),Slices.cells(extent,inner)) == U32.to_nat(count) : Nat},nonzero: {U32.is_zero(count) == False{} : Bool},
    capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool}}

def part_fit(+outer: U32,+extent: U32,+inner: U32,+count: U32,+capacity: U32,+target: Target<outer,extent,inner,count,capacity>,+start: U32,+part_extent: U32,
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),U32.to_nat(extent)) == True{} : Bool}) ->
  Slices.MiddleFit<outer,extent,inner,count,start,part_extent,capacity>:
  Target{exact,nonzero,capacity_ok,count_room} = target
  Slices.MiddleFit{exact,nonzero,room,capacity_ok,count_room}

# The next part starts where this one ends.
def next_start(+outer: U32,+extent: U32,+inner: U32,+count: U32,+capacity: U32,+target: Target<outer,extent,inner,count,capacity>,+start: U32,+part_extent: U32,
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),U32.to_nat(extent)) == True{} : Bool}) ->
  {U32.to_nat((start + part_extent : U32)) == Nat.add(U32.to_nat(start),U32.to_nat(part_extent)) : Nat}:
  Target{exact,nonzero,+capacity_ok,count_room} = target
  +fit = part_fit(outer,extent,inner,count,capacity,target,start,part_extent,room)
  +line_positive = DivisionOrder.right_factor_positive(U32.to_nat(outer),Slices.cells(extent,inner),Slices.fit_cells(outer,extent,inner,count,start,part_extent,capacity,fit))
  Arithmetic.add(start,part_extent,capacity,capacity_ok,
    Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),U32.to_nat(extent),U32.to_nat(capacity),room,
      Order.transitive(U32.to_nat(extent),Slices.cells(extent,inner),U32.to_nat(capacity),
        RowMajor.product_covers(U32.to_nat(extent),U32.to_nat(inner),DivisionOrder.right_factor_positive(U32.to_nat(extent),U32.to_nat(inner),line_positive)),
        Slices.fit_line(outer,extent,inner,count,start,part_extent,capacity,fit))))

# The rest of the parts after one part fits after it.
def rest_room(+start: Nat,+part_extent: Nat,+rest: Nat,+extent: Nat,room: {Nat.is_le(Nat.add(start,Nat.add(part_extent,rest)),extent) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(start,part_extent),rest),extent) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(Nat.add(start,part_extent),rest),Nat.add(start,Nat.add(part_extent,rest)),Addition.associative(start,part_extent,rest)) : {Nat.is_le(_,extent) == True{} : Bool}
  room

def first_room(+start: Nat,+part_extent: Nat,+rest: Nat,+extent: Nat,room: {Nat.is_le(Nat.add(start,Nat.add(part_extent,rest)),extent) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(start,part_extent),extent) == True{} : Bool}:
  Order.transitive(Nat.add(start,part_extent),Nat.add(Nat.add(start,part_extent),rest),extent,Order.le_add(Nat.add(start,part_extent),rest),rest_room(start,part_extent,rest,extent,room))

# The cells of channels below a limit lie below the cells of the limit.
def channel_cell(+channel: Nat,+limit: Nat,+inner: Nat,+offset: Nat,offset_inside: {Nat.is_lt(offset,inner) == True{} : Bool},
  below: {Nat.is_lt(channel,limit) == True{} : Bool}) -> {Nat.is_lt(Nat.add(Nat.mul(channel,inner),offset),Nat.mul(limit,inner)) == True{} : Bool}:
  Bounds.nat_trans(Nat.add(Nat.mul(channel,inner),offset),Nat.add(inner,Nat.mul(channel,inner)),Nat.mul(limit,inner),
    %Order.commutative(Nat.mul(channel,inner),inner) : {Nat.is_lt(Nat.add(Nat.mul(channel,inner),offset),_) == True{} : Bool}
    Order.offset_lt(Nat.mul(channel,inner),offset,inner,offset_inside),
    RowMajor.mul_le(1n+channel,limit,inner,Order.strict_follow(channel,limit,below)))

# The cells of a channel from a limit on lie past the cells of the limit.
def channel_past(+channel: Nat,+limit: Nat,+inner: Nat,+offset: Nat,past: {Nat.is_le(limit,channel) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(limit,inner),Nat.add(Nat.mul(channel,inner),offset)) == True{} : Bool}:
  Order.transitive(Nat.mul(limit,inner),Nat.mul(channel,inner),Nat.add(Nat.mul(channel,inner),offset),RowMajor.mul_le(limit,channel,inner,past),Order.le_add(Nat.mul(channel,inner),offset))

# A cell before the channels of the parts keeps its element.
def appended_below(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,layers: Layers<Part>,+outer: U32,+extent: U32,+inner: U32,+count: U32,+capacity: U32,
  +target: Target<outer,extent,inner,count,capacity>,+start: U32,+previous: Element,+line: Nat,+channel: Nat,+offset: Nat,+counted: Counted(~Part,layers,outer,inner),
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),layers_total(~Part,layers)),U32.to_nat(extent)) == True{} : Bool},
  +below: {Nat.is_lt(channel,U32.to_nat(start)) == True{} : Bool},+line_inside: {Nat.is_lt(line,U32.to_nat(outer)) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(inner)) == True{} : Bool}) ->
  {appended(~Element,~Part,~part_value,lined(~Part,layers,outer,inner),1,[outer,extent,inner],start,previous,U32.from_nat(cell_at(extent,inner,line,channel,offset))) == previous : Element}:
  match layers:
    case Alone{+part,+part_extent,+part_count}:
      +cell = Nat.add(Nat.mul(channel,U32.to_nat(inner)),offset)
      Slices.middle_miss(~Element,~Part,~part_value,outer,extent,inner,count,start,part_extent,capacity,part_fit(outer,extent,inner,count,capacity,target,start,part_extent,room),
        part_count,counted,part,previous,line,cell,Inl{channel_cell(channel,U32.to_nat(start),U32.to_nat(inner),offset,offset_inside,below)},line_inside,
        channel_cell(channel,U32.to_nat(extent),U32.to_nat(inner),offset,offset_inside,
          Bounds.nat_trans(channel,U32.to_nat(start),U32.to_nat(extent),below,
            Order.transitive(U32.to_nat(start),Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),U32.to_nat(extent),Order.le_add(U32.to_nat(start),U32.to_nat(part_extent)),room))))
    case Before{+part,+part_extent,+part_count,+rest}:
      Shape.Both{+exact,+rest_counted} = counted
      +cell = Nat.add(Nat.mul(channel,U32.to_nat(inner)),offset)
      +fit = first_room(U32.to_nat(start),U32.to_nat(part_extent),layers_total(~Part,rest),U32.to_nat(extent),room)
      +moved = next_start(outer,extent,inner,count,capacity,target,start,part_extent,fit)
      +next = (start + part_extent : U32)
      +query = U32.from_nat(cell_at(extent,inner,line,channel,offset))
      %Equal.sym(Element,Slices.slice_element(~Element,~Part,~part_value,Slices.slice_of(1,start,[outer,extent,inner],[outer,part_extent,inner]),part,previous,query),previous,
        Slices.middle_miss(~Element,~Part,~part_value,outer,extent,inner,count,start,part_extent,capacity,part_fit(outer,extent,inner,count,capacity,target,start,part_extent,fit),
          part_count,exact,part,previous,line,cell,Inl{channel_cell(channel,U32.to_nat(start),U32.to_nat(inner),offset,offset_inside,below)},line_inside,
          channel_cell(channel,U32.to_nat(extent),U32.to_nat(inner),offset,offset_inside,
            Bounds.nat_trans(channel,U32.to_nat(start),U32.to_nat(extent),below,
              Order.transitive(U32.to_nat(start),Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),U32.to_nat(extent),Order.le_add(U32.to_nat(start),U32.to_nat(part_extent)),fit))))) :
        {appended(~Element,~Part,~part_value,lined(~Part,rest,outer,inner),1,[outer,extent,inner],next,_,query) == previous : Element}
      appended_below(~Element,~Part,~part_value,rest,outer,extent,inner,count,capacity,target,next,previous,line,channel,offset,rest_counted,
        %Equal.sym(Nat,U32.to_nat(next),Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),moved) : {Nat.is_le(Nat.add(_,layers_total(~Part,rest)),U32.to_nat(extent)) == True{} : Bool}
        rest_room(U32.to_nat(start),U32.to_nat(part_extent),layers_total(~Part,rest),U32.to_nat(extent),room),
        %Equal.sym(Nat,U32.to_nat(next),Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),moved) : {Nat.is_lt(channel,_) == True{} : Bool}
        Bounds.nat_trans(channel,U32.to_nat(start),Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),below,Order.le_add(U32.to_nat(start),U32.to_nat(part_extent))),
        line_inside,offset_inside)

# The cell of a line at a channel past start, read through the slice of a part
# from start: the cell of the slice at that channel of the part.
def sliced_cell(+start: U32,+extent: U32,+inner: U32,+line: Nat,+channel: Nat,+offset: Nat) ->
  {cell_at(extent,inner,line,Nat.add(U32.to_nat(start),channel),offset) == Slices.middle_cell(start,extent,inner,line,Nat.add(Nat.mul(channel,U32.to_nat(inner)),offset)) : Nat}:
  +wide = U32.to_nat(inner)
  +lines = Nat.mul(line,Slices.cells(extent,inner))
  +opening = Nat.mul(U32.to_nat(start),wide)
  +local = Nat.add(Nat.mul(channel,wide),offset)
  Equal.trans(Nat,Nat.add(lines,Nat.add(Nat.mul(Nat.add(U32.to_nat(start),channel),wide),offset)),Nat.add(lines,Nat.add(opening,local)),Nat.add(Nat.add(opening,lines),local),
    Equal.cong(Nat,Nat,value => Nat.add(lines,value),Nat.add(Nat.mul(Nat.add(U32.to_nat(start),channel),wide),offset),Nat.add(opening,local),
      %Equal.sym(Nat,Nat.mul(Nat.add(U32.to_nat(start),channel),wide),Nat.add(opening,Nat.mul(channel,wide)),RowMajor.distrib(U32.to_nat(start),channel,wide)) :
        {Nat.add(_,offset) == Nat.add(opening,local) : Nat}
      Addition.associative(opening,Nat.mul(channel,wide),offset)),
    Equal.trans(Nat,Nat.add(lines,Nat.add(opening,local)),Nat.add(Nat.add(lines,opening),local),Nat.add(Nat.add(opening,lines),local),
      Equal.sym(Nat,Nat.add(Nat.add(lines,opening),local),Nat.add(lines,Nat.add(opening,local)),Addition.associative(lines,opening,local)),
      Equal.cong(Nat,Nat,value => Nat.add(value,local),Nat.add(lines,opening),Nat.add(opening,lines),Order.commutative(lines,opening))))

# The cell at a channel past start of a line holds the cell of the part whose
# extent covers the channel, at that channel of the part: the parts before it
# leave the cell, it writes the cell, and the parts after it start past it.
def appended_middle(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,layers: Layers<Part>,chosen: Bool,+outer: U32,+extent: U32,+inner: U32,+count: U32,+capacity: U32,
  +target: Target<outer,extent,inner,count,capacity>,+start: U32,+previous: Element,+line: Nat,+channel: Nat,+offset: Nat,+counted: Counted(~Part,layers,outer,inner),
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),layers_total(~Part,layers)),U32.to_nat(extent)) == True{} : Bool},
  +line_inside: {Nat.is_lt(line,U32.to_nat(outer)) == True{} : Bool},+inside: {Nat.is_lt(channel,layers_total(~Part,layers)) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(inner)) == True{} : Bool},
  +classified: {Nat.is_lt(channel,U32.to_nat(head_extent(~Part,layers))) == chosen : Bool}) ->
  {appended(~Element,~Part,~part_value,lined(~Part,layers,outer,inner),1,[outer,extent,inner],start,previous,
    U32.from_nat(cell_at(extent,inner,line,Nat.add(U32.to_nat(start),channel),offset))) == along(~Element,~Part,~part_value,layers,inner,line,channel,offset) : Element}:
  match layers chosen:
    case Alone{+part,+part_extent,+part_count} chosen:
      +local = Nat.add(Nat.mul(channel,U32.to_nat(inner)),offset)
      %Equal.sym(Nat,cell_at(extent,inner,line,Nat.add(U32.to_nat(start),channel),offset),Slices.middle_cell(start,extent,inner,line,Nat.add(Nat.mul(channel,U32.to_nat(inner)),offset)),
        sliced_cell(start,extent,inner,line,channel,offset)) :
        {Slices.slice_element(~Element,~Part,~part_value,Slices.slice_of(1,start,[outer,extent,inner],[outer,part_extent,inner]),part,previous,U32.from_nat(_)) ==
          part_value(part,U32.from_nat(cell_at(part_extent,inner,line,channel,offset))) : Element}
      Slices.middle_hit(~Element,~Part,~part_value,outer,extent,inner,count,start,part_extent,capacity,part_fit(outer,extent,inner,count,capacity,target,start,part_extent,room),
        part_count,counted,part,previous,line,local,line_inside,channel_cell(channel,U32.to_nat(part_extent),U32.to_nat(inner),offset,offset_inside,inside))
    case Before{+part,+part_extent,+part_count,+rest} True{}:
      Shape.Both{+exact,+rest_counted} = counted
      +local = Nat.add(Nat.mul(channel,U32.to_nat(inner)),offset)
      +fit = first_room(U32.to_nat(start),U32.to_nat(part_extent),layers_total(~Part,rest),U32.to_nat(extent),room)
      +moved = next_start(outer,extent,inner,count,capacity,target,start,part_extent,fit)
      +next = (start + part_extent : U32)
      +query = U32.from_nat(cell_at(extent,inner,line,Nat.add(U32.to_nat(start),channel),offset))
      +value = part_value(part,U32.from_nat(cell_at(part_extent,inner,line,channel,offset)))
      +slice = Slices.slice_element(~Element,~Part,~part_value,Slices.slice_of(1,start,[outer,extent,inner],[outer,part_extent,inner]),part,previous,query)
      +hit = {
        %Equal.sym(Nat,cell_at(extent,inner,line,Nat.add(U32.to_nat(start),channel),offset),Slices.middle_cell(start,extent,inner,line,Nat.add(Nat.mul(channel,U32.to_nat(inner)),offset)),
        sliced_cell(start,extent,inner,line,channel,offset)) :
          {Slices.slice_element(~Element,~Part,~part_value,Slices.slice_of(1,start,[outer,extent,inner],[outer,part_extent,inner]),part,previous,U32.from_nat(_)) == value : Element}
        Slices.middle_hit(~Element,~Part,~part_value,outer,extent,inner,count,start,part_extent,capacity,part_fit(outer,extent,inner,count,capacity,target,start,part_extent,fit),
          part_count,exact,part,previous,line,local,line_inside,channel_cell(channel,U32.to_nat(part_extent),U32.to_nat(inner),offset,offset_inside,classified))
        : {slice == value : Element}}
      %Equal.sym(Bool,Nat.is_lt(channel,U32.to_nat(part_extent)),True{},classified) :
        {appended(~Element,~Part,~part_value,lined(~Part,rest,outer,inner),1,[outer,extent,inner],next,slice,query) ==
          Bool.pick(Element,_,value,along(~Element,~Part,~part_value,rest,inner,line,Nat.sub(channel,U32.to_nat(part_extent)),offset)) : Element}
      %Equal.sym(Element,slice,value,hit) : {appended(~Element,~Part,~part_value,lined(~Part,rest,outer,inner),1,[outer,extent,inner],next,_,query) == value : Element}
      appended_below(~Element,~Part,~part_value,rest,outer,extent,inner,count,capacity,target,next,value,line,Nat.add(U32.to_nat(start),channel),offset,rest_counted,
        %Equal.sym(Nat,U32.to_nat(next),Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),moved) : {Nat.is_le(Nat.add(_,layers_total(~Part,rest)),U32.to_nat(extent)) == True{} : Bool}
        rest_room(U32.to_nat(start),U32.to_nat(part_extent),layers_total(~Part,rest),U32.to_nat(extent),room),
        %Equal.sym(Nat,U32.to_nat(next),Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),moved) : {Nat.is_lt(Nat.add(U32.to_nat(start),channel),_) == True{} : Bool}
        Order.offset_lt(U32.to_nat(start),channel,U32.to_nat(part_extent),classified),
        line_inside,offset_inside)
    case Before{+part,+part_extent,+part_count,+rest} False{}:
      Shape.Both{+exact,+rest_counted} = counted
      +whole = Nat.add(U32.to_nat(start),channel)
      +cell = Nat.add(Nat.mul(whole,U32.to_nat(inner)),offset)
      +fit = first_room(U32.to_nat(start),U32.to_nat(part_extent),layers_total(~Part,rest),U32.to_nat(extent),room)
      +moved = next_start(outer,extent,inner,count,capacity,target,start,part_extent,fit)
      +next = (start + part_extent : U32)
      +later = Order.not_less(channel,U32.to_nat(part_extent),classified)
      +remaining = Nat.sub(channel,U32.to_nat(part_extent))
      +restored = Order.restore_shift(U32.to_nat(part_extent),channel,later)
      +query = U32.from_nat(cell_at(extent,inner,line,whole,offset))
      +slice = Slices.slice_element(~Element,~Part,~part_value,Slices.slice_of(1,start,[outer,extent,inner],[outer,part_extent,inner]),part,previous,query)
      +whole_inside = Order.offset_lt(U32.to_nat(start),channel,layers_total(~Part,Before{part,part_extent,part_count,rest}),inside)
      +miss = Slices.middle_miss(~Element,~Part,~part_value,outer,extent,inner,count,start,part_extent,capacity,part_fit(outer,extent,inner,count,capacity,target,start,part_extent,fit),
        part_count,exact,part,previous,line,cell,
        Inr{%RowMajor.distrib(U32.to_nat(start),U32.to_nat(part_extent),U32.to_nat(inner)) : {Nat.is_le(_,cell) == True{} : Bool}
          channel_past(whole,Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),U32.to_nat(inner),offset,Order.offset_le(U32.to_nat(start),U32.to_nat(part_extent),channel,later))},
        line_inside,
        channel_cell(whole,U32.to_nat(extent),U32.to_nat(inner),offset,offset_inside,Bounds.nat_trans(whole,Nat.add(U32.to_nat(start),layers_total(~Part,Before{part,part_extent,part_count,rest})),U32.to_nat(extent),whole_inside,room)))
      %Equal.sym(Bool,Nat.is_lt(channel,U32.to_nat(part_extent)),False{},classified) :
        {appended(~Element,~Part,~part_value,lined(~Part,rest,outer,inner),1,[outer,extent,inner],next,slice,query) ==
          Bool.pick(Element,_,part_value(part,U32.from_nat(cell_at(part_extent,inner,line,channel,offset))),along(~Element,~Part,~part_value,rest,inner,line,remaining,offset)) : Element}
      %Equal.sym(Element,slice,previous,miss) :
        {appended(~Element,~Part,~part_value,lined(~Part,rest,outer,inner),1,[outer,extent,inner],next,_,query) == along(~Element,~Part,~part_value,rest,inner,line,remaining,offset) : Element}
      %restored :
        {appended(~Element,~Part,~part_value,lined(~Part,rest,outer,inner),1,[outer,extent,inner],next,previous,U32.from_nat(cell_at(extent,inner,line,Nat.add(U32.to_nat(start),_),offset))) ==
          along(~Element,~Part,~part_value,rest,inner,line,remaining,offset) : Element}
      %Addition.associative(U32.to_nat(start),U32.to_nat(part_extent),remaining) :
        {appended(~Element,~Part,~part_value,lined(~Part,rest,outer,inner),1,[outer,extent,inner],next,previous,U32.from_nat(cell_at(extent,inner,line,_,offset))) ==
          along(~Element,~Part,~part_value,rest,inner,line,remaining,offset) : Element}
      %moved :
        {appended(~Element,~Part,~part_value,lined(~Part,rest,outer,inner),1,[outer,extent,inner],next,previous,U32.from_nat(cell_at(extent,inner,line,Nat.add(_,remaining),offset))) ==
          along(~Element,~Part,~part_value,rest,inner,line,remaining,offset) : Element}
      appended_middle(~Element,~Part,~part_value,rest,Nat.is_lt(remaining,U32.to_nat(head_extent(~Part,rest))),outer,extent,inner,count,capacity,target,next,previous,line,remaining,offset,rest_counted,
        %Equal.sym(Nat,U32.to_nat(next),Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),moved) : {Nat.is_le(Nat.add(_,layers_total(~Part,rest)),U32.to_nat(extent)) == True{} : Bool}
        rest_room(U32.to_nat(start),U32.to_nat(part_extent),layers_total(~Part,rest),U32.to_nat(extent),room),
        line_inside,
        Order.cancel_offset_lt(U32.to_nat(part_extent),remaining,layers_total(~Part,rest),
          %Equal.sym(Nat,Nat.add(U32.to_nat(part_extent),remaining),channel,restored) : {Nat.is_lt(_,Nat.add(U32.to_nat(part_extent),layers_total(~Part,rest))) == True{} : Bool} inside),
        offset_inside,{==})
