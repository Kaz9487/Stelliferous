# Three Normal tensors whose extents broadcast to counted common extents
# select into a Normal tensor of those extents. At every flat index it holds
# the yes value where the mask is nonzero and the no value elsewhere, each
# operand read as it is or through the view its extents stretch to.
import Base
import ../tensor.bend as Tensors
import ../tensor_f32.bend as F
import ../tensor_select.bend as Select
import ../tensor_band.bend as Bands
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_global_shape_count.bend as Global
import ./tensor_gather_values.bend as GatherValues
import ./tensor_broadcast_public.bend as Broadcast
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_band_spread.bend as Spread
import ./tensor_select_public.bend as SelectPublic
import ./tensor_select_band_values.bend as SelectValues
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_unit_broadcast.bend as UnitBroadcast
import ./boolean_logic.bend as Boolean

# The mask, yes and no operands, each read at the common extents.
type Selection<-Logical: Data> is Data:
  Selection{mask: Broadcast.Expanded<Logical>,yes: Broadcast.Expanded<Logical>,no: Broadcast.Expanded<Logical>}

def mask_value(~Logical: Data,~reader: Logical -> U32 -> F32,context: Selection<Logical>,index: U32) -> F32:
  Selection{mask,yes,no} = context
  Broadcast.value(~F32,~Logical,~reader,mask,index)

def yes_value(~Logical: Data,~reader: Logical -> U32 -> F32,context: Selection<Logical>,index: U32) -> F32:
  Selection{mask,yes,no} = context
  Broadcast.value(~F32,~Logical,~reader,yes,index)

def no_value(~Logical: Data,~reader: Logical -> U32 -> F32,context: Selection<Logical>,index: U32) -> F32:
  Selection{mask,yes,no} = context
  Broadcast.value(~F32,~Logical,~reader,no,index)

def mask_of(~Logical: Data,context: Selection<Logical>) -> Broadcast.Expanded<Logical>:
  Selection{mask,yes,no} = context
  mask

def yes_of(~Logical: Data,context: Selection<Logical>) -> Broadcast.Expanded<Logical>:
  Selection{mask,yes,no} = context
  yes

def no_of(~Logical: Data,context: Selection<Logical>) -> Broadcast.Expanded<Logical>:
  Selection{mask,yes,no} = context
  no

def Mask(~Logical: Data,before: Broadcast.Expanded<Logical>,after: Selection<Logical>) -> Data:
  {mask_of(~Logical,after) == before : Broadcast.Expanded<Logical>}

def Yes(~Logical: Data,before: Broadcast.Expanded<Logical>,after: Selection<Logical>) -> Data:
  {yes_of(~Logical,after) == before : Broadcast.Expanded<Logical>}

def No(~Logical: Data,before: Broadcast.Expanded<Logical>,after: Selection<Logical>) -> Data:
  {no_of(~Logical,after) == before : Broadcast.Expanded<Logical>}

def mask_agrees(~Logical: Data,~reader: Logical -> U32 -> F32,+before: Broadcast.Expanded<Logical>,+after: Selection<Logical>,+related: Mask(~Logical,before,after),+index: U32) ->
  {Broadcast.value(~F32,~Logical,~reader,before,index) == mask_value(~Logical,~reader,after,index) : F32}:
  match after:
    case Selection{+mask,+yes,+no}:
      %related : {Broadcast.value(~F32,~Logical,~reader,_,index) == Broadcast.value(~F32,~Logical,~reader,mask,index) : F32}
      {==}

def yes_agrees(~Logical: Data,~reader: Logical -> U32 -> F32,+before: Broadcast.Expanded<Logical>,+after: Selection<Logical>,+related: Yes(~Logical,before,after),+index: U32) ->
  {Broadcast.value(~F32,~Logical,~reader,before,index) == yes_value(~Logical,~reader,after,index) : F32}:
  match after:
    case Selection{+mask,+yes,+no}:
      %related : {Broadcast.value(~F32,~Logical,~reader,_,index) == Broadcast.value(~F32,~Logical,~reader,yes,index) : F32}
      {==}

def no_agrees(~Logical: Data,~reader: Logical -> U32 -> F32,+before: Broadcast.Expanded<Logical>,+after: Selection<Logical>,+related: No(~Logical,before,after),+index: U32) ->
  {Broadcast.value(~F32,~Logical,~reader,before,index) == no_value(~Logical,~reader,after,index) : F32}:
  match after:
    case Selection{+mask,+yes,+no}:
      %related : {Broadcast.value(~F32,~Logical,~reader,_,index) == Broadcast.value(~F32,~Logical,~reader,no,index) : F32}
      {==}

# The contract of a selection of three operands read at the common extents.
def Selected(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +mask_context: Broadcast.Expanded<Logical>,+yes_context: Broadcast.Expanded<Logical>,+no_context: Broadcast.Expanded<Logical>,-tensor: Tensors.Tensor<F32>) -> Data:
  Global.Contract(~F32,~Selection<Logical>,~SelectValues.value(~Selection<Logical>,~mask_value(~Logical,~mv),~yes_value(~Logical,~yv),~no_value(~Logical,~nv)),
    Selection{mask_context,yes_context,no_context},tensor)

# An operand brought to the common extents is a tensor of those extents that
# holds its values read through its expansion: itself when it has them,
# otherwise its repetition to them.
def expanded(~Logical: Data,~reader: Logical -> U32 -> F32,same: Bool,+logical: Logical,+source: List<&2,U32>,+shape: List<&2,U32>,
  +model: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,+count: U32,
  +classified: {Shapes.same_extents(source,shape) == same : Bool},+counted: Global.Counted<shape>,
  +exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},stretched: Broadcast.Stretched<source,count,shape>,
  contract: Global.Contract(~F32,~Logical,~reader,logical,SelectPublic.tensor(model,geometry,source,fill,workers))) ->
  Global.Shaped(~F32,~Broadcast.Expanded<Logical>,~Broadcast.value(~F32,~Logical,~reader),Broadcast.Expanded{logical,Broadcast.expansion_when(same,source,shape)},shape,Select.expanded(same,shape,SelectPublic.tensor(model,geometry,source,fill,workers))):
  match same:
    case True{}:
      +equal = ShapeLaws.same_extents(source,shape,classified)
      %equal : Global.Shaped(~F32,~Broadcast.Expanded<Logical>,~Broadcast.value(~F32,~Logical,~reader),Broadcast.Expanded{logical,None{}},_,SelectPublic.tensor(model,geometry,source,fill,workers))
      Global.shaped(~F32,~Broadcast.Expanded<Logical>,~Broadcast.value(~F32,~Logical,~reader),Broadcast.Expanded{logical,None{}},source,SelectPublic.tensor(model,geometry,source,fill,workers),{==},{==},
        Broadcast.plain(~F32,~Logical,~reader,logical,Model.pack(~F32,model),geometry,source,fill,workers,contract))
    case False{}:
      Broadcast.Stretched{+full,+equation} = stretched
      +view = {Views.View{full,0} : Views.View}
      +viewed = Broadcast.stretched_view(source,count,shape,full,exact,equation)
      %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(source),shape),Some{view},viewed) :
        Global.Shaped(~F32,~Broadcast.Expanded<Logical>,~Broadcast.value(~F32,~Logical,~reader),Broadcast.Expanded{logical,_},shape,Select.expanded(False{},shape,SelectPublic.tensor(model,geometry,source,fill,workers)))
      Broadcast.broadcast_to(~F32,~Logical,~reader,logical,source,shape,view,fill,workers,model,geometry,contract,counted,viewed)

# Operands brought to the common extents meet and select.
def selected(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +mask_context: Broadcast.Expanded<Logical>,+yes_context: Broadcast.Expanded<Logical>,+no_context: Broadcast.Expanded<Logical>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  -mask: Tensors.Tensor<F32>,-yes: Tensors.Tensor<F32>,-no: Tensors.Tensor<F32>,
  mask_shaped: Global.Shaped(~F32,~Broadcast.Expanded<Logical>,~Broadcast.value(~F32,~Logical,~mv),mask_context,shape,mask),yes_shaped: Global.Shaped(~F32,~Broadcast.Expanded<Logical>,~Broadcast.value(~F32,~Logical,~yv),yes_context,shape,yes),
  no_shaped: Global.Shaped(~F32,~Broadcast.Expanded<Logical>,~Broadcast.value(~F32,~Logical,~nv),no_context,shape,no)) ->
  Selected(~Logical,~mv,~yv,~nv,mask_context,yes_context,no_context,Select.matching(shape,fill,workers,mask,yes,no)):
  Shape.Both{mask_usable,mask_contract} = mask_shaped
  Global.Usable{+mm,+mg,+mf,+mw,+me} = mask_usable
  Shape.Both{yes_usable,yes_contract} = yes_shaped
  Global.Usable{+ym,+yg,+yf,+yw,+ye} = yes_usable
  Shape.Both{no_usable,no_contract} = no_shaped
  Global.Usable{+nm,+ng,+nf,+nw,+ne} = no_usable
  +context = {Selection{mask_context,yes_context,no_context} : Selection<Logical>}
  %me :
    Selected(~Logical,~mv,~yv,~nv,mask_context,yes_context,no_context,Select.matching(shape,fill,workers,_,yes,no))
  %ye :
    Selected(~Logical,~mv,~yv,~nv,mask_context,yes_context,no_context,Select.matching(shape,fill,workers,SelectPublic.tensor(mm,mg,shape,mf,mw),_,no))
  %ne :
    Selected(~Logical,~mv,~yv,~nv,mask_context,yes_context,no_context,
      Select.matching(shape,fill,workers,SelectPublic.tensor(mm,mg,shape,mf,mw),SelectPublic.tensor(ym,yg,shape,yf,yw),_))
  SelectPublic.offered(~Selection<Logical>,~mask_value(~Logical,~mv),~yes_value(~Logical,~yv),~no_value(~Logical,~nv),context,shape,mm,ym,nm,mg,yg,ng,mf,yf,nf,mw,yw,nw,fill,workers,
    Global.relabeled(~F32,~Broadcast.Expanded<Logical>,~Selection<Logical>,~Broadcast.value(~F32,~Logical,~mv),~mask_value(~Logical,~mv),~Mask(~Logical),~mask_agrees(~Logical,~mv),
      mask_context,context,{==},Tensors.Tensor{Model.pack(~F32,mm),mg,shape,mf,Tensors.Usable{},mw},
      %Equal.sym(Tensors.Tensor<F32>,SelectPublic.tensor(mm,mg,shape,mf,mw),mask,me) :
        Global.Contract(~F32,~Broadcast.Expanded<Logical>,~Broadcast.value(~F32,~Logical,~mv),mask_context,_)
      mask_contract),
    Global.relabeled(~F32,~Broadcast.Expanded<Logical>,~Selection<Logical>,~Broadcast.value(~F32,~Logical,~yv),~yes_value(~Logical,~yv),~Yes(~Logical),~yes_agrees(~Logical,~yv),
      yes_context,context,{==},Tensors.Tensor{Model.pack(~F32,ym),yg,shape,yf,Tensors.Usable{},yw},
      %Equal.sym(Tensors.Tensor<F32>,SelectPublic.tensor(ym,yg,shape,yf,yw),yes,ye) :
        Global.Contract(~F32,~Broadcast.Expanded<Logical>,~Broadcast.value(~F32,~Logical,~yv),yes_context,_)
      yes_contract),
    Global.relabeled(~F32,~Broadcast.Expanded<Logical>,~Selection<Logical>,~Broadcast.value(~F32,~Logical,~nv),~no_value(~Logical,~nv),~No(~Logical),~no_agrees(~Logical,~nv),
      no_context,context,{==},Tensors.Tensor{Model.pack(~F32,nm),ng,shape,nf,Tensors.Usable{},nw},
      %Equal.sym(Tensors.Tensor<F32>,SelectPublic.tensor(nm,ng,shape,nf,nw),no,ne) :
        Global.Contract(~F32,~Broadcast.Expanded<Logical>,~Broadcast.value(~F32,~Logical,~nv),no_context,_)
      no_contract))

# Each operand is brought to the common extents and the three select.
def expanded_selection(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +mask_context: Logical,+yes_context: Logical,+no_context: Logical,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,+shape: List<&2,U32>,
  +mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,+mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,+mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,
  +mask_count: U32,+yes_count: U32,+no_count: U32,+counted: Global.Counted<shape>,
  +mask_exact: {ShapeLaws.elements(mask_shape) == U32.to_nat(mask_count) : Nat},+yes_exact: {ShapeLaws.elements(yes_shape) == U32.to_nat(yes_count) : Nat},
  +no_exact: {ShapeLaws.elements(no_shape) == U32.to_nat(no_count) : Nat},
  mask_stretched: Broadcast.Stretched<mask_shape,mask_count,shape>,values_stretched: Shape.Both<Broadcast.Stretched<yes_shape,yes_count,shape>,Broadcast.Stretched<no_shape,no_count,shape>>,
  mask_contract: Global.Contract(~F32,~Logical,~mv,mask_context,SelectPublic.tensor(mask,mg,mask_shape,mf,mw)),
  yes_contract: Global.Contract(~F32,~Logical,~yv,yes_context,SelectPublic.tensor(yes,yg,yes_shape,yf,yw)),
  no_contract: Global.Contract(~F32,~Logical,~nv,no_context,SelectPublic.tensor(no,ng,no_shape,nf,nw))) ->
  Selected(~Logical,~mv,~yv,~nv,Broadcast.Expanded{mask_context,Broadcast.expansion(mask_shape,shape)},Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},
    Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
    Select.matching(shape,yf,U32.max(mw,U32.max(yw,nw)),
      Select.expanded(Shapes.same_extents(mask_shape,shape),shape,SelectPublic.tensor(mask,mg,mask_shape,mf,mw)),
      Select.expanded(Shapes.same_extents(yes_shape,shape),shape,SelectPublic.tensor(yes,yg,yes_shape,yf,yw)),
      Select.expanded(Shapes.same_extents(no_shape,shape),shape,SelectPublic.tensor(no,ng,no_shape,nf,nw)))):
  Shape.Both{yes_stretched,no_stretched} = values_stretched
  selected(~Logical,~mv,~yv,~nv,Broadcast.Expanded{mask_context,Broadcast.expansion(mask_shape,shape)},Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},
    Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},shape,yf,U32.max(mw,U32.max(yw,nw)),
    Select.expanded(Shapes.same_extents(mask_shape,shape),shape,SelectPublic.tensor(mask,mg,mask_shape,mf,mw)),
    Select.expanded(Shapes.same_extents(yes_shape,shape),shape,SelectPublic.tensor(yes,yg,yes_shape,yf,yw)),
    Select.expanded(Shapes.same_extents(no_shape,shape),shape,SelectPublic.tensor(no,ng,no_shape,nf,nw)),
    expanded(~Logical,~mv,Shapes.same_extents(mask_shape,shape),mask_context,mask_shape,shape,mask,mg,mf,mw,mask_count,{==},counted,mask_exact,mask_stretched,mask_contract),
    expanded(~Logical,~yv,Shapes.same_extents(yes_shape,shape),yes_context,yes_shape,shape,yes,yg,yf,yw,yes_count,{==},counted,yes_exact,yes_stretched,yes_contract),
    expanded(~Logical,~nv,Shapes.same_extents(no_shape,shape),no_context,no_shape,shape,no,ng,nf,nw,no_count,{==},counted,no_exact,no_stretched,no_contract))

# Some of one shape is Some of another only when the two are one.
def found(+shape: List<&2,U32>,+target: List<&2,U32>,equation: {Some{shape} == Some{target} : Maybe<&2,List<&2,U32>>}) -> {shape == target : List<&2,U32>}:
  Equal.cong(Maybe<&2,List<&2,U32>>,List<&2,U32>,result => Maybe.default(&2,List<&2,U32>,result,shape),Some{shape},Some{target},equation)

# Operands of equal extents have those extents in common.
def same_common(+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,+values: List<&2,U32>,+shape: List<&2,U32>,
  +mask_yes: {mask_shape == yes_shape : List<&2,U32>},+yes_no: {yes_shape == no_shape : List<&2,U32>},
  values_common: {Shapes.broadcast(yes_shape,no_shape) == Some{values} : Maybe<&2,List<&2,U32>>},common: {Shapes.broadcast(mask_shape,values) == Some{shape} : Maybe<&2,List<&2,U32>>}) ->
  {yes_shape == shape : List<&2,U32>}:
  +values_yes = found(yes_shape,values,Equal.trans(Maybe<&2,List<&2,U32>>,Some{yes_shape},Shapes.broadcast(yes_shape,yes_shape),Some{values},
    Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(yes_shape,yes_shape),Some{yes_shape},UnitBroadcast.self_broadcast(yes_shape)),
    %Equal.sym(List<&2,U32>,yes_shape,no_shape,yes_no) : {Shapes.broadcast(yes_shape,_) == Some{values} : Maybe<&2,List<&2,U32>>} values_common))
  found(yes_shape,shape,Equal.trans(Maybe<&2,List<&2,U32>>,Some{yes_shape},Shapes.broadcast(yes_shape,yes_shape),Some{shape},
    Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(yes_shape,yes_shape),Some{yes_shape},UnitBroadcast.self_broadcast(yes_shape)),
    %mask_yes : {Shapes.broadcast(_,yes_shape) == Some{shape} : Maybe<&2,List<&2,U32>>}
    %Equal.sym(List<&2,U32>,yes_shape,values,values_yes) : {Shapes.broadcast(mask_shape,_) == Some{shape} : Maybe<&2,List<&2,U32>>}
    common))

def same_shape(+source: List<&2,U32>,+shape: List<&2,U32>,equal: {source == shape : List<&2,U32>}) -> {Shapes.same_extents(source,shape) == True{} : Bool}:
  %Equal.sym(List<&2,U32>,source,shape,equal) : {Shapes.same_extents(_,shape) == True{} : Bool}
  ShapeLaws.same_extents_reflexive(shape)

# Operands of equal extents select as they are, like operands brought to
# their common extents.
def operands(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,same: Bool,
  +mask_context: Logical,+yes_context: Logical,+no_context: Logical,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,+values: List<&2,U32>,+shape: List<&2,U32>,
  +mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,+mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,+mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,
  +mask_count: U32,+yes_count: U32,+no_count: U32,+count: U32,
  +classified: {Shapes.same_extents(mask_shape,yes_shape) && Shapes.same_extents(yes_shape,no_shape) == same : Bool},
  +values_common: {Shapes.broadcast(yes_shape,no_shape) == Some{values} : Maybe<&2,List<&2,U32>>},+common: {Shapes.broadcast(mask_shape,values) == Some{shape} : Maybe<&2,List<&2,U32>>},
  +counted: Global.Counted<shape>,
  +mask_exact: {ShapeLaws.elements(mask_shape) == U32.to_nat(mask_count) : Nat},+yes_exact: {ShapeLaws.elements(yes_shape) == U32.to_nat(yes_count) : Nat},
  +no_exact: {ShapeLaws.elements(no_shape) == U32.to_nat(no_count) : Nat},
  mask_stretched: Broadcast.Stretched<mask_shape,mask_count,shape>,values_stretched: Shape.Both<Broadcast.Stretched<yes_shape,yes_count,shape>,Broadcast.Stretched<no_shape,no_count,shape>>,
  mask_contract: Global.Contract(~F32,~Logical,~mv,mask_context,SelectPublic.tensor(mask,mg,mask_shape,mf,mw)),
  yes_contract: Global.Contract(~F32,~Logical,~yv,yes_context,SelectPublic.tensor(yes,yg,yes_shape,yf,yw)),
  no_contract: Global.Contract(~F32,~Logical,~nv,no_context,SelectPublic.tensor(no,ng,no_shape,nf,nw))) ->
  Selected(~Logical,~mv,~yv,~nv,Broadcast.Expanded{mask_context,Broadcast.expansion(mask_shape,shape)},Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},
    Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
    Select.shapes(same,shape,Some{count},mask_shape,yes_shape,no_shape,yf,U32.max(mw,U32.max(yw,nw)),
      SelectPublic.tensor(mask,mg,mask_shape,mf,mw),SelectPublic.tensor(yes,yg,yes_shape,yf,yw),SelectPublic.tensor(no,ng,no_shape,nf,nw))):
  match same:
    case True{}:
      +flag = Shapes.same_extents(mask_shape,yes_shape)
      +mask_yes = ShapeLaws.same_extents(mask_shape,yes_shape,Boolean.left(flag,Shapes.same_extents(yes_shape,no_shape),classified))
      +yes_no = ShapeLaws.same_extents(yes_shape,no_shape,Boolean.right(flag,Shapes.same_extents(yes_shape,no_shape),classified))
      +yes_same = same_common(mask_shape,yes_shape,no_shape,values,shape,mask_yes,yes_no,values_common,common)
      +workers = U32.max(mw,U32.max(yw,nw))
      +contexts = {Broadcast.Expanded{mask_context,Broadcast.expansion(mask_shape,shape)} : Broadcast.Expanded<Logical>}
      %same_shape(mask_shape,shape,Equal.trans(List<&2,U32>,mask_shape,yes_shape,shape,mask_yes,yes_same)) :
        Selected(~Logical,~mv,~yv,~nv,contexts,Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
          Select.matching(shape,yf,workers,Select.expanded(_,shape,SelectPublic.tensor(mask,mg,mask_shape,mf,mw)),SelectPublic.tensor(yes,yg,yes_shape,yf,yw),SelectPublic.tensor(no,ng,no_shape,nf,nw)))
      %same_shape(yes_shape,shape,yes_same) :
        Selected(~Logical,~mv,~yv,~nv,contexts,Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
          Select.matching(shape,yf,workers,Select.expanded(Shapes.same_extents(mask_shape,shape),shape,SelectPublic.tensor(mask,mg,mask_shape,mf,mw)),Select.expanded(_,shape,SelectPublic.tensor(yes,yg,yes_shape,yf,yw)),SelectPublic.tensor(no,ng,no_shape,nf,nw)))
      %same_shape(no_shape,shape,Equal.trans(List<&2,U32>,no_shape,yes_shape,shape,Equal.sym(List<&2,U32>,yes_shape,no_shape,yes_no),yes_same)) :
        Selected(~Logical,~mv,~yv,~nv,contexts,Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
          Select.matching(shape,yf,workers,Select.expanded(Shapes.same_extents(mask_shape,shape),shape,SelectPublic.tensor(mask,mg,mask_shape,mf,mw)),Select.expanded(Shapes.same_extents(yes_shape,shape),shape,SelectPublic.tensor(yes,yg,yes_shape,yf,yw)),
            Select.expanded(_,shape,SelectPublic.tensor(no,ng,no_shape,nf,nw))))
      expanded_selection(~Logical,~mv,~yv,~nv,mask_context,yes_context,no_context,mask_shape,yes_shape,no_shape,shape,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,
        mask_count,yes_count,no_count,counted,mask_exact,yes_exact,no_exact,mask_stretched,values_stretched,mask_contract,yes_contract,no_contract)
    case False{}:
      expanded_selection(~Logical,~mv,~yv,~nv,mask_context,yes_context,no_context,mask_shape,yes_shape,no_shape,shape,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,
        mask_count,yes_count,no_count,counted,mask_exact,yes_exact,no_exact,mask_stretched,values_stretched,mask_contract,yes_contract,no_contract)

# The yes and no operands stretch to their common extents, which stretch to
# the common extents of all three.
def values_stretched(+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,+values: List<&2,U32>,+shape: List<&2,U32>,+yes_count: U32,+no_count: U32,
  +yes_exact: {ShapeLaws.elements(yes_shape) == U32.to_nat(yes_count) : Nat},+no_exact: {ShapeLaws.elements(no_shape) == U32.to_nat(no_count) : Nat},
  ranks: Shape.Both<{List.length(&2,U32,values) == Nat.add(Nat.sub(List.length(&2,U32,no_shape),List.length(&2,U32,yes_shape)),List.length(&2,U32,yes_shape)) : Nat},
    {List.length(&2,U32,values) == Nat.add(Nat.sub(List.length(&2,U32,yes_shape),List.length(&2,U32,no_shape)),List.length(&2,U32,no_shape)) : Nat}>,
  +values_rank: {Nat.is_le(List.length(&2,U32,values),List.length(&2,U32,shape)) == True{} : Bool},
  +outer: {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(Views.with_rank(Broadcast.axes_of(values),List.length(&2,U32,shape)),shape)) == True{} : Bool},
  inner: Shape.Both<Broadcast.Stretched<yes_shape,yes_count,values>,Broadcast.Stretched<no_shape,no_count,values>>) ->
  Shape.Both<Broadcast.Stretched<yes_shape,yes_count,shape>,Broadcast.Stretched<no_shape,no_count,shape>>:
  Shape.Both{yes_rank,no_rank} = ranks
  Shape.Both{yes_inner,no_inner} = inner
  Shape.Both{
    Broadcast.stretched_through(yes_shape,yes_count,values,shape,yes_exact,
      Broadcast.padded_covers(List.length(&2,U32,yes_shape),Nat.sub(List.length(&2,U32,no_shape),List.length(&2,U32,yes_shape)),List.length(&2,U32,values),yes_rank),
      values_rank,yes_inner,outer),
    Broadcast.stretched_through(no_shape,no_count,values,shape,no_exact,
      Broadcast.padded_covers(List.length(&2,U32,no_shape),Nat.sub(List.length(&2,U32,yes_shape),List.length(&2,U32,no_shape)),List.length(&2,U32,values),no_rank),
      values_rank,no_inner,outer)}

def common_rank(+mask_shape: List<&2,U32>,+values: List<&2,U32>,+shape: List<&2,U32>,
  ranks: Shape.Both<{List.length(&2,U32,shape) == Nat.add(Nat.sub(List.length(&2,U32,values),List.length(&2,U32,mask_shape)),List.length(&2,U32,mask_shape)) : Nat},
    {List.length(&2,U32,shape) == Nat.add(Nat.sub(List.length(&2,U32,mask_shape),List.length(&2,U32,values)),List.length(&2,U32,values)) : Nat}>) ->
  {Nat.is_le(List.length(&2,U32,values),List.length(&2,U32,shape)) == True{} : Bool}:
  Shape.Both{mask_rank,values_rank} = ranks
  Broadcast.padded_covers(List.length(&2,U32,values),Nat.sub(List.length(&2,U32,mask_shape),List.length(&2,U32,values)),List.length(&2,U32,shape),values_rank)

# The mask stretches to the common extents, and so do the common extents of
# the yes and no operands.
def stretched(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +mask_context: Logical,+yes_context: Logical,+no_context: Logical,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,+values: List<&2,U32>,+shape: List<&2,U32>,
  +mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,+mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,+mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,
  +mask_count: U32,+yes_count: U32,+no_count: U32,+count: U32,
  +values_common: {Shapes.broadcast(yes_shape,no_shape) == Some{values} : Maybe<&2,List<&2,U32>>},+common: {Shapes.broadcast(mask_shape,values) == Some{shape} : Maybe<&2,List<&2,U32>>},
  +counted: Global.Counted<shape>,
  +mask_exact: {ShapeLaws.elements(mask_shape) == U32.to_nat(mask_count) : Nat},+yes_exact: {ShapeLaws.elements(yes_shape) == U32.to_nat(yes_count) : Nat},
  +no_exact: {ShapeLaws.elements(no_shape) == U32.to_nat(no_count) : Nat},
  flat: Broadcast.Stretches(Spread.source_axes(mask_shape,mask_count,List.length(&2,U32,shape)),Views.with_rank(Broadcast.axes_of(values),List.length(&2,U32,shape)),shape),
  mask_contract: Global.Contract(~F32,~Logical,~mv,mask_context,SelectPublic.tensor(mask,mg,mask_shape,mf,mw)),
  yes_contract: Global.Contract(~F32,~Logical,~yv,yes_context,SelectPublic.tensor(yes,yg,yes_shape,yf,yw)),
  no_contract: Global.Contract(~F32,~Logical,~nv,no_context,SelectPublic.tensor(no,ng,no_shape,nf,nw))) ->
  Selected(~Logical,~mv,~yv,~nv,Broadcast.Expanded{mask_context,Broadcast.expansion(mask_shape,shape)},Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},
    Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
    Select.shapes(Shapes.same_extents(mask_shape,yes_shape) && Shapes.same_extents(yes_shape,no_shape),shape,Some{count},mask_shape,yes_shape,no_shape,yf,U32.max(mw,U32.max(yw,nw)),
      SelectPublic.tensor(mask,mg,mask_shape,mf,mw),SelectPublic.tensor(yes,yg,yes_shape,yf,yw),SelectPublic.tensor(no,ng,no_shape,nf,nw))):
  Shape.Both{mask_some,outer} = flat
  operands(~Logical,~mv,~yv,~nv,Shapes.same_extents(mask_shape,yes_shape) && Shapes.same_extents(yes_shape,no_shape),
    mask_context,yes_context,no_context,mask_shape,yes_shape,no_shape,values,shape,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,mask_count,yes_count,no_count,count,
    {==},values_common,common,counted,mask_exact,yes_exact,no_exact,
    Broadcast.stretched_when(Views.stretch_axes(Spread.source_axes(mask_shape,mask_count,List.length(&2,U32,shape)),shape),mask_shape,mask_count,shape,{==},mask_some),
    values_stretched(yes_shape,no_shape,values,shape,yes_count,no_count,yes_exact,no_exact,Broadcast.broadcast_ranks(yes_shape,no_shape,values,values_common),
      common_rank(mask_shape,values,shape,Broadcast.broadcast_ranks(mask_shape,values,shape,common)),outer,
      Broadcast.stretched(yes_shape,no_shape,values,yes_count,no_count,yes_exact,no_exact,values_common)),
    mask_contract,yes_contract,no_contract)

def modeled(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +mask_context: Logical,+yes_context: Logical,+no_context: Logical,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,+values: List<&2,U32>,+shape: List<&2,U32>,
  +mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,+mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,+mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,
  +values_common: {Shapes.broadcast(yes_shape,no_shape) == Some{values} : Maybe<&2,List<&2,U32>>},+common: {Shapes.broadcast(mask_shape,values) == Some{shape} : Maybe<&2,List<&2,U32>>},
  counted: Global.Counted<shape>,
  mask_contract: Global.Contract(~F32,~Logical,~mv,mask_context,SelectPublic.tensor(mask,mg,mask_shape,mf,mw)),
  yes_contract: Global.Contract(~F32,~Logical,~yv,yes_context,SelectPublic.tensor(yes,yg,yes_shape,yf,yw)),
  no_contract: Global.Contract(~F32,~Logical,~nv,no_context,SelectPublic.tensor(no,ng,no_shape,nf,nw))) ->
  Selected(~Logical,~mv,~yv,~nv,Broadcast.Expanded{mask_context,Broadcast.expansion(mask_shape,shape)},Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},
    Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
    F.select(SelectPublic.tensor(mask,mg,mask_shape,mf,mw),SelectPublic.tensor(yes,yg,yes_shape,yf,yw),SelectPublic.tensor(no,ng,no_shape,nf,nw))):
  Global.Counted{+count,+accepted} = counted
  Shape.Both{Shape.Both{+mask_formed,Shape.Both{mask_counted,+mask_coherent}},mask_values} = mask_contract
  Global.Counted{+mask_count,+mask_accepted} = mask_counted
  Shape.Both{Shape.Both{+yes_formed,Shape.Both{yes_counted,+yes_coherent}},yes_values} = yes_contract
  Global.Counted{+yes_count,+yes_accepted} = yes_counted
  Shape.Both{Shape.Both{+no_formed,Shape.Both{no_counted,+no_coherent}},no_values} = no_contract
  Global.Counted{+no_count,+no_accepted} = no_counted
  +workers = U32.max(mw,U32.max(yw,nw))
  +contexts = {Broadcast.Expanded{mask_context,Broadcast.expansion(mask_shape,shape)} : Broadcast.Expanded<Logical>}
  +mask_exact = Global.whole_counted(mask_shape,mask_count,mask_accepted)
  %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(yes_shape,no_shape),Some{values},values_common) :
    Selected(~Logical,~mv,~yv,~nv,contexts,Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
      Select.values_shape(_,mask_shape,yes_shape,no_shape,yf,workers,SelectPublic.tensor(mask,mg,mask_shape,mf,mw),SelectPublic.tensor(yes,yg,yes_shape,yf,yw),SelectPublic.tensor(no,ng,no_shape,nf,nw)))
  %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(mask_shape,values),Some{shape},common) :
    Selected(~Logical,~mv,~yv,~nv,contexts,Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
      Select.masked_shape(_,mask_shape,yes_shape,no_shape,values,yf,workers,SelectPublic.tensor(mask,mg,mask_shape,mf,mw),SelectPublic.tensor(yes,yg,yes_shape,yf,yw),SelectPublic.tensor(no,ng,no_shape,nf,nw)))
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},accepted) :
    Selected(~Logical,~mv,~yv,~nv,contexts,Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
      Select.shapes(Shapes.same_extents(mask_shape,yes_shape) && Shapes.same_extents(yes_shape,no_shape),shape,_,mask_shape,yes_shape,no_shape,yf,workers,SelectPublic.tensor(mask,mg,mask_shape,mf,mw),SelectPublic.tensor(yes,yg,yes_shape,yf,yw),SelectPublic.tensor(no,ng,no_shape,nf,nw)))
  stretched(~Logical,~mv,~yv,~nv,mask_context,yes_context,no_context,mask_shape,yes_shape,no_shape,values,shape,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,
    mask_count,yes_count,no_count,count,values_common,common,Global.Counted{count,accepted},
    mask_exact,Global.whole_counted(yes_shape,yes_count,yes_accepted),Global.whole_counted(no_shape,no_count,no_accepted),
    Broadcast.flat_stretches(mask_shape,values,shape,mask_count,mask_exact,common),
    Shape.Both{Shape.Both{mask_formed,Shape.Both{Global.Counted{mask_count,mask_accepted},mask_coherent}},mask_values},
    Shape.Both{Shape.Both{yes_formed,Shape.Both{Global.Counted{yes_count,yes_accepted},yes_coherent}},yes_values},
    Shape.Both{Shape.Both{no_formed,Shape.Both{Global.Counted{no_count,no_accepted},no_coherent}},no_values})

def observed(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +mask_context: Logical,+yes_context: Logical,+no_context: Logical,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,+values: List<&2,U32>,+shape: List<&2,U32>,
  +mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,+mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,-mask: Bands.Bands(~F32),-yes: Bands.Bands(~F32),-no: Bands.Bands(~F32),
  mask_observed: Model.Observation<F32,mask>,yes_observed: Model.Observation<F32,yes>,no_observed: Model.Observation<F32,no>,
  +values_common: {Shapes.broadcast(yes_shape,no_shape) == Some{values} : Maybe<&2,List<&2,U32>>},+common: {Shapes.broadcast(mask_shape,values) == Some{shape} : Maybe<&2,List<&2,U32>>},
  counted: Global.Counted<shape>,
  mask_contract: Global.Contract(~F32,~Logical,~mv,mask_context,Tensors.Tensor{mask,mg,mask_shape,mf,Tensors.Usable{},mw}),
  yes_contract: Global.Contract(~F32,~Logical,~yv,yes_context,Tensors.Tensor{yes,yg,yes_shape,yf,Tensors.Usable{},yw}),
  no_contract: Global.Contract(~F32,~Logical,~nv,no_context,Tensors.Tensor{no,ng,no_shape,nf,Tensors.Usable{},nw})) ->
  Selected(~Logical,~mv,~yv,~nv,Broadcast.Expanded{mask_context,Broadcast.expansion(mask_shape,shape)},Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},
    Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
    F.select(Tensors.Tensor{mask,mg,mask_shape,mf,Tensors.Usable{},mw},Tensors.Tensor{yes,yg,yes_shape,yf,Tensors.Usable{},yw},Tensors.Tensor{no,ng,no_shape,nf,Tensors.Usable{},nw})):
  Model.Observation{+m,+me} = mask_observed
  Model.Observation{+y,+ye} = yes_observed
  Model.Observation{+n,+ne} = no_observed
  +contexts = {Broadcast.Expanded{mask_context,Broadcast.expansion(mask_shape,shape)} : Broadcast.Expanded<Logical>}
  %me : Selected(~Logical,~mv,~yv,~nv,contexts,Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
    F.select(Tensors.Tensor{_,mg,mask_shape,mf,Tensors.Usable{},mw},Tensors.Tensor{yes,yg,yes_shape,yf,Tensors.Usable{},yw},Tensors.Tensor{no,ng,no_shape,nf,Tensors.Usable{},nw}))
  %ye : Selected(~Logical,~mv,~yv,~nv,contexts,Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
    F.select(SelectPublic.tensor(m,mg,mask_shape,mf,mw),Tensors.Tensor{_,yg,yes_shape,yf,Tensors.Usable{},yw},Tensors.Tensor{no,ng,no_shape,nf,Tensors.Usable{},nw}))
  %ne : Selected(~Logical,~mv,~yv,~nv,contexts,Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
    F.select(SelectPublic.tensor(m,mg,mask_shape,mf,mw),SelectPublic.tensor(y,yg,yes_shape,yf,yw),Tensors.Tensor{_,ng,no_shape,nf,Tensors.Usable{},nw}))
  modeled(~Logical,~mv,~yv,~nv,mask_context,yes_context,no_context,mask_shape,yes_shape,no_shape,values,shape,m,y,n,mg,yg,ng,mf,yf,nf,mw,yw,nw,values_common,common,counted,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,m),mask,me) : Global.Contract(~F32,~Logical,~mv,mask_context,Tensors.Tensor{_,mg,mask_shape,mf,Tensors.Usable{},mw}) mask_contract,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,y),yes,ye) : Global.Contract(~F32,~Logical,~yv,yes_context,Tensors.Tensor{_,yg,yes_shape,yf,Tensors.Usable{},yw}) yes_contract,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,n),no,ne) : Global.Contract(~F32,~Logical,~nv,no_context,Tensors.Tensor{_,ng,no_shape,nf,Tensors.Usable{},nw}) no_contract)

# Three Normal tensors whose extents broadcast to counted common extents
# select into a Normal tensor that holds, at every flat index, the yes value
# where the mask is nonzero and the no value elsewhere, each operand read at
# the common extents.
def broadcast(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +mask_context: Logical,+yes_context: Logical,+no_context: Logical,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,+values: List<&2,U32>,+shape: List<&2,U32>,
  +mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,+mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,mask: Bands.Bands(~F32),yes: Bands.Bands(~F32),no: Bands.Bands(~F32),
  +values_common: {Shapes.broadcast(yes_shape,no_shape) == Some{values} : Maybe<&2,List<&2,U32>>},+common: {Shapes.broadcast(mask_shape,values) == Some{shape} : Maybe<&2,List<&2,U32>>},
  counted: Global.Counted<shape>,
  mask_contract: Global.Contract(~F32,~Logical,~mv,mask_context,Tensors.Tensor{mask,mg,mask_shape,mf,Tensors.Usable{},mw}),
  yes_contract: Global.Contract(~F32,~Logical,~yv,yes_context,Tensors.Tensor{yes,yg,yes_shape,yf,Tensors.Usable{},yw}),
  no_contract: Global.Contract(~F32,~Logical,~nv,no_context,Tensors.Tensor{no,ng,no_shape,nf,Tensors.Usable{},nw})) ->
  Selected(~Logical,~mv,~yv,~nv,Broadcast.Expanded{mask_context,Broadcast.expansion(mask_shape,shape)},Broadcast.Expanded{yes_context,Broadcast.expansion(yes_shape,shape)},
    Broadcast.Expanded{no_context,Broadcast.expansion(no_shape,shape)},
    F.select(Tensors.Tensor{mask,mg,mask_shape,mf,Tensors.Usable{},mw},Tensors.Tensor{yes,yg,yes_shape,yf,Tensors.Usable{},yw},Tensors.Tensor{no,ng,no_shape,nf,Tensors.Usable{},nw})):
  observed(~Logical,~mv,~yv,~nv,mask_context,yes_context,no_context,mask_shape,yes_shape,no_shape,values,shape,mg,yg,ng,mf,yf,nf,mw,yw,nw,mask,yes,no,
    Model.observe(~F32,mask),Model.observe(~F32,yes),Model.observe(~F32,no),values_common,common,counted,mask_contract,yes_contract,no_contract)
