# Bands of one geometry in one plan: the select kernel holds the selected
# values and keeps the bands well formed.
import Base
import ../tensor_select.bend as Select
import ../tensor_band.bend as Bands
import ./tensor_select_band_refinement.bend as Refinement
import ./tensor_select_band_values.bend as Selected
import ./tensor_band_arrive.bend as Arrive
import ./tensor_band_combine_leaf.bend as Leaf
import ./tensor_band_each.bend as Each
import ./tensor_band_shape_refinement.bend as Metadata
import ./tensor_band_shape_count.bend as ShapeCount
import ./tensor_band_values.bend as Values
import ./tensor_band_model.bend as Model
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape

def Facts(~Logical: Data,~reader: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,-bands: Bands.Bands(~F32)) -> Data:
  Shape.Both<Values.Values(~F32,~Logical,~reader,logical,geometry,Model.contents(~F32,bands)),Metadata.WellFormed<F32,geometry,Model.contents(~F32,bands)>>

def dimensions(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +logical: Logical,+geometry: Bands.Geometry,+parallel: Bool,+owners: Model.Shaped<Refinement.Three>,
  choice: Arrive.DimensionChoice(geometry),
  +yes_shape: Metadata.WellFormed<F32,geometry,Model.content(~F32,Refinement.yes(owners))>,no_shape: Metadata.WellFormed<F32,geometry,Model.content(~F32,Refinement.no(owners))>,
  mask_values: Values.Values(~F32,~Logical,~mv,logical,geometry,Model.content(~F32,Refinement.mask(owners))),yes_values: Values.Values(~F32,~Logical,~yv,logical,geometry,Model.content(~F32,Refinement.yes(owners))),no_values: Values.Values(~F32,~Logical,~nv,logical,geometry,Model.content(~F32,Refinement.no(owners)))) ->
  Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,Model.layout(~F32,Refinement.yes(owners)),parallel,Model.pack(~F32,Refinement.mask(owners)),Model.pack(~F32,Refinement.yes(owners)),Model.pack(~F32,Refinement.no(owners)))):
  match choice:
    case Inl{empty}:
      Metadata.WellFormed{metadata,ready,coverage} = yes_shape
      Selected.empty_well_formed(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,parallel,owners,Arrive.empty_counts(~F32,geometry,Refinement.yes(owners),ShapeCount.empty_geometry_classified(geometry,empty),metadata),yes_shape)
    case Inr{Arrive.Dimensions{channels_positive,height_positive,width_positive}}:
      Selected.well_formed(~Logical,~mv,~yv,~nv,logical,geometry,parallel,owners,Arrive.metadata_width_positive(geometry,width_positive),yes_shape,no_shape,mask_values,yes_values,no_values)

# Equal plans keep the three owners in one tree.
type Joined<-mask: Model.Model<F32>,-yes: Model.Model<F32>,-no: Model.Model<F32>> is Data:
  Joined{owners: Model.Shaped<Refinement.Three>,mask_equal: {mask == Refinement.mask(owners) : Model.Model<F32>},yes_equal: {yes == Refinement.yes(owners) : Model.Model<F32>},no_equal: {no == Refinement.no(owners) : Model.Model<F32>}}

def leaf_equal(+model: Buffer.Model<F32>,+first: U32,+rows: U32,+other_first: U32,+other_rows: U32,
  ef: {first == other_first : U32},er: {rows == other_rows : U32}) -> {Model.Leaf{model,first,rows} == Model.Leaf{model,other_first,other_rows} : Model.Model<F32>}:
  %Equal.sym(U32,first,other_first,ef) : {Model.Leaf{model,_,rows} == Model.Leaf{model,other_first,other_rows} : Model.Model<F32>}
  %Equal.sym(U32,rows,other_rows,er) : {Model.Leaf{model,other_first,_} == Model.Leaf{model,other_first,other_rows} : Model.Model<F32>}
  {==}

def fork_equal(+left: Model.Model<F32>,+right: Model.Model<F32>,+other_left: Model.Model<F32>,+other_right: Model.Model<F32>,
  el: {left == other_left : Model.Model<F32>},er: {right == other_right : Model.Model<F32>}) -> {Model.Fork{left,right} == Model.Fork{other_left,other_right} : Model.Model<F32>}:
  %Equal.sym(Model.Model<F32>,left,other_left,el) : {Model.Fork{_,right} == Model.Fork{other_left,other_right} : Model.Model<F32>}
  %Equal.sym(Model.Model<F32>,right,other_right,er) : {Model.Fork{other_left,_} == Model.Fork{other_left,other_right} : Model.Model<F32>}
  {==}

def fork(+ml: Model.Model<F32>,+mr: Model.Model<F32>,+yl: Model.Model<F32>,+yr: Model.Model<F32>,+nl: Model.Model<F32>,+nr: Model.Model<F32>,left: Joined<ml,yl,nl>,right: Joined<mr,yr,nr>) -> Joined<Model.Fork{ml,mr},Model.Fork{yl,yr},Model.Fork{nl,nr}>:
  Joined{+left,lm,ly,ln} = left
  Joined{+right,rm,ry,rn} = right
  Joined{Model.Pieces{left,right},fork_equal(ml,mr,Refinement.mask(left),Refinement.mask(right),lm,rm),fork_equal(yl,yr,Refinement.yes(left),Refinement.yes(right),ly,ry),fork_equal(nl,nr,Refinement.no(left),Refinement.no(right),ln,rn)}

def common(+mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,ym: Each.Common(~F32,yes,mask),yn: Each.Common(~F32,yes,no)) -> Joined<mask,yes,no>:
  match mask yes no:
    case Model.Leaf{+mask,+mf,+mr} Model.Leaf{+yes,+yf,+yr} Model.Leaf{+no,+nf,+nr}:
      Shape.Both{ymf,ymr} = ym
      Shape.Both{ynf,ynr} = yn
      Joined{Model.Piece{Refinement.Three{mask,yes,no},yf,yr},leaf_equal(mask,mf,mr,yf,yr,Equal.sym(U32,yf,mf,ymf),Equal.sym(U32,yr,mr,ymr)),{==},leaf_equal(no,nf,nr,yf,yr,Equal.sym(U32,yf,nf,ynf),Equal.sym(U32,yr,nr,ynr))}
    case Model.Fork{+ml,+mr} Model.Fork{+yl,+yr} Model.Fork{+nl,+nr}:
      Shape.Both{yml,ymr} = ym
      Shape.Both{ynl,ynr} = yn
      fork(ml,mr,yl,yr,nl,nr,common(ml,yl,nl,yml,ynl),common(mr,yr,nr,ymr,ynr))
    case Model.Leaf{m,mf,mr} Model.Fork{yl,yr} no: Empty.absurd(Joined<Model.Leaf{m,mf,mr},Model.Fork{yl,yr},no>,ym)
    case Model.Fork{ml,mr} Model.Leaf{y,yf,yr} no: Empty.absurd(Joined<Model.Fork{ml,mr},Model.Leaf{y,yf,yr},no>,ym)
    case Model.Leaf{m,mf,mr} Model.Leaf{y,yf,yr} Model.Fork{nl,nr}: Empty.absurd(Joined<Model.Leaf{m,mf,mr},Model.Leaf{y,yf,yr},Model.Fork{nl,nr}>,yn)
    case Model.Fork{ml,mr} Model.Fork{yl,yr} Model.Leaf{n,nf,nr}: Empty.absurd(Joined<Model.Fork{ml,mr},Model.Fork{yl,yr},Model.Leaf{n,nf,nr}>,yn)

def matched(+mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,ym: {Model.layout(~F32,yes) == Model.layout(~F32,mask) : Bands.Plan},yn: {Model.layout(~F32,yes) == Model.layout(~F32,no) : Bands.Plan}) -> Joined<mask,yes,no>:
  common(mask,yes,no,Each.matched(~F32,yes,mask,ym),Each.matched(~F32,yes,no,yn))

def joined(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +logical: Logical,+geometry: Bands.Geometry,+parallel: Bool,+mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,joined: Joined<mask,yes,no>,
  yes_shape: Metadata.WellFormed<F32,geometry,Model.content(~F32,yes)>,no_shape: Metadata.WellFormed<F32,geometry,Model.content(~F32,no)>,
  mask_values: Values.Values(~F32,~Logical,~mv,logical,geometry,Model.content(~F32,mask)),yes_values: Values.Values(~F32,~Logical,~yv,logical,geometry,Model.content(~F32,yes)),no_values: Values.Values(~F32,~Logical,~nv,logical,geometry,Model.content(~F32,no))) ->
  Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,Model.layout(~F32,yes),parallel,Model.pack(~F32,mask),Model.pack(~F32,yes),Model.pack(~F32,no))):
  Joined{+owners,+me,+ye,+ne} = joined
  +ys = {%ye : Metadata.WellFormed<F32,geometry,Model.content(~F32,_)> yes_shape : Metadata.WellFormed<F32,geometry,Model.content(~F32,Refinement.yes(owners))>}
  +ns = {%ne : Metadata.WellFormed<F32,geometry,Model.content(~F32,_)> no_shape : Metadata.WellFormed<F32,geometry,Model.content(~F32,Refinement.no(owners))>}
  +m_values = {%me : Values.Values(~F32,~Logical,~mv,logical,geometry,Model.content(~F32,_)) mask_values : Values.Values(~F32,~Logical,~mv,logical,geometry,Model.content(~F32,Refinement.mask(owners)))}
  +y_values = {%ye : Values.Values(~F32,~Logical,~yv,logical,geometry,Model.content(~F32,_)) yes_values : Values.Values(~F32,~Logical,~yv,logical,geometry,Model.content(~F32,Refinement.yes(owners)))}
  +n_values = {%ne : Values.Values(~F32,~Logical,~nv,logical,geometry,Model.content(~F32,_)) no_values : Values.Values(~F32,~Logical,~nv,logical,geometry,Model.content(~F32,Refinement.no(owners)))}
  %Equal.sym(Model.Model<F32>,mask,Refinement.mask(owners),me) : Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,Model.layout(~F32,yes),parallel,Model.pack(~F32,_),Model.pack(~F32,yes),Model.pack(~F32,no)))
  %Equal.sym(Model.Model<F32>,yes,Refinement.yes(owners),ye) : Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,Model.layout(~F32,_),parallel,Model.pack(~F32,Refinement.mask(owners)),Model.pack(~F32,_),Model.pack(~F32,no)))
  %Equal.sym(Model.Model<F32>,no,Refinement.no(owners),ne) : Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,Model.layout(~F32,Refinement.yes(owners)),parallel,Model.pack(~F32,Refinement.mask(owners)),Model.pack(~F32,Refinement.yes(owners)),Model.pack(~F32,_)))
  dimensions(~Logical,~mv,~yv,~nv,logical,geometry,parallel,owners,Arrive.dimension_choice(geometry),ys,ns,m_values,y_values,n_values)

def selected_models(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +logical: Logical,+geometry: Bands.Geometry,+parallel: Bool,+mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,
  ym: {Model.layout(~F32,yes) == Model.layout(~F32,mask) : Bands.Plan},yn: {Model.layout(~F32,yes) == Model.layout(~F32,no) : Bands.Plan},
  yes_shape: Metadata.WellFormed<F32,geometry,Model.content(~F32,yes)>,no_shape: Metadata.WellFormed<F32,geometry,Model.content(~F32,no)>,
  mask_values: Values.Values(~F32,~Logical,~mv,logical,geometry,Model.content(~F32,mask)),yes_values: Values.Values(~F32,~Logical,~yv,logical,geometry,Model.content(~F32,yes)),no_values: Values.Values(~F32,~Logical,~nv,logical,geometry,Model.content(~F32,no))) ->
  Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,Model.layout(~F32,yes),parallel,Model.pack(~F32,mask),Model.pack(~F32,yes),Model.pack(~F32,no))):
  joined(~Logical,~mv,~yv,~nv,logical,geometry,parallel,mask,yes,no,matched(mask,yes,no,ym,yn),yes_shape,no_shape,mask_values,yes_values,no_values)

# Bands observed once: each holds the plan it was cut to.
def model_layout(+model: Model.Model<F32>,+layout: Bands.Plan,-bands: Bands.Bands(~F32),+equation: {Model.pack(~F32,model) == bands : Bands.Bands(~F32)},
  actual: {Model.content_layout(~F32,Model.contents(~F32,bands)) == layout : Bands.Plan}) -> {Model.layout(~F32,model) == layout : Bands.Plan}:
  Equal.trans(Bands.Plan,Model.layout(~F32,model),Model.content_layout(~F32,Model.content(~F32,model)),layout,Model.layout_matches_content(~F32,model),
    %Model.observed_contents(~F32,model,bands,equation) : {Model.content_layout(~F32,_) == layout : Bands.Plan} actual)

def selected_observed(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +logical: Logical,+geometry: Bands.Geometry,+layout: Bands.Plan,+parallel: Bool,-mask: Bands.Bands(~F32),-yes: Bands.Bands(~F32),-no: Bands.Bands(~F32),
  mask_observed: Model.Observation<F32,mask>,yes_observed: Model.Observation<F32,yes>,no_observed: Model.Observation<F32,no>,
  mask_layout: {Model.content_layout(~F32,Model.contents(~F32,mask)) == layout : Bands.Plan},yes_layout: {Model.content_layout(~F32,Model.contents(~F32,yes)) == layout : Bands.Plan},no_layout: {Model.content_layout(~F32,Model.contents(~F32,no)) == layout : Bands.Plan},
  yes_shape: Metadata.WellFormed<F32,geometry,Model.contents(~F32,yes)>,no_shape: Metadata.WellFormed<F32,geometry,Model.contents(~F32,no)>,
  mask_values: Values.Values(~F32,~Logical,~mv,logical,geometry,Model.contents(~F32,mask)),yes_values: Values.Values(~F32,~Logical,~yv,logical,geometry,Model.contents(~F32,yes)),no_values: Values.Values(~F32,~Logical,~nv,logical,geometry,Model.contents(~F32,no))) ->
  Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,layout,parallel,mask,yes,no)):
  Model.Observation{+m,+me} = mask_observed
  Model.Observation{+y,+ye} = yes_observed
  Model.Observation{+n,+ne} = no_observed
  +mc = Model.observed_contents(~F32,m,mask,me)
  +yc = Model.observed_contents(~F32,y,yes,ye)
  +nc = Model.observed_contents(~F32,n,no,ne)
  +ml = model_layout(m,layout,mask,me,mask_layout)
  +yl = model_layout(y,layout,yes,ye,yes_layout)
  +nl = model_layout(n,layout,no,ne,no_layout)
  %me : Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,layout,parallel,_,yes,no))
  %ye : Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,layout,parallel,Model.pack(~F32,m),_,no))
  %ne : Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,layout,parallel,Model.pack(~F32,m),Model.pack(~F32,y),_))
  %yl : Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,_,parallel,Model.pack(~F32,m),Model.pack(~F32,y),Model.pack(~F32,n)))
  selected_models(~Logical,~mv,~yv,~nv,logical,geometry,parallel,m,y,n,
    Equal.trans(Bands.Plan,Model.layout(~F32,y),layout,Model.layout(~F32,m),yl,Equal.sym(Bands.Plan,Model.layout(~F32,m),layout,ml)),
    Equal.trans(Bands.Plan,Model.layout(~F32,y),layout,Model.layout(~F32,n),yl,Equal.sym(Bands.Plan,Model.layout(~F32,n),layout,nl)),
    %yc : Metadata.WellFormed<F32,geometry,_> yes_shape,%nc : Metadata.WellFormed<F32,geometry,_> no_shape,
    %mc : Values.Values(~F32,~Logical,~mv,logical,geometry,_) mask_values,%yc : Values.Values(~F32,~Logical,~yv,logical,geometry,_) yes_values,%nc : Values.Values(~F32,~Logical,~nv,logical,geometry,_) no_values)

def selected(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +logical: Logical,+geometry: Bands.Geometry,+layout: Bands.Plan,+parallel: Bool,mask: Bands.Bands(~F32),yes: Bands.Bands(~F32),no: Bands.Bands(~F32),
  mask_layout: {Model.content_layout(~F32,Model.contents(~F32,mask)) == layout : Bands.Plan},yes_layout: {Model.content_layout(~F32,Model.contents(~F32,yes)) == layout : Bands.Plan},no_layout: {Model.content_layout(~F32,Model.contents(~F32,no)) == layout : Bands.Plan},
  yes_shape: Metadata.WellFormed<F32,geometry,Model.contents(~F32,yes)>,no_shape: Metadata.WellFormed<F32,geometry,Model.contents(~F32,no)>,
  mask_values: Values.Values(~F32,~Logical,~mv,logical,geometry,Model.contents(~F32,mask)),yes_values: Values.Values(~F32,~Logical,~yv,logical,geometry,Model.contents(~F32,yes)),no_values: Values.Values(~F32,~Logical,~nv,logical,geometry,Model.contents(~F32,no))) ->
  Facts(~Logical,~Selected.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,layout,parallel,mask,yes,no)):
  selected_observed(~Logical,~mv,~yv,~nv,logical,geometry,layout,parallel,mask,yes,no,Model.observe(~F32,mask),Model.observe(~F32,yes),Model.observe(~F32,no),mask_layout,yes_layout,no_layout,yes_shape,no_shape,mask_values,yes_values,no_values)