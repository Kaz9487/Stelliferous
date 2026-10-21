# Aligned band leaves hold the select reader's values through the lane value
# theorem.
import Base
import ../tensor_band.bend as Bands
import ../tensor_select.bend as Select
import ./tensor_band_plan.bend as Plan
import ./tensor_band_inspection.bend as Inspection
import ../math_compare.bend as Compare
import ./tensor_select_band_refinement.bend as Refinement
import ./tensor_select_lanes_refinement.bend as Lanes
import ./tensor_select_model_values.bend as LaneValues
import ./tensor_band_values.bend as Values
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as Metadata
import ./tensor_band_arrive.bend as Arrive
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./tensor_band_shape_count.bend as TensorBandShapeCount

# The select reader: the yes value where the mask is nonzero, the no value
# elsewhere.
def value(~Logical: Data,~mask: Logical -> U32 -> F32,~yes: Logical -> U32 -> F32,~no: Logical -> U32 -> F32,logical: Logical,index: U32) -> F32:
  +logical = logical
  +index = index
  Compare.select(mask(logical,index),yes(logical,index),no(logical,index))

def output(+geometry: Bands.Geometry,+rows: U32,+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>) -> Buffer.Content<F32>:
  Lanes.model(U32.to_nat(Bands.local_count(geometry,rows)),0n,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no))

type LeafCells<-Logical: Data,-mv: Logical -> U32 -> F32,-yv: Logical -> U32 -> F32,-nv: Logical -> U32 -> F32> is Data:
  LeafCells{logical: Logical,geometry: Bands.Geometry,first: U32,rows: U32,mask: Buffer.Model<F32>,yes: Buffer.Model<F32>,no: Buffer.Model<F32>,
    yes_ready: Buffer.ContentReady<F32,Buffer.content(~F32,yes)>,no_ready: Buffer.ContentReady<F32,Buffer.content(~F32,no)>,
    counted: {U32.to_nat(Bands.local_count(geometry,rows)) == Buffer.length(~F32,yes) : Nat},
    no_length: {Buffer.length(~F32,no) == Buffer.length(~F32,yes) : Nat},
    meta: LaneValues.Meta<output(geometry,rows,mask,yes,no),Buffer.length(~F32,yes)>,
    mask_values: Values.Leaf(~F32,~Logical,~mv,logical,geometry,first,rows,Buffer.tree(~F32,mask),Buffer.length(~F32,mask)),
    yes_values: Values.Leaf(~F32,~Logical,~yv,logical,geometry,first,rows,Buffer.tree(~F32,yes),Buffer.length(~F32,yes)),
    no_values: Values.Leaf(~F32,~Logical,~nv,logical,geometry,first,rows,Buffer.tree(~F32,no),Buffer.length(~F32,no))}

def cells_geometry(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,evidence: LeafCells<Logical,mv,yv,nv>) -> Bands.Geometry:
  LeafCells{logical,geometry,first,rows,mask,yes,no,yr,nr,counted,nl,meta,m,y,n} = evidence
  geometry

def cells_rows(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,evidence: LeafCells<Logical,mv,yv,nv>) -> U32:
  LeafCells{logical,geometry,first,rows,mask,yes,no,yr,nr,counted,nl,meta,m,y,n} = evidence
  rows

def Cell(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,+evidence: LeafCells<Logical,mv,yv,nv>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  LeafCells{logical,+geometry,first,+rows,+mask,+yes,+no,yr,nr,counted,nl,meta,m,y,n} = evidence
  Values.At(~F32,~Logical,~value(~Logical,~mv,~yv,~nv),logical,geometry,first,rows,Buffer.content_tree(~F32,output(geometry,rows,mask,yes,no)),Buffer.content_length(~F32,output(geometry,rows,mask,yes,no)),channel,row,column)

def selected_readers(+m: F32,+y: F32,+n: F32,+wanted_m: F32,+wanted_y: F32,+wanted_n: F32,
  em: {m == wanted_m : F32},ey: {y == wanted_y : F32},en: {n == wanted_n : F32}) -> {Compare.select(m,y,n) == Compare.select(wanted_m,wanted_y,wanted_n) : F32}:
  %Equal.sym(F32,m,wanted_m,em) : {Compare.select(_,y,n) == Compare.select(wanted_m,wanted_y,wanted_n) : F32}
  %Equal.sym(F32,y,wanted_y,ey) : {Compare.select(wanted_m,_,n) == Compare.select(wanted_m,wanted_y,wanted_n) : F32}
  %Equal.sym(F32,n,wanted_n,en) : {Compare.select(wanted_m,wanted_y,_) == Compare.select(wanted_m,wanted_y,wanted_n) : F32}
  {==}

def cell(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,+evidence: LeafCells<Logical,mv,yv,nv>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,Values.channels(cells_geometry(~Logical,~mv,~yv,~nv,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(cells_rows(~Logical,~mv,~yv,~nv,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,Values.width(cells_geometry(~Logical,~mv,~yv,~nv,evidence))) == True{} : Bool}) -> Cell(~Logical,~mv,~yv,~nv,evidence,channel,row,column):
  LeafCells{+logical,+geometry,+first,+rows,Buffer.Model{+mt,+ml,+mc},Buffer.Model{+yt,+yl,+yc},Buffer.Model{+nt,+nlength,+nc},+yr,+nr,+counted,+nl,LaneValues.Meta{+length,ready},m,y,n} = evidence
  +mask = {Buffer.Model{mt,ml,mc} : Buffer.Model<F32>}
  +yes = {Buffer.Model{yt,yl,yc} : Buffer.Model<F32>}
  +no = {Buffer.Model{nt,nlength,nc} : Buffer.Model<F32>}
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  +position = Values.position(geometry,rows,channel,row,column)
  +index = U32.from_nat(position)
  +pixel = U32.from_nat(Values.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))
  +count = U32.to_nat(Bands.local_count(geometry,rows))
  +mask_cell = Values.leaf_cell(~F32,~Logical,~mv,logical,geometry,first,rows,mt,ml,channel,row,column,m,channel_inside,row_inside,column_inside)
  +yes_cell = Values.leaf_cell(~F32,~Logical,~yv,logical,geometry,first,rows,yt,yl,channel,row,column,y,channel_inside,row_inside,column_inside)
  +no_cell = Values.leaf_cell(~F32,~Logical,~nv,logical,geometry,first,rows,nt,nlength,channel,row,column,n,channel_inside,row_inside,column_inside)
  +mask_equal = Shape.second({Nat.is_lt(position,ml) == True{} : Bool},{Tree.read(F32,mt,index) == mv(logical,pixel) : F32},mask_cell)
  +yes_equal = Shape.second({Nat.is_lt(position,yl) == True{} : Bool},{Tree.read(F32,yt,index) == yv(logical,pixel) : F32},yes_cell)
  +no_equal = Shape.second({Nat.is_lt(position,nlength) == True{} : Bool},{Tree.read(F32,nt,index) == nv(logical,pixel) : F32},no_cell)
  +yes_inside = Shape.first({Nat.is_lt(position,yl) == True{} : Bool},{Tree.read(F32,yt,index) == yv(logical,pixel) : F32},yes_cell)
  +span = {%counted : {Nat.is_le(count,_) == True{} : Bool} Order.reflexive(count) : {Nat.is_le(count,yl) == True{} : Bool}}
  +inside = {%Equal.sym(Nat,count,yl,counted) : {Nat.is_lt(position,_) == True{} : Bool} yes_inside : {Nat.is_lt(position,count) == True{} : Bool}}
  Shape.Both{
    %Equal.sym(Nat,Buffer.content_length(~F32,output(geometry,rows,mask,yes,no)),yl,length) : {Nat.is_lt(position,_) == True{} : Bool} yes_inside,
    Equal.trans(F32,Tree.read(F32,Buffer.content_tree(~F32,output(geometry,rows,mask,yes,no)),index),Buffer.value(~F32,index,output(geometry,rows,mask,yes,no)),Compare.select(mv(logical,pixel),yv(logical,pixel),nv(logical,pixel)),Buffer.read_content(index,output(geometry,rows,mask,yes,no)),
      Equal.trans(F32,Buffer.value(~F32,index,output(geometry,rows,mask,yes,no)),Compare.select(Tree.read(F32,mt,index),Tree.read(F32,yt,index),Tree.read(F32,nt,index)),Compare.select(mv(logical,pixel),yv(logical,pixel),nv(logical,pixel)),
        LaneValues.selected(count,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no),position,yr,nr,nl,span,inside),
        selected_readers(Tree.read(F32,mt,index),Tree.read(F32,yt,index),Tree.read(F32,nt,index),mv(logical,pixel),yv(logical,pixel),nv(logical,pixel),mask_equal,yes_equal,no_equal)))}

def meta_length(+output: Buffer.Content<F32>,+input_length: Nat,evidence: LaneValues.Meta<output,input_length>) -> {Buffer.content_length(~F32,output) == input_length : Nat}:
  LaneValues.Meta{length,ready} = evidence
  length

def meta_ready(+output: Buffer.Content<F32>,+input_length: Nat,evidence: LaneValues.Meta<output,input_length>) -> Buffer.ContentReady<F32,output>:
  LaneValues.Meta{length,ready} = evidence
  ready

def assembled(~Logical: Data,~reader: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,content: Buffer.Content<F32>,
  values: Values.Leaf(~F32,~Logical,~reader,logical,geometry,first,rows,Buffer.content_tree(~F32,content),Buffer.content_length(~F32,content)),
  ready: Buffer.ContentReady<F32,content>,metadata: Metadata.Leaf(geometry,first,rows,Buffer.content_length(~F32,content))) -> Values.PipelineContract(~F32,~Logical,~reader,logical,geometry,Model.RowBand{content,first,rows}):
  Buffer.Content{tree,length} = content
  Shape.Both{values,Shape.Both{ready,metadata}}

def leaf(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,
  +width_positive: {Metadata.width_positive(geometry) == True{} : Bool},+yes_metadata: Metadata.Leaf(geometry,first,rows,Buffer.length(~F32,yes)),+no_metadata: Metadata.Leaf(geometry,first,rows,Buffer.length(~F32,no)),
  +yes_ready: Buffer.Sized<F32,yes>,+no_ready: Buffer.Sized<F32,no>,mask_values: Values.Values(~F32,~Logical,~mv,logical,geometry,Model.RowBand{Buffer.content(~F32,mask),first,rows}),yes_values: Values.Values(~F32,~Logical,~yv,logical,geometry,Model.RowBand{Buffer.content(~F32,yes),first,rows}),no_values: Values.Values(~F32,~Logical,~nv,logical,geometry,Model.RowBand{Buffer.content(~F32,no),first,rows})) ->
  Values.PipelineContract(~F32,~Logical,~value(~Logical,~mv,~yv,~nv),logical,geometry,Refinement.model(geometry,Model.Piece{Refinement.Three{mask,yes,no},first,rows})):
  Buffer.Model{+mt,+ml,+mc} = mask
  Buffer.Model{+yt,+yl,+yc} = yes
  Buffer.Model{+nt,+nl,+nc} = no
  +count = U32.to_nat(Bands.local_count(geometry,rows))
  +counted = TensorBandShapeCount.count_refinement_exact(geometry,rows,yl,Metadata.count(~F32,geometry,first,rows,yes,width_positive,yes_metadata,yes_ready))
  +yr = Buffer.content_ready(~F32,yes,yes_ready)
  +nr = Buffer.content_ready(~F32,no,no_ready)
  +span = {%counted : {Nat.is_le(count,_) == True{} : Bool} Order.reflexive(count) : {Nat.is_le(count,yl) == True{} : Bool}}
  +same_length = Equal.trans(Nat,nl,Values.cell_count(geometry,rows),yl,Values.leaf_length(geometry,first,rows,nl,no_metadata),Equal.sym(Nat,yl,Values.cell_count(geometry,rows),Values.leaf_length(geometry,first,rows,yl,yes_metadata)))
  +meta = LaneValues.meta(count)(0n,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no),yr,span)
  +length = meta_length(output(geometry,rows,mask,yes,no),yl,meta)
  +ready = meta_ready(output(geometry,rows,mask,yes,no),yl,meta)
  assembled(~Logical,~value(~Logical,~mv,~yv,~nv),logical,geometry,first,rows,output(geometry,rows,mask,yes,no),
    Values.cells(~LeafCells<Logical,mv,yv,nv>,~Cell(~Logical,~mv,~yv,~nv),~cells_geometry(~Logical,~mv,~yv,~nv),~cells_rows(~Logical,~mv,~yv,~nv),~cell(~Logical,~mv,~yv,~nv),LeafCells{logical,geometry,first,rows,mask,yes,no,yr,nr,counted,same_length,meta,mask_values,yes_values,no_values}),
    ready,%Equal.sym(Nat,Buffer.content_length(~F32,output(geometry,rows,mask,yes,no)),yl,length) : Metadata.Leaf(geometry,first,rows,_) yes_metadata)

def pipeline(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +logical: Logical,+geometry: Bands.Geometry,owners: Model.Shaped<Refinement.Three>,+width_positive: {Metadata.width_positive(geometry) == True{} : Bool},
  yes_metadata: Metadata.Contents(~F32,geometry,Model.content(~F32,Refinement.yes(owners))),no_metadata: Metadata.Contents(~F32,geometry,Model.content(~F32,Refinement.no(owners))),
  yes_ready: Model.Ready(~F32,Refinement.yes(owners)),no_ready: Model.Ready(~F32,Refinement.no(owners)),
  mask_values: Values.Values(~F32,~Logical,~mv,logical,geometry,Model.content(~F32,Refinement.mask(owners))),yes_values: Values.Values(~F32,~Logical,~yv,logical,geometry,Model.content(~F32,Refinement.yes(owners))),no_values: Values.Values(~F32,~Logical,~nv,logical,geometry,Model.content(~F32,Refinement.no(owners)))) ->
  Values.PipelineContract(~F32,~Logical,~value(~Logical,~mv,~yv,~nv),logical,geometry,Refinement.model(geometry,owners)):
  match owners:
    case Model.Piece{Refinement.Three{Buffer.Model{+mt,+ml,+mc},Buffer.Model{+yt,+yl,+yc},Buffer.Model{+nt,+nl,+nc}},+first,+rows}:
      leaf(~Logical,~mv,~yv,~nv,logical,geometry,first,rows,Buffer.Model{mt,ml,mc},Buffer.Model{yt,yl,yc},Buffer.Model{nt,nl,nc},width_positive,yes_metadata,no_metadata,yes_ready,no_ready,mask_values,yes_values,no_values)
    case Model.Pieces{+left,+right}:
      Shape.Both{ym_left,ym_right} = yes_metadata
      Shape.Both{nm_left,nm_right} = no_metadata
      Shape.Both{yr_left,yr_right} = yes_ready
      Shape.Both{nr_left,nr_right} = no_ready
      Shape.Both{mv_left,mv_right} = mask_values
      Shape.Both{yv_left,yv_right} = yes_values
      Shape.Both{nv_left,nv_right} = no_values
      Values.pipeline_joined(~F32,~Logical,~value(~Logical,~mv,~yv,~nv),logical,geometry,Refinement.model(geometry,left),Refinement.model(geometry,right),
        pipeline(~Logical,~mv,~yv,~nv,logical,geometry,left,width_positive,ym_left,nm_left,yr_left,nr_left,mv_left,yv_left,nv_left),
        pipeline(~Logical,~mv,~yv,~nv,logical,geometry,right,width_positive,ym_right,nm_right,yr_right,nr_right,mv_right,yv_right,nv_right))

def output_layout(+geometry: Bands.Geometry,owners: Model.Shaped<Refinement.Three>) -> {Model.content_layout(~F32,Refinement.model(geometry,owners)) == Model.layout(~F32,Refinement.yes(owners)) : Bands.Plan}:
  match owners:
    case Model.Piece{Refinement.Three{mask,yes,no},first,rows}: {==}
    case Model.Pieces{+left,+right}:
      %Equal.sym(Bands.Plan,Model.content_layout(~F32,Refinement.model(geometry,left)),Model.layout(~F32,Refinement.yes(left)),output_layout(geometry,left)) : {Bands.Split{_,Model.content_layout(~F32,Refinement.model(geometry,right))} == Bands.Split{Model.layout(~F32,Refinement.yes(left)),Model.layout(~F32,Refinement.yes(right))} : Bands.Plan}
      %Equal.sym(Bands.Plan,Model.content_layout(~F32,Refinement.model(geometry,right)),Model.layout(~F32,Refinement.yes(right)),output_layout(geometry,right)) : {Bands.Split{Model.layout(~F32,Refinement.yes(left)),_} == Bands.Split{Model.layout(~F32,Refinement.yes(left)),Model.layout(~F32,Refinement.yes(right))} : Bands.Plan}
      {==}

def well_formed(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,
  +logical: Logical,+geometry: Bands.Geometry,+parallel: Bool,+owners: Model.Shaped<Refinement.Three>,+width_positive: {Metadata.width_positive(geometry) == True{} : Bool},
  yes_shape: Metadata.WellFormed<F32,geometry,Model.content(~F32,Refinement.yes(owners))>,no_shape: Metadata.WellFormed<F32,geometry,Model.content(~F32,Refinement.no(owners))>,
  mask_values: Values.Values(~F32,~Logical,~mv,logical,geometry,Model.content(~F32,Refinement.mask(owners))),yes_values: Values.Values(~F32,~Logical,~yv,logical,geometry,Model.content(~F32,Refinement.yes(owners))),no_values: Values.Values(~F32,~Logical,~nv,logical,geometry,Model.content(~F32,Refinement.no(owners)))) ->
  Shape.Both<Values.Values(~F32,~Logical,~value(~Logical,~mv,~yv,~nv),logical,geometry,
      Model.contents(~F32,Select.selected_bands(geometry,Model.layout(~F32,Refinement.yes(owners)),parallel,Model.pack(~F32,Refinement.mask(owners)),Model.pack(~F32,Refinement.yes(owners)),Model.pack(~F32,Refinement.no(owners))))),
    Metadata.WellFormed<F32,geometry,Model.contents(~F32,Select.selected_bands(geometry,Model.layout(~F32,Refinement.yes(owners)),parallel,Model.pack(~F32,Refinement.mask(owners)),Model.pack(~F32,Refinement.yes(owners)),Model.pack(~F32,Refinement.no(owners))))>>:
  Metadata.WellFormed{ym,yr,coverage} = yes_shape
  Metadata.WellFormed{nm,nr,no_coverage} = no_shape
  %Equal.sym(Model.Content<F32>,Model.contents(~F32,Select.selected_bands(geometry,Model.layout(~F32,Refinement.yes(owners)),parallel,Model.pack(~F32,Refinement.mask(owners)),Model.pack(~F32,Refinement.yes(owners)),Model.pack(~F32,Refinement.no(owners)))),Refinement.model(geometry,owners),Refinement.selected_bands(geometry,parallel,owners)) : Shape.Both<Values.Values(~F32,~Logical,~value(~Logical,~mv,~yv,~nv),logical,geometry,_),Metadata.WellFormed<F32,geometry,_>>
  Values.pipeline_well_formed(~F32,~Logical,~value(~Logical,~mv,~yv,~nv),logical,geometry,Refinement.model(geometry,owners),
    pipeline(~Logical,~mv,~yv,~nv,logical,geometry,owners,width_positive,ym,nm,Model.ready(~F32,Refinement.yes(owners),yr),Model.ready(~F32,Refinement.no(owners),nr),mask_values,yes_values,no_values),
    %Equal.sym(Bands.Plan,Model.content_layout(~F32,Refinement.model(geometry,owners)),Model.layout(~F32,Refinement.yes(owners)),output_layout(geometry,owners)) : Plan.Coverage(_,0n,Inspection.total_rows(geometry))
    %Equal.sym(Bands.Plan,Model.layout(~F32,Refinement.yes(owners)),Model.content_layout(~F32,Model.content(~F32,Refinement.yes(owners))),Model.layout_matches_content(~F32,Refinement.yes(owners))) : Plan.Coverage(_,0n,Inspection.total_rows(geometry)) coverage)

def zero_lane(mask: Buffer.Content<F32>,yes: Buffer.Content<F32>,no: Buffer.Content<F32>) -> {Lanes.model(0n,0n,mask,yes,no) == yes : Buffer.Content<F32>}:
  Buffer.Content{mt,ml} = mask
  Buffer.Content{yt,yl} = yes
  Buffer.Content{nt,nl} = no
  {==}

def empty_model(+geometry: Bands.Geometry,owners: Model.Shaped<Refinement.Three>,counts: Arrive.Counts(~F32,geometry,Refinement.yes(owners))) -> {Refinement.model(geometry,owners) == Model.content(~F32,Refinement.yes(owners)) : Model.Content<F32>}:
  match owners:
    case Model.Piece{Refinement.Three{+mask,+yes,+no},+first,+rows}:
      %Equal.sym(U32,Bands.local_count(geometry,rows),0,counts) : {Model.RowBand{Lanes.model(U32.to_nat(_),0n,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)),first,rows} == Model.RowBand{Buffer.content(~F32,yes),first,rows} : Model.Content<F32>}
      Equal.cong(Buffer.Content<F32>,Model.Content<F32>,content => Model.RowBand{content,first,rows},Lanes.model(0n,0n,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)),Buffer.content(~F32,yes),zero_lane(Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)))
    case Model.Pieces{+left,+right}:
      Shape.Both{lc,rc} = counts
      %Equal.sym(Model.Content<F32>,Refinement.model(geometry,left),Model.content(~F32,Refinement.yes(left)),empty_model(geometry,left,lc)) : {Model.Joined{_,Refinement.model(geometry,right)} == Model.Joined{Model.content(~F32,Refinement.yes(left)),Model.content(~F32,Refinement.yes(right))} : Model.Content<F32>}
      %Equal.sym(Model.Content<F32>,Refinement.model(geometry,right),Model.content(~F32,Refinement.yes(right)),empty_model(geometry,right,rc)) : {Model.Joined{Model.content(~F32,Refinement.yes(left)),_} == Model.Joined{Model.content(~F32,Refinement.yes(left)),Model.content(~F32,Refinement.yes(right))} : Model.Content<F32>}
      {==}

def empty_well_formed(~Logical: Data,~reader: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+parallel: Bool,+owners: Model.Shaped<Refinement.Three>,
  +counts: Arrive.Counts(~F32,geometry,Refinement.yes(owners)),yes_shape: Metadata.WellFormed<F32,geometry,Model.content(~F32,Refinement.yes(owners))>) ->
  Shape.Both<Values.Values(~F32,~Logical,~reader,logical,geometry,Model.contents(~F32,Select.selected_bands(geometry,Model.layout(~F32,Refinement.yes(owners)),parallel,Model.pack(~F32,Refinement.mask(owners)),Model.pack(~F32,Refinement.yes(owners)),Model.pack(~F32,Refinement.no(owners))))),
    Metadata.WellFormed<F32,geometry,Model.contents(~F32,Select.selected_bands(geometry,Model.layout(~F32,Refinement.yes(owners)),parallel,Model.pack(~F32,Refinement.mask(owners)),Model.pack(~F32,Refinement.yes(owners)),Model.pack(~F32,Refinement.no(owners))))>>:
  Metadata.WellFormed{+metadata,+ready,+coverage} = yes_shape
  %Equal.sym(Model.Content<F32>,Model.contents(~F32,Select.selected_bands(geometry,Model.layout(~F32,Refinement.yes(owners)),parallel,Model.pack(~F32,Refinement.mask(owners)),Model.pack(~F32,Refinement.yes(owners)),Model.pack(~F32,Refinement.no(owners)))),Refinement.model(geometry,owners),Refinement.selected_bands(geometry,parallel,owners)) : Shape.Both<Values.Values(~F32,~Logical,~reader,logical,geometry,_),Metadata.WellFormed<F32,geometry,_>>
  %Equal.sym(Model.Content<F32>,Refinement.model(geometry,owners),Model.content(~F32,Refinement.yes(owners)),empty_model(geometry,owners,counts)) : Shape.Both<Values.Values(~F32,~Logical,~reader,logical,geometry,_),Metadata.WellFormed<F32,geometry,_>>
  Shape.Both{Arrive.empty_band_values(~F32,~Logical,~reader,logical,geometry,Refinement.yes(owners),counts,metadata,Model.ready(~F32,Refinement.yes(owners),ready)),Metadata.WellFormed{metadata,ready,coverage}}
