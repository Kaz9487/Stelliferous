# Three aligned owners share the existing each and zip recursion.
import Base
import ../tensor_select.bend as Select
import ../tensor_lanes.bend as NativeLanes
import ../tensor_band.bend as Bands
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition
import ./tensor_band_each.bend as Each
import ./tensor_band_model.bend as Model
import ./storage_buffer_model.bend as Buffer
import ./tensor_select_lanes_refinement.bend as Lanes

# The three buffers a selection reads for one band.
type Three is Data:
  Three{mask: Buffer.Model<F32>,yes: Buffer.Model<F32>,no: Buffer.Model<F32>}

def three_mask(part: Three) -> Buffer.Model<F32>:
  Three{mask,yes,no} = part
  mask

def three_yes(part: Three) -> Buffer.Model<F32>:
  Three{mask,yes,no} = part
  yes

def three_no(part: Three) -> Buffer.Model<F32>:
  Three{mask,yes,no} = part
  no

def three_band(geometry: Bands.Geometry,part: Three,+first: U32,+rows: U32) -> Buffer.Content<F32>:
  Three{mask,yes,no} = part
  Lanes.model(U32.to_nat(Bands.local_count(geometry,rows)),0n,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no))

def mask(owners: Model.Shaped<Three>) -> Model.Model<F32>:
  Model.shaped_model(~Three,~F32,~three_mask,owners)

def yes(owners: Model.Shaped<Three>) -> Model.Model<F32>:
  Model.shaped_model(~Three,~F32,~three_yes,owners)

def no(owners: Model.Shaped<Three>) -> Model.Model<F32>:
  Model.shaped_model(~Three,~F32,~three_no,owners)

def model(+geometry: Bands.Geometry,owners: Model.Shaped<Three>) -> Model.Content<F32>:
  Model.shaped_content(~Three,~F32,~Bands.Geometry,~three_band,geometry,owners)

def paired(+owners: Model.Shaped<Three>) -> Each.Tree(~Bands.Paired<Storage.Buffer<F32>,Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>>):
  Bands.zip(~Storage.Buffer<F32>,~Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>,Model.pack(~F32,yes(owners)),Bands.zip(~Storage.Buffer<F32>,~Storage.Buffer<F32>,Model.pack(~F32,mask(owners)),Model.pack(~F32,no(owners))))

def run(+enabled: Bool,+geometry: Bands.Geometry,+owners: Model.Shaped<Three>) -> Bands.Bands(~F32):
  Bands.each(~Bands.Paired<Storage.Buffer<F32>,Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>>,~Storage.Buffer<F32>,~Unit,~Bands.Geometry,
    ~Bands.unshared,~Select.selected_leaf,~Bands.left_of(~Storage.Buffer<F32>,~Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>),
    Model.layout(~F32,yes(owners)),enabled,geometry,paired(owners),Unit{})

def leaf(+geometry: Bands.Geometry,+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,+first: U32,+rows: U32) ->
  {Model.contents(~F32,Select.selected_leaf(geometry,first,rows,Bands.Both{Buffer.pack(~F32,yes),Bands.Both{Buffer.pack(~F32,mask),Buffer.pack(~F32,no)}},Unit{})) == model(geometry,Model.Piece{Three{mask,yes,no},first,rows}) : Model.Content<F32>}:
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,NativeLanes.select(U32.to_nat(Bands.local_count(geometry,rows)),Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no))),Lanes.model(U32.to_nat(Bands.local_count(geometry,rows)),0n,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)),Lanes.select(U32.to_nat(Bands.local_count(geometry,rows)),mask,yes,no)) :
    {Model.RowBand{_,first,rows} == model(geometry,Model.Piece{Three{mask,yes,no},first,rows}) : Model.Content<F32>}
  {==}

def each(+geometry: Bands.Geometry,owners: Model.Shaped<Three>,+enabled: Bool) -> {Model.contents(~F32,run(enabled,geometry,owners)) == model(geometry,owners) : Model.Content<F32>}:
  match owners:
    case Model.Piece{Three{+mask,+yes,+no},+first,+rows}: leaf(geometry,mask,yes,no,first,rows)
    case Model.Pieces{+left,+right}:
      %Equal.sym(Bands.Bands(~F32),run(enabled,geometry,Model.Pieces{left,right}),
        Partition.Fork{run(enabled,geometry,left),run(enabled,geometry,right)},
        Each.fork(~Bands.Paired<Storage.Buffer<F32>,Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>>,~Storage.Buffer<F32>,~Unit,~Bands.Geometry,
          ~Bands.unshared,~Select.selected_leaf,~Bands.left_of(~Storage.Buffer<F32>,~Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>),
          Model.layout(~F32,yes(left)),Model.layout(~F32,yes(right)),enabled,geometry,paired(left),paired(right),Unit{})) :
        {Model.contents(~F32,_) == Model.Joined{model(geometry,left),model(geometry,right)} : Model.Content<F32>}
      %Equal.sym(Model.Content<F32>,Model.contents(~F32,run(enabled,geometry,left)),model(geometry,left),each(geometry,left,enabled)) :
        {Model.Joined{_,Model.contents(~F32,run(enabled,geometry,right))} == Model.Joined{model(geometry,left),model(geometry,right)} : Model.Content<F32>}
      %Equal.sym(Model.Content<F32>,Model.contents(~F32,run(enabled,geometry,right)),model(geometry,right),each(geometry,right,enabled)) :
        {Model.Joined{model(geometry,left),_} == Model.Joined{model(geometry,left),model(geometry,right)} : Model.Content<F32>}
      {==}

def selected_bands(+geometry: Bands.Geometry,+parallel: Bool,+owners: Model.Shaped<Three>) ->
  {Model.contents(~F32,Select.selected_bands(geometry,Model.layout(~F32,yes(owners)),parallel,Model.pack(~F32,mask(owners)),Model.pack(~F32,yes(owners)),Model.pack(~F32,no(owners)))) == model(geometry,owners) : Model.Content<F32>}:
  each(geometry,owners,parallel)
