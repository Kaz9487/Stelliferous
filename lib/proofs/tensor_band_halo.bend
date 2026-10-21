# Halo preparation and extraction: preserved core contents and the values of
# every allocated halo range.
import Base
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_model.bend as Model
import ./tensor_band_range.bend as BandRange
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ./tensor_band_repartition.bend as BandRepartition
import ../machine_limits.bend as Machine
import ./tensor_band_from_whole.bend as FromWhole
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ../storage_buffer.bend as Storage
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as StorageTree
import ../traversal_partition.bend as Partition

# The actual halo extractor allocates a destination and invokes the shared
# range reader. Its source remains intact and its result uses the range model.
def extract_refinement_result(~Element: Data,geometry: Bands.Geometry,fill: Element,rows: BandWindow.Rows,source: Model.Content<Element>) -> Buffer.Content<Element>:
  BandWindow.Rows{first,count} = rows
  BandRepartition.leaf_buffer(~Element,geometry,fill,first,count,source)

def packed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+rows: BandWindow.Rows,+source: Model.Model<Element>) ->
  {BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,rows,Model.pack(~Element,source))) ==
    BandRange.Content{Model.content(~Element,source),extract_refinement_result(~Element,geometry,fill,rows,Model.content(~Element,source))} : BandRange.Content<Element>}:
  BandWindow.Rows{+first,+count} = rows
  +request = {Bands.Range{geometry,first,count,0} : Bands.Range}
  +length = Bands.local_count(geometry,count)
  Equal.trans(BandRange.Content<Element>,BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,rows,Model.pack(~Element,source))),
    BandRange.written(~Element,request,Model.content(~Element,source),Buffer.contents(~Element,Bands.allocate(~Element,fill,length))),
    BandRange.Content{Model.content(~Element,source),extract_refinement_result(~Element,geometry,fill,rows,Model.content(~Element,source))},
    BandRange.native_range(~Element,request,source,Bands.allocate(~Element,fill,length)),
    Equal.cong(Buffer.Content<Element>,BandRange.Content<Element>,output => BandRange.written(~Element,request,Model.content(~Element,source),output),
      Buffer.contents(~Element,Bands.allocate(~Element,fill,length)),Buffer.Content{StorageTree.fresh(Element,Storage.depth_for(length),fill),U32.to_nat(length)},
      AllocationLaws.contents(~Element,Storage.depth_for(length),fill,U32.to_nat(length))))

type ExtractRefinementContext<-Element: Data> is Data:
  ExtractRefinementContext{geometry: Bands.Geometry,fill: Element,rows: BandWindow.Rows}

def extract_refinement_proposition(~Element: Data,context: ExtractRefinementContext<Element>,-source: Bands.Bands(~Element)) -> Type:
  ExtractRefinementContext{+geometry,+fill,+rows} = context
  {BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,rows,source)) ==
    BandRange.Content{Model.contents(~Element,source),extract_refinement_result(~Element,geometry,fill,rows,Model.contents(~Element,source))} : BandRange.Content<Element>}

def extract(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+rows: BandWindow.Rows,source: Bands.Bands(~Element)) ->
  {BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,rows,source)) ==
    BandRange.Content{Model.contents(~Element,source),extract_refinement_result(~Element,geometry,fill,rows,Model.contents(~Element,source))} : BandRange.Content<Element>}:
  Model.lift(~Element,~ExtractRefinementContext<Element>,~extract_refinement_proposition(~Element),ExtractRefinementContext{geometry,fill,rows},source,Model.observe(~Element,source),model =>
    +model = model
    %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,model)),Model.content(~Element,model),Model.packed_contents(~Element,model)) :
      {BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,rows,Model.pack(~Element,model))) == BandRange.Content{_,extract_refinement_result(~Element,geometry,fill,rows,_)} : BandRange.Content<Element>}
    packed(~Element,geometry,fill,rows,model))

# The real preparation pass preserves core contents and constructs exactly
# the requested upper/lower buffers. No kernel or reduction is modeled here.
# What is seen of one halo, and its owners with their certificates.
type Seen<-Element: Data> is Data:
  Seen{above: Buffer.Content<Element>,below: Buffer.Content<Element>,window: BandWindow.RowWindow}

type Owned<-Element: Data> is Data:
  Owned{above: Buffer.Model<Element>,below: Buffer.Model<Element>,window: BandWindow.RowWindow}

def pack_owned(~Element: Data,owned: Owned<Element>) -> BandWindow.Halo<Element>:
  Owned{above,below,window} = owned
  BandWindow.Halo{Buffer.pack(~Element,above),Buffer.pack(~Element,below),window}

def owned_content(~Element: Data,owned: Owned<Element>) -> Seen<Element>:
  Owned{above,below,window} = owned
  Seen{Buffer.content(~Element,above),Buffer.content(~Element,below),window}

def seen(~Element: Data,halo: BandWindow.Halo<Element>) -> Seen<Element>:
  BandWindow.Halo{above,below,window} = halo
  Seen{Buffer.contents(~Element,above),Buffer.contents(~Element,below),window}

def pack_owners(~Element: Data,owners: Model.Shaped<Owned<Element>>) -> BandWindow.Halos(~Element):
  Model.shaped_pack(~Owned<Element>,~BandWindow.Halo<Element>,~pack_owned(~Element),owners)

def owner_contents(~Element: Data,owners: Model.Shaped<Owned<Element>>) -> Model.Shaped<Seen<Element>>:
  Model.shaped_map(~Owned<Element>,~Seen<Element>,~owned_content(~Element),owners)

type OwnerObservation<-Element: Data,-halos: BandWindow.Halos(~Element)> is Data:
  OwnerObservation{owners: Model.Shaped<Owned<Element>>,equation: {pack_owners(~Element,owners) == halos : BandWindow.Halos(~Element)}}

def observed_halo(~Element: Data,+window: BandWindow.RowWindow,-above: Storage.Buffer<Element>,-below: Storage.Buffer<Element>,
  above_observation: Buffer.Observation<Element,above>,below_observation: Buffer.Observation<Element,below>) ->
  Model.Packed<BandWindow.Halo<Element>,Owned<Element>,pack_owned(~Element),BandWindow.Halo{above,below,window}>:
  Buffer.Observation{+above_model,above_equal} = above_observation
  Buffer.Observation{+below_model,below_equal} = below_observation
  Model.Packed{Owned{above_model,below_model,window},
    %above_equal : {BandWindow.Halo{Buffer.pack(~Element,above_model),Buffer.pack(~Element,below_model),window} == BandWindow.Halo{_,below,window} : BandWindow.Halo<Element>}
    %below_equal : {BandWindow.Halo{Buffer.pack(~Element,above_model),Buffer.pack(~Element,below_model),window} == BandWindow.Halo{Buffer.pack(~Element,above_model),_,window} : BandWindow.Halo<Element>}
    {==}}

def observe_halo(~Element: Data,halo: BandWindow.Halo<Element>) -> Model.Packed<BandWindow.Halo<Element>,Owned<Element>,pack_owned(~Element),halo>:
  BandWindow.Halo{above,below,+window} = halo
  observed_halo(~Element,window,above,below,Buffer.observe(~Element,above),Buffer.observe(~Element,below))

def observed_owners(~Element: Data,-halos: BandWindow.Halos(~Element),observation: Model.ShapedObservation<Owned<Element>,BandWindow.Halo<Element>,pack_owned(~Element),halos>) -> OwnerObservation<Element,halos>:
  Model.ShapedObservation{owners,equation} = observation
  OwnerObservation{owners,equation}

def observe_owners(~Element: Data,halos: BandWindow.Halos(~Element)) -> OwnerObservation<Element,halos>:
  observed_owners(~Element,halos,Model.shaped_observe(~Owned<Element>,~BandWindow.Halo<Element>,~pack_owned(~Element),~observe_halo(~Element),halos))

type Content<-Element: Data> is Data:
  Content{source: Model.Content<Element>,halos: Model.Shaped<Seen<Element>>}

def reflect(~Element: Data,halos: BandWindow.Halos(~Element)) -> Model.Shaped<Seen<Element>>:
  Model.shaped_seen(~BandWindow.Halo<Element>,~Seen<Element>,~seen(~Element),halos)

def packed_owned(~Element: Data,+owned: Owned<Element>) -> {seen(~Element,pack_owned(~Element,owned)) == owned_content(~Element,owned) : Seen<Element>}:
  Owned{+above,+below,+window} = owned
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,above)),Buffer.content(~Element,above),Buffer.packed_contents(~Element,above)) :
    {Seen{_,Buffer.contents(~Element,Buffer.pack(~Element,below)),window} == Seen{Buffer.content(~Element,above),Buffer.content(~Element,below),window} : Seen<Element>}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,below)),Buffer.content(~Element,below),Buffer.packed_contents(~Element,below)) :
    {Seen{Buffer.content(~Element,above),_,window} == Seen{Buffer.content(~Element,above),Buffer.content(~Element,below),window} : Seen<Element>}
  {==}

def packed_owners(~Element: Data,+owners: Model.Shaped<Owned<Element>>) ->
  {reflect(~Element,pack_owners(~Element,owners)) == owner_contents(~Element,owners) : Model.Shaped<Seen<Element>>}:
  Model.shaped_packed(~Owned<Element>,~Seen<Element>,~BandWindow.Halo<Element>,~pack_owned(~Element),~seen(~Element),~owned_content(~Element),~packed_owned(~Element),owners)

def refinement_contents(~Element: Data,owners: Bands.Bands(~Element) & BandWindow.Halos(~Element)) -> Content<Element>:
  (source,halos) = owners
  Content{Model.contents(~Element,source),reflect(~Element,halos)}

def seen_model(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+window: BandWindow.RowWindow,+source: Model.Content<Element>) -> Seen<Element>:
  BandWindow.RowWindow{output,+above,+below,read} = window
  Seen{extract_refinement_result(~Element,geometry,fill,above,source),extract_refinement_result(~Element,geometry,fill,below,source),window}

def leaf_model(~Element: Data,+geometry: Bands.Geometry,+fill: Element,first: U32,rows: U32,+window: BandWindow.RowWindow,+source: Model.Content<Element>) -> Model.Shaped<Seen<Element>>:
  Model.Piece{seen_model(~Element,geometry,fill,window,source),first,rows}
def completed(~Element: Data,first: U32,rows: U32,window: BandWindow.RowWindow,above: Buffer.Content<Element>,lower: BandRange.Content<Element>) -> Content<Element>:
  BandRange.Content{source,below} = lower
  Content{source,Model.Piece{Seen{above,below,window},first,rows}}

def finish(~Element: Data,+first: U32,+rows: U32,+window: BandWindow.RowWindow,above: Storage.Buffer<Element>,lower: Bands.Bands(~Element) & Storage.Buffer<Element>) -> Bands.Bands(~Element) & BandWindow.Halos(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Bands.Bands(~Element) & BandWindow.Halos(~Element)),lower,
    source => below => (source,Partition.Leaf{BandWindow.Halo{above,below,window},first,rows,Unit{}}))

def finished(~Element: Data,+first: U32,+rows: U32,+window: BandWindow.RowWindow,-above: Storage.Buffer<Element>,lower: Bands.Bands(~Element) & Storage.Buffer<Element>) ->
  {refinement_contents(~Element,finish(~Element,first,rows,window,above,lower)) == completed(~Element,first,rows,window,Buffer.contents(~Element,above),BandRange.contents(~Element,lower)) : Content<Element>}:
  (source,below) = lower
  {==}

def after_upper(~Element: Data,geometry: Bands.Geometry,fill: Element,first: U32,rows: U32,+window: BandWindow.RowWindow,upper: BandRange.Content<Element>) -> Content<Element>:
  BandWindow.RowWindow{output,above,+below,read} = window
  BandRange.Content{+source,upper} = upper
  Content{source,Model.Piece{Seen{upper,extract_refinement_result(~Element,geometry,fill,below,source),window},first,rows}}

def below(window: BandWindow.RowWindow) -> BandWindow.Rows:
  BandWindow.RowWindow{output,above,below,read} = window
  below

def lower_packed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+source_model: Model.Model<Element>,-above: Storage.Buffer<Element>) ->
  {refinement_contents(~Element,finish(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below(window),Model.pack(~Element,source_model)))) ==
    after_upper(~Element,geometry,fill,first,rows,window,BandRange.Content{Model.contents(~Element,Model.pack(~Element,source_model)),Buffer.contents(~Element,above)}) : Content<Element>}:
  BandWindow.RowWindow{output,upper_rows,below_rows,read} = window
  -source = Model.pack(~Element,source_model)
  Equal.trans(Content<Element>,refinement_contents(~Element,finish(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below_rows,source))),
    completed(~Element,first,rows,window,Buffer.contents(~Element,above),BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,below_rows,source))),
    after_upper(~Element,geometry,fill,first,rows,window,BandRange.Content{Model.contents(~Element,source),Buffer.contents(~Element,above)}),
    finished(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below_rows,Model.pack(~Element,source_model))),
    Equal.cong(BandRange.Content<Element>,Content<Element>,result => completed(~Element,first,rows,window,Buffer.contents(~Element,above),result),
      BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,below_rows,source)),
      BandRange.Content{Model.contents(~Element,source),extract_refinement_result(~Element,geometry,fill,below_rows,Model.contents(~Element,source))},extract(~Element,geometry,fill,below_rows,Model.pack(~Element,source_model))))

def lower_observed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,-source: Bands.Bands(~Element),-above: Storage.Buffer<Element>,observation: Model.Observation<Element,source>) ->
  {refinement_contents(~Element,finish(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below(window),source))) ==
    after_upper(~Element,geometry,fill,first,rows,window,BandRange.Content{Model.contents(~Element,source),Buffer.contents(~Element,above)}) : Content<Element>}:
  Model.Observation{source_model,equation} = observation
  %equation : {refinement_contents(~Element,finish(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below(window),_))) ==
    after_upper(~Element,geometry,fill,first,rows,window,BandRange.Content{Model.contents(~Element,_),Buffer.contents(~Element,above)}) : Content<Element>}
  lower_packed(~Element,geometry,fill,first,rows,window,source_model,above)

def continued(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,upper: Bands.Bands(~Element) & Storage.Buffer<Element>) -> Bands.Bands(~Element) & BandWindow.Halos(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Bands.Bands(~Element) & BandWindow.Halos(~Element)),upper,
    source => above => finish(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below(window),source)))

def continuation(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,upper: Bands.Bands(~Element) & Storage.Buffer<Element>) ->
  {refinement_contents(~Element,continued(~Element,geometry,fill,first,rows,window,upper)) == after_upper(~Element,geometry,fill,first,rows,window,BandRange.contents(~Element,upper)) : Content<Element>}:
  (source,above) = upper
  lower_observed(~Element,geometry,fill,first,rows,window,source,above,Model.observe(~Element,source))

def halo_packed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+source_model: Model.Model<Element>) ->
  {refinement_contents(~Element,BandWindow.halo(~Element,geometry,fill,first,rows,window,Model.pack(~Element,source_model))) ==
    Content{Model.contents(~Element,Model.pack(~Element,source_model)),leaf_model(~Element,geometry,fill,first,rows,window,Model.contents(~Element,Model.pack(~Element,source_model)))} : Content<Element>}:
  BandWindow.RowWindow{output,above,below,read} = window
  -source = Model.pack(~Element,source_model)
  Equal.trans(Content<Element>,refinement_contents(~Element,BandWindow.halo(~Element,geometry,fill,first,rows,window,source)),
    after_upper(~Element,geometry,fill,first,rows,window,BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,above,source))),
    Content{Model.contents(~Element,source),leaf_model(~Element,geometry,fill,first,rows,window,Model.contents(~Element,source))},
    continuation(~Element,geometry,fill,first,rows,window,BandWindow.extract(~Element,geometry,fill,above,Model.pack(~Element,source_model))),
    Equal.cong(BandRange.Content<Element>,Content<Element>,result => after_upper(~Element,geometry,fill,first,rows,window,result),
      BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,above,source)),
      BandRange.Content{Model.contents(~Element,source),extract_refinement_result(~Element,geometry,fill,above,Model.contents(~Element,source))},extract(~Element,geometry,fill,above,Model.pack(~Element,source_model))))

type LeafContext<-Element: Data> is Data:
  LeafContext{geometry: Bands.Geometry,fill: Element,first: U32,rows: U32,window: BandWindow.RowWindow}

def leaf_proposition(~Element: Data,context: LeafContext<Element>,-source: Bands.Bands(~Element)) -> Type:
  LeafContext{+geometry,+fill,+first,+rows,+window} = context
  {refinement_contents(~Element,BandWindow.halo(~Element,geometry,fill,first,rows,window,source)) ==
    Content{Model.contents(~Element,source),leaf_model(~Element,geometry,fill,first,rows,window,Model.contents(~Element,source))} : Content<Element>}

def halo(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,source: Bands.Bands(~Element)) ->
  {refinement_contents(~Element,BandWindow.halo(~Element,geometry,fill,first,rows,window,source)) ==
    Content{Model.contents(~Element,source),leaf_model(~Element,geometry,fill,first,rows,window,Model.contents(~Element,source))} : Content<Element>}:
  Model.lift(~Element,~LeafContext<Element>,~leaf_proposition(~Element),LeafContext{geometry,fill,first,rows,window},source,Model.observe(~Element,source),model => halo_packed(~Element,geometry,fill,first,rows,window,model))

# What the halo of every band is extracted from.
type Extraction<-Element: Data> is Data:
  Extraction{geometry: BandWindow.Geometry,fill: Element,source: Model.Content<Element>}

def band_model(~Element: Data,+context: Extraction<Element>,+first: U32,+rows: U32) -> Seen<Element>:
  Extraction{+geometry,+fill,+source} = context
  seen_model(~Element,BandWindow.input_geometry(geometry),fill,BandWindow.window(geometry,first,rows),source)

def model(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Content<Element>) -> Model.Shaped<Seen<Element>>:
  Model.shaped_build(~Seen<Element>,~Extraction<Element>,~band_model(~Element),Extraction{geometry,fill,source},layout)
def joined(~Element: Data,left: Model.Shaped<Seen<Element>>,right: Content<Element>) -> Content<Element>:
  Content{source,right} = right
  Content{source,Model.Pieces{left,right}}

def join_native(~Element: Data,left: BandWindow.Halos(~Element),right: Bands.Bands(~Element) & BandWindow.Halos(~Element)) -> Bands.Bands(~Element) & BandWindow.Halos(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~BandWindow.Halos(~Element),~(Bands.Bands(~Element) & BandWindow.Halos(~Element)),right,
    source => right => (source,Partition.Fork{left,right}))

def join_contents(~Element: Data,-left: BandWindow.Halos(~Element),right: Bands.Bands(~Element) & BandWindow.Halos(~Element)) ->
  {refinement_contents(~Element,join_native(~Element,left,right)) == joined(~Element,reflect(~Element,left),refinement_contents(~Element,right)) : Content<Element>}:
  (source,right) = right
  {==}

def after_left(~Element: Data,geometry: BandWindow.Geometry,fill: Element,layout: Bands.Plan,left: Content<Element>) -> Content<Element>:
  Content{+source,left} = left
  Content{source,Model.Pieces{left,model(~Element,geometry,fill,layout,source)}}

def continue_tree(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,+layout: Bands.Plan,left: Bands.Bands(~Element) & BandWindow.Halos(~Element)) -> Bands.Bands(~Element) & BandWindow.Halos(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~BandWindow.Halos(~Element),~(Bands.Bands(~Element) & BandWindow.Halos(~Element)),left,
    source => left => join_native(~Element,left,BandWindow.prepare(~Element,geometry,fill,layout,source)))

def continue_packed(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,+layout: Bands.Plan,+source_model: Model.Model<Element>,-left: BandWindow.Halos(~Element),
  rule: @source: Bands.Bands(~Element) -> {refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)) == Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))} : Content<Element>}) ->
  {refinement_contents(~Element,continue_tree(~Element,geometry,fill,layout,(Model.pack(~Element,source_model),left))) == after_left(~Element,geometry,fill,layout,refinement_contents(~Element,(Model.pack(~Element,source_model),left))) : Content<Element>}:
  -source = Model.pack(~Element,source_model)
  Equal.trans(Content<Element>,refinement_contents(~Element,join_native(~Element,left,BandWindow.prepare(~Element,geometry,fill,layout,source))),
    joined(~Element,reflect(~Element,left),refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source))),
    after_left(~Element,geometry,fill,layout,refinement_contents(~Element,(source,left))),
    join_contents(~Element,left,BandWindow.prepare(~Element,geometry,fill,layout,Model.pack(~Element,source_model))),
    Equal.cong(Content<Element>,Content<Element>,result => joined(~Element,reflect(~Element,left),result),
      refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)),Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))},rule(Model.pack(~Element,source_model))))

def continue_observed(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,+layout: Bands.Plan,-source: Bands.Bands(~Element),-left: BandWindow.Halos(~Element),observation: Model.Observation<Element,source>,
  rule: @source: Bands.Bands(~Element) -> {refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)) == Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))} : Content<Element>}) ->
  {refinement_contents(~Element,continue_tree(~Element,geometry,fill,layout,(source,left))) == after_left(~Element,geometry,fill,layout,refinement_contents(~Element,(source,left))) : Content<Element>}:
  Model.Observation{source_model,equation} = observation
  %equation : {refinement_contents(~Element,continue_tree(~Element,geometry,fill,layout,(_,left))) == after_left(~Element,geometry,fill,layout,refinement_contents(~Element,(_,left))) : Content<Element>}
  continue_packed(~Element,geometry,fill,layout,source_model,left,rule)

def continue_contents(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,+layout: Bands.Plan,left: Bands.Bands(~Element) & BandWindow.Halos(~Element),
  rule: @source: Bands.Bands(~Element) -> {refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)) == Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))} : Content<Element>}) ->
  {refinement_contents(~Element,continue_tree(~Element,geometry,fill,layout,left)) == after_left(~Element,geometry,fill,layout,refinement_contents(~Element,left)) : Content<Element>}:
  (source,left) = left
  continue_observed(~Element,geometry,fill,layout,source,left,Model.observe(~Element,source),rule)

type RefinementContext<-Element: Data> is Data:
  RefinementContext{geometry: BandWindow.Geometry,fill: Element,layout: Bands.Plan}

def refinement_proposition(~Element: Data,context: RefinementContext<Element>,-source: Bands.Bands(~Element)) -> Type:
  RefinementContext{+geometry,+fill,+layout} = context
  {refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)) == Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))} : Content<Element>}

def prepare(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,layout: Bands.Plan,source: Bands.Bands(~Element)) ->
  {refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)) == Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))} : Content<Element>}:
  match layout:
    case Bands.Rows{+first,+rows}: halo(~Element,BandWindow.input_geometry(geometry),fill,first,rows,BandWindow.window(geometry,first,rows),source)
    case Bands.Split{+left,+right}:
      Model.lift(~Element,~RefinementContext<Element>,~refinement_proposition(~Element),RefinementContext{geometry,fill,layout},source,Model.observe(~Element,source),source_model =>
        +source_model = source_model
        -source = Model.pack(~Element,source_model)
        Equal.trans(Content<Element>,refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)),
          after_left(~Element,geometry,fill,right,refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,left,source))),
          Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))},
          continue_contents(~Element,geometry,fill,right,BandWindow.prepare(~Element,geometry,fill,left,Model.pack(~Element,source_model)),input => prepare(~Element,geometry,fill,right,input)),
          Equal.cong(Content<Element>,Content<Element>,result => after_left(~Element,geometry,fill,right,result),
            refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,left,source)),Content{Model.contents(~Element,source),model(~Element,geometry,fill,left,Model.contents(~Element,source))},prepare(~Element,geometry,fill,left,Model.pack(~Element,source_model)))))

# Semantics of the real halo preparation result. Window span obligations are
# explicit; the proof constructs all buffer readback facts from source Values.
def RangeFacts(geometry: Bands.Geometry,rows: BandWindow.Rows,capacity: U32,division_capacity: U32) -> Data:
  BandWindow.Rows{first,count} = rows
  FromWhole.LeafFacts<geometry,first,count,capacity,division_capacity>

def LeafFacts(+geometry: Bands.Geometry,window: BandWindow.RowWindow,+capacity: U32,+division_capacity: U32) -> Data:
  BandWindow.RowWindow{output,above,below,read} = window
  Shape.Both<RangeFacts(geometry,above,capacity,division_capacity),RangeFacts(geometry,below,capacity,division_capacity)>

def RangeValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,rows: BandWindow.Rows,+buffer: Buffer.Content<Element>) -> Data:
  BandWindow.Rows{+first,+count} = rows
  Buffer.Content{tree,length} = buffer
  Shape.Both<Buffer.ContentReady<Element,buffer>,Shape.Both<BandShape.Leaf(geometry,first,count,length),
    BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.RowBand{buffer,first,count})>>

# What the values of one halo are read against.
type Read<-Logical: Data> is Data:
  Read{logical: Logical,geometry: Bands.Geometry}

def seen_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+context: Read<Logical>,+part: Seen<Element>,+first: U32,+rows: U32) -> Data:
  Read{logical,geometry} = context
  Seen{above,below,BandWindow.RowWindow{output,above_rows,below_rows,read}} = part
  Shape.Both<RangeValues(~Element,~Logical,~original,logical,geometry,above_rows,above),RangeValues(~Element,~Logical,~original,logical,geometry,below_rows,below)>

def TreeValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,tree: Model.Shaped<Seen<Element>>) -> Data:
  Model.ShapedAll(~Seen<Element>,~Read<Logical>,~seen_values(~Element,~Logical,~original),Read{logical,geometry},tree)

def range(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+rows: BandWindow.Rows,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  +target: RangeFacts(geometry,rows,capacity,Machine.half_word(limits))) ->
  RangeValues(~Element,~Logical,~original,logical,geometry,rows,extract_refinement_result(~Element,geometry,fill,rows,Model.content(~Element,source))):
  BandWindow.Rows{+first,+count} = rows
  FromWhole.LeafFacts{allocation,contents,metadata} = target
  Shape.Both{BandRepartition.leaf_ready(~Element,geometry,fill,first,count,Model.content(~Element,source),capacity,Machine.half_word(limits),target),
    Shape.Both{metadata,BandRepartition.modeled_values(~Element,~Logical,~original,logical,geometry,fill,Bands.Rows{first,count},source,limits,capacity,domain,shape,values,target)}}

def leaf(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  +shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,+values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  target: LeafFacts(geometry,window,capacity,Machine.half_word(limits))) ->
  TreeValues(~Element,~Logical,~original,logical,geometry,leaf_model(~Element,geometry,fill,first,rows,window,Model.content(~Element,source))):
  BandWindow.RowWindow{output,+above,+below,read} = window
  Shape.Both{above_facts,below_facts} = target
  Shape.Both{range(~Element,~Logical,~original,logical,geometry,fill,above,source,limits,capacity,domain,shape,values,above_facts),
    range(~Element,~Logical,~original,logical,geometry,fill,below,source,limits,capacity,domain,shape,values,below_facts)}
