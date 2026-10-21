# The window apply of any checked leaf. On the plan of its cores, with a halo
# of the same shape for every core, each core runs its call with its halo; a
# call that was not usable makes the result unusable. Window operations prove
# only the rule of their own call, and only about usable calls.
import Base
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition
import ./tensor_band_model.bend as Model
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_band_halo.bend as BandHalo
import ./tensor_band_each.bend as Each

# What holds of every core and its halo. Nothing holds when their shapes
# differ.
def HaloAll(~Element: Data,~Scope: Data,~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+core: Buffer.Content<Element> -> @+halo: BandHalo.Seen<Element> -> Data,
  +scope: Scope,cores: Model.Content<Element>,halos: Model.Shaped<BandHalo.Seen<Element>>) -> Data:
  match cores halos:
    case Model.RowBand{core,first,rows} Model.Piece{halo,halo_first,halo_rows}: First(scope,first,rows,core,halo)
    case Model.Joined{left,right} Model.Pieces{left_halo,right_halo}:
      Shape.Both<HaloAll(~Element,~Scope,~First,scope,left,left_halo),HaloAll(~Element,~Scope,~First,scope,right,right_halo)>
    case cores halos: Empty

def Applied(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> BandWindow.Halo<Element> -> Bands.Transformed(~Element),
  +enabled: Bool,+context: Context,+cores: Model.Model<Element>,halos: Model.Shaped<BandHalo.Owned<Element>>) -> Bands.Bands(~Element) & Bool:
  BandWindow.apply(~Element,~Context,~leaf,Model.layout(~Element,cores),enabled,context,Model.pack(~Element,cores),BandHalo.pack_owners(~Element,halos))

# Every core beside its halo, as the apply pairs them.
def pairs(~Element: Data,+cores: Model.Model<Element>,+halos: Model.Shaped<BandHalo.Owned<Element>>) ->
  Partition.Tree<&1,Bands.Paired<Storage.Buffer<Element>,BandWindow.Halo<Element>>,Unit>:
  Bands.zip(~Storage.Buffer<Element>,~BandWindow.Halo<Element>,Model.pack(~Element,cores),BandHalo.pack_owners(~Element,halos))

# Every band of a checked window apply, from the rule for one call. What the
# rule needs travels as one scope value, from which the context of the call
# is read.
def checked_all(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> BandWindow.Halo<Element> -> Bands.Transformed(~Element),
  ~Scope: Data,~context: @+scope: Scope -> Context,
  ~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+core: Buffer.Content<Element> -> @+halo: BandHalo.Seen<Element> -> Data,
  ~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~rule: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+core: Buffer.Model<Element> -> @+halo: BandHalo.Owned<Element> ->
    @fact: First(scope,first,rows,Buffer.content(~Element,core),BandHalo.owned_content(~Element,halo)) ->
    @accepted: {Each.usable(~Element,Each.checked(~Element,leaf(context(scope),first,rows,Buffer.pack(~Element,core),BandHalo.pack_owned(~Element,halo)))) == True{} : Bool} ->
    Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,leaf(context(scope),first,rows,Buffer.pack(~Element,core),BandHalo.pack_owned(~Element,halo)))))),
  +scope: Scope,cores: Model.Model<Element>,halos: Model.Shaped<BandHalo.Owned<Element>>,+enabled: Bool,
  facts: HaloAll(~Element,~Scope,~First,scope,Model.content(~Element,cores),BandHalo.owner_contents(~Element,halos)),
  accepted: {Each.usable(~Element,Applied(~Element,~Context,~leaf,enabled,context(scope),cores,halos)) == True{} : Bool}) ->
  Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,Applied(~Element,~Context,~leaf,enabled,context(scope),cores,halos)))):
  match cores halos:
    case Model.Leaf{+core,+first,+rows} Model.Piece{+halo,halo_first,halo_rows}: rule(scope,first,rows,core,halo,facts,accepted)
    case Model.Fork{+left,+right} Model.Pieces{+left_halo,+right_halo}:
      Shape.Both{left_facts,right_facts} = facts
      +halves = Each.fork(~Bands.Paired<Storage.Buffer<Element>,BandWindow.Halo<Element>>,~Bands.Checked<Element>,~Unit,~Context,~Bands.unshared,
        ~BandWindow.window_band(~Element,~Context,~leaf),~BandWindow.unwindowed(~Element),Model.layout(~Element,left),Model.layout(~Element,right),enabled,context(scope),
        pairs(~Element,left,left_halo),pairs(~Element,right,right_halo),Unit{})
      +left_mode = enabled
      +right_mode = enabled
      %Equal.sym(Each.Tree(~Bands.Checked<Element>),
        Bands.each(~Bands.Paired<Storage.Buffer<Element>,BandWindow.Halo<Element>>,~Bands.Checked<Element>,~Unit,~Context,~Bands.unshared,
          ~BandWindow.window_band(~Element,~Context,~leaf),~BandWindow.unwindowed(~Element),Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},enabled,context(scope),
          Partition.Fork{pairs(~Element,left,left_halo),pairs(~Element,right,right_halo)},Unit{}),
        Each.forked(~Bands.Paired<Storage.Buffer<Element>,BandWindow.Halo<Element>>,~Bands.Checked<Element>,~Unit,~Context,~Bands.unshared,
          ~BandWindow.window_band(~Element,~Context,~leaf),~BandWindow.unwindowed(~Element),Model.layout(~Element,left),Model.layout(~Element,right),enabled,context(scope),
          pairs(~Element,left,left_halo),pairs(~Element,right,right_halo),Bands.unshared(Unit{})),halves) :
        Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,_))))
      Each.checked_joined(~Element,~Scope,~After,scope,
        Applied(~Element,~Context,~leaf,enabled,context(scope),left,left_halo),Applied(~Element,~Context,~leaf,enabled,context(scope),right,right_halo),
        left_accepted => checked_all(~Element,~Context,~leaf,~Scope,~context,~First,~After,~rule,scope,left,left_halo,enabled,left_facts,left_accepted),
        right_accepted => checked_all(~Element,~Context,~leaf,~Scope,~context,~First,~After,~rule,scope,right,right_halo,enabled,right_facts,right_accepted),
        %halves : {Each.usable(~Element,Each.checked(~Element,_)) == True{} : Bool} accepted)
    case Model.Leaf{core,first,rows} Model.Pieces{left_halo,right_halo}: Empty.absurd(
      Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,
        Applied(~Element,~Context,~leaf,enabled,context(scope),Model.Leaf{core,first,rows},Model.Pieces{left_halo,right_halo})))),facts)
    case Model.Fork{left,right} Model.Piece{halo,halo_first,halo_rows}: Empty.absurd(
      Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,
        Applied(~Element,~Context,~leaf,enabled,context(scope),Model.Fork{left,right},Model.Piece{halo,halo_first,halo_rows})))),facts)

# What holds of every core with the halo the prepare extracts for it, from a
# rule for one core.
def halo_all(~Element: Data,~Scope: Data,~Before: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+core: Buffer.Content<Element> -> Data,
  ~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+core: Buffer.Content<Element> -> @+halo: BandHalo.Seen<Element> -> Data,
  ~extraction: @+scope: Scope -> BandHalo.Extraction<Element>,
  ~rule: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+core: Buffer.Content<Element> -> @fact: Before(scope,first,rows,core) ->
    First(scope,first,rows,core,BandHalo.band_model(~Element,extraction(scope),first,rows)),
  +scope: Scope,cores: Model.Content<Element>,facts: Model.ContentAll(~Element,~Before(scope),cores)) ->
  HaloAll(~Element,~Scope,~First,scope,cores,
    Model.shaped_build(~BandHalo.Seen<Element>,~BandHalo.Extraction<Element>,~BandHalo.band_model(~Element),extraction(scope),Model.content_layout(~Element,cores))):
  match cores:
    case Model.RowBand{+core,+first,+rows}: rule(scope,first,rows,core,facts)
    case Model.Joined{+left,+right}:
      Shape.Both{left_facts,right_facts} = facts
      Shape.Both{halo_all(~Element,~Scope,~Before,~First,~extraction,~rule,scope,left,left_facts),
        halo_all(~Element,~Scope,~Before,~First,~extraction,~rule,scope,right,right_facts)}

# What the prepared cores and halos give the apply: the plan is the plan of
# the cores, and every core and its halo hold the facts of the call.
def Prepared(~Element: Data,~Scope: Data,~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+core: Buffer.Content<Element> -> @+halo: BandHalo.Seen<Element> -> Data,
  +scope: Scope,+layout: Bands.Plan,content: BandHalo.Content<Element>) -> Data:
  BandHalo.Content{+cores,+halos} = content
  Shape.Both<{layout == Model.content_layout(~Element,cores) : Bands.Plan},HaloAll(~Element,~Scope,~First,scope,cores,halos)>

def modeled_all(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> BandWindow.Halo<Element> -> Bands.Transformed(~Element),
  ~Scope: Data,~context: @+scope: Scope -> Context,
  ~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+core: Buffer.Content<Element> -> @+halo: BandHalo.Seen<Element> -> Data,
  ~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~rule: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+core: Buffer.Model<Element> -> @+halo: BandHalo.Owned<Element> ->
    @fact: First(scope,first,rows,Buffer.content(~Element,core),BandHalo.owned_content(~Element,halo)) ->
    @accepted: {Each.usable(~Element,Each.checked(~Element,leaf(context(scope),first,rows,Buffer.pack(~Element,core),BandHalo.pack_owned(~Element,halo)))) == True{} : Bool} ->
    Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,leaf(context(scope),first,rows,Buffer.pack(~Element,core),BandHalo.pack_owned(~Element,halo)))))),
  +scope: Scope,+layout: Bands.Plan,+enabled: Bool,+cores: Model.Model<Element>,+halos: Model.Shaped<BandHalo.Owned<Element>>,
  facts: Prepared(~Element,~Scope,~First,scope,layout,BandHalo.refinement_contents(~Element,(Model.pack(~Element,cores),BandHalo.pack_owners(~Element,halos)))),
  accepted: {Each.usable(~Element,BandWindow.apply(~Element,~Context,~leaf,layout,enabled,context(scope),Model.pack(~Element,cores),BandHalo.pack_owners(~Element,halos))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,
    BandWindow.apply(~Element,~Context,~leaf,layout,enabled,context(scope),Model.pack(~Element,cores),BandHalo.pack_owners(~Element,halos))))):
  Shape.Both{+planned,core_facts} = facts
  +same = Equal.trans(Bands.Plan,layout,Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,cores))),Model.layout(~Element,cores),planned,
    %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,cores)),Model.content(~Element,cores),Model.packed_contents(~Element,cores)) :
      {Model.content_layout(~Element,_) == Model.layout(~Element,cores) : Bands.Plan}
    Equal.sym(Bands.Plan,Model.layout(~Element,cores),Model.content_layout(~Element,Model.content(~Element,cores)),Model.layout_matches_content(~Element,cores)))
  %Equal.sym(Bands.Plan,layout,Model.layout(~Element,cores),same) :
    Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,
      BandWindow.apply(~Element,~Context,~leaf,_,enabled,context(scope),Model.pack(~Element,cores),BandHalo.pack_owners(~Element,halos)))))
  checked_all(~Element,~Context,~leaf,~Scope,~context,~First,~After,~rule,scope,cores,halos,enabled,
    %Model.packed_contents(~Element,cores) :
      HaloAll(~Element,~Scope,~First,scope,_,BandHalo.owner_contents(~Element,halos))
    %BandHalo.packed_owners(~Element,halos) :
      HaloAll(~Element,~Scope,~First,scope,Model.contents(~Element,Model.pack(~Element,cores)),_)
    core_facts,
    %same : {Each.usable(~Element,BandWindow.apply(~Element,~Context,~leaf,_,enabled,context(scope),Model.pack(~Element,cores),BandHalo.pack_owners(~Element,halos))) == True{} : Bool}
    accepted)

# Every band of a window apply on the cores and halos a prepare returned.
def observed_all(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> BandWindow.Halo<Element> -> Bands.Transformed(~Element),
  ~Scope: Data,~context: @+scope: Scope -> Context,
  ~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+core: Buffer.Content<Element> -> @+halo: BandHalo.Seen<Element> -> Data,
  ~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~rule: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+core: Buffer.Model<Element> -> @+halo: BandHalo.Owned<Element> ->
    @fact: First(scope,first,rows,Buffer.content(~Element,core),BandHalo.owned_content(~Element,halo)) ->
    @accepted: {Each.usable(~Element,Each.checked(~Element,leaf(context(scope),first,rows,Buffer.pack(~Element,core),BandHalo.pack_owned(~Element,halo)))) == True{} : Bool} ->
    Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,leaf(context(scope),first,rows,Buffer.pack(~Element,core),BandHalo.pack_owned(~Element,halo)))))),
  +scope: Scope,+layout: Bands.Plan,+enabled: Bool,-cores: Bands.Bands(~Element),-halos: BandWindow.Halos(~Element),
  cores_observation: Model.Observation<Element,cores>,halos_observation: BandHalo.OwnerObservation<Element,halos>,
  facts: Prepared(~Element,~Scope,~First,scope,layout,BandHalo.refinement_contents(~Element,(cores,halos))),
  accepted: {Each.usable(~Element,BandWindow.apply(~Element,~Context,~leaf,layout,enabled,context(scope),cores,halos)) == True{} : Bool}) ->
  Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,BandWindow.apply(~Element,~Context,~leaf,layout,enabled,context(scope),cores,halos)))):
  Model.Observation{+core_model,+core_equal} = cores_observation
  BandHalo.OwnerObservation{+halo_model,+halo_equal} = halos_observation
  %core_equal : Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,BandWindow.apply(~Element,~Context,~leaf,layout,enabled,context(scope),_,halos))))
  %halo_equal : Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,
    BandWindow.apply(~Element,~Context,~leaf,layout,enabled,context(scope),Model.pack(~Element,core_model),_))))
  modeled_all(~Element,~Context,~leaf,~Scope,~context,~First,~After,~rule,scope,layout,enabled,core_model,halo_model,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,core_model),cores,core_equal) :
      Prepared(~Element,~Scope,~First,scope,layout,BandHalo.refinement_contents(~Element,(_,BandHalo.pack_owners(~Element,halo_model))))
    %Equal.sym(BandWindow.Halos(~Element),BandHalo.pack_owners(~Element,halo_model),halos,halo_equal) :
      Prepared(~Element,~Scope,~First,scope,layout,BandHalo.refinement_contents(~Element,(cores,_)))
    facts,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,core_model),cores,core_equal) :
      {Each.usable(~Element,BandWindow.apply(~Element,~Context,~leaf,layout,enabled,context(scope),_,BandHalo.pack_owners(~Element,halo_model))) == True{} : Bool}
    %Equal.sym(BandWindow.Halos(~Element),BandHalo.pack_owners(~Element,halo_model),halos,halo_equal) :
      {Each.usable(~Element,BandWindow.apply(~Element,~Context,~leaf,layout,enabled,context(scope),cores,_)) == True{} : Bool}
    accepted)
