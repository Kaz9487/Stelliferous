# Canonical proof representation of the real band owners and their metadata.
# This representation is used only by refinement proofs, never by execution.
import Base
import ../tensor_band.bend as Bands
import ../traversal_partition.bend as Partition
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./traversal_partition_refinement.bend as PartitionLaws
import ./tensor_band_plan.bend as BandPlan
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition

type Model<-Element: Data> is Data:
  Leaf{buffer: Buffer.Model<Element>,first: U32,rows: U32}
  Fork{left: Model<Element>,right: Model<Element>}

type Content<-Element: Data> is Data:
  RowBand{buffer: Buffer.Content<Element>,first: U32,rows: U32}
  Joined{left: Content<Element>,right: Content<Element>}

def contents(~Element: Data,bands: Bands.Bands(~Element)) -> Content<Element>:
  match bands:
    case Partition.Leaf{buffer,first,rows,details}: RowBand{Buffer.contents(~Element,buffer),first,rows}
    case Partition.Fork{left,right}: Joined{contents(~Element,left),contents(~Element,right)}

def content(~Element: Data,model: Model<Element>) -> Content<Element>:
  match model:
    case Leaf{buffer,first,rows}: RowBand{Buffer.content(~Element,buffer),first,rows}
    case Fork{left,right}: Joined{content(~Element,left),content(~Element,right)}

# The three shapes of a band operation on contents: every band by itself,
# every pair of bands of two trees of one shape, and a tree made from a plan.
# An operation gives the contents of one band; the tree is walked here.
def map(~Element: Data,~band: @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,content: Content<Element>) -> Content<Element>:
  match content:
    case RowBand{buffer,+first,+rows}: RowBand{band(first,rows,buffer),first,rows}
    case Joined{left,right}: Joined{map(~Element,~band,left),map(~Element,~band,right)}

def build(~Element: Data,~band: @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,layout: Bands.Plan) -> Content<Element>:
  match layout:
    case Bands.Rows{+first,+rows}: RowBand{band(first,rows),first,rows}
    case Bands.Split{left,right}: Joined{build(~Element,~band,left),build(~Element,~band,right)}

# A property of every band of a model, and of every band of a content.
def All(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> Data,model: Model<Element>) -> Data:
  match model:
    case Leaf{buffer,first,rows}: Property(first,rows,buffer)
    case Fork{left,right}: Shape.Both<All(~Element,~Property,left),All(~Element,~Property,right)>

def ContentAll(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,content: Content<Element>) -> Data:
  match content:
    case RowBand{buffer,first,rows}: Property(first,rows,buffer)
    case Joined{left,right}: Shape.Both<ContentAll(~Element,~Property,left),ContentAll(~Element,~Property,right)>

def sized_band(~Element: Data,+first: U32,+rows: U32,+buffer: Buffer.Model<Element>) -> Data:
  Buffer.Sized<Element,buffer>

def ready_band(~Element: Data,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  Buffer.ContentReady<Element,buffer>

def Ready(~Element: Data,model: Model<Element>) -> Data:
  All(~Element,~sized_band(~Element),model)

def ContentReady(~Element: Data,content: Content<Element>) -> Data:
  ContentAll(~Element,~ready_band(~Element),content)

def ready(~Element: Data,model: Model<Element>,evidence: ContentReady(~Element,content(~Element,model))) -> Ready(~Element,model):
  match model:
    case Leaf{input,first,rows}: Buffer.ready_content(~Element,input,evidence)
    case Fork{+left,+right}:
      Shape.Both{left_ready,right_ready} = evidence
      Shape.Both{ready(~Element,left,left_ready),ready(~Element,right,right_ready)}

def pack(~Element: Data,model: Model<Element>) -> Bands.Bands(~Element):
  match model:
    case Leaf{buffer,first,rows}: Partition.Leaf{Buffer.pack(~Element,buffer),first,rows,Unit{}}
    case Fork{left,right}: Partition.Fork{pack(~Element,left),pack(~Element,right)}

def packed_contents(~Element: Data,+model: Model<Element>) -> {contents(~Element,pack(~Element,model)) == content(~Element,model) : Content<Element>}:
  match model:
    case Leaf{buffer,+first,+rows}:
      Equal.cong(Buffer.Content<Element>,Content<Element>,buffer => RowBand{buffer,first,rows},
        Buffer.contents(~Element,Buffer.pack(~Element,buffer)),Buffer.content(~Element,buffer),Buffer.packed_contents(~Element,buffer))
    case Fork{+left,+right}:
      %Equal.sym(Content<Element>,contents(~Element,pack(~Element,left)),content(~Element,left),packed_contents(~Element,left)) :
        {Joined{_,contents(~Element,pack(~Element,right))} == Joined{content(~Element,left),content(~Element,right)} : Content<Element>}
      %Equal.sym(Content<Element>,contents(~Element,pack(~Element,right)),content(~Element,right),packed_contents(~Element,right)) :
        {Joined{content(~Element,left),_} == Joined{content(~Element,left),content(~Element,right)} : Content<Element>}
      {==}

def layout(~Element: Data,model: Model<Element>) -> Bands.Plan:
  match model:
    case Leaf{buffer,first,rows}: Bands.Rows{first,rows}
    case Fork{left,right}: Bands.Split{layout(~Element,left),layout(~Element,right)}

def content_layout(~Element: Data,content: Content<Element>) -> Bands.Plan:
  match content:
    case RowBand{buffer,first,rows}: Bands.Rows{first,rows}
    case Joined{left,right}: Bands.Split{content_layout(~Element,left),content_layout(~Element,right)}

def layout_matches_content(~Element: Data,+model: Model<Element>) -> {layout(~Element,model) == content_layout(~Element,content(~Element,model)) : Bands.Plan}:
  match model:
    case Leaf{buffer,first,rows}: {==}
    case Fork{+left,+right}:
      %layout_matches_content(~Element,left) : {Bands.Split{layout(~Element,left),layout(~Element,right)} == Bands.Split{_,content_layout(~Element,content(~Element,right))} : Bands.Plan}
      %layout_matches_content(~Element,right) : {Bands.Split{layout(~Element,left),layout(~Element,right)} == Bands.Split{layout(~Element,left),_} : Bands.Plan}
      {==}

# Every leaf of a plan.
def PlanAll(~Context: Data,~Property: Context -> @+first: U32 -> @+rows: U32 -> Data,+context: Context,layout: Bands.Plan) -> Data:
  match layout:
    case Bands.Rows{first,rows}: Property(context,first,rows)
    case Bands.Split{left,right}: Shape.Both<PlanAll(~Context,~Property,context,left),PlanAll(~Context,~Property,context,right)>

# A fact about every leaf of a plan gives another fact about every leaf.
def plan_one(~Context: Data,~Before: Data,~First: Before -> @+first: U32 -> @+rows: U32 -> Data,~before: @+context: Context -> Before,
  ~After: Data,~Second: After -> @+first: U32 -> @+rows: U32 -> Data,~after: @+context: Context -> After,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+fact: First(before(context),first,rows) -> Second(after(context),first,rows),
  +context: Context,layout: Bands.Plan,facts: PlanAll(~Before,~First,before(context),layout)) -> PlanAll(~After,~Second,after(context),layout):
  match layout:
    case Bands.Rows{+first,+rows}: leaf(context,first,rows,facts)
    case Bands.Split{left,right}:
      Shape.Both{left_facts,right_facts} = facts
      Shape.Both{plan_one(~Context,~Before,~First,~before,~After,~Second,~after,~leaf,context,left,left_facts),
        plan_one(~Context,~Before,~First,~before,~After,~Second,~after,~leaf,context,right,right_facts)}

# A plan with every leaf replaced by a plan of its own.
def plan_map(~Context: Data,~band: @+context: Context -> @+first: U32 -> @+rows: U32 -> Bands.Plan,+context: Context,layout: Bands.Plan) -> Bands.Plan:
  match layout:
    case Bands.Rows{+first,+rows}: band(context,first,rows)
    case Bands.Split{left,right}: Bands.Split{plan_map(~Context,~band,context,left),plan_map(~Context,~band,context,right)}

# A property that needs nothing from the context.
def always(~Element: Data,~Context: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+context: Context) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Property

# The walks below carry facts from band to band. An operation proves its rule
# for one band and passes what the rule needs as one context value: static
# arguments are closed, so the lemma can be used from any definition.

# A tree built from a plan has that plan.
def build_layout(~Element: Data,~Context: Data,~band: @+context: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,+context: Context,layout: Bands.Plan) ->
  {content_layout(~Element,build(~Element,~band(context),layout)) == layout : Bands.Plan}:
  match layout:
    case Bands.Rows{first,rows}: {==}
    case Bands.Split{+left,+right}:
      %Equal.sym(Bands.Plan,content_layout(~Element,build(~Element,~band(context),left)),left,build_layout(~Element,~Context,~band,context,left)) : {Bands.Split{_,content_layout(~Element,build(~Element,~band(context),right))} == Bands.Split{left,right} : Bands.Plan}
      %Equal.sym(Bands.Plan,content_layout(~Element,build(~Element,~band(context),right)),right,build_layout(~Element,~Context,~band,context,right)) : {Bands.Split{left,_} == Bands.Split{left,right} : Bands.Plan}
      {==}

# A tree made from a plan: a fact about every leaf of the plan gives a fact about every band.
def build_all(~Element: Data,~Context: Data,~band: @+context: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,~Rows: Data,~Before: Rows -> @+first: U32 -> @+rows: U32 -> Data,~rows_context: @+context: Context -> Rows,~After: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> Before(rows_context(context),first,rows) -> After(context,first,rows,band(context,first,rows)),
  +context: Context,layout: Bands.Plan,evidence: PlanAll(~Rows,~Before,rows_context(context),layout)) -> ContentAll(~Element,~After(context),build(~Element,~band(context),layout)):
  match layout:
    case Bands.Rows{+first,+rows}: leaf(context,first,rows,evidence)
    case Bands.Split{+left,+right}:
      Shape.Both{left_evidence,right_evidence} = evidence
      Shape.Both{build_all(~Element,~Context,~band,~Rows,~Before,~rows_context,~After,~leaf,context,left,left_evidence),build_all(~Element,~Context,~band,~Rows,~Before,~rows_context,~After,~leaf,context,right,right_evidence)}

# A tree made from a plan whose every leaf makes bands of its own.
def planned(~Element: Data,~part: @+first: U32 -> @+rows: U32 -> Content<Element>,layout: Bands.Plan) -> Content<Element>:
  match layout:
    case Bands.Rows{+first,+rows}: part(first,rows)
    case Bands.Split{left,right}: Joined{planned(~Element,~part,left),planned(~Element,~part,right)}

# A fact about every leaf of the plan that gives a fact about every band its
# leaf makes gives that fact about every band of the tree.
def planned_all(~Element: Data,~Context: Data,~part: @+context: Context -> @+first: U32 -> @+rows: U32 -> Content<Element>,~Rows: Data,~Before: Rows -> @+first: U32 -> @+rows: U32 -> Data,~rows_context: @+context: Context -> Rows,
  ~After: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> Before(rows_context(context),first,rows) -> ContentAll(~Element,~After(context),part(context,first,rows)),
  +context: Context,layout: Bands.Plan,evidence: PlanAll(~Rows,~Before,rows_context(context),layout)) -> ContentAll(~Element,~After(context),planned(~Element,~part(context),layout)):
  match layout:
    case Bands.Rows{+first,+rows}: leaf(context,first,rows,evidence)
    case Bands.Split{+left,+right}:
      Shape.Both{left_evidence,right_evidence} = evidence
      Shape.Both{planned_all(~Element,~Context,~part,~Rows,~Before,~rows_context,~After,~leaf,context,left,left_evidence),
        planned_all(~Element,~Context,~part,~Rows,~Before,~rows_context,~After,~leaf,context,right,right_evidence)}

# A plan that covers rows, whose every leaf makes bands that cover its own
# rows, makes a tree that covers them.
def planned_coverage(~Element: Data,~Context: Data,~part: @+context: Context -> @+first: U32 -> @+rows: U32 -> Content<Element>,~Rows: Data,~Before: Rows -> @+first: U32 -> @+rows: U32 -> Data,~rows_context: @+context: Context -> Rows,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> Before(rows_context(context),first,rows) ->
    BandPlan.Coverage(content_layout(~Element,part(context,first,rows)),U32.to_nat(first),U32.to_nat(rows)),
  +context: Context,layout: Bands.Plan,+first: Nat,+rows: Nat,evidence: PlanAll(~Rows,~Before,rows_context(context),layout),coverage: BandPlan.Coverage(layout,first,rows)) ->
  BandPlan.Coverage(content_layout(~Element,planned(~Element,~part(context),layout)),first,rows):
  match layout:
    case Bands.Rows{+native_first,+native_rows}:
      BandPlan.LeafCoverage{first_equal,rows_equal} = coverage
      %first_equal : BandPlan.Coverage(content_layout(~Element,part(context,native_first,native_rows)),_,rows)
      %rows_equal : BandPlan.Coverage(content_layout(~Element,part(context,native_first,native_rows)),U32.to_nat(native_first),_)
      leaf(context,native_first,native_rows,evidence)
    case Bands.Split{+left,+right}:
      Shape.Both{left_evidence,right_evidence} = evidence
      BandPlan.ForkCoverage{+left_rows,+right_rows,rows_equal,left_coverage,right_coverage} = coverage
      BandPlan.ForkCoverage{left_rows,right_rows,rows_equal,
        planned_coverage(~Element,~Context,~part,~Rows,~Before,~rows_context,~leaf,context,left,first,left_rows,left_evidence,left_coverage),
        planned_coverage(~Element,~Context,~part,~Rows,~Before,~rows_context,~leaf,context,right,Nat.add(first,left_rows),right_rows,right_evidence,right_coverage)}

# A fact about every band of a content follows from another fact about every
# band, band by band.
def content_one(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~After: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> @+first_fact: First(context,first,rows,buffer) -> After(context,first,rows,buffer),
  +context: Context,content: Content<Element>,first_fact: ContentAll(~Element,~First(context),content)) ->
  ContentAll(~Element,~After(context),content):
  match content:
    case RowBand{+buffer,+first,+rows}: leaf(context,first,rows,buffer,first_fact)
    case Joined{+left,+right}:
      Shape.Both{first_fact_left,first_fact_right} = first_fact
      Shape.Both{content_one(~Element,~Context,~First,~After,~leaf,context,left,first_fact_left),content_one(~Element,~Context,~First,~After,~leaf,context,right,first_fact_right)}

# Two facts about every band are one fact about every band, and back.
def Both(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Second: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+context: Context,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  Shape.Both<First(context,first,rows,buffer),Second(context,first,rows,buffer)>

def content_both(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Second: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+context: Context,content: Content<Element>,
  first_fact: ContentAll(~Element,~First(context),content),second_fact: ContentAll(~Element,~Second(context),content)) ->
  ContentAll(~Element,~Both(~Element,~Context,~First,~Second,context),content):
  match content:
    case RowBand{+buffer,+first,+rows}: Shape.Both{first_fact,second_fact}
    case Joined{+left,+right}:
      Shape.Both{first_left,first_right} = first_fact
      Shape.Both{second_left,second_right} = second_fact
      Shape.Both{content_both(~Element,~Context,~First,~Second,context,left,first_left,second_left),content_both(~Element,~Context,~First,~Second,context,right,first_right,second_right)}

def content_first(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Second: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+context: Context,content: Content<Element>,
  facts: ContentAll(~Element,~Both(~Element,~Context,~First,~Second,context),content)) -> ContentAll(~Element,~First(context),content):
  match content:
    case RowBand{+buffer,+first,+rows}:
      Shape.Both{first_fact,second_fact} = facts
      first_fact
    case Joined{+left,+right}:
      Shape.Both{left_facts,right_facts} = facts
      Shape.Both{content_first(~Element,~Context,~First,~Second,context,left,left_facts),content_first(~Element,~Context,~First,~Second,context,right,right_facts)}

def content_second(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Second: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+context: Context,content: Content<Element>,
  facts: ContentAll(~Element,~Both(~Element,~Context,~First,~Second,context),content)) -> ContentAll(~Element,~Second(context),content):
  match content:
    case RowBand{+buffer,+first,+rows}:
      Shape.Both{first_fact,second_fact} = facts
      second_fact
    case Joined{+left,+right}:
      Shape.Both{left_facts,right_facts} = facts
      Shape.Both{content_second(~Element,~Context,~First,~Second,context,left,left_facts),content_second(~Element,~Context,~First,~Second,context,right,right_facts)}

# A ready model with facts about the contents of every band gives a fact
# about the contents of every band, as they are and after a map, or about
# every leaf of the layout.
def ready_one(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~After: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+ready: Buffer.Sized<Element,buffer> -> @+first_fact: First(context,first,rows,Buffer.content(~Element,buffer)) -> After(context,first,rows,Buffer.content(~Element,buffer)),
  +context: Context,model: Model<Element>,ready: Ready(~Element,model),first_fact: ContentAll(~Element,~First(context),content(~Element,model))) ->
  ContentAll(~Element,~After(context),content(~Element,model)):
  match model:
    case Leaf{+buffer,+first,+rows}: leaf(context,first,rows,buffer,ready,first_fact)
    case Fork{+left,+right}:
      Shape.Both{ready_left,ready_right} = ready
      Shape.Both{first_fact_left,first_fact_right} = first_fact
      Shape.Both{ready_one(~Element,~Context,~First,~After,~leaf,context,left,ready_left,first_fact_left),ready_one(~Element,~Context,~First,~After,~leaf,context,right,ready_right,first_fact_right)}

def ready_two(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Second: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~After: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+ready: Buffer.Sized<Element,buffer> -> @+first_fact: First(context,first,rows,Buffer.content(~Element,buffer)) -> @+second_fact: Second(context,first,rows,Buffer.content(~Element,buffer)) -> After(context,first,rows,Buffer.content(~Element,buffer)),
  +context: Context,model: Model<Element>,ready: Ready(~Element,model),first_fact: ContentAll(~Element,~First(context),content(~Element,model)),second_fact: ContentAll(~Element,~Second(context),content(~Element,model))) ->
  ContentAll(~Element,~After(context),content(~Element,model)):
  match model:
    case Leaf{+buffer,+first,+rows}: leaf(context,first,rows,buffer,ready,first_fact,second_fact)
    case Fork{+left,+right}:
      Shape.Both{ready_left,ready_right} = ready
      Shape.Both{first_fact_left,first_fact_right} = first_fact
      Shape.Both{second_fact_left,second_fact_right} = second_fact
      Shape.Both{ready_two(~Element,~Context,~First,~Second,~After,~leaf,context,left,ready_left,first_fact_left,second_fact_left),ready_two(~Element,~Context,~First,~Second,~After,~leaf,context,right,ready_right,first_fact_right,second_fact_right)}

def model_zero(~Element: Data,~Context: Data,~After: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> After(context,first,rows,Buffer.content(~Element,buffer)),
  +context: Context,model: Model<Element>) ->
  ContentAll(~Element,~After(context),content(~Element,model)):
  match model:
    case Leaf{+buffer,+first,+rows}: leaf(context,first,rows,buffer)
    case Fork{+left,+right}: Shape.Both{model_zero(~Element,~Context,~After,~leaf,context,left),model_zero(~Element,~Context,~After,~leaf,context,right)}

def model_one(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~After: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+first_fact: First(context,first,rows,Buffer.content(~Element,buffer)) -> After(context,first,rows,Buffer.content(~Element,buffer)),
  +context: Context,model: Model<Element>,first_fact: ContentAll(~Element,~First(context),content(~Element,model))) ->
  ContentAll(~Element,~After(context),content(~Element,model)):
  match model:
    case Leaf{+buffer,+first,+rows}: leaf(context,first,rows,buffer,first_fact)
    case Fork{+left,+right}:
      Shape.Both{first_fact_left,first_fact_right} = first_fact
      Shape.Both{model_one(~Element,~Context,~First,~After,~leaf,context,left,first_fact_left),model_one(~Element,~Context,~First,~After,~leaf,context,right,first_fact_right)}

def layout_one(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Rows: Data,~After: Rows -> @+first: U32 -> @+rows: U32 -> Data,~rows_context: @+context: Context -> Rows,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+first_fact: First(context,first,rows,Buffer.content(~Element,buffer)) -> After(rows_context(context),first,rows),
  +context: Context,model: Model<Element>,first_fact: ContentAll(~Element,~First(context),content(~Element,model))) ->
  PlanAll(~Rows,~After,rows_context(context),layout(~Element,model)):
  match model:
    case Leaf{+buffer,+first,+rows}: leaf(context,first,rows,buffer,first_fact)
    case Fork{+left,+right}:
      Shape.Both{first_fact_left,first_fact_right} = first_fact
      Shape.Both{layout_one(~Element,~Context,~First,~Rows,~After,~rows_context,~leaf,context,left,first_fact_left),layout_one(~Element,~Context,~First,~Rows,~After,~rows_context,~leaf,context,right,first_fact_right)}

def ready_layout_one(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Rows: Data,~After: Rows -> @+first: U32 -> @+rows: U32 -> Data,~rows_context: @+context: Context -> Rows,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+ready: Buffer.Sized<Element,buffer> -> @+first_fact: First(context,first,rows,Buffer.content(~Element,buffer)) -> After(rows_context(context),first,rows),
  +context: Context,model: Model<Element>,ready: Ready(~Element,model),first_fact: ContentAll(~Element,~First(context),content(~Element,model))) ->
  PlanAll(~Rows,~After,rows_context(context),layout(~Element,model)):
  match model:
    case Leaf{+buffer,+first,+rows}: leaf(context,first,rows,buffer,ready,first_fact)
    case Fork{+left,+right}:
      Shape.Both{ready_left,ready_right} = ready
      Shape.Both{first_fact_left,first_fact_right} = first_fact
      Shape.Both{ready_layout_one(~Element,~Context,~First,~Rows,~After,~rows_context,~leaf,context,left,ready_left,first_fact_left),ready_layout_one(~Element,~Context,~First,~Rows,~After,~rows_context,~leaf,context,right,ready_right,first_fact_right)}

def plan(~Element: Data,+model: Model<Element>) ->
  {Bands.plan(~Element,pack(~Element,model)) == (pack(~Element,model),layout(~Element,model)) : Bands.Bands(~Element) & Bands.Plan}:
  match model:
    case Leaf{buffer,first,rows}: {==}
    case Fork{+left,+right}:
      %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,pack(~Element,left)),(pack(~Element,left),layout(~Element,left)),plan(~Element,left)) :
        {Bands.reduced(~Storage.Buffer<Element>,~Bands.Plan,~Bands.split_plan,_,Bands.plan(~Element,pack(~Element,right))) == (pack(~Element,Fork{left,right}),layout(~Element,Fork{left,right})) : Bands.Bands(~Element) & Bands.Plan}
      %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,pack(~Element,right)),(pack(~Element,right),layout(~Element,right)),plan(~Element,right)) :
        {Bands.reduced(~Storage.Buffer<Element>,~Bands.Plan,~Bands.split_plan,(pack(~Element,left),layout(~Element,left)),_) == (pack(~Element,Fork{left,right}),layout(~Element,Fork{left,right})) : Bands.Bands(~Element) & Bands.Plan}
      {==}

# A state threaded through the bands in order.
def thread(~Element: Data,~State: Data,~Context: Data,~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> State -> State,
  +context: Context,content: Content<Element>,state: State) -> State:
  match content:
    case RowBand{buffer,+first,+rows}: step(context,first,rows,buffer,state)
    case Joined{left,right}: thread(~Element,~State,~Context,~step,context,right,thread(~Element,~State,~Context,~step,context,left,state))

# What every band contributes, in order.
def collect(~Element: Data,~Item: Data,~Context: Data,~leaf: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> List<&2,Item>,
  +context: Context,content: Content<Element>) -> List<&2,Item>:
  match content:
    case RowBand{buffer,+first,+rows}: leaf(context,first,rows,buffer)
    case Joined{left,right}: List.append(&2,Item,collect(~Element,~Item,~Context,~leaf,context,left),collect(~Element,~Item,~Context,~leaf,context,right))

# A value from every band, joined where the tree forks.
def reduce(~Element: Data,~Value: Data,~Context: Data,~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Model<Element> -> Value,~join: Value -> Value -> Value,
  +context: Context,model: Model<Element>) -> Value:
  match model:
    case Leaf{buffer,+first,+rows}: leaf(context,first,rows,buffer)
    case Fork{left,right}: join(reduce(~Element,~Value,~Context,~leaf,~join,context,left),reduce(~Element,~Value,~Context,~leaf,~join,context,right))

def ReduceRule(~Element: Data,~Value: Data,~Context: Data,~run: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Bands.Bands(~Element) & Value,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Model<Element> -> Value) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> ->
  {run(context,first,rows,Buffer.pack(~Element,buffer)) == (Partition.Leaf{Buffer.pack(~Element,buffer),first,rows,Unit{}},leaf(context,first,rows,buffer)) : Bands.Bands(~Element) & Value}

# The reduction of the code leaves the tree as it is and gives the joined
# value. An operation supplies only the equation of its leaf.
def reduced(~Element: Data,~Value: Data,~Context: Data,~run: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Bands.Bands(~Element) & Value,~join: Value -> Value -> Value,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Model<Element> -> Value,~rule: ReduceRule(~Element,~Value,~Context,~run,~leaf),+context: Context,+model: Model<Element>) ->
  {Bands.reduce(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Value,~Context,~run,~join,context,pack(~Element,model)) == (pack(~Element,model),reduce(~Element,~Value,~Context,~leaf,~join,context,model)) : Bands.Bands(~Element) & Value}:
  match model:
    case Leaf{+buffer,+first,+rows}: rule(context,first,rows,buffer)
    case Fork{+left,+right}:
      %Equal.sym(Bands.Bands(~Element) & Value,Bands.reduce(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Value,~Context,~run,~join,context,pack(~Element,left)),(pack(~Element,left),reduce(~Element,~Value,~Context,~leaf,~join,context,left)),reduced(~Element,~Value,~Context,~run,~join,~leaf,~rule,context,left)) :
        {Bands.reduced(~Storage.Buffer<Element>,~Value,~join,_,Bands.reduce(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Value,~Context,~run,~join,context,pack(~Element,right))) == (pack(~Element,Fork{left,right}),reduce(~Element,~Value,~Context,~leaf,~join,context,Fork{left,right})) : Bands.Bands(~Element) & Value}
      %Equal.sym(Bands.Bands(~Element) & Value,Bands.reduce(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Value,~Context,~run,~join,context,pack(~Element,right)),(pack(~Element,right),reduce(~Element,~Value,~Context,~leaf,~join,context,right)),reduced(~Element,~Value,~Context,~run,~join,~leaf,~rule,context,right)) :
        {Bands.reduced(~Storage.Buffer<Element>,~Value,~join,(pack(~Element,left),reduce(~Element,~Value,~Context,~leaf,~join,context,left)),_) == (pack(~Element,Fork{left,right}),reduce(~Element,~Value,~Context,~leaf,~join,context,Fork{left,right})) : Bands.Bands(~Element) & Value}
      {==}

type Observation<-Element: Data,-bands: Bands.Bands(~Element)> is Data:
  Observation{model: Model<Element>,equation: {pack(~Element,model) == bands : Bands.Bands(~Element)}}

def leaf(~Element: Data,+first: U32,+rows: U32,-buffer: Storage.Buffer<Element>,observation: Buffer.Observation<Element,buffer>) ->
  Observation<Element,Partition.Leaf{buffer,first,rows,Unit{}}>:
  Buffer.Observation{+model,equation} = observation
  Observation{Leaf{model,first,rows},Equal.cong(Storage.Buffer<Element>,Bands.Bands(~Element),
    owner => Partition.Leaf{owner,first,rows,Unit{}},Buffer.pack(~Element,model),buffer,equation)}

def fork(~Element: Data,-left: Bands.Bands(~Element),-right: Bands.Bands(~Element),
  left_observation: Observation<Element,left>,right_observation: Observation<Element,right>) -> Observation<Element,Partition.Fork{left,right}>:
  Observation{+left_model,left_equal} = left_observation
  Observation{+right_model,right_equal} = right_observation
  Observation{Fork{left_model,right_model},
    %left_equal : {Partition.Fork{pack(~Element,left_model),pack(~Element,right_model)} == Partition.Fork{_,right} : Bands.Bands(~Element)}
    %right_equal : {Partition.Fork{pack(~Element,left_model),pack(~Element,right_model)} == Partition.Fork{pack(~Element,left_model),_} : Bands.Bands(~Element)}
    {==}}

def observe(~Element: Data,bands: Bands.Bands(~Element)) -> Observation<Element,bands>:
  match bands:
    case Partition.Leaf{buffer,+first,+rows,Unit{}}: leaf(~Element,first,rows,buffer,Buffer.observe(~Element,buffer))
    case Partition.Fork{left,right}: fork(~Element,left,right,observe(~Element,left),observe(~Element,right))

def lift(~Element: Data,~Context: Data,~Proposition: Context -> @-bands: Bands.Bands(~Element) -> Type,+context: Context,-bands: Bands.Bands(~Element),
  observation: Observation<Element,bands>,canonical: @+model: Model<Element> -> Proposition(context,pack(~Element,model))) -> Proposition(context,bands):
  Observation{model,equation} = observation
  %equation : Proposition(context,_)
  canonical(model)

# Bands equal to a packed model have the model's contents.
def kept(~Element: Data,+model: Model<Element>,-bands: Bands.Bands(~Element),equation: {bands == pack(~Element,model) : Bands.Bands(~Element)}) ->
  {contents(~Element,bands) == content(~Element,model) : Content<Element>}:
  %Equal.sym(Bands.Bands(~Element),bands,pack(~Element,model),equation) : {contents(~Element,_) == content(~Element,model) : Content<Element>}
  packed_contents(~Element,model)

# A tree of the shape of a band tree with any leaf: the owners an operation
# keeps beside each band, or what is seen of them. An operation gives the
# leaf and uses the functions and lemmas below.
type Shaped<-Part: Data> is Data:
  Piece{part: Part,first: U32,rows: U32}
  Pieces{left: Shaped<Part>,right: Shaped<Part>}

def shaped_map(~From: Data,~To: Data,~each: From -> To,tree: Shaped<From>) -> Shaped<To>:
  match tree:
    case Piece{part,first,rows}: Piece{each(part),first,rows}
    case Pieces{left,right}: Pieces{shaped_map(~From,~To,~each,left),shaped_map(~From,~To,~each,right)}

# Every piece mapped with its rows.
def shaped_valued(~Part: Data,~Value: Data,~Context: Data,~value: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Value,+context: Context,tree: Shaped<Part>) -> Shaped<Value>:
  match tree:
    case Piece{+part,+first,+rows}: Piece{value(context,part,first,rows),first,rows}
    case Pieces{left,right}: Pieces{shaped_valued(~Part,~Value,~Context,~value,context,left),shaped_valued(~Part,~Value,~Context,~value,context,right)}

def shaped_pack(~Part: Data,~Block: Type,~pack: Part -> Block,tree: Shaped<Part>) -> Partition.Tree<&1,Block,Unit>:
  match tree:
    case Piece{part,first,rows}: Partition.Leaf{pack(part),first,rows,Unit{}}
    case Pieces{left,right}: Partition.Fork{shaped_pack(~Part,~Block,~pack,left),shaped_pack(~Part,~Block,~pack,right)}

def shaped_seen(~Block: Type,~Part: Data,~see: Block -> Part,tree: Partition.Tree<&1,Block,Unit>) -> Shaped<Part>:
  match tree:
    case Partition.Leaf{block,first,rows,details}: Piece{see(block),first,rows}
    case Partition.Fork{left,right}: Pieces{shaped_seen(~Block,~Part,~see,left),shaped_seen(~Block,~Part,~see,right)}

# The pieces of a model tree, and a model tree carried into blocks is its
# pieces packed.
def shaped_parts(~Part: Data,tree: Partition.Tree<&2,Part,Unit>) -> Shaped<Part>:
  match tree:
    case Partition.Leaf{part,first,rows,details}: Piece{part,first,rows}
    case Partition.Fork{left,right}: Pieces{shaped_parts(~Part,left),shaped_parts(~Part,right)}

def shaped_carried(~Part: Data,~Block: Type,~pack: Part -> Block,+tree: Partition.Tree<&2,Part,Unit>) ->
  {PartitionLaws.carry(~&2,~&1,~Part,~Block,~Unit,~pack,tree) == shaped_pack(~Part,~Block,~pack,shaped_parts(~Part,tree)) : Partition.Tree<&1,Block,Unit>}:
  match tree:
    case Partition.Leaf{+part,+first,+rows,Unit{}}: {==}
    case Partition.Fork{+left,+right}:
      %Equal.sym(Partition.Tree<&1,Block,Unit>,PartitionLaws.carry(~&2,~&1,~Part,~Block,~Unit,~pack,left),shaped_pack(~Part,~Block,~pack,shaped_parts(~Part,left)),
        shaped_carried(~Part,~Block,~pack,left)) :
        {Partition.Fork{_,PartitionLaws.carry(~&2,~&1,~Part,~Block,~Unit,~pack,right)} ==
          Partition.Fork{shaped_pack(~Part,~Block,~pack,shaped_parts(~Part,left)),shaped_pack(~Part,~Block,~pack,shaped_parts(~Part,right))} : Partition.Tree<&1,Block,Unit>}
      %Equal.sym(Partition.Tree<&1,Block,Unit>,PartitionLaws.carry(~&2,~&1,~Part,~Block,~Unit,~pack,right),shaped_pack(~Part,~Block,~pack,shaped_parts(~Part,right)),
        shaped_carried(~Part,~Block,~pack,right)) :
        {Partition.Fork{shaped_pack(~Part,~Block,~pack,shaped_parts(~Part,left)),_} ==
          Partition.Fork{shaped_pack(~Part,~Block,~pack,shaped_parts(~Part,left)),shaped_pack(~Part,~Block,~pack,shaped_parts(~Part,right))} : Partition.Tree<&1,Block,Unit>}
      {==}

# The pieces moved to later rows.
def shaped_shift(~Part: Data,+start: U32,tree: Shaped<Part>) -> Shaped<Part>:
  match tree:
    case Piece{part,+first,rows}: Piece{part,(start + first : U32),rows}
    case Pieces{left,right}: Pieces{shaped_shift(~Part,start,left),shaped_shift(~Part,start,right)}

# A state threaded through the pieces in order.
def shaped_thread(~Part: Data,~State: Data,~Context: Data,~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> State -> State,
  +context: Context,tree: Shaped<Part>,state: State) -> State:
  match tree:
    case Piece{part,+first,+rows}: step(context,first,rows,part,state)
    case Pieces{left,right}: shaped_thread(~Part,~State,~Context,~step,context,right,shaped_thread(~Part,~State,~Context,~step,context,left,state))

# The value of the last piece: what a step that replaces the state by a value
# of its piece leaves.
def shaped_last(~Part: Data,~Value: Data,~value: Part -> Value,tree: Shaped<Part>) -> Value:
  match tree:
    case Piece{part,first,rows}: value(part)
    case Pieces{left,right}: shaped_last(~Part,~Value,~value,right)

def replace(~Part: Data,~Value: Data,~value: Part -> Value,~Context: Data,context: Context,+first: U32,+rows: U32,part: Part,state: Value) -> Value:
  value(part)

def replaced(~Part: Data,~Value: Data,~value: Part -> Value,~Context: Data,+context: Context,tree: Shaped<Part>,+state: Value) ->
  {shaped_thread(~Part,~Value,~Context,~replace(~Part,~Value,~value,~Context),context,tree,state) == shaped_last(~Part,~Value,~value,tree) : Value}:
  match tree:
    case Piece{+part,+first,+rows}: {==}
    case Pieces{+left,+right}:
      replaced(~Part,~Value,~value,~Context,context,right,shaped_thread(~Part,~Value,~Context,~replace(~Part,~Value,~value,~Context),context,left,state))

# A tree with the bands of a plan, one part built for every band.
def shaped_build(~Part: Data,~Context: Data,~piece: @+context: Context -> @+first: U32 -> @+rows: U32 -> Part,+context: Context,layout: Bands.Plan) -> Shaped<Part>:
  match layout:
    case Bands.Rows{+first,+rows}: Piece{piece(context,first,rows),first,rows}
    case Bands.Split{left,right}: Pieces{shaped_build(~Part,~Context,~piece,context,left),shaped_build(~Part,~Context,~piece,context,right)}

# A plan with a plan of its own for every leaf.
def shaped_plan(~Part: Data,~band: Part -> @+first: U32 -> @+rows: U32 -> Bands.Plan,tree: Shaped<Part>) -> Bands.Plan:
  match tree:
    case Piece{part,+first,+rows}: band(part,first,rows)
    case Pieces{left,right}: Bands.Split{shaped_plan(~Part,~band,left),shaped_plan(~Part,~band,right)}

# One band tree read out of every part, and the contents every part gives.
def shaped_model(~Part: Data,~Element: Data,~buffer: Part -> Buffer.Model<Element>,tree: Shaped<Part>) -> Model<Element>:
  match tree:
    case Piece{part,first,rows}: Leaf{buffer(part),first,rows}
    case Pieces{left,right}: Fork{shaped_model(~Part,~Element,~buffer,left),shaped_model(~Part,~Element,~buffer,right)}

def shaped_content(~Part: Data,~Element: Data,~Context: Data,~band: Context -> Part -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,
  +context: Context,tree: Shaped<Part>) -> Content<Element>:
  match tree:
    case Piece{part,+first,+rows}: RowBand{band(context,part,first,rows),first,rows}
    case Pieces{left,right}: Joined{shaped_content(~Part,~Element,~Context,~band,context,left),shaped_content(~Part,~Element,~Context,~band,context,right)}

def ShapedAll(~Part: Data,~Context: Data,~Property: @+context: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,+context: Context,tree: Shaped<Part>) -> Data:
  match tree:
    case Piece{+part,+first,+rows}: Property(context,part,first,rows)
    case Pieces{left,right}: Shape.Both<ShapedAll(~Part,~Context,~Property,context,left),ShapedAll(~Part,~Context,~Property,context,right)>

# The plan of the rows of the pieces: the layout of the band model read out
# of them.
def shaped_rows(~Part: Data,tree: Shaped<Part>) -> Bands.Plan:
  match tree:
    case Piece{part,first,rows}: Bands.Rows{first,rows}
    case Pieces{left,right}: Bands.Split{shaped_rows(~Part,left),shaped_rows(~Part,right)}

def shaped_model_layout(~Part: Data,~Element: Data,~buffer: Part -> Buffer.Model<Element>,tree: Shaped<Part>) ->
  {layout(~Element,shaped_model(~Part,~Element,~buffer,tree)) == shaped_rows(~Part,tree) : Bands.Plan}:
  match tree:
    case Piece{+part,+first,+rows}: {==}
    case Pieces{+left,+right}:
      %Equal.sym(Bands.Plan,layout(~Element,shaped_model(~Part,~Element,~buffer,left)),shaped_rows(~Part,left),shaped_model_layout(~Part,~Element,~buffer,left)) :
        {Bands.Split{_,layout(~Element,shaped_model(~Part,~Element,~buffer,right))} == Bands.Split{shaped_rows(~Part,left),shaped_rows(~Part,right)} : Bands.Plan}
      %Equal.sym(Bands.Plan,layout(~Element,shaped_model(~Part,~Element,~buffer,right)),shaped_rows(~Part,right),shaped_model_layout(~Part,~Element,~buffer,right)) :
        {Bands.Split{shaped_rows(~Part,left),_} == Bands.Split{shaped_rows(~Part,left),shaped_rows(~Part,right)} : Bands.Plan}
      {==}

# Pieces moved to later rows cover the moved rows.
def shaped_shift_coverage(~Part: Data,+start: U32,tree: Shaped<Part>,+first: Nat,+rows: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),Nat.add(first,rows)),U32.to_nat(capacity)) == True{} : Bool},
  coverage: BandPlan.Coverage(shaped_rows(~Part,tree),first,rows)) ->
  BandPlan.Coverage(shaped_rows(~Part,shaped_shift(~Part,start,tree)),Nat.add(U32.to_nat(start),first),rows):
  match tree:
    case Piece{+part,+native_first,+native_rows}:
      BandPlan.LeafCoverage{+first_equal,+rows_equal} = coverage
      BandPlan.LeafCoverage{
        Equal.trans(Nat,U32.to_nat((start + native_first : U32)),Nat.add(U32.to_nat(start),U32.to_nat(native_first)),Nat.add(U32.to_nat(start),first),
          Arithmetic.add(start,native_first,capacity,capacity_ok,
            %Equal.sym(Nat,U32.to_nat(native_first),first,first_equal) : {Nat.is_le(Nat.add(U32.to_nat(start),_),U32.to_nat(capacity)) == True{} : Bool}
            Order.transitive(Nat.add(U32.to_nat(start),first),Nat.add(U32.to_nat(start),Nat.add(first,rows)),U32.to_nat(capacity),
              Order.offset_le(U32.to_nat(start),first,Nat.add(first,rows),Order.le_add(first,rows)),room)),
          Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(start),value),U32.to_nat(native_first),first,first_equal)),
        rows_equal}
    case Pieces{+left,+right}:
      BandPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
      +origin = U32.to_nat(start)
      BandPlan.ForkCoverage{left_rows,right_rows,rows_equal,
        shaped_shift_coverage(~Part,start,left,first,left_rows,capacity,capacity_ok,
          Order.transitive(Nat.add(origin,Nat.add(first,left_rows)),Nat.add(origin,Nat.add(first,rows)),U32.to_nat(capacity),
            Order.offset_le(origin,Nat.add(first,left_rows),Nat.add(first,rows),
              Order.offset_le(first,left_rows,rows,
                %Equal.sym(Nat,rows,Nat.add(left_rows,right_rows),rows_equal) : {Nat.is_le(left_rows,_) == True{} : Bool}
                Order.le_add(left_rows,right_rows))),
            room),
          left_coverage),
        %Equal.sym(Nat,Nat.add(Nat.add(origin,first),left_rows),Nat.add(origin,Nat.add(first,left_rows)),Addition.associative(origin,first,left_rows)) :
          BandPlan.Coverage(shaped_rows(~Part,shaped_shift(~Part,start,right)),_,right_rows)
        shaped_shift_coverage(~Part,start,right,Nat.add(first,left_rows),right_rows,capacity,capacity_ok,
          %Equal.sym(Nat,Nat.add(Nat.add(first,left_rows),right_rows),Nat.add(first,Nat.add(left_rows,right_rows)),Addition.associative(first,left_rows,right_rows)) :
            {Nat.is_le(Nat.add(origin,_),U32.to_nat(capacity)) == True{} : Bool}
          %rows_equal : {Nat.is_le(Nat.add(origin,Nat.add(first,_)),U32.to_nat(capacity)) == True{} : Bool}
          room,
          right_coverage)}

# A fact about every piece gives a fact about every band of the band model
# read out of the pieces moved to later rows.
def shaped_content_all(~Part: Data,~Element: Data,~buffer: Part -> Buffer.Model<Element>,~Context: Data,~start: Context -> U32,
  ~Property: @+context: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  ~After: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+band: Buffer.Content<Element> -> Data,
  ~rule: @+context: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> @fact: Property(context,part,first,rows) ->
    After(context,(start(context) + first : U32),rows,Buffer.content(~Element,buffer(part))),
  +context: Context,tree: Shaped<Part>,facts: ShapedAll(~Part,~Context,~Property,context,tree)) ->
  ContentAll(~Element,~After(context),content(~Element,shaped_model(~Part,~Element,~buffer,shaped_shift(~Part,start(context),tree)))):
  match tree:
    case Piece{+part,+first,+rows}: rule(context,part,first,rows,facts)
    case Pieces{+left,+right}:
      Shape.Both{left_facts,right_facts} = facts
      Shape.Both{shaped_content_all(~Part,~Element,~buffer,~Context,~start,~Property,~After,~rule,context,left,left_facts),
        shaped_content_all(~Part,~Element,~buffer,~Context,~start,~Property,~After,~rule,context,right,right_facts)}

# A property of a part under a context read from a larger one.
def viewed(~Part: Data,~Context: Data,~Property: @+context: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  ~Outer: Data,~view: @+outer: Outer -> Context,+outer: Outer,+part: Part,+first: U32,+rows: U32) -> Data:
  Property(view(outer),part,first,rows)

def shaped_all_view(~Part: Data,~Context: Data,~Property: @+context: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  ~Outer: Data,~view: @+outer: Outer -> Context,+outer: Outer,tree: Shaped<Part>,facts: ShapedAll(~Part,~Context,~Property,view(outer),tree)) ->
  ShapedAll(~Part,~Outer,~viewed(~Part,~Context,~Property,~Outer,~view),outer,tree):
  match tree:
    case Piece{part,first,rows}: facts
    case Pieces{left,right}:
      Shape.Both{left_facts,right_facts} = facts
      Shape.Both{shaped_all_view(~Part,~Context,~Property,~Outer,~view,outer,left,left_facts),shaped_all_view(~Part,~Context,~Property,~Outer,~view,outer,right,right_facts)}

# An owner as the pack of a part.
type Packed<-Block: Type,-Part: Data,-pack: Part -> Block,-block: Block> is Data:
  Packed{part: Part,equation: {pack(part) == block : Block}}

type ShapedObservation<-Part: Data,-Block: Type,-pack: Part -> Block,-tree: Partition.Tree<&1,Block,Unit>> is Data:
  ShapedObservation{owners: Shaped<Part>,equation: {shaped_pack(~Part,~Block,~pack,owners) == tree : Partition.Tree<&1,Block,Unit>}}

def shaped_observe_piece(~Part: Data,~Block: Type,~pack: Part -> Block,+first: U32,+rows: U32,-block: Block,observation: Packed<Block,Part,pack,block>) ->
  ShapedObservation<Part,Block,pack,Partition.Leaf{block,first,rows,Unit{}}>:
  Packed{+part,equation} = observation
  ShapedObservation{Piece{part,first,rows},
    %equation : {Partition.Leaf{pack(part),first,rows,Unit{}} == Partition.Leaf{_,first,rows,Unit{}} : Partition.Tree<&1,Block,Unit>}
    {==}}

def shaped_observe_pieces(~Part: Data,~Block: Type,~pack: Part -> Block,-left: Partition.Tree<&1,Block,Unit>,-right: Partition.Tree<&1,Block,Unit>,
  left_observation: ShapedObservation<Part,Block,pack,left>,right_observation: ShapedObservation<Part,Block,pack,right>) -> ShapedObservation<Part,Block,pack,Partition.Fork{left,right}>:
  ShapedObservation{+left_owners,left_equal} = left_observation
  ShapedObservation{+right_owners,right_equal} = right_observation
  ShapedObservation{Pieces{left_owners,right_owners},
    %left_equal : {Partition.Fork{shaped_pack(~Part,~Block,~pack,left_owners),shaped_pack(~Part,~Block,~pack,right_owners)} == Partition.Fork{_,right} : Partition.Tree<&1,Block,Unit>}
    %right_equal : {Partition.Fork{shaped_pack(~Part,~Block,~pack,left_owners),shaped_pack(~Part,~Block,~pack,right_owners)} == Partition.Fork{shaped_pack(~Part,~Block,~pack,left_owners),_} : Partition.Tree<&1,Block,Unit>}
    {==}}

# Every tree of owners is the pack of a tree of parts.
def shaped_observe(~Part: Data,~Block: Type,~pack: Part -> Block,~observe: @block: Block -> Packed<Block,Part,pack,block>,tree: Partition.Tree<&1,Block,Unit>) -> ShapedObservation<Part,Block,pack,tree>:
  match tree:
    case Partition.Leaf{block,+first,+rows,Unit{}}: shaped_observe_piece(~Part,~Block,~pack,first,rows,block,observe(block))
    case Partition.Fork{left,right}:
      shaped_observe_pieces(~Part,~Block,~pack,left,right,shaped_observe(~Part,~Block,~pack,~observe,left),shaped_observe(~Part,~Block,~pack,~observe,right))

# What is seen of a packed tree is the tree of what is seen of every part.
def shaped_packed(~Part: Data,~Seen: Data,~Block: Type,~pack: Part -> Block,~see: Block -> Seen,~content: Part -> Seen,
  ~rule: @+part: Part -> {see(pack(part)) == content(part) : Seen},+tree: Shaped<Part>) ->
  {shaped_seen(~Block,~Seen,~see,shaped_pack(~Part,~Block,~pack,tree)) == shaped_map(~Part,~Seen,~content,tree) : Shaped<Seen>}:
  match tree:
    case Piece{+part,+first,+rows}: Equal.cong(Seen,Shaped<Seen>,value => Piece{value,first,rows},see(pack(part)),content(part),rule(part))
    case Pieces{+left,+right}:
      %Equal.sym(Shaped<Seen>,shaped_seen(~Block,~Seen,~see,shaped_pack(~Part,~Block,~pack,left)),shaped_map(~Part,~Seen,~content,left),shaped_packed(~Part,~Seen,~Block,~pack,~see,~content,~rule,left)) :
        {Pieces{_,shaped_seen(~Block,~Seen,~see,shaped_pack(~Part,~Block,~pack,right))} == Pieces{shaped_map(~Part,~Seen,~content,left),shaped_map(~Part,~Seen,~content,right)} : Shaped<Seen>}
      %Equal.sym(Shaped<Seen>,shaped_seen(~Block,~Seen,~see,shaped_pack(~Part,~Block,~pack,right)),shaped_map(~Part,~Seen,~content,right),shaped_packed(~Part,~Seen,~Block,~pack,~see,~content,~rule,right)) :
        {Pieces{shaped_map(~Part,~Seen,~content,left),_} == Pieces{shaped_map(~Part,~Seen,~content,left),shaped_map(~Part,~Seen,~content,right)} : Shaped<Seen>}
      {==}

# Bands equal to a packed model have that model's content.
def observed_contents(~Element: Data,+model: Model<Element>,-owner: Bands.Bands(~Element),equation: {pack(~Element,model) == owner : Bands.Bands(~Element)}) ->
  {contents(~Element,owner) == content(~Element,model) : Content<Element>}:
  Equal.trans(Content<Element>,contents(~Element,owner),contents(~Element,pack(~Element,model)),content(~Element,model),
    Equal.cong(Bands.Bands(~Element),Content<Element>,bands => contents(~Element,bands),owner,pack(~Element,model),Equal.sym(Bands.Bands(~Element),pack(~Element,model),owner,equation)),
    packed_contents(~Element,model))
