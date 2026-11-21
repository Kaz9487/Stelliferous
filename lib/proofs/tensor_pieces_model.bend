# Canonical proof representation of the real piece owners and their metadata.
# This representation is used only by refinement proofs, never by execution.
import Base
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./traversal_partition_refinement.bend as PartitionLaws
import ./tensor_pieces_plan.bend as PiecesPlan
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ../kernel_gemm.bend as KernelGemm

type Model<-Element: Data> is Data:
  Leaf{buffer: Buffer.Model<Element>,first: U32,rows: U32}
  Fork{left: Model<Element>,right: Model<Element>}

type Content<-Element: Data> is Data:
  PieceContent{buffer: Buffer.Content<Element>,first: U32,rows: U32}
  Joined{left: Content<Element>,right: Content<Element>}

def contents(~Element: Data,pieces: Pieces.Pieces(~Element)) -> Content<Element>:
  match pieces:
    case Partition.Leaf{buffer,first,rows,details}: PieceContent{Buffer.contents(~Element,buffer),first,rows}
    case Partition.Fork{left,right}: Joined{contents(~Element,left),contents(~Element,right)}

def content(~Element: Data,model: Model<Element>) -> Content<Element>:
  match model:
    case Leaf{buffer,first,rows}: PieceContent{Buffer.content(~Element,buffer),first,rows}
    case Fork{left,right}: Joined{content(~Element,left),content(~Element,right)}

# The three shapes of a piece operation on contents: every piece by itself,
# every pair of pieces of two trees of one shape, and a tree made from a plan.
# An operation gives the contents of one piece; the tree is walked here.
def map(~Element: Data,~piece: @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,content: Content<Element>) -> Content<Element>:
  match content:
    case PieceContent{buffer,+first,+rows}: PieceContent{piece(first,rows,buffer),first,rows}
    case Joined{left,right}: Joined{map(~Element,~piece,left),map(~Element,~piece,right)}

def build(~Element: Data,~piece: @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,layout: Pieces.Ranges) -> Content<Element>:
  match layout:
    case Pieces.Range{+first,+rows}: PieceContent{piece(first,rows),first,rows}
    case Pieces.Split{left,right}: Joined{build(~Element,~piece,left),build(~Element,~piece,right)}

def ContentAll(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,content: Content<Element>) -> Data:
  match content:
    case PieceContent{buffer,first,rows}: Property(first,rows,buffer)
    case Joined{left,right}: KernelGemm.Both<ContentAll(~Element,~Property,left),ContentAll(~Element,~Property,right)>

# What one property gives of every piece, another gives when each piece's
# first property implies its second.
def content_map(~Element: Data,~From: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~To: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~implied: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> From(first,rows,buffer) -> To(first,rows,buffer),content: Content<Element>,
  facts: ContentAll(~Element,~From,content)) -> ContentAll(~Element,~To,content):
  match content:
    case PieceContent{buffer,first,rows}: implied(first,rows,buffer,facts)
    case Joined{left,right}:
      KernelGemm.Both{left_facts,right_facts} = facts
      KernelGemm.Both{content_map(~Element,~From,~To,~implied,left,left_facts),content_map(~Element,~From,~To,~implied,right,right_facts)}

def ready_piece(~Element: Data,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  Buffer.ContentReady<Element,buffer>

def ContentReady(~Element: Data,content: Content<Element>) -> Data:
  ContentAll(~Element,~ready_piece(~Element),content)

def pack(~Element: Data,model: Model<Element>) -> Pieces.Pieces(~Element):
  match model:
    case Leaf{buffer,first,rows}: Partition.Leaf{Buffer.pack(~Element,buffer),first,rows,Unit{}}
    case Fork{left,right}: Partition.Fork{pack(~Element,left),pack(~Element,right)}

def packed_contents(~Element: Data,+model: Model<Element>) -> {contents(~Element,pack(~Element,model)) == content(~Element,model) : Content<Element>}:
  match model:
    case Leaf{buffer,+first,+rows}:
      Equal.cong(Buffer.Content<Element>,Content<Element>,buffer => PieceContent{buffer,first,rows},
        Buffer.contents(~Element,Buffer.pack(~Element,buffer)),Buffer.content(~Element,buffer),Buffer.packed_contents(~Element,buffer))
    case Fork{+left,+right}:
      %Equal.sym(Content<Element>,contents(~Element,pack(~Element,left)),content(~Element,left),packed_contents(~Element,left)) :
        {Joined{_,contents(~Element,pack(~Element,right))} == Joined{content(~Element,left),content(~Element,right)} : Content<Element>}
      %Equal.sym(Content<Element>,contents(~Element,pack(~Element,right)),content(~Element,right),packed_contents(~Element,right)) :
        {Joined{content(~Element,left),_} == Joined{content(~Element,left),content(~Element,right)} : Content<Element>}
      {==}

def layout(~Element: Data,model: Model<Element>) -> Pieces.Ranges:
  match model:
    case Leaf{buffer,first,rows}: Pieces.Range{first,rows}
    case Fork{left,right}: Pieces.Split{layout(~Element,left),layout(~Element,right)}

def content_layout(~Element: Data,content: Content<Element>) -> Pieces.Ranges:
  match content:
    case PieceContent{buffer,first,rows}: Pieces.Range{first,rows}
    case Joined{left,right}: Pieces.Split{content_layout(~Element,left),content_layout(~Element,right)}

# A map keeps the ranges of its pieces.
def map_layout(~Element: Data,~piece: @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,content: Content<Element>) ->
  {content_layout(~Element,map(~Element,~piece,content)) == content_layout(~Element,content) : Pieces.Ranges}:
  match content:
    case PieceContent{buffer,+first,+rows}: {==}
    case Joined{+left,+right}:
      %map_layout(~Element,~piece,left) :
        {Pieces.Split{content_layout(~Element,map(~Element,~piece,left)),content_layout(~Element,map(~Element,~piece,right))} == Pieces.Split{_,content_layout(~Element,right)} : Pieces.Ranges}
      %map_layout(~Element,~piece,right) :
        {Pieces.Split{content_layout(~Element,map(~Element,~piece,left)),content_layout(~Element,map(~Element,~piece,right))} ==
          Pieces.Split{content_layout(~Element,map(~Element,~piece,left)),_} : Pieces.Ranges}
      {==}

def layout_matches_content(~Element: Data,+model: Model<Element>) -> {layout(~Element,model) == content_layout(~Element,content(~Element,model)) : Pieces.Ranges}:
  match model:
    case Leaf{buffer,first,rows}: {==}
    case Fork{+left,+right}:
      %layout_matches_content(~Element,left) : {Pieces.Split{layout(~Element,left),layout(~Element,right)} == Pieces.Split{_,content_layout(~Element,content(~Element,right))} : Pieces.Ranges}
      %layout_matches_content(~Element,right) : {Pieces.Split{layout(~Element,left),layout(~Element,right)} == Pieces.Split{layout(~Element,left),_} : Pieces.Ranges}
      {==}

def planned(~Element: Data,~part: @+first: U32 -> @+rows: U32 -> Content<Element>,layout: Pieces.Ranges) -> Content<Element>:
  match layout:
    case Pieces.Range{+first,+rows}: part(first,rows)
    case Pieces.Split{left,right}: Joined{planned(~Element,~part,left),planned(~Element,~part,right)}

def content_one(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~After: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> First(context,first,rows,buffer) -> After(context,first,rows,buffer),
  +context: Context,content: Content<Element>,first_fact: ContentAll(~Element,~First(context),content)) ->
  ContentAll(~Element,~After(context),content):
  match content:
    case PieceContent{+buffer,+first,+rows}: leaf(context,first,rows,buffer,first_fact)
    case Joined{+left,+right}:
      KernelGemm.Both{first_fact_left,first_fact_right} = first_fact
      KernelGemm.Both{content_one(~Element,~Context,~First,~After,~leaf,context,left,first_fact_left),content_one(~Element,~Context,~First,~After,~leaf,context,right,first_fact_right)}

# A fact about every piece that names a context gives, piece by piece, a fact
# that does not.
def content_plain(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~After: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> First(context,first,rows,buffer) -> After(first,rows,buffer),
  +context: Context,content: Content<Element>,first_fact: ContentAll(~Element,~First(context),content)) ->
  ContentAll(~Element,~After,content):
  match content:
    case PieceContent{+buffer,+first,+rows}: leaf(context,first,rows,buffer,first_fact)
    case Joined{+left,+right}:
      KernelGemm.Both{first_fact_left,first_fact_right} = first_fact
      KernelGemm.Both{content_plain(~Element,~Context,~First,~After,~leaf,context,left,first_fact_left),content_plain(~Element,~Context,~First,~After,~leaf,context,right,first_fact_right)}

type Observation<-Element: Data,-pieces: Pieces.Pieces(~Element)> is Data:
  Observation{model: Model<Element>,equation: {pack(~Element,model) == pieces : Pieces.Pieces(~Element)}}

def leaf(~Element: Data,+first: U32,+rows: U32,-buffer: Storage.Buffer<Element>,observation: Buffer.Observation<Element,buffer>) ->
  Observation<Element,Partition.Leaf{buffer,first,rows,Unit{}}>:
  Buffer.Observation{+model,equation} = observation
  Observation{Leaf{model,first,rows},Equal.cong(Storage.Buffer<Element>,Pieces.Pieces(~Element),
    owner => Partition.Leaf{owner,first,rows,Unit{}},Buffer.pack(~Element,model),buffer,equation)}

def fork(~Element: Data,-left: Pieces.Pieces(~Element),-right: Pieces.Pieces(~Element),
  left_observation: Observation<Element,left>,right_observation: Observation<Element,right>) -> Observation<Element,Partition.Fork{left,right}>:
  Observation{+left_model,left_equal} = left_observation
  Observation{+right_model,right_equal} = right_observation
  Observation{Fork{left_model,right_model},
    %left_equal : {Partition.Fork{pack(~Element,left_model),pack(~Element,right_model)} == Partition.Fork{_,right} : Pieces.Pieces(~Element)}
    %right_equal : {Partition.Fork{pack(~Element,left_model),pack(~Element,right_model)} == Partition.Fork{pack(~Element,left_model),_} : Pieces.Pieces(~Element)}
    {==}}

def observe(~Element: Data,pieces: Pieces.Pieces(~Element)) -> Observation<Element,pieces>:
  match pieces:
    case Partition.Leaf{buffer,+first,+rows,Unit{}}: leaf(~Element,first,rows,buffer,Buffer.observe(~Element,buffer))
    case Partition.Fork{left,right}: fork(~Element,left,right,observe(~Element,left),observe(~Element,right))

def kept(~Element: Data,+model: Model<Element>,-pieces: Pieces.Pieces(~Element),equation: {pieces == pack(~Element,model) : Pieces.Pieces(~Element)}) ->
  {contents(~Element,pieces) == content(~Element,model) : Content<Element>}:
  %Equal.sym(Pieces.Pieces(~Element),pieces,pack(~Element,model),equation) : {contents(~Element,_) == content(~Element,model) : Content<Element>}
  packed_contents(~Element,model)

# A tree of the shape of a piece tree with any leaf: the owners an operation
# keeps beside each piece, or what is seen of them. An operation gives the
# leaf and uses the functions and lemmas below.
type Shaped<-Part: Data> is Data:
  Piece{part: Part,first: U32,rows: U32}
  Pieces{left: Shaped<Part>,right: Shaped<Part>}

def shaped_valued(~Part: Data,~Value: Data,~Context: Data,~value: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Value,+context: Context,tree: Shaped<Part>) -> Shaped<Value>:
  match tree:
    case Piece{+part,+first,+rows}: Piece{value(context,part,first,rows),first,rows}
    case Pieces{left,right}: Pieces{shaped_valued(~Part,~Value,~Context,~value,context,left),shaped_valued(~Part,~Value,~Context,~value,context,right)}

def shaped_pack(~Part: Data,~Block: Type,~pack: Part -> Block,tree: Shaped<Part>) -> Partition.Tree<&1,Block,Unit>:
  match tree:
    case Piece{part,first,rows}: Partition.Leaf{pack(part),first,rows,Unit{}}
    case Pieces{left,right}: Partition.Fork{shaped_pack(~Part,~Block,~pack,left),shaped_pack(~Part,~Block,~pack,right)}

# The ordinary buffer model as a tree of arbitrary owned parts.
def buffer_parts(~Element: Data,model: Model<Element>) -> Shaped<Buffer.Model<Element>>:
  match model:
    case Leaf{buffer,first,rows}: Piece{buffer,first,rows}
    case Fork{left,right}: Pieces{buffer_parts(~Element,left),buffer_parts(~Element,right)}

def buffer_parts_packed(~Element: Data,+model: Model<Element>) ->
  {shaped_pack(~Buffer.Model<Element>,~Storage.Buffer<Element>,~Buffer.pack(~Element),buffer_parts(~Element,model)) == pack(~Element,model) : Pieces.Pieces(~Element)}:
  match model:
    case Leaf{buffer,first,rows}: {==}
    case Fork{+left,+right}:
      %buffer_parts_packed(~Element,left) :
        {Partition.Fork{shaped_pack(~Buffer.Model<Element>,~Storage.Buffer<Element>,~Buffer.pack(~Element),buffer_parts(~Element,left)),
          shaped_pack(~Buffer.Model<Element>,~Storage.Buffer<Element>,~Buffer.pack(~Element),buffer_parts(~Element,right))} == Partition.Fork{_,pack(~Element,right)} : Pieces.Pieces(~Element)}
      %buffer_parts_packed(~Element,right) :
        {Partition.Fork{shaped_pack(~Buffer.Model<Element>,~Storage.Buffer<Element>,~Buffer.pack(~Element),buffer_parts(~Element,left)),
          shaped_pack(~Buffer.Model<Element>,~Storage.Buffer<Element>,~Buffer.pack(~Element),buffer_parts(~Element,right))} ==
          Partition.Fork{shaped_pack(~Buffer.Model<Element>,~Storage.Buffer<Element>,~Buffer.pack(~Element),buffer_parts(~Element,left)),_} : Pieces.Pieces(~Element)}
      {==}

def shaped_thread(~Part: Data,~State: Data,~Context: Data,~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> State -> State,
  +context: Context,tree: Shaped<Part>,state: State) -> State:
  match tree:
    case Piece{part,+first,+rows}: step(context,first,rows,part,state)
    case Pieces{left,right}: shaped_thread(~Part,~State,~Context,~step,context,right,shaped_thread(~Part,~State,~Context,~step,context,left,state))

def shaped_plan(~Part: Data,~piece: Part -> @+first: U32 -> @+rows: U32 -> Pieces.Ranges,tree: Shaped<Part>) -> Pieces.Ranges:
  match tree:
    case Piece{part,+first,+rows}: piece(part,first,rows)
    case Pieces{left,right}: Pieces.Split{shaped_plan(~Part,~piece,left),shaped_plan(~Part,~piece,right)}

def shaped_content(~Part: Data,~Element: Data,~Context: Data,~piece: Context -> Part -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,
  +context: Context,tree: Shaped<Part>) -> Content<Element>:
  match tree:
    case Piece{part,+first,+rows}: PieceContent{piece(context,part,first,rows),first,rows}
    case Pieces{left,right}: Joined{shaped_content(~Part,~Element,~Context,~piece,context,left),shaped_content(~Part,~Element,~Context,~piece,context,right)}

def ShapedAll(~Part: Data,~Context: Data,~Property: @+context: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,+context: Context,tree: Shaped<Part>) -> Data:
  match tree:
    case Piece{+part,+first,+rows}: Property(context,part,first,rows)
    case Pieces{left,right}: KernelGemm.Both<ShapedAll(~Part,~Context,~Property,context,left),ShapedAll(~Part,~Context,~Property,context,right)>

def viewed(~Part: Data,~Context: Data,~Property: @+context: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  ~Outer: Data,~view: @+outer: Outer -> Context,+outer: Outer,+part: Part,+first: U32,+rows: U32) -> Data:
  Property(view(outer),part,first,rows)

def shaped_all_view(~Part: Data,~Context: Data,~Property: @+context: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  ~Outer: Data,~view: @+outer: Outer -> Context,+outer: Outer,tree: Shaped<Part>,facts: ShapedAll(~Part,~Context,~Property,view(outer),tree)) ->
  ShapedAll(~Part,~Outer,~viewed(~Part,~Context,~Property,~Outer,~view),outer,tree):
  match tree:
    case Piece{part,first,rows}: facts
    case Pieces{left,right}:
      KernelGemm.Both{left_facts,right_facts} = facts
      KernelGemm.Both{shaped_all_view(~Part,~Context,~Property,~Outer,~view,outer,left,left_facts),shaped_all_view(~Part,~Context,~Property,~Outer,~view,outer,right,right_facts)}

def observed_contents(~Element: Data,+model: Model<Element>,-owner: Pieces.Pieces(~Element),equation: {pack(~Element,model) == owner : Pieces.Pieces(~Element)}) ->
  {contents(~Element,owner) == content(~Element,model) : Content<Element>}:
  Equal.trans(Content<Element>,contents(~Element,owner),contents(~Element,pack(~Element,model)),content(~Element,model),
    Equal.cong(Pieces.Pieces(~Element),Content<Element>,pieces => contents(~Element,pieces),owner,pack(~Element,model),Equal.sym(Pieces.Pieces(~Element),pack(~Element,model),owner,equation)),
    packed_contents(~Element,model))
