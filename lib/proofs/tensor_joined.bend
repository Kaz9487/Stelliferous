# Tensors.joined: the pieces of a tensor back in one buffer hold its values
# in order, a piece of all its cells.
import Base
import ../tensor.bend as Tensors
import ../tensor_move.bend as Move
import ../tensor_pieces.bend as Pieces
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_move_readback.bend as Held
import ../default_machine_limits.bend as Default
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./tensor_count_refinement.bend as Count
import ./tensor_global_shape_count.bend as Global
import ./tensor_placed.bend as Placed
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./tensor_contract.bend as Contract
import ./tensor_shape_refinement.bend as ShapeLaws
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ../kernel_gemm.bend as KernelGemm

# The values of a buffer whose tree and length hold them.
def content_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+count: U32,+buffer: Buffer.Content<Element>,
  leaf: Content.Leaf(~Element,~Context,~value,context,first,count,Buffer.content_tree(~Element,buffer),Buffer.content_length(~Element,buffer))) ->
  Content.piece_values(~Element,~Context,~value,context,first,count,buffer):
  match buffer:
    case Buffer.Content{tree,length}: leaf

# A copy of the cells of covering pieces is a piece of all of them.
def copied_holds(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Held.Holding<Element>,+count: U32,+fill: Element,
  +tiled: Held.Tiled(~Element,holding,0n,U32.to_nat(count)),+values: Held.HoldingValues(~Element,~Context,~value,context,holding),
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Placed.PieceHolds(~Element,~Context,~value,context,0,count,
    Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),0,count,fill)))):
  +leaf = Held.copied_leaf(~Element,~Context,~value,context,holding,0,count,fill,0n,U32.to_nat(count),count,Order.word_le(count,1073741824,2147483648,allocatable,{==}),
    Order.reflexive(U32.to_nat(count)),tiled,values,{==},Order.reflexive(U32.to_nat(count)),allocatable)
  KernelGemm.Both{
    content_values(~Element,~Context,~value,context,0,count,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),0,count,fill))),
      %Held.copy_tree(~Element,holding,0,count,fill) :
        Content.Leaf(~Element,~Context,~value,context,0,count,_,Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),0,count,fill)))))
      %Held.copy_length(~Element,holding,0,count,fill) :
        Content.Leaf(~Element,~Context,~value,context,0,count,Buffer.tree(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),0,count,fill)))),_)
      leaf),
    KernelGemm.Both{Held.copied_length(~Element,holding,0,count,fill),Held.copied_ready(~Element,holding,0,count,fill,allocatable)}}

# Covering held pieces copied.
def held_holds(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+count: U32,+fill: Element,-pieces: Pieces.Pieces(~Element),+layout: Held.Spans,
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},
  +covered: Held.Covered<Element,Context,value,context,Move.held(~Element,pieces),0n,U32.to_nat(count),layout>) ->
  Placed.PieceHolds(~Element,~Context,~value,context,0,count,
    Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Move.held(~Element,pieces),0,count,fill)))):
  Held.Covered{+holding,+packed,+tiled,+held_values,+ordered} = covered
  %Equal.sym(Move.Holding<Element>,Move.held(~Element,pieces),Held.pack(~Element,holding),packed) :
    Placed.PieceHolds(~Element,~Context,~value,context,0,count,
      Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,_,0,count,fill))))
  copied_holds(~Element,~Context,~value,context,holding,count,fill,tiled,held_values,allocatable)

# Well formed pieces of count cells with the values of a tensor, joined.
def joined_holds(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+count: U32,+fill: Element,+model: PiecesModel.Model<Element>,
  +formed: Content.WellFormed<Element,U32.to_nat(count),PiecesModel.content(~Element,model)>,
  +values: Content.Values(~Element,~Context,~value,context,PiecesModel.content(~Element,model)),
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Placed.PieceHolds(~Element,~Context,~value,context,0,count,Buffer.contents(~Element,Tensors.joined(~Element,count,fill,PiecesModel.pack(~Element,model)))):
  match model:
    case PiecesModel.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+rows}:
      Content.WellFormed{+fields,+ready,+coverage} = formed
      Content.Fields{+length_exact,+inside} = fields
      PiecesPlan.LeafCoverage{+first_equal,+rows_equal} = coverage
      %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,Buffer.Model{tree,length,certificate})),Buffer.Content{tree,length},
        Buffer.packed_contents(~Element,Buffer.Model{tree,length,certificate})) :
        Placed.PieceHolds(~Element,~Context,~value,context,0,count,_)
      %Arithmetic.to_nat_injective(first,0,first_equal) : Placed.PieceHolds(~Element,~Context,~value,context,_,count,Buffer.Content{tree,length})
      %Arithmetic.to_nat_injective(rows,count,rows_equal) : Placed.PieceHolds(~Element,~Context,~value,context,first,_,Buffer.Content{tree,length})
      KernelGemm.Both{values,KernelGemm.Both{length_exact,ready}}
    case PiecesModel.Fork{+left,+right}:
      held_holds(~Element,~Context,~value,context,count,fill,PiecesModel.pack(~Element,PiecesModel.Fork{left,right}),
        Held.layout_spans(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,PiecesModel.Fork{left,right})),Held.Ended{}),allocatable,
        Held.held_covered(~Element,~Context,~value,context,PiecesModel.Fork{left,right},U32.to_nat(count),formed,values))

# ---- A piece of all cells is whole storage ----

type Whole<-Element: Data,-Logical: Data> is Data:
  Whole{logical: Logical,tree: Tree.Tree<Element>,length: Nat}

def piece_at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+whole: Whole<Element,Logical>,+index: Nat) -> Data:
  Whole{logical,tree,length} = whole
  Content.At(~Element,~Logical,~original,logical,0,tree,length,index)

def whole_at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+whole: Whole<Element,Logical>,+index: Nat) -> Data:
  Whole{logical,tree,length} = whole
  Contract.At(~Element,~Logical,~original,logical,Buffer.Content{tree,length},index)

def piece_whole_at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+whole: Whole<Element,Logical>,+index: Nat,
  +held: piece_at(~Element,~Logical,~original,whole,index)) -> whole_at(~Element,~Logical,~original,whole,index):
  Whole{+logical,+tree,+length} = whole
  KernelGemm.Both{+selected,+equal} = held
  equal

# A piece from cell zero over all count cells of its buffer holds the values
# of whole storage.
def piece_whole(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+count: U32,+content: Buffer.Content<Element>,
  +holds: Placed.PieceHolds(~Element,~Logical,~original,logical,0,count,content)) -> Contract.Values(~Element,~Logical,~original,logical,content):
  match content:
    case Buffer.Content{+tree,+length}:
      KernelGemm.Both{+values,KernelGemm.Both{+sized,+ready}} = holds
      %Equal.sym(Nat,length,U32.to_nat(count),sized) :
        Members.All(Nat,{index => Contract.At(~Element,~Logical,~original,logical,Buffer.Content{tree,length},index) : Nat -> Data},Ranges.range(_,0n))
      Members.transform_all(~Nat,~Whole<Element,Logical>,~piece_at(~Element,~Logical,~original),~whole_at(~Element,~Logical,~original),
        ~piece_whole_at(~Element,~Logical,~original),Ranges.range(U32.to_nat(count),0n),Whole{logical,tree,length},values)

# Such a piece is the well formed whole storage of a shape of count cells.
def piece_formed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+count: U32,+shape: List<&2,U32>,
  +exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat},+content: Buffer.Content<Element>,
  +holds: Placed.PieceHolds(~Element,~Logical,~original,logical,0,count,content)) -> Contract.WellFormed<Element,shape,content>:
  KernelGemm.Both{+values,KernelGemm.Both{+sized,+ready}} = holds
  Contract.WellFormed{Equal.trans(Nat,Buffer.content_length(~Element,content),U32.to_nat(count),ShapeLaws.elements(shape),sized,
    Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),exact)),ready}

def StoredAt(~Element: Data,+content: Buffer.Content<Element>,+index: Nat) -> Data:
  Content.At(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),content,0,Buffer.content_tree(~Element,content),Buffer.content_length(~Element,content),index)

def stored_at(~Element: Data,+content: Buffer.Content<Element>,+index: Nat,+inside: {Nat.is_lt(index,Buffer.content_length(~Element,content)) == True{} : Bool}) ->
  StoredAt(~Element,content,index):
  match content:
    case Buffer.Content{+tree,+length}: KernelGemm.Both{inside,{==}}

# Ready contents of count cells, as a piece from cell zero, hold their own
# cells.
def stored_holds(~Element: Data,+count: U32,+content: Buffer.Content<Element>,+length_exact: {Buffer.content_length(~Element,content) == U32.to_nat(count) : Nat},
  +ready: Buffer.ContentReady<Element,content>) -> Placed.PieceHolds(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),content,0,count,content):
  match content:
    case Buffer.Content{+tree,+length}:
      KernelGemm.Both{
        Members.range_all(~Buffer.Content<Element>,~StoredAt(~Element),~Buffer.content_length(~Element),~stored_at(~Element),U32.to_nat(count),0n,Buffer.Content{tree,length},
          {%Equal.sym(Nat,length,U32.to_nat(count),length_exact) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool} Order.reflexive(U32.to_nat(count)) :
            {Nat.is_le(U32.to_nat(count),length) == True{} : Bool}}),
        KernelGemm.Both{length_exact,ready}}

# ---- The joined cells of a tensor of the contract ----

def count_allocatable(+expected: Nat,+count: U32,+evidence: Count.Count<expected,count>) -> {U32.is_le(count,1073741824) == True{} : Bool}:
  Count.Count{+exact,+bounded} = evidence
  Order.word_le(count,Machine.capacity(Default.limits()),1073741824,bounded,LimitBounds.capacity_allocatable(Default.limits()))

# The cells of a tensor of the contract joined into one buffer are whole
# storage of its shape that holds its values.
def joined_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+fill: Element,
  +workers: U32,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),
  +kept: Buffer.Model<Element>,
  +equation: {Buffer.pack(~Element,kept) == Tensors.joined(~Element,Tensors.count_or_zero(shape),fill,PiecesModel.pack(~Element,model)) : Storage.Buffer<Element>}) ->
  Held.Both<Contract.WellFormed<Element,shape,Buffer.content(~Element,kept)>,Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,kept))>:
  KernelGemm.Both{KernelGemm.Both{+formed,KernelGemm.Both{+counted,+canonical}},+values} = contract
  +count = Global.count(shape,counted)
  +width = Global.whole_width(shape,count,Global.accepted(shape,counted))
  +evidence = Global.evidence(shape,counted)
  +exact = Count.count_exact(ShapeLaws.elements(shape),count,evidence)
  +back_width = Equal.sym(U32,Tensors.count_or_zero(shape),count,width)
  +joined_exact = {%back_width : {ShapeLaws.elements(shape) == U32.to_nat(_) : Nat} exact :
    {ShapeLaws.elements(shape) == U32.to_nat(Tensors.count_or_zero(shape)) : Nat}}
  +model_contents = PiecesModel.observed_contents(~Element,model,PiecesModel.pack(~Element,model),{==})
  +formed_joined = {%joined_exact : Content.WellFormed<Element,_,PiecesModel.content(~Element,model)>
    %model_contents : Content.WellFormed<Element,ShapeLaws.elements(shape),_>
    formed : Content.WellFormed<Element,U32.to_nat(Tensors.count_or_zero(shape)),PiecesModel.content(~Element,model)>}
  +values_model = {%model_contents : Content.Values(~Element,~Logical,~original,logical,_) values :
    Content.Values(~Element,~Logical,~original,logical,PiecesModel.content(~Element,model))}
  +allocatable = {%back_width : {U32.is_le(_,1073741824) == True{} : Bool} count_allocatable(ShapeLaws.elements(shape),count,evidence) :
    {U32.is_le(Tensors.count_or_zero(shape),1073741824) == True{} : Bool}}
  +holds = joined_holds(~Element,~Logical,~original,logical,Tensors.count_or_zero(shape),fill,model,formed_joined,values_model,allocatable)
  +kept_content = Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.joined(~Element,Tensors.count_or_zero(shape),fill,PiecesModel.pack(~Element,model))),Buffer.contents(~Element,Buffer.pack(~Element,kept)),Buffer.content(~Element,kept),
    Equal.cong(Storage.Buffer<Element>,Buffer.Content<Element>,buffer => Buffer.contents(~Element,buffer),Tensors.joined(~Element,Tensors.count_or_zero(shape),fill,PiecesModel.pack(~Element,model)),Buffer.pack(~Element,kept),
      Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,kept),Tensors.joined(~Element,Tensors.count_or_zero(shape),fill,PiecesModel.pack(~Element,model)),equation)),
    Buffer.packed_contents(~Element,kept))
  +formed_kept = {%kept_content : Contract.WellFormed<Element,shape,_>
    piece_formed(~Element,~Logical,~original,logical,Tensors.count_or_zero(shape),shape,joined_exact,Buffer.contents(~Element,Tensors.joined(~Element,Tensors.count_or_zero(shape),fill,PiecesModel.pack(~Element,model))),holds) :
    Contract.WellFormed<Element,shape,Buffer.content(~Element,kept)>}
  +values_kept = {%kept_content : Contract.Values(~Element,~Logical,~original,logical,_)
    piece_whole(~Element,~Logical,~original,logical,Tensors.count_or_zero(shape),Buffer.contents(~Element,Tensors.joined(~Element,Tensors.count_or_zero(shape),fill,PiecesModel.pack(~Element,model))),holds) :
    Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,kept))}
  Held.Both{formed_kept,values_kept}
