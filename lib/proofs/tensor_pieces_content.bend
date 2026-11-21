# The values a tensor holds: cell first + index of a piece is cell index of
# its buffer, and a piece buffer holds as many cells as its range counts.
import Base
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_pieces_model.bend as Model
import ./tensor_pieces_plan.bend as PiecesPlan
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ../kernel_gemm.bend as KernelGemm

# ---- Shape ----

# A piece of cells [first, first + count) of a tensor of total cells.
type Fields<-first: U32,-count: U32,-length: Nat,-total: Nat> is Data:
  Fields{length_exact: {length == U32.to_nat(count) : Nat},
    inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),total) == True{} : Bool}}

def piece_fields(~Element: Data,+total: Nat,+first: U32,+count: U32,+buffer: Buffer.Content<Element>) -> Data:
  Fields<first,count,Buffer.content_length(~Element,buffer),total>

def Contents(~Element: Data,+total: Nat,content: Model.Content<Element>) -> Data:
  Model.ContentAll(~Element,~piece_fields(~Element,total),content)

# Pieces of a tensor of total cells: every piece fits, owns its storage, and
# the ranges cover the cells in order.
type WellFormed<-Element: Data,-total: Nat,-content: Model.Content<Element>> is Data:
  WellFormed{fields: Contents(~Element,total,content),ready: Model.ContentReady(~Element,content),
    coverage: PiecesPlan.Coverage(Model.content_layout(~Element,content),0n,total)}

# ---- Values ----

# Cell index of a piece buffer holds the value of cell first + index.
def At(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,context: Context,first: U32,input: Tree.Tree<Element>,length: Nat,+index: Nat) -> Data:
  KernelGemm.Both<{Nat.is_lt(index,length) == True{} : Bool},
    {Tree.read(Element,input,U32.from_nat(index)) == value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}>

def Leaf(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+count: U32,+input: Tree.Tree<Element>,+length: Nat) -> Data:
  Members.All(Nat,{index => At(~Element,~Context,~value,context,first,input,length,index) : Nat -> Data},Ranges.range(U32.to_nat(count),0n))

def piece_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+count: U32,+buffer: Buffer.Content<Element>) -> Data:
  Buffer.Content{tree,length} = buffer
  Leaf(~Element,~Context,~value,context,first,count,tree,length)

def Values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,pieces: Model.Content<Element>) -> Data:
  Model.ContentAll(~Element,~piece_values(~Element,~Context,~value,context),pieces)

def leaf_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+count: U32,+input: Tree.Tree<Element>,+length: Nat,
  +index: Nat,values: Leaf(~Element,~Context,~value,context,first,count,input,length),inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  At(~Element,~Context,~value,context,first,input,length,index):
  Members.all_member(Nat,{index => At(~Element,~Context,~value,context,first,input,length,index) : Nat -> Data},index,Ranges.range(U32.to_nat(count),0n),values,
    Members.range_member(U32.to_nat(count),0n,index,inside))

# ---- Other readers ----

# Change a logical context only when its pointwise values agree. The pieces,
# their cells and their buffers stay identical.
def Agreement(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data) -> Type:
  @+before: Before -> @+after: After -> @+related: Related(before,after) -> @+index: U32 -> {before_value(before,index) == after_value(after,index) : Element}

type Moved<-Element: Data,-Before: Data,-After: Data,-Related: Before -> After -> Data> is Data:
  Moved{before: Before,after: After,related: Related(before,after),first: U32,input: Tree.Tree<Element>,length: Nat}

def moved_before(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~Related: Before -> After -> Data,
  +scope: Moved<Element,Before,After,Related>,+index: Nat) -> Data:
  Moved{before,after,related,first,input,length} = scope
  At(~Element,~Before,~before_value,before,first,input,length,index)

def moved_after(~Element: Data,~Before: Data,~After: Data,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  +scope: Moved<Element,Before,After,Related>,+index: Nat) -> Data:
  Moved{before,after,related,first,input,length} = scope
  At(~Element,~After,~after_value,after,first,input,length,index)

def moved_cell(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  ~agreement: Agreement(~Element,~Before,~After,~before_value,~after_value,~Related),+scope: Moved<Element,Before,After,Related>,+index: Nat,
  +cell: moved_before(~Element,~Before,~After,~before_value,~Related,scope,index)) -> moved_after(~Element,~Before,~After,~after_value,~Related,scope,index):
  Moved{+before,+after,+related,+first,+input,+length} = scope
  KernelGemm.Both{+inside,+equal} = cell
  +position = U32.from_nat(Nat.add(U32.to_nat(first),index))
  KernelGemm.Both{inside,Equal.trans(Element,Tree.read(Element,input,U32.from_nat(index)),before_value(before,position),after_value(after,position),
    equal,agreement(before,after,related,position))}

def values(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  ~agreement: Agreement(~Element,~Before,~After,~before_value,~after_value,~Related),+before: Before,+after: After,+related: Related(before,after),
  pieces: Model.Content<Element>,evidence: Values(~Element,~Before,~before_value,before,pieces)) -> Values(~Element,~After,~after_value,after,pieces):
  match pieces:
    case Model.PieceContent{Buffer.Content{+tree,+length},+first,+count}:
      Members.transform_all(~Nat,~Moved<Element,Before,After,Related>,~moved_before(~Element,~Before,~After,~before_value,~Related),~moved_after(~Element,~Before,~After,~after_value,~Related),
        ~moved_cell(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement),Ranges.range(U32.to_nat(count),0n),Moved{before,after,related,first,tree,length},evidence)
    case Model.Joined{left,right}:
      KernelGemm.Both{left_values,right_values} = evidence
      KernelGemm.Both{values(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,left,left_values),
        values(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,right,right_values)}

# Two contexts whose values agree at every index below a bound.
def Below(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,~bound: Nat) -> Type:
  @+before: Before -> @+after: After -> @+related: Related(before,after) -> @+index: U32 -> @+inside: {Nat.is_lt(U32.to_nat(index),bound) == True{} : Bool} ->
    {before_value(before,index) == after_value(after,index) : Element}

# A piece inside a tensor of total cells, at most a bound that fits a word.
type Bounded<-Element: Data,-Before: Data,-After: Data,-Related: Before -> After -> Data,-bound: Nat> is Data:
  Bounded{before: Before,after: After,related: Related(before,after),first: U32,count: U32,input: Tree.Tree<Element>,length: Nat,total: Nat,
    fields: Fields<first,count,length,total>,room: {Nat.is_le(total,bound) == True{} : Bool},word: U32,fits: {Nat.is_le(bound,U32.to_nat(word)) == True{} : Bool}}

def bounded_before(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~Related: Before -> After -> Data,~bound: Nat,
  +scope: Bounded<Element,Before,After,Related,bound>,+index: Nat) -> Data:
  Bounded{before,after,related,first,count,input,length,total,fields,room,word,fits} = scope
  At(~Element,~Before,~before_value,before,first,input,length,index)

def bounded_after(~Element: Data,~Before: Data,~After: Data,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,~bound: Nat,
  +scope: Bounded<Element,Before,After,Related,bound>,+index: Nat) -> Data:
  Bounded{before,after,related,first,count,input,length,total,fields,room,word,fits} = scope
  At(~Element,~After,~after_value,after,first,input,length,index)

def bounded_cell(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,~bound: Nat,
  ~agreement: Below(~Element,~Before,~After,~before_value,~after_value,~Related,~bound),+scope: Bounded<Element,Before,After,Related,bound>,+index: Nat,
  +cell: bounded_before(~Element,~Before,~After,~before_value,~Related,~bound,scope,index)) -> bounded_after(~Element,~Before,~After,~after_value,~Related,~bound,scope,index):
  Bounded{+before,+after,+related,+first,+count,+input,+length,+total,+fields,+room,+word,+fits} = scope
  Fields{+length_exact,+inside_total} = fields
  KernelGemm.Both{+inside,+equal} = cell
  +position = Nat.add(U32.to_nat(first),index)
  +below_total = Bounds.nat_trans(position,Nat.add(U32.to_nat(first),U32.to_nat(count)),total,
    Order.offset_lt(U32.to_nat(first),index,U32.to_nat(count),%length_exact : {Nat.is_lt(index,_) == True{} : Bool} inside),inside_total)
  +below_bound = Bounds.nat_trans(position,total,bound,below_total,room)
  +exact = FullRange.roundtrip(position,word,Order.transitive(position,bound,U32.to_nat(word),Bounds.strict_le(position,bound,below_bound),fits))
  +word_index = U32.from_nat(position)
  KernelGemm.Both{inside,Equal.trans(Element,Tree.read(Element,input,U32.from_nat(index)),before_value(before,word_index),after_value(after,word_index),
    equal,agreement(before,after,related,word_index,%Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,exact) : {Nat.is_lt(_,bound) == True{} : Bool} below_bound))}

# The values of the pieces of a tensor of total cells under another context
# that agrees below a bound of at least the total.
def values_below(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,~bound: Nat,
  ~agreement: Below(~Element,~Before,~After,~before_value,~after_value,~Related,~bound),+before: Before,+after: After,+related: Related(before,after),
  +total: Nat,pieces: Model.Content<Element>,+fields: Contents(~Element,total,pieces),
  +room: {Nat.is_le(total,bound) == True{} : Bool},+word: U32,+fits: {Nat.is_le(bound,U32.to_nat(word)) == True{} : Bool},
  evidence: Values(~Element,~Before,~before_value,before,pieces)) -> Values(~Element,~After,~after_value,after,pieces):
  match pieces:
    case Model.PieceContent{Buffer.Content{+tree,+length},+first,+count}:
      Members.transform_all(~Nat,~Bounded<Element,Before,After,Related,bound>,~bounded_before(~Element,~Before,~After,~before_value,~Related,~bound),
        ~bounded_after(~Element,~Before,~After,~after_value,~Related,~bound),~bounded_cell(~Element,~Before,~After,~before_value,~after_value,~Related,~bound,~agreement),
        Ranges.range(U32.to_nat(count),0n),Bounded{before,after,related,first,count,tree,length,total,fields,room,word,fits},evidence)
    case Model.Joined{left,right}:
      KernelGemm.Both{left_fields,right_fields} = fields
      KernelGemm.Both{left_values,right_values} = evidence
      KernelGemm.Both{values_below(~Element,~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,before,after,related,total,left,left_fields,room,word,fits,left_values),
        values_below(~Element,~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,before,after,related,total,right,right_fields,room,word,fits,right_values)}
