# Tensors.placed puts every output piece in the slot of a layout that starts
# at its first cell.
import Base
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./tensor_pieces_model.bend as PiecesModel
import ./traversal_partition_refinement.bend as PartitionLaws
import ./tensor_task_runs.bend as Runs
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition
import ./tensor_move_readback.bend as Held
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./word_encoding.bend as WordEncoding
import ../kernel_gemm.bend as KernelGemm
import ./tensor_global_shape_count.bend as Global
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_shape_refinement.bend as ShapeLaws

# ---- Layouts in order ----

# Every leaf of a layout starts before bound.
def Before(layout: Pieces.Ranges,+bound: Nat) -> Data:
  match layout:
    case Pieces.Range{+first,count}:
      {Nat.is_lt(U32.to_nat(first),bound) == True{} : Bool}
    case Pieces.Split{+left,+right}:
      Held.Both<Before(left,bound),Before(right,bound)>

# The leaves of a layout start in order.
def Ordered(layout: Pieces.Ranges) -> Data:
  match layout:
    case Pieces.Range{first,count}:
      Unit
    case Pieces.Split{+left,+right}:
      Held.Both<Ordered(left),Held.Both<Ordered(right),Before(left,U32.to_nat(Pieces.first(right)))>>

type Side<-Left: Data,-Right: Data> is Data:
  OnLeft{left: Left}
  OnRight{right: Right}

# The leaf of a layout that starts at cell.
def LeafAt(layout: Pieces.Ranges,+cell: U32) -> Data:
  match layout:
    case Pieces.Range{+first,count}:
      {first == cell : U32}
    case Pieces.Split{+left,+right}:
      Side<LeafAt(left,cell),LeafAt(right,cell)>

# The first leaf of a layout starts at its first cell.
def first_leaf(layout: Pieces.Ranges) -> LeafAt(layout,Pieces.first(layout)):
  match layout:
    case Pieces.Range{+first,count}:
      {==}
    case Pieces.Split{+left,+right}:
      OnLeft{first_leaf(left)}

# A leaf of a layout whose leaves start before bound starts before it.
def leaf_before(layout: Pieces.Ranges,+cell: U32,+bound: Nat,+before: Before(layout,bound),+at: LeafAt(layout,cell)) ->
  {Nat.is_lt(U32.to_nat(cell),bound) == True{} : Bool}:
  match layout:
    case Pieces.Range{+first,count}:
      %at : {Nat.is_lt(U32.to_nat(_),bound) == True{} : Bool}
      before
    case Pieces.Split{+left,+right}:
      Held.Both{+left_before,+right_before} = before
      match at:
        case OnLeft{+inside}:
          leaf_before(left,cell,bound,left_before,inside)
        case OnRight{+inside}:
          leaf_before(right,cell,bound,right_before,inside)

# A leaf of an ordered layout starts at or after its first cell.
def leaf_after(layout: Pieces.Ranges,+cell: U32,+ordered: Ordered(layout),+at: LeafAt(layout,cell)) ->
  {Nat.is_le(U32.to_nat(Pieces.first(layout)),U32.to_nat(cell)) == True{} : Bool}:
  match layout:
    case Pieces.Range{+first,count}:
      %at : {Nat.is_le(U32.to_nat(first),U32.to_nat(_)) == True{} : Bool}
      Order.reflexive(U32.to_nat(first))
    case Pieces.Split{+left,+right}:
      Held.Both{+left_ordered,+rest} = ordered
      Held.Both{+right_ordered,+before} = rest
      match at:
        case OnLeft{+inside}:
          leaf_after(left,cell,left_ordered,inside)
        case OnRight{+inside}:
          Order.transitive(U32.to_nat(Pieces.first(left)),U32.to_nat(Pieces.first(right)),U32.to_nat(cell),
            Bounds.strict_le(U32.to_nat(Pieces.first(left)),U32.to_nat(Pieces.first(right)),
              leaf_before(left,Pieces.first(left),U32.to_nat(Pieces.first(right)),before,first_leaf(left))),
            leaf_after(right,cell,right_ordered,inside))

# ---- Exchange ----

# The pieces with replacement in the place Tensors.exchanged finds, and the
# buffer it takes out.
def put(~Element: Data,model: PiecesModel.Model<Element>,layout: Pieces.Ranges,left_side: Bool,+cell: U32,rep: Buffer.Model<Element>) -> PiecesModel.Model<Element>:
  match model:
    case PiecesModel.Leaf{buffer,first,rows}:
      PiecesModel.Leaf{rep,first,rows}
    case PiecesModel.Fork{left,right}:
      match layout:
        case Pieces.Range{first,count}:
          PiecesModel.Fork{left,right}
        case Pieces.Split{+left_layout,+right_layout}:
          match left_side:
            case True{}:
              PiecesModel.Fork{put(~Element,left,left_layout,Tensors.side_of(left_layout,cell),cell,rep),right}
            case False{}:
              PiecesModel.Fork{left,put(~Element,right,right_layout,Tensors.side_of(right_layout,cell),cell,rep)}

def taken(~Element: Data,model: PiecesModel.Model<Element>,layout: Pieces.Ranges,left_side: Bool,+cell: U32,rep: Buffer.Model<Element>) -> Buffer.Model<Element>:
  match model:
    case PiecesModel.Leaf{buffer,first,rows}:
      buffer
    case PiecesModel.Fork{left,right}:
      match layout:
        case Pieces.Range{first,count}:
          rep
        case Pieces.Split{+left_layout,+right_layout}:
          match left_side:
            case True{}:
              taken(~Element,left,left_layout,Tensors.side_of(left_layout,cell),cell,rep)
            case False{}:
              taken(~Element,right,right_layout,Tensors.side_of(right_layout,cell),cell,rep)

def put_at(~Element: Data,model: PiecesModel.Model<Element>,+layout: Pieces.Ranges,+cell: U32,rep: Buffer.Model<Element>) -> PiecesModel.Model<Element>:
  put(~Element,model,layout,Tensors.side_of(layout,cell),cell,rep)

# ---- Deposit ----

# The slot of span cells from slot after a part of count cells from first:
# the part when it is the whole slot, else the model of the slot the runtime
# copies the part into.
def slot_model(~Element: Data,whole: Bool,+data: Buffer.Model<Element>,+slot: U32,+span: U32,+first: U32,+count: U32,+fill: Element,+rep: Buffer.Model<Element>) ->
  Buffer.Model<Element>:
  match whole:
    case True{}: rep
    case False{}: Buffer.remodel(~Element,Tensors.slot_written(~Element,False{},Buffer.pack(~Element,data),slot,span,first,count,fill,Buffer.pack(~Element,rep)))

def slot_exact(~Element: Data,+whole: Bool,+data: Buffer.Model<Element>,+slot: U32,+span: U32,+first: U32,+count: U32,+fill: Element,+rep: Buffer.Model<Element>) ->
  {Tensors.slot_written(~Element,whole,Buffer.pack(~Element,data),slot,span,first,count,fill,Buffer.pack(~Element,rep)) ==
    Buffer.pack(~Element,slot_model(~Element,whole,data,slot,span,first,count,fill,rep)) : Storage.Buffer<Element>}:
  match whole:
    case True{}: {==}
    case False{}:
      Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,slot_model(~Element,False{},data,slot,span,first,count,fill,rep)),
        Tensors.slot_written(~Element,False{},Buffer.pack(~Element,data),slot,span,first,count,fill,Buffer.pack(~Element,rep)),
        Buffer.remodeled(~Element,Tensors.slot_written(~Element,False{},Buffer.pack(~Element,data),slot,span,first,count,fill,Buffer.pack(~Element,rep))))

# The pieces with the part of count cells from first written in the place
# Tensors.deposited finds.
def deposit(~Element: Data,model: PiecesModel.Model<Element>,layout: Pieces.Ranges,left_side: Bool,+first: U32,+count: U32,+fill: Element,+rep: Buffer.Model<Element>) ->
  PiecesModel.Model<Element>:
  match model:
    case PiecesModel.Leaf{+data,+slot,+span}:
      PiecesModel.Leaf{slot_model(~Element,U32.is_eq(slot,first) && U32.is_eq(span,count),data,slot,span,first,count,fill,rep),slot,span}
    case PiecesModel.Fork{left,right}:
      match layout:
        case Pieces.Range{start,cells}:
          PiecesModel.Fork{left,right}
        case Pieces.Split{+left_layout,+right_layout}:
          match left_side:
            case True{}:
              PiecesModel.Fork{deposit(~Element,left,left_layout,Tensors.side_of(left_layout,first),first,count,fill,rep),right}
            case False{}:
              PiecesModel.Fork{left,deposit(~Element,right,right_layout,Tensors.side_of(right_layout,first),first,count,fill,rep)}

def deposit_at(~Element: Data,model: PiecesModel.Model<Element>,+layout: Pieces.Ranges,+first: U32,+count: U32,+fill: Element,+rep: Buffer.Model<Element>) -> PiecesModel.Model<Element>:
  deposit(~Element,model,layout,Tensors.side_of(layout,first),first,count,fill,rep)

def deposited_exact(~Element: Data,+model: PiecesModel.Model<Element>,+layout: Pieces.Ranges,+left_side: Bool,+first: U32,+count: U32,+fill: Element,+rep: Buffer.Model<Element>) ->
  {Tensors.deposited(~Element,PiecesModel.pack(~Element,model),layout,left_side,first,count,fill,Buffer.pack(~Element,rep)) ==
    PiecesModel.pack(~Element,deposit(~Element,model,layout,left_side,first,count,fill,rep)) : Pieces.Pieces(~Element)}:
  match model:
    case PiecesModel.Leaf{+data,+slot,+span}:
      +whole = U32.is_eq(slot,first) && U32.is_eq(span,count)
      Equal.cong(Storage.Buffer<Element>,Pieces.Pieces(~Element),buffer => Partition.Leaf{buffer,slot,span,Unit{}},
        Tensors.slot_written(~Element,whole,Buffer.pack(~Element,data),slot,span,first,count,fill,Buffer.pack(~Element,rep)),
        Buffer.pack(~Element,slot_model(~Element,whole,data,slot,span,first,count,fill,rep)),slot_exact(~Element,whole,data,slot,span,first,count,fill,rep))
    case PiecesModel.Fork{+left,+right}:
      match layout:
        case Pieces.Range{+start,+cells}:
          match left_side:
            case True{}: {==}
            case False{}: {==}
        case Pieces.Split{+left_layout,+right_layout}:
          match left_side:
            case True{}:
              Equal.cong(Pieces.Pieces(~Element),Pieces.Pieces(~Element),tree => Partition.Fork{tree,PiecesModel.pack(~Element,right)},
                Tensors.deposited(~Element,PiecesModel.pack(~Element,left),left_layout,Tensors.side_of(left_layout,first),first,count,fill,Buffer.pack(~Element,rep)),
                PiecesModel.pack(~Element,deposit(~Element,left,left_layout,Tensors.side_of(left_layout,first),first,count,fill,rep)),
                deposited_exact(~Element,left,left_layout,Tensors.side_of(left_layout,first),first,count,fill,rep))
            case False{}:
              Equal.cong(Pieces.Pieces(~Element),Pieces.Pieces(~Element),tree => Partition.Fork{PiecesModel.pack(~Element,left),tree},
                Tensors.deposited(~Element,PiecesModel.pack(~Element,right),right_layout,Tensors.side_of(right_layout,first),first,count,fill,Buffer.pack(~Element,rep)),
                PiecesModel.pack(~Element,deposit(~Element,right,right_layout,Tensors.side_of(right_layout,first),first,count,fill,rep)),
                deposited_exact(~Element,right,right_layout,Tensors.side_of(right_layout,first),first,count,fill,rep))

# A deposit keeps the layout.
def deposit_layout(~Element: Data,+model: PiecesModel.Model<Element>,+layout: Pieces.Ranges,+left_side: Bool,+first: U32,+count: U32,+fill: Element,+rep: Buffer.Model<Element>) ->
  {PiecesModel.layout(~Element,deposit(~Element,model,layout,left_side,first,count,fill,rep)) == PiecesModel.layout(~Element,model) : Pieces.Ranges}:
  match model:
    case PiecesModel.Leaf{+data,+slot,+span}:
      {==}
    case PiecesModel.Fork{+left,+right}:
      match layout:
        case Pieces.Range{+start,+cells}:
          {==}
        case Pieces.Split{+left_layout,+right_layout}:
          match left_side:
            case True{}:
              %Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,deposit(~Element,left,left_layout,Tensors.side_of(left_layout,first),first,count,fill,rep)),PiecesModel.layout(~Element,left),
                  deposit_layout(~Element,left,left_layout,Tensors.side_of(left_layout,first),first,count,fill,rep)) :
                {Pieces.Split{_,PiecesModel.layout(~Element,right)} == Pieces.Split{PiecesModel.layout(~Element,left),PiecesModel.layout(~Element,right)} : Pieces.Ranges}
              {==}
            case False{}:
              %Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,deposit(~Element,right,right_layout,Tensors.side_of(right_layout,first),first,count,fill,rep)),PiecesModel.layout(~Element,right),
                  deposit_layout(~Element,right,right_layout,Tensors.side_of(right_layout,first),first,count,fill,rep)) :
                {Pieces.Split{PiecesModel.layout(~Element,left),_} == Pieces.Split{PiecesModel.layout(~Element,left),PiecesModel.layout(~Element,right)} : Pieces.Ranges}
              {==}

# A part that starts a slot and has its cells is the whole slot.
def whole_slot(+first: U32,+cell: U32,+rows: U32,+count: U32,+at: {first == cell : U32},+cells: {rows == count : U32}) ->
  {U32.is_eq(first,cell) && U32.is_eq(rows,count) == True{} : Bool}:
  %Equal.sym(U32,first,cell,at) : {U32.is_eq(_,cell) && U32.is_eq(rows,count) == True{} : Bool}
  %Equal.sym(U32,rows,count,cells) : {U32.is_eq(cell,cell) && U32.is_eq(_,count) == True{} : Bool}
  %Equal.sym(Bool,U32.is_eq(cell,cell),True{},Arithmetic.equal_reflexive(cell)) : {_ && U32.is_eq(count,count) == True{} : Bool}
  Arithmetic.equal_reflexive(count)

def exchanged_put(~Element: Data,+model: PiecesModel.Model<Element>,+layout: Pieces.Ranges,+left_side: Bool,+cell: U32,+rep: Buffer.Model<Element>) ->
  {Tensors.exchanged(~Element,PiecesModel.pack(~Element,model),layout,left_side,cell,Buffer.pack(~Element,rep)) ==
    (PiecesModel.pack(~Element,put(~Element,model,layout,left_side,cell,rep)),Buffer.pack(~Element,taken(~Element,model,layout,left_side,cell,rep))) : Pieces.Pieces(~Element) & Storage.Buffer<Element>}:
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      {==}
    case PiecesModel.Fork{+left,+right}:
      match layout:
        case Pieces.Range{+first,+count}:
          match left_side:
            case True{}:
              {==}
            case False{}:
              {==}
        case Pieces.Split{+left_layout,+right_layout}:
          match left_side:
            case True{}:
              %Equal.sym(Pieces.Pieces(~Element) & Storage.Buffer<Element>,
                  Tensors.exchanged(~Element,PiecesModel.pack(~Element,left),left_layout,Tensors.side_of(left_layout,cell),cell,Buffer.pack(~Element,rep)),
                  (PiecesModel.pack(~Element,put(~Element,left,left_layout,Tensors.side_of(left_layout,cell),cell,rep)),
                    Buffer.pack(~Element,taken(~Element,left,left_layout,Tensors.side_of(left_layout,cell),cell,rep))),
                  exchanged_put(~Element,left,left_layout,Tensors.side_of(left_layout,cell),cell,rep)) :
                {Tensors.exchanged_left(~Element,PiecesModel.pack(~Element,right),_) ==
                  (PiecesModel.pack(~Element,put(~Element,PiecesModel.Fork{left,right},Pieces.Split{left_layout,right_layout},True{},cell,rep)),
                    Buffer.pack(~Element,taken(~Element,PiecesModel.Fork{left,right},Pieces.Split{left_layout,right_layout},True{},cell,rep))) :
                  Pieces.Pieces(~Element) & Storage.Buffer<Element>}
              {==}
            case False{}:
              %Equal.sym(Pieces.Pieces(~Element) & Storage.Buffer<Element>,
                  Tensors.exchanged(~Element,PiecesModel.pack(~Element,right),right_layout,Tensors.side_of(right_layout,cell),cell,Buffer.pack(~Element,rep)),
                  (PiecesModel.pack(~Element,put(~Element,right,right_layout,Tensors.side_of(right_layout,cell),cell,rep)),
                    Buffer.pack(~Element,taken(~Element,right,right_layout,Tensors.side_of(right_layout,cell),cell,rep))),
                  exchanged_put(~Element,right,right_layout,Tensors.side_of(right_layout,cell),cell,rep)) :
                {Tensors.exchanged_right(~Element,PiecesModel.pack(~Element,left),_) ==
                  (PiecesModel.pack(~Element,put(~Element,PiecesModel.Fork{left,right},Pieces.Split{left_layout,right_layout},False{},cell,rep)),
                    Buffer.pack(~Element,taken(~Element,PiecesModel.Fork{left,right},Pieces.Split{left_layout,right_layout},False{},cell,rep))) :
                  Pieces.Pieces(~Element) & Storage.Buffer<Element>}
              {==}

# Putting a buffer keeps the layout.
def put_layout(~Element: Data,+model: PiecesModel.Model<Element>,+layout: Pieces.Ranges,+left_side: Bool,+cell: U32,+rep: Buffer.Model<Element>) ->
  {PiecesModel.layout(~Element,put(~Element,model,layout,left_side,cell,rep)) == PiecesModel.layout(~Element,model) : Pieces.Ranges}:
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      {==}
    case PiecesModel.Fork{+left,+right}:
      match layout:
        case Pieces.Range{+first,+count}:
          {==}
        case Pieces.Split{+left_layout,+right_layout}:
          match left_side:
            case True{}:
              %Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,put(~Element,left,left_layout,Tensors.side_of(left_layout,cell),cell,rep)),PiecesModel.layout(~Element,left),
                  put_layout(~Element,left,left_layout,Tensors.side_of(left_layout,cell),cell,rep)) :
                {Pieces.Split{_,PiecesModel.layout(~Element,right)} == Pieces.Split{PiecesModel.layout(~Element,left),PiecesModel.layout(~Element,right)} : Pieces.Ranges}
              {==}
            case False{}:
              %Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,put(~Element,right,right_layout,Tensors.side_of(right_layout,cell),cell,rep)),PiecesModel.layout(~Element,right),
                  put_layout(~Element,right,right_layout,Tensors.side_of(right_layout,cell),cell,rep)) :
                {Pieces.Split{PiecesModel.layout(~Element,left),_} == Pieces.Split{PiecesModel.layout(~Element,left),PiecesModel.layout(~Element,right)} : Pieces.Ranges}
              {==}

# ---- Slots ----

# The empty slots of a layout.
def empty_slots(~Element: Data,layout: Pieces.Ranges,+fill: Element) -> PiecesModel.Model<Element>:
  match layout:
    case Pieces.Range{first,count}:
      PiecesModel.Leaf{Buffer.remodel(~Element,Tensors.buffer(~Element,fill,0)),first,count}
    case Pieces.Split{left,right}:
      PiecesModel.Fork{empty_slots(~Element,left,fill),empty_slots(~Element,right,fill)}

def slots_layout(~Element: Data,layout: Pieces.Ranges,+fill: Element) -> {PiecesModel.layout(~Element,empty_slots(~Element,layout,fill)) == layout : Pieces.Ranges}:
  match layout:
    case Pieces.Range{+first,+count}:
      {==}
    case Pieces.Split{+left,+right}:
      %Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,empty_slots(~Element,left,fill)),left,slots_layout(~Element,left,fill)) :
        {Pieces.Split{_,PiecesModel.layout(~Element,empty_slots(~Element,right,fill))} == Pieces.Split{left,right} : Pieces.Ranges}
      %Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,empty_slots(~Element,right,fill)),right,slots_layout(~Element,right,fill)) :
        {Pieces.Split{left,_} == Pieces.Split{left,right} : Pieces.Ranges}
      {==}

# The slots Tensors.placed starts from are the empty slots of its layout.
def slots_built(~Element: Data,layout: Pieces.Ranges,+fill: Element) ->
  {Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~Tensors.slot_piece(~Element),layout,fill,Unit{}) ==
    (Unit{},PiecesModel.pack(~Element,empty_slots(~Element,layout,fill))) : Unit & Pieces.Pieces(~Element)}:
  match layout:
    case Pieces.Range{+first,+count}:
      %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,0))),Tensors.buffer(~Element,fill,0),
          Buffer.remodeled(~Element,Tensors.buffer(~Element,fill,0))) :
        {Tensors.slot_piece(~Element,fill,first,count,Unit{}) == (Unit{},Partition.Leaf{_,first,count,Unit{}}) : Unit & Pieces.Pieces(~Element)}
      {==}
    case Pieces.Split{+left,+right}:
      %Equal.sym(Unit & Pieces.Pieces(~Element),Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~Tensors.slot_piece(~Element),left,fill,Unit{}),
          (Unit{},PiecesModel.pack(~Element,empty_slots(~Element,left,fill))),slots_built(~Element,left,fill)) :
        {Storage.bind_pair(~Unit,~Pieces.Pieces(~Element),~(Unit & Pieces.Pieces(~Element)),_,state => left_tree =>
            Storage.bind_pair(~Unit,~Pieces.Pieces(~Element),~(Unit & Pieces.Pieces(~Element)),
              Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~Tensors.slot_piece(~Element),right,fill,state),state => right_tree => (state,Partition.Fork{left_tree,right_tree}))) ==
          (Unit{},PiecesModel.pack(~Element,empty_slots(~Element,Pieces.Split{left,right},fill))) : Unit & Pieces.Pieces(~Element)}
      %Equal.sym(Unit & Pieces.Pieces(~Element),Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~Tensors.slot_piece(~Element),right,fill,Unit{}),
          (Unit{},PiecesModel.pack(~Element,empty_slots(~Element,right,fill))),slots_built(~Element,right,fill)) :
        {Storage.bind_pair(~Unit,~Pieces.Pieces(~Element),~(Unit & Pieces.Pieces(~Element)),_,state => right_tree =>
            (state,Partition.Fork{PiecesModel.pack(~Element,empty_slots(~Element,left,fill)),right_tree})) ==
          (Unit{},PiecesModel.pack(~Element,empty_slots(~Element,Pieces.Split{left,right},fill))) : Unit & Pieces.Pieces(~Element)}
      {==}

# ---- Placing in order ----

# The cells of the leaf at cell, found as Tensors.exchange finds it.
def rows_of(layout: Pieces.Ranges,+cell: U32) -> U32:
  match layout:
    case Pieces.Range{first,+count}:
      count
    case Pieces.Split{+left,+right}:
      Bool.pick(U32,U32.is_lt(cell,Pieces.first(right)),rows_of(left,cell),rows_of(right,cell))

# At a leaf of the left part the cell lies before the right part, and at a
# leaf of the right part it does not.
def left_side(+left: Pieces.Ranges,+right: Pieces.Ranges,+cell: U32,+before: Before(left,U32.to_nat(Pieces.first(right))),+at: LeafAt(left,cell)) ->
  {U32.is_lt(cell,Pieces.first(right)) == True{} : Bool}:
  Equal.trans(Bool,U32.is_lt(cell,Pieces.first(right)),Nat.is_lt(U32.to_nat(cell),U32.to_nat(Pieces.first(right))),True{},Comparison.lt(cell,Pieces.first(right)),
    leaf_before(left,cell,U32.to_nat(Pieces.first(right)),before,at))

def right_side(+right: Pieces.Ranges,+cell: U32,+ordered: Ordered(right),+at: LeafAt(right,cell)) ->
  {U32.is_lt(cell,Pieces.first(right)) == False{} : Bool}:
  Equal.trans(Bool,U32.is_lt(cell,Pieces.first(right)),Nat.is_lt(U32.to_nat(cell),U32.to_nat(Pieces.first(right))),False{},Comparison.lt(cell,Pieces.first(right)),
    Order.not_below(U32.to_nat(Pieces.first(right)),U32.to_nat(cell),leaf_after(right,cell,ordered,at)))

# Leaves that start before a bound start before any wider bound.
def before_raised(layout: Pieces.Ranges,+bound: Nat,+wider: Nat,+before: Before(layout,bound),+raised: {Nat.is_le(bound,wider) == True{} : Bool}) -> Before(layout,wider):
  match layout:
    case Pieces.Range{+first,+count}:
      Bounds.nat_trans(U32.to_nat(first),bound,wider,before,raised)
    case Pieces.Split{+left,+right}:
      Held.Both{+left_before,+right_before} = before
      Held.Both{before_raised(left,bound,wider,left_before,raised),before_raised(right,bound,wider,right_before,raised)}

# Cell is the first cell of one of the spans.
def Listed(spans: Held.Spans,+cell: U32) -> Data:
  match spans:
    case Held.Ended{}:
      Empty
    case Held.Spanned{+first,+count,rest}:
      Side<{first == cell : U32},Listed(rest,cell)>

# Every leaf of a part starts one of the spans.
def LeavesListed(part: Pieces.Ranges,+spans: Held.Spans) -> Data:
  match part:
    case Pieces.Range{+first,count}:
      Listed(spans,first)
    case Pieces.Split{+left,+right}:
      Held.Both<LeavesListed(left,spans),LeavesListed(right,spans)>

def listed_later(part: Pieces.Ranges,+rest: Held.Spans,+cell: U32,+listed: Listed(rest,cell)) -> Listed(Held.layout_spans(part,rest),cell):
  match part:
    case Pieces.Range{+first,+count}:
      OnRight{listed}
    case Pieces.Split{+left,+right}:
      listed_later(left,Held.layout_spans(right,rest),cell,listed_later(right,rest,cell,listed))

def leaves_later(part: Pieces.Ranges,+before: Pieces.Ranges,+spans: Held.Spans,+listed: LeavesListed(part,spans)) -> LeavesListed(part,Held.layout_spans(before,spans)):
  match part:
    case Pieces.Range{+first,+count}:
      listed_later(before,spans,first,listed)
    case Pieces.Split{+left,+right}:
      Held.Both{+left_listed,+right_listed} = listed
      Held.Both{leaves_later(left,before,spans,left_listed),leaves_later(right,before,spans,right_listed)}

# The leaves of a part start its own spans.
def own_listed(part: Pieces.Ranges,+rest: Held.Spans) -> LeavesListed(part,Held.layout_spans(part,rest)):
  match part:
    case Pieces.Range{+first,+count}:
      OnLeft{{==}}
    case Pieces.Split{+left,+right}:
      Held.Both{own_listed(left,Held.layout_spans(right,rest)),leaves_later(right,left,Held.layout_spans(right,rest),own_listed(right,rest))}

# Every span of from starts one of the spans of to.
def StartsListed(from: Held.Spans,+to: Held.Spans) -> Data:
  match from:
    case Held.Ended{}:
      Unit
    case Held.Spanned{+first,+count,rest}:
      Held.Both<Listed(to,first),StartsListed(rest,to)>

def listed_moved(from: Held.Spans,+to: Held.Spans,+starts: StartsListed(from,to),+cell: U32,+listed: Listed(from,cell)) -> Listed(to,cell):
  match from:
    case Held.Ended{}:
      Empty.absurd(Listed(to,cell),listed)
    case Held.Spanned{+first,+count,+rest}:
      Held.Both{+here,+others} = starts
      match listed:
        case OnLeft{+same}:
          %same : Listed(to,_)
          here
        case OnRight{+later}:
          listed_moved(rest,to,others,cell,later)

# Leaves listed in spans that each start a span of to are listed in to.
def leaves_moved(part: Pieces.Ranges,+from: Held.Spans,+to: Held.Spans,+starts: StartsListed(from,to),+listed: LeavesListed(part,from)) -> LeavesListed(part,to):
  match part:
    case Pieces.Range{+first,+count}:
      listed_moved(from,to,starts,first,listed)
    case Pieces.Split{+left,+right}:
      Held.Both{+left_listed,+right_listed} = listed
      Held.Both{leaves_moved(left,from,to,starts,left_listed),leaves_moved(right,from,to,starts,right_listed)}

# Every leaf holds a piece with the property, or one of the spans still to
# be placed starts at its first cell.
def Good(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,model: PiecesModel.Model<Element>,+remaining: Held.Spans) -> Data:
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      Side<Listed(remaining,first),Property(first,rows,Buffer.content(~Element,buffer))>
    case PiecesModel.Fork{+left,+right}:
      Held.Both<Good(~Element,~Property,left,remaining),Good(~Element,~Property,right,remaining)>

# Leaves that start before cell, or after it, are not the leaf of a span at
# cell.
def good_drop_before(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+model: PiecesModel.Model<Element>,+cell: U32,+count: U32,
  +rest: Held.Spans,+before: Before(PiecesModel.layout(~Element,model),U32.to_nat(cell)),+good: Good(~Element,~Property,model,Held.Spanned{cell,count,rest})) ->
  Good(~Element,~Property,model,rest):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      match good:
        case OnLeft{+listed}:
          match listed:
            case OnLeft{+same}:
              Empty.absurd(Good(~Element,~Property,PiecesModel.Leaf{buffer,first,rows},rest),
                Held.lt_le_absurd(U32.to_nat(first),U32.to_nat(first),
                  {%same : {Nat.is_lt(U32.to_nat(first),U32.to_nat(_)) == True{} : Bool} before : {Nat.is_lt(U32.to_nat(first),U32.to_nat(first)) == True{} : Bool}},
                  Order.reflexive(U32.to_nat(first))))
            case OnRight{+later}:
              OnLeft{later}
        case OnRight{+holds}:
          OnRight{holds}
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_before,+right_before} = before
      Held.Both{+left_good,+right_good} = good
      Held.Both{good_drop_before(~Element,~Property,left,cell,count,rest,left_before,left_good),good_drop_before(~Element,~Property,right,cell,count,rest,right_before,right_good)}

def good_drop_after(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+model: PiecesModel.Model<Element>,+cell: U32,+count: U32,
  +rest: Held.Spans,+ordered: Ordered(PiecesModel.layout(~Element,model)),
  +low: {Nat.is_lt(U32.to_nat(cell),U32.to_nat(Pieces.first(PiecesModel.layout(~Element,model)))) == True{} : Bool},
  +good: Good(~Element,~Property,model,Held.Spanned{cell,count,rest})) -> Good(~Element,~Property,model,rest):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      match good:
        case OnLeft{+listed}:
          match listed:
            case OnLeft{+same}:
              Empty.absurd(Good(~Element,~Property,PiecesModel.Leaf{buffer,first,rows},rest),
                Held.lt_le_absurd(U32.to_nat(first),U32.to_nat(first),
                  {%same : {Nat.is_lt(U32.to_nat(_),U32.to_nat(first)) == True{} : Bool} low : {Nat.is_lt(U32.to_nat(first),U32.to_nat(first)) == True{} : Bool}},
                  Order.reflexive(U32.to_nat(first))))
            case OnRight{+later}:
              OnLeft{later}
        case OnRight{+holds}:
          OnRight{holds}
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_ordered,+ordered_rest} = ordered
      Held.Both{+right_ordered,+before} = ordered_rest
      Held.Both{+left_good,+right_good} = good
      +left_layout = PiecesModel.layout(~Element,left)
      +right_layout = PiecesModel.layout(~Element,right)
      Held.Both{good_drop_after(~Element,~Property,left,cell,count,rest,left_ordered,low,left_good),
        good_drop_after(~Element,~Property,right,cell,count,rest,right_ordered,
          Bounds.nat_trans(U32.to_nat(cell),U32.to_nat(Pieces.first(left_layout)),U32.to_nat(Pieces.first(right_layout)),low,
            Bounds.strict_le(U32.to_nat(Pieces.first(left_layout)),U32.to_nat(Pieces.first(right_layout)),
              leaf_before(left_layout,Pieces.first(left_layout),U32.to_nat(Pieces.first(right_layout)),before,first_leaf(left_layout)))),right_good)}

# Placing the piece of the next span puts it on its leaf with its property
# and leaves every other leaf as it was.
def placed_step(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+model: PiecesModel.Model<Element>,+cell: U32,+count: U32,
  +rep: Buffer.Model<Element>,+rest: Held.Spans,+ordered: Ordered(PiecesModel.layout(~Element,model)),
  +at: LeafAt(PiecesModel.layout(~Element,model),cell),+rows_eq: {rows_of(PiecesModel.layout(~Element,model),cell) == count : U32},
  +holds: Property(cell,count,Buffer.content(~Element,rep)),+good: Good(~Element,~Property,model,Held.Spanned{cell,count,rest})) ->
  Good(~Element,~Property,put_at(~Element,model,PiecesModel.layout(~Element,model),cell,rep),rest):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      OnRight{{%Equal.sym(U32,first,cell,at) : Property(_,rows,Buffer.content(~Element,rep))
        %Equal.sym(U32,rows,count,rows_eq) : Property(cell,_,Buffer.content(~Element,rep))
        holds : Property(first,rows,Buffer.content(~Element,rep))}}
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_ordered,+ordered_rest} = ordered
      Held.Both{+right_ordered,+before} = ordered_rest
      match at:
        case OnLeft{+inside}:
          Held.Both{+left_good,+right_good} = good
          +left_layout = PiecesModel.layout(~Element,left)
          +side = left_side(left_layout,PiecesModel.layout(~Element,right),cell,before,inside)
          +left_rows = {%side :
              {Bool.pick(U32,_,rows_of(left_layout,cell),rows_of(PiecesModel.layout(~Element,right),cell)) == count : U32}
            rows_eq : {rows_of(left_layout,cell) == count : U32}}
          %Equal.sym(Bool,U32.is_lt(cell,Pieces.first(PiecesModel.layout(~Element,right))),True{},side) :
            Good(~Element,~Property,put(~Element,PiecesModel.Fork{left,right},PiecesModel.layout(~Element,PiecesModel.Fork{left,right}),_,cell,rep),rest)
          Held.Both{placed_step(~Element,~Property,left,cell,count,rep,rest,left_ordered,inside,left_rows,holds,left_good),
            good_drop_after(~Element,~Property,right,cell,count,rest,right_ordered,Bounds.as_nat_lt(cell,Pieces.first(PiecesModel.layout(~Element,right)),side),right_good)}
        case OnRight{+inside}:
          Held.Both{+left_good,+right_good} = good
          +right_layout = PiecesModel.layout(~Element,right)
          +not_side = right_side(right_layout,cell,right_ordered,inside)
          %Equal.sym(Bool,U32.is_lt(cell,Pieces.first(right_layout)),False{},not_side) :
            Good(~Element,~Property,put(~Element,PiecesModel.Fork{left,right},PiecesModel.layout(~Element,PiecesModel.Fork{left,right}),_,cell,rep),rest)
          Held.Both{good_drop_before(~Element,~Property,left,cell,count,rest,
              before_raised(PiecesModel.layout(~Element,left),U32.to_nat(Pieces.first(right_layout)),U32.to_nat(cell),before,leaf_after(right_layout,cell,right_ordered,inside)),left_good),
            placed_step(~Element,~Property,right,cell,count,rep,rest,right_ordered,inside,
              {%not_side :
                  {Bool.pick(U32,_,rows_of(PiecesModel.layout(~Element,left),cell),rows_of(right_layout,cell)) == count : U32}
                rows_eq : {rows_of(right_layout,cell) == count : U32}},holds,right_good)}

# Depositing the piece of the next span puts it on its leaf with its property
# and leaves every other leaf as it was.
def deposit_step(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+model: PiecesModel.Model<Element>,+cell: U32,+count: U32,
  +fill: Element,+rep: Buffer.Model<Element>,+rest: Held.Spans,+ordered: Ordered(PiecesModel.layout(~Element,model)),
  +at: LeafAt(PiecesModel.layout(~Element,model),cell),+rows_eq: {rows_of(PiecesModel.layout(~Element,model),cell) == count : U32},
  +holds: Property(cell,count,Buffer.content(~Element,rep)),+good: Good(~Element,~Property,model,Held.Spanned{cell,count,rest})) ->
  Good(~Element,~Property,deposit_at(~Element,model,PiecesModel.layout(~Element,model),cell,count,fill,rep),rest):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      %Equal.sym(Bool,U32.is_eq(first,cell) && U32.is_eq(rows,count),True{},whole_slot(first,cell,rows,count,at,rows_eq)) :
        Good(~Element,~Property,PiecesModel.Leaf{slot_model(~Element,_,buffer,first,rows,cell,count,fill,rep),first,rows},rest)
      OnRight{{%Equal.sym(U32,first,cell,at) : Property(_,rows,Buffer.content(~Element,rep))
        %Equal.sym(U32,rows,count,rows_eq) : Property(cell,_,Buffer.content(~Element,rep))
        holds : Property(first,rows,Buffer.content(~Element,rep))}}
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_ordered,+ordered_rest} = ordered
      Held.Both{+right_ordered,+before} = ordered_rest
      match at:
        case OnLeft{+inside}:
          Held.Both{+left_good,+right_good} = good
          +left_layout = PiecesModel.layout(~Element,left)
          +side = left_side(left_layout,PiecesModel.layout(~Element,right),cell,before,inside)
          +left_rows = {%side :
              {Bool.pick(U32,_,rows_of(left_layout,cell),rows_of(PiecesModel.layout(~Element,right),cell)) == count : U32}
            rows_eq : {rows_of(left_layout,cell) == count : U32}}
          %Equal.sym(Bool,U32.is_lt(cell,Pieces.first(PiecesModel.layout(~Element,right))),True{},side) :
            Good(~Element,~Property,deposit(~Element,PiecesModel.Fork{left,right},PiecesModel.layout(~Element,PiecesModel.Fork{left,right}),_,cell,count,fill,rep),rest)
          Held.Both{deposit_step(~Element,~Property,left,cell,count,fill,rep,rest,left_ordered,inside,left_rows,holds,left_good),
            good_drop_after(~Element,~Property,right,cell,count,rest,right_ordered,Bounds.as_nat_lt(cell,Pieces.first(PiecesModel.layout(~Element,right)),side),right_good)}
        case OnRight{+inside}:
          Held.Both{+left_good,+right_good} = good
          +right_layout = PiecesModel.layout(~Element,right)
          +not_side = right_side(right_layout,cell,right_ordered,inside)
          %Equal.sym(Bool,U32.is_lt(cell,Pieces.first(right_layout)),False{},not_side) :
            Good(~Element,~Property,deposit(~Element,PiecesModel.Fork{left,right},PiecesModel.layout(~Element,PiecesModel.Fork{left,right}),_,cell,count,fill,rep),rest)
          Held.Both{good_drop_before(~Element,~Property,left,cell,count,rest,
              before_raised(PiecesModel.layout(~Element,left),U32.to_nat(Pieces.first(right_layout)),U32.to_nat(cell),before,leaf_after(right_layout,cell,right_ordered,inside)),left_good),
            deposit_step(~Element,~Property,right,cell,count,fill,rep,rest,right_ordered,inside,
              {%not_side :
                  {Bool.pick(U32,_,rows_of(PiecesModel.layout(~Element,left),cell),rows_of(right_layout,cell)) == count : U32}
                rows_eq : {rows_of(right_layout,cell) == count : U32}},holds,right_good)}

# Before anything is placed, every leaf that starts a span is good.
def listed_good(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+model: PiecesModel.Model<Element>,+spans: Held.Spans,
  +listed: LeavesListed(PiecesModel.layout(~Element,model),spans)) -> Good(~Element,~Property,model,spans):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      OnLeft{listed}
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_listed,+right_listed} = listed
      Held.Both{listed_good(~Element,~Property,left,spans,left_listed),listed_good(~Element,~Property,right,spans,right_listed)}

# With no span left, every leaf holds a piece with the property.
def all_good(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+model: PiecesModel.Model<Element>,
  +good: Good(~Element,~Property,model,Held.Ended{})) -> PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,model)):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      match good:
        case OnLeft{+listed}:
          Empty.absurd(Property(first,rows,Buffer.content(~Element,buffer)),listed)
        case OnRight{+holds}:
          holds
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_good,+right_good} = good
      KernelGemm.Both{all_good(~Element,~Property,left,left_good),all_good(~Element,~Property,right,right_good)}

# ---- Taking pieces out ----

# A later span leaves every leaf as good as before.
def good_later(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+model: PiecesModel.Model<Element>,+cell: U32,+count: U32,
  +gone: Held.Spans,+good: Good(~Element,~Property,model,gone)) -> Good(~Element,~Property,model,Held.Spanned{cell,count,gone}):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      match good:
        case OnLeft{+listed}:
          OnLeft{OnRight{listed}}
        case OnRight{+holds}:
          OnRight{holds}
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_good,+right_good} = good
      Held.Both{good_later(~Element,~Property,left,cell,count,gone,left_good),good_later(~Element,~Property,right,cell,count,gone,right_good)}

# Cell starts none of the spans.
def Fresh(spans: Held.Spans,+cell: U32) -> Data:
  match spans:
    case Held.Ended{}:
      Unit
    case Held.Spanned{+first,+count,rest}:
      Held.Both<{U32.is_eq(first,cell) == False{} : Bool},Fresh(rest,cell)>

def fresh_unlisted(spans: Held.Spans,+cell: U32,+fresh: Fresh(spans,cell),+listed: Listed(spans,cell)) -> Empty:
  match spans:
    case Held.Ended{}:
      listed
    case Held.Spanned{+first,+count,+rest}:
      Held.Both{+differs,+later} = fresh
      match listed:
        case OnLeft{+same}:
          WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_eq(first,cell),True{},Equal.sym(Bool,U32.is_eq(first,cell),False{},differs),
            {%same : {U32.is_eq(first,_) == True{} : Bool} Arithmetic.equal_reflexive(first) : {U32.is_eq(first,cell) == True{} : Bool}}))
        case OnRight{+more}:
          fresh_unlisted(rest,cell,later,more)

# The buffer taken from the leaf at cell holds the property when no span
# taken before starts at cell.
def taken_holds(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+model: PiecesModel.Model<Element>,+cell: U32,
  +rep: Buffer.Model<Element>,+gone: Held.Spans,+ordered: Ordered(PiecesModel.layout(~Element,model)),+at: LeafAt(PiecesModel.layout(~Element,model),cell),
  +fresh: Fresh(gone,cell),+good: Good(~Element,~Property,model,gone)) ->
  Property(cell,rows_of(PiecesModel.layout(~Element,model),cell),
    Buffer.content(~Element,taken(~Element,model,PiecesModel.layout(~Element,model),Tensors.side_of(PiecesModel.layout(~Element,model),cell),cell,rep))):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      match good:
        case OnLeft{+listed}:
          Empty.absurd(Property(cell,rows,Buffer.content(~Element,buffer)),
            fresh_unlisted(gone,cell,fresh,{%at : Listed(gone,_) listed : Listed(gone,cell)}))
        case OnRight{+holds}:
          %at : Property(_,rows,Buffer.content(~Element,buffer))
          holds
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_ordered,+ordered_rest} = ordered
      Held.Both{+right_ordered,+before} = ordered_rest
      match at:
        case OnLeft{+inside}:
          Held.Both{+left_good,+right_good} = good
          +left_layout = PiecesModel.layout(~Element,left)
          +right_layout = PiecesModel.layout(~Element,right)
          %Equal.sym(Bool,U32.is_lt(cell,Pieces.first(right_layout)),True{},left_side(left_layout,right_layout,cell,before,inside)) :
            Property(cell,Bool.pick(U32,_,rows_of(left_layout,cell),rows_of(right_layout,cell)),
              Buffer.content(~Element,taken(~Element,PiecesModel.Fork{left,right},Pieces.Split{left_layout,right_layout},_,cell,rep)))
          taken_holds(~Element,~Property,left,cell,rep,gone,left_ordered,inside,fresh,left_good)
        case OnRight{+inside}:
          Held.Both{+left_good,+right_good} = good
          +left_layout = PiecesModel.layout(~Element,left)
          +right_layout = PiecesModel.layout(~Element,right)
          %Equal.sym(Bool,U32.is_lt(cell,Pieces.first(right_layout)),False{},right_side(right_layout,cell,right_ordered,inside)) :
            Property(cell,Bool.pick(U32,_,rows_of(left_layout,cell),rows_of(right_layout,cell)),
              Buffer.content(~Element,taken(~Element,PiecesModel.Fork{left,right},Pieces.Split{left_layout,right_layout},_,cell,rep)))
          taken_holds(~Element,~Property,right,cell,rep,gone,right_ordered,inside,fresh,right_good)

# Taking the leaf at cell leaves every other leaf as it was and lists cell.
def put_gone(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+model: PiecesModel.Model<Element>,+cell: U32,+count: U32,
  +rep: Buffer.Model<Element>,+gone: Held.Spans,+ordered: Ordered(PiecesModel.layout(~Element,model)),+at: LeafAt(PiecesModel.layout(~Element,model),cell),
  +good: Good(~Element,~Property,model,gone)) ->
  Good(~Element,~Property,put_at(~Element,model,PiecesModel.layout(~Element,model),cell,rep),Held.Spanned{cell,count,gone}):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      OnLeft{OnLeft{Equal.sym(U32,first,cell,at)}}
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_ordered,+ordered_rest} = ordered
      Held.Both{+right_ordered,+before} = ordered_rest
      match at:
        case OnLeft{+inside}:
          Held.Both{+left_good,+right_good} = good
          +left_layout = PiecesModel.layout(~Element,left)
          +right_layout = PiecesModel.layout(~Element,right)
          %Equal.sym(Bool,U32.is_lt(cell,Pieces.first(right_layout)),True{},left_side(left_layout,right_layout,cell,before,inside)) :
            Good(~Element,~Property,put(~Element,PiecesModel.Fork{left,right},Pieces.Split{left_layout,right_layout},_,cell,rep),Held.Spanned{cell,count,gone})
          Held.Both{put_gone(~Element,~Property,left,cell,count,rep,gone,left_ordered,inside,left_good),good_later(~Element,~Property,right,cell,count,gone,right_good)}
        case OnRight{+inside}:
          Held.Both{+left_good,+right_good} = good
          +left_layout = PiecesModel.layout(~Element,left)
          +right_layout = PiecesModel.layout(~Element,right)
          %Equal.sym(Bool,U32.is_lt(cell,Pieces.first(right_layout)),False{},right_side(right_layout,cell,right_ordered,inside)) :
            Good(~Element,~Property,put(~Element,PiecesModel.Fork{left,right},Pieces.Split{left_layout,right_layout},_,cell,rep),Held.Spanned{cell,count,gone})
          Held.Both{good_later(~Element,~Property,left,cell,count,gone,left_good),put_gone(~Element,~Property,right,cell,count,rep,gone,right_ordered,inside,right_good)}

# ---- Placing the pieces of the tasks ----

# The slots after the pieces of one task, and after the pieces of every task,
# are put as Tensors.placed puts them.
def placed_when_model(~Element: Data,empty: Bool,+layout: Pieces.Ranges,+fill: Element,+first: U32,+count: U32,+rep: Buffer.Model<Element>,slots: PiecesModel.Model<Element>) ->
  PiecesModel.Model<Element>:
  match empty:
    case True{}:
      slots
    case False{}:
      deposit_at(~Element,slots,layout,first,count,fill,rep)

def placed_pieces(~Element: Data,+layout: Pieces.Ranges,+fill: Element,pieces: PiecesModel.Model<Element>,slots: PiecesModel.Model<Element>) -> PiecesModel.Model<Element>:
  match pieces:
    case PiecesModel.Leaf{buffer,+first,+rows}:
      placed_when_model(~Element,U32.is_zero(rows),layout,fill,first,rows,buffer,slots)
    case PiecesModel.Fork{left,right}:
      placed_pieces(~Element,layout,fill,right,placed_pieces(~Element,layout,fill,left,slots))

def placed_tasks(~Element: Data,+layout: Pieces.Ranges,+fill: Element,results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,slots: PiecesModel.Model<Element>) -> PiecesModel.Model<Element>:
  match results:
    case Partition.Leaf{pieces,first,count,details}:
      placed_pieces(~Element,layout,fill,pieces,slots)
    case Partition.Fork{left,right}:
      placed_tasks(~Element,layout,fill,right,placed_tasks(~Element,layout,fill,left,slots))

def placed_when_exact(~Element: Data,+empty: Bool,+layout: Pieces.Ranges,+fill: Element,+first: U32,+count: U32,+rep: Buffer.Model<Element>,+slots: PiecesModel.Model<Element>) ->
  {Tensors.placed_when(~Element,empty,Tensors.Depositing{layout,fill},first,count,Buffer.pack(~Element,rep),PiecesModel.pack(~Element,slots)) ==
    PiecesModel.pack(~Element,placed_when_model(~Element,empty,layout,fill,first,count,rep,slots)) : Pieces.Pieces(~Element)}:
  match empty:
    case True{}:
      {==}
    case False{}:
      deposited_exact(~Element,slots,layout,Tensors.side_of(layout,first),first,count,fill,rep)

def placed_pieces_exact(~Element: Data,+layout: Pieces.Ranges,+fill: Element,+pieces: PiecesModel.Model<Element>,+slots: PiecesModel.Model<Element>) ->
  {Pieces.fold(~Storage.Buffer<Element>,~Pieces.Pieces(~Element),~Tensors.Depositing<Element>,~Tensors.placed_piece(~Element),Tensors.Depositing{layout,fill},PiecesModel.pack(~Element,pieces),PiecesModel.pack(~Element,slots)) ==
    PiecesModel.pack(~Element,placed_pieces(~Element,layout,fill,pieces,slots)) : Pieces.Pieces(~Element)}:
  match pieces:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      placed_when_exact(~Element,U32.is_zero(rows),layout,fill,first,rows,buffer,slots)
    case PiecesModel.Fork{+left,+right}:
      %Equal.sym(Pieces.Pieces(~Element),
          Pieces.fold(~Storage.Buffer<Element>,~Pieces.Pieces(~Element),~Tensors.Depositing<Element>,~Tensors.placed_piece(~Element),Tensors.Depositing{layout,fill},PiecesModel.pack(~Element,left),PiecesModel.pack(~Element,slots)),
          PiecesModel.pack(~Element,placed_pieces(~Element,layout,fill,left,slots)),placed_pieces_exact(~Element,layout,fill,left,slots)) :
        {Pieces.fold(~Storage.Buffer<Element>,~Pieces.Pieces(~Element),~Tensors.Depositing<Element>,~Tensors.placed_piece(~Element),Tensors.Depositing{layout,fill},PiecesModel.pack(~Element,right),_) ==
          PiecesModel.pack(~Element,placed_pieces(~Element,layout,fill,right,placed_pieces(~Element,layout,fill,left,slots))) : Pieces.Pieces(~Element)}
      placed_pieces_exact(~Element,layout,fill,right,placed_pieces(~Element,layout,fill,left,slots))

def placed_tasks_exact(~Element: Data,+layout: Pieces.Ranges,+fill: Element,+results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+slots: PiecesModel.Model<Element>) ->
  {Pieces.fold(~Pieces.Pieces(~Element),~Pieces.Pieces(~Element),~Tensors.Depositing<Element>,~Tensors.placed_component(~Element),Tensors.Depositing{layout,fill},
      PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),results),PiecesModel.pack(~Element,slots)) ==
    PiecesModel.pack(~Element,placed_tasks(~Element,layout,fill,results,slots)) : Pieces.Pieces(~Element)}:
  match results:
    case Partition.Leaf{+pieces,+first,+count,+details}:
      placed_pieces_exact(~Element,layout,fill,pieces,slots)
    case Partition.Fork{+left,+right}:
      %Equal.sym(Pieces.Pieces(~Element),
          Pieces.fold(~Pieces.Pieces(~Element),~Pieces.Pieces(~Element),~Tensors.Depositing<Element>,~Tensors.placed_component(~Element),Tensors.Depositing{layout,fill},
            PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),left),PiecesModel.pack(~Element,slots)),
          PiecesModel.pack(~Element,placed_tasks(~Element,layout,fill,left,slots)),placed_tasks_exact(~Element,layout,fill,left,slots)) :
        {Pieces.fold(~Pieces.Pieces(~Element),~Pieces.Pieces(~Element),~Tensors.Depositing<Element>,~Tensors.placed_component(~Element),Tensors.Depositing{layout,fill},
            PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),right),_) ==
          PiecesModel.pack(~Element,placed_tasks(~Element,layout,fill,right,placed_tasks(~Element,layout,fill,left,slots))) : Pieces.Pieces(~Element)}
      placed_tasks_exact(~Element,layout,fill,right,placed_tasks(~Element,layout,fill,left,slots))

# Tensors.placed packs the empty slots of its layout with the pieces of every
# task put in.
def placed_exact(~Element: Data,+layout: Pieces.Ranges,+fill: Element,+results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>) ->
  {Tensors.placed(~Element,layout,fill,PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),results)) ==
    PiecesModel.pack(~Element,placed_tasks(~Element,layout,fill,results,empty_slots(~Element,layout,fill))) : Pieces.Pieces(~Element)}:
  %Equal.sym(Unit & Pieces.Pieces(~Element),Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~Tensors.slot_piece(~Element),layout,fill,Unit{}),
      (Unit{},PiecesModel.pack(~Element,empty_slots(~Element,layout,fill))),slots_built(~Element,layout,fill)) :
    {Pieces.fold(~Pieces.Pieces(~Element),~Pieces.Pieces(~Element),~Tensors.Depositing<Element>,~Tensors.placed_component(~Element),Tensors.Depositing{layout,fill},
        PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),results),Pair.snd(Unit,Pieces.Pieces(~Element),_)) ==
      PiecesModel.pack(~Element,placed_tasks(~Element,layout,fill,results,empty_slots(~Element,layout,fill))) : Pieces.Pieces(~Element)}
  placed_tasks_exact(~Element,layout,fill,results,empty_slots(~Element,layout,fill))

# ---- What the placed slots hold ----

# Putting pieces keeps the layout of the slots.
def placed_when_layout(~Element: Data,+empty: Bool,+layout: Pieces.Ranges,+fill: Element,+first: U32,+count: U32,+rep: Buffer.Model<Element>,+slots: PiecesModel.Model<Element>) ->
  {PiecesModel.layout(~Element,placed_when_model(~Element,empty,layout,fill,first,count,rep,slots)) == PiecesModel.layout(~Element,slots) : Pieces.Ranges}:
  match empty:
    case True{}:
      {==}
    case False{}:
      deposit_layout(~Element,slots,layout,Tensors.side_of(layout,first),first,count,fill,rep)

def placed_pieces_layout(~Element: Data,+layout: Pieces.Ranges,+fill: Element,+pieces: PiecesModel.Model<Element>,+slots: PiecesModel.Model<Element>) ->
  {PiecesModel.layout(~Element,placed_pieces(~Element,layout,fill,pieces,slots)) == PiecesModel.layout(~Element,slots) : Pieces.Ranges}:
  match pieces:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      placed_when_layout(~Element,U32.is_zero(rows),layout,fill,first,rows,buffer,slots)
    case PiecesModel.Fork{+left,+right}:
      Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,placed_pieces(~Element,layout,fill,right,placed_pieces(~Element,layout,fill,left,slots))),
        PiecesModel.layout(~Element,placed_pieces(~Element,layout,fill,left,slots)),PiecesModel.layout(~Element,slots),
        placed_pieces_layout(~Element,layout,fill,right,placed_pieces(~Element,layout,fill,left,slots)),placed_pieces_layout(~Element,layout,fill,left,slots))

def placed_tasks_layout(~Element: Data,+layout: Pieces.Ranges,+fill: Element,+results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+slots: PiecesModel.Model<Element>) ->
  {PiecesModel.layout(~Element,placed_tasks(~Element,layout,fill,results,slots)) == PiecesModel.layout(~Element,slots) : Pieces.Ranges}:
  match results:
    case Partition.Leaf{+pieces,+first,+count,+details}:
      placed_pieces_layout(~Element,layout,fill,pieces,slots)
    case Partition.Fork{+left,+right}:
      Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,placed_tasks(~Element,layout,fill,right,placed_tasks(~Element,layout,fill,left,slots))),
        PiecesModel.layout(~Element,placed_tasks(~Element,layout,fill,left,slots)),PiecesModel.layout(~Element,slots),
        placed_tasks_layout(~Element,layout,fill,right,placed_tasks(~Element,layout,fill,left,slots)),placed_tasks_layout(~Element,layout,fill,left,slots))

# The cell after the last piece of the tasks.
def tasks_end(~Element: Data,results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>) -> Nat:
  match results:
    case Partition.Leaf{pieces,first,count,details}:
      Runs.range_high(PiecesModel.layout(~Element,pieces))
    case Partition.Fork{left,right}:
      tasks_end(~Element,right)

# The ranges of the pieces of the tasks in order, then rest.
def tasks_spans(~Element: Data,results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,rest: Held.Spans) -> Held.Spans:
  match results:
    case Partition.Leaf{pieces,first,count,details}:
      Held.layout_spans(PiecesModel.layout(~Element,pieces),rest)
    case Partition.Fork{left,right}:
      tasks_spans(~Element,left,tasks_spans(~Element,right,rest))

# A piece starts at a leaf of the layout with its cells, is not empty and
# holds the property.
type PieceAt<-Element: Data,-Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,-layout: Pieces.Ranges,-first: U32,-rows: U32,
  -buffer: Buffer.Model<Element>> is Data:
  PieceAt{at: LeafAt(layout,first),cells: {rows_of(layout,first) == rows : U32},filled: {U32.is_zero(rows) == False{} : Bool},holds: Property(first,rows,Buffer.content(~Element,buffer))}

def PiecesAt(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,layout: Pieces.Ranges,pieces: PiecesModel.Model<Element>) -> Data:
  match pieces:
    case PiecesModel.Leaf{buffer,first,rows}:
      PieceAt<Element,Property,layout,first,rows,buffer>
    case PiecesModel.Fork{left,right}:
      Held.Both<PiecesAt(~Element,~Property,layout,left),PiecesAt(~Element,~Property,layout,right)>

def TasksAt(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,layout: Pieces.Ranges,results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>) -> Data:
  match results:
    case Partition.Leaf{pieces,first,count,details}:
      PiecesAt(~Element,~Property,layout,pieces)
    case Partition.Fork{left,right}:
      Held.Both<TasksAt(~Element,~Property,layout,left),TasksAt(~Element,~Property,layout,right)>

# Putting the pieces of a task, and of the tasks, places their spans.
def pieces_good(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+fill: Element,+pieces: PiecesModel.Model<Element>,
  +slots: PiecesModel.Model<Element>,+rest: Held.Spans,+same: {PiecesModel.layout(~Element,slots) == layout : Pieces.Ranges},+ordered: Ordered(layout),
  +facts: PiecesAt(~Element,~Property,layout,pieces),+good: Good(~Element,~Property,slots,Held.layout_spans(PiecesModel.layout(~Element,pieces),rest))) ->
  Good(~Element,~Property,placed_pieces(~Element,layout,fill,pieces,slots),rest):
  match pieces:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      PieceAt{+at,+cells,+filled,+holds} = facts
      +back = Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,slots),layout,same)
      %Equal.sym(Bool,U32.is_zero(rows),False{},filled) : Good(~Element,~Property,placed_when_model(~Element,_,layout,fill,first,rows,buffer,slots),rest)
      %same : Good(~Element,~Property,deposit_at(~Element,slots,_,first,rows,fill,buffer),rest)
      deposit_step(~Element,~Property,slots,first,rows,fill,buffer,rest,
        {%back : Ordered(_) ordered : Ordered(PiecesModel.layout(~Element,slots))},
        {%back : LeafAt(_,first) at : LeafAt(PiecesModel.layout(~Element,slots),first)},
        {%back : {rows_of(_,first) == rows : U32} cells : {rows_of(PiecesModel.layout(~Element,slots),first) == rows : U32}},
        holds,good)
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_facts,+right_facts} = facts
      pieces_good(~Element,~Property,layout,fill,right,placed_pieces(~Element,layout,fill,left,slots),rest,
        Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,placed_pieces(~Element,layout,fill,left,slots)),PiecesModel.layout(~Element,slots),layout,
          placed_pieces_layout(~Element,layout,fill,left,slots),same),
        ordered,right_facts,
        pieces_good(~Element,~Property,layout,fill,left,slots,Held.layout_spans(PiecesModel.layout(~Element,right),rest),same,ordered,left_facts,good))

def tasks_good(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+fill: Element,
  +results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+slots: PiecesModel.Model<Element>,+rest: Held.Spans,+same: {PiecesModel.layout(~Element,slots) == layout : Pieces.Ranges},
  +ordered: Ordered(layout),+facts: TasksAt(~Element,~Property,layout,results),+good: Good(~Element,~Property,slots,tasks_spans(~Element,results,rest))) ->
  Good(~Element,~Property,placed_tasks(~Element,layout,fill,results,slots),rest):
  match results:
    case Partition.Leaf{+pieces,+first,+count,+details}:
      pieces_good(~Element,~Property,layout,fill,pieces,slots,rest,same,ordered,facts,good)
    case Partition.Fork{+left,+right}:
      Held.Both{+left_facts,+right_facts} = facts
      tasks_good(~Element,~Property,layout,fill,right,placed_tasks(~Element,layout,fill,left,slots),rest,
        Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,placed_tasks(~Element,layout,fill,left,slots)),PiecesModel.layout(~Element,slots),layout,
          placed_tasks_layout(~Element,layout,fill,left,slots),same),
        ordered,right_facts,tasks_good(~Element,~Property,layout,fill,left,slots,tasks_spans(~Element,right,rest),same,ordered,left_facts,good))

# ---- Pieces that follow a run ----

# Every piece of the tasks holds the property.
def TasksHold(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>) -> Data:
  match results:
    case Partition.Leaf{pieces,first,count,details}:
      PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,pieces))
    case Partition.Fork{left,right}:
      Held.Both<TasksHold(~Element,~Property,left),TasksHold(~Element,~Property,right)>

# Every range of a list, and every leaf of a part, is a leaf of the layout
# with its cells.
def SpansAt(+layout: Pieces.Ranges,spans: Held.Spans) -> Data:
  match spans:
    case Held.Ended{}:
      Unit
    case Held.Spanned{+first,+count,+rest}:
      Held.Both<LeafAt(layout,first),Held.Both<{rows_of(layout,first) == count : U32},SpansAt(layout,rest)>>

def LeavesAt(+layout: Pieces.Ranges,part: Pieces.Ranges) -> Data:
  match part:
    case Pieces.Range{+first,+count}:
      Held.Both<LeafAt(layout,first),{rows_of(layout,first) == count : U32}>
    case Pieces.Split{left,right}:
      Held.Both<LeavesAt(layout,left),LeavesAt(layout,right)>

def leaves_left(+left: Pieces.Ranges,+right: Pieces.Ranges,+part: Pieces.Ranges,+before: Before(left,U32.to_nat(Pieces.first(right))),+leaves: LeavesAt(left,part)) ->
  LeavesAt(Pieces.Split{left,right},part):
  match part:
    case Pieces.Range{+first,+count}:
      Held.Both{+at,+rows} = leaves
      Held.Both{OnLeft{at},
        {%Equal.sym(Bool,U32.is_lt(first,Pieces.first(right)),True{},left_side(left,right,first,before,at)) :
            {Bool.pick(U32,_,rows_of(left,first),rows_of(right,first)) == count : U32}
          rows : {Bool.pick(U32,U32.is_lt(first,Pieces.first(right)),rows_of(left,first),rows_of(right,first)) == count : U32}}}
    case Pieces.Split{+part_left,+part_right}:
      Held.Both{+left_leaves,+right_leaves} = leaves
      Held.Both{leaves_left(left,right,part_left,before,left_leaves),leaves_left(left,right,part_right,before,right_leaves)}

def leaves_right(+left: Pieces.Ranges,+right: Pieces.Ranges,+part: Pieces.Ranges,+ordered: Ordered(right),+leaves: LeavesAt(right,part)) ->
  LeavesAt(Pieces.Split{left,right},part):
  match part:
    case Pieces.Range{+first,+count}:
      Held.Both{+at,+rows} = leaves
      Held.Both{OnRight{at},
        {%Equal.sym(Bool,U32.is_lt(first,Pieces.first(right)),False{},right_side(right,first,ordered,at)) :
            {Bool.pick(U32,_,rows_of(left,first),rows_of(right,first)) == count : U32}
          rows : {Bool.pick(U32,U32.is_lt(first,Pieces.first(right)),rows_of(left,first),rows_of(right,first)) == count : U32}}}
    case Pieces.Split{+part_left,+part_right}:
      Held.Both{+left_leaves,+right_leaves} = leaves
      Held.Both{leaves_right(left,right,part_left,ordered,left_leaves),leaves_right(left,right,part_right,ordered,right_leaves)}

def own_leaves(layout: Pieces.Ranges,+ordered: Ordered(layout)) -> LeavesAt(layout,layout):
  match layout:
    case Pieces.Range{+first,+count}:
      Held.Both{{==},{==}}
    case Pieces.Split{+left,+right}:
      Held.Both{+left_ordered,+rest} = ordered
      Held.Both{+right_ordered,+before} = rest
      Held.Both{leaves_left(left,right,left,before,own_leaves(left,left_ordered)),leaves_right(left,right,right,right_ordered,own_leaves(right,right_ordered))}

def leaves_spans(+layout: Pieces.Ranges,part: Pieces.Ranges,+rest: Held.Spans,+leaves: LeavesAt(layout,part),+after: SpansAt(layout,rest)) ->
  SpansAt(layout,Held.layout_spans(part,rest)):
  match part:
    case Pieces.Range{+first,+count}:
      Held.Both{+at,+rows} = leaves
      Held.Both{at,Held.Both{rows,after}}
    case Pieces.Split{+left,+right}:
      Held.Both{+left_leaves,+right_leaves} = leaves
      leaves_spans(layout,left,Held.layout_spans(right,rest),left_leaves,leaves_spans(layout,right,rest,right_leaves,after))

# A run is ordered, and its leaves start before its end.
def half_end(+left: Pieces.Ranges,+right: Pieces.Ranges,+glue: {Runs.range_high(left) == Runs.range_low(right) : Nat}) ->
  {Runs.range_high(left) == U32.to_nat(Pieces.first(right)) : Nat}:
  Equal.trans(Nat,Runs.range_high(left),Runs.range_low(right),U32.to_nat(Pieces.first(right)),glue,
    Equal.sym(Nat,U32.to_nat(Pieces.first(right)),Runs.range_low(right),Runs.first_low(right)))

def half_within(+left: Pieces.Ranges,+right: Pieces.Ranges,+unit: Nat,+bound: Nat,+right_run: Runs.Run(right,unit,bound),+glue: {Runs.range_high(left) == Runs.range_low(right) : Nat}) ->
  {Nat.is_le(Runs.range_high(left),Runs.range_high(right)) == True{} : Bool}:
  %Equal.sym(Nat,Runs.range_high(left),Runs.range_low(right),glue) : {Nat.is_le(_,Runs.range_high(right)) == True{} : Bool}
  Bounds.strict_le(Runs.range_low(right),Runs.range_high(right),
    Runs.fits_less(unit,Runs.range_low(right),Runs.range_high(right),bound,Runs.run_fits(right,unit,bound,right_run)))

def run_before(layout: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound)) -> Before(layout,Runs.range_high(layout)):
  match layout:
    case Pieces.Range{+first,+count}:
      Runs.fits_less(unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),bound,run)
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      Held.Both{before_raised(left,Runs.range_high(left),Runs.range_high(right),run_before(left,unit,bound,left_run),half_within(left,right,unit,bound,right_run,glue)),
        run_before(right,unit,bound,right_run)}

def run_ordered(layout: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound)) -> Ordered(layout):
  match layout:
    case Pieces.Range{+first,+count}:
      Unit{}
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      Held.Both{run_ordered(left,unit,bound,left_run),Held.Both{run_ordered(right,unit,bound,right_run),
        {%half_end(left,right,glue) : Before(left,_) run_before(left,unit,bound,left_run) : Before(left,U32.to_nat(Pieces.first(right)))}}}

# The end of a chain of ranges from low.
def chain_end(spans: Held.Spans,+low: Nat) -> Nat:
  match spans:
    case Held.Ended{}:
      low
    case Held.Spanned{first,count,rest}:
      chain_end(rest,Nat.add(U32.to_nat(first),U32.to_nat(count)))

def layout_chain_end(layout: Pieces.Ranges,+rest: Held.Spans,+low: Nat) -> {chain_end(Held.layout_spans(layout,rest),low) == chain_end(rest,Runs.range_high(layout)) : Nat}:
  match layout:
    case Pieces.Range{+first,+count}:
      {==}
    case Pieces.Split{+left,+right}:
      Equal.trans(Nat,chain_end(Held.layout_spans(left,Held.layout_spans(right,rest)),low),chain_end(Held.layout_spans(right,rest),Runs.range_high(left)),
        chain_end(rest,Runs.range_high(right)),layout_chain_end(left,Held.layout_spans(right,rest),low),layout_chain_end(right,rest,Runs.range_high(left)))

type PiecesFound<-Element: Data,-Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,-layout: Pieces.Ranges,-pieces: PiecesModel.Model<Element>,
  -rest: Held.Spans,-low: Nat,-unit: Nat,-bound: Nat> is Data:
  PiecesFound{at: PiecesAt(~Element,~Property,layout,pieces),chain: Runs.Chain(rest,Runs.range_high(PiecesModel.layout(~Element,pieces)),unit,bound),spans: SpansAt(layout,rest)}

def pieces_found_chain(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+pieces: PiecesModel.Model<Element>,
  +rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+found: PiecesFound<Element,Property,layout,pieces,rest,low,unit,bound>) ->
  Runs.Chain(rest,Runs.range_high(PiecesModel.layout(~Element,pieces)),unit,bound):
  PiecesFound{+at,+chain,+spans} = found
  chain

def pieces_found_spans(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+pieces: PiecesModel.Model<Element>,
  +rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+found: PiecesFound<Element,Property,layout,pieces,rest,low,unit,bound>) -> SpansAt(layout,rest):
  PiecesFound{+at,+chain,+spans} = found
  spans

def pieces_found_at(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+pieces: PiecesModel.Model<Element>,
  +rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+found: PiecesFound<Element,Property,layout,pieces,rest,low,unit,bound>) -> PiecesAt(~Element,~Property,layout,pieces):
  PiecesFound{+at,+chain,+spans} = found
  at

def pieces_found(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+pieces: PiecesModel.Model<Element>,
  +rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+chain: Runs.Chain(Held.layout_spans(PiecesModel.layout(~Element,pieces),rest),low,unit,bound),
  +spans: SpansAt(layout,Held.layout_spans(PiecesModel.layout(~Element,pieces),rest)),+holds: PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,pieces))) ->
  PiecesFound<Element,Property,layout,pieces,rest,low,unit,bound>:
  match pieces:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      Held.Both{+fits,+more} = chain
      Held.Both{+starts,+next} = more
      Held.Both{+at,+spans_more} = spans
      Held.Both{+cells,+after} = spans_more
      +positive = Order.cancel_offset_lt(U32.to_nat(first),0n,U32.to_nat(rows),
        {%Equal.sym(Nat,Nat.add(U32.to_nat(first),0n),U32.to_nat(first),Addition.zero_right(U32.to_nat(first))) :
            {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}
          Runs.fits_less(unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),bound,fits) :
          {Nat.is_lt(Nat.add(U32.to_nat(first),0n),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}})
      PiecesFound{PieceAt{at,cells,Arithmetic.word_positive_nonzero(rows,positive),holds},next,after}
    case PiecesModel.Fork{+left,+right}:
      KernelGemm.Both{+left_holds,+right_holds} = holds
      +middle = Runs.range_high(PiecesModel.layout(~Element,left))
      +after_left = Held.layout_spans(PiecesModel.layout(~Element,right),rest)
      +left_found = pieces_found(~Element,~Property,layout,left,after_left,low,unit,bound,chain,spans,left_holds)
      +right_found = pieces_found(~Element,~Property,layout,right,rest,middle,unit,bound,
        pieces_found_chain(~Element,~Property,layout,left,after_left,low,unit,bound,left_found),
        pieces_found_spans(~Element,~Property,layout,left,after_left,low,unit,bound,left_found),right_holds)
      PiecesFound{Held.Both{pieces_found_at(~Element,~Property,layout,left,after_left,low,unit,bound,left_found),
          pieces_found_at(~Element,~Property,layout,right,rest,middle,unit,bound,right_found)},
        pieces_found_chain(~Element,~Property,layout,right,rest,middle,unit,bound,right_found),
        pieces_found_spans(~Element,~Property,layout,right,rest,middle,unit,bound,right_found)}

type TasksFound<-Element: Data,-Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,-layout: Pieces.Ranges,
  -results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,-rest: Held.Spans,-low: Nat,-unit: Nat,-bound: Nat> is Data:
  TasksFound{at: TasksAt(~Element,~Property,layout,results),chain: Runs.Chain(rest,tasks_end(~Element,results),unit,bound),spans: SpansAt(layout,rest)}

def tasks_found_chain(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,
  +results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+found: TasksFound<Element,Property,layout,results,rest,low,unit,bound>) ->
  Runs.Chain(rest,tasks_end(~Element,results),unit,bound):
  TasksFound{+at,+chain,+spans} = found
  chain

def tasks_found_spans(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,
  +results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+found: TasksFound<Element,Property,layout,results,rest,low,unit,bound>) ->
  SpansAt(layout,rest):
  TasksFound{+at,+chain,+spans} = found
  spans

def tasks_found_at(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,
  +results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+found: TasksFound<Element,Property,layout,results,rest,low,unit,bound>) ->
  TasksAt(~Element,~Property,layout,results):
  TasksFound{+at,+chain,+spans} = found
  at

def tasks_found(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,
  +results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+chain: Runs.Chain(tasks_spans(~Element,results,rest),low,unit,bound),
  +spans: SpansAt(layout,tasks_spans(~Element,results,rest)),+holds: TasksHold(~Element,~Property,results)) -> TasksFound<Element,Property,layout,results,rest,low,unit,bound>:
  match results:
    case Partition.Leaf{+pieces,+first,+count,+details}:
      +found = pieces_found(~Element,~Property,layout,pieces,rest,low,unit,bound,chain,spans,holds)
      TasksFound{pieces_found_at(~Element,~Property,layout,pieces,rest,low,unit,bound,found),pieces_found_chain(~Element,~Property,layout,pieces,rest,low,unit,bound,found),
        pieces_found_spans(~Element,~Property,layout,pieces,rest,low,unit,bound,found)}
    case Partition.Fork{+left,+right}:
      Held.Both{+left_holds,+right_holds} = holds
      +middle = tasks_end(~Element,left)
      +after_left = tasks_spans(~Element,right,rest)
      +left_found = tasks_found(~Element,~Property,layout,left,after_left,low,unit,bound,chain,spans,left_holds)
      +right_found = tasks_found(~Element,~Property,layout,right,rest,middle,unit,bound,
        tasks_found_chain(~Element,~Property,layout,left,after_left,low,unit,bound,left_found),
        tasks_found_spans(~Element,~Property,layout,left,after_left,low,unit,bound,left_found),right_holds)
      TasksFound{Held.Both{tasks_found_at(~Element,~Property,layout,left,after_left,low,unit,bound,left_found),
          tasks_found_at(~Element,~Property,layout,right,rest,middle,unit,bound,right_found)},
        tasks_found_chain(~Element,~Property,layout,right,rest,middle,unit,bound,right_found),
        tasks_found_spans(~Element,~Property,layout,right,rest,middle,unit,bound,right_found)}

# ---- The placed slots ----

# Every slot holds a piece with the property when every piece of the tasks
# lands on a leaf with its cells and holds the property, and every leaf starts
# one of the pieces, in any order.
def placed_listed(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+fill: Element,
  +results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+ordered: Ordered(layout),+facts: TasksAt(~Element,~Property,layout,results),
  +listed: LeavesListed(layout,tasks_spans(~Element,results,Held.Ended{}))) ->
  PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,placed_tasks(~Element,layout,fill,results,empty_slots(~Element,layout,fill)))):
  +slots = empty_slots(~Element,layout,fill)
  +kept = slots_layout(~Element,layout,fill)
  all_good(~Element,~Property,placed_tasks(~Element,layout,fill,results,slots),
    tasks_good(~Element,~Property,layout,fill,results,slots,Held.Ended{},kept,ordered,facts,
      listed_good(~Element,~Property,slots,tasks_spans(~Element,results,Held.Ended{}),
        {%Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,slots),layout,kept) : LeavesListed(_,tasks_spans(~Element,results,Held.Ended{}))
          listed : LeavesListed(PiecesModel.layout(~Element,slots),tasks_spans(~Element,results,Held.Ended{}))})))

# When the ranges of the pieces of the tasks in order are the ranges of a
# layout that is a run, every slot holds a piece with the property.
def placed_holds(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+fill: Element,
  +results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+low: Nat,+unit: Nat,+bound: Nat,+chain: Runs.Chain(Held.layout_spans(layout,Held.Ended{}),low,unit,bound),
  +same: {tasks_spans(~Element,results,Held.Ended{}) == Held.layout_spans(layout,Held.Ended{}) : Held.Spans},+holds: TasksHold(~Element,~Property,results)) ->
  PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,placed_tasks(~Element,layout,fill,results,empty_slots(~Element,layout,fill)))):
  +run_facts = Runs.tree_run(layout,Held.Ended{},low,unit,bound,chain)
  +run = Runs.part_run(layout,Held.Ended{},low,unit,bound,run_facts)
  +ordered = run_ordered(layout,unit,bound,run)
  +back = Equal.sym(Held.Spans,tasks_spans(~Element,results,Held.Ended{}),Held.layout_spans(layout,Held.Ended{}),same)
  +found = tasks_found(~Element,~Property,layout,results,Held.Ended{},low,unit,bound,
    {%back : Runs.Chain(_,low,unit,bound) chain : Runs.Chain(tasks_spans(~Element,results,Held.Ended{}),low,unit,bound)},
    {%back : SpansAt(layout,_) leaves_spans(layout,layout,Held.Ended{},own_leaves(layout,ordered),Unit{}) : SpansAt(layout,tasks_spans(~Element,results,Held.Ended{}))},holds)
  placed_listed(~Element,~Property,layout,fill,results,ordered,tasks_found_at(~Element,~Property,layout,results,Held.Ended{},low,unit,bound,found),
    {%back : LeavesListed(layout,_) own_listed(layout,Held.Ended{}) : LeavesListed(layout,tasks_spans(~Element,results,Held.Ended{}))})

# ---- The placed pieces ----

# The pieces placed from the results of tasks: their model, laid as the
# layout, every piece holding the property the pieces of the tasks hold.
type PlacedModel<-Element: Data,-Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,-layout: Pieces.Ranges,
  -pieces: Pieces.Pieces(~Element)> is Data:
  PlacedModel{model: PiecesModel.Model<Element>,exact: {pieces == PiecesModel.pack(~Element,model) : Pieces.Pieces(~Element)},
    laid: {PiecesModel.layout(~Element,model) == layout : Pieces.Ranges},holds: PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,model))}

def placed_model(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+fill: Element,
  +results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+low: Nat,+unit: Nat,+bound: Nat,+chain: Runs.Chain(Held.layout_spans(layout,Held.Ended{}),low,unit,bound),
  +same: {tasks_spans(~Element,results,Held.Ended{}) == Held.layout_spans(layout,Held.Ended{}) : Held.Spans},+holds: TasksHold(~Element,~Property,results)) ->
  PlacedModel<Element,Property,layout,Tensors.placed(~Element,layout,fill,PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),results))>:
  +slots = empty_slots(~Element,layout,fill)
  PlacedModel{placed_tasks(~Element,layout,fill,results,slots),placed_exact(~Element,layout,fill,results),
    Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,placed_tasks(~Element,layout,fill,results,slots)),PiecesModel.layout(~Element,slots),layout,
      placed_tasks_layout(~Element,layout,fill,results,slots),slots_layout(~Element,layout,fill)),
    placed_holds(~Element,~Property,layout,fill,results,low,unit,bound,chain,same,holds)}

# ---- The contract of placed pieces ----

# A piece that holds the values of a reader from its first cell, as many
# cells as its rows, in a ready buffer.
def PieceHolds(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  KernelGemm.Both<Content.piece_values(~Element,~Logical,~original,logical,first,rows,buffer),KernelGemm.Both<Runs.piece_sized(~Element,first,rows,buffer),PiecesModel.ready_piece(~Element,first,rows,buffer)>>

def holds_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>,
  holds: PieceHolds(~Element,~Logical,~original,logical,first,rows,buffer)) -> Content.piece_values(~Element,~Logical,~original,logical,first,rows,buffer):
  KernelGemm.Both{values,rest} = holds
  values

# A piece of a source holds the values of the original at its cells.
def SourceHolds(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: U32,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) ->
  Data:
  PieceHolds(~Element,~Logical,~original,logical,first,rows,buffer)

# A holding ordered as the pieces a component took holds their values.
def ordered_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+model: PiecesModel.Model<Element>,+holding: Held.Holding<Element>,
  +ordered: {Held.order(~Element,holding) == Held.stacked_model(~Element,model,Held.Bottom{}) : Held.Stack<Element>},
  +held: PiecesModel.ContentAll(~Element,~SourceHolds(~Element,~Logical,~original,logical,0),PiecesModel.content(~Element,model))) ->
  Held.HoldingValues(~Element,~Logical,~original,logical,holding):
  Held.holding_values(~Element,~Logical,~original,logical,holding,
    {%Equal.sym(Held.Stack<Element>,Held.order(~Element,holding),Held.stacked_model(~Element,model,Held.Bottom{}),ordered) :
        Held.StackValues(~Element,~Logical,~original,logical,_)
      Held.stacked_values(~Element,~Logical,~original,logical,model,Held.Bottom{},
        PiecesModel.content_map(~Element,~SourceHolds(~Element,~Logical,~original,logical,0),~Content.piece_values(~Element,~Logical,~original,logical),
          ~holds_values(~Element,~Logical,~original,logical),PiecesModel.content(~Element,model),held),Unit{}) :
      Held.StackValues(~Element,~Logical,~original,logical,Held.order(~Element,holding))})

def holds_sized(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>,
  holds: PieceHolds(~Element,~Logical,~original,logical,first,rows,buffer)) -> Runs.piece_sized(~Element,first,rows,buffer):
  KernelGemm.Both{values,rest} = holds
  KernelGemm.Both{sized,ready} = rest
  sized

def holds_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>,
  holds: PieceHolds(~Element,~Logical,~original,logical,first,rows,buffer)) -> PiecesModel.ready_piece(~Element,first,rows,buffer):
  KernelGemm.Both{values,rest} = holds
  KernelGemm.Both{sized,ready} = rest
  ready

# Pieces whose contents lie in the layout of the grid of a counted shape, a
# run from its first cell to its last, each holding the values of a reader,
# are a tensor of the contract of that reader.
def content_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  -pieces: Pieces.Pieces(~Element),+unit: Nat,+counted: Global.Counted<shape>,
  +run: Runs.Run(Pieces.ranges_of_grid(Tensors.grid_of(shape)),unit,ShapeLaws.elements(shape)),
  +low: {Runs.range_low(Pieces.ranges_of_grid(Tensors.grid_of(shape))) == 0n : Nat},
  +high: {Runs.range_high(Pieces.ranges_of_grid(Tensors.grid_of(shape))) == ShapeLaws.elements(shape) : Nat},
  +content: PiecesModel.Content<Element>,+contents_exact: {PiecesModel.contents(~Element,pieces) == content : PiecesModel.Content<Element>},
  +same_layout: {PiecesModel.content_layout(~Element,content) == Pieces.ranges_of_grid(Tensors.grid_of(shape)) : Pieces.Ranges},
  +holds: PiecesModel.ContentAll(~Element,~PieceHolds(~Element,~Logical,~original,logical),content)) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}):
  +layout = Pieces.ranges_of_grid(Tensors.grid_of(shape))
  +total = ShapeLaws.elements(shape)
  +content_run = {%Equal.sym(Pieces.Ranges,PiecesModel.content_layout(~Element,content),layout,same_layout) : Runs.Run(_,unit,total)
    run : Runs.Run(PiecesModel.content_layout(~Element,content),unit,total)}
  +covered = Runs.run_coverage(layout,unit,total,run)
  +covered_low = {%low : PiecesPlan.Coverage(layout,_,Nat.sub(Runs.range_high(layout),_)) covered : PiecesPlan.Coverage(layout,0n,Nat.sub(Runs.range_high(layout),0n))}
  +covered_high = {%high : PiecesPlan.Coverage(layout,0n,Nat.sub(_,0n)) covered_low : PiecesPlan.Coverage(layout,0n,Nat.sub(total,0n))}
  +covered_all = {%Order.subtract_zero(total) : PiecesPlan.Coverage(layout,0n,_) covered_high : PiecesPlan.Coverage(layout,0n,total)}
  +coverage = {%Equal.sym(Pieces.Ranges,PiecesModel.content_layout(~Element,content),layout,same_layout) : PiecesPlan.Coverage(_,0n,total)
    covered_all : PiecesPlan.Coverage(PiecesModel.content_layout(~Element,content),0n,total)}
  +values = PiecesModel.content_one(~Element,~Logical,~PieceHolds(~Element,~Logical,~original),~Content.piece_values(~Element,~Logical,~original),
    ~holds_values(~Element,~Logical,~original),logical,content,holds)
  +sized = PiecesModel.content_plain(~Element,~Logical,~PieceHolds(~Element,~Logical,~original),~Runs.piece_sized(~Element),
    ~holds_sized(~Element,~Logical,~original),logical,content,holds)
  +ready = PiecesModel.content_plain(~Element,~Logical,~PieceHolds(~Element,~Logical,~original),~PiecesModel.ready_piece(~Element),
    ~holds_ready(~Element,~Logical,~original),logical,content,holds)
  +well_formed = {Content.WellFormed{Runs.content_fields(~Element,content,unit,total,content_run,sized),ready,coverage} : Content.WellFormed<Element,total,content>}
  +packed = Equal.sym(PiecesModel.Content<Element>,PiecesModel.contents(~Element,pieces),content,contents_exact)
  Global.contract(~Element,~Logical,~original,logical,shape,fill,workers,pieces,counted,
    {%packed : {PiecesModel.content_layout(~Element,_) == layout : Pieces.Ranges} same_layout : Global.Canonical(~Element,shape,PiecesModel.contents(~Element,pieces))},
    {%packed : KernelGemm.Both<Content.Values(~Element,~Logical,~original,logical,_),Content.WellFormed<Element,total,_>> KernelGemm.Both{values,well_formed} :
      KernelGemm.Both<Content.Values(~Element,~Logical,~original,logical,PiecesModel.contents(~Element,pieces)),Content.WellFormed<Element,total,PiecesModel.contents(~Element,pieces)>>})

# Pieces placed in the layout of the grid of a counted shape, a run from its
# first cell to its last, each holding the values of a reader, are a tensor
# of the contract of that reader.
def placed_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  -pieces: Pieces.Pieces(~Element),+unit: Nat,+counted: Global.Counted<shape>,
  +run: Runs.Run(Pieces.ranges_of_grid(Tensors.grid_of(shape)),unit,ShapeLaws.elements(shape)),
  +low: {Runs.range_low(Pieces.ranges_of_grid(Tensors.grid_of(shape))) == 0n : Nat},
  +high: {Runs.range_high(Pieces.ranges_of_grid(Tensors.grid_of(shape))) == ShapeLaws.elements(shape) : Nat},
  +placed: PlacedModel<Element,PieceHolds(~Element,~Logical,~original,logical),Pieces.ranges_of_grid(Tensors.grid_of(shape)),pieces>) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}):
  PlacedModel{+model,+exact,+laid,+holds} = placed
  %Equal.sym(Pieces.Pieces(~Element),pieces,PiecesModel.pack(~Element,model),exact) : Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers})
  content_contract(~Element,~Logical,~original,logical,shape,fill,workers,PiecesModel.pack(~Element,model),unit,counted,run,low,high,PiecesModel.content(~Element,model),
    PiecesModel.packed_contents(~Element,model),
    Equal.trans(Pieces.Ranges,PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),PiecesModel.layout(~Element,model),Pieces.ranges_of_grid(Tensors.grid_of(shape)),
      Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,model),PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),PiecesModel.layout_matches_content(~Element,model)),laid),
    holds)
