# A write through a view in its banded form. Every band of the target takes,
# through its local view, the band of the part of the same rows; a band then
# holds at each cell what the write of the whole part through the whole view
# leaves there.
import Base
import ../tensor_view.bend as Views
import ../tensor_band.bend as Bands
import ../tensor_band_view.bend as BandViews
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./tensor_contract.bend as Contract
import ./tensor_scatter_values.bend as Scatter
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_band_values.bend as BandValues
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_view_values.bend as ViewValues
import ./tensor_band_view_classified.bend as Classified
import ./tensor_band_view_address.bend as Address
import ./nat_division_order.bend as DivisionOrder
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./view_writes.bend as ViewWrites
import ./tensor_copy_values.bend as Copy
import ./u32_comparison.bend as Comparison
import ./u32_full_range.bend as FullRange
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ../machine_limits.bend as Machine
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_band_schedule.bend as BandSchedule
import ../tensor_layout.bend as Layout
import ../tensor_operations.bend as Operations
import ./storage_shape.bend as Shape
import ./tensor_band_each.bend as Each
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_canonical_view.bend as CanonicalView
import ./tensor_gather_values.bend as GatherValues
import ./tensor_global_shape_count.bend as Global
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_line_values.bend as LineValues

# ---- Order of naturals ----

def below_successor(+value: Nat,+limit: Nat,below: {Nat.is_lt(value,1n+limit) == True{} : Bool}) -> {Nat.is_le(value,limit) == True{} : Bool}:
  match value limit:
    case 0n limit: Order.zero_le(limit)
    case 1n+ +value 0n: Empty.absurd({Nat.is_le(1n+value,0n) == True{} : Bool},Order.below_zero(value,below))
    case 1n+ +value 1n+ +limit: below_successor(value,limit,below)

def compared_below(less: Bool,+value: Nat,+limit: Nat,+compared: {Nat.is_lt(value,limit) == less : Bool},+bounded: {Nat.is_le(value,limit) == True{} : Bool},
  +unequal: {Nat.is_eq(value,limit) == False{} : Bool}) -> {Nat.is_lt(value,limit) == True{} : Bool}:
  match less:
    case True{}: compared
    case False{}:
      +equal = Order.antisymmetric(value,limit,bounded,Order.not_less(value,limit,compared))
      Empty.absurd({Nat.is_lt(value,limit) == True{} : Bool},WordEncoding.false_true(
        %unequal : {_ == True{} : Bool}
        %equal : {Nat.is_eq(value,_) == True{} : Bool}
        Order.equal_reflexive(value)))

# A value at most a limit and other than it is below it.
def unequal_below(+value: Nat,+limit: Nat,+bounded: {Nat.is_le(value,limit) == True{} : Bool},+unequal: {Nat.is_eq(value,limit) == False{} : Bool}) ->
  {Nat.is_lt(value,limit) == True{} : Bool}:
  compared_below(Nat.is_lt(value,limit),value,limit,{==},bounded,unequal)

# ---- Writes in blocks ----

# The element a query holds after count blocks of size writes from base, one
# block after another.
def through(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+size: Nat,+base: Nat,+view: Views.View,+part: Part,
  +query: U32,previous: Element) -> Element:
  match count:
    case 0n: previous
    case 1n+ +rest:
      Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size,Nat.add(base,Nat.mul(rest,size)),view,part),query,
        through(~Element,~Part,~value,rest,size,base,view,part,query,previous))

# The writes of count blocks are read block by block.
def folded(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+size: Nat,+base: Nat,+view: Views.View,+part: Part,
  +query: U32,+previous: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(count,size),base,view,part),query,previous) ==
    through(~Element,~Part,~value,count,size,base,view,part,query,previous) : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +done = Nat.mul(rest,size)
      +earlier = ViewWrites.writes_from(~Element,~Part,~value,done,base,view,part)
      +block = ViewWrites.writes_from(~Element,~Part,~value,size,Nat.add(base,done),view,part)
      +all = ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(1n+rest,size),base,view,part)
      +joined = Equal.trans(List<&2,Writes.Assignment<Element>>,all,ViewWrites.writes_from(~Element,~Part,~value,Nat.add(done,size),base,view,part),
        List.append(&2,Writes.Assignment<Element>,earlier,block),
        Equal.cong(Nat,List<&2,Writes.Assignment<Element>>,amount => ViewWrites.writes_from(~Element,~Part,~value,amount,base,view,part),Nat.add(size,done),Nat.add(done,size),
          Order.commutative(size,done)),
        ViewWrites.split(~Element,~Part,~value,done,size,base,view,part))
      Equal.trans(Element,Readback.selected(~Element,all,query,previous),Readback.selected(~Element,block,query,Readback.selected(~Element,earlier,query,previous)),
        Readback.selected(~Element,block,query,through(~Element,~Part,~value,rest,size,base,view,part,query,previous)),
        Equal.trans(Element,Readback.selected(~Element,all,query,previous),Readback.selected(~Element,List.append(&2,Writes.Assignment<Element>,earlier,block),query,previous),
          Readback.selected(~Element,block,query,Readback.selected(~Element,earlier,query,previous)),
          Equal.cong(List<&2,Writes.Assignment<Element>>,Element,writes => Readback.selected(~Element,writes,query,previous),all,List.append(&2,Writes.Assignment<Element>,earlier,block),joined),
          Readback.selected_append(~Element,earlier,block,query,previous)),
        Equal.cong(Element,Element,element => Readback.selected(~Element,block,query,element),Readback.selected(~Element,earlier,query,previous),
          through(~Element,~Part,~value,rest,size,base,view,part,query,previous),folded(~Element,~Part,~value,rest,size,base,view,part,query,previous)))

# Blocks below the chosen one keep the previous element when each block other
# than the chosen one keeps every element.
def through_before(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,~Context: Data,~size: @+context: Context -> Nat,
  ~base: @+context: Context -> Nat,~view: @+context: Context -> Views.View,~part: @+context: Context -> Part,~query: @+context: Context -> U32,
  ~total: @+context: Context -> Nat,~chosen: @+context: Context -> Nat,
  ~skip: @+context: Context -> @+block: Nat -> @+inside: {Nat.is_lt(block,total(context)) == True{} : Bool} -> @+other: {Nat.is_eq(block,chosen(context)) == False{} : Bool} ->
    @+element: Element ->
    {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(block,size(context))),view(context),part(context)),
      query(context),element) == element : Element},
  +context: Context,count: Nat,+below: {Nat.is_le(count,chosen(context)) == True{} : Bool},+chosen_inside: {Nat.is_lt(chosen(context),total(context)) == True{} : Bool},
  +previous: Element) ->
  {through(~Element,~Part,~value,count,size(context),base(context),view(context),part(context),query(context),previous) == previous : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +rest_below = Bounds.nat_trans(rest,1n+rest,chosen(context),Order.lt_succ(rest,rest,Order.reflexive(rest)),below)
      +rest_inside = Order.le_lt_trans(rest,chosen(context),total(context),Bounds.strict_le(rest,chosen(context),rest_below),chosen_inside)
      +earlier = through(~Element,~Part,~value,rest,size(context),base(context),view(context),part(context),query(context),previous)
      Equal.trans(Element,
        Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(rest,size(context))),view(context),part(context)),
          query(context),earlier),
        earlier,previous,
        skip(context,rest,rest_inside,Order.less_not_equal(rest,chosen(context),rest_below),earlier),
        through_before(~Element,~Part,~value,~Context,~size,~base,~view,~part,~query,~total,~chosen,~skip,context,rest,Bounds.strict_le(rest,chosen(context),rest_below),
          chosen_inside,previous))

def through_step(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,~Context: Data,~size: @+context: Context -> Nat,
  ~base: @+context: Context -> Nat,~view: @+context: Context -> Views.View,~part: @+context: Context -> Part,~query: @+context: Context -> U32,
  ~total: @+context: Context -> Nat,~chosen: @+context: Context -> Nat,
  ~skip: @+context: Context -> @+block: Nat -> @+inside: {Nat.is_lt(block,total(context)) == True{} : Bool} -> @+other: {Nat.is_eq(block,chosen(context)) == False{} : Bool} ->
    @+element: Element ->
    {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(block,size(context))),view(context),part(context)),
      query(context),element) == element : Element},
  +context: Context,+rest: Nat,same: Bool,+compared: {Nat.is_eq(rest,chosen(context)) == same : Bool},
  +above: {Nat.is_lt(chosen(context),1n+rest) == True{} : Bool},+room: {Nat.is_le(1n+rest,total(context)) == True{} : Bool},+previous: Element) ->
  {through(~Element,~Part,~value,1n+rest,size(context),base(context),view(context),part(context),query(context),previous) ==
    Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(chosen(context),size(context))),view(context),part(context)),
      query(context),previous) : Element}:
  match rest same:
    case rest True{}:
      +chosen_inside = Bounds.nat_trans(chosen(context),1n+rest,total(context),above,room)
      +earlier = through(~Element,~Part,~value,rest,size(context),base(context),view(context),part(context),query(context),previous)
      +equal = Order.equal_of_is_eq(rest,chosen(context),compared)
      %equal :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(rest,size(context))),view(context),part(context)),
            query(context),earlier) ==
          Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(_,size(context))),view(context),part(context)),
            query(context),previous) : Element}
      Equal.cong(Element,Element,element =>
          Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(rest,size(context))),view(context),part(context)),
            query(context),element),
        earlier,previous,
        through_before(~Element,~Part,~value,~Context,~size,~base,~view,~part,~query,~total,~chosen,~skip,context,rest,
          %equal : {Nat.is_le(rest,_) == True{} : Bool} Order.reflexive(rest),chosen_inside,previous))
    case 0n False{}:
      Empty.absurd({through(~Element,~Part,~value,1n,size(context),base(context),view(context),part(context),query(context),previous) ==
        Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(chosen(context),size(context))),view(context),part(context)),
          query(context),previous) : Element},
        Order.below_zero(chosen(context),unequal_below(chosen(context),0n,below_successor(chosen(context),0n,above),
          %Order.equal_symmetric(0n,chosen(context)) : {_ == False{} : Bool} compared)))
    case 1n+ +shorter False{}:
      +longer = {1n+shorter : Nat}
      +chosen_below = unequal_below(chosen(context),longer,below_successor(chosen(context),longer,above),
        %Order.equal_symmetric(longer,chosen(context)) : {_ == False{} : Bool} compared)
      +longer_inside = Bounds.nat_trans(longer,1n+longer,total(context),Order.lt_succ(longer,longer,Order.reflexive(longer)),room)
      +earlier = through(~Element,~Part,~value,longer,size(context),base(context),view(context),part(context),query(context),previous)
      Equal.trans(Element,
        Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(longer,size(context))),view(context),part(context)),
          query(context),earlier),
        earlier,
        Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(chosen(context),size(context))),view(context),part(context)),
          query(context),previous),
        skip(context,longer,longer_inside,compared,earlier),
        through_step(~Element,~Part,~value,~Context,~size,~base,~view,~part,~query,~total,~chosen,~skip,context,shorter,Nat.is_eq(shorter,chosen(context)),{==},chosen_below,
          Bounds.strict_le(longer,total(context),longer_inside),previous))

# Count blocks give what the chosen block gives when every other block keeps
# every element.
def through_from(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,~Context: Data,~size: @+context: Context -> Nat,
  ~base: @+context: Context -> Nat,~view: @+context: Context -> Views.View,~part: @+context: Context -> Part,~query: @+context: Context -> U32,
  ~total: @+context: Context -> Nat,~chosen: @+context: Context -> Nat,
  ~skip: @+context: Context -> @+block: Nat -> @+inside: {Nat.is_lt(block,total(context)) == True{} : Bool} -> @+other: {Nat.is_eq(block,chosen(context)) == False{} : Bool} ->
    @+element: Element ->
    {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(block,size(context))),view(context),part(context)),
      query(context),element) == element : Element},
  +context: Context,count: Nat,+above: {Nat.is_lt(chosen(context),count) == True{} : Bool},+room: {Nat.is_le(count,total(context)) == True{} : Bool},+previous: Element) ->
  {through(~Element,~Part,~value,count,size(context),base(context),view(context),part(context),query(context),previous) ==
    Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(chosen(context),size(context))),view(context),part(context)),
      query(context),previous) : Element}:
  match count:
    case 0n:
      Empty.absurd({previous ==
        Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(chosen(context),size(context))),view(context),part(context)),
          query(context),previous) : Element},Order.below_zero(chosen(context),above))
    case 1n+ +rest:
      through_step(~Element,~Part,~value,~Context,~size,~base,~view,~part,~query,~total,~chosen,~skip,context,rest,Nat.is_eq(rest,chosen(context)),{==},above,room,previous)

# Count blocks each of which keeps every element keep the previous element.
def through_kept(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,~Context: Data,~size: @+context: Context -> Nat,
  ~base: @+context: Context -> Nat,~view: @+context: Context -> Views.View,~part: @+context: Context -> Part,~query: @+context: Context -> U32,
  ~total: @+context: Context -> Nat,
  ~keep: @+context: Context -> @+block: Nat -> @+inside: {Nat.is_lt(block,total(context)) == True{} : Bool} -> @+element: Element ->
    {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(block,size(context))),view(context),part(context)),
      query(context),element) == element : Element},
  +context: Context,count: Nat,+room: {Nat.is_le(count,total(context)) == True{} : Bool},+previous: Element) ->
  {through(~Element,~Part,~value,count,size(context),base(context),view(context),part(context),query(context),previous) == previous : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +rest_inside = Bounds.nat_trans(rest,1n+rest,total(context),Order.lt_succ(rest,rest,Order.reflexive(rest)),room)
      +earlier = through(~Element,~Part,~value,rest,size(context),base(context),view(context),part(context),query(context),previous)
      Equal.trans(Element,
        Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(rest,size(context))),view(context),part(context)),
          query(context),earlier),
        earlier,previous,
        keep(context,rest,rest_inside,earlier),
        through_kept(~Element,~Part,~value,~Context,~size,~base,~view,~part,~query,~total,~keep,context,rest,Bounds.strict_le(rest,total(context),rest_inside),previous))

# Two block structures whose blocks give the same element from every element
# read the same.
def through_parallel(~Element: Data,~Left: Data,~left_value: @+part: Left -> @+position: Nat -> Element,~Right: Data,~right_value: @+part: Right -> @+position: Nat -> Element,
  ~Context: Data,~left_size: @+context: Context -> Nat,~left_base: @+context: Context -> Nat,~left_view: @+context: Context -> Views.View,~left_part: @+context: Context -> Left,
  ~left_query: @+context: Context -> U32,~right_size: @+context: Context -> Nat,~right_base: @+context: Context -> Nat,~right_view: @+context: Context -> Views.View,
  ~right_part: @+context: Context -> Right,~right_query: @+context: Context -> U32,~total: @+context: Context -> Nat,
  ~same: @+context: Context -> @+block: Nat -> @+inside: {Nat.is_lt(block,total(context)) == True{} : Bool} -> @+element: Element ->
    {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Left,~left_value,left_size(context),Nat.add(left_base(context),Nat.mul(block,left_size(context))),
        left_view(context),left_part(context)),left_query(context),element) ==
      Readback.selected(~Element,ViewWrites.writes_from(~Element,~Right,~right_value,right_size(context),Nat.add(right_base(context),Nat.mul(block,right_size(context))),
        right_view(context),right_part(context)),right_query(context),element) : Element},
  +context: Context,count: Nat,+room: {Nat.is_le(count,total(context)) == True{} : Bool},+previous: Element) ->
  {through(~Element,~Left,~left_value,count,left_size(context),left_base(context),left_view(context),left_part(context),left_query(context),previous) ==
    through(~Element,~Right,~right_value,count,right_size(context),right_base(context),right_view(context),right_part(context),right_query(context),previous) : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +rest_inside = Bounds.nat_trans(rest,1n+rest,total(context),Order.lt_succ(rest,rest,Order.reflexive(rest)),room)
      +left_earlier = through(~Element,~Left,~left_value,rest,left_size(context),left_base(context),left_view(context),left_part(context),left_query(context),previous)
      +right_earlier = through(~Element,~Right,~right_value,rest,right_size(context),right_base(context),right_view(context),right_part(context),right_query(context),previous)
      %Equal.sym(Element,left_earlier,right_earlier,
          through_parallel(~Element,~Left,~left_value,~Right,~right_value,~Context,~left_size,~left_base,~left_view,~left_part,~left_query,~right_size,~right_base,~right_view,
            ~right_part,~right_query,~total,~same,context,rest,Bounds.strict_le(rest,total(context),rest_inside),previous)) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Left,~left_value,left_size(context),Nat.add(left_base(context),Nat.mul(rest,left_size(context))),
            left_view(context),left_part(context)),left_query(context),_) ==
          Readback.selected(~Element,ViewWrites.writes_from(~Element,~Right,~right_value,right_size(context),Nat.add(right_base(context),Nat.mul(rest,right_size(context))),
            right_view(context),right_part(context)),right_query(context),right_earlier) : Element}
      same(context,rest,rest_inside,right_earlier)

# Writes of which none reaches the query keep the previous element.
def missed_from(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,~Context: Data,~view: @+context: Context -> Views.View,
  ~part: @+context: Context -> Part,~query: @+context: Context -> U32,~base: @+context: Context -> Nat,~size: @+context: Context -> Nat,
  ~miss: @+context: Context -> @+index: Nat -> @+inside: {Nat.is_lt(index,size(context)) == True{} : Bool} ->
    {U32.is_eq(Copy.address(view(context),Nat.add(base(context),index)),query(context)) == False{} : Bool},
  +context: Context,count: Nat,+done: Nat,+room: {Nat.is_le(Nat.add(done,count),size(context)) == True{} : Bool},+previous: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,count,Nat.add(base(context),done),view(context),part(context)),query(context),previous) ==
    previous : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +position = Nat.add(base(context),done)
      +inside = Order.room_index(done,rest,size(context),room)
      %Addition.successor_right(base(context),done) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,rest,_,view(context),part(context)),query(context),
          Bool.pick(Element,U32.is_eq(Copy.address(view(context),position),query(context)),value(part(context),position),previous)) == previous : Element}
      %Equal.sym(Bool,U32.is_eq(Copy.address(view(context),position),query(context)),False{},miss(context,done,inside)) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,rest,Nat.add(base(context),1n+done),view(context),part(context)),query(context),
          Bool.pick(Element,_,value(part(context),position),previous)) == previous : Element}
      missed_from(~Element,~Part,~value,~Context,~view,~part,~query,~base,~size,~miss,context,rest,1n+done,Order.advance_room(done,rest,size(context),room),previous)

# Size writes of which none reaches the query keep the previous element.
def missed(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,~Context: Data,~view: @+context: Context -> Views.View,
  ~part: @+context: Context -> Part,~query: @+context: Context -> U32,~base: @+context: Context -> Nat,~size: @+context: Context -> Nat,
  ~miss: @+context: Context -> @+index: Nat -> @+inside: {Nat.is_lt(index,size(context)) == True{} : Bool} ->
    {U32.is_eq(Copy.address(view(context),Nat.add(base(context),index)),query(context)) == False{} : Bool},
  +context: Context,+previous: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),base(context),view(context),part(context)),query(context),previous) ==
    previous : Element}:
  %Addition.zero_right(base(context)) :
    {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,size(context),_,view(context),part(context)),query(context),previous) == previous : Element}
  missed_from(~Element,~Part,~value,~Context,~view,~part,~query,~base,~size,~miss,context,size(context),0n,Order.reflexive(size(context)),previous)

# Two runs of writes whose positions reach their queries together and carry
# the same elements read the same.
def parallel_from(~Element: Data,~Left: Data,~left_value: @+part: Left -> @+position: Nat -> Element,~Right: Data,~right_value: @+part: Right -> @+position: Nat -> Element,
  ~Context: Data,~left_view: @+context: Context -> Views.View,~left_part: @+context: Context -> Left,~left_query: @+context: Context -> U32,~left_base: @+context: Context -> Nat,
  ~right_view: @+context: Context -> Views.View,~right_part: @+context: Context -> Right,~right_query: @+context: Context -> U32,~right_base: @+context: Context -> Nat,
  ~size: @+context: Context -> Nat,
  ~hits: @+context: Context -> @+index: Nat -> @+inside: {Nat.is_lt(index,size(context)) == True{} : Bool} ->
    {U32.is_eq(Copy.address(left_view(context),Nat.add(left_base(context),index)),left_query(context)) ==
      U32.is_eq(Copy.address(right_view(context),Nat.add(right_base(context),index)),right_query(context)) : Bool},
  ~values: @+context: Context -> @+index: Nat -> @+inside: {Nat.is_lt(index,size(context)) == True{} : Bool} ->
    {left_value(left_part(context),Nat.add(left_base(context),index)) == right_value(right_part(context),Nat.add(right_base(context),index)) : Element},
  +context: Context,count: Nat,+done: Nat,+room: {Nat.is_le(Nat.add(done,count),size(context)) == True{} : Bool},+previous: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Left,~left_value,count,Nat.add(left_base(context),done),left_view(context),left_part(context)),left_query(context),previous) ==
    Readback.selected(~Element,ViewWrites.writes_from(~Element,~Right,~right_value,count,Nat.add(right_base(context),done),right_view(context),right_part(context)),right_query(context),previous) :
    Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +left_position = Nat.add(left_base(context),done)
      +right_position = Nat.add(right_base(context),done)
      +inside = Order.room_index(done,rest,size(context),room)
      +left_hit = U32.is_eq(Copy.address(left_view(context),left_position),left_query(context))
      +right_hit = U32.is_eq(Copy.address(right_view(context),right_position),right_query(context))
      +left_next = Bool.pick(Element,left_hit,left_value(left_part(context),left_position),previous)
      +right_next = Bool.pick(Element,right_hit,right_value(right_part(context),right_position),previous)
      +next_same = {
        %hits(context,done,inside) : {Bool.pick(Element,left_hit,left_value(left_part(context),left_position),previous) == Bool.pick(Element,_,right_value(right_part(context),right_position),previous) : Element}
        %values(context,done,inside) : {Bool.pick(Element,left_hit,left_value(left_part(context),left_position),previous) == Bool.pick(Element,left_hit,_,previous) : Element}
        {==} : {left_next == right_next : Element}}
      %Addition.successor_right(left_base(context),done) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Left,~left_value,rest,_,left_view(context),left_part(context)),left_query(context),left_next) ==
          Readback.selected(~Element,ViewWrites.writes_from(~Element,~Right,~right_value,rest,1n+right_position,right_view(context),right_part(context)),right_query(context),right_next) : Element}
      %Addition.successor_right(right_base(context),done) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Left,~left_value,rest,Nat.add(left_base(context),1n+done),left_view(context),left_part(context)),left_query(context),left_next) ==
          Readback.selected(~Element,ViewWrites.writes_from(~Element,~Right,~right_value,rest,_,right_view(context),right_part(context)),right_query(context),right_next) : Element}
      %next_same :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Left,~left_value,rest,Nat.add(left_base(context),1n+done),left_view(context),left_part(context)),left_query(context),left_next) ==
          Readback.selected(~Element,ViewWrites.writes_from(~Element,~Right,~right_value,rest,Nat.add(right_base(context),1n+done),right_view(context),right_part(context)),right_query(context),_) : Element}
      parallel_from(~Element,~Left,~left_value,~Right,~right_value,~Context,~left_view,~left_part,~left_query,~left_base,~right_view,~right_part,~right_query,~right_base,~size,
        ~hits,~values,context,rest,1n+done,Order.advance_room(done,rest,size(context),room),left_next)

def parallel(~Element: Data,~Left: Data,~left_value: @+part: Left -> @+position: Nat -> Element,~Right: Data,~right_value: @+part: Right -> @+position: Nat -> Element,
  ~Context: Data,~left_view: @+context: Context -> Views.View,~left_part: @+context: Context -> Left,~left_query: @+context: Context -> U32,~left_base: @+context: Context -> Nat,
  ~right_view: @+context: Context -> Views.View,~right_part: @+context: Context -> Right,~right_query: @+context: Context -> U32,~right_base: @+context: Context -> Nat,
  ~size: @+context: Context -> Nat,
  ~hits: @+context: Context -> @+index: Nat -> @+inside: {Nat.is_lt(index,size(context)) == True{} : Bool} ->
    {U32.is_eq(Copy.address(left_view(context),Nat.add(left_base(context),index)),left_query(context)) ==
      U32.is_eq(Copy.address(right_view(context),Nat.add(right_base(context),index)),right_query(context)) : Bool},
  ~values: @+context: Context -> @+index: Nat -> @+inside: {Nat.is_lt(index,size(context)) == True{} : Bool} ->
    {left_value(left_part(context),Nat.add(left_base(context),index)) == right_value(right_part(context),Nat.add(right_base(context),index)) : Element},
  +context: Context,+previous: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Left,~left_value,size(context),left_base(context),left_view(context),left_part(context)),left_query(context),previous) ==
    Readback.selected(~Element,ViewWrites.writes_from(~Element,~Right,~right_value,size(context),right_base(context),right_view(context),right_part(context)),right_query(context),previous) :
    Element}:
  %Addition.zero_right(left_base(context)) :
    {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Left,~left_value,size(context),_,left_view(context),left_part(context)),left_query(context),previous) ==
      Readback.selected(~Element,ViewWrites.writes_from(~Element,~Right,~right_value,size(context),right_base(context),right_view(context),right_part(context)),right_query(context),previous) : Element}
  %Addition.zero_right(right_base(context)) :
    {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Left,~left_value,size(context),Nat.add(left_base(context),0n),left_view(context),left_part(context)),left_query(context),previous) ==
      Readback.selected(~Element,ViewWrites.writes_from(~Element,~Right,~right_value,size(context),_,right_view(context),right_part(context)),right_query(context),previous) : Element}
  parallel_from(~Element,~Left,~left_value,~Right,~right_value,~Context,~left_view,~left_part,~left_query,~left_base,~right_view,~right_part,~right_query,~right_base,~size,
    ~hits,~values,context,size(context),0n,Order.reflexive(size(context)),previous)

# ---- Digits of a radix ----

def equal_offset(+offset: Nat,+left: Nat,+right: Nat) -> {Nat.is_eq(Nat.add(offset,left),Nat.add(offset,right)) == Nat.is_eq(left,right) : Bool}:
  match offset:
    case 0n: {==}
    case 1n+ +offset: equal_offset(offset,left,right)

# A value below a multiple of the width plus a digit below the width lies
# below the next multiple.
def below_next(+high: Nat,+low: Nat,+width: Nat,+low_inside: {Nat.is_lt(low,width) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(high,width),low),Nat.mul(1n+high,width)) == True{} : Bool}:
  %Order.commutative(low,Nat.mul(high,width)) : {Nat.is_lt(_,Nat.add(width,Nat.mul(high,width))) == True{} : Bool}
  Order.add_lt(low,width,Nat.mul(high,width),low_inside)

def radix_below(+high: Nat,+low: Nat,+other_high: Nat,+other_low: Nat,+width: Nat,+low_inside: {Nat.is_lt(low,width) == True{} : Bool},
  +less: {Nat.is_lt(high,other_high) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(high,width),low),Nat.add(Nat.mul(other_high,width),other_low)) == True{} : Bool}:
  Bounds.nat_trans(Nat.add(Nat.mul(high,width),low),Nat.mul(1n+high,width),Nat.add(Nat.mul(other_high,width),other_low),below_next(high,low,width,low_inside),
    Order.transitive(Nat.mul(1n+high,width),Nat.mul(other_high,width),Nat.add(Nat.mul(other_high,width),other_low),
      RowMajor.mul_le(1n+high,other_high,width,Order.strict_follow(high,other_high,less)),Order.le_add(Nat.mul(other_high,width),other_low)))

def radix_compared(less: Bool,more: Bool,+high: Nat,+low: Nat,+other_high: Nat,+other_low: Nat,+width: Nat,
  +low_inside: {Nat.is_lt(low,width) == True{} : Bool},+other_low_inside: {Nat.is_lt(other_low,width) == True{} : Bool},
  +less_compared: {Nat.is_lt(high,other_high) == less : Bool},+more_compared: {Nat.is_lt(other_high,high) == more : Bool}) ->
  {Nat.is_eq(Nat.add(Nat.mul(high,width),low),Nat.add(Nat.mul(other_high,width),other_low)) == (Nat.is_eq(high,other_high) && Nat.is_eq(low,other_low)) : Bool}:
  match less more:
    case True{} more:
      +left = Nat.add(Nat.mul(high,width),low)
      +right = Nat.add(Nat.mul(other_high,width),other_low)
      %Equal.sym(Bool,Nat.is_eq(high,other_high),False{},Order.less_not_equal(high,other_high,less_compared)) : {Nat.is_eq(left,right) == (_ && Nat.is_eq(low,other_low)) : Bool}
      Order.less_not_equal(left,right,radix_below(high,low,other_high,other_low,width,low_inside,less_compared))
    case False{} True{}:
      +left = Nat.add(Nat.mul(high,width),low)
      +right = Nat.add(Nat.mul(other_high,width),other_low)
      %Equal.sym(Bool,Nat.is_eq(high,other_high),False{},
          Equal.trans(Bool,Nat.is_eq(high,other_high),Nat.is_eq(other_high,high),False{},Order.equal_symmetric(high,other_high),Order.less_not_equal(other_high,high,more_compared))) :
        {Nat.is_eq(left,right) == (_ && Nat.is_eq(low,other_low)) : Bool}
      Equal.trans(Bool,Nat.is_eq(left,right),Nat.is_eq(right,left),False{},Order.equal_symmetric(left,right),
        Order.less_not_equal(right,left,radix_below(other_high,other_low,high,low,width,other_low_inside,more_compared)))
    case False{} False{}:
      +left = Nat.add(Nat.mul(high,width),low)
      +right = Nat.add(Nat.mul(other_high,width),other_low)
      +same = Order.antisymmetric(high,other_high,Order.not_less(other_high,high,more_compared),Order.not_less(high,other_high,less_compared))
      %same : {Nat.is_eq(left,Nat.add(Nat.mul(_,width),other_low)) == (Nat.is_eq(high,_) && Nat.is_eq(low,other_low)) : Bool}
      %Equal.sym(Bool,Nat.is_eq(high,high),True{},Order.equal_reflexive(high)) :
        {Nat.is_eq(left,Nat.add(Nat.mul(high,width),other_low)) == (_ && Nat.is_eq(low,other_low)) : Bool}
      equal_offset(Nat.mul(high,width),low,other_low)

# Two values of a high digit and a low digit below the width are equal
# exactly when both digits are.
def radix_equal(+high: Nat,+low: Nat,+other_high: Nat,+other_low: Nat,+width: Nat,
  +low_inside: {Nat.is_lt(low,width) == True{} : Bool},+other_low_inside: {Nat.is_lt(other_low,width) == True{} : Bool}) ->
  {Nat.is_eq(Nat.add(Nat.mul(high,width),low),Nat.add(Nat.mul(other_high,width),other_low)) == (Nat.is_eq(high,other_high) && Nat.is_eq(low,other_low)) : Bool}:
  radix_compared(Nat.is_lt(high,other_high),Nat.is_lt(other_high,high),high,low,other_high,other_low,width,low_inside,other_low_inside,{==},{==})

# Two row-major indices of a channel, a row below the height and a column
# below the width are equal exactly when all three are.
def index_equal(+channel: Nat,+row: Nat,+column: Nat,+other_channel: Nat,+other_row: Nat,+other_column: Nat,+height: Nat,+width: Nat,
  +row_inside: {Nat.is_lt(row,height) == True{} : Bool},+other_row_inside: {Nat.is_lt(other_row,height) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,width) == True{} : Bool},+other_column_inside: {Nat.is_lt(other_column,width) == True{} : Bool}) ->
  {Nat.is_eq(RowMajor.index(channel,height,width,row,column),RowMajor.index(other_channel,height,width,other_row,other_column)) ==
    (Nat.is_eq(channel,other_channel) && (Nat.is_eq(row,other_row) && Nat.is_eq(column,other_column))) : Bool}:
  +plane = Nat.mul(height,width)
  +low = Nat.add(Nat.mul(row,width),column)
  +other_low = Nat.add(Nat.mul(other_row,width),other_column)
  %radix_equal(row,column,other_row,other_column,width,column_inside,other_column_inside) :
    {Nat.is_eq(RowMajor.index(channel,height,width,row,column),RowMajor.index(other_channel,height,width,other_row,other_column)) == (Nat.is_eq(channel,other_channel) && _) : Bool}
  radix_equal(channel,low,other_channel,other_low,plane,RowMajor.radix(row,height,column,width,row_inside,column_inside),
    RowMajor.radix(other_row,height,other_column,width,other_row_inside,other_column_inside))

# Two words of numbers that fit a capacity are equal exactly when the numbers
# are.
def word_equal(+left: Nat,+right: Nat,+capacity: U32,+left_room: {Nat.is_le(left,U32.to_nat(capacity)) == True{} : Bool},
  +right_room: {Nat.is_le(right,U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.is_eq(U32.from_nat(left),U32.from_nat(right)) == Nat.is_eq(left,right) : Bool}:
  %FullRange.roundtrip(left,capacity,left_room) : {U32.is_eq(U32.from_nat(left),U32.from_nat(right)) == Nat.is_eq(_,right) : Bool}
  %FullRange.roundtrip(right,capacity,right_room) : {U32.is_eq(U32.from_nat(left),U32.from_nat(right)) == Nat.is_eq(U32.to_nat(U32.from_nat(left)),_) : Bool}
  Comparison.eq(U32.from_nat(left),U32.from_nat(right))

# ---- Numbers of a banded write ----

# A position of the whole write counted in images, channel blocks, rows and
# columns is a row-major index.
def whole_position(+image: Nat,+channel: Nat,+row: Nat,+column: Nat,+early: Nat,+height: Nat,+late: Nat) ->
  {Nat.add(Nat.add(Nat.add(Nat.mul(image,Nat.mul(early,Nat.mul(height,late))),Nat.mul(channel,Nat.mul(height,late))),Nat.mul(row,late)),column) ==
    RowMajor.index(Nat.add(Nat.mul(image,early),channel),height,late,row,column) : Nat}:
  +plane = Nat.mul(height,late)
  +rest = Nat.add(Nat.mul(row,late),column)
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,early),channel),plane),Nat.add(Nat.mul(Nat.mul(image,early),plane),Nat.mul(channel,plane)),
      RowMajor.distrib(Nat.mul(image,early),channel,plane)) :
    {Nat.add(Nat.add(Nat.add(Nat.mul(image,Nat.mul(early,plane)),Nat.mul(channel,plane)),Nat.mul(row,late)),column) == Nat.add(_,rest) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(image,early),plane),Nat.mul(image,Nat.mul(early,plane)),RowMajor.assoc(image,early,plane)) :
    {Nat.add(Nat.add(Nat.add(Nat.mul(image,Nat.mul(early,plane)),Nat.mul(channel,plane)),Nat.mul(row,late)),column) == Nat.add(Nat.add(_,Nat.mul(channel,plane)),rest) : Nat}
  Addition.associative(Nat.add(Nat.mul(image,Nat.mul(early,plane)),Nat.mul(channel,plane)),Nat.mul(row,late),column)

# A position of a band write counted in channel blocks, rows and columns is a
# row-major index.
def band_position(+channel: Nat,+row: Nat,+column: Nat,+rows: Nat,+late: Nat) ->
  {Nat.add(Nat.add(Nat.mul(channel,Nat.mul(rows,late)),Nat.mul(row,late)),column) == RowMajor.index(channel,rows,late,row,column) : Nat}:
  Addition.associative(Nat.mul(channel,Nat.mul(rows,late)),Nat.mul(row,late),column)

# The pixel of an image, a channel, a row and a column fits a capacity that
# holds every pixel.
def pixel_room(+image: Nat,+channel: Nat,+row: Nat,+column: Nat,+images: Nat,+channels: Nat,+height: Nat,+width: Nat,+capacity: U32,
  image_inside: {Nat.is_lt(image,images) == True{} : Bool},channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},
  row_inside: {Nat.is_lt(row,height) == True{} : Bool},column_inside: {Nat.is_lt(column,width) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(images,Nat.mul(channels,Nat.mul(height,width))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,row,column),U32.to_nat(capacity)) == True{} : Bool}:
  +index = RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,row,column)
  +total = Nat.mul(images,Nat.mul(channels,Nat.mul(height,width)))
  +below = {%RowMajor.assoc(images,channels,Nat.mul(height,width)) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.in_bounds(Nat.add(Nat.mul(image,channels),channel),Nat.mul(images,channels),height,width,row,column,
      RowMajor.radix(image,images,channel,channels,image_inside,channel_inside),row_inside,column_inside) : {Nat.is_lt(index,total) == True{} : Bool}}
  Order.transitive(index,total,U32.to_nat(capacity),Bounds.strict_le(index,total,below),room)

# A cell of a band fits a capacity that holds every cell of the band.
def cell_room(+channel: Nat,+row: Nat,+column: Nat,+channels: Nat,+rows: Nat,+width: Nat,+capacity: U32,
  channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},row_inside: {Nat.is_lt(row,rows) == True{} : Bool},column_inside: {Nat.is_lt(column,width) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(channels,Nat.mul(rows,width)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(RowMajor.index(channel,rows,width,row,column),U32.to_nat(capacity)) == True{} : Bool}:
  +index = RowMajor.index(channel,rows,width,row,column)
  +total = Nat.mul(channels,Nat.mul(rows,width))
  Order.transitive(index,total,U32.to_nat(capacity),Bounds.strict_le(index,total,RowMajor.in_bounds(channel,channels,rows,width,row,column,channel_inside,row_inside,column_inside)),
    room)

# The whole view of a banded form without repeats reads, at the position of an
# image, a channel, a row and a column, the pixel of that image and row at the
# channel and column the view reads.
def whole_plain(+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +image: Nat,+channel: Nat,+row: Nat,+column: Nat,+plane: ViewValues.Plane<batch,channels,height,width>,
  +image_inside: {Nat.is_lt(image,U32.to_nat(batch)) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,ShapeLaws.elements(Views.axis_extents(before))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(height)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ShapeLaws.elements(Views.axis_extents(after))) == True{} : Bool}) ->
  {Copy.address(Views.View{Classified.whole_axes(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width}),start},
      RowMajor.index(Nat.add(Nat.mul(image,ShapeLaws.elements(Views.axis_extents(before))),channel),U32.to_nat(height),ShapeLaws.elements(Views.axis_extents(after)),row,column)) ==
    U32.from_nat(RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),Address.at((height * width : U32),False{},before,after,start,channel,column)),U32.to_nat(height),
      U32.to_nat(width),row,Address.at((height * width : U32),True{},before,after,start,channel,column))) : U32}:
  ViewValues.Plane{+divisor,+plane_exact,+image_exact} = plane
  +view = {Views.View{Classified.whole_axes(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width}),start} : Views.View}
  +leading = Nat.add(Nat.mul(image,ShapeLaws.elements(Views.axis_extents(before))),channel)
  +late_size = ShapeLaws.elements(Views.axis_extents(after))
  +read = U32.from_nat(RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),Address.at((height * width : U32),False{},before,after,start,channel,column)),U32.to_nat(height),
    U32.to_nat(width),row,Address.at((height * width : U32),True{},before,after,start,channel,column)))
  %Order.times_one(U32.to_nat(height)) : {Copy.address(view,RowMajor.index(leading,_,late_size,row,column)) == read : U32}
  %Equal.trans(Nat,Nat.add(Nat.mul(row,1n),0n),Nat.mul(row,1n),row,Addition.zero_right(Nat.mul(row,1n)),Order.times_one(row)) :
    {Copy.address(view,RowMajor.index(leading,Nat.mul(U32.to_nat(height),1n),late_size,_,column)) == read : U32}
  Address.whole_address(before,[],after,start,batch,channels,height,width,image,channel,row,0n,column,divisor,plane_exact,image_exact,image_inside,channel_inside,row_inside,{==},
    column_inside)

# The local view of a band of a banded form without repeats reads, at the
# position of a channel, a row and a column, the cell of the band at the
# channel and column the view reads.
def band_plain(+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,+rows: U32,+capacity: U32,
  +channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,ShapeLaws.elements(Views.axis_extents(before))) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,ShapeLaws.elements(Views.axis_extents(after))) == True{} : Bool},
  +channel_reach: {Nat.is_lt(Address.reach_total((height * width : U32),False{},before,after,start),U32.to_nat(channels)) == True{} : Bool},
  +column_reach: {Nat.is_lt(Address.reach_total((height * width : U32),True{},before,after,start),U32.to_nat(width)) == True{} : Bool},
  +band_exact: {U32.to_nat((rows * width : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(width)) : Nat},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(channels),U32.to_nat((rows * width : U32))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Copy.address(BandViews.local(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width},rows),
      RowMajor.index(channel,U32.to_nat(rows),ShapeLaws.elements(Views.axis_extents(after)),row,column)) ==
    U32.from_nat(RowMajor.index(Address.at((height * width : U32),False{},before,after,start,channel,column),U32.to_nat(rows),U32.to_nat(width),row,
      Address.at((height * width : U32),True{},before,after,start,channel,column))) : U32}:
  +view = BandViews.local(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width},rows)
  +late_size = ShapeLaws.elements(Views.axis_extents(after))
  +read = U32.from_nat(RowMajor.index(Address.at((height * width : U32),False{},before,after,start,channel,column),U32.to_nat(rows),U32.to_nat(width),row,
    Address.at((height * width : U32),True{},before,after,start,channel,column)))
  %Order.times_one(U32.to_nat(rows)) : {Copy.address(view,RowMajor.index(channel,_,late_size,row,column)) == read : U32}
  %Equal.trans(Nat,Nat.add(Nat.mul(row,1n),0n),Nat.mul(row,1n),row,Addition.zero_right(Nat.mul(row,1n)),Order.times_one(row)) :
    {Copy.address(view,RowMajor.index(channel,Nat.mul(U32.to_nat(rows),1n),late_size,_,column)) == read : U32}
  Address.band_address(before,[],after,start,batch,channels,height,width,rows,capacity,channel,row,0n,column,channel_inside,row_inside,{==},column_inside,channel_reach,column_reach,
    band_exact,room)

# ---- One cell of a band ----

# A cell of a band of a write through a banded form without repeats: the form
# over its geometry, the band, the cell, and what the classification and the
# band establish.
type Cell is Data:
  Cell{before: List<&2,Views.Axis>,after: List<&2,Views.Axis>,start: U32,batch: U32,channels: U32,height: U32,width: U32,first: U32,rows: U32,capacity: U32,
    whole_capacity: U32,channel: Nat,row: Nat,column: Nat,fitted: ViewValues.Fitted<before,[],after,start,batch,channels,height,width>,
    plane: ViewValues.Plane<batch,channels,height,width>,
    whole_room: {Nat.is_le(BandGather.elements(Bands.Geometry{batch,channels,height,width}),U32.to_nat(whole_capacity)) == True{} : Bool},
    band_exact: {U32.to_nat((rows * width : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(width)) : Nat},
    band_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),U32.to_nat((rows * width : U32))),U32.to_nat(capacity)) == True{} : Bool},
    rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) == True{} : Bool},
    channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
    column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}}

def view_whole(cell: Cell) -> Views.View:
  Cell{+before,+after,+start,+batch,+channels,+height,+width,first,rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  Views.View{Classified.whole_axes(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width}),start}

def view_local(cell: Cell) -> Views.View:
  Cell{+before,+after,+start,+batch,+channels,+height,+width,first,+rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  BandViews.local(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width},rows)

# The geometry the part of the band arrives in.
def output_of(cell: Cell) -> Bands.Geometry:
  Cell{+before,+after,+start,+batch,+channels,+height,+width,first,rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  BandViews.output(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width})

def cell_first(cell: Cell) -> U32:
  Cell{before,after,start,batch,channels,height,width,+first,rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  first

def cell_rows(cell: Cell) -> U32:
  Cell{before,after,start,batch,channels,height,width,first,+rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  rows

# The elements of the axes before and after the rows, the height, the rows of
# the band and the images.
def early(cell: Cell) -> Nat:
  Cell{+before,after,start,batch,channels,height,width,first,rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  ShapeLaws.elements(Views.axis_extents(before))

def late(cell: Cell) -> Nat:
  Cell{before,+after,start,batch,channels,height,width,first,rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  ShapeLaws.elements(Views.axis_extents(after))

def tall(cell: Cell) -> Nat:
  Cell{before,after,start,batch,channels,+height,width,first,rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  U32.to_nat(height)

def band_rows(cell: Cell) -> Nat:
  Cell{before,after,start,batch,channels,height,width,first,+rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  U32.to_nat(rows)

def images(cell: Cell) -> Nat:
  Cell{before,after,start,+batch,channels,height,width,first,rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  U32.to_nat(batch)

# The cell: its channel, its row in the band and its column, and the image and
# the row in the image of its row.
def cell_channel(cell: Cell) -> Nat:
  Cell{before,after,start,batch,channels,height,width,first,rows,capacity,whole_capacity,+channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  channel

def cell_row(cell: Cell) -> Nat:
  Cell{before,after,start,batch,channels,height,width,first,rows,capacity,whole_capacity,channel,+row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  row

def cell_column(cell: Cell) -> Nat:
  Cell{before,after,start,batch,channels,height,width,first,rows,capacity,whole_capacity,channel,row,+column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  column

def cell_image(cell: Cell) -> Nat:
  Cell{before,after,start,batch,channels,+height,width,+first,rows,capacity,whole_capacity,channel,+row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  Nat.div(Nat.add(U32.to_nat(first),row),U32.to_nat(height))

def cell_inside(cell: Cell) -> Nat:
  Cell{before,after,start,batch,channels,+height,width,+first,rows,capacity,whole_capacity,channel,+row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  Nat.mod(Nat.add(U32.to_nat(first),row),U32.to_nat(height))

# The storage index of the cell in the target, and in its band.
def query_whole(cell: Cell) -> U32:
  Cell{before,after,start,batch,+channels,+height,+width,+first,rows,capacity,whole_capacity,+channel,+row,+column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  +global = Nat.add(U32.to_nat(first),row)
  U32.from_nat(RowMajor.index(Nat.add(Nat.mul(Nat.div(global,U32.to_nat(height)),U32.to_nat(channels)),channel),U32.to_nat(height),U32.to_nat(width),
    Nat.mod(global,U32.to_nat(height)),column))

def query_local(cell: Cell) -> U32:
  Cell{before,after,start,batch,channels,height,+width,first,+rows,capacity,whole_capacity,+channel,+row,+column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  U32.from_nat(RowMajor.index(channel,U32.to_nat(rows),U32.to_nat(width),row,column))

# The channel and the column the view reads at a position of a channel and a
# column of the part.
def at_channel(cell: Cell,+part_channel: Nat,+part_column: Nat) -> Nat:
  Cell{+before,+after,+start,batch,channels,+height,+width,first,rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  Address.at((height * width : U32),False{},before,after,start,part_channel,part_column)

def at_column(cell: Cell,+part_channel: Nat,+part_column: Nat) -> Nat:
  Cell{+before,+after,+start,batch,channels,+height,+width,first,rows,capacity,whole_capacity,channel,row,column,fitted,plane,whole_room,band_exact,band_room,rows_fit,
    channel_inside,row_inside,column_inside} = cell
  Address.at((height * width : U32),True{},before,after,start,part_channel,part_column)

def cell_image_inside(+cell: Cell) -> {Nat.is_lt(cell_image(cell),images(cell)) == True{} : Bool}:
  Cell{+before,+after,+start,+batch,+channels,+height,+width,+first,+rows,+capacity,+whole_capacity,+channel,+row,+column,+fitted,+plane,+whole_room,+band_exact,+band_room,
    +rows_fit,+channel_inside,+row_inside,+column_inside} = cell
  ViewValues.Fitted{batch_positive,+height_positive,channel_reach,column_reach,early_count,late_count,repeated,rows_exact} = fitted
  BandValues.image_below(U32.to_nat(first),U32.to_nat(rows),row,U32.to_nat(batch),U32.to_nat(height),height_positive,row_inside,rows_fit)

def cell_inside_below(+cell: Cell) -> {Nat.is_lt(cell_inside(cell),tall(cell)) == True{} : Bool}:
  Cell{+before,+after,+start,+batch,+channels,+height,+width,+first,+rows,+capacity,+whole_capacity,+channel,+row,+column,+fitted,+plane,+whole_room,+band_exact,+band_room,
    +rows_fit,+channel_inside,+row_inside,+column_inside} = cell
  ViewValues.Fitted{batch_positive,+height_positive,channel_reach,column_reach,early_count,late_count,repeated,rows_exact} = fitted
  DivisionOrder.remainder_small(Nat.add(U32.to_nat(first),row),U32.to_nat(height),height_positive)

def cell_row_inside(+cell: Cell) -> {Nat.is_lt(cell_row(cell),band_rows(cell)) == True{} : Bool}:
  Cell{+before,+after,+start,+batch,+channels,+height,+width,+first,+rows,+capacity,+whole_capacity,+channel,+row,+column,+fitted,+plane,+whole_room,+band_exact,+band_room,
    +rows_fit,+channel_inside,+row_inside,+column_inside} = cell
  row_inside

# ---- Where a write meets the cell ----

# A write of the whole view meets the cell exactly when it is in the image and
# the row of the cell and the view reads the channel and the column of the
# cell.
def whole_hit(+cell: Cell,+part_image: Nat,+part_channel: Nat,+part_row: Nat,+part_column: Nat,
  +image_below: {Nat.is_lt(part_image,images(cell)) == True{} : Bool},+channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},
  +row_below: {Nat.is_lt(part_row,tall(cell)) == True{} : Bool},+column_below: {Nat.is_lt(part_column,late(cell)) == True{} : Bool}) ->
  {U32.is_eq(Copy.address(view_whole(cell),RowMajor.index(Nat.add(Nat.mul(part_image,early(cell)),part_channel),tall(cell),late(cell),part_row,part_column)),query_whole(cell)) ==
    ((Nat.is_eq(part_image,cell_image(cell)) && Nat.is_eq(at_channel(cell,part_channel,part_column),cell_channel(cell))) &&
      (Nat.is_eq(part_row,cell_inside(cell)) && Nat.is_eq(at_column(cell,part_channel,part_column),cell_column(cell)))) : Bool}:
  Cell{+before,+after,+start,+batch,+channels,+height,+width,+first,+rows,+capacity,+whole_capacity,+channel,+row,+column,+fitted,+plane,+whole_room,+band_exact,+band_room,
    +rows_fit,+channel_inside,+row_inside,+column_inside} = cell
  ViewValues.Fitted{batch_positive,+height_positive,+channel_reach,+column_reach,early_count,late_count,repeated,rows_exact} = fitted
  +plane_word = (height * width : U32)
  +h = U32.to_nat(height)
  +w = U32.to_nat(width)
  +c = U32.to_nat(channels)
  +global = Nat.add(U32.to_nat(first),row)
  +image_value = Nat.div(global,h)
  +inside_value = Nat.mod(global,h)
  +at_channel_value = Address.at(plane_word,False{},before,after,start,part_channel,part_column)
  +at_column_value = Address.at(plane_word,True{},before,after,start,part_channel,part_column)
  +at_channel_inside = Order.le_lt_trans(at_channel_value,Address.reach_total(plane_word,False{},before,after,start),c,
    Address.at_within(plane_word,False{},before,after,start,part_channel,part_column,channel_below,column_below),channel_reach)
  +at_column_inside = Order.le_lt_trans(at_column_value,Address.reach_total(plane_word,True{},before,after,start),w,
    Address.at_within(plane_word,True{},before,after,start,part_channel,part_column,channel_below,column_below),column_reach)
  +read = RowMajor.index(Nat.add(Nat.mul(part_image,c),at_channel_value),h,w,part_row,at_column_value)
  +query = RowMajor.index(Nat.add(Nat.mul(image_value,c),channel),h,w,inside_value,column)
  +image_inside = BandValues.image_below(U32.to_nat(first),U32.to_nat(rows),row,U32.to_nat(batch),h,height_positive,row_inside,rows_fit)
  +inside_below = DivisionOrder.remainder_small(global,h,height_positive)
  +read_room = pixel_room(part_image,at_channel_value,part_row,at_column_value,U32.to_nat(batch),c,h,w,whole_capacity,image_below,at_channel_inside,row_below,at_column_inside,
    whole_room)
  +query_room = pixel_room(image_value,channel,inside_value,column,U32.to_nat(batch),c,h,w,whole_capacity,image_inside,channel_inside,inside_below,column_inside,whole_room)
  +met = ((Nat.is_eq(part_image,image_value) && Nat.is_eq(at_channel_value,channel)) && (Nat.is_eq(part_row,inside_value) && Nat.is_eq(at_column_value,column)))
  %Equal.sym(U32,Copy.address(Views.View{Classified.whole_axes(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width}),start},
      RowMajor.index(Nat.add(Nat.mul(part_image,ShapeLaws.elements(Views.axis_extents(before))),part_channel),h,ShapeLaws.elements(Views.axis_extents(after)),part_row,part_column)),
      U32.from_nat(read),
      whole_plain(before,after,start,batch,channels,height,width,part_image,part_channel,part_row,part_column,plane,image_below,channel_below,row_below,column_below)) :
    {U32.is_eq(_,U32.from_nat(query)) == met : Bool}
  Equal.trans(Bool,U32.is_eq(U32.from_nat(read),U32.from_nat(query)),Nat.is_eq(read,query),met,
    word_equal(read,query,whole_capacity,read_room,query_room),
    %radix_equal(part_image,at_channel_value,image_value,channel,c,at_channel_inside,channel_inside) :
      {Nat.is_eq(read,query) == (_ && (Nat.is_eq(part_row,inside_value) && Nat.is_eq(at_column_value,column))) : Bool}
    index_equal(Nat.add(Nat.mul(part_image,c),at_channel_value),part_row,at_column_value,Nat.add(Nat.mul(image_value,c),channel),inside_value,column,h,w,
      row_below,inside_below,at_column_inside,column_inside))

# A write of the local view of the band meets the cell exactly when it is in
# the row of the cell and the view reads the channel and the column of the
# cell.
def band_hit(+cell: Cell,+part_channel: Nat,+part_row: Nat,+part_column: Nat,
  +channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},+row_below: {Nat.is_lt(part_row,band_rows(cell)) == True{} : Bool},
  +column_below: {Nat.is_lt(part_column,late(cell)) == True{} : Bool}) ->
  {U32.is_eq(Copy.address(view_local(cell),RowMajor.index(part_channel,band_rows(cell),late(cell),part_row,part_column)),query_local(cell)) ==
    (Nat.is_eq(at_channel(cell,part_channel,part_column),cell_channel(cell)) &&
      (Nat.is_eq(part_row,cell_row(cell)) && Nat.is_eq(at_column(cell,part_channel,part_column),cell_column(cell)))) : Bool}:
  Cell{+before,+after,+start,+batch,+channels,+height,+width,+first,+rows,+capacity,+whole_capacity,+channel,+row,+column,+fitted,+plane,+whole_room,+band_exact,+band_room,
    +rows_fit,+channel_inside,+row_inside,+column_inside} = cell
  ViewValues.Fitted{batch_positive,height_positive,+channel_reach,+column_reach,early_count,late_count,repeated,rows_exact} = fitted
  +plane_word = (height * width : U32)
  +w = U32.to_nat(width)
  +c = U32.to_nat(channels)
  +r = U32.to_nat(rows)
  +at_channel_value = Address.at(plane_word,False{},before,after,start,part_channel,part_column)
  +at_column_value = Address.at(plane_word,True{},before,after,start,part_channel,part_column)
  +at_channel_inside = Order.le_lt_trans(at_channel_value,Address.reach_total(plane_word,False{},before,after,start),c,
    Address.at_within(plane_word,False{},before,after,start,part_channel,part_column,channel_below,column_below),channel_reach)
  +at_column_inside = Order.le_lt_trans(at_column_value,Address.reach_total(plane_word,True{},before,after,start),w,
    Address.at_within(plane_word,True{},before,after,start,part_channel,part_column,channel_below,column_below),column_reach)
  +read = RowMajor.index(at_channel_value,r,w,part_row,at_column_value)
  +query = RowMajor.index(channel,r,w,row,column)
  +cells_room = {%band_exact : {Nat.is_le(Nat.mul(c,_),U32.to_nat(capacity)) == True{} : Bool} band_room : {Nat.is_le(Nat.mul(c,Nat.mul(r,w)),U32.to_nat(capacity)) == True{} : Bool}}
  +read_room = cell_room(at_channel_value,part_row,at_column_value,c,r,w,capacity,at_channel_inside,row_below,at_column_inside,cells_room)
  +query_room = cell_room(channel,row,column,c,r,w,capacity,channel_inside,row_inside,column_inside,cells_room)
  +met = (Nat.is_eq(at_channel_value,channel) && (Nat.is_eq(part_row,row) && Nat.is_eq(at_column_value,column)))
  %Equal.sym(U32,Copy.address(BandViews.local(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width},rows),
      RowMajor.index(part_channel,r,ShapeLaws.elements(Views.axis_extents(after)),part_row,part_column)),U32.from_nat(read),
      band_plain(before,after,start,batch,channels,height,width,rows,capacity,part_channel,part_row,part_column,channel_below,row_below,column_below,channel_reach,column_reach,
        band_exact,band_room)) :
    {U32.is_eq(_,U32.from_nat(query)) == met : Bool}
  Equal.trans(Bool,U32.is_eq(U32.from_nat(read),U32.from_nat(query)),Nat.is_eq(read,query),met,
    word_equal(read,query,capacity,read_room,query_room),
    index_equal(at_channel_value,part_row,at_column_value,channel,row,column,r,w,row_below,row_inside,at_column_inside,column_inside))

# A write of another image misses the cell.
def other_image_miss(+cell: Cell,+part_image: Nat,+part_channel: Nat,+part_row: Nat,+part_column: Nat,
  +image_below: {Nat.is_lt(part_image,images(cell)) == True{} : Bool},+channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},
  +row_below: {Nat.is_lt(part_row,tall(cell)) == True{} : Bool},+column_below: {Nat.is_lt(part_column,late(cell)) == True{} : Bool},
  +other: {Nat.is_eq(part_image,cell_image(cell)) == False{} : Bool}) ->
  {U32.is_eq(Copy.address(view_whole(cell),Nat.add(Nat.add(Nat.add(Nat.add(0n,Nat.mul(part_image,Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),
      Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),Nat.mul(part_row,late(cell))),part_column)),query_whole(cell)) == False{} : Bool}:
  +position = Nat.add(Nat.add(Nat.add(Nat.mul(part_image,Nat.mul(early(cell),Nat.mul(tall(cell),late(cell)))),Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),
    Nat.mul(part_row,late(cell))),part_column)
  +index = RowMajor.index(Nat.add(Nat.mul(part_image,early(cell)),part_channel),tall(cell),late(cell),part_row,part_column)
  +channel_met = Nat.is_eq(at_channel(cell,part_channel,part_column),cell_channel(cell))
  +rest = (Nat.is_eq(part_row,cell_inside(cell)) && Nat.is_eq(at_column(cell,part_channel,part_column),cell_column(cell)))
  %Equal.sym(Nat,position,index,whole_position(part_image,part_channel,part_row,part_column,early(cell),tall(cell),late(cell))) :
    {U32.is_eq(Copy.address(view_whole(cell),_),query_whole(cell)) == False{} : Bool}
  Equal.trans(Bool,U32.is_eq(Copy.address(view_whole(cell),index),query_whole(cell)),((Nat.is_eq(part_image,cell_image(cell)) && channel_met) && rest),False{},
    whole_hit(cell,part_image,part_channel,part_row,part_column,image_below,channel_below,row_below,column_below),
    %Equal.sym(Bool,Nat.is_eq(part_image,cell_image(cell)),False{},other) : {((_ && channel_met) && rest) == False{} : Bool}
    {==})

# A write of the image of the cell in another row misses the cell.
def row_miss_whole(+cell: Cell,+part_channel: Nat,+part_row: Nat,+part_column: Nat,
  +channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},+row_below: {Nat.is_lt(part_row,tall(cell)) == True{} : Bool},
  +column_below: {Nat.is_lt(part_column,late(cell)) == True{} : Bool},+other: {Nat.is_eq(part_row,cell_inside(cell)) == False{} : Bool}) ->
  {U32.is_eq(Copy.address(view_whole(cell),Nat.add(Nat.add(Nat.add(Nat.add(0n,Nat.mul(cell_image(cell),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),
      Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),Nat.mul(part_row,late(cell))),part_column)),query_whole(cell)) == False{} : Bool}:
  +position = Nat.add(Nat.add(Nat.add(Nat.mul(cell_image(cell),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell)))),Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),
    Nat.mul(part_row,late(cell))),part_column)
  +index = RowMajor.index(Nat.add(Nat.mul(cell_image(cell),early(cell)),part_channel),tall(cell),late(cell),part_row,part_column)
  +head = (Nat.is_eq(cell_image(cell),cell_image(cell)) && Nat.is_eq(at_channel(cell,part_channel,part_column),cell_channel(cell)))
  +column_met = Nat.is_eq(at_column(cell,part_channel,part_column),cell_column(cell))
  %Equal.sym(Nat,position,index,whole_position(cell_image(cell),part_channel,part_row,part_column,early(cell),tall(cell),late(cell))) :
    {U32.is_eq(Copy.address(view_whole(cell),_),query_whole(cell)) == False{} : Bool}
  Equal.trans(Bool,U32.is_eq(Copy.address(view_whole(cell),index),query_whole(cell)),(head && (Nat.is_eq(part_row,cell_inside(cell)) && column_met)),False{},
    whole_hit(cell,cell_image(cell),part_channel,part_row,part_column,cell_image_inside(cell),channel_below,row_below,column_below),
    %Equal.sym(Bool,Nat.is_eq(part_row,cell_inside(cell)),False{},other) : {(head && (_ && column_met)) == False{} : Bool}
    Boolean.and_false_right(head))

# A write of the band in another row misses the cell.
def row_miss_local(+cell: Cell,+part_channel: Nat,+part_row: Nat,+part_column: Nat,
  +channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},+row_below: {Nat.is_lt(part_row,band_rows(cell)) == True{} : Bool},
  +column_below: {Nat.is_lt(part_column,late(cell)) == True{} : Bool},+other: {Nat.is_eq(part_row,cell_row(cell)) == False{} : Bool}) ->
  {U32.is_eq(Copy.address(view_local(cell),Nat.add(Nat.add(Nat.add(0n,Nat.mul(part_channel,Nat.mul(band_rows(cell),late(cell)))),Nat.mul(part_row,late(cell))),part_column)),
    query_local(cell)) == False{} : Bool}:
  +position = Nat.add(Nat.add(Nat.mul(part_channel,Nat.mul(band_rows(cell),late(cell))),Nat.mul(part_row,late(cell))),part_column)
  +index = RowMajor.index(part_channel,band_rows(cell),late(cell),part_row,part_column)
  +head = Nat.is_eq(at_channel(cell,part_channel,part_column),cell_channel(cell))
  +column_met = Nat.is_eq(at_column(cell,part_channel,part_column),cell_column(cell))
  %Equal.sym(Nat,position,index,band_position(part_channel,part_row,part_column,band_rows(cell),late(cell))) :
    {U32.is_eq(Copy.address(view_local(cell),_),query_local(cell)) == False{} : Bool}
  Equal.trans(Bool,U32.is_eq(Copy.address(view_local(cell),index),query_local(cell)),(head && (Nat.is_eq(part_row,cell_row(cell)) && column_met)),False{},
    band_hit(cell,part_channel,part_row,part_column,channel_below,row_below,column_below),
    %Equal.sym(Bool,Nat.is_eq(part_row,cell_row(cell)),False{},other) : {(head && (_ && column_met)) == False{} : Bool}
    Boolean.and_false_right(head))

# In the row of the cell, the whole write and the band write meet the cell at
# the same columns.
def run_hits(+cell: Cell,+part_channel: Nat,+part_column: Nat,
  +channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},+column_below: {Nat.is_lt(part_column,late(cell)) == True{} : Bool}) ->
  {U32.is_eq(Copy.address(view_whole(cell),Nat.add(Nat.add(Nat.add(Nat.add(0n,Nat.mul(cell_image(cell),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),
      Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),Nat.mul(cell_inside(cell),late(cell))),part_column)),query_whole(cell)) ==
    U32.is_eq(Copy.address(view_local(cell),Nat.add(Nat.add(Nat.add(0n,Nat.mul(part_channel,Nat.mul(band_rows(cell),late(cell)))),Nat.mul(cell_row(cell),late(cell))),part_column)),
      query_local(cell)) : Bool}:
  +channel_met = Nat.is_eq(at_channel(cell,part_channel,part_column),cell_channel(cell))
  +column_met = Nat.is_eq(at_column(cell,part_channel,part_column),cell_column(cell))
  +whole_place = Nat.add(Nat.add(Nat.add(Nat.mul(cell_image(cell),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell)))),Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),
    Nat.mul(cell_inside(cell),late(cell))),part_column)
  +whole_index = RowMajor.index(Nat.add(Nat.mul(cell_image(cell),early(cell)),part_channel),tall(cell),late(cell),cell_inside(cell),part_column)
  +band_place = Nat.add(Nat.add(Nat.mul(part_channel,Nat.mul(band_rows(cell),late(cell))),Nat.mul(cell_row(cell),late(cell))),part_column)
  +band_index = RowMajor.index(part_channel,band_rows(cell),late(cell),cell_row(cell),part_column)
  +whole_met = {
    %Equal.sym(Nat,whole_place,whole_index,whole_position(cell_image(cell),part_channel,cell_inside(cell),part_column,early(cell),tall(cell),late(cell))) :
      {U32.is_eq(Copy.address(view_whole(cell),_),query_whole(cell)) == (channel_met && column_met) : Bool}
    %Order.equal_reflexive(cell_image(cell)) :
      {U32.is_eq(Copy.address(view_whole(cell),whole_index),query_whole(cell)) == ((_ && channel_met) && (True{} && column_met)) : Bool}
    %Order.equal_reflexive(cell_inside(cell)) :
      {U32.is_eq(Copy.address(view_whole(cell),whole_index),query_whole(cell)) == ((Nat.is_eq(cell_image(cell),cell_image(cell)) && channel_met) && (_ && column_met)) : Bool}
    whole_hit(cell,cell_image(cell),part_channel,cell_inside(cell),part_column,cell_image_inside(cell),channel_below,cell_inside_below(cell),column_below)
    : {U32.is_eq(Copy.address(view_whole(cell),whole_place),query_whole(cell)) == (channel_met && column_met) : Bool}}
  +band_met = {
    %Equal.sym(Nat,band_place,band_index,band_position(part_channel,cell_row(cell),part_column,band_rows(cell),late(cell))) :
      {U32.is_eq(Copy.address(view_local(cell),_),query_local(cell)) == (channel_met && column_met) : Bool}
    %Order.equal_reflexive(cell_row(cell)) :
      {U32.is_eq(Copy.address(view_local(cell),band_index),query_local(cell)) == (channel_met && (_ && column_met)) : Bool}
    band_hit(cell,part_channel,cell_row(cell),part_column,channel_below,cell_row_inside(cell),column_below)
    : {U32.is_eq(Copy.address(view_local(cell),band_place),query_local(cell)) == (channel_met && column_met) : Bool}}
  Equal.trans(Bool,U32.is_eq(Copy.address(view_whole(cell),whole_place),query_whole(cell)),(channel_met && column_met),
    U32.is_eq(Copy.address(view_local(cell),band_place),query_local(cell)),whole_met,
    Equal.sym(Bool,U32.is_eq(Copy.address(view_local(cell),band_place),query_local(cell)),(channel_met && column_met),band_met))

# In the row of the cell, the whole write and the band write carry the same
# elements: the part band holds the part at the pixel the whole write reads.
def run_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+cell: Cell,+logical: Logical,+tree: Tree.Tree<Element>,+length: Nat,
  +part_values: BandValues.Leaf(~Element,~Logical,~original,logical,output_of(cell),cell_first(cell),cell_rows(cell),tree,length),
  +part_channel: Nat,+part_column: Nat,+channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},+column_below: {Nat.is_lt(part_column,late(cell)) == True{} : Bool}) ->
  {Scatter.logical(~Element,~Logical,~original,logical,Nat.add(Nat.add(Nat.add(Nat.add(0n,Nat.mul(cell_image(cell),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),
      Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),Nat.mul(cell_inside(cell),late(cell))),part_column)) ==
    Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),Buffer.Content{tree,length},
      Nat.add(Nat.add(Nat.add(0n,Nat.mul(part_channel,Nat.mul(band_rows(cell),late(cell)))),Nat.mul(cell_row(cell),late(cell))),part_column)) : Element}:
  Cell{+before,+after,+start,+batch,+channels,+height,+width,+first,+rows,+capacity,+whole_capacity,+channel,+row,+column,+fitted,+plane,+whole_room,+band_exact,+band_room,
    +rows_fit,+channel_inside,+row_inside,+column_inside} = cell
  ViewValues.Fitted{batch_positive,height_positive,channel_reach,column_reach,+early_count,+late_count,repeated,rows_exact} = fitted
  +h = U32.to_nat(height)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +ea = ShapeLaws.elements(Views.axis_extents(after))
  +r = U32.to_nat(rows)
  +global = Nat.add(U32.to_nat(first),row)
  +image_value = Nat.div(global,h)
  +inside_value = Nat.mod(global,h)
  +before_word = BandViews.product_or_zero(Views.axis_extents(before))
  +after_word = BandViews.product_or_zero(Views.axis_extents(after))
  +output = BandViews.output(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width})
  +whole_index = RowMajor.index(Nat.add(Nat.mul(image_value,eb),part_channel),h,ea,inside_value,part_column)
  +band_index = RowMajor.index(part_channel,r,ea,row,part_column)
  +channel_out = {%Equal.sym(Nat,U32.to_nat(before_word),eb,early_count) : {Nat.is_lt(part_channel,_) == True{} : Bool}
    channel_below : {Nat.is_lt(part_channel,U32.to_nat(before_word)) == True{} : Bool}}
  +column_out = {%Equal.sym(Nat,U32.to_nat(after_word),ea,late_count) : {Nat.is_lt(part_column,_) == True{} : Bool}
    column_below : {Nat.is_lt(part_column,U32.to_nat(after_word)) == True{} : Bool}}
  +read = {
    %early_count :
      {Tree.read(Element,tree,U32.from_nat(band_index)) == original(logical,U32.from_nat(RowMajor.index(Nat.add(Nat.mul(image_value,_),part_channel),h,ea,inside_value,part_column))) : Element}
    %late_count :
      {Tree.read(Element,tree,U32.from_nat(RowMajor.index(part_channel,r,_,row,part_column))) ==
        original(logical,U32.from_nat(RowMajor.index(Nat.add(Nat.mul(image_value,U32.to_nat(before_word)),part_channel),h,_,inside_value,part_column))) : Element}
    %Arithmetic.multiply_one(height) :
      {Tree.read(Element,tree,U32.from_nat(RowMajor.index(part_channel,r,U32.to_nat(after_word),row,part_column))) ==
        original(logical,U32.from_nat(RowMajor.index(Nat.add(Nat.mul(Nat.div(global,U32.to_nat(_)),U32.to_nat(before_word)),part_channel),U32.to_nat(_),U32.to_nat(after_word),
          Nat.mod(global,U32.to_nat(_)),part_column))) : Element}
    BandValues.leaf_read(~Element,~Logical,~original,logical,output,first,rows,tree,length,part_channel,row,part_column,part_values,channel_out,row_inside,column_out)
    : {Tree.read(Element,tree,U32.from_nat(band_index)) == original(logical,U32.from_nat(whole_index)) : Element}}
  +whole_place = Nat.add(Nat.add(Nat.add(Nat.mul(image_value,Nat.mul(eb,Nat.mul(h,ea))),Nat.mul(part_channel,Nat.mul(h,ea))),Nat.mul(inside_value,ea)),part_column)
  +band_place = Nat.add(Nat.add(Nat.mul(part_channel,Nat.mul(r,ea)),Nat.mul(row,ea)),part_column)
  %Equal.sym(Nat,whole_place,whole_index,whole_position(image_value,part_channel,inside_value,part_column,eb,h,ea)) :
    {original(logical,U32.from_nat(_)) == Tree.read(Element,tree,U32.from_nat(band_place)) : Element}
  %Equal.sym(Nat,band_place,band_index,band_position(part_channel,row,part_column,r,ea)) :
    {original(logical,U32.from_nat(whole_index)) == Tree.read(Element,tree,U32.from_nat(_)) : Element}
  Equal.sym(Element,Tree.read(Element,tree,U32.from_nat(band_index)),original(logical,U32.from_nat(whole_index)),read)

# ---- The cell of a band reads the whole write ----

# Every row of a channel block of another image keeps every element.
def row_kept(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+cell: Cell,+logical: Logical,+part_image: Nat,
  +image_below: {Nat.is_lt(part_image,images(cell)) == True{} : Bool},+other: {Nat.is_eq(part_image,cell_image(cell)) == False{} : Bool},
  +part_channel: Nat,+channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},+part_row: Nat,+row_below: {Nat.is_lt(part_row,tall(cell)) == True{} : Bool},
  +element: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),late(cell),
      Nat.add(Nat.add(Nat.add(0n,Nat.mul(part_image,Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),
        Nat.mul(part_row,late(cell))),view_whole(cell),logical),query_whole(cell),element) == element : Element}:
  +base = Nat.add(Nat.add(Nat.add(0n,Nat.mul(part_image,Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),
    Nat.mul(part_row,late(cell)))
  missed(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),~Unit,~(context => view_whole(cell)),~(context => logical),~(context => query_whole(cell)),
    ~(context => base),~(context => late(cell)),
    ~(context => part_column => column_below => other_image_miss(cell,part_image,part_channel,part_row,part_column,image_below,channel_below,row_below,column_below,other)),
    Unit{},element)

# Every channel block of another image keeps every element.
def channel_kept(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+cell: Cell,+logical: Logical,+part_image: Nat,
  +image_below: {Nat.is_lt(part_image,images(cell)) == True{} : Bool},+other: {Nat.is_eq(part_image,cell_image(cell)) == False{} : Bool},
  +part_channel: Nat,+channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},+element: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),Nat.mul(tall(cell),late(cell)),
      Nat.add(Nat.add(0n,Nat.mul(part_image,Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),view_whole(cell),logical),
      query_whole(cell),element) == element : Element}:
  +block = Nat.add(Nat.add(0n,Nat.mul(part_image,Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),Nat.mul(part_channel,Nat.mul(tall(cell),late(cell))))
  Equal.trans(Element,
    Readback.selected(~Element,ViewWrites.writes_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),Nat.mul(tall(cell),late(cell)),block,view_whole(cell),logical),
      query_whole(cell),element),
    through(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),tall(cell),late(cell),block,view_whole(cell),logical,query_whole(cell),element),element,
    folded(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),tall(cell),late(cell),block,view_whole(cell),logical,query_whole(cell),element),
    through_kept(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),~Unit,~(context => late(cell)),~(context => block),~(context => view_whole(cell)),
      ~(context => logical),~(context => query_whole(cell)),~(context => tall(cell)),
      ~(context => part_row => row_below => kept =>
        row_kept(~Element,~Logical,~original,cell,logical,part_image,image_below,other,part_channel,channel_below,part_row,row_below,kept)),
      Unit{},tall(cell),Order.reflexive(tall(cell)),element))

# Another image keeps every element.
def image_kept(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+cell: Cell,+logical: Logical,+part_image: Nat,
  +image_below: {Nat.is_lt(part_image,images(cell)) == True{} : Bool},+other: {Nat.is_eq(part_image,cell_image(cell)) == False{} : Bool},+element: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))),
      Nat.add(0n,Nat.mul(part_image,Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),view_whole(cell),logical),query_whole(cell),element) == element : Element}:
  +block = Nat.add(0n,Nat.mul(part_image,Nat.mul(early(cell),Nat.mul(tall(cell),late(cell)))))
  Equal.trans(Element,
    Readback.selected(~Element,ViewWrites.writes_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))),block,
      view_whole(cell),logical),query_whole(cell),element),
    through(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),early(cell),Nat.mul(tall(cell),late(cell)),block,view_whole(cell),logical,query_whole(cell),element),
    element,
    folded(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),early(cell),Nat.mul(tall(cell),late(cell)),block,view_whole(cell),logical,query_whole(cell),element),
    through_kept(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),~Unit,~(context => Nat.mul(tall(cell),late(cell))),~(context => block),~(context => view_whole(cell)),
      ~(context => logical),~(context => query_whole(cell)),~(context => early(cell)),
      ~(context => part_channel => channel_below => kept => channel_kept(~Element,~Logical,~original,cell,logical,part_image,image_below,other,part_channel,channel_below,kept)),
      Unit{},early(cell),Order.reflexive(early(cell)),element))

# Another row of the channel block of the image of the cell keeps every
# element.
def row_skip_whole(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+cell: Cell,+logical: Logical,
  +part_channel: Nat,+channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},+part_row: Nat,+row_below: {Nat.is_lt(part_row,tall(cell)) == True{} : Bool},
  +other: {Nat.is_eq(part_row,cell_inside(cell)) == False{} : Bool},+element: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),late(cell),
      Nat.add(Nat.add(Nat.add(0n,Nat.mul(cell_image(cell),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),
        Nat.mul(part_row,late(cell))),view_whole(cell),logical),query_whole(cell),element) == element : Element}:
  +base = Nat.add(Nat.add(Nat.add(0n,Nat.mul(cell_image(cell),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),
    Nat.mul(part_row,late(cell)))
  missed(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),~Unit,~(context => view_whole(cell)),~(context => logical),~(context => query_whole(cell)),
    ~(context => base),~(context => late(cell)),
    ~(context => part_column => column_below => row_miss_whole(cell,part_channel,part_row,part_column,channel_below,row_below,column_below,other)),
    Unit{},element)

# Another row of the channel block of the band keeps every element.
def row_skip_local(~Element: Data,+cell: Cell,+tree: Tree.Tree<Element>,+length: Nat,
  +part_channel: Nat,+channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},+part_row: Nat,+row_below: {Nat.is_lt(part_row,band_rows(cell)) == True{} : Bool},
  +other: {Nat.is_eq(part_row,cell_row(cell)) == False{} : Bool},+element: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),late(cell),
      Nat.add(Nat.add(0n,Nat.mul(part_channel,Nat.mul(band_rows(cell),late(cell)))),Nat.mul(part_row,late(cell))),view_local(cell),Buffer.Content{tree,length}),
      query_local(cell),element) == element : Element}:
  +content = {Buffer.Content{tree,length} : Buffer.Content<Element>}
  +base = Nat.add(Nat.add(0n,Nat.mul(part_channel,Nat.mul(band_rows(cell),late(cell)))),Nat.mul(part_row,late(cell)))
  missed(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),~Unit,~(context => view_local(cell)),~(context => content),
    ~(context => query_local(cell)),~(context => base),~(context => late(cell)),
    ~(context => part_column => column_below => row_miss_local(cell,part_channel,part_row,part_column,channel_below,row_below,column_below,other)),
    Unit{},element)

# A channel block of the image of the cell and the same channel block of the
# band read the same: both reach the cell only in its row, where they meet it
# together with the same elements.
def channel_same(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+cell: Cell,+logical: Logical,+tree: Tree.Tree<Element>,+length: Nat,
  +part_values: BandValues.Leaf(~Element,~Logical,~original,logical,output_of(cell),cell_first(cell),cell_rows(cell),tree,length),
  +part_channel: Nat,+channel_below: {Nat.is_lt(part_channel,early(cell)) == True{} : Bool},+element: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),Nat.mul(tall(cell),late(cell)),
      Nat.add(Nat.add(0n,Nat.mul(cell_image(cell),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),Nat.mul(part_channel,Nat.mul(tall(cell),late(cell)))),view_whole(cell),logical),
      query_whole(cell),element) ==
    Readback.selected(~Element,ViewWrites.writes_from(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
      Nat.mul(band_rows(cell),late(cell)),Nat.add(0n,Nat.mul(part_channel,Nat.mul(band_rows(cell),late(cell)))),view_local(cell),Buffer.Content{tree,length}),
      query_local(cell),element) : Element}:
  +content = {Buffer.Content{tree,length} : Buffer.Content<Element>}
  +block = Nat.add(Nat.add(0n,Nat.mul(cell_image(cell),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell))))),Nat.mul(part_channel,Nat.mul(tall(cell),late(cell))))
  +band_block = Nat.add(0n,Nat.mul(part_channel,Nat.mul(band_rows(cell),late(cell))))
  +whole_run = Nat.add(block,Nat.mul(cell_inside(cell),late(cell)))
  +band_run = Nat.add(band_block,Nat.mul(cell_row(cell),late(cell)))
  +whole_selected = Readback.selected(~Element,ViewWrites.writes_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),Nat.mul(tall(cell),late(cell)),block,
    view_whole(cell),logical),query_whole(cell),element)
  +whole_through = through(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),tall(cell),late(cell),block,view_whole(cell),logical,query_whole(cell),element)
  +whole_row = Readback.selected(~Element,ViewWrites.writes_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),late(cell),whole_run,view_whole(cell),logical),
    query_whole(cell),element)
  +band_row = Readback.selected(~Element,ViewWrites.writes_from(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),late(cell),
    band_run,view_local(cell),content),query_local(cell),element)
  +band_through = through(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),band_rows(cell),late(cell),band_block,
    view_local(cell),content,query_local(cell),element)
  +band_selected = Readback.selected(~Element,ViewWrites.writes_from(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
    Nat.mul(band_rows(cell),late(cell)),band_block,view_local(cell),content),query_local(cell),element)
  +whole_chosen = through_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),~Unit,~(context => late(cell)),~(context => block),~(context => view_whole(cell)),
    ~(context => logical),~(context => query_whole(cell)),~(context => tall(cell)),~(context => cell_inside(cell)),
    ~(context => part_row => row_below => other => kept => row_skip_whole(~Element,~Logical,~original,cell,logical,part_channel,channel_below,part_row,row_below,other,kept)),
    Unit{},tall(cell),cell_inside_below(cell),Order.reflexive(tall(cell)),element)
  +band_chosen = through_from(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),~Unit,~(context => late(cell)),
    ~(context => band_block),~(context => view_local(cell)),~(context => content),~(context => query_local(cell)),~(context => band_rows(cell)),~(context => cell_row(cell)),
    ~(context => part_row => row_below => other => kept => row_skip_local(~Element,cell,tree,length,part_channel,channel_below,part_row,row_below,other,kept)),
    Unit{},band_rows(cell),cell_row_inside(cell),Order.reflexive(band_rows(cell)),element)
  +run = parallel(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
    ~Unit,~(context => view_whole(cell)),~(context => logical),~(context => query_whole(cell)),~(context => whole_run),~(context => view_local(cell)),~(context => content),
    ~(context => query_local(cell)),~(context => band_run),~(context => late(cell)),
    ~(context => part_column => column_below => run_hits(cell,part_channel,part_column,channel_below,column_below)),
    ~(context => part_column => column_below => run_values(~Element,~Logical,~original,cell,logical,tree,length,part_values,part_channel,part_column,channel_below,column_below)),
    Unit{},element)
  Equal.trans(Element,whole_selected,whole_through,band_selected,
    folded(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),tall(cell),late(cell),block,view_whole(cell),logical,query_whole(cell),element),
    Equal.trans(Element,whole_through,whole_row,band_selected,whole_chosen,
      Equal.trans(Element,whole_row,band_row,band_selected,run,
        Equal.trans(Element,band_row,band_through,band_selected,Equal.sym(Element,band_through,band_row,band_chosen),
          Equal.sym(Element,band_selected,band_through,
            folded(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),band_rows(cell),late(cell),band_block,
              view_local(cell),content,query_local(cell),element))))))

# The whole write read at the storage index of the cell in the target is the
# write of the band through its local view read at the storage index of the
# cell in the band: only the image of the cell reaches it, and there every
# channel block reads as the channel block of the band.
def band_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+cell: Cell,+logical: Logical,+tree: Tree.Tree<Element>,+length: Nat,
  +part_values: BandValues.Leaf(~Element,~Logical,~original,logical,output_of(cell),cell_first(cell),cell_rows(cell),tree,length),+previous: Element) ->
  {Readback.selected(~Element,ViewWrites.writes(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),
      Nat.mul(images(cell),Nat.mul(early(cell),Nat.mul(tall(cell),late(cell)))),view_whole(cell),logical),query_whole(cell),previous) ==
    Readback.selected(~Element,ViewWrites.writes(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
      Nat.mul(early(cell),Nat.mul(band_rows(cell),late(cell))),view_local(cell),Buffer.Content{tree,length}),query_local(cell),previous) : Element}:
  +content = {Buffer.Content{tree,length} : Buffer.Content<Element>}
  +image_size = Nat.mul(early(cell),Nat.mul(tall(cell),late(cell)))
  +block = Nat.add(0n,Nat.mul(cell_image(cell),image_size))
  +whole_selected = Readback.selected(~Element,ViewWrites.writes_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),Nat.mul(images(cell),image_size),0n,
    view_whole(cell),logical),query_whole(cell),previous)
  +whole_images = through(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),images(cell),image_size,0n,view_whole(cell),logical,query_whole(cell),previous)
  +image_selected = Readback.selected(~Element,ViewWrites.writes_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),image_size,block,view_whole(cell),logical),
    query_whole(cell),previous)
  +whole_channels = through(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),early(cell),Nat.mul(tall(cell),late(cell)),block,view_whole(cell),logical,
    query_whole(cell),previous)
  +band_channels = through(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),early(cell),
    Nat.mul(band_rows(cell),late(cell)),0n,view_local(cell),content,query_local(cell),previous)
  +band_selected = Readback.selected(~Element,ViewWrites.writes_from(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
    Nat.mul(early(cell),Nat.mul(band_rows(cell),late(cell))),0n,view_local(cell),content),query_local(cell),previous)
  +chosen = through_from(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),~Unit,~(context => image_size),~(context => 0n),~(context => view_whole(cell)),
    ~(context => logical),~(context => query_whole(cell)),~(context => images(cell)),~(context => cell_image(cell)),
    ~(context => part_image => image_below => other => kept => image_kept(~Element,~Logical,~original,cell,logical,part_image,image_below,other,kept)),
    Unit{},images(cell),cell_image_inside(cell),Order.reflexive(images(cell)),previous)
  +channels = through_parallel(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),~Buffer.Content<Element>,
    ~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),~Unit,~(context => Nat.mul(tall(cell),late(cell))),~(context => block),~(context => view_whole(cell)),
    ~(context => logical),~(context => query_whole(cell)),~(context => Nat.mul(band_rows(cell),late(cell))),~(context => 0n),~(context => view_local(cell)),~(context => content),
    ~(context => query_local(cell)),~(context => early(cell)),
    ~(context => part_channel => channel_below => kept => channel_same(~Element,~Logical,~original,cell,logical,tree,length,part_values,part_channel,channel_below,kept)),
    Unit{},early(cell),Order.reflexive(early(cell)),previous)
  Equal.trans(Element,whole_selected,whole_images,band_selected,
    folded(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),images(cell),image_size,0n,view_whole(cell),logical,query_whole(cell),previous),
    Equal.trans(Element,whole_images,image_selected,band_selected,chosen,
      Equal.trans(Element,image_selected,whole_channels,band_selected,
        folded(~Element,~Logical,~Scatter.logical(~Element,~Logical,~original),early(cell),Nat.mul(tall(cell),late(cell)),block,view_whole(cell),logical,query_whole(cell),previous),
        Equal.trans(Element,whole_channels,band_channels,band_selected,channels,
          Equal.sym(Element,band_selected,band_channels,
            folded(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),early(cell),Nat.mul(band_rows(cell),late(cell)),0n,
              view_local(cell),content,query_local(cell),previous))))))

# ---- Every band of the target ----

# What every band of one write shares: the target and the part, the view and
# its banded form without repeats over the target geometry, and what its
# classification established.
type Scope<-Before: Data,-Part: Data> is Data:
  Scope{target: Before,part: Part,view_axes: List<&2,Views.Axis>,offset: U32,before: List<&2,Views.Axis>,after: List<&2,Views.Axis>,start: U32,geometry: Bands.Geometry,
    limits: Machine.Limits,words: ViewValues.Words(BandViews.Banded{before,[],after,start},geometry),
    classified: Classified.Classified<view_axes,offset,geometry,BandViews.Banded{before,[],after,start}>,
    room: BandGather.Domain<geometry,limits,Machine.capacity(limits)>}

def scope_context(~Before: Data,~Part: Data,+scope: Scope<Before,Part>) -> Layout.Placing:
  Scope{target,part,view_axes,offset,+before,+after,+start,+geometry,limits,words,classified,room} = scope
  Layout.Placing{BandViews.Banded{before,[],after,start},geometry}

def scope_geometry(~Before: Data,~Part: Data,+scope: Scope<Before,Part>) -> Bands.Geometry:
  Scope{target,part,view_axes,offset,before,after,start,+geometry,limits,words,classified,room} = scope
  geometry

# The reader of the result: the whole part written through the view over the
# target, as many writes as the images, the channel blocks, the rows and the
# columns of the banded form.
def written(~Before: Data,~Part: Data,+target: Before,+part: Part,+view: Views.View,+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+geometry: Bands.Geometry) ->
  Scatter.Scatter<Before,Part>:
  Bands.Geometry{+batch,channels,+height,width} = geometry
  Scatter.Scatter{target,part,view,
    Nat.mul(U32.to_nat(batch),Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(height),ShapeLaws.elements(Views.axis_extents(after)))))}

def scope_written(~Before: Data,~Part: Data,+scope: Scope<Before,Part>) -> Scatter.Scatter<Before,Part>:
  Scope{+target,+part,+view_axes,+offset,+before,+after,start,+geometry,limits,words,classified,room} = scope
  written(~Before,~Part,target,part,Views.View{view_axes,offset},before,after,geometry)

# The cells of a band of rows rows, and the extents its scatter writes.
def band_cells(~Before: Data,~Part: Data,+scope: Scope<Before,Part>,+rows: U32) -> Nat:
  Scope{target,part,view_axes,offset,+before,+after,start,geometry,limits,words,classified,room} = scope
  Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(rows),ShapeLaws.elements(Views.axis_extents(after))))

def band_shape(~Before: Data,~Part: Data,+scope: Scope<Before,Part>,+rows: U32) -> List<&2,U32>:
  Scope{target,part,view_axes,offset,before,after,start,+geometry,limits,words,classified,room} = scope
  Bands.Geometry{batch,+channels,height,+width} = geometry
  [channels,rows,width]

# The facts of one band of the target and of the part: their fields, their
# buffers are ready, and their values.
def target_fields(~Element: Data,~Before: Data,~Part: Data,+scope: Scope<Before,Part>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Scope{target,part,view_axes,offset,before,after,start,+geometry,limits,words,classified,room} = scope
  BandShape.band_fields(~Element,geometry)

def target_values(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,+scope: Scope<Before,Part>) ->
  @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Scope{+target,part,view_axes,offset,before,after,start,+geometry,limits,words,classified,room} = scope
  BandValues.band_values(~Element,~Before,~before_value,target,geometry)

def Target(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,+scope: Scope<Before,Part>) ->
  @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Model.Both(~Element,~Scope<Before,Part>,~Model.Both(~Element,~Scope<Before,Part>,~target_fields(~Element,~Before,~Part),
    ~Model.always(~Element,~Scope<Before,Part>,~Model.ready_band(~Element))),~target_values(~Element,~Before,~Part,~before_value),scope)

def part_fields(~Element: Data,~Before: Data,~Part: Data,+scope: Scope<Before,Part>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Scope{target,part,view_axes,offset,+before,+after,+start,+geometry,limits,words,classified,room} = scope
  BandShape.band_fields(~Element,BandViews.output(BandViews.Banded{before,[],after,start},geometry))

def part_values(~Element: Data,~Before: Data,~Part: Data,~part_value: Part -> U32 -> Element,+scope: Scope<Before,Part>) ->
  @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Scope{target,+part,view_axes,offset,+before,+after,+start,+geometry,limits,words,classified,room} = scope
  BandValues.band_values(~Element,~Part,~part_value,part,BandViews.output(BandViews.Banded{before,[],after,start},geometry))

def PartBand(~Element: Data,~Before: Data,~Part: Data,~part_value: Part -> U32 -> Element,+scope: Scope<Before,Part>) ->
  @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Model.Both(~Element,~Scope<Before,Part>,~Model.Both(~Element,~Scope<Before,Part>,~part_fields(~Element,~Before,~Part),
    ~Model.always(~Element,~Scope<Before,Part>,~Model.ready_band(~Element))),~part_values(~Element,~Before,~Part,~part_value),scope)

# A band of the result holds at each cell what the whole write leaves at the
# pixel of that cell.
def OutputValues(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+scope: Scope<Before,Part>) ->
  @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Scope{+target,+part,+view_axes,+offset,+before,+after,start,+geometry,limits,words,classified,room} = scope
  BandValues.band_values(~Element,~Scatter.Scatter<Before,Part>,~Scatter.value(~Element,~Before,~Part,~before_value,~part_value),
    written(~Before,~Part,target,part,Views.View{view_axes,offset},before,after,geometry),geometry)

def Made(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+scope: Scope<Before,Part>) ->
  @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Global.Made(~Element,~Scope<Before,Part>,~OutputValues(~Element,~Before,~Part,~before_value,~part_value),scope)

# What the scatter of the part band through the local view of the band leaves
# over the target band.
def local_writes(~Element: Data,~Before: Data,~Part: Data,+scope: Scope<Before,Part>,+rows: U32,+target_content: Buffer.Content<Element>,+part_content: Buffer.Content<Element>) ->
  Scatter.Scatter<Buffer.Content<Element>,Buffer.Content<Element>>:
  Scope{target,part,view_axes,offset,+before,+after,+start,+geometry,limits,words,classified,room} = scope
  Scatter.Scatter{target_content,part_content,BandViews.local(BandViews.Banded{before,[],after,start},geometry,rows),
    U32.to_nat(U32.from_nat(Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(rows),ShapeLaws.elements(Views.axis_extents(after))))))}

def Local(~Element: Data,~Before: Data,~Part: Data,+scope: Scope<Before,Part>,+rows: U32,+target_content: Buffer.Content<Element>,+part_content: Buffer.Content<Element>,
  -output: Buffer.Content<Element>) -> Data:
  GatherValues.Facts(~Element,~Scatter.Scatter<Buffer.Content<Element>,Buffer.Content<Element>>,
    ~Scatter.value(~Element,~Buffer.Content<Element>,~Buffer.Content<Element>,~Contract.stored(~Element),~Contract.stored(~Element)),
    local_writes(~Element,~Before,~Part,scope,rows,target_content,part_content),band_shape(~Before,~Part,scope,rows),output)

# ---- The numbers of one band ----

# The local view of a band of rows rows holds the elements of the axes before
# the rows, the rows and the axes after them.
def local_elements(+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,+rows: U32) ->
  {ShapeLaws.elements(Views.axis_extents(BandViews.axes(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width},rows))) ==
    Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(rows),ShapeLaws.elements(Views.axis_extents(after)))) : Nat}:
  +plane = (height * width : U32)
  +band = (rows * width : U32)
  +local_before = BandViews.localized_axes(plane,band,before)
  +local_after = BandViews.localized_axes(plane,band,after)
  %Address.extents_localized(plane,band,before) :
    {ShapeLaws.elements(Views.axis_extents(BandViews.axes(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width},rows))) ==
      Nat.mul(ShapeLaws.elements(_),Nat.mul(U32.to_nat(rows),ShapeLaws.elements(Views.axis_extents(after)))) : Nat}
  %Address.extents_localized(plane,band,after) :
    {ShapeLaws.elements(Views.axis_extents(BandViews.axes(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width},rows))) ==
      Nat.mul(ShapeLaws.elements(Views.axis_extents(local_before)),Nat.mul(U32.to_nat(rows),ShapeLaws.elements(_))) : Nat}
  LineValues.axes_elements(local_before,Views.Axis{rows,width} <> local_after)

# The extents of a band of rows rows hold the same elements.
def extents_elements(+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+start: U32,+rows: U32) ->
  {ShapeLaws.elements(BandViews.extents(BandViews.Banded{before,[],after,start},rows)) ==
    Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(rows),ShapeLaws.elements(Views.axis_extents(after)))) : Nat}:
  ShapeLaws.elements_appended(Views.axis_extents(before),rows <> Views.axis_extents(after))

# A band of the part holds the cells of the band, which fit its capacity.
def part_cells(~Element: Data,~Before: Data,~Part: Data,~part_value: Part -> U32 -> Element,+scope: Scope<Before,Part>,+first: U32,+rows: U32,+part_band: Buffer.Model<Element>,
  +part_fact: PartBand(~Element,~Before,~Part,~part_value,scope,first,rows,Buffer.content(~Element,part_band))) ->
  {Buffer.length(~Element,part_band) == band_cells(~Before,~Part,scope,rows) : Nat}:
  Scope{+target,+part,+view_axes,+offset,+before,+after,+start,+geometry,+limits,+words,+classified,+room} = scope
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Shape.Both{+fitted,+plane} = words
  ViewValues.Fitted{batch_positive,height_positive,channel_reach,column_reach,+early_count,+late_count,repeated,rows_exact} = fitted
  Buffer.Model{+tree,+length,+certificate} = part_band
  Shape.Both{Shape.Both{+fields,ready},values} = part_fact
  +before_word = BandViews.product_or_zero(Views.axis_extents(before))
  +after_word = BandViews.product_or_zero(Views.axis_extents(after))
  +output = BandViews.output(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width})
  %early_count : {length == Nat.mul(_,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(Views.axis_extents(after)))) : Nat}
  %late_count : {length == Nat.mul(U32.to_nat(before_word),Nat.mul(U32.to_nat(rows),_)) : Nat}
  BandValues.leaf_length(output,first,rows,length,fields)

def part_count(~Element: Data,~Before: Data,~Part: Data,~part_value: Part -> U32 -> Element,+scope: Scope<Before,Part>,+first: U32,+rows: U32,+part_band: Buffer.Model<Element>,
  +part_fact: PartBand(~Element,~Before,~Part,~part_value,scope,first,rows,Buffer.content(~Element,part_band))) ->
  {U32.to_nat(U32.from_nat(band_cells(~Before,~Part,scope,rows))) == band_cells(~Before,~Part,scope,rows) : Nat}:
  Buffer.Model{+tree,+length,+certificate} = part_band
  Shape.Both{Shape.Both{fields,+ready},values} = part_fact
  +cells = part_cells(~Element,~Before,~Part,~part_value,scope,first,rows,Buffer.Model{tree,length,certificate},part_fact)
  FullRange.roundtrip(band_cells(~Before,~Part,scope,rows),Tree.capacity(Element,tree),
    %cells : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool} ViewValues.content_room(~Element,Buffer.Content{tree,length},ready))

# A band of the target holds the cells of its shape.
def target_cells(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,+scope: Scope<Before,Part>,+first: U32,+rows: U32,+target_band: Buffer.Model<Element>,
  +target_fact: Target(~Element,~Before,~Part,~before_value,scope,first,rows,Buffer.content(~Element,target_band))) ->
  {Buffer.length(~Element,target_band) == ShapeLaws.elements(band_shape(~Before,~Part,scope,rows)) : Nat}:
  Scope{+target,+part,+view_axes,+offset,+before,+after,+start,+geometry,+limits,+words,+classified,+room} = scope
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Buffer.Model{+tree,+length,+certificate} = target_band
  Shape.Both{Shape.Both{+fields,ready},values} = target_fact
  +c = U32.to_nat(channels)
  +r = U32.to_nat(rows)
  %Equal.sym(Nat,Nat.mul(U32.to_nat(width),1n),U32.to_nat(width),Order.times_one(U32.to_nat(width))) : {length == Nat.mul(c,Nat.mul(r,_)) : Nat}
  BandValues.leaf_length(Bands.Geometry{batch,channels,height,width},first,rows,length,fields)

# A band of the target and the band the scatter leaves are ready.
def target_ready(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,+scope: Scope<Before,Part>,+first: U32,+rows: U32,
  +target_band: Buffer.Model<Element>,fact: Target(~Element,~Before,~Part,~before_value,scope,first,rows,Buffer.content(~Element,target_band))) ->
  Buffer.ContentReady<Element,Buffer.content(~Element,target_band)>:
  Shape.Both{Shape.Both{fields,ready},values} = fact
  ready

def local_ready(~Element: Data,~Before: Data,~Part: Data,+scope: Scope<Before,Part>,+rows: U32,+target_content: Buffer.Content<Element>,+part_content: Buffer.Content<Element>,
  +output: Buffer.Content<Element>,local: Local(~Element,~Before,~Part,scope,rows,target_content,part_content,output)) -> Buffer.ContentReady<Element,output>:
  Shape.Both{Contract.WellFormed{length_exact,ready},values} = local
  ready

# ---- The cells of one band ----

# What one band hands to each of its cells.
type PlacedCells<-Element: Data,-Before: Data,-Part: Data,-before_value: Before -> U32 -> Element,-part_value: Part -> U32 -> Element> is Data:
  PlacedCells{scope: Scope<Before,Part>,first: U32,rows: U32,target_band: Buffer.Model<Element>,part_band: Buffer.Model<Element>,model: Buffer.Model<Element>,
    target_fact: Target(~Element,~Before,~Part,~before_value,scope,first,rows,Buffer.content(~Element,target_band)),
    part_fact: PartBand(~Element,~Before,~Part,~part_value,scope,first,rows,Buffer.content(~Element,part_band)),
    local: Local(~Element,~Before,~Part,scope,rows,Buffer.content(~Element,target_band),Buffer.content(~Element,part_band),Buffer.content(~Element,model))}

def cells_geometry(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,
  evidence: PlacedCells<Element,Before,Part,before_value,part_value>) -> Bands.Geometry:
  PlacedCells{scope,first,rows,target_band,part_band,model,target_fact,part_fact,local} = evidence
  Scope{target,part,view_axes,offset,before,after,start,+geometry,limits,words,classified,room} = scope
  geometry

def cells_rows(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,
  evidence: PlacedCells<Element,Before,Part,before_value,part_value>) -> U32:
  PlacedCells{scope,first,+rows,target_band,part_band,model,target_fact,part_fact,local} = evidence
  rows

def CellAt(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,
  +evidence: PlacedCells<Element,Before,Part,before_value,part_value>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  PlacedCells{+scope,+first,+rows,target_band,part_band,+model,target_fact,part_fact,local} = evidence
  Scope{+target,+part,+view_axes,+offset,+before,+after,start,+geometry,limits,words,classified,room} = scope
  BandValues.At(~Element,~Scatter.Scatter<Before,Part>,~Scatter.value(~Element,~Before,~Part,~before_value,~part_value),
    written(~Before,~Part,target,part,Views.View{view_axes,offset},before,after,geometry),geometry,first,rows,
    Buffer.content_tree(~Element,Buffer.content(~Element,model)),Buffer.content_length(~Element,Buffer.content(~Element,model)),channel,row,column)

# One cell of a band of the result. The scatter of the band reads, at the
# cell, the write of the part band over the target cell; that is what the
# whole write leaves at the pixel of the cell.
def placed_cell(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,
  +evidence: PlacedCells<Element,Before,Part,before_value,part_value>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(cells_geometry(~Element,~Before,~Part,~before_value,~part_value,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(cells_rows(~Element,~Before,~Part,~before_value,~part_value,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(cells_geometry(~Element,~Before,~Part,~before_value,~part_value,evidence))) == True{} : Bool}) ->
  CellAt(~Element,~Before,~Part,~before_value,~part_value,evidence,channel,row,column):
  PlacedCells{+scope,+first,+rows,+target_band,+part_band,+model,+target_fact,+part_fact,+local} = evidence
  Scope{+target,+part,+view_axes,+offset,+before,+after,+start,+geometry,+limits,+words,+classified,+room} = scope
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Shape.Both{+fitted,+plane} = words
  Classified.Classified{+axes_exact,+offset_exact,fits} = classified
  BandGather.Domain{capacity_ok,domain_channels,domain_height,domain_width,+machine_room,allocation_room} = room
  Buffer.Model{+target_tree,+target_length,+target_certificate} = target_band
  Buffer.Model{+part_tree,+part_length,+part_certificate} = part_band
  Buffer.Model{+tree,+length,+certificate} = model
  Shape.Both{Shape.Both{+target_fields,+target_ready},+target_values} = target_fact
  Shape.Both{Shape.Both{part_fields,part_ready},+part_values} = part_fact
  Shape.Both{Contract.WellFormed{+length_exact,ready},+values} = local
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +ea = ShapeLaws.elements(Views.axis_extents(after))
  +b = U32.to_nat(batch)
  +h = U32.to_nat(height)
  +w = U32.to_nat(width)
  +c = U32.to_nat(channels)
  +r = U32.to_nat(rows)
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +scope = {Scope{target,part,view_axes,offset,before,after,start,geometry,limits,words,classified,room} : Scope<Before,Part>}
  +banded = {BandViews.Banded{before,[],after,start} : BandViews.Banded}
  +local_view = BandViews.local(banded,geometry,rows)
  +whole = Classified.whole_axes(banded,geometry)
  +position = RowMajor.index(channel,r,w,row,column)
  +global = Nat.add(U32.to_nat(first),row)
  +pixel = BandValues.pixel(geometry,channel,global,column)
  +capacity = Tree.capacity(Element,target_tree)
  +cell_total = Nat.mul(c,Nat.mul(r,w))
  +cells_room = {%BandValues.leaf_length(geometry,first,rows,target_length,target_fields) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    ViewValues.content_room(~Element,Buffer.Content{target_tree,target_length},target_ready) : {Nat.is_le(cell_total,U32.to_nat(capacity)) == True{} : Bool}}
  +band_exact = FullRange.multiply(rows,width,capacity,
    Order.transitive(Nat.mul(r,w),cell_total,U32.to_nat(capacity),DivisionOrder.le_multiple(c,Nat.mul(r,w),Order.positive_bound(channel,c,channel_inside)),cells_room))
  +band_room = {%Equal.sym(Nat,U32.to_nat((rows * width : U32)),Nat.mul(r,w),band_exact) : {Nat.is_le(Nat.mul(c,_),U32.to_nat(capacity)) == True{} : Bool}
    cells_room : {Nat.is_le(Nat.mul(c,U32.to_nat((rows * width : U32))),U32.to_nat(capacity)) == True{} : Bool}}
  +rows_fit = ViewValues.band_rows_fit(geometry,first,rows,target_length,target_fields)
  +cell = {Cell{before,after,start,batch,channels,height,width,first,rows,capacity,Machine.capacity(limits),channel,row,column,fitted,plane,machine_room,band_exact,band_room,
    rows_fit,channel_inside,row_inside,column_inside} : Cell}
  +shape_cells = Equal.cong(Nat,Nat,last => Nat.mul(c,Nat.mul(r,last)),Nat.mul(w,1n),w,Order.times_one(w))
  +length_cells = Equal.trans(Nat,length,ShapeLaws.elements([channels,rows,width]),cell_total,length_exact,shape_cells)
  +bound = {%Equal.sym(Nat,length,cell_total,length_cells) : {Nat.is_lt(position,_) == True{} : Bool}
    BandValues.position_inside(geometry,rows,channel,row,column,channel_inside,row_inside,column_inside) : {Nat.is_lt(position,length) == True{} : Bool}}
  +cells = Nat.mul(eb,Nat.mul(r,ea))
  +content = {Buffer.Content{part_tree,part_length} : Buffer.Content<Element>}
  +part_band = {Buffer.Model{part_tree,part_length,part_certificate} : Buffer.Model<Element>}
  +count_exact = part_count(~Element,~Before,~Part,~part_value,scope,first,rows,part_band,part_fact)
  +previous = before_value(target,U32.from_nat(pixel))
  +target_read = Tree.read(Element,target_tree,U32.from_nat(position))
  +before_read = BandValues.leaf_read(~Element,~Before,~before_value,target,geometry,first,rows,target_tree,target_length,channel,row,column,target_values,
    channel_inside,row_inside,column_inside)
  +local_read = Contract.read(~Element,~Scatter.Scatter<Buffer.Content<Element>,Buffer.Content<Element>>,
    ~Scatter.value(~Element,~Buffer.Content<Element>,~Buffer.Content<Element>,~Contract.stored(~Element),~Contract.stored(~Element)),
    local_writes(~Element,~Before,~Part,scope,rows,Buffer.Content{target_tree,target_length},content),Buffer.Content{tree,length},position,values,bound)
  +band = band_read(~Element,~Part,~part_value,cell,part,part_tree,part_length,part_values,previous)
  +view_same = Equal.trans(Views.View,Views.View{whole,start},Views.View{view_axes,start},Views.View{view_axes,offset},
    Equal.cong(List<&2,Views.Axis>,Views.View,axes => Views.View{axes,start},whole,view_axes,Equal.sym(List<&2,Views.Axis>,view_axes,whole,axes_exact)),
    Equal.cong(U32,Views.View,word => Views.View{view_axes,word},start,offset,offset_exact))
  +first_read = Readback.selected(~Element,ViewWrites.writes(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
    U32.to_nat(U32.from_nat(cells)),local_view,content),U32.from_nat(position),target_read)
  +counted_read = Readback.selected(~Element,ViewWrites.writes(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
    cells,local_view,content),U32.from_nat(position),target_read)
  +band_selected = Readback.selected(~Element,ViewWrites.writes(~Element,~Buffer.Content<Element>,~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
    cells,local_view,content),U32.from_nat(position),previous)
  +whole_count = Nat.mul(b,Nat.mul(eb,Nat.mul(h,ea)))
  +whole_selected = Readback.selected(~Element,ViewWrites.writes(~Element,~Part,~Scatter.logical(~Element,~Part,~part_value),whole_count,Views.View{whole,start},part),
    U32.from_nat(pixel),previous)
  +view_selected = Readback.selected(~Element,ViewWrites.writes(~Element,~Part,~Scatter.logical(~Element,~Part,~part_value),whole_count,Views.View{view_axes,offset},part),
    U32.from_nat(pixel),previous)
  +read = Equal.trans(Element,Tree.read(Element,tree,U32.from_nat(position)),first_read,view_selected,local_read,
    Equal.trans(Element,first_read,counted_read,view_selected,
      Equal.cong(Nat,Element,amount => Readback.selected(~Element,ViewWrites.writes(~Element,~Buffer.Content<Element>,
          ~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),amount,local_view,content),U32.from_nat(position),target_read),
        U32.to_nat(U32.from_nat(cells)),cells,count_exact),
      Equal.trans(Element,counted_read,band_selected,view_selected,
        Equal.cong(Element,Element,element => Readback.selected(~Element,ViewWrites.writes(~Element,~Buffer.Content<Element>,
            ~Scatter.logical(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),cells,local_view,content),U32.from_nat(position),element),
          target_read,previous,before_read),
        Equal.trans(Element,band_selected,whole_selected,view_selected,Equal.sym(Element,whole_selected,band_selected,band),
          Equal.cong(Views.View,Element,view => Readback.selected(~Element,ViewWrites.writes(~Element,~Part,~Scatter.logical(~Element,~Part,~part_value),whole_count,view,part),
              U32.from_nat(pixel),previous),
            Views.View{whole,start},Views.View{view_axes,offset},view_same)))))
  Shape.Both{bound,read}

# The band of the result, ready and holding at every cell what the whole
# write leaves at the pixel of that cell.
def made_placed(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+scope: Scope<Before,Part>,
  +first: U32,+rows: U32,+target_band: Buffer.Model<Element>,+part_band: Buffer.Model<Element>,+model: Buffer.Model<Element>,
  +target_fact: Target(~Element,~Before,~Part,~before_value,scope,first,rows,Buffer.content(~Element,target_band)),
  +part_fact: PartBand(~Element,~Before,~Part,~part_value,scope,first,rows,Buffer.content(~Element,part_band)),
  +local: Local(~Element,~Before,~Part,scope,rows,Buffer.content(~Element,target_band),Buffer.content(~Element,part_band),Buffer.content(~Element,model))) ->
  Made(~Element,~Before,~Part,~before_value,~part_value,scope,first,rows,Buffer.content(~Element,model)):
  Scope{+target,+part,+view_axes,+offset,+before,+after,+start,+geometry,+limits,+words,+classified,+room} = scope
  Buffer.Model{+tree,+length,+certificate} = model
  +scope = {Scope{target,part,view_axes,offset,before,after,start,geometry,limits,words,classified,room} : Scope<Before,Part>}
  +model = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
  Shape.Both{local_ready(~Element,~Before,~Part,scope,rows,Buffer.content(~Element,target_band),Buffer.content(~Element,part_band),Buffer.content(~Element,model),local),
    BandValues.cells(~PlacedCells<Element,Before,Part,before_value,part_value>,~CellAt(~Element,~Before,~Part,~before_value,~part_value),
      ~cells_geometry(~Element,~Before,~Part,~before_value,~part_value),~cells_rows(~Element,~Before,~Part,~before_value,~part_value),
      ~placed_cell(~Element,~Before,~Part,~before_value,~part_value),PlacedCells{scope,first,rows,target_band,part_band,model,target_fact,part_fact,local})}

# The scattered band, named by its buffer model.
def stored_placed(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+scope: Scope<Before,Part>,
  +first: U32,+rows: U32,+target_band: Buffer.Model<Element>,+part_band: Buffer.Model<Element>,
  +target_fact: Target(~Element,~Before,~Part,~before_value,scope,first,rows,Buffer.content(~Element,target_band)),
  +part_fact: PartBand(~Element,~Before,~Part,~part_value,scope,first,rows,Buffer.content(~Element,part_band)),-storage: Storage.Buffer<Element>,
  observation: Buffer.Observation<Element,storage>,
  local: Local(~Element,~Before,~Part,scope,rows,Buffer.content(~Element,target_band),Buffer.content(~Element,part_band),Buffer.contents(~Element,storage))) ->
  Made(~Element,~Before,~Part,~before_value,~part_value,scope,first,rows,Buffer.contents(~Element,storage)):
  Buffer.Observation{+model,+equation} = observation
  %equation : Made(~Element,~Before,~Part,~before_value,~part_value,scope,first,rows,Buffer.contents(~Element,_))
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) :
    Made(~Element,~Before,~Part,~before_value,~part_value,scope,first,rows,_)
  made_placed(~Element,~Before,~Part,~before_value,~part_value,scope,first,rows,target_band,part_band,model,target_fact,part_fact,
    %Buffer.packed_contents(~Element,model) : Local(~Element,~Before,~Part,scope,rows,Buffer.content(~Element,target_band),Buffer.content(~Element,part_band),_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),storage,equation) :
      Local(~Element,~Before,~Part,scope,rows,Buffer.content(~Element,target_band),Buffer.content(~Element,part_band),Buffer.contents(~Element,_))
    local)

# An accepted scatter leaves the band of the result.
def placed_scattered(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,
  outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+scope: Scope<Before,Part>,+first: U32,+rows: U32,
  +target_band: Buffer.Model<Element>,+part_band: Buffer.Model<Element>,
  +target_fact: Target(~Element,~Before,~Part,~before_value,scope,first,rows,Buffer.content(~Element,target_band)),
  +part_fact: PartBand(~Element,~Before,~Part,~part_value,scope,first,rows,Buffer.content(~Element,part_band)),
  local: Local(~Element,~Before,~Part,scope,rows,Buffer.content(~Element,target_band),Buffer.content(~Element,part_band),
    Copy.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome))),
  accepted: {Each.usable(~Element,Each.checked(~Element,Layout.placed_outcome(~Element,first,rows,outcome))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~Made(~Element,~Before,~Part,~before_value,~part_value,scope),
    Model.contents(~Element,Each.made(~Element,Each.checked(~Element,Layout.placed_outcome(~Element,first,rows,outcome))))):
  match outcome:
    case Operations.Accepted{(kept,Storage.Buffer{storage,length,certificate})}:
      stored_placed(~Element,~Before,~Part,~before_value,~part_value,scope,first,rows,target_band,part_band,target_fact,part_fact,Storage.Buffer{storage,length,certificate},
        Buffer.observe(~Element,Storage.Buffer{storage,length,certificate}),local)
    case Operations.Rejected{(kept,target)}:
      Empty.absurd(Model.ContentAll(~Element,~Made(~Element,~Before,~Part,~before_value,~part_value,scope),
        Model.contents(~Element,Each.made(~Element,Each.checked(~Element,Layout.placed_outcome(~Element,first,rows,Operations.Rejected{(kept,target)}))))),
        WordEncoding.false_true(accepted))

# A usable band was written by an accepted scatter.
def placed_accepted(~Element: Data,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+first: U32,+rows: U32,
  accepted: {Each.usable(~Element,Each.checked(~Element,Layout.placed_outcome(~Element,first,rows,outcome))) == True{} : Bool}) ->
  {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) == True{} : Bool}:
  match outcome:
    case Operations.Accepted{(kept,target)}: {==}
    case Operations.Rejected{(kept,target)}: accepted

# One band through its checked scatter: the part band goes through the local
# view of the band over the target band; a usable call leaves the band of the
# result.
def placed_output(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+scope: Scope<Before,Part>,
  +first: U32,+rows: U32,+pair: Each.BandPair<Element>,
  fact: Each.PairAt(~Element,~Scope<Before,Part>,~Target(~Element,~Before,~Part,~before_value),~PartBand(~Element,~Before,~Part,~part_value),scope,pair,first,rows),
  accepted: {Each.usable(~Element,Each.checked(~Element,Bands.transform_pair(~Element,~Layout.Placing,~Layout.placed_band(~Element),scope_context(~Before,~Part,scope),first,rows,
    Each.pair_pack(~Element,pair),Unit{}))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~Made(~Element,~Before,~Part,~before_value,~part_value,scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,
    Bands.transform_pair(~Element,~Layout.Placing,~Layout.placed_band(~Element),scope_context(~Before,~Part,scope),first,rows,Each.pair_pack(~Element,pair),Unit{}))))):
  Scope{+target,+part,+view_axes,+offset,+before,+after,+start,+geometry,+limits,+words,+classified,+room} = scope
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Each.BandPair{+target_band,+part_band} = pair
  Shape.Both{+target_fact,+part_fact} = fact
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +scope = {Scope{target,part,view_axes,offset,before,after,start,geometry,limits,words,classified,room} : Scope<Before,Part>}
  +banded = {BandViews.Banded{before,[],after,start} : BandViews.Banded}
  +extents = BandViews.extents(banded,rows)
  +local = BandViews.local(banded,geometry,rows)
  +cells = band_cells(~Before,~Part,scope,rows)
  +count = U32.from_nat(cells)
  +count_exact = part_count(~Element,~Before,~Part,~part_value,scope,first,rows,part_band,part_fact)
  +part_exact = part_cells(~Element,~Before,~Part,~part_value,scope,first,rows,part_band,part_fact)
  +extents_exact = Equal.trans(Nat,ShapeLaws.elements(extents),cells,U32.to_nat(count),extents_elements(before,after,start,rows),Equal.sym(Nat,U32.to_nat(count),cells,count_exact))
  +source = CanonicalView.canonical_view(extents,count)
  +source_axes = CanonicalView.axes(source)
  +target_axes = BandViews.axes(banded,geometry,rows)
  +target_offset = BandViews.localized((height * width : U32),(rows * width : U32),start)
  +viewed = {%CanonicalView.view(extents,count,extents_exact) :
      {Each.usable(~Element,Each.checked(~Element,Layout.placed_viewed(~Element,_,local,first,rows,Buffer.pack(~Element,target_band),Buffer.pack(~Element,part_band)))) == True{} : Bool}
    accepted : {Each.usable(~Element,Each.checked(~Element,Layout.placed_outcome(~Element,first,rows,Operations.scatter(~Element,Buffer.pack(~Element,part_band),source,Buffer.pack(~Element,target_band),local)))) == True{} : Bool}}
  +source_exact = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(source_axes)),ShapeLaws.elements(extents),Buffer.length(~Element,part_band),
    Equal.cong(List<&2,U32>,Nat,list => ShapeLaws.elements(list),Views.axis_extents(source_axes),extents,CanonicalView.canonical_shape(extents,count,extents_exact)),
    Equal.trans(Nat,ShapeLaws.elements(extents),cells,Buffer.length(~Element,part_band),extents_elements(before,after,start,rows),
      Equal.sym(Nat,Buffer.length(~Element,part_band),cells,part_exact)))
  +exact = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(target_axes)),cells,U32.to_nat(count),local_elements(before,after,start,batch,channels,height,width,rows),
    Equal.sym(Nat,U32.to_nat(count),cells,count_exact))
  +facts = Scatter.modeled_facts(~Element,~Buffer.Content<Element>,~Buffer.Content<Element>,~Contract.stored(~Element),~Contract.stored(~Element),
    Buffer.content(~Element,target_band),Buffer.content(~Element,part_band),target_axes,target_offset,part_band,target_band,count,source_axes,
    band_shape(~Before,~Part,scope,rows),Buffer.ready_content(~Element,target_band,target_ready(~Element,~Before,~Part,~before_value,scope,first,rows,target_band,target_fact)),Contract.stored_values(~Element,Buffer.content(~Element,part_band)),
    Contract.stored_values(~Element,Buffer.content(~Element,target_band)),source_exact,exact,
    target_cells(~Element,~Before,~Part,~before_value,scope,first,rows,target_band,target_fact),placed_accepted(~Element,Operations.scatter(~Element,Buffer.pack(~Element,part_band),source,Buffer.pack(~Element,target_band),local),first,rows,viewed))
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(extents),Some{source},CanonicalView.view(extents,count,extents_exact)) :
    Model.ContentAll(~Element,~Made(~Element,~Before,~Part,~before_value,~part_value,scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,
      Layout.placed_viewed(~Element,_,local,first,rows,Buffer.pack(~Element,target_band),Buffer.pack(~Element,part_band))))))
  placed_scattered(~Element,~Before,~Part,~before_value,~part_value,Operations.scatter(~Element,Buffer.pack(~Element,part_band),source,Buffer.pack(~Element,target_band),local),scope,first,rows,target_band,part_band,target_fact,part_fact,facts,viewed)

# ---- Every band ----

# A Normal result that holds what the whole part written through the view
# over the target leaves at every index.
def PlacedTensor(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+scope: Scope<Before,Part>,
  -tensor: Tensors.Tensor<Element>) -> Data:
  Global.Contract(~Element,~Scatter.Scatter<Before,Part>,~Scatter.value(~Element,~Before,~Part,~before_value,~part_value),scope_written(~Before,~Part,scope),tensor)

# Every band of the target through its checked scatter, on the plan of the
# target bands, with the part in the same plan. A Ready result is a tensor of
# PlacedTensor, which the caller states as its Result.
def placed(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,
  ~Result: @+scope: Scope<Before,Part> -> @-tensor: Tensors.Tensor<Element> -> Data,+scope: Scope<Before,Part>,
  +name: String,+shape: List<&2,U32>,+fill: Element,+workers: U32,+target_model: Model.Model<Element>,+part_model: Model.Model<Element>,
  +common: Each.Common(~Element,target_model,part_model),
  target_facts: Model.ContentAll(~Element,~Target(~Element,~Before,~Part,~before_value,scope),Model.content(~Element,target_model)),
  part_facts: Model.ContentAll(~Element,~PartBand(~Element,~Before,~Part,~part_value,scope),Model.content(~Element,part_model)),
  counted: Global.Counted<shape>,coherent: Global.Coherent(shape,scope_geometry(~Before,~Part,scope)),
  conclude: @+model: Model.Model<Element> ->
    @placed: PlacedTensor(~Element,~Before,~Part,~before_value,~part_value,scope,
      Tensors.Tensor{Model.pack(~Element,model),scope_geometry(~Before,~Part,scope),shape,fill,Tensors.Usable{},workers}) ->
    Result(scope,Tensors.Tensor{Model.pack(~Element,model),scope_geometry(~Before,~Part,scope),shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},scope_geometry(~Before,~Part,scope),shape,fill,workers,
    Bands.transform_paired(~Element,~Layout.Placing,~Layout.placed_band(~Element),Model.layout(~Element,target_model),
      BandSchedule.parallel(scope_geometry(~Before,~Part,scope),workers,12.0,0.0,Model.layout(~Element,target_model)),
      scope_context(~Before,~Part,scope),Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,target_model),Model.pack(~Element,part_model))))) ==
    True{} : Bool}) ->
  Result(scope,Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},scope_geometry(~Before,~Part,scope),shape,fill,
    workers,Bands.transform_paired(~Element,~Layout.Placing,~Layout.placed_band(~Element),Model.layout(~Element,target_model),
      BandSchedule.parallel(scope_geometry(~Before,~Part,scope),workers,12.0,0.0,Model.layout(~Element,target_model)),
      scope_context(~Before,~Part,scope),Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,target_model),Model.pack(~Element,part_model))))):
  Scope{+target,+part,+view_axes,+offset,+before,+after,+start,+geometry,+limits,+words,+classified,+room} = scope
  +scope = {Scope{target,part,view_axes,offset,before,after,start,geometry,limits,words,classified,room} : Scope<Before,Part>}
  +context = {Layout.Placing{BandViews.Banded{before,[],after,start},geometry} : Layout.Placing}
  +parallel = BandSchedule.parallel(geometry,workers,12.0,0.0,Model.layout(~Element,target_model))
  +transform = Each.paired_transformed(~Element,~Layout.Placing,~Layout.placed_band(~Element),parallel,context,target_model,part_model,common)
  %Equal.sym(Bands.Bands(~Element) & Bool,
      Bands.transform_paired(~Element,~Layout.Placing,~Layout.placed_band(~Element),Model.layout(~Element,target_model),parallel,context,
        Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,target_model),Model.pack(~Element,part_model))),
      Each.checked(~Element,Each.paired_map(~Element,~Layout.Placing,~Layout.placed_band(~Element),context,target_model,part_model)),transform) :
    Result(scope,Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},geometry,shape,fill,workers,_))
  Global.transformed(~Element,~Scope<Before,Part>,~OutputValues(~Element,~Before,~Part,~before_value,~part_value),~Result,scope,
    name,Tensors.InvalidArgument{shape},geometry,shape,fill,workers,limits,BandGather.rows_room(geometry,limits,Machine.capacity(limits),room),
    Each.checked(~Element,Each.paired_map(~Element,~Layout.Placing,~Layout.placed_band(~Element),context,target_model,part_model)),
    usable => Each.shaped_checked_all(~Element,~Each.BandPair<Element>,~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Each.pair_pack(~Element),~Layout.Placing,
      ~Bands.transform_pair(~Element,~Layout.Placing,~Layout.placed_band(~Element)),~Scope<Before,Part>,~scope_context(~Before,~Part),
      ~Each.PairAt(~Element,~Scope<Before,Part>,~Target(~Element,~Before,~Part,~before_value),~PartBand(~Element,~Before,~Part,~part_value)),
      ~Made(~Element,~Before,~Part,~before_value,~part_value),~placed_output(~Element,~Before,~Part,~before_value,~part_value),scope,
      Each.paired_pieces(~Element,target_model,part_model),
      Each.paired_all(~Element,~Scope<Before,Part>,~Target(~Element,~Before,~Part,~before_value),~PartBand(~Element,~Before,~Part,~part_value),scope,target_model,part_model,common,
        target_facts,part_facts),usable),
    counted,coherent,bands => normal => facts => conclude(bands,Shape.Both{normal,facts}),
    %transform : {Contract.tensor_ready(~Element,Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},geometry,shape,fill,workers,_)) == True{} : Bool} accepted)
