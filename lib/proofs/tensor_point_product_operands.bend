# The point job reads its rows at their position in the rectangle.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../tensor_point_products.bend as Products
import ../tensor.bend as Tensors
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./binary_radix.bend as Radix
import ./kernel_gemm_point_buffer.bend as PointBuffer
import ./kernel_gemm_point_jobs.bend as Jobs
import ./kernel_gemm_point_readback.bend as Readback
import ./kernel_gemm_point_spec.bend as Spec
import ./kernel_gemm_point_taps.bend as Taps
import ./kernel_gemm_output_groups.bend as Groups
import ./kernel_tile_refinement.bend as Lanes
import ./machine_limit_bounds.bend as Limits
import ./nat_division.bend as NatDivision
import ./nat_division_order.bend as Division
import ./nat_extrema.bend as Extrema
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as Index
import ./storage_allocation_refinement.bend as Allocation
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./tensor_point_product_values.bend as Values
import ./tensor_point_product_lanes.bend as OutputLanes
import ./tensor_point_products.bend as Point
import ./word_arithmetic.bend as WordAddition
import ./word_encoding.bend as WordEncoding
import ./word_multiplication.bend as WordMultiplication
import ./word_power_of_two.bend as Power

def unit_of(+first: U32,+position: Nat) -> Nat:
  Nat.div(Nat.sub(position,U32.to_nat(first)),4n)

def row_of(+first: U32,+position: Nat) -> Nat:
  Nat.mod(Nat.sub(position,U32.to_nat(first)),4n)

def row_small(+first: U32,+position: Nat) -> {Nat.is_le(row_of(first,position),3n) == True{} : Bool}:
  Order.strict_follow(row_of(first,position),4n,Division.remainder_small(Nat.sub(position,U32.to_nat(first)),4n,{==}))

def row_position(+first: U32,+position: Nat,+after: {Nat.is_le(U32.to_nat(first),position) == True{} : Bool}) ->
  {Nat.add(Nat.add(U32.to_nat(first),Nat.mul(unit_of(first,position),4n)),row_of(first,position)) == position : Nat}:
  +start = U32.to_nat(first)
  +offset = Nat.sub(position,start)
  Equal.trans(Nat,Nat.add(Nat.add(start,Nat.mul(unit_of(first,position),4n)),row_of(first,position)),Nat.add(start,offset),position,
    Readback.step_position(start,offset),Order.restore_shift(start,position,after))

def address_sum(+base: Nat,+first: Nat,+unit: Nat,+row: Nat,+column: Nat) ->
  {Nat.add(Nat.add(column,Nat.add(Nat.add(base,first),unit)),row) ==
    Nat.add(base,Nat.add(Nat.add(Nat.add(first,unit),row),column)) : Nat}:
  +start = Nat.add(Nat.add(base,first),unit)
  +rows = Nat.add(first,unit)
  +reversed = Nat.add(row,column)
  %Equal.sym(Nat,Nat.add(column,start),Nat.add(start,column),Order.commutative(column,start)) :
    {Nat.add(_,row) == Nat.add(base,Nat.add(Nat.add(rows,row),column)) : Nat}
  %Equal.sym(Nat,start,Nat.add(base,rows),Addition.associative(base,first,unit)) :
    {Nat.add(Nat.add(_,column),row) == Nat.add(base,Nat.add(Nat.add(rows,row),column)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(base,rows),column),Nat.add(base,Nat.add(rows,column)),Addition.associative(base,rows,column)) :
    {Nat.add(_,row) == Nat.add(base,Nat.add(Nat.add(rows,row),column)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(base,Nat.add(rows,column)),row),Nat.add(base,Nat.add(Nat.add(rows,column),row)),
    Addition.associative(base,Nat.add(rows,column),row)) :
    {_ == Nat.add(base,Nat.add(Nat.add(rows,row),column)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(rows,column),row),Nat.add(rows,Nat.add(column,row)),Addition.associative(rows,column,row)) :
    {Nat.add(base,_) == Nat.add(base,Nat.add(Nat.add(rows,row),column)) : Nat}
  %Equal.sym(Nat,Nat.add(column,row),reversed,Order.commutative(column,row)) :
    {Nat.add(base,Nat.add(rows,_)) == Nat.add(base,Nat.add(Nat.add(rows,row),column)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(rows,row),column),Nat.add(rows,reversed),Addition.associative(rows,row,column)) : {Nat.add(base,Nat.add(rows,reversed)) == Nat.add(base,_) : Nat}
  {==}

def chosen_group(high: Bool,+pair: Nat,+groups: Nat) -> Nat:
  match high:
    case False{}: Nat.mul(pair,2n)
    case True{}: Nat.min(1n+Nat.mul(pair,2n),Nat.sub(groups,1n))

def pair_lower_inside(+group: Nat,+groups: Nat,+inside: {Nat.is_lt(group,groups) == True{} : Bool}) ->
  {Nat.is_le(1n+Nat.mul(OutputLanes.pair_of(group),2n),groups) == True{} : Bool}:
  +lower = Nat.mul(OutputLanes.pair_of(group),2n)
  +remainder = Nat.mod(group,2n)
  +lower_le = {%Equal.sym(Nat,group,Nat.add(lower,remainder),Division.quotient_exact(group,2n,{==})) : {Nat.is_le(lower,_) == True{} : Bool}
    Order.le_add(lower,remainder) : {Nat.is_le(lower,group) == True{} : Bool}}
  Order.strict_follow(lower,groups,Order.le_lt_trans(lower,group,groups,lower_le,inside))

def odd_group(rest: Nat,+group: Nat,+groups: Nat,+pair: Nat,
  +split: {group == Nat.add(Nat.mul(pair,2n),1n+rest) : Nat},
  +small: {Nat.is_lt(1n+rest,2n) == True{} : Bool},+inside: {Nat.is_lt(group,groups) == True{} : Bool}) ->
  {chosen_group(U32.is_eq(U32.from_nat(1n+rest),1),pair,groups) == group : Nat}:
  match rest:
    case 0n:
      +lower = Nat.mul(pair,2n)
      +last = Nat.sub(groups,1n)
      +high = {Equal.trans(Nat,group,Nat.add(lower,1n),1n+lower,split,Order.commutative(lower,1n)) :
        {group == 1n+lower : Nat}}
      %Equal.sym(Nat,group,1n+lower,high) : {Nat.min(1n+lower,last) == _ : Nat}
      Extrema.minimum_when_ordered(1n+lower,last,%high : {Nat.is_le(_,last) == True{} : Bool} Order.before_last(group,groups,inside))
    case 1n+more:
      Empty.absurd({chosen_group(U32.is_eq(U32.from_nat(1n+1n+more),1),pair,groups) == group : Nat},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(more,0n),True{},Equal.sym(Bool,Nat.is_lt(more,0n),False{},NatDivision.lt_zero(more)),small)))

def parity_group(part: Nat,+group: Nat,+groups: Nat,+pair: Nat,
  +split: {group == Nat.add(Nat.mul(pair,2n),part) : Nat},
  +small: {Nat.is_lt(part,2n) == True{} : Bool},+inside: {Nat.is_lt(group,groups) == True{} : Bool}) ->
  {chosen_group(U32.is_eq(U32.from_nat(part),1),pair,groups) == group : Nat}:
  match part:
    case 0n:
      +lower = Nat.mul(pair,2n)
      Equal.sym(Nat,group,lower,
        Equal.trans(Nat,group,Nat.add(lower,0n),lower,split,Addition.zero_right(lower)))
    case 1n+rest:
      odd_group(rest,group,groups,pair,split,small,inside)

def selected_group(+group: Nat,+groups: Nat,+inside: {Nat.is_lt(group,groups) == True{} : Bool}) ->
  {chosen_group(OutputLanes.upper(group),OutputLanes.pair_of(group),groups) == group : Nat}:
  parity_group(Nat.mod(group,2n),group,groups,OutputLanes.pair_of(group),Division.quotient_exact(group,2n,{==}),Division.remainder_small(group,2n,{==}),inside)

def tiled_half(+tree: Tree.Tree<Tile.Tile8>,+lower: U32,+upper: U32,high: Bool) -> Tile.Tile8:
  match high:
    case False{}: Tree.read(Tile.Tile8,tree,lower)
    case True{}: Tree.read(Tile.Tile8,tree,upper)

def pair_half(+tree: Tree.Tree<Tile.Tile8>,+lower: U32,+upper: U32,high: Bool,+lane: Nat,
  +inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Spec.pair_lane(OutputLanes.selected_lane(high,lane),Tile.Tiles{Tree.read(Tile.Tile8,tree,lower),Tree.read(Tile.Tile8,tree,upper)}) ==
    Lanes.select_lane(lane,tiled_half(tree,lower,upper,high)) : F32}:
  match high:
    case False{}:
      %Equal.sym(Bool,Nat.is_lt(lane,8n),True{},inside) : {Spec.half_lane(_,lane,Tree.read(Tile.Tile8,tree,lower),Tree.read(Tile.Tile8,tree,upper)) ==
        Lanes.select_lane(lane,Tree.read(Tile.Tile8,tree,lower)) : F32}
      {==}
    case True{}:
      Equal.sym(F32,Lanes.select_lane(lane,Tree.read(Tile.Tile8,tree,upper)),
        Spec.pair_lane(Nat.add(8n,lane),Tile.Tiles{Tree.read(Tile.Tile8,tree,lower),Tree.read(Tile.Tile8,tree,upper)}),
        OutputLanes.half_lane(True{},Tile.Tiles{Tree.read(Tile.Tile8,tree,lower),Tree.read(Tile.Tile8,tree,upper)},lane,inside))

def addressed_half(~Scope: Data,~address: Scope -> Nat -> U32,+scope: Scope,+tree: Tree.Tree<Tile.Tile8>,+pair: Nat,+groups: Nat,high: Bool,+lane: Nat,
  +inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Spec.pair_lane(OutputLanes.selected_lane(high,lane),
    Tile.Tiles{Tree.read(Tile.Tile8,tree,address(scope,Nat.mul(pair,2n))),Tree.read(Tile.Tile8,tree,address(scope,Nat.min(1n+Nat.mul(pair,2n),Nat.sub(groups,1n))))}) ==
    Lanes.select_lane(lane,Tree.read(Tile.Tile8,tree,address(scope,chosen_group(high,pair,groups)))) : F32}:
  match high:
    case False{}:
      pair_half(tree,address(scope,Nat.mul(pair,2n)),address(scope,Nat.min(1n+Nat.mul(pair,2n),Nat.sub(groups,1n))),False{},lane,inside)
    case True{}:
      pair_half(tree,address(scope,Nat.mul(pair,2n)),address(scope,Nat.min(1n+Nat.mul(pair,2n),Nat.sub(groups,1n))),True{},lane,inside)

def row_address(+base: Nat,+first: Nat,+unit_rows: Nat,+row: Nat,+stride: Nat,+column: Nat) ->
  {Nat.add(Nat.add(column,Nat.add(Nat.add(base,Nat.mul(first,stride)),Nat.mul(unit_rows,stride))),Nat.mul(row,stride)) ==
    Nat.add(base,Nat.add(Nat.mul(Nat.add(Nat.add(first,unit_rows),row),stride),column)) : Nat}:
  +first_cell = Nat.mul(first,stride)
  +unit_cell = Nat.mul(unit_rows,stride)
  +row_cell = Nat.mul(row,stride)
  +start = Nat.add(Nat.add(base,first_cell),unit_cell)
  +preceding = Nat.mul(Nat.add(first,unit_rows),stride)
  +all = Nat.mul(Nat.add(Nat.add(first,unit_rows),row),stride)
  %Equal.sym(Nat,all,Nat.add(preceding,row_cell),Index.distrib(Nat.add(first,unit_rows),row,stride)) :
    {Nat.add(Nat.add(column,start),row_cell) == Nat.add(base,Nat.add(_,column)) : Nat}
  %Equal.sym(Nat,preceding,Nat.add(first_cell,unit_cell),Index.distrib(first,unit_rows,stride)) :
    {Nat.add(Nat.add(column,start),row_cell) == Nat.add(base,Nat.add(Nat.add(_,row_cell),column)) : Nat}
  address_sum(base,first_cell,unit_cell,row_cell,column)
def input_value(+input: Tree.Tree<Tile.Tile8>,+base: Nat,+stride: Nat,+position: Nat,+key: Nat) -> F32:
  Lanes.select_lane(Nat.mod(key,8n),Tree.read(Tile.Tile8,input,U32.from_nat(Nat.add(base,Nat.add(Nat.mul(position,stride),Nat.div(key,8n))))))

def encoded_shift(+before: Nat,+after: Nat,+offset: Nat,+same: {U32.from_nat(before) == U32.from_nat(after) : U32}) ->
  {U32.from_nat(Nat.add(before,offset)) == U32.from_nat(Nat.add(after,offset)) : U32}:
  %WordAddition.native_add(before,offset) :
    {_ == U32.from_nat(Nat.add(after,offset)) : U32}
  %WordAddition.native_add(after,offset) : {(U32.from_nat(before) + U32.from_nat(offset) : U32) == _ : U32}
  %same : {(U32.from_nat(before) + U32.from_nat(offset) : U32) == (_ + U32.from_nat(offset) : U32) : U32}
  {==}

def encoded_start(+base: U32,+first: U32,+stride: U32) ->
  {U32.from_nat(U32.to_nat((base + first * stride : U32))) == U32.from_nat(Nat.add(U32.to_nat(base),Nat.mul(U32.to_nat(first),U32.to_nat(stride)))) : U32}:
  +scaled = Nat.mul(U32.to_nat(first),U32.to_nat(stride))
  %Equal.sym(U32,U32.from_nat(U32.to_nat((base + first * stride : U32))),(base + first * stride : U32),Arithmetic.native_roundtrip((base + first * stride : U32))) :
    {_ == U32.from_nat(Nat.add(U32.to_nat(base),scaled)) : U32}
  %Equal.sym(U32,(first * stride : U32),U32.from_nat(scaled),WordMultiplication.native_multiply(first,stride)) :
    {(base + _ : U32) == U32.from_nat(Nat.add(U32.to_nat(base),scaled)) : U32}
  %Arithmetic.native_roundtrip(base) : {(_ + U32.from_nat(scaled) : U32) == U32.from_nat(Nat.add(U32.to_nat(base),scaled)) : U32}
  WordAddition.native_add(U32.to_nat(base),scaled)

def changed_base(+before: Nat,+after: Nat,+unit: Nat,+row: Nat,+column: Nat,+same: {U32.from_nat(before) == U32.from_nat(after) : U32}) ->
  {U32.from_nat(Nat.add(Nat.add(column,Nat.add(before,unit)),row)) == U32.from_nat(Nat.add(Nat.add(column,Nat.add(after,unit)),row)) : U32}:
  +early = Nat.add(before,unit)
  +late = Nat.add(after,unit)
  %Equal.sym(Nat,Nat.add(column,early),Nat.add(early,column),Order.commutative(column,early)) :
    {U32.from_nat(Nat.add(_,row)) == U32.from_nat(Nat.add(Nat.add(column,late),row)) : U32}
  %Equal.sym(Nat,Nat.add(column,late),Nat.add(late,column),Order.commutative(column,late)) :
    {U32.from_nat(Nat.add(Nat.add(early,column),row)) == U32.from_nat(Nat.add(_,row)) : U32}
  encoded_shift(Nat.add(early,column),Nat.add(late,column),row,encoded_shift(early,late,column,encoded_shift(before,after,unit,same)))

def input_at(+rectangle: Products.Rectangle,+base: U32,+first: U32,+span: U32,+readers: Jobs.Readers,+position: Nat,+key: Nat,
  +after: {Nat.is_le(U32.to_nat(first),position) == True{} : Bool}) ->
  {Taps.physical_input(Readback.Target{Values.matrix_at(rectangle,base,first,span),readers},unit_of(first,position),row_of(first,position),key) ==
    input_value(Taps.input_of(Readback.Target{Values.matrix_at(rectangle,base,first,span),readers}),U32.to_nat(base),
      U32.to_nat(Gemm.blocks(Values.input_inner(rectangle))),position,key) : F32}:
  Products.Rectangle{out_first,+positions,+inner,+channels,+group_major,+grid} = rectangle
  +groups = Gemm.blocks(inner)
  +stride = U32.to_nat(groups)
  +offset = U32.to_nat(base)
  +start = U32.to_nat(first)
  +unit_rows = Nat.mul(unit_of(first,position),4n)
  +row = row_of(first,position)
  +column = Nat.div(key,8n)
  +address = (base + first * groups : U32)
  +scaled = Nat.mul(start,stride)
  +unit_cells = Nat.mul(unit_rows,stride)
  +row_cells = Nat.mul(row,stride)
  +physical = Nat.add(Nat.add(column,Nat.add(U32.to_nat(address),Nat.mul(unit_rows,stride))),Nat.mul(row,stride))
  +normalized = Nat.add(Nat.add(column,Nat.add(Nat.add(offset,scaled),unit_cells)),row_cells)
  +wanted = Nat.add(offset,Nat.add(Nat.mul(position,stride),column))
  +input = Taps.input_of(Readback.Target{Values.matrix_at(rectangle,base,first,span),readers})
  Equal.cong(U32,F32,value => Lanes.select_lane(Nat.mod(key,8n),Tree.read(Tile.Tile8,input,value)),U32.from_nat(physical),U32.from_nat(wanted),
    Equal.trans(U32,U32.from_nat(physical),U32.from_nat(normalized),U32.from_nat(wanted),
      changed_base(U32.to_nat(address),Nat.add(offset,scaled),unit_cells,row_cells,column,encoded_start(base,first,groups)),
      Equal.cong(Nat,U32,U32.from_nat,normalized,wanted,
        %row_position(first,position,after) :
          {normalized == Nat.add(offset,Nat.add(Nat.mul(_,stride),column)) : Nat}
        row_address(offset,start,unit_rows,row,stride,column))))

def panel_position(+base: U32,+tap: U32,+groups: U32,+group: Nat) ->
  {(base + tap * groups + U32.from_nat(group) : U32) ==
    U32.from_nat(Nat.add(Nat.add(U32.to_nat(base),Nat.mul(U32.to_nat(tap),U32.to_nat(groups))),group)) : U32}:
  +end = Nat.mul(U32.to_nat(tap),U32.to_nat(groups))
  +first = U32.to_nat(base)
  %Equal.sym(U32,(tap * groups : U32),U32.from_nat(end),WordMultiplication.native_multiply(tap,groups)) :
    {(base + _ + U32.from_nat(group) : U32) == U32.from_nat(Nat.add(Nat.add(first,end),group)) : U32}
  %Arithmetic.native_roundtrip(base) :
    {(_ + U32.from_nat(end) + U32.from_nat(group) : U32) == U32.from_nat(Nat.add(Nat.add(first,end),group)) : U32}
  %Equal.sym(U32,(U32.from_nat(first) + U32.from_nat(end) : U32),U32.from_nat(Nat.add(first,end)),WordAddition.native_add(first,end)) :
    {(_ + U32.from_nat(group) : U32) == U32.from_nat(Nat.add(Nat.add(first,end),group)) : U32}
  WordAddition.native_add(Nat.add(first,end),group)

def panel_pair_address(+tree: Tree.Tree<Tile.Tile8>,+base: U32,+groups: U32,+inner: U32,+pair: Nat,+tap: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +index_room: {Nat.is_le(Nat.add(Nat.mul(pair,U32.to_nat(inner)),tap),U32.to_nat(cap)) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool},
  +group_words: Gemm.Both<{U32.from_nat(Nat.mul(pair,2n)) == (U32.from_nat(pair) * 2 : U32) : U32},
    {U32.from_nat(Nat.min(1n+Nat.mul(pair,2n),Nat.sub(U32.to_nat(groups),1n))) ==
      U32.min((U32.from_nat(pair) * 2 + 1 : U32),(groups - 1 : U32)) : U32}>) ->
  {PointBuffer.panel_sample(tree,Gemm.PanelPlan{groups,inner,base},U32.from_nat(Nat.add(Nat.mul(pair,U32.to_nat(inner)),tap))) ==
    Tile.Tiles{Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.add(U32.to_nat(base),Nat.mul(tap,U32.to_nat(groups))),Nat.mul(pair,2n)))),
      Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.add(U32.to_nat(base),Nat.mul(tap,U32.to_nat(groups))),
        Nat.min(1n+Nat.mul(pair,2n),Nat.sub(U32.to_nat(groups),1n)))))} : Tile.Tile16}:
  Gemm.Both{+lower_word,+upper_word} = group_words
  +width = U32.to_nat(inner)
  +num_groups = U32.to_nat(groups)
  +positive = Order.positive_bound(tap,width,tap_inside)
  +index = Nat.add(Nat.mul(pair,width),tap)
  +word = U32.from_nat(index)
  +lower = Nat.mul(pair,2n)
  +upper = Nat.min(1n+lower,Nat.sub(num_groups,1n))
  +start = Nat.add(U32.to_nat(base),Nat.mul(tap,num_groups))
  +wanted = {Tile.Tiles{Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(start,lower))),
    Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(start,upper)))} : Tile.Tile16}
  +index_exact = Bounds.roundtrip(index,cap,cap_ok,index_room)
  +pair_exact = Radix.quotient_word(word,inner,pair,tap,width,cap,cap_ok,index_exact,{==},positive,tap_inside,divisor_room)
  +tap_exact = Radix.remainder_word(word,inner,pair,tap,width,cap,cap_ok,index_exact,{==},positive,tap_inside,divisor_room)
  +q = U32.from_nat(pair)
  +t = (word % inner : U32)
  +b = (q * 2 : U32)
  +high = U32.min((b + 1 : U32),(groups - 1 : U32))
  +row = (base + t * groups : U32)
  +panel = {Gemm.PanelPlan{groups,inner,base} : Gemm.PanelPlan}
  +first_address = {%tap_exact : {(row + U32.from_nat(lower) : U32) == U32.from_nat(Nat.add(Nat.add(U32.to_nat(base),Nat.mul(_,num_groups)),lower)) : U32}
    panel_position(base,t,groups,lower) : {(row + U32.from_nat(lower) : U32) == U32.from_nat(Nat.add(start,lower)) : U32}}
  +second_address = {%tap_exact : {(row + U32.from_nat(upper) : U32) == U32.from_nat(Nat.add(Nat.add(U32.to_nat(base),Nat.mul(_,num_groups)),upper)) : U32}
    panel_position(base,t,groups,upper) : {(row + U32.from_nat(upper) : U32) == U32.from_nat(Nat.add(start,upper)) : U32}}
  %Arithmetic.from_nat_exact((word / inner : U32),pair,pair_exact) :
    {Tile.Tiles{Tree.read(Tile.Tile8,tree,(row + 2 * _ : U32)),
      Tree.read(Tile.Tile8,tree,(row + U32.min((2 * _ + 1 : U32),(groups - 1 : U32)) : U32))} == wanted : Tile.Tile16}
  %Equal.sym(U32,(2 * q : U32),b,Arithmetic.multiply_commutative(2,q)) :
    {Tile.Tiles{Tree.read(Tile.Tile8,tree,(row + _ : U32)),
      Tree.read(Tile.Tile8,tree,(row + U32.min((_ + 1 : U32),(groups - 1 : U32)) : U32))} == wanted : Tile.Tile16}
  %lower_word : {Tile.Tiles{Tree.read(Tile.Tile8,tree,(row + _ : U32)),Tree.read(Tile.Tile8,tree,(row + high : U32))} == wanted : Tile.Tile16}
  %upper_word : {Tile.Tiles{Tree.read(Tile.Tile8,tree,(row + U32.from_nat(lower) : U32)),Tree.read(Tile.Tile8,tree,(row + _ : U32))} == wanted : Tile.Tile16}
  %Equal.sym(U32,(row + U32.from_nat(lower) : U32),U32.from_nat(Nat.add(start,lower)),first_address) :
    {Tile.Tiles{Tree.read(Tile.Tile8,tree,_),Tree.read(Tile.Tile8,tree,(row + U32.from_nat(upper) : U32))} == wanted : Tile.Tile16}
  %Equal.sym(U32,(row + U32.from_nat(upper) : U32),U32.from_nat(Nat.add(start,upper)),second_address) :
    {Tile.Tiles{Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(start,lower))),Tree.read(Tile.Tile8,tree,_)} == wanted : Tile.Tile16}
  {==}

def panel_address(+tree: Tree.Tree<Tile.Tile8>,+base: U32,+groups: U32,+inner: U32,+pair: Nat,+tap: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +groups_room: {Nat.is_le(U32.to_nat(groups),U32.to_nat(cap)) == True{} : Bool},
  +index_room: {Nat.is_le(Nat.add(Nat.mul(pair,U32.to_nat(inner)),tap),U32.to_nat(cap)) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool},
  +pair_inside: {Nat.is_le(1n+Nat.mul(pair,2n),U32.to_nat(groups)) == True{} : Bool}) ->
  {PointBuffer.panel_sample(tree,Gemm.PanelPlan{groups,inner,base},U32.from_nat(Nat.add(Nat.mul(pair,U32.to_nat(inner)),tap))) ==
    Tile.Tiles{Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.add(U32.to_nat(base),Nat.mul(tap,U32.to_nat(groups))),Nat.mul(pair,2n)))),
      Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.add(U32.to_nat(base),Nat.mul(tap,U32.to_nat(groups))),
        Nat.min(1n+Nat.mul(pair,2n),Nat.sub(U32.to_nat(groups),1n)))))} : Tile.Tile16}:
  panel_pair_address(tree,base,groups,inner,pair,tap,cap,cap_ok,index_room,divisor_room,tap_inside,
    Readback.pair_group_words(groups,U32.to_nat(groups),{==},cap,cap_ok,groups_room,pair,pair_inside))

def shared_address(inner_tap: Gemm.Both<Nat,Nat>,group: Nat) -> U32:
  Gemm.Both{+inner,+tap} = inner_tap
  U32.from_nat(Nat.add(Nat.mul(group,inner),tap))

def panel_cell_address(start: Nat,group: Nat) -> U32:
  U32.from_nat(Nat.add(start,group))

def bias_address(scope: Unit,group: Nat) -> U32:
  U32.from_nat(group)

def bias_at(+tree: Tree.Tree<Tile.Tile8>,+channels: U32,+group: Nat,+lane: Nat,+limits: Machine.Limits,
  +groups_room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Spec.pair_lane(OutputLanes.selected_lane(OutputLanes.upper(group),lane),Readback.biases(tree,Gemm.blocks(channels),U32.from_nat(OutputLanes.pair_of(group)))) ==
    Lanes.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(group))) : F32}:
  +groups = Gemm.blocks(channels)
  +count = U32.to_nat(groups)
  +pair = OutputLanes.pair_of(group)
  +high = OutputLanes.upper(group)
  +lo = Nat.mul(pair,2n)
  +hi = Nat.min(1n+lo,Nat.sub(count,1n))
  +packed = Readback.biases(tree,groups,U32.from_nat(pair))
  +value = {Tile.Tiles{Tree.read(Tile.Tile8,tree,U32.from_nat(lo)),Tree.read(Tile.Tile8,tree,U32.from_nat(hi))} : Tile.Tile16}
  %Equal.sym(Tile.Tile16,packed,value,Readback.biases_address(tree,groups,count,{==},Machine.capacity(limits),Limits.capacity_fits(limits),
    groups_room,pair,pair_lower_inside(group,count,group_inside))) :
    {Spec.pair_lane(OutputLanes.selected_lane(high,lane),_) == Lanes.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(group))) : F32}
  %selected_group(group,count,group_inside) :
    {Spec.pair_lane(OutputLanes.selected_lane(high,lane),value) == Lanes.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(_))) : F32}
  addressed_half(~Unit,~bias_address,Unit{},tree,pair,count,high,lane,lane_inside)

type PairSpace<-limits: Machine.Limits,-channels: U32,-inner: U32> is Data:
  PairSpace{groups_room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    width_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    weights_room: {Nat.is_le(Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

type PackedTap<-channels: U32,-inner: U32,-index: Nat> is Data:
  PackedTap{depth: Nat,balance: Shape.Balanced(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner))),
    depth_ok: {Nat.is_le(depth,31n) == True{} : Bool},
    capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner))),2147483648) == True{} : Bool},
    fits: {Nat.is_le(PointBuffer.pair_count(channels,inner),U32.to_nat(Tree.capacity(Tile.Tile16,Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner))))) == True{} : Bool},
    inside: {Nat.is_lt(index,PointBuffer.pair_count(channels,inner)) == True{} : Bool},
    index_room: {Nat.is_le(index,PointBuffer.pair_count(channels,inner)) == True{} : Bool}}

def packed_tap_ready(+channels: U32,+inner: U32,+index: Nat,
  +ready: Values.StageReady<Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner)),PointBuffer.pair_count(channels,inner)>,
  +inside: {Nat.is_lt(index,PointBuffer.pair_count(channels,inner)) == True{} : Bool}) -> PackedTap<channels,inner,index>:
  Values.StageReady{+depth,+balance,+depth_ok,+fits} = ready
  +tree = Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner))
  +limit = Order.successor(depth,30n,depth_ok)
  PackedTap{depth,balance,limit,
    %Equal.sym(U32,Tree.capacity(Tile.Tile16,tree),U32.shln(1,depth),Shape.capacity(Tile.Tile16,depth,tree,balance)) :
      {U32.is_le(_,2147483648) == True{} : Bool} Power.native_power_bound(depth,limit),
    fits,inside,Bounds.strict_le(index,PointBuffer.pair_count(channels,inner),inside)}

def packed_tap(+limits: Machine.Limits,+channels: U32,+inner: U32,+group: Nat,+tap: Nat,+space: PairSpace<limits,channels,inner>,
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool}) ->
  PackedTap<channels,inner,Nat.add(Nat.mul(OutputLanes.pair_of(group),U32.to_nat(inner)),tap)>:
  PairSpace{+groups_room,+width_room,+weights_room} = space
  +groups = U32.to_nat(Gemm.blocks(channels))
  +pairs = Gemm.pairs_of(channels)
  +columns = U32.to_nat(pairs)
  +width = U32.to_nat(inner)
  +pair = OutputLanes.pair_of(group)
  +index = Nat.add(Nat.mul(pair,width),tap)
  +word = (pairs * inner : U32)
  +cells = Nat.mul(columns,width)
  +cap = Machine.capacity(limits)
  +cap_ok = Limits.capacity_fits(limits)
  +room = Order.transitive(cells,Nat.mul(groups,width),U32.to_nat(cap),Index.mul_le(columns,groups,width,Groups.pairs_fewer(limits,channels,groups_room)),weights_room)
  +exact = Arithmetic.product_exact(cap,cap_ok,pairs,inner,columns,width,{==},{==},room)
  packed_tap_ready(channels,inner,index,
    %Equal.sym(Tree.Tree<Tile.Tile16>,Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner)),Tree.fresh(Tile.Tile16,Tensors.depth_for(word),Tile.filled_pair(0.0)),
      Allocation.buffer_tree(~Tile.Tile16,Tile.filled_pair(0.0),word)) :
      Values.StageReady<_,PointBuffer.pair_count(channels,inner)>
    Values.initial_ready(limits,word,U32.to_nat(word),{==},
      %Equal.sym(Nat,U32.to_nat(word),cells,exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room),
    %Equal.sym(Nat,U32.to_nat(word),cells,exact) : {Nat.is_lt(index,_) == True{} : Bool}
    Index.radix(pair,columns,tap,width,OutputLanes.pair_inside(limits,channels,group,groups_room,group_inside),tap_inside))

def selected_cell(~Scope: Data,~address: Scope -> Nat -> U32,+scope: Scope,+tree: Tree.Tree<Tile.Tile8>,+groups: Nat,+group: Nat,+lane: Nat,
  +actual: Tile.Tile16,
  +sample: {actual == Tile.Tiles{Tree.read(Tile.Tile8,tree,address(scope,Nat.mul(OutputLanes.pair_of(group),2n))),
    Tree.read(Tile.Tile8,tree,address(scope,Nat.min(1n+Nat.mul(OutputLanes.pair_of(group),2n),Nat.sub(groups,1n))))} : Tile.Tile16},
  +group_inside: {Nat.is_lt(group,groups) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Spec.pair_lane(OutputLanes.selected_lane(OutputLanes.upper(group),lane),actual) ==
    Lanes.select_lane(lane,Tree.read(Tile.Tile8,tree,address(scope,group))) : F32}:
  +high = OutputLanes.upper(group)
  +pair = OutputLanes.pair_of(group)
  +low = Nat.mul(pair,2n)
  +upper = Nat.min(1n+low,Nat.sub(groups,1n))
  +value = {Tile.Tiles{Tree.read(Tile.Tile8,tree,address(scope,low)),Tree.read(Tile.Tile8,tree,address(scope,upper))} : Tile.Tile16}
  %Equal.sym(Tile.Tile16,actual,value,sample) :
    {Spec.pair_lane(OutputLanes.selected_lane(high,lane),_) == Lanes.select_lane(lane,Tree.read(Tile.Tile8,tree,address(scope,group))) : F32}
  %selected_group(group,groups,group_inside) :
    {Spec.pair_lane(OutputLanes.selected_lane(high,lane),value) == Lanes.select_lane(lane,Tree.read(Tile.Tile8,tree,address(scope,_))) : F32}
  addressed_half(~Scope,~address,scope,tree,pair,groups,high,lane,lane_inside)

def shared_weight_cell(+limits: Machine.Limits,+rectangle: Products.Rectangle,+weights: Buffer.Model<Tile.Tile8>,+group: Nat,+lane: Nat,+key: Nat,
  +space: PairSpace<limits,Values.rectangle_channels(rectangle),Values.input_inner(rectangle)>,
  +checked: PackedTap<Values.rectangle_channels(rectangle),Values.input_inner(rectangle),
    Nat.add(Nat.mul(OutputLanes.pair_of(group),U32.to_nat(Values.input_inner(rectangle))),key)>,
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle)))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +key_inside: {Nat.is_lt(key,U32.to_nat(Values.input_inner(rectangle))) == True{} : Bool}) ->
  {Spec.pair_lane(OutputLanes.selected_lane(OutputLanes.upper(group),lane),
    Tree.read(Tile.Tile16,Buffer.content_tree(~Tile.Tile16,Point.shared_pair_model(rectangle,weights)),
      U32.from_nat(Nat.add(Nat.mul(OutputLanes.pair_of(group),U32.to_nat(Values.input_inner(rectangle))),key)))) ==
    Lanes.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,weights),
      shared_address(Gemm.Both{U32.to_nat(Values.input_inner(rectangle)),key},group))) : F32}:
  Products.Rectangle{start,rows,+inner,+channels,major,grid} = rectangle
  PairSpace{+groups_room,+width_room,+weights_room} = space
  PackedTap{+depth,+balance,+depth_ok,+capacity_ok,+fits,+inside,+index_room} = checked
  +pair = OutputLanes.pair_of(group)
  +group_word = Gemm.blocks(channels)
  +groups = U32.to_nat(group_word)
  +width = U32.to_nat(inner)
  +index = Nat.add(Nat.mul(pair,width),key)
  +word = U32.from_nat(index)
  +tree = Buffer.tree(~Tile.Tile8,weights)
  +fresh = Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner))
  +actual = Tree.read(Tile.Tile16,PointBuffer.pair_loop(channels,inner,tree,fresh),word)
  +sampled = PointBuffer.pair_tiles(tree,Gemm.PairPlan{group_word,inner},word)
  +scope = {Gemm.Both{width,key} : Gemm.Both<Nat,Nat>}
  +value = {Tile.Tiles{Tree.read(Tile.Tile8,tree,shared_address(scope,Nat.mul(pair,2n))),
    Tree.read(Tile.Tile8,tree,shared_address(scope,Nat.min(1n+Nat.mul(pair,2n),Nat.sub(groups,1n))))} : Tile.Tile16}
  selected_cell(~Gemm.Both<Nat,Nat>,~shared_address,scope,tree,groups,group,lane,actual,
    Equal.trans(Tile.Tile16,actual,sampled,value,
      PointBuffer.pair_at(channels,inner,tree,fresh,depth,balance,depth_ok,capacity_ok,fits,index,inside),
      PointBuffer.pair_address(tree,group_word,inner,groups,width,{==},{==},Machine.half_word(limits),Limits.half_word_fits(limits),
        Limits.within_half(limits,Nat.mul(groups,width),weights_room),Limits.double_bound(limits,width,width_room),pair,key,key_inside,pair_lower_inside(group,groups,group_inside))),
    group_inside,lane_inside)

def panel_weight_cell(+limits: Machine.Limits,+rectangle: Products.Rectangle,+base: U32,+weights: Buffer.Model<Tile.Tile8>,+group: Nat,+lane: Nat,+key: Nat,
  +space: PairSpace<limits,Values.rectangle_channels(rectangle),Values.input_inner(rectangle)>,
  +checked: PackedTap<Values.rectangle_channels(rectangle),Values.input_inner(rectangle),
    Nat.add(Nat.mul(OutputLanes.pair_of(group),U32.to_nat(Values.input_inner(rectangle))),key)>,
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle)))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},+key_inside: {Nat.is_lt(key,U32.to_nat(Values.input_inner(rectangle))) == True{} : Bool}) ->
  {Spec.pair_lane(OutputLanes.selected_lane(OutputLanes.upper(group),lane),
    Tree.read(Tile.Tile16,Buffer.content_tree(~Tile.Tile16,Point.panel_pair_model(rectangle,base,weights)),
      U32.from_nat(Nat.add(Nat.mul(OutputLanes.pair_of(group),U32.to_nat(Values.input_inner(rectangle))),key)))) ==
    Lanes.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,weights),
      panel_cell_address(Nat.add(U32.to_nat(base),Nat.mul(key,U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle))))),group))) : F32}:
  Products.Rectangle{start,rows,+inner,+channels,major,grid} = rectangle
  PairSpace{+groups_room,+width_room,+weights_room} = space
  PackedTap{+depth,+balance,+depth_ok,+capacity_ok,+fits,+inside,+index_room} = checked
  +pair = OutputLanes.pair_of(group)
  +group_word = Gemm.blocks(channels)
  +groups = U32.to_nat(group_word)
  +width = U32.to_nat(inner)
  +index = Nat.add(Nat.mul(pair,width),key)
  +word = U32.from_nat(index)
  +tree = Buffer.tree(~Tile.Tile8,weights)
  +fresh = Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner))
  +actual = Tree.read(Tile.Tile16,PointBuffer.panel_loop(channels,inner,base,tree,fresh),word)
  +sampled = PointBuffer.panel_sample(tree,Gemm.PanelPlan{group_word,inner,base},word)
  +start = Nat.add(U32.to_nat(base),Nat.mul(key,groups))
  +value = {Tile.Tiles{Tree.read(Tile.Tile8,tree,panel_cell_address(start,Nat.mul(pair,2n))),
    Tree.read(Tile.Tile8,tree,panel_cell_address(start,Nat.min(1n+Nat.mul(pair,2n),Nat.sub(groups,1n))))} : Tile.Tile16}
  +cells = Nat.mul(U32.to_nat(Gemm.pairs_of(channels)),width)
  +capacity = Machine.capacity(limits)
  +room = Order.transitive(cells,Nat.mul(groups,width),U32.to_nat(capacity),
    Index.mul_le(U32.to_nat(Gemm.pairs_of(channels)),groups,width,Groups.pairs_fewer(limits,channels,groups_room)),weights_room)
  +count_exact = Arithmetic.product_exact(capacity,Limits.capacity_fits(limits),Gemm.pairs_of(channels),inner,U32.to_nat(Gemm.pairs_of(channels)),width,{==},{==},room)
  +index_cap = Limits.within_half(limits,index,Order.transitive(index,cells,U32.to_nat(capacity),
    %count_exact : {Nat.is_le(index,_) == True{} : Bool} index_room,room))
  selected_cell(~Nat,~panel_cell_address,start,tree,groups,group,lane,actual,
    Equal.trans(Tile.Tile16,actual,sampled,value,
      PointBuffer.panel_at(channels,inner,base,tree,fresh,depth,balance,depth_ok,capacity_ok,fits,index,inside),
      panel_address(tree,base,group_word,inner,pair,key,Machine.half_word(limits),Limits.half_word_fits(limits),
        Limits.within_half(limits,groups,groups_room),index_cap,Limits.double_bound(limits,width,width_room),key_inside,pair_lower_inside(group,groups,group_inside))),
    group_inside,lane_inside)
