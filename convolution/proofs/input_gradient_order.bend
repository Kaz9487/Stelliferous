# Phase products reindex the existing strided specification in its tap order.
import Base
import ../specification.bend as ConvSpec
import ../input_gradient_specification.bend as Spec
import ../../lib/proofs/input_gradient_phase_taps.bend as Taps
import ../../lib/proofs/ordered_reduction_split.bend as Ordered
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/ordered_matrix.bend as OrderedMatrix

type Flat is Data:
  Flat{problem: Spec.Problem,channel: Nat,y: Nat,x: Nat}

def x_residue(+problem: Spec.Problem,x: Nat) -> Nat:
  Nat.mod(Nat.add(x,Spec.padding(problem)),Spec.stride(problem))

def y_residue(+problem: Spec.Problem,y: Nat) -> Nat:
  Nat.mod(Nat.add(y,Spec.padding(problem)),Spec.stride(problem))

def row_aligned(+problem: Spec.Problem,y: Nat,row: Nat) -> Bool:
  Nat.is_eq(y_residue(problem,y),Nat.mod(row,Spec.stride(problem)))

type Column is Data:
  Column{problem: Spec.Problem,channel: Nat,output: Nat,y: Nat,x: Nat,row: Nat}

def column_product(context: Column,column: Nat) -> ConvSpec.Product:
  Column{+problem,+channel,+output,+y,+x,+row} = context
  +column = column
  ConvSpec.Product{Spec.sample(problem,output,y,x,row,column),Spec.weight(problem,output,channel,row,column)}

def mapped_selected(enabled: Bool,+column: Nat,+tail: List<&2,Nat>,+context: Column) ->
  {Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,Taps.selected(enabled,column,tail),context) ==
    Spec.selected(enabled,column_product(context,column),Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,tail,context)) : List<&2,ConvSpec.Product>}:
  match enabled:
    case False{}: {==}
    case True{}: {==}

def columns_empty(count: Nat,+row: Nat,+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem,
  +unaligned: {row_aligned(problem,y,row) == False{} : Bool}) ->
  {Spec.kernel_columns(count,row,output,channel,y,x,problem) == [] : List<&2,ConvSpec.Product>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Bool,row_aligned(problem,y,row),False{},unaligned) :
        {Spec.selected(_ && Nat.is_eq(x_residue(problem,x),Nat.mod(rest,Spec.stride(problem))),column_product(Column{problem,channel,output,y,x,row},rest),Spec.kernel_columns(rest,row,output,channel,y,x,problem)) == [] : List<&2,ConvSpec.Product>}
      columns_empty(rest,row,output,channel,y,x,problem,unaligned)

def columns_filtered(count: Nat,+row: Nat,+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem,
  +aligned: {row_aligned(problem,y,row) == True{} : Bool}) ->
  {Spec.kernel_columns(count,row,output,channel,y,x,problem) ==
    Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,Taps.filtered(count,x_residue(problem,x),Spec.stride(problem)),Column{problem,channel,output,y,x,row}) : List<&2,ConvSpec.Product>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +context = {Column{problem,channel,output,y,x,row} : Column}
      +enabled = Nat.is_eq(x_residue(problem,x),Nat.mod(rest,Spec.stride(problem)))
      %Equal.sym(Bool,row_aligned(problem,y,row),True{},aligned) :
        {Spec.selected(_ && enabled,column_product(context,rest),Spec.kernel_columns(rest,row,output,channel,y,x,problem)) ==
          Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,Taps.selected(enabled,rest,Taps.filtered(rest,x_residue(problem,x),Spec.stride(problem))),context) : List<&2,ConvSpec.Product>}
      %Equal.sym(List<&2,ConvSpec.Product>,Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,Taps.selected(enabled,rest,Taps.filtered(rest,x_residue(problem,x),Spec.stride(problem))),context),
        Spec.selected(enabled,column_product(context,rest),Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,Taps.filtered(rest,x_residue(problem,x),Spec.stride(problem)),context)),mapped_selected(enabled,rest,Taps.filtered(rest,x_residue(problem,x),Spec.stride(problem)),context)) :
        {Spec.selected(enabled,column_product(context,rest),Spec.kernel_columns(rest,row,output,channel,y,x,problem)) == _ : List<&2,ConvSpec.Product>}
      %Equal.sym(List<&2,ConvSpec.Product>,Spec.kernel_columns(rest,row,output,channel,y,x,problem),
        Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,Taps.filtered(rest,x_residue(problem,x),Spec.stride(problem)),context),columns_filtered(rest,row,output,channel,y,x,problem,aligned)) :
        {Spec.selected(enabled,column_product(context,rest),_) ==
          Spec.selected(enabled,column_product(context,rest),Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,Taps.filtered(rest,x_residue(problem,x),Spec.stride(problem)),context)) : List<&2,ConvSpec.Product>}
      {==}

def phase_columns(+row: Nat,+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem) -> List<&2,ConvSpec.Product>:
  Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,
    Taps.descending(Taps.count(Spec.kernel_size(problem),x_residue(problem,x),Spec.stride(problem)),x_residue(problem,x),Spec.stride(problem)),Column{problem,channel,output,y,x,row})

def columns(+row: Nat,+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem,
  +positive: {Nat.is_lt(0n,Spec.stride(problem)) == True{} : Bool},aligned: {row_aligned(problem,y,row) == True{} : Bool}) ->
  {Spec.kernel_columns(Spec.kernel_size(problem),row,output,channel,y,x,problem) == phase_columns(row,output,channel,y,x,problem) : List<&2,ConvSpec.Product>}:
  %Taps.filtered_count(Spec.kernel_size(problem),Spec.stride(problem),x_residue(problem,x),positive,DivisionOrder.remainder_small(Nat.add(x,Spec.padding(problem)),Spec.stride(problem),positive)) :
    {Spec.kernel_columns(Spec.kernel_size(problem),row,output,channel,y,x,problem) == Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,_,Column{problem,channel,output,y,x,row}) : List<&2,ConvSpec.Product>}
  columns_filtered(Spec.kernel_size(problem),row,output,channel,y,x,problem,aligned)

def rows_from_keys(keys: List<&2,Nat>,+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem) -> List<&2,ConvSpec.Product>:
  match keys:
    case []: []
    case row <> rest: ConvSpec.cat_products(Spec.kernel_columns(Spec.kernel_size(problem),row,output,channel,y,x,problem),rows_from_keys(rest,output,channel,y,x,problem))

def row_selected(enabled: Bool,+rest: Nat,+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem,
  +compared: {row_aligned(problem,y,rest) == enabled : Bool},
  previous: {Spec.kernel_rows(rest,output,channel,y,x,problem) == rows_from_keys(Taps.filtered(rest,y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem) : List<&2,ConvSpec.Product>}) ->
  {Spec.kernel_rows(1n+rest,output,channel,y,x,problem) == rows_from_keys(Taps.selected(enabled,rest,Taps.filtered(rest,y_residue(problem,y),Spec.stride(problem))),output,channel,y,x,problem) : List<&2,ConvSpec.Product>}:
  match enabled:
    case False{}:
      %Equal.sym(List<&2,ConvSpec.Product>,Spec.kernel_columns(Spec.kernel_size(problem),rest,output,channel,y,x,problem),[],columns_empty(Spec.kernel_size(problem),rest,output,channel,y,x,problem,compared)) :
        {ConvSpec.cat_products(_,Spec.kernel_rows(rest,output,channel,y,x,problem)) == rows_from_keys(Taps.filtered(rest,y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem) : List<&2,ConvSpec.Product>}
      previous
    case True{}:
      Equal.cong(List<&2,ConvSpec.Product>,List<&2,ConvSpec.Product>,tail => ConvSpec.cat_products(Spec.kernel_columns(Spec.kernel_size(problem),rest,output,channel,y,x,problem),tail),
        Spec.kernel_rows(rest,output,channel,y,x,problem),rows_from_keys(Taps.filtered(rest,y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem),previous)

def rows_filtered(count: Nat,+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem) ->
  {Spec.kernel_rows(count,output,channel,y,x,problem) == rows_from_keys(Taps.filtered(count,y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem) : List<&2,ConvSpec.Product>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      row_selected(row_aligned(problem,y,rest),rest,output,channel,y,x,problem,{==},rows_filtered(rest,output,channel,y,x,problem))

def phase_rows(count: Nat,+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem) -> List<&2,ConvSpec.Product>:
  match count:
    case 0n: []
    case 1n+ +rest:
      +row = Nat.add(y_residue(problem,y),Nat.mul(rest,Spec.stride(problem)))
      ConvSpec.cat_products(phase_columns(row,output,channel,y,x,problem),phase_rows(rest,output,channel,y,x,problem))

def congruent_row(+residue: Nat,+stride: Nat,+quotient: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +small: {Nat.is_lt(residue,stride) == True{} : Bool}) ->
  {Nat.is_eq(residue,Nat.mod(Nat.add(residue,Nat.mul(quotient,stride)),stride)) == True{} : Bool}:
  %Order.commutative(Nat.mul(quotient,stride),residue) : {Nat.is_eq(residue,Nat.mod(_,stride)) == True{} : Bool}
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(quotient,stride),residue),stride),(quotient,residue),Division.quotient_remainder(quotient,residue,stride,positive,small)) :
    {Nat.is_eq(residue,Pair.snd(Nat,Nat,_)) == True{} : Bool}
  Order.equal_reflexive(residue)

def rows_descending(count: Nat,+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem,
  +positive: {Nat.is_lt(0n,Spec.stride(problem)) == True{} : Bool}) ->
  {rows_from_keys(Taps.descending(count,y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem) == phase_rows(count,output,channel,y,x,problem) : List<&2,ConvSpec.Product>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +row = Nat.add(y_residue(problem,y),Nat.mul(rest,Spec.stride(problem)))
      %Equal.sym(List<&2,ConvSpec.Product>,Spec.kernel_columns(Spec.kernel_size(problem),row,output,channel,y,x,problem),phase_columns(row,output,channel,y,x,problem),
        columns(row,output,channel,y,x,problem,positive,congruent_row(y_residue(problem,y),Spec.stride(problem),rest,positive,DivisionOrder.remainder_small(Nat.add(y,Spec.padding(problem)),Spec.stride(problem),positive)))) :
        {ConvSpec.cat_products(_,rows_from_keys(Taps.descending(rest,y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem)) == phase_rows(1n+rest,output,channel,y,x,problem) : List<&2,ConvSpec.Product>}
      Equal.cong(List<&2,ConvSpec.Product>,List<&2,ConvSpec.Product>,tail => ConvSpec.cat_products(phase_columns(row,output,channel,y,x,problem),tail),
        rows_from_keys(Taps.descending(rest,y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem),phase_rows(rest,output,channel,y,x,problem),rows_descending(rest,output,channel,y,x,problem,positive))

def rows(+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem,
  +positive: {Nat.is_lt(0n,Spec.stride(problem)) == True{} : Bool}) ->
  {Spec.kernel_rows(Spec.kernel_size(problem),output,channel,y,x,problem) == phase_rows(Taps.count(Spec.kernel_size(problem),y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem) : List<&2,ConvSpec.Product>}:
  +count = Taps.count(Spec.kernel_size(problem),y_residue(problem,y),Spec.stride(problem))
  Equal.trans(List<&2,ConvSpec.Product>,Spec.kernel_rows(Spec.kernel_size(problem),output,channel,y,x,problem),rows_from_keys(Taps.descending(count,y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem),phase_rows(count,output,channel,y,x,problem),
    %Taps.filtered_count(Spec.kernel_size(problem),Spec.stride(problem),y_residue(problem,y),positive,DivisionOrder.remainder_small(Nat.add(y,Spec.padding(problem)),Spec.stride(problem),positive)) :
      {Spec.kernel_rows(Spec.kernel_size(problem),output,channel,y,x,problem) == rows_from_keys(_,output,channel,y,x,problem) : List<&2,ConvSpec.Product>}
    rows_filtered(Spec.kernel_size(problem),output,channel,y,x,problem),rows_descending(count,output,channel,y,x,problem,positive))

def phase_outputs(count: Nat,+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem) -> List<&2,ConvSpec.Product>:
  match count:
    case 0n: []
    case 1n+rest:
      ConvSpec.cat_products(phase_rows(Taps.count(Spec.kernel_size(problem),y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem),phase_outputs(rest,1n+output,channel,y,x,problem))

def outputs(count: Nat,+output: Nat,+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem,
  +positive: {Nat.is_lt(0n,Spec.stride(problem)) == True{} : Bool}) ->
  {Spec.output_channels(count,output,channel,y,x,problem) == phase_outputs(count,output,channel,y,x,problem) : List<&2,ConvSpec.Product>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(List<&2,ConvSpec.Product>,Spec.kernel_rows(Spec.kernel_size(problem),output,channel,y,x,problem),phase_rows(Taps.count(Spec.kernel_size(problem),y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem),rows(output,channel,y,x,problem,positive)) :
        {ConvSpec.cat_products(_,Spec.output_channels(rest,1n+output,channel,y,x,problem)) == phase_outputs(1n+rest,output,channel,y,x,problem) : List<&2,ConvSpec.Product>}
      Equal.cong(List<&2,ConvSpec.Product>,List<&2,ConvSpec.Product>,tail => ConvSpec.cat_products(phase_rows(Taps.count(Spec.kernel_size(problem),y_residue(problem,y),Spec.stride(problem)),output,channel,y,x,problem),tail),
        Spec.output_channels(rest,1n+output,channel,y,x,problem),phase_outputs(rest,1n+output,channel,y,x,problem),outputs(rest,1n+output,channel,y,x,problem,positive))

def products(+channel: Nat,+y: Nat,+x: Nat,+problem: Spec.Problem,
  positive: {Nat.is_lt(0n,Spec.stride(problem)) == True{} : Bool}) ->
  {Spec.products(problem,channel,y,x) == phase_outputs(Spec.outputs(problem),0n,channel,y,x,problem) : List<&2,ConvSpec.Product>}:
  outputs(Spec.outputs(problem),0n,channel,y,x,problem,positive)

def flat_rows(context: Flat) -> Nat:
  Flat{+problem,channel,+y,x} = context
  Taps.count(Spec.kernel_size(problem),y_residue(problem,y),Spec.stride(problem))

def flat_columns(context: Flat) -> Nat:
  Flat{+problem,channel,y,+x} = context
  Taps.count(Spec.kernel_size(problem),x_residue(problem,x),Spec.stride(problem))

def flat_area(+context: Flat) -> Nat:
  Nat.mul(flat_rows(context),flat_columns(context))

def flat_index(+context: Flat,+output: Nat,+row: Nat,+column: Nat) -> Nat:
  RowMajor.index(output,flat_rows(context),flat_columns(context),row,column)

def flat_row_tap(+context: Flat,row: Nat) -> Nat:
  Flat{+problem,channel,+y,x} = context
  Nat.add(y_residue(problem,y),Nat.mul(Nat.sub(Nat.sub(flat_rows(context),1n),row),Spec.stride(problem)))

def flat_column_tap(+context: Flat,column: Nat) -> Nat:
  Flat{+problem,channel,y,+x} = context
  Nat.add(x_residue(problem,x),Nat.mul(Nat.sub(Nat.sub(flat_columns(context),1n),column),Spec.stride(problem)))

def flat_product(context: Flat,index: Nat) -> ConvSpec.Product:
  Flat{+problem,+channel,+y,+x} = context
  +index = index
  +pixel = Nat.mod(index,flat_area(context))
  +output = Nat.div(index,flat_area(context))
  +row = flat_row_tap(context,Nat.div(pixel,flat_columns(context)))
  +column = flat_column_tap(context,Nat.mod(pixel,flat_columns(context)))
  column_product(Column{problem,channel,output,y,x,row},column)

def flat_problem(context: Flat) -> Spec.Problem:
  Flat{problem,channel,y,x} = context
  problem

def flat_channel(context: Flat) -> Nat:
  Flat{problem,channel,y,x} = context
  channel

def flat_y(context: Flat) -> Nat:
  Flat{problem,channel,y,x} = context
  y

def flat_x(context: Flat) -> Nat:
  Flat{problem,channel,y,x} = context
  x

def flat_product_at(+context: Flat,+output: Nat,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,flat_rows(context)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,flat_columns(context)) == True{} : Bool}) ->
  {flat_product(context,flat_index(context,output,row,column)) ==
    column_product(Column{flat_problem(context),flat_channel(context),output,flat_y(context),flat_x(context),flat_row_tap(context,row)},flat_column_tap(context,column)) : ConvSpec.Product}:
  Flat{+problem,+channel,+y,+x} = context
  +width = flat_columns(context)
  +rows = flat_rows(context)
  +area = flat_area(context)
  +pixel = Nat.add(Nat.mul(row,width),column)
  +wp = Order.positive_bound(column,width,column_inside)
  +rp = Order.positive_bound(row,rows,row_inside)
  +ap = Order.product_positive(rows,width,rp,wp)
  +pixel_inside = RowMajor.radix(row,rows,column,width,row_inside,column_inside)
  %Equal.sym(Nat,Nat.div(flat_index(context,output,row,column),area),output,Radix.quotient(output,pixel,area,ap,pixel_inside)) :
    {column_product(Column{problem,channel,_,y,x,flat_row_tap(context,Nat.div(Nat.mod(flat_index(context,output,row,column),area),width))},flat_column_tap(context,Nat.mod(Nat.mod(flat_index(context,output,row,column),area),width))) == column_product(Column{problem,channel,output,y,x,flat_row_tap(context,row)},flat_column_tap(context,column)) : ConvSpec.Product}
  %Equal.sym(Nat,Nat.mod(flat_index(context,output,row,column),area),pixel,Radix.remainder(output,pixel,area,ap,pixel_inside)) :
    {column_product(Column{problem,channel,output,y,x,flat_row_tap(context,Nat.div(_,width))},flat_column_tap(context,Nat.mod(_,width))) == column_product(Column{problem,channel,output,y,x,flat_row_tap(context,row)},flat_column_tap(context,column)) : ConvSpec.Product}
  %Equal.sym(Nat,Nat.div(pixel,width),row,Radix.quotient(row,column,width,wp,column_inside)) :
    {column_product(Column{problem,channel,output,y,x,flat_row_tap(context,_)},flat_column_tap(context,Nat.mod(pixel,width))) == column_product(Column{problem,channel,output,y,x,flat_row_tap(context,row)},flat_column_tap(context,column)) : ConvSpec.Product}
  %Equal.sym(Nat,Nat.mod(pixel,width),column,Radix.remainder(row,column,width,wp,column_inside)) :
    {column_product(Column{problem,channel,output,y,x,flat_row_tap(context,row)},flat_column_tap(context,_)) == column_product(Column{problem,channel,output,y,x,flat_row_tap(context,row)},flat_column_tap(context,column)) : ConvSpec.Product}
  {==}

def flat_map_cat(+context: Flat,left: List<&2,Nat>,+right: List<&2,Nat>) ->
  {Ordered.map_keys(~Nat,~ConvSpec.Product,~Flat,~flat_product,OrderedMatrix.cat(Nat,left,right),context) ==
    ConvSpec.cat_products(Ordered.map_keys(~Nat,~ConvSpec.Product,~Flat,~flat_product,left,context),Ordered.map_keys(~Nat,~ConvSpec.Product,~Flat,~flat_product,right,context)) : List<&2,ConvSpec.Product>}:
  match left:
    case []: {==}
    case +head <> +tail:
      Equal.cong(List<&2,ConvSpec.Product>,List<&2,ConvSpec.Product>,rest => flat_product(context,head) <> rest,
        Ordered.map_keys(~Nat,~ConvSpec.Product,~Flat,~flat_product,OrderedMatrix.cat(Nat,tail,right),context),
        ConvSpec.cat_products(Ordered.map_keys(~Nat,~ConvSpec.Product,~Flat,~flat_product,tail,context),Ordered.map_keys(~Nat,~ConvSpec.Product,~Flat,~flat_product,right,context)),flat_map_cat(context,tail,right))

def flat_range(+context: Flat,count: Nat,start: Nat) -> List<&2,ConvSpec.Product>:
  Ordered.map_keys(~Nat,~ConvSpec.Product,~Flat,~flat_product,Ranges.range(count,start),context)

def flat_split(+context: Flat,+left: Nat,+right: Nat,+start: Nat) ->
  {flat_range(context,Nat.add(left,right),start) == ConvSpec.cat_products(flat_range(context,left,start),flat_range(context,right,Nat.add(start,left))) : List<&2,ConvSpec.Product>}:
  %Equal.sym(List<&2,Nat>,Ranges.range(Nat.add(left,right),start),OrderedMatrix.cat(Nat,Ranges.range(left,start),Ranges.range(right,Nat.add(start,left))),Ranges.split(left,right,start)) :
    {Ordered.map_keys(~Nat,~ConvSpec.Product,~Flat,~flat_product,_,context) == ConvSpec.cat_products(flat_range(context,left,start),flat_range(context,right,Nat.add(start,left))) : List<&2,ConvSpec.Product>}
  flat_map_cat(context,Ranges.range(left,start),Ranges.range(right,Nat.add(start,left)))

def flat_next_column(+context: Flat,+output: Nat,+row: Nat,+column: Nat) ->
  {flat_index(context,output,row,1n+column) == 1n+flat_index(context,output,row,column) : Nat}:
  Equal.trans(Nat,flat_index(context,output,row,1n+column),Nat.add(Nat.mul(output,flat_area(context)),1n+Nat.add(Nat.mul(row,flat_columns(context)),column)),1n+flat_index(context,output,row,column),
    Equal.cong(Nat,Nat,pixel => Nat.add(Nat.mul(output,flat_area(context)),pixel),Nat.add(Nat.mul(row,flat_columns(context)),1n+column),1n+Nat.add(Nat.mul(row,flat_columns(context)),column),Addition.successor_right(Nat.mul(row,flat_columns(context)),column)),
    Addition.successor_right(Nat.mul(output,flat_area(context)),Nat.add(Nat.mul(row,flat_columns(context)),column)))

def flat_columns_order(count: Nat,+offset: Nat,+output: Nat,+row: Nat,+context: Flat,
  +row_inside: {Nat.is_lt(row,flat_rows(context)) == True{} : Bool},
  +position: {Nat.add(offset,count) == flat_columns(context) : Nat}) ->
  {flat_range(context,count,flat_index(context,output,row,offset)) ==
    Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,Taps.descending(count,x_residue(flat_problem(context),flat_x(context)),Spec.stride(flat_problem(context))),Column{flat_problem(context),flat_channel(context),output,flat_y(context),flat_x(context),flat_row_tap(context,row)}) : List<&2,ConvSpec.Product>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Flat{+problem,+channel,+y,+x} = context
      +tap = Nat.add(x_residue(problem,flat_x(context)),Nat.mul(rest,Spec.stride(problem)))
      +column = {Column{problem,flat_channel(context),output,flat_y(context),flat_x(context),flat_row_tap(context,row)} : Column}
      +inside = Order.below_complement(offset,rest,flat_columns(context),position)
      +reflected = Order.reflected_complement(offset,rest,flat_columns(context),position)
      +term = Equal.trans(ConvSpec.Product,flat_product(context,flat_index(context,output,row,offset)),column_product(column,flat_column_tap(context,offset)),column_product(column,tap),flat_product_at(context,output,row,offset,row_inside,inside),
        Equal.cong(Nat,ConvSpec.Product,column_product(column),flat_column_tap(context,offset),tap,Equal.cong(Nat,Nat,digit => Nat.add(x_residue(problem,flat_x(context)),Nat.mul(digit,Spec.stride(problem))),Nat.sub(Nat.sub(flat_columns(context),1n),offset),rest,reflected)))
      +advanced = Equal.trans(Nat,Nat.add(1n+offset,rest),Nat.add(offset,1n+rest),flat_columns(context),Equal.sym(Nat,Nat.add(offset,1n+rest),1n+Nat.add(offset,rest),Addition.successor_right(offset,rest)),position)
      %Equal.sym(ConvSpec.Product,flat_product(context,flat_index(context,output,row,offset)),column_product(column,tap),term) :
        {_ <> flat_range(context,rest,1n+flat_index(context,output,row,offset)) == column_product(column,tap) <>
          Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,Taps.descending(rest,x_residue(problem,flat_x(context)),Spec.stride(problem)),column) : List<&2,ConvSpec.Product>}
      %flat_next_column(context,output,row,offset) :
        {column_product(column,tap) <> flat_range(context,rest,_) == column_product(column,tap) <>
          Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,Taps.descending(rest,x_residue(problem,flat_x(context)),Spec.stride(problem)),column) : List<&2,ConvSpec.Product>}
      Equal.cong(List<&2,ConvSpec.Product>,List<&2,ConvSpec.Product>,tail => column_product(column,tap) <> tail,
        flat_range(context,rest,flat_index(context,output,row,1n+offset)),Ordered.map_keys(~Nat,~ConvSpec.Product,~Column,~column_product,Taps.descending(rest,x_residue(problem,flat_x(context)),Spec.stride(problem)),column),flat_columns_order(rest,1n+offset,output,row,context,row_inside,advanced))

def flat_next_row(+context: Flat,+output: Nat,+row: Nat) ->
  {flat_index(context,output,1n+row,0n) == Nat.add(flat_index(context,output,row,0n),flat_columns(context)) : Nat}:
  +prefix = Nat.mul(output,flat_area(context))
  +current = Nat.mul(row,flat_columns(context))
  +width = flat_columns(context)
  Equal.trans(Nat,flat_index(context,output,1n+row,0n),Nat.add(prefix,Nat.add(width,current)),Nat.add(flat_index(context,output,row,0n),width),
    Equal.cong(Nat,Nat,pixel => Nat.add(prefix,pixel),Nat.add(Nat.add(width,current),0n),Nat.add(width,current),Addition.zero_right(Nat.add(width,current))),
    Equal.trans(Nat,Nat.add(prefix,Nat.add(width,current)),Nat.add(prefix,Nat.add(current,width)),Nat.add(flat_index(context,output,row,0n),width),
      Equal.cong(Nat,Nat,pixel => Nat.add(prefix,pixel),Nat.add(width,current),Nat.add(current,width),Order.commutative(width,current)),
      Equal.trans(Nat,Nat.add(prefix,Nat.add(current,width)),Nat.add(Nat.add(prefix,current),width),Nat.add(flat_index(context,output,row,0n),width),Equal.sym(Nat,Nat.add(Nat.add(prefix,current),width),Nat.add(prefix,Nat.add(current,width)),Addition.associative(prefix,current,width)),
        Equal.cong(Nat,Nat,pixel => Nat.add(Nat.add(prefix,pixel),width),current,Nat.add(current,0n),Equal.sym(Nat,Nat.add(current,0n),current,Addition.zero_right(current))))))

def flat_rows_order(count: Nat,+offset: Nat,+output: Nat,+context: Flat,
  +position: {Nat.add(offset,count) == flat_rows(context) : Nat}) ->
  {flat_range(context,Nat.mul(count,flat_columns(context)),flat_index(context,output,offset,0n)) ==
    phase_rows(count,output,flat_channel(context),flat_y(context),flat_x(context),flat_problem(context)) : List<&2,ConvSpec.Product>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Flat{+problem,+channel,+y,+x} = context
      +width = flat_columns(context)
      +start = flat_index(context,output,offset,0n)
      +row = Nat.add(y_residue(problem,y),Nat.mul(rest,Spec.stride(problem)))
      +inside = Order.below_complement(offset,rest,flat_rows(context),position)
      +reflected = Order.reflected_complement(offset,rest,flat_rows(context),position)
      +head = Equal.trans(List<&2,ConvSpec.Product>,flat_range(context,width,start),phase_columns(flat_row_tap(context,offset),output,channel,y,x,problem),phase_columns(row,output,channel,y,x,problem),
        flat_columns_order(width,0n,output,offset,context,inside,{==}),
        Equal.cong(Nat,List<&2,ConvSpec.Product>,row => phase_columns(row,output,channel,y,x,problem),flat_row_tap(context,offset),row,Equal.cong(Nat,Nat,digit => Nat.add(y_residue(problem,y),Nat.mul(digit,Spec.stride(problem))),Nat.sub(Nat.sub(flat_rows(context),1n),offset),rest,reflected)))
      +advanced = Equal.trans(Nat,Nat.add(1n+offset,rest),Nat.add(offset,1n+rest),flat_rows(context),Equal.sym(Nat,Nat.add(offset,1n+rest),1n+Nat.add(offset,rest),Addition.successor_right(offset,rest)),position)
      %Equal.sym(List<&2,ConvSpec.Product>,flat_range(context,Nat.add(width,Nat.mul(rest,width)),start),ConvSpec.cat_products(flat_range(context,width,start),flat_range(context,Nat.mul(rest,width),Nat.add(start,width))),flat_split(context,width,Nat.mul(rest,width),start)) :
        {_ == ConvSpec.cat_products(phase_columns(row,output,channel,y,x,problem),phase_rows(rest,output,channel,y,x,problem)) : List<&2,ConvSpec.Product>}
      %Equal.sym(List<&2,ConvSpec.Product>,flat_range(context,width,start),phase_columns(row,output,channel,y,x,problem),head) :
        {ConvSpec.cat_products(_,flat_range(context,Nat.mul(rest,width),Nat.add(start,width))) == ConvSpec.cat_products(phase_columns(row,output,channel,y,x,problem),phase_rows(rest,output,channel,y,x,problem)) : List<&2,ConvSpec.Product>}
      %flat_next_row(context,output,offset) :
        {ConvSpec.cat_products(phase_columns(row,output,channel,y,x,problem),flat_range(context,Nat.mul(rest,width),_)) == ConvSpec.cat_products(phase_columns(row,output,channel,y,x,problem),phase_rows(rest,output,channel,y,x,problem)) : List<&2,ConvSpec.Product>}
      Equal.cong(List<&2,ConvSpec.Product>,List<&2,ConvSpec.Product>,tail => ConvSpec.cat_products(phase_columns(row,output,channel,y,x,problem),tail),
        flat_range(context,Nat.mul(rest,width),flat_index(context,output,1n+offset,0n)),phase_rows(rest,output,channel,y,x,problem),flat_rows_order(rest,1n+offset,output,context,advanced))

def flat_next_output(+context: Flat,+output: Nat) ->
  {flat_index(context,1n+output,0n,0n) == Nat.add(flat_index(context,output,0n,0n),flat_area(context)) : Nat}:
  +area = flat_area(context)
  +base = Nat.mul(output,area)
  Equal.trans(Nat,flat_index(context,1n+output,0n,0n),Nat.add(area,base),Nat.add(flat_index(context,output,0n,0n),area),Addition.zero_right(Nat.add(area,base)),
    Equal.trans(Nat,Nat.add(area,base),Nat.add(base,area),Nat.add(flat_index(context,output,0n,0n),area),Order.commutative(area,base),
      Equal.cong(Nat,Nat,prefix => Nat.add(prefix,area),base,Nat.add(base,0n),Equal.sym(Nat,Nat.add(base,0n),base,Addition.zero_right(base)))))

def flat_outputs_order(count: Nat,+output: Nat,+context: Flat) ->
  {flat_range(context,Nat.mul(count,flat_area(context)),flat_index(context,output,0n,0n)) ==
    phase_outputs(count,output,flat_channel(context),flat_y(context),flat_x(context),flat_problem(context)) : List<&2,ConvSpec.Product>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Flat{+problem,+channel,+y,+x} = context
      +area = flat_area(context)
      +start = flat_index(context,output,0n,0n)
      +head = phase_rows(flat_rows(context),output,channel,y,x,problem)
      %Equal.sym(List<&2,ConvSpec.Product>,flat_range(context,Nat.add(area,Nat.mul(rest,area)),start),ConvSpec.cat_products(flat_range(context,area,start),flat_range(context,Nat.mul(rest,area),Nat.add(start,area))),flat_split(context,area,Nat.mul(rest,area),start)) :
        {_ == ConvSpec.cat_products(head,phase_outputs(rest,1n+output,channel,y,x,problem)) : List<&2,ConvSpec.Product>}
      %Equal.sym(List<&2,ConvSpec.Product>,flat_range(context,area,start),head,flat_rows_order(flat_rows(context),0n,output,context,{==})) :
        {ConvSpec.cat_products(_,flat_range(context,Nat.mul(rest,area),Nat.add(start,area))) == ConvSpec.cat_products(head,phase_outputs(rest,1n+output,channel,y,x,problem)) : List<&2,ConvSpec.Product>}
      %flat_next_output(context,output) :
        {ConvSpec.cat_products(head,flat_range(context,Nat.mul(rest,area),_)) == ConvSpec.cat_products(head,phase_outputs(rest,1n+output,channel,y,x,problem)) : List<&2,ConvSpec.Product>}
      Equal.cong(List<&2,ConvSpec.Product>,List<&2,ConvSpec.Product>,tail => ConvSpec.cat_products(head,tail),flat_range(context,Nat.mul(rest,area),flat_index(context,1n+output,0n,0n)),phase_outputs(rest,1n+output,channel,y,x,problem),flat_outputs_order(rest,1n+output,context))

def flat_count(+context: Flat) -> Nat:
  Nat.mul(Spec.outputs(flat_problem(context)),flat_area(context))

def flat_products(+context: Flat) -> List<&2,ConvSpec.Product>:
  flat_range(context,flat_count(context),0n)

def flat_spec_products(+context: Flat,+positive: {Nat.is_lt(0n,Spec.stride(flat_problem(context))) == True{} : Bool}) ->
  {flat_products(context) == Spec.products(flat_problem(context),flat_channel(context),flat_y(context),flat_x(context)) : List<&2,ConvSpec.Product>}:
  Flat{+problem,+channel,+y,+x} = context
  Equal.trans(List<&2,ConvSpec.Product>,flat_products(context),phase_outputs(Spec.outputs(problem),0n,channel,y,x,problem),Spec.products(problem,channel,y,x),flat_outputs_order(Spec.outputs(problem),0n,context),
    Equal.sym(List<&2,ConvSpec.Product>,Spec.products(problem,channel,y,x),phase_outputs(Spec.outputs(problem),0n,channel,y,x,problem),products(channel,y,x,problem,positive)))

def empty_value(+context: Flat,+positive: {Nat.is_lt(0n,Spec.stride(flat_problem(context))) == True{} : Bool},
  +empty: {flat_count(context) == 0n : Nat}) ->
  {Spec.value(flat_problem(context),flat_channel(context),flat_y(context),flat_x(context)) == (0.0 + 0.0 : F32) : F32}:
  %flat_spec_products(context,positive) : {(ConvSpec.sum(ConvSpec.multiply(_),0.0) + 0.0 : F32) == (0.0 + 0.0 : F32) : F32}
  %Equal.sym(Nat,flat_count(context),0n,empty) : {(ConvSpec.sum(ConvSpec.multiply(flat_range(context,_,0n)),0.0) + 0.0 : F32) == (0.0 + 0.0 : F32) : F32}
  {==}
