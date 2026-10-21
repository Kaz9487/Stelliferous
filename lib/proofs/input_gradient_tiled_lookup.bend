# Table-indirected tile addresses use the filled row-major slot.
import Base
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../kernel_input_gradient.bend as Kernel
import ./input_gradient_tiled_fill.bend as Fill
import ./input_gradient_tiled_jobs.bend as Jobs
import ./input_gradient_tiled_reduction.bend as Reduction
import ./input_gradient_tiled_offsets.bend as Offsets
import ./input_gradient_tiled_weights.bend as Packing
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_chunk_allocation.bend as Allocation
import ./input_gradient_tiled_chunk_geometry.bend as Chunks
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_band_geometry.bend as Band

import ./input_gradient_tiled_bounds.bend as Groups
import ./input_gradient_tiled_refinement.bend as Filling
import ./storage_tree_model.bend as Tree
import ./row_major_index.bend as RowMajor
import ./binary_radix.bend as Radix
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic

type Query is Data:
  Query{fill: Fill.Query,row: U32,group: U32,output: U32,shift: U32}
def fill(query: Query) -> Fill.Query:
  Query{fill,row,group,output,shift} = query
  fill
def row(query: Query) -> U32:
  Query{fill,row,group,output,shift} = query
  row
def group(query: Query) -> U32:
  Query{fill,row,group,output,shift} = query
  group
def output(query: Query) -> U32:
  Query{fill,row,group,output,shift} = query
  output
def shift(query: Query) -> U32:
  Query{fill,row,group,output,shift} = query
  shift
def span(+query: Query) -> U32:
  Fill.span(Fill.filling(fill(query)))
def groups(+query: Query) -> U32:
  Fill.groups(Fill.filling(fill(query)))
def shifts(+query: Query) -> U32:
  Fill.shifts(Fill.filling(fill(query)))
def block(+query: Query) -> U32:
  Fill.block(Fill.filling(fill(query)))
def block_size(+query: Query) -> Nat:
  Fill.block_size(Fill.filling(fill(query)))
def group_natural(+query: Query) -> Nat:
  Nat.add(Nat.mul(U32.to_nat(row(query)),U32.to_nat(groups(query))),U32.to_nat(group(query)))
def cell_natural(+query: Query) -> Nat:
  Nat.add(Nat.mul(U32.to_nat(output(query)),U32.to_nat(shifts(query))),U32.to_nat(shift(query)))
def address_natural(+query: Query) -> Nat:
  Nat.add(Nat.mul(group_natural(query),block_size(query)),cell_natural(query))
def group_word(+query: Query) -> U32:
  (row(query) * groups(query) + group(query) : U32)
def cell_word(+query: Query) -> U32:
  (output(query) * shifts(query) + shift(query) : U32)
def address(+query: Query) -> U32:
  (group_word(query) * block(query) + cell_word(query) : U32)

type Fields<-query: Query> is Data:
  Fields{position: {Fill.position(fill(query)) == Nat.add(Nat.mul(U32.to_nat(row(query)),U32.to_nat(span(query))),U32.to_nat(group(query))) : Nat},
    output: {Fill.channel(fill(query)) == U32.to_nat(output(query)) : Nat},shift: {Fill.shift(fill(query)) == U32.to_nat(shift(query)) : Nat},
    group_inside: {Nat.is_lt(U32.to_nat(group(query)),U32.to_nat(span(query))) == True{} : Bool}}
def decoded_group(+query: Query,+fields: Fields<query>,+positive: {Nat.is_lt(0n,U32.to_nat(span(query))) == True{} : Bool}) ->
  {Groups.group_position(Fill.position(fill(query)),U32.to_nat(span(query)),U32.to_nat(groups(query))) == group_natural(query) : Nat}:
  Fields{+position,output,shift,+inside} = fields
  +rn = U32.to_nat(row(query))
  +gn = U32.to_nat(group(query))
  +width = U32.to_nat(span(query))
  +index = Nat.add(Nat.mul(rn,width),gn)
  +cg = U32.to_nat(groups(query))
  %Equal.sym(Nat,Fill.position(fill(query)),index,position) : {Groups.group_position(_,width,cg) == group_natural(query) : Nat}
  %Equal.sym(Nat,Nat.div(index,width),rn,Radix.quotient(rn,gn,width,positive,inside)) : {Nat.add(Nat.mul(_,cg),Nat.mod(index,width)) == group_natural(query) : Nat}
  %Equal.sym(Nat,Nat.mod(index,width),gn,Radix.remainder(rn,gn,width,positive,inside)) : {Nat.add(Nat.mul(rn,cg),_) == group_natural(query) : Nat}
  {==}
def cell_inside(+query: Query,+fields: Fields<query>,+facts: Fill.Facts<fill(query)>) ->
  {Nat.is_lt(cell_natural(query),block_size(query)) == True{} : Bool}:
  Fields{position,+oe,+se,group_inside} = fields
  Fill.Facts{capacity_ok,span_positive,span_ordered,double_span,room,position_inside,+output_inside,+shift_inside} = facts
  +outputs = U32.to_nat(Fill.outputs(Fill.filling(fill(query))))
  +shifts = U32.to_nat(shifts(query))
  +product = Nat.mul(outputs,shifts)
  +inside = RowMajor.radix(U32.to_nat(output(query)),outputs,U32.to_nat(shift(query)),shifts,
    %oe : {Nat.is_lt(_,outputs) == True{} : Bool} output_inside,%se : {Nat.is_lt(_,shifts) == True{} : Bool} shift_inside)
  Bounds.nat_trans(cell_natural(query),product,block_size(query),inside,Bounds.strict_le(product,block_size(query),Order.below_sum(product,0n)))
def address_room(+query: Query,+fields: Fields<query>,+facts: Fill.Facts<fill(query)>) ->
  {Nat.is_le(address_natural(query),U32.to_nat(Fill.capacity(fill(query)))) == True{} : Bool}:
  Fill.Facts{capacity_ok,+span_positive,+span_ordered,double_span,+room,+position_inside,output_inside,shift_inside} = facts
  +count = Fill.group_count(fill(query))
  +group_inside = {%decoded_group(query,fields,span_positive) : {Nat.is_lt(_,count) == True{} : Bool}
    Groups.group_inside(Fill.position(fill(query)),Fill.rows(fill(query)),U32.to_nat(span(query)),U32.to_nat(groups(query)),span_positive,span_ordered,position_inside) : {Nat.is_lt(group_natural(query),count) == True{} : Bool}}
  Bounds.strict_le(address_natural(query),U32.to_nat(Fill.capacity(fill(query))),Bounds.nat_trans(address_natural(query),Fill.tile_count(fill(query)),U32.to_nat(Fill.capacity(fill(query))),RowMajor.radix(group_natural(query),count,cell_natural(query),block_size(query),group_inside,cell_inside(query,fields,facts)),room))
def group_room(+query: Query,+fields: Fields<query>,+facts: Fill.Facts<fill(query)>) ->
  {Nat.is_le(group_natural(query),U32.to_nat(Fill.capacity(fill(query)))) == True{} : Bool}:
  Order.transitive(group_natural(query),Nat.mul(group_natural(query),block_size(query)),U32.to_nat(Fill.capacity(fill(query))),RowMajor.product_covers(group_natural(query),block_size(query),Fill.block_positive(Fill.filling(fill(query)))),
    Order.transitive(Nat.mul(group_natural(query),block_size(query)),address_natural(query),U32.to_nat(Fill.capacity(fill(query))),Order.le_add(Nat.mul(group_natural(query),block_size(query)),cell_natural(query)),address_room(query,fields,facts)))
def cell_room(+query: Query,+fields: Fields<query>,+facts: Fill.Facts<fill(query)>) ->
  {Nat.is_le(cell_natural(query),U32.to_nat(Fill.capacity(fill(query)))) == True{} : Bool}:
  Order.transitive(cell_natural(query),address_natural(query),U32.to_nat(Fill.capacity(fill(query))),
    %Order.commutative(cell_natural(query),Nat.mul(group_natural(query),block_size(query))) : {Nat.is_le(cell_natural(query),_) == True{} : Bool} Order.le_add(cell_natural(query),Nat.mul(group_natural(query),block_size(query))),address_room(query,fields,facts))
def group_word_exact(+query: Query,+fields: Fields<query>,+facts: Fill.Facts<fill(query)>) -> {U32.to_nat(group_word(query)) == group_natural(query) : Nat}:
  Fill.Facts{+capacity_ok,span_positive,span_ordered,double_span,room,position_inside,output_inside,shift_inside} = facts
  Arithmetic.multiply_add_exact(row(query),groups(query),group(query),Fill.capacity(fill(query)),capacity_ok,group_room(query,fields,facts))
def cell_word_exact(+query: Query,+fields: Fields<query>,+facts: Fill.Facts<fill(query)>) -> {U32.to_nat(cell_word(query)) == cell_natural(query) : Nat}:
  Fill.Facts{+capacity_ok,span_positive,span_ordered,double_span,room,position_inside,output_inside,shift_inside} = facts
  Arithmetic.multiply_add_exact(output(query),shifts(query),shift(query),Fill.capacity(fill(query)),capacity_ok,cell_room(query,fields,facts))

def address_exact(+query: Query,+fields: Fields<query>,+facts: Fill.Facts<fill(query)>) -> {U32.to_nat(address(query)) == address_natural(query) : Nat}:
  Fill.Facts{+capacity_ok,span_positive,span_ordered,double_span,room,position_inside,output_inside,shift_inside} = facts
  +gn = group_natural(query)
  +bn = block_size(query)
  +cn = cell_natural(query)
  +gw = group_word(query)
  +bw = block(query)
  +cw = cell_word(query)
  +capacity = Fill.capacity(fill(query))
  +ge = group_word_exact(query,fields,facts)
  +be = Fill.block_exact(fill(query),facts)
  +ce = cell_word_exact(query,fields,facts)
  +room = {%Equal.sym(Nat,U32.to_nat(gw),gn,ge) : {Nat.is_le(Nat.add(Nat.mul(_,U32.to_nat(bw)),U32.to_nat(cw)),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(bw),bn,be) : {Nat.is_le(Nat.add(Nat.mul(gn,_),U32.to_nat(cw)),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(cw),cn,ce) : {Nat.is_le(Nat.add(Nat.mul(gn,bn),_),U32.to_nat(capacity)) == True{} : Bool} address_room(query,fields,facts) : {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(gw),U32.to_nat(bw)),U32.to_nat(cw)),U32.to_nat(capacity)) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat(address(query)),Nat.add(Nat.mul(U32.to_nat(gw),U32.to_nat(bw)),U32.to_nat(cw)),address_natural(query),Arithmetic.multiply_add_exact(gw,bw,cw,capacity,capacity_ok,room),
    Equal.trans(Nat,Nat.add(Nat.mul(U32.to_nat(gw),U32.to_nat(bw)),U32.to_nat(cw)),Nat.add(Nat.mul(gn,U32.to_nat(bw)),U32.to_nat(cw)),address_natural(query),Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,U32.to_nat(bw)),U32.to_nat(cw)),U32.to_nat(gw),gn,ge),
      Equal.trans(Nat,Nat.add(Nat.mul(gn,U32.to_nat(bw)),U32.to_nat(cw)),Nat.add(Nat.mul(gn,bn),U32.to_nat(cw)),address_natural(query),Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(gn,value),U32.to_nat(cw)),U32.to_nat(bw),bn,be),Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(gn,bn),value),U32.to_nat(cw),cn,ce))))
def fill_address_finish(+query: Query,+fields: Fields<query>,+state: Filling.Owners,+evidence: Fill.Frame<fill(query),state>,
  +output_equal: {Fill.channel(fill(query)) == U32.to_nat(output(query)) : Nat},
  +shift_equal: {Fill.shift(fill(query)) == U32.to_nat(shift(query)) : Nat},+facts: Fill.Facts<fill(query)>) ->
  {U32.to_nat(Fill.address(fill(query))) == address_natural(query) : Nat}:
  Fill.Facts{capacity_ok,+span_positive,span_ordered,double_span,room,position_inside,output_inside,shift_inside} = facts
  +fq = fill(query)
  +group = Groups.group_position(Fill.position(fq),U32.to_nat(span(query)),U32.to_nat(groups(query)))
  +gn = group_natural(query)
  +bn = block_size(query)
  +sn = U32.to_nat(shifts(query))
  +cell = Nat.add(Nat.mul(Fill.channel(fq),sn),Fill.shift(fq))
  +oe = output_equal
  +se = shift_equal
  Equal.trans(Nat,U32.to_nat(Fill.address(fq)),Nat.add(Nat.mul(group,bn),cell),address_natural(query),Fill.address_exact(fq,state,evidence),
    Equal.trans(Nat,Nat.add(Nat.mul(group,bn),cell),Nat.add(Nat.mul(gn,bn),cell),address_natural(query),Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,bn),cell),group,gn,decoded_group(query,fields,span_positive)),
      Equal.trans(Nat,Nat.add(Nat.mul(gn,bn),cell),Nat.add(Nat.mul(gn,bn),Nat.add(Nat.mul(U32.to_nat(output(query)),sn),Fill.shift(fq))),address_natural(query),Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(gn,bn),Nat.add(Nat.mul(value,sn),Fill.shift(fq))),Fill.channel(fq),U32.to_nat(output(query)),oe),Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(gn,bn),Nat.add(Nat.mul(U32.to_nat(output(query)),sn),value)),Fill.shift(fq),U32.to_nat(shift(query)),se))))
def fill_address_exact(+query: Query,+fields: Fields<query>,+state: Filling.Owners,+evidence: Fill.Frame<fill(query),state>) ->
  {U32.to_nat(Fill.address(fill(query))) == address_natural(query) : Nat}:
  Fields{position,+oe,+se,inside} = fields
  +facts = Fill.frame_facts(fill(query),state,evidence)
  fill_address_finish(query,fields,state,evidence,oe,se,facts)
def addresses_match(+query: Query,+fields: Fields<query>,+state: Filling.Owners,+evidence: Fill.Frame<fill(query),state>) ->
  {Fill.address(fill(query)) == address(query) : U32}:
  Arithmetic.to_nat_injective(Fill.address(fill(query)),address(query),Equal.trans(Nat,U32.to_nat(Fill.address(fill(query))),address_natural(query),U32.to_nat(address(query)),fill_address_exact(query,fields,state,evidence),Equal.sym(Nat,U32.to_nat(address(query)),address_natural(query),address_exact(query,fields,Fill.frame_facts(fill(query),state,evidence)))))
def readback(+query: Query,+fields: Fields<query>,+state: Filling.Owners,+evidence: Fill.Frame<fill(query),state>) ->
  {Tree.read(Tile.Tile8,Fill.tree(Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(fill(query)),0,Fill.filling(fill(query)),state)),address(query)) == Fill.wanted(fill(query)) : Tile.Tile8}:
  +filled = Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(fill(query)),0,Fill.filling(fill(query)),state)
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Fill.tree(filled),address(query)),Fill.read(fill(query),filled),Fill.wanted(fill(query)),Equal.cong(U32,Tile.Tile8,Tree.read(Tile.Tile8,Fill.tree(filled)),address(query),Fill.address(fill(query)),Equal.sym(U32,Fill.address(fill(query)),address(query),addresses_match(query,fields,state,evidence))),Fill.readback(fill(query),state,evidence))

type Read is Data:
  Read{source_row: U32,line: U32,tap_row: U32,chunk_groups: U32,block: U32,group: U32,output: U32,shifts: U32,tap_column: U32,first_shift: U32}
def read_row(read: Read) -> U32:
  Read{row,line,tap,cg,b,group,output,s,column,first} = read
  row
def read_line(read: Read) -> U32:
  Read{row,line,tap,cg,b,group,output,s,column,first} = read
  line
def read_tap_row(read: Read) -> U32:
  Read{row,line,tap,cg,b,group,output,s,column,first} = read
  tap
def read_groups(read: Read) -> U32:
  Read{row,line,tap,cg,b,group,output,s,column,first} = read
  cg
def read_block(read: Read) -> U32:
  Read{row,line,tap,cg,b,group,output,s,column,first} = read
  b
def read_group(read: Read) -> U32:
  Read{row,line,tap,cg,b,group,output,s,column,first} = read
  group
def read_output(read: Read) -> U32:
  Read{row,line,tap,cg,b,group,output,s,column,first} = read
  output
def read_shifts(read: Read) -> U32:
  Read{row,line,tap,cg,b,group,output,s,column,first} = read
  s
def read_tap_column(read: Read) -> U32:
  Read{row,line,tap,cg,b,group,output,s,column,first} = read
  column
def read_first_shift(read: Read) -> U32:
  Read{row,line,tap,cg,b,group,output,s,column,first} = read
  first
def read_row_step(+read: Read) -> U32:
  (read_groups(read) * read_block(read) : U32)
def read_source(+read: Read) -> U32:
  (read_row(read) * read_groups(read) * read_block(read) + read_line(read) * read_row_step(read) + read_group(read) * read_block(read) : U32)
def read_offset(+read: Read) -> U32:
  (read_tap_row(read) * read_row_step(read) + read_output(read) * read_shifts(read) + read_tap_column(read) + read_first_shift(read) : U32)
def read_row_total(+read: Read) -> U32:
  (read_row(read) + read_line(read) + read_tap_row(read) : U32)
def read_shift_total(+read: Read) -> U32:
  (read_tap_column(read) + read_first_shift(read) : U32)
def read_address(+read: Read) -> U32:
  (read_source(read) + read_offset(read) : U32)
def read_canonical(+read: Read) -> U32:
  ((read_row_total(read) * read_groups(read) + read_group(read)) * read_block(read) + read_output(read) * read_shifts(read) + read_shift_total(read) : U32)
def read_tail(+read: Read) -> U32:
  (read_output(read) * read_shifts(read) + read_shift_total(read) : U32)
def read_prefix(+read: Read) -> U32:
  (read_row(read) * read_groups(read) * read_block(read) + read_line(read) * read_row_step(read) : U32)
def offset_tail_matches(+read: Read) ->
  {read_offset(read) == (read_tap_row(read) * read_row_step(read) + read_tail(read) : U32) : U32}:
  +tap = (read_tap_row(read) * read_row_step(read) : U32)
  +output = (read_output(read) * read_shifts(read) : U32)
  +column = read_tap_column(read)
  +first = read_first_shift(read)
  Equal.trans(U32,read_offset(read),((tap + output) + (column + first) : U32),(tap + read_tail(read) : U32),Arithmetic.add_associative((tap + output : U32),column,first),Arithmetic.add_associative(tap,output,(column + first : U32)))
def read_grouped(+read: Read) ->
  {read_address(read) == (((read_prefix(read) + read_tap_row(read) * read_row_step(read)) + read_group(read) * read_block(read)) + read_tail(read) : U32) : U32}:
  +prefix = read_prefix(read)
  +group = (read_group(read) * read_block(read) : U32)
  +tap = (read_tap_row(read) * read_row_step(read) : U32)
  +tail = read_tail(read)
  %Equal.sym(U32,read_offset(read),(tap + tail : U32),offset_tail_matches(read)) : {(read_source(read) + _ : U32) == (((prefix + tap) + group) + tail : U32) : U32}
  Equal.trans(U32,((prefix + group) + (tap + tail) : U32),(prefix + (group + (tap + tail)) : U32),(((prefix + tap) + group) + tail : U32),Arithmetic.add_associative(prefix,group,(tap + tail : U32)),
    Equal.trans(U32,(prefix + (group + (tap + tail)) : U32),(prefix + ((group + tap) + tail) : U32),(((prefix + tap) + group) + tail : U32),Equal.cong(U32,U32,value => (prefix + value : U32),(group + (tap + tail) : U32),((group + tap) + tail : U32),Equal.sym(U32,((group + tap) + tail : U32),(group + (tap + tail) : U32),Arithmetic.add_associative(group,tap,tail))),
      Equal.trans(U32,(prefix + ((group + tap) + tail) : U32),(prefix + ((tap + group) + tail) : U32),(((prefix + tap) + group) + tail : U32),Equal.cong(U32,U32,value => (prefix + (value + tail) : U32),(group + tap : U32),(tap + group : U32),Arithmetic.add_commutative(group,tap)),
        Equal.trans(U32,(prefix + ((tap + group) + tail) : U32),((prefix + (tap + group)) + tail : U32),(((prefix + tap) + group) + tail : U32),Equal.sym(U32,((prefix + (tap + group)) + tail : U32),(prefix + ((tap + group) + tail) : U32),Arithmetic.add_associative(prefix,(tap + group : U32),tail)),Equal.cong(U32,U32,value => (value + tail : U32),(prefix + (tap + group) : U32),((prefix + tap) + group : U32),Equal.sym(U32,((prefix + tap) + group : U32),(prefix + (tap + group) : U32),Arithmetic.add_associative(prefix,tap,group)))))))
def row_terms_matches(+read: Read) ->
  {(read_prefix(read) + read_tap_row(read) * read_row_step(read) : U32) == (read_row_total(read) * read_row_step(read) : U32) : U32}:
  +base = read_row(read)
  +line = read_line(read)
  +tap = read_tap_row(read)
  +groups = read_groups(read)
  +block = read_block(read)
  +step = read_row_step(read)
  +prefix = Equal.trans(U32,read_prefix(read),(base * step + line * step : U32),((base + line) * step : U32),Equal.cong(U32,U32,value => (value + line * step : U32),(base * groups * block : U32),(base * step : U32),Arithmetic.multiply_associative(base,groups,block)),Equal.sym(U32,((base + line) * step : U32),(base * step + line * step : U32),Arithmetic.multiply_additive(base,line,step)))
  Equal.trans(U32,(read_prefix(read) + tap * step : U32),((base + line) * step + tap * step : U32),(read_row_total(read) * step : U32),Equal.cong(U32,U32,value => (value + tap * step : U32),read_prefix(read),((base + line) * step : U32),prefix),Equal.sym(U32,(read_row_total(read) * step : U32),((base + line) * step + tap * step : U32),Arithmetic.multiply_additive((base + line : U32),tap,step)))
def group_row_matches(+read: Read) ->
  {(read_row_total(read) * read_row_step(read) + read_group(read) * read_block(read) : U32) == ((read_row_total(read) * read_groups(read) + read_group(read)) * read_block(read) : U32) : U32}:
  +row = read_row_total(read)
  +groups = read_groups(read)
  +group = read_group(read)
  +block = read_block(read)
  Equal.trans(U32,(row * read_row_step(read) + group * block : U32),(row * groups * block + group * block : U32),((row * groups + group) * block : U32),Equal.cong(U32,U32,value => (value + group * block : U32),(row * read_row_step(read) : U32),(row * groups * block : U32),Equal.sym(U32,(row * groups * block : U32),(row * read_row_step(read) : U32),Arithmetic.multiply_associative(row,groups,block))),Equal.sym(U32,((row * groups + group) * block : U32),(row * groups * block + group * block : U32),Arithmetic.multiply_additive((row * groups : U32),group,block)))
def read_matches_canonical(+read: Read) -> {read_address(read) == read_canonical(read) : U32}:
  +row_terms = (read_prefix(read) + read_tap_row(read) * read_row_step(read) : U32)
  +group = (read_group(read) * read_block(read) : U32)
  +tail = read_tail(read)
  +grouped = ((read_row_total(read) * read_groups(read) + read_group(read)) * read_block(read) : U32)
  Equal.trans(U32,read_address(read),((row_terms + group) + tail : U32),read_canonical(read),read_grouped(read),
    Equal.trans(U32,((row_terms + group) + tail : U32),((read_row_total(read) * read_row_step(read) + group) + tail : U32),read_canonical(read),Equal.cong(U32,U32,value => ((value + group) + tail : U32),row_terms,(read_row_total(read) * read_row_step(read) : U32),row_terms_matches(read)),
      Equal.trans(U32,((read_row_total(read) * read_row_step(read) + group) + tail : U32),(grouped + tail : U32),read_canonical(read),Equal.cong(U32,U32,value => (value + tail : U32),(read_row_total(read) * read_row_step(read) + group : U32),grouped,group_row_matches(read)),Equal.sym(U32,read_canonical(read),(grouped + tail : U32),Arithmetic.add_associative(grouped,(read_output(read) * read_shifts(read) : U32),read_shift_total(read))))))

def context_source(context: Kernel.Context) -> U32:
  Kernel.Context{source,weight} = context
  source
def source_row(phase: Kernel.Phase,chunk: Kernel.Chunk) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,+k,+s,p,gh,gw},Kernel.Axis{+yf,yc,yt,yr,+ys},x,inner} = phase
  Kernel.Chunk{leaf,+row,rows,+low,group} = chunk
  (Band.clipped(yf,row,s) + ys + k - low : U32)
def read_at(+plan: Kernel.Plan,+table: Kernel.Table,+base: U32,+groups: U32,+block: U32,+index: U32,+relative: U32) -> Read:
  Kernel.Plan{+units,+span,inner,source,row_step,group_step,target,target_row,step,plane,channels,columns,wb,ob} = plan
  Kernel.Table{+tap_rows,+tap_columns,table_step,+shifts,+first_shift,table_base} = table
  Read{base,(index / units / span : U32),(relative / tap_columns % tap_rows : U32),groups,block,(index / units % span : U32),(relative / (tap_rows * tap_columns) : U32),shifts,(relative % tap_columns : U32),first_shift}
def chunk_layout(chunk: Kernel.Chunk) -> Kernel.Tiling:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{g,layout},first,count,target,plane},row,rows,low,group} = chunk
  layout
def layout_groups(layout: Kernel.Tiling) -> U32:
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,cr} = layout
  cg
def layout_block(layout: Kernel.Tiling) -> U32:
  Kernel.Tiling{groups,+shifts,+outputs,column_low,kernel,height,width,cg,cr} = layout
  (outputs * shifts + 1 : U32)
def phase_read(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+ob: U32,+index: U32,+relative: U32) -> Read:
  +layout = chunk_layout(chunk)
  read_at(Execution.work_plan(Execution.work(phase,chunk,wb,ob)),Packing.phase_table(phase,layout,ob),source_row(phase,chunk),layout_groups(layout),layout_block(layout),index,relative)
def source_table_matches(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+ob: U32,+index: U32,+relative: U32,+readers: Jobs.Readers) ->
  {(context_source(Reduction.native_context(Jobs.initial_loop(index,Execution.work_plan(Execution.work(phase,chunk,wb,ob)),readers))) + Offsets.value(Packing.phase_table(phase,chunk_layout(chunk),ob),relative) : U32) == read_address(phase_read(phase,chunk,wb,ob,index,relative)) : U32}:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,xs},inner} = phase
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},row,rows,low,group} = chunk
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,cr} = layout
  Jobs.Readers{tiles,weights,offsets} = readers
  {==}

def query_for_read(fill: Fill.Query,read: Read) -> Query:
  +read = read
  Query{fill,read_row_total(read),read_group(read),read_output(read),read_shift_total(read)}
type Layout<-fill: Fill.Query,-read: Read> is Data:
  Layout{groups: {read_groups(read) == Fill.groups(Fill.filling(fill)) : U32},block: {read_block(read) == Fill.block(Fill.filling(fill)) : U32},shifts: {read_shifts(read) == Fill.shifts(Fill.filling(fill)) : U32}}
def canonical_matches_query(+fill: Fill.Query,+read: Read,+layout: Layout<fill,read>) ->
  {read_canonical(read) == address(query_for_read(fill,read)) : U32}:
  Layout{+groups_equal,+block_equal,+shifts_equal} = layout
  +row = read_row_total(read)
  +group = read_group(read)
  +output = read_output(read)
  +shift = read_shift_total(read)
  +groups = Fill.groups(Fill.filling(fill))
  +block = Fill.block(Fill.filling(fill))
  +shifts = Fill.shifts(Fill.filling(fill))
  %Equal.sym(U32,read_groups(read),groups,groups_equal) : {((row * _ + group) * read_block(read) + output * read_shifts(read) + shift : U32) == address(query_for_read(fill,read)) : U32}
  %Equal.sym(U32,read_block(read),block,block_equal) : {((row * groups + group) * _ + output * read_shifts(read) + shift : U32) == address(query_for_read(fill,read)) : U32}
  %Equal.sym(U32,read_shifts(read),shifts,shifts_equal) : {((row * groups + group) * block + output * _ + shift : U32) == address(query_for_read(fill,read)) : U32}
  Arithmetic.add_associative(((row * groups + group) * block : U32),(output * shifts : U32),shift)

def slot_layout(+grid: Chunks.Grid,+first: U32,+target: U32,+plane: U32,+row: Nat,+group: Nat,+source: Execution.Source,+position: Nat,+output: Nat,+shift: Nat,
  +phase: Kernel.Phase,+wb: U32,+ob: U32,+index: U32,+relative: U32) ->
  Layout<Allocation.slot_query(grid,first,row,group,source,position,output,shift),phase_read(phase,Chunks.selected_chunk(grid,first,target,plane,row,group),wb,ob,index,relative)>:
  Chunks.Grid{call,e,c,range,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{cc,co,ch,cw,ck,cs,cp,cgh,cgw} = g
  Execution.Source{data,base,source_plane,source_first,source_rows} = source
  Kernel.Phase{Kernel.Geometry{ci,o,h,w,k,s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,xs},inner} = phase
  Layout{{==},{==},{==}}
