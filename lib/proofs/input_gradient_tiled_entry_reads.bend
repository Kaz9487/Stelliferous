# The entries a phase job reads: the word of entry (output, tap row, tap
# column) of a job is the filled slot of its tile.
import Base
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../kernel_input_gradient.bend as Kernel
import ../kernel_input_gradient_fill.bend as FillKernel
import ./input_gradient_tiled_fill.bend as Fill
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_chunk_allocation.bend as Allocation
import ./input_gradient_tiled_chunk_geometry.bend as Chunks
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_band_geometry.bend as Band
import ./input_gradient_tiled_bounds.bend as Groups
import ./kernel_input_gradient_fill_refinement.bend as Filling
import ./kernel_gemm_spec.bend as GemmSpec
import ./input_gradient_phase_packing.bend as Packing
import ./storage_tree_model.bend as Tree
import ./row_major_index.bend as RowMajor
import ./binary_radix.bend as Radix
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction

# The tile at row, group, output and shift of a filled chunk.
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
  {Tree.read(Tile.Tile8,Filling.tiles(Traversal.indexed_fold(~Filling.Owners,~FillKernel.Filling,~Filling.fill_group,Fill.count(fill(query)),0,Fill.filling(fill(query)),state)),address(query)) == Fill.wanted(fill(query)) : Tile.Tile8}:
  +filled = Traversal.indexed_fold(~Filling.Owners,~FillKernel.Filling,~Filling.fill_group,Fill.count(fill(query)),0,Fill.filling(fill(query)),state)
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Filling.tiles(filled),address(query)),Fill.read(fill(query),filled),Fill.wanted(fill(query)),Equal.cong(U32,Tile.Tile8,Tree.read(Tile.Tile8,Filling.tiles(filled)),address(query),Fill.address(fill(query)),Equal.sym(U32,Fill.address(fill(query)),address(query),addresses_match(query,fields,state,evidence))),Fill.readback(fill(query),state,evidence))

# A job read: the tile row of the chunk's first phase line, the job's line
# and group, and the entry's output, tap row and tap column.
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
# The first entry of the job, and the entry from it, as the job walks them.
def read_base(+read: Read) -> U32:
  (read_row(read) * read_row_step(read) + read_first_shift(read) + read_line(read) * read_row_step(read) + read_group(read) * read_block(read) : U32)
def read_offset(+read: Read) -> U32:
  (read_output(read) * read_shifts(read) + (read_tap_row(read) * read_row_step(read) + read_tap_column(read)) : U32)
def read_row_total(+read: Read) -> U32:
  (read_row(read) + read_line(read) + read_tap_row(read) : U32)
def read_shift_total(+read: Read) -> U32:
  (read_tap_column(read) + read_first_shift(read) : U32)
def read_address(+read: Read) -> U32:
  (read_base(read) + read_offset(read) : U32)
def read_canonical(+read: Read) -> U32:
  ((read_row_total(read) * read_groups(read) + read_group(read)) * read_block(read) + read_output(read) * read_shifts(read) + read_shift_total(read) : U32)

# Both words are the sum of the same terms in one order.
def canonical_terms(+read: Read) ->
  {read_canonical(read) == ((((((read_row(read) * read_row_step(read) + read_line(read) * read_row_step(read) : U32) + read_tap_row(read) * read_row_step(read) : U32) +
    read_group(read) * read_block(read) : U32) + read_output(read) * read_shifts(read) : U32) + read_tap_column(read) : U32) + read_first_shift(read) : U32) : U32}:
  +r = read_row(read)
  +l = read_line(read)
  +a = read_tap_row(read)
  +cg = read_groups(read)
  +block = read_block(read)
  +step = read_row_step(read)
  +g = read_group(read)
  +o = (read_output(read) * read_shifts(read) : U32)
  +c = read_tap_column(read)
  +f = read_first_shift(read)
  +total = read_row_total(read)
  +rows = ((r * step + l * step : U32) + a * step : U32)
  +grouped = Equal.trans(U32,((total * cg + g : U32) * block : U32),(total * cg * block + g * block : U32),(rows + g * block : U32),
    Arithmetic.multiply_additive((total * cg : U32),g,block),
    Equal.cong(U32,U32,value => (value + g * block : U32),(total * cg * block : U32),rows,
      Equal.trans(U32,(total * cg * block : U32),(total * step : U32),rows,Arithmetic.multiply_associative(total,cg,block),
        Equal.trans(U32,(total * step : U32),((r + l : U32) * step + a * step : U32),rows,Arithmetic.multiply_additive((r + l : U32),a,step),
          Equal.cong(U32,U32,value => (value + a * step : U32),((r + l : U32) * step : U32),(r * step + l * step : U32),Arithmetic.multiply_additive(r,l,step))))))
  Equal.trans(U32,read_canonical(read),(((rows + g * block : U32) + o : U32) + (c + f : U32) : U32),((((rows + g * block : U32) + o : U32) + c : U32) + f : U32),
    Equal.cong(U32,U32,value => ((value + o : U32) + (c + f : U32) : U32),((total * cg + g : U32) * block : U32),(rows + g * block : U32),grouped),
    Equal.sym(U32,((((rows + g * block : U32) + o : U32) + c : U32) + f : U32),(((rows + g * block : U32) + o : U32) + (c + f : U32) : U32),
      Arithmetic.add_associative(((rows + g * block : U32) + o : U32),c,f)))

def address_terms(+read: Read) ->
  {read_address(read) == ((((((read_row(read) * read_row_step(read) + read_line(read) * read_row_step(read) : U32) + read_tap_row(read) * read_row_step(read) : U32) +
    read_group(read) * read_block(read) : U32) + read_output(read) * read_shifts(read) : U32) + read_tap_column(read) : U32) + read_first_shift(read) : U32) : U32}:
  +rs = (read_row(read) * read_row_step(read) : U32)
  +ls = (read_line(read) * read_row_step(read) : U32)
  +as = (read_tap_row(read) * read_row_step(read) : U32)
  +gb = (read_group(read) * read_block(read) : U32)
  +o = (read_output(read) * read_shifts(read) : U32)
  +c = read_tap_column(read)
  +f = read_first_shift(read)
  +p = (rs + ls : U32)
  +z = (p + gb : U32)
  +w = (o + (as + c : U32) : U32)
  +front = Equal.trans(U32,(((rs + f : U32) + ls : U32) + gb : U32),((p + f : U32) + gb : U32),(z + f : U32),
    Equal.cong(U32,U32,value => (value + gb : U32),((rs + f : U32) + ls : U32),(p + f : U32),Arithmetic.add_swap(rs,f,ls)),Arithmetic.add_swap(p,f,gb))
  +back = Equal.trans(U32,(z + w : U32),(z + ((o + as : U32) + c : U32) : U32),((((p + as : U32) + gb : U32) + o : U32) + c : U32),
    Equal.cong(U32,U32,value => (z + value : U32),w,((o + as : U32) + c : U32),Equal.sym(U32,((o + as : U32) + c : U32),w,Arithmetic.add_associative(o,as,c))),
    Equal.trans(U32,(z + ((o + as : U32) + c : U32) : U32),((z + (o + as : U32) : U32) + c : U32),((((p + as : U32) + gb : U32) + o : U32) + c : U32),
      Equal.sym(U32,((z + (o + as : U32) : U32) + c : U32),(z + ((o + as : U32) + c : U32) : U32),Arithmetic.add_associative(z,(o + as : U32),c)),
      Equal.cong(U32,U32,value => (value + c : U32),(z + (o + as : U32) : U32),(((p + as : U32) + gb : U32) + o : U32),
        Equal.trans(U32,(z + (o + as : U32) : U32),((z + o : U32) + as : U32),(((p + as : U32) + gb : U32) + o : U32),
          Equal.sym(U32,((z + o : U32) + as : U32),(z + (o + as : U32) : U32),Arithmetic.add_associative(z,o,as)),
          Equal.trans(U32,((z + o : U32) + as : U32),((z + as : U32) + o : U32),(((p + as : U32) + gb : U32) + o : U32),Arithmetic.add_swap(z,o,as),
            Equal.cong(U32,U32,value => (value + o : U32),(z + as : U32),((p + as : U32) + gb : U32),Arithmetic.add_swap(p,gb,as)))))))
  Equal.trans(U32,read_address(read),((z + f : U32) + w : U32),(((((p + as : U32) + gb : U32) + o : U32) + c : U32) + f : U32),
    Equal.cong(U32,U32,value => (value + w : U32),(((rs + f : U32) + ls : U32) + gb : U32),(z + f : U32),front),
    Equal.trans(U32,((z + f : U32) + w : U32),((z + w : U32) + f : U32),(((((p + as : U32) + gb : U32) + o : U32) + c : U32) + f : U32),Arithmetic.add_swap(z,f,w),
      Equal.cong(U32,U32,value => (value + f : U32),(z + w : U32),((((p + as : U32) + gb : U32) + o : U32) + c : U32),back)))

def read_matches_canonical(+read: Read) -> {read_address(read) == read_canonical(read) : U32}:
  Equal.trans(U32,read_address(read),
    ((((((read_row(read) * read_row_step(read) + read_line(read) * read_row_step(read) : U32) + read_tap_row(read) * read_row_step(read) : U32) +
      read_group(read) * read_block(read) : U32) + read_output(read) * read_shifts(read) : U32) + read_tap_column(read) : U32) + read_first_shift(read) : U32),
    read_canonical(read),address_terms(read),
    Equal.sym(U32,read_canonical(read),
      ((((((read_row(read) * read_row_step(read) + read_line(read) * read_row_step(read) : U32) + read_tap_row(read) * read_row_step(read) : U32) +
        read_group(read) * read_block(read) : U32) + read_output(read) * read_shifts(read) : U32) + read_tap_column(read) : U32) + read_first_shift(read) : U32),
      canonical_terms(read)))

# The chunk's layout, its first phase tile row and its first shift.
def chunk_layout(chunk: Kernel.Chunk) -> Kernel.Tiling:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{g,layout},first,count,target},row,rows,low,group} = chunk
  layout
def layout_groups(layout: Kernel.Tiling) -> U32:
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,cr} = layout
  cg
def layout_block(layout: Kernel.Tiling) -> U32:
  Kernel.Tiling{groups,+shifts,+outputs,column_low,kernel,height,width,cg,cr} = layout
  (outputs * shifts + 1 : U32)
def layout_shifts(layout: Kernel.Tiling) -> U32:
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,cr} = layout
  shifts
def source_row(phase: Kernel.Phase,chunk: Kernel.Chunk) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,+k,+s,p,gh,gw},Kernel.Axis{+yf,yc,yt,yr,+ys},x,inner} = phase
  Kernel.Chunk{leaf,+row,rows,+low,group} = chunk
  (Band.clipped(yf,row,s) + ys + k - low : U32)
def first_shift(phase: Kernel.Phase,chunk: Kernel.Chunk) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw},y,Kernel.Axis{xf,xc,xt,xr,+xs},inner} = phase
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{g,Kernel.Tiling{groups,shifts,outputs,+column_low,kernel,height,width,cg,cr}},first,count,target},row,rows,low,group} = chunk
  (xs + k - column_low : U32)

# The read of entry (output, row, column) of job index of a phase chunk.
def job_read(+plan: Kernel.Plan,+source: U32,+groups: U32,+block: U32,+shifts: U32,+first: U32,+index: U32,+output: Nat,+row: Nat,+column: Nat) -> Read:
  Kernel.Plan{+units,+span,inner,source_base,row_step,group_step,target,target_row,step,cells,channels,columns,base,tap_columns,tap_rows,row_jump,output_jump} = plan
  Read{source,(index / units / span : U32),U32.from_nat(row),groups,block,(index / units % span : U32),U32.from_nat(output),shifts,U32.from_nat(column),first}

def phase_read(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+index: U32,+output: Nat,+row: Nat,+column: Nat) -> Read:
  +layout = chunk_layout(chunk)
  job_read(Execution.work_plan(Execution.work(phase,chunk,wb)),source_row(phase,chunk),layout_groups(layout),layout_block(layout),layout_shifts(layout),first_shift(phase,chunk),
    index,output,row,column)


# The entries of a job as a grid: tap rows a row step apart, one plane for
# each output.
def job_grid(+plan: Kernel.Plan,+index: U32) -> GemmSpec.Grid:
  Kernel.Plan{+units,+groups,inner,+source_base,+row_step,+group_step,target_base,target_row,step,cells,channels,columns,weight_base,+tap_columns,+tap_rows,row_jump,+output_jump} = plan
  +index = index
  GemmSpec.Grid{(source_base + index / units / groups * row_step + index / units % groups * group_step : U32),U32.to_nat(row_step),tap_columns,tap_rows,output_jump,0n}

# The planes of a phase grid are the outputs of the tiling apart.
def plane_shifts(+shifts: U32,+band: U32,+columns: U32,+rows: U32) ->
  {GemmSpec.plane_word(band,columns,rows,(shifts - (rows - 1) * band - (columns - 1) : U32)) == shifts : U32}:
  +low = ((rows - 1) * band + (columns - 1) : U32)
  Equal.trans(U32,(low + (shifts - (rows - 1) * band - (columns - 1) : U32) : U32),(low + (shifts - low : U32) : U32),shifts,
    Equal.cong(U32,U32,value => (low + value : U32),(shifts - (rows - 1) * band - (columns - 1) : U32),(shifts - low : U32),
      Arithmetic.subtract_sum(shifts,((rows - 1) * band : U32),(columns - 1 : U32))),
    Equal.trans(U32,(low + (shifts - low : U32) : U32),((shifts - low : U32) + low : U32),shifts,Arithmetic.add_commutative(low,(shifts - low : U32)),
      Subtraction.add_subtract(shifts,low)))

# The runtime base of a job is the read base of its entries.
def base_matches(+row: U32,+step: U32,+shift: U32,+kernel: U32,+low: U32) ->
  {(row * step + shift + kernel - low : U32) == (row * step + (shift + kernel - low : U32) : U32) : U32}:
  Equal.trans(U32,((((row * step : U32) + shift : U32) + kernel : U32) - low : U32),(((row * step : U32) + (shift + kernel : U32) : U32) - low : U32),
    ((row * step : U32) + ((shift + kernel : U32) - low : U32) : U32),
    Equal.cong(U32,U32,value => (value - low : U32),(((row * step : U32) + shift : U32) + kernel : U32),((row * step : U32) + (shift + kernel : U32) : U32),
      Arithmetic.add_associative((row * step : U32),shift,kernel)),
    Arithmetic.subtract_from_sum((row * step : U32),(shift + kernel : U32),low))

# The word of entry (output, row, column) of a job is the address of its read.
def entry_word(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+index: U32,+output: Nat,+row: Nat,+column: Nat,
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(Packing.phase_columns(phase))) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(Packing.phase_rows(phase))) == True{} : Bool}) ->
  {GemmSpec.grid_at_word(job_grid(Execution.work_plan(Execution.work(phase,chunk,wb)),index),output,row,column) == read_address(phase_read(phase,chunk,wb,index,output,row,column)) : U32}:
  Kernel.Phase{Kernel.Geometry{+c,o,+h,+w,+k,+s,p,gh,gw},Kernel.Axis{+yf,+yc,yt,+yr,+ys},Kernel.Axis{+xf,+xc,xt,+xr,+xs},+inner} = phase
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},+first,count,+target},+chunk_row,+rows,+low,+chunk_group} = chunk
  Kernel.Tiling{groups,+shifts,+outputs,+column_low,kernel,height,width,+cg,cr} = layout
  +units = ((c + 7) / 8 : U32)
  +phase_groups = ((xc + 7) / 8 : U32)
  +span = Bool.pick(U32,U32.is_lt(chunk_group,phase_groups),U32.min(cg,(phase_groups - chunk_group : U32)),0)
  +block = (outputs * shifts + 1 : U32)
  +step = (cg * block : U32)
  +start = Band.clipped(yf,chunk_row,s)
  +r = (start + ys + k - low : U32)
  +f = (xs + k - column_low : U32)
  +lr = ((index / units / span : U32) * step : U32)
  +gb = ((index / units % span : U32) * block : U32)
  +jump = (shifts - (yr - 1) * step - (xr - 1) : U32)
  +offset = (U32.from_nat(row) * step + U32.from_nat(column) : U32)
  +source = (r * step + xs + k - column_low : U32)
  Equal.trans(U32,GemmSpec.grid_at_word(GemmSpec.Grid{(source + lr + gb : U32),U32.to_nat(step),xr,yr,jump,0n},output,row,column),
    ((source + lr + gb : U32) + (U32.from_nat(output) * GemmSpec.plane_word(step,xr,yr,jump) + offset : U32) : U32),
    ((((r * step : U32) + f : U32) + lr + gb : U32) + (U32.from_nat(output) * shifts + offset : U32) : U32),
    GemmSpec.grid_offset((source + lr + gb : U32),step,xr,yr,jump,0n,output,row,column,columns_positive,rows_positive),
    Equal.trans(U32,((source + lr + gb : U32) + (U32.from_nat(output) * GemmSpec.plane_word(step,xr,yr,jump) + offset : U32) : U32),
      ((source + lr + gb : U32) + (U32.from_nat(output) * shifts + offset : U32) : U32),
      ((((r * step : U32) + f : U32) + lr + gb : U32) + (U32.from_nat(output) * shifts + offset : U32) : U32),
      Equal.cong(U32,U32,value => ((source + lr + gb : U32) + (U32.from_nat(output) * value + offset : U32) : U32),GemmSpec.plane_word(step,xr,yr,jump),shifts,plane_shifts(shifts,step,xr,yr)),
      Equal.cong(U32,U32,value => ((value + lr + gb : U32) + (U32.from_nat(output) * shifts + offset : U32) : U32),source,((r * step : U32) + f : U32),base_matches(r,step,xs,k,column_low))))

# A read in terms of the fill slot it names.
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

def slot_layout(+grid: Chunks.Grid,+first: U32,+target: U32,+row: Nat,+group: Nat,+source: Execution.Source,+position: Nat,+fill_output: Nat,+shift: Nat,
  +phase: Kernel.Phase,+wb: U32,+index: U32,+output: Nat,+tap_row: Nat,+tap_column: Nat) ->
  Layout<Allocation.slot_query(grid,first,row,group,source,position,fill_output,shift),phase_read(phase,Chunks.selected_chunk(grid,first,target,row,group),wb,index,output,tap_row,tap_column)>:
  Chunks.Grid{call,e,c,range,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{cc,co,ch,cw,ck,cs,cp,cgh,cgw} = g
  Execution.Source{data,base,source_first,source_rows} = source
  Kernel.Phase{Kernel.Geometry{ci,o,h,w,k,s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,xs},inner} = phase
  Layout{{==},{==},{==}}
