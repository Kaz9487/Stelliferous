# The gradient rows a leaf reads from a band lent by the holding of the
# gradient cells: the cells of a reader, written into one buffer for the
# specification, are what the band holds.
import Base
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../tensor_move.bend as Move
import ../tensor_tiled_cells.bend as Cells
import ../storage_buffer.bend as Storage
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./tensor_move_readback.bend as Held
import ./tensor_list_values.bend as ListValues
import ./list_storage_readback.bend as ListRead
import ./tensor_allocation_depth.bend as Depth
import ./input_gradient_tiled_source_rows.bend as Rows
import ./input_gradient_tiled_source_contents.bend as Contents
import ./input_gradient_tiled_source_coverage.bend as Coverage
import ./input_gradient_tiled_source_witness.bend as Witnesses
import ./input_gradient_tiled_band_order.bend as BandOrder
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_copy_geometry.bend as Copy
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_packing.bend as Packing
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./word_subtraction.bend as Subtraction
import ./word_encoding.bend as WordEncoding
import ./encoded_coordinates.bend as Encoded
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_division.bend as NativeDivision
import ./u32_full_range.bend as FullRange
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor

# ---- The cells of a reader as one buffer ----

# The cell of a reader at a cell number.
def holder_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,holder: Holder,index: Nat) -> Tile.Tile8:
  value(holder,U32.from_nat(index))

# The cells [0, count) of a reader written into one buffer.
def gradient_buffer(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+count: U32) -> Storage.Buffer<Tile.Tile8>:
  Storage.from_list(~Tile.Tile8,ListRead.enumerate(~Tile.Tile8,~Holder,~holder_cell(~Holder,~value),U32.to_nat(count),0n,holder),
    Tensors.depth_for(count),Tile.filled(0.0),U32.to_nat(count))
def gradient_model(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+count: U32) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,gradient_buffer(~Holder,~value,holder,count))

def model_value(+model: Buffer.Model<Tile.Tile8>,+index: U32) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,model),index) == Buffer.value(~Tile.Tile8,index,Buffer.content(~Tile.Tile8,model)) : Tile.Tile8}:
  Buffer.Model{tree,length,certificate} = model
  {==}

# A cell of the buffer is the cell of the reader.
def gradient_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+count: U32,+sized: Depth.Sized<count>,+index: Nat,
  +inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,gradient_model(~Holder,~value,holder,count)),U32.from_nat(index)) == value(holder,U32.from_nat(index)) : Tile.Tile8}:
  +values = ListRead.enumerate(~Tile.Tile8,~Holder,~holder_cell(~Holder,~value),U32.to_nat(count),0n,holder)
  +model = gradient_model(~Holder,~value,holder,count)
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,model),U32.from_nat(index)),Buffer.value(~Tile.Tile8,U32.from_nat(index),Buffer.content(~Tile.Tile8,model)),
    value(holder,U32.from_nat(index)),
    model_value(model,U32.from_nat(index)),
    {%Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,model),Buffer.contents(~Tile.Tile8,gradient_buffer(~Holder,~value,holder,count)),Buffer.remodel_content(~Tile.Tile8,gradient_buffer(~Holder,~value,holder,count))) :
      {Buffer.value(~Tile.Tile8,U32.from_nat(index),_) == value(holder,U32.from_nat(index)) : Tile.Tile8}
    Equal.trans(Tile.Tile8,Buffer.value(~Tile.Tile8,U32.from_nat(index),Buffer.contents(~Tile.Tile8,gradient_buffer(~Holder,~value,holder,count))),ListRead.at(~Tile.Tile8,Tile.filled(0.0),index,values),
      value(holder,U32.from_nat(index)),
      ListValues.written(~Tile.Tile8,values,count,Tile.filled(0.0),index,ListRead.enumerate_length(~Tile.Tile8,~Holder,~holder_cell(~Holder,~value),U32.to_nat(count),0n,holder),
        inside,sized),
      ListRead.enumerate_at(~Tile.Tile8,~Holder,~holder_cell(~Holder,~value),Tile.filled(0.0),U32.to_nat(count),index,0n,holder,inside)) :
      {Buffer.value(~Tile.Tile8,U32.from_nat(index),Buffer.content(~Tile.Tile8,model)) == value(holder,U32.from_nat(index)) : Tile.Tile8}})

# ---- Cell words ----

# Cell group of the pixel at row and column of a plane width wide, groups
# cells a pixel, from base on.
def cell_word(+base: U32,+base_n: Nat,+base_exact: {base == U32.from_nat(base_n) : U32},
  +group: U32,+group_n: Nat,+group_exact: {group == U32.from_nat(group_n) : U32},
  +row: U32,+row_n: Nat,+row_exact: {row == U32.from_nat(row_n) : U32},
  +width: U32,+width_n: Nat,+width_exact: {width == U32.from_nat(width_n) : U32},
  +column: U32,+column_n: Nat,+column_exact: {column == U32.from_nat(column_n) : U32},
  +groups: U32,+groups_n: Nat,+groups_exact: {groups == U32.from_nat(groups_n) : U32}) ->
  {((base + group : U32) + (row * width + column : U32) * groups : U32) ==
    U32.from_nat(Nat.add(Nat.add(base_n,group_n),Nat.mul(Nat.add(Nat.mul(row_n,width_n),column_n),groups_n))) : U32}:
  +offset = Nat.add(Nat.mul(row_n,width_n),column_n)
  +result = U32.from_nat(Nat.add(Nat.add(base_n,group_n),Nat.mul(offset,groups_n)))
  %Equal.sym(U32,base,U32.from_nat(base_n),base_exact) : {((_ + group : U32) + (row * width + column : U32) * groups : U32) == result : U32}
  %Equal.sym(U32,group,U32.from_nat(group_n),group_exact) : {((U32.from_nat(base_n) + _ : U32) + (row * width + column : U32) * groups : U32) == result : U32}
  %Equal.sym(U32,row,U32.from_nat(row_n),row_exact) :
    {((U32.from_nat(base_n) + U32.from_nat(group_n) : U32) + (_ * width + column : U32) * groups : U32) == result : U32}
  %Equal.sym(U32,width,U32.from_nat(width_n),width_exact) :
    {((U32.from_nat(base_n) + U32.from_nat(group_n) : U32) + (U32.from_nat(row_n) * _ + column : U32) * groups : U32) == result : U32}
  %Equal.sym(U32,column,U32.from_nat(column_n),column_exact) :
    {((U32.from_nat(base_n) + U32.from_nat(group_n) : U32) + (U32.from_nat(row_n) * U32.from_nat(width_n) + _ : U32) * groups : U32) == result : U32}
  %Equal.sym(U32,groups,U32.from_nat(groups_n),groups_exact) :
    {((U32.from_nat(base_n) + U32.from_nat(group_n) : U32) + (U32.from_nat(row_n) * U32.from_nat(width_n) + U32.from_nat(column_n) : U32) * _ : U32) == result : U32}
  %Equal.sym(U32,(U32.from_nat(row_n) * U32.from_nat(width_n) + U32.from_nat(column_n) : U32),U32.from_nat(offset),Encoded.encoded_madd(row_n,width_n,column_n)) :
    {((U32.from_nat(base_n) + U32.from_nat(group_n) : U32) + _ * U32.from_nat(groups_n) : U32) == result : U32}
  %Equal.sym(U32,(U32.from_nat(offset) * U32.from_nat(groups_n) : U32),U32.from_nat(Nat.mul(offset,groups_n)),Multiplication.native_encoded_multiply(offset,groups_n)) :
    {((U32.from_nat(base_n) + U32.from_nat(group_n) : U32) + _ : U32) == result : U32}
  %Equal.sym(U32,(U32.from_nat(base_n) + U32.from_nat(group_n) : U32),U32.from_nat(Nat.add(base_n,group_n)),WordArithmetic.native_add(base_n,group_n)) :
    {(_ + U32.from_nat(Nat.mul(offset,groups_n)) : U32) == result : U32}
  WordArithmetic.native_add(Nat.add(base_n,group_n),Nat.mul(offset,groups_n))

# The group of eight channels of a channel below a word, as a word.
def group_word(+output: Nat,+limit: U32,+within: {Nat.is_le(output,U32.to_nat(limit)) == True{} : Bool}) ->
  {(U32.from_nat(output) / 8 : U32) == U32.from_nat(Nat.div(output,8n)) : U32}:
  Encoded.encoding((U32.from_nat(output) / 8 : U32),Nat.div(output,8n),
    Equal.trans(Nat,U32.to_nat((U32.from_nat(output) / 8 : U32)),Nat.div(U32.to_nat(U32.from_nat(output)),8n),Nat.div(output,8n),
      NativeDivision.divide(U32.from_nat(output),8,16,{==},{==},{==}),
      Equal.cong(Nat,Nat,value => Nat.div(value,8n),U32.to_nat(U32.from_nat(output)),output,FullRange.roundtrip(output,limit,within))))

# A row counted from a first row, as a word.
def local_word(+first: U32,+row: Nat) -> {(U32.from_nat(Nat.add(U32.to_nat(first),row)) - first : U32) == U32.from_nat(row) : U32}:
  %Arithmetic.native_roundtrip(first) : {(U32.from_nat(Nat.add(U32.to_nat(first),row)) - _ : U32) == U32.from_nat(row) : U32}
  %Order.commutative(row,U32.to_nat(first)) : {(U32.from_nat(_) - U32.from_nat(U32.to_nat(first)) : U32) == U32.from_nat(row) : U32}
  Subtraction.encoded_subtract_add(row,U32.to_nat(first))

# ---- Cell numbers ----

def shuffle(+u: Nat,+q: Nat,+v: Nat) -> {Nat.add(u,Nat.add(q,v)) == Nat.add(Nat.add(u,v),q) : Nat}:
  %Order.commutative(v,q) : {Nat.add(u,_) == Nat.add(Nat.add(u,v),q) : Nat}
  Equal.sym(Nat,Nat.add(Nat.add(u,v),q),Nat.add(u,Nat.add(v,q)),Addition.associative(u,v,q))

def swap_last(+u: Nat,+q: Nat,+v: Nat) -> {Nat.add(Nat.add(u,q),v) == Nat.add(Nat.add(u,v),q) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(u,q),v),Nat.add(u,Nat.add(q,v)),Nat.add(Nat.add(u,v),q),Addition.associative(u,q,v),shuffle(u,q,v))

# The cell of a band row from the first cell of the band, rows from a + c on,
# is the cell of the pixel of row a + c + l: (pixel * g) + q.
def band_number(+a: Nat,+c: Nat,+l: Nat,+w: Nat,+column: Nat,+g: Nat,+q: Nat) ->
  {Nat.add(Nat.mul(Nat.add(a,c),Nat.mul(w,g)),Nat.add(q,Nat.mul(Nat.add(Nat.mul(l,w),column),g))) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.add(a,c),l),w),column),g),q) : Nat}:
  +x = Nat.mul(Nat.add(a,c),w)
  +z = Nat.add(Nat.mul(l,w),column)
  +p = Nat.add(Nat.mul(Nat.add(Nat.add(a,c),l),w),column)
  +summed = Equal.trans(Nat,Nat.add(x,z),Nat.add(Nat.add(x,Nat.mul(l,w)),column),p,
      Equal.sym(Nat,Nat.add(Nat.add(x,Nat.mul(l,w)),column),Nat.add(x,z),Addition.associative(x,Nat.mul(l,w),column)),
      Equal.cong(Nat,Nat,head => Nat.add(head,column),Nat.add(x,Nat.mul(l,w)),Nat.mul(Nat.add(Nat.add(a,c),l),w),
        Equal.sym(Nat,Nat.mul(Nat.add(Nat.add(a,c),l),w),Nat.add(x,Nat.mul(l,w)),RowMajor.distrib(Nat.add(a,c),l,w))))
  %RowMajor.assoc(Nat.add(a,c),w,g) : {Nat.add(_,Nat.add(q,Nat.mul(z,g))) == Nat.add(Nat.mul(p,g),q) : Nat}
  Equal.trans(Nat,Nat.add(Nat.mul(x,g),Nat.add(q,Nat.mul(z,g))),Nat.add(Nat.add(Nat.mul(x,g),Nat.mul(z,g)),q),Nat.add(Nat.mul(p,g),q),
    shuffle(Nat.mul(x,g),q,Nat.mul(z,g)),
    %RowMajor.distrib(x,z,g) : {Nat.add(_,q) == Nat.add(Nat.mul(p,g),q) : Nat}
    Equal.cong(Nat,Nat,sum => Nat.add(Nat.mul(sum,g),q),Nat.add(x,z),p,summed))

# The cell of group q at row c + l and a column of image i: the same cell.
def image_number(+i: Nat,+h: Nat,+c: Nat,+l: Nat,+w: Nat,+column: Nat,+g: Nat,+q: Nat) ->
  {Nat.add(Nat.add(Nat.mul(Nat.mul(i,Nat.mul(h,w)),g),q),Nat.mul(Nat.add(Nat.mul(Nat.add(c,l),w),column),g)) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.add(Nat.mul(i,h),c),l),w),column),g),q) : Nat}:
  +a = Nat.mul(i,h)
  +b = Nat.mul(i,Nat.mul(h,w))
  +y = Nat.add(Nat.mul(Nat.add(c,l),w),column)
  +p = Nat.add(Nat.mul(Nat.add(Nat.add(a,c),l),w),column)
  +scaled = Equal.trans(Nat,Nat.add(Nat.mul(a,w),Nat.mul(Nat.add(c,l),w)),Nat.mul(Nat.add(a,Nat.add(c,l)),w),Nat.mul(Nat.add(Nat.add(a,c),l),w),
      Equal.sym(Nat,Nat.mul(Nat.add(a,Nat.add(c,l)),w),Nat.add(Nat.mul(a,w),Nat.mul(Nat.add(c,l),w)),RowMajor.distrib(a,Nat.add(c,l),w)),
      Equal.cong(Nat,Nat,sum => Nat.mul(sum,w),Nat.add(a,Nat.add(c,l)),Nat.add(Nat.add(a,c),l),
        Equal.sym(Nat,Nat.add(Nat.add(a,c),l),Nat.add(a,Nat.add(c,l)),Addition.associative(a,c,l))))
  +summed = {
    %RowMajor.assoc(i,h,w) : {Nat.add(_,y) == p : Nat}
    Equal.trans(Nat,Nat.add(Nat.mul(a,w),y),Nat.add(Nat.add(Nat.mul(a,w),Nat.mul(Nat.add(c,l),w)),column),p,
      Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(a,w),Nat.mul(Nat.add(c,l),w)),column),Nat.add(Nat.mul(a,w),y),Addition.associative(Nat.mul(a,w),Nat.mul(Nat.add(c,l),w),column)),
      Equal.cong(Nat,Nat,head => Nat.add(head,column),Nat.add(Nat.mul(a,w),Nat.mul(Nat.add(c,l),w)),Nat.mul(Nat.add(Nat.add(a,c),l),w),scaled)) :
    {Nat.add(b,y) == p : Nat}}
  Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(b,g),q),Nat.mul(y,g)),Nat.add(Nat.add(Nat.mul(b,g),Nat.mul(y,g)),q),Nat.add(Nat.mul(p,g),q),
    swap_last(Nat.mul(b,g),q,Nat.mul(y,g)),
    %RowMajor.distrib(b,y,g) : {Nat.add(_,q) == Nat.add(Nat.mul(p,g),q) : Nat}
    Equal.cong(Nat,Nat,sum => Nat.add(Nat.mul(sum,g),q),Nat.add(b,y),p,summed))

# A cell of a pixel of an image of the batch lies among the cells of the batch.
def number_inside(+image: Nat,+images: Nat,+height: Nat,+row: Nat,+width: Nat,+column: Nat,+groups: Nat,+group: Nat,
  +image_inside: {Nat.is_lt(image,images) == True{} : Bool},+row_inside: {Nat.is_lt(row,height) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,width) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(image,height),row),width),column),groups),group),Nat.mul(Nat.mul(Nat.mul(images,height),width),groups)) == True{} : Bool}:
  RowMajor.radix(Nat.add(Nat.mul(Nat.add(Nat.mul(image,height),row),width),column),Nat.mul(Nat.mul(images,height),width),group,groups,
    RowMajor.radix(Nat.add(Nat.mul(image,height),row),Nat.mul(images,height),column,width,RowMajor.radix(image,images,row,height,image_inside,row_inside),column_inside),
    group_inside)

# A cell of a band row lies among the cells of the band.
def band_inside(+row: Nat,+rows: Nat,+width: Nat,+column: Nat,+groups: Nat,+group: Nat,
  +row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+column_inside: {Nat.is_lt(column,width) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,groups) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(group,Nat.mul(Nat.add(Nat.mul(row,width),column),groups)),Nat.mul(rows,Nat.mul(width,groups))) == True{} : Bool}:
  +offset = Nat.add(Nat.mul(row,width),column)
  %Order.commutative(Nat.mul(offset,groups),group) : {Nat.is_lt(_,Nat.mul(rows,Nat.mul(width,groups))) == True{} : Bool}
  %RowMajor.assoc(rows,width,groups) : {Nat.is_lt(Nat.add(Nat.mul(offset,groups),group),_) == True{} : Bool}
  RowMajor.radix(offset,Nat.mul(rows,width),group,groups,RowMajor.radix(row,rows,column,width,row_inside,column_inside),group_inside)

def below_zero(+value: Nat,inside: {Nat.is_lt(value,0n) == True{} : Bool}) -> Empty:
  WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(value,0n),True{},Equal.sym(Bool,Nat.is_lt(value,0n),False{},Division.lt_zero(value)),inside))

# A row of a range of rows lies in the image.
def rows_inside(+first: U32,+rows: U32,+height: U32,range: Rows.Range<first,rows,height>,+row: Nat,
  +inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(first),row),U32.to_nat(height)) == True{} : Bool}:
  match range:
    case Rows.EmptyRows{+empty}:
      Empty.absurd({Nat.is_lt(Nat.add(U32.to_nat(first),row),U32.to_nat(height)) == True{} : Bool},
        below_zero(row,%empty : {Nat.is_lt(row,U32.to_nat(_)) == True{} : Bool} inside))
    case Rows.ImageRows{+contained}:
      Bounds.nat_trans(Nat.add(U32.to_nat(first),row),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(height),
        Order.offset_lt(U32.to_nat(first),row,U32.to_nat(rows),inside),contained)

def copy_row_inside(+band: Bands.Query,+row: Nat,+inside: {Nat.is_lt(row,U32.to_nat(Copy.row_count(band))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(Copy.begin(band)),row),U32.to_nat(Domain.gradient_rows(Bands.geometry(band)))) == True{} : Bool}:
  rows_inside(Copy.begin(band),Copy.row_count(band),Domain.gradient_rows(Bands.geometry(band)),BandOrder.copy_range(band),row,inside)

# ---- A band lent from the holding of the gradient ----

# Where count cells from cell first lie in a holding of cells [low, high):
# nowhere when there are none, otherwise inside.
type Placed<-low: Nat,-high: Nat,-first: U32,-count: U32> is Data:
  EmptyBand{empty: {count == 0 : U32}}
  HeldBand{after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool}}

# Count cells from cell first of a holding of the cells of a reader.
type Lent<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8> is Data:
  Lent{holder: Holder,holding: Held.Holding<Tile.Tile8>,low: Nat,high: Nat,capacity: U32,
    capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
    tiled: Held.Tiled(~Tile.Tile8,holding,low,high),values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),
    first: U32,count: U32,fill: Tile.Tile8,spare: Held.SpareModel<Tile.Tile8>,spare_room: Held.SpareRoom(~Tile.Tile8,spare),placed: Placed<low,high,first,count>,
    allocatable: {U32.is_le(count,1073741824) == True{} : Bool}}

def lent_holder(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lent: Lent<Holder,value>) -> Holder:
  Lent{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,first,count,fill,spare,spare_room,placed,allocatable} = lent
  holder
def lent_first(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lent: Lent<Holder,value>) -> U32:
  Lent{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,first,count,fill,spare,spare_room,placed,allocatable} = lent
  first
def lent_count(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lent: Lent<Holder,value>) -> U32:
  Lent{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,first,count,fill,spare,spare_room,placed,allocatable} = lent
  count
def lent_band(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lent: Lent<Holder,value>) -> Move.Band<Tile.Tile8>:
  Lent{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,first,count,fill,spare,spare_room,placed,allocatable} = lent
  Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),first,count,fill,Held.pack_spare(~Tile.Tile8,spare))

# Cell k of a band that holds cells is the cell first + k of the reader.
def placed_read(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,+low: Nat,+high: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),
  +first: U32,+count: U32,+fill: Tile.Tile8,+spare: Held.SpareModel<Tile.Tile8>,+spare_room: Held.SpareRoom(~Tile.Tile8,spare),placed: Placed<low,high,first,count>,+allocatable: {U32.is_le(count,1073741824) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  Held.BandRead(~Tile.Tile8,~Holder,~value,holder,Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),first,count,fill,Held.pack_spare(~Tile.Tile8,spare)),first,index):
  match placed:
    case EmptyBand{+empty}:
      Empty.absurd(Held.BandRead(~Tile.Tile8,~Holder,~value,holder,Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),first,count,fill,Held.pack_spare(~Tile.Tile8,spare)),first,index),
        below_zero(index,%empty : {Nat.is_lt(index,U32.to_nat(_)) == True{} : Bool} less))
    case HeldBand{+after,+fits}:
      +start = U32.to_nat(first)
      +inside = Bounds.nat_trans(start,Nat.add(start,U32.to_nat(count)),high,
        Order.le_lt_trans(start,Nat.add(start,index),Nat.add(start,U32.to_nat(count)),Order.le_add(start,index),Order.offset_lt(start,index,U32.to_nat(count),less)),fits)
      Held.band_value(~Tile.Tile8,~Holder,~value,holder,holding,first,count,fill,spare,spare_room,low,high,capacity,capacity_ok,bounded,tiled,values,after,inside,fits,allocatable,index,less)

# Cell offset + k of the band is the cell first + k of the reader.
def lent_read(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+lent: Lent<Holder,value>,+index: Nat,
  +less: {Nat.is_lt(index,U32.to_nat(lent_count(~Holder,~value,lent))) == True{} : Bool}) ->
  Held.BandRead(~Tile.Tile8,~Holder,~value,lent_holder(~Holder,~value,lent),lent_band(~Holder,~value,lent),lent_first(~Holder,~value,lent),index):
  Lent{+holder,+holding,+low,+high,+capacity,+capacity_ok,+bounded,+tiled,+values,+first,+count,+fill,+spare,+spare_room,placed,+allocatable} = lent
  placed_read(~Holder,~value,holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,first,count,fill,spare,spare_room,placed,allocatable,index,less)

def band_tree(band: Move.Band<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,band)))

def height_of(+band: Bands.Query) -> Nat:
  U32.to_nat(Domain.gradient_rows(Bands.geometry(band)))
def width_of(+band: Bands.Query) -> Nat:
  U32.to_nat(Domain.gradient_columns(Bands.geometry(band)))

# A band lent for the rows of a band query of an image of the batch: its first
# cell and count are those of the rows it copies, groups cells a pixel, and
# the cells of the batch lie in one buffer from cell base on for the image.
type Banded<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8> is Data:
  Banded{lent: Lent<Holder,value>,band: Bands.Query,groups: Nat,
    groups_exact: {Packing.output_groups(Bands.geometry(band)) == U32.from_nat(groups) : U32},
    images: Nat,image: Nat,image_inside: {Nat.is_lt(image,images) == True{} : Bool},
    first_exact: {lent_first(~Holder,~value,lent) == U32.from_nat(Nat.mul(Nat.add(Nat.mul(image,height_of(band)),U32.to_nat(Copy.begin(band))),Nat.mul(width_of(band),groups))) : U32},
    count_exact: {U32.to_nat(lent_count(~Holder,~value,lent)) == Nat.mul(U32.to_nat(Copy.row_count(band)),Nat.mul(width_of(band),groups)) : Nat},
    base: U32,base_exact: {base == U32.from_nat(Nat.mul(Nat.mul(image,Nat.mul(height_of(band),width_of(band))),groups)) : U32},
    cells: U32,sized: Depth.Sized<cells>,
    cells_exact: {U32.to_nat(cells) == Nat.mul(Nat.mul(Nat.mul(images,height_of(band)),width_of(band)),groups) : Nat},
    outputs: U32,outputs_room: {Nat.is_le(U32.to_nat(outputs),Nat.mul(groups,8n)) == True{} : Bool}}

def banded_lent(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,banded: Banded<Holder,value>) -> Lent<Holder,value>:
  Banded{lent,band,groups,groups_exact,images,image,image_inside,first_exact,count_exact,base,base_exact,cells,sized,cells_exact,outputs,outputs_room} = banded
  lent
def banded_band(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,banded: Banded<Holder,value>) -> Bands.Query:
  Banded{lent,band,groups,groups_exact,images,image,image_inside,first_exact,count_exact,base,base_exact,cells,sized,cells_exact,outputs,outputs_room} = banded
  band
def banded_base(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,banded: Banded<Holder,value>) -> U32:
  Banded{lent,band,groups,groups_exact,images,image,image_inside,first_exact,count_exact,base,base_exact,cells,sized,cells_exact,outputs,outputs_room} = banded
  base
def banded_cells(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,banded: Banded<Holder,value>) -> U32:
  Banded{lent,band,groups,groups_exact,images,image,image_inside,first_exact,count_exact,base,base_exact,cells,sized,cells_exact,outputs,outputs_room} = banded
  cells
def banded_outputs(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,banded: Banded<Holder,value>) -> U32:
  Banded{lent,band,groups,groups_exact,images,image,image_inside,first_exact,count_exact,base,base_exact,cells,sized,cells_exact,outputs,outputs_room} = banded
  outputs

# The rows the band holds, the gradient they come from and the slice a leaf
# reads.
def banded_source(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+banded: Banded<Holder,value>) -> Execution.Source:
  +band = banded_band(~Holder,~value,banded)
  Execution.Source{band_tree(lent_band(~Holder,~value,banded_lent(~Holder,~value,banded))),Held.band_offset(~Tile.Tile8,lent_band(~Holder,~value,banded_lent(~Holder,~value,banded))),Copy.begin(band),Copy.row_count(band)}
def banded_model(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+banded: Banded<Holder,value>) -> Buffer.Model<Tile.Tile8>:
  gradient_model(~Holder,~value,lent_holder(~Holder,~value,banded_lent(~Holder,~value,banded)),banded_cells(~Holder,~value,banded))
def banded_groups(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+banded: Banded<Holder,value>) -> U32:
  Packing.output_groups(Bands.geometry(banded_band(~Holder,~value,banded)))
def banded_slice(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+banded: Banded<Holder,value>) -> Rows.Slice:
  Rows.query_slice(banded_band(~Holder,~value,banded),banded_source(~Holder,~value,banded),banded_model(~Holder,~value,banded),banded_base(~Holder,~value,banded),
    banded_groups(~Holder,~value,banded))

# One cell of the band holds the cell of the gradient it was lent from.
def band_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+lent: Lent<Holder,value>,+band: Bands.Query,+groups: Nat,
  +groups_exact: {Packing.output_groups(Bands.geometry(band)) == U32.from_nat(groups) : U32},
  +images: Nat,+image: Nat,+image_inside: {Nat.is_lt(image,images) == True{} : Bool},
  +first_exact: {lent_first(~Holder,~value,lent) == U32.from_nat(Nat.mul(Nat.add(Nat.mul(image,height_of(band)),U32.to_nat(Copy.begin(band))),Nat.mul(width_of(band),groups))) : U32},
  +count_exact: {U32.to_nat(lent_count(~Holder,~value,lent)) == Nat.mul(U32.to_nat(Copy.row_count(band)),Nat.mul(width_of(band),groups)) : Nat},
  +base: U32,+base_exact: {base == U32.from_nat(Nat.mul(Nat.mul(image,Nat.mul(height_of(band),width_of(band))),groups)) : U32},
  +cells: U32,+sized: Depth.Sized<cells>,
  +cells_exact: {U32.to_nat(cells) == Nat.mul(Nat.mul(Nat.mul(images,height_of(band)),width_of(band)),groups) : Nat},
  +outputs: U32,+outputs_room: {Nat.is_le(U32.to_nat(outputs),Nat.mul(groups,8n)) == True{} : Bool},
  +output: Nat,+row: Nat,+column: Nat,
  +output_inside: {Nat.is_lt(output,U32.to_nat(outputs)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Copy.row_count(band))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,width_of(band)) == True{} : Bool}) ->
  Contents.Cell(Rows.query_slice(band,Execution.Source{band_tree(lent_band(~Holder,~value,lent)),Held.band_offset(~Tile.Tile8,lent_band(~Holder,~value,lent)),Copy.begin(band),Copy.row_count(band)},
    gradient_model(~Holder,~value,lent_holder(~Holder,~value,lent),cells),base,Packing.output_groups(Bands.geometry(band))),output,row,column):
  +holder = lent_holder(~Holder,~value,lent)
  +tree = band_tree(lent_band(~Holder,~value,lent))
  +offset = Held.band_offset(~Tile.Tile8,lent_band(~Holder,~value,lent))
  +model = gradient_model(~Holder,~value,holder,cells)
  +geometry = Bands.geometry(band)
  +begin = Copy.begin(band)
  +gw = Domain.gradient_columns(geometry)
  +groups_word = Packing.output_groups(geometry)
  +first = lent_first(~Holder,~value,lent)
  +h = height_of(band)
  +w = width_of(band)
  +b = U32.to_nat(begin)
  +q = Nat.div(output,8n)
  +lane = (U32.from_nat(output) % 8 : U32)
  +number = Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.add(Nat.mul(image,h),b),row),w),column),groups),q)
  +j = Nat.add(q,Nat.mul(Nat.add(Nat.mul(row,w),column),groups))
  +start = Nat.mul(Nat.add(Nat.mul(image,h),b),Nat.mul(w,groups))
  +image_base = Nat.mul(Nat.mul(image,Nat.mul(h,w)),groups)
  +gidx = Nat.add(Nat.add(image_base,q),Nat.mul(Nat.add(Nat.mul(Nat.add(b,row),w),column),groups))
  +offset_n = U32.to_nat(offset)
  +global_row = U32.from_nat(Nat.add(b,row))
  +local_address = ((offset + (U32.from_nat(output) / 8 : U32) : U32) + ((global_row - begin : U32) * gw + U32.from_nat(column) : U32) * groups_word : U32)
  +global_address = ((base + (U32.from_nat(output) / 8 : U32) : U32) + (global_row * gw + U32.from_nat(column) : U32) * groups_word : U32)
  +group_exact = group_word(output,outputs,Bounds.strict_le(output,U32.to_nat(outputs),output_inside))
  +width_exact = Equal.sym(U32,U32.from_nat(w),gw,Arithmetic.native_roundtrip(gw))
  +offset_exact = Equal.sym(U32,U32.from_nat(offset_n),offset,Arithmetic.native_roundtrip(offset))
  +local_cell_word = cell_word(offset,offset_n,offset_exact,(U32.from_nat(output) / 8 : U32),q,group_exact,
    (global_row - begin : U32),row,local_word(begin,row),gw,w,width_exact,U32.from_nat(column),column,{==},groups_word,groups,groups_exact)
  +global_cell_word = cell_word(base,image_base,base_exact,(U32.from_nat(output) / 8 : U32),q,group_exact,
    global_row,Nat.add(b,row),{==},gw,w,width_exact,U32.from_nat(column),column,{==},groups_word,groups,groups_exact)
  +q_inside = DivisionOrder.quotient_below(output,groups,8n,{==},Bounds.nat_trans(output,U32.to_nat(outputs),Nat.mul(groups,8n),output_inside,outputs_room))
  +j_less = {
    %Equal.sym(Nat,U32.to_nat(lent_count(~Holder,~value,lent)),Nat.mul(U32.to_nat(Copy.row_count(band)),Nat.mul(w,groups)),count_exact) :
      {Nat.is_lt(j,_) == True{} : Bool}
    band_inside(row,U32.to_nat(Copy.row_count(band)),w,column,groups,q,row_inside,column_inside,q_inside) :
    {Nat.is_lt(j,U32.to_nat(lent_count(~Holder,~value,lent))) == True{} : Bool}}
  +number_less = {
    %Equal.sym(Nat,U32.to_nat(cells),Nat.mul(Nat.mul(Nat.mul(images,h),w),groups),cells_exact) : {Nat.is_lt(number,_) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(image,h),b),row),Nat.add(Nat.mul(image,h),Nat.add(b,row)),Addition.associative(Nat.mul(image,h),b,row)) :
      {Nat.is_lt(Nat.add(Nat.mul(Nat.add(Nat.mul(_,w),column),groups),q),Nat.mul(Nat.mul(Nat.mul(images,h),w),groups)) == True{} : Bool}
    number_inside(image,images,h,Nat.add(b,row),w,column,groups,q,image_inside,copy_row_inside(band,row,row_inside),column_inside,q_inside) :
    {Nat.is_lt(number,U32.to_nat(cells)) == True{} : Bool}}
  +moved_word = Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(first),j)),(U32.from_nat(U32.to_nat(first)) + U32.from_nat(j) : U32),U32.from_nat(number),
      Equal.sym(U32,(U32.from_nat(U32.to_nat(first)) + U32.from_nat(j) : U32),U32.from_nat(Nat.add(U32.to_nat(first),j)),WordArithmetic.native_add(U32.to_nat(first),j)),
      {%Equal.sym(U32,U32.from_nat(U32.to_nat(first)),first,Arithmetic.native_roundtrip(first)) : {(_ + U32.from_nat(j) : U32) == U32.from_nat(number) : U32}
      %Equal.sym(U32,first,U32.from_nat(start),first_exact) : {(_ + U32.from_nat(j) : U32) == U32.from_nat(number) : U32}
      Equal.trans(U32,(U32.from_nat(start) + U32.from_nat(j) : U32),U32.from_nat(Nat.add(start,j)),U32.from_nat(number),
        WordArithmetic.native_add(start,j),
        Equal.cong(Nat,U32,U32.from_nat,Nat.add(start,j),number,band_number(Nat.mul(image,h),b,row,w,column,groups,q))) :
      {(U32.from_nat(U32.to_nat(first)) + U32.from_nat(j) : U32) == U32.from_nat(number) : U32}})
  +local_read = Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,tree,local_address),Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(offset_n,j))),value(holder,U32.from_nat(number)),
      Equal.cong(U32,Tile.Tile8,word => Tree.read(Tile.Tile8,tree,word),local_address,U32.from_nat(Nat.add(offset_n,j)),
        Equal.trans(U32,local_address,U32.from_nat(Nat.add(Nat.add(offset_n,q),Nat.mul(Nat.add(Nat.mul(row,w),column),groups))),U32.from_nat(Nat.add(offset_n,j)),
          local_cell_word,
          Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.add(offset_n,q),Nat.mul(Nat.add(Nat.mul(row,w),column),groups)),Nat.add(offset_n,j),
            Addition.associative(offset_n,q,Nat.mul(Nat.add(Nat.mul(row,w),column),groups))))),
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(offset_n,j))),value(holder,U32.from_nat(Nat.add(U32.to_nat(first),j))),value(holder,U32.from_nat(number)),
        lent_read(~Holder,~value,lent,j,j_less),
        Equal.cong(U32,Tile.Tile8,word => value(holder,word),U32.from_nat(Nat.add(U32.to_nat(first),j)),U32.from_nat(number),moved_word)))
  +global_read = Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,model),global_address),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,model),U32.from_nat(number)),
      value(holder,U32.from_nat(number)),
      Equal.cong(U32,Tile.Tile8,word => Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,model),word),global_address,U32.from_nat(number),
        Equal.trans(U32,global_address,U32.from_nat(gidx),U32.from_nat(number),global_cell_word,
          Equal.cong(Nat,U32,U32.from_nat,gidx,number,image_number(image,h,b,row,w,column,groups,q)))),
      gradient_cell(~Holder,~value,holder,cells,sized,number,number_less))
  Equal.cong(Tile.Tile8,F32,cell => Cells.lane_of(cell,lane),Tree.read(Tile.Tile8,tree,local_address),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,model),global_address),
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,tree,local_address),value(holder,U32.from_nat(number)),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,model),global_address),
      local_read,Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,model),global_address),value(holder,U32.from_nat(number)),global_read)))

# ---- Every cell of the band ----

def outputs_of(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,banded: Banded<Holder,value>) -> Nat:
  U32.to_nat(banded_outputs(~Holder,~value,banded))
def rows_of(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,banded: Banded<Holder,value>) -> Nat:
  U32.to_nat(Copy.row_count(banded_band(~Holder,~value,banded)))
def columns_of(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,banded: Banded<Holder,value>) -> Nat:
  width_of(banded_band(~Holder,~value,banded))

def Planes(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+banded: Banded<Holder,value>,+output: Nat) -> Data:
  Members.All(Nat,{row => Members.All(Nat,{column => Contents.Cell(banded_slice(~Holder,~value,banded),output,row,column) : Nat -> Data},
    Ranges.range(columns_of(~Holder,~value,banded),0n)) : Nat -> Data},Ranges.range(rows_of(~Holder,~value,banded),0n))

type Plane<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8> is Data:
  Plane{banded: Banded<Holder,value>,output: Nat,output_inside: {Nat.is_lt(output,outputs_of(~Holder,~value,banded)) == True{} : Bool}}
def plane_banded(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,plane: Plane<Holder,value>) -> Banded<Holder,value>:
  Plane{banded,output,output_inside} = plane
  banded
def plane_output(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,plane: Plane<Holder,value>) -> Nat:
  Plane{banded,output,output_inside} = plane
  output
def plane_rows(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,plane: Plane<Holder,value>) -> Nat:
  rows_of(~Holder,~value,plane_banded(~Holder,~value,plane))
def Lines(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+plane: Plane<Holder,value>,+row: Nat) -> Data:
  Members.All(Nat,{column => Contents.Cell(banded_slice(~Holder,~value,plane_banded(~Holder,~value,plane)),plane_output(~Holder,~value,plane),row,column) : Nat -> Data},
    Ranges.range(columns_of(~Holder,~value,plane_banded(~Holder,~value,plane)),0n))

type Line<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8> is Data:
  Line{plane: Plane<Holder,value>,row: Nat,row_inside: {Nat.is_lt(row,plane_rows(~Holder,~value,plane)) == True{} : Bool}}
def line_plane(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,line: Line<Holder,value>) -> Plane<Holder,value>:
  Line{plane,row,row_inside} = line
  plane
def line_row(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,line: Line<Holder,value>) -> Nat:
  Line{plane,row,row_inside} = line
  row
def line_columns(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,line: Line<Holder,value>) -> Nat:
  columns_of(~Holder,~value,plane_banded(~Holder,~value,line_plane(~Holder,~value,line)))
def Point(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+line: Line<Holder,value>,+column: Nat) -> Data:
  +plane = line_plane(~Holder,~value,line)
  Contents.Cell(banded_slice(~Holder,~value,plane_banded(~Holder,~value,plane)),plane_output(~Holder,~value,plane),line_row(~Holder,~value,line),column)

def point(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+line: Line<Holder,value>,+column: Nat,
  +inside: {Nat.is_lt(column,line_columns(~Holder,~value,line)) == True{} : Bool}) -> Point(~Holder,~value,line,column):
  Line{+plane,+row,+row_inside} = line
  Plane{+banded,+output,+output_inside} = plane
  Banded{+lent,+band,+groups,+groups_exact,+images,+image,+image_inside,+first_exact,+count_exact,+base,+base_exact,+cells,+sized,+cells_exact,+outputs,+outputs_room} = banded
  band_cell(~Holder,~value,lent,band,groups,groups_exact,images,image,image_inside,first_exact,count_exact,base,base_exact,cells,sized,cells_exact,outputs,outputs_room,
    output,row,column,output_inside,row_inside,inside)
def points(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+plane: Plane<Holder,value>,+row: Nat,
  +inside: {Nat.is_lt(row,plane_rows(~Holder,~value,plane)) == True{} : Bool}) -> Lines(~Holder,~value,plane,row):
  +columns = columns_of(~Holder,~value,plane_banded(~Holder,~value,plane))
  Members.range_all(~Line<Holder,value>,~Point(~Holder,~value),~line_columns(~Holder,~value),~point(~Holder,~value),columns,0n,Line{plane,row,inside},Order.reflexive(columns))
def lines(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+banded: Banded<Holder,value>,+output: Nat,
  +inside: {Nat.is_lt(output,outputs_of(~Holder,~value,banded)) == True{} : Bool}) -> Planes(~Holder,~value,banded,output):
  +rows = rows_of(~Holder,~value,banded)
  Members.range_all(~Plane<Holder,value>,~Lines(~Holder,~value),~plane_rows(~Holder,~value),~points(~Holder,~value),rows,0n,Plane{banded,output,inside},Order.reflexive(rows))

# The band holds the rows it copies of the image, as the buffer of the
# gradient holds them.
def band_contents(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+banded: Banded<Holder,value>) ->
  Contents.Content(banded_slice(~Holder,~value,banded),banded_outputs(~Holder,~value,banded)):
  +outputs = outputs_of(~Holder,~value,banded)
  Members.range_all(~Banded<Holder,value>,~Planes(~Holder,~value),~outputs_of(~Holder,~value),~lines(~Holder,~value),outputs,0n,banded,Order.reflexive(outputs))

def band_witness(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+banded: Banded<Holder,value>) ->
  Witnesses.Witness<banded_band(~Holder,~value,banded),banded_source(~Holder,~value,banded),banded_model(~Holder,~value,banded),banded_base(~Holder,~value,banded),
    banded_groups(~Holder,~value,banded),banded_outputs(~Holder,~value,banded)>:
  Witnesses.Retained{BandOrder.copy_range(banded_band(~Holder,~value,banded)),Coverage.Exact{{==},{==}},band_contents(~Holder,~value,banded)}
