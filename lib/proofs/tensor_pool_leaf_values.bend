# Leaf pooling: complete output values over the logical input. The pool is
# two checked sweeps: along the columns of every row into fresh planes
# [..,height,out_width], then along the rows into the fresh output. Each
# output holds the ordered fold, from pad, of the k lines of its window in
# increasing ky; a line is the ordered fold, from pad, of its k samples in
# increasing kx; a line or a sample outside the input is pad. No exchange,
# reassociation or law on the operation is used. `at` states this fold by the
# plane, row and column of a cell, and `cell` shows the two sweeps hold it at
# the index of that cell.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../storage_buffer.bend as Storage
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_contract.bend as Contract
import ./tensor_gather_values.bend as GatherValues
import ./tensor_copy_values.bend as CopyValues
import ./tensor_sweep_values.bend as Sweeps
import ./tensor_operations_refinement.bend as OperationsLaws
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./storage_shape.bend as Shape
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./encoded_coordinates.bend as Encoded
import ./machine_limit_bounds.bend as Limits
import ./boolean_logic.bend as Boolean
import ./ordered_fold_refinement.bend as Fold

# The logical input read through the two sweeps of a pool.
def context(~Element: Data,~Logical: Data,logical: Logical,+pool: Access.Pool,+origin: U32,+pad: Element) -> Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>:
  Sweeps.Swept{Sweeps.Swept{logical,Access.pool_columns(pool),pad},Access.pool_rows(pool,origin),pad}

# ---- One output cell by its coordinates ----

# The samples of one output column along one row of a plane: a sample inside
# the row reads the input at the row-major index of its plane, row and
# column; a sample outside is pad.
type Run<-Element: Data> is Data:
  Run{plane: Nat,height: Nat,width: U32,row: Nat,first: U32,padding: U32,pad: Element}

def run_sample(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,logical: Logical,run: Run<Element>,tap: U32) -> Element:
  Run{+plane,+height,+width,+row,+first,+padding,+pad} = run
  +column = (first + tap - padding : U32)
  Sweeps.logical_when(~Element,~Logical,~original,pad,U32.is_lt(column,width),logical,U32.from_nat(RowMajor.index(plane,height,U32.to_nat(width),row,U32.to_nat(column))))

# The rows of the window of one output cell: a row inside the plane is the
# ordered fold, from pad, of its run; a row outside is pad.
type Cell<-Element: Data> is Data:
  Cell{plane: Nat,height: U32,width: U32,kernel: U32,padding: U32,first: U32,column: U32,pad: Element}

def line_when(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,logical: Logical,
  +plane: Nat,+height: U32,+width: U32,+kernel: U32,+padding: U32,+column: U32,+pad: Element,valid: Bool,+row: U32) -> Element:
  match valid:
    case False{}: pad
    case True{}:
      Fold.values(~Logical,~Element,~Element,~Run<Element>,~run_sample(~Element,~Logical,~original),~operation,U32.to_nat(kernel),0,logical,
        Run{plane,U32.to_nat(height),width,U32.to_nat(row),column,padding,pad},pad)

def line_sample(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,logical: Logical,cell: Cell<Element>,tap: U32) -> Element:
  Cell{+plane,+height,+width,+kernel,+padding,+first,+column,+pad} = cell
  +row = (first + tap - padding : U32)
  line_when(~Element,~Logical,~original,~operation,logical,plane,height,width,kernel,padding,column,pad,U32.is_lt(row,height),row)

# The pool of one output cell of a plane: the ordered fold, from pad, of the k
# rows of its window.
def at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,logical: Logical,+pool: Access.Pool,+origin: U32,+pad: Element,
  +plane: Nat,+row: Nat,+column: Nat) -> Element:
  Access.Pool{+height,+width,+kernel,+stride,+padding,out_height,out_width} = pool
  Fold.values(~Logical,~Element,~Element,~Cell<Element>,~line_sample(~Element,~Logical,~original,~operation),~operation,U32.to_nat(kernel),0,logical,
    Cell{plane,height,width,kernel,padding,(U32.from_nat(row) * stride + origin : U32),(U32.from_nat(column) * stride + 0 : U32),pad},pad)

def unit_collapse(+value: Nat) -> {Nat.add(Nat.mul(value,1n),0n) == value : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(value,1n),0n),Nat.mul(value,1n),value,Addition.zero_right(Nat.mul(value,1n)),Order.times_one(value))

# The row-major index with step one is the index of a list of lines.
def unit_index(+block: Nat,+extent: Nat,+coordinate: Nat) -> {RowMajor.index(block,extent,1n,coordinate,0n) == Nat.add(Nat.mul(block,extent),coordinate) : Nat}:
  %Equal.sym(Nat,Nat.mul(extent,1n),extent,Order.times_one(extent)) : {Nat.add(Nat.mul(block,_),Nat.add(Nat.mul(coordinate,1n),0n)) == Nat.add(Nat.mul(block,extent),coordinate) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.mul(coordinate,1n),0n),coordinate,unit_collapse(coordinate)) : {Nat.add(Nat.mul(block,extent),_) == Nat.add(Nat.mul(block,extent),coordinate) : Nat}
  {==}

def unit_room(+outer: Nat,+rows: Nat,+width: Nat,+limit: Nat,room: {Nat.is_le(Nat.mul(outer,Nat.mul(rows,width)),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.mul(outer,rows),Nat.mul(width,1n)),limit) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(width,1n),width,Order.times_one(width)) : {Nat.is_le(Nat.mul(Nat.mul(outer,rows),_),limit) == True{} : Bool}
  %Equal.sym(Nat,Nat.mul(Nat.mul(outer,rows),width),Nat.mul(outer,Nat.mul(rows,width)),RowMajor.assoc(outer,rows,width)) : {Nat.is_le(_,limit) == True{} : Bool}
  room

# Each positive factor of a product inside a bound is inside it.
def inner_room(+outer: Nat,+middle: Nat,+inner: Nat,+limit: Nat,outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},middle_positive: {Nat.is_lt(0n,middle) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(outer,Nat.mul(middle,inner)),limit) == True{} : Bool}) -> {Nat.is_le(inner,limit) == True{} : Bool}:
  Order.transitive(inner,Nat.mul(middle,inner),limit,
    %RowMajor.multiply_commutative(inner,middle) : {Nat.is_le(inner,_) == True{} : Bool}
    RowMajor.product_covers(inner,middle,middle_positive),
    Order.transitive(Nat.mul(middle,inner),Nat.mul(outer,Nat.mul(middle,inner)),limit,
      %RowMajor.multiply_commutative(Nat.mul(middle,inner),outer) : {Nat.is_le(Nat.mul(middle,inner),_) == True{} : Bool}
      RowMajor.product_covers(Nat.mul(middle,inner),outer,outer_positive),room))

def middle_room(+outer: Nat,+middle: Nat,+inner: Nat,+limit: Nat,outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(outer,Nat.mul(middle,inner)),limit) == True{} : Bool}) -> {Nat.is_le(middle,limit) == True{} : Bool}:
  Order.transitive(middle,Nat.mul(middle,inner),limit,RowMajor.product_covers(middle,inner,inner_positive),
    Order.transitive(Nat.mul(middle,inner),Nat.mul(outer,Nat.mul(middle,inner)),limit,
      %RowMajor.multiply_commutative(Nat.mul(middle,inner),outer) : {Nat.is_le(Nat.mul(middle,inner),_) == True{} : Bool}
      RowMajor.product_covers(Nat.mul(middle,inner),outer,outer_positive),room))

# The native digits of an index (block * out_extent + along) * step + offset.
def digits(+value: Nat,+step: U32,+out_extent: U32,+block: Nat,+along: Nat,+offset: Nat,+limits: Machine.Limits,
  +exact: {Nat.add(Nat.mul(Nat.add(Nat.mul(block,U32.to_nat(out_extent)),along),U32.to_nat(step)),offset) == value : Nat},
  +room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +along_inside: {Nat.is_lt(along,U32.to_nat(out_extent)) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(step)) == True{} : Bool},
  +step_room: {Nat.is_le(U32.to_nat(step),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_room: {Nat.is_le(U32.to_nat(out_extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Sweeps.Decoded<U32.from_nat(value),step,out_extent,block,along,offset>:
  Sweeps.decoded(U32.from_nat(value),step,out_extent,block,along,offset,limits,
    Equal.trans(Nat,U32.to_nat(U32.from_nat(value)),value,Nat.add(Nat.mul(Nat.add(Nat.mul(block,U32.to_nat(out_extent)),along),U32.to_nat(step)),offset),
      Bounds.roundtrip(value,Machine.capacity(limits),Limits.capacity_fits(limits),room),
      Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(block,U32.to_nat(out_extent)),along),U32.to_nat(step)),offset),value,exact)),
    along_inside,offset_inside,step_room,out_room)

# A valid sample of the column sweep, at the planes index of a row, reads the
# input at the row-major index of that row and its column.
def run_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+height: U32,+width: U32,+out_width: U32,
  +plane: Nat,+row: Nat,+planes: Nat,+limits: Machine.Limits,+position: U32,valid: Bool,+column: U32,
  +exact: {U32.is_lt(column,width) == valid : Bool},
  +block: {U32.to_nat(((position / 1 : U32) / out_width : U32)) == Nat.add(Nat.mul(plane,U32.to_nat(height)),row) : Nat},
  +offset: {U32.to_nat((position % 1 : U32)) == 0n : Nat},
  +plane_inside: {Nat.is_lt(plane,planes) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(height)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Boolean.When(valid,{original(logical,(((position / 1 : U32) / out_width : U32) * width * 1 + (position % 1 : U32) + column * 1 : U32)) ==
    original(logical,U32.from_nat(RowMajor.index(plane,U32.to_nat(height),U32.to_nat(width),row,U32.to_nat(column)))) : Element}):
  match valid:
    case False{}: Unit{}
    case True{}:
      +tall = U32.to_nat(height)
      +wide = U32.to_nat(width)
      +line = Nat.add(Nat.mul(plane,tall),row)
      +sample = U32.to_nat(column)
      +index = RowMajor.index(plane,tall,wide,row,sample)
      +word = (((position / 1 : U32) / out_width : U32) * width * 1 + (position % 1 : U32) + column * 1 : U32)
      +address = Equal.trans(Nat,U32.to_nat(word),RowMajor.index(line,wide,1n,sample,0n),index,
        Sweeps.address_exact(((position / 1 : U32) / out_width : U32),width,1,(position % 1 : U32),column,line,0n,Nat.mul(planes,tall),limits,block,offset,
          RowMajor.radix(plane,planes,row,tall,plane_inside,row_inside),{==},Bounds.as_nat_lt(column,width,exact),
          unit_room(planes,tall,wide,U32.to_nat(Machine.capacity(limits)),input_room)),
        Equal.trans(Nat,RowMajor.index(line,wide,1n,sample,0n),Nat.add(Nat.mul(line,wide),sample),index,unit_index(line,wide,sample),RowMajor.mixed(plane,tall,wide,row,sample)))
      %Equal.sym(U32,word,U32.from_nat(index),Encoded.encoding(word,index,address)) : {original(logical,_) == original(logical,U32.from_nat(index)) : Element}
      {==}

# One tap of the column sweep at the planes index of a row and column: the
# same tap of the run of that row.
def run_tap(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+pad: Element,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_width: U32,
  +plane: Nat,+row: Nat,+column: Nat,+planes: Nat,+limits: Machine.Limits,+position: U32,+tap: U32,
  +digits: Sweeps.Decoded<position,1,out_width,Nat.add(Nat.mul(plane,U32.to_nat(height)),row),column,0n>,
  +plane_inside: {Nat.is_lt(plane,planes) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(height)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Sweeps.logical_line(~Element,~Logical,~original,logical,Access.line_at(~Element,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad,position),tap) ==
    run_sample(~Element,~Logical,~original,logical,Run{plane,U32.to_nat(height),width,row,(U32.from_nat(column) * stride + 0 : U32),padding,pad},tap) : Element}:
  Sweeps.Decoded{+block,+along,+offset} = digits
  +base = ((position / 1 : U32) / out_width * width * 1 + position % 1 : U32)
  +sample = (U32.from_nat(column) * stride + 0 + tap - padding : U32)
  %Equal.sym(U32,((position / 1 : U32) % out_width : U32),U32.from_nat(column),Encoded.encoding(((position / 1 : U32) % out_width : U32),column,along)) :
    {Sweeps.logical_line(~Element,~Logical,~original,logical,Access.Line{base,(_ * stride + 0 : U32),width,1,padding,pad},tap) ==
      run_sample(~Element,~Logical,~original,logical,Run{plane,U32.to_nat(height),width,row,(U32.from_nat(column) * stride + 0 : U32),padding,pad},tap) : Element}
  Sweeps.moved(~Element,~Logical,~original,~Logical,~original,logical,logical,pad,U32.is_lt(sample,width),(base + sample * 1 : U32),
    U32.from_nat(RowMajor.index(plane,U32.to_nat(height),U32.to_nat(width),row,U32.to_nat(sample))),
    run_read(~Element,~Logical,~original,logical,height,width,out_width,plane,row,planes,limits,position,U32.is_lt(sample,width),sample,{==},block,offset,plane_inside,row_inside,input_room))

def run_taps(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,remaining: Nat,+tap: U32,+logical: Logical,+pad: Element,
  +height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_width: U32,+plane: Nat,+row: Nat,+column: Nat,+planes: Nat,+limits: Machine.Limits,+position: U32,
  +digits: Sweeps.Decoded<position,1,out_width,Nat.add(Nat.mul(plane,U32.to_nat(height)),row),column,0n>,
  +plane_inside: {Nat.is_lt(plane,planes) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(height)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Fold.Agree(~Element,~Logical,~Access.Line<Element>,~Sweeps.logical_line(~Element,~Logical,~original),~Logical,~Run<Element>,~run_sample(~Element,~Logical,~original),
    logical,Access.line_at(~Element,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad,position),
    logical,Run{plane,U32.to_nat(height),width,row,(U32.from_nat(column) * stride + 0 : U32),padding,pad},remaining,tap):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{run_tap(~Element,~Logical,~original,logical,pad,height,width,kernel,stride,padding,out_width,plane,row,column,planes,limits,position,tap,digits,plane_inside,row_inside,input_room),
        run_taps(~Element,~Logical,~original,rest,(tap + 1 : U32),logical,pad,height,width,kernel,stride,padding,out_width,plane,row,column,planes,limits,position,digits,plane_inside,row_inside,input_room)}

# The column sweep at the planes index of a row and column is the fold of the
# run of that row.
def run_value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,
  +height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_width: U32,+plane: Nat,+row: Nat,+column: Nat,+planes: Nat,+limits: Machine.Limits,
  +plane_inside: {Nat.is_lt(plane,planes) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(height)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(out_width)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +planes_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Sweeps.value(~Element,~Logical,~original,~operation,Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},
      U32.from_nat(RowMajor.index(plane,U32.to_nat(height),U32.to_nat(out_width),row,column))) ==
    Fold.values(~Logical,~Element,~Element,~Run<Element>,~run_sample(~Element,~Logical,~original),~operation,U32.to_nat(kernel),0,logical,
      Run{plane,U32.to_nat(height),width,row,(U32.from_nat(column) * stride + 0 : U32),padding,pad},pad) : Element}:
  +tall = U32.to_nat(height)
  +wide = U32.to_nat(out_width)
  +limit = U32.to_nat(Machine.capacity(limits))
  +line = Nat.add(Nat.mul(plane,tall),row)
  +index = RowMajor.index(plane,tall,wide,row,column)
  +out_room = inner_room(planes,tall,wide,limit,Order.positive_bound(plane,planes,plane_inside),Order.positive_bound(row,tall,row_inside),planes_room)
  +line_digits = digits(index,1,out_width,line,column,0n,limits,
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(line,wide),column),1n),0n),Nat.add(Nat.mul(line,wide),column),index,
      unit_collapse(Nat.add(Nat.mul(line,wide),column)),RowMajor.mixed(plane,tall,wide,row,column)),
    Bounds.strict_le(index,limit,Bounds.nat_trans(index,Nat.mul(planes,Nat.mul(tall,wide)),limit,RowMajor.in_bounds(plane,planes,tall,wide,row,column,plane_inside,row_inside,column_inside),planes_room)),
    column_inside,{==},Order.transitive(1n,wide,limit,Order.strict_follow(0n,wide,Order.positive_bound(column,wide,column_inside)),out_room),out_room)
  Fold.agreed(~Element,~Element,~Logical,~Access.Line<Element>,~Sweeps.logical_line(~Element,~Logical,~original),~Logical,~Run<Element>,~run_sample(~Element,~Logical,~original),~operation,
    U32.to_nat(kernel),0,logical,Access.line_at(~Element,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad,U32.from_nat(index)),
    logical,Run{plane,tall,width,row,(U32.from_nat(column) * stride + 0 : U32),padding,pad},pad,
    run_taps(~Element,~Logical,~original,U32.to_nat(kernel),0,logical,pad,height,width,kernel,stride,padding,out_width,plane,row,column,planes,limits,U32.from_nat(index),line_digits,
      plane_inside,row_inside,input_room))

# A valid row of the row sweep, at the output index of a cell, is the run of
# that row.
def line_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,
  +height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+out_width: U32,+plane: Nat,+column: Nat,+planes: Nat,+limits: Machine.Limits,
  +position: U32,valid: Bool,+row: U32,
  +exact: {U32.is_lt(row,height) == valid : Bool},
  +block: {U32.to_nat(((position / out_width : U32) / out_height : U32)) == plane : Nat},
  +offset: {U32.to_nat((position % out_width : U32)) == column : Nat},
  +plane_inside: {Nat.is_lt(plane,planes) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(out_width)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +planes_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Sweeps.logical_when(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),pad,valid,
      Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},
      (((position / out_width : U32) / out_height : U32) * height * out_width + (position % out_width : U32) + row * out_width : U32)) ==
    line_when(~Element,~Logical,~original,~operation,logical,plane,height,width,kernel,padding,(U32.from_nat(column) * stride + 0 : U32),pad,valid,row) : Element}:
  match valid:
    case False{}: {==}
    case True{}:
      +row_inside = Bounds.as_nat_lt(row,height,exact)
      +index = RowMajor.index(plane,U32.to_nat(height),U32.to_nat(out_width),U32.to_nat(row),column)
      +word = (((position / out_width : U32) / out_height : U32) * height * out_width + (position % out_width : U32) + row * out_width : U32)
      +address = Sweeps.address_exact(((position / out_width : U32) / out_height : U32),height,out_width,(position % out_width : U32),row,plane,column,planes,limits,block,offset,
        plane_inside,column_inside,row_inside,planes_room)
      %Equal.sym(U32,word,U32.from_nat(index),Encoded.encoding(word,index,address)) :
        {Sweeps.value(~Element,~Logical,~original,~operation,Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},_) ==
          Fold.values(~Logical,~Element,~Element,~Run<Element>,~run_sample(~Element,~Logical,~original),~operation,U32.to_nat(kernel),0,logical,
            Run{plane,U32.to_nat(height),width,U32.to_nat(row),(U32.from_nat(column) * stride + 0 : U32),padding,pad},pad) : Element}
      run_value(~Element,~Logical,~original,~operation,logical,pad,height,width,kernel,stride,padding,out_width,plane,U32.to_nat(row),column,planes,limits,
        plane_inside,row_inside,column_inside,input_room,planes_room)

# One tap of the row sweep at the output index of a cell: the same tap of the
# rows of that cell.
def line_tap(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,
  +height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+out_width: U32,+origin: U32,+plane: Nat,+row: Nat,+column: Nat,+planes: Nat,+limits: Machine.Limits,
  +position: U32,+tap: U32,+digits: Sweeps.Decoded<position,out_width,out_height,plane,row,column>,
  +plane_inside: {Nat.is_lt(plane,planes) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(out_width)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +planes_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Sweeps.logical_line(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},
      Access.line_at(~Element,Access.Sweep{height,out_width,kernel,stride,padding,out_height,origin},pad,position),tap) ==
    line_sample(~Element,~Logical,~original,~operation,logical,
      Cell{plane,height,width,kernel,padding,(U32.from_nat(row) * stride + origin : U32),(U32.from_nat(column) * stride + 0 : U32),pad},tap) : Element}:
  Sweeps.Decoded{+block,+along,+offset} = digits
  +sample = (U32.from_nat(row) * stride + origin + tap - padding : U32)
  %Equal.sym(U32,((position / out_width : U32) % out_height : U32),U32.from_nat(row),Encoded.encoding(((position / out_width : U32) % out_height : U32),row,along)) :
    {Sweeps.logical_line(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},
        Access.Line{((position / out_width : U32) / out_height * height * out_width + position % out_width : U32),(_ * stride + origin : U32),height,out_width,padding,pad},tap) ==
      line_sample(~Element,~Logical,~original,~operation,logical,
        Cell{plane,height,width,kernel,padding,(U32.from_nat(row) * stride + origin : U32),(U32.from_nat(column) * stride + 0 : U32),pad},tap) : Element}
  line_read(~Element,~Logical,~original,~operation,logical,pad,height,width,kernel,stride,padding,out_height,out_width,plane,column,planes,limits,position,
    U32.is_lt(sample,height),sample,{==},block,offset,plane_inside,column_inside,input_room,planes_room)

def line_taps(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,remaining: Nat,+tap: U32,+logical: Logical,+pad: Element,
  +height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+out_width: U32,+origin: U32,+plane: Nat,+row: Nat,+column: Nat,+planes: Nat,+limits: Machine.Limits,
  +position: U32,+digits: Sweeps.Decoded<position,out_width,out_height,plane,row,column>,
  +plane_inside: {Nat.is_lt(plane,planes) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(out_width)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +planes_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Fold.Agree(~Element,~Sweeps.Swept<Element,Logical>,~Access.Line<Element>,~Sweeps.logical_line(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation)),
    ~Logical,~Cell<Element>,~line_sample(~Element,~Logical,~original,~operation),
    Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},Access.line_at(~Element,Access.Sweep{height,out_width,kernel,stride,padding,out_height,origin},pad,position),
    logical,Cell{plane,height,width,kernel,padding,(U32.from_nat(row) * stride + origin : U32),(U32.from_nat(column) * stride + 0 : U32),pad},remaining,tap):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{line_tap(~Element,~Logical,~original,~operation,logical,pad,height,width,kernel,stride,padding,out_height,out_width,origin,plane,row,column,planes,limits,position,tap,digits,
          plane_inside,column_inside,input_room,planes_room),
        line_taps(~Element,~Logical,~original,~operation,rest,(tap + 1 : U32),logical,pad,height,width,kernel,stride,padding,out_height,out_width,origin,plane,row,column,planes,limits,position,digits,
          plane_inside,column_inside,input_room,planes_room)}

# The pool of a stack of planes, read at the output index of a plane, row and
# column, is the pool of that cell by its coordinates.
def cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,
  +height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+out_width: U32,+origin: U32,+plane: Nat,+row: Nat,+column: Nat,+planes: Nat,+limits: Machine.Limits,
  +plane_inside: {Nat.is_lt(plane,planes) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(out_height)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(out_width)) == True{} : Bool},
  +input_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +planes_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(planes,Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation,
      context(~Element,~Logical,logical,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},origin,pad),
      U32.from_nat(RowMajor.index(plane,U32.to_nat(out_height),U32.to_nat(out_width),row,column))) ==
    at(~Element,~Logical,~original,~operation,logical,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},origin,pad,plane,row,column) : Element}:
  +tall = U32.to_nat(out_height)
  +wide = U32.to_nat(out_width)
  +limit = U32.to_nat(Machine.capacity(limits))
  +index = RowMajor.index(plane,tall,wide,row,column)
  +plane_positive = Order.positive_bound(plane,planes,plane_inside)
  +cell_digits = digits(index,out_width,out_height,plane,row,column,limits,RowMajor.mixed(plane,tall,wide,row,column),
    Bounds.strict_le(index,limit,Bounds.nat_trans(index,Nat.mul(planes,Nat.mul(tall,wide)),limit,RowMajor.in_bounds(plane,planes,tall,wide,row,column,plane_inside,row_inside,column_inside),output_room)),
    row_inside,column_inside,inner_room(planes,tall,wide,limit,plane_positive,Order.positive_bound(row,tall,row_inside),output_room),
    middle_room(planes,tall,wide,limit,plane_positive,Order.positive_bound(column,wide,column_inside),output_room))
  Fold.agreed(~Element,~Element,~Sweeps.Swept<Element,Logical>,~Access.Line<Element>,~Sweeps.logical_line(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation)),
    ~Logical,~Cell<Element>,~line_sample(~Element,~Logical,~original,~operation),~operation,
    U32.to_nat(kernel),0,Sweeps.Swept{logical,Access.Sweep{width,1,kernel,stride,padding,out_width,0},pad},
    Access.line_at(~Element,Access.Sweep{height,out_width,kernel,stride,padding,out_height,origin},pad,U32.from_nat(index)),
    logical,Cell{plane,height,width,kernel,padding,(U32.from_nat(row) * stride + origin : U32),(U32.from_nat(column) * stride + 0 : U32),pad},pad,
    line_taps(~Element,~Logical,~original,~operation,U32.to_nat(kernel),0,logical,pad,height,width,kernel,stride,padding,out_height,out_width,origin,plane,row,column,planes,limits,
      U32.from_nat(index),cell_digits,plane_inside,column_inside,input_room,planes_room))

def swept_cells(pool: Access.Pool,channels: U32) -> Nat:
  Access.Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(out_width)))

def pooled_cells(pool: Access.Pool,channels: U32) -> Nat:
  Access.Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)))

def lines_of(pool: Access.Pool,+channels: U32) -> Maybe<&2,U32>:
  Access.Pool{+height,width,kernel,stride,padding,out_height,out_width} = pool
  Shapes.product([channels,height],Some{1})

# The outcome of a pool is the outcome of its second sweep.
def finished_content(~Element: Data,input: Storage.Buffer<Element>,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,input,outcome))) == CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) : Buffer.Content<Element>}:
  match outcome:
    case Operations.Accepted{(middle,Storage.Buffer{storage,length,certificate})}: {==}
    case Operations.Rejected{(middle,Storage.Buffer{storage,length,certificate})}: {==}

def finished_accepted(~Element: Data,input: Storage.Buffer<Element>,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,input,outcome))) == GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) : Bool}:
  match outcome:
    case Operations.Accepted{(middle,output)}: {==}
    case Operations.Rejected{(middle,output)}: {==}

# The second sweep reads the planes the first one wrote.
def rows_modeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+input: Buffer.Model<Element>,+middle: Buffer.Model<Element>,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+origin: U32,+channels: U32,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+ready: Buffer.Sized<Element,output>,
  +cells: {Sweeps.output_cells(Access.pool_rows(pool,origin),U32.to_nat(channels)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  facts: GatherValues.Facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},middle_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.Accepted{(Buffer.pack(~Element,input),Buffer.pack(~Element,middle))}))),
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,Buffer.pack(~Element,input),Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output))))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,origin,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,Buffer.pack(~Element,input),Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output)))))):
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+middle_tree,+middle_length,+middle_certificate} = middle
  Shape.Both{Contract.WellFormed{middle_exact,middle_content_ready},middle_values} = facts
  +input = {Buffer.Model{input_tree,input_length,input_certificate} : Buffer.Model<Element>}
  +middle = {Buffer.Model{middle_tree,middle_length,middle_certificate} : Buffer.Model<Element>}
  %Equal.sym(Buffer.Content<Element>,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,Buffer.pack(~Element,input),Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output))))),CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output)))),finished_content(~Element,Buffer.pack(~Element,input),Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output)))) :
    GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,origin,pad),out_shape,_)
  Sweeps.modeled_facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation,Sweeps.Swept{logical,Access.pool_columns(pool),pad},pad,Access.pool_rows(pool,origin),channels,middle,output,out_shape,limits,actual,
    Buffer.ready_content(~Element,middle,
      %Buffer.packed_contents(~Element,middle) : Buffer.ContentReady<Element,_>
      middle_content_ready),
    ready,
    %Buffer.packed_contents(~Element,middle) : Contract.Values(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},_)
    middle_values,
    cells,length_exact,
    %finished_accepted(~Element,Buffer.pack(~Element,input),Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output))) : {_ == True{} : Bool}
    accepted)

def rows_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,-input: Storage.Buffer<Element>,-middle: Storage.Buffer<Element>,
  input_observation: Buffer.Observation<Element,input>,middle_observation: Buffer.Observation<Element,middle>,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+origin: U32,+channels: U32,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+ready: Buffer.Sized<Element,output>,
  +cells: {Sweeps.output_cells(Access.pool_rows(pool,origin),U32.to_nat(channels)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  facts: GatherValues.Facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},middle_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.Accepted{(input,middle)}))),
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,input,Operations.sweep(~Element,~operation,pad,middle,Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output))))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,origin,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,input,Operations.sweep(~Element,~operation,pad,middle,Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output)))))):
  Buffer.Observation{+input_model,+input_equation} = input_observation
  Buffer.Observation{+middle_model,+middle_equation} = middle_observation
  %input_equation : GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,origin,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,_,Operations.sweep(~Element,~operation,pad,middle,Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output))))))
  %middle_equation : GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,origin,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,Buffer.pack(~Element,input_model),Operations.sweep(~Element,~operation,pad,_,Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output))))))
  rows_modeled(~Element,~Logical,~original,~operation,input_model,middle_model,logical,pad,pool,origin,channels,output,middle_shape,out_shape,limits,actual,ready,cells,length_exact,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input_model),input,input_equation) : GatherValues.Facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},middle_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.Accepted{(_,Buffer.pack(~Element,middle_model))})))
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,middle_model),middle,middle_equation) : GatherValues.Facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},middle_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.Accepted{(input,_)})))
    facts,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input_model),input,input_equation) : {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,_,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle_model),Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output))))) == True{} : Bool}
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,middle_model),middle,middle_equation) : {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,input,Operations.sweep(~Element,~operation,pad,_,Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output))))) == True{} : Bool}
    accepted)

def rowed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,first: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+origin: U32,+channels: U32,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+ready: Buffer.Sized<Element,output>,
  +cells: {Sweeps.output_cells(Access.pool_rows(pool,origin),U32.to_nat(channels)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  facts: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,first)) == True{} : Bool} -> GatherValues.Facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},middle_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,first))),
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_rows(~Element,~operation,pad,first,Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output)))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,origin,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_rows(~Element,~operation,pad,first,Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output))))):
  match first:
    case Operations.Accepted{(input,middle)}:
      rows_observed(~Element,~Logical,~original,~operation,input,middle,Buffer.observe(~Element,input),Buffer.observe(~Element,middle),logical,pad,pool,origin,channels,output,middle_shape,out_shape,limits,actual,ready,cells,length_exact,facts({==}),accepted)
    case Operations.Rejected{(input,middle)}:
      Empty.absurd(GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,origin,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_rows(~Element,~operation,pad,Operations.Rejected{(input,middle)},Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output))))),WordEncoding.false_true(accepted))

def swept_lines(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+lines: U32,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+origin: U32,+channels: U32,+input: Buffer.Model<Element>,+middle: Buffer.Model<Element>,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+input_ready: Buffer.Sized<Element,input>,+middle_ready: Buffer.Sized<Element,middle>,+ready: Buffer.Sized<Element,output>,
  +values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +middle_cells: {swept_cells(pool,channels) == Buffer.length(~Element,middle) : Nat},
  +middle_exact: {Buffer.length(~Element,middle) == ShapeLaws.elements(middle_shape) : Nat},
  +cells: {pooled_cells(pool,channels) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  +lines_exact: {lines_of(pool,channels) == Some{lines} : Maybe<&2,U32>},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_rows(~Element,~operation,pad,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,input),Access.pool_columns(pool),lines,Buffer.pack(~Element,middle)),Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output)))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,origin,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_rows(~Element,~operation,pad,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,input),Access.pool_columns(pool),lines,Buffer.pack(~Element,middle)),Access.pool_rows(pool,origin),channels,Buffer.pack(~Element,output))))):
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  +channel_count = U32.to_nat(channels)
  +rows = U32.to_nat(height)
  +columns = U32.to_nat(out_width)
  rowed(~Element,~Logical,~original,~operation,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,input),Access.pool_columns(pool),lines,Buffer.pack(~Element,middle)),logical,pad,pool,origin,channels,output,middle_shape,out_shape,limits,actual,ready,cells,length_exact,
    first_accepted => Sweeps.modeled_facts(~Element,~Logical,~original,~operation,logical,pad,Access.pool_columns(pool),lines,input,middle,middle_shape,limits,actual,input_ready,middle_ready,values,
      Equal.trans(Nat,Nat.mul(U32.to_nat(lines),Nat.mul(columns,1n)),Nat.mul(channel_count,Nat.mul(rows,columns)),Buffer.length(~Element,middle),
        %Equal.sym(Nat,Nat.mul(columns,1n),columns,Order.times_one(columns)) : {Nat.mul(U32.to_nat(lines),_) == Nat.mul(channel_count,Nat.mul(rows,columns)) : Nat}
        %Sweeps.two(channels,height,lines,lines_exact) : {Nat.mul(_,columns) == Nat.mul(channel_count,Nat.mul(rows,columns)) : Nat}
        RowMajor.assoc(channel_count,rows,columns),
        middle_cells),
      middle_exact,first_accepted),
    accepted)

def lined(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,lines: Maybe<&2,U32>,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+origin: U32,+channels: U32,+input: Buffer.Model<Element>,+middle: Buffer.Model<Element>,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+input_ready: Buffer.Sized<Element,input>,+middle_ready: Buffer.Sized<Element,middle>,+ready: Buffer.Sized<Element,output>,
  +values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +middle_cells: {swept_cells(pool,channels) == Buffer.length(~Element,middle) : Nat},
  +middle_exact: {Buffer.length(~Element,middle) == ShapeLaws.elements(middle_shape) : Nat},
  +cells: {pooled_cells(pool,channels) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  +lines_exact: {lines_of(pool,channels) == lines : Maybe<&2,U32>},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_lines(~Element,~operation,pad,lines,Buffer.pack(~Element,input),pool,origin,channels,Buffer.pack(~Element,middle),Buffer.pack(~Element,output)))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,origin,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_lines(~Element,~operation,pad,lines,Buffer.pack(~Element,input),pool,origin,channels,Buffer.pack(~Element,middle),Buffer.pack(~Element,output))))):
  match lines:
    case Some{+lines}:
      swept_lines(~Element,~Logical,~original,~operation,lines,logical,pad,pool,origin,channels,input,middle,output,middle_shape,out_shape,limits,actual,input_ready,middle_ready,ready,values,middle_cells,middle_exact,cells,length_exact,lines_exact,accepted)
    case None{}:
      Empty.absurd(GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,origin,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_lines(~Element,~operation,pad,None{},Buffer.pack(~Element,input),pool,origin,channels,Buffer.pack(~Element,middle),Buffer.pack(~Element,output))))),WordEncoding.false_true(accepted))

# One accepted checked pool of a Ready input with complete values, through
# planes and an output it writes whole: the output content is well formed for
# its shape and holds the pooled value of the logical input at every index.
def pooled_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+origin: U32,+channels: U32,+input: Buffer.Model<Element>,+middle: Buffer.Model<Element>,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+input_ready: Buffer.Sized<Element,input>,+middle_ready: Buffer.Sized<Element,middle>,+ready: Buffer.Sized<Element,output>,
  +values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +middle_cells: {swept_cells(pool,channels) == Buffer.length(~Element,middle) : Nat},
  +middle_exact: {Buffer.length(~Element,middle) == ShapeLaws.elements(middle_shape) : Nat},
  +cells: {pooled_cells(pool,channels) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,origin,channels,Buffer.pack(~Element,middle),Buffer.pack(~Element,output)))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,origin,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,origin,channels,Buffer.pack(~Element,middle),Buffer.pack(~Element,output))))):
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  lined(~Element,~Logical,~original,~operation,lines_of(pool,channels),logical,pad,pool,origin,channels,input,middle,output,middle_shape,out_shape,limits,actual,input_ready,middle_ready,ready,values,middle_cells,middle_exact,cells,length_exact,{==},accepted)

# A fresh buffer, observed as a model, has the length it was allocated with.
def fresh_length(~Element: Data,+fill: Element,+count: U32,+model: Buffer.Model<Element>,
  equation: {Buffer.pack(~Element,model) == Tensors.buffer(~Element,fill,count) : Storage.Buffer<Element>}) ->
  {Buffer.length(~Element,model) == U32.to_nat(count) : Nat}:
  +fresh = {Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)} : Buffer.Content<Element>}
  +allocated = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,model),Buffer.contents(~Element,Buffer.pack(~Element,model)),fresh,
    Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)),
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),Tensors.buffer(~Element,fill,count),equation) : {Buffer.contents(~Element,_) == fresh : Buffer.Content<Element>}
    AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)))
  Equal.trans(Nat,Buffer.length(~Element,model),Buffer.content_length(~Element,Buffer.content(~Element,model)),U32.to_nat(count),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,model)),Buffer.length(~Element,model),Buffer.content_length_of(~Element,model)),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,model),fresh,allocated))
