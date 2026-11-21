# The values the weight gradient products read from the cells two holdings
# hold: the gradient of an output at a position, and the input a tap reads
# there, each a lane of a cell of its reader. Over the bands an image lends,
# the products of a group read these values.
import Base
import ../kernel_weight_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../tensor_move.bend as Move
import ../ordered_matrix.bend as OrderedMatrix
import ../machine_limits.bend as Machine
import ../kernel_gemm.bend as KernelGemm
import ./kernel_weight_gradient_group.bend as Groups
import ./kernel_weight_gradient_packing.bend as Packing
import ./kernel_tile_refinement.bend as TileLaws
import ./tensor_move_readback.bend as Held
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./ordered_reduction_split.bend as Ordered
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./machine_limit_bounds.bend as Limits
import ./nat_ceiling.bend as Ceiling
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./boolean_logic.bend as Logic
import ./u32_full_range.bend as FullRange
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_multiplication.bend as Multiplication

# ---- The values of the cells two readers hold ----

# The cells of the input and of the gradient, each with its reader, and the
# first cell of an image in each.
type HeldSource<-X: Data,-D: Data> is Data:
  HeldSource{input: X,gradient: D,geometry: Kernel.Geometry,x_first: Nat,d_first: Nat}

def held_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,holder: Holder,index: Nat) -> Tile.Tile8:
  value(holder,U32.from_nat(index))

# Output output at a position: lane output % 8 of the cell of group output / 8
# at the position, the groups of a position side by side.
def held_input(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,source: HeldSource<X,D>,output: U32,position: Nat) -> F32:
  HeldSource{input,gradient,geometry,x_first,d_first} = source
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  +output = U32.to_nat(output)
  TileLaws.select_lane(Nat.mod(output,8n),dv(gradient,U32.from_nat(Nat.add(Nat.add(d_first,Nat.mul(position,Ceiling.ceiling(U32.to_nat(o),8n))),Nat.div(output,8n)))))

def held_weight(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,source: HeldSource<X,D>,tap: U32,position: Nat) -> F32:
  HeldSource{input,gradient,+geometry,x_first,d_first} = source
  Kernel.Geometry{c,o,h,w,k,s,p,rows,+columns,taps} = geometry
  +position = position
  Packing.tap_cell(~X,~held_cell(~X,~xv),input,geometry,x_first,tap,U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))))

def held_dot(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,source: HeldSource<X,D>,tap: U32,output: U32,count: Nat,first: Nat,initial: F32) -> F32:
  OrderedMatrix.dot(~U32,~Nat,~U32,~HeldSource<X,D>,~held_input(~X,~xv,~D,~dv),~held_weight(~X,~xv,~D,~dv),Ranges.range(count,first),tap,output,source,initial)

# ---- The bands of an image ----

# The input cells and the gradient cells of an image.
def input_cells(+geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{+c,o,+h,+w,k,s,p,rows,columns,taps} = geometry
  Nat.mul(Nat.mul(Ceiling.ceiling(U32.to_nat(c),8n),U32.to_nat(h)),U32.to_nat(w))

def input_word(+geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{+c,o,+h,+w,k,s,p,rows,columns,taps} = geometry
  ((c + 7) / 8 * h * w : U32)

def gradient_word(+geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,+o,h,w,k,s,p,+rows,+columns,taps} = geometry
  ((o + 7) / 8 * rows * columns : U32)

def input_band(holding: Held.Holding<Tile.Tile8>,+geometry: Kernel.Geometry,+image: U32) -> Move.Band<Tile.Tile8>:
  Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),(image * input_word(geometry) : U32),input_word(geometry),Tile.filled(0.0),Move.spare(~Tile.Tile8,Tile.filled(0.0)))

def gradient_band(holding: Held.Holding<Tile.Tile8>,+geometry: Kernel.Geometry,+image: U32) -> Move.Band<Tile.Tile8>:
  Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),(image * gradient_word(geometry) : U32),gradient_word(geometry),Tile.filled(0.0),Move.spare(~Tile.Tile8,Tile.filled(0.0)))

def band_model(band: Move.Band<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,band))

# What the groups of an image read: the models of its two bands and their
# offsets.
def band_source(+x: Held.Holding<Tile.Tile8>,+d: Held.Holding<Tile.Tile8>,+geometry: Kernel.Geometry,+image: U32) -> Groups.NumericSource:
  Groups.NumericSource{band_model(input_band(x,geometry,image)),band_model(gradient_band(d,geometry,image)),geometry,
    Held.band_offset(~Tile.Tile8,input_band(x,geometry,image)),Held.band_offset(~Tile.Tile8,gradient_band(d,geometry,image))}

# ---- Words ----

def product_word(+left: Nat,+right: U32) -> {(U32.from_nat(left) * right : U32) == U32.from_nat(Nat.mul(left,U32.to_nat(right))) : U32}:
  %Arithmetic.native_roundtrip(right) : {(U32.from_nat(left) * _ : U32) == U32.from_nat(Nat.mul(left,U32.to_nat(right))) : U32}
  Multiplication.native_encoded_multiply(left,U32.to_nat(right))

# The cells of value / 8 rounded up groups of first * second cells.
def cells_word(+limits: Machine.Limits,+value: U32,+first: U32,+second: U32,+room: {Nat.is_le(U32.to_nat(value),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {((value + 7) / 8 * first * second : U32) == U32.from_nat(Nat.mul(Nat.mul(Ceiling.ceiling(U32.to_nat(value),8n),U32.to_nat(first)),U32.to_nat(second))) : U32}:
  +blocks = Ceiling.ceiling(U32.to_nat(value),8n)
  %Equal.sym(U32,((value + 7) / 8 : U32),U32.from_nat(blocks),Groups.blocks_word(limits,value,room)) :
    {((_ * first : U32) * second : U32) == U32.from_nat(Nat.mul(Nat.mul(blocks,U32.to_nat(first)),U32.to_nat(second))) : U32}
  %Equal.sym(U32,(U32.from_nat(blocks) * first : U32),U32.from_nat(Nat.mul(blocks,U32.to_nat(first))),product_word(blocks,first)) :
    {(_ * second : U32) == U32.from_nat(Nat.mul(Nat.mul(blocks,U32.to_nat(first)),U32.to_nat(second))) : U32}
  product_word(Nat.mul(blocks,U32.to_nat(first)),second)

# The count of the cells of an image and its first cell, as numbers.
def count_exact(+word: U32,+count: Nat,+capacity: U32,+exact: {word == U32.from_nat(count) : U32},
  +room: {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool}) -> {U32.to_nat(word) == count : Nat}:
  %Equal.sym(U32,word,U32.from_nat(count),exact) : {U32.to_nat(_) == count : Nat}
  FullRange.roundtrip(count,capacity,room)

def first_exact(+image: Nat,+word: U32,+count: Nat,+capacity: U32,+exact: {word == U32.from_nat(count) : U32},
  +room: {Nat.is_le(Nat.mul(image,count),U32.to_nat(capacity)) == True{} : Bool}) -> {U32.to_nat((U32.from_nat(image) * word : U32)) == Nat.mul(image,count) : Nat}:
  %Equal.sym(U32,word,U32.from_nat(count),exact) : {U32.to_nat((U32.from_nat(image) * _ : U32)) == Nat.mul(image,count) : Nat}
  %Equal.sym(U32,(U32.from_nat(image) * U32.from_nat(count) : U32),U32.from_nat(Nat.mul(image,count)),Multiplication.native_encoded_multiply(image,count)) :
    {U32.to_nat(_) == Nat.mul(image,count) : Nat}
  FullRange.roundtrip(Nat.mul(image,count),capacity,room)

# Cell head + tail of the band of an image of count cells, as the model of
# the band, is the cell the holding holds there.
def band_read(~H: Data,~hv: H -> U32 -> Tile.Tile8,+holder: H,+holding: Held.Holding<Tile.Tile8>,+high: Nat,+limits: Machine.Limits,
  +bounded: {Nat.is_le(high,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,holding,0n,high),+values: Held.HoldingValues(~Tile.Tile8,~H,~hv,holder,holding),
  +image: Nat,+word: U32,+cells: Nat,+exact: {word == U32.from_nat(cells) : U32},
  +room: {Nat.is_le(Nat.add(Nat.mul(image,cells),cells),high) == True{} : Bool},
  +head: Nat,+tail: Nat,+inside: {Nat.is_lt(Nat.add(head,tail),cells) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_model(Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),(U32.from_nat(image) * word : U32),word,Tile.filled(0.0),Move.spare(~Tile.Tile8,Tile.filled(0.0))))),
      U32.from_nat(Nat.add(Nat.add(U32.to_nat(Held.band_offset(~Tile.Tile8,Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),(U32.from_nat(image) * word : U32),word,Tile.filled(0.0),Move.spare(~Tile.Tile8,Tile.filled(0.0))))),head),tail))) ==
    hv(holder,U32.from_nat(Nat.add(Nat.add(Nat.mul(image,cells),head),tail))) : Tile.Tile8}:
  %Equal.sym(Move.Spare<Tile.Tile8>,Move.spare(~Tile.Tile8,Tile.filled(0.0)),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))),Held.spare_packed(~Tile.Tile8,Tile.filled(0.0))) :
    {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_model(Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),(U32.from_nat(image) * word : U32),word,Tile.filled(0.0),_))),
      U32.from_nat(Nat.add(Nat.add(U32.to_nat(Held.band_offset(~Tile.Tile8,Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),(U32.from_nat(image) * word : U32),word,Tile.filled(0.0),_))),head),tail))) ==
    hv(holder,U32.from_nat(Nat.add(Nat.add(Nat.mul(image,cells),head),tail))) : Tile.Tile8}
  +capacity = Machine.capacity(limits)
  +before = Nat.mul(image,cells)
  +whole = Order.transitive(Nat.add(before,cells),high,U32.to_nat(capacity),room,bounded)
  +count_room = Order.drop_left_le(before,cells,U32.to_nat(capacity),whole)
  +counted = count_exact(word,cells,capacity,exact,count_room)
  +started = first_exact(image,word,cells,capacity,exact,Order.transitive(before,Nat.add(before,cells),U32.to_nat(capacity),Order.le_add(before,cells),whole))
  +first = U32.to_nat((U32.from_nat(image) * word : U32))
  +offset = U32.to_nat(Held.band_offset(~Tile.Tile8,Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),(U32.from_nat(image) * word : U32),word,Tile.filled(0.0),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))))))
  +index = Nat.add(head,tail)
  +fits = {%Equal.sym(Nat,first,before,started) : {Nat.is_le(Nat.add(_,U32.to_nat(word)),high) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(word),cells,counted) : {Nat.is_le(Nat.add(before,_),high) == True{} : Bool}
    room : {Nat.is_le(Nat.add(first,U32.to_nat(word)),high) == True{} : Bool}}
  +less = {%Equal.sym(Nat,U32.to_nat(word),cells,counted) : {Nat.is_lt(index,_) == True{} : Bool} inside : {Nat.is_lt(index,U32.to_nat(word)) == True{} : Bool}}
  +read = Held.band_model_value(~Tile.Tile8,~H,~hv,holder,holding,(U32.from_nat(image) * word : U32),word,Tile.filled(0.0),
    Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0)),Held.fresh_spare_room(~Tile.Tile8,Tile.filled(0.0)),0n,high,capacity,Limits.capacity_fits(limits),bounded,
    tiled,values,Order.zero_le(first),fits,Arithmetic.count_within(word,cells,capacity,1073741824,counted,count_room,Limits.capacity_allocatable(limits)),index,less)
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_model(Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),(U32.from_nat(image) * word : U32),word,Tile.filled(0.0),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0)))))),U32.from_nat(Nat.add(Nat.add(offset,head),tail))),
    hv(holder,U32.from_nat(Nat.add(first,index))),hv(holder,U32.from_nat(Nat.add(Nat.add(before,head),tail))),
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_model(Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),(U32.from_nat(image) * word : U32),word,Tile.filled(0.0),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0)))))),U32.from_nat(Nat.add(Nat.add(offset,head),tail))),
      Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_model(Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),(U32.from_nat(image) * word : U32),word,Tile.filled(0.0),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0)))))),U32.from_nat(Nat.add(offset,index))),hv(holder,U32.from_nat(Nat.add(first,index))),
      Equal.cong(Nat,Tile.Tile8,at => Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_model(Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),(U32.from_nat(image) * word : U32),word,Tile.filled(0.0),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0)))))),U32.from_nat(at)),Nat.add(Nat.add(offset,head),tail),Nat.add(offset,index),
        Addition.associative(offset,head,tail)),read),
    Equal.cong(Nat,Tile.Tile8,at => hv(holder,U32.from_nat(at)),Nat.add(first,index),Nat.add(Nat.add(before,head),tail),
      Equal.trans(Nat,Nat.add(first,index),Nat.add(before,index),Nat.add(Nat.add(before,head),tail),
        Equal.cong(Nat,Nat,start => Nat.add(start,index),first,before,started),
        Equal.sym(Nat,Nat.add(Nat.add(before,head),tail),Nat.add(before,index),Addition.associative(before,head,tail)))))

def taps_exact(+geometry: Kernel.Geometry,+domain: Groups.GeometryReady<geometry>) ->
  {Groups.tap_count(geometry) == Nat.mul(Groups.channel_count(geometry),Groups.square_size(geometry)) : Nat}:
  Groups.GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned,+taps_exact,pixels,gradients} = domain
  taps_exact

# A sample inside the image lies in the cells of the image: below the groups
# of its pixel, and the pixel below the pixels of the image.
def sample_inside(+channel: Nat,+channels: Nat,+y: Nat,+x: Nat,+height: Nat,+width: Nat,+padding: Nat,
  +channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},
  +inside: {(Nat.is_le(padding,y) && Nat.is_lt(y,Nat.add(padding,height))) && (Nat.is_le(padding,x) && Nat.is_lt(x,Nat.add(padding,width))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.div(channel,8n),Nat.mul(Nat.add(Nat.mul(Nat.sub(y,padding),width),Nat.sub(x,padding)),Ceiling.ceiling(channels,8n))),
    Nat.mul(Nat.mul(Ceiling.ceiling(channels,8n),height),width)) == True{} : Bool}:
  +rows = Logic.and_true_left(Nat.is_le(padding,y) && Nat.is_lt(y,Nat.add(padding,height)),Nat.is_le(padding,x) && Nat.is_lt(x,Nat.add(padding,width)),inside)
  +columns = Logic.and_true_right(Nat.is_le(padding,y) && Nat.is_lt(y,Nat.add(padding,height)),Nat.is_le(padding,x) && Nat.is_lt(x,Nat.add(padding,width)),inside)
  +row = Order.sub_lt(y,padding,height,Logic.and_true_left(Nat.is_le(padding,y),Nat.is_lt(y,Nat.add(padding,height)),rows),
    Logic.and_true_right(Nat.is_le(padding,y),Nat.is_lt(y,Nat.add(padding,height)),rows))
  +column = Order.sub_lt(x,padding,width,Logic.and_true_left(Nat.is_le(padding,x),Nat.is_lt(x,Nat.add(padding,width)),columns),
    Logic.and_true_right(Nat.is_le(padding,x),Nat.is_lt(x,Nat.add(padding,width)),columns))
  +blocks = Ceiling.ceiling(channels,8n)
  +pixel = Nat.add(Nat.mul(Nat.sub(y,padding),width),Nat.sub(x,padding))
  %Equal.sym(Nat,Nat.mul(Nat.mul(blocks,height),width),Nat.mul(blocks,Nat.mul(height,width)),RowMajor.assoc(blocks,height,width)) :
    {Nat.is_lt(Nat.add(Nat.div(channel,8n),Nat.mul(pixel,blocks)),_) == True{} : Bool}
  %RowMajor.multiply_commutative(Nat.mul(height,width),blocks) : {Nat.is_lt(Nat.add(Nat.div(channel,8n),Nat.mul(pixel,blocks)),_) == True{} : Bool}
  %Order.commutative(Nat.mul(pixel,blocks),Nat.div(channel,8n)) : {Nat.is_lt(_,Nat.mul(Nat.mul(height,width),blocks)) == True{} : Bool}
  RowMajor.radix(pixel,Nat.mul(height,width),Nat.div(channel,8n),blocks,RowMajor.radix(Nat.sub(y,padding),height,Nat.sub(x,padding),width,row,column),
    Groups.eight_block_inside(channels,channel,channel_inside))

# ---- An image lent from two holdings ----

# An image whose input cells and gradient cells two holdings hold from cell
# 0 on, each with the values of its reader.
type Lending<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8> is Data:
  Lending{input: X,gradient: D,x: Held.Holding<Tile.Tile8>,d: Held.Holding<Tile.Tile8>,geometry: Kernel.Geometry,domain: Groups.GeometryReady<geometry>,image: Nat,
    x_high: Nat,x_tiled: Held.Tiled(~Tile.Tile8,x,0n,x_high),x_values: Held.HoldingValues(~Tile.Tile8,~X,~xv,input,x),
    x_room: {Nat.is_le(Nat.add(Nat.mul(image,input_cells(geometry)),input_cells(geometry)),x_high) == True{} : Bool},
    x_bounded: {Nat.is_le(x_high,U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))) == True{} : Bool},
    d_high: Nat,d_tiled: Held.Tiled(~Tile.Tile8,d,0n,d_high),d_values: Held.HoldingValues(~Tile.Tile8,~D,~dv,gradient,d),
    d_room: {Nat.is_le(Nat.add(Nat.mul(image,Groups.gradient_extent(geometry)),Groups.gradient_extent(geometry)),d_high) == True{} : Bool},
    d_bounded: {Nat.is_le(d_high,U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))) == True{} : Bool}}

def lent_geometry(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,lending: Lending<X,xv,D,dv>) -> Kernel.Geometry:
  Lending{input,gradient,x,d,geometry,domain,image,x_high,x_tiled,x_values,x_room,x_bounded,d_high,d_tiled,d_values,d_room,d_bounded} = lending
  geometry

def lent_source(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,lending: Lending<X,xv,D,dv>) -> Groups.NumericSource:
  Lending{input,gradient,+x,+d,+geometry,domain,+image,x_high,x_tiled,x_values,x_room,x_bounded,d_high,d_tiled,d_values,d_room,d_bounded} = lending
  band_source(x,d,geometry,U32.from_nat(image))

def held_source(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,lending: Lending<X,xv,D,dv>) -> HeldSource<X,D>:
  Lending{input,gradient,x,d,+geometry,domain,+image,x_high,x_tiled,x_values,x_room,x_bounded,d_high,d_tiled,d_values,d_room,d_bounded} = lending
  HeldSource{input,gradient,geometry,Nat.mul(image,input_cells(geometry)),Nat.mul(image,Groups.gradient_extent(geometry))}

# The gradient a group reads at a position over the bands is the gradient the
# holding holds there.
def input_agrees(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lending: Lending<X,xv,D,dv>,+output: U32,+position: Nat,
  +output_inside: {Nat.is_lt(U32.to_nat(output),Groups.output_count(lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,Nat.mul(Groups.row_count(lent_geometry(~X,~xv,~D,~dv,lending)),Groups.column_count(lent_geometry(~X,~xv,~D,~dv,lending)))) == True{} : Bool}) ->
  {Groups.numeric_input(lent_source(~X,~xv,~D,~dv,lending),output,position) == held_input(~X,~xv,~D,~dv,held_source(~X,~xv,~D,~dv,lending),output,position) : F32}:
  Lending{+input,+gradient,+x,+d,+geometry,+domain,+image,x_high,x_tiled,x_values,x_room,x_bounded,+d_high,+d_tiled,+d_values,+d_room,+d_bounded} = lending
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  +limits = Groups.domain_limits(shape,domain)
  +cells = Groups.gradient_extent(shape)
  +blocks = Ceiling.ceiling(U32.to_nat(o),8n)
  +area = Nat.mul(U32.to_nat(rows),U32.to_nat(columns))
  +group = Nat.div(U32.to_nat(output),8n)
  +step = Nat.mul(position,blocks)
  +inside = {%Equal.sym(Nat,cells,Nat.mul(blocks,area),RowMajor.assoc(blocks,U32.to_nat(rows),U32.to_nat(columns))) : {Nat.is_lt(Nat.add(step,group),_) == True{} : Bool}
    %RowMajor.multiply_commutative(area,blocks) : {Nat.is_lt(Nat.add(step,group),_) == True{} : Bool}
    RowMajor.radix(position,area,group,blocks,position_inside,Groups.eight_block_inside(U32.to_nat(o),U32.to_nat(output),output_inside)) :
    {Nat.is_lt(Nat.add(step,group),cells) == True{} : Bool}}
  Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),tile),
    Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_model(gradient_band(d,shape,U32.from_nat(image)))),U32.from_nat(Nat.add(Nat.add(U32.to_nat(Held.band_offset(~Tile.Tile8,gradient_band(d,shape,U32.from_nat(image)))),step),group))),
    dv(gradient,U32.from_nat(Nat.add(Nat.add(Nat.mul(image,cells),step),group))),
    band_read(~D,~dv,gradient,d,d_high,limits,d_bounded,d_tiled,d_values,image,gradient_word(shape),cells,
      cells_word(limits,o,rows,columns,Groups.outputs_room(shape,domain)),d_room,step,group,inside))

# The input a tap reads at a position over the bands is the input the holding
# holds there.
def weight_agrees(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lending: Lending<X,xv,D,dv>,+tap: U32,+position: Nat,
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Groups.tap_count(lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool}) ->
  {Groups.numeric_weight(lent_source(~X,~xv,~D,~dv,lending),tap,position) == held_weight(~X,~xv,~D,~dv,held_source(~X,~xv,~D,~dv,lending),tap,position) : F32}:
  Lending{+input,+gradient,+x,+d,+geometry,+domain,+image,+x_high,+x_tiled,+x_values,+x_room,+x_bounded,d_high,d_tiled,d_values,d_room,d_bounded} = lending
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  +limits = Groups.domain_limits(shape,domain)
  +capacity = Machine.capacity(limits)
  +cells = input_cells(shape)
  +blocks = Ceiling.ceiling(U32.to_nat(c),8n)
  +channels = Groups.channels_room(shape,domain)
  +group_word = U32.to_nat(((c + 7) / 8 : U32))
  +groups = count_exact(((c + 7) / 8 : U32),blocks,capacity,Groups.blocks_word(limits,c,channels),
    Order.transitive(blocks,U32.to_nat(c),U32.to_nat(capacity),Ceiling.bounded(U32.to_nat(c),8n,{==}),channels))
  +square = Nat.mul(U32.to_nat(k),U32.to_nat(k))
  +channel = Nat.div(U32.to_nat(tap),square)
  +channel_inside = DivisionOrder.quotient_below(U32.to_nat(tap),U32.to_nat(c),square,Groups.square_positive(shape,domain),
    {%taps_exact(shape,domain) : {Nat.is_lt(U32.to_nat(tap),_) == True{} : Bool} tap_inside : {Nat.is_lt(U32.to_nat(tap),Nat.mul(U32.to_nat(c),square)) == True{} : Bool}})
  +row = U32.from_nat(Nat.div(position,U32.to_nat(columns)))
  +column = U32.from_nat(Nat.mod(position,U32.to_nat(columns)))
  +y = Nat.add(Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(k)),Nat.mul(U32.to_nat(row),U32.to_nat(s)))
  +x_at = Nat.add(Nat.mod(U32.to_nat(tap),U32.to_nat(k)),Nat.mul(U32.to_nat(column),U32.to_nat(s)))
  +pixel = Nat.add(Nat.mul(Nat.sub(y,U32.to_nat(p)),U32.to_nat(w)),Nat.sub(x_at,U32.to_nat(p)))
  Packing.sampled_agrees(~Buffer.Model<Tile.Tile8>,~Packing.model_cell,~X,~held_cell(~X,~xv),band_model(input_band(x,shape,U32.from_nat(image))),input,U32.to_nat(Held.band_offset(~Tile.Tile8,input_band(x,shape,U32.from_nat(image)))),
    Nat.mul(image,cells),channel,y,x_at,U32.to_nat(h),U32.to_nat(w),U32.to_nat(p),group_word,
    inside_true => band_read(~X,~xv,input,x,x_high,limits,x_bounded,x_tiled,x_values,image,input_word(shape),cells,
      cells_word(limits,c,h,w,channels),x_room,Nat.div(channel,8n),Nat.mul(pixel,group_word),
      {%Equal.sym(Nat,group_word,blocks,groups) : {Nat.is_lt(Nat.add(Nat.div(channel,8n),Nat.mul(pixel,_)),cells) == True{} : Bool}
        sample_inside(channel,U32.to_nat(c),y,x_at,U32.to_nat(h),U32.to_nat(w),U32.to_nat(p),channel_inside,inside_true) :
        {Nat.is_lt(Nat.add(Nat.div(channel,8n),Nat.mul(pixel,group_word)),cells) == True{} : Bool}}))

# ---- A product over the bands ----

# One lane of one tap of an image lent from two holdings.
type Lookup<-X: Data,-xv: X -> U32 -> Tile.Tile8,-D: Data,-dv: D -> U32 -> Tile.Tile8> is Data:
  Lookup{lending: Lending<X,xv,D,dv>,tap: U32,output: U32,
    tap_inside: {Nat.is_lt(U32.to_nat(tap),Groups.tap_count(lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
    output_inside: {Nat.is_lt(U32.to_nat(output),Groups.output_count(lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool}}

def lookup_size(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,lookup: Lookup<X,xv,D,dv>) -> Nat:
  Lookup{+lending,tap,output,tap_inside,output_inside} = lookup
  Nat.mul(Groups.row_count(lent_geometry(~X,~xv,~D,~dv,lending)),Groups.column_count(lent_geometry(~X,~xv,~D,~dv,lending)))

def LookupAgrees(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lookup: Lookup<X,xv,D,dv>,+position: Nat) -> Data:
  Lookup{+lending,+tap,+output,tap_inside,output_inside} = lookup
  KernelGemm.Both<{Groups.numeric_input(lent_source(~X,~xv,~D,~dv,lending),output,position) == held_input(~X,~xv,~D,~dv,held_source(~X,~xv,~D,~dv,lending),output,position) : F32},
    {Groups.numeric_weight(lent_source(~X,~xv,~D,~dv,lending),tap,position) == held_weight(~X,~xv,~D,~dv,held_source(~X,~xv,~D,~dv,lending),tap,position) : F32}>

def lookup_agrees(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lookup: Lookup<X,xv,D,dv>,+position: Nat,
  +inside: {Nat.is_lt(position,lookup_size(~X,~xv,~D,~dv,lookup)) == True{} : Bool}) -> LookupAgrees(~X,~xv,~D,~dv,lookup,position):
  Lookup{+lending,+tap,+output,+tap_inside,+output_inside} = lookup
  KernelGemm.Both{input_agrees(~X,~xv,~D,~dv,lending,output,position,output_inside,inside),weight_agrees(~X,~xv,~D,~dv,lending,tap,position,tap_inside)}

# A group product over the bands of an image is the product over the cells
# the holdings hold.
def lent_dot(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+lending: Lending<X,xv,D,dv>,+tap: U32,+output: U32,
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Groups.tap_count(lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +output_inside: {Nat.is_lt(U32.to_nat(output),Groups.output_count(lent_geometry(~X,~xv,~D,~dv,lending))) == True{} : Bool},
  +count: Nat,+first: Nat,+initial: F32,
  +room: {Nat.is_le(Nat.add(first,count),Nat.mul(Groups.row_count(lent_geometry(~X,~xv,~D,~dv,lending)),Groups.column_count(lent_geometry(~X,~xv,~D,~dv,lending)))) == True{} : Bool}) ->
  {Groups.numeric_dot(lent_source(~X,~xv,~D,~dv,lending),tap,output,count,first,initial) == held_dot(~X,~xv,~D,~dv,held_source(~X,~xv,~D,~dv,lending),tap,output,count,first,initial) : F32}:
  Ordered.dot_congruence(~Nat,~U32,~U32,~Groups.NumericSource,~U32,~U32,~HeldSource<X,D>,~Groups.numeric_input,~Groups.numeric_weight,
    ~held_input(~X,~xv,~D,~dv),~held_weight(~X,~xv,~D,~dv),Ranges.range(count,first),tap,output,lent_source(~X,~xv,~D,~dv,lending),tap,output,
    held_source(~X,~xv,~D,~dv,lending),initial,
    Members.range_all(~Lookup<X,xv,D,dv>,~LookupAgrees(~X,~xv,~D,~dv),~lookup_size(~X,~xv,~D,~dv),~lookup_agrees(~X,~xv,~D,~dv),count,first,
      {Lookup{lending,tap,output,tap_inside,output_inside} : Lookup<X,xv,D,dv>},room))
