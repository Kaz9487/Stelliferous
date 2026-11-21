# The values of the channels of cells read from a base on, placed from zero.
import Base
import ../kernel_spatial_packing.bend as Packing
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../storage_buffer.bend as Storage
import ../tensor_grid.bend as Grid
import ../traversal_blocks.bend as Blocks
import ./storage_tree_model.bend as Tree
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./boolean_logic.bend as Boolean
import ./binary_radix.bend as Radix
import ./machine_limit_bounds.bend as Limits
import ./tensor_tiled_values.bend as TiledValues
import ./traversal_indexed_values.bend as IndexedValues
import ./word_multiplication.bend as WordMultiplication
import ./kernel_spatial_packing_refinement.bend as Model
import ./u32_division.bend as WordDivision
import ./encoded_coordinates.bend as Encoded
import ./kernel_tile_refinement.bend as TileLaws
import ./word_arithmetic.bend as WordArithmetic
import ./kernel_channel_lanes.bend as Agree

# Where the cells of a plane start, the plane and the groups of a position.
type At is Data:
  At{base: U32,plane: U32,groups: U32}

# Value c * plane + p: the lane of channel c in the cell of its group at pixel
# p, the cells read from base on.
def at_sample(cells: Tree.Tree<Tile.Tile8>,context: At,index: U32) -> F32:
  At{+base,+plane,+groups} = context
  +position = index
  +channel = (position / plane : U32)
  Agree.lane_at(Agree.bits(channel),Tree.read(Tile.Tile8,cells,((base + (channel / 8 : U32) : U32) + (position % plane : U32) * groups : U32)))

def values_at(+base: U32,+channels: U32,+h: U32,+w: U32,cells: Tree.Tree<Tile.Tile8>) -> Tree.Tree<F32>:
  IndexedValues.run(~Tree.Tree<Tile.Tile8>,~F32,~At,~at_sample,~Blocks.in_place(~At),U32.to_nat((channels * (h * w) : U32)),0,At{base,(h * w : U32),Grid.groups(channels)},cells,
    Tree.fresh(F32,Storage.depth_for((channels * (h * w) : U32)),0.0))

# The lane of a channel in the cell of its group at a pixel, from the base on,
# is the value of the channel at the pixel.
def values_at_agree(+base: U32,+channels: U32,+h: U32,+w: U32,+g: U32,+fits: Agree.Fits<channels,h,w,g>,+x: Tree.Tree<Tile.Tile8>,+c: U32,+iy: U32,+ix: U32,
  bound: {U32.is_lt(c,channels) == True{} : Bool},inside: {(U32.is_lt(iy,h) && U32.is_lt(ix,w)) == True{} : Bool}) ->
  {Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + (iy * w + ix) * g : U32))) ==
    Tree.read(F32,values_at(base,channels,h,w,x),((0 + (c * (h * w) : U32) : U32) + (iy * w + ix) * 1 : U32)) : F32}:
  Agree.Fits{+limits,+grouped,+plane_exact,+total_exact,+bounded} = fits
  +in_channels = bound
  +in_image = inside
  +plane = (h * w : U32)
  +pixel = (iy * w + ix : U32)
  +total = (channels * plane : U32)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +fitting = Limits.capacity_fits(limits)
  +n = U32.to_nat(c)
  +p = U32.to_nat(plane)
  +o = U32.to_nat(pixel)
  +channel_inside = Equal.trans(Bool,Nat.is_lt(n,U32.to_nat(channels)),U32.is_lt(c,channels),True{},Equal.sym(Bool,U32.is_lt(c,channels),Nat.is_lt(n,U32.to_nat(channels)),Comparison.lt(c,channels)),in_channels)
  +rows = Equal.trans(Bool,Nat.is_lt(U32.to_nat(iy),U32.to_nat(h)),U32.is_lt(iy,h),True{},Equal.sym(Bool,U32.is_lt(iy,h),Nat.is_lt(U32.to_nat(iy),U32.to_nat(h)),Comparison.lt(iy,h)),
    Boolean.left(U32.is_lt(iy,h),U32.is_lt(ix,w),in_image))
  +columns = Equal.trans(Bool,Nat.is_lt(U32.to_nat(ix),U32.to_nat(w)),U32.is_lt(ix,w),True{},Equal.sym(Bool,U32.is_lt(ix,w),Nat.is_lt(U32.to_nat(ix),U32.to_nat(w)),Comparison.lt(ix,w)),
    Boolean.right(U32.is_lt(iy,h),U32.is_lt(ix,w),in_image))
  +nat_pixel = Nat.add(Nat.mul(U32.to_nat(iy),U32.to_nat(w)),U32.to_nat(ix))
  +pixel_plane = {%plane_exact : {Nat.is_lt(nat_pixel,_) == True{} : Bool} RowMajor.radix(U32.to_nat(iy),U32.to_nat(h),U32.to_nat(ix),U32.to_nat(w),rows,columns) : {Nat.is_lt(nat_pixel,p) == True{} : Bool}}
  +channels_positive = Order.le_lt_trans(0n,n,U32.to_nat(channels),Order.zero_le(n),channel_inside)
  +within_total = {
    %total_exact : {Nat.is_le(p,_) == True{} : Bool}
    %RowMajor.multiply_commutative(p,U32.to_nat(channels)) : {Nat.is_le(p,_) == True{} : Bool}
    RowMajor.product_covers(p,U32.to_nat(channels),channels_positive) : {Nat.is_le(p,U32.to_nat(total)) == True{} : Bool}}
  +total_room = Bounds.as_nat_le(total,Machine.capacity(limits),bounded)
  +plane_room = Order.transitive(p,U32.to_nat(total),capacity,within_total,total_room)
  +pixel_exact = Arithmetic.multiply_add_exact(iy,w,ix,Machine.capacity(limits),fitting,
    Bounds.strict_le(nat_pixel,capacity,Bounds.nat_trans(nat_pixel,p,capacity,pixel_plane,plane_room)))
  +offset_inside = {%Equal.sym(Nat,o,nat_pixel,pixel_exact) : {Nat.is_lt(_,p) == True{} : Bool} pixel_plane : {Nat.is_lt(o,p) == True{} : Bool}}
  +plane_positive = Order.le_lt_trans(0n,o,p,Order.zero_le(o),offset_inside)
  +mp = Nat.mul(n,p)
  +position = Nat.add(mp,o)
  +selected = TiledValues.inside_rows(n,U32.to_nat(channels),o,p,U32.to_nat(total),total_exact,channel_inside,offset_inside)
  +position_room = Bounds.strict_le(position,capacity,Bounds.nat_trans(position,U32.to_nat(total),capacity,selected,total_room))
  +value_word = U32.from_nat(position)
  +index_exact = Bounds.roundtrip(position,Machine.capacity(limits),fitting,position_room)
  +double = Limits.double_bound(limits,p,plane_room)
  +channel_exact = Radix.quotient_word(value_word,plane,n,o,p,Machine.half_word(limits),Limits.half_word_fits(limits),index_exact,{==},plane_positive,offset_inside,double)
  +pixel_found = Radix.remainder_word(value_word,plane,n,o,p,Machine.half_word(limits),Limits.half_word_fits(limits),index_exact,{==},plane_positive,offset_inside,double)
  +channel_word = Arithmetic.to_nat_injective((value_word / plane : U32),c,channel_exact)
  +pixel_word = Arithmetic.to_nat_injective((value_word % plane : U32),pixel,pixel_found)
  +mp_room = Order.transitive(mp,U32.to_nat(total),capacity,Order.transitive(mp,position,U32.to_nat(total),Order.le_add(mp,o),Bounds.strict_le(position,U32.to_nat(total),selected)),total_room)
  +scaled = WordMultiplication.bounded_multiply(c,plane,Machine.capacity(limits),fitting,mp_room)
  +front = Arithmetic.add(0,(c * plane : U32),Machine.capacity(limits),fitting,
    {%Equal.sym(Nat,U32.to_nat((c * plane : U32)),mp,scaled) : {Nat.is_le(_,capacity) == True{} : Bool} mp_room : {Nat.is_le(U32.to_nat((c * plane : U32)),capacity) == True{} : Bool}})
  +front_value = Equal.trans(Nat,U32.to_nat((0 + c * plane : U32)),U32.to_nat((c * plane : U32)),mp,front,scaled)
  +unit = Equal.cong(U32,Nat,word => U32.to_nat(word),(pixel * 1 : U32),pixel,Arithmetic.multiply_one(pixel))
  +whole = Equal.trans(Nat,Nat.add(U32.to_nat((0 + c * plane : U32)),U32.to_nat((pixel * 1 : U32))),Nat.add(mp,U32.to_nat((pixel * 1 : U32))),position,
    Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat((pixel * 1 : U32))),U32.to_nat((0 + c * plane : U32)),mp,front_value),
    Equal.cong(Nat,Nat,value => Nat.add(mp,value),U32.to_nat((pixel * 1 : U32)),o,unit))
  +address_sum = Arithmetic.add((0 + c * plane : U32),(pixel * 1 : U32),Machine.capacity(limits),fitting,
    {%Equal.sym(Nat,Nat.add(U32.to_nat((0 + c * plane : U32)),U32.to_nat((pixel * 1 : U32))),position,whole) : {Nat.is_le(_,capacity) == True{} : Bool} position_room :
      {Nat.is_le(Nat.add(U32.to_nat((0 + c * plane : U32)),U32.to_nat((pixel * 1 : U32))),capacity) == True{} : Bool}})
  +value_address_word = ((0 + c * plane : U32) + pixel * 1 : U32)
  +address_word = Equal.sym(U32,value_word,value_address_word,
    Arithmetic.from_nat_exact(value_address_word,position,Equal.trans(Nat,U32.to_nat(value_address_word),Nat.add(U32.to_nat((0 + c * plane : U32)),U32.to_nat((pixel * 1 : U32))),position,address_sum,whole)))
  +context = {At{base,plane,Grid.groups(channels)} : At}
  +values = values_at(base,channels,h,w,x)
  +stored = TiledValues.fresh_stored(~Tree.Tree<Tile.Tile8>,~F32,~At,~at_sample,total,0.0,context,x,position,TiledValues.allocatable(limits,total,bounded),selected)
  +sampled = {%Equal.sym(U32,(value_word / plane : U32),c,channel_word) :
      {Agree.lane_at(Agree.bits(_),Tree.read(Tile.Tile8,x,((base + (_ / 8 : U32) : U32) + (value_word % plane : U32) * Grid.groups(channels) : U32))) == Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + pixel * g : U32))) : F32}
    {%Equal.sym(U32,(value_word % plane : U32),pixel,pixel_word) :
        {Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + _ * Grid.groups(channels) : U32))) == Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + pixel * g : U32))) : F32}
      {%grouped :
          {Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + pixel * _ : U32))) == Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + pixel * g : U32))) : F32}
        {==} :
        {Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + pixel * Grid.groups(channels) : U32))) == Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + pixel * g : U32))) : F32}} :
      {Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + (value_word % plane : U32) * Grid.groups(channels) : U32))) == Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + pixel * g : U32))) : F32}} :
    {at_sample(x,context,value_word) == Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + pixel * g : U32))) : F32}}
  Equal.sym(F32,Tree.read(F32,values,value_address_word),Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + pixel * g : U32))),
    Equal.trans(F32,Tree.read(F32,values,value_address_word),Tree.read(F32,values,value_word),Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + pixel * g : U32))),
      Equal.cong(U32,F32,address => Tree.read(F32,values,address),value_address_word,value_word,address_word),
      Equal.trans(F32,Tree.read(F32,values,value_word),at_sample(x,context,value_word),Agree.lane_at(Agree.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + pixel * g : U32))),stored,sampled)))

# The cell of a channel at a pixel word, as one natural number.
def cell_word(+channels: U32,+c: U32,+pixel: U32) ->
  {((0 + c / 8 : U32) + pixel * Grid.groups(channels) : U32) ==
    U32.from_nat(Nat.add(Nat.mul(U32.to_nat(pixel),U32.to_nat(Grid.groups(channels))),Nat.div(U32.to_nat(c),8n))) : U32}:
  +groups = Grid.groups(channels)
  +group = Nat.div(U32.to_nat(c),8n)
  +scaled = (U32.from_nat(U32.to_nat(pixel)) * U32.from_nat(U32.to_nat(groups)) : U32)
  +goal = U32.from_nat(Nat.add(Nat.mul(U32.to_nat(pixel),U32.to_nat(groups)),group))
  %Equal.sym(U32,(0 + (c / 8 : U32) : U32),(c / 8 : U32),Arithmetic.add_zero_left((c / 8 : U32))) : {(_ + pixel * groups : U32) == goal : U32}
  %Arithmetic.from_nat_exact((c / 8 : U32),group,WordDivision.divide(c,8,16,{==},{==},{==})) : {(_ + pixel * groups : U32) == goal : U32}
  %Arithmetic.native_roundtrip(pixel) : {(U32.from_nat(group) + _ * groups : U32) == goal : U32}
  %Arithmetic.native_roundtrip(groups) : {(U32.from_nat(group) + U32.from_nat(U32.to_nat(pixel)) * _ : U32) == goal : U32}
  Equal.trans(U32,(U32.from_nat(group) + scaled : U32),(scaled + U32.from_nat(group) : U32),goal,Arithmetic.add_commutative(U32.from_nat(group),scaled),
    Encoded.encoded_madd(U32.to_nat(pixel),U32.to_nat(groups),group))

# Value channel * plane + offset of cells read from a base is the lane of the
# channel in the cell of its group at the offset, counted from the base.
def values_read(+base: U32,+channels: U32,+h: U32,+w: U32,+fits: Agree.Fits<channels,h,w,Grid.groups(channels)>,+x: Tree.Tree<Tile.Tile8>,+channel: Nat,+offset: Nat,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+offset_inside: {Nat.is_lt(offset,U32.to_nat((h * w : U32))) == True{} : Bool}) ->
  {Tree.read(F32,values_at(base,channels,h,w,x),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat((h * w : U32))),offset))) ==
    TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,x,(base + U32.from_nat(Nat.add(Nat.mul(offset,U32.to_nat(Grid.groups(channels))),Nat.div(channel,8n))) : U32))) : F32}:
  Agree.Fits{+limits,+grouped,+plane_exact,+total_exact,+bounded} = fits
  +plane = (h * w : U32)
  +groups = Grid.groups(channels)
  +total = (channels * plane : U32)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +fitting = Limits.capacity_fits(limits)
  +p = U32.to_nat(plane)
  +position = Nat.add(Nat.mul(channel,p),offset)
  +selected = TiledValues.inside_rows(channel,U32.to_nat(channels),offset,p,U32.to_nat(total),total_exact,channel_inside,offset_inside)
  +total_room = Bounds.as_nat_le(total,Machine.capacity(limits),bounded)
  +position_room = Bounds.strict_le(position,capacity,Bounds.nat_trans(position,U32.to_nat(total),capacity,selected,total_room))
  +word = U32.from_nat(position)
  +word_exact = Bounds.roundtrip(position,Machine.capacity(limits),fitting,position_room)
  +plane_positive = Order.le_lt_trans(0n,offset,p,Order.zero_le(offset),offset_inside)
  +channels_positive = Order.le_lt_trans(0n,channel,U32.to_nat(channels),Order.zero_le(channel),channel_inside)
  +within_total = {
    %total_exact : {Nat.is_le(p,_) == True{} : Bool}
    %RowMajor.multiply_commutative(p,U32.to_nat(channels)) : {Nat.is_le(p,_) == True{} : Bool}
    RowMajor.product_covers(p,U32.to_nat(channels),channels_positive) : {Nat.is_le(p,U32.to_nat(total)) == True{} : Bool}}
  +plane_room = Order.transitive(p,U32.to_nat(total),capacity,within_total,total_room)
  +double = Limits.double_bound(limits,p,plane_room)
  +channel_exact = Radix.quotient_word(word,plane,channel,offset,p,Machine.half_word(limits),Limits.half_word_fits(limits),word_exact,{==},plane_positive,offset_inside,double)
  +pixel_found = Radix.remainder_word(word,plane,channel,offset,p,Machine.half_word(limits),Limits.half_word_fits(limits),word_exact,{==},plane_positive,offset_inside,double)
  +cw = (word / plane : U32)
  +pw = (word % plane : U32)
  +cell = U32.from_nat(Nat.add(Nat.mul(U32.to_nat(pw),U32.to_nat(groups)),Nat.div(U32.to_nat(cw),8n)))
  +address = Equal.trans(U32,((base + cw / 8 : U32) + pw * groups : U32),(base + ((cw / 8 : U32) + pw * groups : U32) : U32),(base + cell : U32),
    Arithmetic.add_associative(base,(cw / 8 : U32),(pw * groups : U32)),
    Equal.cong(U32,U32,value => (base + value : U32),((cw / 8 : U32) + pw * groups : U32),cell,
      Equal.trans(U32,((cw / 8 : U32) + pw * groups : U32),((0 + cw / 8 : U32) + pw * groups : U32),cell,
        Equal.cong(U32,U32,value => (value + pw * groups : U32),(cw / 8 : U32),(0 + cw / 8 : U32),Equal.sym(U32,(0 + cw / 8 : U32),(cw / 8 : U32),Arithmetic.add_zero_left((cw / 8 : U32)))),
        cell_word(channels,cw,pw))))
  +tile = Tree.read(Tile.Tile8,x,((base + cw / 8 : U32) + pw * groups : U32))
  +wanted = TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,x,(base + U32.from_nat(Nat.add(Nat.mul(offset,U32.to_nat(groups)),Nat.div(channel,8n))) : U32)))
  +context = {At{base,plane,groups} : At}
  +stored = TiledValues.fresh_stored(~Tree.Tree<Tile.Tile8>,~F32,~At,~at_sample,total,0.0,context,x,position,TiledValues.allocatable(limits,total,bounded),selected)
  Equal.trans(F32,Tree.read(F32,values_at(base,channels,h,w,x),word),at_sample(x,context,word),wanted,stored,
    Equal.trans(F32,Agree.lane_at(Agree.bits(cw),tile),TileLaws.select_lane(Nat.mod(U32.to_nat(cw),8n),tile),wanted,Agree.lane_bits(cw,tile),
      %Equal.sym(U32,((base + cw / 8 : U32) + pw * groups : U32),(base + cell : U32),address) :
        {TileLaws.select_lane(Nat.mod(U32.to_nat(cw),8n),Tree.read(Tile.Tile8,x,_)) == wanted : F32}
      %Equal.sym(Nat,U32.to_nat(cw),channel,channel_exact) :
        {TileLaws.select_lane(Nat.mod(_,8n),Tree.read(Tile.Tile8,x,(base + U32.from_nat(Nat.add(Nat.mul(U32.to_nat(pw),U32.to_nat(groups)),Nat.div(_,8n))) : U32))) == wanted : F32}
      %Equal.sym(Nat,U32.to_nat(pw),offset,pixel_found) :
        {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,x,(base + U32.from_nat(Nat.add(Nat.mul(_,U32.to_nat(groups)),Nat.div(channel,8n))) : U32))) == wanted : F32}
      {==}))

# The same lane at cell base + offset * groups + channel / 8 as one natural
# number.
def values_cell(+base: U32,+channels: U32,+h: U32,+w: U32,+fits: Agree.Fits<channels,h,w,Grid.groups(channels)>,+x: Tree.Tree<Tile.Tile8>,+channel: Nat,+offset: Nat,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+offset_inside: {Nat.is_lt(offset,U32.to_nat((h * w : U32))) == True{} : Bool}) ->
  {Tree.read(F32,values_at(base,channels,h,w,x),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat((h * w : U32))),offset))) ==
    TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,x,U32.from_nat(Nat.add(U32.to_nat(base),Nat.add(Nat.mul(offset,U32.to_nat(Grid.groups(channels))),Nat.div(channel,8n)))))) : F32}:
  +cell = Nat.add(Nat.mul(offset,U32.to_nat(Grid.groups(channels))),Nat.div(channel,8n))
  +word = Equal.trans(U32,(base + U32.from_nat(cell) : U32),(U32.from_nat(U32.to_nat(base)) + U32.from_nat(cell) : U32),U32.from_nat(Nat.add(U32.to_nat(base),cell)),
    Equal.cong(U32,U32,value => (value + U32.from_nat(cell) : U32),base,U32.from_nat(U32.to_nat(base)),Equal.sym(U32,U32.from_nat(U32.to_nat(base)),base,Arithmetic.native_roundtrip(base))),
    WordArithmetic.native_add(U32.to_nat(base),cell))
  %word :
    {Tree.read(F32,values_at(base,channels,h,w,x),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat((h * w : U32))),offset))) == TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,x,_)) : F32}
  values_read(base,channels,h,w,fits,x,channel,offset,channel_inside,offset_inside)
