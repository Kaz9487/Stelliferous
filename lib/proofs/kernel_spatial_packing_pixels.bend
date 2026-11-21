# Each lane of a packed tile is the input pixel of its position, channel and
# taps: the pass that owns the lane reads it from the planes the transpose
# wrote.
import Base
import ../kernel_spatial_packing.bend as Packing
import ../kernel_packing_samples.bend as PackingSamples
import ../kernel_tile.bend as Tile
import ../math_fp32.bend as FP32
import ./math_fp32_bits.bend as FP32Bits
import ./kernel_tile_refinement.bend as TileLaws
import ./kernel_channel_lanes.bend as Lanes
import ./kernel_spatial_packing_refinement.bend as Model
import ./kernel_spatial_packing_planes.bend as Planes
import ./kernel_spatial_packing_passes.bend as Passes
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./word_subtraction.bend as Subtraction
import ./u32_division.bend as WordDivision
import ./row_major_index.bend as RowMajor
import ./binary_radix.bend as Radix
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./word_encoding.bend as WordEncoding
import ./boolean_logic.bend as Boolean
import ./kernel_spatial_packing_samples.bend as SpatialSamples
import ./kernel_packing_values_at.bend as ValuesAt
import ./storage_tree_equality.bend as TreeEquality
import ./storage_address_bounds.bend as Address

# ---- Lanes of tiles ----

# No lane lies eight or more past lane zero.
def eight_below(+rest: Nat,below: {Nat.is_lt(Nat.add(8n,rest),8n) == True{} : Bool}) -> Empty:
  WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),below))

# Lane index of a pair: the lanes of its first tile, then of its second.
def pair_lane(+index: Nat,+pair: Tile.Tile16) -> F32:
  Tile.Tiles{+low,+high} = pair
  Bool.pick(F32,Nat.is_lt(index,8n),TileLaws.select_lane(index,low),TileLaws.select_lane(Nat.sub(index,8n),high))

# The lanes of a pair from a lane on.
def pair_from(+shift: Nat,+pair: Tile.Tile16) -> Tile.Tile8:
  Tile.Tile{pair_lane(Nat.add(shift,0n),pair),pair_lane(Nat.add(shift,1n),pair),pair_lane(Nat.add(shift,2n),pair),pair_lane(Nat.add(shift,3n),pair),
    pair_lane(Nat.add(shift,4n),pair),pair_lane(Nat.add(shift,5n),pair),pair_lane(Nat.add(shift,6n),pair),pair_lane(Nat.add(shift,7n),pair)}

# The shift the packer names by the bits of a word below eight takes the
# pair's lanes from that lane on.
def ShiftForm(+shift: Nat,+pair: Tile.Tile16) -> Type:
  +word = U32.from_nat(shift)
  {Passes.shifted_at(U32.is_lt(word,4),U32.is_lt((word % 4 : U32),2),U32.is_eq((word % 2 : U32),1),pair) == pair_from(shift,pair) : Tile.Tile8}

def shift_form(+pair: Tile.Tile16,+shift: Nat,below: {Nat.is_lt(shift,8n) == True{} : Bool}) -> ShiftForm(shift,pair):
  Tile.Tiles{+low,+high} = pair
  Tile.Tile{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} = low
  Tile.Tile{+b0,+b1,+b2,+b3,+b4,+b5,+b6,+b7} = high
  match shift:
    case 0n: {==}
    case 1n+ +s1:
      match s1:
        case 0n: {==}
        case 1n+ +s2:
          match s2:
            case 0n: {==}
            case 1n+ +s3:
              match s3:
                case 0n: {==}
                case 1n+ +s4:
                  match s4:
                    case 0n: {==}
                    case 1n+ +s5:
                      match s5:
                        case 0n: {==}
                        case 1n+ +s6:
                          match s6:
                            case 0n: {==}
                            case 1n+ +s7:
                              match s7:
                                case 0n: {==}
                                case 1n+ +rest:
                                  Empty.absurd(ShiftForm(Nat.add(8n,rest),Tile.Tiles{Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7},Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7}}),
                                    eight_below(rest,below))

def SpreadLane(+shift: Nat,+pair: Tile.Tile16,+lane: Nat) -> Type:
  {TileLaws.select_lane(lane,pair_from(shift,pair)) == pair_lane(Nat.add(shift,lane),pair) : F32}

def spread_lane(+shift: Nat,+pair: Tile.Tile16,+lane: Nat,below: {Nat.is_lt(lane,8n) == True{} : Bool}) -> SpreadLane(shift,pair,lane):
  match lane:
    case 0n: {==}
    case 1n+ +l1:
      match l1:
        case 0n: {==}
        case 1n+ +l2:
          match l2:
            case 0n: {==}
            case 1n+ +l3:
              match l3:
                case 0n: {==}
                case 1n+ +l4:
                  match l4:
                    case 0n: {==}
                    case 1n+ +l5:
                      match l5:
                        case 0n: {==}
                        case 1n+ +l6:
                          match l6:
                            case 0n: {==}
                            case 1n+ +l7:
                              match l7:
                                case 0n: {==}
                                case 1n+ +rest: Empty.absurd(SpreadLane(shift,pair,Nat.add(8n,rest)),eight_below(rest,below))

# Lane lane of a shifted pair is lane shift + lane of the pair.
def shifted_lane(+pair: Tile.Tile16,+shift: Nat,+lane: Nat,+shift_below: {Nat.is_lt(shift,8n) == True{} : Bool},+lane_below: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Passes.shifted_at(U32.is_lt(U32.from_nat(shift),4),U32.is_lt((U32.from_nat(shift) % 4 : U32),2),U32.is_eq((U32.from_nat(shift) % 2 : U32),1),pair)) ==
    pair_lane(Nat.add(shift,lane),pair) : F32}:
  %Equal.sym(Tile.Tile8,Passes.shifted_at(U32.is_lt(U32.from_nat(shift),4),U32.is_lt((U32.from_nat(shift) % 4 : U32),2),U32.is_eq((U32.from_nat(shift) % 2 : U32),1),pair),
    pair_from(shift,pair),shift_form(pair,shift,shift_below)) : {TileLaws.select_lane(lane,_) == pair_lane(Nat.add(shift,lane),pair) : F32}
  spread_lane(shift,pair,lane,lane_below)

# A word select by a full or empty mask.
def selected_pick(+inside: Bool,+yes: F32,+no: F32) -> {FP32.selected(Bool.pick(U32,inside,4294967295,0),yes,no) == Bool.pick(F32,inside,yes,no) : F32}:
  match inside:
    case True{}: FP32Bits.selected_all(yes,no)
    case False{}: FP32Bits.selected_none(yes,no)

# A merged tile takes the pass's lanes from its lane on, the kept lanes below.
def MergedLane(+lane: U32,+kept: Tile.Tile8,+tile: Tile.Tile8,+index: Nat) -> Type:
  {TileLaws.select_lane(index,Packing.merged(lane,kept,tile)) ==
    Bool.pick(F32,U32.is_le(lane,U32.from_nat(index)),TileLaws.select_lane(index,tile),TileLaws.select_lane(index,kept)) : F32}

def merged_lane(+lane: U32,+kept: Tile.Tile8,+tile: Tile.Tile8,+index: Nat,below: {Nat.is_lt(index,8n) == True{} : Bool}) -> MergedLane(lane,kept,tile,index):
  Tile.Tile{+x0,+x1,+x2,+x3,+x4,+x5,+x6,+x7} = kept
  Tile.Tile{+y0,+y1,+y2,+y3,+y4,+y5,+y6,+y7} = tile
  match index:
    case 0n: selected_pick(U32.is_le(lane,0),y0,x0)
    case 1n+ +l1:
      match l1:
        case 0n: selected_pick(U32.is_le(lane,1),y1,x1)
        case 1n+ +l2:
          match l2:
            case 0n: selected_pick(U32.is_le(lane,2),y2,x2)
            case 1n+ +l3:
              match l3:
                case 0n: selected_pick(U32.is_le(lane,3),y3,x3)
                case 1n+ +l4:
                  match l4:
                    case 0n: selected_pick(U32.is_le(lane,4),y4,x4)
                    case 1n+ +l5:
                      match l5:
                        case 0n: selected_pick(U32.is_le(lane,5),y5,x5)
                        case 1n+ +l6:
                          match l6:
                            case 0n: selected_pick(U32.is_le(lane,6),y6,x6)
                            case 1n+ +l7:
                              match l7:
                                case 0n: selected_pick(U32.is_le(lane,7),y7,x7)
                                case 1n+ +rest:
                                  Empty.absurd(MergedLane(lane,Tile.Tile{x0,x1,x2,x3,x4,x5,x6,x7},Tile.Tile{y0,y1,y2,y3,y4,y5,y6,y7},Nat.add(8n,rest)),eight_below(rest,below))

# The lanes of a tile of samples are the samples.
def CellsLane(+input: Tree.Tree<Tile.Tile8>,+lanes: Lanes.Bits,+block: Packing.PixelBlock,+index: Nat) -> Type:
  {TileLaws.select_lane(index,Planes.cells_tile(input,lanes,block)) == Planes.cell_value(input,lanes,block,index) : F32}

def cells_lane(+input: Tree.Tree<Tile.Tile8>,+lanes: Lanes.Bits,+block: Packing.PixelBlock,+index: Nat,below: {Nat.is_lt(index,8n) == True{} : Bool}) ->
  CellsLane(input,lanes,block,index):
  match index:
    case 0n: {==}
    case 1n+ +l1:
      match l1:
        case 0n: {==}
        case 1n+ +l2:
          match l2:
            case 0n: {==}
            case 1n+ +l3:
              match l3:
                case 0n: {==}
                case 1n+ +l4:
                  match l4:
                    case 0n: {==}
                    case 1n+ +l5:
                      match l5:
                        case 0n: {==}
                        case 1n+ +l6:
                          match l6:
                            case 0n: {==}
                            case 1n+ +l7:
                              match l7:
                                case 0n: {==}
                                case 1n+ +rest: Empty.absurd(CellsLane(input,lanes,block,Nat.add(8n,rest)),eight_below(rest,below))

# ---- Plane samples ----

# Value value of plane row row: the lane of the cell at the pixel the plane
# reads there, +0 outside the input.
def plane_sample(+context: Planes.Row,+lanes: Lanes.Bits,+row: Nat,+value: Nat) -> F32:
  Planes.Row{+input,+first_row,+first_column,row_tiles,+channel_base,+height,+width,+stride,+groups,channels} = context
  +iy = (first_row + U32.from_nat(row) : U32)
  +ix = (first_column + U32.from_nat(value) * stride : U32)
  PackingSamples.masked(Lanes.lane_at(lanes,Tree.read(Tile.Tile8,input,(channel_base + (iy * width + ix) * groups : U32))),U32.is_lt(iy,height) && U32.is_lt(ix,width))

# Lane index of tile column column lies at value column * 8 + index.
def column_word(+first: U32,+column: Nat,+index: Nat,+stride: U32) ->
  {((first + U32.from_nat(column) * (8 * stride : U32) : U32) + U32.from_nat(index) * stride : U32) ==
    (first + U32.from_nat(Nat.add(Nat.mul(column,8n),index)) * stride : U32) : U32}:
  +tens = U32.from_nat(Nat.mul(column,8n))
  +part = U32.from_nat(index)
  +left = ((first + U32.from_nat(column) * (8 * stride : U32) : U32) + part * stride : U32)
  %WordArithmetic.native_add(Nat.mul(column,8n),index) : {left == (first + _ * stride : U32) : U32}
  %Equal.sym(U32,((tens + part : U32) * stride : U32),(tens * stride + part * stride : U32),Arithmetic.multiply_additive(tens,part,stride)) : {left == (first + _ : U32) : U32}
  %Multiplication.native_encoded_multiply(column,8n) : {left == (first + (_ * stride + part * stride : U32) : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(column) * U32.from_nat(8n) * stride : U32),(U32.from_nat(column) * (U32.from_nat(8n) * stride : U32) : U32),
    Arithmetic.multiply_associative(U32.from_nat(column),U32.from_nat(8n),stride)) : {left == (first + (_ + part * stride : U32) : U32) : U32}
  Arithmetic.add_associative(first,(U32.from_nat(column) * (8 * stride : U32) : U32),(part * stride : U32))

# Lane index of the tile at a spot is the plane value at its column * 8 +
# index.
def tile_sample(+context: Planes.Row,+lanes: Lanes.Bits,+spot: Planes.Spot,+index: Nat,+below: {Nat.is_lt(index,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Planes.lanes_tile(context,lanes,spot)) == plane_sample(context,lanes,Planes.row_of(spot),Nat.add(Nat.mul(Planes.column_of(spot),8n),index)) : F32}:
  Planes.Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
  Planes.Spot{+row,+column} = spot
  +iy = (first_row + U32.from_nat(row) : U32)
  +block = {Packing.PixelBlock{iy,(first_column + U32.from_nat(column) * (8 * stride : U32) : U32),channel_base,height,width,stride,groups} : Packing.PixelBlock}
  Equal.trans(F32,TileLaws.select_lane(index,Planes.cells_tile(input,lanes,block)),Planes.cell_value(input,lanes,block,index),
    plane_sample(Planes.Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels},lanes,row,Nat.add(Nat.mul(column,8n),index)),
    cells_lane(input,lanes,block,index,below),
    Equal.cong(U32,F32,ix => PackingSamples.masked(Lanes.lane_at(lanes,Tree.read(Tile.Tile8,input,(channel_base + (iy * width + ix) * groups : U32))),U32.is_lt(iy,height) && U32.is_lt(ix,width)),
      ((first_column + U32.from_nat(column) * (8 * stride : U32) : U32) + U32.from_nat(index) * stride : U32),
      (first_column + U32.from_nat(Nat.add(Nat.mul(column,8n),index)) * stride : U32),column_word(first_column,column,index,stride)))

# Element * 8 + index splits into the plane row and the value in it.
def element_split(+row: Nat,+column: Nat,+tiles: Nat,+index: Nat) ->
  {Nat.add(Nat.mul(Planes.outer(row,column,tiles),8n),index) == Planes.outer(row,Nat.add(Nat.mul(column,8n),index),Nat.mul(tiles,8n)) : Nat}:
  +head = Nat.mul(Nat.mul(row,tiles),8n)
  %Equal.sym(Nat,Nat.mul(Planes.outer(row,column,tiles),8n),Nat.add(head,Nat.mul(column,8n)),RowMajor.distrib(Nat.mul(row,tiles),column,8n)) : {Nat.add(_,index) == Planes.outer(row,Nat.add(Nat.mul(column,8n),index),Nat.mul(tiles,8n)) : Nat}
  %RowMajor.assoc(row,tiles,8n) : {Nat.add(Nat.add(head,Nat.mul(column,8n)),index) == Nat.add(_,Nat.add(Nat.mul(column,8n),index)) : Nat}
  Addition.associative(head,Nat.mul(column,8n),index)

# The tile at a spot holds the plane values its linear place names.
def tile_linear(+context: Planes.Row,+lanes: Lanes.Bits,+spot: Planes.Spot,+index: Nat,+row: Nat,+value: Nat,+below: {Nat.is_lt(index,8n) == True{} : Bool},
  +column_inside: {Nat.is_lt(Planes.column_of(spot),Planes.row_tiles(context)) == True{} : Bool},
  +value_inside: {Nat.is_lt(value,Nat.mul(Planes.row_tiles(context),8n)) == True{} : Bool},
  +linear: {Nat.add(Nat.mul(Planes.element(spot,Planes.row_tiles(context)),8n),index) == Planes.outer(row,value,Nat.mul(Planes.row_tiles(context),8n)) : Nat}) ->
  {TileLaws.select_lane(index,Planes.lanes_tile(context,lanes,spot)) == plane_sample(context,lanes,row,value) : F32}:
  Planes.Spot{+spot_row,+column} = spot
  +tiles = Planes.row_tiles(context)
  +width = Nat.mul(tiles,8n)
  +own = Nat.add(Nat.mul(column,8n),index)
  +own_inside = RowMajor.radix(column,tiles,index,8n,column_inside,below)
  +positive = Order.le_lt_trans(0n,own,width,Order.zero_le(own),own_inside)
  +same = Equal.trans(Nat,Planes.outer(spot_row,own,width),Nat.add(Nat.mul(Planes.outer(spot_row,column,tiles),8n),index),Planes.outer(row,value,width),
    Equal.sym(Nat,Nat.add(Nat.mul(Planes.outer(spot_row,column,tiles),8n),index),Planes.outer(spot_row,own,width),element_split(spot_row,column,tiles,index)),linear)
  %Radix.row_unique(spot_row,row,width,own,value,positive,own_inside,value_inside,same) : {TileLaws.select_lane(index,Planes.lanes_tile(context,lanes,Planes.Spot{spot_row,column})) == plane_sample(context,lanes,_,value) : F32}
  %Radix.col_unique(spot_row,row,width,own,value,positive,own_inside,value_inside,same) : {TileLaws.select_lane(index,Planes.lanes_tile(context,lanes,Planes.Spot{spot_row,column})) == plane_sample(context,lanes,spot_row,_) : F32}
  tile_sample(context,lanes,Planes.Spot{spot_row,column},index,below)

# Past lane 7 a pair reads the next tile of the plane.
def pair_case(+context: Planes.Row,+lanes: Lanes.Bits,+spot: Planes.Spot,+inside: Bool,+index: Nat,+row: Nat,+value: Nat,
  +exact: {Nat.is_lt(index,8n) == inside : Bool},+below: {Nat.is_lt(index,16n) == True{} : Bool},
  +column_inside: {Nat.is_lt(Planes.column_of(spot),Planes.row_tiles(context)) == True{} : Bool},
  +value_inside: {Nat.is_lt(value,Nat.mul(Planes.row_tiles(context),8n)) == True{} : Bool},
  +linear: {Nat.add(Nat.mul(Planes.element(spot,Planes.row_tiles(context)),8n),index) == Planes.outer(row,value,Nat.mul(Planes.row_tiles(context),8n)) : Nat}) ->
  {Bool.pick(F32,inside,TileLaws.select_lane(index,Planes.lanes_tile(context,lanes,spot)),
    TileLaws.select_lane(Nat.sub(index,8n),Planes.lanes_tile(context,lanes,Planes.next_spot(spot,Planes.row_tiles(context))))) == plane_sample(context,lanes,row,value) : F32}:
  Planes.Spot{+spot_row,+column} = spot
  match inside:
    case True{}: tile_linear(context,lanes,Planes.Spot{spot_row,column},index,row,value,exact,column_inside,value_inside,linear)
    case False{}:
      +tiles = Planes.row_tiles(context)
      +width = Nat.mul(tiles,8n)
      +here = {Planes.Spot{spot_row,column} : Planes.Spot}
      +next = Planes.next_spot(here,tiles)
      +positive = Order.le_lt_trans(0n,column,tiles,Order.zero_le(column),column_inside)
      +over = Order.not_less(index,8n,exact)
      +rest = Nat.sub(index,8n)
      +rest_below = Order.sub_lt(index,8n,8n,over,below)
      +scaled = Nat.mul(Planes.outer(spot_row,column,tiles),8n)
      +next_element = Planes.next_count(spot_row,column,tiles,Nat.is_lt(1n+column,tiles),{==},column_inside)
      +linear_next = {
        %Equal.sym(Nat,Planes.element(next,tiles),1n+Planes.outer(spot_row,column,tiles),next_element) : {Nat.add(Nat.mul(_,8n),rest) == Planes.outer(row,value,width) : Nat}
        %Equal.sym(Nat,Nat.add(Nat.add(8n,scaled),rest),Nat.add(8n,Nat.add(scaled,rest)),Addition.associative(8n,scaled,rest)) : {_ == Planes.outer(row,value,width) : Nat}
        %Equal.sym(Nat,Nat.add(8n,Nat.add(scaled,rest)),Nat.add(scaled,Nat.add(8n,rest)),RowMajor.add_swap(8n,scaled,rest)) : {_ == Planes.outer(row,value,width) : Nat}
        %Equal.sym(Nat,Nat.add(8n,rest),index,Order.restore_shift(8n,index,over)) : {Nat.add(scaled,_) == Planes.outer(row,value,width) : Nat}
        linear : {Nat.add(Nat.mul(Planes.element(next,tiles),8n),rest) == Planes.outer(row,value,width) : Nat}}
      tile_linear(context,lanes,next,rest,row,value,rest_below,Planes.next_column_inside(here,tiles,positive),value_inside,linear_next)

# Lane index of the pair at a spot, below sixteen, is the plane value at its
# linear place element * 8 + index.
def pair_linear(+context: Planes.Row,+lanes: Lanes.Bits,+spot: Planes.Spot,+index: Nat,+row: Nat,+value: Nat,+below: {Nat.is_lt(index,16n) == True{} : Bool},
  +column_inside: {Nat.is_lt(Planes.column_of(spot),Planes.row_tiles(context)) == True{} : Bool},
  +value_inside: {Nat.is_lt(value,Nat.mul(Planes.row_tiles(context),8n)) == True{} : Bool},
  +linear: {Nat.add(Nat.mul(Planes.element(spot,Planes.row_tiles(context)),8n),index) == Planes.outer(row,value,Nat.mul(Planes.row_tiles(context),8n)) : Nat}) ->
  {pair_lane(index,Planes.lanes_pair(context,lanes,spot)) == plane_sample(context,lanes,row,value) : F32}:
  pair_case(context,lanes,spot,Nat.is_lt(index,8n),index,row,value,{==},below,column_inside,value_inside,linear)

# ---- The pass that owns a lane ----

def output_width_of(+window: Packing.Stencil) -> Nat:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,+output_width,inner,groups} = window
  U32.to_nat(output_width)

# The tile a pass of a kernel column writes for a target entry before
# merging.
def pass_shifted(+pairs: Tree.Tree<Tile.Tile16>,+lane: U32,+base: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes,+target: Planes.Spot) ->
  Tile.Tile8:
  +shift = (base % 8 : U32)
  Passes.shifted_at(U32.is_lt(shift,4),U32.is_lt((shift % 4 : U32),2),U32.is_eq((shift % 2 : U32),1),
    Tree.read(Tile.Tile16,pairs,Passes.entry_pair(Passes.pass_at(lane,base,kernel_column,unit,window,planes),target)))

def pass_lane(+lane: U32,+base: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes) ->
  {Passes.lane_of(Passes.pass_at(lane,base,kernel_column,unit,window,planes)) == lane : U32}:
  Packing.Stencil{+base_cell,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  {==}

# After the first pass the lane steps one output row.
def later_lane(+pass: U32,+lane: Nat,+second_lane: U32,+window: Packing.Stencil,+nonzero: {U32.is_zero(pass) == False{} : Bool}) ->
  {Passes.next_lane(pass,U32.from_nat(lane),second_lane,window) == U32.from_nat(Nat.add(lane,output_width_of(window))) : U32}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  %Equal.sym(Bool,U32.is_zero(pass),False{},nonzero) : {Bool.pick(U32,_,second_lane,(U32.from_nat(lane) + output_width : U32)) == U32.from_nat(Nat.add(lane,U32.to_nat(output_width))) : U32}
  %Arithmetic.native_roundtrip(output_width) :
    {(U32.from_nat(lane) + _ : U32) == U32.from_nat(Nat.add(lane,U32.to_nat(output_width))) : U32}
  WordArithmetic.native_add(lane,U32.to_nat(output_width))

# The first pass hands over to the second lane.
def first_lane(+lane: U32,+second_lane: U32,+window: Packing.Stencil) -> {Passes.next_lane(U32.from_nat(0n),lane,second_lane,window) == second_lane : U32}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  {==}

# Below a cap a natural compares as its word does.
def word_le(+left: Nat,+right: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+left_room: {Nat.is_le(left,U32.to_nat(cap)) == True{} : Bool},
  +right_room: {Nat.is_le(right,U32.to_nat(cap)) == True{} : Bool}) -> {U32.is_le(U32.from_nat(left),U32.from_nat(right)) == Nat.is_le(left,right) : Bool}:
  %Equal.sym(Bool,U32.is_le(U32.from_nat(left),U32.from_nat(right)),Nat.is_le(U32.to_nat(U32.from_nat(left)),U32.to_nat(U32.from_nat(right))),
    Comparison.le(U32.from_nat(left),U32.from_nat(right))) : {_ == Nat.is_le(left,right) : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(left)),left,Bounds.roundtrip(left,cap,cap_ok,left_room)) : {Nat.is_le(_,U32.to_nat(U32.from_nat(right))) == Nat.is_le(left,right) : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(right)),right,Bounds.roundtrip(right,cap,cap_ok,right_room)) : {Nat.is_le(left,_) == Nat.is_le(left,right) : Bool}
  {==}

def le_false(+left: Nat,+right: Nat,+less: {Nat.is_lt(right,left) == True{} : Bool}) -> {Nat.is_le(left,right) == False{} : Bool}:
  match left right:
    case 0n +right: Empty.absurd({Nat.is_le(0n,right) == False{} : Bool},
      WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(right,0n),True{},Equal.sym(Bool,Nat.is_lt(right,0n),False{},Division.lt_zero(right)),less)))
    case 1n+ +left 0n: {==}
    case 1n+ +left 1n+ +right: le_false(left,right,less)

# A lane below the merge lane keeps its tile, a lane from it on takes the
# tile of the pass.
def merged_kept(+lane: Nat,+kept: Tile.Tile8,+tile: Tile.Tile8,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +below: {Nat.is_lt(index,8n) == True{} : Bool},+above: {Nat.is_lt(index,lane) == True{} : Bool},+lane_room: {Nat.is_le(lane,U32.to_nat(cap)) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Packing.merged(U32.from_nat(lane),kept,tile)) == TileLaws.select_lane(index,kept) : F32}:
  +index_room = Order.transitive(index,lane,U32.to_nat(cap),Bounds.strict_le(index,lane,above),lane_room)
  %Equal.sym(F32,TileLaws.select_lane(index,Packing.merged(U32.from_nat(lane),kept,tile)),Bool.pick(F32,U32.is_le(U32.from_nat(lane),U32.from_nat(index)),TileLaws.select_lane(index,tile),TileLaws.select_lane(index,kept)),merged_lane(U32.from_nat(lane),kept,tile,index,below)) : {_ == TileLaws.select_lane(index,kept) : F32}
  %Equal.sym(Bool,U32.is_le(U32.from_nat(lane),U32.from_nat(index)),Nat.is_le(lane,index),word_le(lane,index,cap,cap_ok,lane_room,index_room)) : {Bool.pick(F32,_,TileLaws.select_lane(index,tile),TileLaws.select_lane(index,kept)) == TileLaws.select_lane(index,kept) : F32}
  %Equal.sym(Bool,Nat.is_le(lane,index),False{},le_false(lane,index,above)) : {Bool.pick(F32,_,TileLaws.select_lane(index,tile),TileLaws.select_lane(index,kept)) == TileLaws.select_lane(index,kept) : F32}
  {==}

def merged_taken(+lane: Nat,+kept: Tile.Tile8,+tile: Tile.Tile8,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +below: {Nat.is_lt(index,8n) == True{} : Bool},+reached: {Nat.is_le(lane,index) == True{} : Bool},+index_room: {Nat.is_le(index,U32.to_nat(cap)) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Packing.merged(U32.from_nat(lane),kept,tile)) == TileLaws.select_lane(index,tile) : F32}:
  +lane_room = Order.transitive(lane,index,U32.to_nat(cap),reached,index_room)
  %Equal.sym(F32,TileLaws.select_lane(index,Packing.merged(U32.from_nat(lane),kept,tile)),Bool.pick(F32,U32.is_le(U32.from_nat(lane),U32.from_nat(index)),TileLaws.select_lane(index,tile),TileLaws.select_lane(index,kept)),merged_lane(U32.from_nat(lane),kept,tile,index,below)) : {_ == TileLaws.select_lane(index,tile) : F32}
  %Equal.sym(Bool,U32.is_le(U32.from_nat(lane),U32.from_nat(index)),Nat.is_le(lane,index),word_le(lane,index,cap,cap_ok,lane_room,index_room)) : {Bool.pick(F32,_,TileLaws.select_lane(index,tile),TileLaws.select_lane(index,kept)) == TileLaws.select_lane(index,tile) : F32}
  %Equal.sym(Bool,Nat.is_le(lane,index),True{},reached) : {Bool.pick(F32,_,TileLaws.select_lane(index,tile),TileLaws.select_lane(index,kept)) == TileLaws.select_lane(index,tile) : F32}
  {==}

# A later pass whose lane lies past a lane leaves it, and so do the passes
# after it.
def passes_after(n: Nat,+pairs: Tree.Tree<Tile.Tile16>,+pass: Nat,+lane: Nat,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,
  +planes: Packing.Planes,+target: Planes.Spot,+kept: Tile.Tile8,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +below: {Nat.is_lt(index,8n) == True{} : Bool},+later: {Nat.is_lt(0n,pass) == True{} : Bool},+above: {Nat.is_lt(index,lane) == True{} : Bool},
  +lanes_room: {Nat.is_le(Nat.add(lane,Nat.mul(n,output_width_of(window))),U32.to_nat(cap)) == True{} : Bool},
  +passes_room: {Nat.is_le(Nat.add(pass,n),U32.to_nat(cap)) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Passes.passes_tile(n,pairs,U32.from_nat(pass),U32.from_nat(lane),base,second_lane,kernel_column,unit,window,planes,target,kept)) ==
    TileLaws.select_lane(index,kept) : F32}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +ow = output_width_of(window)
      +next = Passes.next_base(base,window,planes)
      +at = Passes.pass_at(U32.from_nat(lane),base,kernel_column,unit,window,planes)
      +shifted = pass_shifted(pairs,U32.from_nat(lane),base,kernel_column,unit,window,planes,target)
      +merged = Packing.merged(U32.from_nat(lane),kept,shifted)
      +tail = Nat.mul(rest,ow)
      +lane_room = Order.transitive(lane,Nat.add(lane,Nat.add(ow,tail)),U32.to_nat(cap),Order.le_add(lane,Nat.add(ow,tail)),lanes_room)
      +pass_room = Order.transitive(pass,Nat.add(pass,1n+rest),U32.to_nat(cap),Order.le_add(pass,1n+rest),passes_room)
      +positive = {
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(pass)),pass,Bounds.roundtrip(pass,cap,cap_ok,pass_room)) : {Nat.is_lt(0n,_) == True{} : Bool}
        later : {Nat.is_lt(0n,U32.to_nat(U32.from_nat(pass))) == True{} : Bool}}
      +nonzero = Arithmetic.word_positive_nonzero(U32.from_nat(pass),positive)
      +rest_lanes = {
        %Equal.sym(Nat,Nat.add(Nat.add(lane,ow),tail),Nat.add(lane,Nat.add(ow,tail)),Addition.associative(lane,ow,tail)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        lanes_room : {Nat.is_le(Nat.add(Nat.add(lane,ow),tail),U32.to_nat(cap)) == True{} : Bool}}
      +rest_passes = {
        %Addition.successor_right(pass,rest) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        passes_room : {Nat.is_le(Nat.add(1n+pass,rest),U32.to_nat(cap)) == True{} : Bool}}
      +onward = passes_after(rest,pairs,1n+pass,Nat.add(lane,ow),next,second_lane,kernel_column,unit,window,planes,target,merged,index,cap,cap_ok,below,{==},
        Order.lt_le(index,lane,Nat.add(lane,ow),above,Order.le_add(lane,ow)),rest_lanes,rest_passes)
      %Equal.sym(U32,(U32.from_nat(pass) + 1 : U32),U32.from_nat(1n+pass),Bounds.increment_encoded(pass)) :
        {TileLaws.select_lane(index,Passes.passes_tile(rest,pairs,_,Passes.next_lane(U32.from_nat(pass),U32.from_nat(lane),second_lane,window),next,second_lane,kernel_column,unit,
          window,planes,target,Passes.entry_value(U32.is_zero(U32.from_nat(pass)),Passes.lane_of(at),kept,shifted))) == TileLaws.select_lane(index,kept) : F32}
      %Equal.sym(U32,Passes.next_lane(U32.from_nat(pass),U32.from_nat(lane),second_lane,window),U32.from_nat(Nat.add(lane,ow)),
        later_lane(U32.from_nat(pass),lane,second_lane,window,nonzero)) :
        {TileLaws.select_lane(index,Passes.passes_tile(rest,pairs,U32.from_nat(1n+pass),_,next,second_lane,kernel_column,unit,
          window,planes,target,Passes.entry_value(U32.is_zero(U32.from_nat(pass)),Passes.lane_of(at),kept,shifted))) == TileLaws.select_lane(index,kept) : F32}
      %Equal.sym(Bool,U32.is_zero(U32.from_nat(pass)),False{},nonzero) :
        {TileLaws.select_lane(index,Passes.passes_tile(rest,pairs,U32.from_nat(1n+pass),U32.from_nat(Nat.add(lane,ow)),next,second_lane,kernel_column,unit,
          window,planes,target,Passes.entry_value(_,Passes.lane_of(at),kept,shifted))) == TileLaws.select_lane(index,kept) : F32}
      %Equal.sym(U32,Passes.lane_of(at),U32.from_nat(lane),pass_lane(U32.from_nat(lane),base,kernel_column,unit,window,planes)) :
        {TileLaws.select_lane(index,Passes.passes_tile(rest,pairs,U32.from_nat(1n+pass),U32.from_nat(Nat.add(lane,ow)),next,second_lane,kernel_column,unit,
          window,planes,target,Packing.merged(_,kept,shifted))) == TileLaws.select_lane(index,kept) : F32}
      Equal.trans(F32,TileLaws.select_lane(index,Passes.passes_tile(rest,pairs,U32.from_nat(1n+pass),U32.from_nat(Nat.add(lane,ow)),next,second_lane,kernel_column,unit,
        window,planes,target,merged)),TileLaws.select_lane(index,merged),TileLaws.select_lane(index,kept),onward,
        merged_kept(lane,kept,shifted,index,cap,cap_ok,below,above,lane_room))

# The first pass writes every lane, the passes from the second lane on leave
# the lanes below it.
def first_owner(+rest: Nat,+pairs: Tree.Tree<Tile.Tile16>,+lane: U32,+base: U32,+second: Nat,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,
  +planes: Packing.Planes,+target: Planes.Spot,+kept: Tile.Tile8,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +below: {Nat.is_lt(index,8n) == True{} : Bool},+owned: {Nat.is_lt(index,second) == True{} : Bool},
  +lanes_room: {Nat.is_le(Nat.add(second,Nat.mul(rest,output_width_of(window))),U32.to_nat(cap)) == True{} : Bool},
  +passes_room: {Nat.is_le(1n+rest,U32.to_nat(cap)) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Passes.passes_tile(1n+rest,pairs,U32.from_nat(0n),lane,base,U32.from_nat(second),kernel_column,unit,window,planes,target,kept)) ==
    TileLaws.select_lane(index,pass_shifted(pairs,lane,base,kernel_column,unit,window,planes,target)) : F32}:
  +next = Passes.next_base(base,window,planes)
  +at = Passes.pass_at(lane,base,kernel_column,unit,window,planes)
  +shifted = pass_shifted(pairs,lane,base,kernel_column,unit,window,planes,target)
  %Equal.sym(U32,Passes.next_lane(U32.from_nat(0n),lane,U32.from_nat(second),window),U32.from_nat(second),first_lane(lane,U32.from_nat(second),window)) :
    {TileLaws.select_lane(index,Passes.passes_tile(rest,pairs,(U32.from_nat(0n) + 1 : U32),_,next,U32.from_nat(second),kernel_column,unit,window,planes,target,
      Passes.entry_value(U32.is_zero(U32.from_nat(0n)),Passes.lane_of(at),kept,shifted))) == TileLaws.select_lane(index,shifted) : F32}
  passes_after(rest,pairs,1n,second,next,U32.from_nat(second),kernel_column,unit,window,planes,target,shifted,index,cap,cap_ok,below,{==},owned,lanes_room,passes_room)

# A later pass whose lane reaches a lane, the next pass's lane past it, owns it.
def later_owner(+rest: Nat,+pairs: Tree.Tree<Tile.Tile16>,+pass: Nat,+lane: Nat,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,
  +planes: Packing.Planes,+target: Planes.Spot,+kept: Tile.Tile8,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +below: {Nat.is_lt(index,8n) == True{} : Bool},+later: {Nat.is_lt(0n,pass) == True{} : Bool},+reached: {Nat.is_le(lane,index) == True{} : Bool},
  +owned: {Nat.is_lt(index,Nat.add(lane,output_width_of(window))) == True{} : Bool},
  +lanes_room: {Nat.is_le(Nat.add(Nat.add(lane,output_width_of(window)),Nat.mul(rest,output_width_of(window))),U32.to_nat(cap)) == True{} : Bool},
  +passes_room: {Nat.is_le(Nat.add(pass,1n+rest),U32.to_nat(cap)) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Passes.passes_tile(1n+rest,pairs,U32.from_nat(pass),U32.from_nat(lane),base,second_lane,kernel_column,unit,window,planes,target,kept)) ==
    TileLaws.select_lane(index,pass_shifted(pairs,U32.from_nat(lane),base,kernel_column,unit,window,planes,target)) : F32}:
  +ow = output_width_of(window)
  +next = Passes.next_base(base,window,planes)
  +at = Passes.pass_at(U32.from_nat(lane),base,kernel_column,unit,window,planes)
  +shifted = pass_shifted(pairs,U32.from_nat(lane),base,kernel_column,unit,window,planes,target)
  +merged = Packing.merged(U32.from_nat(lane),kept,shifted)
  +reach = Nat.add(Nat.add(lane,ow),Nat.mul(rest,ow))
  +pass_room = Order.transitive(pass,Nat.add(pass,1n+rest),U32.to_nat(cap),Order.le_add(pass,1n+rest),passes_room)
  +index_room = Order.transitive(index,reach,U32.to_nat(cap),
    Order.transitive(index,Nat.add(lane,ow),reach,Bounds.strict_le(index,Nat.add(lane,ow),owned),Order.le_add(Nat.add(lane,ow),Nat.mul(rest,ow))),lanes_room)
  +positive = {
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(pass)),pass,Bounds.roundtrip(pass,cap,cap_ok,pass_room)) : {Nat.is_lt(0n,_) == True{} : Bool}
    later : {Nat.is_lt(0n,U32.to_nat(U32.from_nat(pass))) == True{} : Bool}}
  +nonzero = Arithmetic.word_positive_nonzero(U32.from_nat(pass),positive)
  +rest_passes = {
    %Addition.successor_right(pass,rest) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    passes_room : {Nat.is_le(Nat.add(1n+pass,rest),U32.to_nat(cap)) == True{} : Bool}}
  +onward = passes_after(rest,pairs,1n+pass,Nat.add(lane,ow),next,second_lane,kernel_column,unit,window,planes,target,merged,index,cap,cap_ok,below,{==},owned,lanes_room,
    rest_passes)
  %Equal.sym(U32,(U32.from_nat(pass) + 1 : U32),U32.from_nat(1n+pass),Bounds.increment_encoded(pass)) :
    {TileLaws.select_lane(index,Passes.passes_tile(rest,pairs,_,Passes.next_lane(U32.from_nat(pass),U32.from_nat(lane),second_lane,window),next,second_lane,kernel_column,unit,
      window,planes,target,Passes.entry_value(U32.is_zero(U32.from_nat(pass)),Passes.lane_of(at),kept,shifted))) == TileLaws.select_lane(index,shifted) : F32}
  %Equal.sym(U32,Passes.next_lane(U32.from_nat(pass),U32.from_nat(lane),second_lane,window),U32.from_nat(Nat.add(lane,ow)),
    later_lane(U32.from_nat(pass),lane,second_lane,window,nonzero)) :
    {TileLaws.select_lane(index,Passes.passes_tile(rest,pairs,U32.from_nat(1n+pass),_,next,second_lane,kernel_column,unit,
      window,planes,target,Passes.entry_value(U32.is_zero(U32.from_nat(pass)),Passes.lane_of(at),kept,shifted))) == TileLaws.select_lane(index,shifted) : F32}
  %Equal.sym(Bool,U32.is_zero(U32.from_nat(pass)),False{},nonzero) :
    {TileLaws.select_lane(index,Passes.passes_tile(rest,pairs,U32.from_nat(1n+pass),U32.from_nat(Nat.add(lane,ow)),next,second_lane,kernel_column,unit,
      window,planes,target,Passes.entry_value(_,Passes.lane_of(at),kept,shifted))) == TileLaws.select_lane(index,shifted) : F32}
  %Equal.sym(U32,Passes.lane_of(at),U32.from_nat(lane),pass_lane(U32.from_nat(lane),base,kernel_column,unit,window,planes)) :
    {TileLaws.select_lane(index,Passes.passes_tile(rest,pairs,U32.from_nat(1n+pass),U32.from_nat(Nat.add(lane,ow)),next,second_lane,kernel_column,unit,
      window,planes,target,Packing.merged(_,kept,shifted))) == TileLaws.select_lane(index,shifted) : F32}
  Equal.trans(F32,TileLaws.select_lane(index,Passes.passes_tile(rest,pairs,U32.from_nat(1n+pass),U32.from_nat(Nat.add(lane,ow)),next,second_lane,kernel_column,unit,
    window,planes,target,merged)),TileLaws.select_lane(index,merged),TileLaws.select_lane(index,shifted),onward,
    merged_taken(lane,kept,shifted,index,cap,cap_ok,below,reached,index_room))

# The lane and the plane offset of the pass count passes on.
def lane_after(count: Nat,+lane: Nat,+step: Nat) -> Nat:
  match count:
    case 0n: lane
    case 1n+ +rest: lane_after(rest,Nat.add(lane,step),step)

def base_after(count: Nat,+base: U32,+window: Packing.Stencil,+planes: Packing.Planes) -> U32:
  match count:
    case 0n: base
    case 1n+ +rest: base_after(rest,Passes.next_base(base,window,planes),window,planes)

def lane_after_step(count: Nat,+lane: Nat,+step: Nat) -> {lane_after(count,Nat.add(lane,step),step) == Nat.add(lane_after(count,lane,step),step) : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest: lane_after_step(rest,Nat.add(lane,step),step)

# The passes before the owner of a lane leave whatever they write in it to
# the owner.
def later_skip(count: Nat,+rest: Nat,+pairs: Tree.Tree<Tile.Tile16>,+pass: Nat,+lane: Nat,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,
  +window: Packing.Stencil,+planes: Packing.Planes,+target: Planes.Spot,+kept: Tile.Tile8,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +below: {Nat.is_lt(index,8n) == True{} : Bool},+later: {Nat.is_lt(0n,pass) == True{} : Bool},
  +reached: {Nat.is_le(lane_after(count,lane,output_width_of(window)),index) == True{} : Bool},
  +owned: {Nat.is_lt(index,Nat.add(lane_after(count,lane,output_width_of(window)),output_width_of(window))) == True{} : Bool},
  +lanes_room: {Nat.is_le(Nat.add(Nat.add(lane_after(count,lane,output_width_of(window)),output_width_of(window)),Nat.mul(rest,output_width_of(window))),U32.to_nat(cap)) ==
    True{} : Bool},
  +passes_room: {Nat.is_le(Nat.add(pass,Nat.add(count,1n+rest)),U32.to_nat(cap)) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Passes.passes_tile(Nat.add(count,1n+rest),pairs,U32.from_nat(pass),U32.from_nat(lane),base,second_lane,kernel_column,unit,window,planes,target,
    kept)) ==
    TileLaws.select_lane(index,pass_shifted(pairs,U32.from_nat(lane_after(count,lane,output_width_of(window))),base_after(count,base,window,planes),kernel_column,unit,window,
      planes,target)) : F32}:
  match count:
    case 0n: later_owner(rest,pairs,pass,lane,base,second_lane,kernel_column,unit,window,planes,target,kept,index,cap,cap_ok,below,later,reached,owned,lanes_room,passes_room)
    case 1n+ +more:
      +ow = output_width_of(window)
      +next = Passes.next_base(base,window,planes)
      +at = Passes.pass_at(U32.from_nat(lane),base,kernel_column,unit,window,planes)
      +shifted = pass_shifted(pairs,U32.from_nat(lane),base,kernel_column,unit,window,planes,target)
      +passed = Passes.entry_value(U32.is_zero(U32.from_nat(pass)),Passes.lane_of(at),kept,shifted)
      +total = Nat.add(more,1n+rest)
      +pass_room = Order.transitive(pass,Nat.add(pass,1n+total),U32.to_nat(cap),Order.le_add(pass,1n+total),passes_room)
      +positive = {
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(pass)),pass,Bounds.roundtrip(pass,cap,cap_ok,pass_room)) : {Nat.is_lt(0n,_) == True{} : Bool}
        later : {Nat.is_lt(0n,U32.to_nat(U32.from_nat(pass))) == True{} : Bool}}
      +nonzero = Arithmetic.word_positive_nonzero(U32.from_nat(pass),positive)
      +rest_passes = {
        %Addition.successor_right(pass,total) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        passes_room : {Nat.is_le(Nat.add(1n+pass,total),U32.to_nat(cap)) == True{} : Bool}}
      %Equal.sym(U32,(U32.from_nat(pass) + 1 : U32),U32.from_nat(1n+pass),Bounds.increment_encoded(pass)) :
        {TileLaws.select_lane(index,Passes.passes_tile(total,pairs,_,Passes.next_lane(U32.from_nat(pass),U32.from_nat(lane),second_lane,window),next,second_lane,kernel_column,unit,
          window,planes,target,passed)) ==
          TileLaws.select_lane(index,pass_shifted(pairs,U32.from_nat(lane_after(more,Nat.add(lane,ow),ow)),base_after(more,next,window,planes),kernel_column,unit,window,
            planes,target)) : F32}
      %Equal.sym(U32,Passes.next_lane(U32.from_nat(pass),U32.from_nat(lane),second_lane,window),U32.from_nat(Nat.add(lane,ow)),
        later_lane(U32.from_nat(pass),lane,second_lane,window,nonzero)) :
        {TileLaws.select_lane(index,Passes.passes_tile(total,pairs,U32.from_nat(1n+pass),_,next,second_lane,kernel_column,unit,window,planes,target,passed)) ==
          TileLaws.select_lane(index,pass_shifted(pairs,U32.from_nat(lane_after(more,Nat.add(lane,ow),ow)),base_after(more,next,window,planes),kernel_column,unit,window,
            planes,target)) : F32}
      later_skip(more,rest,pairs,1n+pass,Nat.add(lane,ow),next,second_lane,kernel_column,unit,window,planes,target,passed,index,cap,cap_ok,below,{==},reached,owned,lanes_room,
        rest_passes)

# The lane of the pass count passes in, the first pass at lane zero.
def owner_lane(count: Nat,+second: Nat,+step: Nat) -> Nat:
  match count:
    case 0n: 0n
    case 1n+ +rest: lane_after(rest,second,step)

# Lane index of a kernel column's tile is the lane of the pass count passes
# in, when that pass reaches it and the next does not.
def owner(count: Nat,+rest: Nat,+pairs: Tree.Tree<Tile.Tile16>,+base: U32,+second: Nat,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes,
  +target: Planes.Spot,+kept: Tile.Tile8,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+below: {Nat.is_lt(index,8n) == True{} : Bool},
  +reached: {Nat.is_le(owner_lane(count,second,output_width_of(window)),index) == True{} : Bool},
  +owned: {Nat.is_lt(index,lane_after(count,second,output_width_of(window))) == True{} : Bool},
  +lanes_room: {Nat.is_le(Nat.add(lane_after(count,second,output_width_of(window)),Nat.mul(rest,output_width_of(window))),U32.to_nat(cap)) == True{} : Bool},
  +passes_room: {Nat.is_le(Nat.add(count,1n+rest),U32.to_nat(cap)) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Passes.passes_tile(Nat.add(count,1n+rest),pairs,0,0,base,U32.from_nat(second),kernel_column,unit,window,planes,target,kept)) ==
    TileLaws.select_lane(index,pass_shifted(pairs,U32.from_nat(owner_lane(count,second,output_width_of(window))),base_after(count,base,window,planes),kernel_column,unit,
      window,planes,target)) : F32}:
  match count:
    case 0n: first_owner(rest,pairs,U32.from_nat(0n),base,second,kernel_column,unit,window,planes,target,kept,index,cap,cap_ok,below,owned,lanes_room,passes_room)
    case 1n+ +more:
      +ow = output_width_of(window)
      +next = Passes.next_base(base,window,planes)
      +at = Passes.pass_at(0,base,kernel_column,unit,window,planes)
      +shifted = pass_shifted(pairs,0,base,kernel_column,unit,window,planes,target)
      +passed = Passes.entry_value(U32.is_zero(0),Passes.lane_of(at),kept,shifted)
      +total = Nat.add(more,1n+rest)
      +stepped = lane_after_step(more,second,ow)
      +owned_after = {
        %stepped : {Nat.is_lt(index,_) == True{} : Bool}
        owned : {Nat.is_lt(index,Nat.add(lane_after(more,second,ow),ow)) == True{} : Bool}}
      +lanes_after = {
        %stepped : {Nat.is_le(Nat.add(_,Nat.mul(rest,ow)),U32.to_nat(cap)) == True{} : Bool}
        lanes_room : {Nat.is_le(Nat.add(Nat.add(lane_after(more,second,ow),ow),Nat.mul(rest,ow)),U32.to_nat(cap)) == True{} : Bool}}
      %Equal.sym(U32,Passes.next_lane(U32.from_nat(0n),0,U32.from_nat(second),window),U32.from_nat(second),first_lane(0,U32.from_nat(second),window)) :
        {TileLaws.select_lane(index,Passes.passes_tile(total,pairs,(0 + 1 : U32),_,next,U32.from_nat(second),kernel_column,unit,window,planes,target,passed)) ==
          TileLaws.select_lane(index,pass_shifted(pairs,U32.from_nat(lane_after(more,second,ow)),base_after(more,next,window,planes),kernel_column,unit,window,planes,target)) :
          F32}
      later_skip(more,rest,pairs,1n,second,next,U32.from_nat(second),kernel_column,unit,window,planes,target,passed,index,cap,cap_ok,below,{==},reached,owned_after,lanes_after,
        passes_room)

# ---- Which pass owns a lane ----

def lane_closed(count: Nat,+lane: Nat,+step: Nat) -> {lane_after(count,lane,step) == Nat.add(lane,Nat.mul(count,step)) : Nat}:
  match count:
    case 0n: Equal.sym(Nat,Nat.add(lane,0n),lane,Addition.zero_right(lane))
    case 1n+ +rest:
      Equal.trans(Nat,lane_after(rest,Nat.add(lane,step),step),Nat.add(Nat.add(lane,step),Nat.mul(rest,step)),Nat.add(lane,Nat.add(step,Nat.mul(rest,step))),
        lane_closed(rest,Nat.add(lane,step),step),Addition.associative(lane,step,Nat.mul(rest,step)))

# The column of the first position and the second lane make a whole output
# row.
def width_split(+offset: Nat,+width: Nat,+tail: Nat,+inside: {Nat.is_le(offset,width) == True{} : Bool}) ->
  {Nat.add(offset,Nat.add(Nat.sub(width,offset),tail)) == Nat.add(width,tail) : Nat}:
  +second = Nat.sub(width,offset)
  Equal.trans(Nat,Nat.add(offset,Nat.add(second,tail)),Nat.add(Nat.add(offset,second),tail),Nat.add(width,tail),
    Equal.sym(Nat,Nat.add(Nat.add(offset,second),tail),Nat.add(offset,Nat.add(second,tail)),Addition.associative(offset,second,tail)),
    Equal.cong(Nat,Nat,head => Nat.add(head,tail),Nat.add(offset,second),width,Order.restore_shift(offset,width,inside)))

# The pass a lane falls in reaches it.
def owner_reached_at(count: Nat,+offset: Nat,+width: Nat,+index: Nat,+inside: {Nat.is_le(offset,width) == True{} : Bool},
  +reach: {Nat.is_le(Nat.mul(count,width),Nat.add(offset,index)) == True{} : Bool}) ->
  {Nat.is_le(owner_lane(count,Nat.sub(width,offset),width),index) == True{} : Bool}:
  match count:
    case 0n: Order.zero_le(index)
    case 1n+ +rest:
      +second = Nat.sub(width,offset)
      +tail = Nat.mul(rest,width)
      %Equal.sym(Nat,lane_after(rest,second,width),Nat.add(second,tail),lane_closed(rest,second,width)) : {Nat.is_le(_,index) == True{} : Bool}
      Order.cancel_offset_le(offset,Nat.add(second,tail),index,{
        %Equal.sym(Nat,Nat.add(offset,Nat.add(second,tail)),Nat.add(width,tail),width_split(offset,width,tail,inside)) : {Nat.is_le(_,Nat.add(offset,index)) == True{} : Bool}
        reach : {Nat.is_le(Nat.add(offset,Nat.add(second,tail)),Nat.add(offset,index)) == True{} : Bool}})

# The pass a lane falls in: (offset + index) / width.
def owner_of(+offset: Nat,+width: Nat,+index: Nat) -> Nat:
  Nat.div(Nat.add(offset,index),width)

def owner_reached(+offset: Nat,+width: Nat,+index: Nat,+positive: {Nat.is_lt(0n,width) == True{} : Bool},+inside: {Nat.is_le(offset,width) == True{} : Bool}) ->
  {Nat.is_le(owner_lane(owner_of(offset,width,index),Nat.sub(width,offset),width),index) == True{} : Bool}:
  owner_reached_at(owner_of(offset,width,index),offset,width,index,inside,DivisionOrder.product_below(Nat.add(offset,index),width,positive))

# The pass after it starts past the lane.
def owner_owned(+offset: Nat,+width: Nat,+index: Nat,+positive: {Nat.is_lt(0n,width) == True{} : Bool},+inside: {Nat.is_le(offset,width) == True{} : Bool}) ->
  {Nat.is_lt(index,lane_after(owner_of(offset,width,index),Nat.sub(width,offset),width)) == True{} : Bool}:
  +owner = owner_of(offset,width,index)
  +second = Nat.sub(width,offset)
  +tail = Nat.mul(owner,width)
  %Equal.sym(Nat,lane_after(owner,second,width),Nat.add(second,tail),lane_closed(owner,second,width)) : {Nat.is_lt(index,_) == True{} : Bool}
  Order.cancel_offset_lt(offset,index,Nat.add(second,tail),{
    %Equal.sym(Nat,Nat.add(offset,Nat.add(second,tail)),Nat.add(width,tail),width_split(offset,width,tail,inside)) : {Nat.is_lt(Nat.add(offset,index),_) == True{} : Bool}
    DivisionOrder.next_above(Nat.add(offset,index),width,positive) : {Nat.is_lt(Nat.add(offset,index),Nat.add(offset,Nat.add(second,tail))) == True{} : Bool}})

# A count past an owner splits at it.
def count_split(+owner: Nat,+last: Nat,+ordered: {Nat.is_le(owner,last) == True{} : Bool}) -> {Nat.add(owner,1n+Nat.sub(last,owner)) == 1n+last : Nat}:
  Equal.trans(Nat,Nat.add(owner,1n+Nat.sub(last,owner)),1n+Nat.add(owner,Nat.sub(last,owner)),1n+last,Addition.successor_right(owner,Nat.sub(last,owner)),
    Equal.cong(Nat,Nat,value => 1n+value,Nat.add(owner,Nat.sub(last,owner)),last,Order.restore_shift(owner,last,ordered)))

def le_false_less(+left: Nat,+right: Nat,+outside: {Nat.is_le(left,right) == False{} : Bool}) -> {Nat.is_lt(right,left) == True{} : Bool}:
  match left right:
    case 0n +right: Empty.absurd({Nat.is_lt(right,0n) == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_le(0n,right),True{},Equal.sym(Bool,Nat.is_le(0n,right),False{},outside),Order.zero_le(right))))
    case 1n+ +left 0n: {==}
    case 1n+ +left 1n+ +right: le_false_less(left,right,outside)

# The second lane of a unit at a position: the output width less its column.
def second_word(+window: Packing.Stencil,+position: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,output_width_of(window)) == True{} : Bool},+divisor_bound: {Nat.is_le(Nat.double(output_width_of(window)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Passes.second_lane_of(window,position) == U32.from_nat(Nat.sub(output_width_of(window),Nat.mod(U32.to_nat(position),output_width_of(window)))) : U32}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +ow = U32.to_nat(output_width)
  +column = Nat.mod(U32.to_nat(position),ow)
  +exact = WordDivision.modulo(position,output_width,cap,cap_ok,divisor_bound,positive)
  +inside = Bounds.strict_le(column,ow,DivisionOrder.remainder_small(U32.to_nat(position),ow,positive))
  %Arithmetic.native_roundtrip(output_width) : {(_ - position % output_width : U32) == U32.from_nat(Nat.sub(ow,column)) : U32}
  %Arithmetic.from_nat_exact((position % output_width : U32),column,exact) : {(U32.from_nat(ow) - _ : U32) == U32.from_nat(Nat.sub(ow,column)) : U32}
  Subtraction.encoded_subtract(ow,column,inside)

def count_case(+wide: Bool,+output_width: U32,+column: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(output_width)) == True{} : Bool},+divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(output_width)),U32.to_nat(cap)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(output_width),8n),U32.to_nat(cap)) == True{} : Bool},+inside: {Nat.is_le(column,U32.to_nat(output_width)) == True{} : Bool},
  +exact: {Nat.is_le(8n,Nat.sub(U32.to_nat(output_width),column)) == wide : Bool}) ->
  {U32.to_nat(Bool.pick(U32,wide,1,(1 + (8 - U32.from_nat(Nat.sub(U32.to_nat(output_width),column)) + output_width - 1) / output_width : U32))) ==
    1n+owner_of(column,U32.to_nat(output_width),7n) : Nat}:
  match wide:
    case True{}:
      +ow = U32.to_nat(output_width)
      +second = Nat.sub(ow,column)
      +last = Nat.add(column,7n)
      +whole = {
        %Order.restore_shift(column,ow,inside) : {Nat.is_le(Nat.add(column,8n),_) == True{} : Bool}
        Order.offset_le(column,8n,second,exact) : {Nat.is_le(Nat.add(column,8n),ow) == True{} : Bool}}
      +below = Order.lt_le(last,Nat.add(column,8n),ow,Order.offset_lt(column,7n,8n,{==}),whole)
      %Equal.sym(Nat,Nat.div(last,ow),0n,DivisionOrder.small_quotient(last,ow,below)) : {1n == 1n+_ : Nat}
      {==}
    case False{}:
      +ow = U32.to_nat(output_width)
      +second = Nat.sub(ow,column)
      +last = Nat.add(column,7n)
      +narrow = le_false_less(8n,second,exact)
      +short = Nat.sub(8n,second)
      +sum = Nat.add(short,ow)
      +total = Nat.sub(sum,1n)
      +one_room = Order.transitive(1n,ow,sum,Order.strict_follow(0n,ow,positive),Order.right_le(short,ow))
      +sum_exact = Equal.trans(Nat,sum,Nat.add(short,Nat.add(column,second)),Nat.add(column,8n),
        Equal.cong(Nat,Nat,value => Nat.add(short,value),ow,Nat.add(column,second),Equal.sym(Nat,Nat.add(column,second),ow,Order.restore_shift(column,ow,inside))),
        Equal.trans(Nat,Nat.add(short,Nat.add(column,second)),Nat.add(column,Nat.add(short,second)),Nat.add(column,8n),RowMajor.add_swap(short,column,second),
          Equal.cong(Nat,Nat,value => Nat.add(column,value),Nat.add(short,second),8n,
            Equal.trans(Nat,Nat.add(short,second),Nat.add(second,short),8n,Order.commutative(short,second),Order.restore_shift(second,8n,Bounds.strict_le(second,8n,narrow))))))
      +total_exact = Equal.trans(Nat,total,Nat.sub(Nat.add(last,1n),1n),last,
        Equal.cong(Nat,Nat,value => Nat.sub(value,1n),sum,Nat.add(last,1n),
          Equal.trans(Nat,sum,Nat.add(column,8n),Nat.add(last,1n),sum_exact,Equal.sym(Nat,Nat.add(last,1n),Nat.add(column,8n),Addition.associative(column,7n,1n)))),
        DivisionOrder.subtract_sum(last,1n))
      +last_room = Order.transitive(last,Nat.add(ow,7n),U32.to_nat(cap),Order.add_le(column,ow,7n,inside),
        Order.transitive(Nat.add(ow,7n),Nat.add(ow,8n),U32.to_nat(cap),Order.offset_le(ow,7n,8n,{==}),room))
      +total_room = {
        %Equal.sym(Nat,total,last,total_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        last_room : {Nat.is_le(total,U32.to_nat(cap)) == True{} : Bool}}
      +ow_room = Order.transitive(ow,Nat.double(ow),U32.to_nat(cap),Order.le_double(ow),divisor_bound)
      +quotient = Nat.div(last,ow)
      +quotient_room = Order.transitive(1n+quotient,1n+last,U32.to_nat(cap),
        Order.transitive(1n+quotient,1n+Nat.mul(quotient,ow),1n+last,Order.offset_le(1n,quotient,Nat.mul(quotient,ow),RowMajor.product_covers(quotient,ow,positive)),
          Order.offset_le(1n,Nat.mul(quotient,ow),last,DivisionOrder.product_below(last,ow,positive))),
        Order.transitive(1n+last,Nat.add(ow,8n),U32.to_nat(cap),{
          %Addition.successor_right(column,7n) : {Nat.is_le(_,Nat.add(ow,8n)) == True{} : Bool}
          Order.add_le(column,ow,8n,inside) : {Nat.is_le(1n+last,Nat.add(ow,8n)) == True{} : Bool}},room))
      %Arithmetic.native_roundtrip(output_width) : {U32.to_nat((1 + (8 - U32.from_nat(second) + _ - 1) / _ : U32)) == 1n+quotient : Nat}
      %Equal.sym(U32,(U32.from_nat(8n) - U32.from_nat(second) : U32),U32.from_nat(short),Subtraction.encoded_subtract(8n,second,Bounds.strict_le(second,8n,narrow))) :
        {U32.to_nat((1 + (_ + U32.from_nat(ow) - 1) / U32.from_nat(ow) : U32)) == 1n+quotient : Nat}
      %Equal.sym(U32,(U32.from_nat(short) + U32.from_nat(ow) : U32),U32.from_nat(sum),WordArithmetic.native_add(short,ow)) :
        {U32.to_nat((1 + (_ - 1) / U32.from_nat(ow) : U32)) == 1n+quotient : Nat}
      %Equal.sym(U32,(U32.from_nat(sum) - U32.from_nat(1n) : U32),U32.from_nat(total),Subtraction.encoded_subtract(sum,1n,one_room)) :
        {U32.to_nat((1 + _ / U32.from_nat(ow) : U32)) == 1n+quotient : Nat}
      %Equal.sym(U32,(U32.from_nat(total) / U32.from_nat(ow) : U32),U32.from_nat(Nat.div(total,ow)),
        WordDivision.encoded_divide(total,ow,cap,cap_ok,total_room,ow_room,divisor_bound,positive)) :
        {U32.to_nat((1 + _ : U32)) == 1n+quotient : Nat}
      %Equal.sym(Nat,total,last,total_exact) : {U32.to_nat((1 + U32.from_nat(Nat.div(_,ow)) : U32)) == 1n+quotient : Nat}
      %Equal.sym(U32,(U32.from_nat(1n) + U32.from_nat(quotient) : U32),U32.from_nat(1n+quotient),WordArithmetic.native_add(1n,quotient)) : {U32.to_nat(_) == 1n+quotient : Nat}
      Bounds.roundtrip(1n+quotient,cap,cap_ok,quotient_room)

# The passes of a unit: one, and one more for each output row its last
# lane crosses into.
def pass_count_value(+window: Packing.Stencil,+position: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,output_width_of(window)) == True{} : Bool},+divisor_bound: {Nat.is_le(Nat.double(output_width_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(output_width_of(window),8n),U32.to_nat(cap)) == True{} : Bool}) ->
  {Model.pass_count(window,position) == 1n+owner_of(Nat.mod(U32.to_nat(position),output_width_of(window)),output_width_of(window),7n) : Nat}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +ow = U32.to_nat(output_width)
  +column = Nat.mod(U32.to_nat(position),ow)
  +second = Nat.sub(ow,column)
  +second_exact = second_word(window,position,cap,cap_ok,positive,divisor_bound)
  +inside = Bounds.strict_le(column,ow,DivisionOrder.remainder_small(U32.to_nat(position),ow,positive))
  +second_room = Order.transitive(second,ow,U32.to_nat(cap),Order.sub_le(ow,column),Order.transitive(ow,Nat.add(ow,8n),U32.to_nat(cap),Order.le_add(ow,8n),room))
  %Equal.sym(U32,(output_width - position % output_width : U32),U32.from_nat(second),second_exact) :
    {U32.to_nat(Bool.pick(U32,U32.is_le(8,_),1,(1 + (8 - _ + output_width - 1) / output_width : U32))) == 1n+owner_of(column,ow,7n) : Nat}
  %Equal.sym(Bool,U32.is_le(U32.from_nat(8n),U32.from_nat(second)),Nat.is_le(8n,second),word_le(8n,second,cap,cap_ok,
    Order.transitive(8n,Nat.add(ow,8n),U32.to_nat(cap),Order.right_le(ow,8n),room),second_room)) :
    {U32.to_nat(Bool.pick(U32,_,1,(1 + (8 - U32.from_nat(second) + output_width - 1) / output_width : U32))) == 1n+owner_of(column,ow,7n) : Nat}
  count_case(Nat.is_le(8n,second),output_width,column,cap,cap_ok,positive,divisor_bound,room,inside,{==})


# The owner of a lane falls among the passes of its unit.
def owner_inside(+column: Nat,+width: Nat,+index: Nat,+positive: {Nat.is_lt(0n,width) == True{} : Bool},+below: {Nat.is_lt(index,8n) == True{} : Bool}) ->
  {Nat.is_le(owner_of(column,width,index),owner_of(column,width,7n)) == True{} : Bool}:
  DivisionOrder.quotient_le(Nat.add(column,index),Nat.add(column,7n),width,positive,Order.offset_le(column,index,7n,Division.lt_succ_le(index,7n,below)))

# The lanes of the passes up to the last stay within an output row past
# the width.
def lanes_reach(+column: Nat,+width: Nat,+index: Nat,+cap: U32,+positive: {Nat.is_lt(0n,width) == True{} : Bool},+inside: {Nat.is_le(column,width) == True{} : Bool},
  +below: {Nat.is_lt(index,8n) == True{} : Bool},+room: {Nat.is_le(Nat.add(width,8n),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(lane_after(owner_of(column,width,index),Nat.sub(width,column),width),
    Nat.mul(Nat.sub(owner_of(column,width,7n),owner_of(column,width,index)),width)),U32.to_nat(cap)) == True{} : Bool}:
  +owner = owner_of(column,width,index)
  +last = owner_of(column,width,7n)
  +rest = Nat.sub(last,owner)
  +second = Nat.sub(width,column)
  +ordered = owner_inside(column,width,index,positive,below)
  +form = Equal.trans(Nat,Nat.add(lane_after(owner,second,width),Nat.mul(rest,width)),Nat.add(Nat.add(second,Nat.mul(owner,width)),Nat.mul(rest,width)),
    Nat.add(second,Nat.mul(last,width)),
    Equal.cong(Nat,Nat,value => Nat.add(value,Nat.mul(rest,width)),lane_after(owner,second,width),Nat.add(second,Nat.mul(owner,width)),lane_closed(owner,second,width)),
    Equal.trans(Nat,Nat.add(Nat.add(second,Nat.mul(owner,width)),Nat.mul(rest,width)),Nat.add(second,Nat.add(Nat.mul(owner,width),Nat.mul(rest,width))),
      Nat.add(second,Nat.mul(last,width)),Addition.associative(second,Nat.mul(owner,width),Nat.mul(rest,width)),
      Equal.cong(Nat,Nat,value => Nat.add(second,value),Nat.add(Nat.mul(owner,width),Nat.mul(rest,width)),Nat.mul(last,width),
        Equal.trans(Nat,Nat.add(Nat.mul(owner,width),Nat.mul(rest,width)),Nat.mul(Nat.add(owner,rest),width),Nat.mul(last,width),
          Equal.sym(Nat,Nat.mul(Nat.add(owner,rest),width),Nat.add(Nat.mul(owner,width),Nat.mul(rest,width)),RowMajor.distrib(owner,rest,width)),
          Equal.cong(Nat,Nat,value => Nat.mul(value,width),Nat.add(owner,rest),last,Order.restore_shift(owner,last,ordered))))))
  +end = Equal.trans(Nat,Nat.add(second,Nat.add(column,7n)),Nat.add(column,Nat.add(second,7n)),Nat.add(width,7n),RowMajor.add_swap(second,column,7n),
    width_split(column,width,7n,inside))
  %Equal.sym(Nat,Nat.add(lane_after(owner,second,width),Nat.mul(rest,width)),Nat.add(second,Nat.mul(last,width)),form) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
  Order.transitive(Nat.add(second,Nat.mul(last,width)),Nat.add(second,Nat.add(column,7n)),U32.to_nat(cap),
    Order.offset_le(second,Nat.mul(last,width),Nat.add(column,7n),DivisionOrder.product_below(Nat.add(column,7n),width,positive)),
    Order.transitive(Nat.add(second,Nat.add(column,7n)),Nat.add(width,7n),U32.to_nat(cap),{
      %end : {Nat.is_le(Nat.add(second,Nat.add(column,7n)),_) == True{} : Bool}
      Order.reflexive(Nat.add(second,Nat.add(column,7n))) : {Nat.is_le(Nat.add(second,Nat.add(column,7n)),Nat.add(width,7n)) == True{} : Bool}},
      Order.transitive(Nat.add(width,7n),Nat.add(width,8n),U32.to_nat(cap),Order.offset_le(width,7n,8n,{==}),room)))

# The passes of a unit fit below the cap.
def passes_reach(+column: Nat,+width: Nat,+cap: U32,+positive: {Nat.is_lt(0n,width) == True{} : Bool},+inside: {Nat.is_le(column,width) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(width,8n),U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(1n+owner_of(column,width,7n),U32.to_nat(cap)) == True{} : Bool}:
  +last = owner_of(column,width,7n)
  +value = Nat.add(column,7n)
  Order.transitive(1n+last,1n+value,U32.to_nat(cap),
    Order.offset_le(1n,last,value,Order.transitive(last,Nat.mul(last,width),value,RowMajor.product_covers(last,width,positive),DivisionOrder.product_below(value,width,positive))),
    Order.transitive(1n+value,Nat.add(width,8n),U32.to_nat(cap),{
      %Addition.successor_right(column,7n) : {Nat.is_le(_,Nat.add(width,8n)) == True{} : Bool}
      Order.add_le(column,width,8n,inside) : {Nat.is_le(1n+value,Nat.add(width,8n)) == True{} : Bool}},room))

# Lane index of a kernel column's tile of a target entry is that lane of the
# shifted pair of the pass that owns it.
def column_owner(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,
  +unit: U32,+target: Planes.Spot,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+below: {Nat.is_lt(index,8n) == True{} : Bool},
  +positive: {Nat.is_lt(0n,output_width_of(window)) == True{} : Bool},+divisor_bound: {Nat.is_le(Nat.double(output_width_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(output_width_of(window),8n),U32.to_nat(cap)) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Passes.column_tile(pairs,tiles,kernel_column,window,planes,position,unit,target)) ==
    TileLaws.select_lane(index,pass_shifted(pairs,
      U32.from_nat(owner_lane(owner_of(Nat.mod(U32.to_nat(position),output_width_of(window)),output_width_of(window),index),
        Nat.sub(output_width_of(window),Nat.mod(U32.to_nat(position),output_width_of(window))),output_width_of(window))),
      base_after(owner_of(Nat.mod(U32.to_nat(position),output_width_of(window)),output_width_of(window),index),Model.first_base(window,planes,position,kernel_column),window,planes),
      kernel_column,unit,window,planes,target)) : F32}:
  +ow = output_width_of(window)
  +column = Nat.mod(U32.to_nat(position),ow)
  +second = Nat.sub(ow,column)
  +owning = owner_of(column,ow,index)
  +last = owner_of(column,ow,7n)
  +rest = Nat.sub(last,owning)
  +first = Model.first_base(window,planes,position,kernel_column)
  +kept = Tree.read(Tile.Tile8,tiles,Passes.column_slot(window,kernel_column,unit,target))
  +inside = Bounds.strict_le(column,ow,DivisionOrder.remainder_small(U32.to_nat(position),ow,positive))
  +goal = TileLaws.select_lane(index,pass_shifted(pairs,U32.from_nat(owner_lane(owning,second,ow)),base_after(owning,first,window,planes),kernel_column,unit,window,planes,target))
  %Equal.sym(Nat,Model.pass_count(window,position),1n+last,pass_count_value(window,position,cap,cap_ok,positive,divisor_bound,room)) :
    {TileLaws.select_lane(index,Passes.passes_tile(_,pairs,0,0,first,Passes.second_lane_of(window,position),kernel_column,unit,window,planes,target,kept)) == goal : F32}
  %count_split(owning,last,owner_inside(column,ow,index,positive,below)) :
    {TileLaws.select_lane(index,Passes.passes_tile(_,pairs,0,0,first,Passes.second_lane_of(window,position),kernel_column,unit,window,planes,target,kept)) == goal : F32}
  %Equal.sym(U32,Passes.second_lane_of(window,position),U32.from_nat(second),second_word(window,position,cap,cap_ok,positive,divisor_bound)) :
    {TileLaws.select_lane(index,Passes.passes_tile(Nat.add(owning,1n+rest),pairs,0,0,first,_,kernel_column,unit,window,planes,target,kept)) == goal : F32}
  owner(owning,rest,pairs,first,second,kernel_column,unit,window,planes,target,kept,index,cap,cap_ok,below,owner_reached(column,ow,index,positive,inside),
    owner_owned(column,ow,index,positive,inside),lanes_reach(column,ow,index,cap,positive,inside,below,room),
    %Equal.sym(Nat,Nat.add(owning,1n+rest),1n+last,count_split(owning,last,owner_inside(column,ow,index,positive,below))) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    passes_reach(column,ow,cap,positive,inside,room))

# ---- The plane offsets of the passes ----

def stride_of(+window: Packing.Stencil) -> Nat:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,output_width,inner,groups} = window
  U32.to_nat(stride)

def row_width_of(+planes: Packing.Planes) -> Nat:
  Packing.Planes{first_output_row,rows,+row_width,elements} = planes
  U32.to_nat(row_width)

def first_row_of(+planes: Packing.Planes) -> Nat:
  Packing.Planes{+first_output_row,rows,row_width,elements} = planes
  U32.to_nat(first_output_row)

# The plane offset steps stride rows a pass, less the output row it moves on.
def offset_step(+window: Packing.Stencil,+planes: Packing.Planes) -> Nat:
  Nat.sub(Nat.mul(stride_of(window),row_width_of(planes)),output_width_of(window))

def next_word(+offset: Nat,+window: Packing.Stencil,+planes: Packing.Planes,
  +ordered: {Nat.is_le(output_width_of(window),Nat.mul(stride_of(window),row_width_of(planes))) == True{} : Bool}) ->
  {Passes.next_base(U32.from_nat(offset),window,planes) == U32.from_nat(Nat.add(offset,offset_step(window,planes))) : U32}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +s = U32.to_nat(stride)
  +ow = U32.to_nat(output_width)
  +rw = U32.to_nat(row_width)
  +span = Nat.mul(s,rw)
  %Arithmetic.native_roundtrip(stride) : {((U32.from_nat(offset) + _ * row_width : U32) - output_width : U32) == U32.from_nat(Nat.add(offset,Nat.sub(span,ow))) : U32}
  %Arithmetic.native_roundtrip(row_width) : {((U32.from_nat(offset) + U32.from_nat(s) * _ : U32) - output_width : U32) == U32.from_nat(Nat.add(offset,Nat.sub(span,ow))) : U32}
  %Arithmetic.native_roundtrip(output_width) : {((U32.from_nat(offset) + U32.from_nat(s) * U32.from_nat(rw) : U32) - _ : U32) == U32.from_nat(Nat.add(offset,Nat.sub(span,ow))) : U32}
  Equal.trans(U32,((U32.from_nat(offset) + U32.from_nat(s) * U32.from_nat(rw) : U32) - U32.from_nat(ow) : U32),((U32.from_nat(offset) + U32.from_nat(span) : U32) - U32.from_nat(ow) : U32),
    U32.from_nat(Nat.add(offset,Nat.sub(span,ow))),
    Equal.cong(U32,U32,word => ((U32.from_nat(offset) + word : U32) - U32.from_nat(ow) : U32),(U32.from_nat(s) * U32.from_nat(rw) : U32),U32.from_nat(span),
      Multiplication.native_encoded_multiply(s,rw)),
    Equal.trans(U32,((U32.from_nat(offset) + U32.from_nat(span) : U32) - U32.from_nat(ow) : U32),(U32.from_nat(Nat.add(offset,span)) - U32.from_nat(ow) : U32),
      U32.from_nat(Nat.add(offset,Nat.sub(span,ow))),
      Equal.cong(U32,U32,word => (word - U32.from_nat(ow) : U32),(U32.from_nat(offset) + U32.from_nat(span) : U32),U32.from_nat(Nat.add(offset,span)),WordArithmetic.native_add(offset,span)),
      Equal.trans(U32,(U32.from_nat(Nat.add(offset,span)) - U32.from_nat(ow) : U32),U32.from_nat(Nat.sub(Nat.add(offset,span),ow)),U32.from_nat(Nat.add(offset,Nat.sub(span,ow))),
        Subtraction.encoded_subtract(Nat.add(offset,span),ow,Order.transitive(ow,span,Nat.add(offset,span),ordered,Order.right_le(offset,span))),
        Equal.cong(Nat,U32,U32.from_nat,Nat.sub(Nat.add(offset,span),ow),Nat.add(offset,Nat.sub(span,ow)),Order.subtract_from_sum(offset,span,ow,ordered)))))

# The plane offset count passes on.
def base_closed(count: Nat,+offset: Nat,+window: Packing.Stencil,+planes: Packing.Planes,
  +ordered: {Nat.is_le(output_width_of(window),Nat.mul(stride_of(window),row_width_of(planes))) == True{} : Bool}) ->
  {base_after(count,U32.from_nat(offset),window,planes) == U32.from_nat(Nat.add(offset,Nat.mul(count,offset_step(window,planes)))) : U32}:
  match count:
    case 0n: Equal.cong(Nat,U32,U32.from_nat,offset,Nat.add(offset,0n),Equal.sym(Nat,Nat.add(offset,0n),offset,Addition.zero_right(offset)))
    case 1n+ +rest:
      +step = offset_step(window,planes)
      %Equal.sym(U32,Passes.next_base(U32.from_nat(offset),window,planes),U32.from_nat(Nat.add(offset,step)),next_word(offset,window,planes,ordered)) :
        {base_after(rest,_,window,planes) == U32.from_nat(Nat.add(offset,Nat.add(step,Nat.mul(rest,step)))) : U32}
      Equal.trans(U32,base_after(rest,U32.from_nat(Nat.add(offset,step)),window,planes),U32.from_nat(Nat.add(Nat.add(offset,step),Nat.mul(rest,step))),
        U32.from_nat(Nat.add(offset,Nat.add(step,Nat.mul(rest,step)))),base_closed(rest,Nat.add(offset,step),window,planes,ordered),
        Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.add(offset,step),Nat.mul(rest,step)),Nat.add(offset,Nat.add(step,Nat.mul(rest,step))),
          Addition.associative(offset,step,Nat.mul(rest,step))))

# The plane offset of the first pass of a unit at a position: its output row
# past the planes' first, stride plane rows each, its column and the kernel
# column's offset in its phase.
def first_offset(+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+kernel_column: Nat) -> Nat:
  +ow = output_width_of(window)
  +s = stride_of(window)
  Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.sub(Nat.div(U32.to_nat(position),ow),first_row_of(planes)),s),row_width_of(planes)),Nat.mod(U32.to_nat(position),ow)),Nat.div(kernel_column,s))

def first_word(+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+kernel_column: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,output_width_of(window)) == True{} : Bool},+divisor_bound: {Nat.is_le(Nat.double(output_width_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,stride_of(window)) == True{} : Bool},+stride_bound: {Nat.is_le(Nat.double(stride_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +column_room: {Nat.is_le(kernel_column,U32.to_nat(cap)) == True{} : Bool},
  +below_first: {Nat.is_le(first_row_of(planes),Nat.div(U32.to_nat(position),output_width_of(window))) == True{} : Bool}) ->
  {Model.first_base(window,planes,position,U32.from_nat(kernel_column)) == U32.from_nat(first_offset(window,planes,position,kernel_column)) : U32}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +ow = U32.to_nat(output_width)
  +s = U32.to_nat(stride)
  +rw = U32.to_nat(row_width)
  +first = U32.to_nat(first_output_row)
  +row = Nat.div(U32.to_nat(position),ow)
  +column = Nat.mod(U32.to_nat(position),ow)
  +rows_past = Nat.sub(row,first)
  +scaled = Nat.mul(rows_past,s)
  +spanned = Nat.mul(scaled,rw)
  +shift = Nat.div(kernel_column,s)
  +goal = U32.from_nat(Nat.add(Nat.add(spanned,column),shift))
  +s_room = Order.transitive(s,Nat.double(s),U32.to_nat(cap),Order.le_double(s),stride_bound)
  %Arithmetic.from_nat_exact((position / output_width : U32),row,WordDivision.divide(position,output_width,cap,cap_ok,divisor_bound,positive)) :
    {((_ - first_output_row) * stride * row_width + position % output_width + U32.from_nat(kernel_column) / stride : U32) == goal : U32}
  %Arithmetic.from_nat_exact((position % output_width : U32),column,WordDivision.modulo(position,output_width,cap,cap_ok,divisor_bound,positive)) :
    {((U32.from_nat(row) - first_output_row) * stride * row_width + _ + U32.from_nat(kernel_column) / stride : U32) == goal : U32}
  %Arithmetic.native_roundtrip(first_output_row) : {((U32.from_nat(row) - _) * stride * row_width + U32.from_nat(column) + U32.from_nat(kernel_column) / stride : U32) == goal : U32}
  %Arithmetic.native_roundtrip(stride) : {((U32.from_nat(row) - U32.from_nat(first)) * _ * row_width + U32.from_nat(column) + U32.from_nat(kernel_column) / _ : U32) == goal : U32}
  %Arithmetic.native_roundtrip(row_width) : {((U32.from_nat(row) - U32.from_nat(first)) * U32.from_nat(s) * _ + U32.from_nat(column) + U32.from_nat(kernel_column) / U32.from_nat(s) : U32) == goal : U32}
  %Equal.sym(U32,(U32.from_nat(row) - U32.from_nat(first) : U32),U32.from_nat(rows_past),Subtraction.encoded_subtract(row,first,below_first)) :
    {(_ * U32.from_nat(s) * U32.from_nat(rw) + U32.from_nat(column) + U32.from_nat(kernel_column) / U32.from_nat(s) : U32) == goal : U32}
  %Equal.sym(U32,(U32.from_nat(rows_past) * U32.from_nat(s) : U32),U32.from_nat(scaled),Multiplication.native_encoded_multiply(rows_past,s)) :
    {(_ * U32.from_nat(rw) + U32.from_nat(column) + U32.from_nat(kernel_column) / U32.from_nat(s) : U32) == goal : U32}
  %Equal.sym(U32,(U32.from_nat(scaled) * U32.from_nat(rw) : U32),U32.from_nat(spanned),Multiplication.native_encoded_multiply(scaled,rw)) :
    {(_ + U32.from_nat(column) + U32.from_nat(kernel_column) / U32.from_nat(s) : U32) == goal : U32}
  %Equal.sym(U32,(U32.from_nat(spanned) + U32.from_nat(column) : U32),U32.from_nat(Nat.add(spanned,column)),WordArithmetic.native_add(spanned,column)) :
    {(_ + U32.from_nat(kernel_column) / U32.from_nat(s) : U32) == goal : U32}
  %Equal.sym(U32,(U32.from_nat(kernel_column) / U32.from_nat(s) : U32),U32.from_nat(shift),
    WordDivision.encoded_divide(kernel_column,s,cap,cap_ok,column_room,s_room,stride_bound,stride_positive)) :
    {(U32.from_nat(Nat.add(spanned,column)) + _ : U32) == goal : U32}
  WordArithmetic.native_add(Nat.add(spanned,column),shift)

# ---- The pair a pass reads ----

def right_swap(+first: Nat,+second: Nat,+third: Nat) -> {Nat.add(Nat.add(first,second),third) == Nat.add(Nat.add(first,third),second) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(first,second),third),Nat.add(first,Nat.add(second,third)),Nat.add(Nat.add(first,third),second),Addition.associative(first,second,third),
    Equal.trans(Nat,Nat.add(first,Nat.add(second,third)),Nat.add(first,Nat.add(third,second)),Nat.add(Nat.add(first,third),second),
      Equal.cong(Nat,Nat,value => Nat.add(first,value),Nat.add(second,third),Nat.add(third,second),Order.commutative(second,third)),
      Equal.sym(Nat,Nat.add(Nat.add(first,third),second),Nat.add(first,Nat.add(third,second)),Addition.associative(first,third,second))))

# Lane index of the tile a pass writes at plane offset offset is lane
# offset % 8 + index of the pair it reads.
def lane_value(+pairs: Tree.Tree<Tile.Tile16>,+lane: U32,+offset: Nat,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes,+target: Planes.Spot,
  +index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+below: {Nat.is_lt(index,8n) == True{} : Bool},
  +offset_room: {Nat.is_le(offset,U32.to_nat(cap)) == True{} : Bool},+sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool}) ->
  {TileLaws.select_lane(index,pass_shifted(pairs,lane,U32.from_nat(offset),kernel_column,unit,window,planes,target)) ==
    pair_lane(Nat.add(Nat.mod(offset,8n),index),Tree.read(Tile.Tile16,pairs,Passes.entry_pair(Passes.pass_at(lane,U32.from_nat(offset),kernel_column,unit,window,planes),target))) :
    F32}:
  +pair = Tree.read(Tile.Tile16,pairs,Passes.entry_pair(Passes.pass_at(lane,U32.from_nat(offset),kernel_column,unit,window,planes),target))
  +shift = Nat.mod(offset,8n)
  %Equal.sym(U32,(U32.from_nat(offset) % U32.from_nat(8n) : U32),U32.from_nat(shift),
    WordDivision.encoded_modulo(offset,8n,cap,cap_ok,offset_room,Order.transitive(8n,16n,U32.to_nat(cap),{==},sixteen),sixteen,{==})) :
    {TileLaws.select_lane(index,Passes.shifted_at(U32.is_lt(_,4),U32.is_lt((_ % 4 : U32),2),U32.is_eq((_ % 2 : U32),1),pair)) == pair_lane(Nat.add(shift,index),pair) : F32}
  shifted_lane(pair,shift,index,DivisionOrder.remainder_small(offset,8n,{==}),below)

# The pair number a pass reads for an entry, as one natural number.
def pair_index_nat(+lane: U32,+offset: Nat,+kernel_column: Nat,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes,+spot: Planes.Spot,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+column_room: {Nat.is_le(kernel_column,U32.to_nat(cap)) == True{} : Bool},
  +offset_room: {Nat.is_le(offset,U32.to_nat(cap)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride_of(window)) == True{} : Bool},
  +stride_bound: {Nat.is_le(Nat.double(stride_of(window)),U32.to_nat(cap)) == True{} : Bool},+sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},
  +origin_room: {Nat.is_le(Nat.mul(Nat.add(Nat.mul(Nat.mod(kernel_column,stride_of(window)),Planes.elements_of(planes)),Nat.div(offset,8n)),Planes.channel_count(window)),
    U32.to_nat(cap)) == True{} : Bool},
  +step_room: {Nat.is_le(Nat.mul(Planes.tiles_of(planes),Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Passes.pair_number(Passes.pass_at(lane,U32.from_nat(offset),U32.from_nat(kernel_column),unit,window,planes),spot) ==
    Nat.add(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.mod(kernel_column,stride_of(window)),Planes.elements_of(planes)),Nat.div(offset,8n)),Planes.channel_count(window)),Planes.row_of(spot)),
      Nat.mul(Planes.column_of(spot),Nat.mul(Planes.tiles_of(planes),Planes.channel_count(window)))) : Nat}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +channels = (inner / (tap_rows * tap_columns) : U32)
  +count = U32.to_nat(channels)
  +s = U32.to_nat(stride)
  +n = U32.to_nat(elements)
  +tiles = (row_width / 8 : U32)
  +t = U32.to_nat(tiles)
  +phase = Nat.mod(kernel_column,s)
  +quotient = Nat.div(offset,8n)
  +head = Nat.mul(phase,n)
  +origin = Nat.add(head,quotient)
  +value = Nat.mul(origin,count)
  +s_room = Order.transitive(s,Nat.double(s),U32.to_nat(cap),Order.le_double(s),stride_bound)
  +origin_word = {
    %Arithmetic.native_roundtrip(stride) : {(((U32.from_nat(kernel_column) % _) * elements + U32.from_nat(offset) / 8) * channels : U32) == U32.from_nat(value) : U32}
    %Arithmetic.native_roundtrip(elements) : {(((U32.from_nat(kernel_column) % U32.from_nat(s)) * _ + U32.from_nat(offset) / 8) * channels : U32) == U32.from_nat(value) : U32}
    %Arithmetic.native_roundtrip(channels) : {(((U32.from_nat(kernel_column) % U32.from_nat(s)) * U32.from_nat(n) + U32.from_nat(offset) / 8) * _ : U32) == U32.from_nat(value) : U32}
    %Equal.sym(U32,(U32.from_nat(kernel_column) % U32.from_nat(s) : U32),U32.from_nat(phase),
      WordDivision.encoded_modulo(kernel_column,s,cap,cap_ok,column_room,s_room,stride_bound,stride_positive)) :
      {((_ * U32.from_nat(n) + U32.from_nat(offset) / 8) * U32.from_nat(count) : U32) == U32.from_nat(value) : U32}
    %Equal.sym(U32,(U32.from_nat(offset) / U32.from_nat(8n) : U32),U32.from_nat(quotient),
      WordDivision.encoded_divide(offset,8n,cap,cap_ok,offset_room,Order.transitive(8n,16n,U32.to_nat(cap),{==},sixteen),sixteen,{==})) :
      {((U32.from_nat(phase) * U32.from_nat(n) + _) * U32.from_nat(count) : U32) == U32.from_nat(value) : U32}
    %Equal.sym(U32,(U32.from_nat(phase) * U32.from_nat(n) : U32),U32.from_nat(head),Multiplication.native_encoded_multiply(phase,n)) :
      {((_ + U32.from_nat(quotient)) * U32.from_nat(count) : U32) == U32.from_nat(value) : U32}
    %Equal.sym(U32,(U32.from_nat(head) + U32.from_nat(quotient) : U32),U32.from_nat(origin),WordArithmetic.native_add(head,quotient)) :
      {(_ * U32.from_nat(count) : U32) == U32.from_nat(value) : U32}
    Multiplication.native_encoded_multiply(origin,count) :
      {(((U32.from_nat(kernel_column) % stride) * elements + U32.from_nat(offset) / 8) * channels : U32) == U32.from_nat(value) : U32}}
  +origin_exact = Equal.trans(Nat,U32.to_nat((((U32.from_nat(kernel_column) % stride) * elements + U32.from_nat(offset) / 8) * channels : U32)),
    U32.to_nat(U32.from_nat(value)),value,
    Equal.cong(U32,Nat,U32.to_nat,(((U32.from_nat(kernel_column) % stride) * elements + U32.from_nat(offset) / 8) * channels : U32),U32.from_nat(value),origin_word),
    Bounds.roundtrip(value,cap,cap_ok,origin_room))
  +step_word = {
    %Arithmetic.native_roundtrip(tiles) : {(_ * channels : U32) == U32.from_nat(Nat.mul(t,count)) : U32}
    %Arithmetic.native_roundtrip(channels) : {(U32.from_nat(t) * _ : U32) == U32.from_nat(Nat.mul(t,count)) : U32}
    Multiplication.native_encoded_multiply(t,count) : {(tiles * channels : U32) == U32.from_nat(Nat.mul(t,count)) : U32}}
  +step_exact = Equal.trans(Nat,U32.to_nat((tiles * channels : U32)),U32.to_nat(U32.from_nat(Nat.mul(t,count))),Nat.mul(t,count),
    Equal.cong(U32,Nat,U32.to_nat,(tiles * channels : U32),U32.from_nat(Nat.mul(t,count)),step_word),Bounds.roundtrip(Nat.mul(t,count),cap,cap_ok,step_room))
  %Equal.sym(Nat,U32.to_nat((((U32.from_nat(kernel_column) % stride) * elements + U32.from_nat(offset) / 8) * channels : U32)),value,origin_exact) :
    {Nat.add(Nat.add(_,Planes.row_of(spot)),Nat.mul(Planes.column_of(spot),U32.to_nat((tiles * channels : U32)))) ==
      Nat.add(Nat.add(value,Planes.row_of(spot)),Nat.mul(Planes.column_of(spot),Nat.mul(t,count))) : Nat}
  %Equal.sym(Nat,U32.to_nat((tiles * channels : U32)),Nat.mul(t,count),step_exact) :
    {Nat.add(Nat.add(value,Planes.row_of(spot)),Nat.mul(Planes.column_of(spot),_)) == Nat.add(Nat.add(value,Planes.row_of(spot)),Nat.mul(Planes.column_of(spot),Nat.mul(t,count))) : Nat}
  {==}

# The pair number regrouped as plane, element and channel.
def pair_sum(+head: Nat,+quotient: Nat,+channel: Nat,+kernel_row: Nat,+tiles: Nat,+channels: Nat) ->
  {Nat.add(Nat.add(Nat.mul(Nat.add(head,quotient),channels),channel),Nat.mul(kernel_row,Nat.mul(tiles,channels))) ==
    Nat.add(Nat.mul(Nat.add(head,Nat.add(quotient,Nat.mul(kernel_row,tiles))),channels),channel) : Nat}:
  +front = Nat.mul(Nat.add(head,quotient),channels)
  +rows = Nat.mul(kernel_row,tiles)
  Equal.trans(Nat,Nat.add(Nat.add(front,channel),Nat.mul(kernel_row,Nat.mul(tiles,channels))),Nat.add(Nat.add(front,Nat.mul(kernel_row,Nat.mul(tiles,channels))),channel),
    Nat.add(Nat.mul(Nat.add(head,Nat.add(quotient,rows)),channels),channel),right_swap(front,channel,Nat.mul(kernel_row,Nat.mul(tiles,channels))),
    Equal.cong(Nat,Nat,value => Nat.add(value,channel),Nat.add(front,Nat.mul(kernel_row,Nat.mul(tiles,channels))),Nat.mul(Nat.add(head,Nat.add(quotient,rows)),channels),
      Equal.trans(Nat,Nat.add(front,Nat.mul(kernel_row,Nat.mul(tiles,channels))),Nat.add(front,Nat.mul(rows,channels)),Nat.mul(Nat.add(head,Nat.add(quotient,rows)),channels),
        Equal.cong(Nat,Nat,value => Nat.add(front,value),Nat.mul(kernel_row,Nat.mul(tiles,channels)),Nat.mul(rows,channels),
          Equal.sym(Nat,Nat.mul(rows,channels),Nat.mul(kernel_row,Nat.mul(tiles,channels)),RowMajor.assoc(kernel_row,tiles,channels))),
        Equal.trans(Nat,Nat.add(front,Nat.mul(rows,channels)),Nat.mul(Nat.add(Nat.add(head,quotient),rows),channels),Nat.mul(Nat.add(head,Nat.add(quotient,rows)),channels),
          Equal.sym(Nat,Nat.mul(Nat.add(Nat.add(head,quotient),rows),channels),Nat.add(front,Nat.mul(rows,channels)),RowMajor.distrib(Nat.add(head,quotient),rows,channels)),
          Equal.cong(Nat,Nat,value => Nat.mul(value,channels),Nat.add(Nat.add(head,quotient),rows),Nat.add(head,Nat.add(quotient,rows)),
            Addition.associative(head,quotient,rows))))))

# The spot of element quotient + kernel_row * tiles.
def spot_at(+quotient: Nat,+kernel_row: Nat,+tiles: Nat) -> Planes.Spot:
  Planes.Spot{Nat.add(Nat.div(quotient,tiles),kernel_row),Nat.mod(quotient,tiles)}

def element_form(+quotient: Nat,+kernel_row: Nat,+tiles: Nat,+positive: {Nat.is_lt(0n,tiles) == True{} : Bool}) ->
  {Planes.element(spot_at(quotient,kernel_row,tiles),tiles) == Nat.add(quotient,Nat.mul(kernel_row,tiles)) : Nat}:
  +major = Nat.div(quotient,tiles)
  +minor = Nat.mod(quotient,tiles)
  Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(major,kernel_row),tiles),minor),Nat.add(Nat.add(Nat.mul(major,tiles),Nat.mul(kernel_row,tiles)),minor),
    Nat.add(quotient,Nat.mul(kernel_row,tiles)),
    Equal.cong(Nat,Nat,value => Nat.add(value,minor),Nat.mul(Nat.add(major,kernel_row),tiles),Nat.add(Nat.mul(major,tiles),Nat.mul(kernel_row,tiles)),
      RowMajor.distrib(major,kernel_row,tiles)),
    Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(major,tiles),Nat.mul(kernel_row,tiles)),minor),Nat.add(Nat.add(Nat.mul(major,tiles),minor),Nat.mul(kernel_row,tiles)),
      Nat.add(quotient,Nat.mul(kernel_row,tiles)),right_swap(Nat.mul(major,tiles),Nat.mul(kernel_row,tiles),minor),
      Equal.cong(Nat,Nat,value => Nat.add(value,Nat.mul(kernel_row,tiles)),Nat.add(Nat.mul(major,tiles),minor),quotient,
        Equal.sym(Nat,quotient,Nat.add(Nat.mul(major,tiles),minor),DivisionOrder.quotient_exact(quotient,tiles,positive)))))

# The plane number of a phase and a channel.
def plane_number_of(+phase: Nat,+channel: Nat,+channels: Nat) -> Nat:
  Nat.add(Nat.mul(phase,channels),channel)

def number_phase(+phase: Nat,+channel: Nat,+channels: Nat,+inside: {Nat.is_lt(channel,channels) == True{} : Bool}) ->
  {Nat.div(plane_number_of(phase,channel,channels),channels) == phase : Nat}:
  +positive = Order.le_lt_trans(0n,channel,channels,Order.zero_le(channel),inside)
  Equal.trans(Nat,Nat.div(plane_number_of(phase,channel,channels),channels),Nat.add(phase,Nat.div(channel,channels)),phase,
    DivisionOrder.quotient_shift(channel,channels,phase,positive),
    Equal.trans(Nat,Nat.add(phase,Nat.div(channel,channels)),Nat.add(phase,0n),phase,
      Equal.cong(Nat,Nat,value => Nat.add(phase,value),Nat.div(channel,channels),0n,DivisionOrder.small_quotient(channel,channels,inside)),Addition.zero_right(phase)))

def number_channel(+phase: Nat,+channel: Nat,+channels: Nat,+inside: {Nat.is_lt(channel,channels) == True{} : Bool}) ->
  {Nat.mod(plane_number_of(phase,channel,channels),channels) == channel : Nat}:
  +positive = Order.le_lt_trans(0n,channel,channels,Order.zero_le(channel),inside)
  Equal.trans(Nat,Nat.mod(plane_number_of(phase,channel,channels),channels),Nat.mod(channel,channels),channel,
    DivisionOrder.remainder_shift(channel,channels,phase,positive),DivisionOrder.small_remainder(channel,channels,inside))

# The pair a pass reads for entry (channel, kernel row) is the pair of its
# kernel column's phase and the channel at the element of its offset, kernel
# row plane rows down.
def pair_word(+lane: U32,+offset: Nat,+kernel_column: Nat,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes,+channel: Nat,+kernel_row: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+column_room: {Nat.is_le(kernel_column,U32.to_nat(cap)) == True{} : Bool},
  +offset_room: {Nat.is_le(offset,U32.to_nat(cap)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,stride_of(window)) == True{} : Bool},
  +stride_bound: {Nat.is_le(Nat.double(stride_of(window)),U32.to_nat(cap)) == True{} : Bool},+sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},
  +origin_room: {Nat.is_le(Nat.mul(Nat.add(Nat.mul(Nat.mod(kernel_column,stride_of(window)),Planes.elements_of(planes)),Nat.div(offset,8n)),Planes.channel_count(window)),
    U32.to_nat(cap)) == True{} : Bool},
  +step_room: {Nat.is_le(Nat.mul(Planes.tiles_of(planes),Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool},
  +tiles_positive: {Nat.is_lt(0n,Planes.tiles_of(planes)) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,Planes.channel_count(window)) == True{} : Bool}) ->
  {Passes.entry_pair(Passes.pass_at(lane,U32.from_nat(offset),U32.from_nat(kernel_column),unit,window,planes),Planes.Spot{channel,kernel_row}) ==
    Planes.plane_word(Planes.plane_of(plane_number_of(Nat.mod(kernel_column,stride_of(window)),channel,Planes.channel_count(window)),Planes.channel_count(window)),
      Planes.elements_of(planes),Planes.channel_count(window),Planes.element(spot_at(Nat.div(offset,8n),kernel_row,Planes.tiles_of(planes)),Planes.tiles_of(planes))) : U32}:
  +phase = Nat.mod(kernel_column,stride_of(window))
  +channels = Planes.channel_count(window)
  +n = Planes.elements_of(planes)
  +t = Planes.tiles_of(planes)
  +quotient = Nat.div(offset,8n)
  +number = plane_number_of(phase,channel,channels)
  +sum = Nat.add(Nat.mul(Nat.add(Nat.mul(phase,n),Nat.add(quotient,Nat.mul(kernel_row,t))),channels),channel)
  Equal.cong(Nat,U32,U32.from_nat,Passes.pair_number(Passes.pass_at(lane,U32.from_nat(offset),U32.from_nat(kernel_column),unit,window,planes),Planes.Spot{channel,kernel_row}),
    Planes.radix(Nat.div(number,channels),Planes.element(spot_at(quotient,kernel_row,t),t),Nat.mod(number,channels),n,channels),
    Equal.trans(Nat,Passes.pair_number(Passes.pass_at(lane,U32.from_nat(offset),U32.from_nat(kernel_column),unit,window,planes),Planes.Spot{channel,kernel_row}),
      Nat.add(Nat.add(Nat.mul(Nat.add(Nat.mul(phase,n),quotient),channels),channel),Nat.mul(kernel_row,Nat.mul(t,channels))),
      Planes.radix(Nat.div(number,channels),Planes.element(spot_at(quotient,kernel_row,t),t),Nat.mod(number,channels),n,channels),
      pair_index_nat(lane,offset,kernel_column,unit,window,planes,Planes.Spot{channel,kernel_row},cap,cap_ok,column_room,offset_room,stride_positive,stride_bound,sixteen,
        origin_room,step_room),
      Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(Nat.add(Nat.mul(phase,n),quotient),channels),channel),Nat.mul(kernel_row,Nat.mul(t,channels))),sum,
        Planes.radix(Nat.div(number,channels),Planes.element(spot_at(quotient,kernel_row,t),t),Nat.mod(number,channels),n,channels),
        pair_sum(Nat.mul(phase,n),quotient,channel,kernel_row,t,channels),
        %Equal.sym(Nat,Nat.div(number,channels),phase,number_phase(phase,channel,channels,channel_inside)) :
          {sum == Planes.radix(_,Planes.element(spot_at(quotient,kernel_row,t),t),Nat.mod(number,channels),n,channels) : Nat}
        %Equal.sym(Nat,Nat.mod(number,channels),channel,number_channel(phase,channel,channels,channel_inside)) :
          {sum == Planes.radix(phase,Planes.element(spot_at(quotient,kernel_row,t),t),_,n,channels) : Nat}
        %Equal.sym(Nat,Planes.element(spot_at(quotient,kernel_row,t),t),Nat.add(quotient,Nat.mul(kernel_row,t)),element_form(quotient,kernel_row,t,tiles_positive)) :
          {sum == Planes.radix(phase,_,channel,n,channels) : Nat}
        {==})))

# ---- Where the owner's lane lies in the plane ----

# The plane offset of the owner of a lane, the lane added, is plane row
# (output rows past the first) * stride, at the lane's output column past the
# kernel column's offset.
def owner_offset(+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+kernel_column: Nat,+index: Nat) -> Nat:
  +ow = output_width_of(window)
  Nat.add(first_offset(window,planes,position,kernel_column),Nat.mul(owner_of(Nat.mod(U32.to_nat(position),ow),ow,index),offset_step(window,planes)))

def owner_row(+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+index: Nat) -> Nat:
  +ow = output_width_of(window)
  Nat.mul(Nat.add(Nat.sub(Nat.div(U32.to_nat(position),ow),first_row_of(planes)),owner_of(Nat.mod(U32.to_nat(position),ow),ow,index)),stride_of(window))

def owner_value(+window: Packing.Stencil,+position: U32,+kernel_column: Nat,+index: Nat) -> Nat:
  +ow = output_width_of(window)
  Nat.add(Nat.mod(Nat.add(Nat.mod(U32.to_nat(position),ow),index),ow),Nat.div(kernel_column,stride_of(window)))

def offset_split(+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+kernel_column: Nat,+index: Nat,
  +positive: {Nat.is_lt(0n,output_width_of(window)) == True{} : Bool},
  +ordered: {Nat.is_le(output_width_of(window),Nat.mul(stride_of(window),row_width_of(planes))) == True{} : Bool}) ->
  {Nat.add(owner_offset(window,planes,position,kernel_column,index),index) ==
    Nat.add(Nat.mul(owner_row(window,planes,position,index),row_width_of(planes)),owner_value(window,position,kernel_column,index)) : Nat}:
  +ow = output_width_of(window)
  +s = stride_of(window)
  +w = row_width_of(planes)
  +column = Nat.mod(U32.to_nat(position),ow)
  +rows_past = Nat.sub(Nat.div(U32.to_nat(position),ow),first_row_of(planes))
  +head = Nat.mul(Nat.mul(rows_past,s),w)
  +shift = Nat.div(kernel_column,s)
  +owning = owner_of(column,ow,index)
  +step = offset_step(window,planes)
  +moved = Nat.mul(owning,step)
  +crossed = Nat.mul(owning,ow)
  +left = Nat.mod(Nat.add(column,index),ow)
  +value = Nat.add(left,shift)
  +canon = Nat.add(head,Nat.add(crossed,Nat.add(moved,value)))
  +lane_split = DivisionOrder.quotient_exact(Nat.add(column,index),ow,positive)
  +inner = Equal.trans(Nat,Nat.add(Nat.add(Nat.add(crossed,left),shift),moved),Nat.add(Nat.add(crossed,value),moved),Nat.add(crossed,Nat.add(moved,value)),
    Equal.cong(Nat,Nat,part => Nat.add(part,moved),Nat.add(Nat.add(crossed,left),shift),Nat.add(crossed,value),Addition.associative(crossed,left,shift)),
    Equal.trans(Nat,Nat.add(Nat.add(crossed,value),moved),Nat.add(crossed,Nat.add(value,moved)),Nat.add(crossed,Nat.add(moved,value)),Addition.associative(crossed,value,moved),
      Equal.cong(Nat,Nat,part => Nat.add(crossed,part),Nat.add(value,moved),Nat.add(moved,value),Order.commutative(value,moved))))
  +left_canon = Equal.trans(Nat,Nat.add(Nat.add(Nat.add(Nat.add(head,column),shift),moved),index),Nat.add(Nat.add(Nat.add(Nat.add(head,column),shift),index),moved),canon,
    right_swap(Nat.add(Nat.add(head,column),shift),moved,index),
    Equal.trans(Nat,Nat.add(Nat.add(Nat.add(Nat.add(head,column),shift),index),moved),Nat.add(Nat.add(Nat.add(head,Nat.add(crossed,left)),shift),moved),canon,
      Equal.cong(Nat,Nat,part => Nat.add(part,moved),Nat.add(Nat.add(Nat.add(head,column),shift),index),Nat.add(Nat.add(head,Nat.add(crossed,left)),shift),
        Equal.trans(Nat,Nat.add(Nat.add(Nat.add(head,column),shift),index),Nat.add(Nat.add(Nat.add(head,column),index),shift),Nat.add(Nat.add(head,Nat.add(crossed,left)),shift),
          right_swap(Nat.add(head,column),shift,index),
          Equal.cong(Nat,Nat,part => Nat.add(part,shift),Nat.add(Nat.add(head,column),index),Nat.add(head,Nat.add(crossed,left)),
            Equal.trans(Nat,Nat.add(Nat.add(head,column),index),Nat.add(head,Nat.add(column,index)),Nat.add(head,Nat.add(crossed,left)),
              Addition.associative(head,column,index),
              Equal.cong(Nat,Nat,part => Nat.add(head,part),Nat.add(column,index),Nat.add(crossed,left),lane_split))))),
      Equal.trans(Nat,Nat.add(Nat.add(Nat.add(head,Nat.add(crossed,left)),shift),moved),Nat.add(Nat.add(head,Nat.add(Nat.add(crossed,left),shift)),moved),canon,
        Equal.cong(Nat,Nat,part => Nat.add(part,moved),Nat.add(Nat.add(head,Nat.add(crossed,left)),shift),Nat.add(head,Nat.add(Nat.add(crossed,left),shift)),
          Addition.associative(head,Nat.add(crossed,left),shift)),
        Equal.trans(Nat,Nat.add(Nat.add(head,Nat.add(Nat.add(crossed,left),shift)),moved),Nat.add(head,Nat.add(Nat.add(Nat.add(crossed,left),shift),moved)),canon,
          Addition.associative(head,Nat.add(Nat.add(crossed,left),shift),moved),
          Equal.cong(Nat,Nat,part => Nat.add(head,part),Nat.add(Nat.add(Nat.add(crossed,left),shift),moved),Nat.add(crossed,Nat.add(moved,value)),inner)))))
  +span = Nat.mul(s,w)
  +spread = Equal.trans(Nat,Nat.mul(Nat.mul(owning,s),w),Nat.mul(owning,span),Nat.add(crossed,moved),RowMajor.assoc(owning,s,w),
    Equal.trans(Nat,Nat.mul(owning,span),Nat.mul(owning,Nat.add(ow,step)),Nat.add(crossed,moved),
      Equal.cong(Nat,Nat,part => Nat.mul(owning,part),span,Nat.add(ow,step),Equal.sym(Nat,Nat.add(ow,step),span,Order.restore_shift(ow,span,ordered))),
      RowMajor.right_distrib(owning,ow,step)))
  +rows = Equal.trans(Nat,Nat.mul(Nat.mul(Nat.add(rows_past,owning),s),w),Nat.mul(Nat.add(Nat.mul(rows_past,s),Nat.mul(owning,s)),w),Nat.add(head,Nat.add(crossed,moved)),
    Equal.cong(Nat,Nat,part => Nat.mul(part,w),Nat.mul(Nat.add(rows_past,owning),s),Nat.add(Nat.mul(rows_past,s),Nat.mul(owning,s)),RowMajor.distrib(rows_past,owning,s)),
    Equal.trans(Nat,Nat.mul(Nat.add(Nat.mul(rows_past,s),Nat.mul(owning,s)),w),Nat.add(head,Nat.mul(Nat.mul(owning,s),w)),Nat.add(head,Nat.add(crossed,moved)),
      RowMajor.distrib(Nat.mul(rows_past,s),Nat.mul(owning,s),w),
      Equal.cong(Nat,Nat,part => Nat.add(head,part),Nat.mul(Nat.mul(owning,s),w),Nat.add(crossed,moved),spread)))
  +right_canon = Equal.trans(Nat,Nat.add(Nat.mul(Nat.mul(Nat.add(rows_past,owning),s),w),value),Nat.add(Nat.add(head,Nat.add(crossed,moved)),value),canon,
    Equal.cong(Nat,Nat,part => Nat.add(part,value),Nat.mul(Nat.mul(Nat.add(rows_past,owning),s),w),Nat.add(head,Nat.add(crossed,moved)),rows),
    Equal.trans(Nat,Nat.add(Nat.add(head,Nat.add(crossed,moved)),value),Nat.add(head,Nat.add(Nat.add(crossed,moved),value)),canon,
      Addition.associative(head,Nat.add(crossed,moved),value),
      Equal.cong(Nat,Nat,part => Nat.add(head,part),Nat.add(Nat.add(crossed,moved),value),Nat.add(crossed,Nat.add(moved,value)),Addition.associative(crossed,moved,value))))
  Equal.trans(Nat,Nat.add(Nat.add(Nat.add(Nat.add(head,column),shift),moved),index),canon,Nat.add(Nat.mul(Nat.mul(Nat.add(rows_past,owning),s),w),value),left_canon,
    Equal.sym(Nat,Nat.add(Nat.mul(Nat.mul(Nat.add(rows_past,owning),s),w),value),canon,right_canon))

# Element * 8 + lane splits the same way, kernel_row plane rows further.
def lanes_linear(+offset: Nat,+index: Nat,+kernel_row: Nat,+tiles: Nat) ->
  {Nat.add(Nat.mul(Nat.add(Nat.div(offset,8n),Nat.mul(kernel_row,tiles)),8n),Nat.add(Nat.mod(offset,8n),index)) ==
    Nat.add(Nat.add(offset,index),Nat.mul(kernel_row,Nat.mul(tiles,8n))) : Nat}:
  +whole = Nat.mul(Nat.div(offset,8n),8n)
  +part = Nat.mod(offset,8n)
  +down = Nat.mul(kernel_row,Nat.mul(tiles,8n))
  Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.div(offset,8n),Nat.mul(kernel_row,tiles)),8n),Nat.add(part,index)),Nat.add(Nat.add(whole,down),Nat.add(part,index)),
    Nat.add(Nat.add(offset,index),down),
    Equal.cong(Nat,Nat,value => Nat.add(value,Nat.add(part,index)),Nat.mul(Nat.add(Nat.div(offset,8n),Nat.mul(kernel_row,tiles)),8n),Nat.add(whole,down),
      Equal.trans(Nat,Nat.mul(Nat.add(Nat.div(offset,8n),Nat.mul(kernel_row,tiles)),8n),Nat.add(whole,Nat.mul(Nat.mul(kernel_row,tiles),8n)),Nat.add(whole,down),
        RowMajor.distrib(Nat.div(offset,8n),Nat.mul(kernel_row,tiles),8n),
        Equal.cong(Nat,Nat,value => Nat.add(whole,value),Nat.mul(Nat.mul(kernel_row,tiles),8n),down,RowMajor.assoc(kernel_row,tiles,8n)))),
    Equal.trans(Nat,Nat.add(Nat.add(whole,down),Nat.add(part,index)),Nat.add(Nat.add(whole,Nat.add(part,index)),down),Nat.add(Nat.add(offset,index),down),
      right_swap(whole,down,Nat.add(part,index)),
      Equal.cong(Nat,Nat,value => Nat.add(value,down),Nat.add(whole,Nat.add(part,index)),Nat.add(offset,index),
        Equal.trans(Nat,Nat.add(whole,Nat.add(part,index)),Nat.add(Nat.add(whole,part),index),Nat.add(offset,index),
          Equal.sym(Nat,Nat.add(Nat.add(whole,part),index),Nat.add(whole,Nat.add(part,index)),Addition.associative(whole,part,index)),
          Equal.cong(Nat,Nat,value => Nat.add(value,index),Nat.add(whole,part),offset,
            Equal.sym(Nat,offset,Nat.add(whole,part),DivisionOrder.quotient_exact(offset,8n,{==})))))))

# kernel_row plane rows further down.
def rows_down(+row: Nat,+value: Nat,+kernel_row: Nat,+width: Nat) ->
  {Nat.add(Nat.add(Nat.mul(row,width),value),Nat.mul(kernel_row,width)) == Planes.outer(Nat.add(row,kernel_row),value,width) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(row,width),value),Nat.mul(kernel_row,width)),Nat.add(Nat.add(Nat.mul(row,width),Nat.mul(kernel_row,width)),value),
    Planes.outer(Nat.add(row,kernel_row),value,width),right_swap(Nat.mul(row,width),value,Nat.mul(kernel_row,width)),
    Equal.cong(Nat,Nat,part => Nat.add(part,value),Nat.add(Nat.mul(row,width),Nat.mul(kernel_row,width)),Nat.mul(Nat.add(row,kernel_row),width),
      Equal.sym(Nat,Nat.mul(Nat.add(row,kernel_row),width),Nat.add(Nat.mul(row,width),Nat.mul(kernel_row,width)),RowMajor.distrib(row,kernel_row,width))))

# ---- The pixel of a lane ----

# A sample of the input cells: the lane the bits name of the cell at pixel
# (row, column) from a channel base, +0 outside the input.
def cell_at(+input: Tree.Tree<Tile.Tile8>,+lanes: Lanes.Bits,+channel_base: U32,+row: U32,+column: U32,+height: U32,+width: U32,+groups: U32) -> F32:
  PackingSamples.masked(Lanes.lane_at(lanes,Tree.read(Tile.Tile8,input,(channel_base + (row * width + column) * groups : U32))),U32.is_lt(row,height) && U32.is_lt(column,width))

# The pixel lane index of the unit at position reads for entry (channel,
# kernel row) of a kernel column: output position position + index, its row
# and column stride apart, the taps past them, the lane of the channel in the
# cell of its group.
def window_cell(+input: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+position: U32,+channel: Nat,+kernel_row: Nat,+kernel_column: Nat,+index: Nat) -> F32:
  Packing.Stencil{+base,+height,+width,tap_rows,tap_columns,+stride,+row_origin,+column_origin,+output_width,inner,+groups} = window
  +at = (position + U32.from_nat(index) : U32)
  cell_at(input,Lanes.bits(U32.from_nat(channel)),(base + U32.from_nat(channel) / 8 : U32),(at / output_width * stride + U32.from_nat(kernel_row) + row_origin : U32),
    (at % output_width * stride + U32.from_nat(kernel_column) + column_origin : U32),height,width,groups)

def word_right_swap(+first: U32,+second: U32,+third: U32) -> {((first + second : U32) + third : U32) == ((first + third : U32) + second : U32) : U32}:
  Equal.trans(U32,((first + second : U32) + third : U32),(first + (second + third : U32) : U32),((first + third : U32) + second : U32),Arithmetic.add_associative(first,second,third),
    Equal.trans(U32,(first + (second + third : U32) : U32),(first + (third + second : U32) : U32),((first + third : U32) + second : U32),
      Equal.cong(U32,U32,part => (first + part : U32),(second + third : U32),(third + second : U32),Arithmetic.add_commutative(second,third)),
      Equal.sym(U32,((first + third : U32) + second : U32),(first + (third + second : U32) : U32),Arithmetic.add_associative(first,third,second))))

# The output position index lanes on, below the cap.
def lane_position(+position: U32,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),index),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((position + U32.from_nat(index) : U32)) == Nat.add(U32.to_nat(position),index) : Nat}:
  %Arithmetic.native_roundtrip(position) : {U32.to_nat((_ + U32.from_nat(index) : U32)) == Nat.add(U32.to_nat(position),index) : Nat}
  %Equal.sym(U32,(U32.from_nat(U32.to_nat(position)) + U32.from_nat(index) : U32),U32.from_nat(Nat.add(U32.to_nat(position),index)),WordArithmetic.native_add(U32.to_nat(position),index)) : {U32.to_nat(_) == Nat.add(U32.to_nat(position),index) : Nat}
  Bounds.roundtrip(Nat.add(U32.to_nat(position),index),cap,cap_ok,room)

# Position + index lies past the first position's row by the owner's rows.
def lane_split(+position: Nat,+width: Nat,+index: Nat,+positive: {Nat.is_lt(0n,width) == True{} : Bool}) ->
  {Nat.add(position,index) == Nat.add(Nat.mul(Nat.div(position,width),width),Nat.add(Nat.mod(position,width),index)) : Nat}:
  Equal.trans(Nat,Nat.add(position,index),Nat.add(Nat.add(Nat.mul(Nat.div(position,width),width),Nat.mod(position,width)),index),
    Nat.add(Nat.mul(Nat.div(position,width),width),Nat.add(Nat.mod(position,width),index)),
    Equal.cong(Nat,Nat,value => Nat.add(value,index),position,Nat.add(Nat.mul(Nat.div(position,width),width),Nat.mod(position,width)),
      DivisionOrder.quotient_exact(position,width,positive)),
    Addition.associative(Nat.mul(Nat.div(position,width),width),Nat.mod(position,width),index))

# The row of a lane's pixel read from the plane is the row the spec reads.
def row_pixel(+first: U32,+stride: U32,+row_origin: U32,+output_width: U32,+position: U32,+index: Nat,+kernel_row: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(output_width)) == True{} : Bool},
  +divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(output_width)),U32.to_nat(cap)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),index),U32.to_nat(cap)) == True{} : Bool},
  +below_first: {Nat.is_le(U32.to_nat(first),Nat.div(U32.to_nat(position),U32.to_nat(output_width))) == True{} : Bool}) ->
  {((first * stride : U32) + row_origin + U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.sub(Nat.div(U32.to_nat(position),U32.to_nat(output_width)),U32.to_nat(first)),
      owner_of(Nat.mod(U32.to_nat(position),U32.to_nat(output_width)),U32.to_nat(output_width),index)),U32.to_nat(stride)),kernel_row)) : U32) ==
    ((position + U32.from_nat(index) : U32) / output_width * stride + U32.from_nat(kernel_row) + row_origin : U32) : U32}:
  +ow = U32.to_nat(output_width)
  +s = U32.to_nat(stride)
  +f = U32.to_nat(first)
  +p = U32.to_nat(position)
  +row = Nat.div(p,ow)
  +column = Nat.mod(p,ow)
  +owning = owner_of(column,ow,index)
  +reached = Nat.add(row,owning)
  +rk = Nat.add(Nat.mul(Nat.add(Nat.sub(row,f),owning),s),kernel_row)
  +lifted = Nat.add(Nat.mul(reached,s),kernel_row)
  +at = (position + U32.from_nat(index) : U32)
  +quotient_exact = Equal.trans(Nat,U32.to_nat((at / output_width : U32)),Nat.div(U32.to_nat(at),ow),reached,WordDivision.divide(at,output_width,cap,cap_ok,divisor_bound,positive),
    Equal.trans(Nat,Nat.div(U32.to_nat(at),ow),Nat.div(Nat.add(p,index),ow),reached,
      Equal.cong(Nat,Nat,value => Nat.div(value,ow),U32.to_nat(at),Nat.add(p,index),lane_position(position,index,cap,cap_ok,room)),
      Equal.trans(Nat,Nat.div(Nat.add(p,index),ow),Nat.div(Nat.add(Nat.mul(row,ow),Nat.add(column,index)),ow),reached,
        Equal.cong(Nat,Nat,value => Nat.div(value,ow),Nat.add(p,index),Nat.add(Nat.mul(row,ow),Nat.add(column,index)),lane_split(p,ow,index,positive)),
        DivisionOrder.quotient_shift(Nat.add(column,index),ow,row,positive))))
  +rows = Equal.trans(Nat,Nat.add(Nat.mul(f,s),rk),Nat.add(Nat.add(Nat.mul(f,s),Nat.mul(Nat.add(Nat.sub(row,f),owning),s)),kernel_row),lifted,
    Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(f,s),Nat.mul(Nat.add(Nat.sub(row,f),owning),s)),kernel_row),Nat.add(Nat.mul(f,s),rk),
      Addition.associative(Nat.mul(f,s),Nat.mul(Nat.add(Nat.sub(row,f),owning),s),kernel_row)),
    Equal.cong(Nat,Nat,value => Nat.add(value,kernel_row),Nat.add(Nat.mul(f,s),Nat.mul(Nat.add(Nat.sub(row,f),owning),s)),Nat.mul(reached,s),
      Equal.trans(Nat,Nat.add(Nat.mul(f,s),Nat.mul(Nat.add(Nat.sub(row,f),owning),s)),Nat.mul(Nat.add(f,Nat.add(Nat.sub(row,f),owning)),s),Nat.mul(reached,s),
        Equal.sym(Nat,Nat.mul(Nat.add(f,Nat.add(Nat.sub(row,f),owning)),s),Nat.add(Nat.mul(f,s),Nat.mul(Nat.add(Nat.sub(row,f),owning),s)),
          RowMajor.distrib(f,Nat.add(Nat.sub(row,f),owning),s)),
        Equal.cong(Nat,Nat,value => Nat.mul(value,s),Nat.add(f,Nat.add(Nat.sub(row,f),owning)),reached,
          Equal.trans(Nat,Nat.add(f,Nat.add(Nat.sub(row,f),owning)),Nat.add(Nat.add(f,Nat.sub(row,f)),owning),reached,
            Equal.sym(Nat,Nat.add(Nat.add(f,Nat.sub(row,f)),owning),Nat.add(f,Nat.add(Nat.sub(row,f),owning)),Addition.associative(f,Nat.sub(row,f),owning)),
            Equal.cong(Nat,Nat,value => Nat.add(value,owning),Nat.add(f,Nat.sub(row,f)),row,Order.restore_shift(f,row,below_first)))))))
  %Arithmetic.native_roundtrip(stride) :
    {((first * _ : U32) + row_origin + U32.from_nat(rk) : U32) == (at / output_width * _ + U32.from_nat(kernel_row) + row_origin : U32) : U32}
  %Arithmetic.native_roundtrip(first) :
    {((_ * U32.from_nat(s) : U32) + row_origin + U32.from_nat(rk) : U32) == (at / output_width * U32.from_nat(s) + U32.from_nat(kernel_row) + row_origin : U32) : U32}
  %Arithmetic.from_nat_exact((at / output_width : U32),reached,quotient_exact) :
    {((U32.from_nat(f) * U32.from_nat(s) : U32) + row_origin + U32.from_nat(rk) : U32) == (_ * U32.from_nat(s) + U32.from_nat(kernel_row) + row_origin : U32) : U32}
  Equal.trans(U32,((U32.from_nat(f) * U32.from_nat(s) : U32) + row_origin + U32.from_nat(rk) : U32),(U32.from_nat(lifted) + row_origin : U32),
    (U32.from_nat(reached) * U32.from_nat(s) + U32.from_nat(kernel_row) + row_origin : U32),
    Equal.trans(U32,((U32.from_nat(f) * U32.from_nat(s) : U32) + row_origin + U32.from_nat(rk) : U32),((U32.from_nat(Nat.mul(f,s)) + row_origin : U32) + U32.from_nat(rk) : U32),
      (U32.from_nat(lifted) + row_origin : U32),
      Equal.cong(U32,U32,word => ((word + row_origin : U32) + U32.from_nat(rk) : U32),(U32.from_nat(f) * U32.from_nat(s) : U32),U32.from_nat(Nat.mul(f,s)),
        Multiplication.native_encoded_multiply(f,s)),
      Equal.trans(U32,((U32.from_nat(Nat.mul(f,s)) + row_origin : U32) + U32.from_nat(rk) : U32),((U32.from_nat(Nat.mul(f,s)) + U32.from_nat(rk) : U32) + row_origin : U32),
        (U32.from_nat(lifted) + row_origin : U32),word_right_swap(U32.from_nat(Nat.mul(f,s)),row_origin,U32.from_nat(rk)),
        Equal.cong(U32,U32,word => (word + row_origin : U32),(U32.from_nat(Nat.mul(f,s)) + U32.from_nat(rk) : U32),U32.from_nat(lifted),
          Equal.trans(U32,(U32.from_nat(Nat.mul(f,s)) + U32.from_nat(rk) : U32),U32.from_nat(Nat.add(Nat.mul(f,s),rk)),U32.from_nat(lifted),
            WordArithmetic.native_add(Nat.mul(f,s),rk),Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.mul(f,s),rk),lifted,rows))))),
    Equal.sym(U32,(U32.from_nat(reached) * U32.from_nat(s) + U32.from_nat(kernel_row) + row_origin : U32),(U32.from_nat(lifted) + row_origin : U32),
      Equal.cong(U32,U32,word => (word + row_origin : U32),(U32.from_nat(reached) * U32.from_nat(s) + U32.from_nat(kernel_row) : U32),U32.from_nat(lifted),
        Equal.trans(U32,(U32.from_nat(reached) * U32.from_nat(s) + U32.from_nat(kernel_row) : U32),(U32.from_nat(Nat.mul(reached,s)) + U32.from_nat(kernel_row) : U32),
          U32.from_nat(lifted),
          Equal.cong(U32,U32,word => (word + U32.from_nat(kernel_row) : U32),(U32.from_nat(reached) * U32.from_nat(s) : U32),U32.from_nat(Nat.mul(reached,s)),
            Multiplication.native_encoded_multiply(reached,s)),
          WordArithmetic.native_add(Nat.mul(reached,s),kernel_row)))))

# The column of a lane's pixel read from the plane is the column the spec
# reads.
def column_pixel(+stride: U32,+column_origin: U32,+output_width: U32,+position: U32,+index: Nat,+kernel_column: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(output_width)) == True{} : Bool},
  +divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(output_width)),U32.to_nat(cap)) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(position),index),U32.to_nat(cap)) == True{} : Bool}) ->
  {((U32.from_nat(Nat.mod(kernel_column,U32.to_nat(stride))) + column_origin : U32) +
      U32.from_nat(Nat.add(Nat.mod(Nat.add(Nat.mod(U32.to_nat(position),U32.to_nat(output_width)),index),U32.to_nat(output_width)),Nat.div(kernel_column,U32.to_nat(stride)))) * stride : U32) ==
    ((position + U32.from_nat(index) : U32) % output_width * stride + U32.from_nat(kernel_column) + column_origin : U32) : U32}:
  +ow = U32.to_nat(output_width)
  +s = U32.to_nat(stride)
  +p = U32.to_nat(position)
  +column = Nat.mod(p,ow)
  +left = Nat.mod(Nat.add(column,index),ow)
  +phase = Nat.mod(kernel_column,s)
  +shift = Nat.div(kernel_column,s)
  +value = Nat.add(left,shift)
  +spread = Nat.mul(value,s)
  +target = Nat.add(Nat.mul(left,s),kernel_column)
  +at = (position + U32.from_nat(index) : U32)
  +remainder_exact = Equal.trans(Nat,U32.to_nat((at % output_width : U32)),Nat.mod(U32.to_nat(at),ow),left,WordDivision.modulo(at,output_width,cap,cap_ok,divisor_bound,positive),
    Equal.trans(Nat,Nat.mod(U32.to_nat(at),ow),Nat.mod(Nat.add(p,index),ow),left,
      Equal.cong(Nat,Nat,number => Nat.mod(number,ow),U32.to_nat(at),Nat.add(p,index),lane_position(position,index,cap,cap_ok,room)),
      Equal.trans(Nat,Nat.mod(Nat.add(p,index),ow),Nat.mod(Nat.add(Nat.mul(Nat.div(p,ow),ow),Nat.add(column,index)),ow),left,
        Equal.cong(Nat,Nat,number => Nat.mod(number,ow),Nat.add(p,index),Nat.add(Nat.mul(Nat.div(p,ow),ow),Nat.add(column,index)),lane_split(p,ow,index,positive)),
        DivisionOrder.remainder_shift(Nat.add(column,index),ow,Nat.div(p,ow),positive))))
  +columns = Equal.trans(Nat,Nat.add(phase,spread),Nat.add(phase,Nat.add(Nat.mul(left,s),Nat.mul(shift,s))),target,
    Equal.cong(Nat,Nat,number => Nat.add(phase,number),spread,Nat.add(Nat.mul(left,s),Nat.mul(shift,s)),RowMajor.distrib(left,shift,s)),
    Equal.trans(Nat,Nat.add(phase,Nat.add(Nat.mul(left,s),Nat.mul(shift,s))),Nat.add(Nat.mul(left,s),Nat.add(phase,Nat.mul(shift,s))),target,
      RowMajor.add_swap(phase,Nat.mul(left,s),Nat.mul(shift,s)),
      Equal.cong(Nat,Nat,number => Nat.add(Nat.mul(left,s),number),Nat.add(phase,Nat.mul(shift,s)),kernel_column,
        Equal.trans(Nat,Nat.add(phase,Nat.mul(shift,s)),Nat.add(Nat.mul(shift,s),phase),kernel_column,Order.commutative(phase,Nat.mul(shift,s)),
          Equal.sym(Nat,kernel_column,Nat.add(Nat.mul(shift,s),phase),DivisionOrder.quotient_exact(kernel_column,s,stride_positive))))))
  %Arithmetic.native_roundtrip(stride) :
    {((U32.from_nat(phase) + column_origin : U32) + U32.from_nat(value) * _ : U32) == (at % output_width * _ + U32.from_nat(kernel_column) + column_origin : U32) : U32}
  %Arithmetic.from_nat_exact((at % output_width : U32),left,remainder_exact) :
    {((U32.from_nat(phase) + column_origin : U32) + U32.from_nat(value) * U32.from_nat(s) : U32) == (_ * U32.from_nat(s) + U32.from_nat(kernel_column) + column_origin : U32) : U32}
  Equal.trans(U32,((U32.from_nat(phase) + column_origin : U32) + U32.from_nat(value) * U32.from_nat(s) : U32),(U32.from_nat(target) + column_origin : U32),
    (U32.from_nat(left) * U32.from_nat(s) + U32.from_nat(kernel_column) + column_origin : U32),
    Equal.trans(U32,((U32.from_nat(phase) + column_origin : U32) + U32.from_nat(value) * U32.from_nat(s) : U32),((U32.from_nat(phase) + column_origin : U32) + U32.from_nat(spread) : U32),
      (U32.from_nat(target) + column_origin : U32),
      Equal.cong(U32,U32,word => ((U32.from_nat(phase) + column_origin : U32) + word : U32),(U32.from_nat(value) * U32.from_nat(s) : U32),U32.from_nat(spread),
        Multiplication.native_encoded_multiply(value,s)),
      Equal.trans(U32,((U32.from_nat(phase) + column_origin : U32) + U32.from_nat(spread) : U32),((U32.from_nat(phase) + U32.from_nat(spread) : U32) + column_origin : U32),
        (U32.from_nat(target) + column_origin : U32),word_right_swap(U32.from_nat(phase),column_origin,U32.from_nat(spread)),
        Equal.cong(U32,U32,word => (word + column_origin : U32),(U32.from_nat(phase) + U32.from_nat(spread) : U32),U32.from_nat(target),
          Equal.trans(U32,(U32.from_nat(phase) + U32.from_nat(spread) : U32),U32.from_nat(Nat.add(phase,spread)),U32.from_nat(target),WordArithmetic.native_add(phase,spread),
            Equal.cong(Nat,U32,U32.from_nat,Nat.add(phase,spread),target,columns))))),
    Equal.sym(U32,(U32.from_nat(left) * U32.from_nat(s) + U32.from_nat(kernel_column) + column_origin : U32),(U32.from_nat(target) + column_origin : U32),
      Equal.cong(U32,U32,word => (word + column_origin : U32),(U32.from_nat(left) * U32.from_nat(s) + U32.from_nat(kernel_column) : U32),U32.from_nat(target),
        Equal.trans(U32,(U32.from_nat(left) * U32.from_nat(s) + U32.from_nat(kernel_column) : U32),(U32.from_nat(Nat.mul(left,s)) + U32.from_nat(kernel_column) : U32),
          U32.from_nat(target),
          Equal.cong(U32,U32,word => (word + U32.from_nat(kernel_column) : U32),(U32.from_nat(left) * U32.from_nat(s) : U32),U32.from_nat(Nat.mul(left,s)),
            Multiplication.native_encoded_multiply(left,s)),
          WordArithmetic.native_add(Nat.mul(left,s),kernel_column)))))

# The channel and the phase of a plane number, as words.
def channel_word(+window: Packing.Stencil,+phase: Nat,+channel: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Planes.channel_count(window)) == True{} : Bool},
  +channels_bound: {Nat.is_le(Nat.double(Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool},
  +number_room: {Nat.is_le(plane_number_of(phase,channel,Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool}) ->
  {(U32.from_nat(plane_number_of(phase,channel,Planes.channel_count(window))) % Packing.channels_of(window) : U32) == U32.from_nat(channel) : U32}:
  +channels = Planes.channel_count(window)
  +number = plane_number_of(phase,channel,channels)
  +positive = Order.le_lt_trans(0n,channel,channels,Order.zero_le(channel),channel_inside)
  +room = Order.transitive(channels,Nat.double(channels),U32.to_nat(cap),Order.le_double(channels),channels_bound)
  %Arithmetic.native_roundtrip(Packing.channels_of(window)) : {(U32.from_nat(number) % _ : U32) == U32.from_nat(channel) : U32}
  Equal.trans(U32,(U32.from_nat(number) % U32.from_nat(channels) : U32),U32.from_nat(Nat.mod(number,channels)),U32.from_nat(channel),
    WordDivision.encoded_modulo(number,channels,cap,cap_ok,number_room,room,channels_bound,positive),
    Equal.cong(Nat,U32,U32.from_nat,Nat.mod(number,channels),channel,number_channel(phase,channel,channels,channel_inside)))

def phase_word(+window: Packing.Stencil,+phase: Nat,+channel: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Planes.channel_count(window)) == True{} : Bool},
  +channels_bound: {Nat.is_le(Nat.double(Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool},
  +number_room: {Nat.is_le(plane_number_of(phase,channel,Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool}) ->
  {(U32.from_nat(plane_number_of(phase,channel,Planes.channel_count(window))) / Packing.channels_of(window) : U32) == U32.from_nat(phase) : U32}:
  +channels = Planes.channel_count(window)
  +number = plane_number_of(phase,channel,channels)
  +positive = Order.le_lt_trans(0n,channel,channels,Order.zero_le(channel),channel_inside)
  +room = Order.transitive(channels,Nat.double(channels),U32.to_nat(cap),Order.le_double(channels),channels_bound)
  %Arithmetic.native_roundtrip(Packing.channels_of(window)) : {(U32.from_nat(number) / _ : U32) == U32.from_nat(phase) : U32}
  Equal.trans(U32,(U32.from_nat(number) / U32.from_nat(channels) : U32),U32.from_nat(Nat.div(number,channels)),U32.from_nat(phase),
    WordDivision.encoded_divide(number,channels,cap,cap_ok,number_room,room,channels_bound,positive),
    Equal.cong(Nat,U32,U32.from_nat,Nat.div(number,channels),phase,number_phase(phase,channel,channels,channel_inside)))

# The plane value of the owner of a lane is the input pixel the spec's
# window sample of that lane reads.
def plane_pixel(+input: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+channel: Nat,+kernel_row: Nat,+kernel_column: Nat,+index: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+positive: {Nat.is_lt(0n,output_width_of(window)) == True{} : Bool},
  +divisor_bound: {Nat.is_le(Nat.double(output_width_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,stride_of(window)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),index),U32.to_nat(cap)) == True{} : Bool},
  +below_first: {Nat.is_le(first_row_of(planes),Nat.div(U32.to_nat(position),output_width_of(window))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Planes.channel_count(window)) == True{} : Bool},
  +channels_bound: {Nat.is_le(Nat.double(Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool},
  +number_room: {Nat.is_le(plane_number_of(Nat.mod(kernel_column,stride_of(window)),channel,Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool}) ->
  {plane_sample(Planes.context_of(window,planes,input,U32.from_nat(plane_number_of(Nat.mod(kernel_column,stride_of(window)),channel,Planes.channel_count(window)))),
      Lanes.bits((U32.from_nat(plane_number_of(Nat.mod(kernel_column,stride_of(window)),channel,Planes.channel_count(window))) % Packing.channels_of(window) : U32)),
      Nat.add(owner_row(window,planes,position,index),kernel_row),owner_value(window,position,kernel_column,index)) ==
    window_cell(input,window,position,channel,kernel_row,kernel_column,index) : F32}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +planes = {Packing.Planes{first_output_row,rows,row_width,elements} : Packing.Planes}
  +channels = (inner / (tap_rows * tap_columns) : U32)
  +phase = Nat.mod(kernel_column,U32.to_nat(stride))
  +number = U32.from_nat(plane_number_of(phase,channel,U32.to_nat(channels)))
  +row = Nat.add(owner_row(window,planes,position,index),kernel_row)
  +value = owner_value(window,position,kernel_column,index)
  +at = (position + U32.from_nat(index) : U32)
  +row_plane = ((first_output_row * stride : U32) + row_origin + U32.from_nat(row) : U32)
  +row_spec = (at / output_width * stride + U32.from_nat(kernel_row) + row_origin : U32)
  +column_plane = ((number / channels + column_origin : U32) + U32.from_nat(value) * stride : U32)
  +column_spec = (at % output_width * stride + U32.from_nat(kernel_column) + column_origin : U32)
  +picked = (number % channels : U32)
  +columns = Equal.trans(U32,column_plane,((U32.from_nat(phase) + column_origin : U32) + U32.from_nat(value) * stride : U32),column_spec,
    Equal.cong(U32,U32,word => ((word + column_origin : U32) + U32.from_nat(value) * stride : U32),(number / channels : U32),U32.from_nat(phase),
      phase_word(window,phase,channel,cap,cap_ok,channel_inside,channels_bound,number_room)),
    column_pixel(stride,column_origin,output_width,position,index,kernel_column,cap,cap_ok,positive,divisor_bound,stride_positive,room))
  +rows_equal = row_pixel(first_output_row,stride,row_origin,output_width,position,index,kernel_row,cap,cap_ok,positive,divisor_bound,room,below_first)
  +picked_equal = channel_word(window,phase,channel,cap,cap_ok,channel_inside,channels_bound,number_room)
  Equal.trans(F32,cell_at(input,Lanes.bits(picked),(base + picked / 8 : U32),row_plane,column_plane,height,width,groups),
    cell_at(input,Lanes.bits(U32.from_nat(channel)),(base + U32.from_nat(channel) / 8 : U32),row_plane,column_plane,height,width,groups),
    window_cell(input,window,position,channel,kernel_row,kernel_column,index),
    Equal.cong(U32,F32,word => cell_at(input,Lanes.bits(word),(base + word / 8 : U32),row_plane,column_plane,height,width,groups),picked,U32.from_nat(channel),picked_equal),
    Equal.trans(F32,cell_at(input,Lanes.bits(U32.from_nat(channel)),(base + U32.from_nat(channel) / 8 : U32),row_plane,column_plane,height,width,groups),
      cell_at(input,Lanes.bits(U32.from_nat(channel)),(base + U32.from_nat(channel) / 8 : U32),row_spec,column_plane,height,width,groups),
      window_cell(input,window,position,channel,kernel_row,kernel_column,index),
      Equal.cong(U32,F32,word => cell_at(input,Lanes.bits(U32.from_nat(channel)),(base + U32.from_nat(channel) / 8 : U32),word,column_plane,height,width,groups),row_plane,row_spec,
        rows_equal),
      Equal.cong(U32,F32,word => cell_at(input,Lanes.bits(U32.from_nat(channel)),(base + U32.from_nat(channel) / 8 : U32),row_spec,word,height,width,groups),column_plane,
        column_spec,columns)))

# ---- A lane of a kernel column's tile ----

def geometry_cap(+window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+cap: U32,+geometry: Planes.Geometry<window,planes,depth,cap>) ->
  {U32.is_le(cap,2147483648) == True{} : Bool}:
  Planes.Geometry{+room,tiles_bound,tiles_positive,divisor_bound,planes_bound,planes_exact} = geometry
  Planes.Room{+cap_ok,limit,fits,capped,elements_positive,channels_positive} = room
  cap_ok

def geometry_fits(+window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+cap: U32,+geometry: Planes.Geometry<window,planes,depth,cap>) ->
  {Nat.is_le(Nat.mul(Nat.mul(Planes.phases_of(window),Planes.elements_of(planes)),Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool}:
  Planes.Geometry{+room,tiles_bound,tiles_positive,divisor_bound,planes_bound,planes_exact} = geometry
  Planes.Room{cap_ok,limit,+fits,+capped,elements_positive,channels_positive} = room
  Order.transitive(Nat.mul(Nat.mul(Planes.phases_of(window),Planes.elements_of(planes)),Planes.channel_count(window)),U32.to_nat(U32.shln(1,depth)),U32.to_nat(cap),fits,capped)

def geometry_tiles(+window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+cap: U32,+geometry: Planes.Geometry<window,planes,depth,cap>) ->
  {Nat.is_lt(0n,Planes.tiles_of(planes)) == True{} : Bool}:
  Planes.Geometry{room,tiles_bound,+tiles_positive,divisor_bound,planes_bound,planes_exact} = geometry
  tiles_positive

def geometry_divisor(+window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+cap: U32,+geometry: Planes.Geometry<window,planes,depth,cap>) ->
  {Nat.is_le(Nat.double(Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool}:
  Planes.Geometry{room,tiles_bound,tiles_positive,+divisor_bound,planes_bound,planes_exact} = geometry
  divisor_bound

def geometry_planes_bound(+window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+cap: U32,+geometry: Planes.Geometry<window,planes,depth,cap>) ->
  {Nat.is_le(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool}:
  Planes.Geometry{room,tiles_bound,tiles_positive,divisor_bound,+planes_bound,planes_exact} = geometry
  planes_bound

def phases_stride(+window: Packing.Stencil) -> {Planes.phases_of(window) == stride_of(window) : Nat}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  {==}

def context_tiles(+window: Packing.Stencil,+planes: Packing.Planes,+input: Tree.Tree<Tile.Tile8>,+word: U32) ->
  {Planes.row_tiles(Planes.context_of(window,planes,input,word)) == Planes.tiles_of(planes) : Nat}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  {==}

def below_last(+value: Nat,+limit: Nat,+below: {Nat.is_lt(value,limit) == True{} : Bool}) -> {Nat.is_le(value,Nat.sub(limit,1n)) == True{} : Bool}:
  match limit:
    case 0n: Empty.absurd({Nat.is_le(value,Nat.sub(0n,1n)) == True{} : Bool},
      WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(value,0n),True{},Equal.sym(Bool,Nat.is_lt(value,0n),False{},Division.lt_zero(value)),below)))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {Nat.is_le(value,_) == True{} : Bool}
      Division.lt_succ_le(value,rest,below)

# The plane row of the last lane of the unit at a position.
def last_row(+window: Packing.Stencil,+planes: Packing.Planes,+position: U32) -> Nat:
  owner_row(window,planes,position,7n)

# What the lanes of the unit at a position need to read their pixels: the
# planes of the transpose, a cap above every word, the plane rows of every
# lane and tap of the unit and the values of a plane row past every lane.
type Reach<-window: Packing.Stencil,-planes: Packing.Planes,-position: U32,-depth: Nat,-cap: U32> is Data:
  Reach{geometry: Planes.Geometry<window,planes,depth,cap>,sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},
    positive: {Nat.is_lt(0n,output_width_of(window)) == True{} : Bool},width_bound: {Nat.is_le(Nat.double(output_width_of(window)),U32.to_nat(cap)) == True{} : Bool},
    lanes_room: {Nat.is_le(Nat.add(output_width_of(window),8n),U32.to_nat(cap)) == True{} : Bool},stride_positive: {Nat.is_lt(0n,stride_of(window)) == True{} : Bool},
    stride_bound: {Nat.is_le(Nat.double(stride_of(window)),U32.to_nat(cap)) == True{} : Bool},
    ordered: {Nat.is_le(output_width_of(window),Nat.mul(stride_of(window),row_width_of(planes))) == True{} : Bool},
    row_exact: {row_width_of(planes) == Nat.mul(Planes.tiles_of(planes),8n) : Nat},
    width_fits: {Nat.is_le(Nat.add(output_width_of(window),Nat.div(Nat.sub(Passes.tap_columns_of(window),1n),stride_of(window))),Nat.mul(Planes.tiles_of(planes),8n)) == True{} :
      Bool},
    columns_room: {Nat.is_le(Passes.tap_columns_of(window),U32.to_nat(cap)) == True{} : Bool},
    values_room: {Nat.is_le(Nat.mul(Planes.elements_of(planes),8n),U32.to_nat(cap)) == True{} : Bool},
    below_first: {Nat.is_le(first_row_of(planes),Nat.div(U32.to_nat(position),output_width_of(window))) == True{} : Bool},
    covered: {Nat.is_le(Nat.mul(Nat.add(last_row(window,planes,position),Passes.tap_rows_of(window)),Planes.tiles_of(planes)),Planes.elements_of(planes)) == True{} : Bool},
    positions_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(cap)) == True{} : Bool}}

# Lane index of a kernel column's tile of entry (channel, kernel row), packed
# from the planes of the transpose, is the input pixel the spec's window
# sample of that lane reads.
def column_lane(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,
  +unit: U32,+channel: Nat,+kernel_row: Nat,+kernel_column: Nat,+index: Nat,+depth: Nat,+cap: U32,+reach: Reach<window,planes,position,depth,cap>,
  +balanced: Shape.Balanced(Tile.Tile16,depth,pairs),+channel_inside: {Nat.is_lt(channel,Planes.channel_count(window)) == True{} : Bool},
  +row_inside: {Nat.is_lt(kernel_row,Passes.tap_rows_of(window)) == True{} : Bool},+column_inside: {Nat.is_lt(kernel_column,Passes.tap_columns_of(window)) == True{} : Bool},
  +below: {Nat.is_lt(index,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Passes.column_tile(Model.transposed(window,planes,input,pairs),tiles,U32.from_nat(kernel_column),window,planes,position,unit,
    Planes.Spot{channel,kernel_row})) == window_cell(input,window,position,channel,kernel_row,kernel_column,index) : F32}:
  Reach{+geometry,+sixteen,+positive,+width_bound,+lanes_room,+stride_positive,+stride_bound,+ordered,+row_exact,+width_fits,+columns_room,+values_room,+below_first,+covered,
    +positions_room} = reach
  +cap_ok = geometry_cap(window,planes,depth,cap,geometry)
  +tiles_positive = geometry_tiles(window,planes,depth,cap,geometry)
  +ow = output_width_of(window)
  +s = stride_of(window)
  +w = row_width_of(planes)
  +t = Planes.tiles_of(planes)
  +n = Planes.elements_of(planes)
  +channels = Planes.channel_count(window)
  +taps = Passes.tap_rows_of(window)
  +p = U32.to_nat(position)
  +column = Nat.mod(p,ow)
  +second = Nat.sub(ow,column)
  +owning = owner_of(column,ow,index)
  +last = owner_of(column,ow,7n)
  +offset = owner_offset(window,planes,position,kernel_column,index)
  +quotient = Nat.div(offset,8n)
  +spot = spot_at(quotient,kernel_row,t)
  +phase = Nat.mod(kernel_column,s)
  +number = plane_number_of(phase,channel,channels)
  +row = owner_row(window,planes,position,index)
  +top = last_row(window,planes,position)
  +value = owner_value(window,position,kernel_column,index)
  +width8 = Nat.mul(t,8n)
  +target = {Planes.Spot{channel,kernel_row} : Planes.Spot}
  +transposed = Model.transposed(window,planes,input,pairs)
  +lane_word = U32.from_nat(owner_lane(owning,second,ow))
  +at = Passes.pass_at(lane_word,U32.from_nat(offset),U32.from_nat(kernel_column),unit,window,planes)
  +context = Planes.context_of(window,planes,input,U32.from_nat(number))
  +lanes = Lanes.bits((U32.from_nat(number) % Packing.channels_of(window) : U32))
  +shift_index = Nat.add(Nat.mod(offset,8n),index)
  +element = Planes.element(spot,t)
  +down = Nat.mul(kernel_row,width8)
  +first_column = Order.transitive(kernel_column,Passes.tap_columns_of(window),U32.to_nat(cap),Bounds.strict_le(kernel_column,Passes.tap_columns_of(window),column_inside),
    columns_room)
  +phase_inside = DivisionOrder.remainder_small(kernel_column,s,stride_positive)
  +split = Equal.trans(Nat,Nat.add(offset,index),Nat.add(Nat.mul(row,w),value),Nat.add(Nat.mul(row,width8),value),
    offset_split(window,planes,position,kernel_column,index,positive,ordered),
    Equal.cong(Nat,Nat,part => Nat.add(Nat.mul(row,part),value),w,width8,row_exact))
  +sum_exact = Equal.trans(Nat,Nat.add(Nat.add(offset,index),down),Nat.add(Nat.add(Nat.mul(row,width8),value),down),Planes.outer(Nat.add(row,kernel_row),value,width8),
    Equal.cong(Nat,Nat,part => Nat.add(part,down),Nat.add(offset,index),Nat.add(Nat.mul(row,width8),value),split),rows_down(row,value,kernel_row,width8))
  +element_exact = element_form(quotient,kernel_row,t,tiles_positive)
  +linear = Equal.trans(Nat,Nat.add(Nat.mul(element,8n),shift_index),Nat.add(Nat.mul(Nat.add(quotient,Nat.mul(kernel_row,t)),8n),shift_index),
    Planes.outer(Nat.add(row,kernel_row),value,width8),
    Equal.cong(Nat,Nat,part => Nat.add(Nat.mul(part,8n),shift_index),element,Nat.add(quotient,Nat.mul(kernel_row,t)),element_exact),
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(quotient,Nat.mul(kernel_row,t)),8n),shift_index),Nat.add(Nat.add(offset,index),down),Planes.outer(Nat.add(row,kernel_row),value,width8),
      lanes_linear(offset,index,kernel_row,t),sum_exact))
  +left_inside = DivisionOrder.remainder_small(Nat.add(column,index),ow,positive)
  +shift_le = DivisionOrder.quotient_le(kernel_column,Nat.sub(Passes.tap_columns_of(window),1n),s,stride_positive,below_last(kernel_column,Passes.tap_columns_of(window),column_inside))
  +value_inside = Order.lt_le(value,Nat.add(ow,Nat.div(kernel_column,s)),width8,
    Order.add_lt(Nat.mod(Nat.add(column,index),ow),ow,Nat.div(kernel_column,s),left_inside),
    Order.transitive(Nat.add(ow,Nat.div(kernel_column,s)),Nat.add(ow,Nat.div(Nat.sub(Passes.tap_columns_of(window),1n),s)),width8,
      Order.offset_le(ow,Nat.div(kernel_column,s),Nat.div(Nat.sub(Passes.tap_columns_of(window),1n),s),shift_le),width_fits))
  +rows_past = Nat.sub(Nat.div(p,ow),first_row_of(planes))
  +row_le = RowMajor.mul_le(Nat.add(rows_past,owning),Nat.add(rows_past,last),s,Order.offset_le(rows_past,owning,last,owner_inside(column,ow,index,positive,below)))
  +rows_below = Order.le_lt_trans(Nat.add(row,kernel_row),Nat.add(top,kernel_row),Nat.add(top,taps),Order.add_le(row,top,kernel_row,row_le),
    Order.offset_lt(top,kernel_row,taps,row_inside))
  +span_le = {
    %RowMajor.assoc(Nat.add(top,taps),t,8n) : {Nat.is_le(_,Nat.mul(n,8n)) == True{} : Bool}
    RowMajor.mul_le(Nat.mul(Nat.add(top,taps),t),n,8n,covered) : {Nat.is_le(Nat.mul(Nat.add(top,taps),width8),Nat.mul(n,8n)) == True{} : Bool}}
  +outer_below = Order.lt_le(Planes.outer(Nat.add(row,kernel_row),value,width8),Nat.mul(Nat.add(top,taps),width8),Nat.mul(n,8n),
    RowMajor.radix(Nat.add(row,kernel_row),Nat.add(top,taps),value,width8,rows_below,value_inside),span_le)
  +element_inside = RowMajor.cancel_multiply_lt(element,n,8n,
    Order.le_lt_trans(Nat.mul(element,8n),Nat.add(Nat.mul(element,8n),shift_index),Nat.mul(n,8n),Order.le_add(Nat.mul(element,8n),shift_index),{
      %Equal.sym(Nat,Nat.add(Nat.mul(element,8n),shift_index),Planes.outer(Nat.add(row,kernel_row),value,width8),linear) : {Nat.is_lt(_,Nat.mul(n,8n)) == True{} : Bool}
      outer_below : {Nat.is_lt(Nat.add(Nat.mul(element,8n),shift_index),Nat.mul(n,8n)) == True{} : Bool}}))
  +offset_room = Order.transitive(offset,Nat.mul(n,8n),U32.to_nat(cap),
    Bounds.strict_le(offset,Nat.mul(n,8n),Order.le_lt_trans(offset,Nat.add(Nat.add(offset,index),down),Nat.mul(n,8n),
      Order.transitive(offset,Nat.add(offset,index),Nat.add(Nat.add(offset,index),down),Order.le_add(offset,index),Order.le_add(Nat.add(offset,index),down)),{
        %Equal.sym(Nat,Nat.add(Nat.add(offset,index),down),Planes.outer(Nat.add(row,kernel_row),value,width8),sum_exact) : {Nat.is_lt(_,Nat.mul(n,8n)) == True{} : Bool}
        outer_below : {Nat.is_lt(Nat.add(Nat.add(offset,index),down),Nat.mul(n,8n)) == True{} : Bool}})),values_room)
  +quotient_inside = Order.le_lt_trans(quotient,element,n,{
    %Equal.sym(Nat,element,Nat.add(quotient,Nat.mul(kernel_row,t)),element_exact) : {Nat.is_le(quotient,_) == True{} : Bool}
    Order.le_add(quotient,Nat.mul(kernel_row,t)) : {Nat.is_le(quotient,element) == True{} : Bool}},element_inside)
  +fits = {
    %phases_stride(window) : {Nat.is_le(Nat.mul(Nat.mul(_,n),channels),U32.to_nat(cap)) == True{} : Bool}
    geometry_fits(window,planes,depth,cap,geometry) : {Nat.is_le(Nat.mul(Nat.mul(s,n),channels),U32.to_nat(cap)) == True{} : Bool}}
  +origin_room = Order.transitive(Nat.mul(Nat.add(Nat.mul(phase,n),quotient),channels),Nat.mul(Nat.mul(s,n),channels),U32.to_nat(cap),
    RowMajor.mul_le(Nat.add(Nat.mul(phase,n),quotient),Nat.mul(s,n),channels,Bounds.strict_le(Nat.add(Nat.mul(phase,n),quotient),Nat.mul(s,n),
      RowMajor.radix(phase,s,quotient,n,phase_inside,quotient_inside))),fits)
  +taps_positive = Order.le_lt_trans(0n,kernel_row,taps,Order.zero_le(kernel_row),row_inside)
  +tiles_le = Order.transitive(t,Nat.mul(Nat.add(top,taps),t),n,{
    %RowMajor.multiply_commutative(t,Nat.add(top,taps)) : {Nat.is_le(t,_) == True{} : Bool}
    RowMajor.product_covers(t,Nat.add(top,taps),Order.lt_le(0n,taps,Nat.add(top,taps),taps_positive,Order.right_le(top,taps))) :
      {Nat.is_le(t,Nat.mul(Nat.add(top,taps),t)) == True{} : Bool}},covered)
  +elements_le = {
    %RowMajor.multiply_commutative(n,s) : {Nat.is_le(n,_) == True{} : Bool}
    RowMajor.product_covers(n,s,stride_positive) : {Nat.is_le(n,Nat.mul(s,n)) == True{} : Bool}}
  +step_room = Order.transitive(Nat.mul(t,channels),Nat.mul(Nat.mul(s,n),channels),U32.to_nat(cap),
    Order.transitive(Nat.mul(t,channels),Nat.mul(n,channels),Nat.mul(Nat.mul(s,n),channels),RowMajor.mul_le(t,n,channels,tiles_le),
      RowMajor.mul_le(n,Nat.mul(s,n),channels,elements_le)),fits)
  +number_below = RowMajor.radix(phase,s,channel,channels,phase_inside,channel_inside)
  +number_inside = {
    %Equal.sym(Nat,Planes.phases_of(window),s,phases_stride(window)) : {Nat.is_lt(number,Nat.mul(_,channels)) == True{} : Bool}
    number_below : {Nat.is_lt(number,Nat.mul(Planes.phases_of(window),channels)) == True{} : Bool}}
  +number_room = Order.transitive(number,Nat.mul(s,channels),U32.to_nat(cap),Bounds.strict_le(number,Nat.mul(s,channels),number_below),{
    %phases_stride(window) : {Nat.is_le(Nat.mul(_,channels),U32.to_nat(cap)) == True{} : Bool}
    geometry_planes_bound(window,planes,depth,cap,geometry) : {Nat.is_le(Nat.mul(s,channels),U32.to_nat(cap)) == True{} : Bool}})
  +shift_below = Order.lt_le(shift_index,Nat.add(8n,index),16n,Order.add_lt(Nat.mod(offset,8n),8n,index,DivisionOrder.remainder_small(offset,8n,{==})),
    Bounds.strict_le(Nat.add(8n,index),16n,Order.offset_lt(8n,index,8n,below)))
  +position_room = Order.transitive(Nat.add(p,index),Nat.add(p,8n),U32.to_nat(cap),Order.offset_le(p,index,8n,Bounds.strict_le(index,8n,below)),positions_room)
  +rtiles = context_tiles(window,planes,input,U32.from_nat(number))
  +s1 = column_owner(transposed,tiles,U32.from_nat(kernel_column),window,planes,position,unit,target,index,cap,cap_ok,below,positive,width_bound,lanes_room)
  +s2 = Equal.trans(U32,base_after(owning,Model.first_base(window,planes,position,U32.from_nat(kernel_column)),window,planes),
    base_after(owning,U32.from_nat(first_offset(window,planes,position,kernel_column)),window,planes),U32.from_nat(offset),
    Equal.cong(U32,U32,word => base_after(owning,word,window,planes),Model.first_base(window,planes,position,U32.from_nat(kernel_column)),
      U32.from_nat(first_offset(window,planes,position,kernel_column)),
      first_word(window,planes,position,kernel_column,cap,cap_ok,positive,width_bound,stride_positive,stride_bound,first_column,below_first)),
    base_closed(owning,first_offset(window,planes,position,kernel_column),window,planes,ordered))
  +s3 = lane_value(transposed,lane_word,offset,U32.from_nat(kernel_column),unit,window,planes,target,index,cap,cap_ok,below,offset_room,sixteen)
  +s4 = pair_word(lane_word,offset,kernel_column,unit,window,planes,channel,kernel_row,cap,cap_ok,first_column,offset_room,stride_positive,stride_bound,sixteen,origin_room,
    step_room,tiles_positive,channel_inside)
  +s5 = Planes.transposed_pair(input,pairs,window,planes,number,spot,depth,cap,geometry,balanced,number_inside,DivisionOrder.remainder_small(quotient,t,tiles_positive),
    element_inside)
  +s6 = pair_linear(context,lanes,spot,shift_index,Nat.add(row,kernel_row),value,shift_below,{
      %Equal.sym(Nat,Planes.row_tiles(context),t,rtiles) : {Nat.is_lt(Planes.column_of(spot),_) == True{} : Bool}
      DivisionOrder.remainder_small(quotient,t,tiles_positive) : {Nat.is_lt(Planes.column_of(spot),Planes.row_tiles(context)) == True{} : Bool}},{
      %Equal.sym(Nat,Planes.row_tiles(context),t,rtiles) : {Nat.is_lt(value,Nat.mul(_,8n)) == True{} : Bool}
      value_inside : {Nat.is_lt(value,Nat.mul(Planes.row_tiles(context),8n)) == True{} : Bool}},{
      %Equal.sym(Nat,Planes.row_tiles(context),t,rtiles) :
        {Nat.add(Nat.mul(Planes.element(spot,_),8n),shift_index) == Planes.outer(Nat.add(row,kernel_row),value,Nat.mul(_,8n)) : Nat}
      linear : {Nat.add(Nat.mul(Planes.element(spot,Planes.row_tiles(context)),8n),shift_index) == Planes.outer(Nat.add(row,kernel_row),value,Nat.mul(Planes.row_tiles(context),8n)) :
        Nat}})
  +s7 = plane_pixel(input,window,planes,position,channel,kernel_row,kernel_column,index,cap,cap_ok,positive,width_bound,stride_positive,position_room,below_first,channel_inside,
    geometry_divisor(window,planes,depth,cap,geometry),number_room)
  +shifted = pass_shifted(transposed,lane_word,U32.from_nat(offset),U32.from_nat(kernel_column),unit,window,planes,target)
  +read = Tree.read(Tile.Tile16,transposed,Passes.entry_pair(at,target))
  +word = Planes.plane_word(Planes.plane_of(number,channels),n,channels,element)
  Equal.trans(F32,TileLaws.select_lane(index,Passes.column_tile(transposed,tiles,U32.from_nat(kernel_column),window,planes,position,unit,target)),
    TileLaws.select_lane(index,pass_shifted(transposed,lane_word,base_after(owning,Model.first_base(window,planes,position,U32.from_nat(kernel_column)),window,planes),
      U32.from_nat(kernel_column),unit,window,planes,target)),window_cell(input,window,position,channel,kernel_row,kernel_column,index),s1,
    Equal.trans(F32,TileLaws.select_lane(index,pass_shifted(transposed,lane_word,base_after(owning,Model.first_base(window,planes,position,U32.from_nat(kernel_column)),window,planes),
        U32.from_nat(kernel_column),unit,window,planes,target)),TileLaws.select_lane(index,shifted),window_cell(input,window,position,channel,kernel_row,kernel_column,index),
      Equal.cong(U32,F32,base => TileLaws.select_lane(index,pass_shifted(transposed,lane_word,base,U32.from_nat(kernel_column),unit,window,planes,target)),
        base_after(owning,Model.first_base(window,planes,position,U32.from_nat(kernel_column)),window,planes),U32.from_nat(offset),s2),
      Equal.trans(F32,TileLaws.select_lane(index,shifted),pair_lane(shift_index,read),window_cell(input,window,position,channel,kernel_row,kernel_column,index),s3,
        Equal.trans(F32,pair_lane(shift_index,read),pair_lane(shift_index,Tree.read(Tile.Tile16,transposed,word)),window_cell(input,window,position,channel,kernel_row,kernel_column,index),
          Equal.cong(U32,F32,index_word => pair_lane(shift_index,Tree.read(Tile.Tile16,transposed,index_word)),Passes.entry_pair(at,target),word,s4),
          Equal.trans(F32,pair_lane(shift_index,Tree.read(Tile.Tile16,transposed,word)),pair_lane(shift_index,Planes.plane_pair(input,window,planes,number,spot)),
            window_cell(input,window,position,channel,kernel_row,kernel_column,index),
            Equal.cong(Tile.Tile16,F32,pair => pair_lane(shift_index,pair),Tree.read(Tile.Tile16,transposed,word),Planes.plane_pair(input,window,planes,number,spot),s5),
            Equal.trans(F32,pair_lane(shift_index,Planes.lanes_pair(context,lanes,spot)),plane_sample(context,lanes,Nat.add(row,kernel_row),value),
              window_cell(input,window,position,channel,kernel_row,kernel_column,index),s6,s7))))))

# ---- A lane of a packed tile ----

# Lane index of a packed tile of a unit, packed from the planes of the
# transpose, is the input pixel the window sample of that lane reads for the
# tile's channel, kernel row and kernel column.
def pack_lane(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+count: Nat,
  +start: U32,+units: Nat,+depth: Nat,+room: Passes.PackRoom<window,units,depth>,+units_exact: {Model.unit_count(window,count) == units : Nat},
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+query: Nat,+query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +owned: {Nat.is_lt(Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window)),units) == True{} : Bool},+pair_depth: Nat,+cap: U32,
  +reach: Reach<window,planes,Passes.position_of(start,Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window))),pair_depth,cap>,
  +pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),
  +entries_exact: {Model.pass_entries(window) == Nat.mul(Planes.channel_count(window),Passes.tap_rows_of(window)) : Nat},+index: Nat,
  +below: {Nat.is_lt(index,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Tree.read(Tile.Tile8,Model.packed_tiles(window,planes,count,start,Model.transposed(window,planes,input,pairs),tiles),U32.from_nat(query))) ==
    window_cell(input,window,Passes.position_of(start,Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window))),
      Nat.div(Passes.entry_of(query,Model.pass_entries(window),Passes.tap_columns_of(window)),Passes.tap_rows_of(window)),
      Nat.mod(Passes.entry_of(query,Model.pass_entries(window),Passes.tap_columns_of(window)),Passes.tap_rows_of(window)),
      Passes.column_index(query,Passes.tap_columns_of(window)),index) : F32}:
  +entries = Model.pass_entries(window)
  +columns = Passes.tap_columns_of(window)
  +taps = Passes.tap_rows_of(window)
  +owning = Passes.unit_of(query,entries,columns)
  +entry = Passes.entry_of(query,entries,columns)
  +transposed = Model.transposed(window,planes,input,pairs)
  +packed = Model.packed_tiles(window,planes,count,start,transposed,tiles)
  +value = Passes.pack_value(transposed,tiles,window,planes,start,query)
  +taps_positive = Passes.pack_taps(window,units,depth,room)
  +entry_inside = DivisionOrder.remainder_small(Nat.div(query,columns),entries,Passes.pack_entries(window,units,depth,room))
  +channel_inside = DivisionOrder.quotient_below(entry,Planes.channel_count(window),taps,taps_positive,{
    %entries_exact : {Nat.is_lt(entry,_) == True{} : Bool}
    entry_inside : {Nat.is_lt(entry,Nat.mul(Planes.channel_count(window),taps)) == True{} : Bool}})
  +read_value = Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,packed,U32.from_nat(query)),Bool.pick(Tile.Tile8,Nat.is_lt(owning,units),value,Tree.read(Tile.Tile8,tiles,U32.from_nat(query))),
    value,Passes.pack_read(window,planes,count,start,transposed,tiles,units,depth,room,units_exact,balanced,query,query_inside),
    %Equal.sym(Bool,Nat.is_lt(owning,units),True{},owned) : {Bool.pick(Tile.Tile8,_,value,Tree.read(Tile.Tile8,tiles,U32.from_nat(query))) == value : Tile.Tile8}
    {==})
  Equal.trans(F32,TileLaws.select_lane(index,Tree.read(Tile.Tile8,packed,U32.from_nat(query))),TileLaws.select_lane(index,value),
    window_cell(input,window,Passes.position_of(start,owning),Nat.div(entry,taps),Nat.mod(entry,taps),Passes.column_index(query,columns),index),
    Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(index,tile),Tree.read(Tile.Tile8,packed,U32.from_nat(query)),value,read_value),
    column_lane(input,pairs,tiles,window,planes,Passes.position_of(start,owning),U32.from_nat(owning),Nat.div(entry,taps),Nat.mod(entry,taps),Passes.column_index(query,columns),
      index,pair_depth,cap,reach,pairs_balanced,channel_inside,DivisionOrder.remainder_small(entry,taps,taps_positive),
      DivisionOrder.remainder_small(query,columns,Passes.pack_columns(window,units,depth,room)),below))

# ---- Lanes as samples of the values ----

def height_of(+window: Packing.Stencil) -> U32:
  Packing.Stencil{base,+height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  height

def width_of(+window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,+width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  width

def groups_of(+window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,+groups} = window
  groups

# The values of the channels of the cells of a window, from its base on.
def window_values(+window: Packing.Stencil,+input: Tree.Tree<Tile.Tile8>) -> Tree.Tree<F32>:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,stride,row_origin,column_origin,output_width,+inner,groups} = window
  ValuesAt.values_at(base,(inner / (tap_rows * tap_columns) : U32),height,width,input)

# What the values of a window's cells need: the cells fit the channels of the
# window, and the kernel and the inner extent fit the cap.
type ValuesRoom<-window: Packing.Stencil,-cap: U32> is Data:
  ValuesRoom{fits: Lanes.Fits<Packing.channels_of(window),height_of(window),width_of(window),groups_of(window)>,
    taps_bound: {Nat.is_le(Nat.double(Nat.mul(Passes.tap_rows_of(window),Passes.tap_columns_of(window))),U32.to_nat(cap)) == True{} : Bool},
    inner_room: {Nat.is_le(U32.to_nat(Passes.inner_of(window)),U32.to_nat(cap)) == True{} : Bool}}

def reach_cap(+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+depth: Nat,+cap: U32,+reach: Reach<window,planes,position,depth,cap>) ->
  {U32.is_le(cap,2147483648) == True{} : Bool}:
  Reach{+geometry,sixteen,positive,width_bound,lanes_room,stride_positive,stride_bound,ordered,row_exact,width_fits,columns_room,values_room,below_first,covered,
    positions_room} = reach
  geometry_cap(window,planes,depth,cap,geometry)

# A cell of a channel, masked alike, is the value of that channel at that
# pixel.
def values_cell(+base: U32,+channels: U32,+h: U32,+w: U32,+g: U32,+fits: Lanes.Fits<channels,h,w,g>,+x: Tree.Tree<Tile.Tile8>,+c: U32,+iy: U32,+ix: U32,+inside: Bool,
  +bound: {U32.is_lt(c,channels) == True{} : Bool},+exact: {(U32.is_lt(iy,h) && U32.is_lt(ix,w)) == inside : Bool}) ->
  {PackingSamples.masked(Lanes.lane_at(Lanes.bits(c),Tree.read(Tile.Tile8,x,((base + (c / 8 : U32) : U32) + (iy * w + ix) * g : U32))),inside) ==
    PackingSamples.masked(Packing.identity(Tree.read(F32,ValuesAt.values_at(base,channels,h,w,x),((0 + (c * (h * w) : U32) : U32) + (iy * w + ix) * 1 : U32))),inside) : F32}:
  match inside:
    case True{}: ValuesAt.values_at_agree(base,channels,h,w,g,fits,x,c,iy,ix,bound,exact)
    case False{}: {==}

# The kernel column, kernel row and channel words of a reduction index.
def reduction_column(+tap_columns: U32,+reduction: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(reduction,U32.to_nat(cap)) == True{} : Bool},+columns_bound: {Nat.is_le(Nat.double(U32.to_nat(tap_columns)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(tap_columns)) == True{} : Bool}) ->
  {(U32.from_nat(reduction) % tap_columns : U32) == U32.from_nat(Nat.mod(reduction,U32.to_nat(tap_columns))) : U32}:
  +columns = U32.to_nat(tap_columns)
  +columns_room = Order.transitive(columns,Nat.double(columns),U32.to_nat(cap),Order.le_double(columns),columns_bound)
  %Arithmetic.native_roundtrip(tap_columns) : {(U32.from_nat(reduction) % _ : U32) == U32.from_nat(Nat.mod(reduction,columns)) : U32}
  WordDivision.encoded_modulo(reduction,columns,cap,cap_ok,room,columns_room,columns_bound,positive)

def reduction_row(+tap_rows: U32,+tap_columns: U32,+reduction: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(reduction,U32.to_nat(cap)) == True{} : Bool},+rows_bound: {Nat.is_le(Nat.double(U32.to_nat(tap_rows)),U32.to_nat(cap)) == True{} : Bool},
  +columns_bound: {Nat.is_le(Nat.double(U32.to_nat(tap_columns)),U32.to_nat(cap)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(tap_rows)) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(tap_columns)) == True{} : Bool}) ->
  {(U32.from_nat(reduction) / tap_columns % tap_rows : U32) == U32.from_nat(Nat.mod(Nat.div(reduction,U32.to_nat(tap_columns)),U32.to_nat(tap_rows))) : U32}:
  +rows = U32.to_nat(tap_rows)
  +columns = U32.to_nat(tap_columns)
  +limit = U32.to_nat(cap)
  +quotient = Nat.div(reduction,columns)
  +rows_room = Order.transitive(rows,Nat.double(rows),limit,Order.le_double(rows),rows_bound)
  +columns_room = Order.transitive(columns,Nat.double(columns),limit,Order.le_double(columns),columns_bound)
  +quotient_room = Order.transitive(quotient,reduction,limit,DivisionOrder.quotient_at_most(reduction,columns,columns_positive),room)
  %Arithmetic.native_roundtrip(tap_columns) : {(U32.from_nat(reduction) / _ % tap_rows : U32) == U32.from_nat(Nat.mod(quotient,rows)) : U32}
  %Equal.sym(U32,(U32.from_nat(reduction) / U32.from_nat(columns) : U32),U32.from_nat(quotient),
    WordDivision.encoded_divide(reduction,columns,cap,cap_ok,room,columns_room,columns_bound,columns_positive)) :
    {(_ % tap_rows : U32) == U32.from_nat(Nat.mod(quotient,rows)) : U32}
  %Arithmetic.native_roundtrip(tap_rows) : {(U32.from_nat(quotient) % _ : U32) == U32.from_nat(Nat.mod(quotient,rows)) : U32}
  WordDivision.encoded_modulo(quotient,rows,cap,cap_ok,quotient_room,rows_room,rows_bound,rows_positive)

def reduction_channel(+tap_rows: U32,+tap_columns: U32,+reduction: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(reduction,U32.to_nat(cap)) == True{} : Bool},
  +taps_bound: {Nat.is_le(Nat.double(Nat.mul(U32.to_nat(tap_rows),U32.to_nat(tap_columns))),U32.to_nat(cap)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(tap_rows)) == True{} : Bool},+columns_positive: {Nat.is_lt(0n,U32.to_nat(tap_columns)) == True{} : Bool}) ->
  {(U32.from_nat(reduction) / (tap_rows * tap_columns) : U32) == U32.from_nat(Nat.div(Nat.div(reduction,U32.to_nat(tap_columns)),U32.to_nat(tap_rows))) : U32}:
  +rows = U32.to_nat(tap_rows)
  +columns = U32.to_nat(tap_columns)
  +taps = Nat.mul(rows,columns)
  +taps_positive = Order.product_positive(rows,columns,rows_positive,columns_positive)
  +taps_room = Order.transitive(taps,Nat.double(taps),U32.to_nat(cap),Order.le_double(taps),taps_bound)
  %Arithmetic.native_roundtrip(tap_rows) : {(U32.from_nat(reduction) / (_ * tap_columns) : U32) == U32.from_nat(Nat.div(Nat.div(reduction,columns),rows)) : U32}
  %Arithmetic.native_roundtrip(tap_columns) : {(U32.from_nat(reduction) / (U32.from_nat(rows) * _) : U32) == U32.from_nat(Nat.div(Nat.div(reduction,columns),rows)) : U32}
  %Equal.sym(U32,(U32.from_nat(rows) * U32.from_nat(columns) : U32),U32.from_nat(taps),Multiplication.native_encoded_multiply(rows,columns)) :
    {(U32.from_nat(reduction) / _ : U32) == U32.from_nat(Nat.div(Nat.div(reduction,columns),rows)) : U32}
  %DivisionOrder.product_quotient(reduction,rows,columns,rows_positive,columns_positive) : {(U32.from_nat(reduction) / U32.from_nat(taps) : U32) == U32.from_nat(_) : U32}
  WordDivision.encoded_divide(reduction,taps,cap,cap_ok,room,taps_room,taps_bound,taps_positive)

# The spec's sample of values at a position, a channel, a kernel row and a
# kernel column.
def values_sample(+values: Tree.Tree<F32>,+window: Packing.Stencil,+position: U32,+channel: U32,+kernel_row: U32,+kernel_column: U32,+index: Nat) -> F32:
  Packing.Stencil{base,+height,+width,tap_rows,tap_columns,+stride,+row_origin,+column_origin,+output_width,inner,groups} = window
  +at = (position + U32.from_nat(index) : U32)
  +row = (at / output_width * stride + kernel_row + row_origin : U32)
  +column = (at % output_width * stride + kernel_column + column_origin : U32)
  PackingSamples.masked(Packing.identity(Tree.read(F32,values,((0 + (channel * (height * width) : U32) : U32) + (row * width + column) * 1 : U32))),
    U32.is_lt(row,height) && U32.is_lt(column,width))

# Lane index of the window cells at the channel and kernel row of an entry
# and at a kernel column, the entry and the column those of a reduction
# index, is the spec's window sample of the values of the cells at that index.
def cell_sample(+input: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+position: U32,+reduction: Nat,+entry: Nat,+column: Nat,+index: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(reduction,U32.to_nat(cap)) == True{} : Bool},
  +taps_bound: {Nat.is_le(Nat.double(Nat.mul(Passes.tap_rows_of(window),Passes.tap_columns_of(window))),U32.to_nat(cap)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,Passes.tap_rows_of(window)) == True{} : Bool},+columns_positive: {Nat.is_lt(0n,Passes.tap_columns_of(window)) == True{} : Bool},
  +fits: Lanes.Fits<Packing.channels_of(window),height_of(window),width_of(window),groups_of(window)>,
  +entry_exact: {Nat.div(reduction,Passes.tap_columns_of(window)) == entry : Nat},+column_exact: {Nat.mod(reduction,Passes.tap_columns_of(window)) == column : Nat},
  +channel_inside: {Nat.is_lt(Nat.div(entry,Passes.tap_rows_of(window)),Planes.channel_count(window)) == True{} : Bool}) ->
  {window_cell(input,window,position,Nat.div(entry,Passes.tap_rows_of(window)),Nat.mod(entry,Passes.tap_rows_of(window)),column,index) ==
    SpatialSamples.window_sample(~F32,~Packing.identity,window_values(window,input),
      SpatialSamples.window_context(~SpatialSamples.planes,~SpatialSamples.adjacent,SpatialSamples.plain(window),position,U32.from_nat(reduction)),index) : F32}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +rows = U32.to_nat(tap_rows)
  +columns = U32.to_nat(tap_columns)
  +limit = U32.to_nat(cap)
  +taps = Nat.mul(rows,columns)
  +rows_bound = Order.transitive(Nat.double(rows),Nat.double(taps),limit,Order.double_le(rows,taps,RowMajor.product_covers(rows,columns,columns_positive)),taps_bound)
  +columns_cover = {
    %RowMajor.multiply_commutative(columns,rows) : {Nat.is_le(columns,_) == True{} : Bool}
    RowMajor.product_covers(columns,rows,rows_positive) : {Nat.is_le(columns,taps) == True{} : Bool}}
  +columns_bound = Order.transitive(Nat.double(columns),Nat.double(taps),limit,Order.double_le(columns,taps,columns_cover),taps_bound)
  +channel = Nat.div(entry,rows)
  +kernel_row = Nat.mod(entry,rows)
  +channels = (inner / (tap_rows * tap_columns) : U32)
  +values = ValuesAt.values_at(base,channels,height,width,input)
  +word = U32.from_nat(reduction)
  +picked_channel = U32.from_nat(channel)
  +picked_row = U32.from_nat(kernel_row)
  +picked_column = U32.from_nat(column)
  +channel_equal = Equal.trans(U32,(word / (tap_rows * tap_columns) : U32),U32.from_nat(Nat.div(Nat.div(reduction,columns),rows)),picked_channel,
    reduction_channel(tap_rows,tap_columns,reduction,cap,cap_ok,room,taps_bound,rows_positive,columns_positive),
    Equal.cong(Nat,U32,quotient => U32.from_nat(Nat.div(quotient,rows)),Nat.div(reduction,columns),entry,entry_exact))
  +row_equal = Equal.trans(U32,(word / tap_columns % tap_rows : U32),U32.from_nat(Nat.mod(Nat.div(reduction,columns),rows)),picked_row,
    reduction_row(tap_rows,tap_columns,reduction,cap,cap_ok,room,rows_bound,columns_bound,rows_positive,columns_positive),
    Equal.cong(Nat,U32,quotient => U32.from_nat(Nat.mod(quotient,rows)),Nat.div(reduction,columns),entry,entry_exact))
  +column_equal = Equal.trans(U32,(word % tap_columns : U32),U32.from_nat(Nat.mod(reduction,columns)),picked_column,
    reduction_column(tap_columns,reduction,cap,cap_ok,room,columns_bound,columns_positive),
    Equal.cong(Nat,U32,U32.from_nat,Nat.mod(reduction,columns),column,column_exact))
  +quotient_room = Order.transitive(Nat.div(reduction,columns),reduction,limit,DivisionOrder.quotient_at_most(reduction,columns,columns_positive),room)
  +entry_room = {%entry_exact : {Nat.is_le(_,limit) == True{} : Bool} quotient_room : {Nat.is_le(entry,limit) == True{} : Bool}}
  +channel_room = Order.transitive(channel,entry,limit,DivisionOrder.quotient_at_most(entry,rows,rows_positive),entry_room)
  +bound = Equal.trans(Bool,U32.is_lt(picked_channel,channels),Nat.is_lt(U32.to_nat(picked_channel),U32.to_nat(channels)),True{},Comparison.lt(picked_channel,channels),
    %Equal.sym(Nat,U32.to_nat(picked_channel),channel,Bounds.roundtrip(channel,cap,cap_ok,channel_room)) : {Nat.is_lt(_,U32.to_nat(channels)) == True{} : Bool}
    channel_inside)
  +at = (position + U32.from_nat(index) : U32)
  +row = (at / output_width * stride + picked_row + row_origin : U32)
  +spot = (at % output_width * stride + picked_column + column_origin : U32)
  %Equal.sym(U32,(word / (tap_rows * tap_columns) : U32),picked_channel,channel_equal) :
    {window_cell(input,window,position,channel,kernel_row,column,index) ==
      values_sample(values,window,position,_,(word / tap_columns % tap_rows : U32),(word % tap_columns : U32),index) : F32}
  %Equal.sym(U32,(word / tap_columns % tap_rows : U32),picked_row,row_equal) :
    {window_cell(input,window,position,channel,kernel_row,column,index) == values_sample(values,window,position,picked_channel,_,(word % tap_columns : U32),index) : F32}
  %Equal.sym(U32,(word % tap_columns : U32),picked_column,column_equal) :
    {window_cell(input,window,position,channel,kernel_row,column,index) == values_sample(values,window,position,picked_channel,picked_row,_,index) : F32}
  values_cell(base,channels,height,width,groups,fits,input,picked_channel,row,spot,U32.is_lt(row,height) && U32.is_lt(spot,width),bound,{==})

# Lane index of a packed tile of a unit, packed from the planes of the
# transpose, is the spec's window sample of the values of the input cells at
# the unit's position and the tile's reduction index.
def packed_sample(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+count: Nat,
  +start: U32,+units: Nat,+depth: Nat,+room: Passes.PackRoom<window,units,depth>,+units_exact: {Model.unit_count(window,count) == units : Nat},
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+query: Nat,+query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +owned: {Nat.is_lt(Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window)),units) == True{} : Bool},+pair_depth: Nat,+cap: U32,
  +reach: Reach<window,planes,Passes.position_of(start,Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window))),pair_depth,cap>,
  +pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),
  +entries_exact: {Model.pass_entries(window) == Nat.mul(Planes.channel_count(window),Passes.tap_rows_of(window)) : Nat},+values_room: ValuesRoom<window,cap>,+index: Nat,
  +below: {Nat.is_lt(index,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(index,Tree.read(Tile.Tile8,Model.packed_tiles(window,planes,count,start,Model.transposed(window,planes,input,pairs),tiles),U32.from_nat(query))) ==
    SpatialSamples.window_sample(~F32,~Packing.identity,window_values(window,input),
      SpatialSamples.window_context(~SpatialSamples.planes,~SpatialSamples.adjacent,SpatialSamples.plain(window),
        Passes.position_of(start,Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window))),
        U32.from_nat(Nat.mod(query,U32.to_nat(Passes.inner_of(window))))),index) : F32}:
  ValuesRoom{+fits,+taps_bound,+inner_room} = values_room
  +entries = Model.pass_entries(window)
  +columns = Passes.tap_columns_of(window)
  +taps = Passes.tap_rows_of(window)
  +owning = Passes.unit_of(query,entries,columns)
  +entry = Passes.entry_of(query,entries,columns)
  +position = Passes.position_of(start,owning)
  +inner = U32.to_nat(Passes.inner_of(window))
  +reduction = Nat.mod(query,inner)
  +columns_positive = Passes.pack_columns(window,units,depth,room)
  +entries_positive = Passes.pack_entries(window,units,depth,room)
  +taps_positive = Passes.pack_taps(window,units,depth,room)
  +inner_exact = Passes.pack_inner(window,units,depth,room)
  +cap_ok = reach_cap(window,planes,position,pair_depth,cap,reach)
  +inner_positive = {
    %Equal.sym(Nat,inner,Nat.mul(entries,columns),inner_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(entries,columns,entries_positive,columns_positive) : {Nat.is_lt(0n,inner) == True{} : Bool}}
  +reduction_room = Order.transitive(reduction,inner,U32.to_nat(cap),Bounds.strict_le(reduction,inner,DivisionOrder.remainder_small(query,inner,inner_positive)),inner_room)
  +entry_exact = {
    %Equal.sym(Nat,inner,Nat.mul(entries,columns),inner_exact) : {Nat.div(Nat.mod(query,_),columns) == entry : Nat}
    DivisionOrder.remainder_quotient(query,entries,columns,entries_positive,columns_positive) : {Nat.div(reduction,columns) == entry : Nat}}
  +column_exact = {
    %Equal.sym(Nat,inner,Nat.mul(entries,columns),inner_exact) : {Nat.mod(Nat.mod(query,_),columns) == Passes.column_index(query,columns) : Nat}
    DivisionOrder.remainder_remainder(query,entries,columns,entries_positive,columns_positive) : {Nat.mod(reduction,columns) == Passes.column_index(query,columns) : Nat}}
  +entry_inside = DivisionOrder.remainder_small(Nat.div(query,columns),entries,entries_positive)
  +channel_inside = DivisionOrder.quotient_below(entry,Planes.channel_count(window),taps,taps_positive,{
    %entries_exact : {Nat.is_lt(entry,_) == True{} : Bool}
    entry_inside : {Nat.is_lt(entry,Nat.mul(Planes.channel_count(window),taps)) == True{} : Bool}})
  Equal.trans(F32,TileLaws.select_lane(index,Tree.read(Tile.Tile8,Model.packed_tiles(window,planes,count,start,Model.transposed(window,planes,input,pairs),tiles),U32.from_nat(query))),
    window_cell(input,window,position,Nat.div(entry,taps),Nat.mod(entry,taps),Passes.column_index(query,columns),index),
    SpatialSamples.window_sample(~F32,~Packing.identity,window_values(window,input),
      SpatialSamples.window_context(~SpatialSamples.planes,~SpatialSamples.adjacent,SpatialSamples.plain(window),position,U32.from_nat(reduction)),index),
    pack_lane(input,pairs,tiles,window,planes,count,start,units,depth,room,units_exact,balanced,query,query_inside,owned,pair_depth,cap,reach,pairs_balanced,entries_exact,index,below),
    cell_sample(input,window,position,reduction,entry,Passes.column_index(query,columns),index,cap,cap_ok,reduction_room,taps_bound,taps_positive,columns_positive,fits,
      entry_exact,column_exact,channel_inside))

# A packed tile of a unit, packed from the planes of the transpose, is the
# spec's window vector of the values of the input cells at the unit's
# position and the tile's reduction index.
def pack_tile(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+count: Nat,
  +start: U32,+units: Nat,+depth: Nat,+room: Passes.PackRoom<window,units,depth>,+units_exact: {Model.unit_count(window,count) == units : Nat},
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+query: Nat,+query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +owned: {Nat.is_lt(Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window)),units) == True{} : Bool},+pair_depth: Nat,+cap: U32,
  +reach: Reach<window,planes,Passes.position_of(start,Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window))),pair_depth,cap>,
  +pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),
  +entries_exact: {Model.pass_entries(window) == Nat.mul(Planes.channel_count(window),Passes.tap_rows_of(window)) : Nat},+values_room: ValuesRoom<window,cap>) ->
  {Tree.read(Tile.Tile8,Model.packed_tiles(window,planes,count,start,Model.transposed(window,planes,input,pairs),tiles),U32.from_nat(query)) ==
    SpatialSamples.value_vector(window_values(window,input),window,Passes.position_of(start,Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window))),
      U32.from_nat(Nat.mod(query,U32.to_nat(Passes.inner_of(window))))) : Tile.Tile8}:
  +packed = Tree.read(Tile.Tile8,Model.packed_tiles(window,planes,count,start,Model.transposed(window,planes,input,pairs),tiles),U32.from_nat(query))
  +position = Passes.position_of(start,Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window)))
  +reduction = U32.from_nat(Nat.mod(query,U32.to_nat(Passes.inner_of(window))))
  TileLaws.tiles_equal(packed,SpatialSamples.value_vector(window_values(window,input),window,position,reduction),
    packed_sample(input,pairs,tiles,window,planes,count,start,units,depth,room,units_exact,balanced,query,query_inside,owned,pair_depth,cap,reach,pairs_balanced,entries_exact,values_room,0n,{==}),
    packed_sample(input,pairs,tiles,window,planes,count,start,units,depth,room,units_exact,balanced,query,query_inside,owned,pair_depth,cap,reach,pairs_balanced,entries_exact,values_room,1n,{==}),
    packed_sample(input,pairs,tiles,window,planes,count,start,units,depth,room,units_exact,balanced,query,query_inside,owned,pair_depth,cap,reach,pairs_balanced,entries_exact,values_room,2n,{==}),
    packed_sample(input,pairs,tiles,window,planes,count,start,units,depth,room,units_exact,balanced,query,query_inside,owned,pair_depth,cap,reach,pairs_balanced,entries_exact,values_room,3n,{==}),
    packed_sample(input,pairs,tiles,window,planes,count,start,units,depth,room,units_exact,balanced,query,query_inside,owned,pair_depth,cap,reach,pairs_balanced,entries_exact,values_room,4n,{==}),
    packed_sample(input,pairs,tiles,window,planes,count,start,units,depth,room,units_exact,balanced,query,query_inside,owned,pair_depth,cap,reach,pairs_balanced,entries_exact,values_room,5n,{==}),
    packed_sample(input,pairs,tiles,window,planes,count,start,units,depth,room,units_exact,balanced,query,query_inside,owned,pair_depth,cap,reach,pairs_balanced,entries_exact,values_room,6n,{==}),
    packed_sample(input,pairs,tiles,window,planes,count,start,units,depth,room,units_exact,balanced,query,query_inside,owned,pair_depth,cap,reach,pairs_balanced,entries_exact,values_room,7n,{==}))

# ---- The units of a block ----

# One past the last position of the units of a block, and its last position.
def block_end(+start: U32,+units: Nat) -> Nat:
  Nat.add(U32.to_nat(start),Nat.mul(units,8n))

def block_last(+start: U32,+units: Nat) -> Nat:
  Nat.sub(block_end(start,units),1n)

# What the units of a block need: the facts of a reach that hold at every
# position, the first output row of the planes at or before the block's, the
# plane rows of the block's last position and its taps inside the planes, and
# the block inside the cap.
type Block<-window: Packing.Stencil,-planes: Packing.Planes,-start: U32,-units: Nat,-depth: Nat,-cap: U32> is Data:
  Block{geometry: Planes.Geometry<window,planes,depth,cap>,sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},
    positive: {Nat.is_lt(0n,output_width_of(window)) == True{} : Bool},width_bound: {Nat.is_le(Nat.double(output_width_of(window)),U32.to_nat(cap)) == True{} : Bool},
    lanes_room: {Nat.is_le(Nat.add(output_width_of(window),8n),U32.to_nat(cap)) == True{} : Bool},stride_positive: {Nat.is_lt(0n,stride_of(window)) == True{} : Bool},
    stride_bound: {Nat.is_le(Nat.double(stride_of(window)),U32.to_nat(cap)) == True{} : Bool},
    ordered: {Nat.is_le(output_width_of(window),Nat.mul(stride_of(window),row_width_of(planes))) == True{} : Bool},
    row_exact: {row_width_of(planes) == Nat.mul(Planes.tiles_of(planes),8n) : Nat},
    width_fits: {Nat.is_le(Nat.add(output_width_of(window),Nat.div(Nat.sub(Passes.tap_columns_of(window),1n),stride_of(window))),Nat.mul(Planes.tiles_of(planes),8n)) == True{} :
      Bool},
    columns_room: {Nat.is_le(Passes.tap_columns_of(window),U32.to_nat(cap)) == True{} : Bool},
    values_room: {Nat.is_le(Nat.mul(Planes.elements_of(planes),8n),U32.to_nat(cap)) == True{} : Bool},
    below_first: {Nat.is_le(first_row_of(planes),Nat.div(U32.to_nat(start),output_width_of(window))) == True{} : Bool},
    covered: {Nat.is_le(Nat.mul(Nat.add(Nat.mul(Nat.sub(Nat.div(block_last(start,units),output_width_of(window)),first_row_of(planes)),stride_of(window)),
      Passes.tap_rows_of(window)),Planes.tiles_of(planes)),Planes.elements_of(planes)) == True{} : Bool},
    positions_room: {Nat.is_le(block_end(start,units),U32.to_nat(cap)) == True{} : Bool}}

# The output row past the first of a position plus the rows its lane seven
# moves on is the output row of the position seven later past the first.
def seventh_row(+value: Nat,+width: Nat,+first: Nat,+last: Nat,+positive: {Nat.is_lt(0n,width) == True{} : Bool},
  +below_first: {Nat.is_le(first,Nat.div(value,width)) == True{} : Bool},+reached: {Nat.is_le(Nat.add(value,7n),last) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.sub(Nat.div(value,width),first),Nat.div(Nat.add(Nat.mod(value,width),7n),width)),Nat.sub(Nat.div(last,width),first)) == True{} : Bool}:
  +row = Nat.div(value,width)
  +spot = Nat.mod(value,width)
  +moved = Nat.div(Nat.add(spot,7n),width)
  +bound = Nat.div(last,width)
  +sum_equal = Equal.trans(Nat,Nat.div(Nat.add(value,7n),width),Nat.div(Nat.add(Nat.mul(row,width),Nat.add(spot,7n)),width),Nat.add(row,moved),
    Equal.cong(Nat,Nat,total => Nat.div(total,width),Nat.add(value,7n),Nat.add(Nat.mul(row,width),Nat.add(spot,7n)),
      Equal.trans(Nat,Nat.add(value,7n),Nat.add(Nat.add(Nat.mul(row,width),spot),7n),Nat.add(Nat.mul(row,width),Nat.add(spot,7n)),
        Equal.cong(Nat,Nat,total => Nat.add(total,7n),value,Nat.add(Nat.mul(row,width),spot),DivisionOrder.quotient_exact(value,width,positive)),
        Addition.associative(Nat.mul(row,width),spot,7n))),
    DivisionOrder.quotient_shift(Nat.add(spot,7n),width,row,positive))
  +sum_le = {
    %sum_equal : {Nat.is_le(_,bound) == True{} : Bool}
    DivisionOrder.quotient_le(Nat.add(value,7n),last,width,positive,reached) : {Nat.is_le(Nat.add(row,moved),bound) == True{} : Bool}}
  +first_bound = Order.transitive(first,row,bound,below_first,Order.transitive(row,Nat.add(row,moved),bound,Order.le_add(row,moved),sum_le))
  +shifted = {
    %Addition.associative(first,Nat.sub(row,first),moved) : {Nat.is_le(_,Nat.add(first,Nat.sub(bound,first))) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(first,Nat.sub(row,first)),row,Order.restore_shift(first,row,below_first)) :
      {Nat.is_le(Nat.add(_,moved),Nat.add(first,Nat.sub(bound,first))) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(first,Nat.sub(bound,first)),bound,Order.restore_shift(first,bound,first_bound)) : {Nat.is_le(Nat.add(row,moved),_) == True{} : Bool}
    sum_le : {Nat.is_le(Nat.add(first,Nat.add(Nat.sub(row,first),moved)),Nat.add(first,Nat.sub(bound,first))) == True{} : Bool}}
  Order.cancel_offset_le(first,Nat.add(Nat.sub(row,first),moved),Nat.sub(bound,first),shifted)

# The reach of every unit of a block.
def unit_reach(+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+units: Nat,+depth: Nat,+cap: U32,+block: Block<window,planes,start,units,depth,cap>,
  +unit: Nat,+below: {Nat.is_lt(unit,units) == True{} : Bool}) -> Reach<window,planes,Passes.position_of(start,unit),depth,cap>:
  Block{+geometry,+sixteen,+positive,+width_bound,+lanes_room,+stride_positive,+stride_bound,+ordered,+row_exact,+width_fits,+columns_room,+values_room,+below_first,
    +covered,+positions_room} = block
  +ow = output_width_of(window)
  +first = first_row_of(planes)
  +stride = stride_of(window)
  +taps = Passes.tap_rows_of(window)
  +tiles = Planes.tiles_of(planes)
  +elements = Planes.elements_of(planes)
  +limit = U32.to_nat(cap)
  +cap_ok = geometry_cap(window,planes,depth,cap,geometry)
  +begin = U32.to_nat(start)
  +offset = Nat.mul(unit,8n)
  +value = Nat.add(begin,offset)
  +position = Passes.position_of(start,unit)
  +ending = block_end(start,units)
  +next_room = Order.offset_le(begin,Nat.mul(1n+unit,8n),Nat.mul(units,8n),RowMajor.mul_le(1n+unit,units,8n,Order.strict_follow(unit,units,below)))
  +next_equal = Equal.trans(Nat,Nat.add(value,8n),Nat.add(begin,Nat.add(offset,8n)),Nat.add(begin,Nat.mul(1n+unit,8n)),Addition.associative(begin,offset,8n),
    Equal.cong(Nat,Nat,tail => Nat.add(begin,tail),Nat.add(offset,8n),Nat.add(8n,offset),Order.commutative(offset,8n)))
  +step_end = {%Equal.sym(Nat,Nat.add(value,8n),Nat.add(begin,Nat.mul(1n+unit,8n)),next_equal) : {Nat.is_le(_,ending) == True{} : Bool} next_room : {Nat.is_le(Nat.add(value,8n),ending) == True{} : Bool}}
  +step_room = Order.transitive(Nat.add(value,8n),ending,limit,step_end,positions_room)
  +value_room = Order.transitive(value,Nat.add(value,8n),limit,Order.le_add(value,8n),step_room)
  +offset_room = Order.transitive(offset,value,limit,{%Order.commutative(offset,begin) : {Nat.is_le(offset,_) == True{} : Bool} Order.le_add(offset,begin) : {Nat.is_le(offset,value) == True{} : Bool}},
    value_room)
  +word = Equal.trans(U32,position,(start + U32.from_nat(offset) : U32),U32.from_nat(value),
    Equal.cong(U32,U32,scaled => (start + scaled : U32),(U32.from_nat(unit) * 8 : U32),U32.from_nat(offset),Multiplication.native_encoded_multiply(unit,8n)),{
      %Bounds.roundtrip(offset,cap,cap_ok,offset_room) : {(start + U32.from_nat(offset) : U32) == U32.from_nat(Nat.add(begin,_)) : U32}
      Arithmetic.native_add(start,U32.from_nat(offset)) : {(start + U32.from_nat(offset) : U32) == U32.from_nat(value) : U32}})
  +exact = Equal.trans(Nat,U32.to_nat(position),U32.to_nat(U32.from_nat(value)),value,Equal.cong(U32,Nat,U32.to_nat,position,U32.from_nat(value),word),
    Bounds.roundtrip(value,cap,cap_ok,value_room))
  +below_value = Order.transitive(first,Nat.div(begin,ow),Nat.div(value,ow),below_first,DivisionOrder.quotient_le(begin,value,ow,positive,Order.le_add(begin,offset)))
  +reached = below_last(Nat.add(value,7n),ending,Order.succ_le_lt(Nat.add(value,7n),ending,{
    %Addition.successor_right(value,7n) : {Nat.is_le(_,ending) == True{} : Bool}
    step_end : {Nat.is_le(1n+Nat.add(value,7n),ending) == True{} : Bool}}))
  +rows = seventh_row(value,ow,first,block_last(start,units),positive,below_value,reached)
  +plane_rows = Nat.sub(Nat.div(block_last(start,units),ow),first)
  +moved_rows = Nat.add(Nat.sub(Nat.div(value,ow),first),Nat.div(Nat.add(Nat.mod(value,ow),7n),ow))
  +covered_value = Order.transitive(Nat.mul(Nat.add(Nat.mul(moved_rows,stride),taps),tiles),Nat.mul(Nat.add(Nat.mul(plane_rows,stride),taps),tiles),elements,
    RowMajor.mul_le(Nat.add(Nat.mul(moved_rows,stride),taps),Nat.add(Nat.mul(plane_rows,stride),taps),tiles,
      Order.add_le(Nat.mul(moved_rows,stride),Nat.mul(plane_rows,stride),taps,RowMajor.mul_le(moved_rows,plane_rows,stride,rows))),covered)
  +back = Equal.sym(Nat,U32.to_nat(position),value,exact)
  {Reach{geometry,sixteen,positive,width_bound,lanes_room,stride_positive,stride_bound,ordered,row_exact,width_fits,columns_room,values_room,
    %back : {Nat.is_le(first,Nat.div(_,ow)) == True{} : Bool} below_value,
    %back : {Nat.is_le(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.sub(Nat.div(_,ow),first),Nat.div(Nat.add(Nat.mod(_,ow),7n),ow)),stride),taps),tiles),elements) == True{} : Bool}
      covered_value,
    %back : {Nat.is_le(Nat.add(_,8n),limit) == True{} : Bool} step_room} : Reach<window,planes,position,depth,cap>}

# ---- The packed tiles ----

def inner_agree(+window: Packing.Stencil) -> {SpatialSamples.inner_count(window) == U32.to_nat(Passes.inner_of(window)) : Nat}:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  {==}

# The tiles of a pack of the transposed planes of input cells, and the spec's
# packing of the values of the cells into the same tiles.
def packed_of(+window: Packing.Stencil,+planes: Packing.Planes,+count: Nat,+start: U32,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>) ->
  Tree.Tree<Tile.Tile8>:
  Model.packed_tiles(window,planes,count,start,Model.transposed(window,planes,input,pairs),tiles)

def filled_of(+window: Packing.Stencil,+units: Nat,+start: U32,+input: Tree.Tree<Tile.Tile8>,+tiles: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  SpatialSamples.fill(Nat.mul(units,SpatialSamples.inner_count(window)),window_values(window,input),window,start,tiles)

# The inner extent of a pack: its entries times its kernel columns, never zero.
def inner_exact_of(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: Passes.PackRoom<window,units,depth>) ->
  {SpatialSamples.inner_count(window) == Nat.mul(Model.pass_entries(window),Passes.tap_columns_of(window)) : Nat}:
  Equal.trans(Nat,SpatialSamples.inner_count(window),U32.to_nat(Passes.inner_of(window)),Nat.mul(Model.pass_entries(window),Passes.tap_columns_of(window)),inner_agree(window),
    Passes.pack_inner(window,units,depth,room))

def inner_positive_of(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: Passes.PackRoom<window,units,depth>) ->
  {Nat.is_lt(0n,SpatialSamples.inner_count(window)) == True{} : Bool}:
  +entries = Model.pass_entries(window)
  +columns = Passes.tap_columns_of(window)
  %Equal.sym(Nat,SpatialSamples.inner_count(window),Nat.mul(entries,columns),inner_exact_of(window,units,depth,room)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.product_positive(entries,columns,Passes.pack_entries(window,units,depth,room),Passes.pack_columns(window,units,depth,room))

# The unit of a tile is its block of inner tiles.
def unit_exact_of(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: Passes.PackRoom<window,units,depth>,+query: Nat) ->
  {Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window)) == Nat.div(query,SpatialSamples.inner_count(window)) : Nat}:
  +entries = Model.pass_entries(window)
  +columns = Passes.tap_columns_of(window)
  +inner = SpatialSamples.inner_count(window)
  Equal.trans(Nat,Passes.unit_of(query,entries,columns),Nat.div(query,Nat.mul(entries,columns)),Nat.div(query,inner),
    Equal.sym(Nat,Nat.div(query,Nat.mul(entries,columns)),Passes.unit_of(query,entries,columns),
      DivisionOrder.product_quotient(query,entries,columns,Passes.pack_entries(window,units,depth,room),Passes.pack_columns(window,units,depth,room))),
    Equal.cong(Nat,Nat,divisor => Nat.div(query,divisor),Nat.mul(entries,columns),inner,Equal.sym(Nat,inner,Nat.mul(entries,columns),inner_exact_of(window,units,depth,room))))

# The spec's packing of the units fits the tiles.
def filled_room(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: Passes.PackRoom<window,units,depth>,+tiles: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles)) ->
  {Nat.is_le(Nat.mul(units,SpatialSamples.inner_count(window)),U32.to_nat(Tree.capacity(Tile.Tile8,tiles))) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(Tile.Tile8,tiles),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,tiles,balanced)) :
    {Nat.is_le(Nat.mul(units,SpatialSamples.inner_count(window)),U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(Nat,SpatialSamples.inner_count(window),Nat.mul(Model.pass_entries(window),Passes.tap_columns_of(window)),inner_exact_of(window,units,depth,room)) :
    {Nat.is_le(Nat.mul(units,_),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
  Passes.pack_fits(window,units,depth,room)

# A tile of a unit of the pack is the spec's entry.
def packed_owned(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,
  +count: Nat,+start: U32,+units: Nat,+depth: Nat,+pair_depth: Nat,+cap: U32,+room: Passes.PackRoom<window,units,depth>,
  +units_exact: {Model.unit_count(window,count) == units : Nat},+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),
  +entries_exact: {Model.pass_entries(window) == Nat.mul(Planes.channel_count(window),Passes.tap_rows_of(window)) : Nat},+values_room: ValuesRoom<window,cap>,
  +block: Block<window,planes,start,units,pair_depth,cap>,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+query: Nat,
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +owned: {Nat.is_lt(Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window)),units) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,packed_of(window,planes,count,start,input,pairs,tiles),U32.from_nat(query)) ==
    Tree.read(Tile.Tile8,filled_of(window,units,start,input,tiles),U32.from_nat(query)) : Tile.Tile8}:
  +inner = SpatialSamples.inner_count(window)
  +values = window_values(window,input)
  +total = Nat.mul(units,inner)
  +owning = Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window))
  +unit_equal = unit_exact_of(window,units,depth,room,query)
  +block_below = {%unit_equal : {Nat.is_lt(_,units) == True{} : Bool} owned : {Nat.is_lt(Nat.div(query,inner),units) == True{} : Bool}}
  +within = Order.lt_le(query,Nat.mul(1n+Nat.div(query,inner),inner),total,DivisionOrder.next_above(query,inner,inner_positive_of(window,units,depth,room)),
    RowMajor.mul_le(1n+Nat.div(query,inner),units,inner,Order.strict_follow(Nat.div(query,inner),units,block_below)))
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,packed_of(window,planes,count,start,input,pairs,tiles),U32.from_nat(query)),
    SpatialSamples.value_vector(values,window,Passes.position_of(start,owning),U32.from_nat(Nat.mod(query,U32.to_nat(Passes.inner_of(window))))),
    Tree.read(Tile.Tile8,filled_of(window,units,start,input,tiles),U32.from_nat(query)),
    pack_tile(input,pairs,tiles,window,planes,count,start,units,depth,room,units_exact,balanced,query,query_inside,owned,pair_depth,cap,
      unit_reach(window,planes,start,units,pair_depth,cap,block,owning,owned),pairs_balanced,entries_exact,values_room),{
      %Equal.sym(Nat,owning,Nat.div(query,inner),unit_equal) :
        {SpatialSamples.value_vector(values,window,Passes.position_of(start,_),U32.from_nat(Nat.mod(query,U32.to_nat(Passes.inner_of(window))))) ==
          Tree.read(Tile.Tile8,filled_of(window,units,start,input,tiles),U32.from_nat(query)) : Tile.Tile8}
      %inner_agree(window) :
        {SpatialSamples.value_vector(values,window,Passes.position_of(start,Nat.div(query,inner)),U32.from_nat(Nat.mod(query,_))) ==
          Tree.read(Tile.Tile8,filled_of(window,units,start,input,tiles),U32.from_nat(query)) : Tile.Tile8}
      Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,filled_of(window,units,start,input,tiles),U32.from_nat(query)),
        SpatialSamples.packed_entry(SpatialSamples.Entries{values,window,start},query),
        SpatialSamples.fill_entry(total,values,window,start,depth,tiles,balanced,limit,filled_room(window,units,depth,room,tiles,balanced),query,within)) :
        {SpatialSamples.value_vector(values,window,Passes.position_of(start,owning),U32.from_nat(Nat.mod(query,U32.to_nat(Passes.inner_of(window))))) ==
          Tree.read(Tile.Tile8,filled_of(window,units,start,input,tiles),U32.from_nat(query)) : Tile.Tile8}})

# A tile past the units of the pack is kept by both.
def packed_kept(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,
  +count: Nat,+start: U32,+units: Nat,+depth: Nat,+pair_depth: Nat,+cap: U32,+room: Passes.PackRoom<window,units,depth>,
  +units_exact: {Model.unit_count(window,count) == units : Nat},+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),
  +entries_exact: {Model.pass_entries(window) == Nat.mul(Planes.channel_count(window),Passes.tap_rows_of(window)) : Nat},+values_room: ValuesRoom<window,cap>,
  +block: Block<window,planes,start,units,pair_depth,cap>,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+query: Nat,
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +outside: {Nat.is_lt(Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window)),units) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,packed_of(window,planes,count,start,input,pairs,tiles),U32.from_nat(query)) ==
    Tree.read(Tile.Tile8,filled_of(window,units,start,input,tiles),U32.from_nat(query)) : Tile.Tile8}:
  +inner = SpatialSamples.inner_count(window)
  +total = Nat.mul(units,inner)
  +owning = Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window))
  +transposed = Model.transposed(window,planes,input,pairs)
  +past = {%unit_exact_of(window,units,depth,room,query) : {Nat.is_le(units,_) == True{} : Bool}
    Order.not_less(owning,units,outside) : {Nat.is_le(units,Nat.div(query,inner)) == True{} : Bool}}
  +after = Order.transitive(total,Nat.mul(Nat.div(query,inner),inner),query,RowMajor.mul_le(units,Nat.div(query,inner),inner,past),
    DivisionOrder.product_below(query,inner,inner_positive_of(window,units,depth,room)))
  +kept = Tree.read(Tile.Tile8,tiles,U32.from_nat(query))
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,packed_of(window,planes,count,start,input,pairs,tiles),U32.from_nat(query)),kept,
    Tree.read(Tile.Tile8,filled_of(window,units,start,input,tiles),U32.from_nat(query)),
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,packed_of(window,planes,count,start,input,pairs,tiles),U32.from_nat(query)),
      Bool.pick(Tile.Tile8,Nat.is_lt(owning,units),Passes.pack_value(transposed,tiles,window,planes,start,query),kept),kept,
      Passes.pack_read(window,planes,count,start,transposed,tiles,units,depth,room,units_exact,balanced,query,query_inside),
      %Equal.sym(Bool,Nat.is_lt(owning,units),False{},outside) : {Bool.pick(Tile.Tile8,_,Passes.pack_value(transposed,tiles,window,planes,start,query),kept) == kept : Tile.Tile8}
      {==}),
    Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,filled_of(window,units,start,input,tiles),U32.from_nat(query)),kept,
      SpatialSamples.fill_kept(total,window_values(window,input),window,start,depth,tiles,balanced,limit,filled_room(window,units,depth,room,tiles,balanced),query,
        %Equal.sym(U32,Tree.capacity(Tile.Tile8,tiles),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,tiles,balanced)) : {Nat.is_lt(query,U32.to_nat(_)) == True{} : Bool}
          query_inside,after)))

# Tile query of the pack is tile query of the spec's packing.
def packed_case(owned: Bool,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,
  +count: Nat,+start: U32,+units: Nat,+depth: Nat,+pair_depth: Nat,+cap: U32,+room: Passes.PackRoom<window,units,depth>,
  +units_exact: {Model.unit_count(window,count) == units : Nat},+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),
  +entries_exact: {Model.pass_entries(window) == Nat.mul(Planes.channel_count(window),Passes.tap_rows_of(window)) : Nat},+values_room: ValuesRoom<window,cap>,
  +block: Block<window,planes,start,units,pair_depth,cap>,+limit: {Nat.is_le(depth,30n) == True{} : Bool},+query: Nat,
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +exact: {Nat.is_lt(Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window)),units) == owned : Bool}) ->
  {Tree.read(Tile.Tile8,packed_of(window,planes,count,start,input,pairs,tiles),U32.from_nat(query)) ==
    Tree.read(Tile.Tile8,filled_of(window,units,start,input,tiles),U32.from_nat(query)) : Tile.Tile8}:
  match owned:
    case True{}: packed_owned(input,pairs,tiles,window,planes,count,start,units,depth,pair_depth,cap,room,units_exact,balanced,pairs_balanced,entries_exact,values_room,block,limit,query,query_inside,exact)
    case False{}: packed_kept(input,pairs,tiles,window,planes,count,start,units,depth,pair_depth,cap,room,units_exact,balanced,pairs_balanced,entries_exact,values_room,block,limit,query,query_inside,exact)

# Everything a pack of a block reads, for the spec's packing to be its tiles.
type PackCase is Data:
  PackCase{input: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>,window: Packing.Stencil,planes: Packing.Planes,count: Nat,start: U32,
    units: Nat,depth: Nat,pair_depth: Nat,cap: U32,room: Passes.PackRoom<window,units,depth>,units_exact: {Model.unit_count(window,count) == units : Nat},
    balanced: Shape.Balanced(Tile.Tile8,depth,tiles),pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),
    entries_exact: {Model.pass_entries(window) == Nat.mul(Planes.channel_count(window),Passes.tap_rows_of(window)) : Nat},values_room: ValuesRoom<window,cap>,
    block: Block<window,planes,start,units,pair_depth,cap>,limit: {Nat.is_le(depth,30n) == True{} : Bool}}

def case_packed(packing: PackCase) -> Tree.Tree<Tile.Tile8>:
  PackCase{input,pairs,tiles,window,planes,count,start,units,depth,pair_depth,cap,room,units_exact,balanced,pairs_balanced,entries_exact,values_room,block,limit} = packing
  packed_of(window,planes,count,start,input,pairs,tiles)

def case_filled(packing: PackCase) -> Tree.Tree<Tile.Tile8>:
  PackCase{input,pairs,tiles,window,planes,count,start,units,depth,pair_depth,cap,room,units_exact,balanced,pairs_balanced,entries_exact,values_room,block,limit} = packing
  filled_of(window,units,start,input,tiles)

def case_depth(packing: PackCase) -> Nat:
  PackCase{input,pairs,tiles,window,planes,count,start,units,depth,pair_depth,cap,room,units_exact,balanced,pairs_balanced,entries_exact,values_room,block,limit} = packing
  depth

# Every tile inside the tree of a pack case is the spec's.
def case_same(+packing: PackCase,+index: U32,+inside: {U32.is_lt(index,Address.capacity(case_depth(packing))) == True{} : Bool}) ->
  {Tree.fetch(Tile.Tile8,case_packed(packing),Address.capacity(case_depth(packing)),index) == Tree.fetch(Tile.Tile8,case_filled(packing),Address.capacity(case_depth(packing)),index) :
    Tile.Tile8}:
  PackCase{+input,+pairs,+tiles,+window,+planes,+count,+start,+units,+depth,+pair_depth,+cap,+room,+units_exact,+balanced,+pairs_balanced,+entries_exact,+values_room,+block,
    +limit} = packing
  +query = U32.to_nat(index)
  +full_limit = Order.successor(depth,30n,limit)
  +query_inside = {
    %Equal.sym(Nat,U32.to_nat(Address.capacity(depth)),Shape.extent(depth),Address.capacity_value(depth,full_limit)) : {Nat.is_lt(query,_) == True{} : Bool}
    Address.natural_bound(depth,index,full_limit,inside) : {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}}
  +packed = packed_of(window,planes,count,start,input,pairs,tiles)
  +filled = filled_of(window,units,start,input,tiles)
  +packed_balanced = Passes.pack_balanced(window,planes,count,start,Model.transposed(window,planes,input,pairs),tiles,units,depth,room,units_exact,balanced)
  +filled_balanced = SpatialSamples.fill_balanced(Nat.mul(units,SpatialSamples.inner_count(window)),window_values(window,input),window,start,depth,tiles,balanced)
  Equal.trans(Tile.Tile8,Tree.fetch(Tile.Tile8,packed,Address.capacity(depth),index),Tree.read(Tile.Tile8,packed,index),Tree.fetch(Tile.Tile8,filled,Address.capacity(depth),index),
    Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,packed,index),Tree.fetch(Tile.Tile8,packed,Address.capacity(depth),index),
      TreeEquality.read_fetch(~Tile.Tile8,depth,packed,index,packed_balanced,inside)),
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,packed,index),Tree.read(Tile.Tile8,filled,index),Tree.fetch(Tile.Tile8,filled,Address.capacity(depth),index),
      %Arithmetic.native_roundtrip(index) : {Tree.read(Tile.Tile8,packed,_) == Tree.read(Tile.Tile8,filled,_) : Tile.Tile8}
      packed_case(Nat.is_lt(Passes.unit_of(query,Model.pass_entries(window),Passes.tap_columns_of(window)),units),input,pairs,tiles,window,planes,count,start,units,depth,
        pair_depth,cap,room,units_exact,balanced,pairs_balanced,entries_exact,values_room,block,limit,query,query_inside,{==}),
      TreeEquality.read_fetch(~Tile.Tile8,depth,filled,index,filled_balanced,inside)))

# The pack of a block from the transposed planes of input cells is the spec's
# packing of the values of the cells: the value vectors of the block's
# positions at every reduction index, the tiles past them kept.
def pack_fill(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+count: Nat,
  +start: U32,+units: Nat,+depth: Nat,+pair_depth: Nat,+cap: U32,+room: Passes.PackRoom<window,units,depth>,+units_exact: {Model.unit_count(window,count) == units : Nat},
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),
  +entries_exact: {Model.pass_entries(window) == Nat.mul(Planes.channel_count(window),Passes.tap_rows_of(window)) : Nat},+values_room: ValuesRoom<window,cap>,
  +block: Block<window,planes,start,units,pair_depth,cap>,+limit: {Nat.is_le(depth,30n) == True{} : Bool}) ->
  {packed_of(window,planes,count,start,input,pairs,tiles) == filled_of(window,units,start,input,tiles) : Tree.Tree<Tile.Tile8>}:
  +packing = {PackCase{input,pairs,tiles,window,planes,count,start,units,depth,pair_depth,cap,room,units_exact,balanced,pairs_balanced,entries_exact,values_room,block,limit} : PackCase}
  TreeEquality.same_trees(~Tile.Tile8,~PackCase,~case_packed,~case_filled,~case_depth,~case_same,packing,
    Passes.pack_balanced(window,planes,count,start,Model.transposed(window,planes,input,pairs),tiles,units,depth,room,units_exact,balanced),
    SpatialSamples.fill_balanced(Nat.mul(units,SpatialSamples.inner_count(window)),window_values(window,input),window,start,depth,tiles,balanced),
    Order.successor(depth,30n,limit))
