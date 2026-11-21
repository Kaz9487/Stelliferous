# The pairs a shared rectangle builds from bands of its held weights are the
# pairs of a copy of those weights.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_move.bend as Move
import ../tensor_point_products.bend as Products
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./kernel_gemm_certificate.bend as GemmCertificate
import ./kernel_gemm_output_groups.bend as Groups
import ./kernel_gemm_point_buffer.bend as PointBuffer
import ./kernel_gemm_read_into.bend as ReadInto
import ./machine_limit_bounds.bend as Limits
import ./nat_ceiling.bend as Ceiling
import ./nat_division_order.bend as DivisionOrder
import ./nat_extrema.bend as NatExtrema
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./storage_address_bounds.bend as Address
import ./storage_allocation_refinement.bend as Allocation
import ./storage_buffer_model.bend as Buffer
import ./storage_readback.bend as StorageReadback
import ./storage_shape.bend as Shape
import ./storage_tree_equality.bend as TreeEquality
import ./storage_tree_model.bend as Tree
import ./tensor_allocation_depth.bend as Depth
import ./tensor_halo_windows.bend as Halo
import ./tensor_move_readback.bend as Held
import ./tensor_point_product_operands.bend as Operands
import ./tensor_point_product_windows.bend as PointWindows
import ./tensor_point_products.bend as Point
import ./traversal_update_readback.bend as Updates
import ./u32_extrema.bend as U32Extrema
import ./word_arithmetic.bend as WordArithmetic
import ./word_power_of_two.bend as Power

# ---- Where a band writes ----

# A band from target writes position at target + position.
def placed_word(+source: U32,+target: U32,+start: Nat,+position: Nat,+exact: {U32.to_nat(target) == start : Nat}) ->
  {Gemm.tile_placed(Gemm.TilePlan{source,target},U32.from_nat(position)) == U32.from_nat(Nat.add(start,position)) : U32}:
  %Arithmetic.from_nat_exact(target,start,exact) : {(_ + U32.from_nat(position) : U32) == U32.from_nat(Nat.add(start,position)) : U32}
  WordArithmetic.native_add(start,position)

# Positions from position on of a band at start write past a query before them.
def misses_after(count: Nat,+position: Nat,+source: U32,+target: U32,+start: Nat,+query: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(target) == start : Nat},+below: {Nat.is_lt(query,Nat.add(start,position)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.add(start,position),count),U32.to_nat(cap)) == True{} : Bool}) ->
  Updates.Misses(~Gemm.TilePlan,~Gemm.tile_placed,count,U32.from_nat(position),Gemm.TilePlan{source,target},U32.from_nat(query)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +address = Nat.add(start,position)
      +address_room = Bounds.strict_le(address,U32.to_nat(cap),Order.room_index(address,rest,U32.to_nat(cap),room))
      +query_room = Order.transitive(query,address,U32.to_nat(cap),Bounds.strict_le(query,address,below),address_room)
      +moved = Equal.sym(Nat,Nat.add(start,1n+position),1n+address,Addition.successor_right(start,position))
      Gemm.Both{
        %Equal.sym(U32,Gemm.tile_placed(Gemm.TilePlan{source,target},U32.from_nat(position)),U32.from_nat(address),placed_word(source,target,start,position,exact)) :
          {U32.is_eq(_,U32.from_nat(query)) == False{} : Bool}
        Updates.words_apart(address,query,cap,cap_ok,address_room,query_room,
          %Equal.sym(Bool,Nat.is_eq(address,query),Nat.is_eq(query,address),Order.equal_symmetric(address,query)) : {_ == False{} : Bool}
          Order.less_not_equal(query,address,below)),
        %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
          Updates.Misses(~Gemm.TilePlan,~Gemm.tile_placed,rest,_,Gemm.TilePlan{source,target},U32.from_nat(query))
        misses_after(rest,1n+position,source,target,start,query,cap,cap_ok,exact,
          %moved : {Nat.is_lt(query,_) == True{} : Bool} Order.lt_succ(query,address,Bounds.strict_le(query,address,below)),
          %moved : {Nat.is_le(Nat.add(_,rest),U32.to_nat(cap)) == True{} : Bool} Order.advance_room(address,rest,U32.to_nat(cap),room))}

# Positions from position on of a band at start write before a query past them.
def misses_before(count: Nat,+position: Nat,+source: U32,+target: U32,+start: Nat,+query: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(target) == start : Nat},+past: {Nat.is_le(Nat.add(Nat.add(start,position),count),query) == True{} : Bool},
  +query_room: {Nat.is_le(query,U32.to_nat(cap)) == True{} : Bool}) ->
  Updates.Misses(~Gemm.TilePlan,~Gemm.tile_placed,count,U32.from_nat(position),Gemm.TilePlan{source,target},U32.from_nat(query)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +address = Nat.add(start,position)
      +address_below = Order.lt_le(address,Nat.add(address,1n+rest),query,Order.below_sum(address,rest),past)
      +address_room = Bounds.strict_le(address,U32.to_nat(cap),Order.lt_le(address,query,U32.to_nat(cap),address_below,query_room))
      +moved = Equal.sym(Nat,Nat.add(start,1n+position),1n+address,Addition.successor_right(start,position))
      Gemm.Both{
        %Equal.sym(U32,Gemm.tile_placed(Gemm.TilePlan{source,target},U32.from_nat(position)),U32.from_nat(address),placed_word(source,target,start,position,exact)) :
          {U32.is_eq(_,U32.from_nat(query)) == False{} : Bool}
        Updates.words_apart(address,query,cap,cap_ok,address_room,query_room,Order.less_not_equal(address,query,address_below)),
        %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
          Updates.Misses(~Gemm.TilePlan,~Gemm.tile_placed,rest,_,Gemm.TilePlan{source,target},U32.from_nat(query))
        misses_before(rest,1n+position,source,target,start,query,cap,cap_ok,exact,
          %moved : {Nat.is_le(Nat.add(_,rest),query) == True{} : Bool}
            %Addition.successor_right(address,rest) : {Nat.is_le(_,query) == True{} : Bool} past,
          query_room)}

# The tap before of positions from position on lies among them.
def hit_inside(+start: Nat,+position: Nat,+before: Nat,+count: Nat,+inside: {Nat.is_lt(before,count) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(start,Nat.add(before,position)),Nat.add(Nat.add(start,position),count)) == True{} : Bool}:
  %Equal.trans(Nat,Nat.add(Nat.add(start,position),before),Nat.add(start,Nat.add(position,before)),Nat.add(start,Nat.add(before,position)),
    Addition.associative(start,position,before),
    Equal.cong(Nat,Nat,value => Nat.add(start,value),Nat.add(position,before),Nat.add(before,position),Order.commutative(position,before))) :
    {Nat.is_lt(_,Nat.add(Nat.add(start,position),count)) == True{} : Bool}
  Order.offset_lt(Nat.add(start,position),before,count,inside)

# Of the positions from position on, only the one before positions later
# writes start + before + position.
def hits(count: Nat,+position: Nat,before: Nat,+source: U32,+target: U32,+start: Nat,+query: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(target) == start : Nat},+query_exact: {Nat.add(start,Nat.add(before,position)) == query : Nat},
  +room: {Nat.is_le(Nat.add(Nat.add(start,position),count),U32.to_nat(cap)) == True{} : Bool},+inside: {Nat.is_lt(before,count) == True{} : Bool}) ->
  Updates.Hits(~Gemm.TilePlan,~Gemm.tile_placed,count,U32.from_nat(position),Gemm.TilePlan{source,target},U32.from_nat(query),U32.from_nat(Nat.add(before,position)),before):
  match count:
    case 0n: Order.below_zero(before,inside)
    case 1n+ +rest:
      match before:
        case 0n:
          +address = Nat.add(start,position)
          +moved = Equal.sym(Nat,Nat.add(start,1n+position),1n+address,Addition.successor_right(start,position))
          +next_room = {%moved : {Nat.is_le(Nat.add(_,rest),U32.to_nat(cap)) == True{} : Bool} Order.advance_room(address,rest,U32.to_nat(cap),room) :
            {Nat.is_le(Nat.add(Nat.add(start,1n+position),rest),U32.to_nat(cap)) == True{} : Bool}}
          +placed = Equal.sym(U32,Gemm.tile_placed(Gemm.TilePlan{source,target},U32.from_nat(position)),U32.from_nat(address),placed_word(source,target,start,position,exact))
          Gemm.Both{{==},
            Gemm.Both{
              %placed : {U32.is_eq(_,U32.from_nat(query)) == True{} : Bool}
              %query_exact : {U32.is_eq(U32.from_nat(address),U32.from_nat(_)) == True{} : Bool}
              Arithmetic.equal_reflexive(U32.from_nat(address)),
              %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
                Updates.Misses(~Gemm.TilePlan,~Gemm.tile_placed,rest,_,Gemm.TilePlan{source,target},U32.from_nat(query))
              misses_after(rest,1n+position,source,target,start,query,cap,cap_ok,exact,
                %query_exact : {Nat.is_lt(_,Nat.add(start,1n+position)) == True{} : Bool}
                  %moved : {Nat.is_lt(address,_) == True{} : Bool} Order.lt_succ(address,address,Order.reflexive(address)),
                next_room)}}
        case 1n+ +earlier:
          +address = Nat.add(start,position)
          +moved = Equal.sym(Nat,Nat.add(start,1n+position),1n+address,Addition.successor_right(start,position))
          +next_room = {%moved : {Nat.is_le(Nat.add(_,rest),U32.to_nat(cap)) == True{} : Bool} Order.advance_room(address,rest,U32.to_nat(cap),room) :
            {Nat.is_le(Nat.add(Nat.add(start,1n+position),rest),U32.to_nat(cap)) == True{} : Bool}}
          +placed = Equal.sym(U32,Gemm.tile_placed(Gemm.TilePlan{source,target},U32.from_nat(position)),U32.from_nat(address),placed_word(source,target,start,position,exact))
          +query_below = {%query_exact : {Nat.is_lt(_,Nat.add(address,1n+rest)) == True{} : Bool} hit_inside(start,position,1n+earlier,1n+rest,inside) :
            {Nat.is_lt(query,Nat.add(address,1n+rest)) == True{} : Bool}}
          +query_room = Bounds.strict_le(query,U32.to_nat(cap),Order.lt_le(query,Nat.add(address,1n+rest),U32.to_nat(cap),query_below,room))
          +address_below = {%query_exact : {Nat.is_lt(address,_) == True{} : Bool}
            Order.offset_lt(start,position,1n+Nat.add(earlier,position),Order.lt_succ(position,Nat.add(earlier,position),Order.right_le(earlier,position))) :
            {Nat.is_lt(address,query) == True{} : Bool}}
          +address_room = Bounds.strict_le(address,U32.to_nat(cap),Order.lt_le(address,query,U32.to_nat(cap),address_below,query_room))
          Gemm.Both{
            %placed : {U32.is_eq(_,U32.from_nat(query)) == False{} : Bool}
            Updates.words_apart(address,query,cap,cap_ok,address_room,query_room,Order.less_not_equal(address,query,address_below)),
            %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
              Updates.Hits(~Gemm.TilePlan,~Gemm.tile_placed,rest,_,Gemm.TilePlan{source,target},U32.from_nat(query),U32.from_nat(Nat.add(1n+earlier,position)),earlier)
            %Addition.successor_right(earlier,position) :
              Updates.Hits(~Gemm.TilePlan,~Gemm.tile_placed,rest,U32.from_nat(1n+position),Gemm.TilePlan{source,target},U32.from_nat(query),U32.from_nat(_),earlier)
            hits(rest,1n+position,earlier,source,target,start,query,cap,cap_ok,exact,
              %Equal.sym(Nat,Nat.add(earlier,1n+position),1n+Nat.add(earlier,position),Addition.successor_right(earlier,position)) :
                {Nat.add(start,_) == query : Nat} query_exact,
              next_room,inside)}

# Every position from position on of a band writes inside the capacity.
def destinations(count: Nat,+position: Nat,+source: U32,+target: U32,+start: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(target) == start : Nat},+room: {Nat.is_le(Nat.add(Nat.add(start,position),count),U32.to_nat(capacity)) == True{} : Bool}) ->
  Updates.Destinations(~Gemm.TilePlan,~Gemm.tile_placed,count,U32.from_nat(position),Gemm.TilePlan{source,target},capacity):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +address = Nat.add(start,position)
      +below = Order.room_index(address,rest,U32.to_nat(capacity),room)
      +moved = Equal.sym(Nat,Nat.add(start,1n+position),1n+address,Addition.successor_right(start,position))
      Gemm.Both{
        %Equal.sym(U32,Gemm.tile_placed(Gemm.TilePlan{source,target},U32.from_nat(position)),U32.from_nat(address),placed_word(source,target,start,position,exact)) :
          {U32.is_lt(_,capacity) == True{} : Bool}
        Bounds.native_bound(address,capacity,Bounds.roundtrip(address,capacity,capacity_ok,Bounds.strict_le(address,U32.to_nat(capacity),below)),below),
        %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
          Updates.Destinations(~Gemm.TilePlan,~Gemm.tile_placed,rest,_,Gemm.TilePlan{source,target},capacity)
        destinations(rest,1n+position,source,target,start,capacity,capacity_ok,exact,
          %moved : {Nat.is_le(Nat.add(_,rest),U32.to_nat(capacity)) == True{} : Bool} Order.advance_room(address,rest,U32.to_nat(capacity),room))}

# ---- One band into the pairs ----

# The pairs after count tiles of a band, each combined into its pair by half.
def band_loop(~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,count: Nat,plan: Gemm.TilePlan,reader: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>) ->
  Tree.Tree<Tile.Tile16>:
  ReadInto.update_loop(~Tile.Tile8,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~PointBuffer.tile_sample,~half,~Gemm.tile_placed,count,plan,reader,pairs)

# A band leaves a pair it does not write as it was.
def band_kept(~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+count: Nat,+source: U32,+target: U32,+start: Nat,+reader: Tree.Tree<Tile.Tile8>,
  +pairs: Tree.Tree<Tile.Tile16>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,pairs),2147483648) == True{} : Bool},+exact: {U32.to_nat(target) == start : Nat},
  +room: {Nat.is_le(Nat.add(start,count),U32.to_nat(Tree.capacity(Tile.Tile16,pairs))) == True{} : Bool},
  +query: Nat,+query_inside: {Nat.is_lt(query,U32.to_nat(Tree.capacity(Tile.Tile16,pairs))) == True{} : Bool},
  +misses: Updates.Misses(~Gemm.TilePlan,~Gemm.tile_placed,count,0,Gemm.TilePlan{source,target},U32.from_nat(query))) ->
  {Tree.read(Tile.Tile16,band_loop(~half,count,Gemm.TilePlan{source,target},reader,pairs),U32.from_nat(query)) == Tree.read(Tile.Tile16,pairs,U32.from_nat(query)) : Tile.Tile16}:
  +capacity = Tree.capacity(Tile.Tile16,pairs)
  +plan = {Gemm.TilePlan{source,target} : Gemm.TilePlan}
  Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,band_loop(~half,count,plan,reader,pairs),U32.from_nat(query)),
    Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~PointBuffer.tile_sample,~half,~Gemm.tile_placed,count,0,reader,plan,U32.from_nat(query),
      Tree.read(Tile.Tile16,pairs,U32.from_nat(query))),
    Tree.read(Tile.Tile16,pairs,U32.from_nat(query)),
    Updates.update_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~PointBuffer.tile_sample,~half,~Gemm.tile_placed,count,0,reader,plan,pairs,depth,
      U32.from_nat(query),balanced,limit,
      Bounds.native_bound(query,capacity,Bounds.roundtrip(query,capacity,capacity_ok,Bounds.strict_le(query,U32.to_nat(capacity),query_inside)),query_inside),
      destinations(count,0n,source,target,start,capacity,capacity_ok,exact,
        %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) : {Nat.is_le(Nat.add(_,count),U32.to_nat(capacity)) == True{} : Bool} room)),
    Updates.missed(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~PointBuffer.tile_sample,~half,~Gemm.tile_placed,count,0,reader,plan,U32.from_nat(query),
      Tree.read(Tile.Tile16,pairs,U32.from_nat(query)),misses))

# A band combines its tile tap into the pair at start + tap.
def band_written(~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+count: Nat,+source: U32,+target: U32,+start: Nat,+reader: Tree.Tree<Tile.Tile8>,
  +pairs: Tree.Tree<Tile.Tile16>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,pairs),2147483648) == True{} : Bool},+exact: {U32.to_nat(target) == start : Nat},
  +room: {Nat.is_le(Nat.add(start,count),U32.to_nat(Tree.capacity(Tile.Tile16,pairs))) == True{} : Bool},
  +tap: Nat,+tap_inside: {Nat.is_lt(tap,count) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,band_loop(~half,count,Gemm.TilePlan{source,target},reader,pairs),U32.from_nat(Nat.add(start,tap))) ==
    half(Tree.read(Tile.Tile16,pairs,U32.from_nat(Nat.add(start,tap))),PointBuffer.tile_sample(reader,Gemm.TilePlan{source,target},U32.from_nat(tap))) : Tile.Tile16}:
  +capacity = Tree.capacity(Tile.Tile16,pairs)
  +plan = {Gemm.TilePlan{source,target} : Gemm.TilePlan}
  +query = Nat.add(start,tap)
  +room_zero = {%Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) : {Nat.is_le(Nat.add(_,count),U32.to_nat(capacity)) == True{} : Bool} room :
    {Nat.is_le(Nat.add(Nat.add(start,0n),count),U32.to_nat(capacity)) == True{} : Bool}}
  +query_inside = Order.lt_le(query,Nat.add(start,count),U32.to_nat(capacity),Order.offset_lt(start,tap,count,tap_inside),room)
  Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,band_loop(~half,count,plan,reader,pairs),U32.from_nat(query)),
    Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~PointBuffer.tile_sample,~half,~Gemm.tile_placed,count,0,reader,plan,U32.from_nat(query),
      Tree.read(Tile.Tile16,pairs,U32.from_nat(query))),
    half(Tree.read(Tile.Tile16,pairs,U32.from_nat(query)),PointBuffer.tile_sample(reader,plan,U32.from_nat(tap))),
    Updates.update_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~PointBuffer.tile_sample,~half,~Gemm.tile_placed,count,0,reader,plan,pairs,depth,
      U32.from_nat(query),balanced,limit,
      Bounds.native_bound(query,capacity,Bounds.roundtrip(query,capacity,capacity_ok,Bounds.strict_le(query,U32.to_nat(capacity),query_inside)),query_inside),
      destinations(count,0n,source,target,start,capacity,capacity_ok,exact,room_zero)),
    %Addition.zero_right(tap) :
      {Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~PointBuffer.tile_sample,~half,~Gemm.tile_placed,count,0,reader,plan,U32.from_nat(query),
        Tree.read(Tile.Tile16,pairs,U32.from_nat(query))) == half(Tree.read(Tile.Tile16,pairs,U32.from_nat(query)),PointBuffer.tile_sample(reader,plan,U32.from_nat(_))) : Tile.Tile16}
    Updates.hit(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~PointBuffer.tile_sample,~half,~Gemm.tile_placed,count,0,reader,plan,U32.from_nat(query),
      U32.from_nat(Nat.add(tap,0n)),tap,Tree.read(Tile.Tile16,pairs,U32.from_nat(query)),
      hits(count,0n,tap,source,target,start,query,capacity,capacity_ok,exact,
        Equal.cong(Nat,Nat,value => Nat.add(start,value),Nat.add(tap,0n),tap,Addition.zero_right(tap)),room_zero,tap_inside)))

# ---- The state between bands ----

def Lent() -> Type:
  Move.Holding<Tile.Tile8> & Move.Spare<Tile.Tile8>

def Run() -> Type:
  Lent() & Storage.Buffer<Tile.Tile16>

# The pairs so far, of a depth and length, and their value at query.
type PairsAt<-depth: Nat,-length: Nat,-query: Nat,-pairs: Storage.Buffer<Tile.Tile16>,-current: Tile.Tile16> is Data:
  PairsAt{model: Buffer.Model<Tile.Tile16>,packed: {pairs == Buffer.pack(~Tile.Tile16,model) : Storage.Buffer<Tile.Tile16>},
    balanced: Shape.Balanced(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,model)),long: {Buffer.length(~Tile.Tile16,model) == length : Nat},
    read: {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,model),U32.from_nat(query)) == current : Tile.Tile16}}

# The held weights, the spare and the pairs between bands, as models, and the
# value of the pairs at query.
type Packed<-Source: Data,-value: Source -> U32 -> Tile.Tile8,-source: Source,-low: Nat,-high: Nat,-depth: Nat,-length: Nat,-query: Nat,-state: Run(),
  -current: Tile.Tile16> is Data:
  Packed{holding: Held.Holding<Tile.Tile8>,spare: Held.SpareModel<Tile.Tile8>,pairs: Buffer.Model<Tile.Tile16>,
    packed: {state == ((Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare)),Buffer.pack(~Tile.Tile16,pairs)) : Run()},
    tiled: Held.Tiled(~Tile.Tile8,holding,low,high),values: Held.HoldingValues(~Tile.Tile8,~Source,~value,source,holding),room: Held.SpareRoom(~Tile.Tile8,spare),
    balanced: Shape.Balanced(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,pairs)),long: {Buffer.length(~Tile.Tile16,pairs) == length : Nat},
    read: {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,pairs),U32.from_nat(query)) == current : Tile.Tile16}}

# Where a query lies against the count pairs of a band from start.
type Where<-start: Nat,-count: Nat,-query: Nat> is Data:
  Before{below: {Nat.is_lt(query,start) == True{} : Bool}}
  Past{past: {Nat.is_le(Nat.add(start,count),query) == True{} : Bool}}
  At{tap: Nat,exact: {Nat.add(start,tap) == query : Nat},inside: {Nat.is_lt(tap,count) == True{} : Bool}}

# The value a band from first of the weights leaves at the query.
def band_effect(~Source: Data,~value: Source -> U32 -> Tile.Tile8,~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+source: Source,+first: Nat,+start: Nat,+count: Nat,
  +query: Nat,place: Where<start,count,query>,current: Tile.Tile16) -> Tile.Tile16:
  match place:
    case Before{below}: current
    case Past{past}: current
    case At{tap,exact,inside}: half(current,value(source,U32.from_nat(Nat.add(first,tap))))

# ---- One band ----

def lent_band(+first: U32,+inner: U32,+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>) -> Move.Band<Tile.Tile8>:
  Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),first,inner,Products.zero(),Held.pack_spare(~Tile.Tile8,spare))

def band_update(~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+target: U32,+inner: U32,band: Move.Band<Tile.Tile8>,pairs: Storage.Buffer<Tile.Tile16>) ->
  Products.Cells() & Storage.Buffer<Tile.Tile16>:
  Move.Band{lent,buffer,+offset} = band
  GemmBuffer.updated_into(~Tile.Tile8,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~Gemm.tile_read,~GemmCertificate.tile_read,~half,~Gemm.tile_placed,
    U32.to_nat(inner),Gemm.TilePlan{offset,target},buffer,pairs)

def returned_parts(lent: Move.Lent<Tile.Tile8>,result: Products.Cells() & Storage.Buffer<Tile.Tile16>) ->
  {Products.band_returned(lent,result) == (Move.returned(~Tile.Tile8,lent,Pair.fst(Products.Cells(),Storage.Buffer<Tile.Tile16>,result)),
    Pair.snd(Products.Cells(),Storage.Buffer<Tile.Tile16>,result)) : Run()}:
  (buffer,pairs) = result
  {==}

# A band half is the band back in its holding and the updated pairs.
def band_parts(~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+target: U32,+inner: U32,band: Move.Band<Tile.Tile8>,pairs: Storage.Buffer<Tile.Tile16>) ->
  {Products.band_half(~half,target,inner,band,pairs) ==
    (Move.returned(~Tile.Tile8,Held.band_loan(~Tile.Tile8,band),Pair.fst(Products.Cells(),Storage.Buffer<Tile.Tile16>,band_update(~half,target,inner,band,pairs))),
      Pair.snd(Products.Cells(),Storage.Buffer<Tile.Tile16>,band_update(~half,target,inner,band,pairs))) : Run()}:
  Move.Band{lent,buffer,+offset} = band
  returned_parts(lent,GemmBuffer.updated_into(~Tile.Tile8,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~Gemm.tile_read,~GemmCertificate.tile_read,~half,~Gemm.tile_placed,
    U32.to_nat(inner),Gemm.TilePlan{offset,target},buffer,pairs))

def contents_parts(result: Products.Cells() & Storage.Buffer<Tile.Tile16>) ->
  {ReadInto.buffers_contents(~Tile.Tile8,~Tile.Tile16,result) ==
    (Buffer.contents(~Tile.Tile8,Pair.fst(Products.Cells(),Storage.Buffer<Tile.Tile16>,result)),
      Buffer.contents(~Tile.Tile16,Pair.snd(Products.Cells(),Storage.Buffer<Tile.Tile16>,result))) : Buffer.Content<Tile.Tile8> & Buffer.Content<Tile.Tile16>}:
  (buffer,pairs) = result
  {==}

def band_plan(+target: U32,band: Move.Band<Tile.Tile8>) -> Gemm.TilePlan:
  Gemm.TilePlan{Held.band_offset(~Tile.Tile8,band),target}

def band_reader(band: Move.Band<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,band)))

def band_offset_parts(band: Move.Band<Tile.Tile8>) -> {band == Move.Band{Held.band_loan(~Tile.Tile8,band),Held.band_buffer(~Tile.Tile8,band),Held.band_offset(~Tile.Tile8,band)} : Move.Band<Tile.Tile8>}:
  Move.Band{lent,buffer,offset} = band
  {==}

# The buffers an update of a lent band leaves: the band as it was and the
# pairs of the model loop.
def band_contents(~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+target: U32,+inner: U32,+first: U32,+holding: Held.Holding<Tile.Tile8>,
  +spare: Held.SpareModel<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>) ->
  {ReadInto.buffers_contents(~Tile.Tile8,~Tile.Tile16,band_update(~half,target,inner,lent_band(first,inner,holding,spare),Buffer.pack(~Tile.Tile16,pairs))) ==
    (Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,lent_band(first,inner,holding,spare)))),
      Buffer.Content{band_loop(~half,U32.to_nat(inner),band_plan(target,lent_band(first,inner,holding,spare)),band_reader(lent_band(first,inner,holding,spare)),
        Buffer.tree(~Tile.Tile16,pairs)),Buffer.length(~Tile.Tile16,pairs)}) : Buffer.Content<Tile.Tile8> & Buffer.Content<Tile.Tile16>}:
  -band = lent_band(first,inner,holding,spare)
  -source = Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,band))
  %Equal.sym(Move.Band<Tile.Tile8>,lent_band(first,inner,holding,spare),
      Move.Band{Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),Held.band_buffer(~Tile.Tile8,lent_band(first,inner,holding,spare)),
        Held.band_offset(~Tile.Tile8,lent_band(first,inner,holding,spare))},
      band_offset_parts(lent_band(first,inner,holding,spare))) :
    {ReadInto.buffers_contents(~Tile.Tile8,~Tile.Tile16,band_update(~half,target,inner,_,Buffer.pack(~Tile.Tile16,pairs))) ==
      (Buffer.content(~Tile.Tile8,source),Buffer.Content{band_loop(~half,U32.to_nat(inner),band_plan(target,band),band_reader(band),Buffer.tree(~Tile.Tile16,pairs)),
        Buffer.length(~Tile.Tile16,pairs)}) : Buffer.Content<Tile.Tile8> & Buffer.Content<Tile.Tile16>}
  %Buffer.remodeled(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,lent_band(first,inner,holding,spare))) :
    {ReadInto.buffers_contents(~Tile.Tile8,~Tile.Tile16,GemmBuffer.updated_into(~Tile.Tile8,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~Gemm.tile_read,~GemmCertificate.tile_read,
      ~half,~Gemm.tile_placed,U32.to_nat(inner),band_plan(target,band),_,Buffer.pack(~Tile.Tile16,pairs))) ==
      (Buffer.content(~Tile.Tile8,source),Buffer.Content{band_loop(~half,U32.to_nat(inner),band_plan(target,band),band_reader(band),Buffer.tree(~Tile.Tile16,pairs)),
        Buffer.length(~Tile.Tile16,pairs)}) : Buffer.Content<Tile.Tile8> & Buffer.Content<Tile.Tile16>}
  ReadInto.update_model(~Tile.Tile8,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~Gemm.tile_read,~GemmCertificate.tile_read,~PointBuffer.tile_sample,~PointBuffer.tile_matches,
    ~half,~Gemm.tile_placed,U32.to_nat(inner),band_plan(target,lent_band(first,inner,holding,spare)),
    Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,lent_band(first,inner,holding,spare))),pairs)

def offset_word(+offset: U32,+tap: Nat) -> {(offset + U32.from_nat(tap) : U32) == U32.from_nat(Nat.add(U32.to_nat(offset),tap)) : U32}:
  %Arithmetic.native_roundtrip(offset) : {(_ + U32.from_nat(tap) : U32) == U32.from_nat(Nat.add(U32.to_nat(offset),tap)) : U32}
  WordArithmetic.native_add(U32.to_nat(offset),tap)

# The tile tap of a lent band is the weight at first + tap.
def tile_value(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+low: Nat,+high: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(cap)) == True{} : Bool},+first: U32,+inner: U32,+target: U32,
  +holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+spare_room: Held.SpareRoom(~Tile.Tile8,spare),
  +tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Source,~value,source,holding),
  +after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},+fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(inner)),high) == True{} : Bool},
  +allocatable: {U32.is_le(inner,1073741824) == True{} : Bool},+tap: Nat,+inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool}) ->
  {PointBuffer.tile_sample(band_reader(lent_band(first,inner,holding,spare)),band_plan(target,lent_band(first,inner,holding,spare)),U32.from_nat(tap)) ==
    value(source,U32.from_nat(Nat.add(U32.to_nat(first),tap))) : Tile.Tile8}:
  +offset = Held.band_offset(~Tile.Tile8,lent_band(first,inner,holding,spare))
  %Equal.sym(U32,(offset + U32.from_nat(tap) : U32),U32.from_nat(Nat.add(U32.to_nat(offset),tap)),offset_word(offset,tap)) :
    {Tree.read(Tile.Tile8,band_reader(lent_band(first,inner,holding,spare)),_) == value(source,U32.from_nat(Nat.add(U32.to_nat(first),tap))) : Tile.Tile8}
  Held.band_model_value(~Tile.Tile8,~Source,~value,source,holding,first,inner,Products.zero(),spare,spare_room,low,high,cap,cap_ok,bounded,tiled,values,after,fits,
    allocatable,tap,inside)

def halves_equal(~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+left: Tile.Tile16,+right: Tile.Tile16,+low: Tile.Tile8,+high: Tile.Tile8,
  +pairs_same: {left == right : Tile.Tile16},+tiles_same: {low == high : Tile.Tile8}) -> {half(left,low) == half(right,high) : Tile.Tile16}:
  %pairs_same : {half(left,low) == half(_,high) : Tile.Tile16}
  %tiles_same : {half(left,low) == half(left,_) : Tile.Tile16}
  {==}

# The read a lent band leaves at the query.
def band_read(~Source: Data,~value: Source -> U32 -> Tile.Tile8,~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+source: Source,+low: Nat,+high: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(cap)) == True{} : Bool},
  +first: U32,+inner: U32,+target: U32,+start: Nat,+exact: {U32.to_nat(target) == start : Nat},
  +holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+spare_room: Held.SpareRoom(~Tile.Tile8,spare),
  +tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Source,~value,source,holding),
  +after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},+fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(inner)),high) == True{} : Bool},
  +allocatable: {U32.is_le(inner,1073741824) == True{} : Bool},
  +pairs: Tree.Tree<Tile.Tile16>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,pairs),2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,U32.to_nat(inner)),U32.to_nat(Tree.capacity(Tile.Tile16,pairs))) == True{} : Bool},
  +query: Nat,+query_inside: {Nat.is_lt(query,U32.to_nat(Tree.capacity(Tile.Tile16,pairs))) == True{} : Bool},
  +current: Tile.Tile16,+read: {Tree.read(Tile.Tile16,pairs,U32.from_nat(query)) == current : Tile.Tile16},+place: Where<start,U32.to_nat(inner),query>) ->
  {Tree.read(Tile.Tile16,band_loop(~half,U32.to_nat(inner),band_plan(target,lent_band(first,inner,holding,spare)),band_reader(lent_band(first,inner,holding,spare)),pairs),
    U32.from_nat(query)) == band_effect(~Source,~value,~half,source,U32.to_nat(first),start,U32.to_nat(inner),query,place,current) : Tile.Tile16}:
  match place:
    case Before{+below}:
      +offset = Held.band_offset(~Tile.Tile8,lent_band(first,inner,holding,spare))
      +zero_start = Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start))
      Equal.trans(Tile.Tile16,
        Tree.read(Tile.Tile16,band_loop(~half,U32.to_nat(inner),Gemm.TilePlan{offset,target},band_reader(lent_band(first,inner,holding,spare)),pairs),U32.from_nat(query)),
        Tree.read(Tile.Tile16,pairs,U32.from_nat(query)),current,
        band_kept(~half,U32.to_nat(inner),offset,target,start,band_reader(lent_band(first,inner,holding,spare)),pairs,depth,balanced,limit,capacity_ok,exact,room,
          query,query_inside,
          misses_after(U32.to_nat(inner),0n,offset,target,start,query,Tree.capacity(Tile.Tile16,pairs),capacity_ok,exact,
            %zero_start : {Nat.is_lt(query,_) == True{} : Bool} below,
            %zero_start : {Nat.is_le(Nat.add(_,U32.to_nat(inner)),U32.to_nat(Tree.capacity(Tile.Tile16,pairs))) == True{} : Bool} room)),
        read)
    case Past{+past}:
      +offset = Held.band_offset(~Tile.Tile8,lent_band(first,inner,holding,spare))
      +zero_start = Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start))
      Equal.trans(Tile.Tile16,
        Tree.read(Tile.Tile16,band_loop(~half,U32.to_nat(inner),Gemm.TilePlan{offset,target},band_reader(lent_band(first,inner,holding,spare)),pairs),U32.from_nat(query)),
        Tree.read(Tile.Tile16,pairs,U32.from_nat(query)),current,
        band_kept(~half,U32.to_nat(inner),offset,target,start,band_reader(lent_band(first,inner,holding,spare)),pairs,depth,balanced,limit,capacity_ok,exact,room,
          query,query_inside,
          misses_before(U32.to_nat(inner),0n,offset,target,start,query,Tree.capacity(Tile.Tile16,pairs),capacity_ok,exact,
            %zero_start : {Nat.is_le(Nat.add(_,U32.to_nat(inner)),query) == True{} : Bool} past,
            Bounds.strict_le(query,U32.to_nat(Tree.capacity(Tile.Tile16,pairs)),query_inside))),
        read)
    case At{+tap,+tap_exact,+inside}:
      +plan = band_plan(target,lent_band(first,inner,holding,spare))
      +reader = band_reader(lent_band(first,inner,holding,spare))
      +at = Nat.add(start,tap)
      %tap_exact : {Tree.read(Tile.Tile16,band_loop(~half,U32.to_nat(inner),plan,reader,pairs),U32.from_nat(_)) ==
        half(current,value(source,U32.from_nat(Nat.add(U32.to_nat(first),tap)))) : Tile.Tile16}
      Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,band_loop(~half,U32.to_nat(inner),plan,reader,pairs),U32.from_nat(at)),
        half(Tree.read(Tile.Tile16,pairs,U32.from_nat(at)),PointBuffer.tile_sample(reader,plan,U32.from_nat(tap))),
        half(current,value(source,U32.from_nat(Nat.add(U32.to_nat(first),tap)))),
        band_written(~half,U32.to_nat(inner),Held.band_offset(~Tile.Tile8,lent_band(first,inner,holding,spare)),target,start,reader,pairs,depth,balanced,limit,capacity_ok,exact,room,tap,inside),
        halves_equal(~half,Tree.read(Tile.Tile16,pairs,U32.from_nat(at)),current,PointBuffer.tile_sample(reader,plan,U32.from_nat(tap)),
          value(source,U32.from_nat(Nat.add(U32.to_nat(first),tap))),
          %Equal.sym(Nat,at,query,tap_exact) : {Tree.read(Tile.Tile16,pairs,U32.from_nat(_)) == current : Tile.Tile16} read,
          tile_value(~Source,~value,source,low,high,cap,cap_ok,bounded,first,inner,target,holding,spare,spare_room,tiled,values,after,fits,allocatable,tap,inside)))

# A state of a holding covered after a band, its spare and its pairs is
# packed.
def packed_after(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+low: Nat,+high: Nat,+layout: Held.Spans,+depth: Nat,+length: Nat,
  +query: Nat,+current: Tile.Tile16,-state: Run(),-holding_after: Move.Holding<Tile.Tile8>,-spare_after: Move.Spare<Tile.Tile8>,-written: Storage.Buffer<Tile.Tile16>,
  +parts: {state == ((holding_after,spare_after),written) : Run()},
  +covered: Held.CoveredFact(~Tile.Tile8,~Source,~value,source,holding_after,low,high,layout),
  +spare: Held.SpareModel<Tile.Tile8>,+spared: {spare_after == Held.pack_spare(~Tile.Tile8,spare) : Move.Spare<Tile.Tile8>},+room: Held.SpareRoom(~Tile.Tile8,spare),
  +pairs: Buffer.Model<Tile.Tile16>,+paired: {written == Buffer.pack(~Tile.Tile16,pairs) : Storage.Buffer<Tile.Tile16>},
  +balanced: Shape.Balanced(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,pairs)),+long: {Buffer.length(~Tile.Tile16,pairs) == length : Nat},
  +read: {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,pairs),U32.from_nat(query)) == current : Tile.Tile16}) ->
  Packed<Source,value,source,low,high,depth,length,query,state,current>:
  Held.Covered{+model,+held_packed,+tiled,+values,+ordered} = covered
  Packed{model,spare,pairs,
    %held_packed : {state == ((_,Held.pack_spare(~Tile.Tile8,spare)),Buffer.pack(~Tile.Tile16,pairs)) : Run()}
    %spared : {state == ((holding_after,_),Buffer.pack(~Tile.Tile16,pairs)) : Run()}
    %paired : {state == ((holding_after,spare_after),_) : Run()}
    parts,
    tiled,values,room,balanced,long,read}

def lent_buffer(+first: U32,+inner: U32,+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Held.band_buffer(~Tile.Tile8,lent_band(first,inner,holding,spare))

def lent_update(~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+target: U32,+inner: U32,+first: U32,+holding: Held.Holding<Tile.Tile8>,
  +spare: Held.SpareModel<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>) -> Products.Cells() & Storage.Buffer<Tile.Tile16>:
  band_update(~half,target,inner,lent_band(first,inner,holding,spare),Buffer.pack(~Tile.Tile16,pairs))

def lent_returned(~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+target: U32,+inner: U32,+first: U32,+holding: Held.Holding<Tile.Tile8>,
  +spare: Held.SpareModel<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>) -> Storage.Buffer<Tile.Tile8>:
  Pair.fst(Products.Cells(),Storage.Buffer<Tile.Tile16>,lent_update(~half,target,inner,first,holding,spare,pairs))

def lent_pairs(~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+target: U32,+inner: U32,+first: U32,+holding: Held.Holding<Tile.Tile8>,
  +spare: Held.SpareModel<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>) -> Storage.Buffer<Tile.Tile16>:
  Pair.snd(Products.Cells(),Storage.Buffer<Tile.Tile16>,lent_update(~half,target,inner,first,holding,spare,pairs))

# A band lent from held weights into the pairs leaves the next state.
def band_core(~Source: Data,~value: Source -> U32 -> Tile.Tile8,~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+source: Source,+low: Nat,+high: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(cap)) == True{} : Bool},
  +depth: Nat,+limit: {Nat.is_le(depth,31n) == True{} : Bool},+length: Nat,+query: Nat,+current: Tile.Tile16,
  +first: U32,+inner: U32,+target: U32,+start: Nat,+exact: {U32.to_nat(target) == start : Nat},
  +after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},+fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(inner)),high) == True{} : Bool},
  +allocatable: {U32.is_le(inner,1073741824) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,U32.to_nat(inner)),U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+place: Where<start,U32.to_nat(inner),query>,
  +holding: Held.Holding<Tile.Tile8>,+tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Source,~value,source,holding),
  +spare: Held.SpareModel<Tile.Tile8>,+spare_room: Held.SpareRoom(~Tile.Tile8,spare),
  +pairs: Buffer.Model<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,pairs)),+long: {Buffer.length(~Tile.Tile16,pairs) == length : Nat},
  +read: {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,pairs),U32.from_nat(query)) == current : Tile.Tile16}) ->
  Packed<Source,value,source,low,high,depth,length,query,Products.band_half(~half,target,inner,lent_band(first,inner,holding,spare),Buffer.pack(~Tile.Tile16,pairs)),
    band_effect(~Source,~value,~half,source,U32.to_nat(first),start,U32.to_nat(inner),query,place,current)>:
  +effect = band_effect(~Source,~value,~half,source,U32.to_nat(first),start,U32.to_nat(inner),query,place,current)
  +loop = band_loop(~half,U32.to_nat(inner),band_plan(target,lent_band(first,inner,holding,spare)),band_reader(lent_band(first,inner,holding,spare)),Buffer.tree(~Tile.Tile16,pairs))
  +both = Equal.trans(Buffer.Content<Tile.Tile8> & Buffer.Content<Tile.Tile16>,(Buffer.contents(~Tile.Tile8,lent_returned(~half,target,inner,first,holding,spare,pairs)),Buffer.contents(~Tile.Tile16,lent_pairs(~half,target,inner,first,holding,spare,pairs))),
    ReadInto.buffers_contents(~Tile.Tile8,~Tile.Tile16,lent_update(~half,target,inner,first,holding,spare,pairs)),
    (Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,lent_buffer(first,inner,holding,spare))),Buffer.Content{loop,Buffer.length(~Tile.Tile16,pairs)}),
    Equal.sym(Buffer.Content<Tile.Tile8> & Buffer.Content<Tile.Tile16>,ReadInto.buffers_contents(~Tile.Tile8,~Tile.Tile16,lent_update(~half,target,inner,first,holding,spare,pairs)),
      (Buffer.contents(~Tile.Tile8,lent_returned(~half,target,inner,first,holding,spare,pairs)),Buffer.contents(~Tile.Tile16,lent_pairs(~half,target,inner,first,holding,spare,pairs))),contents_parts(lent_update(~half,target,inner,first,holding,spare,pairs))),
    band_contents(~half,target,inner,first,holding,spare,pairs))
  +returned_same = Equal.cong(Buffer.Content<Tile.Tile8> & Buffer.Content<Tile.Tile16>,Buffer.Content<Tile.Tile8>,
    contents => Pair.fst(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile16>,contents),
    (Buffer.contents(~Tile.Tile8,lent_returned(~half,target,inner,first,holding,spare,pairs)),Buffer.contents(~Tile.Tile16,lent_pairs(~half,target,inner,first,holding,spare,pairs))),
    (Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,lent_buffer(first,inner,holding,spare))),Buffer.Content{loop,Buffer.length(~Tile.Tile16,pairs)}),both)
  +written_same = Equal.cong(Buffer.Content<Tile.Tile8> & Buffer.Content<Tile.Tile16>,Buffer.Content<Tile.Tile16>,
    contents => Pair.snd(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile16>,contents),
    (Buffer.contents(~Tile.Tile8,lent_returned(~half,target,inner,first,holding,spare,pairs)),Buffer.contents(~Tile.Tile16,lent_pairs(~half,target,inner,first,holding,spare,pairs))),
    (Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,lent_buffer(first,inner,holding,spare))),Buffer.Content{loop,Buffer.length(~Tile.Tile16,pairs)}),both)
  +back = Buffer.remodel(~Tile.Tile8,lent_returned(~half,target,inner,first,holding,spare,pairs))
  +same_band = Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,back),Buffer.contents(~Tile.Tile8,lent_returned(~half,target,inner,first,holding,spare,pairs)),Buffer.contents(~Tile.Tile8,lent_buffer(first,inner,holding,spare)),
    Buffer.remodel_content(~Tile.Tile8,lent_returned(~half,target,inner,first,holding,spare,pairs)),
    Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,lent_returned(~half,target,inner,first,holding,spare,pairs)),Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,lent_buffer(first,inner,holding,spare))),Buffer.contents(~Tile.Tile8,lent_buffer(first,inner,holding,spare)),
      returned_same,Buffer.remodel_content(~Tile.Tile8,lent_buffer(first,inner,holding,spare))))
  +layout = Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding))
  +next_spare = Held.spare_after(~Tile.Tile8,holding,first,inner,spare,back)
  +after_pairs = Buffer.remodel(~Tile.Tile16,lent_pairs(~half,target,inner,first,holding,spare,pairs))
  +after_content = Equal.trans(Buffer.Content<Tile.Tile16>,Buffer.content(~Tile.Tile16,after_pairs),Buffer.contents(~Tile.Tile16,lent_pairs(~half,target,inner,first,holding,spare,pairs)),Buffer.Content{loop,Buffer.length(~Tile.Tile16,pairs)},
    Buffer.remodel_content(~Tile.Tile16,lent_pairs(~half,target,inner,first,holding,spare,pairs)),written_same)
  +after_tree = Held.content_tree_is(~Tile.Tile16,after_pairs,Buffer.Content{loop,Buffer.length(~Tile.Tile16,pairs)},after_content)
  +after_length = Equal.trans(Nat,Buffer.length(~Tile.Tile16,after_pairs),Buffer.content_length(~Tile.Tile16,Buffer.content(~Tile.Tile16,after_pairs)),Buffer.length(~Tile.Tile16,pairs),
    Equal.sym(Nat,Buffer.content_length(~Tile.Tile16,Buffer.content(~Tile.Tile16,after_pairs)),Buffer.length(~Tile.Tile16,after_pairs),Buffer.content_length_of(~Tile.Tile16,after_pairs)),
    Equal.cong(Buffer.Content<Tile.Tile16>,Nat,content => Buffer.content_length(~Tile.Tile16,content),Buffer.content(~Tile.Tile16,after_pairs),
      Buffer.Content{loop,Buffer.length(~Tile.Tile16,pairs)},after_content))
  +capacity = Shape.capacity(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,pairs),balanced)
  +unshift = Equal.sym(U32,Tree.capacity(Tile.Tile16,Buffer.tree(~Tile.Tile16,pairs)),U32.shln(1,depth),capacity)
  packed_after(~Source,~value,source,low,high,layout,depth,length,query,effect,
    Products.band_half(~half,target,inner,lent_band(first,inner,holding,spare),Buffer.pack(~Tile.Tile16,pairs)),Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),lent_returned(~half,target,inner,first,holding,spare,pairs)),Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),lent_returned(~half,target,inner,first,holding,spare,pairs)),lent_pairs(~half,target,inner,first,holding,spare,pairs),
    Equal.trans(Run(),Products.band_half(~half,target,inner,lent_band(first,inner,holding,spare),Buffer.pack(~Tile.Tile16,pairs)),(Move.returned(~Tile.Tile8,Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),lent_returned(~half,target,inner,first,holding,spare,pairs)),lent_pairs(~half,target,inner,first,holding,spare,pairs)),
      ((Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),lent_returned(~half,target,inner,first,holding,spare,pairs)),Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),lent_returned(~half,target,inner,first,holding,spare,pairs))),lent_pairs(~half,target,inner,first,holding,spare,pairs)),
      band_parts(~half,target,inner,lent_band(first,inner,holding,spare),Buffer.pack(~Tile.Tile16,pairs)),
      Equal.cong(Lent(),Run(),held => (held,lent_pairs(~half,target,inner,first,holding,spare,pairs)),Move.returned(~Tile.Tile8,Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),lent_returned(~half,target,inner,first,holding,spare,pairs)),(Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),lent_returned(~half,target,inner,first,holding,spare,pairs)),Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),lent_returned(~half,target,inner,first,holding,spare,pairs))),
        Held.pair_eta(~Move.Holding<Tile.Tile8>,~Move.Spare<Tile.Tile8>,Move.returned(~Tile.Tile8,Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),lent_returned(~half,target,inner,first,holding,spare,pairs))))),
    %Buffer.remodeled(~Tile.Tile8,lent_returned(~half,target,inner,first,holding,spare,pairs)) : Held.CoveredFact(~Tile.Tile8,~Source,~value,source,Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),_),low,high,layout)
    Held.band_returned(~Tile.Tile8,~Source,~value,source,holding,first,inner,Products.zero(),spare,low,high,tiled,values,layout,{==},back,same_band),
    next_spare,
    %Buffer.remodeled(~Tile.Tile8,lent_returned(~half,target,inner,first,holding,spare,pairs)) : {Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,lent_band(first,inner,holding,spare)),_) == Held.pack_spare(~Tile.Tile8,next_spare) : Move.Spare<Tile.Tile8>}
    Held.band_spare(~Tile.Tile8,holding,first,inner,Products.zero(),spare,back),
    Held.band_spare_room(~Tile.Tile8,holding,first,inner,Products.zero(),spare,spare_room,allocatable,back,same_band),
    after_pairs,Equal.sym(Storage.Buffer<Tile.Tile16>,Buffer.pack(~Tile.Tile16,after_pairs),lent_pairs(~half,target,inner,first,holding,spare,pairs),Buffer.remodeled(~Tile.Tile16,lent_pairs(~half,target,inner,first,holding,spare,pairs))),
    %Equal.sym(Tree.Tree<Tile.Tile16>,Buffer.tree(~Tile.Tile16,after_pairs),loop,after_tree) : Shape.Balanced(Tile.Tile16,depth,_)
    Updates.update_balanced(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~PointBuffer.tile_sample,~half,~Gemm.tile_placed,U32.to_nat(inner),0,
      band_reader(lent_band(first,inner,holding,spare)),band_plan(target,lent_band(first,inner,holding,spare)),Buffer.tree(~Tile.Tile16,pairs),depth,balanced),
    Equal.trans(Nat,Buffer.length(~Tile.Tile16,after_pairs),Buffer.length(~Tile.Tile16,pairs),length,after_length,long),
    %Equal.sym(Tree.Tree<Tile.Tile16>,Buffer.tree(~Tile.Tile16,after_pairs),loop,after_tree) : {Tree.read(Tile.Tile16,_,U32.from_nat(query)) == effect : Tile.Tile16}
    band_read(~Source,~value,~half,source,low,high,cap,cap_ok,bounded,first,inner,target,start,exact,holding,spare,spare_room,tiled,values,after,fits,allocatable,
      Buffer.tree(~Tile.Tile16,pairs),depth,balanced,limit,
      %unshift : {U32.is_le(_,2147483648) == True{} : Bool} Power.native_power_bound(depth,limit),
      %unshift : {Nat.is_le(Nat.add(start,U32.to_nat(inner)),U32.to_nat(_)) == True{} : Bool} room,
      query,%unshift : {Nat.is_lt(query,U32.to_nat(_)) == True{} : Bool} query_inside,
      current,read,place))


# ---- The words of a pair ----

def lower_word(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+base: U32,+inner: U32,+pair: Nat,
  +double_room: {Nat.is_le(Nat.mul(pair,2n),U32.to_nat(cap)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(base),Nat.mul(Nat.mul(pair,2n),U32.to_nat(inner))),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((base + U32.from_nat(pair) * 2 * inner : U32)) == Nat.add(U32.to_nat(base),Nat.mul(Nat.mul(pair,2n),U32.to_nat(inner))) : Nat}:
  +double = Nat.mul(pair,2n)
  +scaled = Nat.mul(double,U32.to_nat(inner))
  +pair_room = Order.transitive(pair,double,U32.to_nat(cap),RowMajor.product_covers(pair,2n,{==}),double_room)
  +double_exact = Arithmetic.product_exact(cap,cap_ok,U32.from_nat(pair),2,pair,2n,Bounds.roundtrip(pair,cap,cap_ok,pair_room),{==},double_room)
  +scaled_exact = Arithmetic.product_exact(cap,cap_ok,(U32.from_nat(pair) * 2 : U32),inner,double,U32.to_nat(inner),double_exact,{==},
    Order.transitive(scaled,Nat.add(U32.to_nat(base),scaled),U32.to_nat(cap),Order.right_le(U32.to_nat(base),scaled),room))
  Arithmetic.sum_exact(cap,cap_ok,base,(U32.from_nat(pair) * 2 * inner : U32),U32.to_nat(base),scaled,{==},scaled_exact,room)

def upper_group(+pair: Nat,+groups: Nat) -> Nat:
  Nat.min(1n+Nat.mul(pair,2n),Nat.sub(groups,1n))

def upper_word(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+base: U32,+groups: U32,+inner: U32,+pair: Nat,
  +lower_fits: {Nat.is_le(1n+Nat.mul(pair,2n),U32.to_nat(groups)) == True{} : Bool},+groups_room: {Nat.is_le(U32.to_nat(groups),U32.to_nat(cap)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(base),Nat.mul(upper_group(pair,U32.to_nat(groups)),U32.to_nat(inner))),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((base + U32.min((U32.from_nat(pair) * 2 + 1 : U32),(groups - 1 : U32)) * inner : U32)) ==
    Nat.add(U32.to_nat(base),Nat.mul(upper_group(pair,U32.to_nat(groups)),U32.to_nat(inner))) : Nat}:
  +count = U32.to_nat(groups)
  +double = Nat.mul(pair,2n)
  +last = Nat.sub(count,1n)
  +upper = upper_group(pair,count)
  +next_word = (U32.from_nat(pair) * 2 + 1 : U32)
  +last_word = (groups - 1 : U32)
  +groups_positive = Order.lt_le(0n,1n+double,count,{==},lower_fits)
  +double_cap = Order.transitive(double,count,U32.to_nat(cap),Order.transitive(double,1n+double,count,Order.successor(double,double,Order.reflexive(double)),lower_fits),groups_room)
  +pair_room = Order.transitive(pair,double,U32.to_nat(cap),RowMajor.product_covers(pair,2n,{==}),double_cap)
  +double_exact = Arithmetic.product_exact(cap,cap_ok,U32.from_nat(pair),2,pair,2n,Bounds.roundtrip(pair,cap,cap_ok,pair_room),{==},double_cap)
  +next_cap = {%Order.commutative(1n,double) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(1n+double,count,U32.to_nat(cap),lower_fits,groups_room) : {Nat.is_le(Nat.add(double,1n),U32.to_nat(cap)) == True{} : Bool}}
  +next_exact = {Equal.trans(Nat,U32.to_nat(next_word),Nat.add(double,1n),1n+double,
      Arithmetic.sum_exact(cap,cap_ok,(U32.from_nat(pair) * 2 : U32),1,double,1n,double_exact,{==},next_cap),Order.commutative(double,1n)) :
    {U32.to_nat(next_word) == 1n+double : Nat}}
  +last_exact = Arithmetic.difference_exact(cap,cap_ok,groups,1,count,1n,{==},{==},Order.strict_follow(0n,count,groups_positive),groups_room)
  +upper_exact = {%next_exact : {U32.to_nat(U32.min(next_word,last_word)) == Nat.min(_,last) : Nat}
    %last_exact : {U32.to_nat(U32.min(next_word,last_word)) == Nat.min(U32.to_nat(next_word),_) : Nat}
    U32Extrema.minimum(next_word,last_word) : {U32.to_nat(U32.min(next_word,last_word)) == upper : Nat}}
  +scaled = Nat.mul(upper,U32.to_nat(inner))
  +scaled_exact = Arithmetic.product_exact(cap,cap_ok,U32.min(next_word,last_word),inner,upper,U32.to_nat(inner),upper_exact,{==},
    Order.transitive(scaled,Nat.add(U32.to_nat(base),scaled),U32.to_nat(cap),Order.right_le(U32.to_nat(base),scaled),room))
  Arithmetic.sum_exact(cap,cap_ok,base,(U32.min(next_word,last_word) * inner : U32),U32.to_nat(base),scaled,{==},scaled_exact,room)

def target_word(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+inner: U32,+pair: Nat,
  +pair_room: {Nat.is_le(pair,U32.to_nat(cap)) == True{} : Bool},+room: {Nat.is_le(Nat.mul(pair,U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((U32.from_nat(pair) * inner : U32)) == Nat.mul(pair,U32.to_nat(inner)) : Nat}:
  Arithmetic.product_exact(cap,cap_ok,U32.from_nat(pair),inner,pair,U32.to_nat(inner),Bounds.roundtrip(pair,cap,cap_ok,pair_room),{==},room)

# A band of group group ends inside the weights of groups groups.
def band_fits(+base: Nat,+group: Nat,+groups: Nat,+width: Nat,+high: Nat,+inside: {Nat.is_le(1n+group,groups) == True{} : Bool},
  +reach: {Nat.is_le(Nat.add(base,Nat.mul(groups,width)),high) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(base,Nat.mul(group,width)),width),high) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(Nat.add(base,Nat.mul(group,width)),width),Nat.add(base,Nat.mul(1n+group,width)),
    Equal.trans(Nat,Nat.add(Nat.add(base,Nat.mul(group,width)),width),Nat.add(base,Nat.add(Nat.mul(group,width),width)),Nat.add(base,Nat.mul(1n+group,width)),
      Addition.associative(base,Nat.mul(group,width),width),
      Equal.cong(Nat,Nat,value => Nat.add(base,value),Nat.add(Nat.mul(group,width),width),Nat.add(width,Nat.mul(group,width)),Order.commutative(Nat.mul(group,width),width)))) :
    {Nat.is_le(_,high) == True{} : Bool}
  Order.transitive(Nat.add(base,Nat.mul(1n+group,width)),Nat.add(base,Nat.mul(groups,width)),high,
    Order.offset_le(base,Nat.mul(1n+group,width),Nat.mul(groups,width),RowMajor.mul_le(1n+group,groups,width,inside)),reach)

# The pair of pair before count pairs fits below 2 * pair + 1 groups.
def lower_inside(+pair: Nat,+pairs: Nat,+groups: Nat,+pairs_exact: {pairs == Ceiling.ceiling(groups,2n) : Nat},+inside: {Nat.is_lt(pair,pairs) == True{} : Bool}) ->
  {Nat.is_le(1n+Nat.mul(pair,2n),groups) == True{} : Bool}:
  Held.by_cases({Nat.is_le(1n+Nat.mul(pair,2n),groups) == True{} : Bool},Nat.is_lt(Nat.mul(pair,2n),groups),
    below => Order.strict_follow(Nat.mul(pair,2n),groups,below),
    beyond => Empty.absurd({Nat.is_le(1n+Nat.mul(pair,2n),groups) == True{} : Bool},
      Held.lt_le_absurd(pair,pairs,inside,
        %Equal.sym(Nat,pairs,Ceiling.ceiling(groups,2n),pairs_exact) : {Nat.is_le(_,pair) == True{} : Bool}
        Ceiling.minimal(groups,2n,pair,{==},Order.not_less(Nat.mul(pair,2n),groups,beyond)))))

def upper_inside(+pair: Nat,+groups: Nat,+lower_fits: {Nat.is_le(1n+Nat.mul(pair,2n),groups) == True{} : Bool}) ->
  {Nat.is_le(1n+upper_group(pair,groups),groups) == True{} : Bool}:
  +groups_positive = Order.lt_le(0n,1n+Nat.mul(pair,2n),groups,{==},lower_fits)
  %Order.positive_has_last(groups,groups_positive) : {Nat.is_le(1n+upper_group(pair,groups),_) == True{} : Bool}
  NatExtrema.minimum_right(1n+Nat.mul(pair,2n),Nat.sub(groups,1n))

# ---- The room of a shared rectangle ----

type PairRoom<-cap: U32,-high: Nat,-base: U32,-groups: U32,-pairs: Nat,-inner: U32,-depth: Nat> is Data:
  PairRoom{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},bounded: {Nat.is_le(high,U32.to_nat(cap)) == True{} : Bool},
    reach: {Nat.is_le(Nat.add(U32.to_nat(base),Nat.mul(U32.to_nat(groups),U32.to_nat(inner))),high) == True{} : Bool},
    groups_room: {Nat.is_le(U32.to_nat(groups),U32.to_nat(cap)) == True{} : Bool},
    pairs_exact: {pairs == Ceiling.ceiling(U32.to_nat(groups),2n) : Nat},
    limit: {Nat.is_le(depth,31n) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(pairs,U32.to_nat(inner)),U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
    allocatable: {U32.is_le(inner,1073741824) == True{} : Bool}}

# A query against the pairs: a tap of a pair, or past every pair.
type Query<-pairs: Nat,-width: Nat,-query: Nat> is Data:
  Inside{pair: Nat,tap: Nat,exact: {Nat.add(Nat.mul(pair,width),tap) == query : Nat},pair_inside: {Nat.is_lt(pair,pairs) == True{} : Bool},
    tap_inside: {Nat.is_lt(tap,width) == True{} : Bool}}
  Beyond{beyond: {Nat.is_le(Nat.mul(pairs,width),query) == True{} : Bool}}

# The pair of the weights at tap of pair: groups 2 pair and 2 pair + 1, the
# last group again for a missing upper one.
def pair_value(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+base: Nat,+groups: Nat,+width: Nat,+pair: Nat,+tap: Nat) -> Tile.Tile16:
  Tile.Tiles{value(source,U32.from_nat(Nat.add(base,Nat.add(Nat.mul(Nat.mul(pair,2n),width),tap)))),
    value(source,U32.from_nat(Nat.add(base,Nat.add(Nat.mul(upper_group(pair,groups),width),tap))))}

# The value at the query after the pairs below done.
def expected(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+base: Nat,+groups: Nat,+width: Nat,+pairs: Nat,+query: Nat,kind: Query<pairs,width,query>,
  +done: Nat) -> Tile.Tile16:
  match kind:
    case Inside{+pair,+tap,exact,pair_inside,tap_inside}:
      Bool.pick(Tile.Tile16,Nat.is_lt(pair,done),pair_value(~Source,~value,source,base,groups,width,pair,tap),Tile.filled_pair(0.0))
    case Beyond{beyond}: Tile.filled_pair(0.0)

# ---- One pair ----

def lower_first(+base: U32,+inner: U32,+pair: Nat) -> U32:
  (base + U32.from_nat(pair) * 2 * inner : U32)

def upper_first(+base: U32,+groups: U32,+inner: U32,+pair: Nat) -> U32:
  (base + U32.min((U32.from_nat(pair) * 2 + 1 : U32),(groups - 1 : U32)) * inner : U32)

def pair_target(+inner: U32,+pair: Nat) -> U32:
  (U32.from_nat(pair) * inner : U32)

# The state after the band of the lower group of pair.
def lowered(+base: U32,+inner: U32,+pair: Nat,+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>) -> Run():
  Products.band_half(~Gemm.lower_half,pair_target(inner,pair),inner,lent_band(lower_first(base,inner,pair),inner,holding,spare),Buffer.pack(~Tile.Tile16,pairs))

# The words of the two bands of a pair and their room.
type PairWords<-high: Nat,-base: U32,-groups: U32,-inner: U32,-depth: Nat,-pair: Nat> is Data:
  PairWords{target: {U32.to_nat(pair_target(inner,pair)) == Nat.mul(pair,U32.to_nat(inner)) : Nat},
    lower: {U32.to_nat(lower_first(base,inner,pair)) == Nat.add(U32.to_nat(base),Nat.mul(Nat.mul(pair,2n),U32.to_nat(inner))) : Nat},
    upper: {U32.to_nat(upper_first(base,groups,inner,pair)) == Nat.add(U32.to_nat(base),Nat.mul(upper_group(pair,U32.to_nat(groups)),U32.to_nat(inner))) : Nat},
    lower_after: {Nat.is_le(U32.to_nat(base),U32.to_nat(lower_first(base,inner,pair))) == True{} : Bool},
    lower_fit: {Nat.is_le(Nat.add(U32.to_nat(lower_first(base,inner,pair)),U32.to_nat(inner)),high) == True{} : Bool},
    upper_after: {Nat.is_le(U32.to_nat(base),U32.to_nat(upper_first(base,groups,inner,pair))) == True{} : Bool},
    upper_fit: {Nat.is_le(Nat.add(U32.to_nat(upper_first(base,groups,inner,pair)),U32.to_nat(inner)),high) == True{} : Bool},
    band_room: {Nat.is_le(Nat.add(Nat.mul(pair,U32.to_nat(inner)),U32.to_nat(inner)),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}}

def pair_words(+cap: U32,+high: Nat,+base: U32,+groups: U32,+total: Nat,+inner: U32,+depth: Nat,+room: PairRoom<cap,high,base,groups,total,inner,depth>,
  +pair: Nat,+pair_inside: {Nat.is_lt(pair,total) == True{} : Bool}) -> PairWords<high,base,groups,inner,depth,pair>:
  PairRoom{+cap_ok,+bounded,+reach,+groups_room,+pairs_exact,+limit,+pairs_room,+allocatable} = room
  +width = U32.to_nat(inner)
  +low = U32.to_nat(base)
  +count = U32.to_nat(groups)
  +start = Nat.mul(pair,width)
  +double = Nat.mul(pair,2n)
  +upper = upper_group(pair,count)
  +lower_fits = lower_inside(pair,total,count,pairs_exact,pair_inside)
  +lower_end = band_fits(low,double,count,width,high,lower_fits,reach)
  +upper_end = band_fits(low,upper,count,width,high,upper_inside(pair,count,lower_fits),reach)
  +lower_sum = Nat.add(low,Nat.mul(double,width))
  +upper_sum = Nat.add(low,Nat.mul(upper,width))
  +lower_room = Order.transitive(lower_sum,high,U32.to_nat(cap),Order.transitive(lower_sum,Nat.add(lower_sum,width),high,Order.le_add(lower_sum,width),lower_end),bounded)
  +upper_room = Order.transitive(upper_sum,high,U32.to_nat(cap),Order.transitive(upper_sum,Nat.add(upper_sum,width),high,Order.le_add(upper_sum,width),upper_end),bounded)
  +double_room = Order.transitive(double,count,U32.to_nat(cap),
    Order.transitive(double,1n+double,count,Order.successor(double,double,Order.reflexive(double)),lower_fits),groups_room)
  +pair_room = Order.transitive(pair,double,U32.to_nat(cap),RowMajor.product_covers(pair,2n,{==}),double_room)
  +start_room = Order.transitive(start,lower_sum,U32.to_nat(cap),
    Order.transitive(start,Nat.mul(double,width),lower_sum,RowMajor.mul_le(pair,double,width,RowMajor.product_covers(pair,2n,{==})),Order.right_le(low,Nat.mul(double,width))),
    lower_room)
  +lower_exact = lower_word(cap,cap_ok,base,inner,pair,double_room,lower_room)
  +upper_exact = upper_word(cap,cap_ok,base,groups,inner,pair,lower_fits,groups_room,upper_room)
  +lower_back = Equal.sym(Nat,U32.to_nat(lower_first(base,inner,pair)),lower_sum,lower_exact)
  +upper_back = Equal.sym(Nat,U32.to_nat(upper_first(base,groups,inner,pair)),upper_sum,upper_exact)
  PairWords{target_word(cap,cap_ok,inner,pair,pair_room,start_room),lower_exact,upper_exact,
    %lower_back : {Nat.is_le(low,_) == True{} : Bool} Order.le_add(low,Nat.mul(double,width)),
    %lower_back : {Nat.is_le(Nat.add(_,width),high) == True{} : Bool} lower_end,
    %upper_back : {Nat.is_le(low,_) == True{} : Bool} Order.le_add(low,Nat.mul(upper,width)),
    %upper_back : {Nat.is_le(Nat.add(_,width),high) == True{} : Bool} upper_end,
    %Order.commutative(width,start) : {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
      Order.transitive(Nat.mul(1n+pair,width),Nat.mul(total,width),U32.to_nat(U32.shln(1,depth)),
        RowMajor.mul_le(1n+pair,total,width,Order.strict_follow(pair,total,pair_inside)),pairs_room)}

def words_lower(+high: Nat,+base: U32,+groups: U32,+inner: U32,+depth: Nat,+pair: Nat,+words: PairWords<high,base,groups,inner,depth,pair>) ->
  {U32.to_nat(lower_first(base,inner,pair)) == Nat.add(U32.to_nat(base),Nat.mul(Nat.mul(pair,2n),U32.to_nat(inner))) : Nat}:
  PairWords{+target,+lower,+upper,+lower_after,+lower_fit,+upper_after,+upper_fit,+band_room} = words
  lower

def words_upper(+high: Nat,+base: U32,+groups: U32,+inner: U32,+depth: Nat,+pair: Nat,+words: PairWords<high,base,groups,inner,depth,pair>) ->
  {U32.to_nat(upper_first(base,groups,inner,pair)) == Nat.add(U32.to_nat(base),Nat.mul(upper_group(pair,U32.to_nat(groups)),U32.to_nat(inner))) : Nat}:
  PairWords{+target,+lower,+upper,+lower_after,+lower_fit,+upper_after,+upper_fit,+band_room} = words
  upper

# The band of the upper group of pair after the band of its lower group.
def upper_core(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+low: Nat,+high: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(cap)) == True{} : Bool},
  +depth: Nat,+limit: {Nat.is_le(depth,31n) == True{} : Bool},+length: Nat,+query: Nat,+current: Tile.Tile16,
  +base: U32,+groups: U32,+inner: U32,+pair: Nat,+start: Nat,+exact: {U32.to_nat(pair_target(inner,pair)) == start : Nat},
  +after: {Nat.is_le(low,U32.to_nat(upper_first(base,groups,inner,pair))) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(upper_first(base,groups,inner,pair)),U32.to_nat(inner)),high) == True{} : Bool},
  +allocatable: {U32.is_le(inner,1073741824) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,U32.to_nat(inner)),U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+place: Where<start,U32.to_nat(inner),query>,
  -state: Run(),+facts: Packed<Source,value,source,low,high,depth,length,query,state,current>) ->
  Packed<Source,value,source,low,high,depth,length,query,Products.upper_band(U32.from_nat(pair),groups,inner,base,state),
    band_effect(~Source,~value,~Gemm.upper_half,source,U32.to_nat(upper_first(base,groups,inner,pair)),start,U32.to_nat(inner),query,place,current)>:
  Packed{+holding,+spare,+pairs,+packed,+tiled,+values,+spare_room,+balanced,+long,+read} = facts
  -effect = band_effect(~Source,~value,~Gemm.upper_half,source,U32.to_nat(upper_first(base,groups,inner,pair)),start,U32.to_nat(inner),query,place,current)
  %Equal.sym(Run(),state,((Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare)),Buffer.pack(~Tile.Tile16,pairs)),packed) :
    Packed<Source,value,source,low,high,depth,length,query,Products.upper_band(U32.from_nat(pair),groups,inner,base,_),effect>
  band_core(~Source,~value,~Gemm.upper_half,source,low,high,cap,cap_ok,bounded,depth,limit,length,query,current,upper_first(base,groups,inner,pair),inner,
    pair_target(inner,pair),start,exact,after,fits,allocatable,room,query_inside,place,holding,tiled,values,spare,spare_room,pairs,balanced,long,read)

# Both bands of pair, each into its half of the pairs.
def pair_cores(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+high: Nat,+cap: U32,+base: U32,+groups: U32,+total: Nat,+inner: U32,
  +depth: Nat,+length: Nat,+query: Nat,+current: Tile.Tile16,+next: Tile.Tile16,+room: PairRoom<cap,high,base,groups,total,inner,depth>,
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+pair: Nat,+words: PairWords<high,base,groups,inner,depth,pair>,
  +place: Where<Nat.mul(pair,U32.to_nat(inner)),U32.to_nat(inner),query>,
  +transition: {band_effect(~Source,~value,~Gemm.upper_half,source,U32.to_nat(upper_first(base,groups,inner,pair)),Nat.mul(pair,U32.to_nat(inner)),U32.to_nat(inner),query,place,
    band_effect(~Source,~value,~Gemm.lower_half,source,U32.to_nat(lower_first(base,inner,pair)),Nat.mul(pair,U32.to_nat(inner)),U32.to_nat(inner),query,place,current)) ==
    next : Tile.Tile16},
  +holding: Held.Holding<Tile.Tile8>,+tiled: Held.Tiled(~Tile.Tile8,holding,U32.to_nat(base),high),+values: Held.HoldingValues(~Tile.Tile8,~Source,~value,source,holding),
  +spare: Held.SpareModel<Tile.Tile8>,+spare_room: Held.SpareRoom(~Tile.Tile8,spare),
  +pairs: Buffer.Model<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,pairs)),+long: {Buffer.length(~Tile.Tile16,pairs) == length : Nat},
  +read: {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,pairs),U32.from_nat(query)) == current : Tile.Tile16}) ->
  Packed<Source,value,source,U32.to_nat(base),high,depth,length,query,Products.upper_band(U32.from_nat(pair),groups,inner,base,lowered(base,inner,pair,holding,spare,pairs)),next>:
  PairRoom{+cap_ok,+bounded,+reach,+groups_room,+pairs_exact,+limit,+pairs_room,+allocatable} = room
  PairWords{+target_exact,+lower_exact,+upper_exact,+lower_after,+lower_fit,+upper_after,+upper_fit,+band_room} = words
  +width = U32.to_nat(inner)
  +low = U32.to_nat(base)
  +start = Nat.mul(pair,width)
  +lower_effect = band_effect(~Source,~value,~Gemm.lower_half,source,U32.to_nat(lower_first(base,inner,pair)),start,width,query,place,current)
  %transition : Packed<Source,value,source,low,high,depth,length,query,Products.upper_band(U32.from_nat(pair),groups,inner,base,lowered(base,inner,pair,holding,spare,pairs)),_>
  upper_core(~Source,~value,source,low,high,cap,cap_ok,bounded,depth,limit,length,query,lower_effect,base,groups,inner,pair,start,target_exact,upper_after,upper_fit,
    allocatable,band_room,query_inside,place,lowered(base,inner,pair,holding,spare,pairs),
    band_core(~Source,~value,~Gemm.lower_half,source,low,high,cap,cap_ok,bounded,depth,limit,length,query,current,
      lower_first(base,inner,pair),inner,pair_target(inner,pair),start,target_exact,lower_after,lower_fit,allocatable,band_room,query_inside,place,
      holding,tiled,values,spare,spare_room,pairs,balanced,long,read))

# ---- Each query through one pair ----

# A pair below the query pair leaves it as it was.
def step_before(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+high: Nat,+cap: U32,+base: U32,+groups: U32,+total: Nat,+inner: U32,
  +depth: Nat,+length: Nat,+query: Nat,+room: PairRoom<cap,high,base,groups,total,inner,depth>,
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+done: Nat,+words: PairWords<high,base,groups,inner,depth,done>,
  +pair: Nat,+tap: Nat,+exact: {Nat.add(Nat.mul(pair,U32.to_nat(inner)),tap) == query : Nat},+tap_inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool},
  +below: {Nat.is_lt(pair,done) == True{} : Bool},
  +holding: Held.Holding<Tile.Tile8>,+tiled: Held.Tiled(~Tile.Tile8,holding,U32.to_nat(base),high),+values: Held.HoldingValues(~Tile.Tile8,~Source,~value,source,holding),
  +spare: Held.SpareModel<Tile.Tile8>,+spare_room: Held.SpareRoom(~Tile.Tile8,spare),
  +pairs: Buffer.Model<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,pairs)),+long: {Buffer.length(~Tile.Tile16,pairs) == length : Nat},
  +read: {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,pairs),U32.from_nat(query)) ==
    Bool.pick(Tile.Tile16,Nat.is_lt(pair,done),pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),pair,tap),Tile.filled_pair(0.0)) :
    Tile.Tile16}) ->
  Packed<Source,value,source,U32.to_nat(base),high,depth,length,query,Products.upper_band(U32.from_nat(done),groups,inner,base,lowered(base,inner,done,holding,spare,pairs)),
    Bool.pick(Tile.Tile16,Nat.is_lt(pair,1n+done),pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),pair,tap),Tile.filled_pair(0.0))>:
  +width = U32.to_nat(inner)
  +sample = pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),width,pair,tap)
  +before_start = {%exact : {Nat.is_lt(_,Nat.mul(done,width)) == True{} : Bool}
    Order.lt_le(Nat.add(Nat.mul(pair,width),tap),Nat.mul(1n+pair,width),Nat.mul(done,width),
      %Order.commutative(Nat.mul(pair,width),width) : {Nat.is_lt(Nat.add(Nat.mul(pair,width),tap),_) == True{} : Bool}
        Order.offset_lt(Nat.mul(pair,width),tap,width,tap_inside),
      RowMajor.mul_le(1n+pair,done,width,Order.strict_follow(pair,done,below))) :
    {Nat.is_lt(query,Nat.mul(done,width)) == True{} : Bool}}
  +below_next = Order.lt_succ(pair,done,Bounds.strict_le(pair,done,below))
  pair_cores(~Source,~value,source,high,cap,base,groups,total,inner,depth,length,query,
    Bool.pick(Tile.Tile16,Nat.is_lt(pair,done),sample,Tile.filled_pair(0.0)),Bool.pick(Tile.Tile16,Nat.is_lt(pair,1n+done),sample,Tile.filled_pair(0.0)),
    room,query_inside,done,words,Before{before_start},
    %Equal.sym(Bool,Nat.is_lt(pair,done),True{},below) :
      {Bool.pick(Tile.Tile16,_,sample,Tile.filled_pair(0.0)) == Bool.pick(Tile.Tile16,Nat.is_lt(pair,1n+done),sample,Tile.filled_pair(0.0)) : Tile.Tile16}
    %Equal.sym(Bool,Nat.is_lt(pair,1n+done),True{},below_next) :
      {Bool.pick(Tile.Tile16,True{},sample,Tile.filled_pair(0.0)) == Bool.pick(Tile.Tile16,_,sample,Tile.filled_pair(0.0)) : Tile.Tile16}
    {==},
    holding,tiled,values,spare,spare_room,pairs,balanced,long,read)

# A pair above the query pair leaves it as it was.
def step_after(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+high: Nat,+cap: U32,+base: U32,+groups: U32,+total: Nat,+inner: U32,
  +depth: Nat,+length: Nat,+query: Nat,+room: PairRoom<cap,high,base,groups,total,inner,depth>,
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+done: Nat,+words: PairWords<high,base,groups,inner,depth,done>,
  +pair: Nat,+tap: Nat,+exact: {Nat.add(Nat.mul(pair,U32.to_nat(inner)),tap) == query : Nat},
  +not_below: {Nat.is_lt(pair,done) == False{} : Bool},+after: {Nat.is_lt(done,pair) == True{} : Bool},
  +holding: Held.Holding<Tile.Tile8>,+tiled: Held.Tiled(~Tile.Tile8,holding,U32.to_nat(base),high),+values: Held.HoldingValues(~Tile.Tile8,~Source,~value,source,holding),
  +spare: Held.SpareModel<Tile.Tile8>,+spare_room: Held.SpareRoom(~Tile.Tile8,spare),
  +pairs: Buffer.Model<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,pairs)),+long: {Buffer.length(~Tile.Tile16,pairs) == length : Nat},
  +read: {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,pairs),U32.from_nat(query)) ==
    Bool.pick(Tile.Tile16,Nat.is_lt(pair,done),pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),pair,tap),Tile.filled_pair(0.0)) :
    Tile.Tile16}) ->
  Packed<Source,value,source,U32.to_nat(base),high,depth,length,query,Products.upper_band(U32.from_nat(done),groups,inner,base,lowered(base,inner,done,holding,spare,pairs)),
    Bool.pick(Tile.Tile16,Nat.is_lt(pair,1n+done),pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),pair,tap),Tile.filled_pair(0.0))>:
  +width = U32.to_nat(inner)
  +sample = pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),width,pair,tap)
  +past_end = {%exact : {Nat.is_le(Nat.add(Nat.mul(done,width),width),_) == True{} : Bool}
    %Order.commutative(width,Nat.mul(done,width)) : {Nat.is_le(_,Nat.add(Nat.mul(pair,width),tap)) == True{} : Bool}
      Order.transitive(Nat.mul(1n+done,width),Nat.mul(pair,width),Nat.add(Nat.mul(pair,width),tap),
        RowMajor.mul_le(1n+done,pair,width,Order.strict_follow(done,pair,after)),Order.le_add(Nat.mul(pair,width),tap)) :
    {Nat.is_le(Nat.add(Nat.mul(done,width),width),query) == True{} : Bool}}
  +not_next = Order.not_below(1n+done,pair,Order.strict_follow(done,pair,after))
  pair_cores(~Source,~value,source,high,cap,base,groups,total,inner,depth,length,query,
    Bool.pick(Tile.Tile16,Nat.is_lt(pair,done),sample,Tile.filled_pair(0.0)),Bool.pick(Tile.Tile16,Nat.is_lt(pair,1n+done),sample,Tile.filled_pair(0.0)),
    room,query_inside,done,words,Past{past_end},
    %Equal.sym(Bool,Nat.is_lt(pair,done),False{},not_below) :
      {Bool.pick(Tile.Tile16,_,sample,Tile.filled_pair(0.0)) == Bool.pick(Tile.Tile16,Nat.is_lt(pair,1n+done),sample,Tile.filled_pair(0.0)) : Tile.Tile16}
    %Equal.sym(Bool,Nat.is_lt(pair,1n+done),False{},not_next) :
      {Bool.pick(Tile.Tile16,False{},sample,Tile.filled_pair(0.0)) == Bool.pick(Tile.Tile16,_,sample,Tile.filled_pair(0.0)) : Tile.Tile16}
    {==},
    holding,tiled,values,spare,spare_room,pairs,balanced,long,read)

# The query pair takes the tiles of both its groups.
def step_same(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+high: Nat,+cap: U32,+base: U32,+groups: U32,+total: Nat,+inner: U32,
  +depth: Nat,+length: Nat,+query: Nat,+room: PairRoom<cap,high,base,groups,total,inner,depth>,
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+done: Nat,+words: PairWords<high,base,groups,inner,depth,done>,
  +pair: Nat,+tap: Nat,+exact: {Nat.add(Nat.mul(pair,U32.to_nat(inner)),tap) == query : Nat},+tap_inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool},
  +not_below: {Nat.is_lt(pair,done) == False{} : Bool},+not_after: {Nat.is_lt(done,pair) == False{} : Bool},
  +holding: Held.Holding<Tile.Tile8>,+tiled: Held.Tiled(~Tile.Tile8,holding,U32.to_nat(base),high),+values: Held.HoldingValues(~Tile.Tile8,~Source,~value,source,holding),
  +spare: Held.SpareModel<Tile.Tile8>,+spare_room: Held.SpareRoom(~Tile.Tile8,spare),
  +pairs: Buffer.Model<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,pairs)),+long: {Buffer.length(~Tile.Tile16,pairs) == length : Nat},
  +read: {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,pairs),U32.from_nat(query)) ==
    Bool.pick(Tile.Tile16,Nat.is_lt(pair,done),pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),pair,tap),Tile.filled_pair(0.0)) :
    Tile.Tile16}) ->
  Packed<Source,value,source,U32.to_nat(base),high,depth,length,query,Products.upper_band(U32.from_nat(done),groups,inner,base,lowered(base,inner,done,holding,spare,pairs)),
    Bool.pick(Tile.Tile16,Nat.is_lt(pair,1n+done),pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),pair,tap),Tile.filled_pair(0.0))>:
  +width = U32.to_nat(inner)
  +low = U32.to_nat(base)
  +count = U32.to_nat(groups)
  +same = Order.antisymmetric(pair,done,Order.not_less(done,pair,not_after),Order.not_less(pair,done,not_below))
  +sample = pair_value(~Source,~value,source,low,count,width,done,tap)
  +lower_sum = Nat.mul(Nat.mul(done,2n),width)
  +upper_sum = Nat.mul(upper_group(done,count),width)
  +now_next = Order.lt_succ(done,done,Order.reflexive(done))
  %Equal.sym(Nat,pair,done,same) :
    Packed<Source,value,source,low,high,depth,length,query,Products.upper_band(U32.from_nat(done),groups,inner,base,lowered(base,inner,done,holding,spare,pairs)),
      Bool.pick(Tile.Tile16,Nat.is_lt(_,1n+done),pair_value(~Source,~value,source,low,count,width,_,tap),Tile.filled_pair(0.0))>
  %Equal.sym(Bool,Nat.is_lt(done,1n+done),True{},now_next) :
    Packed<Source,value,source,low,high,depth,length,query,Products.upper_band(U32.from_nat(done),groups,inner,base,lowered(base,inner,done,holding,spare,pairs)),
      Bool.pick(Tile.Tile16,_,sample,Tile.filled_pair(0.0))>
  pair_cores(~Source,~value,source,high,cap,base,groups,total,inner,depth,length,query,Tile.filled_pair(0.0),sample,room,query_inside,done,words,
    At{tap,%same : {Nat.add(Nat.mul(_,width),tap) == query : Nat} exact,tap_inside},
    %Equal.sym(Nat,U32.to_nat(lower_first(base,inner,done)),Nat.add(low,lower_sum),words_lower(high,base,groups,inner,depth,done,words)) :
      {Tile.Tiles{value(source,U32.from_nat(Nat.add(_,tap))),value(source,U32.from_nat(Nat.add(U32.to_nat(upper_first(base,groups,inner,done)),tap)))} == sample : Tile.Tile16}
    %Equal.sym(Nat,U32.to_nat(upper_first(base,groups,inner,done)),Nat.add(low,upper_sum),words_upper(high,base,groups,inner,depth,done,words)) :
      {Tile.Tiles{value(source,U32.from_nat(Nat.add(Nat.add(low,lower_sum),tap))),value(source,U32.from_nat(Nat.add(_,tap)))} == sample : Tile.Tile16}
    %Equal.sym(Nat,Nat.add(Nat.add(low,lower_sum),tap),Nat.add(low,Nat.add(lower_sum,tap)),Addition.associative(low,lower_sum,tap)) :
      {Tile.Tiles{value(source,U32.from_nat(_)),value(source,U32.from_nat(Nat.add(Nat.add(low,upper_sum),tap)))} == sample : Tile.Tile16}
    %Equal.sym(Nat,Nat.add(Nat.add(low,upper_sum),tap),Nat.add(low,Nat.add(upper_sum,tap)),Addition.associative(low,upper_sum,tap)) :
      {Tile.Tiles{value(source,U32.from_nat(Nat.add(low,Nat.add(lower_sum,tap)))),value(source,U32.from_nat(_))} == sample : Tile.Tile16}
    {==},
    holding,tiled,values,spare,spare_room,pairs,balanced,long,
    %not_below : {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,pairs),U32.from_nat(query)) ==
      Bool.pick(Tile.Tile16,_,pair_value(~Source,~value,source,low,count,width,pair,tap),Tile.filled_pair(0.0)) : Tile.Tile16} read)

# A pair not below the query pair is above it or is it.
def step_from(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+high: Nat,+cap: U32,+base: U32,+groups: U32,+total: Nat,+inner: U32,
  +depth: Nat,+length: Nat,+query: Nat,+room: PairRoom<cap,high,base,groups,total,inner,depth>,
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+done: Nat,+words: PairWords<high,base,groups,inner,depth,done>,
  +pair: Nat,+tap: Nat,+exact: {Nat.add(Nat.mul(pair,U32.to_nat(inner)),tap) == query : Nat},+tap_inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool},
  +not_below: {Nat.is_lt(pair,done) == False{} : Bool},
  +holding: Held.Holding<Tile.Tile8>,+tiled: Held.Tiled(~Tile.Tile8,holding,U32.to_nat(base),high),+values: Held.HoldingValues(~Tile.Tile8,~Source,~value,source,holding),
  +spare: Held.SpareModel<Tile.Tile8>,+spare_room: Held.SpareRoom(~Tile.Tile8,spare),
  +pairs: Buffer.Model<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,pairs)),+long: {Buffer.length(~Tile.Tile16,pairs) == length : Nat},
  +read: {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,pairs),U32.from_nat(query)) ==
    Bool.pick(Tile.Tile16,Nat.is_lt(pair,done),pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),pair,tap),Tile.filled_pair(0.0)) :
    Tile.Tile16}) ->
  Packed<Source,value,source,U32.to_nat(base),high,depth,length,query,Products.upper_band(U32.from_nat(done),groups,inner,base,lowered(base,inner,done,holding,spare,pairs)),
    Bool.pick(Tile.Tile16,Nat.is_lt(pair,1n+done),pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),pair,tap),Tile.filled_pair(0.0))>:
  Held.by_cases(Packed<Source,value,source,U32.to_nat(base),high,depth,length,query,
      Products.upper_band(U32.from_nat(done),groups,inner,base,lowered(base,inner,done,holding,spare,pairs)),
      Bool.pick(Tile.Tile16,Nat.is_lt(pair,1n+done),pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),pair,tap),Tile.filled_pair(0.0))>,
    Nat.is_lt(done,pair),
    after => step_after(~Source,~value,source,high,cap,base,groups,total,inner,depth,length,query,room,query_inside,done,words,pair,tap,exact,not_below,after,
      holding,tiled,values,spare,spare_room,pairs,balanced,long,read),
    not_after => step_same(~Source,~value,source,high,cap,base,groups,total,inner,depth,length,query,room,query_inside,done,words,pair,tap,exact,tap_inside,
      not_below,not_after,holding,tiled,values,spare,spare_room,pairs,balanced,long,read))

# The pair done leaves the expected value of the query.
def pair_step(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+high: Nat,+cap: U32,+base: U32,+groups: U32,+total: Nat,+inner: U32,
  +depth: Nat,+length: Nat,+query: Nat,+room: PairRoom<cap,high,base,groups,total,inner,depth>,
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+kind: Query<total,U32.to_nat(inner),query>,
  +done: Nat,+done_inside: {Nat.is_lt(done,total) == True{} : Bool},
  +holding: Held.Holding<Tile.Tile8>,+tiled: Held.Tiled(~Tile.Tile8,holding,U32.to_nat(base),high),+values: Held.HoldingValues(~Tile.Tile8,~Source,~value,source,holding),
  +spare: Held.SpareModel<Tile.Tile8>,+spare_room: Held.SpareRoom(~Tile.Tile8,spare),
  +pairs: Buffer.Model<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,Buffer.tree(~Tile.Tile16,pairs)),+long: {Buffer.length(~Tile.Tile16,pairs) == length : Nat},
  +read: {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,pairs),U32.from_nat(query)) ==
    expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),total,query,kind,done) : Tile.Tile16}) ->
  Packed<Source,value,source,U32.to_nat(base),high,depth,length,query,Products.upper_band(U32.from_nat(done),groups,inner,base,lowered(base,inner,done,holding,spare,pairs)),
    expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),total,query,kind,1n+done)>:
  match kind:
    case Inside{+pair,+tap,+exact,+pair_inside,+tap_inside}:
      +words = pair_words(cap,high,base,groups,total,inner,depth,room,done,done_inside)
      Held.by_cases(Packed<Source,value,source,U32.to_nat(base),high,depth,length,query,
          Products.upper_band(U32.from_nat(done),groups,inner,base,lowered(base,inner,done,holding,spare,pairs)),
          Bool.pick(Tile.Tile16,Nat.is_lt(pair,1n+done),pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),pair,tap),Tile.filled_pair(0.0))>,
        Nat.is_lt(pair,done),
        below => step_before(~Source,~value,source,high,cap,base,groups,total,inner,depth,length,query,room,query_inside,done,words,pair,tap,exact,tap_inside,below,
          holding,tiled,values,spare,spare_room,pairs,balanced,long,read),
        not_below => step_from(~Source,~value,source,high,cap,base,groups,total,inner,depth,length,query,room,query_inside,done,words,pair,tap,exact,tap_inside,
          not_below,holding,tiled,values,spare,spare_room,pairs,balanced,long,read))
    case Beyond{+beyond}:
      +width = U32.to_nat(inner)
      +past_end = {%Order.commutative(width,Nat.mul(done,width)) : {Nat.is_le(_,query) == True{} : Bool}
        Order.transitive(Nat.mul(1n+done,width),Nat.mul(total,width),query,RowMajor.mul_le(1n+done,total,width,Order.strict_follow(done,total,done_inside)),beyond) :
        {Nat.is_le(Nat.add(Nat.mul(done,width),width),query) == True{} : Bool}}
      pair_cores(~Source,~value,source,high,cap,base,groups,total,inner,depth,length,query,Tile.filled_pair(0.0),Tile.filled_pair(0.0),room,query_inside,done,
        pair_words(cap,high,base,groups,total,inner,depth,room,done,done_inside),Past{past_end},{==},
        holding,tiled,values,spare,spare_room,pairs,balanced,long,read)

# ---- All pairs ----

# The held pairs from pair done on leave the expected value of the query
# after every pair.
def held_loop(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+high: Nat,+cap: U32,+base: U32,+groups: U32,+total: Nat,+inner: U32,
  +depth: Nat,+length: Nat,+query: Nat,+room: PairRoom<cap,high,base,groups,total,inner,depth>,
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+kind: Query<total,U32.to_nat(inner),query>,
  remaining: Nat,+done: Nat,+count: {Nat.add(remaining,done) == total : Nat},-state: Run(),
  +facts: Packed<Source,value,source,U32.to_nat(base),high,depth,length,query,state,
    expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),total,query,kind,done)>) ->
  PairsAt<depth,length,query,Products.held_pairs(remaining,U32.from_nat(done),groups,inner,base,state),
    expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),total,query,kind,total)>:
  match remaining:
    case 0n:
      Packed{+holding,+spare,+pairs,+packed,+tiled,+values,+spare_room,+balanced,+long,+read} = facts
      %Equal.sym(Run(),state,((Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare)),Buffer.pack(~Tile.Tile16,pairs)),packed) :
        PairsAt<depth,length,query,Products.held_pairs(0n,U32.from_nat(done),groups,inner,base,_),
          expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),total,query,kind,total)>
      %count : PairsAt<depth,length,query,Buffer.pack(~Tile.Tile16,pairs),expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),total,query,kind,_)>
      PairsAt{pairs,{==},balanced,long,read}
    case 1n+ +rest:
      Packed{+holding,+spare,+pairs,+packed,+tiled,+values,+spare_room,+balanced,+long,+read} = facts
      +done_inside = {%count : {Nat.is_lt(done,_) == True{} : Bool} Order.lt_succ(done,Nat.add(rest,done),Order.right_le(rest,done)) :
        {Nat.is_lt(done,total) == True{} : Bool}}
      -after = expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(groups),U32.to_nat(inner),total,query,kind,total)
      %Equal.sym(Run(),state,((Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare)),Buffer.pack(~Tile.Tile16,pairs)),packed) :
        PairsAt<depth,length,query,Products.held_pairs(1n+rest,U32.from_nat(done),groups,inner,base,_),after>
      %Equal.sym(U32,(U32.from_nat(done) + 1 : U32),U32.from_nat(1n+done),Bounds.increment_encoded(done)) :
        PairsAt<depth,length,query,Products.held_pairs(rest,_,groups,inner,base,Products.upper_band(U32.from_nat(done),groups,inner,base,lowered(base,inner,done,holding,spare,pairs))),
          after>
      held_loop(~Source,~value,source,high,cap,base,groups,total,inner,depth,length,query,room,query_inside,kind,rest,1n+done,
        Equal.trans(Nat,Nat.add(rest,1n+done),1n+Nat.add(rest,done),total,Addition.successor_right(rest,done),count),
        Products.upper_band(U32.from_nat(done),groups,inner,base,lowered(base,inner,done,holding,spare,pairs)),
        pair_step(~Source,~value,source,high,cap,base,groups,total,inner,depth,length,query,room,query_inside,kind,done,done_inside,
          holding,tiled,values,spare,spare_room,pairs,balanced,long,read))

# ---- The held pairs of a rectangle against the pairs of its copied weights ----

def pairs_word(+channels: U32,+inner: U32) -> U32:
  (Gemm.pairs_of(channels) * inner : U32)

def pairs_depth(+channels: U32,+inner: U32) -> Nat:
  Tensors.depth_for(pairs_word(channels,inner))

# The held weights, an empty spare and the fresh pairs before the first band.
def start_state(+channels: U32,+inner: U32,+holding: Held.Holding<Tile.Tile8>) -> Run():
  ((Held.pack(~Tile.Tile8,holding),Move.spare(~Tile.Tile8,Products.zero())),Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),pairs_word(channels,inner)))

def held_tree(+channels: U32,+inner: U32,+base: U32,+holding: Held.Holding<Tile.Tile8>) -> Tree.Tree<Tile.Tile16>:
  Buffer.content_tree(~Tile.Tile16,Buffer.contents(~Tile.Tile16,
    Products.held_pairs(U32.to_nat(Gemm.pairs_of(channels)),0,Gemm.blocks(channels),inner,base,start_state(channels,inner,holding))))

def copied_tree(+channels: U32,+inner: U32,+weights: Buffer.Model<Tile.Tile8>) -> Tree.Tree<Tile.Tile16>:
  PointBuffer.pair_loop(channels,inner,Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner)))

# The pairs of the pair count of channels and their fresh tree.
def word_exact(+limits: Machine.Limits,+channels: U32,+inner: U32,+space: Operands.PairSpace<limits,channels,inner>) ->
  {U32.to_nat(pairs_word(channels,inner)) == Nat.mul(U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(inner)) : Nat}:
  Operands.PairSpace{+groups_room,+width_room,+weights_room} = space
  +pairs = U32.to_nat(Gemm.pairs_of(channels))
  +width = U32.to_nat(inner)
  Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),Gemm.pairs_of(channels),inner,pairs,width,{==},{==},
    Order.transitive(Nat.mul(pairs,width),Nat.mul(U32.to_nat(Gemm.blocks(channels)),width),U32.to_nat(Machine.capacity(limits)),
      RowMajor.mul_le(pairs,U32.to_nat(Gemm.blocks(channels)),width,Groups.pairs_fewer(limits,channels,groups_room)),weights_room))

def word_sized(+limits: Machine.Limits,+channels: U32,+inner: U32,+space: Operands.PairSpace<limits,channels,inner>) -> Depth.Sized<pairs_word(channels,inner)>:
  Operands.PairSpace{+groups_room,+width_room,+weights_room} = space
  +pairs = U32.to_nat(Gemm.pairs_of(channels))
  +width = U32.to_nat(inner)
  +room = Order.transitive(Nat.mul(pairs,width),Nat.mul(U32.to_nat(Gemm.blocks(channels)),width),U32.to_nat(Machine.capacity(limits)),
    RowMajor.mul_le(pairs,U32.to_nat(Gemm.blocks(channels)),width,Groups.pairs_fewer(limits,channels,groups_room)),weights_room)
  Depth.bounded(pairs_word(channels,inner),Limits.count_allocatable(limits,pairs_word(channels,inner),
    %Equal.sym(Nat,U32.to_nat(pairs_word(channels,inner)),Nat.mul(pairs,width),word_exact(limits,channels,inner,space)) :
      {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room))

def word_limit(+limits: Machine.Limits,+channels: U32,+inner: U32,+space: Operands.PairSpace<limits,channels,inner>) ->
  {Nat.is_le(pairs_depth(channels,inner),31n) == True{} : Bool}:
  Order.successor(pairs_depth(channels,inner),30n,Depth.allocation_depth(pairs_word(channels,inner),word_sized(limits,channels,inner,space)))

def fresh_balanced(+channels: U32,+inner: U32) -> Shape.Balanced(Tile.Tile16,pairs_depth(channels,inner),Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner))):
  +depth = pairs_depth(channels,inner)
  %Equal.sym(Tree.Tree<Tile.Tile16>,Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner)),Tree.fresh(Tile.Tile16,depth,Tile.filled_pair(0.0)),
    Allocation.buffer_tree(~Tile.Tile16,Tile.filled_pair(0.0),pairs_word(channels,inner))) : Shape.Balanced(Tile.Tile16,depth,_)
  Shape.fresh(Tile.Tile16,depth,Tile.filled_pair(0.0))

def fresh_capacity(+channels: U32,+inner: U32) ->
  {Tree.capacity(Tile.Tile16,Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner))) == U32.shln(1,pairs_depth(channels,inner)) : U32}:
  Shape.capacity(Tile.Tile16,pairs_depth(channels,inner),Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner)),fresh_balanced(channels,inner))

def fresh_read(+channels: U32,+inner: U32,+query: Nat) ->
  {Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner)),U32.from_nat(query)) == Tile.filled_pair(0.0) : Tile.Tile16}:
  %Equal.sym(Tree.Tree<Tile.Tile16>,Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner)),Tree.fresh(Tile.Tile16,pairs_depth(channels,inner),Tile.filled_pair(0.0)),
    Allocation.buffer_tree(~Tile.Tile16,Tile.filled_pair(0.0),pairs_word(channels,inner))) :
    {Tree.read(Tile.Tile16,_,U32.from_nat(query)) == Tile.filled_pair(0.0) : Tile.Tile16}
  StorageReadback.read_fresh(~Tile.Tile16,pairs_depth(channels,inner),Tile.filled_pair(0.0),U32.from_nat(query))

def pairs_ceiling(+limits: Machine.Limits,+channels: U32,
  +groups_room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(Gemm.pairs_of(channels)) == Ceiling.ceiling(U32.to_nat(Gemm.blocks(channels)),2n) : Nat}:
  +count = U32.to_nat(Gemm.blocks(channels))
  Equal.trans(Nat,U32.to_nat(Gemm.pairs_of(channels)),Nat.div(Nat.add(count,1n),2n),Ceiling.ceiling(count,2n),
    Groups.pairs_exact(limits,channels,groups_room),Ceiling.shifted_formula(count,2n,{==}))

# ---- Each query ----

def double_positive(+value: Nat,+positive: {Nat.is_lt(0n,value) == True{} : Bool}) -> {Nat.is_lt(0n,Nat.double(value)) == True{} : Bool}:
  match value:
    case 0n: Empty.absurd({Nat.is_lt(0n,Nat.double(0n)) == True{} : Bool},Held.lt_le_absurd(0n,0n,positive,{==}))
    case 1n+ +rest: {==}

def extent_positive(+depth: Nat) -> {Nat.is_lt(0n,Shape.extent(depth)) == True{} : Bool}:
  match depth:
    case 0n: {==}
    case 1n+ +rest: double_positive(Shape.extent(rest),extent_positive(rest))

# A query below pairs of a width lies at a tap of a pair.
def inside_query(+pairs: Nat,+width: Nat,+query: Nat,+inside: {Nat.is_lt(query,Nat.mul(pairs,width)) == True{} : Bool}) -> Query<pairs,width,query>:
  +positive = DivisionOrder.right_factor_positive(pairs,width,Order.le_lt_trans(0n,query,Nat.mul(pairs,width),Order.zero_le(query),inside))
  Inside{Nat.div(query,width),Nat.mod(query,width),
    Equal.sym(Nat,query,Nat.add(Nat.mul(Nat.div(query,width),width),Nat.mod(query,width)),DivisionOrder.quotient_exact(query,width,positive)),
    DivisionOrder.quotient_below(query,pairs,width,positive,inside),DivisionOrder.remainder_small(query,width,positive)}

def query_kind(+pairs: Nat,+width: Nat,+query: Nat) -> Query<pairs,width,query>:
  Held.by_cases(Query<pairs,width,query>,Nat.is_lt(query,Nat.mul(pairs,width)),
    inside => inside_query(pairs,width,query,inside),
    outside => Beyond{Order.not_less(query,Nat.mul(pairs,width),outside)})

# No pair is done before the first.
def expected_start(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+base: Nat,+groups: Nat,+width: Nat,+pairs: Nat,+query: Nat,
  +kind: Query<pairs,width,query>) -> {Tile.filled_pair(0.0) == expected(~Source,~value,source,base,groups,width,pairs,query,kind,0n) : Tile.Tile16}:
  match kind:
    case Inside{+pair,+tap,+exact,+pair_inside,+tap_inside}:
      %Equal.sym(Bool,Nat.is_lt(pair,0n),False{},Order.not_below(0n,pair,Order.zero_le(pair))) :
        {Tile.filled_pair(0.0) == Bool.pick(Tile.Tile16,_,pair_value(~Source,~value,source,base,groups,width,pair,tap),Tile.filled_pair(0.0)) : Tile.Tile16}
      {==}
    case Beyond{+beyond}: {==}

# The held pairs leave the expected value at a query.
def held_at(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,+channels: U32,+inner: U32,+base: U32,
  +holding: Held.Holding<Tile.Tile8>,+cells: U32,+total: Nat,
  +cells_nat: {U32.to_nat(cells) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)) : Nat},
  +room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(cells)),total) == True{} : Bool},
  +window: Halo.TaskWindow<Tile.Tile8,Source,value,source,holding,base,(base + cells : U32),total>,+space: Operands.PairSpace<limits,channels,inner>,
  +query: Nat,+query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,pairs_depth(channels,inner)))) == True{} : Bool},
  +kind: Query<U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(inner),query>) ->
  PairsAt<pairs_depth(channels,inner),Buffer.length(~Tile.Tile16,Point.fresh_pairs(channels,inner)),query,
    Products.held_pairs(U32.to_nat(Gemm.pairs_of(channels)),U32.from_nat(0n),Gemm.blocks(channels),inner,base,start_state(channels,inner,holding)),
    expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner),U32.to_nat(Gemm.pairs_of(channels)),query,kind,
      U32.to_nat(Gemm.pairs_of(channels)))>:
  Halo.TaskWindow{+high,+reaches,+within,+tiled,+values} = window
  Operands.PairSpace{+groups_room,+width_room,+weights_room} = space
  +cap = Machine.capacity(limits)
  +groups = Gemm.blocks(channels)
  +count = U32.to_nat(groups)
  +width = U32.to_nat(inner)
  +pairs = U32.to_nat(Gemm.pairs_of(channels))
  +low = U32.to_nat(base)
  +depth = pairs_depth(channels,inner)
  +fresh = Point.fresh_pairs(channels,inner)
  +reach_word = {%PointWindows.available(limits,base,cells,total,room,contains) : {Nat.is_le(_,high) == True{} : Bool} reaches :
    {Nat.is_le(Nat.add(low,U32.to_nat(cells)),high) == True{} : Bool}}
  +pair_room = {PairRoom{Limits.capacity_fits(limits),Order.transitive(high,total,U32.to_nat(cap),within,room),
    %cells_nat : {Nat.is_le(Nat.add(low,_),high) == True{} : Bool} reach_word,
    groups_room,pairs_ceiling(limits,channels,groups_room),word_limit(limits,channels,inner,space),
    %word_exact(limits,channels,inner,space) : {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
      Depth.allocation_room(pairs_word(channels,inner),word_sized(limits,channels,inner,space)),
    Limits.count_allocatable(limits,inner,width_room)} : PairRoom<cap,high,base,groups,pairs,inner,depth>}
  held_loop(~Source,~value,source,high,cap,base,groups,pairs,inner,depth,Buffer.length(~Tile.Tile16,fresh),query,pair_room,query_inside,kind,pairs,0n,
    Addition.zero_right(pairs),start_state(channels,inner,holding),
    Packed{holding,Held.fresh_spare(~Tile.Tile8,Products.zero()),fresh,
      %Held.spare_packed(~Tile.Tile8,Products.zero()) :
        {start_state(channels,inner,holding) == ((Held.pack(~Tile.Tile8,holding),_),Buffer.pack(~Tile.Tile16,fresh)) : Run()}
      %Equal.sym(Storage.Buffer<Tile.Tile16>,Buffer.pack(~Tile.Tile16,fresh),Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),pairs_word(channels,inner)),Buffer.remodeled(~Tile.Tile16,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),pairs_word(channels,inner)))) :
        {start_state(channels,inner,holding) == ((Held.pack(~Tile.Tile8,holding),Move.spare(~Tile.Tile8,Products.zero())),_) : Run()}
      {==},
      tiled,values,Held.fresh_spare_room(~Tile.Tile8,Products.zero()),fresh_balanced(channels,inner),{==},
      Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,Buffer.tree(~Tile.Tile16,fresh),U32.from_nat(query)),Tile.filled_pair(0.0),
        expected(~Source,~value,source,low,count,width,pairs,query,kind,0n),
        fresh_read(channels,inner,query),expected_start(~Source,~value,source,low,count,width,pairs,query,kind))})

# What the pairs at a depth hold, read through their contents.
def seen_balanced(+depth: Nat,+length: Nat,+query: Nat,-pairs: Storage.Buffer<Tile.Tile16>,+current: Tile.Tile16,+at: PairsAt<depth,length,query,pairs,current>) ->
  Shape.Balanced(Tile.Tile16,depth,Buffer.content_tree(~Tile.Tile16,Buffer.contents(~Tile.Tile16,pairs))):
  PairsAt{+model,+packed,+balanced,+long,+read} = at
  %Equal.sym(Storage.Buffer<Tile.Tile16>,pairs,Buffer.pack(~Tile.Tile16,model),packed) :
    Shape.Balanced(Tile.Tile16,depth,Buffer.content_tree(~Tile.Tile16,Buffer.contents(~Tile.Tile16,_)))
  %Equal.sym(Buffer.Content<Tile.Tile16>,Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,model)),Buffer.content(~Tile.Tile16,model),Buffer.packed_contents(~Tile.Tile16,model)) :
    Shape.Balanced(Tile.Tile16,depth,Buffer.content_tree(~Tile.Tile16,_))
  %Equal.sym(Tree.Tree<Tile.Tile16>,Buffer.content_tree(~Tile.Tile16,Buffer.content(~Tile.Tile16,model)),Buffer.tree(~Tile.Tile16,model),Buffer.content_tree_of(~Tile.Tile16,model)) :
    Shape.Balanced(Tile.Tile16,depth,_)
  balanced

def seen_long(+depth: Nat,+length: Nat,+query: Nat,-pairs: Storage.Buffer<Tile.Tile16>,+current: Tile.Tile16,+at: PairsAt<depth,length,query,pairs,current>) ->
  {Buffer.content_length(~Tile.Tile16,Buffer.contents(~Tile.Tile16,pairs)) == length : Nat}:
  PairsAt{+model,+packed,+balanced,+long,+read} = at
  %Equal.sym(Storage.Buffer<Tile.Tile16>,pairs,Buffer.pack(~Tile.Tile16,model),packed) :
    {Buffer.content_length(~Tile.Tile16,Buffer.contents(~Tile.Tile16,_)) == length : Nat}
  %Equal.sym(Buffer.Content<Tile.Tile16>,Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,model)),Buffer.content(~Tile.Tile16,model),Buffer.packed_contents(~Tile.Tile16,model)) :
    {Buffer.content_length(~Tile.Tile16,_) == length : Nat}
  Equal.trans(Nat,Buffer.content_length(~Tile.Tile16,Buffer.content(~Tile.Tile16,model)),Buffer.length(~Tile.Tile16,model),length,
    Buffer.content_length_of(~Tile.Tile16,model),long)

def seen_read(+depth: Nat,+length: Nat,+query: Nat,-pairs: Storage.Buffer<Tile.Tile16>,+current: Tile.Tile16,+at: PairsAt<depth,length,query,pairs,current>) ->
  {Tree.read(Tile.Tile16,Buffer.content_tree(~Tile.Tile16,Buffer.contents(~Tile.Tile16,pairs)),U32.from_nat(query)) == current : Tile.Tile16}:
  PairsAt{+model,+packed,+balanced,+long,+read} = at
  %Equal.sym(Storage.Buffer<Tile.Tile16>,pairs,Buffer.pack(~Tile.Tile16,model),packed) :
    {Tree.read(Tile.Tile16,Buffer.content_tree(~Tile.Tile16,Buffer.contents(~Tile.Tile16,_)),U32.from_nat(query)) == current : Tile.Tile16}
  %Equal.sym(Buffer.Content<Tile.Tile16>,Buffer.contents(~Tile.Tile16,Buffer.pack(~Tile.Tile16,model)),Buffer.content(~Tile.Tile16,model),Buffer.packed_contents(~Tile.Tile16,model)) :
    {Tree.read(Tile.Tile16,Buffer.content_tree(~Tile.Tile16,_),U32.from_nat(query)) == current : Tile.Tile16}
  %Equal.sym(Tree.Tree<Tile.Tile16>,Buffer.content_tree(~Tile.Tile16,Buffer.content(~Tile.Tile16,model)),Buffer.tree(~Tile.Tile16,model),Buffer.content_tree_of(~Tile.Tile16,model)) :
    {Tree.read(Tile.Tile16,_,U32.from_nat(query)) == current : Tile.Tile16}
  read

# The pairs read from copied weights hold the tiles of both groups of a pair.
def copied_inside(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,+channels: U32,+inner: U32,+base: U32,
  +holding: Held.Holding<Tile.Tile8>,+cells: U32,+total: Nat,
  +cells_nat: {U32.to_nat(cells) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)) : Nat},
  +room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(cells)),total) == True{} : Bool},
  +window: Halo.TaskWindow<Tile.Tile8,Source,value,source,holding,base,(base + cells : U32),total>,+space: Operands.PairSpace<limits,channels,inner>,
  +pair: Nat,+tap: Nat,+pair_inside: {Nat.is_lt(pair,U32.to_nat(Gemm.pairs_of(channels))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,copied_tree(channels,inner,PointWindows.copy_model(holding,base,cells)),U32.from_nat(Nat.add(Nat.mul(pair,U32.to_nat(inner)),tap))) ==
    pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner),pair,tap) : Tile.Tile16}:
  Operands.PairSpace{+groups_room,+width_room,+weights_room} = space
  +groups = Gemm.blocks(channels)
  +count = U32.to_nat(groups)
  +width = U32.to_nat(inner)
  +pairs = U32.to_nat(Gemm.pairs_of(channels))
  +low = U32.to_nat(base)
  +depth = pairs_depth(channels,inner)
  +limit = word_limit(limits,channels,inner,space)
  +weights = Buffer.tree(~Tile.Tile8,PointWindows.copy_model(holding,base,cells))
  +fresh_tree = Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner))
  +index = Nat.add(Nat.mul(pair,width),tap)
  +double = Nat.mul(pair,2n)
  +upper = upper_group(pair,count)
  +lower_index = Nat.add(Nat.mul(double,width),tap)
  +upper_index = Nat.add(Nat.mul(upper,width),tap)
  +lower_fits = lower_inside(pair,pairs,count,pairs_ceiling(limits,channels,groups_room),pair_inside)
  +lower_group = Order.lt_le(double,1n+double,count,Order.lt_succ(double,double,Order.reflexive(double)),lower_fits)
  +upper_group_inside = Order.lt_le(upper,1n+upper,count,Order.lt_succ(upper,upper,Order.reflexive(upper)),upper_inside(pair,count,lower_fits))
  +lower_cell = {%Equal.sym(Nat,U32.to_nat(cells),Nat.mul(count,width),cells_nat) : {Nat.is_lt(lower_index,_) == True{} : Bool}
    RowMajor.radix(double,count,tap,width,lower_group,tap_inside) : {Nat.is_lt(lower_index,U32.to_nat(cells)) == True{} : Bool}}
  +upper_cell = {%Equal.sym(Nat,U32.to_nat(cells),Nat.mul(count,width),cells_nat) : {Nat.is_lt(upper_index,_) == True{} : Bool}
    RowMajor.radix(upper,count,tap,width,upper_group_inside,tap_inside) : {Nat.is_lt(upper_index,U32.to_nat(cells)) == True{} : Bool}}
  +sample = pair_value(~Source,~value,source,low,count,width,pair,tap)
  +read_pair = {Tile.Tiles{Tree.read(Tile.Tile8,weights,U32.from_nat(lower_index)),Tree.read(Tile.Tile8,weights,U32.from_nat(upper_index))} : Tile.Tile16}
  Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,PointBuffer.pair_loop(channels,inner,weights,fresh_tree),U32.from_nat(index)),
    PointBuffer.pair_tiles(weights,Gemm.PairPlan{groups,inner},U32.from_nat(index)),sample,
    PointBuffer.pair_at(channels,inner,weights,fresh_tree,depth,fresh_balanced(channels,inner),limit,
      %Equal.sym(U32,Tree.capacity(Tile.Tile16,fresh_tree),U32.shln(1,depth),fresh_capacity(channels,inner)) : {U32.is_le(_,2147483648) == True{} : Bool}
        Power.native_power_bound(depth,limit),
      %Equal.sym(U32,Tree.capacity(Tile.Tile16,fresh_tree),U32.shln(1,depth),fresh_capacity(channels,inner)) :
        {Nat.is_le(U32.to_nat(pairs_word(channels,inner)),U32.to_nat(_)) == True{} : Bool}
        Depth.allocation_room(pairs_word(channels,inner),word_sized(limits,channels,inner,space)),
      index,
      %Equal.sym(Nat,U32.to_nat(pairs_word(channels,inner)),Nat.mul(pairs,width),word_exact(limits,channels,inner,space)) : {Nat.is_lt(index,_) == True{} : Bool}
        RowMajor.radix(pair,pairs,tap,width,pair_inside,tap_inside)),
    Equal.trans(Tile.Tile16,PointBuffer.pair_tiles(weights,Gemm.PairPlan{groups,inner},U32.from_nat(index)),read_pair,sample,
      PointBuffer.pair_address(weights,groups,inner,count,width,{==},{==},Machine.half_word(limits),Limits.half_word_fits(limits),
        Limits.within_half(limits,Nat.mul(count,width),weights_room),Limits.double_bound(limits,width,width_room),pair,tap,tap_inside,lower_fits),
      %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,weights,U32.from_nat(lower_index)),value(source,U32.from_nat(Nat.add(low,lower_index))),
        PointWindows.copied_read(~Source,~value,source,limits,holding,base,cells,total,room,contains,window,lower_index,lower_cell)) :
        {Tile.Tiles{_,Tree.read(Tile.Tile8,weights,U32.from_nat(upper_index))} == sample : Tile.Tile16}
      %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,weights,U32.from_nat(upper_index)),value(source,U32.from_nat(Nat.add(low,upper_index))),
        PointWindows.copied_read(~Source,~value,source,limits,holding,base,cells,total,room,contains,window,upper_index,upper_cell)) :
        {Tile.Tiles{value(source,U32.from_nat(Nat.add(low,lower_index))),_} == sample : Tile.Tile16}
      {==}))

# The pairs read from copied weights keep the fill past the pairs.
def copied_beyond(+limits: Machine.Limits,+channels: U32,+inner: U32,+weights: Tree.Tree<Tile.Tile8>,+space: Operands.PairSpace<limits,channels,inner>,
  +query: Nat,+query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,pairs_depth(channels,inner)))) == True{} : Bool},
  +beyond: {Nat.is_le(Nat.mul(U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(inner)),query) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,PointBuffer.pair_loop(channels,inner,weights,Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner))),U32.from_nat(query)) ==
    Tile.filled_pair(0.0) : Tile.Tile16}:
  +depth = pairs_depth(channels,inner)
  +limit = word_limit(limits,channels,inner,space)
  +fresh_tree = Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner))
  +capacity = Tree.capacity(Tile.Tile16,fresh_tree)
  +count = PointBuffer.pair_count(channels,inner)
  +plan = {Gemm.PairPlan{Gemm.blocks(channels),inner} : Gemm.PairPlan}
  +word = U32.from_nat(query)
  +capacity_ok = {%Equal.sym(U32,capacity,U32.shln(1,depth),fresh_capacity(channels,inner)) : {U32.is_le(_,2147483648) == True{} : Bool}
    Power.native_power_bound(depth,limit) : {U32.is_le(capacity,2147483648) == True{} : Bool}}
  +fits = {%Equal.sym(U32,capacity,U32.shln(1,depth),fresh_capacity(channels,inner)) : {Nat.is_le(count,U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(pairs_word(channels,inner),word_sized(limits,channels,inner,space)) : {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool}}
  +query_cap = {%Equal.sym(U32,capacity,U32.shln(1,depth),fresh_capacity(channels,inner)) : {Nat.is_lt(query,U32.to_nat(_)) == True{} : Bool}
    query_inside : {Nat.is_lt(query,U32.to_nat(capacity)) == True{} : Bool}}
  +past = {%Equal.sym(Nat,count,Nat.mul(U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(inner)),word_exact(limits,channels,inner,space)) :
    {Nat.is_le(_,query) == True{} : Bool} beyond : {Nat.is_le(count,query) == True{} : Bool}}
  +current = Tree.read(Tile.Tile16,fresh_tree,word)
  -updated = Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile16,~Gemm.PairPlan,~PointBuffer.pair_sample,~Gemm.replaced(~Tile.Tile16),~Gemm.pair_placed,
    count,0,weights,plan,word,current)
  Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,PointBuffer.pair_loop(channels,inner,weights,fresh_tree),word),updated,Tile.filled_pair(0.0),
    Updates.update_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile16,~Gemm.PairPlan,~PointBuffer.pair_sample,~Gemm.replaced(~Tile.Tile16),~Gemm.pair_placed,
      count,0,weights,plan,fresh_tree,depth,word,fresh_balanced(channels,inner),limit,
      Bounds.native_bound(query,capacity,Bounds.roundtrip(query,capacity,capacity_ok,Bounds.strict_le(query,U32.to_nat(capacity),query_cap)),query_cap),
      Updates.identity_destinations(~Gemm.PairPlan,~Gemm.pair_placed,~PointBuffer.pair_identity,count,0n,plan,capacity,capacity_ok,fits)),
    Equal.trans(Tile.Tile16,updated,current,Tile.filled_pair(0.0),
      Updates.missed(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile16,~Gemm.PairPlan,~PointBuffer.pair_sample,~Gemm.replaced(~Tile.Tile16),~Gemm.pair_placed,
        count,0,weights,plan,word,current,
        Updates.identity_past(~Gemm.PairPlan,~Gemm.pair_placed,~PointBuffer.pair_identity,count,0n,plan,query,capacity,capacity_ok,past,
          Bounds.strict_le(query,U32.to_nat(capacity),query_cap))),
      fresh_read(channels,inner,query)))

# The pairs read from copied weights leave the expected value at a query.
def copied_at(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,+channels: U32,+inner: U32,+base: U32,
  +holding: Held.Holding<Tile.Tile8>,+cells: U32,+total: Nat,
  +cells_nat: {U32.to_nat(cells) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)) : Nat},
  +room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(cells)),total) == True{} : Bool},
  +window: Halo.TaskWindow<Tile.Tile8,Source,value,source,holding,base,(base + cells : U32),total>,+space: Operands.PairSpace<limits,channels,inner>,
  +query: Nat,+query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,pairs_depth(channels,inner)))) == True{} : Bool},
  +kind: Query<U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(inner),query>) ->
  {Tree.read(Tile.Tile16,copied_tree(channels,inner,PointWindows.copy_model(holding,base,cells)),U32.from_nat(query)) ==
    expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner),U32.to_nat(Gemm.pairs_of(channels)),query,kind,
      U32.to_nat(Gemm.pairs_of(channels))) : Tile.Tile16}:
  match kind:
    case Inside{+pair,+tap,+exact,+pair_inside,+tap_inside}:
      +copied = copied_tree(channels,inner,PointWindows.copy_model(holding,base,cells))
      +sample = pair_value(~Source,~value,source,U32.to_nat(base),U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner),pair,tap)
      %exact : {Tree.read(Tile.Tile16,copied,U32.from_nat(_)) ==
        Bool.pick(Tile.Tile16,Nat.is_lt(pair,U32.to_nat(Gemm.pairs_of(channels))),sample,Tile.filled_pair(0.0)) : Tile.Tile16}
      %Equal.sym(Bool,Nat.is_lt(pair,U32.to_nat(Gemm.pairs_of(channels))),True{},pair_inside) :
        {Tree.read(Tile.Tile16,copied,U32.from_nat(Nat.add(Nat.mul(pair,U32.to_nat(inner)),tap))) == Bool.pick(Tile.Tile16,_,sample,Tile.filled_pair(0.0)) : Tile.Tile16}
      copied_inside(~Source,~value,source,limits,channels,inner,base,holding,cells,total,cells_nat,room,contains,window,space,pair,tap,pair_inside,tap_inside)
    case Beyond{+beyond}:
      copied_beyond(limits,channels,inner,Buffer.tree(~Tile.Tile8,PointWindows.copy_model(holding,base,cells)),space,query,query_inside,beyond)

# ---- Every query ----

# A rectangle of channels and inner whose weights a window holds from base.
type PairCase<-Source: Data,-value: Source -> U32 -> Tile.Tile8> is Data:
  PairCase{source: Source,limits: Machine.Limits,channels: U32,inner: U32,base: U32,holding: Held.Holding<Tile.Tile8>,cells: U32,total: Nat,
    cells_nat: {U32.to_nat(cells) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)) : Nat},
    room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    contains: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(cells)),total) == True{} : Bool},
    window: Halo.TaskWindow<Tile.Tile8,Source,value,source,holding,base,(base + cells : U32),total>,space: Operands.PairSpace<limits,channels,inner>}

def case_held(~Source: Data,~value: Source -> U32 -> Tile.Tile8,context: PairCase<Source,value>) -> Tree.Tree<Tile.Tile16>:
  PairCase{source,limits,+channels,+inner,+base,+holding,cells,total,cells_nat,room,contains,window,space} = context
  held_tree(channels,inner,base,holding)

def case_copied(~Source: Data,~value: Source -> U32 -> Tile.Tile8,context: PairCase<Source,value>) -> Tree.Tree<Tile.Tile16>:
  PairCase{source,limits,+channels,+inner,+base,+holding,+cells,total,cells_nat,room,contains,window,space} = context
  copied_tree(channels,inner,PointWindows.copy_model(holding,base,cells))

def case_depth(~Source: Data,~value: Source -> U32 -> Tile.Tile8,context: PairCase<Source,value>) -> Nat:
  PairCase{source,limits,+channels,+inner,base,holding,cells,total,cells_nat,room,contains,window,space} = context
  pairs_depth(channels,inner)

# The held pairs at query 0 show their shape.
def held_first(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,+channels: U32,+inner: U32,+base: U32,
  +holding: Held.Holding<Tile.Tile8>,+cells: U32,+total: Nat,
  +cells_nat: {U32.to_nat(cells) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)) : Nat},
  +room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(cells)),total) == True{} : Bool},
  +window: Halo.TaskWindow<Tile.Tile8,Source,value,source,holding,base,(base + cells : U32),total>,+space: Operands.PairSpace<limits,channels,inner>) ->
  PairsAt<pairs_depth(channels,inner),Buffer.length(~Tile.Tile16,Point.fresh_pairs(channels,inner)),0n,Products.held_pairs(U32.to_nat(Gemm.pairs_of(channels)),U32.from_nat(0n),Gemm.blocks(channels),inner,base,start_state(channels,inner,holding)),
    expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner),U32.to_nat(Gemm.pairs_of(channels)),0n,query_kind(U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(inner),0n),U32.to_nat(Gemm.pairs_of(channels)))>:
  +depth = pairs_depth(channels,inner)
  held_at(~Source,~value,source,limits,channels,inner,base,holding,cells,total,cells_nat,room,contains,window,space,0n,
    %Equal.sym(Nat,U32.to_nat(Address.capacity(depth)),Shape.extent(depth),Address.capacity_value(depth,word_limit(limits,channels,inner,space))) :
      {Nat.is_lt(0n,_) == True{} : Bool} extent_positive(depth),
    query_kind(U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(inner),0n))

def held_shape(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,+channels: U32,+inner: U32,+base: U32,
  +holding: Held.Holding<Tile.Tile8>,+cells: U32,+total: Nat,
  +cells_nat: {U32.to_nat(cells) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)) : Nat},
  +room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(cells)),total) == True{} : Bool},
  +window: Halo.TaskWindow<Tile.Tile8,Source,value,source,holding,base,(base + cells : U32),total>,+space: Operands.PairSpace<limits,channels,inner>) ->
  Shape.Balanced(Tile.Tile16,pairs_depth(channels,inner),held_tree(channels,inner,base,holding)):
  seen_balanced(pairs_depth(channels,inner),Buffer.length(~Tile.Tile16,Point.fresh_pairs(channels,inner)),0n,Products.held_pairs(U32.to_nat(Gemm.pairs_of(channels)),U32.from_nat(0n),Gemm.blocks(channels),inner,base,start_state(channels,inner,holding)),expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner),U32.to_nat(Gemm.pairs_of(channels)),0n,query_kind(U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(inner),0n),U32.to_nat(Gemm.pairs_of(channels))),held_first(~Source,~value,source,limits,channels,inner,base,holding,cells,total,cells_nat,room,contains,window,space))

def held_length(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,+channels: U32,+inner: U32,+base: U32,
  +holding: Held.Holding<Tile.Tile8>,+cells: U32,+total: Nat,
  +cells_nat: {U32.to_nat(cells) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)) : Nat},
  +room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(cells)),total) == True{} : Bool},
  +window: Halo.TaskWindow<Tile.Tile8,Source,value,source,holding,base,(base + cells : U32),total>,+space: Operands.PairSpace<limits,channels,inner>) ->
  {Buffer.content_length(~Tile.Tile16,Buffer.contents(~Tile.Tile16,Products.held_pairs(U32.to_nat(Gemm.pairs_of(channels)),U32.from_nat(0n),Gemm.blocks(channels),inner,base,start_state(channels,inner,holding)))) == Buffer.length(~Tile.Tile16,Point.fresh_pairs(channels,inner)) : Nat}:
  seen_long(pairs_depth(channels,inner),Buffer.length(~Tile.Tile16,Point.fresh_pairs(channels,inner)),0n,Products.held_pairs(U32.to_nat(Gemm.pairs_of(channels)),U32.from_nat(0n),Gemm.blocks(channels),inner,base,start_state(channels,inner,holding)),expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner),U32.to_nat(Gemm.pairs_of(channels)),0n,query_kind(U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(inner),0n),U32.to_nat(Gemm.pairs_of(channels))),held_first(~Source,~value,source,limits,channels,inner,base,holding,cells,total,cells_nat,room,contains,window,space))

def copied_shape(+channels: U32,+inner: U32,+weights: Buffer.Model<Tile.Tile8>) ->
  Shape.Balanced(Tile.Tile16,pairs_depth(channels,inner),copied_tree(channels,inner,weights)):
  Updates.update_balanced(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile16,~Gemm.PairPlan,~PointBuffer.pair_sample,~Gemm.replaced(~Tile.Tile16),~Gemm.pair_placed,
    PointBuffer.pair_count(channels,inner),0,Buffer.tree(~Tile.Tile8,weights),Gemm.PairPlan{Gemm.blocks(channels),inner},
    Buffer.tree(~Tile.Tile16,Point.fresh_pairs(channels,inner)),pairs_depth(channels,inner),fresh_balanced(channels,inner))

# The held and the copied pairs agree at every query inside their depth.
def case_same(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+context: PairCase<Source,value>,+index: U32,
  +inside: {U32.is_lt(index,Address.capacity(case_depth(~Source,~value,context))) == True{} : Bool}) ->
  {Tree.fetch(Tile.Tile16,case_held(~Source,~value,context),Address.capacity(case_depth(~Source,~value,context)),index) ==
    Tree.fetch(Tile.Tile16,case_copied(~Source,~value,context),Address.capacity(case_depth(~Source,~value,context)),index) : Tile.Tile16}:
  PairCase{+source,+limits,+channels,+inner,+base,+holding,+cells,+total,+cells_nat,+room,+contains,+window,+space} = context
  +depth = pairs_depth(channels,inner)
  +limit = word_limit(limits,channels,inner,space)
  +query = U32.to_nat(index)
  +pairs = U32.to_nat(Gemm.pairs_of(channels))
  +whole = Address.capacity(depth)
  +query_inside = {%Equal.sym(Nat,U32.to_nat(whole),Shape.extent(depth),Address.capacity_value(depth,limit)) : {Nat.is_lt(query,_) == True{} : Bool}
    Address.natural_bound(depth,index,limit,inside) : {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}}
  +kind = query_kind(pairs,U32.to_nat(inner),query)
  +held = held_tree(channels,inner,base,holding)
  +copied = copied_tree(channels,inner,PointWindows.copy_model(holding,base,cells))
  +wanted = expected(~Source,~value,source,U32.to_nat(base),U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner),pairs,query,kind,pairs)
  Equal.trans(Tile.Tile16,Tree.fetch(Tile.Tile16,held,whole,index),Tree.read(Tile.Tile16,held,index),Tree.fetch(Tile.Tile16,copied,whole,index),
    Equal.sym(Tile.Tile16,Tree.read(Tile.Tile16,held,index),Tree.fetch(Tile.Tile16,held,whole,index),
      TreeEquality.read_fetch(~Tile.Tile16,depth,held,index,held_shape(~Source,~value,source,limits,channels,inner,base,holding,cells,total,cells_nat,room,contains,window,space),inside)),
    Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,held,index),Tree.read(Tile.Tile16,copied,index),Tree.fetch(Tile.Tile16,copied,whole,index),
      %Arithmetic.native_roundtrip(index) : {Tree.read(Tile.Tile16,held,_) == Tree.read(Tile.Tile16,copied,_) : Tile.Tile16}
      Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,held,U32.from_nat(query)),wanted,Tree.read(Tile.Tile16,copied,U32.from_nat(query)),
        seen_read(depth,Buffer.length(~Tile.Tile16,Point.fresh_pairs(channels,inner)),query,
          Products.held_pairs(pairs,U32.from_nat(0n),Gemm.blocks(channels),inner,base,start_state(channels,inner,holding)),wanted,
          held_at(~Source,~value,source,limits,channels,inner,base,holding,cells,total,cells_nat,room,contains,window,space,query,query_inside,kind)),
        Equal.sym(Tile.Tile16,Tree.read(Tile.Tile16,copied,U32.from_nat(query)),wanted,
          copied_at(~Source,~value,source,limits,channels,inner,base,holding,cells,total,cells_nat,room,contains,window,space,query,query_inside,kind))),
      TreeEquality.read_fetch(~Tile.Tile16,depth,copied,index,copied_shape(channels,inner,PointWindows.copy_model(holding,base,cells)),inside)))

# The held pairs of a rectangle are the pairs read from its copied weights.
def held_pairs_copied(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,+first: U32,+positions: U32,+inner: U32,
  +channels: U32,+major: Bool,+grid: Grid.Grid,+base: U32,+holding: Held.Holding<Tile.Tile8>,+cells: U32,+total: Nat,
  +cells_nat: {U32.to_nat(cells) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)) : Nat},
  +room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(cells)),total) == True{} : Bool},
  +window: Halo.TaskWindow<Tile.Tile8,Source,value,source,holding,base,(base + cells : U32),total>,+space: Operands.PairSpace<limits,channels,inner>) ->
  {Point.held_pairs_model(Products.Rectangle{first,positions,inner,channels,major,grid},base,Held.pack(~Tile.Tile8,holding)) ==
    Point.shared_pair_model(Products.Rectangle{first,positions,inner,channels,major,grid},PointWindows.copy_model(holding,base,cells)) : Buffer.Content<Tile.Tile16>}:
  +context = {PairCase{source,limits,channels,inner,base,holding,cells,total,cells_nat,room,contains,window,space} : PairCase<Source,value>}
  +held = Point.held_pairs_model(Products.Rectangle{first,positions,inner,channels,major,grid},base,Held.pack(~Tile.Tile8,holding))
  +copied = copied_tree(channels,inner,PointWindows.copy_model(holding,base,cells))
  +length = Buffer.length(~Tile.Tile16,Point.fresh_pairs(channels,inner))
  +trees = TreeEquality.same_trees(~Tile.Tile16,~PairCase<Source,value>,~case_held(~Source,~value),~case_copied(~Source,~value),~case_depth(~Source,~value),
    ~case_same(~Source,~value),context,held_shape(~Source,~value,source,limits,channels,inner,base,holding,cells,total,cells_nat,room,contains,window,space),copied_shape(channels,inner,PointWindows.copy_model(holding,base,cells)),word_limit(limits,channels,inner,space))
  Equal.trans(Buffer.Content<Tile.Tile16>,held,Buffer.Content{Buffer.content_tree(~Tile.Tile16,held),Buffer.content_length(~Tile.Tile16,held)},
    Buffer.Content{copied,length},
    Buffer.content_eta(~Tile.Tile16,held),
    %Equal.sym(Tree.Tree<Tile.Tile16>,Buffer.content_tree(~Tile.Tile16,held),copied,trees) :
      {Buffer.Content{_,Buffer.content_length(~Tile.Tile16,held)} == Buffer.Content{copied,length} : Buffer.Content<Tile.Tile16>}
    %Equal.sym(Nat,Buffer.content_length(~Tile.Tile16,held),length,held_length(~Source,~value,source,limits,channels,inner,base,holding,cells,total,cells_nat,room,contains,window,space)) :
      {Buffer.Content{copied,_} == Buffer.Content{copied,length} : Buffer.Content<Tile.Tile16>}
    {==})
