# One block of a relane: eight source cells read from the pieces a task
# holds, transposed, and written to the target cells of the block. Every
# read gives the value of its cell and keeps the pieces in their order.

import Base
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../tensor_move.bend as Move
import ../tensor_tiled.bend as Tiled
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_move_readback.bend as Held
import ./boolean_logic.bend as Logic
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./kernel_tile_refinement.bend as TileLaws
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./natural_addition.bend as Addition
import ./word_subtraction.bend as Subtraction
import ./word_encoding.bend as WordEncoding
import ./tensor_move_part.bend as MovePart
import ./row_major_index.bend as RowMajor

# ---- A read on held pieces ----

# A read leaves the holding it resolved.
def current_held(~Element: Data,held: Bool,+index: U32,+fallback: Element,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,
  +below: Held.Stack<Element>) ->
  {Held.read_held(~Element,held,index,fallback,behind,buffer,first,span,below) ==
    (Held.Holding{behind,Held.Held{buffer,first,span,below}},Pair.snd(Held.Holding<Element>,Element,Held.read_held(~Element,held,index,fallback,behind,buffer,first,span,below))) :
    Held.Read(~Element)}:
  match held:
    case True{}: {==}
    case False{}: {==}

def current_pair(~Element: Data,+holding: Held.Holding<Element>,+index: U32,+fallback: Element) ->
  {Held.read_current(~Element,holding,index,fallback) == (holding,Pair.snd(Held.Holding<Element>,Element,Held.read_current(~Element,holding,index,fallback))) : Held.Read(~Element)}:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+span,+below}}:
      current_held(~Element,U32.is_le(first,index) && U32.is_lt(index,(first + span : U32)),index,fallback,behind,buffer,first,span,below)
    case Held.Holding{+behind,Held.Bottom{}}: {==}

# A read of packed pieces is the packed resolved holding and the value the
# model reads.
def read_pair(~Element: Data,+holding: Held.Holding<Element>,+index: U32,+fallback: Element) ->
  {Move.read(~Element,Held.pack(~Element,holding),index,fallback) ==
    (Held.pack(~Element,Held.resolve(~Element,holding,index)),Pair.snd(Held.Holding<Element>,Element,Held.read(~Element,holding,index,fallback))) :
    Move.Holding<Element> & Element}:
  +resolved = Held.resolve(~Element,holding,index)
  Equal.trans(Move.Holding<Element> & Element,Move.read(~Element,Held.pack(~Element,holding),index,fallback),Held.pack_read(~Element,Held.read(~Element,holding,index,fallback)),
    (Held.pack(~Element,resolved),Pair.snd(Held.Holding<Element>,Element,Held.read(~Element,holding,index,fallback))),
    Held.read_packed(~Element,holding,index,fallback),
    Equal.cong(Held.Read(~Element),Move.Holding<Element> & Element,Held.pack_read(~Element),Held.read_current(~Element,resolved,index,fallback),
      (resolved,Pair.snd(Held.Holding<Element>,Element,Held.read_current(~Element,resolved,index,fallback))),current_pair(~Element,resolved,index,fallback)))

# On pieces in order that hold a cell with the values of a reader, a read of
# the cell gives its value.
def sorted_read(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Held.Holding<Element>,+index: U32,+fallback: Element,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Element,Held.order(~Element,holding),low,high),+held: Held.SpanHeld(Held.spans(~Element,Held.order(~Element,holding)),U32.to_nat(index)),
  +values: Held.HoldingValues(~Element,~Context,~value,context,holding)) ->
  {Pair.snd(Held.Holding<Element>,Element,Held.read(~Element,holding,index,fallback)) == value(context,index) : Element}:
  Held.read_current_value(~Element,~Context,~value,context,Held.resolve(~Element,holding,index),index,fallback,capacity,capacity_ok,
    Held.resolve_current(~Element,holding,index,low,high,capacity,capacity_ok,bounded,sorted,held),
    Held.resolve_values(~Element,~Context,~value,context,holding,index,values))

# ---- The eight reads of a block ----

def zero() -> Tile.Tile8:
  Tiled.zero_tile()

# What read index of a block leaves of a reader model, and the cell it gives:
# +0 past the cells it reads.
def next_model(~Model: Data,~moved: Model -> U32 -> Model,+reads: U32,+source: U32,+step: U32,+index: U32,+model: Model) -> Model:
  Bool.pick(Model,U32.is_lt(index,reads),moved(model,(source + index * step : U32)),model)

def next_cell(~Model: Data,~cell: Model -> U32 -> Tile.Tile8,+reads: U32,+source: U32,+step: U32,+index: U32,+model: Model) -> Tile.Tile8:
  Bool.pick(Tile.Tile8,U32.is_lt(index,reads),cell(model,(source + index * step : U32)),zero())

# The model after the eight reads of a block and the eight cells.
def eight_model(~Model: Data,~moved: Model -> U32 -> Model,~cell: Model -> U32 -> Tile.Tile8,+reads: U32,+source: U32,+step: U32,+start: Model) -> Model & Tiled.Eight:
  +m1 = next_model(~Model,~moved,reads,source,step,0,start)
  +m2 = next_model(~Model,~moved,reads,source,step,1,m1)
  +m3 = next_model(~Model,~moved,reads,source,step,2,m2)
  +m4 = next_model(~Model,~moved,reads,source,step,3,m3)
  +m5 = next_model(~Model,~moved,reads,source,step,4,m4)
  +m6 = next_model(~Model,~moved,reads,source,step,5,m5)
  +m7 = next_model(~Model,~moved,reads,source,step,6,m6)
  (next_model(~Model,~moved,reads,source,step,7,m7),Tiled.Eight{next_cell(~Model,~cell,reads,source,step,0,start),next_cell(~Model,~cell,reads,source,step,1,m1),
    next_cell(~Model,~cell,reads,source,step,2,m2),next_cell(~Model,~cell,reads,source,step,3,m3),next_cell(~Model,~cell,reads,source,step,4,m4),
    next_cell(~Model,~cell,reads,source,step,5,m5),next_cell(~Model,~cell,reads,source,step,6,m6),next_cell(~Model,~cell,reads,source,step,7,m7)})

def pack_eight(~Model: Data,~Reader: Type,~pack: Model -> Reader,pair: Model & Tiled.Eight) -> Reader & Tiled.Eight:
  (model,eight) = pair
  (pack(model),eight)

# A wanted read of a packed model is the packed read of the model.
def wanted_packed(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,~Model: Data,~pack: Model -> Reader,~moved: Model -> U32 -> Model,~cell: Model -> U32 -> Tile.Tile8,
  ~packed: @+model: Model -> @+index: U32 -> {read(pack(model),index) == (pack(moved(model,index)),cell(model,index)) : Reader & Tile.Tile8},
  wanted: Bool,+source: U32,+step: U32,+index: U32,+model: Model) ->
  {Tiled.block_read(~Reader,~read,wanted,source,step,index,pack(model)) ==
    (pack(Bool.pick(Model,wanted,moved(model,(source + index * step : U32)),model)),Bool.pick(Tile.Tile8,wanted,cell(model,(source + index * step : U32)),zero())) :
    Reader & Tile.Tile8}:
  match wanted:
    case True{}: packed(model,(source + index * step : U32))
    case False{}: {==}

def next_packed(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,~Model: Data,~pack: Model -> Reader,~moved: Model -> U32 -> Model,~cell: Model -> U32 -> Tile.Tile8,
  ~packed: @+model: Model -> @+index: U32 -> {read(pack(model),index) == (pack(moved(model,index)),cell(model,index)) : Reader & Tile.Tile8},
  +reads: U32,+source: U32,+step: U32,+index: U32,+model: Model) ->
  {Tiled.read_next(~Reader,~read,reads,source,step,index,pack(model)) ==
    (pack(next_model(~Model,~moved,reads,source,step,index,model)),next_cell(~Model,~cell,reads,source,step,index,model)) : Reader & Tile.Tile8}:
  wanted_packed(~Reader,~read,~Model,~pack,~moved,~cell,~packed,U32.is_lt(index,reads),source,step,index,model)

# The eight reads of a block on a packed model are the packed model reads.
def eight_packed(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,~Model: Data,~pack: Model -> Reader,~moved: Model -> U32 -> Model,~cell: Model -> U32 -> Tile.Tile8,
  ~packed: @+model: Model -> @+index: U32 -> {read(pack(model),index) == (pack(moved(model,index)),cell(model,index)) : Reader & Tile.Tile8},
  +reads: U32,+source: U32,+step: U32,+start: Model) ->
  {Tiled.eight_read(~Reader,~read,reads,source,step,pack(start)) ==
    pack_eight(~Model,~Reader,~pack,eight_model(~Model,~moved,~cell,reads,source,step,start)) : Reader & Tiled.Eight}:
  +m1 = next_model(~Model,~moved,reads,source,step,0,start)
  +m2 = next_model(~Model,~moved,reads,source,step,1,m1)
  +m3 = next_model(~Model,~moved,reads,source,step,2,m2)
  +m4 = next_model(~Model,~moved,reads,source,step,3,m3)
  +m5 = next_model(~Model,~moved,reads,source,step,4,m4)
  +m6 = next_model(~Model,~moved,reads,source,step,5,m5)
  +m7 = next_model(~Model,~moved,reads,source,step,6,m6)
  +m8 = next_model(~Model,~moved,reads,source,step,7,m7)
  +v0 = next_cell(~Model,~cell,reads,source,step,0,start)
  +v1 = next_cell(~Model,~cell,reads,source,step,1,m1)
  +v2 = next_cell(~Model,~cell,reads,source,step,2,m2)
  +v3 = next_cell(~Model,~cell,reads,source,step,3,m3)
  +v4 = next_cell(~Model,~cell,reads,source,step,4,m4)
  +v5 = next_cell(~Model,~cell,reads,source,step,5,m5)
  +v6 = next_cell(~Model,~cell,reads,source,step,6,m6)
  +v7 = next_cell(~Model,~cell,reads,source,step,7,m7)
  %Equal.sym(Reader & Tile.Tile8,Tiled.read_next(~Reader,~read,reads,source,step,0,pack(start)),(pack(m1),v0),
      next_packed(~Reader,~read,~Model,~pack,~moved,~cell,~packed,reads,source,step,0,start)) :
    {Tiled.after_first(~Reader,~read,reads,source,step,_) == (pack(m8),Tiled.Eight{v0,v1,v2,v3,v4,v5,v6,v7}) : Reader & Tiled.Eight}
  %Equal.sym(Reader & Tile.Tile8,Tiled.read_next(~Reader,~read,reads,source,step,1,pack(m1)),(pack(m2),v1),
      next_packed(~Reader,~read,~Model,~pack,~moved,~cell,~packed,reads,source,step,1,m1)) :
    {Tiled.after_second(~Reader,~read,reads,source,step,v0,_) == (pack(m8),Tiled.Eight{v0,v1,v2,v3,v4,v5,v6,v7}) : Reader & Tiled.Eight}
  %Equal.sym(Reader & Tile.Tile8,Tiled.read_next(~Reader,~read,reads,source,step,2,pack(m2)),(pack(m3),v2),
      next_packed(~Reader,~read,~Model,~pack,~moved,~cell,~packed,reads,source,step,2,m2)) :
    {Tiled.after_third(~Reader,~read,reads,source,step,v0,v1,_) == (pack(m8),Tiled.Eight{v0,v1,v2,v3,v4,v5,v6,v7}) : Reader & Tiled.Eight}
  %Equal.sym(Reader & Tile.Tile8,Tiled.read_next(~Reader,~read,reads,source,step,3,pack(m3)),(pack(m4),v3),
      next_packed(~Reader,~read,~Model,~pack,~moved,~cell,~packed,reads,source,step,3,m3)) :
    {Tiled.after_fourth(~Reader,~read,reads,source,step,v0,v1,v2,_) == (pack(m8),Tiled.Eight{v0,v1,v2,v3,v4,v5,v6,v7}) : Reader & Tiled.Eight}
  %Equal.sym(Reader & Tile.Tile8,Tiled.read_next(~Reader,~read,reads,source,step,4,pack(m4)),(pack(m5),v4),
      next_packed(~Reader,~read,~Model,~pack,~moved,~cell,~packed,reads,source,step,4,m4)) :
    {Tiled.after_fifth(~Reader,~read,reads,source,step,v0,v1,v2,v3,_) == (pack(m8),Tiled.Eight{v0,v1,v2,v3,v4,v5,v6,v7}) : Reader & Tiled.Eight}
  %Equal.sym(Reader & Tile.Tile8,Tiled.read_next(~Reader,~read,reads,source,step,5,pack(m5)),(pack(m6),v5),
      next_packed(~Reader,~read,~Model,~pack,~moved,~cell,~packed,reads,source,step,5,m5)) :
    {Tiled.after_sixth(~Reader,~read,reads,source,step,v0,v1,v2,v3,v4,_) == (pack(m8),Tiled.Eight{v0,v1,v2,v3,v4,v5,v6,v7}) : Reader & Tiled.Eight}
  %Equal.sym(Reader & Tile.Tile8,Tiled.read_next(~Reader,~read,reads,source,step,6,pack(m6)),(pack(m7),v6),
      next_packed(~Reader,~read,~Model,~pack,~moved,~cell,~packed,reads,source,step,6,m6)) :
    {Tiled.after_seventh(~Reader,~read,reads,source,step,v0,v1,v2,v3,v4,v5,_) == (pack(m8),Tiled.Eight{v0,v1,v2,v3,v4,v5,v6,v7}) : Reader & Tiled.Eight}
  %Equal.sym(Reader & Tile.Tile8,Tiled.read_next(~Reader,~read,reads,source,step,7,pack(m7)),(pack(m8),v7),
      next_packed(~Reader,~read,~Model,~pack,~moved,~cell,~packed,reads,source,step,7,m7)) :
    {Tiled.after_eighth(~Reader,v0,v1,v2,v3,v4,v5,v6,_) == (pack(m8),Tiled.Eight{v0,v1,v2,v3,v4,v5,v6,v7}) : Reader & Tiled.Eight}
  {==}

# Reads through held pieces: each resolves the piece of its cell.
def held_moved(holding: Held.Holding<Tile.Tile8>,index: U32) -> Held.Holding<Tile.Tile8>:
  Held.resolve(~Tile.Tile8,holding,index)

def held_value(holding: Held.Holding<Tile.Tile8>,index: U32) -> Tile.Tile8:
  Pair.snd(Held.Holding<Tile.Tile8>,Tile.Tile8,Held.read(~Tile.Tile8,holding,index,zero()))

def held_pack(holding: Held.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8>:
  Held.pack(~Tile.Tile8,holding)

def held_packed(+holding: Held.Holding<Tile.Tile8>,+index: U32) ->
  {Tiled.held_cell(held_pack(holding),index) == (held_pack(held_moved(holding,index)),held_value(holding,index)) : Move.Holding<Tile.Tile8> & Tile.Tile8}:
  read_pair(~Tile.Tile8,holding,index,zero())

# Reads of one piece: they keep its buffer.
def piece_kept(buffer: Buffer.Model<Tile.Tile8>,index: U32) -> Buffer.Model<Tile.Tile8>:
  buffer

def piece_value(buffer: Buffer.Model<Tile.Tile8>,index: U32) -> Tile.Tile8:
  Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,buffer),index)

def piece_pack(buffer: Buffer.Model<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Buffer.pack(~Tile.Tile8,buffer)

def piece_packed(+buffer: Buffer.Model<Tile.Tile8>,+index: U32) ->
  {Tiled.piece_cell(piece_pack(buffer),index) == (piece_pack(piece_kept(buffer,index)),piece_value(buffer,index)) : Storage.Buffer<Tile.Tile8> & Tile.Tile8}:
  Buffer.read(~Tile.Tile8,buffer,index)

# The holding after read index of a block, and the cell it reads.
def read_holding(+reads: U32,+source: U32,+step: U32,+index: U32,+holding: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  next_model(~Held.Holding<Tile.Tile8>,~held_moved,reads,source,step,index,holding)

def read_cell(+reads: U32,+source: U32,+step: U32,+index: U32,+holding: Held.Holding<Tile.Tile8>) -> Tile.Tile8:
  next_cell(~Held.Holding<Tile.Tile8>,~held_value,reads,source,step,index,holding)

# The eight reads of a block through held pieces.
def reads_model(+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8> & Tiled.Eight:
  eight_model(~Held.Holding<Tile.Tile8>,~held_moved,~held_value,reads,source,step,start)

def pack_reads(pair: Held.Holding<Tile.Tile8> & Tiled.Eight) -> Move.Holding<Tile.Tile8> & Tiled.Eight:
  (holding,eight) = pair
  (Held.pack(~Tile.Tile8,holding),eight)

# The reads of a block that one piece holds whole read its buffer from the
# cell of the source in it.
def piece_returned(+behind: Held.Stack<Tile.Tile8>,+first: U32,+count: U32,+below: Held.Stack<Tile.Tile8>,pair: Buffer.Model<Tile.Tile8> & Tiled.Eight) ->
  Held.Holding<Tile.Tile8> & Tiled.Eight:
  (buffer,eight) = pair
  (Held.Holding{behind,Held.Held{buffer,first,count,below}},eight)

def piece_eight(+reads: U32,+offset: U32,+step: U32,+buffer: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8> & Tiled.Eight:
  eight_model(~Buffer.Model<Tile.Tile8>,~piece_kept,~piece_value,reads,offset,step,buffer)

def piece_model(inside: Bool,+reads: U32,+source: U32,+step: U32,+behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,
  +below: Held.Stack<Tile.Tile8>) -> Held.Holding<Tile.Tile8> & Tiled.Eight:
  match inside:
    case True{}: piece_returned(behind,first,count,below,piece_eight(reads,(source - first : U32),step,buffer))
    case False{}: reads_model(reads,source,step,Held.Holding{behind,Held.Held{buffer,first,count,below}})

def current_model(+reads: U32,+source: U32,+step: U32,holding: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8> & Tiled.Eight:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+count,+below}}:
      piece_model(U32.is_le(reads,Move.inside(reads,source,step,False{},first,count)),reads,source,step,behind,buffer,first,count,below)
    case Held.Holding{+behind,Held.Bottom{}}: reads_model(reads,source,step,Held.Holding{behind,Held.Bottom{}})

# The model of the eight reads of a block: from the piece of its first cell.
def block_model(+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8> & Tiled.Eight:
  current_model(reads,source,step,Held.resolve(~Tile.Tile8,start,source))

def piece_packed_block(inside: Bool,+reads: U32,+source: U32,+step: U32,+behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,
  +below: Held.Stack<Tile.Tile8>) ->
  {Tiled.piece_block(inside,reads,source,step,Held.pack_stack(~Tile.Tile8,behind),Buffer.pack(~Tile.Tile8,buffer),first,count,Held.pack_stack(~Tile.Tile8,below)) ==
    pack_reads(piece_model(inside,reads,source,step,behind,buffer,first,count,below)) : Move.Holding<Tile.Tile8> & Tiled.Eight}:
  match inside:
    case True{}:
      %Equal.sym(Storage.Buffer<Tile.Tile8> & Tiled.Eight,Tiled.eight_read(~Storage.Buffer<Tile.Tile8>,~Tiled.piece_cell,reads,(source - first : U32),step,piece_pack(buffer)),
          pack_eight(~Buffer.Model<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~piece_pack,piece_eight(reads,(source - first : U32),step,buffer)),
          eight_packed(~Storage.Buffer<Tile.Tile8>,~Tiled.piece_cell,~Buffer.Model<Tile.Tile8>,~piece_pack,~piece_kept,~piece_value,~piece_packed,reads,(source - first : U32),step,buffer)) :
        {Tiled.piece_returned(Held.pack_stack(~Tile.Tile8,behind),first,count,Held.pack_stack(~Tile.Tile8,below),_) ==
          pack_reads(piece_model(True{},reads,source,step,behind,buffer,first,count,below)) : Move.Holding<Tile.Tile8> & Tiled.Eight}
      {==}
    case False{}:
      eight_packed(~Move.Holding<Tile.Tile8>,~Tiled.held_cell,~Held.Holding<Tile.Tile8>,~held_pack,~held_moved,~held_value,~held_packed,reads,source,step,
        Held.Holding{behind,Held.Held{buffer,first,count,below}})

def current_packed(+reads: U32,+source: U32,+step: U32,+holding: Held.Holding<Tile.Tile8>) ->
  {Tiled.current_block(reads,source,step,Held.pack(~Tile.Tile8,holding)) == pack_reads(current_model(reads,source,step,holding)) : Move.Holding<Tile.Tile8> & Tiled.Eight}:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+count,+below}}:
      piece_packed_block(U32.is_le(reads,Move.inside(reads,source,step,False{},first,count)),reads,source,step,behind,buffer,first,count,below)
    case Held.Holding{+behind,Held.Bottom{}}:
      eight_packed(~Move.Holding<Tile.Tile8>,~Tiled.held_cell,~Held.Holding<Tile.Tile8>,~held_pack,~held_moved,~held_value,~held_packed,reads,source,step,
        Held.Holding{behind,Held.Bottom{}})

# The eight reads of a block on packed pieces are the packed model reads.
def block_reads_packed(+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>) ->
  {Tiled.block_reads(reads,source,step,Held.pack(~Tile.Tile8,start)) == pack_reads(block_model(reads,source,step,start)) : Move.Holding<Tile.Tile8> & Tiled.Eight}:
  %Equal.sym(Move.Holding<Tile.Tile8>,Move.resolve(~Tile.Tile8,Held.pack(~Tile.Tile8,start),source),Held.pack(~Tile.Tile8,Held.resolve(~Tile.Tile8,start,source)),
      Held.resolve_packed(~Tile.Tile8,start,source)) :
    {Tiled.current_block(reads,source,step,_) == pack_reads(block_model(reads,source,step,start)) : Move.Holding<Tile.Tile8> & Tiled.Eight}
  current_packed(reads,source,step,Held.resolve(~Tile.Tile8,start,source))

# A pick between a model and itself is the model.
def kept_pick(~Model: Data,choice: Bool,+model: Model) -> {Bool.pick(Model,choice,model,model) == model : Model}:
  match choice:
    case True{}: {==}
    case False{}: {==}

# The reads of one piece keep its buffer.
def piece_kept_eight(+reads: U32,+offset: U32,+step: U32,+buffer: Buffer.Model<Tile.Tile8>) ->
  {Pair.fst(Buffer.Model<Tile.Tile8>,Tiled.Eight,piece_eight(reads,offset,step,buffer)) == buffer : Buffer.Model<Tile.Tile8>}:
  +m1 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,0,buffer)
  +m2 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,1,m1)
  +m3 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,2,m2)
  +m4 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,3,m3)
  +m5 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,4,m4)
  +m6 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,5,m5)
  +m7 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,6,m6)
  +e1 = kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(0,reads),buffer)
  +e2 = Equal.trans(Buffer.Model<Tile.Tile8>,m2,m1,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(1,reads),m1),e1)
  +e3 = Equal.trans(Buffer.Model<Tile.Tile8>,m3,m2,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(2,reads),m2),e2)
  +e4 = Equal.trans(Buffer.Model<Tile.Tile8>,m4,m3,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(3,reads),m3),e3)
  +e5 = Equal.trans(Buffer.Model<Tile.Tile8>,m5,m4,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(4,reads),m4),e4)
  +e6 = Equal.trans(Buffer.Model<Tile.Tile8>,m6,m5,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(5,reads),m5),e5)
  +e7 = Equal.trans(Buffer.Model<Tile.Tile8>,m7,m6,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(6,reads),m6),e6)
  Equal.trans(Buffer.Model<Tile.Tile8>,next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,7,m7),m7,buffer,
    kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(7,reads),m7),e7)

# The reads of one piece leave the pieces as they were.
def piece_holding(+reads: U32,+offset: U32,+step: U32,+behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,+below: Held.Stack<Tile.Tile8>) ->
  {Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,piece_returned(behind,first,count,below,piece_eight(reads,offset,step,buffer))) ==
    Held.Holding{behind,Held.Held{buffer,first,count,below}} : Held.Holding<Tile.Tile8>}:
  Equal.cong(Buffer.Model<Tile.Tile8>,Held.Holding<Tile.Tile8>,model => Held.Holding{behind,Held.Held{model,first,count,below}},
    Pair.fst(Buffer.Model<Tile.Tile8>,Tiled.Eight,piece_eight(reads,offset,step,buffer)),buffer,piece_kept_eight(reads,offset,step,buffer))

# ---- What the reads of a block give ----

def picked_order(~Element: Data,wanted: Bool,+index: U32,+holding: Held.Holding<Element>) ->
  {Held.order(~Element,Bool.pick(Held.Holding<Element>,wanted,Held.resolve(~Element,holding,index),holding)) == Held.order(~Element,holding) : Held.Stack<Element>}:
  match wanted:
    case True{}: Held.resolve_order(~Element,holding,index)
    case False{}: {==}

def picked_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,wanted: Bool,+index: U32,+holding: Held.Holding<Element>,
  +values: Held.HoldingValues(~Element,~Context,~value,context,holding)) ->
  Held.HoldingValues(~Element,~Context,~value,context,Bool.pick(Held.Holding<Element>,wanted,Held.resolve(~Element,holding,index),holding)):
  match wanted:
    case True{}: Held.resolve_values(~Element,~Context,~value,context,holding,index,values)
    case False{}: values

# What cell index of a block holds: the value of the source cell it reads,
# or +0 past the cells it reads.
def block_cell(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+reads: U32,+source: U32,+step: U32,+index: U32) -> Tile.Tile8:
  Bool.pick(Tile.Tile8,U32.is_lt(index,reads),value(context,(source + index * step : U32)),zero())

def picked_cell(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,wanted: Bool,+address: U32,+holding: Held.Holding<Tile.Tile8>,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,holding),low,high),
  +held: Logic.When(wanted,Held.SpanHeld(Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)),U32.to_nat(address))),
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,holding)) ->
  {Bool.pick(Tile.Tile8,wanted,Pair.snd(Held.Holding<Tile.Tile8>,Tile.Tile8,Held.read(~Tile.Tile8,holding,address,zero())),zero()) ==
    Bool.pick(Tile.Tile8,wanted,value(context,address),zero()) : Tile.Tile8}:
  match wanted:
    case True{}: sorted_read(~Tile.Tile8,~Context,~value,context,holding,address,zero(),low,high,capacity,capacity_ok,bounded,sorted,held,values)
    case False{}: {==}

# The source cell read index of a block reads is held when the block reads it.
def ReadAt(+ranges: Held.Spans,+reads: U32,+source: U32,+step: U32,+index: U32) -> Data:
  Logic.When(U32.is_lt(index,reads),Held.SpanHeld(ranges,U32.to_nat((source + index * step : U32))))

type ReadsHeld<-ranges: Held.Spans,-reads: U32,-source: U32,-step: U32> is Data:
  ReadsHeld{c0: ReadAt(ranges,reads,source,step,0),c1: ReadAt(ranges,reads,source,step,1),c2: ReadAt(ranges,reads,source,step,2),c3: ReadAt(ranges,reads,source,step,3),
    c4: ReadAt(ranges,reads,source,step,4),c5: ReadAt(ranges,reads,source,step,5),c6: ReadAt(ranges,reads,source,step,6),c7: ReadAt(ranges,reads,source,step,7)}

# A read of a block keeps the pieces in their order.
def step_order(+reads: U32,+source: U32,+step: U32,+index: U32,+holding: Held.Holding<Tile.Tile8>,+start: Held.Holding<Tile.Tile8>,
  +same: {Held.order(~Tile.Tile8,holding) == Held.order(~Tile.Tile8,start) : Held.Stack<Tile.Tile8>}) ->
  {Held.order(~Tile.Tile8,read_holding(reads,source,step,index,holding)) == Held.order(~Tile.Tile8,start) : Held.Stack<Tile.Tile8>}:
  Equal.trans(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,read_holding(reads,source,step,index,holding)),Held.order(~Tile.Tile8,holding),Held.order(~Tile.Tile8,start),
    picked_order(~Tile.Tile8,U32.is_lt(index,reads),(source + index * step : U32),holding),same)

# A read of a block, on pieces in the order of the start, gives the cell of
# the block.
def step_cell(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+reads: U32,+source: U32,+step: U32,+index: U32,+holding: Held.Holding<Tile.Tile8>,
  +start: Held.Holding<Tile.Tile8>,+same: {Held.order(~Tile.Tile8,holding) == Held.order(~Tile.Tile8,start) : Held.Stack<Tile.Tile8>},
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,start),low,high),+held: ReadAt(Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,start)),reads,source,step,index),
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,holding)) ->
  {read_cell(reads,source,step,index,holding) == block_cell(~Context,~value,context,reads,source,step,index) : Tile.Tile8}:
  +back = Equal.sym(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,holding),Held.order(~Tile.Tile8,start),same)
  picked_cell(~Context,~value,context,U32.is_lt(index,reads),(source + index * step : U32),holding,low,high,capacity,capacity_ok,bounded,
    Held.sorted_moved(~Tile.Tile8,Held.order(~Tile.Tile8,start),Held.order(~Tile.Tile8,holding),low,high,back,sorted),
    {%back : Logic.When(U32.is_lt(index,reads),Held.SpanHeld(Held.spans(~Tile.Tile8,_),U32.to_nat((source + index * step : U32))))
      held : Logic.When(U32.is_lt(index,reads),Held.SpanHeld(Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)),U32.to_nat((source + index * step : U32))))},
    values)

def reads_holding(+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,reads_model(reads,source,step,start))

def reads_eight(+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>) -> Tiled.Eight:
  Pair.snd(Held.Holding<Tile.Tile8>,Tiled.Eight,reads_model(reads,source,step,start))

def expected_eight(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+reads: U32,+source: U32,+step: U32) -> Tiled.Eight:
  Tiled.Eight{block_cell(~Context,~value,context,reads,source,step,0),block_cell(~Context,~value,context,reads,source,step,1),
    block_cell(~Context,~value,context,reads,source,step,2),block_cell(~Context,~value,context,reads,source,step,3),
    block_cell(~Context,~value,context,reads,source,step,4),block_cell(~Context,~value,context,reads,source,step,5),
    block_cell(~Context,~value,context,reads,source,step,6),block_cell(~Context,~value,context,reads,source,step,7)}

# The reads of a block keep the pieces in their order.
def reads_order(+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>) ->
  {Held.order(~Tile.Tile8,reads_holding(reads,source,step,start)) == Held.order(~Tile.Tile8,start) : Held.Stack<Tile.Tile8>}:
  +h1 = read_holding(reads,source,step,0,start)
  +h2 = read_holding(reads,source,step,1,h1)
  +h3 = read_holding(reads,source,step,2,h2)
  +h4 = read_holding(reads,source,step,3,h3)
  +h5 = read_holding(reads,source,step,4,h4)
  +h6 = read_holding(reads,source,step,5,h5)
  +h7 = read_holding(reads,source,step,6,h6)
  +o1 = step_order(reads,source,step,0,start,start,{==})
  +o2 = step_order(reads,source,step,1,h1,start,o1)
  +o3 = step_order(reads,source,step,2,h2,start,o2)
  +o4 = step_order(reads,source,step,3,h3,start,o3)
  +o5 = step_order(reads,source,step,4,h4,start,o4)
  +o6 = step_order(reads,source,step,5,h5,start,o5)
  +o7 = step_order(reads,source,step,6,h6,start,o6)
  step_order(reads,source,step,7,h7,start,o7)

# The reads of a block keep the values of the pieces.
def reads_values(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>,
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,start)) ->
  Held.HoldingValues(~Tile.Tile8,~Context,~value,context,reads_holding(reads,source,step,start)):
  +h1 = read_holding(reads,source,step,0,start)
  +h2 = read_holding(reads,source,step,1,h1)
  +h3 = read_holding(reads,source,step,2,h2)
  +h4 = read_holding(reads,source,step,3,h3)
  +h5 = read_holding(reads,source,step,4,h4)
  +h6 = read_holding(reads,source,step,5,h5)
  +h7 = read_holding(reads,source,step,6,h6)
  +v1 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(0,reads),(source + 0 * step : U32),start,values)
  +v2 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(1,reads),(source + 1 * step : U32),h1,v1)
  +v3 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(2,reads),(source + 2 * step : U32),h2,v2)
  +v4 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(3,reads),(source + 3 * step : U32),h3,v3)
  +v5 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(4,reads),(source + 4 * step : U32),h4,v4)
  +v6 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(5,reads),(source + 5 * step : U32),h5,v5)
  +v7 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(6,reads),(source + 6 * step : U32),h6,v6)
  picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(7,reads),(source + 7 * step : U32),h7,v7)

# On pieces in order that hold the cells a block reads with the values of a
# reader, the reads give the cells of the block.
def reads_cells(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,start),low,high),+held: ReadsHeld<Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,start)),reads,source,step>,
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,start)) ->
  {reads_eight(reads,source,step,start) == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}:
  ReadsHeld{+held0,+held1,+held2,+held3,+held4,+held5,+held6,+held7} = held
  +h1 = read_holding(reads,source,step,0,start)
  +h2 = read_holding(reads,source,step,1,h1)
  +h3 = read_holding(reads,source,step,2,h2)
  +h4 = read_holding(reads,source,step,3,h3)
  +h5 = read_holding(reads,source,step,4,h4)
  +h6 = read_holding(reads,source,step,5,h5)
  +h7 = read_holding(reads,source,step,6,h6)
  +o1 = step_order(reads,source,step,0,start,start,{==})
  +o2 = step_order(reads,source,step,1,h1,start,o1)
  +o3 = step_order(reads,source,step,2,h2,start,o2)
  +o4 = step_order(reads,source,step,3,h3,start,o3)
  +o5 = step_order(reads,source,step,4,h4,start,o4)
  +o6 = step_order(reads,source,step,5,h5,start,o5)
  +o7 = step_order(reads,source,step,6,h6,start,o6)
  +v1 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(0,reads),(source + 0 * step : U32),start,values)
  +v2 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(1,reads),(source + 1 * step : U32),h1,v1)
  +v3 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(2,reads),(source + 2 * step : U32),h2,v2)
  +v4 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(3,reads),(source + 3 * step : U32),h3,v3)
  +v5 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(4,reads),(source + 4 * step : U32),h4,v4)
  +v6 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(5,reads),(source + 5 * step : U32),h5,v5)
  +v7 = picked_values(~Tile.Tile8,~Context,~value,context,U32.is_lt(6,reads),(source + 6 * step : U32),h6,v6)
  +e0 = step_cell(~Context,~value,context,reads,source,step,0,start,start,{==},low,high,capacity,capacity_ok,bounded,sorted,held0,values)
  +e1 = step_cell(~Context,~value,context,reads,source,step,1,h1,start,o1,low,high,capacity,capacity_ok,bounded,sorted,held1,v1)
  +e2 = step_cell(~Context,~value,context,reads,source,step,2,h2,start,o2,low,high,capacity,capacity_ok,bounded,sorted,held2,v2)
  +e3 = step_cell(~Context,~value,context,reads,source,step,3,h3,start,o3,low,high,capacity,capacity_ok,bounded,sorted,held3,v3)
  +e4 = step_cell(~Context,~value,context,reads,source,step,4,h4,start,o4,low,high,capacity,capacity_ok,bounded,sorted,held4,v4)
  +e5 = step_cell(~Context,~value,context,reads,source,step,5,h5,start,o5,low,high,capacity,capacity_ok,bounded,sorted,held5,v5)
  +e6 = step_cell(~Context,~value,context,reads,source,step,6,h6,start,o6,low,high,capacity,capacity_ok,bounded,sorted,held6,v6)
  +e7 = step_cell(~Context,~value,context,reads,source,step,7,h7,start,o7,low,high,capacity,capacity_ok,bounded,sorted,held7,v7)
  +b0 = block_cell(~Context,~value,context,reads,source,step,0)
  +b1 = block_cell(~Context,~value,context,reads,source,step,1)
  +b2 = block_cell(~Context,~value,context,reads,source,step,2)
  +b3 = block_cell(~Context,~value,context,reads,source,step,3)
  +b4 = block_cell(~Context,~value,context,reads,source,step,4)
  +b5 = block_cell(~Context,~value,context,reads,source,step,5)
  +b6 = block_cell(~Context,~value,context,reads,source,step,6)
  +r1 = read_cell(reads,source,step,1,h1)
  +r2 = read_cell(reads,source,step,2,h2)
  +r3 = read_cell(reads,source,step,3,h3)
  +r4 = read_cell(reads,source,step,4,h4)
  +r5 = read_cell(reads,source,step,5,h5)
  +r6 = read_cell(reads,source,step,6,h6)
  +r7 = read_cell(reads,source,step,7,h7)
  %Equal.sym(Tile.Tile8,read_cell(reads,source,step,0,start),b0,e0) : {Tiled.Eight{_,r1,r2,r3,r4,r5,r6,r7} == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}
  %Equal.sym(Tile.Tile8,r1,b1,e1) : {Tiled.Eight{b0,_,r2,r3,r4,r5,r6,r7} == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}
  %Equal.sym(Tile.Tile8,r2,b2,e2) : {Tiled.Eight{b0,b1,_,r3,r4,r5,r6,r7} == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}
  %Equal.sym(Tile.Tile8,r3,b3,e3) : {Tiled.Eight{b0,b1,b2,_,r4,r5,r6,r7} == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}
  %Equal.sym(Tile.Tile8,r4,b4,e4) : {Tiled.Eight{b0,b1,b2,b3,_,r5,r6,r7} == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}
  %Equal.sym(Tile.Tile8,r5,b5,e5) : {Tiled.Eight{b0,b1,b2,b3,b4,_,r6,r7} == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}
  %Equal.sym(Tile.Tile8,r6,b6,e6) : {Tiled.Eight{b0,b1,b2,b3,b4,b5,_,r7} == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}
  %Equal.sym(Tile.Tile8,r7,block_cell(~Context,~value,context,reads,source,step,7),e7) :
    {Tiled.Eight{b0,b1,b2,b3,b4,b5,b6,_} == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}
  {==}

# ---- What the reads of a block keep ----

def piece_order(inside: Bool,+reads: U32,+source: U32,+step: U32,+behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,
  +below: Held.Stack<Tile.Tile8>) ->
  {Held.order(~Tile.Tile8,Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,piece_model(inside,reads,source,step,behind,buffer,first,count,below))) ==
    Held.order(~Tile.Tile8,Held.Holding{behind,Held.Held{buffer,first,count,below}}) : Held.Stack<Tile.Tile8>}:
  match inside:
    case True{}:
      Equal.cong(Held.Holding<Tile.Tile8>,Held.Stack<Tile.Tile8>,holding => Held.order(~Tile.Tile8,holding),
        Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,piece_returned(behind,first,count,below,piece_eight(reads,(source - first : U32),step,buffer))),
        Held.Holding{behind,Held.Held{buffer,first,count,below}},piece_holding(reads,(source - first : U32),step,behind,buffer,first,count,below))
    case False{}: reads_order(reads,source,step,Held.Holding{behind,Held.Held{buffer,first,count,below}})

def current_order(+reads: U32,+source: U32,+step: U32,+holding: Held.Holding<Tile.Tile8>) ->
  {Held.order(~Tile.Tile8,Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,current_model(reads,source,step,holding))) == Held.order(~Tile.Tile8,holding) : Held.Stack<Tile.Tile8>}:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+count,+below}}:
      piece_order(U32.is_le(reads,Move.inside(reads,source,step,False{},first,count)),reads,source,step,behind,buffer,first,count,below)
    case Held.Holding{+behind,Held.Bottom{}}: reads_order(reads,source,step,Held.Holding{behind,Held.Bottom{}})

# The reads of a block keep the pieces in their order.
def block_order(+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>) ->
  {Held.order(~Tile.Tile8,Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,block_model(reads,source,step,start))) == Held.order(~Tile.Tile8,start) : Held.Stack<Tile.Tile8>}:
  +resolved = Held.resolve(~Tile.Tile8,start,source)
  Equal.trans(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,current_model(reads,source,step,resolved))),
    Held.order(~Tile.Tile8,resolved),Held.order(~Tile.Tile8,start),current_order(reads,source,step,resolved),Held.resolve_order(~Tile.Tile8,start,source))

def piece_values(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,inside: Bool,+reads: U32,+source: U32,+step: U32,+behind: Held.Stack<Tile.Tile8>,
  +buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,+below: Held.Stack<Tile.Tile8>,
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,Held.Holding{behind,Held.Held{buffer,first,count,below}})) ->
  Held.HoldingValues(~Tile.Tile8,~Context,~value,context,Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,piece_model(inside,reads,source,step,behind,buffer,first,count,below))):
  match inside:
    case True{}:
      {%Equal.sym(Held.Holding<Tile.Tile8>,Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,piece_returned(behind,first,count,below,piece_eight(reads,(source - first : U32),step,buffer))),
          Held.Holding{behind,Held.Held{buffer,first,count,below}},piece_holding(reads,(source - first : U32),step,behind,buffer,first,count,below)) :
        Held.HoldingValues(~Tile.Tile8,~Context,~value,context,_)
        values :
        Held.HoldingValues(~Tile.Tile8,~Context,~value,context,Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,
          piece_returned(behind,first,count,below,piece_eight(reads,(source - first : U32),step,buffer))))}
    case False{}: reads_values(~Context,~value,context,reads,source,step,Held.Holding{behind,Held.Held{buffer,first,count,below}},values)

def current_values(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+reads: U32,+source: U32,+step: U32,+holding: Held.Holding<Tile.Tile8>,
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,holding)) ->
  Held.HoldingValues(~Tile.Tile8,~Context,~value,context,Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,current_model(reads,source,step,holding))):
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+count,+below}}:
      piece_values(~Context,~value,context,U32.is_le(reads,Move.inside(reads,source,step,False{},first,count)),reads,source,step,behind,buffer,first,count,below,values)
    case Held.Holding{+behind,Held.Bottom{}}: reads_values(~Context,~value,context,reads,source,step,Held.Holding{behind,Held.Bottom{}},values)

# The reads of a block keep the values of the pieces.
def block_values(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>,
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,start)) ->
  Held.HoldingValues(~Tile.Tile8,~Context,~value,context,Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,block_model(reads,source,step,start))):
  current_values(~Context,~value,context,reads,source,step,Held.resolve(~Tile.Tile8,start,source),Held.resolve_values(~Tile.Tile8,~Context,~value,context,start,source,values))

# ---- The cells the reads of a block give ----

def below_le(+a: Nat,+b: Nat,+less: {Nat.is_lt(a,b) == True{} : Bool}) -> {Nat.is_le(a,b) == True{} : Bool}:
  Order.transitive(a,1n+a,b,Order.successor(a,a,Order.reflexive(a)),Order.strict_follow(a,b,less))

# A cell of a run that one piece holds whole lies in the piece.
def run_holds(+reads: U32,+source: U32,+step: U32,+index: U32,+first: U32,+count: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +current: Held.Holds<first,count,source,capacity>,+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},
  +step_room: {Nat.is_le(Nat.double(U32.to_nat(step)),U32.to_nat(word)) == True{} : Bool},
  +fits: {U32.is_le(reads,Move.inside(reads,source,step,False{},first,count)) == True{} : Bool},+less: {U32.is_lt(index,reads) == True{} : Bool}) ->
  Held.Holds<first,count,(source + index * step : U32),capacity>:
  Held.Holds{+after,+before,+room} = current
  +lane = U32.to_nat(index)
  +start = U32.to_nat(source)
  +distance = U32.to_nat(step)
  +offset = Nat.mul(lane,distance)
  +end = Nat.add(U32.to_nat(first),U32.to_nat(count))
  +lane_less = Bounds.as_nat_lt(index,reads,less)
  +positive = Order.le_lt_trans(0n,lane,U32.to_nat(reads),Order.zero_le(lane),lane_less)
  +reach = MovePart.fit_reach(reads,step,False{},0,source,first,count,capacity,capacity_ok,room,word,word_ok,step_room,positive,fits)
  +offset_le = RowMajor.mul_le(lane,Nat.sub(U32.to_nat(reads),1n),distance,Order.before_last(lane,U32.to_nat(reads),lane_less))
  +address_below = Order.le_lt_trans(Nat.add(start,offset),Nat.add(start,Nat.mul(Nat.sub(U32.to_nat(reads),1n),distance)),end,
    Order.offset_le(start,offset,Nat.mul(Nat.sub(U32.to_nat(reads),1n),distance),offset_le),reach)
  +address_room = Order.transitive(Nat.add(start,offset),end,U32.to_nat(capacity),below_le(Nat.add(start,offset),end,address_below),room)
  +offset_exact = Arithmetic.product_exact(capacity,capacity_ok,index,step,lane,distance,{==},{==},
    Order.transitive(offset,Nat.add(start,offset),U32.to_nat(capacity),Order.right_le(start,offset),address_room))
  +address_exact = Arithmetic.sum_exact(capacity,capacity_ok,source,(index * step : U32),start,offset,{==},offset_exact,address_room)
  +end_exact = Arithmetic.sum_exact(capacity,capacity_ok,first,count,U32.to_nat(first),U32.to_nat(count),{==},{==},room)
  +first_le = Order.transitive(U32.to_nat(first),start,Nat.add(start,offset),Bounds.as_nat_le(first,source,after),Order.le_add(start,offset))
  Held.Holds{
    Held.u32_le(first,(source + index * step : U32),
      {%Equal.sym(Nat,U32.to_nat((source + index * step : U32)),Nat.add(start,offset),address_exact) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool}
        first_le : {Nat.is_le(U32.to_nat(first),U32.to_nat((source + index * step : U32))) == True{} : Bool}}),
    Held.u32_lt((source + index * step : U32),(first + count : U32),
      {%Equal.sym(Nat,U32.to_nat((source + index * step : U32)),Nat.add(start,offset),address_exact) : {Nat.is_lt(_,U32.to_nat((first + count : U32))) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat((first + count : U32)),end,end_exact) : {Nat.is_lt(Nat.add(start,offset),_) == True{} : Bool}
        address_below : {Nat.is_lt(U32.to_nat((source + index * step : U32)),U32.to_nat((first + count : U32))) == True{} : Bool}}),
    room}

# A cell past the first of a piece, as the piece offsets it.
def shifted_offset(+source: U32,+first: U32,+offset: U32) -> {((source - first : U32) + offset : U32) == ((source + offset : U32) - first : U32) : U32}:
  Equal.trans(U32,((source - first : U32) + offset : U32),(offset + (source - first : U32) : U32),((source + offset : U32) - first : U32),
    Arithmetic.add_commutative((source - first : U32),offset),
    Equal.trans(U32,(offset + (source - first : U32) : U32),((offset + source : U32) - first : U32),((source + offset : U32) - first : U32),
      Equal.sym(U32,((offset + source : U32) - first : U32),(offset + (source - first : U32) : U32),Arithmetic.subtract_from_sum(offset,source,first)),
      Equal.cong(U32,U32,sum => (sum - first : U32),(offset + source : U32),(source + offset : U32),Arithmetic.add_commutative(offset,source))))

# A cell the current piece holds is its value in the buffer of the piece.
def piece_read(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,
  +below: Held.Stack<Tile.Tile8>,+address: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+holds: Held.Holds<first,count,address,capacity>,
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,Held.Holding{behind,Held.Held{buffer,first,count,below}})) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,buffer),(address - first : U32)) == value(context,address) : Tile.Tile8}:
  Held.Holds{+after,+before,+room} = holds
  {%before :
      {Pair.snd(Held.Holding<Tile.Tile8>,Tile.Tile8,Held.read_held(~Tile.Tile8,True{} && _,address,zero(),behind,buffer,first,count,below)) == value(context,address) : Tile.Tile8}
    %after :
      {Pair.snd(Held.Holding<Tile.Tile8>,Tile.Tile8,Held.read_held(~Tile.Tile8,_ && U32.is_lt(address,(first + count : U32)),address,zero(),behind,buffer,first,count,below)) == value(context,address) : Tile.Tile8}
    Held.read_current_value(~Tile.Tile8,~Context,~value,context,Held.Holding{behind,Held.Held{buffer,first,count,below}},address,zero(),capacity,capacity_ok,
      Held.Holds{after,before,room},values) :
    {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,buffer),(address - first : U32)) == value(context,address) : Tile.Tile8}}

# A lane of a block that one piece holds whole is the cell of the block.
def piece_lane(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,wanted: Bool,+reads: U32,+source: U32,+step: U32,+index: U32,
  +behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,+below: Held.Stack<Tile.Tile8>,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,Held.Holding{behind,Held.Held{buffer,first,count,below}}),
  +current: Logic.When(U32.is_lt(0,reads),Held.Holds<first,count,source,capacity>),+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},
  +step_room: {Nat.is_le(Nat.double(U32.to_nat(step)),U32.to_nat(word)) == True{} : Bool},
  +fits: {U32.is_le(reads,Move.inside(reads,source,step,False{},first,count)) == True{} : Bool},+wanted_exact: {U32.is_lt(index,reads) == wanted : Bool}) ->
  {next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,(source - first : U32),step,index,buffer) == block_cell(~Context,~value,context,reads,source,step,index) : Tile.Tile8}:
  match wanted:
    case False{}:
      %Equal.sym(Bool,U32.is_lt(index,reads),False{},wanted_exact) :
        {Bool.pick(Tile.Tile8,_,piece_value(buffer,((source - first : U32) + index * step : U32)),zero()) == Bool.pick(Tile.Tile8,_,value(context,(source + index * step : U32)),zero()) :
          Tile.Tile8}
      {==}
    case True{}:
      +lane_less = Bounds.as_nat_lt(index,reads,wanted_exact)
      +some = Held.u32_lt(0,reads,Order.le_lt_trans(0n,U32.to_nat(index),U32.to_nat(reads),Order.zero_le(U32.to_nat(index)),lane_less))
      +holds = run_holds(reads,source,step,index,first,count,capacity,capacity_ok,{%some : Logic.When(_,Held.Holds<first,count,source,capacity>) current : Held.Holds<first,count,source,capacity>},
        word,word_ok,step_room,fits,wanted_exact)
      %Equal.sym(Bool,U32.is_lt(index,reads),True{},wanted_exact) :
        {Bool.pick(Tile.Tile8,_,piece_value(buffer,((source - first : U32) + index * step : U32)),zero()) == Bool.pick(Tile.Tile8,_,value(context,(source + index * step : U32)),zero()) :
          Tile.Tile8}
      %Equal.sym(U32,((source - first : U32) + index * step : U32),((source + index * step : U32) - first : U32),shifted_offset(source,first,(index * step : U32))) :
        {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,buffer),_) == value(context,(source + index * step : U32)) : Tile.Tile8}
      piece_read(~Context,~value,context,behind,buffer,first,count,below,(source + index * step : U32),capacity,capacity_ok,holds,values)

# The eight cells of reads of one piece, read from given models.
def cells_at(+reads: U32,+offset: U32,+step: U32,+m0: Buffer.Model<Tile.Tile8>,+m1: Buffer.Model<Tile.Tile8>,+m2: Buffer.Model<Tile.Tile8>,+m3: Buffer.Model<Tile.Tile8>,
  +m4: Buffer.Model<Tile.Tile8>,+m5: Buffer.Model<Tile.Tile8>,+m6: Buffer.Model<Tile.Tile8>,+m7: Buffer.Model<Tile.Tile8>) -> Tiled.Eight:
  Tiled.Eight{next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,0,m0),next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,1,m1),
    next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,2,m2),next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,3,m3),
    next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,4,m4),next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,5,m5),
    next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,6,m6),next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,7,m7)}

# The cells of the reads of one piece all come from its buffer.
def piece_cells_kept(+reads: U32,+offset: U32,+step: U32,+buffer: Buffer.Model<Tile.Tile8>) ->
  {Pair.snd(Buffer.Model<Tile.Tile8>,Tiled.Eight,piece_eight(reads,offset,step,buffer)) == cells_at(reads,offset,step,buffer,buffer,buffer,buffer,buffer,buffer,buffer,buffer) : Tiled.Eight}:
  +m1 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,0,buffer)
  +m2 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,1,m1)
  +m3 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,2,m2)
  +m4 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,3,m3)
  +m5 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,4,m4)
  +m6 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,5,m5)
  +m7 = next_model(~Buffer.Model<Tile.Tile8>,~piece_kept,reads,offset,step,6,m6)
  +e1 = kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(0,reads),buffer)
  +e2 = Equal.trans(Buffer.Model<Tile.Tile8>,m2,m1,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(1,reads),m1),e1)
  +e3 = Equal.trans(Buffer.Model<Tile.Tile8>,m3,m2,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(2,reads),m2),e2)
  +e4 = Equal.trans(Buffer.Model<Tile.Tile8>,m4,m3,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(3,reads),m3),e3)
  +e5 = Equal.trans(Buffer.Model<Tile.Tile8>,m5,m4,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(4,reads),m4),e4)
  +e6 = Equal.trans(Buffer.Model<Tile.Tile8>,m6,m5,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(5,reads),m5),e5)
  +e7 = Equal.trans(Buffer.Model<Tile.Tile8>,m7,m6,buffer,kept_pick(~Buffer.Model<Tile.Tile8>,U32.is_lt(6,reads),m6),e6)
  +all = cells_at(reads,offset,step,buffer,buffer,buffer,buffer,buffer,buffer,buffer,buffer)
  %Equal.sym(Buffer.Model<Tile.Tile8>,m1,buffer,e1) : {cells_at(reads,offset,step,buffer,_,m2,m3,m4,m5,m6,m7) == all : Tiled.Eight}
  %Equal.sym(Buffer.Model<Tile.Tile8>,m2,buffer,e2) : {cells_at(reads,offset,step,buffer,buffer,_,m3,m4,m5,m6,m7) == all : Tiled.Eight}
  %Equal.sym(Buffer.Model<Tile.Tile8>,m3,buffer,e3) : {cells_at(reads,offset,step,buffer,buffer,buffer,_,m4,m5,m6,m7) == all : Tiled.Eight}
  %Equal.sym(Buffer.Model<Tile.Tile8>,m4,buffer,e4) : {cells_at(reads,offset,step,buffer,buffer,buffer,buffer,_,m5,m6,m7) == all : Tiled.Eight}
  %Equal.sym(Buffer.Model<Tile.Tile8>,m5,buffer,e5) : {cells_at(reads,offset,step,buffer,buffer,buffer,buffer,buffer,_,m6,m7) == all : Tiled.Eight}
  %Equal.sym(Buffer.Model<Tile.Tile8>,m6,buffer,e6) : {cells_at(reads,offset,step,buffer,buffer,buffer,buffer,buffer,buffer,_,m7) == all : Tiled.Eight}
  %Equal.sym(Buffer.Model<Tile.Tile8>,m7,buffer,e7) : {cells_at(reads,offset,step,buffer,buffer,buffer,buffer,buffer,buffer,buffer,_) == all : Tiled.Eight}
  {==}

def piece_cells(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,inside: Bool,+reads: U32,+source: U32,+step: U32,+behind: Held.Stack<Tile.Tile8>,
  +buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,+below: Held.Stack<Tile.Tile8>,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,Held.Holding{behind,Held.Held{buffer,first,count,below}}),low,high),
  +held: ReadsHeld<Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,Held.Holding{behind,Held.Held{buffer,first,count,below}})),reads,source,step>,
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,Held.Holding{behind,Held.Held{buffer,first,count,below}}),
  +current: Logic.When(U32.is_lt(0,reads),Held.Holds<first,count,source,capacity>),+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},
  +step_room: {Nat.is_le(Nat.double(U32.to_nat(step)),U32.to_nat(word)) == True{} : Bool},
  +inside_exact: {U32.is_le(reads,Move.inside(reads,source,step,False{},first,count)) == inside : Bool}) ->
  {Pair.snd(Held.Holding<Tile.Tile8>,Tiled.Eight,piece_model(inside,reads,source,step,behind,buffer,first,count,below)) == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}:
  match inside:
    case True{}:
      +offset = (source - first : U32)
      +b0 = block_cell(~Context,~value,context,reads,source,step,0)
      +b1 = block_cell(~Context,~value,context,reads,source,step,1)
      +b2 = block_cell(~Context,~value,context,reads,source,step,2)
      +b3 = block_cell(~Context,~value,context,reads,source,step,3)
      +b4 = block_cell(~Context,~value,context,reads,source,step,4)
      +b5 = block_cell(~Context,~value,context,reads,source,step,5)
      +b6 = block_cell(~Context,~value,context,reads,source,step,6)
      +n1 = next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,1,buffer)
      +n2 = next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,2,buffer)
      +n3 = next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,3,buffer)
      +n4 = next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,4,buffer)
      +n5 = next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,5,buffer)
      +n6 = next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,6,buffer)
      +n7 = next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,7,buffer)
      +expected = expected_eight(~Context,~value,context,reads,source,step)
      %Equal.sym(Tiled.Eight,Pair.snd(Buffer.Model<Tile.Tile8>,Tiled.Eight,piece_eight(reads,offset,step,buffer)),cells_at(reads,offset,step,buffer,buffer,buffer,buffer,buffer,buffer,buffer,buffer),
          piece_cells_kept(reads,offset,step,buffer)) :
        {_ == expected : Tiled.Eight}
      %Equal.sym(Tile.Tile8,next_cell(~Buffer.Model<Tile.Tile8>,~piece_value,reads,offset,step,0,buffer),b0,
          piece_lane(~Context,~value,context,U32.is_lt(0,reads),reads,source,step,0,behind,buffer,first,count,below,capacity,capacity_ok,values,current,word,word_ok,step_room,inside_exact,{==})) :
        {Tiled.Eight{_,n1,n2,n3,n4,n5,n6,n7} == expected : Tiled.Eight}
      %Equal.sym(Tile.Tile8,n1,b1,
          piece_lane(~Context,~value,context,U32.is_lt(1,reads),reads,source,step,1,behind,buffer,first,count,below,capacity,capacity_ok,values,current,word,word_ok,step_room,inside_exact,{==})) :
        {Tiled.Eight{b0,_,n2,n3,n4,n5,n6,n7} == expected : Tiled.Eight}
      %Equal.sym(Tile.Tile8,n2,b2,
          piece_lane(~Context,~value,context,U32.is_lt(2,reads),reads,source,step,2,behind,buffer,first,count,below,capacity,capacity_ok,values,current,word,word_ok,step_room,inside_exact,{==})) :
        {Tiled.Eight{b0,b1,_,n3,n4,n5,n6,n7} == expected : Tiled.Eight}
      %Equal.sym(Tile.Tile8,n3,b3,
          piece_lane(~Context,~value,context,U32.is_lt(3,reads),reads,source,step,3,behind,buffer,first,count,below,capacity,capacity_ok,values,current,word,word_ok,step_room,inside_exact,{==})) :
        {Tiled.Eight{b0,b1,b2,_,n4,n5,n6,n7} == expected : Tiled.Eight}
      %Equal.sym(Tile.Tile8,n4,b4,
          piece_lane(~Context,~value,context,U32.is_lt(4,reads),reads,source,step,4,behind,buffer,first,count,below,capacity,capacity_ok,values,current,word,word_ok,step_room,inside_exact,{==})) :
        {Tiled.Eight{b0,b1,b2,b3,_,n5,n6,n7} == expected : Tiled.Eight}
      %Equal.sym(Tile.Tile8,n5,b5,
          piece_lane(~Context,~value,context,U32.is_lt(5,reads),reads,source,step,5,behind,buffer,first,count,below,capacity,capacity_ok,values,current,word,word_ok,step_room,inside_exact,{==})) :
        {Tiled.Eight{b0,b1,b2,b3,b4,_,n6,n7} == expected : Tiled.Eight}
      %Equal.sym(Tile.Tile8,n6,b6,
          piece_lane(~Context,~value,context,U32.is_lt(6,reads),reads,source,step,6,behind,buffer,first,count,below,capacity,capacity_ok,values,current,word,word_ok,step_room,inside_exact,{==})) :
        {Tiled.Eight{b0,b1,b2,b3,b4,b5,_,n7} == expected : Tiled.Eight}
      %Equal.sym(Tile.Tile8,n7,block_cell(~Context,~value,context,reads,source,step,7),
          piece_lane(~Context,~value,context,U32.is_lt(7,reads),reads,source,step,7,behind,buffer,first,count,below,capacity,capacity_ok,values,current,word,word_ok,step_room,inside_exact,{==})) :
        {Tiled.Eight{b0,b1,b2,b3,b4,b5,b6,_} == expected : Tiled.Eight}
      {==}
    case False{}:
      reads_cells(~Context,~value,context,reads,source,step,Held.Holding{behind,Held.Held{buffer,first,count,below}},low,high,capacity,capacity_ok,bounded,sorted,held,values)

def current_cells(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+reads: U32,+source: U32,+step: U32,+holding: Held.Holding<Tile.Tile8>,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,holding),low,high),+held: ReadsHeld<Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)),reads,source,step>,
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,holding),+current: Logic.When(U32.is_lt(0,reads),Held.Current(~Tile.Tile8,holding,source,capacity)),
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+step_room: {Nat.is_le(Nat.double(U32.to_nat(step)),U32.to_nat(word)) == True{} : Bool}) ->
  {Pair.snd(Held.Holding<Tile.Tile8>,Tiled.Eight,current_model(reads,source,step,holding)) == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+count,+below}}:
      piece_cells(~Context,~value,context,U32.is_le(reads,Move.inside(reads,source,step,False{},first,count)),reads,source,step,behind,buffer,first,count,below,
        low,high,capacity,capacity_ok,bounded,sorted,held,values,current,word,word_ok,step_room,{==})
    case Held.Holding{+behind,Held.Bottom{}}:
      reads_cells(~Context,~value,context,reads,source,step,Held.Holding{behind,Held.Bottom{}},low,high,capacity,capacity_ok,bounded,sorted,held,values)

# The piece of the first cell a block reads holds it when the block reads any.
def first_current(wanted: Bool,+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,start),low,high),+first: ReadAt(Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,start)),reads,source,step,0),
  +wanted_exact: {U32.is_lt(0,reads) == wanted : Bool}) ->
  Logic.When(U32.is_lt(0,reads),Held.Current(~Tile.Tile8,Held.resolve(~Tile.Tile8,start,source),source,capacity)):
  match wanted:
    case False{}:
      {%Equal.sym(Bool,U32.is_lt(0,reads),False{},wanted_exact) : Logic.When(_,Held.Current(~Tile.Tile8,Held.resolve(~Tile.Tile8,start,source),source,capacity))
        Unit{} : Logic.When(U32.is_lt(0,reads),Held.Current(~Tile.Tile8,Held.resolve(~Tile.Tile8,start,source),source,capacity))}
    case True{}:
      +source_held = {%Equal.trans(U32,(source + 0 * step : U32),(source + 0 : U32),source,
            Equal.cong(U32,U32,offset => (source + offset : U32),(0 * step : U32),0,Arithmetic.multiply_zero_left(step)),Arithmetic.add_zero(source)) :
          Held.SpanHeld(Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,start)),U32.to_nat(_))
        {%wanted_exact : Logic.When(_,Held.SpanHeld(Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,start)),U32.to_nat((source + 0 * step : U32)))) first :
          Logic.When(True{},Held.SpanHeld(Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,start)),U32.to_nat((source + 0 * step : U32))))} :
        Held.SpanHeld(Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,start)),U32.to_nat(source))}
      {%Equal.sym(Bool,U32.is_lt(0,reads),True{},wanted_exact) : Logic.When(_,Held.Current(~Tile.Tile8,Held.resolve(~Tile.Tile8,start,source),source,capacity))
        Held.resolve_current(~Tile.Tile8,start,source,low,high,capacity,capacity_ok,bounded,sorted,source_held) :
        Logic.When(U32.is_lt(0,reads),Held.Current(~Tile.Tile8,Held.resolve(~Tile.Tile8,start,source),source,capacity))}

# On pieces in order that hold the cells a block reads with the values of a
# reader, the reads of the block give the cells of the block.
def block_cells(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+reads: U32,+source: U32,+step: U32,+start: Held.Holding<Tile.Tile8>,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,start),low,high),+held: ReadsHeld<Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,start)),reads,source,step>,
  +values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,start),
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+step_room: {Nat.is_le(Nat.double(U32.to_nat(step)),U32.to_nat(word)) == True{} : Bool}) ->
  {Pair.snd(Held.Holding<Tile.Tile8>,Tiled.Eight,block_model(reads,source,step,start)) == expected_eight(~Context,~value,context,reads,source,step) : Tiled.Eight}:
  ReadsHeld{+held0,+held1,+held2,+held3,+held4,+held5,+held6,+held7} = held
  +resolved = Held.resolve(~Tile.Tile8,start,source)
  +back = Equal.sym(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,resolved),Held.order(~Tile.Tile8,start),Held.resolve_order(~Tile.Tile8,start,source))
  current_cells(~Context,~value,context,reads,source,step,resolved,low,high,capacity,capacity_ok,bounded,
    Held.sorted_moved(~Tile.Tile8,Held.order(~Tile.Tile8,start),Held.order(~Tile.Tile8,resolved),low,high,back,sorted),
    {%back : ReadsHeld<Held.spans(~Tile.Tile8,_),reads,source,step>
      ReadsHeld{held0,held1,held2,held3,held4,held5,held6,held7} : ReadsHeld<Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,resolved)),reads,source,step>},
    Held.resolve_values(~Tile.Tile8,~Context,~value,context,start,source,values),
    first_current(U32.is_lt(0,reads),reads,source,step,start,low,high,capacity,capacity_ok,bounded,sorted,held0,{==}),
    word,word_ok,step_room)

# ---- The writes of a block ----

# A write into packed contents writes into their tree.
def write_packed(~Element: Data,+index: U32,+value: Element,+model: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model))) ==
    Buffer.Content{Tree.write(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,Buffer.pack(~Element,model))),index,value),
      Buffer.content_length(~Element,Buffer.contents(~Element,Buffer.pack(~Element,model)))} : Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model))),
    Buffer.Content{Tree.write(Element,Buffer.content_tree(~Element,Buffer.content(~Element,model)),index,value),Buffer.content_length(~Element,Buffer.content(~Element,model))},
    Buffer.Content{Tree.write(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,Buffer.pack(~Element,model))),index,value),
      Buffer.content_length(~Element,Buffer.contents(~Element,Buffer.pack(~Element,model)))},
    Buffer.write_value_packed(~Element,index,value,model),
    Equal.cong(Buffer.Content<Element>,Buffer.Content<Element>,content => Buffer.Content{Tree.write(Element,Buffer.content_tree(~Element,content),index,value),Buffer.content_length(~Element,content)},
      Buffer.content(~Element,model),Buffer.contents(~Element,Buffer.pack(~Element,model)),
      Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model))))

# Cell index of eight cells.
def eighth(+index: U32,+eight: Tiled.Eight) -> Tile.Tile8:
  Tiled.Eight{+c0,+c1,+c2,+c3,+c4,+c5,+c6,+c7} = eight
  Bool.pick(Tile.Tile8,U32.is_lt(index,4),
    Bool.pick(Tile.Tile8,U32.is_lt(index,2),Bool.pick(Tile.Tile8,U32.is_zero(index),c0,c1),Bool.pick(Tile.Tile8,U32.is_eq(index,2),c2,c3)),
    Bool.pick(Tile.Tile8,U32.is_lt(index,6),Bool.pick(Tile.Tile8,U32.is_eq(index,4),c4,c5),Bool.pick(Tile.Tile8,U32.is_eq(index,6),c6,c7)))

# The tree the writes of a block leave.
def written_tree(fuel: Nat,more: Bool,+index: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  match fuel more:
    case 0n more: tree
    case 1n+rest False{}: tree
    case 1n+rest True{}:
      written_tree(rest,U32.is_lt((index + 1 : U32),high),(index + 1 : U32),high,(target + stride : U32),stride,eight,Tree.write(Tile.Tile8,tree,target,eighth(index,eight)))

# A write of a tree when it is wanted.
def tree_when(wanted: Bool,+index: U32,+value: Tile.Tile8,tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  match wanted:
    case True{}: Tree.write(Tile.Tile8,tree,index,value)
    case False{}: tree

# The tree the writes of a block leave, lane by lane as the block writes them.
def lanes_tree(+low: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Tiled.Eight{+c0,+c1,+c2,+c3,+c4,+c5,+c6,+c7} = eight
  +start = (target - low * stride : U32)
  tree_when(U32.is_le(low,7) && U32.is_lt(7,high),(start + 7 * stride : U32),c7,
    tree_when(U32.is_le(low,6) && U32.is_lt(6,high),(start + 6 * stride : U32),c6,
      tree_when(U32.is_le(low,5) && U32.is_lt(5,high),(start + 5 * stride : U32),c5,
        tree_when(U32.is_le(low,4) && U32.is_lt(4,high),(start + 4 * stride : U32),c4,
          tree_when(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,
            tree_when(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,
              tree_when(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,
                tree_when(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,tree))))))))

def when_contents(wanted: Bool,+index: U32,+value: Tile.Tile8,output: Storage.Buffer<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,Tiled.cell_written(wanted,index,value,output)) ==
    Buffer.Content{tree_when(wanted,index,value,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))),Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))} :
    Buffer.Content<Tile.Tile8>}:
  match wanted:
    case True{}:
      +observation = Buffer.observe(~Tile.Tile8,output)
      %Buffer.observed_equation(~Tile.Tile8,output,observation) :
        {Buffer.contents(~Tile.Tile8,Storage.write(~Tile.Tile8,index,value,_)) ==
          Buffer.Content{Tree.write(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,_)),index,value),Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,_))} :
          Buffer.Content<Tile.Tile8>}
      write_packed(~Tile.Tile8,index,value,Buffer.observed_model(~Tile.Tile8,output,observation))
    case False{}: Buffer.content_eta(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))

def when_after(wanted: Bool,+index: U32,+value: Tile.Tile8,-output: Storage.Buffer<Tile.Tile8>,-inner: Buffer.Content<Tile.Tile8>,
  +step: {Buffer.contents(~Tile.Tile8,Tiled.cell_written(wanted,index,value,output)) ==
    Buffer.Content{tree_when(wanted,index,value,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))),Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))} :
    Buffer.Content<Tile.Tile8>},
  +same: {Buffer.contents(~Tile.Tile8,output) == inner : Buffer.Content<Tile.Tile8>}) ->
  {Buffer.contents(~Tile.Tile8,Tiled.cell_written(wanted,index,value,output)) ==
    Buffer.Content{tree_when(wanted,index,value,Buffer.content_tree(~Tile.Tile8,inner)),Buffer.content_length(~Tile.Tile8,inner)} : Buffer.Content<Tile.Tile8>}:
  %same :
    {Buffer.contents(~Tile.Tile8,Tiled.cell_written(wanted,index,value,output)) ==
      Buffer.Content{tree_when(wanted,index,value,Buffer.content_tree(~Tile.Tile8,_)),Buffer.content_length(~Tile.Tile8,_)} : Buffer.Content<Tile.Tile8>}
  step

# The contents the writes of a block leave on a buffer model: its lanes on
# the tree of the model.
def lanes_packed(+low: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,+model: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,Tiled.block_writes(low,high,target,stride,eight,Buffer.pack(~Tile.Tile8,model))) ==
    Buffer.Content{lanes_tree(low,high,target,stride,eight,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))),
      Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))} : Buffer.Content<Tile.Tile8>}:
  match eight:
    case Tiled.Eight{+c0,+c1,+c2,+c3,+c4,+c5,+c6,+c7}:
      +start = (target - low * stride : U32)
      +tree = Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))
      +length = Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))
      +t0 = tree_when(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,tree)
      +t1 = tree_when(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,t0)
      +t2 = tree_when(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,t1)
      +t3 = tree_when(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,t2)
      +t4 = tree_when(U32.is_le(low,4) && U32.is_lt(4,high),(start + 4 * stride : U32),c4,t3)
      +t5 = tree_when(U32.is_le(low,5) && U32.is_lt(5,high),(start + 5 * stride : U32),c5,t4)
      +t6 = tree_when(U32.is_le(low,6) && U32.is_lt(6,high),(start + 6 * stride : U32),c6,t5)
      +e0 = when_after(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)),when_contents(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model)),{==})
      +e1 = when_after(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model)),Buffer.Content{t0,length},
        when_contents(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model))),e0)
      +e2 = when_after(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model))),Buffer.Content{t1,length},
        when_contents(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model)))),e1)
      +e3 = when_after(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,Tiled.cell_written(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model)))),Buffer.Content{t2,length},
        when_contents(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,Tiled.cell_written(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model))))),e2)
      +e4 = when_after(U32.is_le(low,4) && U32.is_lt(4,high),(start + 4 * stride : U32),c4,Tiled.cell_written(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,Tiled.cell_written(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model))))),Buffer.Content{t3,length},
        when_contents(U32.is_le(low,4) && U32.is_lt(4,high),(start + 4 * stride : U32),c4,Tiled.cell_written(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,Tiled.cell_written(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model)))))),e3)
      +e5 = when_after(U32.is_le(low,5) && U32.is_lt(5,high),(start + 5 * stride : U32),c5,Tiled.cell_written(U32.is_le(low,4) && U32.is_lt(4,high),(start + 4 * stride : U32),c4,Tiled.cell_written(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,Tiled.cell_written(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model)))))),Buffer.Content{t4,length},
        when_contents(U32.is_le(low,5) && U32.is_lt(5,high),(start + 5 * stride : U32),c5,Tiled.cell_written(U32.is_le(low,4) && U32.is_lt(4,high),(start + 4 * stride : U32),c4,Tiled.cell_written(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,Tiled.cell_written(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model))))))),e4)
      +e6 = when_after(U32.is_le(low,6) && U32.is_lt(6,high),(start + 6 * stride : U32),c6,Tiled.cell_written(U32.is_le(low,5) && U32.is_lt(5,high),(start + 5 * stride : U32),c5,Tiled.cell_written(U32.is_le(low,4) && U32.is_lt(4,high),(start + 4 * stride : U32),c4,Tiled.cell_written(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,Tiled.cell_written(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model))))))),Buffer.Content{t5,length},
        when_contents(U32.is_le(low,6) && U32.is_lt(6,high),(start + 6 * stride : U32),c6,Tiled.cell_written(U32.is_le(low,5) && U32.is_lt(5,high),(start + 5 * stride : U32),c5,Tiled.cell_written(U32.is_le(low,4) && U32.is_lt(4,high),(start + 4 * stride : U32),c4,Tiled.cell_written(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,Tiled.cell_written(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model)))))))),e5)
      when_after(U32.is_le(low,7) && U32.is_lt(7,high),(start + 7 * stride : U32),c7,Tiled.cell_written(U32.is_le(low,6) && U32.is_lt(6,high),(start + 6 * stride : U32),c6,Tiled.cell_written(U32.is_le(low,5) && U32.is_lt(5,high),(start + 5 * stride : U32),c5,Tiled.cell_written(U32.is_le(low,4) && U32.is_lt(4,high),(start + 4 * stride : U32),c4,Tiled.cell_written(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,Tiled.cell_written(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model)))))))),Buffer.Content{t6,length},
        when_contents(U32.is_le(low,7) && U32.is_lt(7,high),(start + 7 * stride : U32),c7,Tiled.cell_written(U32.is_le(low,6) && U32.is_lt(6,high),(start + 6 * stride : U32),c6,Tiled.cell_written(U32.is_le(low,5) && U32.is_lt(5,high),(start + 5 * stride : U32),c5,Tiled.cell_written(U32.is_le(low,4) && U32.is_lt(4,high),(start + 4 * stride : U32),c4,Tiled.cell_written(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,Tiled.cell_written(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,Tiled.cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,Tiled.cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,Buffer.pack(~Tile.Tile8,model))))))))),e6)

# The contents the writes of a block leave: its lanes on the tree of the
# output.
def lanes_contents(+low: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,output: Storage.Buffer<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,Tiled.block_writes(low,high,target,stride,eight,output)) ==
    Buffer.Content{lanes_tree(low,high,target,stride,eight,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))),Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))} :
    Buffer.Content<Tile.Tile8>}:
  +observation = Buffer.observe(~Tile.Tile8,output)
  %Buffer.observed_equation(~Tile.Tile8,output,observation) :
    {Buffer.contents(~Tile.Tile8,Tiled.block_writes(low,high,target,stride,eight,_)) ==
      Buffer.Content{lanes_tree(low,high,target,stride,eight,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,_))),Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,_))} :
      Buffer.Content<Tile.Tile8>}
  lanes_packed(low,high,target,stride,eight,Buffer.observed_model(~Tile.Tile8,output,observation))

# ---- The lanes of a block as a walk ----

# The lanes from lane on, each written when it lies from low to below high.
def lane_walk(fuel: Nat,+lane: U32,+low: U32,+high: U32,+start: U32,+stride: U32,+eight: Tiled.Eight,tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  match fuel:
    case 0n: tree
    case 1n+rest:
      lane_walk(rest,(lane + 1 : U32),low,high,start,stride,eight,
        tree_when(U32.is_le(low,lane) && U32.is_lt(lane,high),(start + lane * stride : U32),eighth(lane,eight),tree))

def nat_le_zero(+value: Nat) -> {Nat.is_le(value,0n) == Nat.is_eq(value,0n) : Bool}:
  match value:
    case 0n: {==}
    case 1n+rest: {==}

def le_zero_word(+word: U32) -> {U32.is_le(word,0) == U32.is_eq(word,0) : Bool}:
  Equal.trans(Bool,U32.is_le(word,0),Nat.is_le(U32.to_nat(word),0n),U32.is_eq(word,0),Comparison.le(word,0),
    Equal.trans(Bool,Nat.is_le(U32.to_nat(word),0n),Nat.is_eq(U32.to_nat(word),0n),U32.is_eq(word,0),nat_le_zero(U32.to_nat(word)),
      Equal.sym(Bool,U32.is_eq(word,0),Nat.is_eq(U32.to_nat(word),0n),Comparison.eq(word,0))))

def one_times(+value: U32) -> {(1 * value : U32) == value : U32}:
  Equal.trans(U32,(1 * value : U32),(value * 1 : U32),value,Arithmetic.multiply_commutative(1,value),Arithmetic.multiply_one(value))

def and_false(+flag: Bool) -> {(flag && False{}) == False{} : Bool}:
  match flag:
    case True{}: {==}
    case False{}: {==}

# The writes of a block are its lanes as a walk from lane 0.
def lanes_walk(+low: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,+tree: Tree.Tree<Tile.Tile8>) ->
  {lanes_tree(low,high,target,stride,eight,tree) == lane_walk(8n,0,low,high,(target - low * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>}:
  match eight:
    case Tiled.Eight{+c0,+c1,+c2,+c3,+c4,+c5,+c6,+c7}:
      +start = (target - low * stride : U32)
      +w1 = U32.is_le(low,1) && U32.is_lt(1,high)
      +runtime = tree_when(w1,(start + stride : U32),c1,tree_when(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,tree))
      +model = tree_when(w1,(start + 1 * stride : U32),c1,tree_when(U32.is_le(low,0) && U32.is_lt(0,high),(start + 0 * stride : U32),c0,tree))
      +inner = {
        %Equal.sym(U32,(1 * stride : U32),stride,one_times(stride)) :
          {runtime == tree_when(w1,(start + _ : U32),c1,tree_when(U32.is_le(low,0) && U32.is_lt(0,high),(start + 0 * stride : U32),c0,tree)) : Tree.Tree<Tile.Tile8>}
        %Equal.sym(U32,(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)) :
          {runtime == tree_when(w1,(start + stride : U32),c1,tree_when(U32.is_le(low,0) && U32.is_lt(0,high),(start + _ : U32),c0,tree)) : Tree.Tree<Tile.Tile8>}
        %Equal.sym(U32,(start + 0 : U32),start,Arithmetic.add_zero(start)) :
          {runtime == tree_when(w1,(start + stride : U32),c1,tree_when(U32.is_le(low,0) && U32.is_lt(0,high),_,c0,tree)) : Tree.Tree<Tile.Tile8>}
        %Equal.sym(Bool,U32.is_le(low,0),U32.is_eq(low,0),le_zero_word(low)) :
          {runtime == tree_when(w1,(start + stride : U32),c1,tree_when(_ && U32.is_lt(0,high),start,c0,tree)) : Tree.Tree<Tile.Tile8>}
        {==} : {runtime == model : Tree.Tree<Tile.Tile8>}}
      Equal.cong(Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,written => lane_walk(6n,2,low,high,start,stride,Tiled.Eight{c0,c1,c2,c3,c4,c5,c6,c7},written),runtime,model,inner)

# One lane on: the lanes left and the next lane.
def next_total(+at: Nat,+rest: Nat,+total: {Nat.add(at,1n+rest) == 8n : Nat}) -> {Nat.add(1n+at,rest) == 8n : Nat}:
  Equal.trans(Nat,Nat.add(1n+at,rest),Nat.add(at,1n+rest),8n,Equal.sym(Nat,Nat.add(at,1n+rest),1n+Nat.add(at,rest),Addition.successor_right(at,rest)),total)

def next_exact(+lane: U32,+at: Nat,+rest: Nat,+lane_exact: {U32.to_nat(lane) == at : Nat},+total: {Nat.add(at,1n+rest) == 8n : Nat}) ->
  {U32.to_nat((lane + 1 : U32)) == 1n+at : Nat}:
  +bound = {%next_total(at,rest,total) : {Nat.is_le(1n+at,_) == True{} : Bool} Order.le_add(1n+at,rest) : {Nat.is_le(1n+at,8n) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat((lane + 1 : U32)),Nat.add(at,1n),1n+at,
    Arithmetic.sum_exact(8,{==},lane,1,at,1n,lane_exact,{==},
      {%Equal.sym(Nat,Nat.add(at,1n),1n+at,Addition.successor_width(at)) : {Nat.is_le(_,8n) == True{} : Bool} bound : {Nat.is_le(Nat.add(at,1n),8n) == True{} : Bool}}),
    Addition.successor_width(at))

# Comparisons of a lane through its number.
def lane_nat_le(+word: U32,+lane: U32,+at: Nat,+lane_exact: {U32.to_nat(lane) == at : Nat},+order: {U32.is_le(word,lane) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(word),at) == True{} : Bool}:
  {%lane_exact : {Nat.is_le(U32.to_nat(word),_) == True{} : Bool} Bounds.as_nat_le(word,lane,order) : {Nat.is_le(U32.to_nat(word),at) == True{} : Bool}}

def lane_nat_past(+lane: U32,+high: U32,+at: Nat,+lane_exact: {U32.to_nat(lane) == at : Nat},+done: {U32.is_lt(lane,high) == False{} : Bool}) ->
  {Nat.is_le(U32.to_nat(high),at) == True{} : Bool}:
  +nat_done = Equal.trans(Bool,Nat.is_lt(U32.to_nat(lane),U32.to_nat(high)),U32.is_lt(lane,high),False{},Equal.sym(Bool,U32.is_lt(lane,high),Nat.is_lt(U32.to_nat(lane),U32.to_nat(high)),Comparison.lt(lane,high)),done)
  {%lane_exact : {Nat.is_le(U32.to_nat(high),_) == True{} : Bool} Order.not_less(U32.to_nat(lane),U32.to_nat(high),nat_done) : {Nat.is_le(U32.to_nat(high),at) == True{} : Bool}}

def lane_not_below(+lane: U32,+high: U32,+at: Nat,+lane_exact: {U32.to_nat(lane) == at : Nat},+past: {Nat.is_le(U32.to_nat(high),at) == True{} : Bool}) ->
  {U32.is_lt(lane,high) == False{} : Bool}:
  Equal.trans(Bool,U32.is_lt(lane,high),Nat.is_lt(U32.to_nat(lane),U32.to_nat(high)),False{},Comparison.lt(lane,high),
    {%Equal.sym(Nat,U32.to_nat(lane),at,lane_exact) : {Nat.is_lt(_,U32.to_nat(high)) == False{} : Bool}
      Order.not_below(U32.to_nat(high),at,past) : {Nat.is_lt(U32.to_nat(lane),U32.to_nat(high)) == False{} : Bool}})

def word_le_lane(+word: U32,+lane: U32,+at: Nat,+lane_exact: {U32.to_nat(lane) == at : Nat},+order: {Nat.is_le(U32.to_nat(word),at) == True{} : Bool}) ->
  {U32.is_le(word,lane) == True{} : Bool}:
  Held.u32_le(word,lane,{%Equal.sym(Nat,U32.to_nat(lane),at,lane_exact) : {Nat.is_le(U32.to_nat(word),_) == True{} : Bool} order : {Nat.is_le(U32.to_nat(word),U32.to_nat(lane)) == True{} : Bool}})

def lane_not_from(+low: U32,+lane: U32,+at: Nat,+lane_exact: {U32.to_nat(lane) == at : Nat},+less: {U32.is_lt(lane,low) == True{} : Bool}) ->
  {U32.is_le(low,lane) == False{} : Bool}:
  +nat_less = {%lane_exact : {Nat.is_lt(_,U32.to_nat(low)) == True{} : Bool} Bounds.as_nat_lt(lane,low,less) : {Nat.is_lt(at,U32.to_nat(low)) == True{} : Bool}}
  Equal.trans(Bool,U32.is_le(low,lane),Nat.is_le(U32.to_nat(low),U32.to_nat(lane)),False{},Comparison.le(low,lane),
    {%Equal.sym(Nat,U32.to_nat(lane),at,lane_exact) : {Nat.is_le(U32.to_nat(low),_) == False{} : Bool}
      Held.lt_not_le(at,U32.to_nat(low),nat_less) : {Nat.is_le(U32.to_nat(low),U32.to_nat(lane)) == False{} : Bool}})

# A walk past high leaves the tree.
def walk_idle(fuel: Nat,+lane: U32,+at: Nat,+low: U32,+high: U32,+start: U32,+stride: U32,+eight: Tiled.Eight,+tree: Tree.Tree<Tile.Tile8>,
  +lane_exact: {U32.to_nat(lane) == at : Nat},+total: {Nat.add(at,fuel) == 8n : Nat},+past: {Nat.is_le(U32.to_nat(high),at) == True{} : Bool}) ->
  {lane_walk(fuel,lane,low,high,start,stride,eight,tree) == tree : Tree.Tree<Tile.Tile8>}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Bool,U32.is_lt(lane,high),False{},lane_not_below(lane,high,at,lane_exact,past)) :
        {lane_walk(rest,(lane + 1 : U32),low,high,start,stride,eight,tree_when(U32.is_le(low,lane) && _,(start + lane * stride : U32),eighth(lane,eight),tree)) == tree : Tree.Tree<Tile.Tile8>}
      %Equal.sym(Bool,U32.is_le(low,lane) && False{},False{},and_false(U32.is_le(low,lane))) :
        {lane_walk(rest,(lane + 1 : U32),low,high,start,stride,eight,tree_when(_,(start + lane * stride : U32),eighth(lane,eight),tree)) == tree : Tree.Tree<Tile.Tile8>}
      walk_idle(rest,(lane + 1 : U32),1n+at,low,high,start,stride,eight,tree,next_exact(lane,at,rest,lane_exact,total),next_total(at,rest,total),
        Order.successor(U32.to_nat(high),at,past))

# A recursive model with no lanes left to write leaves the tree.
def written_none(budget: Nat,+index: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,+tree: Tree.Tree<Tile.Tile8>) ->
  {tree == written_tree(budget,False{},index,high,target,stride,eight,tree) : Tree.Tree<Tile.Tile8>}:
  match budget:
    case 0n: {==}
    case 1n+rest: {==}

def next_address(+start: U32,+lane: U32,+stride: U32) -> {(start + (lane + 1 : U32) * stride : U32) == ((start + lane * stride : U32) + stride : U32) : U32}:
  Equal.trans(U32,(start + (lane + 1 : U32) * stride : U32),(start + (lane * stride + stride : U32) : U32),((start + lane * stride : U32) + stride : U32),
    Equal.cong(U32,U32,offset => (start + offset : U32),((lane + 1 : U32) * stride : U32),(lane * stride + stride : U32),Arithmetic.multiply_successor(lane,stride)),
    Equal.sym(U32,((start + lane * stride : U32) + stride : U32),(start + (lane * stride + stride : U32) : U32),Arithmetic.add_associative(start,(lane * stride : U32),stride)))

# From a lane at or past low, the walk writes what the recursive model writes
# from that lane.
def walk_written(fuel: Nat,budget: Nat,more: Bool,+lane: U32,+at: Nat,+low: U32,+high: U32,+start: U32,+stride: U32,+eight: Tiled.Eight,+tree: Tree.Tree<Tile.Tile8>,
  +more_exact: {U32.is_lt(lane,high) == more : Bool},+lane_exact: {U32.to_nat(lane) == at : Nat},+total: {Nat.add(at,fuel) == 8n : Nat},+from: {U32.is_le(low,lane) == True{} : Bool},
  +enough: {Nat.is_le(fuel,budget) == True{} : Bool},+high_small: {Nat.is_le(U32.to_nat(high),8n) == True{} : Bool}) ->
  {lane_walk(fuel,lane,low,high,start,stride,eight,tree) == written_tree(budget,U32.is_lt(lane,high),lane,high,(start + lane * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>}:
  match fuel budget more:
    case 0n 0n more: {==}
    case 0n 1n+ +spare more:
      +eight_at = Equal.trans(Nat,at,Nat.add(at,0n),8n,Equal.sym(Nat,Nat.add(at,0n),at,Addition.zero_right(at)),total)
      +past = {%Equal.sym(Nat,at,8n,eight_at) : {Nat.is_le(U32.to_nat(high),_) == True{} : Bool} high_small : {Nat.is_le(U32.to_nat(high),at) == True{} : Bool}}
      %Equal.sym(Bool,U32.is_lt(lane,high),False{},lane_not_below(lane,high,at,lane_exact,past)) :
        {tree == written_tree(1n+spare,_,lane,high,(start + lane * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>}
      {==}
    case 1n+ +rest 0n more:
      Empty.absurd({lane_walk(1n+rest,lane,low,high,start,stride,eight,tree) == written_tree(0n,U32.is_lt(lane,high),lane,high,(start + lane * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>},
        WordEncoding.false_true(enough))
    case 1n+ +rest 1n+ +spare True{}:
      +written = Tree.write(Tile.Tile8,tree,(start + lane * stride : U32),eighth(lane,eight))
      +next_exact_here = next_exact(lane,at,rest,lane_exact,total)
      +next_from = word_le_lane(low,(lane + 1 : U32),1n+at,next_exact_here,Order.successor(U32.to_nat(low),at,lane_nat_le(low,lane,at,lane_exact,from)))
      %Equal.sym(Bool,U32.is_lt(lane,high),True{},more_exact) :
        {lane_walk(rest,(lane + 1 : U32),low,high,start,stride,eight,tree_when(U32.is_le(low,lane) && _,(start + lane * stride : U32),eighth(lane,eight),tree)) ==
          written_tree(1n+spare,_,lane,high,(start + lane * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>}
      %Equal.sym(Bool,U32.is_le(low,lane),True{},from) :
        {lane_walk(rest,(lane + 1 : U32),low,high,start,stride,eight,tree_when(_ && True{},(start + lane * stride : U32),eighth(lane,eight),tree)) ==
          written_tree(1n+spare,True{},lane,high,(start + lane * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>}
      %next_address(start,lane,stride) :
        {lane_walk(rest,(lane + 1 : U32),low,high,start,stride,eight,written) == written_tree(spare,U32.is_lt((lane + 1 : U32),high),(lane + 1 : U32),high,_,stride,eight,written) :
          Tree.Tree<Tile.Tile8>}
      walk_written(rest,spare,U32.is_lt((lane + 1 : U32),high),(lane + 1 : U32),1n+at,low,high,start,stride,eight,written,{==},next_exact_here,next_total(at,rest,total),next_from,
        enough,high_small)
    case 1n+ +rest 1n+ +spare False{}:
      %Equal.sym(Bool,U32.is_lt(lane,high),False{},more_exact) :
        {lane_walk(rest,(lane + 1 : U32),low,high,start,stride,eight,tree_when(U32.is_le(low,lane) && _,(start + lane * stride : U32),eighth(lane,eight),tree)) ==
          written_tree(1n+spare,_,lane,high,(start + lane * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>}
      %Equal.sym(Bool,U32.is_le(low,lane) && False{},False{},and_false(U32.is_le(low,lane))) :
        {lane_walk(rest,(lane + 1 : U32),low,high,start,stride,eight,tree_when(_,(start + lane * stride : U32),eighth(lane,eight),tree)) ==
          written_tree(1n+spare,False{},lane,high,(start + lane * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>}
      walk_idle(rest,(lane + 1 : U32),1n+at,low,high,start,stride,eight,tree,next_exact(lane,at,rest,lane_exact,total),next_total(at,rest,total),
        Order.successor(U32.to_nat(high),at,lane_nat_past(lane,high,at,lane_exact,more_exact)))

# From a lane at or below low, the walk writes what the recursive model
# writes from low.
def skip_walk(fuel: Nat,before: Bool,+lane: U32,+at: Nat,+low: U32,+high: U32,+start: U32,+stride: U32,+eight: Tiled.Eight,+tree: Tree.Tree<Tile.Tile8>,
  +before_exact: {U32.is_lt(lane,low) == before : Bool},+lane_exact: {U32.to_nat(lane) == at : Nat},+total: {Nat.add(at,fuel) == 8n : Nat},
  +below: {Nat.is_le(at,U32.to_nat(low)) == True{} : Bool},+high_small: {Nat.is_le(U32.to_nat(high),8n) == True{} : Bool}) ->
  {lane_walk(fuel,lane,low,high,start,stride,eight,tree) == written_tree(8n,U32.is_lt(low,high),low,high,(start + low * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>}:
  match fuel before:
    case 0n before:
      +eight_at = Equal.trans(Nat,at,Nat.add(at,0n),8n,Equal.sym(Nat,Nat.add(at,0n),at,Addition.zero_right(at)),total)
      +low_past = {%eight_at : {Nat.is_le(_,U32.to_nat(low)) == True{} : Bool} below : {Nat.is_le(8n,U32.to_nat(low)) == True{} : Bool}}
      +past = Order.transitive(U32.to_nat(high),8n,U32.to_nat(low),high_small,low_past)
      +not_lt = Equal.trans(Bool,U32.is_lt(low,high),Nat.is_lt(U32.to_nat(low),U32.to_nat(high)),False{},Comparison.lt(low,high),Order.not_below(U32.to_nat(high),U32.to_nat(low),past))
      %Equal.sym(Bool,U32.is_lt(low,high),False{},not_lt) : {tree == written_tree(8n,_,low,high,(start + low * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>}
      {==}
    case 1n+ +rest True{}:
      +nat_less = {%lane_exact : {Nat.is_lt(_,U32.to_nat(low)) == True{} : Bool} Bounds.as_nat_lt(lane,low,before_exact) : {Nat.is_lt(at,U32.to_nat(low)) == True{} : Bool}}
      %Equal.sym(Bool,U32.is_le(low,lane),False{},lane_not_from(low,lane,at,lane_exact,before_exact)) :
        {lane_walk(rest,(lane + 1 : U32),low,high,start,stride,eight,tree_when(_ && U32.is_lt(lane,high),(start + lane * stride : U32),eighth(lane,eight),tree)) ==
          written_tree(8n,U32.is_lt(low,high),low,high,(start + low * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>}
      skip_walk(rest,U32.is_lt((lane + 1 : U32),low),(lane + 1 : U32),1n+at,low,high,start,stride,eight,tree,{==},next_exact(lane,at,rest,lane_exact,total),next_total(at,rest,total),
        Order.strict_follow(at,U32.to_nat(low),nat_less),high_small)
    case 1n+ +rest False{}:
      +same_nat = Order.antisymmetric(at,U32.to_nat(low),below,lane_nat_past(lane,low,at,lane_exact,before_exact))
      +same = Arithmetic.to_nat_injective(lane,low,Equal.trans(Nat,U32.to_nat(lane),at,U32.to_nat(low),lane_exact,same_nat))
      %Equal.sym(U32,lane,low,same) :
        {lane_walk(1n+rest,_,low,high,start,stride,eight,tree) == written_tree(8n,U32.is_lt(low,high),low,high,(start + low * stride : U32),stride,eight,tree) : Tree.Tree<Tile.Tile8>}
      walk_written(1n+rest,8n,U32.is_lt(low,high),low,U32.to_nat(low),low,high,start,stride,eight,tree,{==},{==},
        {%same_nat : {Nat.add(_,1n+rest) == 8n : Nat} total : {Nat.add(U32.to_nat(low),1n+rest) == 8n : Nat}},
        Held.u32_le(low,low,Order.reflexive(U32.to_nat(low))),
        {%total : {Nat.is_le(1n+rest,_) == True{} : Bool} Order.right_le(at,1n+rest) : {Nat.is_le(1n+rest,8n) == True{} : Bool}},high_small)

# The writes of a block with high at most 8 are the recursive model from low.
def lanes_written(+low: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,+tree: Tree.Tree<Tile.Tile8>,+high_small: {U32.is_le(high,8) == True{} : Bool}) ->
  {lanes_tree(low,high,target,stride,eight,tree) == written_tree(8n,U32.is_lt(low,high),low,high,target,stride,eight,tree) : Tree.Tree<Tile.Tile8>}:
  +start = (target - low * stride : U32)
  Equal.trans(Tree.Tree<Tile.Tile8>,lanes_tree(low,high,target,stride,eight,tree),lane_walk(8n,0,low,high,start,stride,eight,tree),
    written_tree(8n,U32.is_lt(low,high),low,high,target,stride,eight,tree),lanes_walk(low,high,target,stride,eight,tree),
    {%Subtraction.add_subtract(target,(low * stride : U32)) :
        {lane_walk(8n,0,low,high,start,stride,eight,tree) == written_tree(8n,U32.is_lt(low,high),low,high,_,stride,eight,tree) : Tree.Tree<Tile.Tile8>}
      skip_walk(8n,U32.is_lt(0,low),0,0n,low,high,start,stride,eight,tree,{==},{==},{==},Order.zero_le(U32.to_nat(low)),Bounds.as_nat_le(high,8,high_small)) :
      {lane_walk(8n,0,low,high,start,stride,eight,tree) == written_tree(8n,U32.is_lt(low,high),low,high,target,stride,eight,tree) : Tree.Tree<Tile.Tile8>}})

# The contents the writes of a block leave on a buffer model: the tree of the
# recursive model.
def writes_packed(+low: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,+model: Buffer.Model<Tile.Tile8>,+high_small: {U32.is_le(high,8) == True{} : Bool}) ->
  {Buffer.contents(~Tile.Tile8,Tiled.block_writes(low,high,target,stride,eight,Buffer.pack(~Tile.Tile8,model))) ==
    Buffer.Content{written_tree(8n,U32.is_lt(low,high),low,high,target,stride,eight,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))),
      Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))} : Buffer.Content<Tile.Tile8>}:
  +tree = Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))
  +length = Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))
  Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Tiled.block_writes(low,high,target,stride,eight,Buffer.pack(~Tile.Tile8,model))),
    Buffer.Content{lanes_tree(low,high,target,stride,eight,tree),length},Buffer.Content{written_tree(8n,U32.is_lt(low,high),low,high,target,stride,eight,tree),length},
    lanes_packed(low,high,target,stride,eight,model),
    Equal.cong(Tree.Tree<Tile.Tile8>,Buffer.Content<Tile.Tile8>,written => Buffer.Content{written,length},lanes_tree(low,high,target,stride,eight,tree),
      written_tree(8n,U32.is_lt(low,high),low,high,target,stride,eight,tree),lanes_written(low,high,target,stride,eight,tree,high_small)))

# The contents the writes of a block leave: the tree of the recursive model.
def writes_contents(+low: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,output: Storage.Buffer<Tile.Tile8>,+high_small: {U32.is_le(high,8) == True{} : Bool}) ->
  {Buffer.contents(~Tile.Tile8,Tiled.block_writes(low,high,target,stride,eight,output)) ==
    Buffer.Content{written_tree(8n,U32.is_lt(low,high),low,high,target,stride,eight,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))),
      Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))} : Buffer.Content<Tile.Tile8>}:
  +observation = Buffer.observe(~Tile.Tile8,output)
  %Buffer.observed_equation(~Tile.Tile8,output,observation) :
    {Buffer.contents(~Tile.Tile8,Tiled.block_writes(low,high,target,stride,eight,_)) ==
      Buffer.Content{written_tree(8n,U32.is_lt(low,high),low,high,target,stride,eight,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,_))),
        Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,_))} : Buffer.Content<Tile.Tile8>}
  writes_packed(low,high,target,stride,eight,Buffer.observed_model(~Tile.Tile8,output,observation),high_small)

# What a cell holds after the writes of a block: the last cell written to
# it, or what it held.
def written_value(fuel: Nat,more: Bool,+index: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,+query: U32,+old: Tile.Tile8) -> Tile.Tile8:
  match fuel more:
    case 0n more: old
    case 1n+rest False{}: old
    case 1n+rest True{}:
      written_value(rest,U32.is_lt((index + 1 : U32),high),(index + 1 : U32),high,(target + stride : U32),stride,eight,query,
        Bool.pick(Tile.Tile8,U32.is_eq(target,query),eighth(index,eight),old))

# Every cell the writes of a block write lies inside a capacity.
def WritesInside(fuel: Nat,more: Bool,+index: U32,+high: U32,+target: U32,+stride: U32,+capacity: U32) -> Data:
  match fuel more:
    case 0n more: Unit
    case 1n+rest False{}: Unit
    case 1n+rest True{}:
      Held.Both<{U32.is_lt(target,capacity) == True{} : Bool},
        WritesInside(rest,U32.is_lt((index + 1 : U32),high),(index + 1 : U32),high,(target + stride : U32),stride,capacity)>

# Writes inside a capacity lie inside every wider one.
def writes_wider(fuel: Nat,more: Bool,+index: U32,+high: U32,+target: U32,+stride: U32,+capacity: U32,+wider: U32,
  +le: {U32.is_le(capacity,wider) == True{} : Bool},+inside: WritesInside(fuel,more,index,high,target,stride,capacity)) ->
  WritesInside(fuel,more,index,high,target,stride,wider):
  match fuel more:
    case 0n more: Unit{}
    case 1n+rest False{}: Unit{}
    case 1n+rest True{}:
      Held.Both{+here,+later} = inside
      Held.Both{Bounds.uint_trans(target,capacity,wider,here,le),
        writes_wider(rest,U32.is_lt((index + 1 : U32),high),(index + 1 : U32),high,(target + stride : U32),stride,capacity,wider,le,later)}

def write_read_when(same: Bool,+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +address: U32,+value: Tile.Tile8,+query: U32,+address_inside: {U32.is_lt(address,Tree.capacity(Tile.Tile8,tree)) == True{} : Bool},
  +query_inside: {U32.is_lt(query,Tree.capacity(Tile.Tile8,tree)) == True{} : Bool},+classified: {U32.is_eq(address,query) == same : Bool}) ->
  {Tree.read(Tile.Tile8,Tree.write(Tile.Tile8,tree,address,value),query) == Bool.pick(Tile.Tile8,same,value,Tree.read(Tile.Tile8,tree,query)) : Tile.Tile8}:
  match same:
    case True{}:
      %Arithmetic.equal_of_is_eq(address,query,classified) : {Tree.read(Tile.Tile8,Tree.write(Tile.Tile8,tree,address,value),_) == value : Tile.Tile8}
      StorageReadback.read_written(~Tile.Tile8,tree,address,value)
    case False{}:
      StorageReadback.read_preserved(~Tile.Tile8,depth,tree,address,query,value,balanced,limit,address_inside,query_inside,classified)

# A read after a write gives the value written there and what the cell held
# elsewhere.
def write_read(+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +address: U32,+value: Tile.Tile8,+query: U32,+address_inside: {U32.is_lt(address,Tree.capacity(Tile.Tile8,tree)) == True{} : Bool},
  +query_inside: {U32.is_lt(query,Tree.capacity(Tile.Tile8,tree)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Tree.write(Tile.Tile8,tree,address,value),query) == Bool.pick(Tile.Tile8,U32.is_eq(address,query),value,Tree.read(Tile.Tile8,tree,query)) : Tile.Tile8}:
  write_read_when(U32.is_eq(address,query),depth,tree,balanced,limit,address,value,query,address_inside,query_inside,{==})

# A cell after the writes of a block holds what written_value says.
def written_read(fuel: Nat,more: Bool,+index: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +inside: WritesInside(fuel,more,index,high,target,stride,Tree.capacity(Tile.Tile8,tree)),+query: U32,
  +query_inside: {U32.is_lt(query,Tree.capacity(Tile.Tile8,tree)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,written_tree(fuel,more,index,high,target,stride,eight,tree),query) ==
    written_value(fuel,more,index,high,target,stride,eight,query,Tree.read(Tile.Tile8,tree,query)) : Tile.Tile8}:
  match fuel more:
    case 0n more: {==}
    case 1n+rest False{}: {==}
    case 1n+ +rest True{}:
      Held.Both{+here,+later} = inside
      +next = (index + 1 : U32)
      +value = eighth(index,eight)
      +capacity = Tree.capacity(Tile.Tile8,tree)
      +written = Tree.write(Tile.Tile8,tree,target,value)
      +same_capacity = StorageReadback.write_capacity(~Tile.Tile8,tree,target,value)
      +back = Equal.sym(U32,Tree.capacity(Tile.Tile8,written),capacity,same_capacity)
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,written_tree(rest,U32.is_lt(next,high),next,high,(target + stride : U32),stride,eight,written),query),
        written_value(rest,U32.is_lt(next,high),next,high,(target + stride : U32),stride,eight,query,Tree.read(Tile.Tile8,written,query)),
        written_value(rest,U32.is_lt(next,high),next,high,(target + stride : U32),stride,eight,query,Bool.pick(Tile.Tile8,U32.is_eq(target,query),value,Tree.read(Tile.Tile8,tree,query))),
        written_read(rest,U32.is_lt(next,high),next,high,(target + stride : U32),stride,eight,depth,written,
          Shape.store(Tile.Tile8,depth,tree,capacity,U32.and(target,U32.sub(capacity,1)),value,balanced),limit,
          {%back : WritesInside(rest,U32.is_lt(next,high),next,high,(target + stride : U32),stride,_) later : WritesInside(rest,U32.is_lt(next,high),next,high,(target + stride : U32),stride,Tree.capacity(Tile.Tile8,written))},
          query,{%back : {U32.is_lt(query,_) == True{} : Bool} query_inside : {U32.is_lt(query,Tree.capacity(Tile.Tile8,written)) == True{} : Bool}}),
        Equal.cong(Tile.Tile8,Tile.Tile8,old => written_value(rest,U32.is_lt(next,high),next,high,(target + stride : U32),stride,eight,query,old),Tree.read(Tile.Tile8,written,query),
          Bool.pick(Tile.Tile8,U32.is_eq(target,query),value,Tree.read(Tile.Tile8,tree,query)),
          write_read(depth,tree,balanced,limit,target,value,query,here,query_inside)))

# ---- The transposed cells ----

# Lane lane of each of the eight cells, in order.
def column(+lane: Nat,+eight: Tiled.Eight) -> Tile.Tile8:
  Tiled.Eight{+c0,+c1,+c2,+c3,+c4,+c5,+c6,+c7} = eight
  Tile.Tile{TileLaws.select_lane(lane,c0),TileLaws.select_lane(lane,c1),TileLaws.select_lane(lane,c2),TileLaws.select_lane(lane,c3),
    TileLaws.select_lane(lane,c4),TileLaws.select_lane(lane,c5),TileLaws.select_lane(lane,c6),TileLaws.select_lane(lane,c7)}

def transposed_lane(+eight: Tiled.Eight,lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {eighth(U32.from_nat(lane),Tiled.transposed(eight)) == column(lane,eight) : Tile.Tile8}:
  match eight:
    case Tiled.Eight{Tile.Tile{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7},Tile.Tile{+b0,+b1,+b2,+b3,+b4,+b5,+b6,+b7},Tile.Tile{+c0,+c1,+c2,+c3,+c4,+c5,+c6,+c7},
        Tile.Tile{+d0,+d1,+d2,+d3,+d4,+d5,+d6,+d7},Tile.Tile{+e0,+e1,+e2,+e3,+e4,+e5,+e6,+e7},Tile.Tile{+f0,+f1,+f2,+f3,+f4,+f5,+f6,+f7},
        Tile.Tile{+g0,+g1,+g2,+g3,+g4,+g5,+g6,+g7},Tile.Tile{+h0,+h1,+h2,+h3,+h4,+h5,+h6,+h7}}:
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
                                    case 1n+beyond:
                                      Empty.absurd({eighth(U32.from_nat(8n+beyond),Tiled.transposed(Tiled.Eight{Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7},
                                        Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7},Tile.Tile{c0,c1,c2,c3,c4,c5,c6,c7},Tile.Tile{d0,d1,d2,d3,d4,d5,d6,d7},
                                        Tile.Tile{e0,e1,e2,e3,e4,e5,e6,e7},Tile.Tile{f0,f1,f2,f3,f4,f5,f6,f7},Tile.Tile{g0,g1,g2,g3,g4,g5,g6,g7},
                                        Tile.Tile{h0,h1,h2,h3,h4,h5,h6,h7}})) ==
                                        column(8n+beyond,Tiled.Eight{Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7},
                                        Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7},Tile.Tile{c0,c1,c2,c3,c4,c5,c6,c7},Tile.Tile{d0,d1,d2,d3,d4,d5,d6,d7},
                                        Tile.Tile{e0,e1,e2,e3,e4,e5,e6,e7},Tile.Tile{f0,f1,f2,f3,f4,f5,f6,f7},Tile.Tile{g0,g1,g2,g3,g4,g5,g6,g7},
                                        Tile.Tile{h0,h1,h2,h3,h4,h5,h6,h7}}) : Tile.Tile8},Order.below_zero(beyond,small))

# Cell index of the transposed cells holds lane index of each of the eight.
def transposed_at(+index: U32,+eight: Tiled.Eight,+small: {Nat.is_lt(U32.to_nat(index),8n) == True{} : Bool}) ->
  {eighth(index,Tiled.transposed(eight)) == column(U32.to_nat(index),eight) : Tile.Tile8}:
  %Arithmetic.native_roundtrip(index) : {eighth(_,Tiled.transposed(eight)) == column(U32.to_nat(index),eight) : Tile.Tile8}
  transposed_lane(eight,U32.to_nat(index),small)

# ---- One block ----

# The pieces a block leaves held, the eight cells it reads and the target
# cells it writes.
def block_holding(+block: Tiled.Block,+start: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  Tiled.Block{source,target,stride,low,high,+reads,+step} = block
  Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,block_model(reads,source,step,start))

def block_eight(+block: Tiled.Block,+start: Held.Holding<Tile.Tile8>) -> Tiled.Eight:
  Tiled.Block{source,target,stride,low,high,+reads,+step} = block
  Pair.snd(Held.Holding<Tile.Tile8>,Tiled.Eight,block_model(reads,source,step,start))

def block_output(+block: Tiled.Block,+eight: Tiled.Eight,output: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Tiled.Block{source,+target,+stride,+low,+high,reads,step} = block
  Tiled.block_writes(low,high,target,stride,Tiled.transposed(eight),output)

# A block on packed pieces leaves the packed model holding and writes the
# transposed cells it reads.
def block_written_packed(+block: Tiled.Block,+start: Held.Holding<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>) ->
  {Tiled.block_written(block,Held.pack(~Tile.Tile8,start),output) ==
    (Held.pack(~Tile.Tile8,block_holding(block,start)),block_output(block,block_eight(block,start),output)) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  match block:
    case Tiled.Block{+source,+target,+stride,+low,+high,+reads,+step}:
      %Equal.sym(Move.Holding<Tile.Tile8> & Tiled.Eight,Tiled.block_reads(reads,source,step,Held.pack(~Tile.Tile8,start)),pack_reads(block_model(reads,source,step,start)),block_reads_packed(reads,source,step,start)) :
        {Tiled.block_moved(Tiled.Block{source,target,stride,low,high,reads,step},_,output) ==
          (Held.pack(~Tile.Tile8,Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,block_model(reads,source,step,start))),Tiled.block_writes(low,high,target,stride,Tiled.transposed(Pair.snd(Held.Holding<Tile.Tile8>,Tiled.Eight,block_model(reads,source,step,start))),output)) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      %Equal.sym(Held.Holding<Tile.Tile8> & Tiled.Eight,block_model(reads,source,step,start),(Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,block_model(reads,source,step,start)),Pair.snd(Held.Holding<Tile.Tile8>,Tiled.Eight,block_model(reads,source,step,start))),Held.pair_eta(~Held.Holding<Tile.Tile8>,~Tiled.Eight,block_model(reads,source,step,start))) :
        {Tiled.block_moved(Tiled.Block{source,target,stride,low,high,reads,step},pack_reads(_),output) ==
          (Held.pack(~Tile.Tile8,Pair.fst(Held.Holding<Tile.Tile8>,Tiled.Eight,block_model(reads,source,step,start))),Tiled.block_writes(low,high,target,stride,Tiled.transposed(Pair.snd(Held.Holding<Tile.Tile8>,Tiled.Eight,block_model(reads,source,step,start))),output)) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      {==}

# ---- Writes that hit a cell with its value ----

def pick_same(+choice: Bool,+value: Tile.Tile8) -> {Bool.pick(Tile.Tile8,choice,value,value) == value : Tile.Tile8}:
  match choice:
    case True{}: {==}
    case False{}: {==}

# A pick that keeps expected when the first choice holds it.
def pick_merge(+first: Bool,+second: Bool,+expected: Tile.Tile8,+old: Tile.Tile8) ->
  {Bool.pick(Tile.Tile8,second,expected,Bool.pick(Tile.Tile8,first,expected,old)) == Bool.pick(Tile.Tile8,first || second,expected,old) : Tile.Tile8}:
  match first:
    case True{}: pick_same(second,expected)
    case False{}: {==}

# Some write of a block hits query.
def writes_hit(fuel: Nat,more: Bool,+index: U32,+high: U32,+target: U32,+stride: U32,+query: U32) -> Bool:
  match fuel more:
    case 0n more: False{}
    case 1n+rest False{}: False{}
    case 1n+rest True{}: U32.is_eq(target,query) || writes_hit(rest,U32.is_lt((index + 1 : U32),high),(index + 1 : U32),high,(target + stride : U32),stride,query)

# Every write of a block that hits query writes expected.
def WritesSound(fuel: Nat,more: Bool,+index: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,+query: U32,+expected: Tile.Tile8) -> Data:
  match fuel more:
    case 0n more: Unit
    case 1n+rest False{}: Unit
    case 1n+rest True{}:
      Held.Both<{Bool.pick(Tile.Tile8,U32.is_eq(target,query),eighth(index,eight),expected) == expected : Tile.Tile8},
        WritesSound(rest,U32.is_lt((index + 1 : U32),high),(index + 1 : U32),high,(target + stride : U32),stride,eight,query,expected)>

def sound_step(hit: Bool,+value: Tile.Tile8,+expected: Tile.Tile8,+old: Tile.Tile8,+later: Bool,
  +sound: {Bool.pick(Tile.Tile8,hit,value,expected) == expected : Tile.Tile8}) ->
  {Bool.pick(Tile.Tile8,later,expected,Bool.pick(Tile.Tile8,hit,value,old)) == Bool.pick(Tile.Tile8,hit || later,expected,old) : Tile.Tile8}:
  match hit:
    case True{}:
      %Equal.sym(Tile.Tile8,value,expected,sound) : {Bool.pick(Tile.Tile8,later,expected,_) == expected : Tile.Tile8}
      pick_same(later,expected)
    case False{}: {==}

# A cell after the writes of a block holds expected when one hit it, and what
# it held otherwise, when every write that hits it writes expected.
def written_sound(fuel: Nat,more: Bool,+index: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,+query: U32,+expected: Tile.Tile8,+old: Tile.Tile8,
  +sound: WritesSound(fuel,more,index,high,target,stride,eight,query,expected)) ->
  {written_value(fuel,more,index,high,target,stride,eight,query,old) == Bool.pick(Tile.Tile8,writes_hit(fuel,more,index,high,target,stride,query),expected,old) : Tile.Tile8}:
  match fuel more:
    case 0n more: {==}
    case 1n+rest False{}: {==}
    case 1n+ +rest True{}:
      Held.Both{+here,+later} = sound
      +next = (index + 1 : U32)
      +hit = U32.is_eq(target,query)
      +value = eighth(index,eight)
      +rest_hit = writes_hit(rest,U32.is_lt(next,high),next,high,(target + stride : U32),stride,query)
      Equal.trans(Tile.Tile8,written_value(rest,U32.is_lt(next,high),next,high,(target + stride : U32),stride,eight,query,Bool.pick(Tile.Tile8,hit,value,old)),
        Bool.pick(Tile.Tile8,rest_hit,expected,Bool.pick(Tile.Tile8,hit,value,old)),Bool.pick(Tile.Tile8,hit || rest_hit,expected,old),
        written_sound(rest,U32.is_lt(next,high),next,high,(target + stride : U32),stride,eight,query,expected,Bool.pick(Tile.Tile8,hit,value,old),later),
        sound_step(hit,value,expected,old,rest_hit,here))

# The writes of a block keep a tree balanced at its depth and its capacity.
def written_balanced(fuel: Nat,more: Bool,+index: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree)) -> Shape.Balanced(Tile.Tile8,depth,written_tree(fuel,more,index,high,target,stride,eight,tree)):
  match fuel more:
    case 0n more: balanced
    case 1n+rest False{}: balanced
    case 1n+ +rest True{}:
      +value = eighth(index,eight)
      +capacity = Tree.capacity(Tile.Tile8,tree)
      written_balanced(rest,U32.is_lt((index + 1 : U32),high),(index + 1 : U32),high,(target + stride : U32),stride,eight,depth,Tree.write(Tile.Tile8,tree,target,value),
        Shape.store(Tile.Tile8,depth,tree,capacity,U32.and(target,U32.sub(capacity,1)),value,balanced))

def written_capacity(fuel: Nat,more: Bool,+index: U32,+high: U32,+target: U32,+stride: U32,+eight: Tiled.Eight,+tree: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,written_tree(fuel,more,index,high,target,stride,eight,tree)) == Tree.capacity(Tile.Tile8,tree) : U32}:
  match fuel more:
    case 0n more: {==}
    case 1n+rest False{}: {==}
    case 1n+ +rest True{}:
      +value = eighth(index,eight)
      +written = Tree.write(Tile.Tile8,tree,target,value)
      Equal.trans(U32,Tree.capacity(Tile.Tile8,written_tree(rest,U32.is_lt((index + 1 : U32),high),(index + 1 : U32),high,(target + stride : U32),stride,eight,written)),
        Tree.capacity(Tile.Tile8,written),Tree.capacity(Tile.Tile8,tree),
        written_capacity(rest,U32.is_lt((index + 1 : U32),high),(index + 1 : U32),high,(target + stride : U32),stride,eight,written),
        StorageReadback.write_capacity(~Tile.Tile8,tree,target,value))
