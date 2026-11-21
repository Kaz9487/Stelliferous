# The gather of a write along the lane axis: the target piece lent whole from
# the head of the target holding, then every target cell the part reaches
# merged with the part cells it reads. Each read keeps the part pieces in
# their order.

import Base
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_move.bend as Move
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_move_readback.bend as Held
import ./tensor_relane_block.bend as Block
import ./tensor_moved.bend as Moved
import ./tensor_tiled_representation.bend as Representation
import ./nat_division_order.bend as DivisionOrder
import ./u32_division.bend as U32Division
import ./u32_extrema.bend as Extrema
import ./nat_extrema.bend as NatExtrema
import ./word_encoding.bend as WordEncoding
import ./row_major_index.bend as RowMajor
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./boolean_logic.bend as Logic
import ./storage_shape.bend as Shape
import ./nat_to_u32_bounds.bend as Bounds
import ./storage_readback.bend as StorageReadback

# ---- The head piece lent whole ----

def word_below_self(+word: U32) -> {U32.is_lt(word,word) == False{} : Bool}:
  Equal.trans(Bool,U32.is_lt(word,word),Nat.is_lt(U32.to_nat(word),U32.to_nat(word)),False{},Comparison.lt(word,word),
    Order.not_below(U32.to_nat(word),U32.to_nat(word),Order.reflexive(U32.to_nat(word))))

def word_up_to_self(+word: U32) -> {U32.is_le(word,word) == True{} : Bool}:
  Equal.trans(Bool,U32.is_le(word,word),Nat.is_le(U32.to_nat(word),U32.to_nat(word)),True{},Comparison.le(word,word),Order.reflexive(U32.to_nat(word)))

# A piece of cells ends after its first cell.
def end_after(+first: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.is_le((first + count : U32),first) == False{} : Bool}:
  +sum = Arithmetic.add(first,count,cap,cap_ok,room)
  Equal.trans(Bool,U32.is_le((first + count : U32),first),Nat.is_le(U32.to_nat((first + count : U32)),U32.to_nat(first)),False{},Comparison.le((first + count : U32),first),
    %Equal.sym(Nat,U32.to_nat((first + count : U32)),Nat.add(U32.to_nat(first),U32.to_nat(count)),sum) : {Nat.is_le(_,U32.to_nat(first)) == False{} : Bool}
    Held.lt_not_le(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),
      {%Addition.zero_right(U32.to_nat(first)) : {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}
        Order.offset_lt(U32.to_nat(first),0n,U32.to_nat(count),positive) : {Nat.is_lt(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}}))

# A piece that does not end at or before a cell stays current.
def advanced_stays(+below: Held.Stack<Tile.Tile8>,+index: U32,+buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,+behind: Held.Stack<Tile.Tile8>) ->
  {Held.advanced(~Tile.Tile8,below,False{},index,buffer,first,count,behind) == Held.Holding{behind,Held.Held{buffer,first,count,below}} : Held.Holding<Tile.Tile8>}:
  match below:
    case Held.Bottom{}: {==}
    case Held.Held{next,next_first,next_count,rest}: {==}

# Resolving the first cell of the head piece keeps it current.
def head_resolved(+buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,+below: Held.Stack<Tile.Tile8>,
  +ends: {U32.is_le((first + count : U32),first) == False{} : Bool}) ->
  {Held.resolve(~Tile.Tile8,Held.Holding{Held.Bottom{},Held.Held{buffer,first,count,below}},first) == Held.Holding{Held.Bottom{},Held.Held{buffer,first,count,below}} :
    Held.Holding<Tile.Tile8>}:
  %Equal.sym(Bool,U32.is_lt(first,first),False{},word_below_self(first)) :
    {Held.located(~Tile.Tile8,_,first,Held.Bottom{},buffer,first,count,below) == Held.Holding{Held.Bottom{},Held.Held{buffer,first,count,below}} : Held.Holding<Tile.Tile8>}
  %Equal.sym(Bool,U32.is_le((first + count : U32),first),False{},ends) :
    {Held.advanced(~Tile.Tile8,below,_,first,buffer,first,count,Held.Bottom{}) == Held.Holding{Held.Bottom{},Held.Held{buffer,first,count,below}} : Held.Holding<Tile.Tile8>}
  advanced_stays(below,first,buffer,first,count,Held.Bottom{})

# The band of the head piece is the piece lent whole.
def head_band(buffer: Storage.Buffer<Tile.Tile8>,+first: U32,+count: U32,below: Move.Stack<Tile.Tile8>,spare: Move.Spare<Tile.Tile8>) ->
  {Move.band_current(~Tile.Tile8,first,count,Tiled.zero_tile(),Move.Holding{Move.Bottom{},Move.Held{buffer,first,count,below}},spare) ==
    Move.Band{Move.Hole{Move.Bottom{},below,first,count,spare},buffer,(first - first : U32)} : Move.Band<Tile.Tile8>}:
  %Equal.sym(Bool,U32.is_le(first,first),True{},word_up_to_self(first)) :
    {Move.band_when(~Tile.Tile8,_ && U32.is_le((first + count : U32),(first + count : U32)),first,count,Tiled.zero_tile(),Move.Bottom{},buffer,first,count,below,spare) ==
      Move.Band{Move.Hole{Move.Bottom{},below,first,count,spare},buffer,(first - first : U32)} : Move.Band<Tile.Tile8>}
  %Equal.sym(Bool,U32.is_le((first + count : U32),(first + count : U32)),True{},word_up_to_self((first + count : U32))) :
    {Move.band_when(~Tile.Tile8,True{} && _,first,count,Tiled.zero_tile(),Move.Bottom{},buffer,first,count,below,spare) ==
      Move.Band{Move.Hole{Move.Bottom{},below,first,count,spare},buffer,(first - first : U32)} : Move.Band<Tile.Tile8>}
  {==}

# The piece lent whole is written in place.
def head_lent(rest: Tensors.Holdings<Tile.Tile8>,+write: Tiled.Write,+first: U32,+count: U32,below: Move.Stack<Tile.Tile8>,spare: Move.Spare<Tile.Tile8>,
  buffer: Storage.Buffer<Tile.Tile8>,+offset: U32) ->
  {Tiled.write_lent(rest,write,first,count,Move.Band{Move.Hole{Move.Bottom{},below,first,count,spare},buffer,offset}) ==
    Tiled.write_part(rest,write,first,count,(Move.Holding{Move.Bottom{},below},buffer)) : Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  %Equal.sym(Bool,U32.is_eq(first,first),True{},Arithmetic.equal_reflexive(first)) :
    {Tiled.write_hole(_ && U32.is_eq(count,count),rest,write,first,count,Move.Bottom{},below,first,count,buffer) ==
      Tiled.write_part(rest,write,first,count,(Move.Holding{Move.Bottom{},below},buffer)) : Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  %Equal.sym(Bool,U32.is_eq(count,count),True{},Arithmetic.equal_reflexive(count)) :
    {Tiled.write_hole(True{} && _,rest,write,first,count,Move.Bottom{},below,first,count,buffer) ==
      Tiled.write_part(rest,write,first,count,(Move.Holding{Move.Bottom{},below},buffer)) : Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  {==}

# A gather from a target holding whose head piece is the target piece takes
# that piece whole and leaves the pieces after it.
def head_gathered(+write: Tiled.Write,+components: Tasks.Components,+fill: Tile.Tile8,+buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,
  +below: Held.Stack<Tile.Tile8>,rest: Tensors.Holdings<Tile.Tile8>,+ends: {U32.is_le((first + count : U32),first) == False{} : Bool}) ->
  {Tiled.write_gather(write,components,first,count,fill,Tensors.Holds{Held.pack(~Tile.Tile8,Held.Holding{Held.Bottom{},Held.Held{buffer,first,count,below}}),rest}) ==
    Tiled.write_part(rest,write,first,count,(Move.Holding{Move.Bottom{},Held.pack_stack(~Tile.Tile8,below)},Buffer.pack(~Tile.Tile8,buffer))) :
    Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  +holding = {Held.Holding{Held.Bottom{},Held.Held{buffer,first,count,below}} : Held.Holding<Tile.Tile8>}
  %Equal.sym(Move.Holding<Tile.Tile8>,Move.resolve(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),first),Held.pack(~Tile.Tile8,Held.resolve(~Tile.Tile8,holding,first)),
      Held.resolve_packed(~Tile.Tile8,holding,first)) :
    {Tiled.write_lent(rest,write,first,count,Move.band_current(~Tile.Tile8,first,count,Tiled.zero_tile(),_,Move.spare(~Tile.Tile8,Tiled.zero_tile()))) ==
      Tiled.write_part(rest,write,first,count,(Move.Holding{Move.Bottom{},Held.pack_stack(~Tile.Tile8,below)},Buffer.pack(~Tile.Tile8,buffer))) :
      Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  %Equal.sym(Held.Holding<Tile.Tile8>,Held.resolve(~Tile.Tile8,holding,first),holding,head_resolved(buffer,first,count,below,ends)) :
    {Tiled.write_lent(rest,write,first,count,Move.band_current(~Tile.Tile8,first,count,Tiled.zero_tile(),Held.pack(~Tile.Tile8,_),Move.spare(~Tile.Tile8,Tiled.zero_tile()))) ==
      Tiled.write_part(rest,write,first,count,(Move.Holding{Move.Bottom{},Held.pack_stack(~Tile.Tile8,below)},Buffer.pack(~Tile.Tile8,buffer))) :
      Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  %Equal.sym(Move.Band<Tile.Tile8>,
      Move.band_current(~Tile.Tile8,first,count,Tiled.zero_tile(),Move.Holding{Move.Bottom{},Move.Held{Buffer.pack(~Tile.Tile8,buffer),first,count,Held.pack_stack(~Tile.Tile8,below)}},Move.spare(~Tile.Tile8,Tiled.zero_tile())),
      Move.Band{Move.Hole{Move.Bottom{},Held.pack_stack(~Tile.Tile8,below),first,count,Move.spare(~Tile.Tile8,Tiled.zero_tile())},Buffer.pack(~Tile.Tile8,buffer),(first - first : U32)},
      head_band(Buffer.pack(~Tile.Tile8,buffer),first,count,Held.pack_stack(~Tile.Tile8,below),Move.spare(~Tile.Tile8,Tiled.zero_tile()))) :
    {Tiled.write_lent(rest,write,first,count,_) ==
      Tiled.write_part(rest,write,first,count,(Move.Holding{Move.Bottom{},Held.pack_stack(~Tile.Tile8,below)},Buffer.pack(~Tile.Tile8,buffer))) :
      Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  head_lent(rest,write,first,count,Held.pack_stack(~Tile.Tile8,below),Move.spare(~Tile.Tile8,Tiled.zero_tile()),Buffer.pack(~Tile.Tile8,buffer),(first - first : U32))

# ---- A read of the part ----

# The part pieces after a read, and the cell it gives: +0 when unwanted.
def read_step(wanted: Bool,+address: U32,+part: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  Bool.pick(Held.Holding<Tile.Tile8>,wanted,Block.held_moved(part,address),part)

def read_value(wanted: Bool,+address: U32,+part: Held.Holding<Tile.Tile8>) -> Tile.Tile8:
  Bool.pick(Tile.Tile8,wanted,Block.held_value(part,address),Tiled.zero_tile())

def read_packed(wanted: Bool,+source: U32,+part: Held.Holding<Tile.Tile8>) ->
  {Tiled.block_read(~Move.Holding<Tile.Tile8>,~Tiled.held_cell,wanted,source,0,0,Held.pack(~Tile.Tile8,part)) ==
    (Held.pack(~Tile.Tile8,read_step(wanted,(source + 0 * 0 : U32),part)),read_value(wanted,(source + 0 * 0 : U32),part)) : Move.Holding<Tile.Tile8> & Tile.Tile8}:
  match wanted:
    case True{}: Block.held_packed(part,(source + 0 * 0 : U32))
    case False{}: {==}

# ---- The cells the part reaches ----

# The part pieces and the target piece while a write merges cells.
type LaneState is Data:
  LaneState{part: Held.Holding<Tile.Tile8>,output: Buffer.Model<Tile.Tile8>}

def state_part(+state: LaneState) -> Held.Holding<Tile.Tile8>:
  LaneState{+part,+output} = state
  part

def state_output(+state: LaneState) -> Buffer.Model<Tile.Tile8>:
  LaneState{+part,+output} = state
  output

def pack_state(+target: Held.Holding<Tile.Tile8>,+state: LaneState) -> Tiled.Writing:
  Tiled.Writing{Held.pack(~Tile.Tile8,target),Held.pack(~Tile.Tile8,state_part(state)),Buffer.pack(~Tile.Tile8,state_output(state))}

# A cell the part reaches: the part cells of its reach read, joined at the
# shift and put over the cell between the channels of the part.
def cell_model(+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+group: U32,+other: U32,+state: LaneState) ->
  LaneState:
  LaneState{+part,+output} = state
  +reach = (8 * group + 8 - start : U32)
  +above = (reach / 8 : U32)
  +shift = (reach % 8 : U32)
  +low_wanted = U32.is_le(8,reach)
  +low = (Tiled.cell_of(position,other,(above - 1 : U32),part_groups,others) + 0 * 0 : U32)
  +high_wanted = U32.is_lt(0,shift) && U32.is_lt(above,part_groups)
  +high = (Tiled.cell_of(position,other,above,part_groups,others) + 0 * 0 : U32)
  +middle = read_step(low_wanted,low,part)
  +local = (Tiled.cell_of(position,other,group,groups,others) - first : U32)
  +value = Tiled.lanes_between(start,(start + width : U32),(8 * group : U32),Tiled.joined(shift,read_value(low_wanted,low,part),read_value(high_wanted,high,middle)),
    Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),local))
  LaneState{read_step(high_wanted,high,middle),Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,local,value,Buffer.pack(~Tile.Tile8,output)))}

def cell_packed(+target: Held.Holding<Tile.Tile8>,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+group: U32,+other: U32,
  +state: LaneState) ->
  {Tiled.touched_cell(Tiled.Write{start,width,groups,part_groups,others,position},first,group,other,pack_state(target,state)) ==
    pack_state(target,cell_model(start,width,groups,part_groups,others,position,first,group,other,state)) : Tiled.Writing}:
  LaneState{+part,+output} = state
  +reach = (8 * group + 8 - start : U32)
  +above = (reach / 8 : U32)
  +shift = (reach % 8 : U32)
  +low_wanted = U32.is_le(8,reach)
  +low_source = Tiled.cell_of(position,other,(above - 1 : U32),part_groups,others)
  +low = (low_source + 0 * 0 : U32)
  +high_wanted = U32.is_lt(0,shift) && U32.is_lt(above,part_groups)
  +high_source = Tiled.cell_of(position,other,above,part_groups,others)
  +high = (high_source + 0 * 0 : U32)
  +middle = read_step(low_wanted,low,part)
  +below = read_value(low_wanted,low,part)
  +after = read_step(high_wanted,high,middle)
  +over = read_value(high_wanted,high,middle)
  +local = (Tiled.cell_of(position,other,group,groups,others) - first : U32)
  +kept = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),local)
  +value = Tiled.lanes_between(start,(start + width : U32),(8 * group : U32),Tiled.joined(shift,below,over),kept)
  %Equal.sym(Move.Holding<Tile.Tile8> & Tile.Tile8,Tiled.block_read(~Move.Holding<Tile.Tile8>,~Tiled.held_cell,low_wanted,low_source,0,0,Held.pack(~Tile.Tile8,part)),
      (Held.pack(~Tile.Tile8,middle),below),read_packed(low_wanted,low_source,part)) :
    {Tiled.touched_low(shift,start,(start + width : U32),(8 * group : U32),local,high_wanted,high_source,Held.pack(~Tile.Tile8,target),Buffer.pack(~Tile.Tile8,output),_) == pack_state(target,LaneState{after,Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,local,value,Buffer.pack(~Tile.Tile8,output)))}) : Tiled.Writing}
  %Equal.sym(Move.Holding<Tile.Tile8> & Tile.Tile8,Tiled.block_read(~Move.Holding<Tile.Tile8>,~Tiled.held_cell,high_wanted,high_source,0,0,Held.pack(~Tile.Tile8,middle)),
      (Held.pack(~Tile.Tile8,after),over),read_packed(high_wanted,high_source,middle)) :
    {Tiled.touched_high(shift,start,(start + width : U32),(8 * group : U32),local,below,Held.pack(~Tile.Tile8,target),Buffer.pack(~Tile.Tile8,output),_) == pack_state(target,LaneState{after,Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,local,value,Buffer.pack(~Tile.Tile8,output)))}) : Tiled.Writing}
  %Equal.sym(Storage.Buffer<Tile.Tile8> & Tile.Tile8,Storage.read(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output),local),(Buffer.pack(~Tile.Tile8,output),kept),
      Buffer.read(~Tile.Tile8,output,local)) :
    {Tiled.touched_written(start,(start + width : U32),(8 * group : U32),local,Tiled.joined(shift,below,over),Held.pack(~Tile.Tile8,target),Held.pack(~Tile.Tile8,after),_) == pack_state(target,LaneState{after,Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,local,value,Buffer.pack(~Tile.Tile8,output)))}) : Tiled.Writing}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,local,value,Buffer.pack(~Tile.Tile8,output)))),Storage.write(~Tile.Tile8,local,value,Buffer.pack(~Tile.Tile8,output)),Buffer.remodeled(~Tile.Tile8,Storage.write(~Tile.Tile8,local,value,Buffer.pack(~Tile.Tile8,output)))) :
    {Tiled.Writing{Held.pack(~Tile.Tile8,target),Held.pack(~Tile.Tile8,after),Storage.write(~Tile.Tile8,local,value,Buffer.pack(~Tile.Tile8,output))} == Tiled.Writing{Held.pack(~Tile.Tile8,target),Held.pack(~Tile.Tile8,after),_} : Tiled.Writing}
  {==}

def when_model(inside: Bool,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+group: U32,+other: U32,
  +state: LaneState) -> LaneState:
  match inside:
    case True{}: cell_model(start,width,groups,part_groups,others,position,first,group,other,state)
    case False{}: state

def when_packed(+target: Held.Holding<Tile.Tile8>,inside: Bool,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+group: U32,+other: U32,
  +state: LaneState) ->
  {Tiled.touched_when(inside,Tiled.Write{start,width,groups,part_groups,others,position},first,group,other,pack_state(target,state)) ==
    pack_state(target,when_model(inside,start,width,groups,part_groups,others,position,first,group,other,state)) : Tiled.Writing}:
  match inside:
    case True{}: cell_packed(target,start,width,groups,part_groups,others,position,first,group,other,state)
    case False{}: {==}

# The other indices of a group, each cell inside the target piece merged.
def others_model(fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,
  +other: U32,+state: LaneState) -> LaneState:
  match fuel:
    case 0n: state
    case 1n+rest:
      +index = Tiled.cell_of(position,other,group,groups,others)
      others_model(rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),
        when_model(U32.is_le(first,index) && U32.is_lt(index,(first + count : U32)),start,width,groups,part_groups,others,position,first,group,other,state))

def others_packed(+target: Held.Holding<Tile.Tile8>,fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,
  +other: U32,+state: LaneState) ->
  {Tiled.touched_others(fuel,Tiled.Write{start,width,groups,part_groups,others,position},first,count,group,other,pack_state(target,state)) ==
    pack_state(target,others_model(fuel,start,width,groups,part_groups,others,position,first,count,group,other,state)) : Tiled.Writing}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +index = Tiled.cell_of(position,other,group,groups,others)
      +inside = U32.is_le(first,index) && U32.is_lt(index,(first + count : U32))
      +next = when_model(inside,start,width,groups,part_groups,others,position,first,group,other,state)
      %Equal.sym(Tiled.Writing,Tiled.touched_when(inside,Tiled.Write{start,width,groups,part_groups,others,position},first,group,other,pack_state(target,state)),
          pack_state(target,next),when_packed(target,inside,start,width,groups,part_groups,others,position,first,group,other,state)) :
        {Tiled.touched_others(rest,Tiled.Write{start,width,groups,part_groups,others,position},first,count,group,(other + 1 : U32),_) ==
          pack_state(target,others_model(rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),next)) : Tiled.Writing}
      others_packed(target,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),next)

# The other indices a group's cells take in the target piece.
def range_model(found: Tiled.Others,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,
  +state: LaneState) -> LaneState:
  Tiled.Others{+low,+high} = found
  others_model(U32.to_nat((U32.max(high,low) - low : U32)),start,width,groups,part_groups,others,position,first,count,group,low,state)

def range_packed(+target: Held.Holding<Tile.Tile8>,+found: Tiled.Others,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,
  +state: LaneState) ->
  {Tiled.touched_range(found,Tiled.Write{start,width,groups,part_groups,others,position},first,count,group,pack_state(target,state)) ==
    pack_state(target,range_model(found,start,width,groups,part_groups,others,position,first,count,group,state)) : Tiled.Writing}:
  Tiled.Others{+low,+high} = found
  others_packed(target,U32.to_nat((U32.max(high,low) - low : U32)),start,width,groups,part_groups,others,position,first,count,group,low,state)

# Every group the part reaches, from the group of its first index on.
def groups_model(fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,
  +state: LaneState) -> LaneState:
  match fuel:
    case 0n: state
    case 1n+rest:
      groups_model(rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),
        range_model(Tiled.others_of(position,groups,others,first,count,group),start,width,groups,part_groups,others,position,first,count,group,state))

def groups_packed(+target: Held.Holding<Tile.Tile8>,fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,
  +state: LaneState) ->
  {Tiled.touched_groups(fuel,Tiled.Write{start,width,groups,part_groups,others,position},first,count,group,pack_state(target,state)) ==
    pack_state(target,groups_model(fuel,start,width,groups,part_groups,others,position,first,count,group,state)) : Tiled.Writing}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +found = Tiled.others_of(position,groups,others,first,count,group)
      +next = range_model(found,start,width,groups,part_groups,others,position,first,count,group,state)
      %Equal.sym(Tiled.Writing,Tiled.touched_range(found,Tiled.Write{start,width,groups,part_groups,others,position},first,count,group,pack_state(target,state)),
          pack_state(target,next),range_packed(target,found,start,width,groups,part_groups,others,position,first,count,group,state)) :
        {Tiled.touched_groups(rest,Tiled.Write{start,width,groups,part_groups,others,position},first,count,(group + 1 : U32),_) ==
          pack_state(target,groups_model(rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),next)) : Tiled.Writing}
      groups_packed(target,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),next)

# ---- What a merged cell leaves ----

# A cell of the part a write reads: its value, or +0 when unwanted.
def part_cell(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,wanted: Bool,+address: U32) -> Tile.Tile8:
  Bool.pick(Tile.Tile8,wanted,pvalue(pcontext,address),Tiled.zero_tile())

# The value a write gives a reached cell over the target cell it keeps.
def touched_value(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+start: U32,+width: U32,+part_groups: U32,+others: U32,
  +position: Grid.Position,+group: U32,+other: U32,+kept: Tile.Tile8) -> Tile.Tile8:
  +reach = (8 * group + 8 - start : U32)
  +above = (reach / 8 : U32)
  +shift = (reach % 8 : U32)
  Tiled.lanes_between(start,(start + width : U32),(8 * group : U32),
    Tiled.joined(shift,part_cell(~PContext,~pvalue,pcontext,U32.is_le(8,reach),(Tiled.cell_of(position,other,(above - 1 : U32),part_groups,others) + 0 * 0 : U32)),
      part_cell(~PContext,~pvalue,pcontext,U32.is_lt(0,shift) && U32.is_lt(above,part_groups),(Tiled.cell_of(position,other,above,part_groups,others) + 0 * 0 : U32))),
    kept)

# The part cells a reached cell reads lie in the pieces held.
def ReadsHeld(+spans: Held.Spans,+start: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+group: U32,+other: U32) -> Data:
  +reach = (8 * group + 8 - start : U32)
  +above = (reach / 8 : U32)
  +shift = (reach % 8 : U32)
  Held.Both<Logic.When(U32.is_le(8,reach),Held.SpanHeld(spans,U32.to_nat((Tiled.cell_of(position,other,(above - 1 : U32),part_groups,others) + 0 * 0 : U32)))),
    Logic.When(U32.is_lt(0,shift) && U32.is_lt(above,part_groups),Held.SpanHeld(spans,U32.to_nat((Tiled.cell_of(position,other,above,part_groups,others) + 0 * 0 : U32))))>

# The part pieces in their order, with the values of the part.
type PartKept<-PContext: Data,-pvalue: PContext -> U32 -> Tile.Tile8,-pcontext: PContext,-stack: Held.Stack<Tile.Tile8>,-part: Held.Holding<Tile.Tile8>> is Data:
  PartKept{ordered: {Held.order(~Tile.Tile8,part) == stack : Held.Stack<Tile.Tile8>},values: Held.HoldingValues(~Tile.Tile8,~PContext,~pvalue,pcontext,part)}

# Sorted part pieces and the part values: what every read needs.
type PartSorted<-stack: Held.Stack<Tile.Tile8>> is Data:
  PartSorted{low: Nat,high: Nat,capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
    sorted: Held.Sorted(~Tile.Tile8,stack,low,high)}

# A read of a held part cell gives its value and keeps the part pieces.
def read_kept(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+wanted: Bool,+address: U32,+part: Held.Holding<Tile.Tile8>,
  +stack: Held.Stack<Tile.Tile8>,+facts: PartSorted<stack>,+kept: PartKept<PContext,pvalue,pcontext,stack,part>,
  +held: Logic.When(wanted,Held.SpanHeld(Held.spans(~Tile.Tile8,stack),U32.to_nat(address)))) ->
  Held.Both<PartKept<PContext,pvalue,pcontext,stack,read_step(wanted,address,part)>,
    {read_value(wanted,address,part) == part_cell(~PContext,~pvalue,pcontext,wanted,address) : Tile.Tile8}>:
  PartSorted{+low,+high,+capacity,+capacity_ok,+bounded,+sorted} = facts
  PartKept{+ordered,+values} = kept
  +back = Equal.sym(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,part),stack,ordered)
  Held.Both{PartKept{Equal.trans(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,read_step(wanted,address,part)),Held.order(~Tile.Tile8,part),stack,
        Block.picked_order(~Tile.Tile8,wanted,address,part),ordered),
      Block.picked_values(~Tile.Tile8,~PContext,~pvalue,pcontext,wanted,address,part,values)},
    Block.picked_cell(~PContext,~pvalue,pcontext,wanted,address,part,low,high,capacity,capacity_ok,bounded,
      {%back : Held.Sorted(~Tile.Tile8,_,low,high) sorted : Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,part),low,high)},
      {%back : Logic.When(wanted,Held.SpanHeld(Held.spans(~Tile.Tile8,_),U32.to_nat(address)))
        held : Logic.When(wanted,Held.SpanHeld(Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,part)),U32.to_nat(address)))},
      values)}

# A buffer written at a cell holds the tree written there and its length.
def written_content(+index: U32,+value: Tile.Tile8,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,index,value,Buffer.pack(~Tile.Tile8,output)))) == Buffer.Content{Tree.write(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),index,value),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output))} : Buffer.Content<Tile.Tile8>}:
  Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,index,value,Buffer.pack(~Tile.Tile8,output)))),Buffer.contents(~Tile.Tile8,Storage.write(~Tile.Tile8,index,value,Buffer.pack(~Tile.Tile8,output))),
    Buffer.Content{Tree.write(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),index,value),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output))},
    Buffer.remodel_content(~Tile.Tile8,Storage.write(~Tile.Tile8,index,value,Buffer.pack(~Tile.Tile8,output))),
    Buffer.write_value_packed(~Tile.Tile8,index,value,output))

def written_length(+index: U32,+value: Tile.Tile8,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,index,value,Buffer.pack(~Tile.Tile8,output))))) == Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)) : Nat}:
  Equal.cong(Buffer.Content<Tile.Tile8>,Nat,Buffer.content_length(~Tile.Tile8),Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,index,value,Buffer.pack(~Tile.Tile8,output)))),Buffer.Content{Tree.write(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),index,value),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output))},
    written_content(index,value,output))

# A merged cell keeps the length of the target piece.
def cell_sized(+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+group: U32,+other: U32,
  +part: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,state_output(cell_model(start,width,groups,part_groups,others,position,first,group,other,LaneState{part,output})))) ==
    Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)) : Nat}:
  +reach = (8 * group + 8 - start : U32)
  +above = (reach / 8 : U32)
  +shift = (reach % 8 : U32)
  +low_wanted = U32.is_le(8,reach)
  +low = (Tiled.cell_of(position,other,(above - 1 : U32),part_groups,others) + 0 * 0 : U32)
  +high_wanted = U32.is_lt(0,shift) && U32.is_lt(above,part_groups)
  +high = (Tiled.cell_of(position,other,above,part_groups,others) + 0 * 0 : U32)
  +middle = read_step(low_wanted,low,part)
  +local = (Tiled.cell_of(position,other,group,groups,others) - first : U32)
  written_length(local,Tiled.lanes_between(start,(start + width : U32),(8 * group : U32),Tiled.joined(shift,read_value(low_wanted,low,part),read_value(high_wanted,high,middle)),
    Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),local)),output)

# A buffer written at a cell holds the tree written there.
def written_tree(+index: U32,+value: Tile.Tile8,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,index,value,Buffer.pack(~Tile.Tile8,output)))) ==
    Tree.write(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),index,value) : Tree.Tree<Tile.Tile8>}:
  +written = Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,index,value,Buffer.pack(~Tile.Tile8,output)))
  +same = written_content(index,value,output)
  %Buffer.content_tree_of(~Tile.Tile8,output) :
    {Buffer.tree(~Tile.Tile8,written) == Tree.write(Tile.Tile8,_,index,value) : Tree.Tree<Tile.Tile8>}
  %Buffer.content_tree_of(~Tile.Tile8,written) :
    {_ == Tree.write(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),index,value) : Tree.Tree<Tile.Tile8>}
  Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8),Buffer.content(~Tile.Tile8,written),
    Buffer.Content{Tree.write(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),index,value),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output))},
    same)

def tile_pick_when(flag: Bool,+first: Tile.Tile8,+second: Tile.Tile8,+other: Tile.Tile8,same: @+hit: {flag == True{} : Bool} -> {first == second : Tile.Tile8}) ->
  {Bool.pick(Tile.Tile8,flag,first,other) == Bool.pick(Tile.Tile8,flag,second,other) : Tile.Tile8}:
  match flag:
    case True{}: same({==})
    case False{}: {==}

# A target piece buffer balanced at a depth, with room for count cells.
type OutputShape<-depth: Nat,-count: U32,-output: Buffer.Model<Tile.Tile8>> is Data:
  OutputShape{balanced: Shape.Balanced(Tile.Tile8,depth,Buffer.tree(~Tile.Tile8,output)),limit: {Nat.is_le(depth,30n) == True{} : Bool},
    room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)))) == True{} : Bool},
    sized: {Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)) == U32.to_nat(count) : Nat}}

def word_below(+word: U32,+count: U32,+bound: U32,+inside: {Nat.is_lt(U32.to_nat(word),U32.to_nat(count)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(count),U32.to_nat(bound)) == True{} : Bool}) -> {U32.is_lt(word,bound) == True{} : Bool}:
  Equal.trans(Bool,U32.is_lt(word,bound),Nat.is_lt(U32.to_nat(word),U32.to_nat(bound)),True{},Comparison.lt(word,bound),
    Bounds.nat_trans(U32.to_nat(word),U32.to_nat(count),U32.to_nat(bound),inside,room))

# The cell the part reaches at an other index and a group, the part read at
# its reach: the reads keep the part pieces and give the part cells.
def cell_part(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,
  +position: Grid.Position,+first: U32,+group: U32,+other: U32,+part: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,+stack: Held.Stack<Tile.Tile8>,
  +facts: PartSorted<stack>,+kept: PartKept<PContext,pvalue,pcontext,stack,part>,+reads: ReadsHeld(Held.spans(~Tile.Tile8,stack),start,part_groups,others,position,group,other)) ->
  Held.Both<PartKept<PContext,pvalue,pcontext,stack,state_part(cell_model(start,width,groups,part_groups,others,position,first,group,other,LaneState{part,output}))>,
    {Buffer.tree(~Tile.Tile8,state_output(cell_model(start,width,groups,part_groups,others,position,first,group,other,LaneState{part,output}))) ==
      Tree.write(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),(Tiled.cell_of(position,other,group,groups,others) - first : U32),
        touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,
          Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),(Tiled.cell_of(position,other,group,groups,others) - first : U32)))) : Tree.Tree<Tile.Tile8>}>:
  Held.Both{+low_held,+high_held} = reads
  +reach = (8 * group + 8 - start : U32)
  +above = (reach / 8 : U32)
  +shift = (reach % 8 : U32)
  +low_wanted = U32.is_le(8,reach)
  +low = (Tiled.cell_of(position,other,(above - 1 : U32),part_groups,others) + 0 * 0 : U32)
  +high_wanted = U32.is_lt(0,shift) && U32.is_lt(above,part_groups)
  +high = (Tiled.cell_of(position,other,above,part_groups,others) + 0 * 0 : U32)
  +middle = read_step(low_wanted,low,part)
  +local = (Tiled.cell_of(position,other,group,groups,others) - first : U32)
  +kept_cell = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),local)
  +low_read = read_kept(~PContext,~pvalue,pcontext,low_wanted,low,part,stack,facts,kept,low_held)
  +middle_kept = Held.first_of(PartKept<PContext,pvalue,pcontext,stack,middle>,{read_value(low_wanted,low,part) == part_cell(~PContext,~pvalue,pcontext,low_wanted,low) : Tile.Tile8},low_read)
  +low_same = Held.second_of(PartKept<PContext,pvalue,pcontext,stack,middle>,{read_value(low_wanted,low,part) == part_cell(~PContext,~pvalue,pcontext,low_wanted,low) : Tile.Tile8},low_read)
  +high_read = read_kept(~PContext,~pvalue,pcontext,high_wanted,high,middle,stack,facts,middle_kept,high_held)
  +after_kept = Held.first_of(PartKept<PContext,pvalue,pcontext,stack,read_step(high_wanted,high,middle)>,
    {read_value(high_wanted,high,middle) == part_cell(~PContext,~pvalue,pcontext,high_wanted,high) : Tile.Tile8},high_read)
  +high_same = Held.second_of(PartKept<PContext,pvalue,pcontext,stack,read_step(high_wanted,high,middle)>,
    {read_value(high_wanted,high,middle) == part_cell(~PContext,~pvalue,pcontext,high_wanted,high) : Tile.Tile8},high_read)
  +value = Tiled.lanes_between(start,(start + width : U32),(8 * group : U32),Tiled.joined(shift,read_value(low_wanted,low,part),read_value(high_wanted,high,middle)),kept_cell)
  +value_same = {%Equal.sym(Tile.Tile8,read_value(low_wanted,low,part),part_cell(~PContext,~pvalue,pcontext,low_wanted,low),low_same) :
      {Tiled.lanes_between(start,(start + width : U32),(8 * group : U32),Tiled.joined(shift,_,read_value(high_wanted,high,middle)),kept_cell) ==
        touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,kept_cell) : Tile.Tile8}
    %Equal.sym(Tile.Tile8,read_value(high_wanted,high,middle),part_cell(~PContext,~pvalue,pcontext,high_wanted,high),high_same) :
      {Tiled.lanes_between(start,(start + width : U32),(8 * group : U32),Tiled.joined(shift,part_cell(~PContext,~pvalue,pcontext,low_wanted,low),_),kept_cell) ==
        touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,kept_cell) : Tile.Tile8}
    {==} : {value == touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,kept_cell) : Tile.Tile8}}
  Held.Both{after_kept,
    Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,local,value,Buffer.pack(~Tile.Tile8,output)))),
      Tree.write(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),local,value),
      Tree.write(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),local,touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,kept_cell)),
      written_tree(local,value,output),
      Equal.cong(Tile.Tile8,Tree.Tree<Tile.Tile8>,cell => Tree.write(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),local,cell),value,
        touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,kept_cell),value_same))}

# After a merged cell the target piece keeps its shape, and a cell holds the
# merged value where the write went, what it held elsewhere.
def cell_query(~value: Tile.Tile8 -> Tile.Tile8,+depth: Nat,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},+local: U32,
  +local_inside: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},+query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Tree.write(Tile.Tile8,tree,local,value(Tree.read(Tile.Tile8,tree,local))),query) ==
    Bool.pick(Tile.Tile8,U32.is_eq(local,query),value(Tree.read(Tile.Tile8,tree,query)),Tree.read(Tile.Tile8,tree,query)) : Tile.Tile8}:
  +capacity = Tree.capacity(Tile.Tile8,tree)
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Tree.write(Tile.Tile8,tree,local,value(Tree.read(Tile.Tile8,tree,local))),query),
    Bool.pick(Tile.Tile8,U32.is_eq(local,query),value(Tree.read(Tile.Tile8,tree,local)),Tree.read(Tile.Tile8,tree,query)),
    Bool.pick(Tile.Tile8,U32.is_eq(local,query),value(Tree.read(Tile.Tile8,tree,query)),Tree.read(Tile.Tile8,tree,query)),
    Block.write_read(depth,tree,balanced,limit,local,value(Tree.read(Tile.Tile8,tree,local)),query,word_below(local,count,capacity,local_inside,room),
      word_below(query,count,capacity,query_inside,room)),
    tile_pick_when(U32.is_eq(local,query),value(Tree.read(Tile.Tile8,tree,local)),value(Tree.read(Tile.Tile8,tree,query)),Tree.read(Tile.Tile8,tree,query),
      hit => Equal.cong(U32,Tile.Tile8,cell => value(Tree.read(Tile.Tile8,tree,cell)),local,query,Arithmetic.equal_of_is_eq(local,query,hit))))

# ---- What the cells of a group leave ----

# A cell inside the target piece lies at a local index below its count.
def local_below(+first: U32,+count: U32,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
  +inside: {U32.is_le(first,index) && U32.is_lt(index,(first + count : U32)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat((index - first : U32)),U32.to_nat(count)) == True{} : Bool}:
  +after = Equal.trans(Bool,Nat.is_le(U32.to_nat(first),U32.to_nat(index)),U32.is_le(first,index),True{},
    Equal.sym(Bool,U32.is_le(first,index),Nat.is_le(U32.to_nat(first),U32.to_nat(index)),Comparison.le(first,index)),
    Logic.left(U32.is_le(first,index),U32.is_lt(index,(first + count : U32)),inside))
  +below_word = Equal.trans(Bool,Nat.is_lt(U32.to_nat(index),U32.to_nat((first + count : U32))),U32.is_lt(index,(first + count : U32)),True{},
    Equal.sym(Bool,U32.is_lt(index,(first + count : U32)),Nat.is_lt(U32.to_nat(index),U32.to_nat((first + count : U32))),Comparison.lt(index,(first + count : U32))),
    Logic.right(U32.is_le(first,index),U32.is_lt(index,(first + count : U32)),inside))
  +sum = Arithmetic.add(first,count,cap,cap_ok,room)
  +below = {%sum : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool} below_word : {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}}
  +index_room = Order.transitive(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),
    Bounds.strict_le(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count)),below),room)
  +difference = Arithmetic.difference_exact(cap,cap_ok,index,first,U32.to_nat(index),U32.to_nat(first),{==},{==},after,index_room)
  %Equal.sym(Nat,U32.to_nat((index - first : U32)),Nat.sub(U32.to_nat(index),U32.to_nat(first)),difference) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool}
  Order.sub_lt(U32.to_nat(index),U32.to_nat(first),U32.to_nat(count),after,below)

# The part cells the cells of a group from other on read, for every cell
# inside the target piece.
def OthersHeld(fuel: Nat,+spans: Held.Spans,+start: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,
  +other: U32) -> Data:
  match fuel:
    case 0n: Unit
    case 1n+rest:
      Held.Both<Logic.When(U32.is_le(first,Tiled.cell_of(position,other,group,groups,others)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,others),(first + count : U32)),
          ReadsHeld(spans,start,part_groups,others,position,group,other)),
        OthersHeld(rest,spans,start,groups,part_groups,others,position,first,count,group,(other + 1 : U32))>

# What a query cell holds after the cells of a group from other on: each
# merged over what the cell held, where the cell is the query.
def others_expected(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,
  +others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+other: U32,+query: U32,+held: Tile.Tile8) -> Tile.Tile8:
  match fuel:
    case 0n: held
    case 1n+rest:
      +index = Tiled.cell_of(position,other,group,groups,others)
      others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,
        Bool.pick(Tile.Tile8,(U32.is_le(first,index) && U32.is_lt(index,(first + count : U32))) && U32.is_eq((index - first : U32),query),
          touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,held),held))

# The part pieces kept, the target piece of its shape, and a query cell.
def StateFacts(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+stack: Held.Stack<Tile.Tile8>,+depth: Nat,+count: U32,+query: U32,
  +state: LaneState,+expected: Tile.Tile8) -> Data:
  Held.Both<PartKept<PContext,pvalue,pcontext,stack,state_part(state)>,
    Held.Both<OutputShape<depth,count,state_output(state)>,{Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(state)),query) == expected : Tile.Tile8}>>

def facts_kept(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+stack: Held.Stack<Tile.Tile8>,+depth: Nat,+count: U32,+query: U32,
  +state: LaneState,+expected: Tile.Tile8,+facts: StateFacts(~PContext,~pvalue,pcontext,stack,depth,count,query,state,expected)) ->
  PartKept<PContext,pvalue,pcontext,stack,state_part(state)>:
  Held.Both{+kept,+more} = facts
  kept

def facts_shape(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+stack: Held.Stack<Tile.Tile8>,+depth: Nat,+count: U32,+query: U32,
  +state: LaneState,+expected: Tile.Tile8,+facts: StateFacts(~PContext,~pvalue,pcontext,stack,depth,count,query,state,expected)) ->
  OutputShape<depth,count,state_output(state)>:
  Held.Both{+kept,+more} = facts
  Held.Both{+shape,+read} = more
  shape

def facts_read(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+stack: Held.Stack<Tile.Tile8>,+depth: Nat,+count: U32,+query: U32,
  +state: LaneState,+expected: Tile.Tile8,+facts: StateFacts(~PContext,~pvalue,pcontext,stack,depth,count,query,state,expected)) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(state)),query) == expected : Tile.Tile8}:
  Held.Both{+kept,+more} = facts
  Held.Both{+shape,+read} = more
  read

# A cell of a group merged when it lies inside the target piece.
def when_facts(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,inside: Bool,+start: U32,+width: U32,+groups: U32,+part_groups: U32,
  +others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+other: U32,+part: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +stack: Held.Stack<Tile.Tile8>,+facts: PartSorted<stack>,+kept: PartKept<PContext,pvalue,pcontext,stack,part>,+depth: Nat,+shape: OutputShape<depth,count,output>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
  +test: {U32.is_le(first,Tiled.cell_of(position,other,group,groups,others)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,others),(first + count : U32)) == inside : Bool},
  +held: Logic.When(inside,ReadsHeld(Held.spans(~Tile.Tile8,stack),start,part_groups,others,position,group,other)),
  +query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}) ->
  StateFacts(~PContext,~pvalue,pcontext,stack,depth,count,query,when_model(inside,start,width,groups,part_groups,others,position,first,group,other,LaneState{part,output}),
    Bool.pick(Tile.Tile8,inside && U32.is_eq((Tiled.cell_of(position,other,group,groups,others) - first : U32),query),
      touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),query)),
      Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),query))):
  match inside:
    case True{}:
      OutputShape{+balanced,+limit,+fits,+sized} = shape
      +local = (Tiled.cell_of(position,other,group,groups,others) - first : U32)
      +tree = Buffer.tree(~Tile.Tile8,output)
      +next = cell_model(start,width,groups,part_groups,others,position,first,group,other,LaneState{part,output})
      +value = touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,Tree.read(Tile.Tile8,tree,local))
      +cell = cell_part(~PContext,~pvalue,pcontext,start,width,groups,part_groups,others,position,first,group,other,part,output,stack,facts,kept,held)
      +next_kept = Held.first_of(PartKept<PContext,pvalue,pcontext,stack,state_part(next)>,
        {Buffer.tree(~Tile.Tile8,state_output(next)) == Tree.write(Tile.Tile8,tree,local,value) : Tree.Tree<Tile.Tile8>},cell)
      +same = Held.second_of(PartKept<PContext,pvalue,pcontext,stack,state_part(next)>,
        {Buffer.tree(~Tile.Tile8,state_output(next)) == Tree.write(Tile.Tile8,tree,local,value) : Tree.Tree<Tile.Tile8>},cell)
      +back = Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,state_output(next)),Tree.write(Tile.Tile8,tree,local,value),same)
      +local_inside = local_below(first,count,Tiled.cell_of(position,other,group,groups,others),cap,cap_ok,room,test)
      Held.Both{next_kept,
        Held.Both{OutputShape{{%back : Shape.Balanced(Tile.Tile8,depth,_)
              Shape.store(Tile.Tile8,depth,tree,Tree.capacity(Tile.Tile8,tree),U32.and(local,U32.sub(Tree.capacity(Tile.Tile8,tree),1)),value,balanced) :
              Shape.Balanced(Tile.Tile8,depth,Buffer.tree(~Tile.Tile8,state_output(next)))},
            limit,
            {%back : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,_))) == True{} : Bool}
              %Equal.sym(U32,Tree.capacity(Tile.Tile8,Tree.write(Tile.Tile8,tree,local,value)),Tree.capacity(Tile.Tile8,tree),StorageReadback.write_capacity(~Tile.Tile8,tree,local,value)) :
                {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
              fits : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(next))))) == True{} : Bool}},
            Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,state_output(next))),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),
              U32.to_nat(count),cell_sized(start,width,groups,part_groups,others,position,first,group,other,part,output),sized)},
          {%back : {Tree.read(Tile.Tile8,_,query) ==
                Bool.pick(Tile.Tile8,U32.is_eq(local,query),touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,Tree.read(Tile.Tile8,tree,query)),
                  Tree.read(Tile.Tile8,tree,query)) : Tile.Tile8}
            cell_query(~(kept_cell => touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,kept_cell)),depth,count,tree,balanced,limit,fits,
              local,local_inside,query,query_inside) :
            {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(next)),query) ==
              Bool.pick(Tile.Tile8,U32.is_eq(local,query),touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,Tree.read(Tile.Tile8,tree,query)),
                Tree.read(Tile.Tile8,tree,query)) : Tile.Tile8}}}}
    case False{}:
      Held.Both{kept,Held.Both{shape,{==}}}

# The cells of a group from other on, each merged when it lies inside the
# target piece.
def others_facts(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,
  +others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+other: U32,+state: LaneState,+stack: Held.Stack<Tile.Tile8>,+facts: PartSorted<stack>,
  +kept: PartKept<PContext,pvalue,pcontext,stack,state_part(state)>,+depth: Nat,+shape: OutputShape<depth,count,state_output(state)>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
  +held: OthersHeld(fuel,Held.spans(~Tile.Tile8,stack),start,groups,part_groups,others,position,first,count,group,other),
  +query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}) ->
  StateFacts(~PContext,~pvalue,pcontext,stack,depth,count,query,others_model(fuel,start,width,groups,part_groups,others,position,first,count,group,other,state),
    others_expected(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,other,query,
      Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(state)),query))):
  match fuel:
    case 0n:
      Held.Both{kept,Held.Both{shape,{==}}}
    case 1n+ +rest:
      LaneState{+part,+output} = state
      Held.Both{+here,+later} = held
      +index = Tiled.cell_of(position,other,group,groups,others)
      +inside = U32.is_le(first,index) && U32.is_lt(index,(first + count : U32))
      +before = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),query)
      +merged = Bool.pick(Tile.Tile8,inside && U32.is_eq((index - first : U32),query),
        touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,before),before)
      +next = when_model(inside,start,width,groups,part_groups,others,position,first,group,other,LaneState{part,output})
      +step = when_facts(~PContext,~pvalue,pcontext,inside,start,width,groups,part_groups,others,position,first,count,group,other,part,output,stack,facts,kept,depth,shape,
        cap,cap_ok,room,{==},here,query,query_inside)
      +after = others_facts(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),next,stack,facts,
        facts_kept(~PContext,~pvalue,pcontext,stack,depth,count,query,next,merged,step),depth,facts_shape(~PContext,~pvalue,pcontext,stack,depth,count,query,next,merged,step),
        cap,cap_ok,room,later,query,query_inside)
      +last = others_model(rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),next)
      +reached = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(next)),query)
      Held.Both{facts_kept(~PContext,~pvalue,pcontext,stack,depth,count,query,last,
          others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,reached),after),
        Held.Both{facts_shape(~PContext,~pvalue,pcontext,stack,depth,count,query,last,
            others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,reached),after),
          Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(last)),query),
            others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,reached),
            others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,merged),
            facts_read(~PContext,~pvalue,pcontext,stack,depth,count,query,last,
              others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,reached),after),
            Equal.cong(Tile.Tile8,Tile.Tile8,cell => others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,
                (other + 1 : U32),query,cell),reached,merged,facts_read(~PContext,~pvalue,pcontext,stack,depth,count,query,next,merged,step)))}}

# ---- What the groups leave ----

# The part cells the cells of a group a target piece holds read.
def RangeHeld(+found: Tiled.Others,+spans: Held.Spans,+start: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,
  +group: U32) -> Data:
  Tiled.Others{+low,+high} = found
  OthersHeld(U32.to_nat((U32.max(high,low) - low : U32)),spans,start,groups,part_groups,others,position,first,count,group,low)

# The part cells the groups from group on read.
def GroupsHeld(fuel: Nat,+spans: Held.Spans,+start: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32) -> Data:
  match fuel:
    case 0n: Unit
    case 1n+rest:
      Held.Both<RangeHeld(Tiled.others_of(position,groups,others,first,count,group),spans,start,groups,part_groups,others,position,first,count,group),
        GroupsHeld(rest,spans,start,groups,part_groups,others,position,first,count,(group + 1 : U32))>

def range_expected(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+found: Tiled.Others,+start: U32,+width: U32,+groups: U32,+part_groups: U32,
  +others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+query: U32,+held: Tile.Tile8) -> Tile.Tile8:
  Tiled.Others{+low,+high} = found
  others_expected(~PContext,~pvalue,pcontext,U32.to_nat((U32.max(high,low) - low : U32)),start,width,groups,part_groups,others,position,first,count,group,low,query,held)

# What a query cell holds after the groups from group on.
def groups_expected(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,
  +others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+query: U32,+held: Tile.Tile8) -> Tile.Tile8:
  match fuel:
    case 0n: held
    case 1n+rest:
      groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,
        range_expected(~PContext,~pvalue,pcontext,Tiled.others_of(position,groups,others,first,count,group),start,width,groups,part_groups,others,position,first,count,group,
          query,held))

def range_facts(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+found: Tiled.Others,+start: U32,+width: U32,+groups: U32,+part_groups: U32,
  +others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+state: LaneState,+stack: Held.Stack<Tile.Tile8>,+facts: PartSorted<stack>,
  +kept: PartKept<PContext,pvalue,pcontext,stack,state_part(state)>,+depth: Nat,+shape: OutputShape<depth,count,state_output(state)>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
  +held: RangeHeld(found,Held.spans(~Tile.Tile8,stack),start,groups,part_groups,others,position,first,count,group),
  +query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}) ->
  StateFacts(~PContext,~pvalue,pcontext,stack,depth,count,query,range_model(found,start,width,groups,part_groups,others,position,first,count,group,state),
    range_expected(~PContext,~pvalue,pcontext,found,start,width,groups,part_groups,others,position,first,count,group,query,
      Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(state)),query))):
  Tiled.Others{+low,+high} = found
  others_facts(~PContext,~pvalue,pcontext,U32.to_nat((U32.max(high,low) - low : U32)),start,width,groups,part_groups,others,position,first,count,group,low,state,stack,facts,
    kept,depth,shape,cap,cap_ok,room,held,query,query_inside)

# The groups from group on, each cell of the target piece merged.
def groups_facts(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,
  +others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+state: LaneState,+stack: Held.Stack<Tile.Tile8>,+facts: PartSorted<stack>,
  +kept: PartKept<PContext,pvalue,pcontext,stack,state_part(state)>,+depth: Nat,+shape: OutputShape<depth,count,state_output(state)>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
  +held: GroupsHeld(fuel,Held.spans(~Tile.Tile8,stack),start,groups,part_groups,others,position,first,count,group),
  +query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}) ->
  StateFacts(~PContext,~pvalue,pcontext,stack,depth,count,query,groups_model(fuel,start,width,groups,part_groups,others,position,first,count,group,state),
    groups_expected(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,query,
      Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(state)),query))):
  match fuel:
    case 0n:
      Held.Both{kept,Held.Both{shape,{==}}}
    case 1n+ +rest:
      Held.Both{+here,+later} = held
      +found = Tiled.others_of(position,groups,others,first,count,group)
      +next = range_model(found,start,width,groups,part_groups,others,position,first,count,group,state)
      +before = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(state)),query)
      +ranged = range_expected(~PContext,~pvalue,pcontext,found,start,width,groups,part_groups,others,position,first,count,group,query,before)
      +step = range_facts(~PContext,~pvalue,pcontext,found,start,width,groups,part_groups,others,position,first,count,group,state,stack,facts,kept,depth,shape,
        cap,cap_ok,room,here,query,query_inside)
      +after = groups_facts(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),next,stack,facts,
        facts_kept(~PContext,~pvalue,pcontext,stack,depth,count,query,next,ranged,step),depth,facts_shape(~PContext,~pvalue,pcontext,stack,depth,count,query,next,ranged,step),
        cap,cap_ok,room,later,query,query_inside)
      +last = groups_model(rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),next)
      +reached = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(next)),query)
      +expected = groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,reached)
      Held.Both{facts_kept(~PContext,~pvalue,pcontext,stack,depth,count,query,last,expected,after),
        Held.Both{facts_shape(~PContext,~pvalue,pcontext,stack,depth,count,query,last,expected,after),
          Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,state_output(last)),query),expected,
            groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,ranged),
            facts_read(~PContext,~pvalue,pcontext,stack,depth,count,query,last,expected,after),
            Equal.cong(Tile.Tile8,Tile.Tile8,cell => groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,
                (group + 1 : U32),query,cell),reached,ranged,facts_read(~PContext,~pvalue,pcontext,stack,depth,count,query,next,ranged,step)))}}

# ---- The model of the gather ----

# The groups a part of width indices from start reaches.
def reached_groups(+start: U32,+width: U32) -> U32:
  Bool.pick(U32,U32.is_zero(width),0,((start + width - 1) / 8 - start / 8 + 1 : U32))

def written_state(+write: Tiled.Write,+first: U32,+count: U32,+state: LaneState) -> LaneState:
  Tiled.Write{+start,+width,+groups,+part_groups,+others,+position} = write
  groups_model(U32.to_nat(reached_groups(start,width)),start,width,groups,part_groups,others,position,first,count,(start / 8 : U32),state)

def parts_model(+write: Tiled.Write,+first: U32,+count: U32,+below: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+others: Moved.Holdings<Tile.Tile8>) ->
  Moved.Gathered<Tile.Tile8>:
  match others:
    case Moved.Holds{+part,+rest}:
      +state = written_state(write,first,count,LaneState{part,buffer})
      Moved.Gathered{Moved.Holds{Held.Holding{Held.Bottom{},below},Moved.Holds{state_part(state),rest}},state_output(state)}
    case Moved.Unheld{}:
      Moved.Gathered{Moved.Holds{Held.Holding{Held.Bottom{},below},Moved.Unheld{}},buffer}

def ahead_model(+write: Tiled.Write,+first: U32,+count: U32,+fill: Tile.Tile8,+ahead: Held.Stack<Tile.Tile8>,+others: Moved.Holdings<Tile.Tile8>) -> Moved.Gathered<Tile.Tile8>:
  match ahead:
    case Held.Held{+buffer,+start,+span,+below}: parts_model(write,first,count,below,buffer,others)
    case Held.Bottom{}: Moved.Gathered{others,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))}

def holding_model(+write: Tiled.Write,+first: U32,+count: U32,+fill: Tile.Tile8,+holding: Held.Holding<Tile.Tile8>,+others: Moved.Holdings<Tile.Tile8>) ->
  Moved.Gathered<Tile.Tile8>:
  Held.Holding{+behind,+ahead} = holding
  ahead_model(write,first,count,fill,ahead,others)

# A lane write gathers, from a target holding whose head piece is the target
# piece, that piece with every cell the part reaches merged.
def lanes_model(+write: Tiled.Write,+components: Tasks.Components,+first: U32,+count: U32,+fill: Tile.Tile8,+holdings: Moved.Holdings<Tile.Tile8>) -> Moved.Gathered<Tile.Tile8>:
  match holdings:
    case Moved.Holds{+holding,+others}: holding_model(write,first,count,fill,holding,others)
    case Moved.Unheld{}: Moved.Gathered{Moved.Unheld{},Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))}

# The model of a lane write from a head piece that is the target piece.
def lanes_gathered(+write: Tiled.Write,+components: Tasks.Components,+fill: Tile.Tile8,+buffer: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,
  +below: Held.Stack<Tile.Tile8>,+part: Held.Holding<Tile.Tile8>,+rest: Moved.Holdings<Tile.Tile8>,+ends: {U32.is_le((first + count : U32),first) == False{} : Bool}) ->
  {Tiled.write_gather(write,components,first,count,fill,Moved.pack_holdings(~Tile.Tile8,Moved.Holds{Held.Holding{Held.Bottom{},Held.Held{buffer,first,count,below}},Moved.Holds{part,rest}})) ==
    (Moved.pack_holdings(~Tile.Tile8,Moved.gathered_holdings(~Tile.Tile8,lanes_model(write,components,first,count,fill,
        Moved.Holds{Held.Holding{Held.Bottom{},Held.Held{buffer,first,count,below}},Moved.Holds{part,rest}}))),
      Buffer.pack(~Tile.Tile8,Moved.gathered_output(~Tile.Tile8,lanes_model(write,components,first,count,fill,
        Moved.Holds{Held.Holding{Held.Bottom{},Held.Held{buffer,first,count,below}},Moved.Holds{part,rest}})))) : Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  Tiled.Write{+start,+width,+groups,+part_groups,+others,+position} = write
  +state = groups_model(U32.to_nat(reached_groups(start,width)),start,width,groups,part_groups,others,position,first,count,(start / 8 : U32),LaneState{part,buffer})
  %Equal.sym(Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,
      Tiled.write_gather(Tiled.Write{start,width,groups,part_groups,others,position},components,first,count,fill,
        Tensors.Holds{Held.pack(~Tile.Tile8,Held.Holding{Held.Bottom{},Held.Held{buffer,first,count,below}}),Moved.pack_holdings(~Tile.Tile8,Moved.Holds{part,rest})}),
      Tiled.write_part(Moved.pack_holdings(~Tile.Tile8,Moved.Holds{part,rest}),Tiled.Write{start,width,groups,part_groups,others,position},first,count,
        (Move.Holding{Move.Bottom{},Held.pack_stack(~Tile.Tile8,below)},Buffer.pack(~Tile.Tile8,buffer))),
      head_gathered(Tiled.Write{start,width,groups,part_groups,others,position},components,fill,buffer,first,count,below,Moved.pack_holdings(~Tile.Tile8,Moved.Holds{part,rest}),ends)) :
    {_ == (Tensors.Holds{Move.Holding{Move.Bottom{},Held.pack_stack(~Tile.Tile8,below)},Tensors.Holds{Held.pack(~Tile.Tile8,state_part(state)),Moved.pack_holdings(~Tile.Tile8,rest)}},
        Buffer.pack(~Tile.Tile8,state_output(state))) : Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  %Equal.sym(Tiled.Writing,
      Tiled.touched_groups(U32.to_nat(reached_groups(start,width)),Tiled.Write{start,width,groups,part_groups,others,position},first,count,(start / 8 : U32),
        pack_state(Held.Holding{Held.Bottom{},below},LaneState{part,buffer})),
      pack_state(Held.Holding{Held.Bottom{},below},state),
      groups_packed(Held.Holding{Held.Bottom{},below},U32.to_nat(reached_groups(start,width)),start,width,groups,part_groups,others,position,
        first,count,(start / 8 : U32),LaneState{part,buffer})) :
    {Tiled.write_done(Moved.pack_holdings(~Tile.Tile8,rest),_) ==
      (Tensors.Holds{Move.Holding{Move.Bottom{},Held.pack_stack(~Tile.Tile8,below)},Tensors.Holds{Held.pack(~Tile.Tile8,state_part(state)),Moved.pack_holdings(~Tile.Tile8,rest)}},
        Buffer.pack(~Tile.Tile8,state_output(state))) : Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  {==}

# ---- The cell a query names ----

# The cells of a lane write: groups by others, both positive, inside the word,
# which a wider word twice holds.
type CellGrid<-groups: U32,-others: U32,-cap: U32> is Data:
  CellGrid{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},
    others_positive: {Nat.is_lt(0n,U32.to_nat(others)) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(U32.to_nat(groups),U32.to_nat(others)),U32.to_nat(cap)) == True{} : Bool},wide: U32,
    wide_ok: {U32.is_le(wide,2147483648) == True{} : Bool},doubled: {Nat.is_le(Nat.double(U32.to_nat(cap)),U32.to_nat(wide)) == True{} : Bool}}

def cap_fits(+groups: U32,+others: U32,+cap: U32,+grid: CellGrid<groups,others,cap>) -> {U32.is_le(cap,2147483648) == True{} : Bool}:
  CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled} = grid
  cap_ok

# The other index and the group of a cell.
def cell_other(+position: Grid.Position,+groups: U32,+others: U32,+cell: Nat) -> Nat:
  Representation.outer_at(position,U32.to_nat(others),U32.to_nat(groups),cell)

def cell_group(+position: Grid.Position,+groups: U32,+others: U32,+cell: Nat) -> Nat:
  Representation.group_at(position,U32.to_nat(others),U32.to_nat(groups),cell)

def word_by_position(position: Grid.Position,+groups: U32,+others: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+other: U32,+group: U32,
  +fits: {Nat.is_le(Representation.index_at(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Tiled.cell_of(position,other,group,groups,others)) ==
    Representation.index_at(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group)) : Nat}:
  match position:
    case Grid.Innermost{}: Arithmetic.multiply_add_exact(other,groups,group,cap,cap_ok,fits)
    case Grid.Outermost{}: Arithmetic.multiply_add_exact(group,others,other,cap,cap_ok,fits)

# The cell of an other index and a group below theirs lies inside the word.
def index_fits(+position: Grid.Position,+groups: U32,+others: U32,+cap: U32,+grid: CellGrid<groups,others,cap>,+other: U32,+group: U32,
  +other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(others)) == True{} : Bool},+group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool}) ->
  {Nat.is_le(Representation.index_at(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group)),U32.to_nat(cap)) == True{} : Bool}:
  CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled} = grid
  +index = Representation.index_at(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group))
  Order.transitive(index,Nat.mul(U32.to_nat(groups),U32.to_nat(others)),U32.to_nat(cap),
    Bounds.strict_le(index,Nat.mul(U32.to_nat(groups),U32.to_nat(others)),
      Representation.index_inside(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group),other_inside,group_inside)),room)

# The cell of an other index and a group, as a number.
def cell_word(+position: Grid.Position,+groups: U32,+others: U32,+cap: U32,+grid: CellGrid<groups,others,cap>,+other: U32,+group: U32,
  +other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(others)) == True{} : Bool},+group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool}) ->
  {U32.to_nat(Tiled.cell_of(position,other,group,groups,others)) ==
    Representation.index_at(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group)) : Nat}:
  word_by_position(position,groups,others,cap,cap_fits(groups,others,cap,grid),other,group,
    index_fits(position,groups,others,cap,grid,other,group,other_inside,group_inside))

# A cell is the cell of its other index and its group.
def cell_rebuilt(position: Grid.Position,+groups: U32,+others: U32,+cap: U32,+grid: CellGrid<groups,others,cap>,+cell: Nat) ->
  {Representation.index_at(position,U32.to_nat(others),U32.to_nat(groups),cell_other(position,groups,others,cell),cell_group(position,groups,others,cell)) == cell : Nat}:
  match position:
    case Grid.Innermost{}:
      CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled} = grid
      %Equal.sym(Nat,Representation.least_one(U32.to_nat(groups)),U32.to_nat(groups),Representation.least_positive(U32.to_nat(groups),groups_positive)) :
        {Nat.add(Nat.mul(Nat.div(cell,_),U32.to_nat(groups)),Nat.mod(cell,_)) == cell : Nat}
      Equal.sym(Nat,cell,Nat.add(Nat.mul(Nat.div(cell,U32.to_nat(groups)),U32.to_nat(groups)),Nat.mod(cell,U32.to_nat(groups))),
        DivisionOrder.quotient_exact(cell,U32.to_nat(groups),groups_positive))
    case Grid.Outermost{}:
      CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled} = grid
      %Equal.sym(Nat,Representation.least_one(U32.to_nat(others)),U32.to_nat(others),Representation.least_positive(U32.to_nat(others),others_positive)) :
        {Nat.add(Nat.mul(Nat.div(cell,_),U32.to_nat(others)),Nat.mod(cell,_)) == cell : Nat}
      Equal.sym(Nat,cell,Nat.add(Nat.mul(Nat.div(cell,U32.to_nat(others)),U32.to_nat(others)),Nat.mod(cell,U32.to_nat(others))),
        DivisionOrder.quotient_exact(cell,U32.to_nat(others),others_positive))

# A target piece of the cells.
type PieceIn<-groups: U32,-others: U32,-cap: U32,-first: U32,-count: U32> is Data:
  PieceIn{grid: CellGrid<groups,others,cap>,
    inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(groups),U32.to_nat(others))) == True{} : Bool}}

def piece_room(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool}:
  PieceIn{CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled},+inside} = piece
  Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(groups),U32.to_nat(others)),U32.to_nat(cap),inside,room)

def piece_grid(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>) -> CellGrid<groups,others,cap>:
  PieceIn{+grid,+inside} = piece
  grid

# The cell of the query of a target piece.
def query_cell(+first: U32,+query: U32) -> Nat:
  Nat.add(U32.to_nat(first),U32.to_nat(query))

def query_below(+first: U32,+count: U32,+query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.is_lt(query_cell(first,query),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}:
  %Equal.sym(Bool,Nat.is_lt(query_cell(first,query),Nat.add(U32.to_nat(first),U32.to_nat(count))),Nat.is_lt(U32.to_nat(query),U32.to_nat(count)),
      Order.lt_offset(U32.to_nat(first),U32.to_nat(query),U32.to_nat(count))) : {_ == True{} : Bool}
  query_inside

def query_cells(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>,+query: U32,
  +query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.is_lt(query_cell(first,query),Nat.mul(U32.to_nat(groups),U32.to_nat(others))) == True{} : Bool}:
  PieceIn{+grid,+inside} = piece
  Bounds.nat_trans(query_cell(first,query),Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(groups),U32.to_nat(others)),
    query_below(first,count,query,query_inside),inside)

# A natural equal to another gives a true test.
def nat_same(+left: Nat,+right: Nat,+same: {left == right : Nat}) -> {Nat.is_eq(left,right) == True{} : Bool}:
  %Equal.sym(Nat,left,right,same) : {Nat.is_eq(_,right) == True{} : Bool}
  Order.equal_reflexive(right)

# A cell of a target piece the runtime finds at the query is the query cell.
def query_found(+position: Grid.Position,+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +other: U32,+group: U32,+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(others)) == True{} : Bool},
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+query: U32,
  +found: {(U32.is_le(first,Tiled.cell_of(position,other,group,groups,others)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,others),(first + count : U32))) &&
      U32.is_eq((Tiled.cell_of(position,other,group,groups,others) - first : U32),query) == True{} : Bool}) ->
  {Nat.is_eq(cell_group(position,groups,others,query_cell(first,query)),U32.to_nat(group)) &&
    Nat.is_eq(cell_other(position,groups,others,query_cell(first,query)),U32.to_nat(other)) == True{} : Bool}:
  +grid = piece_grid(groups,others,cap,first,count,piece)
  +index = Tiled.cell_of(position,other,group,groups,others)
  +cell = query_cell(first,query)
  +named = Representation.index_at(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group))
  +inside = Logic.and_true_left(U32.is_le(first,index) && U32.is_lt(index,(first + count : U32)),U32.is_eq((index - first : U32),query),found)
  +after_word = Logic.and_true_left(U32.is_le(first,index),U32.is_lt(index,(first + count : U32)),inside)
  +after = Equal.trans(Bool,Nat.is_le(U32.to_nat(first),U32.to_nat(index)),U32.is_le(first,index),True{},
    Equal.sym(Bool,U32.is_le(first,index),Nat.is_le(U32.to_nat(first),U32.to_nat(index)),Comparison.le(first,index)),after_word)
  +same = Arithmetic.equal_of_is_eq((index - first : U32),query,Logic.and_true_right(U32.is_le(first,index) && U32.is_lt(index,(first + count : U32)),
    U32.is_eq((index - first : U32),query),found))
  +word = cell_word(position,groups,others,cap,grid,other,group,other_inside,group_inside)
  +index_room = {%Equal.sym(Nat,U32.to_nat(index),named,word) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    index_fits(position,groups,others,cap,grid,other,group,other_inside,group_inside) : {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool}}
  +difference = Arithmetic.difference_exact(cap,cap_fits(groups,others,cap,grid),index,first,U32.to_nat(index),U32.to_nat(first),{==},{==},after,index_room)
  +offset = Equal.trans(Nat,U32.to_nat(query),U32.to_nat((index - first : U32)),Nat.sub(U32.to_nat(index),U32.to_nat(first)),
    Equal.sym(Nat,U32.to_nat((index - first : U32)),U32.to_nat(query),Equal.cong(U32,Nat,U32.to_nat,(index - first : U32),query,same)),difference)
  +at_index = Equal.trans(Nat,cell,Nat.add(U32.to_nat(first),Nat.sub(U32.to_nat(index),U32.to_nat(first))),U32.to_nat(index),
    Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(first),value),U32.to_nat(query),Nat.sub(U32.to_nat(index),U32.to_nat(first)),offset),
    Order.restore_shift(U32.to_nat(first),U32.to_nat(index),after))
  +at_named = Equal.trans(Nat,cell,U32.to_nat(index),named,at_index,word)
  +back = Equal.sym(Nat,cell,named,at_named)
  Logic.and_true(Nat.is_eq(cell_group(position,groups,others,cell),U32.to_nat(group)),Nat.is_eq(cell_other(position,groups,others,cell),U32.to_nat(other)),
    nat_same(cell_group(position,groups,others,cell),U32.to_nat(group),
      {%back : {cell_group(position,groups,others,_) == U32.to_nat(group) : Nat}
        Representation.placed_group(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group),other_inside,group_inside) :
        {cell_group(position,groups,others,cell) == U32.to_nat(group) : Nat}}),
    nat_same(cell_other(position,groups,others,cell),U32.to_nat(other),
      {%back : {cell_other(position,groups,others,_) == U32.to_nat(other) : Nat}
        Representation.placed_outer(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group),other_inside,group_inside) :
        {cell_other(position,groups,others,cell) == U32.to_nat(other) : Nat}}))

# The cell of the other index and the group of the query cell is a cell of
# the target piece at the query.
def query_named(+position: Grid.Position,+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +other: U32,+group: U32,+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(others)) == True{} : Bool},
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+query: U32,
  +query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},
  +named: {Nat.is_eq(cell_group(position,groups,others,query_cell(first,query)),U32.to_nat(group)) &&
    Nat.is_eq(cell_other(position,groups,others,query_cell(first,query)),U32.to_nat(other)) == True{} : Bool}) ->
  {(U32.is_le(first,Tiled.cell_of(position,other,group,groups,others)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,others),(first + count : U32))) &&
      U32.is_eq((Tiled.cell_of(position,other,group,groups,others) - first : U32),query) == True{} : Bool}:
  +grid = piece_grid(groups,others,cap,first,count,piece)
  +cap_ok = cap_fits(groups,others,cap,grid)
  +index = Tiled.cell_of(position,other,group,groups,others)
  +cell = query_cell(first,query)
  +group_same = Order.equal_of_is_eq(cell_group(position,groups,others,cell),U32.to_nat(group),
    Logic.and_true_left(Nat.is_eq(cell_group(position,groups,others,cell),U32.to_nat(group)),Nat.is_eq(cell_other(position,groups,others,cell),U32.to_nat(other)),named))
  +other_same = Order.equal_of_is_eq(cell_other(position,groups,others,cell),U32.to_nat(other),
    Logic.and_true_right(Nat.is_eq(cell_group(position,groups,others,cell),U32.to_nat(group)),Nat.is_eq(cell_other(position,groups,others,cell),U32.to_nat(other)),named))
  +rebuilt = cell_rebuilt(position,groups,others,cap,grid,cell)
  +named_cell = {%group_same : {Representation.index_at(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),_) == cell : Nat}
    %other_same : {Representation.index_at(position,U32.to_nat(others),U32.to_nat(groups),_,cell_group(position,groups,others,cell)) == cell : Nat}
    rebuilt : {Representation.index_at(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group)) == cell : Nat}}
  +index_cell = Equal.trans(Nat,U32.to_nat(index),Representation.index_at(position,U32.to_nat(others),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group)),
    cell,cell_word(position,groups,others,cap,grid,other,group,other_inside,group_inside),named_cell)
  +back = Equal.sym(Nat,U32.to_nat(index),cell,index_cell)
  +after = {%back : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} Order.le_add(U32.to_nat(first),U32.to_nat(query)) :
    {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool}}
  +end_word = Equal.sym(Nat,U32.to_nat((first + count : U32)),Nat.add(U32.to_nat(first),U32.to_nat(count)),Arithmetic.add(first,count,cap,cap_ok,piece_room(groups,others,cap,first,count,piece)))
  +before = {%back : {Nat.is_lt(_,U32.to_nat((first + count : U32))) == True{} : Bool}
    %end_word : {Nat.is_lt(cell,_) == True{} : Bool}
    query_below(first,count,query,query_inside) : {Nat.is_lt(U32.to_nat(index),U32.to_nat((first + count : U32))) == True{} : Bool}}
  +index_room = Order.transitive(U32.to_nat(index),U32.to_nat((first + count : U32)),U32.to_nat(cap),Bounds.strict_le(U32.to_nat(index),U32.to_nat((first + count : U32)),before),
    {%end_word : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} piece_room(groups,others,cap,first,count,piece) :
      {Nat.is_le(U32.to_nat((first + count : U32)),U32.to_nat(cap)) == True{} : Bool}})
  +difference = Arithmetic.difference_exact(cap,cap_ok,index,first,U32.to_nat(index),U32.to_nat(first),{==},{==},after,index_room)
  +offset = Equal.trans(Nat,U32.to_nat((index - first : U32)),Nat.sub(U32.to_nat(index),U32.to_nat(first)),U32.to_nat(query),difference,
    %back : {Nat.sub(_,U32.to_nat(first)) == U32.to_nat(query) : Nat} Order.subtract_shift(U32.to_nat(first),U32.to_nat(query)))
  Logic.and_true(U32.is_le(first,index) && U32.is_lt(index,(first + count : U32)),U32.is_eq((index - first : U32),query),
    Logic.and_true(U32.is_le(first,index),U32.is_lt(index,(first + count : U32)),
      Equal.trans(Bool,U32.is_le(first,index),Nat.is_le(U32.to_nat(first),U32.to_nat(index)),True{},Comparison.le(first,index),after),
      Equal.trans(Bool,U32.is_lt(index,(first + count : U32)),Nat.is_lt(U32.to_nat(index),U32.to_nat((first + count : U32))),True{},Comparison.lt(index,(first + count : U32)),before)),
    Equal.trans(Bool,U32.is_eq((index - first : U32),query),Nat.is_eq(U32.to_nat((index - first : U32)),U32.to_nat(query)),True{},Comparison.eq((index - first : U32),query),
      nat_same(U32.to_nat((index - first : U32)),U32.to_nat(query),offset)))

# The runtime test of a cell of a target piece at a query holds exactly for
# the other index and the group of the query cell.
def cell_is_query(+position: Grid.Position,+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +other: U32,+group: U32,+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(others)) == True{} : Bool},
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+query: U32,
  +query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}) ->
  {(U32.is_le(first,Tiled.cell_of(position,other,group,groups,others)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,others),(first + count : U32))) &&
      U32.is_eq((Tiled.cell_of(position,other,group,groups,others) - first : U32),query) ==
    Nat.is_eq(cell_group(position,groups,others,query_cell(first,query)),U32.to_nat(group)) &&
      Nat.is_eq(cell_other(position,groups,others,query_cell(first,query)),U32.to_nat(other)) : Bool}:
  +index = Tiled.cell_of(position,other,group,groups,others)
  +cell = query_cell(first,query)
  Logic.equivalent((U32.is_le(first,index) && U32.is_lt(index,(first + count : U32))) && U32.is_eq((index - first : U32),query),
    Nat.is_eq(cell_group(position,groups,others,cell),U32.to_nat(group)) && Nat.is_eq(cell_other(position,groups,others,cell),U32.to_nat(other)),
    found => query_found(position,groups,others,cap,first,count,piece,other,group,other_inside,group_inside,query,found),
    named => query_named(position,groups,others,cap,first,count,piece,other,group,other_inside,group_inside,query,query_inside,named))

# ---- The cells a query cell meets ----

# The group and the other index of the query cell.
def query_group(+position: Grid.Position,+groups: U32,+others: U32,+first: U32,+query: U32) -> Nat:
  cell_group(position,groups,others,query_cell(first,query))

def query_other(+position: Grid.Position,+groups: U32,+others: U32,+first: U32,+query: U32) -> Nat:
  cell_other(position,groups,others,query_cell(first,query))

# The runtime test of the cell of a group at an other index against a query.
def cell_test(+position: Grid.Position,+groups: U32,+others: U32,+first: U32,+count: U32,+group: U32,+other: U32,+query: U32) -> Bool:
  +index = Tiled.cell_of(position,other,group,groups,others)
  (U32.is_le(first,index) && U32.is_lt(index,(first + count : U32))) && U32.is_eq((index - first : U32),query)

def tested(+position: Grid.Position,+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +other: U32,+group: U32,+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(others)) == True{} : Bool},
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+query: U32,
  +query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}) ->
  {cell_test(position,groups,others,first,count,group,other,query) == Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group)) && Nat.is_eq(query_other(position,groups,others,first,query),U32.to_nat(other)) : Bool}:
  cell_is_query(position,groups,others,cap,first,count,piece,other,group,other_inside,group_inside,query,query_inside)

def and_false_left(+first: Bool,+second: Bool,+none: {first == False{} : Bool}) -> {first && second == False{} : Bool}:
  %Equal.sym(Bool,first,False{},none) : {_ && second == False{} : Bool}
  {==}

def and_false_right(first: Bool,+second: Bool,+none: {second == False{} : Bool}) -> {first && second == False{} : Bool}:
  match first:
    case True{}: none
    case False{}: {==}

def le_apart_lt(+left: Nat,+right: Nat,+ordered: {Nat.is_le(left,right) == True{} : Bool},+apart: {Nat.is_eq(right,left) == False{} : Bool}) ->
  {Nat.is_lt(left,right) == True{} : Bool}:
  match left right:
    case 0n 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},apart)))
    case 0n 1n+right: {==}
    case 1n+left 0n: Empty.absurd({Nat.is_lt(1n+left,0n) == True{} : Bool},WordEncoding.false_true(ordered))
    case 1n+ +left 1n+ +right: le_apart_lt(left,right,ordered,apart)

def equal_le(+left: Nat,+right: Nat,+same: {Nat.is_eq(left,right) == True{} : Bool}) -> {Nat.is_le(left,right) == True{} : Bool}:
  %Equal.sym(Nat,left,right,Order.equal_of_is_eq(left,right,same)) : {Nat.is_le(_,right) == True{} : Bool}
  Order.reflexive(right)

def word_same(+word: U32,+value: Nat,+same: {Nat.is_eq(value,U32.to_nat(word)) == True{} : Bool}) -> {word == U32.from_nat(value) : U32}:
  Equal.trans(U32,word,U32.from_nat(U32.to_nat(word)),U32.from_nat(value),Equal.sym(U32,U32.from_nat(U32.to_nat(word)),word,Arithmetic.native_roundtrip(word)),
    Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(word),value,Equal.sym(Nat,value,U32.to_nat(word),Order.equal_of_is_eq(value,U32.to_nat(word),same))))

def pick_le(choice: Bool,+first: Nat,+second: Nat,+bound: Nat,+first_le: {Nat.is_le(first,bound) == True{} : Bool},
  +second_le: {Nat.is_le(second,bound) == True{} : Bool}) -> {Nat.is_le(Bool.pick(Nat,choice,first,second),bound) == True{} : Bool}:
  match choice:
    case True{}: first_le
    case False{}: second_le

def pick_ge(choice: Bool,+first: Nat,+second: Nat,+bound: Nat,+first_ge: {Nat.is_le(bound,first) == True{} : Bool},
  +second_ge: {Nat.is_le(bound,second) == True{} : Bool}) -> {Nat.is_le(bound,Bool.pick(Nat,choice,first,second)) == True{} : Bool}:
  match choice:
    case True{}: first_ge
    case False{}: second_ge

def pick_above(choice: Bool,+first: Nat,+second: Nat,+bound: Nat,+first_above: {Nat.is_lt(bound,first) == True{} : Bool},
  +second_above: {Nat.is_lt(bound,second) == True{} : Bool}) -> {Nat.is_lt(bound,Bool.pick(Nat,choice,first,second)) == True{} : Bool}:
  match choice:
    case True{}: first_above
    case False{}: second_above

def max_le(+left: Nat,+right: Nat,+bound: Nat,+left_le: {Nat.is_le(left,bound) == True{} : Bool},+right_le: {Nat.is_le(right,bound) == True{} : Bool}) ->
  {Nat.is_le(Nat.max(left,right),bound) == True{} : Bool}:
  %Equal.sym(Nat,Nat.max(left,right),Bool.pick(Nat,Nat.is_lt(left,right),right,left),NatExtrema.maximum_choice(left,right)) : {Nat.is_le(_,bound) == True{} : Bool}
  pick_le(Nat.is_lt(left,right),right,left,bound,right_le,left_le)

def min_ge(+left: Nat,+right: Nat,+bound: Nat,+left_ge: {Nat.is_le(bound,left) == True{} : Bool},+right_ge: {Nat.is_le(bound,right) == True{} : Bool}) ->
  {Nat.is_le(bound,Nat.min(left,right)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.min(left,right),Bool.pick(Nat,Nat.is_lt(left,right),left,right),NatExtrema.minimum_choice(left,right)) : {Nat.is_le(bound,_) == True{} : Bool}
  pick_ge(Nat.is_lt(left,right),left,right,bound,left_ge,right_ge)

def min_above(+left: Nat,+right: Nat,+bound: Nat,+left_above: {Nat.is_lt(bound,left) == True{} : Bool},+right_above: {Nat.is_lt(bound,right) == True{} : Bool}) ->
  {Nat.is_lt(bound,Nat.min(left,right)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.min(left,right),Bool.pick(Nat,Nat.is_lt(left,right),left,right),NatExtrema.minimum_choice(left,right)) : {Nat.is_lt(bound,_) == True{} : Bool}
  pick_above(Nat.is_lt(left,right),left,right,bound,left_above,right_above)

def word_nat_le(+left: U32,+right: U32,+ordered: {U32.is_le(left,right) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(U32.to_nat(left),U32.to_nat(right)),U32.is_le(left,right),True{},
    Equal.sym(Bool,U32.is_le(left,right),Nat.is_le(U32.to_nat(left),U32.to_nat(right)),Comparison.le(left,right)),ordered)

def below_end(+value: Nat,+more: Nat,+positive: {Nat.is_lt(0n,more) == True{} : Bool}) -> {Nat.is_lt(value,Nat.add(value,more)) == True{} : Bool}:
  %Addition.zero_right(value) : {Nat.is_lt(_,Nat.add(value,more)) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(Nat.add(value,0n),Nat.add(value,more)),Nat.is_lt(0n,more),Order.lt_offset(value,0n,more)) : {_ == True{} : Bool}
  positive

def count_positive(+count: U32,+query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}:
  Representation.below_positive(U32.to_nat(query),U32.to_nat(count),query_inside)

# ---- The bounds of the cells ----

def groups_above(+groups: U32,+others: U32,+cap: U32,+grid: CellGrid<groups,others,cap>) -> {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool}:
  CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled} = grid
  groups_positive

def others_above(+groups: U32,+others: U32,+cap: U32,+grid: CellGrid<groups,others,cap>) -> {Nat.is_lt(0n,U32.to_nat(others)) == True{} : Bool}:
  CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled} = grid
  others_positive

def groups_room(+groups: U32,+others: U32,+cap: U32,+grid: CellGrid<groups,others,cap>) -> {Nat.is_le(U32.to_nat(groups),U32.to_nat(cap)) == True{} : Bool}:
  CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled} = grid
  Order.transitive(U32.to_nat(groups),Nat.mul(U32.to_nat(groups),U32.to_nat(others)),U32.to_nat(cap),
    RowMajor.product_covers(U32.to_nat(groups),U32.to_nat(others),others_positive),room)

def others_room(+groups: U32,+others: U32,+cap: U32,+grid: CellGrid<groups,others,cap>) -> {Nat.is_le(U32.to_nat(others),U32.to_nat(cap)) == True{} : Bool}:
  CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled} = grid
  Order.transitive(U32.to_nat(others),Nat.mul(U32.to_nat(groups),U32.to_nat(others)),U32.to_nat(cap),
    {%RowMajor.multiply_commutative(U32.to_nat(others),U32.to_nat(groups)) : {Nat.is_le(U32.to_nat(others),_) == True{} : Bool}
      RowMajor.product_covers(U32.to_nat(others),U32.to_nat(groups),groups_positive) :
      {Nat.is_le(U32.to_nat(others),Nat.mul(U32.to_nat(groups),U32.to_nat(others))) == True{} : Bool}},
    room)

# A division by a divisor inside the cells is the division of the naturals.
def grid_divide(+groups: U32,+others: U32,+cap: U32,+grid: CellGrid<groups,others,cap>,+value: U32,+divisor: U32,+value_nat: Nat,+divisor_nat: Nat,
  +value_exact: {U32.to_nat(value) == value_nat : Nat},+divisor_exact: {U32.to_nat(divisor) == divisor_nat : Nat},
  +divisor_room: {Nat.is_le(divisor_nat,U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,divisor_nat) == True{} : Bool}) ->
  {U32.to_nat((value / divisor : U32)) == Nat.div(value_nat,divisor_nat) : Nat}:
  CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled} = grid
  U32Division.quotient_exact(wide,wide_ok,value,divisor,value_nat,divisor_nat,value_exact,divisor_exact,
    Order.transitive(Nat.double(divisor_nat),Nat.double(U32.to_nat(cap)),U32.to_nat(wide),Order.double_le(divisor_nat,U32.to_nat(cap),divisor_room),doubled),positive)

def piece_inside(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(groups),U32.to_nat(others))) == True{} : Bool}:
  PieceIn{+grid,+inside} = piece
  inside

# ---- Walking a range of indices ----

def word_next(+word: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+below: {Nat.is_lt(U32.to_nat(word),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((word + 1 : U32)) == 1n+U32.to_nat(word) : Nat}:
  Equal.trans(Nat,U32.to_nat((word + 1 : U32)),Nat.add(U32.to_nat(word),1n),1n+U32.to_nat(word),
    Arithmetic.add(word,1,cap,cap_ok,
      {%Equal.sym(Nat,Nat.add(U32.to_nat(word),1n),1n+U32.to_nat(word),Order.commutative(U32.to_nat(word),1n)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        Order.strict_follow(U32.to_nat(word),U32.to_nat(cap),below) : {Nat.is_le(Nat.add(U32.to_nat(word),1n),U32.to_nat(cap)) == True{} : Bool}}),
    Order.commutative(U32.to_nat(word),1n))

# An index with more of its range after it lies below the extent.
def range_next(+index: U32,+rest: Nat,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(index),1n+rest),U32.to_nat(extent)) == True{} : Bool}) -> {U32.to_nat((index + 1 : U32)) == 1n+U32.to_nat(index) : Nat}:
  word_next(index,cap,cap_ok,Order.lt_le(U32.to_nat(index),U32.to_nat(extent),U32.to_nat(cap),Order.room_index(U32.to_nat(index),rest,U32.to_nat(extent),room),extent_room))

def range_rest(+index: U32,+rest: Nat,+extent: Nat,+next: {U32.to_nat((index + 1 : U32)) == 1n+U32.to_nat(index) : Nat},
  +room: {Nat.is_le(Nat.add(U32.to_nat(index),1n+rest),extent) == True{} : Bool}) -> {Nat.is_le(Nat.add(U32.to_nat((index + 1 : U32)),rest),extent) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((index + 1 : U32)),1n+U32.to_nat(index),next) : {Nat.is_le(Nat.add(_,rest),extent) == True{} : Bool}
  Order.advance_room(U32.to_nat(index),rest,extent,room)

def range_past(+index: U32,+value: Nat,+next: {U32.to_nat((index + 1 : U32)) == 1n+U32.to_nat(index) : Nat},
  +below: {Nat.is_le(value,U32.to_nat(index)) == True{} : Bool}) -> {Nat.is_lt(value,U32.to_nat((index + 1 : U32))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((index + 1 : U32)),1n+U32.to_nat(index),next) : {Nat.is_lt(value,_) == True{} : Bool}
  Order.lt_succ(value,U32.to_nat(index),below)

def range_on(+index: U32,+value: Nat,+next: {U32.to_nat((index + 1 : U32)) == 1n+U32.to_nat(index) : Nat},
  +from: {Nat.is_le(U32.to_nat(index),value) == True{} : Bool},+apart: {Nat.is_eq(value,U32.to_nat(index)) == False{} : Bool}) ->
  {Nat.is_le(U32.to_nat((index + 1 : U32)),value) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((index + 1 : U32)),1n+U32.to_nat(index),next) : {Nat.is_le(_,value) == True{} : Bool}
  Order.strict_follow(U32.to_nat(index),value,le_apart_lt(U32.to_nat(index),value,from,apart))

def range_within(+index: U32,+rest: Nat,+value: Nat,+next: {U32.to_nat((index + 1 : U32)) == 1n+U32.to_nat(index) : Nat},
  +within: {Nat.is_lt(value,Nat.add(U32.to_nat(index),1n+rest)) == True{} : Bool}) -> {Nat.is_lt(value,Nat.add(U32.to_nat((index + 1 : U32)),rest)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((index + 1 : U32)),1n+U32.to_nat(index),next) : {Nat.is_lt(value,Nat.add(_,rest)) == True{} : Bool}
  %Addition.successor_right(U32.to_nat(index),rest) : {Nat.is_lt(value,_) == True{} : Bool}
  within

def range_none(+index: U32,+value: Nat,+from: {Nat.is_le(U32.to_nat(index),value) == True{} : Bool},
  +within: {Nat.is_lt(value,Nat.add(U32.to_nat(index),0n)) == True{} : Bool}) -> Empty:
  WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(value,U32.to_nat(index)),True{},
    Equal.sym(Bool,Nat.is_lt(value,U32.to_nat(index)),False{},Order.not_below(U32.to_nat(index),value,from)),
    {%Addition.zero_right(U32.to_nat(index)) : {Nat.is_lt(value,_) == True{} : Bool} within : {Nat.is_lt(value,U32.to_nat(index)) == True{} : Bool}}))

# ---- What the cells of a group leave at the query cell ----

# A cell that misses the query leaves what the query cell holds.
def others_skip(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+rest: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+other: U32,+query: U32,+held: Tile.Tile8,+miss: {cell_test(position,groups,others,first,count,group,other,query) == False{} : Bool}) -> {others_expected(~PContext,~pvalue,pcontext,1n+rest,start,width,groups,part_groups,others,position,first,count,group,other,query,held) == others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,held) : Tile.Tile8}:
  %Equal.sym(Bool,cell_test(position,groups,others,first,count,group,other,query),False{},miss) :
    {others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,Bool.pick(Tile.Tile8,_,touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,held),held)) == others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,held) : Tile.Tile8}
  {==}

# The cell that is the query merges over what the query cell holds.
def others_take(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+rest: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+other: U32,+query: U32,+held: Tile.Tile8,+hit: {cell_test(position,groups,others,first,count,group,other,query) == True{} : Bool}) -> {others_expected(~PContext,~pvalue,pcontext,1n+rest,start,width,groups,part_groups,others,position,first,count,group,other,query,held) == others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,held)) : Tile.Tile8}:
  %Equal.sym(Bool,cell_test(position,groups,others,first,count,group,other,query),True{},hit) :
    {others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,Bool.pick(Tile.Tile8,_,touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,held),held)) == others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,held)) : Tile.Tile8}
  {==}

# The cells of a group the query cell is not in leave it.
def others_aside(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+other: U32,+query: U32,+held: Tile.Tile8,+cap: U32,+piece: PieceIn<groups,others,cap,first,count>,+group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(other),fuel),U32.to_nat(others)) == True{} : Bool},+apart: {Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group)) == False{} : Bool}) -> {others_expected(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,other,query,held) == held : Tile.Tile8}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +grid = piece_grid(groups,others,cap,first,count,piece)
      +next = range_next(other,rest,others,cap,cap_fits(groups,others,cap,grid),others_room(groups,others,cap,grid),room)
      +same_group = Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group))
      +same_other = Nat.is_eq(query_other(position,groups,others,first,query),U32.to_nat(other))
      +miss = Equal.trans(Bool,cell_test(position,groups,others,first,count,group,other,query),same_group && same_other,False{},
        tested(position,groups,others,cap,first,count,piece,other,group,Order.room_index(U32.to_nat(other),rest,U32.to_nat(others),room),group_inside,query,query_inside),
        and_false_left(same_group,same_other,apart))
      Equal.trans(Tile.Tile8,others_expected(~PContext,~pvalue,pcontext,1n+rest,start,width,groups,part_groups,others,position,first,count,group,other,query,held),others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,held),held,others_skip(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,other,query,held,miss),
        others_aside(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,held,cap,piece,group_inside,query_inside,range_rest(other,rest,U32.to_nat(others),next,room),apart))

# The cells of a group past the query cell leave it.
def others_past(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+other: U32,+query: U32,+held: Tile.Tile8,+cap: U32,+piece: PieceIn<groups,others,cap,first,count>,+group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(other),fuel),U32.to_nat(others)) == True{} : Bool},+past: {Nat.is_lt(query_other(position,groups,others,first,query),U32.to_nat(other)) == True{} : Bool}) -> {others_expected(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,other,query,held) == held : Tile.Tile8}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +grid = piece_grid(groups,others,cap,first,count,piece)
      +next = range_next(other,rest,others,cap,cap_fits(groups,others,cap,grid),others_room(groups,others,cap,grid),room)
      +same_group = Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group))
      +same_other = Nat.is_eq(query_other(position,groups,others,first,query),U32.to_nat(other))
      +miss = Equal.trans(Bool,cell_test(position,groups,others,first,count,group,other,query),same_group && same_other,False{},
        tested(position,groups,others,cap,first,count,piece,other,group,Order.room_index(U32.to_nat(other),rest,U32.to_nat(others),room),group_inside,query,query_inside),
        and_false_right(same_group,same_other,Order.less_not_equal(query_other(position,groups,others,first,query),U32.to_nat(other),past)))
      Equal.trans(Tile.Tile8,others_expected(~PContext,~pvalue,pcontext,1n+rest,start,width,groups,part_groups,others,position,first,count,group,other,query,held),others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,held),held,others_skip(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,other,query,held,miss),
        others_past(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,held,cap,piece,group_inside,query_inside,range_rest(other,rest,U32.to_nat(others),next,room),
          range_past(other,query_other(position,groups,others,first,query),next,Bounds.strict_le(query_other(position,groups,others,first,query),U32.to_nat(other),past))))

# The cells of the query cell's group from an other index up to past the
# query cell's merge it once.
def others_hit(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,fuel: Nat,on: Bool,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+other: U32,+query: U32,+held: Tile.Tile8,+cap: U32,+piece: PieceIn<groups,others,cap,first,count>,+group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(other),fuel),U32.to_nat(others)) == True{} : Bool},+same: {Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group)) == True{} : Bool},
  +from: {Nat.is_le(U32.to_nat(other),query_other(position,groups,others,first,query)) == True{} : Bool},+within: {Nat.is_lt(query_other(position,groups,others,first,query),Nat.add(U32.to_nat(other),fuel)) == True{} : Bool},
  +on_is: {Nat.is_eq(query_other(position,groups,others,first,query),U32.to_nat(other)) == on : Bool}) -> {others_expected(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,other,query,held) == touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,U32.from_nat(query_other(position,groups,others,first,query)),held) : Tile.Tile8}:
  match fuel:
    case 0n:
      Empty.absurd({others_expected(~PContext,~pvalue,pcontext,0n,start,width,groups,part_groups,others,position,first,count,group,other,query,held) == touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,U32.from_nat(query_other(position,groups,others,first,query)),held) : Tile.Tile8},range_none(other,query_other(position,groups,others,first,query),from,within))
    case 1n+ +rest:
      match on:
        case True{}:
          +grid = piece_grid(groups,others,cap,first,count,piece)
          +next = range_next(other,rest,others,cap,cap_fits(groups,others,cap,grid),others_room(groups,others,cap,grid),room)
          +same_group = Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group))
          +hit = Equal.trans(Bool,cell_test(position,groups,others,first,count,group,other,query),same_group && Nat.is_eq(query_other(position,groups,others,first,query),U32.to_nat(other)),True{},
            tested(position,groups,others,cap,first,count,piece,other,group,Order.room_index(U32.to_nat(other),rest,U32.to_nat(others),room),group_inside,query,query_inside),
            Logic.and_true(same_group,Nat.is_eq(query_other(position,groups,others,first,query),U32.to_nat(other)),same,on_is))
          +taken = touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,other,held)
          Equal.trans(Tile.Tile8,others_expected(~PContext,~pvalue,pcontext,1n+rest,start,width,groups,part_groups,others,position,first,count,group,other,query,held),others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,taken),touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,U32.from_nat(query_other(position,groups,others,first,query)),held),others_take(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,other,query,held,hit),
            Equal.trans(Tile.Tile8,others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,taken),taken,touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,U32.from_nat(query_other(position,groups,others,first,query)),held),
              others_past(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,taken,cap,piece,group_inside,query_inside,range_rest(other,rest,U32.to_nat(others),next,room),
                range_past(other,query_other(position,groups,others,first,query),next,equal_le(query_other(position,groups,others,first,query),U32.to_nat(other),on_is))),
              Equal.cong(U32,Tile.Tile8,word => touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,word,held),other,U32.from_nat(query_other(position,groups,others,first,query)),word_same(other,query_other(position,groups,others,first,query),on_is))))
        case False{}:
          +grid = piece_grid(groups,others,cap,first,count,piece)
          +next = range_next(other,rest,others,cap,cap_fits(groups,others,cap,grid),others_room(groups,others,cap,grid),room)
          +same_group = Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group))
          +same_other = Nat.is_eq(query_other(position,groups,others,first,query),U32.to_nat(other))
          +miss = Equal.trans(Bool,cell_test(position,groups,others,first,count,group,other,query),same_group && same_other,False{},
            tested(position,groups,others,cap,first,count,piece,other,group,Order.room_index(U32.to_nat(other),rest,U32.to_nat(others),room),group_inside,query,query_inside),
            and_false_right(same_group,same_other,on_is))
          Equal.trans(Tile.Tile8,others_expected(~PContext,~pvalue,pcontext,1n+rest,start,width,groups,part_groups,others,position,first,count,group,other,query,held),others_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,(other + 1 : U32),query,held),touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,U32.from_nat(query_other(position,groups,others,first,query)),held),others_skip(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,group,other,query,held,miss),
            others_hit(~PContext,~pvalue,pcontext,rest,Nat.is_eq(query_other(position,groups,others,first,query),U32.to_nat((other + 1 : U32))),start,width,groups,part_groups,others,position,first,count,group,
              (other + 1 : U32),query,held,cap,piece,group_inside,query_inside,range_rest(other,rest,U32.to_nat(others),next,room),same,
              range_on(other,query_other(position,groups,others,first,query),next,from,on_is),range_within(other,rest,query_other(position,groups,others,first,query),next,within),{==}))

# ---- The other indices of a group ----

def range_low(+found: Tiled.Others) -> U32:
  Tiled.Others{+low,+high} = found
  low

def range_high(+found: Tiled.Others) -> U32:
  Tiled.Others{+low,+high} = found
  high

def range_top(+found: Tiled.Others) -> U32:
  Tiled.Others{+low,+high} = found
  U32.max(high,low)

def range_fuel(+found: Tiled.Others) -> Nat:
  Tiled.Others{+low,+high} = found
  U32.to_nat((U32.max(high,low) - low : U32))

# The other indices of a range lie within the others.
type RangeFits<-found: Tiled.Others,-others: U32> is Data:
  RangeFits{low_room: {Nat.is_le(U32.to_nat(range_low(found)),U32.to_nat(others)) == True{} : Bool},
    high_room: {Nat.is_le(U32.to_nat(range_high(found)),U32.to_nat(others)) == True{} : Bool}}

# A range of other indices holds a value.
type RangeCovers<-found: Tiled.Others,-value: Nat> is Data:
  RangeCovers{from: {Nat.is_le(U32.to_nat(range_low(found)),value) == True{} : Bool},within: {Nat.is_lt(value,U32.to_nat(range_high(found))) == True{} : Bool}}

def range_top_room(+found: Tiled.Others,+others: U32,+fits: RangeFits<found,others>) -> {Nat.is_le(U32.to_nat(range_top(found)),U32.to_nat(others)) == True{} : Bool}:
  Tiled.Others{+low,+high} = found
  RangeFits{+low_room,+high_room} = fits
  %Equal.sym(Nat,U32.to_nat(U32.max(high,low)),Nat.max(U32.to_nat(high),U32.to_nat(low)),Extrema.maximum(high,low)) : {Nat.is_le(_,U32.to_nat(others)) == True{} : Bool}
  max_le(U32.to_nat(high),U32.to_nat(low),U32.to_nat(others),high_room,low_room)

def range_high_top(+found: Tiled.Others) -> {Nat.is_le(U32.to_nat(range_high(found)),U32.to_nat(range_top(found))) == True{} : Bool}:
  Tiled.Others{+low,+high} = found
  word_nat_le(high,U32.max(high,low),Extrema.maximum_left(high,low))

# The range ends at its top.
def range_end(+found: Tiled.Others,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +top_room: {Nat.is_le(U32.to_nat(range_top(found)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(range_low(found)),range_fuel(found)) == U32.to_nat(range_top(found)) : Nat}:
  Tiled.Others{+low,+high} = found
  +top = U32.max(high,low)
  +below = word_nat_le(low,top,Extrema.maximum_right(high,low))
  +exact = Arithmetic.subtract(top,low,cap,cap_ok,below,Order.transitive(Nat.sub(U32.to_nat(top),U32.to_nat(low)),U32.to_nat(top),U32.to_nat(cap),
    Order.sub_le(U32.to_nat(top),U32.to_nat(low)),top_room))
  Equal.trans(Nat,Nat.add(U32.to_nat(low),U32.to_nat((top - low : U32))),Nat.add(U32.to_nat(low),Nat.sub(U32.to_nat(top),U32.to_nat(low))),U32.to_nat(top),
    Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(low),value),U32.to_nat((top - low : U32)),Nat.sub(U32.to_nat(top),U32.to_nat(low)),exact),
    Order.restore_shift(U32.to_nat(low),U32.to_nat(top),below))

def range_room(+found: Tiled.Others,+others: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +others_cap: {Nat.is_le(U32.to_nat(others),U32.to_nat(cap)) == True{} : Bool},+fits: RangeFits<found,others>) ->
  {Nat.is_le(Nat.add(U32.to_nat(range_low(found)),range_fuel(found)),U32.to_nat(others)) == True{} : Bool}:
  +top_room = range_top_room(found,others,fits)
  %Equal.sym(Nat,Nat.add(U32.to_nat(range_low(found)),range_fuel(found)),U32.to_nat(range_top(found)),
      range_end(found,cap,cap_ok,Order.transitive(U32.to_nat(range_top(found)),U32.to_nat(others),U32.to_nat(cap),top_room,others_cap))) :
    {Nat.is_le(_,U32.to_nat(others)) == True{} : Bool}
  top_room

def range_reaches(+found: Tiled.Others,+value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +top_room: {Nat.is_le(U32.to_nat(range_top(found)),U32.to_nat(cap)) == True{} : Bool},+within: {Nat.is_lt(value,U32.to_nat(range_high(found))) == True{} : Bool}) ->
  {Nat.is_lt(value,Nat.add(U32.to_nat(range_low(found)),range_fuel(found))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(U32.to_nat(range_low(found)),range_fuel(found)),U32.to_nat(range_top(found)),range_end(found,cap,cap_ok,top_room)) : {Nat.is_lt(value,_) == True{} : Bool}
  Order.lt_le(value,U32.to_nat(range_high(found)),U32.to_nat(range_top(found)),within,range_high_top(found))

# The cells of a group the query cell is not in leave it.
def range_aside(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+found: Tiled.Others,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+query: U32,+held: Tile.Tile8,+cap: U32,+piece: PieceIn<groups,others,cap,first,count>,+group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+fits: RangeFits<found,others>,+apart: {Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group)) == False{} : Bool}) -> {range_expected(~PContext,~pvalue,pcontext,found,start,width,groups,part_groups,others,position,first,count,group,query,held) == held : Tile.Tile8}:
  Tiled.Others{+low,+high} = found
  +grid = piece_grid(groups,others,cap,first,count,piece)
  others_aside(~PContext,~pvalue,pcontext,U32.to_nat((U32.max(high,low) - low : U32)),start,width,groups,part_groups,others,position,first,count,group,low,query,held,cap,piece,
    group_inside,query_inside,range_room(Tiled.Others{low,high},others,cap,cap_fits(groups,others,cap,grid),others_room(groups,others,cap,grid),fits),apart)

# The cells of the query cell's group whose range holds its other index merge
# it once.
def range_hit(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+found: Tiled.Others,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+query: U32,+held: Tile.Tile8,+cap: U32,+piece: PieceIn<groups,others,cap,first,count>,+group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+fits: RangeFits<found,others>,+covers: RangeCovers<found,query_other(position,groups,others,first,query)>,+same: {Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group)) == True{} : Bool}) ->
  {range_expected(~PContext,~pvalue,pcontext,found,start,width,groups,part_groups,others,position,first,count,group,query,held) == touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,U32.from_nat(query_other(position,groups,others,first,query)),held) : Tile.Tile8}:
  Tiled.Others{+low,+high} = found
  RangeCovers{+from,+within} = covers
  +grid = piece_grid(groups,others,cap,first,count,piece)
  +cap_ok = cap_fits(groups,others,cap,grid)
  +others_cap = others_room(groups,others,cap,grid)
  +bounds = {Tiled.Others{low,high} : Tiled.Others}
  +top_room = Order.transitive(U32.to_nat(range_top(bounds)),U32.to_nat(others),U32.to_nat(cap),range_top_room(bounds,others,fits),others_cap)
  others_hit(~PContext,~pvalue,pcontext,U32.to_nat((U32.max(high,low) - low : U32)),Nat.is_eq(query_other(position,groups,others,first,query),U32.to_nat(low)),start,width,groups,part_groups,others,position,first,
    count,group,low,query,held,cap,piece,group_inside,query_inside,range_room(bounds,others,cap,cap_ok,others_cap,fits),same,from,
    range_reaches(bounds,query_other(position,groups,others,first,query),cap,cap_ok,top_room,within),{==})

# ---- The ranges of other indices the runtime walks ----

def inner_low(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>) ->
  {U32.to_nat((first / U32.max(groups,1) : U32)) == Nat.div(U32.to_nat(first),U32.to_nat(groups)) : Nat}:
  +grid = piece_grid(groups,others,cap,first,count,piece)
  +positive = groups_above(groups,others,cap,grid)
  grid_divide(groups,others,cap,grid,first,U32.max(groups,1),U32.to_nat(first),U32.to_nat(groups),{==},
    Held.stride_divisor(groups,Arithmetic.word_positive_nonzero(groups,positive)),groups_room(groups,others,cap,grid),positive)

def inner_last(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) -> {U32.to_nat((first + count - 1 : U32)) == Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n) : Nat}:
  +grid = piece_grid(groups,others,cap,first,count,piece)
  +cap_ok = cap_fits(groups,others,cap,grid)
  +total = Nat.add(U32.to_nat(first),U32.to_nat(count))
  +room = piece_room(groups,others,cap,first,count,piece)
  Arithmetic.difference_exact(cap,cap_ok,(first + count : U32),1,total,1n,Arithmetic.sum_exact(cap,cap_ok,first,count,U32.to_nat(first),U32.to_nat(count),{==},{==},room),{==},
    Order.strict_follow(0n,total,Order.lt_le(0n,U32.to_nat(count),total,positive,Order.right_le(U32.to_nat(first),U32.to_nat(count)))),room)

def inner_last_quotient(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) -> {Nat.is_lt(Nat.div(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),U32.to_nat(groups)),U32.to_nat(others)) == True{} : Bool}:
  PieceIn{+grid,+inside} = piece
  +total = Nat.add(U32.to_nat(first),U32.to_nat(count))
  DivisionOrder.quotient_below(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),U32.to_nat(others),U32.to_nat(groups),groups_above(groups,others,cap,grid),
    {%RowMajor.multiply_commutative(U32.to_nat(groups),U32.to_nat(others)) : {Nat.is_lt(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),_) == True{} : Bool}
      Order.lt_le(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),total,Nat.mul(U32.to_nat(groups),U32.to_nat(others)),
        Order.last_below(total,Order.lt_le(0n,U32.to_nat(count),total,positive,Order.right_le(U32.to_nat(first),U32.to_nat(count)))),inside) :
      {Nat.is_lt(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),Nat.mul(U32.to_nat(others),U32.to_nat(groups))) == True{} : Bool}})

def inner_high(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) -> {U32.to_nat(((first + count - 1) / U32.max(groups,1) + 1 : U32)) == 1n+Nat.div(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),U32.to_nat(groups)) : Nat}:
  +grid = piece_grid(groups,others,cap,first,count,piece)
  +divisor_positive = groups_above(groups,others,cap,grid)
  +quotient = ((first + count - 1) / U32.max(groups,1) : U32)
  +exact = grid_divide(groups,others,cap,grid,(first + count - 1 : U32),U32.max(groups,1),Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),U32.to_nat(groups),inner_last(groups,others,cap,first,count,piece,positive),
    Held.stride_divisor(groups,Arithmetic.word_positive_nonzero(groups,divisor_positive)),groups_room(groups,others,cap,grid),divisor_positive)
  %exact : {U32.to_nat((quotient + 1 : U32)) == 1n+_ : Nat}
  word_next(quotient,cap,cap_fits(groups,others,cap,grid),
    {%Equal.sym(Nat,U32.to_nat(quotient),Nat.div(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),U32.to_nat(groups)),exact) : {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool}
      Order.lt_le(Nat.div(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),U32.to_nat(groups)),U32.to_nat(others),U32.to_nat(cap),inner_last_quotient(groups,others,cap,first,count,piece,positive),
        others_room(groups,others,cap,grid)) : {Nat.is_lt(U32.to_nat(quotient),U32.to_nat(cap)) == True{} : Bool}})

# With the group innermost the range runs from the other index of the first
# cell to the one past the last cell's.
def inner_fit(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+group: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) -> RangeFits<Tiled.others_of(Grid.Innermost{},groups,others,first,count,group),others>:
  +grid = piece_grid(groups,others,cap,first,count,piece)
  +divisor_positive = groups_above(groups,others,cap,grid)
  RangeFits{
    {%Equal.sym(Nat,U32.to_nat((first / U32.max(groups,1) : U32)),Nat.div(U32.to_nat(first),U32.to_nat(groups)),inner_low(groups,others,cap,first,count,piece)) : {Nat.is_le(_,U32.to_nat(others)) == True{} : Bool}
      Bounds.strict_le(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(others),
        DivisionOrder.quotient_below(U32.to_nat(first),U32.to_nat(others),U32.to_nat(groups),divisor_positive,
          {%RowMajor.multiply_commutative(U32.to_nat(groups),U32.to_nat(others)) : {Nat.is_lt(U32.to_nat(first),_) == True{} : Bool}
            Order.lt_le(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(groups),U32.to_nat(others)),
              below_end(U32.to_nat(first),U32.to_nat(count),positive),piece_inside(groups,others,cap,first,count,piece)) :
            {Nat.is_lt(U32.to_nat(first),Nat.mul(U32.to_nat(others),U32.to_nat(groups))) == True{} : Bool}})) :
      {Nat.is_le(U32.to_nat((first / U32.max(groups,1) : U32)),U32.to_nat(others)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(((first + count - 1) / U32.max(groups,1) + 1 : U32)),1n+Nat.div(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),U32.to_nat(groups)),inner_high(groups,others,cap,first,count,piece,positive)) : {Nat.is_le(_,U32.to_nat(others)) == True{} : Bool}
      Order.strict_follow(Nat.div(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),U32.to_nat(groups)),U32.to_nat(others),inner_last_quotient(groups,others,cap,first,count,piece,positive)) :
      {Nat.is_le(U32.to_nat(((first + count - 1) / U32.max(groups,1) + 1 : U32)),U32.to_nat(others)) == True{} : Bool}}}

def inner_cover(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+group: U32,+piece: PieceIn<groups,others,cap,first,count>,+query: U32,
  +query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}) ->
  RangeCovers<Tiled.others_of(Grid.Innermost{},groups,others,first,count,group),query_other(Grid.Innermost{},groups,others,first,query)>:
  +grid = piece_grid(groups,others,cap,first,count,piece)
  +divisor_positive = groups_above(groups,others,cap,grid)
  +positive = count_positive(count,query,query_inside)
  +cell = query_cell(first,query)
  +divisor = U32.to_nat(groups)
  +least = Representation.least_positive(divisor,divisor_positive)
  RangeCovers{
    {%Equal.sym(Nat,U32.to_nat((first / U32.max(groups,1) : U32)),Nat.div(U32.to_nat(first),divisor),inner_low(groups,others,cap,first,count,piece)) :
        {Nat.is_le(_,Nat.div(cell,Representation.least_one(divisor))) == True{} : Bool}
      %Equal.sym(Nat,Representation.least_one(divisor),divisor,least) : {Nat.is_le(Nat.div(U32.to_nat(first),divisor),Nat.div(cell,_)) == True{} : Bool}
      DivisionOrder.quotient_le(U32.to_nat(first),cell,divisor,divisor_positive,Order.le_add(U32.to_nat(first),U32.to_nat(query))) :
      {Nat.is_le(U32.to_nat((first / U32.max(groups,1) : U32)),Nat.div(cell,Representation.least_one(divisor))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(((first + count - 1) / U32.max(groups,1) + 1 : U32)),1n+Nat.div(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),divisor),inner_high(groups,others,cap,first,count,piece,positive)) :
        {Nat.is_lt(Nat.div(cell,Representation.least_one(divisor)),_) == True{} : Bool}
      %Equal.sym(Nat,Representation.least_one(divisor),divisor,least) : {Nat.is_lt(Nat.div(cell,_),1n+Nat.div(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),divisor)) == True{} : Bool}
      Order.lt_succ(Nat.div(cell,divisor),Nat.div(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),divisor),DivisionOrder.quotient_le(cell,Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n),divisor,divisor_positive,
        Order.before_last(cell,Nat.add(U32.to_nat(first),U32.to_nat(count)),query_below(first,count,query,query_inside)))) :
      {Nat.is_lt(Nat.div(cell,Representation.least_one(divisor)),U32.to_nat(((first + count - 1) / U32.max(groups,1) + 1 : U32))) == True{} : Bool}}}

# With the group outermost the cells of a group start at the group times the
# others.
def outer_top_room(+groups: U32,+others: U32,+cap: U32,+group: U32,+grid: CellGrid<groups,others,cap>,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(group),U32.to_nat(others)),U32.to_nat(others)),U32.to_nat(cap)) == True{} : Bool}:
  CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled} = grid
  %Order.commutative(U32.to_nat(others),Nat.mul(U32.to_nat(group),U32.to_nat(others))) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
  Order.transitive(Nat.mul(1n+U32.to_nat(group),U32.to_nat(others)),Nat.mul(U32.to_nat(groups),U32.to_nat(others)),U32.to_nat(cap),
    RowMajor.mul_le(1n+U32.to_nat(group),U32.to_nat(groups),U32.to_nat(others),Order.strict_follow(U32.to_nat(group),U32.to_nat(groups),group_inside)),room)

def outer_base(+groups: U32,+others: U32,+cap: U32,+group: U32,+grid: CellGrid<groups,others,cap>,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool}) ->
  {U32.to_nat((group * others : U32)) == Nat.mul(U32.to_nat(group),U32.to_nat(others)) : Nat}:
  +start = Nat.mul(U32.to_nat(group),U32.to_nat(others))
  Arithmetic.product_exact(cap,cap_fits(groups,others,cap,grid),group,others,U32.to_nat(group),U32.to_nat(others),{==},{==},
    Order.transitive(start,Nat.add(start,U32.to_nat(others)),U32.to_nat(cap),Order.le_add(start,U32.to_nat(others)),outer_top_room(groups,others,cap,group,grid,group_inside)))

# A word clamped to the cells of a group, less the group's first cell.
def outer_clamped(+groups: U32,+others: U32,+cap: U32,+group: U32,+grid: CellGrid<groups,others,cap>,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+value: U32,+amount: Nat,+exact: {U32.to_nat(value) == amount : Nat}) ->
  {U32.to_nat((U32.min(U32.max(value,(group * others : U32)),((group * others : U32) + others : U32)) - (group * others : U32) : U32)) == Nat.sub(Nat.min(Nat.max(amount,Nat.mul(U32.to_nat(group),U32.to_nat(others))),Nat.add(Nat.mul(U32.to_nat(group),U32.to_nat(others)),U32.to_nat(others))),Nat.mul(U32.to_nat(group),U32.to_nat(others))) : Nat}:
  +cap_ok = cap_fits(groups,others,cap,grid)
  +base = (group * others : U32)
  +start = Nat.mul(U32.to_nat(group),U32.to_nat(others))
  +finish = Nat.add(start,U32.to_nat(others))
  +base_exact = outer_base(groups,others,cap,group,grid,group_inside)
  +top_room = outer_top_room(groups,others,cap,group,grid,group_inside)
  +finish_exact = Arithmetic.sum_exact(cap,cap_ok,base,others,start,U32.to_nat(others),base_exact,{==},top_room)
  +raised = Nat.max(amount,start)
  +raised_exact = Equal.trans(Nat,U32.to_nat(U32.max(value,base)),Nat.max(U32.to_nat(value),U32.to_nat(base)),raised,Extrema.maximum(value,base),
    %exact : {Nat.max(U32.to_nat(value),U32.to_nat(base)) == Nat.max(_,start) : Nat}
    %base_exact : {Nat.max(U32.to_nat(value),U32.to_nat(base)) == Nat.max(U32.to_nat(value),_) : Nat}
    {==})
  +clamped = Nat.min(raised,finish)
  +clamped_exact = Equal.trans(Nat,U32.to_nat(U32.min(U32.max(value,base),(base + others : U32))),
    Nat.min(U32.to_nat(U32.max(value,base)),U32.to_nat((base + others : U32))),clamped,Extrema.minimum(U32.max(value,base),(base + others : U32)),
    %raised_exact : {Nat.min(U32.to_nat(U32.max(value,base)),U32.to_nat((base + others : U32))) == Nat.min(_,finish) : Nat}
    %finish_exact : {Nat.min(U32.to_nat(U32.max(value,base)),U32.to_nat((base + others : U32))) == Nat.min(U32.to_nat(U32.max(value,base)),_) : Nat}
    {==})
  +above = min_ge(raised,finish,start,NatExtrema.maximum_right(amount,start),Order.le_add(start,U32.to_nat(others)))
  Arithmetic.difference_exact(cap,cap_ok,U32.min(U32.max(value,base),(base + others : U32)),base,clamped,start,clamped_exact,base_exact,above,
    Order.transitive(clamped,finish,U32.to_nat(cap),NatExtrema.minimum_right(raised,finish),top_room))

def outer_clamped_room(+others: U32,+group: U32,+amount: Nat) -> {Nat.is_le(Nat.sub(Nat.min(Nat.max(amount,Nat.mul(U32.to_nat(group),U32.to_nat(others))),Nat.add(Nat.mul(U32.to_nat(group),U32.to_nat(others)),U32.to_nat(others))),Nat.mul(U32.to_nat(group),U32.to_nat(others))),U32.to_nat(others)) == True{} : Bool}:
  +start = Nat.mul(U32.to_nat(group),U32.to_nat(others))
  +finish = Nat.add(start,U32.to_nat(others))
  %Order.subtract_shift(start,U32.to_nat(others)) : {Nat.is_le(Nat.sub(Nat.min(Nat.max(amount,Nat.mul(U32.to_nat(group),U32.to_nat(others))),Nat.add(Nat.mul(U32.to_nat(group),U32.to_nat(others)),U32.to_nat(others))),Nat.mul(U32.to_nat(group),U32.to_nat(others))),_) == True{} : Bool}
  DivisionOrder.subtract_le(Nat.min(Nat.max(amount,start),finish),finish,start,NatExtrema.minimum_right(Nat.max(amount,start),finish))

def outer_fit(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+group: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool}) -> RangeFits<Tiled.others_of(Grid.Outermost{},groups,others,first,count,group),others>:
  +grid = piece_grid(groups,others,cap,first,count,piece)
  +cap_ok = cap_fits(groups,others,cap,grid)
  +total = Nat.add(U32.to_nat(first),U32.to_nat(count))
  RangeFits{
    {%Equal.sym(Nat,U32.to_nat((U32.min(U32.max(first,(group * others : U32)),((group * others : U32) + others : U32)) - (group * others : U32) : U32)),Nat.sub(Nat.min(Nat.max(U32.to_nat(first),Nat.mul(U32.to_nat(group),U32.to_nat(others))),Nat.add(Nat.mul(U32.to_nat(group),U32.to_nat(others)),U32.to_nat(others))),Nat.mul(U32.to_nat(group),U32.to_nat(others))),outer_clamped(groups,others,cap,group,grid,group_inside,first,U32.to_nat(first),{==})) :
        {Nat.is_le(_,U32.to_nat(others)) == True{} : Bool}
      outer_clamped_room(others,group,U32.to_nat(first)) : {Nat.is_le(U32.to_nat((U32.min(U32.max(first,(group * others : U32)),((group * others : U32) + others : U32)) - (group * others : U32) : U32)),U32.to_nat(others)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat((U32.min(U32.max((first + count : U32),(group * others : U32)),((group * others : U32) + others : U32)) - (group * others : U32) : U32)),Nat.sub(Nat.min(Nat.max(total,Nat.mul(U32.to_nat(group),U32.to_nat(others))),Nat.add(Nat.mul(U32.to_nat(group),U32.to_nat(others)),U32.to_nat(others))),Nat.mul(U32.to_nat(group),U32.to_nat(others))),outer_clamped(groups,others,cap,group,grid,group_inside,(first + count : U32),total,
        Arithmetic.add(first,count,cap,cap_ok,piece_room(groups,others,cap,first,count,piece)))) : {Nat.is_le(_,U32.to_nat(others)) == True{} : Bool}
      outer_clamped_room(others,group,total) : {Nat.is_le(U32.to_nat((U32.min(U32.max((first + count : U32),(group * others : U32)),((group * others : U32) + others : U32)) - (group * others : U32) : U32)),U32.to_nat(others)) == True{} : Bool}}}

def outer_cover(+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+group: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},
  +same: {Nat.is_eq(query_group(Grid.Outermost{},groups,others,first,query),U32.to_nat(group)) == True{} : Bool}) ->
  RangeCovers<Tiled.others_of(Grid.Outermost{},groups,others,first,count,group),query_other(Grid.Outermost{},groups,others,first,query)>:
  +grid = piece_grid(groups,others,cap,first,count,piece)
  +cap_ok = cap_fits(groups,others,cap,grid)
  +extent = U32.to_nat(others)
  +positive = others_above(groups,others,cap,grid)
  +least = Representation.least_positive(extent,positive)
  +cell = query_cell(first,query)
  +total = Nat.add(U32.to_nat(first),U32.to_nat(count))
  +start = Nat.mul(U32.to_nat(group),extent)
  +finish = Nat.add(start,extent)
  +place = Nat.mod(cell,extent)
  +quotient = {%least : {Nat.div(cell,_) == U32.to_nat(group) : Nat}
    Order.equal_of_is_eq(query_group(Grid.Outermost{},groups,others,first,query),U32.to_nat(group),same) : {Nat.div(cell,extent) == U32.to_nat(group) : Nat}}
  +placed = {%quotient : {cell == Nat.add(Nat.mul(_,extent),place) : Nat} DivisionOrder.quotient_exact(cell,extent,positive) : {cell == Nat.add(start,place) : Nat}}
  +placed_back = Equal.sym(Nat,cell,Nat.add(start,place),placed)
  +start_le_cell = {%placed_back : {Nat.is_le(start,_) == True{} : Bool} Order.le_add(start,place) : {Nat.is_le(start,cell) == True{} : Bool}}
  +lowest = Nat.min(Nat.max(U32.to_nat(first),start),finish)
  +low_bound = Order.transitive(lowest,Nat.max(U32.to_nat(first),start),cell,NatExtrema.minimum_left(Nat.max(U32.to_nat(first),start),finish),
    max_le(U32.to_nat(first),start,cell,Order.le_add(U32.to_nat(first),U32.to_nat(query)),start_le_cell))
  +from = {%Order.subtract_shift(start,place) : {Nat.is_le(Nat.sub(lowest,start),_) == True{} : Bool}
    %placed : {Nat.is_le(Nat.sub(lowest,start),Nat.sub(_,start)) == True{} : Bool}
    DivisionOrder.subtract_le(lowest,cell,start,low_bound) : {Nat.is_le(Nat.sub(lowest,start),place) == True{} : Bool}}
  +highest = Nat.min(Nat.max(total,start),finish)
  +cell_finish = {%placed_back : {Nat.is_lt(_,finish) == True{} : Bool}
    %Equal.sym(Bool,Nat.is_lt(Nat.add(start,place),finish),Nat.is_lt(place,extent),Order.lt_offset(start,place,extent)) : {_ == True{} : Bool}
    DivisionOrder.remainder_small(cell,extent,positive) : {Nat.is_lt(cell,finish) == True{} : Bool}}
  +high_above = min_above(Nat.max(total,start),finish,cell,
    Order.lt_le(cell,total,Nat.max(total,start),query_below(first,count,query,query_inside),NatExtrema.maximum_left(total,start)),cell_finish)
  +start_le_high = min_ge(Nat.max(total,start),finish,start,NatExtrema.maximum_right(total,start),Order.le_add(start,extent))
  +within = Equal.trans(Bool,Nat.is_lt(place,Nat.sub(highest,start)),Nat.is_lt(Nat.add(start,place),highest),True{},Order.below_difference(start,place,highest,start_le_high),
    {%placed : {Nat.is_lt(_,highest) == True{} : Bool} high_above : {Nat.is_lt(Nat.add(start,place),highest) == True{} : Bool}})
  %Equal.sym(Nat,Representation.least_one(extent),extent,least) :
    RangeCovers<Tiled.others_of(Grid.Outermost{},groups,others,first,count,group),Nat.mod(cell,_)>
  RangeCovers{
    {%Equal.sym(Nat,U32.to_nat((U32.min(U32.max(first,(group * others : U32)),((group * others : U32) + others : U32)) - (group * others : U32) : U32)),Nat.sub(Nat.min(Nat.max(U32.to_nat(first),Nat.mul(U32.to_nat(group),U32.to_nat(others))),Nat.add(Nat.mul(U32.to_nat(group),U32.to_nat(others)),U32.to_nat(others))),Nat.mul(U32.to_nat(group),U32.to_nat(others))),outer_clamped(groups,others,cap,group,grid,group_inside,first,U32.to_nat(first),{==})) :
        {Nat.is_le(_,place) == True{} : Bool}
      from : {Nat.is_le(U32.to_nat((U32.min(U32.max(first,(group * others : U32)),((group * others : U32) + others : U32)) - (group * others : U32) : U32)),place) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat((U32.min(U32.max((first + count : U32),(group * others : U32)),((group * others : U32) + others : U32)) - (group * others : U32) : U32)),Nat.sub(Nat.min(Nat.max(total,Nat.mul(U32.to_nat(group),U32.to_nat(others))),Nat.add(Nat.mul(U32.to_nat(group),U32.to_nat(others)),U32.to_nat(others))),Nat.mul(U32.to_nat(group),U32.to_nat(others))),outer_clamped(groups,others,cap,group,grid,group_inside,(first + count : U32),total,
        Arithmetic.add(first,count,cap,cap_ok,piece_room(groups,others,cap,first,count,piece)))) : {Nat.is_lt(place,_) == True{} : Bool}
      within : {Nat.is_lt(place,U32.to_nat((U32.min(U32.max((first + count : U32),(group * others : U32)),((group * others : U32) + others : U32)) - (group * others : U32) : U32))) == True{} : Bool}}}

def others_fit(position: Grid.Position,+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+group: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) ->
  RangeFits<Tiled.others_of(position,groups,others,first,count,group),others>:
  match position:
    case Grid.Innermost{}: inner_fit(groups,others,cap,first,count,group,piece,positive)
    case Grid.Outermost{}: outer_fit(groups,others,cap,first,count,group,piece,group_inside)

def others_cover(position: Grid.Position,+groups: U32,+others: U32,+cap: U32,+first: U32,+count: U32,+group: U32,+piece: PieceIn<groups,others,cap,first,count>,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},
  +same: {Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group)) == True{} : Bool}) ->
  RangeCovers<Tiled.others_of(position,groups,others,first,count,group),query_other(position,groups,others,first,query)>:
  match position:
    case Grid.Innermost{}: inner_cover(groups,others,cap,first,count,group,piece,query,query_inside)
    case Grid.Outermost{}: outer_cover(groups,others,cap,first,count,group,piece,group_inside,query,query_inside,same)

# ---- What the groups leave at the query cell ----

# The groups past the query cell's leave it.
def groups_past(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+query: U32,+held: Tile.Tile8,+cap: U32,+piece: PieceIn<groups,others,cap,first,count>,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(group),fuel),U32.to_nat(groups)) == True{} : Bool},+past: {Nat.is_lt(query_group(position,groups,others,first,query),U32.to_nat(group)) == True{} : Bool}) -> {groups_expected(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,query,held) == held : Tile.Tile8}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +grid = piece_grid(groups,others,cap,first,count,piece)
      +next = range_next(group,rest,groups,cap,cap_fits(groups,others,cap,grid),groups_room(groups,others,cap,grid),room)
      +group_inside = Order.room_index(U32.to_nat(group),rest,U32.to_nat(groups),room)
      +aside = range_aside(~PContext,~pvalue,pcontext,Tiled.others_of(position,groups,others,first,count,group),start,width,groups,part_groups,others,position,first,count,group,query,held,cap,piece,group_inside,query_inside,
        others_fit(position,groups,others,cap,first,count,group,piece,group_inside,count_positive(count,query,query_inside)),Order.less_not_equal(query_group(position,groups,others,first,query),U32.to_nat(group),past))
      Equal.trans(Tile.Tile8,groups_expected(~PContext,~pvalue,pcontext,1n+rest,start,width,groups,part_groups,others,position,first,count,group,query,held),groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,held),held,
        Equal.cong(Tile.Tile8,Tile.Tile8,cell => groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,cell),range_expected(~PContext,~pvalue,pcontext,Tiled.others_of(position,groups,others,first,count,group),start,width,groups,part_groups,others,position,first,count,group,query,held),held,aside),
        groups_past(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,held,cap,piece,query_inside,range_rest(group,rest,U32.to_nat(groups),next,room),
          range_past(group,query_group(position,groups,others,first,query),next,Bounds.strict_le(query_group(position,groups,others,first,query),U32.to_nat(group),past))))

# The groups before the query cell's leave it.
def groups_beyond(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+query: U32,+held: Tile.Tile8,+cap: U32,+piece: PieceIn<groups,others,cap,first,count>,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(group),fuel),U32.to_nat(groups)) == True{} : Bool},+beyond: {Nat.is_le(Nat.add(U32.to_nat(group),fuel),query_group(position,groups,others,first,query)) == True{} : Bool}) -> {groups_expected(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,query,held) == held : Tile.Tile8}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +grid = piece_grid(groups,others,cap,first,count,piece)
      +next = range_next(group,rest,groups,cap,cap_fits(groups,others,cap,grid),groups_room(groups,others,cap,grid),room)
      +group_inside = Order.room_index(U32.to_nat(group),rest,U32.to_nat(groups),room)
      +apart = Equal.trans(Bool,Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group)),Nat.is_eq(U32.to_nat(group),query_group(position,groups,others,first,query)),False{},Order.equal_symmetric(query_group(position,groups,others,first,query),U32.to_nat(group)),
        Order.less_not_equal(U32.to_nat(group),query_group(position,groups,others,first,query),Order.room_index(U32.to_nat(group),rest,query_group(position,groups,others,first,query),beyond)))
      +aside = range_aside(~PContext,~pvalue,pcontext,Tiled.others_of(position,groups,others,first,count,group),start,width,groups,part_groups,others,position,first,count,group,query,held,cap,piece,group_inside,query_inside,
        others_fit(position,groups,others,cap,first,count,group,piece,group_inside,count_positive(count,query,query_inside)),apart)
      Equal.trans(Tile.Tile8,groups_expected(~PContext,~pvalue,pcontext,1n+rest,start,width,groups,part_groups,others,position,first,count,group,query,held),groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,held),held,
        Equal.cong(Tile.Tile8,Tile.Tile8,cell => groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,cell),range_expected(~PContext,~pvalue,pcontext,Tiled.others_of(position,groups,others,first,count,group),start,width,groups,part_groups,others,position,first,count,group,query,held),held,aside),
        groups_beyond(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,held,cap,piece,query_inside,range_rest(group,rest,U32.to_nat(groups),next,room),range_rest(group,rest,query_group(position,groups,others,first,query),next,beyond)))

# The groups from one up to past the query cell's merge it once.
def groups_hit(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,fuel: Nat,on: Bool,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+query: U32,+held: Tile.Tile8,+cap: U32,+piece: PieceIn<groups,others,cap,first,count>,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(group),fuel),U32.to_nat(groups)) == True{} : Bool},+from: {Nat.is_le(U32.to_nat(group),query_group(position,groups,others,first,query)) == True{} : Bool},
  +within: {Nat.is_lt(query_group(position,groups,others,first,query),Nat.add(U32.to_nat(group),fuel)) == True{} : Bool},+on_is: {Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group)) == on : Bool}) ->
  {groups_expected(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,query,held) == touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,U32.from_nat(query_group(position,groups,others,first,query)),U32.from_nat(query_other(position,groups,others,first,query)),held) : Tile.Tile8}:
  match fuel:
    case 0n:
      Empty.absurd({groups_expected(~PContext,~pvalue,pcontext,0n,start,width,groups,part_groups,others,position,first,count,group,query,held) == touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,U32.from_nat(query_group(position,groups,others,first,query)),U32.from_nat(query_other(position,groups,others,first,query)),held) : Tile.Tile8},range_none(group,query_group(position,groups,others,first,query),from,within))
    case 1n+ +rest:
      match on:
        case True{}:
          +grid = piece_grid(groups,others,cap,first,count,piece)
          +next = range_next(group,rest,groups,cap,cap_fits(groups,others,cap,grid),groups_room(groups,others,cap,grid),room)
          +group_inside = Order.room_index(U32.to_nat(group),rest,U32.to_nat(groups),room)
          +reached = touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,group,U32.from_nat(query_other(position,groups,others,first,query)),held)
          +hit = range_hit(~PContext,~pvalue,pcontext,Tiled.others_of(position,groups,others,first,count,group),start,width,groups,part_groups,others,position,first,count,group,query,held,cap,piece,group_inside,query_inside,
            others_fit(position,groups,others,cap,first,count,group,piece,group_inside,count_positive(count,query,query_inside)),
            others_cover(position,groups,others,cap,first,count,group,piece,group_inside,query,query_inside,on_is),on_is)
          Equal.trans(Tile.Tile8,groups_expected(~PContext,~pvalue,pcontext,1n+rest,start,width,groups,part_groups,others,position,first,count,group,query,held),groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,reached),touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,U32.from_nat(query_group(position,groups,others,first,query)),U32.from_nat(query_other(position,groups,others,first,query)),held),
            Equal.cong(Tile.Tile8,Tile.Tile8,cell => groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,cell),range_expected(~PContext,~pvalue,pcontext,Tiled.others_of(position,groups,others,first,count,group),start,width,groups,part_groups,others,position,first,count,group,query,held),reached,hit),
            Equal.trans(Tile.Tile8,groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,reached),reached,touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,U32.from_nat(query_group(position,groups,others,first,query)),U32.from_nat(query_other(position,groups,others,first,query)),held),
              groups_past(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,reached,cap,piece,query_inside,range_rest(group,rest,U32.to_nat(groups),next,room),
                range_past(group,query_group(position,groups,others,first,query),next,equal_le(query_group(position,groups,others,first,query),U32.to_nat(group),on_is))),
              Equal.cong(U32,Tile.Tile8,word => touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,word,U32.from_nat(query_other(position,groups,others,first,query)),held),group,U32.from_nat(query_group(position,groups,others,first,query)),word_same(group,query_group(position,groups,others,first,query),on_is))))
        case False{}:
          +grid = piece_grid(groups,others,cap,first,count,piece)
          +next = range_next(group,rest,groups,cap,cap_fits(groups,others,cap,grid),groups_room(groups,others,cap,grid),room)
          +group_inside = Order.room_index(U32.to_nat(group),rest,U32.to_nat(groups),room)
          +aside = range_aside(~PContext,~pvalue,pcontext,Tiled.others_of(position,groups,others,first,count,group),start,width,groups,part_groups,others,position,first,count,group,query,held,cap,piece,group_inside,query_inside,
            others_fit(position,groups,others,cap,first,count,group,piece,group_inside,count_positive(count,query,query_inside)),on_is)
          Equal.trans(Tile.Tile8,groups_expected(~PContext,~pvalue,pcontext,1n+rest,start,width,groups,part_groups,others,position,first,count,group,query,held),groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,held),touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,U32.from_nat(query_group(position,groups,others,first,query)),U32.from_nat(query_other(position,groups,others,first,query)),held),
            Equal.cong(Tile.Tile8,Tile.Tile8,cell => groups_expected(~PContext,~pvalue,pcontext,rest,start,width,groups,part_groups,others,position,first,count,(group + 1 : U32),query,cell),range_expected(~PContext,~pvalue,pcontext,Tiled.others_of(position,groups,others,first,count,group),start,width,groups,part_groups,others,position,first,count,group,query,held),held,aside),
            groups_hit(~PContext,~pvalue,pcontext,rest,Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat((group + 1 : U32))),start,width,groups,part_groups,others,position,first,count,
              (group + 1 : U32),query,held,cap,piece,query_inside,range_rest(group,rest,U32.to_nat(groups),next,room),range_on(group,query_group(position,groups,others,first,query),next,from,on_is),
              range_within(group,rest,query_group(position,groups,others,first,query),next,within),{==}))

def groups_upper(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,high: Bool,+fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+query: U32,+held: Tile.Tile8,+cap: U32,+piece: PieceIn<groups,others,cap,first,count>,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(group),fuel),U32.to_nat(groups)) == True{} : Bool},+from: {Nat.is_le(U32.to_nat(group),query_group(position,groups,others,first,query)) == True{} : Bool},
  +high_is: {Nat.is_lt(query_group(position,groups,others,first,query),Nat.add(U32.to_nat(group),fuel)) == high : Bool}) -> {groups_expected(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,query,held) == Bool.pick(Tile.Tile8,high,touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,U32.from_nat(query_group(position,groups,others,first,query)),U32.from_nat(query_other(position,groups,others,first,query)),held),held) : Tile.Tile8}:
  match high:
    case True{}: groups_hit(~PContext,~pvalue,pcontext,fuel,Nat.is_eq(query_group(position,groups,others,first,query),U32.to_nat(group)),start,width,groups,part_groups,others,position,first,count,group,query,held,cap,piece,query_inside,room,from,high_is,{==})
    case False{}: groups_beyond(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,query,held,cap,piece,query_inside,room,Order.not_less(query_group(position,groups,others,first,query),Nat.add(U32.to_nat(group),fuel),high_is))

def groups_cases(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,low: Bool,high: Bool,+fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+query: U32,+held: Tile.Tile8,+cap: U32,+piece: PieceIn<groups,others,cap,first,count>,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(group),fuel),U32.to_nat(groups)) == True{} : Bool},+low_is: {Nat.is_le(U32.to_nat(group),query_group(position,groups,others,first,query)) == low : Bool},
  +high_is: {Nat.is_lt(query_group(position,groups,others,first,query),Nat.add(U32.to_nat(group),fuel)) == high : Bool}) -> {groups_expected(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,query,held) == Bool.pick(Tile.Tile8,low && high,touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,U32.from_nat(query_group(position,groups,others,first,query)),U32.from_nat(query_other(position,groups,others,first,query)),held),held) : Tile.Tile8}:
  match low:
    case True{}: groups_upper(~PContext,~pvalue,pcontext,high,fuel,start,width,groups,part_groups,others,position,first,count,group,query,held,cap,piece,query_inside,room,low_is,high_is)
    case False{}:
      groups_past(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,query,held,cap,piece,query_inside,room,Order.ge_false(query_group(position,groups,others,first,query),U32.to_nat(group),
        Equal.trans(Bool,Nat.is_ge(query_group(position,groups,others,first,query),U32.to_nat(group)),Nat.is_le(U32.to_nat(group),query_group(position,groups,others,first,query)),False{},Order.ge_le(query_group(position,groups,others,first,query),U32.to_nat(group)),low_is)))

# A query cell of the target piece after the groups from group on: merged over
# what it held when its group is one of them, what it held otherwise.
def groups_closed(~PContext: Data,~pvalue: PContext -> U32 -> Tile.Tile8,+pcontext: PContext,+fuel: Nat,+start: U32,+width: U32,+groups: U32,+part_groups: U32,+others: U32,+position: Grid.Position,+first: U32,+count: U32,+group: U32,+query: U32,+held: Tile.Tile8,+cap: U32,+piece: PieceIn<groups,others,cap,first,count>,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(group),fuel),U32.to_nat(groups)) == True{} : Bool}) ->
  {groups_expected(~PContext,~pvalue,pcontext,fuel,start,width,groups,part_groups,others,position,first,count,group,query,held) == Bool.pick(Tile.Tile8,Nat.is_le(U32.to_nat(group),query_group(position,groups,others,first,query)) && Nat.is_lt(query_group(position,groups,others,first,query),Nat.add(U32.to_nat(group),fuel)),touched_value(~PContext,~pvalue,pcontext,start,width,part_groups,others,position,U32.from_nat(query_group(position,groups,others,first,query)),U32.from_nat(query_other(position,groups,others,first,query)),held),held) : Tile.Tile8}:
  groups_cases(~PContext,~pvalue,pcontext,Nat.is_le(U32.to_nat(group),query_group(position,groups,others,first,query)),Nat.is_lt(query_group(position,groups,others,first,query),Nat.add(U32.to_nat(group),fuel)),fuel,start,width,groups,part_groups,others,position,first,count,group,query,held,cap,piece,query_inside,room,{==},{==})
