# A split gathers every target piece from the source pieces its task holds:
# cell by cell through the held pieces, or, with the groups innermost, run by
# run from the one piece that holds a run whole. The model of the holding and
# the buffer it leaves.

import Base
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_move.bend as Move
import ../tensor_tiled.bend as Tiled
import ./storage_buffer_model.bend as Buffer
import ./tensor_move_readback.bend as Held
import ./tensor_moved.bend as Moved
import ./tensor_relane_block.bend as Block
import ./tensor_split_value.bend as SplitValue
import ./tensor_lanes_write_gather.bend as Merge
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./boolean_logic.bend as Logic
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_pieces_content.bend as Content
import ../kernel_gemm.bend as KernelGemm
import ./u32_extrema.bend as Extrema
import ./u32_division.bend as U32Division
import ./tensor_allocation_depth.bend as Depth
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_placed.bend as Placed
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges

# ---- The model of a target cell ----

# The holding a target cell leaves: the source cell of its first lane
# resolved, then the next when its lanes shift and another follows.
def spotted_holding(+width: U32,+source_groups: U32,+spot: Tiled.Spot,+holding: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  +middle = Block.held_moved(holding,SplitValue.spot_read(width,spot))
  Bool.pick(Held.Holding<Tile.Tile8>,SplitValue.spot_further(width,source_groups,spot),Block.held_moved(middle,SplitValue.spot_high(width,spot)),middle)

# The next source cell a target cell reads, +0 when it reads none.
def spotted_high(+width: U32,+source_groups: U32,+spot: Tiled.Spot,+holding: Held.Holding<Tile.Tile8>) -> Tile.Tile8:
  +middle = Block.held_moved(holding,SplitValue.spot_read(width,spot))
  Bool.pick(Tile.Tile8,SplitValue.spot_further(width,source_groups,spot),Block.held_value(middle,SplitValue.spot_high(width,spot)),Block.zero())

# The target cell from the source cells it reads.
def spotted_value(+width: U32,+source_groups: U32,+spot: Tiled.Spot,+holding: Held.Holding<Tile.Tile8>) -> Tile.Tile8:
  SplitValue.spot_cell(width,spot,Block.held_value(holding,SplitValue.spot_read(width,spot)),spotted_high(width,source_groups,spot,holding))

# A target cell of packed pieces is the packed holding it leaves and its
# value.
def spotted_packed(+width: U32,+source_groups: U32,+spot: Tiled.Spot,+holding: Held.Holding<Tile.Tile8>) ->
  {Tiled.split_spotted(width,source_groups,spot,Held.pack(~Tile.Tile8,holding)) ==
    (Held.pack(~Tile.Tile8,spotted_holding(width,source_groups,spot,holding)),spotted_value(width,source_groups,spot,holding)) : Move.Holding<Tile.Tile8> & Tile.Tile8}:
  match spot:
    case Tiled.Spot{+anchor,+group,+base,+step}:
      +first = (anchor * width + group * 8 : U32)
      +cell = (first / 8 : U32)
      +shift = (first % 8 : U32)
      +source = (base + cell * step : U32)
      +further = U32.is_lt(0,shift) && U32.is_lt((cell + 1 : U32),source_groups)
      +middle = Block.held_moved(holding,source)
      +low = Block.held_value(holding,source)
      %Equal.sym(Move.Holding<Tile.Tile8> & Tile.Tile8,Tiled.held_cell(Held.pack(~Tile.Tile8,holding),source),(Held.pack(~Tile.Tile8,middle),low),Block.held_packed(holding,source)) :
        {Tiled.split_low(shift,(group * 8 : U32),width,source,step,further,_) ==
          (Held.pack(~Tile.Tile8,Bool.pick(Held.Holding<Tile.Tile8>,further,Block.held_moved(middle,(source + 1 * step : U32)),middle)),
            SplitValue.spot_cell(width,Tiled.Spot{anchor,group,base,step},low,Bool.pick(Tile.Tile8,further,Block.held_value(middle,(source + 1 * step : U32)),Block.zero()))) :
          Move.Holding<Tile.Tile8> & Tile.Tile8}
      %Equal.sym(Move.Holding<Tile.Tile8> & Tile.Tile8,Tiled.block_read(~Move.Holding<Tile.Tile8>,~Tiled.held_cell,further,source,step,1,Held.pack(~Tile.Tile8,middle)),
          (Held.pack(~Tile.Tile8,Bool.pick(Held.Holding<Tile.Tile8>,further,Block.held_moved(middle,(source + 1 * step : U32)),middle)),
            Bool.pick(Tile.Tile8,further,Block.held_value(middle,(source + 1 * step : U32)),Block.zero())),
          Block.wanted_packed(~Move.Holding<Tile.Tile8>,~Tiled.held_cell,~Held.Holding<Tile.Tile8>,~Block.held_pack,~Block.held_moved,~Block.held_value,~Block.held_packed,
            further,source,step,1,middle)) :
        {Tiled.split_high(shift,(group * 8 : U32),width,low,_) ==
          (Held.pack(~Tile.Tile8,Bool.pick(Held.Holding<Tile.Tile8>,further,Block.held_moved(middle,(source + 1 * step : U32)),middle)),
            SplitValue.spot_cell(width,Tiled.Spot{anchor,group,base,step},low,Bool.pick(Tile.Tile8,further,Block.held_value(middle,(source + 1 * step : U32)),Block.zero()))) :
          Move.Holding<Tile.Tile8> & Tile.Tile8}
      {==}

# The spot of a target cell of a split.
def spot_at(+split: Tiled.Split,+index: U32) -> Tiled.Spot:
  Tiled.Split{+outer,+anchors,+width,+after,+groups,+source_groups,+position} = split
  Tiled.split_spot(position,outer,anchors,after,groups,source_groups,index)

def split_width(+split: Tiled.Split) -> U32:
  Tiled.Split{+outer,+anchors,+width,+after,+groups,+source_groups,+position} = split
  width

def split_source_groups(+split: Tiled.Split) -> U32:
  Tiled.Split{+outer,+anchors,+width,+after,+groups,+source_groups,+position} = split
  source_groups

def cell_holding(+split: Tiled.Split,+index: U32,+holding: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  spotted_holding(split_width(split),split_source_groups(split),spot_at(split,index),holding)

def cell_value(+split: Tiled.Split,+index: U32,+holding: Held.Holding<Tile.Tile8>) -> Tile.Tile8:
  spotted_value(split_width(split),split_source_groups(split),spot_at(split,index),holding)

def cell_packed(+split: Tiled.Split,+index: U32,+holding: Held.Holding<Tile.Tile8>) ->
  {Tiled.split_cell(split,index,Held.pack(~Tile.Tile8,holding)) == (Held.pack(~Tile.Tile8,cell_holding(split,index,holding)),cell_value(split,index,holding)) :
    Move.Holding<Tile.Tile8> & Tile.Tile8}:
  match split:
    case Tiled.Split{+outer,+anchors,+width,+after,+groups,+source_groups,+position}:
      spotted_packed(width,source_groups,Tiled.split_spot(position,outer,anchors,after,groups,source_groups,index),holding)

# ---- The model of the cells of a target piece ----

# A target buffer with a cell written.
def written(+index: U32,+cell: Tile.Tile8,+output: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,index,cell,Buffer.pack(~Tile.Tile8,output)))

# A holding and a target buffer, packed.
def packed(+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  (Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,output))

# The holding and the target buffer after the target cells from local index
# index on, each read through the holding and written at its local index.
def cells_holding(fuel: Nat,+split: Tiled.Split,+first: U32,+index: U32,+holding: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  match fuel:
    case 0n: holding
    case 1n+rest: cells_holding(rest,split,first,(index + 1 : U32),cell_holding(split,(first + index : U32),holding))

def cells_output(fuel: Nat,+split: Tiled.Split,+first: U32,+index: U32,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  match fuel:
    case 0n: output
    case 1n+rest:
      +cell = (first + index : U32)
      cells_output(rest,split,first,(index + 1 : U32),cell_holding(split,cell,holding),written(index,cell_value(split,cell,holding),output))

def cells_packed(fuel: Nat,+split: Tiled.Split,+first: U32,+index: U32,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Tiled.split_cells(fuel,split,first,index,packed(holding,output)) ==
    packed(cells_holding(fuel,split,first,index,holding),cells_output(fuel,split,first,index,holding,output)) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +cell = (first + index : U32)
      +next = cell_holding(split,cell,holding)
      +value = cell_value(split,cell,holding)
      stored = Storage.write(~Tile.Tile8,index,value,Buffer.pack(~Tile.Tile8,output))
      +after_holding = cells_holding(rest,split,first,(index + 1 : U32),next)
      +after_output = cells_output(rest,split,first,(index + 1 : U32),next,written(index,value,output))
      %Equal.sym(Move.Holding<Tile.Tile8> & Tile.Tile8,Tiled.split_cell(split,cell,Held.pack(~Tile.Tile8,holding)),(Held.pack(~Tile.Tile8,next),value),cell_packed(split,cell,holding)) :
        {Tiled.split_cells(rest,split,first,(index + 1 : U32),Tiled.split_written(index,Buffer.pack(~Tile.Tile8,output),_)) == packed(after_holding,after_output) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      %Buffer.remodeled(~Tile.Tile8,stored) :
        {Tiled.split_cells(rest,split,first,(index + 1 : U32),(Held.pack(~Tile.Tile8,next),_)) == packed(after_holding,after_output) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      cells_packed(rest,split,first,(index + 1 : U32),next,written(index,value,output))

# ---- The model of a run ----

# A cell of a run from the piece buffer at local index local: the cell of its
# first lane and, when its lanes shift and another source cell follows, the
# next.
def run_value(+width: U32,+source_groups: U32,+shift: U32,+local: U32,+cell: U32,+group: U32,+input: Buffer.Model<Tile.Tile8>) -> Tile.Tile8:
  +further = U32.is_lt(0,shift) && U32.is_lt((cell + 1 : U32),source_groups)
  +first = (group * 8 : U32)
  Tiled.run_masked(U32.is_lt(width,(first + 8 : U32)),first,width,
    Tiled.joined(shift,Block.piece_value(input,local),Bool.pick(Tile.Tile8,further,Block.piece_value(input,((local + 1 : U32) + 0 * 0 : U32)),Block.zero())))

# The target buffer after the cells of a run from local index local of the
# piece buffer, written from local target index target.
def run_cells_output(fuel: Nat,+width: U32,+source_groups: U32,+shift: U32,+local: U32,+cell: U32,+group: U32,+target: U32,+input: Buffer.Model<Tile.Tile8>,
  +output: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  match fuel:
    case 0n: output
    case 1n+rest:
      run_cells_output(rest,width,source_groups,shift,(local + 1 : U32),(cell + 1 : U32),(group + 1 : U32),(target + 1 : U32),input,
        written(target,run_value(width,source_groups,shift,local,cell,group,input),output))

def run_cells_packed(fuel: Nat,+width: U32,+source_groups: U32,+shift: U32,+local: U32,+cell: U32,+group: U32,+target: U32,+input: Buffer.Model<Tile.Tile8>,
  +output: Buffer.Model<Tile.Tile8>) ->
  {Tiled.run_cells(fuel,width,source_groups,shift,local,cell,group,target,(Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,output))) ==
    (Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,run_cells_output(fuel,width,source_groups,shift,local,cell,group,target,input,output))) :
    Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +further = U32.is_lt(0,shift) && U32.is_lt((cell + 1 : U32),source_groups)
      +first = (group * 8 : U32)
      +low = Block.piece_value(input,local)
      +address = ((local + 1 : U32) + 0 * 0 : U32)
      +high = Bool.pick(Tile.Tile8,further,Block.piece_value(input,address),Block.zero())
      +value = run_value(width,source_groups,shift,local,cell,group,input)
      stored = Storage.write(~Tile.Tile8,target,value,Buffer.pack(~Tile.Tile8,output))
      +after = run_cells_output(rest,width,source_groups,shift,(local + 1 : U32),(cell + 1 : U32),(group + 1 : U32),(target + 1 : U32),input,written(target,value,output))
      %Equal.sym(Storage.Buffer<Tile.Tile8> & Tile.Tile8,Tiled.piece_cell(Buffer.pack(~Tile.Tile8,input),local),(Buffer.pack(~Tile.Tile8,input),low),Block.piece_packed(input,local)) :
        {Tiled.run_cells(rest,width,source_groups,shift,(local + 1 : U32),(cell + 1 : U32),(group + 1 : U32),(target + 1 : U32),
            Tiled.run_written(target,Buffer.pack(~Tile.Tile8,output),Tiled.run_low(shift,first,width,local,further,_))) == (Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,after)) :
          Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      %Equal.sym(Storage.Buffer<Tile.Tile8> & Tile.Tile8,Tiled.block_read(~Storage.Buffer<Tile.Tile8>,~Tiled.piece_cell,further,(local + 1 : U32),0,0,Buffer.pack(~Tile.Tile8,input)),
          (Buffer.pack(~Tile.Tile8,Bool.pick(Buffer.Model<Tile.Tile8>,further,input,input)),high),
          Block.wanted_packed(~Storage.Buffer<Tile.Tile8>,~Tiled.piece_cell,~Buffer.Model<Tile.Tile8>,~Block.piece_pack,~Block.piece_kept,~Block.piece_value,~Block.piece_packed,
            further,(local + 1 : U32),0,0,input)) :
        {Tiled.run_cells(rest,width,source_groups,shift,(local + 1 : U32),(cell + 1 : U32),(group + 1 : U32),(target + 1 : U32),
            Tiled.run_written(target,Buffer.pack(~Tile.Tile8,output),Tiled.run_high(shift,first,width,low,_))) == (Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,after)) :
          Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      %Equal.sym(Buffer.Model<Tile.Tile8>,Bool.pick(Buffer.Model<Tile.Tile8>,further,input,input),input,Block.kept_pick(~Buffer.Model<Tile.Tile8>,further,input)) :
        {Tiled.run_cells(rest,width,source_groups,shift,(local + 1 : U32),(cell + 1 : U32),(group + 1 : U32),(target + 1 : U32),(Buffer.pack(~Tile.Tile8,_),stored)) == (Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,after)) :
          Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      %Buffer.remodeled(~Tile.Tile8,stored) :
        {Tiled.run_cells(rest,width,source_groups,shift,(local + 1 : U32),(cell + 1 : U32),(group + 1 : U32),(target + 1 : U32),(Buffer.pack(~Tile.Tile8,input),_)) == (Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,after)) :
          Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      run_cells_packed(rest,width,source_groups,shift,(local + 1 : U32),(cell + 1 : U32),(group + 1 : U32),(target + 1 : U32),input,written(target,value,output))

# The words of a run.
def run_source(+run: Tiled.Run) -> U32:
  Tiled.Run{+source,shift,group,count,cell} = run
  source

def run_shift(+run: Tiled.Run) -> U32:
  Tiled.Run{source,+shift,group,count,cell} = run
  shift

def run_group(+run: Tiled.Run) -> U32:
  Tiled.Run{source,shift,+group,count,cell} = run
  group

def run_cell(+run: Tiled.Run) -> U32:
  Tiled.Run{source,shift,group,count,+cell} = run
  cell

# A run from the head piece of a resolved holding: from its buffer when the
# piece holds every source cell the run reads, otherwise cell by cell.
def run_held_holding(+split: Tiled.Split,+run: Tiled.Run,inside: Bool,+first: U32,+local: U32,+behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+start: U32,
  +span: U32,+below: Held.Stack<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  match inside:
    case True{}: Held.Holding{behind,Held.Held{buffer,start,span,below}}
    case False{}: cells_holding(U32.to_nat(Tiled.run_count(run)),split,first,local,Held.Holding{behind,Held.Held{buffer,start,span,below}})

def run_held_output(+split: Tiled.Split,+run: Tiled.Run,inside: Bool,+first: U32,+local: U32,+behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+start: U32,
  +span: U32,+below: Held.Stack<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  match inside:
    case True{}:
      run_cells_output(U32.to_nat(Tiled.run_count(run)),split_width(split),split_source_groups(split),run_shift(run),(run_source(run) - start : U32),run_cell(run),run_group(run),
        local,buffer,output)
    case False{}: cells_output(U32.to_nat(Tiled.run_count(run)),split,first,local,Held.Holding{behind,Held.Held{buffer,start,span,below}},output)

# The source cells a run reads: one per cell, and the next after the last
# when its lanes shift and another follows.
def run_reads(+source_groups: U32,+run: Tiled.Run) -> U32:
  +count = Tiled.run_count(run)
  (count + Bool.pick(U32,U32.is_lt(0,run_shift(run)) && U32.is_lt((run_cell(run) + count : U32),source_groups),1,0) : U32)

# Whether the head piece holds every source cell of a run.
def run_inside(+source_groups: U32,+run: Tiled.Run,+start: U32,+span: U32) -> Bool:
  +source = run_source(run)
  U32.is_le(start,source) && U32.is_le((source + run_reads(source_groups,run) : U32),(start + span : U32))

def run_resolved_holding(+split: Tiled.Split,+run: Tiled.Run,+first: U32,+local: U32,+holding: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+start,+span,+below}}:
      run_held_holding(split,run,run_inside(split_source_groups(split),run,start,span),first,local,behind,buffer,start,span,below)
    case Held.Holding{+behind,Held.Bottom{}}: cells_holding(U32.to_nat(Tiled.run_count(run)),split,first,local,Held.Holding{behind,Held.Bottom{}})

def run_resolved_output(+split: Tiled.Split,+run: Tiled.Run,+first: U32,+local: U32,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  Buffer.Model<Tile.Tile8>:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+start,+span,+below}}:
      run_held_output(split,run,run_inside(split_source_groups(split),run,start,span),first,local,behind,buffer,start,span,below,output)
    case Held.Holding{+behind,Held.Bottom{}}: cells_output(U32.to_nat(Tiled.run_count(run)),split,first,local,Held.Holding{behind,Held.Bottom{}},output)

def run_applied_holding(+split: Tiled.Split,+run: Tiled.Run,+first: U32,+local: U32,+holding: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  run_resolved_holding(split,run,first,local,Held.resolve(~Tile.Tile8,holding,run_source(run)))

def run_applied_output(+split: Tiled.Split,+run: Tiled.Run,+first: U32,+local: U32,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  run_resolved_output(split,run,first,local,Held.resolve(~Tile.Tile8,holding,run_source(run)),output)

# The run of a target piece from local cell local, and the local cell after it.
def run_of(+split: Tiled.Split,+first: U32,+local: U32,+count: U32) -> Tiled.Run:
  Tiled.split_run(split,(first + local : U32),(first + count : U32))

def run_next(+split: Tiled.Split,+first: U32,+local: U32,+count: U32) -> U32:
  (local + U32.max(Tiled.run_count(run_of(split,first,local,count)),1) : U32)

# The runs of a target piece from local cell local on, while more.
def runs_holding(fuel: Nat,more: Bool,+split: Tiled.Split,+first: U32,+local: U32,+count: U32,+holding: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  match fuel more:
    case 0n more: holding
    case 1n+rest False{}: holding
    case 1n+rest True{}:
      +next = run_next(split,first,local,count)
      runs_holding(rest,U32.is_lt(next,count),split,first,next,count,run_applied_holding(split,run_of(split,first,local,count),first,local,holding))

def runs_output(fuel: Nat,more: Bool,+split: Tiled.Split,+first: U32,+local: U32,+count: U32,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  Buffer.Model<Tile.Tile8>:
  match fuel more:
    case 0n more: output
    case 1n+rest False{}: output
    case 1n+rest True{}:
      +run = run_of(split,first,local,count)
      +next = run_next(split,first,local,count)
      runs_output(rest,U32.is_lt(next,count),split,first,next,count,run_applied_holding(split,run,first,local,holding),run_applied_output(split,run,first,local,holding,output))

def run_held_packed(+split: Tiled.Split,+run: Tiled.Run,inside: Bool,+first: U32,+local: U32,+behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+start: U32,
  +span: U32,+below: Held.Stack<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Tiled.run_held(split,run,inside,first,local,Held.pack_stack(~Tile.Tile8,behind),Buffer.pack(~Tile.Tile8,buffer),start,span,Held.pack_stack(~Tile.Tile8,below),Buffer.pack(~Tile.Tile8,output)) ==
    packed(run_held_holding(split,run,inside,first,local,behind,buffer,start,span,below),run_held_output(split,run,inside,first,local,behind,buffer,start,span,below,output)) :
    Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  match split:
    case Tiled.Split{+outer,+anchors,+width,+after,+groups,+source_groups,+position}:
      match run:
        case Tiled.Run{+source,+shift,+group,+count,+cell}:
          match inside:
            case True{}:
              +local_source = (source - start : U32)
              %Equal.sym(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,
                  Tiled.run_cells(U32.to_nat(count),width,source_groups,shift,local_source,cell,group,local,(Buffer.pack(~Tile.Tile8,buffer),Buffer.pack(~Tile.Tile8,output))),
                  (Buffer.pack(~Tile.Tile8,buffer),Buffer.pack(~Tile.Tile8,run_cells_output(U32.to_nat(count),width,source_groups,shift,local_source,cell,group,local,buffer,output))),
                  run_cells_packed(U32.to_nat(count),width,source_groups,shift,local_source,cell,group,local,buffer,output)) :
                {Move.piece_returned(~Tile.Tile8,Held.pack_stack(~Tile.Tile8,behind),start,span,Held.pack_stack(~Tile.Tile8,below),_) ==
                  packed(Held.Holding{behind,Held.Held{buffer,start,span,below}},
                    run_cells_output(U32.to_nat(count),width,source_groups,shift,local_source,cell,group,local,buffer,output)) :
                  Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
              {==}
            case False{}:
              cells_packed(U32.to_nat(count),Tiled.Split{outer,anchors,width,after,groups,source_groups,position},first,local,
                Held.Holding{behind,Held.Held{buffer,start,span,below}},output)

def run_resolved_packed(+split: Tiled.Split,+run: Tiled.Run,+first: U32,+local: U32,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Tiled.run_resolved(split,run,first,local,Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,output)) ==
    packed(run_resolved_holding(split,run,first,local,holding),run_resolved_output(split,run,first,local,holding,output)) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  match split:
    case Tiled.Split{+outer,+anchors,+width,+after,+groups,+source_groups,+position}:
      match run:
        case Tiled.Run{+source,+shift,+group,+count,+cell}:
          match holding:
            case Held.Holding{+behind,Held.Held{+buffer,+start,+span,+below}}:
              run_held_packed(Tiled.Split{outer,anchors,width,after,groups,source_groups,position},Tiled.Run{source,shift,group,count,cell},
                run_inside(source_groups,Tiled.Run{source,shift,group,count,cell},start,span),first,local,behind,buffer,start,span,below,output)
            case Held.Holding{+behind,Held.Bottom{}}:
              cells_packed(U32.to_nat(count),Tiled.Split{outer,anchors,width,after,groups,source_groups,position},first,local,Held.Holding{behind,Held.Bottom{}},output)

def run_applied_packed(+split: Tiled.Split,+run: Tiled.Run,+first: U32,+local: U32,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Tiled.run_applied(split,run,first,local,packed(holding,output)) ==
    packed(run_applied_holding(split,run,first,local,holding),run_applied_output(split,run,first,local,holding,output)) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  match run:
    case Tiled.Run{+source,+shift,+group,+count,+cell}:
      +resolved = Held.resolve(~Tile.Tile8,holding,source)
      %Equal.sym(Move.Holding<Tile.Tile8>,Move.resolve(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),source),Held.pack(~Tile.Tile8,resolved),Held.resolve_packed(~Tile.Tile8,holding,source)) :
        {Tiled.run_resolved(split,Tiled.Run{source,shift,group,count,cell},first,local,_,Buffer.pack(~Tile.Tile8,output)) ==
          packed(run_resolved_holding(split,Tiled.Run{source,shift,group,count,cell},first,local,resolved),
            run_resolved_output(split,Tiled.Run{source,shift,group,count,cell},first,local,resolved,output)) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      run_resolved_packed(split,Tiled.Run{source,shift,group,count,cell},first,local,resolved,output)

def runs_packed(fuel: Nat,more: Bool,+split: Tiled.Split,+first: U32,+local: U32,+count: U32,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Tiled.split_runs(fuel,more,split,first,local,count,packed(holding,output)) ==
    packed(runs_holding(fuel,more,split,first,local,count,holding),runs_output(fuel,more,split,first,local,count,holding,output)) :
    Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  match fuel more:
    case 0n more: {==}
    case 1n+ +rest False{}: {==}
    case 1n+ +rest True{}:
      +run = run_of(split,first,local,count)
      +next = run_next(split,first,local,count)
      +middle = run_applied_holding(split,run,first,local,holding)
      +middle_output = run_applied_output(split,run,first,local,holding,output)
      %Equal.sym(Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,Tiled.run_applied(split,run,first,local,packed(holding,output)),packed(middle,middle_output),
          run_applied_packed(split,run,first,local,holding,output)) :
        {Tiled.split_runs(rest,U32.is_lt(next,count),split,first,next,count,_) ==
          packed(runs_holding(rest,U32.is_lt(next,count),split,first,next,count,middle),runs_output(rest,U32.is_lt(next,count),split,first,next,count,middle,middle_output)) :
          Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      runs_packed(rest,U32.is_lt(next,count),split,first,next,count,middle,middle_output)

# ---- The model of a gather ----

# The cells of a target piece: by runs with the groups innermost, otherwise
# cell by cell.
def piece_holding(innermost: Bool,+split: Tiled.Split,+first: U32,+count: U32,+holding: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  match innermost:
    case True{}: runs_holding(U32.to_nat(count),U32.is_lt(0,count),split,first,0,count,holding)
    case False{}: cells_holding(U32.to_nat(count),split,first,0,holding)

def piece_output(innermost: Bool,+split: Tiled.Split,+first: U32,+count: U32,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  match innermost:
    case True{}: runs_output(U32.to_nat(count),U32.is_lt(0,count),split,first,0,count,holding,output)
    case False{}: cells_output(U32.to_nat(count),split,first,0,holding,output)

def piece_packed(innermost: Bool,+split: Tiled.Split,+first: U32,+count: U32,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Tiled.split_written_piece(innermost,split,first,count,packed(holding,output)) ==
    packed(piece_holding(innermost,split,first,count,holding),piece_output(innermost,split,first,count,holding,output)) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  match innermost:
    case True{}: runs_packed(U32.to_nat(count),U32.is_lt(0,count),split,first,0,count,holding,output)
    case False{}: cells_packed(U32.to_nat(count),split,first,0,holding,output)

def fresh_output(+count: U32) -> Storage.Buffer<Tile.Tile8>:
  Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),count)

def innermost_of(+split: Tiled.Split) -> Bool:
  Bool.not(Grid.is_outermost(Tiled.split_position(split)))

# The holding and the buffer a split gathers, as a model.
def split_model(+split: Tiled.Split,+components: Tasks.Components,+first: U32,+count: U32,+fill: Tile.Tile8,+holdings: Moved.Holdings<Tile.Tile8>) -> Moved.Gathered<Tile.Tile8>:
  match holdings:
    case Moved.Holds{+holding,+rest}:
      +innermost = innermost_of(split)
      Moved.Gathered{Moved.Holds{piece_holding(innermost,split,first,count,holding),rest},
        piece_output(innermost,split,first,count,holding,Buffer.remodel(~Tile.Tile8,fresh_output(count)))}
    case Moved.Unheld{}:
      Moved.Gathered{Moved.Unheld{},Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))}

# A split gathers what its model says.
def split_gathered(+split: Tiled.Split,+components: Tasks.Components,+first: U32,+count: U32,+fill: Tile.Tile8,+holdings: Moved.Holdings<Tile.Tile8>) ->
  {Tiled.split_gather(split,components,first,count,fill,Moved.pack_holdings(~Tile.Tile8,holdings)) ==
    (Moved.pack_holdings(~Tile.Tile8,Moved.gathered_holdings(~Tile.Tile8,split_model(split,components,first,count,fill,holdings))),
      Buffer.pack(~Tile.Tile8,Moved.gathered_output(~Tile.Tile8,split_model(split,components,first,count,fill,holdings)))) :
    Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  match holdings:
    case Moved.Holds{+holding,+rest}:
      +innermost = innermost_of(split)
      +fresh = Buffer.remodel(~Tile.Tile8,fresh_output(count))
      +after_holding = piece_holding(innermost,split,first,count,holding)
      +after_output = piece_output(innermost,split,first,count,holding,fresh)
      %Buffer.remodeled(~Tile.Tile8,fresh_output(count)) :
        {Tiled.relane_written(Moved.pack_holdings(~Tile.Tile8,rest),Tiled.split_written_piece(innermost,split,first,count,(Held.pack(~Tile.Tile8,holding),_))) == (Tensors.Holds{Held.pack(~Tile.Tile8,after_holding),Moved.pack_holdings(~Tile.Tile8,rest)},Buffer.pack(~Tile.Tile8,after_output)) :
          Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      %Equal.sym(Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,Tiled.split_written_piece(innermost,split,first,count,packed(holding,fresh)),
          packed(piece_holding(innermost,split,first,count,holding),piece_output(innermost,split,first,count,holding,fresh)),
          piece_packed(innermost,split,first,count,holding,fresh)) :
        {Tiled.relane_written(Moved.pack_holdings(~Tile.Tile8,rest),_) == (Tensors.Holds{Held.pack(~Tile.Tile8,after_holding),Moved.pack_holdings(~Tile.Tile8,rest)},Buffer.pack(~Tile.Tile8,after_output)) : Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      {==}
    case Moved.Unheld{}:
      %Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)) :
        {(Tensors.Unheld{},_) == (Tensors.Unheld{},Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)))) :
          Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      {==}

# ---- What a target cell reads ----

# The source cells a target cell reads lie in the pieces held: the cell of
# its first lane, and the next when a lane inside the width needs it, which
# the cell then reads.
type CellHeld<-spans: Held.Spans,-width: U32,-source_groups: U32,-spot: Tiled.Spot> is Data:
  CellHeld{low: Held.SpanHeld(spans,U32.to_nat(SplitValue.spot_read(width,spot))),
    high: Logic.When(SplitValue.spot_needed(width,spot),Held.Both<{SplitValue.spot_further(width,source_groups,spot) == True{} : Bool},
      Held.SpanHeld(spans,U32.to_nat(SplitValue.spot_high(width,spot)))>)}

# A read, wanted or not, keeps the pieces in their order with their values.
def picked_kept(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+wanted: Bool,+address: U32,+holding: Held.Holding<Tile.Tile8>,
  +stack: Held.Stack<Tile.Tile8>,+kept: Merge.PartKept<Context,value,context,stack,holding>) ->
  Merge.PartKept<Context,value,context,stack,Merge.read_step(wanted,address,holding)>:
  Merge.PartKept{+ordered,+values} = kept
  Merge.PartKept{Equal.trans(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,Merge.read_step(wanted,address,holding)),Held.order(~Tile.Tile8,holding),stack,
      Block.picked_order(~Tile.Tile8,wanted,address,holding),ordered),
    Block.picked_values(~Tile.Tile8,~Context,~value,context,wanted,address,holding,values)}

# A read the cell needs reads its held source cell.
def needed_value(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+further: Bool,+address: U32,+holding: Held.Holding<Tile.Tile8>,
  +stack: Held.Stack<Tile.Tile8>,+facts: Merge.PartSorted<stack>,+kept: Merge.PartKept<Context,value,context,stack,holding>,
  +held: Held.Both<{further == True{} : Bool},Held.SpanHeld(Held.spans(~Tile.Tile8,stack),U32.to_nat(address))>) ->
  {Merge.read_value(further,address,holding) == value(context,address) : Tile.Tile8}:
  Merge.PartSorted{+low,+high,+capacity,+capacity_ok,+bounded,+sorted} = facts
  Merge.PartKept{+ordered,+values} = kept
  Held.Both{+further_true,+span} = held
  +back = Equal.sym(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,holding),stack,ordered)
  %Equal.sym(Bool,further,True{},further_true) : {Bool.pick(Tile.Tile8,_,Block.held_value(holding,address),Tiled.zero_tile()) == value(context,address) : Tile.Tile8}
  Block.picked_cell(~Context,~value,context,True{},address,holding,low,high,capacity,capacity_ok,bounded,
    Held.sorted_moved(~Tile.Tile8,stack,Held.order(~Tile.Tile8,holding),low,high,back,sorted),
    {%back : Held.SpanHeld(Held.spans(~Tile.Tile8,_),U32.to_nat(address)) span : Held.SpanHeld(Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)),U32.to_nat(address))},
    values)

# The next source cell a target cell reads gives its value when the cell
# needs it.
def high_value(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,need: Bool,+further: Bool,+address: U32,+holding: Held.Holding<Tile.Tile8>,
  +stack: Held.Stack<Tile.Tile8>,+facts: Merge.PartSorted<stack>,+kept: Merge.PartKept<Context,value,context,stack,holding>,
  +held: Logic.When(need,Held.Both<{further == True{} : Bool},Held.SpanHeld(Held.spans(~Tile.Tile8,stack),U32.to_nat(address))>)) ->
  Logic.When(need,{Merge.read_value(further,address,holding) == value(context,address) : Tile.Tile8}):
  match need:
    case True{}: needed_value(~Context,~value,context,further,address,holding,stack,facts,kept,held)
    case False{}: Unit{}

# The cells a target cell reads give their values, and the pieces stay in
# their order with their values.
def spotted_kept(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+width: U32,+source_groups: U32,+spot: Tiled.Spot,+holding: Held.Holding<Tile.Tile8>,
  +stack: Held.Stack<Tile.Tile8>,+facts: Merge.PartSorted<stack>,+kept: Merge.PartKept<Context,value,context,stack,holding>,
  +held: CellHeld<Held.spans(~Tile.Tile8,stack),width,source_groups,spot>) ->
  Held.Both<Merge.PartKept<Context,value,context,stack,spotted_holding(width,source_groups,spot,holding)>,
    Held.Both<{Block.held_value(holding,SplitValue.spot_read(width,spot)) == value(context,SplitValue.spot_read(width,spot)) : Tile.Tile8},
      Logic.When(SplitValue.spot_needed(width,spot),{spotted_high(width,source_groups,spot,holding) == value(context,SplitValue.spot_high(width,spot)) : Tile.Tile8})>>:
  CellHeld{+low_held,+high_held} = held
  +read = SplitValue.spot_read(width,spot)
  +next = SplitValue.spot_high(width,spot)
  +further = SplitValue.spot_further(width,source_groups,spot)
  +middle = Block.held_moved(holding,read)
  +first_read = Merge.read_kept(~Context,~value,context,True{},read,holding,stack,facts,kept,low_held)
  +middle_kept = Held.first_of(Merge.PartKept<Context,value,context,stack,middle>,{Block.held_value(holding,read) == value(context,read) : Tile.Tile8},first_read)
  Held.Both{picked_kept(~Context,~value,context,further,next,middle,stack,middle_kept),
    Held.Both{Held.second_of(Merge.PartKept<Context,value,context,stack,middle>,{Block.held_value(holding,read) == value(context,read) : Tile.Tile8},first_read),
      high_value(~Context,~value,context,SplitValue.spot_needed(width,spot),further,next,middle,stack,facts,middle_kept,high_held)}}

# What a target cell of a target piece holds: for every source cell it reads
# at the value of the source and every next one that holds it when a lane
# needs it, the target value at its index.
def CellLaw(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,+context: Context,+target: Target,+split: Tiled.Split,
  +first: U32,+count: U32) -> Type:
  @+local: U32 -> @+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool} -> @+low: Tile.Tile8 -> @+high: Tile.Tile8 ->
  @+low_exact: {low == value(context,SplitValue.spot_read(split_width(split),spot_at(split,(first + local : U32)))) : Tile.Tile8} ->
  @+high_exact: Logic.When(SplitValue.spot_needed(split_width(split),spot_at(split,(first + local : U32))),
    {high == value(context,SplitValue.spot_high(split_width(split),spot_at(split,(first + local : U32)))) : Tile.Tile8}) ->
  {SplitValue.spot_cell(split_width(split),spot_at(split,(first + local : U32)),low,high) == expected(target,(first + local : U32)) : Tile.Tile8}

# Every target cell of a target piece reads source cells the pieces hold.
def CellsHeld(+spans: Held.Spans,+split: Tiled.Split,+first: U32,+count: U32) -> Type:
  @+local: U32 -> @+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool} ->
  CellHeld<spans,split_width(split),split_source_groups(split),spot_at(split,(first + local : U32))>

# A target cell read through the held pieces holds its target value, and the
# pieces stay in their order with their values.
def cell_kept(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,~context: Context,~target: Target,~split: Tiled.Split,
  ~first: U32,~count: U32,~cell_law: CellLaw(~Context,~value,~Target,~expected,context,target,split,first,count),+local: U32,
  +below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},+holding: Held.Holding<Tile.Tile8>,+stack: Held.Stack<Tile.Tile8>,
  +facts: Merge.PartSorted<stack>,+kept: Merge.PartKept<Context,value,context,stack,holding>,
  +held: CellHeld<Held.spans(~Tile.Tile8,stack),split_width(split),split_source_groups(split),spot_at(split,(first + local : U32))>) ->
  Held.Both<Merge.PartKept<Context,value,context,stack,cell_holding(split,(first + local : U32),holding)>,
    {cell_value(split,(first + local : U32),holding) == expected(target,(first + local : U32)) : Tile.Tile8}>:
  +width = split_width(split)
  +source_groups = split_source_groups(split)
  +spot = spot_at(split,(first + local : U32))
  +read = spotted_kept(~Context,~value,context,width,source_groups,spot,holding,stack,facts,kept,held)
  +values = Held.second_of(Merge.PartKept<Context,value,context,stack,spotted_holding(width,source_groups,spot,holding)>,
    Held.Both<{Block.held_value(holding,SplitValue.spot_read(width,spot)) == value(context,SplitValue.spot_read(width,spot)) : Tile.Tile8},
      Logic.When(SplitValue.spot_needed(width,spot),{spotted_high(width,source_groups,spot,holding) == value(context,SplitValue.spot_high(width,spot)) : Tile.Tile8})>,read)
  Held.Both{Held.first_of(Merge.PartKept<Context,value,context,stack,spotted_holding(width,source_groups,spot,holding)>,
      Held.Both<{Block.held_value(holding,SplitValue.spot_read(width,spot)) == value(context,SplitValue.spot_read(width,spot)) : Tile.Tile8},
        Logic.When(SplitValue.spot_needed(width,spot),{spotted_high(width,source_groups,spot,holding) == value(context,SplitValue.spot_high(width,spot)) : Tile.Tile8})>,read),
    cell_law(local,below,Block.held_value(holding,SplitValue.spot_read(width,spot)),spotted_high(width,source_groups,spot,holding),
      Held.first_of({Block.held_value(holding,SplitValue.spot_read(width,spot)) == value(context,SplitValue.spot_read(width,spot)) : Tile.Tile8},
        Logic.When(SplitValue.spot_needed(width,spot),{spotted_high(width,source_groups,spot,holding) == value(context,SplitValue.spot_high(width,spot)) : Tile.Tile8}),values),
      Held.second_of({Block.held_value(holding,SplitValue.spot_read(width,spot)) == value(context,SplitValue.spot_read(width,spot)) : Tile.Tile8},
        Logic.When(SplitValue.spot_needed(width,spot),{spotted_high(width,source_groups,spot,holding) == value(context,SplitValue.spot_high(width,spot)) : Tile.Tile8}),values))}

# ---- What the cells of a target piece leave ----

# A cell of the target piece once the cells below done are written: its
# target value below done, +0 from done on.
def done_cell(~Target: Data,~expected: Target -> U32 -> Tile.Tile8,+target: Target,+first: U32,+query: U32,+done: Nat) -> Tile.Tile8:
  Bool.pick(Tile.Tile8,Nat.is_lt(U32.to_nat(query),done),expected(target,(first + query : U32)),Tiled.zero_tile())

# The pieces in their order with their values, the target piece of its
# shape, and a query cell.
type CellsFacts<-Context: Data,-value: Context -> U32 -> Tile.Tile8,-context: Context,-stack: Held.Stack<Tile.Tile8>,-depth: Nat,-count: U32,-query: U32,
  -holding: Held.Holding<Tile.Tile8>,-output: Buffer.Model<Tile.Tile8>,-expected: Tile.Tile8> is Data:
  CellsFacts{kept: Merge.PartKept<Context,value,context,stack,holding>,shape: Merge.OutputShape<depth,count,output>,
    read: {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),query) == expected : Tile.Tile8}}

# One more cell written below a query: the query holds its value from the
# next cell on when it is the cell written.
def prefix_step(+query: Nat,+done: Nat,+value: Tile.Tile8,+other: Tile.Tile8) ->
  {Bool.pick(Tile.Tile8,Nat.is_eq(done,query),value,Bool.pick(Tile.Tile8,Nat.is_lt(query,done),value,other)) ==
    Bool.pick(Tile.Tile8,Nat.is_lt(query,1n+done),value,other) : Tile.Tile8}:
  match query done:
    case 0n 0n: {==}
    case 0n 1n+ +rest: {==}
    case 1n+ +lower 0n:
      %Equal.sym(Bool,Nat.is_lt(lower,0n),False{},Order.not_below(0n,lower,Order.zero_le(lower))) :
        {other == Bool.pick(Tile.Tile8,_,value,other) : Tile.Tile8}
      {==}
    case 1n+ +lower 1n+ +rest: prefix_step(lower,rest,value,other)

# A cell written at local index local, where the cells below it are written,
# holds its target value: the query holds what it held, or the target value
# when it is the cell written.
def written_query(~Target: Data,~expected: Target -> U32 -> Tile.Tile8,+target: Target,+first: U32,+local: U32,+done: Nat,
  +done_exact: {U32.to_nat(local) == done : Nat},+query: U32,+cell: Tile.Tile8,+cell_exact: {cell == expected(target,(first + local : U32)) : Tile.Tile8},
  +held: Tile.Tile8,+held_exact: {held == done_cell(~Target,~expected,target,first,query,done) : Tile.Tile8}) ->
  {Bool.pick(Tile.Tile8,U32.is_eq(local,query),cell,held) == done_cell(~Target,~expected,target,first,query,1n+done) : Tile.Tile8}:
  +wanted = expected(target,(first + query : U32))
  %Equal.sym(Tile.Tile8,cell,expected(target,(first + local : U32)),cell_exact) :
    {Bool.pick(Tile.Tile8,U32.is_eq(local,query),_,held) == done_cell(~Target,~expected,target,first,query,1n+done) : Tile.Tile8}
  %Equal.sym(Tile.Tile8,held,done_cell(~Target,~expected,target,first,query,done),held_exact) :
    {Bool.pick(Tile.Tile8,U32.is_eq(local,query),expected(target,(first + local : U32)),_) == done_cell(~Target,~expected,target,first,query,1n+done) : Tile.Tile8}
  %Equal.sym(Bool,U32.is_eq(local,query),Nat.is_eq(U32.to_nat(local),U32.to_nat(query)),Comparison.eq(local,query)) :
    {Bool.pick(Tile.Tile8,_,expected(target,(first + local : U32)),done_cell(~Target,~expected,target,first,query,done)) ==
      done_cell(~Target,~expected,target,first,query,1n+done) : Tile.Tile8}
  %Equal.sym(Nat,U32.to_nat(local),done,done_exact) :
    {Bool.pick(Tile.Tile8,Nat.is_eq(_,U32.to_nat(query)),expected(target,(first + local : U32)),done_cell(~Target,~expected,target,first,query,done)) ==
      done_cell(~Target,~expected,target,first,query,1n+done) : Tile.Tile8}
  Equal.trans(Tile.Tile8,Bool.pick(Tile.Tile8,Nat.is_eq(done,U32.to_nat(query)),expected(target,(first + local : U32)),done_cell(~Target,~expected,target,first,query,done)),
    Bool.pick(Tile.Tile8,Nat.is_eq(done,U32.to_nat(query)),wanted,done_cell(~Target,~expected,target,first,query,done)),
    done_cell(~Target,~expected,target,first,query,1n+done),
    Merge.tile_pick_when(Nat.is_eq(done,U32.to_nat(query)),expected(target,(first + local : U32)),wanted,done_cell(~Target,~expected,target,first,query,done),
      hit => Equal.cong(U32,Tile.Tile8,index => expected(target,(first + index : U32)),local,query,
        Arithmetic.equal_of_is_eq(local,query,
          Equal.trans(Bool,U32.is_eq(local,query),Nat.is_eq(U32.to_nat(local),U32.to_nat(query)),True{},Comparison.eq(local,query),
            {%Equal.sym(Nat,U32.to_nat(local),done,done_exact) : {Nat.is_eq(_,U32.to_nat(query)) == True{} : Bool} hit :
              {Nat.is_eq(U32.to_nat(local),U32.to_nat(query)) == True{} : Bool}})))),
    prefix_step(U32.to_nat(query),done,wanted,Tiled.zero_tile()))

# A cell written at local index local below count keeps the target piece of
# its shape and holds its target value at the query.
def written_facts(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,+context: Context,+target: Target,
  +first: U32,+count: U32,+local: U32,+done: Nat,+done_exact: {U32.to_nat(local) == done : Nat},+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},
  +query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+stack: Held.Stack<Tile.Tile8>,+depth: Nat,
  +holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +facts: CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,done)>,
  +next: Held.Holding<Tile.Tile8>,+next_kept: Merge.PartKept<Context,value,context,stack,next>,+cell: Tile.Tile8,
  +cell_exact: {cell == expected(target,(first + local : U32)) : Tile.Tile8}) ->
  CellsFacts<Context,value,context,stack,depth,count,query,next,written(local,cell,output),done_cell(~Target,~expected,target,first,query,1n+done)>:
  CellsFacts{+kept,+shape,+read} = facts
  Merge.OutputShape{+balanced,+limit,+fits,+sized} = shape
  +tree = Buffer.tree(~Tile.Tile8,output)
  +stored = Tree.write(Tile.Tile8,tree,local,cell)
  +back = Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,written(local,cell,output)),stored,Merge.written_tree(local,cell,output))
  +capacity = Tree.capacity(Tile.Tile8,tree)
  CellsFacts{next_kept,
    Merge.OutputShape{{%back : Shape.Balanced(Tile.Tile8,depth,_)
          Shape.store(Tile.Tile8,depth,tree,capacity,U32.and(local,U32.sub(capacity,1)),cell,balanced) :
          Shape.Balanced(Tile.Tile8,depth,Buffer.tree(~Tile.Tile8,written(local,cell,output)))},
        limit,
        {%back : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,_))) == True{} : Bool}
          %Equal.sym(U32,Tree.capacity(Tile.Tile8,stored),capacity,StorageReadback.write_capacity(~Tile.Tile8,tree,local,cell)) :
            {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
          fits : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,written(local,cell,output))))) == True{} : Bool}},
        Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,written(local,cell,output))),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),
          U32.to_nat(count),Merge.written_length(local,cell,output),sized)},
    {%back : {Tree.read(Tile.Tile8,_,query) == done_cell(~Target,~expected,target,first,query,1n+done) : Tile.Tile8}
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,stored,query),Bool.pick(Tile.Tile8,U32.is_eq(local,query),cell,Tree.read(Tile.Tile8,tree,query)),
        done_cell(~Target,~expected,target,first,query,1n+done),
        Block.write_read(depth,tree,balanced,limit,local,cell,query,Merge.word_below(local,count,capacity,below,fits),Merge.word_below(query,count,capacity,query_inside,fits)),
        written_query(~Target,~expected,target,first,local,done,done_exact,query,cell,cell_exact,Tree.read(Tile.Tile8,tree,query),read)) :
      {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,written(local,cell,output)),query) == done_cell(~Target,~expected,target,first,query,1n+done) : Tile.Tile8}}}

# A cell of a target piece read and written: the pieces stay, and the cells
# below the next local index hold their target values.
def cell_facts(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,~context: Context,~target: Target,~split: Tiled.Split,
  ~first: U32,~count: U32,~cell_law: CellLaw(~Context,~value,~Target,~expected,context,target,split,first,count),+local: U32,+done: Nat,
  +done_exact: {U32.to_nat(local) == done : Nat},+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},
  +query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+stack: Held.Stack<Tile.Tile8>,+sorted: Merge.PartSorted<stack>,+depth: Nat,
  +holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +facts: CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,done)>,
  +held: CellHeld<Held.spans(~Tile.Tile8,stack),split_width(split),split_source_groups(split),spot_at(split,(first + local : U32))>) ->
  CellsFacts<Context,value,context,stack,depth,count,query,cell_holding(split,(first + local : U32),holding),
    written(local,cell_value(split,(first + local : U32),holding),output),done_cell(~Target,~expected,target,first,query,1n+done)>:
  CellsFacts{+kept,+shape,+read} = facts
  +index = (first + local : U32)
  +step = cell_kept(~Context,~value,~Target,~expected,~context,~target,~split,~first,~count,~cell_law,local,below,holding,stack,sorted,kept,held)
  written_facts(~Context,~value,~Target,~expected,context,target,first,count,local,done,done_exact,below,query,query_inside,stack,depth,holding,output,
    CellsFacts{kept,shape,read},cell_holding(split,index,holding),
    Held.first_of(Merge.PartKept<Context,value,context,stack,cell_holding(split,index,holding)>,{cell_value(split,index,holding) == expected(target,index) : Tile.Tile8},step),
    cell_value(split,index,holding),
    Held.second_of(Merge.PartKept<Context,value,context,stack,cell_holding(split,index,holding)>,{cell_value(split,index,holding) == expected(target,index) : Tile.Tile8},step))

# The next local index, one past a local index below count.
def next_exact(+local: U32,+done: Nat,+done_exact: {U32.to_nat(local) == done : Nat},+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool},+below: {Nat.is_lt(done,U32.to_nat(count)) == True{} : Bool}) ->
  {U32.to_nat((local + 1 : U32)) == 1n+done : Nat}:
  Equal.trans(Nat,U32.to_nat((local + 1 : U32)),Nat.add(done,1n),1n+done,
    Arithmetic.sum_exact(cap,cap_ok,local,1,done,1n,done_exact,{==},
      {%Equal.sym(Nat,Nat.add(done,1n),1n+done,Addition.successor_width(done)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        Order.transitive(1n+done,U32.to_nat(count),U32.to_nat(cap),Order.strict_follow(done,U32.to_nat(count),below),count_room) :
        {Nat.is_le(Nat.add(done,1n),U32.to_nat(cap)) == True{} : Bool}}),
    Addition.successor_width(done))

# The cells of a target piece from local index local on: the pieces stay, and
# the cells below the local index past them hold their target values.
def cells_facts(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,~context: Context,~target: Target,
  ~split: Tiled.Split,~first: U32,~count: U32,~cell_law: CellLaw(~Context,~value,~Target,~expected,context,target,split,first,count),~stack: Held.Stack<Tile.Tile8>,
  ~cells_held: CellsHeld(Held.spans(~Tile.Tile8,stack),split,first,count),fuel: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool},+local: U32,+done: Nat,+done_exact: {U32.to_nat(local) == done : Nat},+total: Nat,
  +total_exact: {Nat.add(done,fuel) == total : Nat},+total_room: {Nat.is_le(total,U32.to_nat(count)) == True{} : Bool},
  +query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+sorted: Merge.PartSorted<stack>,+depth: Nat,
  +holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +facts: CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,done)>) ->
  CellsFacts<Context,value,context,stack,depth,count,query,cells_holding(fuel,split,first,local,holding),cells_output(fuel,split,first,local,holding,output),
    done_cell(~Target,~expected,target,first,query,total)>:
  match fuel:
    case 0n:
      %total_exact : CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,_)>
      %Equal.sym(Nat,Nat.add(done,0n),done,Addition.zero_right(done)) : CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,_)>
      facts
    case 1n+ +rest:
      +room = {%Equal.sym(Nat,Nat.add(done,1n+rest),total,total_exact) : {Nat.is_le(_,U32.to_nat(count)) == True{} : Bool} total_room : {Nat.is_le(Nat.add(done,1n+rest),U32.to_nat(count)) == True{} : Bool}}
      +done_below = Order.room_index(done,rest,U32.to_nat(count),room)
      +below = {%Equal.sym(Nat,U32.to_nat(local),done,done_exact) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} done_below :
        {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool}}
      +index = (first + local : U32)
      +next = cell_holding(split,index,holding)
      +written_output = written(local,cell_value(split,index,holding),output)
      cells_facts(~Context,~value,~Target,~expected,~context,~target,~split,~first,~count,~cell_law,~stack,~cells_held,rest,cap,cap_ok,count_room,(local + 1 : U32),1n+done,
        next_exact(local,done,done_exact,count,cap,cap_ok,count_room,done_below),total,
        Equal.trans(Nat,Nat.add(1n+done,rest),Nat.add(done,1n+rest),total,Equal.sym(Nat,Nat.add(done,1n+rest),1n+Nat.add(done,rest),Addition.successor_right(done,rest)),total_exact),
        total_room,query,query_inside,sorted,depth,next,written_output,
        cell_facts(~Context,~value,~Target,~expected,~context,~target,~split,~first,~count,~cell_law,local,done,done_exact,below,query,query_inside,stack,sorted,depth,holding,output,facts,
          cells_held(local,below)))

# ---- Words of the runs ----

# A word one past a word of a value below the cap.
def plus_one(+word: U32,+value: Nat,+exact: {U32.to_nat(word) == value : Nat},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_lt(value,U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((word + 1 : U32)) == 1n+value : Nat}:
  Equal.trans(Nat,U32.to_nat((word + 1 : U32)),Nat.add(value,1n),1n+value,
    Arithmetic.sum_exact(cap,cap_ok,word,1,value,1n,exact,{==},
      {%Equal.sym(Nat,Nat.add(value,1n),1n+value,Addition.successor_width(value)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        Order.strict_follow(value,U32.to_nat(cap),room) : {Nat.is_le(Nat.add(value,1n),U32.to_nat(cap)) == True{} : Bool}}),
    Addition.successor_width(value))

# The word of a step one past a word of a base and a step.
def next_step(+word: U32,+base: Nat,+step: Nat,+exact: {U32.to_nat(word) == Nat.add(base,step) : Nat},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_lt(Nat.add(base,step),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((word + 1 : U32)) == Nat.add(base,1n+step) : Nat}:
  Equal.trans(Nat,U32.to_nat((word + 1 : U32)),1n+Nat.add(base,step),Nat.add(base,1n+step),plus_one(word,Nat.add(base,step),exact,cap,cap_ok,room),
    Equal.sym(Nat,Nat.add(base,1n+step),1n+Nat.add(base,step),Addition.successor_right(base,step)))

# A word with nothing added.
def zero_added(+word: U32,+value: Nat,+exact: {U32.to_nat(word) == value : Nat},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(value,U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((word + 0 * 0 : U32)) == value : Nat}:
  Equal.trans(Nat,U32.to_nat((word + 0 * 0 : U32)),Nat.add(value,0n),value,
    Arithmetic.sum_exact(cap,cap_ok,word,(0 * 0 : U32),value,0n,exact,{==},
      {%Equal.sym(Nat,Nat.add(value,0n),value,Addition.zero_right(value)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        room : {Nat.is_le(Nat.add(value,0n),U32.to_nat(cap)) == True{} : Bool}}),
    Addition.zero_right(value))

# ---- What a run reads ----

# The count of the run of a target piece from local cell local.
def run_count_at(+split: Tiled.Split,+first: U32,+local: U32,+count: U32) -> U32:
  Tiled.run_count(run_of(split,first,local,count))

# Whether the lanes of a run shift and another source cell follows its last.
def run_extra(+source_groups: U32,+run: Tiled.Run) -> Bool:
  U32.is_lt(0,run_shift(run)) && U32.is_lt((run_cell(run) + Tiled.run_count(run) : U32),source_groups)

# What a run of a target piece from a local index below count covers: at
# least one cell and none past count, its first source cell the one its first
# target cell reads, and room for its source cells, its groups and the
# source groups.
type RunCounts<-split: Tiled.Split,-first: U32,-local: U32,-count: U32,-cap: U32> is Data:
  RunCounts{positive: {Nat.is_lt(0n,U32.to_nat(run_count_at(split,first,local,count))) == True{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(run_count_at(split,first,local,count))),U32.to_nat(count)) == True{} : Bool},
    first_read: {run_source(run_of(split,first,local,count)) == SplitValue.spot_read(split_width(split),spot_at(split,(first + local : U32))) : U32},
    source_room: {Nat.is_le(Nat.add(U32.to_nat(run_source(run_of(split,first,local,count))),1n+U32.to_nat(run_count_at(split,first,local,count))),U32.to_nat(cap)) == True{} : Bool},
    cells_room: {Nat.is_le(Nat.add(U32.to_nat(run_cell(run_of(split,first,local,count))),1n+U32.to_nat(run_count_at(split,first,local,count))),U32.to_nat(cap)) == True{} : Bool},
    groups_room: {Nat.is_le(Nat.add(U32.to_nat(run_group(run_of(split,first,local,count))),U32.to_nat(run_count_at(split,first,local,count))),U32.to_nat(cap)) == True{} : Bool},
    source_groups_room: {Nat.is_le(U32.to_nat(split_source_groups(split)),U32.to_nat(cap)) == True{} : Bool}}

def RunsCounted(+split: Tiled.Split,+first: U32,+count: U32,+cap: U32) -> Type:
  @+local: U32 -> @+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool} -> RunCounts<split,first,local,count,cap>

# What a cell of a run holds: the source cells of the run at its step joined
# at the shift of the run and cut at the width give the target value of the
# cell, for every next source cell that holds its value when a lane needs it.
def RunLaw(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,+context: Context,+target: Target,+split: Tiled.Split,
  +first: U32,+count: U32) -> Type:
  @+local: U32 -> @+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool} -> @+step: Nat ->
  @+step_inside: {Nat.is_lt(step,U32.to_nat(run_count_at(split,first,local,count))) == True{} : Bool} ->
  @+index: U32 -> @+index_exact: {U32.to_nat(index) == Nat.add(U32.to_nat(local),step) : Nat} ->
  @+group: U32 -> @+group_exact: {U32.to_nat(group) == Nat.add(U32.to_nat(run_group(run_of(split,first,local,count))),step) : Nat} ->
  @+low: Tile.Tile8 -> @+high: Tile.Tile8 ->
  @+low_exact: {low == value(context,U32.from_nat(Nat.add(U32.to_nat(run_source(run_of(split,first,local,count))),step))) : Tile.Tile8} ->
  @+high_exact: Logic.When(SplitValue.spot_needed(split_width(split),spot_at(split,(first + index : U32))),
    {high == value(context,U32.from_nat(1n+Nat.add(U32.to_nat(run_source(run_of(split,first,local,count))),step))) : Tile.Tile8}) ->
  {Tiled.run_masked(U32.is_lt(split_width(split),((group * 8 : U32) + 8 : U32)),(group * 8 : U32),split_width(split),
    Tiled.joined(run_shift(run_of(split,first,local,count)),low,high)) == expected(target,(first + index : U32)) : Tile.Tile8}

# A cell of a run with a lane that needs the next source cell: the lanes of
# the run shift and another source cell follows its cell.
def RunNeeds(+split: Tiled.Split,+first: U32,+count: U32) -> Type:
  @+local: U32 -> @+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool} -> @+step: Nat ->
  @+step_inside: {Nat.is_lt(step,U32.to_nat(run_count_at(split,first,local,count))) == True{} : Bool} ->
  @+index: U32 -> @+index_exact: {U32.to_nat(index) == Nat.add(U32.to_nat(local),step) : Nat} ->
  Logic.When(SplitValue.spot_needed(split_width(split),spot_at(split,(first + index : U32))),
    {Nat.is_lt(0n,U32.to_nat(run_shift(run_of(split,first,local,count)))) &&
      Nat.is_lt(1n+Nat.add(U32.to_nat(run_cell(run_of(split,first,local,count))),step),U32.to_nat(split_source_groups(split))) == True{} : Bool})

# Whether the next source cell of a cell of a run is read, in words and in
# numbers.
def further_exact(+shift: U32,+cell: U32,+cell_value: Nat,+cell_exact: {U32.to_nat(cell) == cell_value : Nat},+source_groups: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_lt(cell_value,U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.is_lt(0,shift) && U32.is_lt((cell + 1 : U32),source_groups) == Nat.is_lt(0n,U32.to_nat(shift)) && Nat.is_lt(1n+cell_value,U32.to_nat(source_groups)) : Bool}:
  %Equal.sym(Bool,U32.is_lt(0,shift),Nat.is_lt(U32.to_nat(0),U32.to_nat(shift)),Comparison.lt(0,shift)) : {_ && U32.is_lt((cell + 1 : U32),source_groups) == Nat.is_lt(0n,U32.to_nat(shift)) && Nat.is_lt(1n+cell_value,U32.to_nat(source_groups)) : Bool}
  %Equal.sym(Bool,U32.is_lt((cell + 1 : U32),source_groups),Nat.is_lt(U32.to_nat((cell + 1 : U32)),U32.to_nat(source_groups)),Comparison.lt((cell + 1 : U32),source_groups)) :
    {Nat.is_lt(0n,U32.to_nat(shift)) && _ == Nat.is_lt(0n,U32.to_nat(shift)) && Nat.is_lt(1n+cell_value,U32.to_nat(source_groups)) : Bool}
  %Equal.sym(Nat,U32.to_nat((cell + 1 : U32)),1n+cell_value,plus_one(cell,cell_value,cell_exact,cap,cap_ok,room)) :
    {Nat.is_lt(0n,U32.to_nat(shift)) && Nat.is_lt(_,U32.to_nat(source_groups)) == Nat.is_lt(0n,U32.to_nat(shift)) && Nat.is_lt(1n+cell_value,U32.to_nat(source_groups)) : Bool}
  {==}

# The source cells a run reads, as a number: one per cell and one more.
def reads_value(extra: Bool,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_lt(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((count + Bool.pick(U32,extra,1,0) : U32)) == Nat.add(U32.to_nat(count),Bool.pick(Nat,extra,1n,0n)) : Nat}:
  match extra:
    case True{}:
      Arithmetic.sum_exact(cap,cap_ok,count,1,U32.to_nat(count),1n,{==},{==},
        {%Equal.sym(Nat,Nat.add(U32.to_nat(count),1n),1n+U32.to_nat(count),Addition.successor_width(U32.to_nat(count))) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
          Order.strict_follow(U32.to_nat(count),U32.to_nat(cap),room) : {Nat.is_le(Nat.add(U32.to_nat(count),1n),U32.to_nat(cap)) == True{} : Bool}})
    case False{}:
      Arithmetic.sum_exact(cap,cap_ok,count,0,U32.to_nat(count),0n,{==},{==},
        {%Equal.sym(Nat,Nat.add(U32.to_nat(count),0n),U32.to_nat(count),Addition.zero_right(U32.to_nat(count))) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
          Bounds.strict_le(U32.to_nat(count),U32.to_nat(cap),room) :
          {Nat.is_le(Nat.add(U32.to_nat(count),0n),U32.to_nat(cap)) == True{} : Bool}})

# A run reads at least its cells, and one more when another source cell
# follows its last.
def reads_extra(+extra: Bool,+count: Nat) -> Logic.When(extra,{Nat.is_lt(count,Nat.add(count,Bool.pick(Nat,extra,1n,0n))) == True{} : Bool}):
  match extra:
    case True{}:
      %Equal.sym(Nat,Nat.add(count,1n),1n+count,Addition.successor_width(count)) : {Nat.is_lt(count,_) == True{} : Bool}
      Order.lt_succ(count,count,Order.reflexive(count))
    case False{}: Unit{}

# The cell after a cell of a run inside the cells the run reads, when the
# cell reads it.
def next_below(fits: Bool,+step: Nat,+count: Nat,+reads: Nat,+cell: Nat,+shift: Nat,+source_groups: Nat,+fits_exact: {Nat.is_lt(1n+step,count) == fits : Bool},
  +step_below: {Nat.is_lt(step,count) == True{} : Bool},+cover: {Nat.is_le(count,reads) == True{} : Bool},
  +last: Logic.When(Nat.is_lt(0n,shift) && Nat.is_lt(Nat.add(cell,count),source_groups),{Nat.is_lt(count,reads) == True{} : Bool}),
  +further: {Nat.is_lt(0n,shift) && Nat.is_lt(1n+Nat.add(cell,step),source_groups) == True{} : Bool}) -> {Nat.is_lt(1n+step,reads) == True{} : Bool}:
  match fits:
    case True{}: Order.lt_le(1n+step,count,reads,fits_exact,cover)
    case False{}:
      +ends = Order.antisymmetric(1n+step,count,Order.strict_follow(step,count,step_below),Order.not_less(1n+step,count,fits_exact))
      +follows = {%Equal.cong(Nat,Nat,next => Nat.add(cell,next),1n+step,count,ends) : {Nat.is_lt(0n,shift) && Nat.is_lt(_,source_groups) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(cell,1n+step),1n+Nat.add(cell,step),Addition.successor_right(cell,step)) :
          {Nat.is_lt(0n,shift) && Nat.is_lt(_,source_groups) == True{} : Bool}
        further : {Nat.is_lt(0n,shift) && Nat.is_lt(Nat.add(cell,count),source_groups) == True{} : Bool}}
      %Equal.sym(Nat,1n+step,count,ends) : {Nat.is_lt(_,reads) == True{} : Bool}
      Logic.required({Nat.is_lt(count,reads) == True{} : Bool},Nat.is_lt(0n,shift) && Nat.is_lt(Nat.add(cell,count),source_groups),last,follows)

# ---- What the cells of a run leave ----

# The values of the head piece of a holding.
def head_leaf(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+start: U32,+span: U32,
  +lower: Held.Stack<Tile.Tile8>,+values: Held.HoldingValues(~Tile.Tile8,~Context,~value,context,Held.Holding{behind,Held.Held{buffer,start,span,lower}})) ->
  Content.Leaf(~Tile.Tile8,~Context,~value,context,start,span,Buffer.tree(~Tile.Tile8,buffer),Buffer.length(~Tile.Tile8,buffer)):
  Held.Both{+back,Held.Both{+leaf,+rest}} = values
  leaf

def at_read(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+first: U32,+input: Tree.Tree<Tile.Tile8>,+length: Nat,+index: Nat,
  +at: Content.At(~Tile.Tile8,~Context,~value,context,first,input,length,index)) ->
  {Tree.read(Tile.Tile8,input,U32.from_nat(index)) == value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Tile.Tile8}:
  KernelGemm.Both{+fits,+read} = at
  read

# A cell of the head piece at a local index inside it holds the source value
# of its index.
def piece_read(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+buffer: Buffer.Model<Tile.Tile8>,+start: U32,+span: U32,
  +leaf: Content.Leaf(~Tile.Tile8,~Context,~value,context,start,span,Buffer.tree(~Tile.Tile8,buffer),Buffer.length(~Tile.Tile8,buffer)),+local: U32,+local_value: Nat,
  +local_exact: {U32.to_nat(local) == local_value : Nat},+inside: {Nat.is_lt(local_value,U32.to_nat(span)) == True{} : Bool}) ->
  {Block.piece_value(buffer,local) == value(context,U32.from_nat(Nat.add(U32.to_nat(start),local_value))) : Tile.Tile8}:
  +tree = Buffer.tree(~Tile.Tile8,buffer)
  %Arithmetic.from_nat_exact(local,local_value,local_exact) : {Tree.read(Tile.Tile8,tree,_) == value(context,U32.from_nat(Nat.add(U32.to_nat(start),local_value))) : Tile.Tile8}
  at_read(~Context,~value,context,start,tree,Buffer.length(~Tile.Tile8,buffer),local_value,
    Content.leaf_cell(~Tile.Tile8,~Context,~value,context,start,span,tree,Buffer.length(~Tile.Tile8,buffer),local_value,leaf,inside))

def facts_kept(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+stack: Held.Stack<Tile.Tile8>,+depth: Nat,+count: U32,+query: U32,
  +holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,+expected: Tile.Tile8,+facts: CellsFacts<Context,value,context,stack,depth,count,query,holding,output,expected>) ->
  Merge.PartKept<Context,value,context,stack,holding>:
  CellsFacts{+kept,+shape,+read} = facts
  kept

# The next source cell of a cell of a run read from the head piece: its
# value when a lane of the cell needs it.
def run_high(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,need: Bool,+further: Bool,+further_value: Bool,
  +further_exact: {further == further_value : Bool},+buffer: Buffer.Model<Tile.Tile8>,+start: U32,+span: U32,
  +leaf: Content.Leaf(~Tile.Tile8,~Context,~value,context,start,span,Buffer.tree(~Tile.Tile8,buffer),Buffer.length(~Tile.Tile8,buffer)),+local: U32,+local_value: Nat,
  +local_exact: {U32.to_nat(local) == local_value : Nat},+source: Nat,+source_exact: {Nat.add(U32.to_nat(start),local_value) == source : Nat},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+span_room: {Nat.is_le(U32.to_nat(span),U32.to_nat(cap)) == True{} : Bool},
  +reads: Logic.When(further_value,{Nat.is_lt(1n+local_value,U32.to_nat(span)) == True{} : Bool}),
  +needs: Logic.When(need,{further_value == True{} : Bool})) ->
  Logic.When(need,{Bool.pick(Tile.Tile8,further,Block.piece_value(buffer,((local + 1 : U32) + 0 * 0 : U32)),Block.zero()) == value(context,U32.from_nat(1n+source)) : Tile.Tile8}):
  match need:
    case True{}:
      +inside = Logic.required({Nat.is_lt(1n+local_value,U32.to_nat(span)) == True{} : Bool},further_value,reads,needs)
      +local_inside = Order.lt_le(local_value,1n+local_value,U32.to_nat(span),Order.lt_succ(local_value,local_value,Order.reflexive(local_value)),
        Bounds.strict_le(1n+local_value,U32.to_nat(span),inside))
      +next_exact = zero_added((local + 1 : U32),1n+local_value,
        plus_one(local,local_value,local_exact,cap,cap_ok,Order.lt_le(local_value,U32.to_nat(span),U32.to_nat(cap),local_inside,span_room)),cap,cap_ok,
        Order.transitive(1n+local_value,U32.to_nat(span),U32.to_nat(cap),Bounds.strict_le(1n+local_value,U32.to_nat(span),inside),span_room))
      %Equal.sym(Bool,further,True{},Equal.trans(Bool,further,further_value,True{},further_exact,needs)) :
        {Bool.pick(Tile.Tile8,_,Block.piece_value(buffer,((local + 1 : U32) + 0 * 0 : U32)),Block.zero()) == value(context,U32.from_nat(1n+source)) : Tile.Tile8}
      %Equal.trans(Nat,Nat.add(U32.to_nat(start),1n+local_value),1n+Nat.add(U32.to_nat(start),local_value),1n+source,Addition.successor_right(U32.to_nat(start),local_value),
            Equal.cong(Nat,Nat,number => 1n+number,Nat.add(U32.to_nat(start),local_value),source,source_exact)) :
        {Block.piece_value(buffer,((local + 1 : U32) + 0 * 0 : U32)) == value(context,U32.from_nat(_)) : Tile.Tile8}
      piece_read(~Context,~value,context,buffer,start,span,leaf,((local + 1 : U32) + 0 * 0 : U32),1n+local_value,next_exact,inside)
    case False{}: Unit{}

# The cells of a run from its step on, read from the head piece that holds
# every source cell they read: the pieces stay, and the cells below the local
# index past them hold their target values.
def run_cells_facts(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,~context: Context,~target: Target,~split: Tiled.Split,
  ~first: U32,~count: U32,~run_law: RunLaw(~Context,~value,~Target,~expected,context,target,split,first,count),~run_needs: RunNeeds(split,first,count),fuel: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool},
  +local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},+done: Nat,+done_exact: {U32.to_nat(local) == done : Nat},
  +counts: RunCounts<split,first,local,count,cap>,+start: U32,+offset: Nat,+reads: Nat,
  +source_split: {Nat.add(U32.to_nat(start),offset) == U32.to_nat(run_source(run_of(split,first,local,count))) : Nat},
  +cover: {Nat.is_le(U32.to_nat(run_count_at(split,first,local,count)),reads) == True{} : Bool},
  +last: Logic.When(Nat.is_lt(0n,U32.to_nat(run_shift(run_of(split,first,local,count)))) &&
      Nat.is_lt(Nat.add(U32.to_nat(run_cell(run_of(split,first,local,count))),U32.to_nat(run_count_at(split,first,local,count))),U32.to_nat(split_source_groups(split))),
    {Nat.is_lt(U32.to_nat(run_count_at(split,first,local,count)),reads) == True{} : Bool}),
  +behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+span: U32,+lower: Held.Stack<Tile.Tile8>,
  +span_reads: {Nat.is_le(Nat.add(offset,reads),U32.to_nat(span)) == True{} : Bool},+span_room: {Nat.is_le(U32.to_nat(span),U32.to_nat(cap)) == True{} : Bool},
  +leaf: Content.Leaf(~Tile.Tile8,~Context,~value,context,start,span,Buffer.tree(~Tile.Tile8,buffer),Buffer.length(~Tile.Tile8,buffer)),
  +step: Nat,+source_local: U32,+cell: U32,+group: U32,+index: U32,+source_local_exact: {U32.to_nat(source_local) == Nat.add(offset,step) : Nat},
  +cell_exact: {U32.to_nat(cell) == Nat.add(U32.to_nat(run_cell(run_of(split,first,local,count))),step) : Nat},
  +group_exact: {U32.to_nat(group) == Nat.add(U32.to_nat(run_group(run_of(split,first,local,count))),step) : Nat},
  +index_exact: {U32.to_nat(index) == Nat.add(done,step) : Nat},+steps: {Nat.add(step,fuel) == U32.to_nat(run_count_at(split,first,local,count)) : Nat},
  +query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+stack: Held.Stack<Tile.Tile8>,+depth: Nat,+output: Buffer.Model<Tile.Tile8>,
  +facts: CellsFacts<Context,value,context,stack,depth,count,query,Held.Holding{behind,Held.Held{buffer,start,span,lower}},output,
    done_cell(~Target,~expected,target,first,query,Nat.add(done,step))>) ->
  CellsFacts<Context,value,context,stack,depth,count,query,Held.Holding{behind,Held.Held{buffer,start,span,lower}},
    run_cells_output(fuel,split_width(split),split_source_groups(split),run_shift(run_of(split,first,local,count)),source_local,cell,group,index,buffer,output),
    done_cell(~Target,~expected,target,first,query,Nat.add(done,U32.to_nat(run_count_at(split,first,local,count))))>:
  match fuel:
    case 0n:
      +ended = Equal.trans(Nat,step,Nat.add(step,0n),U32.to_nat(run_count_at(split,first,local,count)),Equal.sym(Nat,Nat.add(step,0n),step,Addition.zero_right(step)),steps)
      %Equal.cong(Nat,Nat,number => Nat.add(done,number),step,U32.to_nat(run_count_at(split,first,local,count)),ended) :
        CellsFacts<Context,value,context,stack,depth,count,query,Held.Holding{behind,Held.Held{buffer,start,span,lower}},output,done_cell(~Target,~expected,target,first,query,_)>
      facts
    case 1n+ +rest:
      RunCounts{+positive,+room,+first_read,+source_room,+cells_room,+groups_room,+source_groups_room} = counts
      +run = run_of(split,first,local,count)
      +runs = U32.to_nat(run_count_at(split,first,local,count))
      +width = split_width(split)
      +source_groups = split_source_groups(split)
      +shift = run_shift(run)
      +source = U32.to_nat(run_source(run))
      +cell_first = U32.to_nat(run_cell(run))
      +group_first = U32.to_nat(run_group(run))
      +holding = {Held.Holding{behind,Held.Held{buffer,start,span,lower}} : Held.Holding<Tile.Tile8>}
      +step_room = {%Equal.sym(Nat,Nat.add(step,1n+rest),runs,steps) : {Nat.is_le(_,runs) == True{} : Bool} Order.reflexive(runs) :
        {Nat.is_le(Nat.add(step,1n+rest),runs) == True{} : Bool}}
      +step_below = Order.room_index(step,rest,runs,step_room)
      +done_room = {%done_exact : {Nat.is_le(Nat.add(_,runs),U32.to_nat(count)) == True{} : Bool} room : {Nat.is_le(Nat.add(done,runs),U32.to_nat(count)) == True{} : Bool}}
      +target_below = Order.lt_le(Nat.add(done,step),Nat.add(done,runs),U32.to_nat(count),Order.offset_lt(done,step,runs,step_below),done_room)
      +index_below = {%Equal.sym(Nat,U32.to_nat(index),Nat.add(done,step),index_exact) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} target_below :
        {Nat.is_lt(U32.to_nat(index),U32.to_nat(count)) == True{} : Bool}}
      +index_local = {%Equal.sym(Nat,U32.to_nat(local),done,done_exact) : {U32.to_nat(index) == Nat.add(_,step) : Nat} index_exact :
        {U32.to_nat(index) == Nat.add(U32.to_nat(local),step) : Nat}}
      +low_inside = Order.lt_le(Nat.add(offset,step),Nat.add(offset,reads),U32.to_nat(span),Order.offset_lt(offset,step,reads,Order.lt_le(step,runs,reads,step_below,cover)),span_reads)
      +source_step = Equal.trans(Nat,Nat.add(U32.to_nat(start),Nat.add(offset,step)),Nat.add(Nat.add(U32.to_nat(start),offset),step),Nat.add(source,step),
        Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(start),offset),step),Nat.add(U32.to_nat(start),Nat.add(offset,step)),Addition.associative(U32.to_nat(start),offset,step)),
        Equal.cong(Nat,Nat,number => Nat.add(number,step),Nat.add(U32.to_nat(start),offset),source,source_split))
      +low = Block.piece_value(buffer,source_local)
      +low_exact = {%source_step : {low == value(context,U32.from_nat(_)) : Tile.Tile8}
        piece_read(~Context,~value,context,buffer,start,span,leaf,source_local,Nat.add(offset,step),source_local_exact,low_inside) :
        {low == value(context,U32.from_nat(Nat.add(source,step))) : Tile.Tile8}}
      +cell_room = Order.lt_le(Nat.add(cell_first,step),Nat.add(cell_first,1n+runs),U32.to_nat(cap),
        Order.offset_lt(cell_first,step,1n+runs,Order.lt_le(step,runs,1n+runs,step_below,Order.successor(runs,runs,Order.reflexive(runs)))),cells_room)
      +further = U32.is_lt(0,shift) && U32.is_lt((cell + 1 : U32),source_groups)
      +further_value = Nat.is_lt(0n,U32.to_nat(shift)) && Nat.is_lt(1n+Nat.add(cell_first,step),U32.to_nat(source_groups))
      +reads_next = Logic.when_enabled({Nat.is_lt(1n+Nat.add(offset,step),U32.to_nat(span)) == True{} : Bool},further_value,
        true => {%Addition.successor_right(offset,step) : {Nat.is_lt(_,U32.to_nat(span)) == True{} : Bool}
          Order.lt_le(Nat.add(offset,1n+step),Nat.add(offset,reads),U32.to_nat(span),
            Order.offset_lt(offset,1n+step,reads,next_below(Nat.is_lt(1n+step,runs),step,runs,reads,cell_first,U32.to_nat(shift),U32.to_nat(source_groups),{==},step_below,cover,
              last,true)),span_reads) :
          {Nat.is_lt(1n+Nat.add(offset,step),U32.to_nat(span)) == True{} : Bool}})
      +needs = run_needs(local,below,step,step_below,index,index_local)
      +high = Bool.pick(Tile.Tile8,further,Block.piece_value(buffer,((source_local + 1 : U32) + 0 * 0 : U32)),Block.zero())
      +high_exact = run_high(~Context,~value,context,SplitValue.spot_needed(width,spot_at(split,(first + index : U32))),further,further_value,
        further_exact(shift,cell,Nat.add(cell_first,step),cell_exact,source_groups,cap,cap_ok,cell_room),buffer,start,span,leaf,source_local,Nat.add(offset,step),source_local_exact,
        Nat.add(source,step),source_step,cap,cap_ok,span_room,reads_next,needs)
      +written_value = run_value(width,source_groups,shift,source_local,cell,group,buffer)
      +value_exact = run_law(local,below,step,step_below,index,index_local,group,group_exact,low,high,low_exact,high_exact)
      +written_step = written_facts(~Context,~value,~Target,~expected,context,target,first,count,index,Nat.add(done,step),index_exact,index_below,query,query_inside,stack,depth,
        holding,output,facts,holding,facts_kept(~Context,~value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,Nat.add(done,step)),facts),
        written_value,value_exact)
      run_cells_facts(~Context,~value,~Target,~expected,~context,~target,~split,~first,~count,~run_law,~run_needs,rest,cap,cap_ok,count_room,local,below,done,done_exact,counts,
        start,offset,reads,source_split,cover,last,behind,buffer,span,lower,span_reads,span_room,leaf,1n+step,(source_local + 1 : U32),(cell + 1 : U32),(group + 1 : U32),
        (index + 1 : U32),
        next_step(source_local,offset,step,source_local_exact,cap,cap_ok,Order.lt_le(Nat.add(offset,step),U32.to_nat(span),U32.to_nat(cap),low_inside,span_room)),
        next_step(cell,cell_first,step,cell_exact,cap,cap_ok,cell_room),
        next_step(group,group_first,step,group_exact,cap,cap_ok,
          Order.lt_le(Nat.add(group_first,step),Nat.add(group_first,runs),U32.to_nat(cap),Order.offset_lt(group_first,step,runs,step_below),groups_room)),
        next_step(index,done,step,index_exact,cap,cap_ok,Order.lt_le(Nat.add(done,step),U32.to_nat(count),U32.to_nat(cap),target_below,count_room)),
        Equal.trans(Nat,1n+Nat.add(step,rest),Nat.add(step,1n+rest),runs,Equal.sym(Nat,Nat.add(step,1n+rest),1n+Nat.add(step,rest),Addition.successor_right(step,rest)),steps),
        query,query_inside,stack,depth,written(index,written_value,output),
        {%Equal.sym(Nat,Nat.add(done,1n+step),1n+Nat.add(done,step),Addition.successor_right(done,step)) :
            CellsFacts<Context,value,context,stack,depth,count,query,holding,written(index,written_value,output),done_cell(~Target,~expected,target,first,query,_)>
          written_step :
          CellsFacts<Context,value,context,stack,depth,count,query,holding,written(index,written_value,output),done_cell(~Target,~expected,target,first,query,Nat.add(done,1n+step))>})

# ---- What the runs of a target piece leave ----

# The capacity the pieces lie below.
def sorted_capacity(+stack: Held.Stack<Tile.Tile8>,+sorted: Merge.PartSorted<stack>) -> U32:
  Merge.PartSorted{+low,+high,+capacity,+capacity_ok,+bounded,+order} = sorted
  capacity

# A run ends inside the target piece.
def run_done_room(+split: Tiled.Split,+first: U32,+local: U32,+count: U32,+cap: U32,+counts: RunCounts<split,first,local,count,cap>,+done: Nat,
  +done_exact: {U32.to_nat(local) == done : Nat}) -> {Nat.is_le(Nat.add(done,U32.to_nat(run_count_at(split,first,local,count))),U32.to_nat(count)) == True{} : Bool}:
  RunCounts{+positive,+room,+first_read,+source_room,+cells_room,+groups_room,+source_groups_room} = counts
  %done_exact : {Nat.is_le(Nat.add(_,U32.to_nat(run_count_at(split,first,local,count))),U32.to_nat(count)) == True{} : Bool}
  room

# The first source cell of a run lies in the pieces held.
def run_first_held(+split: Tiled.Split,+first: U32,+local: U32,+count: U32,+cap: U32,+counts: RunCounts<split,first,local,count,cap>,+spans: Held.Spans,
  +held: CellHeld<spans,split_width(split),split_source_groups(split),spot_at(split,(first + local : U32))>) ->
  Held.SpanHeld(spans,U32.to_nat(run_source(run_of(split,first,local,count)))):
  RunCounts{+positive,+room,+first_read,+source_room,+cells_room,+groups_room,+source_groups_room} = counts
  CellHeld{+low,+high} = held
  %Equal.sym(U32,run_source(run_of(split,first,local,count)),SplitValue.spot_read(split_width(split),spot_at(split,(first + local : U32))),first_read) :
    Held.SpanHeld(spans,U32.to_nat(_))
  low

# Whether another source cell follows the last of a run, in words and in
# numbers.
def extra_exact(+shift: U32,+cell: U32,+count: U32,+source_groups: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(cell),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.is_lt(0,shift) && U32.is_lt((cell + count : U32),source_groups) ==
    Nat.is_lt(0n,U32.to_nat(shift)) && Nat.is_lt(Nat.add(U32.to_nat(cell),U32.to_nat(count)),U32.to_nat(source_groups)) : Bool}:
  %Equal.sym(Bool,U32.is_lt(0,shift),Nat.is_lt(U32.to_nat(0),U32.to_nat(shift)),Comparison.lt(0,shift)) :
    {_ && U32.is_lt((cell + count : U32),source_groups) == Nat.is_lt(0n,U32.to_nat(shift)) && Nat.is_lt(Nat.add(U32.to_nat(cell),U32.to_nat(count)),U32.to_nat(source_groups)) : Bool}
  %Equal.sym(Bool,U32.is_lt((cell + count : U32),source_groups),Nat.is_lt(U32.to_nat((cell + count : U32)),U32.to_nat(source_groups)),Comparison.lt((cell + count : U32),source_groups)) :
    {Nat.is_lt(0n,U32.to_nat(shift)) && _ == Nat.is_lt(0n,U32.to_nat(shift)) && Nat.is_lt(Nat.add(U32.to_nat(cell),U32.to_nat(count)),U32.to_nat(source_groups)) : Bool}
  %Equal.sym(Nat,U32.to_nat((cell + count : U32)),Nat.add(U32.to_nat(cell),U32.to_nat(count)),
      Arithmetic.sum_exact(cap,cap_ok,cell,count,U32.to_nat(cell),U32.to_nat(count),{==},{==},room)) :
    {Nat.is_lt(0n,U32.to_nat(shift)) && Nat.is_lt(_,U32.to_nat(source_groups)) ==
      Nat.is_lt(0n,U32.to_nat(shift)) && Nat.is_lt(Nat.add(U32.to_nat(cell),U32.to_nat(count)),U32.to_nat(source_groups)) : Bool}
  {==}

# A run reads one more source cell than its cells at most.
def extra_le(extra: Bool,+count: Nat) -> {Nat.is_le(Nat.add(count,Bool.pick(Nat,extra,1n,0n)),1n+count) == True{} : Bool}:
  match extra:
    case True{}:
      %Equal.sym(Nat,Nat.add(count,1n),1n+count,Addition.successor_width(count)) : {Nat.is_le(_,1n+count) == True{} : Bool}
      Order.reflexive(1n+count)
    case False{}:
      %Equal.sym(Nat,Nat.add(count,0n),count,Addition.zero_right(count)) : {Nat.is_le(_,1n+count) == True{} : Bool}
      Order.successor(count,count,Order.reflexive(count))

# A run from the head piece of a resolved holding: from its buffer when the
# piece holds every source cell the run reads, otherwise cell by cell.
def held_facts(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,~context: Context,~target: Target,
  ~split: Tiled.Split,~first: U32,~count: U32,~cell_law: CellLaw(~Context,~value,~Target,~expected,context,target,split,first,count),~stack: Held.Stack<Tile.Tile8>,
  ~cells_held: CellsHeld(Held.spans(~Tile.Tile8,stack),split,first,count),~run_law: RunLaw(~Context,~value,~Target,~expected,context,target,split,first,count),
  ~run_needs: RunNeeds(split,first,count),inside: Bool,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool},+local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},
  +done: Nat,+done_exact: {U32.to_nat(local) == done : Nat},+counts: RunCounts<split,first,local,count,cap>,
  +behind: Held.Stack<Tile.Tile8>,+buffer: Buffer.Model<Tile.Tile8>,+start: U32,+span: U32,+lower: Held.Stack<Tile.Tile8>,
  +inside_exact: {run_inside(split_source_groups(split),run_of(split,first,local,count),start,span) == inside : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool},
  +query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+sorted: Merge.PartSorted<stack>,+depth: Nat,+output: Buffer.Model<Tile.Tile8>,
  +facts: CellsFacts<Context,value,context,stack,depth,count,query,Held.Holding{behind,Held.Held{buffer,start,span,lower}},output,done_cell(~Target,~expected,target,first,query,done)>) ->
  CellsFacts<Context,value,context,stack,depth,count,query,run_held_holding(split,run_of(split,first,local,count),inside,first,local,behind,buffer,start,span,lower),
    run_held_output(split,run_of(split,first,local,count),inside,first,local,behind,buffer,start,span,lower,output),
    done_cell(~Target,~expected,target,first,query,Nat.add(done,U32.to_nat(run_count_at(split,first,local,count))))>:
  match inside:
    case True{}:
      RunCounts{+positive,+room,+first_read,+source_room,+cells_room,+groups_room,+source_groups_room} = counts
      CellsFacts{Merge.PartKept{+ordered,+values},+shape,+read} = facts
      +run = run_of(split,first,local,count)
      +runs = U32.to_nat(run_count_at(split,first,local,count))
      +source_word = run_source(run)
      +source = U32.to_nat(source_word)
      +source_groups = split_source_groups(split)
      +reads_word = run_reads(source_groups,run)
      +extra = run_extra(source_groups,run)
      +reads = Nat.add(runs,Bool.pick(Nat,extra,1n,0n))
      +runs_room = Order.transitive(1n+runs,Nat.add(source,1n+runs),U32.to_nat(cap),Order.right_le(source,1n+runs),source_room)
      +reads_exact = reads_value(extra,run_count_at(split,first,local,count),cap,cap_ok,Order.succ_le_lt(runs,U32.to_nat(cap),runs_room))
      +after = Logic.left(U32.is_le(start,source_word),U32.is_le((source_word + reads_word : U32),(start + span : U32)),inside_exact)
      +fits = Logic.right(U32.is_le(start,source_word),U32.is_le((source_word + reads_word : U32),(start + span : U32)),inside_exact)
      +start_le = Equal.trans(Bool,Nat.is_le(U32.to_nat(start),source),U32.is_le(start,source_word),True{},
        Equal.sym(Bool,U32.is_le(start,source_word),Nat.is_le(U32.to_nat(start),source),Comparison.le(start,source_word)),after)
      +reads_room = Order.transitive(Nat.add(source,reads),Nat.add(source,1n+runs),U32.to_nat(cap),Order.offset_le(source,reads,1n+runs,extra_le(extra,runs)),source_room)
      +fits_nat = {%Arithmetic.sum_exact(cap,cap_ok,start,span,U32.to_nat(start),U32.to_nat(span),{==},{==},piece_room) :
          {Nat.is_le(Nat.add(source,reads),_) == True{} : Bool}
        %Arithmetic.sum_exact(cap,cap_ok,source_word,reads_word,source,reads,{==},reads_exact,reads_room) :
          {Nat.is_le(_,U32.to_nat((start + span : U32))) == True{} : Bool}
        Equal.trans(Bool,Nat.is_le(U32.to_nat((source_word + reads_word : U32)),U32.to_nat((start + span : U32))),U32.is_le((source_word + reads_word : U32),(start + span : U32)),True{},
          Equal.sym(Bool,U32.is_le((source_word + reads_word : U32),(start + span : U32)),Nat.is_le(U32.to_nat((source_word + reads_word : U32)),U32.to_nat((start + span : U32))),
            Comparison.le((source_word + reads_word : U32),(start + span : U32))),fits) :
        {Nat.is_le(Nat.add(source,reads),Nat.add(U32.to_nat(start),U32.to_nat(span))) == True{} : Bool}}
      +offset = Nat.sub(source,U32.to_nat(start))
      +source_split = Order.restore_shift(U32.to_nat(start),source,start_le)
      +span_reads = Order.cancel_offset_le(U32.to_nat(start),Nat.add(offset,reads),U32.to_nat(span),
        {%Equal.trans(Nat,Nat.add(source,reads),Nat.add(Nat.add(U32.to_nat(start),offset),reads),Nat.add(U32.to_nat(start),Nat.add(offset,reads)),
            Equal.cong(Nat,Nat,number => Nat.add(number,reads),source,Nat.add(U32.to_nat(start),offset),Equal.sym(Nat,Nat.add(U32.to_nat(start),offset),source,source_split)),
            Addition.associative(U32.to_nat(start),offset,reads)) :
            {Nat.is_le(_,Nat.add(U32.to_nat(start),U32.to_nat(span))) == True{} : Bool}
          fits_nat : {Nat.is_le(Nat.add(U32.to_nat(start),Nat.add(offset,reads)),Nat.add(U32.to_nat(start),U32.to_nat(span))) == True{} : Bool}})
      +span_room = Order.transitive(U32.to_nat(span),Nat.add(U32.to_nat(start),U32.to_nat(span)),U32.to_nat(cap),Order.right_le(U32.to_nat(start),U32.to_nat(span)),piece_room)
      +cell_first = U32.to_nat(run_cell(run))
      +last = {%extra_exact(run_shift(run),run_cell(run),run_count_at(split,first,local,count),source_groups,cap,cap_ok,
            Order.transitive(Nat.add(cell_first,runs),Nat.add(cell_first,1n+runs),U32.to_nat(cap),Order.offset_le(cell_first,runs,1n+runs,Order.successor(runs,runs,Order.reflexive(runs))),
              cells_room)) :
            Logic.When(_,{Nat.is_lt(runs,reads) == True{} : Bool})
          reads_extra(extra,runs) :
        Logic.When(Nat.is_lt(0n,U32.to_nat(run_shift(run))) && Nat.is_lt(Nat.add(cell_first,runs),U32.to_nat(source_groups)),{Nat.is_lt(runs,reads) == True{} : Bool})}
      +source_room_cap = Order.transitive(source,Nat.add(source,1n+runs),U32.to_nat(cap),Order.le_add(source,1n+runs),source_room)
      +holding = {Held.Holding{behind,Held.Held{buffer,start,span,lower}} : Held.Holding<Tile.Tile8>}
      run_cells_facts(~Context,~value,~Target,~expected,~context,~target,~split,~first,~count,~run_law,~run_needs,runs,cap,cap_ok,count_room,local,below,done,done_exact,
        RunCounts{positive,room,first_read,source_room,cells_room,groups_room,source_groups_room},start,offset,reads,source_split,Order.le_add(runs,Bool.pick(Nat,extra,1n,0n)),last,
        behind,buffer,span,lower,span_reads,span_room,head_leaf(~Context,~value,context,behind,buffer,start,span,lower,values),0n,(source_word - start : U32),run_cell(run),run_group(run),local,
        Equal.trans(Nat,U32.to_nat((source_word - start : U32)),offset,Nat.add(offset,0n),
          Arithmetic.difference_exact(cap,cap_ok,source_word,start,source,U32.to_nat(start),{==},{==},start_le,source_room_cap),Equal.sym(Nat,Nat.add(offset,0n),offset,Addition.zero_right(offset))),
        Equal.sym(Nat,Nat.add(cell_first,0n),cell_first,Addition.zero_right(cell_first)),
        Equal.sym(Nat,Nat.add(U32.to_nat(run_group(run)),0n),U32.to_nat(run_group(run)),Addition.zero_right(U32.to_nat(run_group(run)))),
        Equal.trans(Nat,U32.to_nat(local),done,Nat.add(done,0n),done_exact,Equal.sym(Nat,Nat.add(done,0n),done,Addition.zero_right(done))),{==},
        query,query_inside,stack,depth,output,
        {%Equal.sym(Nat,Nat.add(done,0n),done,Addition.zero_right(done)) :
            CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,_)>
          CellsFacts{Merge.PartKept{ordered,values},shape,read} :
          CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,Nat.add(done,0n))>})
    case False{}:
      cells_facts(~Context,~value,~Target,~expected,~context,~target,~split,~first,~count,~cell_law,~stack,~cells_held,U32.to_nat(run_count_at(split,first,local,count)),
        cap,cap_ok,count_room,local,done,done_exact,Nat.add(done,U32.to_nat(run_count_at(split,first,local,count))),{==},
        run_done_room(split,first,local,count,cap,counts,done,done_exact),query,query_inside,sorted,depth,Held.Holding{behind,Held.Held{buffer,start,span,lower}},output,facts)

# A run from the holding resolved at its first source cell.
def resolved_facts(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,~context: Context,~target: Target,
  ~split: Tiled.Split,~first: U32,~count: U32,~cell_law: CellLaw(~Context,~value,~Target,~expected,context,target,split,first,count),~stack: Held.Stack<Tile.Tile8>,
  ~cells_held: CellsHeld(Held.spans(~Tile.Tile8,stack),split,first,count),~run_law: RunLaw(~Context,~value,~Target,~expected,context,target,split,first,count),
  ~run_needs: RunNeeds(split,first,count),+holding: Held.Holding<Tile.Tile8>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool},+local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},
  +done: Nat,+done_exact: {U32.to_nat(local) == done : Nat},+counts: RunCounts<split,first,local,count,cap>,
  +current: Held.Current(~Tile.Tile8,holding,run_source(run_of(split,first,local,count)),cap),
  +query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+sorted: Merge.PartSorted<stack>,+depth: Nat,+output: Buffer.Model<Tile.Tile8>,
  +facts: CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,done)>) ->
  CellsFacts<Context,value,context,stack,depth,count,query,run_resolved_holding(split,run_of(split,first,local,count),first,local,holding),
    run_resolved_output(split,run_of(split,first,local,count),first,local,holding,output),
    done_cell(~Target,~expected,target,first,query,Nat.add(done,U32.to_nat(run_count_at(split,first,local,count))))>:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+start,+span,+lower}}:
      Held.Holds{+after,+before,+room} = current
      held_facts(~Context,~value,~Target,~expected,~context,~target,~split,~first,~count,~cell_law,~stack,~cells_held,~run_law,~run_needs,
        run_inside(split_source_groups(split),run_of(split,first,local,count),start,span),cap,cap_ok,count_room,local,below,done,done_exact,counts,behind,buffer,start,span,lower,
        {==},room,query,query_inside,sorted,depth,output,facts)
    case Held.Holding{+behind,Held.Bottom{}}:
      cells_facts(~Context,~value,~Target,~expected,~context,~target,~split,~first,~count,~cell_law,~stack,~cells_held,U32.to_nat(run_count_at(split,first,local,count)),
        cap,cap_ok,count_room,local,done,done_exact,Nat.add(done,U32.to_nat(run_count_at(split,first,local,count))),{==},
        run_done_room(split,first,local,count,cap,counts,done,done_exact),query,query_inside,sorted,depth,Held.Holding{behind,Held.Bottom{}},output,facts)

# A run of a target piece from a local index below count.
def applied_facts(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,~context: Context,~target: Target,
  ~split: Tiled.Split,~first: U32,~count: U32,~cell_law: CellLaw(~Context,~value,~Target,~expected,context,target,split,first,count),~stack: Held.Stack<Tile.Tile8>,
  ~cells_held: CellsHeld(Held.spans(~Tile.Tile8,stack),split,first,count),~run_law: RunLaw(~Context,~value,~Target,~expected,context,target,split,first,count),
  ~run_needs: RunNeeds(split,first,count),~cap: U32,~runs_counted: RunsCounted(split,first,count,cap),+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool},+local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},
  +done: Nat,+done_exact: {U32.to_nat(local) == done : Nat},+query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},
  +sorted: Merge.PartSorted<stack>,+capacity_exact: {sorted_capacity(stack,sorted) == cap : U32},+depth: Nat,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +facts: CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,done)>) ->
  CellsFacts<Context,value,context,stack,depth,count,query,run_applied_holding(split,run_of(split,first,local,count),first,local,holding),
    run_applied_output(split,run_of(split,first,local,count),first,local,holding,output),
    done_cell(~Target,~expected,target,first,query,Nat.add(done,U32.to_nat(run_count_at(split,first,local,count))))>:
  Merge.PartSorted{+low,+high,+capacity,+capacity_ok,+bounded,+order} = sorted
  CellsFacts{Merge.PartKept{+ordered,+values},+shape,+read} = facts
  +run = run_of(split,first,local,count)
  +source = run_source(run)
  +counts = runs_counted(local,below)
  +back = Equal.sym(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,holding),stack,ordered)
  +held = {%back : Held.SpanHeld(Held.spans(~Tile.Tile8,_),U32.to_nat(source))
      run_first_held(split,first,local,count,cap,counts,Held.spans(~Tile.Tile8,stack),cells_held(local,below)) :
    Held.SpanHeld(Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)),U32.to_nat(source))}
  +current = {%capacity_exact : Held.Current(~Tile.Tile8,Held.resolve(~Tile.Tile8,holding,source),source,_)
      Held.resolve_current(~Tile.Tile8,holding,source,low,high,capacity,capacity_ok,bounded,Held.sorted_moved(~Tile.Tile8,stack,Held.order(~Tile.Tile8,holding),low,high,back,order),held) :
    Held.Current(~Tile.Tile8,Held.resolve(~Tile.Tile8,holding,source),source,cap)}
  +kept = picked_kept(~Context,~value,context,True{},source,holding,stack,Merge.PartKept{ordered,values})
  resolved_facts(~Context,~value,~Target,~expected,~context,~target,~split,~first,~count,~cell_law,~stack,~cells_held,~run_law,~run_needs,Held.resolve(~Tile.Tile8,holding,source),
    cap,cap_ok,count_room,local,below,done,done_exact,counts,current,query,query_inside,Merge.PartSorted{low,high,capacity,capacity_ok,bounded,order},depth,output,
    CellsFacts{kept,shape,read})

# A run covers at least one cell.
def run_positive(+split: Tiled.Split,+first: U32,+local: U32,+count: U32,+cap: U32,+counts: RunCounts<split,first,local,count,cap>) ->
  {Nat.is_lt(0n,U32.to_nat(run_count_at(split,first,local,count))) == True{} : Bool}:
  RunCounts{+positive,+room,+first_read,+source_room,+cells_room,+groups_room,+source_groups_room} = counts
  positive

# The runs of a target piece from local index local on, while more: the
# pieces stay, and every cell of the target piece holds its target value.
def runs_facts(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,~context: Context,~target: Target,
  ~split: Tiled.Split,~first: U32,~count: U32,~cell_law: CellLaw(~Context,~value,~Target,~expected,context,target,split,first,count),~stack: Held.Stack<Tile.Tile8>,
  ~cells_held: CellsHeld(Held.spans(~Tile.Tile8,stack),split,first,count),~run_law: RunLaw(~Context,~value,~Target,~expected,context,target,split,first,count),
  ~run_needs: RunNeeds(split,first,count),~cap: U32,~runs_counted: RunsCounted(split,first,count,cap),fuel: Nat,more: Bool,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool},+local: U32,+done: Nat,
  +done_exact: {U32.to_nat(local) == done : Nat},+more_exact: {more == Nat.is_lt(done,U32.to_nat(count)) : Bool},+done_room: {Nat.is_le(done,U32.to_nat(count)) == True{} : Bool},
  +fuel_room: {Nat.is_le(U32.to_nat(count),Nat.add(done,fuel)) == True{} : Bool},+query: U32,+query_inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},
  +sorted: Merge.PartSorted<stack>,+capacity_exact: {sorted_capacity(stack,sorted) == cap : U32},+depth: Nat,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +facts: CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,done)>) ->
  CellsFacts<Context,value,context,stack,depth,count,query,runs_holding(fuel,more,split,first,local,count,holding),runs_output(fuel,more,split,first,local,count,holding,output),
    done_cell(~Target,~expected,target,first,query,U32.to_nat(count))>:
  match fuel more:
    case 0n more:
      +reached = Order.antisymmetric(done,U32.to_nat(count),done_room,
        {%Addition.zero_right(done) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool} fuel_room : {Nat.is_le(U32.to_nat(count),done) == True{} : Bool}})
      %reached : CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,_)>
      facts
    case 1n+ +rest False{}:
      +reached = Order.antisymmetric(done,U32.to_nat(count),done_room,
        Order.not_less(done,U32.to_nat(count),Equal.sym(Bool,False{},Nat.is_lt(done,U32.to_nat(count)),more_exact)))
      %reached : CellsFacts<Context,value,context,stack,depth,count,query,holding,output,done_cell(~Target,~expected,target,first,query,_)>
      facts
    case 1n+ +rest True{}:
      +below_done = Equal.sym(Bool,True{},Nat.is_lt(done,U32.to_nat(count)),more_exact)
      +below = {%Equal.sym(Nat,U32.to_nat(local),done,done_exact) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} below_done :
        {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool}}
      +run = run_of(split,first,local,count)
      +runs = U32.to_nat(run_count_at(split,first,local,count))
      +counts = runs_counted(local,below)
      +positive = run_positive(split,first,local,count,cap,counts)
      +after = Nat.add(done,runs)
      +after_room = run_done_room(split,first,local,count,cap,counts,done,done_exact)
      +next = run_next(split,first,local,count)
      +next_exact = Arithmetic.sum_exact(cap,cap_ok,local,U32.max(Tiled.run_count(run),1),done,runs,done_exact,Extrema.max_one_exact(Tiled.run_count(run),runs,{==},positive),
        Order.transitive(after,U32.to_nat(count),U32.to_nat(cap),after_room,count_room))
      +widened = Order.offset_le(done,1n+rest,Nat.add(runs,rest),Order.add_le(1n,runs,rest,Order.positive_one(runs,positive)))
      runs_facts(~Context,~value,~Target,~expected,~context,~target,~split,~first,~count,~cell_law,~stack,~cells_held,~run_law,~run_needs,~cap,~runs_counted,rest,
        U32.is_lt(next,count),cap_ok,count_room,next,after,next_exact,
        Equal.trans(Bool,U32.is_lt(next,count),Nat.is_lt(U32.to_nat(next),U32.to_nat(count)),Nat.is_lt(after,U32.to_nat(count)),Comparison.lt(next,count),
          Equal.cong(Nat,Bool,number => Nat.is_lt(number,U32.to_nat(count)),U32.to_nat(next),after,next_exact)),
        after_room,
        Order.transitive(U32.to_nat(count),Nat.add(done,1n+rest),Nat.add(after,rest),fuel_room,
          {%Equal.sym(Nat,Nat.add(Nat.add(done,runs),rest),Nat.add(done,Nat.add(runs,rest)),Addition.associative(done,runs,rest)) :
              {Nat.is_le(Nat.add(done,1n+rest),_) == True{} : Bool}
            widened : {Nat.is_le(Nat.add(done,1n+rest),Nat.add(after,rest)) == True{} : Bool}}),
        query,query_inside,sorted,capacity_exact,depth,run_applied_holding(split,run,first,local,holding),run_applied_output(split,run,first,local,holding,output),
        applied_facts(~Context,~value,~Target,~expected,~context,~target,~split,~first,~count,~cell_law,~stack,~cells_held,~run_law,~run_needs,~cap,~runs_counted,cap_ok,count_room,
          local,below,done,done_exact,query,query_inside,sorted,capacity_exact,depth,holding,output,facts))

# ---- What a target piece holds ----

def fresh_facts(~Context: Data,~value: Context -> U32 -> Tile.Tile8,~Target: Data,~expected: Target -> U32 -> Tile.Tile8,+context: Context,+target: Target,+first: U32,+count: U32,
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},+query: U32,+stack: Held.Stack<Tile.Tile8>,+holding: Held.Holding<Tile.Tile8>,
  +kept: Merge.PartKept<Context,value,context,stack,holding>) ->
  CellsFacts<Context,value,context,stack,Tensors.depth_for(count),count,query,holding,Buffer.remodel(~Tile.Tile8,fresh_output(count)),
    done_cell(~Target,~expected,target,first,query,0n)>:
  +depth = Tensors.depth_for(count)
  +fresh = Tree.fresh(Tile.Tile8,depth,Tiled.zero_tile())
  +output = Buffer.remodel(~Tile.Tile8,fresh_output(count))
  +content = AllocationLaws.buffer_content(~Tile.Tile8,Tiled.zero_tile(),count)
  +tree_exact = Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,output),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),fresh,
    Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),Buffer.tree(~Tile.Tile8,output),Buffer.content_tree_of(~Tile.Tile8,output)),
    Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8),Buffer.content(~Tile.Tile8,output),Buffer.Content{fresh,U32.to_nat(count)},content))
  +back = Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,output),fresh,tree_exact)
  CellsFacts{kept,
    Merge.OutputShape{{%back : Shape.Balanced(Tile.Tile8,depth,_) Shape.fresh(Tile.Tile8,depth,Tiled.zero_tile()) : Shape.Balanced(Tile.Tile8,depth,Buffer.tree(~Tile.Tile8,output))},
      Depth.allocation_depth(count,Depth.bounded(count,allocatable)),
      {%back : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,_))) == True{} : Bool}
        Bounds.as_nat_le(count,Tree.capacity(Tile.Tile8,fresh),Depth.fresh_fits(~Tile.Tile8,count,Tiled.zero_tile(),allocatable)) :
        {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)))) == True{} : Bool}},
      Equal.cong(Buffer.Content<Tile.Tile8>,Nat,Buffer.content_length(~Tile.Tile8),Buffer.content(~Tile.Tile8,output),Buffer.Content{fresh,U32.to_nat(count)},content)},
    {%back : {Tree.read(Tile.Tile8,_,query) == done_cell(~Target,~expected,target,first,query,0n) : Tile.Tile8}
      %Equal.sym(Bool,Nat.is_lt(U32.to_nat(query),0n),False{},Order.not_below(0n,U32.to_nat(query),Order.zero_le(U32.to_nat(query)))) :
        {Tree.read(Tile.Tile8,fresh,query) == Bool.pick(Tile.Tile8,_,expected(target,(first + query : U32)),Tiled.zero_tile()) : Tile.Tile8}
      StorageReadback.read_fresh(~Tile.Tile8,depth,Tiled.zero_tile(),query) :
      {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),query) == done_cell(~Target,~expected,target,first,query,0n) : Tile.Tile8}}}

# Every cell of a target piece holds its target value.
def QueryHolds(~Target: Data,~expected: Target -> U32 -> Tile.Tile8,+target: Target,+first: U32,+count: U32,+output: Buffer.Model<Tile.Tile8>) -> Type:
  @+query: U32 -> @+inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool} ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),query) == expected(target,(first + query : U32)) : Tile.Tile8}

# A query cell below count, once every cell is written, holds its target
# value.
def written_holds(~Target: Data,~expected: Target -> U32 -> Tile.Tile8,+target: Target,+first: U32,+count: U32,+query: U32,
  +inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool},+held: Tile.Tile8,
  +held_exact: {held == done_cell(~Target,~expected,target,first,query,U32.to_nat(count)) : Tile.Tile8}) -> {held == expected(target,(first + query : U32)) : Tile.Tile8}:
  %inside : {held == Bool.pick(Tile.Tile8,_,expected(target,(first + query : U32)),Tiled.zero_tile()) : Tile.Tile8}
  held_exact

# The cells of a target piece from offset on hold their target values.
def piece_cells(~Target: Data,~expected: Target -> U32 -> Tile.Tile8,~target: Target,~first: U32,~count: U32,~output: Buffer.Model<Tile.Tile8>,
  ~holds: QueryHolds(~Target,~expected,target,first,count,output),remaining: Nat,+offset: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool},+length_exact: {Buffer.length(~Tile.Tile8,output) == U32.to_nat(count) : Nat},
  +left: {Nat.is_le(Nat.add(offset,remaining),U32.to_nat(count)) == True{} : Bool}) ->
  Members.All(Nat,{index => Content.At(~Tile.Tile8,~Target,~expected,target,first,Buffer.tree(~Tile.Tile8,output),Buffer.length(~Tile.Tile8,output),index) : Nat -> Data},
    Ranges.range(remaining,offset)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +index_in = Order.room_index(offset,rest,U32.to_nat(count),left)
      +back = Bounds.roundtrip(offset,cap,cap_ok,Bounds.strict_le(offset,U32.to_nat(cap),Order.lt_le(offset,U32.to_nat(count),U32.to_nat(cap),index_in,count_room)))
      +query = U32.from_nat(offset)
      +query_below = {%Equal.sym(Nat,U32.to_nat(query),offset,back) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} index_in :
        {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}}
      +address = Equal.trans(U32,(first + query : U32),U32.from_nat(Nat.add(U32.to_nat(first),U32.to_nat(query))),U32.from_nat(Nat.add(U32.to_nat(first),offset)),
        Arithmetic.native_add(first,query),Equal.cong(Nat,U32,number => U32.from_nat(Nat.add(U32.to_nat(first),number)),U32.to_nat(query),offset,back))
      KernelGemm.Both{KernelGemm.Both{{%Equal.sym(Nat,Buffer.length(~Tile.Tile8,output),U32.to_nat(count),length_exact) : {Nat.is_lt(offset,_) == True{} : Bool} index_in :
            {Nat.is_lt(offset,Buffer.length(~Tile.Tile8,output)) == True{} : Bool}},
          Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),query),expected(target,(first + query : U32)),
            expected(target,U32.from_nat(Nat.add(U32.to_nat(first),offset))),holds(query,query_below),
            Equal.cong(U32,Tile.Tile8,word => expected(target,word),(first + query : U32),U32.from_nat(Nat.add(U32.to_nat(first),offset)),address))},
        piece_cells(~Target,~expected,~target,~first,~count,~output,~holds,rest,1n+offset,cap,cap_ok,count_room,length_exact,Order.advance_room(offset,rest,U32.to_nat(count),left))}

# A buffer model is its content.
def model_content(+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.content(~Tile.Tile8,output) == Buffer.Content{Buffer.tree(~Tile.Tile8,output),Buffer.length(~Tile.Tile8,output)} : Buffer.Content<Tile.Tile8>}:
  match output:
    case Buffer.Model{+tree,+length,+certificate}: {==}

# A target piece of its shape whose every cell holds its target value.
def piece_holds(~Target: Data,~expected: Target -> U32 -> Tile.Tile8,~target: Target,~first: U32,~count: U32,~output: Buffer.Model<Tile.Tile8>,
  ~holds: QueryHolds(~Target,~expected,target,first,count,output),+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool},+depth: Nat,+shape: Merge.OutputShape<depth,count,output>) ->
  Placed.PieceHolds(~Tile.Tile8,~Target,~expected,target,first,count,Buffer.content(~Tile.Tile8,output)):
  Merge.OutputShape{+balanced,+limit,+fits,+sized} = shape
  +tree = Buffer.tree(~Tile.Tile8,output)
  +length = Buffer.length(~Tile.Tile8,output)
  +length_exact = Equal.trans(Nat,length,Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),U32.to_nat(count),
    Equal.sym(Nat,Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),length,Buffer.content_length_of(~Tile.Tile8,output)),sized)
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,output),Buffer.Content{tree,length},model_content(output)) :
    Placed.PieceHolds(~Tile.Tile8,~Target,~expected,target,first,count,_)
  KernelGemm.Both{piece_cells(~Target,~expected,~target,~first,~count,~output,~holds,U32.to_nat(count),0n,cap,cap_ok,count_room,length_exact,Order.reflexive(U32.to_nat(count))),
    KernelGemm.Both{length_exact,
      Buffer.ContentReady{depth,balanced,limit,
        {%Equal.sym(Nat,length,U32.to_nat(count),length_exact) : {Nat.is_le(_,U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool}
          fits : {Nat.is_le(length,U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool}}}}}
