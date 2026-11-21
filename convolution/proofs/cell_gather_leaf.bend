# Bounds and matching lane values for every store of one cells gather leaf: one
# run of the cells of its positions.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_gather.bend as Gather
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/write_program_every.bend as Every
import ../../lib/proofs/write_program_readback.bend as Readback
import ../../lib/proofs/write_program_observation.bend as Observation
import ../../lib/proofs/kernel_gather_properties.bend as Properties
import ../../lib/proofs/kernel_gather_width.bend as Width
import ../../lib/proofs/kernel_gather_coverage.bend as Coverage
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/proofs/nat_division.bend as Division
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./packed_output_bounds.bend as Output
import ./array_leaf_cells.bend as ArrayLeaf
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./microkernel_storage_execution.bend as Execution
import ./output_values.bend as Values
import ./cell_gather_coordinates.bend as Coordinates
import ./cell_gather_values.bend as CellValues
import ./reduction_coordinate_bounds.bend as Reduction
import ../../lib/kernel_gemm.bend as KernelGemm

type Frame is Data:
  Frame{limits: Machine.Limits,problem: ConvSpec.Problem,storage: Inputs.Contract<problem>,offset: U32,span: U32,
    channel: Nat,row: Nat,column: Nat,capacity: U32,
    valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
    certificate: ArrayLeaf.Geometry<problem,offset,span>,
    channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
    row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
    slots: {Nat.is_le(Nat.mul(U32.to_nat(Coordinates.groups(problem)),U32.to_nat(Padded.spatial(problem))),U32.to_nat(capacity)) == True{} : Bool}}

def plan(+frame: Frame) -> Properties.Copy<Tile.Tile8>:
  Frame{limits,+problem,+storage,+offset,+span,channel,row,column,capacity,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} = frame
  +groups = Coordinates.groups(problem)
  Properties.Copy{ArrayLeaf.leaf_cells(problem,storage,offset,span),1n,
    Gather.gather_width((offset * groups : U32),(Padded.spatial(problem) * groups : U32),(span * groups : U32)),
    (offset * groups : U32),(Padded.spatial(problem) * groups : U32),(span * groups : U32),0}

def offset_of(+frame: Frame) -> U32:
  Frame{limits,problem,storage,+offset,span,channel,row,column,capacity,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} = frame
  offset

def span_of(+frame: Frame) -> U32:
  Frame{limits,problem,storage,offset,+span,channel,row,column,capacity,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} = frame
  span

def capacity(+frame: Frame) -> U32:
  Frame{limits,problem,storage,offset,span,channel,row,column,capacity,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} = frame
  capacity

def query(frame: Frame) -> U32:
  Frame{limits,+problem,storage,offset,span,+channel,+row,+column,capacity,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} = frame
  U32.from_nat(Coordinates.query_index(problem,channel,row,column))

def field(frame: Frame) -> Nat:
  Frame{limits,problem,storage,offset,span,channel,row,column,capacity,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} = frame
  Nat.mod(channel,8n)

def wanted(frame: Frame) -> F32:
  Frame{limits,problem,storage,offset,span,channel,row,column,capacity,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} = frame
  Values.matrix_value(problem,channel,row,column)

def point(frame: Frame) -> Nat:
  Frame{limits,problem,storage,offset,span,channel,row,column,capacity,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} = frame
  Reduction.position(problem,row,column)

def target(+frame: Frame) -> Observation.Query<Nat,F32>:
  Observation.Query{field(frame),query(frame),wanted(frame)}

def writes(frame: Frame) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  Frame{limits,+problem,storage,+offset,+span,channel,row,column,capacity,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} = frame
  Execution.cell_block_writes(MicroLeaf.rows(problem),offset,Padded.spatial(problem),span,ArrayLeaf.leaf_cells(problem,storage,offset,span))

def Bounded(+capacity: U32,address: U32,value: Tile.Tile8) -> Data:
  {U32.is_lt(address,capacity) == True{} : Bool}

def Range(+frame: Frame,+column: Nat) -> Data:
  Frame{limits,+problem,storage,+offset,+span,channel,row,other,capacity,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} = frame
  +groups = Coordinates.groups(problem)
  KernelGemm.Both<{Nat.is_lt(column,U32.to_nat((span * groups : U32))) == True{} : Bool},
    {Nat.is_lt(Nat.add(U32.to_nat((offset * groups : U32)),column),U32.to_nat((Padded.spatial(problem) * groups : U32))) == True{} : Bool}>

# The cells of every position fit the machine as one word.
def cells_word(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {U32.is_le((Padded.spatial(problem) * Coordinates.groups(problem) : U32),Machine.capacity(limits)) == True{} : Bool}:
  +cells = (Padded.spatial(problem) * Coordinates.groups(problem) : U32)
  +machine = Machine.capacity(limits)
  %Equal.sym(Bool,U32.is_le(cells,machine),Nat.is_le(U32.to_nat(cells),U32.to_nat(machine)),Comparison.le(cells,machine)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((Padded.spatial(problem) * Coordinates.groups(problem) : U32)),Nat.mul(U32.to_nat(Padded.spatial(problem)),U32.to_nat(Coordinates.groups(problem))),Coordinates.scaled(limits,problem,Padded.spatial(problem),supported,Coordinates.spatial_bound(limits,problem,valid,supported))) : {Nat.is_le(_,U32.to_nat(machine)) == True{} : Bool}
  Coordinates.cells_bound(limits,problem,valid,supported)

def range(+frame: Frame,+column: Nat,+inside: {Nat.is_lt(column,U32.to_nat(Properties.width(~Tile.Tile8,plan(frame)))) == True{} : Bool}) -> Range(frame,column):
  Frame{+limits,+problem,storage,+offset,+span,channel,row,other,capacity,+valid,+supported,certificate,channel_inside,row_inside,column_inside,slots} = frame
  +groups = Coordinates.groups(problem)
  Width.inside((offset * groups : U32),(Padded.spatial(problem) * groups : U32),(span * groups : U32),column,Machine.capacity(limits),Limits.capacity_fits(limits),
    cells_word(limits,problem,valid,supported),inside)

# A gather leaf has one run of cells.
def bounded_with(+frame: Frame,+group: Nat,+column: Nat,+group_inside: {Nat.is_lt(group,Properties.rows(~Tile.Tile8,plan(frame))) == True{} : Bool},+range: Range(frame,column)) ->
  Bounded(capacity(frame),(Properties.destination(~Tile.Tile8,plan(frame),group) + U32.from_nat(column) : U32),
    Tree.read(Tile.Tile8,Properties.input(~Tile.Tile8,plan(frame)),(Properties.source(~Tile.Tile8,plan(frame),group) + U32.from_nat(column) : U32))):
  Frame{+limits,+problem,+storage,+offset,+span,channel,row,other,+capacity,+valid,+supported,+certificate,channel_inside,row_inside,column_inside,+slots} = frame
  match group:
    case 0n:
      KernelGemm.Both{in_span,+point_inside} = range
      +groups = Coordinates.groups(problem)
      +g = U32.to_nat(groups)
      +destination = (U32.from_nat(0n) * (Padded.spatial(problem) * groups : U32) + (offset * groups : U32) + U32.from_nat(column) : U32)
      +index = Nat.add(Nat.mul(U32.to_nat(offset),g),column)
      +bound = ArrayLeaf.offset_bound(problem,offset,span,certificate)
      +reach = CellValues.reach(limits,problem,offset,column,valid,supported,bound,point_inside)
      %Equal.sym(Bool,U32.is_lt(destination,capacity),Nat.is_lt(U32.to_nat(destination),U32.to_nat(capacity)),Comparison.lt(destination,capacity)) : {_ == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(destination),index,Coordinates.destination(limits,problem,offset,column,valid,supported,bound,reach)) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
      Bounds.nat_trans(index,Nat.mul(U32.to_nat(Padded.spatial(problem)),g),U32.to_nat(capacity),reach,
        %RowMajor.multiply_commutative(g,U32.to_nat(Padded.spatial(problem))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} slots)
    case 1n+ +rest:
      KernelGemm.Both{in_span,+point_inside} = range
      +groups = Coordinates.groups(problem)
      +g = U32.to_nat(groups)
      Empty.absurd({U32.is_lt((U32.from_nat(1n+rest) * (Padded.spatial(problem) * groups : U32) + (offset * groups : U32) + U32.from_nat(column) : U32),capacity) == True{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),group_inside)))

def bounded_cell(+frame: Frame,+group: Nat,+column: Nat,+group_inside: {Nat.is_lt(group,Properties.rows(~Tile.Tile8,plan(frame))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Properties.width(~Tile.Tile8,plan(frame)))) == True{} : Bool}) ->
  Bounded(capacity(frame),(Properties.destination(~Tile.Tile8,plan(frame),group) + U32.from_nat(column) : U32),
    Tree.read(Tile.Tile8,Properties.input(~Tile.Tile8,plan(frame)),(Properties.source(~Tile.Tile8,plan(frame),group) + U32.from_nat(column) : U32))):
  bounded_with(frame,group,column,group_inside,range(frame,column,column_inside))

def matched(+frame: Frame,+group: Nat,+column: Nat,+group_inside: {Nat.is_lt(group,Properties.rows(~Tile.Tile8,plan(frame))) == True{} : Bool},
  +range: Range(frame,column),
  +compared: {U32.is_eq((Properties.destination(~Tile.Tile8,plan(frame),group) + U32.from_nat(column) : U32),query(frame)) == True{} : Bool}) ->
  {CellValues.lane(field(frame),Tree.read(Tile.Tile8,Properties.input(~Tile.Tile8,plan(frame)),(Properties.source(~Tile.Tile8,plan(frame),group) + U32.from_nat(column) : U32))) == wanted(frame) : F32}:
  Frame{+limits,+problem,+storage,+offset,+span,+channel,+row,+other,capacity,+valid,+supported,+certificate,+channel_inside,+row_inside,+column_inside,slots} = frame
  match group:
    case 0n:
      KernelGemm.Both{+in_span,+point_inside} = range
      +groups = Coordinates.groups(problem)
      CellValues.copied_value(limits,problem,storage,offset,span,channel,row,other,column,valid,supported,certificate,channel_inside,row_inside,column_inside,in_span,point_inside,compared)
    case 1n+ +rest:
      KernelGemm.Both{+in_span,+point_inside} = range
      +groups = Coordinates.groups(problem)
      Empty.absurd({CellValues.lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,ArrayLeaf.leaf_cells(problem,storage,offset,span),
          (U32.from_nat(1n+rest) * (span * groups : U32) + 0 + U32.from_nat(column) : U32))) == Values.matrix_value(problem,channel,row,other) : F32},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),group_inside)))

def match_with(equal: Bool,+frame: Frame,+group: Nat,+column: Nat,+group_inside: {Nat.is_lt(group,Properties.rows(~Tile.Tile8,plan(frame))) == True{} : Bool},
  +range: Range(frame,column),
  +compared: {U32.is_eq((Properties.destination(~Tile.Tile8,plan(frame),group) + U32.from_nat(column) : U32),query(frame)) == equal : Bool}) ->
  Boolean.When(equal,{CellValues.lane(field(frame),Tree.read(Tile.Tile8,Properties.input(~Tile.Tile8,plan(frame)),(Properties.source(~Tile.Tile8,plan(frame),group) + U32.from_nat(column) : U32))) == wanted(frame) : F32}):
  match equal:
    case False{}: Unit{}
    case True{}:
      matched(frame,group,column,group_inside,range,compared)

def match_cell(+frame: Frame,+group: Nat,+column: Nat,+group_inside: {Nat.is_lt(group,Properties.rows(~Tile.Tile8,plan(frame))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Properties.width(~Tile.Tile8,plan(frame)))) == True{} : Bool}) ->
  Observation.Holds(~Tile.Tile8,~F32,~Nat,~CellValues.lane,target(frame),(Properties.destination(~Tile.Tile8,plan(frame),group) + U32.from_nat(column) : U32),
    Tree.read(Tile.Tile8,Properties.input(~Tile.Tile8,plan(frame)),(Properties.source(~Tile.Tile8,plan(frame),group) + U32.from_nat(column) : U32))):
  match_with(U32.is_eq((Properties.destination(~Tile.Tile8,plan(frame),group) + U32.from_nat(column) : U32),query(frame)),frame,group,column,group_inside,range(frame,column,column_inside),{==})

def bound_same(+frame: Frame,+address: U32,+value: Tile.Tile8,bounded: Bounded(capacity(frame),address,value)) -> {U32.is_lt(address,capacity(frame)) == True{} : Bool}:
  bounded

def inside(+frame: Frame) -> Readback.Inside(~Tile.Tile8,capacity(frame),writes(frame)):
  Frame{+limits,+problem,+storage,+offset,+span,+channel,+row,+column,+cap,+valid,+supported,+certificate,+channel_inside,+row_inside,+column_inside,+slots} = frame
  +frame = {Frame{limits,problem,storage,offset,span,channel,row,column,cap,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} : Frame}
  Every.inside(~Tile.Tile8,~U32,~Bounded,~Frame,~capacity,~capacity,~bound_same,frame,writes(frame),
    Properties.every(~Tile.Tile8,~Frame,~U32,~plan,~capacity,~Bounded,~bounded_cell,frame))

def matching(+frame: Frame) -> Readback.Matching(~F32,query(frame),wanted(frame),Observation.projected(~Tile.Tile8,~F32,~Nat,~CellValues.lane,writes(frame),field(frame))):
  Frame{+limits,+problem,+storage,+offset,+span,+channel,+row,+column,+capacity,+valid,+supported,+certificate,+channel_inside,+row_inside,+column_inside,+slots} = frame
  +frame = {Frame{limits,problem,storage,offset,span,channel,row,column,capacity,valid,supported,certificate,channel_inside,row_inside,column_inside,slots} : Frame}
  Observation.matching(~Tile.Tile8,~F32,~Nat,~CellValues.lane,field(frame),query(frame),wanted(frame),writes(frame),
    Properties.every(~Tile.Tile8,~Frame,~Observation.Query<Nat,F32>,~plan,~target,~Observation.Holds(~Tile.Tile8,~F32,~Nat,~CellValues.lane),~match_cell,frame))

def covered(+frame: Frame,+local: Nat,+in_span: {Nat.is_lt(local,U32.to_nat(span_of(frame))) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset_of(frame)),local) == point(frame) : Nat}) ->
  {Readback.contains(~Tile.Tile8,query(frame),writes(frame)) == True{} : Bool}:
  Frame{+limits,+problem,+storage,+offset,+span,+channel,+row,+other,capacity,+valid,+supported,+certificate,+channel_inside,+row_inside,+column_inside,slots} = frame
  +groups = Coordinates.groups(problem)
  +g = U32.to_nat(groups)
  +o = U32.to_nat(offset)
  +group = Nat.div(channel,8n)
  +cell = Nat.add(Nat.mul(local,g),group)
  +start = Nat.mul(o,g)
  +query_index = Coordinates.query_index(problem,channel,row,other)
  +bound = ArrayLeaf.offset_bound(problem,offset,span,certificate)
  +group_inside = Coordinates.channel_group(limits,problem,channel,supported,channel_inside)
  +span_cells = (span * groups : U32)
  +offset_cells = (offset * groups : U32)
  +spatial_cells = (Padded.spatial(problem) * groups : U32)
  +indices = Equal.trans(Nat,Nat.add(start,cell),Nat.add(Nat.mul(Nat.add(o,local),g),group),query_index,
    Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(o,local),g),group),Nat.add(start,cell),CellValues.regrouped(o,local,g,group)),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,g),group),Nat.add(o,local),Reduction.position(problem,row,other),position))
  +reach = {%Equal.sym(Nat,Nat.add(start,cell),query_index,indices) : {Nat.is_lt(_,Nat.mul(U32.to_nat(Padded.spatial(problem)),g)) == True{} : Bool}
    Coordinates.query_inside(limits,problem,channel,row,other,valid,supported,channel_inside,row_inside,column_inside) :
    {Nat.is_lt(Nat.add(start,cell),Nat.mul(U32.to_nat(Padded.spatial(problem)),g)) == True{} : Bool}}
  +in_output = {%Equal.sym(Nat,U32.to_nat((offset * Coordinates.groups(problem) : U32)),Nat.mul(U32.to_nat(offset),U32.to_nat(Coordinates.groups(problem))),Coordinates.scaled(limits,problem,offset,supported,bound)) : {Nat.is_lt(Nat.add(_,cell),U32.to_nat(spatial_cells)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat((Padded.spatial(problem) * Coordinates.groups(problem) : U32)),Nat.mul(U32.to_nat(Padded.spatial(problem)),U32.to_nat(Coordinates.groups(problem))),Coordinates.scaled(limits,problem,Padded.spatial(problem),supported,Coordinates.spatial_bound(limits,problem,valid,supported))) : {Nat.is_lt(Nat.add(start,cell),_) == True{} : Bool}
    reach : {Nat.is_lt(Nat.add(U32.to_nat(offset_cells),cell),U32.to_nat(spatial_cells)) == True{} : Bool}}
  +cell_in_span = {%Equal.sym(Nat,U32.to_nat((span * Coordinates.groups(problem) : U32)),Nat.mul(U32.to_nat(span),U32.to_nat(Coordinates.groups(problem))),Coordinates.scaled(limits,problem,span,supported,ArrayLeaf.span_bound(problem,offset,span,certificate))) : {Nat.is_lt(cell,_) == True{} : Bool}
    RowMajor.radix(local,U32.to_nat(span),group,g,in_span,group_inside) : {Nat.is_lt(cell,U32.to_nat(span_cells)) == True{} : Bool}}
  +width = Gather.gather_width(offset_cells,spatial_cells,span_cells)
  +width_inside = Width.covers(offset_cells,spatial_cells,span_cells,cell,Machine.capacity(limits),Limits.capacity_fits(limits),cells_word(limits,problem,valid,supported),cell_in_span,in_output)
  +destination = (U32.from_nat(0n) * spatial_cells + offset_cells + U32.from_nat(cell) : U32)
  +same = Equal.trans(U32,destination,U32.from_nat(Nat.add(start,cell)),U32.from_nat(query_index),
    Encoded.encoding(destination,Nat.add(start,cell),Coordinates.destination(limits,problem,offset,cell,valid,supported,bound,reach)),
    Equal.cong(Nat,U32,value => U32.from_nat(value),Nat.add(start,cell),query_index,indices))
  %same : {Readback.contains(~Tile.Tile8,_,Execution.cell_block_writes(MicroLeaf.rows(problem),offset,Padded.spatial(problem),span,ArrayLeaf.leaf_cells(problem,storage,offset,span))) == True{} : Bool}
  Coverage.copied_cell(~Tile.Tile8,1,width,offset_cells,spatial_cells,span_cells,0,ArrayLeaf.leaf_cells(problem,storage,offset,span),0n,cell,{==},width_inside)
