# Unit u of a phase reads the packed weights of its eight channels: tap t
# of the unit is weight slot u * inner + t of the phase.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../tensor_tiled.bend as Tiled
import ../tensor_view.bend as Views
import ./tensor_copy_values.bend as Values
import ../kernel_tile.bend as Tile
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_packing.bend as Packing
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./input_gradient_phase_window_weights.bend as Window
import ./input_gradient_tiled_weights.bend as Weights
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_phase_index.bend as Index
import ./input_gradient_tiled_pack_values.bend as PackedValues
import ./input_gradient_tiled_job_reads.bend as Reads
import ./kernel_packing_window_coordinates.bend as Coordinates

import ./nat_ceiling.bend as Ceiling
import ./row_major_index.bend as RowMajor
import ./tensor_view_lanes.bend as ViewLanes
import ./natural_addition.bend as Addition
import ./input_gradient_phase_weights.bend as PhaseCells
import ../kernel_gemm.bend as KernelGemm

def block_inside(+phase: Jobs.Phase,+unit: Nat,+inside: {Nat.is_lt(unit,U32.to_nat(Jobs.units(Jobs.geometry(phase)))) == True{} : Bool}) ->
  {Nat.is_lt(unit,Ceiling.ceiling(Domain.channels(Jobs.geometry(phase)),8n)) == True{} : Bool}:
  Jobs.Phase{+call,+e,y,x,yi,xi} = phase
  %Jobs.units_exact(call,e) : {Nat.is_lt(unit,_) == True{} : Bool} inside
def block_count(+phase: Jobs.Phase) -> Nat:
  U32.to_nat(Tiled.groups(Domain.channel_word(Jobs.geometry(phase))))
def block_count_exact(+phase: Jobs.Phase) -> {block_count(phase) == Ceiling.ceiling(Domain.channels(Jobs.geometry(phase)),8n) : Nat}:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Window.block_count_exact(call,U32.from_nat(y),U32.from_nat(x))
def tap_index_inside(+phase: Jobs.Phase,+output: Nat,+row: Nat,+column: Nat,
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(phase))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(phase)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(phase)))) == True{} : Bool}) ->
  {Nat.is_lt(Packing.phase_index(Jobs.context(phase),output,row,column),U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +rows = U32.to_nat(Packing.phase_rows(Jobs.context(phase)))
  +columns = U32.to_nat(Packing.phase_columns(Jobs.context(phase)))
  %Equal.sym(Nat,U32.to_nat(Kernel.phase_inner(Jobs.context(phase))),Nat.mul(Domain.outputs(g),Nat.mul(rows,columns)),PhaseBounds.phase_inner_exact(call,U32.from_nat(y),U32.from_nat(x))) :
    {Nat.is_lt(Packing.phase_index(Jobs.context(phase),output,row,column),_) == True{} : Bool}
  RowMajor.in_bounds(output,Domain.outputs(g),rows,columns,row,column,output_inside,row_inside,column_inside)
def selected_sample(+phase: Jobs.Phase,+original: Buffer.Model<Tile.Tile8>,+block: Nat,+tap: Nat) -> Tile.Tile8:
  Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,original),Values.address(Kernel.weight_view(Jobs.context(phase)),Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap)))

def coordinate_position(+phase: Jobs.Phase,+block: Nat,+output: Nat,+row: Nat,+column: Nat) ->
  {ViewLanes.position(Views.axis_extents_of(Kernel.weight_view(Jobs.context(phase))),[block,output,row,column],0n) ==
    Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),Packing.phase_index(Jobs.context(phase),output,row,column)) : Nat}:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,+o,h,w,k,s,p,gh,gw} = g
  +rows = U32.to_nat(Packing.phase_rows(Jobs.context(phase)))
  +columns = U32.to_nat(Packing.phase_columns(Jobs.context(phase)))
  +area = Nat.mul(rows,columns)
  +outputs = U32.to_nat(o)
  +prefix = Nat.mul(block,outputs)
  +within = Nat.add(Nat.mul(row,columns),column)
  %Equal.sym(Nat,U32.to_nat(Kernel.phase_inner(Jobs.context(phase))),Nat.mul(outputs,area),PhaseBounds.phase_inner_exact(call,U32.from_nat(y),U32.from_nat(x))) :
    {ViewLanes.position(Views.axis_extents_of(Kernel.weight_view(Jobs.context(phase))),[block,output,row,column],0n) == Nat.add(Nat.mul(block,_),Nat.add(Nat.mul(output,area),within)) : Nat}
  Equal.trans(Nat,ViewLanes.position(Views.axis_extents_of(Kernel.weight_view(Jobs.context(phase))),[block,output,row,column],0n),Nat.add(Nat.mul(Nat.add(prefix,output),area),within),Nat.add(Nat.mul(block,Nat.mul(outputs,area)),Nat.add(Nat.mul(output,area),within)),
    RowMajor.mixed(Nat.add(prefix,output),rows,columns,row,column),
    %Equal.sym(Nat,Nat.mul(Nat.add(prefix,output),area),Nat.add(Nat.mul(prefix,area),Nat.mul(output,area)),RowMajor.distrib(prefix,output,area)) :
      {Nat.add(_,within) == Nat.add(Nat.mul(block,Nat.mul(outputs,area)),Nat.add(Nat.mul(output,area),within)) : Nat}
    %Equal.sym(Nat,Nat.mul(prefix,area),Nat.mul(block,Nat.mul(outputs,area)),RowMajor.assoc(block,outputs,area)) :
      {Nat.add(Nat.add(_,Nat.mul(output,area)),within) == Nat.add(Nat.mul(block,Nat.mul(outputs,area)),Nat.add(Nat.mul(output,area),within)) : Nat}
    Addition.associative(Nat.mul(block,Nat.mul(outputs,area)),Nat.mul(output,area),within))

def coordinate_address(+phase: Jobs.Phase,+block: Nat,+output: Nat,+row: Nat,+column: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(phase))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(phase)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(phase)))) == True{} : Bool}) ->
  {Values.address(Kernel.weight_view(Jobs.context(phase)),Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),Packing.phase_index(Jobs.context(phase),output,row,column))) ==
    ViewLanes.view_terms(Kernel.weight_view(Jobs.context(phase)),[block,output,row,column]) : U32}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  %coordinate_position(phase,block,output,row,column) :
    {Values.address(Kernel.weight_view(Jobs.context(phase)),_) == ViewLanes.view_terms(Kernel.weight_view(Jobs.context(phase)),[block,output,row,column]) : U32}
  ViewLanes.view_address(Kernel.weight_view(Jobs.context(phase)),[block,output,row,column],KernelGemm.Both{block_inside,KernelGemm.Both{output_inside,KernelGemm.Both{row_inside,KernelGemm.Both{column_inside,Unit{}}}}})

def selected_cell(+phase: Jobs.Phase,+original: Buffer.Model<Tile.Tile8>,+block: Nat,+output: Nat,+row: Nat,+column: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(phase))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(phase)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(phase)))) == True{} : Bool}) ->
  {selected_sample(phase,original,block,Packing.phase_index(Jobs.context(phase),output,row,column)) == PhaseCells.cell_tile(original,Jobs.context(phase),block,output,row,column) : Tile.Tile8}:
  +context = Jobs.context(phase)
  +index = Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(context))),Packing.phase_index(context,output,row,column))
  Equal.cong(U32,Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,original)),Values.address(Kernel.weight_view(context),index),PhaseCells.cell_address(context,block,output,row,column),
    Equal.trans(U32,Values.address(Kernel.weight_view(context),index),ViewLanes.view_terms(Kernel.weight_view(context),[block,output,row,column]),PhaseCells.cell_address(context,block,output,row,column),
      coordinate_address(phase,block,output,row,column,block_inside,output_inside,row_inside,column_inside),PhaseCells.cell_view_address(context,block,output,row,column)))

def slot_inside(+phase: Jobs.Phase,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap),U32.to_nat(Weights.phase_count(Jobs.context(phase)))) == True{} : Bool}:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  %Equal.sym(Nat,U32.to_nat(Weights.phase_count(Jobs.context(phase))),Nat.mul(block_count(phase),U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),Window.packed_exact(call,U32.from_nat(y),U32.from_nat(x))) :
    {Nat.is_lt(Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap),_) == True{} : Bool}
  RowMajor.radix(block,block_count(phase),tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase))),block_inside,tap_inside)

def packed_query(+phase: Jobs.Phase,+original: Buffer.Model<Tile.Tile8>,+block: Nat,+tap: Nat) -> PackedValues.Query:
  PackedValues.allocated_query(Index.packing_bounds(phase),original,Index.ordinal(phase),Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap))
def packed_state(+phase: Jobs.Phase,+original: Buffer.Model<Tile.Tile8>,+block: Nat,+tap: Nat) -> Weights.Packed:
  +q = packed_query(phase,original,block,tap)
  PackedValues.complete(q,PackedValues.initial(original,Index.packing_bounds(phase)))
def packed_readback(+phase: Jobs.Phase,+original: Buffer.Model<Tile.Tile8>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}) ->
  {PackedValues.read(packed_query(phase,original,block,tap),packed_state(phase,original,block,tap)) == PackedValues.wanted(packed_query(phase,original,block,tap)) : Tile.Tile8}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  PackedValues.allocated_readback(Index.packing_bounds(phase),original,Index.ordinal(phase),Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap),Index.packing_inside(phase),
    %Equal.sym(Kernel.Phase,Kernel.phase_of(g,U32.from_nat(Index.ordinal(phase))),Jobs.context(phase),Index.phase_matches(phase)) : {Nat.is_lt(Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap),U32.to_nat(Weights.phase_count(_))) == True{} : Bool} slot_inside(phase,block,tap,block_inside,tap_inside))

def packed_weight_sample(+phase: Jobs.Phase,+original: Buffer.Model<Tile.Tile8>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}) ->
  {PackedValues.read(packed_query(phase,original,block,tap),packed_state(phase,original,block,tap)) == selected_sample(phase,original,block,tap) : Tile.Tile8}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +q = packed_query(phase,original,block,tap)
  +position = Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap)
  Equal.trans(Tile.Tile8,PackedValues.read(q,packed_state(phase,original,block,tap)),PackedValues.wanted(q),selected_sample(phase,original,block,tap),
    packed_readback(phase,original,block,tap,block_inside,tap_inside),
    Equal.cong(Kernel.Phase,Tile.Tile8,selected => Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,original),Values.address(Kernel.weight_view(selected),position)),Kernel.phase_of(g,U32.from_nat(Index.ordinal(phase))),Jobs.context(phase),Index.phase_matches(phase)))
def full_state(+phase: Jobs.Phase,+original: Buffer.Model<Tile.Tile8>) -> Weights.Packed:
  packed_state(phase,original,0n,0n)
def full_state_matches(+phase: Jobs.Phase,+original: Buffer.Model<Tile.Tile8>,+block: Nat,+tap: Nat) ->
  {packed_state(phase,original,block,tap) == full_state(phase,original) : Weights.Packed}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}

# ---- The taps of a weight slot ----
