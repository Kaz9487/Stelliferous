# Shape preservation reuses the traversal's gather/scatter witnesses.
import Base
import ../tensor_access.bend as Access
import ../math_activation.bend as Act
import ../math_gelu_lanes.bend as Gelu
import ../kernel_tile.bend as Tile
import ../traversal_lanes.bend as Lanes
import ../traversal_cells.bend as Cells
import ./traversal_lanes_witness.bend as LaneWitness
import ./storage_witness.bend as Witness
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./traversal_witness.bend as TraversalWitness

def gathered(-values: Array<F32>,+position: U32,+context: Unit,
  evidence: LaneWitness.Gathered<values,Lanes.gather(~Unit,~Gelu.identity,values,position,context)>) ->
  LaneWitness.Gathered<values,Gelu.gather(values,position,context)>:
  LaneWitness.Gathered{cell,equation} = evidence
  LaneWitness.Gathered{Gelu.second(Gelu.first(cell)),
    Equal.cong(Array<F32> & Tile.Tile8,Array<F32> & Tile.Tile8,Gelu.gathered,
      Lanes.gather(~Unit,~Gelu.identity,values,position,context),(values,cell),equation)}

def gather(+depth: Nat,-values: Array<F32>,+position: U32,+context: Unit,+witness: Witness.Witness(F32,depth,values)) ->
  LaneWitness.Gathered<values,Gelu.gather(values,position,context)>:
  gathered(values,position,context,LaneWitness.gather(~Unit,~Gelu.identity,depth,values,position,context,witness))

def mapped(+depth: Nat,count: Nat,+position: U32,-values: Array<F32>,witness: Witness.Witness(F32,depth,values)) ->
  Witness.Witness(F32,depth,Gelu.map(count,position,values)):
  LaneWitness.rounds_with(~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),~Gelu.gather,~gather,depth,count,Cells.State{values,Lanes.scratch(),position,0,Unit{}},
    LaneWitness.Evidence{values,Lanes.scratch(),position,0,Unit{},{==},witness,Witness.fresh(Tile.Tile8,6n,Tile.filled(0.0))})

def shape(covered: Bool,+depth: Nat,count: Nat,+position: U32,-values: Array<F32>,witness: Witness.Witness(F32,depth,values)) ->
  Witness.Witness(F32,depth,Gelu.map_when(covered,count,position,values)):
  match covered:
    case True{}: mapped(depth,count,position,values,witness)
    case False{}: TraversalWitness.map_in_place(~F32,~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),depth,count,position,Unit{},values,witness)

def map_when(+covered: Bool,+count: Nat,+position: U32,-values: Array<F32>,certificate: Certificate.Certificate<F32,values>) ->
  Certificate.Certificate<F32,Gelu.map_when(covered,count,position,values)>:
  Certificate.Certificate{+depth,equation} = certificate
  Certificate.Certificate{depth,
    %Certificate.restored(F32,depth,Gelu.map_when(covered,count,position,values),shape(covered,depth,count,position,values,Certificate.witness_at(F32,depth,values,equation))) :
      {_ == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,Gelu.map_when(covered,count,position,values)))) : Array<F32>}
    {==}}
