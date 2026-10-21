# The staged GELU gather uses the common arbitrary-gather round refinement.
import Base
import ../tensor_access.bend as Access
import ../math_activation.bend as Act
import ../math_gelu_lanes.bend as Gelu
import ../kernel_tile.bend as Tile
import ../traversal_lanes.bend as Lanes
import ../traversal_array.bend as Traversal
import ./traversal_lanes_refinement.bend as LaneLaws
import ./traversal_lanes_certificate.bend as LaneCertificate
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_refinement.bend as StorageLaws

def read8(+context: Unit,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,+g: F32,read: Array<F32> & F32) ->
  {Gelu.gathered(Lanes.gathered(~Unit,~Gelu.identity,context,a,b,c,d,e,f,g,read)) == Lanes.gathered(~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),context,a,b,c,d,e,f,g,read) : Array<F32> & Tile.Tile8}:
  (values,h) = read
  {==}

def gather(values: Array<F32>,+position: U32,+context: Unit) ->
  {Gelu.gather(values,position,context) == Lanes.gather(~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),values,position,context) : Array<F32> & Tile.Tile8}:
  LaneLaws.gather_transform(~Unit,~Gelu.identity,~Access.apply_element(~F32,~Act.gelu_tanh_value),~(context => read => Gelu.gathered(read)),~read8,values,position,context)

def map(+count: Nat,+position: U32,values: Array<F32>) ->
  {Gelu.map(count,position,values) == Lanes.map(~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),count,position,Unit{},values) : Array<F32>}:
  LaneLaws.map_with(~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),~Gelu.gather,~gather,count,position,Unit{},values)

def map_when(+covered: Bool,+count: Nat,+position: U32,values: Array<F32>) ->
  {Gelu.map_when(covered,count,position,values) == Lanes.map_when(~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),covered,count,position,Unit{},values) : Array<F32>}:
  match covered:
    case False{}: {==}
    case True{}: map(count,position,values)

def certified_observed(+count: Nat,+position: U32,-values: Array<F32>,+depth: Nat,
  equation: {values == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,values))) : Array<F32>},observation: StorageLaws.Observation<F32,values>) ->
  {Gelu.map_when(Lanes.covered(depth,position,count),count,position,values) == Traversal.map_in_place(~F32,~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),count,position,Unit{},values) : Array<F32>}:
  StorageLaws.Observation{+tree,+packed,reflected} = observation
  %packed : {Gelu.map_when(Lanes.covered(depth,position,count),count,position,_) == Traversal.map_in_place(~F32,~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),count,position,Unit{},_) : Array<F32>}
  Equal.trans(Array<F32>,Gelu.map_when(Lanes.covered(depth,position,count),count,position,Tree.pack(F32,tree)),
    Lanes.map_when(~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),Lanes.covered(depth,position,count),count,position,Unit{},Tree.pack(F32,tree)),
    Traversal.map_in_place(~F32,~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),count,position,Unit{},Tree.pack(F32,tree)),
    map_when(Lanes.covered(depth,position,count),count,position,Tree.pack(F32,tree)),
    LaneCertificate.certified(~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),count,position,Unit{},Tree.pack(F32,tree),depth,
      %Equal.sym(Array<F32>,Tree.pack(F32,tree),values,packed) : {_ == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,_))) : Array<F32>}
      equation))

def certified(+count: Nat,+position: U32,values: Array<F32>,+depth: Nat,
  equation: {values == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,values))) : Array<F32>}) ->
  {Gelu.map_when(Lanes.covered(depth,position,count),count,position,values) == Traversal.map_in_place(~F32,~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),count,position,Unit{},values) : Array<F32>}:
  certified_observed(count,position,values,depth,equation,StorageLaws.observe(F32,values))
