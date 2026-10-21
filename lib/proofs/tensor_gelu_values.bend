# The staged mapper is the same ordered scalar map, including the tail.
import Base
import ../tensor_access.bend as Access
import ../math_activation.bend as Act
import ../tensor_gelu.bend as Gelu
import ../math_gelu_lanes.bend as Lanes
import ../traversal_lanes.bend as TraversalLanes
import ../traversal_array.bend as Traversal
import ../storage_buffer.bend as Storage
import ./storage_certificate.bend as Certificate
import ./storage_tree_model.bend as Tree
import ./tensor_operations_refinement.bend as Operations
import ./math_gelu_lanes_refinement.bend as GeluLaws

def mapped(+count: Nat,+position: U32,+tree: Tree.Tree<F32>,+length: Nat,+certificate: Certificate.Certificate<F32,Tree.pack(F32,tree)>) ->
  {Operations.observe_owner(~F32,Gelu.mapped(count,position,Storage.Buffer{Tree.pack(F32,tree),length,certificate})) ==
    Operations.Owner{Traversal.map_in_place(~F32,~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),count,position,Unit{},Tree.pack(F32,tree)),length} : Operations.Owner<&1,Array<F32>>}:
  Certificate.Certificate{+depth,+equation} = certificate
  Equal.cong(Array<F32>,Operations.Owner<&1,Array<F32>>,storage => Operations.Owner{storage,length},
    Lanes.map_when(TraversalLanes.covered(depth,position,count),count,position,Tree.pack(F32,tree)),
    Traversal.map_in_place(~F32,~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),count,position,Unit{},Tree.pack(F32,tree)),
    GeluLaws.certified(count,position,Tree.pack(F32,tree),depth,equation))
