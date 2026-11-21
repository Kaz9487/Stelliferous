# A rectangular stencil packs a contiguous prefix of canonical eight-lane
# entries. Geometry and counter bounds are independent of source values.
import Base
import ../kernel_spatial_packing.bend as Packing
import ../kernel_tile.bend as Tile
import ./kernel_spatial_packing_samples.bend as Samples
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./traversal_map_readback.bend as MapReadback
import ./natural_addition.bend as Addition

def increment_relation(+prefix: Nat,+index: Nat,+q: Nat,located: {index == Nat.add(prefix,q) : Nat}) ->
  {1n+index == Nat.add(prefix,1n+q) : Nat}:
  %Equal.sym(Nat,Nat.add(prefix,1n+q),1n+Nat.add(prefix,q),Addition.successor_right(prefix,q)) : {1n+index == _ : Nat}
  Equal.cong(Nat,Nat,value => 1n+value,index,Nat.add(prefix,q),located)

