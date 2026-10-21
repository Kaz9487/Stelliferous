import ../../lib/proofs/kernel_spatial_packing_refinement.bend as LibPacking
# Erase data buffers from the actual prepare recursion, retaining every tile.
import Base
import ../../lib/traversal_partition.bend as Partition
import ../../lib/tensor_band.bend as Bands
import ../../lib/proofs/tensor_band_walk.bend as Walk
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./packing_storage_refinement.bend as PackingStorage
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws

def Layout() -> Data:
  Partition.Tree<&2,Unit,Unit>

def no_block(+context: Unit,+offset: U32,+span: U32,+details: Unit,block: Tree.Tree<Tile.Tile8>) -> Unit:
  Unit{}

def no_details(+context: Unit,+details: Unit) -> Unit:
  Unit{}

def tiles(t: PackingStorage.Tiles()) -> Layout():
  PartitionLaws.map(~&2,~&2,~Tree.Tree<Tile.Tile8>,~Unit,~Unit,~Unit,~Unit,~no_block,~no_details,Unit{},t)
def prepared(p: PackingStorage.Prepared) -> Layout():
  PackingStorage.Prepared{x,t} = p
  tiles(t)

# The parts of the plan the Array executor runs.
def divide(depth: Nat,+offset: U32,+span: U32) -> Layout():
  Walk.shape(Bands.unit_plan(depth,8,offset,span))

def leaf(+offset: U32,+span: U32,+pair: LibPacking.Packed) -> {prepared(PackingStorage.leaf(offset,span,
      pair)) == Partition.Leaf{Unit{},offset,span,Unit{}} : Layout()}:
  LibPacking.Packed{x,y} = pair
  {==}

def attach(+left: PackingStorage.Tiles(),+p: PackingStorage.Prepared) -> {prepared(PackingStorage.attach(left,
      p)) == Partition.Fork{tiles(left),prepared(p)} : Layout()}:
  PackingStorage.Prepared{x,right} = p
  {==}

def after_left(+depth: Nat,+offset: U32,+span: U32,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,+inner: U32,
  +p: PackingStorage.Prepared,ih: @x: Tree.Tree<F32> -> {prepared(PackingStorage.prepare(depth,
        (offset + span / 16 * 8 : U32),(span - span / 16 * 8 : U32),x,h,w,k,stride,pad,output_width,inner)) == divide(depth,(offset + span / 16 * 8 : U32),
      (span - span / 16 * 8 : U32)) : Layout()}) -> {prepared(PackingStorage.after_left(depth,offset,span,h,w,k,stride,pad,output_width,inner,
      p)) == Partition.Fork{prepared(p),divide(depth,(offset + span / 16 * 8 : U32),(span - span / 16 * 8 : U32))} : Layout()}:
  PackingStorage.Prepared{+x,+left} = p
  %ih(x) : {prepared(PackingStorage.after_left(depth,offset,span,h,w,k,stride,pad,output_width,inner,PackingStorage.Prepared{x,
          left})) == Partition.Fork{tiles(left),_} : Layout()}
  attach(left,PackingStorage.prepare(depth,(offset + span / 16 * 8 : U32),(span - span / 16 * 8 : U32),x,h,w,k,stride,pad,output_width,inner))

law prepare:
  for +depth: Nat
  for +offset: U32
  for +span: U32
  for +x: Tree.Tree<F32>
  for +h: U32
  for +w: U32
  for +k: U32
  for +stride: U32
  for +pad: U32
  for +output_width: U32
  for +inner: U32
  {prepared(PackingStorage.prepare(depth,offset,span,x,h,w,k,stride,pad,output_width,inner)) == divide(depth,offset,span) : Layout()}
def prepare(depth,offset,span,x,h,w,k,stride,pad,output_width,inner):
  match depth:
    case 0n:
      leaf(offset,span,PackingStorage.packing(U32.to_nat((span / 8 * inner : U32)),offset,x,
          Tree.fresh(Tile.Tile8,Storage.allocation_depth((span / 8 * inner : U32)),Tile.filled(0.0)),h,w,k,stride,pad,
          output_width,inner))
    case 1n+ +rest:
      +half = (span / 16 * 8 : U32)
      %prepare(rest,offset,half,x,h,w,k,stride,pad,output_width,inner) : {prepared(PackingStorage.prepare(1n+rest,offset,span,x,h,w,k,stride,pad,
            output_width,inner)) == Partition.Fork{_,divide(rest,(offset + half : U32),(span - half : U32))} : Layout()}
      after_left(rest,offset,span,h,w,k,stride,pad,output_width,inner,PackingStorage.prepare(rest,offset,half,x,h,w,k,stride,pad,output_width,inner),
        x => prepare(rest,(offset + half : U32),(span - half : U32),x,h,w,k,stride,pad,output_width,inner))
