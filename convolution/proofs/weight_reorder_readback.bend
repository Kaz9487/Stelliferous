# Every tile of Weights.run on the real fresh allocation holds the rows it was
# built from. The caller supplies only existing guards and an in-range logical
# coordinate.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_weights.bend as Weights
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/kernel_block_geometry.bend as Geometry
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/kernel_weights_readback.bend as WeightsReadback
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Coordinates
import ./weight_reorder_bounds.bend as WeightBounds

def Goal(+problem: ConvSpec.Problem,+input: Tree.Tree<F32>,+block: Nat,+tap: Nat) -> Data:
  +rows = U32.from_nat(Guards.output_channel_count(problem))
  +inner = Coordinates.native_inner(problem)
  {Tree.read(Tile.Tile8,WeightsLaws.run(rows,inner,input),U32.from_nat(Nat.add(Nat.mul(block,Coordinates.inner(problem)),tap))) ==
    WeightsReadback.sample_linear(input,rows,inner,(U32.from_nat(block) * 8 : U32),U32.from_nat(tap)) : Tile.Tile8}

def allocated_readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+supported: Support.FastSupported<limits,problem>,
  +input: Tree.Tree<F32>,+block: Nat,+tap: Nat,
  block_inside: {Nat.is_lt(block,Geometry.blocks(Guards.output_channel_count(problem))) == True{} : Bool},
  tap_inside: {Nat.is_lt(tap,Coordinates.inner(problem)) == True{} : Bool},ready: WeightBounds.Ready(problem)) -> Goal(problem,input,block,tap):
  Shape.Both{+depth_ok,+room} = ready
  +rows = U32.from_nat(Guards.output_channel_count(problem))
  +inner = Coordinates.native_inner(problem)
  +blocks_exact = WeightBounds.blocks_exact(limits,problem,supported)
  +inner_exact = Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,supported))
  +query = {WeightsLaws.Query{Weights.MappedLayout{rows,inner,inner,Unit{}},input,block,tap,depth_ok,
    %Equal.sym(Nat,U32.to_nat(WeightBounds.blocks(problem)),Geometry.blocks(Guards.output_channel_count(problem)),blocks_exact) :
      {Nat.is_le(Nat.mul(_,U32.to_nat(inner)),U32.to_nat(U32.shln(1,WeightsLaws.mapped_depth(~Unit,Weights.MappedLayout{rows,inner,inner,Unit{}})))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(inner),Coordinates.inner(problem),inner_exact) :
      {Nat.is_le(Nat.mul(Geometry.blocks(Guards.output_channel_count(problem)),_),U32.to_nat(U32.shln(1,WeightsLaws.mapped_depth(~Unit,Weights.MappedLayout{rows,inner,inner,Unit{}})))) == True{} : Bool}
    %WeightBounds.count_exact(limits,problem,valid,supported) : {Nat.is_le(_,U32.to_nat(U32.shln(1,WeightsLaws.mapped_depth(~Unit,Weights.MappedLayout{rows,inner,inner,Unit{}})))) == True{} : Bool}
    room,
    %Equal.sym(Nat,U32.to_nat(WeightBounds.blocks(problem)),Geometry.blocks(Guards.output_channel_count(problem)),blocks_exact) : {Nat.is_lt(block,_) == True{} : Bool}
    block_inside,
    %Equal.sym(Nat,U32.to_nat(inner),Coordinates.inner(problem),inner_exact) : {Nat.is_lt(tap,_) == True{} : Bool}
    tap_inside} : WeightsLaws.Query<Unit>}
  %inner_exact :
    {Tree.read(Tile.Tile8,WeightsLaws.run(rows,inner,input),U32.from_nat(Nat.add(Nat.mul(block,_),tap))) ==
      WeightsReadback.sample_linear(input,rows,inner,(U32.from_nat(block) * 8 : U32),U32.from_nat(tap)) : Tile.Tile8}
  WeightsLaws.mapped_lookup(~Unit,~Weights.identity,query)

def run_readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+supported: Support.FastSupported<limits,problem>,
  +input: Tree.Tree<F32>,+block: Nat,+tap: Nat,
  block_inside: {Nat.is_lt(block,Geometry.blocks(Guards.output_channel_count(problem))) == True{} : Bool},
  tap_inside: {Nat.is_lt(tap,Coordinates.inner(problem)) == True{} : Bool}) -> Goal(problem,input,block,tap):
  allocated_readback(limits,problem,valid,supported,input,block,tap,block_inside,tap_inside,WeightBounds.allocated(limits,problem,valid,supported))
