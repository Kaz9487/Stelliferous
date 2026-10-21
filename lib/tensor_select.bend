# Three-input selection: one band kernel over operands brought to one layout.
import Base
import ./tensor.bend as Tensors
import ./tensor_shape.bend as Shapes
import ./tensor_lanes.bend as Lanes
import ./tensor_band.bend as Bands
import ./tensor_band_schedule.bend as Schedule
import ./storage_buffer.bend as Storage
import ./traversal_partition.bend as Partition

def expanded(same: Bool,shape: List<&2,U32>,tensor: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  match same:
    case True{}: tensor
    case False{}: Tensors.broadcast_to(~F32,tensor,shape)

def selected_leaf(geometry: Bands.Geometry,first: U32,rows: U32,
  input: Bands.Paired<Storage.Buffer<F32>,Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>>,resource: Unit) -> Bands.Bands(~F32):
  match input:
    case Bands.Both{yes,Bands.Both{mask,no}}:
      +rows = rows
      Partition.Leaf{Lanes.select(U32.to_nat(Bands.local_count(geometry,rows)),mask,yes,no),first,rows,Unit{}}
    case Bands.Both{yes,Bands.Only{mask,own_first,own_rows}}: Partition.Leaf{yes,first,rows,Unit{}}
    case Bands.Only{yes,own_first,own_rows}: Partition.Leaf{yes,own_first,own_rows,Unit{}}

def selected_bands(+geometry: Bands.Geometry,+layout: Bands.Plan,+parallel: Bool,mask: Bands.Bands(~F32),yes: Bands.Bands(~F32),no: Bands.Bands(~F32)) -> Bands.Bands(~F32):
  Bands.each(~Bands.Paired<Storage.Buffer<F32>,Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>>,~Storage.Buffer<F32>,~Unit,~Bands.Geometry,
    ~Bands.unshared,~selected_leaf,~Bands.left_of(~Storage.Buffer<F32>,~Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>),
    layout,parallel,geometry,
    Bands.zip(~Storage.Buffer<F32>,~Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>,yes,
      Bands.zip(~Storage.Buffer<F32>,~Storage.Buffer<F32>,mask,no)),Unit{})

# The three operands meet in one layout, each arrives in it, and one band
# kernel selects.
def selected(target: Schedule.Target,shape: List<&2,U32>,fill: F32,+workers: U32,
  mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  Schedule.Target{+geometry,+layout,split} = target
  Tensors.Tensor{selected_bands(geometry,layout,Schedule.parallel(geometry,workers,16.0,0.875,layout),
    Tensors.arrive(~F32,geometry,layout,mask),Tensors.arrive(~F32,geometry,layout,yes),Tensors.arrive(~F32,geometry,layout,no)),
    geometry,shape,fill,Tensors.Usable{},workers}

def matching(+shape: List<&2,U32>,+fill: F32,+workers: U32,mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  Tensors.then(~Tensors.Tensor<F32>,~Schedule.Target,~Tensors.Tensor<F32>,Tensors.offer(~F32,mask),mask => mask_target =>
    Tensors.then(~Tensors.Tensor<F32>,~Schedule.Target,~Tensors.Tensor<F32>,Tensors.offer(~F32,yes),yes => yes_target =>
      Tensors.then(~Tensors.Tensor<F32>,~Schedule.Target,~Tensors.Tensor<F32>,Tensors.offer(~F32,no),no => no_target =>
        selected(Schedule.meet(mask_target,Schedule.meet(yes_target,no_target)),shape,fill,workers,mask,yes,no))))

def shapes(same: Bool,+shape: List<&2,U32>,count: Maybe<&2,U32>,mask_shape: List<&2,U32>,yes_shape: List<&2,U32>,no_shape: List<&2,U32>,fill: F32,workers: U32,
  mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  match same count:
    case True{} Some{count}: matching(shape,fill,workers,mask,yes,no)
    case False{} Some{count}:
      matching(shape,fill,workers,expanded(Shapes.same_extents(mask_shape,shape),shape,mask),
        expanded(Shapes.same_extents(yes_shape,shape),shape,yes),expanded(Shapes.same_extents(no_shape,shape),shape,no))
    case same None{}: Tensors.failed(~F32,"select",Tensors.InvalidShape{shape},shape,fill,workers)

def masked_shape(common: Maybe<&2,List<&2,U32>>,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,values_shape: List<&2,U32>,fill: F32,workers: U32,
  mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  match common:
    case Some{+shape}:
      shapes(Shapes.same_extents(mask_shape,yes_shape) && Shapes.same_extents(yes_shape,no_shape),shape,Tensors.count_of(shape),mask_shape,yes_shape,no_shape,fill,workers,mask,yes,no)
    case None{}: Tensors.failed(~F32,"select",Tensors.ShapeMismatch{mask_shape,values_shape},mask_shape,fill,workers)

def values_shape(common: Maybe<&2,List<&2,U32>>,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,no_shape: List<&2,U32>,fill: F32,workers: U32,
  mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  match common:
    case Some{+shape}: masked_shape(Shapes.broadcast(mask_shape,shape),mask_shape,yes_shape,no_shape,shape,fill,workers,mask,yes,no)
    case None{}: Tensors.failed(~F32,"select",Tensors.ShapeMismatch{yes_shape,no_shape},yes_shape,fill,workers)

def status(status: Tensors.Status,mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,fill: F32,workers: U32,
  mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  match status:
    case Tensors.Failed{problem}: Tensors.with_status(~F32,Tensors.Failed{problem},mask)
    case Tensors.Usable{}: values_shape(Shapes.broadcast(yes_shape,no_shape),mask_shape,yes_shape,no_shape,fill,workers,mask,yes,no)

def select(mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  Tensors.with_metadata(~F32,~Tensors.Tensor<F32>,mask,mask => mask_shape => mf => mt => mw =>
    Tensors.with_metadata(~F32,~Tensors.Tensor<F32>,yes,yes => yes_shape => fill => yt => yw =>
      Tensors.with_metadata(~F32,~Tensors.Tensor<F32>,no,no => no_shape => nf => nt => nw =>
        status(Tensors.first_failure(mt,Tensors.first_failure(yt,nt)),mask_shape,yes_shape,no_shape,fill,U32.max(mw,U32.max(yw,nw)),mask,yes,no))))
