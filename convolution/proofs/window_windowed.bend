# The windowed convolution over an output cell shape: the output pieces it
# places in the grid of the shape hold the values of the convolution, on the
# route the runtime picks for it.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_window_pipeline.bend as WindowPipeline
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/math_activation.bend as Activations
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_tiled_cells.bend as TiledCells
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ./image_problems.bend as ImageProblems
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./reduction_coordinate_bounds.bend as Coordinates
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./blocked_convolution_readback.bend as Blocked
import ./window_cell_contents.bend as Cells
import ./output_values.bend as Values
import ./window_packed_values.bend as Calls
import ./window_packed_call.bend as Call
import ./window_skeleton.bend as Skeleton
import ./window_call_evidence.bend as Evidence
import ./window_band_cover.bend as BandCover
import ./window_packed_band.bend as Band
import ../../lib/proofs/kernel_pipeline_bounds.bend as BlockBounds
import ../../lib/proofs/kernel_pipeline_model.bend as Model
import ../../lib/proofs/nat_order.bend as Order
import ./window_reach.bend as Reach
import ../../lib/proofs/tensor_halo_windows.bend as Windows
import ../../lib/tensor_pieces.bend as Pieces
import ../../lib/tensor_tasks.bend as Tasks
import ../../lib/tensor_halo.bend as Halo
import ../../lib/storage_clone.bend as Clone
import ../../lib/proofs/storage_clone_refinement.bend as CloneLaws
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/tensor_placed.bend as Placed
import ../../lib/proofs/tensor_pieces_content.bend as Content
import ../../lib/proofs/tensor_tiled_padding.bend as Padding
import ../../lib/tensor_tiled_map.bend as TiledMap
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/proofs/tensor_tiled_public.bend as Public
import ../../lib/proofs/kernel_pipeline_activation.bend as ActivationProofs
import ../../lib/proofs/word_power_of_two.bend as Power
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ./window_packed_finish.bend as Finish
import ../../lib/proofs/tensor_task_runs.bend as Runs
import ../../lib/proofs/storage_allocation_refinement.bend as AllocationLaws
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/tensor_grid_pieces.bend as Laws
import ../../lib/proofs/u32_division.bend as U32Division
import ./window_input_cells.bend as InputCells
import ./window_positions.bend as Positions
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/tensor_grid_ranges.bend as GridRanges
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ./window_packed.bend as Packed
import ./window_point_placed.bend as Point

# The pieces of the route a convolution takes, packed or pointwise.
def routed(~custom: F32 -> F32,
  ~Logical: Data,
  ~original: Logical -> U32 -> F32,
  direct: Bool,
  +limits: Machine.Limits,
  +logical: Logical,
  +activation: Activations.Activation,
  +channels: Nat,
  +outputs: Nat,
  +height: Nat,
  +width: Nat,
  +kernel: Nat,
  +stride: Nat,
  +padding: Nat,
  +weight_list: List<&2,F32>,
  +bias_list: List<&2,F32>,
  +groups_out: U32,
  +images: Nat,
  +input_outer: U32,
  +input_after: U32,
  +input_lanes: U32,
  +input_groups: U32,
  +outer_word: U32,
  +after_word: U32,
  +lanes_word: U32,
  +groups_word: U32,
  +upper: U32,
  +upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},
  +area_positive: {Nat.is_lt(0n,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool},
  +area_room: {Nat.is_le(Nat.mul(channels,Nat.mul(height,width)),U32.to_nat(upper)) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,
  +bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +parameters: Inputs.Parameters(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),
  +valid: {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)) == True{} : Bool},
  +word_positive: {Nat.is_lt(0n,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},
  +plane_positive: {Nat.is_lt(0n,ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool},
  +plane_exact: {U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},
  +plane_room: {Nat.is_le(U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outputs_room: {Nat.is_le(outputs,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_exact_out: {U32.to_nat(U32.max(groups_out,1)) == Calls.cell_groups(outputs) : Nat},
  +after_exact: {U32.to_nat(after_word) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},
  +lanes_exact: {U32.to_nat(lanes_word) == outputs : Nat},
  +elements_room: {Nat.is_le(Nat.mul(Nat.mul(images,Representation.lane_count(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}})),Representation.after_count(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_exact: {U32.to_nat(groups_word) == Calls.cell_groups(outputs) : Nat},
  +positive: {Nat.is_lt(0n,Calls.cell_groups(outputs)) == True{} : Bool},
  +input_after_exact: {U32.to_nat(input_after) == Nat.mul(height,width) : Nat},
  +input_lanes_exact: {U32.to_nat(input_lanes) == channels : Nat},
  +input_groups_exact: {U32.to_nat(input_groups) == Cells.group_count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)) : Nat},
  +groups_room: {Nat.is_le(Calls.cell_groups(outputs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +images_count_room: {Nat.is_le(Nat.mul(images,Cells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),Guards.height(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +images_outputs_room: {Nat.is_le(Nat.mul(outputs,Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))),Calls.cell_groups(outputs)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +direct_exact: {WindowPipeline.pointwise(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)) == direct : Bool},
  +total: Nat,
  +shape: Reach.Shape<limits,BandCover.height_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),U32.from_nat(Guards.width(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n))),BandCover.kernel_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),BandCover.stride_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),U32.from_nat(ConvSpec.out_height(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n))),BandCover.width_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),Calls.groups_in(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),groups_out,images,total>,
  +total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outer: U32,
  +extent: U32,
  +unit_cells: U32,
  +block: U32,
  +per: U32,
  +count: U32,
  +prefix: U32,
  +grid_shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +grid_cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +input_grid: Grid.Grid,
  +workers: U32,
  +whole: U32,
  +bytes_per_cell: F32,
  +operations: F32,
  +cut_cell_ns: F32,
  +extra: Tasks.Extra,
  -pieces: Pieces.Pieces(~Tile.Tile8),
  +windowed: Windows.Windowed<Tile.Tile8,Representation.Lifted<Logical>,Representation.lifted(~Logical,~original),Packed.input_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room}),WindowPipeline.Convolution,WindowPipeline.reach_of,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),total,Runs.decision_tasks(Tasks.decide_cut(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),Halo.windows(~Tile.Tile8,~WindowPipeline.Convolution,~WindowPipeline.reach_of,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),input_grid,Tile.filled(0.0),Runs.decision_tasks(Tasks.decide_cut(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),pieces)>,
  +weights: Buffer.Model<Tile.Tile8>,
  +bias: Buffer.Model<Tile.Tile8>,
  +weights_exact: {Buffer.tree(~Tile.Tile8,weights) == Packed.weight_cells_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room}) : Tree.Tree<Tile.Tile8>},
  +bias_exact: {Buffer.tree(~Tile.Tile8,bias) == Packed.bias_cells_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room}) : Tree.Tree<Tile.Tile8>},
  +multiple: Nat,
  +unit_exact: {U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},whole)) == Nat.mul(multiple,U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},
  +unit_room: {Nat.is_le(U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},whole)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(count),Nat.mul(Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))),U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)))) == True{} : Bool},
  +units: Nat,
  +units_exact: {U32.to_nat(unit_cells) == Nat.mul(units,U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},
  +grid_inside: {Nat.is_le(U32.to_nat(count),Packed.cells_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room},Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation))) == True{} : Bool}) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<ImageProblems.Context<Logical>>,~Representation.lifted(~ImageProblems.Context<Logical>,~ImageProblems.value(~custom,~Logical,~original)),Packed.logical_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room},Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation))),Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),WindowPipeline.windowed_route(~custom,direct,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),input_grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Tasks.decide_cut(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},workers,whole,bytes_per_cell,operations,cut_cell_ns,extra),pieces,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias))>:
  match direct:
    case False{}: Packed.packed_placed(~custom,~Logical,~original,limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room,total,shape,total_room,outer,extent,unit_cells,block,per,count,prefix,grid_shape,grid_cells,input_grid,workers,whole,bytes_per_cell,operations,cut_cell_ns,extra,pieces,windowed,weights,bias,weights_exact,bias_exact,multiple,unit_exact,unit_room,inside,units,units_exact,grid_inside)
    case True{}: Point.point_placed(~custom,~Logical,~original,limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room,total,shape,total_room,outer,extent,unit_cells,block,per,count,prefix,grid_shape,grid_cells,input_grid,workers,whole,bytes_per_cell,operations,cut_cell_ns,extra,pieces,windowed,weights,bias,direct_exact,weights_exact,bias_exact,multiple,unit_exact,unit_room,inside,units,units_exact,grid_inside)

# The windowed convolution over the grid of an output cell shape, given the
# facts of that grid.
def windowed_grid(~custom: F32 -> F32,
  ~Logical: Data,
  ~original: Logical -> U32 -> F32,
  +limits: Machine.Limits,
  +logical: Logical,
  +activation: Activations.Activation,
  +channels: Nat,
  +outputs: Nat,
  +height: Nat,
  +width: Nat,
  +kernel: Nat,
  +stride: Nat,
  +padding: Nat,
  +weight_list: List<&2,F32>,
  +bias_list: List<&2,F32>,
  +groups_out: U32,
  +images: Nat,
  +input_outer: U32,
  +input_after: U32,
  +input_lanes: U32,
  +input_groups: U32,
  +outer_word: U32,
  +after_word: U32,
  +lanes_word: U32,
  +groups_word: U32,
  +upper: U32,
  +upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},
  +area_positive: {Nat.is_lt(0n,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool},
  +area_room: {Nat.is_le(Nat.mul(channels,Nat.mul(height,width)),U32.to_nat(upper)) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,
  +bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +parameters: Inputs.Parameters(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),
  +valid: {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)) == True{} : Bool},
  +word_positive: {Nat.is_lt(0n,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},
  +plane_positive: {Nat.is_lt(0n,ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool},
  +plane_exact: {U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},
  +plane_room: {Nat.is_le(U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outputs_room: {Nat.is_le(outputs,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_exact_out: {U32.to_nat(U32.max(groups_out,1)) == Calls.cell_groups(outputs) : Nat},
  +after_exact: {U32.to_nat(after_word) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},
  +lanes_exact: {U32.to_nat(lanes_word) == outputs : Nat},
  +elements_room: {Nat.is_le(Nat.mul(Nat.mul(images,Representation.lane_count(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}})),Representation.after_count(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_exact: {U32.to_nat(groups_word) == Calls.cell_groups(outputs) : Nat},
  +positive: {Nat.is_lt(0n,Calls.cell_groups(outputs)) == True{} : Bool},
  +input_after_exact: {U32.to_nat(input_after) == Nat.mul(height,width) : Nat},
  +input_lanes_exact: {U32.to_nat(input_lanes) == channels : Nat},
  +input_groups_exact: {U32.to_nat(input_groups) == Cells.group_count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)) : Nat},
  +groups_room: {Nat.is_le(Calls.cell_groups(outputs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +images_count_room: {Nat.is_le(Nat.mul(images,Cells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),Guards.height(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +images_outputs_room: {Nat.is_le(Nat.mul(outputs,Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))),Calls.cell_groups(outputs)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +total: Nat,
  +shape: Reach.Shape<limits,BandCover.height_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),U32.from_nat(Guards.width(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n))),BandCover.kernel_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),BandCover.stride_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),U32.from_nat(ConvSpec.out_height(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n))),BandCover.width_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),Calls.groups_in(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),groups_out,images,total>,
  +total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells: List<&2,U32>,
  +facts: Laws.GridFacts<limits,Grid.grid(cells,Grid.piece_target())>,
  +whole_units: GridRanges.Multiple<U32.to_nat(GridRanges.unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target()))),U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))>,
  +counted: {U32.to_nat(Grid.count_of(Grid.grid(cells,Grid.piece_target()))) == ShapeLaws.elements(cells) : Nat},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +bound_cells: {Nat.is_le(ShapeLaws.elements(cells),Nat.mul(Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))),U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)))) == True{} : Bool},
  +input_grid: Grid.Grid,
  +workers: U32,
  +whole: U32,
  +bytes_per_cell: F32,
  +operations: F32,
  +cut_cell_ns: F32,
  +extra: Tasks.Extra,
  -pieces: Pieces.Pieces(~Tile.Tile8),
  +windowed: Windows.Windowed<Tile.Tile8,Representation.Lifted<Logical>,Representation.lifted(~Logical,~original),Packed.input_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room}),WindowPipeline.Convolution,WindowPipeline.reach_of,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),total,Runs.decision_tasks(Tasks.decide_cut(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),Halo.windows(~Tile.Tile8,~WindowPipeline.Convolution,~WindowPipeline.reach_of,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),input_grid,Tile.filled(0.0),Runs.decision_tasks(Tasks.decide_cut(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),pieces)>,
  +weights: Buffer.Model<Tile.Tile8>,
  +bias: Buffer.Model<Tile.Tile8>,
  +weights_exact: {Buffer.tree(~Tile.Tile8,weights) == Packed.weight_cells_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room}) : Tree.Tree<Tile.Tile8>},
  +bias_exact: {Buffer.tree(~Tile.Tile8,bias) == Packed.bias_cells_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room}) : Tree.Tree<Tile.Tile8>},
  +multiple: Nat,
  +cut_exact: {U32.to_nat(Runs.cut_unit_of(Grid.grid(cells,Grid.piece_target()),whole)) == Nat.mul(multiple,U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},
  +cut_room: {Nat.is_le(U32.to_nat(Runs.cut_unit_of(Grid.grid(cells,Grid.piece_target()),whole)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<ImageProblems.Context<Logical>>,~Representation.lifted(~ImageProblems.Context<Logical>,~ImageProblems.value(~custom,~Logical,~original)),Packed.logical_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room},Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation))),Pieces.ranges_of_grid(Grid.grid(cells,Grid.piece_target())),WindowPipeline.windowed(~custom,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),input_grid,Grid.grid(cells,Grid.piece_target()),Tasks.decide_cut(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,cut_cell_ns,extra),pieces,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias))>:
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+grid_shape,+exact,+grid_cells} = facts
  +kept = Equal.trans(U32,unit_cells,GridRanges.grid_unit_cells(Grid.grid(cells,Grid.piece_target())),GridRanges.unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target())),
    Equal.sym(U32,GridRanges.grid_unit_cells(Grid.grid(cells,Grid.piece_target())),unit_cells,Equal.cong(Grid.Grid,U32,grid => GridRanges.grid_unit_cells(grid),Grid.grid(cells,Grid.piece_target()),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact)),
    Equal.trans(U32,GridRanges.grid_unit_cells(Grid.grid(cells,Grid.piece_target())),GridRanges.grid_unit_cells(Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target()),Grid.piece_target())),GridRanges.unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target())),
      Equal.cong(Grid.Grid,U32,grid => GridRanges.grid_unit_cells(grid),Grid.grid(cells,Grid.piece_target()),Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target()),Grid.piece_target()),Laws.grid_units(cells,Grid.piece_target())),
      GridRanges.unit_cells_kept(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target()),Grid.piece_target())))
  +unit_whole = GridRanges.multiple_moved(U32.to_nat(unit_cells),U32.to_nat(GridRanges.unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target()))),U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)),
    Equal.cong(U32,Nat,word => U32.to_nat(word),unit_cells,GridRanges.unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target())),kept),whole_units)
  +units = GridRanges.multiple_times(U32.to_nat(unit_cells),U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)),unit_whole)
  +units_exact = GridRanges.multiple_exact(U32.to_nat(unit_cells),U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)),unit_whole)
  +count_exact = Equal.trans(Nat,U32.to_nat(count),U32.to_nat(Grid.count_of(Grid.grid(cells,Grid.piece_target()))),ShapeLaws.elements(cells),
    Equal.sym(Nat,U32.to_nat(Grid.count_of(Grid.grid(cells,Grid.piece_target()))),U32.to_nat(count),Equal.cong(Grid.Grid,Nat,grid => U32.to_nat(Grid.count_of(grid)),Grid.grid(cells,Grid.piece_target()),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact)),counted)
  +nonzero = Arithmetic.word_positive_nonzero(count,{%Equal.sym(Nat,U32.to_nat(count),ShapeLaws.elements(cells),count_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    nonempty : {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}})
  +inside = {%Equal.sym(Nat,U32.to_nat(count),ShapeLaws.elements(cells),count_exact) : {Nat.is_le(_,Nat.mul(Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))),U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)))) == True{} : Bool}
    bound_cells : {Nat.is_le(U32.to_nat(count),Nat.mul(Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))),U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)))) == True{} : Bool}}
  +grid_inside = {%plane_exact : {Nat.is_le(U32.to_nat(count),Nat.mul(Nat.mul(images,_),Calls.cell_groups(outputs))) == True{} : Bool}
    %groups_exact_out : {Nat.is_le(U32.to_nat(count),Nat.mul(Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))),_)) == True{} : Bool}
    inside : {Nat.is_le(U32.to_nat(count),Nat.mul(Nat.mul(images,ImageProblems.plane(height,width,kernel,stride,padding)),Calls.cell_groups(outputs))) == True{} : Bool}}
  +unit_exact = {%exact : {U32.to_nat(Runs.cut_unit_of(_,whole)) == Nat.mul(multiple,U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat} cut_exact : {U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},whole)) == Nat.mul(multiple,U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat}}
  +unit_room = {%exact : {Nat.is_le(U32.to_nat(Runs.cut_unit_of(_,whole)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} cut_room :
    {Nat.is_le(U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},whole)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +windowed_lit = {%exact : Windows.Windowed<Tile.Tile8,Representation.Lifted<Logical>,Representation.lifted(~Logical,~original),Packed.input_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room}),WindowPipeline.Convolution,WindowPipeline.reach_of,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),total,Runs.decision_tasks(Tasks.decide_cut(_,workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),Halo.windows(~Tile.Tile8,~WindowPipeline.Convolution,~WindowPipeline.reach_of,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),input_grid,Tile.filled(0.0),Runs.decision_tasks(Tasks.decide_cut(_,workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),pieces)> windowed : Windows.Windowed<Tile.Tile8,Representation.Lifted<Logical>,Representation.lifted(~Logical,~original),Packed.input_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room}),WindowPipeline.Convolution,WindowPipeline.reach_of,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),total,Runs.decision_tasks(Tasks.decide_cut(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),Halo.windows(~Tile.Tile8,~WindowPipeline.Convolution,~WindowPipeline.reach_of,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),input_grid,Tile.filled(0.0),Runs.decision_tasks(Tasks.decide_cut(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),pieces)>}
  %Equal.sym(Grid.Grid,Grid.grid(cells,Grid.piece_target()),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<ImageProblems.Context<Logical>>,~Representation.lifted(~ImageProblems.Context<Logical>,~ImageProblems.value(~custom,~Logical,~original)),Packed.logical_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room},Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation))),Pieces.ranges_of_grid(_),WindowPipeline.windowed(~custom,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),input_grid,_,Tasks.decide_cut(_,workers,whole,bytes_per_cell,operations,cut_cell_ns,extra),pieces,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias))>
  %Equal.sym(Bool,U32.is_zero(count),False{},nonzero) : Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<ImageProblems.Context<Logical>>,~Representation.lifted(~ImageProblems.Context<Logical>,~ImageProblems.value(~custom,~Logical,~original)),Packed.logical_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room},Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation))),Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),WindowPipeline.windowed_when(~custom,_,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),input_grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Tasks.decide_cut(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},workers,whole,bytes_per_cell,operations,cut_cell_ns,extra),pieces,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias))>
  routed(~custom,~Logical,~original,WindowPipeline.pointwise(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room,{==},total,shape,total_room,outer,extent,unit_cells,block,per,count,prefix,grid_shape,grid_cells,input_grid,workers,whole,bytes_per_cell,operations,cut_cell_ns,extra,pieces,windowed_lit,weights,bias,weights_exact,bias_exact,multiple,unit_exact,unit_room,inside,units,units_exact,grid_inside)

# The windowed convolution over an output cell shape: its output pieces
# placed in the grid of the shape hold the values of the convolution.
def windowed_cells(~custom: F32 -> F32,
  ~Logical: Data,
  ~original: Logical -> U32 -> F32,
  +limits: Machine.Limits,
  +logical: Logical,
  +activation: Activations.Activation,
  +channels: Nat,
  +outputs: Nat,
  +height: Nat,
  +width: Nat,
  +kernel: Nat,
  +stride: Nat,
  +padding: Nat,
  +weight_list: List<&2,F32>,
  +bias_list: List<&2,F32>,
  +groups_out: U32,
  +images: Nat,
  +input_outer: U32,
  +input_after: U32,
  +input_lanes: U32,
  +input_groups: U32,
  +outer_word: U32,
  +after_word: U32,
  +lanes_word: U32,
  +groups_word: U32,
  +upper: U32,
  +upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},
  +area_positive: {Nat.is_lt(0n,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool},
  +area_room: {Nat.is_le(Nat.mul(channels,Nat.mul(height,width)),U32.to_nat(upper)) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,
  +bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +parameters: Inputs.Parameters(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),
  +valid: {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)) == True{} : Bool},
  +word_positive: {Nat.is_lt(0n,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},
  +plane_positive: {Nat.is_lt(0n,ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool},
  +plane_exact: {U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},
  +plane_room: {Nat.is_le(U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outputs_room: {Nat.is_le(outputs,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_exact_out: {U32.to_nat(U32.max(groups_out,1)) == Calls.cell_groups(outputs) : Nat},
  +after_exact: {U32.to_nat(after_word) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},
  +lanes_exact: {U32.to_nat(lanes_word) == outputs : Nat},
  +elements_room: {Nat.is_le(Nat.mul(Nat.mul(images,Representation.lane_count(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}})),Representation.after_count(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_exact: {U32.to_nat(groups_word) == Calls.cell_groups(outputs) : Nat},
  +positive: {Nat.is_lt(0n,Calls.cell_groups(outputs)) == True{} : Bool},
  +input_after_exact: {U32.to_nat(input_after) == Nat.mul(height,width) : Nat},
  +input_lanes_exact: {U32.to_nat(input_lanes) == channels : Nat},
  +input_groups_exact: {U32.to_nat(input_groups) == Cells.group_count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)) : Nat},
  +groups_room: {Nat.is_le(Calls.cell_groups(outputs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +images_count_room: {Nat.is_le(Nat.mul(images,Cells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),Guards.height(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +images_outputs_room: {Nat.is_le(Nat.mul(outputs,Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))),Calls.cell_groups(outputs)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +total: Nat,
  +shape: Reach.Shape<limits,BandCover.height_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),U32.from_nat(Guards.width(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n))),BandCover.kernel_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),BandCover.stride_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),U32.from_nat(ConvSpec.out_height(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n))),BandCover.width_word(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),Calls.groups_in(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)),groups_out,images,total>,
  +total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells: List<&2,U32>,
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +divides: GridRanges.Multiple<GridRanges.last_extent(cells),U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))>,
  +bound_cells: {Nat.is_le(ShapeLaws.elements(cells),Nat.mul(Nat.mul(images,U32.to_nat(U32.max(WindowPipeline.plane_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))),U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)))) == True{} : Bool},
  +input_grid: Grid.Grid,
  +workers: U32,
  +whole: U32,
  +bytes_per_cell: F32,
  +operations: F32,
  +cut_cell_ns: F32,
  +extra: Tasks.Extra,
  -pieces: Pieces.Pieces(~Tile.Tile8),
  +windowed: Windows.Windowed<Tile.Tile8,Representation.Lifted<Logical>,Representation.lifted(~Logical,~original),Packed.input_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room}),WindowPipeline.Convolution,WindowPipeline.reach_of,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),total,Runs.decision_tasks(Tasks.decide_cut(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),Halo.windows(~Tile.Tile8,~WindowPipeline.Convolution,~WindowPipeline.reach_of,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),input_grid,Tile.filled(0.0),Runs.decision_tasks(Tasks.decide_cut(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),pieces)>,
  +weights: Buffer.Model<Tile.Tile8>,
  +bias: Buffer.Model<Tile.Tile8>,
  +weights_exact: {Buffer.tree(~Tile.Tile8,weights) == Packed.weight_cells_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room}) : Tree.Tree<Tile.Tile8>},
  +bias_exact: {Buffer.tree(~Tile.Tile8,bias) == Packed.bias_cells_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room}) : Tree.Tree<Tile.Tile8>},
  +multiple: Nat,
  +cut_exact: {U32.to_nat(Runs.cut_unit_of(Grid.grid(cells,Grid.piece_target()),whole)) == Nat.mul(multiple,U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},
  +cut_room: {Nat.is_le(U32.to_nat(Runs.cut_unit_of(Grid.grid(cells,Grid.piece_target()),whole)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<ImageProblems.Context<Logical>>,~Representation.lifted(~ImageProblems.Context<Logical>,~ImageProblems.value(~custom,~Logical,~original)),Packed.logical_of(~Logical,~original,Packed.WindowProblem{limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room},Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation))),Pieces.ranges_of_grid(Grid.grid(cells,Grid.piece_target())),WindowPipeline.windowed(~custom,Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation),input_grid,Grid.grid(cells,Grid.piece_target()),Tasks.decide_cut(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,cut_cell_ns,extra),pieces,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias))>:
  windowed_grid(~custom,~Logical,~original,limits,logical,activation,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,images,input_outer,input_after,input_lanes,input_groups,outer_word,after_word,lanes_word,groups_word,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,valid,supported,word_positive,plane_positive,plane_exact,plane_room,outputs_room,groups_exact_out,after_exact,lanes_exact,elements_room,groups_exact,positive,input_after_exact,input_lanes_exact,input_groups_exact,groups_room,images_count_room,images_outputs_room,cells_room,total,shape,total_room,cells,GridRanges.grid_facts(limits,cells,Grid.piece_target(),room,nonempty),
    GridRanges.multiple_through(U32.to_nat(GridRanges.unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target()))),GridRanges.last_extent(cells),U32.to_nat(U32.max(WindowPipeline.output_groups_of(Calls.record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)),
      GridRanges.units_last(limits,cells,U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target(),room,nonempty),divides),
    GridRanges.grid_count(limits,cells,Grid.piece_target(),room,nonempty),nonempty,bound_cells,input_grid,workers,whole,bytes_per_cell,operations,cut_cell_ns,extra,pieces,windowed,weights,bias,weights_exact,bias_exact,multiple,cut_exact,cut_room)
