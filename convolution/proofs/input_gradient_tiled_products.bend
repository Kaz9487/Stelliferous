# Actual filled-input and packed-weight operands denote the specification products.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_rows.bend as Rows
import ../../lib/proofs/kernel_accumulator.bend as Accumulator
import ../../lib/proofs/kernel_weight_gradient_packing.bend as Patches
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/kernel_weights_readback.bend as MappedSamples
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ../../lib/proofs/input_gradient_tiled_job_reads.bend as Reads
import ../../lib/proofs/input_gradient_tiled_product_reads.bend as Runtime
import ../../lib/proofs/input_gradient_tiled_value_reads.bend as Input
import ../../lib/proofs/input_gradient_tiled_source_rows.bend as SourceRows
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_fill.bend as Fill
import ../../lib/proofs/input_gradient_tiled_lookup.bend as Lookup
import ../../lib/proofs/input_gradient_phase_geometry.bend as PhaseGeometry
import ../../lib/proofs/input_gradient_tiled_weight_reads.bend as Weights
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_phase_packing.bend as Packing
import ../../lib/proofs/input_gradient_phase_weights.bend as PhaseWeights
import ../../lib/proofs/input_gradient_phase_bounds.bend as PhaseBounds
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ./input_gradient_sources.bend as Sources
import ./input_gradient_order.bend as Order
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../specification.bend as Specification
import ../../lib/proofs/input_gradient_tiled_reduction.bend as Natural
import ../../lib/proofs/input_gradient_tiled_reduction_indices.bend as Indices
import ../../lib/proofs/kernel_reduction_model.bend as Accumulators
import ../../lib/proofs/input_gradient_tiled_reduction.bend as Reduction
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as PackingValues

def decoded_from(+phase: Jobs.Phase,+output: Nat,+row: Nat,+column: Nat,decoded: Reads.Taps<phase,output,row,column>) -> Sources.WindowDecoded(PhaseBounds.phase_stencil(Jobs.context(phase)),output,row,column):
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Reads.Taps{oe,re,ce} = decoded
  Shape.Both{oe,Shape.Both{re,ce}}
def mapped_half(+upper: Bool,+row: Accumulator.Row,+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+output: Nat,+tap_row: Nat,+tap_column: Nat,
  +enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(phase))) == True{} : Bool},
  +row_inside: {Nat.is_lt(tap_row,U32.to_nat(Packing.phase_rows(Jobs.context(phase)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(tap_column,U32.to_nat(Packing.phase_columns(Jobs.context(phase)))) == True{} : Bool}) ->
  {Accumulator.row(~F32,row,Rows.half(upper,Weights.selected_sample(phase,original,block,Packing.phase_index(Jobs.context(phase),output,tap_row,tap_column)))) ==
    Accumulator.row(~F32,row,Rows.half(upper,PhaseWeights.output_tile(original,(U32.from_nat(block) * 8 : U32),0,PhaseWeights.axis_tap(Packing.phase_y(Jobs.context(phase))),PhaseWeights.axis_tap(Packing.phase_x(Jobs.context(phase))),Jobs.context(phase),output,tap_row,tap_column))) : F32}:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +tap = Packing.phase_index(Jobs.context(phase),output,tap_row,tap_column)
  +native = MappedSamples.sample_linear(Buffer.tree(~F32,original),Domain.channel_word(g),(k * k : U32),(U32.from_nat(block) * 8 : U32),Kernel.weight_offset(Jobs.context(phase),U32.from_nat(tap)))
  Equal.trans(F32,Accumulator.row(~F32,row,Rows.half(upper,Weights.selected_sample(phase,original,block,tap))),Accumulator.row(~F32,row,Rows.half(upper,native)),
    Accumulator.row(~F32,row,Rows.half(upper,PhaseWeights.output_tile(original,(U32.from_nat(block) * 8 : U32),0,PhaseWeights.axis_tap(Packing.phase_y(Jobs.context(phase))),PhaseWeights.axis_tap(Packing.phase_x(Jobs.context(phase))),Jobs.context(phase),output,tap_row,tap_column))),
    Equal.cong(Tile.Tile8,F32,tile => Accumulator.row(~F32,row,Rows.half(upper,tile)),Weights.selected_sample(phase,original,block,tap),native,Weights.selected_sample_phase(phase,original,block,tap)),
    Sources.phase_mapped_weight(upper,row,call,U32.from_nat(y),U32.from_nat(x),(U32.from_nat(block) * 8 : U32),original,output,tap_row,tap_column,decoded_from(phase,output,tap_row,tap_column,Reads.taps(phase,enabled,output,tap_row,tap_column,output_inside,row_inside,column_inside))))

def weight_tile(+reduction: Chunks.Reduction,+original: Buffer.Model<F32>) -> Tile.Tile8:
  +phase = Chunks.phase(Chunks.selection(reduction))
  +block = Nat.div(Chunks.unit(reduction),2n)
  +tap = Packing.phase_index(Jobs.context(phase),Chunks.output(reduction),Chunks.tap_row(reduction),Chunks.tap_column(reduction))
  +query = Weights.packed_query(phase,original,block,tap)
  Weights.weight(Packed.read(query,Packed.complete(query,Packed.initial(query))))
def weight_half_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<F32>,+upper: Bool,+row: Accumulator.Row) ->
  {Accumulator.row(~F32,row,Rows.half(upper,weight_tile(reduction,original))) ==
    Accumulator.row(~F32,row,Rows.half(upper,PhaseWeights.output_tile(original,(U32.from_nat(Nat.div(Chunks.unit(reduction),2n)) * 8 : U32),0,PhaseWeights.axis_tap(Packing.phase_y(Jobs.context(Chunks.phase(Chunks.selection(reduction))))),PhaseWeights.axis_tap(Packing.phase_x(Jobs.context(Chunks.phase(Chunks.selection(reduction))))),Jobs.context(Chunks.phase(Chunks.selection(reduction))),Chunks.output(reduction),Chunks.tap_row(reduction),Chunks.tap_column(reduction)))) : F32}:
  Chunks.Reduction{+selected,line,local_group,+unit,+output,+tr,+tc} = reduction
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,room,fits,+enabled,+ji,+oi,+ri,+coli} = valid
  Jobs.Inside{li,gi,+ui} = ji
  +phase = Chunks.phase(selected)
  +block = Nat.div(unit,2n)
  +tap = Packing.phase_index(Jobs.context(phase),output,tr,tc)
  +block_inside = {%Equal.sym(Nat,Weights.block_count(phase),Ceiling.ceiling(Domain.channels(g),8n),Weights.block_count_exact(phase)) : {Nat.is_lt(block,_) == True{} : Bool} Weights.block_inside(phase,unit,ui) : {Nat.is_lt(block,Weights.block_count(phase)) == True{} : Bool}}
  +tap_inside = Weights.tap_index_inside(phase,output,tr,tc,oi,ri,coli)
  Equal.trans(F32,Accumulator.row(~F32,row,Rows.half(upper,weight_tile(reduction,original))),Accumulator.row(~F32,row,Rows.half(upper,Weights.selected_sample(phase,original,block,tap))),
    Accumulator.row(~F32,row,Rows.half(upper,PhaseWeights.output_tile(original,(U32.from_nat(block) * 8 : U32),0,PhaseWeights.axis_tap(Packing.phase_y(Jobs.context(phase))),PhaseWeights.axis_tap(Packing.phase_x(Jobs.context(phase))),Jobs.context(phase),output,tr,tc))),
    Equal.cong(Tile.Tile8,F32,tile => Accumulator.row(~F32,row,Rows.half(upper,tile)),weight_tile(reduction,original),Weights.selected_sample(phase,original,block,tap),Weights.packed_weight_sample(phase,original,block,tap,block_inside,tap_inside)),mapped_half(upper,row,phase,original,block,output,tr,tc,enabled,oi,ri,coli))

def channel_base(+reduction: Chunks.Reduction) -> U32:
  (U32.from_nat(Nat.div(Chunks.unit(reduction),2n)) * 8 : U32)
def flat(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+reduction: Chunks.Reduction,+image: U32,gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>) -> Order.Flat:
  +selected = Chunks.selection(reduction)
  +g = Jobs.geometry(Chunks.phase(selected))
  Order.Flat{Sources.problem(g,Sources.models_reader(gradient,weights),U32.to_nat(image)),Nat.add(U32.to_nat(channel_base(reduction)),Patches.patch_key(upper,row)),Jobs.physical_row(Chunks.work(selected),Chunks.line(reduction)),
    PhaseBounds.pixel(U32.from_nat(Jobs.phase_x(Chunks.phase(selected))),Domain.stride_word(g),(Input.input_column(reduction) + U32.from_nat(lane) : U32))}
def product(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+reduction: Chunks.Reduction,+weights: Buffer.Model<F32>) -> Specification.Product:
  Specification.Product{Tile.element_at(TileLaws.lane_of(lane),Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))),Accumulator.row(~F32,row,Rows.half(upper,weight_tile(reduction,weights)))}
def product_spec(~upper: Bool,+row: Accumulator.Row,+lane: Nat,+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+image_id: U32,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +column_inside: {Nat.is_lt(Nat.add(U32.to_nat(Input.input_column(reduction)),lane),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_x(Jobs.context(Chunks.phase(Chunks.selection(reduction))))))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(U32.to_nat(channel_base(reduction)),Patches.patch_key(upper,row)),Domain.channels(Jobs.geometry(Chunks.phase(Chunks.selection(reduction))))) == True{} : Bool},
  +range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(reduction))))>,
  covers: SourceRows.Covers<Chunks.leaf_band(reduction,valid),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction))))>,
  stored: SourceRows.Stored<Input.image(reduction,valid,gradient,Packing.input_base(Jobs.context(Chunks.phase(Chunks.selection(reduction))),image_id),Packing.gradient_step(Jobs.context(Chunks.phase(Chunks.selection(reduction))))),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(reduction))))>) ->
  {product(upper,row,lane,reduction,weights) == Order.flat_product(flat(upper,row,lane,reduction,image_id,gradient,weights),Packing.phase_index(Jobs.context(Chunks.phase(Chunks.selection(reduction))),Chunks.output(reduction),Chunks.tap_row(reduction),Chunks.tap_column(reduction))) : Specification.Product}:
  Chunks.Reduction{+selected,+line,+local_group,+unit,+output,+tr,+tc} = reduction
  Chunks.Selection{+query,+y,+x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,leaf_range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,room,fits,enabled,+ji,+oi,+ri,+coli} = valid
  +phase = Jobs.context(Chunks.phase(selected))
  +native_input = Tile.element_at(TileLaws.lane_of(lane),Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction))))
  +native_weight = Accumulator.row(~F32,row,Rows.half(upper,weight_tile(reduction,weights)))
  +pure_input = Tile.element_at(TileLaws.lane_of(lane),Input.phase_input_tile(reduction,image_id,gradient))
  +pure_weight = Accumulator.row(~F32,row,Rows.half(upper,PhaseWeights.output_tile(weights,channel_base(reduction),0,PhaseWeights.axis_tap(Packing.phase_y(phase)),PhaseWeights.axis_tap(Packing.phase_x(phase)),phase,output,tr,tc)))
  +target = Order.flat_product(flat(upper,row,lane,reduction,image_id,gradient,weights),Packing.phase_index(phase,output,tr,tc))
  Equal.trans(Specification.Product,product(upper,row,lane,reduction,weights),Specification.Product{pure_input,pure_weight},target,
    Equal.trans(Specification.Product,Specification.Product{native_input,native_weight},Specification.Product{pure_input,native_weight},Specification.Product{pure_input,pure_weight},
      Equal.cong(F32,Specification.Product,value => Specification.Product{value,native_weight},native_input,pure_input,Input.input_lane(reduction,valid,gradient,image_id,TileLaws.lane_of(lane),range,covers,stored)),
      Equal.cong(F32,Specification.Product,value => Specification.Product{pure_input,value},native_weight,pure_weight,weight_half_matches(reduction,valid,weights,upper,row))),
    Sources.tile_product_spec(~upper,row,lane,call,U32.from_nat(y),U32.from_nat(x),image_id,channel_base(reduction),Jobs.row_native(Chunks.work(selected),line),Input.input_column(reduction),gradient,weights,output,tr,tc,Reads.phase_y_inside(Chunks.phase(selected)),Reads.phase_x_inside(Chunks.phase(selected)),Chunks.leaf_position_inside(reduction,valid),column_inside,oi,ri,coli,channel_inside,lane_inside))

def runtime_product(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+reduction: Chunks.Reduction,+weights: Buffer.Model<F32>) -> Specification.Product:
  Specification.Product{Tile.element_at(TileLaws.lane_of(lane),Runtime.input_read(reduction,weights)),Accumulator.row(~F32,row,Rows.half(upper,Runtime.weight_read(reduction,weights)))}
def runtime_product_spec(~upper: Bool,+row: Accumulator.Row,+lane: Nat,+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+bases: Runtime.Bases<Chunks.selection(reduction)>,+image_id: U32,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +column_inside: {Nat.is_lt(Nat.add(U32.to_nat(Input.input_column(reduction)),lane),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_x(Jobs.context(Chunks.phase(Chunks.selection(reduction))))))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(U32.to_nat(channel_base(reduction)),Patches.patch_key(upper,row)),Domain.channels(Jobs.geometry(Chunks.phase(Chunks.selection(reduction))))) == True{} : Bool},
  +range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(reduction))))>,
  covers: SourceRows.Covers<Chunks.leaf_band(reduction,valid),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction))))>,
  stored: SourceRows.Stored<Input.image(reduction,valid,gradient,Packing.input_base(Jobs.context(Chunks.phase(Chunks.selection(reduction))),image_id),Packing.gradient_step(Jobs.context(Chunks.phase(Chunks.selection(reduction))))),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(reduction))))>) ->
  {runtime_product(upper,row,lane,reduction,weights) == Order.flat_product(flat(upper,row,lane,reduction,image_id,gradient,weights),Packing.phase_index(Jobs.context(Chunks.phase(Chunks.selection(reduction))),Chunks.output(reduction),Chunks.tap_row(reduction),Chunks.tap_column(reduction))) : Specification.Product}:
  +wanted_input = Tile.element_at(TileLaws.lane_of(lane),Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction))))
  +runtime_input = Tile.element_at(TileLaws.lane_of(lane),Runtime.input_read(reduction,weights))
  +wanted_weight = Accumulator.row(~F32,row,Rows.half(upper,weight_tile(reduction,weights)))
  +runtime_weight = Accumulator.row(~F32,row,Rows.half(upper,Runtime.weight_read(reduction,weights)))
  Equal.trans(Specification.Product,runtime_product(upper,row,lane,reduction,weights),product(upper,row,lane,reduction,weights),Order.flat_product(flat(upper,row,lane,reduction,image_id,gradient,weights),Packing.phase_index(Jobs.context(Chunks.phase(Chunks.selection(reduction))),Chunks.output(reduction),Chunks.tap_row(reduction),Chunks.tap_column(reduction))),
    Equal.trans(Specification.Product,Specification.Product{runtime_input,runtime_weight},Specification.Product{wanted_input,runtime_weight},Specification.Product{wanted_input,wanted_weight},
      Equal.cong(F32,Specification.Product,value => Specification.Product{value,runtime_weight},runtime_input,wanted_input,Equal.cong(Tile.Tile8,F32,Tile.element_at(TileLaws.lane_of(lane)),Runtime.input_read(reduction,weights),Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction))),Runtime.input_read_matches(reduction,valid,bases,weights))),
      Equal.cong(F32,Specification.Product,value => Specification.Product{wanted_input,value},runtime_weight,wanted_weight,Equal.cong(Tile.Tile8,F32,tile => Accumulator.row(~F32,row,Rows.half(upper,tile)),Runtime.weight_read(reduction,weights),weight_tile(reduction,weights),Runtime.weight_read_matches(reduction,valid,bases,weights)))),
    product_spec(~upper,row,lane,reduction,valid,image_id,gradient,weights,lane_inside,column_inside,channel_inside,range,covers,stored))

# The scalar operand pair of the reindexed real reduction.
def natural_product(~upper: Bool,+row: Accumulator.Row,+lane: Nat,+seed: Chunks.Reduction,+index: Nat,+weights: Buffer.Model<F32>) -> Specification.Product:
  +state = Runtime.initial(seed,weights)
  +key = {Accumulators.RowLane{row,TileLaws.lane_of(lane)} : Accumulators.RowLane}
  Specification.Product{Natural.natural_input()(Natural.natural_source(state),key,index),Natural.natural_weight(~upper)(Natural.natural_source(state),key,index)}
def addressed_product(~upper: Bool,+row: Accumulator.Row,+lane: Nat,+seed: Chunks.Reduction,+address: U32,+weights: Buffer.Model<F32>) -> Specification.Product:
  +state = Runtime.initial(seed,weights)
  +context = Reduction.native_context(state)
  +key = {Accumulators.RowLane{row,TileLaws.lane_of(lane)} : Accumulators.RowLane}
  +packed = Weights.full_state(Chunks.phase(Chunks.selection(seed)),weights)
  Specification.Product{Accumulators.input_at(key,Reduction.input_tile(Fill.tree(Chunks.filled(seed)),Reduction.ReadContext{context,PackingValues.offset_tree(packed)},address)),Accumulators.weight_at(key,Reduction.weights_half(~upper,PackingValues.weight_tree(packed),Reduction.ReadContext{context,PackingValues.offset_tree(packed)},address))}
def natural_product_matches(~upper: Bool,+row: Accumulator.Row,+lane: Nat,+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+index: Nat,+inside: {Nat.is_lt(index,Indices.count(Chunks.phase(Chunks.selection(seed)))) == True{} : Bool},+weights: Buffer.Model<F32>) ->
  {natural_product(~upper,row,lane,seed,index,weights) == runtime_product(upper,row,lane,Indices.at(seed,index),weights) : Specification.Product}:
  Chunks.Reduction{+selected,line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,+ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  %Equal.sym(U32,U32.from_nat(Nat.add(U32.to_nat(ob),index)),(ob + U32.from_nat(index) : U32),Runtime.natural_address(ob,index)) : {addressed_product(~upper,row,lane,seed,_,weights) == runtime_product(upper,row,lane,Indices.at(seed,index),weights) : Specification.Product}
  %Equal.sym(Nat,index,Runtime.tap(Indices.at(seed,index)),Indices.tap_exact(seed,valid,index,inside)) : {addressed_product(~upper,row,lane,seed,(ob + U32.from_nat(_) : U32),weights) == runtime_product(upper,row,lane,Indices.at(seed,index),weights) : Specification.Product}
  {==}
