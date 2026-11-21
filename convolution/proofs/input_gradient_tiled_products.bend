# Every GEMM step reads a gradient tile and a weight cell of its phase.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ../../lib/proofs/input_gradient_tiled_job_reads.bend as Reads
import ../../lib/proofs/input_gradient_tiled_product_reads.bend as Runtime
import ../../lib/proofs/input_gradient_tiled_value_reads.bend as Input
import ../../lib/proofs/input_gradient_tiled_source_rows.bend as SourceRows
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_fill.bend as Fill
import ../../lib/proofs/kernel_input_gradient_fill_refinement.bend as Filling
import ../../lib/proofs/input_gradient_tiled_entry_reads.bend as Lookup
import ../../lib/proofs/input_gradient_phase_geometry.bend as PhaseGeometry
import ../../lib/proofs/input_gradient_tiled_weight_reads.bend as Weights
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_phase_packing.bend as Packing
import ../../lib/proofs/input_gradient_phase_weights.bend as PhaseWeights
import ../../lib/proofs/input_gradient_phase_bounds.bend as PhaseBounds
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_tiled_store_geometry.bend as Channels
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ./input_gradient_cell_sources.bend as Sources
import ./input_gradient_order.bend as Order
import ../specification.bend as Specification

def weight_tile(+reduction: Chunks.Reduction,+original: Buffer.Model<Tile.Tile8>) -> Tile.Tile8:
  +phase = Chunks.phase(Chunks.selection(reduction))
  +unit = Chunks.unit(reduction)
  +tap = Runtime.tap(reduction)
  Packed.read(Weights.packed_query(phase,original,unit,tap),Weights.packed_state(phase,original,unit,tap))

def weight_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<Tile.Tile8>) ->
  {weight_tile(reduction,original) == PhaseWeights.cell_tile(original,Jobs.context(Chunks.phase(Chunks.selection(reduction))),Chunks.unit(reduction),Chunks.output(reduction),Chunks.tap_row(reduction),Chunks.tap_column(reduction)) : Tile.Tile8}:
  Chunks.Reduction{+selected,line,local_group,+unit,+output,+tr,+tc} = reduction
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,source,chunk_row,chunk_group,wb} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,room,fits,enabled,+ji,+oi,+ri,+coli} = valid
  Jobs.Inside{li,gi,+ui} = ji
  +phase = Chunks.phase(selected)
  +tap = Packing.phase_index(Jobs.context(phase),output,tr,tc)
  +block_inside = {%Equal.sym(Nat,Weights.block_count(phase),Ceiling.ceiling(Domain.channels(g),8n),Weights.block_count_exact(phase)) : {Nat.is_lt(unit,_) == True{} : Bool} Weights.block_inside(phase,unit,ui) : {Nat.is_lt(unit,Weights.block_count(phase)) == True{} : Bool}}
  Equal.trans(Tile.Tile8,weight_tile(reduction,original),Weights.selected_sample(phase,original,unit,tap),PhaseWeights.cell_tile(original,Jobs.context(phase),unit,output,tr,tc),
    Weights.packed_weight_sample(phase,original,unit,tap,block_inside,Weights.tap_index_inside(phase,output,tr,tc,oi,ri,coli)),
    Weights.selected_cell(phase,original,unit,output,tr,tc,block_inside,oi,ri,coli))

# The image a leaf reads lies in the batch.
def ImageInside(+query: Chunks.Query,+image: U32) -> Data:
  {Nat.is_lt(U32.to_nat(image),Domain.batch(GeometryChunks.grid_call(Chunks.grid(query)))) == True{} : Bool}

def channel_base(+reduction: Chunks.Reduction) -> U32:
  (U32.from_nat(Chunks.unit(reduction)) * 8 : U32)

def flat(+channel: Nat,+position: Nat,+reduction: Chunks.Reduction,+image: U32,gradient: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>) -> Order.Flat:
  +selected = Chunks.selection(reduction)
  +g = Jobs.geometry(Chunks.phase(selected))
  Order.Flat{Sources.problem(g,Sources.models_reader(g,gradient,weights),U32.to_nat(image)),Nat.add(U32.to_nat(channel_base(reduction)),channel),Jobs.physical_row(Chunks.work(selected),Chunks.line(reduction)),
    PhaseBounds.pixel(U32.from_nat(Jobs.phase_x(Chunks.phase(selected))),Domain.stride_word(g),(Input.input_column(reduction) + U32.from_nat(position) : U32))}

def product(+channel: Nat,+position: Nat,+reduction: Chunks.Reduction,+weights: Buffer.Model<Tile.Tile8>) -> Specification.Product:
  Specification.Product{TileLaws.select_lane(position,Tree.read(Tile.Tile8,Filling.tiles(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))),TileLaws.select_lane(channel,weight_tile(reduction,weights))}

def product_spec(+channel: Nat,+channel_small: {Nat.is_lt(channel,8n) == True{} : Bool},+position: Nat,+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+image_id: U32,+gradient: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,
  +position_small: {Nat.is_lt(position,8n) == True{} : Bool},
  +image_inside: ImageInside(Chunks.query(Chunks.selection(reduction)),image_id),
  +column_inside: {Nat.is_lt(Nat.add(U32.to_nat(Input.input_column(reduction)),position),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_x(Jobs.context(Chunks.phase(Chunks.selection(reduction))))))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(U32.to_nat(channel_base(reduction)),channel),Domain.channels(Jobs.geometry(Chunks.phase(Chunks.selection(reduction))))) == True{} : Bool},
  +range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(reduction))))>,
  covers: SourceRows.Covers<Chunks.leaf_band(reduction,valid),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction))))>,
  stored: SourceRows.Stored<Input.image(reduction,valid,gradient,Packing.cells_base(Jobs.context(Chunks.phase(Chunks.selection(reduction))),image_id)),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(reduction))))>) ->
  {product(channel,position,reduction,weights) == Order.flat_product(flat(channel,position,reduction,image_id,gradient,weights),Runtime.tap(reduction)) : Specification.Product}:
  Chunks.Reduction{+selected,+line,+local_group,+unit,+output,+tr,+tc} = reduction
  Chunks.Selection{+query,+y,+x,yi,xi} = selected
  Chunks.Query{+grid,first,target,source,chunk_row,chunk_group,wb} = query
  GeometryChunks.Grid{+call,e,count,leaf_range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,room,fits,enabled,ji,+oi,+ri,+coli} = valid
  +phase = Jobs.context(Chunks.phase(selected))
  +read = Tree.read(Tile.Tile8,Filling.tiles(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))
  +native_input = TileLaws.select_lane(position,read)
  +native_weight = TileLaws.select_lane(channel,weight_tile(reduction,weights))
  +lane_input = Tile.element_at(TileLaws.lane_of(position),read)
  +pure_input = Tile.element_at(TileLaws.lane_of(position),Input.phase_input_tile(reduction,image_id,gradient))
  +pure_weight = TileLaws.select_lane(channel,PhaseWeights.cell_tile(weights,phase,unit,output,tr,tc))
  +input_same = Equal.trans(F32,native_input,lane_input,pure_input,TileLaws.selected_lane(position,read),
    Input.input_lane(reduction,valid,gradient,image_id,TileLaws.lane_of(position),range,covers,stored))
  +base_exact = Channels.first_word_exact(Channels.channel_job(reduction,valid))
  +natural_channel = {%base_exact : {Nat.is_lt(Nat.add(_,channel),Domain.channels(g)) == True{} : Bool} channel_inside : {Nat.is_lt(Nat.add(Nat.mul(unit,8n),channel),Domain.channels(g)) == True{} : Bool}}
  Equal.trans(Specification.Product,product(channel,position,reduction,weights),Specification.Product{pure_input,pure_weight},Order.flat_product(flat(channel,position,reduction,image_id,gradient,weights),Runtime.tap(reduction)),
    Equal.trans(Specification.Product,Specification.Product{native_input,native_weight},Specification.Product{pure_input,native_weight},Specification.Product{pure_input,pure_weight},
      Equal.cong(F32,Specification.Product,value => Specification.Product{value,native_weight},native_input,pure_input,input_same),
      Equal.cong(Tile.Tile8,Specification.Product,value => Specification.Product{pure_input,TileLaws.select_lane(channel,value)},weight_tile(reduction,weights),PhaseWeights.cell_tile(weights,phase,unit,output,tr,tc),weight_matches(reduction,valid,weights))),
    %Equal.sym(Nat,U32.to_nat(channel_base(reduction)),Nat.mul(unit,8n),base_exact) :
      {Specification.Product{pure_input,pure_weight} == Order.flat_product(Sources.phase_flat(g,U32.from_nat(y),U32.from_nat(x),Jobs.row_native(Chunks.work(selected),line),(Input.input_column(reduction) + U32.from_nat(position) : U32),image_id,Nat.add(_,channel),gradient,weights),Runtime.tap(reduction)) : Specification.Product}
    Sources.phase_product(call,U32.from_nat(y),U32.from_nat(x),image_id,unit,Jobs.row_native(Chunks.work(selected),line),Input.input_column(reduction),gradient,weights,output,tr,tc,channel,position,image_inside,channel_small,position_small,oi,natural_channel,Reads.phase_y_inside(Chunks.phase(selected)),Reads.phase_x_inside(Chunks.phase(selected)),ri,coli,
      PhaseBounds.row_facts(call,U32.from_nat(y),Jobs.row_native(Chunks.work(selected),line),tr,Reads.phase_y_inside(Chunks.phase(selected)),Chunks.leaf_position_inside(reduction,valid),ri),
      PhaseBounds.column_lane_facts(call,U32.from_nat(x),Input.input_column(reduction),position,tc,Reads.phase_x_inside(Chunks.phase(selected)),column_inside,coli)))
