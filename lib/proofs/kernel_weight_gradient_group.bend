# A group packs its pixel and gradient tiles once and runs one GEMM job for
# every channel block, tap and output block. The tile of output block b at
# tap t of the group reads back, in lane l below the outputs, the ordered dot
# of the tap's pixels and the gradients of output 8 * b + l over the group's
# positions.
import Base
import ../kernel_weight_gradient.bend as Kernel
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_grouped.bend as Grouped
import ../traversal_array.bend as Traversal
import ../ordered_matrix.bend as OrderedMatrix
import ../machine_limits.bend as Machine
import ./kernel_weight_gradient_packing.bend as Packing
import ./traversal_indexed_values.bend as IndexedValues
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_values.bend as TiledValues
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_spec.bend as GemmSpec
import ./kernel_gemm_jobs.bend as GemmJobs
import ./traversal_reduction_refinement.bend as ReductionLaws
import ./view_writes.bend as ViewWrites
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as Traversals
import ./write_program.bend as Writes
import ./storage_shape.bend as Shape
import ./kernel_tile_refinement.bend as TileLaws
import ./list_storage_readback.bend as ListRead
import ./ordered_reduction_split.bend as Ordered
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./word_subtraction.bend as Subtraction
import ./u32_division.bend as NativeDivision
import ./u32_extrema.bend as NativeExtrema
import ./u32_comparison.bend as Comparison
import ./u32_full_range.bend as FullRange
import ./u32_shift.bend as Shift
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_extrema.bend as Extrema
import ./nat_ceiling.bend as Ceiling
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./binary_radix.bend as Radix
import ./row_major_index.bend as RowMajor
import ./encoded_coordinates.bend as Encoded
import ./machine_limit_bounds.bend as Limits
import ./tensor_allocation_depth.bend as AllocationDepth
import ./word_encoding.bend as Words
import ../traversal_blocks.bend as TraversalBlocks

type Group is Data:
  Group{input: Buffer.Model<Tile.Tile8>,gradient: Buffer.Model<Tile.Tile8>,values: Tree.Tree<Tile.Tile8>}

def represent(result: Group) -> (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>:
  Group{input,gradient,values} = result
  ((Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,gradient)),Tree.pack(Tile.Tile8,values))

type Sources is Data:
  Sources{input: Buffer.Model<Tile.Tile8>,gradient: Buffer.Model<Tile.Tile8>}

def group_sources(state: Group) -> Sources:
  Group{input,gradient,values} = state
  Sources{input,gradient}

def group_output(result: Group) -> Tree.Tree<Tile.Tile8>:
  Group{input,gradient,values} = result
  values

def StorageFrame(depth: Nat,capacity: U32,tree: Tree.Tree<Tile.Tile8>) -> Data:
  Gemm.Both<Shape.Balanced(Tile.Tile8,depth,tree),{Tree.capacity(Tile.Tile8,tree) == capacity : U32}>

def geometry_channels(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  c

def geometry_stride(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  s

def geometry_rows(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  rows

def geometry_columns(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  columns

def geometry_outputs(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  o

# ---- The model ----

# The bias tree of an unbiased run.
def no_bias() -> Tree.Tree<F32>:
  Tree.fresh(F32,Storage.depth_for(1),0.0)

# The writes of the jobs over the packed tiles, from a fresh output.
def product(+plan: Kernel.Plan,+jobs: U32,pixels: Packing.Packed,gradients: Packing.Packed,values: Tree.Tree<Tile.Tile8>) -> Group:
  Packing.Packed{input,pixel_tiles} = pixels
  Packing.Packed{gradient,gradient_tiles} = gradients
  Group{input,gradient,Writes.pure(Tile.Tile8,GemmJobs.all_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.advance,~GemmLaws.LaneFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.kept,U32.to_nat(jobs),0,plan,
    GemmJobs.Readers{pixel_tiles,gradient_tiles,no_bias()}),values)}

def product_matches(+plan: Kernel.Plan,+jobs: U32,+pixels: Packing.Packed,+gradients: Packing.Packed,+values: Tree.Tree<Tile.Tile8>) ->
  {Kernel.product(plan,jobs,Packing.owners(pixels),Packing.owners(gradients),Tree.pack(Tile.Tile8,values)) == represent(product(plan,jobs,pixels,gradients,values)) :
    (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>}:
  Packing.Packed{+input,+pixel_tiles} = pixels
  Packing.Packed{+gradient,+gradient_tiles} = gradients
  +depth = Storage.depth_for(1)
  +count = U32.to_nat(jobs)
  +written = Writes.pure(Tile.Tile8,GemmJobs.all_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.advance,~GemmLaws.LaneFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.kept,count,0,plan,
    GemmJobs.Readers{pixel_tiles,gradient_tiles,no_bias()}),values)
  %Equal.sym(Array<F32>,Array.new(F32,depth,0.0),Tree.pack(F32,Tree.fresh(F32,depth,0.0)),StorageLaws.fresh(F32,depth,0.0)) :
    {((Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,gradient)),Gemm.output_of(~Tile.Tile8,~Array<F32>,Gemm.run(~Kernel.Plan,~Kernel.Cursor(),~Tile.Tile8,~Kernel.job,~Kernel.advance,~Gemm.lane_times,~Gemm.write_tiles(~Gemm.unbiased,~Gemm.kept),count,plan,
      Gemm.Operands{Tree.pack(Tile.Tile8,pixel_tiles),Tree.pack(Tile.Tile8,gradient_tiles),_,Tree.pack(Tile.Tile8,values)}))) == ((Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,gradient)),Tree.pack(Tile.Tile8,written)) :
      (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>}
  Equal.cong(Array<Tile.Tile8>,(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>,output => ((Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,gradient)),output),
    Gemm.output_of(~Tile.Tile8,~Array<F32>,Gemm.run(~Kernel.Plan,~Kernel.Cursor(),~Tile.Tile8,~Kernel.job,~Kernel.advance,~Gemm.lane_times,~Gemm.write_tiles(~Gemm.unbiased,~Gemm.kept),count,plan,
      GemmLaws.represent(~Tile.Tile8,GemmLaws.Operands{pixel_tiles,gradient_tiles,no_bias(),Writes.Initial{values}}))),Tree.pack(Tile.Tile8,written),
    GemmJobs.native_output(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.advance,~Gemm.lane_times,~GemmLaws.LaneFirst{},~GemmLaws.lane_first,~GemmLaws.Tiles{},~Gemm.write_tiles(~Gemm.unbiased,~Gemm.kept),~GemmLaws.unbiased_value,~Gemm.kept,
      ~GemmLaws.tiles_stored(~Gemm.unbiased,~GemmLaws.unbiased_value,~GemmLaws.unbiased_fetched,~Gemm.kept),
      count,plan,pixel_tiles,gradient_tiles,no_bias(),values))

def product_sources(+plan: Kernel.Plan,+jobs: U32,+pixels: Packing.Packed,+gradients: Packing.Packed,+values: Tree.Tree<Tile.Tile8>) ->
  {group_sources(product(plan,jobs,pixels,gradients,values)) == Sources{Packing.packed_input(pixels),Packing.packed_input(gradients)} : Sources}:
  Packing.Packed{+input,+pixel_tiles} = pixels
  Packing.Packed{+gradient,+gradient_tiles} = gradients
  {==}

def product_output(+plan: Kernel.Plan,+jobs: U32,+pixels: Packing.Packed,+gradients: Packing.Packed,+values: Tree.Tree<Tile.Tile8>) ->
  {group_output(product(plan,jobs,pixels,gradients,values)) ==
    Writes.pure(Tile.Tile8,GemmJobs.all_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.advance,~GemmLaws.LaneFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.kept,U32.to_nat(jobs),0,plan,
      GemmJobs.Readers{Packing.packed_output(pixels),Packing.packed_output(gradients),no_bias()}),values) : Tree.Tree<Tile.Tile8>}:
  Packing.Packed{+input,+pixel_tiles} = pixels
  Packing.Packed{+gradient,+gradient_tiles} = gradients
  {==}

# The positions [first, first + count) of an image whose input cells start at
# x_base and whose gradient cells start at d_base.
type Band is Data:
  Band{geometry: Kernel.Geometry,x_base: U32,d_base: U32,first: Nat,count: Nat,input: Buffer.Model<Tile.Tile8>,gradient: Buffer.Model<Tile.Tile8>}

def band_plan(band: Band) -> Kernel.Plan:
  Band{+geometry,x_base,d_base,+first,+count,input,gradient} = band
  Kernel.Geometry{+c,+o,h,w,+k,+s,p,rows,+columns,taps} = geometry
  +positions = U32.from_nat(count)
  +start = U32.from_nat(first)
  +first_row = (start / columns : U32)
  +phases = U32.min(s,k)
  +reach = ((k - 1) / phases : U32)
  +band_rows = ((start + positions - 1) / columns - first_row + 1 + reach : U32)
  Kernel.Plan{((o + 7) / 8 : U32),(k * k : U32),k,phases,band_rows,(columns + reach : U32),columns,(start % columns : U32),positions,c,o}

def band_jobs(band: Band) -> U32:
  Band{+geometry,x_base,d_base,first,count,input,gradient} = band
  Kernel.Geometry{+c,+o,h,w,+k,s,p,rows,columns,taps} = geometry
  (((c + 7) / 8 : U32) * k * k * ((o + 7) / 8 : U32) : U32)

def band_pixels(band: Band) -> Packing.Packed:
  Band{+geometry,+x_base,+d_base,+first,+count,input,gradient} = band
  Kernel.Geometry{+c,o,h,w,+k,+s,p,rows,+columns,taps} = geometry
  +positions = U32.from_nat(count)
  +start = U32.from_nat(first)
  +first_row = (start / columns : U32)
  +phases = U32.min(s,k)
  +reach = ((k - 1) / phases : U32)
  +band_rows = ((start + positions - 1) / columns - first_row + 1 + reach : U32)
  +band_columns = (columns + reach : U32)
  +planes = (((c + 7) / 8 : U32) * phases * phases : U32)
  Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,U32.to_nat(planes),0,Kernel.Pixels{geometry,x_base,phases,first_row,band_rows,band_columns},
    Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for((planes * band_rows * band_columns : U32)),Tile.filled(0.0))})

def band_gradients(band: Band) -> Packing.Packed:
  Band{+geometry,+x_base,+d_base,+first,+count,input,gradient} = band
  Kernel.Geometry{c,+o,h,w,k,s,p,+rows,+columns,taps} = geometry
  +positions = U32.from_nat(count)
  +output_blocks = ((o + 7) / 8 : U32)
  Packing.gradients(Kernel.Positions{(d_base + U32.from_nat(first) * output_blocks : U32),output_blocks,positions},
    U32.to_nat((output_blocks * positions : U32)),gradient,Tree.fresh(Tile.Tile8,Storage.depth_for((output_blocks * positions : U32)),Tile.filled(0.0)))

def band_values(band: Band) -> Tree.Tree<Tile.Tile8>:
  Band{+geometry,x_base,d_base,first,count,input,gradient} = band
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  Tree.fresh(Tile.Tile8,Storage.depth_for((((o + 7) / 8 : U32) * taps : U32)),Tile.filled(0.0))

def band_group(+band: Band) -> Group:
  product(band_plan(band),band_jobs(band),band_pixels(band),band_gradients(band),band_values(band))

def range_band(+geometry: Kernel.Geometry,+x_base: U32,+d_base: U32,range: Grouped.Range,input: Buffer.Model<Tile.Tile8>,gradient: Buffer.Model<Tile.Tile8>) -> Band:
  Grouped.Range{first,count} = range
  Band{geometry,x_base,d_base,first,count,input,gradient}

def group(+geometry: Kernel.Geometry,+x_base: U32,+d_base: U32,range: Grouped.Range,input: Buffer.Model<Tile.Tile8>,gradient: Buffer.Model<Tile.Tile8>) -> Group:
  band_group(range_band(geometry,x_base,d_base,range,input,gradient))

# The folds of the runtime group over the packed sources.
def runtime_pixels(+band: Band) -> Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>:
  Band{+geometry,+x_base,+d_base,+first,+count,+input,gradient} = band
  Kernel.Geometry{+c,o,h,w,+k,+s,p,rows,+columns,taps} = geometry
  +positions = U32.from_nat(count)
  +start = U32.from_nat(first)
  +first_row = (start / columns : U32)
  +phases = U32.min(s,k)
  +reach = ((k - 1) / phases : U32)
  +band_rows = ((start + positions - 1) / columns - first_row + 1 + reach : U32)
  +band_columns = (columns + reach : U32)
  +planes = (((c + 7) / 8 : U32) * phases * phases : U32)
  Traversal.indexed_fold(~(Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>),~Kernel.Pixels,~Kernel.pixel_plane,U32.to_nat(planes),0,Kernel.Pixels{geometry,x_base,phases,first_row,band_rows,band_columns},
    (Buffer.pack(~Tile.Tile8,input),Array.new(Tile.Tile8,Storage.depth_for((planes * band_rows * band_columns : U32)),Tile.filled(0.0))))

def runtime_gradients(+band: Band) -> Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>:
  Band{+geometry,+x_base,+d_base,+first,+count,input,+gradient} = band
  Kernel.Geometry{c,+o,h,w,k,s,p,+rows,+columns,taps} = geometry
  +positions = U32.from_nat(count)
  +output_blocks = ((o + 7) / 8 : U32)
  Traversal.finish(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Kernel.Positions,
    Traversal.run(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Kernel.Positions,~Kernel.gradient_read,~TraversalBlocks.in_place(~Kernel.Positions),U32.to_nat((output_blocks * positions : U32)),
      Traversal.State{Buffer.pack(~Tile.Tile8,gradient),Array.new(Tile.Tile8,Storage.depth_for((output_blocks * positions : U32)),Tile.filled(0.0)),0,
        Kernel.Positions{(d_base + U32.from_nat(first) * output_blocks : U32),output_blocks,positions}}))

def pixels_refined(+band: Band) -> {runtime_pixels(band) == Packing.owners(band_pixels(band)) : Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>}:
  Band{+geometry,+x_base,+d_base,+first,+count,+input,+gradient} = band
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +whole = {Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  +positions = U32.from_nat(count)
  +start = U32.from_nat(first)
  +first_row = (start / columns : U32)
  +phases = U32.min(s,k)
  +reach = ((k - 1) / phases : U32)
  +band_rows = ((start + positions - 1) / columns - first_row + 1 + reach : U32)
  +band_columns = (columns + reach : U32)
  +planes = (((c + 7) / 8 : U32) * phases * phases : U32)
  +depth = Storage.depth_for((planes * band_rows * band_columns : U32))
  +context = {Kernel.Pixels{whole,x_base,phases,first_row,band_rows,band_columns} : Kernel.Pixels}
  +start_state = {Packing.Packed{input,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))} : Packing.Packed}
  %Equal.sym(Array<Tile.Tile8>,Array.new(Tile.Tile8,depth,Tile.filled(0.0)),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))),StorageLaws.fresh(Tile.Tile8,depth,Tile.filled(0.0))) :
    {Traversal.indexed_fold(~(Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>),~Kernel.Pixels,~Kernel.pixel_plane,U32.to_nat(planes),0,context,(Buffer.pack(~Tile.Tile8,input),_)) ==
      Packing.owners(Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,U32.to_nat(planes),0,context,start_state)) : Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>}
  Traversals.indexed_fold_refines_model(~(Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>),~Packing.Packed,~Kernel.Pixels,~Kernel.pixel_plane,~Packing.pixel_plane,~Packing.owners,
    ~Packing.pixel_plane_matches,U32.to_nat(planes),0,context,start_state)

def gradients_refined(+band: Band) -> {runtime_gradients(band) == Packing.owners(band_gradients(band)) : Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>}:
  Band{+geometry,+x_base,+d_base,+first,+count,+input,+gradient} = band
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +positions = U32.from_nat(count)
  +output_blocks = ((o + 7) / 8 : U32)
  +depth = Storage.depth_for((output_blocks * positions : U32))
  +total = U32.to_nat((output_blocks * positions : U32))
  +context = {Kernel.Positions{(d_base + U32.from_nat(first) * output_blocks : U32),output_blocks,positions} : Kernel.Positions}
  +fresh = Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))
  %Equal.sym(Array<Tile.Tile8>,Array.new(Tile.Tile8,depth,Tile.filled(0.0)),Tree.pack(Tile.Tile8,fresh),StorageLaws.fresh(Tile.Tile8,depth,Tile.filled(0.0))) :
    {Traversal.finish(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Kernel.Positions,
      Traversal.run(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Kernel.Positions,~Kernel.gradient_read,~TraversalBlocks.in_place(~Kernel.Positions),total,
        Traversal.State{Buffer.pack(~Tile.Tile8,gradient),_,0,context})) ==
      Packing.owners(Packing.gradients(context,total,gradient,fresh)) : Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>}
  Packing.gradients_matches(context,total,gradient,fresh)

def runtime_values(+band: Band) -> Array<Tile.Tile8>:
  Band{+geometry,x_base,d_base,first,count,input,gradient} = band
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  Array.new(Tile.Tile8,Storage.depth_for((((o + 7) / 8 : U32) * taps : U32)),Tile.filled(0.0))

def values_refined(+band: Band) -> {runtime_values(band) == Tree.pack(Tile.Tile8,band_values(band)) : Array<Tile.Tile8>}:
  Band{+geometry,+x_base,+d_base,+first,+count,+input,+gradient} = band
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  StorageLaws.fresh(Tile.Tile8,Storage.depth_for((((o + 7) / 8 : U32) * taps : U32)),Tile.filled(0.0))

def band_matches(+band: Band) ->
  {Kernel.product(band_plan(band),band_jobs(band),runtime_pixels(band),runtime_gradients(band),runtime_values(band)) == represent(band_group(band)) :
    (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>}:
  +plan = band_plan(band)
  +jobs = band_jobs(band)
  Equal.trans((Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>,
    Kernel.product(plan,jobs,runtime_pixels(band),runtime_gradients(band),runtime_values(band)),
    Kernel.product(plan,jobs,Packing.owners(band_pixels(band)),runtime_gradients(band),runtime_values(band)),represent(band_group(band)),
    Equal.cong(Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>,(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>,
      pixels => Kernel.product(plan,jobs,pixels,runtime_gradients(band),runtime_values(band)),runtime_pixels(band),Packing.owners(band_pixels(band)),pixels_refined(band)),
    Equal.trans((Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>,
      Kernel.product(plan,jobs,Packing.owners(band_pixels(band)),runtime_gradients(band),runtime_values(band)),
      Kernel.product(plan,jobs,Packing.owners(band_pixels(band)),Packing.owners(band_gradients(band)),runtime_values(band)),represent(band_group(band)),
      Equal.cong(Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>,(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>,
        gradients => Kernel.product(plan,jobs,Packing.owners(band_pixels(band)),gradients,runtime_values(band)),runtime_gradients(band),Packing.owners(band_gradients(band)),
        gradients_refined(band)),
      Equal.trans((Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>,
        Kernel.product(plan,jobs,Packing.owners(band_pixels(band)),Packing.owners(band_gradients(band)),runtime_values(band)),
        Kernel.product(plan,jobs,Packing.owners(band_pixels(band)),Packing.owners(band_gradients(band)),Tree.pack(Tile.Tile8,band_values(band))),represent(band_group(band)),
        Equal.cong(Array<Tile.Tile8>,(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>,
          values => Kernel.product(plan,jobs,Packing.owners(band_pixels(band)),Packing.owners(band_gradients(band)),values),runtime_values(band),Tree.pack(Tile.Tile8,band_values(band)),
          values_refined(band)),
        product_matches(plan,jobs,band_pixels(band),band_gradients(band),band_values(band)))))

def group_matches(+geometry: Kernel.Geometry,+x_base: U32,+d_base: U32,+range: Grouped.Range,+input: Buffer.Model<Tile.Tile8>,+gradient: Buffer.Model<Tile.Tile8>) ->
  {Kernel.group(geometry,x_base,d_base,range,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,gradient)) == represent(group(geometry,x_base,d_base,range,input,gradient)) :
    (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  Grouped.Range{+first,+count} = range
  band_matches(Band{Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},x_base,d_base,first,count,input,gradient})

# A group gives back the input and the gradient it read.
def group_sources_kept(+geometry: Kernel.Geometry,+x_base: U32,+d_base: U32,+range: Grouped.Range,+input: Buffer.Model<Tile.Tile8>,+gradient: Buffer.Model<Tile.Tile8>) ->
  {group_sources(group(geometry,x_base,d_base,range,input,gradient)) == Sources{input,gradient} : Sources}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  Grouped.Range{+first,+count} = range
  +whole = {Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  +band = {Band{whole,x_base,d_base,first,count,input,gradient} : Band}
  +positions = U32.from_nat(count)
  +start = U32.from_nat(first)
  +first_row = (start / columns : U32)
  +phases = U32.min(s,k)
  +reach = ((k - 1) / phases : U32)
  +band_rows = ((start + positions - 1) / columns - first_row + 1 + reach : U32)
  +band_columns = (columns + reach : U32)
  +planes = (((c + 7) / 8 : U32) * phases * phases : U32)
  %Equal.sym(Sources,group_sources(band_group(band)),Sources{Packing.packed_input(band_pixels(band)),Packing.packed_input(band_gradients(band))},
      product_sources(band_plan(band),band_jobs(band),band_pixels(band),band_gradients(band),band_values(band))) :
    {_ == Sources{input,gradient} : Sources}
  %Equal.sym(Buffer.Model<Tile.Tile8>,Packing.packed_input(band_pixels(band)),input,
      Packing.fold_kept(~Kernel.Pixels,~Packing.pixel_plane,~Packing.pixel_plane_kept,U32.to_nat(planes),0,Kernel.Pixels{whole,x_base,phases,first_row,band_rows,band_columns},
        Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for((planes * band_rows * band_columns : U32)),Tile.filled(0.0))})) :
    {Sources{_,Packing.packed_input(band_gradients(band))} == Sources{input,gradient} : Sources}
  {==}

# ---- Geometry evidence ----

def channel_count(+geometry: Kernel.Geometry) -> Nat:
  U32.to_nat(geometry_channels(geometry))

def output_count(+geometry: Kernel.Geometry) -> Nat:
  U32.to_nat(geometry_outputs(geometry))

def kernel_size(+geometry: Kernel.Geometry) -> Nat:
  U32.to_nat(Packing.geometry_kernel(geometry))

def stride_size(+geometry: Kernel.Geometry) -> Nat:
  U32.to_nat(geometry_stride(geometry))

def row_count(+geometry: Kernel.Geometry) -> Nat:
  U32.to_nat(geometry_rows(geometry))

def column_count(+geometry: Kernel.Geometry) -> Nat:
  U32.to_nat(geometry_columns(geometry))

def tap_count(+geometry: Kernel.Geometry) -> Nat:
  U32.to_nat(Packing.geometry_taps(geometry))

def square_size(+geometry: Kernel.Geometry) -> Nat:
  Nat.mul(kernel_size(geometry),kernel_size(geometry))

# The phases of a tap axis and the tile rows a phase adds to a band.
def phase_count(+geometry: Kernel.Geometry) -> Nat:
  Nat.min(stride_size(geometry),kernel_size(geometry))

def reach(+geometry: Kernel.Geometry) -> Nat:
  Nat.div(Nat.sub(kernel_size(geometry),1n),phase_count(geometry))

def channel_blocks(+geometry: Kernel.Geometry) -> Nat:
  Ceiling.ceiling(channel_count(geometry),8n)

def output_blocks(+geometry: Kernel.Geometry) -> Nat:
  Ceiling.ceiling(output_count(geometry),8n)

# The pixel tiles of the planes of every output row, and the gradient tiles
# of every output position.
def pixel_extent(+geometry: Kernel.Geometry) -> Nat:
  Nat.mul(Nat.mul(Nat.mul(Nat.mul(channel_blocks(geometry),phase_count(geometry)),phase_count(geometry)),Nat.add(row_count(geometry),reach(geometry))),
    Nat.add(column_count(geometry),reach(geometry)))

def gradient_extent(+geometry: Kernel.Geometry) -> Nat:
  Nat.mul(Nat.mul(output_blocks(geometry),row_count(geometry)),column_count(geometry))

# Geometry evidence supplies the packed, job and output allocation bounds.
type GeometryReady<-geometry: Kernel.Geometry> is Data:
  GeometryReady{limits: Machine.Limits,
    columns: {Nat.is_lt(0n,U32.to_nat(geometry_columns(geometry))) == True{} : Bool},
    kernel: {Nat.is_lt(0n,U32.to_nat(Packing.geometry_kernel(geometry))) == True{} : Bool},
    stride: {Nat.is_lt(0n,U32.to_nat(geometry_stride(geometry))) == True{} : Bool},
    square: {Nat.is_le(Nat.mul(U32.to_nat(Packing.geometry_kernel(geometry)),U32.to_nat(Packing.geometry_kernel(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    taps: {Nat.is_lt(0n,U32.to_nat(Packing.geometry_taps(geometry))) == True{} : Bool},
    outputs: {Nat.is_lt(0n,U32.to_nat(geometry_outputs(geometry))) == True{} : Bool},
    count: {Nat.is_le(Nat.mul(U32.to_nat(Packing.geometry_taps(geometry)),U32.to_nat(geometry_outputs(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    aligned: {U32.is_zero((Machine.capacity(limits) % 8 : U32)) == True{} : Bool},
    taps_exact: {tap_count(geometry) == Nat.mul(channel_count(geometry),square_size(geometry)) : Nat},
    pixels: {Nat.is_le(pixel_extent(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    gradients: {Nat.is_le(gradient_extent(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

type GroupPlan is Data:
  GroupPlan{limits: Machine.Limits,geometry: Kernel.Geometry,
    taps_positive: {Nat.is_lt(0n,U32.to_nat(Packing.geometry_taps(geometry))) == True{} : Bool},
    outputs_positive: {Nat.is_lt(0n,U32.to_nat(geometry_outputs(geometry))) == True{} : Bool},
    bound: {Nat.is_le(Nat.mul(U32.to_nat(Packing.geometry_taps(geometry)),U32.to_nat(geometry_outputs(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    aligned: {U32.is_zero((Machine.capacity(limits) % 8 : U32)) == True{} : Bool}}

def plan_taps(plan: GroupPlan) -> U32:
  GroupPlan{limits,geometry,taps_positive,outputs_positive,bound,aligned} = plan
  Packing.geometry_taps(geometry)

def plan_outputs(plan: GroupPlan) -> U32:
  GroupPlan{limits,geometry,taps_positive,outputs_positive,bound,aligned} = plan
  geometry_outputs(geometry)

def plan_limits(plan: GroupPlan) -> Machine.Limits:
  GroupPlan{limits,geometry,taps_positive,outputs_positive,bound,aligned} = plan
  limits

def plan_output_blocks(plan: GroupPlan) -> U32:
  ((plan_outputs(plan) + 7) / 8 : U32)

# The cells of a group: the output blocks at every tap.
def plan_count(+plan: GroupPlan) -> U32:
  (plan_output_blocks(plan) * plan_taps(plan) : U32)

def plan_cells(+plan: GroupPlan) -> Nat:
  Nat.mul(Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n),U32.to_nat(plan_taps(plan)))

def plan_outputs_room(+plan: GroupPlan) -> {Nat.is_le(U32.to_nat(plan_outputs(plan)),U32.to_nat(Machine.capacity(plan_limits(plan)))) == True{} : Bool}:
  GroupPlan{+limits,+geometry,+taps_positive,outputs_positive,+bound,aligned} = plan
  +taps = U32.to_nat(Packing.geometry_taps(geometry))
  +outputs = U32.to_nat(geometry_outputs(geometry))
  Order.transitive(outputs,Nat.mul(taps,outputs),U32.to_nat(Machine.capacity(limits)),
    %Equal.sym(Nat,Nat.mul(taps,outputs),Nat.mul(outputs,taps),RowMajor.multiply_commutative(taps,outputs)) : {Nat.is_le(outputs,_) == True{} : Bool} RowMajor.product_covers(outputs,taps,taps_positive),bound)

def plan_cells_room(+plan: GroupPlan) -> {Nat.is_le(plan_cells(plan),U32.to_nat(Machine.capacity(plan_limits(plan)))) == True{} : Bool}:
  GroupPlan{+limits,+geometry,taps_positive,outputs_positive,+bound,aligned} = plan
  +taps = U32.to_nat(Packing.geometry_taps(geometry))
  +outputs = U32.to_nat(geometry_outputs(geometry))
  +blocks = Ceiling.ceiling(outputs,8n)
  Order.transitive(Nat.mul(blocks,taps),Nat.mul(taps,outputs),U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(outputs,taps) : {Nat.is_le(Nat.mul(blocks,taps),_) == True{} : Bool}
    RowMajor.mul_le(blocks,outputs,taps,Ceiling.bounded(outputs,8n,{==})),bound)

def plan_count_exact(+plan: GroupPlan) -> {U32.to_nat(plan_count(plan)) == plan_cells(plan) : Nat}:
  GroupPlan{+limits,+geometry,taps_positive,outputs_positive,bound,aligned} = plan
  +taps = Packing.geometry_taps(geometry)
  +outputs = geometry_outputs(geometry)
  +blocks = ((outputs + 7) / 8 : U32)
  +blocks_exact = Encoded.eight_blocks_exact(limits,outputs,plan_outputs_room(plan))
  +native_room = {
    %Equal.sym(Nat,U32.to_nat(blocks),Ceiling.ceiling(U32.to_nat(outputs),8n),blocks_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(taps)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    plan_cells_room(plan) : {Nat.is_le(Nat.mul(U32.to_nat(blocks),U32.to_nat(taps)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  %blocks_exact : {U32.to_nat((blocks * taps : U32)) == Nat.mul(_,U32.to_nat(taps)) : Nat}
  Multiplication.bounded_multiply(blocks,taps,Machine.capacity(limits),Limits.capacity_fits(limits),native_room)

def plan_count_room(+plan: GroupPlan) -> {Nat.is_le(U32.to_nat(plan_count(plan)),U32.to_nat(Machine.capacity(plan_limits(plan)))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(plan_count(plan)),plan_cells(plan),plan_count_exact(plan)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(plan_limits(plan)))) == True{} : Bool}
  plan_cells_room(plan)

def plan_allocation(+plan: GroupPlan) -> AllocationDepth.Sized<plan_count(plan)>:
  GroupPlan{+limits,geometry,taps_positive,outputs_positive,bound,aligned} = plan
  AllocationDepth.bounded(plan_count(plan),Encoded.native_le_transitive(plan_count(plan),Machine.capacity(limits),1073741824,
    %Equal.sym(Bool,U32.is_le(plan_count(plan),Machine.capacity(limits)),Nat.is_le(U32.to_nat(plan_count(plan)),U32.to_nat(Machine.capacity(limits))),Comparison.le(plan_count(plan),Machine.capacity(limits))) : {_ == True{} : Bool} plan_count_room(plan),
    Limits.capacity_allocatable(limits)))

def plan_depth(plan: GroupPlan) -> Nat:
  Storage.depth_for(plan_count(plan))

def plan_capacity(plan: GroupPlan) -> U32:
  U32.shln(1,plan_depth(plan))

def allocation_depth_limit(+count: U32,allocation: AllocationDepth.Sized<count>) -> {Nat.is_le(Storage.depth_for(count),30n) == True{} : Bool}:
  AllocationDepth.Sized{depth_ok,room} = allocation
  depth_ok

def plan_depth_limit(+plan: GroupPlan) -> {Nat.is_le(plan_depth(plan),30n) == True{} : Bool}:
  allocation_depth_limit(plan_count(plan),plan_allocation(plan))

def plan_physical_extent(+plan: GroupPlan) -> {Nat.is_le(plan_cells(plan),U32.to_nat(plan_capacity(plan))) == True{} : Bool}:
  %plan_count_exact(plan) : {Nat.is_le(_,U32.to_nat(plan_capacity(plan))) == True{} : Bool}
  AllocationDepth.allocation_room(plan_count(plan),plan_allocation(plan))

def plan_capacity_bound(+plan: GroupPlan) -> {U32.is_le(plan_capacity(plan),2147483648) == True{} : Bool}:
  Shift.power_fits(plan_depth(plan),plan_depth_limit(plan))

def geometry_plan(+geometry: Kernel.Geometry,domain: GeometryReady<geometry>) -> GroupPlan:
  GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  GroupPlan{limits,geometry,taps,outputs,count,aligned}

def eight_block_inside(+total: Nat,+value: Nat,+inside: {Nat.is_lt(value,total) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(value,8n),Ceiling.ceiling(total,8n)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(Nat.div(value,8n),Ceiling.ceiling(total,8n)),Nat.is_lt(Nat.mul(Nat.div(value,8n),8n),total),True{},
    Ceiling.upper_comparison(total,8n,Nat.div(value,8n),{==}),
    Order.le_lt_trans(Nat.mul(Nat.div(value,8n),8n),value,total,DivisionOrder.product_below(value,8n,{==}),inside))

# Lane output % 8 of the cell of tap tap and output block output / 8.
def plan_query_inside(+plan: GroupPlan,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,plan_taps(plan)) == True{} : Bool},+output_inside: {U32.is_lt(output,plan_outputs(plan)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),U32.to_nat(plan_taps(plan))),U32.to_nat(tap)),U32.to_nat(plan_count(plan))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(plan_count(plan)),plan_cells(plan),plan_count_exact(plan)) :
    {Nat.is_lt(Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),U32.to_nat(plan_taps(plan))),U32.to_nat(tap)),_) == True{} : Bool}
  RowMajor.radix(Nat.div(U32.to_nat(output),8n),Ceiling.ceiling(U32.to_nat(plan_outputs(plan)),8n),U32.to_nat(tap),U32.to_nat(plan_taps(plan)),
    eight_block_inside(U32.to_nat(plan_outputs(plan)),U32.to_nat(output),Bounds.as_nat_lt(output,plan_outputs(plan),output_inside)),Bounds.as_nat_lt(tap,plan_taps(plan),tap_inside))

# ---- A ready band ----

# A band of count positions from first, with its geometry evidence.
type BandReady is Data:
  BandReady{geometry: Kernel.Geometry,domain: GeometryReady<geometry>,x_base: U32,d_base: U32,first: Nat,count: Nat,input: Buffer.Model<Tile.Tile8>,gradient: Buffer.Model<Tile.Tile8>,
    nonempty: {Nat.is_lt(0n,count) == True{} : Bool},
    room: {Nat.is_le(Nat.add(first,count),Nat.mul(row_count(geometry),column_count(geometry))) == True{} : Bool}}

def ready_geometry(+ready: BandReady) -> Kernel.Geometry:
  BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  geometry

def ready_domain(+ready: BandReady) -> GeometryReady<ready_geometry(ready)>:
  BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  domain

def ready_first(+ready: BandReady) -> Nat:
  BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  first

def ready_count(+ready: BandReady) -> Nat:
  BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  count

def ready_band(+ready: BandReady) -> Band:
  BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  Band{geometry,x_base,d_base,first,count,input,gradient}

# The tile rows of the band from its first row, and its tile columns.
def first_row(+ready: BandReady) -> Nat:
  Nat.div(ready_first(ready),column_count(ready_geometry(ready)))

def last_row(+ready: BandReady) -> Nat:
  Nat.div(Nat.sub(Nat.add(ready_first(ready),ready_count(ready)),1n),column_count(ready_geometry(ready)))

def band_rows(+ready: BandReady) -> Nat:
  Nat.add(Nat.add(Nat.sub(last_row(ready),first_row(ready)),1n),reach(ready_geometry(ready)))

def band_columns(+ready: BandReady) -> Nat:
  Nat.add(column_count(ready_geometry(ready)),reach(ready_geometry(ready)))

def start_column(+ready: BandReady) -> Nat:
  Nat.mod(ready_first(ready),column_count(ready_geometry(ready)))

# The plan of a band, every field an encoded natural.
def natural_plan(+ready: BandReady) -> Kernel.Plan:
  +geometry = ready_geometry(ready)
  Kernel.Plan{U32.from_nat(output_blocks(geometry)),U32.from_nat(square_size(geometry)),Packing.geometry_kernel(geometry),U32.from_nat(phase_count(geometry)),
    U32.from_nat(band_rows(ready)),U32.from_nat(band_columns(ready)),geometry_columns(geometry),U32.from_nat(start_column(ready)),U32.from_nat(ready_count(ready)),
    geometry_channels(geometry),geometry_outputs(geometry)}

# ---- Encoded band words ----

def capacity_half(+limits: Machine.Limits,+value: Nat,+room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(value,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(value,U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),room,
    Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits)))

def word_quotient(+limits: Machine.Limits,+value: Nat,+divisor: Nat,+value_room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +divisor_room: {Nat.is_le(divisor,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {(U32.from_nat(value) / U32.from_nat(divisor) : U32) == U32.from_nat(Nat.div(value,divisor)) : U32}:
  NativeDivision.encoded_divide(value,divisor,Machine.half_word(limits),Limits.half_word_fits(limits),capacity_half(limits,value,value_room),
    capacity_half(limits,divisor,divisor_room),Limits.double_bound(limits,divisor,divisor_room),positive)

def word_remainder(+limits: Machine.Limits,+value: Nat,+divisor: Nat,+value_room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +divisor_room: {Nat.is_le(divisor,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {(U32.from_nat(value) % U32.from_nat(divisor) : U32) == U32.from_nat(Nat.mod(value,divisor)) : U32}:
  NativeDivision.encoded_modulo(value,divisor,Machine.half_word(limits),Limits.half_word_fits(limits),capacity_half(limits,value,value_room),
    capacity_half(limits,divisor,divisor_room),Limits.double_bound(limits,divisor,divisor_room),positive)

def phases_word(+stride: U32,+kernel: U32) -> {U32.min(stride,kernel) == U32.from_nat(Nat.min(U32.to_nat(stride),U32.to_nat(kernel))) : U32}:
  Encoded.encoding(U32.min(stride,kernel),Nat.min(U32.to_nat(stride),U32.to_nat(kernel)),NativeExtrema.minimum(stride,kernel))

def reach_word(+limits: Machine.Limits,+stride: U32,+kernel: U32,+stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +kernel_positive: {Nat.is_lt(0n,U32.to_nat(kernel)) == True{} : Bool},+kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {((kernel - 1) / U32.min(stride,kernel) : U32) == U32.from_nat(Nat.div(Nat.sub(U32.to_nat(kernel),1n),Nat.min(U32.to_nat(stride),U32.to_nat(kernel)))) : U32}:
  +k = U32.to_nat(kernel)
  +phases = Nat.min(U32.to_nat(stride),k)
  +phases_room = Order.transitive(phases,k,U32.to_nat(Machine.capacity(limits)),Extrema.minimum_right(U32.to_nat(stride),k),kernel_room)
  %Equal.sym(U32,U32.min(stride,kernel),U32.from_nat(phases),phases_word(stride,kernel)) :
    {((kernel - 1) / _ : U32) == U32.from_nat(Nat.div(Nat.sub(k,1n),phases)) : U32}
  %Equal.sym(U32,(kernel - 1 : U32),U32.from_nat(Nat.sub(k,1n)),Arithmetic.native_subtract(kernel,1,Order.positive_one(k,kernel_positive))) :
    {(_ / U32.from_nat(phases) : U32) == U32.from_nat(Nat.div(Nat.sub(k,1n),phases)) : U32}
  word_quotient(limits,Nat.sub(k,1n),phases,Order.transitive(Nat.sub(k,1n),k,U32.to_nat(Machine.capacity(limits)),Order.sub_le(k,1n),kernel_room),phases_room,
    Extrema.minimum_strict_lower(U32.to_nat(stride),k,0n,stride_positive,kernel_positive))

# The tile rows of a band: from the row of its first position to the row of
# its last, and reach more.
def rows_word(+limits: Machine.Limits,+first: Nat,+count: Nat,+columns: U32,+reach: U32,+extra: Nat,+reach_exact: {reach == U32.from_nat(extra) : U32},
  +positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,count) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(first,count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {((U32.from_nat(first) + U32.from_nat(count) - 1) / columns - U32.from_nat(first) / columns + 1 + reach : U32) ==
    U32.from_nat(Nat.add(Nat.add(Nat.sub(Nat.div(Nat.sub(Nat.add(first,count),1n),U32.to_nat(columns)),Nat.div(first,U32.to_nat(columns))),1n),extra)) : U32}:
  +width = U32.to_nat(columns)
  +total = Nat.add(first,count)
  +end = Nat.sub(total,1n)
  +low = Nat.div(first,width)
  +high = Nat.div(end,width)
  +cap = U32.to_nat(Machine.capacity(limits))
  +one_le = Order.transitive(1n,count,total,Order.positive_one(count,nonempty),Order.drop_left_le(first,count,total,Order.reflexive(total)))
  +first_le = {%Equal.sym(Nat,end,Nat.add(first,Nat.sub(count,1n)),Order.subtract_from_sum(first,count,1n,Order.positive_one(count,nonempty))) : {Nat.is_le(first,_) == True{} : Bool}
    Order.le_add(first,Nat.sub(count,1n)) : {Nat.is_le(first,end) == True{} : Bool}}
  +end_room = Order.transitive(end,total,cap,Order.sub_le(total,1n),room)
  +first_room = Order.transitive(first,end,cap,first_le,end_room)
  +wanted = U32.from_nat(Nat.add(Nat.add(Nat.sub(high,low),1n),extra))
  %Arithmetic.native_roundtrip(columns) :
    {((U32.from_nat(first) + U32.from_nat(count) - 1) / _ - U32.from_nat(first) / _ + 1 + reach : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(first) + U32.from_nat(count) : U32),U32.from_nat(total),WordArithmetic.native_add(first,count)) :
    {((_ - 1) / U32.from_nat(width) - U32.from_nat(first) / U32.from_nat(width) + 1 + reach : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(total) - 1 : U32),U32.from_nat(end),Subtraction.encoded_subtract(total,1n,one_le)) :
    {(_ / U32.from_nat(width) - U32.from_nat(first) / U32.from_nat(width) + 1 + reach : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(end) / U32.from_nat(width) : U32),U32.from_nat(high),word_quotient(limits,end,width,end_room,columns_room,positive)) :
    {(_ - U32.from_nat(first) / U32.from_nat(width) + 1 + reach : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(first) / U32.from_nat(width) : U32),U32.from_nat(low),word_quotient(limits,first,width,first_room,columns_room,positive)) :
    {(U32.from_nat(high) - _ + 1 + reach : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(high) - U32.from_nat(low) : U32),U32.from_nat(Nat.sub(high,low)),
      Subtraction.encoded_subtract(high,low,DivisionOrder.quotient_le(first,end,width,positive,first_le))) :
    {(_ + 1 + reach : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.sub(high,low)) + 1 : U32),U32.from_nat(Nat.add(Nat.sub(high,low),1n)),WordArithmetic.native_add(Nat.sub(high,low),1n)) :
    {(_ + reach : U32) == wanted : U32}
  %Equal.sym(U32,reach,U32.from_nat(extra),reach_exact) :
    {(U32.from_nat(Nat.add(Nat.sub(high,low),1n)) + _ : U32) == wanted : U32}
  WordArithmetic.native_add(Nat.add(Nat.sub(high,low),1n),extra)

def sum_word(+columns: U32,+reach: U32,+extra: Nat,+reach_exact: {reach == U32.from_nat(extra) : U32}) ->
  {(columns + reach : U32) == U32.from_nat(Nat.add(U32.to_nat(columns),extra)) : U32}:
  %Arithmetic.native_roundtrip(columns) : {(_ + reach : U32) == U32.from_nat(Nat.add(U32.to_nat(columns),extra)) : U32}
  %Equal.sym(U32,reach,U32.from_nat(extra),reach_exact) : {(U32.from_nat(U32.to_nat(columns)) + _ : U32) == U32.from_nat(Nat.add(U32.to_nat(columns),extra)) : U32}
  WordArithmetic.native_add(U32.to_nat(columns),extra)

def column_word(+limits: Machine.Limits,+first: Nat,+columns: U32,+positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},
  +first_room: {Nat.is_le(first,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {(U32.from_nat(first) % columns : U32) == U32.from_nat(Nat.mod(first,U32.to_nat(columns))) : U32}:
  %Arithmetic.native_roundtrip(columns) : {(U32.from_nat(first) % _ : U32) == U32.from_nat(Nat.mod(first,U32.to_nat(columns))) : U32}
  word_remainder(limits,first,U32.to_nat(columns),first_room,columns_room,positive)

def blocks_word(+limits: Machine.Limits,+value: U32,+room: {Nat.is_le(U32.to_nat(value),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {((value + 7) / 8 : U32) == U32.from_nat(Ceiling.ceiling(U32.to_nat(value),8n)) : U32}:
  Encoded.encoding(((value + 7) / 8 : U32),Ceiling.ceiling(U32.to_nat(value),8n),Encoded.eight_blocks_exact(limits,value,room))

# ---- Band bounds ----

def domain_limits(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) -> Machine.Limits:
  GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  limits

def outputs_room(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) ->
  {Nat.is_le(output_count(geometry),U32.to_nat(Machine.capacity(domain_limits(geometry,domain)))) == True{} : Bool}:
  GeometryReady{+limits,columns,kernel,stride,square,+taps,outputs,+count,aligned,taps_exact,pixels,gradients} = domain
  +t = tap_count(geometry)
  +o = output_count(geometry)
  Order.transitive(o,Nat.mul(t,o),U32.to_nat(Machine.capacity(limits)),
    %Equal.sym(Nat,Nat.mul(t,o),Nat.mul(o,t),RowMajor.multiply_commutative(t,o)) : {Nat.is_le(o,_) == True{} : Bool} RowMajor.product_covers(o,t,taps),count)

def kernel_room(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) ->
  {Nat.is_le(kernel_size(geometry),U32.to_nat(Machine.capacity(domain_limits(geometry,domain)))) == True{} : Bool}:
  GeometryReady{+limits,columns,+kernel,stride,+square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  +k = kernel_size(geometry)
  Order.transitive(k,Nat.mul(k,k),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(k,k,kernel),square)

def output_blocks_positive(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) -> {Nat.is_lt(0n,output_blocks(geometry)) == True{} : Bool}:
  GeometryReady{limits,columns,kernel,stride,square,taps,+outputs,count,aligned,taps_exact,pixels,gradients} = domain
  +o = output_count(geometry)
  RowMajor.left_factor_positive(Ceiling.ceiling(o,8n),8n,Bounds.nat_trans(0n,o,Nat.mul(Ceiling.ceiling(o,8n),8n),outputs,Ceiling.covers(o,8n,{==})))

# The positions of the image fit the capacity.
def area_room(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) ->
  {Nat.is_le(Nat.mul(row_count(geometry),column_count(geometry)),U32.to_nat(Machine.capacity(domain_limits(geometry,domain)))) == True{} : Bool}:
  GeometryReady{+limits,columns,kernel,stride,square,taps,outputs,count,aligned,taps_exact,pixels,+gradients} = domain
  +blocks = output_blocks(geometry)
  +area = Nat.mul(row_count(geometry),column_count(geometry))
  +positive = output_blocks_positive(geometry,domain)
  Order.transitive(area,Nat.mul(blocks,area),U32.to_nat(Machine.capacity(limits)),
    %Equal.sym(Nat,Nat.mul(blocks,area),Nat.mul(area,blocks),RowMajor.multiply_commutative(blocks,area)) : {Nat.is_le(area,_) == True{} : Bool}
    RowMajor.product_covers(area,blocks,positive),
    %RowMajor.assoc(blocks,row_count(geometry),column_count(geometry)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    gradients)

# The positions of a band and its columns fit the capacity.
def total_room(+ready: BandReady) ->
  {Nat.is_le(Nat.add(ready_first(ready),ready_count(ready)),U32.to_nat(Machine.capacity(domain_limits(ready_geometry(ready),ready_domain(ready))))) == True{} : Bool}:
  BandReady{+geometry,+domain,x_base,d_base,+first,+count,input,gradient,nonempty,+room} = ready
  Order.transitive(Nat.add(first,count),Nat.mul(row_count(geometry),column_count(geometry)),U32.to_nat(Machine.capacity(domain_limits(geometry,domain))),room,area_room(geometry,domain))

def rows_positive(+ready: BandReady) -> {Nat.is_lt(0n,row_count(ready_geometry(ready))) == True{} : Bool}:
  BandReady{+geometry,domain,x_base,d_base,+first,+count,input,gradient,+nonempty,+room} = ready
  +area = Nat.mul(row_count(geometry),column_count(geometry))
  RowMajor.left_factor_positive(row_count(geometry),column_count(geometry),
    Bounds.nat_trans(0n,count,area,nonempty,Order.transitive(count,Nat.add(first,count),area,Order.drop_left_le(first,count,Nat.add(first,count),Order.reflexive(Nat.add(first,count))),room)))

def columns_room(+ready: BandReady) -> {Nat.is_le(column_count(ready_geometry(ready)),U32.to_nat(Machine.capacity(domain_limits(ready_geometry(ready),ready_domain(ready))))) == True{} : Bool}:
  BandReady{+geometry,+domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  +positive = rows_positive(ready)
  +r = row_count(geometry)
  +width = column_count(geometry)
  Order.transitive(width,Nat.mul(r,width),U32.to_nat(Machine.capacity(domain_limits(geometry,domain))),
    %Equal.sym(Nat,Nat.mul(r,width),Nat.mul(width,r),RowMajor.multiply_commutative(r,width)) : {Nat.is_le(width,_) == True{} : Bool} RowMajor.product_covers(width,r,positive),
    area_room(geometry,domain))

def first_room(+ready: BandReady) -> {Nat.is_le(ready_first(ready),U32.to_nat(Machine.capacity(domain_limits(ready_geometry(ready),ready_domain(ready))))) == True{} : Bool}:
  BandReady{+geometry,+domain,x_base,d_base,+first,+count,input,gradient,nonempty,room} = ready
  +total = total_room(ready)
  Order.transitive(first,Nat.add(first,count),U32.to_nat(Machine.capacity(domain_limits(geometry,domain))),Order.le_add(first,count),total)

# The runtime plan of a band is its natural plan.
def plan_exact(+ready: BandReady) -> {band_plan(ready_band(ready)) == natural_plan(ready) : Kernel.Plan}:
  BandReady{+geometry,+domain,+x_base,+d_base,+first,+count,+input,+gradient,+nonempty,+room} = ready
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  GeometryReady{+limits,+columns_positive,+kernel_positive,+stride_positive,+square,+taps_positive,+outputs_positive,+count_bound,+aligned,+taps_exact,+pixels,+gradients} = domain
  +total = total_room(ready)
  +width_room = columns_room(ready)
  +start_room = first_room(ready)
  +phases_w = U32.min(s,k)
  +reach_w = ((k - 1) / phases_w : U32)
  +rows_w = ((U32.from_nat(first) + U32.from_nat(count) - 1) / columns - U32.from_nat(first) / columns + 1 + reach_w : U32)
  +extra = Nat.div(Nat.sub(U32.to_nat(k),1n),Nat.min(U32.to_nat(s),U32.to_nat(k)))
  +reach_exact = reach_word(limits,s,k,stride_positive,kernel_positive,kernel_room(geometry,domain))
  +wanted = natural_plan(ready)
  %Equal.sym(U32,rows_w,U32.from_nat(band_rows(ready)),
      rows_word(limits,first,count,columns,reach_w,extra,reach_exact,columns_positive,nonempty,total,width_room)) :
    {Kernel.Plan{((o + 7) / 8 : U32),(k * k : U32),k,phases_w,_,(columns + reach_w : U32),columns,(U32.from_nat(first) % columns : U32),U32.from_nat(count),c,o} == wanted : Kernel.Plan}
  %Equal.sym(U32,(columns + reach_w : U32),U32.from_nat(band_columns(ready)),sum_word(columns,reach_w,extra,reach_exact)) :
    {Kernel.Plan{((o + 7) / 8 : U32),(k * k : U32),k,phases_w,U32.from_nat(band_rows(ready)),_,columns,(U32.from_nat(first) % columns : U32),U32.from_nat(count),c,o} == wanted : Kernel.Plan}
  %Equal.sym(U32,phases_w,U32.from_nat(phase_count(geometry)),phases_word(s,k)) :
    {Kernel.Plan{((o + 7) / 8 : U32),(k * k : U32),k,_,U32.from_nat(band_rows(ready)),U32.from_nat(band_columns(ready)),columns,(U32.from_nat(first) % columns : U32),U32.from_nat(count),c,o} == wanted : Kernel.Plan}
  %Equal.sym(U32,((o + 7) / 8 : U32),U32.from_nat(output_blocks(geometry)),blocks_word(limits,o,outputs_room(geometry,domain))) :
    {Kernel.Plan{_,(k * k : U32),k,U32.from_nat(phase_count(geometry)),U32.from_nat(band_rows(ready)),U32.from_nat(band_columns(ready)),columns,(U32.from_nat(first) % columns : U32),U32.from_nat(count),c,o} == wanted : Kernel.Plan}
  %Equal.sym(U32,(k * k : U32),U32.from_nat(square_size(geometry)),Multiplication.native_multiply(k,k)) :
    {Kernel.Plan{U32.from_nat(output_blocks(geometry)),_,k,U32.from_nat(phase_count(geometry)),U32.from_nat(band_rows(ready)),U32.from_nat(band_columns(ready)),columns,(U32.from_nat(first) % columns : U32),U32.from_nat(count),c,o} == wanted : Kernel.Plan}
  %Equal.sym(U32,(U32.from_nat(first) % columns : U32),U32.from_nat(start_column(ready)),column_word(limits,first,columns,columns_positive,start_room,width_room)) :
    {Kernel.Plan{U32.from_nat(output_blocks(geometry)),U32.from_nat(square_size(geometry)),k,U32.from_nat(phase_count(geometry)),U32.from_nat(band_rows(ready)),U32.from_nat(band_columns(ready)),columns,_,U32.from_nat(count),c,o} == wanted : Kernel.Plan}
  {==}

# ---- Job bounds ----

def ready_limits(+ready: BandReady) -> Machine.Limits:
  domain_limits(ready_geometry(ready),ready_domain(ready))

def ready_capacity(+ready: BandReady) -> Nat:
  U32.to_nat(Machine.capacity(ready_limits(ready)))

def square_positive(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) -> {Nat.is_lt(0n,square_size(geometry)) == True{} : Bool}:
  GeometryReady{limits,columns,+kernel,stride,square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  Order.product_positive(kernel_size(geometry),kernel_size(geometry),kernel,kernel)

def channels_room(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) ->
  {Nat.is_le(channel_count(geometry),U32.to_nat(Machine.capacity(domain_limits(geometry,domain)))) == True{} : Bool}:
  GeometryReady{+limits,columns,kernel,stride,square,taps,+outputs,+count,aligned,+taps_exact,pixels,gradients} = domain
  +c = channel_count(geometry)
  +t = tap_count(geometry)
  +o = output_count(geometry)
  +cap = U32.to_nat(Machine.capacity(limits))
  Order.transitive(c,t,cap,
    %Equal.sym(Nat,t,Nat.mul(c,square_size(geometry)),taps_exact) : {Nat.is_le(c,_) == True{} : Bool} RowMajor.product_covers(c,square_size(geometry),square_positive(geometry,domain)),
    Order.transitive(t,Nat.mul(t,o),cap,RowMajor.product_covers(t,o,outputs),count))

def job_extent(+geometry: Kernel.Geometry) -> Nat:
  Nat.mul(Nat.mul(Nat.mul(channel_blocks(geometry),kernel_size(geometry)),kernel_size(geometry)),output_blocks(geometry))

def jobs_room(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) ->
  {Nat.is_le(job_extent(geometry),U32.to_nat(Machine.capacity(domain_limits(geometry,domain)))) == True{} : Bool}:
  GeometryReady{+limits,columns,kernel,stride,square,taps,outputs,+count,aligned,+taps_exact,pixels,gradients} = domain
  +c = channel_count(geometry)
  +o = output_count(geometry)
  +k = kernel_size(geometry)
  +blocks = channel_blocks(geometry)
  +outs = output_blocks(geometry)
  +cap = U32.to_nat(Machine.capacity(limits))
  +first = RowMajor.mul_le(blocks,c,k,Ceiling.bounded(c,8n,{==}))
  +second = RowMajor.mul_le(Nat.mul(blocks,k),Nat.mul(c,k),k,first)
  +third = RowMajor.mul_le(Nat.mul(Nat.mul(blocks,k),k),Nat.mul(Nat.mul(c,k),k),outs,second)
  +fourth = DivisionOrder.multiply_left_le(Nat.mul(Nat.mul(c,k),k),outs,o,Ceiling.bounded(o,8n,{==}))
  +fifth = {%Equal.sym(Nat,Nat.mul(Nat.mul(c,k),k),Nat.mul(c,Nat.mul(k,k)),RowMajor.assoc(c,k,k)) : {Nat.is_le(Nat.mul(_,o),cap) == True{} : Bool}
    %taps_exact : {Nat.is_le(Nat.mul(_,o),cap) == True{} : Bool} count : {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(c,k),k),o),cap) == True{} : Bool}}
  Order.transitive(job_extent(geometry),Nat.mul(Nat.mul(Nat.mul(c,k),k),outs),cap,third,Order.transitive(Nat.mul(Nat.mul(Nat.mul(c,k),k),outs),Nat.mul(Nat.mul(Nat.mul(c,k),k),o),cap,fourth,fifth))

def output_blocks_room(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) ->
  {Nat.is_le(output_blocks(geometry),U32.to_nat(Machine.capacity(domain_limits(geometry,domain)))) == True{} : Bool}:
  Order.transitive(output_blocks(geometry),output_count(geometry),U32.to_nat(Machine.capacity(domain_limits(geometry,domain))),Ceiling.bounded(output_count(geometry),8n,{==}),
    outputs_room(geometry,domain))

def square_room(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) ->
  {Nat.is_le(square_size(geometry),U32.to_nat(Machine.capacity(domain_limits(geometry,domain)))) == True{} : Bool}:
  GeometryReady{limits,columns,kernel,stride,+square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  square

def phases_positive(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) -> {Nat.is_lt(0n,phase_count(geometry)) == True{} : Bool}:
  GeometryReady{limits,columns,+kernel,+stride,square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  Extrema.minimum_strict_lower(stride_size(geometry),kernel_size(geometry),0n,stride,kernel)

def phases_room(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) ->
  {Nat.is_le(phase_count(geometry),U32.to_nat(Machine.capacity(domain_limits(geometry,domain)))) == True{} : Bool}:
  Order.transitive(phase_count(geometry),kernel_size(geometry),U32.to_nat(Machine.capacity(domain_limits(geometry,domain))),
    Extrema.minimum_right(stride_size(geometry),kernel_size(geometry)),kernel_room(geometry,domain))

def kernel_positive(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) -> {Nat.is_lt(0n,kernel_size(geometry)) == True{} : Bool}:
  GeometryReady{limits,columns,+kernel,stride,square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  kernel

# ---- The jobs of a band ----

def job_count(+ready: BandReady) -> Nat:
  job_extent(ready_geometry(ready))

# Job index is output block index % blocks, tap index / blocks % square and
# channel block index / blocks / square; its tap is row a and column b.
def job_output_block(+ready: BandReady,+index: Nat) -> Nat:
  Nat.mod(index,output_blocks(ready_geometry(ready)))

def job_quotient(+ready: BandReady,+index: Nat) -> Nat:
  Nat.div(index,output_blocks(ready_geometry(ready)))

def job_tap(+ready: BandReady,+index: Nat) -> Nat:
  Nat.mod(job_quotient(ready,index),square_size(ready_geometry(ready)))

def job_block(+ready: BandReady,+index: Nat) -> Nat:
  Nat.div(job_quotient(ready,index),square_size(ready_geometry(ready)))

def job_tap_row(+ready: BandReady,+index: Nat) -> Nat:
  Nat.div(job_tap(ready,index),kernel_size(ready_geometry(ready)))

def job_tap_column(+ready: BandReady,+index: Nat) -> Nat:
  Nat.mod(job_tap(ready,index),kernel_size(ready_geometry(ready)))

# The pixel plane of a job's channel block and tap phase.
def job_plane(+ready: BandReady,+index: Nat) -> Nat:
  +phases = phase_count(ready_geometry(ready))
  Nat.add(Nat.mul(Nat.add(Nat.mul(job_block(ready,index),phases),Nat.mod(job_tap_row(ready,index),phases)),phases),Nat.mod(job_tap_column(ready,index),phases))

# The pixel tile row of a job's first band row.
def job_row_base(+ready: BandReady,+index: Nat) -> Nat:
  Nat.add(Nat.mul(job_plane(ready,index),band_rows(ready)),Nat.div(job_tap_row(ready,index),phase_count(ready_geometry(ready))))

def job_address(+ready: BandReady,+index: Nat) -> Nat:
  Nat.add(Nat.add(Nat.mul(job_row_base(ready,index),band_columns(ready)),start_column(ready)),Nat.div(job_tap_column(ready,index),phase_count(ready_geometry(ready))))

# Row r of a job is stored r rows of square_size after the cell of its first
# channel at its tap in its output block.
def job_store_base(+ready: BandReady,+index: Nat) -> Nat:
  +geometry = ready_geometry(ready)
  Nat.add(Nat.mul(Nat.add(Nat.mul(job_output_block(ready,index),channel_count(geometry)),Nat.mul(job_block(ready,index),8n)),square_size(geometry)),job_tap(ready,index))

def job_words(+row: U32,+depth: Nat,+direct: U32,+address: U32,+column: U32,+columns: U32,+band_rows: U32,+band_columns: U32,+base: U32,+row_stride: U32,
  +rows: U32,+lanes: U32) -> Gemm.Job<Kernel.Cursor()>:
  Gemm.Job{row,depth,Gemm.Both{direct,Gemm.Entries{address,column,0,columns,band_rows,(band_columns - columns + 1 : U32),0}},Gemm.Store{base,row_stride,1,rows,lanes}}

def natural_job(+ready: BandReady,+index: Nat) -> Gemm.Job<Kernel.Cursor()>:
  +geometry = ready_geometry(ready)
  job_words(U32.from_nat(Nat.mul(job_block(ready,index),8n)),ready_count(ready),U32.from_nat(Nat.mul(job_output_block(ready,index),ready_count(ready))),
    U32.from_nat(job_address(ready,index)),U32.from_nat(start_column(ready)),geometry_columns(geometry),U32.from_nat(band_rows(ready)),U32.from_nat(band_columns(ready)),
    U32.from_nat(job_store_base(ready,index)),U32.from_nat(square_size(geometry)),
    U32.min(8,U32.from_nat(Nat.sub(channel_count(geometry),Nat.mul(job_block(ready,index),8n)))),
    U32.min(8,U32.from_nat(Nat.sub(output_count(geometry),Nat.mul(job_output_block(ready,index),8n)))))

# The parts of a job index lie inside their ranges.
type JobInside<-ready: BandReady,-index: Nat> is Data:
  JobInside{index_room: {Nat.is_le(index,ready_capacity(ready)) == True{} : Bool},
    output_block: {Nat.is_lt(job_output_block(ready,index),output_blocks(ready_geometry(ready))) == True{} : Bool},
    quotient_room: {Nat.is_le(job_quotient(ready,index),ready_capacity(ready)) == True{} : Bool},
    tap: {Nat.is_lt(job_tap(ready,index),square_size(ready_geometry(ready))) == True{} : Bool},
    block: {Nat.is_lt(job_block(ready,index),channel_blocks(ready_geometry(ready))) == True{} : Bool},
    tap_row: {Nat.is_lt(job_tap_row(ready,index),kernel_size(ready_geometry(ready))) == True{} : Bool},
    tap_column: {Nat.is_lt(job_tap_column(ready,index),kernel_size(ready_geometry(ready))) == True{} : Bool},
    channels: {Nat.is_lt(Nat.mul(job_block(ready,index),8n),channel_count(ready_geometry(ready))) == True{} : Bool},
    outputs: {Nat.is_lt(Nat.mul(job_output_block(ready,index),8n),output_count(ready_geometry(ready))) == True{} : Bool}}

def job_inside(+ready: BandReady,+index: Nat,+inside: {Nat.is_lt(index,job_count(ready)) == True{} : Bool}) -> JobInside<ready,index>:
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  +cap = ready_capacity(ready)
  +k = kernel_size(geometry)
  +blocks = channel_blocks(geometry)
  +outs = output_blocks(geometry)
  +square = square_size(geometry)
  +outs_positive = output_blocks_positive(geometry,domain)
  +square_ok = square_positive(geometry,domain)
  +k_positive = kernel_positive(geometry,domain)
  +quotient = job_quotient(ready,index)
  +tap = job_tap(ready,index)
  +room = Order.transitive(index,job_count(ready),cap,Bounds.strict_le(index,job_count(ready),inside),jobs_room(geometry,domain))
  +below = {%RowMajor.assoc(blocks,k,k) : {Nat.is_lt(index,Nat.mul(_,outs)) == True{} : Bool} inside : {Nat.is_lt(index,Nat.mul(Nat.mul(blocks,square),outs)) == True{} : Bool}}
  +quotient_below = DivisionOrder.quotient_below(index,Nat.mul(blocks,square),outs,outs_positive,below)
  +quotient_le = Order.transitive(quotient,Nat.mul(quotient,outs),index,RowMajor.product_covers(quotient,outs,outs_positive),DivisionOrder.product_below(index,outs,outs_positive))
  +block_below = DivisionOrder.quotient_below(quotient,blocks,square,square_ok,quotient_below)
  +tap_below = DivisionOrder.remainder_small(quotient,square,square_ok)
  +output_below = DivisionOrder.remainder_small(index,outs,outs_positive)
  JobInside{room,output_below,Order.transitive(quotient,index,cap,quotient_le,room),tap_below,block_below,
    DivisionOrder.quotient_below(tap,k,k,k_positive,tap_below),DivisionOrder.remainder_small(tap,k,k_positive),
    %Ceiling.upper_comparison(channel_count(geometry),8n,job_block(ready,index),{==}) : {_ == True{} : Bool} block_below,
    %Ceiling.upper_comparison(output_count(geometry),8n,job_output_block(ready,index),{==}) : {_ == True{} : Bool} output_below}

# ---- The words of a job ----

def scaled_word(+value: Nat) -> {(U32.from_nat(value) * 8 : U32) == U32.from_nat(Nat.mul(value,8n)) : U32}:
  Multiplication.native_encoded_multiply(value,8n)

def unscaled_word(+limits: Machine.Limits,+value: Nat,+room: {Nat.is_le(Nat.mul(value,8n),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {(U32.from_nat(Nat.mul(value,8n)) / 8 : U32) == U32.from_nat(value) : U32}:
  +sixteen = Bounds.as_nat_le(16,Machine.half_word(limits),Limits.sixteen_within_half(limits))
  +scaled = Nat.mul(value,8n)
  +exact = Equal.trans(Nat,Nat.div(scaled,8n),Nat.div(Nat.add(scaled,0n),8n),value,
    Equal.cong(Nat,Nat,number => Nat.div(number,8n),scaled,Nat.add(scaled,0n),Equal.sym(Nat,Nat.add(scaled,0n),scaled,Addition.zero_right(scaled))),
    Radix.quotient(value,0n,8n,{==},{==}))
  Equal.trans(U32,(U32.from_nat(scaled) / 8 : U32),U32.from_nat(Nat.div(scaled,8n)),U32.from_nat(value),
    NativeDivision.encoded_divide(scaled,8n,Machine.half_word(limits),Limits.half_word_fits(limits),capacity_half(limits,scaled,room),
      Order.transitive(8n,16n,U32.to_nat(Machine.half_word(limits)),{==},sixteen),sixteen,{==}),
    Equal.cong(Nat,U32,U32.from_nat,Nat.div(scaled,8n),value,exact))

def output_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {(U32.from_nat(index) % U32.from_nat(output_blocks(ready_geometry(ready))) * 8 : U32) == U32.from_nat(Nat.mul(job_output_block(ready,index),8n)) : U32}:
  JobInside{+index_room,output_block,quotient_room,tap,block,tap_row,tap_column,channels,outputs} = inside
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  %Equal.sym(U32,(U32.from_nat(index) % U32.from_nat(output_blocks(geometry)) : U32),U32.from_nat(job_output_block(ready,index)),
      word_remainder(ready_limits(ready),index,output_blocks(geometry),index_room,output_blocks_room(geometry,domain),output_blocks_positive(geometry,domain))) :
    {(_ * 8 : U32) == U32.from_nat(Nat.mul(job_output_block(ready,index),8n)) : U32}
  scaled_word(job_output_block(ready,index))

def quotient_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {(U32.from_nat(index) / U32.from_nat(output_blocks(ready_geometry(ready))) : U32) == U32.from_nat(job_quotient(ready,index)) : U32}:
  JobInside{+index_room,output_block,quotient_room,tap,block,tap_row,tap_column,channels,outputs} = inside
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  word_quotient(ready_limits(ready),index,output_blocks(geometry),index_room,output_blocks_room(geometry,domain),output_blocks_positive(geometry,domain))

def tap_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {(U32.from_nat(index) / U32.from_nat(output_blocks(ready_geometry(ready))) % U32.from_nat(square_size(ready_geometry(ready))) : U32) ==
    U32.from_nat(job_tap(ready,index)) : U32}:
  JobInside{index_room,output_block,+quotient_room,tap,block,tap_row,tap_column,channels,outputs} = inside
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  %Equal.sym(U32,(U32.from_nat(index) / U32.from_nat(output_blocks(geometry)) : U32),U32.from_nat(job_quotient(ready,index)),quotient_word(ready,index,inside)) :
    {(_ % U32.from_nat(square_size(geometry)) : U32) == U32.from_nat(job_tap(ready,index)) : U32}
  word_remainder(ready_limits(ready),job_quotient(ready,index),square_size(geometry),quotient_room,square_room(geometry,domain),square_positive(geometry,domain))

def row_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {(U32.from_nat(index) / U32.from_nat(output_blocks(ready_geometry(ready))) / U32.from_nat(square_size(ready_geometry(ready))) * 8 : U32) ==
    U32.from_nat(Nat.mul(job_block(ready,index),8n)) : U32}:
  JobInside{index_room,output_block,+quotient_room,tap,block,tap_row,tap_column,channels,outputs} = inside
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  %Equal.sym(U32,(U32.from_nat(index) / U32.from_nat(output_blocks(geometry)) : U32),U32.from_nat(job_quotient(ready,index)),quotient_word(ready,index,inside)) :
    {(_ / U32.from_nat(square_size(geometry)) * 8 : U32) == U32.from_nat(Nat.mul(job_block(ready,index),8n)) : U32}
  %Equal.sym(U32,(U32.from_nat(job_quotient(ready,index)) / U32.from_nat(square_size(geometry)) : U32),U32.from_nat(job_block(ready,index)),
      word_quotient(ready_limits(ready),job_quotient(ready,index),square_size(geometry),quotient_room,square_room(geometry,domain),square_positive(geometry,domain))) :
    {(_ * 8 : U32) == U32.from_nat(Nat.mul(job_block(ready,index),8n)) : U32}
  scaled_word(job_block(ready,index))

def tap_room(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) -> {Nat.is_le(job_tap(ready,index),ready_capacity(ready)) == True{} : Bool}:
  JobInside{index_room,output_block,quotient_room,+tap,block,tap_row,tap_column,channels,outputs} = inside
  +geometry = ready_geometry(ready)
  Order.transitive(job_tap(ready,index),square_size(geometry),ready_capacity(ready),Bounds.strict_le(job_tap(ready,index),square_size(geometry),tap),square_room(geometry,ready_domain(ready)))

def tap_row_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {(U32.from_nat(job_tap(ready,index)) / Packing.geometry_kernel(ready_geometry(ready)) : U32) == U32.from_nat(job_tap_row(ready,index)) : U32}:
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  %Arithmetic.native_roundtrip(Packing.geometry_kernel(geometry)) : {(U32.from_nat(job_tap(ready,index)) / _ : U32) == U32.from_nat(job_tap_row(ready,index)) : U32}
  word_quotient(ready_limits(ready),job_tap(ready,index),kernel_size(geometry),tap_room(ready,index,inside),kernel_room(geometry,domain),kernel_positive(geometry,domain))

def tap_column_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {(U32.from_nat(job_tap(ready,index)) % Packing.geometry_kernel(ready_geometry(ready)) : U32) == U32.from_nat(job_tap_column(ready,index)) : U32}:
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  %Arithmetic.native_roundtrip(Packing.geometry_kernel(geometry)) : {(U32.from_nat(job_tap(ready,index)) % _ : U32) == U32.from_nat(job_tap_column(ready,index)) : U32}
  word_remainder(ready_limits(ready),job_tap(ready,index),kernel_size(geometry),tap_room(ready,index,inside),kernel_room(geometry,domain),kernel_positive(geometry,domain))

# A tap offset below the kernel splits into its phase and its tile row.
def offset_room(+ready: BandReady,+offset: Nat,+below: {Nat.is_lt(offset,kernel_size(ready_geometry(ready))) == True{} : Bool}) ->
  {Nat.is_le(offset,ready_capacity(ready)) == True{} : Bool}:
  +geometry = ready_geometry(ready)
  Order.transitive(offset,kernel_size(geometry),ready_capacity(ready),Bounds.strict_le(offset,kernel_size(geometry),below),kernel_room(geometry,ready_domain(ready)))

def phase_word(+ready: BandReady,+offset: Nat,+below: {Nat.is_lt(offset,kernel_size(ready_geometry(ready))) == True{} : Bool}) ->
  {(U32.from_nat(offset) % U32.from_nat(phase_count(ready_geometry(ready))) : U32) == U32.from_nat(Nat.mod(offset,phase_count(ready_geometry(ready)))) : U32}:
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  word_remainder(ready_limits(ready),offset,phase_count(geometry),offset_room(ready,offset,below),phases_room(geometry,domain),phases_positive(geometry,domain))

def shift_word(+ready: BandReady,+offset: Nat,+below: {Nat.is_lt(offset,kernel_size(ready_geometry(ready))) == True{} : Bool}) ->
  {(U32.from_nat(offset) / U32.from_nat(phase_count(ready_geometry(ready))) : U32) == U32.from_nat(Nat.div(offset,phase_count(ready_geometry(ready)))) : U32}:
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  word_quotient(ready_limits(ready),offset,phase_count(geometry),offset_room(ready,offset,below),phases_room(geometry,domain),phases_positive(geometry,domain))

def channel_rows_room(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {Nat.is_le(Nat.mul(job_block(ready,index),8n),ready_capacity(ready)) == True{} : Bool}:
  JobInside{index_room,output_block,quotient_room,tap,block,tap_row,tap_column,+channels,outputs} = inside
  +geometry = ready_geometry(ready)
  Order.transitive(Nat.mul(job_block(ready,index),8n),channel_count(geometry),ready_capacity(ready),Bounds.strict_le(Nat.mul(job_block(ready,index),8n),channel_count(geometry),channels),
    channels_room(geometry,ready_domain(ready)))

def output_lanes_room(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {Nat.is_le(Nat.mul(job_output_block(ready,index),8n),ready_capacity(ready)) == True{} : Bool}:
  JobInside{index_room,output_block,quotient_room,tap,block,tap_row,tap_column,channels,+outputs} = inside
  +geometry = ready_geometry(ready)
  Order.transitive(Nat.mul(job_output_block(ready,index),8n),output_count(geometry),ready_capacity(ready),Bounds.strict_le(Nat.mul(job_output_block(ready,index),8n),output_count(geometry),outputs),
    outputs_room(geometry,ready_domain(ready)))

# ---- The fields of a job ----

def direct_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {(U32.from_nat(index) % U32.from_nat(output_blocks(ready_geometry(ready))) * 8 / 8 * U32.from_nat(ready_count(ready)) : U32) ==
    U32.from_nat(Nat.mul(job_output_block(ready,index),ready_count(ready))) : U32}:
  +block = job_output_block(ready,index)
  +wanted = U32.from_nat(Nat.mul(block,ready_count(ready)))
  %Equal.sym(U32,(U32.from_nat(index) % U32.from_nat(output_blocks(ready_geometry(ready))) * 8 : U32),U32.from_nat(Nat.mul(block,8n)),output_word(ready,index,inside)) :
    {(_ / 8 * U32.from_nat(ready_count(ready)) : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.mul(block,8n)) / 8 : U32),U32.from_nat(block),unscaled_word(ready_limits(ready),block,output_lanes_room(ready,index,inside))) :
    {(_ * U32.from_nat(ready_count(ready)) : U32) == wanted : U32}
  Multiplication.native_encoded_multiply(block,ready_count(ready))

def address_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {((((U32.from_nat(index) / U32.from_nat(output_blocks(ready_geometry(ready))) / U32.from_nat(square_size(ready_geometry(ready))) * 8 / 8 * U32.from_nat(phase_count(ready_geometry(ready))) +
      U32.from_nat(index) / U32.from_nat(output_blocks(ready_geometry(ready))) % U32.from_nat(square_size(ready_geometry(ready))) / Packing.geometry_kernel(ready_geometry(ready)) % U32.from_nat(phase_count(ready_geometry(ready)))) *
      U32.from_nat(phase_count(ready_geometry(ready))) +
      U32.from_nat(index) / U32.from_nat(output_blocks(ready_geometry(ready))) % U32.from_nat(square_size(ready_geometry(ready))) % Packing.geometry_kernel(ready_geometry(ready)) % U32.from_nat(phase_count(ready_geometry(ready)))) *
      U32.from_nat(band_rows(ready)) +
      U32.from_nat(index) / U32.from_nat(output_blocks(ready_geometry(ready))) % U32.from_nat(square_size(ready_geometry(ready))) / Packing.geometry_kernel(ready_geometry(ready)) / U32.from_nat(phase_count(ready_geometry(ready)))) *
      U32.from_nat(band_columns(ready)) + U32.from_nat(start_column(ready)) +
      U32.from_nat(index) / U32.from_nat(output_blocks(ready_geometry(ready))) % U32.from_nat(square_size(ready_geometry(ready))) % Packing.geometry_kernel(ready_geometry(ready)) / U32.from_nat(phase_count(ready_geometry(ready))) : U32) ==
    U32.from_nat(job_address(ready,index)) : U32}:
  JobInside{index_room,output_block,quotient_room,tap,block,+tap_row,+tap_column,channels,outputs} = inside
  +geometry = ready_geometry(ready)
  +kernel = Packing.geometry_kernel(geometry)
  +phases = phase_count(geometry)
  +ph = U32.from_nat(phases)
  +br = U32.from_nat(band_rows(ready))
  +bc = U32.from_nat(band_columns(ready))
  +col = U32.from_nat(start_column(ready))
  +rb = job_block(ready,index)
  +kt = job_tap(ready,index)
  +a = job_tap_row(ready,index)
  +b = job_tap_column(ready,index)
  +tap_w = (U32.from_nat(index) / U32.from_nat(output_blocks(geometry)) % U32.from_nat(square_size(geometry)) : U32)
  +wanted = U32.from_nat(job_address(ready,index))
  %Equal.sym(U32,(U32.from_nat(index) / U32.from_nat(output_blocks(geometry)) / U32.from_nat(square_size(geometry)) * 8 : U32),U32.from_nat(Nat.mul(rb,8n)),row_word(ready,index,inside)) :
    {((((_ / 8 * ph + tap_w / kernel % ph) * ph + tap_w % kernel % ph) * br + tap_w / kernel / ph) * bc + col + tap_w % kernel / ph : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.mul(rb,8n)) / 8 : U32),U32.from_nat(rb),unscaled_word(ready_limits(ready),rb,channel_rows_room(ready,index,inside))) :
    {((((_ * ph + tap_w / kernel % ph) * ph + tap_w % kernel % ph) * br + tap_w / kernel / ph) * bc + col + tap_w % kernel / ph : U32) == wanted : U32}
  %Equal.sym(U32,tap_w,U32.from_nat(kt),tap_word(ready,index,inside)) :
    {((((U32.from_nat(rb) * ph + _ / kernel % ph) * ph + _ % kernel % ph) * br + _ / kernel / ph) * bc + col + _ % kernel / ph : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(kt) / kernel : U32),U32.from_nat(a),tap_row_word(ready,index,inside)) :
    {((((U32.from_nat(rb) * ph + _ % ph) * ph + U32.from_nat(kt) % kernel % ph) * br + _ / ph) * bc + col + U32.from_nat(kt) % kernel / ph : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(kt) % kernel : U32),U32.from_nat(b),tap_column_word(ready,index,inside)) :
    {((((U32.from_nat(rb) * ph + U32.from_nat(a) % ph) * ph + _ % ph) * br + U32.from_nat(a) / ph) * bc + col + _ / ph : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(a) % ph : U32),U32.from_nat(Nat.mod(a,phases)),phase_word(ready,a,tap_row)) :
    {((((U32.from_nat(rb) * ph + _) * ph + U32.from_nat(b) % ph) * br + U32.from_nat(a) / ph) * bc + col + U32.from_nat(b) / ph : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(b) % ph : U32),U32.from_nat(Nat.mod(b,phases)),phase_word(ready,b,tap_column)) :
    {((((U32.from_nat(rb) * ph + U32.from_nat(Nat.mod(a,phases))) * ph + _) * br + U32.from_nat(a) / ph) * bc + col + U32.from_nat(b) / ph : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(a) / ph : U32),U32.from_nat(Nat.div(a,phases)),shift_word(ready,a,tap_row)) :
    {((((U32.from_nat(rb) * ph + U32.from_nat(Nat.mod(a,phases))) * ph + U32.from_nat(Nat.mod(b,phases))) * br + _) * bc + col + U32.from_nat(b) / ph : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(b) / ph : U32),U32.from_nat(Nat.div(b,phases)),shift_word(ready,b,tap_column)) :
    {((((U32.from_nat(rb) * ph + U32.from_nat(Nat.mod(a,phases))) * ph + U32.from_nat(Nat.mod(b,phases))) * br + U32.from_nat(Nat.div(a,phases))) * bc + col + _ : U32) == wanted : U32}
  +first = Nat.add(Nat.mul(rb,phases),Nat.mod(a,phases))
  +plane = Nat.add(Nat.mul(first,phases),Nat.mod(b,phases))
  +row = Nat.add(Nat.mul(plane,band_rows(ready)),Nat.div(a,phases))
  %Equal.sym(U32,(U32.from_nat(rb) * ph + U32.from_nat(Nat.mod(a,phases)) : U32),U32.from_nat(first),Encoded.encoded_madd(rb,phases,Nat.mod(a,phases))) :
    {(((_ * ph + U32.from_nat(Nat.mod(b,phases))) * br + U32.from_nat(Nat.div(a,phases))) * bc + col + U32.from_nat(Nat.div(b,phases)) : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(first) * ph + U32.from_nat(Nat.mod(b,phases)) : U32),U32.from_nat(plane),Encoded.encoded_madd(first,phases,Nat.mod(b,phases))) :
    {((_ * br + U32.from_nat(Nat.div(a,phases))) * bc + col + U32.from_nat(Nat.div(b,phases)) : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(plane) * br + U32.from_nat(Nat.div(a,phases)) : U32),U32.from_nat(row),Encoded.encoded_madd(plane,band_rows(ready),Nat.div(a,phases))) :
    {(_ * bc + col + U32.from_nat(Nat.div(b,phases)) : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(row) * bc + col : U32),U32.from_nat(Nat.add(Nat.mul(row,band_columns(ready)),start_column(ready))),Encoded.encoded_madd(row,band_columns(ready),start_column(ready))) :
    {(_ + U32.from_nat(Nat.div(b,phases)) : U32) == wanted : U32}
  WordArithmetic.native_add(Nat.add(Nat.mul(row,band_columns(ready)),start_column(ready)),Nat.div(b,phases))

def base_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {((U32.from_nat(index) % U32.from_nat(output_blocks(ready_geometry(ready))) * 8 / 8 * geometry_channels(ready_geometry(ready)) +
      U32.from_nat(index) / U32.from_nat(output_blocks(ready_geometry(ready))) / U32.from_nat(square_size(ready_geometry(ready))) * 8) * U32.from_nat(square_size(ready_geometry(ready))) +
      U32.from_nat(index) / U32.from_nat(output_blocks(ready_geometry(ready))) % U32.from_nat(square_size(ready_geometry(ready))) : U32) == U32.from_nat(job_store_base(ready,index)) : U32}:
  +geometry = ready_geometry(ready)
  +square = U32.from_nat(square_size(geometry))
  +channels = geometry_channels(geometry)
  +ob = job_output_block(ready,index)
  +rb8 = Nat.mul(job_block(ready,index),8n)
  +kt = job_tap(ready,index)
  +wanted = U32.from_nat(job_store_base(ready,index))
  +ow = (U32.from_nat(index) % U32.from_nat(output_blocks(geometry)) * 8 : U32)
  +rw = (U32.from_nat(index) / U32.from_nat(output_blocks(geometry)) / square * 8 : U32)
  +tw = (U32.from_nat(index) / U32.from_nat(output_blocks(geometry)) % square : U32)
  %Equal.sym(U32,ow,U32.from_nat(Nat.mul(ob,8n)),output_word(ready,index,inside)) :
    {((_ / 8 * channels + rw) * square + tw : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.mul(ob,8n)) / 8 : U32),U32.from_nat(ob),unscaled_word(ready_limits(ready),ob,output_lanes_room(ready,index,inside))) :
    {((_ * channels + rw) * square + tw : U32) == wanted : U32}
  %Equal.sym(U32,rw,U32.from_nat(rb8),row_word(ready,index,inside)) :
    {((U32.from_nat(ob) * channels + _) * square + tw : U32) == wanted : U32}
  %Equal.sym(U32,tw,U32.from_nat(kt),tap_word(ready,index,inside)) :
    {((U32.from_nat(ob) * channels + U32.from_nat(rb8)) * square + _ : U32) == wanted : U32}
  %Arithmetic.native_roundtrip(channels) : {((U32.from_nat(ob) * _ + U32.from_nat(rb8)) * square + U32.from_nat(kt) : U32) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(ob) * U32.from_nat(channel_count(geometry)) + U32.from_nat(rb8) : U32),U32.from_nat(Nat.add(Nat.mul(ob,channel_count(geometry)),rb8)),
      Encoded.encoded_madd(ob,channel_count(geometry),rb8)) :
    {(_ * square + U32.from_nat(kt) : U32) == wanted : U32}
  Encoded.encoded_madd(Nat.add(Nat.mul(ob,channel_count(geometry)),rb8),square_size(geometry),kt)

def rows_left_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {U32.min(8,(geometry_channels(ready_geometry(ready)) - U32.from_nat(index) / U32.from_nat(output_blocks(ready_geometry(ready))) / U32.from_nat(square_size(ready_geometry(ready))) * 8 : U32)) ==
    U32.min(8,U32.from_nat(Nat.sub(channel_count(ready_geometry(ready)),Nat.mul(job_block(ready,index),8n)))) : U32}:
  JobInside{index_room,output_block,quotient_room,tap,block,tap_row,tap_column,+channels,outputs} = inside
  +geometry = ready_geometry(ready)
  +rb8 = Nat.mul(job_block(ready,index),8n)
  +wanted = U32.min(8,U32.from_nat(Nat.sub(channel_count(geometry),rb8)))
  %Equal.sym(U32,(U32.from_nat(index) / U32.from_nat(output_blocks(geometry)) / U32.from_nat(square_size(geometry)) * 8 : U32),U32.from_nat(rb8),row_word(ready,index,inside)) :
    {U32.min(8,(geometry_channels(geometry) - _ : U32)) == wanted : U32}
  %Arithmetic.native_roundtrip(geometry_channels(geometry)) : {U32.min(8,(_ - U32.from_nat(rb8) : U32)) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(channel_count(geometry)) - U32.from_nat(rb8) : U32),U32.from_nat(Nat.sub(channel_count(geometry),rb8)),
      Subtraction.encoded_subtract(channel_count(geometry),rb8,Bounds.strict_le(rb8,channel_count(geometry),channels))) :
    {U32.min(8,_) == wanted : U32}
  {==}

def lanes_left_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {U32.min(8,(geometry_outputs(ready_geometry(ready)) - U32.from_nat(index) % U32.from_nat(output_blocks(ready_geometry(ready))) * 8 : U32)) ==
    U32.min(8,U32.from_nat(Nat.sub(output_count(ready_geometry(ready)),Nat.mul(job_output_block(ready,index),8n)))) : U32}:
  JobInside{index_room,output_block,quotient_room,tap,block,tap_row,tap_column,channels,+outputs} = inside
  +geometry = ready_geometry(ready)
  +ob8 = Nat.mul(job_output_block(ready,index),8n)
  +wanted = U32.min(8,U32.from_nat(Nat.sub(output_count(geometry),ob8)))
  %Equal.sym(U32,(U32.from_nat(index) % U32.from_nat(output_blocks(geometry)) * 8 : U32),U32.from_nat(ob8),output_word(ready,index,inside)) :
    {U32.min(8,(geometry_outputs(geometry) - _ : U32)) == wanted : U32}
  %Arithmetic.native_roundtrip(geometry_outputs(geometry)) : {U32.min(8,(_ - U32.from_nat(ob8) : U32)) == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(output_count(geometry)) - U32.from_nat(ob8) : U32),U32.from_nat(Nat.sub(output_count(geometry),ob8)),
      Subtraction.encoded_subtract(output_count(geometry),ob8,Bounds.strict_le(ob8,output_count(geometry),outputs))) :
    {U32.min(8,_) == wanted : U32}
  {==}

def count_room(+ready: BandReady) -> {Nat.is_le(ready_count(ready),ready_capacity(ready)) == True{} : Bool}:
  Order.drop_left_le(ready_first(ready),ready_count(ready),ready_capacity(ready),total_room(ready))

# Every job of the natural plan is its natural job.
def job_exact(+ready: BandReady,+index: Nat,+within: {Nat.is_lt(index,job_count(ready)) == True{} : Bool}) ->
  {Kernel.job(natural_plan(ready),U32.from_nat(index)) == natural_job(ready,index) : Gemm.Job<Kernel.Cursor()>}:
  +inside = job_inside(ready,index,within)
  +geometry = ready_geometry(ready)
  +n = U32.from_nat(index)
  +blocks = U32.from_nat(output_blocks(geometry))
  +square = U32.from_nat(square_size(geometry))
  +kernel = Packing.geometry_kernel(geometry)
  +ph = U32.from_nat(phase_count(geometry))
  +br = U32.from_nat(band_rows(ready))
  +bc = U32.from_nat(band_columns(ready))
  +col = U32.from_nat(start_column(ready))
  +count = U32.from_nat(ready_count(ready))
  +columns = geometry_columns(geometry)
  +output_w = (n % blocks * 8 : U32)
  +tap_w = (n / blocks % square : U32)
  +row_w = (n / blocks / square * 8 : U32)
  +address_w = ((((row_w / 8 * ph + tap_w / kernel % ph) * ph + tap_w % kernel % ph) * br + tap_w / kernel / ph) * bc + col + tap_w % kernel / ph : U32)
  +base_w = ((output_w / 8 * geometry_channels(geometry) + row_w) * square + tap_w : U32)
  +rows_w = U32.min(8,(geometry_channels(geometry) - row_w : U32))
  +lanes_w = U32.min(8,(geometry_outputs(geometry) - output_w : U32))
  +stride_w = square
  +direct_w = (output_w / 8 * count : U32)
  +wanted = natural_job(ready,index)
  +row_n = U32.from_nat(Nat.mul(job_block(ready,index),8n))
  +direct_n = U32.from_nat(Nat.mul(job_output_block(ready,index),ready_count(ready)))
  +address_n = U32.from_nat(job_address(ready,index))
  +base_n = U32.from_nat(job_store_base(ready,index))
  +stride_n = square
  %Equal.sym(U32,row_w,row_n,row_word(ready,index,inside)) :
    {job_words(_,U32.to_nat(count),direct_w,address_w,col,columns,br,bc,base_w,stride_w,rows_w,lanes_w) == wanted : Gemm.Job<Kernel.Cursor()>}
  %Equal.sym(Nat,U32.to_nat(count),ready_count(ready),FullRange.roundtrip(ready_count(ready),Machine.capacity(ready_limits(ready)),count_room(ready))) :
    {job_words(row_n,_,direct_w,address_w,col,columns,br,bc,base_w,stride_w,rows_w,lanes_w) == wanted : Gemm.Job<Kernel.Cursor()>}
  %Equal.sym(U32,direct_w,direct_n,direct_word(ready,index,inside)) :
    {job_words(row_n,ready_count(ready),_,address_w,col,columns,br,bc,base_w,stride_w,rows_w,lanes_w) == wanted : Gemm.Job<Kernel.Cursor()>}
  %Equal.sym(U32,address_w,address_n,address_word(ready,index,inside)) :
    {job_words(row_n,ready_count(ready),direct_n,_,col,columns,br,bc,base_w,stride_w,rows_w,lanes_w) == wanted : Gemm.Job<Kernel.Cursor()>}
  %Equal.sym(U32,base_w,base_n,base_word(ready,index,inside)) :
    {job_words(row_n,ready_count(ready),direct_n,address_n,col,columns,br,bc,_,stride_w,rows_w,lanes_w) == wanted : Gemm.Job<Kernel.Cursor()>}
  %Equal.sym(U32,rows_w,U32.min(8,U32.from_nat(Nat.sub(channel_count(geometry),Nat.mul(job_block(ready,index),8n)))),rows_left_word(ready,index,inside)) :
    {job_words(row_n,ready_count(ready),direct_n,address_n,col,columns,br,bc,base_n,stride_n,_,lanes_w) == wanted : Gemm.Job<Kernel.Cursor()>}
  %Equal.sym(U32,lanes_w,U32.min(8,U32.from_nat(Nat.sub(output_count(geometry),Nat.mul(job_output_block(ready,index),8n)))),lanes_left_word(ready,index,inside)) :
    {job_words(row_n,ready_count(ready),direct_n,address_n,col,columns,br,bc,base_n,stride_n,U32.min(8,U32.from_nat(Nat.sub(channel_count(geometry),Nat.mul(job_block(ready,index),8n)))),_) ==
      wanted : Gemm.Job<Kernel.Cursor()>}
  {==}

# ---- The rows of a band ----

def ready_x_base(+ready: BandReady) -> U32:
  BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  x_base

def ready_d_base(+ready: BandReady) -> U32:
  BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  d_base

def ready_input(+ready: BandReady) -> Buffer.Model<Tile.Tile8>:
  BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  input

def ready_gradient(+ready: BandReady) -> Buffer.Model<Tile.Tile8>:
  BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  gradient

def band_end(+ready: BandReady) -> Nat:
  Nat.sub(Nat.add(ready_first(ready),ready_count(ready)),1n)

def ready_room(+ready: BandReady) -> {Nat.is_le(Nat.add(ready_first(ready),ready_count(ready)),Nat.mul(row_count(ready_geometry(ready)),column_count(ready_geometry(ready)))) == True{} : Bool}:
  BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,nonempty,+room} = ready
  room

def ready_nonempty(+ready: BandReady) -> {Nat.is_lt(0n,ready_count(ready)) == True{} : Bool}:
  BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,+nonempty,room} = ready
  nonempty

def columns_positive(+ready: BandReady) -> {Nat.is_lt(0n,column_count(ready_geometry(ready))) == True{} : Bool}:
  BandReady{+geometry,+domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  GeometryReady{limits,+columns,kernel,stride,square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  columns

def stride_positive(+ready: BandReady) -> {Nat.is_lt(0n,stride_size(ready_geometry(ready))) == True{} : Bool}:
  BandReady{+geometry,+domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  GeometryReady{limits,columns,kernel,+stride,square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  stride

# The last position of a band lies in the image, in a row from the first.
def end_inside(+ready: BandReady) -> {Nat.is_lt(band_end(ready),Nat.mul(row_count(ready_geometry(ready)),column_count(ready_geometry(ready)))) == True{} : Bool}:
  +total = Nat.add(ready_first(ready),ready_count(ready))
  +positive = Bounds.nat_trans(0n,ready_count(ready),total,ready_nonempty(ready),Order.drop_left_le(ready_first(ready),ready_count(ready),total,Order.reflexive(total)))
  Bounds.nat_trans(band_end(ready),total,Nat.mul(row_count(ready_geometry(ready)),column_count(ready_geometry(ready))),Order.last_below(total,positive),ready_room(ready))

def first_before_end(+ready: BandReady) -> {Nat.is_le(ready_first(ready),band_end(ready)) == True{} : Bool}:
  +first = ready_first(ready)
  +count = ready_count(ready)
  %Equal.sym(Nat,band_end(ready),Nat.add(first,Nat.sub(count,1n)),Order.subtract_from_sum(first,count,1n,Order.positive_one(count,ready_nonempty(ready)))) :
    {Nat.is_le(first,_) == True{} : Bool}
  Order.le_add(first,Nat.sub(count,1n))

def last_inside(+ready: BandReady) -> {Nat.is_lt(last_row(ready),row_count(ready_geometry(ready))) == True{} : Bool}:
  DivisionOrder.quotient_below(band_end(ready),row_count(ready_geometry(ready)),column_count(ready_geometry(ready)),columns_positive(ready),end_inside(ready))

def first_before_last(+ready: BandReady) -> {Nat.is_le(first_row(ready),last_row(ready)) == True{} : Bool}:
  DivisionOrder.quotient_le(ready_first(ready),band_end(ready),column_count(ready_geometry(ready)),columns_positive(ready),first_before_end(ready))

def successor_word(+value: Nat) -> {Nat.add(value,1n) == 1n+value : Nat}:
  Order.commutative(value,1n)

# The band rows end below the image rows and reach.
def rows_within(+ready: BandReady) -> {Nat.is_le(band_rows(ready),Nat.add(row_count(ready_geometry(ready)),reach(ready_geometry(ready)))) == True{} : Bool}:
  +last = last_row(ready)
  +span = Nat.sub(last,first_row(ready))
  +below = Order.le_lt_trans(span,last,row_count(ready_geometry(ready)),Order.sub_le(last,first_row(ready)),last_inside(ready))
  Order.add_le(Nat.add(span,1n),row_count(ready_geometry(ready)),reach(ready_geometry(ready)),
    %Equal.sym(Nat,Nat.add(span,1n),1n+span,successor_word(span)) : {Nat.is_le(_,row_count(ready_geometry(ready))) == True{} : Bool} Order.strict_follow(span,row_count(ready_geometry(ready)),below))

def rows_end(+ready: BandReady) -> {Nat.add(first_row(ready),band_rows(ready)) == Nat.add(Nat.add(last_row(ready),1n),reach(ready_geometry(ready))) : Nat}:
  +first = first_row(ready)
  +last = last_row(ready)
  +span = Nat.sub(last,first)
  +extra = reach(ready_geometry(ready))
  Equal.trans(Nat,Nat.add(first,Nat.add(Nat.add(span,1n),extra)),Nat.add(Nat.add(first,Nat.add(span,1n)),extra),Nat.add(Nat.add(last,1n),extra),
    Equal.sym(Nat,Nat.add(Nat.add(first,Nat.add(span,1n)),extra),Nat.add(first,Nat.add(Nat.add(span,1n),extra)),Addition.associative(first,Nat.add(span,1n),extra)),
    Equal.cong(Nat,Nat,head => Nat.add(head,extra),Nat.add(first,Nat.add(span,1n)),Nat.add(last,1n),
      Equal.trans(Nat,Nat.add(first,Nat.add(span,1n)),Nat.add(Nat.add(first,span),1n),Nat.add(last,1n),
        Equal.sym(Nat,Nat.add(Nat.add(first,span),1n),Nat.add(first,Nat.add(span,1n)),Addition.associative(first,span,1n)),
        Equal.cong(Nat,Nat,head => Nat.add(head,1n),Nat.add(first,span),last,Order.restore_shift(first,last,first_before_last(ready))))))

def rows_end_within(+ready: BandReady) -> {Nat.is_le(Nat.add(first_row(ready),band_rows(ready)),Nat.add(row_count(ready_geometry(ready)),reach(ready_geometry(ready)))) == True{} : Bool}:
  +last = last_row(ready)
  +rows = row_count(ready_geometry(ready))
  %Equal.sym(Nat,Nat.add(first_row(ready),band_rows(ready)),Nat.add(Nat.add(last,1n),reach(ready_geometry(ready))),rows_end(ready)) :
    {Nat.is_le(_,Nat.add(rows,reach(ready_geometry(ready)))) == True{} : Bool}
  Order.add_le(Nat.add(last,1n),rows,reach(ready_geometry(ready)),%Equal.sym(Nat,Nat.add(last,1n),1n+last,successor_word(last)) : {Nat.is_le(_,rows) == True{} : Bool} Order.strict_follow(last,rows,last_inside(ready)))

# ---- Packed band sizes ----

def plane_count(+geometry: Kernel.Geometry) -> Nat:
  Nat.mul(Nat.mul(channel_blocks(geometry),phase_count(geometry)),phase_count(geometry))

def band_area(+ready: BandReady) -> Nat:
  Nat.mul(band_rows(ready),band_columns(ready))

def pixel_count(+ready: BandReady) -> Nat:
  Nat.mul(Nat.mul(plane_count(ready_geometry(ready)),band_rows(ready)),band_columns(ready))

def gradient_count(+ready: BandReady) -> Nat:
  Nat.mul(output_blocks(ready_geometry(ready)),ready_count(ready))

def channels_positive(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) -> {Nat.is_lt(0n,channel_count(geometry)) == True{} : Bool}:
  GeometryReady{limits,columns,kernel,stride,square,+taps,outputs,count,aligned,+taps_exact,pixels,gradients} = domain
  RowMajor.left_factor_positive(channel_count(geometry),square_size(geometry),
    {%taps_exact : {Nat.is_lt(0n,_) == True{} : Bool} taps : {Nat.is_lt(0n,Nat.mul(channel_count(geometry),square_size(geometry))) == True{} : Bool}})

def channel_blocks_positive(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) -> {Nat.is_lt(0n,channel_blocks(geometry)) == True{} : Bool}:
  +c = channel_count(geometry)
  RowMajor.left_factor_positive(Ceiling.ceiling(c,8n),8n,Bounds.nat_trans(0n,c,Nat.mul(Ceiling.ceiling(c,8n),8n),channels_positive(geometry,domain),Ceiling.covers(c,8n,{==})))

def rows_reach_positive(+ready: BandReady) -> {Nat.is_lt(0n,Nat.add(row_count(ready_geometry(ready)),reach(ready_geometry(ready)))) == True{} : Bool}:
  Bounds.nat_trans(0n,row_count(ready_geometry(ready)),Nat.add(row_count(ready_geometry(ready)),reach(ready_geometry(ready))),rows_positive(ready),
    Order.le_add(row_count(ready_geometry(ready)),reach(ready_geometry(ready))))

def band_columns_positive(+ready: BandReady) -> {Nat.is_lt(0n,band_columns(ready)) == True{} : Bool}:
  Bounds.nat_trans(0n,column_count(ready_geometry(ready)),band_columns(ready),columns_positive(ready),Order.le_add(column_count(ready_geometry(ready)),reach(ready_geometry(ready))))

def pixel_extent_room(+ready: BandReady) -> {Nat.is_le(pixel_extent(ready_geometry(ready)),ready_capacity(ready)) == True{} : Bool}:
  BandReady{+geometry,+domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned,taps_exact,+pixels,gradients} = domain
  pixels

def gradient_extent_room(+ready: BandReady) -> {Nat.is_le(gradient_extent(ready_geometry(ready)),ready_capacity(ready)) == True{} : Bool}:
  BandReady{+geometry,+domain,x_base,d_base,first,count,input,gradient,nonempty,room} = ready
  GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned,taps_exact,pixels,+gradients} = domain
  gradients

# The tile rows and columns of every plane fit the capacity.
def plane_area_room(+ready: BandReady) ->
  {Nat.is_le(Nat.mul(Nat.add(row_count(ready_geometry(ready)),reach(ready_geometry(ready))),band_columns(ready)),ready_capacity(ready)) == True{} : Bool}:
  +geometry = ready_geometry(ready)
  +planes = plane_count(geometry)
  +area = Nat.mul(Nat.add(row_count(geometry),reach(geometry)),band_columns(ready))
  +positive = Order.product_positive(Nat.mul(channel_blocks(geometry),phase_count(geometry)),phase_count(geometry),
    Order.product_positive(channel_blocks(geometry),phase_count(geometry),channel_blocks_positive(geometry,ready_domain(ready)),phases_positive(geometry,ready_domain(ready))),
    phases_positive(geometry,ready_domain(ready)))
  Order.transitive(area,Nat.mul(planes,area),ready_capacity(ready),
    %RowMajor.multiply_commutative(area,planes) : {Nat.is_le(area,_) == True{} : Bool} RowMajor.product_covers(area,planes,positive),
    %RowMajor.assoc(planes,Nat.add(row_count(geometry),reach(geometry)),band_columns(ready)) : {Nat.is_le(_,ready_capacity(ready)) == True{} : Bool}
    pixel_extent_room(ready))

def planes_room(+ready: BandReady) -> {Nat.is_le(plane_count(ready_geometry(ready)),ready_capacity(ready)) == True{} : Bool}:
  +geometry = ready_geometry(ready)
  +planes = plane_count(geometry)
  +tall = Nat.add(row_count(geometry),reach(geometry))
  Order.transitive(planes,Nat.mul(planes,tall),ready_capacity(ready),RowMajor.product_covers(planes,tall,rows_reach_positive(ready)),
    Order.transitive(Nat.mul(planes,tall),Nat.mul(Nat.mul(planes,tall),band_columns(ready)),ready_capacity(ready),
      RowMajor.product_covers(Nat.mul(planes,tall),band_columns(ready),band_columns_positive(ready)),pixel_extent_room(ready)))

def pixel_count_room(+ready: BandReady) -> {Nat.is_le(pixel_count(ready),ready_capacity(ready)) == True{} : Bool}:
  +geometry = ready_geometry(ready)
  +planes = plane_count(geometry)
  +tall = Nat.add(row_count(geometry),reach(geometry))
  Order.transitive(pixel_count(ready),Nat.mul(Nat.mul(planes,tall),band_columns(ready)),ready_capacity(ready),
    RowMajor.mul_le(Nat.mul(planes,band_rows(ready)),Nat.mul(planes,tall),band_columns(ready),DivisionOrder.multiply_left_le(planes,band_rows(ready),tall,rows_within(ready))),
    pixel_extent_room(ready))

def gradient_count_room(+ready: BandReady) -> {Nat.is_le(gradient_count(ready),ready_capacity(ready)) == True{} : Bool}:
  +geometry = ready_geometry(ready)
  +area = Nat.mul(row_count(geometry),column_count(geometry))
  +count = ready_count(ready)
  Order.transitive(gradient_count(ready),Nat.mul(output_blocks(geometry),area),ready_capacity(ready),
    DivisionOrder.multiply_left_le(output_blocks(geometry),count,area,Order.drop_left_le(ready_first(ready),count,area,ready_room(ready))),
    %RowMajor.assoc(output_blocks(geometry),row_count(geometry),column_count(geometry)) : {Nat.is_le(_,ready_capacity(ready)) == True{} : Bool}
    gradient_extent_room(ready))

# ---- The packed sources in natural words ----

def natural_pixels(+ready: BandReady) -> Kernel.Pixels:
  Kernel.Pixels{ready_geometry(ready),ready_x_base(ready),U32.from_nat(phase_count(ready_geometry(ready))),U32.from_nat(first_row(ready)),U32.from_nat(band_rows(ready)),
    U32.from_nat(band_columns(ready))}

def pixel_fresh(+ready: BandReady) -> Tree.Tree<Tile.Tile8>:
  Tree.fresh(Tile.Tile8,Storage.depth_for(U32.from_nat(pixel_count(ready))),Tile.filled(0.0))

# The gradient cell of the first position of a band: the groups of a
# position side by side from d_base.
def gradient_base(+ready: BandReady) -> Nat:
  Nat.add(U32.to_nat(ready_d_base(ready)),Nat.mul(ready_first(ready),output_blocks(ready_geometry(ready))))

def natural_positions(+ready: BandReady) -> Kernel.Positions:
  Kernel.Positions{U32.from_nat(gradient_base(ready)),U32.from_nat(output_blocks(ready_geometry(ready))),U32.from_nat(ready_count(ready))}

def gradient_fresh(+ready: BandReady) -> Tree.Tree<Tile.Tile8>:
  Tree.fresh(Tile.Tile8,Storage.depth_for(U32.from_nat(gradient_count(ready))),Tile.filled(0.0))

def first_row_word(+limits: Machine.Limits,+first: Nat,+columns: U32,+positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},
  +first_room: {Nat.is_le(first,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {(U32.from_nat(first) / columns : U32) == U32.from_nat(Nat.div(first,U32.to_nat(columns))) : U32}:
  %Arithmetic.native_roundtrip(columns) : {(U32.from_nat(first) / _ : U32) == U32.from_nat(Nat.div(first,U32.to_nat(columns))) : U32}
  word_quotient(limits,first,U32.to_nat(columns),first_room,columns_room,positive)

# The pixel planes of a band, every word an encoded natural.
def pixels_exact(+ready: BandReady) ->
  {band_pixels(ready_band(ready)) ==
    Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,plane_count(ready_geometry(ready)),0,natural_pixels(ready),
      Packing.Packed{ready_input(ready),pixel_fresh(ready)}) : Packing.Packed}:
  BandReady{+geometry,+domain,+x_base,+d_base,+first,+count,+input,+gradient,+nonempty,+room} = ready
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  GeometryReady{+limits,+columns_positive,+kernel_positive,+stride_positive,+square,+taps_positive,+outputs_positive,+count_bound,+aligned,+taps_exact,+pixels,+gradients} = domain
  +total = total_room(ready)
  +width_room = columns_room(ready)
  +start_room = first_room(ready)
  +planes_inside = planes_room(ready)
  +start = U32.from_nat(first)
  +first_row_w = (start / columns : U32)
  +phases_w = U32.min(s,k)
  +reach_w = ((k - 1) / phases_w : U32)
  +rows_w = ((start + U32.from_nat(count) - 1) / columns - first_row_w + 1 + reach_w : U32)
  +columns_w = (columns + reach_w : U32)
  +blocks_w = ((c + 7) / 8 : U32)
  +extra = Nat.div(Nat.sub(U32.to_nat(k),1n),Nat.min(U32.to_nat(s),U32.to_nat(k)))
  +reach_exact = reach_word(limits,s,k,stride_positive,kernel_positive,kernel_room(geometry,domain))
  +ph = U32.from_nat(phase_count(geometry))
  +rows_n = U32.from_nat(band_rows(ready))
  +columns_n = U32.from_nat(band_columns(ready))
  +blocks = channel_blocks(geometry)
  +planes = plane_count(geometry)
  +wanted = Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,planes,0,natural_pixels(ready),Packing.Packed{input,pixel_fresh(ready)})
  %Equal.sym(U32,rows_w,rows_n,rows_word(limits,first,count,columns,reach_w,extra,reach_exact,columns_positive,nonempty,total,width_room)) :
    {Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,U32.to_nat((blocks_w * phases_w * phases_w : U32)),0,Kernel.Pixels{geometry,x_base,phases_w,first_row_w,_,columns_w},
      Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for(((blocks_w * phases_w * phases_w : U32) * _ * columns_w : U32)),Tile.filled(0.0))}) == wanted : Packing.Packed}
  %Equal.sym(U32,columns_w,columns_n,sum_word(columns,reach_w,extra,reach_exact)) :
    {Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,U32.to_nat((blocks_w * phases_w * phases_w : U32)),0,Kernel.Pixels{geometry,x_base,phases_w,first_row_w,rows_n,_},
      Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for(((blocks_w * phases_w * phases_w : U32) * rows_n * _ : U32)),Tile.filled(0.0))}) == wanted : Packing.Packed}
  %Equal.sym(U32,first_row_w,U32.from_nat(first_row(ready)),first_row_word(limits,first,columns,columns_positive,start_room,width_room)) :
    {Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,U32.to_nat((blocks_w * phases_w * phases_w : U32)),0,Kernel.Pixels{geometry,x_base,phases_w,_,rows_n,columns_n},
      Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for(((blocks_w * phases_w * phases_w : U32) * rows_n * columns_n : U32)),Tile.filled(0.0))}) == wanted : Packing.Packed}
  %Equal.sym(U32,blocks_w,U32.from_nat(blocks),blocks_word(limits,c,channels_room(geometry,domain))) :
    {Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,U32.to_nat((_ * phases_w * phases_w : U32)),0,
      Kernel.Pixels{geometry,x_base,phases_w,U32.from_nat(first_row(ready)),rows_n,columns_n},
      Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for(((_ * phases_w * phases_w : U32) * rows_n * columns_n : U32)),Tile.filled(0.0))}) == wanted : Packing.Packed}
  %Equal.sym(U32,phases_w,ph,phases_word(s,k)) :
    {Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,U32.to_nat((U32.from_nat(blocks) * _ * _ : U32)),0,
      Kernel.Pixels{geometry,x_base,_,U32.from_nat(first_row(ready)),rows_n,columns_n},
      Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for(((U32.from_nat(blocks) * _ * _ : U32) * rows_n * columns_n : U32)),Tile.filled(0.0))}) == wanted : Packing.Packed}
  %Equal.sym(U32,(U32.from_nat(blocks) * ph : U32),U32.from_nat(Nat.mul(blocks,phase_count(geometry))),Multiplication.native_encoded_multiply(blocks,phase_count(geometry))) :
    {Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,U32.to_nat((_ * ph : U32)),0,natural_pixels(ready),
      Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for(((_ * ph : U32) * rows_n * columns_n : U32)),Tile.filled(0.0))}) == wanted : Packing.Packed}
  %Equal.sym(U32,(U32.from_nat(Nat.mul(blocks,phase_count(geometry))) * ph : U32),U32.from_nat(planes),
      Multiplication.native_encoded_multiply(Nat.mul(blocks,phase_count(geometry)),phase_count(geometry))) :
    {Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,U32.to_nat(_),0,natural_pixels(ready),
      Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for((_ * rows_n * columns_n : U32)),Tile.filled(0.0))}) == wanted : Packing.Packed}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(planes)),planes,FullRange.roundtrip(planes,Machine.capacity(limits),planes_inside)) :
    {Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,_,0,natural_pixels(ready),
      Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for((U32.from_nat(planes) * rows_n * columns_n : U32)),Tile.filled(0.0))}) == wanted : Packing.Packed}
  %Equal.sym(U32,(U32.from_nat(planes) * rows_n : U32),U32.from_nat(Nat.mul(planes,band_rows(ready))),Multiplication.native_encoded_multiply(planes,band_rows(ready))) :
    {Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,planes,0,natural_pixels(ready),
      Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for((_ * columns_n : U32)),Tile.filled(0.0))}) == wanted : Packing.Packed}
  %Equal.sym(U32,(U32.from_nat(Nat.mul(planes,band_rows(ready))) * columns_n : U32),U32.from_nat(pixel_count(ready)),
      Multiplication.native_encoded_multiply(Nat.mul(planes,band_rows(ready)),band_columns(ready))) :
    {Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,planes,0,natural_pixels(ready),
      Packing.Packed{input,Tree.fresh(Tile.Tile8,Storage.depth_for(_),Tile.filled(0.0))}) == wanted : Packing.Packed}
  {==}

# The gradient blocks of a band, every word an encoded natural.
def gradients_exact(+ready: BandReady) ->
  {band_gradients(ready_band(ready)) == Packing.gradients(natural_positions(ready),gradient_count(ready),ready_gradient(ready),gradient_fresh(ready)) : Packing.Packed}:
  BandReady{+geometry,+domain,+x_base,+d_base,+first,+count,+input,+gradient,+nonempty,+room} = ready
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +limits = domain_limits(geometry,domain)
  +blocks_w = ((o + 7) / 8 : U32)
  +blocks = output_blocks(geometry)
  +positions = U32.from_nat(count)
  +total = gradient_count(ready)
  +start = U32.from_nat(first)
  +d_n = U32.to_nat(d_base)
  +blocks_n = U32.from_nat(blocks)
  +lead = Nat.mul(first,blocks)
  +wanted = Packing.gradients(natural_positions(ready),total,gradient,gradient_fresh(ready))
  %Equal.sym(U32,blocks_w,blocks_n,blocks_word(limits,o,outputs_room(geometry,domain))) :
    {Packing.gradients(Kernel.Positions{(d_base + start * _ : U32),_,positions},U32.to_nat((_ * positions : U32)),gradient,
      Tree.fresh(Tile.Tile8,Storage.depth_for((_ * positions : U32)),Tile.filled(0.0))) == wanted : Packing.Packed}
  %Equal.sym(U32,(blocks_n * positions : U32),U32.from_nat(total),Multiplication.native_encoded_multiply(blocks,count)) :
    {Packing.gradients(Kernel.Positions{(d_base + start * blocks_n : U32),blocks_n,positions},U32.to_nat(_),gradient,
      Tree.fresh(Tile.Tile8,Storage.depth_for(_),Tile.filled(0.0))) == wanted : Packing.Packed}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(total)),total,FullRange.roundtrip(total,Machine.capacity(limits),gradient_count_room(ready))) :
    {Packing.gradients(Kernel.Positions{(d_base + start * blocks_n : U32),blocks_n,positions},_,gradient,gradient_fresh(ready)) == wanted : Packing.Packed}
  %Arithmetic.native_roundtrip(d_base) :
    {Packing.gradients(Kernel.Positions{(_ + start * blocks_n : U32),blocks_n,positions},total,gradient,gradient_fresh(ready)) == wanted : Packing.Packed}
  %Equal.sym(U32,(start * blocks_n : U32),U32.from_nat(lead),Multiplication.native_encoded_multiply(first,blocks)) :
    {Packing.gradients(Kernel.Positions{(U32.from_nat(d_n) + _ : U32),blocks_n,positions},total,gradient,gradient_fresh(ready)) == wanted : Packing.Packed}
  %Equal.sym(U32,(U32.from_nat(d_n) + U32.from_nat(lead) : U32),U32.from_nat(Nat.add(d_n,lead)),WordArithmetic.native_add(d_n,lead)) :
    {Packing.gradients(Kernel.Positions{_,blocks_n,positions},total,gradient,gradient_fresh(ready)) == wanted : Packing.Packed}
  {==}

# ---- The packed trees of a band ----

def band_columns_room(+ready: BandReady) -> {Nat.is_le(band_columns(ready),ready_capacity(ready)) == True{} : Bool}:
  +tall = Nat.add(row_count(ready_geometry(ready)),reach(ready_geometry(ready)))
  +columns = band_columns(ready)
  Order.transitive(columns,Nat.mul(tall,columns),ready_capacity(ready),
    %RowMajor.multiply_commutative(columns,tall) : {Nat.is_le(columns,_) == True{} : Bool} RowMajor.product_covers(columns,tall,rows_reach_positive(ready)),
    plane_area_room(ready))

def band_rows_positive(+ready: BandReady) -> {Nat.is_lt(0n,band_rows(ready)) == True{} : Bool}:
  +span = Nat.sub(last_row(ready),first_row(ready))
  Bounds.nat_trans(0n,Nat.add(span,1n),band_rows(ready),
    {%Equal.sym(Nat,Nat.add(span,1n),1n+span,successor_word(span)) : {Nat.is_lt(0n,_) == True{} : Bool} {==} : {Nat.is_lt(0n,Nat.add(span,1n)) == True{} : Bool}},
    Order.le_add(Nat.add(span,1n),reach(ready_geometry(ready))))

def pixel_planes(+ready: BandReady) -> Packing.Planes:
  Packing.Planes{ready_input(ready),natural_pixels(ready),band_area(ready),plane_count(ready_geometry(ready))}

def band_area_capacity(+ready: BandReady) -> {Nat.is_le(band_area(ready),ready_capacity(ready)) == True{} : Bool}:
  +tall = Nat.add(row_count(ready_geometry(ready)),reach(ready_geometry(ready)))
  Order.transitive(band_area(ready),Nat.mul(tall,band_columns(ready)),ready_capacity(ready),RowMajor.mul_le(band_rows(ready),tall,band_columns(ready),rows_within(ready)),
    plane_area_room(ready))

# The planes of a band fit a span of the half word.
def pixels_ready(+ready: BandReady) -> Packing.planes_ready(pixel_planes(ready)):
  Packing.PlanesReady{Multiplication.native_encoded_multiply(band_rows(ready),band_columns(ready)),
    FullRange.roundtrip(band_area(ready),Machine.capacity(ready_limits(ready)),band_area_capacity(ready))}

def pixel_tree(+ready: BandReady) -> Tree.Tree<Tile.Tile8>:
  Tree.write_sequence(~Tile.Tile8,List.concat(&2,Tile.Tile8,ListRead.enumerate(~List<&2,Tile.Tile8>,~Packing.Planes,~Packing.planes_row,plane_count(ready_geometry(ready)),0n,pixel_planes(ready))),
    U32.from_nat(0n),pixel_fresh(ready))

def gradient_tree(+ready: BandReady) -> Tree.Tree<Tile.Tile8>:
  Packing.packed_output(Packing.gradients(natural_positions(ready),gradient_count(ready),ready_gradient(ready),gradient_fresh(ready)))

def pixel_tiles_exact(+ready: BandReady) -> {Packing.packed_output(band_pixels(ready_band(ready))) == pixel_tree(ready) : Tree.Tree<Tile.Tile8>}:
  +folded = Traversal.indexed_fold(~Packing.Packed,~Kernel.Pixels,~Packing.pixel_plane,plane_count(ready_geometry(ready)),0,natural_pixels(ready),
    Packing.Packed{ready_input(ready),pixel_fresh(ready)})
  Equal.trans(Tree.Tree<Tile.Tile8>,Packing.packed_output(band_pixels(ready_band(ready))),Packing.packed_output(folded),pixel_tree(ready),
    Equal.cong(Packing.Packed,Tree.Tree<Tile.Tile8>,state => Packing.packed_output(state),band_pixels(ready_band(ready)),folded,pixels_exact(ready)),
    Equal.cong(Packing.Packed,Tree.Tree<Tile.Tile8>,state => Packing.packed_output(state),folded,Packing.Packed{ready_input(ready),pixel_tree(ready)},
      Packing.planes_points(pixel_planes(ready),pixels_ready(ready),pixel_fresh(ready))))

def gradient_tiles_exact(+ready: BandReady) -> {Packing.packed_output(band_gradients(ready_band(ready))) == gradient_tree(ready) : Tree.Tree<Tile.Tile8>}:
  Equal.cong(Packing.Packed,Tree.Tree<Tile.Tile8>,state => Packing.packed_output(state),band_gradients(ready_band(ready)),
    Packing.gradients(natural_positions(ready),gradient_count(ready),ready_gradient(ready),gradient_fresh(ready)),gradients_exact(ready))

def natural_readers(+ready: BandReady) -> GemmJobs.Readers:
  GemmJobs.Readers{pixel_tree(ready),gradient_tree(ready),no_bias()}

# The jobs of a band are its channel blocks by taps by output blocks.
def jobs_exact(+ready: BandReady) -> {U32.to_nat(band_jobs(ready_band(ready))) == job_count(ready) : Nat}:
  BandReady{+geometry,+domain,+x_base,+d_base,+first,+count,+input,+gradient,+nonempty,+room} = ready
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +limits = domain_limits(geometry,domain)
  +blocks = channel_blocks(geometry)
  +outs = output_blocks(geometry)
  +kernel = U32.to_nat(k)
  +wanted = job_count(ready)
  %Equal.sym(U32,((c + 7) / 8 : U32),U32.from_nat(blocks),blocks_word(limits,c,channels_room(geometry,domain))) :
    {U32.to_nat((_ * k * k * ((o + 7) / 8 : U32) : U32)) == wanted : Nat}
  %Equal.sym(U32,((o + 7) / 8 : U32),U32.from_nat(outs),blocks_word(limits,o,outputs_room(geometry,domain))) :
    {U32.to_nat((U32.from_nat(blocks) * k * k * _ : U32)) == wanted : Nat}
  %Arithmetic.native_roundtrip(k) : {U32.to_nat((U32.from_nat(blocks) * _ * _ * U32.from_nat(outs) : U32)) == wanted : Nat}
  %Equal.sym(U32,(U32.from_nat(blocks) * U32.from_nat(kernel) : U32),U32.from_nat(Nat.mul(blocks,kernel)),Multiplication.native_encoded_multiply(blocks,kernel)) :
    {U32.to_nat((_ * U32.from_nat(kernel) * U32.from_nat(outs) : U32)) == wanted : Nat}
  %Equal.sym(U32,(U32.from_nat(Nat.mul(blocks,kernel)) * U32.from_nat(kernel) : U32),U32.from_nat(Nat.mul(Nat.mul(blocks,kernel),kernel)),
      Multiplication.native_encoded_multiply(Nat.mul(blocks,kernel),kernel)) :
    {U32.to_nat((_ * U32.from_nat(outs) : U32)) == wanted : Nat}
  %Equal.sym(U32,(U32.from_nat(Nat.mul(Nat.mul(blocks,kernel),kernel)) * U32.from_nat(outs) : U32),U32.from_nat(wanted),
      Multiplication.native_encoded_multiply(Nat.mul(Nat.mul(blocks,kernel),kernel),outs)) :
    {U32.to_nat(_) == wanted : Nat}
  FullRange.roundtrip(wanted,Machine.capacity(limits),jobs_room(geometry,domain))

def natural_writes(+ready: BandReady) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  GemmJobs.all_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.advance,~GemmLaws.LaneFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.kept,job_count(ready),0,natural_plan(ready),natural_readers(ready))

def band_writes(+band: Band,+jobs: Nat,+plan: Kernel.Plan,+pixels: Tree.Tree<Tile.Tile8>,+gradients: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Writes.pure(Tile.Tile8,GemmJobs.all_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.advance,~GemmLaws.LaneFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.kept,jobs,0,plan,
    GemmJobs.Readers{pixels,gradients,no_bias()}),band_values(band))

# The output of a band is the writes of its natural jobs over its packed trees.
def band_output(+ready: BandReady) -> {group_output(band_group(ready_band(ready))) == Writes.pure(Tile.Tile8,natural_writes(ready),band_values(ready_band(ready))) : Tree.Tree<Tile.Tile8>}:
  +band = ready_band(ready)
  +wanted = Writes.pure(Tile.Tile8,natural_writes(ready),band_values(band))
  %Equal.sym(Tree.Tree<Tile.Tile8>,group_output(band_group(band)),
      band_writes(band,U32.to_nat(band_jobs(band)),band_plan(band),Packing.packed_output(band_pixels(band)),Packing.packed_output(band_gradients(band))),
      product_output(band_plan(band),band_jobs(band),band_pixels(band),band_gradients(band),band_values(band))) :
    {_ == wanted : Tree.Tree<Tile.Tile8>}
  %Equal.sym(Nat,U32.to_nat(band_jobs(band)),job_count(ready),jobs_exact(ready)) :
    {band_writes(band,_,band_plan(band),Packing.packed_output(band_pixels(band)),Packing.packed_output(band_gradients(band))) == wanted : Tree.Tree<Tile.Tile8>}
  %Equal.sym(Kernel.Plan,band_plan(band),natural_plan(ready),plan_exact(ready)) :
    {band_writes(band,job_count(ready),_,Packing.packed_output(band_pixels(band)),Packing.packed_output(band_gradients(band))) == wanted : Tree.Tree<Tile.Tile8>}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Packing.packed_output(band_pixels(band)),pixel_tree(ready),pixel_tiles_exact(ready)) :
    {band_writes(band,job_count(ready),natural_plan(ready),_,Packing.packed_output(band_gradients(band))) == wanted : Tree.Tree<Tile.Tile8>}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Packing.packed_output(band_gradients(band)),gradient_tree(ready),gradient_tiles_exact(ready)) :
    {band_writes(band,job_count(ready),natural_plan(ready),pixel_tree(ready),_) == wanted : Tree.Tree<Tile.Tile8>}
  {==}

# ---- The numeric model ----

# The global position is the reduction key, independently of band
# boundaries.
type NumericSource is Data:
  NumericSource{input: Buffer.Model<Tile.Tile8>,gradient: Buffer.Model<Tile.Tile8>,geometry: Kernel.Geometry,x_base: U32,d_base: U32}

# Output output at a position: lane output % 8 of the cell of group output / 8
# at the position, the groups of a position side by side from d_base.
def numeric_input(source: NumericSource,output: U32,position: Nat) -> F32:
  NumericSource{input,gradient,geometry,x_base,d_base} = source
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  +output = U32.to_nat(output)
  TileLaws.select_lane(Nat.mod(output,8n),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,gradient),
    U32.from_nat(Nat.add(Nat.add(U32.to_nat(d_base),Nat.mul(position,Ceiling.ceiling(U32.to_nat(o),8n))),Nat.div(output,8n)))))

def numeric_weight(source: NumericSource,tap: U32,position: Nat) -> F32:
  NumericSource{input,gradient,+geometry,x_base,d_base} = source
  Kernel.Geometry{c,o,h,w,k,s,p,rows,+columns,taps} = geometry
  +position = position
  Packing.natural_tap_value(input,geometry,x_base,tap,U32.from_nat(Nat.div(position,U32.to_nat(columns))),U32.from_nat(Nat.mod(position,U32.to_nat(columns))))

def shifted_input() -> Gemm.Both<Nat,NumericSource> -> U32 -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~Nat,~U32,~NumericSource,~Nat,~Ordered.shifted_key,~numeric_input)

def shifted_weight() -> Gemm.Both<Nat,NumericSource> -> U32 -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~Nat,~U32,~NumericSource,~Nat,~Ordered.shifted_key,~numeric_weight)

def numeric_dot(source: NumericSource,tap: U32,output: U32,count: Nat,first: Nat,initial: F32) -> F32:
  OrderedMatrix.dot(~U32,~Nat,~U32,~NumericSource,~numeric_input,~numeric_weight,Ranges.range(count,first),tap,output,source,initial)

def numeric_dot_shifted(+source: NumericSource,+tap: U32,+output: U32,+count: Nat,+first: Nat,+initial: F32) ->
  {numeric_dot(source,tap,output,count,first,initial) == OrderedMatrix.dot(~U32,~Nat,~U32,~Gemm.Both<Nat,NumericSource>,~shifted_input(),~shifted_weight(),Ranges.range(count,0n),tap,output,Gemm.Both{first,source},initial) : F32}:
  +mapped = Ordered.map_keys(~Nat,~Nat,~Nat,~Ordered.shifted_key,Ranges.range(count,0n),first)
  +exact = Equal.trans(List<&2,Nat>,mapped,Ranges.range(count,Nat.add(first,0n)),Ranges.range(count,first),Ordered.range_keys_shift(count,0n,first),Equal.cong(Nat,List<&2,Nat>,start => Ranges.range(count,start),Nat.add(first,0n),first,Addition.zero_right(first)))
  %exact :
    {OrderedMatrix.dot(~U32,~Nat,~U32,~NumericSource,~numeric_input,~numeric_weight,_,tap,output,source,initial) == OrderedMatrix.dot(~U32,~Nat,~U32,~Gemm.Both<Nat,NumericSource>,~shifted_input(),~shifted_weight(),Ranges.range(count,0n),tap,output,Gemm.Both{first,source},initial) : F32}
  Ordered.dot_reindexed(~Nat,~Nat,~U32,~U32,~NumericSource,~Nat,~Ordered.shifted_key,~numeric_input,~numeric_weight,Ranges.range(count,0n),first,tap,output,source,initial)

def numeric_source(+ready: BandReady) -> NumericSource:
  NumericSource{ready_input(ready),ready_gradient(ready),ready_geometry(ready),ready_x_base(ready),ready_d_base(ready)}

# ---- The reads of a job ----

def band_rows_room(+ready: BandReady) -> {Nat.is_le(band_rows(ready),ready_capacity(ready)) == True{} : Bool}:
  +tall = Nat.add(row_count(ready_geometry(ready)),reach(ready_geometry(ready)))
  Order.transitive(band_rows(ready),tall,ready_capacity(ready),rows_within(ready),
    Order.transitive(tall,Nat.mul(tall,band_columns(ready)),ready_capacity(ready),RowMajor.product_covers(tall,band_columns(ready),band_columns_positive(ready)),plane_area_room(ready)))

def start_inside(+ready: BandReady) -> {Nat.is_lt(start_column(ready),column_count(ready_geometry(ready))) == True{} : Bool}:
  DivisionOrder.remainder_small(ready_first(ready),column_count(ready_geometry(ready)),columns_positive(ready))

# The positions of a band from its start column fill no more than its tile
# rows of the image columns.
def entry_room(+ready: BandReady) ->
  {Nat.is_le(Nat.add(Nat.add(start_column(ready),0n),ready_count(ready)),Nat.mul(U32.to_nat(U32.from_nat(band_rows(ready))),column_count(ready_geometry(ready)))) == True{} : Bool}:
  +columns = column_count(ready_geometry(ready))
  +positive = columns_positive(ready)
  +first = ready_first(ready)
  +count = ready_count(ready)
  +row = first_row(ready)
  +last = last_row(ready)
  +start = start_column(ready)
  +rows = band_rows(ready)
  +total = Nat.add(first,count)
  +offset = Nat.mul(row,columns)
  +extra = reach(ready_geometry(ready))
  +total_positive = Bounds.nat_trans(0n,count,total,ready_nonempty(ready),Order.drop_left_le(first,count,total,Order.reflexive(total)))
  +below_next = {%Order.positive_has_last(total,total_positive) : {Nat.is_le(_,Nat.mul(1n+last,columns)) == True{} : Bool}
    Order.strict_follow(band_end(ready),Nat.mul(1n+last,columns),DivisionOrder.next_above(band_end(ready),columns,positive)) :
    {Nat.is_le(total,Nat.mul(1n+last,columns)) == True{} : Bool}}
  +rows_cover = {%Equal.sym(Nat,Nat.add(row,rows),Nat.add(Nat.add(last,1n),extra),rows_end(ready)) : {Nat.is_le(1n+last,_) == True{} : Bool}
    %successor_word(last) : {Nat.is_le(_,Nat.add(Nat.add(last,1n),extra)) == True{} : Bool}
    Order.le_add(Nat.add(last,1n),extra) : {Nat.is_le(1n+last,Nat.add(row,rows)) == True{} : Bool}}
  +within = Order.transitive(total,Nat.mul(1n+last,columns),Nat.mul(Nat.add(row,rows),columns),below_next,RowMajor.mul_le(1n+last,Nat.add(row,rows),columns,rows_cover))
  +split = {%DivisionOrder.quotient_exact(first,columns,positive) : {Nat.is_le(Nat.add(_,count),Nat.mul(Nat.add(row,rows),columns)) == True{} : Bool} within :
    {Nat.is_le(Nat.add(Nat.add(offset,start),count),Nat.mul(Nat.add(row,rows),columns)) == True{} : Bool}}
  +joined = {%Addition.associative(offset,start,count) : {Nat.is_le(_,Nat.mul(Nat.add(row,rows),columns)) == True{} : Bool} split :
    {Nat.is_le(Nat.add(offset,Nat.add(start,count)),Nat.mul(Nat.add(row,rows),columns)) == True{} : Bool}}
  +spread = {%RowMajor.distrib(row,rows,columns) : {Nat.is_le(Nat.add(offset,Nat.add(start,count)),_) == True{} : Bool} joined :
    {Nat.is_le(Nat.add(offset,Nat.add(start,count)),Nat.add(offset,Nat.mul(rows,columns))) == True{} : Bool}}
  +fits = Order.cancel_offset_le(offset,Nat.add(start,count),Nat.mul(rows,columns),spread)
  %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) :
    {Nat.is_le(Nat.add(_,count),Nat.mul(U32.to_nat(U32.from_nat(rows)),columns)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(rows)),rows,FullRange.roundtrip(rows,Machine.capacity(ready_limits(ready)),band_rows_room(ready))) :
    {Nat.is_le(Nat.add(start,count),Nat.mul(_,columns)) == True{} : Bool}
  fits

# The pixel tiles of a job: band columns apart, from its plane row and its
# column phase, along the start column of the band.
def job_grid_base(+ready: BandReady,+index: Nat) -> Nat:
  Nat.add(Nat.mul(job_row_base(ready,index),band_columns(ready)),Nat.div(job_tap_column(ready,index),phase_count(ready_geometry(ready))))

def job_grid(+ready: BandReady,+index: Nat) -> GemmSpec.Grid:
  GemmSpec.Grid{U32.from_nat(job_grid_base(ready,index)),band_columns(ready),geometry_columns(ready_geometry(ready)),U32.from_nat(band_rows(ready)),0,start_column(ready)}

def grid_rows_positive(+ready: BandReady,+index: Nat) -> {Nat.is_lt(0n,GemmSpec.grid_rows(job_grid(ready,index))) == True{} : Bool}:
  +rows = band_rows(ready)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(rows)),rows,FullRange.roundtrip(rows,Machine.capacity(ready_limits(ready)),band_rows_room(ready))) : {Nat.is_lt(0n,_) == True{} : Bool}
  band_rows_positive(ready)

def job_entries(+ready: BandReady,+index: Nat) -> Gemm.Entries:
  +columns = geometry_columns(ready_geometry(ready))
  Gemm.Entries{U32.from_nat(job_address(ready,index)),U32.from_nat(start_column(ready)),0,columns,U32.from_nat(band_rows(ready)),(U32.from_nat(band_columns(ready)) - columns + 1 : U32),0}

def grid_start(+ready: BandReady) -> {Nat.divmod(Nat.add(start_column(ready),0n),column_count(ready_geometry(ready))) == (0n,start_column(ready)) : Nat & Nat}:
  +column = start_column(ready)
  %Equal.sym(Nat,Nat.add(column,0n),column,Addition.zero_right(column)) : {Nat.divmod(_,column_count(ready_geometry(ready))) == (0n,column) : Nat & Nat}
  Division.small(column,column_count(ready_geometry(ready)),start_inside(ready))

def grid_address_exact(+ready: BandReady,+index: Nat) ->
  {U32.from_nat(job_address(ready,index)) == (U32.from_nat(job_grid_base(ready,index)) + U32.from_nat(start_column(ready)) : U32) : U32}:
  +row = Nat.mul(job_row_base(ready,index),band_columns(ready))
  +shift = Nat.div(job_tap_column(ready,index),phase_count(ready_geometry(ready)))
  +column = start_column(ready)
  +base = job_grid_base(ready,index)
  +order = Equal.trans(Nat,Nat.add(Nat.add(row,column),shift),Nat.add(row,Nat.add(column,shift)),Nat.add(Nat.add(row,shift),column),
    Addition.associative(row,column,shift),
    Equal.trans(Nat,Nat.add(row,Nat.add(column,shift)),Nat.add(row,Nat.add(shift,column)),Nat.add(Nat.add(row,shift),column),
      Equal.cong(Nat,Nat,tail => Nat.add(row,tail),Nat.add(column,shift),Nat.add(shift,column),Order.commutative(column,shift)),
      Equal.sym(Nat,Nat.add(Nat.add(row,shift),column),Nat.add(row,Nat.add(shift,column)),Addition.associative(row,shift,column))))
  Equal.trans(U32,U32.from_nat(job_address(ready,index)),U32.from_nat(Nat.add(base,column)),(U32.from_nat(base) + U32.from_nat(column) : U32),
    Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.add(row,column),shift),Nat.add(base,column),order),
    Equal.sym(U32,(U32.from_nat(base) + U32.from_nat(column) : U32),U32.from_nat(Nat.add(base,column)),WordArithmetic.native_add(base,column)))

# The entries of a job are its grid from step 0.
def grid_exact(+ready: BandReady,+index: Nat) -> {job_entries(ready,index) == GemmSpec.grid_entry(job_grid(ready,index),0n) : Gemm.Entries}:
  +grid = job_grid(ready,index)
  +columns = geometry_columns(ready_geometry(ready))
  +start = U32.from_nat(start_column(ready))
  +rows = U32.from_nat(band_rows(ready))
  +jump = (U32.from_nat(band_columns(ready)) - columns + 1 : U32)
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(start_column(ready),0n),column_count(ready_geometry(ready))),(0n,start_column(ready)),grid_start(ready)) :
    {job_entries(ready,index) == GemmSpec.grid_cell(grid,_) : Gemm.Entries}
  %Equal.sym(Nat & Nat,Nat.divmod(0n,GemmSpec.grid_rows(grid)),(0n,0n),Division.small(0n,GemmSpec.grid_rows(grid),grid_rows_positive(ready,index))) :
    {job_entries(ready,index) == GemmSpec.grid_placed(grid,_,start_column(ready)) : Gemm.Entries}
  %Equal.sym(U32,U32.from_nat(job_address(ready,index)),(U32.from_nat(job_grid_base(ready,index)) + start : U32),grid_address_exact(ready,index)) :
    {Gemm.Entries{_,start,0,columns,rows,jump,0} == GemmSpec.grid_at(grid,0n,0n,start_column(ready)) : Gemm.Entries}
  {==}

def advanced(+first: U32,+second: Gemm.Entries) ->
  {Kernel.advance(Gemm.Both{first,second}) == Gemm.Step{Pair.fst(U32,U32,Gemm.direct(first)),Pair.fst(U32,Gemm.Entries,Gemm.entry(second)),
    Gemm.Both{Pair.snd(U32,U32,Gemm.direct(first)),Pair.snd(U32,Gemm.Entries,Gemm.entry(second))}} : Gemm.Step<Gemm.Both<U32,Gemm.Entries>>}:
  Gemm.Entries{+address,+column,+row,+columns,+rows,+row_jump,+output_jump} = second
  {==}

# Step t of a job reads gradient tile output block * count + t and the
# pixel tile at step t of its grid.
def job_key(+ready: BandReady,+index: Nat) -> Gemm.Both<Nat,GemmSpec.Grid>:
  Gemm.Both{Nat.mul(job_output_block(ready,index),ready_count(ready)),job_grid(ready,index)}

def job_keys(+ready: BandReady,+index: Nat) -> List<&2,ReductionLaws.AddressPair>:
  Ordered.map_keys(~Nat,~ReductionLaws.AddressPair,~Gemm.Both<Nat,GemmSpec.Grid>,~GemmSpec.both_key(~Nat,~Nat,~GemmSpec.Grid,~GemmSpec.offset_word,~GemmSpec.grid_read),
    Ranges.range(ready_count(ready),0n),job_key(ready,index))

def job_taps(+ready: BandReady,+index: Nat) ->
  {GemmSpec.taps(~Kernel.Cursor(),~Kernel.advance,ready_count(ready),Gemm.Both{U32.from_nat(Nat.mul(job_output_block(ready,index),ready_count(ready))),job_entries(ready,index)}) ==
    job_keys(ready,index) : List<&2,ReductionLaws.AddressPair>}:
  +count = ready_count(ready)
  +base = Nat.mul(job_output_block(ready,index),count)
  +grid = job_grid(ready,index)
  +inputs = GemmSpec.reads(~U32,~Gemm.direct,count,U32.from_nat(base))
  +pixels = Ordered.map_keys(~Nat,~U32,~GemmSpec.Grid,~GemmSpec.grid_read,Ranges.range(count,0n),grid)
  +wanted = job_keys(ready,index)
  %Equal.sym(List<&2,ReductionLaws.AddressPair>,GemmSpec.taps(~Kernel.Cursor(),~Kernel.advance,count,Gemm.Both{U32.from_nat(base),job_entries(ready,index)}),
      GemmSpec.paired(inputs,GemmSpec.reads(~Gemm.Entries,~Gemm.entry,count,job_entries(ready,index))),
      GemmSpec.both_taps(~U32,~Gemm.Entries,~Gemm.direct,~Gemm.entry,~Kernel.advance,~advanced,count,U32.from_nat(base),job_entries(ready,index))) :
    {_ == wanted : List<&2,ReductionLaws.AddressPair>}
  %Equal.sym(Gemm.Entries,job_entries(ready,index),GemmSpec.grid_entry(grid,0n),grid_exact(ready,index)) :
    {GemmSpec.paired(inputs,GemmSpec.reads(~Gemm.Entries,~Gemm.entry,count,_)) == wanted : List<&2,ReductionLaws.AddressPair>}
  %Equal.sym(List<&2,U32>,GemmSpec.reads(~Gemm.Entries,~Gemm.entry,count,GemmSpec.grid_entry(grid,0n)),pixels,
      GemmSpec.entry_reads(count,grid,0n,columns_positive(ready),grid_rows_positive(ready,index))) :
    {GemmSpec.paired(inputs,_) == wanted : List<&2,ReductionLaws.AddressPair>}
  +shifted = Equal.trans(List<&2,U32>,inputs,GemmSpec.reads(~U32,~Gemm.direct,count,U32.from_nat(Nat.add(base,0n))),
    Ordered.map_keys(~Nat,~U32,~Nat,~GemmSpec.offset_word,Ranges.range(count,0n),base),
    Equal.cong(Nat,List<&2,U32>,start => GemmSpec.reads(~U32,~Gemm.direct,count,U32.from_nat(start)),base,Nat.add(base,0n),Equal.sym(Nat,Nat.add(base,0n),base,Addition.zero_right(base))),
    GemmSpec.direct_reads(count,base,0n))
  %Equal.sym(List<&2,U32>,inputs,Ordered.map_keys(~Nat,~U32,~Nat,~GemmSpec.offset_word,Ranges.range(count,0n),base),shifted) :
    {GemmSpec.paired(_,pixels) == wanted : List<&2,ReductionLaws.AddressPair>}
  GemmSpec.paired_keys(~Nat,~Nat,~GemmSpec.Grid,~GemmSpec.offset_word,~GemmSpec.grid_read,Ranges.range(count,0n),base,grid)

# ---- The samples of a step ----

# A fresh tree of count tiles has room for them.
def fresh_sized(+limits: Machine.Limits,+count: Nat,+room: {Nat.is_le(count,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> AllocationDepth.Sized<U32.from_nat(count)>:
  +word = U32.from_nat(count)
  +exact = FullRange.roundtrip(count,Machine.capacity(limits),room)
  AllocationDepth.bounded(word,Encoded.native_le_transitive(word,Machine.capacity(limits),1073741824,
    Equal.trans(Bool,U32.is_le(word,Machine.capacity(limits)),Nat.is_le(U32.to_nat(word),U32.to_nat(Machine.capacity(limits))),True{},Comparison.le(word,Machine.capacity(limits)),
      %Equal.sym(Nat,U32.to_nat(word),count,exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room),
    Limits.capacity_allocatable(limits)))

def fresh_capacity(+limits: Machine.Limits,+count: Nat,+room: {Nat.is_le(count,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(count,U32.to_nat(Tree.capacity(Tile.Tile8,Tree.fresh(Tile.Tile8,Storage.depth_for(U32.from_nat(count)),Tile.filled(0.0))))) == True{} : Bool}:
  +word = U32.from_nat(count)
  +depth = Storage.depth_for(word)
  +tree = Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))
  %Equal.sym(U32,Tree.capacity(Tile.Tile8,tree),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,tree,Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)))) :
    {Nat.is_le(count,U32.to_nat(_)) == True{} : Bool}
  %FullRange.roundtrip(count,Machine.capacity(limits),room) : {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
  AllocationDepth.allocation_room(word,fresh_sized(limits,count,room))

def rows_room(+ready: BandReady) -> {Nat.is_le(row_count(ready_geometry(ready)),ready_capacity(ready)) == True{} : Bool}:
  +rows = row_count(ready_geometry(ready))
  Order.transitive(rows,Nat.mul(rows,column_count(ready_geometry(ready))),ready_capacity(ready),RowMajor.product_covers(rows,column_count(ready_geometry(ready)),columns_positive(ready)),
    area_room(ready_geometry(ready),ready_domain(ready)))

# Step t of a band is position first + t, at its row and column.
def step_position(+ready: BandReady,+step: Nat) -> Nat:
  Nat.add(ready_first(ready),step)

def step_row(+ready: BandReady,+step: Nat) -> Nat:
  Nat.div(step_position(ready,step),column_count(ready_geometry(ready)))

def step_column(+ready: BandReady,+step: Nat) -> Nat:
  Nat.mod(step_position(ready,step),column_count(ready_geometry(ready)))

def position_inside(+ready: BandReady,+step: Nat,+inside: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool}) ->
  {Nat.is_lt(step_position(ready,step),Nat.mul(row_count(ready_geometry(ready)),column_count(ready_geometry(ready)))) == True{} : Bool}:
  Bounds.nat_trans(step_position(ready,step),Nat.add(ready_first(ready),ready_count(ready)),Nat.mul(row_count(ready_geometry(ready)),column_count(ready_geometry(ready))),
    Order.offset_lt(ready_first(ready),step,ready_count(ready),inside),ready_room(ready))

def step_row_inside(+ready: BandReady,+step: Nat,+inside: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool}) ->
  {Nat.is_lt(step_row(ready,step),row_count(ready_geometry(ready))) == True{} : Bool}:
  DivisionOrder.quotient_below(step_position(ready,step),row_count(ready_geometry(ready)),column_count(ready_geometry(ready)),columns_positive(ready),position_inside(ready,step,inside))

def step_column_inside(+ready: BandReady,+step: Nat) -> {Nat.is_lt(step_column(ready,step),column_count(ready_geometry(ready))) == True{} : Bool}:
  DivisionOrder.remainder_small(step_position(ready,step),column_count(ready_geometry(ready)),columns_positive(ready))

def step_row_back(+ready: BandReady,+step: Nat,+inside: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool}) ->
  {step_row(ready,step) == U32.to_nat(U32.from_nat(step_row(ready,step))) : Nat}:
  +row = step_row(ready,step)
  Equal.sym(Nat,U32.to_nat(U32.from_nat(row)),row,FullRange.roundtrip(row,Machine.capacity(ready_limits(ready)),
    Order.transitive(row,row_count(ready_geometry(ready)),ready_capacity(ready),Bounds.strict_le(row,row_count(ready_geometry(ready)),step_row_inside(ready,step,inside)),rows_room(ready))))

def step_column_back(+ready: BandReady,+step: Nat) -> {step_column(ready,step) == U32.to_nat(U32.from_nat(step_column(ready,step))) : Nat}:
  +column = step_column(ready,step)
  Equal.sym(Nat,U32.to_nat(U32.from_nat(column)),column,FullRange.roundtrip(column,Machine.capacity(ready_limits(ready)),
    Order.transitive(column,column_count(ready_geometry(ready)),ready_capacity(ready),Bounds.strict_le(column,column_count(ready_geometry(ready)),step_column_inside(ready,step)),
      columns_room(ready))))

def gradient_sample(+ready: BandReady,+block: Nat,+step: Nat,+lane: Nat,+block_inside: {Nat.is_lt(block,output_blocks(ready_geometry(ready))) == True{} : Bool},
  +step_inside: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool},+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +output_inside: {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),output_count(ready_geometry(ready))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,gradient_tree(ready),U32.from_nat(Nat.add(Nat.mul(block,ready_count(ready)),step)))) ==
    numeric_input(numeric_source(ready),U32.from_nat(Nat.add(Nat.mul(block,8n),lane)),step_position(ready,step)) : F32}:
  BandReady{+geometry,+domain,+x_base,+d_base,+first,+count,+input,+gradient,+nonempty,+room} = ready
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +limits = ready_limits(ready)
  +cap = Machine.capacity(limits)
  +outputs = U32.to_nat(o)
  +blocks = output_blocks(geometry)
  +d_n = U32.to_nat(d_base)
  +position = Nat.add(first,step)
  +output = Nat.add(Nat.mul(block,8n),lane)
  +index = Nat.add(Nat.mul(block,count),step)
  +total = gradient_count(ready)
  +context = natural_positions(ready)
  +fresh = gradient_fresh(ready)
  +depth = Storage.depth_for(U32.from_nat(total))
  +fresh_room = gradient_count_room(ready)
  +limit = allocation_depth_limit(U32.from_nat(total),fresh_sized(limits,total,fresh_room))
  +index_inside = RowMajor.radix(block,blocks,step,count,block_inside,step_inside)
  +index_room = Order.transitive(index,total,U32.to_nat(cap),Bounds.strict_le(index,total,index_inside),fresh_room)
  +count_fits = count_room(ready)
  +stored = Packing.gradients_stored(context,total,gradient,fresh,depth,Tree.capacity(Tile.Tile8,fresh),index,{==},Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)),limit,
    {%Equal.sym(U32,Tree.capacity(Tile.Tile8,fresh),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,fresh,Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)))) :
      {U32.is_le(_,2147483648) == True{} : Bool} Shift.power_fits(depth,limit) : {U32.is_le(Tree.capacity(Tile.Tile8,fresh),2147483648) == True{} : Bool}},
    fresh_capacity(limits,total,fresh_room),index_inside)
  +quotient = Equal.trans(U32,(U32.from_nat(index) / U32.from_nat(count) : U32),U32.from_nat(Nat.div(index,count)),U32.from_nat(block),
    word_quotient(limits,index,count,index_room,count_fits,nonempty),
    Equal.cong(Nat,U32,U32.from_nat,Nat.div(index,count),block,Radix.quotient(block,step,count,nonempty,step_inside)))
  +remainder = Equal.trans(U32,(U32.from_nat(index) % U32.from_nat(count) : U32),U32.from_nat(Nat.mod(index,count)),U32.from_nat(step),
    word_remainder(limits,index,count,index_room,count_fits,nonempty),
    Equal.cong(Nat,U32,U32.from_nat,Nat.mod(index,count),step,Radix.remainder(block,step,count,nonempty,step_inside)))
  +base = gradient_base(ready)
  +moved = Nat.mul(step,blocks)
  +tree = Buffer.tree(~Tile.Tile8,gradient)
  +base_w = U32.from_nat(base)
  +blocks_w = U32.from_nat(blocks)
  +output_back = Equal.sym(Nat,U32.to_nat(U32.from_nat(output)),output,FullRange.roundtrip(output,cap,
    Order.transitive(output,outputs,U32.to_nat(cap),Bounds.strict_le(output,outputs,output_inside),outputs_room(geometry,domain))))
  +regrouped = Equal.trans(Nat,Nat.add(Nat.add(d_n,Nat.mul(first,blocks)),moved),Nat.add(d_n,Nat.add(Nat.mul(first,blocks),moved)),Nat.add(d_n,Nat.mul(position,blocks)),
    Addition.associative(d_n,Nat.mul(first,blocks),moved),
    Equal.cong(Nat,Nat,tail => Nat.add(d_n,tail),Nat.add(Nat.mul(first,blocks),moved),Nat.mul(position,blocks),
      Equal.sym(Nat,Nat.mul(position,blocks),Nat.add(Nat.mul(first,blocks),moved),RowMajor.distrib(first,step,blocks))))
  +cell = Nat.add(Nat.add(d_n,Nat.mul(position,blocks)),block)
  +wanted = numeric_input(numeric_source(ready),U32.from_nat(output),position)
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,gradient_tree(ready),U32.from_nat(index)),Packing.gradient_sample(gradient,context,U32.from_nat(index)),stored) :
    {TileLaws.select_lane(lane,_) == wanted : F32}
  %Equal.sym(U32,(U32.from_nat(index) % U32.from_nat(count) : U32),U32.from_nat(step),remainder) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,(base_w + _ * blocks_w + U32.from_nat(index) / U32.from_nat(count) : U32))) == wanted : F32}
  %Equal.sym(U32,(U32.from_nat(index) / U32.from_nat(count) : U32),U32.from_nat(block),quotient) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,(base_w + U32.from_nat(step) * blocks_w + _ : U32))) == wanted : F32}
  %Equal.sym(U32,(U32.from_nat(step) * blocks_w : U32),U32.from_nat(moved),Multiplication.native_encoded_multiply(step,blocks)) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,(base_w + _ + U32.from_nat(block) : U32))) == wanted : F32}
  %Equal.sym(U32,(base_w + U32.from_nat(moved) : U32),U32.from_nat(Nat.add(base,moved)),WordArithmetic.native_add(base,moved)) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,(_ + U32.from_nat(block) : U32))) == wanted : F32}
  %Equal.sym(U32,(U32.from_nat(Nat.add(base,moved)) + U32.from_nat(block) : U32),U32.from_nat(Nat.add(Nat.add(base,moved),block)),
      WordArithmetic.native_add(Nat.add(base,moved),block)) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,_)) == wanted : F32}
  %Equal.sym(Nat,Nat.add(Nat.add(d_n,Nat.mul(first,blocks)),moved),Nat.add(d_n,Nat.mul(position,blocks)),regrouped) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(_,block)))) == wanted : F32}
  %output_back :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(cell))) ==
      TileLaws.select_lane(Nat.mod(_,8n),Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.add(d_n,Nat.mul(position,blocks)),Nat.div(_,8n))))) : F32}
  %Equal.sym(Nat,Nat.mod(output,8n),lane,Radix.remainder(block,lane,8n,{==},small)) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(cell))) ==
      TileLaws.select_lane(_,Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.add(d_n,Nat.mul(position,blocks)),Nat.div(output,8n))))) : F32}
  %Equal.sym(Nat,Nat.div(output,8n),block,Radix.quotient(block,lane,8n,{==},small)) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(cell))) ==
      TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.add(d_n,Nat.mul(position,blocks)),_)))) : F32}
  {==}

# ---- The pixel tile of a step ----

# Step t lies start column + t into the band from its first row.
def grid_row(+ready: BandReady,+step: Nat) -> Nat:
  Nat.div(Nat.add(start_column(ready),step),column_count(ready_geometry(ready)))

def grid_column(+ready: BandReady,+step: Nat) -> Nat:
  Nat.mod(Nat.add(start_column(ready),step),column_count(ready_geometry(ready)))

def step_split(+ready: BandReady,+step: Nat) ->
  {step_position(ready,step) == Nat.add(Nat.mul(first_row(ready),column_count(ready_geometry(ready))),Nat.add(start_column(ready),step)) : Nat}:
  +columns = column_count(ready_geometry(ready))
  +head = Nat.mul(first_row(ready),columns)
  Equal.trans(Nat,step_position(ready,step),Nat.add(Nat.add(head,start_column(ready)),step),Nat.add(head,Nat.add(start_column(ready),step)),
    Equal.cong(Nat,Nat,first => Nat.add(first,step),ready_first(ready),Nat.add(head,start_column(ready)),DivisionOrder.quotient_exact(ready_first(ready),columns,columns_positive(ready))),
    Addition.associative(head,start_column(ready),step))

def step_row_split(+ready: BandReady,+step: Nat) -> {step_row(ready,step) == Nat.add(first_row(ready),grid_row(ready,step)) : Nat}:
  +columns = column_count(ready_geometry(ready))
  +inner = Nat.add(start_column(ready),step)
  Equal.trans(Nat,step_row(ready,step),Nat.div(Nat.add(Nat.mul(first_row(ready),columns),inner),columns),Nat.add(first_row(ready),grid_row(ready,step)),
    Equal.cong(Nat,Nat,position => Nat.div(position,columns),step_position(ready,step),Nat.add(Nat.mul(first_row(ready),columns),inner),step_split(ready,step)),
    DivisionOrder.quotient_shift(inner,columns,first_row(ready),columns_positive(ready)))

def step_column_split(+ready: BandReady,+step: Nat) -> {step_column(ready,step) == grid_column(ready,step) : Nat}:
  +columns = column_count(ready_geometry(ready))
  +inner = Nat.add(start_column(ready),step)
  Equal.trans(Nat,step_column(ready,step),Nat.mod(Nat.add(Nat.mul(first_row(ready),columns),inner),columns),grid_column(ready,step),
    Equal.cong(Nat,Nat,position => Nat.mod(position,columns),step_position(ready,step),Nat.add(Nat.mul(first_row(ready),columns),inner),step_split(ready,step)),
    DivisionOrder.remainder_shift(inner,columns,first_row(ready),columns_positive(ready)))

# The tile of step t of a job: its grid position moved by the tap rows and
# columns of the phases.
def tap_row_shift(+ready: BandReady,+index: Nat) -> Nat:
  Nat.div(job_tap_row(ready,index),phase_count(ready_geometry(ready)))

def tap_column_shift(+ready: BandReady,+index: Nat) -> Nat:
  Nat.div(job_tap_column(ready,index),phase_count(ready_geometry(ready)))

def tile_row(+ready: BandReady,+index: Nat,+step: Nat) -> Nat:
  Nat.add(grid_row(ready,step),tap_row_shift(ready,index))

def tile_column(+ready: BandReady,+index: Nat,+step: Nat) -> Nat:
  Nat.add(grid_column(ready,step),tap_column_shift(ready,index))

def tile_index(+ready: BandReady,+index: Nat,+step: Nat) -> Nat:
  Nat.add(Nat.mul(tile_row(ready,index,step),band_columns(ready)),tile_column(ready,index,step))

def sheet_row(+ready: BandReady,+index: Nat,+step: Nat) -> Nat:
  Nat.add(step_row(ready,step),tap_row_shift(ready,index))

def sheet_column(+ready: BandReady,+index: Nat,+step: Nat) -> Nat:
  Nat.add(step_column(ready,step),tap_column_shift(ready,index))

# Five terms of a sum in another grouping.
def rearranged_sum(+head: Nat,+first: Nat,+second: Nat,+third: Nat,+fourth: Nat) ->
  {Nat.add(Nat.add(Nat.add(head,first),second),Nat.add(third,fourth)) == Nat.add(head,Nat.add(Nat.add(third,first),Nat.add(fourth,second))) : Nat}:
  +rest = Nat.add(third,fourth)
  +swapped = Equal.trans(Nat,Nat.add(first,Nat.add(second,rest)),Nat.add(first,Nat.add(third,Nat.add(second,fourth))),Nat.add(Nat.add(third,first),Nat.add(fourth,second)),
    Equal.cong(Nat,Nat,tail => Nat.add(first,tail),Nat.add(second,rest),Nat.add(third,Nat.add(second,fourth)),RowMajor.add_swap(second,third,fourth)),
    Equal.trans(Nat,Nat.add(first,Nat.add(third,Nat.add(second,fourth))),Nat.add(third,Nat.add(first,Nat.add(second,fourth))),Nat.add(Nat.add(third,first),Nat.add(fourth,second)),
      RowMajor.add_swap(first,third,Nat.add(second,fourth)),
      Equal.trans(Nat,Nat.add(third,Nat.add(first,Nat.add(second,fourth))),Nat.add(third,Nat.add(first,Nat.add(fourth,second))),Nat.add(Nat.add(third,first),Nat.add(fourth,second)),
        Equal.cong(Nat,Nat,tail => Nat.add(third,Nat.add(first,tail)),Nat.add(second,fourth),Nat.add(fourth,second),Order.commutative(second,fourth)),
        Equal.sym(Nat,Nat.add(Nat.add(third,first),Nat.add(fourth,second)),Nat.add(third,Nat.add(first,Nat.add(fourth,second))),Addition.associative(third,first,Nat.add(fourth,second))))))
  Equal.trans(Nat,Nat.add(Nat.add(Nat.add(head,first),second),rest),Nat.add(Nat.add(head,first),Nat.add(second,rest)),Nat.add(head,Nat.add(Nat.add(third,first),Nat.add(fourth,second))),
    Addition.associative(Nat.add(head,first),second,rest),
    Equal.trans(Nat,Nat.add(Nat.add(head,first),Nat.add(second,rest)),Nat.add(head,Nat.add(first,Nat.add(second,rest))),Nat.add(head,Nat.add(Nat.add(third,first),Nat.add(fourth,second))),
      Addition.associative(head,first,Nat.add(second,rest)),
      Equal.cong(Nat,Nat,tail => Nat.add(head,tail),Nat.add(first,Nat.add(second,rest)),Nat.add(Nat.add(third,first),Nat.add(fourth,second)),swapped)))

# Step t of a job lies in the first plane of its grid.
def grid_row_inside(+ready: BandReady,+index: Nat,+step: Nat,+within: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool}) ->
  {Nat.is_lt(grid_row(ready,step),GemmSpec.grid_rows(job_grid(ready,index))) == True{} : Bool}:
  +columns = column_count(ready_geometry(ready))
  +start = start_column(ready)
  +count = ready_count(ready)
  +rows = GemmSpec.grid_rows(job_grid(ready,index))
  +room = {%Addition.zero_right(start) : {Nat.is_le(Nat.add(_,count),Nat.mul(rows,columns)) == True{} : Bool} entry_room(ready) :
    {Nat.is_le(Nat.add(start,count),Nat.mul(rows,columns)) == True{} : Bool}}
  DivisionOrder.quotient_below(Nat.add(start,step),rows,columns,columns_positive(ready),
    Bounds.nat_trans(Nat.add(start,step),Nat.add(start,count),Nat.mul(rows,columns),Order.offset_lt(start,step,count,within),room))

# Step t of a job reads tile plane * area + tile index of the pixel tree.
def grid_read_exact(+ready: BandReady,+index: Nat,+step: Nat,+within: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool}) ->
  {GemmSpec.grid_read(job_grid(ready,index),step) == U32.from_nat(Nat.add(Nat.mul(job_plane(ready,index),band_area(ready)),tile_index(ready,index,step))) : U32}:
  +grid = job_grid(ready,index)
  +columns = band_columns(ready)
  +plane = job_plane(ready,index)
  +rows = band_rows(ready)
  +down = tap_row_shift(ready,index)
  +across = tap_column_shift(ready,index)
  +row = grid_row(ready,step)
  +column = grid_column(ready,step)
  +head = Nat.mul(plane,band_area(ready))
  +lift = Nat.mul(down,columns)
  +moved = Nat.mul(row,columns)
  +base = job_grid_base(ready,index)
  +wanted = U32.from_nat(Nat.add(head,tile_index(ready,index,step)))
  %Equal.sym(Nat & Nat,Nat.divmod(GemmSpec.grid_position(grid,step),GemmSpec.grid_width(grid)),(row,column),GemmSpec.grid_split(grid,step,columns_positive(ready))) :
    {GemmSpec.grid_word(grid,_) == wanted : U32}
  %Equal.sym(U32,GemmSpec.grid_word(grid,(row,column)),GemmSpec.grid_at_word(grid,0n,row,column),GemmSpec.first_plane(grid,row,column,grid_row_inside(ready,index,step,within))) :
    {_ == wanted : U32}
  %Equal.sym(U32,(U32.from_nat(base) + U32.from_nat(Nat.add(moved,column)) : U32),U32.from_nat(Nat.add(base,Nat.add(moved,column))),
      WordArithmetic.native_add(base,Nat.add(moved,column))) :
    {_ == wanted : U32}
  +base_exact = Equal.cong(Nat,Nat,first => Nat.add(first,across),Nat.mul(Nat.add(Nat.mul(plane,rows),down),columns),Nat.add(head,lift),
    Equal.trans(Nat,Nat.mul(Nat.add(Nat.mul(plane,rows),down),columns),Nat.add(Nat.mul(Nat.mul(plane,rows),columns),lift),Nat.add(head,lift),
      RowMajor.distrib(Nat.mul(plane,rows),down,columns),
      Equal.cong(Nat,Nat,first => Nat.add(first,lift),Nat.mul(Nat.mul(plane,rows),columns),head,RowMajor.assoc(plane,rows,columns))))
  +tile_exact = Equal.cong(Nat,Nat,first => Nat.add(first,Nat.add(column,across)),Nat.mul(Nat.add(row,down),columns),Nat.add(moved,lift),RowMajor.distrib(row,down,columns))
  Equal.cong(Nat,U32,U32.from_nat,Nat.add(base,Nat.add(moved,column)),Nat.add(head,tile_index(ready,index,step)),
    Equal.trans(Nat,Nat.add(base,Nat.add(moved,column)),Nat.add(Nat.add(Nat.add(head,lift),across),Nat.add(moved,column)),Nat.add(head,tile_index(ready,index,step)),
      Equal.cong(Nat,Nat,first => Nat.add(first,Nat.add(moved,column)),base,Nat.add(Nat.add(head,lift),across),base_exact),
      Equal.trans(Nat,Nat.add(Nat.add(Nat.add(head,lift),across),Nat.add(moved,column)),Nat.add(head,Nat.add(Nat.add(moved,lift),Nat.add(column,across))),Nat.add(head,tile_index(ready,index,step)),
        rearranged_sum(head,lift,across,moved,column),
        Equal.cong(Nat,Nat,tail => Nat.add(head,tail),Nat.add(Nat.add(moved,lift),Nat.add(column,across)),tile_index(ready,index,step),
          Equal.sym(Nat,tile_index(ready,index,step),Nat.add(Nat.add(moved,lift),Nat.add(column,across)),tile_exact)))))

# A tap offset moves a tile by no more than the reach.
def shift_within(+ready: BandReady,+offset: Nat,+below: {Nat.is_lt(offset,kernel_size(ready_geometry(ready))) == True{} : Bool}) ->
  {Nat.is_le(Nat.div(offset,phase_count(ready_geometry(ready))),reach(ready_geometry(ready))) == True{} : Bool}:
  +geometry = ready_geometry(ready)
  DivisionOrder.quotient_le(offset,Nat.sub(kernel_size(geometry),1n),phase_count(geometry),phases_positive(geometry,ready_domain(ready)),Order.before_last(offset,kernel_size(geometry),below))

def step_before_end(+ready: BandReady,+step: Nat,+inside: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool}) -> {Nat.is_le(step_position(ready,step),band_end(ready)) == True{} : Bool}:
  Order.before_last(step_position(ready,step),Nat.add(ready_first(ready),ready_count(ready)),Order.offset_lt(ready_first(ready),step,ready_count(ready),inside))

def sheet_row_split(+ready: BandReady,+index: Nat,+step: Nat) -> {sheet_row(ready,index,step) == Nat.add(first_row(ready),tile_row(ready,index,step)) : Nat}:
  +first = first_row(ready)
  Equal.trans(Nat,sheet_row(ready,index,step),Nat.add(Nat.add(first,grid_row(ready,step)),tap_row_shift(ready,index)),Nat.add(first,tile_row(ready,index,step)),
    Equal.cong(Nat,Nat,row => Nat.add(row,tap_row_shift(ready,index)),step_row(ready,step),Nat.add(first,grid_row(ready,step)),step_row_split(ready,step)),
    Addition.associative(first,grid_row(ready,step),tap_row_shift(ready,index)))

# The sheet row of a step lies below the tile rows of the band.
def sheet_row_inside(+ready: BandReady,+index: Nat,+step: Nat,+inside: JobInside<ready,index>,+step_inside: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool}) ->
  {Nat.is_lt(sheet_row(ready,index,step),Nat.add(first_row(ready),band_rows(ready))) == True{} : Bool}:
  JobInside{index_room,output_block,quotient_room,tap,block,+tap_row,tap_column,channels,outputs} = inside
  +last = last_row(ready)
  +extra = reach(ready_geometry(ready))
  +row_le = DivisionOrder.quotient_le(step_position(ready,step),band_end(ready),column_count(ready_geometry(ready)),columns_positive(ready),step_before_end(ready,step,step_inside))
  +sum_le = Encoded.sum_bounds(step_row(ready,step),tap_row_shift(ready,index),last,extra,row_le,shift_within(ready,job_tap_row(ready,index),tap_row))
  +next = {%Equal.sym(Nat,Nat.add(last,1n),1n+last,successor_word(last)) : {Nat.is_lt(last,_) == True{} : Bool} Order.lt_succ(last,last,Order.reflexive(last)) :
    {Nat.is_lt(last,Nat.add(last,1n)) == True{} : Bool}}
  +end = {%Equal.sym(Nat,Nat.add(first_row(ready),band_rows(ready)),Nat.add(Nat.add(last,1n),extra),rows_end(ready)) : {Nat.is_lt(Nat.add(last,extra),_) == True{} : Bool}
    Order.add_lt(last,Nat.add(last,1n),extra,next) : {Nat.is_lt(Nat.add(last,extra),Nat.add(first_row(ready),band_rows(ready))) == True{} : Bool}}
  Order.le_lt_trans(sheet_row(ready,index,step),Nat.add(last,extra),Nat.add(first_row(ready),band_rows(ready)),sum_le,end)

def tile_row_inside(+ready: BandReady,+index: Nat,+step: Nat,+inside: JobInside<ready,index>,+step_inside: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool}) ->
  {Nat.is_lt(tile_row(ready,index,step),band_rows(ready)) == True{} : Bool}:
  Order.cancel_offset_lt(first_row(ready),tile_row(ready,index,step),band_rows(ready),
    %sheet_row_split(ready,index,step) : {Nat.is_lt(_,Nat.add(first_row(ready),band_rows(ready))) == True{} : Bool}
    sheet_row_inside(ready,index,step,inside,step_inside))

def tile_column_inside(+ready: BandReady,+index: Nat,+step: Nat,+inside: JobInside<ready,index>) ->
  {Nat.is_lt(tile_column(ready,index,step),band_columns(ready)) == True{} : Bool}:
  JobInside{index_room,output_block,quotient_room,tap,block,tap_row,+tap_column,channels,outputs} = inside
  +columns = column_count(ready_geometry(ready))
  +across = tap_column_shift(ready,index)
  +column = grid_column(ready,step)
  Bounds.nat_trans(tile_column(ready,index,step),Nat.add(columns,across),band_columns(ready),
    Order.add_lt(column,columns,across,DivisionOrder.remainder_small(Nat.add(start_column(ready),step),columns,columns_positive(ready))),
    Order.offset_le(columns,across,reach(ready_geometry(ready)),shift_within(ready,job_tap_column(ready,index),tap_column)))

def tile_index_inside(+ready: BandReady,+index: Nat,+step: Nat,+inside: JobInside<ready,index>,+step_inside: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool}) ->
  {Nat.is_lt(tile_index(ready,index,step),band_area(ready)) == True{} : Bool}:
  RowMajor.radix(tile_row(ready,index,step),band_rows(ready),tile_column(ready,index,step),band_columns(ready),tile_row_inside(ready,index,step,inside,step_inside),
    tile_column_inside(ready,index,step,inside))

def sheet_column_inside(+ready: BandReady,+index: Nat,+step: Nat,+inside: JobInside<ready,index>) ->
  {Nat.is_lt(sheet_column(ready,index,step),band_columns(ready)) == True{} : Bool}:
  %Equal.sym(Nat,step_column(ready,step),grid_column(ready,step),step_column_split(ready,step)) :
    {Nat.is_lt(Nat.add(_,tap_column_shift(ready,index)),band_columns(ready)) == True{} : Bool}
  tile_column_inside(ready,index,step,inside)

# ---- The pixel value of a step ----

def geometry_height(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  h

def geometry_width(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  w

def geometry_padding(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  p

# Row r of a job is input channel 8 * block + r at its tap.
def cell_channel(+ready: BandReady,+index: Nat,+lane: Nat) -> Nat:
  Nat.add(Nat.mul(job_block(ready,index),8n),lane)

def cell_tap(+ready: BandReady,+index: Nat,+lane: Nat) -> Nat:
  Nat.add(Nat.mul(cell_channel(ready,index,lane),square_size(ready_geometry(ready))),job_tap(ready,index))

def geometry_groups(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  ((c + 7) / 8 : U32)

# The input pixel that tap (a, b) of a row reads at step t.
def pixel_value(+ready: BandReady,+index: Nat,+lane: Nat,+step: Nat) -> F32:
  +geometry = ready_geometry(ready)
  +stride = stride_size(geometry)
  Packing.sampled_point(ready_input(ready),U32.to_nat(ready_x_base(ready)),cell_channel(ready,index,lane),
    Nat.add(job_tap_row(ready,index),Nat.mul(step_row(ready,step),stride)),Nat.add(job_tap_column(ready,index),Nat.mul(step_column(ready,step),stride)),
    U32.to_nat(geometry_height(geometry)),U32.to_nat(geometry_width(geometry)),U32.to_nat(geometry_padding(geometry)),U32.to_nat(geometry_groups(geometry)))

def taps_exact_of(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) -> {tap_count(geometry) == Nat.mul(channel_count(geometry),square_size(geometry)) : Nat}:
  GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned,+taps_exact,pixels,gradients} = domain
  taps_exact

def taps_room(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) ->
  {Nat.is_le(tap_count(geometry),U32.to_nat(Machine.capacity(domain_limits(geometry,domain)))) == True{} : Bool}:
  GeometryReady{+limits,columns,kernel,stride,square,taps,+outputs,+count,aligned,taps_exact,pixels,gradients} = domain
  Order.transitive(tap_count(geometry),Nat.mul(tap_count(geometry),output_count(geometry)),U32.to_nat(Machine.capacity(limits)),
    RowMajor.product_covers(tap_count(geometry),output_count(geometry),outputs),count)

def cell_tap_inside(+ready: BandReady,+index: Nat,+lane: Nat,+inside: JobInside<ready,index>,
  +channel_inside: {Nat.is_lt(cell_channel(ready,index,lane),channel_count(ready_geometry(ready))) == True{} : Bool}) ->
  {Nat.is_lt(cell_tap(ready,index,lane),tap_count(ready_geometry(ready))) == True{} : Bool}:
  JobInside{index_room,output_block,quotient_room,+tap,block,tap_row,tap_column,channels,outputs} = inside
  +geometry = ready_geometry(ready)
  %Equal.sym(Nat,tap_count(geometry),Nat.mul(channel_count(geometry),square_size(geometry)),taps_exact_of(geometry,ready_domain(ready))) :
    {Nat.is_lt(cell_tap(ready,index,lane),_) == True{} : Bool}
  RowMajor.radix(cell_channel(ready,index,lane),channel_count(geometry),job_tap(ready,index),square_size(geometry),channel_inside,tap)

def cell_tap_back(+ready: BandReady,+index: Nat,+lane: Nat,+inside: JobInside<ready,index>,
  +channel_inside: {Nat.is_lt(cell_channel(ready,index,lane),channel_count(ready_geometry(ready))) == True{} : Bool}) ->
  {cell_tap(ready,index,lane) == U32.to_nat(U32.from_nat(cell_tap(ready,index,lane))) : Nat}:
  +tap = cell_tap(ready,index,lane)
  +geometry = ready_geometry(ready)
  Equal.sym(Nat,U32.to_nat(U32.from_nat(tap)),tap,FullRange.roundtrip(tap,Machine.capacity(ready_limits(ready)),
    Order.transitive(tap,tap_count(geometry),ready_capacity(ready),Bounds.strict_le(tap,tap_count(geometry),cell_tap_inside(ready,index,lane,inside,channel_inside)),
      taps_room(geometry,ready_domain(ready)))))

# The tap of a row decodes to its channel, its tap row and its tap column.
def tap_channel_decode(+ready: BandReady,+index: Nat,+lane: Nat,+inside: JobInside<ready,index>) ->
  {Nat.div(cell_tap(ready,index,lane),square_size(ready_geometry(ready))) == cell_channel(ready,index,lane) : Nat}:
  JobInside{index_room,output_block,quotient_room,+tap,block,tap_row,tap_column,channels,outputs} = inside
  Radix.quotient(cell_channel(ready,index,lane),job_tap(ready,index),square_size(ready_geometry(ready)),square_positive(ready_geometry(ready),ready_domain(ready)),tap)

def tap_grouped(+ready: BandReady,+index: Nat,+lane: Nat) ->
  {cell_tap(ready,index,lane) == Nat.add(Nat.mul(Nat.mul(cell_channel(ready,index,lane),kernel_size(ready_geometry(ready))),kernel_size(ready_geometry(ready))),job_tap(ready,index)) : Nat}:
  +k = kernel_size(ready_geometry(ready))
  +channel = cell_channel(ready,index,lane)
  Equal.cong(Nat,Nat,head => Nat.add(head,job_tap(ready,index)),Nat.mul(channel,Nat.mul(k,k)),Nat.mul(Nat.mul(channel,k),k),
    Equal.sym(Nat,Nat.mul(Nat.mul(channel,k),k),Nat.mul(channel,Nat.mul(k,k)),RowMajor.assoc(channel,k,k)))

def tap_row_decode(+ready: BandReady,+index: Nat,+lane: Nat,+inside: JobInside<ready,index>) ->
  {Nat.mod(Nat.div(cell_tap(ready,index,lane),kernel_size(ready_geometry(ready))),kernel_size(ready_geometry(ready))) == job_tap_row(ready,index) : Nat}:
  JobInside{index_room,output_block,quotient_room,tap,block,+tap_row,tap_column,channels,outputs} = inside
  +k = kernel_size(ready_geometry(ready))
  +positive = kernel_positive(ready_geometry(ready),ready_domain(ready))
  +rows = Nat.mul(cell_channel(ready,index,lane),k)
  +quotient = Nat.div(job_tap(ready,index),k)
  +shifted = Equal.trans(Nat,Nat.div(cell_tap(ready,index,lane),k),Nat.div(Nat.add(Nat.mul(rows,k),job_tap(ready,index)),k),Nat.add(rows,quotient),
    Equal.cong(Nat,Nat,value => Nat.div(value,k),cell_tap(ready,index,lane),Nat.add(Nat.mul(rows,k),job_tap(ready,index)),tap_grouped(ready,index,lane)),
    DivisionOrder.quotient_shift(job_tap(ready,index),k,rows,positive))
  Equal.trans(Nat,Nat.mod(Nat.div(cell_tap(ready,index,lane),k),k),Nat.mod(Nat.add(rows,quotient),k),quotient,
    Equal.cong(Nat,Nat,value => Nat.mod(value,k),Nat.div(cell_tap(ready,index,lane),k),Nat.add(rows,quotient),shifted),
    Equal.trans(Nat,Nat.mod(Nat.add(Nat.mul(cell_channel(ready,index,lane),k),quotient),k),Nat.mod(quotient,k),quotient,
      DivisionOrder.remainder_shift(quotient,k,cell_channel(ready,index,lane),positive),
      Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(quotient,k),(0n,quotient),Division.small(quotient,k,tap_row))))

def tap_column_decode(+ready: BandReady,+index: Nat,+lane: Nat) ->
  {Nat.mod(cell_tap(ready,index,lane),kernel_size(ready_geometry(ready))) == job_tap_column(ready,index) : Nat}:
  +k = kernel_size(ready_geometry(ready))
  +positive = kernel_positive(ready_geometry(ready),ready_domain(ready))
  +rows = Nat.mul(cell_channel(ready,index,lane),k)
  Equal.trans(Nat,Nat.mod(cell_tap(ready,index,lane),k),Nat.mod(Nat.add(Nat.mul(rows,k),job_tap(ready,index)),k),job_tap_column(ready,index),
    Equal.cong(Nat,Nat,value => Nat.mod(value,k),cell_tap(ready,index,lane),Nat.add(Nat.mul(rows,k),job_tap(ready,index)),tap_grouped(ready,index,lane)),
    DivisionOrder.remainder_shift(job_tap(ready,index),k,rows,positive))

# The weight sample of a row at step t is the input pixel its tap reads.
def weight_value(+ready: BandReady,+index: Nat,+lane: Nat,+step: Nat,+inside: JobInside<ready,index>,+step_inside: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(cell_channel(ready,index,lane),channel_count(ready_geometry(ready))) == True{} : Bool}) ->
  {numeric_weight(numeric_source(ready),U32.from_nat(cell_tap(ready,index,lane)),step_position(ready,step)) == pixel_value(ready,index,lane,step) : F32}:
  BandReady{+geometry,+domain,+x_base,+d_base,+first,+count,+input,+gradient,+nonempty,+room} = ready
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +tap = cell_tap(ready,index,lane)
  +channel = cell_channel(ready,index,lane)
  +kn = U32.to_nat(k)
  +sn = U32.to_nat(s)
  +hn = U32.to_nat(h)
  +wn = U32.to_nat(w)
  +pn = U32.to_nat(p)
  +xn = U32.to_nat(x_base)
  +gn = U32.to_nat(((c + 7) / 8 : U32))
  +y = step_row(ready,step)
  +x = step_column(ready,step)
  +a = job_tap_row(ready,index)
  +b = job_tap_column(ready,index)
  +wanted = pixel_value(ready,index,lane,step)
  %cell_tap_back(ready,index,lane,inside,channel_inside) :
    {Packing.sampled_point(input,xn,Nat.div(_,Nat.mul(kn,kn)),Nat.add(Nat.mod(Nat.div(_,kn),kn),Nat.mul(U32.to_nat(U32.from_nat(y)),sn)),
      Nat.add(Nat.mod(_,kn),Nat.mul(U32.to_nat(U32.from_nat(x)),sn)),hn,wn,pn,gn) == wanted : F32}
  %Equal.sym(Nat,Nat.div(tap,Nat.mul(kn,kn)),channel,tap_channel_decode(ready,index,lane,inside)) :
    {Packing.sampled_point(input,xn,_,Nat.add(Nat.mod(Nat.div(tap,kn),kn),Nat.mul(U32.to_nat(U32.from_nat(y)),sn)),
      Nat.add(Nat.mod(tap,kn),Nat.mul(U32.to_nat(U32.from_nat(x)),sn)),hn,wn,pn,gn) == wanted : F32}
  %Equal.sym(Nat,Nat.mod(Nat.div(tap,kn),kn),a,tap_row_decode(ready,index,lane,inside)) :
    {Packing.sampled_point(input,xn,channel,Nat.add(_,Nat.mul(U32.to_nat(U32.from_nat(y)),sn)),
      Nat.add(Nat.mod(tap,kn),Nat.mul(U32.to_nat(U32.from_nat(x)),sn)),hn,wn,pn,gn) == wanted : F32}
  %Equal.sym(Nat,Nat.mod(tap,kn),b,tap_column_decode(ready,index,lane)) :
    {Packing.sampled_point(input,xn,channel,Nat.add(a,Nat.mul(U32.to_nat(U32.from_nat(y)),sn)),
      Nat.add(_,Nat.mul(U32.to_nat(U32.from_nat(x)),sn)),hn,wn,pn,gn) == wanted : F32}
  %step_row_back(ready,step,step_inside) :
    {Packing.sampled_point(input,xn,channel,Nat.add(a,Nat.mul(_,sn)),Nat.add(b,Nat.mul(U32.to_nat(U32.from_nat(x)),sn)),hn,wn,pn,gn) == wanted : F32}
  %step_column_back(ready,step) :
    {Packing.sampled_point(input,xn,channel,Nat.add(a,Nat.mul(y,sn)),Nat.add(b,Nat.mul(_,sn)),hn,wn,pn,gn) == wanted : F32}
  {==}

# ---- The pixel tile of a step holds its pixels ----

def word_back(+limits: Machine.Limits,+value: Nat,+room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {value == U32.to_nat(U32.from_nat(value)) : Nat}:
  Equal.sym(Nat,U32.to_nat(U32.from_nat(value)),value,FullRange.roundtrip(value,Machine.capacity(limits),room))

# The plane of a job is its channel block and its tap phases.
def phase_row(+ready: BandReady,+index: Nat) -> Nat:
  Nat.mod(job_tap_row(ready,index),phase_count(ready_geometry(ready)))

def phase_column(+ready: BandReady,+index: Nat) -> Nat:
  Nat.mod(job_tap_column(ready,index),phase_count(ready_geometry(ready)))

def plane_inner(+ready: BandReady,+index: Nat) -> Nat:
  Nat.add(Nat.mul(job_block(ready,index),phase_count(ready_geometry(ready))),phase_row(ready,index))

def phase_row_inside(+ready: BandReady,+index: Nat) -> {Nat.is_lt(phase_row(ready,index),phase_count(ready_geometry(ready))) == True{} : Bool}:
  DivisionOrder.remainder_small(job_tap_row(ready,index),phase_count(ready_geometry(ready)),phases_positive(ready_geometry(ready),ready_domain(ready)))

def phase_column_inside(+ready: BandReady,+index: Nat) -> {Nat.is_lt(phase_column(ready,index),phase_count(ready_geometry(ready))) == True{} : Bool}:
  DivisionOrder.remainder_small(job_tap_column(ready,index),phase_count(ready_geometry(ready)),phases_positive(ready_geometry(ready),ready_domain(ready)))

def inner_inside(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {Nat.is_lt(plane_inner(ready,index),Nat.mul(channel_blocks(ready_geometry(ready)),phase_count(ready_geometry(ready)))) == True{} : Bool}:
  JobInside{index_room,output_block,quotient_room,tap,+block,tap_row,tap_column,channels,outputs} = inside
  RowMajor.radix(job_block(ready,index),channel_blocks(ready_geometry(ready)),phase_row(ready,index),phase_count(ready_geometry(ready)),block,phase_row_inside(ready,index))

def plane_inside(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) -> {Nat.is_lt(job_plane(ready,index),plane_count(ready_geometry(ready))) == True{} : Bool}:
  +geometry = ready_geometry(ready)
  RowMajor.radix(plane_inner(ready,index),Nat.mul(channel_blocks(geometry),phase_count(geometry)),phase_column(ready,index),phase_count(geometry),inner_inside(ready,index,inside),
    phase_column_inside(ready,index))

def plane_room(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) -> {Nat.is_le(job_plane(ready,index),ready_capacity(ready)) == True{} : Bool}:
  +planes = plane_count(ready_geometry(ready))
  Order.transitive(job_plane(ready,index),planes,ready_capacity(ready),Bounds.strict_le(job_plane(ready,index),planes,plane_inside(ready,index,inside)),planes_room(ready))

def inner_room(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) -> {Nat.is_le(plane_inner(ready,index),ready_capacity(ready)) == True{} : Bool}:
  +geometry = ready_geometry(ready)
  +rows = Nat.mul(channel_blocks(geometry),phase_count(geometry))
  Order.transitive(plane_inner(ready,index),rows,ready_capacity(ready),Bounds.strict_le(plane_inner(ready,index),rows,inner_inside(ready,index,inside)),
    Order.transitive(rows,plane_count(geometry),ready_capacity(ready),RowMajor.product_covers(rows,phase_count(geometry),phases_positive(geometry,ready_domain(ready))),planes_room(ready)))

def plane_quotient(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {(U32.from_nat(job_plane(ready,index)) / U32.from_nat(phase_count(ready_geometry(ready))) : U32) == U32.from_nat(plane_inner(ready,index)) : U32}:
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  +phases = phase_count(geometry)
  +plane = job_plane(ready,index)
  Equal.trans(U32,(U32.from_nat(plane) / U32.from_nat(phases) : U32),U32.from_nat(Nat.div(plane,phases)),U32.from_nat(plane_inner(ready,index)),
    word_quotient(ready_limits(ready),plane,phases,plane_room(ready,index,inside),phases_room(geometry,domain),phases_positive(geometry,domain)),
    Equal.cong(Nat,U32,U32.from_nat,Nat.div(plane,phases),plane_inner(ready,index),
      Radix.quotient(plane_inner(ready,index),phase_column(ready,index),phases,phases_positive(geometry,domain),phase_column_inside(ready,index))))

def plane_block_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {(U32.from_nat(job_plane(ready,index)) / U32.from_nat(phase_count(ready_geometry(ready))) / U32.from_nat(phase_count(ready_geometry(ready))) : U32) ==
    U32.from_nat(job_block(ready,index)) : U32}:
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  +phases = phase_count(geometry)
  +ph = U32.from_nat(phases)
  +inner = plane_inner(ready,index)
  Equal.trans(U32,(U32.from_nat(job_plane(ready,index)) / ph / ph : U32),(U32.from_nat(inner) / ph : U32),U32.from_nat(job_block(ready,index)),
    Equal.cong(U32,U32,value => (value / ph : U32),(U32.from_nat(job_plane(ready,index)) / ph : U32),U32.from_nat(inner),plane_quotient(ready,index,inside)),
    Equal.trans(U32,(U32.from_nat(inner) / ph : U32),U32.from_nat(Nat.div(inner,phases)),U32.from_nat(job_block(ready,index)),
      word_quotient(ready_limits(ready),inner,phases,inner_room(ready,index,inside),phases_room(geometry,domain),phases_positive(geometry,domain)),
      Equal.cong(Nat,U32,U32.from_nat,Nat.div(inner,phases),job_block(ready,index),
        Radix.quotient(job_block(ready,index),phase_row(ready,index),phases,phases_positive(geometry,domain),phase_row_inside(ready,index)))))

def plane_row_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {(U32.from_nat(job_plane(ready,index)) / U32.from_nat(phase_count(ready_geometry(ready))) % U32.from_nat(phase_count(ready_geometry(ready))) : U32) ==
    U32.from_nat(phase_row(ready,index)) : U32}:
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  +phases = phase_count(geometry)
  +ph = U32.from_nat(phases)
  +inner = plane_inner(ready,index)
  Equal.trans(U32,(U32.from_nat(job_plane(ready,index)) / ph % ph : U32),(U32.from_nat(inner) % ph : U32),U32.from_nat(phase_row(ready,index)),
    Equal.cong(U32,U32,value => (value % ph : U32),(U32.from_nat(job_plane(ready,index)) / ph : U32),U32.from_nat(inner),plane_quotient(ready,index,inside)),
    Equal.trans(U32,(U32.from_nat(inner) % ph : U32),U32.from_nat(Nat.mod(inner,phases)),U32.from_nat(phase_row(ready,index)),
      word_remainder(ready_limits(ready),inner,phases,inner_room(ready,index,inside),phases_room(geometry,domain),phases_positive(geometry,domain)),
      Equal.cong(Nat,U32,U32.from_nat,Nat.mod(inner,phases),phase_row(ready,index),
        Radix.remainder(job_block(ready,index),phase_row(ready,index),phases,phases_positive(geometry,domain),phase_row_inside(ready,index)))))

def plane_column_word(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>) ->
  {(U32.from_nat(job_plane(ready,index)) % U32.from_nat(phase_count(ready_geometry(ready))) : U32) == U32.from_nat(phase_column(ready,index)) : U32}:
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  +phases = phase_count(geometry)
  +plane = job_plane(ready,index)
  Equal.trans(U32,(U32.from_nat(plane) % U32.from_nat(phases) : U32),U32.from_nat(Nat.mod(plane,phases)),U32.from_nat(phase_column(ready,index)),
    word_remainder(ready_limits(ready),plane,phases,plane_room(ready,index,inside),phases_room(geometry,domain),phases_positive(geometry,domain)),
    Equal.cong(Nat,U32,U32.from_nat,Nat.mod(plane,phases),phase_column(ready,index),
      Radix.remainder(plane_inner(ready,index),phase_column(ready,index),phases,phases_positive(geometry,domain),phase_column_inside(ready,index))))

def rows_end_room(+ready: BandReady) -> {Nat.is_le(Nat.add(first_row(ready),band_rows(ready)),ready_capacity(ready)) == True{} : Bool}:
  +tall = Nat.add(row_count(ready_geometry(ready)),reach(ready_geometry(ready)))
  Order.transitive(Nat.add(first_row(ready),band_rows(ready)),tall,ready_capacity(ready),rows_end_within(ready),
    Order.transitive(tall,Nat.mul(tall,band_columns(ready)),ready_capacity(ready),RowMajor.product_covers(tall,band_columns(ready),band_columns_positive(ready)),plane_area_room(ready)))

def phases_word_room(+ready: BandReady,+value: Nat,+below: {Nat.is_lt(value,phase_count(ready_geometry(ready))) == True{} : Bool}) ->
  {Nat.is_le(value,ready_capacity(ready)) == True{} : Bool}:
  +geometry = ready_geometry(ready)
  Order.transitive(value,phase_count(geometry),ready_capacity(ready),Bounds.strict_le(value,phase_count(geometry),below),phases_room(geometry,ready_domain(ready)))

# Lane r of the pixel tile at step t of a job is the pixel its row reads.
def tile_value(+ready: BandReady,+index: Nat,+step: Nat,+lane: Nat,+inside: JobInside<ready,index>,+step_inside: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool},
  +small: {Nat.is_lt(lane,8n) == True{} : Bool},+channel_inside: {Nat.is_lt(cell_channel(ready,index,lane),channel_count(ready_geometry(ready))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,pixel_tree(ready),GemmSpec.grid_read(job_grid(ready,index),step))) == pixel_value(ready,index,lane,step) : F32}:
  BandReady{+geometry,+domain,+x_base,+d_base,+first,+count,+input,+gradient,+nonempty,+room} = ready
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  JobInside{index_room,output_block,quotient_room,tap,+block_inside,+tap_row,+tap_column,channels,outputs} = inside
  +limits = ready_limits(ready)
  +phases = phase_count(geometry)
  +ph = U32.from_nat(phases)
  +plane = job_plane(ready,index)
  +area = band_area(ready)
  +tile = tile_index(ready,index,step)
  +band = band_columns(ready)
  +fr = first_row(ready)
  +br = band_rows(ready)
  +row = sheet_row(ready,index,step)
  +column = sheet_column(ready,index,step)
  +blocks = job_block(ready,index)
  +am = phase_row(ready,index)
  +bm = phase_column(ready,index)
  +a = job_tap_row(ready,index)
  +b = job_tap_column(ready,index)
  +y = step_row(ready,step)
  +x = step_column(ready,step)
  +rows_end = Nat.add(fr,br)
  +context = pixel_planes(ready)
  +depth = Storage.depth_for(U32.from_nat(pixel_count(ready)))
  +fresh_room = pixel_count_room(ready)
  +inside_tile = tile_index_inside(ready,index,step,inside,step_inside)
  +planes = plane_count(geometry)
  +capacity_room = {%RowMajor.assoc(planes,br,band) : {Nat.is_le(_,U32.to_nat(Tree.capacity(Tile.Tile8,pixel_fresh(ready)))) == True{} : Bool}
    fresh_capacity(limits,pixel_count(ready),fresh_room) : {Nat.is_le(Nat.mul(planes,area),U32.to_nat(Tree.capacity(Tile.Tile8,pixel_fresh(ready)))) == True{} : Bool}}
  +tiles = Packing.blocks_read(~Packing.Planes,~Packing.planes_row,~Packing.planes_width,~Packing.planes_length,planes,context,plane,tile,depth,pixel_fresh(ready),
    Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)),allocation_depth_limit(U32.from_nat(pixel_count(ready)),fresh_sized(limits,pixel_count(ready),fresh_room)),
    capacity_room,plane_inside(ready,index,inside),inside_tile)
  +listed = ListRead.enumerate_at(~Tile.Tile8,~IndexedValues.Sampled<Buffer.Model<Tile.Tile8>,Kernel.Plane>,
    ~IndexedValues.sampled(~Buffer.Model<Tile.Tile8>,~Tile.Tile8,~Kernel.Plane,~Packing.plane_sample),Tile.filled(0.0),area,tile,Nat.mul(plane,area),
    Packing.plane_samples(input,natural_pixels(ready),plane),inside_tile)
  +band_exact = FullRange.roundtrip(band,Machine.capacity(limits),band_columns_room(ready))
  +band_positive = band_columns_positive(ready)
  +column_inside = sheet_column_inside(ready,index,step,inside)
  +rows_end_back = word_back(limits,rows_end,rows_end_room(ready))
  +row_inside = sheet_row_inside(ready,index,step,inside,step_inside)
  +row_back = word_back(limits,row,Order.transitive(row,rows_end,ready_capacity(ready),Bounds.strict_le(row,rows_end,row_inside),rows_end_room(ready)))
  +column_back = word_back(limits,column,Order.transitive(column,band,ready_capacity(ready),Bounds.strict_le(column,band,column_inside),band_columns_room(ready)))
  +band_back = Equal.sym(Nat,U32.to_nat(U32.from_nat(band)),band,band_exact)
  +row_ok = {%row_back : {Nat.is_lt(_,U32.to_nat(U32.from_nat(rows_end))) == True{} : Bool} %rows_end_back : {Nat.is_lt(row,_) == True{} : Bool} row_inside :
    {Nat.is_lt(U32.to_nat(U32.from_nat(row)),U32.to_nat(U32.from_nat(rows_end))) == True{} : Bool}}
  +column_ok = {%column_back : {Nat.is_lt(_,U32.to_nat(U32.from_nat(band))) == True{} : Bool} %band_back : {Nat.is_lt(column,_) == True{} : Bool} column_inside :
    {Nat.is_lt(U32.to_nat(U32.from_nat(column)),U32.to_nat(U32.from_nat(band))) == True{} : Bool}}
  +blocks_room = Order.transitive(blocks,channel_blocks(geometry),ready_capacity(ready),Bounds.strict_le(blocks,channel_blocks(geometry),block_inside),
    Order.transitive(channel_blocks(geometry),Nat.mul(channel_blocks(geometry),phases),ready_capacity(ready),
      RowMajor.product_covers(channel_blocks(geometry),phases,phases_positive(geometry,domain)),
      Order.transitive(Nat.mul(channel_blocks(geometry),phases),planes,ready_capacity(ready),
        RowMajor.product_covers(Nat.mul(channel_blocks(geometry),phases),phases,phases_positive(geometry,domain)),planes_room(ready))))
  +blocks_back = word_back(limits,blocks,blocks_room)
  +am_back = word_back(limits,am,phases_word_room(ready,am,phase_row_inside(ready,index)))
  +bm_back = word_back(limits,bm,phases_word_room(ready,bm,phase_column_inside(ready,index)))
  +xn = U32.to_nat(x_base)
  +gn = U32.to_nat(((c + 7) / 8 : U32))
  +sn = U32.to_nat(s)
  +hn = U32.to_nat(h)
  +wn = U32.to_nat(w)
  +pn = U32.to_nat(p)
  +pw = U32.from_nat(plane)
  +fr_w = U32.from_nat(fr)
  +br_w = U32.from_nat(br)
  +band_w = U32.from_nat(band)
  +ends = (fr_w + br_w : U32)
  +row_w = U32.from_nat(row)
  +column_w = U32.from_nat(column)
  +blocks_w = U32.from_nat(blocks)
  +am_w = U32.from_nat(am)
  +bm_w = U32.from_nat(bm)
  +ends_w = U32.from_nat(rows_end)
  +start = Nat.mul(plane,area)
  +start_w = (pw * br_w * band_w : U32)
  +position_w = U32.from_nat(Nat.add(start,tile))
  +start_exact = Equal.trans(U32,start_w,(U32.from_nat(Nat.mul(plane,br)) * band_w : U32),U32.from_nat(start),
    Equal.cong(U32,U32,value => (value * band_w : U32),(pw * br_w : U32),U32.from_nat(Nat.mul(plane,br)),Multiplication.native_encoded_multiply(plane,br)),
    Equal.trans(U32,(U32.from_nat(Nat.mul(plane,br)) * band_w : U32),U32.from_nat(Nat.mul(Nat.mul(plane,br),band)),U32.from_nat(start),
      Multiplication.native_encoded_multiply(Nat.mul(plane,br),band),Equal.cong(Nat,U32,U32.from_nat,Nat.mul(Nat.mul(plane,br),band),start,RowMajor.assoc(plane,br,band))))
  +within = Equal.trans(U32,(position_w - start_w : U32),(position_w - U32.from_nat(start) : U32),U32.from_nat(tile),
    Equal.cong(U32,U32,value => (position_w - value : U32),start_w,U32.from_nat(start),start_exact),
    Equal.trans(U32,(position_w - U32.from_nat(start) : U32),((U32.from_nat(start) + U32.from_nat(tile) : U32) - U32.from_nat(start) : U32),U32.from_nat(tile),
      Equal.cong(U32,U32,value => (value - U32.from_nat(start) : U32),position_w,(U32.from_nat(start) + U32.from_nat(tile) : U32),
        Equal.sym(U32,(U32.from_nat(start) + U32.from_nat(tile) : U32),position_w,WordArithmetic.native_add(start,tile))),
      Arithmetic.subtract_first(U32.from_nat(start),U32.from_nat(tile))))
  +tile_room = Order.transitive(tile,area,ready_capacity(ready),Bounds.strict_le(tile,area,inside_tile),band_area_capacity(ready))
  +tile_row_n = tile_row(ready,index,step)
  +tile_column_n = tile_column(ready,index,step)
  +tile_column_small = tile_column_inside(ready,index,step,inside)
  +row_exact = Equal.trans(U32,(fr_w + (position_w - start_w : U32) / band_w : U32),(fr_w + U32.from_nat(tile) / band_w : U32),row_w,
    Equal.cong(U32,U32,value => (fr_w + value / band_w : U32),(position_w - start_w : U32),U32.from_nat(tile),within),
    Equal.trans(U32,(fr_w + U32.from_nat(tile) / band_w : U32),(fr_w + U32.from_nat(tile_row_n) : U32),row_w,
      Equal.cong(U32,U32,value => (fr_w + value : U32),(U32.from_nat(tile) / band_w : U32),U32.from_nat(tile_row_n),
        Equal.trans(U32,(U32.from_nat(tile) / band_w : U32),U32.from_nat(Nat.div(tile,band)),U32.from_nat(tile_row_n),
          word_quotient(limits,tile,band,tile_room,band_columns_room(ready),band_positive),
          Equal.cong(Nat,U32,U32.from_nat,Nat.div(tile,band),tile_row_n,Radix.quotient(tile_row_n,tile_column_n,band,band_positive,tile_column_small)))),
      Equal.trans(U32,(fr_w + U32.from_nat(tile_row_n) : U32),U32.from_nat(Nat.add(fr,tile_row_n)),row_w,WordArithmetic.native_add(fr,tile_row_n),
        Equal.cong(Nat,U32,U32.from_nat,Nat.add(fr,tile_row_n),row,Equal.sym(Nat,row,Nat.add(fr,tile_row_n),sheet_row_split(ready,index,step))))))
  +column_split = Equal.cong(Nat,Nat,value => Nat.add(value,tap_column_shift(ready,index)),x,grid_column(ready,step),step_column_split(ready,step))
  +column_exact = Equal.trans(U32,((position_w - start_w : U32) % band_w : U32),(U32.from_nat(tile) % band_w : U32),column_w,
    Equal.cong(U32,U32,value => (value % band_w : U32),(position_w - start_w : U32),U32.from_nat(tile),within),
    Equal.trans(U32,(U32.from_nat(tile) % band_w : U32),U32.from_nat(Nat.mod(tile,band)),column_w,
      word_remainder(limits,tile,band,tile_room,band_columns_room(ready),band_positive),
      Equal.cong(Nat,U32,U32.from_nat,Nat.mod(tile,band),column,
        Equal.trans(Nat,Nat.mod(tile,band),tile_column_n,column,Radix.remainder(tile_row_n,tile_column_n,band,band_positive,tile_column_small),
          Equal.sym(Nat,column,tile_column_n,column_split)))))
  +pixel = Packing.cell_pixel(input,blocks_w,am_w,bm_w,ends_w,band_w,geometry,x_base,row_w,column_w)
  +cell = Packing.plane_cell(input,blocks_w,am_w,bm_w,ends_w,band_w,geometry,x_base,fr_w,band_w,start_w,position_w,row_w,column_w,row_exact,column_exact,
    Packing.PixelReady{stride_positive(ready),row_ok,column_ok})
  +lanes = Packing.cell_pixel_lane(input,blocks_w,am_w,bm_w,ends_w,band_w,c,o,h,w,k,s,p,rows,columns,taps,x_base,row_w,column_w,lane,small)
  +wanted = pixel_value(ready,index,lane,step)
  %Equal.sym(U32,GemmSpec.grid_read(job_grid(ready,index),step),position_w,grid_read_exact(ready,index,step,step_inside)) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,pixel_tree(ready),_)) == wanted : F32}
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,pixel_tree(ready),position_w),ListRead.at(~Tile.Tile8,Tile.filled(0.0),tile,Packing.planes_row(context,plane)),tiles) :
    {TileLaws.select_lane(lane,_) == wanted : F32}
  %Equal.sym(Tile.Tile8,ListRead.at(~Tile.Tile8,Tile.filled(0.0),tile,Packing.planes_row(context,plane)),
      Packing.plane_sample(input,Kernel.plane_of(pw,natural_pixels(ready)),position_w),listed) :
    {TileLaws.select_lane(lane,_) == wanted : F32}
  %Equal.sym(U32,(pw / ph / ph : U32),blocks_w,plane_block_word(ready,index,inside)) :
    {TileLaws.select_lane(lane,Packing.plane_sample(input,Kernel.Plane{(x_base + _ : U32),((pw / ph % ph : U32) - p : U32),((pw % ph : U32) - p : U32),s,w,((c + 7) / 8 : U32),
      Kernel.axis_first((pw / ph % ph : U32),s,p),Kernel.axis_end((pw / ph % ph : U32),s,p,h,ends),Kernel.axis_first((pw % ph : U32),s,p),Kernel.axis_end((pw % ph : U32),s,p,w,band_w),
      fr_w,band_w,start_w},position_w)) == wanted : F32}
  %Equal.sym(U32,(pw / ph % ph : U32),am_w,plane_row_word(ready,index,inside)) :
    {TileLaws.select_lane(lane,Packing.plane_sample(input,Kernel.Plane{(x_base + blocks_w : U32),(_ - p : U32),((pw % ph : U32) - p : U32),s,w,((c + 7) / 8 : U32),
      Kernel.axis_first(_,s,p),Kernel.axis_end(_,s,p,h,ends),Kernel.axis_first((pw % ph : U32),s,p),Kernel.axis_end((pw % ph : U32),s,p,w,band_w),
      fr_w,band_w,start_w},position_w)) == wanted : F32}
  %Equal.sym(U32,(pw % ph : U32),bm_w,plane_column_word(ready,index,inside)) :
    {TileLaws.select_lane(lane,Packing.plane_sample(input,Kernel.Plane{(x_base + blocks_w : U32),(am_w - p : U32),(_ - p : U32),s,w,((c + 7) / 8 : U32),
      Kernel.axis_first(am_w,s,p),Kernel.axis_end(am_w,s,p,h,ends),Kernel.axis_first(_,s,p),Kernel.axis_end(_,s,p,w,band_w),
      fr_w,band_w,start_w},position_w)) == wanted : F32}
  %Equal.sym(U32,ends,ends_w,WordArithmetic.native_add(fr,br)) :
    {TileLaws.select_lane(lane,Packing.plane_sample(input,Kernel.Plane{(x_base + blocks_w : U32),(am_w - p : U32),(bm_w - p : U32),s,w,((c + 7) / 8 : U32),
      Kernel.axis_first(am_w,s,p),Kernel.axis_end(am_w,s,p,h,_),Kernel.axis_first(bm_w,s,p),Kernel.axis_end(bm_w,s,p,w,band_w),
      fr_w,band_w,start_w},position_w)) == wanted : F32}
  %Equal.sym(Tile.Tile8,Packing.plane_sample(input,Packing.block_plane(blocks_w,am_w,bm_w,ends_w,band_w,geometry,x_base,fr_w,band_w,start_w),position_w),
      Packing.numeric_cell(pixel),cell) :
    {TileLaws.select_lane(lane,_) == wanted : F32}
  %Equal.sym(F32,TileLaws.select_lane(lane,Packing.numeric_cell(pixel)),
      Packing.sampled_point(input,xn,Nat.add(Nat.mul(U32.to_nat(blocks_w),8n),lane),
        Nat.add(U32.to_nat(am_w),Nat.mul(U32.to_nat(row_w),sn)),Nat.add(U32.to_nat(bm_w),Nat.mul(U32.to_nat(column_w),sn)),hn,wn,pn,gn),lanes) :
    {_ == wanted : F32}
  %blocks_back :
    {Packing.sampled_point(input,xn,Nat.add(Nat.mul(_,8n),lane),Nat.add(U32.to_nat(am_w),Nat.mul(U32.to_nat(row_w),sn)),
      Nat.add(U32.to_nat(bm_w),Nat.mul(U32.to_nat(column_w),sn)),hn,wn,pn,gn) == wanted : F32}
  +channel = Nat.add(Nat.mul(blocks,8n),lane)
  %am_back :
    {Packing.sampled_point(input,xn,channel,Nat.add(_,Nat.mul(U32.to_nat(row_w),sn)),Nat.add(U32.to_nat(bm_w),Nat.mul(U32.to_nat(column_w),sn)),
      hn,wn,pn,gn) == wanted : F32}
  %row_back :
    {Packing.sampled_point(input,xn,channel,Nat.add(am,Nat.mul(_,sn)),Nat.add(U32.to_nat(bm_w),Nat.mul(U32.to_nat(column_w),sn)),
      hn,wn,pn,gn) == wanted : F32}
  %bm_back :
    {Packing.sampled_point(input,xn,channel,Nat.add(am,Nat.mul(row,sn)),Nat.add(_,Nat.mul(U32.to_nat(column_w),sn)),hn,wn,pn,gn) == wanted : F32}
  %column_back :
    {Packing.sampled_point(input,xn,channel,Nat.add(am,Nat.mul(row,sn)),Nat.add(bm,Nat.mul(_,sn)),hn,wn,pn,gn) == wanted : F32}
  %Equal.sym(Nat,Nat.add(am,Nat.mul(row,sn)),Nat.add(a,Nat.mul(y,sn)),Packing.phase_offset(a,y,sn,U32.to_nat(k),stride_positive(ready),tap_row)) :
    {Packing.sampled_point(input,xn,channel,_,Nat.add(bm,Nat.mul(column,sn)),hn,wn,pn,gn) == wanted : F32}
  %Equal.sym(Nat,Nat.add(bm,Nat.mul(column,sn)),Nat.add(b,Nat.mul(x,sn)),Packing.phase_offset(b,x,sn,U32.to_nat(k),stride_positive(ready),tap_column)) :
    {Packing.sampled_point(input,xn,channel,Nat.add(a,Nat.mul(y,sn)),_,hn,wn,pn,gn) == wanted : F32}
  {==}

# Lane r of the pixel tile at step t of a job is the weight sample of its tap.
def pixel_sample(+ready: BandReady,+index: Nat,+step: Nat,+lane: Nat,+inside: JobInside<ready,index>,+step_inside: {Nat.is_lt(step,ready_count(ready)) == True{} : Bool},
  +small: {Nat.is_lt(lane,8n) == True{} : Bool},+channel_inside: {Nat.is_lt(cell_channel(ready,index,lane),channel_count(ready_geometry(ready))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,pixel_tree(ready),GemmSpec.grid_read(job_grid(ready,index),step))) ==
    numeric_weight(numeric_source(ready),U32.from_nat(cell_tap(ready,index,lane)),step_position(ready,step)) : F32}:
  +weight = numeric_weight(numeric_source(ready),U32.from_nat(cell_tap(ready,index,lane)),step_position(ready,step))
  Equal.trans(F32,TileLaws.select_lane(lane,Tree.read(Tile.Tile8,pixel_tree(ready),GemmSpec.grid_read(job_grid(ready,index),step))),pixel_value(ready,index,lane,step),weight,
    tile_value(ready,index,step,lane,inside,step_inside,small,channel_inside),
    Equal.sym(F32,weight,pixel_value(ready,index,lane,step),weight_value(ready,index,lane,step,inside,step_inside,channel_inside)))

# ---- The value of a cell ----

def gemm_source(+ready: BandReady) -> GemmSpec.Source:
  GemmSpec.Source{pixel_tree(ready),gradient_tree(ready)}

# Lane j of a job is output 8 * output block + j.
def cell_output(+ready: BandReady,+index: Nat,+lane: Nat) -> Nat:
  Nat.add(Nat.mul(job_output_block(ready,index),8n),lane)

# The steps of cell (r, j) of a job.
type CellSteps is Data:
  CellSteps{ready: BandReady,index: Nat,row: Nat,lane: Nat,inside: JobInside<ready,index>,row_small: {Nat.is_lt(row,8n) == True{} : Bool},
    lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},channel_inside: {Nat.is_lt(cell_channel(ready,index,row),channel_count(ready_geometry(ready))) == True{} : Bool},
    output_inside: {Nat.is_lt(cell_output(ready,index,lane),output_count(ready_geometry(ready))) == True{} : Bool}}

def steps_count(context: CellSteps) -> Nat:
  CellSteps{ready,index,row,lane,inside,row_small,lane_small,channel_inside,output_inside} = context
  ready_count(ready)

# Step t of the cell reads the gradient and the pixel the numeric model
# reads at the position of step t.
def StepAgrees(+context: CellSteps,+step: Nat) -> Data:
  CellSteps{+ready,+index,+row,+lane,inside,row_small,lane_small,channel_inside,output_inside} = context
  Gemm.Both<
    {GemmSpec.gemm_input()(Gemm.Both{job_key(ready,index),gemm_source(ready)},lane,step) ==
      shifted_input()(Gemm.Both{ready_first(ready),numeric_source(ready)},U32.from_nat(cell_output(ready,index,lane)),step) : F32},
    {GemmSpec.gemm_weight()(Gemm.Both{job_key(ready,index),gemm_source(ready)},row,step) ==
      shifted_weight()(Gemm.Both{ready_first(ready),numeric_source(ready)},U32.from_nat(cell_tap(ready,index,row)),step) : F32}>

def step_agrees(+context: CellSteps,+step: Nat,+within: {Nat.is_lt(step,steps_count(context)) == True{} : Bool}) -> StepAgrees(context,step):
  CellSteps{+ready,+index,+row,+lane,+inside,+row_small,+lane_small,+channel_inside,+output_inside} = context
  JobInside{index_room,+output_block,quotient_room,tap,block,tap_row,tap_column,channels,outputs} = inside
  Gemm.Both{gradient_sample(ready,job_output_block(ready,index),step,lane,output_block,within,lane_small,output_inside),
    pixel_sample(ready,index,step,row,inside,within,row_small,channel_inside)}

# Cell (r, j) of a job is the numeric dot of its tap and output over the band.
def cell_dot(+ready: BandReady,+index: Nat,+row: Nat,+lane: Nat,+inside: JobInside<ready,index>,+row_small: {Nat.is_lt(row,8n) == True{} : Bool},
  +lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},+channel_inside: {Nat.is_lt(cell_channel(ready,index,row),channel_count(ready_geometry(ready))) == True{} : Bool},
  +output_inside: {Nat.is_lt(cell_output(ready,index,lane),output_count(ready_geometry(ready))) == True{} : Bool}) ->
  {GemmSpec.dot(GemmLaws.LaneFirst{},GemmSpec.taps(~Kernel.Cursor(),~Kernel.advance,ready_count(ready),Gemm.Both{U32.from_nat(Nat.mul(job_output_block(ready,index),ready_count(ready))),job_entries(ready,index)}),
    row,lane,gemm_source(ready),0.0) ==
    numeric_dot(numeric_source(ready),U32.from_nat(cell_tap(ready,index,row)),U32.from_nat(cell_output(ready,index,lane)),ready_count(ready),ready_first(ready),0.0) : F32}:
  +count = ready_count(ready)
  +first = ready_first(ready)
  +source = gemm_source(ready)
  +numeric = numeric_source(ready)
  +tap = U32.from_nat(cell_tap(ready,index,row))
  +output = U32.from_nat(cell_output(ready,index,lane))
  +key = job_key(ready,index)
  +keys = Ranges.range(count,0n)
  +left = {Gemm.Both{key,source} : Gemm.Both<Gemm.Both<Nat,GemmSpec.Grid>,GemmSpec.Source>}
  +right = {Gemm.Both{first,numeric} : Gemm.Both<Nat,NumericSource>}
  +wanted = numeric_dot(numeric,tap,output,count,first,0.0)
  +mapped = OrderedMatrix.dot(~Nat,~Nat,~Nat,~Gemm.Both<Gemm.Both<Nat,GemmSpec.Grid>,GemmSpec.Source>,~GemmSpec.gemm_input(),~GemmSpec.gemm_weight(),keys,row,lane,left,0.0)
  +shifted = OrderedMatrix.dot(~U32,~Nat,~U32,~Gemm.Both<Nat,NumericSource>,~shifted_input(),~shifted_weight(),keys,tap,output,right,0.0)
  +steps = {CellSteps{ready,index,row,lane,inside,row_small,lane_small,channel_inside,output_inside} : CellSteps}
  %Equal.sym(List<&2,ReductionLaws.AddressPair>,
      GemmSpec.taps(~Kernel.Cursor(),~Kernel.advance,count,Gemm.Both{U32.from_nat(Nat.mul(job_output_block(ready,index),count)),job_entries(ready,index)}),job_keys(ready,index),
      job_taps(ready,index)) :
    {GemmSpec.dot(GemmLaws.LaneFirst{},_,row,lane,source,0.0) == wanted : F32}
  Equal.trans(F32,GemmSpec.dot(GemmLaws.LaneFirst{},job_keys(ready,index),row,lane,source,0.0),mapped,wanted,
    Ordered.dot_reindexed(~Nat,~ReductionLaws.AddressPair,~Nat,~Nat,~GemmSpec.Source,~Gemm.Both<Nat,GemmSpec.Grid>,
      ~GemmSpec.both_key(~Nat,~Nat,~GemmSpec.Grid,~GemmSpec.offset_word,~GemmSpec.grid_read),~GemmSpec.lane_element,~GemmSpec.row_element,keys,key,row,lane,source,0.0),
    Equal.trans(F32,mapped,shifted,wanted,
      Ordered.dot_congruence(~Nat,~Nat,~Nat,~Gemm.Both<Gemm.Both<Nat,GemmSpec.Grid>,GemmSpec.Source>,~U32,~U32,~Gemm.Both<Nat,NumericSource>,
        ~GemmSpec.gemm_input(),~GemmSpec.gemm_weight(),~shifted_input(),~shifted_weight(),keys,row,lane,left,tap,output,right,0.0,
        Members.range_all(~CellSteps,~StepAgrees,~steps_count,~step_agrees,count,0n,steps,Order.reflexive(count))),
      Equal.sym(F32,wanted,shifted,numeric_dot_shifted(numeric,tap,output,count,first,0.0))))

# ---- The cells of a band ----

# Cell (tap, block) of the band's output: the tile of output block block at
# tap tap.
type Cell is Data:
  Cell{tap: Nat,block: Nat}

def CellInside(+ready: BandReady,+cell: Cell) -> Data:
  Cell{+tap,+block} = cell
  Gemm.Both<{Nat.is_lt(tap,tap_count(ready_geometry(ready))) == True{} : Bool},{Nat.is_lt(block,output_blocks(ready_geometry(ready))) == True{} : Bool}>

def cell_number(+ready: BandReady,+cell: Cell) -> Nat:
  Cell{+tap,+block} = cell
  Nat.add(Nat.mul(block,tap_count(ready_geometry(ready))),tap)

def cell_word(+ready: BandReady,+cell: Cell) -> U32:
  U32.from_nat(cell_number(ready,cell))

# The lanes of an output block below the outputs.
def block_lanes(+geometry: Kernel.Geometry,+block: Nat) -> U32:
  U32.min(8,U32.from_nat(Nat.sub(output_count(geometry),Nat.mul(block,8n))))

# Output 8 * block + lane at tap tap over the band.
def lane_dot(+ready: BandReady,+tap: Nat,+block: Nat,+lane: Nat) -> F32:
  numeric_dot(numeric_source(ready),U32.from_nat(tap),U32.from_nat(Nat.add(Nat.mul(block,8n),lane)),ready_count(ready),ready_first(ready),0.0)

def lane_dots(+ready: BandReady,+tap: Nat,+block: Nat) -> Tile.Tile8:
  Tile.Tile{lane_dot(ready,tap,block,0n),lane_dot(ready,tap,block,1n),lane_dot(ready,tap,block,2n),lane_dot(ready,tap,block,3n),lane_dot(ready,tap,block,4n),lane_dot(ready,tap,block,5n),lane_dot(ready,tap,block,6n),lane_dot(ready,tap,block,7n)}

# The tile of a cell: the dots of the outputs of its block, +0 past the
# outputs.
def cell_wanted(+ready: BandReady,+cell: Cell) -> Tile.Tile8:
  Cell{+tap,+block} = cell
  GemmLaws.finished_tile(~GemmLaws.unbiased_value,~Gemm.kept,0,block_lanes(ready_geometry(ready),block),no_bias(),lane_dots(ready,tap,block))

def cell_bound(+ready: BandReady) -> Nat:
  Nat.mul(output_blocks(ready_geometry(ready)),tap_count(ready_geometry(ready)))

def taps_positive(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) -> {Nat.is_lt(0n,tap_count(geometry)) == True{} : Bool}:
  GeometryReady{limits,columns,kernel,stride,square,+taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  taps

def cells_room(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>) ->
  {Nat.is_le(Nat.mul(output_blocks(geometry),tap_count(geometry)),U32.to_nat(Machine.capacity(domain_limits(geometry,domain)))) == True{} : Bool}:
  GeometryReady{+limits,+columns,+kernel,+stride,+square,+taps,+outputs,+count,+aligned,+taps_exact,+pixels,+gradients} = domain
  plan_cells_room(GroupPlan{limits,geometry,taps,outputs,count,aligned})

def number_below(+ready: BandReady,+cell: Cell,+valid: CellInside(ready,cell)) -> {Nat.is_lt(cell_number(ready,cell),cell_bound(ready)) == True{} : Bool}:
  Cell{+tap,+block} = cell
  Gemm.Both{+tap_inside,+block_inside} = valid
  RowMajor.radix(block,output_blocks(ready_geometry(ready)),tap,tap_count(ready_geometry(ready)),block_inside,tap_inside)

def number_back(+ready: BandReady,+cell: Cell,+valid: CellInside(ready,cell)) -> {U32.to_nat(cell_word(ready,cell)) == cell_number(ready,cell) : Nat}:
  +number = cell_number(ready,cell)
  +bound = cell_bound(ready)
  FullRange.roundtrip(number,Machine.capacity(ready_limits(ready)),
    Order.transitive(number,bound,ready_capacity(ready),Bounds.strict_le(number,bound,number_below(ready,cell,valid)),cells_room(ready_geometry(ready),ready_domain(ready))))

def cell_below(+ready: BandReady,+cell: Cell,+valid: CellInside(ready,cell)) -> {Nat.is_lt(U32.to_nat(cell_word(ready,cell)),cell_bound(ready)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(cell_word(ready,cell)),cell_number(ready,cell),number_back(ready,cell,valid)) : {Nat.is_lt(_,cell_bound(ready)) == True{} : Bool}
  number_below(ready,cell,valid)

# Two cells at one address are one cell.
def cell_same(+ready: BandReady,+cell: Cell,+other: Cell,+valid: CellInside(ready,cell),+other_valid: CellInside(ready,other),
  +equal: {cell_word(ready,cell) == cell_word(ready,other) : U32}) -> {cell_wanted(ready,cell) == cell_wanted(ready,other) : Tile.Tile8}:
  Cell{+tap,+block} = cell
  Cell{+other_tap,+other_block} = other
  Gemm.Both{+tap_inside,+block_inside} = valid
  Gemm.Both{+other_tap_inside,+other_block_inside} = other_valid
  +key = {Cell{tap,block} : Cell}
  +second = {Cell{other_tap,other_block} : Cell}
  +numbers = Equal.trans(Nat,cell_number(ready,key),U32.to_nat(cell_word(ready,key)),cell_number(ready,second),
    Equal.sym(Nat,U32.to_nat(cell_word(ready,key)),cell_number(ready,key),number_back(ready,key,Gemm.Both{tap_inside,block_inside})),
    Equal.trans(Nat,U32.to_nat(cell_word(ready,key)),U32.to_nat(cell_word(ready,second)),cell_number(ready,second),
      Equal.cong(U32,Nat,U32.to_nat,cell_word(ready,key),cell_word(ready,second),equal),number_back(ready,second,Gemm.Both{other_tap_inside,other_block_inside})))
  +width = tap_count(ready_geometry(ready))
  +positive = taps_positive(ready_geometry(ready),ready_domain(ready))
  %Radix.row_unique(block,other_block,width,tap,other_tap,positive,tap_inside,other_tap_inside,numbers) :
    {cell_wanted(ready,key) == cell_wanted(ready,Cell{other_tap,_}) : Tile.Tile8}
  %Radix.col_unique(block,other_block,width,tap,other_tap,positive,tap_inside,other_tap_inside,numbers) :
    {cell_wanted(ready,key) == cell_wanted(ready,Cell{_,block}) : Tile.Tile8}
  {==}

# ---- Every cell of a job is a cell of the band ----

def rows_small(+ready: BandReady,+index: Nat,+within: {Nat.is_lt(index,job_count(ready)) == True{} : Bool}) ->
  {Nat.is_le(GemmJobs.job_rows(~Kernel.Cursor(),Kernel.job(natural_plan(ready),U32.from_nat(index))),8n) == True{} : Bool}:
  %Equal.sym(Gemm.Job<Kernel.Cursor()>,Kernel.job(natural_plan(ready),U32.from_nat(index)),natural_job(ready,index),job_exact(ready,index,within)) :
    {Nat.is_le(GemmJobs.job_rows(~Kernel.Cursor(),_),8n) == True{} : Bool}
  NativeExtrema.minimum_below(8,U32.from_nat(Nat.sub(channel_count(ready_geometry(ready)),Nat.mul(job_block(ready,index),8n))))

# A job stores one tile for each row.
def columns_small(+ready: BandReady,+index: Nat,+within: {Nat.is_lt(index,job_count(ready)) == True{} : Bool}) ->
  {Nat.is_le(GemmJobs.job_columns(~Kernel.Cursor(),GemmLaws.Tiles{},Kernel.job(natural_plan(ready),U32.from_nat(index))),8n) == True{} : Bool}:
  {==}

# Part p of block b below min(8, total - 8 * b) is element 8 * b + p below total.
def part_left_back(+limits: Machine.Limits,+total: Nat,+block: Nat,+total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(Nat.sub(total,Nat.mul(block,8n)))) == Nat.sub(total,Nat.mul(block,8n)) : Nat}:
  +left = Nat.sub(total,Nat.mul(block,8n))
  FullRange.roundtrip(left,Machine.capacity(limits),Order.transitive(left,total,U32.to_nat(Machine.capacity(limits)),Order.sub_le(total,Nat.mul(block,8n)),total_room))

def part_inside(+limits: Machine.Limits,+total: Nat,+block: Nat,+part: Nat,+total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +block_inside: {Nat.is_lt(Nat.mul(block,8n),total) == True{} : Bool},
  +clipped: {Nat.is_lt(part,U32.to_nat(U32.min(8,U32.from_nat(Nat.sub(total,Nat.mul(block,8n)))))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(block,8n),part),total) == True{} : Bool}:
  +first = Nat.mul(block,8n)
  +left = Nat.sub(total,first)
  +below = {%part_left_back(limits,total,block,total_room) : {Nat.is_lt(part,_) == True{} : Bool}
    ViewWrites.clipped_below(part,8,U32.from_nat(left),clipped) : {Nat.is_lt(part,left) == True{} : Bool}}
  %Order.restore_shift(first,total,Bounds.strict_le(first,total,block_inside)) : {Nat.is_lt(Nat.add(first,part),_) == True{} : Bool}
  Equal.trans(Bool,Nat.is_lt(Nat.add(first,part),Nat.add(first,left)),Nat.is_lt(part,left),True{},Order.lt_offset(first,part,left),below)

def part_clipped(+limits: Machine.Limits,+total: Nat,+block: Nat,+part: Nat,+total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +small: {Nat.is_lt(part,8n) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(Nat.mul(block,8n),part),total) == True{} : Bool}) ->
  {Nat.is_lt(part,U32.to_nat(U32.min(8,U32.from_nat(Nat.sub(total,Nat.mul(block,8n)))))) == True{} : Bool}:
  +first = Nat.mul(block,8n)
  +left = Nat.sub(total,first)
  +lower = Bounds.strict_le(first,total,Order.le_lt_trans(first,Nat.add(first,part),total,Order.le_add(first,part),inside))
  +shifted = {%Equal.sym(Nat,Nat.add(first,left),total,Order.restore_shift(first,total,lower)) : {Nat.is_lt(Nat.add(first,part),_) == True{} : Bool}
    inside : {Nat.is_lt(Nat.add(first,part),Nat.add(first,left)) == True{} : Bool}}
  +below = Equal.trans(Bool,Nat.is_lt(part,left),Nat.is_lt(Nat.add(first,part),Nat.add(first,left)),True{},
    Equal.sym(Bool,Nat.is_lt(Nat.add(first,part),Nat.add(first,left)),Nat.is_lt(part,left),Order.lt_offset(first,part,left)),shifted)
  ViewWrites.clipped(part,8,U32.from_nat(left),small,
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(left)),left,part_left_back(limits,total,block,total_room)) : {Nat.is_lt(part,_) == True{} : Bool}
      below : {Nat.is_lt(part,U32.to_nat(U32.from_nat(left))) == True{} : Bool}})

# The tile of row r of a job at the cell of its first channel plus r rows of
# square_size: the cell of its channel's tap in its output block.
def tile_address(+b: U32,+c: U32,+a: U32,+r: U32,+sw: U32,+k: U32) ->
  {(((b * c + a : U32) * sw + k : U32) + r * sw : U32) == (b * c * sw + ((a + r : U32) * sw + k : U32) : U32) : U32}:
  +p = (b * c * sw : U32)
  +left = (a * sw : U32)
  +right = (r * sw : U32)
  Equal.trans(U32,(((b * c + a : U32) * sw + k : U32) + right : U32),(((b * c + a : U32) * sw + right : U32) + k : U32),(p + ((a + r : U32) * sw + k : U32) : U32),
    Arithmetic.add_swap(((b * c + a : U32) * sw : U32),k,right),
    Equal.trans(U32,(((b * c + a : U32) * sw + right : U32) + k : U32),(((p + left : U32) + right : U32) + k : U32),(p + ((a + r : U32) * sw + k : U32) : U32),
      Equal.cong(U32,U32,head => ((head + right : U32) + k : U32),((b * c + a : U32) * sw : U32),(p + left : U32),Arithmetic.multiply_additive((b * c : U32),a,sw)),
      Equal.trans(U32,(((p + left : U32) + right : U32) + k : U32),((p + (left + right : U32) : U32) + k : U32),(p + ((a + r : U32) * sw + k : U32) : U32),
        Equal.cong(U32,U32,head => (head + k : U32),((p + left : U32) + right : U32),(p + (left + right : U32) : U32),Arithmetic.add_associative(p,left,right)),
        Equal.trans(U32,((p + (left + right : U32) : U32) + k : U32),(p + ((left + right : U32) + k : U32) : U32),(p + ((a + r : U32) * sw + k : U32) : U32),
          Arithmetic.add_associative(p,(left + right : U32),k),
          Equal.cong(U32,U32,sum => (p + (sum + k : U32) : U32),(left + right : U32),((a + r : U32) * sw : U32),
            Equal.sym(U32,((a + r : U32) * sw : U32),(left + right : U32),Arithmetic.multiply_additive(a,r,sw)))))))

def natural_address(+ready: BandReady,+index: Nat,+row: Nat) ->
  {GemmJobs.cell_address(~Kernel.Cursor(),natural_job(ready,index),row,0n) == cell_word(ready,Cell{cell_tap(ready,index,row),job_output_block(ready,index)}) : U32}:
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  +sq = square_size(geometry)
  +cn = channel_count(geometry)
  +ob = job_output_block(ready,index)
  +rb8 = Nat.mul(job_block(ready,index),8n)
  +kt = job_tap(ready,index)
  +channel = cell_channel(ready,index,row)
  +tap = cell_tap(ready,index,row)
  +b = U32.from_nat(ob)
  +c = U32.from_nat(cn)
  +a = U32.from_nat(rb8)
  +r = U32.from_nat(row)
  +sw = U32.from_nat(sq)
  +k = U32.from_nat(kt)
  +base = U32.from_nat(job_store_base(ready,index))
  +target = (b * c * sw + ((a + r : U32) * sw + k : U32) : U32)
  +word = {
    %Equal.sym(Nat,tap_count(geometry),Nat.mul(cn,sq),taps_exact_of(geometry,domain)) :
      {U32.from_nat(Nat.add(Nat.mul(ob,_),tap)) == target : U32}
    %WordArithmetic.native_add(Nat.mul(ob,Nat.mul(cn,sq)),tap) : {_ == target : U32}
    %Multiplication.native_encoded_multiply(ob,Nat.mul(cn,sq)) : {(_ + U32.from_nat(tap) : U32) == target : U32}
    %Multiplication.native_encoded_multiply(cn,sq) : {(b * _ + U32.from_nat(tap) : U32) == target : U32}
    %Arithmetic.multiply_associative(b,c,sw) : {(_ + U32.from_nat(tap) : U32) == target : U32}
    %Encoded.encoded_madd(channel,sq,kt) : {(b * c * sw + _ : U32) == target : U32}
    %WordArithmetic.native_add(rb8,row) : {(b * c * sw + (_ * sw + k : U32) : U32) == target : U32}
    {==} : {cell_word(ready,Cell{tap,ob}) == target : U32}}
  +base_exact = Equal.trans(U32,base,(U32.from_nat(Nat.add(Nat.mul(ob,cn),rb8)) * sw + k : U32),((b * c + a : U32) * sw + k : U32),
    Equal.sym(U32,(U32.from_nat(Nat.add(Nat.mul(ob,cn),rb8)) * sw + k : U32),base,Encoded.encoded_madd(Nat.add(Nat.mul(ob,cn),rb8),sq,kt)),
    Equal.cong(U32,U32,head => (head * sw + k : U32),U32.from_nat(Nat.add(Nat.mul(ob,cn),rb8)),(b * c + a : U32),Equal.sym(U32,(b * c + a : U32),U32.from_nat(Nat.add(Nat.mul(ob,cn),rb8)),Encoded.encoded_madd(ob,cn,rb8))))
  +stored = Equal.trans(U32,((base + r * sw : U32) + 0 : U32),(base + r * sw : U32),target,Arithmetic.add_zero((base + r * sw : U32)),
    Equal.trans(U32,(base + r * sw : U32),(((b * c + a : U32) * sw + k : U32) + r * sw : U32),target,
      Equal.cong(U32,U32,head => (head + r * sw : U32),base,((b * c + a : U32) * sw + k : U32),base_exact),tile_address(b,c,a,r,sw,k)))
  Equal.trans(U32,((base + r * sw : U32) + 0 : U32),target,cell_word(ready,Cell{tap,ob}),stored,Equal.sym(U32,cell_word(ready,Cell{tap,ob}),target,word))

def CellAt(+ready: BandReady,+row: Nat,+job: Gemm.Job<Kernel.Cursor()>) -> Data:
  GemmJobs.Located(~Tile.Tile8,~BandReady,~Cell,~CellInside,~cell_word,~cell_wanted,ready,GemmJobs.cell_address(~Kernel.Cursor(),job,row,0n),
    GemmJobs.cell_value(~Kernel.Cursor(),~Kernel.advance,~GemmLaws.LaneFirst{},~GemmLaws.unbiased_value,~Gemm.kept,GemmLaws.Tiles{},job,natural_readers(ready),row,0n))

# The bias of an unbiased kept tile is never read.
def kept_first(+first: U32,+lanes: U32,+values: Tile.Tile8) ->
  {GemmLaws.finished_tile(~GemmLaws.unbiased_value,~Gemm.kept,first,lanes,no_bias(),values) == GemmLaws.finished_tile(~GemmLaws.unbiased_value,~Gemm.kept,0,lanes,no_bias(),values) : Tile.Tile8}:
  {==}

# Lane l below the lanes of a job's tile of row r is the dot of its output.
def lane_agrees(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>,+row: Nat,+row_small: {Nat.is_lt(row,8n) == True{} : Bool},
  +channel_inside: {Nat.is_lt(cell_channel(ready,index,row),channel_count(ready_geometry(ready))) == True{} : Bool},+lane: Nat,+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,U32.to_nat(block_lanes(ready_geometry(ready),job_output_block(ready,index)))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,GemmSpec.dot_row(GemmLaws.LaneFirst{},GemmSpec.taps(~Kernel.Cursor(),~Kernel.advance,ready_count(ready),
      Gemm.Both{U32.from_nat(Nat.mul(job_output_block(ready,index),ready_count(ready))),job_entries(ready,index)}),row,gemm_source(ready))) ==
    lane_dot(ready,cell_tap(ready,index,row),job_output_block(ready,index),lane) : F32}:
  JobInside{index_room,output_block,quotient_room,tap,block,tap_row,tap_column,channels,+outputs} = inside
  +geometry = ready_geometry(ready)
  +output_inside = part_inside(ready_limits(ready),output_count(geometry),job_output_block(ready,index),lane,outputs_room(geometry,ready_domain(ready)),outputs,lane_inside)
  Equal.trans(F32,TileLaws.select_lane(lane,GemmSpec.dot_row(GemmLaws.LaneFirst{},GemmSpec.taps(~Kernel.Cursor(),~Kernel.advance,ready_count(ready),
      Gemm.Both{U32.from_nat(Nat.mul(job_output_block(ready,index),ready_count(ready))),job_entries(ready,index)}),row,gemm_source(ready))),
    GemmSpec.dot(GemmLaws.LaneFirst{},GemmSpec.taps(~Kernel.Cursor(),~Kernel.advance,ready_count(ready),
      Gemm.Both{U32.from_nat(Nat.mul(job_output_block(ready,index),ready_count(ready))),job_entries(ready,index)}),row,lane,gemm_source(ready),0.0),
    lane_dot(ready,cell_tap(ready,index,row),job_output_block(ready,index),lane),
    GemmSpec.dot_lane(GemmLaws.LaneFirst{},GemmSpec.taps(~Kernel.Cursor(),~Kernel.advance,ready_count(ready),
      Gemm.Both{U32.from_nat(Nat.mul(job_output_block(ready,index),ready_count(ready))),job_entries(ready,index)}),row,lane,gemm_source(ready),lane_small),
    cell_dot(ready,index,row,lane,inside,row_small,lane_small,channel_inside,output_inside))

# The tile a job stores for row r is the tile of its cell.
def cell_tile(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>,+row: Nat,+row_small: {Nat.is_lt(row,8n) == True{} : Bool},
  +channel_inside: {Nat.is_lt(cell_channel(ready,index,row),channel_count(ready_geometry(ready))) == True{} : Bool}) ->
  {GemmJobs.cell_value(~Kernel.Cursor(),~Kernel.advance,~GemmLaws.LaneFirst{},~GemmLaws.unbiased_value,~Gemm.kept,GemmLaws.Tiles{},natural_job(ready,index),natural_readers(ready),row,0n) ==
    cell_wanted(ready,Cell{cell_tap(ready,index,row),job_output_block(ready,index)}) : Tile.Tile8}:
  +geometry = ready_geometry(ready)
  +lanes = block_lanes(geometry,job_output_block(ready,index))
  +values = GemmSpec.dot_row(GemmLaws.LaneFirst{},GemmSpec.taps(~Kernel.Cursor(),~Kernel.advance,ready_count(ready),
    Gemm.Both{U32.from_nat(Nat.mul(job_output_block(ready,index),ready_count(ready))),job_entries(ready,index)}),row,gemm_source(ready))
  +dots = lane_dots(ready,cell_tap(ready,index,row),job_output_block(ready,index))
  Equal.trans(Tile.Tile8,GemmLaws.finished_tile(~GemmLaws.unbiased_value,~Gemm.kept,U32.from_nat(Nat.mul(job_block(ready,index),8n)),lanes,no_bias(),values),
    GemmLaws.finished_tile(~GemmLaws.unbiased_value,~Gemm.kept,0,lanes,no_bias(),values),GemmLaws.finished_tile(~GemmLaws.unbiased_value,~Gemm.kept,0,lanes,no_bias(),dots),
    kept_first(U32.from_nat(Nat.mul(job_block(ready,index),8n)),lanes,values),
    GemmLaws.finished_agree(~GemmLaws.unbiased_value,~Gemm.kept,0,lanes,no_bias(),values,dots,
      lane_inside => lane_agrees(ready,index,inside,row,row_small,channel_inside,0n,{==},Bounds.as_nat_lt(0,lanes,lane_inside)),
      lane_inside => lane_agrees(ready,index,inside,row,row_small,channel_inside,1n,{==},Bounds.as_nat_lt(1,lanes,lane_inside)),
      lane_inside => lane_agrees(ready,index,inside,row,row_small,channel_inside,2n,{==},Bounds.as_nat_lt(2,lanes,lane_inside)),
      lane_inside => lane_agrees(ready,index,inside,row,row_small,channel_inside,3n,{==},Bounds.as_nat_lt(3,lanes,lane_inside)),
      lane_inside => lane_agrees(ready,index,inside,row,row_small,channel_inside,4n,{==},Bounds.as_nat_lt(4,lanes,lane_inside)),
      lane_inside => lane_agrees(ready,index,inside,row,row_small,channel_inside,5n,{==},Bounds.as_nat_lt(5,lanes,lane_inside)),
      lane_inside => lane_agrees(ready,index,inside,row,row_small,channel_inside,6n,{==},Bounds.as_nat_lt(6,lanes,lane_inside)),
      lane_inside => lane_agrees(ready,index,inside,row,row_small,channel_inside,7n,{==},Bounds.as_nat_lt(7,lanes,lane_inside))))

def natural_cell(+ready: BandReady,+index: Nat,+inside: JobInside<ready,index>,+row: Nat,
  +row_inside: {Nat.is_lt(row,GemmJobs.job_rows(~Kernel.Cursor(),natural_job(ready,index))) == True{} : Bool}) -> CellAt(ready,row,natural_job(ready,index)):
  JobInside{index_room,+output_block,quotient_room,tap,block,tap_row,tap_column,+channels,outputs} = inside
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  +limits = ready_limits(ready)
  +row_small = ViewWrites.clipped_small(row,8,U32.from_nat(Nat.sub(channel_count(geometry),Nat.mul(job_block(ready,index),8n))),row_inside)
  +channel_inside = part_inside(limits,channel_count(geometry),job_block(ready,index),row,channels_room(geometry,domain),channels,row_inside)
  GemmJobs.Keyed{Cell{cell_tap(ready,index,row),job_output_block(ready,index)},Gemm.Both{cell_tap_inside(ready,index,row,inside,channel_inside),output_block},
    natural_address(ready,index,row),cell_tile(ready,index,inside,row,row_small,channel_inside)}

def band_cell(+ready: BandReady,+index: Nat,+within: {Nat.is_lt(index,job_count(ready)) == True{} : Bool},+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,GemmJobs.job_rows(~Kernel.Cursor(),Kernel.job(natural_plan(ready),U32.from_nat(index)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,GemmJobs.job_columns(~Kernel.Cursor(),GemmLaws.Tiles{},Kernel.job(natural_plan(ready),U32.from_nat(index)))) == True{} : Bool}) ->
  GemmJobs.Located(~Tile.Tile8,~BandReady,~Cell,~CellInside,~cell_word,~cell_wanted,ready,GemmJobs.cell_address(~Kernel.Cursor(),Kernel.job(natural_plan(ready),U32.from_nat(index)),row,column),
    GemmJobs.cell_value(~Kernel.Cursor(),~Kernel.advance,~GemmLaws.LaneFirst{},~GemmLaws.unbiased_value,~Gemm.kept,GemmLaws.Tiles{},Kernel.job(natural_plan(ready),U32.from_nat(index)),natural_readers(ready),row,column)):
  +exact = job_exact(ready,index,within)
  +job = natural_job(ready,index)
  %Equal.sym(Nat,column,0n,GemmJobs.only_cell(column,column_inside)) :
    GemmJobs.Located(~Tile.Tile8,~BandReady,~Cell,~CellInside,~cell_word,~cell_wanted,ready,GemmJobs.cell_address(~Kernel.Cursor(),Kernel.job(natural_plan(ready),U32.from_nat(index)),row,_),
      GemmJobs.cell_value(~Kernel.Cursor(),~Kernel.advance,~GemmLaws.LaneFirst{},~GemmLaws.unbiased_value,~Gemm.kept,GemmLaws.Tiles{},Kernel.job(natural_plan(ready),U32.from_nat(index)),natural_readers(ready),row,_))
  %Equal.sym(Gemm.Job<Kernel.Cursor()>,Kernel.job(natural_plan(ready),U32.from_nat(index)),job,exact) : CellAt(ready,row,_)
  natural_cell(ready,index,job_inside(ready,index,within),row,
    {%exact : {Nat.is_lt(row,GemmJobs.job_rows(~Kernel.Cursor(),_)) == True{} : Bool} row_inside : {Nat.is_lt(row,GemmJobs.job_rows(~Kernel.Cursor(),job)) == True{} : Bool}})

# ---- Every cell of the band is a cell of a job ----

# Tap t is channel t / (k * k) at tap t % (k * k); channel c is row c % 8 of
# channel block c / 8.
def key_channel(+ready: BandReady,+tap: Nat) -> Nat:
  Nat.div(tap,square_size(ready_geometry(ready)))

def key_tap(+ready: BandReady,+tap: Nat) -> Nat:
  Nat.mod(tap,square_size(ready_geometry(ready)))

def key_block(+ready: BandReady,+tap: Nat) -> Nat:
  Nat.div(key_channel(ready,tap),8n)

def key_row(+ready: BandReady,+tap: Nat) -> Nat:
  Nat.mod(key_channel(ready,tap),8n)

def key_quotient(+ready: BandReady,+tap: Nat) -> Nat:
  Nat.add(Nat.mul(key_block(ready,tap),square_size(ready_geometry(ready))),key_tap(ready,tap))

def key_index(+ready: BandReady,+tap: Nat,+block: Nat) -> Nat:
  Nat.add(Nat.mul(key_quotient(ready,tap),output_blocks(ready_geometry(ready))),block)

def key_channel_inside(+ready: BandReady,+tap: Nat,+inside: {Nat.is_lt(tap,tap_count(ready_geometry(ready))) == True{} : Bool}) ->
  {Nat.is_lt(key_channel(ready,tap),channel_count(ready_geometry(ready))) == True{} : Bool}:
  +geometry = ready_geometry(ready)
  +domain = ready_domain(ready)
  DivisionOrder.quotient_below(tap,channel_count(geometry),square_size(geometry),square_positive(geometry,domain),
    {%taps_exact_of(geometry,domain) : {Nat.is_lt(tap,_) == True{} : Bool} inside : {Nat.is_lt(tap,Nat.mul(channel_count(geometry),square_size(geometry))) == True{} : Bool}})

def key_tap_inside(+ready: BandReady,+tap: Nat) -> {Nat.is_lt(key_tap(ready,tap),square_size(ready_geometry(ready))) == True{} : Bool}:
  DivisionOrder.remainder_small(tap,square_size(ready_geometry(ready)),square_positive(ready_geometry(ready),ready_domain(ready)))

def index_output_block(+ready: BandReady,+tap: Nat,+block: Nat,+inside: {Nat.is_lt(block,output_blocks(ready_geometry(ready))) == True{} : Bool}) ->
  {job_output_block(ready,key_index(ready,tap,block)) == block : Nat}:
  +geometry = ready_geometry(ready)
  Radix.remainder(key_quotient(ready,tap),block,output_blocks(geometry),output_blocks_positive(geometry,ready_domain(ready)),inside)

def index_quotient(+ready: BandReady,+tap: Nat,+block: Nat,+inside: {Nat.is_lt(block,output_blocks(ready_geometry(ready))) == True{} : Bool}) ->
  {job_quotient(ready,key_index(ready,tap,block)) == key_quotient(ready,tap) : Nat}:
  +geometry = ready_geometry(ready)
  Radix.quotient(key_quotient(ready,tap),block,output_blocks(geometry),output_blocks_positive(geometry,ready_domain(ready)),inside)

def index_tap(+ready: BandReady,+tap: Nat,+block: Nat,+inside: {Nat.is_lt(block,output_blocks(ready_geometry(ready))) == True{} : Bool}) ->
  {job_tap(ready,key_index(ready,tap,block)) == key_tap(ready,tap) : Nat}:
  +geometry = ready_geometry(ready)
  +sq = square_size(geometry)
  +index = key_index(ready,tap,block)
  Equal.trans(Nat,job_tap(ready,index),Nat.mod(key_quotient(ready,tap),sq),key_tap(ready,tap),
    Equal.cong(Nat,Nat,value => Nat.mod(value,sq),job_quotient(ready,index),key_quotient(ready,tap),index_quotient(ready,tap,block,inside)),
    Radix.remainder(key_block(ready,tap),key_tap(ready,tap),sq,square_positive(geometry,ready_domain(ready)),key_tap_inside(ready,tap)))

def index_block(+ready: BandReady,+tap: Nat,+block: Nat,+inside: {Nat.is_lt(block,output_blocks(ready_geometry(ready))) == True{} : Bool}) ->
  {job_block(ready,key_index(ready,tap,block)) == key_block(ready,tap) : Nat}:
  +geometry = ready_geometry(ready)
  +sq = square_size(geometry)
  +index = key_index(ready,tap,block)
  Equal.trans(Nat,job_block(ready,index),Nat.div(key_quotient(ready,tap),sq),key_block(ready,tap),
    Equal.cong(Nat,Nat,value => Nat.div(value,sq),job_quotient(ready,index),key_quotient(ready,tap),index_quotient(ready,tap,block,inside)),
    Radix.quotient(key_block(ready,tap),key_tap(ready,tap),sq,square_positive(geometry,ready_domain(ready)),key_tap_inside(ready,tap)))

def key_index_inside(+ready: BandReady,+tap: Nat,+block: Nat,+tap_inside: {Nat.is_lt(tap,tap_count(ready_geometry(ready))) == True{} : Bool},
  +block_inside: {Nat.is_lt(block,output_blocks(ready_geometry(ready))) == True{} : Bool}) -> {Nat.is_lt(key_index(ready,tap,block),job_count(ready)) == True{} : Bool}:
  +geometry = ready_geometry(ready)
  +blocks = channel_blocks(geometry)
  +k = kernel_size(geometry)
  +sq = square_size(geometry)
  +outs = output_blocks(geometry)
  +channel_block_inside = eight_block_inside(channel_count(geometry),key_channel(ready,tap),key_channel_inside(ready,tap,tap_inside))
  +below = RowMajor.radix(key_quotient(ready,tap),Nat.mul(blocks,sq),block,outs,
    RowMajor.radix(key_block(ready,tap),blocks,key_tap(ready,tap),sq,channel_block_inside,key_tap_inside(ready,tap)),block_inside)
  %Equal.sym(Nat,Nat.mul(Nat.mul(blocks,k),k),Nat.mul(blocks,sq),RowMajor.assoc(blocks,k,k)) : {Nat.is_lt(key_index(ready,tap,block),Nat.mul(_,outs)) == True{} : Bool}
  below

def key_tap_back(+ready: BandReady,+tap: Nat,+block: Nat,+inside: {Nat.is_lt(block,output_blocks(ready_geometry(ready))) == True{} : Bool}) ->
  {cell_tap(ready,key_index(ready,tap,block),key_row(ready,tap)) == tap : Nat}:
  +geometry = ready_geometry(ready)
  +sq = square_size(geometry)
  +index = key_index(ready,tap,block)
  +channel = key_channel(ready,tap)
  +row = key_row(ready,tap)
  %Equal.sym(Nat,job_block(ready,index),key_block(ready,tap),index_block(ready,tap,block,inside)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(_,8n),row),sq),job_tap(ready,index)) == tap : Nat}
  %Equal.sym(Nat,job_tap(ready,index),key_tap(ready,tap),index_tap(ready,tap,block,inside)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(key_block(ready,tap),8n),row),sq),_) == tap : Nat}
  %DivisionOrder.quotient_exact(channel,8n,{==}) : {Nat.add(Nat.mul(_,sq),key_tap(ready,tap)) == tap : Nat}
  Equal.sym(Nat,tap,Nat.add(Nat.mul(channel,sq),key_tap(ready,tap)),DivisionOrder.quotient_exact(tap,sq,square_positive(geometry,ready_domain(ready))))

def cell_found(+ready: BandReady,+cell: Cell,+valid: CellInside(ready,cell)) ->
  GemmJobs.Locate(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~GemmLaws.Tiles{},~BandReady,~natural_plan,~job_count,~Cell,~cell_word,ready,cell):
  Cell{+tap,+block} = cell
  Gemm.Both{+tap_inside,+block_inside} = valid
  +geometry = ready_geometry(ready)
  +limits = ready_limits(ready)
  +index = key_index(ready,tap,block)
  +row = key_row(ready,tap)
  +within = key_index_inside(ready,tap,block,tap_inside,block_inside)
  +exact = job_exact(ready,index,within)
  +job = natural_job(ready,index)
  +placed = Kernel.job(natural_plan(ready),U32.from_nat(index))
  +channel_split = {%DivisionOrder.quotient_exact(key_channel(ready,tap),8n,{==}) : {Nat.is_lt(_,channel_count(geometry)) == True{} : Bool}
    key_channel_inside(ready,tap,tap_inside) : {Nat.is_lt(Nat.add(Nat.mul(key_block(ready,tap),8n),row),channel_count(geometry)) == True{} : Bool}}
  +row_inside = {
    %Equal.sym(Gemm.Job<Kernel.Cursor()>,placed,job,exact) : {Nat.is_lt(row,GemmJobs.job_rows(~Kernel.Cursor(),_)) == True{} : Bool}
    %Equal.sym(Nat,job_block(ready,index),key_block(ready,tap),index_block(ready,tap,block,block_inside)) :
      {Nat.is_lt(row,U32.to_nat(U32.min(8,U32.from_nat(Nat.sub(channel_count(geometry),Nat.mul(_,8n)))))) == True{} : Bool}
    part_clipped(limits,channel_count(geometry),key_block(ready,tap),row,channels_room(geometry,ready_domain(ready)),
      DivisionOrder.remainder_small(key_channel(ready,tap),8n,{==}),channel_split) :
    {Nat.is_lt(row,GemmJobs.job_rows(~Kernel.Cursor(),placed)) == True{} : Bool}}
  +address = {
    %Equal.sym(Gemm.Job<Kernel.Cursor()>,placed,job,exact) : {GemmJobs.cell_address(~Kernel.Cursor(),_,row,0n) == cell_word(ready,Cell{tap,block}) : U32}
    %key_tap_back(ready,tap,block,block_inside) : {GemmJobs.cell_address(~Kernel.Cursor(),job,row,0n) == cell_word(ready,Cell{_,block}) : U32}
    %index_output_block(ready,tap,block,block_inside) : {GemmJobs.cell_address(~Kernel.Cursor(),job,row,0n) == cell_word(ready,Cell{cell_tap(ready,index,row),_}) : U32}
    natural_address(ready,index,row) : {GemmJobs.cell_address(~Kernel.Cursor(),placed,row,0n) == cell_word(ready,Cell{tap,block}) : U32}}
  GemmJobs.Found{index,row,0n,within,row_inside,{==},address}

# ---- The output of a band ----

# The output tree of a band is fresh at the depth of the geometry's plan.
def values_depth(+ready: BandReady) -> Nat:
  plan_depth(geometry_plan(ready_geometry(ready),ready_domain(ready)))

def values_balanced(+ready: BandReady) -> Shape.Balanced(Tile.Tile8,values_depth(ready),band_values(ready_band(ready))):
  BandReady{+geometry,+domain,+x_base,+d_base,+first,+count,+input,+gradient,+nonempty,+room} = ready
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  GeometryReady{+limits,+columns_positive,+kernel_positive,+stride_positive,+square,+taps_positive,+outputs_positive,+cells,+aligned,+taps_exact,+pixels,+gradients} = domain
  Shape.fresh(Tile.Tile8,values_depth(ready),Tile.filled(0.0))

def values_depth_ok(+ready: BandReady) -> {Nat.is_le(values_depth(ready),30n) == True{} : Bool}:
  plan_depth_limit(geometry_plan(ready_geometry(ready),ready_domain(ready)))

def values_room(+ready: BandReady) -> {Nat.is_le(cell_bound(ready),U32.to_nat(Tree.capacity(Tile.Tile8,band_values(ready_band(ready))))) == True{} : Bool}:
  BandReady{+geometry,+domain,+x_base,+d_base,+first,+count,+input,+gradient,+nonempty,+room} = ready
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  GeometryReady{+limits,+columns_positive,+kernel_positive,+stride_positive,+square,+taps_positive,+outputs_positive,+cells,+aligned,+taps_exact,+pixels,+gradients} = domain
  +plan = geometry_plan(geometry,domain)
  +depth = plan_depth(plan)
  +tree = Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))
  %Equal.sym(U32,Tree.capacity(Tile.Tile8,tree),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,tree,Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)))) : {Nat.is_le(cell_bound(ready),U32.to_nat(_)) == True{} : Bool}
  plan_physical_extent(plan)

# The tile of a cell of a band's output is the cell's wanted tile.
def band_cells(+ready: BandReady,+tap: Nat,+block: Nat,+tap_inside: {Nat.is_lt(tap,tap_count(ready_geometry(ready))) == True{} : Bool},
  +block_inside: {Nat.is_lt(block,output_blocks(ready_geometry(ready))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,group_output(band_group(ready_band(ready))),cell_word(ready,Cell{tap,block})) == cell_wanted(ready,Cell{tap,block}) : Tile.Tile8}:
  +key = {Cell{tap,block} : Cell}
  +tree = band_values(ready_band(ready))
  %Equal.sym(Tree.Tree<Tile.Tile8>,group_output(band_group(ready_band(ready))),Writes.pure(Tile.Tile8,natural_writes(ready),tree),band_output(ready)) :
    {Tree.read(Tile.Tile8,_,cell_word(ready,key)) == cell_wanted(ready,key) : Tile.Tile8}
  GemmJobs.readback(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.advance,~GemmLaws.LaneFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.kept,~BandReady,~natural_plan,~natural_readers,~job_count,
    ~Cell,~CellInside,~cell_word,~cell_wanted,~rows_small,~columns_small,~band_cell,~cell_bound,~cell_below,~cell_same,~cell_found,
    ready,key,Gemm.Both{tap_inside,block_inside},values_depth(ready),tree,values_balanced(ready),values_depth_ok(ready),values_room(ready))

# Lane lane of the dots of a block is the dot of output 8 * block + lane.
def dots_lane(+ready: BandReady,+tap: Nat,+block: Nat,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,lane_dots(ready,tap,block)) == lane_dot(ready,tap,block,lane) : F32}:
  match lane:
    case 0n: {==}
    case 1n+ +a:
      match a:
        case 0n: {==}
        case 1n+ +b:
          match b:
            case 0n: {==}
            case 1n+ +c:
              match c:
                case 0n: {==}
                case 1n+ +d:
                  match d:
                    case 0n: {==}
                    case 1n+ +e:
                      match e:
                        case 0n: {==}
                        case 1n+ +f:
                          match f:
                            case 0n: {==}
                            case 1n+ +g:
                              match g:
                                case 0n: {==}
                                case 1n+ +h:
                                  Empty.absurd({TileLaws.select_lane(8n+h,lane_dots(ready,tap,block)) == lane_dot(ready,tap,block,8n+h) : F32},
                                    Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(h,0n),True{},Equal.sym(Bool,Nat.is_lt(h,0n),False{},Division.lt_zero(h)),small)))

# Lane output % 8 of the cell of output block output / 8 at a tap of the
# group over a range is the numeric dot of the tap and output over the range.
def range_group_numeric(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>,+x_base: U32,+d_base: U32,+first: Nat,+count: Nat,+input: Buffer.Model<Tile.Tile8>,
  +gradient: Buffer.Model<Tile.Tile8>,+nonempty: {Nat.is_lt(0n,count) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(first,count),Nat.mul(row_count(geometry),column_count(geometry))) == True{} : Bool},+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(geometry)) == True{} : Bool},+output_inside: {U32.is_lt(output,geometry_outputs(geometry)) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),Tree.read(Tile.Tile8,group_output(group(geometry,x_base,d_base,Grouped.Range{first,count},input,gradient)),
      U32.from_nat(Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),tap_count(geometry)),U32.to_nat(tap))))) ==
    numeric_dot(NumericSource{input,gradient,geometry,x_base,d_base},tap,output,count,first,0.0) : F32}:
  +ready = {BandReady{geometry,domain,x_base,d_base,first,count,input,gradient,nonempty,room} : BandReady}
  +limits = domain_limits(geometry,domain)
  +out = U32.to_nat(output)
  +block = Nat.div(out,8n)
  +lane = Nat.mod(out,8n)
  +tap_n = U32.to_nat(tap)
  +small = DivisionOrder.remainder_small(out,8n,{==})
  +source = {NumericSource{input,gradient,geometry,x_base,d_base} : NumericSource}
  +output_n = Bounds.as_nat_lt(output,geometry_outputs(geometry),output_inside)
  +split = {%DivisionOrder.quotient_exact(out,8n,{==}) : {Nat.is_lt(_,output_count(geometry)) == True{} : Bool} output_n :
    {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),output_count(geometry)) == True{} : Bool}}
  +clipped = part_clipped(limits,output_count(geometry),block,lane,outputs_room(geometry,domain),small,split)
  +lanes = block_lanes(geometry,block)
  +lane_word = {
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,FullRange.roundtrip(lane,Machine.capacity(limits),
        Order.transitive(lane,output_count(geometry),U32.to_nat(Machine.capacity(limits)),
          Order.drop_left_le(Nat.mul(block,8n),lane,output_count(geometry),Bounds.strict_le(Nat.add(Nat.mul(block,8n),lane),output_count(geometry),split)),outputs_room(geometry,domain)))) :
      {Nat.is_lt(_,U32.to_nat(lanes)) == True{} : Bool}
    clipped : {Nat.is_lt(U32.to_nat(U32.from_nat(lane)),U32.to_nat(lanes)) == True{} : Bool}}
  +lane_inside = {%Equal.sym(Bool,U32.is_lt(U32.from_nat(lane),lanes),Nat.is_lt(U32.to_nat(U32.from_nat(lane)),U32.to_nat(lanes)),Comparison.lt(U32.from_nat(lane),lanes)) : {_ == True{} : Bool} lane_word : {U32.is_lt(U32.from_nat(lane),lanes) == True{} : Bool}}
  +tile = band_cells(ready,tap_n,block,Bounds.as_nat_lt(tap,Packing.geometry_taps(geometry),tap_inside),eight_block_inside(output_count(geometry),out,output_n))
  +dots = lane_dots(ready,tap_n,block)
  +selected = Equal.trans(F32,TileLaws.select_lane(lane,cell_wanted(ready,Cell{tap_n,block})),TileLaws.select_lane(lane,dots),lane_dot(ready,tap_n,block,lane),
    GemmLaws.finished_lane(~GemmLaws.unbiased_value,~Gemm.kept,0,lanes,no_bias(),dots,lane,lane_inside,small),dots_lane(ready,tap_n,block,lane,small))
  +read = Equal.trans(F32,TileLaws.select_lane(lane,Tree.read(Tile.Tile8,group_output(band_group(ready_band(ready))),cell_word(ready,Cell{tap_n,block}))),
    TileLaws.select_lane(lane,cell_wanted(ready,Cell{tap_n,block})),lane_dot(ready,tap_n,block,lane),
    Equal.cong(Tile.Tile8,F32,value => TileLaws.select_lane(lane,value),Tree.read(Tile.Tile8,group_output(band_group(ready_band(ready))),cell_word(ready,Cell{tap_n,block})),
      cell_wanted(ready,Cell{tap_n,block}),tile),selected)
  %Arithmetic.native_roundtrip(output) : {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,group_output(band_group(ready_band(ready))),cell_word(ready,Cell{tap_n,block}))) ==
    numeric_dot(source,tap,_,count,first,0.0) : F32}
  %Arithmetic.native_roundtrip(tap) : {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,group_output(band_group(ready_band(ready))),cell_word(ready,Cell{tap_n,block}))) ==
    numeric_dot(source,_,U32.from_nat(out),count,first,0.0) : F32}
  %Equal.sym(Nat,out,Nat.add(Nat.mul(block,8n),lane),DivisionOrder.quotient_exact(out,8n,{==})) : {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,group_output(band_group(ready_band(ready))),cell_word(ready,Cell{tap_n,block}))) ==
    numeric_dot(source,U32.from_nat(tap_n),U32.from_nat(_),count,first,0.0) : F32}
  read

# The output of a product is balanced at its depth.
def product_frame(+plan: Kernel.Plan,+jobs: U32,+pixels: Packing.Packed,+gradients: Packing.Packed,+depth: Nat,+values: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,values)) -> StorageFrame(depth,U32.shln(1,depth),group_output(product(plan,jobs,pixels,gradients,values))):
  Packing.Packed{+input,+pixel_tiles} = pixels
  Packing.Packed{+gradient,+gradient_tiles} = gradients
  +writes = GemmJobs.all_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.advance,~GemmLaws.LaneFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.kept,U32.to_nat(jobs),0,plan,
    GemmJobs.Readers{pixel_tiles,gradient_tiles,no_bias()})
  +kept = Writes.writes_preserve_balance(Tile.Tile8,writes,values,depth,balanced)
  Gemm.Both{kept,Shape.capacity(Tile.Tile8,depth,Writes.pure(Tile.Tile8,writes,values),kept)}

# A group's output is a balanced tree of the plan's depth, and it gives back
# the input and the gradient it read.
def group_frame(+geometry: Kernel.Geometry,+domain: GeometryReady<geometry>,+x_base: U32,+d_base: U32,+range: Grouped.Range,+input: Buffer.Model<Tile.Tile8>,+gradient: Buffer.Model<Tile.Tile8>) ->
  Gemm.Both<StorageFrame(plan_depth(geometry_plan(geometry,domain)),plan_capacity(geometry_plan(geometry,domain)),group_output(group(geometry,x_base,d_base,range,input,gradient))),
    {group_sources(group(geometry,x_base,d_base,range,input,gradient)) == Sources{input,gradient} : Sources}>:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  GeometryReady{+limits,+columns_positive,+kernel_positive,+stride_positive,+square,+taps_positive,+outputs_positive,+cells,+aligned,+taps_exact,+pixels,+gradients} = domain
  Grouped.Range{+first,+count} = range
  +band = {Band{Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},x_base,d_base,first,count,input,gradient} : Band}
  +depth = plan_depth(geometry_plan(geometry,domain))
  Gemm.Both{product_frame(band_plan(band),band_jobs(band),band_pixels(band),band_gradients(band),depth,band_values(band),Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0))),
    group_sources_kept(geometry,x_base,d_base,range,input,gradient)}
