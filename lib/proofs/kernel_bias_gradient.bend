# The bias kernel over the pieces that hold the gradient. Every range of
# positions lends the band of its cells, folds them and gives the band back,
# so the holding keeps covering the gradient, and every result is the bias
# model over the cells the holder holds. A lane of a written cell is the
# adjacent tree, over the images, of the adjacent tree of the ordered sums of
# the ranges of its output, +0 past the outputs.
import Base
import ../kernel_parameter_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../tensor_move.bend as Move
import ../storage_buffer.bend as Storage
import ../traversal_grouped.bend as Grouped
import ../traversal_loop.bend as Loop
import ../machine_limits.bend as Machine
import ../kernel_gemm.bend as KernelGemm
import ./tensor_move_readback.bend as Held
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./grouped_reduction_refinement.bend as Reduction
import ./kernel_tile_refinement.bend as TileLaws
import ./list_storage_readback.bend as ListRead
import ./sequence_membership.bend as Members
import ./machine_limit_bounds.bend as Limits
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./u32_full_range.bend as FullRange
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_encoding.bend as WordEncoding
import ./kernel_pipeline_buffer_shape.bend as Footprints
import ./kernel_pipeline_allocation_ready.bend as AllocationReady
import ../tensor.bend as Tensors
import ./tensor_list_values.bend as Listed

# ---- The work ----

def work_images(work: Kernel.Work) -> Nat:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  n

def work_outputs(work: Kernel.Work) -> Nat:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  o

def work_rows(work: Kernel.Work) -> Nat:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  rows

def work_columns(work: Kernel.Work) -> Nat:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  columns

def work_group(work: Kernel.Work) -> Nat:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  g

def work_positions(+work: Kernel.Work) -> Nat:
  Nat.mul(work_rows(work),work_columns(work))

def work_groups(+work: Kernel.Work) -> Nat:
  Kernel.groups(work_outputs(work))

def work_ranges(+work: Kernel.Work) -> List<&2,Grouped.Range>:
  Grouped.ranges(work_rows(work),work_columns(work),work_group(work))

# ---- The values of the bias ----

# Output element of an image at a position: lane element % 8 of the cell its
# holder holds there.
def held_sample(~D: Data,~dv: D -> U32 -> Tile.Tile8,source: D,context: Kernel.Context,position: Nat) -> F32:
  Kernel.Context{+work,+image,+element} = context
  TileLaws.select_lane(Nat.mod(element,8n),dv(source,Kernel.bias_address(Kernel.Context{work,image,element},position)))

def image_values(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,remaining: Nat,+image: Nat,+element: Nat,+work: Kernel.Work,
  +ranges: List<&2,Grouped.Range>,+source: Source) -> List<&2,F32>:
  match remaining:
    case 0n: []
    case 1n+rest:
      Grouped.tree(Reduction.group_values(~Source,~Kernel.Context,~sample,ranges,source,Kernel.Context{work,image,element})) <>
        image_values(~Source,~sample,rest,1n+image,element,work,ranges,source)

def element(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,+work: Kernel.Work,index: Nat,+ranges: List<&2,Grouped.Range>,source: Source) -> F32:
  Kernel.Work{images,c,o,h,w,k,s,p,rows,columns,g} = work
  Grouped.tree(image_values(~Source,~sample,images,0n,index,work,ranges,source))

# Lane lane of a block is output 8 * block + lane, +0 past the outputs.
def lane_value(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,inside: Bool,+work: Kernel.Work,+index: Nat,+ranges: List<&2,Grouped.Range>,+source: Source) -> F32:
  match inside:
    case True{}: element(~Source,~sample,work,index,ranges,source)
    case False{}: 0.0

def block_lane(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,source: Source,context: Kernel.Block,lane: Nat) -> F32:
  Kernel.Block{+work,+block,+ranges} = context
  Kernel.Work{n,c,+o,h,w,k,s,p,rows,columns,g} = work
  +index = Nat.add(Nat.mul(block,8n),lane)
  lane_value(~Source,~sample,Nat.is_lt(index,o),work,index,ranges,source)

def cell_values(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,+source: Source,+context: Kernel.Block) -> Tile.Tile8:
  Tile.Tile{block_lane(~Source,~sample,source,context,0n),block_lane(~Source,~sample,source,context,1n),block_lane(~Source,~sample,source,context,2n),block_lane(~Source,~sample,source,context,3n),
    block_lane(~Source,~sample,source,context,4n),block_lane(~Source,~sample,source,context,5n),block_lane(~Source,~sample,source,context,6n),block_lane(~Source,~sample,source,context,7n)}

# ---- The model of a range ----

# The cells of a range of positions from a total, as the holder holds them.
def held_fold(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+context: Kernel.Context,remaining: Nat,+position: Nat,total: Tile.Tile8) -> Tile.Tile8:
  match remaining:
    case 0n: total
    case 1n+rest: held_fold(~D,~dv,source,context,rest,1n+position,Tile.plus(total,dv(source,Kernel.bias_address(context,position))))

# Output 8 * block + lane is lane lane of block block.
def block_of(+block: Nat,+lane: Nat,+below: {Nat.is_lt(lane,8n) == True{} : Bool}) -> {Nat.div(Nat.add(Nat.mul(block,8n),lane),8n) == block : Nat}:
  Equal.trans(Nat,Nat.div(Nat.add(Nat.mul(block,8n),lane),8n),Nat.add(block,Nat.div(lane,8n)),block,DivisionOrder.quotient_shift(lane,8n,block,{==}),
    Equal.trans(Nat,Nat.add(block,Nat.div(lane,8n)),Nat.add(block,0n),block,
      Equal.cong(Nat & Nat,Nat,pair => Nat.add(block,Pair.fst(Nat,Nat,pair)),Nat.divmod(lane,8n),(0n,lane),Division.small(lane,8n,below)),Addition.zero_right(block)))

def lane_of(+block: Nat,+lane: Nat,+below: {Nat.is_lt(lane,8n) == True{} : Bool}) -> {Nat.mod(Nat.add(Nat.mul(block,8n),lane),8n) == lane : Nat}:
  Equal.trans(Nat,Nat.mod(Nat.add(Nat.mul(block,8n),lane),8n),Nat.mod(lane,8n),lane,DivisionOrder.remainder_shift(lane,8n,block,{==}),
    Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(lane,8n),(0n,lane),Division.small(lane,8n,below)))

def same_address(+work: Kernel.Work,+image: Nat,+left: Nat,+right: Nat,+position: Nat,same: {Nat.div(left,8n) == Nat.div(right,8n) : Nat}) ->
  {Kernel.bias_address(Kernel.Context{work,image,left},position) == Kernel.bias_address(Kernel.Context{work,image,right},position) : U32}:
  Kernel.Work{n,c,+o,h,w,k,s,p,+rows,+columns,g} = work
  Equal.cong(Nat,U32,quotient => U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(rows,columns)),position),Kernel.groups(o)),quotient)),Nat.div(left,8n),Nat.div(right,8n),same)

# The sample of an output is lane lane of the cell of its block.
def sample_lane(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+image: Nat,+block: Nat,+lane: Nat,+position: Nat,+below: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {held_sample(~D,~dv,source,Kernel.Context{work,image,Nat.add(Nat.mul(block,8n),lane)},position) ==
    TileLaws.select_lane(lane,dv(source,Kernel.bias_address(Kernel.Context{work,image,Nat.mul(block,8n)},position))) : F32}:
  +element = Nat.add(Nat.mul(block,8n),lane)
  +start = Nat.mul(block,8n)
  +target = Kernel.bias_address(Kernel.Context{work,image,start},position)
  %Equal.sym(Nat,Nat.mod(element,8n),lane,lane_of(block,lane,below)) :
    {TileLaws.select_lane(_,dv(source,Kernel.bias_address(Kernel.Context{work,image,element},position))) == TileLaws.select_lane(lane,dv(source,target)) : F32}
  %Equal.sym(U32,Kernel.bias_address(Kernel.Context{work,image,element},position),target,
    same_address(work,image,element,start,position,Equal.trans(Nat,Nat.div(element,8n),block,Nat.div(start,8n),block_of(block,lane,below),Equal.sym(Nat,Nat.div(start,8n),block,Division.exact_quotient(block,8n,{==}))))) :
    {TileLaws.select_lane(lane,dv(source,_)) == TileLaws.select_lane(lane,dv(source,target)) : F32}
  {==}

# A lane of the cells of a range is the ordered sum of the output of that lane.
def fold_lane(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+image: Nat,+block: Nat,+lane: Nat,+below: {Nat.is_lt(lane,8n) == True{} : Bool},
  remaining: Nat,+position: Nat,+total: Tile.Tile8) ->
  {TileLaws.select_lane(lane,held_fold(~D,~dv,source,Kernel.Context{work,image,Nat.mul(block,8n)},remaining,position,total)) ==
    Reduction.values(~D,~Kernel.Context,~held_sample(~D,~dv),remaining,position,source,Kernel.Context{work,image,Nat.add(Nat.mul(block,8n),lane)},TileLaws.select_lane(lane,total)) : F32}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +start = {Kernel.Context{work,image,Nat.mul(block,8n)} : Kernel.Context}
      +output = {Kernel.Context{work,image,Nat.add(Nat.mul(block,8n),lane)} : Kernel.Context}
      +value = dv(source,Kernel.bias_address(start,position))
      %Equal.sym(F32,held_sample(~D,~dv,source,output,position),TileLaws.select_lane(lane,value),sample_lane(~D,~dv,source,work,image,block,lane,position,below)) :
        {TileLaws.select_lane(lane,held_fold(~D,~dv,source,start,rest,1n+position,Tile.plus(total,value))) ==
          Reduction.values(~D,~Kernel.Context,~held_sample(~D,~dv),rest,1n+position,source,output,(TileLaws.select_lane(lane,total) + _ : F32)) : F32}
      %TileLaws.added_lane(total,value,lane) :
        {TileLaws.select_lane(lane,held_fold(~D,~dv,source,start,rest,1n+position,Tile.plus(total,value))) ==
          Reduction.values(~D,~Kernel.Context,~held_sample(~D,~dv),rest,1n+position,source,output,_) : F32}
      fold_lane(~D,~dv,source,work,image,block,lane,below,rest,1n+position,Tile.plus(total,value))

# ---- The model of the groups of an image ----

def group_totals(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+context: Kernel.Context,ranges: List<&2,Grouped.Range>) -> List<&2,Tile.Tile8>:
  match ranges:
    case []: []
    case Grouped.Range{first,count} <> rest: held_fold(~D,~dv,source,context,count,first,Tile.filled(0.0)) <> group_totals(~D,~dv,source,context,rest)

# One lane of every cell of a list.
def lanes(+lane: Nat,values: List<&2,Tile.Tile8>) -> List<&2,F32>:
  match values:
    case []: []
    case head <> tail: TileLaws.select_lane(lane,head) <> lanes(lane,tail)

def group_lanes(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+image: Nat,+block: Nat,+lane: Nat,+below: {Nat.is_lt(lane,8n) == True{} : Bool},ranges: List<&2,Grouped.Range>) ->
  {lanes(lane,group_totals(~D,~dv,source,Kernel.Context{work,image,Nat.mul(block,8n)},ranges)) ==
    Reduction.group_values(~D,~Kernel.Context,~held_sample(~D,~dv),ranges,source,Kernel.Context{work,image,Nat.add(Nat.mul(block,8n),lane)}) : List<&2,F32>}:
  match ranges:
    case []: {==}
    case Grouped.Range{+first,+count} <> +rest:
      +start = {Kernel.Context{work,image,Nat.mul(block,8n)} : Kernel.Context}
      +output = {Kernel.Context{work,image,Nat.add(Nat.mul(block,8n),lane)} : Kernel.Context}
      +total = TileLaws.select_lane(lane,held_fold(~D,~dv,source,start,count,first,Tile.filled(0.0)))
      %TileLaws.filled_lane(lane,0.0) :
        {total <> lanes(lane,group_totals(~D,~dv,source,start,rest)) ==
          Reduction.values(~D,~Kernel.Context,~held_sample(~D,~dv),count,first,source,output,_) <> Reduction.group_values(~D,~Kernel.Context,~held_sample(~D,~dv),rest,source,output) : List<&2,F32>}
      %fold_lane(~D,~dv,source,work,image,block,lane,below,count,first,Tile.filled(0.0)) :
        {total <> lanes(lane,group_totals(~D,~dv,source,start,rest)) == _ <> Reduction.group_values(~D,~Kernel.Context,~held_sample(~D,~dv),rest,source,output) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => total <> tail,lanes(lane,group_totals(~D,~dv,source,start,rest)),Reduction.group_values(~D,~Kernel.Context,~held_sample(~D,~dv),rest,source,output),
        group_lanes(~D,~dv,source,work,image,block,lane,below,rest))

def group_length(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+context: Kernel.Context,ranges: List<&2,Grouped.Range>) ->
  {List.length(&2,F32,Reduction.group_values(~D,~Kernel.Context,~held_sample(~D,~dv),ranges,source,context)) == List.length(&2,Grouped.Range,ranges) : Nat}:
  match ranges:
    case []: {==}
    case Grouped.Range{+first,+count} <> +rest:
      %group_length(~D,~dv,source,context,rest) :
        {1n+List.length(&2,F32,Reduction.group_values(~D,~Kernel.Context,~held_sample(~D,~dv),rest,source,context)) == 1n+_ : Nat}
      {==}

# ---- Trees of cells ----

def pairs_lane(+lane: Nat,+values: List<&2,Tile.Tile8>) ->
  {lanes(lane,Grouped.pairs(~&2,~Tile.Tile8,~Unit,~Kernel.tile_added,Unit{},values)) == Grouped.pairs(~&2,~F32,~Unit,~(context => F32.add),Unit{},lanes(lane,values)) : List<&2,F32>}:
  match values:
    case Nil{}: {==}
    case head <> Nil{}: {==}
    case +left <> +right <> +tail:
      +paired = lanes(lane,Grouped.pairs(~&2,~Tile.Tile8,~Unit,~Kernel.tile_added,Unit{},tail))
      %TileLaws.added_lane(left,right,lane) :
        {TileLaws.select_lane(lane,Tile.plus(left,right)) <> paired == _ <> Grouped.pairs(~&2,~F32,~Unit,~(context => F32.add),Unit{},lanes(lane,tail)) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,rest => TileLaws.select_lane(lane,Tile.plus(left,right)) <> rest,paired,Grouped.pairs(~&2,~F32,~Unit,~(context => F32.add),Unit{},lanes(lane,tail)),pairs_lane(lane,tail))

# A lane of the tree of cells is the tree of that lane.
def tree_lane(+lane: Nat,+fuel: Nat,+values: List<&2,Tile.Tile8>) ->
  {TileLaws.select_lane(lane,Kernel.tree(fuel,values)) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},lanes(lane,values)) : F32}:
  match fuel values:
    case 0n Nil{}: TileLaws.filled_lane(lane,0.0)
    case 0n head <> tail: {==}
    case 1n+rest Nil{}: TileLaws.filled_lane(lane,0.0)
    case 1n+rest head <> Nil{}: {==}
    case 1n+ +rest +left <> +right <> +tail:
      +paired = Grouped.pairs(~&2,~Tile.Tile8,~Unit,~Kernel.tile_added,Unit{},left <> right <> tail)
      %pairs_lane(lane,left <> right <> tail) :
        {TileLaws.select_lane(lane,Kernel.tree(rest,paired)) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),rest,Unit{},_) : F32}
      tree_lane(lane,rest,paired)

# A lane of the tree of the groups of an image is the tree of its output.
def image_lane(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+image: Nat,+block: Nat,+lane: Nat,+below: {Nat.is_lt(lane,8n) == True{} : Bool},+ranges: List<&2,Grouped.Range>) ->
  {TileLaws.select_lane(lane,Kernel.tree(List.length(&2,Grouped.Range,ranges),group_totals(~D,~dv,source,Kernel.Context{work,image,Nat.mul(block,8n)},ranges))) ==
    Grouped.tree(Reduction.group_values(~D,~Kernel.Context,~held_sample(~D,~dv),ranges,source,Kernel.Context{work,image,Nat.add(Nat.mul(block,8n),lane)})) : F32}:
  +output = {Kernel.Context{work,image,Nat.add(Nat.mul(block,8n),lane)} : Kernel.Context}
  +totals = group_totals(~D,~dv,source,Kernel.Context{work,image,Nat.mul(block,8n)},ranges)
  +values = Reduction.group_values(~D,~Kernel.Context,~held_sample(~D,~dv),ranges,source,output)
  +fuel = List.length(&2,Grouped.Range,ranges)
  %Equal.sym(Nat,List.length(&2,F32,values),fuel,group_length(~D,~dv,source,output,ranges)) :
    {TileLaws.select_lane(lane,Kernel.tree(fuel,totals)) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),_,Unit{},values) : F32}
  %group_lanes(~D,~dv,source,work,image,block,lane,below,ranges) :
    {TileLaws.select_lane(lane,Kernel.tree(fuel,totals)) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},_) : F32}
  tree_lane(lane,fuel,totals)

# ---- The model of the images ----

def image_totals(~D: Data,~dv: D -> U32 -> Tile.Tile8,remaining: Nat,+image: Nat,+element: Nat,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,+source: D) -> List<&2,Tile.Tile8>:
  match remaining:
    case 0n: []
    case 1n+rest:
      Kernel.tree(List.length(&2,Grouped.Range,ranges),group_totals(~D,~dv,source,Kernel.Context{work,image,element},ranges)) <> image_totals(~D,~dv,rest,1n+image,element,work,ranges,source)

def images_lane(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+block: Nat,+lane: Nat,+below: {Nat.is_lt(lane,8n) == True{} : Bool},remaining: Nat,+image: Nat,+ranges: List<&2,Grouped.Range>) ->
  {lanes(lane,image_totals(~D,~dv,remaining,image,Nat.mul(block,8n),work,ranges,source)) == image_values(~D,~held_sample(~D,~dv),remaining,image,Nat.add(Nat.mul(block,8n),lane),work,ranges,source) : List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +head = TileLaws.select_lane(lane,Kernel.tree(List.length(&2,Grouped.Range,ranges),group_totals(~D,~dv,source,Kernel.Context{work,image,Nat.mul(block,8n)},ranges)))
      %image_lane(~D,~dv,source,work,image,block,lane,below,ranges) :
        {head <> lanes(lane,image_totals(~D,~dv,rest,1n+image,Nat.mul(block,8n),work,ranges,source)) ==
          _ <> image_values(~D,~held_sample(~D,~dv),rest,1n+image,Nat.add(Nat.mul(block,8n),lane),work,ranges,source) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => head <> tail,lanes(lane,image_totals(~D,~dv,rest,1n+image,Nat.mul(block,8n),work,ranges,source)),
        image_values(~D,~held_sample(~D,~dv),rest,1n+image,Nat.add(Nat.mul(block,8n),lane),work,ranges,source),images_lane(~D,~dv,source,work,block,lane,below,rest,1n+image,ranges))

def images_length(~D: Data,~dv: D -> U32 -> Tile.Tile8,remaining: Nat,+image: Nat,+element: Nat,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,+source: D) ->
  {List.length(&2,F32,image_values(~D,~held_sample(~D,~dv),remaining,image,element,work,ranges,source)) == remaining : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %images_length(~D,~dv,rest,1n+image,element,work,ranges,source) :
        {1n+List.length(&2,F32,image_values(~D,~held_sample(~D,~dv),rest,1n+image,element,work,ranges,source)) == 1n+_ : Nat}
      {==}

# ---- The model of a cell ----

# The tree of the images of a block, before the lanes past the outputs are
# cleared.
def block_total(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+block: Nat,+ranges: List<&2,Grouped.Range>) -> Tile.Tile8:
  Kernel.tree(work_images(work),image_totals(~D,~dv,work_images(work),0n,Nat.mul(block,8n),work,ranges,source))

def cell_lane(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+block: Nat,+lane: Nat,+below: {Nat.is_lt(lane,8n) == True{} : Bool},+ranges: List<&2,Grouped.Range>) ->
  {TileLaws.select_lane(lane,block_total(~D,~dv,source,work,block,ranges)) == element(~D,~held_sample(~D,~dv),work,Nat.add(Nat.mul(block,8n),lane),ranges,source) : F32}:
  Kernel.Work{+n,c,o,h,w,k,s,p,rows,columns,g} = work
  +index = Nat.add(Nat.mul(block,8n),lane)
  +values = image_values(~D,~held_sample(~D,~dv),n,0n,index,work,ranges,source)
  +totals = image_totals(~D,~dv,n,0n,Nat.mul(block,8n),work,ranges,source)
  %Equal.sym(Nat,List.length(&2,F32,values),n,images_length(~D,~dv,n,0n,index,work,ranges,source)) :
    {TileLaws.select_lane(lane,Kernel.tree(n,totals)) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),_,Unit{},values) : F32}
  %images_lane(~D,~dv,source,work,block,lane,below,n,0n,ranges) :
    {TileLaws.select_lane(lane,Kernel.tree(n,totals)) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),n,Unit{},_) : F32}
  tree_lane(lane,n,totals)

def picked_value(~D: Data,~dv: D -> U32 -> Tile.Tile8,inside: Bool,+work: Kernel.Work,+index: Nat,+ranges: List<&2,Grouped.Range>,+source: D,+value: F32,
  +same: {value == element(~D,~held_sample(~D,~dv),work,index,ranges,source) : F32}) ->
  {Bool.pick(F32,inside,value,0.0) == lane_value(~D,~held_sample(~D,~dv),inside,work,index,ranges,source) : F32}:
  match inside:
    case True{}: same
    case False{}: {==}

def kept_lane(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+block: Nat,+lane: Nat,+below: {Nat.is_lt(lane,8n) == True{} : Bool},+ranges: List<&2,Grouped.Range>) ->
  {Kernel.lane_kept(work_outputs(work),block,lane,TileLaws.select_lane(lane,block_total(~D,~dv,source,work,block,ranges))) ==
    block_lane(~D,~held_sample(~D,~dv),source,Kernel.Block{work,block,ranges},lane) : F32}:
  Kernel.Work{n,c,+o,h,w,k,s,p,rows,columns,g} = work
  +index = Nat.add(Nat.mul(block,8n),lane)
  picked_value(~D,~dv,Nat.is_lt(index,o),work,index,ranges,source,TileLaws.select_lane(lane,block_total(~D,~dv,source,work,block,ranges)),cell_lane(~D,~dv,source,work,block,lane,below,ranges))

def kept_tile(+outputs: Nat,+block: Nat,+tile: Tile.Tile8) ->
  {Kernel.kept(outputs,block,tile) == Tile.Tile{Kernel.lane_kept(outputs,block,0n,Tile.lane0(tile)),Kernel.lane_kept(outputs,block,1n,Tile.lane1(tile)),Kernel.lane_kept(outputs,block,2n,Tile.lane2(tile)),Kernel.lane_kept(outputs,block,3n,Tile.lane3(tile)),
    Kernel.lane_kept(outputs,block,4n,Tile.lane4(tile)),Kernel.lane_kept(outputs,block,5n,Tile.lane5(tile)),Kernel.lane_kept(outputs,block,6n,Tile.lane6(tile)),Kernel.lane_kept(outputs,block,7n,Tile.lane7(tile))} : Tile.Tile8}:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  {==}

# The cell of a block is the cell of the values of its outputs.
def cell_equal(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+block: Nat,+ranges: List<&2,Grouped.Range>) ->
  {Kernel.kept(work_outputs(work),block,block_total(~D,~dv,source,work,block,ranges)) == cell_values(~D,~held_sample(~D,~dv),source,Kernel.Block{work,block,ranges}) : Tile.Tile8}:
  +tile = block_total(~D,~dv,source,work,block,ranges)
  +outputs = work_outputs(work)
  +laned = {Tile.Tile{Kernel.lane_kept(outputs,block,0n,Tile.lane0(tile)),Kernel.lane_kept(outputs,block,1n,Tile.lane1(tile)),Kernel.lane_kept(outputs,block,2n,Tile.lane2(tile)),Kernel.lane_kept(outputs,block,3n,Tile.lane3(tile)),
    Kernel.lane_kept(outputs,block,4n,Tile.lane4(tile)),Kernel.lane_kept(outputs,block,5n,Tile.lane5(tile)),Kernel.lane_kept(outputs,block,6n,Tile.lane6(tile)),Kernel.lane_kept(outputs,block,7n,Tile.lane7(tile))} : Tile.Tile8}
  Equal.trans(Tile.Tile8,Kernel.kept(outputs,block,tile),laned,cell_values(~D,~held_sample(~D,~dv),source,Kernel.Block{work,block,ranges}),kept_tile(outputs,block,tile),
    TileLaws.tiles_equal(laned,cell_values(~D,~held_sample(~D,~dv),source,Kernel.Block{work,block,ranges}),
      kept_lane(~D,~dv,source,work,block,0n,{==},ranges),kept_lane(~D,~dv,source,work,block,1n,{==},ranges),kept_lane(~D,~dv,source,work,block,2n,{==},ranges),kept_lane(~D,~dv,source,work,block,3n,{==},ranges),
      kept_lane(~D,~dv,source,work,block,4n,{==},ranges),kept_lane(~D,~dv,source,work,block,5n,{==},ranges),kept_lane(~D,~dv,source,work,block,6n,{==},ranges),kept_lane(~D,~dv,source,work,block,7n,{==},ranges)))

# ---- The model of the result ----

def values(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,remaining: Nat,+index: Nat,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,+source: Source) -> List<&2,Tile.Tile8>:
  match remaining:
    case 0n: []
    case 1n+rest: cell_values(~Source,~sample,source,Kernel.Block{work,index,ranges}) <> values(~Source,~sample,rest,1n+index,work,ranges,source)

def values_length(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,remaining: Nat,+index: Nat,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,+source: Source) ->
  {List.length(&2,Tile.Tile8,values(~Source,~sample,remaining,index,work,ranges,source)) == remaining : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %values_length(~Source,~sample,rest,1n+index,work,ranges,source) : {1n+List.length(&2,Tile.Tile8,values(~Source,~sample,rest,1n+index,work,ranges,source)) == 1n+_ : Nat}
      {==}

# The cell at a position of the list is the cell of its block.
def values_at(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,remaining: Nat,position: Nat,+index: Nat,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,+source: Source,
  within: {Nat.is_lt(position,remaining) == True{} : Bool}) ->
  {ListRead.at(~Tile.Tile8,Tile.filled(0.0),position,values(~Source,~sample,remaining,index,work,ranges,source)) ==
    cell_values(~Source,~sample,source,Kernel.Block{work,Nat.add(index,position),ranges}) : Tile.Tile8}:
  match remaining position:
    case 0n 0n: Empty.absurd({Tile.filled(0.0) == cell_values(~Source,~sample,source,Kernel.Block{work,Nat.add(index,0n),ranges}) : Tile.Tile8},WordEncoding.false_true(within))
    case 0n 1n+position: Empty.absurd({Tile.filled(0.0) == cell_values(~Source,~sample,source,Kernel.Block{work,Nat.add(index,1n+position),ranges}) : Tile.Tile8},WordEncoding.false_true(within))
    case 1n+rest 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) :
        {cell_values(~Source,~sample,source,Kernel.Block{work,index,ranges}) == cell_values(~Source,~sample,source,Kernel.Block{work,_,ranges}) : Tile.Tile8}
      {==}
    case 1n+ +rest 1n+ +previous:
      %Equal.sym(Nat,Nat.add(index,1n+previous),1n+Nat.add(index,previous),Addition.successor_right(index,previous)) :
        {ListRead.at(~Tile.Tile8,Tile.filled(0.0),previous,values(~Source,~sample,rest,1n+index,work,ranges,source)) ==
          cell_values(~Source,~sample,source,Kernel.Block{work,_,ranges}) : Tile.Tile8}
      values_at(~Source,~sample,rest,previous,1n+index,work,ranges,source,within)

# The cells of every block of eight outputs.
def bias_values(~D: Data,~dv: D -> U32 -> Tile.Tile8,+work: Kernel.Work,source: D) -> List<&2,Tile.Tile8>:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  values(~D,~held_sample(~D,~dv),Kernel.groups(o),0n,work,Grouped.ranges(rows,columns,g),source)

# Each cell written at its position, one after another.
def written_model(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,remaining: Nat,+index: Nat,+position: U32,+work: Kernel.Work,
  +ranges: List<&2,Grouped.Range>,+source: Source,result: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  match remaining:
    case 0n: result
    case 1n+rest:
      written_model(~Source,~sample,rest,1n+index,(position + 1 : U32),work,ranges,source,
        Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,position,cell_values(~Source,~sample,source,Kernel.Block{work,index,ranges}),Buffer.pack(~Tile.Tile8,result))))

def written_content(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,remaining: Nat,+index: Nat,+position: U32,+work: Kernel.Work,
  +ranges: List<&2,Grouped.Range>,+source: Source,+result: Buffer.Model<Tile.Tile8>) ->
  {Buffer.content(~Tile.Tile8,written_model(~Source,~sample,remaining,index,position,work,ranges,source,result)) ==
    Buffer.Content{Tree.write_sequence(~Tile.Tile8,values(~Source,~sample,remaining,index,work,ranges,source),position,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))),
      Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))} : Buffer.Content<Tile.Tile8>}:
  match remaining:
    case 0n:
      Buffer.Model{tree,length,certificate} = result
      {==}
    case 1n+ +rest:
      +value = cell_values(~Source,~sample,source,Kernel.Block{work,index,ranges})
      +next = Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,position,value,Buffer.pack(~Tile.Tile8,result)))
      +stepped = Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,next),Buffer.contents(~Tile.Tile8,Storage.write(~Tile.Tile8,position,value,Buffer.pack(~Tile.Tile8,result))),
        Buffer.Content{Tree.write(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,result)),position,value),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))},
        Buffer.remodel_content(~Tile.Tile8,Storage.write(~Tile.Tile8,position,value,Buffer.pack(~Tile.Tile8,result))),Buffer.write_value_packed(~Tile.Tile8,position,value,result))
      %stepped :
        {Buffer.content(~Tile.Tile8,written_model(~Source,~sample,rest,1n+index,(position + 1 : U32),work,ranges,source,next)) ==
          Buffer.Content{Tree.write_sequence(~Tile.Tile8,values(~Source,~sample,rest,1n+index,work,ranges,source),(position + 1 : U32),Buffer.content_tree(~Tile.Tile8,_)),
            Buffer.content_length(~Tile.Tile8,_)} : Buffer.Content<Tile.Tile8>}
      written_content(~Source,~sample,rest,1n+index,(position + 1 : U32),work,ranges,source,next)

def bias_written(~D: Data,~dv: D -> U32 -> Tile.Tile8,+work: Kernel.Work,+source: D,+result: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  written_model(~D,~held_sample(~D,~dv),Kernel.groups(o),0n,0,work,Grouped.ranges(rows,columns,g),source,result)

# The result holds the cells of every block from cell 0.
def bias_content(~D: Data,~dv: D -> U32 -> Tile.Tile8,+work: Kernel.Work,+source: D,+result: Buffer.Model<Tile.Tile8>) ->
  {Buffer.content(~Tile.Tile8,bias_written(~D,~dv,work,source,result)) ==
    Buffer.Content{Tree.write_sequence(~Tile.Tile8,bias_values(~D,~dv,work,source),0,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))),
      Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))} : Buffer.Content<Tile.Tile8>}:
  Kernel.Work{n,c,+o,h,w,k,s,p,+rows,+columns,+g} = work
  written_content(~D,~held_sample(~D,~dv),Kernel.groups(o),0n,0,work,Grouped.ranges(rows,columns,g),source,result)

# ---- Numbers of a range ----

# The cells of a range end below the cells of the images.
def range_end(+images: Nat,+positions: Nat,+groups: Nat,+image: Nat,+first: Nat,+count: Nat,+high: Nat,
  +inside: {Nat.is_lt(image,images) == True{} : Bool},+room: {Nat.is_le(Nat.add(first,count),positions) == True{} : Bool},
  +cells: {Nat.is_le(Nat.mul(Nat.mul(images,positions),groups),high) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.add(Nat.add(Nat.mul(image,positions),first),count),groups),high) == True{} : Bool}:
  +start = Nat.mul(image,positions)
  +within = {%Equal.sym(Nat,Nat.add(Nat.add(start,first),count),Nat.add(start,Nat.add(first,count)),Addition.associative(start,first,count)) :
      {Nat.is_le(_,Nat.add(start,positions)) == True{} : Bool}
    Order.offset_le(start,Nat.add(first,count),positions,room) : {Nat.is_le(Nat.add(Nat.add(start,first),count),Nat.add(start,positions)) == True{} : Bool}}
  +image_end = {%Order.commutative(positions,start) : {Nat.is_le(_,Nat.mul(images,positions)) == True{} : Bool}
    RowMajor.mul_le(1n+image,images,positions,Order.strict_follow(image,images,inside)) : {Nat.is_le(Nat.add(start,positions),Nat.mul(images,positions)) == True{} : Bool}}
  Order.transitive(Nat.mul(Nat.add(Nat.add(start,first),count),groups),Nat.mul(Nat.mul(images,positions),groups),high,
    RowMajor.mul_le(Nat.add(Nat.add(start,first),count),Nat.mul(images,positions),groups,
      Order.transitive(Nat.add(Nat.add(start,first),count),Nat.add(start,positions),Nat.mul(images,positions),within,image_end)),cells)

# The band of a range ends at the cell after its last read, below the cells
# of the position after the range.
def span_fits(+start: Nat,+groups: Nat,+quotient: Nat,+last: Nat,+below: {Nat.is_lt(quotient,groups) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(Nat.mul(start,groups),quotient),Nat.add(Nat.mul(last,groups),1n)),Nat.mul(Nat.add(start,1n+last),groups)) == True{} : Bool}:
  +before = Nat.mul(start,groups)
  +after = Nat.mul(last,groups)
  %Equal.sym(Nat,Nat.mul(Nat.add(start,1n+last),groups),Nat.add(before,Nat.mul(1n+last,groups)),RowMajor.distrib(start,1n+last,groups)) :
    {Nat.is_le(Nat.add(Nat.add(before,quotient),Nat.add(after,1n)),_) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.add(before,quotient),Nat.add(after,1n)),Nat.add(before,Nat.add(1n+quotient,after)),
    Equal.trans(Nat,Nat.add(Nat.add(before,quotient),Nat.add(after,1n)),Nat.add(before,Nat.add(quotient,Nat.add(after,1n))),Nat.add(before,Nat.add(1n+quotient,after)),
      Addition.associative(before,quotient,Nat.add(after,1n)),
      Equal.cong(Nat,Nat,value => Nat.add(before,value),Nat.add(quotient,Nat.add(after,1n)),Nat.add(1n+quotient,after),
        Equal.trans(Nat,Nat.add(quotient,Nat.add(after,1n)),Nat.add(quotient,1n+after),Nat.add(1n+quotient,after),
          Equal.cong(Nat,Nat,value => Nat.add(quotient,value),Nat.add(after,1n),1n+after,Addition.successor_width(after)),
          Addition.successor_right(quotient,after))))) :
    {Nat.is_le(_,Nat.add(before,Nat.mul(1n+last,groups))) == True{} : Bool}
  Order.offset_le(before,Nat.add(1n+quotient,after),Nat.mul(1n+last,groups),Order.add_le(1n+quotient,groups,after,Order.strict_follow(quotient,groups,below)))

# Read index of a range is the cell of position first + index.
def cell_shift(+base: Nat,+first: Nat,+index: Nat,+groups: Nat,+quotient: Nat) ->
  {Nat.add(Nat.add(Nat.mul(Nat.add(base,first),groups),quotient),Nat.mul(index,groups)) == Nat.add(Nat.mul(Nat.add(base,Nat.add(first,index)),groups),quotient) : Nat}:
  +before = Nat.mul(Nat.add(base,first),groups)
  +moved = Nat.mul(index,groups)
  +target = Nat.add(Nat.mul(Nat.add(base,Nat.add(first,index)),groups),quotient)
  Equal.trans(Nat,Nat.add(Nat.add(before,quotient),moved),Nat.add(before,Nat.add(quotient,moved)),target,
    Addition.associative(before,quotient,moved),
    Equal.trans(Nat,Nat.add(before,Nat.add(quotient,moved)),Nat.add(before,Nat.add(moved,quotient)),target,
      Equal.cong(Nat,Nat,value => Nat.add(before,value),Nat.add(quotient,moved),Nat.add(moved,quotient),Order.commutative(quotient,moved)),
      Equal.trans(Nat,Nat.add(before,Nat.add(moved,quotient)),Nat.add(Nat.add(before,moved),quotient),target,
        Equal.sym(Nat,Nat.add(Nat.add(before,moved),quotient),Nat.add(before,Nat.add(moved,quotient)),Addition.associative(before,moved,quotient)),
        Equal.cong(Nat,Nat,value => Nat.add(value,quotient),Nat.add(before,moved),Nat.mul(Nat.add(base,Nat.add(first,index)),groups),
          Equal.trans(Nat,Nat.add(before,moved),Nat.mul(Nat.add(Nat.add(base,first),index),groups),Nat.mul(Nat.add(base,Nat.add(first,index)),groups),
            Equal.sym(Nat,Nat.mul(Nat.add(Nat.add(base,first),index),groups),Nat.add(before,moved),RowMajor.distrib(Nat.add(base,first),index,groups)),
            Equal.cong(Nat,Nat,value => Nat.mul(value,groups),Nat.add(Nat.add(base,first),index),Nat.add(base,Nat.add(first,index)),Addition.associative(base,first,index)))))))

# The span of a band of count = 1 + last cells step groups apart.
def span_exact(+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+last: Nat,+groups: Nat,
  +positive: {Nat.is_lt(0n,groups) == True{} : Bool},+groups_room: {Nat.is_le(groups,U32.to_nat(capacity)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(last,groups),1n),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(Bool.pick(U32,U32.is_zero(U32.from_nat(1n+last)),0,((U32.from_nat(1n+last) - 1) * U32.from_nat(groups) + 1 : U32))) == Nat.add(Nat.mul(last,groups),1n) : Nat}:
  +product = Nat.mul(last,groups)
  +product_room = Order.transitive(product,Nat.add(product,1n),U32.to_nat(capacity),Order.le_add(product,1n),room)
  +count_room = Order.transitive(1n+last,Nat.add(product,1n),U32.to_nat(capacity),
    {%Addition.successor_width(last) : {Nat.is_le(_,Nat.add(product,1n)) == True{} : Bool}
      Order.add_le(last,product,1n,RowMajor.product_covers(last,groups,positive)) : {Nat.is_le(1n+last,Nat.add(product,1n)) == True{} : Bool}},room)
  +count = U32.from_nat(1n+last)
  +count_exact = FullRange.roundtrip(1n+last,capacity,count_room)
  +groups_exact = FullRange.roundtrip(groups,capacity,groups_room)
  +nonzero = Arithmetic.word_positive_nonzero(count,
    {%Equal.sym(Nat,U32.to_nat(count),1n+last,count_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
      {==} : {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}})
  +less_exact = Equal.trans(Nat,U32.to_nat((count - 1 : U32)),Nat.sub(1n+last,1n),last,
    Arithmetic.difference_exact(capacity,capacity_ok,count,1,1n+last,1n,count_exact,{==},Order.zero_le(last),count_room),Order.subtract_zero(last))
  +times = Arithmetic.product_exact(capacity,capacity_ok,(count - 1 : U32),U32.from_nat(groups),last,groups,less_exact,groups_exact,product_room)
  %Equal.sym(Bool,U32.is_zero(count),False{},nonzero) : {U32.to_nat(Bool.pick(U32,_,0,((count - 1) * U32.from_nat(groups) + 1 : U32))) == Nat.add(product,1n) : Nat}
  Arithmetic.sum_exact(capacity,capacity_ok,((count - 1) * U32.from_nat(groups) : U32),1,product,1n,times,{==},room)

# ---- The band of a range ----

def range_context(+work: Kernel.Work,+image: Nat,+block: Nat) -> Kernel.Context:
  Kernel.Context{work,image,Nat.mul(block,8n)}

def range_span(+work: Kernel.Work,+image: Nat,+block: Nat,+count: Nat) -> U32:
  +step = Kernel.cell_step(range_context(work,image,block))
  +cells = U32.from_nat(count)
  Bool.pick(U32,U32.is_zero(cells),0,((cells - 1) * step + 1 : U32))

# The band of a range, from a fresh spare.
def held_band(+model: Held.Holding<Tile.Tile8>,+work: Kernel.Work,+image: Nat,+block: Nat,+first: Nat,+count: Nat) -> Move.Band<Tile.Tile8>:
  Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,model),Kernel.bias_address(range_context(work,image,block),first),range_span(work,image,block,count),Tile.filled(0.0),
    Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))))

def range_model(+model: Held.Holding<Tile.Tile8>,+work: Kernel.Work,+image: Nat,+block: Nat,+first: Nat,+count: Nat) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,held_band(model,work,image,block,first,count)))

def range_offset(+model: Held.Holding<Tile.Tile8>,+work: Kernel.Work,+image: Nat,+block: Nat,+first: Nat,+count: Nat) -> Nat:
  U32.to_nat(Held.band_offset(~Tile.Tile8,held_band(model,work,image,block,first,count)))

# Read index of the band of a range of count = 1 + last positions is the
# cell of the holder at position first + index.
def band_cell(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+model: Held.Holding<Tile.Tile8>,+high: Nat,+limits: Machine.Limits,
  +bounded: {Nat.is_le(high,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,model,0n,high),+held_values: Held.HoldingValues(~Tile.Tile8,~D,~dv,source,model),
  +work: Kernel.Work,+image: Nat,+block: Nat,+first: Nat,+last: Nat,
  +block_inside: {Nat.is_lt(block,work_groups(work)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.mul(Nat.add(Nat.add(Nat.mul(image,work_positions(work)),first),1n+last),work_groups(work)),high) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,1n+last) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,range_model(model,work,image,block,first,1n+last)),
      U32.from_nat(Nat.add(range_offset(model,work,image,block,first,1n+last),Nat.mul(index,work_groups(work))))) ==
    dv(source,Kernel.bias_address(range_context(work,image,block),Nat.add(first,index))) : Tile.Tile8}:
  Kernel.Work{+n,+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+g} = work
  +groups = Kernel.groups(o)
  +base = Nat.mul(image,Nat.mul(rows,columns))
  +start = Nat.add(base,first)
  +quotient = Nat.div(Nat.mul(block,8n),8n)
  +address = Nat.add(Nat.mul(start,groups),quotient)
  +span = Nat.add(Nat.mul(last,groups),1n)
  +capacity = Machine.capacity(limits)
  +first_word = U32.from_nat(address)
  +span_word = range_span(work,image,block,1n+last)
  +quotient_below = {%Equal.sym(Nat,quotient,block,Division.exact_quotient(block,8n,{==})) : {Nat.is_lt(_,groups) == True{} : Bool}
    block_inside : {Nat.is_lt(quotient,groups) == True{} : Bool}}
  +ends = Order.transitive(Nat.add(address,span),Nat.mul(Nat.add(start,1n+last),groups),high,span_fits(start,groups,quotient,last,quotient_below),end_room)
  +ends_capacity = Order.transitive(Nat.add(address,span),high,U32.to_nat(capacity),ends,bounded)
  +address_room = Order.transitive(address,Nat.add(address,span),U32.to_nat(capacity),Order.le_add(address,span),ends_capacity)
  +span_room = Order.transitive(span,Nat.add(address,span),U32.to_nat(capacity),Order.right_le(address,span),ends_capacity)
  +end_base = Nat.add(start,1n+last)
  +end_positive = {%Equal.sym(Nat,Nat.add(start,1n+last),1n+Nat.add(start,last),Addition.successor_right(start,last)) : {Nat.is_lt(0n,_) == True{} : Bool}
    {==} : {Nat.is_lt(0n,Nat.add(start,1n+last)) == True{} : Bool}}
  +groups_room = Order.transitive(groups,Nat.mul(end_base,groups),U32.to_nat(capacity),
    {%RowMajor.multiply_commutative(groups,end_base) : {Nat.is_le(groups,_) == True{} : Bool}
      RowMajor.product_covers(groups,end_base,end_positive) : {Nat.is_le(groups,Nat.mul(end_base,groups)) == True{} : Bool}},
    Order.transitive(Nat.mul(end_base,groups),high,U32.to_nat(capacity),end_room,bounded))
  +first_exact = FullRange.roundtrip(address,capacity,address_room)
  +span_counted = span_exact(capacity,Limits.capacity_fits(limits),last,groups,Order.positive_bound(block,groups,block_inside),groups_room,span_room)
  +fits = {%Equal.sym(Nat,U32.to_nat(first_word),address,first_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(span_word)),high) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(span_word),span,span_counted) : {Nat.is_le(Nat.add(address,_),high) == True{} : Bool}
    ends : {Nat.is_le(Nat.add(U32.to_nat(first_word),U32.to_nat(span_word)),high) == True{} : Bool}}
  +less = {%Equal.sym(Nat,U32.to_nat(span_word),span,span_counted) : {Nat.is_lt(Nat.mul(index,groups),_) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.mul(last,groups),1n),1n+Nat.mul(last,groups),Addition.successor_width(Nat.mul(last,groups))) : {Nat.is_lt(Nat.mul(index,groups),_) == True{} : Bool}
    Order.lt_succ(Nat.mul(index,groups),Nat.mul(last,groups),RowMajor.mul_le(index,last,groups,Order.strict_follow(index,1n+last,inside))) :
    {Nat.is_lt(Nat.mul(index,groups),U32.to_nat(span_word)) == True{} : Bool}}
  +read = Held.band_model_value(~Tile.Tile8,~D,~dv,source,model,first_word,span_word,Tile.filled(0.0),Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0)),Held.fresh_spare_room(~Tile.Tile8,Tile.filled(0.0)),0n,high,capacity,Limits.capacity_fits(limits),bounded,tiled,held_values,
    Order.zero_le(U32.to_nat(first_word)),fits,Arithmetic.count_within(span_word,span,capacity,1073741824,span_counted,span_room,Limits.capacity_allocatable(limits)),Nat.mul(index,groups),less)
  +target = Nat.add(Nat.mul(Nat.add(base,Nat.add(first,index)),groups),quotient)
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,range_model(model,work,image,block,first,1n+last)),
      U32.from_nat(Nat.add(range_offset(model,work,image,block,first,1n+last),Nat.mul(index,groups)))),
    dv(source,U32.from_nat(Nat.add(U32.to_nat(first_word),Nat.mul(index,groups)))),dv(source,Kernel.bias_address(range_context(work,image,block),Nat.add(first,index))),read,
    {%Equal.sym(Nat,U32.to_nat(first_word),address,first_exact) :
        {dv(source,U32.from_nat(Nat.add(_,Nat.mul(index,groups)))) == dv(source,Kernel.bias_address(range_context(work,image,block),Nat.add(first,index))) : Tile.Tile8}
      %Equal.sym(Nat,Nat.add(address,Nat.mul(index,groups)),target,cell_shift(base,first,index,groups,quotient)) :
        {dv(source,U32.from_nat(_)) == dv(source,Kernel.bias_address(range_context(work,image,block),Nat.add(first,index))) : Tile.Tile8}
      {==} : {dv(source,U32.from_nat(Nat.add(U32.to_nat(first_word),Nat.mul(index,groups)))) == dv(source,Kernel.bias_address(range_context(work,image,block),Nat.add(first,index))) : Tile.Tile8}})

# ---- The fold of a range ----

# The fold over the model of a band: read index lies index steps after offset.
def fold_total(+tree: Tree.Tree<Tile.Tile8>,+offset: Nat,+step: Nat,remaining: Nat,+index: Nat,total: Tile.Tile8) -> Tile.Tile8:
  match remaining:
    case 0n: total
    case 1n+rest: fold_total(tree,offset,step,rest,1n+index,Tile.plus(total,Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(offset,Nat.mul(index,step))))))

def next_address(+offset: Nat,+step: Nat,+index: Nat) ->
  {(U32.from_nat(Nat.add(offset,Nat.mul(index,step))) + U32.from_nat(step) : U32) == U32.from_nat(Nat.add(offset,Nat.mul(1n+index,step))) : U32}:
  +moved = Nat.mul(index,step)
  %Equal.trans(Nat,Nat.add(Nat.add(offset,moved),step),Nat.add(offset,Nat.add(moved,step)),Nat.add(offset,Nat.mul(1n+index,step)),
    Addition.associative(offset,moved,step),
    Equal.cong(Nat,Nat,value => Nat.add(offset,value),Nat.add(moved,step),Nat.mul(1n+index,step),Order.commutative(moved,step))) :
    {(U32.from_nat(Nat.add(offset,moved)) + U32.from_nat(step) : U32) == U32.from_nat(_) : U32}
  WordArithmetic.native_add(Nat.add(offset,moved),step)

# The loop of a fold on a buffer reads it and keeps it.
def fold_loop(-lent: Move.Lent<Tile.Tile8>,+model: Buffer.Model<Tile.Tile8>,+offset: Nat,+step: Nat,remaining: Nat,+index: Nat,+total: Tile.Tile8) ->
  {Kernel.folded(lent,Loop.run(~Kernel.Folding,~Kernel.fold_step,remaining,Kernel.Folding{Buffer.pack(~Tile.Tile8,model),U32.from_nat(Nat.add(offset,Nat.mul(index,step))),U32.from_nat(step),total})) ==
    (Move.returned_holding(~Tile.Tile8,lent,Buffer.pack(~Tile.Tile8,model)),fold_total(Buffer.tree(~Tile.Tile8,model),offset,step,remaining,index,total)) : Move.Holding<Tile.Tile8> & Tile.Tile8}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +address = U32.from_nat(Nat.add(offset,Nat.mul(index,step)))
      +value = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,model),address)
      %Equal.sym(Storage.Buffer<Tile.Tile8> & Tile.Tile8,Storage.read(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model),address),(Buffer.pack(~Tile.Tile8,model),value),Buffer.read(~Tile.Tile8,model,address)) :
        {Kernel.folded(lent,Loop.run(~Kernel.Folding,~Kernel.fold_step,rest,Kernel.fold_added(address,U32.from_nat(step),total,_))) ==
          (Move.returned_holding(~Tile.Tile8,lent,Buffer.pack(~Tile.Tile8,model)),fold_total(Buffer.tree(~Tile.Tile8,model),offset,step,rest,1n+index,Tile.plus(total,value))) : Move.Holding<Tile.Tile8> & Tile.Tile8}
      %Equal.sym(U32,(address + U32.from_nat(step) : U32),U32.from_nat(Nat.add(offset,Nat.mul(1n+index,step))),next_address(offset,step,index)) :
        {Kernel.folded(lent,Loop.run(~Kernel.Folding,~Kernel.fold_step,rest,Kernel.Folding{Buffer.pack(~Tile.Tile8,model),_,U32.from_nat(step),Tile.plus(total,value)})) ==
          (Move.returned_holding(~Tile.Tile8,lent,Buffer.pack(~Tile.Tile8,model)),fold_total(Buffer.tree(~Tile.Tile8,model),offset,step,rest,1n+index,Tile.plus(total,value))) : Move.Holding<Tile.Tile8> & Tile.Tile8}
      fold_loop(lent,model,offset,step,rest,1n+index,Tile.plus(total,value))

# The fold over the band of a range is the fold over the cells the holder
# holds.
def fold_cells(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+model: Held.Holding<Tile.Tile8>,+high: Nat,+limits: Machine.Limits,
  +bounded: {Nat.is_le(high,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,model,0n,high),+held_values: Held.HoldingValues(~Tile.Tile8,~D,~dv,source,model),
  +work: Kernel.Work,+image: Nat,+block: Nat,+first: Nat,+last: Nat,
  +block_inside: {Nat.is_lt(block,work_groups(work)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.mul(Nat.add(Nat.add(Nat.mul(image,work_positions(work)),first),1n+last),work_groups(work)),high) == True{} : Bool},
  remaining: Nat,+index: Nat,+total: Tile.Tile8,+room: {Nat.is_le(Nat.add(index,remaining),1n+last) == True{} : Bool}) ->
  {fold_total(Buffer.tree(~Tile.Tile8,range_model(model,work,image,block,first,1n+last)),range_offset(model,work,image,block,first,1n+last),work_groups(work),remaining,index,total) ==
    held_fold(~D,~dv,source,range_context(work,image,block),remaining,Nat.add(first,index),total) : Tile.Tile8}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +context = range_context(work,image,block)
      +tree = Buffer.tree(~Tile.Tile8,range_model(model,work,image,block,first,1n+last))
      +offset = range_offset(model,work,image,block,first,1n+last)
      +groups = work_groups(work)
      +value = Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(offset,Nat.mul(index,groups))))
      %Addition.successor_right(first,index) :
        {fold_total(tree,offset,groups,rest,1n+index,Tile.plus(total,value)) ==
          held_fold(~D,~dv,source,context,rest,_,Tile.plus(total,dv(source,Kernel.bias_address(context,Nat.add(first,index))))) : Tile.Tile8}
      %band_cell(~D,~dv,source,model,high,limits,bounded,tiled,held_values,work,image,block,first,last,block_inside,end_room,index,Order.room_index(index,rest,1n+last,room)) :
        {fold_total(tree,offset,groups,rest,1n+index,Tile.plus(total,value)) == held_fold(~D,~dv,source,context,rest,Nat.add(first,1n+index),Tile.plus(total,_)) : Tile.Tile8}
      fold_cells(~D,~dv,source,model,high,limits,bounded,tiled,held_values,work,image,block,first,last,block_inside,end_room,rest,1n+index,Tile.plus(total,value),
        Order.advance_room(index,rest,1n+last,room))

# ---- A holding that keeps covering ----

# What the bias needs of the holding of the gradient: it holds every cell of
# the images from cell 0, below the capacity.
type Gradient<-work: Kernel.Work,-high: Nat> is Data:
  Gradient{limits: Machine.Limits,bounded: {Nat.is_le(high,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    cells: {Nat.is_le(Nat.mul(Nat.mul(work_images(work),work_positions(work)),work_groups(work)),high) == True{} : Bool},
    columns_positive: {Nat.is_lt(0n,work_columns(work)) == True{} : Bool},group_positive: {Nat.is_lt(0n,work_group(work)) == True{} : Bool}}

def gradient_limits(-work: Kernel.Work,-high: Nat,+ready: Gradient<work,high>) -> Machine.Limits:
  Gradient{limits,bounded,cells,columns_positive,group_positive} = ready
  limits

def gradient_bounded(-work: Kernel.Work,-high: Nat,+ready: Gradient<work,high>) ->
  {Nat.is_le(high,U32.to_nat(Machine.capacity(gradient_limits(work,high,ready)))) == True{} : Bool}:
  Gradient{limits,bounded,cells,columns_positive,group_positive} = ready
  bounded

def gradient_cells(-work: Kernel.Work,-high: Nat,+ready: Gradient<work,high>) ->
  {Nat.is_le(Nat.mul(Nat.mul(work_images(work),work_positions(work)),work_groups(work)),high) == True{} : Bool}:
  Gradient{limits,bounded,cells,columns_positive,group_positive} = ready
  cells

def gradient_columns(-work: Kernel.Work,-high: Nat,+ready: Gradient<work,high>) -> {Nat.is_lt(0n,work_columns(work)) == True{} : Bool}:
  Gradient{limits,bounded,cells,columns_positive,group_positive} = ready
  columns_positive

def gradient_group(-work: Kernel.Work,-high: Nat,+ready: Gradient<work,high>) -> {Nat.is_lt(0n,work_group(work)) == True{} : Bool}:
  Gradient{limits,bounded,cells,columns_positive,group_positive} = ready
  group_positive

# A step that gives back a holding that still covers the gradient, with the
# value of the model.
type Kept<-D: Data,-dv: D -> U32 -> Tile.Tile8,-source: D,-high: Nat,-layout: Held.Spans,-Value: Type,-result: Move.Holding<Tile.Tile8> & Value,-expected: Value> is Data:
  Kept{covered: Held.Covered<Tile.Tile8,D,dv,source,Pair.fst(Move.Holding<Tile.Tile8>,Value,result),0n,high,layout>,
    split: {result == (Pair.fst(Move.Holding<Tile.Tile8>,Value,result),expected) : Move.Holding<Tile.Tile8> & Value}}

def kept_covered(~D: Data,~dv: D -> U32 -> Tile.Tile8,~Value: Type,-source: D,-high: Nat,-layout: Held.Spans,-result: Move.Holding<Tile.Tile8> & Value,-expected: Value,
  +kept: Kept<D,dv,source,high,layout,Value,result,expected>) -> Held.Covered<Tile.Tile8,D,dv,source,Pair.fst(Move.Holding<Tile.Tile8>,Value,result),0n,high,layout>:
  Kept{covered,split} = kept
  covered

def kept_split(~D: Data,~dv: D -> U32 -> Tile.Tile8,~Value: Type,-source: D,-high: Nat,-layout: Held.Spans,-result: Move.Holding<Tile.Tile8> & Value,-expected: Value,
  +kept: Kept<D,dv,source,high,layout,Value,result,expected>) -> {result == (Pair.fst(Move.Holding<Tile.Tile8>,Value,result),expected) : Move.Holding<Tile.Tile8> & Value}:
  Kept{covered,split} = kept
  split

# ---- The runtime over the holding ----

def banded_eta(+step: U32,+count: Nat,banded: Move.Band<Tile.Tile8>) ->
  {Kernel.fold_banded(step,count,banded) ==
    Kernel.folded(Held.band_loan(~Tile.Tile8,banded),Loop.run(~Kernel.Folding,~Kernel.fold_step,count,Kernel.Folding{Held.band_buffer(~Tile.Tile8,banded),Held.band_offset(~Tile.Tile8,banded),step,Tile.filled(0.0)})) :
    Move.Holding<Tile.Tile8> & Tile.Tile8}:
  Move.Band{lent,band,offset} = banded
  {==}

def start_word(+offset: U32) -> {U32.from_nat(Nat.add(U32.to_nat(offset),0n)) == offset : U32}:
  %Equal.sym(Nat,Nat.add(U32.to_nat(offset),0n),U32.to_nat(offset),Addition.zero_right(U32.to_nat(offset))) : {U32.from_nat(_) == offset : U32}
  Arithmetic.native_roundtrip(offset)

def step_word(+work: Kernel.Work,+image: Nat,+element: Nat) -> {Kernel.cell_step(Kernel.Context{work,image,element}) == U32.from_nat(work_groups(work)) : U32}:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  {==}

# A range lends the band of its cells, folds them and gives the band back.
def fold_done(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+high: Nat,+layout: Held.Spans,+ready: Gradient<work,high>,
  +image: Nat,+block: Nat,+first: Nat,+count: Nat,
  +image_inside: {Nat.is_lt(image,work_images(work)) == True{} : Bool},+block_inside: {Nat.is_lt(block,work_groups(work)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,count) == True{} : Bool},+range_room: {Nat.is_le(Nat.add(first,count),work_positions(work)) == True{} : Bool},
  -reader: Move.Holding<Tile.Tile8>,covered: Held.Covered<Tile.Tile8,D,dv,source,reader,0n,high,layout>) ->
  Kept<D,dv,source,high,layout,Tile.Tile8,Kernel.fold(range_context(work,image,block),Grouped.Range{first,count},reader),
    held_fold(~D,~dv,source,range_context(work,image,block),count,first,Tile.filled(0.0))>:
  match count:
    case 0n:
      Empty.absurd(Kept<D,dv,source,high,layout,Tile.Tile8,Kernel.fold(range_context(work,image,block),Grouped.Range{first,0n},reader),
        held_fold(~D,~dv,source,range_context(work,image,block),0n,first,Tile.filled(0.0))>,WordEncoding.false_true(positive))
    case 1n+ +last:
      Held.Covered{+model,+packed,+tiled,+held_values,+ordered} = covered
      +limits = gradient_limits(work,high,ready)
      +bounded = gradient_bounded(work,high,ready)
      +end_room = range_end(work_images(work),work_positions(work),work_groups(work),image,first,1n+last,high,image_inside,range_room,gradient_cells(work,high,ready))
      +context = range_context(work,image,block)
      +step = Kernel.cell_step(context)
      +groups = work_groups(work)
      +bm = range_model(model,work,image,block,first,1n+last)
      +off = range_offset(model,work,image,block,first,1n+last)
      +expected = held_fold(~D,~dv,source,context,1n+last,first,Tile.filled(0.0))
      +total = fold_total(Buffer.tree(~Tile.Tile8,bm),off,groups,1n+last,0n,Tile.filled(0.0))
      %Equal.sym(Move.Holding<Tile.Tile8>,reader,Held.pack(~Tile.Tile8,model),packed) :
        Kept<D,dv,source,high,layout,Tile.Tile8,Kernel.fold(context,Grouped.Range{first,1n+last},_),expected>
      %Equal.sym(Move.Spare<Tile.Tile8>,Move.spare(~Tile.Tile8,Tile.filled(0.0)),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))),Held.spare_packed(~Tile.Tile8,Tile.filled(0.0))) :
        Kept<D,dv,source,high,layout,Tile.Tile8,Kernel.fold_banded(step,1n+last,Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,model),Kernel.bias_address(context,first),range_span(work,image,block,1n+last),
          Tile.filled(0.0),_)),expected>
      %Equal.sym(Move.Holding<Tile.Tile8> & Tile.Tile8,Kernel.fold_banded(step,1n+last,held_band(model,work,image,block,first,1n+last)),
          Kernel.folded(Held.band_loan(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)),Loop.run(~Kernel.Folding,~Kernel.fold_step,1n+last,
            Kernel.Folding{Held.band_buffer(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)),Held.band_offset(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)),step,Tile.filled(0.0)})),
          banded_eta(step,1n+last,held_band(model,work,image,block,first,1n+last))) :
        Kept<D,dv,source,high,layout,Tile.Tile8,_,expected>
      %Buffer.remodeled(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,held_band(model,work,image,block,first,1n+last))) :
        Kept<D,dv,source,high,layout,Tile.Tile8,Kernel.folded(Held.band_loan(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)),Loop.run(~Kernel.Folding,~Kernel.fold_step,1n+last,
          Kernel.Folding{_,Held.band_offset(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)),step,Tile.filled(0.0)})),expected>
      %start_word(Held.band_offset(~Tile.Tile8,held_band(model,work,image,block,first,1n+last))) :
        Kept<D,dv,source,high,layout,Tile.Tile8,Kernel.folded(Held.band_loan(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)),Loop.run(~Kernel.Folding,~Kernel.fold_step,1n+last,
          Kernel.Folding{Buffer.pack(~Tile.Tile8,bm),_,step,Tile.filled(0.0)})),expected>
      %Equal.sym(U32,step,U32.from_nat(groups),step_word(work,image,Nat.mul(block,8n))) :
        Kept<D,dv,source,high,layout,Tile.Tile8,Kernel.folded(Held.band_loan(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)),Loop.run(~Kernel.Folding,~Kernel.fold_step,1n+last,
          Kernel.Folding{Buffer.pack(~Tile.Tile8,bm),U32.from_nat(Nat.add(off,0n)),_,Tile.filled(0.0)})),expected>
      %Equal.sym(Move.Holding<Tile.Tile8> & Tile.Tile8,
          Kernel.folded(Held.band_loan(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)),Loop.run(~Kernel.Folding,~Kernel.fold_step,1n+last,
            Kernel.Folding{Buffer.pack(~Tile.Tile8,bm),U32.from_nat(Nat.add(off,Nat.mul(0n,groups))),U32.from_nat(groups),Tile.filled(0.0)})),
          (Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)),Buffer.pack(~Tile.Tile8,bm)),total),
          fold_loop(Held.band_loan(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)),bm,off,groups,1n+last,0n,Tile.filled(0.0))) :
        Kept<D,dv,source,high,layout,Tile.Tile8,_,expected>
      Kept{Held.band_returned(~Tile.Tile8,~D,~dv,source,model,Kernel.bias_address(context,first),range_span(work,image,block,1n+last),Tile.filled(0.0),Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0)),0n,high,tiled,held_values,layout,ordered,bm,
          Buffer.remodel_content(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)))),
        Equal.cong(Tile.Tile8,Move.Holding<Tile.Tile8> & Tile.Tile8,value => (Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,held_band(model,work,image,block,first,1n+last)),Buffer.pack(~Tile.Tile8,bm)),value),
          total,expected,
          Equal.trans(Tile.Tile8,total,held_fold(~D,~dv,source,context,1n+last,Nat.add(first,0n),Tile.filled(0.0)),expected,
            fold_cells(~D,~dv,source,model,high,limits,bounded,tiled,held_values,work,image,block,first,last,block_inside,end_room,1n+last,0n,Tile.filled(0.0),Order.reflexive(1n+last)),
            Equal.cong(Nat,Tile.Tile8,position => held_fold(~D,~dv,source,context,1n+last,position,Tile.filled(0.0)),Nat.add(first,0n),first,Addition.zero_right(first))))}

# The ranges of an image, each folded over its band.
def ranges_done(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+high: Nat,+layout: Held.Spans,+ready: Gradient<work,high>,
  +image: Nat,+block: Nat,+image_inside: {Nat.is_lt(image,work_images(work)) == True{} : Bool},+block_inside: {Nat.is_lt(block,work_groups(work)) == True{} : Bool},
  ranges: List<&2,Grouped.Range>,
  +bounded: Members.All(Grouped.Range,{range => Reduction.RangeBounded(work_rows(work),work_columns(work),work_group(work),range) : Grouped.Range -> Data},ranges),
  -reader: Move.Holding<Tile.Tile8>,covered: Held.Covered<Tile.Tile8,D,dv,source,reader,0n,high,layout>) ->
  Kept<D,dv,source,high,layout,List<&2,Tile.Tile8>,Kernel.fold_ranges(range_context(work,image,block),ranges,reader),group_totals(~D,~dv,source,range_context(work,image,block),ranges)>:
  match ranges:
    case []: Kept{covered,{==}}
    case Grouped.Range{+first,+count} <> +rest:
      KernelGemm.Both{+head,+tail} = bounded
      KernelGemm.Both{+positive,KernelGemm.Both{+most,+room}} = head
      +context = range_context(work,image,block)
      +total = held_fold(~D,~dv,source,context,count,first,Tile.filled(0.0))
      +totals = group_totals(~D,~dv,source,context,rest)
      +folded = fold_done(~D,~dv,source,work,high,layout,ready,image,block,first,count,image_inside,block_inside,positive,room,reader,covered)
      +others = ranges_done(~D,~dv,source,work,high,layout,ready,image,block,image_inside,block_inside,rest,tail,
        Pair.fst(Move.Holding<Tile.Tile8>,Tile.Tile8,Kernel.fold(context,Grouped.Range{first,count},reader)),
        kept_covered(~D,~dv,~Tile.Tile8,source,high,layout,Kernel.fold(context,Grouped.Range{first,count},reader),total,folded))
      %Equal.sym(Move.Holding<Tile.Tile8> & Tile.Tile8,Kernel.fold(context,Grouped.Range{first,count},reader),
          (Pair.fst(Move.Holding<Tile.Tile8>,Tile.Tile8,Kernel.fold(context,Grouped.Range{first,count},reader)),total),
          kept_split(~D,~dv,~Tile.Tile8,source,high,layout,Kernel.fold(context,Grouped.Range{first,count},reader),total,folded)) :
        Kept<D,dv,source,high,layout,List<&2,Tile.Tile8>,Kernel.group_tail(_,reader => Kernel.fold_ranges(context,rest,reader)),total <> totals>
      %Equal.sym(Move.Holding<Tile.Tile8> & List<&2,Tile.Tile8>,
          Kernel.fold_ranges(context,rest,Pair.fst(Move.Holding<Tile.Tile8>,Tile.Tile8,Kernel.fold(context,Grouped.Range{first,count},reader))),
          (Pair.fst(Move.Holding<Tile.Tile8>,List<&2,Tile.Tile8>,Kernel.fold_ranges(context,rest,Pair.fst(Move.Holding<Tile.Tile8>,Tile.Tile8,Kernel.fold(context,Grouped.Range{first,count},reader)))),totals),
          kept_split(~D,~dv,~List<&2,Tile.Tile8>,source,high,layout,
            Kernel.fold_ranges(context,rest,Pair.fst(Move.Holding<Tile.Tile8>,Tile.Tile8,Kernel.fold(context,Grouped.Range{first,count},reader))),totals,others)) :
        Kept<D,dv,source,high,layout,List<&2,Tile.Tile8>,Kernel.prepended(total,_),total <> totals>
      Kept{kept_covered(~D,~dv,~List<&2,Tile.Tile8>,source,high,layout,
          Kernel.fold_ranges(context,rest,Pair.fst(Move.Holding<Tile.Tile8>,Tile.Tile8,Kernel.fold(context,Grouped.Range{first,count},reader))),totals,others),{==}}

# The images from image on: each folds its ranges and takes their tree.
def images_done(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+high: Nat,+layout: Held.Spans,+ready: Gradient<work,high>,
  +block: Nat,+block_inside: {Nat.is_lt(block,work_groups(work)) == True{} : Bool},
  remaining: Nat,+image: Nat,+room: {Nat.is_le(Nat.add(image,remaining),work_images(work)) == True{} : Bool},
  -reader: Move.Holding<Tile.Tile8>,covered: Held.Covered<Tile.Tile8,D,dv,source,reader,0n,high,layout>) ->
  Kept<D,dv,source,high,layout,List<&2,Tile.Tile8>,Kernel.image_values(remaining,image,Nat.mul(block,8n),work,work_ranges(work),reader),
    image_totals(~D,~dv,remaining,image,Nat.mul(block,8n),work,work_ranges(work),source)>:
  match remaining:
    case 0n: Kept{covered,{==}}
    case 1n+ +rest:
      +ranges = work_ranges(work)
      +context = range_context(work,image,block)
      +fuel = List.length(&2,Grouped.Range,ranges)
      +totals = group_totals(~D,~dv,source,context,ranges)
      +later = image_totals(~D,~dv,rest,1n+image,Nat.mul(block,8n),work,ranges,source)
      +folds = ranges_done(~D,~dv,source,work,high,layout,ready,image,block,Order.room_index(image,rest,work_images(work),room),block_inside,ranges,
        Reduction.ranges_bounded(work_rows(work),work_columns(work),work_group(work),gradient_columns(work,high,ready),gradient_group(work,high,ready)),reader,covered)
      +next = images_done(~D,~dv,source,work,high,layout,ready,block,block_inside,rest,1n+image,Order.advance_room(image,rest,work_images(work),room),
        Pair.fst(Move.Holding<Tile.Tile8>,List<&2,Tile.Tile8>,Kernel.fold_ranges(context,ranges,reader)),
        kept_covered(~D,~dv,~List<&2,Tile.Tile8>,source,high,layout,Kernel.fold_ranges(context,ranges,reader),totals,folds))
      %Equal.sym(Move.Holding<Tile.Tile8> & List<&2,Tile.Tile8>,Kernel.fold_ranges(context,ranges,reader),
          (Pair.fst(Move.Holding<Tile.Tile8>,List<&2,Tile.Tile8>,Kernel.fold_ranges(context,ranges,reader)),totals),
          kept_split(~D,~dv,~List<&2,Tile.Tile8>,source,high,layout,Kernel.fold_ranges(context,ranges,reader),totals,folds)) :
        Kept<D,dv,source,high,layout,List<&2,Tile.Tile8>,
          Kernel.group_tail(Kernel.reduced(fuel,_),reader => Kernel.image_values(rest,1n+image,Nat.mul(block,8n),work,ranges,reader)),Kernel.tree(fuel,totals) <> later>
      %Equal.sym(Move.Holding<Tile.Tile8> & List<&2,Tile.Tile8>,
          Kernel.image_values(rest,1n+image,Nat.mul(block,8n),work,ranges,Pair.fst(Move.Holding<Tile.Tile8>,List<&2,Tile.Tile8>,Kernel.fold_ranges(context,ranges,reader))),
          (Pair.fst(Move.Holding<Tile.Tile8>,List<&2,Tile.Tile8>,
            Kernel.image_values(rest,1n+image,Nat.mul(block,8n),work,ranges,Pair.fst(Move.Holding<Tile.Tile8>,List<&2,Tile.Tile8>,Kernel.fold_ranges(context,ranges,reader)))),later),
          kept_split(~D,~dv,~List<&2,Tile.Tile8>,source,high,layout,
            Kernel.image_values(rest,1n+image,Nat.mul(block,8n),work,ranges,Pair.fst(Move.Holding<Tile.Tile8>,List<&2,Tile.Tile8>,Kernel.fold_ranges(context,ranges,reader))),later,next)) :
        Kept<D,dv,source,high,layout,List<&2,Tile.Tile8>,Kernel.prepended(Kernel.tree(fuel,totals),_),Kernel.tree(fuel,totals) <> later>
      Kept{kept_covered(~D,~dv,~List<&2,Tile.Tile8>,source,high,layout,
          Kernel.image_values(rest,1n+image,Nat.mul(block,8n),work,ranges,Pair.fst(Move.Holding<Tile.Tile8>,List<&2,Tile.Tile8>,Kernel.fold_ranges(context,ranges,reader))),later,next),{==}}

# The cell of a block: the tree of its images, +0 past the outputs.
def cell_done(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+high: Nat,+layout: Held.Spans,+ready: Gradient<work,high>,
  +block: Nat,+block_inside: {Nat.is_lt(block,work_groups(work)) == True{} : Bool},
  -reader: Move.Holding<Tile.Tile8>,covered: Held.Covered<Tile.Tile8,D,dv,source,reader,0n,high,layout>) ->
  Kept<D,dv,source,high,layout,Tile.Tile8,Kernel.cell(Kernel.Block{work,block,work_ranges(work)},reader),
    cell_values(~D,~held_sample(~D,~dv),source,Kernel.Block{work,block,work_ranges(work)})>:
  Kernel.Work{+n,+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+g} = work
  +ranges = work_ranges(work)
  +totals = image_totals(~D,~dv,n,0n,Nat.mul(block,8n),work,ranges,source)
  +images = images_done(~D,~dv,source,work,high,layout,ready,block,block_inside,n,0n,Order.reflexive(n),reader,covered)
  %Equal.sym(Move.Holding<Tile.Tile8> & List<&2,Tile.Tile8>,Kernel.image_values(n,0n,Nat.mul(block,8n),work,ranges,reader),
      (Pair.fst(Move.Holding<Tile.Tile8>,List<&2,Tile.Tile8>,Kernel.image_values(n,0n,Nat.mul(block,8n),work,ranges,reader)),totals),
      kept_split(~D,~dv,~List<&2,Tile.Tile8>,source,high,layout,Kernel.image_values(n,0n,Nat.mul(block,8n),work,ranges,reader),totals,images)) :
    Kept<D,dv,source,high,layout,Tile.Tile8,Kernel.cell_kept(o,block,Kernel.reduced(n,_)),cell_values(~D,~held_sample(~D,~dv),source,Kernel.Block{work,block,ranges})>
  Kept{kept_covered(~D,~dv,~List<&2,Tile.Tile8>,source,high,layout,Kernel.image_values(n,0n,Nat.mul(block,8n),work,ranges,reader),totals,images),
    Equal.cong(Tile.Tile8,Move.Holding<Tile.Tile8> & Tile.Tile8,tile => (Pair.fst(Move.Holding<Tile.Tile8>,List<&2,Tile.Tile8>,Kernel.image_values(n,0n,Nat.mul(block,8n),work,ranges,reader)),tile),
      Kernel.kept(o,block,Kernel.tree(n,totals)),cell_values(~D,~held_sample(~D,~dv),source,Kernel.Block{work,block,ranges}),cell_equal(~D,~dv,source,work,block,ranges))}

# The cells of the blocks from index on, written from position on.
def written_done(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+high: Nat,+layout: Held.Spans,+ready: Gradient<work,high>,
  remaining: Nat,+index: Nat,+position: U32,+room: {Nat.is_le(Nat.add(index,remaining),work_groups(work)) == True{} : Bool},
  -reader: Move.Holding<Tile.Tile8>,covered: Held.Covered<Tile.Tile8,D,dv,source,reader,0n,high,layout>,+result: Buffer.Model<Tile.Tile8>) ->
  Kept<D,dv,source,high,layout,Storage.Buffer<Tile.Tile8>,Kernel.written(remaining,index,position,work,work_ranges(work),reader,Buffer.pack(~Tile.Tile8,result)),
    Buffer.pack(~Tile.Tile8,written_model(~D,~held_sample(~D,~dv),remaining,index,position,work,work_ranges(work),source,result))>:
  match remaining:
    case 0n: Kept{covered,{==}}
    case 1n+ +rest:
      +ranges = work_ranges(work)
      +block = {Kernel.Block{work,index,ranges} : Kernel.Block}
      +value = cell_values(~D,~held_sample(~D,~dv),source,block)
      +next = Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,position,value,Buffer.pack(~Tile.Tile8,result)))
      +cell = cell_done(~D,~dv,source,work,high,layout,ready,index,Order.room_index(index,rest,work_groups(work),room),reader,covered)
      %Equal.sym(Move.Holding<Tile.Tile8> & Tile.Tile8,Kernel.cell(block,reader),(Pair.fst(Move.Holding<Tile.Tile8>,Tile.Tile8,Kernel.cell(block,reader)),value),
          kept_split(~D,~dv,~Tile.Tile8,source,high,layout,Kernel.cell(block,reader),value,cell)) :
        Kept<D,dv,source,high,layout,Storage.Buffer<Tile.Tile8>,
          Kernel.stored(position,Buffer.pack(~Tile.Tile8,result),_,reader => result => Kernel.written(rest,1n+index,(position + 1 : U32),work,ranges,reader,result)),
          Buffer.pack(~Tile.Tile8,written_model(~D,~held_sample(~D,~dv),rest,1n+index,(position + 1 : U32),work,ranges,source,next))>
      %Buffer.remodeled(~Tile.Tile8,Storage.write(~Tile.Tile8,position,value,Buffer.pack(~Tile.Tile8,result))) :
        Kept<D,dv,source,high,layout,Storage.Buffer<Tile.Tile8>,
          Kernel.written(rest,1n+index,(position + 1 : U32),work,ranges,Pair.fst(Move.Holding<Tile.Tile8>,Tile.Tile8,Kernel.cell(block,reader)),_),
          Buffer.pack(~Tile.Tile8,written_model(~D,~held_sample(~D,~dv),rest,1n+index,(position + 1 : U32),work,ranges,source,next))>
      written_done(~D,~dv,source,work,high,layout,ready,rest,1n+index,(position + 1 : U32),Order.advance_room(index,rest,work_groups(work),room),
        Pair.fst(Move.Holding<Tile.Tile8>,Tile.Tile8,Kernel.cell(block,reader)),kept_covered(~D,~dv,~Tile.Tile8,source,high,layout,Kernel.cell(block,reader),value,cell),next)

# The bias over a holding that covers the gradient writes the cells of every
# block and gives back a holding that still covers it.
def bias_done(~D: Data,~dv: D -> U32 -> Tile.Tile8,+source: D,+work: Kernel.Work,+high: Nat,+layout: Held.Spans,+ready: Gradient<work,high>,
  -reader: Move.Holding<Tile.Tile8>,covered: Held.Covered<Tile.Tile8,D,dv,source,reader,0n,high,layout>,+result: Buffer.Model<Tile.Tile8>) ->
  Kept<D,dv,source,high,layout,Storage.Buffer<Tile.Tile8>,Kernel.bias(work,reader,Buffer.pack(~Tile.Tile8,result)),Buffer.pack(~Tile.Tile8,bias_written(~D,~dv,work,source,result))>:
  Kernel.Work{+n,+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+g} = work
  written_done(~D,~dv,source,work,high,layout,ready,Kernel.groups(o),0n,0,Order.reflexive(Kernel.groups(o)),reader,covered,result)

# ---- The cells the bias writes are ready ----

# The cells the bias writes keep the depth and the length of the buffer.
def written_footprint(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,remaining: Nat,+index: Nat,+position: U32,+work: Kernel.Work,
  +ranges: List<&2,Grouped.Range>,+source: Source,+result: Buffer.Model<Tile.Tile8>) ->
  {Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,written_model(~Source,~sample,remaining,index,position,work,ranges,source,result))) ==
    Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,result)) : Footprints.Footprint}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +value = cell_values(~Source,~sample,source,Kernel.Block{work,index,ranges})
      +next = Buffer.remodel(~Tile.Tile8,Storage.write(~Tile.Tile8,position,value,Buffer.pack(~Tile.Tile8,result)))
      Equal.trans(Footprints.Footprint,
        Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,written_model(~Source,~sample,rest,1n+index,(position + 1 : U32),work,ranges,source,next))),
        Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,next)),Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,result)),
        written_footprint(~Source,~sample,rest,1n+index,(position + 1 : U32),work,ranges,source,next),
        Equal.trans(Footprints.Footprint,Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,next)),
          Footprints.footprint(~Tile.Tile8,Storage.write(~Tile.Tile8,position,value,Buffer.pack(~Tile.Tile8,result))),Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,result)),
          Equal.cong(Storage.Buffer<Tile.Tile8>,Footprints.Footprint,buffer => Footprints.footprint(~Tile.Tile8,buffer),Buffer.pack(~Tile.Tile8,next),
            Storage.write(~Tile.Tile8,position,value,Buffer.pack(~Tile.Tile8,result)),Buffer.remodeled(~Tile.Tile8,Storage.write(~Tile.Tile8,position,value,Buffer.pack(~Tile.Tile8,result)))),
          Footprints.write_footprint(~Tile.Tile8,position,value,result)))

def bias_footprint(~D: Data,~dv: D -> U32 -> Tile.Tile8,+work: Kernel.Work,+source: D,+result: Buffer.Model<Tile.Tile8>) ->
  {Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias_written(~D,~dv,work,source,result))) == Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,result)) : Footprints.Footprint}:
  Kernel.Work{n,c,+o,h,w,k,s,p,+rows,+columns,+g} = work
  written_footprint(~D,~held_sample(~D,~dv),Kernel.groups(o),0n,0,work,Grouped.ranges(rows,columns,g),source,result)

# The cells the bias writes into a fresh buffer have its footprint.
def fresh_footprint(~D: Data,~dv: D -> U32 -> Tile.Tile8,+work: Kernel.Work,+source: D,+fill: Tile.Tile8,+count: U32) ->
  {Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias_written(~D,~dv,work,source,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))))) ==
    Footprints.footprint(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)) : Footprints.Footprint}:
  Equal.trans(Footprints.Footprint,
    Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias_written(~D,~dv,work,source,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))))),
    Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)))),
    Footprints.footprint(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)),
    bias_footprint(~D,~dv,work,source,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))),
    Equal.cong(Storage.Buffer<Tile.Tile8>,Footprints.Footprint,buffer => Footprints.footprint(~Tile.Tile8,buffer),
      Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))),Tensors.buffer(~Tile.Tile8,fill,count),
      Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))))

# The cells the bias writes into a fresh buffer of a count within 2^30 are
# ready.
def bias_ready(~D: Data,~dv: D -> U32 -> Tile.Tile8,+work: Kernel.Work,+source: D,+fill: Tile.Tile8,+count: U32,+within: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Buffer.ContentReady<Tile.Tile8,Buffer.content(~Tile.Tile8,bias_written(~D,~dv,work,source,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))))>:
  AllocationReady.bounded(~Tile.Tile8,fill,count,bias_written(~D,~dv,work,source,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))),
    fresh_footprint(~D,~dv,work,source,fill,count),within)

# The cells the bias writes into a fresh buffer are as many as it has.
def bias_length(~D: Data,~dv: D -> U32 -> Tile.Tile8,+work: Kernel.Work,+source: D,+fill: Tile.Tile8,+count: U32) ->
  {Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,bias_written(~D,~dv,work,source,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))))) ==
    U32.to_nat(count) : Nat}:
  +written = bias_written(~D,~dv,work,source,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)))
  +allocated = {Footprints.Footprint{Tensors.depth_for(count),U32.to_nat(count)} : Footprints.Footprint}
  Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,written)),Buffer.length(~Tile.Tile8,written),U32.to_nat(count),
    Buffer.content_length_of(~Tile.Tile8,written),
    Equal.trans(Nat,Buffer.length(~Tile.Tile8,written),Footprints.length(Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,written))),U32.to_nat(count),
      Equal.sym(Nat,Footprints.length(Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,written))),Buffer.length(~Tile.Tile8,written),
        Footprints.footprint_length(~Tile.Tile8,written)),
      Equal.cong(Footprints.Footprint,Nat,footprint => Footprints.length(footprint),Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,written)),allocated,
        Equal.trans(Footprints.Footprint,Footprints.footprint(~Tile.Tile8,Buffer.pack(~Tile.Tile8,written)),Footprints.footprint(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)),
          allocated,fresh_footprint(~D,~dv,work,source,fill,count),Footprints.allocated(~Tile.Tile8,fill,count)))))

# A block of the cells the bias writes into a fresh buffer of its groups holds
# the cell of the bias model.
def bias_value(~D: Data,~dv: D -> U32 -> Tile.Tile8,+work: Kernel.Work,+source: D,+fill: Tile.Tile8,+count: U32,+within: {U32.is_le(count,1073741824) == True{} : Bool},
  +groups_exact: {work_groups(work) == U32.to_nat(count) : Nat},+block: Nat,+inside: {Nat.is_lt(block,U32.to_nat(count)) == True{} : Bool}) ->
  {Buffer.value(~Tile.Tile8,U32.from_nat(block),Buffer.content(~Tile.Tile8,bias_written(~D,~dv,work,source,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))))) ==
    cell_values(~D,~held_sample(~D,~dv),source,Kernel.Block{work,block,work_ranges(work)}) : Tile.Tile8}:
  Kernel.Work{+n,+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+g} = work
  +start = Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))
  +ranges = Grouped.ranges(rows,columns,g)
  +values = bias_values(~D,~dv,work,source)
  +groups = Kernel.groups(o)
  +tree = Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,start))
  +length = Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,start))
  +cell = cell_values(~D,~held_sample(~D,~dv),source,Kernel.Block{work,block,ranges})
  +ready = {%Buffer.content_eta(~Tile.Tile8,Buffer.content(~Tile.Tile8,start)) : Buffer.ContentReady<Tile.Tile8,_>
    AllocationReady.bounded(~Tile.Tile8,fill,count,start,
      Equal.cong(Storage.Buffer<Tile.Tile8>,Footprints.Footprint,buffer => Footprints.footprint(~Tile.Tile8,buffer),Buffer.pack(~Tile.Tile8,start),Tensors.buffer(~Tile.Tile8,fill,count),
        Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))),within) :
    Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>}
  +counted = Equal.trans(Nat,List.length(&2,Tile.Tile8,values),groups,U32.to_nat(count),
    values_length(~D,~held_sample(~D,~dv),groups,0n,work,ranges,source),groups_exact)
  +fits = {%Equal.sym(Nat,List.length(&2,Tile.Tile8,values),U32.to_nat(count),counted) : {Nat.is_le(_,length) == True{} : Bool}
    %Equal.sym(Nat,length,U32.to_nat(count),Footprints.allocated_length(~Tile.Tile8,fill,count)) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool}
    Order.reflexive(U32.to_nat(count)) : {Nat.is_le(List.length(&2,Tile.Tile8,values),length) == True{} : Bool}}
  +listed = {%Equal.sym(Nat,List.length(&2,Tile.Tile8,values),U32.to_nat(count),counted) : {Nat.is_lt(block,_) == True{} : Bool} inside :
    {Nat.is_lt(block,List.length(&2,Tile.Tile8,values)) == True{} : Bool}}
  +below = {%Equal.sym(Nat,groups,U32.to_nat(count),groups_exact) : {Nat.is_lt(block,_) == True{} : Bool} inside : {Nat.is_lt(block,groups) == True{} : Bool}}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,bias_written(~D,~dv,work,source,start)),Buffer.Content{Tree.write_sequence(~Tile.Tile8,values,0,tree),length},
      bias_content(~D,~dv,work,source,start)) :
    {Buffer.value(~Tile.Tile8,U32.from_nat(block),_) == cell : Tile.Tile8}
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Tree.write_sequence(~Tile.Tile8,values,0,tree),U32.from_nat(block)),ListRead.at(~Tile.Tile8,Tile.filled(0.0),block,values),cell,
    Listed.sequence_value(~Tile.Tile8,values,tree,length,Tile.filled(0.0),ready,fits,block,listed),
    values_at(~D,~held_sample(~D,~dv),groups,block,0n,work,ranges,source,below))
