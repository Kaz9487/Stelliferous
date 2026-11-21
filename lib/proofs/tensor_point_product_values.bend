# Readback of the position blocks that fill the staged point-product output.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_point_products.bend as Products
import ../traversal_array.bend as Traversal
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./kernel_gemm_point_buffer.bend as PointBuffer
import ./kernel_gemm_point_jobs.bend as PointJobs
import ./kernel_gemm_point_readback.bend as Readback
import ./kernel_gemm_point_refinement.bend as PointLaws
import ./kernel_gemm_output_groups.bend as Groups
import ./machine_limit_bounds.bend as Limits
import ./nat_ceiling.bend as Ceiling
import ./nat_division_order.bend as DivisionOrder
import ./nat_extrema.bend as Extrema
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as Index
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as StorageShape
import ./storage_tree_model.bend as Tree
import ./tensor_allocation_depth.bend as Depth
import ./tensor_grid_pieces.bend as GridLaws
import ./tensor_move_readback.bend as Held
import ./tensor_point_products.bend as Point
import ./u32_extrema.bend as WordExtrema
import ./u32_division.bend as U32Division
import ./word_power_of_two.bend as Power
import ./word_encoding.bend as Encoding
import ./write_program.bend as Writes

def block_rows(+positions: U32,+first: U32,+span: U32) -> U32:
  U32.min(span,(positions - first : U32))

def block_end(+positions: U32,+first: U32,+span: U32) -> U32:
  (first + block_rows(positions,first,span) : U32)

type Span<-limits: Machine.Limits,-inner: U32> is Data:
  Span{positive: {Nat.is_lt(0n,U32.to_nat(Products.block_positions(inner))) == True{} : Bool},
    room: {Nat.is_le(U32.to_nat(Products.block_positions(inner)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def budget_span(+limits: Machine.Limits,+inner: U32,+tiles: U32,
  +inner_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +four_within: {Nat.is_le(4n,U32.to_nat(tiles)) == True{} : Bool},
  +budget_room: {Nat.is_le(U32.to_nat(tiles),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Held.Both<{Nat.is_lt(0n,U32.to_nat((4 * U32.max(1,(tiles / Gemm.blocks(inner) / 4 : U32)) : U32))) == True{} : Bool},
    {Nat.is_le(U32.to_nat((4 * U32.max(1,(tiles / Gemm.blocks(inner) / 4 : U32)) : U32)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}>:
  +budget = U32.to_nat(tiles)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +groups = Gemm.blocks(inner)
  +ng = U32.to_nat(groups)
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +gp = Groups.groups_positive(limits,inner,inner_room,inner_positive)
  +outer = (tiles / groups : U32)
  +outer_nat = Nat.div(budget,ng)
  +outer_exact = U32Division.divide(tiles,groups,half,half_ok,Limits.double_bound(limits,ng,Groups.groups_fit(limits,inner,inner_room)),gp)
  +inner_exact = U32Division.quotient_exact(half,half_ok,outer,4,outer_nat,4n,outer_exact,{==},
    Limits.within_half(limits,8n,Order.transitive(8n,16n,capacity,{==},Bounds.as_nat_le(16,Machine.capacity(limits),Limits.sixteen_within_capacity(limits)))),{==})
  +clip = U32.max(1,(outer / 4 : U32))
  +width = Nat.max(1n,Nat.div(outer_nat,4n))
  +width_exact = Equal.trans(Nat,U32.to_nat(clip),Nat.max(1n,U32.to_nat((outer / 4 : U32))),width,
    WordExtrema.maximum(1,(outer / 4 : U32)),Equal.cong(Nat,Nat,n => Nat.max(1n,n),U32.to_nat((outer / 4 : U32)),Nat.div(outer_nat,4n),inner_exact))
  +width_room = Extrema.maximum_upper(1n,Nat.div(outer_nat,4n),Nat.div(budget,4n),
    DivisionOrder.quotient_le(4n,budget,4n,{==},four_within),
    DivisionOrder.quotient_le(outer_nat,budget,4n,{==},DivisionOrder.quotient_at_most(budget,ng,gp)))
  +wide = Order.transitive(Nat.mul(width,4n),Nat.mul(Nat.div(budget,4n),4n),capacity,Index.mul_le(width,Nat.div(budget,4n),4n,width_room),
    Order.transitive(Nat.mul(Nat.div(budget,4n),4n),budget,capacity,DivisionOrder.product_below(budget,4n,{==}),budget_room))
  +exact = Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),4,clip,4n,width,{==},width_exact,
    %Index.multiply_commutative(width,4n) : {Nat.is_le(_,capacity) == True{} : Bool} wide)
  +size = (4 * clip : U32)
  Held.Both{%Equal.sym(Nat,U32.to_nat(size),Nat.mul(4n,width),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
      Order.product_positive(4n,width,{==},Order.lt_le(0n,1n,width,{==},Extrema.maximum_left(1n,Nat.div(outer_nat,4n)))),
    %Equal.sym(Nat,U32.to_nat(size),Nat.mul(4n,width),exact) : {Nat.is_le(_,capacity) == True{} : Bool}
    %Index.multiply_commutative(width,4n) : {Nat.is_le(_,capacity) == True{} : Bool} wide}

def actual_span(+limits: Machine.Limits,+inner: U32,
  +checked: Held.Both<{Nat.is_lt(0n,U32.to_nat(Products.block_positions(inner))) == True{} : Bool},
    {Nat.is_le(U32.to_nat(Products.block_positions(inner)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}>) -> Span<limits,inner>:
  Held.Both{positive,room} = checked
  Span{positive,room}

def block_span(+limits: Machine.Limits,+inner: U32,
  +inner_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +budget_room: {Nat.is_le(U32.to_nat(Gemm.block_tiles()),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Span<limits,inner>:
  actual_span(limits,inner,budget_span(limits,inner,Gemm.block_tiles(),inner_room,inner_positive,Bounds.as_nat_le(4,Gemm.block_tiles(),{==}),budget_room))

type StepWindow<-positions: U32,-first: U32,-span: U32> is Data:
  StepWindow{rows: {U32.to_nat(block_rows(positions,first,span)) == Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(positions),U32.to_nat(first))) : Nat},
    end: {U32.to_nat(block_end(positions,first,span)) == Nat.add(U32.to_nat(first),U32.to_nat(block_rows(positions,first,span))) : Nat},
    inside: {Nat.is_le(U32.to_nat(block_end(positions,first,span)),U32.to_nat(positions)) == True{} : Bool},
    positive: {Nat.is_lt(0n,U32.to_nat(block_rows(positions,first,span))) == True{} : Bool}}

def step_inside(+positions: U32,+first: U32,+span: U32,+step: StepWindow<positions,first,span>) ->
  {Nat.is_le(U32.to_nat(block_end(positions,first,span)),U32.to_nat(positions)) == True{} : Bool}:
  StepWindow{rows,end,inside,positive} = step
  inside

def step_window(+limits: Machine.Limits,+positions: U32,+first: U32,+span: U32,
  +span_positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},+before: {Nat.is_lt(U32.to_nat(first),U32.to_nat(positions)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(positions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> StepWindow<positions,first,span>:
  +capacity = Machine.capacity(limits)
  +total = U32.to_nat(positions)
  +low = U32.to_nat(first)
  +width = U32.to_nat(span)
  +left = Nat.sub(total,low)
  +left_word = (positions - first : U32)
  +fits = Bounds.strict_le(low,total,before)
  +left_exact = Arithmetic.subtract(positions,first,capacity,Limits.capacity_fits(limits),fits,
    Order.transitive(left,total,U32.to_nat(capacity),Order.sub_le(total,low),room))
  +size = Nat.min(width,left)
  +rows = block_rows(positions,first,span)
  +rows_exact = Equal.trans(Nat,U32.to_nat(rows),Nat.min(width,U32.to_nat(left_word)),size,WordExtrema.minimum(span,left_word),
    Equal.cong(Nat,Nat,available => Nat.min(width,available),U32.to_nat(left_word),left,left_exact))
  +end = Nat.add(low,U32.to_nat(rows))
  +end_inside = {%Equal.sym(Nat,U32.to_nat(rows),size,rows_exact) : {Nat.is_le(Nat.add(low,_),total) == True{} : Bool}
    %Order.restore_shift(low,total,fits) : {Nat.is_le(Nat.add(low,size),_) == True{} : Bool}
    Order.offset_le(low,size,left,Extrema.minimum_right(width,left)) : {Nat.is_le(end,total) == True{} : Bool}}
  +end_exact = Arithmetic.add(first,rows,capacity,Limits.capacity_fits(limits),Order.transitive(end,total,U32.to_nat(capacity),end_inside,room))
  StepWindow{rows_exact,end_exact,
    Held.nat_le_moved(U32.to_nat(block_end(positions,first,span)),end,total,end_exact,end_inside),
    %Equal.sym(Nat,U32.to_nat(rows),size,rows_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Extrema.minimum_strict_lower(width,left,0n,span_positive,Order.positive_difference(total,low,before))}

def Fuel(remaining: Nat,+available: Nat,+span: Nat) -> Data:
  match remaining:
    case 0n: {available == 0n : Nat}
    case 1n+ +rest:
      Held.Both<{Nat.is_lt(Nat.mul(rest,span),available) == True{} : Bool},{Nat.is_le(available,Nat.mul(1n+rest,span)) == True{} : Bool}>

def fuel_room(+tail: Nat,+available: Nat,+span: Nat,+enough: Fuel(1n+tail,available,span)) ->
  {Nat.is_lt(0n,available) == True{} : Bool}:
  Held.Both{+lower,upper} = enough
  Order.le_lt_trans(0n,Nat.mul(tail,span),available,Order.zero_le(Nat.mul(tail,span)),lower)

def next_fuel(tail: Nat,+available: Nat,+span: Nat,+enough: Fuel(1n+tail,available,span)) ->
  Fuel(tail,Nat.sub(available,Nat.min(span,available)),span):
  match tail:
    case 0n:
      Held.Both{+lower,+upper} = enough
      +inside = {%Addition.zero_right(span) : {Nat.is_le(available,_) == True{} : Bool} upper : {Nat.is_le(available,span) == True{} : Bool}}
      %Equal.sym(Nat,Nat.min(span,available),Nat.min(available,span),Extrema.minimum_commutative(span,available)) : {Nat.sub(available,_) == 0n : Nat}
      %Equal.sym(Nat,Nat.min(available,span),available,Extrema.minimum_when_ordered(available,span,inside)) : {Nat.sub(available,_) == 0n : Nat}
      Order.subtract_below(available,available,Order.reflexive(available))
    case 1n+ +rest:
      Held.Both{+lower,+upper} = enough
      +first = Order.transitive(span,Nat.mul(1n+rest,span),available,DivisionOrder.le_multiple(1n+rest,span,{==}),
        Bounds.strict_le(Nat.mul(1n+rest,span),available,lower))
      +shift = Nat.sub(available,span)
      +restored = Order.restore_shift(span,available,first)
      %Equal.sym(Nat,Nat.min(span,available),span,Extrema.minimum_when_ordered(span,available,first)) : Fuel(1n+rest,Nat.sub(available,_),span)
      Held.Both{Order.cancel_offset_lt(span,Nat.mul(rest,span),shift,
        %Equal.sym(Nat,Nat.add(span,shift),available,restored) : {Nat.is_lt(Nat.mul(1n+rest,span),_) == True{} : Bool} lower),
        Order.cancel_offset_le(span,shift,Nat.mul(1n+rest,span),
          %Equal.sym(Nat,Nat.add(span,shift),available,restored) : {Nat.is_le(_,Nat.mul(1n+(1n+rest),span)) == True{} : Bool} upper)}

def Blocks(remaining: Nat,+positions: U32,+first: U32,+span: U32) -> Data:
  match remaining:
    case 0n: {U32.to_nat(first) == U32.to_nat(positions) : Nat}
    case 1n+ +rest:
      Held.Both<StepWindow<positions,first,span>,Blocks(rest,positions,block_end(positions,first,span),span)>

def clipped_remainder(+positions: U32,+first: U32,+span: U32,
  +low: {Nat.is_le(U32.to_nat(first),U32.to_nat(positions)) == True{} : Bool},+step: StepWindow<positions,first,span>) ->
  {Nat.sub(U32.to_nat(positions),U32.to_nat(block_end(positions,first,span))) ==
    Nat.sub(Nat.sub(U32.to_nat(positions),U32.to_nat(first)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(positions),U32.to_nat(first)))) : Nat}:
  StepWindow{+rows_exact,+end_exact,inside,positive} = step
  +total = U32.to_nat(positions)
  +start = U32.to_nat(first)
  +left = Nat.sub(total,start)
  +rows = U32.to_nat(block_rows(positions,first,span))
  +size = Nat.min(U32.to_nat(span),left)
  %rows_exact : {Nat.sub(total,U32.to_nat(block_end(positions,first,span))) == Nat.sub(left,_) : Nat}
  %Equal.sym(Nat,U32.to_nat(block_end(positions,first,span)),Nat.add(start,rows),end_exact) : {Nat.sub(total,_) == Nat.sub(left,rows) : Nat}
  %Order.restore_shift(start,total,low) : {Nat.sub(_,Nat.add(start,rows)) == Nat.sub(left,rows) : Nat}
  Index.subtract_common_prefix(start,left,rows)

def block_schedule(+limits: Machine.Limits,remaining: Nat,+positions: U32,+first: U32,+span: U32,
  +span_positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +low: {Nat.is_le(U32.to_nat(first),U32.to_nat(positions)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(positions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fuel: Fuel(remaining,Nat.sub(U32.to_nat(positions),U32.to_nat(first)),U32.to_nat(span))) ->
  Blocks(remaining,positions,first,span):
  match remaining:
    case 0n:
      +start = U32.to_nat(first)
      +total = U32.to_nat(positions)
      +left = Nat.sub(total,start)
      Equal.trans(Nat,start,Nat.add(start,left),total,
        Equal.sym(Nat,Nat.add(start,left),start,
          %Equal.sym(Nat,left,0n,fuel) : {Nat.add(start,_) == start : Nat} Addition.zero_right(start)),
        Order.restore_shift(start,total,low))
    case 1n+ +rest:
      +start = U32.to_nat(first)
      +total = U32.to_nat(positions)
      +left = Nat.sub(total,start)
      +width = U32.to_nat(span)
      +end = block_end(positions,first,span)
      +strict = Order.difference_positive(start,total,fuel_room(rest,left,width,fuel))
      +step = step_window(limits,positions,first,span,span_positive,strict,room)
      +moved = clipped_remainder(positions,first,span,low,step)
      Held.Both{step,block_schedule(limits,rest,positions,end,span,span_positive,step_inside(positions,first,span,step),room,
        %Equal.sym(Nat,Nat.sub(total,U32.to_nat(end)),Nat.sub(left,Nat.min(width,left)),moved) : Fuel(rest,_,width)
        next_fuel(rest,left,width,fuel))}

def ceiling_fuel(remaining: Nat,+available: Nat,+span: Nat,+positive: {Nat.is_lt(0n,span) == True{} : Bool},
  +exact: {Ceiling.ceiling(available,span) == remaining : Nat}) -> Fuel(remaining,available,span):
  match remaining:
    case 0n:
      Order.zero_below(available,
        %exact : {Nat.is_le(available,Nat.mul(_,span)) == True{} : Bool}
        Ceiling.covers(available,span,positive))
    case 1n+ +rest:
      Held.Both{
        %Ceiling.upper_comparison(available,span,rest,positive) : {_ == True{} : Bool}
        %Equal.sym(Nat,Ceiling.ceiling(available,span),1n+rest,exact) : {Nat.is_lt(rest,_) == True{} : Bool}
        Order.lt_succ(rest,rest,Order.reflexive(rest)),
        %exact : {Nat.is_le(available,Nat.mul(_,span)) == True{} : Bool}
        Ceiling.covers(available,span,positive)}

def actual_blocks(+limits: Machine.Limits,+positions: U32,+span: U32,
  +positions_room: {Nat.is_le(U32.to_nat(positions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span_room: {Nat.is_le(U32.to_nat(span),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool}) ->
  Blocks(U32.to_nat(Grid.divided_up(positions,span)),positions,0,span):
  +remaining = U32.to_nat(Grid.divided_up(positions,span))
  +total = U32.to_nat(positions)
  +width = U32.to_nat(span)
  block_schedule(limits,remaining,positions,0,span,positive,Order.zero_le(total),positions_room,
    %Equal.sym(Nat,Nat.sub(total,0n),total,Order.subtract_zero(total)) : Fuel(remaining,_,width)
    ceiling_fuel(remaining,total,width,positive,Equal.sym(Nat,remaining,Ceiling.ceiling(total,width),
      GridLaws.divided_exact(limits,positions,span,positions_room,span_room,positive))))

def stage_tree(state: PointBuffer.PointContents) -> Tree.Tree<Tile.Tile16>:
  Buffer.content_tree(~Tile.Tile16,Point.staged_content(state))

def state_readers(state: PointBuffer.PointContents) -> PointJobs.Readers:
  PointBuffer.PointContents{Buffer.Content{bias,bias_length},Buffer.Content{cells,cell_length},Buffer.Content{pairs,pair_length},staged} = state
  PointJobs.Readers{bias,cells,pairs}

def state_stage(+state: PointBuffer.PointContents,tree: Tree.Tree<Tile.Tile16>) -> PointBuffer.PointContents:
  PointBuffer.PointContents{bias,cells,pairs,Buffer.Content{staged,length}} = state
  PointBuffer.PointContents{bias,cells,pairs,Buffer.Content{tree,length}}

def set_stage(+state: PointBuffer.PointContents,+tree: Tree.Tree<Tile.Tile16>) ->
  {stage_tree(state_stage(state,tree)) == tree : Tree.Tree<Tile.Tile16>}:
  PointBuffer.PointContents{bias,cells,pairs,Buffer.Content{staged,length}} = state
  {==}

def staged_readers(+state: PointBuffer.PointContents,+tree: Tree.Tree<Tile.Tile16>) ->
  {state_readers(state_stage(state,tree)) == state_readers(state) : PointJobs.Readers}:
  PointBuffer.PointContents{Buffer.Content{bias,bias_length},Buffer.Content{cells,cell_length},Buffer.Content{pairs,pair_length},Buffer.Content{staged,length}} = state
  {==}

def make_operands(readers: PointJobs.Readers,program: Writes.Program<Tile.Tile16>) -> PointJobs.Operands:
  PointJobs.Readers{bias,input,pairs} = readers
  PointJobs.Operands{bias,input,pairs,program}

def restore_contents(+operands: PointJobs.Operands,+lengths: PointBuffer.PointLengths,+readers: PointJobs.Readers,+tree: Tree.Tree<Tile.Tile16>,
  +reads: {PointJobs.readers_of(operands) == readers : PointJobs.Readers},+writes: {PointJobs.staged_tree(operands) == tree : Tree.Tree<Tile.Tile16>}) ->
  {PointBuffer.operand_contents(operands,lengths) == PointBuffer.operand_contents(make_operands(readers,Writes.Initial{tree}),lengths) : PointBuffer.PointContents}:
  PointJobs.Operands{+bias,+input,+pairs,+stage} = operands
  PointBuffer.PointLengths{+bias_length,+input_length,+pair_length,+stage_length} = lengths
  %reads : {PointBuffer.operand_contents(PointJobs.Operands{bias,input,pairs,stage},lengths) == PointBuffer.operand_contents(
    make_operands(_,Writes.Initial{tree}),lengths) : PointBuffer.PointContents}
  %writes : {PointBuffer.operand_contents(PointJobs.Operands{bias,input,pairs,stage},lengths) == PointBuffer.operand_contents(
    make_operands(PointJobs.Readers{bias,input,pairs},Writes.Initial{_}),lengths) : PointBuffer.PointContents}
  {==}

def StepWrites(~run: Point.Run(),~accumulate: PointJobs.Accumulate()) -> Type:
  @+matrix: Gemm.PointMatrix -> @+state: PointBuffer.PointContents ->
    {Point.block_step(~run,matrix,state) == state_stage(state,Writes.pure(Tile.Tile16,
      PointJobs.all_writes_by(~accumulate,U32.to_nat(Gemm.point_count(matrix)),0,matrix,state_readers(state)),stage_tree(state))) : PointBuffer.PointContents}

def whole_frame(+matrix: Gemm.PointMatrix,+state: PointBuffer.PointContents) ->
  {Point.block_step(~PointJobs.run,matrix,state) == state_stage(state,Writes.pure(Tile.Tile16,
    PointJobs.all_writes_by(~PointJobs.job_accumulators_with(~Gemm.row_times,~PointJobs.whole_inner),U32.to_nat(Gemm.point_count(matrix)),0,matrix,state_readers(state)),stage_tree(state))) :
    PointBuffer.PointContents}:
  PointBuffer.PointContents{Buffer.Content{+bias,+bias_length},Buffer.Content{+input,+input_length},Buffer.Content{+pairs,+pair_length},Buffer.Content{+stage,+stage_length}} = state
  +operands = {PointJobs.Operands{bias,input,pairs,Writes.Initial{stage}} : PointJobs.Operands}
  +lengths = {PointBuffer.PointLengths{bias_length,input_length,pair_length,stage_length} : PointBuffer.PointLengths}
  +count = U32.to_nat(Gemm.point_count(matrix))
  restore_contents(PointJobs.run(count,matrix,operands),lengths,PointJobs.Readers{bias,input,pairs},
    Writes.pure(Tile.Tile16,PointJobs.all_writes(count,0,matrix,PointJobs.Readers{bias,input,pairs}),stage),
    PointBuffer.run_readers(count,matrix,operands),PointJobs.run_writes(count,matrix,operands))

def tail_readers(count: Nat,+first: U32,+matrix: Gemm.PointMatrix,+operands: PointJobs.Operands) ->
  {PointJobs.readers_of(Traversal.indexed_fold(~PointJobs.Operands,~Gemm.PointMatrix,~PointJobs.job_with(~Gemm.panel_times,~PointLaws.tail_reduced),count,first,matrix,operands)) ==
    PointJobs.readers_of(operands) : PointJobs.Readers}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +next = PointJobs.job_with(~Gemm.panel_times,~PointLaws.tail_reduced,first,matrix,operands)
      Equal.trans(PointJobs.Readers,
        PointJobs.readers_of(Traversal.indexed_fold(~PointJobs.Operands,~Gemm.PointMatrix,~PointJobs.job_with(~Gemm.panel_times,~PointLaws.tail_reduced),rest,(first + 1 : U32),matrix,next)),
        PointJobs.readers_of(next),PointJobs.readers_of(operands),
        tail_readers(rest,(first + 1 : U32),matrix,next),PointJobs.readers_kept_with(~Gemm.panel_times,~PointLaws.tail_reduced,first,matrix,operands))

def tail_frame(+matrix: Gemm.PointMatrix,+state: PointBuffer.PointContents) ->
  {Point.block_step(~PointJobs.run_tail,matrix,state) == state_stage(state,Writes.pure(Tile.Tile16,
    PointJobs.all_writes_by(~PointJobs.job_accumulators_with(~Gemm.panel_times,~PointLaws.tail_reduced),U32.to_nat(Gemm.point_count(matrix)),0,matrix,state_readers(state)),stage_tree(state))) :
    PointBuffer.PointContents}:
  PointBuffer.PointContents{Buffer.Content{+bias,+bias_length},Buffer.Content{+input,+input_length},Buffer.Content{+pairs,+pair_length},Buffer.Content{+stage,+stage_length}} = state
  +operands = {PointJobs.Operands{bias,input,pairs,Writes.Initial{stage}} : PointJobs.Operands}
  +lengths = {PointBuffer.PointLengths{bias_length,input_length,pair_length,stage_length} : PointBuffer.PointLengths}
  +count = U32.to_nat(Gemm.point_count(matrix))
  restore_contents(PointJobs.run_tail(count,matrix,operands),lengths,PointJobs.Readers{bias,input,pairs},
    Writes.pure(Tile.Tile16,PointJobs.tail_writes(count,0,matrix,PointJobs.Readers{bias,input,pairs}),stage),
    tail_readers(count,0,matrix,operands),PointJobs.run_tail_writes(count,matrix,operands))

def stage_bias(+state: PointBuffer.PointContents,+stage: Tree.Tree<Tile.Tile16>) ->
  {Point.stored_bias(state_stage(state,stage)) == Point.stored_bias(state) : Buffer.Content<Tile.Tile8>}:
  PointBuffer.PointContents{bias,input,pairs,Buffer.Content{previous,length}} = state
  {==}

def whole_bias(+matrix: Gemm.PointMatrix,+state: PointBuffer.PointContents) ->
  {Point.stored_bias(Point.block_step(~PointJobs.run,matrix,state)) == Point.stored_bias(state) : Buffer.Content<Tile.Tile8>}:
  +after = Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(matrix)),0,matrix,state_readers(state)),stage_tree(state))
  %Equal.sym(PointBuffer.PointContents,Point.block_step(~PointJobs.run,matrix,state),state_stage(state,after),whole_frame(matrix,state)) :
    {Point.stored_bias(_) == Point.stored_bias(state) : Buffer.Content<Tile.Tile8>}
  stage_bias(state,after)

def tail_bias(+matrix: Gemm.PointMatrix,+state: PointBuffer.PointContents) ->
  {Point.stored_bias(Point.block_step(~PointJobs.run_tail,matrix,state)) == Point.stored_bias(state) : Buffer.Content<Tile.Tile8>}:
  +after = Writes.pure(Tile.Tile16,PointJobs.tail_writes(U32.to_nat(Gemm.point_count(matrix)),0,matrix,state_readers(state)),stage_tree(state))
  %Equal.sym(PointBuffer.PointContents,Point.block_step(~PointJobs.run_tail,matrix,state),state_stage(state,after),tail_frame(matrix,state)) :
    {Point.stored_bias(_) == Point.stored_bias(state) : Buffer.Content<Tile.Tile8>}
  stage_bias(state,after)

def matrix_at(+rectangle: Products.Rectangle,+input_base: U32,+first: U32,+span: U32) -> Gemm.PointMatrix:
  Products.Rectangle{output_first,+positions,+inner,+channels,group_major,grid} = rectangle
  +groups = Gemm.blocks(inner)
  Gemm.PointMatrix{Gemm.TileMatrix{channels,block_rows(positions,first,span),inner,first,block_end(positions,first,span)},
    (input_base + first * groups : U32),groups}

def rectangle_positions(+rectangle: Products.Rectangle) -> U32:
  Products.Rectangle{first,positions,inner,channels,group_major,grid} = rectangle
  positions

def rectangle_channels(+rectangle: Products.Rectangle) -> U32:
  Products.Rectangle{first,positions,inner,channels,group_major,grid} = rectangle
  channels

type RectangleSpace<-limits: Machine.Limits,-positions: U32,-channels: U32> is Data:
  RectangleSpace{groups: {Nat.is_lt(0n,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
    group_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(Gemm.blocks(channels)),2n),2n),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(U32.to_nat(positions),U32.to_nat(Gemm.blocks(channels))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def pair_room(+limits: Machine.Limits) -> {Nat.is_le(4n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(4n,16n,U32.to_nat(Machine.capacity(limits)),{==},Bounds.as_nat_le(16,Machine.capacity(limits),Limits.sixteen_within_capacity(limits)))

def checked_run(+limits: Machine.Limits,+rectangle: Products.Rectangle,+input_base: U32,+first: U32,+span: U32,+readers: PointJobs.Readers,
  +space: RectangleSpace<limits,rectangle_positions(rectangle),rectangle_channels(rectangle)>,
  +step: StepWindow<rectangle_positions(rectangle),first,span>) ->
  Readback.Run<Readback.Target{matrix_at(rectangle,input_base,first,span),readers},Machine.half_word(limits)>:
  Products.Rectangle{output_first,+positions,+inner,+channels,group_major,grid} = rectangle
  RectangleSpace{+groups_positive,+group_room,+output_room} = space
  StepWindow{+rows_exact,+end_exact,+inside,+positive} = step
  +rows = block_rows(positions,first,span)
  +end = block_end(positions,first,span)
  +groups = U32.to_nat(Gemm.blocks(channels))
  +inner_groups = Gemm.blocks(inner)
  +end_room = Order.transitive(Nat.mul(U32.to_nat(end),groups),Nat.mul(U32.to_nat(positions),groups),U32.to_nat(Machine.capacity(limits)),
    Index.mul_le(U32.to_nat(end),U32.to_nat(positions),groups,inside),output_room)
  Readback.call_run(channels,rows,inner,first,end,(input_base + first * inner_groups : U32),inner_groups,readers,
    Machine.half_word(limits),Limits.half_word_fits(limits),U32.to_nat(Machine.capacity(limits)),Limits.double_bound(limits,U32.to_nat(Machine.capacity(limits)),
      Order.reflexive(U32.to_nat(Machine.capacity(limits)))),
    groups,{==},groups_positive,group_room,pair_room(limits),positive,end_exact,
    %end_exact : {Nat.is_le(Nat.mul(_,groups),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} end_room)

def checked_block(+limits: Machine.Limits,+rectangle: Products.Rectangle,+input_base: U32,+first: U32,+span: U32,+readers: PointJobs.Readers,
  +space: RectangleSpace<limits,rectangle_positions(rectangle),rectangle_channels(rectangle)>,+step: StepWindow<rectangle_positions(rectangle),first,span>) -> Readback.Checked:
  Readback.Checked{Readback.Target{matrix_at(rectangle,input_base,first,span),readers},Machine.half_word(limits),
    checked_run(limits,rectangle,input_base,first,span,readers,space,step)}

def written_tree(~accumulate: PointJobs.Accumulate(),+checked: Readback.Checked,+tree: Tree.Tree<Tile.Tile16>) -> Tree.Tree<Tile.Tile16>:
  Writes.pure(Tile.Tile16,Readback.checked_writes_by(~accumulate,checked),tree)

def block_frame(~run: Point.Run(),~accumulate: PointJobs.Accumulate(),~write: StepWrites(~run,~accumulate),
  +checked: Readback.Checked,+state: PointBuffer.PointContents,
  +readers_match: {Readback.target_readers(Readback.checked_target(checked)) == state_readers(state) : PointJobs.Readers}) ->
  {Point.block_step(~run,Readback.target_matrix(Readback.checked_target(checked)),state) ==
    state_stage(state,written_tree(~accumulate,checked,stage_tree(state))) : PointBuffer.PointContents}:
  +target = Readback.checked_target(checked)
  +matrix = Readback.target_matrix(target)
  +reader = Readback.target_readers(target)
  +count = U32.to_nat(Gemm.point_count(matrix))
  +before = stage_tree(state)
  +assignments = PointJobs.all_writes_by(~accumulate,count,0,matrix,reader)
  %Readback.checked_stores_by(~accumulate,checked) :
    {Point.block_step(~run,matrix,state) == state_stage(state,Writes.pure(Tile.Tile16,_,before)) : PointBuffer.PointContents}
  %Equal.sym(PointJobs.Readers,reader,state_readers(state),readers_match) :
    {Point.block_step(~run,matrix,state) == state_stage(state,Writes.pure(Tile.Tile16,PointJobs.all_writes_by(~accumulate,count,0,matrix,_),before)) : PointBuffer.PointContents}
  write(matrix,state)

type StageReady<-tree: Tree.Tree<Tile.Tile16>,-cells: Nat> is Data:
  StageReady{depth: Nat,balanced: StorageShape.Balanced(Tile.Tile16,depth,tree),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
    room: {Nat.is_le(cells,U32.to_nat(Tree.capacity(Tile.Tile16,tree))) == True{} : Bool}}

def ready_written(~accumulate: PointJobs.Accumulate(),+checked: Readback.Checked,+tree: Tree.Tree<Tile.Tile16>,+cells: Nat,+ready: StageReady<tree,cells>) ->
  StageReady<written_tree(~accumulate,checked,tree),cells>:
  StageReady{+depth,+balance,+depth_ok,+room} = ready
  +writes = Readback.checked_writes_by(~accumulate,checked)
  +written = written_tree(~accumulate,checked,tree)
  +capacity = Tree.capacity(Tile.Tile16,tree)
  StageReady{depth,Writes.writes_preserve_balance(Tile.Tile16,writes,tree,depth,balance),depth_ok,
    %Equal.sym(U32,Tree.capacity(Tile.Tile16,written),capacity,Writes.writes_preserve_capacity(Tile.Tile16,writes,tree,depth,balance)) :
      {Nat.is_le(cells,U32.to_nat(_)) == True{} : Bool} room}

def stage_width(+rectangle: Products.Rectangle) -> Nat:
  U32.to_nat(Gemm.pairs_of(rectangle_channels(rectangle)))

def stage_cells(+rectangle: Products.Rectangle) -> Nat:
  Nat.mul(U32.to_nat(rectangle_positions(rectangle)),stage_width(rectangle))

def stage_read(+state: PointBuffer.PointContents,+index: Nat) -> Tile.Tile16:
  Tree.read(Tile.Tile16,stage_tree(state),U32.from_nat(index))

def step_keeps(~accumulate: PointJobs.Accumulate(),+checked: Readback.Checked,+state: PointBuffer.PointContents,+query: Nat,+total: Nat,
  +before: {Nat.is_lt(query,Nat.mul(U32.to_nat(Readback.target_start(Readback.checked_target(checked))),U32.to_nat(Readback.target_pairs(Readback.checked_target(checked))))) == True{} : Bool},
  +end_fits: {Nat.is_le(Readback.checked_bound(checked),total) == True{} : Bool},
  +query_inside: {Nat.is_lt(query,total) == True{} : Bool},
  +ready: StageReady<stage_tree(state),total>) ->
  {stage_read(state_stage(state,written_tree(~accumulate,checked,stage_tree(state))),query) == stage_read(state,query) : Tile.Tile16}:
  StageReady{+depth,+balanced,+depth_ok,+room} = ready
  +tree = stage_tree(state)
  +capacity = Tree.capacity(Tile.Tile16,tree)
  +capacity_ok = {%Equal.sym(U32,capacity,U32.shln(1,depth),StorageShape.capacity(Tile.Tile16,depth,tree,balanced)) : {U32.is_le(_,2147483648) == True{} : Bool}
    Power.native_power_bound(depth,Order.successor(depth,30n,depth_ok)) : {U32.is_le(capacity,2147483648) == True{} : Bool}}
  +query_fits = Order.lt_le(query,total,U32.to_nat(capacity),query_inside,room)
  +encoded = Bounds.roundtrip(query,capacity,capacity_ok,Bounds.strict_le(query,U32.to_nat(capacity),query_fits))
  +written = written_tree(~accumulate,checked,tree)
  %Equal.sym(Tree.Tree<Tile.Tile16>,stage_tree(state_stage(state,written)),written,set_stage(state,written)) :
    {Tree.read(Tile.Tile16,_,U32.from_nat(query)) == stage_read(state,query) : Tile.Tile16}
  Readback.spot_keeps_by(~accumulate,checked,query,before,depth,tree,balanced,depth_ok,
    Order.transitive(Readback.checked_bound(checked),total,U32.to_nat(capacity),end_fits,room),
    Bounds.native_bound(query,capacity,encoded,query_fits))

def slot(+position: Nat,+width: Nat,+pair: Nat) -> Nat:
  Nat.add(Nat.mul(position,width),pair)

def local_value(~accumulate: PointJobs.Accumulate(),+checked: Readback.Checked,+position: Nat,+pair: Nat) -> Tile.Tile16:
  +target = Readback.checked_target(checked)
  +offset = Nat.sub(position,U32.to_nat(Readback.target_start(target)))
  Readback.stored_value_by(~accumulate,target,U32.from_nat(pair),U32.from_nat(Nat.div(offset,4n)),Nat.mod(offset,4n))

def step_stores(~accumulate: PointJobs.Accumulate(),+checked: Readback.Checked,+state: PointBuffer.PointContents,+position: Nat,+pair: Nat,+total: Nat,
  +pair_inside: {Nat.is_lt(pair,U32.to_nat(Readback.target_pairs(Readback.checked_target(checked)))) == True{} : Bool},
  +after_start: {Nat.is_le(U32.to_nat(Readback.target_start(Readback.checked_target(checked))),position) == True{} : Bool},
  +before_end: {Nat.is_lt(position,U32.to_nat(Readback.target_end(Readback.checked_target(checked)))) == True{} : Bool},
  +end_fits: {Nat.is_le(Readback.checked_bound(checked),total) == True{} : Bool},+ready: StageReady<stage_tree(state),total>) ->
  {stage_read(state_stage(state,written_tree(~accumulate,checked,stage_tree(state))),
      slot(position,U32.to_nat(Readback.target_pairs(Readback.checked_target(checked))),pair)) == local_value(~accumulate,checked,position,pair) : Tile.Tile16}:
  StageReady{+depth,+balanced,+depth_ok,+room} = ready
  +target = Readback.checked_target(checked)
  +width = U32.to_nat(Readback.target_pairs(target))
  +first = U32.to_nat(Readback.target_start(target))
  +offset = Nat.sub(position,first)
  +restored = Order.restore_shift(first,position,after_start)
  +tree = stage_tree(state)
  +written = written_tree(~accumulate,checked,tree)
  %restored : {stage_read(state_stage(state,written),slot(_,width,pair)) == local_value(~accumulate,checked,position,pair) : Tile.Tile16}
  %Equal.sym(Tree.Tree<Tile.Tile16>,stage_tree(state_stage(state,written)),written,set_stage(state,written)) :
    {Tree.read(Tile.Tile16,_,U32.from_nat(slot(Nat.add(first,offset),width,pair))) == local_value(~accumulate,checked,position,pair) : Tile.Tile16}
  Readback.stored_at_by(~accumulate,checked,offset,pair,pair_inside,
    %Equal.sym(Nat,Nat.add(first,offset),position,restored) : {Nat.is_lt(_,U32.to_nat(Readback.target_end(target))) == True{} : Bool} before_end,
    depth,tree,balanced,depth_ok,Order.transitive(Readback.checked_bound(checked),total,U32.to_nat(Tree.capacity(Tile.Tile16,tree)),end_fits,room))

def end_fits(+rectangle: Products.Rectangle,+first: U32,+span: U32,+step: StepWindow<rectangle_positions(rectangle),first,span>) ->
  {Nat.is_le(Nat.mul(U32.to_nat(block_end(rectangle_positions(rectangle),first,span)),stage_width(rectangle)),stage_cells(rectangle)) == True{} : Bool}:
  Index.mul_le(U32.to_nat(block_end(rectangle_positions(rectangle),first,span)),U32.to_nat(rectangle_positions(rectangle)),stage_width(rectangle),
    step_inside(rectangle_positions(rectangle),first,span,step))

def next_ready(~accumulate: PointJobs.Accumulate(),+checked: Readback.Checked,+state: PointBuffer.PointContents,+cells: Nat,
  +ready: StageReady<stage_tree(state),cells>) ->
  StageReady<stage_tree(state_stage(state,written_tree(~accumulate,checked,stage_tree(state)))),cells>:
  +written = written_tree(~accumulate,checked,stage_tree(state))
  %Equal.sym(Tree.Tree<Tile.Tile16>,stage_tree(state_stage(state,written)),written,set_stage(state,written)) : StageReady<_,cells>
  ready_written(~accumulate,checked,stage_tree(state),cells,ready)

def blocks_keep(~run: Point.Run(),~accumulate: PointJobs.Accumulate(),~write: StepWrites(~run,~accumulate),
  remaining: Nat,+limits: Machine.Limits,+rectangle: Products.Rectangle,+input_base: U32,+first: U32,+span: U32,+state: PointBuffer.PointContents,
  +space: RectangleSpace<limits,rectangle_positions(rectangle),rectangle_channels(rectangle)>,
  +blocks: Blocks(remaining,rectangle_positions(rectangle),first,span),+ready: StageReady<stage_tree(state),stage_cells(rectangle)>,
  +query: Nat,+before: {Nat.is_lt(query,Nat.mul(U32.to_nat(first),stage_width(rectangle))) == True{} : Bool},
  +inside: {Nat.is_lt(query,stage_cells(rectangle)) == True{} : Bool}) ->
  {stage_read(Point.blocks_model(~run,remaining,rectangle,input_base,first,span,state),query) == stage_read(state,query) : Tile.Tile16}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Products.Rectangle{+output_first,+positions,+inner,+channels,+group_major,+grid} = rectangle
      Held.Both{+step,+following} = blocks
      StepWindow{+rows_exact,+end_exact,+end_inside,+positive} = step
      +next = block_end(positions,first,span)
      +rows = block_rows(positions,first,span)
      +width = stage_width(rectangle)
      +checked = checked_block(limits,rectangle,input_base,first,span,state_readers(state),space,step)
      +written = written_tree(~accumulate,checked,stage_tree(state))
      +later = state_stage(state,written)
      +keeps = step_keeps(~accumulate,checked,state,query,stage_cells(rectangle),before,end_fits(rectangle,first,span,step),inside,ready)
      +start_le = {%Equal.sym(Nat,U32.to_nat(next),Nat.add(U32.to_nat(first),U32.to_nat(rows)),end_exact) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool}
        Order.le_add(U32.to_nat(first),U32.to_nat(rows)) : {Nat.is_le(U32.to_nat(first),U32.to_nat(next)) == True{} : Bool}}
      +moves = Order.lt_le(query,Nat.mul(U32.to_nat(first),width),Nat.mul(U32.to_nat(next),width),before,Index.mul_le(U32.to_nat(first),U32.to_nat(next),width,start_le))
      +matrix = matrix_at(rectangle,input_base,first,span)
      %Equal.sym(PointBuffer.PointContents,Point.block_step(~run,matrix,state),later,block_frame(~run,~accumulate,~write,checked,state,{==})) :
        {stage_read(Point.blocks_model(~run,rest,rectangle,input_base,next,span,_),query) == stage_read(state,query) : Tile.Tile16}
      Equal.trans(Tile.Tile16,stage_read(Point.blocks_model(~run,rest,rectangle,input_base,next,span,later),query),stage_read(later,query),stage_read(state,query),
        blocks_keep(~run,~accumulate,~write,rest,limits,rectangle,input_base,next,span,later,space,following,next_ready(~accumulate,checked,state,stage_cells(rectangle),ready),
          query,moves,inside),keeps)

def block_value(~accumulate: PointJobs.Accumulate(),+rectangle: Products.Rectangle,+input_base: U32,+first: U32,+span: U32,+readers: PointJobs.Readers,
  +position: Nat,+pair: Nat) -> Tile.Tile16:
  +offset = Nat.sub(position,U32.to_nat(first))
  Readback.stored_value_by(~accumulate,Readback.Target{matrix_at(rectangle,input_base,first,span),readers},
    U32.from_nat(pair),U32.from_nat(Nat.div(offset,4n)),Nat.mod(offset,4n))

type BlockHit<-accumulate: PointJobs.Accumulate(),-rectangle: Products.Rectangle,-input_base: U32,-span: U32,-readers: PointJobs.Readers,
  -position: Nat,-pair: Nat,-value: Tile.Tile16> is Data:
  BlockHit{first: U32,after: {Nat.is_le(U32.to_nat(first),position) == True{} : Bool},
    before: {Nat.is_lt(position,U32.to_nat(block_end(rectangle_positions(rectangle),first,span))) == True{} : Bool},
    exact: {value == block_value(~accumulate,rectangle,input_base,first,span,readers,position,pair) : Tile.Tile16}}

def find_in_block(~run: Point.Run(),~accumulate: PointJobs.Accumulate(),~write: StepWrites(~run,~accumulate),
  use_first: Bool,+rest: Nat,+limits: Machine.Limits,+rectangle: Products.Rectangle,+input_base: U32,+first: U32,+span: U32,+state: PointBuffer.PointContents,
  +space: RectangleSpace<limits,rectangle_positions(rectangle),rectangle_channels(rectangle)>,
  +step: StepWindow<rectangle_positions(rectangle),first,span>,
  +following: Blocks(rest,rectangle_positions(rectangle),block_end(rectangle_positions(rectangle),first,span),span),
  +ready: StageReady<stage_tree(state),stage_cells(rectangle)>,
  +position: Nat,+pair: Nat,+pair_inside: {Nat.is_lt(pair,stage_width(rectangle)) == True{} : Bool},
  +after_start: {Nat.is_le(U32.to_nat(first),position) == True{} : Bool},
  +below_total: {Nat.is_lt(position,U32.to_nat(rectangle_positions(rectangle))) == True{} : Bool},
  +choice: {Nat.is_lt(position,U32.to_nat(block_end(rectangle_positions(rectangle),first,span))) == use_first : Bool},
  continue: @+next_state: PointBuffer.PointContents -> @+next_ready: StageReady<stage_tree(next_state),stage_cells(rectangle)> ->
    @+after_end: {Nat.is_le(U32.to_nat(block_end(rectangle_positions(rectangle),first,span)),position) == True{} : Bool} ->
    BlockHit<accumulate,rectangle,input_base,span,state_readers(next_state),position,pair,
      stage_read(Point.blocks_model(~run,rest,rectangle,input_base,block_end(rectangle_positions(rectangle),first,span),span,next_state),slot(position,stage_width(rectangle),pair))>) ->
  BlockHit<accumulate,rectangle,input_base,span,state_readers(state),position,pair,
    stage_read(Point.blocks_model(~run,1n+rest,rectangle,input_base,first,span,state),slot(position,stage_width(rectangle),pair))>:
  match use_first:
    case True{}:
      Products.Rectangle{+output_first,+positions,+inner,+channels,+group_major,+grid} = rectangle
      +next = block_end(positions,first,span)
      +width = stage_width(rectangle)
      +index = slot(position,width,pair)
      +checked = checked_block(limits,rectangle,input_base,first,span,state_readers(state),space,step)
      +later = state_stage(state,written_tree(~accumulate,checked,stage_tree(state)))
      +matrix = matrix_at(rectangle,input_base,first,span)
      %Equal.sym(PointBuffer.PointContents,Point.block_step(~run,matrix,state),later,block_frame(~run,~accumulate,~write,checked,state,{==})) :
        BlockHit<accumulate,rectangle,input_base,span,state_readers(state),position,pair,stage_read(Point.blocks_model(~run,rest,rectangle,input_base,next,span,_),index)>
      BlockHit{first,after_start,choice,
        Equal.trans(Tile.Tile16,stage_read(Point.blocks_model(~run,rest,rectangle,input_base,next,span,later),index),stage_read(later,index),block_value(~accumulate,rectangle,input_base,first,span,state_readers(state),position,pair),
          blocks_keep(~run,~accumulate,~write,rest,limits,rectangle,input_base,next,span,later,space,following,next_ready(~accumulate,checked,state,stage_cells(rectangle),ready),
            index,Index.radix(position,U32.to_nat(next),pair,width,choice,pair_inside),Index.radix(position,U32.to_nat(positions),pair,width,below_total,pair_inside)),
          step_stores(~accumulate,checked,state,position,pair,stage_cells(rectangle),pair_inside,after_start,choice,end_fits(rectangle,first,span,step),ready))}
    case False{}:
      Products.Rectangle{+output_first,+positions,+inner,+channels,+group_major,+grid} = rectangle
      +next = block_end(positions,first,span)
      +index = slot(position,stage_width(rectangle),pair)
      +checked = checked_block(limits,rectangle,input_base,first,span,state_readers(state),space,step)
      +written = written_tree(~accumulate,checked,stage_tree(state))
      +later = state_stage(state,written)
      +matrix = matrix_at(rectangle,input_base,first,span)
      +too_early = Order.not_less(position,U32.to_nat(next),choice)
      %Equal.sym(PointBuffer.PointContents,Point.block_step(~run,matrix,state),later,block_frame(~run,~accumulate,~write,checked,state,{==})) :
        BlockHit<accumulate,rectangle,input_base,span,state_readers(state),position,pair,stage_read(Point.blocks_model(~run,rest,rectangle,input_base,next,span,_),index)>
      %staged_readers(state,written) :
        BlockHit<accumulate,rectangle,input_base,span,_,position,pair,stage_read(Point.blocks_model(~run,rest,rectangle,input_base,next,span,later),index)>
      continue(later,next_ready(~accumulate,checked,state,stage_cells(rectangle),ready),too_early)

def blocks_hit(~run: Point.Run(),~accumulate: PointJobs.Accumulate(),~write: StepWrites(~run,~accumulate),
  remaining: Nat,+limits: Machine.Limits,+rectangle: Products.Rectangle,+input_base: U32,+first: U32,+span: U32,+state: PointBuffer.PointContents,
  +space: RectangleSpace<limits,rectangle_positions(rectangle),rectangle_channels(rectangle)>,
  +blocks: Blocks(remaining,rectangle_positions(rectangle),first,span),+ready: StageReady<stage_tree(state),stage_cells(rectangle)>,
  +position: Nat,+pair: Nat,+pair_inside: {Nat.is_lt(pair,stage_width(rectangle)) == True{} : Bool},
  +after_start: {Nat.is_le(U32.to_nat(first),position) == True{} : Bool},
  +below_total: {Nat.is_lt(position,U32.to_nat(rectangle_positions(rectangle))) == True{} : Bool}) ->
  BlockHit<accumulate,rectangle,input_base,span,state_readers(state),position,pair,
    stage_read(Point.blocks_model(~run,remaining,rectangle,input_base,first,span,state),slot(position,stage_width(rectangle),pair))>:
  match remaining:
    case 0n:
      +total = U32.to_nat(rectangle_positions(rectangle))
      +start = U32.to_nat(first)
      +contradiction = Order.le_lt_trans(start,position,total,after_start,below_total)
      Empty.absurd(BlockHit<accumulate,rectangle,input_base,span,state_readers(state),position,pair,stage_read(state,slot(position,stage_width(rectangle),pair))>,
        Encoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(total,total),True{},
          Equal.sym(Bool,Nat.is_lt(total,total),False{},Order.not_below(total,total,Order.reflexive(total))),
          %blocks : {Nat.is_lt(_,total) == True{} : Bool} contradiction)))
    case 1n+ +rest:
      Held.Both{+step,+following} = blocks
      +end = block_end(rectangle_positions(rectangle),first,span)
      find_in_block(~run,~accumulate,~write,Nat.is_lt(position,U32.to_nat(end)),rest,limits,rectangle,input_base,first,span,state,space,step,following,ready,
        position,pair,pair_inside,after_start,below_total,{==},
        after => next_ready => after_end =>
          blocks_hit(~run,~accumulate,~write,rest,limits,rectangle,input_base,end,span,after,space,following,next_ready,position,pair,pair_inside,after_end,below_total))

def initial_ready(+limits: Machine.Limits,+count: U32,+cells: Nat,+exact: {U32.to_nat(count) == cells : Nat},
  +room: {Nat.is_le(cells,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  StageReady<Tree.fresh(Tile.Tile16,Tensors.depth_for(count),Tile.filled_pair(0.0)),cells>:
  +bounded = Limits.count_allocatable(limits,count,
    %Equal.sym(Nat,U32.to_nat(count),cells,exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room)
  +shape = Depth.bounded(count,bounded)
  +depth = Tensors.depth_for(count)
  +tree = Tree.fresh(Tile.Tile16,depth,Tile.filled_pair(0.0))
  +balance = StorageShape.fresh(Tile.Tile16,depth,Tile.filled_pair(0.0))
  StageReady{depth,balance,Depth.allocation_depth(count,shape),
    %exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Tile.Tile16,tree))) == True{} : Bool}
    %Equal.sym(U32,Tree.capacity(Tile.Tile16,tree),U32.shln(1,depth),StorageShape.capacity(Tile.Tile16,depth,tree,balance)) :
      {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,shape)}

def input_inner(+rectangle: Products.Rectangle) -> U32:
  Products.Rectangle{first,positions,inner,channels,group_major,grid} = rectangle
  inner

def rectangle_input(+input: Buffer.Content<Tile.Tile8>,+pairs: Buffer.Content<Tile.Tile16>,+bias: Buffer.Content<Tile.Tile8>) -> PointJobs.Readers:
  PointJobs.Readers{Buffer.content_tree(~Tile.Tile8,bias),Buffer.content_tree(~Tile.Tile8,input),Buffer.content_tree(~Tile.Tile16,pairs)}

# At every pair of a rectangle the actual loop returns the value stored by its one owning block.
def computed_hit(~run: Point.Run(),~accumulate: PointJobs.Accumulate(),~write: StepWrites(~run,~accumulate),
  +limits: Machine.Limits,+rectangle: Products.Rectangle,+input_base: U32,+input: Buffer.Content<Tile.Tile8>,+pairs: Buffer.Content<Tile.Tile16>,+bias: Buffer.Content<Tile.Tile8>,
  +space: RectangleSpace<limits,rectangle_positions(rectangle),rectangle_channels(rectangle)>,
  +pos_room: {Nat.is_le(U32.to_nat(rectangle_positions(rectangle)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span_room: {Nat.is_le(U32.to_nat(Products.block_positions(input_inner(rectangle))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(Products.block_positions(input_inner(rectangle)))) == True{} : Bool},
  +stage_room: {Nat.is_le(stage_cells(rectangle),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +position: Nat,+pair: Nat,+position_inside: {Nat.is_lt(position,U32.to_nat(rectangle_positions(rectangle))) == True{} : Bool},
  +pair_inside: {Nat.is_lt(pair,stage_width(rectangle)) == True{} : Bool}) ->
  BlockHit<accumulate,rectangle,input_base,Products.block_positions(input_inner(rectangle)),rectangle_input(input,pairs,bias),position,pair,
    Tree.read(Tile.Tile16,Buffer.content_tree(~Tile.Tile16,Point.computed_staged(~run,rectangle,input_base,input,pairs,bias)),
      U32.from_nat(slot(position,stage_width(rectangle),pair)))>:
  Products.Rectangle{+output_first,+positions,+inner,+channels,+group_major,+grid} = rectangle
  Buffer.Content{+input_tree,+input_length} = input
  Buffer.Content{+pair_tree,+pair_length} = pairs
  Buffer.Content{+bias_tree,+bias_length} = bias
  +span = Products.block_positions(inner)
  +width = Gemm.pairs_of(channels)
  +count = (width * positions : U32)
  +initial = {PointBuffer.PointContents{bias,input,pairs,Buffer.Content{Tree.fresh(Tile.Tile16,Tensors.depth_for(count),Tile.filled_pair(0.0)),U32.to_nat(count)}} : PointBuffer.PointContents}
  +room = {%Index.multiply_commutative(U32.to_nat(positions),U32.to_nat(width)) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} stage_room :
    {Nat.is_le(Nat.mul(U32.to_nat(width),U32.to_nat(positions)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +exact = Equal.trans(Nat,U32.to_nat(count),Nat.mul(U32.to_nat(width),U32.to_nat(positions)),stage_cells(rectangle),
    Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),width,positions,U32.to_nat(width),U32.to_nat(positions),{==},{==},room),
    Index.multiply_commutative(U32.to_nat(width),U32.to_nat(positions)))
  blocks_hit(~run,~accumulate,~write,U32.to_nat(Grid.divided_up(positions,span)),limits,rectangle,input_base,0,span,initial,space,
    actual_blocks(limits,positions,span,pos_room,span_room,span_positive),initial_ready(limits,count,stage_cells(rectangle),exact,stage_room),
    position,pair,pair_inside,Order.zero_le(position),position_inside)
