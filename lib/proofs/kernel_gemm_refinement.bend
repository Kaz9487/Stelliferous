# Execution refinement for the whole GEMM schedule. The native flat loop
# refines the shared ordered reduction; stores use the actual output model.
# Address coverage is separate.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../kernel_rows.bend as Rows
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../traversal_reduction.bend as Reduction
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./loop_projection.bend as Projection
import ./kernel_reduction_model.bend as ReductionModel
import ./kernel_output_refinement.bend as KernelOutputLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./write_program.bend as Writes
import ./traversal_lanes_refinement.bend as Lanes
import ../traversal_lanes.bend as LanesNative

type Buffers is Data:
  Buffers{input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Writes.Program<F32>}

def represent(buffers: Buffers) -> Gemm.Buffers:
  Buffers{input,weights,bias,output} = buffers
  Gemm.Buffers{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output))}

# A model reduction state as the native loop state. The loop index is the
# model weight index; the model input index is the input base plus that index.
def represent_loop(bias: Tree.Tree<F32>,output: Writes.Program<F32>,context: Gemm.Context,state: ReductionModel.State()) -> Gemm.Loop:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
    weight_index,context,accumulators}

def step_matches(~upper: Bool,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,
  +index: U32,+row: U32,+position: U32,+rows: U32,+columns: U32,+input_base: U32,+weight_base: U32,+accumulators: Rows.Rows<Tile.Tile8>) ->
  {Gemm.step(~upper,Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
      index,Gemm.Context{row,position,rows,columns,input_base,weight_base},accumulators}) ==
    Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
      (index + 1 : U32),Gemm.Context{row,position,rows,columns,input_base,weight_base},
      ReductionModel.update(weight_base,accumulators,ReductionModel.input_tile(input,weight_base,(input_base + index : U32)),ReductionModel.weights_half(~upper,weights,weight_base,index))} :
    Gemm.Loop}:
  -input_array = Tree.pack(Tile.Tile8,input)
  -weight_array = Tree.pack(Tile.Tile8,weights)
  -bias_array = Tree.pack(F32,bias)
  -output_array = Tree.pack(F32,Writes.evaluate(F32,output))
  +context = {Gemm.Context{row,position,rows,columns,input_base,weight_base} : Gemm.Context}
  -result = {Gemm.Loop{input_array,weight_array,bias_array,output_array,(index + 1 : U32),context,
    ReductionModel.update(weight_base,accumulators,ReductionModel.input_tile(input,weight_base,(input_base + index : U32)),ReductionModel.weights_half(~upper,weights,weight_base,index))} : Gemm.Loop}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,weight_array,(weight_base + index : U32)),
    (weight_array,Tree.read(Tile.Tile8,weights,(weight_base + index : U32))),StorageLaws.read(Tile.Tile8,weights,(weight_base + index : U32))) :
    {Gemm.with_weights(~upper,input_array,bias_array,output_array,index,context,input_base,accumulators,_) == result : Gemm.Loop}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,input_array,(input_base + index : U32)),
    (input_array,Tree.read(Tile.Tile8,input,(input_base + index : U32))),StorageLaws.read(Tile.Tile8,input,(input_base + index : U32))) :
    {Gemm.with_input(weight_array,bias_array,output_array,index,context,accumulators,Rows.half(upper,Tree.read(Tile.Tile8,weights,(weight_base + index : U32))),_) ==
      result : Gemm.Loop}
  {==}

# The native loop reads input at input_base + index and weights at
# weight_base + index; the model advances an absolute input index instead.
def loop_matches(~upper: Bool,count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,
  +index: U32,+row: U32,+position: U32,+rows: U32,+columns: U32,+input_base: U32,+weight_base: U32,+accumulators: Rows.Rows<Tile.Tile8>) ->
  {Loop.run(~Gemm.Loop,~Gemm.step(~upper),count,
      Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
        index,Gemm.Context{row,position,rows,columns,input_base,weight_base},accumulators}) ==
    represent_loop(bias,output,Gemm.Context{row,position,rows,columns,input_base,weight_base},
      ReductionModel.run(~upper,count,Reduction.State{input,weights,(input_base + index : U32),index,weight_base,accumulators})) : Gemm.Loop}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +context = {Gemm.Context{row,position,rows,columns,input_base,weight_base} : Gemm.Context}
      -start = {Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
        index,context,accumulators} : Gemm.Loop}
      +next = ReductionModel.update(weight_base,accumulators,ReductionModel.input_tile(input,weight_base,(input_base + index : U32)),ReductionModel.weights_half(~upper,weights,weight_base,index))
      -stepped = {Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
        (index + 1 : U32),context,next} : Gemm.Loop}
      %Equal.sym(Gemm.Loop,Gemm.step(~upper,start),stepped,
        step_matches(~upper,input,weights,bias,output,index,row,position,rows,columns,input_base,weight_base,accumulators)) :
        {Loop.run(~Gemm.Loop,~Gemm.step(~upper),rest,_) ==
          represent_loop(bias,output,context,ReductionModel.run(~upper,1n+rest,Reduction.State{input,weights,(input_base + index : U32),index,weight_base,accumulators})) : Gemm.Loop}
      %Equal.sym(U32,((input_base + index : U32) + 1 : U32),(input_base + (index + 1 : U32) : U32),Arithmetic.add_associative(input_base,index,1)) :
        {Loop.run(~Gemm.Loop,~Gemm.step(~upper),rest,stepped) ==
          represent_loop(bias,output,context,ReductionModel.run(~upper,rest,Reduction.State{input,weights,_,(index + 1 : U32),weight_base,next})) : Gemm.Loop}
      loop_matches(~upper,rest,input,weights,bias,output,(index + 1 : U32),row,position,rows,columns,input_base,weight_base,next)

def after_write(input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,owners: KernelOutputLaws.Owners) -> Buffers:
  KernelOutputLaws.Owners{bias,output} = owners
  Buffers{input,weights,bias,output}

def written(+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+owners: KernelOutputLaws.Owners) ->
  {Gemm.after_write(Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),KernelOutputLaws.represent(owners)) == represent(after_write(input,weights,owners)) : Gemm.Buffers}:
  KernelOutputLaws.Owners{bias,output} = owners
  {==}

def finish(~biased: Bool,bias: Tree.Tree<F32>,output: Writes.Program<F32>,context: Gemm.Context,state: ReductionModel.State()) -> Buffers:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  after_write(input,weights,KernelOutputLaws.write(biased,context,accumulators,bias,output))

def finished(~biased: Bool,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+context: Gemm.Context,+state: ReductionModel.State()) ->
  {Gemm.finish(~biased,represent_loop(bias,output,context,state)) == represent(finish(~biased,bias,output,context,state)) : Gemm.Buffers}:
  Reduction.State{+input,+weights,input_index,weight_index,base,+accumulators} = state
  %Equal.sym(Array<F32> & Array<F32>,Gemm.write(biased,context,accumulators,Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output))),
    KernelOutputLaws.represent(KernelOutputLaws.write(biased,context,accumulators,bias,output)),KernelOutputLaws.write_matches(biased,context,accumulators,bias,output)) :
    {Gemm.after_write(Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),_) ==
      represent(after_write(input,weights,KernelOutputLaws.write(biased,context,accumulators,bias,output))) : Gemm.Buffers}
  written(input,weights,KernelOutputLaws.write(biased,context,accumulators,bias,output))

def reduce(upper: Bool,count: Nat,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,context: Gemm.Context) -> ReductionModel.State():
  Gemm.Context{row,position,rows,columns,+input_base,+weight_base} = context
  ReductionModel.selected(upper,count,ReductionModel.initial(input,weights,input_base,0,weight_base))

def reduced_with(~biased: Bool,~upper: Bool,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,
  +row: U32,+position: U32,+rows: U32,+columns: U32,+input_base: U32,+weight_base: U32,+accumulators: Rows.Rows<Tile.Tile8>) ->
  {Loop.run_then(~Gemm.Loop,~Gemm.Buffers,~Gemm.step(~upper),~Gemm.finish(~biased),count,
      Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
        0,Gemm.Context{row,position,rows,columns,input_base,weight_base},accumulators}) ==
    represent(finish(~biased,bias,output,Gemm.Context{row,position,rows,columns,input_base,weight_base},
      ReductionModel.run(~upper,count,Reduction.State{input,weights,input_base,0,weight_base,accumulators}))) : Gemm.Buffers}:
  +context = {Gemm.Context{row,position,rows,columns,input_base,weight_base} : Gemm.Context}
  +zeros = accumulators
  -start = {Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
    0,context,zeros} : Gemm.Loop}
  +model = ReductionModel.run(~upper,count,Reduction.State{input,weights,input_base,0,weight_base,accumulators})
  %Equal.sym(Gemm.Buffers,Loop.run_then(~Gemm.Loop,~Gemm.Buffers,~Gemm.step(~upper),~Gemm.finish(~biased),count,start),
    Gemm.finish(~biased,Loop.run(~Gemm.Loop,~Gemm.step(~upper),count,start)),
    Projection.run_then_finishes(~Gemm.Loop,~Gemm.Buffers,~Gemm.step(~upper),~Gemm.finish(~biased),count,start)) :
    {_ == represent(finish(~biased,bias,output,context,model)) : Gemm.Buffers}
  %Equal.sym(Gemm.Loop,Loop.run(~Gemm.Loop,~Gemm.step(~upper),count,start),
    represent_loop(bias,output,context,ReductionModel.run(~upper,count,Reduction.State{input,weights,(input_base + 0 : U32),0,weight_base,zeros})),
    loop_matches(~upper,count,input,weights,bias,output,0,row,position,rows,columns,input_base,weight_base,zeros)) :
    {Gemm.finish(~biased,_) == represent(finish(~biased,bias,output,context,model)) : Gemm.Buffers}
  %Equal.sym(U32,(input_base + 0 : U32),input_base,Arithmetic.add_zero(input_base)) :
    {Gemm.finish(~biased,represent_loop(bias,output,context,ReductionModel.run(~upper,count,Reduction.State{input,weights,_,0,weight_base,zeros}))) ==
      represent(finish(~biased,bias,output,context,model)) : Gemm.Buffers}
  finished(~biased,bias,output,context,model)

def reduced(~biased: Bool,~upper: Bool,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,
  +row: U32,+position: U32,+rows: U32,+columns: U32,+input_base: U32,+weight_base: U32) ->
  {Loop.run_then(~Gemm.Loop,~Gemm.Buffers,~Gemm.step(~upper),~Gemm.finish(~biased),count,
      Gemm.Loop{Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,Writes.evaluate(F32,output)),
        0,Gemm.Context{row,position,rows,columns,input_base,weight_base},Rows.filled(~Tile.Tile8,Tile.filled(0.0))}) ==
    represent(finish(~biased,bias,output,Gemm.Context{row,position,rows,columns,input_base,weight_base},
      ReductionModel.run(~upper,count,ReductionModel.initial(input,weights,input_base,0,weight_base)))) : Gemm.Buffers}:
  reduced_with(~biased,~upper,count,input,weights,bias,output,row,position,rows,columns,input_base,weight_base,Rows.filled(~Tile.Tile8,Tile.filled(0.0)))

def execute_with(~biased: Bool,job: Gemm.Job,buffers: Buffers,accumulators: Rows.Rows<Tile.Tile8>) -> Buffers:
  Gemm.Job{+context,upper,count} = job
  Gemm.Context{row,position,rows,columns,+input_base,+weight_base} = context
  Buffers{input,weights,bias,output} = buffers
  finish(~biased,bias,output,context,ReductionModel.selected(upper,count,Reduction.State{input,weights,input_base,0,weight_base,accumulators}))

def executed_with(~biased: Bool,+upper: Bool,+count: Nat,+context: Gemm.Context,+buffers: Buffers,+accumulators: Rows.Rows<Tile.Tile8>) ->
  {Gemm.execute_with(~biased,Gemm.Job{context,upper,count},represent(buffers),accumulators) == represent(execute_with(~biased,Gemm.Job{context,upper,count},buffers,accumulators)) : Gemm.Buffers}:
  match upper:
    case False{}:
      Gemm.Context{+row,+position,+rows,+columns,+input_base,+weight_base} = context
      Buffers{+input,+weights,+bias,+output} = buffers
      reduced_with(~biased,~False{},count,input,weights,bias,output,row,position,rows,columns,input_base,weight_base,accumulators)
    case True{}:
      Gemm.Context{+row,+position,+rows,+columns,+input_base,+weight_base} = context
      Buffers{+input,+weights,+bias,+output} = buffers
      reduced_with(~biased,~True{},count,input,weights,bias,output,row,position,rows,columns,input_base,weight_base,accumulators)

def execute_with_matches(~biased: Bool,+job: Gemm.Job,+buffers: Buffers,+accumulators: Rows.Rows<Tile.Tile8>) ->
  {Gemm.execute_with(~biased,job,represent(buffers),accumulators) == represent(execute_with(~biased,job,buffers,accumulators)) : Gemm.Buffers}:
  Gemm.Job{+context,+upper,+count} = job
  executed_with(~biased,upper,count,context,buffers,accumulators)

def seed(+output: Tree.Tree<F32>,+address: U32,+columns: U32) -> Rows.Rows<Tile.Tile8>:
  +p1 = (address + columns : U32)
  +p2 = (p1 + columns : U32)
  +p3 = (p2 + columns : U32)
  Rows.Rows{Lanes.cell(~Unit,~(context => value => value),output,address,Unit{}),Lanes.cell(~Unit,~(context => value => value),output,p1,Unit{}),
    Lanes.cell(~Unit,~(context => value => value),output,p2,Unit{}),Lanes.cell(~Unit,~(context => value => value),output,p3,Unit{})}

def execute_seeded(+job: Gemm.Job,+buffers: Buffers) -> Buffers:
  Gemm.Job{Gemm.Context{row,position,rows,+columns,input_base,weight_base},upper,count} = job
  Buffers{input,weights,bias,output} = buffers
  execute_with(~False{},job,buffers,seed(Writes.evaluate(F32,output),(row * columns + position : U32),columns))

def execute_seeded_matches(+job: Gemm.Job,+buffers: Buffers) ->
  {Gemm.execute_seeded(job,represent(buffers)) == represent(execute_seeded(job,buffers)) : Gemm.Buffers}:
  Gemm.Job{Gemm.Context{+row,+position,rows,+columns,input_base,weight_base},upper,count} = job
  Buffers{+input,+weights,+bias,+output} = buffers
  +tree = Writes.evaluate(F32,output)
  +address = (row * columns + position : U32)
  +p1 = (address + columns : U32)
  +p2 = (p1 + columns : U32)
  +p3 = (p2 + columns : U32)
  +a = Lanes.cell(~Unit,~(context => value => value),tree,address,Unit{})
  +b = Lanes.cell(~Unit,~(context => value => value),tree,p1,Unit{})
  +c = Lanes.cell(~Unit,~(context => value => value),tree,p2,Unit{})
  +d = Lanes.cell(~Unit,~(context => value => value),tree,p3,Unit{})
  %Equal.sym(Array<F32> & Tile.Tile8,
    LanesNative.gather(~Unit,~(context => value => value),Tree.pack(F32,tree),address,Unit{}),(Tree.pack(F32,tree),a),
    Lanes.gathered(~Unit,~(context => value => value),tree,address,Unit{})) :
    {Gemm.seeded_first(Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),job,address,columns,_) == represent(execute_seeded(job,buffers)) : Gemm.Buffers}
  %Equal.sym(Array<F32> & Tile.Tile8,
    LanesNative.gather(~Unit,~(context => value => value),Tree.pack(F32,tree),p1,Unit{}),(Tree.pack(F32,tree),b),
    Lanes.gathered(~Unit,~(context => value => value),tree,p1,Unit{})) :
    {Gemm.seeded_second(Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),job,p1,columns,a,_) == represent(execute_seeded(job,buffers)) : Gemm.Buffers}
  %Equal.sym(Array<F32> & Tile.Tile8,
    LanesNative.gather(~Unit,~(context => value => value),Tree.pack(F32,tree),p2,Unit{}),(Tree.pack(F32,tree),c),
    Lanes.gathered(~Unit,~(context => value => value),tree,p2,Unit{})) :
    {Gemm.seeded_third(Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),job,p2,columns,a,b,_) == represent(execute_seeded(job,buffers)) : Gemm.Buffers}
  %Equal.sym(Array<F32> & Tile.Tile8,
    LanesNative.gather(~Unit,~(context => value => value),Tree.pack(F32,tree),p3,Unit{}),(Tree.pack(F32,tree),d),
    Lanes.gathered(~Unit,~(context => value => value),tree,p3,Unit{})) :
    {Gemm.seeded_result(Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),job,a,b,c,_) == represent(execute_seeded(job,buffers)) : Gemm.Buffers}
  execute_with_matches(~False{},job,buffers,Rows.Rows{a,b,c,d})

def seeded_step(index: U32,layout: Gemm.Layout,buffers: Buffers) -> Buffers:
  execute_seeded(Gemm.job(layout,index),buffers)

def seeded_step_matches(+index: U32,+layout: Gemm.Layout,+buffers: Buffers) ->
  {Gemm.seeded_job(index,layout,represent(buffers)) == represent(seeded_step(index,layout,buffers)) : Gemm.Buffers}:
  execute_seeded_matches(Gemm.job(layout,index),buffers)

def continue_keeping(+rows: U32,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Tree.Tree<F32>,+columns: U32,inner: U32) -> Buffers:
  Traversal.indexed_fold(~Buffers,~Gemm.Layout,~seeded_step,U32.to_nat(Gemm.job_count(rows,columns)),0,Gemm.Layout{rows,columns,inner,0,columns},Buffers{input,weights,bias,Writes.Initial{output}})

def continue_keeping_matches(+rows: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+columns: U32,+inner: U32) ->
  {Gemm.continue_keeping(rows,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),columns,inner) ==
    represent(continue_keeping(rows,input,weights,bias,output,columns,inner)) : Gemm.Buffers}:
  ArrayLaws.indexed_fold_refines_model(~Gemm.Buffers,~Buffers,~Gemm.Layout,~Gemm.seeded_job,~seeded_step,~represent,~seeded_step_matches,
    U32.to_nat(Gemm.job_count(rows,columns)),0,Gemm.Layout{rows,columns,inner,0,columns},Buffers{input,weights,bias,Writes.Initial{output}})

def execute(~biased: Bool,job: Gemm.Job,buffers: Buffers) -> Buffers:
  Gemm.Job{+context,upper,count} = job
  Buffers{input,weights,bias,output} = buffers
  finish(~biased,bias,output,context,reduce(upper,count,input,weights,context))

def executed(~biased: Bool,+upper: Bool,+count: Nat,+context: Gemm.Context,+buffers: Buffers) ->
  {Gemm.execute(~biased,Gemm.Job{context,upper,count},represent(buffers)) == represent(execute(~biased,Gemm.Job{context,upper,count},buffers)) : Gemm.Buffers}:
  match upper:
    case False{}:
      Gemm.Context{+row,+position,+rows,+columns,+input_base,+weight_base} = context
      Buffers{+input,+weights,+bias,+output} = buffers
      reduced(~biased,~False{},count,input,weights,bias,output,row,position,rows,columns,input_base,weight_base)
    case True{}:
      Gemm.Context{+row,+position,+rows,+columns,+input_base,+weight_base} = context
      Buffers{+input,+weights,+bias,+output} = buffers
      reduced(~biased,~True{},count,input,weights,bias,output,row,position,rows,columns,input_base,weight_base)

def execute_matches(~biased: Bool,+job: Gemm.Job,+buffers: Buffers) ->
  {Gemm.execute(~biased,job,represent(buffers)) == represent(execute(~biased,job,buffers)) : Gemm.Buffers}:
  Gemm.Job{+context,+upper,+count} = job
  executed(~biased,upper,count,context,buffers)

def step(~biased: Bool,index: U32,layout: Gemm.Layout,buffers: Buffers) -> Buffers:
  execute(~biased,Gemm.job(layout,index),buffers)

def step_job_matches(~biased: Bool,+index: U32,+layout: Gemm.Layout,+buffers: Buffers) ->
  {Gemm.step_job(~biased,index,layout,represent(buffers)) == represent(step(~biased,index,layout,buffers)) : Gemm.Buffers}:
  execute_matches(~biased,Gemm.job(layout,index),buffers)

def result(buffers: Buffers) -> Tree.Tree<F32>:
  Buffers{input,weights,bias,output} = buffers
  Writes.evaluate(F32,output)

def result_matches(+buffers: Buffers) ->
  {Gemm.result(represent(buffers)) == Tree.pack(F32,result(buffers)) : Array<F32>}:
  Buffers{input,weights,bias,output} = buffers
  {==}

def run_at(~biased: Bool,+rows: U32,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Tree.Tree<F32>,+columns: U32,inner: U32,start: U32,output_columns: U32) -> Tree.Tree<F32>:
  result(Traversal.indexed_fold(~Buffers,~Gemm.Layout,~step(~biased),U32.to_nat(Gemm.job_count(rows,columns)),0,
    Gemm.Layout{rows,columns,inner,start,output_columns},Buffers{input,weights,bias,Writes.Initial{output}}))

def run_at_matches(~biased: Bool,+rows: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+columns: U32,+inner: U32,+start: U32,+output_columns: U32) ->
  {Gemm.run_at(~biased,rows,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),columns,inner,start,output_columns) ==
    Tree.pack(F32,run_at(~biased,rows,input,weights,bias,output,columns,inner,start,output_columns)) : Array<F32>}:
  +count = U32.to_nat(Gemm.job_count(rows,columns))
  +layout = {Gemm.Layout{rows,columns,inner,start,output_columns} : Gemm.Layout}
  +buffers = {Buffers{input,weights,bias,Writes.Initial{output}} : Buffers}
  +final = Traversal.indexed_fold(~Buffers,~Gemm.Layout,~step(~biased),count,0,layout,buffers)
  %Equal.sym(Gemm.Buffers,Traversal.indexed_fold(~Gemm.Buffers,~Gemm.Layout,~Gemm.step_job(~biased),count,0,layout,represent(buffers)),
    represent(final),ArrayLaws.indexed_fold_refines_model(~Gemm.Buffers,~Buffers,~Gemm.Layout,~Gemm.step_job(~biased),~step(~biased),~represent,~step_job_matches(~biased),count,0,layout,buffers)) :
    {Gemm.result(_) == Tree.pack(F32,result(final)) : Array<F32>}
  result_matches(final)

def run(~biased: Bool,+rows: U32,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Tree.Tree<F32>,+columns: U32,inner: U32) -> Tree.Tree<F32>:
  run_at(~biased,rows,input,weights,bias,output,columns,inner,0,columns)

def run_matches(~biased: Bool,+rows: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+columns: U32,+inner: U32) ->
  {Gemm.run(~biased,rows,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),columns,inner) ==
    Tree.pack(F32,run(~biased,rows,input,weights,bias,output,columns,inner)) : Array<F32>}:
  run_at_matches(~biased,rows,input,weights,bias,output,columns,inner,0,columns)

def keeping_matches(~biased: Bool,+rows: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+columns: U32,+inner: U32,+start: U32,+output_columns: U32) ->
  {Gemm.run_keeping(~biased,rows,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,weights),Tree.pack(F32,bias),Tree.pack(F32,output),columns,inner,start,output_columns) ==
    represent(Traversal.indexed_fold(~Buffers,~Gemm.Layout,~step(~biased),U32.to_nat(Gemm.job_count(rows,columns)),0,Gemm.Layout{rows,columns,inner,start,output_columns},Buffers{input,weights,bias,Writes.Initial{output}})) : Gemm.Buffers}:
  ArrayLaws.indexed_fold_refines_model(~Gemm.Buffers,~Buffers,~Gemm.Layout,~Gemm.step_job(~biased),~step(~biased),~represent,~step_job_matches(~biased),
    U32.to_nat(Gemm.job_count(rows,columns)),0,Gemm.Layout{rows,columns,inner,start,output_columns},Buffers{input,weights,bias,Writes.Initial{output}})
