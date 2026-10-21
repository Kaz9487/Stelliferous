# Internal tensor read layouts and unchecked traversal kernels.
import Base
import ./tensor_view.bend as Views
import ./traversal_array.bend as Traversal
import ./traversal_loop.bend as Loop
import ./tensor_cursor.bend as Cursor

# A strided operation visits the source view in row-major order. It proceeds
# in runs of the innermost axis: every element of a run but the last advances
# only the address, and the last takes the ordinary cursor step, which carries
# into the outer axes. The element loop is a small native loop whatever else
# shares the cursor; stepping the whole cursor per element costs several calls
# with a wide state once more than one operation uses it. inside and last are
# the same element action, stepping a stride and the whole cursor.
def last_of_run(~Element: Data,extent: Nat,stride: U32,rewind: U32,rest: Cursor.Depth1(),
  state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>) -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>:
  Traversal.State{input,output,position,Cursor.Stride{address,step}} = state
  Traversal.State{input,output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n,stride,rewind},rest},address}}

def run_axis(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,remaining: Nat,extent: Nat,+stride: U32,rewind: U32,rest: Cursor.Depth1(),+address: U32,
  input: Array<Element>,output: Array<Element>,position: U32) -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>:
  match remaining:
    case 0n:
      last(Traversal.State{input,output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,0n,stride,rewind},rest},address}})
    case 1n+count:
      last(last_of_run(~Element,extent,stride,rewind,rest,
        Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~inside,count,Traversal.State{input,output,position,Cursor.Stride{address,stride}})))

def run_step(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>) ->
  Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>:
  Traversal.State{input,output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,remaining,stride,rewind},rest},address}} = state
  run_axis(~Element,~inside,~last,remaining,extent,stride,rewind,rest,address,input,output,position)

# Runs are used when the cursor starts a full run and count is a whole number
# of runs; any other call steps element by element. Both visit the same
# elements in the same order.
def run_guarded(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,full_run: Bool,whole_runs: Bool,runs: Nat,count: Nat,cursor: Cursor.Cursor,destination: U32,
  input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  match full_run whole_runs:
    case True{} True{}:
      Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,
        Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~run_step(~Element,~inside,~last),runs,
          Traversal.State{input,output,destination,cursor}))
    case full_run whole_runs:
      Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,
        Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~last,count,Traversal.State{input,output,destination,cursor}))

def run_remaining(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,remaining: Nat,+extent: Nat,stride: U32,rewind: U32,rest: Cursor.Depth1(),address: U32,+count: Nat,destination: U32,
  input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  match remaining:
    case 0n:
      Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,
        Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~last,count,
          Traversal.State{input,output,destination,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,0n,stride,rewind},rest},address}}))
    case 1n+ +inner_count:
      +runs = Nat.div(count,1n+inner_count)
      run_guarded(~Element,~inside,~last,Nat.is_eq(extent,1n+inner_count),Nat.is_eq(Nat.mul(runs,1n+inner_count),count),runs,count,
        Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n+inner_count,stride,rewind},rest},address},destination,input,output)

def run_strided(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,count: Nat,cursor: Cursor.Cursor,destination: U32,input: Array<Element>,output: Array<Element>) ->
  Array<Element> & Array<Element>:
  Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,remaining,stride,rewind},rest},address} = cursor
  run_remaining(~Element,~inside,~last,remaining,extent,stride,rewind,rest,address,count,destination,input,output)

def copy_strided(~Element: Data,count: Nat,cursor: Cursor.Cursor,destination: U32,input: Array<Element>,output: Array<Element>) ->
  Array<Element> & Array<Element>:
  run_strided(~Element,~Traversal.visit_step(~Array<Element>,~Element,~Cursor.Stride,~Cursor.read_stride(~Element)),
    ~Traversal.visit_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element)),count,cursor,destination,input,output)

# output[destination + i] = operation(output[destination + i], element i of the view).
def update_strided(~Element: Data,~operation: Element -> Element -> Element,count: Nat,cursor: Cursor.Cursor,destination: U32,
  input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  run_strided(~Element,~Traversal.visit_update_step(~Array<Element>,~Element,~Cursor.Stride,~Cursor.read_stride(~Element),~operation),
    ~Traversal.visit_update_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation),count,cursor,destination,input,output)

# output[address of element i of the cursor] = input[source + i].
def scatter_strided(~Element: Data,count: Nat,cursor: Cursor.Cursor,source: U32,input: Array<Element>,output: Array<Element>) ->
  Array<Element> & Array<Element>:
  run_strided(~Element,~Traversal.spread_step(~Element,~Cursor.Stride,~Cursor.target_stride),
    ~Traversal.spread_step(~Element,~Cursor.Cursor,~Cursor.target),count,cursor,source,input,output)

def scatter_run(~Element: Data,count: Nat,target: Views.View,source: U32,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  scatter_strided(~Element,count,Cursor.prepare(target),source,input,output)

def copy_run(~Element: Data,contiguous: Bool,count: Nat,source: Views.View,destination: U32,
  input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  match contiguous:
    case True{}:
      Views.View{axes,offset} = source
      Traversal.copy(~Element,count,offset,destination,input,output)
    case False{}: copy_strided(~Element,count,Cursor.prepare(source),destination,input,output)

def zero_contiguous(+view: Views.View) -> Bool:
  Views.View{axes,offset} = view
  U32.is_zero(offset) && Views.is_contiguous(view)

def combine_run(~Element: Data,~operation: Element -> Element -> Element,fast: Bool,count: Nat,
  right_view: Views.View,destination: U32,right: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  match fast:
    case True{}:
      Traversal.finish(~Array<Element>,~Element,~Unit,
        Traversal.update(~Array<Element>,~Element,~Element,~Unit,~Traversal.read_index(~Element),~operation,~Traversal.index,count,
          Traversal.State{right,output,0,Unit{}}))
    case False{}: update_strided(~Element,~operation,count,Cursor.prepare(right_view),destination,right,output)

def apply_element(~Element: Data,~operation: Element -> Element,context: Unit,value: Element) -> Element:
  operation(value)

# Reduction of the lines of a view, one output at a time. The cursor walks
# the view without its last axis, which has extent values stride apart; a
# line is runs consecutive runs of that axis. Output i folds the values of
# line i in the view's row-major order, from initial, and stores finish of
# the fold.
type Runs is Data:
  Runs{cursor: Cursor.Cursor,extent: U32,stride: U32}

type Lines<-Accumulator: Data> is Data:
  Lines{cursor: Cursor.Cursor,runs: Nat,extent: U32,stride: U32,initial: Accumulator}

def ran(~Element: Data,~Accumulator: Data,axes: Cursor.Levels(),address: U32,+extent: U32,+stride: U32,+position: U32,
  result: Array<Element> & Accumulator) -> Traversal.State<&1,&2,Array<Element>,Accumulator,Runs>:
  (input,folded) = result
  Traversal.State{input,folded,(position + 1 : U32),Runs{Cursor.advance(axes,address),extent,stride}}

# One run folded into the accumulator; the cursor moves to the next run.
def fold_run(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,
  state: Traversal.State<&1,&2,Array<Element>,Accumulator,Runs>) -> Traversal.State<&1,&2,Array<Element>,Accumulator,Runs>:
  Traversal.State{input,folded,+position,Runs{Cursor.Cursor{axes,+address},+extent,+stride}} = state
  ran(~Element,~Accumulator,axes,address,extent,stride,position,
    Traversal.fold_rounds(~Array<Element>,~Element,~Accumulator,~Traversal.Strided,~Traversal.strided(~Element),~operation,U32.to_nat(extent),0,input,folded,
      Traversal.Strided{address,stride}))

def folded_line(~Element: Data,~Accumulator: Data,~finish: Accumulator -> Element,+runs: Nat,+initial: Accumulator,
  state: Traversal.State<&1,&2,Array<Element>,Accumulator,Runs>) -> Array<Element> & Traversal.Visited<Lines<Accumulator>,Element>:
  Traversal.State{input,folded,position,Runs{cursor,extent,stride}} = state
  (input,Traversal.Visited{Lines{cursor,runs,extent,stride,initial},finish(folded)})

def fold_line(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  input: Array<Element>,context: Lines<Accumulator>) -> Array<Element> & Traversal.Visited<Lines<Accumulator>,Element>:
  Lines{cursor,+runs,extent,stride,+initial} = context
  folded_line(~Element,~Accumulator,~finish,runs,initial,
    Loop.run(~Traversal.State<&1,&2,Array<Element>,Accumulator,Runs>,~fold_run(~Element,~Accumulator,~operation),runs,
      Traversal.State{input,initial,0,Runs{cursor,extent,stride}}))

# count lines of runs runs; view is the view without its last axis.
def reduce_lines(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  count: Nat,view: Views.View,+runs: Nat,+extent: U32,+stride: U32,+initial: Accumulator,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  Traversal.finish(~Array<Element>,~Element,~Lines<Accumulator>,
    Traversal.visit(~Array<Element>,~Element,~Lines<Accumulator>,~fold_line(~Element,~Accumulator,~operation,~finish),count,
      Traversal.State{input,output,0,Lines{Cursor.prepare(view),runs,extent,stride,initial}}))

# The geometry of a two-dimensional window reduction over the last two axes.
# It runs as two sweeps: along the columns of every row, then along the rows.
type Pool is Data:
  Pool{height: U32,width: U32,kernel: U32,stride: U32,padding: U32,out_height: U32,out_width: U32}

def window_read(~Element: Data,+pad: Element,valid: Bool,input: Array<Element>,index: U32) -> Array<Element> & Element:
  match valid:
    case False{}: (input,pad)
    case True{}: Array.get(Element,input,index)

# Window reduction along one axis. step is the number of elements after the
# axis, the same in the input and the output. Output position p keeps its
# step offset p % step, lies at coordinate (p / step) % out_extent of the
# axis, and folds kernel samples in increasing coordinate from pad; a sample
# outside the axis reads pad. origin is the coordinate at which the window of
# output coordinate zero starts before padding: zero for a whole axis, the
# stored row for the rows of a taller image.
type Sweep is Data:
  Sweep{extent: U32,step: U32,kernel: U32,stride: U32,padding: U32,out_extent: U32,origin: U32}

type SweepContext<-Element: Data> is Data:
  SweepContext{sweep: Sweep,pad: Element}

type Line<-Element: Data> is Data:
  Line{base: U32,first: U32,extent: U32,step: U32,padding: U32,pad: Element}

def read_line(~Element: Data,input: Array<Element>,line: Line<Element>,position: U32) -> Array<Element> & Element:
  Line{+base,+first,+extent,+step,+padding,+pad} = line
  +coordinate = (first + position - padding : U32)
  window_read(~Element,pad,U32.is_lt(coordinate,extent),input,(base + coordinate * step : U32))

def line_at(~Element: Data,+sweep: Sweep,+pad: Element,+position: U32) -> Line<Element>:
  Sweep{+extent,+step,kernel,+stride,+padding,+out_extent,+origin} = sweep
  +along = (position / step : U32)
  Line{(along / out_extent * extent * step + position % step : U32),(along % out_extent * stride + origin : U32),extent,step,padding,pad}

def read_swept(~Element: Data,~operation: Element -> Element -> Element,input: Array<Element>,context: SweepContext<Element>,position: U32) ->
  Array<Element> & Element:
  SweepContext{+sweep,+pad} = context
  Sweep{extent,step,+kernel,stride,padding,out_extent,origin} = sweep
  Traversal.fold(~Array<Element>,~Element,~Element,~Line<Element>,~read_line(~Element),~operation,U32.to_nat(kernel),0,input,pad,
    line_at(~Element,sweep,pad,position))

def swept_index(~Element: Data,context: SweepContext<Element>,position: U32) -> U32:
  position

def sweep_run(~Element: Data,~operation: Element -> Element -> Element,count: Nat,context: SweepContext<Element>,
  input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  Traversal.finish(~Array<Element>,~Element,~SweepContext<Element>,
    Traversal.run(~Array<Element>,~Element,~SweepContext<Element>,~read_swept(~Element,~operation),~swept_index(~Element),count,
      Traversal.State{input,output,0,context}))

# The sweep along the columns of a pool: every stored row is one line.
def pool_columns(pool: Pool) -> Sweep:
  Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  Sweep{width,1,kernel,stride,padding,out_width,0}

# The sweep along the rows of the column-swept planes. The stored rows may be
# a window of a taller image; see Sweep for origin.
def pool_rows(pool: Pool,origin: U32) -> Sweep:
  Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  Sweep{height,out_width,kernel,stride,padding,out_height,origin}
