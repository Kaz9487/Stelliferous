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

# A ternary update reads the mask and the second operand at one position
# from a pair of arrays.
type Choice<-Element: Data> is Data:
  Choice{mask: Element,no: Element}

def choice_joined(~Element: Data,mask: Array<Element> & Element,no: Array<Element> & Element) -> (Array<Element> & Array<Element>) & Choice<Element>:
  (mask,mask_value) = mask
  (no,no_value) = no
  ((mask,no),Choice{mask_value,no_value})

def choice_read(~Element: Data,reader: (Array<Element> & Array<Element>),context: Unit,position: U32) -> (Array<Element> & Array<Element>) & Choice<Element>:
  (mask,no) = reader
  +position = position
  choice_joined(~Element,Array.get(Element,mask,position),Array.get(Element,no,position))

def chosen(~Element: Data,~choose: Element -> Element -> Element -> Element,yes: Element,choice: Choice<Element>) -> Element:
  Choice{mask,no} = choice
  choose(mask,yes,no)

# output[i] = choose(mask[i], output[i], no[i]) for the first count positions.
def choose_run(~Element: Data,~choose: Element -> Element -> Element -> Element,count: Nat,mask: Array<Element>,no: Array<Element>,output: Array<Element>) ->
  (Array<Element> & Array<Element>) & Array<Element>:
  Traversal.finish(~(Array<Element> & Array<Element>),~Element,~Unit,
    Traversal.update(~(Array<Element> & Array<Element>),~Element,~Choice<Element>,~Unit,~choice_read(~Element),~chosen(~Element,~choose),~Traversal.index,count,
      Traversal.State{(mask,no),output,0,Unit{}}))

def apply_element(~Element: Data,~operation: Element -> Element,context: Unit,value: Element) -> Element:
  operation(value)

# The geometry of a two-dimensional window reduction over the last two axes.
# It runs as two sweeps: along the columns of every row, then along the rows.
type Pool is Data:
  Pool{height: U32,width: U32,kernel: U32,stride: U32,padding: U32,out_height: U32,out_width: U32}
