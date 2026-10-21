# Checked operations preserve all owners and their balanced-storage certificates.
# Element functions enter hot loops only as direct templates.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./traversal_array.bend as Traversal
import ./tensor_access.bend as Access
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/traversal_certificate.bend as TraversalCertificate
import ./proofs/traversal_witness.bend as TraversalWitness
import ./proofs/tensor_access_certificate.bend as AccessCertificate
import ./traversal_lanes.bend as Lanes
import ./kernel_tile.bend as Tile
import ./proofs/traversal_lanes_certificate.bend as LanesCertificate

type Outcome<-Value: Type> is Type:
  Accepted{value: Value}
  Rejected{value: Value}

def valid_buffer(+length: Nat,capacity: U32,view: Views.View) -> Bool:
  Nat.is_le(length,U32.to_nat(capacity)) && Views.fits(view,U32.from_nat(length))

def pair_buffers(~Element: Data,input_length: Nat,output_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})

def copy_when(~Element: Data,valid: Bool,count: Maybe<&2,U32>,+source: Views.View,target: Views.View,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match valid count:
    case True{} Some{+count}:
      Views.View{axes,+offset} = target
      Storage.Buffer{input,input_length,input_certificate} = input
      Storage.Buffer{output,output_length,output_certificate} = output
      +contiguous = Views.is_contiguous(source)
      Accepted{pair_buffers(~Element,input_length,output_length,
        Access.copy_run(~Element,contiguous,U32.to_nat(count),source,offset,input,output),
        AccessCertificate.copy(~Element,contiguous,U32.to_nat(count),source,offset,input,output,input_certificate,output_certificate))}
    case valid count: Rejected{(input,output)}

# output[offset of target + i] = element i of source read with the flagged
# axes in reverse, in row-major order of the result; an axis without a flag
# is read forward. The guard checks the source itself, and the copy runs
# through its mirrored view.
def copy(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,+reversed: List<&2,Bool>,
  output: Storage.Buffer<Element>,+target: Views.View) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_depth,input_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_depth,output_equation} = output_certificate
  copy_when(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(target) &&
    valid_buffer(input_length,U32.shln(1,input_depth),source) &&
    valid_buffer(output_length,U32.shln(1,output_depth),target),Views.numel(target),Views.mirrored(source,reversed),target,
    Storage.Buffer{input,input_length,Certificate.Certificate{input_depth,input_equation}},Storage.Buffer{output,output_length,Certificate.Certificate{output_depth,output_equation}})

# output[address of element i of target] = input[offset of source + i]: a
# contiguous source written through a strided target view, in row-major order.
def scatter_counted(~Element: Data,+count: Nat,+target: Views.View,+offset: U32,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{output,output_length,output_certificate} = output
  pair_buffers(~Element,input_length,output_length,
    Access.scatter_run(~Element,count,target,offset,input,output),
    AccessCertificate.scatter(~Element,count,target,offset,input,output,input_certificate,output_certificate))

def scatter_when(~Element: Data,valid: Bool,count: Maybe<&2,U32>,source: Views.View,+target: Views.View,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match valid count:
    case True{} Some{+count}:
      Views.View{axes,+offset} = source
      Accepted{scatter_counted(~Element,U32.to_nat(count),target,offset,input,output)}
    case valid count: Rejected{(input,output)}

def scatter(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,
  output: Storage.Buffer<Element>,+target: Views.View) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_depth,input_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_depth,output_equation} = output_certificate
  scatter_when(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(source) &&
    valid_buffer(input_length,U32.shln(1,input_depth),source) &&
    valid_buffer(output_length,U32.shln(1,output_depth),target),Views.numel(target),source,target,
    Storage.Buffer{input,input_length,Certificate.Certificate{input_depth,input_equation}},Storage.Buffer{output,output_length,Certificate.Certificate{output_depth,output_equation}})

def combine_buffers(~Element: Data,left_length: Nat,right_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  (right,left) = result
  TraversalCertificate.Certificates{right_certificate,left_certificate} = certificates
  (Storage.Buffer{left,left_length,left_certificate},Storage.Buffer{right,right_length,right_certificate})

def combine_kernel(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+right_view: Views.View,+offset: U32,
  left: Storage.Buffer<Element>,right: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Storage.Buffer{left,left_length,left_certificate} = left
  Storage.Buffer{right,right_length,right_certificate} = right
  +fast = Access.zero_contiguous(right_view) && U32.is_zero(offset)
  combine_buffers(~Element,left_length,right_length,
    Access.combine_run(~Element,~operation,fast,count,right_view,offset,right,left),
    AccessCertificate.combine(~Element,~operation,fast,count,right_view,offset,right,left,right_certificate,left_certificate))

def combine_when_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,
  valid: Bool,count: Maybe<&2,U32>,right_view: Views.View,target: Views.View,left: Storage.Buffer<Element>,right: Storage.Buffer<Element>) ->
  Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match valid count:
    case True{} Some{count}:
      Views.View{axes,offset} = target
      Accepted{kernel(U32.to_nat(count),right_view,offset,left,right)}
    case valid count: Rejected{(left,right)}

def combine_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,
  left: Storage.Buffer<Element>,+left_view: Views.View,right: Storage.Buffer<Element>,+right_view: Views.View) ->
  Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Storage.Buffer{left,+left_length,left_certificate} = left
  Certificate.Certificate{+left_depth,left_equation} = left_certificate
  Storage.Buffer{right,+right_length,right_certificate} = right
  Certificate.Certificate{+right_depth,right_equation} = right_certificate
  combine_when_with(~Element,~kernel,Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) &&
    valid_buffer(left_length,U32.shln(1,left_depth),left_view) &&
    valid_buffer(right_length,U32.shln(1,right_depth),right_view),Views.numel(left_view),right_view,left_view,
    Storage.Buffer{left,left_length,Certificate.Certificate{left_depth,left_equation}},Storage.Buffer{right,right_length,Certificate.Certificate{right_depth,right_equation}})

def combine(~Element: Data,~operation: Element -> Element -> Element,
  left: Storage.Buffer<Element>,+left_view: Views.View,right: Storage.Buffer<Element>,+right_view: Views.View) ->
  Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  combine_with(~Element,~combine_kernel(~Element,~operation),left,left_view,right,right_view)

# A mapper is the ordered map of count values from position on a certified
# Buffer. mapped applies the operation to one value at a time.
def mapped(~Element: Data,~operation: Element -> Element,+count: Nat,+position: U32,buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Storage.Buffer{storage,length,certificate} = buffer
  Storage.Buffer{Traversal.map_in_place(~Element,~Unit,~Access.apply_element(~Element,~operation),count,position,Unit{},storage),length,
    TraversalCertificate.map_in_place(~Element,~Unit,~Access.apply_element(~Element,~operation),count,position,Unit{},storage,certificate)}

# The same map with the values of each round of eight computed together: an
# operation written with FP32 arithmetic alone then compiles to vector code.
# The certificate's depth decides whether the array holds the positions.
def lanes(~Context: Data,~operation: Context -> F32 -> F32,+count: Nat,+position: U32,+context: Context,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Storage.Buffer{storage,length,Certificate.Certificate{+depth,equation}} = buffer
  Storage.Buffer{Lanes.map_when(~Context,~operation,Lanes.covered(depth,position,count),count,position,context,storage),length,
    LanesCertificate.map_when(~Context,~operation,Lanes.covered(depth,position,count),count,position,context,storage,Certificate.Certificate{depth,equation})}

def mapped_lanes(~operation: F32 -> F32,+count: Nat,+position: U32,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  lanes(~Unit,~Access.apply_element(~F32,~operation),count,position,Unit{},buffer)

# A mapper without a context, as the Unit-context case.
def without_context(~Element: Data,~mapper: @+count: Nat -> @+position: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element>,+count: Nat,+position: U32,+context: Unit,buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  mapper(count,position,buffer)

def holds(count: Maybe<&2,U32>,+length: Nat,capacity: U32) -> Bool:
  match count:
    case None{}: False{}
    case Some{count}: Nat.is_le(U32.to_nat(count),length) && Nat.is_le(length,U32.to_nat(capacity))

def small(value: U32) -> Bool:
  U32.is_le(value,65535)

# The window geometry is exact and every sample address stays in U32.
def pool_geometry(+pool: Access.Pool) -> Bool:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  small(height) && small(width) && small(kernel) && small(stride) && small(padding) && U32.is_lt(0,kernel) && U32.is_lt(0,stride) &&
    U32.is_le(kernel,(height + 2 * padding : U32)) && U32.is_le(kernel,(width + 2 * padding : U32)) &&
    U32.is_eq(out_height,((height + 2 * padding - kernel : U32) / stride + 1 : U32)) &&
    U32.is_eq(out_width,((width + 2 * padding - kernel : U32) / stride + 1 : U32))

# The unchecked sweep of two owners; the callers establish its bounds.
def swept(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+context: Access.SweepContext<Element>,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{output,output_length,output_certificate} = output
  pair_buffers(~Element,input_length,output_length,
    Access.sweep_run(~Element,~operation,count,context,input,output),
    AccessCertificate.sweep(~Element,~operation,count,context,input,output,input_certificate,output_certificate))

def sweep_when(~Element: Data,~operation: Element -> Element -> Element,valid: Bool,count: Maybe<&2,U32>,+context: Access.SweepContext<Element>,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match valid count:
    case True{} Some{count}: Accepted{swept(~Element,~operation,U32.to_nat(count),context,input,output)}
    case valid count: Rejected{(input,output)}

# Every window fits the padded axis and every sample address stays in U32.
# The axis may be a window of rows of a taller image, whose first output
# window starts at origin.
def sweep_geometry(+sweep: Access.Sweep) -> Bool:
  Access.Sweep{+extent,+step,+kernel,+stride,+padding,out_extent,+origin} = sweep
  small(extent) && small(kernel) && small(stride) && small(padding) && small(origin) && U32.is_lt(0,kernel) && U32.is_lt(0,stride) &&
    U32.is_lt(0,step) && U32.is_le(kernel,(extent + 2 * padding : U32))

def sweep_input(+sweep: Access.Sweep,+outer: U32) -> Maybe<&2,U32>:
  Access.Sweep{+extent,+step,kernel,stride,padding,out_extent,origin} = sweep
  Shapes.product([outer,extent,step],Some{1})

def sweep_output(+sweep: Access.Sweep,+outer: U32) -> Maybe<&2,U32>:
  Access.Sweep{extent,+step,kernel,stride,padding,+out_extent,origin} = sweep
  Shapes.product([outer,out_extent,step],Some{1})

# Windows along one axis of outer contiguous blocks; see Access.Sweep.
def sweep(~Element: Data,~operation: Element -> Element -> Element,+pad: Element,input: Storage.Buffer<Element>,+sweep: Access.Sweep,+outer: U32,
  output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_depth,input_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_depth,output_equation} = output_certificate
  sweep_when(~Element,~operation,sweep_geometry(sweep) && holds(sweep_input(sweep,outer),input_length,U32.shln(1,input_depth)) &&
    holds(sweep_output(sweep,outer),output_length,U32.shln(1,output_depth)),sweep_output(sweep,outer),Access.SweepContext{sweep,pad},
    Storage.Buffer{input,input_length,Certificate.Certificate{input_depth,input_equation}},Storage.Buffer{output,output_length,Certificate.Certificate{output_depth,output_equation}})

# The input and the output of a pool around its column-swept planes.
def pool_finished(~Element: Data,input: Storage.Buffer<Element>,result: Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match result:
    case Accepted{(middle,output)}: Accepted{(input,output)}
    case Rejected{(middle,output)}: Rejected{(input,output)}

def pool_rows(~Element: Data,~operation: Element -> Element -> Element,+pad: Element,result: Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,
  +rows: Access.Sweep,+channels: U32,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match result:
    case Accepted{(input,middle)}: pool_finished(~Element,input,sweep(~Element,~operation,pad,middle,rows,channels,output))
    case Rejected{(input,middle)}: Rejected{(input,output)}

def pool_lines(~Element: Data,~operation: Element -> Element -> Element,+pad: Element,lines: Maybe<&2,U32>,input: Storage.Buffer<Element>,+pool: Access.Pool,
  +origin: U32,+channels: U32,middle: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match lines:
    case None{}: Rejected{(input,output)}
    case Some{+lines}:
      pool_rows(~Element,~operation,pad,sweep(~Element,~operation,pad,input,Access.pool_columns(pool),lines,middle),Access.pool_rows(pool,origin),channels,output)

# Windows over the last two axes of channels contiguous planes: a sweep along
# the columns of every row into middle, [channels,height,out_width], then a
# sweep along the rows into output. Each sweep checks its own geometry and
# counts. The stored rows may be a window of a taller image; origin is then
# the row where the window of the first output row starts (see Access.Sweep).
def pool(~Element: Data,~operation: Element -> Element -> Element,+pad: Element,input: Storage.Buffer<Element>,+pool: Access.Pool,+origin: U32,+channels: U32,
  middle: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  pool_lines(~Element,~operation,pad,Shapes.product([channels,height],Some{1}),input,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},origin,channels,middle,output)

# The axes of a line before its last axis, and the last axis; a line of no
# axis is one value.
def prepend_run(axis: Views.Axis,runs: List<&2,Views.Axis> & Views.Axis) -> List<&2,Views.Axis> & Views.Axis:
  (outer,last) = runs
  (axis <> outer,last)

def runs_after(rest: List<&2,Views.Axis>,axis: Views.Axis) -> List<&2,Views.Axis> & Views.Axis:
  match rest:
    case []: ([],axis)
    case next <> tail: prepend_run(axis,runs_after(tail,next))

def line_runs(line: List<&2,Views.Axis>) -> List<&2,Views.Axis> & Views.Axis:
  match line:
    case []: ([],Views.Axis{1,0})
    case axis <> rest: runs_after(rest,axis)

def elements_of(axes: List<&2,Views.Axis>) -> Nat:
  match axes:
    case []: 1n
    case Views.Axis{extent,stride} <> rest: Nat.mul(U32.to_nat(extent),elements_of(rest))

# The lines, unchecked; the callers establish the bounds.
def lined(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +count: Nat,+view: Views.View,+runs: Nat,+extent: U32,+stride: U32,+initial: Accumulator,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{output,output_length,output_certificate} = output
  pair_buffers(~Element,input_length,output_length,
    Access.reduce_lines(~Element,~Accumulator,~operation,~finish,count,view,runs,extent,stride,initial,input,output),
    AccessCertificate.lines(~Element,~Accumulator,~operation,~finish,count,view,runs,extent,stride,initial,input,output,input_certificate,output_certificate))

def runs_lined(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +count: Nat,+kept: List<&2,Views.Axis>,+offset: U32,runs: List<&2,Views.Axis> & Views.Axis,+initial: Accumulator,input: Storage.Buffer<Element>,
  output: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  (+outer,Views.Axis{+extent,+stride}) = runs
  lined(~Element,~Accumulator,~operation,~finish,count,Views.View{List.append(&2,Views.Axis,kept,outer),offset},elements_of(outer),extent,stride,initial,input,output)

def lines_when(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  valid: Bool,count: Maybe<&2,U32>,+kept: List<&2,Views.Axis>,+line: List<&2,Views.Axis>,+offset: U32,+initial: Accumulator,
  input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match valid count:
    case True{} Some{+count}:
      Accepted{runs_lined(~Element,~Accumulator,~operation,~finish,U32.to_nat(count),kept,offset,line_runs(line),initial,input,output)}
    case valid count: Rejected{(input,output)}

# The lines of a view: line i holds the values of the last reduced axes at
# index i of the axes before them. Output i folds line i in row-major order
# from initial and holds finish of the fold. The view must fit the input and
# the outputs the output.
def reduce_lines(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  input: Storage.Buffer<Element>,+axes: List<&2,Views.Axis>,+reduced: Nat,+offset: U32,+initial: Accumulator,output: Storage.Buffer<Element>) ->
  Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_depth,input_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_depth,output_equation} = output_certificate
  +kept_count = Nat.sub(List.length(&2,Views.Axis,axes),reduced)
  +kept = List.take(&2,Views.Axis,axes,kept_count)
  source = {Views.View{axes,offset} : Views.View}
  outputs = {Views.View{kept,0} : Views.View}
  counted = {Views.View{kept,0} : Views.View}
  lines_when(~Element,~Accumulator,~operation,~finish,
    valid_buffer(input_length,U32.shln(1,input_depth),source) && holds(Views.numel(outputs),output_length,U32.shln(1,output_depth)),
    Views.numel(counted),kept,List.drop(&2,Views.Axis,axes,kept_count),offset,initial,
    Storage.Buffer{input,input_length,Certificate.Certificate{input_depth,input_equation}},Storage.Buffer{output,output_length,Certificate.Certificate{output_depth,output_equation}})

# A staged tile transform uses the same certified buffer map and scalar tail.
def lanes_tiles(~Context: Data,~operation: Context -> F32 -> F32,~transform: Context -> Tile.Tile8 -> Tile.Tile8,+count: Nat,+position: U32,+context: Context,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Storage.Buffer{storage,length,Certificate.Certificate{+depth,equation}} = buffer
  Storage.Buffer{Lanes.map_with_when(~Context,~operation,~Lanes.gather_transformed(~Context,~transform),Lanes.covered(depth,position,count),count,position,context,storage),length,
    LanesCertificate.transformed_map_when(~Context,~operation,~transform,Lanes.covered(depth,position,count),count,position,context,storage,Certificate.Certificate{depth,equation})}
