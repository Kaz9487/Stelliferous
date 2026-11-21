# One kernel row of the window engine over the cells of a segment. Every
# kernel column of an output cell reads the input row the band lends, or pad
# outside the input, and the weight of its tap; the kernel row then updates
# the cell in place. Where the lent row holds the cells of the input, the
# cells of the segment take the value of the kernel row over the input.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../storage_buffer.bend as Storage
import ../tensor_window.bend as Window
import ./tensor_window_values.bend as Values
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./boolean_logic.bend as Boolean
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_pieces_combine_leaf.bend as Updates
import ./storage_shape.bend as Shape
import ./nat_interval.bend as Interval
import ./tensor_allocation_depth.bend as AllocationDepth
import ./u32_division.bend as U32Division
import ./nat_division_order.bend as DivisionOrder
import ./u32_comparison.bend as Comparison
import ../tensor_move.bend as Move
import ./tensor_move_readback.bend as Held
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition

# ---- What a window reads ----

# The input cells [N,H,W,groups] a holder keeps.
type Input<-Holder: Data> is Data:
  Input{holder: Holder,height: U32,width: U32,groups: U32}

def input_read(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,source: Input<Holder>,image: Nat,channel: Nat,row: U32,column: U32) -> Tile.Tile8:
  Input{holder,+height,+width,+groups} = source
  value(holder,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat(row)),U32.to_nat(width)),U32.to_nat(column)),
    U32.to_nat(groups)),channel)))

# The parameter cells of weighting: the weight cells of every group, kernel
# by kernel, and then the bias cells.
type Weights is Data:
  Weights{tree: Tree.Tree<Tile.Tile8>,kernel: U32,groups: U32}

def weight_read(source: Weights,channel: Nat,kernel_row: U32,column: U32) -> Tile.Tile8:
  Weights{tree,+kernel,groups} = source
  Tree.read(Tile.Tile8,tree,((kernel_row * kernel : U32) + U32.from_nat(channel) * kernel * kernel + column : U32))

def bias_read(source: Weights,channel: Nat) -> Tile.Tile8:
  Weights{tree,+kernel,+groups} = source
  Tree.read(Tile.Tile8,tree,(groups * kernel * kernel + U32.from_nat(channel) : U32))

def weight_runtime(+tree: Tree.Tree<Tile.Tile8>,+kernel: U32,+groups: U32,+group: U32,+kernel_row: U32,+column: U32) ->
  {weight_read(Weights{tree,kernel,groups},U32.to_nat(group),kernel_row,column) == Tree.read(Tile.Tile8,tree,((kernel_row * kernel : U32) + group * kernel * kernel + column : U32)) :
    Tile.Tile8}:
  %Arithmetic.native_roundtrip(group) :
    {weight_read(Weights{tree,kernel,groups},U32.to_nat(group),kernel_row,column) == Tree.read(Tile.Tile8,tree,((kernel_row * kernel : U32) + _ * kernel * kernel + column : U32)) :
      Tile.Tile8}
  {==}

# ---- The kind and the fold of the engine ----

def of_sampling(kind: Window.Sampling) -> Values.Sampling<Tile.Tile8>:
  match kind:
    case Window.Pooling{pad}: Values.Pooling{pad}
    case Window.Weighting{}: Values.Weighting{Tile.filled(0.0)}

def folded_of(folded: Window.Folded) -> Values.Folded<Tile.Tile8>:
  Window.Folded{output,line} = folded
  Values.Folded{output,line}

def unfolded(folded: Values.Folded<Tile.Tile8>) -> Window.Folded:
  Values.Folded{output,line} = folded
  Window.Folded{output,line}

def unfolded_of(+folded: Window.Folded) -> {folded == unfolded(folded_of(folded)) : Window.Folded}:
  match folded:
    case Window.Folded{+output,+line}: {==}

def pad_of(+kind: Window.Sampling) -> {Window.sample_pad(kind) == Values.start(~Tile.Tile8,of_sampling(kind)) : Tile.Tile8}:
  match kind:
    case Window.Pooling{+pad}: {==}
    case Window.Weighting{}: {==}

def tapped_of(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+kind: Window.Sampling,+folded: Window.Folded,+value: Tile.Tile8,+weight: Tile.Tile8) ->
  {folded_of(Window.tapped(~operation,kind,folded,value,weight)) ==
    Values.tapped(~Tile.Tile8,~operation,~Tile.update_lanes,of_sampling(kind),folded_of(folded),value,weight) : Values.Folded<Tile.Tile8>}:
  match kind folded:
    case Window.Pooling{+pad} Window.Folded{+output,+line}: {==}
    case Window.Weighting{} Window.Folded{+output,+line}: {==}

# Pooling reads no weight.
def masked_tapped(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+kind: Window.Sampling,+folded: Window.Folded,+value: Tile.Tile8,+weight: Tile.Tile8,+fallback: Tile.Tile8) ->
  {Window.tapped(~operation,kind,folded,value,Bool.pick(Tile.Tile8,Window.weighted(kind),weight,fallback)) == Window.tapped(~operation,kind,folded,value,weight) : Window.Folded}:
  match kind folded:
    case Window.Pooling{+pad} Window.Folded{+output,+line}: {==}
    case Window.Weighting{} Window.Folded{+output,+line}: {==}

def closed_of(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+kind: Window.Sampling,+folded: Values.Folded<Tile.Tile8>) ->
  {Window.row_closed(~operation,kind,unfolded(folded)) == Values.closed(~Tile.Tile8,~operation,of_sampling(kind),folded) : Tile.Tile8}:
  match kind folded:
    case Window.Pooling{+pad} Values.Folded{+output,+line}: {==}
    case Window.Weighting{} Values.Folded{+output,+line}: {==}

# ---- The kernel columns of one cell ----

# The reads of the kernel columns from column on of one cell on a lent row:
# where a sample is inside the input, the lent row holds its input cell.
def ReadsAgree(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,count: Nat,+column: U32,+input: Input<Holder>,+row: Tree.Tree<Tile.Tile8>,+padding: U32,+width: U32,
  +groups: U32,+stored: Bool,+base: U32,+left: U32,+group: U32,+image: Nat,+y: U32) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      KernelGemm.Both<Boolean.When(stored && U32.is_lt((left + column - padding : U32),width),
          {Tree.read(Tile.Tile8,row,(base + (left + column - padding : U32) * groups + group : U32)) ==
            input_read(~Holder,~value,input,image,U32.to_nat(group),y,(left + column - padding : U32)) : Tile.Tile8}),
        ReadsAgree(~Holder,~value,rest,(column + 1 : U32),input,row,padding,width,groups,stored,base,left,group,image,y)>

def picked_same(valid: Bool,+left: Tile.Tile8,+right: Tile.Tile8,+fallback: Tile.Tile8,same: Boolean.When(valid,{left == right : Tile.Tile8})) ->
  {Bool.pick(Tile.Tile8,valid,left,fallback) == Bool.pick(Tile.Tile8,valid,right,fallback) : Tile.Tile8}:
  match valid:
    case False{}: {==}
    case True{}: same

# The sample of a kernel column on a lent row is the sample of the window.
def sampled(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,
  +stored: Bool,+base: U32,+left: U32,+weights: U32,+column: U32,+group: U32,+input: Input<Holder>,+image: Nat,+top: U32,+y: U32,+row: Buffer.Model<Tile.Tile8>,
  +stored_exact: {U32.is_lt(y,height) == stored : Bool},
  same: Boolean.When(stored && U32.is_lt((left + column - padding : U32),width),
    {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,row),(base + (left + column - padding : U32) * groups + group : U32)) ==
      input_read(~Holder,~value,input,image,U32.to_nat(group),y,(left + column - padding : U32)) : Tile.Tile8})) ->
  {Window.sample(Window.Line{Window.Taps{kernel,stride,padding,height,width,groups},stored,base,left,weights},column,group,Window.sample_pad(kind),Buffer.pack(~Tile.Tile8,row)) ==
    (Buffer.pack(~Tile.Tile8,row),Values.sample(~Tile.Tile8,~Input<Holder>,~input_read(~Holder,~value),input,Window.Taps{kernel,stride,padding,height,width,groups},
      Values.Cell{image,U32.to_nat(group),top,left},y,column,Values.start(~Tile.Tile8,of_sampling(kind)))) : Storage.Buffer<Tile.Tile8> & Tile.Tile8}:
  +x = (left + column - padding : U32)
  +index = (base + x * groups + group : U32)
  +read = input_read(~Holder,~value,input,image,U32.to_nat(group),y,x)
  %pad_of(kind) : {Window.sample(Window.Line{Window.Taps{kernel,stride,padding,height,width,groups},stored,base,left,weights},column,group,Window.sample_pad(kind),Buffer.pack(~Tile.Tile8,row)) == (Buffer.pack(~Tile.Tile8,row),Bool.pick(Tile.Tile8,U32.is_lt(y,height) && U32.is_lt(x,width),read,_)) : Storage.Buffer<Tile.Tile8> & Tile.Tile8}
  %Equal.sym(Bool,U32.is_lt(y,height),stored,stored_exact) :
    {Window.sample(Window.Line{Window.Taps{kernel,stride,padding,height,width,groups},stored,base,left,weights},column,group,Window.sample_pad(kind),Buffer.pack(~Tile.Tile8,row)) == (Buffer.pack(~Tile.Tile8,row),Bool.pick(Tile.Tile8,_ && U32.is_lt(x,width),read,Window.sample_pad(kind))) : Storage.Buffer<Tile.Tile8> & Tile.Tile8}
  Equal.trans(Storage.Buffer<Tile.Tile8> & Tile.Tile8,Window.sample(Window.Line{Window.Taps{kernel,stride,padding,height,width,groups},stored,base,left,weights},column,group,Window.sample_pad(kind),Buffer.pack(~Tile.Tile8,row)),
    (Buffer.pack(~Tile.Tile8,row),Bool.pick(Tile.Tile8,stored && U32.is_lt(x,width),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,row),index),Window.sample_pad(kind))),
    (Buffer.pack(~Tile.Tile8,row),Bool.pick(Tile.Tile8,stored && U32.is_lt(x,width),read,Window.sample_pad(kind))),
    Buffer.read_masked(~Tile.Tile8,Window.sample_pad(kind),stored && U32.is_lt(x,width),row,index),
    Equal.cong(Tile.Tile8,Storage.Buffer<Tile.Tile8> & Tile.Tile8,picked => (Buffer.pack(~Tile.Tile8,row),picked),
      Bool.pick(Tile.Tile8,stored && U32.is_lt(x,width),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,row),index),Window.sample_pad(kind)),
      Bool.pick(Tile.Tile8,stored && U32.is_lt(x,width),read,Window.sample_pad(kind)),
      picked_same(stored && U32.is_lt(x,width),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,row),index),read,Window.sample_pad(kind),same)))

# The kernel columns of one cell on a lent row fold the samples of its window
# with the weights of its taps.
def taps_model(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,+column: U32,+kind: Window.Sampling,
  +kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+stored: Bool,+base: U32,+left: U32,+kernel_row: U32,+group: U32,+input: Input<Holder>,
  +image: Nat,+top: U32,+row: Buffer.Model<Tile.Tile8>,+parameters: Buffer.Model<Tile.Tile8>,+folded: Window.Folded,
  +stored_exact: {U32.is_lt((top + kernel_row - padding : U32),height) == stored : Bool},
  agreed: ReadsAgree(~Holder,~value,fuel,column,input,Buffer.tree(~Tile.Tile8,row),padding,width,groups,stored,base,left,group,image,(top + kernel_row - padding : U32))) ->
  {Window.taps(~operation,fuel,column,kind,Window.Line{Window.Taps{kernel,stride,padding,height,width,groups},stored,base,left,(kernel_row * kernel : U32)},group,
      Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),folded}) ==
    Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),
      unfolded(Values.columns(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,fuel,column,input,
        Weights{Buffer.tree(~Tile.Tile8,parameters),kernel,groups},of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},
        Values.Cell{image,U32.to_nat(group),top,left},(top + kernel_row - padding : U32),kernel_row,folded_of(folded)))} : Window.Tapping}:
  match fuel:
    case 0n:
      Equal.cong(Window.Folded,Window.Tapping,kept => Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),kept},folded,unfolded(folded_of(folded)),
        unfolded_of(folded))
    case 1n+ +rest:
      KernelGemm.Both{+same,+later} = agreed
      +taps = {Window.Taps{kernel,stride,padding,height,width,groups} : Window.Taps}
      +line = {Window.Line{taps,stored,base,left,(kernel_row * kernel : U32)} : Window.Line}
      +y = (top + kernel_row - padding : U32)
      +weights = {Weights{Buffer.tree(~Tile.Tile8,parameters),kernel,groups} : Weights}
      +cell = {Values.Cell{image,U32.to_nat(group),top,left} : Values.Cell}
      +sample = Values.sample(~Tile.Tile8,~Input<Holder>,~input_read(~Holder,~value),input,taps,cell,y,column,Values.start(~Tile.Tile8,of_sampling(kind)))
      +index = ((kernel_row * kernel : U32) + group * kernel * kernel + column : U32)
      +weight = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,parameters),index)
      +next = Window.tapped(~operation,kind,folded,sample,weight)
      Equal.trans(Window.Tapping,
        Window.taps(~operation,1n+rest,column,kind,line,group,Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),folded}),
        Window.taps(~operation,rest,(column + 1 : U32),kind,line,group,Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),next}),
        Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),
          unfolded(Values.columns(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,1n+rest,column,input,weights,
            of_sampling(kind),taps,cell,y,kernel_row,folded_of(folded)))},
        Equal.trans(Window.Tapping,
          Window.taps(~operation,1n+rest,column,kind,line,group,Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),folded}),
          Window.taps(~operation,rest,(column + 1 : U32),kind,line,group,
            Window.tap_read(~operation,kind,folded,sample,Storage.read_masked(~Tile.Tile8,Tile.filled(0.0),Window.weighted(kind),Buffer.pack(~Tile.Tile8,parameters),index),
              Buffer.pack(~Tile.Tile8,row))),
          Window.taps(~operation,rest,(column + 1 : U32),kind,line,group,Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),next}),
          Equal.cong(Storage.Buffer<Tile.Tile8> & Tile.Tile8,Window.Tapping,
            read => Window.taps(~operation,rest,(column + 1 : U32),kind,line,group,Window.tap_sampled(~operation,kind,index,folded,Buffer.pack(~Tile.Tile8,parameters),read)),
            Window.sample(line,column,group,Window.sample_pad(kind),Buffer.pack(~Tile.Tile8,row)),(Buffer.pack(~Tile.Tile8,row),sample),
            sampled(~Holder,~value,kind,kernel,stride,padding,height,width,groups,stored,base,left,(kernel_row * kernel : U32),column,group,input,image,top,y,row,stored_exact,same)),
          Equal.trans(Window.Tapping,
            Window.taps(~operation,rest,(column + 1 : U32),kind,line,group,
              Window.tap_read(~operation,kind,folded,sample,Storage.read_masked(~Tile.Tile8,Tile.filled(0.0),Window.weighted(kind),Buffer.pack(~Tile.Tile8,parameters),index),
                Buffer.pack(~Tile.Tile8,row))),
            Window.taps(~operation,rest,(column + 1 : U32),kind,line,group,
              Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),
                Window.tapped(~operation,kind,folded,sample,Bool.pick(Tile.Tile8,Window.weighted(kind),weight,Tile.filled(0.0)))}),
            Window.taps(~operation,rest,(column + 1 : U32),kind,line,group,Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),next}),
            Equal.cong(Storage.Buffer<Tile.Tile8> & Tile.Tile8,Window.Tapping,
              read => Window.taps(~operation,rest,(column + 1 : U32),kind,line,group,Window.tap_read(~operation,kind,folded,sample,read,Buffer.pack(~Tile.Tile8,row))),
              Storage.read_masked(~Tile.Tile8,Tile.filled(0.0),Window.weighted(kind),Buffer.pack(~Tile.Tile8,parameters),index),
              (Buffer.pack(~Tile.Tile8,parameters),Bool.pick(Tile.Tile8,Window.weighted(kind),weight,Tile.filled(0.0))),
              Buffer.read_masked(~Tile.Tile8,Tile.filled(0.0),Window.weighted(kind),parameters,index)),
            Equal.cong(Window.Folded,Window.Tapping,
              kept => Window.taps(~operation,rest,(column + 1 : U32),kind,line,group,Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),kept}),
              Window.tapped(~operation,kind,folded,sample,Bool.pick(Tile.Tile8,Window.weighted(kind),weight,Tile.filled(0.0))),next,
              masked_tapped(~operation,kind,folded,sample,weight,Tile.filled(0.0))))),
        Equal.trans(Window.Tapping,
          Window.taps(~operation,rest,(column + 1 : U32),kind,line,group,Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),next}),
          Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),
            unfolded(Values.columns(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,rest,(column + 1 : U32),input,
              weights,of_sampling(kind),taps,cell,y,kernel_row,folded_of(next)))},
          Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),
            unfolded(Values.columns(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,1n+rest,column,input,weights,
              of_sampling(kind),taps,cell,y,kernel_row,folded_of(folded)))},
          taps_model(~Holder,~value,~operation,rest,(column + 1 : U32),kind,kernel,stride,padding,height,width,groups,stored,base,left,kernel_row,group,input,image,top,row,
            parameters,next,stored_exact,later),
          Equal.cong(Values.Folded<Tile.Tile8>,Window.Tapping,
            started => Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),
              unfolded(Values.columns(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,rest,(column + 1 : U32),input,
                weights,of_sampling(kind),taps,cell,y,kernel_row,started))},
            folded_of(next),
            Values.tapped(~Tile.Tile8,~operation,~Tile.update_lanes,of_sampling(kind),folded_of(folded),sample,weight_read(weights,U32.to_nat(group),kernel_row,column)),
            Equal.trans(Values.Folded<Tile.Tile8>,folded_of(next),
              Values.tapped(~Tile.Tile8,~operation,~Tile.update_lanes,of_sampling(kind),folded_of(folded),sample,weight),
              Values.tapped(~Tile.Tile8,~operation,~Tile.update_lanes,of_sampling(kind),folded_of(folded),sample,weight_read(weights,U32.to_nat(group),kernel_row,column)),
              tapped_of(~operation,kind,folded,sample,weight),
              Equal.cong(Tile.Tile8,Values.Folded<Tile.Tile8>,
                taken => Values.tapped(~Tile.Tile8,~operation,~Tile.update_lanes,of_sampling(kind),folded_of(folded),sample,taken),
                weight,weight_read(weights,U32.to_nat(group),kernel_row,column),
                Equal.sym(Tile.Tile8,weight_read(weights,U32.to_nat(group),kernel_row,column),weight,
                  weight_runtime(Buffer.tree(~Tile.Tile8,parameters),kernel,groups,group,kernel_row,column)))))))

# ---- The cells of a segment ----

# What one kernel row over a segment reads: the input, the parameters, the
# kind and the taps, the image, the first row and column of the windows of
# the segment, the kernel row and the first output cell.
type Row<-Holder: Data> is Data:
  Row{input: Input<Holder>,weights: Weights,kind: Values.Sampling<Tile.Tile8>,taps: Window.Taps,image: Nat,top: U32,left: U32,kernel_row: U32,start: U32}

# The window of one cell of a segment and what it reads.
type Tap<-Holder: Data> is Data:
  Tap{input: Input<Holder>,weights: Weights,kind: Values.Sampling<Tile.Tile8>,taps: Window.Taps,cell: Values.Cell,kernel_row: U32}

def cell_at(+taps: Window.Taps,+image: Nat,+top: U32,+left: U32,+cell: U32) -> Values.Cell:
  Window.Taps{kernel,+stride,padding,height,width,+groups} = taps
  Values.Cell{image,U32.to_nat((cell % U32.max(groups,1) : U32)),top,(left + (cell / U32.max(groups,1) : U32) * stride : U32)}

def tap_at(~Holder: Data,row: Row<Holder>,unit: Unit,position: U32) -> Tap<Holder>:
  Row{input,weights,kind,+taps,+image,+top,+left,kernel_row,+start} = row
  Tap{input,weights,kind,taps,cell_at(taps,image,top,left,(position - start : U32)),kernel_row}

# A kernel row added to one cell.
def row_step(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,accumulator: Tile.Tile8,tap: Tap<Holder>) -> Tile.Tile8:
  Tap{+input,+weights,+kind,+taps,+cell,+kernel_row} = tap
  Values.row_value(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,input,weights,kind,taps,cell,kernel_row,accumulator)

# The cells of a segment from cell on: their kernel columns read inside the
# lent row.
def CellsAgree(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,count: Nat,+cell: U32,+input: Input<Holder>,+row: Tree.Tree<Tile.Tile8>,+kernel: U32,+stride: U32,
  +padding: U32,+width: U32,+groups: U32,+stored: Bool,+base: U32,+left: U32,+image: Nat,+y: U32) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      KernelGemm.Both<ReadsAgree(~Holder,~value,U32.to_nat(kernel),0,input,row,padding,width,groups,stored,base,(left + (cell / U32.max(groups,1) : U32) * stride : U32),
          (cell % U32.max(groups,1) : U32),image,y),
        CellsAgree(~Holder,~value,rest,(cell + 1 : U32),input,row,kernel,stride,padding,width,groups,stored,base,left,image,y)>

# The buffers a kernel row gives back: the lent row and the parameters as they
# were, and the output as a model of a tree and a length.
type SweptModel<-swept: Window.Swept,-row: Buffer.Model<Tile.Tile8>,-parameters: Buffer.Model<Tile.Tile8>,-tree: Tree.Tree<Tile.Tile8>,-length: Nat> is Data:
  SweptModel{output: Buffer.Model<Tile.Tile8>,
    packed: {swept == Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)} : Window.Swept},
    tree: {Buffer.tree(~Tile.Tile8,output) == tree : Tree.Tree<Tile.Tile8>},length: {Buffer.length(~Tile.Tile8,output) == length : Nat}}

def written_buffer(~Element: Data,+index: U32,+value: Element,+model: Buffer.Model<Element>) -> Buffer.Model<Element>:
  Buffer.observed_model(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model)),
    Buffer.observe(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model))))

def written_tree(~Element: Data,+index: U32,+value: Element,+model: Buffer.Model<Element>) ->
  {Buffer.tree(~Element,written_buffer(~Element,index,value,model)) == Tree.write(Element,Buffer.tree(~Element,model),index,value) : Tree.Tree<Element>}:
  +next = written_buffer(~Element,index,value,model)
  Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,next),Buffer.content_tree(~Element,Buffer.content(~Element,next)),Tree.write(Element,Buffer.tree(~Element,model),index,value),
    Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,next)),Buffer.tree(~Element,next),Buffer.content_tree_of(~Element,next)),
    Equal.trans(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,next)),
      Buffer.content_tree(~Element,Buffer.contents(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model)))),
      Tree.write(Element,Buffer.tree(~Element,model),index,value),
      Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.content(~Element,next),
        Buffer.contents(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model))),
        Buffer.observed_content(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model)),
          Buffer.observe(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model))))),
      Equal.trans(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.contents(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model)))),
        Tree.write(Element,Buffer.content_tree(~Element,Buffer.content(~Element,model)),index,value),Tree.write(Element,Buffer.tree(~Element,model),index,value),
        Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.contents(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model))),
          Buffer.Content{Tree.write(Element,Buffer.content_tree(~Element,Buffer.content(~Element,model)),index,value),Buffer.content_length(~Element,Buffer.content(~Element,model))},
          Buffer.write_value_packed(~Element,index,value,model)),
        Equal.cong(Tree.Tree<Element>,Tree.Tree<Element>,tree => Tree.write(Element,tree,index,value),Buffer.content_tree(~Element,Buffer.content(~Element,model)),
          Buffer.tree(~Element,model),Buffer.content_tree_of(~Element,model)))))

def written_length(~Element: Data,+index: U32,+value: Element,+model: Buffer.Model<Element>) ->
  {Buffer.length(~Element,written_buffer(~Element,index,value,model)) == Buffer.length(~Element,model) : Nat}:
  +next = written_buffer(~Element,index,value,model)
  Equal.trans(Nat,Buffer.length(~Element,next),Buffer.content_length(~Element,Buffer.content(~Element,next)),Buffer.length(~Element,model),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,next)),Buffer.length(~Element,next),Buffer.content_length_of(~Element,next)),
    Equal.trans(Nat,Buffer.content_length(~Element,Buffer.content(~Element,next)),
      Buffer.content_length(~Element,Buffer.contents(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model)))),Buffer.length(~Element,model),
      Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,next),
        Buffer.contents(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model))),
        Buffer.observed_content(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model)),
          Buffer.observe(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model))))),
      Equal.trans(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model)))),
        Buffer.content_length(~Element,Buffer.content(~Element,model)),Buffer.length(~Element,model),
        Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model))),
          Buffer.Content{Tree.write(Element,Buffer.content_tree(~Element,Buffer.content(~Element,model)),index,value),Buffer.content_length(~Element,Buffer.content(~Element,model))},
          Buffer.write_value_packed(~Element,index,value,model)),
        Buffer.content_length_of(~Element,model))))

# The new value of one cell of a segment is the kernel row added to it.
def cell_value(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+kind: Window.Sampling,+kernel: U32,+stride: U32,
  +padding: U32,+height: U32,+width: U32,+groups: U32,+kernel_row: U32,+start: U32,+cell: U32,+input: Input<Holder>,+image: Nat,+top: U32,+left: U32,
  +parameters: Buffer.Model<Tile.Tile8>,+accumulator: Tile.Tile8) ->
  {Window.row_closed(~operation,kind,unfolded(Values.columns(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,
      U32.to_nat(kernel),0,input,Weights{Buffer.tree(~Tile.Tile8,parameters),kernel,groups},of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},
      cell_at(Window.Taps{kernel,stride,padding,height,width,groups},image,top,left,cell),(top + kernel_row - padding : U32),kernel_row,
      folded_of(Window.Folded{accumulator,Window.sample_pad(kind)})))) ==
    row_step(~Holder,~value,~operation,accumulator,tap_at(~Holder,Row{input,Weights{Buffer.tree(~Tile.Tile8,parameters),kernel,groups},of_sampling(kind),
      Window.Taps{kernel,stride,padding,height,width,groups},image,top,left,kernel_row,start},Unit{},(start + cell : U32))) : Tile.Tile8}:
  +taps = {Window.Taps{kernel,stride,padding,height,width,groups} : Window.Taps}
  +weights = {Weights{Buffer.tree(~Tile.Tile8,parameters),kernel,groups} : Weights}
  %Equal.sym(U32,((start + cell : U32) - start : U32),cell,Arithmetic.subtract_first(start,cell)) :
    {Window.row_closed(~operation,kind,unfolded(Values.columns(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,
        U32.to_nat(kernel),0,input,weights,of_sampling(kind),taps,cell_at(taps,image,top,left,cell),(top + kernel_row - padding : U32),kernel_row,
        Values.Folded{accumulator,Window.sample_pad(kind)}))) ==
      Values.row_value(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,input,weights,of_sampling(kind),taps,
        cell_at(taps,image,top,left,_),kernel_row,accumulator) : Tile.Tile8}
  %pad_of(kind) :
    {Window.row_closed(~operation,kind,unfolded(Values.columns(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,
        U32.to_nat(kernel),0,input,weights,of_sampling(kind),taps,cell_at(taps,image,top,left,cell),(top + kernel_row - padding : U32),kernel_row,
        Values.Folded{accumulator,Window.sample_pad(kind)}))) ==
      Values.closed(~Tile.Tile8,~operation,of_sampling(kind),Values.columns(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,
        ~weight_read,U32.to_nat(kernel),0,input,weights,of_sampling(kind),taps,cell_at(taps,image,top,left,cell),(top + kernel_row - padding : U32),kernel_row,
        Values.Folded{accumulator,_})) : Tile.Tile8}
  closed_of(~operation,kind,Values.columns(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,U32.to_nat(kernel),0,
    input,weights,of_sampling(kind),taps,cell_at(taps,image,top,left,cell),(top + kernel_row - padding : U32),kernel_row,Values.Folded{accumulator,Window.sample_pad(kind)}))

def swept_moved(-left: Window.Swept,-right: Window.Swept,+row: Buffer.Model<Tile.Tile8>,+parameters: Buffer.Model<Tile.Tile8>,-left_tree: Tree.Tree<Tile.Tile8>,
  -right_tree: Tree.Tree<Tile.Tile8>,-left_length: Nat,-right_length: Nat,model: SweptModel<right,row,parameters,right_tree,right_length>,
  +same: {left == right : Window.Swept},+tree_same: {right_tree == left_tree : Tree.Tree<Tile.Tile8>},+length_same: {right_length == left_length : Nat}) ->
  SweptModel<left,row,parameters,left_tree,left_length>:
  SweptModel{+output,+packed,+tree,+length} = model
  SweptModel{output,
    Equal.trans(Window.Swept,left,right,Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)},same,packed),
    Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,output),right_tree,left_tree,tree,tree_same),
    Equal.trans(Nat,Buffer.length(~Tile.Tile8,output),right_length,left_length,length,length_same)}

# One kernel row over the cells of a segment from cell on updates every one of
# them in place with the kernel row over the input.
def cells_model(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,+cell: U32,+kind: Window.Sampling,
  +kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+stored: Bool,+base: U32,+left: U32,+kernel_row: U32,+start: U32,+input: Input<Holder>,
  +image: Nat,+top: U32,+row: Buffer.Model<Tile.Tile8>,+parameters: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +stored_exact: {U32.is_lt((top + kernel_row - padding : U32),height) == stored : Bool},
  agreed: CellsAgree(~Holder,~value,fuel,cell,input,Buffer.tree(~Tile.Tile8,row),kernel,stride,padding,width,groups,stored,base,left,image,(top + kernel_row - padding : U32))) ->
  SweptModel<Window.cells(~operation,fuel,cell,kind,Window.Line{Window.Taps{kernel,stride,padding,height,width,groups},stored,base,left,(kernel_row * kernel : U32)},start,
      Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)}),row,parameters,
    Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),fuel,(start + cell : U32),
      Row{input,Weights{Buffer.tree(~Tile.Tile8,parameters),kernel,groups},of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},image,top,left,kernel_row,start},
      Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)>:
  match fuel:
    case 0n: SweptModel{output,{==},{==},{==}}
    case 1n+ +rest:
      KernelGemm.Both{+reads,+later} = agreed
      +taps = {Window.Taps{kernel,stride,padding,height,width,groups} : Window.Taps}
      +line = {Window.Line{taps,stored,base,left,(kernel_row * kernel : U32)} : Window.Line}
      +group = (cell % U32.max(groups,1) : U32)
      +shifted = (left + (cell / U32.max(groups,1) : U32) * stride : U32)
      +moved = {Window.Line{taps,stored,base,shifted,(kernel_row * kernel : U32)} : Window.Line}
      +index = (start + cell : U32)
      +old = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),index)
      +weights = {Weights{Buffer.tree(~Tile.Tile8,parameters),kernel,groups} : Weights}
      +reader = {Row{input,weights,of_sampling(kind),taps,image,top,left,kernel_row,start} : Row<Holder>}
      +fresh = row_step(~Holder,~value,~operation,old,tap_at(~Holder,reader,Unit{},index))
      +next = written_buffer(~Tile.Tile8,index,fresh,output)
      +columns = Values.columns(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,U32.to_nat(kernel),0,input,weights,
        of_sampling(kind),taps,Values.Cell{image,U32.to_nat(group),top,shifted},(top + kernel_row - padding : U32),kernel_row,folded_of(Window.Folded{old,Window.sample_pad(kind)}))
      swept_moved(
        Window.cells(~operation,1n+rest,cell,kind,line,start,Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)}),
        Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,next)}),
        row,parameters,
        Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),1n+rest,index,reader,Buffer.tree(~Tile.Tile8,output)),
        Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),rest,(start + (cell + 1 : U32) : U32),reader,
          Buffer.tree(~Tile.Tile8,next)),
        Buffer.length(~Tile.Tile8,output),Buffer.length(~Tile.Tile8,next),
        cells_model(~Holder,~value,~operation,rest,(cell + 1 : U32),kind,kernel,stride,padding,height,width,groups,stored,base,left,kernel_row,start,input,image,top,row,
          parameters,next,stored_exact,later),
        Equal.trans(Window.Swept,
          Window.cells(~operation,1n+rest,cell,kind,line,start,Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)}),
          Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,
            Window.cell_swept(~operation,kind,index,Buffer.pack(~Tile.Tile8,output),
              Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),unfolded(columns)})),
          Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,next)}),
          Equal.trans(Window.Swept,
            Window.cells(~operation,1n+rest,cell,kind,line,start,Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)}),
            Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,
              Window.cell_started(~operation,kind,moved,group,index,Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),(Buffer.pack(~Tile.Tile8,output),old))),
            Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,
              Window.cell_swept(~operation,kind,index,Buffer.pack(~Tile.Tile8,output),
                Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),unfolded(columns)})),
            Equal.cong(Storage.Buffer<Tile.Tile8> & Tile.Tile8,Window.Swept,
              read => Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,
                Window.cell_started(~operation,kind,moved,group,index,Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),read)),
              Storage.read(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output),index),(Buffer.pack(~Tile.Tile8,output),old),Buffer.read(~Tile.Tile8,output,index)),
            Equal.cong(Window.Tapping,Window.Swept,
              tapping => Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,Window.cell_swept(~operation,kind,index,Buffer.pack(~Tile.Tile8,output),tapping)),
              Window.taps(~operation,U32.to_nat(kernel),0,kind,moved,group,
                Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Window.Folded{old,Window.sample_pad(kind)}}),
              Window.Tapping{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),unfolded(columns)},
              taps_model(~Holder,~value,~operation,U32.to_nat(kernel),0,kind,kernel,stride,padding,height,width,groups,stored,base,shifted,kernel_row,group,input,image,top,row,
                parameters,Window.Folded{old,Window.sample_pad(kind)},stored_exact,reads))),
          Equal.trans(Window.Swept,
            Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,
              Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),
                Storage.write(~Tile.Tile8,index,Window.row_closed(~operation,kind,unfolded(columns)),Buffer.pack(~Tile.Tile8,output))}),
            Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,
              Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Storage.write(~Tile.Tile8,index,fresh,Buffer.pack(~Tile.Tile8,output))}),
            Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,next)}),
            Equal.cong(Tile.Tile8,Window.Swept,
              written => Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,
                Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),Storage.write(~Tile.Tile8,index,written,Buffer.pack(~Tile.Tile8,output))}),
              Window.row_closed(~operation,kind,unfolded(columns)),fresh,
              cell_value(~Holder,~value,~operation,kind,kernel,stride,padding,height,width,groups,kernel_row,start,cell,input,image,top,left,parameters,old)),
            Equal.cong(Storage.Buffer<Tile.Tile8>,Window.Swept,
              buffer => Window.cells(~operation,rest,(cell + 1 : U32),kind,line,start,Window.Swept{Buffer.pack(~Tile.Tile8,row),Buffer.pack(~Tile.Tile8,parameters),buffer}),
              Storage.write(~Tile.Tile8,index,fresh,Buffer.pack(~Tile.Tile8,output)),Buffer.pack(~Tile.Tile8,next),
              Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,next),Storage.write(~Tile.Tile8,index,fresh,Buffer.pack(~Tile.Tile8,output)),
                Buffer.observed_equation(~Tile.Tile8,Storage.write(~Tile.Tile8,index,fresh,Buffer.pack(~Tile.Tile8,output)),
                  Buffer.observe(~Tile.Tile8,Storage.write(~Tile.Tile8,index,fresh,Buffer.pack(~Tile.Tile8,output)))))))),
        Equal.trans(Tree.Tree<Tile.Tile8>,
          Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),rest,(start + (cell + 1 : U32) : U32),reader,
            Buffer.tree(~Tile.Tile8,next)),
          Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),rest,(index + 1 : U32),reader,Buffer.tree(~Tile.Tile8,next)),
          Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),1n+rest,index,reader,Buffer.tree(~Tile.Tile8,output)),
          Equal.cong(U32,Tree.Tree<Tile.Tile8>,
            position => Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),rest,position,reader,Buffer.tree(~Tile.Tile8,next)),
            (start + (cell + 1 : U32) : U32),(index + 1 : U32),Equal.sym(U32,(index + 1 : U32),(start + (cell + 1 : U32) : U32),Arithmetic.add_associative(start,cell,1))),
          Equal.cong(Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,
            tree => Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),rest,(index + 1 : U32),reader,tree),
            Buffer.tree(~Tile.Tile8,next),Tree.write(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),index,fresh),written_tree(~Tile.Tile8,index,fresh,output))),
        written_length(~Tile.Tile8,index,fresh,output))

# ---- A stored input row lent by a band ----

# The first input cell of row y of an image.
def row_start(+image: U32,+height: U32,+y: U32,+width: U32,+groups: U32) -> Nat:
  Nat.mul(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y)),U32.to_nat(width)),U32.to_nat(groups))

# The band of a stored input row.
def row_band(+holding: Held.Holding<Tile.Tile8>,+image: U32,+height: U32,+y: U32,+width: U32,+groups: U32,+fill: Tile.Tile8) -> Move.Band<Tile.Tile8>:
  Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),((image * height + y) * width * groups : U32),(width * groups : U32),fill,Move.spare(~Tile.Tile8,fill))

# The band of a stored input row with its fresh spare as a model.
def model_band(+holding: Held.Holding<Tile.Tile8>,+image: U32,+height: U32,+y: U32,+width: U32,+groups: U32,+fill: Tile.Tile8) -> Move.Band<Tile.Tile8>:
  Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),((image * height + y) * width * groups : U32),(width * groups : U32),fill,
    Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,fill)))

def band_model(+holding: Held.Holding<Tile.Tile8>,+image: U32,+height: U32,+y: U32,+width: U32,+groups: U32,+fill: Tile.Tile8) ->
  {model_band(holding,image,height,y,width,groups,fill) == row_band(holding,image,height,y,width,groups,fill) : Move.Band<Tile.Tile8>}:
  Equal.cong(Move.Spare<Tile.Tile8>,Move.Band<Tile.Tile8>,
    spare => Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),((image * height + y) * width * groups : U32),(width * groups : U32),fill,spare),
    Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,fill)),Move.spare(~Tile.Tile8,fill),
    Equal.sym(Move.Spare<Tile.Tile8>,Move.spare(~Tile.Tile8,fill),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,fill)),Held.spare_packed(~Tile.Tile8,fill)))

def band_row(band: Move.Band<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,band))

def band_tree(+holding: Held.Holding<Tile.Tile8>,+image: U32,+height: U32,+y: U32,+width: U32,+groups: U32,+fill: Tile.Tile8) ->
  {Buffer.tree(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,fill))) == Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,row_band(holding,image,height,y,width,groups,fill)))) : Tree.Tree<Tile.Tile8>}:
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,fill))),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,fill)))),
    Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,row_band(holding,image,height,y,width,groups,fill)))),
    Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,fill)))),Buffer.tree(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,fill))),
      Buffer.content_tree_of(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,fill)))),
    Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8),Buffer.content(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,fill))),
      Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,row_band(holding,image,height,y,width,groups,fill))),Buffer.remodel_content(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,row_band(holding,image,height,y,width,groups,fill)))))

# The facts of a stored row: the holding keeps it, and it fits the words.
type RowRoom<-low: Nat,-high: Nat,-capacity: U32,-image: U32,-height: U32,-y: U32,-width: U32,-groups: U32> is Data:
  RowRoom{after: {Nat.is_le(low,row_start(image,height,y,width,groups)) == True{} : Bool},
    fits: {Nat.is_le(Nat.add(row_start(image,height,y,width,groups),Nat.mul(U32.to_nat(width),U32.to_nat(groups))),high) == True{} : Bool},
    first_exact: {U32.to_nat(((image * height + y) * width * groups : U32)) == row_start(image,height,y,width,groups) : Nat},
    count_exact: {U32.to_nat((width * groups : U32)) == Nat.mul(U32.to_nat(width),U32.to_nat(groups)) : Nat},
    allocatable: {U32.is_le((width * groups : U32),1073741824) == True{} : Bool}}

# A cell of a stored row inside the input, read from its band, is its input
# cell.
def band_read(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,+low: Nat,+high: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),+fill: Tile.Tile8,
  +height: U32,+width: U32,+groups: U32,+image: U32,+y: U32,+room: RowRoom<low,high,capacity,image,height,y,width,groups>,+x: U32,+group: U32,
  +x_inside: {Nat.is_lt(U32.to_nat(x),U32.to_nat(width)) == True{} : Bool},+group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,fill))),
      (Held.band_offset(~Tile.Tile8,row_band(holding,image,height,y,width,groups,fill)) + x * groups + group : U32)) ==
    input_read(~Holder,~value,Input{holder,height,width,groups},U32.to_nat(image),U32.to_nat(group),y,x) : Tile.Tile8}:
  RowRoom{+after,+fits,+first_exact,+count_exact,+allocatable} = room
  %band_model(holding,image,height,y,width,groups,fill) :
    {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_row(_)),
      (Held.band_offset(~Tile.Tile8,_) + x * groups + group : U32)) ==
    input_read(~Holder,~value,Input{holder,height,width,groups},U32.to_nat(image),U32.to_nat(group),y,x) : Tile.Tile8}
  +first = ((image * height + y) * width * groups : U32)
  +start = row_start(image,height,y,width,groups)
  +area = Nat.mul(U32.to_nat(width),U32.to_nat(groups))
  +offset = Held.band_offset(~Tile.Tile8,model_band(holding,image,height,y,width,groups,fill))
  +index = Nat.add(Nat.mul(U32.to_nat(x),U32.to_nat(groups)),U32.to_nat(group))
  +less = RowMajor.radix(U32.to_nat(x),U32.to_nat(width),U32.to_nat(group),U32.to_nat(groups),x_inside,group_inside)
  +point = Bounds.nat_trans(Nat.add(start,index),Nat.add(start,area),high,Order.offset_lt(start,index,area,less),fits)
  +start_inside = Order.le_lt_trans(start,Nat.add(start,index),high,Order.le_add(start,index),point)
  +area_room = Order.transitive(area,Nat.add(start,area),U32.to_nat(capacity),Order.right_le(start,area),Order.transitive(Nat.add(start,area),high,U32.to_nat(capacity),fits,bounded))
  +index_room = Order.transitive(index,area,U32.to_nat(capacity),Bounds.strict_le(index,area,less),area_room)
  +address = Equal.trans(U32,(offset + x * groups + group : U32),(offset + (x * groups + group : U32) : U32),U32.from_nat(Nat.add(U32.to_nat(offset),index)),
    Arithmetic.add_associative(offset,(x * groups : U32),group),
    Equal.trans(U32,(offset + (x * groups + group : U32) : U32),U32.from_nat(Nat.add(U32.to_nat(offset),U32.to_nat((x * groups + group : U32)))),
      U32.from_nat(Nat.add(U32.to_nat(offset),index)),
      Arithmetic.native_add(offset,(x * groups + group : U32)),
      Equal.cong(Nat,U32,local => U32.from_nat(Nat.add(U32.to_nat(offset),local)),U32.to_nat((x * groups + group : U32)),index,
        Arithmetic.multiply_add_exact(x,groups,group,capacity,capacity_ok,index_room))))
  +cell = Equal.trans(Nat,Nat.add(U32.to_nat(first),index),Nat.add(start,index),
    Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y)),U32.to_nat(width)),U32.to_nat(x)),U32.to_nat(groups)),U32.to_nat(group)),
    Equal.cong(Nat,Nat,begin => Nat.add(begin,index),U32.to_nat(first),start,first_exact),
    Equal.trans(Nat,Nat.add(start,index),
      Nat.add(Nat.add(start,Nat.mul(U32.to_nat(x),U32.to_nat(groups))),U32.to_nat(group)),
      Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y)),U32.to_nat(width)),U32.to_nat(x)),U32.to_nat(groups)),U32.to_nat(group)),
      Equal.sym(Nat,Nat.add(Nat.add(start,Nat.mul(U32.to_nat(x),U32.to_nat(groups))),U32.to_nat(group)),Nat.add(start,index),
        Addition.associative(start,Nat.mul(U32.to_nat(x),U32.to_nat(groups)),U32.to_nat(group))),
      Equal.cong(Nat,Nat,whole => Nat.add(whole,U32.to_nat(group)),Nat.add(start,Nat.mul(U32.to_nat(x),U32.to_nat(groups))),
        Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y)),U32.to_nat(width)),U32.to_nat(x)),U32.to_nat(groups)),
        Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y)),U32.to_nat(width)),U32.to_nat(x)),U32.to_nat(groups)),
          Nat.add(start,Nat.mul(U32.to_nat(x),U32.to_nat(groups))),
          RowMajor.distrib(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y)),U32.to_nat(width)),U32.to_nat(x),U32.to_nat(groups))))))
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_row(model_band(holding,image,height,y,width,groups,fill))),(offset + x * groups + group : U32)),
    Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,model_band(holding,image,height,y,width,groups,fill)))),U32.from_nat(Nat.add(U32.to_nat(offset),index))),
    input_read(~Holder,~value,Input{holder,height,width,groups},U32.to_nat(image),U32.to_nat(group),y,x),
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_row(model_band(holding,image,height,y,width,groups,fill))),(offset + x * groups + group : U32)),
      Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,model_band(holding,image,height,y,width,groups,fill)))),(offset + x * groups + group : U32)),
      Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,model_band(holding,image,height,y,width,groups,fill)))),U32.from_nat(Nat.add(U32.to_nat(offset),index))),
      Equal.cong(Tree.Tree<Tile.Tile8>,Tile.Tile8,tree => Tree.read(Tile.Tile8,tree,(offset + x * groups + group : U32)),Buffer.tree(~Tile.Tile8,band_row(model_band(holding,image,height,y,width,groups,fill))),
        Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,model_band(holding,image,height,y,width,groups,fill)))),band_tree(holding,image,height,y,width,groups,fill)),
      Equal.cong(U32,Tile.Tile8,word => Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,model_band(holding,image,height,y,width,groups,fill)))),word),
        (offset + x * groups + group : U32),U32.from_nat(Nat.add(U32.to_nat(offset),index)),address)),
    Equal.trans(Tile.Tile8,
      Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,model_band(holding,image,height,y,width,groups,fill)))),U32.from_nat(Nat.add(U32.to_nat(offset),index))),
      value(holder,U32.from_nat(Nat.add(U32.to_nat(first),index))),
      input_read(~Holder,~value,Input{holder,height,width,groups},U32.to_nat(image),U32.to_nat(group),y,x),
      Held.band_value(~Tile.Tile8,~Holder,~value,holder,holding,first,(width * groups : U32),fill,
        Held.fresh_spare(~Tile.Tile8,fill),Held.fresh_spare_room(~Tile.Tile8,fill),low,high,capacity,capacity_ok,bounded,tiled,values,
      %Equal.sym(Nat,U32.to_nat(first),start,first_exact) : {Nat.is_le(low,_) == True{} : Bool} after,
      %Equal.sym(Nat,U32.to_nat(first),start,first_exact) : {Nat.is_lt(_,high) == True{} : Bool} start_inside,
      %Equal.sym(Nat,U32.to_nat(first),start,first_exact) : {Nat.is_le(Nat.add(_,U32.to_nat((width * groups : U32))),high) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat((width * groups : U32)),area,count_exact) : {Nat.is_le(Nat.add(start,_),high) == True{} : Bool} fits,
      allocatable,index,
      %Equal.sym(Nat,U32.to_nat((width * groups : U32)),area,count_exact) : {Nat.is_lt(index,_) == True{} : Bool} less),
      Equal.cong(Nat,Tile.Tile8,position => value(holder,U32.from_nat(position)),Nat.add(U32.to_nat(first),index),
        Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y)),U32.to_nat(width)),U32.to_nat(x)),U32.to_nat(groups)),U32.to_nat(group)),
        cell)))

# The group of a cell of a segment is below the groups.
def group_inside(+cell: U32,+groups: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +groups_room: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(capacity)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat((cell % U32.max(groups,1) : U32)),U32.to_nat(groups)) == True{} : Bool}:
  %Equal.sym(U32,U32.max(groups,1),groups,AllocationDepth.positive_max(groups,positive)) :
    {Nat.is_lt(U32.to_nat((cell % _ : U32)),U32.to_nat(groups)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((cell % groups : U32)),Nat.mod(U32.to_nat(cell),U32.to_nat(groups)),U32Division.modulo(cell,groups,capacity,capacity_ok,groups_room,positive)) :
    {Nat.is_lt(_,U32.to_nat(groups)) == True{} : Bool}
  DivisionOrder.remainder_small(U32.to_nat(cell),U32.to_nat(groups),positive)

# A kernel column of a stored row reads its input cell where it is inside.
def inside_read(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,valid: Bool,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,+low: Nat,+high: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),+fill: Tile.Tile8,
  +height: U32,+width: U32,+groups: U32,+image: U32,+y: U32,+room: RowRoom<low,high,capacity,image,height,y,width,groups>,+x: U32,+group: U32,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+classified: {U32.is_lt(x,width) == valid : Bool}) ->
  Boolean.When(valid,{Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,fill))),
      (Held.band_offset(~Tile.Tile8,row_band(holding,image,height,y,width,groups,fill)) + x * groups + group : U32)) ==
    input_read(~Holder,~value,Input{holder,height,width,groups},U32.to_nat(image),U32.to_nat(group),y,x) : Tile.Tile8}):
  match valid:
    case False{}: Unit{}
    case True{}:
      band_read(~Holder,~value,holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,fill,height,width,groups,image,y,room,x,group,
        Equal.trans(Bool,Nat.is_lt(U32.to_nat(x),U32.to_nat(width)),U32.is_lt(x,width),True{},Equal.sym(Bool,U32.is_lt(x,width),Nat.is_lt(U32.to_nat(x),U32.to_nat(width)),
          Comparison.lt(x,width)),classified),group_inside)

def band_columns(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,count: Nat,+column: U32,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,+low: Nat,+high: Nat,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),+fill: Tile.Tile8,
  +padding: U32,+height: U32,+width: U32,+groups: U32,+image: U32,+y: U32,+room: RowRoom<low,high,capacity,image,height,y,width,groups>,+left: U32,+group: U32,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool}) ->
  ReadsAgree(~Holder,~value,count,column,Input{holder,height,width,groups},Buffer.tree(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,fill))),padding,width,
    groups,True{},Held.band_offset(~Tile.Tile8,row_band(holding,image,height,y,width,groups,fill)),left,group,U32.to_nat(image),y):
  match count:
    case 0n: Unit{}
    case 1n+rest:
      KernelGemm.Both{
        inside_read(~Holder,~value,U32.is_lt((left + column - padding : U32),width),holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,fill,height,width,groups,
          image,y,room,(left + column - padding : U32),group,group_inside,{==}),
        band_columns(~Holder,~value,rest,(column + 1 : U32),holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,fill,padding,height,width,groups,image,y,room,
          left,group,group_inside)}

def band_cells(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,count: Nat,+cell: U32,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,+low: Nat,+high: Nat,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),+fill: Tile.Tile8,
  +kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+image: U32,+y: U32,+room: RowRoom<low,high,capacity,image,height,y,width,groups>,+left: U32,
  +groups_room: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(capacity)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool}) ->
  CellsAgree(~Holder,~value,count,cell,Input{holder,height,width,groups},Buffer.tree(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,fill))),kernel,stride,
    padding,width,groups,True{},Held.band_offset(~Tile.Tile8,row_band(holding,image,height,y,width,groups,fill)),left,U32.to_nat(image),y):
  match count:
    case 0n: Unit{}
    case 1n+rest:
      KernelGemm.Both{
        band_columns(~Holder,~value,U32.to_nat(kernel),0,holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,fill,padding,height,width,groups,image,y,room,
          (left + (cell / U32.max(groups,1) : U32) * stride : U32),(cell % U32.max(groups,1) : U32),
          group_inside(cell,groups,capacity,capacity_ok,groups_room,groups_positive)),
        band_cells(~Holder,~value,rest,(cell + 1 : U32),holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,fill,kernel,stride,padding,height,width,groups,image,
          y,room,left,groups_room,groups_positive)}

# An input row outside the input reads pad everywhere.
def unstored_columns(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,count: Nat,+column: U32,+input: Input<Holder>,+row: Tree.Tree<Tile.Tile8>,+padding: U32,+width: U32,
  +groups: U32,+base: U32,+left: U32,+group: U32,+image: Nat,+y: U32) ->
  ReadsAgree(~Holder,~value,count,column,input,row,padding,width,groups,False{},base,left,group,image,y):
  match count:
    case 0n: Unit{}
    case 1n+rest: KernelGemm.Both{Unit{},unstored_columns(~Holder,~value,rest,(column + 1 : U32),input,row,padding,width,groups,base,left,group,image,y)}

def unstored_cells(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,count: Nat,+cell: U32,+input: Input<Holder>,+row: Tree.Tree<Tile.Tile8>,+kernel: U32,+stride: U32,
  +padding: U32,+width: U32,+groups: U32,+base: U32,+left: U32,+image: Nat,+y: U32) ->
  CellsAgree(~Holder,~value,count,cell,input,row,kernel,stride,padding,width,groups,False{},base,left,image,y):
  match count:
    case 0n: Unit{}
    case 1n+rest:
      KernelGemm.Both{
        unstored_columns(~Holder,~value,U32.to_nat(kernel),0,input,row,padding,width,groups,base,(left + (cell / U32.max(groups,1) : U32) * stride : U32),
          (cell % U32.max(groups,1) : U32),image,y),
        unstored_cells(~Holder,~value,rest,(cell + 1 : U32),input,row,kernel,stride,padding,width,groups,base,left,image,y)}

# ---- The kernel rows of a segment ----

# A task after kernel rows: the holding the bands went back into, which still
# covers its range with the same values, and the output as a model of a tree
# and a length.
type RowedModel<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-rowed: Window.Rowed,-low: Nat,-high: Nat,-layout: Held.Spans,
  -parameters: Buffer.Model<Tile.Tile8>,-tree: Tree.Tree<Tile.Tile8>,-length: Nat> is Data:
  RowedModel{holding: Held.Holding<Tile.Tile8>,output: Buffer.Model<Tile.Tile8>,
    packed: {rowed == Window.Rowed{Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)} : Window.Rowed},
    tiled: Held.Tiled(~Tile.Tile8,holding,low,high),values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),
    ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},
    tree: {Buffer.tree(~Tile.Tile8,output) == tree : Tree.Tree<Tile.Tile8>},length: {Buffer.length(~Tile.Tile8,output) == length : Nat}}

def rowed_holding(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-rowed: Window.Rowed,-low: Nat,-high: Nat,-layout: Held.Spans,-parameters: Buffer.Model<Tile.Tile8>,-tree: Tree.Tree<Tile.Tile8>,-length: Nat,+model: RowedModel<Holder,value,holder,rowed,low,high,layout,parameters,tree,length>) ->
  Held.Holding<Tile.Tile8>:
  RowedModel{+holding,+output,+packed,+tiled,+values,+ordered,+tree,+length} = model
  holding

def rowed_output(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-rowed: Window.Rowed,-low: Nat,-high: Nat,-layout: Held.Spans,-parameters: Buffer.Model<Tile.Tile8>,-tree: Tree.Tree<Tile.Tile8>,-length: Nat,+model: RowedModel<Holder,value,holder,rowed,low,high,layout,parameters,tree,length>) ->
  Buffer.Model<Tile.Tile8>:
  RowedModel{+holding,+output,+packed,+tiled,+values,+ordered,+tree,+length} = model
  output

def rowed_packed(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-rowed: Window.Rowed,-low: Nat,-high: Nat,-layout: Held.Spans,-parameters: Buffer.Model<Tile.Tile8>,-tree: Tree.Tree<Tile.Tile8>,-length: Nat,+model: RowedModel<Holder,value,holder,rowed,low,high,layout,parameters,tree,length>) ->
  {rowed == Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,rowed,low,high,layout,parameters,tree,length,model)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,rowed,low,high,layout,parameters,tree,length,model))} : Window.Rowed}:
  RowedModel{+holding,+output,+packed,+tiled,+values,+ordered,+tree,+length} = model
  packed

def rowed_tiled(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-rowed: Window.Rowed,-low: Nat,-high: Nat,-layout: Held.Spans,-parameters: Buffer.Model<Tile.Tile8>,-tree: Tree.Tree<Tile.Tile8>,-length: Nat,+model: RowedModel<Holder,value,holder,rowed,low,high,layout,parameters,tree,length>) ->
  Held.Tiled(~Tile.Tile8,rowed_holding(~Holder,~value,holder,rowed,low,high,layout,parameters,tree,length,model),low,high):
  RowedModel{+holding,+output,+packed,+tiled,+values,+ordered,+tree,+length} = model
  tiled

def rowed_values(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-rowed: Window.Rowed,-low: Nat,-high: Nat,-layout: Held.Spans,-parameters: Buffer.Model<Tile.Tile8>,-tree: Tree.Tree<Tile.Tile8>,-length: Nat,+model: RowedModel<Holder,value,holder,rowed,low,high,layout,parameters,tree,length>) ->
  Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,rowed_holding(~Holder,~value,holder,rowed,low,high,layout,parameters,tree,length,model)):
  RowedModel{+holding,+output,+packed,+tiled,+values,+ordered,+tree,+length} = model
  values

def rowed_ordered(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-rowed: Window.Rowed,-low: Nat,-high: Nat,-layout: Held.Spans,-parameters: Buffer.Model<Tile.Tile8>,-tree: Tree.Tree<Tile.Tile8>,-length: Nat,+model: RowedModel<Holder,value,holder,rowed,low,high,layout,parameters,tree,length>) ->
  {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,rowed_holding(~Holder,~value,holder,rowed,low,high,layout,parameters,tree,length,model))) == layout : Held.Spans}:
  RowedModel{+holding,+output,+packed,+tiled,+values,+ordered,+tree,+length} = model
  ordered

def rowed_tree(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-rowed: Window.Rowed,-low: Nat,-high: Nat,-layout: Held.Spans,-parameters: Buffer.Model<Tile.Tile8>,-tree: Tree.Tree<Tile.Tile8>,-length: Nat,+model: RowedModel<Holder,value,holder,rowed,low,high,layout,parameters,tree,length>) ->
  {Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,rowed,low,high,layout,parameters,tree,length,model)) == tree : Tree.Tree<Tile.Tile8>}:
  RowedModel{+holding,+output,+packed,+tiled,+values,+ordered,+tree,+length} = model
  tree

def rowed_length(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-rowed: Window.Rowed,-low: Nat,-high: Nat,-layout: Held.Spans,-parameters: Buffer.Model<Tile.Tile8>,-tree: Tree.Tree<Tile.Tile8>,-length: Nat,+model: RowedModel<Holder,value,holder,rowed,low,high,layout,parameters,tree,length>) ->
  {Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,rowed,low,high,layout,parameters,tree,length,model)) == length : Nat}:
  RowedModel{+holding,+output,+packed,+tiled,+values,+ordered,+tree,+length} = model
  length

# A kernel row on a band runs the cells on the lent buffer and gives the band
# back.
def band_eta(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,band: Move.Band<Tile.Tile8>,+kind: Window.Sampling,+taps: Window.Taps,+start: U32,+count: U32,+image: U32,
  +top: U32,+left: U32,+weights: U32,parameters: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) ->
  {Window.row_banded(~operation,kind,taps,Window.Segment{start,count,image,top,left},True{},weights,band,parameters,output) ==
    Window.swept_back(Held.band_loan(~Tile.Tile8,band),Window.cells(~operation,U32.to_nat(count),0,kind,Window.Line{taps,True{},Held.band_offset(~Tile.Tile8,band),left,weights},
      start,Window.Swept{Held.band_buffer(~Tile.Tile8,band),parameters,output})) : Window.Rowed}:
  match band:
    case Move.Band{lent,buffer,+offset}: {==}

# The tree of the output after one kernel row over a segment.
def row_tree(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+kind: Window.Sampling,+kernel: U32,
  +stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,+kernel_row: U32,+weights: Tree.Tree<Tile.Tile8>,
  +tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),U32.to_nat(count),(start + 0 : U32),
    Row{Input{holder,height,width,groups},Weights{weights,kernel,groups},of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},U32.to_nat(image),top,left,
      kernel_row,start},tree)

def stored_joined(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,
  +low: Nat,+high: Nat,+layout: Held.Spans,+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+start: U32,+count: U32,
  +image: U32,+top: U32,+left: U32,+kernel_row: U32,+parameters: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  swept: SweptModel<Window.cells(~operation,U32.to_nat(count),0,kind,Window.Line{Window.Taps{kernel,stride,padding,height,width,groups},True{},
      Held.band_offset(~Tile.Tile8,row_band(holding,image,height,(top + kernel_row - padding : U32),width,groups,Window.sample_pad(kind))),left,(kernel_row * kernel : U32)},start,
      Window.Swept{Buffer.pack(~Tile.Tile8,band_row(row_band(holding,image,height,(top + kernel_row - padding : U32),width,groups,Window.sample_pad(kind)))),
        Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)}),
    band_row(row_band(holding,image,height,(top + kernel_row - padding : U32),width,groups,Window.sample_pad(kind))),parameters,
    row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),
      Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)>,
  covered: Held.Covered<Tile.Tile8,Holder,value,holder,Move.returned_holding(~Tile.Tile8,
      Held.band_loan(~Tile.Tile8,row_band(holding,image,height,(top + kernel_row - padding : U32),width,groups,Window.sample_pad(kind))),
      Buffer.pack(~Tile.Tile8,band_row(row_band(holding,image,height,(top + kernel_row - padding : U32),width,groups,Window.sample_pad(kind))))),low,high,layout>) ->
  RowedModel<Holder,value,holder,Window.row_applied(~operation,True{},kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},
      (top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),
    low,high,layout,parameters,
    row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),
      Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)>:
  SweptModel{+written,+swept_packed,+written_tree,+written_length} = swept
  Held.Covered{+next,+next_packed,+next_tiled,+next_values,+next_ordered} = covered
  +y = (top + kernel_row - padding : U32)
  +taps = {Window.Taps{kernel,stride,padding,height,width,groups} : Window.Taps}
  +line = {Window.Line{taps,True{},Held.band_offset(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),left,(kernel_row * kernel : U32)} : Window.Line}
  RowedModel{next,written,
    Equal.trans(Window.Rowed,
      Window.row_banded(~operation,kind,taps,Window.Segment{start,count,image,top,left},True{},(kernel_row * kernel : U32),
        row_band(holding,image,height,y,width,groups,Window.sample_pad(kind)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),
      Window.swept_back(Held.band_loan(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
        Window.cells(~operation,U32.to_nat(count),0,kind,line,start,Window.Swept{Held.band_buffer(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
          Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)})),
      Window.Rowed{Held.pack(~Tile.Tile8,next),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,written)},
      band_eta(~operation,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind)),kind,taps,start,count,image,top,left,(kernel_row * kernel : U32),
        Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),
      Equal.trans(Window.Rowed,
        Window.swept_back(Held.band_loan(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
          Window.cells(~operation,U32.to_nat(count),0,kind,line,start,Window.Swept{Held.band_buffer(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
            Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)})),
        Window.swept_back(Held.band_loan(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
          Window.cells(~operation,U32.to_nat(count),0,kind,line,start,Window.Swept{Buffer.pack(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,Window.sample_pad(kind)))),
            Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)})),
        Window.Rowed{Held.pack(~Tile.Tile8,next),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,written)},
        Equal.cong(Storage.Buffer<Tile.Tile8>,Window.Rowed,
          buffer => Window.swept_back(Held.band_loan(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
            Window.cells(~operation,U32.to_nat(count),0,kind,line,start,Window.Swept{buffer,Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)})),
          Held.band_buffer(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
          Buffer.pack(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,Window.sample_pad(kind)))),
          Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,Window.sample_pad(kind)))),
            Held.band_buffer(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
            Buffer.remodeled(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind)))))),
        Equal.trans(Window.Rowed,
          Window.swept_back(Held.band_loan(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
            Window.cells(~operation,U32.to_nat(count),0,kind,line,start,Window.Swept{Buffer.pack(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,Window.sample_pad(kind)))),
              Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)})),
          Window.Rowed{Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
            Buffer.pack(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))))),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,written)},
          Window.Rowed{Held.pack(~Tile.Tile8,next),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,written)},
          Equal.cong(Window.Swept,Window.Rowed,swept => Window.swept_back(Held.band_loan(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),swept),
            Window.cells(~operation,U32.to_nat(count),0,kind,line,start,Window.Swept{Buffer.pack(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,Window.sample_pad(kind)))),
              Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)}),
            Window.Swept{Buffer.pack(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,Window.sample_pad(kind)))),Buffer.pack(~Tile.Tile8,parameters),
              Buffer.pack(~Tile.Tile8,written)},
            swept_packed),
          Equal.cong(Move.Holding<Tile.Tile8>,Window.Rowed,window => Window.Rowed{window,Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,written)},
            Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
              Buffer.pack(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))))),
            Held.pack(~Tile.Tile8,next),next_packed)))),
    next_tiled,next_values,next_ordered,written_tree,written_length}

# A kernel row on a stored input row: the band lent from the holding goes
# back into it and the cells of the segment take the row.
def stored_row(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),+layout: Held.Spans,
  +ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,
  +width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,+kernel_row: U32,+parameters: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +groups_room: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(capacity)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},
  +stored: {U32.is_lt((top + kernel_row - padding : U32),height) == True{} : Bool},
  +room: RowRoom<low,high,capacity,image,height,(top + kernel_row - padding : U32),width,groups>) ->
  RowedModel<Holder,value,holder,Window.row_applied(~operation,True{},kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},
      (top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),
    low,high,layout,parameters,
    row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),
      Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)>:
  +y = (top + kernel_row - padding : U32)
  stored_joined(~Holder,~value,~operation,holder,holding,low,high,layout,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,parameters,output,
    cells_model(~Holder,~value,~operation,U32.to_nat(count),0,kind,kernel,stride,padding,height,width,groups,True{},
      Held.band_offset(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),left,kernel_row,start,Input{holder,height,width,groups},
      U32.to_nat(image),top,band_row(row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),parameters,output,stored,
      band_cells(~Holder,~value,U32.to_nat(count),0,holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,Window.sample_pad(kind),kernel,stride,padding,height,width,
        groups,image,y,room,left,groups_room,groups_positive)),
    {%band_model(holding,image,height,y,width,groups,Window.sample_pad(kind)) :
      Held.Covered<Tile.Tile8,Holder,value,holder,Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,_),Buffer.pack(~Tile.Tile8,band_row(_))),low,high,layout>
    Held.band_returned(~Tile.Tile8,~Holder,~value,holder,holding,((image * height + y) * width * groups : U32),(width * groups : U32),Window.sample_pad(kind),
      Held.fresh_spare(~Tile.Tile8,Window.sample_pad(kind)),low,high,tiled,values,layout,ordered,band_row(model_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),
      Buffer.remodel_content(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,model_band(holding,image,height,y,width,groups,Window.sample_pad(kind))))) :
    Held.Covered<Tile.Tile8,Holder,value,holder,Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))),Buffer.pack(~Tile.Tile8,band_row(row_band(holding,image,height,y,width,groups,Window.sample_pad(kind))))),
      low,high,layout>})

def unstored_joined(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,
  +low: Nat,+high: Nat,+tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),+layout: Held.Spans,
  +ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,
  +width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,+kernel_row: U32,+parameters: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  swept: SweptModel<Window.cells(~operation,U32.to_nat(count),0,kind,Window.Line{Window.Taps{kernel,stride,padding,height,width,groups},False{},0,left,(kernel_row * kernel : U32)},
      start,Window.Swept{Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n))),Buffer.pack(~Tile.Tile8,parameters),
        Buffer.pack(~Tile.Tile8,output)}),
    Buffer.remodel(~Tile.Tile8,Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n)),parameters,
    row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),
      Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)>) ->
  RowedModel<Holder,value,holder,Window.row_applied(~operation,False{},kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},
      (top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),
    low,high,layout,parameters,
    row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),
      Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)>:
  SweptModel{+written,+swept_packed,+written_tree,+written_length} = swept
  +line = {Window.Line{Window.Taps{kernel,stride,padding,height,width,groups},False{},0,left,(kernel_row * kernel : U32)} : Window.Line}
  RowedModel{holding,written,
    Equal.trans(Window.Rowed,
      Window.rowed_unstored(Held.pack(~Tile.Tile8,holding),Window.cells(~operation,U32.to_nat(count),0,kind,line,start,
        Window.Swept{Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)})),
      Window.rowed_unstored(Held.pack(~Tile.Tile8,holding),Window.cells(~operation,U32.to_nat(count),0,kind,line,start,
        Window.Swept{Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n))),Buffer.pack(~Tile.Tile8,parameters),
          Buffer.pack(~Tile.Tile8,output)})),
      Window.Rowed{Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,written)},
      Equal.cong(Storage.Buffer<Tile.Tile8>,Window.Rowed,
        buffer => Window.rowed_unstored(Held.pack(~Tile.Tile8,holding),Window.cells(~operation,U32.to_nat(count),0,kind,line,start,
          Window.Swept{buffer,Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)})),
        Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n),Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n))),
        Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n))),
          Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n),Buffer.remodeled(~Tile.Tile8,Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n)))),
      Equal.cong(Window.Swept,Window.Rowed,swept => Window.rowed_unstored(Held.pack(~Tile.Tile8,holding),swept),
        Window.cells(~operation,U32.to_nat(count),0,kind,line,start,
          Window.Swept{Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n))),Buffer.pack(~Tile.Tile8,parameters),
            Buffer.pack(~Tile.Tile8,output)}),
        Window.Swept{Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n))),Buffer.pack(~Tile.Tile8,parameters),
          Buffer.pack(~Tile.Tile8,written)},
        swept_packed)),
    tiled,values,ordered,written_tree,written_length}

# A kernel row on an input row outside the input reads pad and keeps the
# holding.
def unstored_row(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,
  +low: Nat,+high: Nat,+tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),+layout: Held.Spans,
  +ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,
  +width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,+kernel_row: U32,+parameters: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +outside: {U32.is_lt((top + kernel_row - padding : U32),height) == False{} : Bool}) ->
  RowedModel<Holder,value,holder,Window.row_applied(~operation,False{},kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},
      (top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),
    low,high,layout,parameters,
    row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),
      Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)>:
  +empty = Buffer.remodel(~Tile.Tile8,Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n))
  unstored_joined(~Holder,~value,~operation,holder,holding,low,high,tiled,values,layout,ordered,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,
    parameters,output,
    cells_model(~Holder,~value,~operation,U32.to_nat(count),0,kind,kernel,stride,padding,height,width,groups,False{},0,left,kernel_row,start,Input{holder,height,width,groups},
      U32.to_nat(image),top,empty,parameters,output,outside,
      unstored_cells(~Holder,~value,U32.to_nat(count),0,Input{holder,height,width,groups},Buffer.tree(~Tile.Tile8,empty),kernel,stride,padding,width,groups,0,left,U32.to_nat(image),
        (top + kernel_row - padding : U32))))

# One kernel row over a segment.
def row_model(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,stored: Bool,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),+layout: Held.Spans,
  +ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,
  +width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,+kernel_row: U32,+parameters: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +groups_room: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(capacity)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},
  +classified: {U32.is_lt((top + kernel_row - padding : U32),height) == stored : Bool},
  +room: Boolean.When(stored,RowRoom<low,high,capacity,image,height,(top + kernel_row - padding : U32),width,groups>)) ->
  RowedModel<Holder,value,holder,Window.row_applied(~operation,stored,kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},
      (top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),
    low,high,layout,parameters,
    row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),
      Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)>:
  match stored:
    case True{}:
      stored_row(~Holder,~value,~operation,holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,kind,kernel,stride,padding,height,width,groups,start,
        count,image,top,left,kernel_row,parameters,output,groups_room,groups_positive,classified,room)
    case False{}:
      unstored_row(~Holder,~value,~operation,holder,holding,low,high,tiled,values,layout,ordered,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,
        kernel_row,parameters,output,classified)

# The input rows a segment reads: each stored one lies in the holding.
def RowsRoom(count: Nat,+kernel_row: U32,+low: Nat,+high: Nat,+capacity: U32,+image: U32,+height: U32,+top: U32,+padding: U32,+width: U32,+groups: U32) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      KernelGemm.Both<Boolean.When(U32.is_lt((top + kernel_row - padding : U32),height),RowRoom<low,high,capacity,image,height,(top + kernel_row - padding : U32),width,groups>),
        RowsRoom(rest,(kernel_row + 1 : U32),low,high,capacity,image,height,top,padding,width,groups)>

# The tree of the output after the kernel rows from kernel_row on.
def rows_tree(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,+kernel_row: U32,+holder: Holder,+kind: Window.Sampling,+kernel: U32,
  +stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,+weights: Tree.Tree<Tile.Tile8>,
  tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  match fuel:
    case 0n: tree
    case 1n+rest:
      rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,weights,
        row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree))

# The kernel rows from kernel_row on over a segment, in order.
def rows_model(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,+kernel_row: U32,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),+layout: Held.Spans,
  +ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,
  +width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,+parameters: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +groups_room: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(capacity)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},
  rooms: RowsRoom(fuel,kernel_row,low,high,capacity,image,height,top,padding,width,groups)) ->
  RowedModel<Holder,value,holder,Window.kernel_rows(~operation,fuel,kernel_row,kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,fuel,kernel_row,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)>:
  match fuel:
    case 0n: RowedModel{holding,output,{==},tiled,values,ordered,{==},{==}}
    case 1n+ +rest:
      KernelGemm.Both{+room,+later} = rooms
      +first = row_model(~Holder,~value,~operation,U32.is_lt((top + kernel_row - padding : U32),height),holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,kind,kernel,stride,
        padding,height,width,groups,start,count,image,top,left,kernel_row,parameters,output,groups_room,groups_positive,{==},room)
      +next = rows_model(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first),low,high,capacity,capacity_ok,bounded,rowed_tiled(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first),rowed_values(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first),layout,rowed_ordered(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first),kind,kernel,
        stride,padding,height,width,groups,start,count,image,top,left,parameters,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first),groups_room,groups_positive,later)
      RowedModel{rowed_holding(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next),rowed_output(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next),
        Equal.trans(Window.Rowed,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output))),Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),
          Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next))},
          Equal.cong(Window.Rowed,Window.Rowed,rowed => Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},rowed),Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),
            Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))},rowed_packed(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),
          rowed_packed(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next)),
        rowed_tiled(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next),rowed_values(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next),rowed_ordered(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next),
        Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next)),rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output))),rowed_tree(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next),
          Equal.cong(Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,tree => rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),tree),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),rowed_tree(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),
        Equal.trans(Nat,Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next)),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.length(~Tile.Tile8,output),rowed_length(~Holder,~value,holder,Window.kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},Window.Rowed{Held.pack(~Tile.Tile8,rowed_holding(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}),low,high,layout,parameters,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))),Buffer.length(~Tile.Tile8,rowed_output(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first)),next),rowed_length(~Holder,~value,holder,Window.row_applied(~operation,U32.is_lt((top + kernel_row - padding : U32),height),kind,Window.Taps{kernel,stride,padding,height,width,groups},Window.Segment{start,count,image,top,left},(top + kernel_row - padding : U32),(kernel_row * kernel : U32),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters),Buffer.pack(~Tile.Tile8,output)),low,high,layout,parameters,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output),first))}

# ---- What the kernel rows leave in the cells ----

def row_capacity(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+kind: Window.Sampling,+kernel: U32,
  +stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,+kernel_row: U32,+weights: Tree.Tree<Tile.Tile8>,
  +tree: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree)) ==
    Tree.capacity(Tile.Tile8,tree) : U32}:
  Updates.updated_capacity(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),U32.to_nat(count),(start + 0 : U32),
    Row{Input{holder,height,width,groups},Weights{weights,kernel,groups},of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},U32.to_nat(image),top,left,
      kernel_row,start},tree)

def row_balanced(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+kind: Window.Sampling,+kernel: U32,
  +stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,+kernel_row: U32,+weights: Tree.Tree<Tile.Tile8>,
  +tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tree)) ->
  Shape.Balanced(Tile.Tile8,depth,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree)):
  Updates.updated_balanced(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),U32.to_nat(count),(start + 0 : U32),
    Row{Input{holder,height,width,groups},Weights{weights,kernel,groups},of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},U32.to_nat(image),top,left,
      kernel_row,start},tree,depth,balanced)

def rows_capacity(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,+kernel_row: U32,+holder: Holder,
  +kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,
  +weights: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,rows_tree(~Holder,~value,~operation,fuel,kernel_row,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,weights,tree)) ==
    Tree.capacity(Tile.Tile8,tree) : U32}:
  match fuel:
    case 0n: {==}
    case 1n+rest:
      +next = row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree)
      Equal.trans(U32,
        Tree.capacity(Tile.Tile8,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,
          weights,next)),
        Tree.capacity(Tile.Tile8,next),Tree.capacity(Tile.Tile8,tree),
        rows_capacity(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,weights,next),
        row_capacity(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree))

def rows_balanced(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,+kernel_row: U32,+holder: Holder,
  +kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,
  +weights: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tree)) ->
  Shape.Balanced(Tile.Tile8,depth,rows_tree(~Holder,~value,~operation,fuel,kernel_row,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,weights,
    tree)):
  match fuel:
    case 0n: balanced
    case 1n+rest:
      rows_balanced(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,weights,
        row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree),depth,
        row_balanced(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree,depth,balanced))

# The first cell of a segment as a word.
def start_word(+start: U32,+offset: Nat,+start_exact: {U32.to_nat(start) == offset : Nat}) -> {(start + 0 : U32) == U32.from_nat(offset) : U32}:
  Equal.trans(U32,(start + 0 : U32),start,U32.from_nat(offset),Arithmetic.add_zero(start),
    Equal.trans(U32,start,U32.from_nat(U32.to_nat(start)),U32.from_nat(offset),Equal.sym(U32,U32.from_nat(U32.to_nat(start)),start,Arithmetic.native_roundtrip(start)),
      Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(start),offset,start_exact)))

# Cell offset + index less the first cell is index.
def cell_word(+start: U32,+offset: Nat,+index: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +start_exact: {U32.to_nat(start) == offset : Nat},+room: {Nat.is_le(Nat.add(offset,index),U32.to_nat(capacity)) == True{} : Bool}) ->
  {(U32.from_nat(Nat.add(offset,index)) - start : U32) == U32.from_nat(index) : U32}:
  +index_room = Order.transitive(index,Nat.add(offset,index),U32.to_nat(capacity),Order.right_le(offset,index),room)
  +sum = Equal.trans(U32,U32.from_nat(Nat.add(offset,index)),U32.from_nat(Nat.add(U32.to_nat(start),U32.to_nat(U32.from_nat(index)))),(start + U32.from_nat(index) : U32),
    Equal.cong(Nat,U32,U32.from_nat,Nat.add(offset,index),Nat.add(U32.to_nat(start),U32.to_nat(U32.from_nat(index))),
      Equal.sym(Nat,Nat.add(U32.to_nat(start),U32.to_nat(U32.from_nat(index))),Nat.add(offset,index),
        Equal.trans(Nat,Nat.add(U32.to_nat(start),U32.to_nat(U32.from_nat(index))),Nat.add(offset,U32.to_nat(U32.from_nat(index))),Nat.add(offset,index),
          Equal.cong(Nat,Nat,first => Nat.add(first,U32.to_nat(U32.from_nat(index))),U32.to_nat(start),offset,start_exact),
          Equal.cong(Nat,Nat,local => Nat.add(offset,local),U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,capacity,capacity_ok,index_room))))),
    Equal.sym(U32,(start + U32.from_nat(index) : U32),U32.from_nat(Nat.add(U32.to_nat(start),U32.to_nat(U32.from_nat(index)))),Arithmetic.native_add(start,U32.from_nat(index))))
  Equal.trans(U32,(U32.from_nat(Nat.add(offset,index)) - start : U32),((start + U32.from_nat(index) : U32) - start : U32),U32.from_nat(index),
    Equal.cong(U32,U32,word => (word - start : U32),U32.from_nat(Nat.add(offset,index)),(start + U32.from_nat(index) : U32),sum),
    Arithmetic.subtract_first(start,U32.from_nat(index)))

# One kernel row adds its row to every cell of its segment.
def row_inside(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+kind: Window.Sampling,+kernel: U32,
  +stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,+kernel_row: U32,+weights: Tree.Tree<Tile.Tile8>,
  +tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(Tile.Tile8,tree) == capacity : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,tree),
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+offset: Nat,+start_exact: {U32.to_nat(start) == offset : Nat},
  +room: {Nat.is_le(Nat.add(offset,U32.to_nat(count)),U32.to_nat(capacity)) == True{} : Bool},+index: Nat,+selected: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree),
      U32.from_nat(Nat.add(offset,index))) ==
    Values.row_value(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,Input{holder,height,width,groups},
      Weights{weights,kernel,groups},of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},
      cell_at(Window.Taps{kernel,stride,padding,height,width,groups},U32.to_nat(image),top,left,U32.from_nat(index)),kernel_row,
      Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(offset,index)))) : Tile.Tile8}:
  +taps = {Window.Taps{kernel,stride,padding,height,width,groups} : Window.Taps}
  +reader = {Row{Input{holder,height,width,groups},Weights{weights,kernel,groups},of_sampling(kind),taps,U32.to_nat(image),top,left,kernel_row,start} : Row<Holder>}
  +query = U32.from_nat(Nat.add(offset,index))
  +old = Tree.read(Tile.Tile8,tree,query)
  %Equal.sym(U32,(start + 0 : U32),U32.from_nat(offset),start_word(start,offset,start_exact)) :
    {Tree.read(Tile.Tile8,Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),U32.to_nat(count),_,reader,tree),query) ==
      Values.row_value(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,Input{holder,height,width,groups},
        Weights{weights,kernel,groups},of_sampling(kind),taps,cell_at(taps,U32.to_nat(image),top,left,U32.from_nat(index)),kernel_row,old) : Tile.Tile8}
  Equal.trans(Tile.Tile8,
    Tree.read(Tile.Tile8,Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),U32.to_nat(count),U32.from_nat(offset),reader,
      tree),query),
    Values.row_value(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,Input{holder,height,width,groups},
      Weights{weights,kernel,groups},of_sampling(kind),taps,cell_at(taps,U32.to_nat(image),top,left,(query - start : U32)),kernel_row,old),
    Values.row_value(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,Input{holder,height,width,groups},
      Weights{weights,kernel,groups},of_sampling(kind),taps,cell_at(taps,U32.to_nat(image),top,left,U32.from_nat(index)),kernel_row,old),
    Updates.updated_inside(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),U32.to_nat(count),offset,index,reader,tree,depth,capacity,
      capacity_equal,balanced,limit,capacity_ok,room,selected),
    Equal.cong(U32,Tile.Tile8,
      cell => Values.row_value(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,Input{holder,height,width,groups},
        Weights{weights,kernel,groups},of_sampling(kind),taps,cell_at(taps,U32.to_nat(image),top,left,cell),kernel_row,old),
      (query - start : U32),U32.from_nat(index),
      cell_word(start,offset,index,capacity,capacity_ok,start_exact,
        Order.transitive(Nat.add(offset,index),Nat.add(offset,U32.to_nat(count)),U32.to_nat(capacity),
          Order.offset_le(offset,index,U32.to_nat(count),Bounds.strict_le(index,U32.to_nat(count),selected)),room))))

# One kernel row keeps every cell outside its segment.
def row_outside(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+kind: Window.Sampling,+kernel: U32,
  +stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,+kernel_row: U32,+weights: Tree.Tree<Tile.Tile8>,
  +tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(Tile.Tile8,tree) == capacity : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,tree),
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+offset: Nat,+start_exact: {U32.to_nat(start) == offset : Nat},
  +room: {Nat.is_le(Nat.add(offset,U32.to_nat(count)),U32.to_nat(capacity)) == True{} : Bool},+query: U32,+query_inside: {U32.is_lt(query,capacity) == True{} : Bool},
  +separate: Interval.Outside<offset,U32.to_nat(count),U32.to_nat(query)>) ->
  {Tree.read(Tile.Tile8,row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree),query) ==
    Tree.read(Tile.Tile8,tree,query) : Tile.Tile8}:
  +reader = {Row{Input{holder,height,width,groups},Weights{weights,kernel,groups},of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},U32.to_nat(image),top,
    left,kernel_row,start} : Row<Holder>}
  %Equal.sym(U32,(start + 0 : U32),U32.from_nat(offset),start_word(start,offset,start_exact)) :
    {Tree.read(Tile.Tile8,Updates.updated(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),U32.to_nat(count),_,reader,tree),query) ==
      Tree.read(Tile.Tile8,tree,query) : Tile.Tile8}
  Updates.updated_outside(~Tile.Tile8,~Row<Holder>,~Tap<Holder>,~tap_at(~Holder),~row_step(~Holder,~value,~operation),U32.to_nat(count),offset,query,reader,tree,depth,capacity,
    capacity_equal,balanced,limit,capacity_ok,room,query_inside,separate)

# The kernel rows from kernel_row on fold their rows into every cell of the
# segment.
def rows_inside(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,+kernel_row: U32,+holder: Holder,
  +kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,
  +weights: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(Tile.Tile8,tree) == capacity : U32},
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+offset: Nat,
  +start_exact: {U32.to_nat(start) == offset : Nat},+room: {Nat.is_le(Nat.add(offset,U32.to_nat(count)),U32.to_nat(capacity)) == True{} : Bool},+index: Nat,
  +selected: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,rows_tree(~Holder,~value,~operation,fuel,kernel_row,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,weights,tree),
      U32.from_nat(Nat.add(offset,index))) ==
    Values.rows(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,fuel,kernel_row,Input{holder,height,width,groups},
      Weights{weights,kernel,groups},of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},
      cell_at(Window.Taps{kernel,stride,padding,height,width,groups},U32.to_nat(image),top,left,U32.from_nat(index)),
      Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(offset,index)))) : Tile.Tile8}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +taps = {Window.Taps{kernel,stride,padding,height,width,groups} : Window.Taps}
      +cell = cell_at(taps,U32.to_nat(image),top,left,U32.from_nat(index))
      +query = U32.from_nat(Nat.add(offset,index))
      +next = row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree)
      Equal.trans(Tile.Tile8,
        Tree.read(Tile.Tile8,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,
          weights,next),query),
        Values.rows(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,rest,(kernel_row + 1 : U32),
          Input{holder,height,width,groups},Weights{weights,kernel,groups},of_sampling(kind),taps,cell,Tree.read(Tile.Tile8,next,query)),
        Values.rows(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,rest,(kernel_row + 1 : U32),
          Input{holder,height,width,groups},Weights{weights,kernel,groups},of_sampling(kind),taps,cell,
          Values.row_value(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,Input{holder,height,width,groups},
            Weights{weights,kernel,groups},of_sampling(kind),taps,cell,kernel_row,Tree.read(Tile.Tile8,tree,query))),
        rows_inside(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,weights,next,depth,
          capacity,
          Equal.trans(U32,Tree.capacity(Tile.Tile8,next),Tree.capacity(Tile.Tile8,tree),capacity,
            row_capacity(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree),capacity_equal),
          row_balanced(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree,depth,balanced),
          limit,capacity_ok,offset,start_exact,room,index,selected),
        Equal.cong(Tile.Tile8,Tile.Tile8,
          started => Values.rows(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,rest,(kernel_row + 1 : U32),
            Input{holder,height,width,groups},Weights{weights,kernel,groups},of_sampling(kind),taps,cell,started),
          Tree.read(Tile.Tile8,next,query),
          Values.row_value(~Tile.Tile8,~operation,~Tile.update_lanes,~Input<Holder>,~input_read(~Holder,~value),~Weights,~weight_read,Input{holder,height,width,groups},
            Weights{weights,kernel,groups},of_sampling(kind),taps,cell,kernel_row,Tree.read(Tile.Tile8,tree,query)),
          row_inside(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree,depth,capacity,
            capacity_equal,balanced,limit,capacity_ok,offset,start_exact,room,index,selected)))

# The kernel rows keep every cell outside their segment.
def rows_outside(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,+kernel_row: U32,+holder: Holder,
  +kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+start: U32,+count: U32,+image: U32,+top: U32,+left: U32,
  +weights: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(Tile.Tile8,tree) == capacity : U32},
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+offset: Nat,
  +start_exact: {U32.to_nat(start) == offset : Nat},+room: {Nat.is_le(Nat.add(offset,U32.to_nat(count)),U32.to_nat(capacity)) == True{} : Bool},+query: U32,
  +query_inside: {U32.is_lt(query,capacity) == True{} : Bool},+separate: Interval.Outside<offset,U32.to_nat(count),U32.to_nat(query)>) ->
  {Tree.read(Tile.Tile8,rows_tree(~Holder,~value,~operation,fuel,kernel_row,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,weights,tree),query) ==
    Tree.read(Tile.Tile8,tree,query) : Tile.Tile8}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +next = row_tree(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree)
      Equal.trans(Tile.Tile8,
        Tree.read(Tile.Tile8,rows_tree(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,
          weights,next),query),
        Tree.read(Tile.Tile8,next,query),Tree.read(Tile.Tile8,tree,query),
        rows_outside(~Holder,~value,~operation,rest,(kernel_row + 1 : U32),holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,weights,next,depth,
          capacity,
          Equal.trans(U32,Tree.capacity(Tile.Tile8,next),Tree.capacity(Tile.Tile8,tree),capacity,
            row_capacity(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree),capacity_equal),
          row_balanced(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree,depth,balanced),
          limit,capacity_ok,offset,start_exact,room,query,query_inside,separate),
        row_outside(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,start,count,image,top,left,kernel_row,weights,tree,depth,capacity,
          capacity_equal,balanced,limit,capacity_ok,offset,start_exact,room,query,query_inside,separate))
