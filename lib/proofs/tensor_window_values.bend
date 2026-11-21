# The value of a window operation at one output cell, for any element.
# Pooling folds the samples of every kernel row from pad in increasing
# column, and the output folds the rows from pad in increasing row.
# Weighting adds every sample times the weight of its tap to the output from
# zero, in row-major order of the taps, and then the bias of its channel. A
# sample outside the input reads pad. A lane of the value over tiles is the
# value over the lanes when the samples and the parameters agree lane by lane.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor_tiled_map.bend as TiledMap
import ../tensor_window.bend as Window
import ./kernel_tile_refinement.bend as TileLaws

# ---- The value of a cell ----

# What a window does with its samples, and the element it starts from.
type Sampling<-Element: Data> is Data:
  Pooling{pad: Element}
  Weighting{zero: Element}

def start(~Element: Data,kind: Sampling<Element>) -> Element:
  match kind:
    case Pooling{pad}: pad
    case Weighting{zero}: zero

# A kernel row folded: the output so far and, for pooling, the row so far.
type Folded<-Element: Data> is Data:
  Folded{output: Element,line: Element}

def tapped(~Element: Data,~operation: Element -> Element -> Element,~update: Element -> Element -> Element -> Element,kind: Sampling<Element>,folded: Folded<Element>,
  value: Element,weight: Element) -> Folded<Element>:
  match kind folded:
    case Pooling{pad} Folded{output,line}: Folded{output,operation(line,value)}
    case Weighting{zero} Folded{output,line}: Folded{update(output,value,weight),line}

def closed(~Element: Data,~operation: Element -> Element -> Element,kind: Sampling<Element>,folded: Folded<Element>) -> Element:
  match kind folded:
    case Pooling{pad} Folded{output,line}: operation(output,line)
    case Weighting{zero} Folded{output,line}: output

def kernel_of(taps: Window.Taps) -> U32:
  Window.Taps{kernel,stride,padding,height,width,groups} = taps
  kernel

def padding_of(taps: Window.Taps) -> U32:
  Window.Taps{kernel,stride,padding,height,width,groups} = taps
  padding

# The window of one output cell: the image and the channel it reads, and the
# row and the column its window starts at before padding.
type Cell is Data:
  Cell{image: Nat,channel: Nat,top: U32,left: U32}

def channel_of(cell: Cell) -> Nat:
  Cell{image,channel,top,left} = cell
  channel

def top_of(cell: Cell) -> U32:
  Cell{image,channel,top,left} = cell
  top

# The input row a kernel row of a window reads.
def row_of(+taps: Window.Taps,+cell: Cell,+kernel_row: U32) -> U32:
  (top_of(cell) + kernel_row - padding_of(taps) : U32)

# The sample at a kernel column of a window on an input row: pad outside the
# input.
def sample(~Element: Data,~Source: Data,~read: Source -> Nat -> Nat -> U32 -> U32 -> Element,+source: Source,+taps: Window.Taps,+cell: Cell,+row: U32,+column: U32,
  +pad: Element) -> Element:
  Window.Taps{kernel,stride,+padding,+height,+width,groups} = taps
  Cell{+image,+channel,top,+left} = cell
  +x = (left + column - padding : U32)
  Bool.pick(Element,U32.is_lt(row,height) && U32.is_lt(x,width),read(source,image,channel,row,x),pad)

# The kernel columns of a kernel row from column on.
def columns(~Element: Data,~operation: Element -> Element -> Element,~update: Element -> Element -> Element -> Element,~Source: Data,
  ~read: Source -> Nat -> Nat -> U32 -> U32 -> Element,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> Element,count: Nat,+column: U32,+source: Source,
  +parameters: Parameters,+kind: Sampling<Element>,+taps: Window.Taps,+cell: Cell,+row: U32,+kernel_row: U32,folded: Folded<Element>) -> Folded<Element>:
  match count:
    case 0n: folded
    case 1n+rest:
      columns(~Element,~operation,~update,~Source,~read,~Parameters,~weight,rest,(column + 1 : U32),source,parameters,kind,taps,cell,row,kernel_row,
        tapped(~Element,~operation,~update,kind,folded,sample(~Element,~Source,~read,source,taps,cell,row,column,start(~Element,kind)),
          weight(parameters,channel_of(cell),kernel_row,column)))

# One kernel row of a window added to the output so far.
def row_value(~Element: Data,~operation: Element -> Element -> Element,~update: Element -> Element -> Element -> Element,~Source: Data,
  ~read: Source -> Nat -> Nat -> U32 -> U32 -> Element,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> Element,+source: Source,+parameters: Parameters,
  +kind: Sampling<Element>,+taps: Window.Taps,+cell: Cell,+kernel_row: U32,accumulator: Element) -> Element:
  closed(~Element,~operation,kind,columns(~Element,~operation,~update,~Source,~read,~Parameters,~weight,U32.to_nat(kernel_of(taps)),0,source,parameters,kind,taps,cell,
    row_of(taps,cell,kernel_row),kernel_row,Folded{accumulator,start(~Element,kind)}))

# The kernel rows of a window from kernel_row on.
def rows(~Element: Data,~operation: Element -> Element -> Element,~update: Element -> Element -> Element -> Element,~Source: Data,
  ~read: Source -> Nat -> Nat -> U32 -> U32 -> Element,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> Element,count: Nat,+kernel_row: U32,+source: Source,
  +parameters: Parameters,+kind: Sampling<Element>,+taps: Window.Taps,+cell: Cell,accumulator: Element) -> Element:
  match count:
    case 0n: accumulator
    case 1n+rest:
      rows(~Element,~operation,~update,~Source,~read,~Parameters,~weight,rest,(kernel_row + 1 : U32),source,parameters,kind,taps,cell,
        row_value(~Element,~operation,~update,~Source,~read,~Parameters,~weight,source,parameters,kind,taps,cell,kernel_row,accumulator))

def finished(~Element: Data,~add: Element -> Element -> Element,~Parameters: Data,~bias: Parameters -> Nat -> Element,kind: Sampling<Element>,+parameters: Parameters,
  +channel: Nat,value: Element) -> Element:
  match kind:
    case Pooling{pad}: value
    case Weighting{zero}: add(value,bias(parameters,channel))

# The value of a window at one output cell.
def at(~Element: Data,~operation: Element -> Element -> Element,~update: Element -> Element -> Element -> Element,~add: Element -> Element -> Element,~Source: Data,
  ~read: Source -> Nat -> Nat -> U32 -> U32 -> Element,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> Element,~bias: Parameters -> Nat -> Element,
  +source: Source,+parameters: Parameters,+kind: Sampling<Element>,+taps: Window.Taps,+cell: Cell) -> Element:
  finished(~Element,~add,~Parameters,~bias,kind,parameters,channel_of(cell),
    rows(~Element,~operation,~update,~Source,~read,~Parameters,~weight,U32.to_nat(kernel_of(taps)),0,source,parameters,kind,taps,cell,start(~Element,kind)))

# ---- One lane of the value over tiles ----

# The multiply-add of one lane.
def lane_update(accumulator: F32,input: F32,weight: F32) -> F32:
  (accumulator + (input * weight : F32) : F32)

def lane_add(left: F32,right: F32) -> F32:
  (left + right : F32)

# The kind over tiles of a kind over lanes.
def lifted(kind: Sampling<F32>) -> Sampling<Tile.Tile8>:
  match kind:
    case Pooling{+pad}: Pooling{Tile.filled(pad)}
    case Weighting{+zero}: Weighting{Tile.filled(zero)}

def lane_folded(+lane: Nat,folded: Folded<Tile.Tile8>) -> Folded<F32>:
  Folded{output,line} = folded
  Folded{TileLaws.select_lane(lane,output),TileLaws.select_lane(lane,line)}

def paired_lane(~operation: F32 -> F32 -> F32,+left: Tile.Tile8,+right: Tile.Tile8,+lane: Nat) ->
  {TileLaws.select_lane(lane,TiledMap.pairs(~operation,left,right)) == operation(TileLaws.select_lane(lane,left),TileLaws.select_lane(lane,right)) : F32}:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = left
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = right
  match lane:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  match d:
                    case 0n: {==}
                    case 1n+e:
                      match e:
                        case 0n: {==}
                        case 1n+f:
                          match f:
                            case 0n: {==}
                            case 1n+g: {==}

def start_lane(+lane: Nat,+kind: Sampling<F32>) -> {TileLaws.select_lane(lane,start(~Tile.Tile8,lifted(kind))) == start(~F32,kind) : F32}:
  match kind:
    case Pooling{+pad}: TileLaws.filled_lane(lane,pad)
    case Weighting{+zero}: TileLaws.filled_lane(lane,zero)

def tapped_lane(~operation: F32 -> F32 -> F32,+lane: Nat,+kind: Sampling<F32>,+folded: Folded<Tile.Tile8>,+cell_value: Tile.Tile8,+cell_weight: Tile.Tile8,+value: F32,
  +weight: F32,+value_lane: {TileLaws.select_lane(lane,cell_value) == value : F32},+weight_lane: {TileLaws.select_lane(lane,cell_weight) == weight : F32}) ->
  {lane_folded(lane,tapped(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,lifted(kind),folded,cell_value,cell_weight)) ==
    tapped(~F32,~operation,~lane_update,kind,lane_folded(lane,folded),value,weight) : Folded<F32>}:
  match kind folded:
    case Pooling{+pad} Folded{+output,+line}:
      %value_lane : {Folded{TileLaws.select_lane(lane,output),TileLaws.select_lane(lane,TiledMap.pairs(~operation,line,cell_value))} ==
        Folded{TileLaws.select_lane(lane,output),operation(TileLaws.select_lane(lane,line),_)} : Folded<F32>}
      Equal.cong(F32,Folded<F32>,lined => Folded{TileLaws.select_lane(lane,output),lined},TileLaws.select_lane(lane,TiledMap.pairs(~operation,line,cell_value)),
        operation(TileLaws.select_lane(lane,line),TileLaws.select_lane(lane,cell_value)),paired_lane(~operation,line,cell_value,lane))
    case Weighting{+zero} Folded{+output,+line}:
      %value_lane : {Folded{TileLaws.select_lane(lane,Tile.update_lanes(output,cell_value,cell_weight)),TileLaws.select_lane(lane,line)} ==
        Folded{lane_update(TileLaws.select_lane(lane,output),_,weight),TileLaws.select_lane(lane,line)} : Folded<F32>}
      %weight_lane : {Folded{TileLaws.select_lane(lane,Tile.update_lanes(output,cell_value,cell_weight)),TileLaws.select_lane(lane,line)} ==
        Folded{lane_update(TileLaws.select_lane(lane,output),TileLaws.select_lane(lane,cell_value),_),TileLaws.select_lane(lane,line)} : Folded<F32>}
      Equal.cong(F32,Folded<F32>,summed => Folded{summed,TileLaws.select_lane(lane,line)},TileLaws.select_lane(lane,Tile.update_lanes(output,cell_value,cell_weight)),
        lane_update(TileLaws.select_lane(lane,output),TileLaws.select_lane(lane,cell_value),TileLaws.select_lane(lane,cell_weight)),
        TileLaws.updated_lane(output,cell_value,cell_weight,lane))

def closed_lane(~operation: F32 -> F32 -> F32,+lane: Nat,+kind: Sampling<F32>,+folded: Folded<Tile.Tile8>) ->
  {TileLaws.select_lane(lane,closed(~Tile.Tile8,~TiledMap.pairs(~operation),lifted(kind),folded)) == closed(~F32,~operation,kind,lane_folded(lane,folded)) : F32}:
  match kind folded:
    case Pooling{+pad} Folded{+output,+line}: paired_lane(~operation,output,line,lane)
    case Weighting{+zero} Folded{+output,+line}: {==}

# A tap whose sample and weight over tiles hold, at a lane, the sample and
# the weight over lanes.
def TapAgrees(~Cells: Data,~cell_read: Cells -> Nat -> Nat -> U32 -> U32 -> Tile.Tile8,~Values: Data,~value_read: Values -> Nat -> Nat -> U32 -> U32 -> F32,
  ~CellParameters: Data,~cell_weight: CellParameters -> Nat -> U32 -> U32 -> Tile.Tile8,~ValueParameters: Data,~value_weight: ValueParameters -> Nat -> U32 -> U32 -> F32,
  +lane: Nat,+cells: Cells,+values: Values,+cell_parameters: CellParameters,+value_parameters: ValueParameters,+kind: Sampling<F32>,+taps: Window.Taps,+image: Nat,
  +group: Nat,+channel: Nat,+top: U32,+left: U32,+row: U32,+kernel_row: U32,+column: U32) -> Data:
  KernelGemm.Both<{TileLaws.select_lane(lane,sample(~Tile.Tile8,~Cells,~cell_read,cells,taps,Cell{image,group,top,left},row,column,start(~Tile.Tile8,lifted(kind)))) ==
      sample(~F32,~Values,~value_read,values,taps,Cell{image,channel,top,left},row,column,start(~F32,kind)) : F32},
    {TileLaws.select_lane(lane,cell_weight(cell_parameters,group,kernel_row,column)) == value_weight(value_parameters,channel,kernel_row,column) : F32}>

def ColumnsAgree(~Cells: Data,~cell_read: Cells -> Nat -> Nat -> U32 -> U32 -> Tile.Tile8,~Values: Data,~value_read: Values -> Nat -> Nat -> U32 -> U32 -> F32,
  ~CellParameters: Data,~cell_weight: CellParameters -> Nat -> U32 -> U32 -> Tile.Tile8,~ValueParameters: Data,~value_weight: ValueParameters -> Nat -> U32 -> U32 -> F32,
  count: Nat,+column: U32,+lane: Nat,+cells: Cells,+values: Values,+cell_parameters: CellParameters,+value_parameters: ValueParameters,+kind: Sampling<F32>,
  +taps: Window.Taps,+image: Nat,+group: Nat,+channel: Nat,+top: U32,+left: U32,+row: U32,+kernel_row: U32) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      KernelGemm.Both<TapAgrees(~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,lane,cells,values,cell_parameters,
          value_parameters,kind,taps,image,group,channel,top,left,row,kernel_row,column),
        ColumnsAgree(~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,rest,(column + 1 : U32),lane,cells,values,
          cell_parameters,value_parameters,kind,taps,image,group,channel,top,left,row,kernel_row)>

def columns_lane(~operation: F32 -> F32 -> F32,~Cells: Data,~cell_read: Cells -> Nat -> Nat -> U32 -> U32 -> Tile.Tile8,~Values: Data,
  ~value_read: Values -> Nat -> Nat -> U32 -> U32 -> F32,~CellParameters: Data,~cell_weight: CellParameters -> Nat -> U32 -> U32 -> Tile.Tile8,~ValueParameters: Data,
  ~value_weight: ValueParameters -> Nat -> U32 -> U32 -> F32,count: Nat,+column: U32,+lane: Nat,+cells: Cells,+values: Values,+cell_parameters: CellParameters,
  +value_parameters: ValueParameters,+kind: Sampling<F32>,+taps: Window.Taps,+image: Nat,+group: Nat,+channel: Nat,+top: U32,+left: U32,+row: U32,+kernel_row: U32,
  +folded: Folded<Tile.Tile8>,
  agreed: ColumnsAgree(~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,count,column,lane,cells,values,cell_parameters,
    value_parameters,kind,taps,image,group,channel,top,left,row,kernel_row)) ->
  {lane_folded(lane,columns(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Cells,~cell_read,~CellParameters,~cell_weight,count,column,cells,cell_parameters,
      lifted(kind),taps,Cell{image,group,top,left},row,kernel_row,folded)) ==
    columns(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,count,column,values,value_parameters,kind,taps,Cell{image,channel,top,left},
      row,kernel_row,lane_folded(lane,folded)) : Folded<F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      KernelGemm.Both{+tap,+later} = agreed
      KernelGemm.Both{+sampled,+weighted} = tap
      +cell_sample = sample(~Tile.Tile8,~Cells,~cell_read,cells,taps,Cell{image,group,top,left},row,column,start(~Tile.Tile8,lifted(kind)))
      +value_sample = sample(~F32,~Values,~value_read,values,taps,Cell{image,channel,top,left},row,column,start(~F32,kind))
      +next = tapped(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,lifted(kind),folded,cell_sample,cell_weight(cell_parameters,group,kernel_row,column))
      Equal.trans(Folded<F32>,
        lane_folded(lane,columns(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Cells,~cell_read,~CellParameters,~cell_weight,rest,(column + 1 : U32),cells,
          cell_parameters,lifted(kind),taps,Cell{image,group,top,left},row,kernel_row,next)),
        columns(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,rest,(column + 1 : U32),values,value_parameters,kind,taps,
          Cell{image,channel,top,left},row,kernel_row,lane_folded(lane,next)),
        columns(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,rest,(column + 1 : U32),values,value_parameters,kind,taps,
          Cell{image,channel,top,left},row,kernel_row,
          tapped(~F32,~operation,~lane_update,kind,lane_folded(lane,folded),value_sample,value_weight(value_parameters,channel,kernel_row,column))),
        columns_lane(~operation,~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,rest,(column + 1 : U32),lane,cells,values,
          cell_parameters,value_parameters,kind,taps,image,group,channel,top,left,row,kernel_row,next,later),
        Equal.cong(Folded<F32>,Folded<F32>,
          started => columns(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,rest,(column + 1 : U32),values,value_parameters,kind,taps,
            Cell{image,channel,top,left},row,kernel_row,started),
          lane_folded(lane,next),
          tapped(~F32,~operation,~lane_update,kind,lane_folded(lane,folded),value_sample,value_weight(value_parameters,channel,kernel_row,column)),
          tapped_lane(~operation,lane,kind,folded,cell_sample,cell_weight(cell_parameters,group,kernel_row,column),value_sample,
            value_weight(value_parameters,channel,kernel_row,column),sampled,weighted)))

# The kernel rows of a window from kernel_row on whose taps agree.
def RowsAgree(~Cells: Data,~cell_read: Cells -> Nat -> Nat -> U32 -> U32 -> Tile.Tile8,~Values: Data,~value_read: Values -> Nat -> Nat -> U32 -> U32 -> F32,
  ~CellParameters: Data,~cell_weight: CellParameters -> Nat -> U32 -> U32 -> Tile.Tile8,~ValueParameters: Data,~value_weight: ValueParameters -> Nat -> U32 -> U32 -> F32,
  count: Nat,+kernel_row: U32,+lane: Nat,+cells: Cells,+values: Values,+cell_parameters: CellParameters,+value_parameters: ValueParameters,+kind: Sampling<F32>,
  +taps: Window.Taps,+image: Nat,+group: Nat,+channel: Nat,+top: U32,+left: U32) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      KernelGemm.Both<ColumnsAgree(~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,U32.to_nat(kernel_of(taps)),0,lane,
          cells,values,cell_parameters,value_parameters,kind,taps,image,group,channel,top,left,row_of(taps,Cell{image,group,top,left},kernel_row),kernel_row),
        RowsAgree(~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,rest,(kernel_row + 1 : U32),lane,cells,values,
          cell_parameters,value_parameters,kind,taps,image,group,channel,top,left)>

def row_lane(~operation: F32 -> F32 -> F32,~Cells: Data,~cell_read: Cells -> Nat -> Nat -> U32 -> U32 -> Tile.Tile8,~Values: Data,
  ~value_read: Values -> Nat -> Nat -> U32 -> U32 -> F32,~CellParameters: Data,~cell_weight: CellParameters -> Nat -> U32 -> U32 -> Tile.Tile8,~ValueParameters: Data,
  ~value_weight: ValueParameters -> Nat -> U32 -> U32 -> F32,+kernel_row: U32,+lane: Nat,+cells: Cells,+values: Values,+cell_parameters: CellParameters,
  +value_parameters: ValueParameters,+kind: Sampling<F32>,+taps: Window.Taps,+image: Nat,+group: Nat,+channel: Nat,+top: U32,+left: U32,+accumulator: Tile.Tile8,
  agreed: ColumnsAgree(~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,U32.to_nat(kernel_of(taps)),0,lane,cells,values,
    cell_parameters,value_parameters,kind,taps,image,group,channel,top,left,row_of(taps,Cell{image,group,top,left},kernel_row),kernel_row)) ->
  {TileLaws.select_lane(lane,row_value(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Cells,~cell_read,~CellParameters,~cell_weight,cells,cell_parameters,
      lifted(kind),taps,Cell{image,group,top,left},kernel_row,accumulator)) ==
    row_value(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,values,value_parameters,kind,taps,Cell{image,channel,top,left},kernel_row,
      TileLaws.select_lane(lane,accumulator)) : F32}:
  +row = row_of(taps,Cell{image,group,top,left},kernel_row)
  +count = U32.to_nat(kernel_of(taps))
  +swept = columns(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Cells,~cell_read,~CellParameters,~cell_weight,count,0,cells,cell_parameters,lifted(kind),
    taps,Cell{image,group,top,left},row,kernel_row,Folded{accumulator,start(~Tile.Tile8,lifted(kind))})
  Equal.trans(F32,TileLaws.select_lane(lane,closed(~Tile.Tile8,~TiledMap.pairs(~operation),lifted(kind),swept)),closed(~F32,~operation,kind,lane_folded(lane,swept)),
    closed(~F32,~operation,kind,columns(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,count,0,values,value_parameters,kind,taps,
    Cell{image,channel,top,left},row,kernel_row,Folded{TileLaws.select_lane(lane,accumulator),start(~F32,kind)})),
    closed_lane(~operation,lane,kind,swept),
    Equal.cong(Folded<F32>,F32,folded => closed(~F32,~operation,kind,folded),lane_folded(lane,swept),columns(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,count,0,values,value_parameters,kind,taps,
    Cell{image,channel,top,left},row,kernel_row,Folded{TileLaws.select_lane(lane,accumulator),start(~F32,kind)}),
      Equal.trans(Folded<F32>,lane_folded(lane,swept),columns(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,count,0,values,value_parameters,kind,taps,
    Cell{image,channel,top,left},row,kernel_row,Folded{TileLaws.select_lane(lane,accumulator),TileLaws.select_lane(lane,start(~Tile.Tile8,lifted(kind)))}),columns(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,count,0,values,value_parameters,kind,taps,
    Cell{image,channel,top,left},row,kernel_row,Folded{TileLaws.select_lane(lane,accumulator),start(~F32,kind)}),
        columns_lane(~operation,~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,count,0,lane,cells,values,cell_parameters,
          value_parameters,kind,taps,image,group,channel,top,left,row,kernel_row,Folded{accumulator,start(~Tile.Tile8,lifted(kind))},agreed),
        Equal.cong(F32,Folded<F32>,start => columns(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,count,0,values,value_parameters,kind,taps,
    Cell{image,channel,top,left},row,kernel_row,Folded{TileLaws.select_lane(lane,accumulator),start}),TileLaws.select_lane(lane,start(~Tile.Tile8,lifted(kind))),start(~F32,kind),start_lane(lane,kind)))))

def rows_lane(~operation: F32 -> F32 -> F32,~Cells: Data,~cell_read: Cells -> Nat -> Nat -> U32 -> U32 -> Tile.Tile8,~Values: Data,
  ~value_read: Values -> Nat -> Nat -> U32 -> U32 -> F32,~CellParameters: Data,~cell_weight: CellParameters -> Nat -> U32 -> U32 -> Tile.Tile8,~ValueParameters: Data,
  ~value_weight: ValueParameters -> Nat -> U32 -> U32 -> F32,count: Nat,+kernel_row: U32,+lane: Nat,+cells: Cells,+values: Values,+cell_parameters: CellParameters,
  +value_parameters: ValueParameters,+kind: Sampling<F32>,+taps: Window.Taps,+image: Nat,+group: Nat,+channel: Nat,+top: U32,+left: U32,+accumulator: Tile.Tile8,
  agreed: RowsAgree(~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,count,kernel_row,lane,cells,values,cell_parameters,
    value_parameters,kind,taps,image,group,channel,top,left)) ->
  {TileLaws.select_lane(lane,rows(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Cells,~cell_read,~CellParameters,~cell_weight,count,kernel_row,cells,
      cell_parameters,lifted(kind),taps,Cell{image,group,top,left},accumulator)) ==
    rows(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,count,kernel_row,values,value_parameters,kind,taps,Cell{image,channel,top,left},
      TileLaws.select_lane(lane,accumulator)) : F32}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      KernelGemm.Both{+row,+later} = agreed
      +next = row_value(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Cells,~cell_read,~CellParameters,~cell_weight,cells,cell_parameters,lifted(kind),taps,
        Cell{image,group,top,left},kernel_row,accumulator)
      Equal.trans(F32,
        TileLaws.select_lane(lane,rows(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Cells,~cell_read,~CellParameters,~cell_weight,rest,(kernel_row + 1 : U32),
          cells,cell_parameters,lifted(kind),taps,Cell{image,group,top,left},next)),
        rows(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,rest,(kernel_row + 1 : U32),values,value_parameters,kind,taps,
          Cell{image,channel,top,left},TileLaws.select_lane(lane,next)),
        rows(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,rest,(kernel_row + 1 : U32),values,value_parameters,kind,taps,
          Cell{image,channel,top,left},
          row_value(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,values,value_parameters,kind,taps,Cell{image,channel,top,left},kernel_row,
            TileLaws.select_lane(lane,accumulator))),
        rows_lane(~operation,~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,rest,(kernel_row + 1 : U32),lane,cells,values,
          cell_parameters,value_parameters,kind,taps,image,group,channel,top,left,next,later),
        Equal.cong(F32,F32,
          started => rows(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,rest,(kernel_row + 1 : U32),values,value_parameters,kind,taps,
            Cell{image,channel,top,left},started),
          TileLaws.select_lane(lane,next),
          row_value(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,values,value_parameters,kind,taps,Cell{image,channel,top,left},kernel_row,
            TileLaws.select_lane(lane,accumulator)),
          row_lane(~operation,~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,kernel_row,lane,cells,values,cell_parameters,
            value_parameters,kind,taps,image,group,channel,top,left,accumulator,row)))

def finished_lane(~CellParameters: Data,~cell_bias: CellParameters -> Nat -> Tile.Tile8,~ValueParameters: Data,~value_bias: ValueParameters -> Nat -> F32,+lane: Nat,
  +kind: Sampling<F32>,+cell_parameters: CellParameters,+value_parameters: ValueParameters,+group: Nat,+channel: Nat,+cell: Tile.Tile8,+value: F32,
  +value_lane: {TileLaws.select_lane(lane,cell) == value : F32},
  +bias_lane: {TileLaws.select_lane(lane,cell_bias(cell_parameters,group)) == value_bias(value_parameters,channel) : F32}) ->
  {TileLaws.select_lane(lane,finished(~Tile.Tile8,~Tile.plus,~CellParameters,~cell_bias,lifted(kind),cell_parameters,group,cell)) ==
    finished(~F32,~lane_add,~ValueParameters,~value_bias,kind,value_parameters,channel,value) : F32}:
  match kind:
    case Pooling{+pad}: value_lane
    case Weighting{+zero}:
      %value_lane : {TileLaws.select_lane(lane,Tile.plus(cell,cell_bias(cell_parameters,group))) == lane_add(_,value_bias(value_parameters,channel)) : F32}
      %bias_lane : {TileLaws.select_lane(lane,Tile.plus(cell,cell_bias(cell_parameters,group))) == lane_add(TileLaws.select_lane(lane,cell),_) : F32}
      TileLaws.added_lane(cell,cell_bias(cell_parameters,group),lane)

# A lane of the value of a window over tiles is the value over lanes when its
# taps and its bias agree.
def at_lane(~operation: F32 -> F32 -> F32,~Cells: Data,~cell_read: Cells -> Nat -> Nat -> U32 -> U32 -> Tile.Tile8,~Values: Data,
  ~value_read: Values -> Nat -> Nat -> U32 -> U32 -> F32,~CellParameters: Data,~cell_weight: CellParameters -> Nat -> U32 -> U32 -> Tile.Tile8,
  ~cell_bias: CellParameters -> Nat -> Tile.Tile8,~ValueParameters: Data,~value_weight: ValueParameters -> Nat -> U32 -> U32 -> F32,
  ~value_bias: ValueParameters -> Nat -> F32,+lane: Nat,+cells: Cells,+values: Values,+cell_parameters: CellParameters,+value_parameters: ValueParameters,
  +kind: Sampling<F32>,+taps: Window.Taps,+image: Nat,+group: Nat,+channel: Nat,+top: U32,+left: U32,
  agreed: RowsAgree(~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,U32.to_nat(kernel_of(taps)),0,lane,cells,values,
    cell_parameters,value_parameters,kind,taps,image,group,channel,top,left),
  +bias_lane: {TileLaws.select_lane(lane,cell_bias(cell_parameters,group)) == value_bias(value_parameters,channel) : F32}) ->
  {TileLaws.select_lane(lane,at(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Tile.plus,~Cells,~cell_read,~CellParameters,~cell_weight,~cell_bias,cells,
      cell_parameters,lifted(kind),taps,Cell{image,group,top,left})) ==
    at(~F32,~operation,~lane_update,~lane_add,~Values,~value_read,~ValueParameters,~value_weight,~value_bias,values,value_parameters,kind,taps,
      Cell{image,channel,top,left}) : F32}:
  +count = U32.to_nat(kernel_of(taps))
  +window = rows(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Cells,~cell_read,~CellParameters,~cell_weight,count,0,cells,cell_parameters,lifted(kind),
    taps,Cell{image,group,top,left},start(~Tile.Tile8,lifted(kind)))
  finished_lane(~CellParameters,~cell_bias,~ValueParameters,~value_bias,lane,kind,cell_parameters,value_parameters,group,channel,window,rows(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,count,0,values,value_parameters,kind,taps,
    Cell{image,channel,top,left},start(~F32,kind)),
    Equal.trans(F32,TileLaws.select_lane(lane,window),rows(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,count,0,values,value_parameters,kind,taps,
    Cell{image,channel,top,left},TileLaws.select_lane(lane,start(~Tile.Tile8,lifted(kind)))),rows(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,count,0,values,value_parameters,kind,taps,
    Cell{image,channel,top,left},start(~F32,kind)),
      rows_lane(~operation,~Cells,~cell_read,~Values,~value_read,~CellParameters,~cell_weight,~ValueParameters,~value_weight,count,0,lane,cells,values,cell_parameters,
        value_parameters,kind,taps,image,group,channel,top,left,start(~Tile.Tile8,lifted(kind)),agreed),
      Equal.cong(F32,F32,start => rows(~F32,~operation,~lane_update,~Values,~value_read,~ValueParameters,~value_weight,count,0,values,value_parameters,kind,taps,
    Cell{image,channel,top,left},start),TileLaws.select_lane(lane,start(~Tile.Tile8,lifted(kind))),start(~F32,kind),start_lane(lane,kind))),
    bias_lane)
