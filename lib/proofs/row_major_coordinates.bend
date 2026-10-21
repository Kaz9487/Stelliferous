# Every bounded flat index has bounded channel, row and column coordinates.
import Base
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order

type Coordinates<-channels: Nat,-rows: Nat,-width: Nat,-position: Nat> is Data:
  Coordinates{channel: Nat,row: Nat,column: Nat,
    channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},row_inside: {Nat.is_lt(row,rows) == True{} : Bool},column_inside: {Nat.is_lt(column,width) == True{} : Bool},
    exact: {position == RowMajor.index(channel,rows,width,row,column) : Nat}}

def locate(+channels: Nat,+rows: Nat,+width: Nat,+position: Nat,+width_positive: {Nat.is_lt(0n,width) == True{} : Bool},
  +inside: {Nat.is_lt(position,Nat.mul(channels,Nat.mul(rows,width))) == True{} : Bool}) -> Coordinates<channels,rows,width,position>:
  +plane = Nat.mul(rows,width)
  +positive = RowMajor.left_factor_positive(plane,channels,
    %RowMajor.multiply_commutative(channels,plane) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,position,Nat.mul(channels,plane),Order.zero_le(position),inside))
  +channel = Nat.div(position,plane)
  +within = Nat.mod(position,plane)
  +split = DivisionOrder.quotient_exact(position,plane,positive)
  +within_inside = DivisionOrder.remainder_small(position,plane,positive)
  +row = Nat.div(within,width)
  +column = Nat.mod(within,width)
  +pixel = DivisionOrder.quotient_exact(within,width,width_positive)
  Coordinates{channel,row,column,
    RowMajor.row_inside(channel,channels,plane,within,%split : {Nat.is_lt(_,Nat.mul(channels,plane)) == True{} : Bool} inside),
    RowMajor.row_inside(row,rows,width,column,%pixel : {Nat.is_lt(_,Nat.mul(rows,width)) == True{} : Bool} within_inside),DivisionOrder.remainder_small(within,width,width_positive),
    Equal.trans(Nat,position,Nat.add(Nat.mul(channel,plane),within),RowMajor.index(channel,rows,width,row,column),split,
      Equal.cong(Nat,Nat,index => Nat.add(Nat.mul(channel,plane),index),within,Nat.add(Nat.mul(row,width),column),pixel))}

# The pure coordinate values are shared by callers that carry bounds separately.
type Values is Data:
  Values{channel: Nat,row: Nat,column: Nat}
def values(+rows: Nat,+width: Nat,+position: Nat) -> Values:
  +within = Nat.mod(position,Nat.mul(rows,width))
  Values{Nat.div(position,Nat.mul(rows,width)),Nat.div(within,width),Nat.mod(within,width)}
def channel(value: Values) -> Nat:
  Values{channel,row,column} = value
  channel
def row(value: Values) -> Nat:
  Values{channel,row,column} = value
  row
def column(value: Values) -> Nat:
  Values{channel,row,column} = value
  column
def coordinate_values(+channels: Nat,+rows: Nat,+width: Nat,+position: Nat,coordinates: Coordinates<channels,rows,width,position>) -> Values:
  Coordinates{channel,row,column,channel_inside,row_inside,column_inside,exact} = coordinates
  Values{channel,row,column}
def located_values(+channels: Nat,+rows: Nat,+width: Nat,+position: Nat,+width_positive: {Nat.is_lt(0n,width) == True{} : Bool},+inside: {Nat.is_lt(position,Nat.mul(channels,Nat.mul(rows,width))) == True{} : Bool}) ->
  {coordinate_values(channels,rows,width,position,locate(channels,rows,width,position,width_positive,inside)) == values(rows,width,position) : Values}:
  {==}

type Bounded<-channels: Nat,-rows: Nat,-width: Nat,-value: Values> is Data:
  Bounded{channel: {Nat.is_lt(channel(value),channels) == True{} : Bool},row: {Nat.is_lt(row(value),rows) == True{} : Bool},column: {Nat.is_lt(column(value),width) == True{} : Bool}}
def channel_below(+channels: Nat,+rows: Nat,+width: Nat,+value: Values,bounds: Bounded<channels,rows,width,value>) -> {Nat.is_lt(channel(value),channels) == True{} : Bool}:
  Bounded{c,r,w} = bounds
  c
def row_below(+channels: Nat,+rows: Nat,+width: Nat,+value: Values,bounds: Bounded<channels,rows,width,value>) -> {Nat.is_lt(row(value),rows) == True{} : Bool}:
  Bounded{c,r,w} = bounds
  r
def column_below(+channels: Nat,+rows: Nat,+width: Nat,+value: Values,bounds: Bounded<channels,rows,width,value>) -> {Nat.is_lt(column(value),width) == True{} : Bool}:
  Bounded{c,r,w} = bounds
  w
def coordinate_bounds(+channels: Nat,+rows: Nat,+width: Nat,+position: Nat,+coordinates: Coordinates<channels,rows,width,position>) -> Bounded<channels,rows,width,coordinate_values(channels,rows,width,position,coordinates)>:
  Coordinates{channel,row,column,channel_inside,row_inside,column_inside,exact} = coordinates
  Bounded{channel_inside,row_inside,column_inside}
def values_inside(+channels: Nat,+rows: Nat,+width: Nat,+position: Nat,+width_positive: {Nat.is_lt(0n,width) == True{} : Bool},+inside: {Nat.is_lt(position,Nat.mul(channels,Nat.mul(rows,width))) == True{} : Bool}) -> Bounded<channels,rows,width,values(rows,width,position)>:
  %located_values(channels,rows,width,position,width_positive,inside) : Bounded<channels,rows,width,_>
  coordinate_bounds(channels,rows,width,position,locate(channels,rows,width,position,width_positive,inside))

def coordinate_exact(+channels: Nat,+rows: Nat,+width: Nat,+position: Nat,coordinates: Coordinates<channels,rows,width,position>) ->
  {position == RowMajor.index(channel(coordinate_values(channels,rows,width,position,coordinates)),rows,width,row(coordinate_values(channels,rows,width,position,coordinates)),column(coordinate_values(channels,rows,width,position,coordinates))) : Nat}:
  Coordinates{channel,row,column,channel_inside,row_inside,column_inside,exact} = coordinates
  exact
def values_exact(+channels: Nat,+rows: Nat,+width: Nat,+position: Nat,+width_positive: {Nat.is_lt(0n,width) == True{} : Bool},+inside: {Nat.is_lt(position,Nat.mul(channels,Nat.mul(rows,width))) == True{} : Bool}) ->
  {position == RowMajor.index(channel(values(rows,width,position)),rows,width,row(values(rows,width,position)),column(values(rows,width,position))) : Nat}:
  %located_values(channels,rows,width,position,width_positive,inside) : {position == RowMajor.index(channel(_),rows,width,row(_),column(_)) : Nat}
  coordinate_exact(channels,rows,width,position,locate(channels,rows,width,position,width_positive,inside))

