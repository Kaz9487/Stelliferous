# Row-major index of a four-axis kernel view and its channel exchange.
import Base
import ../tensor_view.bend as Views
import ./tensor_row_major_coordinates.bend as Coordinates
import ./nat_order.bend as Order
import ./tensor_shape_refinement.bend as ShapeLaws
import ../kernel_gemm.bend as KernelGemm

# The actual strides of a four-axis view, including inactive unit axes.
type Strides is Data:
  Strides{outer: U32,inner: U32,row: U32,column: U32}

def points(+a: U32,+b: U32,+kernel: U32,strides: Strides,c0: Nat,c1: Nat,c2: Nat,c3: Nat,+mirror: Bool) -> List<&2,Coordinates.Point>:
  Strides{outer,inner,row,column} = strides
  [Coordinates.Point{a,c0,outer,False{}},Coordinates.Point{b,c1,inner,False{}},Coordinates.Point{kernel,c2,row,mirror},Coordinates.Point{kernel,c3,column,mirror}]

def index(+b: U32,+kernel: U32,+c0: Nat,+c1: Nat,c2: Nat,c3: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(c0,U32.to_nat(b)),c1),U32.to_nat(kernel)),c2),U32.to_nat(kernel)),c3)

def position(+a: U32,+b: U32,+kernel: U32,+strides: Strides,+c0: Nat,+c1: Nat,+c2: Nat,+c3: Nat,+mirror: Bool) ->
  {Coordinates.position(points(a,b,kernel,strides,c0,c1,c2,c3,mirror),0n) == index(b,kernel,c0,c1,c2,c3) : Nat}:
  Strides{outer,inner,row,column} = strides
  {==}

def valid(+a: U32,+b: U32,+kernel: U32,strides: Strides,+c0: Nat,+c1: Nat,+c2: Nat,+c3: Nat,+mirror: Bool,
  a_inside: {Nat.is_lt(c0,U32.to_nat(a)) == True{} : Bool},b_inside: {Nat.is_lt(c1,U32.to_nat(b)) == True{} : Bool},
  row_inside: {Nat.is_lt(c2,U32.to_nat(kernel)) == True{} : Bool},column_inside: {Nat.is_lt(c3,U32.to_nat(kernel)) == True{} : Bool}) ->
  Coordinates.Valid(points(a,b,kernel,strides,c0,c1,c2,c3,mirror)):
  Strides{outer,inner,row,column} = strides
  KernelGemm.Both{a_inside,KernelGemm.Both{b_inside,KernelGemm.Both{row_inside,KernelGemm.Both{column_inside,Unit{}}}}}

def inside(+a: U32,+b: U32,+kernel: U32,+c0: Nat,+c1: Nat,+c2: Nat,+c3: Nat,
  a_inside: {Nat.is_lt(c0,U32.to_nat(a)) == True{} : Bool},b_inside: {Nat.is_lt(c1,U32.to_nat(b)) == True{} : Bool},
  row_inside: {Nat.is_lt(c2,U32.to_nat(kernel)) == True{} : Bool},column_inside: {Nat.is_lt(c3,U32.to_nat(kernel)) == True{} : Bool}) ->
  {Nat.is_lt(index(b,kernel,c0,c1,c2,c3),ShapeLaws.elements([a,b,kernel,kernel])) == True{} : Bool}:
  +strides = {Strides{0,0,0,0} : Strides}
  %position(a,b,kernel,strides,c0,c1,c2,c3,False{}) : {Nat.is_lt(_,ShapeLaws.elements([a,b,kernel,kernel])) == True{} : Bool}
  Coordinates.inside(points(a,b,kernel,strides,c0,c1,c2,c3,False{}),valid(a,b,kernel,strides,c0,c1,c2,c3,False{},a_inside,b_inside,row_inside,column_inside))

def elements_four(+a: U32,+b: U32,+kernel: U32) ->
  {ShapeLaws.elements([a,b,kernel,kernel]) == Nat.mul(U32.to_nat(a),Nat.mul(U32.to_nat(b),Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)))) : Nat}:
  Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(a),Nat.mul(U32.to_nat(b),Nat.mul(U32.to_nat(kernel),value))),Nat.mul(U32.to_nat(kernel),1n),U32.to_nat(kernel),Order.times_one(U32.to_nat(kernel)))
