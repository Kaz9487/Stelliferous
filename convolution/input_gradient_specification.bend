# Semantic contract of the gradient of the input of a convolution of any stride:
# independent of native kernels. Nat coordinates, row-major CHW ordering, and an
# explicitly ordered F32 reduction.
import Base
import ./specification.bend as ConvSpec

# The shape is the shape of the convolution. gradient is the gradient of its
# output [co, out_height, out_width], weights are its weights [co, ci, k, k].
type Problem is Data:
  Problem{shape: ConvSpec.Shape,gradient: List<&2,F32>,weights: List<&2,F32>}

def channels(p: Problem) -> Nat:
  Problem{shape,gradient,weights} = p
  ConvSpec.Shape{ci,co,height,width,kernel,stride,pad} = shape
  ci

def outputs(p: Problem) -> Nat:
  Problem{shape,gradient,weights} = p
  ConvSpec.Shape{ci,co,height,width,kernel,stride,pad} = shape
  co

def kernel_size(p: Problem) -> Nat:
  Problem{shape,gradient,weights} = p
  ConvSpec.Shape{ci,co,height,width,kernel,stride,pad} = shape
  kernel

def padding(p: Problem) -> Nat:
  Problem{shape,gradient,weights} = p
  ConvSpec.Shape{ci,co,height,width,kernel,stride,pad} = shape
  pad

def out_height(p: Problem) -> Nat:
  Problem{shape,gradient,weights} = p
  ConvSpec.Shape{ci,co,height,width,kernel,stride,pad} = shape
  ConvSpec.axis_size(height,kernel,stride,pad)

def out_width(p: Problem) -> Nat:
  Problem{shape,gradient,weights} = p
  ConvSpec.Shape{ci,co,height,width,kernel,stride,pad} = shape
  ConvSpec.axis_size(width,kernel,stride,pad)

def output_gradient(p: Problem) -> List<&2,F32>:
  Problem{shape,gradient,weights} = p
  gradient

def weight(p: Problem, oc: Nat, ic: Nat, ky: Nat, kx: Nat) -> F32:
  Problem{shape,gradient,weights} = p
  ConvSpec.Shape{+ci,co,height,width,+kernel,stride,pad} = shape
  ConvSpec.at(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(oc,ci),ic),kernel),ky),kernel),kx),weights)

def stride(p: Problem) -> Nat:
  Problem{ConvSpec.Shape{ci,co,h,w,k,s,pad},gradient,weights} = p
  s

def aligned(+p: Problem,y: Nat,x: Nat,row: Nat,column: Nat) -> Bool:
  +s = stride(p)
  +pad = padding(p)
  Nat.is_eq(Nat.mod(Nat.add(y,pad),s),Nat.mod(row,s)) &&
    Nat.is_eq(Nat.mod(Nat.add(x,pad),s),Nat.mod(column,s))

# A congruent tap outside the output gradient multiplies zero by its weight.
def sample(+p: Problem,oc: Nat,y: Nat,x: Nat,+row: Nat,+column: Nat) -> F32:
  +s = stride(p)
  +rows = out_height(p)
  +columns = out_width(p)
  +py = Nat.add(y,padding(p))
  +px = Nat.add(x,padding(p))
  ConvSpec.sample_if(Nat.is_le(row,py) && Nat.is_lt(py,Nat.add(row,Nat.mul(s,rows))) &&
    Nat.is_le(column,px) && Nat.is_lt(px,Nat.add(column,Nat.mul(s,columns))),
    Nat.add(Nat.mul(oc,Nat.mul(rows,columns)),Nat.add(Nat.mul(Nat.div(Nat.sub(py,row),s),columns),Nat.div(Nat.sub(px,column),s))),output_gradient(p))

# The products of input element (ic, y, x): oc outermost and increasing, the
# kernel row next and decreasing, the kernel column innermost and decreasing,
# keeping the taps congruent to the padded position modulo the stride.
def selected(enabled: Bool,product: ConvSpec.Product,tail: List<&2,ConvSpec.Product>) -> List<&2,ConvSpec.Product>:
  match enabled:
    case False{}: tail
    case True{}: product <> tail

def kernel_columns(n: Nat,+row: Nat,+oc: Nat,+ic: Nat,+y: Nat,+x: Nat,+p: Problem) -> List<&2,ConvSpec.Product>:
  match n:
    case 0n: Nil{}
    case 1n+ +rest:
      selected(aligned(p,y,x,row,rest),ConvSpec.Product{sample(p,oc,y,x,row,rest),weight(p,oc,ic,row,rest)},kernel_columns(rest,row,oc,ic,y,x,p))

def kernel_rows(n: Nat,+oc: Nat,+ic: Nat,+y: Nat,+x: Nat,+p: Problem) -> List<&2,ConvSpec.Product>:
  match n:
    case 0n: Nil{}
    case 1n+ +rest:
      ConvSpec.cat_products(kernel_columns(kernel_size(p),rest,oc,ic,y,x,p),kernel_rows(rest,oc,ic,y,x,p))

def output_channels(n: Nat,+oc: Nat,+ic: Nat,+y: Nat,+x: Nat,+p: Problem) -> List<&2,ConvSpec.Product>:
  match n:
    case 0n: Nil{}
    case 1n+rest:
      ConvSpec.cat_products(kernel_rows(kernel_size(p),oc,ic,y,x,p),output_channels(rest,1n+oc,ic,y,x,p))

def products(+p: Problem,ic: Nat,y: Nat,x: Nat) -> List<&2,ConvSpec.Product>:
  output_channels(outputs(p),0n,ic,y,x,p)

# The gradient of input element (ic, y, x): its products multiplied and added
# from +0.0 in that order, then +0.0 added last.
def value(+p: Problem,ic: Nat,y: Nat,x: Nat) -> F32:
  (ConvSpec.sum(ConvSpec.multiply(products(p,ic,y,x)),0.0) + 0.0 : F32)
