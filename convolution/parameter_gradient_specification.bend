# The weight and bias gradient contract. Natural coordinates and explicit
# row-major terms. The groups and the pair tree are those of the library.
# The source of the values and its lookup are parameters.
import Base
import ./specification.bend as ConvSpec
import ../lib/traversal_grouped.bend as Grouped

type Shape is Data:
  Shape{images: Nat,channels: Nat,outputs: Nat,height: Nat,width: Nat,kernel: Nat,stride: Nat,padding: Nat,rows: Nat,columns: Nat,group: Nat}

type Problem<-Source: Data> is Data:
  Problem{shape: Shape,input: Source,gradient: Source}

def sampled(~Source: Data,~at: Source -> Nat -> F32,inside: Bool,input: Source,index: Nat) -> F32:
  match inside:
    case False{}: 0.0
    case True{}: at(input,index)

# The position is u*Wo+v, increasing within every group. Padding zeros are
# multiplied by d, including when d is nonfinite; no term is skipped.
def weight_term(~Source: Data,~at: Source -> Nat -> F32,p: Problem<Source>,+image: Nat,+element: Nat,+position: Nat) -> F32:
  Problem{Shape{n,+c,+o,+h,+w,+k,+s,+pad,+rows,+columns,g},input,gradient} = p
  +t = Nat.mod(element,k)
  +r = Nat.mod(Nat.div(element,k),k)
  +channel = Nat.mod(Nat.div(element,Nat.mul(k,k)),c)
  +output = Nat.div(element,Nat.mul(c,Nat.mul(k,k)))
  +y = Nat.add(r,Nat.mul(Nat.div(position,columns),s))
  +x = Nat.add(t,Nat.mul(Nat.mod(position,columns),s))
  (at(gradient,Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),output),rows),columns),position)) *
    sampled(~Source,~at,Nat.is_le(pad,y) && Nat.is_lt(y,Nat.add(pad,h)) && Nat.is_le(pad,x) && Nat.is_lt(x,Nat.add(pad,w)),input,
      Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,c),channel),h),w),Nat.mul(Nat.sub(y,pad),w)),Nat.sub(x,pad))) : F32)

def bias_term(~Source: Data,~at: Source -> Nat -> F32,p: Problem<Source>,+image: Nat,+element: Nat,+position: Nat) -> F32:
  Problem{Shape{n,c,+o,h,w,k,s,pad,rows,columns,g},input,gradient} = p
  at(gradient,Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),element),rows),columns),position))

# Explicit term enumeration, rather than a native loop or a machine index.
def terms(~Source: Data,~term: @p: Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,remaining: Nat,+position: Nat,+p: Problem<Source>,+image: Nat,+element: Nat) -> List<&2,F32>:
  match remaining:
    case 0n: []
    case 1n+rest: term(p,image,element,position) <> terms(~Source,~term,rest,1n+position,p,image,element)

def group_values(~Source: Data,~term: @p: Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,groups: List<&2,Grouped.Range>,+p: Problem<Source>,+image: Nat,+element: Nat) -> List<&2,F32>:
  match groups:
    case []: []
    case Grouped.Range{first,count} <> rest:
      ConvSpec.sum(terms(~Source,~term,count,first,p,image,element),0.0) <> group_values(~Source,~term,rest,p,image,element)

def image_values(~Source: Data,~term: @p: Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,remaining: Nat,+image: Nat,+element: Nat,+p: Problem<Source>,+groups: List<&2,Grouped.Range>) -> List<&2,F32>:
  match remaining:
    case 0n: []
    case 1n+rest:
      Grouped.tree(group_values(~Source,~term,groups,p,image,element)) <> image_values(~Source,~term,rest,1n+image,element,p,groups)

# Tree of images of trees of groups. No final +0.0 is added.
def value(~Source: Data,~term: @p: Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,+p: Problem<Source>,element: Nat) -> F32:
  Problem{Shape{n,c,o,h,w,k,s,pad,rows,columns,g},input,gradient} = p
  Grouped.tree(image_values(~Source,~term,n,0n,element,p,Grouped.ranges(rows,columns,g)))

def weight(~Source: Data,~at: Source -> Nat -> F32,p: Problem<Source>,element: Nat) -> F32:
  value(~Source,~weight_term(~Source,~at),p,element)

def bias(~Source: Data,~at: Source -> Nat -> F32,p: Problem<Source>,element: Nat) -> F32:
  value(~Source,~bias_term(~Source,~at),p,element)
