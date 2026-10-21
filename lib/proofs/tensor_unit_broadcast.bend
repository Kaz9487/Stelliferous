# Keeping a singleton axis preserves rank and broadcasts back to the source.
import Base
import ../tensor_shape.bend as Shapes
import ../tensor_layout.bend as Layout
import ./tensor_band_shape_refinement.bend as BandShape
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding

def rank(+shape: List<&2,U32>,axis: Nat) -> {List.length(&2,U32,Layout.with_extent(shape,axis,1)) == List.length(&2,U32,shape) : Nat}:
  match shape axis:
    case Nil{} axis: {==}
    case Con{head,tail} 0n: {==}
    case Con{head,+tail} 1n+ +rest:
      %rank(tail,rest) : {1n+List.length(&2,U32,Layout.with_extent(tail,rest,1)) == 1n+_ : Nat}
      {==}

def own_rank(+shape: List<&2,U32>) -> {Shapes.with_rank(shape,List.length(&2,U32,shape)) == shape : List<&2,U32>}:
  %Addition.zero_right(List.length(&2,U32,shape)) : {List.append(&2,U32,List.replicate(U32,Nat.sub(_,List.length(&2,U32,shape)),1),shape) == shape : List<&2,U32>}
  %Equal.sym(Nat,Nat.sub(Nat.add(List.length(&2,U32,shape),0n),List.length(&2,U32,shape)),0n,Order.subtract_shift(List.length(&2,U32,shape),0n)) : {List.append(&2,U32,List.replicate(U32,_,1),shape) == shape : List<&2,U32>}
  {==}

def singleton(+head: U32,+same: Bool,+left_one: Bool,
  +equal: {U32.is_eq(head,1) == same : Bool},+one: {U32.is_eq(head,1) == left_one : Bool}) -> {Shapes.broadcast_extent(same,left_one,True{},head,1) == Some{head} : Maybe<&2,U32>}:
  match same left_one:
    case True{} left_one: {==}
    case False{} False{}: {==}
    case False{} True{}:
      Empty.absurd({Some{1} == Some{head} : Maybe<&2,U32>},
        WordEncoding.false_true(%equal : {_ == True{} : Bool} one))

def self_rank(+shape: List<&2,U32>) -> {Shapes.broadcast_same_rank(shape,shape) == Some{shape} : Maybe<&2,List<&2,U32>>}:
  match shape:
    case Nil{}: {==}
    case Con{+head,+tail}:
      %Equal.sym(Bool,U32.is_eq(head,head),True{},BandShape.equal_words(head,head,{==})) : {Shapes.prepend_extent(Shapes.broadcast_extent(_,U32.is_eq(head,1),U32.is_eq(head,1),head,head),Shapes.broadcast_same_rank(tail,tail)) == Some{Con{head,tail}} : Maybe<&2,List<&2,U32>>}
      %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast_same_rank(tail,tail),Some{tail},self_rank(tail)) : {Shapes.prepend_extent(Some{head},_) == Some{Con{head,tail}} : Maybe<&2,List<&2,U32>>}
      {==}

# Equal shapes do not add leading singleton axes to one another.
def self_broadcast(+shape: List<&2,U32>) -> {Shapes.broadcast(shape,shape) == Some{shape} : Maybe<&2,List<&2,U32>>}:
  %Equal.sym(List<&2,U32>,Shapes.with_rank(shape,List.length(&2,U32,shape)),shape,own_rank(shape)) : {Shapes.broadcast_same_rank(_,_) == Some{shape} : Maybe<&2,List<&2,U32>>}
  self_rank(shape)

def same_rank(+shape: List<&2,U32>,axis: Nat) -> {Shapes.broadcast_same_rank(shape,Layout.with_extent(shape,axis,1)) == Some{shape} : Maybe<&2,List<&2,U32>>}:
  match shape axis:
    case Nil{} axis: {==}
    case Con{+head,+tail} 0n:
      %Equal.sym(Bool,U32.is_eq(1,1),True{},BandShape.equal_words(1,1,{==})) : {Shapes.prepend_extent(Shapes.broadcast_extent(U32.is_eq(head,1),U32.is_eq(head,1),_,head,1),Shapes.broadcast_same_rank(tail,tail)) == Some{Con{head,tail}} : Maybe<&2,List<&2,U32>>}
      %Equal.sym(Maybe<&2,U32>,Shapes.broadcast_extent(U32.is_eq(head,1),U32.is_eq(head,1),True{},head,1),Some{head},singleton(head,U32.is_eq(head,1),U32.is_eq(head,1),{==},{==})) : {Shapes.prepend_extent(_,Shapes.broadcast_same_rank(tail,tail)) == Some{Con{head,tail}} : Maybe<&2,List<&2,U32>>}
      %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast_same_rank(tail,tail),Some{tail},self_rank(tail)) : {Shapes.prepend_extent(Some{head},_) == Some{Con{head,tail}} : Maybe<&2,List<&2,U32>>}
      {==}
    case Con{+head,+tail} 1n+ +rest:
      %Equal.sym(Bool,U32.is_eq(head,head),True{},BandShape.equal_words(head,head,{==})) : {Shapes.prepend_extent(Shapes.broadcast_extent(_,U32.is_eq(head,1),U32.is_eq(head,1),head,head),Shapes.broadcast_same_rank(tail,Layout.with_extent(tail,rest,1))) == Some{Con{head,tail}} : Maybe<&2,List<&2,U32>>}
      %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast_same_rank(tail,Layout.with_extent(tail,rest,1)),Some{tail},same_rank(tail,rest)) : {Shapes.prepend_extent(Some{head},_) == Some{Con{head,tail}} : Maybe<&2,List<&2,U32>>}
      {==}

def broadcast(+shape: List<&2,U32>,+axis: Nat) -> {Shapes.broadcast(shape,Layout.with_extent(shape,axis,1)) == Some{shape} : Maybe<&2,List<&2,U32>>}:
  %Equal.sym(Nat,List.length(&2,U32,Layout.with_extent(shape,axis,1)),List.length(&2,U32,shape),rank(shape,axis)) : {Shapes.broadcast_same_rank(Shapes.with_rank(shape,_),Shapes.with_rank(Layout.with_extent(shape,axis,1),List.length(&2,U32,shape))) == Some{shape} : Maybe<&2,List<&2,U32>>}
  %Equal.sym(List<&2,U32>,Shapes.with_rank(shape,List.length(&2,U32,shape)),shape,own_rank(shape)) : {Shapes.broadcast_same_rank(_,Shapes.with_rank(Layout.with_extent(shape,axis,1),List.length(&2,U32,shape))) == Some{shape} : Maybe<&2,List<&2,U32>>}
  %rank(shape,axis) : {Shapes.broadcast_same_rank(shape,Shapes.with_rank(Layout.with_extent(shape,axis,1),_)) == Some{shape} : Maybe<&2,List<&2,U32>>}
  %Equal.sym(List<&2,U32>,Shapes.with_rank(Layout.with_extent(shape,axis,1),List.length(&2,U32,Layout.with_extent(shape,axis,1))),Layout.with_extent(shape,axis,1),own_rank(Layout.with_extent(shape,axis,1))) : {Shapes.broadcast_same_rank(shape,_) == Some{shape} : Maybe<&2,List<&2,U32>>}
  same_rank(shape,axis)
