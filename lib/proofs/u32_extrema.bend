# Native interval endpoints agree with their unbounded minimum/maximum.
import Base
import ./u32_comparison.bend as Comparison
import ./nat_extrema.bend as Extrema


def selected(choice: Bool,+left: U32,+right: U32) ->
  {U32.to_nat(Bool.pick(U32,choice,left,right)) == Bool.pick(Nat,choice,U32.to_nat(left),U32.to_nat(right)) : Nat}:
  match choice:
    case False{}: {==}
    case True{}: {==}

def minimum(+left: U32,+right: U32) -> {U32.to_nat(U32.min(left,right)) == Nat.min(U32.to_nat(left),U32.to_nat(right)) : Nat}:
  U32{+left_word} = left
  U32{+right_word} = right
  +left = {U32{left_word} : U32}
  +right = {U32{right_word} : U32}
  Equal.trans(Nat,U32.to_nat(U32.min(left,right)),Bool.pick(Nat,U32.is_lt(left,right),U32.to_nat(left),U32.to_nat(right)),Nat.min(U32.to_nat(left),U32.to_nat(right)),
    selected(U32.is_lt(left,right),left,right),
    %Equal.sym(Bool,U32.is_lt(left,right),Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),Comparison.lt(left,right)) : {Bool.pick(Nat,_,U32.to_nat(left),U32.to_nat(right)) == Nat.min(U32.to_nat(left),U32.to_nat(right)) : Nat}
    Equal.sym(Nat,Nat.min(U32.to_nat(left),U32.to_nat(right)),Bool.pick(Nat,Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(left),U32.to_nat(right)),Extrema.minimum_choice(U32.to_nat(left),U32.to_nat(right))))

def maximum(+left: U32,+right: U32) -> {U32.to_nat(U32.max(left,right)) == Nat.max(U32.to_nat(left),U32.to_nat(right)) : Nat}:
  U32{+left_word} = left
  U32{+right_word} = right
  +left = {U32{left_word} : U32}
  +right = {U32{right_word} : U32}
  Equal.trans(Nat,U32.to_nat(U32.max(left,right)),Bool.pick(Nat,U32.is_lt(left,right),U32.to_nat(right),U32.to_nat(left)),Nat.max(U32.to_nat(left),U32.to_nat(right)),
    selected(U32.is_lt(left,right),right,left),
    %Equal.sym(Bool,U32.is_lt(left,right),Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),Comparison.lt(left,right)) : {Bool.pick(Nat,_,U32.to_nat(right),U32.to_nat(left)) == Nat.max(U32.to_nat(left),U32.to_nat(right)) : Nat}
    Equal.sym(Nat,Nat.max(U32.to_nat(left),U32.to_nat(right)),Bool.pick(Nat,Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(right),U32.to_nat(left)),Extrema.maximum_choice(U32.to_nat(left),U32.to_nat(right))))

# Base opens both arguments before it selects. These equations restore the
# selection form for arguments that are not constructor terms.
def minimum_pick(left: U32,right: U32) -> {U32.min(left,right) == Bool.pick(U32,U32.is_lt(left,right),left,right) : U32}:
  U32{left_word} = left
  U32{right_word} = right
  {==}

def pick_minimum(left: U32,right: U32) -> {Bool.pick(U32,U32.is_lt(left,right),left,right) == U32.min(left,right) : U32}:
  U32{left_word} = left
  U32{right_word} = right
  {==}

def pick_maximum(left: U32,right: U32) -> {Bool.pick(U32,U32.is_lt(left,right),right,left) == U32.max(left,right) : U32}:
  U32{left_word} = left
  U32{right_word} = right
  {==}

def minimum_left(+left: U32,+right: U32) -> {U32.is_le(U32.min(left,right),left) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(U32.min(left,right),left),Nat.is_le(U32.to_nat(U32.min(left,right)),U32.to_nat(left)),Comparison.le(U32.min(left,right),left)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.min(left,right)),Nat.min(U32.to_nat(left),U32.to_nat(right)),minimum(left,right)) : {Nat.is_le(_,U32.to_nat(left)) == True{} : Bool}
  Extrema.minimum_left(U32.to_nat(left),U32.to_nat(right))

def minimum_right(+left: U32,+right: U32) -> {U32.is_le(U32.min(left,right),right) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(U32.min(left,right),right),Nat.is_le(U32.to_nat(U32.min(left,right)),U32.to_nat(right)),Comparison.le(U32.min(left,right),right)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.min(left,right)),Nat.min(U32.to_nat(left),U32.to_nat(right)),minimum(left,right)) : {Nat.is_le(_,U32.to_nat(right)) == True{} : Bool}
  Extrema.minimum_right(U32.to_nat(left),U32.to_nat(right))

def maximum_left(+left: U32,+right: U32) -> {U32.is_le(left,U32.max(left,right)) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(left,U32.max(left,right)),Nat.is_le(U32.to_nat(left),U32.to_nat(U32.max(left,right))),Comparison.le(left,U32.max(left,right))) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.max(left,right)),Nat.max(U32.to_nat(left),U32.to_nat(right)),maximum(left,right)) : {Nat.is_le(U32.to_nat(left),_) == True{} : Bool}
  Extrema.maximum_left(U32.to_nat(left),U32.to_nat(right))

def maximum_right(+left: U32,+right: U32) -> {U32.is_le(right,U32.max(left,right)) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(right,U32.max(left,right)),Nat.is_le(U32.to_nat(right),U32.to_nat(U32.max(left,right))),Comparison.le(right,U32.max(left,right))) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.max(left,right)),Nat.max(U32.to_nat(left),U32.to_nat(right)),maximum(left,right)) : {Nat.is_le(U32.to_nat(right),_) == True{} : Bool}
  Extrema.maximum_right(U32.to_nat(left),U32.to_nat(right))

def maximum_upper(+left: U32,+right: U32,+upper: U32,lu: {U32.is_le(left,upper) == True{} : Bool},ru: {U32.is_le(right,upper) == True{} : Bool}) ->
  {U32.is_le(U32.max(left,right),upper) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(U32.max(left,right),upper),Nat.is_le(U32.to_nat(U32.max(left,right)),U32.to_nat(upper)),Comparison.le(U32.max(left,right),upper)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.max(left,right)),Nat.max(U32.to_nat(left),U32.to_nat(right)),maximum(left,right)) : {Nat.is_le(_,U32.to_nat(upper)) == True{} : Bool}
  Extrema.maximum_upper(U32.to_nat(left),U32.to_nat(right),U32.to_nat(upper),%Comparison.le(left,upper) : {_ == True{} : Bool} lu,%Comparison.le(right,upper) : {_ == True{} : Bool} ru)
