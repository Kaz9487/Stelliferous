# Inductive whole-write-program invariant. A covered index receives its
# specified F32 value; unrelated writes preserve it. No FP32 algebra is used.
import Base
import ./scalar_write_refinement.bend as ScalarWriteRefinement
import ../../lib/proofs/nat_division.bend as Division
import ../specification.bend as ConvSpec
import ../../lib/proofs/word_encoding.bend as WordEncoding

def set_length(+i: Nat,+v: F32,+xs: List<&2,F32>) -> {ConvSpec.length(ScalarWriteRefinement.set(i,v,xs)) == ConvSpec.length(xs) : Nat}:
  match i xs:
    case 0n Nil{}:
      {==}
    case 0n x <> tail:
      {==}
    case 1n+i Nil{}:
      {==}
    case 1n+ +i +x <> +tail:
      %set_length(i,v,tail) : {1n+ConvSpec.length(ScalarWriteRefinement.set(i,v,tail)) == 1n+_ : Nat}
      {==}

def set_room(+query: Nat,+i: Nat,+v: F32,+xs: List<&2,F32>,room: {Nat.is_lt(query,ConvSpec.length(xs)) == True{} : Bool}) -> {Nat.is_lt(query,
    ConvSpec.length(ScalarWriteRefinement.set(i,v,xs))) == True{} : Bool}:
  %Equal.sym(Nat,ConvSpec.length(ScalarWriteRefinement.set(i,v,xs)),ConvSpec.length(xs),set_length(i,v,xs)) : {Nat.is_lt(query,_) == True{} : Bool}
  room

def at_set(+query: Nat,+i: Nat,+v: F32,+xs: List<&2,F32>,room: {Nat.is_lt(query,ConvSpec.length(xs)) == True{} : Bool}) -> {ConvSpec.at(query,
    ScalarWriteRefinement.set(i,v,xs)) == Bool.pick(F32,Nat.is_eq(query,i),v,ConvSpec.at(query,xs)) : F32}:
  match query i xs:
    case query i Nil{}:
      Empty.absurd({ConvSpec.at(query,ScalarWriteRefinement.set(i,v,[])) == Bool.pick(F32,Nat.is_eq(query,i),v,ConvSpec.at(query,[])) : F32},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(query,0n),True{},Equal.sym(Bool,Nat.is_lt(query,0n),False{},
              Division.lt_zero(query)),room)))
    case 0n 0n x <> tail:
      {==}
    case 0n 1n+i x <> tail:
      {==}
    case 1n+query 0n x <> tail:
      {==}
    case 1n+query 1n+i x <> tail:
      at_set(query,i,v,tail,room)
