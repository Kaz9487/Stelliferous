# Equality of two sequences is preserved by concatenation. No FP32 algebra.
import Base
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ../ordered_reference.bend as OrderedReference

def cat(-A: Data,+xs: List<&2,A>,+ys: List<&2,A>,+us: List<&2,A>,+vs: List<&2,A>,left: {xs == us : List<&2,A>},right: {ys == vs : List<&2,
    A>}) -> {OrderedMatrix.cat(A,xs,ys) == OrderedMatrix.cat(A,us,vs) : List<&2,A>}:
  %left : {OrderedMatrix.cat(A,xs,ys) == OrderedMatrix.cat(A,_,vs) : List<&2,A>}
  %right : {OrderedMatrix.cat(A,xs,ys) == OrderedMatrix.cat(A,xs,_) : List<&2,A>}
  {==}

def stream_cat(+left: List<&2,F32>,+right: List<&2,F32>) -> {OrderedMatrix.cat(F32,left,right) == OrderedReference.cat(left,right) : List<&2,F32>}:
  match left:
    case Nil{}: {==}
    case +head <> +tail:
      %stream_cat(tail,right) : {head <> OrderedMatrix.cat(F32,tail,right) == head <> _ : List<&2,F32>}
      {==}
