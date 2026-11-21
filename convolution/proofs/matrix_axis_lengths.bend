import Base
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ../matrix_specification.bend as MatrixSpec

def length(-A: Data,xs: List<&2,A>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case h <> tail:
      1n+length(A,tail)

law cat_length:
  for -A: Data
  for +a: List<&2,A>
  for +b: List<&2,A>
  {length(A,OrderedMatrix.cat(A,a,b)) == Nat.add(length(A,a),length(A,b)) : Nat}
def cat_length(A,a,b):
  match a:
    case Nil{}:
      {==}
    case h <> +tail:
      %cat_length(A,tail,b) : {1n+length(A,OrderedMatrix.cat(A,tail,b)) == 1n+_ : Nat}
      {==}

def columns(+n: Nat,+output_column_index: Nat,+output_row_index: Nat) -> {length(MatrixSpec.Pixel,MatrixSpec.columns(n,output_column_index,output_row_index)) == n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %columns(rest,1n+output_column_index,output_row_index) : {1n+length(MatrixSpec.Pixel,MatrixSpec.columns(rest,1n+output_column_index,output_row_index)) == 1n+_ : Nat}
      {==}
def rows(+n: Nat,+output_row_index: Nat,+output_width: Nat) -> {length(MatrixSpec.Pixel,MatrixSpec.rows(n,output_row_index,output_width)) == Nat.mul(n,output_width) : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %columns(output_width,0n,output_row_index) : {length(MatrixSpec.Pixel,MatrixSpec.rows(1n+rest,output_row_index,output_width)) == Nat.add(_,Nat.mul(rest,output_width)) : Nat}
      %rows(rest,1n+output_row_index,output_width) : {length(MatrixSpec.Pixel,MatrixSpec.rows(1n+rest,output_row_index,output_width)) == Nat.add(length(MatrixSpec.Pixel,MatrixSpec.columns(output_width,
              0n,output_row_index)),_) : Nat}
      cat_length(MatrixSpec.Pixel,MatrixSpec.columns(output_width,0n,output_row_index),MatrixSpec.rows(rest,1n+output_row_index,output_width))
def channels(+n: Nat,+output_channel_index: Nat) -> {length(Nat,MatrixSpec.channels(n,output_channel_index)) == n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %channels(rest,1n+output_channel_index) : {1n+length(Nat,MatrixSpec.channels(rest,1n+output_channel_index)) == 1n+_ : Nat}
      {==}
