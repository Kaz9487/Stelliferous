# Complete FP32 output value: ordered reduction followed by one bias addition.
import Base
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ./convolution_guard_facts.bend as Guards

def matrix_dot(indices: List<&2,MatrixSpec.Reduction>,p: ConvSpec.Problem,output_channel_index: Nat,output_row_index: Nat,output_column_index: Nat,accumulator: F32) -> F32:
  OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,indices,output_channel_index,
    MatrixSpec.Position{output_row_index,output_column_index},p,accumulator)

def matrix_value(+problem: ConvSpec.Problem,+channel: Nat,row: Nat,column: Nat) -> F32:
  (matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,row,column,0.0) + ConvSpec.bias_at(problem,channel) : F32)

