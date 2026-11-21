# Bounds for the actual output allocation. No requested buffer size or
# successful write is supplied as a premise.
import Base
import ../specification.bend as ConvSpec
import ./convolution_guard_facts.bend as Guards

def channel_positive(+problem: ConvSpec.Problem,valid: Guards.Valid<problem>) -> {Nat.is_lt(0n,Guards.output_channel_count(problem)) == True{} : Bool}:
  Guards.Valid{input_channels,output_channels,height,width,kernel,stride,fits_height,fits_width,input_length,weight_length,bias_length} = valid
  output_channels

