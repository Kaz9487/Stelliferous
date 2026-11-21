import Base
import ./traversal_array.bend as Traversal

# Clip once per leaf. A channel is one contiguous copy; padding never becomes
# a store into the next channel. Both owners stay affine across the channel loop.
def gather_width_when(inside: Bool,offset: U32,output_positions: U32,span: U32) -> U32:
  match inside:
    case False{}: 0
    case True{}: U32.min(span,(output_positions - offset : U32))

def gather_width(+offset: U32,+output_positions: U32,span: U32) -> U32:
  gather_width_when(U32.is_lt(offset,output_positions),offset,output_positions,span)

def gather_channel(~Element: Data,+channel: U32,+width: U32,+offset: U32,+output_positions: U32,+span: U32,+source_start: U32,
  owners: Array<Element> & Array<Element>) -> Array<Element> & Array<Element>:
  (input_storage,output_storage) = owners
  Traversal.copy(~Element,U32.to_nat(width),(channel * span + source_start : U32),
    (channel * output_positions + offset : U32),input_storage,output_storage)

def gather_channels(~Element: Data,remaining: Nat,+channel: U32,+width: U32,+offset: U32,+output_positions: U32,+span: U32,+source_start: U32,
  owners: Array<Element> & Array<Element>) -> Array<Element> & Array<Element>:
  match remaining:
    case 0n: owners
    case 1n+rest:
      gather_channels(~Element,rest,(channel + 1 : U32),width,offset,output_positions,span,source_start,
        gather_channel(~Element,channel,width,offset,output_positions,span,source_start,owners))

def gathered_output(~Element: Data,owners: Array<Element> & Array<Element>) -> Array<Element>:
  (input_storage,output_storage) = owners
  output_storage

def gather_block(~Element: Data,channels: U32,+offset: U32,+output_positions: U32,+span: U32,input_storage: Array<Element>,output_storage: Array<Element>) -> Array<Element>:
  gathered_output(~Element,gather_channels(~Element,U32.to_nat(channels),0,gather_width(offset,output_positions,span),offset,output_positions,span,0,
    (input_storage,output_storage)))
