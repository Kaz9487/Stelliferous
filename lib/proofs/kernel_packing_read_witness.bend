import Base
import ../kernel_spatial_packing.bend as Packing
import ../kernel_packing_samples.bend as PackingSamples
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness
import ../kernel_tile.bend as Tile
import ./kernel_tile_witness.bend as TileWitness

def family(~Element: Data,-input: Array<Element>) -> Data:
  Witness.ArrayWitness<Element,input>

def masked(~Element: Data,~field: Element -> F32,-input: Array<Element>,+inside: Bool,-result: Array<Element> & Element,observed: Witness.Read<Element,input,result>) ->
  TraversalWitness.Sample<Array<Element>,F32,input,PackingSamples.mask_field(~Element,~field,inside,result)>:
  Witness.Read{value,equation} = observed
  %Equal.sym(Array<Element> & Element,result,(input,value),equation) : TraversalWitness.Sample<Array<Element>,F32,input,PackingSamples.mask_field(~Element,~field,inside,_)>
  TraversalWitness.Sample{PackingSamples.masked(field(value),inside),{==}}

def masked_read(~Element: Data,~field: Element -> F32,-input: Array<Element>,+inside: Bool,-result: Array<Element> & Element,observed: Witness.Read<Element,input,result>) ->
  TraversalWitness.Sample<Array<Element>,F32,input,Packing.masked_field(~Element,inside,Packing.field_of(~Element,~field,result))>:
  Witness.Read{value,equation} = observed
  %Equal.sym(Array<Element> & Element,result,(input,value),equation) :
    TraversalWitness.Sample<Array<Element>,F32,input,Packing.masked_field(~Element,inside,Packing.field_of(~Element,~field,_))>
  TraversalWitness.Sample{PackingSamples.masked(field(value),inside),{==}}

def pixel(~Element: Data,~field: Element -> F32,-input: Array<Element>,+row: U32,+column: U32,+channel: U32,+height: U32,+width: U32,+step: U32,evidence: family(~Element,input)) ->
  TraversalWitness.Sample<Array<Element>,F32,input,Packing.pixel(~Element,~field,input,row,column,channel,height,width,step)>:
  +index = (channel + (row * width + column) * step : U32)
  masked_read(~Element,~field,input,U32.is_lt(row,height) && U32.is_lt(column,width),Array.get(Element,input,index),Witness.observed(Element,input,index,evidence))

def row_read(~Element: Data,~field: Element -> F32,-input: Array<Element>,context: Packing.PixelBlock,index: Nat,evidence: family(~Element,input)) ->
  TraversalWitness.Sample<Array<Element>,F32,input,Packing.read_block_pixel(~Element,~field,input,context,index)>:
  Packing.PixelBlock{row,column,channel,height,width,+stride,step} = context
  pixel(~Element,~field,input,row,(column + U32.from_nat(index) * stride : U32),channel,height,width,step,evidence)

def row_block(~Element: Data,~field: Element -> F32,-input: Array<Element>,+row: U32,+column: U32,+channel: U32,+height: U32,+width: U32,+stride: U32,+next_row: U32,+offset: U32,+output_width: U32,
  +kernel_column: U32,+padding: U32,+step: U32,+evidence: family(~Element,input)) ->
  TraversalWitness.Sample<Array<Element>,Tile.Tile8,input,Packing.block(~Element,~field,input,row,column,channel,height,width,stride,next_row,offset,output_width,kernel_column,padding,step)>:
  TileWitness.tail(~Array<Element>,~Packing.PixelBlock,~family(~Element),~Packing.read_block_pixel(~Element,~field),~row_read(~Element,~field),input,
    Packing.PixelBlock{row,column,channel,height,width,stride,step},
    Packing.pixel(~Element,~field,input,row,column,channel,height,width,step),evidence,pixel(~Element,~field,input,row,column,channel,height,width,step,evidence))
