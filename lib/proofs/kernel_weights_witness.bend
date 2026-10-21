import Base
import ../kernel_weights.bend as Weights
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness

# Missing rows are zero-filled without changing the source owner.
def conditional(-input: Array<F32>,valid: Bool,+index: U32,evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,F32,input,Storage.read_when(~F32,0.0,valid,input,index)>:
  match valid:
    case False{}: TraversalWitness.Sample{0.0,{==}}
    case True{}: TraversalWitness.array_sample(~F32,input,Array.get(F32,input,index),Witness.observed(F32,input,index,evidence))

def eighth(-input: Array<F32>,-a: F32,-b: F32,-c: F32,-d: F32,-e: F32,-f: F32,-g: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.eighth(a,b,c,d,e,f,g,result)>:
  TraversalWitness.Sample{h,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,h),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.eighth(a,b,c,d,e,f,g,_)>
  TraversalWitness.Sample{Tile.Tile{a,b,c,d,e,f,g,h},{==}}

def seventh_rows(+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,-input: Array<F32>,-a: F32,-b: F32,-c: F32,-d: F32,-e: F32,-f: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.seventh_rows(rows,row,index,bases,a,b,c,d,e,f,result)>:
  TraversalWitness.Sample{g,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,g),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.seventh_rows(rows,row,index,bases,a,b,c,d,e,f,_)>
  eighth(input,a,b,c,d,e,f,g,Storage.read_when(~F32,0.0,U32.is_lt((row + 7 : U32),rows),input,(Weights.base7(bases) + index : U32)),
    conditional(input,U32.is_lt((row + 7 : U32),rows),(Weights.base7(bases) + index : U32),evidence))

def sixth_rows(+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,-input: Array<F32>,-a: F32,-b: F32,-c: F32,-d: F32,-e: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.sixth_rows(rows,row,index,bases,a,b,c,d,e,result)>:
  TraversalWitness.Sample{f,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,f),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.sixth_rows(rows,row,index,bases,a,b,c,d,e,_)>
  seventh_rows(rows,row,index,bases,input,a,b,c,d,e,f,Storage.read_when(~F32,0.0,U32.is_lt((row + 6 : U32),rows),input,(Weights.base6(bases) + index : U32)),
    conditional(input,U32.is_lt((row + 6 : U32),rows),(Weights.base6(bases) + index : U32),evidence),evidence)

def fifth_rows(+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,-input: Array<F32>,-a: F32,-b: F32,-c: F32,-d: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.fifth_rows(rows,row,index,bases,a,b,c,d,result)>:
  TraversalWitness.Sample{e,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,e),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.fifth_rows(rows,row,index,bases,a,b,c,d,_)>
  sixth_rows(rows,row,index,bases,input,a,b,c,d,e,Storage.read_when(~F32,0.0,U32.is_lt((row + 5 : U32),rows),input,(Weights.base5(bases) + index : U32)),
    conditional(input,U32.is_lt((row + 5 : U32),rows),(Weights.base5(bases) + index : U32),evidence),evidence)

def fourth_rows(+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,-input: Array<F32>,-a: F32,-b: F32,-c: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.fourth_rows(rows,row,index,bases,a,b,c,result)>:
  TraversalWitness.Sample{d,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,d),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.fourth_rows(rows,row,index,bases,a,b,c,_)>
  fifth_rows(rows,row,index,bases,input,a,b,c,d,Storage.read_when(~F32,0.0,U32.is_lt((row + 4 : U32),rows),input,(Weights.base4(bases) + index : U32)),
    conditional(input,U32.is_lt((row + 4 : U32),rows),(Weights.base4(bases) + index : U32),evidence),evidence)

def third_rows(+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,-input: Array<F32>,-a: F32,-b: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.third_rows(rows,row,index,bases,a,b,result)>:
  TraversalWitness.Sample{c,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,c),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.third_rows(rows,row,index,bases,a,b,_)>
  fourth_rows(rows,row,index,bases,input,a,b,c,Storage.read_when(~F32,0.0,U32.is_lt((row + 3 : U32),rows),input,(Weights.base3(bases) + index : U32)),
    conditional(input,U32.is_lt((row + 3 : U32),rows),(Weights.base3(bases) + index : U32),evidence),evidence)

def second_rows(+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,-input: Array<F32>,-a: F32,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.second_rows(rows,row,index,bases,a,result)>:
  TraversalWitness.Sample{b,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,b),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.second_rows(rows,row,index,bases,a,_)>
  third_rows(rows,row,index,bases,input,a,b,Storage.read_when(~F32,0.0,U32.is_lt((row + 2 : U32),rows),input,(Weights.base2(bases) + index : U32)),
    conditional(input,U32.is_lt((row + 2 : U32),rows),(Weights.base2(bases) + index : U32),evidence),evidence)

def first_rows(+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,-input: Array<F32>,-result: Array<F32> & F32,
  observed: TraversalWitness.Sample<Array<F32>,F32,input,result>,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.first_rows(rows,row,index,bases,result)>:
  TraversalWitness.Sample{a,equation} = observed
  %Equal.sym(Array<F32> & F32,result,(input,a),equation) :
    TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.first_rows(rows,row,index,bases,_)>
  second_rows(rows,row,index,bases,input,a,Storage.read_when(~F32,0.0,U32.is_lt((row + 1 : U32),rows),input,(Weights.base1(bases) + index : U32)),
    conditional(input,U32.is_lt((row + 1 : U32),rows),(Weights.base1(bases) + index : U32),evidence),evidence)

def read_rows(-input: Array<F32>,+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,+evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.read_rows(input,rows,row,index,bases)>:
  first_rows(rows,row,index,bases,input,Storage.read_when(~F32,0.0,U32.is_lt(row,rows),input,(Weights.base0(bases) + index : U32)),
    conditional(input,U32.is_lt(row,rows),(Weights.base0(bases) + index : U32),evidence),evidence)

def read_mapped_rows(~Context: Data,~offset: Context -> U32 -> U32,-input: Array<F32>,context: Weights.MappedRows<Context>,index: U32,
  evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Weights.read_mapped_rows(~Context,~offset,input,context,index)>:
  Weights.MappedRows{rows,row,destination,context,bases} = context
  read_rows(input,rows,row,offset(context,index),bases,evidence)
