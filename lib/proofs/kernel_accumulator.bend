import Base
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ../math_sequence.bend as Sequence
import ../math_fp32.bend as Fp32
import ./fp32_arithmetic_laws.bend as ArithmeticLaws
import ./indexed_sequence_laws.bend as SequenceLaws

# These projections are proof observations, not runtime representations.
type Row is Data:
  R0{}
  R1{}
  R2{}
  R3{}

def row(~Element: Data,index: Row,values: Rows.Rows<Element>) -> Element:
  match index values:
    case R0{} Rows.Rows{a,b,c,d}: a
    case R1{} Rows.Rows{a,b,c,d}: b
    case R2{} Rows.Rows{a,b,c,d}: c
    case R3{} Rows.Rows{a,b,c,d}: d

def update_vector(+accumulator: Tile.Tile8,+input: Tile.Tile8,+weight: F32) ->
  {Tile.view(Tile.update(accumulator,input,weight)) == Fp32.zip_mac(8n,Tile.view(accumulator),Tile.view(input),weight) : Sequence.Values(F32,8n)}:
  Tile.Tile{a,b,c,d,e,f,g,h} = accumulator
  Tile.Tile{x0,x1,x2,x3,x4,x5,x6,x7} = input
  {==}

def update_at(+index: Sequence.Index(8n),+accumulator: Tile.Tile8,+input: Tile.Tile8,+weight: F32) ->
  {Tile.element_at(index,Tile.update(accumulator,input,weight)) ==
    Fp32.multiply_add(Tile.element_at(index,accumulator),Tile.element_at(index,input),weight) : F32}:
  %Equal.sym(Sequence.Values(F32,8n),Tile.view(Tile.update(accumulator,input,weight)),Fp32.zip_mac(8n,Tile.view(accumulator),Tile.view(input),weight),
    update_vector(accumulator,input,weight)) :
    {Sequence.get(F32,8n,index,_) == Fp32.multiply_add(Tile.element_at(index,accumulator),Tile.element_at(index,input),weight) : F32}
  ArithmeticLaws.zip_mac_at_index(8n,index,Tile.view(accumulator),Tile.view(input),weight)

def initial_at(+index: Sequence.Index(8n),+value: F32) -> {Tile.element_at(index,Tile.filled(value)) == value : F32}:
  SequenceLaws.get_replicated(F32,8n,index,value)

def row_update(+index: Row,+accumulators: Rows.Rows<Tile.Tile8>,+input: Tile.Tile8,+weights: Rows.Rows<F32>) ->
  {row(~Tile.Tile8,index,Rows.map(~Tile.Tile8,~Tile.Tile8,~F32,~Tile.update,accumulators,input,weights)) ==
    Tile.update(row(~Tile.Tile8,index,accumulators),input,row(~F32,index,weights)) : Tile.Tile8}:
  match index accumulators weights:
    case R0{} Rows.Rows{a,b,c,d} Rows.Rows{u,v,w,x}: {==}
    case R1{} Rows.Rows{a,b,c,d} Rows.Rows{u,v,w,x}: {==}
    case R2{} Rows.Rows{a,b,c,d} Rows.Rows{u,v,w,x}: {==}
    case R3{} Rows.Rows{a,b,c,d} Rows.Rows{u,v,w,x}: {==}

def row_filled(~Element: Data,index: Row,+value: Element) -> {row(~Element,index,Rows.filled(~Element,value)) == value : Element}:
  match index:
    case R0{}: {==}
    case R1{}: {==}
    case R2{}: {==}
    case R3{}: {==}

def row_of(index: Nat) -> Row:
  match index:
    case 0n: R0{}
    case 1n+a:
      match a:
        case 0n: R1{}
        case 1n+b:
          match b:
            case 0n: R2{}
            case 1n+c: R3{}

def select_row(~Element: Data,index: Nat,values: Rows.Rows<Element>) -> Element:
  match index:
    case 0n: Rows.first(~Element,values)
    case 1n+a:
      match a:
        case 0n: Rows.second(~Element,values)
        case 1n+b:
          match b:
            case 0n: Rows.third(~Element,values)
            case 1n+c: Rows.fourth(~Element,values)

def selection_matches(~Element: Data,+index: Nat,+values: Rows.Rows<Element>) ->
  {select_row(~Element,index,values) == row(~Element,row_of(index),values) : Element}:
  match index:
    case 0n:
      Rows.Rows{a,b,c,d} = values
      {==}
    case 1n+ +a:
      match a:
        case 0n:
          Rows.Rows{a,b,c,d} = values
          {==}
        case 1n+ +b:
          match b:
            case 0n:
              Rows.Rows{a,b,c,d} = values
              {==}
            case 1n+ +c:
              Rows.Rows{a,b,c,d} = values
              {==}
