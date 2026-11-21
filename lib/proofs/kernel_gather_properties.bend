# Lift a property of each copied cell to the actual gather write program.
import Base
import ./kernel_gather_blocks.bend as Blocks
import ./contiguous_copy.bend as Contiguous
import ./storage_tree_model.bend as Tree
import ./write_program_every.bend as Every
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ../kernel_gemm.bend as KernelGemm

type Copy<-Element: Data> is Data:
  Copy{input: Tree.Tree<Element>,rows: Nat,width: U32,offset: U32,spatial: U32,span: U32,source_start: U32}

def input(~Element: Data,copy: Copy<Element>) -> Tree.Tree<Element>:
  Copy{input,rows,width,offset,spatial,span,source_start} = copy
  input

def rows(~Element: Data,copy: Copy<Element>) -> Nat:
  Copy{input,rows,width,offset,spatial,span,source_start} = copy
  rows

def width(~Element: Data,copy: Copy<Element>) -> U32:
  Copy{input,rows,width,offset,spatial,span,source_start} = copy
  width

def offset(~Element: Data,copy: Copy<Element>) -> U32:
  Copy{input,rows,width,offset,spatial,span,source_start} = copy
  offset

def spatial(~Element: Data,copy: Copy<Element>) -> U32:
  Copy{input,rows,width,offset,spatial,span,source_start} = copy
  spatial

def span(~Element: Data,copy: Copy<Element>) -> U32:
  Copy{input,rows,width,offset,spatial,span,source_start} = copy
  span

def source_start(~Element: Data,copy: Copy<Element>) -> U32:
  Copy{input,rows,width,offset,spatial,span,source_start} = copy
  source_start

def source(~Element: Data,+copy: Copy<Element>,row: Nat) -> U32:
  (U32.from_nat(row) * span(~Element,copy) + source_start(~Element,copy) : U32)

def destination(~Element: Data,+copy: Copy<Element>,row: Nat) -> U32:
  (U32.from_nat(row) * spatial(~Element,copy) + offset(~Element,copy) : U32)

def CellLaw(~Element: Data,~Environment: Data,~Target: Data,~plan: @+environment: Environment -> Copy<Element>,~target: @+environment: Environment -> Target,
  ~Holds: @+target: Target -> U32 -> Element -> Data) -> Type:
  @+environment: Environment -> @+row: Nat -> @+column: Nat ->
  @+row_inside: {Nat.is_lt(row,rows(~Element,plan(environment))) == True{} : Bool} ->
  @+column_inside: {Nat.is_lt(column,U32.to_nat(width(~Element,plan(environment)))) == True{} : Bool} ->
  Holds(target(environment),(destination(~Element,plan(environment),row) + U32.from_nat(column) : U32),
    Tree.read(Element,input(~Element,plan(environment)),(source(~Element,plan(environment),row) + U32.from_nat(column) : U32)))

def columns(~Element: Data,~Environment: Data,~Target: Data,~plan: @+environment: Environment -> Copy<Element>,~target: @+environment: Environment -> Target,
  ~Holds: @+target: Target -> U32 -> Element -> Data,~cell: CellLaw(~Element,~Environment,~Target,~plan,~target,~Holds),
  +environment: Environment,count: Nat,+row: Nat,+column: Nat,+row_inside: {Nat.is_lt(row,rows(~Element,plan(environment))) == True{} : Bool},
  +room: {Nat.add(column,count) == U32.to_nat(width(~Element,plan(environment))) : Nat}) ->
  Every.Every(~Element,~Target,~Holds,target(environment),Contiguous.writes(~Element,count,input(~Element,plan(environment)),U32.from_nat(column),source(~Element,plan(environment),row),destination(~Element,plan(environment),row))):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{cell(environment,row,column,row_inside,Order.below_complement(column,rest,U32.to_nat(width(~Element,plan(environment))),room)),
        %Equal.sym(U32,(U32.from_nat(column) + 1 : U32),U32.from_nat(1n+column),Bounds.add_one(U32.from_nat(column))) :
          Every.Every(~Element,~Target,~Holds,target(environment),Contiguous.writes(~Element,rest,input(~Element,plan(environment)),_,source(~Element,plan(environment),row),destination(~Element,plan(environment),row)))
        columns(~Element,~Environment,~Target,~plan,~target,~Holds,~cell,environment,rest,row,1n+column,row_inside,
          Equal.trans(Nat,1n+Nat.add(column,rest),Nat.add(column,1n+rest),U32.to_nat(width(~Element,plan(environment))),
            Equal.sym(Nat,Nat.add(column,1n+rest),1n+Nat.add(column,rest),Addition.successor_right(column,rest)),room))}

def channels(~Element: Data,~Environment: Data,~Target: Data,~plan: @+environment: Environment -> Copy<Element>,~target: @+environment: Environment -> Target,
  ~Holds: @+target: Target -> U32 -> Element -> Data,~cell: CellLaw(~Element,~Environment,~Target,~plan,~target,~Holds),
  +environment: Environment,count: Nat,+row: Nat,+room: {Nat.add(row,count) == rows(~Element,plan(environment)) : Nat}) ->
  Every.Every(~Element,~Target,~Holds,target(environment),Blocks.writes(~Element,count,U32.from_nat(row),width(~Element,plan(environment)),offset(~Element,plan(environment)),
    spatial(~Element,plan(environment)),span(~Element,plan(environment)),source_start(~Element,plan(environment)),input(~Element,plan(environment)))):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +copy = plan(environment)
      +head = Contiguous.writes(~Element,U32.to_nat(width(~Element,copy)),input(~Element,copy),0,source(~Element,copy,row),destination(~Element,copy,row))
      +tail = Blocks.writes(~Element,rest,(U32.from_nat(row) + 1 : U32),width(~Element,copy),offset(~Element,copy),spatial(~Element,copy),span(~Element,copy),source_start(~Element,copy),input(~Element,copy))
      Every.appended(~Element,~Target,~Holds,target(environment),head,tail,
        columns(~Element,~Environment,~Target,~plan,~target,~Holds,~cell,environment,U32.to_nat(width(~Element,copy)),row,0n,Order.below_complement(row,rest,rows(~Element,copy),room),{==}),
        %Equal.sym(U32,(U32.from_nat(row) + 1 : U32),U32.from_nat(1n+row),Bounds.add_one(U32.from_nat(row))) :
          Every.Every(~Element,~Target,~Holds,target(environment),Blocks.writes(~Element,rest,_,width(~Element,copy),offset(~Element,copy),spatial(~Element,copy),span(~Element,copy),source_start(~Element,copy),input(~Element,copy)))
        channels(~Element,~Environment,~Target,~plan,~target,~Holds,~cell,environment,rest,1n+row,
          Equal.trans(Nat,1n+Nat.add(row,rest),Nat.add(row,1n+rest),rows(~Element,copy),Equal.sym(Nat,Nat.add(row,1n+rest),1n+Nat.add(row,rest),Addition.successor_right(row,rest)),room)))

def every(~Element: Data,~Environment: Data,~Target: Data,~plan: @+environment: Environment -> Copy<Element>,~target: @+environment: Environment -> Target,
  ~Holds: @+target: Target -> U32 -> Element -> Data,~cell: CellLaw(~Element,~Environment,~Target,~plan,~target,~Holds),+environment: Environment) ->
  Every.Every(~Element,~Target,~Holds,target(environment),Blocks.writes(~Element,rows(~Element,plan(environment)),0,width(~Element,plan(environment)),offset(~Element,plan(environment)),
    spatial(~Element,plan(environment)),span(~Element,plan(environment)),source_start(~Element,plan(environment)),input(~Element,plan(environment)))):
  channels(~Element,~Environment,~Target,~plan,~target,~Holds,~cell,environment,rows(~Element,plan(environment)),0n,{==})
