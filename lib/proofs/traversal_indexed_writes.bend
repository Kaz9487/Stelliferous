# An immutable reader and an ordered destination cursor produce this exact
# write program, the consecutive writes of its values. Indexed maps and other
# contiguous output traversals share it.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../tensor_indexed.bend as Indexed
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_access_refinement.bend as AccessLaws
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes
import ./consecutive_writes.bend as Consecutive
import ./nat_to_u32_bounds.bend as Bounds

type Input<-Reader: Data,-Context: Data> is Data:
  Input{reader: Reader,context: Context}

def value(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,input: Input<Reader,Context>,position: U32) -> Element:
  Input{reader,context} = input
  sample(reader,context,position)

def writes(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,count: Nat,+position: U32,+context: Context) -> List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+rest: Writes.Put{position,value(context,position)} <> writes(~Element,~Context,~value,rest,(position + 1 : U32),context)

def loop(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,count: Nat,+position: U32,+reader: Reader,+context: Context,+output: Tree.Tree<Element>) ->
  {AccessLaws.finish_model(~Reader,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Context>,
    ~ArrayLaws.model_step(~Reader,~Element,~Context,~sample,~Indexed.destination(~Context)),count,Traversal.State{reader,output,position,context})) ==
    AccessLaws.ModelResult{reader,Writes.pure(Element,writes(~Element,~Input<Reader,Context>,~value(~Reader,~Element,~Context,~sample),count,position,Input{reader,context}),output)} : AccessLaws.ModelResult<Reader,Element>}:
  match count:
    case 0n: {==}
    case 1n+rest: loop(~Reader,~Element,~Context,~sample,rest,(position + 1 : U32),reader,context,Tree.write(Element,output,position,sample(reader,context,position)))

# The value of a position as a number.
def at(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+position: Nat) -> Element:
  value(context,U32.from_nat(position))

# The writes from a position are the consecutive writes of its values.
def line(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,count: Nat,+position: Nat,+context: Context) ->
  {writes(~Element,~Context,~value,count,U32.from_nat(position),context) ==
    Consecutive.writes(~Element,~Context,~at(~Element,~Context,~value),count,0n,position,context) : List<&2,Writes.Assignment<Element>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +start = U32.from_nat(position)
      +head = {Writes.Put{start,value(context,start)} : Writes.Assignment<Element>}
      %Equal.sym(U32,(start + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(start)) :
        {head <> writes(~Element,~Context,~value,rest,_,context) ==
          head <> Consecutive.writes(~Element,~Context,~at(~Element,~Context,~value),rest,0n,1n+position,context) : List<&2,Writes.Assignment<Element>>}
      Equal.cong(List<&2,Writes.Assignment<Element>>,List<&2,Writes.Assignment<Element>>,tail => head <> tail,
        writes(~Element,~Context,~value,rest,U32.from_nat(1n+position),context),Consecutive.writes(~Element,~Context,~at(~Element,~Context,~value),rest,0n,1n+position,context),
        line(~Element,~Context,~value,rest,1n+position,context))

# After the writes from zero, a position inside the count holds its value.
def read(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+count: Nat,+context: Context,+output: Tree.Tree<Element>,+depth: Nat,+index: Nat,
  +balanced: Shape.Balanced(Element,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +selected: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,writes(~Element,~Context,~value,count,0,context),output),U32.from_nat(index)) == value(context,U32.from_nat(index)) : Element}:
  %Equal.sym(List<&2,Writes.Assignment<Element>>,writes(~Element,~Context,~value,count,U32.from_nat(0n),context),
    Consecutive.writes(~Element,~Context,~at(~Element,~Context,~value),count,0n,0n,context),line(~Element,~Context,~value,count,0n,context)) :
    {Tree.read(Element,Writes.pure(Element,_,output),U32.from_nat(index)) == value(context,U32.from_nat(index)) : Element}
  Consecutive.read(~Element,~Context,~at(~Element,~Context,~value),count,0n,context,output,depth,index,balanced,depth_ok,capacity_ok,room,selected)
