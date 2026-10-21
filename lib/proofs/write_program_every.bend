# A property that every assignment of a write list has. The property reads
# an environment passed at run time, so the template stays closed.
import Base
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./boolean_logic.bend as Boolean

def Every(~Element: Data,~Environment: Data,~Holds: @+environment: Environment -> U32 -> Element -> Data,+environment: Environment,writes: List<&2,Writes.Assignment<Element>>) -> Data:
  match writes:
    case []: Unit
    case Writes.Put{address,value} <> rest:
      Shape.Both<Holds(environment,address,value),Every(~Element,~Environment,~Holds,environment,rest)>

def appended(~Element: Data,~Environment: Data,~Holds: @+environment: Environment -> U32 -> Element -> Data,+environment: Environment,
  first: List<&2,Writes.Assignment<Element>>,+second: List<&2,Writes.Assignment<Element>>,
  first_holds: Every(~Element,~Environment,~Holds,environment,first),+second_holds: Every(~Element,~Environment,~Holds,environment,second)) ->
  Every(~Element,~Environment,~Holds,environment,List.append(&2,Writes.Assignment<Element>,first,second)):
  match first:
    case []: second_holds
    case Writes.Put{address,value} <> rest:
      Shape.Both{here,later} = first_holds
      Shape.Both{here,appended(~Element,~Environment,~Holds,environment,rest,second,later,second_holds)}

# What a readback needs of a write list, from a property of every
# assignment: the addresses are inside, the assignments at a queried address
# store the wanted value, or none is at it. The context gives the target of
# the property and what is asked.
def inside(~Element: Data,~Target: Data,~Holds: @+target: Target -> U32 -> Element -> Data,~Context: Data,~target: @+context: Context -> Target,
  ~capacity: @+context: Context -> U32,
  ~bounded: @+context: Context -> @+address: U32 -> @+value: Element -> Holds(target(context),address,value) -> {U32.is_lt(address,capacity(context)) == True{} : Bool},
  +context: Context,writes: List<&2,Writes.Assignment<Element>>,holds: Every(~Element,~Target,~Holds,target(context),writes)) -> Readback.Inside(~Element,capacity(context),writes):
  match writes:
    case []: Unit{}
    case Writes.Put{+address,+value} <> rest:
      Shape.Both{here,later} = holds
      Shape.Both{bounded(context,address,value,here),inside(~Element,~Target,~Holds,~Context,~target,~capacity,~bounded,context,rest,later)}
def matching(~Element: Data,~Target: Data,~Holds: @+target: Target -> U32 -> Element -> Data,~Context: Data,~target: @+context: Context -> Target,
  ~query: @+context: Context -> U32,~wanted: @+context: Context -> Element,
  ~matches: @+context: Context -> @+address: U32 -> @+value: Element -> Holds(target(context),address,value) -> Boolean.When(U32.is_eq(address,query(context)),{value == wanted(context) : Element}),
  +context: Context,writes: List<&2,Writes.Assignment<Element>>,holds: Every(~Element,~Target,~Holds,target(context),writes)) -> Readback.Matching(~Element,query(context),wanted(context),writes):
  match writes:
    case []: Unit{}
    case Writes.Put{+address,+value} <> rest:
      Shape.Both{here,later} = holds
      Shape.Both{matches(context,address,value,here),matching(~Element,~Target,~Holds,~Context,~target,~query,~wanted,~matches,context,rest,later)}
def excludes(~Target: Data,~Holds: @+target: Target -> U32 -> F32 -> Data,~Context: Data,~target: @+context: Context -> Target,~query: @+context: Context -> U32,
  ~different: @+context: Context -> @+address: U32 -> @+value: F32 -> Holds(target(context),address,value) -> {U32.is_eq(address,query(context)) == False{} : Bool},
  +context: Context,writes: List<&2,Writes.Assignment<F32>>,holds: Every(~F32,~Target,~Holds,target(context),writes)) -> Readback.Excludes(query(context),writes):
  match writes:
    case []: Unit{}
    case Writes.Put{+address,+value} <> rest:
      Shape.Both{here,later} = holds
      Shape.Both{different(context,address,value,here),excludes(~Target,~Holds,~Context,~target,~query,~different,context,rest,later)}
