# The place of the first largest element of a sequence, in increasing order.
# A NaN counts as larger than any number, and the first NaN stays.
import Base

type Largest is Data:
  Largest{value: F32,place: U32,next: U32}

def start() -> Largest:
  Largest{F32.div(F32.neg(1.0),0.0),0,0}

def larger(+element: F32,+value: F32) -> Bool:
  F32.is_lt(value,element) || (F32.is_eq(value,value) && Bool.not(F32.is_eq(element,element)))

def chosen(larger: Bool,+value: F32,+place: U32,+element: F32,+next: U32) -> Largest:
  match larger:
    case True{}: Largest{element,next,(next + 1 : U32)}
    case False{}: Largest{value,place,(next + 1 : U32)}

def step(state: Largest,element: F32) -> Largest:
  Largest{+value,+place,+next} = state
  +element = element
  chosen(larger(element,value),value,place,element,next)

def place(state: Largest) -> F32:
  Largest{value,place,next} = state
  U32.to_f32(place)
