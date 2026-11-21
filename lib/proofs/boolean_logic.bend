# Fold a sequence of conditions and recover any member by one induction.
import Base
import ./word_encoding.bend as WordEncoding
import ../kernel_gemm.bend as KernelGemm

# Evidence is required only for an enabled branch.
def When(enabled: Bool,Property: Data) -> Data:
  match enabled:
    case False{}: Unit
    case True{}: Property

# Evidence of a property is evidence for either branch.
def when_held(-Property: Data,enabled: Bool,+held: Property) -> When(enabled,Property):
  match enabled:
    case False{}: Unit{}
    case True{}: held

# Evidence that follows from the branch being enabled.
def when_enabled(-Property: Data,enabled: Bool,held: {enabled == True{} : Bool} -> Property) -> When(enabled,Property):
  match enabled:
    case False{}: Unit{}
    case True{}: held({==})

# The evidence of a branch known to be enabled.
def required(-Property: Data,enabled: Bool,when: When(enabled,Property),truth: {enabled == True{} : Bool}) -> Property:
  match enabled:
    case False{}: Empty.absurd(Property,WordEncoding.false_true(truth))
    case True{}: when

def choose_matching(-Element: Data,+first: Bool,+rest: Bool,+actual: Element,+wanted: Element,+old: Element,
  correct: When(first,{actual == wanted : Element})) ->
  {Bool.pick(Element,rest,wanted,Bool.pick(Element,first,actual,old)) == Bool.pick(Element,first || rest,wanted,old) : Element}:
  match first rest:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: correct
    case True{} True{}: {==}

def left(+first: Bool,+second: Bool,truth: {Bool.and(first,second) == True{} : Bool}) -> {first == True{} : Bool}:
  match first:
    case False{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(truth))
    case True{}: {==}

def right(+first: Bool,+second: Bool,truth: {Bool.and(first,second) == True{} : Bool}) -> {second == True{} : Bool}:
  match first:
    case False{}: Empty.absurd({second == True{} : Bool},WordEncoding.false_true(truth))
    case True{}: truth

# Stored in reverse evaluation order so all preserves left-associated &&.
def all(conditions: List<&2,Bool>) -> Bool:
  match conditions:
    case Nil{}: True{}
    case last <> prefix: Bool.and(all(prefix),last)

# Evidence for every condition, in the same reverse order as all.
def Holds(conditions: List<&2,Bool>) -> Data:
  match conditions:
    case Nil{}: Unit
    case last <> prefix: KernelGemm.Both<{last == True{} : Bool},Holds(prefix)>

def all_hold(+conditions: List<&2,Bool>,+holds: Holds(conditions)) -> {all(conditions) == True{} : Bool}:
  match conditions:
    case Nil{}: {==}
    case +last <> +prefix:
      KernelGemm.Both{+here,+rest} = holds
      %Equal.sym(Bool,all(prefix),True{},all_hold(prefix,rest)) : {Bool.and(_,last) == True{} : Bool}
      here

# A missing condition is True, the neutral element, not an array read.
def condition_at(conditions: List<&2,Bool>,index: Nat) -> Bool:
  match conditions:
    case Nil{}: True{}
    case last <> prefix:
      match index:
        case 0n: last
        case 1n+previous: condition_at(prefix,previous)

law every_condition_holds:
  for +conditions: List<&2,Bool>
  for +index: Nat
  for truth: {all(conditions) == True{} : Bool}
  {condition_at(conditions,index) == True{} : Bool}

def every_condition_holds(conditions,index,truth):
  match conditions:
    case Nil{}: {==}
    case +last <> +prefix:
      match index:
        case 0n: right(all(prefix),last,truth)
        case 1n+previous:
          every_condition_holds(prefix,previous,left(all(prefix),last,truth))

# Disjunction identities shared by coverage proofs.
def or_true_right(+value: Bool) -> {value || True{} == True{} : Bool}:
  match value:
    case False{}: {==}
    case True{}: {==}

def or_false_left(+first: Bool,+second: Bool,proof: {first || second == False{} : Bool}) -> {first == False{} : Bool}:
  match first:
    case False{}: {==}
    case True{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},proof)))

def or_false_right(+first: Bool,+second: Bool,proof: {first || second == False{} : Bool}) -> {second == False{} : Bool}:
  match first:
    case False{}: proof
    case True{}: Empty.absurd({second == False{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},proof)))

def or_associative(+first: Bool,+second: Bool,+third: Bool) ->
  {first || (second || third) == (first || second) || third : Bool}:
  match first:
    case False{}: {==}
    case True{}: {==}

# Boolean predicates agree when each truth witness implies the other.
def equivalent(+first: Bool,+second: Bool,
  forward: {first == True{} : Bool} -> {second == True{} : Bool},
  backward: {second == True{} : Bool} -> {first == True{} : Bool}) -> {first == second : Bool}:
  match first second:
    case False{} False{}: {==}
    case True{} True{}: {==}
    case True{} False{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(forward({==})))
    case False{} True{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(backward({==})))

def and_associative(+first: Bool,+second: Bool,+third: Bool) -> {(first && second) && third == first && (second && third) : Bool}:
  match first:
    case False{}: {==}
    case True{}: {==}

def and_true(+a: Bool,+b: Bool,ha: {a == True{} : Bool},hb: {b == True{} : Bool}) -> {a && b == True{} : Bool}:
  %Equal.sym(Bool,a,True{},ha) : {_ && b == True{} : Bool}
  hb

# The parts of a true conjunction.
def and_true_left(+first: Bool,+second: Bool,both: {first && second == True{} : Bool}) -> {first == True{} : Bool}:
  match first:
    case True{}: {==}
    case False{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(both))

def and_true_right(+first: Bool,+second: Bool,both: {first && second == True{} : Bool}) -> {second == True{} : Bool}:
  match first:
    case True{}: both
    case False{}: Empty.absurd({second == True{} : Bool},WordEncoding.false_true(both))

def not_true(value: Bool,evidence: {Bool.not(value) == True{} : Bool}) -> {value == False{} : Bool}:
  match value:
    case False{}: {==}
    case True{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(evidence))
