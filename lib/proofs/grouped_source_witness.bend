# Erased affine owner evidence for the same grouped traversal. Unlike the
# value model, this certificate proof never reflects an owner into live Data.
import Base
import ../traversal_grouped.bend as Grouped
import ../traversal_loop.bend as Loop
import ./traversal_witness.bend as Witness

type StateEvidence<-Reader: Type,-Context: Data,-reader: Reader,-state: Grouped.State<Reader,Context>> is Data:
  StateEvidence{-total: F32,position: Nat,context: Context,
    equation: {state == Grouped.State{reader,context,position,total} : Grouped.State<Reader,Context>}}

def advanced(~Reader: Type,~Context: Data,-reader: Reader,+context: Context,+position: Nat,-total: F32,-result: Reader & F32,
  evidence: Witness.Sample<Reader,F32,reader,result>) ->
  StateEvidence<Reader,Context,reader,Grouped.advanced(~Reader,~Context,context,position,total,result)>:
  Witness.Sample{value,equation} = evidence
  %Equal.sym(Reader & F32,result,(reader,value),equation) : StateEvidence<Reader,Context,reader,Grouped.advanced(~Reader,~Context,context,position,total,_)>
  StateEvidence{(total + value : F32),1n+position,context,{==}}

def step(~Reader: Type,~Context: Data,~Evidence: @-reader: Reader -> Data,~read: Reader -> Context -> Nat -> Reader & F32,
  ~preserves: @-reader: Reader -> @context: Context -> @position: Nat -> Evidence(reader) -> Witness.Sample<Reader,F32,reader,read(reader,context,position)>,
  -reader: Reader,-state: Grouped.State<Reader,Context>,source: Evidence(reader),evidence: StateEvidence<Reader,Context,reader,state>) ->
  StateEvidence<Reader,Context,reader,Grouped.step(~Reader,~Context,~read,state)>:
  StateEvidence{total,+position,+context,equation} = evidence
  %Equal.sym(Grouped.State<Reader,Context>,state,Grouped.State{reader,context,position,total},equation) : StateEvidence<Reader,Context,reader,Grouped.step(~Reader,~Context,~read,_)>
  advanced(~Reader,~Context,reader,context,position,total,read(reader,context,position),preserves(reader,context,position,source))

def loop(~Reader: Type,~Context: Data,~Evidence: @-reader: Reader -> Data,~read: Reader -> Context -> Nat -> Reader & F32,
  ~preserves: @-reader: Reader -> @context: Context -> @position: Nat -> Evidence(reader) -> Witness.Sample<Reader,F32,reader,read(reader,context,position)>,
  count: Nat,-reader: Reader,-state: Grouped.State<Reader,Context>,+source: Evidence(reader),evidence: StateEvidence<Reader,Context,reader,state>) ->
  StateEvidence<Reader,Context,reader,Loop.run(~Grouped.State<Reader,Context>,~Grouped.step(~Reader,~Context,~read),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      loop(~Reader,~Context,~Evidence,~read,~preserves,rest,reader,Grouped.step(~Reader,~Context,~read,state),source,
        step(~Reader,~Context,~Evidence,~read,~preserves,reader,state,source,evidence))

def finished(~Reader: Type,~Context: Data,-reader: Reader,-state: Grouped.State<Reader,Context>,evidence: StateEvidence<Reader,Context,reader,state>) ->
  Witness.Sample<Reader,F32,reader,Grouped.finished(~Reader,~Context,state)>:
  StateEvidence{total,position,context,equation} = evidence
  %Equal.sym(Grouped.State<Reader,Context>,state,Grouped.State{reader,context,position,total},equation) : Witness.Sample<Reader,F32,reader,Grouped.finished(~Reader,~Context,_)>
  Witness.Sample{total,{==}}

def fold(~Reader: Type,~Context: Data,~Evidence: @-reader: Reader -> Data,~read: Reader -> Context -> Nat -> Reader & F32,
  ~preserves: @-reader: Reader -> @context: Context -> @position: Nat -> Evidence(reader) -> Witness.Sample<Reader,F32,reader,read(reader,context,position)>,
  count: Nat,position: Nat,-reader: Reader,context: Context,source: Evidence(reader)) ->
  Witness.Sample<Reader,F32,reader,Grouped.fold(~Reader,~Context,~read,count,position,reader,context)>:
  finished(~Reader,~Context,reader,Loop.run(~Grouped.State<Reader,Context>,~Grouped.step(~Reader,~Context,~read),count,Grouped.State{reader,context,position,0.0}),
    loop(~Reader,~Context,~Evidence,~read,~preserves,count,reader,Grouped.State{reader,context,position,0.0},source,StateEvidence{0.0,position,context,{==}}))

def prepended(~Reader: Type,-reader: Reader,-head: F32,-result: Reader & List<&2,F32>,evidence: Witness.Sample<Reader,List<&2,F32>,reader,result>) ->
  Witness.Sample<Reader,List<&2,F32>,reader,Grouped.prepend(~Reader,head,result)>:
  Witness.Sample{tail,equation} = evidence
  %Equal.sym(Reader & List<&2,F32>,result,(reader,tail),equation) : Witness.Sample<Reader,List<&2,F32>,reader,Grouped.prepend(~Reader,head,_)>
  Witness.Sample{head <> tail,{==}}

def group_tail(~Reader: Type,~Context: Data,~read: Reader -> Context -> Nat -> Reader & F32,
  -reader: Reader,+context: Context,+rest: List<&2,Grouped.Range>,-result: Reader & F32,
  evidence: Witness.Sample<Reader,F32,reader,result>,tail_evidence: Witness.Sample<Reader,List<&2,F32>,reader,Grouped.fold_ranges(~Reader,~Context,~read,rest,reader,context)>) ->
  Witness.Sample<Reader,List<&2,F32>,reader,Grouped.grouped_tail(~Reader,result,reader => Grouped.fold_ranges(~Reader,~Context,~read,rest,reader,context))>:
  Witness.Sample{head,equation} = evidence
  %Equal.sym(Reader & F32,result,(reader,head),equation) :
    Witness.Sample<Reader,List<&2,F32>,reader,Grouped.grouped_tail(~Reader,_,reader => Grouped.fold_ranges(~Reader,~Context,~read,rest,reader,context))>
  prepended(~Reader,reader,head,Grouped.fold_ranges(~Reader,~Context,~read,rest,reader,context),tail_evidence)

def groups(~Reader: Type,~Context: Data,~Evidence: @-reader: Reader -> Data,~read: Reader -> Context -> Nat -> Reader & F32,
  ~preserves: @-reader: Reader -> @context: Context -> @position: Nat -> Evidence(reader) -> Witness.Sample<Reader,F32,reader,read(reader,context,position)>,
  ranges: List<&2,Grouped.Range>,-reader: Reader,+context: Context,+source: Evidence(reader)) ->
  Witness.Sample<Reader,List<&2,F32>,reader,Grouped.fold_ranges(~Reader,~Context,~read,ranges,reader,context)>:
  match ranges:
    case []: Witness.Sample{[],{==}}
    case Grouped.Range{first,count} <> +rest:
      group_tail(~Reader,~Context,~read,reader,context,rest,Grouped.fold(~Reader,~Context,~read,count,first,reader,context),
        fold(~Reader,~Context,~Evidence,~read,~preserves,count,first,reader,context,source),
        groups(~Reader,~Context,~Evidence,~read,~preserves,rest,reader,context,source))

def reduced(~Reader: Type,-reader: Reader,-result: Reader & List<&2,F32>,evidence: Witness.Sample<Reader,List<&2,F32>,reader,result>) ->
  Witness.Sample<Reader,F32,reader,Grouped.reduced(~Reader,result)>:
  Witness.Sample{values,equation} = evidence
  %Equal.sym(Reader & List<&2,F32>,result,(reader,values),equation) : Witness.Sample<Reader,F32,reader,Grouped.reduced(~Reader,_)>
  Witness.Sample{Grouped.tree(values),{==}}

def run(~Reader: Type,~Context: Data,~Evidence: @-reader: Reader -> Data,~read: Reader -> Context -> Nat -> Reader & F32,
  ~preserves: @-reader: Reader -> @context: Context -> @position: Nat -> Evidence(reader) -> Witness.Sample<Reader,F32,reader,read(reader,context,position)>,
  ranges: List<&2,Grouped.Range>,-reader: Reader,context: Context,source: Evidence(reader)) ->
  Witness.Sample<Reader,F32,reader,Grouped.run(~Reader,~Context,~read,ranges,reader,context)>:
  reduced(~Reader,reader,Grouped.fold_ranges(~Reader,~Context,~read,ranges,reader,context),groups(~Reader,~Context,~Evidence,~read,~preserves,ranges,reader,context,source))
