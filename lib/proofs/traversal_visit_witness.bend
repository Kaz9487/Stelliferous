# Shape and owner preservation for traversals with consumed cursor state.
import Base
import ../traversal_array.bend as Traversal
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness

type Sample<-Reader: Type,-Element: Data,-Context: Data,-reader: Reader,-result: Reader & Traversal.Visited<Context,Element>> is Data:
  Sample{context: Context,-value: Element,equation: {result == (reader,Traversal.Visited{context,value}) : Reader & Traversal.Visited<Context,Element>}}

def store(~Reader: Type,~Element: Data,~Context: Data,+depth: Nat,-reader: Reader,+position: U32,-output: Array<Element>,
  -result: Reader & Traversal.Visited<Context,Element>,output_evidence: Witness.Witness(Element,depth,output),sample: Sample<Reader,Element,Context,reader,result>) ->
  TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_store(~Reader,~Element,~Context,position,output,result)>:
  match sample:
    case Sample{context,value,equation}:
      %Equal.sym(Reader & Traversal.Visited<Context,Element>,result,(reader,Traversal.Visited{context,value}),equation) :
        TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_store(~Reader,~Element,~Context,position,output,_)>
      TraversalWitness.StateEvidence{Array.set(Element,output,position,value),(position + 1 : U32),context,{==},Witness.write(Element,depth,output,position,value,output_evidence)}

def step(~Reader: Type,~Element: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,
  ~read_preserves: @-reader: Reader -> @context: Context -> ReaderEvidence(reader) -> Sample<Reader,Element,Context,reader,read(reader,context)>,
  +depth: Nat,-reader: Reader,-state: Traversal.State<&1,&1,Reader,Array<Element>,Context>,reader_evidence: ReaderEvidence(reader),
  evidence: TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,state>) ->
  TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_step(~Reader,~Element,~Context,~read,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,position,context,equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Reader,Array<Element>,Context>,state,Traversal.State{reader,output,position,context},equation) :
        TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_step(~Reader,~Element,~Context,~read,_)>
      store(~Reader,~Element,~Context,depth,reader,position,output,read(reader,context),output_evidence,read_preserves(reader,context,reader_evidence))

def update_sample(~Reader: Type,~Element: Data,~Context: Data,~combine: Element -> Element -> Element,
  +depth: Nat,-reader: Reader,+position: U32,-output: Array<Element>,-current: Element,-result: Reader & Traversal.Visited<Context,Element>,
  output_evidence: Witness.Witness(Element,depth,output),sample: Sample<Reader,Element,Context,reader,result>) ->
  TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_sample(~Reader,~Element,~Context,~combine,position,output,current,result)>:
  match sample:
    case Sample{context,value,equation}:
      %Equal.sym(Reader & Traversal.Visited<Context,Element>,result,(reader,Traversal.Visited{context,value}),equation) :
        TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_sample(~Reader,~Element,~Context,~combine,position,output,current,_)>
      TraversalWitness.StateEvidence{Array.set(Element,output,position,combine(current,value)),(position + 1 : U32),context,{==},
        Witness.write(Element,depth,output,position,combine(current,value),output_evidence)}

def update_current(~Reader: Type,~Element: Data,~Context: Data,~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,
  ~combine: Element -> Element -> Element,+depth: Nat,-reader: Reader,position: U32,context: Context,-output: Array<Element>,-result: Array<Element> & Element,
  output_evidence: Witness.Witness(Element,depth,output),observed: Witness.Read<Element,output,result>,sample: Sample<Reader,Element,Context,reader,read(reader,context)>) ->
  TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_current(~Reader,~Element,~Context,~read,~combine,position,context,reader,result)>:
  match observed:
    case Witness.Read{value,equation}:
      %Equal.sym(Array<Element> & Element,result,(output,value),equation) :
        TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_current(~Reader,~Element,~Context,~read,~combine,position,context,reader,_)>
      update_sample(~Reader,~Element,~Context,~combine,depth,reader,position,output,value,read(reader,context),output_evidence,sample)

def update_step(~Reader: Type,~Element: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,~combine: Element -> Element -> Element,
  ~read_preserves: @-reader: Reader -> @context: Context -> ReaderEvidence(reader) -> Sample<Reader,Element,Context,reader,read(reader,context)>,
  +depth: Nat,-reader: Reader,-state: Traversal.State<&1,&1,Reader,Array<Element>,Context>,reader_evidence: ReaderEvidence(reader),
  evidence: TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,state>) ->
  TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_step(~Reader,~Element,~Context,~read,~combine,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,+position,+context,equation,+output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Reader,Array<Element>,Context>,state,Traversal.State{reader,output,position,context},equation) :
        TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_step(~Reader,~Element,~Context,~read,~combine,_)>
      update_current(~Reader,~Element,~Context,~read,~combine,depth,reader,position,context,output,Array.get(Element,output,position),output_evidence,
        Witness.read(Element,depth,output,position,output_evidence),read_preserves(reader,context,reader_evidence))

