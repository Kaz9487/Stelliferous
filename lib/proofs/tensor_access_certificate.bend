# Discharge the reader premises for the actual tensor traversal kernels.
# Public operations use only the flat certificates returned at their boundary.
import Base
import ../tensor_access.bend as Access
import ../tensor_view.bend as Views
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../tensor_cursor.bend as Cursor
import ./tensor_cursor_witness.bend as CursorWitness
import ./traversal_visit_witness.bend as VisitWitness
import ./storage_certificate.bend as Certificate
import ./storage_tree_model.bend as Tree
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness
import ./traversal_certificate.bend as TraversalCertificate

def read_index(~Element: Data,-input: Array<Element>,context: Unit,index: U32,evidence: Witness.ArrayWitness<Element,input>) ->
  TraversalWitness.Sample<Array<Element>,Element,input,Traversal.read_index(~Element,input,context,index)>:
  TraversalWitness.array_sample(~Element,input,Array.get(Element,input,index),Witness.observed(Element,input,index,evidence))

type CopyCall is Data:
  CopyCall{fast: Bool,count: Nat,source: Views.View,destination: U32}

def copy_operation(~Element: Data,call: CopyCall,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  CopyCall{fast,count,source,destination} = call
  Access.copy_run(~Element,fast,count,source,destination,input,output)

def combine_operation(~Element: Data,~operation: Element -> Element -> Element,call: CopyCall,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  CopyCall{fast,count,source,destination} = call
  Access.combine_run(~Element,~operation,fast,count,source,destination,input,output)

# A strided operation keeps its owners through each part of a run: the stride
# steps, the state handed to the cursor, and the cursor step that ends the run.
# The lemmas are generic in the two element steps and in the evidence that
# each of them keeps.
def last_of_run(~Element: Data,+depth: Nat,extent: Nat,stride: U32,rewind: U32,rest: Cursor.Depth1(),-input: Array<Element>,
  -state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,
  evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Stride,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Access.last_of_run(~Element,extent,stride,rewind,rest,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,position,Cursor.Stride{address,step},equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,state,Traversal.State{input,output,position,Cursor.Stride{address,step}},equation) :
        TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Access.last_of_run(~Element,extent,stride,rewind,rest,_)>
      TraversalWitness.StateEvidence{output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n,stride,rewind},rest},address},{==},output_evidence}

# A step keeps the evidence of its state.
def Keeps(~Element: Data,~Context: Data,~step: Traversal.State<&1,&1,Array<Element>,Array<Element>,Context> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Context>) -> Type:
  @+depth: Nat -> @-input: Array<Element> -> @-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Context> -> Witness.ArrayWitness<Element,input> ->
    TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,state> -> TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,step(state)>

def loop(~Element: Data,~Context: Data,~step: Traversal.State<&1,&1,Array<Element>,Array<Element>,Context> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Context>,
  ~keeps: Keeps(~Element,~Context,~step),
  +depth: Nat,count: Nat,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Context>,+input_evidence: Witness.ArrayWitness<Element,input>,
  evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Context>,~step,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      loop(~Element,~Context,~step,~keeps,depth,rest,input,step(state),input_evidence,keeps(depth,input,state,input_evidence,evidence))

def run_axis(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,remaining: Nat,extent: Nat,+stride: U32,rewind: U32,rest: Cursor.Depth1(),+address: U32,
  -input: Array<Element>,-output: Array<Element>,position: U32,+input_evidence: Witness.ArrayWitness<Element,input>,
  output_evidence: Witness.Witness(Element,depth,output)) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Access.run_axis(~Element,~inside,~last,remaining,extent,stride,rewind,rest,address,input,output,position)>:
  match remaining:
    case 0n:
      last_keeps(depth,input,Traversal.State{input,output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,0n,stride,rewind},rest},address}},input_evidence,
        TraversalWitness.StateEvidence{output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,0n,stride,rewind},rest},address},{==},output_evidence})
    case 1n+count:
      last_keeps(depth,input,
        Access.last_of_run(~Element,extent,stride,rewind,rest,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~inside,count,Traversal.State{input,output,position,Cursor.Stride{address,stride}})),
        input_evidence,
        last_of_run(~Element,depth,extent,stride,rewind,rest,input,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~inside,count,Traversal.State{input,output,position,Cursor.Stride{address,stride}}),
          loop(~Element,~Cursor.Stride,~inside,~inside_keeps,depth,count,input,Traversal.State{input,output,position,Cursor.Stride{address,stride}},input_evidence,
            TraversalWitness.StateEvidence{output,position,Cursor.Stride{address,stride},{==},output_evidence})))

def run_step(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  +input_evidence: Witness.ArrayWitness<Element,input>,evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Access.run_step(~Element,~inside,~last,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,remaining,stride,rewind},rest},address},equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,state,
        Traversal.State{input,output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,remaining,stride,rewind},rest},address}},equation) :
        TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Access.run_step(~Element,~inside,~last,_)>
      run_axis(~Element,~inside,~last,~inside_keeps,~last_keeps,depth,remaining,extent,stride,rewind,rest,address,input,output,position,input_evidence,output_evidence)

def run_loop(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,count: Nat,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  +input_evidence: Witness.ArrayWitness<Element,input>,evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~Access.run_step(~Element,~inside,~last),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      run_loop(~Element,~inside,~last,~inside_keeps,~last_keeps,depth,rest,input,Access.run_step(~Element,~inside,~last,state),input_evidence,
        run_step(~Element,~inside,~last,~inside_keeps,~last_keeps,depth,input,state,input_evidence,evidence))

def last_loop(~Element: Data,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,count: Nat,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  +input_evidence: Witness.ArrayWitness<Element,input>,evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~last,count,state)>:
  loop(~Element,~Cursor.Cursor,~last,~last_keeps,depth,count,input,state,input_evidence,evidence)

def stepped(~Element: Data,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,count: Nat,cursor: Cursor.Cursor,destination: U32,-input: Array<Element>,-output: Array<Element>,
  input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,
    Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~last,count,
      Traversal.State{input,output,destination,cursor}))>:
  TraversalCertificate.completed(~Array<Element>,~Element,~Cursor.Cursor,~TraversalWitness.array_evidence(~Element),
    ~Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~last),~last_loop(~Element,~last,~last_keeps),
    count,destination,cursor,input,output,input_evidence,depth,equation)

def guarded(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,full_run: Bool,whole_runs: Bool,runs: Nat,count: Nat,cursor: Cursor.Cursor,destination: U32,
  -input: Array<Element>,-output: Array<Element>,input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Access.run_guarded(~Element,~inside,~last,full_run,whole_runs,runs,count,cursor,destination,input,output)>:
  match full_run whole_runs:
    case True{} True{}:
      TraversalCertificate.completed(~Array<Element>,~Element,~Cursor.Cursor,~TraversalWitness.array_evidence(~Element),
        ~Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~Access.run_step(~Element,~inside,~last)),~run_loop(~Element,~inside,~last,~inside_keeps,~last_keeps),
        runs,destination,cursor,input,output,input_evidence,depth,equation)
    case True{} False{}: stepped(~Element,~last,~last_keeps,depth,count,cursor,destination,input,output,input_evidence,equation)
    case False{} whole_runs: stepped(~Element,~last,~last_keeps,depth,count,cursor,destination,input,output,input_evidence,equation)

def remaining_run(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,remaining: Nat,+extent: Nat,stride: U32,rewind: U32,rest: Cursor.Depth1(),address: U32,+count: Nat,destination: U32,
  -input: Array<Element>,-output: Array<Element>,input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Access.run_remaining(~Element,~inside,~last,remaining,extent,stride,rewind,rest,address,count,destination,input,output)>:
  match remaining:
    case 0n:
      stepped(~Element,~last,~last_keeps,depth,count,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,0n,stride,rewind},rest},address},destination,input,output,input_evidence,equation)
    case 1n+ +inner_count:
      guarded(~Element,~inside,~last,~inside_keeps,~last_keeps,depth,Nat.is_eq(extent,1n+inner_count),Nat.is_eq(Nat.mul(Nat.div(count,1n+inner_count),1n+inner_count),count),Nat.div(count,1n+inner_count),count,
        Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n+inner_count,stride,rewind},rest},address},destination,input,output,input_evidence,equation)

def strided(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,count: Nat,cursor: Cursor.Cursor,destination: U32,-input: Array<Element>,-output: Array<Element>,
  input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Access.run_strided(~Element,~inside,~last,count,cursor,destination,input,output)>:
  Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,remaining,stride,rewind},rest},address} = cursor
  remaining_run(~Element,~inside,~last,~inside_keeps,~last_keeps,depth,remaining,extent,stride,rewind,rest,address,count,destination,input,output,input_evidence,equation)

def copied_strided(~Element: Data,+depth: Nat,count: Nat,cursor: Cursor.Cursor,destination: U32,-input: Array<Element>,-output: Array<Element>,
  input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Access.copy_strided(~Element,count,cursor,destination,input,output)>:
  strided(~Element,~Traversal.visit_step(~Array<Element>,~Element,~Cursor.Stride,~Cursor.read_stride(~Element)),
    ~Traversal.visit_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element)),
    ~VisitWitness.step(~Array<Element>,~Element,~Cursor.Stride,~TraversalWitness.array_evidence(~Element),~Cursor.read_stride(~Element),~CursorWitness.read_stride(~Element)),
    ~VisitWitness.step(~Array<Element>,~Element,~Cursor.Cursor,~TraversalWitness.array_evidence(~Element),~Cursor.read(~Element),~CursorWitness.read(~Element)),
    depth,count,cursor,destination,input,output,input_evidence,equation)

def updated_strided(~Element: Data,~operation: Element -> Element -> Element,+depth: Nat,count: Nat,cursor: Cursor.Cursor,destination: U32,
  -input: Array<Element>,-output: Array<Element>,input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Access.update_strided(~Element,~operation,count,cursor,destination,input,output)>:
  strided(~Element,~Traversal.visit_update_step(~Array<Element>,~Element,~Cursor.Stride,~Cursor.read_stride(~Element),~operation),
    ~Traversal.visit_update_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation),
    ~VisitWitness.update_step(~Array<Element>,~Element,~Cursor.Stride,~TraversalWitness.array_evidence(~Element),~Cursor.read_stride(~Element),~operation,~CursorWitness.read_stride(~Element)),
    ~VisitWitness.update_step(~Array<Element>,~Element,~Cursor.Cursor,~TraversalWitness.array_evidence(~Element),~Cursor.read(~Element),~operation,~CursorWitness.read(~Element)),
    depth,count,cursor,destination,input,output,input_evidence,equation)

def copied(~Element: Data,+depth: Nat,call: CopyCall,-input: Array<Element>,-output: Array<Element>,input_certificate: Certificate.Certificate<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,copy_operation(~Element,call,input,output)>:
  match call:
    case CopyCall{True{},count,Views.View{axes,offset},destination}:
      TraversalCertificate.completed(~Array<Element>,~Element,~Traversal.Offsets,~TraversalWitness.array_evidence(~Element),
        ~Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.source_offset(~Element),~Traversal.destination_offset),
        ~TraversalWitness.run(~Array<Element>,~Element,~Traversal.Offsets,~TraversalWitness.array_evidence(~Element),~Traversal.source_offset(~Element),~Traversal.destination_offset,~TraversalWitness.offset_read(~Element)),
        count,0,Traversal.Offsets{offset,destination},input,output,Certificate.witness(Element,input,input_certificate),depth,equation)
    case CopyCall{False{},count,source,destination}:
      copied_strided(~Element,depth,count,Cursor.prepare(source),destination,input,output,Certificate.witness(Element,input,input_certificate),equation)

def combined(~Element: Data,~operation: Element -> Element -> Element,+depth: Nat,call: CopyCall,-input: Array<Element>,-output: Array<Element>,
  input_certificate: Certificate.Certificate<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,combine_operation(~Element,~operation,call,input,output)>:
  match call:
    case CopyCall{True{},count,source,destination}:
      TraversalCertificate.completed(~Array<Element>,~Element,~Unit,~TraversalWitness.array_evidence(~Element),
        ~Traversal.update(~Array<Element>,~Element,~Element,~Unit,~Traversal.read_index(~Element),~operation,~Traversal.index),
        ~TraversalWitness.update(~Array<Element>,~Element,~Element,~Unit,~TraversalWitness.array_evidence(~Element),~Traversal.read_index(~Element),~operation,~Traversal.index,~read_index(~Element)),
        count,0,Unit{},input,output,Certificate.witness(Element,input,input_certificate),depth,equation)
    case CopyCall{False{},count,source,destination}:
      updated_strided(~Element,~operation,depth,count,Cursor.prepare(source),destination,input,output,Certificate.witness(Element,input,input_certificate),equation)

def copy(~Element: Data,fast: Bool,count: Nat,source: Views.View,destination: U32,-input: Array<Element>,-output: Array<Element>,
  input_certificate: Certificate.Certificate<Element,input>,output_certificate: Certificate.Certificate<Element,output>) ->
  TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),Access.copy_run(~Element,fast,count,source,destination,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Element>,~Element,~CopyCall,~Certificate.family(~Element),~copy_operation(~Element),~copied(~Element),
    CopyCall{fast,count,source,destination},input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)

def combine(~Element: Data,~operation: Element -> Element -> Element,fast: Bool,count: Nat,source: Views.View,destination: U32,
  -input: Array<Element>,-output: Array<Element>,input_certificate: Certificate.Certificate<Element,input>,output_certificate: Certificate.Certificate<Element,output>) ->
  TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),Access.combine_run(~Element,~operation,fast,count,source,destination,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Element>,~Element,~CopyCall,~Certificate.family(~Element),~combine_operation(~Element,~operation),~combined(~Element,~operation),
    CopyCall{fast,count,source,destination},input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)

# The reader of a ternary update is a pair of arrays; every read hands both
# back unchanged.
type Paired<-Element: Data,-reader: (Array<Element> & Array<Element>)> is Data:
  Paired{-mask: Array<Element>,-no: Array<Element>,equation: {reader == (mask,no) : (Array<Element> & Array<Element>)},
    mask_witness: Witness.ArrayWitness<Element,mask>,no_witness: Witness.ArrayWitness<Element,no>}

def paired(~Element: Data,-reader: (Array<Element> & Array<Element>)) -> Data:
  Paired<Element,reader>

def choice_sample(~Element: Data,-mask: Array<Element>,-no: Array<Element>,-mask_result: Array<Element> & Element,-no_result: Array<Element> & Element,
  mask_read: Witness.Read<Element,mask,mask_result>,no_read: Witness.Read<Element,no,no_result>) ->
  TraversalWitness.Sample<(Array<Element> & Array<Element>),Access.Choice<Element>,(mask,no),Access.choice_joined(~Element,mask_result,no_result)>:
  match mask_read:
    case Witness.Read{mask_value,mask_equation}:
      match no_read:
        case Witness.Read{no_value,no_equation}:
          %Equal.sym(Array<Element> & Element,mask_result,(mask,mask_value),mask_equation) :
            TraversalWitness.Sample<(Array<Element> & Array<Element>),Access.Choice<Element>,(mask,no),Access.choice_joined(~Element,_,no_result)>
          %Equal.sym(Array<Element> & Element,no_result,(no,no_value),no_equation) :
            TraversalWitness.Sample<(Array<Element> & Array<Element>),Access.Choice<Element>,(mask,no),Access.choice_joined(~Element,(mask,mask_value),_)>
          TraversalWitness.Sample{Access.Choice{mask_value,no_value},{==}}

def choice_at(~Element: Data,-reader: (Array<Element> & Array<Element>),context: Unit,+index: U32,evidence: Paired<Element,reader>) ->
  TraversalWitness.Sample<(Array<Element> & Array<Element>),Access.Choice<Element>,reader,Access.choice_read(~Element,reader,context,index)>:
  Paired{mask,no,equation,mask_witness,no_witness} = evidence
  %Equal.sym((Array<Element> & Array<Element>),reader,(mask,no),equation) :
    TraversalWitness.Sample<(Array<Element> & Array<Element>),Access.Choice<Element>,_,Access.choice_read(~Element,_,context,index)>
  choice_sample(~Element,mask,no,Array.get(Element,mask,index),Array.get(Element,no,index),
    Witness.observed(Element,mask,index,mask_witness),Witness.observed(Element,no,index,no_witness))

def choice_read(~Element: Data,-reader: (Array<Element> & Array<Element>),context: Unit,index: U32,evidence: Paired<Element,reader>) ->
  TraversalWitness.Sample<(Array<Element> & Array<Element>),Access.Choice<Element>,reader,Access.choice_read(~Element,reader,context,index)>:
  choice_at(~Element,reader,context,index,evidence)

def chosen_equation(~Element: Data,~choose: Element -> Element -> Element -> Element,count: Nat,+depth: Nat,-mask: Array<Element>,-no: Array<Element>,-output: Array<Element>,
  mask_certificate: Certificate.Certificate<Element,mask>,no_certificate: Certificate.Certificate<Element,no>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  {TraversalWitness.second(~(Array<Element> & Array<Element>),~Array<Element>,Access.choose_run(~Element,~choose,count,mask,no,output)) ==
    Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,
      TraversalWitness.second(~(Array<Element> & Array<Element>),~Array<Element>,Access.choose_run(~Element,~choose,count,mask,no,output))))) : Array<Element>}:
  %Certificate.restored(Element,depth,TraversalWitness.second(~(Array<Element> & Array<Element>),~Array<Element>,Access.choose_run(~Element,~choose,count,mask,no,output)),
    TraversalWitness.output_witness(~(Array<Element> & Array<Element>),~Element,(mask,no),depth,Access.choose_run(~Element,~choose,count,mask,no,output),
      TraversalCertificate.completed(~(Array<Element> & Array<Element>),~Element,~Unit,~paired(~Element),
        ~Traversal.update(~(Array<Element> & Array<Element>),~Element,~Access.Choice<Element>,~Unit,~Access.choice_read(~Element),~Access.chosen(~Element,~choose),~Traversal.index),
        ~TraversalWitness.update(~(Array<Element> & Array<Element>),~Element,~Access.Choice<Element>,~Unit,~paired(~Element),~Access.choice_read(~Element),~Access.chosen(~Element,~choose),
          ~Traversal.index,~choice_read(~Element)),
        count,0,Unit{},(mask,no),output,Paired{mask,no,{==},Certificate.witness(Element,mask,mask_certificate),Certificate.witness(Element,no,no_certificate)},depth,equation))) :
    {_ == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,
      TraversalWitness.second(~(Array<Element> & Array<Element>),~Array<Element>,Access.choose_run(~Element,~choose,count,mask,no,output))))) : Array<Element>}
  {==}

# The output of a ternary update keeps the certificate of the array it
# updates.
def choose(~Element: Data,~choose: Element -> Element -> Element -> Element,count: Nat,-mask: Array<Element>,-no: Array<Element>,-output: Array<Element>,
  mask_certificate: Certificate.Certificate<Element,mask>,no_certificate: Certificate.Certificate<Element,no>,output_certificate: Certificate.Certificate<Element,output>) ->
  Certificate.Certificate<Element,TraversalWitness.second(~(Array<Element> & Array<Element>),~Array<Element>,Access.choose_run(~Element,~choose,count,mask,no,output))>:
  Certificate.Certificate{+depth,+equation} = output_certificate
  Certificate.Certificate{depth,chosen_equation(~Element,~choose,count,depth,mask,no,output,mask_certificate,no_certificate,equation)}

def range_stated(~Element: Data,~Accumulator: Data,-input: Array<Element>,-state: Traversal.State<&1,&2,Array<Element>,Accumulator,Traversal.Strided>,
  evidence: TraversalWitness.FoldEvidence<Array<Element>,Accumulator,Traversal.Strided,input,state>,certificate: Certificate.Certificate<Element,input>) ->
  Certificate.Certificate<Element,TraversalWitness.first(~Array<Element>,~Accumulator,Traversal.finish_fold(~Array<Element>,~Accumulator,~Traversal.Strided,state))>:
  %TraversalWitness.folded_reader(~Array<Element>,~Accumulator,~Traversal.Strided,input,state,evidence) : Certificate.Certificate<Element,_>
  certificate

# The input a fold of a range hands back keeps its certificate. The fold
# evidence stays inside the rewrite, so native code does not fold again.
def fold_range(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,+count: Nat,+position: U32,+step: U32,-initial: Accumulator,-input: Array<Element>,
  +certificate: Certificate.Certificate<Element,input>) ->
  Certificate.Certificate<Element,TraversalWitness.first(~Array<Element>,~Accumulator,
    Traversal.fold(~Array<Element>,~Element,~Accumulator,~Traversal.Strided,~Traversal.strided(~Element),~operation,count,0,input,initial,Traversal.Strided{position,step}))>:
  %TraversalWitness.folded_reader(~Array<Element>,~Accumulator,~Traversal.Strided,input,
    Loop.run(~Traversal.State<&1,&2,Array<Element>,Accumulator,Traversal.Strided>,
      ~Traversal.fold_step(~Array<Element>,~Element,~Accumulator,~Traversal.Strided,~Traversal.strided(~Element),~operation),count,
      Traversal.State{input,initial,0,Traversal.Strided{position,step}}),
    TraversalWitness.fold_run(~Array<Element>,~Element,~Accumulator,~Traversal.Strided,~TraversalWitness.array_evidence(~Element),~Traversal.strided(~Element),~operation,
      ~TraversalWitness.strided_read(~Element),count,input,Traversal.State{input,initial,0,Traversal.Strided{position,step}},Certificate.witness(Element,input,certificate),
      TraversalWitness.FoldEvidence{initial,0,Traversal.Strided{position,step},{==}})) :
    Certificate.Certificate<Element,_>
  certificate
