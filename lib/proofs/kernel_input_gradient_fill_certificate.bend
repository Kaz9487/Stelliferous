# The fill reads its source and writes only its tiles: the source comes back
# unchanged and the tiles keep their certificate.
import Base
import ../kernel_input_gradient_fill.bend as Fill
import ../kernel_spatial_packing.bend as Spatial
import ../kernel_tile.bend as Tile
import ../traversal_loop.bend as Loop
import ../traversal_array.bend as Traversal
import ./storage_witness.bend as Witness
import ./storage_certificate.bend as Certificate
import ./storage_tree_model.bend as Tree
import ./traversal_witness.bend as TraversalWitness
import ./traversal_certificate.bend as TraversalCertificate
import ./kernel_packing_read_witness.bend as PackingReadWitness

# A fill state over the source input: some tiles, with their witness.
type Filled<-input: Array<Tile.Tile8>,-depth: Nat,-state: Fill.Fill> is Data:
  Filled{-tiles: Array<Tile.Tile8>,index: U32,row: U32,column: U32,first: U32,base: U32,groups: U32,height: U32,width: U32,shifts: Nat,channel: U32,
    equation: {state == Fill.Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} : Fill.Fill},
    balanced: Witness.Witness(Tile.Tile8,depth,tiles)}

def stored(+depth: Nat,-input: Array<Tile.Tile8>,-tiles: Array<Tile.Tile8>,+index: U32,+row: U32,+column: U32,+first: U32,+base: U32,+groups: U32,+height: U32,+width: U32,+shifts: Nat,+channel: U32,
  -result: Array<Tile.Tile8> & Tile.Tile8,balanced: Witness.Witness(Tile.Tile8,depth,tiles),sample: TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,input,result>) ->
  Filled<input,depth,Fill.filled(tiles,index,row,column,first,base,groups,height,width,shifts,channel,result)>:
  TraversalWitness.Sample{value,equation} = sample
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,result,(input,value),equation) : Filled<input,depth,Fill.filled(tiles,index,row,column,first,base,groups,height,width,shifts,channel,_)>
  Filled{Array.set(Tile.Tile8,tiles,index,value),(index + 1 : U32),row,(column + 1 : U32),first,base,groups,height,width,shifts,channel,{==},
    Witness.write(Tile.Tile8,depth,tiles,index,value,balanced)}

def shift(~field: Tile.Tile8 -> F32,+depth: Nat,-input: Array<Tile.Tile8>,-state: Fill.Fill,+input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,evidence: Filled<input,depth,state>) ->
  Filled<input,depth,Fill.fill_shift(~field,state)>:
  Filled{tiles,+index,+row,+column,+first,+base,+groups,+height,+width,+shifts,+channel,equation,balanced} = evidence
  %Equal.sym(Fill.Fill,state,Fill.Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel},equation) : Filled<input,depth,Fill.fill_shift(~field,_)>
  stored(depth,input,tiles,index,row,column,first,base,groups,height,width,shifts,channel,
    Spatial.block(~Tile.Tile8,~field,input,row,column,(base + channel / 8 : U32),height,width,1,0,0,0,0,0,groups),balanced,
    PackingReadWitness.row_block(~Tile.Tile8,~field,input,row,column,(base + channel / 8 : U32),height,width,1,0,0,0,0,0,groups,input_evidence))

def shift_loop(~field: Tile.Tile8 -> F32,+depth: Nat,count: Nat,-input: Array<Tile.Tile8>,-state: Fill.Fill,+input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,evidence: Filled<input,depth,state>) ->
  Filled<input,depth,Loop.run(~Fill.Fill,~Fill.fill_shift(~field),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: shift_loop(~field,depth,rest,input,Fill.fill_shift(~field,state),input_evidence,shift(~field,depth,input,state,input_evidence,evidence))

def next(-depth: Nat,-input: Array<Tile.Tile8>,-state: Fill.Fill,evidence: Filled<input,depth,state>) -> Filled<input,depth,Fill.next_channel(state)>:
  Filled{tiles,index,row,column,+first,+base,+groups,height,width,shifts,channel,equation,balanced} = evidence
  %Equal.sym(Fill.Fill,state,Fill.Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel},equation) : Filled<input,depth,Fill.next_channel(_)>
  Filled{tiles,index,row,first,first,base,groups,height,width,shifts,(channel + 1 : U32),{==},balanced}

def lane(~field: Tile.Tile8 -> F32,+depth: Nat,-input: Array<Tile.Tile8>,-state: Fill.Fill,+input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,evidence: Filled<input,depth,state>) ->
  Filled<input,depth,Fill.fill_lane(~field,state)>:
  Filled{tiles,index,row,column,first,base,groups,height,width,+shifts,+channel,equation,balanced} = evidence
  -start = {Fill.Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} : Fill.Fill}
  %Equal.sym(Fill.Fill,state,start,equation) : Filled<input,depth,Fill.fill_lane(~field,_)>
  next(depth,input,Loop.run(~Fill.Fill,~Fill.fill_shift(~field),shifts,start),
    shift_loop(~field,depth,shifts,input,start,input_evidence,Filled{tiles,index,row,column,first,base,groups,height,width,shifts,channel,{==},balanced}))

def selected(half: Bool,low: Bool,first: Bool,+depth: Nat,-input: Array<Tile.Tile8>,-state: Fill.Fill,+input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,evidence: Filled<input,depth,state>) ->
  Filled<input,depth,Fill.fill_selected(half,low,first,state)>:
  match half low first:
    case True{} True{} True{}: lane(~Tile.lane0,depth,input,state,input_evidence,evidence)
    case True{} True{} False{}: lane(~Tile.lane1,depth,input,state,input_evidence,evidence)
    case True{} False{} True{}: lane(~Tile.lane2,depth,input,state,input_evidence,evidence)
    case True{} False{} False{}: lane(~Tile.lane3,depth,input,state,input_evidence,evidence)
    case False{} True{} True{}: lane(~Tile.lane4,depth,input,state,input_evidence,evidence)
    case False{} True{} False{}: lane(~Tile.lane5,depth,input,state,input_evidence,evidence)
    case False{} False{} True{}: lane(~Tile.lane6,depth,input,state,input_evidence,evidence)
    case False{} False{} False{}: lane(~Tile.lane7,depth,input,state,input_evidence,evidence)

def channel(+depth: Nat,-input: Array<Tile.Tile8>,-state: Fill.Fill,+input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,evidence: Filled<input,depth,state>) ->
  Filled<input,depth,Fill.fill_channel(state)>:
  Filled{tiles,index,row,column,first,base,groups,height,width,shifts,+channel,equation,balanced} = evidence
  -start = {Fill.Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} : Fill.Fill}
  %Equal.sym(Fill.Fill,state,start,equation) : Filled<input,depth,Fill.fill_channel(_)>
  selected(U32.is_lt((channel % 8 : U32),4),U32.is_lt((channel % 8 % 4 : U32),2),U32.is_zero((channel % 8 % 2 : U32)),depth,input,start,input_evidence,
    Filled{tiles,index,row,column,first,base,groups,height,width,shifts,channel,{==},balanced})

def channel_loop(+depth: Nat,count: Nat,-input: Array<Tile.Tile8>,-state: Fill.Fill,+input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,evidence: Filled<input,depth,state>) ->
  Filled<input,depth,Loop.run(~Fill.Fill,~Fill.fill_channel,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: channel_loop(depth,rest,input,Fill.fill_channel(state),input_evidence,channel(depth,input,state,input_evidence,evidence))

def finished(-depth: Nat,-input: Array<Tile.Tile8>,-state: Fill.Fill,evidence: Filled<input,depth,state>) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile8,input,depth,Fill.fill_finished(state)>:
  Filled{tiles,index,row,column,first,base,groups,height,width,shifts,channel,equation,balanced} = evidence
  %Equal.sym(Fill.Fill,state,Fill.Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel},equation) :
    TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile8,input,depth,Fill.fill_finished(_)>
  TraversalWitness.ResultEvidence{tiles,{==},balanced}

def group(+depth: Nat,-input: Array<Tile.Tile8>,+index: U32,+context: Fill.Filling,-tiles: Array<Tile.Tile8>,+input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,
  balanced: Witness.Witness(Tile.Tile8,depth,tiles)) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile8,input,depth,Fill.fill_group(index,context,(input,tiles))>:
  Fill.Filling{+shifts,+outputs,+column_low,+kernel,+width,+chunk_groups,+low,+base,+groups,+first,+source_rows,+group,+span} = context
  +column = ((index % span + group) * 8 + column_low - kernel : U32)
  -start = {Fill.Fill{input,tiles,((index / span * chunk_groups + index % span) * (outputs * shifts + 1) : U32),(index / span + low - kernel - first : U32),column,column,
    base,groups,source_rows,width,U32.to_nat(shifts),0} : Fill.Fill}
  finished(depth,input,Loop.run(~Fill.Fill,~Fill.fill_channel,U32.to_nat(outputs),start),
    channel_loop(depth,U32.to_nat(outputs),input,start,input_evidence,
      Filled{tiles,((index / span * chunk_groups + index % span) * (outputs * shifts + 1) : U32),(index / span + low - kernel - first : U32),column,column,
        base,groups,source_rows,width,U32.to_nat(shifts),0,{==},balanced}))

def group_loop(+depth: Nat,count: Nat,+index: U32,+context: Fill.Filling,-input: Array<Tile.Tile8>,-owners: Array<Tile.Tile8> & Array<Tile.Tile8>,
  +input_evidence: Witness.ArrayWitness<Tile.Tile8,input>,evidence: TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile8,input,depth,owners>) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile8,input,depth,
    Traversal.indexed_fold(~(Array<Tile.Tile8> & Array<Tile.Tile8>),~Fill.Filling,~Fill.fill_group,count,index,context,owners)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      TraversalWitness.ResultEvidence{tiles,equation,balanced} = evidence
      %Equal.sym(Array<Tile.Tile8> & Array<Tile.Tile8>,owners,(input,tiles),equation) :
        TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile8,input,depth,
          Traversal.indexed_fold(~(Array<Tile.Tile8> & Array<Tile.Tile8>),~Fill.Filling,~Fill.fill_group,1n+rest,index,context,_)>
      group_loop(depth,rest,(index + 1 : U32),context,input,Fill.fill_group(index,context,(input,tiles)),input_evidence,
        group(depth,input,index,context,tiles,input_evidence,balanced))

def preserves(+depth: Nat,call: Fill.Call,-input: Array<Tile.Tile8>,-tiles: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<Tile.Tile8,input>,
  equation: {tiles == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,depth,Tree.reflect(Tile.Tile8,tiles))) : Array<Tile.Tile8>}) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile8,input,depth,Fill.run(call,input,tiles)>:
  Fill.Call{count,+filling} = call
  group_loop(depth,count,0,filling,input,(input,tiles),Certificate.witness(Tile.Tile8,input,input_certificate),
    TraversalWitness.ResultEvidence{tiles,{==},Certificate.witness_at(Tile.Tile8,depth,tiles,equation)})

def certificates(call: Fill.Call,-input: Array<Tile.Tile8>,-tiles: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<Tile.Tile8,input>,
  tile_certificate: Certificate.Certificate<Tile.Tile8,tiles>) ->
  TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile8,Certificate.family(~Tile.Tile8),Fill.run(call,input,tiles)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Tile.Tile8>,~Tile.Tile8,~Fill.Call,~Certificate.family(~Tile.Tile8),~Fill.run,~preserves,
    call,input,tiles,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},tile_certificate)
