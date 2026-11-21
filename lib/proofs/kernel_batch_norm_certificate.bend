# The loops of batch normalization read their sources and write their
# targets: every read hands its arrays back, so each array keeps its depth.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_batch_norm.bend as Kernel
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../math_sum.bend as Sum
import ../tensor_operations.bend as Operations
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness
import ./traversal_certificate.bend as TraversalCertificate

def Read(-input: Array<Tile.Tile8>,-result: Array<Tile.Tile8> & Tile.Tile8) -> Data:
  TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,input,result>

def observed(-input: Array<Tile.Tile8>,+index: U32,evidence: Witness.ArrayWitness<Tile.Tile8,input>) -> Read(input,Array.get(Tile.Tile8,input,index)):
  TraversalWitness.array_sample(~Tile.Tile8,input,Array.get(Tile.Tile8,input,index),Witness.observed(Tile.Tile8,input,index,evidence))

# ---- Array reads ----

def normal_sample(-input: Array<Tile.Tile8>,+group: Kernel.Normal,-result: Array<Tile.Tile8> & Tile.Tile8,evidence: Read(input,result)) ->
  Read(input,Kernel.normal_cell(group,result)):
  match evidence:
    case TraversalWitness.Sample{value,equation}:
      %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,result,(input,value),equation) : Read(input,Kernel.normal_cell(group,_))
      TraversalWitness.Sample{Kernel.normalized(group,value),{==}}

def read_normalized(-input: Array<Tile.Tile8>,context: Kernel.Grouped<Kernel.Normal>,position: U32,evidence: Witness.ArrayWitness<Tile.Tile8,input>) ->
  Read(input,Kernel.read_normalized(input,context,position)):
  Kernel.Grouped{+group,+start,+step,count} = context
  normal_sample(input,group,Array.get(Tile.Tile8,input,(start + position * step : U32)),observed(input,(start + position * step : U32),evidence))

def output_sample(-input: Array<Tile.Tile8>,+group: Kernel.Normal,-result: Array<Tile.Tile8> & Tile.Tile8,evidence: Read(input,result)) ->
  Read(input,Kernel.output_cell(group,result)):
  match evidence:
    case TraversalWitness.Sample{value,equation}:
      %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,result,(input,value),equation) : Read(input,Kernel.output_cell(group,_))
      TraversalWitness.Sample{Kernel.output(group,value),{==}}

def read_output(-input: Array<Tile.Tile8>,context: Kernel.Grouped<Kernel.Normal>,position: U32,evidence: Witness.ArrayWitness<Tile.Tile8,input>) ->
  Read(input,Kernel.read_output(input,context,position)):
  Kernel.Grouped{+group,+start,+step,count} = context
  output_sample(input,group,Array.get(Tile.Tile8,input,(start + position * step : U32)),observed(input,(start + position * step : U32),evidence))

def back_sample(-reader: Array<Tile.Tile8>,+group: Kernel.Back,-result: Array<Tile.Tile8> & Tile.Tile8,evidence: Read(reader,result)) ->
  TraversalWitness.Sample<Array<Tile.Tile8>,Kernel.Sampled,reader,Kernel.back_sample(group,result)>:
  Kernel.Back{+beta,+gamma,+scale,+total} = group
  match evidence:
    case TraversalWitness.Sample{value,equation}:
      %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,result,(reader,value),equation) :
        TraversalWitness.Sample<Array<Tile.Tile8>,Kernel.Sampled,reader,Kernel.back_sample(Kernel.Back{beta,gamma,scale,total},_)>
      TraversalWitness.Sample{Kernel.Sampled{Tile.times(value,gamma),Kernel.Back{beta,gamma,scale,total}},{==}}

def back_read(-reader: Array<Tile.Tile8>,context: Kernel.Grouped<Kernel.Back>,position: U32,evidence: Witness.ArrayWitness<Tile.Tile8,reader>) ->
  TraversalWitness.Sample<Array<Tile.Tile8>,Kernel.Sampled,reader,Kernel.back_read(reader,context,position)>:
  Kernel.Grouped{+group,+start,+step,count} = context
  back_sample(reader,group,Array.get(Tile.Tile8,reader,(start + position * step : U32)),observed(reader,(start + position * step : U32),evidence))

def centered_sample(-input: Array<Tile.Tile8>,+mean: Tile.Tile8,-result: Array<Tile.Tile8> & Tile.Tile8,evidence: Read(input,result)) ->
  Read(input,Kernel.centered_cell(mean,result)):
  match evidence:
    case TraversalWitness.Sample{value,equation}:
      %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,result,(input,value),equation) : Read(input,Kernel.centered_cell(mean,_))
      TraversalWitness.Sample{Kernel.squared(Tile.minus(value,mean)),{==}}

def read_centered(-input: Array<Tile.Tile8>,context: Traversal.Spaced<Tile.Tile8>,position: U32,evidence: Witness.ArrayWitness<Tile.Tile8,input>) ->
  Read(input,Kernel.read_centered(input,context,position)):
  Traversal.Spaced{+base,+stride,+mean} = context
  centered_sample(input,mean,Array.get(Tile.Tile8,input,(base + position * stride : U32)),observed(input,(base + position * stride : U32),evidence))

# ---- Two arrays ----

type LeavesEvidence<-leaves: Kernel.Leaves()> is Data:
  LeavesEvidence{-left: Array<Tile.Tile8>,-right: Array<Tile.Tile8>,equation: {leaves == (left,right) : Kernel.Leaves()},
    left_evidence: Witness.ArrayWitness<Tile.Tile8,left>,right_evidence: Witness.ArrayWitness<Tile.Tile8,right>}

def leaves_evidence(-leaves: Kernel.Leaves()) -> Data:
  LeavesEvidence<leaves>

def product_right(-left: Array<Tile.Tile8>,-right: Array<Tile.Tile8>,-a: Tile.Tile8,+index: U32,read: Witness.Read<Tile.Tile8,right,Array.get(Tile.Tile8,right,index)>) ->
  TraversalWitness.Sample<Kernel.Leaves(),Tile.Tile8,(left,right),Kernel.product_right(left,a,Array.get(Tile.Tile8,right,index))>:
  Witness.Read{value,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,right,index),(right,value),equation) :
    TraversalWitness.Sample<Kernel.Leaves(),Tile.Tile8,(left,right),Kernel.product_right(left,a,_)>
  TraversalWitness.Sample{Tile.times(a,value),{==}}

def product_left(-left: Array<Tile.Tile8>,-right: Array<Tile.Tile8>,+index: U32,left_read: Witness.Read<Tile.Tile8,left,Array.get(Tile.Tile8,left,index)>,
  right_evidence: Witness.ArrayWitness<Tile.Tile8,right>) ->
  TraversalWitness.Sample<Kernel.Leaves(),Tile.Tile8,(left,right),Kernel.product_left(right,index,Array.get(Tile.Tile8,left,index))>:
  Witness.Read{value,equation} = left_read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,left,index),(left,value),equation) :
    TraversalWitness.Sample<Kernel.Leaves(),Tile.Tile8,(left,right),Kernel.product_left(right,index,_)>
  product_right(left,right,value,index,Witness.observed(Tile.Tile8,right,index,right_evidence))

def read_product(-leaves: Kernel.Leaves(),context: Traversal.Spaced<Unit>,position: U32,evidence: LeavesEvidence<leaves>) ->
  TraversalWitness.Sample<Kernel.Leaves(),Tile.Tile8,leaves,Kernel.read_product(leaves,context,position)>:
  Traversal.Spaced{+base,+stride,Unit{}} = context
  LeavesEvidence{left,right,equation,left_evidence,right_evidence} = evidence
  %Equal.sym(Kernel.Leaves(),leaves,(left,right),equation) :
    TraversalWitness.Sample<Kernel.Leaves(),Tile.Tile8,_,Kernel.read_product(_,Traversal.Spaced{base,stride,Unit{}},position)>
  +index = (base + position * stride : U32)
  product_left(left,right,index,Witness.observed(Tile.Tile8,left,index,left_evidence),right_evidence)

# Two arrays with certificates of their own.
type LeavesOwned<-leaves: Kernel.Leaves()> is Data:
  LeavesOwned{-left: Array<Tile.Tile8>,-right: Array<Tile.Tile8>,equation: {leaves == (left,right) : Kernel.Leaves()},
    left_certificate: Certificate.Certificate<Tile.Tile8,left>,right_certificate: Certificate.Certificate<Tile.Tile8,right>}

def observe_leaves(-leaves: Kernel.Leaves(),owned: LeavesOwned<leaves>) -> LeavesEvidence<leaves>:
  LeavesOwned{left,right,equation,left_certificate,right_certificate} = owned
  LeavesEvidence{left,right,equation,Certificate.witness(Tile.Tile8,left,left_certificate),Certificate.witness(Tile.Tile8,right,right_certificate)}

# The certificates of the two arrays.
type LeavesCertificates<-leaves: Kernel.Leaves()> is Data:
  LeavesCertificates{left: Certificate.Certificate<Tile.Tile8,TraversalWitness.first(~Array<Tile.Tile8>,~Array<Tile.Tile8>,leaves)>,
    right: Certificate.Certificate<Tile.Tile8,TraversalWitness.second(~Array<Tile.Tile8>,~Array<Tile.Tile8>,leaves)>}

def owned_certificates(-leaves: Kernel.Leaves(),owned: LeavesOwned<leaves>) -> LeavesCertificates<leaves>:
  LeavesOwned{left,right,equation,left_certificate,right_certificate} = owned
  %Equal.sym(Kernel.Leaves(),leaves,(left,right),equation) : LeavesCertificates<_>
  LeavesCertificates{left_certificate,right_certificate}

def leaves_owned(-leaves: Kernel.Leaves()) -> Data:
  LeavesOwned<leaves>

def owned_leaves(-left: Array<Tile.Tile8>,-right: Array<Tile.Tile8>,+left_depth: Nat,
  +left_equation: {left == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,left_depth,Tree.reflect(Tile.Tile8,left))) : Array<Tile.Tile8>},+right_depth: Nat,
  +right_equation: {right == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,right_depth,Tree.reflect(Tile.Tile8,right))) : Array<Tile.Tile8>}) -> LeavesOwned<(left,right)>:
  LeavesOwned{left,right,{==},Certificate.Certificate{left_depth,left_equation},Certificate.Certificate{right_depth,right_equation}}

# ---- Loops ----

# A run of products from two arrays into the slots hands both arrays back and
# updates the slots in place.
def product_updates(+run: Traversal.Steps<Traversal.Spaced<Unit>>,-left: Array<Tile.Tile8>,-right: Array<Tile.Tile8>,-slots: Array<Sum.TileParts>,
  left_certificate: Certificate.Certificate<Tile.Tile8,left>,right_certificate: Certificate.Certificate<Tile.Tile8,right>,
  slots_certificate: Certificate.Certificate<Sum.TileParts,slots>) ->
  TraversalCertificate.Certificates<Kernel.Leaves(),Sum.TileParts,leaves_owned,
    Traversal.updated(~Kernel.Leaves(),~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~Kernel.read_product,~Sum.tile_slot_step,~Operations.slot_of(~Unit),run,(left,right),slots)>:
  Certificate.Certificate{+left_depth,+left_equation} = left_certificate
  Certificate.Certificate{+right_depth,+right_equation} = right_certificate
  TraversalCertificate.updated_certificates(~Kernel.Leaves(),~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~leaves_evidence,~leaves_owned,~Kernel.read_product,
    ~Sum.tile_slot_step,~Operations.slot_of(~Unit),~read_product,~observe_leaves,run,(left,right),slots,
    owned_leaves(left,right,left_depth,left_equation,right_depth,right_equation),owned_leaves(left,right,left_depth,left_equation,right_depth,right_equation),
    owned_leaves(left,right,left_depth,left_equation,right_depth,right_equation),slots_certificate)

def array_preserves(~Group: Data,~read: Array<Tile.Tile8> -> Kernel.Grouped<Group> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,
  ~read_preserves: @-input: Array<Tile.Tile8> -> @context: Kernel.Grouped<Group> -> @index: U32 -> Witness.ArrayWitness<Tile.Tile8,input> -> Read(input,read(input,context,index)),
  +depth: Nat,context: Kernel.Grouped<Group>,-input: Array<Tile.Tile8>,-output: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<Tile.Tile8,input>,
  equation: {output == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,depth,Tree.reflect(Tile.Tile8,output))) : Array<Tile.Tile8>}) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile8,input,depth,Kernel.mapped(~Array<Tile.Tile8>,~Group,~read,context,input,output)>:
  +context = context
  TraversalCertificate.completed(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~TraversalWitness.array_evidence(~Tile.Tile8),
    ~Traversal.run(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~read,~Kernel.place(~Group)),
    ~TraversalWitness.run(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~TraversalWitness.array_evidence(~Tile.Tile8),~read,~Kernel.place(~Group),~read_preserves),
    U32.to_nat(Kernel.count_of(~Group,context)),0,context,input,output,Certificate.witness(Tile.Tile8,input,input_certificate),depth,equation)

# The input handed back and the target of a loop that reads one array.
def array_certificates(~Group: Data,~read: Array<Tile.Tile8> -> Kernel.Grouped<Group> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,
  ~read_preserves: @-input: Array<Tile.Tile8> -> @context: Kernel.Grouped<Group> -> @index: U32 -> Witness.ArrayWitness<Tile.Tile8,input> -> Read(input,read(input,context,index)),
  +context: Kernel.Grouped<Group>,-input: Array<Tile.Tile8>,-output: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<Tile.Tile8,input>,
  output_certificate: Certificate.Certificate<Tile.Tile8,output>) ->
  TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile8,Certificate.family(~Tile.Tile8),Kernel.mapped(~Array<Tile.Tile8>,~Group,~read,context,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~Certificate.family(~Tile.Tile8),~Kernel.mapped(~Array<Tile.Tile8>,~Group,~read),
    ~array_preserves(~Group,~read,~read_preserves),context,input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},
    Certificate.Certificate{depth,equation},output_certificate)

def updated_preserves(+depth: Nat,context: Kernel.Grouped<Kernel.Back>,-xhat: Array<Tile.Tile8>,-dy: Array<Tile.Tile8>,xhat_certificate: Certificate.Certificate<Tile.Tile8,xhat>,
  equation: {dy == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,depth,Tree.reflect(Tile.Tile8,dy))) : Array<Tile.Tile8>}) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile8,xhat,depth,Kernel.updated(context,xhat,dy)>:
  +context = context
  TraversalCertificate.completed(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Kernel.Back>,~TraversalWitness.array_evidence(~Tile.Tile8),
    ~Traversal.update(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Sampled,~Kernel.Grouped<Kernel.Back>,~Kernel.back_read,~Kernel.back_combine,~Kernel.place(~Kernel.Back)),
    ~TraversalWitness.update(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Sampled,~Kernel.Grouped<Kernel.Back>,~TraversalWitness.array_evidence(~Tile.Tile8),~Kernel.back_read,~Kernel.back_combine,
      ~Kernel.place(~Kernel.Back),~back_read),
    U32.to_nat(Kernel.count_of(~Kernel.Back,context)),0,context,xhat,dy,Certificate.witness(Tile.Tile8,xhat,xhat_certificate),depth,equation)

# xhat handed back and dy updated in place.
def updated_certificates(+context: Kernel.Grouped<Kernel.Back>,-xhat: Array<Tile.Tile8>,-dy: Array<Tile.Tile8>,xhat_certificate: Certificate.Certificate<Tile.Tile8,xhat>,
  dy_certificate: Certificate.Certificate<Tile.Tile8,dy>) ->
  TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile8,Certificate.family(~Tile.Tile8),Kernel.updated(context,xhat,dy)>:
  Certificate.Certificate{+depth,+equation} = xhat_certificate
  TraversalCertificate.operation(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Kernel.Back>,~Certificate.family(~Tile.Tile8),~Kernel.updated,~updated_preserves,
    context,xhat,dy,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},dy_certificate)
