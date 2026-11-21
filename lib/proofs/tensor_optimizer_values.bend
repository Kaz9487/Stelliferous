# An SGD step composes the scaling of the gradient by the rate and the
# subtraction from the parameter: every element is parameter - gradient * rate
# in FP32, for a parameter and a gradient of one shape in any layout.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor.bend as Tensors
import ../tensor_f32.bend as F
import ../tensor_optimizer.bend as Optimizer
import ../tensor_tiled.bend as Tiled
import ../tensor_grid.bend as Grid
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./word_encoding.bend as Words
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_combine_leaf.bend as Leaf
import ./tensor_function_values.bend as FunctionValues
import ./tensor_binary_values.bend as Binary
import ./tensor_tiled_public.bend as Public
import ./tensor_copy_contract.bend as CopyContract

# The parameter, and the rate with the gradient.
def step_context(~Parameter: Data,~Gradient: Data,+parameter: Parameter,+gradient: Gradient,+rate: F32) -> Leaf.Context<Parameter,PiecesMap.Context<F32,Gradient>>:
  Leaf.Context{parameter,PiecesMap.Context{rate,gradient}}

def sgd(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+parameter: Parameter,+gradient: Gradient,+rate: F32,
  +shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  parameter_tiled: Tiled.Tiled,+parameter_contract: Public.Contract(~Parameter,~parameter_value,parameter,shape,parameter_tiled),
  gradient_tiled: Tiled.Tiled,+gradient_contract: Public.Contract(~Gradient,~gradient_value,gradient,shape,gradient_tiled)) ->
  Public.Contract(~Leaf.Context<Parameter,PiecesMap.Context<F32,Gradient>>,~Leaf.value(~F32,~Parameter,~PiecesMap.Context<F32,Gradient>,~F32.sub,~parameter_value,~PiecesMap.value(~F32,~F32,~Gradient,~F.scalar_value(~F32.mul),~gradient_value)),step_context(~Parameter,~Gradient,parameter,gradient,rate),shape,
    Optimizer.sgd(parameter_tiled,gradient_tiled,rate)):
  Public.by_packed(~Gradient,~gradient_value,(tiled => Public.Contract(~Leaf.Context<Parameter,PiecesMap.Context<F32,Gradient>>,~Leaf.value(~F32,~Parameter,~PiecesMap.Context<F32,Gradient>,~F32.sub,~parameter_value,~PiecesMap.value(~F32,~F32,~Gradient,~F.scalar_value(~F32.mul),~gradient_value)),
      step_context(~Parameter,~Gradient,parameter,gradient,rate),shape,Optimizer.sgd(parameter_tiled,tiled,rate))),gradient,shape,gradient_tiled,gradient_contract,
    model => cells => fill => workers => axis => lanes => position => contract =>
      Binary.same(~Parameter,~parameter_value,~PiecesMap.Context<F32,Gradient>,~PiecesMap.value(~F32,~F32,~Gradient,~F.scalar_value(~F32.mul),~gradient_value),~F32.sub,parameter,PiecesMap.Context{rate,gradient},"sub",1.0,
        limits,default_capacity,shape,nonempty,rank_room,parameter_tiled,parameter_contract,F.mul_scalar(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position},rate),
        FunctionValues.scalar_public(~Gradient,~gradient_value,~F32.mul,rate,1.0)(gradient,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position},contract)))

# ---- With momentum and weight decay ----

# The direction of a step: the gradient, plus the parameter times a decay that
# is not zero.
type Direction<-Parameter: Data,-Gradient: Data> is Data:
  Direction{zero: Bool,decay: F32,parameter: Parameter,gradient: Gradient}

def direction_value(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,context: Direction<Parameter,Gradient>,index: U32) -> F32:
  match context:
    case Direction{zero,+decay,parameter,gradient}:
      match zero:
        case True{}: gradient_value(gradient,index)
        case False{}:
          +index = index
          F32.add(gradient_value(gradient,index),F32.mul(parameter_value(parameter,index),decay))

# What a step applies: the direction alone for a momentum of zero or a first
# step, otherwise the previous velocity times the momentum plus the direction.
type Applied<-Along: Data,-Before: Data> is Data:
  Applied{still: Bool,moving: Bool,momentum: F32,along: Along,before: Before}

def applied_value(~Along: Data,~along_value: Along -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,context: Applied<Along,Before>,index: U32) -> F32:
  match context:
    case Applied{still,moving,+momentum,along,before}:
      match still:
        case True{}: along_value(along,index)
        case False{}:
          match moving:
            case False{}: along_value(along,index)
            case True{}:
              +index = index
              F32.add(F32.mul(before_value(before,index),momentum),along_value(along,index))

# A velocity that is moving exactly when the flag says so, and then holds the
# values of its reader.
def Kept(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+moving: Bool,velocity: Optimizer.Velocity) -> Data:
  match velocity:
    case Optimizer.Still{}: {moving == False{} : Bool}
    case Optimizer.Moving{tensor,rest}: KernelGemm.Both<{moving == True{} : Bool},Public.Contract(~Logical,~original,logical,shape,tensor)>

# The parameter after a step and the velocity that step leaves.
def Stepped(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Moved: Data,~moved_value: Moved -> U32 -> F32,+parameter: Parameter,+moved: Moved,+rate: F32,+shape: List<&2,U32>,+moving: Bool,
  result: Tiled.Tiled & Optimizer.Velocity) -> Data:
  (after,velocity) = result
  KernelGemm.Both<Public.Contract(~Leaf.Context<Parameter,PiecesMap.Context<F32,Moved>>,~Leaf.value(~F32,~Parameter,~PiecesMap.Context<F32,Moved>,~F32.sub,~parameter_value,~PiecesMap.value(~F32,~F32,~Moved,~F.scalar_value(~F32.mul),~moved_value)),step_context(~Parameter,~Moved,parameter,moved,rate),shape,after),
    Kept(~Moved,~moved_value,moved,shape,moving,velocity)>

# ---- Readers of one step ----

# A tensor of the contract for one reader is a tensor of the contract for
# every reader that agrees with it at every index.
def reread_then(~Before: Data,~After: Data,~before_value: Before -> U32 -> F32,~after_value: After -> U32 -> F32,~Related: Before -> After -> Data,
  ~agreement: Content.Agreement(~F32,~Before,~After,~before_value,~after_value,~Related),~Goal: Tiled.Tiled -> Data,+before: Before,+after: After,+related: Related(before,after),
  +values: List<&2,U32>,tiled: Tiled.Tiled,+contract: Public.Contract(~Before,~before_value,before,values,tiled),
  body: @tiled: Tiled.Tiled -> @+contract: Public.Contract(~After,~after_value,after,values,tiled) -> Goal(tiled)) -> Goal(tiled):
  Public.by_packed(~Before,~before_value,Goal,before,values,tiled,contract,
    model => cells => fill => workers => axis => lanes => position => packed =>
      body(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position},Public.reread(~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position},packed)))

# Whether a step applies the direction alone.
def alone(still: Bool,moving: Bool) -> Bool:
  match still:
    case True{}: True{}
    case False{}: Bool.not(moving)

# An application that reads the direction alone.
def Alone(~Along: Data,~Before: Data,along: Along,applied: Applied<Along,Before>) -> Data:
  match applied:
    case Applied{still,moving,momentum,other,before}: KernelGemm.Both<{alone(still,moving) == True{} : Bool},{other == along : Along}>

def alone_value(~Along: Data,~along_value: Along -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,still: Bool,moving: Bool,+momentum: F32,+along: Along,+before: Before,
  +reads: {alone(still,moving) == True{} : Bool},+index: U32) ->
  {along_value(along,index) == applied_value(~Along,~along_value,~Before,~before_value,Applied{still,moving,momentum,along,before},index) : F32}:
  match still:
    case True{}: {==}
    case False{}:
      match moving:
        case False{}: {==}
        case True{}:
          Empty.absurd({along_value(along,index) == applied_value(~Along,~along_value,~Before,~before_value,Applied{False{},True{},momentum,along,before},index) : F32},Words.false_true(reads))

def alone_agree(~Along: Data,~along_value: Along -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,+along: Along,+applied: Applied<Along,Before>,+related: Alone(~Along,~Before,along,applied),+index: U32) ->
  {along_value(along,index) == applied_value(~Along,~along_value,~Before,~before_value,applied,index) : F32}:
  match applied:
    case Applied{+still,+moving,+momentum,+other,+before}:
      KernelGemm.Both{+reads,+same} = related
      %Equal.sym(Along,other,along,same) : {along_value(along,index) == applied_value(~Along,~along_value,~Before,~before_value,Applied{still,moving,momentum,_,before},index) : F32}
      alone_value(~Along,~along_value,~Before,~before_value,still,moving,momentum,along,before,reads,index)

# velocity * momentum + direction as an application.
def accumulated_of(~Along: Data,~Before: Data,context: Leaf.Context<PiecesMap.Context<F32,Before>,Along>) -> Applied<Along,Before>:
  match context:
    case Leaf.Context{scaled,along}:
      match scaled:
        case PiecesMap.Context{momentum,before}: Applied{False{},True{},momentum,along,before}

def Accumulated(~Along: Data,~Before: Data,context: Leaf.Context<PiecesMap.Context<F32,Before>,Along>,applied: Applied<Along,Before>) -> Data:
  {applied == accumulated_of(~Along,~Before,context) : Applied<Along,Before>}

def accumulated_value(~Along: Data,~along_value: Along -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,+context: Leaf.Context<PiecesMap.Context<F32,Before>,Along>,+index: U32) ->
  {Leaf.value(~F32,~PiecesMap.Context<F32,Before>,~Along,~F32.add,~PiecesMap.value(~F32,~F32,~Before,~F.scalar_value(~F32.mul),~before_value),~along_value)(context,index) == applied_value(~Along,~along_value,~Before,~before_value,accumulated_of(~Along,~Before,context),index) : F32}:
  match context:
    case Leaf.Context{+scaled,+along}:
      match scaled:
        case PiecesMap.Context{+momentum,+before}: {==}

def accumulated_agree(~Along: Data,~along_value: Along -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,+context: Leaf.Context<PiecesMap.Context<F32,Before>,Along>,+applied: Applied<Along,Before>,+related: Accumulated(~Along,~Before,context,applied),+index: U32) ->
  {Leaf.value(~F32,~PiecesMap.Context<F32,Before>,~Along,~F32.add,~PiecesMap.value(~F32,~F32,~Before,~F.scalar_value(~F32.mul),~before_value),~along_value)(context,index) == applied_value(~Along,~along_value,~Before,~before_value,applied,index) : F32}:
  %Equal.sym(Applied<Along,Before>,applied,accumulated_of(~Along,~Before,context),related) :
    {Leaf.value(~F32,~PiecesMap.Context<F32,Before>,~Along,~F32.add,~PiecesMap.value(~F32,~F32,~Before,~F.scalar_value(~F32.mul),~before_value),~along_value)(context,index) == applied_value(~Along,~along_value,~Before,~before_value,_,index) : F32}
  accumulated_value(~Along,~along_value,~Before,~before_value,context,index)

# A direction that reads the gradient alone.
def Undecayed(~Parameter: Data,~Gradient: Data,gradient: Gradient,direction: Direction<Parameter,Gradient>) -> Data:
  match direction:
    case Direction{zero,decay,parameter,other}: KernelGemm.Both<{zero == True{} : Bool},{other == gradient : Gradient}>

def undecayed_value(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,zero: Bool,+decay: F32,+parameter: Parameter,+gradient: Gradient,
  +reads: {zero == True{} : Bool},+index: U32) -> {gradient_value(gradient,index) == direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value)(Direction{zero,decay,parameter,gradient},index) : F32}:
  match zero:
    case True{}: {==}
    case False{}:
      Empty.absurd({gradient_value(gradient,index) == direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value)(Direction{False{},decay,parameter,gradient},index) : F32},Words.false_true(reads))

def undecayed_agree(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+gradient: Gradient,+direction: Direction<Parameter,Gradient>,
  +related: Undecayed(~Parameter,~Gradient,gradient,direction),+index: U32) -> {gradient_value(gradient,index) == direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value)(direction,index) : F32}:
  match direction:
    case Direction{+zero,+decay,+parameter,+other}:
      KernelGemm.Both{+reads,+same} = related
      %Equal.sym(Gradient,other,gradient,same) : {gradient_value(gradient,index) == direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value)(Direction{zero,decay,parameter,_},index) : F32}
      undecayed_value(~Parameter,~parameter_value,~Gradient,~gradient_value,zero,decay,parameter,gradient,reads,index)

# gradient + parameter * decay as a direction.
def decayed_of(~Parameter: Data,~Gradient: Data,context: Leaf.Context<Gradient,PiecesMap.Context<F32,Parameter>>) -> Direction<Parameter,Gradient>:
  match context:
    case Leaf.Context{gradient,scaled}:
      match scaled:
        case PiecesMap.Context{decay,parameter}: Direction{False{},decay,parameter,gradient}

def Decayed(~Parameter: Data,~Gradient: Data,context: Leaf.Context<Gradient,PiecesMap.Context<F32,Parameter>>,direction: Direction<Parameter,Gradient>) -> Data:
  {direction == decayed_of(~Parameter,~Gradient,context) : Direction<Parameter,Gradient>}

def decayed_value(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+context: Leaf.Context<Gradient,PiecesMap.Context<F32,Parameter>>,+index: U32) ->
  {Leaf.value(~F32,~Gradient,~PiecesMap.Context<F32,Parameter>,~F32.add,~gradient_value,~PiecesMap.value(~F32,~F32,~Parameter,~F.scalar_value(~F32.mul),~parameter_value))(context,index) == direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value)(decayed_of(~Parameter,~Gradient,context),index) : F32}:
  match context:
    case Leaf.Context{+gradient,+scaled}:
      match scaled:
        case PiecesMap.Context{+decay,+parameter}: {==}

def decayed_agree(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+context: Leaf.Context<Gradient,PiecesMap.Context<F32,Parameter>>,+direction: Direction<Parameter,Gradient>,
  +related: Decayed(~Parameter,~Gradient,context,direction),+index: U32) -> {Leaf.value(~F32,~Gradient,~PiecesMap.Context<F32,Parameter>,~F32.add,~gradient_value,~PiecesMap.value(~F32,~F32,~Parameter,~F.scalar_value(~F32.mul),~parameter_value))(context,index) == direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value)(direction,index) : F32}:
  %Equal.sym(Direction<Parameter,Gradient>,direction,decayed_of(~Parameter,~Gradient,context),related) : {Leaf.value(~F32,~Gradient,~PiecesMap.Context<F32,Parameter>,~F32.add,~gradient_value,~PiecesMap.value(~F32,~F32,~Parameter,~F.scalar_value(~F32.mul),~parameter_value))(context,index) == direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value)(_,index) : F32}
  decayed_value(~Parameter,~parameter_value,~Gradient,~gradient_value,context,index)

# ---- One step ----

# A step that keeps what it applies as the velocity.
def kept_step(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Moved: Data,~moved_value: Moved -> U32 -> F32,+parameter: Parameter,+moved: Moved,+rate: F32,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  before: Tiled.Tiled,+before_contract: Public.Contract(~Parameter,~parameter_value,parameter,shape,before),
  tensor: Tiled.Tiled,+contract: Public.Contract(~Moved,~moved_value,moved,shape,tensor)) ->
  Stepped(~Parameter,~parameter_value,~Moved,~moved_value,parameter,moved,rate,shape,True{},Optimizer.descended(before,rate,Optimizer.kept_velocity(F.copy(tensor)))):
  CopyContract.tiled_copied(~Moved,~moved_value,~(pair => Stepped(~Parameter,~parameter_value,~Moved,~moved_value,parameter,moved,rate,shape,True{},
      Optimizer.descended(before,rate,Optimizer.kept_velocity(pair)))),moved,shape,tensor,contract,
    first => second => first_contract => second_contract =>
      KernelGemm.Both{sgd(~Parameter,~parameter_value,~Moved,~moved_value,parameter,moved,rate,shape,limits,default_capacity,nonempty,rank_room,before,before_contract,first,first_contract),KernelGemm.Both{{==},second_contract}})

# The first step of a momentum that is not zero keeps the direction.
def first_step(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Along: Data,~along_value: Along -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,+parameter: Parameter,+along: Along,+previous: Before,+rate: F32,+momentum: F32,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  before: Tiled.Tiled,+before_contract: Public.Contract(~Parameter,~parameter_value,parameter,shape,before),
  direction: Tiled.Tiled,+direction_contract: Public.Contract(~Along,~along_value,along,shape,direction)) ->
  Stepped(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{False{},False{},momentum,along,previous},rate,shape,True{},
    Optimizer.descended(before,rate,Optimizer.kept_velocity(F.copy(direction)))):
  reread_then(~Along,~Applied<Along,Before>,~along_value,~applied_value(~Along,~along_value,~Before,~before_value),~Alone(~Along,~Before),~alone_agree(~Along,~along_value,~Before,~before_value),
    ~(tiled => Stepped(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{False{},False{},momentum,along,previous},rate,shape,True{},
      Optimizer.descended(before,rate,Optimizer.kept_velocity(F.copy(tiled))))),
    along,Applied{False{},False{},momentum,along,previous},KernelGemm.Both{{==},{==}},shape,direction,direction_contract,
    tiled => contract =>
      kept_step(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{False{},False{},momentum,along,previous},rate,shape,limits,default_capacity,nonempty,rank_room,before,before_contract,tiled,contract))

# A later step keeps velocity * momentum + direction.
def later_step(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Along: Data,~along_value: Along -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,+parameter: Parameter,+along: Along,+previous: Before,+rate: F32,+momentum: F32,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  before: Tiled.Tiled,+before_contract: Public.Contract(~Parameter,~parameter_value,parameter,shape,before),
  direction: Tiled.Tiled,+direction_contract: Public.Contract(~Along,~along_value,along,shape,direction),
  velocity: Tiled.Tiled,+velocity_contract: Public.Contract(~Before,~before_value,previous,shape,velocity)) ->
  Stepped(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{False{},True{},momentum,along,previous},rate,shape,True{},
    Optimizer.descended(before,rate,Optimizer.kept_velocity(F.copy(F.add(F.mul_scalar(velocity,momentum),direction))))):
  Public.by_packed(~Before,~before_value,(tiled => Stepped(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{False{},True{},momentum,along,previous},rate,shape,True{},
      Optimizer.descended(before,rate,Optimizer.kept_velocity(F.copy(F.add(F.mul_scalar(tiled,momentum),direction)))))),previous,shape,velocity,velocity_contract,
    vm => vc => vf => vw => va => vl => vp => scaled_contract =>
      Public.by_packed(~Along,~along_value,(tiled => Stepped(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{False{},True{},momentum,along,previous},rate,shape,True{},
          Optimizer.descended(before,rate,Optimizer.kept_velocity(F.copy(F.add(F.mul_scalar(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vc,vf,Tensors.Usable{},vw},va,vl,vp},momentum),tiled)))))),along,shape,direction,direction_contract,
        dm => dc => df => dw => da => dl => dp => along_contract =>
          kept_step(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{False{},True{},momentum,along,previous},rate,shape,limits,default_capacity,nonempty,rank_room,before,before_contract,
            F.add(F.mul_scalar(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vc,vf,Tensors.Usable{},vw},va,vl,vp},momentum),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dc,df,Tensors.Usable{},dw},da,dl,dp}),
            Public.reread(~Leaf.Context<PiecesMap.Context<F32,Before>,Along>,~Applied<Along,Before>,~Leaf.value(~F32,~PiecesMap.Context<F32,Before>,~Along,~F32.add,~PiecesMap.value(~F32,~F32,~Before,~F.scalar_value(~F32.mul),~before_value),~along_value),~applied_value(~Along,~along_value,~Before,~before_value),~Accumulated(~Along,~Before),~accumulated_agree(~Along,~along_value,~Before,~before_value),
              Leaf.Context{PiecesMap.Context{momentum,previous},along},Applied{False{},True{},momentum,along,previous},{==},shape,
              F.add(F.mul_scalar(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vc,vf,Tensors.Usable{},vw},va,vl,vp},momentum),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dc,df,Tensors.Usable{},dw},da,dl,dp}),
              Binary.same(~PiecesMap.Context<F32,Before>,~PiecesMap.value(~F32,~F32,~Before,~F.scalar_value(~F32.mul),~before_value),~Along,~along_value,~F32.add,PiecesMap.Context{momentum,previous},along,"add",1.0,
                limits,default_capacity,shape,nonempty,rank_room,F.mul_scalar(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vc,vf,Tensors.Usable{},vw},va,vl,vp},momentum),
                FunctionValues.scalar_public(~Before,~before_value,~F32.mul,momentum,1.0)(previous,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vc,vf,Tensors.Usable{},vw},va,vl,vp},scaled_contract),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dc,df,Tensors.Usable{},dw},da,dl,dp},along_contract)))))

def accumulated_holds(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Along: Data,~along_value: Along -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,+parameter: Parameter,+along: Along,+previous: Before,+rate: F32,+momentum: F32,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+moving: Bool,
  before: Tiled.Tiled,+before_contract: Public.Contract(~Parameter,~parameter_value,parameter,shape,before),
  direction: Tiled.Tiled,+direction_contract: Public.Contract(~Along,~along_value,along,shape,direction),
  velocity: Optimizer.Velocity,+kept: Kept(~Before,~before_value,previous,shape,moving,velocity)) ->
  Stepped(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{False{},moving,momentum,along,previous},rate,shape,True{},
    Optimizer.descended(before,rate,Optimizer.accumulated(velocity,direction,momentum))):
  match velocity:
    case Optimizer.Still{}:
      %Equal.sym(Bool,moving,False{},kept) : Stepped(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{False{},_,momentum,along,previous},rate,shape,True{},
        Optimizer.descended(before,rate,Optimizer.kept_velocity(F.copy(direction))))
      first_step(~Parameter,~parameter_value,~Along,~along_value,~Before,~before_value,parameter,along,previous,rate,momentum,shape,limits,default_capacity,nonempty,rank_room,before,before_contract,direction,direction_contract)
    case Optimizer.Moving{tensor,rest}:
      KernelGemm.Both{+is_moving,+tensor_contract} = kept
      %Equal.sym(Bool,moving,True{},is_moving) : Stepped(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{False{},_,momentum,along,previous},rate,shape,True{},
        Optimizer.descended(before,rate,Optimizer.kept_velocity(F.copy(F.add(F.mul_scalar(tensor,momentum),direction)))))
      later_step(~Parameter,~parameter_value,~Along,~along_value,~Before,~before_value,parameter,along,previous,rate,momentum,shape,limits,default_capacity,nonempty,rank_room,before,before_contract,direction,direction_contract,tensor,tensor_contract)

# A velocity remains after a step exactly when the momentum is not zero.
def moved_holds(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Along: Data,~along_value: Along -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,still: Bool,+parameter: Parameter,+along: Along,+previous: Before,+rate: F32,+momentum: F32,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+moving: Bool,
  before: Tiled.Tiled,+before_contract: Public.Contract(~Parameter,~parameter_value,parameter,shape,before),
  direction: Tiled.Tiled,+direction_contract: Public.Contract(~Along,~along_value,along,shape,direction),
  velocity: Optimizer.Velocity,+kept: Kept(~Before,~before_value,previous,shape,moving,velocity)) ->
  Stepped(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{still,moving,momentum,along,previous},rate,shape,Bool.not(still),
    Optimizer.descended(before,rate,Optimizer.moved(still,velocity,direction,momentum))):
  match still:
    case True{}:
      reread_then(~Along,~Applied<Along,Before>,~along_value,~applied_value(~Along,~along_value,~Before,~before_value),~Alone(~Along,~Before),~alone_agree(~Along,~along_value,~Before,~before_value),
        ~(tiled => Stepped(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{True{},moving,momentum,along,previous},rate,shape,False{},
          Optimizer.descended(before,rate,Optimizer.moved(True{},velocity,tiled,momentum)))),
        along,Applied{True{},moving,momentum,along,previous},KernelGemm.Both{{==},{==}},shape,direction,direction_contract,
        tiled => contract =>
          KernelGemm.Both{sgd(~Parameter,~parameter_value,~Applied<Along,Before>,~applied_value(~Along,~along_value,~Before,~before_value),parameter,Applied{True{},moving,momentum,along,previous},rate,shape,limits,default_capacity,nonempty,rank_room,before,before_contract,tiled,contract),{==}})
    case False{}:
      accumulated_holds(~Parameter,~parameter_value,~Along,~along_value,~Before,~before_value,parameter,along,previous,rate,momentum,shape,limits,default_capacity,nonempty,rank_room,moving,before,before_contract,direction,direction_contract,velocity,kept)

# The direction is the gradient plus a copy of the parameter times the decay.
def decayed_step(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,+parameter: Parameter,+gradient: Gradient,+previous: Before,+rate: F32,+momentum: F32,+decay: F32,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+moving: Bool,
  first: Tiled.Tiled,+first_contract: Public.Contract(~Parameter,~parameter_value,parameter,shape,first),
  second: Tiled.Tiled,+second_contract: Public.Contract(~Parameter,~parameter_value,parameter,shape,second),
  tensor: Tiled.Tiled,+tensor_contract: Public.Contract(~Gradient,~gradient_value,gradient,shape,tensor),
  velocity: Optimizer.Velocity,+kept: Kept(~Before,~before_value,previous,shape,moving,velocity)) ->
  Stepped(~Parameter,~parameter_value,~Applied<Direction<Parameter,Gradient>,Before>,~applied_value(~Direction<Parameter,Gradient>,~direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value),~Before,~before_value),
    parameter,Applied{F32.is_eq(momentum,0.0),moving,momentum,Direction{False{},decay,parameter,gradient},previous},rate,shape,Bool.not(F32.is_eq(momentum,0.0)),
    Optimizer.descended(first,rate,Optimizer.moved(F32.is_eq(momentum,0.0),velocity,F.add(tensor,F.mul_scalar(second,decay)),momentum))):
  Public.by_packed(~Parameter,~parameter_value,(tiled => Stepped(~Parameter,~parameter_value,~Applied<Direction<Parameter,Gradient>,Before>,~applied_value(~Direction<Parameter,Gradient>,~direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value),~Before,~before_value),
      parameter,Applied{F32.is_eq(momentum,0.0),moving,momentum,Direction{False{},decay,parameter,gradient},previous},rate,shape,Bool.not(F32.is_eq(momentum,0.0)),
      Optimizer.descended(first,rate,Optimizer.moved(F32.is_eq(momentum,0.0),velocity,F.add(tensor,F.mul_scalar(tiled,decay)),momentum)))),parameter,shape,second,second_contract,
    sm => sc => sf => sw => sa => sl => sp => scaled_contract =>
      Public.by_packed(~Gradient,~gradient_value,(tiled => Stepped(~Parameter,~parameter_value,~Applied<Direction<Parameter,Gradient>,Before>,~applied_value(~Direction<Parameter,Gradient>,~direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value),~Before,~before_value),
          parameter,Applied{F32.is_eq(momentum,0.0),moving,momentum,Direction{False{},decay,parameter,gradient},previous},rate,shape,Bool.not(F32.is_eq(momentum,0.0)),
          Optimizer.descended(first,rate,Optimizer.moved(F32.is_eq(momentum,0.0),velocity,F.add(tiled,F.mul_scalar(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,sm),sc,sf,Tensors.Usable{},sw},sa,sl,sp},decay)),momentum)))),gradient,shape,tensor,tensor_contract,
        tm => tc => tf => tw => ta => tl => tp => gradient_contract =>
          moved_holds(~Parameter,~parameter_value,~Direction<Parameter,Gradient>,~direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value),~Before,~before_value,F32.is_eq(momentum,0.0),parameter,Direction{False{},decay,parameter,gradient},previous,rate,momentum,shape,limits,default_capacity,nonempty,rank_room,moving,
            first,first_contract,F.add(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),tc,tf,Tensors.Usable{},tw},ta,tl,tp},F.mul_scalar(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,sm),sc,sf,Tensors.Usable{},sw},sa,sl,sp},decay)),
            Public.reread(~Leaf.Context<Gradient,PiecesMap.Context<F32,Parameter>>,~Direction<Parameter,Gradient>,~Leaf.value(~F32,~Gradient,~PiecesMap.Context<F32,Parameter>,~F32.add,~gradient_value,~PiecesMap.value(~F32,~F32,~Parameter,~F.scalar_value(~F32.mul),~parameter_value)),~direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value),~Decayed(~Parameter,~Gradient),~decayed_agree(~Parameter,~parameter_value,~Gradient,~gradient_value),
              Leaf.Context{gradient,PiecesMap.Context{decay,parameter}},Direction{False{},decay,parameter,gradient},{==},shape,F.add(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),tc,tf,Tensors.Usable{},tw},ta,tl,tp},F.mul_scalar(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,sm),sc,sf,Tensors.Usable{},sw},sa,sl,sp},decay)),
              Binary.same(~Gradient,~gradient_value,~PiecesMap.Context<F32,Parameter>,~PiecesMap.value(~F32,~F32,~Parameter,~F.scalar_value(~F32.mul),~parameter_value),~F32.add,gradient,PiecesMap.Context{decay,parameter},"add",1.0,
                limits,default_capacity,shape,nonempty,rank_room,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),tc,tf,Tensors.Usable{},tw},ta,tl,tp},gradient_contract,F.mul_scalar(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,sm),sc,sf,Tensors.Usable{},sw},sa,sl,sp},decay),
                FunctionValues.scalar_public(~Parameter,~parameter_value,~F32.mul,decay,1.0)(parameter,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,sm),sc,sf,Tensors.Usable{},sw},sa,sl,sp},scaled_contract))),velocity,kept)))

def decayed_holds(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,zero: Bool,+parameter: Parameter,+gradient: Gradient,+previous: Before,+rate: F32,+momentum: F32,+decay: F32,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+moving: Bool,
  before: Tiled.Tiled,+before_contract: Public.Contract(~Parameter,~parameter_value,parameter,shape,before),
  tensor: Tiled.Tiled,+tensor_contract: Public.Contract(~Gradient,~gradient_value,gradient,shape,tensor),
  velocity: Optimizer.Velocity,+kept: Kept(~Before,~before_value,previous,shape,moving,velocity)) ->
  Stepped(~Parameter,~parameter_value,~Applied<Direction<Parameter,Gradient>,Before>,~applied_value(~Direction<Parameter,Gradient>,~direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value),~Before,~before_value),
    parameter,Applied{F32.is_eq(momentum,0.0),moving,momentum,Direction{zero,decay,parameter,gradient},previous},rate,shape,Bool.not(F32.is_eq(momentum,0.0)),
    Optimizer.stepped(velocity,rate,momentum,Optimizer.decayed(zero,before,tensor,decay))):
  match zero:
    case True{}:
      reread_then(~Gradient,~Direction<Parameter,Gradient>,~gradient_value,~direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value),~Undecayed(~Parameter,~Gradient),~undecayed_agree(~Parameter,~parameter_value,~Gradient,~gradient_value),
        ~(tiled => Stepped(~Parameter,~parameter_value,~Applied<Direction<Parameter,Gradient>,Before>,~applied_value(~Direction<Parameter,Gradient>,~direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value),~Before,~before_value),
          parameter,Applied{F32.is_eq(momentum,0.0),moving,momentum,Direction{True{},decay,parameter,gradient},previous},rate,shape,Bool.not(F32.is_eq(momentum,0.0)),
          Optimizer.stepped(velocity,rate,momentum,Optimizer.decayed(True{},before,tiled,decay)))),
        gradient,Direction{True{},decay,parameter,gradient},KernelGemm.Both{{==},{==}},shape,tensor,tensor_contract,
        tiled => contract =>
          moved_holds(~Parameter,~parameter_value,~Direction<Parameter,Gradient>,~direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value),~Before,~before_value,F32.is_eq(momentum,0.0),parameter,Direction{True{},decay,parameter,gradient},previous,rate,momentum,shape,limits,default_capacity,nonempty,rank_room,moving,
            before,before_contract,tiled,contract,velocity,kept))
    case False{}:
      CopyContract.tiled_copied(~Parameter,~parameter_value,~(pair => Stepped(~Parameter,~parameter_value,~Applied<Direction<Parameter,Gradient>,Before>,~applied_value(~Direction<Parameter,Gradient>,~direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value),~Before,~before_value),
          parameter,Applied{F32.is_eq(momentum,0.0),moving,momentum,Direction{False{},decay,parameter,gradient},previous},rate,shape,Bool.not(F32.is_eq(momentum,0.0)),
          Optimizer.stepped(velocity,rate,momentum,Optimizer.decayed_copies(tensor,decay,pair)))),parameter,shape,before,before_contract,
        first => second => first_contract => second_contract =>
          decayed_step(~Parameter,~parameter_value,~Gradient,~gradient_value,~Before,~before_value,parameter,gradient,previous,rate,momentum,decay,shape,limits,default_capacity,nonempty,rank_room,moving,first,first_contract,second,second_contract,tensor,tensor_contract,velocity,kept))

# A parameter, a gradient and a previous velocity of one shape without a zero
# extent, each a tiled tensor of the contract: the parameter after the step
# holds parameter - applied * rate at every index. The velocity left holds
# what was applied, and it moves exactly when the momentum is not zero, so it
# is the previous velocity of the next step.
def sgd_with(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Before: Data,~before_value: Before -> U32 -> F32,+parameter: Parameter,+gradient: Gradient,+previous: Before,+rate: F32,+momentum: F32,+decay: F32,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+moving: Bool,
  before: Tiled.Tiled,+before_contract: Public.Contract(~Parameter,~parameter_value,parameter,shape,before),
  tensor: Tiled.Tiled,+tensor_contract: Public.Contract(~Gradient,~gradient_value,gradient,shape,tensor),
  velocity: Optimizer.Velocity,+kept: Kept(~Before,~before_value,previous,shape,moving,velocity)) ->
  Stepped(~Parameter,~parameter_value,~Applied<Direction<Parameter,Gradient>,Before>,~applied_value(~Direction<Parameter,Gradient>,~direction_value(~Parameter,~parameter_value,~Gradient,~gradient_value),~Before,~before_value),
    parameter,Applied{F32.is_eq(momentum,0.0),moving,momentum,Direction{F32.is_eq(decay,0.0),decay,parameter,gradient},previous},rate,shape,Bool.not(F32.is_eq(momentum,0.0)),
    Optimizer.sgd_with(before,tensor,velocity,rate,momentum,decay)):
  decayed_holds(~Parameter,~parameter_value,~Gradient,~gradient_value,~Before,~before_value,F32.is_eq(decay,0.0),parameter,gradient,previous,rate,momentum,decay,shape,limits,default_capacity,nonempty,rank_room,moving,before,before_contract,tensor,tensor_contract,velocity,kept)

# ---- Velocities as models ----

# A velocity whose tensors are the packed cells of models: a proof reads it as
# often as it needs.
type VelocityModel is Data:
  StillModel{}
  MovingModel{model: PiecesModel.Model<Tile.Tile8>,shape: List<&2,U32>,fill: Tile.Tile8,status: Tensors.Status,workers: U32,axis: Nat,lanes: U32,position: Grid.Position,
    rest: VelocityModel}

def velocity_of(model: VelocityModel) -> Optimizer.Velocity:
  match model:
    case StillModel{}: Optimizer.Still{}
    case MovingModel{cells,+shape,+fill,+status,+workers,+axis,+lanes,+position,rest}:
      Optimizer.Moving{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,cells),shape,fill,status,workers},axis,lanes,position},velocity_of(rest)}

type VelocityObserved<-velocity: Optimizer.Velocity> is Data:
  VelocityObserved{model: VelocityModel,equation: {velocity_of(model) == velocity : Optimizer.Velocity}}

def moving_observed(-tensor: Tiled.Tiled,+tensor_observed: Public.Observed<tensor>,-rest: Optimizer.Velocity,+rest_observed: VelocityObserved<rest>) ->
  VelocityObserved<Optimizer.Moving{tensor,rest}>:
  Public.Observed{+model,+shape,+fill,+status,+workers,+axis,+lanes,+position,+tensor_equation} = tensor_observed
  VelocityObserved{+rest_model,+rest_equation} = rest_observed
  VelocityObserved{MovingModel{model,shape,fill,status,workers,axis,lanes,position,rest_model},
    {%tensor_equation : {Optimizer.Moving{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,status,workers},axis,lanes,position},velocity_of(rest_model)} == Optimizer.Moving{_,rest} : Optimizer.Velocity}
      {%rest_equation : {Optimizer.Moving{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,status,workers},axis,lanes,position},velocity_of(rest_model)} == Optimizer.Moving{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,status,workers},axis,lanes,position},_} : Optimizer.Velocity}
        {==} :
        {Optimizer.Moving{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,status,workers},axis,lanes,position},velocity_of(rest_model)} == Optimizer.Moving{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,status,workers},axis,lanes,position},rest} : Optimizer.Velocity}} :
      {Optimizer.Moving{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,status,workers},axis,lanes,position},velocity_of(rest_model)} == Optimizer.Moving{tensor,rest} : Optimizer.Velocity}}}

# Any velocity as the velocity of a model.
def observe_velocity(velocity: Optimizer.Velocity) -> VelocityObserved<velocity>:
  match velocity:
    case Optimizer.Still{}: VelocityObserved{StillModel{},{==}}
    case Optimizer.Moving{tensor,rest}: moving_observed(tensor,Public.observe(tensor),rest,observe_velocity(rest))
