# Mean squared error composes copy, subtraction, square, reduction and scaling.
import Base
import ../tensor.bend as Tensors
import ../tensor_loss.bend as Loss
import ../tensor_f32.bend as F32Tensor
import ../tensor_shape.bend as Shapes
import ../tensor_access.bend as Access
import ../tensor_grid.bend as Grid
import ../tensor_layout.bend as Layout
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_map.bend as TiledMap
import ../tensor_tiled_reduce.bend as TiledReduce
import ../math_sum.bend as Sum
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_combine_leaf.bend as Leaf
import ./tensor_partials_values.bend as PartialsValues
import ./tensor_reduce_values.bend as Reduce
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_map.bend as TiledMaps
import ./tensor_tiled_sums.bend as Sums
import ./tensor_function_values.bend as FunctionValues
import ./tensor_binary_values.bend as Binary
import ./tensor_copy_contract.bend as Copy
import ./word_encoding.bend as Words

# A loss or gradient of the values divided by the count for Mean, as it is for
# Sum.
def Normalized(~Logical: Data,~original: Logical -> U32 -> F32,reduction: Loss.Reduction,+count: F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled) -> Data:
  match reduction:
    case Loss.Mean{}: Public.Contract(~PiecesMap.Context<F32,Logical>,~PiecesMap.value(~F32,~F32,~Logical,~F32Tensor.scalar_value(~F32.div),~original),PiecesMap.Context{count,logical},values,tiled)
    case Loss.Sum{}: Public.Contract(~Logical,~original,logical,values,tiled)

# The rank-0 loss holds the sum of the squares of the difference of prediction
# and target, the gradient holds the difference times two, and the target
# returns as it was given.
def Contract(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+shape: List<&2,U32>,
  +reduction: Loss.Reduction,result: Loss.Tensor() & (Loss.Tensor() & Loss.Tensor())) -> Data:
  (loss,(gradient,returned)) = result
  KernelGemm.Both<Normalized(~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),reduction,U32.to_f32(Tensors.count_or_zero(shape)),Sums.sum_box(~PiecesMap.Context<Unit,Leaf.Context<Prediction,Target>>,~PiecesMap.value(~F32,~Unit,~Leaf.Context<Prediction,Target>,~Access.apply_element(~F32,~Loss.square),~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value)),PiecesMap.Context{Unit{},Leaf.Context{prediction,target}},shape,TiledReduce.every_axis(List.length(&2,U32,shape),0)),Layout.extents_at(shape,Reduce.kept_axes(shape,TiledReduce.every_axis(List.length(&2,U32,shape),0))),loss),
    KernelGemm.Both<Normalized(~PiecesMap.Context<F32,Leaf.Context<Prediction,Target>>,~PiecesMap.value(~F32,~F32,~Leaf.Context<Prediction,Target>,~F32Tensor.scalar_value(~F32.mul),~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value)),reduction,U32.to_f32(Tensors.count_or_zero(shape)),PiecesMap.Context{2.0,Leaf.Context{prediction,target}},shape,gradient),
      Public.Contract(~Target,~target_value,target,shape,returned)>>

# A loss or gradient of the contract, divided by the count for Mean.
def normalized(~Logical: Data,~original: Logical -> U32 -> F32,reduction: Loss.Reduction,+count: F32,+logical: Logical,+values: List<&2,U32>,tiled: Loss.Tensor(),
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) -> Normalized(~Logical,~original,reduction,count,logical,values,Loss.reduced(reduction,count,tiled)):
  match reduction:
    case Loss.Mean{}: FunctionValues.scalar_public(~Logical,~original,~F32.div,count,1.0)(logical,values,tiled,contract)
    case Loss.Sum{}: contract
# The loss and gradient of the two copies of a difference of the contract.
def squared(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+reduction: Loss.Reduction,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},-returned: Loss.Tensor(),
  +first: PiecesModel.Model<Tile.Tile8>,+second: PiecesModel.Model<Tile.Tile8>,+cells: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +first_contract: Public.Contract(~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,first),cells,fill,Tensors.Usable{},workers},axis,lanes,position}),
  +second_contract: Public.Contract(~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,second),cells,fill,Tensors.Usable{},workers},axis,lanes,position}),
  +returned_contract: Public.Contract(~Target,~target_value,target,shape,returned)) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(shape)),returned,(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,first),cells,fill,Tensors.Usable{},workers},axis,lanes,position},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,second),cells,fill,Tensors.Usable{},workers},axis,lanes,position}))):
  KernelGemm.Both{
    normalized(~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),reduction,U32.to_f32(Tensors.count_or_zero(shape)),Sums.sum_box(~PiecesMap.Context<Unit,Leaf.Context<Prediction,Target>>,~PiecesMap.value(~F32,~Unit,~Leaf.Context<Prediction,Target>,~Access.apply_element(~F32,~Loss.square),~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value)),PiecesMap.Context{Unit{},Leaf.Context{prediction,target}},shape,TiledReduce.every_axis(List.length(&2,U32,shape),0)),Layout.extents_at(shape,Reduce.kept_axes(shape,TiledReduce.every_axis(List.length(&2,U32,shape),0))),TiledReduce.sum_all("mse_loss",F32Tensor.tiled_lanes(~Loss.square,1.0,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,second),cells,fill,Tensors.Usable{},workers},axis,lanes,position})),
      Sums.tiled_sum_all(~PiecesMap.Context<Unit,Leaf.Context<Prediction,Target>>,~PiecesMap.value(~F32,~Unit,~Leaf.Context<Prediction,Target>,~Access.apply_element(~F32,~Loss.square),~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value)),PiecesMap.Context{Unit{},Leaf.Context{prediction,target}},limits,"mse_loss",shape,F32Tensor.tiled_lanes(~Loss.square,1.0,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,second),cells,fill,Tensors.Usable{},workers},axis,lanes,position}),
        FunctionValues.public(~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),~Loss.square,1.0)(Leaf.Context{prediction,target},shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,second),cells,fill,Tensors.Usable{},workers},axis,lanes,position},second_contract),
        default_capacity,rank_room,nonempty,cap,cap_ok,count_room)),
    KernelGemm.Both{
      normalized(~PiecesMap.Context<F32,Leaf.Context<Prediction,Target>>,~PiecesMap.value(~F32,~F32,~Leaf.Context<Prediction,Target>,~F32Tensor.scalar_value(~F32.mul),~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value)),reduction,U32.to_f32(Tensors.count_or_zero(shape)),PiecesMap.Context{2.0,Leaf.Context{prediction,target}},shape,F32Tensor.mul_scalar(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,first),cells,fill,Tensors.Usable{},workers},axis,lanes,position},2.0),
        FunctionValues.scalar_public(~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),~F32.mul,2.0,1.0)(Leaf.Context{prediction,target},shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,first),cells,fill,Tensors.Usable{},workers},axis,lanes,position},first_contract)),
      returned_contract}}
# A difference of the contract, measured as a tensor of the shape and copied.
def measured_model(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+reduction: Loss.Reduction,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},-returned: Loss.Tensor(),
  +model: PiecesModel.Model<Tile.Tile8>,+cells: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,evidence: Copy.Copied<Tile.Tile8,model>,
  +contract: Public.Contract(~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position}),
  +returned_contract: Public.Contract(~Target,~target_value,target,shape,returned)) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_counted(reduction,returned,F32Tensor.shape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position}))):
  Copy.Copied{+first,+second,+equation,+first_content,+second_content} = evidence
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  +pair_equation = Copy.copied_equation(~Tile.Tile8,32.0,model,cells,fill,workers,first,second,equation)
  +before = {Representation.Lifted{Leaf.Context{prediction,target},Public.geometry_of(shape,axis,position)} : Representation.Lifted<Leaf.Context<Prediction,Target>>}
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(cells,axis,lanes,position),shape,restored) : Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(_)),returned,F32Tensor.copy(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position})))
  %Equal.sym(Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>,Tensors.copied(~Tile.Tile8,32.0,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers}),(Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,first),cells,fill,Tensors.Usable{},workers},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,second),cells,fill,Tensors.Usable{},workers}),pair_equation) :
    Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(shape)),returned,Tiled.copied(axis,lanes,position,_)))
  squared(~Prediction,~Target,~prediction_value,~target_value,prediction,target,reduction,shape,limits,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,returned,first,second,cells,fill,workers,axis,lanes,position,
    KernelGemm.Both{Copy.kept(~Tile.Tile8,~Representation.Lifted<Leaf.Context<Prediction,Target>>,~Representation.lifted(~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value)),before,model,first,cells,fill,workers,first_content,global),Public.Laid{inside,shaped,restored,counted,pinned}},
    KernelGemm.Both{Copy.kept(~Tile.Tile8,~Representation.Lifted<Leaf.Context<Prediction,Target>>,~Representation.lifted(~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value)),before,model,second,cells,fill,workers,second_content,global),Public.Laid{inside,shaped,restored,counted,pinned}},
    returned_contract)

def measured(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+reduction: Loss.Reduction,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},-returned: Loss.Tensor(),
  -difference: Loss.Tensor(),packed: Public.Packed<difference>,+contract: Public.Contract(~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},shape,difference),
  +returned_contract: Public.Contract(~Target,~target_value,target,shape,returned)) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_counted(reduction,returned,F32Tensor.shape(difference))):
  Public.Packed{+model,+cells,+fill,+workers,+axis,+lanes,+position,+equation} = packed
  %equation : Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_counted(reduction,returned,F32Tensor.shape(_)))
  measured_model(~Prediction,~Target,~prediction_value,~target_value,prediction,target,reduction,shape,limits,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,returned,model,cells,fill,workers,axis,lanes,position,Copy.copied(~Tile.Tile8,model),
    %Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position},difference,equation) : Public.Contract(~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},shape,_) contract,
    returned_contract)
# Operands of the same shape: the difference of the prediction and the kept
# copy of the target.
def kept_model(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+reduction: Loss.Reduction,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},-returned: Loss.Tensor(),
  +pm: PiecesModel.Model<Tile.Tile8>,+ps: List<&2,U32>,+pf: Tile.Tile8,+pw: U32,+pa: Nat,+pl: U32,+pp: Grid.Position,+km: PiecesModel.Model<Tile.Tile8>,+ks: List<&2,U32>,+kf: Tile.Tile8,+kw: U32,+ka: Nat,+kl: U32,+kp: Grid.Position,
  +prediction_contract: Public.Contract(~Prediction,~prediction_value,prediction,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp}),+returned_contract: Public.Contract(~Target,~target_value,target,shape,returned),
  +kept_contract: Public.Contract(~Target,~target_value,target,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,km),ks,kf,Tensors.Usable{},kw},ka,kl,kp})) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_kept(reduction,True{},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},(returned,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,km),ks,kf,Tensors.Usable{},kw},ka,kl,kp}))):
  +contract = Binary.same(~Prediction,~prediction_value,~Target,~target_value,~F32.sub,prediction,target,"mse_loss",1.0,limits,default_capacity,shape,nonempty,rank_room,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},prediction_contract,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,km),ks,kf,Tensors.Usable{},kw},ka,kl,kp},kept_contract)
  measured(~Prediction,~Target,~prediction_value,~target_value,prediction,target,reduction,shape,limits,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,returned,F32Tensor.binary(~F32.sub,"mse_loss",1.0,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,km),ks,kf,Tensors.Usable{},kw},ka,kl,kp}),Public.packed(~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},shape,F32Tensor.binary(~F32.sub,"mse_loss",1.0,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,km),ks,kf,Tensors.Usable{},kw},ka,kl,kp}),contract),contract,returned_contract)

def kept_packed(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+reduction: Loss.Reduction,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},-returned: Loss.Tensor(),
  +pm: PiecesModel.Model<Tile.Tile8>,+ps: List<&2,U32>,+pf: Tile.Tile8,+pw: U32,+pa: Nat,+pl: U32,+pp: Grid.Position,-kept: Loss.Tensor(),packed: Public.Packed<kept>,
  +prediction_contract: Public.Contract(~Prediction,~prediction_value,prediction,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp}),+returned_contract: Public.Contract(~Target,~target_value,target,shape,returned),
  +kept_contract: Public.Contract(~Target,~target_value,target,shape,kept)) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_kept(reduction,True{},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},(returned,kept))):
  Public.Packed{+km,+ks,+kf,+kw,+ka,+kl,+kp,+equation} = packed
  %equation : Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_kept(reduction,True{},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},(returned,_)))
  kept_model(~Prediction,~Target,~prediction_value,~target_value,prediction,target,reduction,shape,limits,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,returned,pm,ps,pf,pw,pa,pl,pp,km,ks,kf,kw,ka,kl,kp,prediction_contract,returned_contract,
    %Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,km),ks,kf,Tensors.Usable{},kw},ka,kl,kp},kept,equation) : Public.Contract(~Target,~target_value,target,shape,_) kept_contract)
# The target is copied: one copy returns, the other meets the prediction.
def selected(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+reduction: Loss.Reduction,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},same: Bool,+target_shape: List<&2,U32>,
  +pm: PiecesModel.Model<Tile.Tile8>,+ps: List<&2,U32>,+pf: Tile.Tile8,+pw: U32,+pa: Nat,+pl: U32,+pp: Grid.Position,+classified: {Shapes.same_extents(shape,target_shape) == same : Bool},
  +prediction_contract: Public.Contract(~Prediction,~prediction_value,prediction,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp}),pair: Loss.Tensor() & Loss.Tensor(),+copies: Copy.Copies(~Target,~target_value,target,target_shape,pair),
  +matched: {Shapes.same_extents(shape,target_shape) == True{} : Bool}) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_kept(reduction,same,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},pair)):
  match same:
    case True{}:
      (returned,kept) = pair
      KernelGemm.Both{+returned_contract,+kept_contract} = copies
      +to_shape = Equal.sym(List<&2,U32>,shape,target_shape,ShapeLaws.same_extents(shape,target_shape,classified))
      kept_packed(~Prediction,~Target,~prediction_value,~target_value,prediction,target,reduction,shape,limits,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,returned,pm,ps,pf,pw,pa,pl,pp,kept,
        Public.packed(~Target,~target_value,target,shape,kept,%to_shape : Public.Contract(~Target,~target_value,target,_,kept) kept_contract),
        prediction_contract,%to_shape : Public.Contract(~Target,~target_value,target,_,returned) returned_contract,
        %to_shape : Public.Contract(~Target,~target_value,target,_,kept) kept_contract)
    case False{}:
      (returned,kept) = pair
      Empty.absurd(Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_kept(reduction,False{},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},(returned,kept))),
        Words.false_true(Equal.trans(Bool,False{},Shapes.same_extents(shape,target_shape),True{},Equal.sym(Bool,Shapes.same_extents(shape,target_shape),False{},classified),matched)))
def mse_model(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+reduction: Loss.Reduction,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},+target_shape: List<&2,U32>,
  +pm: PiecesModel.Model<Tile.Tile8>,+ps: List<&2,U32>,+pf: Tile.Tile8,+pw: U32,+pa: Nat,+pl: U32,+pp: Grid.Position,+tm: PiecesModel.Model<Tile.Tile8>,+ts: List<&2,U32>,+tf: Tile.Tile8,+tw: U32,+ta: Nat,+tl: U32,+tp: Grid.Position,
  +prediction_contract: Public.Contract(~Prediction,~prediction_value,prediction,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp}),+target_contract: Public.Contract(~Target,~target_value,target,target_shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,Tensors.Usable{},tw},ta,tl,tp}),
  +matched: {Shapes.same_extents(shape,target_shape) == True{} : Bool}) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,Tensors.Usable{},tw},ta,tl,tp},reduction)):
  KernelGemm.Both{+prediction_global,Public.Laid{+prediction_inside,+prediction_shaped,+prediction_restored,+prediction_counted,+prediction_pinned}} = prediction_contract
  KernelGemm.Both{+target_global,Public.Laid{+target_inside,+target_shaped,+target_restored,+target_counted,+target_pinned}} = target_contract
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(ps,pa,pl,pp),shape,prediction_restored) : Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_kept(reduction,Shapes.same_extents(_,Tiled.logical_shape(ts,ta,tl,tp)),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},F32Tensor.copy(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,Tensors.Usable{},tw},ta,tl,tp})))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(ts,ta,tl,tp),target_shape,target_restored) : Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_kept(reduction,Shapes.same_extents(shape,_),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},F32Tensor.copy(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,Tensors.Usable{},tw},ta,tl,tp})))
  selected(~Prediction,~Target,~prediction_value,~target_value,prediction,target,reduction,shape,limits,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,Shapes.same_extents(shape,target_shape),target_shape,pm,ps,pf,pw,pa,pl,pp,{==},prediction_contract,F32Tensor.copy(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,Tensors.Usable{},tw},ta,tl,tp}),
    Copy.tiled(~Target,~target_value,target,target_shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,Tensors.Usable{},tw},ta,tl,tp},target_contract),
    matched)

def mse_packed(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+reduction: Loss.Reduction,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},+target_shape: List<&2,U32>,
  -prediction_tiled: Loss.Tensor(),prediction_packed: Public.Packed<prediction_tiled>,-target_tiled: Loss.Tensor(),target_packed: Public.Packed<target_tiled>,
  +prediction_contract: Public.Contract(~Prediction,~prediction_value,prediction,shape,prediction_tiled),+target_contract: Public.Contract(~Target,~target_value,target,target_shape,target_tiled),
  +matched: {Shapes.same_extents(shape,target_shape) == True{} : Bool}) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse(prediction_tiled,target_tiled,reduction)):
  Public.Packed{+pm,+ps,+pf,+pw,+pa,+pl,+pp,+prediction_equation} = prediction_packed
  Public.Packed{+tm,+ts,+tf,+tw,+ta,+tl,+tp,+target_equation} = target_packed
  %prediction_equation : Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse(_,target_tiled,reduction))
  %target_equation : Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},_,reduction))
  mse_model(~Prediction,~Target,~prediction_value,~target_value,prediction,target,reduction,shape,limits,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,target_shape,pm,ps,pf,pw,pa,pl,pp,tm,ts,tf,tw,ta,tl,tp,
    %Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pm),ps,pf,Tensors.Usable{},pw},pa,pl,pp},prediction_tiled,prediction_equation) : Public.Contract(~Prediction,~prediction_value,prediction,shape,_) prediction_contract,
    %Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,Tensors.Usable{},tw},ta,tl,tp},target_tiled,target_equation) : Public.Contract(~Target,~target_value,target,target_shape,_) target_contract,matched)

# The loss and gradient of a prediction against a target, both tiled tensors
# of the contract, of the same shape.
def mse(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+reduction: Loss.Reduction,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},+target_shape: List<&2,U32>,
  prediction_tiled: Loss.Tensor(),+prediction_contract: Public.Contract(~Prediction,~prediction_value,prediction,shape,prediction_tiled),
  target_tiled: Loss.Tensor(),+target_contract: Public.Contract(~Target,~target_value,target,target_shape,target_tiled),
  +matched: {Shapes.same_extents(shape,target_shape) == True{} : Bool}) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse(prediction_tiled,target_tiled,reduction)):
  mse_packed(~Prediction,~Target,~prediction_value,~target_value,prediction,target,reduction,shape,limits,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,target_shape,
    prediction_tiled,Public.packed(~Prediction,~prediction_value,prediction,shape,prediction_tiled,prediction_contract),
    target_tiled,Public.packed(~Target,~target_value,target,target_shape,target_tiled,target_contract),
    prediction_contract,target_contract,matched)
