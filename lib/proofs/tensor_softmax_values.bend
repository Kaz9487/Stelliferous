# Softmax along one axis of a tensor of a nonempty shape: each line along the
# axis less its maximum, folded in increasing index from -inf, then the
# exponentials of those differences divided by their sum along the line, the
# sum of math_sum.bend. The maximum and the sum of a line reach every element
# of it through the view that stretches the shape with the axis of extent 1.
import Base
import ../tensor.bend as Tensors
import ../tensor_f32.bend as F32Tensor
import ../tensor_view.bend as Views
import ../tensor_access.bend as Access
import ../tensor_grid.bend as Grid
import ../tensor_layout.bend as Layout
import ../tensor_shape.bend as Shapes
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_map.bend as TiledMap
import ../tensor_tiled_reduce.bend as TiledReduce
import ../math_choice.bend as Choice
import ../math_fp32.bend as Arithmetic
import ../math_sum.bend as Sum
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_contract.bend as TensorContract
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_combine_leaf.bend as CombineLeaf
import ./tensor_view_gather.bend as ViewGather
import ./tensor_view_part.bend as ViewPart
import ./tensor_grid_box.bend as Box
import ./tensor_reduce_values.bend as Reduce
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_unit_broadcast.bend as UnitBroadcast
import ./tensor_unit_line.bend as UnitLine
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_map.bend as TiledMaps
import ./tensor_tiled_reduce.bend as ReduceLaws
import ./tensor_tiled_reshape.bend as TiledReshape
import ./tensor_binary_values.bend as Binary
import ./tensor_copy_contract.bend as Copy

# The result at i is exp(x[i] - m) / s, with m the maximum of the line of i
# and s the sum of the exponentials of that line.
def Softmax(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,result: Tiled.Tiled) -> Data:
  Public.Contract(~CombineLeaf.Context<ViewGather.Viewing<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>,ViewGather.Viewing<Reduce.Lines<Sum.Lanes,Reduce.Ordered<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>>>>,~CombineLeaf.value(~F32,~ViewGather.Viewing<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>,~ViewGather.Viewing<Reduce.Lines<Sum.Lanes,Reduce.Ordered<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>>>,~F32.div,~ViewGather.viewed(~F32,~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)))))),~ViewGather.viewed(~F32,~Reduce.Lines<Sum.Lanes,Reduce.Ordered<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>>,~Reduce.line_value(~F32,~Sum.Lanes,~Sum.step,~Sum.total,~Reduce.Ordered<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>,~Reduce.through(~F32,~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))))))))),CombineLeaf.Context{ViewGather.Viewing{PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{Sum.zero(),Reduce.Ordered{PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}},shape,result)

# The axis is a valid axis of the reduction.
def axis_below(+shape: List<&2,U32>,+axis: U32,+inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)) == True{} : Bool}) ->
  {Layout.all_below([axis],List.length(&2,U32,shape)) == True{} : Bool}:
  %Equal.sym(Bool,Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)),True{},inside) : {(_ && True{}) == True{} : Bool}
  {==}

# ---- The exponentials divided by their sum ----

def divided(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)) == True{} : Bool},
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+own_stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +unit_stretched: {Tensors.stretched(Tensors.view_of(Layout.with_extent(shape,U32.to_nat(axis),1)),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +em1: PiecesModel.Model<Tile.Tile8>,+em2: PiecesModel.Model<Tile.Tile8>,+es: List<&2,U32>,+ef: Tile.Tile8,+ew: U32,+ea: Nat,+el: U32,+ep: Grid.Position,+e1_contract: Public.Contract(~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))))),PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em1),es,ef,Tensors.Usable{},ew},ea,el,ep}),+e2_contract: Public.Contract(~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))))),PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em2),es,ef,Tensors.Usable{},ew},ea,el,ep})) -> Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax_divided(axis,Layout.with_extent(shape,U32.to_nat(axis),1),(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em1),es,ef,Tensors.Usable{},ew},ea,el,ep},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em2),es,ef,Tensors.Usable{},ew},ea,el,ep}))):
  KernelGemm.Both{+e1_global,Public.Laid{+e1_inside,+e1_shaped,+e1_restored,+e1_counted,+e1_pinned}} = e1_contract
  +nonzero = CountCompletion.positive_not_zero(shape,nonempty)
  +summed = ReduceLaws.reduce_axes(~Sum.Lanes,~Sum.step,~Sum.total,~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))))),PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},limits,"softmax",6.0,Sum.zero(),[axis],shape,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em2),es,ef,Tensors.Usable{},ew},ea,el,ep},e2_contract,{==},axis_below(shape,axis,inside),default_capacity,rank_room,nonempty,cap,cap_ok,count_room)
  +reshaped = UnitLine.kept_reshaped(~Reduce.Lines<Sum.Lanes,Reduce.Ordered<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>>,~Reduce.line_value(~F32,~Sum.Lanes,~Sum.step,~Sum.total,~Reduce.Ordered<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>,~Reduce.through(~F32,~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))))))),Reduce.Lines{Sum.zero(),Reduce.Ordered{PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},limits,shape,U32.to_nat(axis),TiledReduce.reduce_axes(~Sum.Lanes,~Sum.step,~Sum.total,"softmax",6.0,Sum.zero(),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em2),es,ef,Tensors.Usable{},ew},ea,el,ep},[axis]),default_capacity,rank_room,inside,nonempty,summed)
  Binary.broadcast(~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))))),~Reduce.Lines<Sum.Lanes,Reduce.Ordered<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>>,~Reduce.line_value(~F32,~Sum.Lanes,~Sum.step,~Sum.total,~Reduce.Ordered<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>,~Reduce.through(~F32,~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))))))),~F32.div,PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},Reduce.Lines{Sum.zero(),Reduce.Ordered{PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},"div",1.0,limits,default_capacity,shape,Layout.with_extent(shape,U32.to_nat(axis),1),shape,
    laxes,loffset,raxes,roffset,UnitBroadcast.broadcast(shape,U32.to_nat(axis)),own_stretched,unit_stretched,nonzero,UnitLine.unit_nonzero(shape,U32.to_nat(axis),inside,nonempty),nonzero,rank_room,UnitLine.unit_room(limits,shape,U32.to_nat(axis),rank_room),rank_room,e1_counted,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em1),es,ef,Tensors.Usable{},ew},ea,el,ep},e1_contract,F32Tensor.reshape(TiledReduce.reduce_axes(~Sum.Lanes,~Sum.step,~Sum.total,"softmax",6.0,Sum.zero(),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em2),es,ef,Tensors.Usable{},ew},ea,el,ep},[axis]),Layout.with_extent(shape,U32.to_nat(axis),1)),reshaped)

def exponentials_copies(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)) == True{} : Bool},
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+own_stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +unit_stretched: {Tensors.stretched(Tensors.view_of(Layout.with_extent(shape,U32.to_nat(axis),1)),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +em: PiecesModel.Model<Tile.Tile8>,+es: List<&2,U32>,+ef: Tile.Tile8,+ew: U32,+ea: Nat,+el: U32,+ep: Grid.Position,exponentials: Copy.Copied<Tile.Tile8,em>,+exponentials_contract: Public.Contract(~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))))),PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em),es,ef,Tensors.Usable{},ew},ea,el,ep})) ->
  Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax_divided(axis,Layout.with_extent(shape,U32.to_nat(axis),1),F32Tensor.copy(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em),es,ef,Tensors.Usable{},ew},ea,el,ep}))):
  Copy.Copied{+em1,+em2,+exponentials_equation,+exponentials_first_content,+exponentials_second_content} = exponentials
  KernelGemm.Both{+exponentials_global,+exponentials_laid} = exponentials_contract
  +exponentials_pair = Copy.copied_equation(~Tile.Tile8,32.0,em,es,ef,ew,em1,em2,exponentials_equation)
  +exponentials_before = {Representation.Lifted{PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},Public.geometry_of(shape,ea,ep)} : Representation.Lifted<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>}
  %Equal.sym(Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>,Tensors.copied(~Tile.Tile8,32.0,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em),es,ef,Tensors.Usable{},ew}),(Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em1),es,ef,Tensors.Usable{},ew},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em2),es,ef,Tensors.Usable{},ew}),exponentials_pair) :
    Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax_divided(axis,Layout.with_extent(shape,U32.to_nat(axis),1),Tiled.copied(ea,el,ep,_)))
  divided(~Logical,~original,logical,limits,shape,axis,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,inside,laxes,loffset,raxes,roffset,own_stretched,unit_stretched,em1,em2,es,ef,ew,ea,el,ep,
    KernelGemm.Both{Copy.kept(~Tile.Tile8,~Representation.Lifted<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>,~Representation.lifted(~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)))))),exponentials_before,em,em1,es,ef,ew,exponentials_first_content,exponentials_global),exponentials_laid},
    KernelGemm.Both{Copy.kept(~Tile.Tile8,~Representation.Lifted<PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>>,~Representation.lifted(~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)))))),exponentials_before,em,em2,es,ef,ew,exponentials_second_content,exponentials_global),exponentials_laid})

def exponentials_packed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)) == True{} : Bool},
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+own_stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +unit_stretched: {Tensors.stretched(Tensors.view_of(Layout.with_extent(shape,U32.to_nat(axis),1)),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  -e_tiled: Tiled.Tiled,e_packed: Public.Packed<e_tiled>,+e_contract: Public.Contract(~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))))),PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},shape,e_tiled)) -> Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax_divided(axis,Layout.with_extent(shape,U32.to_nat(axis),1),F32Tensor.copy(e_tiled))):
  Public.Packed{+em,+es,+ef,+ew,+ea,+el,+ep,+e_equation} = e_packed
  %e_equation : Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax_divided(axis,Layout.with_extent(shape,U32.to_nat(axis),1),F32Tensor.copy(_)))
  exponentials_copies(~Logical,~original,logical,limits,shape,axis,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,inside,laxes,loffset,raxes,roffset,own_stretched,unit_stretched,em,es,ef,ew,ea,el,ep,Copy.copied(~Tile.Tile8,em),
    %Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em),es,ef,Tensors.Usable{},ew},ea,el,ep},e_tiled,e_equation) : Public.Contract(~PiecesMap.Context<Unit,CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>>,~PiecesMap.value(~F32,~Unit,~CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>,~Access.apply_element(~F32,~Arithmetic.exp),~CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~F32.sub,~ViewGather.viewed(~F32,~Logical,~original),~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32.max,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))))),PiecesMap.Context{Unit{},CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},raxes,roffset,[]}}},shape,_) e_contract)

# ---- The lines less their maximum and their exponentials ----

def Centering(~Logical: Data) -> Data:
  CombineLeaf.Context<ViewGather.Viewing<Logical>,ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>>

def centered_value(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32 -> F32,~maximum: F32 -> F32 -> F32) -> Centering(~Logical) -> U32 -> F32:
  CombineLeaf.value(~F32,~ViewGather.Viewing<Logical>,~ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>,~operation,~ViewGather.viewed(~F32,~Logical,~original),
    ~ViewGather.viewed(~F32,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~maximum,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))))

def ExpMap(~Logical: Data) -> Data:
  PiecesMap.Context<Unit,Centering(~Logical)>

def exponentiated(~Logical: Data,~original: Logical -> U32 -> F32) -> ExpMap(~Logical) -> U32 -> F32:
  PiecesMap.value(~F32,~Unit,~Centering(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~centered_value(~Logical,~original,~F32.sub,~F32.max))

def maximum_selected(flag: Bool,+left: F32,+right: F32) ->
  {Choice.chosen(~F32,flag,right,left) == Bool.pick(F32,flag,right,left) : F32}:
  match flag:
    case False{}: {==}
    case True{}: {==}

def maximum_same(+left: F32,+right: F32) -> {F32Tensor.softmax_maximum(left,right) == F32.max(left,right) : F32}:
  match left right:
    case F32{+x} F32{+y}: maximum_selected(F32.is_lt(F32{x},F32{y}),F32{x},F32{y})

def maximum_fold(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,count: Nat,+position: Nat,+state: F32) ->
  {Reduce.line_fold(~F32,~F32,~F32Tensor.softmax_maximum,~Logical,~original,logical,count,position,state) ==
    Reduce.line_fold(~F32,~F32,~F32.max,~Logical,~original,logical,count,position,state) : F32}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +value = original(logical,U32.from_nat(position))
      +plain = F32.max(state,value)
      +specialized = F32Tensor.softmax_maximum(state,value)
      Equal.trans(F32,Reduce.line_fold(~F32,~F32,~F32Tensor.softmax_maximum,~Logical,~original,logical,rest,1n+position,specialized),
        Reduce.line_fold(~F32,~F32,~F32Tensor.softmax_maximum,~Logical,~original,logical,rest,1n+position,plain),
        Reduce.line_fold(~F32,~F32,~F32.max,~Logical,~original,logical,rest,1n+position,plain),
        Equal.cong(F32,F32,maximum => Reduce.line_fold(~F32,~F32,~F32Tensor.softmax_maximum,~Logical,~original,logical,rest,1n+position,maximum),
          specialized,plain,maximum_same(state,value)),
        maximum_fold(~Logical,~original,logical,rest,1n+position,plain))

def ExpAgreement(~Logical: Data,values: Centering(~Logical),mapped: ExpMap(~Logical)) -> Data:
  {PiecesMap.Context{Unit{},values} == mapped : ExpMap(~Logical)}

def exp_agrees(~Logical: Data,~original: Logical -> U32 -> F32,+before: Centering(~Logical),+after: ExpMap(~Logical),
  +related: ExpAgreement(~Logical,before,after),+index: U32) ->
  {centered_value(~Logical,~original,~F32Tensor.exp_centered,~F32Tensor.softmax_maximum)(before,index) == exponentiated(~Logical,~original)(after,index) : F32}:
  match before:
    case CombineLeaf.Context{+left,ViewGather.Viewing{Reduce.Lines{+initial,+logical,+length},+axes,+offset,+flags}}:
      +right = {ViewGather.Viewing{Reduce.Lines{initial,logical,length},axes,offset,flags} : ViewGather.Viewing<Reduce.Lines<F32,Reduce.Ordered<Logical>>>}
      %related : {centered_value(~Logical,~original,~F32Tensor.exp_centered,~F32Tensor.softmax_maximum)(CombineLeaf.Context{left,right},index) == exponentiated(~Logical,~original)(_,index) : F32}
      +line = (Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,flags)),U32.to_nat(index)) : U32)
      Equal.cong(F32,F32,maximum => F32Tensor.exp_centered(ViewGather.viewed(~F32,~Logical,~original,left,index),maximum),
        Reduce.line_fold(~F32,~F32,~F32Tensor.softmax_maximum,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),logical,length,Nat.mul(U32.to_nat(line),length),initial),
        Reduce.line_fold(~F32,~F32,~F32.max,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),logical,length,Nat.mul(U32.to_nat(line),length),initial),
        maximum_fold(~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),logical,length,Nat.mul(U32.to_nat(line),length),initial))

# The two copies of the input: the maximum of the lines of the second, and
# the exponential of each difference written by the binary pass.
def centered(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)) == True{} : Bool},
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+own_stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +unit_stretched: {Tensors.stretched(Tensors.view_of(Layout.with_extent(shape,U32.to_nat(axis),1)),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +om1: PiecesModel.Model<Tile.Tile8>,+om2: PiecesModel.Model<Tile.Tile8>,+os: List<&2,U32>,+of: Tile.Tile8,+ow: U32,+oa: Nat,+ol: U32,+op: Grid.Position,+o1_contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om1),os,of,Tensors.Usable{},ow},oa,ol,op}),+o2_contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om2),os,of,Tensors.Usable{},ow},oa,ol,op})) ->
  Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax_divided(axis,Layout.with_extent(shape,U32.to_nat(axis),1),F32Tensor.copy(F32Tensor.softmax_exponentials(axis,Layout.with_extent(shape,U32.to_nat(axis),1),(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om1),os,of,Tensors.Usable{},ow},oa,ol,op},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om2),os,of,Tensors.Usable{},ow},oa,ol,op}))))):
  KernelGemm.Both{+o1_global,Public.Laid{+o1_inside,+o1_shaped,+o1_restored,+o1_counted,+o1_pinned}} = o1_contract
  +nonzero = CountCompletion.positive_not_zero(shape,nonempty)
  +maximum = ReduceLaws.reduce_axes(~F32,~F32Tensor.softmax_maximum,~Tensors.same(~F32),~Logical,~original,logical,limits,"softmax",1.0,F32.div(F32.neg(1.0),0.0),[axis],shape,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om2),os,of,Tensors.Usable{},ow},oa,ol,op},o2_contract,{==},axis_below(shape,axis,inside),default_capacity,rank_room,nonempty,cap,cap_ok,count_room)
  +reshaped = UnitLine.kept_reshaped(~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32Tensor.softmax_maximum,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)),Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))},limits,shape,U32.to_nat(axis),TiledReduce.reduce_axes(~F32,~F32Tensor.softmax_maximum,~Tensors.same(~F32),"softmax",1.0,F32.div(F32.neg(1.0),0.0),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om2),os,of,Tensors.Usable{},ow},oa,ol,op},[axis]),default_capacity,rank_room,inside,nonempty,maximum)
  +maxima = {Reduce.Lines{F32.div(F32.neg(1.0),0.0),Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Reduce.kept_axes(shape,[axis]),Reduce.reduced_axes(shape,[axis])))},
    ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[axis])))} : Reduce.Lines<F32,Reduce.Ordered<Logical>>}
  +views = {CombineLeaf.Context{ViewGather.Viewing{logical,laxes,loffset,[]},ViewGather.Viewing{maxima,raxes,roffset,[]}} : Centering(~Logical)}
  +unit = Layout.with_extent(shape,U32.to_nat(axis),1)
  +fused = Binary.broadcast(~Logical,~original,~Reduce.Lines<F32,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~F32,~F32Tensor.softmax_maximum,~Tensors.same(~F32),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)),
    ~F32Tensor.exp_centered,logical,maxima,"sub",5.5,limits,default_capacity,shape,unit,shape,laxes,loffset,raxes,roffset,UnitBroadcast.broadcast(shape,U32.to_nat(axis)),own_stretched,unit_stretched,
    nonzero,UnitLine.unit_nonzero(shape,U32.to_nat(axis),inside,nonempty),nonzero,rank_room,UnitLine.unit_room(limits,shape,U32.to_nat(axis),rank_room),rank_room,o1_counted,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om1),os,of,Tensors.Usable{},ow},oa,ol,op},o1_contract,
    F32Tensor.reshape(TiledReduce.reduce_axes(~F32,~F32Tensor.softmax_maximum,~Tensors.same(~F32),"softmax",1.0,F32.div(F32.neg(1.0),0.0),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om2),os,of,Tensors.Usable{},ow},oa,ol,op},[axis]),unit),reshaped)
  +exp_context = {PiecesMap.Context{Unit{},views} : ExpMap(~Logical)}
  +exp_contract = Public.reread(~Centering(~Logical),~ExpMap(~Logical),~centered_value(~Logical,~original,~F32Tensor.exp_centered,~F32Tensor.softmax_maximum),
    ~exponentiated(~Logical,~original),~ExpAgreement(~Logical),~exp_agrees(~Logical,~original),views,exp_context,{==},shape,
    F32Tensor.softmax_exponentials(axis,unit,(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om1),os,of,Tensors.Usable{},ow},oa,ol,op},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om2),os,of,Tensors.Usable{},ow},oa,ol,op})),fused)
  +exp_packed = Public.packed(~ExpMap(~Logical),~exponentiated(~Logical,~original),exp_context,shape,
    F32Tensor.softmax_exponentials(axis,unit,(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om1),os,of,Tensors.Usable{},ow},oa,ol,op},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om2),os,of,Tensors.Usable{},ow},oa,ol,op})),exp_contract)
  exponentials_packed(~Logical,~original,logical,limits,shape,axis,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,inside,laxes,loffset,raxes,roffset,own_stretched,unit_stretched,
    F32Tensor.softmax_exponentials(axis,unit,(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om1),os,of,Tensors.Usable{},ow},oa,ol,op},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om2),os,of,Tensors.Usable{},ow},oa,ol,op})),exp_packed,exp_contract)


def input_copies(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)) == True{} : Bool},
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+own_stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +unit_stretched: {Tensors.stretched(Tensors.view_of(Layout.with_extent(shape,U32.to_nat(axis),1)),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +om: PiecesModel.Model<Tile.Tile8>,+os: List<&2,U32>,+of: Tile.Tile8,+ow: U32,+oa: Nat,+ol: U32,+op: Grid.Position,input: Copy.Copied<Tile.Tile8,om>,+input_contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om),os,of,Tensors.Usable{},ow},oa,ol,op})) ->
  Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax_divided(axis,Layout.with_extent(shape,U32.to_nat(axis),1),F32Tensor.copy(F32Tensor.softmax_exponentials(axis,Layout.with_extent(shape,U32.to_nat(axis),1),F32Tensor.copy(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om),os,of,Tensors.Usable{},ow},oa,ol,op}))))):
  Copy.Copied{+om1,+om2,+input_equation,+input_first_content,+input_second_content} = input
  KernelGemm.Both{+input_global,+input_laid} = input_contract
  +input_pair = Copy.copied_equation(~Tile.Tile8,32.0,om,os,of,ow,om1,om2,input_equation)
  +input_before = {Representation.Lifted{logical,Public.geometry_of(shape,oa,op)} : Representation.Lifted<Logical>}
  %Equal.sym(Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>,Tensors.copied(~Tile.Tile8,32.0,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om),os,of,Tensors.Usable{},ow}),(Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om1),os,of,Tensors.Usable{},ow},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om2),os,of,Tensors.Usable{},ow}),input_pair) :
    Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax_divided(axis,Layout.with_extent(shape,U32.to_nat(axis),1),F32Tensor.copy(F32Tensor.softmax_exponentials(axis,Layout.with_extent(shape,U32.to_nat(axis),1),Tiled.copied(oa,ol,op,_)))))
  centered(~Logical,~original,logical,limits,shape,axis,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,inside,laxes,loffset,raxes,roffset,own_stretched,unit_stretched,om1,om2,os,of,ow,oa,ol,op,
    KernelGemm.Both{Copy.kept(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),input_before,om,om1,os,of,ow,input_first_content,input_global),input_laid},
    KernelGemm.Both{Copy.kept(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),input_before,om,om2,os,of,ow,input_second_content,input_global),input_laid})

# ---- The input ----

def softmax_model(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)) == True{} : Bool},
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+own_stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +unit_stretched: {Tensors.stretched(Tensors.view_of(Layout.with_extent(shape,U32.to_nat(axis),1)),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +om: PiecesModel.Model<Tile.Tile8>,+os: List<&2,U32>,+of: Tile.Tile8,+ow: U32,+oa: Nat,+ol: U32,+op: Grid.Position,+contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om),os,of,Tensors.Usable{},ow},oa,ol,op})) -> Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om),os,of,Tensors.Usable{},ow},oa,ol,op},axis)):
  KernelGemm.Both{+global,Public.Laid{+within,+shaped,+restored,+counted,+pinned}} = contract
  +to_shape = Equal.cong(List<&2,U32>,Tiled.Tiled,extents => F32Tensor.softmax_valid(Tensors.usable(Tensors.Usable{}) && Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,extents)),Shapes.has_zero(extents),extents,axis,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om),os,of,Tensors.Usable{},ow},oa,ol,op}),Tiled.logical_shape(os,oa,ol,op),shape,restored)
  +empty = CountCompletion.positive_not_zero(shape,nonempty)
  %Equal.sym(Tiled.Tiled,F32Tensor.softmax_valid(Tensors.usable(Tensors.Usable{}) && Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,Tiled.logical_shape(os,oa,ol,op))),Shapes.has_zero(Tiled.logical_shape(os,oa,ol,op)),Tiled.logical_shape(os,oa,ol,op),axis,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om),os,of,Tensors.Usable{},ow},oa,ol,op}),F32Tensor.softmax_valid(Tensors.usable(Tensors.Usable{}) && Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)),Shapes.has_zero(shape),shape,axis,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om),os,of,Tensors.Usable{},ow},oa,ol,op}),to_shape) : Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,_)
  %Equal.sym(Bool,Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)),True{},inside) : Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax_valid(Tensors.usable(Tensors.Usable{}) && _,Shapes.has_zero(shape),shape,axis,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om),os,of,Tensors.Usable{},ow},oa,ol,op}))
  %Equal.sym(Bool,Shapes.has_zero(shape),False{},empty) : Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax_valid(Tensors.usable(Tensors.Usable{}) && True{},_,shape,axis,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om),os,of,Tensors.Usable{},ow},oa,ol,op}))
  input_copies(~Logical,~original,logical,limits,shape,axis,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,inside,laxes,loffset,raxes,roffset,own_stretched,unit_stretched,om,os,of,ow,oa,ol,op,Copy.copied(~Tile.Tile8,om),contract)

def softmax_packed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)) == True{} : Bool},
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+own_stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +unit_stretched: {Tensors.stretched(Tensors.view_of(Layout.with_extent(shape,U32.to_nat(axis),1)),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  -tiled: Tiled.Tiled,packed: Public.Packed<tiled>,+contract: Public.Contract(~Logical,~original,logical,shape,tiled)) ->
  Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax(tiled,axis)):
  Public.Packed{+om,+os,+of,+ow,+oa,+ol,+op,+equation} = packed
  %equation : Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax(_,axis))
  softmax_model(~Logical,~original,logical,limits,shape,axis,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,inside,laxes,loffset,raxes,roffset,own_stretched,unit_stretched,om,os,of,ow,oa,ol,op,
    %Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,om),os,of,Tensors.Usable{},ow},oa,ol,op},tiled,equation) : Public.Contract(~Logical,~original,logical,shape,_) contract)

# A tensor of its contract, of a nonempty shape with the axis inside it, and
# the views that read the shape and the shape with the axis of extent 1
# stretched to the shape: the softmax holds Softmax.
def softmax(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)) == True{} : Bool},
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+own_stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +unit_stretched: {Tensors.stretched(Tensors.view_of(Layout.with_extent(shape,U32.to_nat(axis),1)),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  tiled: Tiled.Tiled,+contract: Public.Contract(~Logical,~original,logical,shape,tiled)) -> Softmax(~Logical,~original,logical,shape,axis,laxes,loffset,raxes,roffset,F32Tensor.softmax(tiled,axis)):
  softmax_packed(~Logical,~original,logical,limits,shape,axis,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,inside,laxes,loffset,raxes,roffset,own_stretched,unit_stretched,tiled,Public.packed(~Logical,~original,logical,shape,tiled,contract),contract)
