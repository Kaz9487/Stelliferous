# Two tiled operands of one shape combine cell by cell, the lanes of the right
# on the axis of the left, so the result holds the operation of the two values
# at every flat index. Operands of other extents broadcast: the result holds at
# every index of the common extents the operation of the two values the
# operands read through their views stretched to those extents.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor_f32.bend as F
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_broadcast.bend as TiledBroadcast
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_combine_leaf.bend as Leaf
import ./tensor_view_gather.bend as ViewGather
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_storage_count.bend as StorageCount
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_broadcast.bend as Broadcast
import ./tensor_unit_broadcast.bend as UnitBroadcast

# ---- The call on packed operands ----

# Packed operands combine as their extents decide.
def packed_call(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~Held: @tiled: Tiled.Tiled -> Data,~operation: F32 -> F32 -> F32,
  +left: Left,+right: Right,+name: String,+operations: F32,+ls: List<&2,U32>,+rs: List<&2,U32>,
  +lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lo: Nat,+ll: U32,+lp: Grid.Position,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +left_contract: Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}),
  +usable: Held(F.binary_usable(~operation,name,operations,Shapes.same_extents(ls,rs),ls,rs,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}))) ->
  Held(F.binary(~operation,name,operations,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})):
  KernelGemm.Both{+left_global,Public.Laid{+left_inside,+left_shaped,+left_restored,+left_counted,+left_pinned}} = left_contract
  KernelGemm.Both{+right_global,Public.Laid{+right_inside,+right_shaped,+right_restored,+right_counted,+right_pinned}} = right_contract
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(lc,lo,ll,lp),ls,left_restored) :
    Held(F.binary_usable(~operation,name,operations,Shapes.same_extents(_,Tiled.logical_shape(rc,ro,rl,rp)),_,Tiled.logical_shape(rc,ro,rl,rp),
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(rc,ro,rl,rp),rs,right_restored) :
    Held(F.binary_usable(~operation,name,operations,Shapes.same_extents(ls,_),ls,_,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}))
  usable

# ---- One shape ----

def same_packed(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+shape: List<&2,U32>,
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lo: Nat,+ll: U32,+lp: Grid.Position,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +left_contract: Public.Contract(~Left,~left_value,left,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp}),
  +right_contract: Public.Contract(~Right,~right_value,right,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})) ->
  Public.Contract(~Leaf.Context<Left,Right>,~Leaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left,right},shape,
    F.binary(~operation,name,operations,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})):
  packed_call(~Left,~left_value,~Right,~right_value,
    ~(tiled => Public.Contract(~Leaf.Context<Left,Right>,~Leaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left,right},shape,tiled)),~operation,
    left,right,name,operations,shape,shape,lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,left_contract,right_contract,
    {%Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},ShapeLaws.same_extents_reflexive(shape)) :
        Public.Contract(~Leaf.Context<Left,Right>,~Leaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left,right},shape,
          F.binary_usable(~operation,name,operations,_,shape,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
            Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}))
      Broadcast.same(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,name,operations,
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},
        limits,default_capacity,nonempty,rank_room,left_contract,right_contract) :
      Public.Contract(~Leaf.Context<Left,Right>,~Leaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left,right},shape,
        F.binary_usable(~operation,name,operations,Shapes.same_extents(shape,shape),shape,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
          Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}))})

def same_described(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+shape: List<&2,U32>,
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  -left_tiled: Tiled.Tiled,+left_packed: Public.Packed<left_tiled>,-right_tiled: Tiled.Tiled,+right_packed: Public.Packed<right_tiled>,
  +left_contract: Public.Contract(~Left,~left_value,left,shape,left_tiled),+right_contract: Public.Contract(~Right,~right_value,right,shape,right_tiled)) ->
  Public.Contract(~Leaf.Context<Left,Right>,~Leaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left,right},shape,F.binary(~operation,name,operations,left_tiled,right_tiled)):
  Public.Packed{+lm,+lc,+lf,+lw,+lo,+ll,+lp,+left_equation} = left_packed
  Public.Packed{+rm,+rc,+rf,+rw,+ro,+rl,+rp,+right_equation} = right_packed
  %left_equation : Public.Contract(~Leaf.Context<Left,Right>,~Leaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left,right},shape,
    F.binary(~operation,name,operations,_,right_tiled))
  %right_equation : Public.Contract(~Leaf.Context<Left,Right>,~Leaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left,right},shape,
    F.binary(~operation,name,operations,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},_))
  same_packed(~Left,~left_value,~Right,~right_value,~operation,left,right,name,operations,limits,default_capacity,shape,nonempty,rank_room,lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},left_tiled,left_equation) :
      Public.Contract(~Left,~left_value,left,shape,_) left_contract :
      Public.Contract(~Left,~left_value,left,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp})},
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},right_tiled,right_equation) :
      Public.Contract(~Right,~right_value,right,shape,_) right_contract :
      Public.Contract(~Right,~right_value,right,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})})

# Two tiled operands of the contract and of one shape without a zero extent
# combine into a tensor of the shape that holds the operation of the two
# values at every flat index.
def same(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+shape: List<&2,U32>,
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  left_tiled: Tiled.Tiled,+left_contract: Public.Contract(~Left,~left_value,left,shape,left_tiled),
  right_tiled: Tiled.Tiled,+right_contract: Public.Contract(~Right,~right_value,right,shape,right_tiled)) ->
  Public.Contract(~Leaf.Context<Left,Right>,~Leaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left,right},shape,F.binary(~operation,name,operations,left_tiled,right_tiled)):
  same_described(~Left,~left_value,~Right,~right_value,~operation,left,right,name,operations,limits,default_capacity,shape,nonempty,rank_room,
    left_tiled,Public.packed(~Left,~left_value,left,shape,left_tiled,left_contract),right_tiled,Public.packed(~Right,~right_value,right,shape,right_tiled,right_contract),
    left_contract,right_contract)

# ---- Broadcasting ----

def extents_or(found: Maybe<&2,List<&2,U32>>,+default: List<&2,U32>) -> List<&2,U32>:
  match found:
    case Some{extents}: extents
    case None{}: default

# Equal found extents are the same extents.
def found_same(+left: List<&2,U32>,+right: List<&2,U32>,+same: {Some{left} == Some{right} : Maybe<&2,List<&2,U32>>}) -> {left == right : List<&2,U32>}:
  Equal.cong(Maybe<&2,List<&2,U32>>,List<&2,U32>,found => extents_or(found,left),Some{left},Some{right},same)

# Operands of one shape read through their own views.
def broadcast_same(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+shape: List<&2,U32>,
  +lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lo: Nat,+ll: U32,+lp: Grid.Position,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +left_stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +nonzero: {Shapes.has_zero(shape) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp}),
  +right_contract: Public.Contract(~Right,~right_value,right,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})) ->
  Broadcast.BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,
    TiledBroadcast.same(~operation,name,operations,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})):
  KernelGemm.Both{+left_global,Public.Laid{+left_inside,+left_shaped,+left_restored,+left_counted,+left_pinned}} = left_contract
  +count = Global.count(shape,left_counted)
  +exact = Global.whole_counted(shape,count,Global.accepted(shape,left_counted))
  +positive = CountCompletion.positive_elements(shape,nonzero)
  +room = StorageCount.counted_room(limits,shape,left_counted,default_capacity)
  Broadcast.own_right(~ViewGather.Viewing<Left>,~ViewGather.viewed(~F32,~Left,~left_value),~Right,~right_value,~operation,ViewGather.Viewing{left,laxes,loffset,[]},right,
    shape,raxes,roffset,count,exact,right_stretched,limits,positive,room,
    TiledBroadcast.same(~operation,name,operations,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}),
    Broadcast.own_left(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,count,exact,left_stretched,limits,positive,room,
      TiledBroadcast.same(~operation,name,operations,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}),
      Broadcast.same(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,name,operations,
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},
        limits,default_capacity,positive,rank_room,left_contract,right_contract)))

# Operands of one shape, or of extents that broadcast.
def broadcast_usable(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,same: Bool,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,
  +lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lo: Nat,+ll: U32,+lp: Grid.Position,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +shape: List<&2,U32>,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+classified: {Shapes.same_extents(ls,rs) == same : Bool},
  +common: {Shapes.broadcast(ls,rs) == Some{shape} : Maybe<&2,List<&2,U32>>},
  +left_stretched: {Tensors.stretched(Tensors.view_of(ls),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<shape>,
  +left_contract: Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})) ->
  Broadcast.BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,
    F.binary_usable(~operation,name,operations,same,ls,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})):
  match same:
    case True{}:
      +equal = ShapeLaws.same_extents(ls,rs,classified)
      +own = Equal.trans(Maybe<&2,List<&2,U32>>,Some{ls},Shapes.broadcast(ls,ls),Some{shape},
        Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(ls,ls),Some{ls},UnitBroadcast.self_broadcast(ls)),
        {%Equal.sym(List<&2,U32>,ls,rs,equal) : {Shapes.broadcast(ls,_) == Some{shape} : Maybe<&2,List<&2,U32>>} common : {Shapes.broadcast(ls,ls) == Some{shape} : Maybe<&2,List<&2,U32>>}})
      +left_shape = found_same(ls,shape,own)
      +right_ls = {%Equal.sym(List<&2,U32>,ls,rs,equal) : {Tensors.stretched(Tensors.view_of(_),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>} right_stretched :
        {Tensors.stretched(Tensors.view_of(ls),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>}}
      +right_own = {%Equal.sym(List<&2,U32>,ls,shape,left_shape) : {Tensors.stretched(Tensors.view_of(ls),_) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>} right_ls :
        {Tensors.stretched(Tensors.view_of(ls),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>}}
      %left_shape : Broadcast.BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,_,laxes,loffset,raxes,roffset,
        TiledBroadcast.same(~operation,name,operations,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
          Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}))
      broadcast_same(~Left,~left_value,~Right,~right_value,~operation,left,right,name,operations,limits,default_capacity,ls,lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,
        laxes,loffset,raxes,roffset,
        {%Equal.sym(List<&2,U32>,ls,shape,left_shape) : {Tensors.stretched(Tensors.view_of(ls),_) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>} left_stretched :
          {Tensors.stretched(Tensors.view_of(ls),ls) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>}},
        right_own,
        left_nonzero,rank_room,left_contract,
        {%Equal.sym(List<&2,U32>,ls,rs,equal) :
          Public.Contract(~Right,~right_value,right,_,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}) right_contract :
          Public.Contract(~Right,~right_value,right,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})})
    case False{}:
      Broadcast.combined(~Left,~left_value,~Right,~right_value,~operation,left,right,name,operations,limits,default_capacity,ls,rs,
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},
        shape,laxes,loffset,raxes,roffset,common,left_stretched,right_stretched,left_nonzero,right_nonzero,shape_nonzero,rank_room,right_rank_room,shape_rank,counted,
        left_contract,right_contract)

def broadcast_described(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,
  -left_tiled: Tiled.Tiled,+left_packed: Public.Packed<left_tiled>,-right_tiled: Tiled.Tiled,+right_packed: Public.Packed<right_tiled>,
  +shape: List<&2,U32>,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +common: {Shapes.broadcast(ls,rs) == Some{shape} : Maybe<&2,List<&2,U32>>},
  +left_stretched: {Tensors.stretched(Tensors.view_of(ls),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<shape>,
  +left_contract: Public.Contract(~Left,~left_value,left,ls,left_tiled),+right_contract: Public.Contract(~Right,~right_value,right,rs,right_tiled)) ->
  Broadcast.BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,F.binary(~operation,name,operations,left_tiled,right_tiled)):
  Public.Packed{+lm,+lc,+lf,+lw,+lo,+ll,+lp,+left_equation} = left_packed
  Public.Packed{+rm,+rc,+rf,+rw,+ro,+rl,+rp,+right_equation} = right_packed
  %left_equation : Broadcast.BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,
    F.binary(~operation,name,operations,_,right_tiled))
  %right_equation : Broadcast.BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,
    F.binary(~operation,name,operations,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},_))
  packed_call(~Left,~left_value,~Right,~right_value,
    ~(tiled => Broadcast.BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,tiled)),~operation,
    left,right,name,operations,ls,rs,lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},left_tiled,left_equation) :
      Public.Contract(~Left,~left_value,left,ls,_) left_contract :
      Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp})},
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},right_tiled,right_equation) :
      Public.Contract(~Right,~right_value,right,rs,_) right_contract :
      Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})},
    broadcast_usable(~Left,~left_value,~Right,~right_value,~operation,Shapes.same_extents(ls,rs),left,right,name,operations,limits,default_capacity,ls,rs,
      lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,shape,laxes,loffset,raxes,roffset,{==},common,left_stretched,right_stretched,left_nonzero,right_nonzero,shape_nonzero,
      rank_room,right_rank_room,shape_rank,counted,
      {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},left_tiled,left_equation) :
        Public.Contract(~Left,~left_value,left,ls,_) left_contract :
        Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp})},
      {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},right_tiled,right_equation) :
        Public.Contract(~Right,~right_value,right,rs,_) right_contract :
        Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})}))

# Two tiled operands of the contract and of extents without a zero under
# NumPy broadcasting: the result holds at every index of the common extents
# the operation of the two values the operands read through their views
# stretched to those extents.
def broadcast(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,
  +shape: List<&2,U32>,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +common: {Shapes.broadcast(ls,rs) == Some{shape} : Maybe<&2,List<&2,U32>>},
  +left_stretched: {Tensors.stretched(Tensors.view_of(ls),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<shape>,
  left_tiled: Tiled.Tiled,+left_contract: Public.Contract(~Left,~left_value,left,ls,left_tiled),
  right_tiled: Tiled.Tiled,+right_contract: Public.Contract(~Right,~right_value,right,rs,right_tiled)) ->
  Broadcast.BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,F.binary(~operation,name,operations,left_tiled,right_tiled)):
  broadcast_described(~Left,~left_value,~Right,~right_value,~operation,left,right,name,operations,limits,default_capacity,ls,rs,
    left_tiled,Public.packed(~Left,~left_value,left,ls,left_tiled,left_contract),right_tiled,Public.packed(~Right,~right_value,right,rs,right_tiled,right_contract),
    shape,laxes,loffset,raxes,roffset,common,left_stretched,right_stretched,left_nonzero,right_nonzero,shape_nonzero,rank_room,right_rank_room,shape_rank,counted,
    left_contract,right_contract)

# ---- The public rules ----

# The rule for operands of one shape: a tensor of the shape that holds the
# operation of the two values.
def SameLaw(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  ~operation: F32 -> F32 -> F32,~call: Tiled.Tiled -> Tiled.Tiled -> Tiled.Tiled) -> Type:
  @+left: Left -> @+right: Right -> @+limits: Machine.Limits -> @+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32} ->
  @+shape: List<&2,U32> -> @+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool} ->
  @+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} ->
  @left_tiled: Tiled.Tiled -> @+left_contract: Public.Contract(~Left,~left_value,left,shape,left_tiled) ->
  @right_tiled: Tiled.Tiled -> @+right_contract: Public.Contract(~Right,~right_value,right,shape,right_tiled) ->
  Public.Contract(~Leaf.Context<Left,Right>,~Leaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left,right},shape,call(left_tiled,right_tiled))

# The rule for operands that broadcast: the operation of the two operands read
# through their views stretched to the common extents.
def BroadcastLaw(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  ~operation: F32 -> F32 -> F32,~call: Tiled.Tiled -> Tiled.Tiled -> Tiled.Tiled) -> Type:
  @+left: Left -> @+right: Right -> @+limits: Machine.Limits -> @+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32} ->
  @+left_shape: List<&2,U32> -> @+right_shape: List<&2,U32> -> @+shape: List<&2,U32> ->
  @+laxes: List<&2,Views.Axis> -> @+loffset: U32 -> @+raxes: List<&2,Views.Axis> -> @+roffset: U32 ->
  @+common: {Shapes.broadcast(left_shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>} ->
  @+left_stretched: {Tensors.stretched(Tensors.view_of(left_shape),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>} ->
  @+right_stretched: {Tensors.stretched(Tensors.view_of(right_shape),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>} ->
  @+left_nonzero: {Shapes.has_zero(left_shape) == False{} : Bool} -> @+right_nonzero: {Shapes.has_zero(right_shape) == False{} : Bool} ->
  @+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool} ->
  @+rank_room: {Nat.is_le(List.length(&2,U32,left_shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} ->
  @+right_rank_room: {Nat.is_le(List.length(&2,U32,right_shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} ->
  @+shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} -> @+counted: Global.Counted<shape> ->
  @left_tiled: Tiled.Tiled -> @+left_contract: Public.Contract(~Left,~left_value,left,left_shape,left_tiled) ->
  @right_tiled: Tiled.Tiled -> @+right_contract: Public.Contract(~Right,~right_value,right,right_shape,right_tiled) ->
  Broadcast.BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,call(left_tiled,right_tiled))

type Rules<-Same: Type,-Broadcasting: Type> is Type:
  Rules{same: Same,broadcasting: Broadcasting}

def PublicLaw(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  ~operation: F32 -> F32 -> F32,~call: Tiled.Tiled -> Tiled.Tiled -> Tiled.Tiled) -> Type:
  Rules<SameLaw(~Left,~Right,~left_value,~right_value,~operation,~call),BroadcastLaw(~Left,~Right,~left_value,~right_value,~operation,~call)>

# Both rules of a binary operation.
def public(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,
  ~name: String,~cost: F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~operation,~F.binary(~operation,name,cost)):
  Rules{
    left => right => limits => default_capacity => shape => nonempty => rank_room => left_tiled => left_contract => right_tiled => right_contract =>
      same(~Left,~left_value,~Right,~right_value,~operation,left,right,name,cost,limits,default_capacity,shape,nonempty,rank_room,left_tiled,left_contract,right_tiled,right_contract),
    left => right => limits => default_capacity => left_shape => right_shape => shape => laxes => loffset => raxes => roffset => common => left_stretched => right_stretched =>
      left_nonzero => right_nonzero => shape_nonzero => rank_room => right_rank_room => shape_rank => counted => left_tiled => left_contract => right_tiled => right_contract =>
      broadcast(~Left,~left_value,~Right,~right_value,~operation,left,right,name,cost,limits,default_capacity,left_shape,right_shape,shape,laxes,loffset,raxes,roffset,common,
        left_stretched,right_stretched,left_nonzero,right_nonzero,shape_nonzero,rank_room,right_rank_room,shape_rank,counted,left_tiled,left_contract,right_tiled,right_contract)}
