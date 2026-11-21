# The tiled form of concat: every part, arranged on one lane axis with the
# groups innermost, placed into the joined cells holds the values of every
# part where it lies.

import Base
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ./tensor_global_shape_count.bend as Global
import ./tensor_placing_public.bend as Written
import ./tensor_move_readback.bend as Held
import ./tensor_pieces_model.bend as PiecesModel
import ../tensor_pieces.bend as Pieces
import ../tensor_tiled_concat.bend as TiledConcat
import ../tensor_grid.bend as Grid
import ../tensor_layout.bend as Layout
import ../tensor_tiled.bend as Tiled
import ./tensor_tiled_public.bend as Public
import ./tensor_cell_room.bend as Rooms
import ../machine_limits.bend as Machine
import ./nat_division.bend as Division
import ./row_major_index.bend as RowMajor
import ./encoded_coordinates.bend as Encoded
import ./u32_full_range.bend as FullRange
import ./binary_radix.bend as Radix
import ./u32_division.bend as U32Division
import ./natural_addition.bend as Addition
import ./u32_comparison.bend as Comparison
import ./boolean_logic.bend as Boolean
import ./tensor_tiled_placing.bend as Placing
import ../tensor_shape.bend as Shapes
import ./nat_ceiling.bend as Ceiling
import ./machine_limit_bounds.bend as Limits
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_placing_facts.bend as Facts
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_view_moved.bend as ViewMoved
import ./tensor_tiled_set_slice.bend as SetSlice
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding
import ./tensor_relane_facts.bend as RelaneFacts
import ../default_machine_limits.bend as Default
import ./tensor_storage_count.bend as StorageCount
import ../tensor_tasks.bend as Tasks
import ./tensor_box_digits.bend as Digits
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_target_cells.bend as Targets
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_relane.bend as Relane
import ./tensor_placing_gather.bend as Gathered
import ../kernel_gemm.bend as KernelGemm

# ---- The parts ----

# Every part is tiled values of the contract of its logical with its values.
def PartsHeld(~Part: Data,~part_value: Part -> U32 -> F32,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,parts: TiledConcat.Parts) -> Data:
  match logicals shapes:
    case Con{+part,+logicals} Con{+values,+shapes}:
      match parts:
        case TiledConcat.Single{tensor}:
          Held.Both<Public.Contract(~Part,~part_value,part,values,tensor),Held.Both<{logicals == [] : List<&2,Part>},{shapes == [] : List<&2,List<&2,U32>>}>>
        case TiledConcat.Cons{tensor,rest}:
          Held.Both<Public.Contract(~Part,~part_value,part,values,tensor),PartsHeld(~Part,~part_value,logicals,shapes,rest)>
    case Con{+part,+logicals} Nil{}: Empty
    case Nil{} +shapes: Empty

# The values of the joined tensor: those of the first part with the total
# extent along axis.
def joined_values(+shapes: List<&2,List<&2,U32>>,+axis: U32) -> List<&2,U32>:
  match shapes:
    case Con{+first,+rest}: Layout.with_extent(first,U32.to_nat(axis),Written.survey_total(Con{first,rest},U32.to_nat(axis),0))
    case Nil{}: []

# A usable tiled part as the model of its pieces.
type PartModel is Data:
  PartModel{pieces: PiecesModel.Model<Tile.Tile8>,shape: List<&2,U32>,fill: Tile.Tile8,workers: U32,lane: Nat,lanes: U32,position: Grid.Position}

def part_of(+model: PartModel) -> Tiled.Tiled:
  PartModel{+pieces,+shape,+fill,+workers,+lane,+lanes,+position} = model
  Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),shape,fill,Tensors.Usable{},workers},lane,lanes,position}

# Usable parts as models.
type PartsModel is Data:
  SingleModel{part: PartModel}
  ConsModel{part: PartModel,rest: PartsModel}

def parts_of(+model: PartsModel) -> TiledConcat.Parts:
  match model:
    case SingleModel{+part}: TiledConcat.Single{part_of(part)}
    case ConsModel{+part,+rest}: TiledConcat.Cons{part_of(part),parts_of(rest)}

# Parts that a model describes.
type Observed<-parts: TiledConcat.Parts> is Data:
  Observed{model: PartsModel,equation: {parts_of(model) == parts : TiledConcat.Parts}}

def single_observed(-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+lane: Nat,+lanes: U32,+position: Grid.Position,
  +usable: Global.Usable<Tile.Tile8,shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>) ->
  Observed<TiledConcat.Single{Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position}}>:
  Global.Usable{+model,+model_fill,+model_workers,+equation} = usable
  Observed{SingleModel{PartModel{model,shape,model_fill,model_workers,lane,lanes,position}},
    %equation : {TiledConcat.Single{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},lane,lanes,position}} ==
      TiledConcat.Single{Tiled.Tiled{_,lane,lanes,position}} : TiledConcat.Parts}
    {==}}

def cons_observed(-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+lane: Nat,+lanes: U32,+position: Grid.Position,-rest: TiledConcat.Parts,
  +usable: Global.Usable<Tile.Tile8,shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>,+later: Observed<rest>) ->
  Observed<TiledConcat.Cons{Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},rest}>:
  Global.Usable{+model,+model_fill,+model_workers,+equation} = usable
  Observed{+rest_model,+rest_equation} = later
  Observed{ConsModel{PartModel{model,shape,model_fill,model_workers,lane,lanes,position},rest_model},
    %equation : {TiledConcat.Cons{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},lane,lanes,position},parts_of(rest_model)} == TiledConcat.Cons{Tiled.Tiled{_,lane,lanes,position},rest} : TiledConcat.Parts}
    %rest_equation : {TiledConcat.Cons{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},lane,lanes,position},parts_of(rest_model)} == TiledConcat.Cons{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},lane,lanes,position},_} : TiledConcat.Parts}
    {==}}

# Parts of the contract are usable: a model describes them.
def observe_parts(~Part: Data,~part_value: Part -> U32 -> F32,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,parts: TiledConcat.Parts,
  +held: PartsHeld(~Part,~part_value,logicals,shapes,parts)) -> Observed<parts>:
  match logicals shapes:
    case Con{+part,+logicals} Con{+values,+shapes}:
      match parts:
        case TiledConcat.Single{tensor}:
          match tensor:
            case Tiled.Tiled{cells,+lane,+lanes,+position}:
              match cells:
                case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
                  single_observed(pieces,shape,fill,workers,lane,lanes,position,Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==}))
                case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
                  Held.Both{+contract,+ends} = held
                  Empty.absurd(Observed<TiledConcat.Single{Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},lane,lanes,position}}>,contract)
        case TiledConcat.Cons{tensor,rest}:
          match tensor:
            case Tiled.Tiled{cells,+lane,+lanes,+position}:
              match cells:
                case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
                  Held.Both{+contract,+held_rest} = held
                  cons_observed(pieces,shape,fill,workers,lane,lanes,position,rest,Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==}),
                    observe_parts(~Part,~part_value,logicals,shapes,rest,held_rest))
                case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
                  Held.Both{+contract,+held_rest} = held
                  Empty.absurd(Observed<TiledConcat.Cons{Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},lane,lanes,position},rest}>,contract)
    case Con{+part,+logicals} Nil{}: Empty.absurd(Observed<parts>,held)
    case Nil{} +shapes: Empty.absurd(Observed<parts>,held)

# ---- The survey of the runtime ----

# What the survey of the runtime finds of usable parts with these values.
def survey_fits(+shapes: List<&2,List<&2,U32>>,+first: List<&2,U32>,+axis: Nat,fits: Bool,+total: U32) -> Bool:
  match shapes:
    case Con{+values,+rest}:
      survey_fits(rest,first,axis,fits && TiledConcat.fitting(first,axis,Tensors.Usable{},values) && U32.is_le(Layout.extent_at(values,axis),(4294967295 - total : U32)),
        (total + Layout.extent_at(values,axis) : U32))
    case Nil{}: fits

# Whether no part follows.
def last_of(+shapes: List<&2,List<&2,U32>>) -> Bool:
  match shapes:
    case Con{+values,+rest}: False{}
    case Nil{}: True{}

def survey_whole(+shapes: List<&2,List<&2,U32>>,+axis: Nat,whole: Bool) -> Bool:
  match shapes:
    case Con{+values,+rest}: survey_whole(rest,axis,whole && (last_of(rest) || U32.is_zero((Layout.extent_at(values,axis) % 8 : U32))))
    case Nil{}: whole

# More parts have more values.
def more_parts(~Part: Data,~part_value: Part -> U32 -> F32,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,-parts: TiledConcat.Parts,
  +held: PartsHeld(~Part,~part_value,logicals,shapes,parts)) -> {last_of(shapes) == False{} : Bool}:
  match logicals shapes:
    case Con{+part,+logicals} Con{+values,+shapes}: {==}
    case Con{+part,+logicals} Nil{}: Empty.absurd({last_of([]) == False{} : Bool},held)
    case Nil{} +shapes: Empty.absurd({last_of(shapes) == False{} : Bool},held)

# The survey of usable parts of the contract finds what their values give.
def surveyed(~Part: Data,~part_value: Part -> U32 -> F32,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,+model: PartsModel,+first: List<&2,U32>,+axis: Nat,+lane: Nat,
  +fits: Bool,+whole: Bool,+total: U32,+held: PartsHeld(~Part,~part_value,logicals,shapes,parts_of(model))) ->
  {TiledConcat.survey(parts_of(model),first,axis,lane,TiledConcat.Fit{fits,whole,total}) ==
    (parts_of(model),TiledConcat.Fit{survey_fits(shapes,first,axis,fits,total),survey_whole(shapes,axis,whole),Written.survey_total(shapes,axis,total)}) : TiledConcat.Parts & TiledConcat.Fit}:
  match logicals shapes:
    case Con{+part,+logicals} Con{+values,+shapes}:
      match model:
        case SingleModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position}}:
          Held.Both{KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}},Held.Both{+named,+last}} = held
          %Equal.sym(List<&2,List<&2,U32>>,shapes,[],last) :
            {TiledConcat.survey(TiledConcat.Single{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position}},first,axis,lane,TiledConcat.Fit{fits,whole,total}) ==
              (TiledConcat.Single{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position}},TiledConcat.Fit{survey_fits(Con{values,_},first,axis,fits,total),survey_whole(Con{values,_},axis,whole),Written.survey_total(Con{values,_},axis,total)}) :
              TiledConcat.Parts & TiledConcat.Fit}
          %Equal.sym(List<&2,U32>,Tiled.logical_shape(cells_shape,own,lanes,position),values,restored) :
            {(TiledConcat.Single{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position}},TiledConcat.Fit{fits && TiledConcat.fitting(first,axis,Tensors.Usable{},_) && U32.is_le(Layout.extent_at(_,axis),(4294967295 - total : U32)),
                whole && (True{} || U32.is_zero((Layout.extent_at(_,axis) % 8 : U32))),(total + Layout.extent_at(_,axis) : U32)}) ==
              (TiledConcat.Single{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position}},TiledConcat.Fit{survey_fits([values],first,axis,fits,total),survey_whole([values],axis,whole),Written.survey_total([values],axis,total)}) :
              TiledConcat.Parts & TiledConcat.Fit}
          {==}
        case ConsModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position},+rest}:
          Held.Both{KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}},+held_rest} = held
          +next_fits = fits && TiledConcat.fitting(first,axis,Tensors.Usable{},values) && U32.is_le(Layout.extent_at(values,axis),(4294967295 - total : U32))
          +next_total = (total + Layout.extent_at(values,axis) : U32)
          +next_whole = whole && (False{} || U32.is_zero((Layout.extent_at(values,axis) % 8 : U32)))
          +later = {TiledConcat.Fit{survey_fits(shapes,first,axis,next_fits,next_total),survey_whole(shapes,axis,next_whole),Written.survey_total(shapes,axis,next_total)} : TiledConcat.Fit}
          %Equal.sym(Bool,last_of(shapes),False{},more_parts(~Part,~part_value,logicals,shapes,parts_of(rest),held_rest)) :
            {TiledConcat.survey(TiledConcat.Cons{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},parts_of(rest)},first,axis,lane,TiledConcat.Fit{fits,whole,total}) ==
              (TiledConcat.Cons{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},parts_of(rest)},TiledConcat.Fit{survey_fits(shapes,first,axis,next_fits,next_total),survey_whole(shapes,axis,whole && (_ || U32.is_zero((Layout.extent_at(values,axis) % 8 : U32)))),
                Written.survey_total(shapes,axis,next_total)}) : TiledConcat.Parts & TiledConcat.Fit}
          %Equal.sym(List<&2,U32>,Tiled.logical_shape(cells_shape,own,lanes,position),values,restored) :
            {TiledConcat.prepend(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},TiledConcat.survey(parts_of(rest),first,axis,lane,TiledConcat.Fit{fits && TiledConcat.fitting(first,axis,Tensors.Usable{},_) &&
                U32.is_le(Layout.extent_at(_,axis),(4294967295 - total : U32)),whole && (False{} || U32.is_zero((Layout.extent_at(_,axis) % 8 : U32))),(total + Layout.extent_at(_,axis) : U32)})) ==
              (TiledConcat.Cons{Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},parts_of(rest)},later) : TiledConcat.Parts & TiledConcat.Fit}
          Equal.cong(TiledConcat.Parts & TiledConcat.Fit,TiledConcat.Parts & TiledConcat.Fit,result => TiledConcat.prepend(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},result),
            TiledConcat.survey(parts_of(rest),first,axis,lane,TiledConcat.Fit{next_fits,next_whole,next_total}),(parts_of(rest),later),
            surveyed(~Part,~part_value,logicals,shapes,rest,first,axis,lane,next_fits,next_whole,next_total,held_rest))
    case Con{+part,+logicals} Nil{}:
      Empty.absurd({TiledConcat.survey(parts_of(model),first,axis,lane,TiledConcat.Fit{fits,whole,total}) ==
          (parts_of(model),TiledConcat.Fit{survey_fits([],first,axis,fits,total),survey_whole([],axis,whole),Written.survey_total([],axis,total)}) :
          TiledConcat.Parts & TiledConcat.Fit},held)
    case Nil{} +shapes:
      Empty.absurd({TiledConcat.survey(parts_of(model),first,axis,lane,TiledConcat.Fit{fits,whole,total}) ==
          (parts_of(model),TiledConcat.Fit{survey_fits(shapes,first,axis,fits,total),survey_whole(shapes,axis,whole),Written.survey_total(shapes,axis,total)}) :
          TiledConcat.Parts & TiledConcat.Fit},held)

# ---- Groups along the axis ----

# A word that counts whole groups of eight is aligned.
def aligned_of(+word: U32,+groups: Nat,+exact: {U32.to_nat(word) == Nat.mul(groups,8n) : Nat}) -> {U32.is_zero((word % 8 : U32)) == True{} : Bool}:
  +zero = Equal.trans(Nat,U32.to_nat((word % 8 : U32)),Nat.mod(U32.to_nat(word),8n),0n,U32Division.modulo(word,8,2147483648,{==},{==},{==}),
    {%Equal.sym(Nat,U32.to_nat(word),Nat.mul(groups,8n),exact) : {Nat.mod(_,8n) == 0n : Nat}
      %Addition.zero_right(Nat.mul(groups,8n)) : {Nat.mod(_,8n) == 0n : Nat}
      Radix.remainder(groups,0n,8n,{==},{==}) : {Nat.mod(U32.to_nat(word),8n) == 0n : Nat}})
  %Equal.sym(Bool,U32.is_eq((word % 8 : U32),0),Nat.is_eq(U32.to_nat((word % 8 : U32)),0n),Comparison.eq((word % 8 : U32),0)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((word % 8 : U32)),0n,zero) : {Nat.is_eq(_,0n) == True{} : Bool}
  {==}

# Two aligned words with a sum within the capacity have an aligned sum.
def aligned_sum(+limits: Machine.Limits,+first: U32,+second: U32,+first_aligned: {U32.is_zero((first % 8 : U32)) == True{} : Bool},
  +second_aligned: {U32.is_zero((second % 8 : U32)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(second)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.is_zero(((first + second : U32) % 8 : U32)) == True{} : Bool}:
  +a = U32.to_nat((first / 8 : U32))
  +b = U32.to_nat((second / 8 : U32))
  aligned_of((first + second : U32),Nat.add(a,b),
    Equal.trans(Nat,U32.to_nat((first + second : U32)),Nat.add(U32.to_nat(first),U32.to_nat(second)),Nat.mul(Nat.add(a,b),8n),
      FullRange.add(first,second,Machine.capacity(limits),room),
      Equal.trans(Nat,Nat.add(U32.to_nat(first),U32.to_nat(second)),Nat.add(Nat.mul(a,8n),Nat.mul(b,8n)),Nat.mul(Nat.add(a,b),8n),
        {%Radix.divide_eight_exact_multiple(first,first_aligned) : {Nat.add(_,U32.to_nat(second)) == Nat.add(Nat.mul(a,8n),Nat.mul(b,8n)) : Nat}
          %Radix.divide_eight_exact_multiple(second,second_aligned) : {Nat.add(Nat.mul(a,8n),_) == Nat.add(Nat.mul(a,8n),Nat.mul(b,8n)) : Nat}
          {==} : {Nat.add(U32.to_nat(first),U32.to_nat(second)) == Nat.add(Nat.mul(a,8n),Nat.mul(b,8n)) : Nat}},
        Equal.sym(Nat,Nat.mul(Nat.add(a,b),8n),Nat.add(Nat.mul(a,8n),Nat.mul(b,8n)),RowMajor.distrib(a,b,8n)))))

# Whether every part before the last fills its groups, from a start.
def whole_split(+shapes: List<&2,List<&2,U32>>,+axis: Nat,+whole: Bool) -> {survey_whole(shapes,axis,whole) == whole && survey_whole(shapes,axis,True{}) : Bool}:
  match shapes:
    case Con{+values,+rest}:
      +step = last_of(rest) || U32.is_zero((Layout.extent_at(values,axis) % 8 : U32))
      +later = survey_whole(rest,axis,True{})
      Equal.trans(Bool,survey_whole(rest,axis,whole && step),(whole && step) && later,whole && survey_whole(rest,axis,step),
        whole_split(rest,axis,whole && step),
        Equal.trans(Bool,(whole && step) && later,whole && (step && later),whole && survey_whole(rest,axis,step),Boolean.and_associative(whole,step,later),
          Equal.cong(Bool,Bool,found => whole && found,step && later,survey_whole(rest,axis,step),Equal.sym(Bool,survey_whole(rest,axis,step),step && later,whole_split(rest,axis,step)))))
    case Nil{}: Equal.sym(Bool,whole && True{},whole,Placing.and_true_right(whole))

# The lanes of the joined tensor on the axis start a group at every part and
# every part before the last fills its groups, or lie elsewhere.
def LanesKept(+lane: Nat,+mark: Nat,+start: U32,+shapes: List<&2,List<&2,U32>>) -> Data:
  {Bool.not(Nat.is_eq(lane,mark)) || (U32.is_zero((start % 8 : U32)) && survey_whole(shapes,mark,True{})) == True{} : Bool}

# The last part ends the axis, the others fill their groups.
def rest_cases(+rest: List<&2,List<&2,U32>>,+mark: Nat,+start: U32,+values: List<&2,U32>,+extent: U32,
  +stepped: {last_of(rest) || U32.is_zero((Layout.extent_at(values,mark) % 8 : U32)) == True{} : Bool},
  +total: {Written.survey_total(Con{values,rest},mark,start) == extent : U32}) ->
  {U32.is_zero((Layout.extent_at(values,mark) % 8 : U32)) || U32.is_eq((start + Layout.extent_at(values,mark) : U32),extent) == True{} : Bool}:
  match rest:
    case Nil{}:
      %Equal.sym(Bool,U32.is_eq((start + Layout.extent_at(values,mark) : U32),extent),True{},Encoded.equal_words((start + Layout.extent_at(values,mark) : U32),extent,total)) :
        {U32.is_zero((Layout.extent_at(values,mark) % 8 : U32)) || _ == True{} : Bool}
      Boolean.or_true_right(U32.is_zero((Layout.extent_at(values,mark) % 8 : U32)))
    case Con{+next,+later}:
      %Equal.sym(Bool,U32.is_zero((Layout.extent_at(values,mark) % 8 : U32)),True{},stepped) :
        {_ || U32.is_eq((start + Layout.extent_at(values,mark) : U32),extent) == True{} : Bool}
      {==}

# A part keeps the lanes of the joined tensor.
def part_keeps(same: Bool,+lane: Nat,+mark: Nat,+start: U32,+values: List<&2,U32>,+rest: List<&2,List<&2,U32>>,+extent: U32,
  +classified: {Nat.is_eq(lane,mark) == same : Bool},+kept: LanesKept(lane,mark,start,Con{values,rest}),
  +total: {Written.survey_total(Con{values,rest},mark,start) == extent : U32}) ->
  {SetSlice.kept_lanes(lane,mark,start,Layout.extent_at(values,mark),extent) == True{} : Bool}:
  match same:
    case False{}:
      +width = Layout.extent_at(values,mark)
      +rest_kept = U32.is_zero((width % 8 : U32)) || U32.is_eq((start + width : U32),extent)
      %Equal.sym(Bool,Nat.is_eq(lane,mark),False{},classified) : {Bool.not(_) || (U32.is_zero((start % 8 : U32)) && rest_kept) == True{} : Bool}
      {==}
    case True{}:
      +width = Layout.extent_at(values,mark)
      +rest_kept = U32.is_zero((width % 8 : U32)) || U32.is_eq((start + width : U32),extent)
      +inner = {%classified : {Bool.not(_) || (U32.is_zero((start % 8 : U32)) && survey_whole(Con{values,rest},mark,True{})) == True{} : Bool} kept :
        {U32.is_zero((start % 8 : U32)) && survey_whole(Con{values,rest},mark,True{}) == True{} : Bool}}
      +aligned = Boolean.left(U32.is_zero((start % 8 : U32)),survey_whole(Con{values,rest},mark,True{}),inner)
      +step = last_of(rest) || U32.is_zero((width % 8 : U32))
      +wholes = {%whole_split(rest,mark,step) : {_ == True{} : Bool} Boolean.right(U32.is_zero((start % 8 : U32)),survey_whole(Con{values,rest},mark,True{}),inner) :
        {step && survey_whole(rest,mark,True{}) == True{} : Bool}}
      +stepped = Boolean.left(step,survey_whole(rest,mark,True{}),wholes)
      %Equal.sym(Bool,Nat.is_eq(lane,mark),True{},classified) : {Bool.not(_) || (U32.is_zero((start % 8 : U32)) && rest_kept) == True{} : Bool}
      %Equal.sym(Bool,U32.is_zero((start % 8 : U32)),True{},aligned) : {_ && rest_kept == True{} : Bool}
      rest_cases(rest,mark,start,values,extent,stepped,total)

# The lanes stay kept from the next part on.
def next_kept(same: Bool,+limits: Machine.Limits,+lane: Nat,+mark: Nat,+start: U32,+values: List<&2,U32>,+next: List<&2,U32>,+rest: List<&2,List<&2,U32>>,
  +classified: {Nat.is_eq(lane,mark) == same : Bool},+kept: LanesKept(lane,mark,start,Con{values,Con{next,rest}}),
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(Layout.extent_at(values,mark))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  LanesKept(lane,mark,(start + Layout.extent_at(values,mark) : U32),Con{next,rest}):
  match same:
    case False{}:
      +width = Layout.extent_at(values,mark)
      %Equal.sym(Bool,Nat.is_eq(lane,mark),False{},classified) :
        {Bool.not(_) || (U32.is_zero(((start + width : U32) % 8 : U32)) && survey_whole(Con{next,rest},mark,True{})) == True{} : Bool}
      {==}
    case True{}:
      +width = Layout.extent_at(values,mark)
      +inner = {%classified : {Bool.not(_) || (U32.is_zero((start % 8 : U32)) && survey_whole(Con{values,Con{next,rest}},mark,True{})) == True{} : Bool} kept :
        {U32.is_zero((start % 8 : U32)) && survey_whole(Con{values,Con{next,rest}},mark,True{}) == True{} : Bool}}
      +aligned = Boolean.left(U32.is_zero((start % 8 : U32)),survey_whole(Con{values,Con{next,rest}},mark,True{}),inner)
      +step = {False{} || U32.is_zero((width % 8 : U32)) : Bool}
      +wholes = {%whole_split(Con{next,rest},mark,step) : {_ == True{} : Bool} Boolean.right(U32.is_zero((start % 8 : U32)),survey_whole(Con{values,Con{next,rest}},mark,True{}),inner) :
        {step && survey_whole(Con{next,rest},mark,True{}) == True{} : Bool}}
      %Equal.sym(Bool,Nat.is_eq(lane,mark),True{},classified) : {Bool.not(_) || (U32.is_zero(((start + width : U32) % 8 : U32)) && survey_whole(Con{next,rest},mark,True{})) == True{} : Bool}
      Boolean.and_true(U32.is_zero(((start + width : U32) % 8 : U32)),survey_whole(Con{next,rest},mark,True{}),
        aligned_sum(limits,start,width,aligned,Boolean.left(step,survey_whole(Con{next,rest},mark,True{}),wholes),room),
        Boolean.right(step,survey_whole(Con{next,rest},mark,True{}),wholes))

# ---- The cells of a part ----

# The cells of a part that fits the joined tensor from start fit the joined
# cells at the cell axis from the group of the start, and are placed at the
# cells of its shifts.
def part_cells(+limits: Machine.Limits,+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32,+lane: Nat,+position: Grid.Position,
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +part_room: {Nat.is_le(ShapeLaws.elements(part_values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},+nonempty: {Shapes.has_zero(part_values) == False{} : Bool},
  +keeps: {SetSlice.kept_lanes(lane,U32.to_nat(axis),start,Layout.extent_at(part_values,U32.to_nat(axis)),Layout.extent_at(values,U32.to_nat(axis))) == True{} : Bool}) ->
  Held.Both<{Layout.slice_fits(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(part_values,lane,position),U32.from_nat(Tiled.cell_axis(position,U32.to_nat(axis),lane,List.length(&2,U32,values))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start),0) == True{} : Bool},Held.Both<{Shapes.has_zero(Tiled.cell_shape(part_values,lane,position)) == False{} : Bool},
    Held.Both<{Layout.shifts_at(List.length(&2,U32,Tiled.cell_shape(part_values,lane,position)),U32.from_nat(Tiled.cell_axis(position,U32.to_nat(axis),lane,List.length(&2,U32,values))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start),0) == Placing.cell_shifts(Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),lane,position) : List<&2,U32>},
      Facts.PlacedIn(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(part_values,lane,position),Placing.cell_shifts(Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),lane,position))>>>:
  +rank = List.length(&2,U32,values)
  +mark = U32.to_nat(axis)
  +cap = Machine.capacity(limits)
  +cells = Tiled.cell_shape(values,lane,position)
  +part_cells = Tiled.cell_shape(part_values,lane,position)
  +cell_axis = Tiled.cell_axis(position,U32.to_nat(axis),lane,List.length(&2,U32,values))
  +cell_start = Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start)
  +shifts = Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0)
  +marked_fits = SetSlice.marked_of(limits,values,part_values,axis,start,rank_room,fits)
  +ranked = SetSlice.fits_length(values,part_values,mark,start,marked_fits)
  +mark_inside = SetSlice.fits_inside(values,part_values,mark,start,marked_fits)
  +part_mark_inside = {%Equal.sym(Nat,List.length(&2,U32,part_values),rank,ranked) : {Nat.is_lt(mark,_) == True{} : Bool} mark_inside :
    {Nat.is_lt(mark,List.length(&2,U32,part_values)) == True{} : Bool}}
  +part_rank_room = {%Equal.sym(Nat,List.length(&2,U32,part_values),rank,ranked) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} rank_room :
    {Nat.is_le(List.length(&2,U32,part_values),U32.to_nat(cap)) == True{} : Bool}}
  +part_positive = CountCompletion.positive_elements(part_values,nonempty)
  +cells_nonempty = ViewMoved.none_zero(part_cells,Rooms.cells_positive(limits,part_values,lane,position,part_positive,part_room))
  +cells_marked = SetSlice.cell_fits(limits,values,part_values,mark,start,lane,position,Layout.extent_at(part_values,mark),Layout.extent_at(values,mark),marked_fits,keeps,
    Rooms.lane_room(limits,values,mark,mark_inside,positive,room))
  +cell_ranked = Rooms.cell_rank(values,lane,position)
  +part_cell_ranked = Rooms.cell_rank(part_values,lane,position)
  +cells_rank_room = {%Equal.sym(Nat,List.length(&2,U32,cells),rank,cell_ranked) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} rank_room :
    {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(cap)) == True{} : Bool}}
  +part_cells_rank = Equal.trans(Nat,List.length(&2,U32,part_cells),List.length(&2,U32,part_values),rank,part_cell_ranked,ranked)
  +part_cells_rank_room = {%Equal.sym(Nat,List.length(&2,U32,part_cells),rank,part_cells_rank) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} rank_room :
    {Nat.is_le(List.length(&2,U32,part_cells),U32.to_nat(cap)) == True{} : Bool}}
  +axis_room = Order.transitive(cell_axis,List.length(&2,U32,cells),U32.to_nat(cap),Bounds.strict_le(cell_axis,List.length(&2,U32,cells),SetSlice.fits_inside(cells,part_cells,cell_axis,cell_start,cells_marked)),cells_rank_room)
  +axis_back = Bounds.roundtrip(cell_axis,cap,Limits.capacity_fits(limits),axis_room)
  +cells_fits = Equal.trans(Bool,Layout.slice_fits(cells,part_cells,U32.from_nat(cell_axis),cell_start,0),SetSlice.fits_marked(cells,part_cells,cell_axis,cell_start),True{},
    SetSlice.fits_mark_eq(cells,part_cells,U32.from_nat(cell_axis),cell_start,0n,cell_axis,cap,Limits.capacity_fits(limits),cells_rank_room,axis_back),cells_marked)
  +marks = SetSlice.shifts_marked(List.length(&2,U32,part_values),axis,start,0n,mark,cap,Limits.capacity_fits(limits),part_rank_room,{==})
  +part_axis = Equal.cong(Nat,Nat,count => Tiled.cell_axis(position,mark,lane,count),rank,List.length(&2,U32,part_values),Equal.sym(Nat,List.length(&2,U32,part_values),rank,ranked))
  +cells_shifts = Equal.trans(List<&2,U32>,Layout.shifts_at(List.length(&2,U32,part_cells),U32.from_nat(cell_axis),cell_start,0),SetSlice.marked(List.length(&2,U32,part_cells),cell_axis,cell_start),
      Placing.cell_shifts(shifts,lane,position),
    SetSlice.shifts_marked(List.length(&2,U32,part_cells),U32.from_nat(cell_axis),cell_start,0n,cell_axis,cap,Limits.capacity_fits(limits),part_cells_rank_room,axis_back),
    Equal.trans(List<&2,U32>,SetSlice.marked(List.length(&2,U32,part_cells),cell_axis,cell_start),SetSlice.marked(List.length(&2,U32,part_values),cell_axis,cell_start),Placing.cell_shifts(shifts,lane,position),
      Equal.cong(Nat,List<&2,U32>,count => SetSlice.marked(count,cell_axis,cell_start),List.length(&2,U32,part_cells),List.length(&2,U32,part_values),part_cell_ranked),
      Equal.trans(List<&2,U32>,SetSlice.marked(List.length(&2,U32,part_values),cell_axis,cell_start),
          SetSlice.marked(List.length(&2,U32,part_values),Tiled.cell_axis(position,mark,lane,List.length(&2,U32,part_values)),cell_start),Placing.cell_shifts(shifts,lane,position),
        Equal.cong(Nat,List<&2,U32>,found => SetSlice.marked(List.length(&2,U32,part_values),found,cell_start),cell_axis,Tiled.cell_axis(position,mark,lane,List.length(&2,U32,part_values)),part_axis),
        Equal.trans(List<&2,U32>,SetSlice.marked(List.length(&2,U32,part_values),Tiled.cell_axis(position,mark,lane,List.length(&2,U32,part_values)),cell_start),
            Placing.cell_shifts(SetSlice.marked(List.length(&2,U32,part_values),mark,start),lane,position),Placing.cell_shifts(shifts,lane,position),
          Equal.sym(List<&2,U32>,Placing.cell_shifts(SetSlice.marked(List.length(&2,U32,part_values),mark,start),lane,position),
            SetSlice.marked(List.length(&2,U32,part_values),Tiled.cell_axis(position,mark,lane,List.length(&2,U32,part_values)),cell_start),
            SetSlice.cell_marked(List.length(&2,U32,part_values),mark,start,lane,position,part_mark_inside)),
          Equal.cong(List<&2,U32>,List<&2,U32>,moved => Placing.cell_shifts(moved,lane,position),SetSlice.marked(List.length(&2,U32,part_values),mark,start),shifts,
            Equal.sym(List<&2,U32>,shifts,SetSlice.marked(List.length(&2,U32,part_values),mark,start),marks))))))
  +cells_same = Equal.trans(Nat,List.length(&2,U32,cells),rank,List.length(&2,U32,part_cells),cell_ranked,Equal.sym(Nat,List.length(&2,U32,part_cells),rank,part_cells_rank))
  Held.Both{cells_fits,Held.Both{cells_nonempty,Held.Both{cells_shifts,
    {%cells_shifts : Facts.PlacedIn(cells,part_cells,_)
      %cells_same : Facts.PlacedIn(cells,part_cells,Layout.shifts_at(_,U32.from_nat(cell_axis),cell_start,0))
      Written.slice_in(cells,part_cells,U32.from_nat(cell_axis),cell_start,0,cells_fits) : Facts.PlacedIn(cells,part_cells,Placing.cell_shifts(shifts,lane,position))}}}}

# ---- The joined values ----

# A part that fits a target at the mark, given the extent of another part
# that fits it at the mark, is that part.
def same_fit(target: List<&2,U32>,part: List<&2,U32>,other: List<&2,U32>,mark: Nat,+start: U32,+other_start: U32,
  +fits: {SetSlice.fits_marked(target,part,mark,start) == True{} : Bool},+other_fits: {SetSlice.fits_marked(target,other,mark,other_start) == True{} : Bool}) ->
  {Layout.with_extent(part,mark,Grid.lanes_at(other,mark)) == other : List<&2,U32>}:
  match target part other mark:
    case Con{+extent,+rest} Con{+width,+part_rest} Con{+own,+other_rest} 0n:
      +part_same = SetSlice.same_listed(rest,part_rest,Boolean.right(U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),SetSlice.same_extents(rest,part_rest),fits))
      +other_same = SetSlice.same_listed(rest,other_rest,Boolean.right(U32.is_le(other_start,extent) && U32.is_le(own,(extent - other_start : U32)),SetSlice.same_extents(rest,other_rest),other_fits))
      Equal.cong(List<&2,U32>,List<&2,U32>,later => own <> later,part_rest,other_rest,
        Equal.trans(List<&2,U32>,part_rest,rest,other_rest,part_same,Equal.sym(List<&2,U32>,other_rest,rest,other_same)))
    case Con{+extent,+rest} Con{+width,+part_rest} Con{+own,+other_rest} 1n+ +next:
      +equal = Arithmetic.equal_of_is_eq(extent,width,Boolean.left(U32.is_eq(extent,width),SetSlice.fits_marked(rest,part_rest,next,start),fits))
      +other_equal = Arithmetic.equal_of_is_eq(extent,own,Boolean.left(U32.is_eq(extent,own),SetSlice.fits_marked(rest,other_rest,next,other_start),other_fits))
      %Equal.trans(U32,own,extent,width,Equal.sym(U32,extent,own,other_equal),equal) :
        {_ <> Layout.with_extent(part_rest,next,Grid.lanes_at(other_rest,next)) == own <> other_rest : List<&2,U32>}
      %same_fit(rest,part_rest,other_rest,next,start,other_start,Boolean.right(U32.is_eq(extent,width),SetSlice.fits_marked(rest,part_rest,next,start),fits),
          Boolean.right(U32.is_eq(extent,own),SetSlice.fits_marked(rest,other_rest,next,other_start),other_fits)) :
        {own <> Layout.with_extent(part_rest,next,Grid.lanes_at(other_rest,next)) == own <> _ : List<&2,U32>}
      {==}
    case Con{+extent,+rest} Con{+width,+part_rest} Nil{} +mark:
      Empty.absurd({Layout.with_extent(Con{width,part_rest},mark,Grid.lanes_at(Nil{},mark)) == Nil{} : List<&2,U32>},WordEncoding.false_true(other_fits))
    case Con{+extent,+rest} Nil{} +other +mark:
      Empty.absurd({Layout.with_extent(Nil{},mark,Grid.lanes_at(other,mark)) == other : List<&2,U32>},WordEncoding.false_true(fits))
    case Nil{} +part +other +mark:
      Empty.absurd({Layout.with_extent(part,mark,Grid.lanes_at(other,mark)) == other : List<&2,U32>},WordEncoding.false_true(fits))

# The extent set at an axis inside the extents is the one there.
def set_lane(values: List<&2,U32>,axis: Nat,+word: U32,+inside: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  {Grid.lanes_at(Layout.with_extent(values,axis,word),axis) == word : U32}:
  match values axis:
    case Con{+extent,+rest} 0n: {==}
    case Con{+extent,+rest} 1n+ +next: set_lane(rest,next,word,inside)
    case Nil{} +axis: Empty.absurd({Grid.lanes_at(Layout.with_extent(Nil{},axis,word),axis) == word : U32},Order.below_zero(axis,inside))

# A part that fits the joined values fits the survey of the runtime.
def fitting_part(+values: List<&2,U32>,+first: List<&2,U32>,+part_values: List<&2,U32>,+mark: Nat,+start: U32,
  +first_fits: {SetSlice.fits_marked(values,first,mark,0) == True{} : Bool},+fits: {SetSlice.fits_marked(values,part_values,mark,start) == True{} : Bool},
  +nonempty: {Shapes.has_zero(part_values) == False{} : Bool}) ->
  {TiledConcat.fitting(first,mark,Tensors.Usable{},part_values) == True{} : Bool}:
  +first_inside = {%Equal.sym(Nat,List.length(&2,U32,first),List.length(&2,U32,values),SetSlice.fits_length(values,first,mark,0,first_fits)) :
      {Nat.is_lt(mark,_) == True{} : Bool}
    SetSlice.fits_inside(values,first,mark,0,first_fits) : {Nat.is_lt(mark,List.length(&2,U32,first)) == True{} : Bool}}
  %Equal.sym(U32,Layout.extent_at(first,mark),Grid.lanes_at(first,mark),SetSlice.extent_lanes(first,mark,first_inside)) :
    {Tensors.usable(Tensors.Usable{}) && Bool.not(Shapes.has_zero(part_values)) && Shapes.same_extents(Layout.with_extent(part_values,mark,_),first) == True{} : Bool}
  %Equal.sym(List<&2,U32>,Layout.with_extent(part_values,mark,Grid.lanes_at(first,mark)),first,same_fit(values,part_values,first,mark,start,0,fits,first_fits)) :
    {Tensors.usable(Tensors.Usable{}) && Bool.not(Shapes.has_zero(part_values)) && Shapes.same_extents(_,first) == True{} : Bool}
  %Equal.sym(Bool,Shapes.has_zero(part_values),False{},nonempty) :
    {Tensors.usable(Tensors.Usable{}) && Bool.not(_) && Shapes.same_extents(first,first) == True{} : Bool}
  %Equal.sym(Bool,Shapes.same_extents(first,first),True{},ShapeLaws.same_extents_reflexive(first)) :
    {Tensors.usable(Tensors.Usable{}) && Bool.not(False{}) && _ == True{} : Bool}
  {==}

# A width that ends within an extent from start has room in a word after
# start.
def word_after(+start: U32,+width: U32,+extent: U32,+top: U32,+top_exact: {top == 4294967295 : U32},
  +within: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(extent)) == True{} : Bool}) ->
  {U32.is_le(width,(top - start : U32)) == True{} : Bool}:
  +s = U32.to_nat(start)
  +w = U32.to_nat(width)
  +most = U32.to_nat(top)
  +start_bound = {%Equal.sym(U32,top,4294967295,top_exact) : {U32.is_le(start,_) == True{} : Bool} FullRange.maximum(start) : {U32.is_le(start,top) == True{} : Bool}}
  +extent_bound = {%Equal.sym(U32,top,4294967295,top_exact) : {U32.is_le(extent,_) == True{} : Bool} FullRange.maximum(extent) : {U32.is_le(extent,top) == True{} : Bool}}
  +ordered = Bounds.as_nat_le(start,top,start_bound)
  +bounded = Order.transitive(Nat.add(s,w),U32.to_nat(extent),most,within,Bounds.as_nat_le(extent,top,extent_bound))
  +exact = FullRange.subtract(top,start,top,ordered,Order.sub_le(most,s))
  %Equal.sym(Bool,U32.is_le(width,(top - start : U32)),Nat.is_le(w,U32.to_nat((top - start : U32))),Comparison.le(width,(top - start : U32))) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((top - start : U32)),Nat.sub(most,s),exact) : {Nat.is_le(w,_) == True{} : Bool}
  %Order.le_offset(s,w,Nat.sub(most,s)) : {_ == True{} : Bool}
  %Equal.sym(Nat,Nat.add(s,Nat.sub(most,s)),most,Order.restore_shift(s,most,ordered)) : {Nat.is_le(Nat.add(s,w),_) == True{} : Bool}
  bounded

# A part that fits the joined values from start has its extent room in a
# word after start.
def word_room(+values: List<&2,U32>,+part_values: List<&2,U32>,+mark: Nat,+start: U32,+fits: {SetSlice.fits_marked(values,part_values,mark,start) == True{} : Bool}) ->
  {U32.is_le(Layout.extent_at(part_values,mark),(4294967295 - start : U32)) == True{} : Bool}:
  +inside = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),SetSlice.fits_length(values,part_values,mark,start,fits)) :
      {Nat.is_lt(mark,_) == True{} : Bool}
    SetSlice.fits_inside(values,part_values,mark,start,fits) : {Nat.is_lt(mark,List.length(&2,U32,part_values)) == True{} : Bool}}
  %Equal.sym(U32,Layout.extent_at(part_values,mark),Grid.lanes_at(part_values,mark),SetSlice.extent_lanes(part_values,mark,inside)) : {U32.is_le(_,(4294967295 - start : U32)) == True{} : Bool}
  word_after(start,Grid.lanes_at(part_values,mark),Grid.lanes_at(values,mark),4294967295,{==},
    Written.slice_axis(True{},Grid.lanes_at(values,mark),Grid.lanes_at(part_values,mark),start,SetSlice.fits_hit(values,part_values,mark,start,fits)))

# Parts that fit the joined values one after another from start fit the
# survey of the runtime.
def fits_surveyed(+shapes: List<&2,List<&2,U32>>,+values: List<&2,U32>,+first: List<&2,U32>,+axis: U32,+start: U32,+limits: Machine.Limits,
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +first_fits: {SetSlice.fits_marked(values,first,U32.to_nat(axis),0) == True{} : Bool},+fit: Written.ShapesFit(shapes,values,axis,start)) ->
  {survey_fits(shapes,first,U32.to_nat(axis),True{},start) == True{} : Bool}:
  match shapes:
    case Con{+part_values,+rest}:
      Held.Both{+fits,Held.Both{+nonempty,+later}} = fit
      +mark = U32.to_nat(axis)
      +marked_fits = SetSlice.marked_of(limits,values,part_values,axis,start,rank_room,fits)
      +next = (start + Layout.extent_at(part_values,mark) : U32)
      %Equal.sym(Bool,TiledConcat.fitting(first,mark,Tensors.Usable{},part_values),True{},fitting_part(values,first,part_values,mark,start,first_fits,marked_fits,nonempty)) :
        {survey_fits(rest,first,mark,True{} && _ && U32.is_le(Layout.extent_at(part_values,mark),(4294967295 - start : U32)),next) == True{} : Bool}
      %Equal.sym(Bool,U32.is_le(Layout.extent_at(part_values,mark),(4294967295 - start : U32)),True{},word_room(values,part_values,mark,start,marked_fits)) :
        {survey_fits(rest,first,mark,True{} && True{} && _,next) == True{} : Bool}
      fits_surveyed(rest,values,first,axis,next,limits,rank_room,first_fits,later)
    case Nil{}: {==}

# ---- The lane axis of the joined cells ----

# The lane axis of the joined cells lies within the joined values and keeps
# its lanes: off the axis, or along it when every part before the last fills
# its groups.
def lane_cases(same: Bool,whole: Bool,+shapes: List<&2,List<&2,U32>>,+first: List<&2,U32>,+mark: Nat,+own: Nat,
  +same_classified: {Nat.is_eq(mark,own) == same : Bool},+whole_classified: {survey_whole(shapes,mark,True{}) == whole : Bool},
  +inside: {Nat.is_lt(mark,List.length(&2,U32,first)) == True{} : Bool},+own_inside: {Nat.is_le(own,List.length(&2,U32,first)) == True{} : Bool}) ->
  Held.Both<{Nat.is_le(TiledConcat.concat_lane(first,mark,own,whole),List.length(&2,U32,first)) == True{} : Bool},LanesKept(TiledConcat.concat_lane(first,mark,own,whole),mark,0,shapes)>:
  match same whole:
    case True{} False{}:
      +rank = List.length(&2,U32,first)
      +longest = Tiled.longest_other(first,mark,0n,rank,2)
      +apart = Equal.trans(Bool,Nat.is_eq(rank,mark),Nat.is_eq(mark,rank),False{},Order.equal_symmetric(rank,mark),Order.less_not_equal(mark,rank,inside))
      %Equal.sym(Bool,Nat.is_eq(mark,own),True{},same_classified) :
        Held.Both<{Nat.is_le(Bool.pick(Nat,_ && Bool.not(False{}),longest,own),rank) == True{} : Bool},LanesKept(Bool.pick(Nat,_ && Bool.not(False{}),longest,own),mark,0,shapes)>
      Held.Both{SetSlice.longest_within(first,mark,0n,rank,2,rank,Order.reflexive(rank),Order.reflexive(rank)),
        {%Equal.sym(Bool,Nat.is_eq(longest,mark),False{},SetSlice.longest_apart(first,mark,0n,rank,2,apart)) :
            {Bool.not(_) || (U32.is_zero((0 % 8 : U32)) && survey_whole(shapes,mark,True{})) == True{} : Bool}
          {==} : LanesKept(longest,mark,0,shapes)}}
    case True{} True{}:
      +rank = List.length(&2,U32,first)
      +longest = Tiled.longest_other(first,mark,0n,rank,2)
      %Equal.sym(Bool,Nat.is_eq(mark,own),True{},same_classified) :
        Held.Both<{Nat.is_le(Bool.pick(Nat,_ && Bool.not(True{}),longest,own),rank) == True{} : Bool},LanesKept(Bool.pick(Nat,_ && Bool.not(True{}),longest,own),mark,0,shapes)>
      Held.Both{own_inside,
        {%Equal.sym(Bool,survey_whole(shapes,mark,True{}),True{},whole_classified) : {Bool.not(Nat.is_eq(own,mark)) || (U32.is_zero((0 % 8 : U32)) && _) == True{} : Bool}
          Boolean.or_true_right(Bool.not(Nat.is_eq(own,mark))) : LanesKept(own,mark,0,shapes)}}
    case False{} True{}:
      +rank = List.length(&2,U32,first)
      +longest = Tiled.longest_other(first,mark,0n,rank,2)
      %Equal.sym(Bool,Nat.is_eq(mark,own),False{},same_classified) :
        Held.Both<{Nat.is_le(Bool.pick(Nat,_ && Bool.not(True{}),longest,own),rank) == True{} : Bool},LanesKept(Bool.pick(Nat,_ && Bool.not(True{}),longest,own),mark,0,shapes)>
      Held.Both{own_inside,
        {%Equal.sym(Bool,survey_whole(shapes,mark,True{}),True{},whole_classified) : {Bool.not(Nat.is_eq(own,mark)) || (U32.is_zero((0 % 8 : U32)) && _) == True{} : Bool}
          Boolean.or_true_right(Bool.not(Nat.is_eq(own,mark))) : LanesKept(own,mark,0,shapes)}}
    case False{} False{}:
      +rank = List.length(&2,U32,first)
      +longest = Tiled.longest_other(first,mark,0n,rank,2)
      +swapped = Equal.trans(Bool,Nat.is_eq(own,mark),Nat.is_eq(mark,own),False{},Order.equal_symmetric(own,mark),same_classified)
      %Equal.sym(Bool,Nat.is_eq(mark,own),False{},same_classified) :
        Held.Both<{Nat.is_le(Bool.pick(Nat,_ && Bool.not(False{}),longest,own),rank) == True{} : Bool},LanesKept(Bool.pick(Nat,_ && Bool.not(False{}),longest,own),mark,0,shapes)>
      Held.Both{own_inside,
        {%Equal.sym(Bool,Nat.is_eq(own,mark),False{},swapped) : {Bool.not(_) || (U32.is_zero((0 % 8 : U32)) && survey_whole(shapes,mark,True{})) == True{} : Bool}
          {==} : LanesKept(own,mark,0,shapes)}}

# ---- The cells along the axis ----

# The place of an axis among the other axes before the lane axis.
def place_same(less: Bool,+mark: Nat,+lane: Nat,+classified: {Nat.is_lt(mark,lane) == less : Bool},+apart: {Nat.is_eq(mark,lane) == False{} : Bool}) ->
  {Bool.pick(Nat,less,mark,Nat.sub(mark,1n)) == RelaneFacts.inner_place(lane,True{},mark) : Nat}:
  match less:
    case True{}:
      %Equal.sym(Bool,Nat.is_lt(lane,mark),False{},Order.not_below(mark,lane,Bounds.strict_le(mark,lane,classified))) : {mark == Bool.pick(Nat,_,Nat.sub(mark,1n),mark) : Nat}
      {==}
    case False{}:
      +after = RelaneFacts.le_apart(lane,mark,Order.not_less(mark,lane,classified),Equal.trans(Bool,Nat.is_eq(lane,mark),Nat.is_eq(mark,lane),False{},Order.equal_symmetric(lane,mark),apart))
      %Equal.sym(Bool,Nat.is_lt(lane,mark),True{},after) : {Nat.sub(mark,1n) == Bool.pick(Nat,_,Nat.sub(mark,1n),mark) : Nat}
      {==}

def appended_extent(+values: List<&2,U32>,+mark: Nat,+lane: Nat,+groups: U32,
  +inner: Held.Both<{Grid.lanes_at(List.append(&2,U32,Grid.removed(values,lane),[groups]),RelaneFacts.inner_place(lane,True{},mark)) == Grid.lanes_at(values,mark) : U32},
    {Nat.is_lt(RelaneFacts.inner_place(lane,True{},mark),List.length(&2,U32,List.append(&2,U32,Grid.removed(values,lane),[groups]))) == True{} : Bool}>) ->
  {Grid.lanes_at(List.append(&2,U32,Grid.removed(values,lane),[groups]),RelaneFacts.inner_place(lane,True{},mark)) == Grid.lanes_at(values,mark) : U32}:
  Held.Both{+same,+below} = inner
  same

# The extent of the cells of values at the cell axis of an axis: the groups
# of its lanes when it is the lane axis, its extent otherwise.
def lane_extent(within: Bool,same: Bool,+values: List<&2,U32>,+mark: Nat,+lane: Nat,
  +classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+same_classified: {Nat.is_eq(mark,lane) == same : Bool},
  +inside: {Nat.is_lt(mark,List.length(&2,U32,values)) == True{} : Bool}) ->
  {Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Tiled.cell_axis(Grid.Innermost{},mark,lane,List.length(&2,U32,values))) ==
    Bool.pick(U32,same,Grid.groups(Grid.lanes_at(values,mark)),Grid.lanes_at(values,mark)) : U32}:
  match within same:
    case False{} False{}:
      +rank = List.length(&2,U32,values)
      %Equal.sym(Bool,Nat.is_lt(lane,rank),False{},classified) :
        {Grid.lanes_at(Grid.placed(_,Grid.Innermost{},Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),Bool.pick(Nat,_,Tiled.cell_axis_at(Grid.Innermost{},mark,lane,rank),mark)) ==
          Grid.lanes_at(values,mark) : U32}
      %Equal.sym(List<&2,U32>,Grid.removed(values,lane),values,Public.removed_past(values,lane,classified)) : {Grid.lanes_at(_,mark) == Grid.lanes_at(values,mark) : U32}
      {==}
    case False{} True{}:
      +rank = List.length(&2,U32,values)
      +apart = Order.less_not_equal(mark,lane,Bounds.nat_trans(mark,rank,lane,inside,Order.not_less(lane,rank,classified)))
      Empty.absurd({Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Tiled.cell_axis(Grid.Innermost{},mark,lane,rank)) == Bool.pick(U32,True{},Grid.groups(Grid.lanes_at(values,mark)),Grid.lanes_at(values,mark)) : U32},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(mark,lane),True{},Equal.sym(Bool,Nat.is_eq(mark,lane),False{},apart),same_classified)))
    case True{} True{}:
      +rank = List.length(&2,U32,values)
      +others = Grid.removed(values,lane)
      +groups = Grid.groups(Grid.lanes_at(values,lane))
      %Equal.sym(Bool,Nat.is_lt(lane,rank),True{},classified) :
        {Grid.lanes_at(Grid.placed(_,Grid.Innermost{},others,groups),Bool.pick(Nat,_,Tiled.cell_axis_at(Grid.Innermost{},mark,lane,rank),mark)) == Grid.groups(Grid.lanes_at(values,mark)) : U32}
      %Equal.sym(Bool,Nat.is_eq(mark,lane),True{},same_classified) :
        {Grid.lanes_at(List.append(&2,U32,others,[groups]),Bool.pick(Nat,_,Nat.sub(rank,1n),Bool.pick(Nat,Nat.is_lt(mark,lane),mark,Nat.sub(mark,1n)))) == Grid.groups(Grid.lanes_at(values,mark)) : U32}
      %Equal.sym(Nat,mark,lane,Order.equal_of_is_eq(mark,lane,same_classified)) : {Grid.lanes_at(List.append(&2,U32,others,[groups]),Nat.sub(rank,1n)) == Grid.groups(Grid.lanes_at(values,_)) : U32}
      %Public.removed_length(values,lane,classified) : {Grid.lanes_at(List.append(&2,U32,others,[groups]),Nat.sub(_,1n)) == groups : U32}
      %Equal.sym(Nat,Nat.sub(1n+List.length(&2,U32,others),1n),List.length(&2,U32,others),Order.subtract_zero(List.length(&2,U32,others))) :
        {Grid.lanes_at(List.append(&2,U32,others,[groups]),_) == groups : U32}
      RelaneFacts.appended_last(others,groups)
    case True{} False{}:
      +rank = List.length(&2,U32,values)
      +others = Grid.removed(values,lane)
      +groups = Grid.groups(Grid.lanes_at(values,lane))
      %Equal.sym(Bool,Nat.is_lt(lane,rank),True{},classified) :
        {Grid.lanes_at(Grid.placed(_,Grid.Innermost{},others,groups),Bool.pick(Nat,_,Tiled.cell_axis_at(Grid.Innermost{},mark,lane,rank),mark)) == Grid.lanes_at(values,mark) : U32}
      %Equal.sym(Bool,Nat.is_eq(mark,lane),False{},same_classified) :
        {Grid.lanes_at(List.append(&2,U32,others,[groups]),Bool.pick(Nat,_,Nat.sub(rank,1n),Bool.pick(Nat,Nat.is_lt(mark,lane),mark,Nat.sub(mark,1n)))) == Grid.lanes_at(values,mark) : U32}
      %Equal.sym(Nat,Bool.pick(Nat,Nat.is_lt(mark,lane),mark,Nat.sub(mark,1n)),RelaneFacts.inner_place(lane,True{},mark),place_same(Nat.is_lt(mark,lane),mark,lane,{==},same_classified)) :
        {Grid.lanes_at(List.append(&2,U32,others,[groups]),_) == Grid.lanes_at(values,mark) : U32}
      appended_extent(values,mark,lane,groups,
        RelaneFacts.innermost_lanes(lane,mark,values,groups,RelaneFacts.inner_both(lane,mark,values,RelaneFacts.inner_lanes(lane,mark,values,same_classified,inside,classified),classified)))

# Along the lane axis a part before the last starts and ends on a group.
def kept_aligned(same: Bool,+lane: Nat,+mark: Nat,+start: U32,+values: List<&2,U32>,+next: List<&2,U32>,+rest: List<&2,List<&2,U32>>,
  +classified: {Nat.is_eq(lane,mark) == same : Bool},+kept: LanesKept(lane,mark,start,Con{values,Con{next,rest}})) ->
  {Bool.not(same) || (U32.is_zero((start % 8 : U32)) && U32.is_zero((Layout.extent_at(values,mark) % 8 : U32))) == True{} : Bool}:
  match same:
    case False{}: {==}
    case True{}:
      +width = Layout.extent_at(values,mark)
      +inner = {%classified : {Bool.not(_) || (U32.is_zero((start % 8 : U32)) && survey_whole(Con{values,Con{next,rest}},mark,True{})) == True{} : Bool} kept :
        {U32.is_zero((start % 8 : U32)) && survey_whole(Con{values,Con{next,rest}},mark,True{}) == True{} : Bool}}
      +step = {False{} || U32.is_zero((width % 8 : U32)) : Bool}
      +wholes = {%whole_split(Con{next,rest},mark,step) : {_ == True{} : Bool} Boolean.right(U32.is_zero((start % 8 : U32)),survey_whole(Con{values,Con{next,rest}},mark,True{}),inner) :
        {step && survey_whole(Con{next,rest},mark,True{}) == True{} : Bool}}
      Boolean.and_true(U32.is_zero((start % 8 : U32)),U32.is_zero((width % 8 : U32)),Boolean.left(U32.is_zero((start % 8 : U32)),survey_whole(Con{values,Con{next,rest}},mark,True{}),inner),
        Boolean.left(step,survey_whole(Con{next,rest},mark,True{}),wholes))

# Along the lane axis the last part starts on a group.
def last_aligned(same: Bool,+lane: Nat,+mark: Nat,+start: U32,+values: List<&2,U32>,
  +classified: {Nat.is_eq(lane,mark) == same : Bool},+kept: LanesKept(lane,mark,start,[values])) ->
  {Bool.not(same) || U32.is_zero((start % 8 : U32)) == True{} : Bool}:
  match same:
    case False{}: {==}
    case True{}:
      +inner = {%classified : {Bool.not(_) || (U32.is_zero((start % 8 : U32)) && survey_whole([values],mark,True{})) == True{} : Bool} kept :
        {U32.is_zero((start % 8 : U32)) && survey_whole([values],mark,True{}) == True{} : Bool}}
      Boolean.left(U32.is_zero((start % 8 : U32)),survey_whole([values],mark,True{}),inner)

# A group start and a whole number of groups end on the group of the sum.
def next_start(same: Bool,+limits: Machine.Limits,+start: U32,+width: U32,
  +aligned: {Bool.not(same) || (U32.is_zero((start % 8 : U32)) && U32.is_zero((width % 8 : U32))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {(Bool.pick(U32,same,(start / 8 : U32),start) + Bool.pick(U32,same,Grid.groups(width),width) : U32) == Bool.pick(U32,same,((start + width : U32) / 8 : U32),(start + width : U32)) : U32}:
  match same:
    case False{}: {==}
    case True{}:
      +cap = U32.to_nat(Machine.capacity(limits))
      +s_aligned = Boolean.left(U32.is_zero((start % 8 : U32)),U32.is_zero((width % 8 : U32)),aligned)
      +w_aligned = Boolean.right(U32.is_zero((start % 8 : U32)),U32.is_zero((width % 8 : U32)),aligned)
      +a = U32.to_nat((start / 8 : U32))
      +b = U32.to_nat((width / 8 : U32))
      +w_room = Order.transitive(U32.to_nat(width),Nat.add(U32.to_nat(start),U32.to_nat(width)),cap,Order.right_le(U32.to_nat(start),U32.to_nat(width)),room)
      +s_exact = Radix.divide_eight_exact_multiple(start,s_aligned)
      +w_exact = Radix.divide_eight_exact_multiple(width,w_aligned)
      +sum_aligned = aligned_sum(limits,start,width,s_aligned,w_aligned,room)
      +q = U32.to_nat(((start + width : U32) / 8 : U32))
      +q_exact = Radix.divide_eight_exact_multiple((start + width : U32),sum_aligned)
      +parts = {%s_exact : {Nat.add(_,U32.to_nat(width)) == Nat.add(Nat.mul(a,8n),Nat.mul(b,8n)) : Nat}
          %w_exact : {Nat.add(Nat.mul(a,8n),_) == Nat.add(Nat.mul(a,8n),Nat.mul(b,8n)) : Nat}
          {==} : {Nat.add(U32.to_nat(start),U32.to_nat(width)) == Nat.add(Nat.mul(a,8n),Nat.mul(b,8n)) : Nat}}
      +sum_groups = Equal.trans(Nat,Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.add(Nat.mul(a,8n),Nat.mul(b,8n)),Nat.mul(Nat.add(a,b),8n),parts,
        Equal.sym(Nat,Nat.mul(Nat.add(a,b),8n),Nat.add(Nat.mul(a,8n),Nat.mul(b,8n)),RowMajor.distrib(a,b,8n)))
      +whole = Equal.trans(Nat,Nat.mul(q,8n),U32.to_nat((start + width : U32)),Nat.mul(Nat.add(a,b),8n),q_exact,
        Equal.trans(Nat,U32.to_nat((start + width : U32)),Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.mul(Nat.add(a,b),8n),FullRange.add(start,width,Machine.capacity(limits),room),sum_groups))
      +q_same = Radix.row_unique(q,Nat.add(a,b),8n,0n,0n,{==},{==},{==},Equal.cong(Nat,Nat,value => Nat.add(value,0n),Nat.mul(q,8n),Nat.mul(Nat.add(a,b),8n),whole))
      +g_exact = Equal.trans(Nat,U32.to_nat(Grid.groups(width)),Ceiling.ceiling(U32.to_nat(width),8n),b,Encoded.eight_blocks_exact(limits,width,w_room),
        Equal.trans(Nat,Ceiling.ceiling(U32.to_nat(width),8n),Ceiling.ceiling(Nat.add(Nat.mul(b,8n),0n),8n),b,
          Equal.cong(Nat,Nat,value => Ceiling.ceiling(value,8n),U32.to_nat(width),Nat.add(Nat.mul(b,8n),0n),
            Equal.trans(Nat,U32.to_nat(width),Nat.mul(b,8n),Nat.add(Nat.mul(b,8n),0n),Equal.sym(Nat,Nat.mul(b,8n),U32.to_nat(width),w_exact),
              Equal.sym(Nat,Nat.add(Nat.mul(b,8n),0n),Nat.mul(b,8n),Addition.zero_right(Nat.mul(b,8n))))),
          Equal.trans(Nat,Ceiling.ceiling(Nat.add(Nat.mul(b,8n),0n),8n),Nat.add(b,Ceiling.ceiling(0n,8n)),b,
            Equal.sym(Nat,Nat.add(b,Ceiling.ceiling(0n,8n)),Ceiling.ceiling(Nat.add(Nat.mul(b,8n),0n),8n),Ceiling.ceiling_after(b,0n)),
            Addition.zero_right(b))))
      +g_room = {%Equal.sym(Nat,U32.to_nat(Grid.groups(width)),b,g_exact) : {Nat.is_le(Nat.add(a,_),cap) == True{} : Bool}
        Order.transitive(Nat.add(a,b),Nat.mul(Nat.add(a,b),8n),cap,RowMajor.product_covers(Nat.add(a,b),8n,{==}),
          {%sum_groups : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(Nat.mul(Nat.add(a,b),8n),cap) == True{} : Bool}}) :
        {Nat.is_le(Nat.add(a,U32.to_nat(Grid.groups(width))),cap) == True{} : Bool}}
      +left = Equal.trans(Nat,U32.to_nat(((start / 8 : U32) + Grid.groups(width) : U32)),Nat.add(a,U32.to_nat(Grid.groups(width))),Nat.add(a,b),
        FullRange.add((start / 8 : U32),Grid.groups(width),Machine.capacity(limits),g_room),
        Equal.cong(Nat,Nat,value => Nat.add(a,value),U32.to_nat(Grid.groups(width)),b,g_exact))
      Arithmetic.to_nat_injective(((start / 8 : U32) + Grid.groups(width) : U32),((start + width : U32) / 8 : U32),
        Equal.trans(Nat,U32.to_nat(((start / 8 : U32) + Grid.groups(width) : U32)),Nat.add(a,b),q,left,Equal.sym(Nat,q,Nat.add(a,b),q_same)))

# A group start and the last width end on the groups of the extent.
def last_start(same: Bool,+limits: Machine.Limits,+start: U32,+width: U32,+extent: U32,
  +aligned: {Bool.not(same) || U32.is_zero((start % 8 : U32)) == True{} : Bool},+sum: {(start + width : U32) == extent : U32},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {(Bool.pick(U32,same,(start / 8 : U32),start) + Bool.pick(U32,same,Grid.groups(width),width) : U32) == Bool.pick(U32,same,Grid.groups(extent),extent) : U32}:
  match same:
    case False{}: sum
    case True{}:
      +cap = U32.to_nat(Machine.capacity(limits))
      +s = U32.to_nat(start)
      +w = U32.to_nat(width)
      +a = U32.to_nat((start / 8 : U32))
      +w_room = Order.transitive(w,Nat.add(s,w),cap,Order.right_le(s,w),room)
      +s_exact = Radix.divide_eight_exact_multiple(start,aligned)
      +e_nat = Equal.trans(Nat,U32.to_nat(extent),U32.to_nat((start + width : U32)),Nat.add(s,w),
        Equal.cong(U32,Nat,word => U32.to_nat(word),extent,(start + width : U32),Equal.sym(U32,(start + width : U32),extent,sum)),
        FullRange.add(start,width,Machine.capacity(limits),room))
      +e_room = {%Equal.sym(Nat,U32.to_nat(extent),Nat.add(s,w),e_nat) : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(U32.to_nat(extent),cap) == True{} : Bool}}
      +c = Ceiling.ceiling(Nat.add(s,w),8n)
      +left_c = Equal.trans(Nat,Nat.add(a,U32.to_nat(Grid.groups(width))),Nat.add(a,Ceiling.ceiling(w,8n)),c,
        Equal.cong(Nat,Nat,value => Nat.add(a,value),U32.to_nat(Grid.groups(width)),Ceiling.ceiling(w,8n),Encoded.eight_blocks_exact(limits,width,w_room)),
        Equal.trans(Nat,Nat.add(a,Ceiling.ceiling(w,8n)),Ceiling.ceiling(Nat.add(Nat.mul(a,8n),w),8n),c,Ceiling.ceiling_after(a,w),
          Equal.cong(Nat,Nat,value => Ceiling.ceiling(Nat.add(value,w),8n),Nat.mul(a,8n),s,s_exact)))
      +c_room = Order.transitive(c,Nat.add(s,w),cap,Ceiling.minimal(Nat.add(s,w),8n,Nat.add(s,w),{==},RowMajor.product_covers(Nat.add(s,w),8n,{==})),room)
      +g_room = {%Equal.sym(Nat,Nat.add(a,U32.to_nat(Grid.groups(width))),c,left_c) : {Nat.is_le(_,cap) == True{} : Bool} c_room :
        {Nat.is_le(Nat.add(a,U32.to_nat(Grid.groups(width))),cap) == True{} : Bool}}
      +left = Equal.trans(Nat,U32.to_nat(((start / 8 : U32) + Grid.groups(width) : U32)),Nat.add(a,U32.to_nat(Grid.groups(width))),c,
        FullRange.add((start / 8 : U32),Grid.groups(width),Machine.capacity(limits),g_room),left_c)
      +right = Equal.trans(Nat,U32.to_nat(Grid.groups(extent)),Ceiling.ceiling(U32.to_nat(extent),8n),c,Encoded.eight_blocks_exact(limits,extent,e_room),
        Equal.cong(Nat,Nat,value => Ceiling.ceiling(value,8n),U32.to_nat(extent),Nat.add(s,w),e_nat))
      Arithmetic.to_nat_injective(((start / 8 : U32) + Grid.groups(width) : U32),Grid.groups(extent),
        Equal.trans(Nat,U32.to_nat(((start / 8 : U32) + Grid.groups(width) : U32)),c,U32.to_nat(Grid.groups(extent)),left,Equal.sym(Nat,U32.to_nat(Grid.groups(extent)),c,right)))

# The cells of a part before the last end where the cells of the next part
# start.
def cells_next(+limits: Machine.Limits,+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32,+lane: Nat,
  +axis_back: {U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))) == Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)) : Nat},+ranked: {List.length(&2,U32,part_values) == List.length(&2,U32,values) : Nat},
  +mark_inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,part_values)) == True{} : Bool},+cell_inside: {Nat.is_lt(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),List.length(&2,U32,Tiled.cell_shape(part_values,lane,Grid.Innermost{}))) == True{} : Bool},
  +aligned: {Bool.not(Nat.is_eq(U32.to_nat(axis),lane)) || (U32.is_zero((start % 8 : U32)) && U32.is_zero((Layout.extent_at(part_values,U32.to_nat(axis)) % 8 : U32))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(Layout.extent_at(part_values,U32.to_nat(axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {(Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start) + Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32) == Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),((start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32) / 8 : U32),(start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32)) : U32}:
  +mark = U32.to_nat(axis)
  +same = Nat.is_eq(mark,lane)
  +cell_axis = Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))
  +width = Grid.lanes_at(part_values,mark)
  +extent_same = SetSlice.extent_lanes(part_values,mark,mark_inside)
  +cell_extent = Equal.trans(U32,Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))),Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),cell_axis),Bool.pick(U32,same,Grid.groups(width),width),
    Equal.cong(Nat,U32,place => Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),place),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),cell_axis,axis_back),
    Equal.trans(U32,Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),cell_axis),Grid.lanes_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),cell_axis),Bool.pick(U32,same,Grid.groups(width),width),
      SetSlice.extent_lanes(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),cell_axis,cell_inside),
      Equal.trans(U32,Grid.lanes_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),cell_axis),Grid.lanes_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),Tiled.cell_axis(Grid.Innermost{},mark,lane,List.length(&2,U32,part_values))),Bool.pick(U32,same,Grid.groups(width),width),
        Equal.cong(Nat,U32,count => Grid.lanes_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),Tiled.cell_axis(Grid.Innermost{},mark,lane,count)),List.length(&2,U32,values),List.length(&2,U32,part_values),
          Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked)),
        lane_extent(Nat.is_lt(lane,List.length(&2,U32,part_values)),same,part_values,mark,lane,{==},{==},mark_inside))))
  %Equal.sym(U32,Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))),Bool.pick(U32,same,Grid.groups(width),width),cell_extent) :
    {(Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start) + _ : U32) == Bool.pick(U32,same,((start + Layout.extent_at(part_values,mark) : U32) / 8 : U32),(start + Layout.extent_at(part_values,mark) : U32)) : U32}
  %Equal.sym(U32,Layout.extent_at(part_values,mark),width,extent_same) :
    {(Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start) + Bool.pick(U32,same,Grid.groups(width),width) : U32) == Bool.pick(U32,same,((start + _ : U32) / 8 : U32),(start + _ : U32)) : U32}
  next_start(same,limits,start,width,
    {%extent_same : {Bool.not(same) || (U32.is_zero((start % 8 : U32)) && U32.is_zero((_ % 8 : U32))) == True{} : Bool} aligned :
      {Bool.not(same) || (U32.is_zero((start % 8 : U32)) && U32.is_zero((width % 8 : U32))) == True{} : Bool}},
    {%extent_same : {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(_)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room : {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})

# The cells of the last part end the cells of the joined values.
def cells_last(+limits: Machine.Limits,+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32,+lane: Nat,
  +axis_back: {U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))) == Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)) : Nat},+ranked: {List.length(&2,U32,part_values) == List.length(&2,U32,values) : Nat},
  +mark_inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,part_values)) == True{} : Bool},+cell_inside: {Nat.is_lt(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),List.length(&2,U32,Tiled.cell_shape(part_values,lane,Grid.Innermost{}))) == True{} : Bool},
  +aligned: {Bool.not(Nat.is_eq(U32.to_nat(axis),lane)) || U32.is_zero((start % 8 : U32)) == True{} : Bool},
  +sum: {(start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32) == Layout.extent_at(values,U32.to_nat(axis)) : U32},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(Layout.extent_at(part_values,U32.to_nat(axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {(Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start) + Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32) == Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32}:
  +mark = U32.to_nat(axis)
  +same = Nat.is_eq(mark,lane)
  +cell_axis = Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))
  +width = Grid.lanes_at(part_values,mark)
  +extent = Grid.lanes_at(values,mark)
  +values_inside = {%ranked : {Nat.is_lt(mark,_) == True{} : Bool} mark_inside : {Nat.is_lt(mark,List.length(&2,U32,values)) == True{} : Bool}}
  +extent_same = SetSlice.extent_lanes(part_values,mark,mark_inside)
  +extent_value = SetSlice.extent_lanes(values,mark,values_inside)
  +cell_extent = Equal.trans(U32,Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))),Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),cell_axis),Bool.pick(U32,same,Grid.groups(width),width),
    Equal.cong(Nat,U32,place => Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),place),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),cell_axis,axis_back),
    Equal.trans(U32,Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),cell_axis),Grid.lanes_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),cell_axis),Bool.pick(U32,same,Grid.groups(width),width),
      SetSlice.extent_lanes(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),cell_axis,cell_inside),
      Equal.trans(U32,Grid.lanes_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),cell_axis),Grid.lanes_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),Tiled.cell_axis(Grid.Innermost{},mark,lane,List.length(&2,U32,part_values))),Bool.pick(U32,same,Grid.groups(width),width),
        Equal.cong(Nat,U32,count => Grid.lanes_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),Tiled.cell_axis(Grid.Innermost{},mark,lane,count)),List.length(&2,U32,values),List.length(&2,U32,part_values),
          Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked)),
        lane_extent(Nat.is_lt(lane,List.length(&2,U32,part_values)),same,part_values,mark,lane,{==},{==},mark_inside))))
  +cells_extent = Equal.trans(U32,Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))),Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),cell_axis),Bool.pick(U32,same,Grid.groups(extent),extent),
    Equal.cong(Nat,U32,place => Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),place),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),cell_axis,axis_back),
    lane_extent(Nat.is_lt(lane,List.length(&2,U32,values)),same,values,mark,lane,{==},{==},values_inside))
  %Equal.sym(U32,Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))),Bool.pick(U32,same,Grid.groups(width),width),cell_extent) : {(Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start) + _ : U32) == Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32}
  %Equal.sym(U32,Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))),Bool.pick(U32,same,Grid.groups(extent),extent),cells_extent) : {(Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start) + Bool.pick(U32,same,Grid.groups(width),width) : U32) == _ : U32}
  last_start(same,limits,start,width,extent,aligned,
    {%extent_same : {(start + _ : U32) == extent : U32} %extent_value : {(start + Layout.extent_at(part_values,mark) : U32) == _ : U32} sum :
      {(start + width : U32) == extent : U32}},
    {%extent_same : {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(_)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room : {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})

# ---- What every part brings ----

# Every part counts its values within the capacity.
def PartsRoom(+limits: Machine.Limits,shapes: List<&2,List<&2,U32>>) -> Data:
  match shapes:
    case Con{+values,shapes}: Held.Both<{Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},PartsRoom(limits,shapes)>
    case Nil{}: Unit

def parts_room(~Part: Data,~part_value: Part -> U32 -> F32,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,+model: PartsModel,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+held: PartsHeld(~Part,~part_value,logicals,shapes,parts_of(model))) ->
  PartsRoom(limits,shapes):
  match logicals shapes:
    case Con{+part,+logicals} Con{+values,+shapes}:
      match model:
        case SingleModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position}}:
          Held.Both{KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}},Held.Both{+named,+last}} = held
          %Equal.sym(List<&2,List<&2,U32>>,shapes,[],last) : PartsRoom(limits,Con{values,_})
          Held.Both{StorageCount.counted_room(limits,values,counted,default_capacity),Unit{}}
        case ConsModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position},+rest}:
          Held.Both{KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}},+held_rest} = held
          Held.Both{StorageCount.counted_room(limits,values,counted,default_capacity),parts_room(~Part,~part_value,logicals,shapes,rest,limits,default_capacity,held_rest)}
    case Con{+part,+logicals} Nil{}: Empty.absurd(PartsRoom(limits,[]),held)
    case Nil{} +shapes: Empty.absurd(PartsRoom(limits,shapes),held)

# Parts that fit the joined values one after another stay within a word.
def shapes_valid(+shapes: List<&2,List<&2,U32>>,+values: List<&2,U32>,+axis: U32,+start: U32,+limits: Machine.Limits,
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fit: Written.ShapesFit(shapes,values,axis,start)) ->
  {Written.survey_valid(shapes,U32.to_nat(axis),True{},start) == True{} : Bool}:
  match shapes:
    case Con{+part_values,+rest}:
      Held.Both{+fits,Held.Both{+nonempty,+later}} = fit
      +mark = U32.to_nat(axis)
      %Equal.sym(Bool,U32.is_le(Layout.extent_at(part_values,mark),(4294967295 - start : U32)),True{},
          word_room(values,part_values,mark,start,SetSlice.marked_of(limits,values,part_values,axis,start,rank_room,fits))) :
        {Written.survey_valid(rest,mark,True{} && _,(start + Layout.extent_at(part_values,mark) : U32)) == True{} : Bool}
      shapes_valid(rest,values,axis,(start + Layout.extent_at(part_values,mark) : U32),limits,rank_room,later)
    case Nil{}: {==}

# ---- Shifts and segments of the cells ----

# The cells of every shift.
def cell_shift_lists(shifts: List<&2,List<&2,U32>>,+lane: Nat) -> List<&2,List<&2,U32>>:
  match shifts:
    case Con{+shift,shifts}: Placing.cell_shifts(shift,lane,Grid.Innermost{}) <> cell_shift_lists(shifts,lane)
    case Nil{}: []

# The segments of parts one after another are those of their shifts.
def segments_shifts(shapes: List<&2,List<&2,U32>>,+axis: U32,+start: U32) ->
  {Written.segments_of(shapes,axis,start) == Placing.value_segments(shapes,Written.shifts_of(shapes,axis,start)) : List<&2,List<&2,List<&2,Tasks.Segment>>>}:
  match shapes:
    case Con{+shape,+rest}:
      +next = (start + Layout.extent_at(shape,U32.to_nat(axis)) : U32)
      Equal.cong(List<&2,List<&2,List<&2,Tasks.Segment>>>,List<&2,List<&2,List<&2,Tasks.Segment>>>,later => Layout.placed_segments(shape,Layout.shifts_at(List.length(&2,U32,shape),axis,start,0)) <> later,
        Written.segments_of(rest,axis,next),Placing.value_segments(rest,Written.shifts_of(rest,axis,next)),segments_shifts(rest,axis,next))
    case Nil{}: {==}

# The segments of the cells of parts are those of their cells at the cells of
# their shifts.
def cells_segments(shapes: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,+lane: Nat) ->
  {Placing.cell_segments(shapes,shifts,lane,Grid.Innermost{}) == Placing.value_segments(Placing.cell_sources(shapes,lane,Grid.Innermost{}),cell_shift_lists(shifts,lane)) : List<&2,List<&2,List<&2,Tasks.Segment>>>}:
  match shapes shifts:
    case Con{+shape,+rest} Con{+shift,+later}:
      Equal.cong(List<&2,List<&2,List<&2,Tasks.Segment>>>,List<&2,List<&2,List<&2,Tasks.Segment>>>,others => Layout.placed_segments(Tiled.cell_shape(shape,lane,Grid.Innermost{}),Placing.cell_shifts(shift,lane,Grid.Innermost{})) <> others,
        Placing.cell_segments(rest,later,lane,Grid.Innermost{}),Placing.value_segments(Placing.cell_sources(rest,lane,Grid.Innermost{}),cell_shift_lists(later,lane)),cells_segments(rest,later,lane))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+shift,+later}: {==}
    case Con{+shape,+rest} Nil{}: {==}

# ---- The facts of every part ----

# What a placing of cells needs of a part: its lanes inside it or on the
# unit axis.
def SourceFact(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+source: List<&2,U32>,+shift: List<&2,U32>,+lane: Nat) -> Data:
  match within:
    case True{}:
      Placing.Lanes<limits,Machine.half_word(limits),Grid.lanes_at(values,lane),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),source,shift,lane,Grid.Innermost{}>
    case False{}: Placing.Units<limits,Machine.half_word(limits),List.length(&2,U32,values),source,shift,lane>

def SourcesFacts(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,shapes: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,+lane: Nat) -> Data:
  match within:
    case True{}:
      Placing.SourcesLanes(limits,Machine.half_word(limits),Grid.lanes_at(values,lane),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),shapes,shifts,lane,Grid.Innermost{})
    case False{}: Placing.SourcesUnits(limits,Machine.half_word(limits),List.length(&2,U32,values),shapes,shifts,lane)

def facts_cons(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+source: List<&2,U32>,+shift: List<&2,U32>,+shapes: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+lane: Nat,
  +head: SourceFact(within,limits,values,source,shift,lane),+rest: SourcesFacts(within,limits,values,shapes,shifts,lane)) ->
  SourcesFacts(within,limits,values,Con{source,shapes},Con{shift,shifts},lane):
  match within:
    case True{}: KernelGemm.Both{head,rest}
    case False{}: KernelGemm.Both{head,rest}

def facts_nil(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+lane: Nat) -> SourcesFacts(within,limits,values,[],[],lane):
  match within:
    case True{}: Unit{}
    case False{}: Unit{}

# A part that fits the joined values with its lanes kept and its cells placed
# brings what a placing of cells needs.
def part_fact(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32,+lane: Nat,
  +classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+lane_inside: {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +part_room: {Nat.is_le(ShapeLaws.elements(part_values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},+nonempty: {Shapes.has_zero(part_values) == False{} : Bool},
  +keeps: {SetSlice.kept_lanes(lane,U32.to_nat(axis),start,Layout.extent_at(part_values,U32.to_nat(axis)),Layout.extent_at(values,U32.to_nat(axis))) == True{} : Bool},
  +cell_placed: Facts.PlacedIn(Tiled.cell_shape(values,lane,Grid.Innermost{}),Tiled.cell_shape(part_values,lane,Grid.Innermost{}),Placing.cell_shifts(Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),lane,Grid.Innermost{}))) ->
  SourceFact(within,limits,values,part_values,Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),lane):
  match within:
    case True{}:
      +mark = U32.to_nat(axis)
      +shifts = Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0)
      +marked_fits = SetSlice.marked_of(limits,values,part_values,axis,start,rank_room,fits)
      +ranked = SetSlice.fits_length(values,part_values,mark,start,marked_fits)
      +mark_inside = SetSlice.fits_inside(values,part_values,mark,start,marked_fits)
      +part_mark_inside = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_lt(mark,_) == True{} : Bool} mark_inside :
        {Nat.is_lt(mark,List.length(&2,U32,part_values)) == True{} : Bool}}
      +source_within = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_lt(lane,_) == True{} : Bool} classified :
        {Nat.is_lt(lane,List.length(&2,U32,part_values)) == True{} : Bool}}
      +part_rank_room = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
        {Nat.is_le(List.length(&2,U32,part_values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
      +placed = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : Facts.PlacedIn(values,part_values,Layout.shifts_at(_,axis,start,0))
        Written.slice_in(values,part_values,axis,start,0,fits) : Facts.PlacedIn(values,part_values,shifts)}
      +part_positive = CountCompletion.positive_elements(part_values,nonempty)
      +keeps_lanes = {%SetSlice.extent_lanes(part_values,mark,part_mark_inside) : {SetSlice.kept_lanes(lane,mark,start,_,Grid.lanes_at(values,mark)) == True{} : Bool}
        %SetSlice.extent_lanes(values,mark,mark_inside) : {SetSlice.kept_lanes(lane,mark,start,Layout.extent_at(part_values,mark),_) == True{} : Bool}
        keeps : {SetSlice.kept_lanes(lane,mark,start,Grid.lanes_at(part_values,mark),Grid.lanes_at(values,mark)) == True{} : Bool}}
      +marks = SetSlice.shifts_marked(List.length(&2,U32,part_values),axis,start,0n,mark,Machine.capacity(limits),Limits.capacity_fits(limits),part_rank_room,{==})
      +lane_facts = {%Equal.sym(List<&2,U32>,shifts,SetSlice.marked(List.length(&2,U32,part_values),mark,start),marks) :
          KernelGemm.Both<{U32.is_zero((Grid.lanes_at(_,lane) % 8 : U32)) == True{} : Bool},Placing.Ending<Grid.lanes_at(part_values,lane),Grid.lanes_at(_,lane),Grid.lanes_at(values,lane)>>
        %Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) :
          KernelGemm.Both<{U32.is_zero((Grid.lanes_at(SetSlice.marked(_,mark,start),lane) % 8 : U32)) == True{} : Bool},
            Placing.Ending<Grid.lanes_at(part_values,lane),Grid.lanes_at(SetSlice.marked(_,mark,start),lane),Grid.lanes_at(values,lane)>>
        SetSlice.part_lane(Nat.is_eq(lane,mark),values,part_values,mark,start,lane,{==},marked_fits,classified,keeps_lanes) :
          KernelGemm.Both<{U32.is_zero((Grid.lanes_at(shifts,lane) % 8 : U32)) == True{} : Bool},Placing.Ending<Grid.lanes_at(part_values,lane),Grid.lanes_at(shifts,lane),Grid.lanes_at(values,lane)>>}
      SetSlice.source_lanes(limits,values,part_values,shifts,lane,Grid.Innermost{},classified,source_within,ranked,SetSlice.shifts_length(List.length(&2,U32,part_values),axis,start,0),placed,cell_placed,
        room,positive,Rooms.cells_room(limits,values,lane,Grid.Innermost{},positive,room),Rooms.cells_positive(limits,values,lane,Grid.Innermost{},positive,room),
        part_room,part_positive,Rooms.cells_room(limits,part_values,lane,Grid.Innermost{},part_positive,part_room),lane_facts)
    case False{}:
      +mark = U32.to_nat(axis)
      +shifts = Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0)
      +marked_fits = SetSlice.marked_of(limits,values,part_values,axis,start,rank_room,fits)
      +ranked = SetSlice.fits_length(values,part_values,mark,start,marked_fits)
      +past = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_lt(lane,_) == False{} : Bool} classified :
        {Nat.is_lt(lane,List.length(&2,U32,part_values)) == False{} : Bool}}
      +inside = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_le(lane,_) == True{} : Bool} lane_inside :
        {Nat.is_le(lane,List.length(&2,U32,part_values)) == True{} : Bool}}
      +placed = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : Facts.PlacedIn(values,part_values,Layout.shifts_at(_,axis,start,0))
        Written.slice_in(values,part_values,axis,start,0,fits) : Facts.PlacedIn(values,part_values,shifts)}
      SetSlice.source_units(limits,values,part_values,shifts,lane,past,inside,ranked,SetSlice.shifts_length(List.length(&2,U32,part_values),axis,start,0),placed,room,positive,part_room)

# ---- The cells of the parts one after another ----

# What the cells of a part bring.
def PartCells(+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32,+lane: Nat,+position: Grid.Position) -> Data:
  Held.Both<{Layout.slice_fits(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(part_values,lane,position),U32.from_nat(Tiled.cell_axis(position,U32.to_nat(axis),lane,List.length(&2,U32,values))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start),0) == True{} : Bool},Held.Both<{Shapes.has_zero(Tiled.cell_shape(part_values,lane,position)) == False{} : Bool},
    Held.Both<{Layout.shifts_at(List.length(&2,U32,Tiled.cell_shape(part_values,lane,position)),U32.from_nat(Tiled.cell_axis(position,U32.to_nat(axis),lane,List.length(&2,U32,values))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start),0) == Placing.cell_shifts(Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),lane,position) : List<&2,U32>},
      Facts.PlacedIn(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(part_values,lane,position),Placing.cell_shifts(Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),lane,position))>>>

def placed_of(+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32,+lane: Nat,+parts: PartCells(values,part_values,axis,start,lane,Grid.Innermost{})) ->
  Facts.PlacedIn(Tiled.cell_shape(values,lane,Grid.Innermost{}),Tiled.cell_shape(part_values,lane,Grid.Innermost{}),Placing.cell_shifts(Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),lane,Grid.Innermost{})):
  Held.Both{+fits,Held.Both{+nonempty,Held.Both{+shifts,+placed}}} = parts
  placed

# What the cells of parts from start hold: they fit the joined cells one after
# another at the cells of the shifts of the parts and end the joined cells,
# and every part brings what a placing of cells needs.
type Walked<-limits: Machine.Limits,-values: List<&2,U32>,-axis: U32,-lane: Nat,-shapes: List<&2,List<&2,U32>>,-start: U32> is Data:
  Walked{fit: Written.ShapesFit(Placing.cell_sources(shapes,lane,Grid.Innermost{}),Tiled.cell_shape(values,lane,Grid.Innermost{}),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start)),
    shifts: {Written.shifts_of(Placing.cell_sources(shapes,lane,Grid.Innermost{}),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start)) == cell_shift_lists(Written.shifts_of(shapes,axis,start),lane) : List<&2,List<&2,U32>>},
    total: {Written.survey_total(Placing.cell_sources(shapes,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start)) == Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32},
    facts: SourcesFacts(Nat.is_lt(lane,List.length(&2,U32,values)),limits,values,shapes,Written.shifts_of(shapes,axis,start),lane)}

def walked_more(+limits: Machine.Limits,+values: List<&2,U32>,+axis: U32,+start: U32,+lane: Nat,+part_values: List<&2,U32>,+next: List<&2,U32>,+rest: List<&2,List<&2,U32>>,
  +ahead: {(Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start) + Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32) == Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),((start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32) / 8 : U32),(start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32)) : U32},+parts: PartCells(values,part_values,axis,start,lane,Grid.Innermost{}),
  +head: SourceFact(Nat.is_lt(lane,List.length(&2,U32,values)),limits,values,part_values,Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),lane),+later: Walked<limits,values,axis,lane,Con{next,rest},(start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32)>) ->
  Walked<limits,values,axis,lane,Con{part_values,Con{next,rest}},start>:
  Held.Both{+cells_fits,Held.Both{+cells_nonempty,Held.Both{+cells_shifts,+cells_in}}} = parts
  Walked{+fit,+shifts,+total,+facts} = later
  +rest_cells = Placing.cell_sources(Con{next,rest},lane,Grid.Innermost{})
  +moved = (Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start) + Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32)
  +tail = Written.shifts_of(Con{next,rest},axis,(start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32))
  +own = Placing.cell_shifts(Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),lane,Grid.Innermost{})
  Walked{Held.Both{cells_fits,Held.Both{cells_nonempty,
      {%Equal.sym(U32,moved,Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),((start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32) / 8 : U32),(start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32)),ahead) : Written.ShapesFit(rest_cells,Tiled.cell_shape(values,lane,Grid.Innermost{}),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),_) fit : Written.ShapesFit(rest_cells,Tiled.cell_shape(values,lane,Grid.Innermost{}),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),moved)}}},
    {%Equal.sym(List<&2,U32>,Layout.shifts_at(List.length(&2,U32,Tiled.cell_shape(part_values,lane,Grid.Innermost{})),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start),0),own,cells_shifts) :
        {_ <> Written.shifts_of(rest_cells,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),moved) == own <> cell_shift_lists(tail,lane) : List<&2,List<&2,U32>>}
      %Equal.sym(U32,moved,Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),((start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32) / 8 : U32),(start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32)),ahead) : {own <> Written.shifts_of(rest_cells,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),_) == own <> cell_shift_lists(tail,lane) : List<&2,List<&2,U32>>}
      Equal.cong(List<&2,List<&2,U32>>,List<&2,List<&2,U32>>,others => own <> others,Written.shifts_of(rest_cells,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),((start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32) / 8 : U32),(start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32))),cell_shift_lists(tail,lane),shifts) :
      {Written.shifts_of(Placing.cell_sources(Con{part_values,Con{next,rest}},lane,Grid.Innermost{}),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start)) == cell_shift_lists(Written.shifts_of(Con{part_values,Con{next,rest}},axis,start),lane) :
        List<&2,List<&2,U32>>}},
    {%Equal.sym(U32,moved,Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),((start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32) / 8 : U32),(start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32)),ahead) : {Written.survey_total(rest_cells,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),_) == Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32} total :
      {Written.survey_total(rest_cells,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),moved) == Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32}},
    facts_cons(Nat.is_lt(lane,List.length(&2,U32,values)),limits,values,part_values,Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),Con{next,rest},tail,lane,head,facts)}

def walked_last(+limits: Machine.Limits,+values: List<&2,U32>,+axis: U32,+start: U32,+lane: Nat,+part_values: List<&2,U32>,
  +ends: {(Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start) + Layout.extent_at(Tiled.cell_shape(part_values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32) == Grid.lanes_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32},+parts: PartCells(values,part_values,axis,start,lane,Grid.Innermost{}),
  +head: SourceFact(Nat.is_lt(lane,List.length(&2,U32,values)),limits,values,part_values,Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),lane)) ->
  Walked<limits,values,axis,lane,[part_values],start>:
  Held.Both{+cells_fits,Held.Both{+cells_nonempty,Held.Both{+cells_shifts,+cells_in}}} = parts
  +within = Nat.is_lt(lane,List.length(&2,U32,values))
  Walked{Held.Both{cells_fits,Held.Both{cells_nonempty,Unit{}}},
    Equal.cong(List<&2,U32>,List<&2,List<&2,U32>>,shift => [shift],Layout.shifts_at(List.length(&2,U32,Tiled.cell_shape(part_values,lane,Grid.Innermost{})),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start),0),Placing.cell_shifts(Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),lane,Grid.Innermost{}),cells_shifts),
    ends,facts_cons(within,limits,values,part_values,Layout.shifts_at(List.length(&2,U32,part_values),axis,start,0),[],[],lane,head,facts_nil(within,limits,values,lane))}

# The cells of parts that fit the joined values one after another from start
# with their lanes kept and that end the joined values.
def cells_walk(+shapes: List<&2,List<&2,U32>>,+values: List<&2,U32>,+axis: U32,+start: U32,+lane: Nat,+limits: Machine.Limits,
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +lane_inside: {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool},+axis_back: {U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))) == Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)) : Nat},
  +more: {last_of(shapes) == False{} : Bool},+fit: Written.ShapesFit(shapes,values,axis,start),+kept: LanesKept(lane,U32.to_nat(axis),start,shapes),
  +total: {Written.survey_total(shapes,U32.to_nat(axis),start) == Layout.extent_at(values,U32.to_nat(axis)) : U32},+rooms: PartsRoom(limits,shapes)) ->
  Walked<limits,values,axis,lane,shapes,start>:
  match shapes:
    case Con{+part_values,+rest}:
      match rest:
        case Con{+next,+later}:
          Held.Both{+fits,Held.Both{+nonempty,+later_fit}} = fit
          Held.Both{+part_room,+later_rooms} = rooms
          +mark = U32.to_nat(axis)
          +marked_fits = SetSlice.marked_of(limits,values,part_values,axis,start,rank_room,fits)
          +ranked = SetSlice.fits_length(values,part_values,mark,start,marked_fits)
          +values_inside = SetSlice.fits_inside(values,part_values,mark,start,marked_fits)
          +mark_inside = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_lt(mark,_) == True{} : Bool} values_inside :
            {Nat.is_lt(mark,List.length(&2,U32,part_values)) == True{} : Bool}}
          +lane_room = Rooms.lane_room(limits,values,mark,values_inside,positive,room)
          +within = Written.slice_axis(True{},Grid.lanes_at(values,mark),Grid.lanes_at(part_values,mark),start,SetSlice.fits_hit(values,part_values,mark,start,marked_fits))
          +sum_room = {%Equal.sym(U32,Layout.extent_at(part_values,U32.to_nat(axis)),Grid.lanes_at(part_values,mark),SetSlice.extent_lanes(part_values,mark,mark_inside)) :
              {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(_)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
            Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(Grid.lanes_at(part_values,mark))),U32.to_nat(Grid.lanes_at(values,mark)),U32.to_nat(Machine.capacity(limits)),within,lane_room) :
            {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(Layout.extent_at(part_values,U32.to_nat(axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
          +keeps = part_keeps(Nat.is_eq(lane,mark),lane,mark,start,part_values,Con{next,later},Layout.extent_at(values,mark),{==},kept,total)
          +cells_marked = SetSlice.cell_fits(limits,values,part_values,mark,start,lane,Grid.Innermost{},Layout.extent_at(part_values,U32.to_nat(axis)),Layout.extent_at(values,mark),marked_fits,keeps,lane_room)
          +cell_inside = {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(part_values,lane,Grid.Innermost{})),List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),SetSlice.fits_length(Tiled.cell_shape(values,lane,Grid.Innermost{}),Tiled.cell_shape(part_values,lane,Grid.Innermost{}),Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start),cells_marked)) :
              {Nat.is_lt(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),_) == True{} : Bool}
            SetSlice.fits_inside(Tiled.cell_shape(values,lane,Grid.Innermost{}),Tiled.cell_shape(part_values,lane,Grid.Innermost{}),Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start),cells_marked) : {Nat.is_lt(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),List.length(&2,U32,Tiled.cell_shape(part_values,lane,Grid.Innermost{}))) == True{} : Bool}}
          +aligned = {%Order.equal_symmetric(lane,mark) : {Bool.not(_) || (U32.is_zero((start % 8 : U32)) && U32.is_zero((Layout.extent_at(part_values,U32.to_nat(axis)) % 8 : U32))) == True{} : Bool}
            kept_aligned(Nat.is_eq(lane,mark),lane,mark,start,part_values,next,later,{==},kept) :
            {Bool.not(Nat.is_eq(mark,lane)) || (U32.is_zero((start % 8 : U32)) && U32.is_zero((Layout.extent_at(part_values,U32.to_nat(axis)) % 8 : U32))) == True{} : Bool}}
          +parts = part_cells(limits,values,part_values,axis,start,lane,Grid.Innermost{},rank_room,room,positive,part_room,fits,nonempty,keeps)
          walked_more(limits,values,axis,start,lane,part_values,next,later,
            cells_next(limits,values,part_values,axis,start,lane,axis_back,ranked,mark_inside,cell_inside,aligned,sum_room),parts,
            part_fact(Nat.is_lt(lane,List.length(&2,U32,values)),limits,values,part_values,axis,start,lane,{==},lane_inside,rank_room,room,positive,part_room,fits,nonempty,keeps,
              placed_of(values,part_values,axis,start,lane,parts)),
            cells_walk(Con{next,later},values,axis,(start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32),lane,limits,rank_room,room,positive,lane_inside,axis_back,{==},later_fit,
              next_kept(Nat.is_eq(lane,mark),limits,lane,mark,start,part_values,next,later,{==},kept,sum_room),total,later_rooms))
        case Nil{}:
          Held.Both{+fits,Held.Both{+nonempty,+later_fit}} = fit
          Held.Both{+part_room,+later_rooms} = rooms
          +mark = U32.to_nat(axis)
          +marked_fits = SetSlice.marked_of(limits,values,part_values,axis,start,rank_room,fits)
          +ranked = SetSlice.fits_length(values,part_values,mark,start,marked_fits)
          +values_inside = SetSlice.fits_inside(values,part_values,mark,start,marked_fits)
          +mark_inside = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_lt(mark,_) == True{} : Bool} values_inside :
            {Nat.is_lt(mark,List.length(&2,U32,part_values)) == True{} : Bool}}
          +lane_room = Rooms.lane_room(limits,values,mark,values_inside,positive,room)
          +within = Written.slice_axis(True{},Grid.lanes_at(values,mark),Grid.lanes_at(part_values,mark),start,SetSlice.fits_hit(values,part_values,mark,start,marked_fits))
          +sum_room = {%Equal.sym(U32,Layout.extent_at(part_values,U32.to_nat(axis)),Grid.lanes_at(part_values,mark),SetSlice.extent_lanes(part_values,mark,mark_inside)) :
              {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(_)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
            Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(Grid.lanes_at(part_values,mark))),U32.to_nat(Grid.lanes_at(values,mark)),U32.to_nat(Machine.capacity(limits)),within,lane_room) :
            {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(Layout.extent_at(part_values,U32.to_nat(axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
          +keeps = part_keeps(Nat.is_eq(lane,mark),lane,mark,start,part_values,[],Layout.extent_at(values,mark),{==},kept,total)
          +cells_marked = SetSlice.cell_fits(limits,values,part_values,mark,start,lane,Grid.Innermost{},Layout.extent_at(part_values,U32.to_nat(axis)),Layout.extent_at(values,mark),marked_fits,keeps,lane_room)
          +cell_inside = {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(part_values,lane,Grid.Innermost{})),List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),SetSlice.fits_length(Tiled.cell_shape(values,lane,Grid.Innermost{}),Tiled.cell_shape(part_values,lane,Grid.Innermost{}),Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start),cells_marked)) :
              {Nat.is_lt(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),_) == True{} : Bool}
            SetSlice.fits_inside(Tiled.cell_shape(values,lane,Grid.Innermost{}),Tiled.cell_shape(part_values,lane,Grid.Innermost{}),Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(start / 8 : U32),start),cells_marked) : {Nat.is_lt(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),List.length(&2,U32,Tiled.cell_shape(part_values,lane,Grid.Innermost{}))) == True{} : Bool}}
          +aligned = {%Order.equal_symmetric(lane,mark) : {Bool.not(_) || U32.is_zero((start % 8 : U32)) == True{} : Bool}
            last_aligned(Nat.is_eq(lane,mark),lane,mark,start,part_values,{==},kept) : {Bool.not(Nat.is_eq(mark,lane)) || U32.is_zero((start % 8 : U32)) == True{} : Bool}}
          +parts = part_cells(limits,values,part_values,axis,start,lane,Grid.Innermost{},rank_room,room,positive,part_room,fits,nonempty,keeps)
          walked_last(limits,values,axis,start,lane,part_values,
            cells_last(limits,values,part_values,axis,start,lane,axis_back,ranked,mark_inside,cell_inside,aligned,total,sum_room),parts,
            part_fact(Nat.is_lt(lane,List.length(&2,U32,values)),limits,values,part_values,axis,start,lane,{==},lane_inside,rank_room,room,positive,part_room,fits,nonempty,keeps,
              placed_of(values,part_values,axis,start,lane,parts)))
    case Nil{}: Empty.absurd(Walked<limits,values,axis,lane,[],start>,WordEncoding.false_true(Equal.sym(Bool,True{},False{},more)))

# ---- Every cell of the joined cells lies in a part ----

# The digit at an axis.
def nth(digits: List<&2,Nat>,index: Nat) -> Nat:
  match digits index:
    case Con{+digit,+rest} 0n: digit
    case Con{+digit,+rest} 1n+ +next: nth(rest,next)
    case Nil{} +index: 0n

# Digits below their extents lie below the extent at every axis.
def nth_below(extents: List<&2,U32>,digits: List<&2,Nat>,index: Nat,+below: Coordinates.DigitsBelow(extents,digits)) ->
  {Nat.is_lt(nth(digits,index),U32.to_nat(Grid.lanes_at(extents,index))) == True{} : Bool}:
  match extents digits index:
    case Con{+extent,+rest} Con{+digit,+more} 0n:
      KernelGemm.Both{+small,+later} = below
      small
    case Con{+extent,+rest} Con{+digit,+more} 1n+ +next:
      KernelGemm.Both{+small,+later} = below
      nth_below(rest,more,next,later)
    case Nil{} Nil{} +index: {==}
    case Con{+extent,+rest} Nil{} +index: Empty.absurd({Nat.is_lt(nth(Nil{},index),U32.to_nat(Grid.lanes_at(Con{extent,rest},index))) == True{} : Bool},below)
    case Nil{} Con{+digit,+more} +index: Empty.absurd({Nat.is_lt(nth(Con{digit,more},index),U32.to_nat(Grid.lanes_at(Nil{},index))) == True{} : Bool},below)

# Digits below a target lie inside a part that fits it at the mark from start
# when the digit at the mark lies from start before the end of the part.
def inside_marked(target: List<&2,U32>,source: List<&2,U32>,digits: List<&2,Nat>,mark: Nat,+start: U32,
  +fits: {SetSlice.fits_marked(target,source,mark,start) == True{} : Bool},+below: Coordinates.DigitsBelow(target,digits),
  +low: {Nat.is_le(U32.to_nat(start),nth(digits,mark)) == True{} : Bool},
  +high: {Nat.is_lt(nth(digits,mark),Nat.add(U32.to_nat(start),U32.to_nat(Grid.lanes_at(source,mark)))) == True{} : Bool}) ->
  {Placing.inside(source,SetSlice.marked(List.length(&2,U32,source),mark,start),digits) == True{} : Bool}:
  match target source digits mark:
    case Con{+extent,+rest} Con{+width,+part_rest} Con{+digit,+more} 0n:
      KernelGemm.Both{+small,+later} = below
      +same = SetSlice.same_listed(rest,part_rest,Boolean.right(U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),SetSlice.same_extents(rest,part_rest),fits))
      Boolean.and_true(Placing.axis_inside(width,start,digit),Placing.inside(part_rest,SetSlice.zeros(List.length(&2,U32,part_rest)),more),
        Boolean.and_true(Nat.is_le(U32.to_nat(start),digit),Nat.is_lt(digit,Nat.add(U32.to_nat(start),U32.to_nat(width))),low,high),
        {%Equal.sym(List<&2,U32>,part_rest,rest,same) : {Placing.inside(_,SetSlice.zeros(List.length(&2,U32,_)),more) == True{} : Bool}
          SetSlice.inside_zeros(rest,more,later) : {Placing.inside(part_rest,SetSlice.zeros(List.length(&2,U32,part_rest)),more) == True{} : Bool}})
    case Con{+extent,+rest} Con{+width,+part_rest} Con{+digit,+more} 1n+ +next:
      KernelGemm.Both{+small,+later} = below
      +equal = Arithmetic.equal_of_is_eq(extent,width,Boolean.left(U32.is_eq(extent,width),SetSlice.fits_marked(rest,part_rest,next,start),fits))
      Boolean.and_true(Placing.axis_inside(width,0,digit),Placing.inside(part_rest,SetSlice.marked(List.length(&2,U32,part_rest),next,start),more),
        Boolean.and_true(Nat.is_le(0n,digit),Nat.is_lt(digit,U32.to_nat(width)),Order.zero_le(digit),
          {%equal : {Nat.is_lt(digit,U32.to_nat(_)) == True{} : Bool} small : {Nat.is_lt(digit,U32.to_nat(width)) == True{} : Bool}}),
        inside_marked(rest,part_rest,more,next,start,Boolean.right(U32.is_eq(extent,width),SetSlice.fits_marked(rest,part_rest,next,start),fits),later,low,high))
    case Con{+extent,+rest} Con{+width,+part_rest} Nil{} +mark:
      Empty.absurd({Placing.inside(Con{width,part_rest},SetSlice.marked(List.length(&2,U32,Con{width,part_rest}),mark,start),Nil{}) == True{} : Bool},below)
    case Con{+extent,+rest} Nil{} +digits +mark:
      Empty.absurd({Placing.inside(Nil{},SetSlice.marked(0n,mark,start),digits) == True{} : Bool},WordEncoding.false_true(fits))
    case Nil{} +source +digits +mark:
      Empty.absurd({Placing.inside(source,SetSlice.marked(List.length(&2,U32,source),mark,start),digits) == True{} : Bool},WordEncoding.false_true(fits))

# Some source placed at its shifts holds the words.
def plain_covers(sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,+words: List<&2,U32>) -> Bool:
  match sources shifts:
    case Con{+source,sources} Con{+shift,shifts}: Placing.inside_words(source,shift,words) || plain_covers(sources,shifts,words)
    case sources shifts: False{}

# The cells of some part hold the cell at the words when some part placed at
# the cells of its shifts does.
def covers_cells(shapes: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,+words: List<&2,U32>,+lane: Nat) ->
  {Placing.covers(shapes,shifts,words,lane,Grid.Innermost{}) == plain_covers(Placing.cell_sources(shapes,lane,Grid.Innermost{}),cell_shift_lists(shifts,lane),words) : Bool}:
  match shapes shifts:
    case Con{+shape,+rest} Con{+shift,+later}:
      Equal.cong(Bool,Bool,others => Placing.inside_words(Tiled.cell_shape(shape,lane,Grid.Innermost{}),Placing.cell_shifts(shift,lane,Grid.Innermost{}),words) || others,
        Placing.covers(rest,later,words,lane,Grid.Innermost{}),plain_covers(Placing.cell_sources(rest,lane,Grid.Innermost{}),cell_shift_lists(later,lane),words),covers_cells(rest,later,words,lane))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+shift,+later}: {==}
    case Con{+shape,+rest} Nil{}: {==}

# Parts that fit a target one after another from start and end it hold every
# value of the target at or past start along the axis.
def walk_covers(shapes: List<&2,List<&2,U32>>,+target: List<&2,U32>,+axis: U32,+start: U32,+limits: Machine.Limits,+word: U32,
  +rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(target)) == True{} : Bool},
  +fit: Written.ShapesFit(shapes,target,axis,start),+total: {Written.survey_total(shapes,U32.to_nat(axis),start) == Grid.lanes_at(target,U32.to_nat(axis)) : U32},
  +low: {Nat.is_le(U32.to_nat(start),nth(Targets.cell_coordinates(target,U32.to_nat(word)),U32.to_nat(axis))) == True{} : Bool}) ->
  {plain_covers(shapes,Written.shifts_of(shapes,axis,start),Grid.coordinates(target,word)) == True{} : Bool}:
  match shapes:
    case Con{+source,+rest}:
      Held.Both{+fits,Held.Both{+nonempty,+later}} = fit
      +mark = U32.to_nat(axis)
      +digits = Targets.cell_coordinates(target,U32.to_nat(word))
      +digit = nth(digits,mark)
      +words = Grid.coordinates(target,word)
      +shift = Layout.shifts_at(List.length(&2,U32,source),axis,start,0)
      +next = (start + Layout.extent_at(source,mark) : U32)
      +later_covers = plain_covers(rest,Written.shifts_of(rest,axis,next),words)
      +marked_fits = SetSlice.marked_of(limits,target,source,axis,start,rank_room,fits)
      +ranked = SetSlice.fits_length(target,source,mark,start,marked_fits)
      +target_inside = SetSlice.fits_inside(target,source,mark,start,marked_fits)
      +source_inside = {%Equal.sym(Nat,List.length(&2,U32,source),List.length(&2,U32,target),ranked) : {Nat.is_lt(mark,_) == True{} : Bool} target_inside :
        {Nat.is_lt(mark,List.length(&2,U32,source)) == True{} : Bool}}
      +source_rank_room = {%Equal.sym(Nat,List.length(&2,U32,source),List.length(&2,U32,target),ranked) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
        {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
      +placed = {%Equal.sym(Nat,List.length(&2,U32,source),List.length(&2,U32,target),ranked) : Facts.PlacedIn(target,source,Layout.shifts_at(_,axis,start,0))
        Written.slice_in(target,source,axis,start,0,fits) : Facts.PlacedIn(target,source,shift)}
      +exact = Placing.inside_nat(source,shift,words,Machine.half_word(limits),Limits.half_word_fits(limits),SetSlice.placed_room(target,source,shift,limits,placed,room,positive))
      +digits_exact = Digits.coordinates_exact(target,word,Machine.half_word(limits),Limits.half_word_fits(limits),Limits.double_bound(limits,ShapeLaws.elements(target),room),inside_word)
      +marks = SetSlice.shifts_marked(List.length(&2,U32,source),axis,start,0n,mark,Machine.capacity(limits),Limits.capacity_fits(limits),source_rank_room,{==})
      +width = Grid.lanes_at(source,mark)
      +within = Written.slice_axis(True{},Grid.lanes_at(target,mark),width,start,SetSlice.fits_hit(target,source,mark,start,marked_fits))
      +sum_room = Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(Grid.lanes_at(target,mark)),U32.to_nat(Machine.capacity(limits)),within,Rooms.lane_room(limits,target,mark,target_inside,positive,room))
      +extent_same = SetSlice.extent_lanes(source,mark,source_inside)
      +below = Digits.coordinates_below(target,U32.to_nat(word),inside_word)
      Held.by_cases({Placing.inside_words(source,shift,words) || later_covers == True{} : Bool},Nat.is_lt(digit,Nat.add(U32.to_nat(start),U32.to_nat(width))),
        +hit => {%Equal.sym(Bool,Placing.inside_words(source,shift,words),True{},
              Equal.trans(Bool,Placing.inside_words(source,shift,words),Placing.inside(source,shift,Digits.words_nat(words)),True{},exact,
                {%Equal.sym(List<&2,Nat>,Digits.words_nat(words),digits,digits_exact) : {Placing.inside(source,shift,_) == True{} : Bool}
                  %Equal.sym(List<&2,U32>,shift,SetSlice.marked(List.length(&2,U32,source),mark,start),marks) : {Placing.inside(source,_,digits) == True{} : Bool}
                  inside_marked(target,source,digits,mark,start,marked_fits,below,low,hit) :
                  {Placing.inside(source,shift,Digits.words_nat(words)) == True{} : Bool}})) :
            {_ || later_covers == True{} : Bool}
          {==} : {Placing.inside_words(source,shift,words) || later_covers == True{} : Bool}},
        +miss => {%Equal.sym(Bool,later_covers,True{},
              walk_covers(rest,target,axis,next,limits,word,rank_room,room,positive,inside_word,later,total,
                {%Equal.sym(Nat,U32.to_nat(next),Nat.add(U32.to_nat(start),U32.to_nat(Layout.extent_at(source,mark))),
                    FullRange.add(start,Layout.extent_at(source,mark),Machine.capacity(limits),
                      {%Equal.sym(U32,Layout.extent_at(source,mark),width,extent_same) : {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(_)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} sum_room :
                        {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(Layout.extent_at(source,mark))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})) :
                    {Nat.is_le(_,digit) == True{} : Bool}
                  %Equal.sym(U32,Layout.extent_at(source,mark),width,extent_same) : {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(_)),digit) == True{} : Bool}
                  Order.not_less(digit,Nat.add(U32.to_nat(start),U32.to_nat(width)),miss) : {Nat.is_le(U32.to_nat(next),digit) == True{} : Bool}})) :
            {Placing.inside_words(source,shift,words) || _ == True{} : Bool}
          Boolean.or_true_right(Placing.inside_words(source,shift,words)) : {Placing.inside_words(source,shift,words) || later_covers == True{} : Bool}})
    case Nil{}:
      +mark = U32.to_nat(axis)
      +digit = nth(Targets.cell_coordinates(target,U32.to_nat(word)),mark)
      +below = {%Equal.sym(U32,start,Grid.lanes_at(target,mark),total) : {Nat.is_lt(digit,U32.to_nat(_)) == True{} : Bool}
        nth_below(target,Targets.cell_coordinates(target,U32.to_nat(word)),mark,Digits.coordinates_below(target,U32.to_nat(word),inside_word)) :
        {Nat.is_lt(digit,U32.to_nat(start)) == True{} : Bool}}
      Empty.absurd({plain_covers([],Written.shifts_of([],axis,start),Grid.coordinates(target,word)) == True{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(digit,U32.to_nat(start)),True{},Equal.sym(Bool,Nat.is_lt(digit,U32.to_nat(start)),False{},Order.not_below(U32.to_nat(start),digit,low)),below)))

# ---- The cells of the parts as inputs ----

# The cells of every part arranged on the lane axis, as the inputs of a
# concatenation of cells.
type CellParts<-Part: Data,-part_value: Part -> U32 -> F32,-logicals: List<&2,Part>,-shapes: List<&2,List<&2,U32>>,-model: PartsModel,-lane: Nat> is Data:
  CellParts{inputs: Written.Inputs<Tile.Tile8,Representation.Lifted<Part>>,
    parts: {Written.parts_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs) == TiledConcat.cells(parts_of(model),lane) : Tensors.Parts<Tile.Tile8>},
    held: Written.InputsHeld(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),inputs),
    sources: {Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs) == Placing.cell_sources(shapes,lane,Grid.Innermost{}) : List<&2,List<&2,U32>>},
    named: {Written.logicals_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs) == Placing.lifted_logicals(~Part,logicals,shapes,lane,Grid.Innermost{}) : List<&2,Representation.Lifted<Part>>}}

def single_cells(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+part_values: List<&2,U32>,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,
  +pieces: PiecesModel.Model<Tile.Tile8>,+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+own: Nat,+lanes: U32,+position: Grid.Position,+lane: Nat,
  +named: {logicals == [] : List<&2,Part>},+last: {shapes == [] : List<&2,List<&2,U32>>},
  +shaped: Public.ShapedLaid(~Part,~part_value,part,part_values,Tiled.Tiled{Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},lane,Grid.Innermost{})),lane,Grid.lanes_at(part_values,lane),Grid.Innermost{}})) ->
  CellParts<Part,part_value,Con{part,logicals},Con{part_values,shapes},SingleModel{PartModel{pieces,cells_shape,fill,workers,own,lanes,position}},lane>:
  KernelGemm.Both{KernelGemm.Both{Global.Usable{+cell_model,+cell_fill,+cell_workers,+equation},+global},+laid} = shaped
  +lifted = {Representation.Lifted{part,Public.geometry_of(part_values,lane,Grid.Innermost{})} : Representation.Lifted<Part>}
  +cells = Tiled.cell_shape(part_values,lane,Grid.Innermost{})
  CellParts{Written.Last{lifted,cell_model,cells,cell_fill,cell_workers},
    Equal.cong(Tensors.Tensor<Tile.Tile8>,Tensors.Parts<Tile.Tile8>,tensor => Tensors.Single{tensor},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,cell_model),cells,cell_fill,Tensors.Usable{},cell_workers},Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},lane,Grid.Innermost{})),equation),
    {%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,cell_model),cells,cell_fill,Tensors.Usable{},cell_workers},Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},lane,Grid.Innermost{})),equation) : Global.Contract(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),lifted,_) global :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),lifted,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,cell_model),cells,cell_fill,Tensors.Usable{},cell_workers})},
    {%Equal.sym(List<&2,List<&2,U32>>,shapes,[],last) : {[cells] == cells <> Placing.cell_sources(_,lane,Grid.Innermost{}) : List<&2,List<&2,U32>>} {==} :
      {[cells] == cells <> Placing.cell_sources(shapes,lane,Grid.Innermost{}) : List<&2,List<&2,U32>>}},
    {%Equal.sym(List<&2,Part>,logicals,[],named) : {[lifted] == lifted <> Placing.lifted_logicals(~Part,_,shapes,lane,Grid.Innermost{}) : List<&2,Representation.Lifted<Part>>}
      %Equal.sym(List<&2,List<&2,U32>>,shapes,[],last) : {[lifted] == lifted <> Placing.lifted_logicals(~Part,[],_,lane,Grid.Innermost{}) : List<&2,Representation.Lifted<Part>>}
      {==} : {[lifted] == lifted <> Placing.lifted_logicals(~Part,logicals,shapes,lane,Grid.Innermost{}) : List<&2,Representation.Lifted<Part>>}}}

def cons_cells(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+part_values: List<&2,U32>,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,
  +pieces: PiecesModel.Model<Tile.Tile8>,+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+own: Nat,+lanes: U32,+position: Grid.Position,+rest: PartsModel,+lane: Nat,
  +shaped: Public.ShapedLaid(~Part,~part_value,part,part_values,Tiled.Tiled{Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},lane,Grid.Innermost{})),lane,Grid.lanes_at(part_values,lane),Grid.Innermost{}}),
  +later: CellParts<Part,part_value,logicals,shapes,rest,lane>) ->
  CellParts<Part,part_value,Con{part,logicals},Con{part_values,shapes},ConsModel{PartModel{pieces,cells_shape,fill,workers,own,lanes,position},rest},lane>:
  KernelGemm.Both{KernelGemm.Both{Global.Usable{+cell_model,+cell_fill,+cell_workers,+equation},+global},+laid} = shaped
  CellParts{+rest_inputs,+rest_parts,+rest_held,+rest_sources,+rest_named} = later
  +lifted = {Representation.Lifted{part,Public.geometry_of(part_values,lane,Grid.Innermost{})} : Representation.Lifted<Part>}
  +cells = Tiled.cell_shape(part_values,lane,Grid.Innermost{})
  CellParts{Written.More{lifted,cell_model,cells,cell_fill,cell_workers,rest_inputs},
    {%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,cell_model),cells,cell_fill,Tensors.Usable{},cell_workers},Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},lane,Grid.Innermost{})),equation) : {Tensors.Cons{_,Written.parts_of(~Tile.Tile8,~Representation.Lifted<Part>,rest_inputs)} == Tensors.Cons{Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},lane,Grid.Innermost{})),TiledConcat.cells(parts_of(rest),lane)} : Tensors.Parts<Tile.Tile8>}
      Equal.cong(Tensors.Parts<Tile.Tile8>,Tensors.Parts<Tile.Tile8>,later_parts => Tensors.Cons{Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},lane,Grid.Innermost{})),later_parts},Written.parts_of(~Tile.Tile8,~Representation.Lifted<Part>,rest_inputs),TiledConcat.cells(parts_of(rest),lane),rest_parts) :
      {Tensors.Cons{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,cell_model),cells,cell_fill,Tensors.Usable{},cell_workers},Written.parts_of(~Tile.Tile8,~Representation.Lifted<Part>,rest_inputs)} == Tensors.Cons{Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},lane,Grid.Innermost{})),TiledConcat.cells(parts_of(rest),lane)} : Tensors.Parts<Tile.Tile8>}},
    Held.Both{{%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,cell_model),cells,cell_fill,Tensors.Usable{},cell_workers},Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},lane,Grid.Innermost{})),equation) : Global.Contract(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),lifted,_) global :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),lifted,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,cell_model),cells,cell_fill,Tensors.Usable{},cell_workers})},rest_held},
    Equal.cong(List<&2,List<&2,U32>>,List<&2,List<&2,U32>>,later_shapes => cells <> later_shapes,Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,rest_inputs),
      Placing.cell_sources(shapes,lane,Grid.Innermost{}),rest_sources),
    Equal.cong(List<&2,Representation.Lifted<Part>>,List<&2,Representation.Lifted<Part>>,later_named => lifted <> later_named,Written.logicals_of(~Tile.Tile8,~Representation.Lifted<Part>,rest_inputs),
      Placing.lifted_logicals(~Part,logicals,shapes,lane,Grid.Innermost{}),rest_named)}

# The cells of parts of the contract that fit the joined values, arranged on
# the lane axis, as inputs of the contract of their lifted values.
def cell_parts(~Part: Data,~part_value: Part -> U32 -> F32,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,+model: PartsModel,+values: List<&2,U32>,+axis: U32,+start: U32,
  +lane: Nat,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+lane_inside: {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool},
  +fit: Written.ShapesFit(shapes,values,axis,start),+held: PartsHeld(~Part,~part_value,logicals,shapes,parts_of(model))) ->
  CellParts<Part,part_value,logicals,shapes,model,lane>:
  match logicals shapes:
    case Con{+part,+logicals} Con{+part_values,+shapes}:
      match model:
        case SingleModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position}}:
          Held.Both{+fits,Held.Both{+nonempty,+later}} = fit
          Held.Both{+contract,Held.Both{+named,+last}} = held
          +ranked = SetSlice.fits_length(values,part_values,U32.to_nat(axis),start,SetSlice.marked_of(limits,values,part_values,axis,start,rank_room,fits))
          single_cells(~Part,~part_value,part,part_values,logicals,shapes,pieces,cells_shape,fill,workers,own,lanes,position,lane,named,last,
            Relane.arranged_cells(~Part,~part_value,part,part_values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},lane,Grid.Innermost{},limits,default_capacity,
              {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_le(lane,_) == True{} : Bool} lane_inside :
                {Nat.is_le(lane,List.length(&2,U32,part_values)) == True{} : Bool}},
              CountCompletion.positive_elements(part_values,nonempty),
              {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
                {Nat.is_le(List.length(&2,U32,part_values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
              contract))
        case ConsModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position},+rest}:
          Held.Both{+fits,Held.Both{+nonempty,+later}} = fit
          Held.Both{+contract,+held_rest} = held
          +ranked = SetSlice.fits_length(values,part_values,U32.to_nat(axis),start,SetSlice.marked_of(limits,values,part_values,axis,start,rank_room,fits))
          cons_cells(~Part,~part_value,part,part_values,logicals,shapes,pieces,cells_shape,fill,workers,own,lanes,position,rest,lane,
            Relane.arranged_cells(~Part,~part_value,part,part_values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,pieces),cells_shape,fill,Tensors.Usable{},workers},own,lanes,position},lane,Grid.Innermost{},limits,default_capacity,
              {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_le(lane,_) == True{} : Bool} lane_inside :
                {Nat.is_le(lane,List.length(&2,U32,part_values)) == True{} : Bool}},
              CountCompletion.positive_elements(part_values,nonempty),
              {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
                {Nat.is_le(List.length(&2,U32,part_values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
              contract),
            cell_parts(~Part,~part_value,logicals,shapes,rest,values,axis,(start + Layout.extent_at(part_values,U32.to_nat(axis)) : U32),lane,limits,default_capacity,rank_room,lane_inside,later,held_rest))
    case Con{+part,+logicals} Nil{}: Empty.absurd(CellParts<Part,part_value,Con{part,logicals},[],model,lane>,held)
    case Nil{} +shapes: Empty.absurd(CellParts<Part,part_value,[],shapes,model,lane>,held)

# The first shape of inputs heads their shapes.
def head_of(shapes: List<&2,List<&2,U32>>) -> List<&2,U32>:
  match shapes:
    case Con{+shape,+rest}: shape
    case Nil{}: []

def first_head(~Element: Data,~Part: Data,+inputs: Written.Inputs<Element,Part>) ->
  {Written.first_shape(~Element,~Part,inputs) == head_of(Written.shapes_of(~Element,~Part,inputs)) : List<&2,U32>}:
  match inputs:
    case Written.Last{+part,+model,+shape,+fill,+workers}: {==}
    case Written.More{+part,+model,+shape,+fill,+workers,+rest}: {==}

def sources_length(shapes: List<&2,List<&2,U32>>,+lane: Nat) ->
  {List.length(&2,List<&2,U32>,Placing.cell_sources(shapes,lane,Grid.Innermost{})) == List.length(&2,List<&2,U32>,shapes) : Nat}:
  match shapes:
    case Con{+shape,+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,List<&2,U32>,Placing.cell_sources(rest,lane,Grid.Innermost{})),List.length(&2,List<&2,U32>,rest),sources_length(rest,lane))
    case Nil{}: {==}

# A part that fits a target at the mark, given the extent of the target
# there, is the target.
def marked_set(target: List<&2,U32>,part: List<&2,U32>,mark: Nat,+start: U32,+fits: {SetSlice.fits_marked(target,part,mark,start) == True{} : Bool}) ->
  {Layout.with_extent(part,mark,Grid.lanes_at(target,mark)) == target : List<&2,U32>}:
  match target part mark:
    case Con{+extent,+rest} Con{+width,+part_rest} 0n:
      %SetSlice.same_listed(rest,part_rest,Boolean.right(U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),SetSlice.same_extents(rest,part_rest),fits)) :
        {extent <> part_rest == extent <> _ : List<&2,U32>}
      {==}
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +next:
      %Arithmetic.equal_of_is_eq(extent,width,Boolean.left(U32.is_eq(extent,width),SetSlice.fits_marked(rest,part_rest,next,start),fits)) :
        {_ <> Layout.with_extent(part_rest,next,Grid.lanes_at(rest,next)) == extent <> rest : List<&2,U32>}
      %marked_set(rest,part_rest,next,start,Boolean.right(U32.is_eq(extent,width),SetSlice.fits_marked(rest,part_rest,next,start),fits)) :
        {extent <> Layout.with_extent(part_rest,next,Grid.lanes_at(rest,next)) == extent <> _ : List<&2,U32>}
      {==}
    case Con{+extent,+rest} Nil{} +mark:
      Empty.absurd({Layout.with_extent(Nil{},mark,Grid.lanes_at(Con{extent,rest},mark)) == Con{extent,rest} : List<&2,U32>},WordEncoding.false_true(fits))
    case Nil{} +part +mark:
      Empty.absurd({Layout.with_extent(part,mark,Grid.lanes_at(Nil{},mark)) == Nil{} : List<&2,U32>},WordEncoding.false_true(fits))

# ---- The first part ----

def both_left(~First: Data,~Second: Data,+both: Held.Both<First,Second>) -> First:
  Held.Both{+left,+right} = both
  left

def both_right(~First: Data,~Second: Data,+both: Held.Both<First,Second>) -> Second:
  Held.Both{+left,+right} = both
  right

# The extents and the lane axis of the first part of a model.
def first_logical(+model: PartsModel) -> List<&2,U32>:
  match model:
    case SingleModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position}}: Tiled.logical_shape(cells_shape,own,lanes,position)
    case ConsModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position},+rest}: Tiled.logical_shape(cells_shape,own,lanes,position)

def first_lane(+model: PartsModel) -> Nat:
  match model:
    case SingleModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position}}: own
    case ConsModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position},+rest}: own

# The runtime leads with the extents and the lane axis of the first part.
def lead_model(+model: PartsModel) -> {TiledConcat.lead(parts_of(model)) == (parts_of(model),TiledConcat.Lead{first_logical(model),first_lane(model)}) : TiledConcat.Parts & TiledConcat.Lead}:
  match model:
    case SingleModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position}}: {==}
    case ConsModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position},+rest}: {==}

def first_values(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+logicals: List<&2,Part>,+first: List<&2,U32>,+shapes: List<&2,List<&2,U32>>,+model: PartsModel,
  +held: PartsHeld(~Part,~part_value,Con{part,logicals},Con{first,shapes},parts_of(model))) -> {first_logical(model) == first : List<&2,U32>}:
  match model:
    case SingleModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position}}:
      Held.Both{KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}},+ends} = held
      restored
    case ConsModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position},+rest}:
      Held.Both{KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}},+later} = held
      restored

def first_lane_inside(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+logicals: List<&2,Part>,+first: List<&2,U32>,+shapes: List<&2,List<&2,U32>>,+model: PartsModel,
  +held: PartsHeld(~Part,~part_value,Con{part,logicals},Con{first,shapes},parts_of(model))) -> {Nat.is_le(first_lane(model),List.length(&2,U32,first)) == True{} : Bool}:
  match model:
    case SingleModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position}}:
      Held.Both{KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}},+ends} = held
      inside
    case ConsModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position},+rest}:
      Held.Both{KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}},+later} = held
      inside

# Every part has a logical.
def held_named(~Part: Data,~part_value: Part -> U32 -> F32,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,+model: PartsModel,
  +held: PartsHeld(~Part,~part_value,logicals,shapes,parts_of(model))) -> {List.length(&2,Part,logicals) == List.length(&2,List<&2,U32>,shapes) : Nat}:
  match logicals shapes:
    case Con{+part,+logicals} Con{+values,+shapes}:
      match model:
        case SingleModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position}}:
          Held.Both{+contract,Held.Both{+named,+last}} = held
          %Equal.sym(List<&2,Part>,logicals,[],named) : {1n+List.length(&2,Part,_) == 1n+List.length(&2,List<&2,U32>,shapes) : Nat}
          %Equal.sym(List<&2,List<&2,U32>>,shapes,[],last) : {1n+List.length(&2,Part,[]) == 1n+List.length(&2,List<&2,U32>,_) : Nat}
          {==}
        case ConsModel{PartModel{+pieces,+cells_shape,+fill,+workers,+own,+lanes,+position},+rest}:
          Held.Both{+contract,+held_rest} = held
          Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Part,logicals),List.length(&2,List<&2,U32>,shapes),held_named(~Part,~part_value,logicals,shapes,rest,held_rest))
    case Con{+part,+logicals} Nil{}: Empty.absurd({List.length(&2,Part,Con{part,logicals}) == 0n : Nat},held)
    case Nil{} +shapes: Empty.absurd({0n == List.length(&2,List<&2,U32>,shapes) : Nat},held)

# The lanes of the joined values on the lane axis: the joined extent on the
# axis, those of the first part elsewhere.
def joined_lanes(same: Bool,+first: List<&2,U32>,+mark: Nat,+lane: Nat,+total: U32,+classified: {Nat.is_eq(mark,lane) == same : Bool},
  +inside: {Nat.is_lt(mark,List.length(&2,U32,first)) == True{} : Bool},+fits: {SetSlice.fits_marked(Layout.with_extent(first,mark,total),first,mark,0) == True{} : Bool}) ->
  {Grid.lanes_at(Layout.with_extent(first,mark,total),lane) == Bool.pick(U32,same,total,Grid.lanes_at(first,lane)) : U32}:
  match same:
    case True{}:
      %Order.equal_of_is_eq(mark,lane,classified) : {Grid.lanes_at(Layout.with_extent(first,mark,total),_) == total : U32}
      set_lane(first,mark,total,inside)
    case False{}:
      SetSlice.fits_other(Layout.with_extent(first,mark,total),first,mark,lane,0,fits,
        Equal.trans(Bool,Nat.is_eq(lane,mark),Nat.is_eq(mark,lane),False{},Order.equal_symmetric(lane,mark),classified))

# ---- The joined cells hold the joined values ----

# Cells that hold the lifted placing of the cells of the parts hold the lifted
# values of the parts placed one after another.
def restate(~Part: Data,~part_value: Part -> U32 -> F32,within: Bool,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,+values: List<&2,U32>,+axis: U32,+lane: Nat,
  +limits: Machine.Limits,+fill: Tile.Tile8,+classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},
  +inside: {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +named: {List.length(&2,Part,logicals) == List.length(&2,List<&2,U32>,shapes) : Nat},
  +facts: SourcesFacts(within,limits,values,shapes,Written.shifts_of(shapes,axis,0),lane),
  covered: @+word: U32 -> @+inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,lane,Grid.Innermost{}))) == True{} : Bool} ->
    {Placing.covers(shapes,Written.shifts_of(shapes,axis,0),Grid.coordinates(Tiled.cell_shape(values,lane,Grid.Innermost{}),word),lane,Grid.Innermost{}) == True{} : Bool},
  -tensor: Tensors.Tensor<Tile.Tile8>,
  +shaped: Global.Shaped(~Tile.Tile8,~Gathered.Placings<Representation.Lifted<Part>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),Gathered.Placings{Placing.lifted_logicals(~Part,logicals,shapes,lane,Grid.Innermost{}),
    Placing.cell_segments(shapes,Written.shifts_of(shapes,axis,0),lane,Grid.Innermost{}),Placing.cell_sources(shapes,lane,Grid.Innermost{}),Tiled.cell_shape(values,lane,Grid.Innermost{}),fill},Tiled.cell_shape(values,lane,Grid.Innermost{}),tensor)) ->
  Global.Shaped(~Tile.Tile8,~Representation.Lifted<Gathered.Placings<Part,F32>>,~Representation.lifted(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value)),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(shapes,Written.shifts_of(shapes,axis,0)),shapes,values,0.0},
    Public.geometry_of(values,lane,Grid.Innermost{})},Tiled.cell_shape(values,lane,Grid.Innermost{}),tensor):
  match within:
    case True{}:
      +cells = Tiled.cell_shape(values,lane,Grid.Innermost{})
      +shifts = Written.shifts_of(shapes,axis,0)
      +cells_space = Rooms.cells_room(limits,values,lane,Grid.Innermost{},positive,room)
      +target = {Placing.Target{Limits.half_word_fits(limits),inside,Limits.double_bound(limits,ShapeLaws.elements(cells),cells_space),Limits.double_bound(limits,ShapeLaws.elements(values),room),
          Rooms.removed_room(limits,values,lane,classified,positive,room)} : Placing.Target<limits,Machine.half_word(limits),values,lane,Grid.Innermost{}>}
      Global.restated_below(~Tile.Tile8,~Gathered.Placings<Representation.Lifted<Part>,Tile.Tile8>,~Representation.Lifted<Gathered.Placings<Part,F32>>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),~Representation.lifted(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value)),~ShapeLaws.elements(cells),
        Gathered.Placings{Placing.lifted_logicals(~Part,logicals,shapes,lane,Grid.Innermost{}),Placing.cell_segments(shapes,shifts,lane,Grid.Innermost{}),Placing.cell_sources(shapes,lane,Grid.Innermost{}),cells,fill},
        Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(shapes,shifts),shapes,values,0.0},Public.geometry_of(values,lane,Grid.Innermost{})},
        index => below => Placing.cell_within(~Part,~part_value,logicals,shapes,shifts,limits,Machine.half_word(limits),values,lane,Grid.Innermost{},fill,0.0,index,target,classified,named,facts,below,
          covered(index,below)),
        cells,tensor,shaped,Order.reflexive(ShapeLaws.elements(cells)),Machine.capacity(limits),cells_space)
    case False{}:
      +cells = Tiled.cell_shape(values,lane,Grid.Innermost{})
      +shifts = Written.shifts_of(shapes,axis,0)
      +cells_space = Rooms.cells_room(limits,values,lane,Grid.Innermost{},positive,room)
      +target = {Placing.Target{Limits.half_word_fits(limits),inside,Limits.double_bound(limits,ShapeLaws.elements(cells),cells_space),Limits.double_bound(limits,ShapeLaws.elements(values),room),
          {%Equal.sym(List<&2,U32>,Grid.removed(values,lane),values,Public.removed_past(values,lane,classified)) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
            room : {Nat.is_le(ShapeLaws.elements(Grid.removed(values,lane)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}} :
        Placing.Target<limits,Machine.half_word(limits),values,lane,Grid.Innermost{}>}
      Global.restated_below(~Tile.Tile8,~Gathered.Placings<Representation.Lifted<Part>,Tile.Tile8>,~Representation.Lifted<Gathered.Placings<Part,F32>>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),~Representation.lifted(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value)),~ShapeLaws.elements(cells),
        Gathered.Placings{Placing.lifted_logicals(~Part,logicals,shapes,lane,Grid.Innermost{}),Placing.cell_segments(shapes,shifts,lane,Grid.Innermost{}),Placing.cell_sources(shapes,lane,Grid.Innermost{}),cells,fill},
        Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(shapes,shifts),shapes,values,0.0},Public.geometry_of(values,lane,Grid.Innermost{})},
        index => below => Placing.cell_unit(~Part,~part_value,logicals,shapes,shifts,limits,Machine.half_word(limits),values,lane,Grid.Innermost{},fill,0.0,index,target,classified,named,facts,below,
          covered(index,below)),
        cells,tensor,shaped,Order.reflexive(ShapeLaws.elements(cells)),Machine.capacity(limits),cells_space)

# The cells of the parts concatenated along the cell axis hold the joined
# values.
def concat_assembled(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+others: List<&2,Part>,+first: List<&2,U32>,+rest: List<&2,List<&2,U32>>,+model: PartsModel,
  +values: List<&2,U32>,+axis: U32,+lane: Nat,+lanes: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+parts_count: {Nat.is_le(List.length(&2,List<&2,U32>,Con{first,rest}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +lane_inside: {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(values) == Some{count} : Maybe<&2,U32>},+pinned: {Grid.lanes_at(values,lane) == lanes : U32},
  +named: {List.length(&2,Part,Con{part,others}) == List.length(&2,List<&2,U32>,Con{first,rest}) : Nat},
  +walked: Walked<limits,values,axis,lane,Con{first,rest},0>,+cells: CellParts<Part,part_value,Con{part,others},Con{first,rest},model,lane>) ->
  Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{Con{part,others},Written.segments_of(Con{first,rest},axis,0),Con{first,rest},values,0.0},values,Tiled.Tiled{Layout.concat(~Tile.Tile8,TiledConcat.cells(parts_of(model),lane),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),lane,lanes,Grid.Innermost{}}):
  Walked{+walk_fit,+walk_shifts,+walk_total,+walk_facts} = walked
  Held.Both{+first_cells_fits,Held.Both{+first_cells_nonempty,+later_cells}} = walk_fit
  CellParts{+inputs,+parts_eq,+cells_held,+sources_eq,+named_eq} = cells
  +cv = Tiled.cell_shape(values,lane,Grid.Innermost{})
  +cw = Tiled.cell_shape(first,lane,Grid.Innermost{})
  +csources = Placing.cell_sources(Con{first,rest},lane,Grid.Innermost{})
  +shifts0 = Written.shifts_of(Con{first,rest},axis,0)
  +cst0 = {Written.same_pick(Nat.is_eq(U32.to_nat(axis),lane),0) : {Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(0 / 8 : U32),0) == 0 : U32}}
  +fit0 = {%cst0 : Written.ShapesFit(csources,cv,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),_) walk_fit : Written.ShapesFit(csources,cv,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0)}
  +shifts0_eq = {%cst0 : {Written.shifts_of(csources,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),_) == cell_shift_lists(shifts0,lane) : List<&2,List<&2,U32>>} walk_shifts :
    {Written.shifts_of(csources,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0) == cell_shift_lists(shifts0,lane) : List<&2,List<&2,U32>>}}
  +total0 = {%cst0 : {Written.survey_total(csources,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),_) == Grid.lanes_at(cv,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32} walk_total : {Written.survey_total(csources,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),0) == Grid.lanes_at(cv,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) : U32}}
  +cells_rank_room = {%Equal.sym(Nat,List.length(&2,U32,cv),List.length(&2,U32,values),Rooms.cell_rank(values,lane,Grid.Innermost{})) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    rank_room : {Nat.is_le(List.length(&2,U32,cv),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +cells_space = Rooms.cells_room(limits,values,lane,Grid.Innermost{},positive,room)
  +cells_filled = Rooms.cells_positive(limits,values,lane,Grid.Innermost{},positive,room)
  +cells_counted = StorageCount.counted_within(limits,cv,cells_space,default_capacity)
  +first_marked = SetSlice.marked_of(limits,cv,cw,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(0 / 8 : U32),0),cells_rank_room,first_cells_fits)
  +first_length = SetSlice.fits_length(cv,cw,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(0 / 8 : U32),0),first_marked)
  +icm_inside = {%Equal.sym(Nat,List.length(&2,U32,cw),List.length(&2,U32,cv),first_length) : {Nat.is_lt(U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),_) == True{} : Bool}
    SetSlice.fits_inside(cv,cw,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(0 / 8 : U32),0),first_marked) : {Nat.is_lt(U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),List.length(&2,U32,cw)) == True{} : Bool}}
  +first_eq = Equal.trans(List<&2,U32>,Written.first_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs),head_of(Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs)),cw,first_head(~Tile.Tile8,~Representation.Lifted<Part>,inputs),
    Equal.cong(List<&2,List<&2,U32>>,List<&2,U32>,listed => head_of(listed),Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),csources,sources_eq))
  +joined_eq = {%Equal.sym(List<&2,U32>,Written.first_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs),cw,first_eq) : {Layout.with_extent(_,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),Written.survey_total(Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),0)) == cv : List<&2,U32>}
    %Equal.sym(List<&2,List<&2,U32>>,Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),csources,sources_eq) : {Layout.with_extent(cw,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),Written.survey_total(_,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),0)) == cv : List<&2,U32>}
    %Equal.sym(U32,Written.survey_total(csources,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),0),Grid.lanes_at(cv,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))),total0) : {Layout.with_extent(cw,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),_) == cv : List<&2,U32>}
    marked_set(cv,cw,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(0 / 8 : U32),0),first_marked) : {Written.joined_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))) == cv : List<&2,U32>}}
  +segments_eq = Equal.trans(List<&2,List<&2,List<&2,Tasks.Segment>>>,Written.segments_of(csources,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0),Placing.value_segments(csources,Written.shifts_of(csources,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0)),Placing.cell_segments(Con{first,rest},shifts0,lane,Grid.Innermost{}),
    segments_shifts(csources,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0),
    Equal.trans(List<&2,List<&2,List<&2,Tasks.Segment>>>,Placing.value_segments(csources,Written.shifts_of(csources,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0)),Placing.value_segments(csources,cell_shift_lists(shifts0,lane)),Placing.cell_segments(Con{first,rest},shifts0,lane,Grid.Innermost{}),
      Equal.cong(List<&2,List<&2,U32>>,List<&2,List<&2,List<&2,Tasks.Segment>>>,listed => Placing.value_segments(csources,listed),Written.shifts_of(csources,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0),cell_shift_lists(shifts0,lane),shifts0_eq),
      Equal.sym(List<&2,List<&2,List<&2,Tasks.Segment>>>,Placing.cell_segments(Con{first,rest},shifts0,lane,Grid.Innermost{}),Placing.value_segments(csources,cell_shift_lists(shifts0,lane)),cells_segments(Con{first,rest},shifts0,lane))))
  +fill = Written.first_fill(~Tile.Tile8,~Representation.Lifted<Part>,inputs)
  +lifted_names = Placing.lifted_logicals(~Part,Con{part,others},Con{first,rest},lane,Grid.Innermost{})
  +rank_in = {%Equal.sym(List<&2,U32>,Written.first_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs),cw,first_eq) : {Nat.is_le(List.length(&2,U32,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(Nat,List.length(&2,U32,cw),List.length(&2,U32,cv),first_length) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    cells_rank_room : {Nat.is_le(List.length(&2,U32,Written.first_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +count_in = {%Written.shapes_count(~Tile.Tile8,~Representation.Lifted<Part>,inputs) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(List<&2,List<&2,U32>>,Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),csources,sources_eq) : {Nat.is_le(List.length(&2,List<&2,U32>,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(Nat,List.length(&2,List<&2,U32>,csources),List.length(&2,List<&2,U32>,Con{first,rest}),sources_length(Con{first,rest},lane)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    parts_count : {Nat.is_le(Written.inputs_count(~Tile.Tile8,~Representation.Lifted<Part>,inputs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +valid_in = {%Equal.sym(List<&2,U32>,Written.first_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs),cw,first_eq) :
      {Written.survey_valid(Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),Nat.is_lt(U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),List.length(&2,U32,_)),0) == True{} : Bool}
    %Equal.sym(List<&2,List<&2,U32>>,Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),csources,sources_eq) : {Written.survey_valid(_,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),Nat.is_lt(U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),List.length(&2,U32,cw)),0) == True{} : Bool}
    %Equal.sym(Bool,Nat.is_lt(U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),List.length(&2,U32,cw)),True{},icm_inside) : {Written.survey_valid(csources,U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),_,0) == True{} : Bool}
    shapes_valid(csources,cv,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0,limits,cells_rank_room,fit0) :
    {Written.survey_valid(Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),Nat.is_lt(U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),List.length(&2,U32,Written.first_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs))),0) == True{} : Bool}}
  +fit_in = {%Equal.sym(List<&2,U32>,Written.joined_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),cv,joined_eq) : Written.ShapesFit(Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),_,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0)
    %Equal.sym(List<&2,List<&2,U32>>,Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),csources,sources_eq) : Written.ShapesFit(_,cv,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0)
    fit0 : Written.ShapesFit(Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),Written.joined_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0)}
  +nonempty_in = {%Equal.sym(List<&2,U32>,Written.joined_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),cv,joined_eq) : {Shapes.has_zero(_) == False{} : Bool}
    ViewMoved.none_zero(cv,cells_filled) : {Shapes.has_zero(Written.joined_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) == False{} : Bool}}
  +counted_in = {%Equal.sym(List<&2,U32>,Written.joined_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),cv,joined_eq) : {Tensors.count_of(_) == Some{Global.count(cv,cells_counted)} : Maybe<&2,U32>}
    Global.accepted(cv,cells_counted) : {Tensors.count_of(Written.joined_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))) == Some{Global.count(cv,cells_counted)} : Maybe<&2,U32>}}
  +written = {%segments_eq : Global.Shaped(~Tile.Tile8,~Gathered.Placings<Representation.Lifted<Part>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),Gathered.Placings{lifted_names,_,csources,cv,fill},cv,Layout.concat(~Tile.Tile8,TiledConcat.cells(parts_of(model),lane),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))))
    %parts_eq : Global.Shaped(~Tile.Tile8,~Gathered.Placings<Representation.Lifted<Part>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),Gathered.Placings{lifted_names,Written.segments_of(csources,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0),csources,cv,fill},cv,Layout.concat(~Tile.Tile8,_,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))))
    %joined_eq : Global.Shaped(~Tile.Tile8,~Gathered.Placings<Representation.Lifted<Part>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),Gathered.Placings{lifted_names,Written.segments_of(csources,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0),csources,_,fill},_,
      Layout.concat(~Tile.Tile8,Written.parts_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))))
    %sources_eq : Global.Shaped(~Tile.Tile8,~Gathered.Placings<Representation.Lifted<Part>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),Gathered.Placings{lifted_names,Written.segments_of(_,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0),_,Written.joined_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),fill},
      Written.joined_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),Layout.concat(~Tile.Tile8,Written.parts_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))))
    %named_eq : Global.Shaped(~Tile.Tile8,~Gathered.Placings<Representation.Lifted<Part>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),Gathered.Placings{_,Written.segments_of(Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0),Written.shapes_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),Written.joined_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),fill},
      Written.joined_shape(~Tile.Tile8,~Representation.Lifted<Part>,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),Layout.concat(~Tile.Tile8,Written.parts_of(~Tile.Tile8,~Representation.Lifted<Part>,inputs),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))))
    Written.concat(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),limits,inputs,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),cells_held,default_capacity,rank_in,count_in,valid_in,fit_in,nonempty_in,
      Global.count(cv,cells_counted),counted_in) :
    Global.Shaped(~Tile.Tile8,~Gathered.Placings<Representation.Lifted<Part>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),Gathered.Placings{lifted_names,Placing.cell_segments(Con{first,rest},shifts0,lane,Grid.Innermost{}),csources,cv,fill},cv,Layout.concat(~Tile.Tile8,TiledConcat.cells(parts_of(model),lane),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))))}
  +restated = {%Equal.sym(List<&2,List<&2,List<&2,Tasks.Segment>>>,Written.segments_of(Con{first,rest},axis,0),Placing.value_segments(Con{first,rest},shifts0),segments_shifts(Con{first,rest},axis,0)) :
      Global.Shaped(~Tile.Tile8,~Representation.Lifted<Gathered.Placings<Part,F32>>,~Representation.lifted(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value)),Representation.Lifted{Gathered.Placings{Con{part,others},_,Con{first,rest},values,0.0},Public.geometry_of(values,lane,Grid.Innermost{})},cv,Layout.concat(~Tile.Tile8,TiledConcat.cells(parts_of(model),lane),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))))
    restate(~Part,~part_value,Nat.is_lt(lane,List.length(&2,U32,values)),Con{part,others},Con{first,rest},values,axis,lane,limits,fill,{==},lane_inside,room,positive,named,walk_facts,
      index => below => {%Equal.sym(Bool,Placing.covers(Con{first,rest},shifts0,Grid.coordinates(cv,index),lane,Grid.Innermost{}),plain_covers(csources,cell_shift_lists(shifts0,lane),Grid.coordinates(cv,index)),
            covers_cells(Con{first,rest},shifts0,Grid.coordinates(cv,index),lane)) : {_ == True{} : Bool}
          %shifts0_eq : {plain_covers(csources,_,Grid.coordinates(cv,index)) == True{} : Bool}
          walk_covers(csources,cv,U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))),0,limits,index,cells_rank_room,cells_space,cells_filled,below,fit0,total0,Order.zero_le(nth(Targets.cell_coordinates(cv,U32.to_nat(index)),U32.to_nat(U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values))))))) :
          {Placing.covers(Con{first,rest},shifts0,Grid.coordinates(cv,index),lane,Grid.Innermost{}) == True{} : Bool}},
      Layout.concat(~Tile.Tile8,TiledConcat.cells(parts_of(model),lane),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),written) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Gathered.Placings<Part,F32>>,~Representation.lifted(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value)),Representation.Lifted{Gathered.Placings{Con{part,others},Written.segments_of(Con{first,rest},axis,0),Con{first,rest},values,0.0},Public.geometry_of(values,lane,Grid.Innermost{})},cv,Layout.concat(~Tile.Tile8,TiledConcat.cells(parts_of(model),lane),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))))}
  +laid = {Public.Laid{lane_inside,{==},{%pinned : {Tiled.logical_shape(cv,lane,_,Grid.Innermost{}) == values : List<&2,U32>} Public.cells_restored(values,lane,Grid.Innermost{}) :
      {Tiled.logical_shape(cv,lane,lanes,Grid.Innermost{}) == values : List<&2,U32>}},Global.Counted{count,counted},pinned} : Public.Laid<cv,lane,lanes,Grid.Innermost{},values>}
  Public.laid_contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{Con{part,others},Written.segments_of(Con{first,rest},axis,0),Con{first,rest},values,0.0},values,Layout.concat(~Tile.Tile8,TiledConcat.cells(parts_of(model),lane),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)))),lane,lanes,Grid.Innermost{},KernelGemm.Both{restated,laid})

# Parts of a model of the contract that fit the joined values one after
# another make the joined values.
def concat_modeled(~Part: Data,~part_value: Part -> U32 -> F32,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,+model: PartsModel,+axis: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+held: PartsHeld(~Part,~part_value,logicals,shapes,parts_of(model)),
  +rank_room: {Nat.is_le(List.length(&2,U32,joined_values(shapes,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+parts_count: {Nat.is_le(List.length(&2,List<&2,U32>,shapes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fit: Written.ShapesFit(shapes,joined_values(shapes,axis),axis,0),+count: U32,+counted: {Tensors.count_of(joined_values(shapes,axis)) == Some{count} : Maybe<&2,U32>}) ->
  Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{logicals,Written.segments_of(shapes,axis,0),shapes,joined_values(shapes,axis),0.0},
    joined_values(shapes,axis),TiledConcat.concat(parts_of(model),axis)):
  match logicals shapes:
    case Con{+part,+others} Con{+first,+rest}:
      Held.Both{+first_slice,Held.Both{+first_nonempty,+later_fit}} = fit
      +mark = U32.to_nat(axis)
      +values = joined_values(Con{first,rest},axis)
      +total = Written.survey_total(Con{first,rest},mark,0)
      +own = first_lane(model)
      +first_same = first_values(~Part,~part_value,part,others,first,rest,model,held)
      +own_inside = first_lane_inside(~Part,~part_value,part,others,first,rest,model,held)
      +first_marked = SetSlice.marked_of(limits,values,first,axis,0,rank_room,first_slice)
      +ranked = SetSlice.fits_length(values,first,mark,0,first_marked)
      +mark_inside_v = SetSlice.fits_inside(values,first,mark,0,first_marked)
      +mark_inside = {%Equal.sym(Nat,List.length(&2,U32,first),List.length(&2,U32,values),ranked) : {Nat.is_lt(mark,_) == True{} : Bool} mark_inside_v :
        {Nat.is_lt(mark,List.length(&2,U32,first)) == True{} : Bool}}
      +whole = survey_whole(Con{first,rest},mark,True{})
      +lane = TiledConcat.concat_lane(first,mark,own,whole)
      +lane_facts = lane_cases(Nat.is_eq(mark,own),whole,Con{first,rest},first,mark,own,{==},{==},mark_inside,own_inside)
      +lane_first = both_left(~{Nat.is_le(lane,List.length(&2,U32,first)) == True{} : Bool},~LanesKept(lane,mark,0,Con{first,rest}),lane_facts)
      +kept = both_right(~{Nat.is_le(lane,List.length(&2,U32,first)) == True{} : Bool},~LanesKept(lane,mark,0,Con{first,rest}),lane_facts)
      +lane_inside = {%ranked : {Nat.is_le(lane,_) == True{} : Bool} lane_first : {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool}}
      +positive = Facts.placed_positive(values,first,Layout.shifts_at(List.length(&2,U32,values),axis,0,0),Written.slice_in(values,first,axis,0,0,first_slice),
        CountCompletion.positive_elements(first,first_nonempty))
      +room = StorageCount.counted_room(limits,values,Global.Counted{count,counted},default_capacity)
      +total_eq = Equal.sym(U32,Layout.extent_at(values,mark),total,
        Equal.trans(U32,Layout.extent_at(values,mark),Grid.lanes_at(values,mark),total,SetSlice.extent_lanes(values,mark,mark_inside_v),set_lane(first,mark,total,mark_inside)))
      +first_keeps = part_keeps(Nat.is_eq(lane,mark),lane,mark,0,first,rest,Layout.extent_at(values,mark),{==},kept,total_eq)
      +lane_room = Rooms.lane_room(limits,values,mark,mark_inside_v,positive,room)
      +first_cells = SetSlice.cell_fits(limits,values,first,mark,0,lane,Grid.Innermost{},Layout.extent_at(first,mark),Layout.extent_at(values,mark),first_marked,first_keeps,lane_room)
      +cells_rank_room = {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),List.length(&2,U32,values),Rooms.cell_rank(values,lane,Grid.Innermost{})) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
        rank_room : {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
      +axis_room = Order.transitive(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits)),
        Bounds.strict_le(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),SetSlice.fits_inside(Tiled.cell_shape(values,lane,Grid.Innermost{}),Tiled.cell_shape(first,lane,Grid.Innermost{}),Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),(0 / 8 : U32),0),first_cells)),cells_rank_room)
      +axis_back = Bounds.roundtrip(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,List.length(&2,U32,values)),Machine.capacity(limits),Limits.capacity_fits(limits),axis_room)
      +lanes = Bool.pick(U32,Nat.is_eq(mark,lane),total,Grid.lanes_at(first,lane))
      %Equal.sym(TiledConcat.Parts & TiledConcat.Lead,TiledConcat.lead(parts_of(model)),(parts_of(model),TiledConcat.Lead{first_logical(model),own}),lead_model(model)) :
        Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{Con{part,others},Written.segments_of(Con{first,rest},axis,0),Con{first,rest},values,0.0},values,TiledConcat.concat_led(axis,_))
      %Equal.sym(List<&2,U32>,first_logical(model),first,first_same) :
        Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{Con{part,others},Written.segments_of(Con{first,rest},axis,0),Con{first,rest},values,0.0},values,TiledConcat.concat_surveyed(_,axis,own,TiledConcat.survey(parts_of(model),_,mark,own,TiledConcat.Fit{True{},True{},0})))
      %Equal.sym(TiledConcat.Parts & TiledConcat.Fit,TiledConcat.survey(parts_of(model),first,mark,own,TiledConcat.Fit{True{},True{},0}),
          (parts_of(model),TiledConcat.Fit{survey_fits(Con{first,rest},first,mark,True{},0),whole,total}),surveyed(~Part,~part_value,Con{part,others},Con{first,rest},model,first,mark,own,True{},True{},0,held)) :
        Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{Con{part,others},Written.segments_of(Con{first,rest},axis,0),Con{first,rest},values,0.0},values,TiledConcat.concat_surveyed(first,axis,own,_))
      %Equal.sym(Bool,survey_fits(Con{first,rest},first,mark,True{},0),True{},fits_surveyed(Con{first,rest},values,first,axis,0,limits,rank_room,first_marked,fit)) :
        Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{Con{part,others},Written.segments_of(Con{first,rest},axis,0),Con{first,rest},values,0.0},values,TiledConcat.concat_at(first,axis,lane,_,total,parts_of(model)))
      %Equal.sym(Bool,Nat.is_lt(mark,List.length(&2,U32,first)),True{},mark_inside) :
        Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{Con{part,others},Written.segments_of(Con{first,rest},axis,0),Con{first,rest},values,0.0},values,TiledConcat.concat_fitted(True{} && _ && Nat.is_le(lane,List.length(&2,U32,first)),axis,lane,lanes,List.length(&2,U32,first),parts_of(model)))
      %Equal.sym(Bool,Nat.is_le(lane,List.length(&2,U32,first)),True{},lane_first) :
        Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{Con{part,others},Written.segments_of(Con{first,rest},axis,0),Con{first,rest},values,0.0},values,TiledConcat.concat_fitted(True{} && True{} && _,axis,lane,lanes,List.length(&2,U32,first),parts_of(model)))
      %Equal.sym(Nat,List.length(&2,U32,first),List.length(&2,U32,values),ranked) :
        Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{Con{part,others},Written.segments_of(Con{first,rest},axis,0),Con{first,rest},values,0.0},values,Tiled.Tiled{Layout.concat(~Tile.Tile8,TiledConcat.cells(parts_of(model),lane),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},mark,lane,_))),lane,lanes,Grid.Innermost{}})
      concat_assembled(~Part,~part_value,part,others,first,rest,model,values,axis,lane,lanes,limits,default_capacity,rank_room,parts_count,room,positive,lane_inside,count,counted,
        joined_lanes(Nat.is_eq(mark,lane),first,mark,lane,total,{==},mark_inside,first_marked),held_named(~Part,~part_value,Con{part,others},Con{first,rest},model,held),
        cells_walk(Con{first,rest},values,axis,0,lane,limits,rank_room,room,positive,lane_inside,axis_back,{==},fit,kept,total_eq,parts_room(~Part,~part_value,Con{part,others},Con{first,rest},model,limits,default_capacity,held)),
        cell_parts(~Part,~part_value,Con{part,others},Con{first,rest},model,values,axis,0,lane,limits,default_capacity,rank_room,lane_inside,fit,held))
    case Con{+part,+others} Nil{}:
      Empty.absurd(Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{Con{part,others},Written.segments_of([],axis,0),[],joined_values([],axis),0.0},
        joined_values([],axis),TiledConcat.concat(parts_of(model),axis)),held)
    case Nil{} +shapes:
      Empty.absurd(Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{[],Written.segments_of(shapes,axis,0),shapes,joined_values(shapes,axis),0.0},
        joined_values(shapes,axis),TiledConcat.concat(parts_of(model),axis)),held)

def concat_observed(~Part: Data,~part_value: Part -> U32 -> F32,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,-parts: TiledConcat.Parts,+axis: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+observed: Observed<parts>,+held: PartsHeld(~Part,~part_value,logicals,shapes,parts),
  +rank_room: {Nat.is_le(List.length(&2,U32,joined_values(shapes,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+parts_count: {Nat.is_le(List.length(&2,List<&2,U32>,shapes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fit: Written.ShapesFit(shapes,joined_values(shapes,axis),axis,0),+count: U32,+counted: {Tensors.count_of(joined_values(shapes,axis)) == Some{count} : Maybe<&2,U32>}) ->
  Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{logicals,Written.segments_of(shapes,axis,0),shapes,joined_values(shapes,axis),0.0},
    joined_values(shapes,axis),TiledConcat.concat(parts,axis)):
  Observed{+model,+equation} = observed
  %equation : Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{logicals,Written.segments_of(shapes,axis,0),shapes,joined_values(shapes,axis),0.0},
    joined_values(shapes,axis),TiledConcat.concat(_,axis))
  concat_modeled(~Part,~part_value,logicals,shapes,model,axis,limits,default_capacity,
    {%Equal.sym(TiledConcat.Parts,parts_of(model),parts,equation) : PartsHeld(~Part,~part_value,logicals,shapes,_) held : PartsHeld(~Part,~part_value,logicals,shapes,parts_of(model))},
    rank_room,parts_count,fit,count,counted)

# ---- The tiled form of concat ----

# Parts of the contract, each with cells, that fit the joined values one after
# another along the axis, the joined values counted within the capacity, make
# a tiled tensor of the contract of every part placed where it lies.
def concat(~Part: Data,~part_value: Part -> U32 -> F32,+logicals: List<&2,Part>,+shapes: List<&2,List<&2,U32>>,parts: TiledConcat.Parts,+axis: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+held: PartsHeld(~Part,~part_value,logicals,shapes,parts),
  +rank_room: {Nat.is_le(List.length(&2,U32,joined_values(shapes,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+parts_count: {Nat.is_le(List.length(&2,List<&2,U32>,shapes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fit: Written.ShapesFit(shapes,joined_values(shapes,axis),axis,0),+count: U32,+counted: {Tensors.count_of(joined_values(shapes,axis)) == Some{count} : Maybe<&2,U32>}) ->
  Public.Contract(~Gathered.Placings<Part,F32>,~Gathered.placing_value(~F32,~Part,~part_value),Gathered.Placings{logicals,Written.segments_of(shapes,axis,0),shapes,joined_values(shapes,axis),0.0},
    joined_values(shapes,axis),TiledConcat.concat(parts,axis)):
  concat_observed(~Part,~part_value,logicals,shapes,parts,axis,limits,default_capacity,observe_parts(~Part,~part_value,logicals,shapes,parts,held),held,rank_room,parts_count,fit,count,counted)
