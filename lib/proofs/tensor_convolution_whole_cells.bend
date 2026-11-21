# The whole cells a convolution keeps together in one task: one cell, or the
# output cells of whole images, a whole number of output units inside the
# capacity.
import Base
import ../tensor_convolution.bend as TensorConvolution
import ../tensor_grid.bend as Grid
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./nat_extrema.bend as NatExtrema
import ./u32_extrema.bend as Extrema
import ./tensor_grid_pieces.bend as Laws
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_shape_refinement.bend as ShapeLaws

# The cells of a piece of the grid of a shape of whole images hold at most
# its images.
def piece_images_of(+limits: Machine.Limits,+cells: List<&2,U32>,+image: U32,+images: Nat,+image_positive: {Nat.is_lt(0n,U32.to_nat(image)) == True{} : Bool},
  +image_room: {Nat.is_le(U32.to_nat(image),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +exact: {ShapeLaws.elements(cells) == Nat.mul(images,U32.to_nat(image)) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +counted: {U32.to_nat(Grid.count_of(Grid.grid(cells,Grid.piece_target()))) == ShapeLaws.elements(cells) : Nat},
  +facts: Laws.GridFacts<limits,Grid.grid(cells,Grid.piece_target())>) ->
  {Nat.is_le(U32.to_nat(TensorConvolution.images_of(Grid.grid(cells,Grid.piece_target()),image)),images) == True{} : Bool}:
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+grid_exact,+grid_cells} = facts
  Laws.Shape{+outer_positive,+unit_positive,+block_positive,+block_below,+per_exact,+shape_room} = shape
  +units = U32.to_nat(unit_cells)
  +blocks = U32.to_nat(block)
  +extents = U32.to_nat(extent)
  +outers = U32.to_nat(outer)
  +size = U32.to_nat(image)
  +elements = ShapeLaws.elements(cells)
  +count_cells = {%grid_exact : {U32.to_nat(Grid.count_of(_)) == elements : Nat} counted : {U32.to_nat(count) == elements : Nat}}
  +spread = DivisionOrder.le_multiple(outers,Nat.mul(extents,units),outer_positive)
  +turned = {%RowMajor.multiply_commutative(extents,units) : {Nat.is_le(_,Nat.mul(outers,Nat.mul(extents,units))) == True{} : Bool}
    spread : {Nat.is_le(Nat.mul(units,extents),Nat.mul(outers,Nat.mul(extents,units))) == True{} : Bool}}
  +grouped = {%Equal.sym(Nat,Nat.mul(Nat.mul(outers,extents),units),Nat.mul(outers,Nat.mul(extents,units)),RowMajor.assoc(outers,extents,units)) :
      {Nat.is_le(Nat.mul(units,extents),_) == True{} : Bool}
    turned : {Nat.is_le(Nat.mul(units,extents),Nat.mul(Nat.mul(outers,extents),units)) == True{} : Bool}}
  +counted_cells = {%Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(outers,extents),units),grid_cells) : {Nat.is_le(Nat.mul(units,extents),_) == True{} : Bool}
    grouped : {Nat.is_le(Nat.mul(units,extents),U32.to_nat(count)) == True{} : Bool}}
  +extent_cells = {%count_cells : {Nat.is_le(Nat.mul(units,extents),_) == True{} : Bool} counted_cells : {Nat.is_le(Nat.mul(units,extents),elements) == True{} : Bool}}
  +piece_le = Order.transitive(Nat.mul(units,blocks),Nat.mul(units,extents),elements,RowMajor.mul_le_left(units,blocks,extents,block_below),extent_cells)
  +product = Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),unit_cells,block,units,blocks,{==},{==},
    Order.transitive(Nat.mul(units,blocks),elements,U32.to_nat(Machine.capacity(limits)),piece_le,room))
  +divisor_exact = Extrema.max_one_exact(image,size,{==},image_positive)
  +divisor_room = {%Equal.sym(Nat,U32.to_nat(U32.max(image,1)),size,divisor_exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    image_room : {Nat.is_le(U32.to_nat(U32.max(image,1)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +divisor_positive = {%Equal.sym(Nat,U32.to_nat(U32.max(image,1)),size,divisor_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    image_positive : {Nat.is_lt(0n,U32.to_nat(U32.max(image,1))) == True{} : Bool}}
  +quotient = Laws.word_quotient(limits,(unit_cells * block : U32),U32.max(image,1),divisor_room,divisor_positive)
  +quotient_product = {%product : {U32.to_nat(((unit_cells * block) / U32.max(image,1) : U32)) == Nat.div(_,U32.to_nat(U32.max(image,1))) : Nat}
    quotient : {U32.to_nat(((unit_cells * block) / U32.max(image,1) : U32)) == Nat.div(Nat.mul(units,blocks),U32.to_nat(U32.max(image,1))) : Nat}}
  +quotient_exact = {%divisor_exact : {U32.to_nat(((unit_cells * block) / U32.max(image,1) : U32)) == Nat.div(Nat.mul(units,blocks),_) : Nat}
    quotient_product : {U32.to_nat(((unit_cells * block) / U32.max(image,1) : U32)) == Nat.div(Nat.mul(units,blocks),size) : Nat}}
  +bounded = DivisionOrder.quotient_le(Nat.mul(units,blocks),elements,size,image_positive,piece_le)
  +whole = {%Equal.sym(Nat,elements,Nat.mul(images,size),exact) : {Nat.div(_,size) == images : Nat}
    DivisionOrder.shifted_small_quotient(0n,size,images,image_positive,image_positive) : {Nat.div(elements,size) == images : Nat}}
  +below = {%whole : {Nat.is_le(Nat.div(Nat.mul(units,blocks),size),_) == True{} : Bool} bounded : {Nat.is_le(Nat.div(Nat.mul(units,blocks),size),images) == True{} : Bool}}
  +held = {%Equal.sym(Nat,U32.to_nat(((unit_cells * block) / U32.max(image,1) : U32)),Nat.div(Nat.mul(units,blocks),size),quotient_exact) : {Nat.is_le(_,images) == True{} : Bool}
    below : {Nat.is_le(U32.to_nat(((unit_cells * block) / U32.max(image,1) : U32)),images) == True{} : Bool}}
  %Equal.sym(Grid.Grid,Grid.grid(cells,Grid.piece_target()),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},grid_exact) :
    {Nat.is_le(U32.to_nat(TensorConvolution.images_of(_,image)),images) == True{} : Bool}
  held

def piece_images(+limits: Machine.Limits,+cells: List<&2,U32>,+room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},+image: U32,+images: Nat,+image_positive: {Nat.is_lt(0n,U32.to_nat(image)) == True{} : Bool},
  +image_room: {Nat.is_le(U32.to_nat(image),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +exact: {ShapeLaws.elements(cells) == Nat.mul(images,U32.to_nat(image)) : Nat}) ->
  {Nat.is_le(U32.to_nat(TensorConvolution.images_of(Grid.grid(cells,Grid.piece_target()),image)),images) == True{} : Bool}:
  piece_images_of(limits,cells,image,images,image_positive,image_room,exact,room,GridRanges.grid_count(limits,cells,Grid.piece_target(),room,nonempty),
    GridRanges.grid_facts(limits,cells,Grid.piece_target(),room,nonempty))

# The images the pieces of the input and output grids hold, at least one.
def images_held(+input_cells: List<&2,U32>,+cells: List<&2,U32>,+image: U32,+output_image: U32) -> U32:
  U32.max(U32.max(TensorConvolution.images_of(Grid.grid(input_cells,Grid.piece_target()),image),TensorConvolution.images_of(Grid.grid(cells,Grid.piece_target()),output_image)),1)

# The whole cells of whole images keep a whole number of output units.
def whole_picked(+held: Bool,+limits: Machine.Limits,+input_cells: List<&2,U32>,+cells: List<&2,U32>,+image: U32,+output_image: U32,+images: Nat,+unit: Nat,
  +input_room: {Nat.is_le(ShapeLaws.elements(input_cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+input_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(input_cells)) == True{} : Bool},
  +output_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+output_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +image_positive: {Nat.is_lt(0n,U32.to_nat(image)) == True{} : Bool},+image_room: {Nat.is_le(U32.to_nat(image),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_positive: {Nat.is_lt(0n,U32.to_nat(output_image)) == True{} : Bool},+output_image_room: {Nat.is_le(U32.to_nat(output_image),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +input_exact: {ShapeLaws.elements(input_cells) == Nat.mul(images,U32.to_nat(image)) : Nat},+output_exact: {ShapeLaws.elements(cells) == Nat.mul(images,U32.to_nat(output_image)) : Nat},
  +output_units: GridRanges.Multiple<U32.to_nat(output_image),unit>) ->
  GridRanges.WholeCells<limits,Bool.pick(U32,held,(output_image * images_held(input_cells,cells,image,output_image) : U32),1),unit>:
  match held:
    case False{}: GridRanges.SingleCell{{==}}
    case True{}:
      GridRanges.Multiple{+times,+times_exact} = output_units
      +inner = TensorConvolution.images_of(Grid.grid(input_cells,Grid.piece_target()),image)
      +outer = TensorConvolution.images_of(Grid.grid(cells,Grid.piece_target()),output_image)
      +either = U32.max(inner,outer)
      +kept = images_held(input_cells,cells,image,output_image)
      +size = U32.to_nat(output_image)
      +positive_cells = {%output_exact : {Nat.is_lt(0n,_) == True{} : Bool} output_nonempty : {Nat.is_lt(0n,Nat.mul(images,size)) == True{} : Bool}}
      +images_positive = RowMajor.left_factor_positive(images,size,positive_cells)
      +either_below = NatExtrema.maximum_upper(U32.to_nat(inner),U32.to_nat(outer),images,
        piece_images(limits,input_cells,input_room,input_nonempty,image,images,image_positive,image_room,input_exact),
        piece_images(limits,cells,output_room,output_nonempty,output_image,images,output_positive,output_image_room,output_exact))
      +kept_below = NatExtrema.maximum_upper(Nat.max(U32.to_nat(inner),U32.to_nat(outer)),1n,images,either_below,Order.strict_follow(0n,images,images_positive))
      +kept_exact = Equal.trans(Nat,U32.to_nat(kept),Nat.max(U32.to_nat(either),1n),Nat.max(Nat.max(U32.to_nat(inner),U32.to_nat(outer)),1n),Extrema.maximum(either,1),
        Equal.cong(Nat,Nat,value => Nat.max(value,1n),U32.to_nat(either),Nat.max(U32.to_nat(inner),U32.to_nat(outer)),Extrema.maximum(inner,outer)))
      +held_below = {%Equal.sym(Nat,U32.to_nat(kept),Nat.max(Nat.max(U32.to_nat(inner),U32.to_nat(outer)),1n),kept_exact) : {Nat.is_le(_,images) == True{} : Bool}
        kept_below : {Nat.is_le(U32.to_nat(kept),images) == True{} : Bool}}
      +turned = {%RowMajor.multiply_commutative(size,images) : {Nat.is_le(Nat.mul(size,U32.to_nat(kept)),_) == True{} : Bool}
        RowMajor.mul_le_left(size,U32.to_nat(kept),images,held_below) : {Nat.is_le(Nat.mul(size,U32.to_nat(kept)),Nat.mul(images,size)) == True{} : Bool}}
      +within = {%Equal.sym(Nat,ShapeLaws.elements(cells),Nat.mul(images,size),output_exact) : {Nat.is_le(Nat.mul(size,U32.to_nat(kept)),_) == True{} : Bool}
        turned : {Nat.is_le(Nat.mul(size,U32.to_nat(kept)),ShapeLaws.elements(cells)) == True{} : Bool}}
      +product_room = Order.transitive(Nat.mul(size,U32.to_nat(kept)),ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits)),within,output_room)
      +product = Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),output_image,kept,size,U32.to_nat(kept),{==},{==},product_room)
      +whole_room = {%Equal.sym(Nat,U32.to_nat((output_image * kept : U32)),Nat.mul(size,U32.to_nat(kept)),product) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
        product_room : {Nat.is_le(U32.to_nat((output_image * kept : U32)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
      +count = U32.to_nat(kept)
      +regrouped = Equal.trans(Nat,Nat.mul(size,count),Nat.mul(Nat.mul(times,unit),count),Nat.mul(Nat.mul(times,count),unit),
        Equal.cong(Nat,Nat,value => Nat.mul(value,count),size,Nat.mul(times,unit),times_exact),
        Equal.trans(Nat,Nat.mul(Nat.mul(times,unit),count),Nat.mul(times,Nat.mul(unit,count)),Nat.mul(Nat.mul(times,count),unit),RowMajor.assoc(times,unit,count),
          Equal.trans(Nat,Nat.mul(times,Nat.mul(unit,count)),Nat.mul(times,Nat.mul(count,unit)),Nat.mul(Nat.mul(times,count),unit),
            Equal.cong(Nat,Nat,value => Nat.mul(times,value),Nat.mul(unit,count),Nat.mul(count,unit),RowMajor.multiply_commutative(unit,count)),
            Equal.sym(Nat,Nat.mul(Nat.mul(times,count),unit),Nat.mul(times,Nat.mul(count,unit)),RowMajor.assoc(times,count,unit)))))
      GridRanges.WholeUnits{GridRanges.Multiple{Nat.mul(times,count),Equal.trans(Nat,U32.to_nat((output_image * kept : U32)),Nat.mul(size,count),Nat.mul(Nat.mul(times,count),unit),product,regrouped)},whole_room}

# The whole cells a convolution keeps together are one cell or the output
# cells of whole images.
def whole_units(+limits: Machine.Limits,+lead: U32,+input_cells: List<&2,U32>,+cells: List<&2,U32>,+image: U32,+output_image: U32,+images: Nat,+unit: Nat,
  +input_room: {Nat.is_le(ShapeLaws.elements(input_cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+input_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(input_cells)) == True{} : Bool},
  +output_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+output_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +image_positive: {Nat.is_lt(0n,U32.to_nat(image)) == True{} : Bool},+image_room: {Nat.is_le(U32.to_nat(image),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_positive: {Nat.is_lt(0n,U32.to_nat(output_image)) == True{} : Bool},+output_image_room: {Nat.is_le(U32.to_nat(output_image),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +input_exact: {ShapeLaws.elements(input_cells) == Nat.mul(images,U32.to_nat(image)) : Nat},+output_exact: {ShapeLaws.elements(cells) == Nat.mul(images,U32.to_nat(output_image)) : Nat},
  +output_units: GridRanges.Multiple<U32.to_nat(output_image),unit>) ->
  GridRanges.WholeCells<limits,TensorConvolution.whole_cells(lead,Grid.grid(input_cells,Grid.piece_target()),Grid.grid(cells,Grid.piece_target()),image,output_image),unit>:
  whole_picked(U32.is_le(TensorConvolution.prefix_of(Grid.grid(input_cells,Grid.piece_target())),lead) && U32.is_le(TensorConvolution.prefix_of(Grid.grid(cells,Grid.piece_target())),lead),
    limits,input_cells,cells,image,output_image,images,unit,input_room,input_nonempty,output_room,output_nonempty,image_positive,image_room,output_positive,output_image_room,
    input_exact,output_exact,output_units)
