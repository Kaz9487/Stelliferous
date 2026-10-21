# A retained gradient row slice has the same padded reads as its original image.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ../kernel_packing_samples.bend as Masks
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_copy_geometry.bend as Copy
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_extents.bend as Extents
import ./input_gradient_phase_domain.bend as Domain
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./kernel_spatial_packing_samples.bend as Samples
import ./padded_interval.bend as Padded
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_division.bend as Division
import ./nat_to_u32_bounds.bend as Bounds
import ./machine_limit_bounds.bend as Limits
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as Words
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./boolean_logic.bend as Boolean
import ./storage_shape.bend as Shape

type Range<-first: U32,-rows: U32,-height: U32> is Data:
  EmptyRows{empty: {rows == 0 : U32}}
  ImageRows{contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool}}
def range_room(+first: U32,+rows: U32,+height: U32,+limits: Machine.Limits,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool},
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(rows),U32.to_nat(first)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  %Order.commutative(U32.to_nat(first),U32.to_nat(rows)) : {Nat.is_le(_,U32.to_nat(Machine.coordinate(limits))) == True{} : Bool} Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(height),U32.to_nat(Machine.coordinate(limits)),contained,height_room)
def local_image_global(+first: U32,+rows: U32,+height: U32,+limits: Machine.Limits,+row: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool},
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool},
  interval: Shape.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat(row)) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}>) -> {U32.is_lt(row,height) == True{} : Bool}:
  Shape.Both{lower,+upper} = interval
  %Equal.sym(Bool,U32.is_lt(row,height),Nat.is_lt(U32.to_nat(row),U32.to_nat(height)),Comparison.lt(row,height)) : {_ == True{} : Bool}
  Bounds.nat_trans(U32.to_nat(row),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(height),upper,contained)
def local_implies_global(+first: U32,+rows: U32,+height: U32,+limits: Machine.Limits,+row: U32,+range: Range<first,rows,height>,
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool},
  +local: {U32.is_lt((row - first : U32),rows) == True{} : Bool}) -> {U32.is_lt(row,height) == True{} : Bool}:
  match range:
    case EmptyRows{+empty}:
      Empty.absurd({U32.is_lt(row,height) == True{} : Bool},Words.false_true(
        %Division.lt_zero(U32.to_nat((row - first : U32))) : {_ == True{} : Bool}
        %Comparison.lt((row - first : U32),0) : {_ == True{} : Bool}
        %empty : {U32.is_lt((row - first : U32),_) == True{} : Bool} local))
    case ImageRows{+contained}:
      local_image_global(first,rows,height,limits,row,contained,height_room,Padded.forward(row,first,rows,Machine.coordinate(limits),Limits.coordinate_fits(limits),range_room(first,rows,height,limits,contained,height_room),local))

def covered_local(+first: U32,+rows: U32,+height: U32,+limits: Machine.Limits,+row: U32,+range: Range<first,rows,height>,
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(first),U32.to_nat(row)) == True{} : Bool},
  +upper: {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}) ->
  {U32.is_lt((row - first : U32),rows) == True{} : Bool}:
  match range:
    case EmptyRows{+empty}:
      Empty.absurd({U32.is_lt((row - first : U32),rows) == True{} : Bool},Words.false_true(
        %Order.not_below(U32.to_nat(first),U32.to_nat(row),lower) : {_ == True{} : Bool}
        %Addition.zero_right(U32.to_nat(first)) : {Nat.is_lt(U32.to_nat(row),_) == True{} : Bool}
        %Equal.cong(U32,Nat,U32.to_nat,rows,0,empty) : {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(first),_)) == True{} : Bool} upper))
    case ImageRows{+contained}:
      Padded.backward(row,first,rows,Machine.coordinate(limits),Limits.coordinate_fits(limits),range_room(first,rows,height,limits,contained,height_room),lower,upper)
def coverage_local(+first: U32,+rows: U32,+height: U32,+limits: Machine.Limits,+row: U32,+range: Range<first,rows,height>,
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool},
  interval: Shape.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat(row)) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}>) ->
  {U32.is_lt((row - first : U32),rows) == True{} : Bool}:
  Shape.Both{+lo,+hi} = interval
  covered_local(first,rows,height,limits,row,range,height_room,lo,hi)
def row_mask_equal(+first: U32,+rows: U32,+height: U32,+limits: Machine.Limits,+row: U32,+range: Range<first,rows,height>,
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool},
  coverage: {U32.is_lt(row,height) == True{} : Bool} ->
    Shape.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat(row)) == True{} : Bool},
      {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}>) ->
  {U32.is_lt((row - first : U32),rows) == U32.is_lt(row,height) : Bool}:
  Boolean.equivalent(U32.is_lt((row - first : U32),rows),U32.is_lt(row,height),
    local => local_implies_global(first,rows,height,limits,row,range,height_room,local),
    global => coverage_local(first,rows,height,limits,row,range,height_room,coverage(global)))

type Slice is Data:
  Slice{source: Execution.Source,original: Buffer.Model<F32>,global_base: U32,global_plane: U32,height: U32,width: U32}
def source(image: Slice) -> Execution.Source:
  Slice{source,original,base,plane,h,w} = image
  source
def original(image: Slice) -> Buffer.Model<F32>:
  Slice{source,original,base,plane,h,w} = image
  original
def global_base(image: Slice) -> U32:
  Slice{source,original,base,plane,h,w} = image
  base
def global_plane(image: Slice) -> U32:
  Slice{source,original,base,plane,h,w} = image
  plane
def height(image: Slice) -> U32:
  Slice{source,original,base,plane,h,w} = image
  h
def width(image: Slice) -> U32:
  Slice{source,original,base,plane,h,w} = image
  w
def local_row(+image: Slice,+row: U32) -> U32:
  (row - Execution.source_first(source(image)) : U32)
def local_mask(+image: Slice,+row: U32,+column: U32) -> Bool:
  U32.is_lt(local_row(image,row),Execution.source_rows(source(image))) && U32.is_lt(column,width(image))
def global_mask(+image: Slice,+row: U32,+column: U32) -> Bool:
  U32.is_lt(row,height(image)) && U32.is_lt(column,width(image))
def local_read(+image: Slice,+output: U32,+row: U32,+column: U32) -> F32:
  Tree.read(F32,Execution.source_data(source(image)),(Execution.source_base(source(image)) + output * Execution.source_plane(source(image)) + local_row(image,row) * width(image) + column : U32))
def global_read(+image: Slice,+output: U32,+row: U32,+column: U32) -> F32:
  Tree.read(F32,Buffer.tree(~F32,original(image)),(global_base(image) + output * global_plane(image) + row * width(image) + column : U32))
def local_pixel(+image: Slice,+output: U32,+row: U32,+column: U32) -> F32:
  Samples.value(Execution.source_data(source(image)),local_row(image,row),column,(Execution.source_base(source(image)) + output * Execution.source_plane(source(image)) : U32),Execution.source_rows(source(image)),width(image))
def global_pixel(+image: Slice,+output: U32,+row: U32,+column: U32) -> F32:
  Samples.value(Buffer.tree(~F32,original(image)),row,column,(global_base(image) + output * global_plane(image) : U32),height(image),width(image))
def pixel_same_case(enabled: Bool,+image: Slice,+output: U32,+row: U32,+column: U32,
  +local_same: {local_mask(image,row,column) == enabled : Bool},+global_same: {global_mask(image,row,column) == enabled : Bool},
  +correct: Boolean.When(enabled,{local_read(image,output,row,column) == global_read(image,output,row,column) : F32})) ->
  {local_pixel(image,output,row,column) == global_pixel(image,output,row,column) : F32}:
  match enabled:
    case False{}:
      %Equal.sym(Bool,local_mask(image,row,column),False{},local_same) : {Masks.masked(local_read(image,output,row,column),_) == global_pixel(image,output,row,column) : F32}
      %Equal.sym(Bool,global_mask(image,row,column),False{},global_same) : {0.0 == Masks.masked(global_read(image,output,row,column),_) : F32}
      {==}
    case True{}:
      %Equal.sym(Bool,local_mask(image,row,column),True{},local_same) : {Masks.masked(local_read(image,output,row,column),_) == global_pixel(image,output,row,column) : F32}
      %Equal.sym(Bool,global_mask(image,row,column),True{},global_same) : {local_read(image,output,row,column) == Masks.masked(global_read(image,output,row,column),_) : F32}
      correct

def conditional_correct(enabled: Bool,+image: Slice,+output: U32,+row: U32,+column: U32,
  +same: {global_mask(image,row,column) == enabled : Bool},
  correct: {global_mask(image,row,column) == True{} : Bool} -> {local_read(image,output,row,column) == global_read(image,output,row,column) : F32}) ->
  Boolean.When(enabled,{local_read(image,output,row,column) == global_read(image,output,row,column) : F32}):
  match enabled:
    case False{}: Unit{}
    case True{}: correct(same)
def pixel_same(+image: Slice,+limits: Machine.Limits,+output: U32,+row: U32,+column: U32,
  +range: Range<Execution.source_first(source(image)),Execution.source_rows(source(image)),height(image)>,
  +height_room: {Nat.is_le(U32.to_nat(height(image)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool},
  coverage: {U32.is_lt(row,height(image)) == True{} : Bool} ->
    Shape.Both<{Nat.is_le(U32.to_nat(Execution.source_first(source(image))),U32.to_nat(row)) == True{} : Bool},
      {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(Execution.source_first(source(image))),U32.to_nat(Execution.source_rows(source(image))))) == True{} : Bool}>,
  correct: {global_mask(image,row,column) == True{} : Bool} -> {local_read(image,output,row,column) == global_read(image,output,row,column) : F32}) ->
  {local_pixel(image,output,row,column) == global_pixel(image,output,row,column) : F32}:
  +row_same = row_mask_equal(Execution.source_first(source(image)),Execution.source_rows(source(image)),height(image),limits,row,range,height_room,coverage)
  +mask_same = Equal.cong(Bool,Bool,inside => inside && U32.is_lt(column,width(image)),U32.is_lt(local_row(image,row),Execution.source_rows(source(image))),U32.is_lt(row,height(image)),row_same)
  pixel_same_case(global_mask(image,row,column),image,output,row,column,mask_same,{==},conditional_correct(global_mask(image,row,column),image,output,row,column,{==},correct))

# Source contracts describe ordinary retained gradient rows and their contents.
type Covers<-query: Bands.Query,-first: U32,-rows: U32> is Type:
  Covers{at: @+row: U32 ->
    @+lower: {Nat.is_le(U32.to_nat(Copy.begin(query)),U32.to_nat(row)) == True{} : Bool} ->
    @+upper: {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(Copy.begin(query)),U32.to_nat(Copy.row_count(query)))) == True{} : Bool} ->
    Shape.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat(row)) == True{} : Bool},
      {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}>}
type Stored<-image: Slice,-outputs: U32> is Type:
  Stored{at: @+output: U32 -> @+row: U32 -> @+column: U32 ->
    @+output_inside: {U32.is_lt(output,outputs) == True{} : Bool} ->
    @+row_inside: {U32.is_lt(row,height(image)) == True{} : Bool} ->
    @+column_inside: {U32.is_lt(column,width(image)) == True{} : Bool} ->
    @+lower: {Nat.is_le(U32.to_nat(Execution.source_first(source(image))),U32.to_nat(row)) == True{} : Bool} ->
    @+upper: {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(Execution.source_first(source(image))),U32.to_nat(Execution.source_rows(source(image))))) == True{} : Bool} ->
    {local_read(image,output,row,column) == global_read(image,output,row,column) : F32}}
def covered_row_finish(+query: Bands.Query,+coordinate: U32,+first: U32,+rows: U32,
  at: @+row: U32 -> @+lower: {Nat.is_le(U32.to_nat(Copy.begin(query)),U32.to_nat(row)) == True{} : Bool} ->
    @+upper: {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(Copy.begin(query)),U32.to_nat(Copy.row_count(query)))) == True{} : Bool} ->
    Shape.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat(row)) == True{} : Bool},{Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}>,
  interval: Shape.Both<{Nat.is_le(U32.to_nat(Copy.begin(query)),U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32)),Nat.add(U32.to_nat(Copy.begin(query)),U32.to_nat(Copy.row_count(query)))) == True{} : Bool}>) ->
  Shape.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32)),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}>:
  Shape.Both{+lo,+hi} = interval
  at((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32),lo,hi)
def covered_row(+query: Bands.Query,+coordinate: U32,+first: U32,+rows: U32,covers: Covers<query,first,rows>,
  +lower: {U32.is_le(Kernel.band_low(Bands.geometry(query),Bands.first(query),Bands.last(query)),coordinate) == True{} : Bool},
  +upper: {U32.is_lt(coordinate,Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query))) == True{} : Bool},
  +row_inside: {U32.is_lt((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32),Domain.gradient_rows(Bands.geometry(query))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32)),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}>:
  Covers{at} = covers
  covered_row_finish(query,coordinate,first,rows,at,Copy.copied_interval(query,coordinate,lower,upper,row_inside))

def Interval(+first: U32,+rows: U32,+row: U32) -> Data:
  Shape.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat(row)) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}>
def retained_case(enabled: Bool,+query: Bands.Query,+coordinate: U32,+first: U32,+rows: U32,
  covers: Covers<query,first,rows>,
  +lower: {U32.is_le(Kernel.band_low(Bands.geometry(query),Bands.first(query),Bands.last(query)),coordinate) == True{} : Bool},
  +upper: {U32.is_lt(coordinate,Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query))) == True{} : Bool},
  +same: {U32.is_lt((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32),Domain.gradient_rows(Bands.geometry(query))) == enabled : Bool}) ->
  Boolean.When(enabled,Interval(first,rows,(coordinate - Domain.kernel_word(Bands.geometry(query)) : U32))):
  match enabled:
    case False{}: Unit{}
    case True{}: covered_row(query,coordinate,first,rows,covers,lower,upper,same)
def interval_required(enabled: Bool,+first: U32,+rows: U32,+row: U32,
  interval: Boolean.When(enabled,Interval(first,rows,row)),truth: {enabled == True{} : Bool}) -> Interval(first,rows,row):
  match enabled:
    case False{}: Empty.absurd(Interval(first,rows,row),Words.false_true(truth))
    case True{}: interval

def stored_read(+image: Slice,+outputs: U32,+output: U32,+row: U32,+column: U32,
  stored: Stored<image,outputs>,+output_inside: {U32.is_lt(output,outputs) == True{} : Bool},
  +enabled: {global_mask(image,row,column) == True{} : Bool},
  interval: Interval(Execution.source_first(source(image)),Execution.source_rows(source(image)),row)) ->
  {local_read(image,output,row,column) == global_read(image,output,row,column) : F32}:
  Stored{at} = stored
  Shape.Both{+lo,+hi} = interval
  at(output,row,column,output_inside,Boolean.left(U32.is_lt(row,height(image)),U32.is_lt(column,width(image)),enabled),
    Boolean.right(U32.is_lt(row,height(image)),U32.is_lt(column,width(image)),enabled),lo,hi)
def retained_pixel(+image: Slice,+limits: Machine.Limits,+outputs: U32,+output: U32,+row: U32,+column: U32,
  +range: Range<Execution.source_first(source(image)),Execution.source_rows(source(image)),height(image)>,
  +height_room: {Nat.is_le(U32.to_nat(height(image)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool},
  +interval: Boolean.When(U32.is_lt(row,height(image)),Interval(Execution.source_first(source(image)),Execution.source_rows(source(image)),row)),
  stored: Stored<image,outputs>,+output_inside: {U32.is_lt(output,outputs) == True{} : Bool}) ->
  {local_pixel(image,output,row,column) == global_pixel(image,output,row,column) : F32}:
  pixel_same(image,limits,output,row,column,range,height_room,
    truth => interval_required(U32.is_lt(row,height(image)),Execution.source_first(source(image)),Execution.source_rows(source(image)),row,interval,truth),
    enabled =>
      +enabled = enabled
      stored_read(image,outputs,output,row,column,stored,output_inside,enabled,
        interval_required(U32.is_lt(row,height(image)),Execution.source_first(source(image)),Execution.source_rows(source(image)),row,interval,
          Boolean.left(U32.is_lt(row,height(image)),U32.is_lt(column,width(image)),enabled))))

def query_slice(+query: Bands.Query,input: Execution.Source,original: Buffer.Model<F32>,base: U32,plane: U32) -> Slice:
  Slice{input,original,base,plane,Domain.gradient_rows(Bands.geometry(query)),Domain.gradient_columns(Bands.geometry(query))}
def query_height_room(+query: Bands.Query) ->
  {Nat.is_le(U32.to_nat(Domain.gradient_rows(Bands.geometry(query))),U32.to_nat(Machine.coordinate(Bands.query_limits(query)))) == True{} : Bool}:
  Bands.Query{Extents.Axis{+call,size,size_room,e},first,last,ordered,room} = query
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Order.transitive(Domain.gradient_height(g),Nat.add(Domain.gradient_height(g),Domain.kernel(g)),U32.to_nat(Machine.coordinate(Bands.query_limits(query))),
    Order.le_add(Domain.gradient_height(g),Domain.kernel(g)),Copy.query_gradient_kernel_room(query))
def halo_pixel(+query: Bands.Query,+input: Execution.Source,+original: Buffer.Model<F32>,+base: U32,+plane: U32,
  +outputs: U32,+output: U32,+coordinate: U32,+column: U32,
  +range: Range<Execution.source_first(input),Execution.source_rows(input),Domain.gradient_rows(Bands.geometry(query))>,
  covers: Covers<query,Execution.source_first(input),Execution.source_rows(input)>,
  stored: Stored<query_slice(query,input,original,base,plane),outputs>,
  +output_inside: {U32.is_lt(output,outputs) == True{} : Bool},
  +lower: {U32.is_le(Kernel.band_low(Bands.geometry(query),Bands.first(query),Bands.last(query)),coordinate) == True{} : Bool},
  +upper: {U32.is_lt(coordinate,Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query))) == True{} : Bool}) ->
  {local_pixel(query_slice(query,input,original,base,plane),output,(coordinate - Domain.kernel_word(Bands.geometry(query)) : U32),column) ==
    global_pixel(query_slice(query,input,original,base,plane),output,(coordinate - Domain.kernel_word(Bands.geometry(query)) : U32),column) : F32}:
  retained_pixel(query_slice(query,input,original,base,plane),Bands.query_limits(query),outputs,output,(coordinate - Domain.kernel_word(Bands.geometry(query)) : U32),column,
    range,query_height_room(query),
    retained_case(U32.is_lt((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32),Domain.gradient_rows(Bands.geometry(query))),query,coordinate,Execution.source_first(input),Execution.source_rows(input),covers,lower,upper,{==}),
    stored,output_inside)

def full_source(+query: Bands.Query,original: Buffer.Model<F32>,base: U32,plane: U32) -> Execution.Source:
  Execution.Source{Buffer.tree(~F32,original),base,plane,0,Domain.gradient_rows(Bands.geometry(query))}
def full_range(+query: Bands.Query) -> Range<0,Domain.gradient_rows(Bands.geometry(query)),Domain.gradient_rows(Bands.geometry(query))>:
  ImageRows{Order.reflexive(U32.to_nat(Domain.gradient_rows(Bands.geometry(query))))}
def full_cover(+query: Bands.Query) -> Covers<query,0,Domain.gradient_rows(Bands.geometry(query))>:
  Bands.Query{Extents.Axis{+call,size,size_room,e},first,last,ordered,room} = query
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Covers{row => lower => upper =>
    +row = row
    Shape.Both{Order.zero_le(U32.to_nat(row)),Copy.required_row_inside(query,row,lower,upper)}}
def copy_cover(+query: Bands.Query) -> Covers<query,Kernel.copy_start(Bands.geometry(query),Bands.first(query),Bands.rows_word(query)),Kernel.copy_count(Bands.geometry(query),Bands.first(query),Bands.rows_word(query))>:
  %Equal.sym(U32,Kernel.copy_start(Bands.geometry(query),Bands.first(query),Bands.rows_word(query)),Copy.begin(query),Copy.begin_matches(query)) :
    Covers<query,_,Kernel.copy_count(Bands.geometry(query),Bands.first(query),Bands.rows_word(query))>
  %Equal.sym(U32,Kernel.copy_count(Bands.geometry(query),Bands.first(query),Bands.rows_word(query)),Copy.row_count(query),Copy.count_matches(query)) : Covers<query,Copy.begin(query),_>
  Covers{row => lower => upper => Shape.Both{lower,upper}}
def full_read(+query: Bands.Query,+original: Buffer.Model<F32>,+base: U32,+plane: U32,+output: U32,+row: U32,+column: U32) ->
  {local_read(query_slice(query,full_source(query,original,base,plane),original,base,plane),output,row,column) ==
    global_read(query_slice(query,full_source(query,original,base,plane),original,base,plane),output,row,column) : F32}:
  %Equal.sym(U32,(row - 0 : U32),row,Arithmetic.subtract_zero(row)) :
    {Tree.read(F32,Buffer.tree(~F32,original),(base + output * plane + _ * Domain.gradient_columns(Bands.geometry(query)) + column : U32)) ==
      global_read(query_slice(query,full_source(query,original,base,plane),original,base,plane),output,row,column) : F32}
  {==}
def full_stored(+query: Bands.Query,+original: Buffer.Model<F32>,+base: U32,+plane: U32,+outputs: U32) ->
  Stored<query_slice(query,full_source(query,original,base,plane),original,base,plane),outputs>:
  Stored{output => row => column => output_inside => row_inside => column_inside => lower => upper => full_read(query,original,base,plane,output,row,column)}
