# F.matmul: the products of the rows of left and the matrix of right of
# their image, by the image panels: every image multiplies its rows with its
# own matrix of right, read where it lies.
import Base
import ../../lib/tensor_view.bend as Views
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/tensor_grid.bend as Grid
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/tensor_tiled_public.bend as TiledPublic
import ../../lib/proofs/tensor_cell_source.bend as CellSource
import ../../lib/proofs/tensor_tiled_swapped.bend as Swapped
import ../../lib/proofs/tensor_view_reads.bend as ViewReads
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/word_multiplication.bend as WordMultiplication
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_view_gather.bend as ViewGather
import ../../lib/proofs/tensor_grid_box.bend as Box
import ../../lib/proofs/tensor_view_part.bend as ViewPart
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/contiguous_copy_readback.bend as CopyReadback
import ../../lib/proofs/tensor_tiled_entry.bend as TiledEntry
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/word_power_of_two.bend as Power
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ./image_problems.bend as ImageProblems
import ./input_storage.bend as Inputs
import ../../lib/machine_limits.bend as Machine
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/tensor_full_values.bend as FullValues
import ./product_values.bend as Products
import ./linear_call.bend as Linear
import ./panel_point_product_contract.bend as PanelContract
import ../specification.bend as ConvSpec
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./list_array_roundtrip.bend as ListArrayRoundtrip
import ./convolution_public.bend as Run
import ./convolution_call.bend as Call
import ./window_packed_values.bend as Calls
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/tensor_matrix_products.bend as TiledProducts
import ../../lib/kernel_window_pipeline.bend as WindowPipeline
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/default_machine_limits.bend as Default
import ../../lib/tensor_pieces.bend as Pieces
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/tensor_view_bounds.bend as ViewBounds
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/tensor_shape.bend as Shapes
import ./convolution_parameters.bend as Parameters
import ../../lib/proofs/tensor_pieces_model.bend as PiecesModel
import ../../lib/proofs/tensor_joined.bend as Joined
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/tensor_view_moved.bend as ViewMoved
import ../../lib/proofs/tensor_storage_count.bend as StorageCount
import ../../lib/proofs/tensor_relabeled.bend as Relabel
import ../../lib/proofs/tensor_count_completion.bend as CountCompletion
import ../../lib/proofs/tensor_tiled_relane.bend as Relane
import ../../lib/proofs/tensor_cell_room.bend as CellRoom
import ../../lib/tensor_f32.bend as F
import ../../lib/tensor_tiled_layout.bend as TiledLayout
import ../../lib/proofs/tensor_tiled_reshape.bend as Reshape

# What the runtime guard of the products of matrices checks.
type Guarded<-count: U32,-rows: U32,-inner: U32,-outputs: U32,-limits: Machine.Limits> is Data:
  Guarded{count_positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
    inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},outputs_positive: {Nat.is_lt(0n,U32.to_nat(outputs)) == True{} : Bool},
    outputs_room: {Nat.is_le(U32.to_nat(outputs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    inner_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    products_room: {Nat.is_le(Nat.mul(U32.to_nat(count),Nat.mul(U32.to_nat(outputs),U32.to_nat(rows))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def guarded(+count: U32,+rows: U32,+inner: U32,+outputs: U32,+limits: Machine.Limits,+supported: {TiledProducts.supported(count,rows,inner,outputs,limits) == True{} : Bool}) ->
  Guarded<count,rows,inner,outputs,limits>:
  +capacity = U32.to_nat(Machine.capacity(limits))
  +a = U32.is_lt(0,count)
  +b = U32.is_lt(0,rows)
  +c = U32.is_lt(0,inner)
  +d = U32.is_lt(0,outputs)
  +e = Nat.is_le(Nat.add(U32.to_nat(rows),7n),capacity)
  +f = Nat.is_le(Nat.add(U32.to_nat(outputs),7n),capacity)
  +g = Nat.is_le(U32.to_nat(inner),capacity)
  +h = Nat.is_le(Nat.mul(U32.to_nat(count),Nat.mul(U32.to_nat(outputs),U32.to_nat(rows))),capacity)
  +upto_g = Boolean.left(a && b && c && d && e && f && g,h,supported)
  +upto_f = Boolean.left(a && b && c && d && e && f,g,upto_g)
  +upto_e = Boolean.left(a && b && c && d && e,f,upto_f)
  +upto_d = Boolean.left(a && b && c && d,e,upto_e)
  +upto_c = Boolean.left(a && b && c,d,upto_d)
  +upto_b = Boolean.left(a && b,c,upto_c)
  Guarded{Bounds.as_nat_lt(0,count,Boolean.left(a,b,upto_b)),Bounds.as_nat_lt(0,rows,Boolean.right(a,b,upto_b)),Bounds.as_nat_lt(0,inner,Boolean.right(a && b,c,upto_c)),
    Bounds.as_nat_lt(0,outputs,Boolean.right(a && b && c,d,upto_d)),
    Order.transitive(U32.to_nat(outputs),Nat.add(U32.to_nat(outputs),7n),capacity,Order.le_add(U32.to_nat(outputs),7n),Boolean.right(a && b && c && d && e,f,upto_f)),
    Boolean.right(a && b && c && d && e && f,g,upto_g),Boolean.right(a && b && c && d && e && f && g,h,supported)}

# Packed rows of left, matrices of right and a bias of +0 that the guard
# supports hold the products of matrices.
def matrices_packed(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,+left: Left,+right: Right,+count: U32,+rows: U32,+inner: U32,+outputs: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +guard: Guarded<count,rows,inner,outputs,limits>,+lm: PiecesModel.Model<Tile.Tile8>,+ls: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+la: Nat,+ll: U32,+lp: Grid.Position,+left_contract: TiledPublic.Contract(~Left,~left_value,left,[count,rows,inner],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),ls,lf,Tensors.Usable{},lw},la,ll,lp}),
  +rm: PiecesModel.Model<Tile.Tile8>,+rs: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ra: Nat,+rl: U32,+rp: Grid.Position,+right_contract: TiledPublic.Contract(~Right,~right_value,right,[count,inner,outputs],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rs,rf,Tensors.Usable{},rw},ra,rl,rp}),
  +qm: PiecesModel.Model<Tile.Tile8>,+qs: List<&2,U32>,+qf: Tile.Tile8,+qw: U32,+qa: Nat,+ql: U32,+qp: Grid.Position,+bias_contract: TiledPublic.Contract(~F32,~FullValues.constant(~F32),0.0,[outputs],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,qm),qs,qf,Tensors.Usable{},qw},qa,ql,qp}),+workers: U32) ->
  TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(outputs)},[count,rows,outputs],TiledProducts.lines(count,rows,inner,outputs,workers,2n,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),ls,lf,Tensors.Usable{},lw},la,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rs,rf,Tensors.Usable{},rw},ra,rl,rp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,qm),qs,qf,Tensors.Usable{},qw},qa,ql,qp})):
  Guarded{+count_positive,+rows_positive,+inner_positive,+outputs_positive,+outputs_room,+inner_room,+products_room} = guard
  KernelGemm.Both{+right_global,TiledPublic.Laid{+right_inside,+right_shaped,+right_restored,+right_counted,+right_pinned}} = right_contract
  +C = U32.to_nat(count)
  +R = U32.to_nat(rows)
  +I = U32.to_nat(inner)
  +O = U32.to_nat(outputs)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +seven_room = Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits))
  +logical_room = {%Equal.sym(Nat,ShapeLaws.elements([count,rows,outputs]),Nat.mul(C,Nat.mul(R,O)),ShapeLaws.elements_three(count,rows,outputs)) :
      {Nat.is_le(_,capacity) == True{} : Bool}
    %RowMajor.multiply_commutative(O,R) : {Nat.is_le(Nat.mul(C,_),capacity) == True{} : Bool} products_room :
    {Nat.is_le(ShapeLaws.elements([count,rows,outputs]),capacity) == True{} : Bool}}
  +left_nonempty = {%Equal.sym(Nat,Nat.mul(I,1n),I,Order.times_one(I)) : {Nat.is_lt(0n,Nat.mul(C,Nat.mul(R,_))) == True{} : Bool}
    Order.product_positive(C,Nat.mul(R,I),count_positive,Order.product_positive(R,I,rows_positive,inner_positive)) : {Nat.is_lt(0n,ShapeLaws.elements([count,rows,inner])) == True{} : Bool}}
  +right_nonempty = {%Equal.sym(Nat,Nat.mul(O,1n),O,Order.times_one(O)) : {Nat.is_lt(0n,Nat.mul(C,Nat.mul(I,_))) == True{} : Bool}
    Order.product_positive(C,Nat.mul(I,O),count_positive,Order.product_positive(I,O,inner_positive,outputs_positive)) : {Nat.is_lt(0n,ShapeLaws.elements([count,inner,outputs])) == True{} : Bool}}
  +bias_nonempty = {%Equal.sym(Nat,Nat.mul(O,1n),O,Order.times_one(O)) : {Nat.is_lt(0n,_) == True{} : Bool} outputs_positive : {Nat.is_lt(0n,ShapeLaws.elements([outputs])) == True{} : Bool}}
  PanelContract.panel_arranged(~Left,~left_value,~Right,~right_value,left,right,limits,count,rows,inner,outputs,workers,default_capacity,count_positive,rows_positive,
    inner_positive,outputs_positive,inner_room,outputs_room,logical_room,StorageCount.counted_room(limits,[count,inner,outputs],right_counted,default_capacity),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),ls,lf,Tensors.Usable{},lw},la,ll,lp},Relane.arranged_held(~Left,~left_value,left,[count,rows,inner],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),ls,lf,Tensors.Usable{},lw},la,ll,lp},2n,Grid.Innermost{},limits,default_capacity,{==},left_nonempty,
      Order.transitive(3n,7n,capacity,{==},seven_room),left_contract),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rs,rf,Tensors.Usable{},rw},ra,rl,rp},Relane.arranged_held(~Right,~right_value,right,[count,inner,outputs],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rs,rf,Tensors.Usable{},rw},ra,rl,rp},2n,Grid.Innermost{},limits,default_capacity,{==},right_nonempty,
      Order.transitive(3n,7n,capacity,{==},seven_room),KernelGemm.Both{right_global,TiledPublic.Laid{right_inside,right_shaped,right_restored,right_counted,right_pinned}}),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,qm),qs,qf,Tensors.Usable{},qw},qa,ql,qp},Relane.arranged_held(~F32,~FullValues.constant(~F32),0.0,[outputs],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,qm),qs,qf,Tensors.Usable{},qw},qa,ql,qp},0n,Grid.Innermost{},limits,default_capacity,{==},bias_nonempty,
      Order.transitive(1n,7n,capacity,{==},seven_room),bias_contract))

# Rows of left [count,rows,inner], matrices of right [count,inner,outputs] and
# a bias of +0 of the contract that the guard supports hold the products of
# matrices.
def matrices_contract(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,+left: Left,+right: Right,+count: U32,+rows: U32,+inner: U32,+outputs: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +guard: Guarded<count,rows,inner,outputs,limits>,left_tiled: Tiled.Tiled,+left_contract: TiledPublic.Contract(~Left,~left_value,left,[count,rows,inner],left_tiled),
  right_tiled: Tiled.Tiled,+right_contract: TiledPublic.Contract(~Right,~right_value,right,[count,inner,outputs],right_tiled),
  bias_tiled: Tiled.Tiled,+bias_contract: TiledPublic.Contract(~F32,~FullValues.constant(~F32),0.0,[outputs],bias_tiled),+workers: U32) ->
  TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(outputs)},[count,rows,outputs],TiledProducts.lines(count,rows,inner,outputs,workers,2n,left_tiled,right_tiled,bias_tiled)):
  TiledPublic.by_packed(~Left,~left_value,(tiled => TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(outputs)},[count,rows,outputs],TiledProducts.lines(count,rows,inner,outputs,workers,2n,tiled,right_tiled,bias_tiled))),left,[count,rows,inner],left_tiled,left_contract,
    lm => ls => lf => lw => la => ll => lp => left_packed => TiledPublic.by_packed(~Right,~right_value,(tiled => TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(outputs)},[count,rows,outputs],TiledProducts.lines(count,rows,inner,outputs,workers,2n,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),ls,lf,Tensors.Usable{},lw},la,ll,lp},tiled,bias_tiled))),right,[count,inner,outputs],right_tiled,right_contract,
      rm => rs => rf => rw => ra => rl => rp => right_packed => TiledPublic.by_packed(~F32,~FullValues.constant(~F32),(tiled => TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(outputs)},[count,rows,outputs],TiledProducts.lines(count,rows,inner,outputs,workers,2n,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),ls,lf,Tensors.Usable{},lw},la,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rs,rf,Tensors.Usable{},rw},ra,rl,rp},tiled))),0.0,[outputs],bias_tiled,bias_contract,
        qm => qs => qf => qw => qa => ql => qp => bias_packed =>
          matrices_packed(~Left,~left_value,~Right,~right_value,left,right,count,rows,inner,outputs,limits,default_capacity,guard,lm,ls,lf,lw,la,ll,lp,left_packed,rm,rs,rf,rw,ra,rl,rp,right_packed,qm,qs,qf,qw,qa,ql,qp,bias_packed,workers))))

# ---- F.matmul ----

# The matrices of right: one that every row of left shares, or one an image.
def images_of(+leading: List<&2,U32>,+right_leading: List<&2,U32>) -> U32:
  Bool.pick(U32,List.is_empty(&2,U32,right_leading),1,Tensors.count_or_zero(leading))

# The rows of left that one matrix of right serves.
def served_rows(+leading: List<&2,U32>,+right_leading: List<&2,U32>,+rows: U32) -> U32:
  Bool.pick(U32,List.is_empty(&2,U32,right_leading),(Tensors.count_or_zero(leading) * rows : U32),rows)

# The elements of left, right and the result read as the images.
type Counts<-leading: List<&2,U32>,-right_leading: List<&2,U32>,-rows: U32,-inner: U32,-outputs: U32,-images: U32,-image_rows: U32> is Data:
  Counts{left_exact: {ShapeLaws.elements(List.append(&2,U32,leading,[rows,inner])) == ShapeLaws.elements([images,image_rows,inner]) : Nat},
    right_exact: {ShapeLaws.elements(List.append(&2,U32,right_leading,[inner,outputs])) == ShapeLaws.elements([images,inner,outputs]) : Nat},
    output_exact: {ShapeLaws.elements(List.append(&2,U32,leading,[rows,outputs])) == ShapeLaws.elements([images,image_rows,outputs]) : Nat}}

# Extents that are the same or none.
def paired_same(same: Bool,+paired: {(same || False{}) == True{} : Bool}) -> {same == True{} : Bool}:
  match same:
    case True{}: {==}
    case False{}: paired

# The leading axes as one axis of images.
def leading_counts(+leading: List<&2,U32>,+head: U32,+last: U32,+leading_exact: {U32.to_nat(Tensors.count_or_zero(leading)) == ShapeLaws.elements(leading) : Nat}) ->
  {ShapeLaws.elements(List.append(&2,U32,leading,[head,last])) == ShapeLaws.elements([Tensors.count_or_zero(leading),head,last]) : Nat}:
  %Equal.sym(Nat,U32.to_nat(Tensors.count_or_zero(leading)),ShapeLaws.elements(leading),leading_exact) :
    {ShapeLaws.elements(List.append(&2,U32,leading,[head,last])) == Nat.mul(_,ShapeLaws.elements([head,last])) : Nat}
  ShapeLaws.elements_appended(leading,[head,last])

# The leading axes and the rows as the rows of one image.
def shared_counts(+leading: List<&2,U32>,+rows: U32,+last: U32,
  +all_exact: {U32.to_nat((Tensors.count_or_zero(leading) * rows : U32)) == Nat.mul(ShapeLaws.elements(leading),U32.to_nat(rows)) : Nat}) ->
  {ShapeLaws.elements(List.append(&2,U32,leading,[rows,last])) == ShapeLaws.elements([1,(Tensors.count_or_zero(leading) * rows : U32),last]) : Nat}:
  +all = (Tensors.count_or_zero(leading) * rows : U32)
  +E = ShapeLaws.elements(leading)
  +R = U32.to_nat(rows)
  +L = Nat.mul(U32.to_nat(last),1n)
  %Equal.sym(Nat,Nat.add(Nat.mul(U32.to_nat(all),L),0n),Nat.mul(U32.to_nat(all),L),Addition.zero_right(Nat.mul(U32.to_nat(all),L))) :
    {ShapeLaws.elements(List.append(&2,U32,leading,[rows,last])) == _ : Nat}
  %Equal.sym(Nat,U32.to_nat(all),Nat.mul(E,R),all_exact) : {ShapeLaws.elements(List.append(&2,U32,leading,[rows,last])) == Nat.mul(_,L) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(E,R),L),Nat.mul(E,Nat.mul(R,L)),RowMajor.assoc(E,R,L)) : {ShapeLaws.elements(List.append(&2,U32,leading,[rows,last])) == _ : Nat}
  ShapeLaws.elements_appended(leading,[rows,last])

# Left, right and the result read as the images the runtime forms.
def counts(+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+leading: List<&2,U32>,right_leading: List<&2,U32>,
  +rows: U32,+inner: U32,+outputs: U32,+x_counted: Global.Counted<List.append(&2,U32,leading,[rows,inner])>,
  +paired: {(Shapes.same_extents(right_leading,leading) || List.is_empty(&2,U32,right_leading)) == True{} : Bool},
  +guard: Guarded<images_of(leading,right_leading),served_rows(leading,right_leading,rows),inner,outputs,limits>) ->
  Counts<leading,right_leading,rows,inner,outputs,images_of(leading,right_leading),served_rows(leading,right_leading,rows)>:
  match right_leading:
    case []:
      Guarded{+count_positive,+all_positive,+inner_positive,+outputs_positive,+outputs_room,+inner_room,+products_room} = guard
      +count = Tensors.count_or_zero(leading)
      +E = ShapeLaws.elements(leading)
      +R = U32.to_nat(rows)
      +I = U32.to_nat(inner)
      +inner_one = {%Equal.sym(Nat,Nat.mul(I,1n),I,Order.times_one(I)) : {Nat.is_lt(0n,_) == True{} : Bool} inner_positive : {Nat.is_lt(0n,Nat.mul(I,1n)) == True{} : Bool}}
      +tail_positive = {Order.product_positive(R,Nat.mul(I,1n),Linear.factor_cases(R,count,rows,{==},all_positive),inner_one) : {Nat.is_lt(0n,ShapeLaws.elements([rows,inner])) == True{} : Bool}}
      +leading_exact = Linear.leading_word(limits,default_capacity,leading,[rows,inner],x_counted,tail_positive)
      +rows_within = {%Equal.sym(Nat,ShapeLaws.elements(List.append(&2,U32,leading,[rows,inner])),Nat.mul(E,ShapeLaws.elements([rows,inner])),ShapeLaws.elements_appended(leading,[rows,inner])) :
          {Nat.is_le(Nat.mul(E,R),_) == True{} : Bool}
        %RowMajor.assoc(E,R,Nat.mul(I,1n)) : {Nat.is_le(Nat.mul(E,R),_) == True{} : Bool}
        RowMajor.product_covers(Nat.mul(E,R),Nat.mul(I,1n),inner_one) : {Nat.is_le(Nat.mul(E,R),ShapeLaws.elements(List.append(&2,U32,leading,[rows,inner]))) == True{} : Bool}}
      +all_exact = Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),count,rows,E,R,leading_exact,{==},
        Order.transitive(Nat.mul(E,R),ShapeLaws.elements(List.append(&2,U32,leading,[rows,inner])),U32.to_nat(Machine.capacity(limits)),rows_within,StorageCount.counted_room(limits,List.append(&2,U32,leading,[rows,inner]),x_counted,default_capacity)))
      +plane = Nat.mul(I,Nat.mul(U32.to_nat(outputs),1n))
      Counts{shared_counts(leading,rows,inner,all_exact),Equal.sym(Nat,Nat.add(plane,0n),plane,Addition.zero_right(plane)),shared_counts(leading,rows,outputs,all_exact)}
    case +extent <> +rest:
      Guarded{+count_positive,+rows_positive,+inner_positive,+outputs_positive,+outputs_room,+inner_room,+products_room} = guard
      +same = ShapeLaws.same_extents(extent <> rest,leading,paired_same(Shapes.same_extents(extent <> rest,leading),paired))
      +I = U32.to_nat(inner)
      +inner_one = {%Equal.sym(Nat,Nat.mul(I,1n),I,Order.times_one(I)) : {Nat.is_lt(0n,_) == True{} : Bool} inner_positive : {Nat.is_lt(0n,Nat.mul(I,1n)) == True{} : Bool}}
      +leading_exact = Linear.leading_word(limits,default_capacity,leading,[rows,inner],x_counted,Order.product_positive(U32.to_nat(rows),Nat.mul(I,1n),rows_positive,inner_one))
      Counts{leading_counts(leading,rows,inner,leading_exact),
        {%Equal.sym(List<&2,U32>,extent <> rest,leading,same) :
            {ShapeLaws.elements(List.append(&2,U32,_,[inner,outputs])) == ShapeLaws.elements([Tensors.count_or_zero(leading),inner,outputs]) : Nat}
          leading_counts(leading,inner,outputs,leading_exact) :
          {ShapeLaws.elements(List.append(&2,U32,extent <> rest,[inner,outputs])) == ShapeLaws.elements([Tensors.count_or_zero(leading),inner,outputs]) : Nat}},
        leading_counts(leading,rows,outputs,leading_exact)}

# Left and right of the contract read as the images, with a bias of +0, hold
# the products of matrices read back in the shape of left.
def products_reshaped(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,+left: Left,+right: Right,+leading: List<&2,U32>,+right_leading: List<&2,U32>,
  +rows: U32,+inner: U32,+outputs: U32,+images: U32,+image_rows: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_contract: TiledPublic.Contract(~Left,~left_value,left,List.append(&2,U32,leading,[rows,inner]),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}),+x_counted: Global.Counted<List.append(&2,U32,leading,[rows,inner])>,
  +ym: PiecesModel.Model<Tile.Tile8>,+ys: List<&2,U32>,+yf: Tile.Tile8,+yw: U32,+ya: Nat,+yl: U32,+yp: Grid.Position,+y_contract: TiledPublic.Contract(~Right,~right_value,right,List.append(&2,U32,right_leading,[inner,outputs]),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp}),+y_counted: Global.Counted<List.append(&2,U32,right_leading,[inner,outputs])>,
  +counts: Counts<leading,right_leading,rows,inner,outputs,images,image_rows>,+guard: Guarded<images,image_rows,inner,outputs,limits>,
  +rank_room: {Nat.is_le(List.length(&2,U32,List.append(&2,U32,leading,[rows,inner])),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank: {Nat.is_le(List.length(&2,U32,List.append(&2,U32,right_leading,[inner,outputs])),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+workers: U32) ->
  TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(image_rows),U32.to_nat(inner),U32.to_nat(outputs)},List.append(&2,U32,leading,[rows,outputs]),TiledLayout.reshape(TiledProducts.lines(images,image_rows,inner,outputs,workers,2n,TiledLayout.reshape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},[images,image_rows,inner]),TiledLayout.reshape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},[images,inner,outputs]),
    Tiled.zeros([outputs])),List.append(&2,U32,leading,[rows,outputs]))):
  Counts{+left_exact,+right_exact,+output_exact} = counts
  Guarded{+count_positive,+rows_positive,+inner_positive,+outputs_positive,+outputs_room,+inner_room,+products_room} = guard
  +C = U32.to_nat(images)
  +R = U32.to_nat(image_rows)
  +I = U32.to_nat(inner)
  +O = U32.to_nat(outputs)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +seven_room = Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits))
  +left_positive = {%Equal.sym(Nat,ShapeLaws.elements(List.append(&2,U32,leading,[rows,inner])),ShapeLaws.elements([images,image_rows,inner]),left_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(I,1n),I,Order.times_one(I)) : {Nat.is_lt(0n,Nat.mul(C,Nat.mul(R,_))) == True{} : Bool}
    Order.product_positive(C,Nat.mul(R,I),count_positive,Order.product_positive(R,I,rows_positive,inner_positive)) : {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,leading,[rows,inner]))) == True{} : Bool}}
  +right_positive = {%Equal.sym(Nat,ShapeLaws.elements(List.append(&2,U32,right_leading,[inner,outputs])),ShapeLaws.elements([images,inner,outputs]),right_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(O,1n),O,Order.times_one(O)) : {Nat.is_lt(0n,Nat.mul(C,Nat.mul(I,_))) == True{} : Bool}
    Order.product_positive(C,Nat.mul(I,O),count_positive,Order.product_positive(I,O,inner_positive,outputs_positive)) : {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,right_leading,[inner,outputs]))) == True{} : Bool}}
  +left_same = Equal.trans(Maybe<&2,U32>,Tensors.count_of([images,image_rows,inner]),Some{Global.count(List.append(&2,U32,leading,[rows,inner]),x_counted)},Tensors.count_of(List.append(&2,U32,leading,[rows,inner])),
    Relabel.counted_same(List.append(&2,U32,leading,[rows,inner]),[images,image_rows,inner],x_counted,left_exact),
    Equal.sym(Maybe<&2,U32>,Tensors.count_of(List.append(&2,U32,leading,[rows,inner])),Some{Global.count(List.append(&2,U32,leading,[rows,inner]),x_counted)},Global.accepted(List.append(&2,U32,leading,[rows,inner]),x_counted)))
  +right_same = Equal.trans(Maybe<&2,U32>,Tensors.count_of([images,inner,outputs]),Some{Global.count(List.append(&2,U32,right_leading,[inner,outputs]),y_counted)},Tensors.count_of(List.append(&2,U32,right_leading,[inner,outputs])),
    Relabel.counted_same(List.append(&2,U32,right_leading,[inner,outputs]),[images,inner,outputs],y_counted,right_exact),
    Equal.sym(Maybe<&2,U32>,Tensors.count_of(List.append(&2,U32,right_leading,[inner,outputs])),Some{Global.count(List.append(&2,U32,right_leading,[inner,outputs]),y_counted)},Global.accepted(List.append(&2,U32,right_leading,[inner,outputs]),y_counted)))
  +left_reshaped = Reshape.reshaped(~Left,~left_value,left,limits,List.append(&2,U32,leading,[rows,inner]),[images,image_rows,inner],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},CountCompletion.positive_not_zero(List.append(&2,U32,leading,[rows,inner]),left_positive),left_same,rank_room,Order.transitive(3n,7n,capacity,{==},seven_room),default_capacity,x_contract)
  +right_reshaped = Reshape.reshaped(~Right,~right_value,right,limits,List.append(&2,U32,right_leading,[inner,outputs]),[images,inner,outputs],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},CountCompletion.positive_not_zero(List.append(&2,U32,right_leading,[inner,outputs]),right_positive),right_same,right_rank,Order.transitive(3n,7n,capacity,{==},seven_room),default_capacity,y_contract)
  +output_values = {%Equal.sym(Nat,Nat.mul(O,1n),O,Order.times_one(O)) : {Nat.mul(C,Nat.mul(R,_)) == Nat.mul(C,Nat.mul(O,R)) : Nat}
    Equal.cong(Nat,Nat,value => Nat.mul(C,value),Nat.mul(R,O),Nat.mul(O,R),RowMajor.multiply_commutative(R,O)) : {ShapeLaws.elements([images,image_rows,outputs]) == Nat.mul(C,Nat.mul(O,R)) : Nat}}
  +output_counted = Linear.counted_by(limits,default_capacity,[images,image_rows,outputs],Nat.mul(C,Nat.mul(O,R)),output_values,products_room)
  +output_positive = {%Equal.sym(Nat,ShapeLaws.elements([images,image_rows,outputs]),Nat.mul(C,Nat.mul(O,R)),output_values) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(C,Nat.mul(O,R),count_positive,Order.product_positive(O,R,outputs_positive,rows_positive)) : {Nat.is_lt(0n,ShapeLaws.elements([images,image_rows,outputs])) == True{} : Bool}}
  +output_same = Equal.trans(Maybe<&2,U32>,Tensors.count_of(List.append(&2,U32,leading,[rows,outputs])),Some{Global.count([images,image_rows,outputs],output_counted)},Tensors.count_of([images,image_rows,outputs]),
    Relabel.counted_same([images,image_rows,outputs],List.append(&2,U32,leading,[rows,outputs]),output_counted,Equal.sym(Nat,ShapeLaws.elements(List.append(&2,U32,leading,[rows,outputs])),ShapeLaws.elements([images,image_rows,outputs]),output_exact)),
    Equal.sym(Maybe<&2,U32>,Tensors.count_of([images,image_rows,outputs]),Some{Global.count([images,image_rows,outputs],output_counted)},Global.accepted([images,image_rows,outputs],output_counted)))
  +bias_counted = Linear.counted_by(limits,default_capacity,[outputs],O,Order.times_one(O),outputs_room)
  +bias_nonzero = CountCompletion.positive_not_zero([outputs],{%Equal.sym(Nat,Nat.mul(O,1n),O,Order.times_one(O)) : {Nat.is_lt(0n,_) == True{} : Bool} outputs_positive :
    {Nat.is_lt(0n,ShapeLaws.elements([outputs])) == True{} : Bool}})
  +bias_zeros = FullValues.zeros([outputs],Global.count([outputs],bias_counted),Global.accepted([outputs],bias_counted),bias_nonzero)
  Reshape.reshaped(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,R,I,O},limits,[images,image_rows,outputs],List.append(&2,U32,leading,[rows,outputs]),
    TiledProducts.lines(images,image_rows,inner,outputs,workers,2n,TiledLayout.reshape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},[images,image_rows,inner]),TiledLayout.reshape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},[images,inner,outputs]),Tiled.zeros([outputs])),
    CountCompletion.positive_not_zero([images,image_rows,outputs],output_positive),output_same,Order.transitive(3n,7n,capacity,{==},seven_room),
    {%Reshape.tail_length(leading,rows,inner,rows,outputs) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
      {Nat.is_le(List.length(&2,U32,List.append(&2,U32,leading,[rows,outputs])),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},default_capacity,
    matrices_contract(~Left,~left_value,~Right,~right_value,left,right,images,image_rows,inner,outputs,limits,default_capacity,guard,
      TiledLayout.reshape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},[images,image_rows,inner]),left_reshaped,TiledLayout.reshape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},[images,inner,outputs]),right_reshaped,Tiled.zeros([outputs]),bias_zeros,workers))

# Packed left and right of the contract that the runtime guard supports hold
# the products of matrices.
def packed_matmul(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,+left: Left,+right: Right,+leading: List<&2,U32>,+right_leading: List<&2,U32>,
  +rows: U32,+inner: U32,+outputs: U32,+limits: Machine.Limits,+limits_exact: {limits == Default.limits() : Machine.Limits},
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_contract: TiledPublic.Contract(~Left,~left_value,left,List.append(&2,U32,leading,[rows,inner]),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}),
  +ym: PiecesModel.Model<Tile.Tile8>,+ys: List<&2,U32>,+yf: Tile.Tile8,+yw: U32,+ya: Nat,+yl: U32,+yp: Grid.Position,+y_contract: TiledPublic.Contract(~Right,~right_value,right,List.append(&2,U32,right_leading,[inner,outputs]),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp}),
  +paired: {(Shapes.same_extents(right_leading,leading) || List.is_empty(&2,U32,right_leading)) == True{} : Bool},
  +supported: {TiledProducts.supported(images_of(leading,right_leading),served_rows(leading,right_leading,rows),inner,outputs,limits) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,List.append(&2,U32,leading,[rows,inner])),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank: {Nat.is_le(List.length(&2,U32,List.append(&2,U32,right_leading,[inner,outputs])),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(served_rows(leading,right_leading,rows)),U32.to_nat(inner),U32.to_nat(outputs)},List.append(&2,U32,leading,[rows,outputs]),F.matmul(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp})):
  KernelGemm.Both{+x_global,TiledPublic.Laid{+x_inside,+x_shaped,+x_restored,+x_counted,+x_pinned}} = x_contract
  KernelGemm.Both{+y_global,TiledPublic.Laid{+y_inside,+y_shaped,+y_restored,+y_counted,+y_pinned}} = y_contract
  +default_capacity = {Equal.cong(Machine.Limits,U32,given => Machine.capacity(given),limits,Default.limits(),limits_exact) :
    {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}}
  +workers = U32.max(xw,yw)
  +guard = guarded(images_of(leading,right_leading),served_rows(leading,right_leading,rows),inner,outputs,limits,supported)
  +guard_true = Boolean.and_true(U32.is_eq(inner,inner) && (Shapes.same_extents(right_leading,leading) || List.is_empty(&2,U32,right_leading)),TiledProducts.supported(images_of(leading,right_leading),served_rows(leading,right_leading,rows),inner,outputs,limits),
    Boolean.and_true(U32.is_eq(inner,inner),Shapes.same_extents(right_leading,leading) || List.is_empty(&2,U32,right_leading),Arithmetic.equal_reflexive(inner),paired),supported)
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(ys,ya,yl,yp),List.append(&2,U32,right_leading,[inner,outputs]),y_restored) :
    TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(served_rows(leading,right_leading,rows)),U32.to_nat(inner),U32.to_nat(outputs)},List.append(&2,U32,leading,[rows,outputs]),F.matmul_found(F.factors(Tiled.logical_shape(xs,xa,xl,xp)),F.factors(_),workers,Default.limits(),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp}))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(xs,xa,xl,xp),List.append(&2,U32,leading,[rows,inner]),x_restored) :
    TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(served_rows(leading,right_leading,rows)),U32.to_nat(inner),U32.to_nat(outputs)},List.append(&2,U32,leading,[rows,outputs]),F.matmul_found(F.factors(_),F.factors(List.append(&2,U32,right_leading,[inner,outputs])),workers,Default.limits(),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp}))
  %limits_exact : TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(served_rows(leading,right_leading,rows)),U32.to_nat(inner),U32.to_nat(outputs)},List.append(&2,U32,leading,[rows,outputs]),F.matmul_found(F.factors(List.append(&2,U32,leading,[rows,inner])),F.factors(List.append(&2,U32,right_leading,[inner,outputs])),workers,_,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp}))
  %Equal.sym(Maybe<&2,F.Factors>,F.factors(List.append(&2,U32,leading,[rows,inner])),Some{F.Factors{leading,rows,inner}},Linear.factors_appended(leading,rows,inner)) :
    TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(served_rows(leading,right_leading,rows)),U32.to_nat(inner),U32.to_nat(outputs)},List.append(&2,U32,leading,[rows,outputs]),F.matmul_found(_,F.factors(List.append(&2,U32,right_leading,[inner,outputs])),workers,limits,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp}))
  %Equal.sym(Maybe<&2,F.Factors>,F.factors(List.append(&2,U32,right_leading,[inner,outputs])),Some{F.Factors{right_leading,inner,outputs}},Linear.factors_appended(right_leading,inner,outputs)) :
    TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(served_rows(leading,right_leading,rows)),U32.to_nat(inner),U32.to_nat(outputs)},List.append(&2,U32,leading,[rows,outputs]),F.matmul_found(Some{F.Factors{leading,rows,inner}},_,workers,limits,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp}))
  %Equal.sym(Bool,U32.is_eq(inner,inner) && (Shapes.same_extents(right_leading,leading) || List.is_empty(&2,U32,right_leading)) && TiledProducts.supported(images_of(leading,right_leading),served_rows(leading,right_leading,rows),inner,outputs,limits),True{},guard_true) :
    TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(served_rows(leading,right_leading,rows)),U32.to_nat(inner),U32.to_nat(outputs)},List.append(&2,U32,leading,[rows,outputs]),F.matmul_inner(_,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},images_of(leading,right_leading),served_rows(leading,right_leading,rows),leading,rows,inner,outputs,workers))
  products_reshaped(~Left,~left_value,~Right,~right_value,left,right,leading,right_leading,rows,inner,outputs,images_of(leading,right_leading),served_rows(leading,right_leading,rows),limits,default_capacity,
    xm,xs,xf,xw,xa,xl,xp,x_contract,x_counted,ym,ys,yf,yw,ya,yl,yp,y_contract,y_counted,counts(limits,default_capacity,leading,right_leading,rows,inner,outputs,x_counted,paired,guard),guard,rank_room,right_rank,workers)

# The public F.matmul: left [...,rows,inner] and right [...,inner,outputs] of
# the contract, with the same axes in front or none on the right, that the
# runtime guard supports hold at row r and column j the products of row r of
# left and column j of the matrix of right that serves r, added in increasing
# inner index from +0.0.
def matmul_contract(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,+left: Left,+right: Right,+leading: List<&2,U32>,+right_leading: List<&2,U32>,
  +rows: U32,+inner: U32,+outputs: U32,+limits: Machine.Limits,+limits_exact: {limits == Default.limits() : Machine.Limits},x: Tiled.Tiled,y: Tiled.Tiled,
  +x_contract: TiledPublic.Contract(~Left,~left_value,left,List.append(&2,U32,leading,[rows,inner]),x),+y_contract: TiledPublic.Contract(~Right,~right_value,right,List.append(&2,U32,right_leading,[inner,outputs]),y),
  +paired: {(Shapes.same_extents(right_leading,leading) || List.is_empty(&2,U32,right_leading)) == True{} : Bool},
  +supported: {TiledProducts.supported(images_of(leading,right_leading),served_rows(leading,right_leading,rows),inner,outputs,limits) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,List.append(&2,U32,leading,[rows,inner])),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank: {Nat.is_le(List.length(&2,U32,List.append(&2,U32,right_leading,[inner,outputs])),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(served_rows(leading,right_leading,rows)),U32.to_nat(inner),U32.to_nat(outputs)},List.append(&2,U32,leading,[rows,outputs]),F.matmul(x,y)):
  TiledPublic.by_packed(~Left,~left_value,(tiled => TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(served_rows(leading,right_leading,rows)),U32.to_nat(inner),U32.to_nat(outputs)},List.append(&2,U32,leading,[rows,outputs]),F.matmul(tiled,y))),left,List.append(&2,U32,leading,[rows,inner]),x,x_contract,
    xm => xs => xf => xw => xa => xl => xp => x_packed => TiledPublic.by_packed(~Right,~right_value,(tiled => TiledPublic.Contract(~Products.Batched<Left,Right>,~Products.batched_value(~Left,~left_value,~Right,~right_value),Products.Batched{left,right,U32.to_nat(served_rows(leading,right_leading,rows)),U32.to_nat(inner),U32.to_nat(outputs)},List.append(&2,U32,leading,[rows,outputs]),F.matmul(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},tiled))),right,List.append(&2,U32,right_leading,[inner,outputs]),y,y_contract,
      ym => ys => yf => yw => ya => yl => yp => y_packed =>
        packed_matmul(~Left,~left_value,~Right,~right_value,left,right,leading,right_leading,rows,inner,outputs,limits,limits_exact,
          xm,xs,xf,xw,xa,xl,xp,x_packed,ym,ys,yf,yw,ya,yl,yp,y_packed,paired,supported,rank_room,right_rank)))
