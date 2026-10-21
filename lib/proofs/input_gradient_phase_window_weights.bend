# The phase supplies allocation and query bounds to the shared mapped reorder.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_weights.bend as Weights
import ../machine_limits.bend as Machine
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./input_gradient_phase_packing.bend as Phase
import ./input_gradient_phase_separation.bend as Separation
import ./kernel_weights_refinement.bend as Mapped
import ./tensor_allocation_depth.bend as Depth
import ./machine_limit_bounds.bend as Limits
import ./nat_ceiling.bend as Ceiling
import ./nat_order.bend as Order
import ./nat_division_order.bend as Division
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./row_major_index.bend as RowMajor
import ./u32_division.bend as NativeDivision
import ../kernel_tile.bend as Tile
import ./storage_tree_model.bend as Tree
import ./kernel_weights_readback.bend as Samples
import ./encoded_coordinates.bend as Encoded
import ./word_multiplication.bend as Multiplication

def output_step_product(+output: U32,+channels: U32,+kernel: U32) -> {(output * (channels * kernel * kernel : U32) : U32) == (output * channels * kernel * kernel : U32) : U32}:
  +o = U32.to_nat(output)
  +c = U32.to_nat(channels)
  +k = U32.to_nat(kernel)
  +step = Nat.mul(Nat.mul(c,k),k)
  +old = Nat.mul(Nat.mul(Nat.mul(o,c),k),k)
  +fresh = Nat.mul(o,step)
  +associated = Equal.trans(Nat,old,Nat.mul(Nat.mul(o,Nat.mul(c,k)),k),fresh,
    Equal.cong(Nat,Nat,n => Nat.mul(n,k),Nat.mul(Nat.mul(o,c),k),Nat.mul(o,Nat.mul(c,k)),RowMajor.assoc(o,c,k)),RowMajor.assoc(o,Nat.mul(c,k),k))
  %Arithmetic.native_roundtrip(output) : {(_ * (channels * kernel * kernel : U32) : U32) == (output * channels * kernel * kernel : U32) : U32}
  %Equal.sym(U32,(channels * kernel * kernel : U32),U32.from_nat(step),Encoded.product_three_encoded(channels,kernel,kernel)) : {(U32.from_nat(o) * _ : U32) == (output * channels * kernel * kernel : U32) : U32}
  Equal.trans(U32,(U32.from_nat(o) * U32.from_nat(step) : U32),U32.from_nat(fresh),(output * channels * kernel * kernel : U32),Multiplication.native_encoded_multiply(o,step),
    Equal.trans(U32,U32.from_nat(fresh),U32.from_nat(old),(output * channels * kernel * kernel : U32),Equal.cong(Nat,U32,U32.from_nat,fresh,old,Equal.sym(Nat,old,fresh,associated)),
      Equal.sym(U32,(output * channels * kernel * kernel : U32),U32.from_nat(old),Encoded.product_four_encoded(output,channels,kernel,kernel))))

def offset_matches(+context: Kernel.Phase,+index: U32) -> {Kernel.weight_window_offset(Kernel.weight_window(context),index) == Kernel.weight_offset(context,index) : U32}:
  {==}

def slim_sample(+input: Tree.Tree<F32>,+rows: U32,+row_stride: U32,+row: U32,+context: Kernel.Phase,+index: U32) ->
  {Samples.sample_linear(input,rows,row_stride,row,Kernel.weight_window_offset(Kernel.weight_window(context),index)) ==
    Samples.sample_linear(input,rows,row_stride,row,Kernel.weight_offset(context,index)) : Tile.Tile8}:
  Equal.cong(U32,Tile.Tile8,key => Samples.sample_linear(input,rows,row_stride,row,key),Kernel.weight_window_offset(Kernel.weight_window(context),index),Kernel.weight_offset(context,index),offset_matches(context,index))

def layout(+geometry: Kernel.Geometry,+y: U32,+x: U32) -> Weights.MappedLayout<Kernel.Phase>:
  Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw} = geometry
  +context = Kernel.phase(geometry,y,x)
  Weights.MappedLayout{c,Kernel.phase_inner(context),(k * k : U32),context}

def block_count_exact(+call: Domain.Call,+y: U32,+x: U32) ->
  {Mapped.mapped_blocks(~Kernel.Phase,layout(Domain.geometry(call),y,x)) == Ceiling.ceiling(Domain.channels(Domain.geometry(call)),8n) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Equal.trans(Nat,U32.to_nat(Separation.block_count(geometry)),Nat.div(Nat.add(Domain.channels(geometry),7n),8n),Ceiling.ceiling(Domain.channels(geometry),8n),
    Separation.block_count_exact(call),Ceiling.shifted_formula(Domain.channels(geometry),8n,{==}))

def blocks_room(+call: Domain.Call,+y: U32,+x: U32) ->
  {Nat.is_le(Mapped.mapped_blocks(~Kernel.Phase,layout(Domain.geometry(call),y,x)),Domain.channels(Domain.geometry(call))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  %Equal.sym(Nat,Mapped.mapped_blocks(~Kernel.Phase,layout(geometry,y,x)),Ceiling.ceiling(Domain.channels(geometry),8n),block_count_exact(call,y,x)) : {Nat.is_le(_,Domain.channels(geometry)) == True{} : Bool}
  Ceiling.bounded(Domain.channels(geometry),8n,{==})

def packed_room(+call: Domain.Call,+y: U32,+x: U32) ->
  {Nat.is_le(Nat.mul(Mapped.mapped_blocks(~Kernel.Phase,layout(Domain.geometry(call),y,x)),U32.to_nat(Mapped.mapped_inner(~Kernel.Phase,layout(Domain.geometry(call),y,x)))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +context = Kernel.phase(geometry,y,x)
  +blocks = Mapped.mapped_blocks(~Kernel.Phase,layout(geometry,y,x))
  +area = Phase.phase_area(context)
  +square = Nat.mul(U32.to_nat(k),U32.to_nat(k))
  +full = Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(c),square))
  %Equal.sym(Nat,U32.to_nat(Kernel.phase_inner(context)),Nat.mul(U32.to_nat(o),area),PhaseBounds.phase_inner_exact(call,y,x)) : {Nat.is_le(Nat.mul(blocks,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Order.transitive(Nat.mul(blocks,Nat.mul(U32.to_nat(o),area)),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(o),area)),U32.to_nat(Machine.capacity(limits)),
    RowMajor.mul_le(blocks,U32.to_nat(c),Nat.mul(U32.to_nat(o),area),blocks_room(call,y,x)),
    Order.transitive(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(o),area)),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(o),square)),U32.to_nat(Machine.capacity(limits)),
      Division.multiply_left_le(U32.to_nat(c),Nat.mul(U32.to_nat(o),area),Nat.mul(U32.to_nat(o),square),Division.multiply_left_le(U32.to_nat(o),area,square,PhaseBounds.phase_area_room(call,y,x))),
      %Equal.sym(Nat,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(o),square)),full,
        Equal.trans(Nat,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(o),square)),Nat.mul(Nat.mul(U32.to_nat(c),U32.to_nat(o)),square),full,
          Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(c),U32.to_nat(o)),square),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(o),square)),RowMajor.assoc(U32.to_nat(c),U32.to_nat(o),square)),
          Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat(c),U32.to_nat(o)),square),Nat.mul(Nat.mul(U32.to_nat(o),U32.to_nat(c)),square),full,
            Equal.cong(Nat,Nat,value => Nat.mul(value,square),Nat.mul(U32.to_nat(c),U32.to_nat(o)),Nat.mul(U32.to_nat(o),U32.to_nat(c)),RowMajor.multiply_commutative(U32.to_nat(c),U32.to_nat(o))),RowMajor.assoc(U32.to_nat(o),U32.to_nat(c),square)))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      Domain.weight_room(call)))

def packed_exact(+call: Domain.Call,+y: U32,+x: U32) ->
  {U32.to_nat((((Domain.channel_word(Domain.geometry(call)) + 7) / 8) * Kernel.phase_inner(Kernel.phase(Domain.geometry(call),y,x)) : U32)) ==
    Nat.mul(Mapped.mapped_blocks(~Kernel.Phase,layout(Domain.geometry(call),y,x)),U32.to_nat(Mapped.mapped_inner(~Kernel.Phase,layout(Domain.geometry(call),y,x)))) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Arithmetic.multiply(((c + 7) / 8 : U32),Kernel.phase_inner(Kernel.phase(geometry,y,x)),Machine.capacity(limits),Limits.capacity_fits(limits),packed_room(call,y,x))

def allocation(+call: Domain.Call,+y: U32,+x: U32) ->
  Depth.Sized<(((Domain.channel_word(Domain.geometry(call)) + 7) / 8) * Kernel.phase_inner(Kernel.phase(Domain.geometry(call),y,x)) : U32)>:
  Domain.Call{geometry,batch,limits,facts} = call
  +count = (((Domain.channel_word(geometry) + 7) / 8) * Kernel.phase_inner(Kernel.phase(geometry,y,x)) : U32)
  Depth.bounded_with(count,Machine.capacity(limits),Limits.capacity_fits(limits),Limits.capacity_allocatable(limits),
    NativeDivision.nat_le_native(1,Machine.capacity(limits),Order.transitive(1n,7n,U32.to_nat(Machine.capacity(limits)),{==},Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits)))),
    NativeDivision.nat_le_native(count,Machine.capacity(limits),
      %Equal.sym(Nat,U32.to_nat(count),Nat.mul(Mapped.mapped_blocks(~Kernel.Phase,layout(geometry,y,x)),U32.to_nat(Mapped.mapped_inner(~Kernel.Phase,layout(geometry,y,x)))),packed_exact(call,y,x)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      packed_room(call,y,x)))
