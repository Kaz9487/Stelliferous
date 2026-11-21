# The weight and bias gradient contract read through value readers, and the
# terms of the kernel models over lifted cells as the terms of the contract.

import Base
import ../parameter_gradient_specification.bend as Spec
import ../../lib/tensor_parameter_gradient.bend as Parameter
import ../../lib/kernel_parameter_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_tiled_cells.bend as Cells
import ../../lib/proofs/kernel_bias_gradient.bend as Bias
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/u32_full_range.bend as FullRange
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_order.bend as Order
import ../specification.bend as ConvSpec
import ../../lib/proofs/grouped_reduction_refinement.bend as Reduction
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/traversal_grouped.bend as Grouped
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/kernel_weight_gradient.bend as WeightKernel
import ../../lib/proofs/kernel_weight_gradient_held.bend as Sources
import ../../lib/proofs/kernel_weight_gradient_group.bend as Groups
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/boolean_logic.bend as Logic
import ../../lib/proofs/kernel_weight_gradient_packing.bend as Packing
import ../../lib/proofs/kernel_weight_gradient_images.bend as Images
import ../../lib/proofs/traversal_array_refinement.bend as Traversals

# ---- Readers as the contract's source ----

# The shape of the contract of a work.
def shape(work: Kernel.Work) -> Spec.Shape:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Spec.Shape{n,c,o,h,w,k,s,p,rows,columns,g}

# The input and the output gradient, each with its own reader.
type Operand<-Input: Data,-Gradient: Data> is Data:
  InputOperand{context: Input}
  GradientOperand{context: Gradient}

def operand_at(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,operand: Operand<Input,Gradient>,index: Nat) -> F32:
  match operand:
    case InputOperand{context}: input_value(context,U32.from_nat(index))
    case GradientOperand{context}: gradient_value(context,U32.from_nat(index))

# The output gradient alone, for the bias.
def gradient_at(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,context: Gradient,index: Nat) -> F32:
  gradient_value(context,U32.from_nat(index))

# The weight gradient of the readers' values at an index of the weights.
type WeightRead<-Input: Data,-Gradient: Data> is Data:
  WeightRead{input: Input,gradient: Gradient,input_image: Maybe<&2,Parameter.ImageGeometry>,gradient_image: Maybe<&2,Parameter.ImageGeometry>,
    kernel: U32,stride: U32,padding: U32,group: U32}

def weight_imaged(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,
  input_image: Maybe<&2,Parameter.ImageGeometry>,gradient_image: Maybe<&2,Parameter.ImageGeometry>,+kernel: U32,+stride: U32,+padding: U32,+group: U32,
  input: Input,gradient: Gradient,index: U32) -> F32:
  match input_image gradient_image:
    case Some{x} Some{d}:
      Spec.weight(~Operand<Input,Gradient>,~operand_at(~Input,~input_value,~Gradient,~gradient_value),
        Spec.Problem{shape(Parameter.work(x,d,kernel,stride,padding,group)),InputOperand{input},GradientOperand{gradient}},U32.to_nat(index))
    case x d: 0.0

def weight_at(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,context: WeightRead<Input,Gradient>,index: U32) -> F32:
  WeightRead{input,gradient,input_image,gradient_image,+kernel,+stride,+padding,+group} = context
  weight_imaged(~Input,~input_value,~Gradient,~gradient_value,input_image,gradient_image,kernel,stride,padding,group,input,gradient,index)

# The bias gradient of the gradient reader's values at an output.
type BiasRead<-Gradient: Data> is Data:
  BiasRead{gradient: Gradient,image: Maybe<&2,Parameter.ImageGeometry>,group: U32}

def bias_imaged(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,image: Maybe<&2,Parameter.ImageGeometry>,+group: U32,+gradient: Gradient,index: U32) -> F32:
  match image:
    case None{}: 0.0
    case Some{d}:
      Spec.bias(~Gradient,~gradient_at(~Gradient,~gradient_value),Spec.Problem{shape(Parameter.bias_work(d,group)),gradient,gradient},U32.to_nat(index))

def bias_at(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,context: BiasRead<Gradient>,index: U32) -> F32:
  BiasRead{+gradient,image,+group} = context
  bias_imaged(~Gradient,~gradient_value,image,group,gradient,index)

# ---- The bias terms over lifted cells ----

# The logical index of an output at a position of an image is the index the
# contract reads.
def index_exact(+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,+image: Nat,+position: Nat,+element: Nat,+outputs: Nat,+plane: Nat,
  +after_exact: {U32.to_nat(after) == plane : Nat},+lanes_exact: {U32.to_nat(lanes) == outputs : Nat},+position_inside: {Nat.is_lt(position,plane) == True{} : Bool}) ->
  {Representation.logical_index(Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}},Nat.add(Nat.mul(image,plane),position),element) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(image,outputs),element),plane),position) : Nat}:
  +outer = Nat.add(Nat.mul(image,plane),position)
  +least = Representation.least_one(plane)
  +target = Nat.add(Nat.mul(Nat.add(Nat.mul(image,outputs),element),plane),position)
  %Equal.sym(Nat,U32.to_nat(after),plane,after_exact) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(outer,Representation.least_one(_)),U32.to_nat(lanes)),element),Representation.least_one(_)),Nat.mod(outer,Representation.least_one(_))) ==
      target : Nat}
  %Equal.sym(Nat,U32.to_nat(lanes),outputs,lanes_exact) : {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(outer,least),_),element),least),Nat.mod(outer,least)) == target : Nat}
  %Equal.sym(Nat & Nat,Nat.divmod(outer,least),(image,position),Representation.split_at(image,position,plane,position_inside)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),outputs),element),least),Pair.snd(Nat,Nat,_)) == target : Nat}
  %Equal.sym(Nat,least,plane,Representation.least_positive(plane,Representation.below_positive(position,plane,position_inside))) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(image,outputs),element),_),position) == target : Nat}
  {==}

# A sample of the bias model over the lifted output gradient, at a lane of a
# block of outputs inside the outputs, is the term the contract reads.
def bias_sample(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +n: Nat,+c: Nat,+o: Nat,+h: Nat,+w: Nat,+k: Nat,+s: Nat,+p: Nat,+rows: Nat,+columns: Nat,+g: Nat,+image: Nat,+block: Nat,+lane: Nat,+position: Nat,
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+below: {Nat.is_lt(lane,8n) == True{} : Bool},+capacity: U32,
  +room: {Nat.is_le(Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(rows,columns)),position),Kernel.groups(o)),block),U32.to_nat(capacity)) == True{} : Bool},
  +after_exact: {U32.to_nat(after) == Nat.mul(rows,columns) : Nat},+lanes_exact: {U32.to_nat(lanes) == o : Nat},+groups_exact: {U32.to_nat(groups) == Kernel.groups(o) : Nat},
  +outer_inside: {Nat.is_lt(Nat.add(Nat.mul(image,Nat.mul(rows,columns)),position),U32.to_nat(outer_word)) == True{} : Bool},
  +block_inside: {Nat.is_lt(block,U32.to_nat(groups)) == True{} : Bool},
  +element_inside: {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),o) == True{} : Bool},+position_inside: {Nat.is_lt(position,Nat.mul(rows,columns)) == True{} : Bool}) ->
  {Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}},
      Kernel.Context{Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g},image,Nat.add(Nat.mul(block,8n),lane)},position) ==
    Spec.bias_term(~DL,~gradient_at(~DL,~d_value),Spec.Problem{shape(Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g}),logical,logical},image,Nat.add(Nat.mul(block,8n),lane),position) : F32}:
  +element = Nat.add(Nat.mul(block,8n),lane)
  +plane = Nat.mul(rows,columns)
  +outer = Nat.add(Nat.mul(image,plane),position)
  +geometry = {Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}} : Cells.Cells}
  +holder = {Representation.Lifted{logical,geometry} : Representation.Lifted<DL>}
  +number = Nat.add(Nat.mul(outer,Kernel.groups(o)),block)
  +spec = d_value(logical,U32.from_nat(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),element),rows),columns),position)))
  +exact = Equal.trans(Nat,U32.to_nat(U32.from_nat(number)),number,Nat.add(Nat.mul(outer,U32.to_nat(groups)),block),FullRange.roundtrip(number,capacity,room),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(outer,value),block),Kernel.groups(o),U32.to_nat(groups),Equal.sym(Nat,U32.to_nat(groups),Kernel.groups(o),groups_exact)))
  +lanes_inside = {%Equal.sym(Nat,U32.to_nat(lanes),o,lanes_exact) : {Nat.is_lt(element,_) == True{} : Bool} element_inside :
    {Nat.is_lt(element,U32.to_nat(lanes)) == True{} : Bool}}
  %Equal.sym(Nat,Nat.div(element,8n),block,Bias.block_of(block,lane,below)) :
    {TileLaws.select_lane(Nat.mod(element,8n),Representation.lifted(~DL,~d_value,holder,U32.from_nat(Nat.add(Nat.mul(outer,Kernel.groups(o)),_)))) == spec : F32}
  %Equal.sym(Nat,Nat.mod(element,8n),lane,Bias.lane_of(block,lane,below)) :
    {TileLaws.select_lane(_,Representation.lifted(~DL,~d_value,holder,U32.from_nat(number))) == spec : F32}
  %Equal.sym(Tile.Tile8,Representation.lifted(~DL,~d_value,holder,U32.from_nat(number)),Representation.cell(~DL,~d_value,logical,geometry,outer,block),
      Representation.decoded(~DL,~d_value,logical,geometry,outer,block,U32.from_nat(number),exact,outer_inside,block_inside)) :
    {TileLaws.select_lane(lane,_) == spec : F32}
  %Equal.sym(F32,TileLaws.select_lane(lane,Representation.cell(~DL,~d_value,logical,geometry,outer,block)),Representation.lane(~DL,~d_value,logical,geometry,outer,block,lane),
      Representation.cell_lane(~DL,~d_value,logical,geometry,outer,block,lane,small)) :
    {_ == spec : F32}
  %Order.commutative(Nat.mul(block,8n),lane) :
    {Bool.pick(F32,Nat.is_lt(_,U32.to_nat(lanes)),d_value(logical,U32.from_nat(Representation.logical_index(geometry,outer,_))),0.0) == spec : F32}
  %Equal.sym(Bool,Nat.is_lt(element,U32.to_nat(lanes)),True{},lanes_inside) :
    {Bool.pick(F32,_,d_value(logical,U32.from_nat(Representation.logical_index(geometry,outer,element))),0.0) == spec : F32}
  %Equal.sym(Nat,Representation.logical_index(geometry,outer,element),Nat.add(Nat.mul(Nat.add(Nat.mul(image,o),element),plane),position),
      index_exact(outer_word,after,lanes,groups,image,position,element,o,plane,after_exact,lanes_exact,position_inside)) :
    {d_value(logical,U32.from_nat(_)) == spec : F32}
  %RowMajor.assoc(Nat.add(Nat.mul(image,o),element),rows,columns) : {d_value(logical,U32.from_nat(Nat.add(_,position))) == spec : F32}
  {==}

# The samples of the bias model over a range of positions fold to the sum of
# the terms the contract reads there.
def bias_fold(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +n: Nat,+c: Nat,+o: Nat,+h: Nat,+w: Nat,+k: Nat,+s: Nat,+p: Nat,+rows: Nat,+columns: Nat,+g: Nat,+image: Nat,+block: Nat,+lane: Nat,
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+below: {Nat.is_lt(lane,8n) == True{} : Bool},+capacity: U32,
  +after_exact: {U32.to_nat(after) == Nat.mul(rows,columns) : Nat},+lanes_exact: {U32.to_nat(lanes) == o : Nat},+groups_exact: {U32.to_nat(groups) == Kernel.groups(o) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(n,Nat.mul(rows,columns)) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(rows,columns)),Kernel.groups(o)),U32.to_nat(capacity)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,n) == True{} : Bool},+block_inside: {Nat.is_lt(block,U32.to_nat(groups)) == True{} : Bool},
  +element_inside: {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),o) == True{} : Bool},
  remaining: Nat,+position: Nat,+end_inside: {Nat.is_le(Nat.add(position,remaining),Nat.mul(rows,columns)) == True{} : Bool},total: F32) ->
  {Reduction.values(~Representation.Lifted<DL>,~Kernel.Context,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),remaining,position,
      Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}},
      Kernel.Context{Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g},image,Nat.add(Nat.mul(block,8n),lane)},total) ==
    ConvSpec.sum(Spec.terms(~DL,~Spec.bias_term(~DL,~gradient_at(~DL,~d_value)),remaining,position,
      Spec.Problem{shape(Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g}),logical,logical},image,Nat.add(Nat.mul(block,8n),lane)),total) : F32}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +plane = Nat.mul(rows,columns)
      +element = Nat.add(Nat.mul(block,8n),lane)
      +work = {Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} : Kernel.Work}
      +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
      +context = {Kernel.Context{work,image,element} : Kernel.Context}
      +problem = {Spec.Problem{shape(work),logical,logical} : Spec.Problem<DL>}
      +outer = Nat.add(Nat.mul(image,plane),position)
      +position_inside = Order.room_index(position,rest,plane,end_inside)
      +outer_below = RowMajor.radix(image,n,position,plane,image_inside,position_inside)
      +outer_inside = {%Equal.sym(Nat,U32.to_nat(outer_word),Nat.mul(n,plane),outer_exact) : {Nat.is_lt(outer,_) == True{} : Bool} outer_below :
        {Nat.is_lt(outer,U32.to_nat(outer_word)) == True{} : Bool}}
      +blocks_inside = {%groups_exact : {Nat.is_lt(block,_) == True{} : Bool} block_inside : {Nat.is_lt(block,Kernel.groups(o)) == True{} : Bool}}
      +number_below = RowMajor.radix(outer,Nat.mul(n,plane),block,Kernel.groups(o),outer_below,blocks_inside)
      +room = Order.transitive(Nat.add(Nat.mul(outer,Kernel.groups(o)),block),Nat.mul(Nat.mul(n,plane),Kernel.groups(o)),U32.to_nat(capacity),
        Bounds.strict_le(Nat.add(Nat.mul(outer,Kernel.groups(o)),block),Nat.mul(Nat.mul(n,plane),Kernel.groups(o)),number_below),total_room)
      +same = bias_sample(~DL,~d_value,logical,outer_word,after,lanes,groups,n,c,o,h,w,k,s,p,rows,columns,g,image,block,lane,position,small,below,capacity,room,
        after_exact,lanes_exact,groups_exact,outer_inside,block_inside,element_inside,position_inside)
      +term = Spec.bias_term(~DL,~gradient_at(~DL,~d_value),problem,image,element,position)
      %Equal.sym(F32,Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,context,position),term,same) :
        {Reduction.values(~Representation.Lifted<DL>,~Kernel.Context,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),rest,1n+position,holder,context,
            (total + _ : F32)) ==
          ConvSpec.sum(Spec.terms(~DL,~Spec.bias_term(~DL,~gradient_at(~DL,~d_value)),rest,1n+position,problem,image,element),(total + term : F32)) : F32}
      bias_fold(~DL,~d_value,logical,outer_word,after,lanes,groups,n,c,o,h,w,k,s,p,rows,columns,g,image,block,lane,small,below,capacity,after_exact,lanes_exact,groups_exact,
        outer_exact,total_room,image_inside,block_inside,element_inside,rest,1n+position,Order.advance_room(position,rest,plane,end_inside),(total + term : F32))

# The group sums of the bias model over bounded ranges are the group sums the
# contract reads.
def bias_groups(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +n: Nat,+c: Nat,+o: Nat,+h: Nat,+w: Nat,+k: Nat,+s: Nat,+p: Nat,+rows: Nat,+columns: Nat,+g: Nat,+image: Nat,+block: Nat,+lane: Nat,
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+below: {Nat.is_lt(lane,8n) == True{} : Bool},+capacity: U32,
  +after_exact: {U32.to_nat(after) == Nat.mul(rows,columns) : Nat},+lanes_exact: {U32.to_nat(lanes) == o : Nat},+groups_exact: {U32.to_nat(groups) == Kernel.groups(o) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(n,Nat.mul(rows,columns)) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(rows,columns)),Kernel.groups(o)),U32.to_nat(capacity)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,n) == True{} : Bool},+block_inside: {Nat.is_lt(block,U32.to_nat(groups)) == True{} : Bool},
  +element_inside: {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),o) == True{} : Bool},
  ranges: List<&2,Grouped.Range>,+bounded: Members.All(Grouped.Range,{range => Reduction.RangeBounded(rows,columns,g,range) : Grouped.Range -> Data},ranges)) ->
  {Reduction.group_values(~Representation.Lifted<DL>,~Kernel.Context,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),ranges,
      Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}},
      Kernel.Context{Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g},image,Nat.add(Nat.mul(block,8n),lane)}) ==
    Spec.group_values(~DL,~Spec.bias_term(~DL,~gradient_at(~DL,~d_value)),ranges,Spec.Problem{shape(Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g}),logical,logical},image,
      Nat.add(Nat.mul(block,8n),lane)) : List<&2,F32>}:
  match ranges:
    case []: {==}
    case Grouped.Range{+first,+count} <> +rest:
      KernelGemm.Both{+range_bound,+others} = bounded
      KernelGemm.Both{positive,+inner} = range_bound
      KernelGemm.Both{fits,+end} = inner
      +element = Nat.add(Nat.mul(block,8n),lane)
      +work = {Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} : Kernel.Work}
      +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
      +context = {Kernel.Context{work,image,element} : Kernel.Context}
      +problem = {Spec.Problem{shape(work),logical,logical} : Spec.Problem<DL>}
      +model_head = Reduction.values(~Representation.Lifted<DL>,~Kernel.Context,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),count,first,holder,
        context,0.0)
      +spec_head = ConvSpec.sum(Spec.terms(~DL,~Spec.bias_term(~DL,~gradient_at(~DL,~d_value)),count,first,problem,image,element),0.0)
      +model_rest = Reduction.group_values(~Representation.Lifted<DL>,~Kernel.Context,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),rest,
        holder,context)
      +spec_rest = Spec.group_values(~DL,~Spec.bias_term(~DL,~gradient_at(~DL,~d_value)),rest,problem,image,element)
      Equal.trans(List<&2,F32>,model_head <> model_rest,spec_head <> model_rest,spec_head <> spec_rest,
        Equal.cong(F32,List<&2,F32>,value => value <> model_rest,model_head,spec_head,
          bias_fold(~DL,~d_value,logical,outer_word,after,lanes,groups,n,c,o,h,w,k,s,p,rows,columns,g,image,block,lane,small,below,capacity,after_exact,lanes_exact,groups_exact,
            outer_exact,total_room,image_inside,block_inside,element_inside,count,first,
            end,
            0.0)),
        Equal.cong(List<&2,F32>,List<&2,F32>,values => spec_head <> values,model_rest,spec_rest,
          bias_groups(~DL,~d_value,logical,outer_word,after,lanes,groups,n,c,o,h,w,k,s,p,rows,columns,g,image,block,lane,small,below,capacity,after_exact,lanes_exact,groups_exact,
            outer_exact,total_room,image_inside,block_inside,element_inside,rest,others)))

# The image sums of the bias model are the image sums the contract reads.
def bias_images(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +n: Nat,+c: Nat,+o: Nat,+h: Nat,+w: Nat,+k: Nat,+s: Nat,+p: Nat,+rows: Nat,+columns: Nat,+g: Nat,+block: Nat,+lane: Nat,
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+below: {Nat.is_lt(lane,8n) == True{} : Bool},+capacity: U32,
  +after_exact: {U32.to_nat(after) == Nat.mul(rows,columns) : Nat},+lanes_exact: {U32.to_nat(lanes) == o : Nat},+groups_exact: {U32.to_nat(groups) == Kernel.groups(o) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(n,Nat.mul(rows,columns)) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(rows,columns)),Kernel.groups(o)),U32.to_nat(capacity)) == True{} : Bool},
  +block_inside: {Nat.is_lt(block,U32.to_nat(groups)) == True{} : Bool},+element_inside: {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),o) == True{} : Bool},
  +ranges: List<&2,Grouped.Range>,+bounded: Members.All(Grouped.Range,{range => Reduction.RangeBounded(rows,columns,g,range) : Grouped.Range -> Data},ranges),
  remaining: Nat,+image: Nat,+end: {Nat.is_le(Nat.add(image,remaining),n) == True{} : Bool}) ->
  {Bias.image_values(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),remaining,image,Nat.add(Nat.mul(block,8n),lane),
      Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g},ranges,Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}}) ==
    Spec.image_values(~DL,~Spec.bias_term(~DL,~gradient_at(~DL,~d_value)),remaining,image,Nat.add(Nat.mul(block,8n),lane),
      Spec.Problem{shape(Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g}),logical,logical},ranges) : List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +element = Nat.add(Nat.mul(block,8n),lane)
      +work = {Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} : Kernel.Work}
      +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
      +problem = {Spec.Problem{shape(work),logical,logical} : Spec.Problem<DL>}
      +model_head = Grouped.tree(Reduction.group_values(~Representation.Lifted<DL>,~Kernel.Context,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),
        ranges,holder,Kernel.Context{work,image,element}))
      +spec_head = Grouped.tree(Spec.group_values(~DL,~Spec.bias_term(~DL,~gradient_at(~DL,~d_value)),ranges,problem,image,element))
      +model_rest = Bias.image_values(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),rest,1n+image,element,work,ranges,
        holder)
      +spec_rest = Spec.image_values(~DL,~Spec.bias_term(~DL,~gradient_at(~DL,~d_value)),rest,1n+image,element,problem,ranges)
      Equal.trans(List<&2,F32>,model_head <> model_rest,spec_head <> model_rest,spec_head <> spec_rest,
        Equal.cong(F32,List<&2,F32>,value => value <> model_rest,model_head,spec_head,
          Equal.cong(List<&2,F32>,F32,values => Grouped.tree(values),
            Reduction.group_values(~Representation.Lifted<DL>,~Kernel.Context,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),
              ranges,holder,Kernel.Context{work,image,element}),
            Spec.group_values(~DL,~Spec.bias_term(~DL,~gradient_at(~DL,~d_value)),ranges,problem,image,element),
            bias_groups(~DL,~d_value,logical,outer_word,after,lanes,groups,n,c,o,h,w,k,s,p,rows,columns,g,image,block,lane,small,below,capacity,after_exact,lanes_exact,
              groups_exact,outer_exact,total_room,Order.room_index(image,rest,n,end),block_inside,element_inside,ranges,bounded))),
        Equal.cong(List<&2,F32>,List<&2,F32>,values => spec_head <> values,model_rest,spec_rest,
          bias_images(~DL,~d_value,logical,outer_word,after,lanes,groups,n,c,o,h,w,k,s,p,rows,columns,g,block,lane,small,below,capacity,after_exact,lanes_exact,groups_exact,
            outer_exact,total_room,block_inside,element_inside,ranges,bounded,rest,1n+image,Order.advance_room(image,rest,n,end))))

# A lane inside the outputs of a block of the bias model over the lifted
# output gradient is the bias the contract reads.
def bias_lane(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +n: Nat,+c: Nat,+o: Nat,+h: Nat,+w: Nat,+k: Nat,+s: Nat,+p: Nat,+rows: Nat,+columns: Nat,+g: Nat,+block: Nat,+lane: Nat,
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+below: {Nat.is_lt(lane,8n) == True{} : Bool},+capacity: U32,
  +after_exact: {U32.to_nat(after) == Nat.mul(rows,columns) : Nat},+lanes_exact: {U32.to_nat(lanes) == o : Nat},+groups_exact: {U32.to_nat(groups) == Kernel.groups(o) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(n,Nat.mul(rows,columns)) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(rows,columns)),Kernel.groups(o)),U32.to_nat(capacity)) == True{} : Bool},
  +block_inside: {Nat.is_lt(block,U32.to_nat(groups)) == True{} : Bool},+element_inside: {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),o) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,columns) == True{} : Bool},+group_positive: {Nat.is_lt(0n,g) == True{} : Bool}) ->
  {Bias.block_lane(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),
      Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}},
      Kernel.Block{Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g},block,Bias.work_ranges(Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g})},lane) ==
    Spec.bias(~DL,~gradient_at(~DL,~d_value),Spec.Problem{shape(Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g}),logical,logical},Nat.add(Nat.mul(block,8n),lane)) : F32}:
  +element = Nat.add(Nat.mul(block,8n),lane)
  +work = {Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} : Kernel.Work}
  +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
  +problem = {Spec.Problem{shape(work),logical,logical} : Spec.Problem<DL>}
  +ranges = Grouped.ranges(rows,columns,g)
  +spec = Spec.bias(~DL,~gradient_at(~DL,~d_value),problem,element)
  %Equal.sym(Bool,Nat.is_lt(element,o),True{},element_inside) :
    {Bias.lane_value(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),_,work,element,ranges,holder) == spec : F32}
  Equal.cong(List<&2,F32>,F32,values => Grouped.tree(values),
    Bias.image_values(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),n,0n,element,work,ranges,holder),
    Spec.image_values(~DL,~Spec.bias_term(~DL,~gradient_at(~DL,~d_value)),n,0n,element,problem,ranges),
    bias_images(~DL,~d_value,logical,outer_word,after,lanes,groups,n,c,o,h,w,k,s,p,rows,columns,g,block,lane,small,below,capacity,after_exact,lanes_exact,groups_exact,
      outer_exact,total_room,block_inside,element_inside,ranges,Reduction.ranges_bounded(rows,columns,g,columns_positive,group_positive),n,0n,Order.reflexive(n)))

# ---- The weight terms over lifted cells ----

# A lane of a lifted [n, o, rows, columns] layout, at an element inside the
# lanes and a position inside the plane of an image, is the value the
# contract reads there.
def lifted_lane(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +n: Nat,+o: Nat,+rows: Nat,+columns: Nat,+image: Nat,+element: Nat,+position: Nat,+capacity: U32,
  +after_exact: {U32.to_nat(after) == Nat.mul(rows,columns) : Nat},+lanes_exact: {U32.to_nat(lanes) == o : Nat},+groups_exact: {U32.to_nat(groups) == Kernel.groups(o) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(n,Nat.mul(rows,columns)) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(rows,columns)),Kernel.groups(o)),U32.to_nat(capacity)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,n) == True{} : Bool},+element_inside: {Nat.is_lt(element,o) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,Nat.mul(rows,columns)) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(element,8n),Representation.lifted(~DL,~d_value,Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}},
      U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(rows,columns)),position),Kernel.groups(o)),Nat.div(element,8n))))) ==
    d_value(logical,U32.from_nat(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),element),rows),columns),position))) : F32}:
  +plane = Nat.mul(rows,columns)
  +block = Nat.div(element,8n)
  +lane = Nat.mod(element,8n)
  +outer = Nat.add(Nat.mul(image,plane),position)
  +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
  +below = DivisionOrder.remainder_small(element,8n,{==})
  +small = Division.lt_succ_le(lane,7n,below)
  +split = DivisionOrder.quotient_exact(element,8n,{==})
  +inside = {%split : {Nat.is_lt(_,o) == True{} : Bool} element_inside : {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),o) == True{} : Bool}}
  +blocks_inside = DivisionOrder.quotient_below(element,Ceiling.ceiling(o,8n),8n,{==},Bounds.nat_trans(element,o,Nat.mul(Ceiling.ceiling(o,8n),8n),element_inside,Ceiling.covers(o,8n,{==})))
  +groups_inside = {%Equal.sym(Nat,Kernel.groups(o),Ceiling.ceiling(o,8n),Ceiling.shifted_formula(o,8n,{==})) : {Nat.is_lt(block,_) == True{} : Bool} blocks_inside :
    {Nat.is_lt(block,Kernel.groups(o)) == True{} : Bool}}
  +block_inside = {%Equal.sym(Nat,U32.to_nat(groups),Kernel.groups(o),groups_exact) : {Nat.is_lt(block,_) == True{} : Bool} groups_inside :
    {Nat.is_lt(block,U32.to_nat(groups)) == True{} : Bool}}
  +outer_below = RowMajor.radix(image,n,position,plane,image_inside,position_inside)
  +outer_inside = {%Equal.sym(Nat,U32.to_nat(outer_word),Nat.mul(n,plane),outer_exact) : {Nat.is_lt(outer,_) == True{} : Bool} outer_below :
    {Nat.is_lt(outer,U32.to_nat(outer_word)) == True{} : Bool}}
  +number_below = RowMajor.radix(outer,Nat.mul(n,plane),block,Kernel.groups(o),outer_below,groups_inside)
  +room = Order.transitive(Nat.add(Nat.mul(outer,Kernel.groups(o)),block),Nat.mul(Nat.mul(n,plane),Kernel.groups(o)),U32.to_nat(capacity),
    Bounds.strict_le(Nat.add(Nat.mul(outer,Kernel.groups(o)),block),Nat.mul(Nat.mul(n,plane),Kernel.groups(o)),number_below),total_room)
  %Equal.sym(Nat,element,Nat.add(Nat.mul(block,8n),lane),split) :
    {TileLaws.select_lane(Nat.mod(_,8n),Representation.lifted(~DL,~d_value,holder,U32.from_nat(Nat.add(Nat.mul(outer,Kernel.groups(o)),Nat.div(_,8n))))) ==
      d_value(logical,U32.from_nat(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),_),rows),columns),position))) : F32}
  bias_sample(~DL,~d_value,logical,outer_word,after,lanes,groups,n,0n,o,0n,0n,0n,0n,0n,rows,columns,0n,image,block,lane,position,small,below,capacity,room,after_exact,
    lanes_exact,groups_exact,outer_inside,block_inside,inside,position_inside)

# The cell the weight model reads at a position of an image is the cell of
# that position in the lifted layout.
def gradient_address(+image: Nat,+position: Nat,+blocks: Nat,+rows: Nat,+columns: Nat,+block: Nat) ->
  {Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,rows),columns)),Nat.mul(position,blocks)),block) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(rows,columns)),position),blocks),block) : Nat}:
  +plane = Nat.mul(rows,columns)
  +model = Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,rows),columns)),Nat.mul(position,blocks)),block)
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,plane),position),blocks),Nat.add(Nat.mul(Nat.mul(image,plane),blocks),Nat.mul(position,blocks)),
      RowMajor.distrib(Nat.mul(image,plane),position,blocks)) :
    {model == Nat.add(_,block) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(image,plane),blocks),Nat.mul(image,Nat.mul(plane,blocks)),RowMajor.assoc(image,plane,blocks)) :
    {model == Nat.add(Nat.add(_,Nat.mul(position,blocks)),block) : Nat}
  %Equal.trans(Nat,Nat.mul(Nat.mul(blocks,rows),columns),Nat.mul(blocks,plane),Nat.mul(plane,blocks),RowMajor.assoc(blocks,rows,columns),
      RowMajor.multiply_commutative(blocks,plane)) :
    {model == Nat.add(Nat.add(Nat.mul(image,_),Nat.mul(position,blocks)),block) : Nat}
  {==}

def input_address(+image: Nat,+pixel: Nat,+blocks: Nat,+height: Nat,+width: Nat,+block: Nat) ->
  {Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,height),width)),block),Nat.mul(pixel,blocks)) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(height,width)),pixel),blocks),block) : Nat}:
  +base = Nat.mul(image,Nat.mul(Nat.mul(blocks,height),width))
  +offset = Nat.mul(pixel,blocks)
  +model = Nat.add(Nat.add(base,block),offset)
  %gradient_address(image,pixel,blocks,height,width,block) : {model == _ : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(base,offset),block),Nat.add(base,Nat.add(offset,block)),Addition.associative(base,offset,block)) : {model == _ : Nat}
  %Order.commutative(block,offset) : {model == Nat.add(base,_) : Nat}
  Addition.associative(base,block,offset)

# The output gradient the weight model reads at an output and a position of
# an image of the lifted gradient is the value the contract reads there.
def gradient_sample(~X: Data,~xv: X -> U32 -> Tile.Tile8,~DL: Data,~d_value: DL -> U32 -> F32,+input: X,+logical: DL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +c: U32,+o: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+rows: U32,+columns: U32,+taps: U32,+x_first: Nat,+n: Nat,+image: Nat,+output: U32,+position: Nat,+capacity: U32,
  +after_exact: {U32.to_nat(after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},+lanes_exact: {U32.to_nat(lanes) == U32.to_nat(o) : Nat},
  +groups_exact: {U32.to_nat(groups) == Kernel.groups(U32.to_nat(o)) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Kernel.groups(U32.to_nat(o))),U32.to_nat(capacity)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,n) == True{} : Bool},+output_inside: {Nat.is_lt(U32.to_nat(output),U32.to_nat(o)) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}) ->
  {Sources.held_input(~X,~xv,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
      Sources.HeldSource{input,Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}},WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},x_first,
        Nat.mul(image,Groups.gradient_extent(WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps}))},output,position) ==
    d_value(logical,U32.from_nat(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(o)),U32.to_nat(output)),U32.to_nat(rows)),U32.to_nat(columns)),position))) : F32}:
  +on = U32.to_nat(output)
  +blocks = Ceiling.ceiling(U32.to_nat(o),8n)
  +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
  +spec = d_value(logical,U32.from_nat(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(o)),on),U32.to_nat(rows)),U32.to_nat(columns)),position)))
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,U32.to_nat(rows)),U32.to_nat(columns))),Nat.mul(position,blocks)),Nat.div(on,8n)),
      Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),position),blocks),Nat.div(on,8n)),
      gradient_address(image,position,blocks,U32.to_nat(rows),U32.to_nat(columns),Nat.div(on,8n))) :
    {TileLaws.select_lane(Nat.mod(on,8n),Representation.lifted(~DL,~d_value,holder,U32.from_nat(_))) == spec : F32}
  %Ceiling.shifted_formula(U32.to_nat(o),8n,{==}) :
    {TileLaws.select_lane(Nat.mod(on,8n),Representation.lifted(~DL,~d_value,holder,
      U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),position),_),Nat.div(on,8n))))) == spec : F32}
  lifted_lane(~DL,~d_value,logical,outer_word,after,lanes,groups,n,U32.to_nat(o),U32.to_nat(rows),U32.to_nat(columns),image,on,position,capacity,after_exact,lanes_exact,
    groups_exact,outer_exact,total_room,image_inside,output_inside,position_inside)

# A value picked inside the image is the sample the contract reads when it
# is the value there.
def inside_sampled(~Source: Data,~at: Source -> Nat -> F32,inside: Bool,+left: F32,+input: Source,+index: Nat,
  same: @inside_true: {inside == True{} : Bool} -> {left == at(input,index) : F32}) ->
  {Bool.pick(F32,inside,left,0.0) == Spec.sampled(~Source,~at,inside,input,index) : F32}:
  match inside:
    case False{}: {==}
    case True{}: same({==})

# Inside the image, the input lane the weight model reads at a pixel is the
# value the contract reads there.
def input_inside(~XL: Data,~x_value: XL -> U32 -> F32,+logical: XL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +n: Nat,+c: Nat,+h: Nat,+w: Nat,+p: Nat,+image: Nat,+channel: Nat,+y: Nat,+x: Nat,+capacity: U32,
  +after_exact: {U32.to_nat(after) == Nat.mul(h,w) : Nat},+lanes_exact: {U32.to_nat(lanes) == c : Nat},+groups_exact: {U32.to_nat(groups) == Kernel.groups(c) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(n,Nat.mul(h,w)) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(h,w)),Kernel.groups(c)),U32.to_nat(capacity)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,n) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,c) == True{} : Bool},
  +inside_true: {((Nat.is_le(p,y) && Nat.is_lt(y,Nat.add(p,h))) && Nat.is_le(p,x)) && Nat.is_lt(x,Nat.add(p,w)) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Representation.lifted(~XL,~x_value,Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}},
      U32.from_nat(Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(Ceiling.ceiling(c,8n),h),w)),Nat.div(channel,8n)),
        Nat.mul(Nat.add(Nat.mul(Nat.sub(y,p),w),Nat.sub(x,p)),Ceiling.ceiling(c,8n)))))) ==
    x_value(logical,U32.from_nat(Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,c),channel),h),w),Nat.mul(Nat.sub(y,p),w)),Nat.sub(x,p)))) : F32}:
  +blocks = Ceiling.ceiling(c,8n)
  +block = Nat.div(channel,8n)
  +row = Nat.sub(y,p)
  +column = Nat.sub(x,p)
  +pixel = Nat.add(Nat.mul(row,w),column)
  +first = Nat.mul(Nat.mul(Nat.add(Nat.mul(image,c),channel),h),w)
  +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}} : Representation.Lifted<XL>}
  +spec = x_value(logical,U32.from_nat(Nat.add(Nat.add(first,Nat.mul(row,w)),column)))
  +front = Logic.and_true_left((Nat.is_le(p,y) && Nat.is_lt(y,Nat.add(p,h))) && Nat.is_le(p,x),Nat.is_lt(x,Nat.add(p,w)),inside_true)
  +x_below = Logic.and_true_right((Nat.is_le(p,y) && Nat.is_lt(y,Nat.add(p,h))) && Nat.is_le(p,x),Nat.is_lt(x,Nat.add(p,w)),inside_true)
  +rows_true = Logic.and_true_left(Nat.is_le(p,y) && Nat.is_lt(y,Nat.add(p,h)),Nat.is_le(p,x),front)
  +x_above = Logic.and_true_right(Nat.is_le(p,y) && Nat.is_lt(y,Nat.add(p,h)),Nat.is_le(p,x),front)
  +y_above = Logic.and_true_left(Nat.is_le(p,y),Nat.is_lt(y,Nat.add(p,h)),rows_true)
  +y_below = Logic.and_true_right(Nat.is_le(p,y),Nat.is_lt(y,Nat.add(p,h)),rows_true)
  +pixel_inside = RowMajor.radix(row,h,column,w,Order.sub_lt(y,p,h,y_above,y_below),Order.sub_lt(x,p,w,x_above,x_below))
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(image,Nat.mul(Nat.mul(blocks,h),w)),block),Nat.mul(pixel,blocks)),Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(h,w)),pixel),blocks),block),
      input_address(image,pixel,blocks,h,w,block)) :
    {TileLaws.select_lane(Nat.mod(channel,8n),Representation.lifted(~XL,~x_value,holder,U32.from_nat(_))) == spec : F32}
  %Ceiling.shifted_formula(c,8n,{==}) :
    {TileLaws.select_lane(Nat.mod(channel,8n),Representation.lifted(~XL,~x_value,holder,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(h,w)),pixel),_),block)))) == spec : F32}
  %Equal.sym(Nat,Nat.add(Nat.add(first,Nat.mul(row,w)),column),Nat.add(first,pixel),Addition.associative(first,Nat.mul(row,w),column)) :
    {TileLaws.select_lane(Nat.mod(channel,8n),Representation.lifted(~XL,~x_value,holder,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(h,w)),pixel),Kernel.groups(c)),block)))) ==
      x_value(logical,U32.from_nat(_)) : F32}
  lifted_lane(~XL,~x_value,logical,outer_word,after,lanes,groups,n,c,h,w,image,channel,pixel,capacity,after_exact,lanes_exact,groups_exact,outer_exact,total_room,image_inside,
    channel_inside,pixel_inside)

# The input the weight model reads at a tap and a position of an image of
# the lifted input is the sample the contract reads there.
def input_sample(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+logical: XL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +gradient: Representation.Lifted<DL>,+c: U32,+o: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+rows: U32,+columns: U32,+taps: U32,+d_first: Nat,
  +n: Nat,+image: Nat,+tap: U32,+position: Nat,+capacity: U32,
  +after_exact: {U32.to_nat(after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},+lanes_exact: {U32.to_nat(lanes) == U32.to_nat(c) : Nat},
  +groups_exact: {U32.to_nat(groups) == Kernel.groups(U32.to_nat(c)) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))),Kernel.groups(U32.to_nat(c))),U32.to_nat(capacity)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,n) == True{} : Bool},+blocks_exact: {U32.to_nat(((c + 7) / 8 : U32)) == Ceiling.ceiling(U32.to_nat(c),8n) : Nat},
  +square_positive: {Nat.is_lt(0n,Nat.mul(U32.to_nat(k),U32.to_nat(k))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}) ->
  {Sources.held_weight(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
      Sources.HeldSource{Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}},gradient,WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},
        Nat.mul(image,Sources.input_cells(WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps})),d_first},tap,position) ==
    Spec.sampled(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),
      ((Nat.is_le(U32.to_nat(p),Nat.add(Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(k)),Nat.mul(Nat.div(position,U32.to_nat(columns)),U32.to_nat(s)))) &&
        Nat.is_lt(Nat.add(Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(k)),Nat.mul(Nat.div(position,U32.to_nat(columns)),U32.to_nat(s))),Nat.add(U32.to_nat(p),U32.to_nat(h)))) &&
        Nat.is_le(U32.to_nat(p),Nat.add(Nat.mod(U32.to_nat(tap),U32.to_nat(k)),Nat.mul(Nat.mod(position,U32.to_nat(columns)),U32.to_nat(s))))) &&
        Nat.is_lt(Nat.add(Nat.mod(U32.to_nat(tap),U32.to_nat(k)),Nat.mul(Nat.mod(position,U32.to_nat(columns)),U32.to_nat(s))),Nat.add(U32.to_nat(p),U32.to_nat(w))),
      InputOperand{logical},
      Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(c)),Nat.div(U32.to_nat(tap),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(h)),U32.to_nat(w)),
        Nat.mul(Nat.sub(Nat.add(Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(k)),Nat.mul(Nat.div(position,U32.to_nat(columns)),U32.to_nat(s))),U32.to_nat(p)),U32.to_nat(w))),
        Nat.sub(Nat.add(Nat.mod(U32.to_nat(tap),U32.to_nat(k)),Nat.mul(Nat.mod(position,U32.to_nat(columns)),U32.to_nat(s))),U32.to_nat(p)))) : F32}:
  +kn = U32.to_nat(k)
  +sn = U32.to_nat(s)
  +pn = U32.to_nat(p)
  +hn = U32.to_nat(h)
  +wn = U32.to_nat(w)
  +cn = U32.to_nat(c)
  +tn = U32.to_nat(tap)
  +row = Nat.div(position,U32.to_nat(columns))
  +column = Nat.mod(position,U32.to_nat(columns))
  +channel = Nat.div(tn,Nat.mul(kn,kn))
  +kernel_row = Nat.mod(Nat.div(tn,kn),kn)
  +kernel_column = Nat.mod(tn,kn)
  +y = Nat.add(kernel_row,Nat.mul(row,sn))
  +x = Nat.add(kernel_column,Nat.mul(column,sn))
  +blocks = Ceiling.ceiling(cn,8n)
  +first = Nat.mul(image,Nat.mul(Nat.mul(blocks,hn),wn))
  +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}} : Representation.Lifted<XL>}
  +inside = ((Nat.is_le(pn,y) && Nat.is_lt(y,Nat.add(pn,hn))) && Nat.is_le(pn,x)) && Nat.is_lt(x,Nat.add(pn,wn))
  +index = Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,cn),channel),hn),wn),Nat.mul(Nat.sub(y,pn),wn)),Nat.sub(x,pn))
  +address = Nat.add(Nat.add(first,Nat.div(channel,8n)),Nat.mul(Nat.add(Nat.mul(Nat.sub(y,pn),wn),Nat.sub(x,pn)),blocks))
  +rhs = Spec.sampled(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),inside,InputOperand{logical},index)
  +row_exact = FullRange.roundtrip(row,rows,Bounds.strict_le(row,U32.to_nat(rows),
    DivisionOrder.quotient_below(position,U32.to_nat(rows),U32.to_nat(columns),columns_positive,position_inside)))
  +column_exact = FullRange.roundtrip(column,columns,Bounds.strict_le(column,U32.to_nat(columns),DivisionOrder.remainder_small(position,U32.to_nat(columns),columns_positive)))
  +channel_inside = DivisionOrder.quotient_below(tn,cn,Nat.mul(kn,kn),square_positive,tap_inside)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(row)),row,row_exact) :
    {Packing.sampled_cell(~Representation.Lifted<XL>,~Sources.held_cell(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value)),holder,first,channel,
      Nat.add(kernel_row,Nat.mul(_,sn)),Nat.add(kernel_column,Nat.mul(U32.to_nat(U32.from_nat(column)),sn)),hn,wn,pn,U32.to_nat(((c + 7) / 8 : U32))) == rhs : F32}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(column)),column,column_exact) :
    {Packing.sampled_cell(~Representation.Lifted<XL>,~Sources.held_cell(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value)),holder,first,channel,
      y,Nat.add(kernel_column,Nat.mul(_,sn)),hn,wn,pn,U32.to_nat(((c + 7) / 8 : U32))) == rhs : F32}
  %Equal.sym(Nat,U32.to_nat(((c + 7) / 8 : U32)),blocks,blocks_exact) :
    {Packing.sampled_cell(~Representation.Lifted<XL>,~Sources.held_cell(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value)),holder,first,channel,
      y,x,hn,wn,pn,_) == rhs : F32}
  %Logic.and_associative(Nat.is_le(pn,y) && Nat.is_lt(y,Nat.add(pn,hn)),Nat.is_le(pn,x),Nat.is_lt(x,Nat.add(pn,wn))) :
    {Bool.pick(F32,_,TileLaws.select_lane(Nat.mod(channel,8n),Representation.lifted(~XL,~x_value,holder,U32.from_nat(address))),0.0) == rhs : F32}
  inside_sampled(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),inside,
    TileLaws.select_lane(Nat.mod(channel,8n),Representation.lifted(~XL,~x_value,holder,U32.from_nat(address))),InputOperand{logical},index,
    inside_true => input_inside(~XL,~x_value,logical,outer_word,after,lanes,groups,n,cn,hn,wn,pn,image,channel,y,x,capacity,after_exact,lanes_exact,groups_exact,outer_exact,
      total_room,image_inside,channel_inside,inside_true))

# The weight term of the contract at the decoded output, channel and kernel
# position of an element.
def coordinate_term(~Source: Data,~at: Source -> Nat -> F32,+p: Spec.Problem<Source>,+image: Nat,+output: Nat,+channel: Nat,+kernel_row: Nat,+kernel_column: Nat,
  +position: Nat) -> F32:
  Spec.Problem{Spec.Shape{n,+c,+o,+h,+w,k,+s,+pad,+rows,+columns,g},input,gradient} = p
  +y = Nat.add(kernel_row,Nat.mul(Nat.div(position,columns),s))
  +x = Nat.add(kernel_column,Nat.mul(Nat.mod(position,columns),s))
  (at(gradient,Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),output),rows),columns),position)) *
    Spec.sampled(~Source,~at,Nat.is_le(pad,y) && Nat.is_lt(y,Nat.add(pad,h)) && Nat.is_le(pad,x) && Nat.is_lt(x,Nat.add(pad,w)),input,
      Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,c),channel),h),w),Nat.mul(Nat.sub(y,pad),w)),Nat.sub(x,pad))) : F32)

# A product of the weight model over the lifted input and output gradient,
# at a tap and an output inside the weights and a position inside the plane
# of an image, is the term the weight contract reads.
def weight_sample(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,
  +x_logical: XL,+x_outer: U32,+x_after: U32,+x_lanes: U32,+x_groups: U32,+d_logical: DL,+d_outer: U32,+d_after: U32,+d_lanes: U32,+d_groups: U32,
  +c: U32,+o: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+rows: U32,+columns: U32,+taps: U32,+n: Nat,+g: Nat,+image: Nat,+tap: U32,+output: U32,+position: Nat,
  +capacity: U32,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},+x_lanes_exact: {U32.to_nat(x_lanes) == U32.to_nat(c) : Nat},
  +x_groups_exact: {U32.to_nat(x_groups) == Kernel.groups(U32.to_nat(c)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +x_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))),Kernel.groups(U32.to_nat(c))),U32.to_nat(capacity)) == True{} : Bool},
  +blocks_exact: {U32.to_nat(((c + 7) / 8 : U32)) == Ceiling.ceiling(U32.to_nat(c),8n) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},+d_lanes_exact: {U32.to_nat(d_lanes) == U32.to_nat(o) : Nat},
  +d_groups_exact: {U32.to_nat(d_groups) == Kernel.groups(U32.to_nat(o)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +d_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Kernel.groups(U32.to_nat(o))),U32.to_nat(capacity)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,n) == True{} : Bool},+channels_positive: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},
  +kernel_positive: {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool},+columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) == True{} : Bool},
  +output_inside: {Nat.is_lt(U32.to_nat(output),U32.to_nat(o)) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}) ->
  {(Sources.held_input(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        Sources.HeldSource{Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}},
          Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}},WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},
          Nat.mul(image,Sources.input_cells(WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps})),
          Nat.mul(image,Groups.gradient_extent(WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps}))},output,position) *
      Sources.held_weight(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        Sources.HeldSource{Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}},
          Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}},WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},
          Nat.mul(image,Sources.input_cells(WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps})),
          Nat.mul(image,Groups.gradient_extent(WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps}))},tap,position) : F32) ==
    Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),
      Spec.Problem{shape(Kernel.Work{n,U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p),U32.to_nat(rows),U32.to_nat(columns),g}),
        InputOperand{x_logical},GradientOperand{d_logical}},
      image,Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap)),position) : F32}:
  +cn = U32.to_nat(c)
  +kn = U32.to_nat(k)
  +tn = U32.to_nat(tap)
  +on = U32.to_nat(output)
  +geometry = {WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : WeightKernel.Geometry}
  +x_holder = {Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}} : Representation.Lifted<XL>}
  +d_holder = {Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
  +source = {Sources.HeldSource{x_holder,d_holder,geometry,Nat.mul(image,Sources.input_cells(geometry)),Nat.mul(image,Groups.gradient_extent(geometry))} :
    Sources.HeldSource<Representation.Lifted<XL>,Representation.Lifted<DL>>}
  +held_gradient = Sources.held_input(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),source,output,
    position)
  +held_sample = Sources.held_weight(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),source,tap,
    position)
  +problem = {Spec.Problem{shape(Kernel.Work{n,cn,U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),kn,U32.to_nat(s),U32.to_nat(p),U32.to_nat(rows),U32.to_nat(columns),g}),
    InputOperand{x_logical},GradientOperand{d_logical}} : Spec.Problem<Operand<XL,DL>>}
  +element = Nat.add(Nat.mul(on,Nat.mul(cn,Nat.mul(kn,kn))),tn)
  +square_positive = Order.product_positive(kn,kn,kernel_positive,kernel_positive)
  +taps_positive = Order.product_positive(cn,Nat.mul(kn,kn),channels_positive,square_positive)
  +gradient_value = d_value(d_logical,U32.from_nat(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(o)),on),U32.to_nat(rows)),U32.to_nat(columns)),position)))
  +lhs = (held_gradient * held_sample : F32)
  +term = coordinate_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),problem,image,on,Nat.div(tn,Nat.mul(kn,kn)),Nat.mod(Nat.div(tn,kn),kn),Nat.mod(tn,kn),position)
  %Equal.sym(Nat,Nat.div(element,Nat.mul(cn,Nat.mul(kn,kn))),on,Packing.weight_output_decode(on,cn,kn,tn,taps_positive,tap_inside)) :
    {lhs == coordinate_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),problem,image,_,Nat.mod(Nat.div(element,Nat.mul(kn,kn)),cn),Nat.mod(Nat.div(element,kn),kn),
      Nat.mod(element,kn),position) : F32}
  %Equal.sym(Nat,Nat.mod(Nat.div(element,Nat.mul(kn,kn)),cn),Nat.div(tn,Nat.mul(kn,kn)),
      Packing.weight_channel_decode(on,cn,kn,tn,channels_positive,square_positive,tap_inside)) :
    {lhs == coordinate_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),problem,image,on,_,Nat.mod(Nat.div(element,kn),kn),Nat.mod(element,kn),position) : F32}
  %Equal.sym(Nat,Nat.mod(Nat.div(element,kn),kn),Nat.mod(Nat.div(tn,kn),kn),Packing.weight_row_decode(on,cn,kn,tn,kernel_positive)) :
    {lhs == coordinate_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),problem,image,on,Nat.div(tn,Nat.mul(kn,kn)),_,Nat.mod(element,kn),position) : F32}
  %Equal.sym(Nat,Nat.mod(element,kn),Nat.mod(tn,kn),Packing.weight_column_decode(on,cn,kn,tn,kernel_positive)) :
    {lhs == coordinate_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),problem,image,on,Nat.div(tn,Nat.mul(kn,kn)),Nat.mod(Nat.div(tn,kn),kn),_,position) : F32}
  %Equal.sym(F32,held_gradient,gradient_value,
      gradient_sample(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~DL,~d_value,x_holder,d_logical,d_outer,d_after,d_lanes,d_groups,c,o,h,w,k,s,p,rows,columns,taps,
        Nat.mul(image,Sources.input_cells(geometry)),n,image,output,position,capacity,d_after_exact,d_lanes_exact,d_groups_exact,d_outer_exact,d_room,image_inside,output_inside,
        position_inside)) :
    {(_ * held_sample : F32) == term : F32}
  %Equal.sym(F32,held_sample,
      Spec.sampled(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),
        ((Nat.is_le(U32.to_nat(p),Nat.add(Nat.mod(Nat.div(tn,kn),kn),Nat.mul(Nat.div(position,U32.to_nat(columns)),U32.to_nat(s)))) &&
          Nat.is_lt(Nat.add(Nat.mod(Nat.div(tn,kn),kn),Nat.mul(Nat.div(position,U32.to_nat(columns)),U32.to_nat(s))),Nat.add(U32.to_nat(p),U32.to_nat(h)))) &&
          Nat.is_le(U32.to_nat(p),Nat.add(Nat.mod(tn,kn),Nat.mul(Nat.mod(position,U32.to_nat(columns)),U32.to_nat(s))))) &&
          Nat.is_lt(Nat.add(Nat.mod(tn,kn),Nat.mul(Nat.mod(position,U32.to_nat(columns)),U32.to_nat(s))),Nat.add(U32.to_nat(p),U32.to_nat(w))),
        InputOperand{x_logical},
        Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,cn),Nat.div(tn,Nat.mul(kn,kn))),U32.to_nat(h)),U32.to_nat(w)),
          Nat.mul(Nat.sub(Nat.add(Nat.mod(Nat.div(tn,kn),kn),Nat.mul(Nat.div(position,U32.to_nat(columns)),U32.to_nat(s))),U32.to_nat(p)),U32.to_nat(w))),
          Nat.sub(Nat.add(Nat.mod(tn,kn),Nat.mul(Nat.mod(position,U32.to_nat(columns)),U32.to_nat(s))),U32.to_nat(p)))),
      input_sample(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,x_lanes,x_groups,d_holder,c,o,h,w,k,s,p,rows,columns,taps,
        Nat.mul(image,Groups.gradient_extent(geometry)),n,image,tap,position,capacity,x_after_exact,x_lanes_exact,x_groups_exact,x_outer_exact,x_room,image_inside,blocks_exact,
        square_positive,tap_inside,columns_positive,position_inside)) :
    {(gradient_value * _ : F32) == term : F32}
  {==}

# The cells the weight model reads for an image of the lifted input and
# output gradient, and the contract over their readers.
def weight_source(~XL: Data,~DL: Data,+x: Representation.Lifted<XL>,+d: Representation.Lifted<DL>,+geometry: WeightKernel.Geometry,+image: Nat) ->
  Sources.HeldSource<Representation.Lifted<XL>,Representation.Lifted<DL>>:
  Sources.HeldSource{x,d,geometry,Nat.mul(image,Sources.input_cells(geometry)),Nat.mul(image,Groups.gradient_extent(geometry))}

def weight_problem(~XL: Data,~DL: Data,+x_logical: XL,+d_logical: DL,+geometry: WeightKernel.Geometry,+n: Nat,+g: Nat) -> Spec.Problem<Operand<XL,DL>>:
  WeightKernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,taps} = geometry
  Spec.Problem{shape(Kernel.Work{n,U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p),U32.to_nat(rows),U32.to_nat(columns),g}),
    InputOperand{x_logical},GradientOperand{d_logical}}

# The products of the weight model over positions [position, position +
# remaining) of an image fold to the sum of the terms the contract reads.
def weight_fold(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,
  +x_logical: XL,+x_outer: U32,+x_after: U32,+x_lanes: U32,+x_groups: U32,+d_logical: DL,+d_outer: U32,+d_after: U32,+d_lanes: U32,+d_groups: U32,
  +c: U32,+o: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+rows: U32,+columns: U32,+taps: U32,+n: Nat,+g: Nat,+image: Nat,+tap: U32,+output: U32,+capacity: U32,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},+x_lanes_exact: {U32.to_nat(x_lanes) == U32.to_nat(c) : Nat},
  +x_groups_exact: {U32.to_nat(x_groups) == Kernel.groups(U32.to_nat(c)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +x_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))),Kernel.groups(U32.to_nat(c))),U32.to_nat(capacity)) == True{} : Bool},
  +blocks_exact: {U32.to_nat(((c + 7) / 8 : U32)) == Ceiling.ceiling(U32.to_nat(c),8n) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},+d_lanes_exact: {U32.to_nat(d_lanes) == U32.to_nat(o) : Nat},
  +d_groups_exact: {U32.to_nat(d_groups) == Kernel.groups(U32.to_nat(o)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +d_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Kernel.groups(U32.to_nat(o))),U32.to_nat(capacity)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,n) == True{} : Bool},+channels_positive: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},
  +kernel_positive: {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool},+columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) == True{} : Bool},
  +output_inside: {Nat.is_lt(U32.to_nat(output),U32.to_nat(o)) == True{} : Bool},
  remaining: Nat,+position: Nat,+end_inside: {Nat.is_le(Nat.add(position,remaining),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool},total: F32) ->
  {Sources.held_dot(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
      weight_source(~XL,~DL,Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}},
        Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}},WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},image),
      tap,output,remaining,position,total) ==
    ConvSpec.sum(Spec.terms(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),remaining,position,
      weight_problem(~XL,~DL,x_logical,d_logical,WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},n,g),image,
      Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap))),total) : F32}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +plane = Nat.mul(U32.to_nat(rows),U32.to_nat(columns))
      +geometry = {WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : WeightKernel.Geometry}
      +x_holder = {Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}} : Representation.Lifted<XL>}
      +d_holder = {Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
      +source = weight_source(~XL,~DL,x_holder,d_holder,geometry,image)
      +problem = weight_problem(~XL,~DL,x_logical,d_logical,geometry,n,g)
      +element = Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap))
      +product = (Sources.held_input(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),source,
          output,position) *
        Sources.held_weight(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),source,tap,position) : F32)
      +term = Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),problem,image,element,position)
      +same = weight_sample(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,x_lanes,x_groups,d_logical,d_outer,d_after,d_lanes,d_groups,c,o,h,w,k,s,p,rows,columns,taps,n,g,
        image,tap,output,position,capacity,x_after_exact,x_lanes_exact,x_groups_exact,x_outer_exact,x_room,blocks_exact,d_after_exact,d_lanes_exact,d_groups_exact,d_outer_exact,
        d_room,image_inside,channels_positive,kernel_positive,columns_positive,tap_inside,output_inside,Order.room_index(position,rest,plane,end_inside))
      %Equal.sym(F32,product,term,same) :
        {Sources.held_dot(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),source,tap,output,rest,
            1n+position,(total + _ : F32)) ==
          ConvSpec.sum(Spec.terms(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),rest,1n+position,problem,image,element),
            (total + term : F32)) : F32}
      weight_fold(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,x_lanes,x_groups,d_logical,d_outer,d_after,d_lanes,d_groups,c,o,h,w,k,s,p,rows,columns,taps,n,g,image,tap,
        output,capacity,x_after_exact,x_lanes_exact,x_groups_exact,x_outer_exact,x_room,blocks_exact,d_after_exact,d_lanes_exact,d_groups_exact,d_outer_exact,d_room,image_inside,
        channels_positive,kernel_positive,columns_positive,tap_inside,output_inside,rest,1n+position,Order.advance_room(position,rest,plane,end_inside),(total + term : F32))

# The contract has one group sum per range and one image sum per image.
def group_values_length(~Source: Data,~term: @p: Spec.Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,ranges: List<&2,Grouped.Range>,
  +p: Spec.Problem<Source>,+image: Nat,+element: Nat) ->
  {List.length(&2,F32,Spec.group_values(~Source,~term,ranges,p,image,element)) == List.length(&2,Grouped.Range,ranges) : Nat}:
  match ranges:
    case []: {==}
    case Grouped.Range{+first,+count} <> +rest:
      Equal.cong(Nat,Nat,value => 1n+value,List.length(&2,F32,Spec.group_values(~Source,~term,rest,p,image,element)),List.length(&2,Grouped.Range,rest),
        group_values_length(~Source,~term,rest,p,image,element))

def image_values_length(~Source: Data,~term: @p: Spec.Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,remaining: Nat,+image: Nat,+element: Nat,
  +p: Spec.Problem<Source>,+groups: List<&2,Grouped.Range>) ->
  {List.length(&2,F32,Spec.image_values(~Source,~term,remaining,image,element,p,groups)) == remaining : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,value => 1n+value,List.length(&2,F32,Spec.image_values(~Source,~term,rest,1n+image,element,p,groups)),rest,
        image_values_length(~Source,~term,rest,1n+image,element,p,groups))

# The group sums of the weight model over bounded ranges are the group sums
# the contract reads.
def weight_groups(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,
  +x_logical: XL,+x_outer: U32,+x_after: U32,+x_lanes: U32,+x_groups: U32,+d_logical: DL,+d_outer: U32,+d_after: U32,+d_lanes: U32,+d_groups: U32,
  +c: U32,+o: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+rows: U32,+columns: U32,+taps: U32,+n: Nat,+g: Nat,+image: Nat,+tap: U32,+output: U32,+capacity: U32,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},+x_lanes_exact: {U32.to_nat(x_lanes) == U32.to_nat(c) : Nat},
  +x_groups_exact: {U32.to_nat(x_groups) == Kernel.groups(U32.to_nat(c)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +x_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))),Kernel.groups(U32.to_nat(c))),U32.to_nat(capacity)) == True{} : Bool},
  +blocks_exact: {U32.to_nat(((c + 7) / 8 : U32)) == Ceiling.ceiling(U32.to_nat(c),8n) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},+d_lanes_exact: {U32.to_nat(d_lanes) == U32.to_nat(o) : Nat},
  +d_groups_exact: {U32.to_nat(d_groups) == Kernel.groups(U32.to_nat(o)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +d_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Kernel.groups(U32.to_nat(o))),U32.to_nat(capacity)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,n) == True{} : Bool},+channels_positive: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},
  +kernel_positive: {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool},+columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) == True{} : Bool},
  +output_inside: {Nat.is_lt(U32.to_nat(output),U32.to_nat(o)) == True{} : Bool},+size: Nat,
  ranges: List<&2,Grouped.Range>,
  +bounded: Members.All(Grouped.Range,{range => Reduction.RangeBounded(U32.to_nat(rows),U32.to_nat(columns),size,range) : Grouped.Range -> Data},ranges)) ->
  {Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Images.HeldLane<Representation.Lifted<XL>,Representation.Lifted<DL>>,
      ~Images.held_sample(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),ranges,
      Images.HeldLane{weight_source(~XL,~DL,Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}},
        Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}},WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},image),tap,output}) ==
    Spec.group_values(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),ranges,
      weight_problem(~XL,~DL,x_logical,d_logical,WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},n,g),image,
      Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap))) : List<&2,F32>}:
  match ranges:
    case []: {==}
    case Grouped.Range{+first,+count} <> +rest:
      KernelGemm.Both{+range_bound,+others} = bounded
      KernelGemm.Both{positive,+inner} = range_bound
      KernelGemm.Both{fits,+end} = inner
      +geometry = {WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : WeightKernel.Geometry}
      +x_holder = {Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}} : Representation.Lifted<XL>}
      +d_holder = {Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
      +lane = {Images.HeldLane{weight_source(~XL,~DL,x_holder,d_holder,geometry,image),tap,output} : Images.HeldLane<Representation.Lifted<XL>,Representation.Lifted<DL>>}
      +problem = weight_problem(~XL,~DL,x_logical,d_logical,geometry,n,g)
      +element = Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap))
      +model_head = Images.held_sample(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),lane,
        Grouped.Range{first,count})
      +spec_head = ConvSpec.sum(Spec.terms(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),count,first,problem,image,element),0.0)
      +model_rest = Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Images.HeldLane<Representation.Lifted<XL>,Representation.Lifted<DL>>,
        ~Images.held_sample(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),rest,lane)
      +spec_rest = Spec.group_values(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),rest,problem,image,element)
      Equal.trans(List<&2,F32>,model_head <> model_rest,spec_head <> model_rest,spec_head <> spec_rest,
        Equal.cong(F32,List<&2,F32>,value => value <> model_rest,model_head,spec_head,
          weight_fold(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,x_lanes,x_groups,d_logical,d_outer,d_after,d_lanes,d_groups,c,o,h,w,k,s,p,rows,columns,taps,n,g,image,
            tap,output,capacity,x_after_exact,x_lanes_exact,x_groups_exact,x_outer_exact,x_room,blocks_exact,d_after_exact,d_lanes_exact,d_groups_exact,d_outer_exact,d_room,
            image_inside,channels_positive,kernel_positive,columns_positive,tap_inside,output_inside,count,first,end,0.0)),
        Equal.cong(List<&2,F32>,List<&2,F32>,values => spec_head <> values,model_rest,spec_rest,
          weight_groups(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,x_lanes,x_groups,d_logical,d_outer,d_after,d_lanes,d_groups,c,o,h,w,k,s,p,rows,columns,taps,n,g,
            image,tap,output,capacity,x_after_exact,x_lanes_exact,x_groups_exact,x_outer_exact,x_room,blocks_exact,d_after_exact,d_lanes_exact,d_groups_exact,d_outer_exact,
            d_room,image_inside,channels_positive,kernel_positive,columns_positive,tap_inside,output_inside,size,rest,others)))

# The adjacent tree of the group sums of an image of the weight model is the
# image sum the contract reads.
def weight_image(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,
  +x_logical: XL,+x_outer: U32,+x_after: U32,+x_lanes: U32,+x_groups: U32,+d_logical: DL,+d_outer: U32,+d_after: U32,+d_lanes: U32,+d_groups: U32,
  +c: U32,+o: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+rows: U32,+columns: U32,+taps: U32,+n: Nat,+g: Nat,+image: Nat,+tap: U32,+output: U32,+capacity: U32,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},+x_lanes_exact: {U32.to_nat(x_lanes) == U32.to_nat(c) : Nat},
  +x_groups_exact: {U32.to_nat(x_groups) == Kernel.groups(U32.to_nat(c)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +x_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))),Kernel.groups(U32.to_nat(c))),U32.to_nat(capacity)) == True{} : Bool},
  +blocks_exact: {U32.to_nat(((c + 7) / 8 : U32)) == Ceiling.ceiling(U32.to_nat(c),8n) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},+d_lanes_exact: {U32.to_nat(d_lanes) == U32.to_nat(o) : Nat},
  +d_groups_exact: {U32.to_nat(d_groups) == Kernel.groups(U32.to_nat(o)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +d_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Kernel.groups(U32.to_nat(o))),U32.to_nat(capacity)) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,n) == True{} : Bool},+channels_positive: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},
  +kernel_positive: {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool},+columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) == True{} : Bool},
  +output_inside: {Nat.is_lt(U32.to_nat(output),U32.to_nat(o)) == True{} : Bool},+size: Nat,
  +ranges: List<&2,Grouped.Range>,
  +bounded: Members.All(Grouped.Range,{range => Reduction.RangeBounded(U32.to_nat(rows),U32.to_nat(columns),size,range) : Grouped.Range -> Data},ranges)) ->
  {Images.held_image(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
      Images.HeldLane{weight_source(~XL,~DL,Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}},
        Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}},WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},image),tap,output},ranges) ==
    Grouped.tree(Spec.group_values(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),ranges,
      weight_problem(~XL,~DL,x_logical,d_logical,WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},n,g),image,
      Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap)))) : F32}:
  +geometry = {WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : WeightKernel.Geometry}
  +x_holder = {Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}} : Representation.Lifted<XL>}
  +d_holder = {Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
  +lane = {Images.HeldLane{weight_source(~XL,~DL,x_holder,d_holder,geometry,image),tap,output} : Images.HeldLane<Representation.Lifted<XL>,Representation.Lifted<DL>>}
  +problem = weight_problem(~XL,~DL,x_logical,d_logical,geometry,n,g)
  +element = Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap))
  +collected = Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~Images.HeldLane<Representation.Lifted<XL>,Representation.Lifted<DL>>,
    ~Images.held_sample(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),ranges,lane)
  +values = Spec.group_values(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),ranges,problem,image,element)
  +same = weight_groups(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,x_lanes,x_groups,d_logical,d_outer,d_after,d_lanes,d_groups,c,o,h,w,k,s,p,rows,columns,taps,n,g,
    image,tap,output,capacity,x_after_exact,x_lanes_exact,x_groups_exact,x_outer_exact,x_room,blocks_exact,d_after_exact,d_lanes_exact,d_groups_exact,d_outer_exact,d_room,
    image_inside,channels_positive,kernel_positive,columns_positive,tap_inside,output_inside,size,ranges,bounded)
  %Equal.sym(List<&2,F32>,collected,values,same) :
    {Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),List.length(&2,Grouped.Range,ranges),Unit{},_) == Grouped.tree(values) : F32}
  %group_values_length(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),ranges,problem,image,element) :
    {Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),_,Unit{},values) == Grouped.tree(values) : F32}
  {==}

# The image sums of the weight model over a batch of the lifted input and
# output gradient are the image sums the contract reads.
def weight_images(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,
  +x_logical: XL,+x_outer: U32,+x_after: U32,+x_lanes: U32,+x_groups: U32,+d_logical: DL,+d_outer: U32,+d_after: U32,+d_lanes: U32,+d_groups: U32,
  +c: U32,+o: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+rows: U32,+columns: U32,+taps: U32,+n: Nat,+g: Nat,+tap: U32,+output: U32,+capacity: U32,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},+x_lanes_exact: {U32.to_nat(x_lanes) == U32.to_nat(c) : Nat},
  +x_groups_exact: {U32.to_nat(x_groups) == Kernel.groups(U32.to_nat(c)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +x_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))),Kernel.groups(U32.to_nat(c))),U32.to_nat(capacity)) == True{} : Bool},
  +blocks_exact: {U32.to_nat(((c + 7) / 8 : U32)) == Ceiling.ceiling(U32.to_nat(c),8n) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},+d_lanes_exact: {U32.to_nat(d_lanes) == U32.to_nat(o) : Nat},
  +d_groups_exact: {U32.to_nat(d_groups) == Kernel.groups(U32.to_nat(o)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +d_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Kernel.groups(U32.to_nat(o))),U32.to_nat(capacity)) == True{} : Bool},
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},
  +kernel_positive: {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool},+columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) == True{} : Bool},
  +output_inside: {Nat.is_lt(U32.to_nat(output),U32.to_nat(o)) == True{} : Bool},
  +batch: Images.Batch<Representation.Lifted<XL>,Representation.lifted(~XL,~x_value),Representation.Lifted<DL>,Representation.lifted(~DL,~d_value)>,
  +input_exact: {Images.batch_input(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch) ==
    Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}} : Representation.Lifted<XL>},
  +gradient_exact: {Images.batch_gradient(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch) ==
    Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}} : Representation.Lifted<DL>},
  +geometry_exact: {Images.batch_geometry(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch) ==
    WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : WeightKernel.Geometry},+size: Nat,
  +bounded: Members.All(Grouped.Range,{range => Reduction.RangeBounded(U32.to_nat(rows),U32.to_nat(columns),size,range) : Grouped.Range -> Data},
    Images.batch_ranges(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch)),
  remaining: Nat,+image: Nat,+end: {Nat.is_le(Nat.add(image,remaining),n) == True{} : Bool}) ->
  {Images.held_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),remaining,image,batch,tap,
      output) ==
    Spec.image_values(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),remaining,image,
      Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap)),
      weight_problem(~XL,~DL,x_logical,d_logical,WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},n,g),
      Images.batch_ranges(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch)) :
    List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +geometry = {WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : WeightKernel.Geometry}
      +x_holder = {Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}} : Representation.Lifted<XL>}
      +d_holder = {Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
      +ranges = Images.batch_ranges(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch)
      +problem = weight_problem(~XL,~DL,x_logical,d_logical,geometry,n,g)
      +element = Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap))
      +batch_input = Images.batch_input(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch)
      +batch_gradient = Images.batch_gradient(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        batch)
      +batch_geometry = Images.batch_geometry(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        batch)
      +model_lane = Images.batch_lane(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch,
        image,tap,output)
      +lane = {Images.HeldLane{weight_source(~XL,~DL,x_holder,d_holder,geometry,image),tap,output} : Images.HeldLane<Representation.Lifted<XL>,Representation.Lifted<DL>>}
      +lane_exact = {%input_exact :
          {model_lane == Images.HeldLane{Sources.HeldSource{_,d_holder,geometry,Nat.mul(image,Sources.input_cells(geometry)),Nat.mul(image,Groups.gradient_extent(geometry))},
            tap,output} : Images.HeldLane<Representation.Lifted<XL>,Representation.Lifted<DL>>}
        %gradient_exact :
          {model_lane == Images.HeldLane{Sources.HeldSource{batch_input,_,geometry,Nat.mul(image,Sources.input_cells(geometry)),Nat.mul(image,Groups.gradient_extent(geometry))},
            tap,output} : Images.HeldLane<Representation.Lifted<XL>,Representation.Lifted<DL>>}
        %geometry_exact :
          {model_lane == Images.HeldLane{Sources.HeldSource{batch_input,batch_gradient,_,Nat.mul(image,Sources.input_cells(_)),Nat.mul(image,Groups.gradient_extent(_))},
            tap,output} : Images.HeldLane<Representation.Lifted<XL>,Representation.Lifted<DL>>}
        Images.batch_lane_parts(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch,image,tap,
          output) :
        {model_lane == lane : Images.HeldLane<Representation.Lifted<XL>,Representation.Lifted<DL>>}}
      +model_head = Images.held_image(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),model_lane,
        ranges)
      +lane_head = Images.held_image(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),lane,ranges)
      +spec_head = Grouped.tree(Spec.group_values(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),ranges,problem,image,element))
      +model_rest = Images.held_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),rest,
        1n+image,batch,tap,output)
      +spec_rest = Spec.image_values(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),rest,1n+image,element,problem,ranges)
      +head = Equal.trans(F32,model_head,lane_head,spec_head,
        Equal.cong(Images.HeldLane<Representation.Lifted<XL>,Representation.Lifted<DL>>,F32,
          held => Images.held_image(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),held,ranges),
          model_lane,lane,lane_exact),
        weight_image(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,x_lanes,x_groups,d_logical,d_outer,d_after,d_lanes,d_groups,c,o,h,w,k,s,p,rows,columns,taps,n,g,image,
          tap,output,capacity,x_after_exact,x_lanes_exact,x_groups_exact,x_outer_exact,x_room,blocks_exact,d_after_exact,d_lanes_exact,d_groups_exact,d_outer_exact,d_room,
          Order.room_index(image,rest,n,end),channels_positive,kernel_positive,columns_positive,tap_inside,output_inside,size,ranges,bounded))
      Equal.trans(List<&2,F32>,model_head <> model_rest,spec_head <> model_rest,spec_head <> spec_rest,
        Equal.cong(F32,List<&2,F32>,value => value <> model_rest,model_head,spec_head,head),
        Equal.cong(List<&2,F32>,List<&2,F32>,values => spec_head <> values,model_rest,spec_rest,
          weight_images(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,x_lanes,x_groups,d_logical,d_outer,d_after,d_lanes,d_groups,c,o,h,w,k,s,p,rows,columns,taps,n,g,
            tap,output,capacity,x_after_exact,x_lanes_exact,x_groups_exact,x_outer_exact,x_room,blocks_exact,d_after_exact,d_lanes_exact,d_groups_exact,d_outer_exact,d_room,
            channels_positive,kernel_positive,columns_positive,tap_inside,output_inside,batch,input_exact,gradient_exact,geometry_exact,size,bounded,rest,1n+image,
            Order.advance_room(image,rest,n,end))))

# A lane of the weight model over a batch of the lifted input and output
# gradient, at a tap and an output inside the weights, is the weight
# gradient the contract reads.
def weight_lane(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,
  +x_logical: XL,+x_outer: U32,+x_after: U32,+x_lanes: U32,+x_groups: U32,+d_logical: DL,+d_outer: U32,+d_after: U32,+d_lanes: U32,+d_groups: U32,
  +c: U32,+o: U32,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+rows: U32,+columns: U32,+taps: U32,+n: Nat,+g: Nat,+tap: U32,+output: U32,+capacity: U32,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},+x_lanes_exact: {U32.to_nat(x_lanes) == U32.to_nat(c) : Nat},
  +x_groups_exact: {U32.to_nat(x_groups) == Kernel.groups(U32.to_nat(c)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +x_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(h),U32.to_nat(w))),Kernel.groups(U32.to_nat(c))),U32.to_nat(capacity)) == True{} : Bool},
  +blocks_exact: {U32.to_nat(((c + 7) / 8 : U32)) == Ceiling.ceiling(U32.to_nat(c),8n) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},+d_lanes_exact: {U32.to_nat(d_lanes) == U32.to_nat(o) : Nat},
  +d_groups_exact: {U32.to_nat(d_groups) == Kernel.groups(U32.to_nat(o)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +d_room: {Nat.is_le(Nat.mul(Nat.mul(n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Kernel.groups(U32.to_nat(o))),U32.to_nat(capacity)) == True{} : Bool},
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},
  +kernel_positive: {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool},+columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},
  +group_positive: {Nat.is_lt(0n,g) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) == True{} : Bool},
  +output_inside: {Nat.is_lt(U32.to_nat(output),U32.to_nat(o)) == True{} : Bool},
  +batch: Images.Batch<Representation.Lifted<XL>,Representation.lifted(~XL,~x_value),Representation.Lifted<DL>,Representation.lifted(~DL,~d_value)>,
  +input_exact: {Images.batch_input(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch) ==
    Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,x_lanes,x_groups,Grid.Innermost{}}} : Representation.Lifted<XL>},
  +gradient_exact: {Images.batch_gradient(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch) ==
    Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,d_lanes,d_groups,Grid.Innermost{}}} : Representation.Lifted<DL>},
  +geometry_exact: {Images.batch_geometry(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch) ==
    WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : WeightKernel.Geometry},
  +images_exact: {Images.batch_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch) ==
    n : Nat},
  +ranges_exact: {Images.batch_ranges(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch) ==
    Grouped.ranges(U32.to_nat(rows),U32.to_nat(columns),g) : List<&2,Grouped.Range>}) ->
  {Images.lane_levels(Images.batch_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch),
      Images.held_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        Images.batch_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch),0n,batch,
        tap,output)) ==
    Spec.weight(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),weight_problem(~XL,~DL,x_logical,d_logical,WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},n,g),
      Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap))) : F32}:
  +geometry = {WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : WeightKernel.Geometry}
  +ranges = Images.batch_ranges(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch)
  +whole = Grouped.ranges(U32.to_nat(rows),U32.to_nat(columns),g)
  +problem = weight_problem(~XL,~DL,x_logical,d_logical,geometry,n,g)
  +element = Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap))
  +weight = Spec.weight(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value),problem,element)
  +bounded = {%Equal.sym(List<&2,Grouped.Range>,ranges,whole,ranges_exact) :
      Members.All(Grouped.Range,{range => Reduction.RangeBounded(U32.to_nat(rows),U32.to_nat(columns),g,range) : Grouped.Range -> Data},_)
    Reduction.ranges_bounded(U32.to_nat(rows),U32.to_nat(columns),g,columns_positive,group_positive) :
    Members.All(Grouped.Range,{range => Reduction.RangeBounded(U32.to_nat(rows),U32.to_nat(columns),g,range) : Grouped.Range -> Data},ranges)}
  +model = Images.held_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),n,0n,batch,tap,output)
  +values = Spec.image_values(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),n,0n,element,problem,ranges)
  +same = weight_images(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,x_lanes,x_groups,d_logical,d_outer,d_after,d_lanes,d_groups,c,o,h,w,k,s,p,rows,columns,taps,n,g,
    tap,output,capacity,x_after_exact,x_lanes_exact,x_groups_exact,x_outer_exact,x_room,blocks_exact,d_after_exact,d_lanes_exact,d_groups_exact,d_outer_exact,d_room,
    channels_positive,kernel_positive,columns_positive,tap_inside,output_inside,batch,input_exact,gradient_exact,geometry_exact,g,bounded,n,0n,Order.reflexive(n))
  %Equal.sym(Nat,Images.batch_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch),n,
      images_exact) :
    {Images.lane_levels(_,Images.held_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
      _,0n,batch,tap,output)) == weight : F32}
  %Equal.sym(List<&2,F32>,model,values,same) : {Images.lane_levels(n,_) == weight : F32}
  %Equal.sym(List<&2,Grouped.Range>,ranges,whole,ranges_exact) :
    {Images.lane_levels(n,Spec.image_values(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),n,0n,element,problem,_)) == weight : F32}
  %image_values_length(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),n,0n,element,problem,whole) :
    {Images.lane_levels(_,Spec.image_values(~Operand<XL,DL>,~Spec.weight_term(~Operand<XL,DL>,~operand_at(~XL,~x_value,~DL,~d_value)),n,0n,element,problem,whole)) == weight : F32}
  {==}
