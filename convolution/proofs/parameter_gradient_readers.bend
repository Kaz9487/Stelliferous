# The weight and bias gradient contract read through value readers. The
# contract reads the input and the output gradient only inside their extents,
# so storage that holds the readers' values gives the readers' contract.
import Base
import ../parameter_gradient_specification.bend as Spec
import ../specification.bend as ConvSpec
import ../../lib/traversal_grouped.bend as Grouped
import ../../lib/tensor_parameter_gradient.bend as Parameter
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/grouped_reduction_refinement.bend as GroupedRanges
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/boolean_logic.bend as Logic
import ./parameter_gradient_bridge.bend as Bridge

# ---- Readers as the contract's source ----

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
        Spec.Problem{Bridge.shape(Parameter.work(x,d,kernel,stride,padding,group)),InputOperand{input},GradientOperand{gradient}},U32.to_nat(index))
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
      Spec.bias(~Gradient,~gradient_at(~Gradient,~gradient_value),Spec.Problem{Bridge.shape(Parameter.bias_work(d,group)),gradient,gradient},U32.to_nat(index))

def bias_at(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,context: BiasRead<Gradient>,index: U32) -> F32:
  BiasRead{+gradient,image,+group} = context
  bias_imaged(~Gradient,~gradient_value,image,group,gradient,index)

# ---- One contract value from agreeing terms ----

def terms_agree(~First: Data,~first_term: @p: Spec.Problem<First> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  ~Second: Data,~second_term: @p: Spec.Problem<Second> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  ~Scope: Data,~first: Scope -> Spec.Problem<First>,~second: Scope -> Spec.Problem<Second>,~first_element: Scope -> Nat,~second_element: Scope -> Nat,~images: Scope -> Nat,~rows: Scope -> Nat,~columns: Scope -> Nat,
  ~agree: @+scope: Scope -> @+image: Nat -> @+position: Nat -> @+image_inside: {Nat.is_lt(image,images(scope)) == True{} : Bool} ->
    @+position_inside: {Nat.is_lt(position,Nat.mul(rows(scope),columns(scope))) == True{} : Bool} ->
    {first_term(first(scope),image,first_element(scope),position) == second_term(second(scope),image,second_element(scope),position) : F32},
  remaining: Nat,+position: Nat,+scope: Scope,+image: Nat,+image_inside: {Nat.is_lt(image,images(scope)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,remaining),Nat.mul(rows(scope),columns(scope))) == True{} : Bool}) ->
  {Spec.terms(~First,~first_term,remaining,position,first(scope),image,first_element(scope)) == Spec.terms(~Second,~second_term,remaining,position,second(scope),image,second_element(scope)) : List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(F32,first_term(first(scope),image,first_element(scope),position),second_term(second(scope),image,second_element(scope),position),
        agree(scope,image,position,image_inside,Order.room_index(position,rest,Nat.mul(rows(scope),columns(scope)),room))) :
        {_ <> Spec.terms(~First,~first_term,rest,1n+position,first(scope),image,first_element(scope)) ==
          second_term(second(scope),image,second_element(scope),position) <> Spec.terms(~Second,~second_term,rest,1n+position,second(scope),image,second_element(scope)) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => second_term(second(scope),image,second_element(scope),position) <> tail,
        Spec.terms(~First,~first_term,rest,1n+position,first(scope),image,first_element(scope)),Spec.terms(~Second,~second_term,rest,1n+position,second(scope),image,second_element(scope)),
        terms_agree(~First,~first_term,~Second,~second_term,~Scope,~first,~second,~first_element,~second_element,~images,~rows,~columns,~agree,rest,1n+position,scope,image,image_inside,
          Order.advance_room(position,rest,Nat.mul(rows(scope),columns(scope)),room)))

def groups_agree(~First: Data,~first_term: @p: Spec.Problem<First> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  ~Second: Data,~second_term: @p: Spec.Problem<Second> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  ~Scope: Data,~first: Scope -> Spec.Problem<First>,~second: Scope -> Spec.Problem<Second>,~first_element: Scope -> Nat,~second_element: Scope -> Nat,~images: Scope -> Nat,~rows: Scope -> Nat,~columns: Scope -> Nat,
  ~group: Scope -> Nat,
  ~agree: @+scope: Scope -> @+image: Nat -> @+position: Nat -> @+image_inside: {Nat.is_lt(image,images(scope)) == True{} : Bool} ->
    @+position_inside: {Nat.is_lt(position,Nat.mul(rows(scope),columns(scope))) == True{} : Bool} ->
    {first_term(first(scope),image,first_element(scope),position) == second_term(second(scope),image,second_element(scope),position) : F32},
  groups: List<&2,Grouped.Range>,+scope: Scope,+image: Nat,+image_inside: {Nat.is_lt(image,images(scope)) == True{} : Bool},
  +bounded: Members.All(Grouped.Range,{range => GroupedRanges.RangeBounded(rows(scope),columns(scope),group(scope),range) : Grouped.Range -> Data},groups)) ->
  {Spec.group_values(~First,~first_term,groups,first(scope),image,first_element(scope)) == Spec.group_values(~Second,~second_term,groups,second(scope),image,second_element(scope)) : List<&2,F32>}:
  match groups:
    case []: {==}
    case Grouped.Range{+start,+count} <> +rest:
      Shape.Both{Shape.Both{+nonempty,Shape.Both{+small,+room}},+tail} = bounded
      +head = Equal.cong(List<&2,F32>,F32,values => ConvSpec.sum(values,0.0),
        Spec.terms(~First,~first_term,count,start,first(scope),image,first_element(scope)),Spec.terms(~Second,~second_term,count,start,second(scope),image,second_element(scope)),
        terms_agree(~First,~first_term,~Second,~second_term,~Scope,~first,~second,~first_element,~second_element,~images,~rows,~columns,~agree,count,start,scope,image,image_inside,room))
      %Equal.sym(F32,ConvSpec.sum(Spec.terms(~First,~first_term,count,start,first(scope),image,first_element(scope)),0.0),
        ConvSpec.sum(Spec.terms(~Second,~second_term,count,start,second(scope),image,second_element(scope)),0.0),head) :
        {_ <> Spec.group_values(~First,~first_term,rest,first(scope),image,first_element(scope)) ==
          ConvSpec.sum(Spec.terms(~Second,~second_term,count,start,second(scope),image,second_element(scope)),0.0) <> Spec.group_values(~Second,~second_term,rest,second(scope),image,second_element(scope)) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,values => ConvSpec.sum(Spec.terms(~Second,~second_term,count,start,second(scope),image,second_element(scope)),0.0) <> values,
        Spec.group_values(~First,~first_term,rest,first(scope),image,first_element(scope)),Spec.group_values(~Second,~second_term,rest,second(scope),image,second_element(scope)),
        groups_agree(~First,~first_term,~Second,~second_term,~Scope,~first,~second,~first_element,~second_element,~images,~rows,~columns,~group,~agree,rest,scope,image,image_inside,tail))

def images_agree(~First: Data,~first_term: @p: Spec.Problem<First> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  ~Second: Data,~second_term: @p: Spec.Problem<Second> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  ~Scope: Data,~first: Scope -> Spec.Problem<First>,~second: Scope -> Spec.Problem<Second>,~first_element: Scope -> Nat,~second_element: Scope -> Nat,~images: Scope -> Nat,~rows: Scope -> Nat,~columns: Scope -> Nat,
  ~group: Scope -> Nat,
  ~agree: @+scope: Scope -> @+image: Nat -> @+position: Nat -> @+image_inside: {Nat.is_lt(image,images(scope)) == True{} : Bool} ->
    @+position_inside: {Nat.is_lt(position,Nat.mul(rows(scope),columns(scope))) == True{} : Bool} ->
    {first_term(first(scope),image,first_element(scope),position) == second_term(second(scope),image,second_element(scope),position) : F32},
  remaining: Nat,+image: Nat,+scope: Scope,+room: {Nat.is_le(Nat.add(image,remaining),images(scope)) == True{} : Bool},
  +bounded: Members.All(Grouped.Range,{range => GroupedRanges.RangeBounded(rows(scope),columns(scope),group(scope),range) : Grouped.Range -> Data},
    Grouped.ranges(rows(scope),columns(scope),group(scope)))) ->
  {Spec.image_values(~First,~first_term,remaining,image,first_element(scope),first(scope),Grouped.ranges(rows(scope),columns(scope),group(scope))) ==
    Spec.image_values(~Second,~second_term,remaining,image,second_element(scope),second(scope),Grouped.ranges(rows(scope),columns(scope),group(scope))) : List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +groups = Grouped.ranges(rows(scope),columns(scope),group(scope))
      +head = Equal.cong(List<&2,F32>,F32,values => Grouped.tree(values),
        Spec.group_values(~First,~first_term,groups,first(scope),image,first_element(scope)),Spec.group_values(~Second,~second_term,groups,second(scope),image,second_element(scope)),
        groups_agree(~First,~first_term,~Second,~second_term,~Scope,~first,~second,~first_element,~second_element,~images,~rows,~columns,~group,~agree,groups,scope,image,
          Order.room_index(image,rest,images(scope),room),bounded))
      %Equal.sym(F32,Grouped.tree(Spec.group_values(~First,~first_term,groups,first(scope),image,first_element(scope))),
        Grouped.tree(Spec.group_values(~Second,~second_term,groups,second(scope),image,second_element(scope))),head) :
        {_ <> Spec.image_values(~First,~first_term,rest,1n+image,first_element(scope),first(scope),groups) ==
          Grouped.tree(Spec.group_values(~Second,~second_term,groups,second(scope),image,second_element(scope))) <> Spec.image_values(~Second,~second_term,rest,1n+image,second_element(scope),second(scope),groups) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,values => Grouped.tree(Spec.group_values(~Second,~second_term,groups,second(scope),image,second_element(scope))) <> values,
        Spec.image_values(~First,~first_term,rest,1n+image,first_element(scope),first(scope),groups),Spec.image_values(~Second,~second_term,rest,1n+image,second_element(scope),second(scope),groups),
        images_agree(~First,~first_term,~Second,~second_term,~Scope,~first,~second,~first_element,~second_element,~images,~rows,~columns,~group,~agree,rest,1n+image,scope,
          Order.advance_room(image,rest,images(scope),room),bounded))

# ---- Reads inside the extents ----

# The output gradient address of a term lies inside the gradient.
def gradient_inside(+images: Nat,+outputs: Nat,+rows: Nat,+columns: Nat,+image: Nat,+output: Nat,+position: Nat,
  +image_inside: {Nat.is_lt(image,images) == True{} : Bool},+output_inside: {Nat.is_lt(output,outputs) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,Nat.mul(rows,columns)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,outputs),output),rows),columns),position),Nat.mul(Nat.mul(images,outputs),Nat.mul(rows,columns))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(Nat.mul(Nat.add(Nat.mul(image,outputs),output),rows),columns),Nat.mul(Nat.add(Nat.mul(image,outputs),output),Nat.mul(rows,columns)),
    RowMajor.assoc(Nat.add(Nat.mul(image,outputs),output),rows,columns)) :
    {Nat.is_lt(Nat.add(_,position),Nat.mul(Nat.mul(images,outputs),Nat.mul(rows,columns))) == True{} : Bool}
  RowMajor.radix(Nat.add(Nat.mul(image,outputs),output),Nat.mul(images,outputs),position,Nat.mul(rows,columns),
    RowMajor.radix(image,images,output,outputs,image_inside,output_inside),position_inside)

# The input address a sampled term reads lies inside the input.
def input_inside(+images: Nat,+channels: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+columns: Nat,
  +image: Nat,+element: Nat,+position: Nat,+image_inside: {Nat.is_lt(image,images) == True{} : Bool},+channels_positive: {Nat.is_lt(0n,channels) == True{} : Bool},
  +truth: {Nat.is_le(padding,Nat.add(Nat.mod(Nat.div(element,kernel),kernel),Nat.mul(Nat.div(position,columns),stride))) &&
    Nat.is_lt(Nat.add(Nat.mod(Nat.div(element,kernel),kernel),Nat.mul(Nat.div(position,columns),stride)),Nat.add(padding,height)) &&
    Nat.is_le(padding,Nat.add(Nat.mod(element,kernel),Nat.mul(Nat.mod(position,columns),stride))) &&
    Nat.is_lt(Nat.add(Nat.mod(element,kernel),Nat.mul(Nat.mod(position,columns),stride)),Nat.add(padding,width)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,channels),Nat.mod(Nat.div(element,Nat.mul(kernel,kernel)),channels)),height),width),
      Nat.mul(Nat.sub(Nat.add(Nat.mod(Nat.div(element,kernel),kernel),Nat.mul(Nat.div(position,columns),stride)),padding),width)),
      Nat.sub(Nat.add(Nat.mod(element,kernel),Nat.mul(Nat.mod(position,columns),stride)),padding)),
    Nat.mul(Nat.mul(images,channels),Nat.mul(height,width))) == True{} : Bool}:
  +y = Nat.add(Nat.mod(Nat.div(element,kernel),kernel),Nat.mul(Nat.div(position,columns),stride))
  +x = Nat.add(Nat.mod(element,kernel),Nat.mul(Nat.mod(position,columns),stride))
  +plane = Nat.add(Nat.mul(image,channels),Nat.mod(Nat.div(element,Nat.mul(kernel,kernel)),channels))
  +rows_and_low = Logic.left(Nat.is_le(padding,y) && Nat.is_lt(y,Nat.add(padding,height)) && Nat.is_le(padding,x),Nat.is_lt(x,Nat.add(padding,width)),truth)
  +x_upper = Logic.right(Nat.is_le(padding,y) && Nat.is_lt(y,Nat.add(padding,height)) && Nat.is_le(padding,x),Nat.is_lt(x,Nat.add(padding,width)),truth)
  +rows = Logic.left(Nat.is_le(padding,y) && Nat.is_lt(y,Nat.add(padding,height)),Nat.is_le(padding,x),rows_and_low)
  +x_lower = Logic.right(Nat.is_le(padding,y) && Nat.is_lt(y,Nat.add(padding,height)),Nat.is_le(padding,x),rows_and_low)
  +y_lower = Logic.left(Nat.is_le(padding,y),Nat.is_lt(y,Nat.add(padding,height)),rows)
  +y_upper = Logic.right(Nat.is_le(padding,y),Nat.is_lt(y,Nat.add(padding,height)),rows)
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(Nat.mul(plane,height),width),Nat.mul(Nat.sub(y,padding),width)),Nat.sub(x,padding)),
    RowMajor.index(plane,height,width,Nat.sub(y,padding),Nat.sub(x,padding)),RowMajor.expanded(plane,height,width,Nat.sub(y,padding),Nat.sub(x,padding))) :
    {Nat.is_lt(_,Nat.mul(Nat.mul(images,channels),Nat.mul(height,width))) == True{} : Bool}
  RowMajor.in_bounds(plane,Nat.mul(images,channels),height,width,Nat.sub(y,padding),Nat.sub(x,padding),
    RowMajor.radix(image,images,Nat.mod(Nat.div(element,Nat.mul(kernel,kernel)),channels),channels,image_inside,
      DivisionOrder.remainder_small(Nat.div(element,Nat.mul(kernel,kernel)),channels,channels_positive)),
    Order.sub_lt(y,padding,height,y_lower,y_upper),Order.sub_lt(x,padding,width,x_lower,x_upper))

def sampled_agrees(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,inside: Bool,
  +tree: Tree.Tree<F32>,+length: Nat,+context: Input,+values: Contract.Values(~F32,~Input,~input_value,context,Buffer.Content{tree,length}),+index: Nat,
  within: @truth: {inside == True{} : Bool} -> {Nat.is_lt(index,length) == True{} : Bool}) ->
  {Spec.sampled(~Tree.Tree<F32>,~Bridge.at,inside,tree,index) ==
    Spec.sampled(~Operand<Input,Gradient>,~operand_at(~Input,~input_value,~Gradient,~gradient_value),inside,InputOperand{context},index) : F32}:
  match inside:
    case False{}: {==}
    case True{}: Contract.read(~F32,~Input,~input_value,context,Buffer.Content{tree,length},index,values,within({==}))

# ---- The weight contract ----

# Storage holding the input and the gradient, each with the values of its
# reader inside its extents, at one weight element.
type WeightScope<-Input: Data,-input_value: Input -> U32 -> F32,-Gradient: Data,-gradient_value: Gradient -> U32 -> F32> is Data:
  WeightScope{images: Nat,channels: Nat,outputs: Nat,height: Nat,width: Nat,kernel: Nat,stride: Nat,padding: Nat,rows: Nat,columns: Nat,group: Nat,
    input_tree: Tree.Tree<F32>,input_length: Nat,input_context: Input,
    input_values: Contract.Values(~F32,~Input,~input_value,input_context,Buffer.Content{input_tree,input_length}),
    input_room: {Nat.is_le(Nat.mul(Nat.mul(images,channels),Nat.mul(height,width)),input_length) == True{} : Bool},
    gradient_tree: Tree.Tree<F32>,gradient_length: Nat,gradient_context: Gradient,
    gradient_values: Contract.Values(~F32,~Gradient,~gradient_value,gradient_context,Buffer.Content{gradient_tree,gradient_length}),
    gradient_room: {Nat.is_le(Nat.mul(Nat.mul(images,outputs),Nat.mul(rows,columns)),gradient_length) == True{} : Bool},
    element: Nat,element_inside: {Nat.is_lt(element,Nat.mul(outputs,Nat.mul(channels,Nat.mul(kernel,kernel)))) == True{} : Bool},
    kernel_positive: {Nat.is_lt(0n,kernel) == True{} : Bool},channels_positive: {Nat.is_lt(0n,channels) == True{} : Bool},
    columns_positive: {Nat.is_lt(0n,columns) == True{} : Bool},group_positive: {Nat.is_lt(0n,group) == True{} : Bool}}

def weight_stored(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,
  scope: WeightScope<Input,input_value,Gradient,gradient_value>) -> Spec.Problem<Tree.Tree<F32>>:
  WeightScope{n,c,o,h,w,k,s,pad,rows,columns,g,input_tree,input_length,input_context,input_values,input_room,
    gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,kernel_positive,channels_positive,columns_positive,group_positive} = scope
  Spec.Problem{Spec.Shape{n,c,o,h,w,k,s,pad,rows,columns,g},input_tree,gradient_tree}

def weight_read(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,
  scope: WeightScope<Input,input_value,Gradient,gradient_value>) -> Spec.Problem<Operand<Input,Gradient>>:
  WeightScope{n,c,o,h,w,k,s,pad,rows,columns,g,input_tree,input_length,input_context,input_values,input_room,
    gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,kernel_positive,channels_positive,columns_positive,group_positive} = scope
  Spec.Problem{Spec.Shape{n,c,o,h,w,k,s,pad,rows,columns,g},InputOperand{input_context},GradientOperand{gradient_context}}

def weight_element(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,
  scope: WeightScope<Input,input_value,Gradient,gradient_value>) -> Nat:
  WeightScope{n,c,o,h,w,k,s,pad,rows,columns,g,input_tree,input_length,input_context,input_values,input_room,
    gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,kernel_positive,channels_positive,columns_positive,group_positive} = scope
  element

def weight_images(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,
  scope: WeightScope<Input,input_value,Gradient,gradient_value>) -> Nat:
  WeightScope{n,c,o,h,w,k,s,pad,rows,columns,g,input_tree,input_length,input_context,input_values,input_room,
    gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,kernel_positive,channels_positive,columns_positive,group_positive} = scope
  n

def weight_rows(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,
  scope: WeightScope<Input,input_value,Gradient,gradient_value>) -> Nat:
  WeightScope{n,c,o,h,w,k,s,pad,rows,columns,g,input_tree,input_length,input_context,input_values,input_room,
    gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,kernel_positive,channels_positive,columns_positive,group_positive} = scope
  rows

def weight_columns(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,
  scope: WeightScope<Input,input_value,Gradient,gradient_value>) -> Nat:
  WeightScope{n,c,o,h,w,k,s,pad,rows,columns,g,input_tree,input_length,input_context,input_values,input_room,
    gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,kernel_positive,channels_positive,columns_positive,group_positive} = scope
  columns

def weight_group(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,
  scope: WeightScope<Input,input_value,Gradient,gradient_value>) -> Nat:
  WeightScope{n,c,o,h,w,k,s,pad,rows,columns,g,input_tree,input_length,input_context,input_values,input_room,
    gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,kernel_positive,channels_positive,columns_positive,group_positive} = scope
  g

# One weight term reads the same values from the storage and the readers.
def weight_term_agrees(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,
  +scope: WeightScope<Input,input_value,Gradient,gradient_value>,+image: Nat,+position: Nat,
  +image_inside: {Nat.is_lt(image,weight_images(~Input,~input_value,~Gradient,~gradient_value,scope)) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,Nat.mul(weight_rows(~Input,~input_value,~Gradient,~gradient_value,scope),weight_columns(~Input,~input_value,~Gradient,~gradient_value,scope))) == True{} : Bool}) ->
  {Spec.weight_term(~Tree.Tree<F32>,~Bridge.at,weight_stored(~Input,~input_value,~Gradient,~gradient_value,scope),image,weight_element(~Input,~input_value,~Gradient,~gradient_value,scope),position) ==
    Spec.weight_term(~Operand<Input,Gradient>,~operand_at(~Input,~input_value,~Gradient,~gradient_value),weight_read(~Input,~input_value,~Gradient,~gradient_value,scope),image,
      weight_element(~Input,~input_value,~Gradient,~gradient_value,scope),position) : F32}:
  WeightScope{+n,+c,+o,+h,+w,+k,+s,+pad,+rows,+columns,+g,+input_tree,+input_length,+input_context,+input_values,+input_room,
    +gradient_tree,+gradient_length,+gradient_context,+gradient_values,+gradient_room,+element,+element_inside,+kernel_positive,+channels_positive,+columns_positive,+group_positive} = scope
  +output = Nat.div(element,Nat.mul(c,Nat.mul(k,k)))
  +address = Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),output),rows),columns),position)
  +y = Nat.add(Nat.mod(Nat.div(element,k),k),Nat.mul(Nat.div(position,columns),s))
  +x = Nat.add(Nat.mod(element,k),Nat.mul(Nat.mod(position,columns),s))
  +inside = Nat.is_le(pad,y) && Nat.is_lt(y,Nat.add(pad,h)) && Nat.is_le(pad,x) && Nat.is_lt(x,Nat.add(pad,w))
  +sample = Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,c),Nat.mod(Nat.div(element,Nat.mul(k,k)),c)),h),w),Nat.mul(Nat.sub(y,pad),w)),Nat.sub(x,pad))
  +output_inside = DivisionOrder.quotient_below(element,o,Nat.mul(c,Nat.mul(k,k)),
    Order.product_positive(c,Nat.mul(k,k),channels_positive,Order.product_positive(k,k,kernel_positive,kernel_positive)),element_inside)
  +gradient_equal = Contract.read(~F32,~Gradient,~gradient_value,gradient_context,Buffer.Content{gradient_tree,gradient_length},address,gradient_values,
    Bounds.nat_trans(address,Nat.mul(Nat.mul(n,o),Nat.mul(rows,columns)),gradient_length,gradient_inside(n,o,rows,columns,image,output,position,image_inside,output_inside,position_inside),gradient_room))
  +sample_equal = sampled_agrees(~Input,~input_value,~Gradient,~gradient_value,inside,input_tree,input_length,input_context,input_values,sample,
    truth => Bounds.nat_trans(sample,Nat.mul(Nat.mul(n,c),Nat.mul(h,w)),input_length,
      input_inside(n,c,h,w,k,s,pad,columns,image,element,position,image_inside,channels_positive,truth),input_room))
  %Equal.sym(F32,Tree.read(F32,gradient_tree,U32.from_nat(address)),gradient_value(gradient_context,U32.from_nat(address)),gradient_equal) :
    {(_ * Spec.sampled(~Tree.Tree<F32>,~Bridge.at,inside,input_tree,sample) : F32) ==
      (gradient_value(gradient_context,U32.from_nat(address)) * Spec.sampled(~Operand<Input,Gradient>,~operand_at(~Input,~input_value,~Gradient,~gradient_value),inside,InputOperand{input_context},sample) : F32) : F32}
  Equal.cong(F32,F32,value => (gradient_value(gradient_context,U32.from_nat(address)) * value : F32),
    Spec.sampled(~Tree.Tree<F32>,~Bridge.at,inside,input_tree,sample),
    Spec.sampled(~Operand<Input,Gradient>,~operand_at(~Input,~input_value,~Gradient,~gradient_value),inside,InputOperand{input_context},sample),sample_equal)

# A weight element of storage that holds the readers' values is the
# readers' weight element.
def weight_agrees(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,
  +scope: WeightScope<Input,input_value,Gradient,gradient_value>) ->
  {Spec.weight(~Tree.Tree<F32>,~Bridge.at,weight_stored(~Input,~input_value,~Gradient,~gradient_value,scope),weight_element(~Input,~input_value,~Gradient,~gradient_value,scope)) ==
    Spec.weight(~Operand<Input,Gradient>,~operand_at(~Input,~input_value,~Gradient,~gradient_value),weight_read(~Input,~input_value,~Gradient,~gradient_value,scope),
      weight_element(~Input,~input_value,~Gradient,~gradient_value,scope)) : F32}:
  WeightScope{+n,+c,+o,+h,+w,+k,+s,+pad,+rows,+columns,+g,+input_tree,+input_length,+input_context,+input_values,+input_room,
    +gradient_tree,+gradient_length,+gradient_context,+gradient_values,+gradient_room,+element,+element_inside,+kernel_positive,+channels_positive,+columns_positive,+group_positive} = scope
  +scope = {WeightScope{n,c,o,h,w,k,s,pad,rows,columns,g,input_tree,input_length,input_context,input_values,input_room,
    gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,kernel_positive,channels_positive,columns_positive,group_positive} :
    WeightScope<Input,input_value,Gradient,gradient_value>}
  Equal.cong(List<&2,F32>,F32,values => Grouped.tree(values),
    Spec.image_values(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~Bridge.at),n,0n,element,weight_stored(~Input,~input_value,~Gradient,~gradient_value,scope),Grouped.ranges(rows,columns,g)),
    Spec.image_values(~Operand<Input,Gradient>,~Spec.weight_term(~Operand<Input,Gradient>,~operand_at(~Input,~input_value,~Gradient,~gradient_value)),n,0n,element,
      weight_read(~Input,~input_value,~Gradient,~gradient_value,scope),Grouped.ranges(rows,columns,g)),
    images_agree(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~Bridge.at),~Operand<Input,Gradient>,~Spec.weight_term(~Operand<Input,Gradient>,~operand_at(~Input,~input_value,~Gradient,~gradient_value)),
      ~WeightScope<Input,input_value,Gradient,gradient_value>,~weight_stored(~Input,~input_value,~Gradient,~gradient_value),~weight_read(~Input,~input_value,~Gradient,~gradient_value),
      ~weight_element(~Input,~input_value,~Gradient,~gradient_value),~weight_element(~Input,~input_value,~Gradient,~gradient_value),~weight_images(~Input,~input_value,~Gradient,~gradient_value),~weight_rows(~Input,~input_value,~Gradient,~gradient_value),
      ~weight_columns(~Input,~input_value,~Gradient,~gradient_value),~weight_group(~Input,~input_value,~Gradient,~gradient_value),
      ~weight_term_agrees(~Input,~input_value,~Gradient,~gradient_value),n,0n,scope,Order.reflexive(n),
      GroupedRanges.ranges_bounded(rows,columns,g,columns_positive,group_positive)))

# ---- The bias contract ----

type BiasScope<-Gradient: Data,-gradient_value: Gradient -> U32 -> F32> is Data:
  BiasScope{images: Nat,outputs: Nat,rows: Nat,columns: Nat,group: Nat,
    gradient_tree: Tree.Tree<F32>,gradient_length: Nat,gradient_context: Gradient,
    gradient_values: Contract.Values(~F32,~Gradient,~gradient_value,gradient_context,Buffer.Content{gradient_tree,gradient_length}),
    gradient_room: {Nat.is_le(Nat.mul(Nat.mul(images,outputs),Nat.mul(rows,columns)),gradient_length) == True{} : Bool},
    element: Nat,element_inside: {Nat.is_lt(element,outputs) == True{} : Bool},
    columns_positive: {Nat.is_lt(0n,columns) == True{} : Bool},group_positive: {Nat.is_lt(0n,group) == True{} : Bool}}

def bias_stored(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,scope: BiasScope<Gradient,gradient_value>) -> Spec.Problem<Tree.Tree<F32>>:
  BiasScope{n,o,rows,columns,g,gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,columns_positive,group_positive} = scope
  Spec.Problem{Spec.Shape{n,0n,o,0n,0n,1n,1n,0n,rows,columns,g},Tree.fresh(F32,0n,0.0),gradient_tree}

def bias_read(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,scope: BiasScope<Gradient,gradient_value>) -> Spec.Problem<Gradient>:
  BiasScope{n,o,rows,columns,g,gradient_tree,gradient_length,+gradient_context,gradient_values,gradient_room,element,element_inside,columns_positive,group_positive} = scope
  Spec.Problem{Spec.Shape{n,0n,o,0n,0n,1n,1n,0n,rows,columns,g},gradient_context,gradient_context}

def bias_element(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,scope: BiasScope<Gradient,gradient_value>) -> Nat:
  BiasScope{n,o,rows,columns,g,gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,columns_positive,group_positive} = scope
  element

def bias_images(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,scope: BiasScope<Gradient,gradient_value>) -> Nat:
  BiasScope{n,o,rows,columns,g,gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,columns_positive,group_positive} = scope
  n

def bias_rows(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,scope: BiasScope<Gradient,gradient_value>) -> Nat:
  BiasScope{n,o,rows,columns,g,gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,columns_positive,group_positive} = scope
  rows

def bias_columns(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,scope: BiasScope<Gradient,gradient_value>) -> Nat:
  BiasScope{n,o,rows,columns,g,gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,columns_positive,group_positive} = scope
  columns

def bias_group(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,scope: BiasScope<Gradient,gradient_value>) -> Nat:
  BiasScope{n,o,rows,columns,g,gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,columns_positive,group_positive} = scope
  g

def bias_term_agrees(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+scope: BiasScope<Gradient,gradient_value>,+image: Nat,+position: Nat,
  +image_inside: {Nat.is_lt(image,bias_images(~Gradient,~gradient_value,scope)) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,Nat.mul(bias_rows(~Gradient,~gradient_value,scope),bias_columns(~Gradient,~gradient_value,scope))) == True{} : Bool}) ->
  {Spec.bias_term(~Tree.Tree<F32>,~Bridge.at,bias_stored(~Gradient,~gradient_value,scope),image,bias_element(~Gradient,~gradient_value,scope),position) ==
    Spec.bias_term(~Gradient,~gradient_at(~Gradient,~gradient_value),bias_read(~Gradient,~gradient_value,scope),image,bias_element(~Gradient,~gradient_value,scope),position) : F32}:
  BiasScope{+n,+o,+rows,+columns,+g,+gradient_tree,+gradient_length,+gradient_context,+gradient_values,+gradient_room,+element,+element_inside,+columns_positive,+group_positive} = scope
  +address = Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),element),rows),columns),position)
  Contract.read(~F32,~Gradient,~gradient_value,gradient_context,Buffer.Content{gradient_tree,gradient_length},address,gradient_values,
    Bounds.nat_trans(address,Nat.mul(Nat.mul(n,o),Nat.mul(rows,columns)),gradient_length,gradient_inside(n,o,rows,columns,image,element,position,image_inside,element_inside,position_inside),gradient_room))

def bias_agrees(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+scope: BiasScope<Gradient,gradient_value>) ->
  {Spec.bias(~Tree.Tree<F32>,~Bridge.at,bias_stored(~Gradient,~gradient_value,scope),bias_element(~Gradient,~gradient_value,scope)) ==
    Spec.bias(~Gradient,~gradient_at(~Gradient,~gradient_value),bias_read(~Gradient,~gradient_value,scope),bias_element(~Gradient,~gradient_value,scope)) : F32}:
  BiasScope{+n,+o,+rows,+columns,+g,+gradient_tree,+gradient_length,+gradient_context,+gradient_values,+gradient_room,+element,+element_inside,+columns_positive,+group_positive} = scope
  +scope = {BiasScope{n,o,rows,columns,g,gradient_tree,gradient_length,gradient_context,gradient_values,gradient_room,element,element_inside,columns_positive,group_positive} :
    BiasScope<Gradient,gradient_value>}
  Equal.cong(List<&2,F32>,F32,values => Grouped.tree(values),
    Spec.image_values(~Tree.Tree<F32>,~Spec.bias_term(~Tree.Tree<F32>,~Bridge.at),n,0n,element,bias_stored(~Gradient,~gradient_value,scope),Grouped.ranges(rows,columns,g)),
    Spec.image_values(~Gradient,~Spec.bias_term(~Gradient,~gradient_at(~Gradient,~gradient_value)),n,0n,element,bias_read(~Gradient,~gradient_value,scope),Grouped.ranges(rows,columns,g)),
    images_agree(~Tree.Tree<F32>,~Spec.bias_term(~Tree.Tree<F32>,~Bridge.at),~Gradient,~Spec.bias_term(~Gradient,~gradient_at(~Gradient,~gradient_value)),
      ~BiasScope<Gradient,gradient_value>,~bias_stored(~Gradient,~gradient_value),~bias_read(~Gradient,~gradient_value),~bias_element(~Gradient,~gradient_value),~bias_element(~Gradient,~gradient_value),
      ~bias_images(~Gradient,~gradient_value),~bias_rows(~Gradient,~gradient_value),~bias_columns(~Gradient,~gradient_value),~bias_group(~Gradient,~gradient_value),
      ~bias_term_agrees(~Gradient,~gradient_value),n,0n,scope,Order.reflexive(n),
      GroupedRanges.ranges_bounded(rows,columns,g,columns_positive,group_positive)))

# ---- Extents of image shapes ----

def cells(image: Parameter.ImageGeometry) -> Nat:
  Parameter.ImageGeometry{n,c,h,w} = image
  Nat.mul(Nat.mul(U32.to_nat(n),U32.to_nat(c)),Nat.mul(U32.to_nat(h),U32.to_nat(w)))

def ImageCells(+shape: List<&2,U32>,result: Maybe<&2,Parameter.ImageGeometry>) -> Data:
  match result:
    case Some{image}: {ShapeLaws.elements(shape) == cells(image) : Nat}
    case None{}: Unit

def three_cells(+c: U32,+h: U32,+w: U32) -> {ShapeLaws.elements([c,h,w]) == cells(Parameter.ImageGeometry{1,c,h,w}) : Nat}:
  %Equal.sym(Nat,Nat.mul(U32.to_nat(w),1n),U32.to_nat(w),Order.times_one(U32.to_nat(w))) :
    {Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),_)) == cells(Parameter.ImageGeometry{1,c,h,w}) : Nat}
  %Equal.sym(Nat,Nat.add(U32.to_nat(c),0n),U32.to_nat(c),Addition.zero_right(U32.to_nat(c))) :
    {Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w))) == Nat.mul(_,Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat}
  {==}

def four_cells(+n: U32,+c: U32,+h: U32,+w: U32) -> {ShapeLaws.elements([n,c,h,w]) == cells(Parameter.ImageGeometry{n,c,h,w}) : Nat}:
  %Equal.sym(Nat,Nat.mul(U32.to_nat(w),1n),U32.to_nat(w),Order.times_one(U32.to_nat(w))) :
    {Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),_))) == cells(Parameter.ImageGeometry{n,c,h,w}) : Nat}
  Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(n),U32.to_nat(c)),Nat.mul(U32.to_nat(h),U32.to_nat(w))),Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w)))),
    RowMajor.assoc(U32.to_nat(n),U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w))))

# An image shape has as many elements as its images, channels, rows and
# columns together.
def image_cells(+shape: List<&2,U32>) -> ImageCells(shape,Parameter.image(shape)):
  match shape:
    case Nil{}: Unit{}
    case Con{+first,+after_first}:
      match after_first:
        case Nil{}: Unit{}
        case Con{+second,+after_second}:
          match after_second:
            case Nil{}: Unit{}
            case Con{+third,+after_third}:
              match after_third:
                case Nil{}: three_cells(first,second,third)
                case Con{+fourth,+after_fourth}:
                  match after_fourth:
                    case Nil{}: four_cells(first,second,third,fourth)
                    case Con{+fifth,+rest}: Unit{}
