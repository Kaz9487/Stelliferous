# Closed bridge from native readers to the independently enumerated Nat terms.
# No source-read agreement is requested from public callers.
import Base
import ../parameter_gradient_specification.bend as Spec
import ../specification.bend as ConvSpec
import ../../lib/kernel_parameter_gradient.bend as Kernel
import ../../lib/traversal_grouped.bend as Grouped
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/grouped_reduction_refinement.bend as Reduction
import ../../lib/proofs/parameter_gradient_refinement.bend as Model
import ../../lib/proofs/list_storage_readback.bend as ListRead

import ../../lib/kernel_weight_gradient.bend as WeightKernel
import ../../lib/proofs/kernel_weight_gradient_group.bend as WeightGroups
import ../../lib/proofs/kernel_weight_gradient_packing.bend as WeightPacking
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/u32_full_range.bend as FullRange
import ../../lib/proofs/boolean_logic.bend as Logic
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/traversal_array_refinement.bend as Traversals
import ../../lib/proofs/ordered_reduction_split.bend as Ordered
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/kernel_weight_gradient_refinement.bend as WeightModel

def shape(work: Kernel.Work) -> Spec.Shape:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Spec.Shape{n,c,o,h,w,k,s,p,rows,columns,g}

def at(source: Tree.Tree<F32>,index: Nat) -> F32:
  Tree.read(F32,source,U32.from_nat(index))

def weight_problem(work: Kernel.Work,source: Model.Source) -> Spec.Problem<Tree.Tree<F32>>:
  Model.Source{input,gradient} = source
  Spec.Problem{shape(work),input,gradient}

def bias_problem(work: Kernel.Work,source: Tree.Tree<F32>) -> Spec.Problem<Tree.Tree<F32>>:
  Spec.Problem{shape(work),Tree.fresh(F32,0n,0.0),source}

def sampled(inside: Bool,+source: Tree.Tree<F32>,+index: Nat) -> {Bool.pick(F32,inside,at(source,index),0.0) == Spec.sampled(~Tree.Tree<F32>,~at,inside,source,index) : F32}:
  match inside:
    case False{}: {==}
    case True{}: {==}

def bias_term(+source: Tree.Tree<F32>,+work: Kernel.Work,+image: Nat,+element: Nat,+position: Nat) -> {Model.bias_sample(source,Kernel.Context{work,image,element},position) == Spec.bias_term(~Tree.Tree<F32>,~at,bias_problem(work,source),image,element,position) : F32}:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  {==}

def fold(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,
  ~problem: Kernel.Work -> Source -> Spec.Problem<Tree.Tree<F32>>,
  ~term: @p: Spec.Problem<Tree.Tree<F32>> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  ~matches: @+source: Source -> @+work: Kernel.Work -> @+image: Nat -> @+element: Nat -> @+position: Nat -> {sample(source,Kernel.Context{work,image,element},position) == term(problem(work,source),image,element,position) : F32},remaining: Nat,+position: Nat,+source: Source,+work: Kernel.Work,+image: Nat,+element: Nat,+total: F32) ->
  {Reduction.values(~Source,~Kernel.Context,~sample,remaining,position,source,Kernel.Context{work,image,element},total) == ConvSpec.sum(Spec.terms(~Tree.Tree<F32>,~term,remaining,position,problem(work,source),image,element),total) : F32}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(F32,sample(source,Kernel.Context{work,image,element},position),term(problem(work,source),image,element,position),matches(source,work,image,element,position)) :
        {Reduction.values(~Source,~Kernel.Context,~sample,rest,1n+position,source,Kernel.Context{work,image,element},(total + _ : F32)) == ConvSpec.sum(Spec.terms(~Tree.Tree<F32>,~term,rest,1n+position,problem(work,source),image,element),(total + term(problem(work,source),image,element,position) : F32)) : F32}
      fold(~Source,~sample,~problem,~term,~matches,rest,1n+position,source,work,image,element,(total + term(problem(work,source),image,element,position) : F32))

def group_lists(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,
  ~problem: Kernel.Work -> Source -> Spec.Problem<Tree.Tree<F32>>,
  ~term: @p: Spec.Problem<Tree.Tree<F32>> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  ~matches: @+source: Source -> @+work: Kernel.Work -> @+image: Nat -> @+element: Nat -> @+position: Nat -> {sample(source,Kernel.Context{work,image,element},position) == term(problem(work,source),image,element,position) : F32},groups: List<&2,Grouped.Range>,+source: Source,+work: Kernel.Work,+image: Nat,+element: Nat) ->
  {Reduction.group_values(~Source,~Kernel.Context,~sample,groups,source,Kernel.Context{work,image,element}) == Spec.group_values(~Tree.Tree<F32>,~term,groups,problem(work,source),image,element) : List<&2,F32>}:
  match groups:
    case []: {==}
    case Grouped.Range{+first,+count} <> +rest:
      %fold(~Source,~sample,~problem,~term,~matches,count,first,source,work,image,element,0.0) :
        {Reduction.values(~Source,~Kernel.Context,~sample,count,first,source,Kernel.Context{work,image,element},0.0) <> Reduction.group_values(~Source,~Kernel.Context,~sample,rest,source,Kernel.Context{work,image,element}) == _ <> Spec.group_values(~Tree.Tree<F32>,~term,rest,problem(work,source),image,element) : List<&2,F32>}
      %group_lists(~Source,~sample,~problem,~term,~matches,rest,source,work,image,element) :
        {Reduction.values(~Source,~Kernel.Context,~sample,count,first,source,Kernel.Context{work,image,element},0.0) <> Reduction.group_values(~Source,~Kernel.Context,~sample,rest,source,Kernel.Context{work,image,element}) == Reduction.values(~Source,~Kernel.Context,~sample,count,first,source,Kernel.Context{work,image,element},0.0) <> _ : List<&2,F32>}
      {==}

def image_value(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,
  ~problem: Kernel.Work -> Source -> Spec.Problem<Tree.Tree<F32>>,
  ~term: @p: Spec.Problem<Tree.Tree<F32>> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  ~matches: @+source: Source -> @+work: Kernel.Work -> @+image: Nat -> @+element: Nat -> @+position: Nat -> {sample(source,Kernel.Context{work,image,element},position) == term(problem(work,source),image,element,position) : F32},+groups: List<&2,Grouped.Range>,+source: Source,+work: Kernel.Work,+image: Nat,+element: Nat) ->
  {Grouped.tree(Reduction.group_values(~Source,~Kernel.Context,~sample,groups,source,Kernel.Context{work,image,element})) == Grouped.tree(Spec.group_values(~Tree.Tree<F32>,~term,groups,problem(work,source),image,element)) : F32}:
  Equal.cong(List<&2,F32>,F32,values => Grouped.tree(values),Reduction.group_values(~Source,~Kernel.Context,~sample,groups,source,Kernel.Context{work,image,element}),Spec.group_values(~Tree.Tree<F32>,~term,groups,problem(work,source),image,element),group_lists(~Source,~sample,~problem,~term,~matches,groups,source,work,image,element))

def images(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,
  ~problem: Kernel.Work -> Source -> Spec.Problem<Tree.Tree<F32>>,
  ~term: @p: Spec.Problem<Tree.Tree<F32>> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  ~matches: @+source: Source -> @+work: Kernel.Work -> @+image: Nat -> @+element: Nat -> @+position: Nat -> {sample(source,Kernel.Context{work,image,element},position) == term(problem(work,source),image,element,position) : F32},remaining: Nat,+image: Nat,+element: Nat,+work: Kernel.Work,+groups: List<&2,Grouped.Range>,+source: Source) ->
  {Model.image_values(~Source,~sample,remaining,image,element,work,groups,source) == Spec.image_values(~Tree.Tree<F32>,~term,remaining,image,element,problem(work,source),groups) : List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %image_value(~Source,~sample,~problem,~term,~matches,groups,source,work,image,element) :
        {Grouped.tree(Reduction.group_values(~Source,~Kernel.Context,~sample,groups,source,Kernel.Context{work,image,element})) <> Model.image_values(~Source,~sample,rest,1n+image,element,work,groups,source) == _ <> Spec.image_values(~Tree.Tree<F32>,~term,rest,1n+image,element,problem(work,source),groups) : List<&2,F32>}
      %images(~Source,~sample,~problem,~term,~matches,rest,1n+image,element,work,groups,source) :
        {Grouped.tree(Reduction.group_values(~Source,~Kernel.Context,~sample,groups,source,Kernel.Context{work,image,element})) <> Model.image_values(~Source,~sample,rest,1n+image,element,work,groups,source) == Grouped.tree(Reduction.group_values(~Source,~Kernel.Context,~sample,groups,source,Kernel.Context{work,image,element})) <> _ : List<&2,F32>}
      {==}

def rows_of(work: Kernel.Work) -> Nat:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  rows

def columns_of(work: Kernel.Work) -> Nat:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  columns

def group_of(work: Kernel.Work) -> Nat:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  g


def enumerate(~Context: Data,~value: Context -> Nat -> F32,remaining: Nat,first: Nat,context: Context) -> List<&2,F32>:
  ListRead.enumerate(~F32,~Context,~value,remaining,first,context)

def enumerate_length(~Context: Data,~value: Context -> Nat -> F32,+remaining: Nat,+first: Nat,+context: Context) -> {List.length(&2,F32,enumerate(~Context,~value,remaining,first,context)) == remaining : Nat}:
  ListRead.enumerate_length(~F32,~Context,~value,remaining,first,context)

def enumerate_at(~Context: Data,~value: Context -> Nat -> F32,+remaining: Nat,+index: Nat,+first: Nat,+context: Context,+within: {Nat.is_lt(index,remaining) == True{} : Bool}) ->
  {ListRead.at(~F32,0.0,index,enumerate(~Context,~value,remaining,first,context)) == value(context,Nat.add(first,index)) : F32}:
  ListRead.enumerate_at(~F32,~Context,~value,0.0,remaining,index,first,context,within)

def bias_element(+work: Kernel.Work,+source: Tree.Tree<F32>,+element: Nat) ->
  {Model.element(~Tree.Tree<F32>,~Model.bias_sample,work,element,Grouped.ranges(rows_of(work),columns_of(work),group_of(work)),source) == Spec.bias(~Tree.Tree<F32>,~at,bias_problem(work,source),element) : F32}:
  Kernel.Work{+n,c,o,h,w,k,s,p,+rows,+columns,+g} = work
  Equal.cong(List<&2,F32>,F32,values => Grouped.tree(values),Model.image_values(~Tree.Tree<F32>,~Model.bias_sample,n,0n,element,work,Grouped.ranges(rows,columns,g),source),Spec.image_values(~Tree.Tree<F32>,~Spec.bias_term(~Tree.Tree<F32>,~at),n,0n,element,bias_problem(work,source),Grouped.ranges(rows,columns,g)),images(~Tree.Tree<F32>,~Model.bias_sample,~bias_problem,~Spec.bias_term(~Tree.Tree<F32>,~at),~bias_term,n,0n,element,work,Grouped.ranges(rows,columns,g),source))

def bias_list(remaining: Nat,+index: Nat,+work: Kernel.Work,+source: Tree.Tree<F32>) -> List<&2,F32>:
  enumerate(~Spec.Problem<Tree.Tree<F32>>,~Spec.bias(~Tree.Tree<F32>,~at),remaining,index,bias_problem(work,source))

def bias_output(remaining: Nat,+index: Nat,+work: Kernel.Work,+source: Tree.Tree<F32>) ->
  {Model.values(~Tree.Tree<F32>,~Model.bias_sample,remaining,index,work,Grouped.ranges(rows_of(work),columns_of(work),group_of(work)),source) == bias_list(remaining,index,work,source) : List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %bias_element(work,source,index) : {Model.element(~Tree.Tree<F32>,~Model.bias_sample,work,index,Grouped.ranges(rows_of(work),columns_of(work),group_of(work)),source) <> Model.values(~Tree.Tree<F32>,~Model.bias_sample,rest,1n+index,work,Grouped.ranges(rows_of(work),columns_of(work),group_of(work)),source) == _ <> bias_list(rest,1n+index,work,source) : List<&2,F32>}
      %bias_output(rest,1n+index,work,source) : {Model.element(~Tree.Tree<F32>,~Model.bias_sample,work,index,Grouped.ranges(rows_of(work),columns_of(work),group_of(work)),source) <> Model.values(~Tree.Tree<F32>,~Model.bias_sample,rest,1n+index,work,Grouped.ranges(rows_of(work),columns_of(work),group_of(work)),source) == Model.element(~Tree.Tree<F32>,~Model.bias_sample,work,index,Grouped.ranges(rows_of(work),columns_of(work),group_of(work)),source) <> _ : List<&2,F32>}
      {==}

# Decoded coordinates connect the packed reader to each specification term.
def coordinate_term(+p: Spec.Problem<Tree.Tree<F32>>,+image: Nat,+output: Nat,+channel: Nat,+kr: Nat,+kc: Nat,+row: Nat,+column: Nat,+position: Nat) -> F32:
  Spec.Problem{Spec.Shape{n,+c,+o,+h,+w,k,+s,+pad,+rows,+columns,g},input,gradient} = p
  +y = Nat.add(kr,Nat.mul(row,s))
  +x = Nat.add(kc,Nat.mul(column,s))
  (at(gradient,Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),output),rows),columns),position)) *
    Spec.sampled(~Tree.Tree<F32>,~at,Nat.is_le(pad,y) && Nat.is_lt(y,Nat.add(pad,h)) && Nat.is_le(pad,x) && Nat.is_lt(x,Nat.add(pad,w)),input,
      Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,c),channel),h),w),Nat.mul(Nat.sub(y,pad),w)),Nat.sub(x,pad))) : F32)

def packed_problem(source: WeightGroups.NumericSource,images: Nat,group: Nat) -> Spec.Problem<Tree.Tree<F32>>:
  WeightGroups.NumericSource{input,gradient,WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},image} = source
  Spec.Problem{Spec.Shape{images,U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p),U32.to_nat(rows),U32.to_nat(columns),group},Buffer.tree(~F32,input),Buffer.tree(~F32,gradient)}

def packed_geometry(source: WeightGroups.NumericSource) -> WeightKernel.Geometry:
  WeightGroups.NumericSource{input,gradient,geometry,image} = source
  geometry

def packed_image(source: WeightGroups.NumericSource) -> U32:
  WeightGroups.NumericSource{input,gradient,geometry,image} = source
  image

def packed_weight_coordinates(+source: WeightGroups.NumericSource,+images: Nat,+group: Nat,+tap: U32,+output: U32,+position: Nat,
  +row_exact: {U32.to_nat(U32.from_nat(Nat.div(position,U32.to_nat(WeightGroups.geometry_columns(packed_geometry(source)))))) == Nat.div(position,U32.to_nat(WeightGroups.geometry_columns(packed_geometry(source)))) : Nat},
  +column_exact: {U32.to_nat(U32.from_nat(Nat.mod(position,U32.to_nat(WeightGroups.geometry_columns(packed_geometry(source)))))) == Nat.mod(position,U32.to_nat(WeightGroups.geometry_columns(packed_geometry(source)))) : Nat}) ->
  {(WeightGroups.numeric_input(source,output,position) * WeightGroups.numeric_weight(source,tap,position) : F32) ==
    coordinate_term(packed_problem(source,images,group),U32.to_nat(packed_image(source)),U32.to_nat(output),Nat.div(U32.to_nat(tap),Nat.mul(U32.to_nat(WeightPacking.geometry_kernel(packed_geometry(source))),U32.to_nat(WeightPacking.geometry_kernel(packed_geometry(source))))),Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(WeightPacking.geometry_kernel(packed_geometry(source)))),U32.to_nat(WeightPacking.geometry_kernel(packed_geometry(source)))),Nat.mod(U32.to_nat(tap),U32.to_nat(WeightPacking.geometry_kernel(packed_geometry(source)))),Nat.div(position,U32.to_nat(WeightGroups.geometry_columns(packed_geometry(source)))),Nat.mod(position,U32.to_nat(WeightGroups.geometry_columns(packed_geometry(source)))),position) : F32}:
  WeightGroups.NumericSource{+input,+gradient,+geometry,+image} = source
  WeightKernel.Geometry{+c,+o,+h,+w,+k,+s,+pad,+rows,+columns,taps} = geometry
  +ch = Nat.div(U32.to_nat(tap),Nat.mul(U32.to_nat(k),U32.to_nat(k)))
  +kr = Nat.mod(Nat.div(U32.to_nat(tap),U32.to_nat(k)),U32.to_nat(k))
  +kc = Nat.mod(U32.to_nat(tap),U32.to_nat(k))
  +row = Nat.div(position,U32.to_nat(columns))
  +col = Nat.mod(position,U32.to_nat(columns))
  +base = WeightPacking.tap_base(geometry,image,tap)
  +y = Nat.add(kr,Nat.mul(row,U32.to_nat(s)))
  +x = Nat.add(kc,Nat.mul(col,U32.to_nat(s)))
  +a = Nat.is_le(U32.to_nat(pad),y)
  +b = Nat.is_lt(y,Nat.add(U32.to_nat(pad),U32.to_nat(h)))
  +d = Nat.is_le(U32.to_nat(pad),x)
  +e = Nat.is_lt(x,Nat.add(U32.to_nat(pad),U32.to_nat(w)))
  +address = Nat.add(Nat.add(base,Nat.mul(Nat.sub(y,U32.to_nat(pad)),U32.to_nat(w))),Nat.sub(x,U32.to_nat(pad)))
  +value = WeightGroups.numeric_input(source,output,position)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(row)),row,row_exact) :
    {(value * WeightPacking.sampled_pixel(Buffer.tree(~F32,input),base,_,U32.to_nat(U32.from_nat(col)),kr,kc,U32.to_nat(h),U32.to_nat(w),U32.to_nat(s),U32.to_nat(pad)) : F32) == coordinate_term(packed_problem(source,images,group),U32.to_nat(image),U32.to_nat(output),ch,kr,kc,row,col,position) : F32}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(col)),col,column_exact) :
    {(value * WeightPacking.sampled_pixel(Buffer.tree(~F32,input),base,row,_,kr,kc,U32.to_nat(h),U32.to_nat(w),U32.to_nat(s),U32.to_nat(pad)) : F32) == coordinate_term(packed_problem(source,images,group),U32.to_nat(image),U32.to_nat(output),ch,kr,kc,row,col,position) : F32}
  %Logic.and_associative((a && b),d,e) :
    {(value * Bool.pick(F32,_,at(Buffer.tree(~F32,input),address),0.0) : F32) == coordinate_term(packed_problem(source,images,group),U32.to_nat(image),U32.to_nat(output),ch,kr,kc,row,col,position) : F32}
  Equal.cong(F32,F32,sample => (value * sample : F32),Bool.pick(F32,((a && b) && d) && e,at(Buffer.tree(~F32,input),address),0.0),Spec.sampled(~Tree.Tree<F32>,~at,((a && b) && d) && e,Buffer.tree(~F32,input),address),sampled(((a && b) && d) && e,Buffer.tree(~F32,input),address))

def packed_channels(source: WeightGroups.NumericSource) -> U32:
  WeightGroups.NumericSource{input,gradient,WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps},image} = source
  c

def PackedLeaf(geometry: WeightKernel.Geometry,+tap: U32) -> Data:
  WeightKernel.Geometry{c,o,h,w,+k,s,p,rows,columns,taps} = geometry
  Shape.Both<{Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},
    Shape.Both<{Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool},
      Shape.Both<{Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},
        {Nat.is_lt(U32.to_nat(tap),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) == True{} : Bool}>>>

def packed_weight_term(+source: WeightGroups.NumericSource,+images: Nat,+group: Nat,+tap: U32,+output: U32,+position: Nat,
  domain: PackedLeaf(packed_geometry(source),tap),
  +position_inside: {Nat.is_lt(position,Nat.mul(U32.to_nat(WeightGroups.geometry_rows(packed_geometry(source))),U32.to_nat(WeightGroups.geometry_columns(packed_geometry(source))))) == True{} : Bool}) ->
  {(WeightGroups.numeric_input(source,output,position) * WeightGroups.numeric_weight(source,tap,position) : F32) ==
    Spec.weight_term(~Tree.Tree<F32>,~at,packed_problem(source,images,group),U32.to_nat(packed_image(source)),Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(packed_channels(source)),Nat.mul(U32.to_nat(WeightPacking.geometry_kernel(packed_geometry(source))),U32.to_nat(WeightPacking.geometry_kernel(packed_geometry(source)))))),U32.to_nat(tap)),position) : F32}:
  WeightGroups.NumericSource{+input,+gradient,+geometry,+image} = source
  WeightKernel.Geometry{+c,o,h,w,+k,s,p,+rows,+columns,taps} = geometry
  Shape.Both{+cp,Shape.Both{+kp,Shape.Both{+colp,+tip}}} = domain
  +square_positive = Order.product_positive(U32.to_nat(k),U32.to_nat(k),kp,kp)
  +taps_positive = Order.product_positive(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)),cp,square_positive)
  +cn = U32.to_nat(c)
  +kn = U32.to_nat(k)
  +tn = U32.to_nat(tap)
  +on = U32.to_nat(output)
  +element = Nat.add(Nat.mul(on,Nat.mul(cn,Nat.mul(kn,kn))),tn)
  +row = Nat.div(position,U32.to_nat(columns))
  +col = Nat.mod(position,U32.to_nat(columns))
  +problem = packed_problem(source,images,group)
  +value = (WeightGroups.numeric_input(source,output,position) * WeightGroups.numeric_weight(source,tap,position) : F32)
  %Equal.sym(Nat,Nat.div(element,Nat.mul(cn,Nat.mul(kn,kn))),on,WeightPacking.weight_output_decode(on,cn,kn,tn,taps_positive,tip)) :
    {value == coordinate_term(problem,U32.to_nat(image),_,Nat.mod(Nat.div(element,Nat.mul(kn,kn)),cn),Nat.mod(Nat.div(element,kn),kn),Nat.mod(element,kn),row,col,position) : F32}
  %Equal.sym(Nat,Nat.mod(Nat.div(element,Nat.mul(kn,kn)),cn),Nat.div(tn,Nat.mul(kn,kn)),WeightPacking.weight_channel_decode(on,cn,kn,tn,cp,square_positive,tip)) :
    {value == coordinate_term(problem,U32.to_nat(image),on,_,Nat.mod(Nat.div(element,kn),kn),Nat.mod(element,kn),row,col,position) : F32}
  %Equal.sym(Nat,Nat.mod(Nat.div(element,kn),kn),Nat.mod(Nat.div(tn,kn),kn),WeightPacking.weight_row_decode(on,cn,kn,tn,kp)) :
    {value == coordinate_term(problem,U32.to_nat(image),on,Nat.div(tn,Nat.mul(kn,kn)),_,Nat.mod(element,kn),row,col,position) : F32}
  %Equal.sym(Nat,Nat.mod(element,kn),Nat.mod(tn,kn),WeightPacking.weight_column_decode(on,cn,kn,tn,kp)) :
    {value == coordinate_term(problem,U32.to_nat(image),on,Nat.div(tn,Nat.mul(kn,kn)),Nat.mod(Nat.div(tn,kn),kn),_,row,col,position) : F32}
  packed_weight_coordinates(source,images,group,tap,output,position,
    FullRange.roundtrip(row,rows,Bounds.strict_le(row,U32.to_nat(rows),DivisionOrder.quotient_below(position,U32.to_nat(rows),U32.to_nat(columns),colp,position_inside))),
    FullRange.roundtrip(col,columns,Bounds.strict_le(col,U32.to_nat(columns),DivisionOrder.remainder_small(position,U32.to_nat(columns),colp))))

type TermsQuery<-Source: Data> is Data:
  TermsQuery{problem: Spec.Problem<Source>,image: Nat,element: Nat}

def term_query(~Source: Data,~term: @p: Spec.Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  query: TermsQuery<Source>,position: Nat) -> F32:
  TermsQuery{problem,image,element} = query
  term(problem,image,element,position)

def terms_problem(~Source: Data,query: TermsQuery<Source>) -> Spec.Problem<Source>:
  TermsQuery{problem,image,element} = query
  problem

def terms_image(~Source: Data,query: TermsQuery<Source>) -> Nat:
  TermsQuery{problem,image,element} = query
  image

def terms_element(~Source: Data,query: TermsQuery<Source>) -> Nat:
  TermsQuery{problem,image,element} = query
  element

def terms_enumerated(~Source: Data,~term: @p: Spec.Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  +query: TermsQuery<Source>,count: Nat,+first: Nat) ->
  {Spec.terms(~Source,~term,count,first,terms_problem(~Source,query),terms_image(~Source,query),terms_element(~Source,query)) ==
    ListRead.enumerate(~F32,~TermsQuery<Source>,~term_query(~Source,~term),count,first,query) : List<&2,F32>}:
  TermsQuery{+problem,+image,+element} = query
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,F32>,List<&2,F32>,values => term(problem,image,element,first) <> values,
        Spec.terms(~Source,~term,rest,1n+first,problem,image,element),ListRead.enumerate(~F32,~TermsQuery<Source>,~term_query(~Source,~term),rest,1n+first,query),terms_enumerated(~Source,~term,query,rest,1n+first))

def sum_fold(values: List<&2,F32>,+initial: F32) ->
  {ConvSpec.sum(values,initial) == List.foldl(~&2,~F32,~F32,~F32.add,values,initial) : F32}:
  match values:
    case Nil{}: {==}
    case value <> rest: sum_fold(rest,(initial + value : F32))

type PackedTerms is Data:
  PackedTerms{source: WeightGroups.NumericSource,images: Nat,group: Nat,tap: U32,output: U32,domain: PackedLeaf(packed_geometry(source),tap)}

def packed_terms_query(query: PackedTerms) -> TermsQuery<Tree.Tree<F32>>:
  PackedTerms{+source,images,group,tap,output,domain} = query
  +k = U32.to_nat(WeightPacking.geometry_kernel(packed_geometry(source)))
  TermsQuery{packed_problem(source,images,group),U32.to_nat(packed_image(source)),Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(packed_channels(source)),Nat.mul(k,k))),U32.to_nat(tap))}

def packed_terms_size(query: PackedTerms) -> Nat:
  PackedTerms{+source,images,group,tap,output,domain} = query
  Nat.mul(U32.to_nat(WeightGroups.geometry_rows(packed_geometry(source))),U32.to_nat(WeightGroups.geometry_columns(packed_geometry(source))))

def PackedTermMatches(+query: PackedTerms,+position: Nat) -> Data:
  PackedTerms{+source,images,group,tap,output,domain} = query
  {(WeightGroups.numeric_input(source,output,position) * WeightGroups.numeric_weight(source,tap,position) : F32) ==
    term_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),packed_terms_query(query),position) : F32}

def packed_term_matches(+query: PackedTerms,+position: Nat,+inside: {Nat.is_lt(position,packed_terms_size(query)) == True{} : Bool}) -> PackedTermMatches(query,position):
  PackedTerms{+source,+images,+group,+tap,+output,domain} = query
  packed_weight_term(source,images,group,tap,output,position,domain,inside)

def packed_terms_dot(query: PackedTerms,count: Nat,first: Nat,initial: F32) -> F32:
  PackedTerms{source,images,group,tap,output,domain} = query
  WeightGroups.numeric_dot(source,tap,output,count,first,initial)

def packed_group_terms(+query: PackedTerms,+count: Nat,+first: Nat,+initial: F32,
  +room: {Nat.is_le(Nat.add(first,count),packed_terms_size(query)) == True{} : Bool}) ->
  {packed_terms_dot(query,count,first,initial) ==
    ConvSpec.sum(Spec.terms(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),count,first,
      terms_problem(~Tree.Tree<F32>,packed_terms_query(query)),terms_image(~Tree.Tree<F32>,packed_terms_query(query)),terms_element(~Tree.Tree<F32>,packed_terms_query(query))),initial) : F32}:
  PackedTerms{+source,images,group,+tap,+output,domain} = query
  +target = packed_terms_query(query)
  +agreement = Members.range_all(~PackedTerms,~PackedTermMatches,~packed_terms_size,~packed_term_matches,count,first,query,room)
  %Equal.sym(List<&2,F32>,Spec.terms(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),count,first,terms_problem(~Tree.Tree<F32>,target),terms_image(~Tree.Tree<F32>,target),terms_element(~Tree.Tree<F32>,target)),ListRead.enumerate(~F32,~TermsQuery<Tree.Tree<F32>>,~term_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),count,first,target),terms_enumerated(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),target,count,first)) :
    {WeightGroups.numeric_dot(source,tap,output,count,first,initial) == ConvSpec.sum(_,initial) : F32}
  %Traversals.collect_values_context_range(~F32,~TermsQuery<Tree.Tree<F32>>,~term_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),count,first,target) :
    {WeightGroups.numeric_dot(source,tap,output,count,first,initial) == ConvSpec.sum(_,initial) : F32}
  Equal.trans(F32,WeightGroups.numeric_dot(source,tap,output,count,first,initial),
    List.foldl(~&2,~F32,~F32,~F32.add,Traversals.collect_values_context(~&2,~F32,~Nat,~TermsQuery<Tree.Tree<F32>>,~term_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),Ranges.range(count,first),target),initial),
    ConvSpec.sum(Traversals.collect_values_context(~&2,~F32,~Nat,~TermsQuery<Tree.Tree<F32>>,~term_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),Ranges.range(count,first),target),initial),
    Ordered.dot_fold_agrees(~Nat,~U32,~U32,~WeightGroups.NumericSource,~TermsQuery<Tree.Tree<F32>>,~WeightGroups.numeric_input,~WeightGroups.numeric_weight,~term_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),Ranges.range(count,first),tap,output,source,target,initial,agreement),
    Equal.sym(F32,ConvSpec.sum(Traversals.collect_values_context(~&2,~F32,~Nat,~TermsQuery<Tree.Tree<F32>>,~term_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),Ranges.range(count,first),target),initial),List.foldl(~&2,~F32,~F32,~F32.add,Traversals.collect_values_context(~&2,~F32,~Nat,~TermsQuery<Tree.Tree<F32>>,~term_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),Ranges.range(count,first),target),initial),sum_fold(Traversals.collect_values_context(~&2,~F32,~Nat,~TermsQuery<Tree.Tree<F32>>,~term_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),Ranges.range(count,first),target),initial)))

def terms_group(~Source: Data,~term: @p: Spec.Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  query: TermsQuery<Source>,range: Grouped.Range) -> F32:
  Grouped.Range{first,count} = range
  +query = query
  ConvSpec.sum(Spec.terms(~Source,~term,count,first,terms_problem(~Source,query),terms_image(~Source,query),terms_element(~Source,query)),0.0)

def groups_mapped(~Source: Data,~term: @p: Spec.Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  +query: TermsQuery<Source>,groups: List<&2,Grouped.Range>) ->
  {Spec.group_values(~Source,~term,groups,terms_problem(~Source,query),terms_image(~Source,query),terms_element(~Source,query)) ==
    Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~TermsQuery<Source>,~terms_group(~Source,~term),groups,query) : List<&2,F32>}:
  TermsQuery{+problem,+image,+element} = query
  match groups:
    case Nil{}: {==}
    case Grouped.Range{+first,+count} <> +rest:
      Equal.cong(List<&2,F32>,List<&2,F32>,values => ConvSpec.sum(Spec.terms(~Source,~term,count,first,problem,image,element),0.0) <> values,
        Spec.group_values(~Source,~term,rest,problem,image,element),Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~TermsQuery<Source>,~terms_group(~Source,~term),rest,query),groups_mapped(~Source,~term,query,rest))

def packed_group_value(query: PackedTerms,range: Grouped.Range) -> F32:
  Grouped.Range{first,count} = range
  packed_terms_dot(query,count,first,0.0)

def PackedGroupRoom(+query: PackedTerms,+range: Grouped.Range) -> Data:
  Grouped.Range{first,count} = range
  {Nat.is_le(Nat.add(first,count),packed_terms_size(query)) == True{} : Bool}

def PackedGroupMatches(+query: PackedTerms,+range: Grouped.Range) -> Data:
  {packed_group_value(query,range) == terms_group(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),packed_terms_query(query),range) : F32}

def packed_group_matches(+query: PackedTerms,+range: Grouped.Range,+room: PackedGroupRoom(query,range)) -> PackedGroupMatches(query,range):
  Grouped.Range{+first,+count} = range
  packed_group_terms(query,count,first,0.0,room)

def packed_groups_spec(+query: PackedTerms,+groups: List<&2,Grouped.Range>,
  rooms: Members.All(Grouped.Range,{range => PackedGroupRoom(query,range) : Grouped.Range -> Data},groups)) ->
  {Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~PackedTerms,~packed_group_value,groups,query) ==
    Spec.group_values(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),groups,terms_problem(~Tree.Tree<F32>,packed_terms_query(query)),terms_image(~Tree.Tree<F32>,packed_terms_query(query)),terms_element(~Tree.Tree<F32>,packed_terms_query(query))) : List<&2,F32>}:
  +target = packed_terms_query(query)
  +agreements = Members.transform_all(~Grouped.Range,~PackedTerms,~PackedGroupRoom,~PackedGroupMatches,~packed_group_matches,groups,query,rooms)
  %Equal.sym(List<&2,F32>,Spec.group_values(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),groups,terms_problem(~Tree.Tree<F32>,target),terms_image(~Tree.Tree<F32>,target),terms_element(~Tree.Tree<F32>,target)),Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~TermsQuery<Tree.Tree<F32>>,~terms_group(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),groups,target),groups_mapped(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),target,groups)) :
    {Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~PackedTerms,~packed_group_value,groups,query) == _ : List<&2,F32>}
  Traversals.collect_values_context_agrees(~F32,~Grouped.Range,~PackedTerms,~TermsQuery<Tree.Tree<F32>>,~packed_group_value,~terms_group(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),groups,query,target,agreements)

type ImagesQuery<-Source: Data> is Data:
  ImagesQuery{problem: Spec.Problem<Source>,element: Nat,groups: List<&2,Grouped.Range>}

def image_query(~Source: Data,~term: @p: Spec.Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  query: ImagesQuery<Source>,image: Nat) -> F32:
  ImagesQuery{problem,element,groups} = query
  Grouped.tree(Spec.group_values(~Source,~term,groups,problem,image,element))

def images_problem(~Source: Data,query: ImagesQuery<Source>) -> Spec.Problem<Source>:
  ImagesQuery{problem,element,groups} = query
  problem

def images_element(~Source: Data,query: ImagesQuery<Source>) -> Nat:
  ImagesQuery{problem,element,groups} = query
  element

def images_groups(~Source: Data,query: ImagesQuery<Source>) -> List<&2,Grouped.Range>:
  ImagesQuery{problem,element,groups} = query
  groups

def images_enumerated(~Source: Data,~term: @p: Spec.Problem<Source> -> @+image: Nat -> @+element: Nat -> @+position: Nat -> F32,
  +query: ImagesQuery<Source>,count: Nat,+first: Nat) ->
  {Spec.image_values(~Source,~term,count,first,images_element(~Source,query),images_problem(~Source,query),images_groups(~Source,query)) ==
    ListRead.enumerate(~F32,~ImagesQuery<Source>,~image_query(~Source,~term),count,first,query) : List<&2,F32>}:
  ImagesQuery{+problem,+element,+groups} = query
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,F32>,List<&2,F32>,values => Grouped.tree(Spec.group_values(~Source,~term,groups,problem,first,element)) <> values,
        Spec.image_values(~Source,~term,rest,1n+first,element,problem,groups),ListRead.enumerate(~F32,~ImagesQuery<Source>,~image_query(~Source,~term),rest,1n+first,query),images_enumerated(~Source,~term,query,rest,1n+first))

type SpecBatchQuery is Data:
  SpecBatchQuery{batch: WeightModel.BatchQuery,
    leaf: PackedLeaf(packed_geometry(WeightModel.group_query_numeric_source(WeightModel.batch_query_group(batch))),WeightModel.group_query_tap(WeightModel.batch_query_group(batch)))}

def spec_batch_terms(query: SpecBatchQuery,image: U32) -> PackedTerms:
  SpecBatchQuery{WeightModel.BatchQuery{WeightModel.GroupQuery{geometry,domain,size,old_image,input,gradient,tap,output,tap_inside,output_inside},range_rows,range_columns,range_size,images,ranges_frame},leaf} = query
  PackedTerms{WeightGroups.NumericSource{input,gradient,geometry,image},U32.to_nat(images),U32.to_nat(size),tap,output,leaf}

def spec_batch_images(query: SpecBatchQuery) -> ImagesQuery<Tree.Tree<F32>>:
  SpecBatchQuery{+batch,leaf} = query
  WeightModel.BatchQuery{group,+range_rows,+range_columns,+range_size,images,ranges_frame} = batch
  +terms = packed_terms_query(spec_batch_terms(query,0))
  ImagesQuery{terms_problem(~Tree.Tree<F32>,terms),terms_element(~Tree.Tree<F32>,terms),Grouped.ranges(range_rows,range_columns,range_size)}

def SpecBatchRangeBefore(+context: Shape.Both<SpecBatchQuery,U32>,+range: Grouped.Range) -> Data:
  Shape.Both{SpecBatchQuery{batch,leaf},image} = context
  WeightModel.group_query_key(WeightModel.batch_query_group(batch),range)

def SpecBatchRangeAfter(+context: Shape.Both<SpecBatchQuery,U32>,+range: Grouped.Range) -> Data:
  Shape.Both{query,image} = context
  PackedGroupRoom(spec_batch_terms(query,image),range)

def spec_batch_range_room(+context: Shape.Both<SpecBatchQuery,U32>,+range: Grouped.Range,+before: SpecBatchRangeBefore(context,range)) -> SpecBatchRangeAfter(context,range):
  Shape.Both{SpecBatchQuery{WeightModel.BatchQuery{WeightModel.GroupQuery{+geometry,domain,size,old_image,input,gradient,tap,output,tap_inside,output_inside},range_rows,range_columns,range_size,images,ranges_frame},leaf},image} = context
  WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Grouped.Range{first,count} = range
  Shape.Both{nonempty,Shape.Both{bounded,room}} = before
  room

def spec_batch_rooms(+query: SpecBatchQuery,+image: U32) ->
  Members.All(Grouped.Range,{range => PackedGroupRoom(spec_batch_terms(query,image),range) : Grouped.Range -> Data},images_groups(~Tree.Tree<F32>,spec_batch_images(query))):
  SpecBatchQuery{WeightModel.BatchQuery{+group,+range_rows,+range_columns,+range_size,images,+ranges_frame},leaf} = query
  WeightModel.GroupQuery{+geometry,domain,size,old_image,input,gradient,tap,output,tap_inside,output_inside} = group
  WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Members.transform_all(~Grouped.Range,~Shape.Both<SpecBatchQuery,U32>,~SpecBatchRangeBefore,~SpecBatchRangeAfter,~spec_batch_range_room,Grouped.ranges(range_rows,range_columns,range_size),Shape.Both{query,image},ranges_frame)

def spec_batch_native(query: SpecBatchQuery) -> WeightModel.BatchQuery:
  SpecBatchQuery{batch,leaf} = query
  batch

def spec_batch_size(query: SpecBatchQuery) -> Nat:
  SpecBatchQuery{WeightModel.BatchQuery{group,range_rows,range_columns,range_size,images,ranges_frame},leaf} = query
  U32.to_nat(images)

def SpecBatchGroupMatching(+context: Shape.Both<SpecBatchQuery,U32>,+range: Grouped.Range) -> Data:
  Shape.Both{+query,+image} = context
  {WeightModel.group_query_sample(WeightModel.group_query_at(WeightModel.batch_query_group(spec_batch_native(query)),image),range) == packed_group_value(spec_batch_terms(query,image),range) : F32}

def spec_batch_group_matching(+context: Shape.Both<SpecBatchQuery,U32>,+range: Grouped.Range,+before: SpecBatchRangeBefore(context,range)) -> SpecBatchGroupMatching(context,range):
  Shape.Both{SpecBatchQuery{WeightModel.BatchQuery{WeightModel.GroupQuery{+geometry,+domain,size,old_image,input,gradient,tap,output,tap_inside,output_inside},range_rows,range_columns,range_size,images,ranges_frame},leaf},image} = context
  WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  WeightGroups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned} = domain
  Grouped.Range{first,remaining} = range
  {==}

def spec_batch_group_list(+query: SpecBatchQuery,+image: U32) ->
  {Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~WeightModel.GroupQuery,~WeightModel.group_query_sample,images_groups(~Tree.Tree<F32>,spec_batch_images(query)),WeightModel.group_query_at(WeightModel.batch_query_group(spec_batch_native(query)),image)) ==
    Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~PackedTerms,~packed_group_value,images_groups(~Tree.Tree<F32>,spec_batch_images(query)),spec_batch_terms(query,image)) : List<&2,F32>}:
  SpecBatchQuery{WeightModel.BatchQuery{+group,+range_rows,+range_columns,+range_size,images,+ranges_frame},leaf} = query
  WeightModel.GroupQuery{+geometry,+domain,size,old_image,input,gradient,tap,output,tap_inside,output_inside} = group
  WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  WeightGroups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned} = domain
  +ranges = Grouped.ranges(range_rows,range_columns,range_size)
  +agreements = Members.transform_all(~Grouped.Range,~Shape.Both<SpecBatchQuery,U32>,~SpecBatchRangeBefore,~SpecBatchGroupMatching,~spec_batch_group_matching,ranges,Shape.Both{query,image},ranges_frame)
  Traversals.collect_values_context_agrees(~F32,~Grouped.Range,~WeightModel.GroupQuery,~PackedTerms,~WeightModel.group_query_sample,~packed_group_value,ranges,WeightModel.group_query_at(group,image),spec_batch_terms(query,image),agreements)

def SpecImageMatches(+query: SpecBatchQuery,+image: Nat) -> Data:
  {WeightModel.batch_query_sample(spec_batch_native(query),image) ==
    image_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),spec_batch_images(query),image) : F32}

def spec_batch_image(+query: SpecBatchQuery,+image: Nat,+inside: {Nat.is_lt(image,spec_batch_size(query)) == True{} : Bool}) -> SpecImageMatches(query,image):
  SpecBatchQuery{WeightModel.BatchQuery{+group,+range_rows,+range_columns,+range_size,+images,ranges_frame},+leaf} = query
  WeightModel.GroupQuery{+geometry,+domain,size,old_image,input,gradient,tap,output,tap_inside,output_inside} = group
  WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  WeightGroups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned} = domain
  +ranges = Grouped.ranges(range_rows,range_columns,range_size)
  +current = spec_batch_terms(query,U32.from_nat(image))
  +target = spec_batch_images(query)
  +terms = packed_terms_query(current)
  +groups_equal = packed_groups_spec(current,ranges,spec_batch_rooms(query,U32.from_nat(image)))
  %FullRange.roundtrip(image,images,Bounds.strict_le(image,U32.to_nat(images),inside)) :
    {WeightModel.batch_query_sample(spec_batch_native(query),image) == image_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),target,_) : F32}
  Equal.cong(List<&2,F32>,F32,values => Grouped.tree(values),
    Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~WeightModel.GroupQuery,~WeightModel.group_query_sample,ranges,WeightModel.group_query_at(group,U32.from_nat(image))),
    Spec.group_values(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),ranges,terms_problem(~Tree.Tree<F32>,terms),terms_image(~Tree.Tree<F32>,terms),terms_element(~Tree.Tree<F32>,terms)),
    Equal.trans(List<&2,F32>,
      Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~WeightModel.GroupQuery,~WeightModel.group_query_sample,ranges,WeightModel.group_query_at(group,U32.from_nat(image))),
      Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~PackedTerms,~packed_group_value,ranges,current),
      Spec.group_values(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),ranges,terms_problem(~Tree.Tree<F32>,terms),terms_image(~Tree.Tree<F32>,terms),terms_element(~Tree.Tree<F32>,terms)),spec_batch_group_list(query,U32.from_nat(image)),groups_equal))

def spec_batch_model_list(+query: SpecBatchQuery) ->
  {ListRead.enumerate(~F32,~WeightModel.BatchQuery,~WeightModel.batch_query_sample,spec_batch_size(query),0n,spec_batch_native(query)) ==
    ListRead.enumerate(~F32,~ImagesQuery<Tree.Tree<F32>>,~image_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),spec_batch_size(query),0n,spec_batch_images(query)) : List<&2,F32>}:
  +count = spec_batch_size(query)
  +native = spec_batch_native(query)
  +target = spec_batch_images(query)
  +agreements = Members.range_all(~SpecBatchQuery,~SpecImageMatches,~spec_batch_size,~spec_batch_image,count,0n,query,Order.reflexive(count))
  %Traversals.collect_values_context_range(~F32,~WeightModel.BatchQuery,~WeightModel.batch_query_sample,count,0n,native) :
    {_ == ListRead.enumerate(~F32,~ImagesQuery<Tree.Tree<F32>>,~image_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),count,0n,target) : List<&2,F32>}
  %Traversals.collect_values_context_range(~F32,~ImagesQuery<Tree.Tree<F32>>,~image_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),count,0n,target) :
    {Traversals.collect_values_context(~&2,~F32,~Nat,~WeightModel.BatchQuery,~WeightModel.batch_query_sample,Ranges.range(count,0n),native) == _ : List<&2,F32>}
  Traversals.collect_values_context_agrees(~F32,~Nat,~WeightModel.BatchQuery,~ImagesQuery<Tree.Tree<F32>>,~WeightModel.batch_query_sample,~image_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),Ranges.range(count,0n),native,target,agreements)

def spec_batch_lists(+query: SpecBatchQuery) ->
  {ListRead.enumerate(~F32,~WeightModel.BatchQuery,~WeightModel.batch_query_sample,spec_batch_size(query),0n,spec_batch_native(query)) ==
    Spec.image_values(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),spec_batch_size(query),0n,images_element(~Tree.Tree<F32>,spec_batch_images(query)),images_problem(~Tree.Tree<F32>,spec_batch_images(query)),images_groups(~Tree.Tree<F32>,spec_batch_images(query))) : List<&2,F32>}:
  +target = spec_batch_images(query)
  %Equal.sym(List<&2,F32>,Spec.image_values(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),spec_batch_size(query),0n,images_element(~Tree.Tree<F32>,target),images_problem(~Tree.Tree<F32>,target),images_groups(~Tree.Tree<F32>,target)),ListRead.enumerate(~F32,~ImagesQuery<Tree.Tree<F32>>,~image_query(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at)),spec_batch_size(query),0n,target),images_enumerated(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),target,spec_batch_size(query),0n)) :
    {ListRead.enumerate(~F32,~WeightModel.BatchQuery,~WeightModel.batch_query_sample,spec_batch_size(query),0n,spec_batch_native(query)) == _ : List<&2,F32>}
  spec_batch_model_list(query)

def spec_batch_tree(+query: SpecBatchQuery) ->
  {Grouped.tree(ListRead.enumerate(~F32,~WeightModel.BatchQuery,~WeightModel.batch_query_sample,spec_batch_size(query),0n,spec_batch_native(query))) ==
    Grouped.tree(Spec.image_values(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),spec_batch_size(query),0n,images_element(~Tree.Tree<F32>,spec_batch_images(query)),images_problem(~Tree.Tree<F32>,spec_batch_images(query)),images_groups(~Tree.Tree<F32>,spec_batch_images(query)))) : F32}:
  Equal.cong(List<&2,F32>,F32,values => Grouped.tree(values),ListRead.enumerate(~F32,~WeightModel.BatchQuery,~WeightModel.batch_query_sample,spec_batch_size(query),0n,spec_batch_native(query)),Spec.image_values(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~at),spec_batch_size(query),0n,images_element(~Tree.Tree<F32>,spec_batch_images(query)),images_problem(~Tree.Tree<F32>,spec_batch_images(query)),images_groups(~Tree.Tree<F32>,spec_batch_images(query))),spec_batch_lists(query))

def group_spec_problem(+query: WeightModel.GroupQuery,images: U32) -> Spec.Problem<Tree.Tree<F32>>:
  packed_problem(WeightModel.group_query_numeric_source(query),U32.to_nat(images),U32.to_nat(WeightModel.group_query_size(query)))

def group_spec_element(query: WeightModel.GroupQuery) -> Nat:
  WeightModel.GroupQuery{WeightKernel.Geometry{c,o,h,w,+k,s,p,rows,columns,taps},domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  Nat.add(Nat.mul(U32.to_nat(output),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(tap))

def native_weight_spec(+query: WeightModel.GroupQuery,+images: U32,
  +leaf: PackedLeaf(packed_geometry(WeightModel.group_query_numeric_source(query)),WeightModel.group_query_tap(query)),
  +size_positive: {Nat.is_lt(0n,U32.to_nat(WeightModel.group_query_size(query))) == True{} : Bool}) ->
  {ListRead.at(~F32,0.0,WeightModel.group_query_element(query),WeightModel.weight_values(WeightModel.query_weight(WeightModel.native_batch_query(query,images,size_positive)))) ==
    Spec.weight(~Tree.Tree<F32>,~at,group_spec_problem(query,images),group_spec_element(query)) : F32}:
  WeightModel.GroupQuery{+geometry,+domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  WeightGroups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned} = domain
  +batch = WeightModel.native_batch_query(query,images,size_positive)
  +spec_query = {SpecBatchQuery{batch,leaf} : SpecBatchQuery}
  Equal.trans(F32,ListRead.at(~F32,0.0,WeightModel.group_query_element(query),WeightModel.weight_values(WeightModel.query_weight(batch))),
    Grouped.tree(ListRead.enumerate(~F32,~WeightModel.BatchQuery,~WeightModel.batch_query_sample,U32.to_nat(images),0n,batch)),
    Spec.weight(~Tree.Tree<F32>,~at,group_spec_problem(query,images),group_spec_element(query)),WeightModel.query_weight_numeric(batch),spec_batch_tree(spec_query))

def native_weight_element(+query: WeightModel.GroupQuery,
  +taps_exact: {U32.to_nat(WeightPacking.geometry_taps(WeightModel.packed_query_geometry(query))) == Nat.mul(U32.to_nat(packed_channels(WeightModel.group_query_numeric_source(query))),Nat.mul(U32.to_nat(WeightPacking.geometry_kernel(WeightModel.packed_query_geometry(query))),U32.to_nat(WeightPacking.geometry_kernel(WeightModel.packed_query_geometry(query))))) : Nat}) ->
  {WeightModel.group_query_element(query) == group_spec_element(query) : Nat}:
  WeightModel.GroupQuery{geometry,domain,size,image,input,gradient,tap,+output,tap_inside,output_inside} = query
  WeightKernel.Geometry{c,o,h,w,+k,s,p,rows,columns,taps} = geometry
  Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(U32.to_nat(output),value),U32.to_nat(tap)),U32.to_nat(taps),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))),taps_exact)
