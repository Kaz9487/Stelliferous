# These lemmas cover whole tensors: a ready call is exactly the checked core
# operation, so the core theorems in tensor_operations_refinement describe its
# storage. Broadcasting names the stretched views that reach the core
# operation; tensor_broadcast gives their extents. A failed operand passes its
# problem on and its storage through unchanged. Results keep the operands'
# largest worker count. Band representation and schedule refinements require
# separate connecting lemmas.
import Base
import ./tensor_operations_refinement.bend as OperationsLaws
import ../storage_buffer.bend as Storage
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_operations.bend as Operations
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_band.bend as Bands
import ../traversal_partition.bend as Partition
import ./tensor_whole_leaf.bend as WholeLeaf

# Elementwise combination.

# Incompatible extents fail and leave the left storage as it was.
def combine_incompatible(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+operations: F32,+name: String,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,+left_workers: U32,
  right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+right_workers: U32,
  different: {Shapes.same_extents(left_shape,right_shape) == False{} : Bool},
  incompatible: {Shapes.broadcast(left_shape,right_shape) == None{} : Maybe<&2,List<&2,U32>>}) ->
  {Tensors.combine_with(~Element,~kernel,name,operations,Tensors.whole(~Element,left,left_shape,fill,Tensors.Usable{},left_workers),
      Tensors.whole(~Element,right,right_shape,right_fill,Tensors.Usable{},right_workers)) ==
    Tensors.whole(~Element,left,left_shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.ShapeMismatch{left_shape,right_shape}}},U32.max(left_workers,right_workers)) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Shapes.same_extents(left_shape,right_shape),False{},different) :
    {Tensors.combine_shapes(~Element,~kernel,name,operations,_,Partition.Leaf{left,0,1,Unit{}},Bands.Geometry{1,1,1,Tensors.count_or_zero(left_shape)},left_shape,fill,Tensors.Usable{},left_workers,
      Partition.Leaf{right,0,1,Unit{}},Bands.Geometry{1,1,1,Tensors.count_or_zero(right_shape)},right_shape,right_fill,Tensors.Usable{},right_workers) ==
    Tensors.whole(~Element,left,left_shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.ShapeMismatch{left_shape,right_shape}}},U32.max(left_workers,right_workers)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(left_shape,right_shape),None{},incompatible) :
    {Tensors.combine_broadcast_with(~Element,~kernel,name,operations,Tensors.Usable{},_,U32.max(left_workers,right_workers),
      Partition.Leaf{left,0,1,Unit{}},Bands.Geometry{1,1,1,Tensors.count_or_zero(left_shape)},left_shape,fill,
      Partition.Leaf{right,0,1,Unit{}},Bands.Geometry{1,1,1,Tensors.count_or_zero(right_shape)},right_shape,right_fill) ==
    Tensors.whole(~Element,left,left_shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.ShapeMismatch{left_shape,right_shape}}},U32.max(left_workers,right_workers)) : Tensors.Tensor<Element>}
  {==}

def failed_shapes(~Element: Data,~kernel: OperationsLaws.Kernel(~Element),+name: String,+operations: F32,same: Bool,+problem: Tensors.Problem,+left_status: Tensors.Status,+right_status: Tensors.Status,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,+left_workers: U32,
  right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+right_workers: U32,
  failed: {Tensors.first_failure(left_status,right_status) == Tensors.Failed{problem} : Tensors.Status}) ->
  {Tensors.combine_shapes(~Element,~kernel,name,operations,same,Partition.Leaf{left,0,1,Unit{}},Bands.Geometry{1,1,1,Tensors.count_or_zero(left_shape)},left_shape,fill,left_status,left_workers,
      Partition.Leaf{right,0,1,Unit{}},Bands.Geometry{1,1,1,Tensors.count_or_zero(right_shape)},right_shape,right_fill,right_status,right_workers) ==
    Tensors.whole(~Element,left,left_shape,fill,Tensors.Failed{problem},U32.max(left_workers,right_workers)) : Tensors.Tensor<Element>}:
  match same:
    case True{}:
      %Equal.sym(Tensors.Status,Tensors.first_failure(left_status,right_status),Tensors.Failed{problem},failed) :
        {Tensors.combine_equal_with(~Element,~kernel,operations,_,U32.max(left_workers,right_workers),left_shape,
            Partition.Leaf{left,0,1,Unit{}},Bands.Geometry{1,1,1,Tensors.count_or_zero(left_shape)},fill,Partition.Leaf{right,0,1,Unit{}},Bands.Geometry{1,1,1,Tensors.count_or_zero(right_shape)},right_fill) ==
          Tensors.whole(~Element,left,left_shape,fill,Tensors.Failed{problem},U32.max(left_workers,right_workers)) : Tensors.Tensor<Element>}
      {==}
    case False{}:
      %Equal.sym(Tensors.Status,Tensors.first_failure(left_status,right_status),Tensors.Failed{problem},failed) :
        {Tensors.combine_broadcast_with(~Element,~kernel,name,operations,_,Shapes.broadcast(left_shape,right_shape),U32.max(left_workers,right_workers),
            Partition.Leaf{left,0,1,Unit{}},Bands.Geometry{1,1,1,Tensors.count_or_zero(left_shape)},left_shape,fill,
            Partition.Leaf{right,0,1,Unit{}},Bands.Geometry{1,1,1,Tensors.count_or_zero(right_shape)},right_shape,right_fill) ==
          Tensors.whole(~Element,left,left_shape,fill,Tensors.Failed{problem},U32.max(left_workers,right_workers)) : Tensors.Tensor<Element>}
      {==}

def combine_left_failed(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+operations: F32,+name: String,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,+problem: Tensors.Problem,+left_workers: U32,
  right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+right_status: Tensors.Status,+right_workers: U32) ->
  {Tensors.combine_with(~Element,~kernel,name,operations,Tensors.whole(~Element,left,left_shape,fill,Tensors.Failed{problem},left_workers),
      Tensors.whole(~Element,right,right_shape,right_fill,right_status,right_workers)) ==
    Tensors.whole(~Element,left,left_shape,fill,Tensors.Failed{problem},U32.max(left_workers,right_workers)) : Tensors.Tensor<Element>}:
  failed_shapes(~Element,~kernel,name,operations,Shapes.same_extents(left_shape,right_shape),problem,Tensors.Failed{problem},right_status,
    left,left_shape,fill,left_workers,right,right_shape,right_fill,right_workers,{==})

def combine_right_failed(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+operations: F32,+name: String,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,+left_workers: U32,
  right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+problem: Tensors.Problem,+right_workers: U32) ->
  {Tensors.combine_with(~Element,~kernel,name,operations,Tensors.whole(~Element,left,left_shape,fill,Tensors.Usable{},left_workers),
      Tensors.whole(~Element,right,right_shape,right_fill,Tensors.Failed{problem},right_workers)) ==
    Tensors.whole(~Element,left,left_shape,fill,Tensors.Failed{problem},U32.max(left_workers,right_workers)) : Tensors.Tensor<Element>}:
  failed_shapes(~Element,~kernel,name,operations,Shapes.same_extents(left_shape,right_shape),problem,Tensors.Usable{},Tensors.Failed{problem},
    left,left_shape,fill,left_workers,right,right_shape,right_fill,right_workers,{==})

# Unary application and indexed application.

def apply_failed(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,+context: Context,+operations: F32,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+problem: Tensors.Problem,
  +workers: U32) ->
  {Tensors.apply_context(~Element,~Context,~mapper,context,operations,Tensors.whole(~Element,storage,shape,fill,Tensors.Failed{problem},workers)) ==
    Tensors.whole(~Element,storage,shape,fill,Tensors.Failed{problem},workers) : Tensors.Tensor<Element>}:
  {==}

# check is the only operation that observes and clears a failure.

def check_ready(~Element: Data,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Tensors.check(~Element,Tensors.whole(~Element,storage,shape,fill,Tensors.Usable{},workers)) ==
    Done{Tensors.whole(~Element,storage,shape,fill,Tensors.Usable{},workers)} : Result<&1,&1,Tensors.Tensor<Element> & Tensors.Problem,Tensors.Tensor<Element>>}:
  {==}

def check_failed(~Element: Data,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+problem: Tensors.Problem,+workers: U32) ->
  {Tensors.check(~Element,Tensors.whole(~Element,storage,shape,fill,Tensors.Failed{problem},workers)) ==
    Fail{(Tensors.whole(~Element,storage,shape,fill,Tensors.Usable{},workers),problem)} : Result<&1,&1,Tensors.Tensor<Element> & Tensors.Problem,Tensors.Tensor<Element>>}:
  {==}

# A constant padding with padded extents is one write of the tensor through
# the interior view, unless nothing is added.
def pad_constant_selected(~Element: Data,+value: Element,tensor: Tensors.Tensor<Element>,+before: List<&2,U32>,+after: List<&2,U32>,+padded: List<&2,U32>) ->
  {Layout.pad_border(~Element,Layout.Constant{value},tensor,before,after,Some{padded}) ==
    Layout.pad_constant(~Element,Layout.all_zero(before) && Layout.all_zero(after),value,tensor,before,padded) : Tensors.Tensor<Element>}:
  {==}
