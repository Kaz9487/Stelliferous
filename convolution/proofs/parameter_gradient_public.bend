# The weight and bias gradients of storage holding the readers' values are
# Ready results of the readers' gradient contract, and the calls give their
# sources back holding the same values.
import Base
import ../parameter_gradient_specification.bend as Spec
import ./parameter_gradient_bridge.bend as Bridge
import ./parameter_gradient_readers.bend as Readers
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_parameter_gradient.bend as Parameter
import ../../lib/kernel_parameter_gradient.bend as Kernel
import ../../lib/kernel_parameter_gradient_buffer.bend as KernelBuffer
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/parameter_gradient_refinement.bend as Model
import ../../lib/proofs/parameter_gradient_buffer_refinement.bend as BufferModel
import ../../lib/proofs/tensor_list_values.bend as Listed
import ../../lib/proofs/list_storage_readback.bend as ListRead
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_gather_values.bend as GatherValues
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/word_encoding.bend as Words
import ../../lib/kernel_weight_gradient.bend as WeightKernel
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/proofs/kernel_weight_gradient_refinement.bend as WeightModel
import ../../lib/proofs/kernel_weight_gradient_group.bend as WeightGroups
import ../../lib/proofs/kernel_weight_gradient_packing.bend as WeightPacking
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/boolean_logic.bend as Logic
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/u32_full_range.bend as FullRange
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_count_refinement.bend as Count
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/default_machine_limits.bend as Default
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/tensor_whole_leaf.bend as WholeLeaf
import ../../lib/proofs/storage_certificate.bend as StorageCertificate
import ../../lib/proofs/parameter_gradient_certificate.bend as SourceCertificate
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_allocation_refinement.bend as AllocationLaws
import ../../lib/traversal_grouped.bend as Grouped
import ../../lib/tensor_band.bend as Bands
import ../../lib/proofs/grouped_reduction_refinement.bend as GroupedRanges
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/tensor_band_from_whole.bend as FromWhole
import ../../lib/proofs/tensor_band_each.bend as BandEach
import ../../lib/proofs/tensor_band_model.bend as BandModel
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_count.bend as BandCount
import ../../lib/proofs/tensor_band_inspection.bend as Inspection
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/word_multiplication.bend as Multiplication
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/kernel_weight_gradient_parts.bend as KParts

def weight_result(result: Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>)) -> Tensors.Tensor<F32>:
  (gradient,sources) = result
  gradient

def bias_result(result: Tensors.Tensor<F32> & Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  (gradient,source) = result
  gradient

def returned_sources(result: Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>)) -> Tensors.Tensor<F32> & Tensors.Tensor<F32>:
  (gradient,sources) = result
  sources

def returned_source(result: Tensors.Tensor<F32> & Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  (gradient,source) = result
  source

# The input and the output gradient come back as tensors of their shapes
# that hold their readers' values.
def Returned(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+input_context: Input,+gradient_context: Gradient,
  +xs: List<&2,U32>,+ds: List<&2,U32>,sources: Tensors.Tensor<F32> & Tensors.Tensor<F32>) -> Data:
  (input,gradient) = sources
  Shape.Both<Global.Shaped(~F32,~Input,~input_value,input_context,xs,input),Global.Shaped(~F32,~Gradient,~gradient_value,gradient_context,ds,gradient)>

# A weight gradient call: the gradient holds the readers' weight gradient,
# and the sources come back.
def WeightCall(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+input_context: Input,+gradient_context: Gradient,
  +xi: Maybe<&2,Parameter.ImageGeometry>,+di: Maybe<&2,Parameter.ImageGeometry>,+k: U32,+s: U32,+p: U32,+g: U32,+xs: List<&2,U32>,+ds: List<&2,U32>,
  result: Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>)) -> Data:
  Shape.Both<Global.Contract(~F32,~Readers.WeightRead<Input,Gradient>,~Readers.weight_at(~Input,~input_value,~Gradient,~gradient_value),
      Readers.WeightRead{input_context,gradient_context,xi,di,k,s,p,g},weight_result(result)),
    Returned(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,xs,ds,returned_sources(result))>

# A bias gradient call: the gradient holds the reader's bias gradient, and
# the output gradient comes back.
def BiasCall(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+gradient_context: Gradient,+di: Maybe<&2,Parameter.ImageGeometry>,+g: U32,+shape: List<&2,U32>,
  result: Tensors.Tensor<F32> & Tensors.Tensor<F32>) -> Data:
  Shape.Both<Global.Contract(~F32,~Readers.BiasRead<Gradient>,~Readers.bias_at(~Gradient,~gradient_value),Readers.BiasRead{gradient_context,di,g},bias_result(result)),
    Global.Shaped(~F32,~Gradient,~gradient_value,gradient_context,shape,returned_source(result))>

def bias_count(work: Kernel.Work) -> Nat:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  o

def weight_list(+work: Kernel.Work,x: Buffer.Model<F32>,d: Buffer.Model<F32>) -> List<&2,F32>:
  WeightModel.weight_values(WeightModel.weight(work,x,d))

def bias_list(+work: Kernel.Work,d: Buffer.Model<F32>) -> List<&2,F32>:
  Bridge.bias_list(bias_count(work),0n,work,Buffer.tree(~F32,d))

def bias_values_listed(+work: Kernel.Work,+d: Buffer.Model<F32>) ->
  {Model.bias_values(work,Buffer.tree(~F32,d)) == bias_list(work,d) : List<&2,F32>}:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Bridge.bias_output(bias_count(work),0n,work,Buffer.tree(~F32,d))

def bias_finished(+shape: List<&2,U32>,+fill: F32,+status: Tensors.Status,+workers: U32,+outputs: U32,result: Storage.Buffer<F32> & Storage.Buffer<F32>) ->
  {bias_result(Parameter.bias_finished(shape,fill,status,workers,outputs,result)) == Tensors.whole(~F32,BufferModel.written(result),[outputs],0.0,Tensors.Usable{},workers) :
    Tensors.Tensor<F32>}:
  (d,values) = result
  {==}

def weight_shape(xi: Parameter.ImageGeometry,di: Parameter.ImageGeometry,+k: U32) -> List<&2,U32>:
  Parameter.ImageGeometry{n,c,h,w} = xi
  Parameter.ImageGeometry{dn,o,rows,columns} = di
  [o,c,k,k]

def bias_shape(di: Parameter.ImageGeometry) -> List<&2,U32>:
  Parameter.ImageGeometry{n,o,rows,columns} = di
  [o]

# Every output written into fresh storage of count elements: the storage
# holds the bias list from 0.
def bias_stored(+work: Kernel.Work,+d: Buffer.Model<F32>,+count: U32) ->
  {Buffer.contents(~F32,BufferModel.written(KernelBuffer.bias(work,Buffer.pack(~F32,d),Tensors.buffer(~F32,0.0,count)))) ==
    Buffer.Content{Tree.write_sequence(~F32,bias_list(work,d),0,Tree.fresh(F32,Tensors.depth_for(count),0.0)),U32.to_nat(count)} : Buffer.Content<F32>}:
  +fresh = Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,count))
  +allocated = {Buffer.Content{Tree.fresh(F32,Tensors.depth_for(count),0.0),U32.to_nat(count)} : Buffer.Content<F32>}
  +listed = {Buffer.Content{Tree.write_sequence(~F32,bias_list(work,d),0,Tree.fresh(F32,Tensors.depth_for(count),0.0)),U32.to_nat(count)} : Buffer.Content<F32>}
  +fresh_content = Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,fresh),Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count)),allocated,
    Buffer.remodel_content(~F32,Tensors.buffer(~F32,0.0,count)),AllocationLaws.contents(~F32,Tensors.depth_for(count),0.0,U32.to_nat(count)))
  %Buffer.remodeled(~F32,Tensors.buffer(~F32,0.0,count)) :
    {Buffer.contents(~F32,BufferModel.written(KernelBuffer.bias(work,Buffer.pack(~F32,d),_))) == listed : Buffer.Content<F32>}
  Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,BufferModel.written(KernelBuffer.bias(work,Buffer.pack(~F32,d),Buffer.pack(~F32,fresh)))),
    Buffer.content(~F32,Model.bias_written(work,Buffer.tree(~F32,d),fresh)),listed,
    BufferModel.bias(work,d,fresh),
    Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,Model.bias_written(work,Buffer.tree(~F32,d),fresh)),
      Buffer.Content{Tree.write_sequence(~F32,Model.bias_values(work,Buffer.tree(~F32,d)),0,Buffer.content_tree(~F32,Buffer.content(~F32,fresh))),
        Buffer.content_length(~F32,Buffer.content(~F32,fresh))},
      listed,
      Model.bias_content(work,Buffer.tree(~F32,d),fresh),
      %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,fresh),allocated,fresh_content) :
        {Buffer.Content{Tree.write_sequence(~F32,Model.bias_values(work,Buffer.tree(~F32,d)),0,Buffer.content_tree(~F32,_)),Buffer.content_length(~F32,_)} == listed :
          Buffer.Content<F32>}
      %Equal.sym(List<&2,F32>,Model.bias_values(work,Buffer.tree(~F32,d)),bias_list(work,d),bias_values_listed(work,d)) :
        {Buffer.Content{Tree.write_sequence(~F32,_,0,Tree.fresh(F32,Tensors.depth_for(count),0.0)),U32.to_nat(count)} == listed : Buffer.Content<F32>}
      {==}))

# The bias kernel gives back the storage it read: its contents hold the
# reader's values.
def bias_kept_source(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+gradient_context: Gradient,+shape: List<&2,U32>,+fill: F32,+workers: U32,+outputs: U32,
  +tree: Tree.Tree<F32>,+length: Nat,+facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,shape,Buffer.Content{tree,length}),
  result: Array<F32> & Storage.Buffer<F32>,certificate: StorageCertificate.Certificate<F32,SourceCertificate.source(~Array<F32>,result)>,
  same: {SourceCertificate.source(~Array<F32>,result) == Tree.pack(F32,tree) : Array<F32>}) ->
  Global.Shaped(~F32,~Gradient,~gradient_value,gradient_context,shape,
    returned_source(Parameter.bias_finished(shape,fill,Tensors.Usable{},workers,outputs,KernelBuffer.bias_kept(length,result,certificate)))):
  (storage,values) = result
  +reflected = Equal.trans(Tree.Tree<F32>,Tree.reflect(F32,storage),Tree.reflect(F32,Tree.pack(F32,tree)),tree,
    Equal.cong(Array<F32>,Tree.Tree<F32>,array => Tree.reflect(F32,array),storage,Tree.pack(F32,tree),same),StorageLaws.reflect_packed(F32,tree))
  WholeLeaf.stored_shaped(~F32,~Gradient,~gradient_value,gradient_context,Storage.Buffer{storage,length,certificate},shape,fill,workers,
    Buffer.observe(~F32,Storage.Buffer{storage,length,certificate}),
    %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,storage),tree,reflected) : GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,shape,Buffer.Content{_,length})
    facts)

def bias_returned(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+gradient_context: Gradient,+di: Parameter.ImageGeometry,+g: U32,+d: Buffer.Model<F32>,
  +shape: List<&2,U32>,+fill: F32,+workers: U32,+count: U32,+facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,shape,Buffer.content(~F32,d))) ->
  Global.Shaped(~F32,~Gradient,~gradient_value,gradient_context,shape,
    returned_source(Parameter.bias_counted(Some{count},di,g,Buffer.pack(~F32,d),shape,fill,Tensors.Usable{},workers))):
  Parameter.ImageGeometry{+n,+o,+rows,+columns} = di
  Buffer.Model{+tree,+length,+certificate} = d
  +work = Parameter.bias_work(Parameter.ImageGeometry{n,o,rows,columns},g)
  bias_kept_source(~Gradient,~gradient_value,gradient_context,shape,fill,workers,o,tree,length,facts,
    Kernel.bias(work,Tree.pack(F32,tree),Tensors.buffer(~F32,0.0,count)),SourceCertificate.bias(work,Tree.pack(F32,tree),Tensors.buffer(~F32,0.0,count),certificate),
    SourceCertificate.bias_owner(work,Tree.pack(F32,tree),Tensors.buffer(~F32,0.0,count),certificate))

def weight_sources(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,input: Tree.Tree<F32>,gradient: Tree.Tree<F32>,index: Nat) -> F32:
  Spec.weight(~Tree.Tree<F32>,~Bridge.at,Bridge.weight_problem(Parameter.work(xi,di,k,s,p,g),Model.Source{input,gradient}),index)

def bias_sources(+di: Parameter.ImageGeometry,+g: U32,gradient: Tree.Tree<F32>,index: Nat) -> F32:
  Spec.bias(~Tree.Tree<F32>,~Bridge.at,Bridge.bias_problem(Parameter.bias_work(di,g),gradient),index)

def weight_value(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,x: Buffer.Model<F32>,d: Buffer.Model<F32>,index: Nat) -> F32:
  weight_sources(xi,di,k,s,p,g,Buffer.tree(~F32,x),Buffer.tree(~F32,d),index)

def bias_value(+di: Parameter.ImageGeometry,+g: U32,d: Buffer.Model<F32>,index: Nat) -> F32:
  bias_sources(di,g,Buffer.tree(~F32,d),index)

def image_count(image: Parameter.ImageGeometry) -> U32:
  Parameter.ImageGeometry{n,c,h,w} = image
  n

type WeightPositive<-xi: Parameter.ImageGeometry,-di: Parameter.ImageGeometry,-k: U32,-s: U32,-g: U32> is Data:
  WeightPositive{input: {Parameter.positive(xi) == True{} : Bool},gradient: {Parameter.positive(di) == True{} : Bool},
    kernel: {U32.is_lt(0,k) == True{} : Bool},stride: {U32.is_lt(0,s) == True{} : Bool},group: {U32.is_lt(0,g) == True{} : Bool},
    images: {U32.is_eq(image_count(xi),image_count(di)) == True{} : Bool}}

# The conditions of a valid call other than the count of its weight.
def weight_conditions(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32) -> Bool:
  Parameter.ImageGeometry{+n,+c,+h,+w} = xi
  Parameter.ImageGeometry{dn,+o,rows,columns} = di
  Parameter.positive(xi) && Parameter.positive(di) && U32.is_eq(n,dn) && U32.is_lt(0,k) && U32.is_lt(0,s) && U32.is_lt(0,g) &&
    Nat.is_le(U32.to_nat(k),Nat.add(U32.to_nat(h),Nat.mul(2n,U32.to_nat(p)))) &&
    Nat.is_le(U32.to_nat(k),Nat.add(U32.to_nat(w),Nat.mul(2n,U32.to_nat(p)))) &&
    Nat.is_eq(U32.to_nat(rows),KernelShape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))) &&
    Nat.is_eq(U32.to_nat(columns),KernelShape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)))

# A valid call counts its weight.
def weight_fits(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +valid: {Parameter.valid(xi,di,k,s,p,g) == True{} : Bool}) -> {Parameter.fits(Tensors.count_of(weight_shape(xi,di,k))) == True{} : Bool}:
  Parameter.ImageGeometry{+n,+c,+h,+w} = xi
  Parameter.ImageGeometry{+dn,+o,+rows,+columns} = di
  Logic.right(weight_conditions(xi,di,k,s,p,g),Parameter.fits(Tensors.count_of([o,c,k,k])),valid)

def weight_positive(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +valid: {Parameter.valid(xi,di,k,s,p,g) == True{} : Bool}) -> WeightPositive<xi,di,k,s,g>:
  Parameter.ImageGeometry{+n,+c,+h,+w} = xi
  Parameter.ImageGeometry{+dn,+o,+rows,+columns} = di
  +ab = Parameter.positive(xi) && Parameter.positive(di)
  +abc = ab && U32.is_eq(n,dn)
  +ak = abc && U32.is_lt(0,k)
  +as = ak && U32.is_lt(0,s)
  +ag = as && U32.is_lt(0,g)
  +height = Nat.is_le(U32.to_nat(k),Nat.add(U32.to_nat(h),Nat.mul(2n,U32.to_nat(p))))
  +width = Nat.is_le(U32.to_nat(k),Nat.add(U32.to_nat(w),Nat.mul(2n,U32.to_nat(p))))
  +row = Nat.is_eq(U32.to_nat(rows),KernelShape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)))
  +column = Nat.is_eq(U32.to_nat(columns),KernelShape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)))
  +ah = ag && height
  +aw = ah && width
  +ar = aw && row
  +ac_true = Logic.left(weight_conditions(xi,di,k,s,p,g),Parameter.fits(Tensors.count_of([o,c,k,k])),valid)
  +ar_true = Logic.left(ar,column,ac_true)
  +aw_true = Logic.left(aw,row,ar_true)
  +ah_true = Logic.left(ah,width,aw_true)
  +ag_true = Logic.left(ag,height,ah_true)
  +as_true = Logic.left(as,U32.is_lt(0,g),ag_true)
  +ak_true = Logic.left(ak,U32.is_lt(0,s),as_true)
  +abc_true = Logic.left(abc,U32.is_lt(0,k),ak_true)
  +ab_true = Logic.left(ab,U32.is_eq(n,dn),abc_true)
  WeightPositive{Logic.left(Parameter.positive(xi),Parameter.positive(di),ab_true),Logic.right(Parameter.positive(xi),Parameter.positive(di),ab_true),Logic.right(abc,U32.is_lt(0,k),ak_true),Logic.right(ak,U32.is_lt(0,s),as_true),Logic.right(as,U32.is_lt(0,g),ag_true),
    Logic.right(ab,U32.is_eq(n,dn),abc_true)}

def weight_geometry(xi: Parameter.ImageGeometry,di: Parameter.ImageGeometry,+k: U32,s: U32,p: U32) -> WeightKernel.Geometry:
  Parameter.ImageGeometry{n,+c,h,w} = xi
  Parameter.ImageGeometry{dn,o,rows,columns} = di
  WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))}

def weight_tap_count(xi: Parameter.ImageGeometry,+k: U32) -> Nat:
  Parameter.ImageGeometry{n,c,h,w} = xi
  Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))

def weight_shape_count(+o: U32,+c: U32,+k: U32) ->
  {ShapeLaws.elements([o,c,k,k]) == Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) : Nat}:
  Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),value))),Nat.mul(U32.to_nat(k),1n),U32.to_nat(k),Order.times_one(U32.to_nat(k)))

type WeightGeometry<-xi: Parameter.ImageGeometry,-di: Parameter.ImageGeometry,-k: U32,-s: U32,-p: U32> is Data:
  WeightGeometry{domain: WeightGroups.GeometryReady<weight_geometry(xi,di,k,s,p)>,
    taps_exact: {U32.to_nat(WeightPacking.geometry_taps(weight_geometry(xi,di,k,s,p))) == weight_tap_count(xi,k) : Nat}}

def weight_geometry_ready_limits(+limits: Machine.Limits,
  +aligned: {U32.is_eq((Machine.capacity(limits) % 8 : U32),0) == True{} : Bool},+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +positive: WeightPositive<xi,di,k,s,g>,+count: U32,
  +exact: {ShapeLaws.elements(weight_shape(xi,di,k)) == U32.to_nat(count) : Nat},
  +bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) -> WeightGeometry<xi,di,k,s,p>:
  Parameter.ImageGeometry{+n,+c,+h,+w} = xi
  Parameter.ImageGeometry{+dn,+o,+rows,+columns} = di
  WeightPositive{+xp,+dp,+kp,+sp,gp,images} = positive
  +xc = Logic.left((U32.is_lt(0,n) && U32.is_lt(0,c)) && U32.is_lt(0,h),U32.is_lt(0,w),xp)
  +nc = Logic.left(U32.is_lt(0,n) && U32.is_lt(0,c),U32.is_lt(0,h),xc)
  +cp_word = Logic.right(U32.is_lt(0,n),U32.is_lt(0,c),nc)
  +dr = Logic.left((U32.is_lt(0,dn) && U32.is_lt(0,o)) && U32.is_lt(0,rows),U32.is_lt(0,columns),dp)
  +no = Logic.left(U32.is_lt(0,dn) && U32.is_lt(0,o),U32.is_lt(0,rows),dr)
  +op_word = Logic.right(U32.is_lt(0,dn),U32.is_lt(0,o),no)
  +columns_word = Logic.right((U32.is_lt(0,dn) && U32.is_lt(0,o)) && U32.is_lt(0,rows),U32.is_lt(0,columns),dp)
  +cp = {%Comparison.lt(0,c) : {_ == True{} : Bool} cp_word : {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool}}
  +op = {%Comparison.lt(0,o) : {_ == True{} : Bool} op_word : {Nat.is_lt(0n,U32.to_nat(o)) == True{} : Bool}}
  +kernel_positive = {%Comparison.lt(0,k) : {_ == True{} : Bool} kp : {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool}}
  +stride_positive = {%Comparison.lt(0,s) : {_ == True{} : Bool} sp : {Nat.is_lt(0n,U32.to_nat(s)) == True{} : Bool}}
  +columns_positive = {%Comparison.lt(0,columns) : {_ == True{} : Bool} columns_word : {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool}}
  +square = Nat.mul(U32.to_nat(k),U32.to_nat(k))
  +taps = weight_tap_count(xi,k)
  +total = Nat.mul(U32.to_nat(o),taps)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +square_positive = Order.product_positive(U32.to_nat(k),U32.to_nat(k),kernel_positive,kernel_positive)
  +taps_positive = Order.product_positive(U32.to_nat(c),square,cp,square_positive)
  +total_bound = {%weight_shape_count(o,c,k) : {Nat.is_le(_,cap) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements([o,c,k,k]),U32.to_nat(count),exact) : {Nat.is_le(_,cap) == True{} : Bool}
    %Comparison.le(count,capacity) : {_ == True{} : Bool} bounded : {Nat.is_le(total,cap) == True{} : Bool}}
  +taps_bound = Order.transitive(taps,total,cap,
    %RowMajor.multiply_commutative(taps,U32.to_nat(o)) : {Nat.is_le(taps,_) == True{} : Bool} RowMajor.product_covers(taps,U32.to_nat(o),op),total_bound)
  +square_bound = Order.transitive(square,taps,cap,
    %RowMajor.multiply_commutative(square,U32.to_nat(c)) : {Nat.is_le(square,_) == True{} : Bool} RowMajor.product_covers(square,U32.to_nat(c),cp),taps_bound)
  +taps_exact = FullRange.roundtrip(taps,capacity,taps_bound)
  +native_taps_positive = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(taps)),taps,taps_exact) : {Nat.is_lt(0n,_) == True{} : Bool} taps_positive : {Nat.is_lt(0n,U32.to_nat(U32.from_nat(taps))) == True{} : Bool}}
  +native_count_bound = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(taps)),taps,taps_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(o)),cap) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(taps,U32.to_nat(o)),total,RowMajor.multiply_commutative(taps,U32.to_nat(o))) : {Nat.is_le(_,cap) == True{} : Bool}
    total_bound : {Nat.is_le(Nat.mul(U32.to_nat(U32.from_nat(taps)),U32.to_nat(o)),cap) == True{} : Bool}}
  WeightGeometry{WeightGroups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square_bound,native_taps_positive,op,native_count_bound,aligned},taps_exact}

def weight_geometry_ready(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +positive: WeightPositive<xi,di,k,s,g>,+count: U32,counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),count>) -> WeightGeometry<xi,di,k,s,p>:
  Count.Count{exact,bounded} = counted
  weight_geometry_ready_limits(Default.limits(),{==},xi,di,k,s,p,g,positive,count,exact,bounded)

def weight_list_numeric_domain(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+x: Buffer.Model<F32>,+d: Buffer.Model<F32>,+index: Nat,
  +positive: WeightPositive<xi,di,k,s,g>,+geometry_ready: WeightGeometry<xi,di,k,s,p>,+count: U32,
  +counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),count>,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {ListRead.at(~F32,0.0,index,weight_list(Parameter.work(xi,di,k,s,p,g),x,d)) == weight_value(xi,di,k,s,p,g,x,d,index) : F32}:
  Parameter.ImageGeometry{+n,+c,+h,+w} = xi
  Parameter.ImageGeometry{dn,+o,rows,+columns} = di
  WeightPositive{+xp,dp,kp,sp,+gp,images} = positive
  WeightGeometry{+domain,+taps_exact} = geometry_ready
  WeightGroups.GeometryReady{limits,+columns_positive,+kernel_positive,stride_positive,square,+native_taps_positive,outputs_positive,bound,aligned} = domain
  Count.Count{+exact,bounded} = counted
  +taps = weight_tap_count(xi,k)
  +native_taps = WeightPacking.geometry_taps(weight_geometry(xi,di,k,s,p))
  +taps_positive = {%taps_exact : {Nat.is_lt(0n,_) == True{} : Bool} native_taps_positive : {Nat.is_lt(0n,taps) == True{} : Bool}}
  +index_inside = {%weight_shape_count(o,c,k) : {Nat.is_lt(index,_) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements([o,c,k,k]),U32.to_nat(count),exact) : {Nat.is_lt(index,_) == True{} : Bool}
    inside : {Nat.is_lt(index,Nat.mul(U32.to_nat(o),taps)) == True{} : Bool}}
  +tap = Nat.mod(index,taps)
  +output = Nat.div(index,taps)
  +tap_inside = DivisionOrder.remainder_small(index,taps,taps_positive)
  +output_inside = DivisionOrder.quotient_below(index,U32.to_nat(o),taps,taps_positive,index_inside)
  +tap_word = U32.from_nat(tap)
  +output_word = U32.from_nat(output)
  +tap_exact = FullRange.roundtrip(tap,native_taps,
    %Equal.sym(Nat,U32.to_nat(native_taps),taps,taps_exact) : {Nat.is_le(tap,_) == True{} : Bool}
    Bounds.strict_le(tap,taps,tap_inside))
  +output_exact = FullRange.roundtrip(output,o,Bounds.strict_le(output,U32.to_nat(o),output_inside))
  +native_tap_inside = {%Equal.sym(Bool,U32.is_lt(tap_word,native_taps),Nat.is_lt(U32.to_nat(tap_word),U32.to_nat(native_taps)),Comparison.lt(tap_word,native_taps)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(tap_word),tap,tap_exact) : {Nat.is_lt(_,U32.to_nat(native_taps)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(native_taps),taps,taps_exact) : {Nat.is_lt(tap,_) == True{} : Bool}
    tap_inside : {U32.is_lt(tap_word,native_taps) == True{} : Bool}}
  +native_output_inside = {%Equal.sym(Bool,U32.is_lt(output_word,o),Nat.is_lt(U32.to_nat(output_word),U32.to_nat(o)),Comparison.lt(output_word,o)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(output_word),output,output_exact) : {Nat.is_lt(_,U32.to_nat(o)) == True{} : Bool}
    output_inside : {U32.is_lt(output_word,o) == True{} : Bool}}
  +xc = Logic.left((U32.is_lt(0,n) && U32.is_lt(0,c)) && U32.is_lt(0,h),U32.is_lt(0,w),xp)
  +nc = Logic.left(U32.is_lt(0,n) && U32.is_lt(0,c),U32.is_lt(0,h),xc)
  +cp_word = Logic.right(U32.is_lt(0,n),U32.is_lt(0,c),nc)
  +cp = {%Comparison.lt(0,c) : {_ == True{} : Bool} cp_word : {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool}}
  +size_positive = {%Comparison.lt(0,g) : {_ == True{} : Bool} gp : {Nat.is_lt(0n,U32.to_nat(g)) == True{} : Bool}}
  +query = {WeightModel.GroupQuery{weight_geometry(xi,di,k,s,p),domain,g,0,x,d,tap_word,output_word,native_tap_inside,native_output_inside} : WeightModel.GroupQuery}
  +leaf = {Shape.Both{cp,Shape.Both{kernel_positive,Shape.Both{columns_positive,
    %Equal.sym(Nat,U32.to_nat(tap_word),tap,tap_exact) : {Nat.is_lt(_,taps) == True{} : Bool} tap_inside}}} : Bridge.PackedLeaf(weight_geometry(xi,di,k,s,p),tap_word)}
  +spec_index = {%Equal.sym(Nat,U32.to_nat(output_word),output,output_exact) : {Nat.add(Nat.mul(_,taps),U32.to_nat(tap_word)) == index : Nat}
    %Equal.sym(Nat,U32.to_nat(tap_word),tap,tap_exact) : {Nat.add(Nat.mul(output,taps),_) == index : Nat}
    Equal.sym(Nat,index,Nat.add(Nat.mul(output,taps),tap),DivisionOrder.quotient_exact(index,taps,taps_positive)) : {Bridge.group_spec_element(query) == index : Nat}}
  +actual_index = Equal.trans(Nat,WeightModel.group_query_element(query),Bridge.group_spec_element(query),index,Bridge.native_weight_element(query,taps_exact),spec_index)
  +work_equal = WeightModel.native_query_weight(query,n,size_positive,{==})
  %actual_index : {ListRead.at(~F32,0.0,_,weight_list(Parameter.work(xi,di,k,s,p,g),x,d)) == weight_value(xi,di,k,s,p,g,x,d,index) : F32}
  %work_equal : {ListRead.at(~F32,0.0,WeightModel.group_query_element(query),WeightModel.weight_values(_)) == weight_value(xi,di,k,s,p,g,x,d,index) : F32}
  %spec_index : {ListRead.at(~F32,0.0,WeightModel.group_query_element(query),WeightModel.weight_values(WeightModel.query_weight(WeightModel.native_batch_query(query,n,size_positive)))) == Spec.weight(~Tree.Tree<F32>,~Bridge.at,Bridge.group_spec_problem(query,n),_) : F32}
  Bridge.native_weight_spec(query,n,leaf,size_positive)

# ---- A part of the outputs is a slice of the weight gradient ----

# The two reads of one weight term.
def gradient_read(gradient: Tree.Tree<F32>,+outputs: Nat,+rows: Nat,+columns: Nat,+image: Nat,+output: Nat,+position: Nat) -> F32:
  Bridge.at(gradient,Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,outputs),output),rows),columns),position))

def input_read(input: Tree.Tree<F32>,+c: Nat,+h: Nat,+w: Nat,+s: Nat,+pad: Nat,+columns: Nat,+image: Nat,+t: Nat,+r: Nat,+channel: Nat,+position: Nat) -> F32:
  +y = Nat.add(r,Nat.mul(Nat.div(position,columns),s))
  +x = Nat.add(t,Nat.mul(Nat.mod(position,columns),s))
  Spec.sampled(~Tree.Tree<F32>,~Bridge.at,Nat.is_le(pad,y) && Nat.is_lt(y,Nat.add(pad,h)) && Nat.is_le(pad,x) && Nat.is_lt(x,Nat.add(pad,w)),input,
    Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,c),channel),h),w),Nat.mul(Nat.sub(y,pad),w)),Nat.sub(x,pad)))

def term_read(+n: Nat,+c: Nat,+o: Nat,+h: Nat,+w: Nat,+k: Nat,+s: Nat,+pad: Nat,+rows: Nat,+columns: Nat,+g: Nat,+input: Tree.Tree<F32>,+gradient: Tree.Tree<F32>,
  +image: Nat,+element: Nat,+position: Nat) ->
  {Spec.weight_term(~Tree.Tree<F32>,~Bridge.at,Spec.Problem{Spec.Shape{n,c,o,h,w,k,s,pad,rows,columns,g},input,gradient},image,element,position) ==
    (gradient_read(gradient,o,rows,columns,image,Nat.div(element,Nat.mul(c,Nat.mul(k,k))),position) *
      input_read(input,c,h,w,s,pad,columns,image,Nat.mod(element,k),Nat.mod(Nat.div(element,k),k),Nat.mod(Nat.div(element,Nat.mul(k,k)),c),position) : F32) : F32}:
  {==}

# An element first outputs later reads the same tap of the output first
# places later.
def shifted_taps(+first: Nat,+c: Nat,+k: Nat) -> {Nat.mul(first,Nat.mul(c,Nat.mul(k,k))) == Nat.mul(Nat.mul(Nat.mul(first,c),k),k) : Nat}:
  %RowMajor.assoc(first,c,Nat.mul(k,k)) : {_ == Nat.mul(Nat.mul(Nat.mul(first,c),k),k) : Nat}
  Equal.sym(Nat,Nat.mul(Nat.mul(Nat.mul(first,c),k),k),Nat.mul(Nat.mul(first,c),Nat.mul(k,k)),RowMajor.assoc(Nat.mul(first,c),k,k))

def shifted_column(+first: Nat,+c: Nat,+k: Nat,+element: Nat,+kernel_positive: {Nat.is_lt(0n,k) == True{} : Bool}) ->
  {Nat.mod(Nat.add(Nat.mul(first,Nat.mul(c,Nat.mul(k,k))),element),k) == Nat.mod(element,k) : Nat}:
  %Equal.sym(Nat,Nat.mul(first,Nat.mul(c,Nat.mul(k,k))),Nat.mul(Nat.mul(Nat.mul(first,c),k),k),shifted_taps(first,c,k)) :
    {Nat.mod(Nat.add(_,element),k) == Nat.mod(element,k) : Nat}
  DivisionOrder.remainder_shift(element,k,Nat.mul(Nat.mul(first,c),k),kernel_positive)

def shifted_row(+first: Nat,+c: Nat,+k: Nat,+element: Nat,+kernel_positive: {Nat.is_lt(0n,k) == True{} : Bool}) ->
  {Nat.mod(Nat.div(Nat.add(Nat.mul(first,Nat.mul(c,Nat.mul(k,k))),element),k),k) == Nat.mod(Nat.div(element,k),k) : Nat}:
  %Equal.sym(Nat,Nat.mul(first,Nat.mul(c,Nat.mul(k,k))),Nat.mul(Nat.mul(Nat.mul(first,c),k),k),shifted_taps(first,c,k)) :
    {Nat.mod(Nat.div(Nat.add(_,element),k),k) == Nat.mod(Nat.div(element,k),k) : Nat}
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(Nat.mul(Nat.mul(first,c),k),k),element),k),Nat.add(Nat.mul(Nat.mul(first,c),k),Nat.div(element,k)),
      DivisionOrder.quotient_shift(element,k,Nat.mul(Nat.mul(first,c),k),kernel_positive)) :
    {Nat.mod(_,k) == Nat.mod(Nat.div(element,k),k) : Nat}
  DivisionOrder.remainder_shift(Nat.div(element,k),k,Nat.mul(first,c),kernel_positive)

def shifted_channel(+first: Nat,+c: Nat,+k: Nat,+element: Nat,+square_positive: {Nat.is_lt(0n,Nat.mul(k,k)) == True{} : Bool},
  +channels_positive: {Nat.is_lt(0n,c) == True{} : Bool}) ->
  {Nat.mod(Nat.div(Nat.add(Nat.mul(first,Nat.mul(c,Nat.mul(k,k))),element),Nat.mul(k,k)),c) == Nat.mod(Nat.div(element,Nat.mul(k,k)),c) : Nat}:
  %RowMajor.assoc(first,c,Nat.mul(k,k)) : {Nat.mod(Nat.div(Nat.add(_,element),Nat.mul(k,k)),c) == Nat.mod(Nat.div(element,Nat.mul(k,k)),c) : Nat}
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(Nat.mul(first,c),Nat.mul(k,k)),element),Nat.mul(k,k)),Nat.add(Nat.mul(first,c),Nat.div(element,Nat.mul(k,k))),
      DivisionOrder.quotient_shift(element,Nat.mul(k,k),Nat.mul(first,c),square_positive)) :
    {Nat.mod(_,c) == Nat.mod(Nat.div(element,Nat.mul(k,k)),c) : Nat}
  DivisionOrder.remainder_shift(Nat.div(element,Nat.mul(k,k)),c,first,channels_positive)

# A gradient address as a row-major index of images of output rows.
def spec_index(+image: Nat,+outputs: Nat,+output: Nat,+rows: Nat,+columns: Nat,+position: Nat) ->
  {Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,outputs),output),rows),columns),position) == RowMajor.index(image,outputs,Nat.mul(rows,columns),output,position) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.mul(Nat.add(Nat.mul(image,outputs),output),rows),columns),Nat.mul(Nat.add(Nat.mul(image,outputs),output),Nat.mul(rows,columns)),
      RowMajor.assoc(Nat.add(Nat.mul(image,outputs),output),rows,columns)) :
    {Nat.add(_,position) == RowMajor.index(image,outputs,Nat.mul(rows,columns),output,position) : Nat}
  RowMajor.mixed(image,outputs,Nat.mul(rows,columns),output,position)

# The output gradient seen as images of output rows, as the parts copy it.
def slice_geometry(n: Nat,o: Nat,rows: Nat,columns: Nat) -> Bands.Geometry:
  Bands.Geometry{1,U32.from_nat(n),U32.from_nat(o),U32.from_nat(Nat.mul(rows,columns))}

# A pixel of the geometry is the gradient address of its image and row.
def pixel_index(+n: Nat,+o: Nat,+rows: Nat,+columns: Nat,+image: Nat,+row: Nat,+position: Nat,
  +outputs_exact: {U32.to_nat(U32.from_nat(o)) == o : Nat},+area_exact: {U32.to_nat(U32.from_nat(Nat.mul(rows,columns))) == Nat.mul(rows,columns) : Nat},
  +row_inside: {Nat.is_lt(row,o) == True{} : Bool},+outputs_positive: {Nat.is_lt(0n,o) == True{} : Bool}) ->
  {BandValues.pixel(slice_geometry(n,o,rows,columns),image,row,position) == Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),row),rows),columns),position) : Nat}:
  +small = Division.unique(row,o,0n,row,{==},outputs_positive,row_inside)
  +quotient = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(row,o),(0n,row),small)
  +remainder = Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(row,o),(0n,row),small)
  +images_native = U32.to_nat(U32.from_nat(n))
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(o)),o,outputs_exact) :
    {RowMajor.index(Nat.add(Nat.mul(Nat.div(row,_),images_native),image),_,U32.to_nat(U32.from_nat(Nat.mul(rows,columns))),Nat.mod(row,_),position) ==
      Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),row),rows),columns),position) : Nat}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Nat.mul(rows,columns))),Nat.mul(rows,columns),area_exact) :
    {RowMajor.index(Nat.add(Nat.mul(Nat.div(row,o),images_native),image),o,_,Nat.mod(row,o),position) ==
      Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),row),rows),columns),position) : Nat}
  %Equal.sym(Nat,Nat.div(row,o),0n,quotient) :
    {RowMajor.index(Nat.add(Nat.mul(_,images_native),image),o,Nat.mul(rows,columns),Nat.mod(row,o),position) ==
      Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),row),rows),columns),position) : Nat}
  %Equal.sym(Nat,Nat.mod(row,o),row,remainder) :
    {RowMajor.index(image,o,Nat.mul(rows,columns),_,position) == Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),row),rows),columns),position) : Nat}
  Equal.sym(Nat,Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),row),rows),columns),position),RowMajor.index(image,o,Nat.mul(rows,columns),row,position),
    spec_index(image,o,row,rows,columns,position))

# One cell of a part's gradient rows is the cell of the whole gradient its
# output lies at.
def slice_cell(+n: Nat,+o: Nat,+rows: Nat,+columns: Nat,+first: U32,+count: U32,+gradient: Tree.Tree<F32>,+band: Tree.Tree<F32>,+length: Nat,
  +image: Nat,+output: Nat,+position: Nat,
  +outputs_exact: {U32.to_nat(U32.from_nat(o)) == o : Nat},+area_exact: {U32.to_nat(U32.from_nat(Nat.mul(rows,columns))) == Nat.mul(rows,columns) : Nat},
  +row_inside: {Nat.is_lt(Nat.add(U32.to_nat(first),output),o) == True{} : Bool},+outputs_positive: {Nat.is_lt(0n,o) == True{} : Bool},
  cell: BandValues.At(~F32,~Tree.Tree<F32>,~BandValues.stored_value(~F32),gradient,slice_geometry(n,o,rows,columns),first,count,band,length,image,output,position)) ->
  {gradient_read(band,U32.to_nat(count),rows,columns,image,output,position) == gradient_read(gradient,o,rows,columns,image,Nat.add(U32.to_nat(first),output),position) : F32}:
  Shape.Both{bound,+read} = cell
  +part_address = Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(count)),output),rows),columns),position)
  +whole_address = Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),Nat.add(U32.to_nat(first),output)),rows),columns),position)
  +part_equal = Equal.trans(Nat,BandValues.position(slice_geometry(n,o,rows,columns),count,image,output,position),
    RowMajor.index(image,U32.to_nat(count),Nat.mul(rows,columns),output,position),part_address,
    Equal.cong(Nat,Nat,width => RowMajor.index(image,U32.to_nat(count),width,output,position),U32.to_nat(U32.from_nat(Nat.mul(rows,columns))),Nat.mul(rows,columns),area_exact),
    Equal.sym(Nat,part_address,RowMajor.index(image,U32.to_nat(count),Nat.mul(rows,columns),output,position),spec_index(image,U32.to_nat(count),output,rows,columns,position)))
  %part_equal : {Tree.read(F32,band,U32.from_nat(_)) == Tree.read(F32,gradient,U32.from_nat(whole_address)) : F32}
  %pixel_index(n,o,rows,columns,image,Nat.add(U32.to_nat(first),output),position,outputs_exact,area_exact,row_inside,outputs_positive) :
    {Tree.read(F32,band,U32.from_nat(BandValues.position(slice_geometry(n,o,rows,columns),count,image,output,position))) == Tree.read(F32,gradient,U32.from_nat(_)) : F32}
  read

# A part's weight problem beside the whole one: an element of the part is the
# element of the whole first outputs later.
type Slice is Data:
  Slice{n: Nat,c: Nat,o: Nat,h: Nat,w: Nat,k: Nat,s: Nat,pad: Nat,rows: Nat,columns: Nat,g: Nat,first: U32,count: U32,element: Nat,
    input: Tree.Tree<F32>,gradient: Tree.Tree<F32>,length: Nat,
    band: BandValues.Values(~F32,~Tree.Tree<F32>,~BandValues.stored_value(~F32),gradient,slice_geometry(n,o,rows,columns),
      FromWhole.model(~F32,slice_geometry(n,o,rows,columns),0.0,Bands.Rows{first,count},Buffer.Content{gradient,length})),
    images_exact: {U32.to_nat(U32.from_nat(n)) == n : Nat},outputs_exact: {U32.to_nat(U32.from_nat(o)) == o : Nat},
    area_exact: {U32.to_nat(U32.from_nat(Nat.mul(rows,columns))) == Nat.mul(rows,columns) : Nat},
    room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),o) == True{} : Bool},
    element_inside: {Nat.is_lt(element,Nat.mul(U32.to_nat(count),Nat.mul(c,Nat.mul(k,k)))) == True{} : Bool},
    kernel_positive: {Nat.is_lt(0n,k) == True{} : Bool},channels_positive: {Nat.is_lt(0n,c) == True{} : Bool},
    columns_positive: {Nat.is_lt(0n,columns) == True{} : Bool},group_positive: {Nat.is_lt(0n,g) == True{} : Bool}}

def slice_band(+n: Nat,+o: Nat,+rows: Nat,+columns: Nat,+first: U32,+count: U32,+gradient: Tree.Tree<F32>,+length: Nat) -> Tree.Tree<F32>:
  Buffer.content_tree(~F32,FromWhole.band(~F32,slice_geometry(n,o,rows,columns),0.0,Buffer.Content{gradient,length},first,count))

def slice_part(scope: Slice) -> Spec.Problem<Tree.Tree<F32>>:
  Slice{+n,+c,+o,+h,+w,+k,+s,+pad,+rows,+columns,+g,+first,+count,element,+input,+gradient,+length,band,images_exact,outputs_exact,area_exact,room,element_inside,
    kernel_positive,channels_positive,columns_positive,group_positive} = scope
  Spec.Problem{Spec.Shape{n,c,U32.to_nat(count),h,w,k,s,pad,rows,columns,g},input,slice_band(n,o,rows,columns,first,count,gradient,length)}

def slice_whole(scope: Slice) -> Spec.Problem<Tree.Tree<F32>>:
  Slice{+n,+c,+o,+h,+w,+k,+s,+pad,+rows,+columns,+g,first,count,element,+input,+gradient,length,band,images_exact,outputs_exact,area_exact,room,element_inside,
    kernel_positive,channels_positive,columns_positive,group_positive} = scope
  Spec.Problem{Spec.Shape{n,c,o,h,w,k,s,pad,rows,columns,g},input,gradient}

def slice_element(scope: Slice) -> Nat:
  Slice{n,c,o,h,w,k,s,pad,rows,columns,g,first,count,element,input,gradient,length,band,images_exact,outputs_exact,area_exact,room,element_inside,
    kernel_positive,channels_positive,columns_positive,group_positive} = scope
  element

def slice_shifted(scope: Slice) -> Nat:
  Slice{n,+c,o,h,w,+k,s,pad,rows,columns,g,+first,count,+element,input,gradient,length,band,images_exact,outputs_exact,area_exact,room,element_inside,
    kernel_positive,channels_positive,columns_positive,group_positive} = scope
  Nat.add(Nat.mul(U32.to_nat(first),Nat.mul(c,Nat.mul(k,k))),element)

def slice_images(scope: Slice) -> Nat:
  Slice{n,c,o,h,w,k,s,pad,rows,columns,g,first,count,element,input,gradient,length,band,images_exact,outputs_exact,area_exact,room,element_inside,
    kernel_positive,channels_positive,columns_positive,group_positive} = scope
  n

def slice_rows(scope: Slice) -> Nat:
  Slice{n,c,o,h,w,k,s,pad,rows,columns,g,first,count,element,input,gradient,length,band,images_exact,outputs_exact,area_exact,room,element_inside,
    kernel_positive,channels_positive,columns_positive,group_positive} = scope
  rows

def slice_columns(scope: Slice) -> Nat:
  Slice{n,c,o,h,w,k,s,pad,rows,columns,g,first,count,element,input,gradient,length,band,images_exact,outputs_exact,area_exact,room,element_inside,
    kernel_positive,channels_positive,columns_positive,group_positive} = scope
  columns

def slice_group(scope: Slice) -> Nat:
  Slice{n,c,o,h,w,k,s,pad,rows,columns,g,first,count,element,input,gradient,length,band,images_exact,outputs_exact,area_exact,room,element_inside,
    kernel_positive,channels_positive,columns_positive,group_positive} = scope
  g

def slice_term(+scope: Slice,+image: Nat,+position: Nat,+image_inside: {Nat.is_lt(image,slice_images(scope)) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,Nat.mul(slice_rows(scope),slice_columns(scope))) == True{} : Bool}) ->
  {Spec.weight_term(~Tree.Tree<F32>,~Bridge.at,slice_part(scope),image,slice_element(scope),position) ==
    Spec.weight_term(~Tree.Tree<F32>,~Bridge.at,slice_whole(scope),image,slice_shifted(scope),position) : F32}:
  Slice{+n,+c,+o,+h,+w,+k,+s,+pad,+rows,+columns,+g,+first,+count,+element,+input,+gradient,+length,+band,+images_exact,+outputs_exact,+area_exact,+room,+element_inside,
    +kernel_positive,+channels_positive,+columns_positive,+group_positive} = scope
  +square = Nat.mul(k,k)
  +taps = Nat.mul(c,square)
  +square_positive = Order.product_positive(k,k,kernel_positive,kernel_positive)
  +taps_positive = Order.product_positive(c,square,channels_positive,square_positive)
  +output = Nat.div(element,taps)
  +start = U32.to_nat(first)
  +shifted = Nat.add(Nat.mul(start,taps),element)
  +output_inside = DivisionOrder.quotient_below(element,U32.to_nat(count),taps,taps_positive,element_inside)
  +row_inside = Bounds.nat_trans(Nat.add(start,output),Nat.add(start,U32.to_nat(count)),o,
    Order.offset_lt(start,output,U32.to_nat(count),output_inside),room)
  +outputs_positive = Order.le_lt_trans(0n,Nat.add(start,output),o,Order.zero_le(Nat.add(start,output)),row_inside)
  +band_tree = slice_band(n,o,rows,columns,first,count,gradient,length)
  +band_length = U32.to_nat(Bands.local_count(slice_geometry(n,o,rows,columns),count))
  +reads = slice_cell(n,o,rows,columns,first,count,gradient,band_tree,band_length,image,output,position,outputs_exact,area_exact,row_inside,outputs_positive,
    BandValues.channels_cell(~F32,~Tree.Tree<F32>,~BandValues.stored_value(~F32),gradient,slice_geometry(n,o,rows,columns),first,count,band_tree,band_length,
      BandValues.channels(slice_geometry(n,o,rows,columns)),image,output,position,band,
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(n)),n,images_exact) : {Nat.is_lt(image,_) == True{} : Bool} image_inside,
      output_inside,
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(Nat.mul(rows,columns))),Nat.mul(rows,columns),area_exact) : {Nat.is_lt(position,_) == True{} : Bool} position_inside))
  +sample = input_read(input,c,h,w,s,pad,columns,image,Nat.mod(element,k),Nat.mod(Nat.div(element,k),k),Nat.mod(Nat.div(element,square),c),position)
  Equal.trans(F32,Spec.weight_term(~Tree.Tree<F32>,~Bridge.at,Spec.Problem{Spec.Shape{n,c,U32.to_nat(count),h,w,k,s,pad,rows,columns,g},input,band_tree},image,element,position),
    (gradient_read(band_tree,U32.to_nat(count),rows,columns,image,output,position) * sample : F32),
    Spec.weight_term(~Tree.Tree<F32>,~Bridge.at,Spec.Problem{Spec.Shape{n,c,o,h,w,k,s,pad,rows,columns,g},input,gradient},image,shifted,position),
    term_read(n,c,U32.to_nat(count),h,w,k,s,pad,rows,columns,g,input,band_tree,image,element,position),
    Equal.trans(F32,(gradient_read(band_tree,U32.to_nat(count),rows,columns,image,output,position) * sample : F32),
      (gradient_read(gradient,o,rows,columns,image,Nat.add(start,output),position) * sample : F32),
      Spec.weight_term(~Tree.Tree<F32>,~Bridge.at,Spec.Problem{Spec.Shape{n,c,o,h,w,k,s,pad,rows,columns,g},input,gradient},image,shifted,position),
      Equal.cong(F32,F32,read => (read * sample : F32),gradient_read(band_tree,U32.to_nat(count),rows,columns,image,output,position),
        gradient_read(gradient,o,rows,columns,image,Nat.add(start,output),position),reads),
      %Equal.sym(F32,Spec.weight_term(~Tree.Tree<F32>,~Bridge.at,Spec.Problem{Spec.Shape{n,c,o,h,w,k,s,pad,rows,columns,g},input,gradient},image,shifted,position),
          (gradient_read(gradient,o,rows,columns,image,Nat.div(shifted,taps),position) *
            input_read(input,c,h,w,s,pad,columns,image,Nat.mod(shifted,k),Nat.mod(Nat.div(shifted,k),k),Nat.mod(Nat.div(shifted,square),c),position) : F32),
          term_read(n,c,o,h,w,k,s,pad,rows,columns,g,input,gradient,image,shifted,position)) :
        {(gradient_read(gradient,o,rows,columns,image,Nat.add(start,output),position) * sample : F32) == _ : F32}
      %Equal.sym(Nat,Nat.div(shifted,taps),Nat.add(start,output),DivisionOrder.quotient_shift(element,taps,start,taps_positive)) :
        {(gradient_read(gradient,o,rows,columns,image,Nat.add(start,output),position) * sample : F32) ==
          (gradient_read(gradient,o,rows,columns,image,_,position) *
            input_read(input,c,h,w,s,pad,columns,image,Nat.mod(shifted,k),Nat.mod(Nat.div(shifted,k),k),Nat.mod(Nat.div(shifted,square),c),position) : F32) : F32}
      %Equal.sym(Nat,Nat.mod(shifted,k),Nat.mod(element,k),shifted_column(start,c,k,element,kernel_positive)) :
        {(gradient_read(gradient,o,rows,columns,image,Nat.add(start,output),position) * sample : F32) ==
          (gradient_read(gradient,o,rows,columns,image,Nat.add(start,output),position) *
            input_read(input,c,h,w,s,pad,columns,image,_,Nat.mod(Nat.div(shifted,k),k),Nat.mod(Nat.div(shifted,square),c),position) : F32) : F32}
      %Equal.sym(Nat,Nat.mod(Nat.div(shifted,k),k),Nat.mod(Nat.div(element,k),k),shifted_row(start,c,k,element,kernel_positive)) :
        {(gradient_read(gradient,o,rows,columns,image,Nat.add(start,output),position) * sample : F32) ==
          (gradient_read(gradient,o,rows,columns,image,Nat.add(start,output),position) *
            input_read(input,c,h,w,s,pad,columns,image,Nat.mod(element,k),_,Nat.mod(Nat.div(shifted,square),c),position) : F32) : F32}
      %Equal.sym(Nat,Nat.mod(Nat.div(shifted,square),c),Nat.mod(Nat.div(element,square),c),shifted_channel(start,c,k,element,square_positive,channels_positive)) :
        {(gradient_read(gradient,o,rows,columns,image,Nat.add(start,output),position) * sample : F32) ==
          (gradient_read(gradient,o,rows,columns,image,Nat.add(start,output),position) *
            input_read(input,c,h,w,s,pad,columns,image,Nat.mod(element,k),Nat.mod(Nat.div(element,k),k),_,position) : F32) : F32}
      {==}))

# The weight gradient of a part is the slice of the whole one at its outputs.
def slice_weight(+scope: Slice) ->
  {Spec.weight(~Tree.Tree<F32>,~Bridge.at,slice_part(scope),slice_element(scope)) == Spec.weight(~Tree.Tree<F32>,~Bridge.at,slice_whole(scope),slice_shifted(scope)) : F32}:
  Slice{+n,+c,+o,+h,+w,+k,+s,+pad,+rows,+columns,+g,+first,+count,+element,+input,+gradient,+length,+band,+images_exact,+outputs_exact,+area_exact,+room,+element_inside,
    +kernel_positive,+channels_positive,+columns_positive,+group_positive} = scope
  +scope = {Slice{n,c,o,h,w,k,s,pad,rows,columns,g,first,count,element,input,gradient,length,band,images_exact,outputs_exact,area_exact,room,element_inside,
    kernel_positive,channels_positive,columns_positive,group_positive} : Slice}
  Equal.cong(List<&2,F32>,F32,values => Grouped.tree(values),
    Spec.image_values(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~Bridge.at),n,0n,element,slice_part(scope),Grouped.ranges(rows,columns,g)),
    Spec.image_values(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~Bridge.at),n,0n,slice_shifted(scope),slice_whole(scope),Grouped.ranges(rows,columns,g)),
    Readers.images_agree(~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~Bridge.at),~Tree.Tree<F32>,~Spec.weight_term(~Tree.Tree<F32>,~Bridge.at),
      ~Slice,~slice_part,~slice_whole,~slice_element,~slice_shifted,~slice_images,~slice_rows,~slice_columns,~slice_group,~slice_term,n,0n,scope,Order.reflexive(n),
      GroupedRanges.ranges_bounded(rows,columns,g,columns_positive,group_positive)))

# ---- The contract of the results ----

# A weight of storage holding the readers' values is the readers' weight
# gradient there.
def weight_agreement(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+input_context: Input,+gradient_context: Gradient,
  +xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +x: Buffer.Model<F32>,+xs: List<&2,U32>,+x_facts: GatherValues.Facts(~F32,~Input,~input_value,input_context,xs,Buffer.content(~F32,x)),+x_cells: Readers.ImageCells(xs,Some{xi}),
  +d: Buffer.Model<F32>,+ds: List<&2,U32>,+d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,ds,Buffer.content(~F32,d)),+d_cells: Readers.ImageCells(ds,Some{di}),
  +positive: WeightPositive<xi,di,k,s,g>,+count: U32,+counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),count>,+values: List<&2,F32>,
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool},+numeric: {ListRead.at(~F32,0.0,index,values) == weight_value(xi,di,k,s,p,g,x,d,index) : F32}) ->
  {ListRead.at(~F32,0.0,index,values) ==
    Readers.weight_at(~Input,~input_value,~Gradient,~gradient_value)(Readers.WeightRead{input_context,gradient_context,Some{xi},Some{di},k,s,p,g},U32.from_nat(index)) : F32}:
  Parameter.ImageGeometry{+n,+c,+h,+w} = xi
  Parameter.ImageGeometry{+dn,+o,+rows,+columns} = di
  Buffer.Model{+xt,+xl,+xc} = x
  Shape.Both{Contract.WellFormed{+x_length,x_ready},+x_values} = x_facts
  Buffer.Model{+dt,+dl,+dc} = d
  Shape.Both{Contract.WellFormed{+d_length,d_ready},+d_values} = d_facts
  WeightPositive{+xp,+dp,+kp,sp,+gp,+images} = positive
  Count.Count{+exact,+bounded} = counted
  +xc_true = Logic.left((U32.is_lt(0,n) && U32.is_lt(0,c)) && U32.is_lt(0,h),U32.is_lt(0,w),xp)
  +nc_true = Logic.left(U32.is_lt(0,n) && U32.is_lt(0,c),U32.is_lt(0,h),xc_true)
  +cp_word = Logic.right(U32.is_lt(0,n),U32.is_lt(0,c),nc_true)
  +columns_word = Logic.right((U32.is_lt(0,dn) && U32.is_lt(0,o)) && U32.is_lt(0,rows),U32.is_lt(0,columns),dp)
  +channels_positive = {%Comparison.lt(0,c) : {_ == True{} : Bool} cp_word : {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool}}
  +columns_positive = {%Comparison.lt(0,columns) : {_ == True{} : Bool} columns_word : {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool}}
  +kernel_positive = {%Comparison.lt(0,k) : {_ == True{} : Bool} kp : {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool}}
  +group_positive = {%Comparison.lt(0,g) : {_ == True{} : Bool} gp : {Nat.is_lt(0n,U32.to_nat(g)) == True{} : Bool}}
  +same_images = Order.equal_of_is_eq(U32.to_nat(n),U32.to_nat(dn),%Comparison.eq(n,dn) : {_ == True{} : Bool} images)
  +x_room = {%Equal.trans(Nat,xl,ShapeLaws.elements(xs),Readers.cells(xi),x_length,x_cells) : {Nat.is_le(_,xl) == True{} : Bool} Order.reflexive(xl) :
    {Nat.is_le(Readers.cells(xi),xl) == True{} : Bool}}
  +d_room = {%Equal.sym(Nat,U32.to_nat(n),U32.to_nat(dn),same_images) : {Nat.is_le(Nat.mul(Nat.mul(_,U32.to_nat(o)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),dl) == True{} : Bool}
    %Equal.trans(Nat,dl,ShapeLaws.elements(ds),Readers.cells(di),d_length,d_cells) : {Nat.is_le(_,dl) == True{} : Bool} Order.reflexive(dl) :
    {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(n),U32.to_nat(o)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),dl) == True{} : Bool}}
  +element_inside = {%weight_shape_count(o,c,k) : {Nat.is_lt(index,_) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements([o,c,k,k]),U32.to_nat(count),exact) : {Nat.is_lt(index,_) == True{} : Bool} inside :
    {Nat.is_lt(index,Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))) == True{} : Bool}}
  +scope = {Readers.WeightScope{U32.to_nat(n),U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p),U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(g),
    xt,xl,input_context,x_values,x_room,dt,dl,gradient_context,d_values,d_room,index,element_inside,kernel_positive,channels_positive,columns_positive,group_positive} :
    Readers.WeightScope<Input,input_value,Gradient,gradient_value>}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,FullRange.roundtrip(index,count,Bounds.strict_le(index,U32.to_nat(count),inside))) :
    {ListRead.at(~F32,0.0,index,values) ==
      Spec.weight(~Readers.Operand<Input,Gradient>,~Readers.operand_at(~Input,~input_value,~Gradient,~gradient_value),
        Spec.Problem{Bridge.shape(Parameter.work(xi,di,k,s,p,g)),Readers.InputOperand{input_context},Readers.GradientOperand{gradient_context}},_) : F32}
  Equal.trans(F32,ListRead.at(~F32,0.0,index,values),
    weight_value(xi,di,k,s,p,g,Buffer.Model{xt,xl,xc},Buffer.Model{dt,dl,dc},index),
    Spec.weight(~Readers.Operand<Input,Gradient>,~Readers.operand_at(~Input,~input_value,~Gradient,~gradient_value),Readers.weight_read(~Input,~input_value,~Gradient,~gradient_value,scope),index),
    numeric,Readers.weight_agrees(~Input,~input_value,~Gradient,~gradient_value,scope))

# ---- The weight gradient of the parts ----

def weight_taps_exact(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,geometry: WeightGeometry<xi,di,k,s,p>) ->
  {U32.to_nat(WeightPacking.geometry_taps(weight_geometry(xi,di,k,s,p))) == weight_tap_count(xi,k) : Nat}:
  WeightGeometry{domain,taps_exact} = geometry
  taps_exact

def positive_images(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+g: U32,positive: WeightPositive<xi,di,k,s,g>) ->
  {U32.is_eq(image_count(xi),image_count(di)) == True{} : Bool}:
  WeightPositive{input,gradient,kernel,stride,group,images} = positive
  images

# The weight gradient holds outputs times taps values.
def weight_elements(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+positive: WeightPositive<xi,di,k,s,g>,
  +count: U32,+counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),count>) ->
  {Nat.mul(U32.to_nat(WeightKernel.outputs(Parameter.work(xi,di,k,s,p,g))),U32.to_nat(WeightKernel.taps(Parameter.work(xi,di,k,s,p,g)))) == ShapeLaws.elements(weight_shape(xi,di,k)) : Nat}:
  Parameter.ImageGeometry{+n,+c,+h,+w} = xi
  Parameter.ImageGeometry{+dn,+o,+oh,+ow} = di
  +taps_exact = weight_taps_exact(xi,di,k,s,p,weight_geometry_ready(xi,di,k,s,p,g,positive,count,counted))
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(U32.to_nat(o))),U32.to_nat(o),Equal.cong(U32,Nat,word => U32.to_nat(word),U32.from_nat(U32.to_nat(o)),o,Arithmetic.native_roundtrip(o))) :
    {Nat.mul(_,U32.to_nat(WeightKernel.taps(Parameter.work(xi,di,k,s,p,g)))) == ShapeLaws.elements([o,c,k,k]) : Nat}
  %Equal.sym(Nat,U32.to_nat(WeightPacking.geometry_taps(weight_geometry(xi,di,k,s,p))),weight_tap_count(xi,k),taps_exact) :
    {Nat.mul(U32.to_nat(o),_) == ShapeLaws.elements([o,c,k,k]) : Nat}
  Equal.sym(Nat,ShapeLaws.elements([o,c,k,k]),Nat.mul(U32.to_nat(o),weight_tap_count(xi,k)),weight_shape_count(o,c,k))

# The count of the weight shape is the count the kernel writes.
def work_count(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+positive: WeightPositive<xi,di,k,s,g>,
  +count: U32,+counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),count>) ->
  Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),KParts.count(Parameter.work(xi,di,k,s,p,g))>:
  Count.Count{+exact,+bounded} = counted
  +work = Parameter.work(xi,di,k,s,p,g)
  +outputs = WeightKernel.outputs(work)
  +taps = WeightKernel.taps(work)
  +elements = ShapeLaws.elements(weight_shape(xi,di,k))
  +same = {
    %Equal.sym(U32,(outputs * taps : U32),U32.from_nat(Nat.mul(U32.to_nat(outputs),U32.to_nat(taps))),Multiplication.native_multiply(outputs,taps)) : {_ == count : U32}
    %Equal.sym(Nat,Nat.mul(U32.to_nat(outputs),U32.to_nat(taps)),elements,weight_elements(xi,di,k,s,p,g,positive,count,Count.Count{exact,bounded})) : {U32.from_nat(_) == count : U32}
    %Equal.sym(Nat,elements,U32.to_nat(count),exact) : {U32.from_nat(_) == count : U32}
    Arithmetic.native_roundtrip(count) : {KParts.count(work) == count : U32}}
  %Equal.sym(U32,KParts.count(work),count,same) : Count.Count<elements,_>
  Count.Count{exact,bounded}

def weight_counted(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+positive: WeightPositive<xi,di,k,s,g>,
  found: Maybe<&2,U32>,+found_equal: {Tensors.count_of(weight_shape(xi,di,k)) == found : Maybe<&2,U32>},fits: {Parameter.fits(found) == True{} : Bool}) ->
  Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),KParts.count(Parameter.work(xi,di,k,s,p,g))>:
  match found:
    case None{}: Empty.absurd(Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),KParts.count(Parameter.work(xi,di,k,s,p,g))>,Words.false_true(fits))
    case Some{+count}: work_count(xi,di,k,s,p,g,positive,count,Count.count_of(weight_shape(xi,di,k),count,found_equal))

# Every output fits a half of the words.
def outputs_fit(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+positive: WeightPositive<xi,di,k,s,g>,
  +counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),KParts.count(Parameter.work(xi,di,k,s,p,g))>) ->
  {U32.is_le(WeightKernel.outputs(Parameter.work(xi,di,k,s,p,g)),2147483648) == True{} : Bool}:
  Parameter.ImageGeometry{+n,+c,+h,+w} = xi
  Parameter.ImageGeometry{+dn,+o,+oh,+ow} = di
  WeightPositive{+xp,+dp,+kp,+sp,+gp,+images} = positive
  Count.Count{+exact,+bounded} = counted
  +work = Parameter.work(xi,di,k,s,p,g)
  +outputs = WeightKernel.outputs(work)
  +taps = WeightKernel.taps(work)
  +capacity = Machine.capacity(Default.limits())
  +facts = {WeightPositive{xp,dp,kp,sp,gp,images} : WeightPositive<xi,di,k,s,g>}
  +taps_exact = weight_taps_exact(xi,di,k,s,p,weight_geometry_ready(xi,di,k,s,p,g,facts,KParts.count(work),Count.Count{exact,bounded}))
  +channels_word = Logic.right(U32.is_lt(0,n),U32.is_lt(0,c),Logic.left(U32.is_lt(0,n) && U32.is_lt(0,c),U32.is_lt(0,h),Logic.left((U32.is_lt(0,n) && U32.is_lt(0,c)) && U32.is_lt(0,h),U32.is_lt(0,w),xp)))
  +channels_positive = {%Comparison.lt(0,c) : {_ == True{} : Bool} channels_word : {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool}}
  +kernel_positive = {%Comparison.lt(0,k) : {_ == True{} : Bool} kp : {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool}}
  +taps_positive = {%Equal.sym(Nat,U32.to_nat(WeightPacking.geometry_taps(weight_geometry(xi,di,k,s,p))),weight_tap_count(xi,k),taps_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)),channels_positive,Order.product_positive(U32.to_nat(k),U32.to_nat(k),kernel_positive,kernel_positive)) :
    {Nat.is_lt(0n,U32.to_nat(taps)) == True{} : Bool}}
  +product = Nat.mul(U32.to_nat(outputs),U32.to_nat(taps))
  +product_room = {%Equal.sym(Nat,product,U32.to_nat(KParts.count(work)),
      Equal.trans(Nat,product,ShapeLaws.elements([o,c,k,k]),U32.to_nat(KParts.count(work)),weight_elements(xi,di,k,s,p,g,facts,KParts.count(work),Count.Count{exact,bounded}),exact)) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Bounds.as_nat_le(KParts.count(work),capacity,bounded) : {Nat.is_le(product,U32.to_nat(capacity)) == True{} : Bool}}
  Encoded.native_le_transitive(outputs,capacity,2147483648,
    %Equal.sym(Bool,U32.is_le(outputs,capacity),Nat.is_le(U32.to_nat(outputs),U32.to_nat(capacity)),Comparison.le(outputs,capacity)) : {_ == True{} : Bool}
    Order.transitive(U32.to_nat(outputs),product,U32.to_nat(capacity),RowMajor.product_covers(U32.to_nat(outputs),U32.to_nat(taps),taps_positive),product_room),
    Limits.capacity_fits(Default.limits()))

# The output gradient seen as images of output rows: its cells, and the
# cells of one row of every image.
def work_cells(work: Kernel.Work) -> Nat:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Nat.mul(Nat.mul(n,o),Nat.mul(rows,columns))

def work_area(work: Kernel.Work) -> Nat:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Nat.mul(rows,columns)

# The rows of the output gradient fit the machine, and the cells of a row
# are exact as a word.
type GradientRoom<-work: Kernel.Work> is Data:
  GradientRoom{domain: BandGather.Domain<KParts.gradient_geometry(work),Default.limits(),Machine.capacity(Default.limits())>,
    area_exact: {U32.to_nat(U32.from_nat(work_area(work))) == work_area(work) : Nat}}

def gradient_room(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+capacity: U32,+positive: WeightPositive<xi,di,k,s,g>,
  +cells_room: {Nat.is_le(work_cells(Parameter.work(xi,di,k,s,p,g)),U32.to_nat(capacity)) == True{} : Bool},
  +capacity_bounded: {U32.is_le(capacity,Machine.capacity(Default.limits())) == True{} : Bool}) -> GradientRoom<Parameter.work(xi,di,k,s,p,g)>:
  Parameter.ImageGeometry{+n,+c,+h,+w} = xi
  Parameter.ImageGeometry{+dn,+o,+rows,+columns} = di
  WeightPositive{+xp,+dp,kp,sp,gp,images} = positive
  +images_word = Logic.left(U32.is_lt(0,n),U32.is_lt(0,c),Logic.left(U32.is_lt(0,n) && U32.is_lt(0,c),U32.is_lt(0,h),Logic.left((U32.is_lt(0,n) && U32.is_lt(0,c)) && U32.is_lt(0,h),U32.is_lt(0,w),xp)))
  +rows_first = Logic.left((U32.is_lt(0,dn) && U32.is_lt(0,o)) && U32.is_lt(0,rows),U32.is_lt(0,columns),dp)
  +outputs_word = Logic.right(U32.is_lt(0,dn),U32.is_lt(0,o),Logic.left(U32.is_lt(0,dn) && U32.is_lt(0,o),U32.is_lt(0,rows),rows_first))
  +rows_word = Logic.right(U32.is_lt(0,dn) && U32.is_lt(0,o),U32.is_lt(0,rows),rows_first)
  +columns_word = Logic.right((U32.is_lt(0,dn) && U32.is_lt(0,o)) && U32.is_lt(0,rows),U32.is_lt(0,columns),dp)
  +images_positive = {%Comparison.lt(0,n) : {_ == True{} : Bool} images_word : {Nat.is_lt(0n,U32.to_nat(n)) == True{} : Bool}}
  +outputs_positive = {%Comparison.lt(0,o) : {_ == True{} : Bool} outputs_word : {Nat.is_lt(0n,U32.to_nat(o)) == True{} : Bool}}
  +rows_positive = {%Comparison.lt(0,rows) : {_ == True{} : Bool} rows_word : {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}}
  +columns_positive = {%Comparison.lt(0,columns) : {_ == True{} : Bool} columns_word : {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool}}
  +images = U32.to_nat(n)
  +outputs = U32.to_nat(o)
  +area = Nat.mul(U32.to_nat(rows),U32.to_nat(columns))
  +plane = Nat.mul(images,outputs)
  +cells = Nat.mul(plane,area)
  +cap = U32.to_nat(capacity)
  +area_room = Order.transitive(area,cells,cap,
    %RowMajor.multiply_commutative(area,plane) : {Nat.is_le(area,_) == True{} : Bool}
    RowMajor.product_covers(area,plane,Order.product_positive(images,outputs,images_positive,outputs_positive)),cells_room)
  +area_exact = FullRange.roundtrip(area,capacity,area_room)
  +cells_exact = FullRange.roundtrip(cells,capacity,cells_room)
  +images_exact = Equal.cong(U32,Nat,word => U32.to_nat(word),U32.from_nat(images),n,Arithmetic.native_roundtrip(n))
  +outputs_exact = Equal.cong(U32,Nat,word => U32.to_nat(word),U32.from_nat(outputs),o,Arithmetic.native_roundtrip(o))
  +geometry = slice_geometry(images,outputs,U32.to_nat(rows),U32.to_nat(columns))
  +count = U32.from_nat(cells)
  +elements_equal = {
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(images)),images,images_exact) :
      {Nat.mul(1n,Nat.mul(_,Nat.mul(U32.to_nat(U32.from_nat(outputs)),U32.to_nat(U32.from_nat(area))))) == cells : Nat}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(outputs)),outputs,outputs_exact) :
      {Nat.mul(1n,Nat.mul(images,Nat.mul(_,U32.to_nat(U32.from_nat(area))))) == cells : Nat}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(area)),area,area_exact) :
      {Nat.mul(1n,Nat.mul(images,Nat.mul(outputs,_))) == cells : Nat}
    Equal.trans(Nat,Nat.mul(1n,Nat.mul(images,Nat.mul(outputs,area))),Nat.mul(images,Nat.mul(outputs,area)),cells,
      Equal.trans(Nat,Nat.mul(1n,Nat.mul(images,Nat.mul(outputs,area))),Nat.mul(Nat.mul(images,Nat.mul(outputs,area)),1n),Nat.mul(images,Nat.mul(outputs,area)),
        RowMajor.multiply_commutative(1n,Nat.mul(images,Nat.mul(outputs,area))),Order.times_one(Nat.mul(images,Nat.mul(outputs,area)))),
      Equal.sym(Nat,cells,Nat.mul(images,Nat.mul(outputs,area)),RowMajor.assoc(images,outputs,area))) :
    {BandGather.elements(geometry) == cells : Nat}}
  +counted = {Count.Count{
      Equal.trans(Nat,ShapeLaws.elements(BandCount.extents(geometry)),BandGather.elements(geometry),U32.to_nat(count),BandCount.domain_elements(geometry),
        Equal.trans(Nat,BandGather.elements(geometry),cells,U32.to_nat(count),elements_equal,Equal.sym(Nat,U32.to_nat(count),cells,cells_exact))),
      Encoded.native_le_transitive(count,capacity,Machine.capacity(Default.limits()),
        %Equal.sym(Bool,U32.is_le(count,capacity),Nat.is_le(U32.to_nat(count),cap),Comparison.le(count,capacity)) : {_ == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(count),cells,cells_exact) : {Nat.is_le(_,cap) == True{} : Bool} cells_room,
        capacity_bounded)} :
    Count.Count<ShapeLaws.elements(BandCount.extents(geometry)),count>}
  GradientRoom{
    BandCount.evidence(geometry,count,
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(images)),images,images_exact) : {Nat.is_lt(0n,_) == True{} : Bool} images_positive,
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(outputs)),outputs,outputs_exact) : {Nat.is_lt(0n,_) == True{} : Bool} outputs_positive,
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(area)),area,area_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.product_positive(U32.to_nat(rows),U32.to_nat(columns),rows_positive,columns_positive),
      counted),
    area_exact}

# The owner of the output gradient holds its cells.
def gradient_room_of(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+positive: WeightPositive<xi,di,k,s,g>,
  +gradient: Buffer.Model<F32>,ready: Buffer.Sized<F32,gradient>,
  +cells_length: {Nat.is_le(work_cells(Parameter.work(xi,di,k,s,p,g)),Buffer.length(~F32,gradient)) == True{} : Bool}) -> GradientRoom<Parameter.work(xi,di,k,s,p,g)>:
  Buffer.Sized{+depth_ok,+room} = ready
  gradient_room(xi,di,k,s,p,g,Tree.capacity(F32,Buffer.tree(~F32,gradient)),positive,
    Order.transitive(work_cells(Parameter.work(xi,di,k,s,p,g)),Buffer.length(~F32,gradient),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,gradient))),cells_length,room),
    Buffer.capacity_allocatable(~F32,gradient,depth_ok))

# A call whose outputs the parts compute: what every part reads.
type WeightParts is Data:
  WeightParts{xi: Parameter.ImageGeometry,di: Parameter.ImageGeometry,k: U32,s: U32,p: U32,g: U32,x: Buffer.Model<F32>,d: Buffer.Model<F32>,
    positive: WeightPositive<xi,di,k,s,g>,
    counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),KParts.count(Parameter.work(xi,di,k,s,p,g))>,
    room: GradientRoom<Parameter.work(xi,di,k,s,p,g)>}

def parts_work(+scope: WeightParts) -> Kernel.Work:
  WeightParts{xi,di,k,s,p,g,x,d,positive,counted,room} = scope
  Parameter.work(xi,di,k,s,p,g)

def parts_fill(+scope: WeightParts) -> F32:
  0.0

def parts_limit(+scope: WeightParts) -> Nat:
  U32.to_nat(WeightKernel.outputs(parts_work(scope)))

def parts_view(+scope: WeightParts) -> BandEach.BuiltScope<Kernel.Work,KParts.OwnersSeen>:
  WeightParts{xi,di,k,s,p,g,x,d,positive,counted,room} = scope
  BandEach.BuiltScope{Parameter.work(xi,di,k,s,p,g),KParts.OwnersSeen{Buffer.content(~F32,x),Buffer.content(~F32,d)}}

def parts_before(+scope: WeightParts) -> KParts.OwnersSeen:
  WeightParts{xi,di,k,s,p,g,x,d,positive,counted,room} = scope
  KParts.OwnersSeen{Buffer.content(~F32,x),Buffer.content(~F32,d)}

def parts_value(+scope: WeightParts,+index: Nat) -> F32:
  WeightParts{+xi,+di,+k,+s,+p,+g,x,d,positive,counted,room} = scope
  weight_value(xi,di,k,s,p,g,x,d,index)

# A part reads the weight gradient of its outputs: the outputs first
# onwards of the whole call.
def parts_read(+scope: WeightParts,+part: KParts.OwnersModel,+first: U32,+rows: U32,
  +fact: BandModel.viewed(~KParts.OwnersModel,~BandEach.BuiltScope<Kernel.Work,KParts.OwnersSeen>,
    ~BandEach.built_piece(~KParts.OwnersModel,~KParts.OwnersSeen,~KParts.OwnersSeen,~Kernel.Work,~KParts.seen_model,~KParts.piece),~WeightParts,~parts_view,scope,part,first,rows),
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),parts_limit(scope)) == True{} : Bool},
  +local: Nat,+inside: {Nat.is_lt(local,Nat.mul(U32.to_nat(rows),KParts.work_width(parts_work(scope)))) == True{} : Bool}) ->
  {ListRead.at(~F32,parts_fill(scope),local,KParts.owner_list(parts_work(scope),part,first,rows)) ==
    parts_value(scope,Nat.add(Nat.mul(U32.to_nat(first),KParts.work_width(parts_work(scope))),local)) : F32}:
  WeightParts{+xi,+di,+k,+s,+p,+g,+x,+d,+positive,+counted,+gradient_room} = scope
  Parameter.ImageGeometry{+n,+c,+h,+w} = xi
  Parameter.ImageGeometry{+dn,+o,+oh,+ow} = di
  Buffer.Model{+xt,+xl,+xc} = x
  Buffer.Model{+dt,+dl,+dc} = d
  WeightPositive{+xp,+dp,+kp,+sp,+gp,+images} = positive
  Count.Count{+exact,+bounded} = counted
  GradientRoom{+domain,+area_exact} = gradient_room
  KParts.OwnersModel{+input,+gradient} = part
  +part_di = {Parameter.ImageGeometry{dn,rows,oh,ow} : Parameter.ImageGeometry}
  +work = Parameter.work(xi,di,k,s,p,g)
  +part_work = Parameter.work(xi,part_di,k,s,p,g)
  +taps = weight_tap_count(xi,k)
  +width = U32.to_nat(WeightKernel.taps(work))
  +capacity = Machine.capacity(Default.limits())
  +taps_exact = weight_taps_exact(xi,di,k,s,p,weight_geometry_ready(xi,di,k,s,p,g,WeightPositive{xp,dp,kp,sp,gp,images},KParts.count(work),Count.Count{exact,bounded}))
  +kernel_positive = {%Comparison.lt(0,k) : {_ == True{} : Bool} kp : {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool}}
  +channels_word = Logic.right(U32.is_lt(0,n),U32.is_lt(0,c),Logic.left(U32.is_lt(0,n) && U32.is_lt(0,c),U32.is_lt(0,h),Logic.left((U32.is_lt(0,n) && U32.is_lt(0,c)) && U32.is_lt(0,h),U32.is_lt(0,w),xp)))
  +channels_positive = {%Comparison.lt(0,c) : {_ == True{} : Bool} channels_word : {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool}}
  +columns_word = Logic.right((U32.is_lt(0,dn) && U32.is_lt(0,o)) && U32.is_lt(0,oh),U32.is_lt(0,ow),dp)
  +columns_positive = {%Comparison.lt(0,ow) : {_ == True{} : Bool} columns_word : {Nat.is_lt(0n,U32.to_nat(ow)) == True{} : Bool}}
  +group_positive = {%Comparison.lt(0,g) : {_ == True{} : Bool} gp : {Nat.is_lt(0n,U32.to_nat(g)) == True{} : Bool}}
  +outputs_word = Logic.right(U32.is_lt(0,dn),U32.is_lt(0,o),Logic.left(U32.is_lt(0,dn) && U32.is_lt(0,o),U32.is_lt(0,oh),Logic.left((U32.is_lt(0,dn) && U32.is_lt(0,o)) && U32.is_lt(0,oh),U32.is_lt(0,ow),dp)))
  +outputs_positive = {%Comparison.lt(0,o) : {_ == True{} : Bool} outputs_word : {Nat.is_lt(0n,U32.to_nat(o)) == True{} : Bool}}
  +taps_positive = Order.product_positive(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)),channels_positive,Order.product_positive(U32.to_nat(k),U32.to_nat(k),kernel_positive,kernel_positive))
  +rows_positive = RowMajor.left_factor_positive(U32.to_nat(rows),width,Order.positive_bound(local,Nat.mul(U32.to_nat(rows),width),inside))
  +rows_word = {%Equal.sym(Bool,U32.is_lt(0,rows),Nat.is_lt(0n,U32.to_nat(rows)),Comparison.lt(0,rows)) : {_ == True{} : Bool} rows_positive : {U32.is_lt(0,rows) == True{} : Bool}}
  +part_gradient = {%Equal.sym(Bool,U32.is_lt(0,rows),True{},rows_word) : {((U32.is_lt(0,dn) && _) && U32.is_lt(0,oh)) && U32.is_lt(0,ow) == True{} : Bool}
    %outputs_word : {((U32.is_lt(0,dn) && _) && U32.is_lt(0,oh)) && U32.is_lt(0,ow) == True{} : Bool} dp : {Parameter.positive(part_di) == True{} : Bool}}
  +part_positive = {WeightPositive{xp,part_gradient,kp,sp,gp,images} : WeightPositive<xi,part_di,k,s,g>}
  +images_exact = Equal.cong(U32,Nat,word => U32.to_nat(word),U32.from_nat(U32.to_nat(n)),n,Arithmetic.native_roundtrip(n))
  +outputs_exact = Equal.cong(U32,Nat,word => U32.to_nat(word),U32.from_nat(U32.to_nat(o)),o,Arithmetic.native_roundtrip(o))
  +outputs_room = {%outputs_exact : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool} room :
    {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(o)) == True{} : Bool}}
  +rows_room = Order.drop_left_le(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(o),outputs_room)
  +whole_exact = Equal.trans(Nat,Nat.mul(U32.to_nat(o),taps),ShapeLaws.elements([o,c,k,k]),U32.to_nat(KParts.count(work)),
    Equal.sym(Nat,ShapeLaws.elements([o,c,k,k]),Nat.mul(U32.to_nat(o),taps),weight_shape_count(o,c,k)),exact)
  +whole_room = {%Equal.sym(Nat,Nat.mul(U32.to_nat(o),taps),U32.to_nat(KParts.count(work)),whole_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Bounds.as_nat_le(KParts.count(work),capacity,bounded) : {Nat.is_le(Nat.mul(U32.to_nat(o),taps),U32.to_nat(capacity)) == True{} : Bool}}
  +part_value = Nat.mul(U32.to_nat(rows),taps)
  +part_room = Order.transitive(part_value,Nat.mul(U32.to_nat(o),taps),U32.to_nat(capacity),RowMajor.mul_le(U32.to_nat(rows),U32.to_nat(o),taps,rows_room),whole_room)
  +part_exact = FullRange.roundtrip(part_value,capacity,part_room)
  +part_count = U32.from_nat(part_value)
  +part_counted = {Count.Count{
      Equal.trans(Nat,ShapeLaws.elements([rows,c,k,k]),part_value,U32.to_nat(part_count),weight_shape_count(rows,c,k),Equal.sym(Nat,U32.to_nat(part_count),part_value,part_exact)),
      %Equal.sym(Bool,U32.is_le(part_count,capacity),Nat.is_le(U32.to_nat(part_count),U32.to_nat(capacity)),Comparison.le(part_count,capacity)) : {_ == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(part_count),part_value,part_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} part_room} :
    Count.Count<ShapeLaws.elements(weight_shape(xi,part_di,k)),part_count>}
  +element_inside = {%taps_exact : {Nat.is_lt(local,Nat.mul(U32.to_nat(rows),_)) == True{} : Bool} inside : {Nat.is_lt(local,part_value) == True{} : Bool}}
  +part_inside = {%Equal.sym(Nat,U32.to_nat(part_count),part_value,part_exact) : {Nat.is_lt(local,_) == True{} : Bool} element_inside :
    {Nat.is_lt(local,U32.to_nat(part_count)) == True{} : Bool}}
  +numeric = weight_list_numeric_domain(xi,part_di,k,s,p,g,input,gradient,local,part_positive,
    weight_geometry_ready(xi,part_di,k,s,p,g,part_positive,part_count,part_counted),part_count,part_counted,part_inside)
  +listed = {%Arithmetic.native_roundtrip(rows) :
      {WeightModel.weight_values(WeightModel.weight(part_work,input,gradient)) ==
        WeightModel.transpose(U32.to_nat(_),0,_,U32.to_nat(WeightKernel.taps(part_work)),WeightGroups.group_output(WeightModel.values(part_work,input,gradient))) : List<&2,F32>}
    WeightModel.weight_list(part_work,input,gradient) : {weight_list(part_work,input,gradient) == KParts.owner_list(work,KParts.OwnersModel{input,gradient},first,rows) : List<&2,F32>}}
  +geometry = slice_geometry(U32.to_nat(n),U32.to_nat(o),U32.to_nat(oh),U32.to_nat(ow))
  +band = FromWhole.band(~F32,geometry,0.0,Buffer.Content{dt,dl},first,rows)
  +seen_input = Equal.cong(KParts.OwnersSeen,Buffer.Content<F32>,seen => Pair.fst(Buffer.Content<F32>,Buffer.Content<F32>,KParts.seen_pair(seen)),
    KParts.seen_model(KParts.OwnersModel{input,gradient}),KParts.OwnersSeen{Buffer.Content{xt,xl},band},fact)
  +seen_gradient = Equal.cong(KParts.OwnersSeen,Buffer.Content<F32>,seen => Pair.snd(Buffer.Content<F32>,Buffer.Content<F32>,KParts.seen_pair(seen)),
    KParts.seen_model(KParts.OwnersModel{input,gradient}),KParts.OwnersSeen{Buffer.Content{xt,xl},band},fact)
  +input_tree = Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,input),Buffer.content_tree(~F32,Buffer.content(~F32,input)),xt,
    Equal.sym(Tree.Tree<F32>,Buffer.content_tree(~F32,Buffer.content(~F32,input)),Buffer.tree(~F32,input),Buffer.content_tree_of(~F32,input)),
    Equal.cong(Buffer.Content<F32>,Tree.Tree<F32>,content => Buffer.content_tree(~F32,content),Buffer.content(~F32,input),Buffer.Content{xt,xl},seen_input))
  +gradient_tree = Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,gradient),Buffer.content_tree(~F32,Buffer.content(~F32,gradient)),Buffer.content_tree(~F32,band),
    Equal.sym(Tree.Tree<F32>,Buffer.content_tree(~F32,Buffer.content(~F32,gradient)),Buffer.tree(~F32,gradient),Buffer.content_tree_of(~F32,gradient)),
    Equal.cong(Buffer.Content<F32>,Tree.Tree<F32>,content => Buffer.content_tree(~F32,content),Buffer.content(~F32,gradient),band,seen_gradient))
  +height = U32.from_nat(U32.to_nat(o))
  +height_positive = {%Equal.sym(Nat,U32.to_nat(height),U32.to_nat(o),outputs_exact) : {Nat.is_lt(0n,_) == True{} : Bool} outputs_positive :
    {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool}}
  +height_room = {%Equal.sym(Nat,U32.to_nat(height),U32.to_nat(o),outputs_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(U32.to_nat(o),Nat.mul(U32.to_nat(o),taps),U32.to_nat(capacity),RowMajor.product_covers(U32.to_nat(o),taps,taps_positive),whole_room) :
    {Nat.is_le(U32.to_nat(height),U32.to_nat(capacity)) == True{} : Bool}}
  +offset_room = Order.transitive(Nat.add(Nat.mod(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(rows)),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(height),
    Order.add_le(Nat.mod(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(first),U32.to_nat(rows),DivisionOrder.remainder_le(U32.to_nat(first),U32.to_nat(height),height_positive)),
    room)
  +span = {BandShape.RowSpan{
      Inspection.local_guard(height,first,rows,Machine.half_word(Default.limits()),Limits.half_word_fits(Default.limits()),height_positive,
        Limits.double_bound(Default.limits(),U32.to_nat(height),height_room),offset_room),
      %Equal.sym(Nat,Nat.mul(1n,U32.to_nat(height)),U32.to_nat(height),
          Equal.trans(Nat,Nat.mul(1n,U32.to_nat(height)),Nat.mul(U32.to_nat(height),1n),U32.to_nat(height),
            RowMajor.multiply_commutative(1n,U32.to_nat(height)),Order.times_one(U32.to_nat(height)))) :
        {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool}
      room} :
    BandShape.RowSpan<1,height,first,rows>}
  +band_values = FromWhole.semantics_leaf(~F32,geometry,0.0,first,rows,dt,dl,capacity,Machine.half_word(Default.limits()),
    FromWhole.planned_band(FromWhole.Owner{geometry,Default.limits(),capacity,domain},first,rows,span))
  +slice = {Slice{U32.to_nat(n),U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p),U32.to_nat(oh),U32.to_nat(ow),U32.to_nat(g),
    first,rows,local,xt,dt,dl,band_values,images_exact,outputs_exact,area_exact,outputs_room,element_inside,kernel_positive,channels_positive,columns_positive,group_positive} : Slice}
  %listed :
    {ListRead.at(~F32,0.0,local,_) == weight_value(xi,di,k,s,p,g,Buffer.Model{xt,xl,xc},Buffer.Model{dt,dl,dc},Nat.add(Nat.mul(U32.to_nat(first),width),local)) : F32}
  %Equal.sym(Nat,U32.to_nat(WeightPacking.geometry_taps(weight_geometry(xi,di,k,s,p))),taps,taps_exact) :
    {ListRead.at(~F32,0.0,local,weight_list(part_work,input,gradient)) ==
      weight_value(xi,di,k,s,p,g,Buffer.Model{xt,xl,xc},Buffer.Model{dt,dl,dc},Nat.add(Nat.mul(U32.to_nat(first),_),local)) : F32}
  Equal.trans(F32,ListRead.at(~F32,0.0,local,weight_list(part_work,input,gradient)),weight_value(xi,part_di,k,s,p,g,input,gradient,local),
    weight_value(xi,di,k,s,p,g,Buffer.Model{xt,xl,xc},Buffer.Model{dt,dl,dc},Nat.add(Nat.mul(U32.to_nat(first),taps),local)),
    numeric,
    %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,input),xt,input_tree) :
      {Spec.weight(~Tree.Tree<F32>,~Bridge.at,Bridge.weight_problem(part_work,Model.Source{_,Buffer.tree(~F32,gradient)}),local) ==
        weight_value(xi,di,k,s,p,g,Buffer.Model{xt,xl,xc},Buffer.Model{dt,dl,dc},Nat.add(Nat.mul(U32.to_nat(first),taps),local)) : F32}
    %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,gradient),Buffer.content_tree(~F32,band),gradient_tree) :
      {Spec.weight(~Tree.Tree<F32>,~Bridge.at,Bridge.weight_problem(part_work,Model.Source{xt,_}),local) ==
        weight_value(xi,di,k,s,p,g,Buffer.Model{xt,xl,xc},Buffer.Model{dt,dl,dc},Nat.add(Nat.mul(U32.to_nat(first),taps),local)) : F32}
    slice_weight(slice))

# The parts list the weight gradient of the whole call.
def parts_agree(+scope: WeightParts,+depth: Nat,+owners: BandModel.Shaped<KParts.OwnersModel>,
  +seen: {BandModel.shaped_map(~KParts.OwnersModel,~KParts.OwnersSeen,~KParts.seen_model,owners) ==
    BandModel.shaped_build(~KParts.OwnersSeen,~KParts.OwnersSeen,~KParts.piece(parts_work(scope)),parts_before(scope),
      KParts.plan(depth,parts_work(scope))) : BandModel.Shaped<KParts.OwnersSeen>},
  +capacity_ok: {U32.is_le(WeightKernel.outputs(parts_work(scope)),2147483648) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,Nat.mul(U32.to_nat(WeightKernel.outputs(parts_work(scope))),KParts.work_width(parts_work(scope)))) == True{} : Bool}) ->
  {ListRead.at(~F32,0.0,index,KParts.parts_list(parts_work(scope),owners)) == parts_value(scope,index) : F32}:
  WeightParts{+xi,+di,+k,+s,+p,+g,+x,+d,+positive,+counted,+room} = scope
  +scope = {WeightParts{xi,di,k,s,p,g,x,d,positive,counted,room} : WeightParts}
  +work = Parameter.work(xi,di,k,s,p,g)
  +before = {KParts.OwnersSeen{Buffer.content(~F32,x),Buffer.content(~F32,d)} : KParts.OwnersSeen}
  BandEach.shaped_list_at(~KParts.OwnersModel,~F32,~Kernel.Work,~KParts.owner_list,~KParts.work_width,~KParts.owner_length,
    ~WeightParts,~parts_work,~parts_fill,~parts_limit,
    ~BandModel.viewed(~KParts.OwnersModel,~BandEach.BuiltScope<Kernel.Work,KParts.OwnersSeen>,
      ~BandEach.built_piece(~KParts.OwnersModel,~KParts.OwnersSeen,~KParts.OwnersSeen,~Kernel.Work,~KParts.seen_model,~KParts.piece),~WeightParts,~parts_view),
    ~parts_value,~parts_read,
    scope,owners,0n,U32.to_nat(WeightKernel.outputs(work)),
    KParts.owners_coverage(depth,work,owners,capacity_ok,KParts.owners_plan(depth,work,before,owners,seen)),
    BandModel.shaped_all_view(~KParts.OwnersModel,~BandEach.BuiltScope<Kernel.Work,KParts.OwnersSeen>,
      ~BandEach.built_piece(~KParts.OwnersModel,~KParts.OwnersSeen,~KParts.OwnersSeen,~Kernel.Work,~KParts.seen_model,~KParts.piece),~WeightParts,~parts_view,scope,owners,
      BandEach.shaped_built_all(~KParts.OwnersModel,~KParts.OwnersSeen,~KParts.OwnersSeen,~Kernel.Work,~KParts.seen_model,~KParts.piece,work,before,owners,KParts.plan(depth,work),seen)),
    Order.reflexive(U32.to_nat(WeightKernel.outputs(work))),index,inside)

# An index of the weight count lies in the outputs times the taps.
def work_inside(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+positive: WeightPositive<xi,di,k,s,g>,
  +count: U32,+counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),count>,+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.is_lt(index,Nat.mul(U32.to_nat(WeightKernel.outputs(Parameter.work(xi,di,k,s,p,g))),U32.to_nat(WeightKernel.taps(Parameter.work(xi,di,k,s,p,g))))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(U32.to_nat(WeightKernel.outputs(Parameter.work(xi,di,k,s,p,g))),U32.to_nat(WeightKernel.taps(Parameter.work(xi,di,k,s,p,g)))),ShapeLaws.elements(weight_shape(xi,di,k)),
      weight_elements(xi,di,k,s,p,g,positive,count,counted)) :
    {Nat.is_lt(index,_) == True{} : Bool}
  %Equal.sym(Nat,ShapeLaws.elements(weight_shape(xi,di,k)),U32.to_nat(count),Count.count_exact(ShapeLaws.elements(weight_shape(xi,di,k)),count,counted)) :
    {Nat.is_lt(index,_) == True{} : Bool}
  inside

# The serial kernel lists the weight gradient of the call.
def serial_numeric(+xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+x: Buffer.Model<F32>,+d: Buffer.Model<F32>,
  +positive: WeightPositive<xi,di,k,s,g>,+count: U32,+counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),count>,
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {ListRead.at(~F32,0.0,index,KParts.serial_list(Parameter.work(xi,di,k,s,p,g),x,d)) == weight_value(xi,di,k,s,p,g,x,d,index) : F32}:
  %WeightModel.weight_list(Parameter.work(xi,di,k,s,p,g),x,d) : {ListRead.at(~F32,0.0,index,_) == weight_value(xi,di,k,s,p,g,x,d,index) : F32}
  weight_list_numeric_domain(xi,di,k,s,p,g,x,d,index,positive,weight_geometry_ready(xi,di,k,s,p,g,positive,count,counted),count,counted,inside)

# Storage holding a list of the weight gradient, and the sources given back
# holding what they held, make a weight call.
def weight_stored(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+input_context: Input,+gradient_context: Gradient,
  +xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +x: Buffer.Model<F32>,+xs: List<&2,U32>,+xf: F32,+xw: U32,+x_facts: GatherValues.Facts(~F32,~Input,~input_value,input_context,xs,Buffer.content(~F32,x)),+x_cells: Readers.ImageCells(xs,Some{xi}),
  +d: Buffer.Model<F32>,+ds: List<&2,U32>,+df: F32,+dw: U32,+d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,ds,Buffer.content(~F32,d)),+d_cells: Readers.ImageCells(ds,Some{di}),
  +positive: WeightPositive<xi,di,k,s,g>,+counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),KParts.count(Parameter.work(xi,di,k,s,p,g))>,
  result: (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>,+values: List<&2,F32>,
  +contents: {KParts.returned(result) ==
    ((Buffer.content(~F32,x),Buffer.content(~F32,d)),
      Buffer.Content{Tree.write_sequence(~F32,values,0,KParts.fresh(Parameter.work(xi,di,k,s,p,g))),U32.to_nat(KParts.count(Parameter.work(xi,di,k,s,p,g)))}) :
    (Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>},
  +length: {List.length(&2,F32,values) ==
    Nat.mul(U32.to_nat(WeightKernel.outputs(Parameter.work(xi,di,k,s,p,g))),U32.to_nat(WeightKernel.taps(Parameter.work(xi,di,k,s,p,g)))) : Nat},
  numeric: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),count> -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~F32,0.0,index,values) == weight_value(xi,di,k,s,p,g,x,d,index) : F32}) ->
  WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,Some{xi},Some{di},k,s,p,g,xs,ds,
    Parameter.weight_finished(xs,xf,Tensors.Usable{},xw,ds,df,Tensors.Usable{},dw,weight_shape(xi,di,k),result)):
  ((input,gradient),weight) = result
  +work = Parameter.work(xi,di,k,s,p,g)
  +count = KParts.count(work)
  +shape = weight_shape(xi,di,k)
  +weight_content = {Buffer.Content{Tree.write_sequence(~F32,values,0,KParts.fresh(work)),U32.to_nat(count)} : Buffer.Content<F32>}
  +context = {Readers.WeightRead{input_context,gradient_context,Some{xi},Some{di},k,s,p,g} : Readers.WeightRead<Input,Gradient>}
  +sources = Equal.cong((Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>,Buffer.Content<F32> & Buffer.Content<F32>,
    pair => Pair.fst(Buffer.Content<F32> & Buffer.Content<F32>,Buffer.Content<F32>,pair),
    KParts.returned(((input,gradient),weight)),((Buffer.content(~F32,x),Buffer.content(~F32,d)),weight_content),contents)
  +input_kept = Equal.cong(Buffer.Content<F32> & Buffer.Content<F32>,Buffer.Content<F32>,pair => Pair.fst(Buffer.Content<F32>,Buffer.Content<F32>,pair),
    (Buffer.contents(~F32,input),Buffer.contents(~F32,gradient)),(Buffer.content(~F32,x),Buffer.content(~F32,d)),sources)
  +gradient_kept = Equal.cong(Buffer.Content<F32> & Buffer.Content<F32>,Buffer.Content<F32>,pair => Pair.snd(Buffer.Content<F32>,Buffer.Content<F32>,pair),
    (Buffer.contents(~F32,input),Buffer.contents(~F32,gradient)),(Buffer.content(~F32,x),Buffer.content(~F32,d)),sources)
  +weight_kept = Equal.cong((Buffer.Content<F32> & Buffer.Content<F32>) & Buffer.Content<F32>,Buffer.Content<F32>,
    pair => Pair.snd(Buffer.Content<F32> & Buffer.Content<F32>,Buffer.Content<F32>,pair),
    KParts.returned(((input,gradient),weight)),((Buffer.content(~F32,x),Buffer.content(~F32,d)),weight_content),contents)
  +length_exact = Equal.trans(Nat,List.length(&2,F32,values),
    Nat.mul(U32.to_nat(WeightKernel.outputs(work)),U32.to_nat(WeightKernel.taps(work))),U32.to_nat(count),length,
    Equal.trans(Nat,Nat.mul(U32.to_nat(WeightKernel.outputs(work)),U32.to_nat(WeightKernel.taps(work))),ShapeLaws.elements(shape),U32.to_nat(count),
      weight_elements(xi,di,k,s,p,g,positive,count,counted),Count.count_exact(ShapeLaws.elements(shape),count,counted)))
  Shape.Both{
    Listed.stored_listed(~F32,~Readers.WeightRead<Input,Gradient>,~Readers.weight_at(~Input,~input_value,~Gradient,~gradient_value),context,weight,shape,values,0.0,count,U32.max(xw,dw),
      Buffer.observe(~F32,weight),weight_kept,counted,length_exact,
      listed_count => listed_counted => index => inside =>
        weight_agreement(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,xi,di,k,s,p,g,x,xs,x_facts,x_cells,d,ds,d_facts,d_cells,positive,
          listed_count,listed_counted,values,index,inside,numeric(listed_count,listed_counted,index,inside))),
    Shape.Both{
      WholeLeaf.stored_shaped(~F32,~Input,~input_value,input_context,input,xs,xf,xw,Buffer.observe(~F32,input),
        %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,input),Buffer.content(~F32,x),input_kept) : GatherValues.Facts(~F32,~Input,~input_value,input_context,xs,_) x_facts),
      WholeLeaf.stored_shaped(~F32,~Gradient,~gradient_value,gradient_context,gradient,ds,df,dw,Buffer.observe(~F32,gradient),
        %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,gradient),Buffer.content(~F32,d),gradient_kept) : GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,ds,_) d_facts)}}

# The parts of the outputs make a weight call.
def weight_parts_stored(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+depth: Nat,+input_context: Input,+gradient_context: Gradient,
  +xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +x: Buffer.Model<F32>,+xs: List<&2,U32>,+xf: F32,+xw: U32,+x_facts: GatherValues.Facts(~F32,~Input,~input_value,input_context,xs,Buffer.content(~F32,x)),+x_cells: Readers.ImageCells(xs,Some{xi}),
  +d: Buffer.Model<F32>,+ds: List<&2,U32>,+df: F32,+dw: U32,+d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,ds,Buffer.content(~F32,d)),+d_cells: Readers.ImageCells(ds,Some{di}),
  +positive: WeightPositive<xi,di,k,s,g>,+counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),KParts.count(Parameter.work(xi,di,k,s,p,g))>,
  +room: GradientRoom<Parameter.work(xi,di,k,s,p,g)>,+capacity_ok: {U32.is_le(WeightKernel.outputs(Parameter.work(xi,di,k,s,p,g)),2147483648) == True{} : Bool},
  parts: KParts.Parts<Parameter.work(xi,di,k,s,p,g),KParts.plan(depth,Parameter.work(xi,di,k,s,p,g)),KParts.OwnersSeen{Buffer.content(~F32,x),Buffer.content(~F32,d)},
    WeightKernel.parts(depth,Parameter.work(xi,di,k,s,p,g),Buffer.pack(~F32,x),Buffer.pack(~F32,d))>) ->
  WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,Some{xi},Some{di},k,s,p,g,xs,ds,
    Parameter.weight_finished(xs,xf,Tensors.Usable{},xw,ds,df,Tensors.Usable{},dw,weight_shape(xi,di,k),WeightKernel.parts(depth,Parameter.work(xi,di,k,s,p,g),Buffer.pack(~F32,x),Buffer.pack(~F32,d)))):
  KParts.Parts{+owners,+seen,+contents} = parts
  +work = Parameter.work(xi,di,k,s,p,g)
  +scope = {WeightParts{xi,di,k,s,p,g,x,d,positive,counted,room} : WeightParts}
  weight_stored(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,xi,di,k,s,p,g,x,xs,xf,xw,x_facts,x_cells,d,ds,df,dw,d_facts,d_cells,positive,counted,
    WeightKernel.parts(depth,work,Buffer.pack(~F32,x),Buffer.pack(~F32,d)),KParts.parts_list(work,owners),contents,
    KParts.parts_length(depth,work,owners,capacity_ok,KParts.owners_plan(depth,work,KParts.OwnersSeen{Buffer.content(~F32,x),Buffer.content(~F32,d)},owners,seen)),
    listed_count => listed_counted => index => inside =>
      parts_agree(scope,depth,owners,seen,capacity_ok,index,work_inside(xi,di,k,s,p,g,positive,listed_count,listed_counted,index,inside)))

# One part, or the parts of the outputs, as the call chose.
def weight_routed(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,depth: Nat,+input_context: Input,+gradient_context: Gradient,
  +xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +x: Buffer.Model<F32>,+xs: List<&2,U32>,+xf: F32,+xw: U32,+x_facts: GatherValues.Facts(~F32,~Input,~input_value,input_context,xs,Buffer.content(~F32,x)),+x_cells: Readers.ImageCells(xs,Some{xi}),
  +d: Buffer.Model<F32>,+ds: List<&2,U32>,+df: F32,+dw: U32,+d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,ds,Buffer.content(~F32,d)),+d_cells: Readers.ImageCells(ds,Some{di}),
  +positive: WeightPositive<xi,di,k,s,g>,+counted: Count.Count<ShapeLaws.elements(weight_shape(xi,di,k)),KParts.count(Parameter.work(xi,di,k,s,p,g))>,
  +room: GradientRoom<Parameter.work(xi,di,k,s,p,g)>,+capacity_ok: {U32.is_le(WeightKernel.outputs(Parameter.work(xi,di,k,s,p,g)),2147483648) == True{} : Bool}) ->
  WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,Some{xi},Some{di},k,s,p,g,xs,ds,
    Parameter.weight_finished(xs,xf,Tensors.Usable{},xw,ds,df,Tensors.Usable{},dw,weight_shape(xi,di,k),WeightKernel.routed(depth,Parameter.work(xi,di,k,s,p,g),Buffer.pack(~F32,x),Buffer.pack(~F32,d)))):
  match depth:
    case 0n:
      +work = Parameter.work(xi,di,k,s,p,g)
      weight_stored(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,xi,di,k,s,p,g,x,xs,xf,xw,x_facts,x_cells,d,ds,df,dw,d_facts,d_cells,positive,counted,
        WeightKernel.weight(work,Buffer.pack(~F32,x),Buffer.pack(~F32,d)),KParts.serial_list(work,x,d),KParts.serial(work,x,d),
        WeightModel.transpose_length(U32.to_nat(WeightKernel.outputs(work)),0,WeightKernel.outputs(work),U32.to_nat(WeightKernel.taps(work)),
          WeightGroups.group_output(WeightModel.values(work,x,d))),
        listed_count => listed_counted => index => inside => serial_numeric(xi,di,k,s,p,g,x,d,positive,listed_count,listed_counted,index,inside))
    case 1n+ +rest:
      weight_parts_stored(~Input,~input_value,~Gradient,~gradient_value,1n+rest,input_context,gradient_context,xi,di,k,s,p,g,x,xs,xf,xw,x_facts,x_cells,d,ds,df,dw,d_facts,d_cells,
        positive,counted,room,capacity_ok,KParts.parts(1n+rest,Parameter.work(xi,di,k,s,p,g),x,d,capacity_ok))

# An accepted weight call: the gradient of the readers' values, and the
# sources back.
def weight_called(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+input_context: Input,+gradient_context: Gradient,
  +xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +x: Buffer.Model<F32>,+xs: List<&2,U32>,+xf: F32,+xw: U32,+x_facts: GatherValues.Facts(~F32,~Input,~input_value,input_context,xs,Buffer.content(~F32,x)),+x_cells: Readers.ImageCells(xs,Some{xi}),
  +d: Buffer.Model<F32>,+ds: List<&2,U32>,+df: F32,+dw: U32,+d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,ds,Buffer.content(~F32,d)),+d_cells: Readers.ImageCells(ds,Some{di}),
  +valid: {Parameter.valid(xi,di,k,s,p,g) == True{} : Bool}) ->
  WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,Some{xi},Some{di},k,s,p,g,xs,ds,
    Parameter.weight_selected(True{},xi,di,k,s,p,g,Buffer.pack(~F32,x),xs,xf,Tensors.Usable{},xw,Buffer.pack(~F32,d),ds,df,Tensors.Usable{},dw)):
  Parameter.ImageGeometry{+n,+c,+h,+w} = xi
  Parameter.ImageGeometry{+dn,+o,+oh,+ow} = di
  Buffer.Model{+dt,+dl,+dc} = d
  Shape.Both{Contract.WellFormed{+d_length,+d_ready},+d_values} = d_facts
  +work = Parameter.work(xi,di,k,s,p,g)
  +gradient = {Buffer.Model{dt,dl,dc} : Buffer.Model<F32>}
  +positive = weight_positive(xi,di,k,s,p,g,valid)
  +counted = weight_counted(xi,di,k,s,p,g,positive,Tensors.count_of(weight_shape(xi,di,k)),{==},weight_fits(xi,di,k,s,p,g,valid))
  +same_images = Order.equal_of_is_eq(U32.to_nat(n),U32.to_nat(dn),%Comparison.eq(n,dn) : {_ == True{} : Bool} positive_images(xi,di,k,s,g,positive))
  +cells_length = {%Equal.sym(Nat,U32.to_nat(n),U32.to_nat(dn),same_images) : {Nat.mul(Nat.mul(_,U32.to_nat(o)),Nat.mul(U32.to_nat(oh),U32.to_nat(ow))) == dl : Nat}
    Equal.sym(Nat,dl,Readers.cells(di),Equal.trans(Nat,dl,ShapeLaws.elements(ds),Readers.cells(di),d_length,d_cells)) : {work_cells(work) == dl : Nat}}
  +room = gradient_room_of(xi,di,k,s,p,g,positive,gradient,Buffer.ready_content(~F32,gradient,d_ready),
    %cells_length : {Nat.is_le(work_cells(work),_) == True{} : Bool} Order.reflexive(work_cells(work)))
  weight_routed(~Input,~input_value,~Gradient,~gradient_value,WeightKernel.depth(work,U32.max(xw,dw)),input_context,gradient_context,xi,di,k,s,p,g,x,xs,xf,xw,x_facts,x_cells,
    gradient,ds,df,dw,Shape.Both{Contract.WellFormed{d_length,d_ready},d_values},d_cells,positive,counted,room,outputs_fit(xi,di,k,s,p,g,positive,counted))

def weight_valid(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,accepted_shape: Bool,+input_context: Input,+gradient_context: Gradient,
  +xi: Parameter.ImageGeometry,+di: Parameter.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +x: Buffer.Model<F32>,+xs: List<&2,U32>,+xf: F32,+xw: U32,+x_facts: GatherValues.Facts(~F32,~Input,~input_value,input_context,xs,Buffer.content(~F32,x)),+x_cells: Readers.ImageCells(xs,Some{xi}),
  +d: Buffer.Model<F32>,+ds: List<&2,U32>,+df: F32,+dw: U32,+d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,ds,Buffer.content(~F32,d)),+d_cells: Readers.ImageCells(ds,Some{di}),
  +valid: {Parameter.valid(xi,di,k,s,p,g) == accepted_shape : Bool},
  accepted: {Contract.tensor_ready(~F32,weight_result(Parameter.weight_selected(accepted_shape,xi,di,k,s,p,g,Buffer.pack(~F32,x),xs,xf,Tensors.Usable{},xw,Buffer.pack(~F32,d),ds,df,Tensors.Usable{},dw))) == True{} : Bool}) ->
  WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,Some{xi},Some{di},k,s,p,g,xs,ds,
    Parameter.weight_selected(accepted_shape,xi,di,k,s,p,g,Buffer.pack(~F32,x),xs,xf,Tensors.Usable{},xw,Buffer.pack(~F32,d),ds,df,Tensors.Usable{},dw)):
  match accepted_shape:
    case False{}:
      Empty.absurd(WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,Some{xi},Some{di},k,s,p,g,xs,ds,
        Parameter.weight_selected(False{},xi,di,k,s,p,g,Buffer.pack(~F32,x),xs,xf,Tensors.Usable{},xw,Buffer.pack(~F32,d),ds,df,Tensors.Usable{},dw)),Words.false_true(accepted))
    case True{}:
      weight_called(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,xi,di,k,s,p,g,x,xs,xf,xw,x_facts,x_cells,d,ds,df,dw,d_facts,d_cells,valid)

# The weight gradient of storage holding the readers' values, in any shapes.
def weight_shaped(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,xi: Maybe<&2,Parameter.ImageGeometry>,di: Maybe<&2,Parameter.ImageGeometry>,+input_context: Input,+gradient_context: Gradient,
  +k: U32,+s: U32,+p: U32,+g: U32,
  +x: Buffer.Model<F32>,+xs: List<&2,U32>,+xf: F32,+xw: U32,+x_facts: GatherValues.Facts(~F32,~Input,~input_value,input_context,xs,Buffer.content(~F32,x)),+x_cells: Readers.ImageCells(xs,xi),
  +d: Buffer.Model<F32>,+ds: List<&2,U32>,+df: F32,+dw: U32,+d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,ds,Buffer.content(~F32,d)),+d_cells: Readers.ImageCells(ds,di),
  accepted: {Contract.tensor_ready(~F32,weight_result(Parameter.weight_shaped(xi,di,Tensors.Usable{},Tensors.Usable{},k,s,p,g,Buffer.pack(~F32,x),xs,xf,xw,Buffer.pack(~F32,d),ds,df,dw))) == True{} : Bool}) ->
  WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,xi,di,k,s,p,g,xs,ds,
    Parameter.weight_shaped(xi,di,Tensors.Usable{},Tensors.Usable{},k,s,p,g,Buffer.pack(~F32,x),xs,xf,xw,Buffer.pack(~F32,d),ds,df,dw)):
  match xi di:
    case Some{+xi} Some{+di}:
      weight_valid(~Input,~input_value,~Gradient,~gradient_value,Parameter.valid(xi,di,k,s,p,g),input_context,gradient_context,xi,di,k,s,p,g,x,xs,xf,xw,x_facts,x_cells,d,ds,df,dw,d_facts,d_cells,{==},accepted)
    case Some{xi} None{}:
      Empty.absurd(WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,Some{xi},None{},k,s,p,g,xs,ds,
        Parameter.weight_shaped(Some{xi},None{},Tensors.Usable{},Tensors.Usable{},k,s,p,g,Buffer.pack(~F32,x),xs,xf,xw,Buffer.pack(~F32,d),ds,df,dw)),Words.false_true(accepted))
    case None{} Some{di}:
      Empty.absurd(WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,None{},Some{di},k,s,p,g,xs,ds,
        Parameter.weight_shaped(None{},Some{di},Tensors.Usable{},Tensors.Usable{},k,s,p,g,Buffer.pack(~F32,x),xs,xf,xw,Buffer.pack(~F32,d),ds,df,dw)),Words.false_true(accepted))
    case None{} None{}:
      Empty.absurd(WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,None{},None{},k,s,p,g,xs,ds,
        Parameter.weight_shaped(None{},None{},Tensors.Usable{},Tensors.Usable{},k,s,p,g,Buffer.pack(~F32,x),xs,xf,xw,Buffer.pack(~F32,d),ds,df,dw)),Words.false_true(accepted))

# An output of the bias gradient of storage holding the reader's values is
# the reader's bias gradient there.
def bias_agreement(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+gradient_context: Gradient,+di: Parameter.ImageGeometry,+g: U32,
  +d: Buffer.Model<F32>,+shape: List<&2,U32>,+facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,shape,Buffer.content(~F32,d)),
  +cells: Readers.ImageCells(shape,Some{di}),+positive: {Parameter.positive(di) == True{} : Bool},+group: {U32.is_lt(0,g) == True{} : Bool},
  +count: U32,+counted: Count.Count<ShapeLaws.elements(bias_shape(di)),count>,+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {ListRead.at(~F32,0.0,index,bias_list(Parameter.bias_work(di,g),d)) == Readers.bias_at(~Gradient,~gradient_value)(Readers.BiasRead{gradient_context,Some{di},g},U32.from_nat(index)) : F32}:
  Parameter.ImageGeometry{+n,+o,+rows,+columns} = di
  Buffer.Model{+dt,+dl,+dc} = d
  Shape.Both{Contract.WellFormed{+length,ready},+values} = facts
  Count.Count{+exact,bounded} = counted
  +columns_word = Logic.right((U32.is_lt(0,n) && U32.is_lt(0,o)) && U32.is_lt(0,rows),U32.is_lt(0,columns),positive)
  +columns_positive = {%Comparison.lt(0,columns) : {_ == True{} : Bool} columns_word : {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool}}
  +group_positive = {%Comparison.lt(0,g) : {_ == True{} : Bool} group : {Nat.is_lt(0n,U32.to_nat(g)) == True{} : Bool}}
  +room = {%Equal.trans(Nat,dl,ShapeLaws.elements(shape),Readers.cells(di),length,cells) : {Nat.is_le(_,dl) == True{} : Bool} Order.reflexive(dl) :
    {Nat.is_le(Readers.cells(di),dl) == True{} : Bool}}
  +element_inside = {%Order.times_one(U32.to_nat(o)) : {Nat.is_lt(index,_) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements([o]),U32.to_nat(count),exact) : {Nat.is_lt(index,_) == True{} : Bool} inside :
    {Nat.is_lt(index,U32.to_nat(o)) == True{} : Bool}}
  +scope = {Readers.BiasScope{U32.to_nat(n),U32.to_nat(o),U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(g),dt,dl,gradient_context,values,room,index,element_inside,columns_positive,group_positive} :
    Readers.BiasScope<Gradient,gradient_value>}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,FullRange.roundtrip(index,count,Bounds.strict_le(index,U32.to_nat(count),inside))) :
    {ListRead.at(~F32,0.0,index,bias_list(Parameter.bias_work(di,g),Buffer.Model{dt,dl,dc})) ==
      Spec.bias(~Gradient,~Readers.gradient_at(~Gradient,~gradient_value),Spec.Problem{Bridge.shape(Parameter.bias_work(di,g)),gradient_context,gradient_context},_) : F32}
  Equal.trans(F32,ListRead.at(~F32,0.0,index,bias_list(Parameter.bias_work(di,g),Buffer.Model{dt,dl,dc})),
    bias_value(di,g,Buffer.Model{dt,dl,dc},index),
    Spec.bias(~Gradient,~Readers.gradient_at(~Gradient,~gradient_value),Readers.bias_read(~Gradient,~gradient_value,scope),index),
    Bridge.enumerate_at(~Spec.Problem<Tree.Tree<F32>>,~Spec.bias(~Tree.Tree<F32>,~Bridge.at),U32.to_nat(o),index,0n,
      Bridge.bias_problem(Parameter.bias_work(di,g),dt),element_inside),
    Readers.bias_agrees(~Gradient,~gradient_value,scope))

def bias_accepted(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+gradient_context: Gradient,+di: Parameter.ImageGeometry,+g: U32,+d: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,+facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,shape,Buffer.content(~F32,d)),
  +cells: Readers.ImageCells(shape,Some{di}),+positive: {Parameter.positive(di) == True{} : Bool},+group: {U32.is_lt(0,g) == True{} : Bool},
  +count: U32,+counts: {Tensors.count_of(bias_shape(di)) == Some{count} : Maybe<&2,U32>}) ->
  Global.Contract(~F32,~Readers.BiasRead<Gradient>,~Readers.bias_at(~Gradient,~gradient_value),Readers.BiasRead{gradient_context,Some{di},g},
    bias_result(Parameter.bias_counted(Some{count},di,g,Buffer.pack(~F32,d),shape,fill,Tensors.Usable{},workers))):
  Parameter.ImageGeometry{+n,+o,+rows,+columns} = di
  +image = {Parameter.ImageGeometry{n,o,rows,columns} : Parameter.ImageGeometry}
  +work = Parameter.bias_work(image,g)
  +values = bias_list(work,d)
  +context = {Readers.BiasRead{gradient_context,Some{image},g} : Readers.BiasRead<Gradient>}
  +counted = Count.count_of([o],count,counts)
  +length_exact = Equal.trans(Nat,List.length(&2,F32,values),U32.to_nat(o),U32.to_nat(count),
    Bridge.enumerate_length(~Spec.Problem<Tree.Tree<F32>>,~Spec.bias(~Tree.Tree<F32>,~Bridge.at),U32.to_nat(o),0n,Bridge.bias_problem(work,Buffer.tree(~F32,d))),
    Equal.trans(Nat,U32.to_nat(o),ShapeLaws.elements([o]),U32.to_nat(count),Equal.sym(Nat,ShapeLaws.elements([o]),U32.to_nat(o),Order.times_one(U32.to_nat(o))),
      Count.count_exact(ShapeLaws.elements([o]),count,counted)))
  %Equal.sym(Tensors.Tensor<F32>,
      bias_result(Parameter.bias_finished(shape,fill,Tensors.Usable{},workers,o,KernelBuffer.bias(work,Buffer.pack(~F32,d),Tensors.buffer(~F32,0.0,count)))),
      Tensors.whole(~F32,BufferModel.written(KernelBuffer.bias(work,Buffer.pack(~F32,d),Tensors.buffer(~F32,0.0,count))),[o],0.0,Tensors.Usable{},workers),
      bias_finished(shape,fill,Tensors.Usable{},workers,o,KernelBuffer.bias(work,Buffer.pack(~F32,d),Tensors.buffer(~F32,0.0,count)))) :
    Global.Contract(~F32,~Readers.BiasRead<Gradient>,~Readers.bias_at(~Gradient,~gradient_value),context,_)
  Listed.stored_listed(~F32,~Readers.BiasRead<Gradient>,~Readers.bias_at(~Gradient,~gradient_value),context,
    BufferModel.written(KernelBuffer.bias(work,Buffer.pack(~F32,d),Tensors.buffer(~F32,0.0,count))),[o],values,0.0,count,workers,
    Buffer.observe(~F32,BufferModel.written(KernelBuffer.bias(work,Buffer.pack(~F32,d),Tensors.buffer(~F32,0.0,count)))),
    bias_stored(work,d,count),counted,length_exact,
    listed_count => listed_counted => index => inside =>
      bias_agreement(~Gradient,~gradient_value,gradient_context,image,g,d,shape,facts,cells,positive,group,listed_count,listed_counted,index,inside))

# The outputs fit a count of [o]: the result and the returned output gradient.
def bias_counted(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,count: Maybe<&2,U32>,+gradient_context: Gradient,+di: Parameter.ImageGeometry,+g: U32,+d: Buffer.Model<F32>,
  +shape: List<&2,U32>,+fill: F32,+workers: U32,+facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,shape,Buffer.content(~F32,d)),
  +cells: Readers.ImageCells(shape,Some{di}),+positive: {Parameter.positive(di) == True{} : Bool},+group: {U32.is_lt(0,g) == True{} : Bool},
  +counts: {Tensors.count_of(bias_shape(di)) == count : Maybe<&2,U32>},
  accepted: {Contract.tensor_ready(~F32,bias_result(Parameter.bias_counted(count,di,g,Buffer.pack(~F32,d),shape,fill,Tensors.Usable{},workers))) == True{} : Bool}) ->
  BiasCall(~Gradient,~gradient_value,gradient_context,Some{di},g,shape,Parameter.bias_counted(count,di,g,Buffer.pack(~F32,d),shape,fill,Tensors.Usable{},workers)):
  match count:
    case Some{+count}:
      Shape.Both{
        bias_accepted(~Gradient,~gradient_value,gradient_context,di,g,d,shape,fill,workers,facts,cells,positive,group,count,counts),
        bias_returned(~Gradient,~gradient_value,gradient_context,di,g,d,shape,fill,workers,count,facts)}
    case None{}:
      Parameter.ImageGeometry{+n,+o,+rows,+columns} = di
      Empty.absurd(BiasCall(~Gradient,~gradient_value,gradient_context,Some{Parameter.ImageGeometry{n,o,rows,columns}},g,shape,
        Parameter.bias_counted(None{},Parameter.ImageGeometry{n,o,rows,columns},g,Buffer.pack(~F32,d),shape,fill,Tensors.Usable{},workers)),Words.false_true(accepted))

def bias_valid(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,accepted_shape: Bool,+gradient_context: Gradient,+di: Parameter.ImageGeometry,+g: U32,+d: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,+facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,shape,Buffer.content(~F32,d)),
  +cells: Readers.ImageCells(shape,Some{di}),+gate: {Parameter.positive(di) && U32.is_lt(0,g) == accepted_shape : Bool},
  accepted: {Contract.tensor_ready(~F32,bias_result(Parameter.bias_selected(accepted_shape,di,g,Buffer.pack(~F32,d),shape,fill,Tensors.Usable{},workers))) == True{} : Bool}) ->
  BiasCall(~Gradient,~gradient_value,gradient_context,Some{di},g,shape,Parameter.bias_selected(accepted_shape,di,g,Buffer.pack(~F32,d),shape,fill,Tensors.Usable{},workers)):
  match accepted_shape:
    case False{}:
      Empty.absurd(BiasCall(~Gradient,~gradient_value,gradient_context,Some{di},g,shape,Parameter.bias_selected(False{},di,g,Buffer.pack(~F32,d),shape,fill,Tensors.Usable{},workers)),
        Words.false_true(accepted))
    case True{}:
      Parameter.ImageGeometry{+n,+o,+rows,+columns} = di
      bias_counted(~Gradient,~gradient_value,Tensors.count_of([o]),gradient_context,Parameter.ImageGeometry{n,o,rows,columns},g,d,shape,fill,workers,facts,cells,
        Logic.left(Parameter.positive(Parameter.ImageGeometry{n,o,rows,columns}),U32.is_lt(0,g),gate),
        Logic.right(Parameter.positive(Parameter.ImageGeometry{n,o,rows,columns}),U32.is_lt(0,g),gate),{==},accepted)

# The bias gradient of storage holding the reader's values, in any shape.
def bias_shaped(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,image: Maybe<&2,Parameter.ImageGeometry>,+gradient_context: Gradient,+g: U32,+d: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,+facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,shape,Buffer.content(~F32,d)),+cells: Readers.ImageCells(shape,image),
  accepted: {Contract.tensor_ready(~F32,bias_result(Parameter.bias_shaped(image,g,Buffer.pack(~F32,d),shape,fill,Tensors.Usable{},workers))) == True{} : Bool}) ->
  BiasCall(~Gradient,~gradient_value,gradient_context,image,g,shape,Parameter.bias_shaped(image,g,Buffer.pack(~F32,d),shape,fill,Tensors.Usable{},workers)):
  match image:
    case Some{+di}:
      bias_valid(~Gradient,~gradient_value,Parameter.positive(di) && U32.is_lt(0,g),gradient_context,di,g,d,shape,fill,workers,facts,cells,{==},accepted)
    case None{}:
      Empty.absurd(BiasCall(~Gradient,~gradient_value,gradient_context,None{},g,shape,Parameter.bias_shaped(None{},g,Buffer.pack(~F32,d),shape,fill,Tensors.Usable{},workers)),
        Words.false_true(accepted))
