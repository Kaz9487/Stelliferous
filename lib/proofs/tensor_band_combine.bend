# The binary band kernel: two bands of one geometry in one plan combine leaf
# by leaf into well formed bands that hold the operation of their values,
# including bands of an empty geometry.
import Base
import ./tensor_band_combine_leaf.bend as BandCombineLeaf
import ./tensor_band_each.bend as Each
import ./tensor_band_model.bend as Model
import ../tensor_band.bend as Bands
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./storage_shape.bend as Shape
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_inspection.bend as Inspection
import ./storage_tree_model.bend as Tree
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_arrive.bend as Arrive
import ./tensor_band_shape_count.bend as BandShapeCount
import ./storage_refinement.bend as StorageLaws
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_vector_descriptor_refinement.bend as TensorVectorDescriptorRefinement
import ./tensor_vector_view_refinement.bend as Vector



def equation(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+geometry: Bands.Geometry,+first: U32,+rows: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  +left_metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,left)),+right_metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,right)),
  +left_ready: Buffer.Sized<Element,left>,+right_ready: Buffer.Sized<Element,right>) ->
  {Buffer.contents(~Element,Bands.combine_leaf_with(~Element,~kernel,Bands.local_count(geometry,rows),Buffer.pack(~Element,left),Buffer.pack(~Element,right))) ==
    BandCombineLeaf.leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right)) : Buffer.Content<Element>}:
  +left_exact = BandShapeCount.count_refinement_exact(geometry,rows,Buffer.length(~Element,left),BandShape.count(~Element,geometry,first,rows,left,width_positive,left_metadata,left_ready))
  +right_exact = BandShapeCount.count_refinement_exact(geometry,rows,Buffer.length(~Element,right),BandShape.count(~Element,geometry,first,rows,right,width_positive,right_metadata,right_ready))
  +descriptor = TensorVectorDescriptorRefinement.from_count(geometry,rows,Buffer.length(~Element,left),BandShape.count(~Element,geometry,first,rows,left,width_positive,left_metadata,left_ready))
  Equal.trans(Buffer.Content<Element>,
    Buffer.contents(~Element,Bands.combine_leaf_with(~Element,~kernel,Bands.local_count(geometry,rows),Buffer.pack(~Element,left),Buffer.pack(~Element,right))),
    BandCombineLeaf.refinement_leaf_model(~Element,~operation,Bands.local_count(geometry,rows),left,right),
    BandCombineLeaf.leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right)),
    BandCombineLeaf.refinement_leaf(~Element,~operation,~kernel,~kernel_law,Bands.local_count(geometry,rows),left,right),
    BandCombineLeaf.checked_leaf_model(~Element,~operation,Bands.local_count(geometry,rows),left,right,left_exact,right_exact,left_ready,right_ready,descriptor))

def LeafContract(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,+content: Buffer.Content<Element>) -> Data:
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.RowBand{content,first,rows}),
    Shape.Both<Buffer.ContentReady<Element,content>,BandShape.Leaf(geometry,first,rows,Buffer.content_length(~Element,content))>>

def leaf(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+first: U32,+rows: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},
  +left_metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,left)),+right_metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,right)),
  +left_ready: Buffer.Sized<Element,left>,+right_ready: Buffer.Sized<Element,right>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.RowBand{Buffer.content(~Element,left),first,rows}),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.RowBand{Buffer.content(~Element,right),first,rows})) ->
  LeafContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,first,rows,
    Buffer.contents(~Element,Bands.combine_leaf_with(~Element,~kernel,Bands.local_count(geometry,rows),Buffer.pack(~Element,left),Buffer.pack(~Element,right)))):
  match left right:
    case Buffer.Model{+left_tree,+left_length,+left_certificate} Buffer.Model{+right_tree,+right_length,+right_certificate}:
      +left = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
      +right = {Buffer.Model{right_tree,right_length,right_certificate} : Buffer.Model<Element>}
      %Equal.sym(Buffer.Content<Element>,
        Buffer.contents(~Element,Bands.combine_leaf_with(~Element,~kernel,Bands.local_count(geometry,rows),Buffer.pack(~Element,left),Buffer.pack(~Element,right))),
        BandCombineLeaf.leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right)),
        equation(~Element,~operation,~kernel,~kernel_law,geometry,first,rows,left,right,width_positive,left_metadata,right_metadata,left_ready,right_ready)) :
        LeafContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,first,rows,_)
      Shape.Both{BandCombineLeaf.semantics_leaf(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,geometry,first,rows,left,right,
          left_ready,left_values,right_values),
        Shape.Both{BandCombineLeaf.leaf_ready(~Element,~operation,left,right,left_ready),left_metadata}}

def leaf_contract(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+first: U32,+rows: U32,content: Buffer.Content<Element>,
  evidence: LeafContract(~Element,~Context,~value,context,geometry,first,rows,content)) ->
  BandValues.PipelineContract(~Element,~Context,~value,context,geometry,Model.RowBand{content,first,rows}):
  Buffer.Content{tree,length} = content
  evidence

def pipeline_model(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},common: Each.Common(~Element,left,right),
  left_metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,left)),right_metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,right)),
  left_ready: Model.Ready(~Element,left),right_ready: Model.Ready(~Element,right),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  BandValues.PipelineContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
    BandCombineLeaf.model(~Element,~operation,geometry,left,right)):
  match left right:
    case Model.Leaf{Buffer.Model{+left_tree,+left_length,+left_certificate},+first,+rows}
      Model.Leaf{Buffer.Model{+right_tree,+right_length,+right_certificate},+other_first,+other_rows}:
      Shape.Both{+same_first,+same_rows} = common
      +left = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
      +right = {Buffer.Model{right_tree,right_length,right_certificate} : Buffer.Model<Element>}
      +right_metadata = {
        %Equal.sym(U32,first,other_first,same_first) : BandShape.Leaf(geometry,_,other_rows,Buffer.length(~Element,right))
        right_metadata : BandShape.Leaf(geometry,first,other_rows,Buffer.length(~Element,right))}
      +right_metadata = {
        %Equal.sym(U32,rows,other_rows,same_rows) : BandShape.Leaf(geometry,first,_,Buffer.length(~Element,right))
        right_metadata : BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,right))}
      +right_values = {
        %Equal.sym(U32,first,other_first,same_first) : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.RowBand{Buffer.content(~Element,right),_,other_rows})
        right_values : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.RowBand{Buffer.content(~Element,right),first,other_rows})}
      +right_values = {
        %Equal.sym(U32,rows,other_rows,same_rows) : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.RowBand{Buffer.content(~Element,right),first,_})
        right_values : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.RowBand{Buffer.content(~Element,right),first,rows})}
      leaf_contract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,first,rows,
        BandCombineLeaf.refinement_leaf_model(~Element,~operation,Bands.local_count(geometry,rows),left,right),
        %BandCombineLeaf.refinement_leaf(~Element,~operation,~kernel,~kernel_law,Bands.local_count(geometry,rows),left,right) :
          LeafContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,first,rows,_)
        leaf(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,first,rows,left,right,
          width_positive,left_metadata,right_metadata,left_ready,right_ready,left_values,right_values))
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      Shape.Both{left_common,right_common} = common
      Shape.Both{left_left_metadata,left_right_metadata} = left_metadata
      Shape.Both{right_left_metadata,right_right_metadata} = right_metadata
      Shape.Both{left_left_ready,left_right_ready} = left_ready
      Shape.Both{right_left_ready,right_right_ready} = right_ready
      Shape.Both{left_left_values,left_right_values} = left_values
      Shape.Both{right_left_values,right_right_values} = right_values
      BandValues.pipeline_joined(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
        BandCombineLeaf.model(~Element,~operation,geometry,left,other_left),BandCombineLeaf.model(~Element,~operation,geometry,right,other_right),
        pipeline_model(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,left,other_left,width_positive,left_common,
          left_left_metadata,right_left_metadata,left_left_ready,right_left_ready,left_left_values,right_left_values),
        pipeline_model(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,right,other_right,width_positive,right_common,
          left_right_metadata,right_right_metadata,left_right_ready,right_right_ready,left_right_values,right_right_values))
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{+other_left,+other_right}:
      Empty.absurd(BandValues.PipelineContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
        BandCombineLeaf.model(~Element,~operation,geometry,Model.Leaf{buffer,first,rows},Model.Fork{other_left,other_right})),common)
    case Model.Fork{+left,+right} Model.Leaf{+buffer,+first,+rows}:
      Empty.absurd(BandValues.PipelineContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
        BandCombineLeaf.model(~Element,~operation,geometry,Model.Fork{left,right},Model.Leaf{buffer,first,rows})),common)

def pipeline_combine(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},same: {Model.layout(~Element,left) == Model.layout(~Element,right) : Bands.Plan},
  left_metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,left)),right_metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,right)),
  left_ready: Model.Ready(~Element,left),right_ready: Model.Ready(~Element,right),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  BandValues.PipelineContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
    Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))),
    BandCombineLeaf.model(~Element,~operation,geometry,left,right),BandCombineLeaf.combine(~Element,~operation,~kernel,~kernel_law,geometry,parallel,left,right)) :
    BandValues.PipelineContract(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,_)
  pipeline_model(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,left,right,width_positive,Each.matched(~Element,left,right,same),
    left_metadata,right_metadata,left_ready,right_ready,left_values,right_values)

def combine_well_formed(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},same: {Model.layout(~Element,left) == Model.layout(~Element,right) : Bands.Plan},
  left_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right))) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))>>:
  BandShape.WellFormed{left_metadata,left_ready,coverage} = left_shape
  BandShape.WellFormed{right_metadata,right_ready,right_coverage} = right_shape
  BandValues.pipeline_well_formed(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
    Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))),
    pipeline_combine(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,left,right,width_positive,same,
      left_metadata,right_metadata,Model.ready(~Element,left,left_ready),Model.ready(~Element,right,right_ready),left_values,right_values),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))),
      Model.layout(~Element,left),BandCombineLeaf.combine_layout(~Element,~operation,~kernel,~kernel_law,geometry,parallel,left,right)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    %Equal.sym(Bands.Plan,Model.layout(~Element,left),Model.content_layout(~Element,Model.content(~Element,left)),Model.layout_matches_content(~Element,left)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    coverage)

def observed_layout(~Element: Data,+model: Model.Model<Element>,-owner: Bands.Bands(~Element),equation: {Model.pack(~Element,model) == owner : Bands.Bands(~Element)}) ->
  {Model.layout(~Element,model) == Model.content_layout(~Element,Model.contents(~Element,owner)) : Bands.Plan}:
  Equal.trans(Bands.Plan,Model.layout(~Element,model),Model.content_layout(~Element,Model.content(~Element,model)),Model.content_layout(~Element,Model.contents(~Element,owner)),
    Model.layout_matches_content(~Element,model),
    Equal.sym(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,owner)),Model.content_layout(~Element,Model.content(~Element,model)),
      Equal.cong(Model.Content<Element>,Bands.Plan,content => Model.content_layout(~Element,content),Model.contents(~Element,owner),Model.content(~Element,model),Model.observed_contents(~Element,model,owner,equation))))

def owners_observed(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,-left: Bands.Bands(~Element),-right: Bands.Bands(~Element),
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},+same: {Model.content_layout(~Element,Model.contents(~Element,left)) == Model.content_layout(~Element,Model.contents(~Element,right)) : Bands.Plan},
  left_shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.contents(~Element,left)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.contents(~Element,right)),
  left_observation: Model.Observation<Element,left>,right_observation: Model.Observation<Element,right>) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))>>:
  Model.Observation{+left_model,+left_equation} = left_observation
  Model.Observation{+right_model,+right_equation} = right_observation
  +left_content = Model.observed_contents(~Element,left_model,left,left_equation)
  +right_content = Model.observed_contents(~Element,right_model,right,right_equation)
  +left_shape = {
    %left_content : BandShape.WellFormed<Element,geometry,_>
    left_shape : BandShape.WellFormed<Element,geometry,Model.content(~Element,left_model)>}
  +right_shape = {
    %right_content : BandShape.WellFormed<Element,geometry,_>
    right_shape : BandShape.WellFormed<Element,geometry,Model.content(~Element,right_model)>}
  +left_values = {
    %left_content : BandValues.Values(~Element,~Left,~left_value,left_context,geometry,_)
    left_values : BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left_model))}
  +right_values = {
    %right_content : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,_)
    right_values : BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.content(~Element,right_model))}
  +matched = Equal.trans(Bands.Plan,Model.layout(~Element,left_model),Model.content_layout(~Element,Model.contents(~Element,left)),Model.layout(~Element,right_model),
    observed_layout(~Element,left_model,left,left_equation),
    Equal.trans(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,left)),Model.content_layout(~Element,Model.contents(~Element,right)),Model.layout(~Element,right_model),
      same,Equal.sym(Bands.Plan,Model.layout(~Element,right_model),Model.content_layout(~Element,Model.contents(~Element,right)),observed_layout(~Element,right_model,right,right_equation))))
  %left_equation :
    Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
        Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,_,right))),
      BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,_,right))>>
  %right_equation :
    Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
        Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left_model),_))),
      BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left_model),_))>>
  combine_well_formed(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,left_model,right_model,width_positive,matched,
    left_shape,right_shape,left_values,right_values)

def combine_owners(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,left: Bands.Bands(~Element),right: Bands.Bands(~Element),
  width_positive: {BandShape.width_positive(geometry) == True{} : Bool},same: {Model.content_layout(~Element,Model.contents(~Element,left)) == Model.content_layout(~Element,Model.contents(~Element,right)) : Bands.Plan},
  left_shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.contents(~Element,left)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.contents(~Element,right))) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))>>:
  owners_observed(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,left,right,width_positive,same,left_shape,right_shape,left_values,right_values,
    Model.observe(~Element,left),Model.observe(~Element,right))

# A zero-count binary traversal preserves its left content even when the
# checked call rejects. Empty geometries derive these counts from source data.
def empty_refinement_selected(~Element: Data,~operation: Element -> Element -> Element,accepted: Bool,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,+left_length: Nat,+right_length: Nat) ->
  {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
      OperationsLaws.PairCall{accepted,Some{0},Vector.vector(0),Vector.vector(0),right_length,left_length},right,left)) == Buffer.Content{left,left_length} : Buffer.Content<Element>}:
  match accepted:
    case False{}:
      Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,left_length},Tree.reflect(Element,Tree.pack(Element,left)),left,StorageLaws.reflect_packed(Element,left))
    case True{}:
      Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,left_length},Tree.reflect(Element,Tree.pack(Element,left)),left,StorageLaws.reflect_packed(Element,left))

def leaf_model(~Element: Data,~operation: Element -> Element -> Element,left: Buffer.Model<Element>,right: Buffer.Model<Element>) ->
  {BandCombineLeaf.refinement_leaf_model(~Element,~operation,0,left,right) == Buffer.content(~Element,left) : Buffer.Content<Element>}:
  match left right:
    case Buffer.Model{+left_tree,+left_length,+left_certificate} Buffer.Model{+right_tree,+right_length,+right_certificate}:
      +left = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
      +right = {Buffer.Model{right_tree,right_length,right_certificate} : Buffer.Model<Element>}
      Equal.trans(Buffer.Content<Element>,BandCombineLeaf.refinement_leaf_model(~Element,~operation,0,left,right),Buffer.Content{Tree.reflect(Element,Tree.pack(Element,left_tree)),left_length},Buffer.Content{left_tree,left_length},
        empty_refinement_selected(~Element,~operation,BandCombineLeaf.guard(~Element,0,left,right),Tree.reflect(Element,Tree.pack(Element,left_tree)),Tree.reflect(Element,Tree.pack(Element,right_tree)),left_length,right_length),
        Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,left_length},Tree.reflect(Element,Tree.pack(Element,left_tree)),left_tree,StorageLaws.reflect_packed(Element,left_tree)))

def empty_refinement_model(~Element: Data,~operation: Element -> Element -> Element,+geometry: Bands.Geometry,left: Model.Model<Element>,right: Model.Model<Element>,counts: Arrive.Counts(~Element,geometry,left)) ->
  {BandCombineLeaf.model(~Element,~operation,geometry,left,right) == Model.content(~Element,left) : Model.Content<Element>}:
  match left right:
    case Model.Leaf{+left,+first,+rows} Model.Leaf{+right,other_first,other_rows}:
      Equal.cong(Buffer.Content<Element>,Model.Content<Element>,buffer => Model.RowBand{buffer,first,rows},BandCombineLeaf.refinement_leaf_model(~Element,~operation,Bands.local_count(geometry,rows),left,right),Buffer.content(~Element,left),
        %Equal.sym(U32,Bands.local_count(geometry,rows),0,counts) : {BandCombineLeaf.refinement_leaf_model(~Element,~operation,_,left,right) == Buffer.content(~Element,left) : Buffer.Content<Element>}
        leaf_model(~Element,~operation,left,right))
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      Shape.Both{left_counts,right_counts} = counts
      %Equal.sym(Model.Content<Element>,BandCombineLeaf.model(~Element,~operation,geometry,left,other_left),Model.content(~Element,left),empty_refinement_model(~Element,~operation,geometry,left,other_left,left_counts)) :
        {Model.Joined{_,BandCombineLeaf.model(~Element,~operation,geometry,right,other_right)} == Model.Joined{Model.content(~Element,left),Model.content(~Element,right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,BandCombineLeaf.model(~Element,~operation,geometry,right,other_right),Model.content(~Element,right),empty_refinement_model(~Element,~operation,geometry,right,other_right,right_counts)) :
        {Model.Joined{Model.content(~Element,left),_} == Model.Joined{Model.content(~Element,left),Model.content(~Element,right)} : Model.Content<Element>}
      {==}
    case Model.Leaf{buffer,first,rows} Model.Fork{other_left,other_right}: {==}
    case Model.Fork{left,right} Model.Leaf{buffer,first,rows}: {==}

def empty_refinement_combine(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>,counts: Arrive.Counts(~Element,geometry,left)) ->
  {Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))) == Model.content(~Element,left) : Model.Content<Element>}:
  Equal.trans(Model.Content<Element>,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))),
    BandCombineLeaf.model(~Element,~operation,geometry,left,right),Model.content(~Element,left),BandCombineLeaf.combine(~Element,~operation,~kernel,~kernel_law,geometry,parallel,left,right),empty_refinement_model(~Element,~operation,geometry,left,right,counts))

def combined_well_formed(~Element: Data,~Context: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +counts: Arrive.Counts(~Element,geometry,left),source: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))>>:
  BandShape.WellFormed{+metadata,+ready,coverage} = source
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right))),Model.content(~Element,left),
    empty_refinement_combine(~Element,~operation,~kernel,~kernel_law,geometry,parallel,left,right,counts)) :
    Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,_),BandShape.WellFormed<Element,geometry,_>>
  Shape.Both{Arrive.empty_band_values(~Element,~Context,~value,context,geometry,left,counts,metadata,Model.ready(~Element,left,ready)),BandShape.WellFormed{metadata,ready,coverage}}

def modeled(~Element: Data,~Context: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~value: Context -> U32 -> Element,
  +context: Context,+geometry: Bands.Geometry,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>,empty: BandShapeCount.EmptyGeometry<geometry>,
  source: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left),Model.pack(~Element,right)))>>:
  BandShape.WellFormed{+metadata,ready,coverage} = source
  combined_well_formed(~Element,~Context,~operation,~kernel,~kernel_law,~value,context,geometry,parallel,left,right,
    Arrive.empty_counts(~Element,geometry,left,empty,metadata),BandShape.WellFormed{metadata,ready,coverage})

def empty_alignment_observed(~Element: Data,~Context: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~value: Context -> U32 -> Element,
  +context: Context,+geometry: Bands.Geometry,+parallel: Bool,-left: Bands.Bands(~Element),-right: Bands.Bands(~Element),+empty: BandShapeCount.EmptyGeometry<geometry>,
  source: BandShape.WellFormed<Element,geometry,Model.contents(~Element,left)>,left_observation: Model.Observation<Element,left>,right_observation: Model.Observation<Element,right>) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))>>:
  Model.Observation{+left_model,+left_equation} = left_observation
  Model.Observation{+right_model,+right_equation} = right_observation
  +left_content = Model.observed_contents(~Element,left_model,left,left_equation)
  +canonical = {
    %left_content : BandShape.WellFormed<Element,geometry,_>
    source : BandShape.WellFormed<Element,geometry,Model.content(~Element,left_model)>}
  %left_equation :
    Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,_,right))),
      BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,_,right))>>
  %right_equation :
    Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left_model),_))),
      BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,Model.pack(~Element,left_model),_))>>
  modeled(~Element,~Context,~operation,~kernel,~kernel_law,~value,context,geometry,parallel,left_model,right_model,empty,canonical)

def empty_alignment_owners(~Element: Data,~Context: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~value: Context -> U32 -> Element,
  +context: Context,+geometry: Bands.Geometry,+parallel: Bool,left: Bands.Bands(~Element),right: Bands.Bands(~Element),empty: BandShapeCount.EmptyGeometry<geometry>,
  source: BandShape.WellFormed<Element,geometry,Model.contents(~Element,left)>) ->
  Shape.Both<BandValues.Values(~Element,~Context,~value,context,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))>>:
  empty_alignment_observed(~Element,~Context,~operation,~kernel,~kernel_law,~value,context,geometry,parallel,left,right,empty,source,Model.observe(~Element,left),Model.observe(~Element,right))

# Bands of one geometry in one plan combine into well formed bands that hold
# the operation of their values. An empty geometry is decided here once.
def combined(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+parallel: Bool,left: Bands.Bands(~Element),right: Bands.Bands(~Element),
  same: {Model.content_layout(~Element,Model.contents(~Element,left)) == Model.content_layout(~Element,Model.contents(~Element,right)) : Bands.Plan},
  left_shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,left)>,right_shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,right)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.contents(~Element,left)),
  right_values: BandValues.Values(~Element,~Right,~right_value,right_context,geometry,Model.contents(~Element,right)),choice: Arrive.DimensionChoice(geometry)) ->
  Shape.Both<BandValues.Values(~Element,~BandCombineLeaf.Context<Left,Right>,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),BandCombineLeaf.Context{left_context,right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))>>:
  match choice:
    case Inr{dimensions}:
      Arrive.Dimensions{channels_positive,height_positive,width_positive} = dimensions
      combine_owners(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,left,right,
        Arrive.metadata_width_positive(geometry,width_positive),same,left_shape,right_shape,left_values,right_values)
    case Inl{empty}:
      empty_alignment_owners(~Element,~BandCombineLeaf.Context<Left,Right>,~operation,~kernel,~kernel_law,~BandCombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),
        BandCombineLeaf.Context{left_context,right_context},geometry,parallel,left,right,BandShapeCount.empty_geometry_classified(geometry,empty),left_shape)
