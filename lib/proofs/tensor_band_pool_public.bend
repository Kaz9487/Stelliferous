# The public pool of a band tensor. The input arrives in the layout of its
# planes: its own plan when its bands split the image rows of its shape,
# otherwise its planes as the channels of one image in one band. One band
# pool runs there. When the call returns Ready with a nonempty output, the
# result is Normal and holds the ordered pool of the logical input at every
# index.
import Base
import ../tensor.bend as Tensors
import ../tensor_window.bend as Window
import ../tensor_layout.bend as Layout
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ../tensor_band_pool.bend as BandPool
import ../tensor_band_schedule.bend as Schedule
import ../tensor_access.bend as Access
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../tensor_operations.bend as Operations
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_count.bend as BandCount
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./tensor_band_target.bend as Target
import ./tensor_band_arrive.bend as Arrive
import ./tensor_band_pool_execution.bend as BandPoolExecution
import ./tensor_band_pool_samples.bend as BandPoolSamples
import ./tensor_band_pool_window_values.bend as BandPoolWindowValues
import ./tensor_band_pool_output_domain.bend as OutputDomain
import ./tensor_pool_scalar_bounds.bend as Scalar
import ./tensor_contract.bend as Contract
import ./tensor_pool_leaf_values.bend as PoolLeaf
import ./tensor_sweep_values.bend as Sweeps
import ./tensor_band_map.bend as BandMap
import ./storage_shape.bend as Shape
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding

def window_geometry(input: Bands.Geometry,+kernel: U32,+stride: U32,+padding: U32) -> BandWindow.Geometry:
  Bands.Geometry{images,channels,+height,+width} = input
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,
    Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)}

def planes_shape(+shape: List<&2,U32>,+geometry: BandWindow.Geometry) -> List<&2,U32>:
  Window.middle_shape(shape,OutputDomain.pool(geometry))

# The pool the call reads from a shape of planes is the pool of their
# geometry.
def planes_pool(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,
  planes: BandCount.Planes<shape,images,channels,height,width>) ->
  {Window.geometry(shape,kernel,stride,padding) == OutputDomain.pool(window_geometry(Bands.Geometry{images,channels,height,width},kernel,stride,padding)) : Access.Pool}:
  BandCount.Planes{ranked,+rows,+columns,leading} = planes
  +row_extent = Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n))
  +column_extent = Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n))
  +geometry = window_geometry(Bands.Geometry{images,channels,height,width},kernel,stride,padding)
  %Equal.sym(U32,row_extent,height,rows) :
    {Access.Pool{_,column_extent,kernel,stride,padding,Window.out_extent(_,kernel,stride,padding),Window.out_extent(column_extent,kernel,stride,padding)} ==
      OutputDomain.pool(geometry) : Access.Pool}
  %Equal.sym(U32,column_extent,width,columns) :
    {Access.Pool{height,_,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(_,kernel,stride,padding)} ==
      OutputDomain.pool(geometry) : Access.Pool}
  {==}

# What the checks of the call give: a legal pool on positive axes, and
# counts of the column-swept planes and of the output.
def valid_axes(+shape: List<&2,U32>,+pool: Access.Pool,valid: {Window.valid(shape,pool) == True{} : Bool}) ->
  {Operations.pool_geometry(pool) && U32.is_lt(0,BandPoolWindowValues.pool_height(pool)) && U32.is_lt(0,BandPoolWindowValues.pool_width(pool)) == True{} : Bool}:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  +axes = (Operations.pool_geometry(pool) && U32.is_lt(0,height) && U32.is_lt(0,width) : Bool)
  +middle = Views.has_count(Tensors.count_of(Window.middle_shape(shape,pool)))
  Boolean.left(axes,middle,Boolean.left((axes && middle : Bool),Views.has_count(Tensors.count_of(Window.output_shape(shape,pool))),valid))

def valid_middle(+shape: List<&2,U32>,+pool: Access.Pool,valid: {Window.valid(shape,pool) == True{} : Bool}) ->
  {Views.has_count(Tensors.count_of(Window.middle_shape(shape,pool))) == True{} : Bool}:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  +axes = (Operations.pool_geometry(pool) && U32.is_lt(0,height) && U32.is_lt(0,width) : Bool)
  +middle = Views.has_count(Tensors.count_of(Window.middle_shape(shape,pool)))
  Boolean.right(axes,middle,Boolean.left((axes && middle : Bool),Views.has_count(Tensors.count_of(Window.output_shape(shape,pool))),valid))

def valid_output(+shape: List<&2,U32>,+pool: Access.Pool,valid: {Window.valid(shape,pool) == True{} : Bool}) ->
  {Views.has_count(Tensors.count_of(Window.output_shape(shape,pool))) == True{} : Bool}:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  +axes = (Operations.pool_geometry(pool) && U32.is_lt(0,height) && U32.is_lt(0,width) : Bool)
  Boolean.right((axes && Views.has_count(Tensors.count_of(Window.middle_shape(shape,pool))) : Bool),Views.has_count(Tensors.count_of(Window.output_shape(shape,pool))),valid)

def counted_result(+shape: List<&2,U32>,result: Maybe<&2,U32>,accepted: {Tensors.count_of(shape) == result : Maybe<&2,U32>},
  found: {Views.has_count(result) == True{} : Bool}) -> Global.Counted<shape>:
  match result:
    case Some{count}: Global.Counted{count,accepted}
    case None{}: Empty.absurd(Global.Counted<shape>,WordEncoding.false_true(found))

# The checks of the call, the planes of the shape and a nonempty output
# select the band pool of the geometry of the planes.
def selected(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,
  +matched: {BandCount.planes_rows(shape,Bands.Geometry{images,channels,height,width}) == True{} : Bool},
  +axes: {Operations.pool_geometry(OutputDomain.pool(window_geometry(Bands.Geometry{images,channels,height,width},kernel,stride,padding))) &&
    U32.is_lt(0,height) && U32.is_lt(0,width) == True{} : Bool},
  +nonempty: {Shapes.has_zero(OutputDomain.output_shape(shape,window_geometry(Bands.Geometry{images,channels,height,width},kernel,stride,padding))) == False{} : Bool}) ->
  {OutputDomain.enabled(shape,window_geometry(Bands.Geometry{images,channels,height,width},kernel,stride,padding)) == True{} : Bool}:
  +geometry = window_geometry(Bands.Geometry{images,channels,height,width},kernel,stride,padding)
  +legal = Operations.pool_geometry(OutputDomain.pool(geometry))
  +rows = U32.is_lt(0,height)
  +columns = U32.is_lt(0,width)
  +shape_ok = BandCount.planes_rows(shape,Bands.Geometry{images,channels,height,width})
  +pair = Boolean.left((legal && rows : Bool),columns,axes)
  Boolean.and_true((shape_ok && legal && rows && columns : Bool),Bool.not(Shapes.has_zero(OutputDomain.output_shape(shape,geometry))),
    Boolean.and_true((shape_ok && legal && rows : Bool),columns,
      Boolean.and_true((shape_ok && legal : Bool),rows,Boolean.and_true(shape_ok,legal,matched,Boolean.left(legal,rows,pair)),Boolean.right(legal,rows,pair)),
      Boolean.right((legal && rows : Bool),columns,axes)),
    %Equal.sym(Bool,Shapes.has_zero(OutputDomain.output_shape(shape,geometry)),False{},nonempty) : {Bool.not(_) == True{} : Bool} {==})

# Every element of the result is the pool of its pixel: the fold, from pad,
# of the k lines of its window, each the fold, from pad, of its k samples.
def Contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,
  +logical: Logical,+pool: Access.Pool,+pad: Element,-tensor: Tensors.Tensor<Element>) -> Data:
  Global.Contract(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),
    PoolLeaf.context(~Element,~Logical,logical,pool,0,pad),tensor)

# The band route of the call: the pool of every band, checked.
def band_route(~Element: Data,~operation: Element -> Element -> Element,+name: String,+pad: Element,+input: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +kernel: U32,+stride: U32,+padding: U32,bands: Bands.Bands(~Element)) -> Tensors.Tensor<Element>:
  +geometry = window_geometry(input,kernel,stride,padding)
  Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},BandPoolExecution.values_output_geometry(geometry),OutputDomain.output_shape(shape,geometry),fill,workers,
    BandPool.execute(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},bands))

# What a result of the band route holds, for the call the scope describes.
def RouteContract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+scope: BandPoolSamples.Scope<Element,Logical,original>,
  -tensor: Tensors.Tensor<Element>) -> Data:
  BandPoolSamples.Scope{machine,+logical,shape,+geometry,+pad,fill,workers,planes,count,source,selected,planes_counted,counted,well_formed,values,envelope} = scope
  Contract(~Element,~Logical,~original,~operation,logical,OutputDomain.pool(geometry),pad,tensor)

# A Ready result of the band route on well formed bands of the input
# geometry with complete values has the contract of the pool.
def band_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +machine: Scalar.Observation,+logical: Logical,+input: Bands.Geometry,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,+source: Model.Model<Element>,
  +selection: {OutputDomain.enabled(shape,window_geometry(input,kernel,stride,padding)) == True{} : Bool},
  planes_output: Global.Counted<planes_shape(shape,window_geometry(input,kernel,stride,padding))>,
  output: Global.Counted<OutputDomain.output_shape(shape,window_geometry(input,kernel,stride,padding))>,input_counted: Global.Counted<shape>,
  well_formed: BandShape.WellFormed<Element,input,Model.contents(~Element,Model.pack(~Element,source))>,
  source_values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,Model.pack(~Element,source))),
  accepted: {Contract.tensor_ready(~Element,band_route(~Element,~operation,name,pad,input,shape,fill,workers,kernel,stride,padding,Model.pack(~Element,source))) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,OutputDomain.pool(window_geometry(input,kernel,stride,padding)),pad,
    band_route(~Element,~operation,name,pad,input,shape,fill,workers,kernel,stride,padding,Model.pack(~Element,source))):
  Bands.Geometry{+images,+channels,+height,+width} = input
  Global.Counted{+planes,+planes_accepted} = planes_output
  Global.Counted{+count,+counted} = output
  +input = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +geometry = window_geometry(input,kernel,stride,padding)
  +out_geometry = BandPoolExecution.values_output_geometry(geometry)
  +scope = {BandPoolSamples.Scope{machine,logical,shape,geometry,pad,fill,workers,planes,count,source,selection,planes_accepted,counted,
    %Model.packed_contents(~Element,source) : BandShape.WellFormed<Element,input,_> well_formed,
    %Model.packed_contents(~Element,source) : BandValues.Values(~Element,~Logical,~original,logical,input,_) source_values,
    BandPoolSamples.source_envelope(machine,shape,geometry,input_counted,selection)} : BandPoolSamples.Scope<Element,Logical,original>}
  Global.transformed(~Element,~BandPoolSamples.Scope<Element,Logical,original>,~BandPoolSamples.OutputValues(~Element,~Logical,~original,~operation),~RouteContract(~Element,~Logical,~original,~operation),scope,
    name,Tensors.InvalidArgument{shape},out_geometry,OutputDomain.output_shape(shape,geometry),fill,workers,Scalar.limits(machine),
    BandGather.rows_room(out_geometry,Scalar.limits(machine),Machine.capacity(Scalar.limits(machine)),OutputDomain.from_selector(machine,shape,geometry,count,selection,counted)),
    BandPool.execute(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Model.pack(~Element,source)),
    usable => BandPoolSamples.executed(~Element,~Logical,~original,~operation,scope,usable),
    Global.Counted{count,counted},OutputDomain.count_shape_elements(shape,geometry,OutputDomain.selected_shape(shape,geometry,selection)),
    bands => normal => facts => Shape.Both{normal,facts},
    accepted)

# Arrived bands are read through their model.
def ran(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +machine: Scalar.Observation,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32,+layout: Bands.Plan,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,
  +kernel: U32,+stride: U32,+padding: U32,-bands: Bands.Bands(~Element),observation: Model.Observation<Element,bands>,
  arrived: Arrive.Arrived(~Element,~Logical,~original,logical,Bands.Geometry{images,channels,height,width},layout,bands),
  +selection: {OutputDomain.enabled(shape,window_geometry(Bands.Geometry{images,channels,height,width},kernel,stride,padding)) == True{} : Bool},
  planes_output: Global.Counted<planes_shape(shape,window_geometry(Bands.Geometry{images,channels,height,width},kernel,stride,padding))>,
  output: Global.Counted<OutputDomain.output_shape(shape,window_geometry(Bands.Geometry{images,channels,height,width},kernel,stride,padding))>,input_counted: Global.Counted<shape>,
  accepted: {Contract.tensor_ready(~Element,band_route(~Element,~operation,name,pad,Bands.Geometry{images,channels,height,width},shape,fill,workers,kernel,stride,padding,bands)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,OutputDomain.pool(window_geometry(Bands.Geometry{images,channels,height,width},kernel,stride,padding)),pad,
    band_route(~Element,~operation,name,pad,Bands.Geometry{images,channels,height,width},shape,fill,workers,kernel,stride,padding,bands)):
  Model.Observation{+source,+equation} = observation
  Shape.Both{same,Shape.Both{well_formed,values}} = arrived
  %equation : Contract(~Element,~Logical,~original,~operation,logical,OutputDomain.pool(window_geometry(Bands.Geometry{images,channels,height,width},kernel,stride,padding)),pad,
    band_route(~Element,~operation,name,pad,Bands.Geometry{images,channels,height,width},shape,fill,workers,kernel,stride,padding,_))
  band_contract(~Element,~Logical,~original,~operation,name,machine,logical,Bands.Geometry{images,channels,height,width},shape,pad,fill,workers,kernel,stride,padding,source,
    selection,planes_output,output,input_counted,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,source),bands,equation) : BandShape.WellFormed<Element,Bands.Geometry{images,channels,height,width},Model.contents(~Element,_)> well_formed,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,source),bands,equation) :
      BandValues.Values(~Element,~Logical,~original,logical,Bands.Geometry{images,channels,height,width},Model.contents(~Element,_)) values,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,source),bands,equation) :
      {Contract.tensor_ready(~Element,band_route(~Element,~operation,name,pad,Bands.Geometry{images,channels,height,width},shape,fill,workers,kernel,stride,padding,_)) == True{} : Bool} accepted)

# The input arrives in a layout of the geometry whose planes its shape holds,
# and the band pool runs there.
def landed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32,+layout: Bands.Plan,+split: Bool,+other: Bands.Geometry,+model: Model.Model<Element>,
  +shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  +arrival: Target.Arrival<shape,Bands.Geometry{images,channels,height,width},layout>,
  +matched: {BandCount.planes_rows(shape,Bands.Geometry{images,channels,height,width}) == True{} : Bool},
  +valid: {Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) == True{} : Bool},
  +nonempty: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  well_formed: BandShape.WellFormed<Element,other,Model.content(~Element,model)>,coherent: Global.Coherent(shape,other),
  values: BandValues.Values(~Element,~Logical,~original,logical,other,Model.content(~Element,model)),
  accepted: {Contract.tensor_ready(~Element,Window.planned(~Element,~operation,name,pad,Window.geometry(shape,kernel,stride,padding),
    Tensors.Tensor{Model.pack(~Element,model),other,shape,fill,Tensors.Usable{},workers},shape,fill,workers,
    Schedule.Target{Bands.Geometry{images,channels,height,width},layout,split})) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,Window.geometry(shape,kernel,stride,padding),pad,
    Window.planned(~Element,~operation,name,pad,Window.geometry(shape,kernel,stride,padding),Tensors.Tensor{Model.pack(~Element,model),other,shape,fill,Tensors.Usable{},workers},shape,fill,workers,
      Schedule.Target{Bands.Geometry{images,channels,height,width},layout,split})):
  +input = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +geometry = window_geometry(input,kernel,stride,padding)
  +pooled = planes_pool(shape,images,channels,height,width,kernel,stride,padding,BandCount.decoded_planes(shape,images,channels,height,width,matched))
  +valid_pool = {%pooled : {Window.valid(shape,_) == True{} : Bool} valid : {Window.valid(shape,OutputDomain.pool(geometry)) == True{} : Bool}}
  +nonempty_pool = {%pooled : {Shapes.has_zero(Window.output_shape(shape,_)) == False{} : Bool} nonempty :
    {Shapes.has_zero(OutputDomain.output_shape(shape,geometry)) == False{} : Bool}}
  %Equal.sym(Access.Pool,Window.geometry(shape,kernel,stride,padding),OutputDomain.pool(geometry),pooled) :
    Contract(~Element,~Logical,~original,~operation,logical,_,pad,
      Window.band_execute(~Element,~operation,name,pad,_,input,Tensors.arrive(~Element,input,layout,Tensors.Tensor{Model.pack(~Element,model),other,shape,fill,Tensors.Usable{},workers}),
        shape,Window.output_shape(shape,_),fill,workers))
  ran(~Element,~Logical,~original,~operation,name,Scalar.observe_envelope(),logical,images,channels,height,width,layout,shape,pad,fill,workers,kernel,stride,padding,
    Tensors.arrive(~Element,input,layout,Tensors.Tensor{Model.pack(~Element,model),other,shape,fill,Tensors.Usable{},workers}),
    Model.observe(~Element,Tensors.arrive(~Element,input,layout,Tensors.Tensor{Model.pack(~Element,model),other,shape,fill,Tensors.Usable{},workers})),
    Arrive.arrived(~Element,~Logical,~original,logical,shape,input,layout,other,fill,Tensors.Usable{},workers,model,arrival,well_formed,coherent,values),
    selected(shape,images,channels,height,width,kernel,stride,padding,matched,valid_axes(shape,OutputDomain.pool(geometry),valid_pool),nonempty_pool),
    counted_result(planes_shape(shape,geometry),Tensors.count_of(planes_shape(shape,geometry)),{==},valid_middle(shape,OutputDomain.pool(geometry),valid_pool)),
    counted_result(OutputDomain.output_shape(shape,geometry),Tensors.count_of(OutputDomain.output_shape(shape,geometry)),{==},valid_output(shape,OutputDomain.pool(geometry),valid_pool)),
    Target.arrival_counted(shape,input,layout,arrival),
    %pooled : {Contract.tensor_ready(~Element,Window.band_execute(~Element,~operation,name,pad,_,input,
      Tensors.arrive(~Element,input,layout,Tensors.Tensor{Model.pack(~Element,model),other,shape,fill,Tensors.Usable{},workers}),shape,Window.output_shape(shape,_),fill,workers)) == True{} : Bool}
    accepted)

# The planes of a counted shape, as the channels of one image, fit the
# machine.
def plane_domain(+shape: List<&2,U32>,+planes: U32,+rows: U32,+columns: U32,+kernel: U32,+stride: U32,+padding: U32,+counted: Global.Counted<shape>,
  +coherent: Global.Coherent(shape,Bands.Geometry{1,planes,rows,columns}),
  +selection: {OutputDomain.enabled(shape,window_geometry(Bands.Geometry{1,planes,rows,columns},kernel,stride,padding)) == True{} : Bool}) ->
  BandGather.Domain<Bands.Geometry{1,planes,rows,columns},Default.limits(),Machine.capacity(Default.limits())>:
  +geometry = window_geometry(Bands.Geometry{1,planes,rows,columns},kernel,stride,padding)
  Global.domain(shape,Bands.Geometry{1,planes,rows,columns},counted,coherent,
    BandPoolSamples.input_channels_positive(geometry,OutputDomain.positive_from_selector(shape,geometry,selection)),
    OutputDomain.selected_height(shape,geometry,selection),OutputDomain.selected_width(shape,geometry,selection))

# The layout the call chooses: the input's own when its bands split the image
# rows of its shape, otherwise its planes as the channels of one image in one
# band.
def chosen(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  rows: Bool,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32,+model: Model.Model<Element>,
  +shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,+planes: U32,
  +compared: {Bands.image_rows(shape,Bands.Geometry{images,channels,height,width}) == rows : Bool},
  +planes_exact: {Layout.planes(shape) == Some{planes} : Maybe<&2,U32>},
  +valid: {Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) == True{} : Bool},
  +nonempty: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Window.planned(~Element,~operation,name,pad,Window.geometry(shape,kernel,stride,padding),
    Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers},shape,fill,workers,
    Schedule.plane_chosen(rows,planes,Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n)),Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n)),
      Schedule.Target{Bands.Geometry{images,channels,height,width},Model.layout(~Element,model),Schedule.splits(Model.layout(~Element,model))}))) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,Window.geometry(shape,kernel,stride,padding),pad,
    Window.planned(~Element,~operation,name,pad,Window.geometry(shape,kernel,stride,padding),
      Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers},shape,fill,workers,
      Schedule.plane_chosen(rows,planes,Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n)),Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n)),
        Schedule.Target{Bands.Geometry{images,channels,height,width},Model.layout(~Element,model),Schedule.splits(Model.layout(~Element,model))}))):
  match rows:
    case True{}:
      Shape.Both{Shape.Both{well_formed,Shape.Both{+counted,+coherent}},values} = contract
      +input = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
      +owned = {%Model.packed_contents(~Element,model) : BandShape.WellFormed<Element,input,_> well_formed : BandShape.WellFormed<Element,input,Model.content(~Element,model)>}
      landed(~Element,~Logical,~original,~operation,name,logical,images,channels,height,width,Model.layout(~Element,model),Schedule.splits(Model.layout(~Element,model)),input,model,
        shape,pad,fill,workers,kernel,stride,padding,
        Target.owner_arrival(~Element,shape,input,Model.layout(~Element,model),Target.Owner{model,owned,counted,coherent,{==}}),
        BandCount.image_planes(shape,images,channels,height,width,compared),valid,nonempty,owned,coherent,
        %Model.packed_contents(~Element,model) : BandValues.Values(~Element,~Logical,~original,logical,input,_) values,accepted)
    case False{}:
      Shape.Both{Shape.Both{well_formed,Shape.Both{+counted,+coherent}},values} = contract
      +rank = List.length(&2,U32,shape)
      +plane_rows = Layout.extent_at(shape,Nat.sub(rank,2n))
      +plane_columns = Layout.extent_at(shape,Nat.sub(rank,1n))
      +plane = {Bands.Geometry{1,planes,plane_rows,plane_columns} : Bands.Geometry}
      +matched = BandCount.plane_planes(shape,planes,planes_exact)
      +plane_coherent = BandCount.planes_coherent(shape,1,planes,plane_rows,plane_columns,BandCount.decoded_planes(shape,1,planes,plane_rows,plane_columns,matched))
      +geometry = window_geometry(plane,kernel,stride,padding)
      +pooled = planes_pool(shape,1,planes,plane_rows,plane_columns,kernel,stride,padding,BandCount.decoded_planes(shape,1,planes,plane_rows,plane_columns,matched))
      +selection = selected(shape,1,planes,plane_rows,plane_columns,kernel,stride,padding,matched,
        valid_axes(shape,OutputDomain.pool(geometry),%pooled : {Window.valid(shape,_) == True{} : Bool} valid),
        %pooled : {Shapes.has_zero(Window.output_shape(shape,_)) == False{} : Bool} nonempty)
      landed(~Element,~Logical,~original,~operation,name,logical,1,planes,plane_rows,plane_columns,Bands.unsplit(plane),False{},Bands.Geometry{images,channels,height,width},model,
        shape,pad,fill,workers,kernel,stride,padding,
        Target.Arrival{counted,plane_coherent,
          Target.split_planned(plane,Default.limits(),Machine.capacity(Default.limits()),0n,
            plane_domain(shape,planes,plane_rows,plane_columns,kernel,stride,padding,counted,plane_coherent,selection))},
        matched,valid,nonempty,
        %Model.packed_contents(~Element,model) : BandShape.WellFormed<Element,Bands.Geometry{images,channels,height,width},_> well_formed,coherent,
        %Model.packed_contents(~Element,model) : BandValues.Values(~Element,~Logical,~original,logical,Bands.Geometry{images,channels,height,width},_) values,accepted)

# A Normal input offers its own geometry and plan to the plane layout.
def offered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32,+model: Model.Model<Element>,
  +shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,+planes: U32,
  +planes_exact: {Layout.planes(shape) == Some{planes} : Maybe<&2,U32>},
  +valid: {Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) == True{} : Bool},
  +nonempty: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Window.offered(~Element,~operation,name,pad,Window.geometry(shape,kernel,stride,padding),planes,shape,fill,workers,
    Tensors.offer(~Element,Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers}))) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,Window.geometry(shape,kernel,stride,padding),pad,
    Window.offered(~Element,~operation,name,pad,Window.geometry(shape,kernel,stride,padding),planes,shape,fill,workers,
      Tensors.offer(~Element,Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers}))):
  +offer = Target.offer(~Element,model,Bands.Geometry{images,channels,height,width},shape,fill,workers)
  +layout = Model.layout(~Element,model)
  %Equal.sym(Tensors.Tensor<Element> & Schedule.Target,
    Tensors.offer(~Element,Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers}),
    (Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers},
      Schedule.Target{Bands.Geometry{images,channels,height,width},layout,Schedule.splits(layout)}),offer) :
    Contract(~Element,~Logical,~original,~operation,logical,Window.geometry(shape,kernel,stride,padding),pad,
      Window.offered(~Element,~operation,name,pad,Window.geometry(shape,kernel,stride,padding),planes,shape,fill,workers,_))
  chosen(~Element,~Logical,~original,~operation,name,Bands.image_rows(shape,Bands.Geometry{images,channels,height,width}),logical,images,channels,height,width,model,
    shape,pad,fill,workers,kernel,stride,padding,planes,{==},planes_exact,valid,nonempty,contract,
    %offer : {Contract.tensor_ready(~Element,Window.offered(~Element,~operation,name,pad,Window.geometry(shape,kernel,stride,padding),planes,shape,fill,workers,_)) == True{} : Bool} accepted)

# The checks of the call pass for a Ready result, and a nonempty output runs
# the band pool.
def checked(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  valid: Bool,planes: Maybe<&2,U32>,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32,+model: Model.Model<Element>,
  +shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  +valid_exact: {Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) == valid : Bool},
  +planes_exact: {Layout.planes(shape) == planes : Maybe<&2,U32>},
  +nonempty: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Window.checked(~Element,~operation,name,valid,planes,pad,Window.geometry(shape,kernel,stride,padding),
    Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers},shape,fill,workers)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,Window.geometry(shape,kernel,stride,padding),pad,
    Window.checked(~Element,~operation,name,valid,planes,pad,Window.geometry(shape,kernel,stride,padding),
      Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers},shape,fill,workers)):
  match valid planes:
    case True{} Some{+planes}:
      %Equal.sym(Bool,Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))),False{},nonempty) :
        Contract(~Element,~Logical,~original,~operation,logical,Window.geometry(shape,kernel,stride,padding),pad,
          Window.nonempty(~Element,~operation,name,_,pad,Window.geometry(shape,kernel,stride,padding),planes,
            Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers},shape,fill,workers))
      offered(~Element,~Logical,~original,~operation,name,logical,images,channels,height,width,model,shape,pad,fill,workers,kernel,stride,padding,planes,
        planes_exact,valid_exact,nonempty,contract,
        %nonempty :
          {Contract.tensor_ready(~Element,Window.nonempty(~Element,~operation,name,_,pad,Window.geometry(shape,kernel,stride,padding),planes,
            Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers},shape,fill,workers)) == True{} : Bool}
        accepted)
    case True{} None{}:
      Empty.absurd(Contract(~Element,~Logical,~original,~operation,logical,Window.geometry(shape,kernel,stride,padding),pad,
        Window.checked(~Element,~operation,name,True{},None{},pad,Window.geometry(shape,kernel,stride,padding),
          Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers},shape,fill,workers)),
        WordEncoding.false_true(accepted))
    case False{} planes:
      Empty.absurd(Contract(~Element,~Logical,~original,~operation,logical,Window.geometry(shape,kernel,stride,padding),pad,
        Window.checked(~Element,~operation,name,False{},planes,pad,Window.geometry(shape,kernel,stride,padding),
          Tensors.Tensor{Model.pack(~Element,model),Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers},shape,fill,workers)),
        WordEncoding.false_true(accepted))

def observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,-bands: Bands.Bands(~Element),+input: Bands.Geometry,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  observation: Model.Observation<Element,bands>,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,input,shape,fill,Tensors.Usable{},workers}),
  +nonempty: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  accepted: {Contract.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,Window.geometry(shape,kernel,stride,padding),pad,
    Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  Bands.Geometry{+images,+channels,+height,+width} = input
  Model.Observation{+model,+equation} = observation
  %equation : Contract(~Element,~Logical,~original,~operation,logical,Window.geometry(shape,kernel,stride,padding),pad,
    Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{_,Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers},kernel,stride,padding))
  checked(~Element,~Logical,~original,~operation,name,Window.valid(shape,Window.geometry(shape,kernel,stride,padding)),Layout.planes(shape),logical,images,channels,height,width,model,
    shape,pad,fill,workers,kernel,stride,padding,{==},{==},nonempty,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) :
      Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{_,Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers}) contract,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) :
      {Contract.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{_,Bands.Geometry{images,channels,height,width},shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}
      accepted)

# The public pool of a Ready tensor with Normal state and complete values,
# whatever its layout. When the call returns Ready with a nonempty output,
# the result is Normal and holds the ordered pool of the logical input at
# every index.
def pool2d(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,bands: Bands.Bands(~Element),+input: Bands.Geometry,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,input,shape,fill,Tensors.Usable{},workers}),
  +nonempty: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  accepted: {Contract.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,Window.geometry(shape,kernel,stride,padding),pad,
    Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  observed(~Element,~Logical,~original,~operation,name,logical,bands,input,shape,pad,fill,workers,kernel,stride,padding,Model.observe(~Element,bands),contract,nonempty,accepted)
