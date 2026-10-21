# Locate the phase and chunk of an arbitrary in-band input coordinate.
import Base
import ./input_gradient_tiled_reduction_indices.bend as Indices
import ./row_major_coordinates.bend as GridCoordinates
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ../machine_limits.bend as Machine
import ./input_gradient_tiled_band_geometry.bend as Samples
import ./natural_addition.bend as Addition
import ./input_gradient_phase_geometry.bend as PhaseGeometry
import ../kernel_input_gradient.bend as Kernel
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_point.bend as Point
import ./input_gradient_tiled_chunks.bend as Partition
import ./input_gradient_tiled_bands.bend as Bands
import ./input_gradient_tiled_capacity.bend as Capacity
import ./input_gradient_phase_packing.bend as Packing
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_ceiling.bend as Ceiling
import ./nat_division_order.bend as Division
import ./storage_shape.bend as Shape

type Problem is Data:
  Problem{query: Chunks.Query,point: Domain.Point,
    inside: Domain.Inside<Chunks.geometry(query),GeometryChunks.grid_batch(Chunks.grid(query)),point>,
    lower: {Nat.is_le(U32.to_nat(Chunks.first(query)),Domain.row(point)) == True{} : Bool},
    upper: {Nat.is_lt(Domain.row(point),Nat.add(U32.to_nat(Chunks.first(query)),U32.to_nat(GeometryChunks.grid_count(Chunks.grid(query))))) == True{} : Bool},
    contained: {Nat.is_le(Nat.add(U32.to_nat(Chunks.first(query)),U32.to_nat(GeometryChunks.grid_count(Chunks.grid(query)))),Domain.height(Chunks.geometry(query))) == True{} : Bool}}
def query(problem: Problem) -> Chunks.Query:
  Problem{query,point,inside,lower,upper,contained} = problem
  query
def point(problem: Problem) -> Domain.Point:
  Problem{query,point,inside,lower,upper,contained} = problem
  point
def point_inside(problem: Problem) -> Domain.Inside<Chunks.geometry(query(problem)),GeometryChunks.grid_batch(Chunks.grid(query(problem))),point(problem)>:
  Problem{query,point,inside,lower,upper,contained} = problem
  inside
def row_lower(problem: Problem) -> {Nat.is_le(U32.to_nat(Chunks.first(query(problem))),Domain.row(point(problem))) == True{} : Bool}:
  Problem{query,point,inside,lower,upper,contained} = problem
  lower
def row_upper(problem: Problem) -> {Nat.is_lt(Domain.row(point(problem)),Nat.add(U32.to_nat(Chunks.first(query(problem))),U32.to_nat(GeometryChunks.grid_count(Chunks.grid(query(problem)))))) == True{} : Bool}:
  Problem{query,point,inside,lower,upper,contained} = problem
  upper
def leaf_contained(problem: Problem) -> {Nat.is_le(Nat.add(U32.to_nat(Chunks.first(query(problem))),U32.to_nat(GeometryChunks.grid_count(Chunks.grid(query(problem))))),Domain.height(Chunks.geometry(query(problem)))) == True{} : Bool}:
  Problem{query,point,inside,lower,upper,contained} = problem
  contained
def axes(+problem: Problem) -> Point.Axes<Chunks.geometry(query(problem)),point(problem)>:
  Problem{+query,+point,+inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{call,e,count,range,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Point.axes(call,point,inside)
def phase_y(+problem: Problem) -> Nat:
  U32.to_nat(Point.first(Domain.row(point(problem)),Domain.stride_word(Chunks.geometry(query(problem)))))
def phase_x(+problem: Problem) -> Nat:
  U32.to_nat(Point.first(Domain.column(point(problem)),Domain.stride_word(Chunks.geometry(query(problem)))))
def y_inside_from(+problem: Problem,evidence: Point.Axes<Chunks.geometry(query(problem)),point(problem)>) ->
  {Nat.is_lt(phase_y(problem),U32.to_nat(U32.min(Domain.stride_word(Chunks.geometry(query(problem))),Domain.height_word(Chunks.geometry(query(problem)))))) == True{} : Bool}:
  Point.Axes{fy,fx,fyl,fxl,re,ce,ri,ci,rp,cp} = evidence
  fyl
def x_inside_from(+problem: Problem,evidence: Point.Axes<Chunks.geometry(query(problem)),point(problem)>) ->
  {Nat.is_lt(phase_x(problem),U32.to_nat(U32.min(Domain.stride_word(Chunks.geometry(query(problem))),Domain.width_word(Chunks.geometry(query(problem)))))) == True{} : Bool}:
  Point.Axes{fy,fx,fyl,fxl,re,ce,ri,ci,rp,cp} = evidence
  fxl
def phase(+problem: Problem) -> Jobs.Phase:
  Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{call,e,count,range,rp,rr} = grid
  Jobs.Phase{call,e,phase_y(problem),phase_x(problem),y_inside_from(problem,axes(problem)),x_inside_from(problem,axes(problem))}
def row_position(+problem: Problem) -> Nat:
  U32.to_nat(Point.local(Domain.row(point(problem)),Domain.stride_word(Chunks.geometry(query(problem)))))
def column_position(+problem: Problem) -> Nat:
  U32.to_nat(Point.local(Domain.column(point(problem)),Domain.stride_word(Chunks.geometry(query(problem)))))
def relative_row(+problem: Problem) -> Nat:
  Nat.sub(Domain.row(point(problem)),U32.to_nat(Chunks.first(query(problem))))
def column_group(+problem: Problem) -> Nat:
  Nat.div(column_position(problem),8n)
def located_query(+problem: Problem) -> Chunks.Query:
  Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{grid,first,target,plane,source,row,group,wb,ob} = query
  Chunks.Query{grid,first,target,plane,source,relative_row(problem),column_group(problem),wb,ob}
def located(+problem: Problem) -> Chunks.Selection:
  Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Selection{located_query(problem),phase_y(problem),phase_x(problem),y_inside_from(problem,axes(problem)),x_inside_from(problem,axes(problem))}
def relative_row_inside(+problem: Problem) -> {Nat.is_lt(relative_row(problem),U32.to_nat(GeometryChunks.grid_count(Chunks.grid(query(problem))))) == True{} : Bool}:
  Problem{+query,+point,inside,+lower,+upper,contained} = problem
  Order.sub_lt(Domain.row(point),U32.to_nat(Chunks.first(query)),U32.to_nat(GeometryChunks.grid_count(Chunks.grid(query))),lower,upper)
def phase_matches(+problem: Problem) -> {Jobs.context(phase(problem)) == Point.phase(Chunks.geometry(query(problem)),point(problem)) : Kernel.Phase}:
  Problem{+query,+point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +y = Point.first(Domain.row(point),Domain.stride_word(g))
  +x = Point.first(Domain.column(point),Domain.stride_word(g))
  Equal.trans(Kernel.Phase,Jobs.context(phase(problem)),Kernel.phase(g,y,U32.from_nat(U32.to_nat(x))),Point.phase(g,point),Equal.cong(U32,Kernel.Phase,first => Kernel.phase(g,first,U32.from_nat(U32.to_nat(x))),U32.from_nat(U32.to_nat(y)),y,Arithmetic.native_roundtrip(y)),Equal.cong(U32,Kernel.Phase,first => Kernel.phase(g,y,first),U32.from_nat(U32.to_nat(x)),x,Arithmetic.native_roundtrip(x)))
def row_axis_inside_from(+problem: Problem,evidence: Point.Axes<Chunks.geometry(query(problem)),point(problem)>) ->
  {Nat.is_lt(row_position(problem),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_y(Point.phase(Chunks.geometry(query(problem)),point(problem)))))) == True{} : Bool}:
  Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Point.Axes{fy,fx,fyl,fxl,re,ce,ri,ci,rp0,cp} = evidence
  ri
def column_axis_inside_from(+problem: Problem,evidence: Point.Axes<Chunks.geometry(query(problem)),point(problem)>) ->
  {Nat.is_lt(column_position(problem),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_x(Point.phase(Chunks.geometry(query(problem)),point(problem)))))) == True{} : Bool}:
  Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Point.Axes{fy,fx,fyl,fxl,re,ce,ri,ci,rp0,cp} = evidence
  ci
def row_axis_inside(+problem: Problem) -> {Nat.is_lt(row_position(problem),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_y(Jobs.context(phase(problem)))))) == True{} : Bool}:
  %Equal.sym(Kernel.Phase,Jobs.context(phase(problem)),Point.phase(Chunks.geometry(query(problem)),point(problem)),phase_matches(problem)) : {Nat.is_lt(row_position(problem),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_y(_)))) == True{} : Bool}
  row_axis_inside_from(problem,axes(problem))
def column_axis_inside(+problem: Problem) -> {Nat.is_lt(column_position(problem),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_x(Jobs.context(phase(problem)))))) == True{} : Bool}:
  %Equal.sym(Kernel.Phase,Jobs.context(phase(problem)),Point.phase(Chunks.geometry(query(problem)),point(problem)),phase_matches(problem)) : {Nat.is_lt(column_position(problem),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_x(_)))) == True{} : Bool}
  column_axis_inside_from(problem,axes(problem))
def chunk_inside(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) -> Partition.Inside<GeometryChunks.query(Chunks.grid(query(problem)),relative_row(problem),column_group(problem))>:
  Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +selected_phase = phase(problem)
  +active = Jobs.active_phase(selected_phase,enabled)
  GeometryChunks.phase_inside(grid,phase_x(problem),x_inside_from(problem,axes(problem)),Jobs.column_active(selected_phase,active),relative_row(problem),relative_row_inside(problem),column_position(problem),column_axis_inside(problem))
def stride_positive(+problem: Problem) -> {Nat.is_lt(0n,Domain.stride(Chunks.geometry(query(problem)))) == True{} : Bool}:
  Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{call,e,count,range,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Domain.stride_positive(call)
def row_pixel_from(+problem: Problem,evidence: Point.Axes<Chunks.geometry(query(problem)),point(problem)>) -> {Nat.add(phase_y(problem),Nat.mul(row_position(problem),Domain.stride(Chunks.geometry(query(problem))))) == Domain.row(point(problem)) : Nat}:
  Point.Axes{+fy,fx,fyl,fxl,+re,ce,ri,ci,rp,cp} = evidence
  +position = Domain.row(point(problem))
  +stride = Domain.stride(Chunks.geometry(query(problem)))
  %Equal.sym(Nat,phase_y(problem),Nat.mod(position,stride),fy) : {Nat.add(_,Nat.mul(row_position(problem),stride)) == position : Nat}
  %Equal.sym(Nat,row_position(problem),Nat.div(position,stride),re) : {Nat.add(Nat.mod(position,stride),Nat.mul(_,stride)) == position : Nat}
  Equal.trans(Nat,Nat.add(Nat.mod(position,stride),Nat.mul(Nat.div(position,stride),stride)),Nat.add(Nat.mul(Nat.div(position,stride),stride),Nat.mod(position,stride)),position,Order.commutative(Nat.mod(position,stride),Nat.mul(Nat.div(position,stride),stride)),Equal.sym(Nat,position,Nat.add(Nat.mul(Nat.div(position,stride),stride),Nat.mod(position,stride)),Division.quotient_exact(position,stride,stride_positive(problem))))
def column_pixel_from(+problem: Problem,evidence: Point.Axes<Chunks.geometry(query(problem)),point(problem)>) -> {Nat.add(phase_x(problem),Nat.mul(column_position(problem),Domain.stride(Chunks.geometry(query(problem))))) == Domain.column(point(problem)) : Nat}:
  Point.Axes{fy,+fx,fyl,fxl,re,+ce,ri,ci,rp,cp} = evidence
  +position = Domain.column(point(problem))
  +stride = Domain.stride(Chunks.geometry(query(problem)))
  %Equal.sym(Nat,phase_x(problem),Nat.mod(position,stride),fx) : {Nat.add(_,Nat.mul(column_position(problem),stride)) == position : Nat}
  %Equal.sym(Nat,column_position(problem),Nat.div(position,stride),ce) : {Nat.add(Nat.mod(position,stride),Nat.mul(_,stride)) == position : Nat}
  Equal.trans(Nat,Nat.add(Nat.mod(position,stride),Nat.mul(Nat.div(position,stride),stride)),Nat.add(Nat.mul(Nat.div(position,stride),stride),Nat.mod(position,stride)),position,Order.commutative(Nat.mod(position,stride),Nat.mul(Nat.div(position,stride),stride)),Equal.sym(Nat,position,Nat.add(Nat.mul(Nat.div(position,stride),stride),Nat.mod(position,stride)),Division.quotient_exact(position,stride,stride_positive(problem))))
def partition_rows_positive(+query: Partition.Query,inside: Partition.Inside<query>) -> {Nat.is_lt(0n,Partition.chunk_rows(query)) == True{} : Bool}:
  Partition.Inside{rp,gp,ri,gi} = inside
  rp
def chunk_rows_positive(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) -> {Nat.is_lt(0n,Partition.chunk_rows(GeometryChunks.query(Chunks.grid(query(problem)),relative_row(problem),column_group(problem)))) == True{} : Bool}:
  partition_rows_positive(GeometryChunks.query(Chunks.grid(query(problem)),relative_row(problem),column_group(problem)),chunk_inside(problem,enabled))
def row_decomposition(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) ->
  {Nat.add(U32.to_nat(GeometryChunks.selected_start(Chunks.grid(query(problem)),Chunks.first(query(problem)),relative_row(problem),column_group(problem))),Nat.mod(relative_row(problem),Partition.chunk_rows(GeometryChunks.query(Chunks.grid(query(problem)),relative_row(problem),column_group(problem))))) == Domain.row(point(problem)) : Nat}:
  Problem{+query,+point,inside,+lower,upper,+contained} = problem
  Chunks.Query{+grid,+first,target,plane,source,row,group,wb,ob} = query
  +relative = relative_row(problem)
  +group_index = column_group(problem)
  +pure_query = GeometryChunks.query(grid,relative,group_index)
  +width = Partition.chunk_rows(pure_query)
  +offset = GeometryChunks.row_offset_natural(grid,relative,group_index)
  +local = Nat.mod(relative,width)
  +start = GeometryChunks.selected_start(grid,first,relative,group_index)
  %Equal.sym(Nat,U32.to_nat(start),Nat.add(U32.to_nat(first),offset),GeometryChunks.selected_start_exact(grid,first,relative,group_index,chunk_inside(problem,enabled),contained)) : {Nat.add(_,local) == Domain.row(point) : Nat}
  Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(first),offset),local),Nat.add(U32.to_nat(first),Nat.add(offset,local)),Domain.row(point),Addition.associative(U32.to_nat(first),offset,local),
    Equal.trans(Nat,Nat.add(U32.to_nat(first),Nat.add(offset,local)),Nat.add(U32.to_nat(first),relative),Domain.row(point),Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(first),value),Nat.add(offset,local),relative,
      Equal.sym(Nat,relative,Nat.add(offset,local),Division.quotient_exact(relative,width,chunk_rows_positive(problem,enabled)))),Order.restore_shift(U32.to_nat(first),Domain.row(point),lower)))
def selected_row_lower(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) ->
  {Nat.is_le(U32.to_nat(GeometryChunks.selected_start(Chunks.grid(query(problem)),Chunks.first(query(problem)),relative_row(problem),column_group(problem))),Domain.row(point(problem))) == True{} : Bool}:
  +grid = Chunks.grid(query(problem))
  +start = U32.to_nat(GeometryChunks.selected_start(grid,Chunks.first(query(problem)),relative_row(problem),column_group(problem)))
  +local = Nat.mod(relative_row(problem),Partition.chunk_rows(GeometryChunks.query(grid,relative_row(problem),column_group(problem))))
  %row_decomposition(problem,enabled) : {Nat.is_le(start,_) == True{} : Bool}
  Order.le_add(start,local)
def selected_row_upper(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) ->
  {Nat.is_lt(Domain.row(point(problem)),Nat.add(U32.to_nat(GeometryChunks.selected_start(Chunks.grid(query(problem)),Chunks.first(query(problem)),relative_row(problem),column_group(problem))),U32.to_nat(GeometryChunks.selected_rows(Chunks.grid(query(problem)),relative_row(problem),column_group(problem))))) == True{} : Bool}:
  +grid = Chunks.grid(query(problem))
  +row = relative_row(problem)
  +group = column_group(problem)
  +start = U32.to_nat(GeometryChunks.selected_start(grid,Chunks.first(query(problem)),row,group))
  +local = Nat.mod(row,Partition.chunk_rows(GeometryChunks.query(grid,row,group)))
  +rows = U32.to_nat(GeometryChunks.selected_rows(grid,row,group))
  %row_decomposition(problem,enabled) : {Nat.is_lt(_,Nat.add(start,rows)) == True{} : Bool}
  Order.offset_lt(start,local,rows,GeometryChunks.selected_row_inside(grid,row,group,chunk_inside(problem,enabled)))
def work(+problem: Problem) -> Jobs.Work:
  Chunks.work(located(problem))
def contained(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Jobs.chunk_row(Jobs.chunk(work(problem)))),U32.to_nat(Jobs.chunk_rows(Jobs.chunk(work(problem))))),Domain.height(Jobs.geometry(Jobs.phase(work(problem))))) == True{} : Bool}:
  Problem{+query,point,inside,lower,upper,+contained} = problem
  Chunks.Query{+grid,+first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  GeometryChunks.selected_band_contains(grid,first,relative_row(problem),column_group(problem),chunk_inside(problem,enabled),contained)
def word_stride_positive(+problem: Problem) -> {Nat.is_lt(0n,U32.to_nat(Domain.stride_word(Chunks.geometry(query(problem))))) == True{} : Bool}:
  Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Domain.stride_positive(call)
def band_pixel(+problem: Problem,+positive: {Nat.is_lt(0n,U32.to_nat(Domain.stride_word(Chunks.geometry(query(problem))))) == True{} : Bool}) ->
  {Bands.position(Samples.model(Jobs.row_axis(Jobs.phase(work(problem))),Domain.stride_word(Chunks.geometry(query(problem))),Jobs.chunk_row(Jobs.chunk(work(problem))),(Jobs.chunk_row(Jobs.chunk(work(problem))) + Jobs.chunk_rows(Jobs.chunk(work(problem))) : U32),positive),row_position(problem)) == Domain.row(point(problem)) : Nat}:
  Problem{+query,+point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,+h,w,+k,+s,+p,gh,gw} = g
  +first_word = Point.first(Domain.row(point),s)
  %Equal.sym(U32,U32.from_nat(phase_y(problem)),first_word,Arithmetic.native_roundtrip(first_word)) : {Nat.add(U32.to_nat(_),Nat.mul(row_position(problem),U32.to_nat(s))) == Domain.row(point) : Nat}
  row_pixel_from(problem,axes(problem))
def row_indices(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat(Jobs.start(work(problem))),row_position(problem)) == True{} : Bool},{Nat.is_lt(row_position(problem),U32.to_nat(Jobs.end(work(problem)))) == True{} : Bool}>:
  Problem{+query,location,inside,lower,upper,leaf_contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,+s,p,gh,gw} = g
  +height_room = Domain.height_room(call)
  +selected_work = work(problem)
  +first = Jobs.chunk_row(Jobs.chunk(selected_work))
  +rows = Jobs.chunk_rows(Jobs.chunk(selected_work))
  +last = (first + rows : U32)
  +room = contained(problem,enabled)
  +first_room = Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(Machine.dimension(l)),Order.le_add(U32.to_nat(first),U32.to_nat(rows)),Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Domain.height(g),U32.to_nat(Machine.dimension(l)),room,height_room))
  +last_room = {%Equal.sym(Nat,U32.to_nat(last),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Jobs.chunk_end_exact(selected_work,room)) : {Nat.is_le(_,U32.to_nat(Machine.dimension(l))) == True{} : Bool} Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Domain.height(g),U32.to_nat(Machine.dimension(l)),room,height_room) : {Nat.is_le(U32.to_nat(last),U32.to_nat(Machine.dimension(l))) == True{} : Bool}}
  +positive = word_stride_positive(problem)
  +pixel = Bands.position(Samples.model(Jobs.row_axis(Jobs.phase(selected_work)),s,first,last,positive),row_position(problem))
  Samples.member_indices(Jobs.row_axis(Jobs.phase(selected_work)),s,first,last,l,first_room,last_room,Domain.stride_room(call),positive,row_position(problem),row_axis_inside(problem),
    %Equal.sym(Nat,pixel,Domain.row(point(problem)),band_pixel(problem,positive)) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} selected_row_lower(problem,enabled),
    %Equal.sym(Nat,pixel,Domain.row(point(problem)),band_pixel(problem,positive)) : {Nat.is_lt(_,U32.to_nat(last)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(last),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Jobs.chunk_end_exact(selected_work,room)) : {Nat.is_lt(Domain.row(point(problem)),_) == True{} : Bool} selected_row_upper(problem,enabled))
def line(+problem: Problem) -> Nat:
  Nat.sub(row_position(problem),U32.to_nat(Jobs.start(work(problem))))
def line_inside_from(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool},
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(Jobs.start(work(problem))),row_position(problem)) == True{} : Bool},{Nat.is_lt(row_position(problem),U32.to_nat(Jobs.end(work(problem)))) == True{} : Bool}>) ->
  {Nat.is_lt(line(problem),U32.to_nat(Jobs.lines(work(problem)))) == True{} : Bool}:
  Shape.Both{+lo,+hi} = bounds
  +first = Jobs.start(work(problem))
  +last = Jobs.end(work(problem))
  +position = row_position(problem)
  +ordered = Order.le_lt_trans(U32.to_nat(first),position,U32.to_nat(last),lo,hi)
  +selected = {%Equal.sym(Bool,U32.is_lt(first,last),Nat.is_lt(U32.to_nat(first),U32.to_nat(last)),Comparison.lt(first,last)) : {_ == True{} : Bool} ordered : {U32.is_lt(first,last) == True{} : Bool}}
  +difference = FullRange.subtract(last,first,last,Bounds.strict_le(U32.to_nat(first),U32.to_nat(last),ordered),Order.sub_le(U32.to_nat(last),U32.to_nat(first)))
  %Equal.sym(Bool,U32.is_lt(first,last),True{},selected) : {Nat.is_lt(line(problem),U32.to_nat(Bool.pick(U32,_,(last - first : U32),0))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((last - first : U32)),Nat.sub(U32.to_nat(last),U32.to_nat(first)),difference) : {Nat.is_lt(line(problem),_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(Nat.sub(position,U32.to_nat(first)),Nat.sub(U32.to_nat(last),U32.to_nat(first))),Nat.is_lt(Nat.add(U32.to_nat(first),Nat.sub(position,U32.to_nat(first))),U32.to_nat(last)),Order.subtraction_lt(U32.to_nat(last),U32.to_nat(first),Nat.sub(position,U32.to_nat(first)))) : {_ == True{} : Bool}
  %Equal.sym(Nat,Nat.add(U32.to_nat(first),Nat.sub(position,U32.to_nat(first))),position,Order.restore_shift(U32.to_nat(first),position,lo)) : {Nat.is_lt(_,U32.to_nat(last)) == True{} : Bool}
  hi
def line_inside(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) -> {Nat.is_lt(line(problem),U32.to_nat(Jobs.lines(work(problem)))) == True{} : Bool}:
  line_inside_from(problem,enabled,row_indices(problem,enabled))
def group(+problem: Problem) -> Nat:
  Nat.sub(column_group(problem),U32.to_nat(Jobs.chunk_group(Jobs.chunk(work(problem)))))
def group_offset_exact(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) ->
  {U32.to_nat(Jobs.chunk_group(Jobs.chunk(work(problem)))) == GeometryChunks.group_offset_natural(Chunks.grid(query(problem)),relative_row(problem),column_group(problem)) : Nat}:
  Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  GeometryChunks.group_offset_exact(grid,relative_row(problem),column_group(problem),chunk_inside(problem,enabled))
def partition_groups_positive(+query: Partition.Query,inside: Partition.Inside<query>) -> {Nat.is_lt(0n,Partition.chunk_groups(query)) == True{} : Bool}:
  Partition.Inside{rp,gp,ri,gi} = inside
  gp
def chunk_groups_positive(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) -> {Nat.is_lt(0n,Partition.chunk_groups(GeometryChunks.query(Chunks.grid(query(problem)),relative_row(problem),column_group(problem)))) == True{} : Bool}:
  partition_groups_positive(GeometryChunks.query(Chunks.grid(query(problem)),relative_row(problem),column_group(problem)),chunk_inside(problem,enabled))
def group_lower(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) -> {Nat.is_le(U32.to_nat(Jobs.chunk_group(Jobs.chunk(work(problem)))),column_group(problem)) == True{} : Bool}:
  +grid = Chunks.grid(query(problem))
  +pure = GeometryChunks.query(grid,relative_row(problem),column_group(problem))
  %Equal.sym(Nat,U32.to_nat(Jobs.chunk_group(Jobs.chunk(work(problem)))),GeometryChunks.group_offset_natural(grid,relative_row(problem),column_group(problem)),group_offset_exact(problem,enabled)) : {Nat.is_le(_,column_group(problem)) == True{} : Bool}
  Division.product_below(column_group(problem),Partition.chunk_groups(pure),chunk_groups_positive(problem,enabled))
def group_local_exact(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) ->
  {group(problem) == Nat.mod(column_group(problem),Partition.chunk_groups(GeometryChunks.query(Chunks.grid(query(problem)),relative_row(problem),column_group(problem)))) : Nat}:
  +grid = Chunks.grid(query(problem))
  +position = column_group(problem)
  +width = Partition.chunk_groups(GeometryChunks.query(grid,relative_row(problem),position))
  +offset = GeometryChunks.group_offset_natural(grid,relative_row(problem),position)
  +local = Nat.mod(position,width)
  %Equal.sym(Nat,U32.to_nat(Jobs.chunk_group(Jobs.chunk(work(problem)))),offset,group_offset_exact(problem,enabled)) : {Nat.sub(position,_) == local : Nat}
  %Equal.sym(Nat,position,Nat.add(offset,local),Division.quotient_exact(position,width,chunk_groups_positive(problem,enabled))) : {Nat.sub(_,offset) == local : Nat}
  Order.subtract_shift(offset,local)
def phase_group_inside(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) -> {Nat.is_lt(column_group(problem),U32.to_nat(Jobs.native_groups(Jobs.phase(work(problem))))) == True{} : Bool}:
  Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +selected_phase = phase(problem)
  +active = Jobs.active_phase(selected_phase,enabled)
  +count = U32.to_nat(Jobs.column_count(selected_phase))
  %Equal.sym(Nat,U32.to_nat(Jobs.native_groups(selected_phase)),Ceiling.ceiling(count,8n),Jobs.phase_groups_exact(Jobs.columns(selected_phase),phase_x(problem),x_inside_from(problem,axes(problem)),Jobs.column_active(selected_phase,active))) : {Nat.is_lt(column_group(problem),_) == True{} : Bool}
  Partition.chunks_inside(column_position(problem),count,8n,{==},column_axis_inside(problem))
def group_inside(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) -> {Nat.is_lt(group(problem),U32.to_nat(Jobs.groups(work(problem)))) == True{} : Bool}:
  Problem{+q0,location,inside,lower,upper,leaf_contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group_index,wb,ob} = q0
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +selected_work = work(problem)
  +width = Capacity.layout_chunk_groups(Kernel.tiling(Chunks.geometry(query(problem))))
  Jobs.group_span_selected(Jobs.chunk_group(Jobs.chunk(selected_work)),Jobs.native_groups(Jobs.phase(selected_work)),width,column_group(problem),group_lower(problem,enabled),phase_group_inside(problem,enabled),
    %Equal.sym(Nat,group(problem),Nat.mod(column_group(problem),Partition.chunk_groups(GeometryChunks.query(Chunks.grid(query(problem)),relative_row(problem),column_group(problem)))),group_local_exact(problem,enabled)) : {Nat.is_lt(_,U32.to_nat(width)) == True{} : Bool}
    Division.remainder_small(column_group(problem),U32.to_nat(width),chunk_groups_positive(problem,enabled)))
def unit(+problem: Problem) -> Nat:
  Nat.div(Domain.channel(point(problem)),4n)
def row_in_unit(+problem: Problem) -> Nat:
  Nat.mod(Domain.channel(point(problem)),4n)
def lane(+problem: Problem) -> Nat:
  Nat.mod(column_position(problem),8n)
def unit_inside(+problem: Problem) -> {Nat.is_lt(unit(problem),U32.to_nat(Jobs.native_units(work(problem)))) == True{} : Bool}:
  Problem{+query,+point,+inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,+e,count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Domain.Inside{ii,+ci,ri,xi} = inside
  %Equal.sym(Nat,U32.to_nat(Jobs.units(g)),Ceiling.ceiling(Domain.channels(g),4n),Jobs.units_exact(call,e)) : {Nat.is_lt(unit(problem),_) == True{} : Bool}
  Partition.chunks_inside(Domain.channel(point),Domain.channels(g),4n,{==},ci)
def job_inside(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool}) -> Jobs.Inside<work(problem),line(problem),group(problem),unit(problem)>:
  Jobs.Inside{line_inside(problem,enabled),group_inside(problem,enabled),unit_inside(problem)}
def row_small(+problem: Problem) -> {Nat.is_lt(row_in_unit(problem),4n) == True{} : Bool}:
  Division.remainder_small(Domain.channel(point(problem)),4n,{==})
def lane_small(+problem: Problem) -> {Nat.is_lt(lane(problem),8n) == True{} : Bool}:
  Division.remainder_small(column_position(problem),8n,{==})
def reduction(+problem: Problem) -> Chunks.Reduction:
  +coordinates = Indices.values(phase(problem),0n)
  Chunks.Reduction{located(problem),line(problem),group(problem),unit(problem),GridCoordinates.channel(coordinates),GridCoordinates.row(coordinates),GridCoordinates.column(coordinates)}
def valid_from(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(query(problem))))) == True{} : Bool},
  coordinates: GridCoordinates.Bounded<Domain.outputs(Jobs.geometry(phase(problem))),Indices.rows(phase(problem)),Indices.columns(phase(problem)),Indices.values(phase(problem),0n)>) -> Chunks.Valid<reduction(problem)>:
  Problem{+query,point,inside,lower,upper,+leaf_contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  GridCoordinates.Bounded{oi,ri,ci} = coordinates
  Chunks.Valid{chunk_inside(problem,enabled),leaf_contained,fits,enabled,job_inside(problem,enabled),oi,ri,ci}
def valid(+problem: Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(query(problem))))) == True{} : Bool}) -> Chunks.Valid<reduction(problem)>:
  valid_from(problem,enabled,fits,Indices.bounded(phase(problem),enabled,0n,PhaseBounds.phase_inner_positive(Jobs.context(phase(problem)),enabled)))

# The phase fold supplies packing cursors without changing the located producer.
def with_bases(problem: Problem,weight_base: U32,offset_base: U32) -> Problem:
  Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{grid,first,target,plane,source,row,group,wb,ob} = query
  Problem{Chunks.Query{grid,first,target,plane,source,row,group,weight_base,offset_base},point,inside,lower,upper,contained}
