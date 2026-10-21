# A residue with no congruent kernel position has the exact positive-zero specification.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/proofs/input_gradient_tiled_producer.bend as Producer
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_phase_point.bend as Point
import ../../lib/proofs/input_gradient_phase_bounds.bend as PhaseBounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ./input_gradient_sources.bend as Sources
import ./input_gradient_order.bend as Order
import ../input_gradient_specification.bend as Specification

def flat(+problem: Producer.Problem,+image: U32,gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>) -> Order.Flat:
  +point = Producer.point(problem)
  Order.Flat{Sources.problem(Chunks.geometry(Producer.query(problem)),Sources.models_reader(gradient,weights),U32.to_nat(image)),Domain.channel(point),Domain.row(point),Domain.column(point)}
def native_flat(+problem: Producer.Problem,+image: U32,gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>) -> Order.Flat:
  +point = Producer.point(problem)
  +g = Chunks.geometry(Producer.query(problem))
  +s = Domain.stride_word(g)
  Sources.phase_flat(g,Point.first(Domain.row(point),s),Point.first(Domain.column(point),s),Point.local(Domain.row(point),s),Point.local(Domain.column(point),s),image,Domain.channel(point),gradient,weights)
def flat_matches(+problem: Producer.Problem,+image: U32,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>) -> {native_flat(problem,image,gradient,weights) == flat(problem,image,gradient,weights) : Order.Flat}:
  Producer.Problem{+query,+point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,+s,p,gh,gw} = g
  +source = Sources.problem(g,Sources.models_reader(gradient,weights),U32.to_nat(image))
  +channel = Domain.channel(point)
  +y = PhaseBounds.pixel(Point.first(Domain.row(point),s),s,Point.local(Domain.row(point),s))
  +x = PhaseBounds.pixel(Point.first(Domain.column(point),s),s,Point.local(Domain.column(point),s))
  Equal.trans(Order.Flat,native_flat(problem,image,gradient,weights),Order.Flat{source,channel,Domain.row(point),x},flat(problem,image,gradient,weights),
    Equal.cong(Nat,Order.Flat,row => Order.Flat{source,channel,row,x},y,Domain.row(point),Producer.row_pixel_from(problem,Producer.axes(problem))),
    Equal.cong(Nat,Order.Flat,column => Order.Flat{source,channel,Domain.row(point),column},x,Domain.column(point),Producer.column_pixel_from(problem,Producer.axes(problem))))
def count_matches(+problem: Producer.Problem,+image: U32,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>) ->
  {U32.to_nat(Kernel.phase_inner(Jobs.context(Producer.phase(problem)))) == Order.flat_count(flat(problem,image,gradient,weights)) : Nat}:
  Producer.Problem{+query,+point,+inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,+h,+w,k,+s,p,gh,gw} = g
  Domain.Inside{ii,ci,+yi,+xi} = inside
  +stride_positive = Domain.stride_positive(call)
  +height_room = Domain.height_room(call)
  +width_room = Domain.width_room(call)
  Equal.trans(Nat,U32.to_nat(Kernel.phase_inner(Jobs.context(Producer.phase(problem)))),Order.flat_count(native_flat(problem,image,gradient,weights)),Order.flat_count(flat(problem,image,gradient,weights)),
    %Equal.sym(Kernel.Phase,Jobs.context(Producer.phase(problem)),Point.phase(g,point),Producer.phase_matches(problem)) : {U32.to_nat(Kernel.phase_inner(_)) == Order.flat_count(native_flat(problem,image,gradient,weights)) : Nat}
    Sources.phase_spec_count(call,Point.first(Domain.row(point),s),Point.first(Domain.column(point),s),Point.local(Domain.row(point),s),Point.local(Domain.column(point),s),image,Domain.channel(point),gradient,weights,Point.first_inside(Domain.row(point),h,s,l,stride_positive,yi,height_room),Point.first_inside(Domain.column(point),w,s,l,stride_positive,xi,width_room)),
    Equal.cong(Order.Flat,Nat,context => Order.flat_count(context),native_flat(problem,image,gradient,weights),flat(problem,image,gradient,weights),flat_matches(problem,image,gradient,weights)))
def value(+problem: Producer.Problem,+image: U32,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,
  +empty: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == True{} : Bool}) ->
  {Specification.value(Order.flat_problem(flat(problem,image,gradient,weights)),Domain.channel(Producer.point(problem)),Domain.row(Producer.point(problem)),Domain.column(Producer.point(problem))) == (0.0 + 0.0 : F32) : F32}:
  Producer.Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +context = flat(problem,image,gradient,weights)
  +count = Kernel.phase_inner(Jobs.context(Producer.phase(problem)))
  +zero = Equal.trans(Nat,Order.flat_count(context),U32.to_nat(count),0n,Equal.sym(Nat,U32.to_nat(count),Order.flat_count(context),count_matches(problem,image,gradient,weights)),Equal.cong(U32,Nat,U32.to_nat,count,0,Arithmetic.equal_of_is_eq(count,0,empty)))
  Order.empty_value(context,Domain.stride_positive(call),zero)
