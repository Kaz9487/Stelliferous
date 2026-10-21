# Normalize the actual job store address into channel, row and column coordinates.
import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./input_gradient_tiled_store_geometry.bend as Channels
import ./input_gradient_tiled_store_columns.bend as Columns
import ./input_gradient_tiled_value_reads.bend as Input
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./input_gradient_phase_geometry.bend as PhaseGeometry
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ../kernel_input_gradient.bend as Kernel
import ../kernel_strided_store.bend as Stores
import ./kernel_accumulator.bend as Accumulator
import ./kernel_strided_store_writes.bend as Writes
import ./input_gradient_tiled_job_store.bend as Store
import ./input_gradient_tiled_jobs.bend as NativeJobs
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_job_reads.bend as Reads
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_tiled_product_reads.bend as Runtime
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_lookup.bend as Lookup
import ./input_gradient_tiled_output_layout.bend as Output
import ./storage_shape.bend as Shape
import ./bounded_u32_arithmetic.bend as Arithmetic

def row_product(+first: U32,+start: U32,+line: U32,+stride: U32,+low: U32,+width: U32) ->
  {((first + (start + line : U32) * stride - low : U32) * width : U32) == (((first + start * stride - low : U32) * width) + (line * (stride * width : U32)) : U32) : U32}:
  +base = (first + start * stride - low : U32)
  +delta = (line * stride : U32)
  %Equal.sym(U32,((start + line : U32) * stride : U32),((start * stride) + (line * stride) : U32),Arithmetic.multiply_additive(start,line,stride)) : {((first + _ - low : U32) * width : U32) == ((base * width) + (line * (stride * width : U32)) : U32) : U32}
  %Arithmetic.add_associative(first,(start * stride : U32),delta) : {((_ - low : U32) * width : U32) == ((base * width) + (line * (stride * width : U32)) : U32) : U32}
  %Arithmetic.add_after_subtract((first + start * stride : U32),low,delta) : {(_ * width : U32) == ((base * width) + (line * (stride * width : U32)) : U32) : U32}
  Equal.trans(U32,((base + delta : U32) * width : U32),((base * width) + (delta * width) : U32),((base * width) + (line * (stride * width : U32)) : U32),Arithmetic.multiply_additive(base,delta,width),Equal.cong(U32,U32,value => (base * width + value : U32),(delta * width : U32),(line * (stride * width : U32) : U32),Arithmetic.multiply_associative(line,stride,width)))
def column_product(+first: U32,+chunk: U32,+group: U32,+lane: U32,+stride: U32) ->
  {(first + ((chunk + group : U32) * 8 + lane : U32) * stride : U32) == (first + chunk * 8 * stride + group * 8 * stride + lane * stride : U32) : U32}:
  +base = ((chunk + group : U32) * 8 : U32)
  +left = (chunk * 8 * stride : U32)
  +right = (group * 8 * stride : U32)
  +last = (lane * stride : U32)
  +combined = {
    Equal.trans(U32,(base * stride : U32),((chunk + group : U32) * (8 * stride : U32) : U32),(left + right : U32),Arithmetic.multiply_associative((chunk + group : U32),8,stride),
      Equal.trans(U32,((chunk + group : U32) * (8 * stride : U32) : U32),((chunk * (8 * stride : U32)) + (group * (8 * stride : U32)) : U32),(left + right : U32),Arithmetic.multiply_additive(chunk,group,(8 * stride : U32)),
        Equal.trans(U32,((chunk * (8 * stride : U32)) + (group * (8 * stride : U32)) : U32),(left + (group * (8 * stride : U32)) : U32),(left + right : U32),
          Equal.cong(U32,U32,value => (value + group * (8 * stride : U32) : U32),(chunk * (8 * stride : U32) : U32),left,Equal.sym(U32,left,(chunk * (8 * stride : U32) : U32),Arithmetic.multiply_associative(chunk,8,stride))),
          Equal.cong(U32,U32,value => (left + value : U32),(group * (8 * stride : U32) : U32),right,Equal.sym(U32,right,(group * (8 * stride : U32) : U32),Arithmetic.multiply_associative(group,8,stride)))))) : {(base * stride : U32) == (left + right : U32) : U32}}
  %Equal.sym(U32,((base + lane : U32) * stride : U32),((base * stride) + last : U32),Arithmetic.multiply_additive(base,lane,stride)) : {(first + _ : U32) == (first + left + right + last : U32) : U32}
  %Equal.sym(U32,(base * stride : U32),(left + right : U32),combined) : {(first + (_ + last : U32) : U32) == (first + left + right + last : U32) : U32}
  Equal.sym(U32,(first + left + right + last : U32),(first + ((left + right : U32) + last : U32) : U32),
    Equal.trans(U32,(first + left + right + last : U32),((first + (left + right : U32) : U32) + last : U32),(first + ((left + right : U32) + last : U32) : U32),Equal.cong(U32,U32,value => (value + last : U32),(first + left + right : U32),(first + (left + right : U32) : U32),Arithmetic.add_associative(first,left,right)),Arithmetic.add_associative(first,(left + right : U32),last)))

# Coefficients of the real plan: target, initial row/column, chunk group,
# channel unit, row/group increments, accumulator row and accumulator lane.
def store_coefficients(+t: U32,+rb: U32,+xb: U32,+cb: U32,+ub: U32,+rd: U32,+gb: U32,+ur: U32,+ln: U32) ->
  {((((((((t + rb : U32) + xb : U32) + cb : U32) + ub : U32) + rd : U32) + gb : U32) + ur : U32) + ln : U32) == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}:
  +step_0 = Arithmetic.add_associative(((((((t + rb : U32) + xb : U32) + cb : U32) + ub : U32) + rd : U32) + gb : U32),ur,ln)
  +step_1 = Arithmetic.add_associative((((((t + rb : U32) + xb : U32) + cb : U32) + ub : U32) + rd : U32),gb,(ur + ln : U32))
  +step_2 = Arithmetic.add_associative(((((t + rb : U32) + xb : U32) + cb : U32) + ub : U32),rd,(gb + (ur + ln : U32) : U32))
  +step_3 = Arithmetic.add_associative((((t + rb : U32) + xb : U32) + cb : U32),ub,(rd + (gb + (ur + ln : U32) : U32) : U32))
  +step_4 = Arithmetic.add_associative(((t + rb : U32) + xb : U32),cb,(ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32))
  +step_5 = Arithmetic.add_associative((t + rb : U32),xb,(cb + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32))
  +step_6 = Arithmetic.add_associative(t,rb,(xb + (cb + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32))
  +step_7 = Equal.cong(U32,U32,value => (t + (rb + (xb + value : U32) : U32) : U32),(cb + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32),((cb + ub : U32) + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32),Equal.sym(U32,((cb + ub : U32) + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32),(cb + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32),Arithmetic.add_associative(cb,ub,(rd + (gb + (ur + ln : U32) : U32) : U32))))
  +step_8 = Equal.cong(U32,U32,value => (t + (rb + (xb + (value + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(cb + ub : U32),(ub + cb : U32),Arithmetic.add_commutative(cb,ub))
  +step_9 = Equal.cong(U32,U32,value => (t + (rb + (xb + value : U32) : U32) : U32),((ub + cb : U32) + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32),(ub + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32),Arithmetic.add_associative(ub,cb,(rd + (gb + (ur + ln : U32) : U32) : U32)))
  +step_10 = Equal.cong(U32,U32,value => (t + (rb + value : U32) : U32),(xb + (ub + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32),((xb + ub : U32) + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32),Equal.sym(U32,((xb + ub : U32) + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32),(xb + (ub + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32),Arithmetic.add_associative(xb,ub,(cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32))))
  +step_11 = Equal.cong(U32,U32,value => (t + (rb + (value + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(xb + ub : U32),(ub + xb : U32),Arithmetic.add_commutative(xb,ub))
  +step_12 = Equal.cong(U32,U32,value => (t + (rb + value : U32) : U32),((ub + xb : U32) + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32),(ub + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32),Arithmetic.add_associative(ub,xb,(cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32)))
  +step_13 = Equal.cong(U32,U32,value => (t + value : U32),(rb + (ub + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),((rb + ub : U32) + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32),Equal.sym(U32,((rb + ub : U32) + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32),(rb + (ub + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),Arithmetic.add_associative(rb,ub,(xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32))))
  +step_14 = Equal.cong(U32,U32,value => (t + (value + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(rb + ub : U32),(ub + rb : U32),Arithmetic.add_commutative(rb,ub))
  +step_15 = Equal.cong(U32,U32,value => (t + value : U32),((ub + rb : U32) + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32),(ub + (rb + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),Arithmetic.add_associative(ub,rb,(xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32)))
  +step_16 = Equal.cong(U32,U32,value => (t + (ub + (rb + (xb + (cb + (rd + value : U32) : U32) : U32) : U32) : U32) : U32),(gb + (ur + ln : U32) : U32),((gb + ur : U32) + ln : U32),Equal.sym(U32,((gb + ur : U32) + ln : U32),(gb + (ur + ln : U32) : U32),Arithmetic.add_associative(gb,ur,ln)))
  +step_17 = Equal.cong(U32,U32,value => (t + (ub + (rb + (xb + (cb + (rd + (value + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(gb + ur : U32),(ur + gb : U32),Arithmetic.add_commutative(gb,ur))
  +step_18 = Equal.cong(U32,U32,value => (t + (ub + (rb + (xb + (cb + (rd + value : U32) : U32) : U32) : U32) : U32) : U32),((ur + gb : U32) + ln : U32),(ur + (gb + ln : U32) : U32),Arithmetic.add_associative(ur,gb,ln))
  +step_19 = Equal.cong(U32,U32,value => (t + (ub + (rb + (xb + (cb + value : U32) : U32) : U32) : U32) : U32),(rd + (ur + (gb + ln : U32) : U32) : U32),((rd + ur : U32) + (gb + ln : U32) : U32),Equal.sym(U32,((rd + ur : U32) + (gb + ln : U32) : U32),(rd + (ur + (gb + ln : U32) : U32) : U32),Arithmetic.add_associative(rd,ur,(gb + ln : U32))))
  +step_20 = Equal.cong(U32,U32,value => (t + (ub + (rb + (xb + (cb + (value + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(rd + ur : U32),(ur + rd : U32),Arithmetic.add_commutative(rd,ur))
  +step_21 = Equal.cong(U32,U32,value => (t + (ub + (rb + (xb + (cb + value : U32) : U32) : U32) : U32) : U32),((ur + rd : U32) + (gb + ln : U32) : U32),(ur + (rd + (gb + ln : U32) : U32) : U32),Arithmetic.add_associative(ur,rd,(gb + ln : U32)))
  +step_22 = Equal.cong(U32,U32,value => (t + (ub + (rb + (xb + value : U32) : U32) : U32) : U32),(cb + (ur + (rd + (gb + ln : U32) : U32) : U32) : U32),((cb + ur : U32) + (rd + (gb + ln : U32) : U32) : U32),Equal.sym(U32,((cb + ur : U32) + (rd + (gb + ln : U32) : U32) : U32),(cb + (ur + (rd + (gb + ln : U32) : U32) : U32) : U32),Arithmetic.add_associative(cb,ur,(rd + (gb + ln : U32) : U32))))
  +step_23 = Equal.cong(U32,U32,value => (t + (ub + (rb + (xb + (value + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(cb + ur : U32),(ur + cb : U32),Arithmetic.add_commutative(cb,ur))
  +step_24 = Equal.cong(U32,U32,value => (t + (ub + (rb + (xb + value : U32) : U32) : U32) : U32),((ur + cb : U32) + (rd + (gb + ln : U32) : U32) : U32),(ur + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32),Arithmetic.add_associative(ur,cb,(rd + (gb + ln : U32) : U32)))
  +step_25 = Equal.cong(U32,U32,value => (t + (ub + (rb + value : U32) : U32) : U32),(xb + (ur + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32),((xb + ur : U32) + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32),Equal.sym(U32,((xb + ur : U32) + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32),(xb + (ur + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32),Arithmetic.add_associative(xb,ur,(cb + (rd + (gb + ln : U32) : U32) : U32))))
  +step_26 = Equal.cong(U32,U32,value => (t + (ub + (rb + (value + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(xb + ur : U32),(ur + xb : U32),Arithmetic.add_commutative(xb,ur))
  +step_27 = Equal.cong(U32,U32,value => (t + (ub + (rb + value : U32) : U32) : U32),((ur + xb : U32) + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32),(ur + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32),Arithmetic.add_associative(ur,xb,(cb + (rd + (gb + ln : U32) : U32) : U32)))
  +step_28 = Equal.cong(U32,U32,value => (t + (ub + value : U32) : U32),(rb + (ur + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32),((rb + ur : U32) + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32),Equal.sym(U32,((rb + ur : U32) + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32),(rb + (ur + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32),Arithmetic.add_associative(rb,ur,(xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32))))
  +step_29 = Equal.cong(U32,U32,value => (t + (ub + (value + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(rb + ur : U32),(ur + rb : U32),Arithmetic.add_commutative(rb,ur))
  +step_30 = Equal.cong(U32,U32,value => (t + (ub + value : U32) : U32),((ur + rb : U32) + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32),(ur + (rb + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32),Arithmetic.add_associative(ur,rb,(xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32)))
  +step_31 = Equal.cong(U32,U32,value => (t + (ub + (ur + (rb + (xb + value : U32) : U32) : U32) : U32) : U32),(cb + (rd + (gb + ln : U32) : U32) : U32),((cb + rd : U32) + (gb + ln : U32) : U32),Equal.sym(U32,((cb + rd : U32) + (gb + ln : U32) : U32),(cb + (rd + (gb + ln : U32) : U32) : U32),Arithmetic.add_associative(cb,rd,(gb + ln : U32))))
  +step_32 = Equal.cong(U32,U32,value => (t + (ub + (ur + (rb + (xb + (value + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(cb + rd : U32),(rd + cb : U32),Arithmetic.add_commutative(cb,rd))
  +step_33 = Equal.cong(U32,U32,value => (t + (ub + (ur + (rb + (xb + value : U32) : U32) : U32) : U32) : U32),((rd + cb : U32) + (gb + ln : U32) : U32),(rd + (cb + (gb + ln : U32) : U32) : U32),Arithmetic.add_associative(rd,cb,(gb + ln : U32)))
  +step_34 = Equal.cong(U32,U32,value => (t + (ub + (ur + (rb + value : U32) : U32) : U32) : U32),(xb + (rd + (cb + (gb + ln : U32) : U32) : U32) : U32),((xb + rd : U32) + (cb + (gb + ln : U32) : U32) : U32),Equal.sym(U32,((xb + rd : U32) + (cb + (gb + ln : U32) : U32) : U32),(xb + (rd + (cb + (gb + ln : U32) : U32) : U32) : U32),Arithmetic.add_associative(xb,rd,(cb + (gb + ln : U32) : U32))))
  +step_35 = Equal.cong(U32,U32,value => (t + (ub + (ur + (rb + (value + (cb + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(xb + rd : U32),(rd + xb : U32),Arithmetic.add_commutative(xb,rd))
  +step_36 = Equal.cong(U32,U32,value => (t + (ub + (ur + (rb + value : U32) : U32) : U32) : U32),((rd + xb : U32) + (cb + (gb + ln : U32) : U32) : U32),(rd + (xb + (cb + (gb + ln : U32) : U32) : U32) : U32),Arithmetic.add_associative(rd,xb,(cb + (gb + ln : U32) : U32)))
  +step_37 = Equal.sym(U32,(t + (ub + (ur + (rb + (rd + ((xb + cb : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + (rd + (xb + (cb + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),Equal.cong(U32,U32,value => (t + (ub + (ur + (rb + (rd + value : U32) : U32) : U32) : U32) : U32),((xb + cb : U32) + (gb + ln : U32) : U32),(xb + (cb + (gb + ln : U32) : U32) : U32),Arithmetic.add_associative(xb,cb,(gb + ln : U32))))
  +step_38 = Equal.sym(U32,(t + (ub + (ur + (rb + (rd + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + (rd + ((xb + cb : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),Equal.cong(U32,U32,value => (t + (ub + (ur + (rb + (rd + value : U32) : U32) : U32) : U32) : U32),(((xb + cb : U32) + gb : U32) + ln : U32),((xb + cb : U32) + (gb + ln : U32) : U32),Arithmetic.add_associative((xb + cb : U32),gb,ln)))
  +step_39 = Equal.sym(U32,(t + (ub + (ur + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + (rd + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32) : U32) : U32),Equal.cong(U32,U32,value => (t + (ub + (ur + value : U32) : U32) : U32),((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32),(rb + (rd + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32),Arithmetic.add_associative(rb,rd,(((xb + cb : U32) + gb : U32) + ln : U32))))
  +step_40 = Equal.sym(U32,(t + ((ub + ur : U32) + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32),(t + (ub + (ur + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32) : U32),Equal.cong(U32,U32,value => (t + value : U32),((ub + ur : U32) + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32),(ub + (ur + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32),Arithmetic.add_associative(ub,ur,((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32))))
  +step_41 = Equal.sym(U32,((t + (ub + ur : U32) : U32) + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32),(t + ((ub + ur : U32) + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32),Arithmetic.add_associative(t,(ub + ur : U32),((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32)))
  +step_42 = Equal.sym(U32,(((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32),((t + (ub + ur : U32) : U32) + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32),Arithmetic.add_associative((t + (ub + ur : U32) : U32),(rb + rd : U32),(((xb + cb : U32) + gb : U32) + ln : U32)))
  %Equal.sym(U32,((((((((t + rb : U32) + xb : U32) + cb : U32) + ub : U32) + rd : U32) + gb : U32) + ur : U32) + ln : U32),(((((((t + rb : U32) + xb : U32) + cb : U32) + ub : U32) + rd : U32) + gb : U32) + (ur + ln : U32) : U32),step_0) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(((((((t + rb : U32) + xb : U32) + cb : U32) + ub : U32) + rd : U32) + gb : U32) + (ur + ln : U32) : U32),((((((t + rb : U32) + xb : U32) + cb : U32) + ub : U32) + rd : U32) + (gb + (ur + ln : U32) : U32) : U32),step_1) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,((((((t + rb : U32) + xb : U32) + cb : U32) + ub : U32) + rd : U32) + (gb + (ur + ln : U32) : U32) : U32),(((((t + rb : U32) + xb : U32) + cb : U32) + ub : U32) + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32),step_2) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(((((t + rb : U32) + xb : U32) + cb : U32) + ub : U32) + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32),((((t + rb : U32) + xb : U32) + cb : U32) + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32),step_3) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,((((t + rb : U32) + xb : U32) + cb : U32) + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32),(((t + rb : U32) + xb : U32) + (cb + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32),step_4) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(((t + rb : U32) + xb : U32) + (cb + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32),((t + rb : U32) + (xb + (cb + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_5) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,((t + rb : U32) + (xb + (cb + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (rb + (xb + (cb + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_6) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (rb + (xb + (cb + (ub + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (rb + (xb + ((cb + ub : U32) + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_7) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (rb + (xb + ((cb + ub : U32) + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (rb + (xb + ((ub + cb : U32) + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_8) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (rb + (xb + ((ub + cb : U32) + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (rb + (xb + (ub + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_9) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (rb + (xb + (ub + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (rb + ((xb + ub : U32) + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_10) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (rb + ((xb + ub : U32) + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (rb + ((ub + xb : U32) + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_11) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (rb + ((ub + xb : U32) + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (rb + (ub + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_12) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (rb + (ub + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + ((rb + ub : U32) + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_13) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + ((rb + ub : U32) + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + ((ub + rb : U32) + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_14) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + ((ub + rb : U32) + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_15) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + (xb + (cb + (rd + (gb + (ur + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + (xb + (cb + (rd + ((gb + ur : U32) + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_16) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + (xb + (cb + (rd + ((gb + ur : U32) + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + (xb + (cb + (rd + ((ur + gb : U32) + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_17) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + (xb + (cb + (rd + ((ur + gb : U32) + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + (xb + (cb + (rd + (ur + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_18) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + (xb + (cb + (rd + (ur + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + (xb + (cb + ((rd + ur : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_19) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + (xb + (cb + ((rd + ur : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + (xb + (cb + ((ur + rd : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_20) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + (xb + (cb + ((ur + rd : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + (xb + (cb + (ur + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_21) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + (xb + (cb + (ur + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + (xb + ((cb + ur : U32) + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_22) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + (xb + ((cb + ur : U32) + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + (xb + ((ur + cb : U32) + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_23) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + (xb + ((ur + cb : U32) + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + (xb + (ur + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_24) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + (xb + (ur + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + ((xb + ur : U32) + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_25) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + ((xb + ur : U32) + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + ((ur + xb : U32) + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_26) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + ((ur + xb : U32) + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (rb + (ur + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_27) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (rb + (ur + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + ((rb + ur : U32) + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_28) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + ((rb + ur : U32) + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + ((ur + rb : U32) + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_29) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + ((ur + rb : U32) + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_30) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (ur + (rb + (xb + (cb + (rd + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + (xb + ((cb + rd : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_31) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (ur + (rb + (xb + ((cb + rd : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + (xb + ((rd + cb : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_32) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (ur + (rb + (xb + ((rd + cb : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + (xb + (rd + (cb + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_33) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (ur + (rb + (xb + (rd + (cb + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + ((xb + rd : U32) + (cb + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_34) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (ur + (rb + ((xb + rd : U32) + (cb + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + ((rd + xb : U32) + (cb + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_35) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (ur + (rb + ((rd + xb : U32) + (cb + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + (rd + (xb + (cb + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_36) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (ur + (rb + (rd + (xb + (cb + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + (rd + ((xb + cb : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),step_37) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (ur + (rb + (rd + ((xb + cb : U32) + (gb + ln : U32) : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + (rb + (rd + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32) : U32) : U32),step_38) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (ur + (rb + (rd + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32) : U32) : U32),(t + (ub + (ur + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32) : U32),step_39) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + (ub + (ur + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32) : U32),(t + ((ub + ur : U32) + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32),step_40) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,(t + ((ub + ur : U32) + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32) : U32),((t + (ub + ur : U32) : U32) + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32),step_41) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  %Equal.sym(U32,((t + (ub + ur : U32) : U32) + ((rb + rd : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32),(((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32),step_42) : {_ == (((t + (ub + ur : U32) : U32) + (rb + rd : U32) : U32) + (((xb + cb : U32) + gb : U32) + ln : U32) : U32) : U32}
  {==}
def expanded(+target: U32,+y: U32,+start: U32,+line: U32,+stride: U32,+first: U32,+width: U32,+x: U32,+chunk: U32,+group: U32,+unit: U32,+row: U32,+lane: U32,+plane: U32) -> U32:
  (target + (y + start * stride - first : U32) * width + x + chunk * 8 * stride + unit * 4 * plane + line * (stride * width : U32) + group * 8 * stride + row * plane + lane * stride : U32)
def normalized(+target: U32,+y: U32,+start: U32,+line: U32,+stride: U32,+first: U32,+width: U32,+x: U32,+chunk: U32,+group: U32,+unit: U32,+row: U32,+lane: U32,+plane: U32) -> U32:
  (target + (unit * 4 + row : U32) * plane + (y + (start + line : U32) * stride - first : U32) * width + (x + ((chunk + group : U32) * 8 + lane : U32) * stride : U32) : U32)
def normalized_matches(+target: U32,+y: U32,+start: U32,+line: U32,+stride: U32,+first: U32,+width: U32,+x: U32,+chunk: U32,+group: U32,+unit: U32,+row: U32,+lane: U32,+plane: U32) ->
  {expanded(target,y,start,line,stride,first,width,x,chunk,group,unit,row,lane,plane) == normalized(target,y,start,line,stride,first,width,x,chunk,group,unit,row,lane,plane) : U32}:
  +channels = ((unit * 4 * plane) + (row * plane) : U32)
  +rows = (((y + start * stride - first : U32) * width) + (line * (stride * width : U32)) : U32)
  +columns = (x + chunk * 8 * stride + group * 8 * stride + lane * stride : U32)
  +collected = (target + channels + rows + columns : U32)
  Equal.trans(U32,expanded(target,y,start,line,stride,first,width,x,chunk,group,unit,row,lane,plane),collected,normalized(target,y,start,line,stride,first,width,x,chunk,group,unit,row,lane,plane),store_coefficients(target,((y + start * stride - first : U32) * width : U32),x,(chunk * 8 * stride : U32),(unit * 4 * plane : U32),(line * (stride * width : U32) : U32),(group * 8 * stride : U32),(row * plane : U32),(lane * stride : U32)),
    %Equal.sym(U32,((unit * 4 + row : U32) * plane : U32),channels,Arithmetic.multiply_additive((unit * 4 : U32),row,plane)) : {collected == (target + _ + (y + (start + line : U32) * stride - first : U32) * width + (x + ((chunk + group : U32) * 8 + lane : U32) * stride : U32) : U32) : U32}
    %Equal.sym(U32,((y + (start + line : U32) * stride - first : U32) * width : U32),rows,row_product(y,start,line,stride,first,width)) : {collected == (target + channels + _ + (x + ((chunk + group : U32) * 8 + lane : U32) * stride : U32) : U32) : U32}
    %Equal.sym(U32,(x + ((chunk + group : U32) * 8 + lane : U32) * stride : U32),columns,column_product(x,chunk,group,lane,stride)) : {collected == (target + channels + rows + _ : U32) : U32}
    {==})
def cell_formula(+store: Stores.Store,+row: Nat,+lane: Nat) ->
  {Writes.cell_address(store,row,lane) == (Writes.store_target(store) + U32.from_nat(row) * Writes.store_plane(store) + U32.from_nat(lane) * Writes.store_step(store) : U32) : U32}:
  Stores.Store{+target,+step,+plane,rows,count} = store
  %Equal.sym(U32,Writes.address(target,plane,row),(target + U32.from_nat(row) * plane : U32),Writes.encoded_address(target,plane,row)) : {Writes.address(_,step,lane) == (target + U32.from_nat(row) * plane + U32.from_nat(lane) * step : U32) : U32}
  Writes.encoded_address((target + U32.from_nat(row) * plane : U32),step,lane)
def actual_expanded(+row_key: Accumulator.Row,+lane: Nat,+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) ->
  {Writes.cell_address(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed)),Store.row_number(row_key),lane) == expanded(Chunks.target(Chunks.query(Chunks.selection(seed))),U32.from_nat(Jobs.phase_y(Chunks.phase(Chunks.selection(seed)))),Jobs.start(Chunks.work(Chunks.selection(seed))),U32.from_nat(Chunks.line(seed)),Domain.stride_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))),Chunks.first(Chunks.query(Chunks.selection(seed))),Domain.width_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))),U32.from_nat(Jobs.phase_x(Chunks.phase(Chunks.selection(seed)))),Jobs.chunk_group(Chunks.chunk(Chunks.query(Chunks.selection(seed)))),U32.from_nat(Chunks.local_group(seed)),U32.from_nat(Chunks.unit(seed)),U32.from_nat(Store.row_number(row_key)),U32.from_nat(lane),Chunks.plane(Chunks.query(Chunks.selection(seed)))) : U32}:
  Chunks.Reduction{+selected,+line,+group,+unit,out,tr,tc} = seed
  Chunks.Selection{+query,+y,+x,yi,xi} = selected
  Chunks.Query{+grid,+first,+target,+plane,source,chunk_row,chunk_group,+wb,+ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,+w,k,+s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,+enabled,+ji,oi,ri,coli} = valid
  +work = Chunks.work(selected)
  +phase = Chunks.phase(selected)
  +active = Jobs.active_phase(phase,enabled)
  +sample = Chunks.sample(seed)
  +native_unit = Runtime.plan_unit(work,wb,ob,line,group,unit)
  +native_line = Lookup.read_line(sample)
  +native_group = Lookup.read_group(sample)
  +row = U32.from_nat(Store.row_number(row_key))
  +lane_word = U32.from_nat(lane)
  +cg = Jobs.chunk_group(Chunks.chunk(query))
  +start = Jobs.start(work)
  +unit_same = Runtime.plan_unit_exact(work,active,wb,ob,line,group,unit,ji)
  +counters = Reads.read_job_counters(work,active,wb,ob,line,group,unit,U32.from_nat(Runtime.tap(seed)),ji)
  +line_same = Shape.second({native_group == U32.from_nat(group) : U32},{native_line == U32.from_nat(line) : U32},counters)
  +group_same = Shape.first({native_group == U32.from_nat(group) : U32},{native_line == U32.from_nat(line) : U32},counters)
  +initial = expanded(target,U32.from_nat(y),start,native_line,s,first,w,U32.from_nat(x),cg,native_group,native_unit,row,lane_word,plane)
  +with_unit = expanded(target,U32.from_nat(y),start,native_line,s,first,w,U32.from_nat(x),cg,native_group,U32.from_nat(unit),row,lane_word,plane)
  +with_line = expanded(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,native_group,U32.from_nat(unit),row,lane_word,plane)
  +wanted = expanded(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,U32.from_nat(group),U32.from_nat(unit),row,lane_word,plane)
  Equal.trans(U32,Writes.cell_address(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed)),Store.row_number(row_key),lane),initial,wanted,
    cell_formula(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed)),Store.row_number(row_key),lane),
    Equal.trans(U32,initial,with_unit,wanted,Equal.cong(U32,U32,value => expanded(target,U32.from_nat(y),start,native_line,s,first,w,U32.from_nat(x),cg,native_group,value,row,lane_word,plane),native_unit,U32.from_nat(unit),unit_same),
      Equal.trans(U32,with_unit,with_line,wanted,Equal.cong(U32,U32,value => expanded(target,U32.from_nat(y),start,value,s,first,w,U32.from_nat(x),cg,native_group,U32.from_nat(unit),row,lane_word,plane),native_line,U32.from_nat(line),line_same),
        Equal.cong(U32,U32,value => expanded(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,value,U32.from_nat(unit),row,lane_word,plane),native_group,U32.from_nat(group),group_same))))
def physical_row_word(+seed: Chunks.Reduction) -> U32:
  (U32.from_nat(Jobs.phase_y(Chunks.phase(Chunks.selection(seed)))) + Jobs.row_native(Chunks.work(Chunks.selection(seed)),Chunks.line(seed)) * Domain.stride_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))) : U32)
def physical_row_exact(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) -> {U32.to_nat(physical_row_word(seed)) == Jobs.physical_row(Chunks.work(Chunks.selection(seed)),Chunks.line(seed)) : Nat}:
  Chunks.Reduction{+selected,+line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,+y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,+h,w,+k,+s,+p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,enabled,ji,oi,ri,coli} = valid
  PhaseGeometry.position_exact(U32.from_nat(y),h,k,s,p,Jobs.row_native(Chunks.work(selected),line),Machine.coordinate(l),Limits.coordinate_fits(l),
    PhaseBounds.height_coordinate_room(call),PhaseBounds.stride_double_room(call),Domain.stride_positive(call),Reads.phase_y_inside(Chunks.phase(selected)),Chunks.leaf_position_inside(seed,valid))
def local_row_word(+seed: Chunks.Reduction) -> U32:
  (physical_row_word(seed) - Chunks.first(Chunks.query(Chunks.selection(seed))) : U32)
def local_row_exact(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) ->
  {U32.to_nat(local_row_word(seed)) == Nat.sub(Jobs.physical_row(Chunks.work(Chunks.selection(seed)),Chunks.line(seed)),U32.to_nat(Chunks.first(Chunks.query(Chunks.selection(seed))))) : Nat}:
  Chunks.Reduction{+selected,+line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,+first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,+count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,+contained,fits,enabled,ji,oi,ri,coli} = valid
  +position = Jobs.physical_row(Chunks.work(selected),line)
  +row_span = Chunks.row_in_leaf(seed,valid)
  +lower = Shape.first({Nat.is_le(U32.to_nat(first),position) == True{} : Bool},{Nat.is_lt(position,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool},row_span)
  +upper = Shape.second({Nat.is_le(U32.to_nat(first),position) == True{} : Bool},{Nat.is_lt(position,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool},row_span)
  +height_inside = Bounds.nat_trans(position,Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(g),upper,contained)
  +capacity = Machine.capacity(l)
  +row_room = Limits.dimension_capacity(l,position,Order.transitive(position,Domain.height(g),U32.to_nat(Machine.dimension(l)),Bounds.strict_le(position,Domain.height(g),height_inside),Domain.height_room(call)))
  %physical_row_exact(seed,valid) : {U32.to_nat(local_row_word(seed)) == Nat.sub(_,U32.to_nat(first)) : Nat}
  Arithmetic.subtract(physical_row_word(seed),first,capacity,Limits.capacity_fits(l),
    %Equal.sym(Nat,U32.to_nat(physical_row_word(seed)),position,physical_row_exact(seed,valid)) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} lower,
    %Equal.sym(Nat,U32.to_nat(physical_row_word(seed)),position,physical_row_exact(seed,valid)) : {Nat.is_le(Nat.sub(_,U32.to_nat(first)),U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(Nat.sub(position,U32.to_nat(first)),position,U32.to_nat(capacity),Order.sub_le(position,U32.to_nat(first)),row_room))
def normalized_coordinates(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+row: Accumulator.Row,+lane: Nat) ->
  {normalized(Chunks.target(Chunks.query(Chunks.selection(seed))),U32.from_nat(Jobs.phase_y(Chunks.phase(Chunks.selection(seed)))),Jobs.start(Chunks.work(Chunks.selection(seed))),U32.from_nat(Chunks.line(seed)),Domain.stride_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))),Chunks.first(Chunks.query(Chunks.selection(seed))),Domain.width_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))),U32.from_nat(Jobs.phase_x(Chunks.phase(Chunks.selection(seed)))),Jobs.chunk_group(Chunks.chunk(Chunks.query(Chunks.selection(seed)))),U32.from_nat(Chunks.local_group(seed)),U32.from_nat(Chunks.unit(seed)),U32.from_nat(Store.row_number(row)),U32.from_nat(lane),Chunks.plane(Chunks.query(Chunks.selection(seed)))) ==
    Output.coordinate_address(Chunks.query(Chunks.selection(seed)),Channels.channel_word(Channels.channel_job(seed,valid),row),local_row_word(seed),Columns.physical_word(seed,lane)) : U32}:
  Chunks.Reduction{+selected,line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,enabled,+ji,oi,ri,coli} = valid
  Jobs.Inside{li,gi,ui} = ji
  +channel = Channels.channel_word(Channels.channel_job(seed,valid),row)
  +row_word = local_row_word(seed)
  +phase = Chunks.phase(selected)
  +source_column = Input.input_column(seed)
  +group_column = Columns.first_word(Columns.column_job(seed,valid))
  Equal.cong(U32,U32,value => Output.coordinate_address(query,channel,row_word,(U32.from_nat(Jobs.phase_x(phase)) + (value + U32.from_nat(lane) : U32) * Domain.stride_word(g) : U32)),group_column,source_column,Equal.sym(U32,source_column,group_column,Columns.input_column_matches(seed,valid)))
def actual_address(+row: Accumulator.Row,+lane: Nat,+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,
  +row_inside: {Nat.is_lt(Store.row_number(row),U32.to_nat(Writes.store_rows(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,U32.to_nat(Writes.store_count(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))))) == True{} : Bool}) ->
  {Writes.cell_address(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed)),Store.row_number(row),lane) == Output.native_address(Chunks.query(Chunks.selection(seed)),Output.job_point(seed,valid,row,lane,row_inside,lane_inside)) : U32}:
  Chunks.Reduction{+selected,+line,+group,+unit,out,tr,tc} = seed
  Chunks.Selection{+query,+y,+x,yi,xi} = selected
  Chunks.Query{+grid,+first,+target,+plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,+w,k,+s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,enabled,+ji,oi,ri,coli} = valid
  Jobs.Inside{li,gi,ui} = ji
  +point = Output.job_point(seed,valid,row,lane,row_inside,lane_inside)
  +start = Jobs.start(Chunks.work(selected))
  +cg = Jobs.chunk_group(Chunks.chunk(query))
  +row_number = U32.from_nat(Store.row_number(row))
  +lane_word = U32.from_nat(lane)
  +expanded_word = expanded(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,U32.from_nat(group),U32.from_nat(unit),row_number,lane_word,plane)
  +normalized_word = normalized(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,U32.from_nat(group),U32.from_nat(unit),row_number,lane_word,plane)
  +channel_job = Channels.channel_job(seed,valid)
  +channel_word = Channels.channel_word(channel_job,row)
  +row_word = local_row_word(seed)
  +column_word = Columns.physical_word(seed,lane)
  +coordinate_word = Output.coordinate_address(query,channel_word,row_word,column_word)
  +wanted = Output.native_address(query,point)
  Equal.trans(U32,Writes.cell_address(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed)),Store.row_number(row),lane),expanded_word,wanted,actual_expanded(row,lane,seed,valid),
    Equal.trans(U32,expanded_word,normalized_word,wanted,normalized_matches(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,U32.from_nat(group),U32.from_nat(unit),row_number,lane_word,plane),
      Equal.trans(U32,normalized_word,coordinate_word,wanted,normalized_coordinates(seed,valid,row,lane),
        Output.coordinate_address_matches(query,point,channel_word,row_word,column_word,
          Channels.channel_word_exact(channel_job,row,%Channels.rows_matches(seed,valid) : {Nat.is_lt(Store.row_number(row),U32.to_nat(_)) == True{} : Bool} row_inside),local_row_exact(seed,valid),Columns.physical_word_exact(seed,valid,lane,lane_inside)))))
