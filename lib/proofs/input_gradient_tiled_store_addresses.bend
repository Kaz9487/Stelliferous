# The address row r of a phase job writes is its tile group, row and column
# in the leaf output.
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
import ./kernel_gemm_jobs.bend as GemmJobs
import ./encoded_coordinates.bend as Encoded
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_job_reads.bend as Reads
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_tiled_product_reads.bend as Runtime
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_entry_reads.bend as Lookup
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
def column_product(+first: U32,+chunk: U32,+group: U32,+offset: U32,+stride: U32) ->
  {(first + ((chunk + group : U32) * 8 + offset : U32) * stride : U32) == (first + chunk * 8 * stride + group * 8 * stride + offset * stride : U32) : U32}:
  +base = ((chunk + group : U32) * 8 : U32)
  +left = (chunk * 8 * stride : U32)
  +right = (group * 8 * stride : U32)
  +last = (offset * stride : U32)
  +combined = {
    Equal.trans(U32,(base * stride : U32),((chunk + group : U32) * (8 * stride : U32) : U32),(left + right : U32),Arithmetic.multiply_associative((chunk + group : U32),8,stride),
      Equal.trans(U32,((chunk + group : U32) * (8 * stride : U32) : U32),((chunk * (8 * stride : U32)) + (group * (8 * stride : U32)) : U32),(left + right : U32),Arithmetic.multiply_additive(chunk,group,(8 * stride : U32)),
        Equal.trans(U32,((chunk * (8 * stride : U32)) + (group * (8 * stride : U32)) : U32),(left + (group * (8 * stride : U32)) : U32),(left + right : U32),
          Equal.cong(U32,U32,value => (value + group * (8 * stride : U32) : U32),(chunk * (8 * stride : U32) : U32),left,Equal.sym(U32,left,(chunk * (8 * stride : U32) : U32),Arithmetic.multiply_associative(chunk,8,stride))),
          Equal.cong(U32,U32,value => (left + value : U32),(group * (8 * stride : U32) : U32),right,Equal.sym(U32,right,(group * (8 * stride : U32) : U32),Arithmetic.multiply_associative(group,8,stride)))))) : {(base * stride : U32) == (left + right : U32) : U32}}
  %Equal.sym(U32,((base + offset : U32) * stride : U32),((base * stride) + last : U32),Arithmetic.multiply_additive(base,offset,stride)) : {(first + _ : U32) == (first + left + right + last : U32) : U32}
  %Equal.sym(U32,(base * stride : U32),(left + right : U32),combined) : {(first + (_ + last : U32) : U32) == (first + left + right + last : U32) : U32}
  Equal.sym(U32,(first + left + right + last : U32),(first + ((left + right : U32) + last : U32) : U32),
    Equal.trans(U32,(first + left + right + last : U32),((first + (left + right : U32) : U32) + last : U32),(first + ((left + right : U32) + last : U32) : U32),Equal.cong(U32,U32,value => (value + last : U32),(first + left + right : U32),(first + (left + right : U32) : U32),Arithmetic.add_associative(first,left,right)),Arithmetic.add_associative(first,(left + right : U32),last)))

# Eight summands of a store address in the order the address is normalized.
def collect(+t: U32,+p: U32,+xc: U32,+cc: U32,+u: U32,+l: U32,+g: U32,+r: U32) ->
  {((((t + ((p + xc : U32) + cc : U32) : U32) + u : U32) + (l + g : U32) : U32) + r : U32) == (t + (((p + l : U32) + (((xc + cc : U32) + g : U32) + r : U32) : U32) + u : U32) : U32) : U32}:
  +x1 = ((p + xc : U32) + cc : U32)
  +k = (xc + cc : U32)
  +m = ((l + g : U32) + r : U32)
  +wanted = (t + (((p + l : U32) + ((k + g : U32) + r : U32) : U32) + u : U32) : U32)
  %Equal.sym(U32,((((t + x1 : U32) + u : U32) + (l + g : U32) : U32) + r : U32),(((t + x1 : U32) + u : U32) + m : U32),Arithmetic.add_associative(((t + x1 : U32) + u : U32),(l + g : U32),r)) :
    {_ == wanted : U32}
  %Equal.sym(U32,(((t + x1 : U32) + u : U32) + m : U32),((t + x1 : U32) + (u + m : U32) : U32),Arithmetic.add_associative((t + x1 : U32),u,m)) :
    {_ == wanted : U32}
  %Equal.sym(U32,((t + x1 : U32) + (u + m : U32) : U32),(t + (x1 + (u + m : U32) : U32) : U32),Arithmetic.add_associative(t,x1,(u + m : U32))) :
    {_ == wanted : U32}
  %Equal.sym(U32,(u + m : U32),(m + u : U32),Arithmetic.add_commutative(u,m)) :
    {(t + (x1 + _ : U32) : U32) == wanted : U32}
  %Arithmetic.add_associative(x1,m,u) :
    {(t + _ : U32) == wanted : U32}
  %Equal.sym(U32,x1,(p + k : U32),Arithmetic.add_associative(p,xc,cc)) :
    {(t + ((_ + m : U32) + u : U32) : U32) == wanted : U32}
  %Equal.sym(U32,((p + k : U32) + m : U32),(p + (k + m : U32) : U32),Arithmetic.add_associative(p,k,m)) :
    {(t + (_ + u : U32) : U32) == wanted : U32}
  %Arithmetic.add_associative(k,(l + g : U32),r) :
    {(t + ((p + _ : U32) + u : U32) : U32) == wanted : U32}
  %Arithmetic.add_associative(k,l,g) :
    {(t + ((p + (_ + r : U32) : U32) + u : U32) : U32) == wanted : U32}
  %Equal.sym(U32,(k + l : U32),(l + k : U32),Arithmetic.add_commutative(k,l)) :
    {(t + ((p + ((_ + g : U32) + r : U32) : U32) + u : U32) : U32) == wanted : U32}
  %Equal.sym(U32,((l + k : U32) + g : U32),(l + (k + g : U32) : U32),Arithmetic.add_associative(l,k,g)) :
    {(t + ((p + (_ + r : U32) : U32) + u : U32) : U32) == wanted : U32}
  %Equal.sym(U32,((l + (k + g : U32) : U32) + r : U32),(l + ((k + g : U32) + r : U32) : U32),Arithmetic.add_associative(l,(k + g : U32),r)) :
    {(t + ((p + _ : U32) + u : U32) : U32) == wanted : U32}
  %Arithmetic.add_associative(p,l,((k + g : U32) + r : U32)) :
    {(t + (_ + u : U32) : U32) == wanted : U32}
  {==}
# The address a store writes, with the parentheses of the plan: the cell of
# the first column, the unit, the line and column group, then the row.
def expanded(+target: U32,+y: U32,+start: U32,+line: U32,+stride: U32,+first: U32,+width: U32,+x: U32,+chunk: U32,+group: U32,+unit: U32,+row: U32,+cells: U32) -> U32:
  ((target + (((y + start * stride - first : U32) * width : U32) + x + chunk * 8 * stride : U32) * cells : U32) + unit + (line * (stride * width : U32) + group * 8 * stride : U32) * cells + row * (stride * cells : U32) : U32)
# The same address as row, column and tile group of the output.
def normalized(+target: U32,+y: U32,+start: U32,+line: U32,+stride: U32,+first: U32,+width: U32,+x: U32,+chunk: U32,+group: U32,+unit: U32,+row: U32,+cells: U32) -> U32:
  (target + (((y + (start + line : U32) * stride - first : U32) * width + (x + ((chunk + group : U32) * 8 + row : U32) * stride : U32) : U32) * cells + unit : U32) : U32)
# Both as the sum of the target, the unit and six products with the cells.
def spread(+target: U32,+y: U32,+start: U32,+line: U32,+stride: U32,+first: U32,+width: U32,+x: U32,+chunk: U32,+group: U32,+unit: U32,+row: U32,+cells: U32) -> U32:
  +rows_low = ((y + start * stride - first : U32) * width : U32)
  +lines = (line * (stride * width : U32) : U32)
  +chunks = (chunk * 8 * stride : U32)
  +groups = (group * 8 * stride : U32)
  +offset = (row * stride : U32)
  (target + ((((rows_low * cells : U32) + (lines * cells : U32) : U32) + ((((x * cells : U32) + (chunks * cells : U32) : U32) + (groups * cells : U32) : U32) + (offset * cells : U32) : U32) : U32) + unit : U32) : U32)
def expanded_spread(+target: U32,+y: U32,+start: U32,+line: U32,+stride: U32,+first: U32,+width: U32,+x: U32,+chunk: U32,+group: U32,+unit: U32,+row: U32,+cells: U32) ->
  {expanded(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells) == spread(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells) : U32}:
  +rows_low = ((y + start * stride - first : U32) * width : U32)
  +lines = (line * (stride * width : U32) : U32)
  +chunks = (chunk * 8 * stride : U32)
  +groups = (group * 8 * stride : U32)
  +offset = (row * stride : U32)
  +wanted = spread(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells)
  %Arithmetic.multiply_associative(row,stride,cells) :
    {((((target + ((rows_low + x : U32) + chunks : U32) * cells : U32) + unit : U32) + (lines + groups : U32) * cells : U32) + _ : U32) == wanted : U32}
  %Equal.sym(U32,((lines + groups : U32) * cells : U32),((lines * cells : U32) + (groups * cells : U32) : U32),Arithmetic.multiply_additive(lines,groups,cells)) :
    {((((target + ((rows_low + x : U32) + chunks : U32) * cells : U32) + unit : U32) + _ : U32) + (offset * cells : U32) : U32) == wanted : U32}
  %Equal.sym(U32,(((rows_low + x : U32) + chunks : U32) * cells : U32),(((rows_low + x : U32) * cells : U32) + (chunks * cells : U32) : U32),Arithmetic.multiply_additive((rows_low + x : U32),chunks,cells)) :
    {((((target + _ : U32) + unit : U32) + ((lines * cells : U32) + (groups * cells : U32) : U32) : U32) + (offset * cells : U32) : U32) == wanted : U32}
  %Equal.sym(U32,((rows_low + x : U32) * cells : U32),((rows_low * cells : U32) + (x * cells : U32) : U32),Arithmetic.multiply_additive(rows_low,x,cells)) :
    {((((target + (_ + (chunks * cells : U32) : U32) : U32) + unit : U32) + ((lines * cells : U32) + (groups * cells : U32) : U32) : U32) + (offset * cells : U32) : U32) == wanted : U32}
  collect(target,(rows_low * cells : U32),(x * cells : U32),(chunks * cells : U32),unit,(lines * cells : U32),(groups * cells : U32),(offset * cells : U32))
def normalized_spread(+target: U32,+y: U32,+start: U32,+line: U32,+stride: U32,+first: U32,+width: U32,+x: U32,+chunk: U32,+group: U32,+unit: U32,+row: U32,+cells: U32) ->
  {normalized(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells) == spread(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells) : U32}:
  +rows_low = ((y + start * stride - first : U32) * width : U32)
  +lines = (line * (stride * width : U32) : U32)
  +chunks = (chunk * 8 * stride : U32)
  +groups = (group * 8 * stride : U32)
  +offset = (row * stride : U32)
  +columns = (((x + chunks : U32) + groups : U32) + offset : U32)
  +wanted = spread(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells)
  +rows_high = ((y + (start + line : U32) * stride - first : U32) * width : U32)
  +columns_high = (x + ((chunk + group : U32) * 8 + row : U32) * stride : U32)
  %Equal.sym(U32,rows_high,(rows_low + lines : U32),row_product(y,start,line,stride,first,width)) :
    {(target + ((_ + columns_high : U32) * cells + unit : U32) : U32) == wanted : U32}
  %Equal.sym(U32,columns_high,columns,column_product(x,chunk,group,row,stride)) :
    {(target + (((rows_low + lines : U32) + _ : U32) * cells + unit : U32) : U32) == wanted : U32}
  %Equal.sym(U32,(((rows_low + lines : U32) + columns : U32) * cells : U32),(((rows_low + lines : U32) * cells : U32) + (columns * cells : U32) : U32),Arithmetic.multiply_additive((rows_low + lines : U32),columns,cells)) :
    {(target + (_ + unit : U32) : U32) == wanted : U32}
  %Equal.sym(U32,((rows_low + lines : U32) * cells : U32),((rows_low * cells : U32) + (lines * cells : U32) : U32),Arithmetic.multiply_additive(rows_low,lines,cells)) :
    {(target + ((_ + (columns * cells : U32) : U32) + unit : U32) : U32) == wanted : U32}
  %Equal.sym(U32,(columns * cells : U32),((((x + chunks : U32) + groups : U32) * cells : U32) + (offset * cells : U32) : U32),Arithmetic.multiply_additive(((x + chunks : U32) + groups : U32),offset,cells)) :
    {(target + ((((rows_low * cells : U32) + (lines * cells : U32) : U32) + _ : U32) + unit : U32) : U32) == wanted : U32}
  %Equal.sym(U32,(((x + chunks : U32) + groups : U32) * cells : U32),(((x + chunks : U32) * cells : U32) + (groups * cells : U32) : U32),Arithmetic.multiply_additive((x + chunks : U32),groups,cells)) :
    {(target + ((((rows_low * cells : U32) + (lines * cells : U32) : U32) + (_ + (offset * cells : U32) : U32) : U32) + unit : U32) : U32) == wanted : U32}
  %Equal.sym(U32,((x + chunks : U32) * cells : U32),((x * cells : U32) + (chunks * cells : U32) : U32),Arithmetic.multiply_additive(x,chunks,cells)) :
    {(target + ((((rows_low * cells : U32) + (lines * cells : U32) : U32) + ((_ + (groups * cells : U32) : U32) + (offset * cells : U32) : U32) : U32) + unit : U32) : U32) == wanted : U32}
  {==}
def normalized_matches(+target: U32,+y: U32,+start: U32,+line: U32,+stride: U32,+first: U32,+width: U32,+x: U32,+chunk: U32,+group: U32,+unit: U32,+row: U32,+cells: U32) ->
  {expanded(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells) == normalized(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells) : U32}:
  Equal.trans(U32,expanded(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells),spread(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells),
    normalized(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells),
    expanded_spread(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells),
    Equal.sym(U32,normalized(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells),spread(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells),
      normalized_spread(target,y,start,line,stride,first,width,x,chunk,group,unit,row,cells)))
def actual_expanded(+row: Nat,+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) ->
  {GemmJobs.cell_address(~Kernel.Cursor(),Runtime.job(seed),row,0n) == expanded(Chunks.target(Chunks.query(Chunks.selection(seed))),U32.from_nat(Jobs.phase_y(Chunks.phase(Chunks.selection(seed)))),Jobs.start(Chunks.work(Chunks.selection(seed))),U32.from_nat(Chunks.line(seed)),Domain.stride_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))),Chunks.first(Chunks.query(Chunks.selection(seed))),Domain.width_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))),U32.from_nat(Jobs.phase_x(Chunks.phase(Chunks.selection(seed)))),Jobs.chunk_group(Chunks.chunk(Chunks.query(Chunks.selection(seed)))),U32.from_nat(Chunks.local_group(seed)),U32.from_nat(Chunks.unit(seed)),U32.from_nat(row),Output.cells(Chunks.query(Chunks.selection(seed)))) : U32}:
  Chunks.Reduction{+selected,+line,+group,+unit,+out,+tr,+tc} = seed
  Chunks.Selection{+query,+y,+x,yi,xi} = selected
  Chunks.Query{+grid,+first,+target,source,chunk_row,chunk_group,+wb} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{+c,o,h,+w,k,+s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,+enabled,+ji,oi,ri,coli} = valid
  +work = Chunks.work(selected)
  +active = Jobs.active_phase(Chunks.phase(selected),enabled)
  +sample = Chunks.sample(seed)
  +native_unit = (Runtime.job_index(seed) % Jobs.native_units(work) : U32)
  +native_line = Lookup.read_line(sample)
  +native_group = Lookup.read_group(sample)
  +row_word = U32.from_nat(row)
  +cg = Jobs.chunk_group(Chunks.chunk(query))
  +units = Output.cells(query)
  +start = Jobs.start(work)
  +unit_same = Encoded.encoding(native_unit,unit,Jobs.unit_decode(work,active,line,group,unit,ji))
  +counters = Reads.read_job_counters(work,active,wb,line,group,unit,out,tr,tc,ji)
  +line_same = Shape.second({native_group == U32.from_nat(group) : U32},{native_line == U32.from_nat(line) : U32},counters)
  +group_same = Shape.first({native_group == U32.from_nat(group) : U32},{native_line == U32.from_nat(line) : U32},counters)
  +initial = expanded(target,U32.from_nat(y),start,native_line,s,first,w,U32.from_nat(x),cg,native_group,native_unit,row_word,units)
  +with_unit = expanded(target,U32.from_nat(y),start,native_line,s,first,w,U32.from_nat(x),cg,native_group,U32.from_nat(unit),row_word,units)
  +with_line = expanded(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,native_group,U32.from_nat(unit),row_word,units)
  +wanted = expanded(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,U32.from_nat(group),U32.from_nat(unit),row_word,units)
  Equal.trans(U32,(initial + 0 : U32),initial,wanted,Arithmetic.add_zero(initial),
    Equal.trans(U32,initial,with_unit,wanted,Equal.cong(U32,U32,value => expanded(target,U32.from_nat(y),start,native_line,s,first,w,U32.from_nat(x),cg,native_group,value,row_word,units),native_unit,U32.from_nat(unit),unit_same),
      Equal.trans(U32,with_unit,with_line,wanted,Equal.cong(U32,U32,value => expanded(target,U32.from_nat(y),start,value,s,first,w,U32.from_nat(x),cg,native_group,U32.from_nat(unit),row_word,units),native_line,U32.from_nat(line),line_same),
        Equal.cong(U32,U32,value => expanded(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,value,U32.from_nat(unit),row_word,units),native_group,U32.from_nat(group),group_same))))
def physical_row_word(+seed: Chunks.Reduction) -> U32:
  (U32.from_nat(Jobs.phase_y(Chunks.phase(Chunks.selection(seed)))) + Jobs.row_native(Chunks.work(Chunks.selection(seed)),Chunks.line(seed)) * Domain.stride_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))) : U32)
def physical_row_exact(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) -> {U32.to_nat(physical_row_word(seed)) == Jobs.physical_row(Chunks.work(Chunks.selection(seed)),Chunks.line(seed)) : Nat}:
  Chunks.Reduction{+selected,+line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,+y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,source,chunk_row,chunk_group,wb} = query
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
  Chunks.Query{+grid,+first,target,source,chunk_row,chunk_group,wb} = query
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
def normalized_coordinates(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+row: Nat) ->
  {normalized(Chunks.target(Chunks.query(Chunks.selection(seed))),U32.from_nat(Jobs.phase_y(Chunks.phase(Chunks.selection(seed)))),Jobs.start(Chunks.work(Chunks.selection(seed))),U32.from_nat(Chunks.line(seed)),Domain.stride_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))),Chunks.first(Chunks.query(Chunks.selection(seed))),Domain.width_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))),U32.from_nat(Jobs.phase_x(Chunks.phase(Chunks.selection(seed)))),Jobs.chunk_group(Chunks.chunk(Chunks.query(Chunks.selection(seed)))),U32.from_nat(Chunks.local_group(seed)),U32.from_nat(Chunks.unit(seed)),U32.from_nat(row),Output.cells(Chunks.query(Chunks.selection(seed)))) ==
    Output.coordinate_address(Chunks.query(Chunks.selection(seed)),U32.from_nat(Chunks.unit(seed)),local_row_word(seed),Columns.physical_word(seed,row)) : U32}:
  Chunks.Reduction{+selected,line,group,+unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,source,chunk_row,chunk_group,wb} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,enabled,+ji,oi,ri,coli} = valid
  Jobs.Inside{li,gi,ui} = ji
  +row_word = local_row_word(seed)
  +phase = Chunks.phase(selected)
  +source_column = Input.input_column(seed)
  +group_column = Columns.first_word(Columns.column_job(seed,valid))
  Equal.cong(U32,U32,value => Output.coordinate_address(query,U32.from_nat(unit),row_word,(U32.from_nat(Jobs.phase_x(phase)) + (value + U32.from_nat(row) : U32) * Domain.stride_word(g) : U32)),group_column,source_column,Equal.sym(U32,source_column,group_column,Columns.input_column_matches(seed,valid)))
def actual_address(+row: Nat,+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,
  +row_inside: {Nat.is_lt(row,GemmJobs.job_rows(~Kernel.Cursor(),Runtime.job(seed))) == True{} : Bool}) ->
  {GemmJobs.cell_address(~Kernel.Cursor(),Runtime.job(seed),row,0n) == Output.native_address(Chunks.query(Chunks.selection(seed)),Output.job_point(seed,valid,row,row_inside)) : U32}:
  Chunks.Reduction{+selected,+line,+group,+unit,out,tr,tc} = seed
  Chunks.Selection{+query,+y,+x,yi,xi} = selected
  Chunks.Query{+grid,+first,+target,source,chunk_row,chunk_group,wb} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{+c,o,h,+w,k,+s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,enabled,+ji,oi,ri,coli} = valid
  Jobs.Inside{li,gi,ui} = ji
  +point = Output.job_point(seed,valid,row,row_inside)
  +start = Jobs.start(Chunks.work(selected))
  +cg = Jobs.chunk_group(Chunks.chunk(query))
  +units = Output.cells(query)
  +row_number = U32.from_nat(row)
  +expanded_word = expanded(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,U32.from_nat(group),U32.from_nat(unit),row_number,units)
  +normalized_word = normalized(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,U32.from_nat(group),U32.from_nat(unit),row_number,units)
  +group_word = U32.from_nat(unit)
  +row_word = local_row_word(seed)
  +column_word = Columns.physical_word(seed,row)
  +coordinate_word = Output.coordinate_address(query,group_word,row_word,column_word)
  +wanted = Output.native_address(query,point)
  Equal.trans(U32,GemmJobs.cell_address(~Kernel.Cursor(),Runtime.job(seed),row,0n),expanded_word,wanted,actual_expanded(row,seed,valid),
    Equal.trans(U32,expanded_word,normalized_word,wanted,normalized_matches(target,U32.from_nat(y),start,U32.from_nat(line),s,first,w,U32.from_nat(x),cg,U32.from_nat(group),U32.from_nat(unit),row_number,units),
      Equal.trans(U32,normalized_word,coordinate_word,wanted,normalized_coordinates(seed,valid,row),
        Output.coordinate_address_matches(query,point,group_word,row_word,column_word,
          Channels.unit_exact(Channels.channel_job(seed,valid)),local_row_exact(seed,valid),Columns.physical_word_exact(seed,valid,row,row_inside)))))
