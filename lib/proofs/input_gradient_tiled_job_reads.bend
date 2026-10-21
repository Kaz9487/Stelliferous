# Actual job and reduction counters select bounded filled and packed entries.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_geometry.bend as Geometry
import ./input_gradient_tiled_capacity.bend as Capacity
import ./input_gradient_tiled_lookup.bend as Lookup
import ./input_gradient_tiled_extents.bend as Extents
import ./input_gradient_tiled_columns.bend as Columns
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_band_geometry.bend as Samples
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./input_gradient_phase_packing.bend as Packing
import ./kernel_packing_window_coordinates.bend as Coordinates
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./encoded_coordinates.bend as Encoded
import ./storage_shape.bend as Shape
import ./bounded_u32_arithmetic.bend as Arithmetic

def phase_y_inside(+phase: Jobs.Phase) -> {Nat.is_lt(U32.to_nat(U32.from_nat(Jobs.phase_y(phase))),Domain.height(Jobs.geometry(phase))) == True{} : Bool}:
  Jobs.Phase{+call,e,+y,x,+yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +query = Jobs.rows(phase)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(y)),y,Extents.index_exact(query,y,yi)) : {Nat.is_lt(_,Domain.height(g)) == True{} : Bool}
  Bounds.nat_trans(y,Extents.count(query),Domain.height(g),yi,Extents.count_size_room(query))
def phase_x_inside(+phase: Jobs.Phase) -> {Nat.is_lt(U32.to_nat(U32.from_nat(Jobs.phase_x(phase))),Domain.width(Jobs.geometry(phase))) == True{} : Bool}:
  Jobs.Phase{+call,e,y,+x,yi,+xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +query = Jobs.columns(phase)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(x)),x,Extents.index_exact(query,x,xi)) : {Nat.is_lt(_,Domain.width(g)) == True{} : Bool}
  Bounds.nat_trans(x,Extents.count(query),Domain.width(g),xi,Extents.count_size_room(query))
def coordinates(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool}) -> Coordinates.Geometry:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  PhaseBounds.window_geometry(call,U32.from_nat(y),U32.from_nat(x),phase_y_inside(phase),phase_x_inside(phase),enabled)
def coordinates_rows(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool}) ->
  {Coordinates.row_word(coordinates(phase,enabled)) == Packing.phase_rows(Jobs.context(phase)) : U32}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}
def coordinates_columns(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool}) ->
  {Coordinates.column_word(coordinates(phase,enabled)) == Packing.phase_columns(Jobs.context(phase)) : U32}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}
def coordinates_outputs(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool}) ->
  {Coordinates.channels(coordinates(phase,enabled)) == Domain.outputs(Jobs.geometry(phase)) : Nat}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}
def coordinates_index(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool},+output: Nat,+row: Nat,+column: Nat) ->
  {Coordinates.reduction(coordinates(phase,enabled),output,row,column) == Packing.phase_index(Jobs.context(phase),output,row,column) : Nat}:
  +cr = Coordinates.rows(coordinates(phase,enabled))
  +cc = Coordinates.columns(coordinates(phase,enabled))
  +pr = U32.to_nat(Packing.phase_rows(Jobs.context(phase)))
  +pc = U32.to_nat(Packing.phase_columns(Jobs.context(phase)))
  %Equal.sym(U32,Coordinates.row_word(coordinates(phase,enabled)),Packing.phase_rows(Jobs.context(phase)),coordinates_rows(phase,enabled)) : {RowMajor.index(output,U32.to_nat(_),cc,row,column) == Packing.phase_index(Jobs.context(phase),output,row,column) : Nat}
  %Equal.sym(U32,Coordinates.column_word(coordinates(phase,enabled)),Packing.phase_columns(Jobs.context(phase)),coordinates_columns(phase,enabled)) : {RowMajor.index(output,pr,U32.to_nat(_),row,column) == Packing.phase_index(Jobs.context(phase),output,row,column) : Nat}
  {==}
def decode(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool},+output: Nat,+row: Nat,+column: Nat,
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(phase))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(phase)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(phase)))) == True{} : Bool}) ->
  Coordinates.Decoded<coordinates(phase,enabled),output,row,column>:
  +q = coordinates(phase,enabled)
  Coordinates.decode(q,output,row,column,
    %Equal.sym(Nat,Coordinates.channels(q),Domain.outputs(Jobs.geometry(phase)),coordinates_outputs(phase,enabled)) : {Nat.is_lt(output,_) == True{} : Bool} output_inside,
    %Equal.sym(U32,Coordinates.row_word(q),Packing.phase_rows(Jobs.context(phase)),coordinates_rows(phase,enabled)) : {Nat.is_lt(row,U32.to_nat(_)) == True{} : Bool} row_inside,
    %Equal.sym(U32,Coordinates.column_word(q),Packing.phase_columns(Jobs.context(phase)),coordinates_columns(phase,enabled)) : {Nat.is_lt(column,U32.to_nat(_)) == True{} : Bool} column_inside)

type Taps<-phase: Jobs.Phase,-output: Nat,-row: Nat,-column: Nat> is Data:
  Taps{output_equal: {(U32.from_nat(Packing.phase_index(Jobs.context(phase),output,row,column)) / (Packing.phase_rows(Jobs.context(phase)) * Packing.phase_columns(Jobs.context(phase))) : U32) == U32.from_nat(output) : U32},
    row_equal: {(U32.from_nat(Packing.phase_index(Jobs.context(phase),output,row,column)) / Packing.phase_columns(Jobs.context(phase)) % Packing.phase_rows(Jobs.context(phase)) : U32) == U32.from_nat(row) : U32},
    column_equal: {(U32.from_nat(Packing.phase_index(Jobs.context(phase),output,row,column)) % Packing.phase_columns(Jobs.context(phase)) : U32) == U32.from_nat(column) : U32}}
def taps_from(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool},+output: Nat,+row: Nat,+column: Nat,
  decoded: Coordinates.Decoded<coordinates(phase,enabled),output,row,column>) -> Taps<phase,output,row,column>:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Coordinates.Decoded{+oe,+re,+ce} = decoded
  +same = coordinates_index(phase,enabled,output,row,column)
  Taps{
    %same : {(U32.from_nat(_) / (Packing.phase_rows(Jobs.context(phase)) * Packing.phase_columns(Jobs.context(phase))) : U32) == U32.from_nat(output) : U32} oe,
    %same : {(U32.from_nat(_) / Packing.phase_columns(Jobs.context(phase)) % Packing.phase_rows(Jobs.context(phase)) : U32) == U32.from_nat(row) : U32} re,
    %same : {(U32.from_nat(_) % Packing.phase_columns(Jobs.context(phase)) : U32) == U32.from_nat(column) : U32} ce}
def taps(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool},+output: Nat,+row: Nat,+column: Nat,
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(phase))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(phase)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(phase)))) == True{} : Bool}) -> Taps<phase,output,row,column>:
  taps_from(phase,enabled,output,row,column,decode(phase,enabled,output,row,column,output_inside,row_inside,column_inside))

def read(+work: Jobs.Work,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+relative: U32) -> Lookup.Read:
  Lookup.phase_read(Jobs.context(Jobs.phase(work)),Jobs.chunk(work),wb,ob,Jobs.job_index(work,line,group,unit),relative)

def read_job_counters(+work: Jobs.Work,+active: Jobs.Active<Jobs.phase(work)>,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+relative: U32,
  +inside: Jobs.Inside<work,line,group,unit>) ->
  Shape.Both<{Lookup.read_group(read(work,wb,ob,line,group,unit,relative)) == U32.from_nat(group) : U32},
    {Lookup.read_line(read(work,wb,ob,line,group,unit,relative)) == U32.from_nat(line) : U32}>:
  Jobs.Work{Jobs.Phase{+call,e,y,x,yi,xi},+chunk} = work
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},chunk_row,chunk_rows,low,chunk_group} = chunk
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,chunk_size} = layout
  +sample = read(work,wb,ob,line,group,unit,relative)
  Shape.Both{Encoded.encoding(Lookup.read_group(sample),group,Jobs.group_decode(work,active,line,group,unit,inside)),
    Encoded.encoding(Lookup.read_line(sample),line,Jobs.line_decode(work,active,line,group,unit,inside))}

def chunk_low(chunk: Kernel.Chunk) -> U32:
  Kernel.Chunk{leaf,row,rows,low,group} = chunk
  low
def source_line_matches(+work: Jobs.Work,+line: U32) ->
  {(Lookup.source_row(Jobs.context(Jobs.phase(work)),Jobs.chunk(work)) + line : U32) ==
    (((Jobs.start(work) + line) + Geometry.axis_low(Jobs.geometry(Jobs.phase(work)),U32.from_nat(Jobs.phase_y(Jobs.phase(work))),Domain.height_word(Jobs.geometry(Jobs.phase(work))))) - chunk_low(Jobs.chunk(work)) : U32) : U32}:
  Jobs.Work{Jobs.Phase{+call,e,y,x,yi,xi},+chunk} = work
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,layout},first,count,target,plane},row,rows,+low,group} = chunk
  +start = Jobs.start(work)
  +base = Geometry.axis_low(g,U32.from_nat(y),h)
  +origin = (start + base : U32)
  +ys = Packing.axis_source(Packing.phase_y(Jobs.context(Jobs.phase(work))))
  +grouped = Equal.cong(U32,U32,value => ((value - low) + line : U32),(start + ys + k : U32),origin,Arithmetic.add_associative(start,ys,k))
  +sum = Equal.trans(U32,(origin + line : U32),(start + (base + line) : U32),((start + line) + base : U32),Arithmetic.add_associative(start,base,line),
    Equal.trans(U32,(start + (base + line) : U32),(start + (line + base) : U32),((start + line) + base : U32),Equal.cong(U32,U32,value => (start + value : U32),(base + line : U32),(line + base : U32),Arithmetic.add_commutative(base,line)),Equal.sym(U32,((start + line) + base : U32),(start + (line + base) : U32),Arithmetic.add_associative(start,line,base))))
  Equal.trans(U32,(Lookup.source_row(Jobs.context(Jobs.phase(work)),chunk) + line : U32),((origin - low) + line : U32),(((start + line) + base) - low : U32),grouped,
    Equal.trans(U32,((origin - low) + line : U32),((origin + line) - low : U32),(((start + line) + base) - low : U32),Arithmetic.add_after_subtract(origin,low,line),
      Equal.cong(U32,U32,value => (value - low : U32),(origin + line : U32),((start + line) + base : U32),sum)))

def row_source_relative(+work: Jobs.Work,+active: Jobs.Active<Jobs.phase(work)>,+line: Nat,+tap: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(Jobs.chunk_row(Jobs.chunk(work))),U32.to_nat(Jobs.chunk_rows(Jobs.chunk(work)))),Domain.height(Jobs.geometry(Jobs.phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(Jobs.lines(work))) == True{} : Bool},
  +low_matches: {chunk_low(Jobs.chunk(work)) == (Kernel.band_low(Jobs.geometry(Jobs.phase(work)),Jobs.chunk_row(Jobs.chunk(work)),(Jobs.chunk_row(Jobs.chunk(work)) + Jobs.chunk_rows(Jobs.chunk(work)) : U32)) : U32) : U32}) ->
  {(Lookup.source_row(Jobs.context(Jobs.phase(work)),Jobs.chunk(work)) + U32.from_nat(line) + tap : U32) ==
    Bands.relative(Jobs.work_band(work,contained),Samples.native_tapped(Jobs.row_selection(work,active,line,contained,inside),Jobs.row_native(work,line),tap)) : U32}:
  Jobs.Work{+phase,+chunk} = work
  Jobs.Phase{+call,e,+y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,layout},first,count,target,plane},+row,+rows,+chunk_low,group} = chunk
  +query = Jobs.work_band(work,contained)
  +position = Jobs.row_native(work,line)
  +base = Geometry.axis_low(g,U32.from_nat(y),h)
  +low = Kernel.band_low(g,row,(row + rows : U32))
  +raw = (Lookup.source_row(Jobs.context(phase),chunk) + U32.from_nat(line) + tap : U32)
  +shifted = (position + base - chunk_low + tap : U32)
  +wanted = Bands.relative(query,Samples.native_tapped(Jobs.row_selection(work,active,line,contained,inside),position,tap))
  +moved = {%Equal.sym(U32,chunk_low,low,low_matches) : {(position + base - _ + tap : U32) == wanted : U32}
    Bands.relative_native_matches(query,Jobs.work_band_rows(work,contained),y,Jobs.work_phase_y_inside(work,contained),Jobs.row_band_used(work,active,line,contained,inside),position,tap) : {shifted == wanted : U32}}
  Equal.trans(U32,raw,shifted,wanted,
    Equal.cong(U32,U32,value => (value + tap : U32),(Lookup.source_row(Jobs.context(phase),chunk) + U32.from_nat(line) : U32),(position + base - chunk_low : U32),source_line_matches(work,U32.from_nat(line))),
    moved)

def coordinate_words_exact(+query: Coordinates.Geometry,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,Coordinates.rows(query)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,Coordinates.columns(query)) == True{} : Bool}) ->
  Shape.Both<{U32.to_nat(U32.from_nat(row)) == row : Nat},{U32.to_nat(U32.from_nat(column)) == column : Nat}>:
  Coordinates.Geometry{window,channels,+cap,+word_ok,rp,cp,ip,wp,ie,ce,+rr,+cr,ar,ir,wr} = query
  +rn = Coordinates.rows(query)
  +cn = Coordinates.columns(query)
  Shape.Both{Bounds.roundtrip(row,cap,word_ok,Bounds.strict_le(row,U32.to_nat(cap),Bounds.nat_trans(row,rn,U32.to_nat(cap),row_inside,Order.transitive(rn,Nat.double(rn),U32.to_nat(cap),Order.le_double(rn),rr)))),
    Bounds.roundtrip(column,cap,word_ok,Bounds.strict_le(column,U32.to_nat(cap),Bounds.nat_trans(column,cn,U32.to_nat(cap),column_inside,Order.transitive(cn,Nat.double(cn),U32.to_nat(cap),Order.le_double(cn),cr))))}
def tap_words_exact(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool},+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(phase)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(phase)))) == True{} : Bool}) ->
  Shape.Both<{U32.to_nat(U32.from_nat(row)) == row : Nat},{U32.to_nat(U32.from_nat(column)) == column : Nat}>:
  +q = coordinates(phase,enabled)
  coordinate_words_exact(q,row,column,
    %Equal.sym(U32,Coordinates.row_word(q),Packing.phase_rows(Jobs.context(phase)),coordinates_rows(phase,enabled)) : {Nat.is_lt(row,U32.to_nat(_)) == True{} : Bool} row_inside,
    %Equal.sym(U32,Coordinates.column_word(q),Packing.phase_columns(Jobs.context(phase)),coordinates_columns(phase,enabled)) : {Nat.is_lt(column,U32.to_nat(_)) == True{} : Bool} column_inside)

def read_base_matches(+work: Jobs.Work,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+relative: U32) ->
  {Lookup.read_row(read(work,wb,ob,line,group,unit,relative)) == Lookup.source_row(Jobs.context(Jobs.phase(work)),Jobs.chunk(work)) : U32}:
  Jobs.Work{Jobs.Phase{+call,e,y,x,yi,xi},chunk} = work
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},chunk_row,chunk_rows,low,chunk_group} = chunk
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,chunk_size} = layout
  {==}
def read_row_relative_finish(+work: Jobs.Work,+active: Jobs.Active<Jobs.phase(work)>,+line: Nat,+tap: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(Jobs.chunk_row(Jobs.chunk(work))),U32.to_nat(Jobs.chunk_rows(Jobs.chunk(work)))),Domain.height(Jobs.geometry(Jobs.phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(Jobs.lines(work))) == True{} : Bool},
  +low_matches: {chunk_low(Jobs.chunk(work)) == (Kernel.band_low(Jobs.geometry(Jobs.phase(work)),Jobs.chunk_row(Jobs.chunk(work)),(Jobs.chunk_row(Jobs.chunk(work)) + Jobs.chunk_rows(Jobs.chunk(work)) : U32)) : U32) : U32},
  +sample: Lookup.Read,+base_matches: {Lookup.read_row(sample) == Lookup.source_row(Jobs.context(Jobs.phase(work)),Jobs.chunk(work)) : U32},
  +line_matches: {Lookup.read_line(sample) == U32.from_nat(line) : U32},+tap_matches: {Lookup.read_tap_row(sample) == U32.from_nat(tap) : U32}) ->
  {Lookup.read_row_total(sample) == Bands.relative(Jobs.work_band(work,contained),Samples.native_tapped(Jobs.row_selection(work,active,line,contained,inside),Jobs.row_native(work,line),U32.from_nat(tap))) : U32}:
  +wanted = Bands.relative(Jobs.work_band(work,contained),Samples.native_tapped(Jobs.row_selection(work,active,line,contained,inside),Jobs.row_native(work,line),U32.from_nat(tap)))
  %Equal.sym(U32,Lookup.read_row(sample),Lookup.source_row(Jobs.context(Jobs.phase(work)),Jobs.chunk(work)),base_matches) : {(_ + Lookup.read_line(sample) + Lookup.read_tap_row(sample) : U32) == wanted : U32}
  %Equal.sym(U32,Lookup.read_line(sample),U32.from_nat(line),line_matches) : {(Lookup.source_row(Jobs.context(Jobs.phase(work)),Jobs.chunk(work)) + _ + Lookup.read_tap_row(sample) : U32) == wanted : U32}
  %Equal.sym(U32,Lookup.read_tap_row(sample),U32.from_nat(tap),tap_matches) : {(Lookup.source_row(Jobs.context(Jobs.phase(work)),Jobs.chunk(work)) + U32.from_nat(line) + _ : U32) == wanted : U32}
  row_source_relative(work,active,line,U32.from_nat(tap),contained,inside,low_matches)

def read_job_line(+work: Jobs.Work,+active: Jobs.Active<Jobs.phase(work)>,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+relative: U32,
  +inside: Jobs.Inside<work,line,group,unit>) -> {Lookup.read_line(read(work,wb,ob,line,group,unit,relative)) == U32.from_nat(line) : U32}:
  Shape.second({Lookup.read_group(read(work,wb,ob,line,group,unit,relative)) == U32.from_nat(group) : U32},
    {Lookup.read_line(read(work,wb,ob,line,group,unit,relative)) == U32.from_nat(line) : U32},read_job_counters(work,active,wb,ob,line,group,unit,relative,inside))
def read_tap_row_from(+work: Jobs.Work,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+output: Nat,+row: Nat,+column: Nat,
  decoded: Taps<Jobs.phase(work),output,row,column>) ->
  {Lookup.read_tap_row(read(work,wb,ob,line,group,unit,U32.from_nat(Packing.phase_index(Jobs.context(Jobs.phase(work)),output,row,column)))) == U32.from_nat(row) : U32}:
  Jobs.Work{Jobs.Phase{+call,e,y,x,yi,xi},chunk} = work
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},chunk_row,chunk_rows,low,chunk_group} = chunk
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,chunk_size} = layout
  Taps{oe,re,ce} = decoded
  re
def read_row_relative(+work: Jobs.Work,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Jobs.phase(work))),0) == False{} : Bool},+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+output: Nat,+row: Nat,+column: Nat,
  +inside: Jobs.Inside<work,line,group,unit>,
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(Jobs.phase(work)))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(Jobs.phase(work))))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(Jobs.phase(work))))) == True{} : Bool},
  +contained: {Nat.is_le(Nat.add(U32.to_nat(Jobs.chunk_row(Jobs.chunk(work))),U32.to_nat(Jobs.chunk_rows(Jobs.chunk(work)))),Domain.height(Jobs.geometry(Jobs.phase(work)))) == True{} : Bool},
  +low_matches: {chunk_low(Jobs.chunk(work)) == (Kernel.band_low(Jobs.geometry(Jobs.phase(work)),Jobs.chunk_row(Jobs.chunk(work)),(Jobs.chunk_row(Jobs.chunk(work)) + Jobs.chunk_rows(Jobs.chunk(work)) : U32)) : U32) : U32}) ->
  {Lookup.read_row_total(read(work,wb,ob,line,group,unit,U32.from_nat(Packing.phase_index(Jobs.context(Jobs.phase(work)),output,row,column)))) ==
    Bands.relative(Jobs.work_band(work,contained),Samples.native_tapped(Jobs.row_selection(work,Jobs.active_phase(Jobs.phase(work),enabled),line,contained,Jobs.line_inside(work,line,group,unit,inside)),Jobs.row_native(work,line),U32.from_nat(row))) : U32}:
  Jobs.Inside{+li,gi,ui} = inside
  +active = Jobs.active_phase(Jobs.phase(work),enabled)
  +relative = U32.from_nat(Packing.phase_index(Jobs.context(Jobs.phase(work)),output,row,column))
  read_row_relative_finish(work,active,line,row,contained,li,low_matches,read(work,wb,ob,line,group,unit,relative),read_base_matches(work,wb,ob,line,group,unit,relative),
    read_job_line(work,active,wb,ob,line,group,unit,relative,inside),read_tap_row_from(work,wb,ob,line,group,unit,output,row,column,taps(Jobs.phase(work),enabled,output,row,column,output_inside,row_inside,column_inside)))

def read_row_inside(+work: Jobs.Work,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Jobs.phase(work))),0) == False{} : Bool},+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+output: Nat,+row: Nat,+column: Nat,
  +inside: Jobs.Inside<work,line,group,unit>,
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(Jobs.phase(work)))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(Jobs.phase(work))))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(Jobs.phase(work))))) == True{} : Bool},
  +contained: {Nat.is_le(Nat.add(U32.to_nat(Jobs.chunk_row(Jobs.chunk(work))),U32.to_nat(Jobs.chunk_rows(Jobs.chunk(work)))),Domain.height(Jobs.geometry(Jobs.phase(work)))) == True{} : Bool},
  +low_matches: {chunk_low(Jobs.chunk(work)) == (Kernel.band_low(Jobs.geometry(Jobs.phase(work)),Jobs.chunk_row(Jobs.chunk(work)),(Jobs.chunk_row(Jobs.chunk(work)) + Jobs.chunk_rows(Jobs.chunk(work)) : U32)) : U32) : U32}) ->
  {Nat.is_lt(U32.to_nat(Lookup.read_row_total(read(work,wb,ob,line,group,unit,U32.from_nat(Packing.phase_index(Jobs.context(Jobs.phase(work)),output,row,column))))),U32.to_nat(Kernel.band_rows(Jobs.geometry(Jobs.phase(work)),Jobs.chunk_row(Jobs.chunk(work)),Jobs.chunk_rows(Jobs.chunk(work))))) == True{} : Bool}:
  Jobs.Inside{+li,gi,ui} = inside
  +active = Jobs.active_phase(Jobs.phase(work),enabled)
  +exact = Shape.first({U32.to_nat(U32.from_nat(row)) == row : Nat},{U32.to_nat(U32.from_nat(column)) == column : Nat},tap_words_exact(Jobs.phase(work),enabled,row,column,row_inside,column_inside))
  +selected = Jobs.row_selection(work,active,line,contained,li)
  +tap_inside = {
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(row)),row,exact) : {Nat.is_lt(_,Samples.selection_taps(selected)) == True{} : Bool}
    %Equal.sym(Nat,Samples.selection_taps(selected),U32.to_nat(Packing.phase_rows(Jobs.context(Jobs.phase(work)))),Jobs.row_selection_taps(work,active,line,contained,li)) : {Nat.is_lt(row,_) == True{} : Bool} row_inside : {Nat.is_lt(U32.to_nat(U32.from_nat(row)),Samples.selection_taps(selected)) == True{} : Bool}}
  +sample = read(work,wb,ob,line,group,unit,U32.from_nat(Packing.phase_index(Jobs.context(Jobs.phase(work)),output,row,column)))
  +wanted = Bands.relative(Jobs.work_band(work,contained),Samples.native_tapped(selected,Jobs.row_native(work,line),U32.from_nat(row)))
  %Equal.sym(U32,Lookup.read_row_total(sample),wanted,read_row_relative(work,enabled,wb,ob,line,group,unit,output,row,column,inside,output_inside,row_inside,column_inside,contained,low_matches)) :
    {Nat.is_lt(U32.to_nat(_),U32.to_nat(Kernel.band_rows(Jobs.geometry(Jobs.phase(work)),Jobs.chunk_row(Jobs.chunk(work)),Jobs.chunk_rows(Jobs.chunk(work))))) == True{} : Bool}
  Jobs.row_tapped_relative(work,active,line,U32.from_nat(row),contained,li,tap_inside)

def first_shift_matches(+work: Jobs.Work,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+relative: U32,
  +layout_matches: {Lookup.chunk_layout(Jobs.chunk(work)) == Kernel.tiling(Jobs.geometry(Jobs.phase(work))) : Kernel.Tiling}) ->
  {Lookup.read_first_shift(read(work,wb,ob,line,group,unit,relative)) ==
    (Geometry.axis_low(Jobs.geometry(Jobs.phase(work)),U32.from_nat(Jobs.phase_x(Jobs.phase(work))),Domain.width_word(Jobs.geometry(Jobs.phase(work)))) - Columns.layout_column_low(Kernel.tiling(Jobs.geometry(Jobs.phase(work)))) : U32) : U32}:
  Jobs.Work{Jobs.Phase{+call,e,y,+x,yi,xi},+chunk} = work
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,+w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},row,rows,low,chunk_group} = chunk
  Kernel.Tiling{groups,shifts,outputs,+column_low,kernel,height,width,cg,chunk_size} = layout
  Equal.cong(U32,U32,value => (Geometry.axis_low(g,U32.from_nat(x),w) - value : U32),column_low,Columns.layout_column_low(Kernel.tiling(g)),Equal.cong(Kernel.Tiling,U32,Columns.layout_column_low,layout,Kernel.tiling(g),layout_matches))
def read_shift_relative_finish(+work: Jobs.Work,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+relative: U32,+tap: Nat,
  +layout_matches: {Lookup.chunk_layout(Jobs.chunk(work)) == Kernel.tiling(Jobs.geometry(Jobs.phase(work))) : Kernel.Tiling},
  +tap_matches: {Lookup.read_tap_column(read(work,wb,ob,line,group,unit,relative)) == U32.from_nat(tap) : U32}) ->
  {Lookup.read_shift_total(read(work,wb,ob,line,group,unit,relative)) == ((Geometry.axis_low(Jobs.geometry(Jobs.phase(work)),U32.from_nat(Jobs.phase_x(Jobs.phase(work))),Domain.width_word(Jobs.geometry(Jobs.phase(work)))) - Columns.layout_column_low(Kernel.tiling(Jobs.geometry(Jobs.phase(work))))) + U32.from_nat(tap) : U32) : U32}:
  Jobs.Work{Jobs.Phase{+call,e,y,+x,yi,xi},chunk} = work
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +sample = read(work,wb,ob,line,group,unit,relative)
  +base = (Geometry.axis_low(g,U32.from_nat(x),w) - Columns.layout_column_low(Kernel.tiling(g)) : U32)
  %Equal.sym(U32,Lookup.read_tap_column(sample),U32.from_nat(tap),tap_matches) : {(_ + Lookup.read_first_shift(sample) : U32) == (base + U32.from_nat(tap) : U32) : U32}
  %Equal.sym(U32,Lookup.read_first_shift(sample),base,first_shift_matches(work,wb,ob,line,group,unit,relative,layout_matches)) : {(U32.from_nat(tap) + _ : U32) == (base + U32.from_nat(tap) : U32) : U32}
  Arithmetic.add_commutative(U32.from_nat(tap),base)

def read_tap_column_from(+work: Jobs.Work,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+output: Nat,+row: Nat,+column: Nat,
  decoded: Taps<Jobs.phase(work),output,row,column>) ->
  {Lookup.read_tap_column(read(work,wb,ob,line,group,unit,U32.from_nat(Packing.phase_index(Jobs.context(Jobs.phase(work)),output,row,column)))) == U32.from_nat(column) : U32}:
  Jobs.Work{Jobs.Phase{+call,e,y,x,yi,xi},chunk} = work
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},chunk_row,chunk_rows,low,chunk_group} = chunk
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,chunk_size} = layout
  Taps{oe,re,ce} = decoded
  ce
def read_shift_inside(+work: Jobs.Work,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Jobs.phase(work))),0) == False{} : Bool},+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+output: Nat,+row: Nat,+column: Nat,
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(Jobs.phase(work)))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(Jobs.phase(work))))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(Jobs.phase(work))))) == True{} : Bool},
  +layout_matches: {Lookup.chunk_layout(Jobs.chunk(work)) == Kernel.tiling(Jobs.geometry(Jobs.phase(work))) : Kernel.Tiling}) ->
  {Nat.is_lt(U32.to_nat(Lookup.read_shift_total(read(work,wb,ob,line,group,unit,U32.from_nat(Packing.phase_index(Jobs.context(Jobs.phase(work)),output,row,column))))),U32.to_nat(Columns.layout_shifts(Kernel.tiling(Jobs.geometry(Jobs.phase(work)))))) == True{} : Bool}:
  Jobs.Work{+phase,chunk} = work
  Jobs.Phase{+call,+e,y,+x,yi,+xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +relative = U32.from_nat(Packing.phase_index(Jobs.context(phase),output,row,column))
  +sample = read(work,wb,ob,line,group,unit,relative)
  +wanted = ((Geometry.axis_low(g,U32.from_nat(x),w) - Columns.layout_column_low(Kernel.tiling(g))) + U32.from_nat(column) : U32)
  +tap_matches = read_tap_column_from(work,wb,ob,line,group,unit,output,row,column,taps(phase,enabled,output,row,column,output_inside,row_inside,column_inside))
  +same = read_shift_relative_finish(work,wb,ob,line,group,unit,relative,column,layout_matches,tap_matches)
  +exact = Shape.second({U32.to_nat(U32.from_nat(row)) == row : Nat},{U32.to_nat(U32.from_nat(column)) == column : Nat},tap_words_exact(phase,enabled,row,column,row_inside,column_inside))
  %Equal.sym(U32,Lookup.read_shift_total(sample),wanted,same) : {Nat.is_lt(U32.to_nat(_),U32.to_nat(Columns.layout_shifts(Kernel.tiling(g)))) == True{} : Bool}
  Columns.column_shift_inside(call,e,x,xi,Jobs.column_active(phase,Jobs.active_phase(phase,enabled)),U32.from_nat(column),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(column)),column,exact) : {Nat.is_lt(_,U32.to_nat(Extents.axis_taps(Jobs.columns(phase),U32.from_nat(x)))) == True{} : Bool} column_inside)

def filled_span(+work: Jobs.Work) -> U32:
  +layout = Lookup.chunk_layout(Jobs.chunk(work))
  U32.min(Lookup.layout_groups(layout),(Capacity.layout_groups(layout) - Jobs.chunk_group(Jobs.chunk(work)) : U32))
def job_span_inside(+work: Jobs.Work,+active: Jobs.Active<Jobs.phase(work)>,
  +layout_matches: {Lookup.chunk_layout(Jobs.chunk(work)) == Kernel.tiling(Jobs.geometry(Jobs.phase(work))) : Kernel.Tiling}) ->
  {Nat.is_le(U32.to_nat(Jobs.groups(work)),U32.to_nat(filled_span(work))) == True{} : Bool}:
  Jobs.Work{+phase,+chunk} = work
  Jobs.Phase{+call,+e,y,+x,yi,+xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},row,rows,low,+group} = chunk
  Kernel.Tiling{+groups,shifts,outputs,column_low,kernel,height,width,+cg,chunk_size} = layout
  +full = Capacity.layout_groups(Kernel.tiling(g))
  +same = Equal.cong(Kernel.Tiling,U32,Capacity.layout_groups,layout,Kernel.tiling(g),layout_matches)
  %Equal.sym(U32,groups,full,same) : {Nat.is_le(U32.to_nat(Jobs.groups(work)),U32.to_nat(U32.min(cg,(_ - group : U32)))) == True{} : Bool}
  Jobs.group_span_monotone(group,Jobs.native_groups(phase),cg,full,Jobs.phase_groups_bounded(call,e,x,xi,Jobs.column_active(phase,active)))
def read_group_inside(+work: Jobs.Work,+active: Jobs.Active<Jobs.phase(work)>,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+relative: U32,
  +inside: Jobs.Inside<work,line,group,unit>,
  +layout_matches: {Lookup.chunk_layout(Jobs.chunk(work)) == Kernel.tiling(Jobs.geometry(Jobs.phase(work))) : Kernel.Tiling}) ->
  {Nat.is_lt(U32.to_nat(Lookup.read_group(read(work,wb,ob,line,group,unit,relative))),U32.to_nat(filled_span(work))) == True{} : Bool}:
  Jobs.Work{Jobs.Phase{+call,e,y,x,yi,xi},chunk} = work
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},chunk_row,chunk_rows,low,chunk_group} = chunk
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,chunk_size} = layout
  Jobs.Inside{li,+gi,ui} = inside
  +sample = read(work,wb,ob,line,group,unit,relative)
  %Equal.sym(Nat,U32.to_nat(Lookup.read_group(sample)),group,Jobs.group_decode(work,active,line,group,unit,inside)) : {Nat.is_lt(_,U32.to_nat(filled_span(work))) == True{} : Bool}
  Bounds.nat_trans(group,U32.to_nat(Jobs.groups(work)),U32.to_nat(filled_span(work)),gi,job_span_inside(work,active,layout_matches))

def output_word_exact(+phase: Jobs.Phase,+output: Nat,+inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(phase))) == True{} : Bool}) -> {U32.to_nat(U32.from_nat(output)) == output : Nat}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  +limits = Domain.limits(call)
  Bounds.roundtrip(output,Machine.dimension(limits),Limits.dimension_fits(limits),Bounds.strict_le(output,U32.to_nat(Machine.dimension(limits)),Bounds.nat_trans(output,Domain.outputs(Domain.geometry(call)),U32.to_nat(Machine.dimension(limits)),inside,Domain.outputs_room(call))))
def read_output_from(+work: Jobs.Work,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+output: Nat,+row: Nat,+column: Nat,
  decoded: Taps<Jobs.phase(work),output,row,column>) ->
  {Lookup.read_output(read(work,wb,ob,line,group,unit,U32.from_nat(Packing.phase_index(Jobs.context(Jobs.phase(work)),output,row,column)))) == U32.from_nat(output) : U32}:
  Jobs.Work{Jobs.Phase{+call,e,y,x,yi,xi},chunk} = work
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},chunk_row,chunk_rows,low,chunk_group} = chunk
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,chunk_size} = layout
  Taps{oe,re,ce} = decoded
  oe
def read_output_inside(+work: Jobs.Work,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Jobs.phase(work))),0) == False{} : Bool},+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+output: Nat,+row: Nat,+column: Nat,
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(Jobs.phase(work)))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(Jobs.phase(work))))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(Jobs.phase(work))))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(Lookup.read_output(read(work,wb,ob,line,group,unit,U32.from_nat(Packing.phase_index(Jobs.context(Jobs.phase(work)),output,row,column))))),Domain.outputs(Jobs.geometry(Jobs.phase(work)))) == True{} : Bool}:
  +sample = read(work,wb,ob,line,group,unit,U32.from_nat(Packing.phase_index(Jobs.context(Jobs.phase(work)),output,row,column)))
  %Equal.sym(U32,Lookup.read_output(sample),U32.from_nat(output),read_output_from(work,wb,ob,line,group,unit,output,row,column,taps(Jobs.phase(work),enabled,output,row,column,output_inside,row_inside,column_inside))) : {Nat.is_lt(U32.to_nat(_),Domain.outputs(Jobs.geometry(Jobs.phase(work)))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(output)),output,output_word_exact(Jobs.phase(work),output,output_inside)) : {Nat.is_lt(_,Domain.outputs(Jobs.geometry(Jobs.phase(work)))) == True{} : Bool}
  output_inside
