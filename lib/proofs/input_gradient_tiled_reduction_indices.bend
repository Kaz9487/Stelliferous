# Every native MAC ordinal denotes exactly one bounded phase product.
import Base
import ../kernel_input_gradient.bend as Kernel
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_job_reads.bend as Reads
import ./input_gradient_phase_packing.bend as Packing
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./kernel_packing_window_coordinates.bend as Window
import ./row_major_coordinates.bend as Coordinates
import ./input_gradient_tiled_band_extents.bend as Bands

def count(+phase: Jobs.Phase) -> Nat:
  U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))
def rows(+phase: Jobs.Phase) -> Nat:
  U32.to_nat(Packing.phase_rows(Jobs.context(phase)))
def columns(+phase: Jobs.Phase) -> Nat:
  U32.to_nat(Packing.phase_columns(Jobs.context(phase)))
def values(+phase: Jobs.Phase,index: Nat) -> Coordinates.Values:
  Coordinates.values(rows(phase),columns(phase),index)
def count_exact(+phase: Jobs.Phase) -> {count(phase) == Nat.mul(Domain.outputs(Jobs.geometry(phase)),Nat.mul(rows(phase),columns(phase))) : Nat}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  PhaseBounds.phase_inner_exact(call,U32.from_nat(y),U32.from_nat(x))
def columns_positive_from(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool},geometry: Window.Geometry) ->
  {Nat.is_lt(0n,Window.columns(geometry)) == True{} : Bool}:
  Window.Geometry{window,channels,capacity,word_ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = geometry
  cp
def columns_positive(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool}) -> {Nat.is_lt(0n,columns(phase)) == True{} : Bool}:
  %Reads.coordinates_columns(phase,enabled) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool}
  columns_positive_from(phase,enabled,Reads.coordinates(phase,enabled))
def bounded(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,count(phase)) == True{} : Bool}) ->
  Coordinates.Bounded<Domain.outputs(Jobs.geometry(phase)),rows(phase),columns(phase),values(phase,index)>:
  Coordinates.values_inside(Domain.outputs(Jobs.geometry(phase)),rows(phase),columns(phase),index,columns_positive(phase,enabled),%count_exact(phase) : {Nat.is_lt(index,_) == True{} : Bool} inside)
def exact(+phase: Jobs.Phase,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase)),0) == False{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,count(phase)) == True{} : Bool}) ->
  {index == Packing.phase_index(Jobs.context(phase),Coordinates.channel(values(phase,index)),Coordinates.row(values(phase,index)),Coordinates.column(values(phase,index))) : Nat}:
  Coordinates.values_exact(Domain.outputs(Jobs.geometry(phase)),rows(phase),columns(phase),index,columns_positive(phase,enabled),%count_exact(phase) : {Nat.is_lt(index,_) == True{} : Bool} inside)
def at(+seed: Chunks.Reduction,+index: Nat) -> Chunks.Reduction:
  Chunks.Reduction{+selected,line,group,unit,out,row,column} = seed
  +value = values(Chunks.phase(selected),index)
  Chunks.Reduction{selected,line,group,unit,Coordinates.channel(value),Coordinates.row(value),Coordinates.column(value)}
def valid_at_from(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+index: Nat,coordinates: Coordinates.Bounded<Domain.outputs(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))),rows(Chunks.phase(Chunks.selection(seed))),columns(Chunks.phase(Chunks.selection(seed))),values(Chunks.phase(Chunks.selection(seed)),index)>) -> Chunks.Valid<at(seed,index)>:
  Chunks.Reduction{+selected,line,group,unit,out,row,column} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,enabled,ji,oi,ri,coli} = valid
  Coordinates.Bounded{new_output,new_row,new_column} = coordinates
  Chunks.Valid{ci,contained,fits,enabled,ji,new_output,new_row,new_column}
def valid_at(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+index: Nat,+inside: {Nat.is_lt(index,count(Chunks.phase(Chunks.selection(seed)))) == True{} : Bool}) -> Chunks.Valid<at(seed,index)>:
  Chunks.Valid{ci,contained,fits,+enabled,ji,oi,ri,coli} = valid
  valid_at_from(seed,valid,index,bounded(Chunks.phase(Chunks.selection(seed)),enabled,index,inside))
def tap_exact(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+index: Nat,+inside: {Nat.is_lt(index,count(Chunks.phase(Chunks.selection(seed)))) == True{} : Bool}) ->
  {index == Packing.phase_index(Jobs.context(Chunks.phase(Chunks.selection(at(seed,index)))),Chunks.output(at(seed,index)),Chunks.tap_row(at(seed,index)),Chunks.tap_column(at(seed,index))) : Nat}:
  Chunks.Reduction{+selected,line,group,unit,out,row,column} = seed
  Chunks.Valid{ci,contained,fits,enabled,ji,oi,ri,coli} = valid
  exact(Chunks.phase(selected),enabled,index,inside)
def leaf_band_at_from(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+index: Nat,+coordinates: Coordinates.Bounded<Domain.outputs(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))),rows(Chunks.phase(Chunks.selection(seed))),columns(Chunks.phase(Chunks.selection(seed))),values(Chunks.phase(Chunks.selection(seed)),index)>) ->
  {Chunks.leaf_band(at(seed,index),valid_at_from(seed,valid,index,coordinates)) == Chunks.leaf_band(seed,valid) : Bands.Query}:
  Chunks.Reduction{Chunks.Selection{Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob},y,x,yi,xi},line,group,unit,out,row,column} = seed
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,enabled,ji,oi,ri,coli} = valid
  Coordinates.Bounded{new_output,new_row,new_column} = coordinates
  {==}
def leaf_band_at(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+index: Nat,+inside: {Nat.is_lt(index,count(Chunks.phase(Chunks.selection(seed)))) == True{} : Bool}) ->
  {Chunks.leaf_band(at(seed,index),valid_at(seed,valid,index,inside)) == Chunks.leaf_band(seed,valid) : Bands.Query}:
  Chunks.Valid{ci,contained,fits,+enabled,ji,oi,ri,coli} = valid
  leaf_band_at_from(seed,valid,index,bounded(Chunks.phase(Chunks.selection(seed)),enabled,index,inside))
