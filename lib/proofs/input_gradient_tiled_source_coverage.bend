# Retained rows may be the full image, the requested copy, or a containing interval.
import Base
import ../kernel_input_gradient.bend as Kernel
import ./nat_to_u32_bounds.bend as Bounds
import ./input_gradient_tiled_source_rows.bend as Rows
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_copy_geometry.bend as Copy
import ./input_gradient_phase_domain.bend as Domain
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./storage_shape.bend as Shape

type Contains<-required_first: U32,-required_rows: U32,-first: U32,-rows: U32,-height: U32> is Data:
  Full{first: {first == 0 : U32},rows: {rows == height : U32}}
  Exact{first: {first == required_first : U32},rows: {rows == required_rows : U32}}
  Superset{lower: {Nat.is_le(U32.to_nat(first),U32.to_nat(required_first)) == True{} : Bool},upper: {Nat.is_le(Nat.add(U32.to_nat(required_first),U32.to_nat(required_rows)),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}}
  Empty{rows: {required_rows == 0 : U32}}
def cover(+query: Bands.Query,+first: U32,+rows: U32,contains: Contains<Copy.begin(query),Copy.row_count(query),first,rows,Domain.gradient_rows(Bands.geometry(query))>) -> Rows.Covers<query,first,rows>:
  match contains:
    case Full{+fe,+re}:
      %Equal.sym(U32,first,0,fe) : Rows.Covers<query,_,rows>
      %Equal.sym(U32,rows,Domain.gradient_rows(Bands.geometry(query)),re) : Rows.Covers<query,0,_>
      Rows.full_cover(query)
    case Exact{+fe,+re}:
      %Equal.sym(U32,first,Copy.begin(query),fe) : Rows.Covers<query,_,rows>
      %Equal.sym(U32,rows,Copy.row_count(query),re) : Rows.Covers<query,Copy.begin(query),_>
      %Copy.begin_matches(query) : Rows.Covers<query,_,Copy.row_count(query)>
      %Copy.count_matches(query) : Rows.Covers<query,Kernel.copy_start(Bands.geometry(query),Bands.first(query),Bands.rows_word(query)),_>
      Rows.copy_cover(query)
    case Superset{+lower,+upper}:
      Rows.Covers{row => lo => hi => Shape.Both{
        Order.transitive(U32.to_nat(first),U32.to_nat(Copy.begin(query)),U32.to_nat(row),lower,lo),
        Bounds.nat_trans(U32.to_nat(row),Nat.add(U32.to_nat(Copy.begin(query)),U32.to_nat(Copy.row_count(query))),Nat.add(U32.to_nat(first),U32.to_nat(rows)),hi,upper)}}
    case Empty{+none}:
      Rows.Covers{row => lo => hi => Empty.absurd(Shape.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat(row)) == True{} : Bool},{Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool}>,
        WordEncoding.false_true(%Order.not_below(U32.to_nat(Copy.begin(query)),U32.to_nat(row),lo) : {_ == True{} : Bool}
          %Addition.zero_right(U32.to_nat(Copy.begin(query))) : {Nat.is_lt(U32.to_nat(row),_) == True{} : Bool}
          %none : {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(Copy.begin(query)),U32.to_nat(_))) == True{} : Bool}
          hi))}
