# Caller-provided gradient rows: their range, coverage, and ordinary cell contents.
import Base
import ./input_gradient_tiled_source_rows.bend as Rows
import ./input_gradient_tiled_source_contents.bend as Contents
import ./input_gradient_tiled_source_coverage.bend as Coverage
import ./input_gradient_tiled_copy_geometry.bend as Copy
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_phase_domain.bend as Domain
import ./storage_buffer_model.bend as Buffer

type Witness<-band: Bands.Query,-source: Execution.Source,-original: Buffer.Model<F32>,-base: U32,-plane: U32,-outputs: U32> is Data:
  Full{same: {source == Rows.full_source(band,original,base,plane) : Execution.Source}}
  Retained{range: Rows.Range<Execution.source_first(source),Execution.source_rows(source),Domain.gradient_rows(Bands.geometry(band))>,
    coverage: Coverage.Contains<Copy.begin(band),Copy.row_count(band),Execution.source_first(source),Execution.source_rows(source),Domain.gradient_rows(Bands.geometry(band))>,
    contents: Contents.Content(Rows.query_slice(band,source,original,base,plane),outputs)}
def range(+band: Bands.Query,+source: Execution.Source,+original: Buffer.Model<F32>,+base: U32,+plane: U32,+outputs: U32,
  witness: Witness<band,source,original,base,plane,outputs>) -> Rows.Range<Execution.source_first(source),Execution.source_rows(source),Domain.gradient_rows(Bands.geometry(band))>:
  match witness:
    case Full{+same}:
      %Equal.sym(Execution.Source,source,Rows.full_source(band,original,base,plane),same) : Rows.Range<Execution.source_first(_),Execution.source_rows(_),Domain.gradient_rows(Bands.geometry(band))>
      Rows.full_range(band)
    case Retained{range,coverage,contents}: range
def cover(+band: Bands.Query,+source: Execution.Source,+original: Buffer.Model<F32>,+base: U32,+plane: U32,+outputs: U32,
  witness: Witness<band,source,original,base,plane,outputs>) -> Rows.Covers<band,Execution.source_first(source),Execution.source_rows(source)>:
  match witness:
    case Full{+same}:
      %Equal.sym(Execution.Source,source,Rows.full_source(band,original,base,plane),same) : Rows.Covers<band,Execution.source_first(_),Execution.source_rows(_)>
      Rows.full_cover(band)
    case Retained{range,coverage,contents}: Coverage.cover(band,Execution.source_first(source),Execution.source_rows(source),coverage)
def stored(+band: Bands.Query,+source: Execution.Source,+original: Buffer.Model<F32>,+base: U32,+plane: U32,+outputs: U32,
  witness: Witness<band,source,original,base,plane,outputs>) -> Rows.Stored<Rows.query_slice(band,source,original,base,plane),outputs>:
  match witness:
    case Full{+same}:
      %Equal.sym(Execution.Source,source,Rows.full_source(band,original,base,plane),same) : Rows.Stored<Rows.query_slice(band,_,original,base,plane),outputs>
      Rows.full_stored(band,original,base,plane,outputs)
    case Retained{range,coverage,contents}: Contents.stored(Rows.query_slice(band,source,original,base,plane),outputs,contents)
