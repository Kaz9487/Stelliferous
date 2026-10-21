# Four rows of eight accumulator lanes stored at strided addresses. Each value
# is the fold of its products from +0, then +0.0.
import Base
import ./kernel_tile.bend as Tile
import ./kernel_rows.bend as Rows
import ./kernel_gemm.bend as Gemm

# Row r, lane i goes to target + r * plane + i * step, for r < rows and i < count.
type Store is Data:
  Store{target: U32,step: U32,plane: U32,rows: U32,count: U32}

def row(valid: Bool,output: Array<F32>,+target: U32,+step: U32,+count: U32,+accumulator: Tile.Tile8) -> Array<F32>:
  match valid:
    case False{}: output
    case True{}:
      output = Gemm.store_lane(U32.is_lt(0,count),output,target,(Tile.lane0(accumulator) + 0.0 : F32))
      output = Gemm.store_lane(U32.is_lt(1,count),output,(target + step : U32),(Tile.lane1(accumulator) + 0.0 : F32))
      output = Gemm.store_lane(U32.is_lt(2,count),output,(target + 2 * step : U32),(Tile.lane2(accumulator) + 0.0 : F32))
      output = Gemm.store_lane(U32.is_lt(3,count),output,(target + 3 * step : U32),(Tile.lane3(accumulator) + 0.0 : F32))
      output = Gemm.store_lane(U32.is_lt(4,count),output,(target + 4 * step : U32),(Tile.lane4(accumulator) + 0.0 : F32))
      output = Gemm.store_lane(U32.is_lt(5,count),output,(target + 5 * step : U32),(Tile.lane5(accumulator) + 0.0 : F32))
      output = Gemm.store_lane(U32.is_lt(6,count),output,(target + 6 * step : U32),(Tile.lane6(accumulator) + 0.0 : F32))
      Gemm.store_lane(U32.is_lt(7,count),output,(target + 7 * step : U32),(Tile.lane7(accumulator) + 0.0 : F32))

def block(+store: Store,+accumulators: Rows.Rows<Tile.Tile8>,output: Array<F32>) -> Array<F32>:
  Store{+target,+step,+plane,+rows,+count} = store
  output = row(U32.is_lt(0,rows),output,target,step,count,Rows.first(~Tile.Tile8,accumulators))
  output = row(U32.is_lt(1,rows),output,(target + plane : U32),step,count,Rows.second(~Tile.Tile8,accumulators))
  output = row(U32.is_lt(2,rows),output,(target + 2 * plane : U32),step,count,Rows.third(~Tile.Tile8,accumulators))
  row(U32.is_lt(3,rows),output,(target + 3 * plane : U32),step,count,Rows.fourth(~Tile.Tile8,accumulators))
