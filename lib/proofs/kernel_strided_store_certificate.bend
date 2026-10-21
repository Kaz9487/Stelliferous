# A block store keeps the storage shape, whatever values it writes. Only the
# final flat certificate is live; the witness follows the writes beneath an
# erased rewrite.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_rows.bend as Rows
import ../kernel_gemm.bend as Gemm
import ../kernel_strided_store.bend as Stores
import ./storage_witness.bend as Witness
import ./storage_certificate.bend as Certificate
import ./storage_tree_model.bend as Tree
import ./kernel_gemm_certificate.bend as GemmCertificate

def row(+depth: Nat,valid: Bool,-output: Array<F32>,+target: U32,+step: U32,+count: U32,-accumulator: Tile.Tile8,
  evidence: Witness.Witness(F32,depth,output)) -> Witness.Witness(F32,depth,Stores.row(valid,output,target,step,count,accumulator)):
  match valid:
    case False{}: evidence
    case True{}:
      -o1 = Gemm.store_lane(U32.is_lt(0,count),output,target,(Tile.lane0(accumulator) + 0.0 : F32))
      -o2 = Gemm.store_lane(U32.is_lt(1,count),o1,(target + step : U32),(Tile.lane1(accumulator) + 0.0 : F32))
      -o3 = Gemm.store_lane(U32.is_lt(2,count),o2,(target + 2 * step : U32),(Tile.lane2(accumulator) + 0.0 : F32))
      -o4 = Gemm.store_lane(U32.is_lt(3,count),o3,(target + 3 * step : U32),(Tile.lane3(accumulator) + 0.0 : F32))
      -o5 = Gemm.store_lane(U32.is_lt(4,count),o4,(target + 4 * step : U32),(Tile.lane4(accumulator) + 0.0 : F32))
      -o6 = Gemm.store_lane(U32.is_lt(5,count),o5,(target + 5 * step : U32),(Tile.lane5(accumulator) + 0.0 : F32))
      -o7 = Gemm.store_lane(U32.is_lt(6,count),o6,(target + 6 * step : U32),(Tile.lane6(accumulator) + 0.0 : F32))
      evidence = GemmCertificate.store_lane(depth,U32.is_lt(0,count),output,target,(Tile.lane0(accumulator) + 0.0 : F32),evidence)
      evidence = GemmCertificate.store_lane(depth,U32.is_lt(1,count),o1,(target + step : U32),(Tile.lane1(accumulator) + 0.0 : F32),evidence)
      evidence = GemmCertificate.store_lane(depth,U32.is_lt(2,count),o2,(target + 2 * step : U32),(Tile.lane2(accumulator) + 0.0 : F32),evidence)
      evidence = GemmCertificate.store_lane(depth,U32.is_lt(3,count),o3,(target + 3 * step : U32),(Tile.lane3(accumulator) + 0.0 : F32),evidence)
      evidence = GemmCertificate.store_lane(depth,U32.is_lt(4,count),o4,(target + 4 * step : U32),(Tile.lane4(accumulator) + 0.0 : F32),evidence)
      evidence = GemmCertificate.store_lane(depth,U32.is_lt(5,count),o5,(target + 5 * step : U32),(Tile.lane5(accumulator) + 0.0 : F32),evidence)
      evidence = GemmCertificate.store_lane(depth,U32.is_lt(6,count),o6,(target + 6 * step : U32),(Tile.lane6(accumulator) + 0.0 : F32),evidence)
      GemmCertificate.store_lane(depth,U32.is_lt(7,count),o7,(target + 7 * step : U32),(Tile.lane7(accumulator) + 0.0 : F32),evidence)

def block(+depth: Nat,+store: Stores.Store,-accumulators: Rows.Rows<Tile.Tile8>,-output: Array<F32>,
  evidence: Witness.Witness(F32,depth,output)) -> Witness.Witness(F32,depth,Stores.block(store,accumulators,output)):
  Stores.Store{+target,+step,+plane,+rows,+count} = store
  -o1 = Stores.row(U32.is_lt(0,rows),output,target,step,count,Rows.first(~Tile.Tile8,accumulators))
  -o2 = Stores.row(U32.is_lt(1,rows),o1,(target + plane : U32),step,count,Rows.second(~Tile.Tile8,accumulators))
  -o3 = Stores.row(U32.is_lt(2,rows),o2,(target + 2 * plane : U32),step,count,Rows.third(~Tile.Tile8,accumulators))
  evidence = row(depth,U32.is_lt(0,rows),output,target,step,count,Rows.first(~Tile.Tile8,accumulators),evidence)
  evidence = row(depth,U32.is_lt(1,rows),o1,(target + plane : U32),step,count,Rows.second(~Tile.Tile8,accumulators),evidence)
  evidence = row(depth,U32.is_lt(2,rows),o2,(target + 2 * plane : U32),step,count,Rows.third(~Tile.Tile8,accumulators),evidence)
  row(depth,U32.is_lt(3,rows),o3,(target + 3 * plane : U32),step,count,Rows.fourth(~Tile.Tile8,accumulators),evidence)

def equation(+depth: Nat,+store: Stores.Store,-accumulators: Rows.Rows<Tile.Tile8>,-output: Array<F32>,
  balanced: {output == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,output))) : Array<F32>}) ->
  {Stores.block(store,accumulators,output) == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,Stores.block(store,accumulators,output)))) : Array<F32>}:
  %Certificate.restored(F32,depth,Stores.block(store,accumulators,output),
    block(depth,store,accumulators,output,Certificate.witness_at(F32,depth,output,balanced))) :
    {_ == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,Stores.block(store,accumulators,output)))) : Array<F32>}
  {==}

def certificate(+store: Stores.Store,-accumulators: Rows.Rows<Tile.Tile8>,-output: Array<F32>,certificate: Certificate.Certificate<F32,output>) ->
  Certificate.Certificate<F32,Stores.block(store,accumulators,output)>:
  Certificate.Certificate{+depth,balanced} = certificate
  Certificate.Certificate{depth,equation(depth,store,accumulators,output,balanced)}
