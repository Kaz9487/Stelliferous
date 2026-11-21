# Two tile products that differ only in their row tiles store the same tree
# when every stored channel has the same ordered dot over both row trees.
# Link names the caller's relation between the product and the other row
# tiles.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ./storage_tree_model.bend as Tree
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_spec.bend as GemmSpec
import ./kernel_gemm_readback.bend as GemmReadback
import ./kernel_tile_refinement.bend as TileLaws
import ./binary_radix.bend as Radix

def TileAgreement(~Link: GemmReadback.Product -> Tree.Tree<Tile.Tile8> -> Data) -> Type:
  @+product: GemmReadback.Product -> @+other: Tree.Tree<Tile.Tile8> -> @+channel: Nat -> @+column: Nat -> @+link: Link(product,other) ->
  @+channel_inside: {Nat.is_lt(channel,GemmReadback.row_count(product)) == True{} : Bool} ->
  @+column_inside: {Nat.is_lt(Nat.add(GemmReadback.start_column(product),column),GemmReadback.output_width(product)) == True{} : Bool} ->
  {GemmReadback.row_dot(product,GemmReadback.product_row_tiles(product),channel,column) == GemmReadback.row_dot(product,other,channel,column) : F32}

# The product over the other row tiles.
def rerowed(+product: GemmReadback.Product,+other: Tree.Tree<Tile.Tile8>) -> GemmReadback.Product:
  GemmReadback.Product{rows,columns,inner,start,output,previous,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  GemmReadback.Product{rows,columns,inner,start,output,other,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room}

# One tile job of a product with the other row tiles.
type TileRetiled<-Link: GemmReadback.Product -> Tree.Tree<Tile.Tile8> -> Data> is Data:
  TileRetiled{product: GemmReadback.Product,other: Tree.Tree<Tile.Tile8>,unit: Nat,group: Nat,link: Link(product,other),
    unit_inside: {Nat.is_lt(unit,GemmReadback.units(product)) == True{} : Bool},group_inside: {Nat.is_lt(group,GemmReadback.product_groups(product)) == True{} : Bool}}

def tile_placement(~Link: GemmReadback.Product -> Tree.Tree<Tile.Tile8> -> Data,+job: TileRetiled<Link>) -> GemmReadback.Placement:
  TileRetiled{product,other,unit,group,link,unit_inside,group_inside} = job
  GemmReadback.Placement{product,unit,group,unit_inside,group_inside}

def tile_retiled_job(~Link: GemmReadback.Product -> Tree.Tree<Tile.Tile8> -> Data,+job: TileRetiled<Link>) -> Gemm.Job<Gemm.Direct()>:
  GemmReadback.tile_job(tile_placement(~Link,job))

# Lane lane of a stored tile has the same dot over both row trees.
def tile_lane_agrees(~Link: GemmReadback.Product -> Tree.Tree<Tile.Tile8> -> Data,~agree: TileAgreement(~Link),
  +product: GemmReadback.Product,+other: Tree.Tree<Tile.Tile8>,+link: Link(product,other),+unit: Nat,+unit_inside: {Nat.is_lt(unit,GemmReadback.units(product)) == True{} : Bool},
  +position: Nat,+position_inside: {Nat.is_lt(Nat.add(GemmReadback.start_column(product),position),GemmReadback.output_width(product)) == True{} : Bool},
  +lane: Nat,+inside: {U32.is_lt(U32.from_nat(lane),U32.min(8,(GemmReadback.product_rows(product) - (U32.from_nat(unit) * 8 : U32) : U32))) == True{} : Bool},
  +small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,GemmSpec.dot_row(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(GemmReadback.product_inner(product)),
      Gemm.Both{(U32.from_nat(unit) * GemmReadback.product_inner(product) : U32),(U32.from_nat(Nat.div(position,8n)) * GemmReadback.product_inner(product) : U32)}),
      Nat.mod(position,8n),GemmSpec.Source{GemmReadback.product_row_tiles(product),GemmReadback.product_lane_tiles(product)})) ==
    TileLaws.select_lane(lane,GemmSpec.dot_row(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(GemmReadback.product_inner(product)),
      Gemm.Both{(U32.from_nat(unit) * GemmReadback.product_inner(product) : U32),(U32.from_nat(Nat.div(position,8n)) * GemmReadback.product_inner(product) : U32)}),
      Nat.mod(position,8n),GemmSpec.Source{other,GemmReadback.product_lane_tiles(product)})) : F32}:
  +inner = GemmReadback.product_inner(product)
  +channel = Nat.add(Nat.mul(unit,8n),lane)
  +row = Nat.mod(position,8n)
  +taps = GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(inner),Gemm.Both{(U32.from_nat(unit) * inner : U32),(U32.from_nat(Nat.div(position,8n)) * inner : U32)})
  +left = {GemmSpec.Source{GemmReadback.product_row_tiles(product),GemmReadback.product_lane_tiles(product)} : GemmSpec.Source}
  +right = {GemmSpec.Source{other,GemmReadback.product_lane_tiles(product)} : GemmSpec.Source}
  +agreed = {
    %Radix.quotient(unit,lane,8n,{==},small) :
      {GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(inner),Gemm.Both{(U32.from_nat(_) * inner : U32),(U32.from_nat(Nat.div(position,8n)) * inner : U32)}),row,lane,left,0.0) ==
        GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(inner),Gemm.Both{(U32.from_nat(_) * inner : U32),(U32.from_nat(Nat.div(position,8n)) * inner : U32)}),row,lane,right,0.0) : F32}
    %Radix.remainder(unit,lane,8n,{==},small) :
      {GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(inner),Gemm.Both{(U32.from_nat(Nat.div(channel,8n)) * inner : U32),(U32.from_nat(Nat.div(position,8n)) * inner : U32)}),row,_,left,0.0) ==
        GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(inner),Gemm.Both{(U32.from_nat(Nat.div(channel,8n)) * inner : U32),(U32.from_nat(Nat.div(position,8n)) * inner : U32)}),row,_,right,0.0) : F32}
    agree(product,other,channel,position,link,GemmReadback.lane_channel(product,unit,unit_inside,lane,small,inside),position_inside) :
    {GemmSpec.dot(GemmLaws.RowFirst{},taps,row,lane,left,0.0) == GemmSpec.dot(GemmLaws.RowFirst{},taps,row,lane,right,0.0) : F32}}
  Equal.trans(F32,TileLaws.select_lane(lane,GemmSpec.dot_row(GemmLaws.RowFirst{},taps,row,left)),GemmSpec.dot(GemmLaws.RowFirst{},taps,row,lane,left,0.0),
    TileLaws.select_lane(lane,GemmSpec.dot_row(GemmLaws.RowFirst{},taps,row,right)),
    GemmSpec.dot_lane(GemmLaws.RowFirst{},taps,row,lane,left,small),
    Equal.trans(F32,GemmSpec.dot(GemmLaws.RowFirst{},taps,row,lane,left,0.0),GemmSpec.dot(GemmLaws.RowFirst{},taps,row,lane,right,0.0),
      TileLaws.select_lane(lane,GemmSpec.dot_row(GemmLaws.RowFirst{},taps,row,right)),agreed,
      Equal.sym(F32,TileLaws.select_lane(lane,GemmSpec.dot_row(GemmLaws.RowFirst{},taps,row,right)),GemmSpec.dot(GemmLaws.RowFirst{},taps,row,lane,right,0.0),
        GemmSpec.dot_lane(GemmLaws.RowFirst{},taps,row,lane,right,small))))

