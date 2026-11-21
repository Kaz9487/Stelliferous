# Cell GEMM preserves equal ordered dots when it changes its row tiles.
import Base
import ./u32_extrema.bend as U32Extrema
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ./storage_tree_model.bend as Tree
import ./write_program.bend as Writes
import ./kernel_gemm_refinement.bend as Laws
import ./kernel_gemm_readback.bend as Readback
import ./kernel_gemm_jobs.bend as Jobs
import ./kernel_gemm_spec.bend as Spec
import ./kernel_gemm_congruence.bend as Congruence
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds

type Retiled<-Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data> is Data:
  Retiled{job: Congruence.TileRetiled<Link>,bias: Tree.Tree<Tile.Tile8>}

def job(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,+state: Retiled<Link>) -> Gemm.Job<Gemm.Direct()>:
  Retiled{current,bias} = state
  Congruence.tile_retiled_job(~Link,current)

def left(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,+state: Retiled<Link>) -> Jobs.CellReaders:
  Retiled{Congruence.TileRetiled{product,other,unit,group,link,unit_inside,group_inside},bias} = state
  Readback.cell_readers(Readback.CellProduct{product,bias})

def right(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,+state: Retiled<Link>) -> Jobs.CellReaders:
  Retiled{Congruence.TileRetiled{product,other,unit,group,link,unit_inside,group_inside},bias} = state
  Readback.cell_readers(Readback.CellProduct{Congruence.rerowed(product,other),bias})

def virtual_job(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,+state: Retiled<Link>) -> Gemm.Job<Gemm.Direct()>:
  Jobs.cell_virtual_job(job(~Link,state))

def virtual_left(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,+state: Retiled<Link>) -> Jobs.Readers:
  Jobs.cell_virtual_readers(job(~Link,state),left(~Link,state))

def virtual_right(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,+state: Retiled<Link>) -> Jobs.Readers:
  Jobs.cell_virtual_readers(job(~Link,state),right(~Link,state))

def rows_small(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,+state: Retiled<Link>) ->
  {Nat.is_le(Jobs.job_rows(~Gemm.Direct(),virtual_job(~Link,state)),8n) == True{} : Bool}:
  Retiled{Congruence.TileRetiled{+product,other,+unit,+group,link,unit_inside,group_inside},bias} = state
  U32Extrema.minimum_below(8,(Readback.product_output(product) - Readback.position_word(product,Readback.job_index(product,unit,group)) : U32))

def columns_small(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,+state: Retiled<Link>) ->
  {Nat.is_le(Jobs.job_columns(~Gemm.Direct(),Laws.Tiles{},virtual_job(~Link,state)),8n) == True{} : Bool}:
  {==}

def dot_row(+product: Readback.Product,+unit: Nat,+position: Nat) -> Tile.Tile8:
  +inner = Readback.product_inner(product)
  Spec.dot_row(Laws.RowFirst{},Spec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(inner),
    Gemm.Both{(U32.from_nat(unit) * inner : U32),(U32.from_nat(Nat.div(position,8n)) * inner : U32)}),Nat.mod(position,8n),
    Spec.Source{Readback.product_row_tiles(product),Readback.product_lane_tiles(product)})

def wanted_agree(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,~agree: Congruence.TileAgreement(~Link),
  +product: Readback.Product,+other: Tree.Tree<Tile.Tile8>,+link: Link(product,other),+bias: Tree.Tree<Tile.Tile8>,+unit: Nat,
  +unit_inside: {Nat.is_lt(unit,Readback.units(product)) == True{} : Bool},+position: Nat,
  +position_inside: {Nat.is_lt(Nat.add(Readback.start_column(product),position),Readback.output_width(product)) == True{} : Bool}) ->
  {Readback.cell_wanted(Readback.CellProduct{product,bias},Readback.TileSpot{unit,position}) ==
    Readback.cell_wanted(Readback.CellProduct{Congruence.rerowed(product,other),bias},Readback.TileSpot{unit,position}) : Tile.Tile8}:
  Readback.Product{+rows,+columns,+inner,+start,+output,+previous,+lane_tiles,+unused,+limit,+groups,+limit_ok,+columns_exact,+rows_room,+area_room,+end_room,+output_room} = product
  +product = {Readback.Product{rows,columns,inner,start,output,previous,lane_tiles,unused,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} : Readback.Product}
  +first = (U32.from_nat(unit) * 8 : U32)
  +lanes = U32.min(8,(rows - first : U32))
  Laws.finished_agree(~Laws.biased_value,~Gemm.plus,0,lanes,Laws.bias_tree(Tree.read(Tile.Tile8,bias,(first / 8 : U32))),
    dot_row(product,unit,position),dot_row(Congruence.rerowed(product,other),unit,position),
    inside => Congruence.tile_lane_agrees(~Link,~agree,product,other,link,unit,unit_inside,position,position_inside,0n,inside,{==}),
    inside => Congruence.tile_lane_agrees(~Link,~agree,product,other,link,unit,unit_inside,position,position_inside,1n,inside,{==}),
    inside => Congruence.tile_lane_agrees(~Link,~agree,product,other,link,unit,unit_inside,position,position_inside,2n,inside,{==}),
    inside => Congruence.tile_lane_agrees(~Link,~agree,product,other,link,unit,unit_inside,position,position_inside,3n,inside,{==}),
    inside => Congruence.tile_lane_agrees(~Link,~agree,product,other,link,unit,unit_inside,position,position_inside,4n,inside,{==}),
    inside => Congruence.tile_lane_agrees(~Link,~agree,product,other,link,unit,unit_inside,position,position_inside,5n,inside,{==}),
    inside => Congruence.tile_lane_agrees(~Link,~agree,product,other,link,unit,unit_inside,position,position_inside,6n,inside,{==}),
    inside => Congruence.tile_lane_agrees(~Link,~agree,product,other,link,unit,unit_inside,position,position_inside,7n,inside,{==}))

def cell_agree(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,~agree: Congruence.TileAgreement(~Link),
  +state: Retiled<Link>,+row: Nat,+column: Nat,+row_inside: {Nat.is_lt(row,Jobs.job_rows(~Gemm.Direct(),virtual_job(~Link,state))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,Jobs.job_columns(~Gemm.Direct(),Laws.Tiles{},virtual_job(~Link,state))) == True{} : Bool}) ->
  {Jobs.cell_value(~Gemm.Direct(),~Gemm.directs,~Laws.RowFirst{},~Laws.biased_value,~Gemm.plus,Laws.Tiles{},virtual_job(~Link,state),virtual_left(~Link,state),row,column) ==
    Jobs.cell_value(~Gemm.Direct(),~Gemm.directs,~Laws.RowFirst{},~Laws.biased_value,~Gemm.plus,Laws.Tiles{},virtual_job(~Link,state),virtual_right(~Link,state),row,column) : Tile.Tile8}:
  Retiled{Congruence.TileRetiled{+product,+other,+unit,+group,+link,+unit_inside,+group_inside},+bias} = state
  Readback.Product{+rows,+columns,+inner,+start,+output,+previous,+lane_tiles,+unused,+limit,+groups,+limit_ok,+columns_exact,+rows_room,+area_room,+end_room,+output_room} = product
  +product = {Readback.Product{rows,columns,inner,start,output,previous,lane_tiles,unused,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} : Readback.Product}
  +other_product = Congruence.rerowed(product,other)
  +placement = {Readback.Placement{product,unit,group,unit_inside,group_inside} : Readback.Placement}
  +other_placement = {Readback.Placement{other_product,unit,group,unit_inside,group_inside} : Readback.Placement}
  +current = Readback.tile_job(placement)
  +spot = {Readback.TileSpot{unit,Readback.cell_column(placement,row)} : Readback.TileSpot}
  +first = Readback.cell_readers(Readback.CellProduct{product,bias})
  +second = Readback.cell_readers(Readback.CellProduct{other_product,bias})
  %Equal.sym(Nat,column,0n,Jobs.only_cell(column,column_inside)) :
    {Jobs.cell_job_value(current,first,row,_) == Jobs.cell_job_value(current,second,row,_) : Tile.Tile8}
  Equal.trans(Tile.Tile8,Jobs.cell_job_value(current,first,row,0n),Readback.cell_wanted(Readback.CellProduct{product,bias},spot),Jobs.cell_job_value(current,second,row,0n),
    Readback.cell_tile_value(placement,bias,row,row_inside),
    Equal.trans(Tile.Tile8,Readback.cell_wanted(Readback.CellProduct{product,bias},spot),Readback.cell_wanted(Readback.CellProduct{other_product,bias},spot),Jobs.cell_job_value(current,second,row,0n),
      wanted_agree(~Link,~agree,product,other,link,bias,unit,unit_inside,Readback.cell_column(placement,row),Readback.column_inside(placement,row,row_inside)),
      Equal.sym(Tile.Tile8,Jobs.cell_job_value(current,second,row,0n),Readback.cell_wanted(Readback.CellProduct{other_product,bias},spot),Readback.cell_tile_value(other_placement,bias,row,row_inside))))

def job_agree(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,~agree: Congruence.TileAgreement(~Link),+state: Retiled<Link>) ->
  {Jobs.cell_job_writes(job(~Link,state),left(~Link,state)) == Jobs.cell_job_writes(job(~Link,state),right(~Link,state)) : List<&2,Writes.Assignment<Tile.Tile8>>}:
  +current = job(~Link,state)
  +first = left(~Link,state)
  +second = right(~Link,state)
  Equal.trans(List<&2,Writes.Assignment<Tile.Tile8>>,Jobs.cell_job_writes(current,first),
    Jobs.job_writes(~Gemm.Direct(),~Gemm.directs,~Laws.RowFirst{},~Laws.Tiles{},~Laws.biased_value,~Gemm.plus,virtual_job(~Link,state),virtual_left(~Link,state)),Jobs.cell_job_writes(current,second),
    Jobs.cell_job_as_virtual(current,first),
    Equal.trans(List<&2,Writes.Assignment<Tile.Tile8>>,
      Jobs.job_writes(~Gemm.Direct(),~Gemm.directs,~Laws.RowFirst{},~Laws.Tiles{},~Laws.biased_value,~Gemm.plus,virtual_job(~Link,state),virtual_left(~Link,state)),
      Jobs.job_writes(~Gemm.Direct(),~Gemm.directs,~Laws.RowFirst{},~Laws.Tiles{},~Laws.biased_value,~Gemm.plus,virtual_job(~Link,state),virtual_right(~Link,state)),Jobs.cell_job_writes(current,second),
      Jobs.job_agree(~Gemm.Direct(),~Gemm.directs,~Laws.RowFirst{},~Laws.Tiles{},~Laws.biased_value,~Gemm.plus,~Retiled<Link>,~virtual_job(~Link),~virtual_left(~Link),~virtual_right(~Link),
        ~rows_small(~Link),~columns_small(~Link),~cell_agree(~Link,~agree),state),
      Equal.sym(List<&2,Writes.Assignment<Tile.Tile8>>,Jobs.cell_job_writes(current,second),
        Jobs.job_writes(~Gemm.Direct(),~Gemm.directs,~Laws.RowFirst{},~Laws.Tiles{},~Laws.biased_value,~Gemm.plus,virtual_job(~Link,state),virtual_right(~Link,state)),Jobs.cell_job_as_virtual(current,second))))

def numbered_agree(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,~agree: Congruence.TileAgreement(~Link),
  +product: Readback.Product,+other: Tree.Tree<Tile.Tile8>,+link: Link(product,other),+bias: Tree.Tree<Tile.Tile8>,+index: Nat,+number: Readback.Numbered<product,index>) ->
  {Jobs.cell_job_writes(Gemm.tile_major(Readback.tile_matrix(product),U32.from_nat(index)),Readback.cell_readers(Readback.CellProduct{product,bias})) ==
    Jobs.cell_job_writes(Gemm.tile_major(Readback.tile_matrix(product),U32.from_nat(index)),Readback.cell_readers(Readback.CellProduct{Congruence.rerowed(product,other),bias})) : List<&2,Writes.Assignment<Tile.Tile8>>}:
  Readback.Numbered{+unit,+group,+unit_inside,+group_inside,+exact} = number
  %Equal.sym(Nat,index,Readback.index_number(product,unit,group),exact) :
    {Jobs.cell_job_writes(Gemm.tile_major(Readback.tile_matrix(product),U32.from_nat(_)),Readback.cell_readers(Readback.CellProduct{product,bias})) ==
      Jobs.cell_job_writes(Gemm.tile_major(Readback.tile_matrix(product),U32.from_nat(_)),Readback.cell_readers(Readback.CellProduct{Congruence.rerowed(product,other),bias})) : List<&2,Writes.Assignment<Tile.Tile8>>}
  job_agree(~Link,~agree,Retiled{Congruence.TileRetiled{product,other,unit,group,link,unit_inside,group_inside},bias})

def all_agree(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,~agree: Congruence.TileAgreement(~Link),
  +product: Readback.Product,+other: Tree.Tree<Tile.Tile8>,+link: Link(product,other),+bias: Tree.Tree<Tile.Tile8>,count: Nat,+first: Nat,
  +room: {Nat.add(first,count) == Readback.jobs(product) : Nat}) ->
  {Jobs.cell_all_writes(count,U32.from_nat(first),Readback.tile_matrix(product),Readback.cell_readers(Readback.CellProduct{product,bias})) ==
    Jobs.cell_all_writes(count,U32.from_nat(first),Readback.tile_matrix(product),Readback.cell_readers(Readback.CellProduct{Congruence.rerowed(product,other),bias})) : List<&2,Writes.Assignment<Tile.Tile8>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +matrix = Readback.tile_matrix(product)
      +left = Readback.cell_readers(Readback.CellProduct{product,bias})
      +right = Readback.cell_readers(Readback.CellProduct{Congruence.rerowed(product,other),bias})
      +left_here = Jobs.cell_job_writes(Gemm.tile_major(matrix,U32.from_nat(first)),left)
      +right_here = Jobs.cell_job_writes(Gemm.tile_major(matrix,U32.from_nat(first)),right)
      +left_later = Jobs.cell_all_writes(rest,U32.from_nat(1n+first),matrix,left)
      +right_later = Jobs.cell_all_writes(rest,U32.from_nat(1n+first),matrix,right)
      +moved = Equal.trans(Nat,1n+Nat.add(first,rest),Nat.add(first,1n+rest),Readback.jobs(product),Equal.sym(Nat,Nat.add(first,1n+rest),1n+Nat.add(first,rest),Addition.successor_right(first,rest)),room)
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        {List.append(&2,Writes.Assignment<Tile.Tile8>,left_here,Jobs.cell_all_writes(rest,_,matrix,left)) ==
          List.append(&2,Writes.Assignment<Tile.Tile8>,right_here,Jobs.cell_all_writes(rest,_,matrix,right)) : List<&2,Writes.Assignment<Tile.Tile8>>}
      Equal.trans(List<&2,Writes.Assignment<Tile.Tile8>>,List.append(&2,Writes.Assignment<Tile.Tile8>,left_here,left_later),List.append(&2,Writes.Assignment<Tile.Tile8>,right_here,left_later),List.append(&2,Writes.Assignment<Tile.Tile8>,right_here,right_later),
        Equal.cong(List<&2,Writes.Assignment<Tile.Tile8>>,List<&2,Writes.Assignment<Tile.Tile8>>,writes => List.append(&2,Writes.Assignment<Tile.Tile8>,writes,left_later),left_here,right_here,
          numbered_agree(~Link,~agree,product,other,link,bias,first,Readback.numbered(product,first,Order.below_complement(first,rest,Readback.jobs(product),room)))),
        Equal.cong(List<&2,Writes.Assignment<Tile.Tile8>>,List<&2,Writes.Assignment<Tile.Tile8>>,writes => List.append(&2,Writes.Assignment<Tile.Tile8>,right_here,writes),left_later,right_later,
          all_agree(~Link,~agree,product,other,link,bias,rest,1n+first,moved)))

def product_agree(~Link: Readback.Product -> Tree.Tree<Tile.Tile8> -> Data,~agree: Congruence.TileAgreement(~Link),
  +product: Readback.Product,+other: Tree.Tree<Tile.Tile8>,+link: Link(product,other),+bias: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) ->
  {Readback.cell_product_tree(Readback.CellProduct{product,bias},tree) ==
    Readback.cell_tree(Readback.tile_matrix(product),other,Readback.product_lane_tiles(product),bias,tree) : Tree.Tree<Tile.Tile8>}:
  Readback.Product{+rows,+columns,+inner,+start,+output,+previous,+lane_tiles,+unused,+limit,+groups,+limit_ok,+columns_exact,+rows_room,+area_room,+end_room,+output_room} = product
  +product = {Readback.Product{rows,columns,inner,start,output,previous,lane_tiles,unused,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} : Readback.Product}
  +matrix = Readback.tile_matrix(product)
  +count = U32.to_nat(Gemm.tile_jobs(matrix))
  +left = Jobs.cell_all_writes(count,0,matrix,Readback.cell_readers(Readback.CellProduct{product,bias}))
  +right = Jobs.cell_all_writes(count,0,matrix,Readback.cell_readers(Readback.CellProduct{Congruence.rerowed(product,other),bias}))
  +wanted = Readback.cell_tree(matrix,other,lane_tiles,bias,tree)
  Equal.trans(Tree.Tree<Tile.Tile8>,Readback.cell_product_tree(Readback.CellProduct{product,bias},tree),Writes.pure(Tile.Tile8,left,tree),wanted,
    Readback.cell_tree_writes(matrix,previous,lane_tiles,bias,tree),
    Equal.trans(Tree.Tree<Tile.Tile8>,Writes.pure(Tile.Tile8,left,tree),Writes.pure(Tile.Tile8,right,tree),wanted,
      Equal.cong(List<&2,Writes.Assignment<Tile.Tile8>>,Tree.Tree<Tile.Tile8>,writes => Writes.pure(Tile.Tile8,writes,tree),left,right,
        all_agree(~Link,~agree,product,other,link,bias,count,0n,Readback.jobs_exact(product))),
      Equal.sym(Tree.Tree<Tile.Tile8>,wanted,Writes.pure(Tile.Tile8,right,tree),Readback.cell_tree_writes(matrix,other,lane_tiles,bias,tree))))
