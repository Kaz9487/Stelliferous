# Readback of the phase table at its packed base, including depth 31 arrays.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../traversal_array.bend as Traversal
import ./input_gradient_tiled_refinement.bend as Execution
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./traversal_indexed_values.bend as Indexed
import ./traversal_array_refinement.bend as Arrays

def base(table: Kernel.Table) -> U32:
  Kernel.Table{tap_rows,tap_columns,row_step,shifts,first_shift,base} = table
  base

def value(table: Kernel.Table,index: U32) -> U32:
  Kernel.Table{+tap_rows,+tap_columns,+row_step,+shifts,+first_shift,base} = table
  +index = index
  (index / tap_columns % tap_rows * row_step + index / (tap_rows * tap_columns) * shifts + index % tap_columns + first_shift : U32)

type Query is Data:
  Query{table: Kernel.Table,count: Nat,index: Nat,depth: Nat,capacity: U32}

def table(query: Query) -> Kernel.Table:
  Query{table,count,index,depth,capacity} = query
  table
def count(query: Query) -> Nat:
  Query{table,count,index,depth,capacity} = query
  count
def index(query: Query) -> Nat:
  Query{table,count,index,depth,capacity} = query
  index
def depth(query: Query) -> Nat:
  Query{table,count,index,depth,capacity} = query
  depth
def capacity(query: Query) -> U32:
  Query{table,count,index,depth,capacity} = query
  capacity
def address(+query: Query) -> U32:
  (base(table(query)) + U32.from_nat(index(query)) : U32)
def wanted(query: Query) -> U32:
  Query{table,count,index,depth,capacity} = query
  value(table,U32.from_nat(index))
def read(query: Query,tree: Tree.Tree<U32>) -> U32:
  Tree.read(U32,tree,address(query))
def update(position: U32,query: Query,tree: Tree.Tree<U32>) -> Tree.Tree<U32>:
  Execution.offset(position,table(query),tree)

type Frame<-query: Query,-tree: Tree.Tree<U32>> is Data:
  Frame{capacity_equal: {Tree.capacity(U32,tree) == capacity(query) : U32},balanced: Shape.Balanced(U32,depth(query),tree),
    depth_ok: {Nat.is_le(depth(query),31n) == True{} : Bool},capacity_ok: {U32.is_le(capacity(query),2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(base(table(query))),count(query)),U32.to_nat(capacity(query))) == True{} : Bool},
    selected: {Nat.is_lt(index(query),count(query)) == True{} : Bool}}

def affine(query: Query) -> Indexed.AffineQuery<Kernel.Table>:
  Query{+table,count,index,depth,capacity} = query
  Indexed.AffineQuery{table,base(table),count,index,depth,capacity}

def affine_step_matches(+position: U32,+query: Query,+tree: Tree.Tree<U32>) ->
  {update(position,query,tree) == Indexed.affine_update(~U32,~Kernel.Table,~value,position,affine(query),tree) : Tree.Tree<U32>}:
  Query{+table,count,index,depth,capacity} = query
  Kernel.Table{tap_rows,tap_columns,row_step,shifts,first_shift,base} = table
  {==}

def same_step(+position: U32,+query: Query,+tree: Tree.Tree<U32>) ->
  {update(position,query,tree) == Execution.offset(position,table(query),tree) : Tree.Tree<U32>}:
  {==}

def readback(+query: Query,+tree: Tree.Tree<U32>,+evidence: Frame<query,tree>) ->
  {Tree.read(U32,Traversal.indexed_fold(~Tree.Tree<U32>,~Kernel.Table,~Execution.offset,count(query),0,table(query),tree),address(query)) == wanted(query) : U32}:
  Query{selected_table,count,index,depth,capacity} = query
  Frame{capacity_equal,balanced,depth_ok,capacity_ok,room,selected} = evidence
  +ready = {Indexed.AffineFrame{capacity_equal,balanced,depth_ok,capacity_ok,room,selected} : Indexed.AffineFrame<U32,Kernel.Table,affine(query),tree>}
  %Arrays.indexed_fold_change_context(~Tree.Tree<U32>,~Query,~Kernel.Table,~table,~update,~Execution.offset,~same_step,count,0,query,tree) :
    {read(query,_) == wanted(query) : U32}
  %Equal.sym(Tree.Tree<U32>,Traversal.indexed_fold(~Tree.Tree<U32>,~Query,~update,count,0,query,tree),
    Traversal.indexed_fold(~Tree.Tree<U32>,~Indexed.AffineQuery<Kernel.Table>,~Indexed.affine_update(~U32,~Kernel.Table,~value),count,0,affine(query),tree),
    Arrays.indexed_fold_change_context(~Tree.Tree<U32>,~Query,~Indexed.AffineQuery<Kernel.Table>,~affine,~update,~Indexed.affine_update(~U32,~Kernel.Table,~value),~affine_step_matches,count,0,query,tree)) :
    {read(query,_) == wanted(query) : U32}
  Indexed.affine_readback(~U32,~Kernel.Table,~value,affine(query),tree,ready)

def sampled(unit: Unit,table: Kernel.Table,index: U32) -> U32:
  value(table,index)
def placed(table: Kernel.Table,index: U32) -> U32:
  (base(table) + index : U32)
def placed_table(query: Indexed.Query<Unit,Kernel.Table>) -> Kernel.Table:
  Indexed.query_context(~Unit,~Kernel.Table,query)
def placed_same(+position: U32,+query: Indexed.Query<Unit,Kernel.Table>,+tree: Tree.Tree<U32>) ->
  {Indexed.placed_update(~Unit,~U32,~Kernel.Table,~sampled,~placed,position,query,tree) == Execution.offset(position,placed_table(query),tree) : Tree.Tree<U32>}:
  Indexed.Query{count,index,depth,capacity,Unit{},table} = query
  Kernel.Table{tap_rows,tap_columns,row_step,shifts,first_shift,base} = table
  {==}

def written_run(+count: Nat,+table: Kernel.Table,+tree: Tree.Tree<U32>) ->
  {Indexed.run(~Unit,~U32,~Kernel.Table,~sampled,~placed,count,0,table,Unit{},tree) ==
    Traversal.indexed_fold(~Tree.Tree<U32>,~Kernel.Table,~Execution.offset,count,0,table,tree) : Tree.Tree<U32>}:
  +query = {Indexed.Query{0n,0n,0n,0,Unit{},table} : Indexed.Query<Unit,Kernel.Table>}
  Equal.trans(Tree.Tree<U32>,Indexed.run(~Unit,~U32,~Kernel.Table,~sampled,~placed,count,0,table,Unit{},tree),
    Traversal.indexed_fold(~Tree.Tree<U32>,~Indexed.Query<Unit,Kernel.Table>,~Indexed.placed_update(~Unit,~U32,~Kernel.Table,~sampled,~placed),count,0,query,tree),
    Traversal.indexed_fold(~Tree.Tree<U32>,~Kernel.Table,~Execution.offset,count,0,table,tree),
    Indexed.placed_folded(~Unit,~U32,~Kernel.Table,~sampled,~placed,query,count,0,tree),
    Arrays.indexed_fold_change_context(~Tree.Tree<U32>,~Indexed.Query<Unit,Kernel.Table>,~Kernel.Table,~placed_table,
      ~Indexed.placed_update(~Unit,~U32,~Kernel.Table,~sampled,~placed),~Execution.offset,~placed_same,count,0,query,tree))

def capacity_kept(+count: Nat,+table: Kernel.Table,+tree: Tree.Tree<U32>) ->
  {Tree.capacity(U32,Traversal.indexed_fold(~Tree.Tree<U32>,~Kernel.Table,~Execution.offset,count,0,table,tree)) == Tree.capacity(U32,tree) : U32}:
  %written_run(count,table,tree) : {Tree.capacity(U32,_) == Tree.capacity(U32,tree) : U32}
  Indexed.capacity(~Unit,~U32,~Kernel.Table,~sampled,~placed,count,0,table,Unit{},tree)

def balanced(+count: Nat,+table: Kernel.Table,+tree: Tree.Tree<U32>,+depth: Nat,evidence: Shape.Balanced(U32,depth,tree)) ->
  Shape.Balanced(U32,depth,Traversal.indexed_fold(~Tree.Tree<U32>,~Kernel.Table,~Execution.offset,count,0,table,tree)):
  %written_run(count,table,tree) : Shape.Balanced(U32,depth,_)
  Indexed.balanced(~Unit,~U32,~Kernel.Table,~sampled,~placed,count,0,table,Unit{},tree,depth,evidence)

def read_before(+query: Query,+address: U32,+tree: Tree.Tree<U32>,+evidence: Frame<query,tree>,
  before: {Nat.is_lt(U32.to_nat(address),U32.to_nat(base(table(query)))) == True{} : Bool},
  inside: {U32.is_lt(address,capacity(query)) == True{} : Bool}) ->
  {Tree.read(U32,Traversal.indexed_fold(~Tree.Tree<U32>,~Kernel.Table,~Execution.offset,count(query),0,table(query),tree),address) == Tree.read(U32,tree,address) : U32}:
  Query{selected_table,count,index,depth,capacity} = query
  Frame{capacity_equal,balanced,depth_ok,capacity_ok,room,selected} = evidence
  +ready = {Indexed.AffineFrame{capacity_equal,balanced,depth_ok,capacity_ok,room,selected} : Indexed.AffineFrame<U32,Kernel.Table,affine(query),tree>}
  %Arrays.indexed_fold_change_context(~Tree.Tree<U32>,~Query,~Kernel.Table,~table,~update,~Execution.offset,~same_step,count,0,query,tree) :
    {Tree.read(U32,_,address) == Tree.read(U32,tree,address) : U32}
  %Equal.sym(Tree.Tree<U32>,Traversal.indexed_fold(~Tree.Tree<U32>,~Query,~update,count,0,query,tree),
    Traversal.indexed_fold(~Tree.Tree<U32>,~Indexed.AffineQuery<Kernel.Table>,~Indexed.affine_update(~U32,~Kernel.Table,~value),count,0,affine(query),tree),
    Arrays.indexed_fold_change_context(~Tree.Tree<U32>,~Query,~Indexed.AffineQuery<Kernel.Table>,~affine,~update,~Indexed.affine_update(~U32,~Kernel.Table,~value),~affine_step_matches,count,0,query,tree)) :
    {Tree.read(U32,_,address) == Tree.read(U32,tree,address) : U32}
  Indexed.affine_read_before(~U32,~Kernel.Table,~value,affine(query),address,tree,ready,before,inside)
