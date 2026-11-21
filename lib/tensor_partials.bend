# The order of a sum across rows. A row is a line of the axis before the
# trailing kept axes, and its partial is the fold of its values. The cells of
# the trailing axes interleave in the rows: every index of them has rows of
# its own, as if those axes came before the axis of the rows. Row i begins
# from context i modulo the count of contexts. The partials of
# the rows an output adds merge in one tree over the reduced axes, outermost
# first: over the indices of an axis a left-complete binary tree whose left
# side holds the largest power of two below the count, each of its leaves the
# tree over the next axis, the last axis merging the partials. The tree
# depends on the shape only, never on the pieces, the tasks or the workers.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_view.bend as Views
import ./tensor_pieces.bend as Pieces
import ./tensor_grid.bend as Grid
import ./tensor.bend as Tensors
import ./tensor_line_chain.bend as LineChain
import ./tensor_layout.bend as Layout
import ./tensor_shape.bend as Shapes
import ./traversal_partition.bend as Partition

# The tree runs as one pass per axis of the box, innermost first, on runs of
# partials. A run holds one partial for every index of the kept axes inside
# the axis of the pass, in their row-major order, and the passes start from a
# run for every partial. The pass of a kept axis joins the runs of each of its
# lines in order. The pass of a reduced axis merges the runs of each line in
# its tree, a run at a time at every position. The tree of a line is a binary
# counter: the run of index i merges with the latest of the counter as many
# times as i has trailing ones, the latest after the earlier, and the line
# ends merging the counter from the latest down.

# Two runs merged at every position, left after right.
def merged_runs(~Partial: Data,~merge: Partial -> Partial -> Partial,left: List<&2,Partial>,right: List<&2,Partial>) -> List<&2,Partial>:
  match left right:
    case Con{low,lows} Con{high,highs}: merge(low,high) <> merged_runs(~Partial,~merge,lows,highs)
    case left right: []

def ones_below(remaining: Nat,odd: Bool,+index: U32) -> Nat:
  match remaining odd:
    case 1n+more True{}: 1n+ones_below(more,U32.is_eq(U32.and(U32.shr(index),1),1),U32.shr(index))
    case remaining odd: 0n

# The trailing ones of index.
def trailing_ones(+index: U32) -> Nat:
  ones_below(32n,U32.is_eq(U32.and(index,1),1),index)

def counted(~Partial: Data,~merge: Partial -> Partial -> Partial,counter: List<&2,List<&2,Partial>>,merges: Nat,run: List<&2,Partial>) -> List<&2,List<&2,Partial>>:
  match counter merges:
    case Con{latest,earlier} 1n+more: counted(~Partial,~merge,earlier,more,merged_runs(~Partial,~merge,latest,run))
    case counter merges: run <> counter

def counter_merged(~Partial: Data,~merge: Partial -> Partial -> Partial,counter: List<&2,List<&2,Partial>>,run: List<&2,Partial>) -> List<&2,Partial>:
  match counter:
    case Nil{}: run
    case Con{earlier,rest}: counter_merged(~Partial,~merge,rest,merged_runs(~Partial,~merge,earlier,run))

def counter_total(~Partial: Data,~merge: Partial -> Partial -> Partial,counter: List<&2,List<&2,Partial>>) -> List<&2,Partial>:
  match counter:
    case Nil{}: []
    case Con{latest,earlier}: counter_merged(~Partial,~merge,earlier,latest)

# The runs of the lines of a reduced axis of extent indices, last first after
# done; last says whether the run in front ends its line.
def reduced_pass(~Partial: Data,~merge: Partial -> Partial -> Partial,runs: List<&2,List<&2,Partial>>,last: Bool,+extent: U32,+index: U32,counter: List<&2,List<&2,Partial>>,
  done: List<&2,List<&2,Partial>>) -> List<&2,List<&2,Partial>>:
  match runs last:
    case Nil{} last: done
    case Con{run,rest} True{}:
      reduced_pass(~Partial,~merge,rest,U32.is_eq(extent,1),extent,0,[],counter_total(~Partial,~merge,counted(~Partial,~merge,counter,trailing_ones(index),run)) <> done)
    case Con{run,rest} False{}:
      reduced_pass(~Partial,~merge,rest,U32.is_eq((index + 2 : U32),extent),extent,(index + 1 : U32),counted(~Partial,~merge,counter,trailing_ones(index),run),done)

def reversed_onto(~Partial: Data,run: List<&2,Partial>,joined: List<&2,Partial>) -> List<&2,Partial>:
  match run:
    case Nil{}: joined
    case Con{partial,rest}: reversed_onto(~Partial,rest,partial <> joined)

# The runs of the lines of a kept axis joined, last first after done; joined
# holds the line so far reversed.
def kept_pass(~Partial: Data,runs: List<&2,List<&2,Partial>>,last: Bool,+extent: U32,+index: U32,joined: List<&2,Partial>,done: List<&2,List<&2,Partial>>) -> List<&2,List<&2,Partial>>:
  match runs last:
    case Nil{} last: done
    case Con{run,rest} True{}: kept_pass(~Partial,rest,U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,reversed_onto(~Partial,run,joined)) <> done)
    case Con{run,rest} False{}: kept_pass(~Partial,rest,U32.is_eq((index + 2 : U32),extent),extent,(index + 1 : U32),reversed_onto(~Partial,run,joined),done)

def axis_pass(~Partial: Data,~merge: Partial -> Partial -> Partial,reduced: Bool,+extent: U32,runs: List<&2,List<&2,Partial>>) -> List<&2,List<&2,Partial>>:
  match reduced:
    case True{}: List.reverse(&2,List<&2,Partial>,reduced_pass(~Partial,~merge,runs,U32.is_eq(extent,1),extent,0,[],[]))
    case False{}: List.reverse(&2,List<&2,Partial>,kept_pass(~Partial,runs,U32.is_eq(extent,1),extent,0,[],[]))

# The passes of the axes, innermost first: extents and flags reversed.
def passes(~Partial: Data,~merge: Partial -> Partial -> Partial,extents: List<&2,U32>,flags: List<&2,Bool>,runs: List<&2,List<&2,Partial>>) -> List<&2,List<&2,Partial>>:
  match extents flags:
    case Con{+extent,rest} Con{flag,others}: passes(~Partial,~merge,rest,others,axis_pass(~Partial,~merge,flag,extent,runs))
    case extents flags: runs

def singles(~Partial: Data,partials: List<&2,Partial>) -> List<&2,List<&2,Partial>>:
  match partials:
    case Nil{}: []
    case Con{partial,rest}: [partial] <> singles(~Partial,rest)

def only_run(~Partial: Data,runs: List<&2,List<&2,Partial>>) -> List<&2,Partial>:
  match runs:
    case Con{run,rest}: run
    case Nil{}: []

# ---- Reductions in this order ----

def flag_removed(flags: List<&2,Bool>,axis: Nat) -> List<&2,Bool>:
  match flags axis:
    case Nil{} axis: []
    case Con{flag,rest} 0n: rest
    case Con{flag,rest} 1n+next: flag <> flag_removed(rest,next)

def flag_at(flags: List<&2,Bool>,axis: Nat) -> Bool:
  match flags axis:
    case Nil{} axis: False{}
    case Con{flag,rest} 0n: flag
    case Con{flag,rest} 1n+next: flag_at(rest,next)

def extents_removed(count: Nat,extents: List<&2,U32>,+axis: Nat) -> List<&2,U32>:
  match count:
    case 0n: extents
    case 1n+rest: extents_removed(rest,Grid.removed(extents,axis),axis)

def flags_removed(count: Nat,flags: List<&2,Bool>,+axis: Nat) -> List<&2,Bool>:
  match count:
    case 0n: flags
    case 1n+rest: flags_removed(rest,flag_removed(flags,axis),axis)

# The extents and flags of the box: the spanned axes of a reduced row from
# first on removed.
def row_extents(reduced: Bool,extents: List<&2,U32>,+first: Nat,+spanned: Nat) -> List<&2,U32>:
  match reduced:
    case True{}: extents_removed(spanned,extents,first)
    case False{}: extents

def row_flags(reduced: Bool,flags: List<&2,Bool>,+first: Nat,+spanned: Nat) -> List<&2,Bool>:
  match reduced:
    case True{}: flags_removed(spanned,flags,first)
    case False{}: flags

# The partials of the outputs of a box of rows, given the partials of the
# rows in the row-major order of the box without the row axis.
def box_outputs(~Partial: Data,~merge: Partial -> Partial -> Partial,+extents: List<&2,U32>,+flags: List<&2,Bool>,partials: List<&2,Partial>) -> List<&2,Partial>:
  only_run(~Partial,passes(~Partial,~merge,List.reverse(&2,U32,extents),List.reverse(&2,Bool,flags),singles(~Partial,partials)))

def carried_joined(~Block: Type,~Partial: Data,~merge: Partial -> Partial -> Partial,+extents: List<&2,U32>,+flags: List<&2,Bool>,
  folded: Partition.Tree<&1,Block,Unit> & List<&2,Partial>) -> Partition.Tree<&1,Block,Unit> & List<&2,Partial>:
  (tree,partials) = folded
  (tree,box_outputs(~Partial,~merge,extents,flags,partials))

def cells_per_outer(+grid: Grid.Grid) -> U32:
  Grid.Grid{+count,prefix,+outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  U32.max((count / U32.max(outer,1) : U32),1)

def regrouped_back(~Block: Type,~Fill: Data,~Partial: Data,~vacant: Fill -> Block,+grid: Grid.Grid,+fill: Fill,result: Partition.Tree<&1,Block,Unit> & List<&2,Partial>) -> Partition.Tree<&1,Block,Unit> & List<&2,Partial>:
  (tree,partials) = result
  (Pieces.blocks_regrouped(~Block,~Fill,~vacant,Pieces.ranges_of_grid(grid),fill,tree),partials)

# The rows of the spanned axes from first on, the cells of the axes after them
# apart, or every cell a row when the axes are kept.
def row_line(reduced: Bool,+shape: List<&2,U32>,+first: Nat,+spanned: Nat) -> LineChain.Line:
  match reduced:
    case True{}:
      LineChain.Line{Grid.product(List.take(&2,U32,List.drop(&2,U32,shape,first),spanned)),Grid.product(List.drop(&2,U32,shape,Nat.add(first,spanned)))}
    case False{}: LineChain.Line{1,1}

# The merged partials of the outputs of a reduction of the pieces of cells of
# shape, no extent zero, in the row-major order of the kept axes, and the
# pieces back in the tree of their grid. The last trailing axes are kept and
# interleave in the rows of the spanned axes before them, which are reduced or
# all kept. fold folds
# count cells of a row step apart from position into a row state; a block too
# short for a task is vacant. rows and steps count one cell. The rows
# fold across the pieces in order, then their partials merge in the tree.
def read(~Block: Type,~Fill: Data,~Context: Data,~Row: Type,~Partial: Data,~begin: Context -> Row,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Row -> Block -> Block & Row,~close: Row -> Partial,~merge: Partial -> Partial -> Partial,~vacant: Fill -> Block,+contexts: List<&2,Context>,+rows: F32,+steps: F32,+shape: List<&2,U32>,+flags: List<&2,Bool>,+trailing: Nat,+spanned: Nat,+fill: Fill,+workers: U32,
  pieces: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Block,Unit> & List<&2,Partial>:
  +grid = Tensors.grid_of(shape)
  +first = Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned))
  +reduced = flag_at(flags,first)
  +line = row_line(reduced,shape,first,spanned)
  regrouped_back(~Block,~Fill,~Partial,~vacant,grid,fill,
    carried_joined(~Block,~Partial,~merge,row_extents(reduced,shape,first,spanned),row_flags(reduced,flags,first,spanned),
      LineChain.folds(~Block,~Fill,~Context,~Row,~Partial,~begin,~fold,~close,~vacant,contexts,line,(LineChain.cells_of(line) / cells_per_outer(grid) : U32),rows,steps,
        grid,fill,workers,pieces)))

def finished(~Element: Data,~Partial: Data,~finish: Partial -> Element,partials: List<&2,Partial>) -> List<&2,Element>:
  match partials:
    case Nil{}: []
    case Con{partial,rest}: finish(partial) <> finished(~Element,~Partial,~finish,rest)

def kept_extent(reduced: Bool,extent: U32,rest: List<&2,U32>) -> List<&2,U32>:
  match reduced:
    case True{}: rest
    case False{}: extent <> rest

def kept_extents(extents: List<&2,U32>,flags: List<&2,Bool>) -> List<&2,U32>:
  match extents flags:
    case Con{extent,rest} Con{flag,others}: kept_extent(flag,extent,kept_extents(rest,others))
    case extents flags: []

def partials_only(~Block: Type,~Partial: Data,result: Partition.Tree<&1,Block,Unit> & List<&2,Partial>) -> List<&2,Partial>:
  (pieces,partials) = result
  partials

# empty: a kept extent is zero, every output then fill. unread: a reduced
# extent is zero, every output then the finish of zero.
def reduce_sized(~Element: Data,~Context: Data,~Row: Type,~Partial: Data,~begin: Context -> Row,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Row -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Row,~close: Row -> Partial,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,
  empty: Bool,unread: Bool,+operations: F32,+context: Context,+zero: Partial,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+flags: List<&2,Bool>,+fill: Element,
  +workers: U32) -> Tensors.Tensor<Element>:
  match empty unread:
    case True{} unread: Tensors.Tensor{Tensors.filled(~Element,kept_extents(shape,flags),fill),kept_extents(shape,flags),fill,Tensors.Usable{},workers}
    case False{} True{}: Tensors.Tensor{Tensors.filled(~Element,kept_extents(shape,flags),finish(zero)),kept_extents(shape,flags),fill,Tensors.Usable{},workers}
    case False{} False{}:
      Tensors.with_workers(~Element,Tensors.from_list(~Element,kept_extents(shape,flags),
        finished(~Element,~Partial,~finish,partials_only(~Storage.Buffer<Element>,~Partial,
          read(~Storage.Buffer<Element>,~Element,~Context,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),
            [context],1.0,operations,shape,flags,0n,1n,fill,workers,pieces))),fill),workers)

def reduce_listed(~Element: Data,~Context: Data,~Row: Type,~Partial: Data,~begin: Context -> Row,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Row -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Row,~close: Row -> Partial,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,
  +name: String,status: Tensors.Status,valid: Bool,+operations: F32,+context: Context,+zero: Partial,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+flags: List<&2,Bool>,
  +fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match status valid:
    case Tensors.Usable{} True{}:
      reduce_sized(~Element,~Context,~Row,~Partial,~begin,~fold,~close,~merge,~finish,Shapes.has_zero(kept_extents(shape,flags)),Shapes.has_zero(shape),
        operations,context,zero,pieces,shape,flags,fill,workers)
    case Tensors.Usable{} False{}: Tensors.failed(~Element,name,Tensors.InvalidArgument{shape},kept_extents(shape,flags),fill,workers)
    case Tensors.Failed{problem} valid: Tensors.Tensor{pieces,kept_extents(shape,flags),fill,Tensors.Failed{problem},workers}

# The reduction of the listed axes in the order of this file: every row
# folded from begin of the context by fold and closed, the partials merged in the tree, and
# finish of the merged partial. The result has the extents of the axes kept.
# operations counts the steps of one value. A row is a line of the last axis
# when it is reduced, otherwise every value.
def reduce(~Element: Data,~Context: Data,~Row: Type,~Partial: Data,~begin: Context -> Row,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Row -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Row,~close: Row -> Partial,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,
  name: String,+operations: F32,+context: Context,+zero: Partial,tensor: Tensors.Tensor<Element>,+axes: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.Tensor{pieces,+shape,+fill,status,+workers} = tensor
  +rank = List.length(&2,U32,shape)
  +reduced = Layout.naturals(axes)
  reduce_listed(~Element,~Context,~Row,~Partial,~begin,~fold,~close,~merge,~finish,name,status,Views.unique_axes(reduced) && Layout.all_below(axes,rank),operations,context,zero,pieces,shape,
    Layout.reduced_flags(rank,0n,reduced),fill,workers)
