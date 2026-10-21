import Base
import ../kernel_weights.bend as Weights
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws

def value_at(+tree: Tree.Tree<F32>,rows: U32,row: U32,index: U32) -> F32:
  Bool.pick(F32,U32.is_lt(row,rows),Tree.read(F32,tree,index),0.0)

def value(+tree: Tree.Tree<F32>,rows: U32,+inner: U32,+row: U32,index: U32) -> F32:
  value_at(tree,rows,row,(row * inner + index : U32))

def sample_rows(+tree: Tree.Tree<F32>,+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases) -> Tile.Tile8:
  Tile.Tile{value_at(tree,rows,row,(Weights.base0(bases) + index : U32)),value_at(tree,rows,(row + 1 : U32),(Weights.base1(bases) + index : U32)),value_at(tree,rows,(row + 2 : U32),(Weights.base2(bases) + index : U32)),value_at(tree,rows,(row + 3 : U32),(Weights.base3(bases) + index : U32)),value_at(tree,rows,(row + 4 : U32),(Weights.base4(bases) + index : U32)),value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}

# The tile at one inner index of eight rows from row, row_stride weights
# apart.
def sample_linear(tree: Tree.Tree<F32>,rows: U32,row_stride: U32,+row: U32,index: U32) -> Tile.Tile8:
  sample_rows(tree,rows,row,index,Weights.linear_bases(~Unit,Unit{},row_stride,row))

def conditional(+tree: Tree.Tree<F32>,+valid: Bool,+index: U32) ->
  {Storage.read_when(~F32,0.0,valid,Tree.pack(F32,tree),index) ==
    (Tree.pack(F32,tree),Bool.pick(F32,valid,Tree.read(F32,tree,index),0.0)) : Array<F32> & F32}:
  match valid:
    case False{}: {==}
    case True{}: StorageLaws.read(F32,tree,index)

def seventh_rows(+tree: Tree.Tree<F32>,+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,+g: F32) ->
  {Weights.seventh_rows(rows,row,index,bases,a,b,c,d,e,f,(Tree.pack(F32,tree),g)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,f,g,value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 7 : U32),rows),Tree.pack(F32,tree),(Weights.base7(bases) + index : U32)),
    (Tree.pack(F32,tree),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))),conditional(tree,U32.is_lt((row + 7 : U32),rows),(Weights.base7(bases) + index : U32))) :
    {Weights.eighth(a,b,c,d,e,f,g,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,f,g,value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}
  {==}

def sixth_rows(+tree: Tree.Tree<F32>,+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32) ->
  {Weights.sixth_rows(rows,row,index,bases,a,b,c,d,e,(Tree.pack(F32,tree),f)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,f,value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 6 : U32),rows),Tree.pack(F32,tree),(Weights.base6(bases) + index : U32)),
    (Tree.pack(F32,tree),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32))),conditional(tree,U32.is_lt((row + 6 : U32),rows),(Weights.base6(bases) + index : U32))) :
    {Weights.seventh_rows(rows,row,index,bases,a,b,c,d,e,f,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,f,value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}
  seventh_rows(tree,rows,row,index,bases,a,b,c,d,e,f,value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)))

def fifth_rows(+tree: Tree.Tree<F32>,+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32) ->
  {Weights.fifth_rows(rows,row,index,bases,a,b,c,d,(Tree.pack(F32,tree),e)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 5 : U32),rows),Tree.pack(F32,tree),(Weights.base5(bases) + index : U32)),
    (Tree.pack(F32,tree),value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32))),conditional(tree,U32.is_lt((row + 5 : U32),rows),(Weights.base5(bases) + index : U32))) :
    {Weights.sixth_rows(rows,row,index,bases,a,b,c,d,e,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,e,value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}
  sixth_rows(tree,rows,row,index,bases,a,b,c,d,e,value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)))

def fourth_rows(+tree: Tree.Tree<F32>,+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,+a: F32,+b: F32,+c: F32,+d: F32) ->
  {Weights.fourth_rows(rows,row,index,bases,a,b,c,(Tree.pack(F32,tree),d)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,value_at(tree,rows,(row + 4 : U32),(Weights.base4(bases) + index : U32)),value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 4 : U32),rows),Tree.pack(F32,tree),(Weights.base4(bases) + index : U32)),
    (Tree.pack(F32,tree),value_at(tree,rows,(row + 4 : U32),(Weights.base4(bases) + index : U32))),conditional(tree,U32.is_lt((row + 4 : U32),rows),(Weights.base4(bases) + index : U32))) :
    {Weights.fifth_rows(rows,row,index,bases,a,b,c,d,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,c,d,value_at(tree,rows,(row + 4 : U32),(Weights.base4(bases) + index : U32)),value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}
  fifth_rows(tree,rows,row,index,bases,a,b,c,d,value_at(tree,rows,(row + 4 : U32),(Weights.base4(bases) + index : U32)))

def third_rows(+tree: Tree.Tree<F32>,+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,+a: F32,+b: F32,+c: F32) ->
  {Weights.third_rows(rows,row,index,bases,a,b,(Tree.pack(F32,tree),c)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,c,value_at(tree,rows,(row + 3 : U32),(Weights.base3(bases) + index : U32)),value_at(tree,rows,(row + 4 : U32),(Weights.base4(bases) + index : U32)),value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 3 : U32),rows),Tree.pack(F32,tree),(Weights.base3(bases) + index : U32)),
    (Tree.pack(F32,tree),value_at(tree,rows,(row + 3 : U32),(Weights.base3(bases) + index : U32))),conditional(tree,U32.is_lt((row + 3 : U32),rows),(Weights.base3(bases) + index : U32))) :
    {Weights.fourth_rows(rows,row,index,bases,a,b,c,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,c,value_at(tree,rows,(row + 3 : U32),(Weights.base3(bases) + index : U32)),value_at(tree,rows,(row + 4 : U32),(Weights.base4(bases) + index : U32)),value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}
  fourth_rows(tree,rows,row,index,bases,a,b,c,value_at(tree,rows,(row + 3 : U32),(Weights.base3(bases) + index : U32)))

def second_rows(+tree: Tree.Tree<F32>,+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,+a: F32,+b: F32) ->
  {Weights.second_rows(rows,row,index,bases,a,(Tree.pack(F32,tree),b)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,b,value_at(tree,rows,(row + 2 : U32),(Weights.base2(bases) + index : U32)),value_at(tree,rows,(row + 3 : U32),(Weights.base3(bases) + index : U32)),value_at(tree,rows,(row + 4 : U32),(Weights.base4(bases) + index : U32)),value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 2 : U32),rows),Tree.pack(F32,tree),(Weights.base2(bases) + index : U32)),
    (Tree.pack(F32,tree),value_at(tree,rows,(row + 2 : U32),(Weights.base2(bases) + index : U32))),conditional(tree,U32.is_lt((row + 2 : U32),rows),(Weights.base2(bases) + index : U32))) :
    {Weights.third_rows(rows,row,index,bases,a,b,_) == (Tree.pack(F32,tree),Tile.Tile{a,b,value_at(tree,rows,(row + 2 : U32),(Weights.base2(bases) + index : U32)),value_at(tree,rows,(row + 3 : U32),(Weights.base3(bases) + index : U32)),value_at(tree,rows,(row + 4 : U32),(Weights.base4(bases) + index : U32)),value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}
  third_rows(tree,rows,row,index,bases,a,b,value_at(tree,rows,(row + 2 : U32),(Weights.base2(bases) + index : U32)))

def first_rows(+tree: Tree.Tree<F32>,+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases,+a: F32) ->
  {Weights.first_rows(rows,row,index,bases,(Tree.pack(F32,tree),a)) ==
    (Tree.pack(F32,tree),Tile.Tile{a,value_at(tree,rows,(row + 1 : U32),(Weights.base1(bases) + index : U32)),value_at(tree,rows,(row + 2 : U32),(Weights.base2(bases) + index : U32)),value_at(tree,rows,(row + 3 : U32),(Weights.base3(bases) + index : U32)),value_at(tree,rows,(row + 4 : U32),(Weights.base4(bases) + index : U32)),value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt((row + 1 : U32),rows),Tree.pack(F32,tree),(Weights.base1(bases) + index : U32)),
    (Tree.pack(F32,tree),value_at(tree,rows,(row + 1 : U32),(Weights.base1(bases) + index : U32))),conditional(tree,U32.is_lt((row + 1 : U32),rows),(Weights.base1(bases) + index : U32))) :
    {Weights.second_rows(rows,row,index,bases,a,_) == (Tree.pack(F32,tree),Tile.Tile{a,value_at(tree,rows,(row + 1 : U32),(Weights.base1(bases) + index : U32)),value_at(tree,rows,(row + 2 : U32),(Weights.base2(bases) + index : U32)),value_at(tree,rows,(row + 3 : U32),(Weights.base3(bases) + index : U32)),value_at(tree,rows,(row + 4 : U32),(Weights.base4(bases) + index : U32)),value_at(tree,rows,(row + 5 : U32),(Weights.base5(bases) + index : U32)),value_at(tree,rows,(row + 6 : U32),(Weights.base6(bases) + index : U32)),value_at(tree,rows,(row + 7 : U32),(Weights.base7(bases) + index : U32))}) : Array<F32> & Tile.Tile8}
  second_rows(tree,rows,row,index,bases,a,value_at(tree,rows,(row + 1 : U32),(Weights.base1(bases) + index : U32)))

def read_rows(+tree: Tree.Tree<F32>,+rows: U32,+row: U32,+index: U32,+bases: Weights.Bases) ->
  {Weights.read_rows(Tree.pack(F32,tree),rows,row,index,bases) == (Tree.pack(F32,tree),sample_rows(tree,rows,row,index,bases)) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,Storage.read_when(~F32,0.0,U32.is_lt(row,rows),Tree.pack(F32,tree),(Weights.base0(bases) + index : U32)),
    (Tree.pack(F32,tree),value_at(tree,rows,row,(Weights.base0(bases) + index : U32))),conditional(tree,U32.is_lt(row,rows),(Weights.base0(bases) + index : U32))) :
    {Weights.first_rows(rows,row,index,bases,_) == (Tree.pack(F32,tree),sample_rows(tree,rows,row,index,bases)) : Array<F32> & Tile.Tile8}
  first_rows(tree,rows,row,index,bases,value_at(tree,rows,row,(Weights.base0(bases) + index : U32)))

def sample_mapped_rows(~Context: Data,~offset: Context -> U32 -> U32,tree: Tree.Tree<F32>,context: Weights.MappedRows<Context>,index: U32) -> Tile.Tile8:
  Weights.MappedRows{rows,row,destination,context,bases} = context
  sample_rows(tree,rows,row,offset(context,index),bases)

def read_mapped_rows(~Context: Data,~offset: Context -> U32 -> U32,+tree: Tree.Tree<F32>,+context: Weights.MappedRows<Context>,+index: U32) ->
  {Weights.read_mapped_rows(~Context,~offset,Tree.pack(F32,tree),context,index) == (Tree.pack(F32,tree),sample_mapped_rows(~Context,~offset,tree,context,index)) : Array<F32> & Tile.Tile8}:
  Weights.MappedRows{rows,row,destination,context,bases} = context
  read_rows(tree,rows,row,offset(context,index),bases)
