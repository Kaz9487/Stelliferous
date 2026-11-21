# The groups of batch normalization: the parameters of group g are the tiles
# of cell g of the statistics and parameters [C], and a lane of a group's
# output is the formula of its lane.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_batch_norm.bend as Kernel
import ../tensor_batch_norm.bend as BN
import ./tensor_contract.bend as Contract
import ./tensor_batch_norm_passes.bend as Passes
import ./kernel_tile_refinement.bend as TileLaws
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding

# ---- Groups read from lists ----

# The parameter before a count of the list of a reader is the reader's value
# at the offset plus that count.
def parameter_listed(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+fallback: Element,count: Nat,+offset: Nat,+context: Context,index: Nat,
  inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Passes.parameter(~Element,Contract.listed(~Element,~Context,~reader,count,offset,context),index,fallback) == reader(context,U32.from_nat(Nat.add(offset,index))) : Element}:
  match count index:
    case 0n 0n: Empty.absurd({fallback == reader(context,U32.from_nat(Nat.add(offset,0n))) : Element},WordEncoding.false_true(inside))
    case 0n 1n+previous: Empty.absurd({fallback == reader(context,U32.from_nat(Nat.add(offset,1n+previous))) : Element},WordEncoding.false_true(inside))
    case 1n+rest 0n:
      %Equal.sym(Nat,Nat.add(offset,0n),offset,Addition.zero_right(offset)) : {reader(context,U32.from_nat(offset)) == reader(context,U32.from_nat(_)) : Element}
      {==}
    case 1n+ +rest 1n+ +previous:
      %Equal.sym(Nat,Nat.add(offset,1n+previous),1n+Nat.add(offset,previous),Addition.successor_right(offset,previous)) :
        {Passes.parameter(~Element,Contract.listed(~Element,~Context,~reader,rest,1n+offset,context),previous,fallback) == reader(context,U32.from_nat(_)) : Element}
      parameter_listed(~Element,~Context,~reader,fallback,rest,1n+offset,context,previous,inside)

# What forward reads of the groups: the cells of the mean, the deviation,
# gamma and beta.
type Normals<-M: Data,-D: Data,-G: Data,-B: Data> is Data:
  Normals{mean: M,deviation: D,gamma: G,beta: B}

def normal_at(~M: Data,~mr: M -> U32 -> Tile.Tile8,~D: Data,~dr: D -> U32 -> Tile.Tile8,~G: Data,~gr: G -> U32 -> Tile.Tile8,~B: Data,~br: B -> U32 -> Tile.Tile8,
  context: Normals<M,D,G,B>,index: U32) -> Kernel.Normal:
  Normals{mean,deviation,gamma,beta} = context
  +index = index
  Kernel.Normal{mr(mean,index),dr(deviation,index),gr(gamma,index),br(beta,index)}

# The groups of the lists of four readers are the list of their groups.
def normal_listed(~M: Data,~mr: M -> U32 -> Tile.Tile8,~D: Data,~dr: D -> U32 -> Tile.Tile8,~G: Data,~gr: G -> U32 -> Tile.Tile8,~B: Data,~br: B -> U32 -> Tile.Tile8,
  count: Nat,+offset: Nat,+mean: M,+deviation: D,+gamma: G,+beta: B) ->
  {BN.normal_groups(Contract.listed(~Tile.Tile8,~M,~mr,count,offset,mean),Contract.listed(~Tile.Tile8,~D,~dr,count,offset,deviation),
      Contract.listed(~Tile.Tile8,~G,~gr,count,offset,gamma),Contract.listed(~Tile.Tile8,~B,~br,count,offset,beta)) ==
    Contract.listed(~Kernel.Normal,~Normals<M,D,G,B>,~normal_at(~M,~mr,~D,~dr,~G,~gr,~B,~br),count,offset,Normals{mean,deviation,gamma,beta}) : List<&2,Kernel.Normal>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,Kernel.Normal>,List<&2,Kernel.Normal>,tail => Kernel.Normal{mr(mean,U32.from_nat(offset)),dr(deviation,U32.from_nat(offset)),gr(gamma,U32.from_nat(offset)),br(beta,U32.from_nat(offset))} <> tail,
        BN.normal_groups(Contract.listed(~Tile.Tile8,~M,~mr,rest,1n+offset,mean),Contract.listed(~Tile.Tile8,~D,~dr,rest,1n+offset,deviation),
          Contract.listed(~Tile.Tile8,~G,~gr,rest,1n+offset,gamma),Contract.listed(~Tile.Tile8,~B,~br,rest,1n+offset,beta)),
        Contract.listed(~Kernel.Normal,~Normals<M,D,G,B>,~normal_at(~M,~mr,~D,~dr,~G,~gr,~B,~br),rest,1n+offset,Normals{mean,deviation,gamma,beta}),
        normal_listed(~M,~mr,~D,~dr,~G,~gr,~B,~br,rest,1n+offset,mean,deviation,gamma,beta))

# What backward reads of the groups: the cells of the two sums and of the
# scale, and the number of values of a channel.
type Backs<-B: Data,-G: Data,-S: Data> is Data:
  Backs{beta: B,gamma: G,scale: S,total: F32}

def back_at(~B: Data,~br: B -> U32 -> Tile.Tile8,~G: Data,~gr: G -> U32 -> Tile.Tile8,~S: Data,~sr: S -> U32 -> Tile.Tile8,context: Backs<B,G,S>,index: U32) -> Kernel.Back:
  Backs{beta,gamma,scale,total} = context
  +index = index
  Kernel.Back{br(beta,index),gr(gamma,index),sr(scale,index),total}

def back_listed(~B: Data,~br: B -> U32 -> Tile.Tile8,~G: Data,~gr: G -> U32 -> Tile.Tile8,~S: Data,~sr: S -> U32 -> Tile.Tile8,
  count: Nat,+offset: Nat,+beta: B,+gamma: G,+scale: S,+total: F32) ->
  {BN.back_groups(total,Contract.listed(~Tile.Tile8,~B,~br,count,offset,beta),Contract.listed(~Tile.Tile8,~G,~gr,count,offset,gamma),Contract.listed(~Tile.Tile8,~S,~sr,count,offset,scale)) ==
    Contract.listed(~Kernel.Back,~Backs<B,G,S>,~back_at(~B,~br,~G,~gr,~S,~sr),count,offset,Backs{beta,gamma,scale,total}) : List<&2,Kernel.Back>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,Kernel.Back>,List<&2,Kernel.Back>,tail => Kernel.Back{br(beta,U32.from_nat(offset)),gr(gamma,U32.from_nat(offset)),sr(scale,U32.from_nat(offset)),total} <> tail,
        BN.back_groups(total,Contract.listed(~Tile.Tile8,~B,~br,rest,1n+offset,beta),Contract.listed(~Tile.Tile8,~G,~gr,rest,1n+offset,gamma),Contract.listed(~Tile.Tile8,~S,~sr,rest,1n+offset,scale)),
        Contract.listed(~Kernel.Back,~Backs<B,G,S>,~back_at(~B,~br,~G,~gr,~S,~sr),rest,1n+offset,Backs{beta,gamma,scale,total}),
        back_listed(~B,~br,~G,~gr,~S,~sr,rest,1n+offset,beta,gamma,scale,total))

# ---- The lanes of a group ----
