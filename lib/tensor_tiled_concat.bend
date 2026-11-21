# Concatenation of tiled parts. When every part is Usable, nonempty and of the
# shape of the first part but along the axis, every part takes its lanes on
# the lane axis of the first part with the groups innermost and the cells of
# the parts are concatenated along the axis. Along the lane axis every part
# before the last fills its groups, otherwise the lanes move to the longest
# other axis. Other parts concatenate with every part on its unit axis, one
# value a cell, which reports them.
import Base
import ./tensor.bend as Tensors
import ./tensor_shape.bend as Shapes
import ./tensor_layout.bend as Layout
import ./tensor_grid.bend as Grid
import ./tensor_tiled.bend as Tiled
import ./kernel_tile.bend as Tile

# The parts of a concatenation, in order.
type Parts is Type:
  Single{tensor: Tiled.Tiled}
  Cons{tensor: Tiled.Tiled,rest: Parts}

# Whether every part so far fits, whether those before the last fill their
# groups along the axis, and their extents along the axis.
type Fit is Data:
  Fit{fits: Bool,whole: Bool,total: U32}

# A part fits when it is Usable and nonempty and has the shape of the first
# part but along the axis.
def fitting(+first: List<&2,U32>,+axis: Nat,status: Tensors.Status,+shape: List<&2,U32>) -> Bool:
  Tensors.usable(status) && Bool.not(Shapes.has_zero(shape)) &&
    Shapes.same_extents(Layout.with_extent(shape,axis,Layout.extent_at(first,axis)),first)

def fit_part(fit: Fit,+first: List<&2,U32>,+axis: Nat,+lane: Nat,+last: Bool,status: Tensors.Status,+shape: List<&2,U32>) -> Fit:
  Fit{fits,whole,+total} = fit
  +width = Layout.extent_at(shape,axis)
  Fit{fits && fitting(first,axis,status,shape) && U32.is_le(width,(4294967295 - total : U32)),whole && (last || U32.is_zero((width % 8 : U32))),(total + width : U32)}

def prepend(tensor: Tiled.Tiled,result: Parts & Fit) -> Parts & Fit:
  (rest,fit) = result
  (Cons{tensor,rest},fit)

def survey(parts: Parts,+first: List<&2,U32>,+axis: Nat,+lane: Nat,fit: Fit) -> Parts & Fit:
  match parts:
    case Single{tensor}:
      Tiled.with_metadata(~(Parts & Fit),tensor,tensor => shape => status => workers =>
        +shape = shape
        (Single{tensor},fit_part(fit,first,axis,lane,True{},status,shape)))
    case Cons{tensor,rest}:
      Tiled.with_metadata(~(Parts & Fit),tensor,tensor => shape => status => workers =>
        +shape = shape
        prepend(tensor,survey(rest,first,axis,lane,fit_part(fit,first,axis,lane,False{},status,shape))))

# The cells of every part with its lanes on the lane axis, the groups
# innermost.
def part_cells(tensor: Tiled.Tiled,+lane: Nat) -> Tensors.Tensor<Tile.Tile8>:
  Tiled.cells_of(Tiled.arranged(tensor,lane,Grid.Innermost{}))

def cells(parts: Parts,+lane: Nat) -> Tensors.Parts<Tile.Tile8>:
  match parts:
    case Single{tensor}: Tensors.Single{part_cells(tensor,lane)}
    case Cons{tensor,rest}: Tensors.Cons{part_cells(tensor,lane),cells(rest,lane)}

# The cells of every part on its unit axis.
def units(parts: Parts) -> Tensors.Parts<Tile.Tile8>:
  match parts:
    case Single{tensor}: Tensors.Single{Tiled.unit_cells(tensor)}
    case Cons{tensor,rest}: Tensors.Cons{Tiled.unit_cells(tensor),units(rest)}

def concat_fitted(fits: Bool,+axis: U32,+lane: Nat,+lanes: U32,+rank: Nat,parts: Parts) -> Tiled.Tiled:
  match fits:
    case True{}: Tiled.Tiled{Layout.concat(~Tile.Tile8,cells(parts,lane),U32.from_nat(Tiled.cell_axis(Grid.Innermost{},U32.to_nat(axis),lane,rank))),lane,lanes,Grid.Innermost{}}
    case False{}: Tiled.Tiled{Layout.concat(~Tile.Tile8,units(parts),axis),rank,1,Grid.Innermost{}}

# The lane axis of the first part, or the longest other axis when the parts
# cut its groups.
def concat_lane(+first: List<&2,U32>,+axis: Nat,+lane: Nat,+whole: Bool) -> Nat:
  Bool.pick(Nat,Nat.is_eq(axis,lane) && Bool.not(whole),Tiled.longest_other(first,axis,0n,List.length(&2,U32,first),2),lane)

def concat_at(+first: List<&2,U32>,+axis: U32,+lane: Nat,+fits: Bool,+total: U32,parts: Parts) -> Tiled.Tiled:
  +rank = List.length(&2,U32,first)
  concat_fitted(fits && Nat.is_lt(U32.to_nat(axis),rank) && Nat.is_le(lane,rank),axis,lane,
    Bool.pick(U32,Nat.is_eq(U32.to_nat(axis),lane),total,Grid.lanes_at(first,lane)),rank,parts)

def concat_surveyed(+first: List<&2,U32>,+axis: U32,+lane: Nat,surveyed: Parts & Fit) -> Tiled.Tiled:
  (parts,fit) = surveyed
  Fit{+fits,+whole,+total} = fit
  concat_at(first,axis,concat_lane(first,U32.to_nat(axis),lane,whole),fits,total,parts)

# The extents and the lane axis of the first part.
type Lead is Data:
  Lead{shape: List<&2,U32>,lane: Nat}

def led(started: Tiled.Tiled & Nat,+shape: List<&2,U32>) -> Tiled.Tiled & Lead:
  (tensor,+lane) = started
  (tensor,Lead{shape,lane})

def first_lead(tensor: Tiled.Tiled) -> Tiled.Tiled & Lead:
  Tiled.with_metadata(~(Tiled.Tiled & Lead),tensor,tensor => shape => status => workers => led(Tiled.lane_axis_of(tensor),shape))

def single_lead(pair: Tiled.Tiled & Lead) -> Parts & Lead:
  (tensor,lead) = pair
  (Single{tensor},lead)

def cons_lead(rest: Parts,pair: Tiled.Tiled & Lead) -> Parts & Lead:
  (tensor,lead) = pair
  (Cons{tensor,rest},lead)

def lead(parts: Parts) -> Parts & Lead:
  match parts:
    case Single{tensor}: single_lead(first_lead(tensor))
    case Cons{tensor,rest}: cons_lead(rest,first_lead(tensor))

def concat_led(+axis: U32,led: Parts & Lead) -> Tiled.Tiled:
  (parts,Lead{+first,+lane}) = led
  concat_surveyed(first,axis,lane,survey(parts,first,U32.to_nat(axis),lane,Fit{True{},True{},0}))

# Layout.concat of the values of tiled parts, on the cells when the parts fit.
def concat(parts: Parts,+axis: U32) -> Tiled.Tiled:
  concat_led(axis,lead(parts))
