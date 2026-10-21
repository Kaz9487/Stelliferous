# Two arithmetic stages for each eight-value cell.
import Base
import ./tensor_access.bend as Access
import ./math_activation.bend as Act
import ./math_fp32.bend as G
import ./kernel_tile.bend as Tile
import ./traversal_lanes.bend as Lanes
import ./traversal_cells.bend as Cells
import ./traversal_array.bend as Traversal

type Arguments is Data:
  Arguments{error: Tile.Tile8,head: Tile.Tile8,original: Tile.Tile8}

def arguments(a: G.Argument,b: G.Argument,c: G.Argument,d: G.Argument,e: G.Argument,f: G.Argument,g: G.Argument,h: G.Argument) -> Arguments:
  G.Argument{ax,aw,ae} = a
  G.Argument{bx,bw,be} = b
  G.Argument{cx,cw,ce} = c
  G.Argument{dx,dw,de} = d
  G.Argument{ex,ew,ee} = e
  G.Argument{fx,fw,fe} = f
  G.Argument{gx,gw,ge} = g
  G.Argument{hx,hw,he} = h
  Arguments{Tile.Tile{ae,be,ce,de,ee,fe,ge,he},Tile.Tile{aw,bw,cw,dw,ew,fw,gw,hw},Tile.Tile{ax,bx,cx,dx,ex,fx,gx,hx}}

def first(tile: Tile.Tile8) -> Arguments:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  arguments(G.gelu_prepared(a),G.gelu_prepared(b),G.gelu_prepared(c),G.gelu_prepared(d),G.gelu_prepared(e),G.gelu_prepared(f),G.gelu_prepared(g),G.gelu_prepared(h))

def second(args: Arguments) -> Tile.Tile8:
  Arguments{error,head,original} = args
  Tile.Tile{ae,be,ce,de,ee,fe,ge,he} = error
  Tile.Tile{aw,bw,cw,dw,ew,fw,gw,hw} = head
  Tile.Tile{ax,bx,cx,dx,ex,fx,gx,hx} = original
  Tile.Tile{G.gelu_finish(G.Argument{ax,aw,ae}),G.gelu_finish(G.Argument{bx,bw,be}),G.gelu_finish(G.Argument{cx,cw,ce}),G.gelu_finish(G.Argument{dx,dw,de}),G.gelu_finish(G.Argument{ex,ew,ee}),G.gelu_finish(G.Argument{fx,fw,fe}),G.gelu_finish(G.Argument{gx,gw,ge}),G.gelu_finish(G.Argument{hx,hw,he})}

def gathered(read: Array<F32> & Tile.Tile8) -> Array<F32> & Tile.Tile8:
  (values,tile) = read
  (values,second(first(tile)))

def identity(context: Unit,value: F32) -> F32:
  value

def gather(values: Array<F32>,position: U32,context: Unit) -> Array<F32> & Tile.Tile8:
  gathered(Lanes.gather(~Unit,~identity,values,position,context))

def map(count: Nat,position: U32,values: Array<F32>) -> Array<F32>:
  Lanes.rounds_with(~Array<F32>,~Unit,~gather,~Lanes.scatter,~Lanes.finish(~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value)),count,Cells.State{values,Lanes.scratch(),position,0,Unit{}})

def map_when(covered: Bool,count: Nat,position: U32,values: Array<F32>) -> Array<F32>:
  match covered:
    case True{}: map(count,position,values)
    case False{}: Traversal.map_in_place(~F32,~Unit,~Access.apply_element(~F32,~Act.gelu_tanh_value),count,position,Unit{},values)
