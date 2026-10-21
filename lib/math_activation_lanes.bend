# Arithmetic stages across each eight-value cell.
import Base
import ./math_fp32.bend as Arithmetic
import ./kernel_tile.bend as Tile

type Arguments is Data:
  Arguments{original: Tile.Tile8,value: Tile.Tile8,scaled: Tile.Tile8}

def arguments(a: Arithmetic.DerivativeArgument,b: Arithmetic.DerivativeArgument,c: Arithmetic.DerivativeArgument,d: Arithmetic.DerivativeArgument,e: Arithmetic.DerivativeArgument,f: Arithmetic.DerivativeArgument,g: Arithmetic.DerivativeArgument,h: Arithmetic.DerivativeArgument) -> Arguments:
  Arithmetic.DerivativeArgument{ax,av,ascale} = a
  Arithmetic.DerivativeArgument{bx,bv,bs} = b
  Arithmetic.DerivativeArgument{cx,cv,cs} = c
  Arithmetic.DerivativeArgument{dx,dv,ds} = d
  Arithmetic.DerivativeArgument{ex,ev,es} = e
  Arithmetic.DerivativeArgument{fx,fv,fs} = f
  Arithmetic.DerivativeArgument{gx,gv,gs} = g
  Arithmetic.DerivativeArgument{hx,hv,hs} = h
  Arguments{Tile.Tile{ax,bx,cx,dx,ex,fx,gx,hx},Tile.Tile{av,bv,cv,dv,ev,fv,gv,hv},Tile.Tile{ascale,bs,cs,ds,es,fs,gs,hs}}

def silu_first(tile: Tile.Tile8) -> Arguments:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  arguments(Arithmetic.silu_derivative_prepare(a),Arithmetic.silu_derivative_prepare(b),Arithmetic.silu_derivative_prepare(c),Arithmetic.silu_derivative_prepare(d),Arithmetic.silu_derivative_prepare(e),Arithmetic.silu_derivative_prepare(f),Arithmetic.silu_derivative_prepare(g),Arithmetic.silu_derivative_prepare(h))

def finished(~finish: Arithmetic.DerivativeArgument -> F32,args: Arguments) -> Tile.Tile8:
  Arguments{original,value,scaled} = args
  Tile.Tile{ax,bx,cx,dx,ex,fx,gx,hx} = original
  Tile.Tile{av,bv,cv,dv,ev,fv,gv,hv} = value
  Tile.Tile{ascale,bs,cs,ds,es,fs,gs,hs} = scaled
  Tile.Tile{finish(Arithmetic.DerivativeArgument{ax,av,ascale}),finish(Arithmetic.DerivativeArgument{bx,bv,bs}),finish(Arithmetic.DerivativeArgument{cx,cv,cs}),finish(Arithmetic.DerivativeArgument{dx,dv,ds}),finish(Arithmetic.DerivativeArgument{ex,ev,es}),finish(Arithmetic.DerivativeArgument{fx,fv,fs}),finish(Arithmetic.DerivativeArgument{gx,gv,gs}),finish(Arithmetic.DerivativeArgument{hx,hv,hs})}

def silu(context: Unit,tile: Tile.Tile8) -> Tile.Tile8:
  finished(~Arithmetic.silu_derivative_finish,silu_first(tile))

type GeluPrepared is Data:
  GeluPrepared{original: Tile.Tile8,head: Tile.Tile8,low: Tile.Tile8,slope: Tile.Tile8,slope_error: Tile.Tile8}

def gelu_prepared(a: Arithmetic.GeluPrepared,b: Arithmetic.GeluPrepared,c: Arithmetic.GeluPrepared,d: Arithmetic.GeluPrepared,e: Arithmetic.GeluPrepared,f: Arithmetic.GeluPrepared,g: Arithmetic.GeluPrepared,h: Arithmetic.GeluPrepared) -> GeluPrepared:
  Arithmetic.GeluPrepared{a0,a1,a2,a3,a4} = a
  Arithmetic.GeluPrepared{b0,b1,b2,b3,b4} = b
  Arithmetic.GeluPrepared{c0,c1,c2,c3,c4} = c
  Arithmetic.GeluPrepared{d0,d1,d2,d3,d4} = d
  Arithmetic.GeluPrepared{e0,e1,e2,e3,e4} = e
  Arithmetic.GeluPrepared{f0,f1,f2,f3,f4} = f
  Arithmetic.GeluPrepared{g0,g1,g2,g3,g4} = g
  Arithmetic.GeluPrepared{h0,h1,h2,h3,h4} = h
  GeluPrepared{Tile.Tile{a0,b0,c0,d0,e0,f0,g0,h0},Tile.Tile{a1,b1,c1,d1,e1,f1,g1,h1},Tile.Tile{a2,b2,c2,d2,e2,f2,g2,h2},Tile.Tile{a3,b3,c3,d3,e3,f3,g3,h3},Tile.Tile{a4,b4,c4,d4,e4,f4,g4,h4}}

def gelu_first(tile: Tile.Tile8) -> GeluPrepared:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  gelu_prepared(Arithmetic.gelu_derivative_prepare(a),Arithmetic.gelu_derivative_prepare(b),Arithmetic.gelu_derivative_prepare(c),Arithmetic.gelu_derivative_prepare(d),Arithmetic.gelu_derivative_prepare(e),Arithmetic.gelu_derivative_prepare(f),Arithmetic.gelu_derivative_prepare(g),Arithmetic.gelu_derivative_prepare(h))

type GeluReduced is Data:
  GeluReduced{original: Tile.Tile8,power: Tile.Tile8,denominator: Tile.Tile8,error: Tile.Tile8,slope: Tile.Tile8,slope_error: Tile.Tile8}

def gelu_reduced(a: Arithmetic.GeluReduced,b: Arithmetic.GeluReduced,c: Arithmetic.GeluReduced,d: Arithmetic.GeluReduced,e: Arithmetic.GeluReduced,f: Arithmetic.GeluReduced,g: Arithmetic.GeluReduced,h: Arithmetic.GeluReduced) -> GeluReduced:
  Arithmetic.GeluReduced{a0,a1,a2,a3,a4,a5} = a
  Arithmetic.GeluReduced{b0,b1,b2,b3,b4,b5} = b
  Arithmetic.GeluReduced{c0,c1,c2,c3,c4,c5} = c
  Arithmetic.GeluReduced{d0,d1,d2,d3,d4,d5} = d
  Arithmetic.GeluReduced{e0,e1,e2,e3,e4,e5} = e
  Arithmetic.GeluReduced{f0,f1,f2,f3,f4,f5} = f
  Arithmetic.GeluReduced{g0,g1,g2,g3,g4,g5} = g
  Arithmetic.GeluReduced{h0,h1,h2,h3,h4,h5} = h
  GeluReduced{Tile.Tile{a0,b0,c0,d0,e0,f0,g0,h0},Tile.Tile{a1,b1,c1,d1,e1,f1,g1,h1},Tile.Tile{a2,b2,c2,d2,e2,f2,g2,h2},Tile.Tile{a3,b3,c3,d3,e3,f3,g3,h3},Tile.Tile{a4,b4,c4,d4,e4,f4,g4,h4},Tile.Tile{a5,b5,c5,d5,e5,f5,g5,h5}}

def gelu_second(args: GeluPrepared) -> GeluReduced:
  GeluPrepared{original,head,low,slope,slope_error} = args
  Tile.Tile{a0,b0,c0,d0,e0,f0,g0,h0} = original
  Tile.Tile{a1,b1,c1,d1,e1,f1,g1,h1} = head
  Tile.Tile{a2,b2,c2,d2,e2,f2,g2,h2} = low
  Tile.Tile{a3,b3,c3,d3,e3,f3,g3,h3} = slope
  Tile.Tile{a4,b4,c4,d4,e4,f4,g4,h4} = slope_error
  gelu_reduced(Arithmetic.gelu_derivative_exponential(Arithmetic.GeluPrepared{a0,a1,a2,a3,a4}),Arithmetic.gelu_derivative_exponential(Arithmetic.GeluPrepared{b0,b1,b2,b3,b4}),Arithmetic.gelu_derivative_exponential(Arithmetic.GeluPrepared{c0,c1,c2,c3,c4}),Arithmetic.gelu_derivative_exponential(Arithmetic.GeluPrepared{d0,d1,d2,d3,d4}),Arithmetic.gelu_derivative_exponential(Arithmetic.GeluPrepared{e0,e1,e2,e3,e4}),Arithmetic.gelu_derivative_exponential(Arithmetic.GeluPrepared{f0,f1,f2,f3,f4}),Arithmetic.gelu_derivative_exponential(Arithmetic.GeluPrepared{g0,g1,g2,g3,g4}),Arithmetic.gelu_derivative_exponential(Arithmetic.GeluPrepared{h0,h1,h2,h3,h4}))

def gelu_third(args: GeluReduced) -> Arguments:
  GeluReduced{original,power,denominator,error,slope,slope_error} = args
  Tile.Tile{a0,b0,c0,d0,e0,f0,g0,h0} = original
  Tile.Tile{a1,b1,c1,d1,e1,f1,g1,h1} = power
  Tile.Tile{a2,b2,c2,d2,e2,f2,g2,h2} = denominator
  Tile.Tile{a3,b3,c3,d3,e3,f3,g3,h3} = error
  Tile.Tile{a4,b4,c4,d4,e4,f4,g4,h4} = slope
  Tile.Tile{a5,b5,c5,d5,e5,f5,g5,h5} = slope_error
  arguments(Arithmetic.gelu_derivative_evaluated(Arithmetic.GeluReduced{a0,a1,a2,a3,a4,a5}),Arithmetic.gelu_derivative_evaluated(Arithmetic.GeluReduced{b0,b1,b2,b3,b4,b5}),Arithmetic.gelu_derivative_evaluated(Arithmetic.GeluReduced{c0,c1,c2,c3,c4,c5}),Arithmetic.gelu_derivative_evaluated(Arithmetic.GeluReduced{d0,d1,d2,d3,d4,d5}),Arithmetic.gelu_derivative_evaluated(Arithmetic.GeluReduced{e0,e1,e2,e3,e4,e5}),Arithmetic.gelu_derivative_evaluated(Arithmetic.GeluReduced{f0,f1,f2,f3,f4,f5}),Arithmetic.gelu_derivative_evaluated(Arithmetic.GeluReduced{g0,g1,g2,g3,g4,g5}),Arithmetic.gelu_derivative_evaluated(Arithmetic.GeluReduced{h0,h1,h2,h3,h4,h5}))

def gelu_tanh(context: Unit,tile: Tile.Tile8) -> Tile.Tile8:
  finished(~Arithmetic.gelu_derivative_finish,gelu_third(gelu_second(gelu_first(tile))))
