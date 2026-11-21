# Selection acts directly on cells.
import Base
import ./math_compare.bend as Compare
import ./kernel_tile.bend as Tile

def select_tile(mask: Tile.Tile8,yes: Tile.Tile8,no: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a,b,c,d,e,f,g,h} = mask
  Tile.Tile{i,j,k,l,m,n,o,p} = yes
  Tile.Tile{q,r,s,t,u,v,w,x} = no
  Tile.Tile{Compare.select(a,i,q),Compare.select(b,j,r),Compare.select(c,k,s),Compare.select(d,l,t),
    Compare.select(e,m,u),Compare.select(f,n,v),Compare.select(g,o,w),Compare.select(h,p,x)}
