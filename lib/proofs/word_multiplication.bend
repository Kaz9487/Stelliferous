import Base
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./word_arithmetic.bend as WordArithmetic
import ./nat_to_u32_bounds.bend as Bounds

def truncate(n: Nat,w: Word(1n+n)) -> Word(n):
  match n:
    case 0n:
      WNil{}
    case 1n+p:
      WCon{b,t} = w
      WCon{b,truncate(p,t)}

def truncate_zero(+n: Nat) -> {truncate(n,Word.zero(1n+n)) == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      %truncate_zero(p) : {WCon{False{},truncate(p,Word.zero(1n+p))} == WCon{False{},_} : Word(1n+p)}
      {==}

def truncate_inc(+n: Nat,+w: Word(1n+n)) -> {truncate(n,Word.inc(1n+n,w)) == Word.inc(n,truncate(n,w)) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      match w:
        case WCon{False{},t}:
          {==}
        case WCon{True{},+t}:
          %truncate_inc(p,t) : {WCon{False{},truncate(p,Word.inc(1n+p,t))} == WCon{False{},_} : Word(1n+p)}
          {==}

def truncate_encode(+a: Nat,+n: Nat) -> {truncate(n,WordArithmetic.encode(a,1n+n)) == WordArithmetic.encode(a,n) : Word(n)}:
  match a:
    case 0n:
      truncate_zero(n)
    case 1n+ +p:
      %Equal.sym(Word(n),truncate(n,Word.inc(1n+n,WordArithmetic.encode(p,1n+n))),Word.inc(n,truncate(n,WordArithmetic.encode(p,1n+n))),
        truncate_inc(n,WordArithmetic.encode(p,1n+n))) : {_ == Word.inc(n,WordArithmetic.encode(p,n)) : Word(n)}
      %truncate_encode(p,n) : {Word.inc(n,truncate(n,WordArithmetic.encode(p,1n+n))) == Word.inc(n,_) : Word(n)}
      {==}

def put(+n: Nat,+b: Bool,+w: Word(n)) -> {Word.shl.put(n,b,w) == truncate(n,WCon{b,w}) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      WCon{+c,+t} = w
      %put(p,c,t) : {WCon{b,Word.shl.put(p,c,t)} == WCon{b,_} : Word(1n+p)}
      {==}

def shift(+n: Nat,+w: Word(1n+n)) -> {Word.shl(1n+n,w) == WCon{False{},truncate(n,w)} : Word(1n+n)}:
  WCon{+b,+t} = w
  %put(n,b,t) : {WCon{False{},Word.shl.put(n,b,t)} == WCon{False{},_} : Word(1n+n)}
  {==}

def encoded_double(+a: Nat,+n: Nat) -> {WordArithmetic.encode(Nat.double(a),1n+n) == WCon{False{},WordArithmetic.encode(a,n)} : Word(1n+n)}:
  %Equal.sym(Word(1n+n),WordArithmetic.encode(Nat.double(a),1n+n),Word.shl(1n+n,WordArithmetic.encode(a,1n+n)),WordArithmetic.encode_double(a,
      1n+n)) : {_ == WCon{False{},WordArithmetic.encode(a,n)} : Word(1n+n)}
  %Equal.sym(Word(1n+n),Word.shl(1n+n,WordArithmetic.encode(a,1n+n)),WCon{False{},truncate(n,WordArithmetic.encode(a,1n+n))},shift(n,
      WordArithmetic.encode(a,1n+n))) : {_ == WCon{False{},WordArithmetic.encode(a,n)} : Word(1n+n)}
  %truncate_encode(a,n) : {WCon{False{},truncate(n,WordArithmetic.encode(a,1n+n))} == WCon{False{},_} : Word(1n+n)}
  {==}

law word_roundtrip:
  for +n: Nat
  for +w: Word(n)
  {WordArithmetic.encode(Word.to_nat(n,w),n) == w : Word(n)}
def word_roundtrip(n,w):
  match n:
    case 0n:
      WNil{} = w
      {==}
    case 1n+ +p:
      match w:
        case WCon{False{},+t}:
          %Equal.sym(Word(1n+p),WordArithmetic.encode(Nat.double(Word.to_nat(p,t)),1n+p),WCon{False{},WordArithmetic.encode(Word.to_nat(p,t),p)},
            encoded_double(Word.to_nat(p,t),p)) : {_ == WCon{False{},t} : Word(1n+p)}
          %Equal.sym(Word(p),WordArithmetic.encode(Word.to_nat(p,t),p),t,word_roundtrip(p,t)) : {WCon{False{},_} == WCon{False{},t} : Word(1n+p)}
          {==}
        case WCon{True{},+t}:
          %Equal.sym(Word(1n+p),WordArithmetic.encode(Nat.double(Word.to_nat(p,t)),1n+p),WCon{False{},WordArithmetic.encode(Word.to_nat(p,t),p)},
            encoded_double(Word.to_nat(p,t),p)) : {Word.inc(1n+p,_) == WCon{True{},t} : Word(1n+p)}
          %Equal.sym(Word(p),WordArithmetic.encode(Word.to_nat(p,t),p),t,word_roundtrip(p,t)) : {WCon{True{},_} == WCon{True{},t} : Word(1n+p)}
          {==}

def double_mul(+a: Nat,+b: Nat) -> {Nat.mul(Nat.double(a),b) == Nat.mul(a,Nat.double(b)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+ +p:
      %Addition.double(b) : {Nat.add(b,Nat.add(b,Nat.mul(Nat.double(p),b))) == Nat.add(_,Nat.mul(p,Nat.double(b))) : Nat}
      %Equal.sym(Nat,Nat.add(Nat.add(b,b),Nat.mul(p,Nat.double(b))),Nat.add(b,Nat.add(b,Nat.mul(p,Nat.double(b)))),Addition.associative(b,b,Nat.mul(p,
            Nat.double(b)))) : {Nat.add(b,Nat.add(b,Nat.mul(Nat.double(p),b))) == _ : Nat}
      %double_mul(p,b) : {Nat.add(b,Nat.add(b,Nat.mul(Nat.double(p),b))) == Nat.add(b,Nat.add(b,_)) : Nat}
      {==}

law multiply_go:
  for +m: Nat
  for +n: Nat
  for +w: Word(m)
  for +b: Nat
  for +acc: Nat
  {Word.mul.go(n,m,w,WordArithmetic.encode(b,n),WordArithmetic.encode(acc,n)) == WordArithmetic.encode(Nat.add(acc,Nat.mul(Word.to_nat(m,w),b)),
      n) : Word(n)}
def multiply_go(m,n,w,b,acc):
  match m:
    case 0n:
      %Equal.sym(Nat,Nat.add(acc,0n),acc,Addition.zero_right(acc)) : {WordArithmetic.encode(acc,n) == WordArithmetic.encode(_,n) : Word(n)}
      {==}
    case 1n+ +p:
      match w:
        case WCon{False{},+t}:
          %WordArithmetic.encode_double(b,n) : {Word.mul.go(n,p,t,_,WordArithmetic.encode(acc,n)) == WordArithmetic.encode(Nat.add(acc,
                Nat.mul(Nat.double(Word.to_nat(p,t)),b)),n) : Word(n)}
          %Equal.sym(Nat,Nat.mul(Nat.double(Word.to_nat(p,t)),b),Nat.mul(Word.to_nat(p,t),Nat.double(b)),double_mul(Word.to_nat(p,t),
              b)) : {Word.mul.go(n,p,t,WordArithmetic.encode(Nat.double(b),n),WordArithmetic.encode(acc,n)) == WordArithmetic.encode(Nat.add(acc,
                _),n) : Word(n)}
          multiply_go(p,n,t,Nat.double(b),acc)
        case WCon{True{},+t}:
          %WordArithmetic.encode_double(b,n) : {Word.mul.go(n,p,t,_,Word.add(n,WordArithmetic.encode(acc,n),WordArithmetic.encode(b,
                  n))) == WordArithmetic.encode(Nat.add(acc,Nat.add(b,Nat.mul(Nat.double(Word.to_nat(p,t)),b))),n) : Word(n)}
          %Equal.sym(Word(n),Word.add(n,WordArithmetic.encode(acc,n),WordArithmetic.encode(b,n)),WordArithmetic.encode(Nat.add(acc,b),n),
            WordArithmetic.encode_add(acc,b,n)) : {Word.mul.go(n,p,t,WordArithmetic.encode(Nat.double(b),n),
              _) == WordArithmetic.encode(Nat.add(acc,Nat.add(b,Nat.mul(Nat.double(Word.to_nat(p,t)),b))),n) : Word(n)}
          %Equal.sym(Nat,Nat.mul(Nat.double(Word.to_nat(p,t)),b),Nat.mul(Word.to_nat(p,t),Nat.double(b)),double_mul(Word.to_nat(p,t),
              b)) : {Word.mul.go(n,p,t,WordArithmetic.encode(Nat.double(b),n),WordArithmetic.encode(Nat.add(acc,b),
                n)) == WordArithmetic.encode(Nat.add(acc,Nat.add(b,_)),n) : Word(n)}
          %Addition.associative(acc,b,Nat.mul(Word.to_nat(p,t),Nat.double(b))) : {Word.mul.go(n,p,t,WordArithmetic.encode(Nat.double(b),n),
              WordArithmetic.encode(Nat.add(acc,b),n)) == WordArithmetic.encode(_,n) : Word(n)}
          multiply_go(p,n,t,Nat.double(b),Nat.add(acc,b))

def multiply(+n: Nat,+a: Word(n),+b: Word(n)) -> {Word.mul(n,a,b) == WordArithmetic.encode(Nat.mul(Word.to_nat(n,a),Word.to_nat(n,b)),n) : Word(n)}:
  %word_roundtrip(n,b) : {Word.mul.go(n,n,a,_,Word.zero(n)) == WordArithmetic.encode(Nat.mul(Word.to_nat(n,a),Word.to_nat(n,b)),n) : Word(n)}
  multiply_go(n,n,a,Word.to_nat(n,b),0n)

def native_multiply(+a: U32,+b: U32) -> {U32.mul(a,b) == U32.from_nat(Nat.mul(U32.to_nat(a),U32.to_nat(b))) : U32}:
  match a b:
    case U32{+x} U32{+y}:
      %Equal.sym(U32,U32.from_nat(Nat.mul(Word.to_nat(32n,x),Word.to_nat(32n,y))),U32{WordArithmetic.encode(Nat.mul(Word.to_nat(32n,x),
              Word.to_nat(32n,y)),32n)},WordArithmetic.native_encoding(Nat.mul(Word.to_nat(32n,x),Word.to_nat(32n,y)))) : {U32{Word.mul(32n,x,
            y)} == _ : U32}
      %multiply(32n,x,y) : {U32{Word.mul(32n,x,y)} == U32{_} : U32}
      {==}

def bounded_multiply(+a: U32,+b: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},h: {Nat.is_le(Nat.mul(U32.to_nat(a),
        U32.to_nat(b)),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(U32.mul(a,b)) == Nat.mul(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  %Equal.sym(U32,U32.mul(a,b),U32.from_nat(Nat.mul(U32.to_nat(a),U32.to_nat(b))),native_multiply(a,b)) : {U32.to_nat(_) == Nat.mul(U32.to_nat(a),
      U32.to_nat(b)) : Nat}
  Bounds.roundtrip(Nat.mul(U32.to_nat(a),U32.to_nat(b)),cap,cap_ok,h)

# Encoding respects products even when the natural operands exceed a word.
def encoded_product_roundtrip(+value: Nat,remaining: Nat,+width: Nat) ->
  {WordArithmetic.encode(Nat.mul(Word.to_nat(width,WordArithmetic.encode(value,width)),remaining),width) == WordArithmetic.encode(Nat.mul(value,remaining),width) : Word(width)}:
  match remaining:
    case 0n:
      Equal.cong(Nat,Word(width),n => WordArithmetic.encode(n,width),Nat.mul(Word.to_nat(width,WordArithmetic.encode(value,width)),0n),Nat.mul(value,0n),
        Equal.trans(Nat,Nat.mul(Word.to_nat(width,WordArithmetic.encode(value,width)),0n),0n,Nat.mul(value,0n),RowMajor.multiply_zero_right(Word.to_nat(width,WordArithmetic.encode(value,width))),Equal.sym(Nat,Nat.mul(value,0n),0n,RowMajor.multiply_zero_right(value))))
    case 1n+ +rest:
      +decoded = Word.to_nat(width,WordArithmetic.encode(value,width))
      %Equal.sym(Nat,Nat.mul(decoded,1n+rest),Nat.add(decoded,Nat.mul(decoded,rest)),RowMajor.multiply_successor_right(decoded,rest)) :
        {WordArithmetic.encode(_,width) == WordArithmetic.encode(Nat.mul(value,1n+rest),width) : Word(width)}
      %Equal.sym(Nat,Nat.mul(value,1n+rest),Nat.add(value,Nat.mul(value,rest)),RowMajor.multiply_successor_right(value,rest)) :
        {WordArithmetic.encode(Nat.add(decoded,Nat.mul(decoded,rest)),width) == WordArithmetic.encode(_,width) : Word(width)}
      %WordArithmetic.encode_add(decoded,Nat.mul(decoded,rest),width) :
        {_ == WordArithmetic.encode(Nat.add(value,Nat.mul(value,rest)),width) : Word(width)}
      %WordArithmetic.encode_add(value,Nat.mul(value,rest),width) :
        {Word.add(width,WordArithmetic.encode(decoded,width),WordArithmetic.encode(Nat.mul(decoded,rest),width)) == _ : Word(width)}
      %Equal.sym(Word(width),WordArithmetic.encode(decoded,width),WordArithmetic.encode(value,width),word_roundtrip(width,WordArithmetic.encode(value,width))) :
        {Word.add(width,_,WordArithmetic.encode(Nat.mul(decoded,rest),width)) == Word.add(width,WordArithmetic.encode(value,width),WordArithmetic.encode(Nat.mul(value,rest),width)) : Word(width)}
      %Equal.sym(Word(width),WordArithmetic.encode(Nat.mul(decoded,rest),width),WordArithmetic.encode(Nat.mul(value,rest),width),encoded_product_roundtrip(value,rest,width)) :
        {Word.add(width,WordArithmetic.encode(value,width),_) == Word.add(width,WordArithmetic.encode(value,width),WordArithmetic.encode(Nat.mul(value,rest),width)) : Word(width)}
      {==}

def encoded_multiply(+left: Nat,+right: Nat,+width: Nat) ->
  {Word.mul(width,WordArithmetic.encode(left,width),WordArithmetic.encode(right,width)) == WordArithmetic.encode(Nat.mul(left,right),width) : Word(width)}:
  +a = Word.to_nat(width,WordArithmetic.encode(left,width))
  +b = Word.to_nat(width,WordArithmetic.encode(right,width))
  %Equal.sym(Word(width),Word.mul(width,WordArithmetic.encode(left,width),WordArithmetic.encode(right,width)),WordArithmetic.encode(Nat.mul(a,b),width),multiply(width,WordArithmetic.encode(left,width),WordArithmetic.encode(right,width))) :
    {_ == WordArithmetic.encode(Nat.mul(left,right),width) : Word(width)}
  %Equal.sym(Word(width),WordArithmetic.encode(Nat.mul(a,b),width),WordArithmetic.encode(Nat.mul(left,b),width),encoded_product_roundtrip(left,b,width)) :
    {_ == WordArithmetic.encode(Nat.mul(left,right),width) : Word(width)}
  %Equal.sym(Nat,Nat.mul(left,b),Nat.mul(b,left),RowMajor.multiply_commutative(left,b)) :
    {WordArithmetic.encode(_,width) == WordArithmetic.encode(Nat.mul(left,right),width) : Word(width)}
  %Equal.sym(Word(width),WordArithmetic.encode(Nat.mul(b,left),width),WordArithmetic.encode(Nat.mul(right,left),width),encoded_product_roundtrip(right,left,width)) :
    {_ == WordArithmetic.encode(Nat.mul(left,right),width) : Word(width)}
  Equal.cong(Nat,Word(width),n => WordArithmetic.encode(n,width),Nat.mul(right,left),Nat.mul(left,right),RowMajor.multiply_commutative(right,left))

def native_encoded_multiply(+left: Nat,+right: Nat) ->
  {(U32.from_nat(left) * U32.from_nat(right) : U32) == U32.from_nat(Nat.mul(left,right)) : U32}:
  %Equal.sym(U32,U32.from_nat(left),U32{WordArithmetic.encode(left,32n)},WordArithmetic.native_encoding(left)) :
    {(_ * U32.from_nat(right) : U32) == U32.from_nat(Nat.mul(left,right)) : U32}
  %Equal.sym(U32,U32.from_nat(right),U32{WordArithmetic.encode(right,32n)},WordArithmetic.native_encoding(right)) :
    {(U32{WordArithmetic.encode(left,32n)} * _ : U32) == U32.from_nat(Nat.mul(left,right)) : U32}
  %Equal.sym(U32,U32.from_nat(Nat.mul(left,right)),U32{WordArithmetic.encode(Nat.mul(left,right),32n)},WordArithmetic.native_encoding(Nat.mul(left,right))) :
    {U32{Word.mul(32n,WordArithmetic.encode(left,32n),WordArithmetic.encode(right,32n))} == _ : U32}
  Equal.cong(Word(32n),U32,word => U32{word},Word.mul(32n,WordArithmetic.encode(left,32n),WordArithmetic.encode(right,32n)),WordArithmetic.encode(Nat.mul(left,right),32n),encoded_multiply(left,right,32n))
