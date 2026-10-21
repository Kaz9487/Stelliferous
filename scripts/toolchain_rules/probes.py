"""Small Bend programs that pin down the rules in docs/toolchain_rules.md.

CHECKER maps a probe name to (expected, source). expected is "pass" or a
fragment of the message the official checker must print. A source can also
be a fixture dictionary with main text and relative files.
CODEGEN maps a probe name to (present, absent, shared, source): strings that
must and must not occur in the generated C, and whether "none", "some" or
"all" constructors are shared. verify.py writes each source under out/ after
an `import Base` line and runs the pinned compiler on it.
"""

# Entry shared by the code generation probes: it calls run(n) with a runtime value.
MAIN = '''
def pick(zero: Bool) -> String:
  match zero:
    case True{}: "zero"
    case False{}: "other"

def done(value: U32) -> IO(Unit):
  IO.print(pick(U32.is_zero(value)))

def main() -> IO(Unit):
  IO.bind(Nat, Unit, IO.now(), n => done(run(n)))
'''

CHECKER = {
    'match_order_bad': ("can't be matched in this position", '''
def f(a: Bool, b: Bool) -> Bool:
  match b:
    case True{}:
      match a:
        case True{}: True{}
        case False{}: False{}
    case False{}: False{}
'''),
    'match_order_ok': ('pass', '''
def f(a: Bool, b: Bool) -> Bool:
  match a:
    case True{}:
      match b:
        case True{}: True{}
        case False{}: False{}
    case False{}: False{}
'''),
    'match_after_let': ("can't be matched in this position", '''
def k(a: Bool, b: Bool) -> Bool:
  x = a
  match b:
    case True{}: x
    case False{}: x
'''),
    'match_computed': ('a match cannot scrutinize a computed value', '''
def flip(a: Bool) -> Bool:
  match a:
    case True{}: False{}
    case False{}: True{}

def g(a: Bool) -> Bool:
  match flip(a):
    case True{}: True{}
    case False{}: False{}
'''),
    'match_local': ('a match cannot scrutinize a local binder', '''
def flip(a: Bool) -> Bool:
  match a:
    case True{}: False{}
    case False{}: True{}

def h(a: Bool) -> Bool:
  x = flip(a)
  match x:
    case True{}: True{}
    case False{}: False{}
'''),
    'erased_scrutinee': ('a live scrutinee (a - scrutinee matches only in a dead region)', '''
def era(-a: Bool) -> Bool:
  match a:
    case True{}: True{}
    case False{}: False{}
'''),
    'use_twice': ('x (consumed more than once)', '''
def dup(x: Nat) -> Nat:
  Nat.add(x, x)
'''),
    'use_twice_plus': ('pass', '''
def dup(+x: Nat) -> Nat:
  Nat.add(x, x)
'''),
    'plus_needs_data': ('can be used many times, so its type must be Data', '''
def keep(+f: Nat -> Nat) -> Nat:
  f(0n)
'''),
    'tilde_not_leading': ('a plain binder (only leading binders take ~)', '''
def t(a: Nat, ~b: Nat) -> Nat:
  a
'''),
    'tilde_on_plain_call': ('expected : a term', '''
def same(A: Type, x: A) -> A:
  x

def use() -> Nat:
  same(~Nat, 1n)
'''),
    'law_template_ok': ('pass', '''
law same:
  for ~f: Nat -> Nat
  for +x: Nat
  {f(x) == f(x) : Nat}
def same(f, x):
  {==}

def keep(n: Nat) -> Nat:
  n

def use(+x: Nat) -> {keep(x) == keep(x) : Nat}:
  same(~keep, x)
'''),
    'non_decreasing': ('a decreasing self-call', '''
def spin(n: Nat) -> Nat:
  spin(n)
'''),
    'non_tail_recursion_ok': ('pass', '''
def count(n: Nat) -> Nat:
  match n:
    case 0n: 0n
    case 1n+p: 1n+count(p)
'''),
    'lambda_self_call_ok': ('pass', '''
def then(+result: U32, next: U32 -> U32) -> U32:
  next(result)

def loop(n: Nat) -> U32:
  match n:
    case 0n: 0
    case 1n+p: then(5, value => loop(p))
'''),
    'lambda_self_call_bad': ('a decreasing self-call', '''
def then(+result: U32, next: U32 -> U32) -> U32:
  next(result)

def loop(n: Nat) -> U32:
  then(5, value => loop(n))
'''),
    'rewrite_direction_ok': ('pass', '''
def rw(+a: Nat, +b: Nat, e: {a == b : Nat}, base: {Nat.add(a, 0n) == a : Nat}) -> {Nat.add(b, 0n) == b : Nat}:
  %e : {Nat.add(_, 0n) == _ : Nat}
  base
'''),
    'rewrite_direction_bad': ('expected : {Nat.add(b, 0n) == b : Nat}', '''
def rw(+a: Nat, +b: Nat, e: {b == a : Nat}, base: {Nat.add(a, 0n) == a : Nat}) -> {Nat.add(b, 0n) == b : Nat}:
  %e : {Nat.add(_, 0n) == _ : Nat}
  base
'''),
    'nat_compute_10000000': ('pass', '''
def huge() -> {Nat.is_le(10000000n, 20000000n) == True{} : Bool}:
  {==}
'''),
    'u32_compute_large': ('pass', '''
def words() -> {U32.is_le(1073741824, 2147483648) == True{} : Bool}:
  {==}
'''),
    'to_nat_large_compare': ('pass', '''
def bound() -> {Nat.is_le(1n, U32.to_nat(1073741824)) == True{} : Bool}:
  {==}
'''),
    'to_nat_equal_100000': ('pass', '''
def same() -> {Nat.is_le(U32.to_nat(100000), U32.to_nat(100000)) == True{} : Bool}:
  {==}
'''),
    'to_nat_symbolic_bound': ('pass', '''
def bound(+x: Nat, room: {Nat.is_le(x, U32.to_nat(1073741824)) == True{} : Bool}) -> {Nat.is_le(x, U32.to_nat(1073741824)) == True{} : Bool}:
  room
'''),
    'to_nat_mixed_literal': ('the machine stack overflowed', '''
def mixed(+x: Nat, room: {Nat.is_le(x, U32.to_nat(1073741824)) == True{} : Bool}) -> {Nat.is_le(x, 1073741824n) == True{} : Bool}:
  room
'''),
}

# After a match on box the statement reads size(Box{n}); a term spelled with the field n differs from it.
FIELD_SPELLING = '''
type Box is Data:
  Box{size: Nat}

def size(box: Box) -> Nat:
  Box{size} = box
  size

def keep(+x: Nat, room: {Nat.is_le(x, BOUND) == True{} : Bool}) -> {Nat.is_le(x, BOUND) == True{} : Bool}:
  room

def spelled(+box: Box, room: {Nat.is_le(size(box), BOUND) == True{} : Bool}) -> {Nat.is_le(size(box), BOUND) == True{} : Bool}:
  Box{n} = box
  keep(TERM, room)
'''
CHECKER['field_spelling_overflow'] = ('the machine stack overflowed', FIELD_SPELLING.replace('BOUND', 'U32.to_nat(1073741824)').replace('TERM', 'n'))
CHECKER['projection_spelling_ok'] = ('pass', FIELD_SPELLING.replace('BOUND', 'U32.to_nat(1073741824)').replace('TERM', 'size(box)'))
CHECKER['projection_let_ok'] = ('pass', FIELD_SPELLING.replace('BOUND', 'U32.to_nat(1073741824)').replace('  keep(TERM, room)', '  +m = size(box)\n  keep(m, room)'))
CHECKER['small_field_spelling_ok'] = ('pass', FIELD_SPELLING.replace('BOUND', '3n').replace('TERM', 'n'))

# A derived accessor that matches its record is stuck on an abstract record; one defined through the base accessor unfolds.
ACCESSORS = '''
type Box is Data:
  Box{size: Nat,extra: Nat}

def size(box: Box) -> Nat:
  Box{size,extra} = box
  size

def through(box: Box) -> Nat:
  size(box)

def matched(box: Box) -> Nat:
  Box{size,extra} = box
  size

def same(+box: Box, known: {DERIVED(box) == 0n : Nat}) -> {size(box) == 0n : Nat}:
  known
'''
# A template nobody calls is still checked, with each ~ parameter an opaque constant.
CHECKER['template_uncalled_checked'] = ('observed : 0n', '''
def wrong(~n: Nat) -> {n == 0n : Nat}:
  {==}
''')
CHECKER['template_function_uncalled_checked'] = ('observed : 0n', '''
def wrong(~f: Nat -> Nat, +n: Nat) -> {f(n) == 0n : Nat}:
  {==}
''')
CHECKER['accessor_through_ok'] = ('pass', ACCESSORS.replace('DERIVED', 'through'))
CHECKER['accessor_matched_bad'] = ('observed : {matched(box) == 0n : Nat}', ACCESSORS.replace('DERIVED', 'matched'))



def evidence_chain(steps):
    """One declaration that passes the same evidence through a lemma once per step."""
    return ('''
def step(+n: Nat,evidence: {n == n : Nat}) -> {n == n : Nat}:
  evidence

def chain(+n: Nat,evidence: {n == n : Nat}) -> {n == n : Nat}:
''' + '  evidence = step(n,evidence)\n' * steps + '  evidence\n')


CHECKER['declaration_4000_steps'] = ('pass', evidence_chain(4000))
CHECKER['declaration_12000_steps'] = ('the machine stack overflowed', evidence_chain(12000))

def chain(steps):
    """A def with one let per step; enough steps make the native function DFAR."""
    lets = ''.join(f'  x{step} = (x{step - 1} * 3 + {step} : U32)\n' for step in range(1, steps))
    return (f'\ndef body(a: U32) -> U32:\n  x0 = (a + 1 : U32)\n{lets}  (x{steps - 1} + 5 : U32)\n\n'
            'def run(n: Nat) -> U32:\n  body(U32.from_nat(n))\n')


CODEGEN = {
    'shapes': (['FID_NON_TAIL_K', 'define FID_MIXED ', 'FID_FORKS_J', 'define FID_KID ', 'WL_AGAIN(FID_KID)'], ['FID_TAIL_ONLY', 'FID_WRAPPER'], 'none', '''
def tail_only(n: Nat, acc: U32) -> U32:
  match n:
    case 0n: acc
    case 1n+p: tail_only(p, (acc + 1 : U32))

def wrapper(n: Nat) -> U32:
  (tail_only(n, 0) + 3 : U32)

def non_tail(n: Nat) -> U32:
  match n:
    case 0n: 0
    case 1n+p: (1 + non_tail(p) : U32)

def mixed(+n: Nat) -> U32:
  (wrapper(n) + non_tail(n) : U32)

def kid(n: Nat, acc: U32) -> U32:
  match n:
    case 0n: acc
    case 1n+p: kid(p, (acc + 2 : U32))

def forks(+n: Nat) -> U32:
  left right = kid(n, 0) kid(n, 1)
  (left + right : U32)

def run(+n: Nat) -> U32:
  (mixed(n) + forks(n) : U32)
''' + MAIN),
    'fork_erased_last': (['FID_TREE_J'], [], 'none', '''
def tree(depth: Nat, +index: U32, -tag: Nat) -> U32:
  match depth:
    case 0n: (index * 3 + 1 : U32)
    case 1n+ +rest:
      left right = tree(rest, (index * 2 : U32), tag) tree(rest, (index * 2 + 1 : U32), tag)
      (left + right : U32)

def run(n: Nat) -> U32:
  tree(n, 1, 0n)
''' + MAIN),
    'inline_small': (['INLINE Term spin_'], ['FAR Term spin_'], 'none', '''
def body(a: U32) -> U32:
  x0 = (a + 1 : U32)
  x1 = (x0 * 3 : U32)
  (x1 + 5 : U32)

def run(n: Nat) -> U32:
  body(U32.from_nat(n))
''' + MAIN),
    'far_large': (['DFAR Term spin_'], [], 'none', chain(400) + MAIN),
    'share_none': ([], [], 'none', '''
def total(xs: List<&2,U32>, acc: U32) -> U32:
  match xs:
    case Nil{}: acc
    case Con{head,tail}: total(tail, (acc + head : U32))

def build(n: Nat, acc: List<&2,U32>) -> List<&2,U32>:
  match n:
    case 0n: acc
    case 1n+p: build(p, 7 <> acc)

def run(n: Nat) -> U32:
  total(build(n, []), 0)
''' + MAIN),
    'share_list': ([], [], 'some', '''
def total(xs: List<&2,U32>, acc: U32) -> U32:
  match xs:
    case Nil{}: acc
    case Con{head,tail}: total(tail, (acc + head : U32))

def build(n: Nat, acc: List<&2,U32>) -> List<&2,U32>:
  match n:
    case 0n: acc
    case 1n+p: build(p, 7 <> acc)

def both(+xs: List<&2,U32>) -> U32:
  (total(xs, 0) + total(xs, 1) : U32)

def run(n: Nat) -> U32:
  both(build(n, []))
''' + MAIN),
    'share_live_parameter': ([], [], 'some', '''
def total(xs: List<&2,U32>, acc: U32) -> U32:
  match xs:
    case Nil{}: acc
    case Con{head,tail}: total(tail, (acc + head : U32))

def build(n: Nat, acc: List<&2,U32>) -> List<&2,U32>:
  match n:
    case 0n: acc
    case 1n+p: build(p, 7 <> acc)

def twice(A: Data, +x: A) -> A & A:
  (x, x)

def first(pair: List<&2,U32> & List<&2,U32>) -> List<&2,U32>:
  (left, right) = pair
  left

def run(n: Nat) -> U32:
  total(first(twice(List<&2,U32>, build(n, []))), 0)
''' + MAIN),
    'share_template_parameter': ([], [], 'some', '''
def total(xs: List<&2,U32>, acc: U32) -> U32:
  match xs:
    case Nil{}: acc
    case Con{head,tail}: total(tail, (acc + head : U32))

def build(n: Nat, acc: List<&2,U32>) -> List<&2,U32>:
  match n:
    case 0n: acc
    case 1n+p: build(p, 7 <> acc)

def twice(~A: Data, +x: A) -> A & A:
  (x, x)

def first(pair: List<&2,U32> & List<&2,U32>) -> List<&2,U32>:
  (left, right) = pair
  left

def run(n: Nat) -> U32:
  total(first(twice(~List<&2,U32>, build(n, []))), 0)
''' + MAIN),
    'share_family_match_live_index': ([], [], 'some', '''
def total(xs: List<&2,U32>, acc: U32) -> U32:
  match xs:
    case Nil{}: acc
    case Con{head,tail}: total(tail, (acc + head : U32))

def build(n: Nat, acc: List<&2,U32>) -> List<&2,U32>:
  match n:
    case 0n: acc
    case 1n+p: build(p, 7 <> acc)

def Family(n: Nat) -> Data:
  match n:
    case 0n: List<&2,U32>
    case 1n+p: List<&2,U32>

def twice(+n: Nat, +x: Family(n)) -> Family(n) & Family(n):
  (x, x)

def first(-n: Nat, pair: Family(n) & Family(n)) -> Family(n):
  (left, right) = pair
  left

def cast(+n: Nat, xs: List<&2,U32>) -> Family(n):
  match n:
    case 0n: xs
    case 1n+p: xs

def back(+n: Nat, xs: Family(n)) -> List<&2,U32>:
  match n:
    case 0n: xs
    case 1n+p: xs

def run(+n: Nat) -> U32:
  total(back(n, first(n, twice(n, cast(n, build(n, []))))), 0)
''' + MAIN),
    'call_template': (['INLINE Term spin_'], ['define FID_RUN ', 'define FID_INCREMENT ', 'term_clo(FID_RUN_C'], 'none', '''
def increment(value: U32) -> U32:
  (value + 1 : U32)

def apply(~operation: U32 -> U32, value: U32) -> U32:
  operation(operation(value))

def run(n: Nat) -> U32:
  apply(~increment, U32.from_nat(n))
''' + MAIN),
    'call_closure': (['define FID_RUN ', 'term_clo(FID_RUN_C'], [], 'none', '''
def increment(value: U32) -> U32:
  (value + 1 : U32)

def apply(operation: U32 -> U32, value: U32) -> U32:
  operation(value)

def run(n: Nat) -> U32:
  apply(increment, U32.from_nat(n))
''' + MAIN),
    'leaf_task': (['define FID_LEAF ', 'define FID_TREE '], ['define FID_HEAVY '], 'none', '''
# The leaf work is a fork-free def called directly by the parallel let, so
# the runtime keeps each leaf as its own task for the work turn.
def heavy(n: Nat, acc: U32) -> U32:
  match n:
    case 0n: acc
    case 1n+p: heavy(p, (acc * 1664525 + 1013904223 : U32))

def leaf(n: Nat, seed: U32) -> U32:
  heavy(n, seed)

def tree(depth: Nat, +n: Nat, +seed: U32) -> U32:
  match depth:
    case 0n: heavy(n, seed)
    case 1n:
      left right = leaf(n, (seed * 2 : U32)) leaf(n, (seed * 2 + 1 : U32))
      (left + right : U32)
    case 2n+ +rest:
      left right = tree(1n+rest, n, (seed * 2 : U32)) tree(1n+rest, n, (seed * 2 + 1 : U32))
      (left + right : U32)

def depth_of(leaves: U32) -> Nat:
  match leaves:
    case 16: 4n
    case 8: 3n
    case _: 2n

def run(+now: Nat) -> U32:
  tree(depth_of(U32.from_nat(Nat.mod(now, 1n))), U32.to_nat(400000000), 1)

def main() -> IO(Unit):
  IO.bind(Nat, Unit, IO.now(), now => IO.print(U32.show(run(now))))
'''),
    'leaf_inline': (['define FID_TREE '], ['define FID_HEAVY ', 'define FID_LEAF '], 'none', '''
# The leaf work is a flat loop called inside the forking recursion.
def heavy(n: Nat, acc: U32) -> U32:
  match n:
    case 0n: acc
    case 1n+p: heavy(p, (acc * 1664525 + 1013904223 : U32))

def tree(depth: Nat, +n: Nat, +seed: U32) -> U32:
  match depth:
    case 0n: heavy(n, seed)
    case 1n+ +rest:
      left right = tree(rest, n, (seed * 2 : U32)) tree(rest, n, (seed * 2 + 1 : U32))
      (left + right : U32)

def depth_of(leaves: U32) -> Nat:
  match leaves:
    case 16: 4n
    case 8: 3n
    case _: 2n

def run(+now: Nat) -> U32:
  tree(depth_of(U32.from_nat(Nat.mod(now, 1n))), U32.to_nat(400000000), 1)

def main() -> IO(Unit):
  IO.bind(Nat, Unit, IO.now(), now => IO.print(U32.show(run(now))))
'''),
}

# Rule H4: whether the runtime treats a task of a codegen probe as fork-free.
TASKS = {
    'leaf_task': {'FID_LEAF': 'fork-free', 'FID_TREE': 'may fork'},
    'leaf_inline': {'FID_TREE': 'may fork'},
}

CHECKER.update({
    'match_constructor': ('already a constructor', """
def f() -> Bool:
  match True{}:
    case True{}: True{}
    case False{}: False{}
"""),
    'match_variable_column': ('pass', """
def f(a: Bool, b: Bool) -> Bool:
  match b:
    case other:
      match a:
        case True{}: other
        case False{}: False{}
"""),
    'match_first_row': ('pass', """
def f(a: Bool) -> Bool:
  match a:
    case _: False{}
    case True{}: True{}

def check() -> {f(True{}) == False{} : Bool}:
  {==}
"""),
    'list_pattern': ('pass', """
def head(xs: List<&2,U32>) -> U32:
  match xs:
    case Nil{}: 0
    case Con{head,tail}: head
"""),
})

CHECKER.update({
    'usage_once_to_many': ('pass', '''
def twice(+n: Nat) -> Nat:
  Nat.add(n,n)
def f(n: Nat) -> Nat:
  twice(n)
'''),
    'usage_many_field': ('pass', '''
type Box is Data:
  Box{+n: Nat}
def f(box: Box) -> Nat:
  Box{n} = box
  Nat.add(n,n)
'''),
    'usage_proof_reuse': ('pass', '''
def f(+n: Nat, +e: {n == n : Nat}) -> {n == n : Nat} & {n == n : Nat}:
  (e,e)
'''),
    'usage_branches': ('pass', '''
def f(b: Bool, n: Nat) -> Nat:
  match b:
    case True{}: n
    case False{}: n
'''),
})

CHECKER.update({
    'template_runtime_rejected': ('a template applied to closed ~ arguments', '''
def take(~n: Nat) -> Nat:
  n
def f(n: Nat) -> Nat:
  take(~n)
'''),
    'forward_live_rejected': ('a filled definition', '''
def first() -> Nat:
  later()
def later() -> Nat:
  0n
'''),
    'law_unfolds': ('pass', '''
law value:
  Nat
def value():
  3n
def check() -> {value() == 3n : Nat}:
  {==}
'''),
    'stuck_same_head': ('pass', '''
def f(n: Nat) -> Nat:
  match n:
    case 0n: 0n
    case 1n+p: p
def check(+n: Nat) -> {f(n) == f(n) : Nat}:
  {==}
'''),
})

CHECKER.update({
    'constructor_inference_bad': ('an annotated term (cannot infer)', '''
def f() -> Bool:
  x = True{}
  x
'''),
    'constructor_annotated': ('pass', '''
def f() -> Bool:
  +x = {True{} : Bool}
  x
'''),
    'function_eta': ('pass', '''
def check(+n: Nat) -> {(x => Nat.add(n,x)) == Nat.add(n) : Nat -> Nat}:
  {==}
'''),
    'kind_data_to_type': ('pass', '''
def kind(A: Type) -> Type:
  A
def good() -> Type:
  kind(Nat)
'''),
})

CHECKER.update({
    'abstract_nat_bound': ('pass', '''
def f(+x: Nat, +bound: Nat, e: {Nat.is_le(x,bound) == True{} : Bool}) -> {Nat.is_le(x,bound) == True{} : Bool}:
  e
'''),
    'literal_constructor_equal': ('pass', '''
def f() -> {3n == 1n+2n : Nat}:
  {==}
'''),
})

# Native loops retain cooperative error polling. Host worklist dispatch uses tail calls.
CODEGEN['shapes'][0].extend(['u32 wpoll = 0;', 'WL_SPIN', 'WL_DYN((u32)STK(0))'])

CODEGEN.update({
    'template_record': (['INLINE Term spin_'], [], 'none', """
type Counts is Data:
  Counts{+count: U32}
def count(n: Nat, acc: U32) -> U32:
  match n:
    case 0n: acc
    case 1n+p: count(p, (acc * 1664525 + 1013904223 : U32))
def counts() -> Counts:
  Counts{count(400000n, 1)}
def get(c: Counts) -> U32:
  Counts{n} = c
  n
def twice(~c: Counts, x: U32) -> U32:
  (get(c) + get(c) + x : U32)
def run(n: Nat) -> U32:
  twice(~counts(), U32.from_nat(n))
""" + MAIN),
    'flat_record': (['INLINE Term spin_'], ['CID_PAIR'], 'none', """
type Words is Data:
  Words{+a: U32,+b: U32}
def total(+p: Words) -> U32:
  Words{a,b} = p
  (a + b + a : U32)
def run(n: Nat) -> U32:
  total(Words{U32.from_nat(n), 7})
""" + MAIN),
})

CODEGEN.update({
    'array_blocks': (['blk_copy(e,', 'blk_half(e,', 'blk_at('], [], 'none', """
def take_value(pair: Array<U32> & U32) -> U32:
  (a,x) = pair
  x
def split(a: Array<U32>) -> U32:
  match a:
    case ALeaf{x}: x
    case ANode{left,right}:
      take_value(Array.get(U32,left,0))
def copies(pair: Array<U32> & Array<U32>) -> U32:
  (left,right) = pair
  (split(left) + split(right) : U32)
def run(n: Nat) -> U32:
  copies(Array.clone(U32,Array.new(U32,3n,U32.from_nat(n))))
""" + MAIN),
    'file_bytes': (['file_read_bytes_pack', 'io_list(e, w->data, w->size)'], [], 'none', """
def opened(r: Result<&1,&1,U32 & String,File>) -> IO(Unit):
  match r:
    case Fail{error}: IO.print("open failed")
    case Done{file}:
      IO.bind(File & Result<&1,&1,U32 & String,List<&2,U32>>,Unit,
        File.read_bytes(file,16), result => IO.print("read"))
def main() -> IO(Unit):
  IO.bind(Result<&1,&1,U32 & String,File>,Unit,File.open("data.bin","r"),opened)
"""),
})

CHECKER.update({
    'array_index_wrap': ('pass', '''
def second(pair: Array<U32> & U32) -> U32:
  (a,x) = pair
  x
def f() -> {second(Array.get(U32,ANode{ALeaf{3},ALeaf{7}},3)) == 7 : U32}:
  {==}
'''),
})

CHECKER.update({
    'list_pattern_postfix': ('expected : a term', '''
def f(xs: List<&2,U32>, ys: List<&2,U32>) -> U32:
  match xs ys:
    case tail []: 0
    case _ _: 1
'''),
    'operator_annotation': ('pass', '''
def f(a: U32,b: U32) -> U32:
  (a + b : U32)
'''),
})

CODEGEN['template_keys'] = (['FID_LOOP_0', 'FID_LOOP_1'], [], 'none', """
def loop(~tag: Nat, n: Nat) -> U32:
  match n:
    case 0n: U32.from_nat(tag)
    case 1n+p: (1 + loop(~tag,p) : U32)
def run(+n: Nat) -> U32:
  (loop(~1n,n) + loop(~Nat.add(0n,1n),n) : U32)
""" + MAIN)

# These spelling diagnostics are compared with the official-checker probes.
LINT = {
    'match_after_let': {'A2'},
    'match_computed': {'A3'},
    'list_pattern_postfix': {'A7'},
    'tilde_not_leading': {'C1'},
    'tilde_on_plain_call': {'C2'},
    'constructor_inference_bad': {'D4'},
    'to_nat_symbolic_bound': {'E1'},
    'usage_branches': set(),
    'use_twice': set(),
    'match_variable_column': set(),
}

# CLI completion and trust checks are separate from local term checking.
CHECKER.update({
    'template_marks_omitted': ('pass', '''
def choose(~a: Nat, ~b: Nat, n: Nat) -> Nat:
  Nat.add(a, Nat.add(b,n))
def use() -> Nat:
  choose(1n,2n,3n)
'''),
    'template_marks_partial': ('pass', '''
def choose(~a: Nat, ~b: Nat, n: Nat) -> Nat:
  Nat.add(a, Nat.add(b,n))
def use() -> Nat:
  choose(~1n,2n,3n)
'''),
    'todo_rejected': ('TODO found', '''
def unfinished() -> Nat:
  ?TODO
'''),
    'unfilled_law_rejected': ('TODO found', '''
law unfinished:
  Nat
'''),
    'unsafe_type_dependency': ('rely on unsafe or foreign code', '''
@unsafe
def hidden() -> Data:
  Nat
law claim:
  {hidden() == hidden() : Type}
def claim():
  {==}
'''),
    'array_fork_trust': ('relies on unsafe or foreign code', '''
def split(a: Array<U32>) -> Array<U32> & Array<U32>:
  Array.fork(U32,a)
'''),
    'parallel_binder_scope': ('a defined name', '''
def use() -> Nat:
  x y = {1n : Nat} Nat.add(x,1n)
  Nat.add(x,y)
'''),
    'match_missing_arm': ('cases for False', '''
def use(b: Bool) -> Nat:
  match b:
    case True{}: 1n
'''),
    'empty_live': ('pass', '''
type Void is Data:
def eliminate(v: Void) -> Nat:
  match v:
'''),
    'empty_erased': ('a live scrutinee', '''
type Void is Data:
def eliminate(-v: Void) -> Nat:
  match v:
'''),
})

# Expected (exit success, output fragment) for independent execution routes.
# Native runs use verify.py's recorded clang flags, not CLI build defaults.
EXECUTION = {
    'word_edges': ('''
def main() -> U32:
  U32.add(U32.div(9,0),U32.add(U32.mod(9,0),U32.shln(7,32n)))
''', {mode: (True, '9') for mode in ('eval', 'js', 'c')}),
    'word_wrap': ('''
def main() -> U32:
  U32.add(4294967295,2)
''', {mode: (True, '1') for mode in ('eval', 'js', 'c')}),
    'nat_runtime_limit': ('''
def main() -> Nat:
  Nat.mul(U32.to_nat(4294967295),U32.to_nat(4294967295))
''', {mode: (False, 'a Nat past the largest immediate 2^48-1') for mode in ('js', 'c')}),
    'unsafe_codegen': ('''
@unsafe
def keep(n: Nat) -> Nat:
  n
def main() -> Nat:
  keep(3n)
''', {'check': (False, 'rely on unsafe or foreign code'),
      'eval': (True, '3n'), 'js': (True, '3n'), 'c': (True, '3n')}),
    'array_wrap_runtime': ('''
def second(pair: Array<U32> & U32) -> U32:
  (a,x) = pair
  x
def main() -> U32:
  second(Array.get(U32,ANode{ALeaf{3},ALeaf{7}},3))
''', {mode: (True, '7') for mode in ('eval', 'js', 'c')}),
}

CHECKER['template_marks_gap'] = ('expected : a term', """
def choose(~a: Nat, ~b: Nat, n: Nat) -> Nat:
  Nat.add(a,Nat.add(b,n))
def use() -> Nat:
  choose(1n,~2n,3n)
""")

LINT.update({'template_marks_partial': set(), 'template_marks_omitted': set(),
             'template_marks_gap': {'C2'}})

CODEGEN['share_existential'] = (['term_keep(e,', 'CID_HIDE'], [], 'some', """
type Hidden is Data:
  Hide{-T: Data, +value: T}
def discard(h: Hidden, n: Nat) -> U32:
  match h:
    case Hide{T,x}: U32.from_nat(n)
def retain(h: Hidden, n: Nat) -> Hidden:
  match n:
    case 0n: h
    case 1n+p: retain(h,p)
def duplicate(+h: Hidden, +n: Nat) -> U32:
  a b = retain(h,n) retain(h,n)
  (discard(a,n) + discard(b,n) : U32)
def run(+n: Nat) -> U32:
  duplicate(Hide{List<&2,Nat>,Con{n,Nil{}}},n)
""" + MAIN)

EXECUTION['f32_opaque_native'] = ('''
def main() -> F32:
  F32.add(16777216.0,1.0)
''', {'eval': (True, 'F32.add(16777216.0, 1.0)'), 'js': (True, '16777216.0'),
      'c': (True, '16777216.0')})
CHECKER['f32_opaque_equation'] = ('expected : F32.add(16777216.0, 1.0)', '''
def equality() -> {F32.add(16777216.0,1.0) == 16777216.0 : F32}:
  {==}
''')

BORROW_SCAN = '''
def scan(xs: List<&2,U32>) -> U32:
  match xs:
    case Nil{}: 0
    case Con{x,tail}: 1
'''
CODEGEN['own_single'] = ([], [], 'none', BORROW_SCAN + '''
def run(n: Nat) -> U32:
  scan(Con{U32.from_nat(n),Con{2,Nil{}}})
''' + MAIN)
CODEGEN['borrow_repeated'] = ([], [], 'none', BORROW_SCAN + '''
def run(n: Nat) -> U32:
  +xs = {Con{U32.from_nat(n),Con{2,Nil{}}} : List<&2,U32>}
  (scan(xs) + scan(xs) : U32)
''' + MAIN)

# Check emitted native/segment bodies, excluding runtime helper definitions.
SEGMENTS = {
    'own_single': {'spin_0': (['ctr_take(', 'spare_free(', 'term_sink('], ['term_peek('])},
    'borrow_repeated': {'spin_0': (['term_peek('], ['ctr_take(', 'spare_free(', 'term_sink('])},
}
CONSTRUCTORS = {
    'share_existential': {'CID_HIDE': 1, 'CID_WCON': 1, 'CID_TRUE': 1,
                          'CID_FALSE': 1, 'CID_CON': 0},
}

CODEGEN['share_concrete'] = (['term_keep(e,', 'CID_HIDE'], [], 'some',
    CODEGEN['share_existential'][3]
        .replace('Hide{-T: Data, +value: T}', 'Hide{+value: List<&2,Nat>}')
        .replace('Hide{T,x}', 'Hide{x}')
        .replace('Hide{List<&2,Nat>,Con{n,Nil{}}}', 'Hide{Con{n,Nil{}}}'))
CONSTRUCTORS['share_concrete'] = {'CID_HIDE': 1, 'CID_CON': 1, 'CID_WCON': 0,
                                'CID_TRUE': 0, 'CID_FALSE': 0}

CHECKER['import_alias_identity'] = ('pass', {
    'files': {'item.bend': 'import Base\ntype Item is Data:\n  Item{+n: Nat}\n'},
    'main': '''
import ./item.bend as A
import ./item.bend as B
def same(x: A.Item) -> B.Item:
  x
''',
})
CHECKER['import_cycle'] = ('cycle', {
    'files': {'loop.bend': 'import ./main.bend as Main\n'},
    'main': 'import ./loop.bend as Loop\n',
})
CHECKER['unprintable_main'] = ('pass', 'def main() -> Type:\n  Nat\n')
EXECUTION['unprintable_main'] = ('def main() -> Type:\n  Nat\n', {
    'emit-c': (False, 'cannot be printed'),
    'emit-js': (False, 'cannot be printed'),
    'eval': (True, 'Nat'),
})

FUSE_WORK = '''
def count(n: Nat, acc: U32) -> U32:
  match n:
    case 0n: acc
    case 1n+p: count(p,(acc + 1 : U32))
def work(+n: Nat) -> U32:
  a b = count(n,0) count(n,1)
  (a + b : U32)
'''
CODEGEN['fuse_once'] = (['FID_COUNT'], ['#define FID_WORK '], 'none',
    FUSE_WORK + 'def run(n: Nat) -> U32:\n  work(n)\n' + MAIN)
CODEGEN['fuse_twice'] = (['#define FID_WORK ', 'FID_COUNT'], [], 'none',
    FUSE_WORK + 'def run(+n: Nat) -> U32:\n  (work(n) + work(n) : U32)\n' + MAIN)

CODEGEN['call_closure_fork'] = (['FID_CLO_APPLY'], [], 'none', '''
def unit(value: U32) -> U32:
  (value + 1 : U32)
def increment(value: U32) -> U32:
  +copy = value
  a b = unit(copy) unit(copy)
  (a + b : U32)
def apply(operation: U32 -> U32, value: U32) -> U32:
  operation(value)
def run(n: Nat) -> U32:
  apply(increment,U32.from_nat(n))
''' + MAIN)
TASKS['call_closure'] = {'FID_RUN': 'fork-free', 'FID_CLO_APPLY': 'fork-free'}
TASKS['call_closure_fork'] = {'FID_RUN': 'may fork', 'FID_CLO_APPLY': 'may fork'}
