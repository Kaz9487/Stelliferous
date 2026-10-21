# Bend toolchain rules

[English](toolchain_rules.md) | [繁體中文](toolchain_rules.zh-TW.md)

These are the rules of the official checker, compiler and runtime that this project's code and proofs depend on. They were read from the unmodified official Bend sources (`bend.ts`, `comp.ts`, `main.ts` and `base.bend` under `.tools/bend/bend2/`) and are confirmed by small programs. They were last reviewed for **Bend v2.0.37**.

Each rule is supported in up to three ways:

- **Source.** The passage that decides the rule. [anchors.json](../scripts/toolchain_rules/anchors.json) holds one exact string per rule, so a change in the sources is noticed.
- **Probe.** A small program in [probes.py](../scripts/toolchain_rules/probes.py) that the pinned checker or compiler must treat as recorded. A rule marked *(probe: name)* has one.
- **Inference.** A rule marked *(inferred)* follows from the source and has no probe.

## Tools

| Command, from the repository root | Use |
| --- | --- |
| `python -B scripts/toolchain_rules/lint.py FILE_OR_DIR ...` | Before a checker run: scan Bend sources for spellings the checker rejects. It only reads files and takes seconds |
| `python -B scripts/toolchain_rules/verify.py` | After changing the Bend pin: check the anchors, the checker's list of error messages and every probe. It prints `RULES HOLD` or each difference |
| `CC=clang-19 python -B scripts/toolchain_rules/verify.py --assembly --probe shapes` | Keep the generated C, assembly, optimization remarks and static instruction counts of codegen probes under `out/toolchain_rules` |
| `CC=clang-19 python -B scripts/toolchain_rules/verify.py --inspect GENERATED.c` | The same for an existing generated C file |
| `CC=clang-19 python -B scripts/toolchain_rules/verify.py --runtime` | Also run the execution probes through the evaluator, generated JS and native C |
| `CC=clang-19 python -B scripts/toolchain_rules/verify.py --official .tools/bend/tests/compile/array_generic_cell.bend` | Run selected official tests and compare their `#|` output |
| `python -B scripts/toolchain_rules/verify.py --snapshot` | After the rules were reread and corrected for a new version: record that version and its messages |
| `python -B scripts/toolchain_rules/sweep.py OUT.json` | After an upgrade, when the full proof run fails: check files bottom-up and report those that fail on their own. It starts the checker once per file and is slow |
| `python -B scripts/toolchain_rules/fork_flags.py GENERATED.c NAME ...` | List which tasks in generated C are fork-free (rule H4) |
| `python3 -B scripts/toolchain_rules/leaf_workers.py` | On Linux: run the two leaf probes of rule H4 and count the leaves that run in a grow turn |
| `python -B benchmarks/model/timeline_claims.py TIMELINE_DIR` | Show how many workers are busy in each operation of a [model timeline](../benchmarks/model/README.md) |

A single file is checked with `node scripts/bend_launcher.mjs FILE --check-only`, which prints `ALL PROOFS CHECK` on success. [diagnostics.txt](../scripts/toolchain_rules/diagnostics.txt) snapshots literal fragments on `Err`/`parse_fail` call lines in `bend.ts`. It does not cover every CLI, compiler or runtime error.

## Official pipeline

The stages of the official toolchain, their entry points and what each passes on.

| Stage | Official implementation and output | What to know |
| --- | --- | --- |
| Load and parse | `main.ts:book_read`, `bend.ts:book_load/parse_book/body_flatten/match_flatten`: canonical real paths and aliases, case trees, first-order terms | Imports precede declarations. Reimporting a real path reuses its namespace. A path still loading is an import cycle |
| Check | `book_valid/def_check/term_check/term_infer`: annotated `Def.e`, quantities, descent, template instances | Types and bodies are checked, unused definitions too. Compiler reachability cannot hide a checker error |
| Convert | `term_higher/term_wnf/term_compare`: higher-order substitutions and shared weak-head cells | Weak-head evaluation unfolds saturated definitions and demanded heads. `term_snf` also normalizes every subterm |
| Complete and certify | `book_read`, `cli_verdict/book_promises` | `?TODO` and unfilled non-Base laws fail completion. `--check-only` additionally rejects transitive unsafe/foreign dependencies, including types and constructor fields |
| Kernel verdict | `safe.ts:safe_book/safe_check`, `bendtt.lean` | `--verdict` elaborates the accepted book and invokes BendTT, `--check-only` does not. Out-of-scope elaboration fails |
| Select backend | `main.ts:book_run/cli_emit`, `comp.ts:io_run/js_lib/compile_book` | A pure plain run uses `term_snf`. An IO plain run compiles to JS. `.js`, `.mjs` and `.c` outputs have separate paths. An executable build adds the native compiler |
| Discover and lower | `file_book/fun_of/lay_of/anf`: reachable definitions, live arguments, layouts, cuts and forks | Checked types drive erasure and layout. Non-tail worklist calls need continuations. Constants and dead lets can disappear |
| Emit and converge | `compile_book/emit_*`: native spins, segments, ownership and heat facts | Re-emission grows `own`, `hot`, `stat`. Then segment reachability, fork propagation and device reachability produce `FID_T`, `CID_T` and code |
| Execute | embedded C/JS runtime and `effs/*`: allocation, tasks, IO, host/device lanes | Base natives and effects are trusted |

The official CLI builds an executable with `-std=c11 -O3 -lpthread -lm`, and this project adds `-march=native -ffp-contract=off`. Generated code is not a proof verdict: `unsafe_codegen` fails `--check-only` but runs on all three execution paths.

## A. Checker: match

The body of a definition is compiled into a case tree (`match_flatten` in `bend.ts`). The checker keeps a list of parameters that are not yet lambdas. A match on the first of them destructs it and puts its fields at the front. A match on a later one first turns the parameters before it into lambdas, and those can no longer be matched.

- **A1** Match parameters, and the fields they produce, from left to right. After a later one is destructed, an earlier one cannot be: `can't be matched in this position`. The limit is the order, not the number of scrutinees. *(probes: `match_order_bad`, `match_order_ok`)*
- **A2** After a let, including `+x = ...`, no outer parameter can be matched. Write all parameter matches before the first let. *(probe: `match_after_let`)*
- **A3** A computed value cannot be matched: `a match cannot scrutinize a computed value`. Write a helper def that takes the value as a parameter. *(probe: `match_computed`)*
- **A4** A let-bound local cannot be matched: `cannot scrutinize a local binder`. Use a helper def as in A3. *(probe: `match_local`)*
- **A5** A value that is already a constructor cannot be matched again: `can't be matched (this value is already a constructor: bind its fields directly)`. Bind its fields directly, or fold the pattern into the outer case.
- **A6** Live code cannot match an erased (`-`) parameter: `a live scrutinee`. Make the parameter live, or match a live value that determines it. *(probe: `erased_scrutinee`)*
- **A7** Write list patterns as `Con{head,tail}` and `Nil{}`. `tail []` parses as an array read.

- **A8** A variable-only pattern column substitutes its binder without closing earlier parameters. The order restriction in A1 applies to constructor columns. *(probe: `match_variable_column`)*
- **A9** Rows have first-match priority. A constructor anywhere in a column makes that column strict even when an earlier catch-all wins. *(probe: `match_first_row`)*

- **A10** Exhaustiveness checks the remaining constructors. An empty match needs an empty live datatype (or a live empty context). An erased empty parameter cannot justify live elimination. *(probes: `match_missing_arm`, `empty_live`, `empty_erased`)*

## B. Checker: usage

A variable has one of three quantities: `-` (none), unmarked (once) and `+` (many). A live variable is used at most once. Type positions, `-` arguments, both sides of an equation and the motive of a rewrite are checked as dead code and do not count as uses.

- **B1** An unmarked value used twice fails with `x (consumed more than once)`. The usual causes are using a whole parameter after it was destructed, and a local alias that already consumed the value. *(probes: `use_twice`, `use_twice_plus`)*
- **B2** The type of `+x` must be Data: `its type must be Data`. Function types, Array and Buffer are not Data. *(probe: `plus_needs_data`)*
- **B3** Passing an argument to a `+` parameter counts as one use, so a once-usable Data value may be passed to a `+` parameter.
- **B4** A pattern binding has the field's quantity times the scrutinee's. A `+` field of a node matched once is still usable many times.
- **B5** An equation `{a == b : T}` is Data, because its evidence is erased. A proof term can be bound with `+` and reused.

- **B6** Sequential expressions add their usage maps (`uses_add`). Alternative match arms join them (`quant_join`), so a once-usable variable may occur once in each arm. *(probe: `usage_branches`)*

- **B7** Every value in `x y = a b` is checked in the outer context, so `b` cannot refer to the new `x`. Branch usage joins, but simultaneous let values add their uses. *(probe: `parallel_binder_scope`)*

## C. Checker: templates, laws and recursion

- **C1** `~k: T` may appear only at the front of a parameter list. *(probe: `tilde_not_leading`)*
- **C2** A call may mark a leading prefix of the callee’s declared `~` arguments, or omit all marks. Marks cannot resume after an unmarked argument or exceed the declared count. The parser accepts partial prefixes even though its overview comment says “each … or none”. A law declares these parameters with leading `for ~name: T` lines. *(probes: `template_marks_omitted`, `template_marks_partial`, `template_marks_gap`, `tilde_on_plain_call`, `law_template_ok`)*
- **C3** A template's `~` arguments must be closed: `a template applied to closed ~ arguments`. The template body is checked once, with each `~` parameter an opaque constant, so it holds for every instance whether or not anything calls it. The first live call also creates the instance at its arguments and checks it again. A template may instantiate itself at most 64 levels deep. *(probes: `template_uncalled_checked`, `template_function_uncalled_checked`)*
- **C4** A self-call must decrease. The arguments are compared with the definition's parameters from left to right: each is passed unchanged until one is smaller. Erased parameters are skipped. Otherwise `a decreasing self-call`. Non-tail recursion is accepted. *(probes: `non_decreasing`, `non_tail_recursion_ok`)*
- **C8** A self-call inside a lambda is held to C4 like any other. A loop over a computed result (A3) can therefore be a def that takes the result and a lambda, matches the result and calls the lambda, while the lambda calls the loop on a smaller argument. No mutual recursion is needed. *(probes: `lambda_self_call_ok`, `lambda_self_call_bad`, from [nuxyel](https://github.com/nuxyel), [bend-ml](https://github.com/nuxyel/bend-ml))*
- **C5** Live code cannot refer to a def that has no body yet. So there is no mutual recursion and no forward reference: `a filled definition (an unfilled law is a dead claim ...)`.
- **C6** A law is visible but stuck until it is filled, and unfolds afterwards. Declaring a law first does not keep it opaque later.
- **C7** A stuck call is canonical: two of them are compared by the name of the def and then argument by argument.

- **C9** Template instances are cached by `term_key(term_lower(argument))`, not by normalizing arguments. Different spellings can create separate instances even when they compute the same value. The serialized key is limited to 32,768 characters. *(source: `def_inst`)*

- **C10** Local acceptance of `?TODO` is provisional: `book_read` rejects all remaining holes and unfilled non-Base laws. `book_promises` then traces unsafe/foreign references through bodies, types and constructor fields for a proof verdict. Base foreign effects are trusted, but Base `@unsafe` functions are not exempt. *(probes: `todo_rejected`, `unfilled_law_rejected`, `unsafe_type_dependency`, `array_fork_trust`)*

## D. Checker: rewrite and type comparison

- **D1** In `%e : P; f` with `e : {a == b : T}`, **the current goal must be `P[_ := b]` and the body `f` proves `P[_ := a]`**. `_` may occur several times. For the other direction use `Equal.sym`. *(probes: `rewrite_direction_ok`, `rewrite_direction_bad`)*
- **D2** `term_compare` first compares with an empty book, so definitions stay rigid. On failure it retries with the real book and weak-head reduction. Both passes stop at identical objects, and successful equality merges shared cells. A spelling difference can expose an expensive subtree but does not force every subtree to normalize. Functions compare up to eta. *(probes: `function_eta`, `field_spelling_overflow`, `projection_spelling_ok`)*
- **D3** Two spellings that mean the same but normalize differently need an equation lemma. For large terms, `Equal.cong` with a lambda is more stable than a long motive.
- **D4** Write `+x = {Ctor{...} : Type}`. A constructor needs a target type to be checked.
- **D6** A match on a parameter, `Box{n} = box`, rewrites the goal and the context with `box := Box{n}`, so they read `size(Box{n})`, not `n`. A term written with the field `n` is then spelled differently from a statement written with `size(box)`, and D2 retries comparison with unfolding. For a statement that contains a large closed Nat or a long chain of arithmetic, this overflows the stack (E1, E5). For a small statement it passes. After such a match, write a term with the projection that the statement uses (`size(box)`), and use the field only where a reduction needs the constructor. A let `+m = size(box)` keeps that spelling: the first pass reads through it. *(probes: `field_spelling_overflow`, `projection_spelling_ok`, `projection_let_ok`, `small_field_spelling_ok`)*
- **D7** A def that matches its argument is stuck while the argument is abstract. Two accessors of the same field therefore differ on an abstract record if one of them matches it: `matched(box)` (a match) stays apart from `size(box)`. Define a derived accessor through the accessor it extends, `through(box) = size(box)`, without a match. It then unfolds to that spelling on any record. *(probes: `accessor_through_ok`, `accessor_matched_bad`)*
- **D5** Write operators as `(a + b : T)`, put spaces around comparisons, and parenthesize a compound first type argument of `X<A, B>`.

- **D8** Goal conversion uses directional `LE`: Data fits Type, function domains compare in reverse, and a datatype with fewer remaining constructors fits one with more. Equation endpoints and type arguments use exact `EQ`. *(source: `compare_go`, probe: `kind_data_to_type`)*

## E. Checker: Base definitions and large numbers

- **E0** In Base, `U32.min`, `U32.max`, `U32.div`, `U32.mod`, `F32.min` and `F32.max` match both arguments before they unfold, so a call on abstract variables stays stuck in a proof. Destruct both arguments into `U32{..}` or `F32{..}` first, or use the lemmas in `lib/proofs/u32_extrema.bend` and `lib/proofs/u32_division.bend`. Use `Pair.fst(Nat,Nat,..)` and `Pair.snd(Nat,Nat,..)` on `Nat.divmod`.

A literal is one node that unfolds one layer at a time when it is read. A U32 is a structure of 32 bits and is small. A Nat is unary, so any computation that walks a whole Nat costs time proportional to its value.

| Case | Result | Probe |
| --- | --- | --- |
| `Nat.is_le(10000000n, 20000000n) == True` by `{==}` | Passes in about a second | `nat_compute_10000000` |
| `U32.is_le(1073741824, 2147483648) == True` by `{==}` | Passes | `u32_compute_large` |
| `Nat.is_le(1n, U32.to_nat(1073741824)) == True` by `{==}` | Passes, one layer is unfolded | `to_nat_large_compare` |
| Both sides are `U32.to_nat(100000)` | Passes in about two seconds | `to_nat_equal_100000` |
| A type contains a closed `U32.to_nat(1073741824)` and is compared with an **identically spelled** type | Passes: the two are compared before any def is unfolded | `to_nat_symbolic_bound` |
| The same, with the other side spelled as the literal `1073741824n` | **Stack overflow** | `to_nat_mixed_literal` |
| `Nat.is_le` on two Nat literals of 2^30 | Does not finish in 60 seconds | none, tried by hand |

- **E1** Keep closed, computable large Nats out of types: for example `U32.to_nat(1073741824)`, or `U32.to_nat(Machine.capacity(default))` after unfolding. When the two sides are spelled differently, the comparison normalizes both and its recursion depth is proportional to the value. The mixed-literal probe overflows at 2^30 on the selected runtime, and the message gives no location.
- **E2** Keep such a quantity **abstract**. Make it a parameter (`limits: Machine.Limits`, `capacity: U32`), use it only through equations and inequalities about it, and substitute the actual value in the last step. `count_within` in `lib/proofs/bounded_u32_arithmetic.bend` and `lib/proofs/checked_word_completion.bend` are examples.
- **E3** Compare concrete numbers on U32 (`U32.is_le`), not after converting them to Nat.
- **E4** Small Nat literal computations, up to about 10^7, can use `{==}`. Their time is proportional to the value.
- **E5** Nested lets and type comparisons recurse on the JavaScript stack. The overflow boundary depends on the engine and term shape, not a language limit. `declaration_4000_steps` and `declaration_12000_steps` record two points for the selected runtime. Split a proof where a step deserves a name.

- **E6** Checker Nats are constructor terms, but native arithmetic has a `2^48−1` immediate ceiling and fails on overflow. U32 operations wrap at 32 bits. Division by zero gives 0, remainder by zero preserves the dividend, and shifts of at least 32 give 0. F32 arithmetic is opaque in Base proofs and implemented by native operations (`Math.fround` in JS). A numeric result from C/JS is not a `{==}` proof. *(execution probes: `word_edges`, `word_wrap`, `nat_runtime_limit`, `f32_opaque_native`, checker probe: `f32_opaque_equation`)*

## F. Compiler: the shape of a compiled def

How to read the generated C:

- `spin_N`: a native function. A def that exists only in this form has no `FID_NAME` of its own.
- `FID_NAME` with `WL_CASE(FID_NAME)`: a worklist segment.
- `FID_NAME_K<n>`: the continuation after a non-tail call. `FID_NAME_J<n>`: the join of a fork. `FID_NAME_C<n>`: a closure.
- `WL_AGAIN(FID_NAME)`: a tail self-jump inside a segment.

Rules:

- **F1** A def is **flat** when all of these hold: its body has no parallel let (one let binding two or more names), it has no bang call `f!(..)` or closure call, it calls itself only in tail position, and every def it calls directly is flat. *(probe: `shapes`)*
- **F2** A flat def normally emits a native C `spin_N`, with flattened machine-word parameters and results in `o[]`. Tail self-calls become loops. Non-flat execution uses worklist segments, but fusion and dead-code removal can erase a def’s individual boundary (F12). *(probe: `shapes`)*
- **F3** Being non-flat spreads to callers, not to callees. *(probe: `shapes`: `mixed` calls a non-tail-recursive def and gets `FID_MIXED`, while the flat `wrapper` it also calls stays a `spin`)*
- **F4** `SPIN_FAR = 8192`: a `spin_N` with fewer than 8,192 characters of generated C is `INLINE`, a longer one is `DFAR`. `DFAR` is a real function call only in a device build. On the CPU it is `INLINE`, so clang decides whether to inline it. *(probes: `inline_small`, `far_large`)*
- **F5** `FOLD_FUEL = 8192` is an unfolding budget, not a strict output-size or run-time bound. `emit_open` resets it, and emitting a nested native spin saves/restores the caller’s budget. `emit_fold` charges visited nodes after an eligible flat, fully applied body unfolds, recursively tries its calls, and memoizes results. Cache hits do not pay again. This does not bound clang inlining or every emitted node. *(source: `emit_open/emit_native/emit_fold/emit_unfold`)*
- **F6** **A def called directly by a parallel let becomes a task**, even when it is flat. It gets its own worklist segment, and its tail recursion becomes a self-jump in that segment, not a loop in a `spin`. So a hot loop should not be the direct child of a fork. Put it in a def one level below. *(probe: `shapes`: `kid` has `FID_KID` and `WL_AGAIN(FID_KID)`)*
- **F7** `INLINE` is a request to the C compiler, not a guarantee. A chain of small `spin_N` functions that pass a wide state through `o[]` is merged into one loop only while the C compiler chooses to inline every link, and it stops doing so when the chain gains more call sites. So a per-element helper shared by several operations can cost several real calls per element in a full program and nothing in a program that links one operation. Keep the per-element path a small loop of its own and step wide state once per run. Measure inside a program that links the other users. *(measured, no probe: the element cursor of a strided copy took 1.3 ns per element alone and 6.7 ns once combine and reduce were linked, and Clang 19 recovered it only with both inline thresholds at 20000, and was slower at 5000)*
- **F8** A parallel let is a fork only while both values stay calls. Its join is `FID_NAME_J<n>` in the C, so inspect the join together with the call segments to confirm the fork. An erased argument stays in its call (`anf`). *(probe: `fork_erased_last`, from [nuxyel](https://github.com/nuxyel), [bend-ml](https://github.com/nuxyel/bend-ml))*

- **F9** Each native tail loop emits `WL_SPIN`, which increments an error-poll counter and reads the error state every 4,096 iterations. Clang can remove polls when it proves they have no effect. Flatness alone does not imply vectorization. Inspect the loop and clang's vectorization remarks. *(probe: `shapes`)*
- **F10** On the CPU, each worklist segment is a `noinline` C function. `WL_JMP` makes a direct mandatory tail call, while `WL_DYN` indexes `wl_tab` for an indirect tail call. Continuation dispatch has register, stack and branch costs even without allocating another frame. *(probe: `shapes`)*

- **F11** U32 literal matches that cover most of `0..max` can become constant tables. `mat_rows` requires hits over half of `0..max` and one default body, and `emit_row` must also reduce each result to an allowed word expression. `TAB_AT` clamps to the last, default entry. Sparse or nonconstant cases keep branches. This Bend optimization precedes clang. *(official test: `u32_table_popcount`, dynamically exercises indices beyond the last literal)*

- **F12** A non-flat callee can fuse into a caller at a single reachable reference when the tail destination and return layouts permit it, provided it is neither a bang nor foreign call. `file_book` counts sites, `emit_body/emit_fuse` emits its body in the caller, and final reachability removes unused segments. `fuse_once` has no `FID_WORK`, while the two calls in `fuse_twice` retain it. A missing FID alone does not prove flatness or absence of scheduling work. *(probes: `fuse_once`, `fuse_twice`)*

## G. Compiler: layout and sharing

- **G1** Layout (`lay_of`). A non-recursive data type is flattened into machine words and allocates nothing. U32 and F32 take 32 bits, Nat 64. `Array`, `IO.OP` and recursive types (lists, trees) are heap boxes. A type wider than `WIDE = 247` words becomes a box. When the parameters together exceed it, multi-word parameters become boxes. A continuation segment, which carries every live value across a non-flat call, is not boxed this way: more than 247 live words fails compilation with `an arity over 247` (`seg_open`). Carry a variable number of tensors in a recursive type, which is one box.
- **G2** Sharing (`facts_hot`). A value used more than once marks its type hot. That type's constructors become shared, a match on it goes through `ctr_take`, which checks the reference count, and the heat spreads to the field types. *(probes: `share_none` shares nothing, and `share_list`, which uses a list twice, shares some constructors)* The project met this with `Views.View`: a view that one definition handed to two others, and a view carried through a loop as a parameter, each made every `View` constructor shared. With the first, the four-thread model measured 136.7 against 143.2 ms and 132.1 against 145.3 ms in two alternating comparisons (interquartile range about 25 ms). After building the view once per use it measured 130.5 against 130.9 ms. A view built in an argument position makes that parameter owned for every caller (G8). A caller that passes on a `+` view it also uses elsewhere then shares it, so the reductions bind a new view to a name before passing it. `scripts/check_generated_c.py` rejects a shared `View` in the model, the benchmarks and the integration entries. The check the launcher runs on every compilation leaves it out, because a program's own definitions may share a view.
- **G3** `facts_hot` follows datatype fields, lambda bodies and stuck type-family arms. A forced unresolved `All`, `Var` or `App` sets `"*"` unless an erased parameter can pass heat back to its caller. `"*"` forces heat when subsequent bindings are visited. It does **not** simply set every `CID_T` entry. `share_existential` follows the erased-field fallback: Hide/WCon/Bool are hot, Con is not (4/22 entries). The concrete-field counterpart heats Hide/Con/Nil instead (3/22). These flags expose different conservative paths, not a proof of arbitrary existential-program correctness. *(probes: `share_existential`, `share_concrete`, plus the resolved generic/family probes)*
- **G4** A binding counts its uses: the last use takes the value and earlier uses share it. Sharing a flattened value copies words, not nodes.
- **G5** Templates and closures. A `~` template argument is instantiated by the checker, so the compiler sees a direct call. *(probe: `call_template`: the whole chain is a `spin`)* A function passed as an ordinary parameter is a closure, and **the def that calls a closure is not flat**. *(probe: `call_closure`)* A function stored in a record was measured 5.85–6.5 times slower than a template.
- **G6** A generic `~Context: Data` with `+context` heats only the instantiated type. *(probe: `share_template_parameter`)*
- **G7** A template substitutes its closed argument expression at each use. In `template_record`, reading one computed field twice emits two calls to the counting loop, which both survive clang. Compute the field once and pass its value as a `+` parameter when sharing is intended. A closure delays computation but does not memoize it, and adds the dispatch cost in G5. *(probe: `template_record`, from [nuxyel](https://github.com/nuxyel), [bend-ml](https://github.com/nuxyel/bend-ml))*

- **G8** Emission repeats while the monotonically growing `own`, `hot` or `stat` sets gain entries. Each pass rebuilds borrowing/lending and emitted segments. Transitive lends propagate through `graph_close`, and unlent borrowed parameters become owned. Layout caches are separate from this convergence test. `own_single` and `borrow_repeated` keep the same `scan` body: its native code uses `ctr_take/term_sink/spare_free` in the former and `term_peek` without disposal in the latter. Array cells use the element datatype’s **open** layout so callers agree. A bare unresolved element type is rejected, including by JS. *(source: `compile_book/lay_el`, official tests: `array_generic_cell`, `array_open_element`, `array_open_element_family`)*

- **G9** Constants and borrowed nodes have different disposal rules. `ctr_build` puts static fields in `STAT_IMG` and adds their constructor to `stat`, and `node_fields` uses `ctr_take` for owned hot **or static** constructors. Thus `ctr_take` alone is not proof of runtime sharing. Packed zero/one-word nodes and static addresses are trivial for `term_sink`. An owned heap subtree is dropped iteratively with reference counts. *(source: `ctr_build/node_fields/term_triv/term_drop`, probes: `own_single`, `borrow_repeated`)*

## H. Runtime: scheduling, memory and files

- **H1** A fork (parallel let) creates a join task and one kid per call. When the kids run in sequence, one frame serves all steps.
- **H2** Workers rotate over rings. A lane in a grow turn skips a task that is fork-free (`fid_nofk` in `monk_step`). The host grows one ring after another and then drains them. The host thread also takes units in every turn, next to `threads - 1` pool threads, and a thread yields a bounded number of times before it sleeps (`POOL_WAIT`). There is no work stealing. One parallel step is therefore one grow turn plus one work turn, and the grow turn is not meant to finish leaves. A planner must count whole turns.
- **H3** A parallel let transfers its arguments. Passing a uniquely owned Array does not itself imply aliasing. Explicit `Array.fork` creates aliases of one block and is `@unsafe`, and its users fail `--check-only` (C10). Shared handles use atomic redirect counts. `Array.clone` copies the outer block with `blk_copy`. Boxed elements are retained/shared, not recursively deep-copied. *(probe: `array_fork_trust`, source: `blk_copy/blk_fill/blk_keep`)*
- **H4** **A fork-free leaf task is deferred to the work turn.** The mechanism, in `cube_run`, `row_grow`, `monk_step`, `task_deal` and `emit_jump`:
  1. At the top fork, the host puts the two kids on **two different rows** (`ring_flip`).
  2. In the grow turn, one worker owns a row. It runs the tasks that may fork, and the kids they fork stay in **the same row** (`ring_pick`). When a turn has fewer rows than threads and a grow runs past 1 ms, the forks it makes after that are dealt over all rows through the shared cursor, and grow turns repeat until none deals (`row_grow`, `cube_run`).
  3. If a kid is the forking recursion itself and the leaf is a flat def inlined into it, that worker runs **one whole leaf inside the grow turn**. After it finishes that task, the row's grow turn ends.
  4. In the work turn, the 128 rings of a row are divided into units (`pool_step`: more units when few rows are in use, eight when many), and each thread claims the next unit with an atomic counter. The remaining leaves run here.

  In the `leaf_inline` probe, a tree of depth two with four leaves runs two leaves in the grow turn on two workers and two in the work turn on two workers. The total is the time of two leaves, not one. With 16 leaves it is two in the grow turn and 14 in the work turn.

  The remedy: where a child is a leaf, the parallel let calls a def that **neither forks nor calls a closure**. By F6 it becomes a task, and its row in the `FID_T` table has mask 2 set (bit index 1, which `fork_flags.py` prints). The grow turn skips it, so all leaves run in the work turn in different units. The hot loop goes in a flat def one level below (F6). Avoiding closure calls is sufficient, not necessary: `Clo~apply` reaches every emitted closure, and is marked may-fork only if that graph reaches a fork. `call_closure` keeps RUN and CLO_APPLY fork-free, and `call_closure_fork` marks both may-fork. Flatness and fork-freedom are different analyses.

  *(probes: `leaf_inline` and `leaf_task`. `verify.py` checks the `FID_T` flags of their tasks. `leaf_workers.py` runs them with 4, 8 and 16 leaves at two and four threads and requires two leaves in a grow turn for the inlined form and none for the fork-free form. With four leaves each, measured on an 8-thread virtual machine: 97.6 and 97.7 ms at two threads, and **97.5 ms against 48.9 ms** at four. The inlined form is no faster at four threads than at two.)*

- **H5** An Array or a Buffer is one block of 2^c cells (`blk_new`), taken from a free list of its class (`heap_alloc`). A block of n cells takes the next power of two, up to twice the cells it holds. A freed block returns to the list of its class and serves only that class. Blocks are neither split nor merged. Count memory in blocks, not elements. *(source, from [nuxyel](https://github.com/nuxyel), [bend-ml](https://github.com/nuxyel/bend-ml))*
- **H6** `File.read_bytes` and `File.read_at` return a list with one cell per byte, so building and walking that list is the cost of a load. Read in chunks and convert each chunk before reading the next. A reader in foreign code avoids the list, but a file that imports it fails `--verdict` (bendlang/bend#1573 asks for an Array read in Base). *(source, from [nuxyel](https://github.com/nuxyel), [bend-ml](https://github.com/nuxyel/bend-ml))*

- **H7** Array indices wrap with a mask derived from the block class. `Array.get` at index 3 of a two-element array reads index 1. Index validity is a library contract, not a runtime bounds exception. *(probe: `array_index_wrap`)*
- **H8** Destructing an Array into halves uses `blk_half`, allocating and copying each part. This differs from `Array.fork` handles, which alias storage. For buffers, two 32-bit cells fit in one 64-bit heap word. Count copied cells and allocator words separately. *(probe: `array_blocks`)*

- **H9** Allocation is per lane and size class. The host recycles HOT, then a parked COLD generation, then the shared bank, before taking a fresh quantum. A full free generation moves to COLD/bank. Returning a block to a free list does not mean returning memory to the OS. Shared boxed values add a redirect cell with a 24-bit copy count, with fail-stop on overflow. *(source: `heap_alloc_miss/heap_hand/heap_free/rfc_bump`)*

- **H10** Host and device share generated semantics but not dispatch machinery. Host segments use `musttail` and optional `preserve_none`, while device segments are cases in one kernel. Bang reachability and closure reachability select device code. `dev` has separate grow/work/bank-pack passes, and a host assembly result does not measure those passes. JS uses direct calls/tail trampolines and sequential lets, without the C worklist parallelism. *(source: `compile_book`, runtime Dialect/Dev, `js_open/js_def/run_loop`, source-reviewed, GPU not executed)*

## I. What this means for design

| Decision | Mechanism and evidence |
| --- | --- |
| Separate tree scheduling from leaf loops | Forks, closure calls and non-tail recursion prevent flatness (F1, G5). Use a template leaf, a fork-free task boundary and a native loop below it (F6, H4) |
| Compare the whole linked program | Inspect constructor sharing, task flags, surviving calls and vectorization remarks. Reachable callers can change ownership and clang inlining (F7, G8) |
| Count work before explaining time | Dynamic instruction counts count executed work. Static assembly counts describe code size and loop bodies. Equal instruction counts with different cycles can also reflect cache misses, contention, frequency or branch prediction. They do not identify scheduling or alignment as the cause |
| Preserve computed values deliberately | Template expressions can run more than once (G7). Array clones and pattern splits copy storage (H3, H8). Byte reads allocate a list (H6) |
| Keep proof bounds abstract | Rigid comparison can avoid expansion, but a later conversion may expose a large closed Nat (D2, E1, E2) |

`verify.py --inspect GENERATED.c` keeps the assembly and remarks of one linked program. Compare two programs only with the same input, worker count, target and flags. The instruction-first approach comes from [nuxyel](https://github.com/nuxyel), [bend-ml](https://github.com/nuxyel/bend-ml).

- **A function in a record field stays a closure.** Direct F32 arithmetic in `traversal_array.update` becomes a native loop. The same update through a record field enters `CLO_APPLY` twice and allocates a one-word closure per element (F1, F6, G5, G7). `benchmarks/checks/measure_arithmetic_dispatch.py` measures both.

## J. What lint.py finds

| Rule | Finding |
| --- | --- |
| A2 | A match or destructuring after a let |
| A3 | A match or destructuring of a computed value |
| A7 | A pattern followed by `[]` |
| C1 | A `~` parameter that is not leading |
| C2 | Excess or non-leading `~` call arguments, resolved through import aliases |
| D4 | A let of a constructor without a type annotation |
| E1 | `U32.to_nat(large number)`, or a Nat literal of six or more digits |
| E1? | `U32.to_nat(f(X.limits()))`: a default machine limit converted to Nat. The question mark means a risk, not a certain failure |

On the accepted sources of `lib` and `convolution` it reports no findings. It looks at spelling only: it finds neither usage errors (B1), constructor-column order nor type mismatches, and C2 checks only calls that write `~` marks to a declaration it resolves. `verify.py` compares lint findings with checker probes.
