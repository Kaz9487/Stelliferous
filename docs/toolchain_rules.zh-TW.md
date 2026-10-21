# Bend 工具鏈規則

[English](toolchain_rules.md) | [繁體中文](toolchain_rules.zh-TW.md)

這裡整理本專案的程式與證明所依賴的官方 checker、編譯器與 runtime 規則。規則是從未修改的官方 Bend 原始碼讀出來的（`.tools/bend/bend2/` 下的 `bend.ts`、`comp.ts`、`main.ts`、`base.bend`），並以小程式確認。最近一次核對的版本是 **Bend v2.0.37**。

每條規則最多有三種依據：

- **原始碼**：決定該規則的程式段。[anchors.json](../scripts/toolchain_rules/anchors.json) 為每條規則存一段完全相同的字串，原始碼一改就會被發現。
- **Probe**：[probes.py](../scripts/toolchain_rules/probes.py) 裡的小程式，固定版本的 checker 或編譯器對它的處理必須和記錄一致。標示 *（probe：名稱）* 的規則有 probe。
- **推論**：標示 *（推論）* 的規則由原始碼推出，沒有 probe。

## 工具

| 指令（在 repository 根目錄執行） | 用途 |
| --- | --- |
| `python -B scripts/toolchain_rules/lint.py 檔案或資料夾 ...` | 送 checker 之前：掃描 Bend 原始碼中 checker 會拒絕的寫法。只讀檔案，幾秒就跑完 |
| `python -B scripts/toolchain_rules/verify.py` | 更換 Bend 版本之後：檢查錨點、checker 的錯誤訊息清單與每個 probe。印出 `RULES HOLD` 或逐項差異 |
| `CC=clang-19 python -B scripts/toolchain_rules/verify.py --assembly --probe shapes` | 在 `out/toolchain_rules` 保留 codegen probe 的 C、組合語言、最佳化診斷與靜態指令數 |
| `CC=clang-19 python -B scripts/toolchain_rules/verify.py --inspect GENERATED.c` | 對已生成的 C 做同樣的事 |
| `CC=clang-19 python -B scripts/toolchain_rules/verify.py --runtime` | 另外經 evaluator、生成的 JS 與原生 C 執行 execution probe |
| `CC=clang-19 python -B scripts/toolchain_rules/verify.py --official .tools/bend/tests/compile/array_generic_cell.bend` | 執行指定的官方測試，比對 `#|` 的輸出 |
| `python -B scripts/toolchain_rules/verify.py --snapshot` | 依新版本重讀並修正規則之後：記錄該版本與它的訊息清單 |
| `python -B scripts/toolchain_rules/sweep.py OUT.json` | 升級後完整證明失敗時：由下往上逐檔檢查，回報「自己壞掉」的檔。每個檔啟動一次 checker，所以很慢 |
| `python -B scripts/toolchain_rules/fork_flags.py GENERATED.c 名稱 ...` | 列出生成的 C 中哪些 task 不含 fork（規則 H4） |
| `python3 -B scripts/toolchain_rules/leaf_workers.py` | 在 Linux 上：執行規則 H4 的兩個 leaf probe，計算在成長輪執行的 leaf 數 |
| `python -B benchmarks/model/timeline_claims.py TIMELINE_DIR` | 顯示[模型時間軸](../benchmarks/model/README.zh-TW.md)中每個運算有幾個 worker 在工作 |

單一檔案用 `node scripts/bend_launcher.mjs FILE --check-only` 檢查，通過時印出 `ALL PROOFS CHECK`。[diagnostics.txt](../scripts/toolchain_rules/diagnostics.txt) 記錄 `bend.ts` 中 `Err`／`parse_fail` 呼叫所在行的字串片段，沒有涵蓋全部 CLI、編譯器或 runtime 錯誤。

## 官方流程

官方工具鏈的各階段、入口與每一階段交給下一階段的東西。

| 階段 | 官方實作與產物 | 要知道的事 |
| --- | --- | --- |
| 載入、解析 | `main.ts:book_read`、`bend.ts:book_load/parse_book/body_flatten/match_flatten`：實體路徑與別名、case tree、一階 term | import 在宣告之前。同一實體路徑重用 namespace，仍在載入中的路徑代表循環匯入 |
| 檢查 | `book_valid/def_check/term_check/term_infer`：帶型別的 `Def.e`、用量、遞減、樣板實例 | 型別與本體都要檢查，沒有被呼叫的定義也要檢查。編譯器的可達性刪除不能隱藏 checker 錯誤 |
| 轉換比較 | `term_higher/term_wnf/term_compare`：高階替換與共享的弱首正規形 cell | 弱首求值展開已足量套用的定義與需要的頭部。`term_snf` 還會正規化每個子項 |
| 完整性、可信依賴 | `book_read`、`cli_verdict/book_promises` | `?TODO` 與未填的非 Base law 過不了完整性檢查。`--check-only` 另拒絕傳遞的 unsafe／foreign 依賴，包含型別及建構子欄位 |
| Kernel verdict | `safe.ts:safe_book/safe_check`、`bendtt.lean` | `--verdict` 把已接受的 book elaboration 後交給 BendTT，`--check-only` 不會。超出 elaboration 範圍會失敗 |
| 選擇後端 | `main.ts:book_run/cli_emit`、`comp.ts:io_run/js_lib/compile_book` | 純值的直接執行使用 `term_snf`，IO 的直接執行編譯成 JS。`.js`、`.mjs`、`.c` 有各自路徑，產生執行檔還會呼叫原生編譯器 |
| 尋找依賴、降低表示層級 | `file_book/fun_of/lay_of/anf`：可達定義、live 引數、配置、cut 與 fork | 已檢查的型別決定擦除與配置。非尾端的 worklist 呼叫需要 continuation。常數與未使用 let 可能消失 |
| 產生、收斂 | `compile_book/emit_*`：native spin、segment、擁有權與共享分析結果 | 重複產生時累積 `own`、`hot`、`stat`，接著計算 segment 可達性、fork 傳播與裝置可達性，產生 `FID_T`、`CID_T` 與程式碼 |
| 執行 | 內嵌 C／JS runtime 與 `effs/*`：配置、task、IO、host／device 路徑 | Base native 與 effects 是受信任的 |

官方 CLI 產生執行檔用 `-std=c11 -O3 -lpthread -lm`，本專案另加 `-march=native -ffp-contract=off`。產生程式碼不等於通過 proof verdict：`unsafe_codegen` 過不了 `--check-only`，但三條執行路徑都能跑。

## A. checker：match

定義的本體會被編譯成一棵 case tree（`bend.ts` 的 `match_flatten`）。checker 保留一串「還沒變成 lambda 的參數」。match 這一串的第一個，會把它拆開，欄位放到最前面。match 後面的某一個，會先把它前面的參數變成 lambda，那些參數之後就不能再 match。

- **A1** 參數（以及拆出來的欄位）要由左到右依序 match。先拆了後面的，前面的就不能再拆：`can't be matched in this position`。限制是順序，不是 scrutinee 的個數。*（probe：`match_order_bad`、`match_order_ok`）*
- **A2** let（包括 `+x = ...`）之後，外層的參數都不能再 match。所有對參數的 match 要寫在第一個 let 之前。*（probe：`match_after_let`）*
- **A3** 不能 match 計算出來的值：`a match cannot scrutinize a computed value`。寫一個以該值為參數的輔助 def。*（probe：`match_computed`）*
- **A4** 不能 match let 綁定的區域變數：`cannot scrutinize a local binder`。做法同 A3。*（probe：`match_local`）*
- **A5** 已經是建構子的值不能再 match：`can't be matched (this value is already a constructor: bind its fields directly)`。直接綁它的欄位，或把 pattern 併進外層的 case。
- **A6** live 的程式不能 match erased（`-`）參數：`a live scrutinee`。把參數改成 live，或改 match 一個能決定它的 live 值。*（probe：`erased_scrutinee`）*
- **A7** list 的 pattern 寫成 `Con{head,tail}` 與 `Nil{}`。`tail []` 會被解析成讀取陣列元素。

- **A8** 只有變數模式的欄會替換綁定名稱，不會關閉前面的參數。A1 的順序限制適用於有建構子的欄。*（probe：`match_variable_column`）*
- **A9** 第一個符合的列優先。欄內只要有建構子模式，就會先拆解該欄，即使前面的萬用模式已經勝出。*（probe：`match_first_row`）*

- **A10** 窮盡性檢查剩餘的建構子。空 match 需要空的 live 資料型別（或含 live 空型別的 context）。erased 空參數不能用來推導 live 結果。*（probe：`match_missing_arm`、`empty_live`、`empty_erased`）*

## B. checker：用量

變數的用量有三種：`-`（不用）、不加標記（一次）、`+`（多次）。live 的變數最多用一次。型別位置、`-` 引數、等式的兩端、rewrite 的 motive 都當成 dead 程式檢查，不算用量。

- **B1** 沒有標記的值用兩次，會報 `x (consumed more than once)`。常見原因有兩種：參數被拆開之後又整個使用，或區域別名已經把值用掉。*（probe：`use_twice`、`use_twice_plus`）*
- **B2** `+x` 的型別必須是 Data：`its type must be Data`。函式型別、Array、Buffer 都不是 Data。*（probe：`plus_needs_data`）*
- **B3** 把引數傳給 `+` 參數只算一次使用，所以只能用一次的 Data 值可以傳給 `+` 參數。
- **B4** pattern 綁定的用量等於欄位的用量乘上 scrutinee 的用量。只 match 一次的節點，它的 `+` 欄位仍然可以用多次。
- **B5** 等式 `{a == b : T}` 是 Data，因為它的證據會被抹除。證明項可以用 `+` 綁定並重複使用。

- **B6** 連續求值用 `uses_add` 相加用量，互斥分支用 `quant_join` 合併。因此只能使用一次的變數，可以在每個分支各出現一次。*（probe：`usage_branches`）*

- **B7** `x y = a b` 的每個值都在外層 context 檢查，`b` 不能引用新綁定的 `x`。分支合併用量，但平行 let 的各個值會相加用量。*（probe：`parallel_binder_scope`）*

## C. checker：樣板、law 與遞迴

- **C1** `~k: T` 只能出現在參數列的最前面。*（probe：`tilde_not_leading`）*
- **C2** 呼叫可對已宣告 `~` 引數的前面一段加上 `~`，也可全部省略。標記不能在未標記引數之後重新出現，也不能超過宣告數目。parser 接受只標記前面一段，儘管概述註解寫的是「全部或全省」。law 用前置的 `for ~name: T` 宣告。*（probe：`template_marks_omitted`、`template_marks_partial`、`template_marks_gap`、`tilde_on_plain_call`、`law_template_ok`）*
- **C3** 樣板的 `~` 引數必須是封閉的：`a template applied to closed ~ arguments`。樣板本體會檢查一次，每個 `~` 參數當成不透明的常數，所以不論有沒有被呼叫，它都對每個實例成立。第一次 live 呼叫時，還會用實際引數產生實例並再檢查一次。樣板自我實例化最多 64 層。*（probe：`template_uncalled_checked`、`template_function_uncalled_checked`）*
- **C4** 自我呼叫必須遞減。引數由左到右與定義的參數比較：每個照原樣傳，直到某一個變小。erased 的參數會跳過。否則報 `a decreasing self-call`。非尾遞迴可以通過。*（probe：`non_decreasing`、`non_tail_recursion_ok`）*
- **C8** lambda 裡的自我呼叫和其他自我呼叫一樣要符合 C4。所以對計算出來的結果做的迴圈（A3）可以寫成一個 def：接收結果與一個 lambda，match 結果後呼叫 lambda，lambda 再以較小的引數呼叫迴圈。不需要互相遞迴。*（probe：`lambda_self_call_ok`、`lambda_self_call_bad`，來自 [nuxyel](https://github.com/nuxyel), [bend-ml](https://github.com/nuxyel/bend-ml)）*
- **C5** live 的程式不能引用還沒有本體的 def。所以沒有互相遞迴，也沒有往前引用：`a filled definition (an unfilled law is a dead claim ...)`。
- **C6** law 在填入之前可見但卡住（stuck），填入之後就會展開。先宣告 law，並不會讓它之後保持不透明。
- **C7** 卡住的呼叫是 canonical 的：比較時先比 def 的名字，再逐一比引數。

- **C9** 樣板實例用 `term_key(term_lower(argument))` 作快取鍵，不先正規化引數。算出同一值的不同寫法可能產生不同實例。序列化的鍵限制為 32,768 字元。*（原始碼：`def_inst`）*

- **C10** 局部接受 `?TODO` 只是暫時結果：`book_read` 會拒絕剩餘 hole 與未填的非 Base law。proof verdict 的 `book_promises` 再沿本體、型別與建構子欄位追查 unsafe／foreign 引用。Base foreign effects 受信任，但 Base 的 `@unsafe` 函式沒有豁免。*（probe：`todo_rejected`、`unfilled_law_rejected`、`unsafe_type_dependency`、`array_fork_trust`）*

## D. checker：rewrite 與型別比較

- **D1** `%e : P; f` 中，設 `e : {a == b : T}`，**目前的目標必須是 `P[_ := b]`，body `f` 要證 `P[_ := a]`**。`_` 可以出現多次。另一個方向用 `Equal.sym`。*（probe：`rewrite_direction_ok`、`rewrite_direction_bad`）*
- **D2** `term_compare` 先用空 book 比較，讓定義保持不展開。失敗後改用實際 book 與弱頭化簡。兩輪都會跳過相同物件，成功的等式比較還會合併共享節點。寫法不同可能暴露昂貴的子樹，但不保證每個子樹都被正規化。函式比較考慮 eta。*（probe：`function_eta`、`field_spelling_overflow`、`projection_spelling_ok`）*
- **D3** 兩個寫法意思相同但 normal form 不同時，需要一條等式引理。對大型的項，`Equal.cong` 加 lambda 比寫長的 motive 穩定。
- **D4** 寫成 `+x = {Ctor{...} : Type}`。建構子需要目標型別才能檢查。
- **D6** 對參數做 match（`Box{n} = box`）時，目標和 context 會改寫成 `box := Box{n}`，所以讀到的是 `size(Box{n})`，不是 `n`。用欄位 `n` 寫的項，和用 `size(box)` 寫的敘述寫法不同，D2 就會允許展開後重新比較。敘述含有大的封閉 Nat 或長串算術時會堆疊溢位（E1、E5），敘述小時則會通過。這種 match 之後，項要用敘述所用的投影（`size(box)`）來寫，欄位只用在需要建構子才能化簡的地方。let `+m = size(box)` 會保留這個寫法，第一輪比對會穿過它。*（probe：`field_spelling_overflow`、`projection_spelling_ok`、`projection_let_ok`、`small_field_spelling_ok`）*
- **D7** 對參數做 match 的 def，在參數抽象時會卡住。所以同一個欄位的兩個存取函式，只要其中一個自己 match 記錄，在抽象記錄上就不會相等：`matched(box)`（有 match）和 `size(box)` 不同。衍生的存取函式要透過它延伸的那個存取函式來定義，寫成 `through(box) = size(box)`，不要 match，這樣對任何記錄都會展開成相同寫法。*（probe：`accessor_through_ok`、`accessor_matched_bad`）*
- **D5** 運算子寫成 `(a + b : T)`，比較運算的兩側加空白，`X<A, B>` 的第一個型別引數如果是複合型別要加括號。

- **D8** 目標轉換用有方向的 `LE`：Data 可當 Type，函式參數型別反向比較，剩餘建構子較少的型別可符合剩餘較多的型別。等式兩端與型別引數用精確的 `EQ`。*（原始碼：`compare_go`，probe：`kind_data_to_type`）*

## E. checker：Base 的定義與大數

- **E0** Base 中，`U32.min`、`U32.max`、`U32.div`、`U32.mod`、`F32.min`、`F32.max` 先 match 兩個引數才展開，所以在證明裡對抽象變數的呼叫會卡住。先把兩個引數拆成 `U32{..}` 或 `F32{..}`，或使用 `lib/proofs/u32_extrema.bend` 與 `lib/proofs/u32_division.bend` 中的引理。商與餘數使用 `Nat.divmod` 上的 `Pair.fst(Nat,Nat,..)` 與 `Pair.snd(Nat,Nat,..)`。

字面值是一個節點，被讀取時才一層一層展開。U32 是 32 個 bit 的結構，本身很小。Nat 是一元表示，所以任何走完整個 Nat 的計算，時間都和數值成正比。

| 情況 | 結果 | Probe |
| --- | --- | --- |
| `Nat.is_le(10000000n, 20000000n) == True` 用 `{==}` | 通過，約一秒 | `nat_compute_10000000` |
| `U32.is_le(1073741824, 2147483648) == True` 用 `{==}` | 通過 | `u32_compute_large` |
| `Nat.is_le(1n, U32.to_nat(1073741824)) == True` 用 `{==}` | 通過，只展開一層 | `to_nat_large_compare` |
| 兩邊都是 `U32.to_nat(100000)` | 通過，約兩秒 | `to_nat_equal_100000` |
| 型別中有封閉的 `U32.to_nat(1073741824)`，並與另一個**寫法完全相同**的型別比較 | 通過：展開任何 def 之前先比較兩者 | `to_nat_symbolic_bound` |
| 同上，但另一邊寫成字面值 `1073741824n` | **堆疊溢位** | `to_nat_mixed_literal` |
| 對兩個 2^30 的 Nat 字面值做 `Nat.is_le` | 60 秒內跑不完 | 無，手動試過 |

- **E1** 不要讓封閉、可以被算出來的大 Nat 出現在型別裡：例如 `U32.to_nat(1073741824)`，或展開後的 `U32.to_nat(Machine.capacity(default))`。兩邊寫法不同時，比較會把兩邊都正規化，遞迴深度和數值成正比。目前執行環境的混合字面值探針在 2^30 溢位，訊息不給位置。
- **E2** 讓這種量保持**抽象**。把它寫成參數（`limits: Machine.Limits`、`capacity: U32`），只透過關於它的等式與不等式使用，最後一步才代入實際的值。`lib/proofs/bounded_u32_arithmetic.bend` 的 `count_within` 與 `lib/proofs/checked_word_completion.bend` 是例子。
- **E3** 具體數字的大小比較在 U32 上做（`U32.is_le`），不要轉成 Nat 再比。
- **E4** 小的 Nat 字面值計算（到大約 10^7）可以用 `{==}`，時間和數值成正比。
- **E5** 巢狀 let 與型別比較會使用 JavaScript 呼叫堆疊。溢位界線取決於引擎與項的形狀，不是語言限制。`declaration_4000_steps` 與 `declaration_12000_steps` 記錄目前執行環境的兩個觀測點。某一步值得有名稱時再拆開證明。

- **E6** checker 的 Nat 是建構子 term，但 native 算術有 `2^48−1` immediate 上限，溢位會失敗。U32 算術以 32 bit 環繞。除零回 0、餘數除零保留被除數，位移至少 32 位回 0。F32 算術在 Base 證明中不透明，由原生運算實作（JS 用 `Math.fround`）。C／JS 算出的數值不等於 `{==}` 證明。*（執行 probe：`word_edges`、`word_wrap`、`nat_runtime_limit`、`f32_opaque_native`，checker probe：`f32_opaque_equation`）*

## F. 編譯器：一個 def 編譯出來的形狀

生成的 C 怎麼讀：

- `spin_N`：原生函式。只以這種形式存在的 def，沒有自己的 `FID_名稱`。
- `FID_名稱` 加上 `WL_CASE(FID_名稱)`：worklist 的 segment。
- `FID_名稱_K<n>`：非尾端呼叫之後的接續。`FID_名稱_J<n>`：fork 的匯合。`FID_名稱_C<n>`：閉包。
- `WL_AGAIN(FID_名稱)`：segment 內的尾端自我跳轉。

規則：

- **F1** 一個 def 是 **flat**，要同時滿足：本體沒有平行 let（一個 let 綁定兩個以上的名字）、沒有 bang 呼叫 `f!(..)` 或閉包呼叫、只在尾端呼叫自己、它直接呼叫的每個 def 也都是 flat。*（probe：`shapes`）*
- **F2** flat def 通常產生原生 C `spin_N`，參數攤平成機器字，結果寫進 `o[]`，尾端自呼叫成為迴圈。非 flat 執行走 worklist segment，但融合與不可達程式碼刪除可能消去個別 def 的邊界（F12）。*（probe：`shapes`）*
- **F3** 不 flat 會往呼叫者傳，不會往被呼叫者傳。*（probe：`shapes`：`mixed` 呼叫了一個非尾遞迴的 def，所以有 `FID_MIXED`，它同時呼叫的 flat `wrapper` 仍然是 `spin`）*
- **F4** `SPIN_FAR = 8192`：生成的 C 少於 8,192 個字元的 `spin_N` 是 `INLINE`，更長的是 `DFAR`。`DFAR` 只有在 device 建置時才是真正的函式呼叫，在 CPU 上就是 `INLINE`，由 clang 決定是否內聯。*（probe：`inline_small`、`far_large`）*
- **F5** `FOLD_FUEL = 8192` 是展開預算，不是嚴格的輸出大小或執行時間上限。`emit_open` 重設它，產生內層 native spin 時保存、還原呼叫者預算。`emit_fold` 在符合條件、已足量套用的 flat 本體展開後扣除走訪節點數，再遞迴嘗試內部呼叫，結果會快取，快取命中不再扣款。這不限制 clang 內聯，也不涵蓋每個生成節點。*（原始碼：`emit_open/emit_native/emit_fold/emit_unfold`）*
- **F6** **被平行 let 直接呼叫的 def 會變成 task**，即使它是 flat。它得到自己的 worklist segment，尾遞迴變成 segment 內的自我跳轉，不是 `spin` 裡的迴圈。所以熱迴圈不要直接當 fork 的子呼叫，要把它放在下一層的 def 裡。*（probe：`shapes`：`kid` 有 `FID_KID` 與 `WL_AGAIN(FID_KID)`）*
- **F7** `INLINE` 是對 C 編譯器的請求，不是保證。一串以 `o[]` 傳遞寬狀態的小 `spin_N`，只有在 C 編譯器願意把每一環都 inline 時才會合成一個迴圈。呼叫點變多，它就不再這樣做。所以多個運算共用的逐元素輔助函式，在完整程式裡可能每個元素花好幾次真正的呼叫，在只連結一個運算的程式裡卻沒有成本。逐元素的路徑要是一個自己的小迴圈，寬狀態每段只前進一次。量測要在連結了其他使用者的程式裡做。*（量測，無探針：非連續複製的逐元素游標單獨是每元素 1.3 ns，連結 combine 與 reduce 後是 6.7 ns，Clang 19 只有在兩個 inline 門檻都設為 20000 時才恢復，設為 5000 反而更慢）*
- **F8** 平行 let 只有在兩個值都還是呼叫時才是 fork。它的 join 是 C 裡的 `FID_NAME_J<n>`，檢查 join 與呼叫 segment，才能確認 fork。erased 的引數留在它的呼叫裡（`anf`）。*（probe：`fork_erased_last`，來自 [nuxyel](https://github.com/nuxyel), [bend-ml](https://github.com/nuxyel/bend-ml)）*

- **F9** 原生尾迴圈展開 `WL_SPIN`，每輪增加錯誤輪詢計數，每 4,096 輪讀取錯誤狀態。clang 證明輪詢沒有影響時可以刪除它。flat 不代表一定能向量化，要檢查迴圈與 clang 的向量化診斷。*（probe：`shapes`）*
- **F10** CPU 的每個 worklist segment 都是 `noinline` C 函式。`WL_JMP` 是直接的強制尾呼叫，`WL_DYN` 則索引 `wl_tab` 作間接尾呼叫。即使不配置新的 frame，接續分派仍有暫存器、堆疊與跳轉成本。*（probe：`shapes`）*

- **F11** 密集的 U32 literal match 可以成為常數表。`mat_rows` 要求命中數超過 `0..max` 的一半且只有一個 default 本體，`emit_row` 還必須把每列結果化成允許的 word 運算式。`TAB_AT` 把超界索引夾到最後的 default 格。稀疏或非常數案例保留分支。這是 clang 之前的 Bend 最佳化。*（官方測試：`u32_table_popcount`，動態執行超過最後 literal 的索引）*

- **F12** 非 flat callee 若只有一個可達引用、尾端目的地與回傳配置符合條件，且不是 bang 或 foreign 呼叫，也能融合進 caller。`file_book` 計數呼叫點，`emit_body/emit_fuse` 在 caller 產生本體，最後依可達性移除未使用 segment。`fuse_once` 沒有 `FID_WORK`，兩次呼叫的 `fuse_twice` 則保留。因此沒有 FID 不能單獨證明 flat，也不能證明沒有排程工作。*（probe：`fuse_once`、`fuse_twice`）*

## G. 編譯器：版面與共享

- **G1** 版面（`lay_of`）。非遞迴的資料型別攤平成機器字，不配置記憶體。U32 與 F32 佔 32 位元，Nat 佔 64 位元。`Array`、`IO.OP` 與遞迴型別（list、樹）是 heap 上的 box。寬度超過 `WIDE = 247` 個字的型別變成 box。參數合計超過時，多字的參數變成 box。延續段（continuation segment）帶著跨過非 flat 呼叫的每個存活值，不會這樣變成 box：存活的字超過 247 個，編譯就以 `an arity over 247` 失敗（`seg_open`）。數量不定的張量放在一個遞迴型別裡，它是一個 box。
- **G2** 共享（`facts_hot`）。一個值被使用超過一次，它的型別就被標成 hot。該型別的建構子變成共享的，對它的 match 改走會檢查引用計數的 `ctr_take`，而且 hot 會傳到欄位的型別。*（probe：`share_none` 沒有共享，`share_list` 把一個 list 用兩次，部分建構子變成共享）*本專案在 `Views.View` 上遇過：一個定義把同一個 view 交給另外兩個定義，以及迴圈把 view 當參數帶過每次迭代，都讓所有 `View` 建構子變成共享。前者使四執行緒模型在兩次交替量測中為 136.7 對 143.2 ms、132.1 對 145.3 ms（四分位距約 25 ms）。改成每次使用各建一次 view 後為 130.5 對 130.9 ms。在參數位置直接建出的 view 會讓該參數對所有呼叫者都成為擁有的（G8）。之後呼叫者把一個在別處也用到的 `+` view 傳下去時就會共享它，所以 reduction 先把新 view 綁定到一個名字再傳。`scripts/check_generated_c.py` 對模型、benchmark 與整合入口會拒絕共享的 `View`。launcher 對每次編譯做的檢查不含這一項，因為程式自己的定義可能共享 view。
- **G3** `facts_hot` 追蹤資料型別欄位、lambda 本體與卡住的型別族分支。強制分析仍無法解析的 `All`、`Var` 或 `App` 時，若不能透過 erased 參數把共享需求傳回呼叫者，就設定 `"*"`。後續走訪 binding 時會強制加熱其型別，**不是**直接把每列 `CID_T` 設為共享。`share_existential` 沿 erased 欄位的後備路徑：Hide／WCon／Bool 為 hot，Con 不是（4/22）。具體欄位對照則加熱 Hide／Con／Nil（3/22）。這些旗標展示不同的保守分析路徑，不是任意 existential 程式的正確性證明。*（probe：`share_existential`、`share_concrete`，另有可解析的泛型／型別族探針）*
- **G4** 綁定會計算自己的使用次數：最後一次使用取走值，之前的使用是共享。攤平的值被共享時，複製的是字，不是節點。
- **G5** 樣板與閉包。`~` 樣板引數由 checker 實例化，所以編譯器看到的是直接呼叫。*（probe：`call_template`：整條鏈都是 `spin`）* 當成一般參數傳入的函式是閉包，而且**呼叫閉包的 def 不是 flat**。*（probe：`call_closure`）* 存在 record 裡的函式，量測結果比樣板慢 5.85–6.5 倍。
- **G6** 泛型的 `~Context: Data` 配 `+context`，只會把實例化後的那個型別標成 hot。*（probe：`share_template_parameter`）*
- **G7** 樣板在每個使用處代入封閉引數的運算式。`template_record` 讀同一個計算欄位兩次，會生成兩次計數迴圈呼叫，clang 之後兩個迴圈仍在。要共享結果時，先算一次再把值當 `+` 參數傳入。閉包只延後計算，不會記住結果，還會增加 G5 的分派成本。*（probe：`template_record`，來自 [nuxyel](https://github.com/nuxyel), [bend-ml](https://github.com/nuxyel/bend-ml)）*

- **G8** 只要單調累積的 `own`、`hot` 或 `stat` 新增項目，就再次產生程式碼。每輪重建借用／出借與 segment。`graph_close` 傳播間接出借，未出借的借用參數轉為 owned。配置快取不屬於這個收斂判斷。`own_single` 與 `borrow_repeated` 保留相同 `scan` 本體：前者 native code 使用 `ctr_take/term_sink/spare_free`，後者用 `term_peek` 且不釋放。Array 儲存格使用元素資料型別的**開放**配置，讓呼叫者一致。無法解析的裸元素型別會被拒絕，JS 也一樣。*（原始碼：`compile_book/lay_el`，官方測試：`array_generic_cell`、`array_open_element`、`array_open_element_family`）*

- **G9** 常數與借用節點的釋放規則不同。`ctr_build` 把靜態欄位放進 `STAT_IMG`，並將建構子加入 `stat`，`node_fields` 對 owned 且 hot **或 static** 的建構子使用 `ctr_take`。所以只看到 `ctr_take` 不能證明執行期共享。packed 的零／單 word 節點與靜態位址對 `term_sink` 而言是 trivial。owned heap 子樹則以迭代和引用計數釋放。*（原始碼：`ctr_build/node_fields/term_triv/term_drop`，probe：`own_single`、`borrow_repeated`）*

## H. runtime：排程、記憶體與檔案

- **H1** fork（平行 let）產生一個 join task，每個呼叫是一個 kid。kid 依序執行時，一個 frame 供所有步驟使用。
- **H2** worker 在 ring 上輪轉。成長輪（grow）中的 lane 會跳過不含 fork 的 task（`monk_step` 裡的 `fid_nofk`）。host 一個 ring 接一個 ring 地成長，然後清空。host 執行緒也在每一輪領取單位，另有 `threads - 1` 個 pool 執行緒，執行緒睡著之前會先讓出有限次數（`POOL_WAIT`）。沒有 work stealing。所以一次平行就是一輪成長加一輪工作，成長輪本來不該完成 leaf。planner 要以完整的輪次計算。
- **H3** 平行 let 轉移其引數。傳遞唯一擁有的 Array 本身不代表產生別名。明確的 `Array.fork` 讓兩個 handle 指向同一區塊，且標為 `@unsafe`，使用者過不了 `--check-only`（C10）。共享 handle 使用原子 redirect 計數。`Array.clone` 用 `blk_copy` 複製外層區塊。boxed 元素增加保留／共享，不會遞迴深拷貝。*（probe：`array_fork_trust`，原始碼：`blk_copy/blk_fill/blk_keep`）*
- **H4** **不含 fork 的葉 task 會延後到 work turn。** 機制在 `cube_run`、`row_grow`、`monk_step`、`task_deal`、`emit_jump`：
  1. 在最上層的 fork，host 把兩個 kid 放到**兩個不同的 row**（`ring_flip`）。
  2. 成長輪中，一個 row 由一個 worker 負責。它執行可能 fork 的 task，這些 task fork 出來的 kid 留在**同一個 row**（`ring_pick`）。當一輪用到的 row 少於執行緒，而某次成長超過 1 ms，之後 fork 出來的 kid 會經共用的游標分到所有 row，成長輪重複到沒有再分出為止（`row_grow`、`cube_run`）。
  3. 如果 kid 就是會 fork 的遞迴本身，而 leaf 是被 inline 進去的 flat def，這個 worker 會**在成長輪裡做完一整個 leaf**。做完這個 task 之後，這個 row 的成長輪就結束。
  4. 工作輪中，一個 row 的 128 個 ring 分成若干單位（`pool_step`：使用的 row 少時單位多，row 多時是八個），各執行緒以原子計數器領取下一個單位。剩下的 leaf 在這裡執行。

  所以深度二、四個 leaf 的樹，在成長輪用兩個 worker 做兩個 leaf，在工作輪用兩個 worker 做另外兩個。總時間是兩個 leaf 的時間，不是一個。16 個 leaf 時是兩個在成長輪、14 個在工作輪。

  做法：子節點是 leaf 時，平行 let 呼叫一個**既不 fork、也不呼叫閉包**的 def。依 F6 它會變成 task，它在 `FID_T` 表中那一列的 mask 2 會被設定（位元索引 1，`fork_flags.py` 會印出來）。成長輪會跳過它，所有 leaf 都在工作輪的不同單位中執行。熱迴圈放在再下一層的 flat def（F6）。避開閉包呼叫是充分條件，不是必要條件：`Clo~apply` 連到所有已產生的閉包，只有這張圖能到達 fork 才標為 may-fork。`call_closure` 的 RUN 與 CLO_APPLY 都是 fork-free，`call_closure_fork` 則都為 may-fork。flat 與 fork-free 是不同分析。

  *（probe：`leaf_inline` 與 `leaf_task`。`verify.py` 檢查它們的 task 在 `FID_T` 的旗標。`leaf_workers.py` 以 4、8、16 個 leaf 在二與四執行緒執行它們，要求 inline 的寫法有兩個 leaf 在成長輪執行，不含 fork 的寫法為零個。各四個 leaf 時，在一台 8 執行緒的虛擬機上量到：二執行緒 97.6 與 97.7 ms，四執行緒 **97.5 ms 對 48.9 ms**。inline 的寫法在四執行緒並不比二執行緒快。）*

- **H5** Array 或 Buffer 是一個 2^c 格的區塊（`blk_new`），從它那一級的 free list 取得（`heap_alloc`）。n 格的區塊取下一個二的次方，最多是實際格數的兩倍。釋放的區塊回到它那一級的 list，只供同一級使用。區塊不會被切開或合併。記憶體要以區塊計算，不是以元素計算。*（原始碼，來自 [nuxyel](https://github.com/nuxyel), [bend-ml](https://github.com/nuxyel/bend-ml)）*
- **H6** `File.read_bytes` 與 `File.read_at` 回傳每個位元組一格的 list，所以建立與走訪這個 list 就是載入的成本。分段讀取，每段轉換完再讀下一段。用外部程式碼寫的讀取器不經過 list，但匯入它的檔案過不了 `--verdict`（bendlang/bend#1573 請求在 Base 加入讀成 Array 的函式）。*（原始碼，來自 [nuxyel](https://github.com/nuxyel), [bend-ml](https://github.com/nuxyel/bend-ml)）*

- **H7** Array 索引用區塊類別導出的遮罩環繞。兩個元素的 Array 用索引 3 讀到索引 1。有效索引是函式庫的合約，不是執行期的越界例外。*（probe：`array_index_wrap`）*
- **H8** 把 Array 拆成兩半會用 `blk_half` 配置並複製各半部，與共享同一儲存的 `Array.fork` handle 不同。Buffer 的兩個 32 位元儲存格佔一個 64 位元 heap word，複製格數與配置字數要分別計算。*（probe：`array_blocks`）*

- **H9** 配置按 lane 與大小 class 管理。host 依序取 HOT、暫存的 COLD 世代、共享 bank，最後才拿新的 quantum。累積一整個釋放世代後移往 COLD／bank。回到 free list 不等於把記憶體還給作業系統。共享 boxed 值另有 redirect cell，使用 24-bit 副本計數，溢位會失敗。*（原始碼：`heap_alloc_miss/heap_hand/heap_free/rfc_bump`）*

- **H10** host 與 device 共用生成語義，但 dispatch 機制不同。host segment 使用 `musttail` 與可用時的 `preserve_none`，device segment 是單一 kernel 的 case。bang 與閉包可達性決定裝置程式碼。`dev` 分成 grow／work／bank-pack 等 pass，host 組合語言結果沒有量到這些 pass。JS 使用直接呼叫／尾端 trampoline 與循序 let，沒有 C worklist 的平行執行。*（原始碼：`compile_book`、runtime Dialect/Dev、`js_open/js_def/run_loop`，已追查原始碼，未執行 GPU）*

## I. 對設計的意義

| 決定 | 機制與證據 |
| --- | --- |
| 分開樹的排程與葉的迴圈 | fork、閉包呼叫與非尾遞迴會阻止 flat（F1、G5）。葉用樣板傳入，設不含 fork 的 task 邊界，再於下一層放原生迴圈（F6、H4） |
| 比較整個連結後的程式 | 檢查建構子共享、task 旗標、留下的呼叫與向量化診斷。可到達的呼叫者會改變擁有權與 clang 內聯（F7、G8） |
| 先數工作再解釋時間 | 動態指令數計算執行過的工作，靜態組譯計數描述程式大小與迴圈本體。同樣指令數但週期不同，也可能來自快取未命中、競爭、頻率或分支預測，不能據此認定是排程或對齊 |
| 明確保留已算出的值 | 樣板運算式可能執行多次（G7）。Array 複製與模式拆解會複製儲存（H3、H8），byte 讀取會配置串列（H6） |
| 證明界限保持抽象 | 不展開定義的比較可避開計算，但後續轉換仍可能暴露大的封閉 Nat（D2、E1、E2） |

`verify.py --inspect GENERATED.c` 保留一個連結後程式的組合語言與診斷。比較兩個程式時，輸入、工作者數、目標與旗標都要相同。先看指令數的做法來自 [nuxyel](https://github.com/nuxyel)、[bend-ml](https://github.com/nuxyel/bend-ml)。

- **record 欄位裡的函式仍是閉包。** `traversal_array.update` 直接做 F32 算術時是原生迴圈。同樣的更新經過 record 欄位時，每個元素進入 `CLO_APPLY` 兩次，並配置一個 word 的閉包（F1、F6、G5、G7）。`benchmarks/checks/measure_arithmetic_dispatch.py` 量這兩種。

## J. lint.py 會找到什麼

| 規則 | 找到的寫法 |
| --- | --- |
| A2 | let 之後的 match 或拆解 |
| A3 | match 或拆解一個計算出來的值 |
| A7 | pattern 後面接 `[]` |
| C1 | `~` 參數不在最前面 |
| C2 | 呼叫的 `~` 引數過多或不是連續的前綴，會解析 import 別名 |
| D4 | let 一個沒有型別標註的建構子 |
| E1 | `U32.to_nat(大數字)`，或六位數以上的 Nat 字面值 |
| E1? | `U32.to_nat(f(X.limits()))`：把預設的機器界限轉成 Nat。問號表示有風險，不一定失敗 |

在 `lib` 與 `convolution` 已驗收的原始碼上，它不回報任何一筆。它只看寫法：找不到用量錯誤（B1）、建構子欄的順序與型別不符，C2 也只檢查明寫 `~` 且解析得到宣告的呼叫。`verify.py` 拿 lint 的結果和 checker probe 對照。
