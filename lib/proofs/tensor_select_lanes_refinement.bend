# The ternary lane rounds and three tail kernels observed through content.
import Base
import ../tensor_lanes.bend as Lanes
import ../traversal_lanes.bend as Unary
import ../traversal_cells.bend as Cells
import ../kernel_tile.bend as Tile
import ../tensor_operations.bend as Operations
import ../tensor_view.bend as Views
import ../storage_buffer.bend as Storage
import ../math_compare.bend as Compare
import ./tensor_lanes_refinement.bend as Binary
import ./traversal_lanes_refinement.bend as UnaryLaws
import ./traversal_cells_refinement.bend as CellLaws
import ./traversal_map_readback.bend as MapReadback
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_clone_refinement.bend as CloneLaws
import ./storage_refinement.bend as StorageLaws
import ./word_encoding.bend as WordEncoding
import ./nat_division.bend as Division

# A tail kernel's two contents. This is the existing binary tail refinement,
# projected before the binary wrapper chooses its result owner.
def scalar_tail(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  {CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,left),Buffer.pack(~F32,right))) ==
    (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~Binary.pair_value(~operation),count,U32.from_nat(start),Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}:
  Equal.trans(Buffer.Content<F32> & Buffer.Content<F32>,
    CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,left),Buffer.pack(~F32,right))),
    Binary.pair_contents(Lanes.pair_finish(~operation,count,Cells.State{Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},Unary.scratch(),U32.from_nat(start),0,Unit{}})),
    (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~Binary.pair_value(~operation),count,U32.from_nat(start),Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)),
    Equal.sym(Buffer.Content<F32> & Buffer.Content<F32>,Binary.pair_contents(Lanes.pair_result(Operations.combine_kernel(~F32,~operation,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,left),Buffer.pack(~F32,right)))),
      CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,left),Buffer.pack(~F32,right))),
      Binary.pair_result_contents(Operations.combine_kernel(~F32,~operation,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,left),Buffer.pack(~F32,right)))),
    %Equal.sym(Array<Tile.Tile8>,Unary.scratch(),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0))),StorageLaws.fresh(Tile.Tile8,6n,Tile.filled(0.0))) :
      {Binary.pair_contents(Lanes.pair_finish(~operation,count,Cells.State{Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},_,U32.from_nat(start),0,Unit{}})) ==
        (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~Binary.pair_value(~operation),count,U32.from_nat(start),Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}
    Binary.pair_tail(~operation,count,start,0,left,right,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0)),small))

def first_content(pair: Buffer.Content<F32> & Buffer.Content<F32>) -> Buffer.Content<F32>:
  (left,right) = pair
  left

def second_content(pair: Buffer.Content<F32> & Buffer.Content<F32>) -> Buffer.Content<F32>:
  (left,right) = pair
  right

def mapped(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+left: Buffer.Content<F32>,+right: Buffer.Content<F32>) -> Buffer.Content<F32>:
  Buffer.Content{left,length} = left
  Buffer.Content{right,right_length} = right
  Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~Binary.pair_value(~operation),count,U32.from_nat(start),right,left),length}

type TailStep<-operation: F32 -> F32 -> F32,-count: Nat,-start: Nat,-left: Buffer.Model<F32>,-right: Buffer.Model<F32>,-pair: Storage.Buffer<F32> & Storage.Buffer<F32>> is Data:
  TailStep{output: Buffer.Model<F32>,source: Buffer.Model<F32>,equation: {(Buffer.pack(~F32,output),Buffer.pack(~F32,source)) == pair : Storage.Buffer<F32> & Storage.Buffer<F32>},
    output_content: {Buffer.content(~F32,output) == mapped(~operation,count,start,Buffer.content(~F32,left),Buffer.content(~F32,right)) : Buffer.Content<F32>},
    source_content: {Buffer.content(~F32,source) == Buffer.content(~F32,right) : Buffer.Content<F32>}}

def tail_observed(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,-pair: Storage.Buffer<F32> & Storage.Buffer<F32>,
  observation: CloneLaws.Observation<F32,pair>,
  +content: {CloneLaws.contents(~F32,pair) == (mapped(~operation,count,start,Buffer.content(~F32,left),Buffer.content(~F32,right)),Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}) -> TailStep<operation,count,start,left,right,pair>:
  CloneLaws.Observation{+output,+source,+equation} = observation
  +observed = {
    %Buffer.packed_contents(~F32,output) : {(_,Buffer.content(~F32,source)) == (mapped(~operation,count,start,Buffer.content(~F32,left),Buffer.content(~F32,right)),Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}
    %Buffer.packed_contents(~F32,source) : {(Buffer.contents(~F32,Buffer.pack(~F32,output)),_) == (mapped(~operation,count,start,Buffer.content(~F32,left),Buffer.content(~F32,right)),Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}
    %Equal.sym(Storage.Buffer<F32> & Storage.Buffer<F32>,(Buffer.pack(~F32,output),Buffer.pack(~F32,source)),pair,equation) : {CloneLaws.contents(~F32,_) == (mapped(~operation,count,start,Buffer.content(~F32,left),Buffer.content(~F32,right)),Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}
    content : {(Buffer.content(~F32,output),Buffer.content(~F32,source)) == (mapped(~operation,count,start,Buffer.content(~F32,left),Buffer.content(~F32,right)),Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}}
  TailStep{output,source,equation,
    Equal.cong(Buffer.Content<F32> & Buffer.Content<F32>,Buffer.Content<F32>,first_content,(Buffer.content(~F32,output),Buffer.content(~F32,source)),(mapped(~operation,count,start,Buffer.content(~F32,left),Buffer.content(~F32,right)),Buffer.content(~F32,right)),observed),
    Equal.cong(Buffer.Content<F32> & Buffer.Content<F32>,Buffer.Content<F32>,second_content,(Buffer.content(~F32,output),Buffer.content(~F32,source)),(mapped(~operation,count,start,Buffer.content(~F32,left),Buffer.content(~F32,right)),Buffer.content(~F32,right)),observed)}

def tail_step(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  TailStep<operation,count,start,left,right,Operations.combine_kernel(~F32,~operation,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,left),Buffer.pack(~F32,right))>:
  Buffer.Model{+lt,+ll,+lc} = left
  Buffer.Model{+rt,+rl,+rc} = right
  +left = {Buffer.Model{lt,ll,lc} : Buffer.Model<F32>}
  +right = {Buffer.Model{rt,rl,rc} : Buffer.Model<F32>}
  tail_observed(~operation,count,start,left,right,Operations.combine_kernel(~F32,~operation,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,left),Buffer.pack(~F32,right)),
    CloneLaws.observe_pair(~F32,Operations.combine_kernel(~F32,~operation,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,left),Buffer.pack(~F32,right))),scalar_tail(~operation,count,start,left,right,small))

def tail_model(+count: Nat,+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>) -> Buffer.Content<F32>:
  mapped(~Compare.union,count,start,mapped(~Compare.selected,count,start,yes,mask),mapped(~Compare.rejected,count,start,no,mask))

def joined_content(+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,+count: Nat,+start: Nat,
  evidence: TailStep<Compare.union,count,start,yes,no,Operations.combine_kernel(~F32,~Compare.union,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,yes),Buffer.pack(~F32,no))>) ->
  {Buffer.contents(~F32,Lanes.select_result(Lanes.select_joined(Buffer.pack(~F32,mask),Operations.combine_kernel(~F32,~Compare.union,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,yes),Buffer.pack(~F32,no))))) == mapped(~Compare.union,count,start,Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}:
  TailStep{+output,+source,+equation,output_content,source_content} = evidence
  %equation : {Buffer.contents(~F32,Lanes.select_result(Lanes.select_joined(Buffer.pack(~F32,mask),_))) == mapped(~Compare.union,count,start,Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,output)),Buffer.content(~F32,output),Buffer.packed_contents(~F32,output)) : {_ == mapped(~Compare.union,count,start,Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}
  output_content

def rejected_content(+count: Nat,+start: Nat,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,+mask: Buffer.Model<F32>,
  evidence: TailStep<Compare.rejected,count,start,no,mask,Operations.combine_kernel(~F32,~Compare.rejected,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,no),Buffer.pack(~F32,mask))>,
  +small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  {Buffer.contents(~F32,Lanes.select_result(Lanes.select_rejected(count,U32.from_nat(start),Buffer.pack(~F32,yes),Operations.combine_kernel(~F32,~Compare.rejected,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,no),Buffer.pack(~F32,mask))))) == mapped(~Compare.union,count,start,Buffer.content(~F32,yes),mapped(~Compare.rejected,count,start,Buffer.content(~F32,no),Buffer.content(~F32,mask))) : Buffer.Content<F32>}:
  TailStep{+output,+source,+equation,+output_content,source_content} = evidence
  %equation : {Buffer.contents(~F32,Lanes.select_result(Lanes.select_rejected(count,U32.from_nat(start),Buffer.pack(~F32,yes),_))) == mapped(~Compare.union,count,start,Buffer.content(~F32,yes),mapped(~Compare.rejected,count,start,Buffer.content(~F32,no),Buffer.content(~F32,mask))) : Buffer.Content<F32>}
  %output_content : {Buffer.contents(~F32,Lanes.select_result(Lanes.select_joined(Buffer.pack(~F32,source),Operations.combine_kernel(~F32,~Compare.union,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,yes),Buffer.pack(~F32,output))))) == mapped(~Compare.union,count,start,Buffer.content(~F32,yes),_) : Buffer.Content<F32>}
  joined_content(source,yes,output,count,start,tail_step(~Compare.union,count,start,yes,output,small))

def chosen_content(+count: Nat,+start: Nat,+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,
  evidence: TailStep<Compare.selected,count,start,yes,mask,Operations.combine_kernel(~F32,~Compare.selected,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,yes),Buffer.pack(~F32,mask))>,
  +small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  {Buffer.contents(~F32,Lanes.select_result(Lanes.select_chosen(count,U32.from_nat(start),Buffer.pack(~F32,no),Operations.combine_kernel(~F32,~Compare.selected,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,yes),Buffer.pack(~F32,mask))))) == tail_model(count,start,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}:
  TailStep{+output,+source,+equation,+output_content,+source_content} = evidence
  %equation : {Buffer.contents(~F32,Lanes.select_result(Lanes.select_chosen(count,U32.from_nat(start),Buffer.pack(~F32,no),_))) == tail_model(count,start,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}
  %source_content : {Buffer.contents(~F32,Lanes.select_result(Lanes.select_rejected(count,U32.from_nat(start),Buffer.pack(~F32,output),Operations.combine_kernel(~F32,~Compare.rejected,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,no),Buffer.pack(~F32,source))))) == mapped(~Compare.union,count,start,mapped(~Compare.selected,count,start,Buffer.content(~F32,yes),Buffer.content(~F32,mask)),mapped(~Compare.rejected,count,start,Buffer.content(~F32,no),_)) : Buffer.Content<F32>}
  %output_content : {Buffer.contents(~F32,Lanes.select_result(Lanes.select_rejected(count,U32.from_nat(start),Buffer.pack(~F32,output),Operations.combine_kernel(~F32,~Compare.rejected,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),Buffer.pack(~F32,no),Buffer.pack(~F32,source))))) == mapped(~Compare.union,count,start,_,mapped(~Compare.rejected,count,start,Buffer.content(~F32,no),Buffer.content(~F32,source))) : Buffer.Content<F32>}
  rejected_content(count,start,output,no,source,tail_step(~Compare.rejected,count,start,no,source,small),small)

def tail(+count: Nat,+start: Nat,+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,+scratch: Tree.Tree<Tile.Tile8>,+slot: U32,
  +small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  {Buffer.contents(~F32,Lanes.select_result(Lanes.select_finish(count,Cells.State{Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)},Tree.pack(Tile.Tile8,scratch),U32.from_nat(start),slot,Unit{}}))) == tail_model(count,start,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}:
  chosen_content(count,start,mask,yes,no,tail_step(~Compare.selected,count,start,yes,mask,small),small)

# The read-only mask and no owners surround the yes owner that scatter writes.
type Contents is Data:
  Contents{mask: Buffer.Content<F32>,yes: Buffer.Content<F32>,no: Buffer.Content<F32>}

def contents(owners: Lanes.Selected) -> Contents:
  Lanes.Selected{mask,yes,no} = owners
  Contents{Buffer.contents(~F32,mask),Buffer.contents(~F32,yes),Buffer.contents(~F32,no)}

def read_contents(result: Lanes.Selected & Tile.Tile8) -> Contents & Tile.Tile8:
  CellLaws.observe_gather(~Lanes.Selected,~Contents,~Tile.Tile8,~contents,result)

def read_model(mask: Buffer.Content<F32> & Tile.Tile8,yes: Buffer.Content<F32> & Tile.Tile8,no: Buffer.Content<F32> & Tile.Tile8) -> Contents & Tile.Tile8:
  (mask,a) = mask
  (yes,b) = yes
  (no,c) = no
  (Contents{mask,yes,no},Lanes.select_tile(a,b,c))

def reads(mask: Storage.Buffer<F32> & Tile.Tile8,yes: Storage.Buffer<F32> & Tile.Tile8,no: Storage.Buffer<F32> & Tile.Tile8) ->
  {read_contents(Lanes.select_reads(mask,yes,no)) == read_model(Binary.observed_read(mask),Binary.observed_read(yes),Binary.observed_read(no)) : Contents & Tile.Tile8}:
  (mask,a) = mask
  (yes,b) = yes
  (no,c) = no
  {==}

def cell(+mask: Tree.Tree<F32>,+yes: Tree.Tree<F32>,+no: Tree.Tree<F32>,+position: U32) -> Tile.Tile8:
  Lanes.select_tile(UnaryLaws.cell(~Unit,~Lanes.identity,mask,position,Unit{}),UnaryLaws.cell(~Unit,~Lanes.identity,yes,position,Unit{}),UnaryLaws.cell(~Unit,~Lanes.identity,no,position,Unit{}))

def gather(+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,+position: U32) ->
  {read_contents(Lanes.select_gather(Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)},position,Unit{})) ==
    (Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},cell(Buffer.tree(~F32,mask),Buffer.tree(~F32,yes),Buffer.tree(~F32,no),position)) : Contents & Tile.Tile8}:
  Equal.trans(Contents & Tile.Tile8,
    read_contents(Lanes.select_gather(Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)},position,Unit{})),
    read_model(Binary.observed_read(Lanes.read(Buffer.pack(~F32,mask),position)),Binary.observed_read(Lanes.read(Buffer.pack(~F32,yes),position)),Binary.observed_read(Lanes.read(Buffer.pack(~F32,no),position))),
    (Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},cell(Buffer.tree(~F32,mask),Buffer.tree(~F32,yes),Buffer.tree(~F32,no),position)),
    reads(Lanes.read(Buffer.pack(~F32,mask),position),Lanes.read(Buffer.pack(~F32,yes),position),Lanes.read(Buffer.pack(~F32,no),position)),
    %Equal.sym(Buffer.Content<F32> & Tile.Tile8,Binary.observed_read(Lanes.read(Buffer.pack(~F32,mask),position)),(Buffer.content(~F32,mask),UnaryLaws.cell(~Unit,~Lanes.identity,Buffer.tree(~F32,mask),position,Unit{})),Binary.read(mask,position)) :
      {read_model(_,Binary.observed_read(Lanes.read(Buffer.pack(~F32,yes),position)),Binary.observed_read(Lanes.read(Buffer.pack(~F32,no),position))) ==
        (Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},cell(Buffer.tree(~F32,mask),Buffer.tree(~F32,yes),Buffer.tree(~F32,no),position)) : Contents & Tile.Tile8}
    %Equal.sym(Buffer.Content<F32> & Tile.Tile8,Binary.observed_read(Lanes.read(Buffer.pack(~F32,yes),position)),(Buffer.content(~F32,yes),UnaryLaws.cell(~Unit,~Lanes.identity,Buffer.tree(~F32,yes),position,Unit{})),Binary.read(yes,position)) :
      {read_model((Buffer.content(~F32,mask),UnaryLaws.cell(~Unit,~Lanes.identity,Buffer.tree(~F32,mask),position,Unit{})),_,Binary.observed_read(Lanes.read(Buffer.pack(~F32,no),position))) ==
        (Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},cell(Buffer.tree(~F32,mask),Buffer.tree(~F32,yes),Buffer.tree(~F32,no),position)) : Contents & Tile.Tile8}
    %Equal.sym(Buffer.Content<F32> & Tile.Tile8,Binary.observed_read(Lanes.read(Buffer.pack(~F32,no),position)),(Buffer.content(~F32,no),UnaryLaws.cell(~Unit,~Lanes.identity,Buffer.tree(~F32,no),position,Unit{})),Binary.read(no,position)) :
      {read_model((Buffer.content(~F32,mask),UnaryLaws.cell(~Unit,~Lanes.identity,Buffer.tree(~F32,mask),position,Unit{})),(Buffer.content(~F32,yes),UnaryLaws.cell(~Unit,~Lanes.identity,Buffer.tree(~F32,yes),position,Unit{})),_) ==
        (Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},cell(Buffer.tree(~F32,mask),Buffer.tree(~F32,yes),Buffer.tree(~F32,no),position)) : Contents & Tile.Tile8}
    {==})

def scatter_model(owners: Contents,position: U32,tile: Tile.Tile8) -> Contents:
  Contents{mask,Buffer.Content{yes,length},no} = owners
  Contents{mask,Buffer.Content{Binary.written(yes,position,tile),length},no}

def packed_contents(+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>) ->
  {contents(Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)}) == Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)} : Contents}:
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,mask)),Buffer.content(~F32,mask),Buffer.packed_contents(~F32,mask)) :
    {Contents{_,Buffer.contents(~F32,Buffer.pack(~F32,yes)),Buffer.contents(~F32,Buffer.pack(~F32,no))} == Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)} : Contents}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,yes)),Buffer.content(~F32,yes),Buffer.packed_contents(~F32,yes)) :
    {Contents{Buffer.content(~F32,mask),_,Buffer.contents(~F32,Buffer.pack(~F32,no))} == Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)} : Contents}
  Equal.cong(Buffer.Content<F32>,Contents,content => Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),content},Buffer.contents(~F32,Buffer.pack(~F32,no)),Buffer.content(~F32,no),Buffer.packed_contents(~F32,no))

def scatter(+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,+position: U32,+tile: Tile.Tile8) ->
  {contents(Lanes.select_scatter(Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)},position,tile)) ==
    scatter_model(Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},position,tile) : Contents}:
  Buffer.Model{+yt,+yl,+yc} = yes
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,mask)),Buffer.content(~F32,mask),Buffer.packed_contents(~F32,mask)) :
    {Contents{_,Buffer.contents(~F32,Lanes.write(Buffer.pack(~F32,yes),position,tile)),Buffer.contents(~F32,Buffer.pack(~F32,no))} == Contents{Buffer.content(~F32,mask),Buffer.Content{Binary.written(yt,position,tile),yl},Buffer.content(~F32,no)} : Contents}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,no)),Buffer.content(~F32,no),Buffer.packed_contents(~F32,no)) :
    {Contents{Buffer.content(~F32,mask),Buffer.contents(~F32,Lanes.write(Buffer.pack(~F32,yes),position,tile)),_} == Contents{Buffer.content(~F32,mask),Buffer.Content{Binary.written(yt,position,tile),yl},Buffer.content(~F32,no)} : Contents}
  Equal.cong(Buffer.Content<F32>,Contents,content => Contents{Buffer.content(~F32,mask),content,Buffer.content(~F32,no)},Buffer.contents(~F32,Lanes.write(Buffer.pack(~F32,yes),position,tile)),Buffer.Content{Binary.written(yt,position,tile),yl},Binary.write(yes,position,tile))

def scatter_seen(-mask: Storage.Buffer<F32>,-yes: Storage.Buffer<F32>,-no: Storage.Buffer<F32>,+position: U32,+tile: Tile.Tile8,
  mask_seen: Buffer.Observation<F32,mask>,yes_seen: Buffer.Observation<F32,yes>,no_seen: Buffer.Observation<F32,no>) ->
  {contents(Lanes.select_scatter(Lanes.Selected{mask,yes,no},position,tile)) == scatter_model(contents(Lanes.Selected{mask,yes,no}),position,tile) : Contents}:
  Buffer.Observation{+m,mask_equal} = mask_seen
  Buffer.Observation{+y,yes_equal} = yes_seen
  Buffer.Observation{+n,no_equal} = no_seen
  %mask_equal : {contents(Lanes.select_scatter(Lanes.Selected{_,yes,no},position,tile)) == scatter_model(contents(Lanes.Selected{_,yes,no}),position,tile) : Contents}
  %yes_equal : {contents(Lanes.select_scatter(Lanes.Selected{Buffer.pack(~F32,m),_,no},position,tile)) == scatter_model(contents(Lanes.Selected{Buffer.pack(~F32,m),_,no}),position,tile) : Contents}
  %no_equal : {contents(Lanes.select_scatter(Lanes.Selected{Buffer.pack(~F32,m),Buffer.pack(~F32,y),_},position,tile)) == scatter_model(contents(Lanes.Selected{Buffer.pack(~F32,m),Buffer.pack(~F32,y),_}),position,tile) : Contents}
  Equal.trans(Contents,contents(Lanes.select_scatter(Lanes.Selected{Buffer.pack(~F32,m),Buffer.pack(~F32,y),Buffer.pack(~F32,n)},position,tile)),
    scatter_model(Contents{Buffer.content(~F32,m),Buffer.content(~F32,y),Buffer.content(~F32,n)},position,tile),
    scatter_model(contents(Lanes.Selected{Buffer.pack(~F32,m),Buffer.pack(~F32,y),Buffer.pack(~F32,n)}),position,tile),scatter(m,y,n,position,tile),
    Equal.cong(Contents,Contents,owners => scatter_model(owners,position,tile),Contents{Buffer.content(~F32,m),Buffer.content(~F32,y),Buffer.content(~F32,n)},contents(Lanes.Selected{Buffer.pack(~F32,m),Buffer.pack(~F32,y),Buffer.pack(~F32,n)}),
      Equal.sym(Contents,contents(Lanes.Selected{Buffer.pack(~F32,m),Buffer.pack(~F32,y),Buffer.pack(~F32,n)}),Contents{Buffer.content(~F32,m),Buffer.content(~F32,y),Buffer.content(~F32,n)},packed_contents(m,y,n))))

def scatter_native(owners: Lanes.Selected,+position: U32,+tile: Tile.Tile8) ->
  {contents(Lanes.select_scatter(owners,position,tile)) == scatter_model(contents(owners),position,tile) : Contents}:
  Lanes.Selected{mask,yes,no} = owners
  scatter_seen(mask,yes,no,position,tile,Buffer.observe(~F32,mask),Buffer.observe(~F32,yes),Buffer.observe(~F32,no))

def packed_state(mask: Buffer.Model<F32>,yes: Buffer.Model<F32>,no: Buffer.Model<F32>,scratch: Tree.Tree<Tile.Tile8>,position: U32,slot: U32) -> Cells.State<Lanes.Selected,Tile.Tile8,Unit>:
  Cells.State{Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)},Tree.pack(Tile.Tile8,scratch),position,slot,Unit{}}

def state_contents(state: Cells.State<Lanes.Selected,Tile.Tile8,Unit>) -> CellLaws.Observed<Contents,Tile.Tile8,Unit>:
  CellLaws.observe(~Lanes.Selected,~Contents,~Tile.Tile8,~Unit,~contents,state)

def round(+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,+scratch: Tree.Tree<Tile.Tile8>,+position: U32,+slot: U32) ->
  {state_contents(Cells.step(~Lanes.Selected,~Tile.Tile8,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~8,packed_state(mask,yes,no,scratch,position,slot))) ==
    CellLaws.model_stored(~Contents,~Tile.Tile8,~Unit,~scatter_model,~8,position,slot,Unit{},scratch,
      (Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},cell(Buffer.tree(~F32,mask),Buffer.tree(~F32,yes),Buffer.tree(~F32,no),position))) : CellLaws.Observed<Contents,Tile.Tile8,Unit>}:
  Equal.trans(CellLaws.Observed<Contents,Tile.Tile8,Unit>,
    state_contents(Cells.step(~Lanes.Selected,~Tile.Tile8,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~8,packed_state(mask,yes,no,scratch,position,slot))),
    CellLaws.model_stored(~Contents,~Tile.Tile8,~Unit,~scatter_model,~8,position,slot,Unit{},scratch,read_contents(Lanes.select_gather(Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)},position,Unit{}))),
    CellLaws.model_stored(~Contents,~Tile.Tile8,~Unit,~scatter_model,~8,position,slot,Unit{},scratch,(Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},cell(Buffer.tree(~F32,mask),Buffer.tree(~F32,yes),Buffer.tree(~F32,no),position))),
    CellLaws.stored_observed(~Lanes.Selected,~Contents,~Tile.Tile8,~Unit,~contents,~Lanes.select_scatter,~scatter_model,~8,~scatter_native,position,slot,Unit{},scratch,
      Lanes.select_gather(Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)},position,Unit{})),
    Equal.cong(Contents & Tile.Tile8,CellLaws.Observed<Contents,Tile.Tile8,Unit>,result => CellLaws.model_stored(~Contents,~Tile.Tile8,~Unit,~scatter_model,~8,position,slot,Unit{},scratch,result),
      read_contents(Lanes.select_gather(Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)},position,Unit{})),
      (Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},cell(Buffer.tree(~F32,mask),Buffer.tree(~F32,yes),Buffer.tree(~F32,no),position)),gather(mask,yes,no,position)))

# This model retains the actual eight-wide writes and scalar tail. Its value
# theorem below uses the indexed map readback.
def model(count: Nat,+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>) -> Buffer.Content<F32>:
  match count:
    case 1n+1n+1n+1n+1n+1n+1n+1n+rest:
      model(rest,8n+start,mask,Buffer.Content{Binary.written(Buffer.content_tree(~F32,yes),U32.from_nat(start),cell(Buffer.content_tree(~F32,mask),Buffer.content_tree(~F32,yes),Buffer.content_tree(~F32,no),U32.from_nat(start))),Buffer.content_length(~F32,yes)},no)
    case few: tail_model(few,start,mask,yes,no)

def packed_state_contents(+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,+scratch: Tree.Tree<Tile.Tile8>,+position: U32,+slot: U32) ->
  {state_contents(packed_state(mask,yes,no,scratch,position,slot)) == CellLaws.Observed{Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},scratch,position,slot,Unit{}} : CellLaws.Observed<Contents,Tile.Tile8,Unit>}:
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,scratch)),scratch,StorageLaws.reflect_packed(Tile.Tile8,scratch)) :
    {CellLaws.Observed{contents(Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)}),_,position,slot,Unit{}} == CellLaws.Observed{Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},scratch,position,slot,Unit{}} : CellLaws.Observed<Contents,Tile.Tile8,Unit>}
  Equal.cong(Contents,CellLaws.Observed<Contents,Tile.Tile8,Unit>,owners => CellLaws.Observed{owners,scratch,position,slot,Unit{}},contents(Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)}),Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},packed_contents(mask,yes,no))

type StateObservation<-state: Cells.State<Lanes.Selected,Tile.Tile8,Unit>> is Data:
  StateObservation{mask: Buffer.Model<F32>,yes: Buffer.Model<F32>,no: Buffer.Model<F32>,scratch: Tree.Tree<Tile.Tile8>,position: U32,slot: U32,
    equation: {packed_state(mask,yes,no,scratch,position,slot) == state : Cells.State<Lanes.Selected,Tile.Tile8,Unit>}}

def state_seen(-mask: Storage.Buffer<F32>,-yes: Storage.Buffer<F32>,-no: Storage.Buffer<F32>,-scratch: Array<Tile.Tile8>,+position: U32,+slot: U32,
  mask_seen: Buffer.Observation<F32,mask>,yes_seen: Buffer.Observation<F32,yes>,no_seen: Buffer.Observation<F32,no>,scratch_seen: StorageLaws.Observation<Tile.Tile8,scratch>) ->
  StateObservation<Cells.State{Lanes.Selected{mask,yes,no},scratch,position,slot,Unit{}}>:
  Buffer.Observation{+m,mask_equal} = mask_seen
  Buffer.Observation{+y,yes_equal} = yes_seen
  Buffer.Observation{+n,no_equal} = no_seen
  StorageLaws.Observation{+tree,packed,reflected} = scratch_seen
  StateObservation{m,y,n,tree,position,slot,
    %mask_equal : {packed_state(m,y,n,tree,position,slot) == Cells.State{Lanes.Selected{_,yes,no},scratch,position,slot,Unit{}} : Cells.State<Lanes.Selected,Tile.Tile8,Unit>}
    %yes_equal : {packed_state(m,y,n,tree,position,slot) == Cells.State{Lanes.Selected{Buffer.pack(~F32,m),_,no},scratch,position,slot,Unit{}} : Cells.State<Lanes.Selected,Tile.Tile8,Unit>}
    %no_equal : {packed_state(m,y,n,tree,position,slot) == Cells.State{Lanes.Selected{Buffer.pack(~F32,m),Buffer.pack(~F32,y),_},scratch,position,slot,Unit{}} : Cells.State<Lanes.Selected,Tile.Tile8,Unit>}
    %packed : {packed_state(m,y,n,tree,position,slot) == Cells.State{Lanes.Selected{Buffer.pack(~F32,m),Buffer.pack(~F32,y),Buffer.pack(~F32,n)},_,position,slot,Unit{}} : Cells.State<Lanes.Selected,Tile.Tile8,Unit>}
    {==}}

def observe_state(state: Cells.State<Lanes.Selected,Tile.Tile8,Unit>) -> StateObservation<state>:
  Cells.State{Lanes.Selected{mask,yes,no},scratch,+position,+slot,Unit{}} = state
  state_seen(mask,yes,no,scratch,position,slot,Buffer.observe(~F32,mask),Buffer.observe(~F32,yes),Buffer.observe(~F32,no),StorageLaws.observe(Tile.Tile8,scratch))

def rounds_property(count: Nat) -> Type:
  @+mask: Buffer.Model<F32> -> @+yes: Buffer.Model<F32> -> @+no: Buffer.Model<F32> -> @+scratch: Tree.Tree<Tile.Tile8> -> @+start: Nat -> @+slot: U32 ->
  {Buffer.contents(~F32,Lanes.select_result(Unary.rounds_with(~Lanes.Selected,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~Lanes.select_finish,count,packed_state(mask,yes,no,scratch,U32.from_nat(start),slot)))) == model(count,start,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}

def Continuation(count: Nat,-state: Cells.State<Lanes.Selected,Tile.Tile8,Unit>,observed: CellLaws.Observed<Contents,Tile.Tile8,Unit>) -> Type:
  CellLaws.Observed{Contents{mask,yes,no},scratch,position,slot,Unit{}} = observed
  @+start: Nat -> @position_equal: {position == U32.from_nat(start) : U32} ->
  {Buffer.contents(~F32,Lanes.select_result(Unary.rounds_with(~Lanes.Selected,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~Lanes.select_finish,count,state))) == model(count,start,mask,yes,no) : Buffer.Content<F32>}

def continue_canonical(+count: Nat,previous: rounds_property(count),+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,+scratch: Tree.Tree<Tile.Tile8>,+position: U32,+slot: U32) ->
  Continuation(count,packed_state(mask,yes,no,scratch,position,slot),CellLaws.Observed{Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},scratch,position,slot,Unit{}}):
  start => position_equal =>
    %Equal.sym(U32,position,U32.from_nat(start),position_equal) :
      {Buffer.contents(~F32,Lanes.select_result(Unary.rounds_with(~Lanes.Selected,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~Lanes.select_finish,count,packed_state(mask,yes,no,scratch,_,slot)))) == model(count,start,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}
    previous(mask,yes,no,scratch,start,slot)

def continue_seen(+count: Nat,previous: rounds_property(count),-state: Cells.State<Lanes.Selected,Tile.Tile8,Unit>,observation: StateObservation<state>) -> Continuation(count,state,state_contents(state)):
  StateObservation{+mask,+yes,+no,+scratch,+position,+slot,equation} = observation
  %equation : Continuation(count,_,state_contents(_))
  %Equal.sym(CellLaws.Observed<Contents,Tile.Tile8,Unit>,state_contents(packed_state(mask,yes,no,scratch,position,slot)),CellLaws.Observed{Contents{Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)},scratch,position,slot,Unit{}},packed_state_contents(mask,yes,no,scratch,position,slot)) :
    Continuation(count,packed_state(mask,yes,no,scratch,position,slot),_)
  continue_canonical(count,previous,mask,yes,no,scratch,position,slot)

def few_model(+count: Nat,+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  {model(count,start,mask,yes,no) == tail_model(count,start,mask,yes,no) : Buffer.Content<F32>}:
  match count:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 1n+1n+1n+1n+1n+1n+1n+1n+ +rest:
      Empty.absurd({model(8n+rest,start,mask,yes,no) == tail_model(8n+rest,start,mask,yes,no) : Buffer.Content<F32>},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),small)))

def few_rounds(+count: Nat,small: {Nat.is_lt(count,8n) == True{} : Bool}) -> rounds_property(count):
  +small = small
  mask => yes => no => scratch => start => slot =>
    %Equal.sym(Lanes.Selected,Unary.rounds_with(~Lanes.Selected,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~Lanes.select_finish,count,packed_state(mask,yes,no,scratch,U32.from_nat(start),slot)),Lanes.select_finish(count,packed_state(mask,yes,no,scratch,U32.from_nat(start),slot)),UnaryLaws.finish_small(~Lanes.Selected,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~Lanes.select_finish,count,packed_state(mask,yes,no,scratch,U32.from_nat(start),slot),small)) :
      {Buffer.contents(~F32,Lanes.select_result(_)) == model(count,start,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}
    %Equal.sym(Buffer.Content<F32>,model(count,start,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)),tail_model(count,start,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)),few_model(count,start,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no),small)) :
      {Buffer.contents(~F32,Lanes.select_result(Lanes.select_finish(count,packed_state(mask,yes,no,scratch,U32.from_nat(start),slot)))) == _ : Buffer.Content<F32>}
    tail(count,start,mask,yes,no,scratch,slot,small)

def more_canonical(+rest: Nat,previous: rounds_property(rest),+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>,+scratch: Tree.Tree<Tile.Tile8>,+start: Nat,+slot: U32) ->
  {Buffer.contents(~F32,Lanes.select_result(Unary.rounds_with(~Lanes.Selected,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~Lanes.select_finish,8n+rest,packed_state(mask,yes,no,scratch,U32.from_nat(start),slot)))) == model(8n+rest,start,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}:
  Buffer.Model{+mt,+ml,+mc} = mask
  Buffer.Model{+yt,+yl,+yc} = yes
  Buffer.Model{+nt,+nl,+nc} = no
  +position = U32.from_nat(start)
  +tile = cell(Buffer.tree(~F32,mask),yt,Buffer.tree(~F32,no),position)
  +next = {Buffer.Content{Binary.written(yt,position,tile),yl} : Buffer.Content<F32>}
  +next_scratch = Tree.write(Tile.Tile8,scratch,slot,tile)
  +turn = U32.and((slot + 1 : U32),63)
  continued = {
    %round(mask,yes,no,scratch,position,slot) : Continuation(rest,Cells.step(~Lanes.Selected,~Tile.Tile8,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~8,packed_state(mask,yes,no,scratch,position,slot)),_)
    continue_seen(rest,previous,Cells.step(~Lanes.Selected,~Tile.Tile8,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~8,packed_state(mask,yes,no,scratch,position,slot)),observe_state(Cells.step(~Lanes.Selected,~Tile.Tile8,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~8,packed_state(mask,yes,no,scratch,position,slot)))) :
    Continuation(rest,Cells.step(~Lanes.Selected,~Tile.Tile8,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~8,packed_state(mask,yes,no,scratch,position,slot)),CellLaws.Observed{Contents{Buffer.content(~F32,mask),next,Buffer.content(~F32,no)},next_scratch,(position + 8 : U32),turn,Unit{}})}
  continued(8n+start,UnaryLaws.after_round(start))

def more_rounds(+rest: Nat,previous: rounds_property(rest)) -> rounds_property(8n+rest):
  mask => yes => no => scratch => start => slot => more_canonical(rest,previous,mask,yes,no,scratch,start,slot)

def rounds(count: Nat) -> rounds_property(count):
  UnaryLaws.by_rounds(~rounds_property,~few_rounds,~more_rounds,count)

def select(+count: Nat,+mask: Buffer.Model<F32>,+yes: Buffer.Model<F32>,+no: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Lanes.select(count,Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no))) == model(count,0n,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}:
  %Equal.sym(Array<Tile.Tile8>,Unary.scratch(),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0))),StorageLaws.fresh(Tile.Tile8,6n,Tile.filled(0.0))) :
    {Buffer.contents(~F32,Lanes.select_result(Unary.rounds_with(~Lanes.Selected,~Unit,~Lanes.select_gather,~Lanes.select_scatter,~Lanes.select_finish,count,Cells.State{Lanes.Selected{Buffer.pack(~F32,mask),Buffer.pack(~F32,yes),Buffer.pack(~F32,no)},_,0,0,Unit{}}))) == model(count,0n,Buffer.content(~F32,mask),Buffer.content(~F32,yes),Buffer.content(~F32,no)) : Buffer.Content<F32>}
  rounds(count)(mask,yes,no,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0)),0n,0)
