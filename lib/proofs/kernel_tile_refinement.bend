# A shared native/model reader contract; pair composition owns the generic step.
import Base
import ./paired_read_refinement.bend as ReadLaws
import ../kernel_block.bend as Block
import ../math_sequence.bend as Sequence
import ../kernel_tile.bend as Tile
import ./word_encoding.bend as WordEncoding
import ./nat_division.bend as Division

type SampleContext<-Model: Data,-Context: Data> is Data:
  SampleContext{model: Model,context: Context}

def sample_context(~Model: Data,~Context: Data,~sample: Model -> Context -> Nat -> F32,pair: SampleContext<Model,Context>,index: Nat) -> F32:
  SampleContext{model,context} = pair
  sample(model,context,index)

def head_sample(~Model: Data,~Context: Data,~sample: Model -> Context -> Nat -> F32,model: Model,combined: Tile.HeadContext<Context>,
  index: Nat) -> F32:
  Tile.HeadContext{head,context} = combined
  match index:
    case 0n: head
    case 1n+rest: sample(model,context,1n+rest)

def head_read_refines(
  ~Native: Type,~Model: Data,~Context: Data,
  ~pack: Model -> Native,~read: Native -> Context -> Nat -> Native & F32,
  ~sample: Model -> Context -> Nat -> F32,
  ~read_rule: @+model: Model -> @+context: Context -> @+index: Nat -> {read(pack(model),context,index) == (pack(model),sample(model,context,
        index)) : Native & F32},
  +model: Model,+combined: Tile.HeadContext<Context>,+index: Nat
) -> {Tile.read_with_head(~Native,~Context,~read,pack(model),combined,index) == (pack(model),head_sample(~Model,~Context,~sample,model,combined,
    index)) : Native & F32}:
  Tile.HeadContext{head,context} = combined
  match index:
    case 0n: {==}
    case 1n+rest: read_rule(model,context,1n+rest)

def read_pair_with_head_refines(
  ~Native: Type,~Model: Data,~Context: Data,
  ~pack: Model -> Native,~read: Native -> Context -> Nat -> Native & F32,
  ~sample: Model -> Context -> Nat -> F32,
  ~read_rule: @+model: Model -> @+context: Context -> @+index: Nat -> {read(pack(model),context,index) == (pack(model),sample(model,context,
        index)) : Native & F32},
  +model: Model,+context: Tile.HeadContext<Context>,+index: Nat
) -> {Block.read_pair(~Native,~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~Native,~Context,~read),~1n,pack(model),context,
  index) == (pack(model),ReadLaws.pair_sample(~Model,~Tile.HeadContext<Context>,~F32,~head_sample(~Model,~Context,~sample),~1n,model,context,
    index)) : Native & Block.Twin<F32>}:
  ReadLaws.read_pair_refines(~Native,~Model,~Tile.HeadContext<Context>,~F32,~pack,~Tile.read_with_head(~Native,~Context,~read),
    ~head_sample(~Model,~Context,~sample),~1n,model,context,index,
    head_read_refines(~Native,~Model,~Context,~pack,~read,~sample,~read_rule,model,context,index),
    head_read_refines(~Native,~Model,~Context,~pack,~read,~sample,~read_rule,model,context,Nat.add(index,1n)))

def read_pair_of_pairs_refines(
  ~Native: Type,~Model: Data,~Context: Data,
  ~pack: Model -> Native,~read: Native -> Context -> Nat -> Native & F32,
  ~sample: Model -> Context -> Nat -> F32,
  ~read_rule: @+model: Model -> @+context: Context -> @+index: Nat -> {read(pack(model),context,index) == (pack(model),sample(model,context,
        index)) : Native & F32},
  +model: Model,+context: Tile.HeadContext<Context>,+index: Nat
) -> {Block.read_pair(~Native,~Tile.HeadContext<Context>,~Block.Twin<F32>,~Block.read_pair(~Native,
    ~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~Native,~Context,~read),~1n),~2n,pack(model),context,index) == (pack(model),
  ReadLaws.pair_sample(~Model,~Tile.HeadContext<Context>,~Block.Twin<F32>,~ReadLaws.pair_sample(~Model,~Tile.HeadContext<Context>,
      ~F32,~head_sample(~Model,~Context,~sample),~1n),~2n,model,context,index)) : Native & Block.Twin<Block.Twin<F32>>}:
  ReadLaws.read_pair_refines(~Native,~Model,~Tile.HeadContext<Context>,~Block.Twin<F32>,~pack,~Block.read_pair(~Native,
      ~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~Native,~Context,~read),~1n),~ReadLaws.pair_sample(~Model,~Tile.HeadContext<Context>,
      ~F32,~head_sample(~Model,~Context,~sample),~1n),~2n,model,context,index,
    read_pair_with_head_refines(~Native,~Model,~Context,~pack,~read,~sample,~read_rule,model,context,index),
    read_pair_with_head_refines(~Native,~Model,~Context,~pack,~read,~sample,~read_rule,model,context,Nat.add(index,2n)))

def read_tile_pairs_refines(
  ~Native: Type,~Model: Data,~Context: Data,
  ~pack: Model -> Native,~read: Native -> Context -> Nat -> Native & F32,
  ~sample: Model -> Context -> Nat -> F32,
  ~read_rule: @+model: Model -> @+context: Context -> @+index: Nat -> {read(pack(model),context,index) == (pack(model),sample(model,context,
        index)) : Native & F32},
  +model: Model,+context: Tile.HeadContext<Context>,+index: Nat
) -> {Block.read_pair(~Native,~Tile.HeadContext<Context>,~Block.Twin<Block.Twin<F32>>,
  ~Block.read_pair(~Native,~Tile.HeadContext<Context>,~Block.Twin<F32>,~Block.read_pair(~Native,
      ~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~Native,~Context,~read),~1n),~2n),~4n,pack(model),context,index) == (pack(model),
  ReadLaws.pair_sample(~Model,~Tile.HeadContext<Context>,~Block.Twin<Block.Twin<F32>>,~ReadLaws.pair_sample(~Model,
      ~Tile.HeadContext<Context>,~Block.Twin<F32>,~ReadLaws.pair_sample(~Model,~Tile.HeadContext<Context>,~F32,~head_sample(~Model,
        ~Context,~sample),~1n),~2n),~4n,model,context,index)) : Native & Block.Twin<Block.Twin<Block.Twin<F32>>>}:
  ReadLaws.read_pair_refines(~Native,~Model,~Tile.HeadContext<Context>,~Block.Twin<Block.Twin<F32>>,~pack,
    ~Block.read_pair(~Native,~Tile.HeadContext<Context>,~Block.Twin<F32>,~Block.read_pair(~Native,
        ~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~Native,~Context,~read),~1n),~2n),~ReadLaws.pair_sample(~Model,
      ~Tile.HeadContext<Context>,~Block.Twin<F32>,~ReadLaws.pair_sample(~Model,~Tile.HeadContext<Context>,~F32,~head_sample(~Model,
          ~Context,~sample),~1n),~2n),~4n,model,context,index,
    read_pair_of_pairs_refines(~Native,~Model,~Context,~pack,~read,~sample,~read_rule,model,context,index),
    read_pair_of_pairs_refines(~Native,~Model,~Context,~pack,~read,~sample,~read_rule,model,context,Nat.add(index,4n)))

def tail_block_refines(
  ~Native: Type,~Model: Data,~Context: Data,
  ~pack: Model -> Native,
  ~read: Native -> Context -> Nat -> Native & F32,
  ~sample: Model -> Context -> Nat -> F32,
  ~read_rule: @+model: Model -> @+context: Context -> @+index: Nat -> {read(pack(model),context,index) == (pack(model),sample(model,context,
        index)) : Native & F32},
  +model: Model,+context: Context,+head: F32
) -> {Tile.read_tail(~Native,~Context,~read,(pack(model),head),context) == (pack(model),Tile.from_sequence(Sequence.Elements{head,
      Sequence.tabulate(~F32,~SampleContext<Model,Context>,~sample_context(~Model,~Context,~sample),7n,1n,SampleContext{model,
        context})})) : Native & Tile.Tile8}:
  %read_tile_pairs_refines(~Native,~Model,~Context,~pack,~read,~sample,~read_rule,model,Tile.HeadContext{head,context},
    0n) : {Tile.finish_pairs(~Native,Block.read_pair(~Native,~Tile.HeadContext<Context>,~Block.Twin<Block.Twin<F32>>,
        ~Block.read_pair(~Native,~Tile.HeadContext<Context>,~Block.Twin<F32>,~Block.read_pair(~Native,
          ~Tile.HeadContext<Context>,~F32,~Tile.read_with_head(~Native,~Context,~read),~1n),~2n),~4n,pack(model),Tile.HeadContext{head,context},
        0n)) == Tile.finish_pairs(~Native,_) : Native & Tile.Tile8}
  {==}

# Tile projections and extensional equality used by GEMM callers.
def lane_of(lane: Nat) -> Sequence.Index(8n):
  match lane:
    case 0n: Sequence.First{}
    case 1n+a:
      match a:
        case 0n: Sequence.Next{Sequence.First{}}
        case 1n+b:
          match b:
            case 0n: Sequence.Next{Sequence.Next{Sequence.First{}}}
            case 1n+c:
              match c:
                case 0n: Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}
                case 1n+d:
                  match d:
                    case 0n: Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}
                    case 1n+e:
                      match e:
                        case 0n: Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}
                        case 1n+f:
                          match f:
                            case 0n: Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}
                            case 1n+g: Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}}

# A lane below eight is its own index.
def lane_value_of(+lane: Nat,small: {Nat.is_lt(lane,8n) == True{} : Bool}) -> {Sequence.index_value(8n,lane_of(lane)) == lane : Nat}:
  match lane:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  match d:
                    case 0n: {==}
                    case 1n+e:
                      match e:
                        case 0n: {==}
                        case 1n+f:
                          match f:
                            case 0n: {==}
                            case 1n+g:
                              match g:
                                case 0n: {==}
                                case 1n+h:
                                  Empty.absurd({Sequence.index_value(8n,lane_of(8n+h)) == 8n+h : Nat},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(h,0n),True{},Equal.sym(Bool,Nat.is_lt(h,0n),False{},Division.lt_zero(h)),small)))

def select_lane(lane: Nat,values: Tile.Tile8) -> F32:
  match lane:
    case 0n: Tile.lane0(values)
    case 1n+a:
      match a:
        case 0n: Tile.lane1(values)
        case 1n+b:
          match b:
            case 0n: Tile.lane2(values)
            case 1n+c:
              match c:
                case 0n: Tile.lane3(values)
                case 1n+d:
                  match d:
                    case 0n: Tile.lane4(values)
                    case 1n+e:
                      match e:
                        case 0n: Tile.lane5(values)
                        case 1n+f:
                          match f:
                            case 0n: Tile.lane6(values)
                            case 1n+g: Tile.lane7(values)

# Every lane of a filled tile holds its value.
def filled_lane(+lane: Nat,+value: F32) -> {select_lane(lane,Tile.filled(value)) == value : F32}:
  match lane:
    case 0n: {==}
    case 1n+ +a:
      match a:
        case 0n: {==}
        case 1n+ +b:
          match b:
            case 0n: {==}
            case 1n+ +c:
              match c:
                case 0n: {==}
                case 1n+ +d:
                  match d:
                    case 0n: {==}
                    case 1n+ +e:
                      match e:
                        case 0n: {==}
                        case 1n+ +f:
                          match f:
                            case 0n: {==}
                            case 1n+g: {==}

def selected_lane(+lane: Nat,+values: Tile.Tile8) ->
  {select_lane(lane,values) == Tile.element_at(lane_of(lane),values) : F32}:
  match lane:
    case 0n:
      Tile.Tile{a,b,c,d,e,f,g,h} = values
      {==}
    case 1n+ +a:
      match a:
        case 0n:
          Tile.Tile{a,b,c,d,e,f,g,h} = values
          {==}
        case 1n+ +b:
          match b:
            case 0n:
              Tile.Tile{a,b,c,d,e,f,g,h} = values
              {==}
            case 1n+ +c:
              match c:
                case 0n:
                  Tile.Tile{a,b,c,d,e,f,g,h} = values
                  {==}
                case 1n+ +d:
                  match d:
                    case 0n:
                      Tile.Tile{a,b,c,d,e,f,g,h} = values
                      {==}
                    case 1n+ +e:
                      match e:
                        case 0n:
                          Tile.Tile{a,b,c,d,e,f,g,h} = values
                          {==}
                        case 1n+ +f:
                          match f:
                            case 0n:
                              Tile.Tile{a,b,c,d,e,f,g,h} = values
                              {==}
                            case 1n+ +g:
                              Tile.Tile{a,b,c,d,e,f,g,h} = values
                              {==}

def tile_eta(+values: Tile.Tile8) -> {values == Tile.Tile{Tile.lane0(values),Tile.lane1(values),Tile.lane2(values),Tile.lane3(values),Tile.lane4(values),Tile.lane5(values),Tile.lane6(values),Tile.lane7(values)} : Tile.Tile8}:
  Tile.Tile{a,b,c,d,e,f,g,h} = values
  {==}

def tiles_equal(+left: Tile.Tile8,+right: Tile.Tile8,
  e0: {Tile.lane0(left) == Tile.lane0(right) : F32},e1: {Tile.lane1(left) == Tile.lane1(right) : F32},e2: {Tile.lane2(left) == Tile.lane2(right) : F32},e3: {Tile.lane3(left) == Tile.lane3(right) : F32},e4: {Tile.lane4(left) == Tile.lane4(right) : F32},e5: {Tile.lane5(left) == Tile.lane5(right) : F32},e6: {Tile.lane6(left) == Tile.lane6(right) : F32},e7: {Tile.lane7(left) == Tile.lane7(right) : F32}) -> {left == right : Tile.Tile8}:
  %Equal.sym(Tile.Tile8,left,Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),Tile.lane6(left),Tile.lane7(left)},tile_eta(left)) : {_ == right : Tile.Tile8}
  %Equal.sym(Tile.Tile8,right,Tile.Tile{Tile.lane0(right),Tile.lane1(right),Tile.lane2(right),Tile.lane3(right),Tile.lane4(right),Tile.lane5(right),Tile.lane6(right),Tile.lane7(right)},tile_eta(right)) :
    {Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),Tile.lane6(left),Tile.lane7(left)} == _ : Tile.Tile8}
  %e0 : {Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),Tile.lane6(left),Tile.lane7(left)} == Tile.Tile{_,Tile.lane1(right),Tile.lane2(right),Tile.lane3(right),Tile.lane4(right),Tile.lane5(right),Tile.lane6(right),Tile.lane7(right)} : Tile.Tile8}
  %e1 : {Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),Tile.lane6(left),Tile.lane7(left)} == Tile.Tile{Tile.lane0(left),_,Tile.lane2(right),Tile.lane3(right),Tile.lane4(right),Tile.lane5(right),Tile.lane6(right),Tile.lane7(right)} : Tile.Tile8}
  %e2 : {Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),Tile.lane6(left),Tile.lane7(left)} == Tile.Tile{Tile.lane0(left),Tile.lane1(left),_,Tile.lane3(right),Tile.lane4(right),Tile.lane5(right),Tile.lane6(right),Tile.lane7(right)} : Tile.Tile8}
  %e3 : {Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),Tile.lane6(left),Tile.lane7(left)} == Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),_,Tile.lane4(right),Tile.lane5(right),Tile.lane6(right),Tile.lane7(right)} : Tile.Tile8}
  %e4 : {Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),Tile.lane6(left),Tile.lane7(left)} == Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),_,Tile.lane5(right),Tile.lane6(right),Tile.lane7(right)} : Tile.Tile8}
  %e5 : {Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),Tile.lane6(left),Tile.lane7(left)} == Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),_,Tile.lane6(right),Tile.lane7(right)} : Tile.Tile8}
  %e6 : {Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),Tile.lane6(left),Tile.lane7(left)} == Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),_,Tile.lane7(right)} : Tile.Tile8}
  %e7 : {Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),Tile.lane6(left),Tile.lane7(left)} == Tile.Tile{Tile.lane0(left),Tile.lane1(left),Tile.lane2(left),Tile.lane3(left),Tile.lane4(left),Tile.lane5(left),Tile.lane6(left),_} : Tile.Tile8}
  {==}

# A lane of the update of tiles is the update of their lanes.
def updated_lane(+accumulator: Tile.Tile8,+input: Tile.Tile8,+weight: Tile.Tile8,+lane: Nat) ->
  {select_lane(lane,Tile.update_lanes(accumulator,input,weight)) == (select_lane(lane,accumulator) + (select_lane(lane,input) * select_lane(lane,weight) : F32) : F32) : F32}:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = accumulator
  Tile.Tile{x0,x1,x2,x3,x4,x5,x6,x7} = input
  Tile.Tile{w0,w1,w2,w3,w4,w5,w6,w7} = weight
  match lane:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  match d:
                    case 0n: {==}
                    case 1n+e:
                      match e:
                        case 0n: {==}
                        case 1n+f:
                          match f:
                            case 0n: {==}
                            case 1n+g: {==}

# A lane of an update by one value is the lane plus the product.
def updated_by_value(~times: F32 -> F32 -> F32,+accumulator: Tile.Tile8,+lane_tile: Tile.Tile8,+row_value: F32,+lane: Nat) ->
  {select_lane(lane,Tile.update(~times,accumulator,lane_tile,row_value)) ==
    (select_lane(lane,accumulator) + times(select_lane(lane,lane_tile),row_value) : F32) : F32}:
  Tile.Tile{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} = accumulator
  Tile.Tile{+x0,+x1,+x2,+x3,+x4,+x5,+x6,+x7} = lane_tile
  match lane:
    case 0n: {==}
    case 1n+ +a:
      match a:
        case 0n: {==}
        case 1n+ +b:
          match b:
            case 0n: {==}
            case 1n+ +c:
              match c:
                case 0n: {==}
                case 1n+ +d:
                  match d:
                    case 0n: {==}
                    case 1n+ +e:
                      match e:
                        case 0n: {==}
                        case 1n+ +f:
                          match f:
                            case 0n: {==}
                            case 1n+ +g: {==}

# A lane of the sum of tiles is the sum of their lanes.
def added_lane(+left: Tile.Tile8,+right: Tile.Tile8,+lane: Nat) ->
  {select_lane(lane,Tile.plus(left,right)) == (select_lane(lane,left) + select_lane(lane,right) : F32) : F32}:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = left
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = right
  match lane:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  match d:
                    case 0n: {==}
                    case 1n+e:
                      match e:
                        case 0n: {==}
                        case 1n+f:
                          match f:
                            case 0n: {==}
                            case 1n+g: {==}

def times_lane(+left: Tile.Tile8,+right: Tile.Tile8,+lane: Nat) ->
  {select_lane(lane,Tile.times(left,right)) == (select_lane(lane,left) * select_lane(lane,right) : F32) : F32}:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = left
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = right
  match lane:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  match d:
                    case 0n: {==}
                    case 1n+e:
                      match e:
                        case 0n: {==}
                        case 1n+f:
                          match f:
                            case 0n: {==}
                            case 1n+g: {==}

def square_lane(+left: Tile.Tile8,+right: Tile.Tile8,+lane: Nat) ->
  {select_lane(lane,Tile.times(Tile.minus(left,right),Tile.minus(left,right))) ==
    ((select_lane(lane,left) - select_lane(lane,right) : F32) * (select_lane(lane,left) - select_lane(lane,right) : F32) : F32) : F32}:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = left
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = right
  match lane:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  match d:
                    case 0n: {==}
                    case 1n+e:
                      match e:
                        case 0n: {==}
                        case 1n+f:
                          match f:
                            case 0n: {==}
                            case 1n+g: {==}

# A lane of a sum of tiles is the sum of their lanes.
def plus_lane(+left: Tile.Tile8,+right: Tile.Tile8,+lane: Nat) ->
  {select_lane(lane,Tile.plus(left,right)) == (select_lane(lane,left) + select_lane(lane,right) : F32) : F32}:
  Tile.Tile{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} = left
  Tile.Tile{+b0,+b1,+b2,+b3,+b4,+b5,+b6,+b7} = right
  match lane:
    case 0n: {==}
    case 1n+ +a:
      match a:
        case 0n: {==}
        case 1n+ +b:
          match b:
            case 0n: {==}
            case 1n+ +c:
              match c:
                case 0n: {==}
                case 1n+ +d:
                  match d:
                    case 0n: {==}
                    case 1n+ +e:
                      match e:
                        case 0n: {==}
                        case 1n+ +f:
                          match f:
                            case 0n: {==}
                            case 1n+g: {==}
