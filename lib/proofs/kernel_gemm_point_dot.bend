# A point job agrees with logical readers when its sample and bias reads agree.
import Base
import ../kernel_gemm.bend as Gemm
import ../ordered_matrix.bend as Matrix
import ./kernel_gemm_point_jobs.bend as Jobs
import ./kernel_gemm_point_keys.bend as Keys
import ./kernel_gemm_point_readback.bend as Readback
import ./kernel_gemm_point_spec.bend as Spec
import ./kernel_gemm_point_taps.bend as Taps
import ./ordered_ranges.bend as Ranges
import ./ordered_reduction_split.bend as Ordered
import ./sequence_membership.bend as Members

def OneSample(~Reader: Data,~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,
  +target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat,+reader: Reader,+position: Nat,+channel: Nat,+key: Nat) -> Data:
  Gemm.Both<{Taps.physical_input(target,unit,row,key) == input(reader,position,key) : F32},
    {Taps.physical_weight(target,pair,lane,key) == weight(reader,channel,key) : F32}>

def SampleList(~Reader: Data,~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,
  +target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat,+reader: Reader,+position: Nat,+channel: Nat,+keys: List<&2,Nat>) -> Data:
  Members.All(Nat,key => OneSample(~Reader,~input,~weight,target,pair,unit,row,lane,reader,position,channel,key),keys)

def Samples(~Reader: Data,~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,
  +target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat,+reader: Reader,+position: Nat,+channel: Nat) -> Data:
  SampleList(~Reader,~input,~weight,target,pair,unit,row,lane,reader,position,channel,Ranges.range(U32.to_nat(Taps.inner_of(Readback.target_matrix(target))),0n))

def mapped_input(~Reader: Data,~input: Reader -> Nat -> Nat -> F32) -> Gemm.Both<Unit,Reader> -> Nat -> Keys.Key -> F32:
  Ordered.mapped_sample(~Keys.Key,~Nat,~Nat,~Reader,~Unit,~Taps.key_number,~input)

def mapped_weight(~Reader: Data,~weight: Reader -> Nat -> Nat -> F32) -> Gemm.Both<Unit,Reader> -> Nat -> Keys.Key -> F32:
  Ordered.mapped_sample(~Keys.Key,~Nat,~Nat,~Reader,~Unit,~Taps.key_number,~weight)

def KeyMatches(~Reader: Data,~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,
  +target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat,+reader: Reader,+position: Nat,+channel: Nat,key: Keys.Key) -> Data:
  Gemm.Both<{Keys.key_input()(Taps.sample_origin(target,pair,unit),row,key) == input(reader,position,Keys.key_channel(key)) : F32},
    {Keys.key_weight()(Taps.sample_origin(target,pair,unit),lane,key) == weight(reader,channel,Keys.key_channel(key)) : F32}>

def match_key(~Reader: Data,~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,
  +target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat,+reader: Reader,+position: Nat,+channel: Nat,+key: Keys.Key,
  +row_inside: {Nat.is_le(row,3n) == True{} : Bool},+lane_inside: Taps.LaneInside(key),
  +samples: Gemm.Both<{Taps.physical_input(target,unit,row,Keys.key_channel(key)) == input(reader,position,Keys.key_channel(key)) : F32},
    {Taps.physical_weight(target,pair,lane,Keys.key_channel(key)) == weight(reader,channel,Keys.key_channel(key)) : F32}>) ->
  KeyMatches(~Reader,~input,~weight,target,pair,unit,row,lane,reader,position,channel,key):
  Gemm.Both{+input_agrees,+weight_agrees} = samples
  +number = Keys.key_channel(key)
  +origin = Taps.sample_origin(target,pair,unit)
  Gemm.Both{Equal.trans(F32,Keys.key_input()(origin,row,key),Taps.physical_input(target,unit,row,number),input(reader,position,number),
    Taps.input_key(target,pair,unit,row,key,row_inside,lane_inside),input_agrees),
    Equal.trans(F32,Keys.key_weight()(origin,lane,key),Taps.physical_weight(target,pair,lane,number),weight(reader,channel,number),
      Taps.weight_key(target,pair,unit,lane,key),weight_agrees)}

def logical_dot(~Reader: Data,~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,
  +reader: Reader,+position: Nat,+channel: Nat,+width: Nat) -> F32:
  Matrix.dot(~Nat,~Nat,~Nat,~Reader,~input,~weight,Ranges.range(width,0n),channel,position,reader,0.0)

def matching(~Reader: Data,~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,
  +target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat,+reader: Reader,+position: Nat,+channel: Nat,
  keys: List<&2,Keys.Key>,+safe: Members.All(Keys.Key,Taps.LaneInside,keys),
  +row_inside: {Nat.is_le(row,3n) == True{} : Bool},
  +samples: SampleList(~Reader,~input,~weight,target,pair,unit,row,lane,reader,position,channel,Ordered.map_keys(~Keys.Key,~Nat,~Unit,~Taps.key_number,keys,Unit{}))) ->
  Ordered.MatchingSamples(~Keys.Key,~Nat,~Nat,~Gemm.Both<Keys.Origin,Spec.Sampled>,~Nat,~Nat,~Gemm.Both<Unit,Reader>,~Keys.key_input(),~Keys.key_weight(),
    ~mapped_input(~Reader,~input),~mapped_weight(~Reader,~weight),keys,lane,row,Taps.sample_origin(target,pair,unit),channel,position,Gemm.Both{Unit{},reader}):
  match keys:
    case []: Unit{}
    case +head <> +rest:
      Gemm.Both{+lane_inside,+rest_safe} = safe
      Gemm.Both{+sample,+rest_samples} = samples
      Gemm.Both{match_key(~Reader,~input,~weight,target,pair,unit,row,lane,reader,position,channel,head,row_inside,lane_inside,sample),
        matching(~Reader,~input,~weight,target,pair,unit,row,lane,reader,position,channel,rest,rest_safe,row_inside,rest_samples)}

def logical_keys(~Reader: Data,~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,
  +target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat,+reader: Reader,+position: Nat,+channel: Nat,
  +keys: List<&2,Keys.Key>,+safe: Members.All(Keys.Key,Taps.LaneInside,keys),
  +sequence: {Ordered.map_keys(~Keys.Key,~Nat,~Unit,~Taps.key_number,keys,Unit{}) == Ranges.range(U32.to_nat(Taps.inner_of(Readback.target_matrix(target))),0n) : List<&2,Nat>},
  +row_inside: {Nat.is_le(row,3n) == True{} : Bool},+samples: Samples(~Reader,~input,~weight,target,pair,unit,row,lane,reader,position,channel)) ->
  {Taps.keyed_dot(keys,target,pair,unit,row,lane) == logical_dot(~Reader,~input,~weight,reader,position,channel,U32.to_nat(Taps.inner_of(Readback.target_matrix(target)))) : F32}:
  +width = U32.to_nat(Taps.inner_of(Readback.target_matrix(target)))
  +keyed = Taps.keyed_dot(keys,target,pair,unit,row,lane)
  +mapped = Matrix.dot(~Nat,~Keys.Key,~Nat,~Gemm.Both<Unit,Reader>,~mapped_input(~Reader,~input),~mapped_weight(~Reader,~weight),
    keys,channel,position,Gemm.Both{Unit{},reader},0.0)
  +wanted = logical_dot(~Reader,~input,~weight,reader,position,channel,width)
  +agrees = matching(~Reader,~input,~weight,target,pair,unit,row,lane,reader,position,channel,keys,safe,row_inside,
    %Equal.sym(List<&2,Nat>,Ordered.map_keys(~Keys.Key,~Nat,~Unit,~Taps.key_number,keys,Unit{}),Ranges.range(width,0n),sequence) :
      SampleList(~Reader,~input,~weight,target,pair,unit,row,lane,reader,position,channel,_)
    samples)
  Equal.trans(F32,keyed,mapped,wanted,
    Ordered.dot_congruence(~Keys.Key,~Nat,~Nat,~Gemm.Both<Keys.Origin,Spec.Sampled>,~Nat,~Nat,~Gemm.Both<Unit,Reader>,~Keys.key_input(),~Keys.key_weight(),
      ~mapped_input(~Reader,~input),~mapped_weight(~Reader,~weight),keys,lane,row,Taps.sample_origin(target,pair,unit),channel,position,Gemm.Both{Unit{},reader},0.0,agrees),
    Equal.sym(F32,wanted,mapped,
      %sequence : {Matrix.dot(~Nat,~Nat,~Nat,~Reader,~input,~weight,_,channel,position,reader,0.0) == mapped : F32}
      Ordered.dot_reindexed(~Keys.Key,~Nat,~Nat,~Nat,~Reader,~Unit,~Taps.key_number,~input,~weight,keys,Unit{},channel,position,reader,0.0)))

def logical_lane(~accumulate: Jobs.Accumulate(),~Reader: Data,~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,
  +target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat,+reader: Reader,+position: Nat,+channel: Nat,+bias: F32,
  +row_inside: {Nat.is_le(row,3n) == True{} : Bool},+samples: Samples(~Reader,~input,~weight,target,pair,unit,row,lane,reader,position,channel),
  +bias_agrees: {Taps.bias_lane(target,pair,lane) == bias : F32},+ordered: Taps.OrderedLane<accumulate,target,pair,unit,row,lane>) ->
  {Spec.pair_lane(lane,Readback.stored_value_by(~accumulate,target,U32.from_nat(pair),U32.from_nat(unit),row)) ==
    (logical_dot(~Reader,~input,~weight,reader,position,channel,U32.to_nat(Taps.inner_of(Readback.target_matrix(target)))) + bias : F32) : F32}:
  Taps.OrderedLane{+keys,+safe,+sequence,+exact} = ordered
  +result = Spec.pair_lane(lane,Readback.stored_value_by(~accumulate,target,U32.from_nat(pair),U32.from_nat(unit),row))
  +physical = Taps.keyed_dot(keys,target,pair,unit,row,lane)
  +logical = logical_dot(~Reader,~input,~weight,reader,position,channel,U32.to_nat(Taps.inner_of(Readback.target_matrix(target))))
  +offset = Taps.bias_lane(target,pair,lane)
  Equal.trans(F32,result,(physical + offset : F32),(logical + bias : F32),exact,
    %bias_agrees : {(physical + offset : F32) == (logical + _ : F32) : F32}
    Equal.cong(F32,F32,value => (value + offset : F32),physical,logical,
      logical_keys(~Reader,~input,~weight,target,pair,unit,row,lane,reader,position,channel,keys,safe,sequence,row_inside,samples)))
