# F.random_uniform and F.random_below hold, at every row-major index, the
# generator of that index and nothing else: the value depends only on the seed,
# the stream and the index. F.arange holds its index plus the start.
import Base
import ../math_random.bend as Random
import ../tensor.bend as Tensors
import ../tensor_f32.bend as F
import ../tensor_shape.bend as Shapes
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_map.bend as TiledMap
import ./tensor_full_values.bend as Full
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_tiled_indexed_values.bend as IndexedValues
import ./tensor_tiled_map.bend as TiledMaps
import ./tensor_tiled_public.bend as Public

# The index of every element of a tensor of counted shape, carried in its bits.
def counter_value() -> F32 -> U32 -> F32:
  IndexedValues.indexed_value(~Random.counter,~F32,~Full.constant(~F32))

def counters(+shape: List<&2,U32>,+count: U32,+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},+nonempty: {Shapes.has_zero(shape) == False{} : Bool}) ->
  Public.Contract(~F32,~counter_value(),0.0,shape,F.apply_indexed(~Random.counter,1.0,F.zeros(shape))):
  IndexedValues.indexed(~Random.counter,~F32,~Full.constant(~F32),1.0,0.0,shape,Tiled.zeros(shape),Full.zeros(shape,count,counts,nonempty))

# A generator is read through the counter its element carries.
def uniform_value() -> PiecesMap.Context<Random.Key,F32> -> U32 -> F32:
  PiecesMap.value(~F32,~Random.Key,~F32,~Random.uniform_of_counter,~counter_value())

def uniform_context(+seed: U32,+stream: U32) -> PiecesMap.Context<Random.Key,F32>:
  PiecesMap.Context{Random.Seeded{seed,stream},0.0}

def random_uniform(+shape: List<&2,U32>,+seed: U32,+stream: U32,+count: U32,+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool}) ->
  Public.Contract(~PiecesMap.Context<Random.Key,F32>,~uniform_value(),uniform_context(seed,stream),shape,F.random_uniform(shape,seed,stream)):
  TiledMaps.map(~Random.Key,~F32,~Random.uniform_of_counter,~TiledMap.lanes(~Random.Key,~Random.uniform_of_counter),~TiledMaps.lanes_law(~Random.Key,~Random.uniform_of_counter),
    ~counter_value(),Random.Seeded{seed,stream},14.375,0.0,shape,F.apply_indexed(~Random.counter,1.0,F.zeros(shape)),counters(shape,count,counts,nonempty))

def below_value() -> PiecesMap.Context<Random.Bounded,F32> -> U32 -> F32:
  PiecesMap.value(~F32,~Random.Bounded,~F32,~Random.below_of_counter,~counter_value())

def below_context(+bound: U32,+seed: U32,+stream: U32) -> PiecesMap.Context<Random.Bounded,F32>:
  PiecesMap.Context{Random.Bounded{bound,Random.Seeded{seed,stream}},0.0}

def random_below(+shape: List<&2,U32>,+bound: U32,+seed: U32,+stream: U32,+count: U32,+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool}) ->
  Public.Contract(~PiecesMap.Context<Random.Bounded,F32>,~below_value(),below_context(bound,seed,stream),shape,F.random_below(shape,bound,seed,stream)):
  TiledMaps.map(~Random.Bounded,~F32,~Random.below_of_counter,~TiledMap.lanes(~Random.Bounded,~Random.below_of_counter),~TiledMaps.lanes_law(~Random.Bounded,~Random.below_of_counter),
    ~counter_value(),Random.Bounded{bound,Random.Seeded{seed,stream}},15.875,0.0,shape,F.apply_indexed(~Random.counter,1.0,F.zeros(shape)),counters(shape,count,counts,nonempty))

def index_value() -> F32 -> U32 -> F32:
  IndexedValues.indexed_value(~(index => value => U32.to_f32(index)),~F32,~Full.constant(~F32))

def arange_value() -> PiecesMap.Context<F32,F32> -> U32 -> F32:
  PiecesMap.value(~F32,~F32,~F32,~F.scalar_value(~F32.add),~index_value())

def arange(+start: F32,+count: U32,+counted: U32,+counts: {Tensors.count_of([count]) == Some{counted} : Maybe<&2,U32>},
  +nonempty: {Shapes.has_zero([count]) == False{} : Bool}) ->
  Public.Contract(~PiecesMap.Context<F32,F32>,~arange_value(),PiecesMap.Context{start,0.0},[count],F.arange(start,count)):
  TiledMaps.map(~F32,~F32,~F.scalar_value(~F32.add),~TiledMap.lanes(~F32,~F.scalar_value(~F32.add)),~TiledMaps.lanes_law(~F32,~F.scalar_value(~F32.add)),
    ~index_value(),start,1.0,0.0,[count],F.apply_indexed(~(index => value => U32.to_f32(index)),1.0,F.zeros([count])),
    IndexedValues.indexed(~(index => value => U32.to_f32(index)),~F32,~Full.constant(~F32),1.0,0.0,[count],Tiled.zeros([count]),Full.zeros([count],counted,counts,nonempty)))
