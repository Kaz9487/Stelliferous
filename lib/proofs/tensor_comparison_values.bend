# The six public comparisons of two tensors are binary operations, and the
# comparisons with a scalar and clamp maps of the cells.
import Base
import ../tensor_f32.bend as F32Tensor
import ../tensor_tiled_map.bend as TiledMap
import ./tensor_binary_values.bend as Binary
import ./tensor_function_values.bend as FunctionValues
import ./tensor_tiled_map.bend as TiledMaps
import ../math_compare.bend as Compare

def less(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  Binary.PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.less_elements,~F32Tensor.less):
  Binary.public(~Left,~Right,~left_value,~right_value,~Compare.less_elements,~"less",~2.625)

def less_equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  Binary.PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.less_equal_elements,~F32Tensor.less_equal):
  Binary.public(~Left,~Right,~left_value,~right_value,~Compare.less_equal_elements,~"less_equal",~2.625)

def greater(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  Binary.PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.greater_elements,~F32Tensor.greater):
  Binary.public(~Left,~Right,~left_value,~right_value,~Compare.greater_elements,~"greater",~2.625)

def greater_equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  Binary.PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.greater_equal_elements,~F32Tensor.greater_equal):
  Binary.public(~Left,~Right,~left_value,~right_value,~Compare.greater_equal_elements,~"greater_equal",~2.625)

def equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  Binary.PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.equal_elements,~F32Tensor.equal):
  Binary.public(~Left,~Right,~left_value,~right_value,~Compare.equal_elements,~"equal",~1.375)

def not_equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  Binary.PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.not_equal_elements,~F32Tensor.not_equal):
  Binary.public(~Left,~Right,~left_value,~right_value,~Compare.not_equal_elements,~"not_equal",~1.375)

# A comparison with a scalar holds 1.0 or 0.0 by the comparison of the value
# with the scalar at every index, and clamp the value clamped to the bounds.
def less_scalar(~Logical: Data,~original: Logical -> U32 -> F32,+value: F32) ->
  FunctionValues.ContextPublicLaw(~F32,~value,~Logical,~original,~Compare.less,~(tensor => F32Tensor.less_scalar(tensor,value))):
  FunctionValues.public_context(~F32,~Logical,~original,~Compare.less,~TiledMap.lanes(~F32,~Compare.less),~TiledMaps.lanes_law(~F32,~Compare.less),value,2.75)

def less_equal_scalar(~Logical: Data,~original: Logical -> U32 -> F32,+value: F32) ->
  FunctionValues.ContextPublicLaw(~F32,~value,~Logical,~original,~Compare.less_equal,~(tensor => F32Tensor.less_equal_scalar(tensor,value))):
  FunctionValues.public_context(~F32,~Logical,~original,~Compare.less_equal,~TiledMap.lanes(~F32,~Compare.less_equal),~TiledMaps.lanes_law(~F32,~Compare.less_equal),value,2.625)

def greater_scalar(~Logical: Data,~original: Logical -> U32 -> F32,+value: F32) ->
  FunctionValues.ContextPublicLaw(~F32,~value,~Logical,~original,~Compare.greater,~(tensor => F32Tensor.greater_scalar(tensor,value))):
  FunctionValues.public_context(~F32,~Logical,~original,~Compare.greater,~TiledMap.lanes(~F32,~Compare.greater),~TiledMaps.lanes_law(~F32,~Compare.greater),value,2.625)

def greater_equal_scalar(~Logical: Data,~original: Logical -> U32 -> F32,+value: F32) ->
  FunctionValues.ContextPublicLaw(~F32,~value,~Logical,~original,~Compare.greater_equal,~(tensor => F32Tensor.greater_equal_scalar(tensor,value))):
  FunctionValues.public_context(~F32,~Logical,~original,~Compare.greater_equal,~TiledMap.lanes(~F32,~Compare.greater_equal),~TiledMaps.lanes_law(~F32,~Compare.greater_equal),value,2.75)

def equal_scalar(~Logical: Data,~original: Logical -> U32 -> F32,+value: F32) ->
  FunctionValues.ContextPublicLaw(~F32,~value,~Logical,~original,~Compare.equal,~(tensor => F32Tensor.equal_scalar(tensor,value))):
  FunctionValues.public_context(~F32,~Logical,~original,~Compare.equal,~TiledMap.lanes(~F32,~Compare.equal),~TiledMaps.lanes_law(~F32,~Compare.equal),value,1.5)

def not_equal_scalar(~Logical: Data,~original: Logical -> U32 -> F32,+value: F32) ->
  FunctionValues.ContextPublicLaw(~F32,~value,~Logical,~original,~Compare.not_equal,~(tensor => F32Tensor.not_equal_scalar(tensor,value))):
  FunctionValues.public_context(~F32,~Logical,~original,~Compare.not_equal,~TiledMap.lanes(~F32,~Compare.not_equal),~TiledMaps.lanes_law(~F32,~Compare.not_equal),value,1.5)

def clamp(~Logical: Data,~original: Logical -> U32 -> F32,+low: F32,+high: F32) ->
  FunctionValues.ContextPublicLaw(~Compare.Bounds,~Compare.Bounds{low,high},~Logical,~original,~Compare.clamp,~(tensor => F32Tensor.clamp(tensor,low,high))):
  FunctionValues.public_context(~Compare.Bounds,~Logical,~original,~Compare.clamp,~TiledMap.lanes(~Compare.Bounds,~Compare.clamp),
    ~TiledMaps.lanes_law(~Compare.Bounds,~Compare.clamp),Compare.Bounds{low,high},0.5)
