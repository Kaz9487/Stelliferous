# Guard facts for actual public Buffer calls. Balance is recovered from Buffer;
# the only logical premises are the successful branches tested by the API.
import Base
import ./bounded_u32_arithmetic.bend as Arithmetic
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ./storage_certificate.bend as Certificate
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_contiguous_bounds.bend as ContiguousBounds
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_bounds.bend as CursorBounds
import ./traversal_write_bounds.bend as WriteBounds
import ./boolean_logic.bend as Boolean
import ./u32_full_range.bend as FullRange

def valid(~Element: Data,buffer: Storage.Buffer<Element>,view: Views.View) -> Bool:
  Storage.Buffer{storage,length,certificate} = buffer
  Operations.valid_buffer(length,Storage.capacity(~Element,storage,certificate),view)

def combine_guard(~Element: Data,left: Storage.Buffer<Element>,+left_view: Views.View,right: Storage.Buffer<Element>,+right_view: Views.View) -> Bool:
  Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) && valid(~Element,left,left_view) && valid(~Element,right,right_view)
