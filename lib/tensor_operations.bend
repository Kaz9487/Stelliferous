# Checked operations preserve all owners and their balanced-storage certificates.
# Element functions enter hot loops only as direct templates.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./traversal_array.bend as Traversal
import ./tensor_access.bend as Access
import ./kernel_tile.bend as Tile
import ./math_sum.bend as Sum
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/traversal_certificate.bend as TraversalCertificate
import ./proofs/traversal_witness.bend as TraversalWitness
import ./proofs/tensor_access_certificate.bend as AccessCertificate

type Outcome<-Value: Type> is Type:
  Accepted{value: Value}
  Rejected{value: Value}

def valid_buffer(+length: Nat,capacity: U32,view: Views.View) -> Bool:
  Nat.is_le(length,U32.to_nat(capacity)) && Views.fits(view,U32.from_nat(length))

def pair_buffers(~Element: Data,input_length: Nat,output_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})

def combine_buffers(~Element: Data,left_length: Nat,right_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  (right,left) = result
  TraversalCertificate.Certificates{right_certificate,left_certificate} = certificates
  (Storage.Buffer{left,left_length,left_certificate},Storage.Buffer{right,right_length,right_certificate})

# yes[i] = choose(mask[i], yes[i], no[i]) for the first count positions; the
# mask and no buffers are consumed.
def chosen_buffer(~Element: Data,length: Nat,result: (Array<Element> & Array<Element>) & Array<Element>,
  certificate: Certificate.Certificate<Element,TraversalWitness.second(~(Array<Element> & Array<Element>),~Array<Element>,result)>) -> Storage.Buffer<Element>:
  (reader,output) = result
  Storage.Buffer{output,length,certificate}

def choose_kernel(~Element: Data,~choose: Element -> Element -> Element -> Element,+count: Nat,
  mask: Storage.Buffer<Element>,yes: Storage.Buffer<Element>,no: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Storage.Buffer{mask,mask_length,mask_certificate} = mask
  Storage.Buffer{yes,yes_length,yes_certificate} = yes
  Storage.Buffer{no,no_length,no_certificate} = no
  chosen_buffer(~Element,yes_length,Access.choose_run(~Element,~choose,count,mask,no,yes),
    AccessCertificate.choose(~Element,~choose,count,mask,no,yes,mask_certificate,no_certificate,yes_certificate))

def combine_kernel(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+right_view: Views.View,+offset: U32,
  left: Storage.Buffer<Element>,right: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  # the view built again for each use: a View used twice shares every View
  Views.View{+axes,+view_offset} = right_view
  Storage.Buffer{left,left_length,left_certificate} = left
  Storage.Buffer{right,right_length,right_certificate} = right
  +fast = Access.zero_contiguous(Views.View{axes,view_offset}) && U32.is_zero(offset)
  combine_buffers(~Element,left_length,right_length,
    Access.combine_run(~Element,~operation,fast,count,Views.View{axes,view_offset},offset,right,left),
    AccessCertificate.combine(~Element,~operation,fast,count,Views.View{axes,view_offset},offset,right,left,right_certificate,left_certificate))

def combine_when_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,
  valid: Bool,count: Maybe<&2,U32>,right_view: Views.View,target: Views.View,left: Storage.Buffer<Element>,right: Storage.Buffer<Element>) ->
  Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  match valid count:
    case True{} Some{count}:
      Views.View{axes,offset} = target
      Accepted{kernel(U32.to_nat(count),right_view,offset,left,right)}
    case valid count: Rejected{(left,right)}

def combine_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,
  left: Storage.Buffer<Element>,+left_view: Views.View,right: Storage.Buffer<Element>,+right_view: Views.View) ->
  Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  # the views built again for each use: a View used twice shares every View
  Storage.Buffer{left,+left_length,left_certificate} = left
  Certificate.Certificate{+left_depth,left_equation} = left_certificate
  Views.View{+left_axes,+left_offset} = left_view
  Storage.Buffer{right,+right_length,right_certificate} = right
  Certificate.Certificate{+right_depth,right_equation} = right_certificate
  Views.View{+right_axes,+right_offset} = right_view
  combine_when_with(~Element,~kernel,Shapes.same(Views.shape(Views.View{left_axes,left_offset}),Views.shape(Views.View{right_axes,right_offset})) &&
    Views.is_contiguous(Views.View{left_axes,left_offset}) &&
    valid_buffer(left_length,U32.shln(1,left_depth),Views.View{left_axes,left_offset}) &&
    valid_buffer(right_length,U32.shln(1,right_depth),Views.View{right_axes,right_offset}),Views.numel(Views.View{left_axes,left_offset}),
    Views.View{right_axes,right_offset},Views.View{left_axes,left_offset},
    Storage.Buffer{left,left_length,Certificate.Certificate{left_depth,left_equation}},Storage.Buffer{right,right_length,Certificate.Certificate{right_depth,right_equation}})

# A mapper is the ordered map of count values from position on a certified
# Buffer. mapped_context applies the operation to one value at a time with
# its context.
def mapped_context(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+count: Nat,+position: U32,+context: Context,buffer: Storage.Buffer<Element>) ->
  Storage.Buffer<Element>:
  Storage.Buffer{storage,length,certificate} = buffer
  Storage.Buffer{Traversal.map_in_place(~Element,~Context,~operation,count,position,context,storage),length,
    TraversalCertificate.map_in_place(~Element,~Context,~operation,count,position,context,storage,certificate)}

def small(value: U32) -> Bool:
  U32.is_le(value,65535)

# The window geometry is exact and every sample address stays in U32.
def pool_geometry(+pool: Access.Pool) -> Bool:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  small(height) && small(width) && small(kernel) && small(stride) && small(padding) && U32.is_lt(0,kernel) && U32.is_lt(0,stride) &&
    U32.is_le(kernel,(height + 2 * padding : U32)) && U32.is_le(kernel,(width + 2 * padding : U32)) &&
    U32.is_eq(out_height,((height + 2 * padding - kernel : U32) / stride + 1 : U32)) &&
    U32.is_eq(out_width,((width + 2 * padding - kernel : U32) / stride + 1 : U32))

def range_folded(~Element: Data,~Accumulator: Data,+length: Nat,result: Array<Element> & Accumulator,
  certificate: Certificate.Certificate<Element,TraversalWitness.first(~Array<Element>,~Accumulator,result)>) -> Storage.Buffer<Element> & Accumulator:
  (input,accumulator) = result
  (Storage.Buffer{input,length,certificate},accumulator)

# The fold of count values from position on, step apart, into initial, in
# order; the buffer comes back unchanged.
def fold_range(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,+count: Nat,+position: U32,+step: U32,initial: Accumulator,
  buffer: Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Accumulator:
  Storage.Buffer{input,+length,+certificate} = buffer
  range_folded(~Element,~Accumulator,length,
    Traversal.fold(~Array<Element>,~Element,~Accumulator,~Traversal.Strided,~Traversal.strided(~Element),~operation,count,0,input,initial,Traversal.Strided{position,step}),
    AccessCertificate.fold_range(~Element,~Accumulator,~operation,count,position,step,initial,input,certificate))

# ---- Eight sums at once in place ----

# The sums of the eight lanes of a row: slot k of parts holds the parts of the
# values whose index in the row is k modulo 8, phase counts the values, and
# every read takes extra.
type TileSlots<-Extra: Data> is Type:
  TileSlots{parts: Storage.Buffer<Sum.TileParts>,phase: U32,extra: Extra}

def slots_begun(~Extra: Data,extra: Extra) -> TileSlots<Extra>:
  TileSlots{Storage.allocate(~Sum.TileParts,3n,Sum.tile_parts_zero(),8n),0,extra}

# The slot of the value at position, positions counting the values of the row.
def slot_of(~Extra: Data,context: Traversal.Spaced<Extra>,position: U32) -> U32:
  U32.and(position,7)

# The slots merge in a tree read in place: ((0 1)(2 3))((4 5)(6 7)).
def merged_after(left: Sum.TileParts,second: Array<Sum.TileParts> & Sum.TileParts) -> Array<Sum.TileParts> & Sum.TileParts:
  (slots,right) = second
  (slots,Sum.tile_merge(left,right))

def pair_rest(+index: U32,first: Array<Sum.TileParts> & Sum.TileParts) -> Array<Sum.TileParts> & Sum.TileParts:
  (slots,left) = first
  merged_after(left,Array.get(Sum.TileParts,slots,(index + 1 : U32)))

def pair_at(+index: U32,slots: Array<Sum.TileParts>) -> Array<Sum.TileParts> & Sum.TileParts:
  pair_rest(index,Array.get(Sum.TileParts,slots,index))

def quad_rest(+index: U32,first: Array<Sum.TileParts> & Sum.TileParts) -> Array<Sum.TileParts> & Sum.TileParts:
  (slots,left) = first
  merged_after(left,pair_at((index + 2 : U32),slots))

def quad_at(+index: U32,slots: Array<Sum.TileParts>) -> Array<Sum.TileParts> & Sum.TileParts:
  quad_rest(index,pair_at(index,slots))

def eight_rest(first: Array<Sum.TileParts> & Sum.TileParts) -> Sum.TileParts:
  (slots,left) = first
  Pair.snd(Array<Sum.TileParts>,Sum.TileParts,merged_after(left,quad_at(4,slots)))

def slots_closed(~Extra: Data,row: TileSlots<Extra>) -> Sum.TileParts:
  TileSlots{parts,phase,extra} = row
  Storage.Buffer{slots,length,certificate} = parts
  eight_rest(quad_at(0,slots))

def slots_buffers(~Extra: Data,+length: Nat,+slots_length: Nat,+phase: U32,extra: Extra,result: Array<Tile.Tile8> & Array<Sum.TileParts>,
  certificates: TraversalCertificate.Certificates<Array<Tile.Tile8>,Sum.TileParts,Certificate.family(~Tile.Tile8),result>) -> Storage.Buffer<Tile.Tile8> & TileSlots<Extra>:
  (input,slots) = result
  TraversalCertificate.Certificates{certificate,slots_certificate} = certificates
  (Storage.Buffer{input,length,certificate},TileSlots{Storage.Buffer{slots,slots_length,slots_certificate},phase,extra})

# count cells from position on, step apart, read into the slots in place, the
# buffer handed back.
def cells_slots(~Extra: Data,~read: Array<Tile.Tile8> -> Traversal.Spaced<Extra> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,
  ~read_preserves: @-reader: Array<Tile.Tile8> -> @context: Traversal.Spaced<Extra> -> @index: U32 -> TraversalWitness.array_evidence(~Tile.Tile8,reader) ->
    TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,reader,read(reader,context,index)>,
  +count: Nat,+position: U32,+step: U32,initial: TileSlots<Extra>,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8> & TileSlots<Extra>:
  TileSlots{parts,+phase,+extra} = initial
  Storage.Buffer{slots,+slots_length,slots_certificate} = parts
  Storage.Buffer{input,+length,certificate} = buffer
  Certificate.Certificate{+depth,+equation} = certificate
  +run = {Traversal.Steps{count,phase,Traversal.Spaced{(position - phase * step : U32),step,extra}} : Traversal.Steps<Traversal.Spaced<Extra>>}
  slots_buffers(~Extra,length,slots_length,(phase + U32.from_nat(count) : U32),extra,
    Traversal.updated(~Array<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~read,~Sum.tile_slot_step,~slot_of(~Extra),run,input,slots),
    TraversalCertificate.updated_certificates(~Array<Tile.Tile8>,~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Extra>,~TraversalWitness.array_evidence(~Tile.Tile8),
      ~Certificate.family(~Tile.Tile8),~read,~Sum.tile_slot_step,~slot_of(~Extra),~read_preserves,~TraversalCertificate.array_observed(~Tile.Tile8),run,input,slots,
      Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},slots_certificate))
