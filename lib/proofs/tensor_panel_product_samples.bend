# Coordinates and storage bounds for a single image of a tiled operand.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../tensor_grid.bend as Grid
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./kernel_gemm_output_groups.bend as Groups
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./kernel_tile_refinement.bend as TileLaws
import ./row_major_index.bend as Index
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_halo_windows.bend as Halo
import ./tensor_move_readback.bend as Held
import ./tensor_point_product_samples.bend as Samples
import ./tensor_point_product_windows.bend as Bands
import ./tensor_shape_refinement.bend as Shape
import ./tensor_tiled_representation.bend as Representation

def cell_width(+rows: U32,+channels: U32) -> U32:
  (rows * Gemm.blocks(channels) : U32)

def row_total(+images: U32,+rows: U32) -> Nat:
  Nat.mul(U32.to_nat(images),U32.to_nat(rows))

def image_size(+rows: U32,+channels: U32) -> Nat:
  Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels)))

def physical_size(+images: U32,+rows: U32,+channels: U32) -> Nat:
  Nat.mul(row_total(images,rows),U32.to_nat(Gemm.blocks(channels)))

def cell_count(+images: U32,+rows: U32,+channels: U32) ->
  {Shape.elements([images,rows,Gemm.blocks(channels)]) == physical_size(images,rows,channels) : Nat}:
  Equal.trans(Nat,Shape.elements([images,rows,Gemm.blocks(channels)]),Nat.mul(U32.to_nat(images),image_size(rows,channels)),physical_size(images,rows,channels),
    Shape.elements_three(images,rows,Gemm.blocks(channels)),
    Equal.sym(Nat,physical_size(images,rows,channels),Nat.mul(U32.to_nat(images),image_size(rows,channels)),
      Index.assoc(U32.to_nat(images),U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels)))))

type Size<-limits: Machine.Limits,-images: U32,-rows: U32,-channels: U32> is Data:
  Size{width: {U32.to_nat(cell_width(rows,channels)) == image_size(rows,channels) : Nat},
    outer: {U32.to_nat((images * rows : U32)) == row_total(images,rows) : Nat},
    image_room: {Nat.is_le(image_size(rows,channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    room: {Nat.is_le(physical_size(images,rows,channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def size(+limits: Machine.Limits,+images: U32,+rows: U32,+channels: U32,
  +images_positive: {Nat.is_lt(0n,U32.to_nat(images)) == True{} : Bool},
  +groups_positive: {Nat.is_lt(0n,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +room: {Nat.is_le(Shape.elements([images,rows,Gemm.blocks(channels)]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Size<limits,images,rows,channels>:
  +count = U32.to_nat(images)
  +rn = U32.to_nat(rows)
  +groups = U32.to_nat(Gemm.blocks(channels))
  +outer = row_total(images,rows)
  +width = image_size(rows,channels)
  +total = physical_size(images,rows,channels)
  +cap = Machine.capacity(limits)
  +bound = U32.to_nat(cap)
  +fits = {%cell_count(images,rows,channels) : {Nat.is_le(_,bound) == True{} : Bool} room : {Nat.is_le(total,bound) == True{} : Bool}}
  +area_room = {%Index.assoc(count,rn,groups) : {Nat.is_le(_,bound) == True{} : Bool} fits : {Nat.is_le(Nat.mul(count,width),bound) == True{} : Bool}}
  +image_room = Order.transitive(width,Nat.mul(count,width),bound,
    %Index.multiply_commutative(width,count) : {Nat.is_le(width,_) == True{} : Bool} Index.product_covers(width,count,images_positive),area_room)
  Size{Arithmetic.product_exact(cap,Limits.capacity_fits(limits),rows,Gemm.blocks(channels),rn,groups,{==},{==},image_room),
    Arithmetic.product_exact(cap,Limits.capacity_fits(limits),images,rows,count,rn,{==},{==},
      Order.transitive(outer,total,bound,Index.product_covers(outer,groups,groups_positive),fits)),image_room,fits}

def band_start(+image: U32,+rows: U32,+channels: U32) -> U32:
  (image * cell_width(rows,channels) : U32)

def one_more(+first: Nat) -> {Nat.add(first,1n) == 1n+first : Nat}:
  Order.commutative(first,1n)

def first_plus_one(+first: Nat,+width: Nat) -> {Nat.mul(Nat.add(first,1n),width) == Nat.add(Nat.mul(first,width),width) : Nat}:
  Equal.trans(Nat,Nat.mul(Nat.add(first,1n),width),Nat.add(Nat.mul(first,width),Nat.mul(1n,width)),Nat.add(Nat.mul(first,width),width),
    Index.distrib(first,1n,width),Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(first,width),value),Nat.mul(1n,width),width,Addition.zero_right(width)))

type ImageBand<-images: U32,-rows: U32,-channels: U32,-image: U32> is Data:
  ImageBand{start: {U32.to_nat(band_start(image,rows,channels)) == Nat.mul(Nat.mul(U32.to_nat(image),U32.to_nat(rows)),U32.to_nat(Gemm.blocks(channels))) : Nat},
    inside: {Nat.is_le(Nat.add(U32.to_nat(band_start(image,rows,channels)),U32.to_nat(cell_width(rows,channels))),physical_size(images,rows,channels)) == True{} : Bool}}

def image_band(+limits: Machine.Limits,+images: U32,+rows: U32,+channels: U32,+image: U32,
  +inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},facts: Size<limits,images,rows,channels>) -> ImageBand<images,rows,channels,image>:
  Size{+width_exact,outer_exact,image_room,+room} = facts
  +first = U32.to_nat(image)
  +width_word = cell_width(rows,channels)
  +width = image_size(rows,channels)
  +total = physical_size(images,rows,channels)
  +cap = Machine.capacity(limits)
  +last = Nat.add(first,1n)
  +end_at = Nat.add(Nat.mul(first,width),width)
  +end = {%Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(rows)),U32.to_nat(Gemm.blocks(channels))),Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels)))),
    Index.assoc(U32.to_nat(images),U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels)))) : {Nat.is_le(end_at,_) == True{} : Bool}
    %first_plus_one(first,width) : {Nat.is_le(_,Nat.mul(U32.to_nat(images),width)) == True{} : Bool}
    Index.mul_le(last,U32.to_nat(images),width,
      %Equal.sym(Nat,last,1n+first,one_more(first)) : {Nat.is_le(_,U32.to_nat(images)) == True{} : Bool} Order.strict_follow(first,U32.to_nat(images),inside)) :
    {Nat.is_le(end_at,total) == True{} : Bool}}
  +word_start = Arithmetic.product_exact(cap,Limits.capacity_fits(limits),image,width_word,first,width,{==},width_exact,
    Order.transitive(Nat.mul(first,width),end_at,U32.to_nat(cap),Order.le_add(Nat.mul(first,width),width),Order.transitive(end_at,total,U32.to_nat(cap),end,room)))
  ImageBand{Equal.trans(Nat,U32.to_nat(band_start(image,rows,channels)),Nat.mul(first,width),Nat.mul(Nat.mul(first,U32.to_nat(rows)),U32.to_nat(Gemm.blocks(channels))),
      word_start,Equal.sym(Nat,Nat.mul(Nat.mul(first,U32.to_nat(rows)),U32.to_nat(Gemm.blocks(channels))),Nat.mul(first,width),Index.assoc(first,U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels))))),
    %Equal.sym(Nat,U32.to_nat(band_start(image,rows,channels)),Nat.mul(first,width),word_start) : {Nat.is_le(Nat.add(_,U32.to_nat(width_word)),total) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(width_word),width,width_exact) : {Nat.is_le(Nat.add(Nat.mul(first,width),_),total) == True{} : Bool} end}

def ImageWindow(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+images: U32,+rows: U32,+channels: U32,+image: U32,+holding: Held.Holding<Tile.Tile8>) -> Data:
  Halo.TaskWindow<Tile.Tile8,Representation.Lifted<Source>,Representation.lifted(~Source,~value),
    Representation.Lifted{source,Samples.row_geometry((images * rows : U32),channels)},holding,band_start(image,rows,channels),
    (band_start(image,rows,channels) + cell_width(rows,channels) : U32),Shape.elements([images,rows,Gemm.blocks(channels)])>

def SampleAt(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+rows: U32,+channels: U32,+image: U32,+holding: Held.Holding<Tile.Tile8>,+row: Nat,+group: Nat,+lane: Nat) -> Data:
  {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Bands.band_model(holding,band_start(image,rows,channels),cell_width(rows,channels))),
    U32.from_nat(Nat.add(U32.to_nat(Bands.band_base(holding,band_start(image,rows,channels),cell_width(rows,channels))),Nat.add(Nat.mul(row,U32.to_nat(Gemm.blocks(channels))),group))))) ==
    value(source,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(rows)),row),U32.to_nat(channels)),Nat.add(lane,Nat.mul(group,8n))))) : F32}

def sample_in_band(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+limits: Machine.Limits,+images: U32,+rows: U32,+channels: U32,+image: U32,+holding: Held.Holding<Tile.Tile8>,
  +image_inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},size: Size<limits,images,rows,channels>,
  band: ImageBand<images,rows,channels,image>,window: ImageWindow(~Source,~value,source,images,rows,channels,image,holding),+row: Nat,+group: Nat,+lane: Nat,
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},+active: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool}) ->
  SampleAt(~Source,~value,source,rows,channels,image,holding,row,group,lane):
  Size{+width_exact,+outer_exact,image_room,+room} = size
  ImageBand{+start_exact,+contains} = band
  +outer = (images * rows : U32)
  +start = band_start(image,rows,channels)
  +count = cell_width(rows,channels)
  +before = Nat.mul(U32.to_nat(image),U32.to_nat(rows))
  +groups = U32.to_nat(Gemm.blocks(channels))
  +total = physical_size(images,rows,channels)
  +global = Nat.add(before,row)
  +shape = {[images,rows,Gemm.blocks(channels)] : List<&2,U32>}
  +reader = {Representation.Lifted{source,Samples.row_geometry(outer,channels)} : Representation.Lifted<Source>}
  Samples.band_row_sample(~Source,~value,source,limits,outer,channels,holding,start,count,before,U32.to_nat(rows),start_exact,width_exact,
    %Equal.sym(Nat,U32.to_nat(outer),row_total(images,rows),outer_exact) : {Nat.is_le(Nat.mul(_,groups),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room,
    %Equal.sym(Nat,U32.to_nat(outer),row_total(images,rows),outer_exact) : {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),Nat.mul(_,groups)) == True{} : Bool} contains,
    %Equal.sym(Nat,U32.to_nat(outer),row_total(images,rows),outer_exact) : Halo.TaskWindow<Tile.Tile8,Representation.Lifted<Source>,Representation.lifted(~Source,~value),reader,holding,start,(start + count : U32),Nat.mul(_,groups)>
    %cell_count(images,rows,channels) : Halo.TaskWindow<Tile.Tile8,Representation.Lifted<Source>,Representation.lifted(~Source,~value),reader,holding,start,(start + count : U32),_> window,
    row,group,lane,row_inside,
    %Equal.sym(Nat,U32.to_nat(outer),row_total(images,rows),outer_exact) : {Nat.is_lt(global,_) == True{} : Bool} Index.radix(U32.to_nat(image),U32.to_nat(images),row,U32.to_nat(rows),image_inside,row_inside),group_inside,lane_inside,active)

def sample(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+limits: Machine.Limits,+images: U32,+rows: U32,+channels: U32,+image: U32,+holding: Held.Holding<Tile.Tile8>,
  +image_inside: {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool},+size: Size<limits,images,rows,channels>,
  window: ImageWindow(~Source,~value,source,images,rows,channels,image,holding),+row: Nat,+group: Nat,+lane: Nat,
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},+active: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool}) ->
  SampleAt(~Source,~value,source,rows,channels,image,holding,row,group,lane):
  sample_in_band(~Source,~value,source,limits,images,rows,channels,image,holding,image_inside,size,
    image_band(limits,images,rows,channels,image,image_inside,size),window,row,group,lane,row_inside,group_inside,lane_inside,active)
