# Row-layout costs follow the actual halo ranges and allocation capacities.
# Preparation is serial. Fork copies weights and bias, never input cores.
import Base
import ./kernel_cost.bend as Cost
import ./kernel_planning.bend as KernelPlanning
import ./tensor_band.bend as Bands
import ./tensor_band_window.bend as BandWindow
import ./kernel_window.bend as KernelWindow

# The whole batch is one parallel region: span is its costliest leaf path,
# leaves its leaves, across every image.
type Movement is Data:
  Movement{work: F32,span: F32,preparation: F32,minimum: F32,leaves: U32}

type Plan is Data:
  Plan{depth: U32,parallel: Bool,estimated_ns: F32}

def transfer(+count: U32,byte_ns: F32) -> F32:
  ((U32.to_f32(count) * 4.0 + KernelPlanning.payload_float(Nat.mul(4n,KernelPlanning.capacity(count)))) * byte_ns : F32)

def leaf_cost(preparation: F32,+tail: F32,cost: KernelPlanning.Movement) -> Movement:
  KernelPlanning.Movement{packed,output,copies,span,total,minimum} = cost
  Movement{(total + tail : F32),(span + tail : F32),preparation,minimum,1}

# BandWindow.assemble reuses the core exactly when both row ranges agree.
def assembly(channels: U32,width: U32,first: U32,rows: U32,window_first: U32,+window_rows: U32,byte_ns: F32) -> F32:
  Bool.pick(F32,U32.is_eq(first,window_first) && U32.is_eq(rows,window_rows),0.0,transfer((channels * window_rows * width : U32),byte_ns))

def leaf_window(+work: KernelPlanning.Work,first: U32,count: U32,window: BandWindow.RowWindow,+config: Cost.Configuration) -> Movement:
  match work window config:
    case KernelPlanning.Work{positions,inner,rows,owners,KernelPlanning.Geometry{+channels,h,+width,k,s,p,+ow},activation} BandWindow.RowWindow{BandWindow.Rows{out_first,+out_count},BandWindow.Rows{af,+ac},BandWindow.Rows{bf,+bc},BandWindow.Rows{wf,+wc}} Cost.Configuration{mac,+byte_ns,operation_ns,grain,wake}:
      leaf_cost((transfer((channels * ac * width : U32),byte_ns) + transfer((channels * bc * width : U32),byte_ns) : F32),
        assembly(channels,width,first,count,wf,wc,byte_ns),
        KernelPlanning.leaf_cost(KernelWindow.padded((out_count * ow : U32)),(out_count * ow : U32),work,config))

def merge(clone: F32,left: Movement,right: Movement) -> Movement:
  Movement{lw,ls,lp,lmin,ln} = left
  Movement{rw,rs,rp,rmin,rn} = right
  +clone = clone
  Movement{(clone + lw + rw : F32),(clone + F32.max(ls,rs) : F32),(lp + rp : F32),F32.min(lmin,rmin),(ln + rn : U32)}

def clone_cost(work: KernelPlanning.Work,config: Cost.Configuration) -> F32:
  Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} = config
  (KernelPlanning.payload_float(KernelPlanning.fork_bytes(work,0n)) * byte_ns : F32)

def reorder_cost(work: KernelPlanning.Work,config: Cost.Configuration) -> F32:
  KernelPlanning.reorder_ns(work,config)

# What the movement of a layout is priced with.
type Priced is Data:
  Priced{workers: U32,work: KernelPlanning.Work,window: BandWindow.Geometry,config: Cost.Configuration}

def leaf_movement(context: Priced,first: U32,count: U32) -> Movement:
  Priced{workers,work,+window,config} = context
  +first = first
  +count = count
  leaf_window(work,first,count,BandWindow.window(window,first,count),config)

def split_movement(context: Priced,left: Bands.Plan,right: Bands.Plan,first: Movement,second: Movement) -> Movement:
  Priced{workers,+work,window,+config} = context
  merge(clone_cost(work,config),first,second)

def movement(layout: Bands.Plan,+workers: U32,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+config: Cost.Configuration) -> Movement:
  Bands.plan_fold(~Movement,~Priced,~leaf_movement,~split_movement,layout,Priced{workers,work,window,config})

def estimate(parallel: Bool,+workers: U32,initial_copy: F32,cost: Movement) -> F32:
  Movement{+work,+span,preparation,minimum,+leaves} = cost
  (initial_copy + preparation + Bool.pick(F32,parallel,Cost.region_ns(workers,leaves,span),work) : F32)

def choose(allowed: Bool,depth: U32,parallel: Bool,+time: F32,best: Plan) -> Plan:
  Plan{old_depth,old_parallel,+old_time} = best
  Bool.pick(Plan,allowed && F32.is_lt(time,old_time),Plan{depth,parallel,time},Plan{old_depth,old_parallel,old_time})

# What a transfer of bands is priced with.
type Moved is Data:
  Moved{geometry: Bands.Geometry,byte_ns: F32}

def leaf_transfer(context: Moved,first: U32,rows: U32) -> F32:
  Moved{geometry,byte_ns} = context
  transfer(Bands.local_count(geometry,rows),byte_ns)

def add_transfer(first: F32,second: F32) -> F32:
  (first + second : F32)

def transfer_layout(layout: Bands.Plan,+geometry: Bands.Geometry,+byte_ns: F32) -> F32:
  Bands.plan_fold(~F32,~Moved,~leaf_transfer,~Bands.plain(~F32,~Moved,~add_transfer),layout,Moved{geometry,byte_ns})

# The two wake rounds of the one parallel region.
def wake_cost(config: Cost.Configuration) -> F32:
  Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} = config
  (2.0 * wake : F32)

def candidate(+depth: U32,+workers: U32,+initial_copy: F32,+wake: F32,+cost: Movement,config: Cost.Configuration,best: Plan) -> Plan:
  match cost config:
    case Movement{work,span,preparation,minimum,leaves} Cost.Configuration{mac,byte_ns,operation_ns,grain,round}:
      choose(F32.is_le(grain,work),depth,True{},(estimate(True{},workers,initial_copy,cost) + wake : F32),best)

def initial_candidate(depth: U32,+layout: Bands.Plan,+workers: U32,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+config: Cost.Configuration,best: Plan) -> Plan:
  Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} = config
  candidate(depth,workers,(transfer_layout(layout,BandWindow.input_geometry(window),byte_ns) + reorder_cost(work,config) : F32),
    wake_cost(config),movement(layout,workers,work,window,config),config,best)

def candidates(remaining: Nat,allowed: Bool,+depth: U32,+tasks: U32,+ceiling: U32,+workers: U32,+batch: U32,+height: U32,
  +work: KernelPlanning.Work,+window: BandWindow.Geometry,+config: Cost.Configuration,best: Plan) -> Plan:
  match remaining allowed:
    case 0n allowed: best
    case 1n+rest False{}: best
    case 1n+rest True{}:
      +next_tasks = (tasks * 2 : U32)
      candidates(rest,U32.is_le(next_tasks,ceiling) && U32.is_le(next_tasks,height),(depth + 1 : U32),next_tasks,ceiling,workers,batch,height,
        work,window,config,initial_candidate(depth,Bands.image_plan(U32.to_nat(batch),0,U32.to_nat(depth),height),workers,work,window,config,best))

def unsplit_baseline(+work: KernelPlanning.Work,window: BandWindow.Geometry,+config: Cost.Configuration) -> F32:
  match work window config:
    case KernelPlanning.Work{+positions,inner,+rows,owners,geometry,activation} BandWindow.Geometry{Bands.Geometry{+batch,channels,height,width},kernel,stride,padding,oh,ow} Cost.Configuration{mac,+byte_ns,operation_ns,grain,wake}:
      +one = KernelPlanning.estimate(0,work,config,KernelPlanning.movement(0n,0,KernelWindow.padded(positions),KernelPlanning.root_rows(work),work,config),1)
      +output_size = (rows * positions : U32)
      +images = U32.to_f32(batch)
      Bool.pick(F32,U32.is_le(batch,1),one,
        (reorder_cost(work,config) + images * (one - reorder_cost(work,config) + transfer((channels * height * width : U32),byte_ns) + U32.to_f32(output_size) * 4.0 * byte_ns) +
          KernelPlanning.payload_float(Nat.mul(4n,KernelPlanning.capacity((batch * output_size : U32)))) * byte_ns + (images - 1.0) * clone_cost(work,config) : F32))

# Grain bounds the operation's work; the two wake rounds are charged separately.
def select(+work: KernelPlanning.Work,+window: BandWindow.Geometry,workers: U32,+config: Cost.Configuration) -> Plan:
  match work window:
    case KernelPlanning.Work{positions,inner,rows,owners,KernelPlanning.Geometry{channels,+height,width,k,s,p,ow},activation} BandWindow.Geometry{Bands.Geometry{batch,ci,h,w},kernel,stride,padding,oh,owidth}:
      +workers = U32.max(1,U32.min(workers,858993459))
      candidates(32n,U32.is_lt(1,workers) && (U32.is_lt(1,batch) || U32.is_le(2,height)),0,1,(workers * 5 : U32),workers,batch,height,work,window,config,
        Plan{0,False{},unsplit_baseline(work,window,config)})

type LayoutPlan is Data:
  LayoutPlan{layout: Bands.Plan,parallel: Bool,estimated_ns: F32,repartition_ns: F32}

def retained_parallel(window: BandWindow.Geometry,workers: U32,config: Cost.Configuration,cost: Movement) -> Bool:
  BandWindow.Geometry{Bands.Geometry{+batch,channels,height,width},kernel,stride,padding,oh,ow} = window
  Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} = config
  Movement{+total,+span,preparation,minimum,+leaves} = cost
  +workers = U32.max(1,U32.min(workers,858993459))
  U32.is_lt(1,workers) && Nat.is_le(U32.to_nat(leaves),Nat.mul(U32.to_nat(batch),U32.to_nat((workers * 5 : U32)))) &&
    F32.is_le(grain,total) && F32.is_lt((Cost.region_ns(workers,leaves,span) + 2.0 * wake : F32),total)

def layout_cost(layout: Bands.Plan,+parallel: Bool,workers: U32,+repartition: F32,wake: F32,reorder: F32,cost: Movement) -> LayoutPlan:
  LayoutPlan{layout,parallel,(reorder + estimate(parallel,workers,repartition,cost) + Bool.pick(F32,parallel,wake,0.0) : F32),repartition}

def evaluated(+layout: Bands.Plan,+previous: Bands.Plan,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+workers: U32,+config: Cost.Configuration,+cost: Movement) -> LayoutPlan:
  Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} = config
  layout_cost(layout,retained_parallel(window,workers,config,cost),workers,
    Bool.pick(F32,Bands.same_plan(layout,previous),0.0,transfer_layout(layout,BandWindow.input_geometry(window),byte_ns)),wake_cost(config),reorder_cost(work,config),cost)

def prefer(candidate: LayoutPlan,best: LayoutPlan) -> LayoutPlan:
  match candidate best:
    case LayoutPlan{layout,parallel,+time,copies} LayoutPlan{old,old_parallel,+old_time,old_copies}:
      Bool.pick(LayoutPlan,F32.is_lt(time,old_time),LayoutPlan{layout,parallel,time,copies},LayoutPlan{old,old_parallel,old_time,old_copies})

def layout_candidate(+layout: Bands.Plan,previous: Bands.Plan,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+workers: U32,+config: Cost.Configuration,best: LayoutPlan) -> LayoutPlan:
  prefer(evaluated(layout,previous,work,window,workers,config,movement(layout,workers,work,window,config)),best)

def layouts(remaining: Nat,allowed: Bool,+depth: U32,+tasks: U32,+ceiling: U32,+height: U32,+batch: U32,+previous: Bands.Plan,
  +work: KernelPlanning.Work,+window: BandWindow.Geometry,+workers: U32,+config: Cost.Configuration,best: LayoutPlan) -> LayoutPlan:
  match remaining allowed:
    case 0n allowed: best
    case 1n+rest False{}: best
    case 1n+rest True{}:
      +next_tasks = (tasks * 2 : U32)
      layouts(rest,U32.is_le(next_tasks,ceiling) && U32.is_le(next_tasks,height),(depth + 1 : U32),next_tasks,ceiling,height,batch,previous,work,window,workers,config,
        layout_candidate(Bands.image_plan(U32.to_nat(batch),0,U32.to_nat(depth),height),previous,work,window,workers,config,best))

def replan(+previous: Bands.Plan,+work: KernelPlanning.Work,+window: BandWindow.Geometry,workers: U32,+config: Cost.Configuration) -> LayoutPlan:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,oh,ow} = window
  +workers = U32.max(1,U32.min(workers,858993459))
  layouts(32n,True{},0,1,Bool.pick(U32,U32.is_eq(workers,1),1,(workers * 5 : U32)),height,batch,previous,work,window,workers,config,
    evaluated(previous,previous,work,window,workers,config,movement(previous,workers,work,window,config)))

# The first split of every image as a layout plan: nothing to repartition.
def first_split(plan: Plan,+window: BandWindow.Geometry) -> LayoutPlan:
  Plan{depth,parallel,time} = plan
  BandWindow.Geometry{Bands.Geometry{batch,channels,+height,width},kernel,stride,padding,oh,ow} = window
  LayoutPlan{Bands.image_plan(U32.to_nat(batch),0,U32.to_nat(depth),height),parallel,time,0.0}

# The layout a window operation runs in: chosen again from the input's plan
# when the input holds the window's geometry, otherwise the first split.
def layout_for(same: Bool,+previous: Bands.Plan,+work: KernelPlanning.Work,+window: BandWindow.Geometry,workers: U32,+config: Cost.Configuration) -> LayoutPlan:
  match same:
    case True{}: replan(previous,work,window,workers,config)
    case False{}: first_split(select(work,window,workers,config),window)
