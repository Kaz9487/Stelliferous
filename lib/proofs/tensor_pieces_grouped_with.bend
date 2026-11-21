# Pieces.grouped_with over the windows of tasks: every task runs its leaf on
# the window of its range and its share of the resource, and the result is
# the tree of the task results.
import Base
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ../tensor_move.bend as Move
import ./traversal_partition_refinement.bend as PartitionLaws
import ./tensor_pieces_walk.bend as Walk
import ./tensor_halo_windows.bend as Windows
import ./tensor_move_readback.bend as Held

# What the leaf of one task gives: a result and a resource after it, the
# result with its property.
type TaskFact<-Element: Data,-Output: Type,-Resource: Type,-Context: Data,-OModel: Data,-RModel: Data,-share: Resource -> Resource & Resource,-leaf: Context -> U32 -> U32 -> Move.Holding<Element> -> Resource -> Partition.Tree<&1,Output,Unit> & Resource,-pack_o: OModel -> Output,-pack_r: RModel -> Resource,-split_r: RModel -> PartitionLaws.Split<RModel>,-Scope: Data,-Property: @+scope: Scope -> @+first: U32 -> @+count: U32 -> @+result: Partition.Tree<&2,OModel,Unit> -> Data,-scope: Scope,-context: Context,-holding: Held.Holding<Element>,-first: U32,-count: U32,-resource: RModel> is Data:
  TaskFact{result: Partition.Tree<&2,OModel,Unit>,rest: RModel,
    exact: {leaf(context,first,count,Held.pack(~Element,holding),pack_r(resource)) ==
      (PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,result),pack_r(rest)) : Partition.Tree<&1,Output,Unit> & Resource},
    holds: Property(scope,first,count,result)}

# The facts of the tasks of a layout over windows of the same shape, every
# split sharing the resource of its tasks.
def TaskFacts(~Element: Data,~Output: Type,~Resource: Type,~Context: Data,~OModel: Data,~RModel: Data,~share: Resource -> Resource & Resource,~leaf: Context -> U32 -> U32 -> Move.Holding<Element> -> Resource -> Partition.Tree<&1,Output,Unit> & Resource,~pack_o: OModel -> Output,~pack_r: RModel -> Resource,~split_r: RModel -> PartitionLaws.Split<RModel>,~Scope: Data,~Property: @+scope: Scope -> @+first: U32 -> @+count: U32 -> @+result: Partition.Tree<&2,OModel,Unit> -> Data,+scope: Scope,+context: Context,tasks: Pieces.Ranges,windows: Windows.WindowModel<Element>,+resource: RModel) -> Data:
  match tasks windows:
    case Pieces.Range{+first,+count} Windows.WindowLeaf{+holding,+task_first,+task_count}:
      TaskFact<Element,Output,Resource,Context,OModel,RModel,share,leaf,pack_o,pack_r,split_r,Scope,Property,scope,context,holding,task_first,task_count,resource>
    case Pieces.Split{+left,+right} Windows.WindowFork{+left_windows,+right_windows}:
      Held.Both<TaskFacts(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,left,left_windows,PartitionLaws.first(~RModel,split_r(resource))),
        TaskFacts(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,right,right_windows,PartitionLaws.second(~RModel,split_r(resource)))>
    case tasks windows:
      Empty

# The results of the tasks, joined as their layout.
def results(~Element: Data,~Output: Type,~Resource: Type,~Context: Data,~OModel: Data,~RModel: Data,~share: Resource -> Resource & Resource,~leaf: Context -> U32 -> U32 -> Move.Holding<Element> -> Resource -> Partition.Tree<&1,Output,Unit> & Resource,~pack_o: OModel -> Output,~pack_r: RModel -> Resource,~split_r: RModel -> PartitionLaws.Split<RModel>,~Scope: Data,~Property: @+scope: Scope -> @+first: U32 -> @+count: U32 -> @+result: Partition.Tree<&2,OModel,Unit> -> Data,+scope: Scope,+context: Context,tasks: Pieces.Ranges,windows: Windows.WindowModel<Element>,+resource: RModel,
  facts: TaskFacts(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,tasks,windows,resource)) -> Partition.Tree<&2,OModel,Unit>:
  match tasks:
    case Pieces.Range{+first,+count}:
      match windows:
        case Windows.WindowLeaf{+holding,+task_first,+task_count}:
          TaskFact{result,rest,exact,holds} = facts
          result
        case Windows.WindowFork{+left_windows,+right_windows}:
          Empty.absurd(Partition.Tree<&2,OModel,Unit>,facts)
    case Pieces.Split{+left,+right}:
      match windows:
        case Windows.WindowLeaf{+holding,+task_first,+task_count}:
          Empty.absurd(Partition.Tree<&2,OModel,Unit>,facts)
        case Windows.WindowFork{+left_windows,+right_windows}:
          Held.Both{left_facts,right_facts} = facts
          Partition.Fork{results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,left,left_windows,PartitionLaws.first(~RModel,split_r(resource)),left_facts),
            results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,right,right_windows,PartitionLaws.second(~RModel,split_r(resource)),right_facts)}

# The walk of the tasks from any mode is the tree of the task results.
def walked(~Element: Data,~Output: Type,~Resource: Type,~Context: Data,~OModel: Data,~RModel: Data,~share: Resource -> Resource & Resource,~leaf: Context -> U32 -> U32 -> Move.Holding<Element> -> Resource -> Partition.Tree<&1,Output,Unit> & Resource,~pack_o: OModel -> Output,~pack_r: RModel -> Resource,~split_r: RModel -> PartitionLaws.Split<RModel>,~Scope: Data,~Property: @+scope: Scope -> @+first: U32 -> @+count: U32 -> @+result: Partition.Tree<&2,OModel,Unit> -> Data,~share_matches: @+resource: RModel -> {share(pack_r(resource)) == (pack_r(PartitionLaws.first(~RModel,split_r(resource))),pack_r(PartitionLaws.second(~RModel,split_r(resource)))) : Resource & Resource},
  tasks: Pieces.Ranges,+mode: Bool,+enabled: Bool,+scope: Scope,+context: Context,+windows: Windows.WindowModel<Element>,+resource: RModel,
  +facts: TaskFacts(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,tasks,windows,resource)) ->
  {Pieces.each_part(~Pieces.Pieced(~Move.Holding<Element>,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Move.Holding<Element>,~Resource,~Context,~share),~Pieces.task_walk(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),~Pieces.task_kept(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),tasks,mode,enabled,context,(Windows.pack_windows(~Element,windows),pack_r(resource))) ==
    PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,tasks,windows,resource,facts)) : Partition.Tree<&1,Output,Unit>}:
  match tasks:
    case Pieces.Range{+first,+count}:
      match windows:
        case Windows.WindowLeaf{+holding,+task_first,+task_count}:
          TaskFact{+result,+rest,+exact,+holds} = facts
          %Equal.sym(Partition.Tree<&1,Output,Unit> & Resource,leaf(context,task_first,task_count,Held.pack(~Element,holding),pack_r(resource)),
              (PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,result),pack_r(rest)),exact) :
            {Pair.fst(Partition.Tree<&1,Output,Unit>,Resource,Pieces.walk_result(~Output,~Resource,Pieces.leaf_walked(~Output,~Resource,Pieces.NoLefts{},_))) == PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,result) : Partition.Tree<&1,Output,Unit>}
          {==}
        case Windows.WindowFork{+left_windows,+right_windows}:
          Empty.absurd({Pieces.each_part(~Pieces.Pieced(~Move.Holding<Element>,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Move.Holding<Element>,~Resource,~Context,~share),~Pieces.task_walk(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),~Pieces.task_kept(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),tasks,mode,enabled,context,(Windows.pack_windows(~Element,windows),pack_r(resource))) ==
            PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,tasks,windows,resource,facts)) : Partition.Tree<&1,Output,Unit>},facts)
    case Pieces.Split{+left,+right}:
      match windows:
        case Windows.WindowLeaf{+holding,+task_first,+task_count}:
          Empty.absurd({Pieces.each_part(~Pieces.Pieced(~Move.Holding<Element>,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Move.Holding<Element>,~Resource,~Context,~share),~Pieces.task_walk(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),~Pieces.task_kept(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),tasks,mode,enabled,context,(Windows.pack_windows(~Element,windows),pack_r(resource))) ==
            PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,tasks,windows,resource,facts)) : Partition.Tree<&1,Output,Unit>},facts)
        case Windows.WindowFork{+left_windows,+right_windows}:
          Held.Both{+left_facts,+right_facts} = facts
          +left_resource = PartitionLaws.first(~RModel,split_r(resource))
          +right_resource = PartitionLaws.second(~RModel,split_r(resource))
          %Equal.sym(Partition.Tree<&1,Output,Unit>,Pieces.each_part(~Pieces.Pieced(~Move.Holding<Element>,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Move.Holding<Element>,~Resource,~Context,~share),~Pieces.task_walk(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),~Pieces.task_kept(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),Pieces.Split{left,right},mode,enabled,context,(Windows.pack_windows(~Element,Windows.WindowFork{left_windows,right_windows}),pack_r(resource))),
              Walk.sequential(~Pieces.Pieced(~Move.Holding<Element>,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Move.Holding<Element>,~Resource,~Context,~share),~Pieces.task_walk(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),~Pieces.task_kept(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),left,right,enabled,context,Pieces.piece_divide(~Move.Holding<Element>,~Resource,~Context,~share,context,left,right,(Windows.pack_windows(~Element,Windows.WindowFork{left_windows,right_windows}),pack_r(resource)))),
              Walk.fork(~Pieces.Pieced(~Move.Holding<Element>,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Move.Holding<Element>,~Resource,~Context,~share),~Pieces.task_walk(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),~Pieces.task_kept(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),
                left,right,mode,enabled,context,(Windows.pack_windows(~Element,Windows.WindowFork{left_windows,right_windows}),pack_r(resource)))) :
            {_ == PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,Pieces.Split{left,right},Windows.WindowFork{left_windows,right_windows},resource,facts)) : Partition.Tree<&1,Output,Unit>}
          %Equal.sym(Resource & Resource,share(pack_r(resource)),(pack_r(left_resource),pack_r(right_resource)),share_matches(resource)) :
            {Walk.sequential(~Pieces.Pieced(~Move.Holding<Element>,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Move.Holding<Element>,~Resource,~Context,~share),~Pieces.task_walk(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),~Pieces.task_kept(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),left,right,enabled,context,Pieces.shared_halves(~Move.Holding<Element>,~Resource,Windows.pack_windows(~Element,left_windows),Windows.pack_windows(~Element,right_windows),_)) == PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,Pieces.Split{left,right},Windows.WindowFork{left_windows,right_windows},resource,facts)) : Partition.Tree<&1,Output,Unit>}
          %Equal.sym(Partition.Tree<&1,Output,Unit>,Pieces.each_part(~Pieces.Pieced(~Move.Holding<Element>,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Move.Holding<Element>,~Resource,~Context,~share),~Pieces.task_walk(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),~Pieces.task_kept(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),left,enabled,enabled,context,(Windows.pack_windows(~Element,left_windows),pack_r(left_resource))),
              PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,left,left_windows,left_resource,left_facts)),
              walked(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,~share_matches,left,enabled,enabled,scope,context,left_windows,left_resource,left_facts)) :
            {Partition.Fork{_,Pieces.each_part(~Pieces.Pieced(~Move.Holding<Element>,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Move.Holding<Element>,~Resource,~Context,~share),~Pieces.task_walk(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),~Pieces.task_kept(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),right,enabled,enabled,context,(Windows.pack_windows(~Element,right_windows),pack_r(right_resource)))} == PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,Pieces.Split{left,right},Windows.WindowFork{left_windows,right_windows},resource,facts)) : Partition.Tree<&1,Output,Unit>}
          %Equal.sym(Partition.Tree<&1,Output,Unit>,Pieces.each_part(~Pieces.Pieced(~Move.Holding<Element>,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Move.Holding<Element>,~Resource,~Context,~share),~Pieces.task_walk(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),~Pieces.task_kept(~Move.Holding<Element>,~Output,~Resource,~Context,~leaf),right,enabled,enabled,context,(Windows.pack_windows(~Element,right_windows),pack_r(right_resource))),
              PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,right,right_windows,right_resource,right_facts)),
              walked(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,~share_matches,right,enabled,enabled,scope,context,right_windows,right_resource,right_facts)) :
            {Partition.Fork{PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,left,left_windows,left_resource,left_facts)),_} == PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,Pieces.Split{left,right},Windows.WindowFork{left_windows,right_windows},resource,facts)) : Partition.Tree<&1,Output,Unit>}
          {==}

# Grouped with a shared resource, the tasks give the tree of their results.
def grouped(~Element: Data,~Output: Type,~Resource: Type,~Context: Data,~OModel: Data,~RModel: Data,~share: Resource -> Resource & Resource,~leaf: Context -> U32 -> U32 -> Move.Holding<Element> -> Resource -> Partition.Tree<&1,Output,Unit> & Resource,~pack_o: OModel -> Output,~pack_r: RModel -> Resource,~split_r: RModel -> PartitionLaws.Split<RModel>,~Scope: Data,~Property: @+scope: Scope -> @+first: U32 -> @+count: U32 -> @+result: Partition.Tree<&2,OModel,Unit> -> Data,~share_matches: @+resource: RModel -> {share(pack_r(resource)) == (pack_r(PartitionLaws.first(~RModel,split_r(resource))),pack_r(PartitionLaws.second(~RModel,split_r(resource)))) : Resource & Resource},
  tasks: Pieces.Ranges,+enabled: Bool,+scope: Scope,+context: Context,+windows: Windows.WindowModel<Element>,+resource: RModel,
  +facts: TaskFacts(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,tasks,windows,resource)) ->
  {Pieces.grouped_with(~Move.Holding<Element>,~Output,~Resource,~Context,~share,~leaf,tasks,enabled,context,Windows.pack_windows(~Element,windows),pack_r(resource)) ==
    PartitionLaws.carry(~&2,~&1,~OModel,~Output,~Unit,~pack_o,results(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,scope,context,tasks,windows,resource,facts)) : Partition.Tree<&1,Output,Unit>}:
  walked(~Element,~Output,~Resource,~Context,~OModel,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Scope,~Property,~share_matches,tasks,enabled,enabled,scope,context,windows,resource,facts)
