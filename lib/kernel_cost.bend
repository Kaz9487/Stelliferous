# Prediction parameters, not machine bounds or correctness certificates. The
# worker count is not a profile value: callers pass the runtime's pool size.
import Base

# The prices a plan is estimated with. A cell counts rows, vector
# instructions of eight lanes with its tile loads and stores among them, and
# steps, scalar instructions. A target cell of a moving operation also pays
# one move through a view, and a block copy pays its bytes.
type Configuration is Data:
  Configuration{row_ns: F32,byte_ns: F32,step_ns: F32,move_ns: F32,grain_ns: F32,wake_round_ns: F32,dispatch_ns: F32}

# One cell's work from its counts.
def cell_ns(+rows: F32,+steps: F32,config: Configuration) -> F32:
  Configuration{row_ns,byte_ns,step_ns,move_ns,grain,wake,dispatch} = config
  (rows * row_ns + steps * step_ns : F32)

# One target cell of a moving operation: its move and its own rows and steps.
def moved_ns(+rows: F32,+steps: F32,config: Configuration) -> F32:
  Configuration{row_ns,byte_ns,step_ns,move_ns,grain,wake,dispatch} = config
  (move_ns + rows * row_ns + steps * step_ns : F32)

# The bytes a block copy writes into a new buffer.
def copy_ns(+bytes: F32,config: Configuration) -> F32:
  Configuration{row_ns,byte_ns,step_ns,move_ns,grain,wake,dispatch} = config
  (bytes * byte_ns : F32)

# The official runtime grows fork tasks and queues fork-free tasks for its
# work round (monk_step, row_grow, cube_run). No fixed number of leaves can
# be subtracted as already completed by growth. Price complete waves of at
# most workers leaves, using the largest path (including clones) per wave.
# This is a conservative scheduling estimate, not a bound on the runtime's
# static ring distribution. Callers charge the two wake rounds separately.
def region_ns(+workers: U32,+leaves: U32,+span: F32) -> F32:
  +pool = U32.max(workers,1)
  +waves = (leaves / pool + Bool.pick(U32,U32.is_zero((leaves % pool : U32)),0,1) : U32)
  (span * U32.to_f32(waves) : F32)
