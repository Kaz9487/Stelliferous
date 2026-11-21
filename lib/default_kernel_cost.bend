# The prices of the reference machine, one thread each:
# scripts/toolchain_rules/planner_prices.py measures a row, a step, a copied
# byte and a moved cell, and scripts/toolchain_rules/wake_rounds.py checks
# that two wake rounds cover a measured region (rule H11).
import Base
import ./kernel_cost.bend as Cost

# A multiply-add row of eight lanes of the GEMM block.
def row_ns() -> F32:
  0.113

# One scalar step of a loop.
def step_ns() -> F32:
  0.122

# One byte a block copy writes.
def byte_ns() -> F32:
  0.036

# One cell moved through a view.
def move_ns() -> F32:
  5.9

# The least predicted work of a leaf of the Array engine.
def grain_ns() -> F32:
  250000.0

# One round of waking workers.
def wake_round_ns() -> F32:
  20000.0

# Dispatching one more task of a region, about 0.12 us at four threads.
def dispatch_ns() -> F32:
  120.0

def configuration() -> Cost.Configuration:
  Cost.Configuration{row_ns(),byte_ns(),step_ns(),move_ns(),grain_ns(),wake_round_ns(),dispatch_ns()}
