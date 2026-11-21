# Real compact addresses and padding decisions, derived from prepare and guards.
import Base
import ../specification.bend as ConvSpec

def spatial(+problem: ConvSpec.Problem) -> Nat:
  Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))

