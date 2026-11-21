# Machine limits carry binary safety evidence. Nat proofs quantify over Limits
# rather than specializing their types to a large unary bound.
import Base

def word_max() -> U32:
  4294967295

type Facts<-d: U32,-c: U32,-k: U32,-h: U32> is Data:
  Facts{
    dimension_fits: {U32.is_le(d,2147483648) == True{} : Bool},
    capacity_fits: {U32.is_le(c,2147483648) == True{} : Bool},
    coordinate_fits: {U32.is_le(k,2147483648) == True{} : Bool},
    half_word_fits: {U32.is_le(h,2147483648) == True{} : Bool},
    dimension_allocatable: {U32.is_le(d,1073741824) == True{} : Bool},
    capacity_allocatable: {U32.is_le(c,1073741824) == True{} : Bool},
    dimension_below_half: {U32.is_lt(d,2147483648) == True{} : Bool},
    capacity_below_half: {U32.is_lt(c,2147483648) == True{} : Bool},
    dimension_within_capacity: {U32.is_le(d,c) == True{} : Bool},
    dimension_within_coordinate: {U32.is_le(d,k) == True{} : Bool},
    doubled_dimension_within_coordinate: {U32.is_le(U32.shl(d),k) == True{} : Bool},
    capacity_doubles_to_half: {U32.shl(c) == h : U32},
    sixteen_within_capacity: {U32.is_le(16,c) == True{} : Bool},
    sixteen_within_half: {U32.is_le(16,h) == True{} : Bool},
    capacity_within_half: {U32.is_le(c,h) == True{} : Bool}
  }

type Limits is Data:
  Limits{dimension: U32,capacity: U32,coordinate: U32,half_word: U32,facts: Facts<dimension,capacity,coordinate,half_word>}

def dimension(limits: Limits) -> U32:
  Limits{d,c,k,h,facts} = limits
  d
def capacity(limits: Limits) -> U32:
  Limits{d,c,k,h,facts} = limits
  c
def coordinate(limits: Limits) -> U32:
  Limits{d,c,k,h,facts} = limits
  k
def half_word(limits: Limits) -> U32:
  Limits{d,c,k,h,facts} = limits
  h
