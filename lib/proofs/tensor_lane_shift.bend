# Lanes of cells joined at a shift and of one cell put over another between
# two lanes.

import Base
import ../kernel_tile.bend as Tile
import ../tensor_tiled.bend as Tiled
import ../math_choice.bend as Choice
import ./kernel_tile_refinement.bend as TileLaws
import ./word_encoding.bend as WordEncoding

# ---- Joined cells ----

# Lane lane of a cell joined at a shift: lane lane + shift of the low cell,
# or of the high cell past the low one.
def joined_value(+shift: Nat,+lane: Nat,+low: Tile.Tile8,+high: Tile.Tile8) -> F32:
  +moved = Nat.add(lane,shift)
  Bool.pick(F32,Nat.is_lt(moved,8n),TileLaws.select_lane(moved,low),TileLaws.select_lane(Nat.sub(moved,8n),high))

def joined_at_0(lane: Nat,+low: Tile.Tile8,+high: Tile.Tile8,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.joined(0,low,high)) == joined_value(0n,lane,low,high) : F32}:
  match lane:
    case 0n:
      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
      {==}
    case 1n+ +next_1:
      match next_1:
        case 0n:
          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
          {==}
        case 1n+ +next_2:
          match next_2:
            case 0n:
              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
              {==}
            case 1n+ +next_3:
              match next_3:
                case 0n:
                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                  {==}
                case 1n+ +next_4:
                  match next_4:
                    case 0n:
                      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                      {==}
                    case 1n+ +next_5:
                      match next_5:
                        case 0n:
                          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                          {==}
                        case 1n+ +next_6:
                          match next_6:
                            case 0n:
                              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                              {==}
                            case 1n+ +next_7:
                              match next_7:
                                case 0n:
                                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                                  {==}
                                case 1n+beyond:
                                  Empty.absurd({TileLaws.select_lane(8n+beyond,Tiled.joined(0,low,high)) == joined_value(0n,8n+beyond,low,high) : F32},WordEncoding.false_true(small))

def joined_at_1(lane: Nat,+low: Tile.Tile8,+high: Tile.Tile8,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.joined(1,low,high)) == joined_value(1n,lane,low,high) : F32}:
  match lane:
    case 0n:
      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
      {==}
    case 1n+ +next_1:
      match next_1:
        case 0n:
          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
          {==}
        case 1n+ +next_2:
          match next_2:
            case 0n:
              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
              {==}
            case 1n+ +next_3:
              match next_3:
                case 0n:
                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                  {==}
                case 1n+ +next_4:
                  match next_4:
                    case 0n:
                      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                      {==}
                    case 1n+ +next_5:
                      match next_5:
                        case 0n:
                          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                          {==}
                        case 1n+ +next_6:
                          match next_6:
                            case 0n:
                              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                              {==}
                            case 1n+ +next_7:
                              match next_7:
                                case 0n:
                                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                                  {==}
                                case 1n+beyond:
                                  Empty.absurd({TileLaws.select_lane(8n+beyond,Tiled.joined(1,low,high)) == joined_value(1n,8n+beyond,low,high) : F32},WordEncoding.false_true(small))

def joined_at_2(lane: Nat,+low: Tile.Tile8,+high: Tile.Tile8,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.joined(2,low,high)) == joined_value(2n,lane,low,high) : F32}:
  match lane:
    case 0n:
      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
      {==}
    case 1n+ +next_1:
      match next_1:
        case 0n:
          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
          {==}
        case 1n+ +next_2:
          match next_2:
            case 0n:
              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
              {==}
            case 1n+ +next_3:
              match next_3:
                case 0n:
                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                  {==}
                case 1n+ +next_4:
                  match next_4:
                    case 0n:
                      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                      {==}
                    case 1n+ +next_5:
                      match next_5:
                        case 0n:
                          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                          {==}
                        case 1n+ +next_6:
                          match next_6:
                            case 0n:
                              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                              {==}
                            case 1n+ +next_7:
                              match next_7:
                                case 0n:
                                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                                  {==}
                                case 1n+beyond:
                                  Empty.absurd({TileLaws.select_lane(8n+beyond,Tiled.joined(2,low,high)) == joined_value(2n,8n+beyond,low,high) : F32},WordEncoding.false_true(small))

def joined_at_3(lane: Nat,+low: Tile.Tile8,+high: Tile.Tile8,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.joined(3,low,high)) == joined_value(3n,lane,low,high) : F32}:
  match lane:
    case 0n:
      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
      {==}
    case 1n+ +next_1:
      match next_1:
        case 0n:
          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
          {==}
        case 1n+ +next_2:
          match next_2:
            case 0n:
              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
              {==}
            case 1n+ +next_3:
              match next_3:
                case 0n:
                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                  {==}
                case 1n+ +next_4:
                  match next_4:
                    case 0n:
                      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                      {==}
                    case 1n+ +next_5:
                      match next_5:
                        case 0n:
                          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                          {==}
                        case 1n+ +next_6:
                          match next_6:
                            case 0n:
                              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                              {==}
                            case 1n+ +next_7:
                              match next_7:
                                case 0n:
                                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                                  {==}
                                case 1n+beyond:
                                  Empty.absurd({TileLaws.select_lane(8n+beyond,Tiled.joined(3,low,high)) == joined_value(3n,8n+beyond,low,high) : F32},WordEncoding.false_true(small))

def joined_at_4(lane: Nat,+low: Tile.Tile8,+high: Tile.Tile8,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.joined(4,low,high)) == joined_value(4n,lane,low,high) : F32}:
  match lane:
    case 0n:
      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
      {==}
    case 1n+ +next_1:
      match next_1:
        case 0n:
          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
          {==}
        case 1n+ +next_2:
          match next_2:
            case 0n:
              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
              {==}
            case 1n+ +next_3:
              match next_3:
                case 0n:
                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                  {==}
                case 1n+ +next_4:
                  match next_4:
                    case 0n:
                      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                      {==}
                    case 1n+ +next_5:
                      match next_5:
                        case 0n:
                          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                          {==}
                        case 1n+ +next_6:
                          match next_6:
                            case 0n:
                              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                              {==}
                            case 1n+ +next_7:
                              match next_7:
                                case 0n:
                                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                                  {==}
                                case 1n+beyond:
                                  Empty.absurd({TileLaws.select_lane(8n+beyond,Tiled.joined(4,low,high)) == joined_value(4n,8n+beyond,low,high) : F32},WordEncoding.false_true(small))

def joined_at_5(lane: Nat,+low: Tile.Tile8,+high: Tile.Tile8,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.joined(5,low,high)) == joined_value(5n,lane,low,high) : F32}:
  match lane:
    case 0n:
      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
      {==}
    case 1n+ +next_1:
      match next_1:
        case 0n:
          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
          {==}
        case 1n+ +next_2:
          match next_2:
            case 0n:
              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
              {==}
            case 1n+ +next_3:
              match next_3:
                case 0n:
                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                  {==}
                case 1n+ +next_4:
                  match next_4:
                    case 0n:
                      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                      {==}
                    case 1n+ +next_5:
                      match next_5:
                        case 0n:
                          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                          {==}
                        case 1n+ +next_6:
                          match next_6:
                            case 0n:
                              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                              {==}
                            case 1n+ +next_7:
                              match next_7:
                                case 0n:
                                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                                  {==}
                                case 1n+beyond:
                                  Empty.absurd({TileLaws.select_lane(8n+beyond,Tiled.joined(5,low,high)) == joined_value(5n,8n+beyond,low,high) : F32},WordEncoding.false_true(small))

def joined_at_6(lane: Nat,+low: Tile.Tile8,+high: Tile.Tile8,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.joined(6,low,high)) == joined_value(6n,lane,low,high) : F32}:
  match lane:
    case 0n:
      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
      {==}
    case 1n+ +next_1:
      match next_1:
        case 0n:
          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
          {==}
        case 1n+ +next_2:
          match next_2:
            case 0n:
              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
              {==}
            case 1n+ +next_3:
              match next_3:
                case 0n:
                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                  {==}
                case 1n+ +next_4:
                  match next_4:
                    case 0n:
                      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                      {==}
                    case 1n+ +next_5:
                      match next_5:
                        case 0n:
                          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                          {==}
                        case 1n+ +next_6:
                          match next_6:
                            case 0n:
                              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                              {==}
                            case 1n+ +next_7:
                              match next_7:
                                case 0n:
                                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                                  {==}
                                case 1n+beyond:
                                  Empty.absurd({TileLaws.select_lane(8n+beyond,Tiled.joined(6,low,high)) == joined_value(6n,8n+beyond,low,high) : F32},WordEncoding.false_true(small))

def joined_at_7(lane: Nat,+low: Tile.Tile8,+high: Tile.Tile8,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.joined(7,low,high)) == joined_value(7n,lane,low,high) : F32}:
  match lane:
    case 0n:
      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
      {==}
    case 1n+ +next_1:
      match next_1:
        case 0n:
          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
          {==}
        case 1n+ +next_2:
          match next_2:
            case 0n:
              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
              {==}
            case 1n+ +next_3:
              match next_3:
                case 0n:
                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                  {==}
                case 1n+ +next_4:
                  match next_4:
                    case 0n:
                      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                      Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                      {==}
                    case 1n+ +next_5:
                      match next_5:
                        case 0n:
                          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                          Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                          {==}
                        case 1n+ +next_6:
                          match next_6:
                            case 0n:
                              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                              Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                              {==}
                            case 1n+ +next_7:
                              match next_7:
                                case 0n:
                                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
                                  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
                                  {==}
                                case 1n+beyond:
                                  Empty.absurd({TileLaws.select_lane(8n+beyond,Tiled.joined(7,low,high)) == joined_value(7n,8n+beyond,low,high) : F32},WordEncoding.false_true(small))

# Lane lane of a cell joined at a shift below eight.
def joined_lane(shift: Nat,+lane: Nat,+low: Tile.Tile8,+high: Tile.Tile8,+shift_small: {Nat.is_le(shift,7n) == True{} : Bool},+lane_small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),low,high)) == joined_value(shift,lane,low,high) : F32}:
  match shift:
    case 0n:
      joined_at_0(lane,low,high,lane_small)
    case 1n+ +more_1:
      match more_1:
        case 0n:
          joined_at_1(lane,low,high,lane_small)
        case 1n+ +more_2:
          match more_2:
            case 0n:
              joined_at_2(lane,low,high,lane_small)
            case 1n+ +more_3:
              match more_3:
                case 0n:
                  joined_at_3(lane,low,high,lane_small)
                case 1n+ +more_4:
                  match more_4:
                    case 0n:
                      joined_at_4(lane,low,high,lane_small)
                    case 1n+ +more_5:
                      match more_5:
                        case 0n:
                          joined_at_5(lane,low,high,lane_small)
                        case 1n+ +more_6:
                          match more_6:
                            case 0n:
                              joined_at_6(lane,low,high,lane_small)
                            case 1n+ +more_7:
                              match more_7:
                                case 0n:
                                  joined_at_7(lane,low,high,lane_small)
                                case 1n+beyond:
                                  Empty.absurd({TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(8n+beyond),low,high)) == joined_value(8n+beyond,lane,low,high) : F32},WordEncoding.false_true(shift_small))

# ---- Lanes between two bounds ----

# Lane lane of the lanes of inside whose channel, base plus the lane, lies
# from start below end, the others of outside.
def between_value(+start: U32,+end: U32,+base: U32,+lane: Nat,+inside: Tile.Tile8,+outside: Tile.Tile8) -> F32:
  Bool.pick(F32,U32.is_le(start,(base + U32.from_nat(lane) : U32)) && U32.is_lt((base + U32.from_nat(lane) : U32),end),TileLaws.select_lane(lane,inside),TileLaws.select_lane(lane,outside))

def chosen_pick(flag: Bool,+yes: F32,+no: F32) -> {Choice.chosen(~F32,flag,yes,no) == Bool.pick(F32,flag,yes,no) : F32}:
  match flag:
    case True{}: {==}
    case False{}: {==}

def between_lane(lane: Nat,+start: U32,+end: U32,+base: U32,+inside: Tile.Tile8,+outside: Tile.Tile8,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.lanes_between(start,end,base,inside,outside)) == between_value(start,end,base,lane,inside,outside) : F32}:
  match lane:
    case 0n:
      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = inside
      Tile.Tile{k0,k1,k2,k3,k4,k5,k6,k7} = outside
      chosen_pick(U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0)
    case 1n+ +next_1:
      match next_1:
        case 0n:
          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = inside
          Tile.Tile{k0,k1,k2,k3,k4,k5,k6,k7} = outside
          chosen_pick(U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1)
        case 1n+ +next_2:
          match next_2:
            case 0n:
              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = inside
              Tile.Tile{k0,k1,k2,k3,k4,k5,k6,k7} = outside
              chosen_pick(U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2)
            case 1n+ +next_3:
              match next_3:
                case 0n:
                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = inside
                  Tile.Tile{k0,k1,k2,k3,k4,k5,k6,k7} = outside
                  chosen_pick(U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3)
                case 1n+ +next_4:
                  match next_4:
                    case 0n:
                      Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = inside
                      Tile.Tile{k0,k1,k2,k3,k4,k5,k6,k7} = outside
                      chosen_pick(U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4)
                    case 1n+ +next_5:
                      match next_5:
                        case 0n:
                          Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = inside
                          Tile.Tile{k0,k1,k2,k3,k4,k5,k6,k7} = outside
                          chosen_pick(U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5)
                        case 1n+ +next_6:
                          match next_6:
                            case 0n:
                              Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = inside
                              Tile.Tile{k0,k1,k2,k3,k4,k5,k6,k7} = outside
                              chosen_pick(U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6)
                            case 1n+ +next_7:
                              match next_7:
                                case 0n:
                                  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = inside
                                  Tile.Tile{k0,k1,k2,k3,k4,k5,k6,k7} = outside
                                  chosen_pick(U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)
                                case 1n+beyond:
                                  Empty.absurd({TileLaws.select_lane(8n+beyond,Tiled.lanes_between(start,end,base,inside,outside)) == between_value(start,end,base,8n+beyond,inside,outside) : F32},WordEncoding.false_true(small))

def chosen_again(flag: Bool,+yes: F32,+no: F32) -> {Choice.chosen(~F32,flag,yes,Choice.chosen(~F32,flag,yes,no)) == Choice.chosen(~F32,flag,yes,no) : F32}:
  match flag:
    case True{}: {==}
    case False{}: {==}

# Putting the same lanes over a cell twice puts them once.
def between_again(+start: U32,+end: U32,+base: U32,+inside: Tile.Tile8,+outside: Tile.Tile8) ->
  {Tiled.lanes_between(start,end,base,inside,Tiled.lanes_between(start,end,base,inside,outside)) == Tiled.lanes_between(start,end,base,inside,outside) : Tile.Tile8}:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = inside
  Tile.Tile{k0,k1,k2,k3,k4,k5,k6,k7} = outside
  %Equal.sym(F32,Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0)),Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),chosen_again(U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0)) :
    {Tile.Tile{_,Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1)),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2)),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3)),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4)),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5)),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6)),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7))} == Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)} : Tile.Tile8}
  %Equal.sym(F32,Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1)),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),chosen_again(U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1)) :
    {Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),_,Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2)),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3)),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4)),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5)),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6)),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7))} == Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)} : Tile.Tile8}
  %Equal.sym(F32,Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2)),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),chosen_again(U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2)) :
    {Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),_,Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3)),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4)),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5)),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6)),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7))} == Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)} : Tile.Tile8}
  %Equal.sym(F32,Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3)),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),chosen_again(U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3)) :
    {Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),_,Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4)),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5)),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6)),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7))} == Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)} : Tile.Tile8}
  %Equal.sym(F32,Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4)),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),chosen_again(U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4)) :
    {Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),_,Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5)),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6)),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7))} == Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)} : Tile.Tile8}
  %Equal.sym(F32,Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5)),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),chosen_again(U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5)) :
    {Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),_,Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6)),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7))} == Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)} : Tile.Tile8}
  %Equal.sym(F32,Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6)),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6),chosen_again(U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6)) :
    {Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),_,Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7))} == Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)} : Tile.Tile8}
  %Equal.sym(F32,Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7),chosen_again(U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)) :
    {Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6),_} == Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6),Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)} : Tile.Tile8}
  {==}
