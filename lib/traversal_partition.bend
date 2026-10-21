# A work partition is independent of its element representation and ownership.
# Storage remains in the leaf payload; offset/span describe its logical range.
import Base

type Tree<ownership,-Block: Kind(ownership),-Details: Data> is Kind(ownership):
  Leaf{data: Block,offset: U32,span: U32,details: Details}
  Fork{left: Tree<ownership,Block,Details>,right: Tree<ownership,Block,Details>}

def gather(~ownership: Quant,~Output: Kind(ownership),~copy: U32 -> U32 -> U32 -> U32 -> Output -> Output -> Output,
  partitions: Tree<ownership,Output,U32>,output: Output,+positions: U32) -> Output:
  match partitions:
    case Leaf{input,offset,span,rows}: copy(rows,offset,positions,span,input,output)
    case Fork{left,right}: gather(~ownership,~Output,~copy,right,gather(~ownership,~Output,~copy,left,output,positions),positions)
