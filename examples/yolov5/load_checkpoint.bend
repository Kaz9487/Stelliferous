# Loads the official checkpoint in Bend and writes every tensor out as a .npy
# file under its PyTorch name, for the comparison with torch.load, then the
# same tensors with the batch normalizations folded, for the comparison with
# the fused PyTorch model.
import Base
import ../../lib/tensor_f32.bend as F
import ./batch_norm.bend as BatchNorm

def text(result: Result<&1,&1,U32 & String,String>,fallback: String) -> String:
  match result:
    case Done{value}: value
    case Fail{error}: fallback

def variable(+name: String,fallback: String) -> IO(String):
  do IO<String>:
    result : Result<&1,&1,U32 & String,String> <- IO.get_env(name)
    IO.pure(String,text(result,fallback))

# One timed load.
def loaded(+path: String) -> IO(F.Weights()):
  do IO<F.Weights()>:
    start : Nat <- IO.now()
    weights : F.Weights() <- F.load_checkpoint(path)
    end : Nat <- IO.now()
    IO.print("load " ++ Nat.show(Nat.sub(end,start)) ++ " ms")
    IO.pure(F.Weights(),weights)

# CHECKPOINT: the .pt file. CHECKPOINT_OUTPUT and CHECKPOINT_FOLDED: the
# folders for names.txt and <name>.npy.
def main() -> IO(Unit):
  do IO<Unit>:
    path : String <- variable("CHECKPOINT","data/weights/yolov5n-v7.0.pt")
    output : String <- variable("CHECKPOINT_OUTPUT","out/results/checkpoint/bend")
    folded : String <- variable("CHECKPOINT_FOLDED","out/results/checkpoint/folded")
    weights : F.Weights() <- loaded(path)
    saved : F.Weights() <- F.save_weights(output,weights)
    saved : F.Weights() <- F.save_weights(folded,BatchNorm.folded_weights(F.run_keeping(Unit,BatchNorm.fold_all(),saved)))
    IO.pure(Unit,Unit{})
