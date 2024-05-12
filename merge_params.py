from llama import ModelArgs, Transformer, LLaMA
from pathlib import Path
import torch
import json
import os
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

ckpt_dir = '/data/z0/heehoon/llama/llama_pretrained/30B'
world_size = 4
max_seq_len = 512
max_batch_size = 12


def load(
) -> LLaMA:
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank > 0:
        return 0
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_values = []
    for i in range(world_size):
        ckpt_path = checkpoints[i]
        print("Loading ", i)
        ckpt_values.append(torch.load(ckpt_path, map_location="cpu"))
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    #torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_args.vocab_size = 32000
    print(model_args)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    for i in range(world_size):
        model.load_state_dict(ckpt_values[i], strict=False)
        print(i)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    print(model)
    for name, param in model.named_parameters():
        print(name, param.shape)

    generator = LLaMA(model)
    return generator

if __name__ == "__main__":
    load()


