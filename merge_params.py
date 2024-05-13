from llama import ModelArgs, Transformer, LLaMA
from pathlib import Path
import torch
import json
import os
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

ckpt_dir = '/data/z0/heehoon/llama/llama_pretrained/30B'
#ckpt_dir = './'
world_size = 4
max_seq_len = 512
max_batch_size = 12


def load(
) -> LLaMA:
    #torch.distributed.init_process_group("nccl")
    #initialize_model_parallel(world_size)
    #local_rank = int(os.environ.get("LOCAL_RANK", -1))
    #if local_rank > 0:
    #    return 0
    checkpoints = sorted(Path('./').glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    #torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.set_default_tensor_type(torch.FloatTensor)
    model_args.vocab_size = 32000
    print(model_args)
    model = Transformer(model_args)
    for i in range(world_size):
        ckpt_path = checkpoints[i]
        print("Loading ", i)
        model.load_state_dict(torch.load(ckpt_path, map_location="cuda:"+str(i)), strict=False)
    model.to(torch.device("cuda"))
    for name, param in model.named_parameters():
        print(name, param.shape)
    generator = LLaMA(model)
    return generator

if __name__ == "__main__":
    load()


