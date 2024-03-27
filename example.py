# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import math

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

import jsonlines

from torch.profiler import profile, record_function, ProfilerActivity

from sentence_transformers import SentenceTransformer, util
import time
start_time = time.time()
sentences = ["I'm happy", "I'm full of happiness"]

sen_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

num_sample = 10000

def num_token(sentence: str):
    return len(sentence.split(' '))

def sen_cmp(sen0, sen1):
    #Compute embedding for both lists
    embedding_1= sen_model.encode(sen0, convert_to_tensor=True)
    embedding_2 = sen_model.encode(sen1, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2)[0][0]

question_lines=[]
answer_lines=[]
answer_indexes=[]

result_values = torch.empty((10000, 4), dtype=torch.float32)
sort_tuples = []

real_i = 0
num_skip_sample = 0
with jsonlines.open("hellaswag_val.jsonl") as f:
    for i, line in enumerate(f.iter()):
        if i >= num_sample+num_skip_sample:
            break
        if i < num_skip_sample:
            continue
        try:
            new_sen = line['ctx']
            new_ans_list = line['endings']
            new_ans_idx = line['label']
        except KeyError as e:
            print('error in ' + str(i)) 
            continue
        question_lines.append(new_sen)
        answer_lines.append(new_ans_list)
        answer_indexes.append(new_ans_idx)

        num_token_sen = num_token(new_sen)
        for j in range(4):
            sort_tuples.append((num_token_sen + num_token(new_ans_list[j]),real_i,j))
        real_i = real_i + 1
        print(str(i)+'th input processing..')

new_tuples = sorted(sort_tuples, key=lambda tup: tup[0])



def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def num_chr(sentence: str):
    return sentence.count('')-sentence.count(' ')-1


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,# 32
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]
    '''
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )
    '''
    print(answer_lines[2])
    normalizing_token = ["Answer:", "Answer:","Answer:","Answer:"]
    results = []
    results_bf = []
    results2 = []
    num_batch = math.floor(len(question_lines)*4 / max_batch_size)
    print("HI",num_batch)
    for i in range(num_batch):
        gen_small = []
        question_for_gen = []
        answer_for_gen = []
        num_chr_list = []
        num_token_list = []
        for j in range(max_batch_size):
            a = i*max_batch_size+j
            question = question_lines[new_tuples[a][1]]
            answer = answer_lines[new_tuples[a][1]][new_tuples[a][2]]

            question_for_gen.append(question)
            answer_for_gen.append(answer)
            num_chr_list.append(num_chr(answer))
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        #    with record_function("model_inference"):
        gen_small = generator.generate(
            question_for_gen, answer_for_gen, max_gen_len=256, temperature=temperature, top_p=top_p
        )
        #print(prof.key_averages().table(sort_by="cuda_time_total",row_limit=20))
        print('chr_list : ', num_chr_list)

        result_small = gen_small.div(torch.tensor(num_chr_list).cuda())
        print(result_small)
        for j in range(max_batch_size):
            a = i*max_batch_size+j
            result_values[new_tuples[a][1],new_tuples[a][2]] = result_small[j]
        print(str(i)+'th batch processing..')

    results = torch.argmax(result_values[:num_sample,:],dim=1).tolist()
    print(results)
    num_correct = 0
    num_total = 0
    for i, result in enumerate(results):
        print(answer_indexes[i])
        if answer_indexes[i] == result:
            num_correct = num_correct + 1
        num_total = num_total + 1
        print(str(i)+'th cmp processing..')

    print(num_correct/num_total * 100)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    fire.Fire(main)
