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

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

import jsonlines

from torch.profiler import profile, record_function, ProfilerActivity

from sentence_transformers import SentenceTransformer, util
import time
start_time = time.time()
sentences = ["I'm happy", "I'm full of happiness"]

sen_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

num_sample = 9996

def num_token(sentence: str):
    return len(sentence.split(' '))

def sen_cmp(sen0, sen1):
    #Compute embedding for both lists
    embedding_1= sen_model.encode(sen0, convert_to_tensor=True)
    embedding_2 = sen_model.encode(sen1, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2)[0][0]

def num_chr(sentence: str):
    return sentence.count('')-sentence.count(' ')-1

question_lines=[]
answer_lines=[]
answer_indexes=[]
answer_chr_nums = []

result_values = torch.empty((10000, 4), dtype=torch.float32)
sort_tuples = []

tokenizer_path = '/data/z0/heehoon/llama/llama_pretrained/tokenizer.model'
tokenizer_path = './tokenizer.model'
tokenizer = Tokenizer(model_path=tokenizer_path)
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
        question_token = tokenizer.encode(new_sen, bos=True, eos=False)
        answer_token_list = [ tokenizer.encode(new_ans, bos=False, eos=True) for new_ans in new_ans_list]
        answer_chr_num = [ num_chr(a) for a in new_ans_list]

        question_lines.append(question_token)
        answer_lines.append(answer_token_list)
        answer_indexes.append(new_ans_idx)
        answer_chr_nums.append(answer_chr_num)

        num_token_sen = len(question_token)
        answers_for_sort = []
        for j in range(4):
            answers_for_sort.append( (len(answer_token_list[j]), j ))
        sorted_answer_index_list = [ t[1] for t in sorted(answers_for_sort, key=lambda tup: tup[0])]
        #print(sorted_answer_index_list)

        for j in range(4):
            #sort_tuples.append((num_token_sen + num_token(new_ans_list[j]),real_i,j))
            sort_tuples.append((num_token_sen ,real_i,sorted_answer_index_list[j]))
        real_i = real_i + 1
        print(str(i)+'th input processing..')

new_tuples = sorted(sort_tuples, key=lambda tup: tup[0])
#new_tuples = sort_tuples

torch.manual_seed(1)

num_gpus = 4

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    print("Loading")
    device = torch.device("cuda")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    print(model_args)
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    for i, ckpt_path in enumerate(checkpoints):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        print(i)
    #model.to(device)
    #print(local_rank, model)

    generator = LLaMA(model)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 12,# 32
):
    generator = load(
        ckpt_dir, tokenizer_path, max_seq_len, max_batch_size
    )

    print(answer_lines[2])
    normalizing_token = ["Answer:", "Answer:","Answer:","Answer:"]
    results = []
    results_bf = []
    results2 = []
    num_batch = math.floor(len(question_lines)*4 / max_batch_size)
    print("HI",num_batch)
    """
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/before_pipeline'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
    prof.start()
    """
    # 0  4, 8, 
    # 1  4+1,8+1,
    # 2
    # 3
    # 12 16 20
    offset = 0
    for i in range(num_batch):
        gen_small = []
        question_for_gen = []
        answer_for_gen = []
        num_chr_list = []
        num_token_list = []
        for j in range(max_batch_size):
            a = (i//4)*max_batch_size*4 + (i%4)+j*4
            question = question_lines[new_tuples[a][1]]
            answer = answer_lines[new_tuples[a][1]][new_tuples[a][2]]
            ans_num_chr = answer_chr_nums[new_tuples[a][1]][new_tuples[a][2]]

            question_for_gen.append(question)
            answer_for_gen.append(answer)
            num_chr_list.append(ans_num_chr)
        gen_small = generator.generate(
            question_for_gen, answer_for_gen, max_gen_len=256, temperature=temperature, top_p=top_p 
        )
        #prof.step()
        #print('chr_list : ', num_chr_list)

        #print(gen_small)
        result_small = gen_small.div(torch.tensor(num_chr_list).to('cuda:3'))
        for j in range(max_batch_size):
            #a = i*max_batch_size+j
            a = (i//4)*max_batch_size*4 + (i%4)+j*4
            result_values[new_tuples[a][1],new_tuples[a][2]] = result_small[j]
        print(str(i)+'th batch processing..')

    #prof.stop()
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
