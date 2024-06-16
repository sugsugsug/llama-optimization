# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch
import torch.nn.functional as F

#from llama.tokenizer import Tokenizer
from llama.model import Transformer

from torch.profiler import record_function

import os

class LLaMA:
    ii=0
    cache_tensor=0
    def __init__(self, model: Transformer):#, tokenizer: Tokenizer):
        self.model = model
        #self.tokenizer = tokenizer
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    def generate(
        self,
        prompts: List[int],
        prompt_answers: List[int], # L * 4
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[int]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = prompts #[self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        answer_tokens = prompt_answers #[self.tokenizer.encode(x, bos=False, eos=True) for x in prompt_answers]
        expanded_tokens = []
        for i, tok in enumerate(prompt_tokens):
            expanded_tokens.append(tok+answer_tokens[i])

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        #min_prompt_size = len(prompt_tokens[0])
        #max_prompt_size = len(prompt_tokens[0])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        result_prob = None
        tokens_with_answer = torch.zeros(1,1)
        max_tokens_with_answer = 0

        result_prob = torch.zeros(len(prompt_tokens)).to('cuda')
        tokens_with_answer = torch.full((bsz, total_len), 0).to('cuda').long()
        for k, t in enumerate(expanded_tokens):
            tokens_with_answer[k, : len(t)] = torch.tensor(t).long()
            if max_tokens_with_answer < len(t):
                max_tokens_with_answer = len(t)

        print(f'{self.local_rank} im here2')
        #logits = self.model.forward(tokens_with_answer[:, :max_tokens_with_answer], 0)
        if LLaMA.ii % 4 == 0:
            logits = self.model.forward(tokens_with_answer[:, :max_tokens_with_answer], 0)
        else:
            logits = self.model.forward(tokens_with_answer[:, min_prompt_size:max_tokens_with_answer], min_prompt_size)
        if self.local_rank in [0,1,2]:
            LLaMA.ii = LLaMA.ii + 1
            return None 

        probs = F.log_softmax(logits / temperature, dim=-1)
        if LLaMA.ii % 4 == 0:
            LLaMA.cache_tensor = torch.cat([probs[i:(i+1),len(prompt_tokens[i])-1:len(prompt_tokens[i]),:] for i in range(len(expanded_tokens))],dim=0)
        for i in range(len(expanded_tokens)):
            probs_for_token = probs[i,:,:]
            #mask = torch.arange(len(prompt_tokens[i])-1,len(expanded_tokens[i])-1)
            if LLaMA.ii % 4 == 0:
                mask = torch.arange(len(prompt_tokens[i])-1,len(expanded_tokens[i])-1)
            else:
                diff = len(prompt_tokens[i]) - min_prompt_size
                mask = torch.arange(diff, diff+len(answer_tokens[i]))
                probs_for_token = torch.cat((LLaMA.cache_tensor[i,:,:],probs_for_token[:-1,:]),dim=0)
            result_prob[i] = probs_for_token[mask,answer_tokens[i]].sum()
        LLaMA.ii = LLaMA.ii + 1
        return result_prob



def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
