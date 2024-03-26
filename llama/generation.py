# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch
import torch.nn.functional as F

from llama.tokenizer import Tokenizer
from llama.model import Transformer

from torch.profiler import record_function

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        prompt_answers: List[str], # L * 4
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[int]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        #print(prompt_tokens)
        answer_tokens = [self.tokenizer.encode(x, bos=False, eos=True) for x in prompt_answers]
        expanded_tokens = []
        for i, tok in enumerate(prompt_tokens):
            expanded_tokens.append(tok+answer_tokens[i])

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        #min_prompt_size = len(prompt_tokens[0])
        #max_prompt_size = len(prompt_tokens[0])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        result_prob = torch.zeros(len(prompt_tokens)).cuda()
        tokens_with_answer = torch.full((bsz, total_len), 0).cuda().long()
        max_tokens_with_answer = 0
        for k, t in enumerate(expanded_tokens):
            tokens_with_answer[k, : len(t)] = torch.tensor(t).long()
            if max_tokens_with_answer < len(t):
                max_tokens_with_answer = len(t)

        ##
        #print(tokens.shape)
        #tokens = tokens.repeat_interleave(repeats=4, dim=0)
        #print(len(expanded_tokens))
        input_text_mask_with_answer = tokens_with_answer != self.tokenizer.pad_id


        print("max ",max_tokens_with_answer)
        print("shape? ", tokens_with_answer.shape)
        #logits = self.model.forward(tokens_with_answer[:, :max_tokens_with_answer], 0)
        max_process = 4
        logit_array = []
        for i in range(max_tokens_with_answer//max_process):
            max_index = min((i+1)*max_process, max_tokens_with_answer)
            print("before: ",tokens_with_answer[:,i*max_process:max_index].shape)
            small_logits = self.model.forward(tokens_with_answer[:, i*max_process:max_index], i*max_process)
            #logit_array.append(small_logits)
            print("small ", small_logits.shape)
            print("done ",i*max_process)
        logits = torch.cat(logit_array, dim=1)
        print(logits.shape)

        probs = F.log_softmax(logits / temperature, dim=-1)
        for i in range(len(expanded_tokens)):
            probs_for_token = probs[i,max_tokens_with_answer*i:max_tokens_with_answer*(i+1),:]
            mask = torch.arange(len(prompt_tokens[i]),len(expanded_tokens[i]))
            result_prob[i] = probs_for_token[mask,answer_tokens[i]].sum()
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
