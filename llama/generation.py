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
        prompt_answers: List[List[str]], # L * 4
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[int]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        #print(prompt_tokens)
        answer_tokens = [[self.tokenizer.encode(x, bos=False, eos=True) for x in one_question] for one_question in prompt_answers]
        expanded_tokens = []
        with record_function("make_token"):
            for i, tok in enumerate(prompt_tokens):
                for j in range(4):
                    expanded_tokens.append(tok+answer_tokens[i][j])
        #print(expanded_tokens)

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        #min_prompt_size = len(prompt_tokens[0])
        #max_prompt_size = len(prompt_tokens[0])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        result_prob = torch.zeros(len(prompt_tokens)*4).cuda()
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        tokens_with_answer = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        tokens_with_answer = tokens_with_answer.repeat(4,1)
        with record_function("make_token2"):
            for k, t in enumerate(prompt_tokens):
                tokens[k, : len(t)] = torch.tensor(t).long()
            for k, t in enumerate(expanded_tokens):
                tokens_with_answer[k, : len(t)] = torch.tensor(t).long()
        ##
        #print(tokens.shape)
        tokens = tokens.repeat_interleave(repeats=4, dim=0)
        #print(len(expanded_tokens))
        tokens = tokens[:16,:]
        expanded_tokens = expanded_tokens[:16]

        tokens_with_answer = tokens_with_answer[:16,:]

        input_text_mask = tokens != self.tokenizer.pad_id
        input_text_mask_with_answer = tokens_with_answer != self.tokenizer.pad_id

        start_pos = min_prompt_size
        prev_pos = 0

        zero_dummy = torch.zeros(input_text_mask_with_answer.shape[0]).cuda()

        for cur_pos in range(start_pos, total_len):
            with record_function("forward_real_Model"):
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
                print(logits.shape)
            if temperature > 0:
                # next_token = sample_top_p(probs, top_p)
                #print(probs.shape)
          
                ls = []
                with record_function("append_token"):
                    for i in range(len(expanded_tokens)):
                        if cur_pos < len(expanded_tokens[i]):
                            ls.append(expanded_tokens[i][cur_pos])
                        else:
                            ls.append(0)

                probs = F.log_softmax(logits / temperature, dim=-1)
                token_prob = probs[torch.arange(probs.size(0)), ls]
                
                token_prob = torch.where(input_text_mask_with_answer[:,cur_pos], token_prob, zero_dummy)
                
                result_prob = torch.add(result_prob, token_prob)
                
                #print(input_text_mask_with_answer[:,cur_pos])
                #print(token_prob)
                #print(result_prob)
                if not torch.is_nonzero(torch.count_nonzero(token_prob)):
                    break

                next_token = torch.tensor(ls).cuda()
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            #print(tokens[:,cur_pos-2:cur_pos+2])
            prev_pos = cur_pos

        '''
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
    
        return decoded
        '''
        
        return result_prob
        #result = torch.argmax(result_prob.reshape((-1,4)), dim=1)
        #return result



def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
