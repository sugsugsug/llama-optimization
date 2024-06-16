# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import os
import time

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads #// fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.cache_k = nn.Parameter(torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        #).cuda()
        ))
        self.cache_v = nn.Parameter(torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        #).cuda()
        ))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        #self.cache_k = self.cache_k.to(xq)
        #self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class CircularQueue:
    def __init__(self, size):
        self.size = size
        self.queue = [None] * size
        self.front = self.rear = -1

    def is_full(self):
        return (self.rear + 1) % self.size == self.front

    def is_empty(self):
        return self.front == -1

    def enqueue(self, data):
        if self.is_full():
            print("Queue is full")
            return

        if self.front == -1:
            self.front = 0

        self.rear = (self.rear + 1) % self.size
        self.queue[self.rear] = data

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty")
            return None

        data = self.queue[self.front]
        self.queue[self.front] = None

        if self.front == self.rear:
            self.front = self.rear = -1
        else:
            self.front = (self.front + 1) % self.size

        return data

    def display(self):
        if self.is_empty():
            print("Queue is empty")
            return

        if self.rear >= self.front:
            print("Queue:", ' '.join([str(self.queue[i]) for i in range(self.front, self.rear + 1)]))
        else:
            print("Queue:", ' '.join([str(self.queue[i]) for i in range(self.front, self.size)] + [str(self.queue[i]) for i in range(0, self.rear + 1)]))



class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))

        self.layers = torch.nn.ModuleList()
        if self.local_rank == 0:
            self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        for layer_id in range(params.n_layers//4):
            if self.local_rank == 0:
                self.layers.append(TransformerBlock(layer_id, params))
            else:
                self.layers.append(nn.Identity())
        for layer_id in range(params.n_layers//4, params.n_layers//4 * 2):
            if self.local_rank == 1:
                self.layers.append(TransformerBlock(layer_id, params))
            else: 
                self.layers.append(nn.Identity())
        for layer_id in range(params.n_layers//4 * 2, params.n_layers//4 * 3):
            if self.local_rank == 2:
                self.layers.append(TransformerBlock(layer_id, params))
            else: 
                self.layers.append(nn.Identity())
        for layer_id in range(params.n_layers//4 * 3, params.n_layers):
            if self.local_rank == 3:
                self.layers.append(TransformerBlock(layer_id, params))
            else: 
                self.layers.append(nn.Identity())

        if self.local_rank == 3:
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)
            self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.tag = 0
        self.total_wait = 0
        #self.tensor_ref = CircularQueue(100)
        self.tensor_ref = []

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        h = None
        local_rank = self.local_rank
        _bsz, seqlen = tokens.shape 
        # Get tensor from previous GPU
        tensor = None
        if local_rank > 0:
            #tensor = torch.zeros((_bsz, seqlen, self.params.dim), dtype=torch.cuda.HalfTensor).to(local_rank)
            tensor = torch.zeros((_bsz, seqlen, self.params.dim), dtype=torch.float16, device=torch.device('cuda'))
            req = dist.irecv(tensor=tensor, src=local_rank-1, tag=self.tag)
            start = time.time()
            req.wait()
            self.total_wait = time.time() - start
            print(f'TOTAL_WAIT{local_rank}: {self.total_wait} sec in tag{self.tag} ')
            '''
            print(f'TOTAL_WAIT{local_rank}: {self.total_wait} sec in tag{self.tag}, got{tensor[0,0,0]}')
            if self.tag != tensor[0,0,0]:
                print('NOT MATCH CODE: UGYEONG')
            '''
            h = tensor
            print(local_rank, tensor.shape)


        if local_rank == 0:
            h = self.tok_embeddings(tokens)

        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if start_pos != -1:
            mask = torch.full((1, 1, seqlen, start_pos+seqlen), float("-inf"), device='cuda')
            mask = torch.triu(mask, diagonal=start_pos + 1)
            mask = mask.type_as(h if local_rank==0 else tensor)

        # Process
        for i, layer in enumerate(self.layers):
            if i < local_rank*15:
                continue
            if i >= (local_rank+1)*15:
                break

            h = layer(h, start_pos, freqs_cis, mask)


        # Send
        if local_rank in [0,1,2]:
            tensor = h
            print(local_rank, tensor.shape)
            req = dist.isend(tensor=tensor, dst=local_rank+1, tag=self.tag)

            self.tag += 1
            return None
        elif local_rank == 3:
            h = self.norm(h)
            output = self.output(h[:, :, :])  
            self.tag += 1
            return output.float()
        else:
            raise Exception('not allowed local_rank')
