#!/bin/bash

hostname

torchrun --nproc_per_node 4 example.py --ckpt_dir ./ --tokenizer_path ./tokenizer.model

exit 0
