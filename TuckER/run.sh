#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

cmd="python3.8 main.py --dataset FB60K-NYT10 --num_iterations 500 --batch_size 256 --lr 0.003 --dr 0.99 --edim 256 --rdim 256 --input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1"

echo "Executing $cmd"

$cmd
