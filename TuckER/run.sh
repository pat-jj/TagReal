#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

cmd="python3.6 main.py --dataset Wiki27K --num_iterations 500 --batch_size 256 --lr 0.0005 --dr 1.0 --edim 256 --rdim 256 --input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1 --model tucker"

echo "Executing $cmd"

$cmd
