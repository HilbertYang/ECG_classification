#!/bin/bash
srun --partition=gpu \
     --gres=gpu:1 \
     --cpus-per-task=4 \
     --mem=32G \
     --time=01:00:00 \
     --pty bash