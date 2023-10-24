#!/bin/bash

# ./script/command.sh ressl-cifa10               1 1 "python -u ressl.py --dataset cifar10      --k 4096  --m 0.99 "
# ./script/command.sh ressl-cifar100             1 1 "python -u ressl.py --dataset cifar100     --k 4096  --m 0.99 "
# ./script/command.sh ressl-stl10                1 1 "python -u ressl.py --dataset stl10        --k 16384 --m 0.996"
# ./script/command.sh ressl-tinyimagenet         1 1 "python -u ressl.py --dataset tinyimagenet --k 16384 --m 0.996"
# ./script/command.sh eval_cos-ressl-cifar10         1 1 "python -u linear_eval.py --dataset cifar10       --checkpoint ressl-cifar10.pth       --s cos  "
# ./script/command.sh eval_cos-ressl-cifar100        1 1 "python -u linear_eval.py --dataset cifar100      --checkpoint ressl-cifar100.pth      --s cos  "
# ./script/command.sh eval_cos-ressl-stl10           1 1 "python -u linear_eval.py --dataset stl10         --checkpoint ressl-stl10.pth         --s cos  "
# ./script/command.sh eval_cos-ressl-tinyimagenet    1 1 "python -u linear_eval.py --dataset tinyimagenet  --checkpoint ressl-tinyimagenet.pth  --s cos  "
