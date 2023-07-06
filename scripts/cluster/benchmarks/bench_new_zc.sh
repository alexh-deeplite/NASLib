#!/bin/bash

searchspace=nasbench201
dataset=cifar10
predictor=min_depth
# predictor=snip
start_seed=9000
experiment=correlation


# make configs
bash ./scripts/bash_scripts/benchmarks/create_configs.sh $searchspace $dataset 9000 $predictor
echo "made config"

# run benchmark
bash ./scripts/cluster/benchmarks/run.sh $searchspace $dataset $predictor $start_seed $experiment