#!/bin/bash

searchspace=nasbench201
datasets=(cifar100)
predictors=(min_depth max_depth)
# predictor=snip
start_seed=9000
experiment=benchmarks

for dataset in "${datasets[@]}"
do
for predictor in "${predictors[@]}"
do
# # make configs
# bash ./scripts/bash_scripts/benchmarks/create_configs.sh $searchspace $dataset 9000 $predictor
# echo "made config"

start_idx=0
for i in 0
do
bash ./scripts/cluster/benchmarks/run.sh $searchspace $dataset $predictor $start_seed $experiment $start_idx 1000
start_idx=$((start_idx + 1000))
done

# predictor=max_depth
# start_idx=0
# # n_models=
# # run benchmark
# for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
# do
# bash ./scripts/cluster/benchmarks/run.sh $searchspace $dataset $predictor $start_seed $experiment $start_idx 1000
# start_idx=$((start_idx + 1000))
# done
done
done