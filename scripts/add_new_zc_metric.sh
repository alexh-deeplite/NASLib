#!/bin/bash

args=$1


searchspaces=(nasbench301)
n_rounds=12
datasets=(cifar10) # svhn scifar100 ninapro)
predictors=(min_depth)
start_seed=9000
experiment=benchmarks


# create configs
for searchspace in $searchspaces; do
for dataset in $datasets; do
for predictor in $predictors; do
    echo $searchspace 
    echo $dataset
    echo $predictor
    echo $start_seed
    if [[ $args == *"benchmark"* ]]; then
        # benchmark
        bash scripts/bash_scripts/benchmarks/create_configs.sh $searchspace $dataset $start_seed $predictor
        start_idx=0
        for idx in `seq 0 $n_rounds`; do
            bash ./scripts/cluster/benchmarks/run.sh $searchspace $dataset $predictor $start_seed benchmarks $start_idx 1000
            start_idx=$((start_idx + 1000))
        done
        python scripts/merge_benchmarks.py --dataset $dataset --search_space $searchspace
    fi


    if [[ $args == *"correlation"* ]]; then
        # correlation
        for seed in {0..10}; do
            scripts/bash_scripts/correlation/create_configs.sh $searchspace $dataset $((9000 + seed)) $predictor
            # run correlation
            bash ./scripts/cluster/correlation/run.sh $searchspace $dataset $predictor $((start_seed + seed)) correlation
        done
    fi

    if [[ $args == *"xgb"* ]]; then
        # xgb_only_zc
        ks=(4)
        train_sizes=(5) # 8 14 24 42 71) #121 205 347 589 1000)
        for k in "${ks[@]}"; do
            bash scripts/bash_scripts/xgb_correlation/create_configs.sh xgb_only_zc $searchspace $dataset $start_seed $k "synflow flops params ${predictor}"
            for size in "${train_sizes[@]}"; do
                bash ./scripts/cluster/xgb_correlation/run.sh $searchspace $dataset $size $start_seed xgb_only_zc $k 50
            done
        done
    fi
done
done
done




