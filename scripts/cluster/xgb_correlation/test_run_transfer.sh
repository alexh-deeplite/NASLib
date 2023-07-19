#!/bin/bash

# train_sizes=(1000) # 14 24 42 71 121 1000)
train_size=1000
test_train_sizes=(5 8 14 42) # 71 121 1000)
searchspace=nasbench201
datasets=(cifar10) # cifar100 ImageNet16-120)
test_searchpace=nasbench301
test_dataset=cifar10
# ks=(1 2 3 4 5 6 7 8 9 10 11 12 13)
ks=(13)
# datasets=(cifar10)
start_seed=9000
n_seeds=5
experiment=$1
predictors=(adapt)

#"synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov epe_nas params grasp"


if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

for predictor in "${predictors[@]}"
do
for k in "${ks[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for test_train_size in "${test_train_sizes[@]}"
        do
        bash scripts/bash_scripts/xgb_correlation/create_configs_finetune.sh $experiment $searchspace $dataset $start_seed $k "synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov epe_nas params grasp" $predictor $test_searchpace $test_dataset $test_train_size # "synflow flops params"
        bash ./scripts/cluster/xgb_correlation/run_transfer.sh $searchspace $dataset $train_size $start_seed $experiment $k $n_seeds $test_train_size
        done
    done
done
done

# bash scripts/cluster/xgb_correlation/run.sh nasbench201 cifar10 $train_size 0 xgb_only_zc 3 5

# "synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov epe_nas params grasp min_depth max_depth"