#!/bin/bash

train_sizes=(5 8 14 24 42 71 121 205 347 589 1000)
searchspace=nasbench201
datasets=(cifar10 cifar100 ImageNet16-120)
# ks=(1 2 3 4 5 6 7 8 9 10 11 12 13)
ks=(13)
# datasets=(cifar10)
start_seed=9000

experiment=$1

#"synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov epe_nas params grasp"


if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

for k in "${ks[@]}"
do
    for dataset in "${datasets[@]}"
    do
        bash scripts/bash_scripts/xgb_correlation/create_configs.sh $experiment $searchspace $dataset $start_seed $k "synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov epe_nas params grasp"
        for size in "${train_sizes[@]}"
        do
            bash ./scripts/cluster/xgb_correlation/run.sh $searchspace $dataset $size $start_seed $experiment $k 50
        done
    done
done


# bash scripts/cluster/xgb_correlation/run.sh nasbench201 cifar10 $train_size 0 xgb_only_zc 3 5