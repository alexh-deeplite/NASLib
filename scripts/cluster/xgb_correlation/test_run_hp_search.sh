#!/bin/bash

# train_sizes=(1000)
size=1000
searchspace=nasbench201
datasets=(cifar10) # cifar100 ImageNet16-120)
# ks=(1 2 3 4 5 6 7 8 9 10 11 12 13)
ks=(13)
# datasets=(cifar10)
start_seed=9000
n_seeds=10
experiment=$1
predictors=(mlp)

#"synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov epe_nas params grasp"
params=( "hidden_layer_size" "hidden_layer_size" "hidden_layer_size" "hidden_layer_size" "hidden_layer_size" "hidden_layer_size" )
values=( 020020 050050 100100 020020020 050050050 100100100 )
# params=( "learning_rate_init" "learning_rate_init" "learning_rate_init" "learning_rate_init" "learning_rate_init" "learning_rate_init" "learning_rate_init")
# values=( .0001 .0005 .001 .005 .01 .05 .1 )
# declare -A pairs=( ["hidden_layer_size"]=30 ["hidden_layer_size"]=60 ["hidden_layer_size"]=100 )

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
        for idx in "${!params[@]}"
        do
            param=${params[$idx]}
            value=${values[$idx]}
            bash scripts/bash_scripts/xgb_correlation/create_configs_hp.sh $experiment $searchspace $dataset $start_seed $k "synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov epe_nas params grasp" $predictor $param $value # "synflow flops params"
            # for size in "${train_sizes[@]}"
        
            bash ./scripts/cluster/xgb_correlation/run_hp_search.sh $searchspace $dataset $size $start_seed $experiment $k $n_seeds
        done
    done
done
done

# bash scripts/cluster/xgb_correlation/run.sh nasbench201 cifar10 $train_size 0 xgb_only_zc 3 5

# "synflow plain l2_norm flops snip grad_norm nwot zen fisher jacov epe_nas params grasp min_depth max_depth"