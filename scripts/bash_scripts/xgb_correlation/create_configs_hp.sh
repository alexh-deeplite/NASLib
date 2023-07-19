#!/bin/bash

# search space and datasets:
experiment=$1
search_space=$2
dataset=$3
start_seed=$4
k=$5
zc_names=$6
predictor=$7
test_search_space=$8
test_dataset=$9
param=$8
value=$9

# train_size=$7

if [ -z "$experiment" ]
then
    echo Experiment argument not provided
    exit 1
fi

if [ -z "$search_space" ]
then
    echo Search space argument not provided
    exit 1
fi

if [ -z "$dataset" ]
then
    echo Dataset argument not provided
    exit 1
fi

if [ -z "$start_seed" ]
then
    start_seed=0
fi

if [ -z "$predictor" ]
then
    predictor="xgb"
fi

if [ -z "$test_dataset" ]
then
    test_dataset=$dataset
fi

if [ -z "$test_search_space" ]
then
    test_search_space=$search_space
fi


out_dir=run
trials=100
end_seed=$(($start_seed + $trials - 1))
train_sizes=(5 8 14 24 42 71 121 205 347 589 1000 2000 4000 8000)
# train_sizes=(100)
test_size=100
config_root=configs

# if [[ "$search_space" == "transbench101_micro"  ||  "$search_space" == "transbench101_macro" ]]; then
#     zc_names="flops params snip jacov grad_norm plain fisher grasp l2_norm nwot zen"
# else
#     zc_names="flops params snip jacov grad_norm plain fisher grasp l2_norm nwot zen epe_nas synflow"
# fi

if [[ "$experiment" == "xgb_only_zc" ]]; then
    echo xgb_only_zc
    zc_ensemble=True
    zc_only=True
fi


if [[ "$experiment" == "model_only_zc" ]]; then
    echo xgb_only_zc
    zc_ensemble=True
    zc_only=True
fi

if [[ "$experiment" == *"transfer_model_only_zc"* ]]; then
    echo xgb_only_zc
    zc_ensemble=True
    zc_only=True
fi

if [[ "$experiment" == "xgb_only_adjacency" ]]; then
    echo xgb_only_adjacency
    zc_ensemble="False"
    zc_only="False"
fi



echo bash $zc_ensemble $zc_only

if [ -z "$zc_ensemble" ]
then
    echo zc_ensemble not set
    exit 1
fi

if [ -z "$zc_only" ]
then
    echo zc_only not set
    exit 1
fi

for train_size in "${train_sizes[@]}"
do
echo predictor $predictor
echo $param
echo $value
python scripts/create_configs_xgb_correlation_hp.py --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --search_space $search_space --config_root=$config_root --zc_names $zc_names \
    --train_size $train_size --experiment $experiment --zc_ensemble $zc_ensemble --zc_only $zc_only \
    --test_size $test_size --k $k --predictor $predictor --test_dataset $test_dataset --test_search_space $test_search_space  --param $param --value $value
done
