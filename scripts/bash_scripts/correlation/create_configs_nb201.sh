#!/bin/bash

searchspace=nasbench201
datasets=(cifar10 cifar100) # ImageNet16-120 svhn scifar100 ninapro)
predictors=(min_depth max_depth)

for predictor in "${predictors[@]}"
do
for dataset in "${datasets[@]}"
do
for seed in {0..10}
do
    scripts/bash_scripts/correlation/create_configs.sh $searchspace $dataset $((9000 + seed)) $predictor
done
done
done