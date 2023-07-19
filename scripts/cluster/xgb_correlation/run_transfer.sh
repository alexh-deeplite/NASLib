#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #,ml_gpu-rtx2080 #ml_gpu-rtx2080     # bosch_gpu-rtx2080    #alldlc_gpu-rtx2080     # partition (queue)
#SBATCH -o logs/%x.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%x.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -a 0 # array size
#SBATCH --mem=5G
#SBATCH --job-name="XGB_ZC_CORRELATION"

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
file=naslib/runners/bbo/model_transfer_runner.py

searchspace=$1
dataset=$2
train_size=$3
start_seed=$4
experiment=$5
k=$6
n_seeds=$7
test_train_size=$8

if [ -z "$searchspace" ]
then
    echo "Search space argument not provided"
    exit 1
fi

if [ -z "$dataset" ]
then
    echo "Dataset argument not provided"
    exit 1
fi

if [ -z "$train_size" ]
then
    echo "Train size argument not provided"
    exit 1
fi

if [ -z "$start_seed" ]
then
    echo "Start seed not provided"
    exit 1
fi

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

if [ -z "$test_train_size" ]
then
    test_train_size=0
fi

if [[ $experiment == "finetune_transfer_model_only_zc" ]]; then
    script=naslib/runners/bbo/test_model_transfer_runner_finetune.py
fi

start=`date +%s`
for t in $(seq 1 $n_seeds)
do
    seed=$(($start_seed + $t - 1))
    echo seed $seed
    python $script --config-file configs/${experiment}/${test_train_size}/$k/${searchspace}-${start_seed}/${dataset}/config_${seed}.yaml
done
# echo "waiting for seeds to train"
# wait
# echo "seeds completed"
end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
