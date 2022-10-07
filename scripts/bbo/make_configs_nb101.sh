export OMP_NUM_THREADS=2
# optimizers=(rs)
# optimizers=(rs re ls npenas bananas)
optimizers=(npenas bananas)
zerocosts=(none predictor api)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

if [[ $optimizers == bananas* ]]
then
  acq_fn_optimization=mutation
else
  acq_fn_optimization=random_sampling
fi

# folders:
base_file=naslib
out_dir=run_cpu

# bbo-bs or predictor-bs
config_type=bbo-bs

# search space / data:
search_space=nasbench101
dataset=cifar10

fidelity=-1
epochs=200
predictor=var_sparse_gp

# trials / seeds:
trials=1 # FIXME 10
end_seed=$(($start_seed + $trials - 1))

# create config files
for i in $(seq 0 $((${#dataset[@]}-1)) )
do 
    dataset=${dataset[$i]}
    echo $dataset
    for i in $(seq 0 $((${#optimizers[@]}-1)) )
	do
		for j in $(seq 0 $((${#zerocosts[@]}-1)) )
		do
			optimizer=${optimizers[$i]}
			zerocost=${zerocosts[$j]}
			python create_configs.py \
			--start_seed $start_seed --trials $trials \
			--out_dir $out_dir --dataset=$dataset --config_type $config_type \
			--search_space $search_space --optimizer $optimizer \
			--acq_fn_optimization $acq_fn_optimization --predictor $predictor \
			--fidelity $fidelity --epochs $epochs --zerocost $zerocost
		done
	done
done


echo 'configs are ready, check config folder ...'
