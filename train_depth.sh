srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 python train_depth.py --gin_configs configs/freenerf/llff3_freenerf_plus.gin
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 python train_depth.py --gin_configs configs/freenerf/llff3_freenerf_plus2.gin