srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 python train_depth.py --gin_configs configs/freenerf/llff3_freenerf_plus5.gin
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 python train_depth.py --gin_configs configs/freenerf/llff3_freenerf_plus6.gin
srun -p priority --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 python train_depth.py --gin_configs configs/freenerf/llff3_freenerf_plus7.gin