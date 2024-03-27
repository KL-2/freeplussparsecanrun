# evaluation on 15 scenes:
# scan30,scan34,scan41,scan45, scan82,scan103, scan38, scan21
# scan40, scan55, scan63, scan31, scan8, scan110, scan114, 

# CUDA_VISIBLE_DEVICES=0  python train_llff_dtu.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan1" --postfix "debug1" 

# 定义一个数组
# my_array=(1 2 3 4 5 7 9 14)
#指令：bash scripts/eval_dtu.sh
my_array=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
# my_array=(1)
# 使用 for 循环遍历数组
for i in "${my_array[@]}"
do
    echo "n_input_views is $i"
    CUDA_VISIBLE_DEVICES=0  python train_depth.py --gin_configs configs/dtu3_freenerf_plus.gin --checkpoint_dir "dtu3_"  --dataset_id "scan118" --n_input_views "$i" --postfix "sparsity$i" # && \
done
my_array=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
# my_array=(1)
# 使用 for 循环遍历数组
for i in "${my_array[@]}"
do
    echo "n_input_views is $i"
    CUDA_VISIBLE_DEVICES=0  python eval.py --gin_configs configs/dtu3_freenerf_plus.gin --checkpoint_dir "dtu3_"  --dataset_id "scan118" --n_input_views "$i" --postfix "sparsity$i" # && \
done