# 定义一个数组
# my_array=(1 2 3 4 5 7 9 14)
#指令：bash scripts/eval_llff.sh
my_array=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
# 使用 for 循环遍历数组
for i in "${my_array[@]}"
do
    echo "n_input_views is $i"
    CUDA_VISIBLE_DEVICES=0  python train_depth.py --gin_configs configs/llff3_freenerf_plus.gin --checkpoint_dir "llff3_"  --dataset_id "flower" --n_input_views "$i" --postfix "sparsity$i" # && \
done

my_array=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
# 使用 for 循环遍历数组
for i in "${my_array[@]}"
do
    echo "n_input_views is $i"
    CUDA_VISIBLE_DEVICES=0  python eval.py --gin_configs configs/llff3_freenerf_plus.gin --checkpoint_dir "llff3_"  --dataset_id "flower" --n_input_views "$i" --postfix "sparsity$i" # && \
done
