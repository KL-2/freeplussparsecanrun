# evaluation on 15 scenes:
# scan30,scan34,scan41,scan45, scan82,scan103, scan38, scan21
# scan40, scan55, scan63, scan31, scan8, scan110, scan114, 

CUDA_VISIBLE_DEVICES=0  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan1" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=1  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan4" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=2  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan9" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=3  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan10" --postfix "debug1" 

CUDA_VISIBLE_DEVICES=4  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan11" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=5  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan12" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=6  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan13" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=7  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan15" --postfix "debug1" 


CUDA_VISIBLE_DEVICES=0  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan23" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=1  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan24" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=2  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan29" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=3  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan32" --postfix "debug1" 

CUDA_VISIBLE_DEVICES=4  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan33" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=5  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan34" --postfix "debug1" 
CUDA_VISIBLE_DEVICES=6  python eval.py --gin_configs configs/dtu3.gin --checkpoint_dir "dtu3_"  --dataset_id "scan48" --postfix "debug1" 

