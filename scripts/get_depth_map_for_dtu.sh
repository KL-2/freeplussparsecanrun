
###################################################################################
# get depth maps from large pre-trained models
# evaluation on 15 scenes:
# dataset_id: scan40, scan55, 63, 
# 110,114 21, 
# 30, 31, 8, #
# 34, 41,45,
# 82,103, 38

# benchmark=DTU #DTU 
# dataset_id=scan9
# root_path=/media/deep/HardDisk4T-new/datasets/DTU/Rectified
benchmark=DTU # LLFF
root_path=/home/user/software/freeplussparse/data/DTU/Rectified/
# python get_depth_map_for_llff_dtu.py --root_path $root_path --benchmark $benchmark --dataset_id $dataset_id


# for dataset_id in  scan1  scan4  scan9  scan10  scan11  scan12  scan13 scan15 scan23 scan24 scan29 scan32 scan33 scan34 scan48
for dataset_id in  scan118
do
    python get_depth_map_for_llff_dtu.py --root_path $root_path --benchmark $benchmark --dataset_id $dataset_id 
done
