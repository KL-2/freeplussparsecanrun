###################################################################################
# #dataset_id: fern  flower  fortress  horns  leaves  orchids  room  trex
# benchmark=LLFF # LLFF
dataset_id=flower
# root_path=/media/deep/HardDisk4T-new/datasets/nerf_llff_data-20220519T122018Z-001/nerf_llff_data/
# python get_depth_map.py --root_path $root_path --benchmark $benchmark --dataset_id $dataset_id 


# dataset_id: fern  flower  fortress  horns  leaves  orchids  room  trex
benchmark=LLFF # LLFF
root_path=/home/user/software/freeplussparse/data/nerf_data/nerf_llff_data/

#for dataset_id in  flower  fortress  horns  leaves  orchids  room  trex
for dataset_id in  flower
do
    python get_depth_map_for_llff_dtu.py --root_path $root_path --benchmark $benchmark --dataset_id $dataset_id 
done

