export PYTHONPATH=$PWD:$PYTHONPATH
# # train
# python my_tools/create_disparity_map_for_city_vps.py \
#     --restore_ckpt /media/tuan/Daten/project/projects/stereo_matching/RAFT_Stereo_pretrained/models/raftstereo-eth3d.pth \
#     -l /media/tuan/Daten/dataset/vps_dataset/cityscapes_vps/cityscapes_vps/train/img \
#     -r /media/tuan/Daten/project/Downloads/rightImg8bit_sequence_trainvaltest/rightImg8bit_sequence/val \
#     --output_directory /media/tuan/Daten/dataset/vps_dataset/cityscapes_vps/cityscapes_vps/train/disparity 

# echo "Generate for train set done!"

# # val
# python my_tools/create_disparity_map_for_city_vps.py \
#     --restore_ckpt /media/tuan/Daten/project/projects/stereo_matching/RAFT_Stereo_pretrained/models/raftstereo-eth3d.pth \
#     -l /media/tuan/Daten/dataset/vps_dataset/cityscapes_vps/cityscapes_vps/val/img \
#     -r /media/tuan/Daten/project/Downloads/rightImg8bit_sequence_trainvaltest/rightImg8bit_sequence/val \
#     --output_directory /media/tuan/Daten/dataset/vps_dataset/cityscapes_vps/cityscapes_vps/val/disparity 

# echo "Generate for val set done!"

# val_split
python my_tools/create_disparity_map_for_city_vps.py \
    --restore_ckpt /media/tuan/Daten/project/projects/stereo_matching/RAFT_Stereo_pretrained/models/raftstereo-eth3d.pth \
    -l /media/tuan/Daten/dataset/vps_dataset/cityscapes_vps/cityscapes_vps/val_split_2/img_all \
    -r /media/tuan/Daten/project/Downloads/rightImg8bit_sequence_trainvaltest/rightImg8bit_sequence/val \
    --existed_directory /media/tuan/Daten/dataset/vps_dataset/cityscapes_vps/cityscapes_vps/val_split_2/disparity \
    --output_directory /media/tuan/Daten/dataset/vps_dataset/cityscapes_vps/cityscapes_vps/val_split_2/disparity_all 

echo "Generate for val_split_all set done!"