import glob
import os
import shutil

if __name__=="__main__":
    
    src_path_dir = "/media/tuan/Daten/project/projects/stereo_matching/RAFT_Stereo_pretrained/val_wrongname"
    dst_path_dir = "/media/tuan/Daten/project/projects/stereo_matching/RAFT_Stereo_pretrained/val"

    for src_sub_path in glob.glob("{}/*".format(src_path_dir)):

        city_name = src_sub_path.split("/")[-1]
        dst_sub_path = os.path.join(dst_path_dir, city_name)
        os.makedirs(dst_sub_path, exist_ok=True)

        for src_img_path in glob.glob("{}/*".format(src_sub_path)):
            img_name = src_img_path.split("/")[-1].split(".")[0]
            new_img_name = "_".join(img_name.split("_")[:-1]) + "_disparity.npy"
            dst_img_path = os.path.join(dst_sub_path, new_img_name)
            shutil.copy(src_img_path, dst_img_path)