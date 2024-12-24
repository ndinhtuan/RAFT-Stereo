import sys
sys.path.append('core')

import os
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from demo import load_image, DEVICE

class CityscapesVPSDisparityCreator(object):

    def __init__(self, args: any) -> None:
        
        self.__left_imgs_dir = args.left_imgs_dir
        self.__left_imgs_path = glob.glob("{}/*.png".format(self.__left_imgs_dir))
        self.__right_imgs_dir = args.right_imgs_dir
        self.__args = args

        if not os.path.isdir(self.__args.output_directory):
            os.makedirs(self.__args.output_directory)

        self.__model = self.__load_model()

    def __get_right_img_path(self, left_img_path: str) -> str:
        """
        Get corresponding right image for the provided left image, for example: 0074_0439_frankfurt_000001_009839_newImg8bit.png
        """

        name_left_img = left_img_path.split("/")[-1]
        city_name = name_left_img.split("_")[2]
        name_right_img = name_left_img[len("0000_0000_"):].replace("newImg8bit", "rightImg8bit").replace("leftImg8bit", "rightImg8bit")
        
        return os.path.join(self.__right_imgs_dir, city_name, name_right_img)

    def __load_model(self) -> torch.nn.DataParallel:

        model = torch.nn.DataParallel(RAFTStereo(self.__args), device_ids=[0])
        model.load_state_dict(torch.load(self.__args.restore_ckpt))

        model = model.module
        model.to(DEVICE)
        model.eval()

        return model
    
    def __predict_one(self, img_left_path: str, img_right_path: str) -> torch.Tensor:
        
        image1 = load_image(img_left_path)
        image2 = load_image(img_right_path)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        _, flow_up = self.__model(image1, image2, iters=self.__args.valid_iters, test_mode=True)
        flow_up = padder.unpad(flow_up).squeeze()
        depth = -flow_up.cpu().numpy().squeeze()

        return depth
    
    def run(self) -> None:

        with torch.no_grad():

            for left_img_path in tqdm(self.__left_imgs_path):

                right_img_path = self.__get_right_img_path(left_img_path=left_img_path)
                disparity = self.__predict_one(img_left_path=left_img_path, img_right_path=right_img_path)

                # Save disparity
                name_left_img = left_img_path.split("/")[-1]
                path_to_save = os.path.join(self.__args.output_directory, name_left_img.replace("png", "npy"))
                np.save(path_to_save, disparity)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs_dir', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs_dir', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    cityvps_disparity_creator = CityscapesVPSDisparityCreator(args=args)
    cityvps_disparity_creator.run()

if __name__=="__main__":
    
    main()