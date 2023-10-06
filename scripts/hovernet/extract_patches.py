"""extract_patches.py

Patch extraction script.
"""
import sys
sys.path.append('/data/yuxinyi/hover_net') # Change it to your hover_net path
import re
import glob
import os
import tqdm
import pathlib

import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = False

    model_mode = 'fast'
    if model_mode == 'original':
        win_size = [270, 270]
        step_size = [80, 80]
    if model_mode == 'fast':
        win_size = [256, 256]
        step_size = [164, 164]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.
    if extract_type == 'valid':
        _extract_type = '_valid'
    else:
        _extract_type = ''
    
    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "kumar"
    prop = 0.11
    save_root = f"hovernet_dataset/monuseg" # change this
    os.makedirs(save_root, exist_ok=True)

    # for 10%/20%/50%/100% labeled:
    dataset_info = {
        f"prop{prop:.1f}": {
            "img": (".png", f"/data/yuxinyi/semantic-diffusion-model/monuseg/allpatch256x256_128/train_6class_prop{prop:.2f}/images"),
            "ann": (".tif", f"/data/yuxinyi/semantic-diffusion-model/monuseg/allpatch256x256_128/train_6class_prop{prop:.2f}/instance_labels"),
        },
    }
    # # for 10%/20%/50%/100% augmented:
    # step1 = 150000 # for example
    # step2 = 300000 # for example
    # num_samples = 512 # for example
    # s = 2.0
    # dataset_info = {
    #     f"prop{prop:.1f}+_syn_{num_samples}": {
    #         "img": (".png", f"../../results/monuseg_mask/prop{prop:.1f}/{step1}_{num_samples}/finetune_{step2}_s{s}/fake_images"),
    #         "ann": (".mat", f"../../results/monuseg_mask/prop{prop:.1f}/{step1}_{num_samples}/instances"),
    #     },
    # }
    

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%dx%d_%dx%d%s/" % (
            save_root,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
            _extract_type,
        )
        file_list = glob.glob(patterning("%s/*%s" % (img_dir, img_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )

            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()
