import os, glob
import math
import random

from skimage import io
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from scipy import ndimage
from scipy.ndimage import measurements
from skimage import morphology as morph

from .utils import center_pad_to_shape, cropping_center, get_bounding_box


def load_data(
    *,
    data_root,
    mask_type,
    task=['image2mask', 'mask2image'],
    image_name='images',
    inst_name='instance_labels',
    batch_size=4,
    deterministic=False,
    random_flip=True,
    random_rotate=True,
    seed=1,
):

    if not data_root:
        raise ValueError("unspecified data root")

    image_dir = os.path.join(data_root, image_name)
    inst_dir = os.path.join(data_root, inst_name)
    dataset = ImageDataset(
        image_dir,
        inst_dir,
        task=task,
        mask_type=mask_type,
        random_flip=random_flip,
        random_rotate=random_rotate,
    )
    print(f'Task {task}, Number of samples: {len(dataset)}')

    if deterministic:
        # setup_seed(seed)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(
            self,
            image_dir,
            instance_dir,
            task=['image2mask', 'mask2image'],
            mask_type=['sdm', 'hover'],
            random_flip=True,
            random_rotate=True,
    ):
        super().__init__()
        self.task = task
        self.mask_type = mask_type
        self.local_images = glob.glob(f'{image_dir}/*.png')
        self.local_instances = glob.glob(f'{instance_dir}/*.tif')
        self.random_flip = random_flip
        self.random_rotate = random_rotate

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):

        image_path = self.local_images[idx]
        inst_path = self.local_instances[idx]
        image = io.imread(image_path).astype(np.float32)
        image = image / 127.5 - 1.0
        inst = io.imread(inst_path) # uint32
        # print(image.dtype, inst.dtype)

        ## format mask
        sem = np.zeros_like(inst).astype(np.float32)
        sem[inst > 0] = 1.0
        sem[inst == 0] = -1.0
        if self.mask_type == 'sdm':
            edge = get_edges(inst, radius=1).astype(np.float32)
            edge[edge == 0] = -1.0
            mask = np.concatenate([sem[:,:,None], -sem[:,:,None], edge[:,:,None]], axis=-1)
        elif self.mask_type == 'hover':
            hv = get_hv(inst).astype(np.float32)
            mask = np.concatenate([sem[:,:,None], hv], axis=-1)
        else:
            raise NotImplementedError

        if self.random_flip and random.random() < 0.5:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()

        if self.random_rotate and random.random() < 0.5:
            rot_k = random.choice(range(1, 4))
            image = np.rot90(image, k=rot_k).copy()
            mask = np.rot90(mask, k=rot_k).copy()

        out_dict = {}
        out_dict['image_path'] = image_path
        out_dict['inst_path'] = inst_path

        if self.task == 'image2mask': # condition is image
            out_dict['y'] = np.transpose(image.copy(), [2, 0, 1])
            return np.transpose(mask, [2, 0, 1]), out_dict
        elif self.task == 'mask2image': # condition is mask
            out_dict['y'] = np.transpose(mask.copy(), [2, 0, 1])
            return np.transpose(image, [2, 0, 1]), out_dict
        else:
            raise NotImplementedError

def get_edges(t, radius=0):
    edge = np.zeros_like(t)
    edge[:, 1:] = edge[:, 1:] | (t[:, 1:] != t[:, :-1])
    edge[:, :-1] = edge[:, :-1] | (t[:, 1:] != t[:, :-1])
    edge[1:, :] = edge[1:, :] | (t[1:, :] != t[:-1, :])
    edge[:-1, :] = edge[:-1, :] | (t[1:, :] != t[:-1, :])
    if radius > 0:
        footprint = morph.disk(radius)
        edge = morph.binary_dilation(edge, footprint)
    return edge

def get_hv(ann):
    """Input annotation must be of original shape.

    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    orig_ann = crop_ann = ann.copy()  # instance ID map
    crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        inst_map = inst_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0]: inst_box[1], inst_box[2]: inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([x_map, y_map]) # (h, w, 2)
    return hv_map

