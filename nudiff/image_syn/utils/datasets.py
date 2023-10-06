import os, glob
import math
import random
from tqdm import tqdm
from skimage import io
from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from scipy import ndimage
from scipy.ndimage import measurements
from skimage import morphology as morph

from nudiff.image_syn.hovernet.utils import center_pad_to_shape, cropping_center, get_bounding_box

def load_data(
    *,
    data_root,
    image_name='images',
    inst_name='instance_labels',
    batch_size=4,
    random_flip=True,
    random_rotate=True,
):

    if not data_root:
        raise ValueError("unspecified data root")

    image_dir = os.path.join(data_root, image_name)
    inst_dir = os.path.join(data_root, inst_name)
    dataset = ImageDataset(
        image_dir,
        inst_dir,
        random_flip=random_flip,
        random_rotate=random_rotate,
    )
    print(f'Number of samples: {len(dataset)}')

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )
    while True:
        yield from loader
    
def get_batch(dataset, batch_idx):

    # Set up the datastructures
    im_size = dataset[0][0].size()
    batch_size = len(batch_idx)
    batch_data = torch.empty((batch_size, *im_size))
    batch_labels = torch.empty((batch_size, 1))

    # Add data to datastructures
    for i, data_idx in enumerate(batch_idx):
        data, label = dataset[data_idx]
        batch_data[i] = data
        batch_labels[i] = label

    return batch_data, batch_labels


def load_batch(
    *,
    data_root,
    image_name='images',
    inst_name='instance_labels',
    batch_size=9,
    seed=42,
): 
    image_dir = os.path.join(data_root, image_name)
    inst_dir = os.path.join(data_root, inst_name)
    dataset = ImageDataset(
        image_dir,
        inst_dir,
        random_flip=False,
        random_rotate=False,
    )
    print(f'Viz data len: {len(dataset)}')
    
    random.seed(seed)
    batch_ids = random.sample(range(len(dataset)), batch_size)
    # print(type(dataset[0][0]))
    image_size = dataset[0][0].shape
    batch = torch.empty((batch_size, *image_size))
    cond = {k: [] for k in dataset[0][1].keys()}
    for i, idx in enumerate(batch_ids):
        image, out_dict = dataset[idx]
        batch[i] = torch.tensor(image)
        cond['label'].append(torch.tensor(out_dict['label']))
        cond['image_path'].append(out_dict['image_path'])
        cond['inst_path'].append(out_dict['inst_path'])
    cond['label'] = torch.stack(cond['label'])
    # print(f"viz image: {batch.shape}, viz label: {cond['label'].shape}")
    # print(f"viz paths: {cond['image_path']}")
    
    return batch, cond
    
def load_batches(
    *,
    data_root,
    image_name='images',
    inst_name='instance_labels',
    batch_size=9,
): 
    image_dir = os.path.join(data_root, image_name)
    inst_dir = os.path.join(data_root, inst_name)
    dataset = ImageDataset(
        image_dir,
        inst_dir,
        random_flip=False,
        random_rotate=False,
    )
    print(f'Viz data len: {len(dataset)}')
    
    # batch_ids = random.sample(range(len(dataset)), batch_size)
    batches, conds = [], []
    starts = np.arange(0, len(dataset), batch_size)
    for start in tqdm(starts):
        end = min(start + batch_size, len(dataset))
        batch_ids = np.arange(start, end)
        cur_batch_size = end - start
        # print(batch_ids)
        image_size = dataset[0][0].shape
        batch = torch.empty((cur_batch_size, *image_size))
        cond = {k: [] for k in dataset[0][1].keys()}
        for i, idx in enumerate(batch_ids):
            image, out_dict = dataset[idx]
            batch[i] = torch.tensor(image)
            cond['label'].append(torch.tensor(out_dict['label']))
            cond['image_path'].append(out_dict['image_path'])
            cond['inst_path'].append(out_dict['inst_path'])
        cond['label'] = torch.stack(cond['label'])
        batches.append(batch)
        conds.append(cond)
    
    return batches, conds

def load_masks(
    *,
    data_root,
    res_dir,
    batch_size=4,
    raw=False,
): 

    if raw:
        dataset = RawMaskDataset(data_root)
    else:
        dataset = MaskDataset(data_root, res_dir)
    print(f'Viz data len: {len(dataset)}')
    
    # batch_ids = random.sample(range(len(dataset)), batch_size)
    batches, conds = [], []
    starts = np.arange(0, len(dataset), batch_size)
    for start in tqdm(starts):
        end = min(start + batch_size, len(dataset))
        batch_ids = np.arange(start, end)
        # print(batch_ids)
        cond = {k: [] for k in dataset[0].keys()}
        for i, idx in enumerate(batch_ids):
            out_dict = dataset[idx]
            cond['label'].append(torch.tensor(out_dict['label']))
            cond['path'].append(out_dict['path'])
        cond['label'] = torch.stack(cond['label'])
        conds.append(cond)
    
    return conds


class ImageDataset(Dataset):
    def __init__(
            self,
            image_dir,
            instance_dir,
            random_flip=True,
            random_rotate=True,
    ):
        super().__init__()
        self.local_images = sorted(glob.glob(f'{image_dir}/*.png'))
        self.local_instances = sorted(glob.glob(f'{instance_dir}/*.tif'))
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
        sem = np.zeros(image.shape[:2])
        sem[inst > 0] = 1
        # sem[inst == 0] = -1
        hv = get_hv(inst).astype(np.float32)
        # print(sem.shape, hv.shape)
        mask = np.concatenate([sem[:,:,None], hv], axis=-1)

        if self.random_flip and random.random() < 0.5:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()

        if self.random_rotate and random.random() < 0.5:
            rot_k = random.choice(range(1, 4))
            image = np.rot90(image, k=rot_k).copy()
            mask = np.rot90(mask, k=rot_k).copy()

        out_dict = {'image_path': image_path,
                    'inst_path': inst_path,
                    'label': mask.copy()}

        return np.transpose(image, [2, 0, 1]), out_dict

class MaskDataset(Dataset):
    def __init__(
            self,
            mask_dir,
            res_dir=None,
    ):
        super().__init__()
        local_masks = sorted(glob.glob(f'{mask_dir}/*.png'))
        # existing fake images
        all_names = [os.path.basename(x) for x in local_masks]
        if res_dir:
            exists = sorted(glob.glob(f'{res_dir}/fake_images/*.png'))
            exist_names = [os.path.basename(x) for x in exists]
            print(f'{len(exist_names)} fake images already generated.')
        else:
            exist_names = []
        mask_names = list(set(all_names) - set(exist_names))
        self.local_masks = sorted([x for x in local_masks if os.path.basename(x) in mask_names])
        print(f'{len(self.local_masks)} fake images to be generated.')

    def __len__(self):
        return len(self.local_masks)

    def __getitem__(self, idx):

        mask_path = self.local_masks[idx]
        mask = io.imread(mask_path).astype(np.float32)
        # print(image.dtype, inst.dtype)
        
        label = mask.copy()
        label[:,:,0] = label[:,:,0] / 255
        label[:,:,0] = label[:,:,0] > 0.5
        label[:,:,1:] = label[:,:,1:] / 127.5 - 1

        out_dict = {'path': mask_path,
                    'label': label.astype(np.float32)}

        return out_dict
    
class RawMaskDataset(Dataset):
    def __init__(
            self,
            mask_dir,
    ):
        super().__init__()
        self.local_masks = sorted(glob.glob(f'{mask_dir}/*.npy'))

    def __len__(self):
        return len(self.local_masks)

    def __getitem__(self, idx):

        mask_path = self.local_masks[idx]
        mask = np.load(mask_path)
        # print(image.dtype, inst.dtype)
        
        label = mask.copy()
        label[:,:,0] = label[:,:,0] > 0.5
        label[:,:,1:] = label[:,:,1:] * 2 - 1

        out_dict = {'path': mask_path,
                    'label': label.astype(np.float32)}

        return out_dict

    
def get_edges(t):
    edge = np.zeros_like(t)
    edge[:, 1:] = edge[:, 1:] | (t[:, 1:] != t[:, :-1])
    edge[:, :-1] = edge[:, :-1] | (t[:, 1:] != t[:, :-1])
    edge[1:, :] = edge[1:, :] | (t[1:, :] != t[:-1, :])
    edge[:-1, :] = edge[:-1, :] | (t[1:, :] != t[:-1, :])
    return edge

def get_hv(ann):
    """Input annotation must be of original shape.

    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    orig_ann = fixed_ann = crop_ann = ann.copy()  # instance ID map
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

