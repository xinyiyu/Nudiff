"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import imageio
import numpy as np

import torch as th
import torch.distributed as dist
import torchvision as tv

from nudiff.image_syn.utils.datasets import load_data, load_batch, load_batches, load_masks

from nudiff.image_syn.utils import dist_util, logger
from nudiff.image_syn.utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

threads = '4'
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    print(dist_util.dev())
    model.to(dist_util.dev())

    logger.log("creating data loader...")

    if args.input_mask:
        data = load_masks(data_root=args.data_dir, res_dir=args.results_path, batch_size=args.batch_size, raw=False)
    else:
        data = load_batches(data_root=args.data_dir, batch_size=args.batch_size)

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if args.input_mask:
        sample_path = os.path.join(args.results_path, 'fake_images')
        os.makedirs(sample_path, exist_ok=True)
        logger.log("sampling...")
        conds = data
        n_samples = 0
        for cond in conds:
            model_kwargs = {'y': cond['label'].permute(0, 3, 1, 2), 's': args.s}
            sample_fn = diffusion.p_sample_loop
            sample = sample_fn(model, model_kwargs['y'].shape, clip_denoised=True, model_kwargs=model_kwargs, progress=True)
            sample = (sample + 1) / 2
            label = cond['label'].permute(0, 3, 1, 2)
            label[:,1:,:,:] = (label[:,1:,:,:] + 1) / 2
            for j in range(sample.shape[0]):
                tv.utils.save_image(sample[j], os.path.join(sample_path, f"{cond['path'][j].split('/')[-1].split('.')[0]}.png"))

            n_samples += sample.shape[0]
            logger.log(f"created {n_samples} samples")

            if n_samples >= args.num_samples:
                break
    else:
        label_path = os.path.join(args.results_path, 'labels')
        os.makedirs(label_path, exist_ok=True)
        sample_path = os.path.join(args.results_path, 'samples')
        os.makedirs(sample_path, exist_ok=True)
        image_path = os.path.join(args.results_path, 'images')
        os.makedirs(image_path, exist_ok=True)
        logger.log("sampling...")
        batches, conds = data[0], data[1]
        n_samples = 0
        for batch, cond in zip(batches, conds):
            model_kwargs = {'y': cond['label'].permute(0, 3, 1, 2), 's': args.s}
            print(model_kwargs['y'].shape)
            sample_fn = diffusion.p_sample_loop
            sample = sample_fn(model, model_kwargs['y'].shape, clip_denoised=True, model_kwargs=model_kwargs, progress=True)
            sample = (sample + 1) / 2
            label = cond['label'].permute(0, 3, 1, 2)
            label[:,1:,:,:] = (label[:,1:,:,:] + 1) / 2
            image = (batch + 1.0) / 2.0
            for j in range(sample.shape[0]):
                tv.utils.save_image(image[j], os.path.join(image_path, cond['image_path'][j].split('/')[-1].split('.')[0] + '.png'))
                tv.utils.save_image(sample[j], os.path.join(sample_path, f"{cond['image_path'][j].split('/')[-1].split('.')[0]}.png"))
                tv.utils.save_image(label[j], os.path.join(label_path, cond['image_path'][j].split('/')[-1].split('.')[0] + '.png'))

            n_samples += sample.shape[0]
            logger.log(f"created {n_samples} samples")

            if n_samples >= args.num_samples:
                break

    dist.barrier()
    logger.log("sampling complete")

## changed
def preprocess_input(data):
    input_semantics = data['label'].permute(0, 3, 1, 2) # bhwc -> bchw
    cond = {'y': input_semantics}
    return cond


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=100,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        train_set=False,
        s=1.0,
        input_mask=False,
        unet='unet',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
