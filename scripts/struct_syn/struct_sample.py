"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import imageio
import random
import string
import sys
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision as tv

from nudiff.struct_syn import logger
from nudiff.struct_syn.script_util import (
    NUM_CLASSES,
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

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def main():
    args = create_argparser().parse_args()
    print(args.in_channels)

    logger.configure()
    device = 'cuda'

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    get_model_size(model)
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    raw_path = os.path.join(args.results_path, 'raw')
    os.makedirs(raw_path, exist_ok=True)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    n_samples = 0
    while n_samples < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, frames = sample_fn(
            model,
            (args.batch_size, args.in_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
        )
        sample = (sample + 1) / 2.0


        for j in range(sample.shape[0]):
            filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            tv.utils.save_image(sample[j], os.path.join(sample_path, f'{filename}.png'))
            np.save(os.path.join(raw_path, f'{filename}.npy'), sample[j].cpu().numpy().transpose(1,2,0).astype(np.float32))
            
        n_samples += sample.shape[0]
        print(f'Current number of samples: {n_samples}')
        logger.log(f"created {n_samples} samples")

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        model_path="",
        results_path="",
        in_channels=3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
