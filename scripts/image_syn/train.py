"""
Train a diffusion model on images.
"""

import os, sys
import argparse

from nudiff.image_syn.utils import dist_util, logger
from nudiff.image_syn.utils.datasets import load_data, load_batch
from nudiff.image_syn.cond.resample import create_named_schedule_sampler
from nudiff.image_syn.utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from nudiff.image_syn.src.run_desc import CycleTrainLoop
from nudiff.image_syn.hovernet.utils import convert_pytorch_checkpoint
import torch

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
    logger.info("Program executed via:\n%s\n" % ' '.join(sys.argv).replace("--", " \\ \n\t--"))

    logger.log("creating unet model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    model_size = get_model_size(model)
    print('Unet model size: {:.3f}MB'.format(model_size))
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_root=args.data_dir,
        batch_size=args.batch_size,
    )
    viz_data = load_batch(
        data_root=args.viz_data_dir,
        batch_size=args.viz_batch_size,
    )

    logger.log("training...")
    CycleTrainLoop(
        model=model,
        crop_size=164,
        diffusion=diffusion,
        data=data,
        viz_data=viz_data,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        drop_rate=args.drop_rate,
        log_interval=args.log_interval,
        viz_interval=args.viz_interval,
        save_interval=args.save_interval,
        max_iterations=args.max_iterations,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        cycle=args.cycle,
        warmup=args.warmup,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        viz_data_dir='',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.01,
        lr_anneal_steps=0,
        cycle=10000,
        warmup=2000,
        batch_size=1,
        viz_batch_size=9,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        drop_rate=0.0,
        log_interval=10,
        save_interval=10000,
        viz_interval=100,
        max_iterations=150000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        is_train=True,
        unet='unet',
        attention='CrossAttention',
        attention_mode=1,
        encoder_resolutions='',
        decoder_resolutions='32,16,8',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

if __name__ == "__main__":
    main()
