import copy
import functools
import os
import warnings
import blobfile as bf
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TTF
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from collections import OrderedDict

from nudiff.image_syn.hovernet.utils import crop_to_shape, dice_loss, mse_loss, msge_loss, xentropy_loss, center_pad_to_shape, cropping_center
from nudiff.image_syn.utils.train_util import TrainLoop, get_blob_logdir
from nudiff.image_syn.utils import dist_util, logger
from nudiff.image_syn.utils.fp16_util import MixedPrecisionTrainer
from nudiff.image_syn.cond.nn import update_ema
from nudiff.image_syn.cond.resample import LossAwareSampler, UniformSampler
from nudiff.image_syn.utils.scheduler import CosineAnnealingWarmupRestarts

import matplotlib.pyplot as plt
from skimage import io
from warnings import warn

loss_opts = {
        "np": {"bce": 1, "dice": 1},
        "hv": {"mse": 1, "msge": 1},
    }
loss_func_dict = {
    "bce": xentropy_loss,
    "dice": dice_loss,
    "mse": mse_loss,
    "msge": msge_loss,
}

def process_diff_out(x):
    # clip
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2 * 255.0
    return x

def log_loss_dict(losses):
    for key, values in losses.items():
        if hasattr(values, '__len__'):
            logger.logkv_mean(key, values.mean().item())
        else:
            logger.logkv_mean(key, values)

def show(imgs, save=None):
    if imgs.shape[0] == 9:
        fig, axs = plt.subplots(3, 3, figsize=(8, 8), dpi=100)
        n = 0
        for i in range(3):
            for j in range(3):
                img = imgs[n].detach().cpu()
                img = TTF.to_pil_image(img)
                axs[i, j].imshow(np.asarray(img))
                axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                n += 1
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(save)
    elif imgs.shape[0] == 4:
        fig, axs = plt.subplots(2, 2, figsize=(6, 6), dpi=100)
        n = 0
        for i in range(2):
            for j in range(2):
                img = imgs[n].detach().cpu()
                img = TTF.to_pil_image(img)
                axs[i, j].imshow(np.asarray(img))
                axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                n += 1
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(save)
    elif imgs.shape[0] == 1:
        img = imgs[0].detach().cpu().numpy().transpose(1, 2, 0)
        io.imsave(save, img)
    else:
        raise NotImplementedError

class CycleTrainLoop(TrainLoop):
    
    def __init__(self, crop_size, viz_interval, viz_data, cycle=10000, warmup=2000, *args, **kwargs):
        self.crop_size = crop_size # int
        self.viz_interval = viz_interval
        self.viz_data = viz_data
        super().__init__(*args, **kwargs)
        self.model_viz = self.model
        # add scheduler
        if cycle and warmup:
            self.scheduler = CosineAnnealingWarmupRestarts(self.opt,
                                          first_cycle_steps=cycle,
                                          cycle_mult=1.0,
                                          max_lr=self.lr,
                                          min_lr=0,
                                          warmup_steps=warmup,
                                          gamma=1.0)
    
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        # self._anneal_lr()
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        print(f"Current lr: {self.opt.param_groups[0]['lr']}")
        self.log_step()

    def run_loop(self):
        while (
            self.step + self.resume_step < self.max_iterations
        ):
            batch, cond = next(self.data)
            cond = self.preprocess_input(cond) # bchw
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step > 0 and self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.step % self.viz_interval == 0:
                self.visualize()
            self.step += 1
        # Visualize and save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.viz_interval != 0:
            self.visualize()
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        
            
    # visualize the generated samples by current model
    def visualize(self):
        
        print('*' * 10, 'Visualize', '*' * 10)
        params = self.mp_trainer.master_params
        state_dict = self.mp_trainer.master_params_to_state_dict(params)
        self.model_viz.load_state_dict(state_dict)
        self.model_viz.to(dist_util.dev())
        if self.use_fp16:
            self.model_viz.convert_to_fp16()
        self.model_viz.eval()
        print(f'Loaded model state of step {self.step} for visualization.')

        print(f'Sampling...')
        model_kwargs = {'y': self.viz_data[1]['label'].permute(0, 3, 1, 2), 's': 2.0}
        sample_fn = self.diffusion.p_sample_loop
        # sample_fn = self.diffusion.ddim_sample_loop
        sample = sample_fn(self.model_viz, self.viz_data[0].shape, clip_denoised=True, model_kwargs=model_kwargs, progress=True)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        print(f'sample: {sample.shape}')
        print('Finish sampling.')
            
        print('Saving visualization results...')
        real_image = (self.viz_data[0] + 1) / 2 # nchw, [0, 1]
        fake_image = sample / 255 # [0, 1]
        input_mask = self.viz_data[1]['label'].permute(0, 3, 1, 2).to(torch.float32) # nchw
        filename_real = f'real_image_{(self.step+self.resume_step):06d}.png'
        filename_fake = f'fake_image_{(self.step+self.resume_step):06d}.png'
        filename_input_mask = f'input_mask_{(self.step+self.resume_step):06d}.png'
        proced_input_mask = input_mask.clone() # nchw
        proced_input_mask[:,1:,:,:] = (proced_input_mask[:,1:,:,:] + 1) / 2
        show(real_image, os.path.join(get_blob_logdir(), filename_real)) # [0, 1]
        show(fake_image, os.path.join(get_blob_logdir(), filename_fake)) # [0, 1]
        show(proced_input_mask, os.path.join(get_blob_logdir(), filename_input_mask)) # [0, 1]
        print('*' * 10, 'Visualize', '*' * 10)

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            #### diffusion loss
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            diff_loss = (losses["loss"] * weights).mean()
            if torch.isnan(diff_loss).any():
                warn(f'Diffusion loss has NaNs!\n{diff_loss}')
            log_loss_dict(
                {k: v * weights for k, v in losses.items() if k != 'pred_xstart'}
            )

            #### combine two losses
            self.mp_trainer.backward(diff_loss)
            print(f'Diffusion loss: {diff_loss.cpu().item():.4f}')


