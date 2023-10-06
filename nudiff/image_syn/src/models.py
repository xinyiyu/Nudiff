"""Wrap sdm and hovernet into one model"""
import torch as th
from torch import nn
from nudiff.image_syn.cond.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from nudiff.image_syn.cond.nn import mean_flat

class WarppedGaussianDiffusion(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            vb_out = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            terms["loss"] = vb_out["output"]
            terms['pred_xstart'] = vb_out['pred_xstart']
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), y=model_kwargs['y'])

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                vb_out = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )
                terms["vb"] = vb_out["output"]
                terms['pred_xstart'] = vb_out['pred_xstart']
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

class WarppedModel(nn.Module):
    def __init__(self, diff, unet, seg, *args, **kwargs):
        self.diff = diff
        self.unet = unet
        self.seg = seg

    def forward(self, x, t, y=None):

        if y is not None:
            model_kwargs = {'y': y}
        else:
            model_kwargs = {}
        diff_out = self.diff.p_mean_variance(self.unet, x, t, clip_denoised=True, model_kwargs=model_kwargs) # dict
        pred_xstart_ = diff_out['pred_xstart']
        pred_xstart = self.process_diff_out(pred_xstart_)
        seg_out = self.seg(pred_xstart) # dict
        return diff_out, seg_out

    def process_diff_out(self, x):
        x = (x + 1) / 2
        return x

