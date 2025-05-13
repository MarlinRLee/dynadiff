# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from functools import partial
from typing import List

import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import DDIMScheduler, DDPMScheduler, VersatileDiffusionDualGuidedPipeline
from diffusers.models import DualTransformer2DModel
from model.fmri_mlp import FmriMLPConfig
from model.peft_utils import add_adapter
from peft import LoraConfig
from torch import Tensor


class DiffusionOutput(tp.NamedTuple):
    image: Tensor
    losses: dict = None
    t_diffusion: Tensor = None
    brain_embeddings: dict = None


class VersatileDiffusionConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["VersatileDiffusion"] = "VersatileDiffusion"

    vd_cache_dir: str = "/fsx-brainai/marlenec/vd_cache_dir"
    in_dim: int = 15724
    num_inference_steps: int = 20


    diffusion_noise_offset: bool = False
    prediction_type: str = "epsilon"
    noise_cubic_sampling: bool = False
    drop_rate_clsfree: float = 0.1
    trainable_unet_layers: str = "lora"

    training_strategy: tp.Literal["w/_difloss", "w/o_difloss"] = "w/_difloss"


    brain_modules_config: dict[str, FmriMLPConfig] | None = None




    def build(
        self, brain_n_in_channels: int | None = None, brain_temp_dim: int | None = None
    ) -> nn.Module:
        return VersatileDiffusion(
            config=self,
            brain_n_in_channels=brain_n_in_channels,
            brain_temp_dim=brain_temp_dim,
        )


class VersatileDiffusion(nn.Module):
    """End-to-end finetuning on brain signals"""

    def __init__(
        self,
        config: VersatileDiffusionConfig | None = None,
        brain_n_in_channels: int | None = None,
        brain_temp_dim: int | None = None,
    ):
        super().__init__()
        config = config if config is not None else VersatileDiffusionConfig()
        self.config = config
        self.drop_rate_clsfree = config.drop_rate_clsfree
        self.guidance_scale = 3.5
        self.diffusion_noise_offset = config.diffusion_noise_offset
        self.prediction_type = config.prediction_type
        self.noise_cubic_sampling = config.noise_cubic_sampling
        in_dim = config.in_dim
        self.brain_modules_config = config.brain_modules_config
        self.training_strategy = config.training_strategy


        print("VD cache dir is : ", config.vd_cache_dir)
        try:
            vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
                config.vd_cache_dir
            )
        except:
            print("Downloading Versatile Diffusion to", config.vd_cache_dir)
            vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
                "shi-labs/versatile-diffusion", cache_dir=config.vd_cache_dir
            )
        vd_pipe.vae.eval()
        vd_pipe.scheduler = DDPMScheduler.from_pretrained(
            "shi-labs/versatile-diffusion",
            subfolder="scheduler",
        )
        self.num_inference_steps = 20

        text_image_ratio = (
            0.0
        )
        for name, module in vd_pipe.image_unet.named_modules():
            if isinstance(module, DualTransformer2DModel):
                module.mix_ratio = text_image_ratio
                for i, type in enumerate(("text", "image")):
                    if type == "text":
                        module.condition_lengths[i] = 77
                        module.transformer_index_for_condition[i] = (
                            1
                        )
                    else:
                        module.condition_lengths[i] = 257
                        module.transformer_index_for_condition[i] = (
                            0
                        )

        self.unet = vd_pipe.image_unet
        self.vae = vd_pipe.vae
        self.noise_scheduler = vd_pipe.scheduler
        self.eval_noise_scheduler = DDIMScheduler.from_pretrained(
            "shi-labs/versatile-diffusion",
            subfolder="scheduler",
        )
        self.unet.enable_xformers_memory_efficient_attention()


        self.vae.state_dict = partial(state_dict_cust, self=self.vae)


        if (
            self.brain_modules_config is not None
            and "blurry" in self.brain_modules_config.keys()
        ):
            num_spa_channels = 4
            orig_inpt_0_weights = self.unet.conv_in.weight
            dst_input_blocks = torch.zeros(
                (320, 4 + num_spa_channels, 3, 3), dtype=self.unet.conv_in.weight.dtype
            )
            dst_input_blocks[:, :4, :, :] = orig_inpt_0_weights
            self.unet.conv_in.weight = torch.nn.Parameter(dst_input_blocks)

        del vd_pipe


        self.has_channel_merger = False
        self.brain_modules = nn.ModuleDict()
        if self.brain_modules_config is not None:

            for brain_module_name in self.brain_modules_config.keys():
                brain_module = self.brain_modules_config[brain_module_name]
                brain_module = brain_module.build(

                    n_in_channels=brain_n_in_channels,
                    n_outputs=257 * 768,
                )
                self.brain_modules[brain_module_name] = brain_module


        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        if (
            self.brain_modules_config is not None
            and "blurry" in self.brain_modules_config.keys()
        ):
            set_requires_grad(self.unet.conv_in, True)
        if config.trainable_unet_layers == "lora":

            self.unet._hf_peft_config_loaded = False
            self.unet.peft_config = {}
            unet_lora_config = LoraConfig(
                r=4,
                lora_alpha=4,
                target_modules=[
                    "attn2.to_k",
                    "attn2.to_q",
                    "attn2.to_v",
                    "attn2.to_out.0",
                ],
            )
            add_adapter(self.unet, unet_lora_config)
        elif config.trainable_unet_layers == "no":
            pass
        else:
            raise ValueError("config.trainable_unet_layers is unknown")

    def get_condition(
        self, brain: Tensor, subject_idx: Tensor, **kwargs
    ) -> tp.Dict:
        brain_embeddings = {
            name: head(
                x=brain, subject_ids=subject_idx,
            )
            for name, head in self.brain_modules.items()
        }
        BS = brain_embeddings["clip_image"]["MSELoss"].size(0)
        brain_embeddings["clip_image"]["MSELoss"] = brain_embeddings["clip_image"][
            "MSELoss"
        ].reshape(BS, -1, 768)
        if "Clip" in brain_embeddings["clip_image"]:
            brain_embeddings["clip_image"]["Clip"] = brain_embeddings["clip_image"][
                "Clip"
            ].reshape(BS, -1, 768)

        if "blurry" in brain_embeddings:
            brain_embeddings["blurry"]["MSELoss"] = F.interpolate(
                brain_embeddings["blurry"]["MSELoss"],
                [64, 64],
                mode="bicubic",
                antialias=True,
                align_corners=False,
            )
        return brain_embeddings

    def compute_diffusion_loss(self, img, brain_embeddings: tp.Dict) -> DiffusionOutput:

        brain_clip_embeddings = brain_embeddings["clip_image"]["MSELoss"]
        brain_blurry_embeddings = None
        if "blurry" in brain_embeddings:
            brain_blurry_embeddings = brain_embeddings["blurry"]["MSELoss"]


        img = 2 * img - 1
        img = img.to(dtype=self.vae.dtype, device=self.vae.device)

        latent = (
            self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor
        )


        losses = {}
        timesteps = None

        bsz = latent.size(0)

        if self.noise_cubic_sampling:
            timesteps = torch.rand((bsz,), device=latent.device)
            timesteps = (
                1 - timesteps**3
            ) * self.noise_scheduler.config.num_train_timesteps
            timesteps = timesteps.long().to(self.noise_scheduler.timesteps.dtype)
            timesteps = timesteps.clamp(
                0, self.noise_scheduler.config.num_train_timesteps - 1
            )
        else:
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latent.device,
            )
            timesteps = timesteps.long()


        if self.diffusion_noise_offset:
            noise = torch.randn_like(latent) + 0.1 * torch.randn(
                latent.shape[0], latent.shape[1], 1, 1, device=latent.device
            )
            noise = noise.to(self.unet.dtype)
        else:
            noise = torch.randn_like(latent)
        noisy_latents = self.noise_scheduler.add_noise(latent, noise, timesteps)


        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":

            target = self.noise_scheduler.get_velocity(latent, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )



        brain_clip_embeddings = torch.cat(
            [
                torch.zeros(len(brain_clip_embeddings), 77, 768)
                .to(self.unet.dtype)
                .to(self.unet.device),
                brain_clip_embeddings,
            ],
            dim=1,
        )


        mask = (
            torch.rand(
                size=(len(brain_clip_embeddings),),
                device=brain_clip_embeddings.device,
            )
            < self.drop_rate_clsfree
        )
        if self.drop_rate_clsfree > 0.0:
            brain_clip_embeddings[mask] = 0

        if brain_blurry_embeddings != None:
            noisy_latents = torch.cat(
                (noisy_latents, brain_blurry_embeddings.to(self.unet.dtype)), dim=1
            ).to(self.unet.dtype)

            losses["blurry"] = F.mse_loss(
                brain_blurry_embeddings, latent, reduction="mean"
            )


        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=brain_clip_embeddings.to(self.unet.dtype),
            cross_attention_kwargs=None,

            return_dict=False,
        )[0]
        dif_losses = F.mse_loss(noise_pred, target, reduction="mean")
        losses["diffusion"] = dif_losses

        rec_image = noise_pred

        return DiffusionOutput(
            image=rec_image,
            losses=losses,
            t_diffusion=timesteps,
            brain_embeddings=brain_embeddings,
        )

    def forward(
        self,
        brain: Tensor,
        subject_idx: Tensor,
        img: Tensor = None,
        is_img_gen_mode: bool = False,
        seed=0,
        return_interm_noisy: bool = False,
        uncond: bool = False,
        blurry_recons_extern: Tensor = None,
        **kwargs,
    ) -> DiffusionOutput:

        brain_embeddings = self.get_condition(brain, subject_idx)
        if not is_img_gen_mode:
            if self.training_strategy == "w/o_difloss":
                return DiffusionOutput(
                    image=img, losses={}, brain_embeddings=brain_embeddings
                )
            return self.compute_diffusion_loss(img, brain_embeddings)
        else:


            return self.reconstruction_from_clipbrainimage(
                brain_embeddings,


                img_lowlevel=blurry_recons_extern,
                num_inference_steps=self.num_inference_steps,
                recons_per_sample=1,
                guidance_scale=self.guidance_scale,
                img2img_strength=0.85,
                seed=seed,
                verbose=False,
                img_variations=False,
                return_interm_noisy=return_interm_noisy,
                uncond=uncond,

            )

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def reconstruction_from_clipbrainimage(
        self,

        brain_embeddings,

        text_token=None,

        img_lowlevel=None,
        num_inference_steps=20,
        recons_per_sample=1,
        guidance_scale=3.5,
        img2img_strength=0.85,

        seed=0,


        verbose=False,
        img_variations=False,
        return_interm_noisy=False,
        uncond=False,



    ):






        brain_clip_embeddings = brain_embeddings["clip_image"]["MSELoss"]
        img_lowlevel_latent = None
        if "blurry" in brain_embeddings:
            img_lowlevel_latent = brain_embeddings["blurry"]["MSELoss"]

        batchsize = brain_clip_embeddings.size(0)
        with torch.no_grad():
            brain_recons = None
            if img_lowlevel is not None:
                img_lowlevel = img_lowlevel.to(self.unet.dtype).to(self.unet.device)

            if self.unet is not None:
                do_classifier_free_guidance = guidance_scale > 1.0

            generator = torch.Generator(device=self.unet.device)
            generator.manual_seed(seed)

            if uncond:
                input_embedding = torch.zeros_like(brain_clip_embeddings)
            else:
                input_embedding = (
                    brain_clip_embeddings
                )
            if verbose:
                print("input_embedding", input_embedding.shape)

            if text_token is not None:
                prompt_embeds = text_token.repeat(recons_per_sample, 1, 1)
            else:
                prompt_embeds = (
                    torch.zeros(len(input_embedding), 77, 768)
                    .to(self.unet.dtype)
                    .to(self.unet.device)
                )
            if verbose:
                print("prompt!", prompt_embeds.shape)

            if do_classifier_free_guidance:
                input_embedding = (
                    torch.cat([torch.zeros_like(input_embedding), input_embedding])
                    .to(self.unet.dtype)
                    .to(self.unet.device)
                )
                prompt_embeds = (
                    torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds])
                    .to(self.unet.dtype)
                    .to(self.unet.device)
                )




            if not img_variations:

                input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)





            self.noise_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps, device=self.unet.device
            )


            batch_size = (
                input_embedding.shape[0] // 2
            )

            if (
                img_lowlevel is not None
                and not self.sanity_check_blurry
                and img2img_strength != 1.0
            ):
                print("use img_lowlevel for img2img initialization")
                init_timestep = min(
                    int(num_inference_steps * img2img_strength), num_inference_steps
                )
                t_start = max(num_inference_steps - init_timestep, 0)
                timesteps = self.noise_scheduler.timesteps[t_start:]
                latent_timestep = timesteps[:1].repeat(batch_size)

                if verbose:
                    print("img_lowlevel", img_lowlevel.shape)

                img_lowlevel_embeddings = 2 * img_lowlevel - 1
                if verbose:
                    print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
                init_latents = self.vae.encode(
                    img_lowlevel_embeddings.to(self.vae.dtype)
                ).latent_dist.sample(
                    generator
                )
                init_latents = self.vae.config.scaling_factor * init_latents
                init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)
                print("init with low level")
                noise = torch.randn(
                    [recons_per_sample, 4, 64, 64],
                    device=self.unet.device,
                    generator=generator,
                    dtype=input_embedding.dtype,
                )
                init_latents = self.noise_scheduler.add_noise(
                    init_latents, noise, latent_timestep
                )
                latents = init_latents
            else:
                timesteps = self.noise_scheduler.timesteps
                latents = torch.randn(
                    [recons_per_sample * batchsize, 4, 64, 64],
                    device=self.unet.device,
                    generator=generator,
                    dtype=input_embedding.dtype,
                )
                latents = latents * self.noise_scheduler.init_noise_sigma


            interm_noisy = []
            for i, t in enumerate(timesteps):


                if img_lowlevel_latent != None:
                    latents = torch.cat((latents, img_lowlevel_latent), dim=1)

                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.noise_scheduler.scale_model_input(
                    latent_model_input, t
                )

                if verbose:
                    print("latent_model_input", latent_model_input.shape)
                if verbose:
                    print("input_embedding", input_embedding.shape)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=input_embedding,
                ).sample


                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )








                if img_lowlevel_latent != None:
                    latents = latents[:, :4]
                res_prev = self.noise_scheduler.step(noise_pred, t, latents)
                latents = res_prev.prev_sample

                if return_interm_noisy and i % 4 == 0:
                    pred_orig = res_prev.pred_original_sample
                    interm_noisy.append(
                        T.Resize((64, 64))(self.decode_latents(pred_orig).detach().cpu())
                    )

            recons = self.decode_latents(latents).detach().cpu()

            brain_recons = recons.unsqueeze(0)




        return DiffusionOutput(image=brain_recons[0], brain_embeddings=brain_embeddings)

    def collect_parameters(
        self,
    ) -> List[nn.Parameter]:
        """Return the trainable parameters of the model.

        Returns:
            model parameter_dict
        """

        model_parameters = {n: p for n, p in self.named_parameters() if p.requires_grad}

        return [v for _, v in model_parameters.items()]


def state_dict_cust(*args, destination=None, prefix="", keep_vars=False, self=None):
    return OrderedDict()


def set_requires_grad(module, value):
    for n, p in module.named_parameters():
        p.requires_grad = value
