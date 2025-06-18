# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
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
from PIL import Image
import os
import random

import lightning.pytorch as pl
from metrics.image_metrics import compute_image_generation_metrics
from deepspeed.ops.adam import DeepSpeedCPUAdam
import wandb

torch.set_float32_matmul_precision('high')

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

    learning_rate: float = 1e-4




    def build(
        self, brain_n_in_channels: int | None = None, brain_temp_dim: int | None = None
    ) -> nn.Module:
        return VersatileDiffusion(
            config=self,
            brain_n_in_channels=brain_n_in_channels,
            brain_temp_dim=brain_temp_dim,
        )


class VersatileDiffusion(pl.LightningModule):
    """End-to-end finetuning on brain signals"""

    def __init__(
        self,
        config: VersatileDiffusionConfig | None = None,
        brain_n_in_channels: int | None = None,
        brain_temp_dim: int | None = None,
        log_image_freq: int = 1,
        compute_metrics_freq: int = 5,
        num_metric_batches: int = 50,
        num_eval_images: int = 5,
        full_validate: bool = False
    ):
        super().__init__()
        config = config if config is not None else VersatileDiffusionConfig()
        self.config = config
        # This will save all hyperparameters passed to __init__ to the checkpoint
        self.save_hyperparameters(ignore=['config']) 

        self.drop_rate_clsfree = config.drop_rate_clsfree
        self.guidance_scale = 3.5
        self.diffusion_noise_offset = config.diffusion_noise_offset
        self.prediction_type = config.prediction_type
        self.noise_cubic_sampling = config.noise_cubic_sampling
        self.brain_modules_config = config.brain_modules_config
        self.training_strategy = config.training_strategy
        self.brain_n_in_channels = brain_n_in_channels
        self.brain_temp_dim = brain_temp_dim
        self.log_image_freq = log_image_freq
        self.compute_metrics_freq = compute_metrics_freq
        self.num_metric_batches = num_metric_batches
        self.num_eval_images = num_eval_images

        self.full_validate = full_validate

        self.is_metrics_epoch = False
        self.is_log_image_epoch = False

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
            print("Creating lora")
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
            print("no training unet")
            pass
        else:
            raise ValueError("config.trainable_unet_layers is unknown")
        
        self._validation_step_outputs_GT = []
        self._validation_step_outputs_GEN = []
        self._validation_step_outputs_PENULTIMATE = []

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
    
    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = DeepSpeedCPUAdam(
            trainable_params, 
            lr=1e-3, 
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        return optimizer
    
    def training_step(self, batch, batch_idx):

        brain = batch["brain"]
        subject_idx = batch["subject_idx"]
        img = batch["img"]

        model_output = self(
            brain=brain,
            subject_idx=subject_idx,
            img=img,
            is_img_gen_mode=False
        )

        loss = model_output.losses["diffusion"]
        self.log('train/diffusion_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step."""
        brain = batch["brain"]
        subject_idx = batch["subject_idx"]
        img = batch["img"]
        
        # 1. Calculate validation loss
        val_loss_output = self(brain=brain, subject_idx=subject_idx, img=img, is_img_gen_mode=False)
        val_loss = val_loss_output.losses["diffusion"]
        self.log('val/loss', val_loss, on_step=False, on_epoch=True, sync_dist=False)

        # 2. Generate images for logging/metrics
        self.is_metrics_epoch = (self.current_epoch % self.compute_metrics_freq == 0) or self.trainer.is_last_batch or self.trainer.sanity_checking
        self.is_log_image_epoch = (self.current_epoch % self.log_image_freq == 0) or self.trainer.is_last_batch or self.trainer.sanity_checking


        should_do_full_pass = self.full_validate or\
           (self.is_metrics_epoch and batch_idx < self.num_metric_batches) or \
           (self.is_log_image_epoch and batch_idx == 0)
        if should_do_full_pass:
            
            gen_output = self(brain=brain, subject_idx=subject_idx, img=img, is_img_gen_mode=True)
    
            if (gen_output.brain_embeddings and 'clip_image' in gen_output.brain_embeddings 
                and 'penultimate_state' in gen_output.brain_embeddings['clip_image']):
                pen_state = gen_output.brain_embeddings['clip_image']['penultimate_state']
                self._validation_step_outputs_PENULTIMATE.append(pen_state.cpu())
            
            # Convert to PIL for metrics and logging
            gen_batch = (gen_output.image.permute(0, 2, 3, 1) * 255.0).to(torch.uint8).cpu()
            gt_batch = (img.permute(0, 2, 3, 1) * 255.0).to(torch.uint8).cpu()

            self._validation_step_outputs_GEN.append(gen_batch)
            self._validation_step_outputs_GT.append(gt_batch)
            
        return val_loss

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch to aggregate and log."""
        if self.is_metrics_epoch or self.is_log_image_epoch:
            
            # Part 1: Each process saves its local results to a temporary file.
            temp_dir = os.path.join(self.trainer.log_dir or ".", "temp_val_files")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Define unique file paths for each process (rank)
            gen_path = os.path.join(temp_dir, f"gen_rank_{self.global_rank}.pt")
            gt_path = os.path.join(temp_dir, f"gt_rank_{self.global_rank}.pt")
            
            # Save the lists of CPU tensors
            torch.save(self._validation_step_outputs_GEN, gen_path)
            torch.save(self._validation_step_outputs_GT, gt_path)
            
            # Clear the memory on each process
            self._validation_step_outputs_GEN.clear()
            self._validation_step_outputs_GT.clear()

            # Handle penultimate states separately
            if self._validation_step_outputs_PENULTIMATE:
                penultimate_path = os.path.join(temp_dir, f"pen_rank_{self.global_rank}.pt")
                torch.save(self._validation_step_outputs_PENULTIMATE, penultimate_path)
                self._validation_step_outputs_PENULTIMATE.clear()

            # Part 2: Wait for all processes to finish writing to disk.
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # Part 3: Only the main process (rank 0) loads all files and computes metrics.
            if self.global_rank == 0:
                all_gen_batches = []
                all_gt_batches = []
                all_pen_batches = []

                # Loop through the files from all ranks
                for i in range(self.trainer.world_size):
                    gen_path_i = os.path.join(temp_dir, f"gen_rank_{i}.pt")
                    gt_path_i = os.path.join(temp_dir, f"gt_rank_{i}.pt")
                    
                    all_gen_batches.extend(torch.load(gen_path_i))
                    all_gt_batches.extend(torch.load(gt_path_i))
                    
                    # Clean up file after loading
                    os.remove(gen_path_i)
                    os.remove(gt_path_i)

                    # Load penultimate states if they exist
                    penultimate_path_i = os.path.join(temp_dir, f"pen_rank_{i}.pt")
                    if os.path.exists(penultimate_path_i):
                        all_pen_batches.extend(torch.load(penultimate_path_i))
                        os.remove(penultimate_path_i)

                # Clean up the temporary directory
                os.rmdir(temp_dir)

                # Concatenate all results in CPU RAM
                all_gen_images = torch.cat(all_gen_batches, dim=0)
                all_gt_images = torch.cat(all_gt_batches, dim=0)

                # Process penultimate states if they were collected
                if all_pen_batches:
                    gathered_penultimate_states = torch.cat(all_pen_batches, dim=0)
                    if self.is_metrics_epoch and not self.trainer.sanity_checking:
                        save_dir = os.path.join(self.trainer.log_dir or "lightning_logs", "penultimate_fmri_states")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"epoch_{self.current_epoch}.pt")
                        torch.save(gathered_penultimate_states.cpu(), save_path)
                        self.print(f"Saved penultimate fMRI states to {save_path}")

                # --- Your existing logic for logging and metrics calculation now runs on CPU tensors ---
                pil_gen_images = [Image.fromarray(img.numpy()) for img in all_gen_images]
                pil_gt_images = [Image.fromarray(img.numpy()) for img in all_gt_images]

                if self.full_validate or self.is_log_image_epoch:
                    wandb_images = []
                    for i, (gen_pil, gt_pil) in enumerate(zip(pil_gen_images[:self.num_eval_images], pil_gt_images[:self.num_eval_images])):
                        wandb_images.append(wandb.Image(gen_pil, caption=f"Generated Image {i+1} (Epoch {self.current_epoch})"))
                        wandb_images.append(wandb.Image(gt_pil, caption=f"Ground Truth {i+1} (Epoch {self.current_epoch})"))
                    if not self.trainer.sanity_checking:
                        self.logger.experiment.log({"val/generated_vs_ground_truth": wandb_images, "epoch": self.current_epoch})

                if self.full_validate or self.is_metrics_epoch:
                    print(f"Computing metrics on {len(pil_gen_images)} generated images...")
                    image_gen_metrics = compute_image_generation_metrics(
                        preds=pil_gen_images,
                        trues=pil_gt_images,
                        device=self.device
                    )
                    if not self.trainer.sanity_checking:
                        self.log_dict({f"val/image_metrics/{k}": v for k, v in image_gen_metrics.items()}, sync_dist=False)
                    print(f"Epoch {self.current_epoch} Image Generation Metrics: {image_gen_metrics}")
        else:
                        # Clear the memory on each process
            self._validation_step_outputs_GEN.clear()
            self._validation_step_outputs_GT.clear()
    def on_save_checkpoint(self, checkpoint: tp.Dict[str, tp.Any]) -> None:
        """
        Called by Lightning before saving a checkpoint.
        We use this to remove the VAE weights from the state_dict,
        as it is frozen and we don't need to save it.
        """
        # Keys are typically like 'state_dict', 'optimizer_states', 'epoch', etc.
        if 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
            # Find all keys belonging to the VAE and remove them
            keys_to_remove = [k for k in model_state_dict.keys() if k.startswith("vae.")]
            if keys_to_remove:
                print(f"INFO: Removing {len(keys_to_remove)} VAE keys from checkpoint before saving.")
                for k in keys_to_remove:
                    del model_state_dict[k]

    def load_state_dict(self, state_dict: tp.Dict[str, tp.Any], strict: bool = True):
        """
        Override the default load_state_dict to use strict=False.
        This allows us to load our partial checkpoints (which are missing VAE weights)
        without raising a "Missing keys" error. The VAE will keep its pretrained weights
        from the initial loading in __init__.
        """
        print("INFO: Custom load_state_dict called. Loading checkpoint with strict=False.")
        # The 'super()' call refers to the parent LightningModule's method
        super().load_state_dict(state_dict, strict=False)

def set_requires_grad(module, value):
    for n, p in module.named_parameters():
        p.requires_grad = value

