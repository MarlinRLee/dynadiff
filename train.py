import torch
import os
import lightning.pytorch as pl
from config.cfg import get_cfg
from model.models import VersatileDiffusionConfig, VersatileDiffusion
from metrics.image_metrics import compute_image_generation_metrics
from data import NeuroImagesDataModuleConfig
from dataclasses import dataclass, field
from tqdm import tqdm
import wandb
import numpy as np
from PIL import Image

# Import DeepSpeed
import deepspeed
from deepspeed.accelerator import get_accelerator
import argparse 


print(f"DeepSpeed is initializing...")

@dataclass
class TrainingConfig:
    subjects: list[int] = field(default_factory=lambda: [1,2,5,7])
    cache: str = field(default="./cache")
    seed: int = field(default=42)
    vd_cache_dir: str = field(default="./versatile_diffusion")
    checkpoint_dir: str = field(default="./training_checkpoints")
    log_dir: str = field(default="./lightning_logs")
    log_interval: int = field(default=10)
    num_eval_images: int = field(default=5)
    eval_freq: int = field(default=1)

# --- Argument Parsing (Minimal for DeepSpeed and your specific script args) ---
parser = argparse.ArgumentParser(description='DeepSpeed Training Script')
# DeepSpeed arguments (required to load ds_config.json)
parser = deepspeed.add_config_arguments(parser)

parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')

args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.local_rank)
    print(f"Process {args.local_rank}: Set CUDA device to {args.local_rank}")
else:
    print(f"CUDA not available. Running on CPU (Process {args.local_rank})")

deepspeed.init_distributed(dist_backend='nccl')

script_args = TrainingConfig()

# Only seed on rank 0, DeepSpeed handles distributed seeding internally
if deepspeed.comm.get_rank() == 0:
    pl.seed_everything(script_args.seed)
    wandb.init(project="versatile-diffusion-fmri", config=vars(script_args)) # versatile-diffusion-fmri



cfg = get_cfg(
    subjects=script_args.subjects,
    averaged_trial=False,
    cache=script_args.cache,
    seed=script_args.seed,
    vd_cache_dir=script_args.vd_cache_dir,
    custom_infra=None,
)

os.makedirs(script_args.checkpoint_dir, exist_ok=True)
os.makedirs(script_args.log_dir, exist_ok=True)
print(f"Saving checkpoints to: {script_args.checkpoint_dir}")
print(f"Logging to: {script_args.log_dir}")

print("Preparing data info for ALL subjects...")
data_module_config = NeuroImagesDataModuleConfig(**cfg['data'])
data_module = data_module_config.build()

train_dataloader = data_module.train_dataloader()
if deepspeed.comm.get_rank() == 0: # Only print from rank 0
    print(f"Training dataset size: {len(data_module.train_dataset)} samples.")
    print(f"Number of training batches: {len(train_dataloader)}")

val_dataloader = data_module.val_dataloader()
if deepspeed.comm.get_rank() == 0: # Only print from rank 0
    print(f"Validation dataset size: {len(data_module.eval_dataset)} samples.")
    print(f"Number of validation batches: {len(val_dataloader)}")


if deepspeed.comm.get_rank() == 0: # Only print from rank 0
    sample_brain_input = data_module.eval_dataset[0]["brain"]
    brain_n_in_channels, brain_temp_dim = sample_brain_input.size()
    print(f"Brain input dimensions: {brain_n_in_channels} channels, {brain_temp_dim} temporal dimension.")

print("Instantiating model...")
vd_config = VersatileDiffusionConfig(**cfg['versatilediffusion_config'])

model = VersatileDiffusion(
    config=vd_config,
    brain_n_in_channels=16724,#make sure these match the sample ones
    brain_temp_dim=6
)

trainable_params = model.collect_parameters()

# Initialize DeepSpeed - pass the `args` object from argparse, which contains ds_config path
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    args=args, # Pass the parsed args object; DeepSpeed will extract its config path
    model=model,
    model_parameters=trainable_params,
    training_data=data_module.train_dataset
)

if deepspeed.comm.get_rank() == 0:
    # for name, param in model_engine.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable (DeepSpeed): {name}, Shape: {param.shape}")
    print(f"Number of trainable parameters (DeepSpeed): {sum(p.numel() for p in model_engine.parameters() if p.requires_grad)}")
    print(f"DeepSpeed Optimizer: {optimizer}")

# Get total_num_steps and micro_batch_size_per_gpu from DeepSpeed engine for local loop logic/logging
max_training_steps = lr_scheduler.total_num_steps # Get from scheduler

# Calculate epochs based on DeepSpeed's steps
total_steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
num_epochs_from_steps = max_training_steps // total_steps_per_epoch

if max_training_steps % total_steps_per_epoch != 0:
    num_epochs_from_steps += 1

if deepspeed.comm.get_rank() == 0:
    print(f"Calculated approximate epochs to reach {max_training_steps} steps: {num_epochs_from_steps}")


print("Starting training...")
global_step = 0
for epoch in range(1, num_epochs_from_steps + 1):
    model_engine.train()
    total_train_loss = 0

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs_from_steps} (Training)", disable=(deepspeed.comm.get_rank() != 0))):

        brain = batch["brain"].to(model_engine.device)
        subject_idx = batch["subject_idx"].to(model_engine.device)
        img = batch["img"].to(model_engine.device)

        brain = brain.half()

        model_output = model_engine(
            brain=brain,
            subject_idx=subject_idx,
            img=img,
            is_img_gen_mode=False,
        )

        current_loss = model_output.losses["diffusion"]

        model_engine.backward(current_loss)
        model_engine.step()

        global_step += 1

    # --- Validation Loop ---
    if deepspeed.comm.get_rank() == 0:
        model_engine.eval()
        total_val_loss = 0
        generated_images = []
        ground_truth_images = []
        wandb_images = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch}/{num_epochs_from_steps} (Validation)")):
                brain = batch["brain"].to(model_engine.device)
                subject_idx = batch["subject_idx"].to(model_engine.device)
                img = batch["img"].to(model_engine.device)
                brain = brain.half()

                model_output = model_engine(
                    brain=brain,
                    subject_idx=subject_idx,
                    img=img,
                    is_img_gen_mode=False,
                )

                current_val_loss = model_output.losses["diffusion"]
                total_val_loss += current_val_loss.item()

                if (epoch % script_args.eval_freq == 0 or epoch == num_epochs_from_steps) and model_engine.train_micro_batch_size_per_gpu() * batch_idx < script_args.num_eval_images:
                    generated_output = model_engine(
                        brain=brain,
                        subject_idx=subject_idx,
                        is_img_gen_mode=True,
                    )
                    for i in range(generated_output.image.shape[0]):
                        gen_img_tensor = generated_output.image[i].cpu()
                        gen_img_np = (gen_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        gen_pil_img = Image.fromarray(gen_img_np)

                        gt_img_tensor = img[i].cpu() # Assuming 'img' is your ground truth batch tensor
                        gt_img_np = (gt_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        gt_pil_img = Image.fromarray(gt_img_np)

                        generated_images.append(gen_pil_img)
                        ground_truth_images.append(gt_pil_img)

                        wandb_images.append(wandb.Image(gen_pil_img, caption=f"Generated Image Batch {batch_idx+1}, Image {i+1} (Epoch {epoch})"))
                        wandb_images.append(wandb.Image(gt_pil_img, caption=f"Ground Truth Image Batch {batch_idx+1}, Image {i+1} (Epoch {epoch})"))

        avg_epoch_val_loss = total_val_loss / len(val_dataloader)

        wandb.log({"val/epoch_loss": avg_epoch_val_loss, "epoch": epoch, "global_step": global_step})
        if (epoch % script_args.eval_freq == 0 or epoch == num_epochs_from_steps):
            wandb.log({"val/generated_vs_ground_truth_images": wandb_images, "epoch": epoch, "global_step": global_step})

            print("Computing image generation metrics...")
            image_gen_metrics = compute_image_generation_metrics(
                preds=generated_images,
                trues=ground_truth_images,
                device=model_engine.device
            )
            wandb.log({f"val/image_metrics/{k}": v for k, v in image_gen_metrics.items()}, step=global_step)
            print(f"Epoch {epoch} Image Generation Metrics: {image_gen_metrics}")

        model_engine.save_checkpoint(script_args.checkpoint_dir, epoch)
        print(f"Saved DeepSpeed checkpoint for epoch {epoch} to {script_args.checkpoint_dir}")

if deepspeed.comm.get_rank() == 0:
    print("Training complete!")
    wandb.finish()