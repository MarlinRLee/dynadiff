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
    log_image_freq: int = field(default=1)
    compute_metrics_freq: int = field(default=5)
    num_metric_batches : int = field(default=10)
    resume_from_checkpoint: bool = field(default=True, metadata={"help": "Resume training from the last checkpoint."})

# --- Argument Parsing (Minimal for DeepSpeed and your specific script args) ---
parser = argparse.ArgumentParser(description='DeepSpeed Training Script')
# DeepSpeed arguments (required to load ds_config.json)
parser = deepspeed.add_config_arguments(parser)

parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')

args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.local_rank)
    print(f"INFO: Rank {args.local_rank} is setting device to torch.cuda.current_device(): {torch.cuda.current_device()}")
else:
    print("WARNING: CUDA not available, running on CPU.")

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

if deepspeed.comm.get_rank() == 0:
    os.makedirs(script_args.checkpoint_dir, exist_ok=True)
    os.makedirs(script_args.log_dir, exist_ok=True)
    
print(f"Saving checkpoints to: {script_args.checkpoint_dir}")
print(f"Logging to: {script_args.log_dir}")

print("Preparing data info for ALL subjects...")
data_module_config = NeuroImagesDataModuleConfig(**cfg['data'])
data_module = data_module_config.build()


val_dataloader = data_module.val_dataloader()
if deepspeed.comm.get_rank() == 0: # Only print from rank 0
    print(f"Validation dataset size: {len(data_module.eval_dataset)} samples.")
    print(f"Number of validation batches: {len(val_dataloader)}")


sample_brain_input = data_module.eval_dataset[0]["brain"]
brain_n_in_channels, brain_temp_dim = sample_brain_input.size()
if deepspeed.comm.get_rank() == 0: # Only print from rank 0
    print(f"Brain input dimensions: {brain_n_in_channels} channels, {brain_temp_dim} temporal dimension.")

print("Instantiating model...")
vd_config = VersatileDiffusionConfig(**cfg['versatilediffusion_config'])

model = VersatileDiffusion(
    config=vd_config,
    brain_n_in_channels = brain_n_in_channels,#make sure these match the sample ones
    brain_temp_dim = brain_temp_dim
)

trainable_params = model.collect_parameters()


model_engine, optimizer, deepspeed_dataloader, lr_scheduler = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=trainable_params,
    training_data=data_module.train_dataset 
)

start_epoch = 1
if script_args.resume_from_checkpoint:
    load_path, client_sd = model_engine.load_checkpoint(script_args.checkpoint_dir, load_module_strict=False)
    if load_path is None:
        print(f"Could not find a checkpoint to resume from in {script_args.checkpoint_dir}, starting from scratch.")
    else:
        # The load_checkpoint function returns a dictionary with metadata.
        # 'epoch' is a common key, but check what your saving logic provides.
        start_epoch = client_sd.get('epoch', 1) + 1
        print(f"Resuming training from epoch {start_epoch}")

if deepspeed.comm.get_rank() == 0:
    # for name, param in model_engine.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable (DeepSpeed): {name}, Shape: {param.shape}")
    print(f"Number of trainable parameters (DeepSpeed): {sum(p.numel() for p in model_engine.parameters() if p.requires_grad)}")
    print(f"DeepSpeed Optimizer: {optimizer}")
    print(f"DeepSpeed DataLoader batch size: {deepspeed_dataloader.batch_size}")


# Get total_num_steps and micro_batch_size_per_gpu from DeepSpeed engine for local loop logic/logging
max_training_steps = lr_scheduler.total_num_steps # Get from scheduler

# Calculate epochs based on DeepSpeed's steps
total_steps_per_epoch = len(deepspeed_dataloader) // model_engine.gradient_accumulation_steps()
num_epochs_from_steps = max_training_steps // total_steps_per_epoch

if max_training_steps % total_steps_per_epoch != 0:
    num_epochs_from_steps += 1

if deepspeed.comm.get_rank() == 0:
    print(f"Calculated approximate epochs to reach {max_training_steps} steps: {num_epochs_from_steps}")


print("Starting training...")
for epoch in range(1, num_epochs_from_steps + 1):#change to 1 once I know model saving works
    model_engine.train()
    total_train_loss = 0

    for batch_idx, batch in enumerate(tqdm(deepspeed_dataloader, desc=f"Epoch {epoch}/{num_epochs_from_steps} (Training)", disable=(deepspeed.comm.get_rank() != 0))):
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

    # --- Validation Loop --
    torch.distributed.barrier()
    model_engine.eval()
    total_val_loss_tensor = torch.tensor(0.0).to(model_engine.device)
    local_generated_images = []
    local_ground_truth_images = []
    wandb_images = []


    is_metrics_epoch = (epoch % script_args.compute_metrics_freq == 0) or (epoch == num_epochs_from_steps) or (epoch == 1)
    is_log_image_epoch = (epoch % script_args.log_image_freq == 0) or (epoch == num_epochs_from_steps) or (epoch == 1)
    with torch.no_grad():
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch} (Validation)", disable=(deepspeed.comm.get_rank() != 0))
        for batch_idx, batch in enumerate(val_pbar):
            brain = batch["brain"].to(model_engine.device).half()
            subject_idx = batch["subject_idx"].to(model_engine.device)
            img = batch["img"].to(model_engine.device)

            model_output = model_engine(
                brain=brain,
                subject_idx=subject_idx,
                img=img,
                is_img_gen_mode=False,
            )

            current_val_loss = model_output.losses["diffusion"]
            total_val_loss_tensor += current_val_loss
            #log_image_freq, compute_metrics_freq
            if (is_metrics_epoch and batch_idx < script_args.num_metric_batches) or\
                (is_log_image_epoch and batch_idx == 0 and deepspeed.comm.get_rank() == 0):
                generated_output = model_engine(
                    brain=brain,
                    subject_idx=subject_idx,
                    img=img,
                    is_img_gen_mode=True,
                )
                gen_batch_np = (generated_output.image.permute(0, 2, 3, 1) * 255.0).to(torch.uint8).cpu().numpy()
                gt_batch_np = (img.permute(0, 2, 3, 1) * 255.0).to(torch.uint8).cpu().numpy()
                gen_pil_list = [Image.fromarray(arr) for arr in gen_batch_np]
                gt_pil_list = [Image.fromarray(arr) for arr in gt_batch_np]
                # If calculating metrics for the whole dataset, each rank collects its own images.
                if (is_metrics_epoch and batch_idx < script_args.num_metric_batches):
                    local_generated_images.extend(gen_pil_list)
                    local_ground_truth_images.extend(gt_pil_list)

                # If logging images, only rank 0 prepares its first batch for wandb.
                if deepspeed.comm.get_rank() == 0 and (is_log_image_epoch and batch_idx == 0):
                    for i, (gen_pil, gt_pil) in enumerate(zip(gen_pil_list, gt_pil_list)):
                        wandb_images.append(wandb.Image(gen_pil, caption=f"Generated Image Batch {batch_idx+1}, Image {i+1} (Epoch {epoch})"))
                        wandb_images.append(wandb.Image(gt_pil, caption=f"Ground Truth Image Batch {batch_idx+1}, Image {i+1} (Epoch {epoch})"))



    torch.distributed.all_reduce(total_val_loss_tensor, op=torch.distributed.ReduceOp.SUM)

    final_generated_images = []
    final_ground_truth_images = []

    if is_metrics_epoch:
        gathered_gen_images = [None] * deepspeed.comm.get_world_size()
        gathered_gt_images = [None] * deepspeed.comm.get_world_size()

        torch.distributed.gather_object(local_generated_images, gathered_gen_images if deepspeed.comm.get_rank() == 0 else None, dst=0)
        torch.distributed.gather_object(local_ground_truth_images, gathered_gt_images if deepspeed.comm.get_rank() == 0 else None, dst=0)

        if deepspeed.comm.get_rank() == 0:
            # On rank 0, flatten the list of lists into a single list for metrics.
            final_generated_images = [item for sublist in gathered_gen_images for item in sublist]
            final_ground_truth_images = [item for sublist in gathered_gt_images for item in sublist]



    if deepspeed.comm.get_rank() == 0:
        num_val_samples = len(data_module.eval_dataset)
        avg_epoch_val_loss = total_val_loss_tensor.item() / num_val_samples
        print(f"Epoch {epoch} | Average Validation Loss: {avg_epoch_val_loss}")
        wandb.log({"val/epoch_loss": avg_epoch_val_loss, "epoch": epoch})
        
        if is_log_image_epoch:
                wandb.log({"val/generated_vs_ground_truth_images": wandb_images, "epoch": epoch})

        if is_metrics_epoch:
            if not final_generated_images:
                print("Warning: Metrics epoch but no images were generated/gathered.")
            else:
                image_gen_metrics = compute_image_generation_metrics(
                    preds=final_generated_images,
                    trues=final_ground_truth_images,
                    device=model_engine.device
                )
                wandb.log({f"val/image_metrics/{k}": v for k, v in image_gen_metrics.items()}, step=epoch)
                print(f"Epoch {epoch} Image Generation Metrics: {image_gen_metrics}")

    
    model_engine.save_checkpoint(script_args.checkpoint_dir, epoch)

    torch.distributed.barrier()

if deepspeed.comm.get_rank() == 0:
    print("Training complete!")
    wandb.finish()