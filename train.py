import torch
import os
import lightning.pytorch as pl
from config.cfg import get_cfg
from model.models import VersatileDiffusionConfig, VersatileDiffusion
from metrics.image_metrics import compute_image_generation_metrics
from data import NeuroImagesDataModuleConfig
from dataclasses import dataclass, field
from tqdm import tqdm
import torch.optim as optim
import wandb
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@dataclass
class TrainingConfig:
    subjects: list[int] = field(default_factory=list)
    learning_rate: float = field(default=1e-3)
    weight_decay: float = field(default=0.01) # Added weight decay
    beta1: float = field(default=0.9) # Added beta1
    beta2: float = field(default=0.999) # Added beta2
    epochs: int = field(default=10) # This will be overridden by max_training_steps for better control
    batch_size: int = field(default=40)
    max_training_steps: int = field(default=60000) # Directly set for 60k steps
    warmup_steps: int = field(default=1000) # Added for linear warmup
    cache: str = field(default="./cache")
    seed: int = field(default=42)
    vd_cache_dir: str = field(default="./versatile_diffusion")
    checkpoint_dir: str = field(default="./training_checkpoints")
    log_dir: str = field(default="./lightning_logs")
    log_interval: int = field(default=10)
    accelerator: str = field(default="cuda") # Typically handled by DeepSpeed directly or Lightning's DeepSpeed strategy
    devices: int = field(default=8) # Set to 8 A100 GPUs
    precision: str = field(default="16-mixed") # Set to float16 precision
    num_eval_images: int = field(default=5)
    eval_freq: int = field(default=1) # Evaluate every epoch (or can be changed to steps later)


args = TrainingConfig(subjects = [1, 2, 5, 7])

pl.seed_everything(args.seed)

# --- Initialize Weights & Biases ---
wandb.init(
    project="versatile-diffusion-fmri", # Replace with your project name
    config=args, # Logs all attributes of the TrainingConfig dataclass
    dir=args.log_dir, # Store wandb files in the same log directory
    entity="ramnie-university-of-minnesota-twin-cities"
)


wandb.run.name = f"training_s{'_'.join(map(str, args.subjects))}_lr{args.learning_rate}_epochs{args.epochs}"

cfg = get_cfg(
    subjects=args.subjects,
    averaged_trial=False, # Assuming training is typically on unaveraged trials
    cache=args.cache,
    seed=args.seed,
    vd_cache_dir=args.vd_cache_dir,
    custom_infra=None,
)

os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
print(f"Saving checkpoints to: {args.checkpoint_dir}")
print(f"Logging to: {args.log_dir}")

print("Preparing data info for ALL subjects...")
    
data_module_config = NeuroImagesDataModuleConfig(**cfg['data'])
data_module = data_module_config.build()

train_dataloader = data_module.train_dataloader()
print(f"Training dataset size: {len(data_module.train_dataset)} samples.")
print(f"Number of training batches: {len(train_dataloader)}")

val_dataloader = data_module.val_dataloader()
print(f"Validation dataset size: {len(data_module.eval_dataset)} samples.")
print(f"Number of validation batches: {len(val_dataloader)}")

sample_brain_input = data_module.eval_dataset[0]["brain"]
brain_n_in_channels, brain_temp_dim = sample_brain_input.size()
print(f"Brain input dimensions: {brain_n_in_channels} channels, {brain_temp_dim} temporal dimension.")

print("Instantiating model...")
vd_config = VersatileDiffusionConfig(**cfg['versatilediffusion_config'])

model = VersatileDiffusion(
    config=vd_config,
    brain_n_in_channels=brain_n_in_channels,
    brain_temp_dim=brain_temp_dim,
).to(device)

if False:#to load a checkpoint
    checkpoint_file = "model_epoch_1.pt"
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)

    print(f"Loading model from checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    # Load the state_dict into the model
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully!")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}, Shape: {param.shape}")

# Define optimizer
trainable_params = model.collect_parameters()
print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
optimizer = optim.AdamW(
    trainable_params,
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    betas=(args.beta1, args.beta2)
)
print(f"Optimizer: {optimizer}")

total_steps_per_epoch = len(train_dataloader)
num_epochs_from_steps = args.max_training_steps // total_steps_per_epoch
if args.max_training_steps % total_steps_per_epoch != 0:
    num_epochs_from_steps += 1 # Ensure all steps are covered
print(f"Calculated approximate epochs to reach {args.max_training_steps} steps: {num_epochs_from_steps}")


# Learning Rate Scheduler
def lr_lambda(current_step: int):
    if current_step < args.warmup_steps:
        return float(current_step) / float(max(1, args.warmup_steps))
    progress = float(current_step - args.warmup_steps) / float(max(1, args.max_training_steps - args.warmup_steps))
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
print(f"Learning rate scheduler: {scheduler}")

# For mixed precision training, you would typically use torch.cuda.amp.GradScaler
# or a framework like DeepSpeed/Lightning.
scaler = torch.amp.GradScaler(enabled=(args.precision == "16-mixed"))

print("Starting training...")
global_step = 0


print("Starting training...")
for epoch in range(1, args.epochs + 1):
    model.train()

    if global_step >= args.max_training_steps:
        print(f"Reached {args.max_training_steps} training steps. Stopping training.")
        break

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs} (Training)")):
        if global_step >= args.max_training_steps:
            print(f"Reached {args.max_training_steps} training steps. Stopping training in batch loop.")
            break

        optimizer.zero_grad()

        brain = batch["brain"].to(device)
        subject_idx = batch["subject_idx"].to(device)
        img = batch["img"].to(device) # Stimuli given to patient

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(args.precision == "16-mixed")):
            model_output = model(
                brain=brain,
                subject_idx=subject_idx,
                img=img,
                is_img_gen_mode=False,
            )

            current_loss = model_output.losses["diffusion"]

        scaler.scale(current_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        wandb.log(
            {
                "train/batch_loss": current_loss.item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            },
            step=global_step
        )
        global_step += 1
    
    # --- Validation Loop ---
    model.eval()
    total_val_loss = 0
    generated_images = []
    ground_truth_images = []
    wandb_images = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch}/{args.epochs} (Validation)")):
            brain = batch["brain"].to(device)
            subject_idx = batch["subject_idx"].to(device)
            img = batch["img"].to(device)

            model_output = model(
                brain=brain,
                subject_idx=subject_idx,
                img=img,
                is_img_gen_mode=False,
            )

            current_val_loss = model_output.losses["diffusion"]

            total_val_loss += current_val_loss.item()

            if (epoch + 1 )% args.eval_freq == 0 and batch_idx < args.num_eval_images:
                generated_output = model(
                    brain=brain,
                    subject_idx=subject_idx,
                    is_img_gen_mode=True,
                )
                # Convert generated tensor to PIL Image
                gen_img_tensor = generated_output.image[0].cpu()
                gen_img_np = (gen_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                gen_pil_img = Image.fromarray(gen_img_np)

                # Convert ground truth image tensor to PIL Image
                gt_img_tensor = img[0].cpu()
                gt_img_np = (gt_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                gt_pil_img = Image.fromarray(gt_img_np)

                generated_images.append(gen_pil_img)
                ground_truth_images.append(gt_pil_img)

                wandb_images.append(wandb.Image(gen_pil_img, caption=f"Generated Image {batch_idx+1} (Epoch {epoch})"))
                wandb_images.append(wandb.Image(gt_pil_img, caption=f"Ground Truth Image {batch_idx+1} (Epoch {epoch})"))

    if (epoch + 1 )% args.eval_freq == 0:
        wandb.log({"val/generated_vs_ground_truth_images": wandb_images, "epoch": epoch})

    avg_epoch_val_loss = total_val_loss / len(val_dataloader)

    wandb.log({"val/epoch_loss": avg_epoch_val_loss, "epoch": epoch})

    if (epoch + 1 )% args.eval_freq == 0:
        # Compute image generation metrics
        print("Computing image generation metrics...")
        image_gen_metrics = compute_image_generation_metrics(
            preds=generated_images,
            trues=ground_truth_images,
            device=device
        )
        wandb.log({f"val/image_metrics/{k}": v for k, v in image_gen_metrics.items()})
        print(f"Epoch {epoch} Image Generation Metrics: {image_gen_metrics}")
        

    checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

print("Training complete!")