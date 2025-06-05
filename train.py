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
    learning_rate: float = field(default=1e-4)
    epochs: int = field(default=10)
    batch_size: int = field(default=32)
    cache: str = field(default="./cache")
    seed: int = field(default=42)
    vd_cache_dir: str = field(default="./versatile_diffusion")
    checkpoint_dir: str = field(default="./training_checkpoints")
    log_dir: str = field(default="./lightning_logs")
    log_interval: int = field(default=10)
    accelerator: str = field(default="cuda")
    devices: int = field(default="auto") 
    precision: str = field(default="16-mixed")
    num_eval_images: int = field(default=5)
    eval_freq: int = field(default=1)

args = TrainingConfig(subjects = [1,2,5])#,7

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

if True:
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
optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)
print(f"Optimizer: {optimizer}")

print("Starting training...")
for epoch in range(1, args.epochs + 1):
    model.train()
    total_train_loss = 0

    """
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs} (Training)")):
        optimizer.zero_grad()

        brain = batch["brain"].to(device)
        subject_idx = batch["subject_idx"].to(device)
        img = batch["img"].to(device) # Stimuli given to patient

        # Forward pass: compute outputs and losses
        model_output = model(
            brain=brain,
            subject_idx=subject_idx,
            img=img,
            is_img_gen_mode=False,
        )

        # Sum up the losses from the model_output
        current_loss = model_output.losses["diffusion"]

        current_loss.backward()

        # Update model parameters
        optimizer.step()

        wandb.log({"train/batch_loss": current_loss.item()}, step=(epoch-1)*len(train_dataloader) + batch_idx)
    """
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

            if (epoch + 1 )% args.eval_freq == 0 and batch_idx < 10:
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
        wandb.log({f"val/image_metrics/{k}": v for k, v in image_gen_metrics.items()}, "epoch": epoch)
        print(f"Epoch {epoch} Image Generation Metrics: {image_gen_metrics}")
        

    checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

print("Training complete!")