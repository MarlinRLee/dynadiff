import torch
import os
import glob
import argparse
from dataclasses import dataclass, field

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy

# --- Project-specific imports ---
from config.cfg import get_cfg 
from model.models import VersatileDiffusion, VersatileDiffusionConfig
from data import NeuroImagesDataModuleConfig

@dataclass
class TrainingConfig:
    # Data and Path Configs
    subjects: list[int] = field(default_factory=lambda: [1, 2, 5, 7])
    cache: str = field(default="./cache")
    vd_cache_dir: str = field(default="./versatile_diffusion")
    checkpoint_dir: str = field(default="./training_checkpoints")
    log_dir: str = field(default="./lightning_logs")

    averaged_trial: bool = field(default=False)
    
    # Training Control
    max_epochs: int = field(default=100)
    seed: int = field(default=42)
    resume_from_checkpoint: str = field(default=None, metadata={"help": "Path to a specific checkpoint to resume from."}) # type: ignore
    auto_resume: bool = field(default=True, metadata={"help": "Automatically resume from the last checkpoint in checkpoint_dir."})

    # Validation and Logging Control
    log_image_freq: int = field(default=1, metadata={"help": "Log images every N epochs."})
    compute_metrics_freq: int = field(default=1, metadata={"help": "Compute expensive metrics every N epochs."})#5
    num_metric_batches: int = field(default=20, metadata={"help": "Number of validation batches to use for computing metrics."})
    num_eval_images: int = field(default=10, metadata={"help": "Number of image pairs to log to W&B."})
    
    # W&B Project
    wandb_project: str = field(default="versatile-diffusion-fmri_temp")

def find_last_checkpoint(checkpoint_dir: str):
    """Finds the latest-saved .ckpt file in a directory."""
    if not os.path.isdir(checkpoint_dir):
        return None
    
    # Find the 'last.ckpt' symlink, which is the most reliable way
    last_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        # Read the symlink to get the actual file path
        return os.path.realpath(last_ckpt_path)

    # Fallback to finding the most recently modified file if 'last.ckpt' doesn't exist
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if not list_of_files:
        return None
        
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def main():
    script_args = TrainingConfig()
    pl.seed_everything(script_args.seed)

    # --- 1. Setup Data ---
    cfg = get_cfg(
        subjects=script_args.subjects,
        averaged_trial=script_args.averaged_trial,
        cache=script_args.cache,
        seed=script_args.seed,
        vd_cache_dir=script_args.vd_cache_dir,
        custom_infra=None
    )
    
    data_module_config = NeuroImagesDataModuleConfig(**cfg['data'])
    data_module = data_module_config.build()
    data_module.setup(stage="fit")
    sample_brain = data_module.train_dataset[0]['brain']
    brain_n_in_channels, brain_temp_dim = sample_brain.shape

    # --- 2. Find Checkpoint for Resumption ---
    ckpt_path = script_args.resume_from_checkpoint
    if not ckpt_path and script_args.auto_resume:
        ckpt_path = find_last_checkpoint(script_args.checkpoint_dir)

    # --- 3. Setup Model ---
    # Load model-specific configurations from your YAML
    vd_config = VersatileDiffusionConfig(**cfg['versatilediffusion_config'])

    model_args = {
        "config": vd_config,
        "brain_n_in_channels": brain_n_in_channels,
        "brain_temp_dim": brain_temp_dim,
        "log_image_freq": script_args.log_image_freq,
        "compute_metrics_freq": script_args.compute_metrics_freq,
        "num_metric_batches": script_args.num_metric_batches,
        "num_eval_images": script_args.num_eval_images,
        "full_validate": False
    }

    model = VersatileDiffusion(**model_args)

    state_dict = torch.load("/scratch.global/lee02328/dynadiff/merged_dynadiff_padded.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)

    # --- 4. Setup Logging & Checkpointing ---
    wandb_logger = WandbLogger(
        project=script_args.wandb_project,
        save_dir=script_args.log_dir,
    )
    
    # Configure checkpointing to save the best models and the last model
    checkpoint_callback = ModelCheckpoint(
        dirpath=script_args.checkpoint_dir,
        filename='vd-fmri-{epoch:02d}-{val/loss:.2f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min',
        save_last=True
    )

    # --- 5. Configure the Trainer ---
    strategy = DeepSpeedStrategy(
        stage=2, 
        offload_optimizer=True,
        offload_parameters=False
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=strategy,
        max_epochs=script_args.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        #log_every_n_steps=10,
        enable_checkpointing=True,
    )
    
    print("hi")
    # --- 6. Start Training ---
    #train_loader = data_module.train_dataloader()
    trainer.fit(model=model, datamodule=data_module)#, ckpt_path=ckpt_path
    #fit

if __name__ == "__main__":
    main()