from config.cfg import get_cfg
from data import NeuroImagesDataModuleConfig, NsdDatasetConfig
import time

"""
Main function to run the preprocessing for train and test datasets.
"""
print("--- Starting Data Preprocessing ---")
start_time = time.time()

# --- 1. Setup Configuration ---
cfg = get_cfg(
    subjects=[1, 2, 5, 7],#1, 2, 5, 7
    averaged_trial=False, 
    cache="./cache",
    seed=42,
    vd_cache_dir="./versatile_diffusion",
    custom_infra=None
)
base_data_cfg = cfg['data']

# --- 2. Process TRAINING data ---
print("\n[1/2] Preparing TRAINING dataset...")

# Step 2a: Create the configuration dictionary for the child model
train_nsd_config_dict = {
    **base_data_cfg['nsd_dataset_config'],
    "dataset_split": "train",
    "averaged": False,
}
train_nsd_config_obj = NsdDatasetConfig(**train_nsd_config_dict)

train_kwargs = base_data_cfg.copy()
train_kwargs['nsd_dataset_config'] = train_nsd_config_obj

# Step 2c: Unpack the modified arguments. Now there is no conflict.
train_datamodule_config = NeuroImagesDataModuleConfig(**train_kwargs)

print(f"Loaded configuration for subjects: {train_datamodule_config.nsd_dataset_config.subject_ids}")
print(f"Processed data will be saved to: {train_datamodule_config.nsd_dataset_config.processed_nsddata_path}")

train_datamodule_config.build()
print("-> Training data preprocessing complete.")

# --- 3. Process TEST data ---
print("\n[2/2] Preparing TEST (validation) dataset...")

test_nsd_config_dict = {
    **base_data_cfg['nsd_dataset_config'],
    "dataset_split": "test",
    "averaged": False,
}
test_nsd_config_obj = NsdDatasetConfig(**test_nsd_config_dict)

# Apply the same fix here
test_kwargs = base_data_cfg.copy()
test_kwargs['nsd_dataset_config'] = test_nsd_config_obj
test_kwargs['test_groupbyimg'] = "unaveraged"

test_datamodule_config = NeuroImagesDataModuleConfig(**test_kwargs)

test_datamodule_config.build()
print("-> Test data preprocessing complete.")

end_time = time.time()
print(f"\n--- Preprocessing finished successfully in {end_time - start_time:.2f} seconds. ---")
print("You are now ready to run the training script.")