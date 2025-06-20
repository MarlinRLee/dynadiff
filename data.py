# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import typing as tp
from collections import defaultdict
from pathlib import Path

import nibabel
import nilearn.signal
import numpy as np
import pandas as pd
import pydantic
import torch
from exca.map import MapInfra
from lightning.pytorch import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader
import os
import concurrent.futures
import multiprocessing
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

TR_s = 4 / 3

MAX_FMRI_FEATURES = 16724
class NsdDataset(torch.utils.data.Dataset):
    def __init__(self, processed_data_paths: list[dict]):
        self.processed_data_paths = processed_data_paths

    def __len__(self):
        return len(self.processed_data_paths)

    def __getitem__(self, idx):
        data_info = self.processed_data_paths[idx]

        # Load preprocessed data from .npy files
        fmri = torch.from_numpy(np.load(data_info["fmri_path"])).float()

        image_np = np.load(data_info["image_path"]).astype(np.float32) / 255.0
        image = torch.from_numpy(image_np).permute(2, 0, 1)

        subject_id = torch.tensor(data_info["subject_idx"], dtype=torch.long)

        current_features, current_timepoints = fmri.shape
        if current_features < MAX_FMRI_FEATURES:
            padding_needed = MAX_FMRI_FEATURES - current_features
            # F.pad expects (padding_left, padding_right, padding_top, padding_bottom, ...)
            # For a 2D tensor (Features, Timepoints), we pad the 'Features' dimension (dim 0)
            # (0, 0) for timepoints (dim 1), (0, padding_needed) for features (dim 0)
            fmri = F.pad(fmri, (0, 0, 0, padding_needed))

        return {"brain": fmri, "img": image, "subject_idx": subject_id}


class ImageEvent(pydantic.BaseModel):
    im_fp: str | Path
    nifti_fp: str | Path
    roi_fp: str | Path
    start_idx: int
    end_idx: int
    subject_id: int


class NsdDatasetConfig(pydantic.BaseModel):
    nsddata_path: str
    subject_ids: list[int]
    offset: float = 4.6
    duration: float = 8.0
    seed: int = 42
    averaged: bool = False
    infra: MapInfra = MapInfra(version="1")
    dataset_split: tp.Literal["train", "val", "test"] = "train"
    processed_data_root: str = "processed_nsd_data"

    def create_event_list(self, subject_id) -> list[ImageEvent]:
        nsddata_path = Path(self.nsddata_path).resolve()
        test_im_ids = np.load(nsddata_path / "test_images_ids.npy")
        subject_to_14run_sessions = {
            1: (21, 38),
            5: (21, 38),
            2: (21, 30),
            7: (21, 30),
        }

        events = []
        for session in range(1, 41):
            runs = (
                range(2, 14)
                if subject_to_14run_sessions[subject_id][0]
                <= session
                <= subject_to_14run_sessions[subject_id][1]
                else range(1, 13)
            )
            for run in runs:
                run_id = f"session{session:02d}_run{run:02d}"
                path_to_df = (
                    nsddata_path
                    / f"nsddata_timeseries/ppdata/subj{subject_id:02d}/func1pt8mm/design/design_{run_id}.tsv"
                )
                if not path_to_df.exists():
                    # logger.warning(f"Design file not found for subj{subject_id:02d} {run_id}. Skipping.")
                    continue # Skip if file doesn't exist

                im_ids = pd.read_csv(path_to_df, header=None).iloc[:, 0].to_list()

                nifti_fp = (
                    nsddata_path
                    / f"nsddata_timeseries/ppdata/subj{subject_id:02d}/func1pt8mm/timeseries/timeseries_{run_id}.nii.gz"
                )
                roi_fp = (
                    nsddata_path
                    / f"nsddata/ppdata/subj{int(subject_id):02d}/func1pt8mm/roi/nsdgeneral.nii.gz"
                )
                for timestep, image_id in enumerate(im_ids):
                    if image_id == 0:
                        continue
                    
                    #and ((self.dataset_split == "test" and image_id in test_im_ids) or (self.dataset_split == "train" and image_id not in test_im_ids)):
                    #and image_id in test_im_ids:# for testing on smaller machine that can only load test data
                    if (self.dataset_split == "test" and image_id in test_im_ids) or (self.dataset_split == "train" and image_id not in test_im_ids):
                        im_fp = nsddata_path / f"nsd_stimuli/{image_id-1}.png"

                        start_idx = timestep + int(round(self.offset / TR_s))
                        end_idx = timestep + int(
                            round(self.offset + self.duration) / TR_s
                        )

                        events.append(
                            ImageEvent(
                                im_fp=im_fp,
                                nifti_fp=nifti_fp,
                                roi_fp=roi_fp,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                subject_id=subject_id
                            )
                        )
        return events

    def _process_single_event(self, event: ImageEvent, subject_cache_dir: Path):
        image_id_from_path = int(event.im_fp.stem) + 1
        unique_event_id = f"{image_id_from_path}_{event.start_idx}_{event.end_idx}"
        fmri_output_path = subject_cache_dir / f"fmri_{unique_event_id}.npy"
        image_output_path = subject_cache_dir / f"image_{unique_event_id}.npy"

        if fmri_output_path.exists() and image_output_path.exists():
            return {
                "fmri_path": str(fmri_output_path),
                "image_path": str(image_output_path),
                "subject_idx_original": event.subject_id,
                "image_id_original": image_id_from_path
            }

        # If not cached, perform the actual processing
        nifti = nibabel.load(event.nifti_fp, mmap=True)
        nifti = nifti.slicer[..., :225]
        roi_np = nibabel.load(event.roi_fp, mmap=True).get_fdata()
        nifti_data = nifti.get_fdata()[roi_np > 0]

        # z-score across run and detrend
        nifti_data = nifti_data.T
        shape = nifti_data.shape
        nifti_data = nilearn.signal.clean(
            nifti_data.reshape(shape[0], -1),
            detrend=True,
            high_pass=None,
            t_r=TR_s,
            standardize="zscore_sample",
        )
        nifti_data = nifti_data.reshape(shape).T

        # Extract the relevant fMRI segment
        fmri_segment = nifti_data[..., event.start_idx : event.end_idx]

        image = np.array(
            Image.open(event.im_fp).convert("RGB").resize((512, 512), Image.BILINEAR),
            dtype=np.uint8,
        )

        np.save(fmri_output_path, fmri_segment)
        np.save(image_output_path, image)

        return {
            "fmri_path": str(fmri_output_path),
            "image_path": str(image_output_path),
            "subject_idx_original": event.subject_id,
            "image_id_original": image_id_from_path
        }

    @infra.apply(item_uid=str, exclude_from_cache_uid=("averaged", "seed"))
    def prepare(self, events: tp.List[ImageEvent]) -> tp.Iterator[dict]:
        processed_data_root_path = Path(self.processed_data_root)

        # Group events by subject to manage subject_cache_dir more cleanly
        events_by_subject = defaultdict(list)
        for event in events:
            events_by_subject[event.subject_id].append(event)

        all_results = []
        for sub_id, sub_events in events_by_subject.items():
            subject_cache_dir = processed_data_root_path / f"subj{sub_id:02d}" / self.dataset_split
            os.makedirs(subject_cache_dir, exist_ok=True)

            with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = [executor.submit(self._process_single_event, event, subject_cache_dir) for event in sub_events]


                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    all_results.append(future.result())

        return iter(all_results)


    def build(self):
        all_processed_data_info = [] # List to store paths to processed data files

        # Map actual subject_id to a contiguous 0-indexed ID for SubjectLayers
        subject_id_map = {sub_id: i for i, sub_id in enumerate(self.subject_ids)}

        for sub_id in self.subject_ids:
            subject_events = self.create_event_list(subject_id=sub_id)

            data = self.prepare(subject_events)

            processed_event_info = list(data) # Collect all yielded results


            if self.dataset_split == "train":
                if self.averaged:
                    raise NotImplementedError
                else:
                    selected_data_info = processed_event_info
            else:
                grouped_events_by_image = defaultdict(list)
                for info in processed_event_info:
                    grouped_events_by_image[info["image_id_original"]].append(info)
                selected_data_info = [] 

                if self.averaged:
                    raise NotImplementedError
                else:
                    random.seed(self.seed + sub_id) # Use a different seed per subject for random selection
                    for img_id in grouped_events_by_image:
                        selected_data_info.append(random.choice(grouped_events_by_image[img_id]))

            # Append subject-specific selected data info to the master list
            for item_info in selected_data_info:
                # Store the mapped 0-indexed subject ID for the NsdDataset
                item_info["subject_idx"] = subject_id_map[item_info["subject_idx_original"]]
                all_processed_data_info.append(item_info)
        
        # Shuffle the combined list of data paths to mix subjects in batches
        random.seed(self.seed) # Use global seed for overall shuffle
        random.shuffle(all_processed_data_info)

        return NsdDataset(processed_data_paths=all_processed_data_info)




class NeuroImagesDataModuleConfig(pydantic.BaseModel):
    name: tp.Literal["NeuroImagesDataModuleConfig"] = "NeuroImagesDataModuleConfig"
    model_config = pydantic.ConfigDict(extra="forbid")

    nsd_dataset_config: NsdDatasetConfig

    pin_memory: bool = True
    workers: int = 0
    batch_size: int

    test_groupbyimg: tp.Literal["averaged", "unaveraged"] | None = None

    def build(
        self,
    ) -> LightningDataModule:
        data_module = NeuroImagesDataModule(
            config=self,
        )
        data_module.setup()
        return data_module


class NeuroImagesDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config if config is not None else NeuroImagesDataModuleConfig()
        self.nsd_dataset_config = config.nsd_dataset_config
        self.train_nsd_dataset_config = self.config.nsd_dataset_config.copy()


        self.batch_size = config.batch_size
        self.workers = config.workers
        self.pin_memory = config.pin_memory
        
        self.test_groupbyimg = config.test_groupbyimg

        self.train_dataset = None
        self.eval_dataset = None # This will be used for validation and testing

    def setup(self, stage: tp.Optional[str] = None):
            all_subjects = [1, 2, 5, 7]
            # Logic to prepare datasets for different stages
            if stage == "fit" or stage is None: # 'fit' is for training
                #print("Preparing training dataset...")
                # Modify the config for training
                train_config = self.nsd_dataset_config.copy(
                    update={"subject_ids": all_subjects, "averaged": False, "dataset_split": "train"}
                )
                self.train_dataset = train_config.build()
                #print(f"Number of samples in training dataset across all subjects: {len(self.train_dataset)}")
            if stage == "validate" or stage == "test" or stage is None:
                    # Create a config for validation/test data (e.g., using test split for all subjects)
                    eval_config = self.nsd_dataset_config.copy(
                        update={"subject_ids": all_subjects, "dataset_split": "test"} # Use test split for eval
                    )
                    if self.test_groupbyimg == "averaged":
                        eval_config.averaged = True
                    elif self.test_groupbyimg == "unaveraged":
                        eval_config.averaged = False
                    self.eval_dataset = eval_config.build()
                    #print(f"Number of samples in evaluation/test dataset across all subjects: {len(self.eval_dataset)}")


    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            sampler=sampler,
            drop_last=True,
        )

    def val_dataloader(self):
        sampler = DistributedSampler(self.eval_dataset, shuffle=False)
        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            sampler=sampler,
            drop_last=False,
        )

    def test_dataloader(self):
        return self.val_dataloader()