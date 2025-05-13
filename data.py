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

TR_s = 4 / 3


class NsdDataset(torch.utils.data.Dataset):
    def __init__(self, fmris: np.ndarray, images: np.ndarray):
        self.fmris = fmris
        self.images = images

    def __len__(self):
        return len(self.fmris)

    def __getitem__(self, idx):
        fmri = torch.from_numpy(self.fmris[idx]).float()
        image = torch.from_numpy(self.images[idx]).long().permute(2, 0, 1) / 255.0
        subject_id = torch.tensor(
            0, dtype=torch.long
        )  # Single-subject models, for compatibility
        return {"brain": fmri, "img": image, "subject_idx": subject_id}


class ImageEvent(pydantic.BaseModel):
    im_fp: str | Path
    nifti_fp: str | Path
    roi_fp: str | Path
    start_idx: int
    end_idx: int


class NsdDatasetConfig(pydantic.BaseModel):
    nsddata_path: str
    subject_id: int
    offset: float = 4.6
    duration: float = 8.0
    seed: int = 42
    averaged: bool = False
    infra: MapInfra = MapInfra(version="1")

    def create_event_list(self) -> list[ImageEvent]:
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
                if subject_to_14run_sessions[self.subject_id][0]
                <= session
                <= subject_to_14run_sessions[self.subject_id][1]
                else range(1, 13)
            )
            for run in runs:
                run_id = f"session{session:02d}_run{run:02d}"
                path_to_df = (
                    nsddata_path
                    / f"nsddata_timeseries/ppdata/subj{self.subject_id:02d}/func1pt8mm/design/design_{run_id}.tsv"
                )
                im_ids = pd.read_csv(path_to_df, header=None).iloc[:, 0].to_list()
                for timestep, image_id in enumerate(im_ids):
                    if image_id != 0 and image_id in test_im_ids:
                        im_fp = nsddata_path / f"nsd_stimuli/{image_id-1}.png"
                        nifti_fp = (
                            nsddata_path
                            / f"nsddata_timeseries/ppdata/subj{self.subject_id:02d}/func1pt8mm/timeseries/timeseries_{run_id}.nii.gz"
                        )
                        roi_fp = (
                            nsddata_path
                            / f"nsddata/ppdata/subj{int(self.subject_id):02d}/func1pt8mm/roi/nsdgeneral.nii.gz"
                        )
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
                            )
                        )
        return events

    @infra.apply(item_uid=str, exclude_from_cache_uid=("averaged", "seed"))
    def prepare(self, events: list[ImageEvent]) -> tp.Iterator[tp.Any]:
        for event in events:
            nifti = nibabel.load(event.nifti_fp, mmap=True)
            nifti = nifti.slicer[..., :225]
            roi_np = nibabel.load(event.roi_fp, mmap=True).get_fdata()
            nifti_data = nifti.get_fdata()[roi_np > 0]

            # z-score across run and detrend
            nifti_data = nifti_data.T  # set time as first dim
            shape = nifti_data.shape
            nifti_data = nilearn.signal.clean(
                nifti_data.reshape(shape[0], -1),
                detrend=True,
                high_pass=None,
                t_r=TR_s,
                standardize="zscore_sample",
            )
            nifti_data = nifti_data.reshape(shape).T

            image = np.array(
                Image.open(event.im_fp).convert("RGB").resize((512, 512), Image.BILINEAR),
                dtype=np.uint8,
            )

            yield {
                "brain": nifti_data[..., event.start_idx : event.end_idx],
                "img": image,
            }

    def build(self):
        events = self.create_event_list()
        data = self.prepare(events)
        data = list(data)

        grouped_events = defaultdict(list)
        for idx, event in enumerate(events):
            grouped_events[event.im_fp].append(idx)
        if self.averaged:
            averaged_data_list = []
            for im_fp in grouped_events:
                averaged_brain = np.mean(
                    [data[idx]["brain"] for idx in grouped_events[im_fp]], axis=0
                )
                averaged_data = data[grouped_events[im_fp][0]].copy()
                averaged_data["brain"] = averaged_brain
                averaged_data_list.append(averaged_data)
            data = averaged_data_list
        else:
            random.seed(self.seed)
            data = [
                data[random.choice(grouped_events[im_fp])] for im_fp in grouped_events
            ]

        fmri = np.stack([item["brain"] for item in data], axis=0)
        images = np.stack([item["img"] for item in data], axis=0)
        return NsdDataset(fmris=fmri, images=images)


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
        config = config if config is not None else NeuroImagesDataModuleConfig()
        self.nsd_dataset_config = config.nsd_dataset_config

        self.batch_size = config.batch_size
        self.workers = config.workers
        self.pin_memory = config.pin_memory

        self.test_groupbyimg = config.test_groupbyimg

    def setup(self, stage: tp.Optional[str] = None):

        self.eval_dataset = self.nsd_dataset_config.build()

        print(
            "Number of samples in test dataset:",
            len(self.eval_dataset),
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return self.val_dataloader()