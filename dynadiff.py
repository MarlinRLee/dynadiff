# Lightning module for model training and evaluation# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import typing as tp
from pathlib import Path

import deepspeed
import numpy as np
import pydantic
import torch
import torchvision.transforms as T
from data import NeuroImagesDataModuleConfig
from exca import MapInfra, TaskInfra
from metrics.image_metrics import compute_image_generation_metrics
from model.models import VersatileDiffusionConfig
from torch import nn
from tqdm import tqdm


class DynaDiffEval(pydantic.BaseModel):
    data: NeuroImagesDataModuleConfig
    seed: int = 33
    versatilediffusion_config: VersatileDiffusionConfig
    strategy: str = "auto"
    device: str = "cuda"
    checkpoint_path: str = "./checkpoints"
    infra: TaskInfra = TaskInfra(version="1")
    image_generation_infra: MapInfra = MapInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[list[str]] = [
        "device",
        "seed",
    ]

    def model_post_init(self, __context: tp.Any) -> None:
        if self.infra.folder is None:
            msg = "infra.folder needs to be specified to save the results."
            raise ValueError(msg)
        self.data.workers = self.infra.cpus_per_task
        self.infra.folder = Path(self.infra.folder)
        self.infra.folder.mkdir(exist_ok=True, parents=True)
        self.image_generation_infra.folder = self.infra.folder

    def _get_brain_model(self, data_module) -> nn.Module:

        brain_n_in_channels, brain_temp_dim = data_module.eval_dataset[0]["brain"].size()

        copy_versatilediffusion_config = self.infra.clone_obj().versatilediffusion_config
        if copy_versatilediffusion_config.brain_modules_config is not None:
            for k in copy_versatilediffusion_config.brain_modules_config:
                copy_versatilediffusion_config.brain_modules_config[k].n_subjects = (
                    1
                )

        brain_model = copy_versatilediffusion_config.build(
            brain_n_in_channels, brain_temp_dim
        )

        checkpoint_dir = (
            Path(self.checkpoint_path).resolve()
            / f"sub{self.data.nsd_dataset_config.subject_id:02d}"
        )

        print(f"Loading checkpoint at: {checkpoint_dir}")

        state_dict = (
            deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
                checkpoint_dir=checkpoint_dir,
                tag="checkpoint",
            )
        )

        state_dict = {
            k[len("model.") :]: v
            for k, v in state_dict.items()
            if k[: len("model.")] == "model."
        }
        brain_model.load_state_dict(state_dict, strict=False)
        brain_model = brain_model.eval()
        return brain_model

    @image_generation_infra.apply(item_uid=str, cache_type="MemmapArrayFile")
    def generate_images(self, images_idx: list[int]) -> tp.Iterator[np.ndarray]:
        data = self.data.build()
        brain_model = self._get_brain_model(data).to(self.device)
        print("IMG IDX ARE: ", images_idx)
        with torch.no_grad():
            for img_idx in images_idx:
                ipt_data = data.eval_dataset[img_idx]
                ipt_data = {
                    k: (
                        v[None, ...].to(device=self.device)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in ipt_data.items()
                }

                if self.data.test_groupbyimg == "unaveraged":
                    ipt_data = {
                        k: v[0] if isinstance(v, torch.Tensor) else v
                        for k, v in ipt_data.items()
                    }
                    print("UNAVG", ipt_data["brain"].size())
                else:
                    batch_out = brain_model(**ipt_data, is_img_gen_mode=True).image
                for image in batch_out:
                    yield image.cpu().numpy()

    def prepare(self):
        torch.cuda.manual_seed(self.seed)
        data = self.data.build()
        return data, self.generate_images(list(range(len(data.eval_dataset))))

    @infra.apply
    def run(self):
        config_path = Path(self.infra.folder) / "config.yaml"
        if not config_path.exists():
            os.makedirs(self.infra.folder, exist_ok=True)
            self.infra.config(uid=False, exclude_defaults=False).to_yaml(config_path)

        data, recons = self.prepare()
        
        average_id = 'averaged' if self.data.nsd_dataset_config.averaged else 'unaveraged'
        subject_id = f"subj{self.data.nsd_dataset_config.subject_id:02d}"
                
        folder = self.image_generation_infra.folder / f"reconstructions_{subject_id}_{average_id}"
        folder.mkdir(exist_ok=True, parents=True)
        metrics_recimg = []
        metrics_gtimg = []
        for i, image in tqdm(enumerate(recons), total=len(data.eval_dataset)):
            recimg = T.ToPILImage()((image.transpose(1, 2, 0) * 255).astype(np.uint8))

            recimg.save(folder / f"{i}.png")
            metrics_recimg.append(recimg)

            gtimg = data.eval_dataset[i]["img"]
            gtimg = T.ToPILImage()((gtimg * 255).to(torch.uint8))
            gtimg.save(folder / f"{i}_gt.png")
            metrics_gtimg.append(gtimg)

        metrics = compute_image_generation_metrics(metrics_gtimg, metrics_recimg)

        mean_values = {k: float(v) for k, v in metrics.items() if "scores" not in k}
        with open("oss_metrics.json", "w") as f:
            json.dump(mean_values, f, indent=4)
        print(mean_values)

    def compute_miou(self):
        from metrics.mIOU.evaluate_img_gen import compute_miou

        data, recons = self.prepare()
        recons = [
            T.ToPILImage()((image.transpose(1, 2, 0) * 255).astype(np.uint8))
            for image in recons
        ]
        gtims = [
            T.ToPILImage()((x["img"] * 255).to(torch.uint8)) for x in data.eval_dataset
        ]
        miou = compute_miou(recons, gtims, eval_res=512)
        print(f"mIoU: {miou}")
