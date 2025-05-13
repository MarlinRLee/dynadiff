# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import subprocess
import tempfile
import typing as tp
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TR
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm, trange

from .eval_miou import SegmentationMetric


def compute_miou(
    preds: tp.List[str | Path | Image.Image],
    trues: tp.List[str | Path | Image.Image],
    eval_res: int = 512,
):

    def openLbl(path):
        lbl = Image.open(str(path))
        lbl = (
            TR.Resize((eval_res, eval_res), interpolation=InterpolationMode.NEAREST)(
                TR.functional.to_tensor(lbl)
            )
            * 255
        )

        return lbl

    def process_img(img):
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        img = img.resize((eval_res, eval_res))
        return img

    preds = [process_img(x.copy()) for x in preds]
    trues = [process_img(x.copy()) for x in trues]
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        folder_gt = folder / "gts" / "images"
        folder_pred = folder / "preds" / "images"
        folder_gt.mkdir(exist_ok=True, parents=True)
        folder_pred.mkdir(exist_ok=True, parents=True)
        pred_labels_gt = folder_gt.parent.joinpath("pred_label")
        pred_labels_gt.mkdir(exist_ok=True)
        pred_labels_pred = folder_pred.parent.joinpath("pred_label")
        pred_labels_pred.mkdir(exist_ok=True)

        images_dir = os.path.join(folder, "images")
        os.makedirs(images_dir)

        for idx, (pred, true) in tqdm(enumerate(zip(preds, trues)), total=len(trues)):
            pred.save(folder_pred / f"pred_{idx}.png")
            true.save(folder_gt / f"gt_{idx}.png")

        bashCommand = f"bash metrics/mIOU/segment_vitadapter.sh {str(folder_gt.parent)}"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output, error)
        bashCommand = f"bash metrics/mIOU/segment_vitadapter.sh {str(folder_pred.parent)}"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output, error)

        seg_metric = SegmentationMetric(172)

        tot_miou = []
        for idx in trange(len(trues)):
            pred_lbl_path = pred_labels_pred / f"pred_{idx}.png"
            gt_lbl_path = pred_labels_gt / f"gt_{idx}.png"
            pred_lbl = openLbl(pred_lbl_path)
            gt_lbl = openLbl(gt_lbl_path)


            bool_mask = torch.ones_like(gt_lbl, dtype=torch.bool)

            seg_metric.update(pred_lbl, gt_lbl, bool_mask)
            miou = seg_metric.get()
            seg_metric.reset()
            tot_miou.append(miou)
        return np.mean(tot_miou)
