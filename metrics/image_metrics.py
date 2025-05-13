# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code modified from https://github.com/ozcelikfu/brain-diffuser evaluation scripts

from pathlib import Path

import clip
import numpy as np
import scipy as sp
import torch
import torchvision.models as tvmodels
import torchvision.transforms as transforms
import torchvision.transforms as T
from dreamsim import dreamsim
from PIL import Image
from scipy.stats import binom
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _compute_image_generation_features(images, emb_batch_size=32, device="cuda:0"):
    class batch_generator_external_images(Dataset):
        def __init__(self, images: list, net_name="clip"):
            self.images = images
            self.net_name = net_name

            if self.net_name == "clip":
                self.normalize = transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                )
            else:
                self.normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

        def __getitem__(self, idx):
            img = self.images[idx]
            img = T.functional.resize(img, (224, 224))
            img = T.functional.to_tensor(img).float()
            img = self.normalize(img)
            return img

        def __len__(self):
            return len(self.images)

    global feat_list
    feat_list = []

    def fn(module, inputs, outputs):
        feat_list.append(outputs.cpu().numpy())

    net_list = [
        ("inceptionv3", "avgpool"),
        ("clip", "final"),
        ("alexnet", 2),
        ("alexnet", 5),
        ("efficientnet", "avgpool"),
        ("swav", "avgpool"),
    ]

    net = None

    result = dict()

    for net_name, layer in net_list:
        feat_list = []
        print(net_name, layer)
        dataset = batch_generator_external_images(images=images, net_name=net_name)
        loader = DataLoader(dataset, emb_batch_size, shuffle=False)

        if net_name == "inceptionv3":
            net = tvmodels.inception_v3(pretrained=True)
            if layer == "avgpool":
                net.avgpool.register_forward_hook(fn)
            elif layer == "lastconv":
                net.Mixed_7c.register_forward_hook(fn)

        elif net_name == "alexnet":
            net = tvmodels.alexnet(pretrained=True)
            if layer == 2:
                net.features[4].register_forward_hook(fn)
            elif layer == 5:
                net.features[11].register_forward_hook(fn)
            elif layer == 7:
                net.classifier[5].register_forward_hook(fn)

        elif net_name == "clip":
            model, _ = clip.load("ViT-L/14", device=device)
            net = model.visual
            net = net.to(torch.float32)
            if layer == 7:
                net.transformer.resblocks[7].register_forward_hook(fn)
            elif layer == 12:
                net.transformer.resblocks[12].register_forward_hook(fn)
            elif layer == "final":
                net.register_forward_hook(fn)

        elif net_name == "efficientnet":
            net = tvmodels.efficientnet_b1(weights=True)
            net.avgpool.register_forward_hook(fn)

        elif net_name == "swav":
            net = torch.hub.load("facebookresearch/swav:main", "resnet50")
            net.avgpool.register_forward_hook(fn)

        net = net.to(device)
        net.eval()

        with torch.no_grad():
            for i, x in tqdm(enumerate(loader), total=len(loader)):
                x = x.to(device)
                _ = net(x)
        if net_name == "clip":
            if layer == 7 or layer == 12:
                feat_list = np.concatenate(feat_list, axis=1).transpose((1, 0, 2))
            else:
                feat_list = np.concatenate(feat_list)
        else:
            feat_list = np.concatenate(feat_list)

        result[net_name + "-" + str(layer)] = feat_list

    return result


def _pairwise_corr_all(ground_truth, predictions):
    r = np.corrcoef(ground_truth, predictions)
    r = r[: len(ground_truth), len(ground_truth) :]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    perf = np.mean(success_cnt) / (len(ground_truth) - 1)
    p = 1 - binom.cdf(
        perf * len(ground_truth) * (len(ground_truth) - 1),
        len(ground_truth) * (len(ground_truth) - 1),
        0.5,
    )

    return perf, p


def compute_dreamsim(
    preds: list[str | Path | Image.Image],
    trues: list[str | Path | Image.Image],
    device: str = "cuda:0",
):
    
    
    model_dreamsim, preprocess_dreamsim = dreamsim(
        pretrained=True,
    )
    
    dreamsim_list = []
    for pred, true in zip(preds, trues):
        dreamsim_list += [
                model_dreamsim(
                    preprocess_dreamsim(pred).to(device),
                    preprocess_dreamsim(true).to(device),
                ).cpu().numpy().item()
            ]
    return np.array(dreamsim_list).mean()


def compute_image_generation_metrics(
    preds,
    trues,
    imsize_for_pixel_level_metrics=425,
    emb_batch_size=32,
    device="cuda:0",
):
    result = dict()

    assert len(preds) == len(trues)
    n = len(preds)

    all_images = trues + preds
    feats = _compute_image_generation_features(
        all_images, emb_batch_size=emb_batch_size, device=device
    )

    gt_feats = dict()
    eval_feats = dict()

    for metric_name in feats:
        gt_feats[metric_name] = feats[metric_name][:n]
        eval_feats[metric_name] = feats[metric_name][n:]

    distance_fn = sp.spatial.distance.correlation
    for metric_name in gt_feats.keys():
        gt_feat = gt_feats[metric_name]
        eval_feat = eval_feats[metric_name]
        n = len(gt_feat)

        gt_feat = gt_feat.reshape((len(gt_feat), -1))
        eval_feat = eval_feat.reshape((len(eval_feat), -1))

        net_name, _ = metric_name.split("-")

        if net_name in ["efficientnet", "swav"]:
            distances = np.array(
                [distance_fn(gt_feat[i], eval_feat[i]) for i in range(n)]
            )
            result[metric_name] = distances.mean()
        else:
            result[metric_name] = _pairwise_corr_all(gt_feat, eval_feat)[0]

    ssim_list = []
    pixcorr_list = []
    for i in range(n):
        gen_image = preds[i].resize(
            (imsize_for_pixel_level_metrics, imsize_for_pixel_level_metrics)
        )
        gt_image = trues[i].resize(
            (imsize_for_pixel_level_metrics, imsize_for_pixel_level_metrics)
        )

        gen_image = np.array(gen_image) / 255.0
        gt_image = np.array(gt_image) / 255.0
        pixcorr_res = np.corrcoef(gt_image.reshape(1, -1), gen_image.reshape(1, -1))[0, 1]
        pixcorr_list.append(pixcorr_res)
        gen_image = rgb2gray(gen_image)
        gt_image = rgb2gray(gt_image)
        ssim_res = ssim(
            gen_image,
            gt_image,
            multichannel=True,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0,
        )
        ssim_list.append(ssim_res)

    ssim_list = np.array(ssim_list)
    pixcorr_list = np.array(pixcorr_list)
    result["pixcorr"] = pixcorr_list.mean()
    result["ssim"] = ssim_list.mean()
    result['dreamsim'] = compute_dreamsim(preds, trues, device=device)

    return result
