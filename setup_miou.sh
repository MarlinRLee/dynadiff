#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

git clone git@github.com:czczup/ViT-Adapter.git

# # Installing MMSegmentation v0.20.2. as per https://github.com/czczup/ViT-Adapter/tree/main/segmentation#usage
# # recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
# for Mask2Former
pip install mmdet==2.22.0
pip install mmsegmentation==0.20.2
pushd ViT-Adapter/segmentation
ln -s ../detection/ops ./
cd ops
# compile deformable attention
sh make.sh
popd

# Additional dependencies
pip install tqdm scipy==1.7.3 

# Download checkpoint
wget -O mask2former_beitv2_adapter_large_896_80k_cocostuff164k.zip https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask2former_beitv2_adapter_large_896_80k_cocostuff164k.zip && unzip mask2former_beitv2_adapter_large_896_80k_cocostuff164k.zip -d ViT-Adapter/segmentation/models/ && rm mask2former_beitv2_adapter_large_896_80k_cocostuff164k.zip