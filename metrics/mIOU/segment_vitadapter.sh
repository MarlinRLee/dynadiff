#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

eval "$(conda shell.bash hook)"
conda activate miou
echo "$(which pip)"
echo "$2" "$1"

CONFIG="ViT-Adapter/segmentation/configs/coco_stuff164k/mask2former_beitv2_adapter_large_896_80k_cocostuff164k_ss.py"
CHECKPOINT="ViT-Adapter/segmentation/models/mask2former_beitv2_adapter_large_896_80k_cocostuff164k.pth"


python metrics/mIOU/segment_vitadapter.py $CONFIG $CHECKPOINT --work-dir "$1"