#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

pip install torch==2.4.0 torchvision==0.19 torchaudio==2.4.0 scikit-image==0.24.0

git clone git@github.com:huggingface/diffusers.git
pushd diffusers
git checkout v0.32.0-release
patch -p1 -d . < ../vd_patch.diff
pip install -e .
popd

pip install lightning==2.4.0 deepspeed==0.15.1 x_transformers
pip install xformers==0.0.28.post1 hydra-core==1.3.2 h5py==3.11.0 peft==0.13.0 
pip install git+https://github.com/openai/CLIP.git
pip install retina-face

# Downgrade tensorflow
pip install tensorflow==2.13.0

# Compatible dreamsim version
pip install dreamsim==0.2.0

pip install typing-extensions==4.13.2 exca nibabel nilearn