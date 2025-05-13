# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code taken from diffusers library (src/diffusers/loaders/peft.py with checkout v0.32.0-release) and modified to ensure compatibility with versatile diffusion model."""

from typing import List, Union

from peft import PeftConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer


def set_adapter_layers(model, enabled=True):
    from peft.tuners.tuners_utils import BaseTunerLayer

    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            # The recent version of PEFT needs to call `enable_adapters` instead
            if hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled=enabled)
            else:
                module.disable_adapters = not enabled


def add_adapter(model, adapter_config, adapter_name: str = "default") -> None:
    r"""
    Adds a new adapter to the current model for training. If no adapter name is passed, a default name is assigned
    to the adapter to follow the convention of the PEFT library.

    If you are not familiar with adapters and PEFT methods, we invite you to read more about them in the PEFT
    [documentation](https://huggingface.co/docs/peft).

    Args:
        adapter_config (`[~peft.PeftConfig]`):
            The configuration of the adapter to add; supported adapters are non-prefix tuning and adaption prompt
            methods.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.
    """
    # check_peft_version(min_version=MIN_PEFT_VERSION)

    # if not is_peft_available():
    #     raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

    if not model._hf_peft_config_loaded:
        model._hf_peft_config_loaded = True
    elif adapter_name in model.peft_config:
        raise ValueError(
            f"Adapter with name {adapter_name} already exists. Please use a different name."
        )

    if not isinstance(adapter_config, PeftConfig):
        raise ValueError(
            f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead."
        )

    # Unlike transformers, here we don't need to retrieve the name_or_path of the unet as the loading logic is
    # handled by the `load_lora_layers` or `StableDiffusionLoraLoaderMixin`. Therefore we set it to `None` here.
    adapter_config.base_model_name_or_path = None
    inject_adapter_in_model(adapter_config, model, adapter_name)
    set_adapter(model, adapter_name)


def set_adapter(model, adapter_name: Union[str, List[str]]) -> None:
    """
    Sets a specific adapter by forcing the model to only use that adapter and disables the other adapters.

    If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
    [documentation](https://huggingface.co/docs/peft).

    Args:
        adapter_name (Union[str, List[str]])):
            The list of adapters to set or the adapter name in the case of a single adapter.
    """
    # check_peft_version(min_version=MIN_PEFT_VERSION)

    if not model._hf_peft_config_loaded:
        raise ValueError("No adapter loaded. Please load an adapter first.")

    if isinstance(adapter_name, str):
        adapter_name = [adapter_name]

    missing = set(adapter_name) - set(model.peft_config)
    if len(missing) > 0:
        raise ValueError(
            f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
            f" current loaded adapters are: {list(model.peft_config.keys())}"
        )

    _adapters_has_been_set = False

    for _, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            if hasattr(module, "set_adapter"):
                module.set_adapter(adapter_name)
            # Previous versions of PEFT does not support multi-adapter inference
            elif not hasattr(module, "set_adapter") and len(adapter_name) != 1:
                raise ValueError(
                    "You are trying to set multiple adapters and you have a PEFT version that does not support multi-adapter inference. Please upgrade to the latest version of PEFT."
                    " `pip install -U peft` or `pip install -U git+https://github.com/huggingface/peft.git`"
                )
            else:
                module.active_adapter = adapter_name
            _adapters_has_been_set = True

    if not _adapters_has_been_set:
        raise ValueError(
            "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
        )
