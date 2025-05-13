# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from exca import ConfDict


def get_cfg(
    subject: int,
    averaged_trial: bool,
    # save_recons_to: str,
    cache: str,
    seed: int,
    vd_cache_dir: str,
    custom_infra: dict | None = None,
) -> ConfDict:
    """
    Get the configuration for the evaluation task.

    Args:
        subject (int): Subject number.
        averaged_trial (bool): Flag to indicate if averaged trial is used.
        cache (str): Directory for caching data.
        seed (int): Seed for RNG.
        vd_cache_dir (str): Directory for caching Versatile Diffusion.
        custom_infra (dict, optional): Custom TaskInfra/MapInfra configuration. Defaults to None (compute locally).

    Returns:
        ConfDict: Configuration dictionary for the evaluation task.
    """

    with open("config/config.yaml", "r") as f:
        config = ConfDict.from_yaml(f)

    config["versatilediffusion_config.vd_cache_dir"] = vd_cache_dir
    config["seed"] = seed
    config["data.nsd_dataset_config.seed"] = seed
    config["data.nsd_dataset_config.averaged"] = averaged_trial
    config["data.nsd_dataset_config.subject_id"] = subject

    local_infra = {
        "cluster": None,
        "folder": cache,
    }

    config["infra"] = local_infra

    if custom_infra is not None:
        assert all(
            [
                key
                in [
                    "task_infra_data",
                    "map_infra_image_generation",
                ]
                for key in custom_infra
            ]
        ), "Infra can be specified only for 'task_infra_data' preparation and 'map_infra_image_generation'"



    config["data.nsd_dataset_config.infra"] = custom_infra["task_infra_data"] if custom_infra is not None else local_infra
    config["image_generation_infra"] = custom_infra["map_infra_image_generation"] if custom_infra is not None else local_infra
    
    

    return config
