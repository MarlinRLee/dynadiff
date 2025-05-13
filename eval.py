# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import yaml
from config.cfg import get_cfg
from dynadiff import DynaDiffEval


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluation script for Dynadiff")
    parser.add_argument(
        "--subject",
        type=int,
        choices=[1, 2, 5, 7],
        help="Subject identifier (must be 1, 2, 5, or 7)",
    )

    parser.add_argument(
        "--averaged-trial",
        action="store_true",
        help="Reconstruct an image from each (1000) averaged test trial instead",
    )

    parser.add_argument(
        "--cache",
        type=str,
        default="./cache",
        help="Folder used to prepare and store fMRI data. Defaults to ./cache.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Seed for RNG (default: 3)",
    )

    parser.add_argument(
        "--vd_cache_dir",
        type=str,
        default="./versatile_diffusion",
        help="Folder to cache Versatile Diffusion. Defaults to ./versatile_diffusion.",
    )

    parser.add_argument(
        "--infra-yaml",
        type=str,
        default=None,
        help="Path to infra.yaml config file for data preparation and image generation."
        "Defaults to None, i.e. using local compute only",
    )

    parser.add_argument(
        "--compute-miou",
        action="store_true",
        help="Compute mIoU for the stimulus / reconstruction pairs.",
    )

    args = parser.parse_args()

    custom_infra = None
    if args.infra_yaml is not None:
        print(f"Using custom infra config at: {args.infra_yaml}")
        with open(args.infra_yaml, "r") as f:
            custom_infra = yaml.safe_load(f)

    print(f"Evaluating subject: sub{args.subject}")
    print(f"Using averaged trials: {args.averaged_trial}")
    print(f"Preparing data in: {args.cache}")
    print(f"Seed: {args.seed}")
    print(f"Caching Versatile Diffusion model in : {args.vd_cache_dir}")

    cfg = get_cfg(
        args.subject,
        args.averaged_trial,
        args.cache,
        args.seed,
        args.vd_cache_dir,
        custom_infra,
    )
    
    average_id = 'averaged' if cfg['data']['nsd_dataset_config']['averaged'] else 'unaveraged'
    # subject_id = f"subj{cfg.data.nsd_dataset_config.subject_id:02d}"
    subject_id = f"subj{cfg['data']['nsd_dataset_config']['subject_id']:02d}"
                
    folder = Path(cfg['image_generation_infra']['folder']) / f"reconstructions_{subject_id}_{average_id}"
    print(
        f"Saving reconstructions to: {folder}"
    )

    task = DynaDiffEval(**cfg)
    task.prepare()
    task.run()

    if args.compute_miou:
        print("Computing mIoU")
        task.compute_miou()


if __name__ == "__main__":
    evaluate()
