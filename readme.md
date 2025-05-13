<!-- Copyright (c) Meta Platforms, Inc. and affiliates. -->
<!-- All rights reserved. -->
<!--  -->
<!-- This source code is licensed under the license found in the -->
<!-- LICENSE file in the root directory of this source tree. -->

# Dynadiff: Single-stage Decoding of Images from Continuously Evolving fMRI

This repository contains the official code for evaluating the **Dynadiff** model: **Dy**namic **N**eural **A**ctivity **Diff**usion for Image Reconstruction.

## Create the environment

Create a conda environment for running and evaluating the **Dynadiff** model using the following command.
The repository was tested with CUDA 12.2.0, cuDNN 8.8.1.3 (for CUDA 12.0), and GCC 12.2.0. We strongly recommend using this configuration.


```bash
conda create --name dynadiff python==3.10 -y
conda activate dynadiff
chmod +x setup.sh
./setup.sh
```


## Download Model Weights

The **Dynadiff** model weights are available on [HuggingFace](https://huggingface.co/facebook/dynadiff). 
The `subj*` folders should be stored inside `./checkpoints`.

## Prepare the NSD Data

Access to the [Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org/) requires filling out this [form](https://docs.google.com/forms/d/e/1FAIpQLSduTPeZo54uEMKD-ihXmRhx0hBDdLHNsVyeo_kCb8qbyAkXuQ/viewform) and agreeing to the dataset's Terms & Conditions.
After obtaining access, the data for evaluating **Dynadiff** can be downloaded by running the following command:

```
python prepare_data.py \
    --nsd_bucket "s3://nsd_bucket_name" \
    --path "./nsddata"
```

* By default, the **Dynadiff** evaluation script expects the NSD data to be downloaded and prepared in the folder `./nsddata`, as done with this command. This can be changed to any path `my/nsddata/path` by setting the config key: `data.nsd_dataset_config.nsddata_path: my/nsddata/path`.
* Additional `s3` arguments can be specified using the string argument `--aws_args`.



## Evaluate the Model
### Locally
Data preparation does not use GPU acceleration. Image reconstruction requires 8Gb of VRAM. 

The NSD data prepared for the **Dynadiff** evaluation will be cached by `exca` inside the folder `./cache` by default. This can be set to another folder using the config key `data.nsd_dataset_config.infra.folder`.

To reconstruct images from fMRI timeseries recorded for subject `$SUBJECT_ID` (e.g., `SUBJECT_ID=1`), run:
```bash
python eval.py --subject $SUBJECT_ID
```

By default, the reconstructed images (`{index}.png`) are stored along the stimuli (`{index}_gt.png`) in the `./cache/reconstructions_{subj_id}_{average_mode}` folder. This can be modified using the key `infra.folder` in `config/config.yaml`.

### On a SLURM cluster
Both data preparation and image reconstruction can be distributed and accelerated on a SLURM cluster via the [`exca`](https://github.com/facebookresearch/exca) library. 
The SLURM resources are specified in the YAML file `custom_infra.yaml` using:

* `task_infra_data` for data preparation (handled by an `exca.TaskInfra` instance).
* `map_infra_image_generation` for image generation (an`exca.MapInfra` instance). 
 
 Evaluation using these resources is launched with:

```bash
python eval.py --subject $SUBJECT_ID --infra-yaml custom_infra.yaml
```


## Computing mIoU for stimulus / reconstruction pairs
To accommodate incompatibilities between the `dynadiff` environment and requirements for computing the mIoU segmentation metric using the [`ViT-Adapter`](https://github.com/czczup/ViT-Adapter/tree/main/segmentation) repository, we use an additional conda environment `miou`. The `eval.py` script should still be run in the `dynadiff` environment and will automatically switch to the `miou` environment to compute the mIoU metric locally:

```bash
conda create --name miou python==3.7 -y
conda activate miou
chmod +x setup_miou.sh
./setup_miou.sh
```
Pass the `--compute-miou` flag to compute mIoU for stimulus / reconstruction pairs:
```bash
conda activate dynadiff
python eval.py --subject $SUBJECT_ID --compute-miou
```

## Contributing
See the CONTRIBUTING file for how to help out.

## License
`dynadiff` is MIT licensed, as found in the LICENSE file. Also check-out Meta Open Source [Terms of Use](https://opensource.fb.com/legal/terms/) and [Privacy Policy](https://opensource.fb.com/legal/privacy/).
