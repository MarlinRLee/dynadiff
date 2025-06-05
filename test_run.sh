#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100g
#SBATCH --tmp=30g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:2

# module load cuda/11.8.0-gcc-7.2.0-xqzqlf2

source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh

conda activate dynadiff5

#conda create --name dynadiff4 python==3.10 -y

#chmod +x setup.sh
#./setup.sh

deepspeed --num_gpus=2 train.py --deepspeed_config ds_config.json