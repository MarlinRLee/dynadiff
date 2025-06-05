#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10 
#SBATCH --mem=64G  
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu 

cd /scratch.global/lee02328/dynadiff/

export PATH="$HOME/.local/bin:$PATH"

# Activate Conda environment
source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh 
conda activate dynadiff

python prepare_data.py --nsd_bucket "s3://natural-scenes-dataset" --path "./nsddata" --aws_args '"--no-sign-request"'