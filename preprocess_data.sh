#!/bin/bash -l
#SBATCH --job-name=nsd-preprocessing  
#SBATCH --time=6:00:00               # 12 hours should be sufficient, adjust if needed
#SBATCH --nodes=1                     # Preprocessing runs on a single node
#SBATCH --ntasks=1                    # A single main process to manage the script
#SBATCH --cpus-per-task=16            # Request many CPUs for multiprocessing
#SBATCH --mem=30g                    # Request significant memory (e.g., 100GB)
#SBATCH --mail-type=ALL               # Get notifications for start, end, and failure
#SBATCH --mail-user=lee02328@umn.edu

source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate dynadiff5

cd /scratch.global/lee02328/dynadiff/

python process_data.py