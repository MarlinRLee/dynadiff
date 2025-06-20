#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --tmp=30g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:2

module purge

# Load the CUDA module first (which is tied to GCC 7.2.0)
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2

# Load the newer GCC. This module load puts gcc/9.2.0's bin in PATH.
module load gcc/9.2.0

# Explicitly ensure the newer GCC's bin directory is at the very front of PATH.
export PATH="/common/software/install/migrated/gcc/9.2.0/bin:$PATH"

# Explicitly add the newer GCC's lib64 directory to LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="/common/software/install/migrated/gcc/9.2.0/lib64:$LD_LIBRARY_PATH"

export TRITON_CACHE_DIR="/scratch.local/lee02328/.triton_cache"

#export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export NCCL_TIMEOUT=7200s
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG=WARN

source /home/boleydl/lee02328/miniconda3/etc/profile.d/conda.sh
conda activate dynadiff5

cd /scratch.global/lee02328/dynadiff/
# Your DeepSpeed training command
#deepspeed train.py --deepspeed_config ds_config.json
srun python train.py

#jupyter notebook --no-browser --ip=0.0.0.0 --port=8888