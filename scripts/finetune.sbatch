#!/bin/bash

#SBATCH --job-name=mosaic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output=./out/finetune.out
#SBATCH --account=class
#SBATCH --priority=4294967293

module purge;

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

singularity exec --nv \
  --overlay /scratch/$USER/my_env/overlay-50G-10M.ext3:rw \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; pip install -r requirements.txt; python finetune.py --mp"
