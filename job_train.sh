#!/bin/bash
 
#SBATCH --job-name=Train
#SBATCH --output=log_slurm/%j_%x.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey0993-gpu

module load ffmpeg/4.4.1
python src/train.py --khan
