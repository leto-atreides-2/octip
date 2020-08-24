#!/bin/bash

#SBATCH -p 2080GPU
#SBATCH -J 3
#SBATCH --output=ah3_%j.out
#SBATCH --gres=gpu
#SBATCH -c6


srun singularity exec --bind /data_GPU/:/data_GPU/ --nv /data_GPU/wenpei/Containers/pytorch-latest.simg python ah3_trainer.py 
