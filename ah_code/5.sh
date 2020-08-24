#!/bin/bash

#SBATCH -p 2080GPU,1080GPU
#SBATCH -J 5-train
#SBATCH --output=ft_%j.out
#SBATCH --gres=gpu
#SBATCH -c 6


srun singularity exec --bind /data_GPU/:/data_GPU/ --nv /data_GPU/wenpei/Containers/pytorch-latest.simg python ah3_finetuning.py 
