#!/bin/bash
#SBATCH --chdir /home/jesslen/trajnetplusplus/trajnetbaselines
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks 1
#SBATCH --mem 120000
#SBATCH --time 60:00:00 

source /home/jesslen/venvs/trajnetv/bin/activate
srun python -m trajnetbaselines.vae.trainer --path synth_data --type nn --epochs 10
