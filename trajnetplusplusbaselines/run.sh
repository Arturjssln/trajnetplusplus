#!/bin/bash
#SBATCH --chdir /home/jesslen/trajnetplusplus/trajnetbaselines
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks 1
#SBATCH --mem 120000
#SBATCH --time 60:00:00 

source /home/jesslen/venvs/trajnetv/bin/activate
srun python -m trajnetbaselines.vae.trainer --epochs 15 --step_size 6 --path data_test --goals --type nn 
