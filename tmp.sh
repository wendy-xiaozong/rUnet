#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mem=187G  # memory
#SBATCH --cpus-per-task=48
#SBATCH --output=log-%j.out  # %N for node name, %j for jobID
#SBATCH --time=00-01:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send emailS
#SBATCH --mail-type=ALL

LOG_DIR=/home/jueqi/projects/def-jlevman/jueqi/rUnet_log
tensorboard --logdir="$LOG_DIR" --host 0.0.0.0
