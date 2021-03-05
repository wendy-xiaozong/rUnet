#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10  #maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=tensorboard-%j.out  # %N for node name, %j for jobID
#SBATCH --time=01-00:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send email
#SBATCH --mail-type=ALL

LOG_DIR=/home/jueqi/projects/def-jlevman/jueqi/rUnet_log
tensorboard --logdir="$LOG_DIR"