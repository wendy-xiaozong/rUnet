#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mem=187G  # memory
#SBATCH --cpus-per-task=48
#SBATCH --output=log-%j.out  # %N for node name, %j for jobID
#SBATCH --time=01-00:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send emailS
#SBATCH --mail-type=ALL

module load python/3.6 cuda cudnn gcc/8.3.0
source /home/jueqi/projects/def-jlevman/jueqi/ENV/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"

LOG_DIR=/home/jueqi/projects/def-jlevman/jueqi/rUnet_log
tensorboard --logdir="$LOG_DIR" --host 0.0.0.0
