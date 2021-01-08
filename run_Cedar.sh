#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --gres=gpu:v100l:4  # on Cedar
#SBATCH --mem=192000M  # memory
#SBATCH --cpus-per-task=32
#SBATCH --output=kidney-%j.out  # %N for node name, %j for jobID
#SBATCH --time=01-00:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send emailS
#SBATCH --mail-type=ALL

module load python/3.6 cuda cudnn gcc/8.3.0
SOURCEDIR=/home/jueqi/projects/def-jlevman/jueqi/

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=^docker0,lo

# force to synchronization, can pinpoint the exact number of lines of error code where our memory operation is observed
CUDA_LAUNCH_BLOCKING=1

# Prepare virtualenv
#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"
#pip install -r $SOURCEDIR/requirements.txt && echo "$(date +"%T"):  install successfully!"
source /home/jueqi/projects/def-jlevman/jueqi/ENV/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"

echo -e '\n'
cd $SLURM_TMPDIR
mkdir work
echo "$(date +"%T"):  Copying data"
tar -xf /home/jueqi/projects/def-jlevman/jueqi/Data/BraTS/BraTS_18-20.tar -C work && echo "$(date +"%T"):  Copied data"
# cp /home/jueqi/projects/def-jlevman/jueqi/Data/Kaggle-RSNA/features.csv work/ && echo "$(date +"%T"):  Copied data"

cd work

GPUS=4
RUN=6
ENCODER_NAME=se_resnext50_32x4d
THRESHOLD=0.3
LEARNING_RATE=0.001
MODEL=Unet
BATCH_SIZE=10
LOSS=bce
LOG_DIR=/home/jueqi/projects/def-jlevman/jueqi/kidney_log

# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"
tensorboard --logdir="$LOG_DIR" --host 0.0.0.0 & python3 /home/jueqi/projects/def-jlevman/jueqi/kidney/7/project/main.py \
       --gpus=$GPUS \
       --run=$RUN \
       --loss=$LOSS \
       --batch_size=$BATCH_SIZE \
       --encoder_name=$ENCODER_NAME \
       --learning_rate=$LEARNING_RATE \
       --tensor_board_logger="$LOG_DIR" \
       --model="$MODEL" && echo "$(date +"%T"):  Finished running!"

#       --checkpoint_file="epoch=76-val_dice=0.43038.ckpt" \