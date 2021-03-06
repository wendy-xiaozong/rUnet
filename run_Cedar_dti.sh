#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --gres=gpu:v100l:4  # on Cedar
#SBATCH --mem=192000M  # memory
#SBATCH --cpus-per-task=32
#SBATCH --output=runet-%j.out  # %N for node name, %j for jobID
#SBATCH --time=01-00:00     # time (DD-HH:MM)
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
# tar -xf /home/jueqi/projects/def-jlevman/jueqi/Data/BraTS/BraTS_18-20.tar -C work && echo "$(date +"%T"):  Copied data"
tar -xf /home/jueqi/projects/def-jlevman/jueqi/Data/DTI/dti_preprocessed.tar -C work && echo "$(date +"%T"):  Copied data"

cd work

GPUS=4
BATCH_SIZE=1
LOSS=l2  # l1 l2 smoothl1
TASK=diffusion_fa   # t1t2 diffusion_adc diffusion_fa
ACTIVATION=LeakyReLU # LeakyReLU ReLU
NORMALIZATION=Batch # Batch Group InstanceNorm3d
WEIGHT_DECAY=1e-8
IN_CHANNELS=288
CLIP_MIN=2
CLIP_MAX=4
X_image=t1
y_image=t2
LEARNING_RATE=1e-5

LOG_DIR=/home/jueqi/projects/def-jlevman/jueqi/rUnet_final_result

# run script
echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"
tensorboard --logdir="$LOG_DIR" --host 0.0.0.0 & python3 /home/jueqi/projects/def-jlevman/jueqi/rUnet/4/project/main.py \
       --gpus=$GPUS \
       --in_channels=$IN_CHANNELS \
       --loss="$LOSS" \
       --clip_min=$CLIP_MIN \
       --clip_max=$CLIP_MAX \
       --batch_size=$BATCH_SIZE \
       --weight_decay=$WEIGHT_DECAY \
       --activation="$ACTIVATION" \
       --normalization="$NORMALIZATION" \
       --X_image="$X_image" \
       --y_image="$y_image" \
       --task="$TASK" \
       --checkpoint_file="epoch=35-val_loss=0.00214-val_MAE=0.50440.ckpt" \
       --learning_rate=$LEARNING_RATE \
       --tensor_board_logger="$LOG_DIR" && echo "$(date +"%T"):  Finished running!"

#       --fast_dev_run \
#       --checkpoint_file="epoch=290-val_loss=4.86729e-09.ckpt" \
# tar -cf /home/jueqi/projects/def-jlevman/jueqi/Data/DTI/dti_preprocessed.tar 1.npz 2.npz 3.npz 4.npz 5.npz