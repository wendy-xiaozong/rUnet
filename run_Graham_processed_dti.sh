#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mem=128000M  # memory
#SBATCH --cpus-per-task=32
#SBATCH --output=runet-%j.out  # %N for node name, %j for jobID
#SBATCH --time=00-06:00     # time (DD-HH:MM)
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
tar -xf /home/jueqi/projects/def-jlevman/jueqi/Data/DTI/diffusion.tar -C work && echo "$(date +"%T"):  Copied data"

cd work

python3 /home/jueqi/projects/def-jlevman/jueqi/rUnet/1/project/load_images.py &&
tar -cf /home/jueqi/projects/def-jlevman/jueqi/Data/DTI/dti_preprocessed.tar 1.npz 2.npz 3.npz 4.npz 5.npz

