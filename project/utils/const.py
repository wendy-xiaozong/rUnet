import os
from pathlib import Path

# Some code is adapted from: https://github.com/DM-Berger/unet-learn/blob/master/src/train/load.py
COMPUTECANADA = False  # are we running on Compute Canada
IN_COMPUTE_CAN_JOB = False  # are we running inside a Compute Canada Job

TMP = os.environ.get("SLURM_TMPDIR")
ACT = os.environ.get("SLURM_ACCOUNT")


if ACT:  # we are on Compute Canada, but not in a job script, so we don't want to run too much
    COMPUTECANADA = True
if TMP:  # running inside Compute Canada
    COMPUTECANADA = True
    IN_COMPUTE_CAN_JOB = True

if COMPUTECANADA:
    DATA_ROOT = Path(str(TMP)).resolve() / "work"
else:
    DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

IMAGESIZE = 128
