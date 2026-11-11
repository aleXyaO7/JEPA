#!/usr/bin/env bash
#SBATCH --job-name=lewm
#SBATCH --account=raiselab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
#
# LeWorldModel (end-to-end JEPA, stable-worldmodel reference script) on one environment.
# Usage:  sbatch -J lewm_tworoom slurm/lewm_train.sbatch tworoom   [extra hydra overrides]
#   data configs: tworoom | dmc (Reacher) | ogb (OGBench Cube) | pusht
# Each run gets its own output_model_name + subdir, so checkpoints never collide.
# Re-submitting the same DATA resumes from <run_dir>/<name>_weights.ckpt.
set -euo pipefail

DATA="$1"; shift
SWM=/sfs/weka/scratch/qxe3fj/JEPA/src/stable-worldmodel
PY=/scratch/qxe3fj/.conda/envs/swm/bin/python

unset VIRTUAL_ENV PYTHONPATH
export PYTHONNOUSERSITE=1
export STABLEWM_HOME=/sfs/weka/scratch/qxe3fj/JEPA/stablewm_home
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

cd "$SWM/scripts/train"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "data=$DATA overrides: $*"

"$PY" lewm.py \
  data="$DATA" \
  output_model_name="lewm_$DATA" \
  subdir="lewm_$DATA" \
  num_workers="$SLURM_CPUS_PER_TASK" \
  "$@"
