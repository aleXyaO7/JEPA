#!/usr/bin/env bash
#SBATCH --job-name=lewm_train
#SBATCH --account=raiselab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-06:00:00
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
#
# Train LeWM (stable-worldmodel scripts/train/lewm.py, default hyperparameters)
# on an env that has NO official checkpoint. Data must exist first:
#   $STABLEWM_HOME/datasets/<data>.h5   (from slurm/collect_data.sbatch)
# Usage:  sbatch -J lewm_train_dmc_cheetah slurm/lewm_train.sbatch dmc_cheetah [hydra overrides]
#   <data> = a config in conf_overrides/lewm/data/ (ogb_scene, dmc_cheetah, ...)
#   EPOCHS (default 100, the LeWM default) sets trainer.max_epochs.
# Measured (A6000): ~3.6-4.1 it/s at batch 128. 2000 eps x 200 steps -> ~2.6k it/epoch,
# 100 epochs ~ 20 h (+val ~22 h); 30 h limit.
# Weights: $STABLEWM_HOME/checkpoints/lewm_<data>/weights_epoch_<k>.pt (+ config.json),
# loadable with policy=lewm_<data> in eval scripts. Re-submitting resumes if
# <run_dir>/lewm_<data>_weights.ckpt exists (mid-run resume NOT verified).
set -euo pipefail

DATA="$1"; shift
EPOCHS="${EPOCHS:-100}"
SWM=/sfs/weka/scratch/qxe3fj/JEPA/src/stable-worldmodel
PY=/scratch/qxe3fj/.conda/envs/swm/bin/python

unset VIRTUAL_ENV
export PYTHONPATH="$SWM"     # library from source (PyPI 0.1.1 lacks .planning)
export PYTHONNOUSERSITE=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export STABLEWM_HOME=/sfs/weka/scratch/qxe3fj/JEPA/stablewm_home
export MUJOCO_GL=egl

[ -f "$STABLEWM_HOME/datasets/$DATA.h5" ] || { echo "!! missing dataset $STABLEWM_HOME/datasets/$DATA.h5 -- run collect_data first"; exit 1; }

cd "$SWM/scripts/train"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "data=$DATA epochs=$EPOCHS overrides: $*"

"$PY" lewm.py \
  --config-dir /sfs/weka/scratch/qxe3fj/JEPA/conf_overrides/lewm \
  data="$DATA" \
  output_model_name="lewm_$DATA" \
  subdir="lewm_$DATA" \
  trainer.max_epochs="$EPOCHS" \
  num_workers="$SLURM_CPUS_PER_TASK" \
  "$@"
