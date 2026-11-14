#!/usr/bin/env bash
#SBATCH --job-name=dinowm_train
#SBATCH --account=raiselab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
#
# Train DINO-WM (frozen DINOv2 + ViT predictor; temporal-straightening train.py,
# straightening OFF = plain DINO-WM) on the DINO-WM deformable envs, which have
# no official checkpoint. Data already downloaded under $DATASET_DIR/deformable.
# Usage:  sbatch -J dinowm_train_rope slurm/dinowm_train.sbatch <rope|granular>
# Settings follow DINO-WM for deformables: frameskip=1, num_hist=1, 20 epochs.
# Est. ~1 h on an A100 (1000 episodes x 20 steps).
# NOTE: planning on these envs needs NVIDIA FleX (not installed), so these
# checkpoints can be evaluated for prediction error but not planning success yet.
set -euo pipefail

OBJ="$1"; shift
E=/scratch/qxe3fj/.conda/envs/ts310
REPO=/sfs/weka/scratch/qxe3fj/temporal-straightening

unset VIRTUAL_ENV
export PATH="$E/bin:$PATH" PYTHONNOUSERSITE=1 WANDB_MODE=disabled HYDRA_FULL_ERROR=1
export DATASET_DIR=/sfs/weka/scratch/qxe3fj/data
export PYTHONPATH="$(ls -d $HOME/.cache/torch/hub/facebookresearch_dinov2_* | head -1)"

cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv
python train.py --config-name train.yaml \
  env=deformable_env env.kwargs.object_name="$OBJ" env.dataset.object_name="$OBJ" \
  +env.save_name="$OBJ" frameskip=1 num_hist=1 \
  encoder=dino training.straighten=False training.epochs="${EPOCHS:-20}" \
  env.num_workers="$SLURM_CPUS_PER_TASK" \
  ckpt_base_path=/sfs/weka/scratch/qxe3fj/JEPA/checkpoints/trained/dinowm \
  "$@"
