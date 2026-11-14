#!/usr/bin/env bash
#SBATCH --job-name=convert_antmaze
#SBATCH --account=raiselab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
# Re-render OGBench visual-antmaze-medium-navigate expert data at 224px into
# $STABLEWM_HOME/datasets/ogb_antmaze.h5 (2000 x 200-step episodes, ~60 GB).
# GPU only for EGL rendering. ~400k frames; expect < 1 h at GPU-EGL speeds.
set -euo pipefail
unset VIRTUAL_ENV
export PYTHONPATH=/sfs/weka/scratch/qxe3fj/JEPA/src/stable-worldmodel PYTHONNOUSERSITE=1
export STABLEWM_HOME=/sfs/weka/scratch/qxe3fj/JEPA/stablewm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
/scratch/qxe3fj/.conda/envs/swm/bin/python /sfs/weka/scratch/qxe3fj/JEPA/scripts/convert_ogbench_antmaze.py \
  /sfs/weka/scratch/qxe3fj/JEPA/raw_downloads/ogbench/visual-antmaze-medium-navigate-v0.npz "$@"
