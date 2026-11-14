#!/usr/bin/env bash
#SBATCH --job-name=collect
#SBATCH --account=raiselab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
#
# Collect an offline dataset for an env WITHOUT an official JEPA checkpoint.
# Measured (A6000): 20 eps = 24 s (DMC cheetah), 50 s (OGB scene) -> 2000 eps ~25-70 min; 3 h limit.
# GPU partition only because MuJoCo renders through EGL (OSMesa fails for DMC/OGBench).
# Usage:  sbatch -J collect_dmc_cheetah slurm/collect_data.sbatch dmc_cheetah [--episodes N]
# Output: $STABLEWM_HOME/datasets/<env_key>.h5  (~60 GB at the default 2000 episodes)
set -euo pipefail

KEY="$1"; shift
unset VIRTUAL_ENV
export PYTHONPATH=/sfs/weka/scratch/qxe3fj/JEPA/src/stable-worldmodel
export PYTHONNOUSERSITE=1
export STABLEWM_HOME=/sfs/weka/scratch/qxe3fj/JEPA/stablewm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

/scratch/qxe3fj/.conda/envs/swm/bin/python /sfs/weka/scratch/qxe3fj/JEPA/scripts/collect_data.py "$KEY" "$@"
