#!/usr/bin/env bash
#SBATCH --job-name=bench_speed
#SBATCH --account=raiselab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
#
# Measure (1) LeWM training throughput and (2) data-collection throughput, to
# size the time limits of the real collection/training jobs. Writes only to a
# throwaway STABLEWM_HOME (official data is symlinked in read-only).
set -uo pipefail
SWM=/sfs/weka/scratch/qxe3fj/JEPA/src/stable-worldmodel
PY=/scratch/qxe3fj/.conda/envs/swm/bin/python
BENCH=/sfs/weka/scratch/qxe3fj/JEPA/bench_tmp
mkdir -p $BENCH/datasets
ln -sfn /sfs/weka/scratch/qxe3fj/JEPA/stablewm_home/datasets/tworoom.h5 $BENCH/datasets/tworoom.h5

unset VIRTUAL_ENV
export PYTHONPATH="$SWM" PYTHONNOUSERSITE=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export STABLEWM_HOME=$BENCH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "===== LeWM train benchmark (tworoom, 300 steps, batch 128) ====="
cd "$SWM/scripts/train"
T0=$(date +%s)
"$PY" lewm.py data=tworoom output_model_name=bench subdir=bench \
  num_workers=8 +trainer.max_steps=300 +trainer.limit_val_batches=5 2>&1 \
  | tr '\r' '\n' | grep -E "it/s|s/it|Epoch 0" | tail -5
echo "LEWM_300_STEPS_SECONDS=$(( $(date +%s) - T0 ))"

for K in dmc_cheetah ogb_scene; do
  echo "===== collect benchmark: $K (20 episodes x 200 steps) ====="
  T0=$(date +%s)
  "$PY" /sfs/weka/scratch/qxe3fj/JEPA/scripts/collect_data.py $K --episodes 20 2>&1 | grep -E "\[collect\]|Error" | tail -2
  echo "COLLECT_${K}_20EP_SECONDS=$(( $(date +%s) - T0 ))"
done
ls -la $BENCH/datasets
