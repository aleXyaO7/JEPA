#!/usr/bin/env bash
#SBATCH --job-name=lewm_eval
#SBATCH --account=raiselab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
#
# INFERENCE ONLY: MPC (CEM) with the official pretrained LeWM checkpoint
# (HF quentinll/lewm-<env>, cached in $STABLEWM_HOME/checkpoints) via
# stable-worldmodel scripts/plan/eval_wm.py. No training.
# Usage:  sbatch -J lewm_eval_tworoom slurm/lewm_eval.sbatch <pusht|tworoom|reacher|cube>
# Goal offsets swept via GOAL_OFFSETS (default "25 50": paper setting, then 2x farther);
# eval budget = 2 x offset, as in the shipped configs.
set -uo pipefail

ENVN="$1"; shift
GOAL_OFFSETS="${GOAL_OFFSETS:-25 50}"
SWM=/sfs/weka/scratch/qxe3fj/JEPA/src/stable-worldmodel
PY=/scratch/qxe3fj/.conda/envs/swm/bin/python

case "$ENVN" in
  pusht)   CFG=pusht;   CKPT=lewm-pusht;    EXTRA=() ;;
  tworoom) CFG=tworoom; CKPT=lewm-tworooms; EXTRA=() ;;
  reacher) CFG=reacher; CKPT=lewm-reacher;  EXTRA=(eval.dataset_name=reacher.h5) ;;
  cube)    CFG=cube;    CKPT=lewm-cube;     EXTRA=() ;;
  *) echo "unknown env $ENVN"; exit 1 ;;
esac

unset VIRTUAL_ENV
# Import stable_worldmodel from the cloned source: the PyPI 0.1.1 build lacks the
# `stable_worldmodel.planning` module that scripts/plan configs reference.
export PYTHONPATH="$SWM"
export PYTHONNOUSERSITE=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export STABLEWM_HOME=/sfs/weka/scratch/qxe3fj/JEPA/stablewm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

cd "$SWM/scripts/plan"
nvidia-smi --query-gpu=name,memory.total --format=csv
for G in $GOAL_OFFSETS; do
  B=$((2 * G))
  echo "===== $ENVN  policy=$CKPT  goal_offset=$G  eval_budget=$B ====="
  "$PY" eval_wm.py --config-name "$CFG" policy="$CKPT" \
    eval.goal_offset_steps="$G" eval.eval_budget="$B" \
    output.filename="${ENVN}_offset${G}_results.txt" \
    hydra.run.dir="/sfs/weka/scratch/qxe3fj/JEPA/results/lewm/$ENVN/offset_$G" \
    "${EXTRA[@]}" "$@"
done
echo "results: $STABLEWM_HOME/checkpoints/$CKPT/*_results.txt"
