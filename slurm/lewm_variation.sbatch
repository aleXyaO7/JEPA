#!/usr/bin/env bash
#SBATCH --job-name=lewm_var
#SBATCH --account=raiselab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
#
# INFERENCE ONLY: LeWM variation (distribution-shift) sweep with the official
# pretrained checkpoint. One factor of variation per run, normal goal distance
# (25 steps), 50 episodes, same start/goal pairs as the baseline eval.
# Usage:  sbatch -J lewm_var_pusht slurm/lewm_variation.sbatch <pusht|tworoom>
set -uo pipefail

ENVN="$1"; shift
SWM=/sfs/weka/scratch/qxe3fj/JEPA/src/stable-worldmodel
PY=/scratch/qxe3fj/.conda/envs/swm/bin/python
OUT=/sfs/weka/scratch/qxe3fj/JEPA/results/lewm_variation/$ENVN

case "$ENVN" in
  pusht)
    CFG=pusht; CKPT=lewm-pusht
    # 'none' = no-shift baseline through the same code path (sanity check vs 88%).
    FACTORS="${FACTORS:-none background.color agent.color block.color goal.color agent.scale block.scale agent.shape block.shape}" ;;
  tworoom)
    CFG=tworoom; CKPT=lewm-tworooms
    # Wall/door geometry is excluded: it would invalidate the replayed start/goal positions.
    FACTORS="${FACTORS:-none background.color agent.color target.color wall.color door.color agent.radius=12 target.radius agent.speed}" ;;
  *) echo "unknown env $ENVN"; exit 1 ;;
esac

unset VIRTUAL_ENV
export PYTHONPATH="$SWM"            # library from source (PyPI build lacks .planning)
export PYTHONNOUSERSITE=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export STABLEWM_HOME=/sfs/weka/scratch/qxe3fj/JEPA/stablewm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

cd "$SWM/scripts/plan"
nvidia-smi --query-gpu=name,memory.total --format=csv
for F in $FACTORS; do
  echo "===== $ENVN  variation=$F ====="
  "$PY" /sfs/weka/scratch/qxe3fj/JEPA/scripts/eval_variation.py \
    --config-name "$CFG" policy="$CKPT" \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    +variation="$F" +out_dir="$OUT" \
    hydra.run.dir="$OUT/$F/hydra" "$@"
done
echo "== summary =="; grep -h '"success_rate"\|"variation"' "$OUT"/*/result.json | paste - -
