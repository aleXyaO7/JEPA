#!/usr/bin/env bash
#SBATCH --job-name=dinowm_eval
#SBATCH --account=raiselab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
#
# INFERENCE ONLY: plan with the official pretrained DINO-WM checkpoint (CEM, open-loop,
# temporal-straightening/plan.py) and report success rate. No training.
# Usage:  sbatch -J dinowm_eval_pusht slurm/dinowm_eval.sbatch <pusht|wall_single|point_maze>
# Goal horizons (env steps) swept via GOAL_HS (default "25 50": paper setting, then 2x farther).
set -uo pipefail

MODEL="$1"; shift
GOAL_HS="${GOAL_HS:-25 50}"
N_EVALS="${N_EVALS:-50}"
SEED="${SEED:-100}"
REPO=/sfs/weka/scratch/qxe3fj/temporal-straightening
E=/scratch/qxe3fj/.conda/envs/ts310
CKPT=/sfs/weka/scratch/qxe3fj/JEPA/checkpoints/dino_wm/outputs/$MODEL
OUT=/sfs/weka/scratch/qxe3fj/JEPA/results/dinowm/$MODEL

unset VIRTUAL_ENV
export PATH="$E/bin:$PATH" PYTHONNOUSERSITE=1 WANDB_MODE=disabled HYDRA_FULL_ERROR=1
export DATASET_DIR=/sfs/weka/scratch/qxe3fj/data
export MUJOCO_PY_FORCE_CPU=1 MUJOCO_GL=osmesa
export LD_LIBRARY_PATH="$E/lib:${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"
export PYTHONPATH="$(ls -d $HOME/.cache/torch/hub/facebookresearch_dinov2_* | head -1)"

EXTRA=()
[ "$MODEL" = pusht ] && EXTRA+=(objective.alpha=1)   # README: PushT needs alpha=1

cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p "$OUT"
for H in $GOAL_HS; do
  echo "===== $MODEL  goal_H=$H  n_evals=$N_EVALS  seed=$SEED ====="
  # plan_cem_fixed.yaml = repo conf/plan_cem.yaml minus its duplicate `sweep:` key (YAML error)
  python plan.py --config-dir /sfs/weka/scratch/qxe3fj/JEPA/conf_overrides --config-name plan_cem_fixed.yaml \
    ckpt_base_path="$CKPT" model_epoch=latest \
    goal_H="$H" planner.sub_planner.horizon="$H" planner.n_taken_actions="$H" \
    n_evals="$N_EVALS" seed="$SEED" decode_for_viz=false \
    hydra.run.dir="$OUT/goalH_${H}_seed${SEED}" \
    "${EXTRA[@]}" "$@" 2>&1 | tee "$OUT/goalH_${H}_seed${SEED}.log"
  grep -E "Success rate" "$OUT/goalH_${H}_seed${SEED}.log" | tail -1
done
