#!/usr/bin/env bash
#SBATCH --account=raiselab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
#
# INFERENCE ONLY. Re-run the normal-goal-distance evals with full trajectory
# logging (same seeds/configs as the original runs -> same success rates) and
# keep the first successful + first failed episode per environment.
# Usage: sbatch -J trace_lewm   --time=01:00:00 slurm/trace_trajectories.sbatch lewm  [pusht tworoom reacher cube]
#        sbatch -J trace_dinowm_pusht --time=02:30:00 slurm/trace_trajectories.sbatch dinowm pusht
set -uo pipefail
KIND="$1"; shift
J=/sfs/weka/scratch/qxe3fj/JEPA
OUTROOT=$J/results/trajectories
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ "$KIND" = lewm ]; then
  SWM=$J/src/stable-worldmodel
  unset VIRTUAL_ENV
  export PYTHONPATH="$SWM" PYTHONNOUSERSITE=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
  export STABLEWM_HOME=$J/stablewm_home MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  cd "$SWM/scripts/plan"
  ENVS=("$@"); [ ${#ENVS[@]} -eq 0 ] && ENVS=(pusht tworoom reacher cube)
  for ENVN in "${ENVS[@]}"; do
    case "$ENVN" in
      pusht)   CFG=pusht;   CKPT=lewm-pusht;    EXTRA=() ;;
      tworoom) CFG=tworoom; CKPT=lewm-tworooms; EXTRA=() ;;
      reacher) CFG=reacher; CKPT=lewm-reacher;  EXTRA=(eval.dataset_name=reacher.h5) ;;
      cube)    CFG=cube;    CKPT=lewm-cube;     EXTRA=() ;;
    esac
    OUT=$OUTROOT/lewm_$ENVN
    echo "===== trace LeWM $ENVN (goal offset 25, budget 50) ====="
    TRACE_OUT=$OUT /scratch/qxe3fj/.conda/envs/swm/bin/python $J/scripts/trace_lewm.py --config-path $SWM/scripts/plan/config \
      --config-name "$CFG" policy="$CKPT" eval.goal_offset_steps=25 eval.eval_budget=50 \
      output.filename="trace_${ENVN}_offset25_results.txt" hydra.run.dir="$OUT/hydra" "${EXTRA[@]}"
  done
else
  MODEL="$1"
  E=/scratch/qxe3fj/.conda/envs/ts310
  export PATH="$E/bin:$PATH" PYTHONNOUSERSITE=1 WANDB_MODE=disabled HYDRA_FULL_ERROR=1
  export DATASET_DIR=/sfs/weka/scratch/qxe3fj/data MUJOCO_PY_FORCE_CPU=1 MUJOCO_GL=osmesa
  export LD_LIBRARY_PATH="$E/lib:${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"
  export PYTHONPATH="$(ls -d $HOME/.cache/torch/hub/facebookresearch_dinov2_* | head -1)"
  EXTRA=(); [ "$MODEL" = pusht ] && EXTRA+=(objective.alpha=1)
  OUT=$OUTROOT/dinowm_$MODEL
  echo "===== trace DINO-WM $MODEL (goal_H 25, 50 evals, seed 100) ====="
  cd /sfs/weka/scratch/qxe3fj/temporal-straightening
  TRACE_OUT=$OUT python $J/scripts/trace_dinowm.py --config-dir $J/conf_overrides --config-name plan_cem_fixed.yaml \
    ckpt_base_path=$J/checkpoints/dino_wm/outputs/$MODEL model_epoch=latest \
    goal_H=25 planner.sub_planner.horizon=25 planner.n_taken_actions=25 \
    n_evals=50 seed=100 decode_for_viz=false hydra.run.dir="$OUT/hydra" "${EXTRA[@]}"
fi
