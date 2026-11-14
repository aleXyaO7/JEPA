#!/usr/bin/env bash
#SBATCH --job-name=intphys2_vjepa2
#SBATCH --account=raiselab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
#
# INFERENCE ONLY: V-JEPA 2 ViT-L (native checkpoint incl. predictor) surprise on
# IntPhys 2 with Meta's prediction_evals code (src/IntPhys2), then pairwise scoring.
# Usage: sbatch -J intphys2_debug slurm/intphys2_vjepa2.sbatch debug   (60 videos)
#        sbatch -J intphys2_main --time=<from debug timing> slurm/intphys2_vjepa2.sbatch main
set -euo pipefail
SPLIT="$1"
J=/sfs/weka/scratch/qxe3fj/JEPA
PY=/scratch/qxe3fj/.conda/envs/ip2venv/bin/python
export PYTHONNOUSERSITE=1
cd $J/src/IntPhys2/prediction_evals
nvidia-smi --query-gpu=name,memory.total --format=csv
t0=$(date +%s)
$PY -m evals.main --debugmode True --devices cuda:0 --fname $J/conf_overrides/intphys2/vjepa2_vitl_$SPLIT.yaml
echo "eval seconds: $(( $(date +%s) - t0 ))"
OUT=$J/results/intphys2/vjepa2_vitl/intphys_v2/intphys2-$SPLIT-vitl
case $SPLIT in debug) META=Debug;; main) META=Main;; esac
for f in $OUT/losses_*.pth; do
  $PY $J/scripts/score_intphys2.py "$f" $J/raw_downloads/intphys2/$META/metadata.csv "${f%.pth}_scores.json"
done
