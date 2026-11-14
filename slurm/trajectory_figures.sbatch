#!/usr/bin/env bash
#SBATCH --job-name=traj_figures
#SBATCH --account=raiselab
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/sfs/weka/scratch/qxe3fj/JEPA/logs/%x_%j.out
# Build figures for every traced trajectory under results/trajectories/*.
set -uo pipefail
J=/sfs/weka/scratch/qxe3fj/JEPA
export PYTHONNOUSERSITE=1
SPECS=()
for d in $J/results/trajectories/*/; do
  [ -f "$d/trace_meta.json" ] && SPECS+=("$(basename $d)=$d")
done
/scratch/qxe3fj/.conda/envs/swm/bin/python $J/scripts/make_trajectory_figures.py $J/figures/trajectories "${SPECS[@]}"
