# Where do JEPA world models break?

We ran inference with two published JEPA-style latent world models on seven goal-reaching tasks. Each task was run at two difficulties: the goal distance used in the papers, and roughly twice that distance. No models were trained or fine-tuned. All numbers come from the authors' released checkpoints.

- **DINO-WM** ([Zhou et al.](https://arxiv.org/abs/2411.04983)): frozen DINOv2 patch features with a learned predictor. Checkpoints come from the authors' [OSF release](https://osf.io/bmw48/).
- **LeWM / LeWorldModel** ([paper](https://arxiv.org/abs/2603.19312)): an end-to-end JEPA trained from pixels. Checkpoints are on Hugging Face: `quentinll/lewm-{pusht,tworooms,reacher,cube}`.

## Results

Success rate over 50 episodes per setting, one seed each. "Normal" is the paper's goal distance (25 env steps). "Harder" is about 2× that.

| Model | Environment | Normal | Harder | Change |
|---|---|---|---|---|
| DINO-WM | PushT | 64% | **6%** (50 steps) | **−58** |
| LeWM | PushT | 88% | 52% (50 steps) | −36 |
| LeWM | TwoRoom | 86% | 50% (50 steps) | −36 |
| LeWM | OGBench-Cube | 74% | 52% (50 steps) | −22 |
| DINO-WM | PointMaze (UMaze) | 98% | 88% (50 steps) | −10 |
| LeWM | Reacher | 62% | 56% (50 steps) | −6 |
| DINO-WM | Wall | 82% | 80% (45 steps) | −2 |

The same data is in [`results/summary.csv`](results/summary.csv), with links to the raw result file for each number.

### Takeaways
- **Contact-rich pushing (PushT) breaks first.** DINO-WM collapses almost completely at the longer horizon, and LeWM loses over a third of its successes.
- **LeWM degrades on TwoRoom.** This matches the LeWM paper, which reports TwoRoom as a weakness.
- **Navigation is robust.** PointMaze and Wall barely change. Wall could only be stretched to 45 steps, because its dataset episodes are 50 steps long and goals are drawn from those episodes.

### Caveats
- With n = 50 and one seed, each number carries roughly ±7 points of noise. Differences under about 10 points are not meaningful yet.
- **The planners differ, so the two models are not directly comparable.** DINO-WM uses open-loop CEM: it plans once and executes the whole plan. LeWM uses CEM with replanning (MPC), so it can correct mistakes. Part of DINO-WM's PushT collapse may be planner failure, not world-model failure.
- The two models also sample start/goal pairs with different code and different seeds (DINO-WM seed 100, LeWM seed 42).

## Setup

**DINO-WM** was run with `plan.py` from the [temporal-straightening](https://github.com/agentic-learning-ai-lab/temporal-straightening) codebase (a DINO-WM fork):
- Planner config `plan_cem.yaml`: CEM, 10 optimisation steps, 200 samples, top-30.
- `goal_H` ∈ {25, 50} (45 for Wall).
- PushT uses `objective.alpha=1`, following the README there.

**LeWM** was run with `scripts/plan/eval_wm.py` from [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), using the configs shipped in that repo:
- CEM with 300 samples, horizon 5, replan every 5 steps.
- `goal_offset_steps` ∈ {25, 50}, with an action budget of 2× the offset.

Hardware: one NVIDIA A6000 per job. Run time:
- DINO-WM: about 1.5 h per environment for PushT and PointMaze, about 0.3 h for Wall.
- LeWM: about 4–5 min per environment.

## Repository layout

```
results/
  summary.csv                      # the table above
  dinowm/<env>/goalH_<H>_seed100.log   # full planning log; final "Success rate" line is the result
  dinowm/<env>/goalH_<H>_seed100/      # per-run hydra config + logs.json (per-episode metrics)
  lewm/<env>/offset_<G>/*_results.txt  # config + metrics (incl. per-episode successes)
logs/                              # SLURM stdout of the final runs
  failed_attempts/                 # earlier runs that crashed (config/env bugs), kept for transparency
slurm/
  dinowm_eval.sbatch               # DINO-WM inference launcher
  lewm_eval.sbatch                 # LeWM inference launcher
  lewm_train.sbatch                # training launcher; written but NOT used for these results
conf_overrides/plan_cem_fixed.yaml # plan_cem.yaml with a duplicate YAML key removed
fetch_lewm_data.sh                 # downloads the LeWM evaluation datasets from Hugging Face
```

Not included, because of size or third-party licensing: model checkpoints, datasets (~243 GB), the stable-worldmodel source, and rollout videos.

## Reproducing

The launchers hard-code paths on the original cluster (UVA HPC, SLURM). To rerun elsewhere, first edit the paths at the top of each `slurm/*.sbatch` file, then:
1. Download the DINO-WM checkpoints (`outputs.zip` from the OSF link above) and datasets.
2. Run `fetch_lewm_data.sh` and download the LeWM checkpoints from Hugging Face.
3. `sbatch slurm/dinowm_eval.sbatch <pusht|wall_single|point_maze>`, and `sbatch slurm/lewm_eval.sbatch <pusht|tworoom|reacher|cube>`.

Environment gotchas we hit:
- LeWM checkpoints need `transformers<5` and `datasets<5`.
- The PyPI build of `stable-worldmodel` (0.1.1) lacks `stable_worldmodel.planning`, so import it from a source checkout.
- DINO-WM PointMaze needs MuJoCo 2.1 and `mujoco_py`.
