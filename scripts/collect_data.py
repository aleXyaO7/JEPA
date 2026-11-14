"""Collect offline trajectories (pixels + actions) for LeWM training.

Only for environments WITHOUT an official pretrained JEPA checkpoint. Writes an
HDF5 dataset to $STABLEWM_HOME/datasets/<name>.h5, the layout lewm.py reads.

  python collect_data.py <env_key> [--episodes N] [--debug]

Policies:
  * ogb_expert : OGBench's scripted oracle (no checkpoint needed), Scene only.
  * random     : uniform random actions. Every other env uses this: the SAC
                 experts that stable-worldmodel's collect_dmc.py expects are not
                 published, and JEPA world models only need (pixels, action)
                 transitions. Random data under-covers goal-relevant states in
                 locomotion tasks; see README for that caveat.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy

# key: (env id, extra World kwargs, policy, max_episode_steps, default episodes)
# 2000 episodes x 200 steps ~ 60 GB of HDF5 pixels per env (~155 KB/frame).
ENVS = {
    # OGBench manipulation / navigation
    'ogb_scene': (
        'swm/OGBScene-v0',
        dict(multiview=False, width=224, height=224, visualize_info=False,
             terminate_at_goal=False, mode='data_collection'),
        'ogb_expert', 200, 2000,
    ),
    'ogb_pointmaze': (
        'swm/OGBMaze-v0',
        dict(loco_env_type='point', maze_env_type='maze', maze_type='medium',
             ob_type='pixels'),
        'random', 200, 2000,
    ),
    # DeepMind Control (random actions; expert SAC checkpoints are unpublished)
    'dmc_cheetah': ('swm/CheetahDMControl-v0', {}, 'random', 200, 2000),
    'dmc_walker': ('swm/WalkerDMControl-v0', {}, 'random', 200, 2000),
    'dmc_hopper': ('swm/HopperDMControl-v0', {}, 'random', 200, 2000),
    'dmc_quadruped': ('swm/QuadrupedDMControl-v0', {}, 'random', 200, 2000),
    'dmc_finger': ('swm/FingerDMControl-v0', {}, 'random', 200, 2000),
    'dmc_ballincup': ('swm/BallInCupDMControl-v0', {}, 'random', 200, 2000),
    'dmc_cartpole': ('swm/CartpoleDMControl-v0', {}, 'random', 200, 2000),
    'dmc_pendulum': ('swm/PendulumDMControl-v0', {}, 'random', 200, 2000),
    'dmc_acrobot': ('swm/AcrobotDMControl-v0', {}, 'random', 200, 2000),
    # Gymnasium classic control (continuous-action only: LeWM's action encoder
    # expects a continuous action vector, so discrete CartPole/Acrobot are out)
    'cc_mountaincar': ('swm/MountainCarContinuousControl-v0', {}, 'random', 200, 2000),
    'cc_pendulum': ('swm/PendulumControl-v1', {}, 'random', 200, 2000),
    # 2D point maze with the default (fixed) wall layout; layout generalisation
    # is tested later at eval time by resampling the `walls` factor.
    'simple_pointmaze': ('swm/SimplePointMaze-v0', {}, 'random', 200, 2000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('env', choices=sorted(ENVS))
    ap.add_argument('--episodes', type=int, default=None)
    ap.add_argument('--num-envs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=3072)
    ap.add_argument('--debug', action='store_true',
                    help='2 episodes of 20 steps into <name>_debug.h5')
    args = ap.parse_args()

    env_id, kwargs, policy, max_steps, n_default = ENVS[args.env]
    episodes = args.episodes or n_default
    name = args.env
    if args.debug:
        episodes, max_steps, name = 2, 20, f'{args.env}_debug'

    world = swm.World(
        env_id,
        num_envs=min(args.num_envs, episodes),
        image_shape=(224, 224),
        max_episode_steps=max_steps,
        **kwargs,
    )
    if policy == 'ogb_expert':
        from stable_worldmodel.envs.ogbench import ExpertPolicy
        world.set_policy(ExpertPolicy())
    else:
        world.set_policy(RandomPolicy(seed=args.seed))

    out = Path(swm.data.utils.get_cache_dir()) / 'datasets' / f'{name}.h5'
    out.parent.mkdir(parents=True, exist_ok=True)
    world.collect(out, episodes=episodes, seed=args.seed, format='hdf5')
    print(f'[collect] {env_id} -> {out} ({episodes} episodes x <= {max_steps} steps)')


if __name__ == '__main__':
    main()
