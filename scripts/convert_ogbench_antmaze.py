"""Build a LeWM training set for Visual AntMaze from OGBench's expert data.

A random policy barely moves the ant, so we use OGBench's
visual-antmaze-medium-navigate-v0 trajectories (a pretrained ant locomotion
policy following maze waypoints). Each stored state (qpos/qvel) is replayed in
stable-worldmodel's swm/OGBMaze-v0 (ant, medium, pixels) and re-rendered at
224x224 with OGBench's tracking camera ('back') and position-coloured
floor (verified: ~2/255 mean abs diff vs the dataset's own 64px frames), so training frames look exactly like what the env renders at eval
time. The OGBench episodes (1000 steps) are cut into 200-step chunks.

  python convert_ogbench_antmaze.py <ogbench.npz> [--episodes 2000] [--chunk 200]
Output: $STABLEWM_HOME/datasets/ogb_antmaze.h5
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import numpy as np
from tqdm import tqdm

import stable_worldmodel as swm  # noqa: F401  (registers swm/ envs)
from stable_worldmodel.data.formats.hdf5 import HDF5Writer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--episodes', type=int, default=2000)
    ap.add_argument('--chunk', type=int, default=200)
    ap.add_argument('--name', default='ogb_antmaze')
    args = ap.parse_args()

    d = np.load(args.npz)
    print({k: d[k].shape for k in d.files})
    qpos, qvel, act = d['qpos'], d['qvel'], d['actions']
    term = d['terminals'].astype(bool)
    ends = np.nonzero(term)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])

    env = gym.make('swm/OGBMaze-v0', loco_env_type='ant', maze_env_type='maze',
                   maze_type='medium', ob_type='pixels', width=224, height=224,
                   camera_name='back')  # OGBench visual-* tracking camera
    # OGBench's visual envs colour the floor by (x, y) so the agent can infer
    # its position from an egocentric frame; swm turns this off by default.
    env.reset(seed=0, options={'variation': [], 'variation_values': {
        'floor.pixel_encoding': np.array(1)}})
    base = env.unwrapped

    out = Path(os.environ['STABLEWM_HOME']) / 'datasets' / f'{args.name}.h5'
    n_ep = 0
    with HDF5Writer(out, mode='overwrite') as w, tqdm(total=args.episodes) as bar:
        for s, e in zip(starts, ends):
            for c0 in range(s, e + 1 - args.chunk + 1, args.chunk):
                if n_ep >= args.episodes:
                    break
                idx = np.arange(c0, c0 + args.chunk)
                pix = np.empty((len(idx), 224, 224, 3), np.uint8)
                for j, t in enumerate(idx):
                    base.set_state(qpos[t], qvel[t])
                    pix[j] = np.asarray(base.render())
                w.write_episode({
                    'pixels': pix,
                    'action': act[idx].astype(np.float32),
                    'qpos': qpos[idx].astype(np.float32),
                    'qvel': qvel[idx].astype(np.float32),
                    'pos_agent': qpos[idx, :2].astype(np.float32),
                    'ep_idx': np.full(len(idx), n_ep, np.int32),
                    'step_idx': np.arange(len(idx), dtype=np.int64),
                })
                n_ep += 1
                bar.update(1)
            if n_ep >= args.episodes:
                break
    print(f'[convert] {n_ep} episodes x {args.chunk} steps -> {out}')


if __name__ == '__main__':
    main()
