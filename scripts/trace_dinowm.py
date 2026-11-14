"""Run temporal-straightening plan.py UNCHANGED, saving the final rollouts.

plan.py's final evaluation (PlanEvaluator.eval_actions(..., filename=
'output_final')) rolls the planned actions out in the real simulator and
discards the trajectories. This wrapper keeps them, and also decodes the world
model's *imagined* rollout of the same actions with the checkpoint's decoder.
Planning itself is untouched (decode_for_viz stays false), so success rates
match the stock run with the same seed.

Output (TRACE_OUT dir):
  all_episodes.npz : success, executed actions, real env states for all episodes
  ep<i>_{success,failure}.npz : real frames, imagined (decoded) frames, start and
                     goal frames, states, actions for the first success / failure.
Usage: same args as plan.py, run from the temporal-straightening repo, plus TRACE_OUT.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = os.environ.get('TS_REPO', '/sfs/weka/scratch/qxe3fj/temporal-straightening')
sys.path.insert(0, REPO)
import runpy  # noqa: E402
from planning.evaluator import PlanEvaluator  # noqa: E402
from utils import move_to_device  # noqa: E402
from einops import rearrange  # noqa: E402

OUT = Path(os.environ['TRACE_OUT'])
OUT.mkdir(parents=True, exist_ok=True)
_orig = PlanEvaluator.eval_actions


def _np(x):
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)


def _to_uint8_hwc(v):
    """(..., C, H, W) or (..., H, W, C), in [-1,1], [0,1] or [0,255] -> uint8 HWC."""
    v = _np(v)
    if v.shape[-1] not in (1, 3) and v.shape[-3] in (1, 3):
        v = np.moveaxis(v, -3, -1)
    if v.dtype != np.uint8:
        lo, hi = float(v.min()), float(v.max())
        if lo < -0.01:
            v = (v + 1) / 2
        if hi > 1.5:
            v = v / 255.0
        v = (np.clip(v, 0, 1) * 255).round().astype(np.uint8)
    return v


def eval_actions(self, actions, action_len=None, filename='output', save_video=False):
    logs, successes, e_obses, e_states = _orig(
        self, actions, action_len, filename=filename, save_video=save_video
    )
    if filename != 'output_final':
        return logs, successes, e_obses, e_states

    succ = np.asarray(successes, dtype=bool).reshape(-1)
    exec_actions = rearrange(actions.cpu(), 'b t (f d) -> b (t f) d', f=self.frameskip)
    exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
    real = _to_uint8_hwc(e_obses['visual'])           # (n, T, H, W, C)
    states = _np(e_states)
    start = _to_uint8_hwc(self.obs_0['visual'])[:, -1]
    goal = _to_uint8_hwc(self.obs_g['visual'])[:, -1]

    # Decode the imagined rollout only for the episodes we keep (memory).
    keep = [int(np.flatnonzero(c)[0]) for c in (succ, ~succ) if c.any()]
    imagined = {}
    if keep and getattr(self.wm, 'decoder', None) is not None:
        with torch.no_grad():
            obs0 = move_to_device(self.preprocessor.transform_obs(
                {k: v[keep] for k, v in self.obs_0.items()}), self.device)
            z, _ = self.wm.rollout(obs_0=obs0, act=actions[keep])
            vis = _to_uint8_hwc(self.wm.decode_obs(z)[0]['visual'])  # (k, t+1, H, W, C)
        imagined = {i: vis[j] for j, i in enumerate(keep)}

    np.savez_compressed(
        OUT / 'all_episodes.npz', success=succ, actions=exec_actions, states=states,
        state_0=_np(self.state_0), state_g=_np(self.state_g), frameskip=self.frameskip,
    )
    picked = {}
    for tag, cond in (('success', succ), ('failure', ~succ)):
        idx = np.flatnonzero(cond)
        if len(idx) == 0:
            continue
        i = int(idx[0])
        extra = {'imagined': imagined[i]} if i in imagined else {}
        np.savez_compressed(
            OUT / f'ep{i}_{tag}.npz', frames=real[i], start=start[i], goal=goal[i],
            states=states[i], actions=exec_actions[i], state_0=_np(self.state_0)[i],
            state_g=_np(self.state_g)[i], frameskip=self.frameskip, **extra,
        )
        picked[tag] = i
    meta = {'n_episodes': int(len(succ)), 'success_rate': float(succ.mean() * 100),
            'picked': picked, 'real_steps': int(real.shape[1]),
            'imagined_steps': {str(k): int(v.shape[0]) for k, v in imagined.items()},
            'frameskip': int(self.frameskip), 'argv': sys.argv[1:]}
    (OUT / 'trace_meta.json').write_text(json.dumps(meta, indent=2))
    print('[trace]', json.dumps(meta), flush=True)
    return logs, successes, e_obses, e_states


PlanEvaluator.eval_actions = eval_actions

if __name__ == '__main__':
    os.chdir(REPO)
    # Execute plan.py as __main__ so its @hydra.main resolves conf/ next to plan.py;
    # it imports the already-patched planning.evaluator module from sys.modules.
    sys.argv[0] = os.path.join(REPO, 'plan.py')
    runpy.run_path(sys.argv[0], run_name='__main__')
