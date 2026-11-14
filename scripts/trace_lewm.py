"""Run stable-worldmodel's stock eval_wm.py UNCHANGED while recording every step.

Two methods of swm.World are wrapped (no library edits):
  * _get_actions : records the action the CEM-MPC policy executes.
  * _run         : chains a logger after eval_wm's own on_step callback, so
                   it sees the same infos (goal re-injected) that eval_wm sees.
Success rates therefore match the stock eval exactly (same seed, same batch).

Output (TRACE_OUT dir):
  all_episodes.npz : per-episode success, first-success step, actions and
                     low-dimensional info keys (state/proprio/pos_*) for all
                     episodes (no pixels).
  ep<i>_{success,failure}.npz : full per-step frames + goal frame + actions +
                     low-dim info for the FIRST successful and FIRST failed
                     episode (by episode index).
Usage: identical to eval_wm.py, plus env TRACE_OUT=/path/to/out
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

PLAN_DIR = os.environ.get(
    'SWM_PLAN_DIR', '/sfs/weka/scratch/qxe3fj/JEPA/src/stable-worldmodel/scripts/plan'
)
sys.path.insert(0, PLAN_DIR)

import stable_worldmodel as swm  # noqa: E402
import eval_wm  # noqa: E402  (its @hydra.main resolves ./config relative to eval_wm.py)

OUT = Path(os.environ['TRACE_OUT'])
MIN_SUCCESS_STEP = 5
OUT.mkdir(parents=True, exist_ok=True)

LOG = {'frames': [], 'goal': None, 'actions': [], 'lowdim': [], 'term': []}
STATE = {'calls': 0, 'active': False}
_orig_get_actions = swm.World._get_actions
_orig_run = swm.World._run


def _last(x):
    x = np.asarray(x)
    return x[:, -1] if x.ndim >= 2 and x.shape[1] >= 1 else x


def _lowdim(infos):
    out = {}
    for k, v in infos.items():
        if k in ('pixels', 'goal') or k.startswith('_'):
            continue
        try:
            a = np.asarray(v)
        except Exception:
            continue
        if a.dtype.kind not in 'fiub' or a.ndim < 2:
            continue
        a = _last(a)  # drop history dim -> (n, ...)
        if a[0].size <= 64:
            out[k] = a.astype(np.float32).copy()
    return out


def _get_actions(self):
    a = _orig_get_actions(self)
    if STATE['active']:
        LOG['actions'].append(np.asarray(a, dtype=np.float32).copy())
    return a


def _record(world, term):
    pix = np.asarray(world.infos['pixels'])
    LOG['frames'].append((pix[:, -1] if pix.ndim == 5 else pix).copy())
    LOG['lowdim'].append(_lowdim(world.infos))
    LOG['term'].append(np.asarray(term, dtype=bool).copy())


def _run(self, *args, on_step=None, **kw):
    # eval_wm may call evaluate twice (compile warm-up); keep the last call only.
    for k in ('frames', 'actions', 'lowdim', 'term'):
        LOG[k] = []
    STATE['active'] = True
    STATE['calls'] += 1
    # Initial observation: the env is already reset and the dataset goal injected.
    g = np.asarray(self.infos['goal'])
    LOG['goal'] = (g[:, -1] if g.ndim == 5 else g).copy()
    _record(self, np.zeros(self.num_envs, dtype=bool))

    def logged(world, mask):
        if on_step is not None:
            on_step(world, mask)
        _record(world, world.terminateds)

    try:
        return _orig_run(self, *args, on_step=logged, **kw)
    finally:
        STATE['active'] = False


swm.World._get_actions = _get_actions
swm.World._run = _run


def dump():
    frames = np.stack(LOG['frames'])            # (T+1, n, H, W, C): reset obs + T steps
    term = np.stack(LOG['term'])                # (T+1, n)
    acts = np.stack(LOG['actions'])             # (T, n, A)  action leading to frame t+1
    keys = sorted(set().union(*[d.keys() for d in LOG['lowdim']]))
    low = {k: np.stack([d[k] for d in LOG['lowdim'] if k in d]) for k in keys
           if all(k in d for d in LOG['lowdim'])}
    success = term.any(0)
    first = np.where(success, term.argmax(0), -1)
    n = frames.shape[1]
    np.savez_compressed(
        OUT / 'all_episodes.npz', success=success, first_success_step=first,
        actions=acts.transpose(1, 0, 2), **{f'info_{k}': v.swapaxes(0, 1) for k, v in low.items()},
    )
    picked = {}
    # Prefer a non-trivial success (goal not already met within the first 5
    # steps; e.g. ~1/3 of Cube successes happen at step 1).
    nontrivial = success & (first >= MIN_SUCCESS_STEP)
    for tag, cond in (('success', nontrivial if nontrivial.any() else success),
                      ('failure', ~success)):
        idx = np.flatnonzero(cond)
        if len(idx) == 0:
            continue
        i = int(idx[0])
        np.savez_compressed(
            OUT / f'ep{i}_{tag}.npz', frames=frames[:, i], goal=LOG['goal'][i],
            actions=acts[:, i], terminated=term[:, i], first_success_step=first[i],
            **{f'info_{k}': v[:, i] for k, v in low.items()},
        )
        picked[tag] = i
    meta = {'n_episodes': int(n), 'success_rate': float(success.mean() * 100),
            'trivial_successes_lt5': int((success & (first < MIN_SUCCESS_STEP)).sum()),
            'picked': picked, 'steps_recorded': int(frames.shape[0]),
            'lowdim_keys': list(low), 'argv': sys.argv[1:]}
    (OUT / 'trace_meta.json').write_text(json.dumps(meta, indent=2))
    print('[trace]', json.dumps(meta))


if __name__ == '__main__':
    try:
        eval_wm.run()
    finally:
        if LOG['frames']:
            dump()
