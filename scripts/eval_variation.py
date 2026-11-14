"""Variation (distribution-shift) stress test for pretrained LeWM checkpoints.

INFERENCE ONLY. Same protocol as stable-worldmodel's scripts/plan/eval_wm.py
(start/goal pairs replayed from the expert dataset, CEM-MPC planning), plus one
change: each run resamples ONE factor of variation (e.g. block.color) at reset,
then re-renders BOTH the current frame and the goal image inside the varied
scene. Without the re-render, the planner would compare a varied observation
against a default-appearance goal image taken from the dataset.

Run (from stable-worldmodel/scripts/plan so the shipped configs resolve):
  python eval_variation.py --config-name pusht policy=lewm-pusht \
      +variation=block.color +out_dir=/path/to/results

`+variation=none` reproduces the no-shift baseline through this same code
path; use it as the sanity check against the stock eval_wm.py numbers.
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal

CONFIG_DIR = os.environ.get(
    'SWM_PLAN_CONFIG',
    '/sfs/weka/scratch/qxe3fj/JEPA/src/stable-worldmodel/scripts/plan/config',
)

# Per-env dataset columns holding the low-dim state used by _set_state.
STATE_KEYS = {
    'swm/PushT-v1': ('state', 'goal_state'),
    'swm/TwoRoom-v1': ('pos_agent', 'goal_pos_agent'),
}


def img_transform(cfg, dtype=torch.float32):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def episode_col(dataset):
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


def render_resized(env, hw):
    img = np.asarray(env.render())
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = img.transpose(1, 2, 0)
    img = img.astype(np.uint8)
    h, w = hw
    return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def render_goal(env, goal_state, init_state, hw):
    """Render the goal configuration in the current (varied) scene, then put
    the env back exactly as it was."""
    if hasattr(env, 'block') and hasattr(env, 'agent'):
        # PushT: _set_state() advances physics one tick, so it cannot be used
        # to restore. Pose the pymunk bodies directly and snapshot/restore them.
        a, b = env.agent, env.block
        saved = (
            tuple(a.position), tuple(a.velocity),
            tuple(b.position), float(b.angle), tuple(b.velocity),
            float(b.angular_velocity),
        )
        g = np.asarray(goal_state, dtype=np.float64).tolist()
        # Angle BEFORE position (pymunk applies position relative to the
        # current angle when the centre of gravity is offset), as _set_state does.
        a.position = g[:2]
        b.angle = g[4]
        b.position = g[2:4]
        # debug_draw uses cached shape transforms: refresh without stepping.
        env.space.reindex_shapes_for_body(a)
        env.space.reindex_shapes_for_body(b)
        img = render_resized(env, hw)
        a.position, a.velocity = saved[0], saved[1]
        b.angle = saved[3]
        b.position, b.velocity, b.angular_velocity = saved[2], saved[4], saved[5]
        env.space.reindex_shapes_for_body(a)
        env.space.reindex_shapes_for_body(b)
        return img
    # TwoRoom: _set_state() only sets the agent position.
    env._set_state(goal_state)
    img = render_resized(env, hw)
    env._set_state(init_state)
    return img


def evaluate_varied(world, dataset, episodes_idx, start_steps, cfg, variation):
    """Dataset-replay evaluation with one resampled factor of variation."""
    n = len(episodes_idx)
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, episodes_idx, start_steps, cfg.eval.goal_offset_steps
    )
    seeds = [int(cfg.seed) * 1000 + i for i in range(n)]
    if variation == 'none':
        opts = None
    elif '=' in variation:
        # Fixed value, e.g. "agent.radius=12" (random draws of some factors
        # violate env constraints such as door size >= 1.1 x agent radius).
        key, val = variation.split('=', 1)
        opts = {
            'variation': [],
            'variation_values': {key: np.array([float(val)], dtype=np.float32)},
        }
    else:
        opts = {'variation': [variation]}
    world.reset(seed=seeds, options=opts)

    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    for i in range(n):
        _apply_callables(
            world.envs.envs[i].unwrapped, callables, {**init_rows[i], **goal_rows[i]}
        )

    # Re-render current frame and goal image inside the (possibly varied) scene.
    s_key, g_key = STATE_KEYS[cfg.world.env_name]
    pix = world.infos['pixels']  # (n, T, H, W, C)
    hw = pix.shape[2:4]
    goals = []
    for i in range(n):
        env = world.envs.envs[i].unwrapped
        if hasattr(env, 'space'):
            # PushT: pymunk's draw cache lags one physics tick behind the bodies;
            # sync it so the start frame and the restore check use true poses.
            for body in (env.agent, env.block):
                env.space.reindex_shapes_for_body(body)
        cur = render_resized(env, hw)
        goal = render_goal(env, goal_rows[i][g_key], init_rows[i][s_key], hw)
        assert np.array_equal(render_resized(env, hw), cur), 'state restore failed'
        pix[i] = cur[None]
        goals.append(goal)
    goal_img = np.broadcast_to(
        np.stack(goals)[:, None], pix.shape
    ).copy()

    # Inject low-dim dataset columns (proprio/state + goal_*) as eval_wm does,
    # but NOT pixels/goal images, which come from the re-render above.
    prefix = pix.shape[:2]
    goal_keys = set(goal_rows[0])
    for rows in (init_rows, goal_rows):
        for key in rows[0]:
            if key in ('action', 'pixels', 'goal'):
                continue
            if key not in world.infos and key not in goal_keys:
                continue
            vals = [r[key] for r in rows]
            if not all(isinstance(v, np.ndarray) for v in vals):
                continue
            if len({v.shape for v in vals}) > 1:
                continue
            st = np.stack(vals)
            world.infos[key] = np.broadcast_to(
                st[:, None, ...], prefix + st.shape[1:]
            ).copy()
    world.infos['goal'] = goal_img
    snapshot = {k: world.infos[k].copy() for k in goal_keys if k in world.infos}
    snapshot['goal'] = goal_img

    successes = np.zeros(n, dtype=bool)

    def on_step(w, mask):
        w.infos.update({k: v.copy() for k, v in snapshot.items()})
        successes[:] |= w.terminateds

    world._run(max_steps=cfg.eval.eval_budget, mode='wait', on_step=on_step)
    return {
        'success_rate': float(successes.sum()) / n * 100.0,
        'episode_successes': successes.tolist(),
        'goal_frames': goals,
        'start_frames': [pix[i, -1] for i in range(min(n, 8))],
    }


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name='pusht')
def run(cfg: DictConfig):
    variation = str(cfg.get('variation', 'none'))
    out_dir = Path(cfg.get('out_dir', '.')) / variation
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {'pixels': img_transform(cfg), 'goal': img_transform(cfg)}

    dataset = swm.data.load_dataset(
        cfg.eval.dataset_name, keys_to_cache=list(cfg.dataset.keys_to_cache)
    )
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == 'pixels':
            continue
        p = preprocessing.StandardScaler()
        d = dataset.get_col_data(col)
        p.fit(d[~np.isnan(d).any(axis=1)])
        process[col] = p
        if col != 'action':
            process[f'goal_{col}'] = p

    model = swm.wm.utils.load_pretrained(cfg.policy).to('cuda').eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    solver = hydra.utils.instantiate(
        cfg.solver,
        cost=swm.planning.ShootingCostEvaluator(
            model, hydra.utils.instantiate(cfg.objective)
        ),
    )
    world.set_policy(
        swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(**cfg.plan_config),
            process=process,
            transform=transform,
        )
    )

    # Same start/goal sampling as eval_wm.py (same seed -> same episodes).
    col = episode_col(dataset)
    ep_ids = dataset.get_col_data(col)
    step = dataset.get_col_data('step_idx')
    # episode length = max(step_idx) + 1, as in eval_wm.get_episodes_length
    uniq, inv = np.unique(ep_ids, return_inverse=True)
    max_step = np.full(len(uniq), -1, dtype=np.int64)
    np.maximum.at(max_step, inv, step.astype(np.int64))
    max_start = (max_step + 1 - cfg.eval.goal_offset_steps - 1)[inv]
    valid = np.nonzero(step <= max_start)[0]
    rng = np.random.default_rng(cfg.seed)
    pick = np.sort(
        valid[rng.choice(len(valid), size=cfg.eval.num_eval, replace=False)]
    )
    episodes_idx = ep_ids[pick].tolist()
    start_steps = step[pick].tolist()

    t0 = time.time()
    res = evaluate_varied(world, dataset, episodes_idx, start_steps, cfg, variation)
    dt = time.time() - t0

    # Contact sheet of the first few varied start/goal frames, for inspection.
    sheet = np.concatenate(
        [np.concatenate([s, g], axis=0) for s, g in zip(res['start_frames'], res['goal_frames'])],
        axis=1,
    )
    Image.fromarray(sheet).save(out_dir / 'start_top_goal_bottom.png')

    summary = {
        'env': cfg.world.env_name,
        'policy': cfg.policy,
        'variation': variation,
        'success_rate': res['success_rate'],
        'n_episodes': cfg.eval.num_eval,
        'goal_offset_steps': cfg.eval.goal_offset_steps,
        'eval_budget': cfg.eval.eval_budget,
        'seed': cfg.seed,
        'eval_seconds': dt,
        'episode_successes': res['episode_successes'],
    }
    (out_dir / 'result.json').write_text(json.dumps(summary, indent=2))
    print(f"[variation] {cfg.world.env_name} {variation}: {res['success_rate']:.1f}% ({dt:.0f}s)")


if __name__ == '__main__':
    run()
