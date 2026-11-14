"""Figures for traced trajectories (from trace_lewm.py / trace_dinowm.py).

For every ep<i>_{success,failure}.npz in each trace dir, writes to
<figdir>/<label>/:
  ep<i>_<tag>_filmstrip.png : start ... evenly spaced frames ... final | goal
                              (DINO-WM: second row = world-model imagination
                               decoded at the same time steps)
  ep<i>_<tag>_panels.png    : 2-D path (agent / object) with start + goal,
                              distance-to-goal over time, actions over time
  ep<i>_<tag>.gif           : animation, current frame next to the goal frame
and <label>/summary.png (success vs failure side by side).

  python make_trajectory_figures.py <figdir> <label>=<trace_dir> [...]
"""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

N_FRAMES = 8
OK, BAD = '#1a9850', '#d73027'



def _load(p):
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _frame_ids(T, n=N_FRAMES):
    return np.unique(np.linspace(0, T - 1, n).round().astype(int))


def _border(ax, color):
    for s in ax.spines.values():
        s.set_edgecolor(color)
        s.set_linewidth(3)


def filmstrip(d, title, out, success_step=None):
    frames = d['frames']
    T = len(frames)
    imag = d.get('imagined')
    if imag is not None:
        # Align real frames with the world model's predictions (one per action
        # chunk of `frameskip` env steps) so the two rows are directly comparable.
        fs_ = int(d.get('frameskip', 1))
        ids = _frame_ids(len(imag), min(N_FRAMES, len(imag))) * fs_
        ids = ids[ids < T]
    else:
        ids = _frame_ids(T)
    rows = 2 if imag is not None else 1
    cols = len(ids) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 2.1 * rows + 0.4), squeeze=False)
    fs = int(d.get('frameskip', 1)) if imag is not None else 1
    for j, t in enumerate(ids):
        ax = axes[0, j]
        ax.imshow(frames[t])
        lab = f't={t}'
        if success_step is not None and success_step >= 0 and t >= success_step:
            lab += ' ✓'
        ax.set_title(lab, fontsize=8)
        if imag is not None:
            k = min(int(round(t / fs)), len(imag) - 1)
            axes[1, j].imshow(imag[k])
            axes[1, j].set_title(f'imagined t={k * fs}', fontsize=8)
    axes[0, -1].imshow(d['goal'])
    axes[0, -1].set_title('GOAL', fontsize=9, fontweight='bold')
    _border(axes[0, -1], '#4575b4')
    if imag is not None:
        axes[1, -1].axis('off')
        axes[0, 0].set_ylabel('real env', fontsize=9)
        axes[1, 0].set_ylabel('world model', fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# (label, current key, goal key, xy slice) for 2-D paths + distances, LeWM envs.
LEWM_POS = [
    ('agent', 'info_pos_agent', 'info_goal_pos_agent', slice(0, 2)),                 # TwoRoom
    ('agent', 'info_proprio', 'info_goal_proprio', slice(0, 2)),                     # TwoRoom (proprio = agent xy)
    ('fingertip', 'info_finger_pos', 'info_goal_finger_pos', slice(0, 2)),           # Reacher
    ('block (xy)', 'info_privileged/block_0_pos', 'info_goal_privileged_block_0_pos', slice(0, 2)),  # Cube
    ('gripper (xy)', 'info_proprio/effector_pos', 'info_goal_proprio_effector_pos', slice(0, 2)),    # Cube
]
LEWM_DIST_ONLY = [
    ('joint angles', 'info_qpos', 'info_goal_qpos', slice(None)),                    # Reacher
]


def _is_pusht_state(d):
    return 'info_state' in d and d['info_state'].ndim == 2 and d['info_state'].shape[1] == 7


def _paths(d, kind):
    """Return list of (name, xy (T,2), goal_xy or None).

    LeWM: env-side keys such as pos_agent/block_pose are stale at t=0 (the env
    reports its pre-reset pose before the dataset start state is applied), so
    PushT uses the injected/updated 7-D `state` = [agent xy, block xy, angle, vel].
    """
    out = []
    if kind == 'lewm':
        if _is_pusht_state(d):
            s, g = d['info_state'], d['info_goal_state'][-1]
            out += [('agent', s[:, :2], g[:2]), ('block', s[:, 2:4], g[2:4])]
        else:
            for name, cur, goal, sl in LEWM_POS:
                if cur in d and not any(o[0] == name for o in out):
                    g = d[goal][-1, sl] if goal in d else None
                    out.append((name, d[cur][1:, sl], g))  # t=0 env fields are stale
    else:
        s, g = d['states'], d['state_g']
        out.append(('agent', s[:, :2], g[:2]))
        if s.shape[1] >= 7:  # PushT: block xy at 2:4
            out.append(('block', s[:, 2:4], g[2:4]))
    return out


def _pusht_dists(s, g):
    return {
        'agent pos': np.linalg.norm(s[:, :2] - g[:2], axis=1),
        'block pos': np.linalg.norm(s[:, 2:4] - g[2:4], axis=1),
        'block angle (rad)': np.abs((s[:, 4] - g[4] + np.pi) % (2 * np.pi) - np.pi),
    }


def _dist_series(d, kind):
    """Distance-to-goal curves on position-like quantities: {label: (T,)}."""
    if kind == 'lewm':
        if _is_pusht_state(d):
            return _pusht_dists(d['info_state'], d['info_goal_state'][-1])
        res = {}
        for name, cur, goal, _ in LEWM_POS + LEWM_DIST_ONLY:
            if cur in d and goal in d and name not in res:
                # skip t=0: env-side fields are stale until the first step
                res[name] = np.linalg.norm(d[cur][1:] - d[goal][1:], axis=1)
        return res
    s, g = d['states'], d['state_g']
    if s.shape[1] >= 7:
        return _pusht_dists(s, g)
    return {'agent pos': np.linalg.norm(s[:, :2] - g[:2], axis=1)}


def _arrows(ax, xy, color, n=6):
    """Direction-of-travel arrowheads at ~n points spaced evenly by path length."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total = seg.sum()
    if total <= 0:
        return
    cum = np.concatenate([[0], np.cumsum(seg)])
    # Direction is measured over a stretch of >= 3% of the plot extent, so
    # arrows show the heading rather than per-step jitter.
    tiny = 0.03 * max(float(np.linalg.norm(xy.max(0) - xy.min(0))), 1e-9)
    for frac in (np.arange(n) + 0.5) / n:
        i = int(np.searchsorted(cum, frac * total)) - 1
        i = min(max(i, 0), len(seg) - 1)
        # skip stationary stretches: look ahead for a visible displacement
        j = i + 1
        while j < len(xy) - 1 and np.linalg.norm(xy[j] - xy[i]) < tiny:
            j += 1
        if np.linalg.norm(xy[j] - xy[i]) < tiny:
            continue
        ax.annotate('', xy=xy[j], xytext=xy[i], zorder=5,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5,
                                    mutation_scale=16, shrinkA=0, shrinkB=0))


def panels(d, kind, title, out, success_step=None):
    paths = _paths(d, kind)
    dist = _dist_series(d, kind)
    acts = d['actions']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    if paths:
        for (name, xy, goal), c in zip(paths, plt.cm.tab10.colors):
            ax.plot(xy[:, 0], xy[:, 1], '-', color=c, lw=1.5, label=name)
            _arrows(ax, xy, c)
            ax.scatter(*xy[0], color=c, marker='o', s=50, edgecolor='k', zorder=3)
            ax.scatter(*xy[-1], color=c, marker='s', s=50, edgecolor='k', zorder=3)
            if goal is not None:
                ax.scatter(*goal, color=c, marker='*', s=220, edgecolor='k', zorder=4)
        ax.set_aspect('equal', adjustable='datalim')
        if max(np.abs(xy).max() for _, xy, _ in paths) > 20:
            ax.invert_yaxis()  # pixel coordinates (y down), matches the frames
        ax.legend(fontsize=8, title='o start  ■ end  ★ goal', title_fontsize=7)
        ax.set_title('path (state space)')
    else:
        ax.text(0.5, 0.5, 'no 2-D state logged', ha='center', va='center')
        ax.set_axis_off()
    ax = axes[1]
    if dist:
        for name, v in dist.items():
            ax.plot(v, label=name)
        if success_step is not None and success_step >= 0:
            ax.axvline(success_step, color=OK, ls='--', label='success')
        ax.set_xlabel('env step')
        ax.set_title('distance to goal')
        ax.legend(fontsize=8)
    else:
        ax.set_axis_off()
    ax = axes[2]
    for k in range(acts.shape[1]):
        ax.plot(acts[:, k], lw=1, label=f'a{k}')
    ax.set_xlabel('env step')
    ax.set_title('executed actions')
    if acts.shape[1] <= 8:
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def gif(d, out, fps=8):
    goal = d['goal']
    frames = [np.concatenate([f, goal], axis=1) for f in d['frames']]
    imageio.mimsave(out, frames, duration=1000 / fps, loop=0)


def summary(items, label, out):
    fig, axes = plt.subplots(len(items), 1, figsize=(16, 2.4 * len(items)), squeeze=False)
    for ax, (tag, png) in zip(axes[:, 0], items):
        ax.imshow(imageio.imread(png))
        ax.set_axis_off()
    fig.suptitle(label, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(figdir, *specs):
    figdir = Path(figdir)
    for spec in specs:
        label, tdir = spec.split('=', 1)
        tdir = Path(tdir)
        meta = json.loads((tdir / 'trace_meta.json').read_text())
        kind = 'dinowm' if 'real_steps' in meta else 'lewm'
        od = figdir / label
        od.mkdir(parents=True, exist_ok=True)
        items = []
        for p in sorted(tdir.glob('ep*_*.npz')):
            d = _load(p)
            tag = p.stem.split('_')[-1]
            if kind == 'lewm':
                ss = int(d['first_success_step'])
            else:
                ss = None
            title = (f'{label} | episode {p.stem.split("_")[0][2:]} | '
                     f'{tag.upper()} | {len(d["frames"]) - 1} steps'
                     + (f' | success at step {ss}' if ss is not None and ss >= 0 else ''))
            fs = od / f'{p.stem}_filmstrip.png'
            filmstrip(d, title, fs, ss)
            panels(d, kind, title, od / f'{p.stem}_panels.png', ss)
            gif(d, od / f'{p.stem}.gif')
            items.append((tag, fs))
        if items:
            summary(sorted(items, key=lambda x: x[0] != 'success'),
                    f'{label}: success rate {meta["success_rate"]:.0f}% over {meta["n_episodes"]} episodes',
                    od / 'summary.png')
        print(f'[figs] {label}: {len(items)} trajectories -> {od}')


if __name__ == '__main__':
    main(*sys.argv[1:])
