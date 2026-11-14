"""Score V-JEPA 2 surprise values on IntPhys 2 (violation-of-expectation).

Input: the losses_*.pth written by IntPhys2/prediction_evals (evals/intphys2/eval.py):
  losses: (n_videos, n_context_lengths, T) L1 latent prediction error over time
  names : video file names
Surprise of a video = max over time of the prediction error (as in
jepa-intuitive-physics). Each IntPhys 2 scene has possible and impossible
versions of the same event; a pair counts as correct when the impossible video
gets the higher surprise (relative classification; chance = 50%).

  python score_intphys2.py <losses.pth> <metadata.csv> [out.json]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CONDITION = {'immutability_texture': 'immutability', 'continuity_swap': 'continuity'}


def pair_acc(df, col):
    correct = total = 0
    for _, g in df.groupby('scene'):
        pos = g.loc[g.possible, col].values
        imp = g.loc[~g.possible, col].values
        for p, i in itertools.product(pos, imp):
            total += 1
            correct += float(i > p) + 0.5 * float(i == p)
    return (100.0 * correct / total if total else float('nan')), total


def main(loss_path, meta_path, out_path=None):
    d = torch.load(loss_path, map_location='cpu', weights_only=False)
    losses = d['losses'].float().numpy()          # (N, C, T); padding is 0
    names = [Path(n).stem for n in d['names']]
    ctxts = list(d['context_lengths'])
    surprise = losses.max(axis=2)                  # (N, C)

    meta = pd.read_csv(meta_path)
    meta['stem'] = meta['file_name'].map(lambda s: Path(s).stem)
    df = pd.DataFrame({'stem': names})
    for j, c in enumerate(ctxts):
        df[f'ctx{c}'] = surprise[:, j]
    df = df.merge(meta, on='stem', how='inner')
    # Scenes are identified by game + environment + scene index.
    df['scene'] = df['game_name'].astype(str) + '|' + df['env'].astype(str) + '|' + df['SceneIndex'].astype(str)
    df['possible'] = df['type'].str.contains('_Possible')
    df['cond'] = df['condition'].map(lambda c: CONDITION.get(c, c))

    res = {'n_videos': int(len(df)), 'frame_step': d.get('frame_step'), 'per_context': {}}
    for c in ctxts:
        col = f'ctx{c}'
        acc, n = pair_acc(df, col)
        r = {'all': acc, 'n_pairs': n}
        for key in ('Difficulty', 'cond', 'Camera'):
            if key in df:
                for v, g in df.groupby(key):
                    r[f'{key}={v}'] = pair_acc(g, col)[0]
        res['per_context'][c] = r
    best = max(res['per_context'], key=lambda c: res['per_context'][c]['all'])
    res['best_context'] = best
    res['note'] = ('best_context is chosen on the same data (optimistic); '
                   'report per-context numbers or a fixed context for fair comparison.')

    for c, r in res['per_context'].items():
        extra = '  '.join(f'{k}:{v:.1f}' for k, v in r.items() if k not in ('all', 'n_pairs'))
        print(f'ctx={c:>3}  acc={r["all"]:.1f}%  (pairs={r["n_pairs"]})  {extra}')
    if out_path:
        Path(out_path).write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
