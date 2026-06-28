"""
beta_sweep_summarize.py — Build unified comparison across beta values.

After running aggregate_coverage_recover.py at multiple beta values, this
script reads the resulting per-beta JSONs and produces a single Markdown
table showing how the population-level result moves with beta. Used to
back the §5 appendix claim "fixed once, results stable".

Reads:  {sweep_dir}/beta_{B:.1f}/coverage_recover_results.json
Writes: {sweep_dir}/beta_sweep_summary.md
        {sweep_dir}/beta_sweep_summary.json
"""
import os
import json
import argparse
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_dir', required=True,
                   help='Dir containing beta_{B}/coverage_recover_results.json subdirs')
    p.add_argument('--betas', nargs='+', type=float, default=[0.3, 0.5, 0.7])
    return p.parse_args()


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)

    rows = []
    for beta in args.betas:
        # Format with one decimal to match the directory naming convention
        sub = sweep_dir / f'beta_{beta:.1f}'
        jpath = sub / 'coverage_recover_results.json'
        if not jpath.exists():
            print(f'  MISSING: {jpath}')
            continue

        with open(jpath) as f:
            data = json.load(f)

        pop = data.get('population', {})
        cells = data.get('cell_results', {})

        n_cells = pop.get('n_cells', 0)
        mean_d  = pop.get('mean_across_cells', float('nan'))
        p_less  = pop.get('wilcoxon_p_less', float('nan'))
        n_neg   = pop.get('n_negative', 0)

        # Compute c*_val statistics across cells
        c_stars = [v['c_star_val'] for v in cells.values()
                   if 'c_star_val' in v]
        median_cstar = float(np.median(c_stars)) if c_stars else float('nan')
        min_cstar    = float(np.min(c_stars))    if c_stars else float('nan')
        max_cstar    = float(np.max(c_stars))    if c_stars else float('nan')

        # Per-cell delta range (best, worst, median)
        deltas = [v['mean_delta'] for v in cells.values()
                  if 'mean_delta' in v]
        median_d = float(np.median(deltas)) if deltas else float('nan')
        best_d   = float(np.min(deltas))    if deltas else float('nan')   # most negative
        worst_d  = float(np.max(deltas))    if deltas else float('nan')   # least negative

        n_fallback = sum(1 for v in cells.values()
                         if v.get('c_star_val', 0) >= 0.999)

        rows.append({
            'beta':         beta,
            'n_cells':      n_cells,
            'mean_delta':   mean_d,
            'median_delta': median_d,
            'best_delta':   best_d,
            'worst_delta':  worst_d,
            'p_less':       p_less,
            'n_improving':  n_neg,
            'median_cstar': median_cstar,
            'min_cstar':    min_cstar,
            'max_cstar':    max_cstar,
            'n_fallback':   n_fallback,
        })

    # ---- Markdown ----
    md = ['# β robustness sweep — population-level summary', '']
    md.append('## Population-level test')
    md.append('')
    md.append('| β | n_cells | mean Δ | median Δ | best Δ | worst Δ | Wilcoxon p (less) | cells improving | fallback cells |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["beta"]:.1f} | {r["n_cells"]} | '
            f'{r["mean_delta"]:+.4f} | {r["median_delta"]:+.4f} | '
            f'{r["best_delta"]:+.4f} | {r["worst_delta"]:+.4f} | '
            f'{r["p_less"]:.5f} | {r["n_improving"]}/{r["n_cells"]} | '
            f'{r["n_fallback"]}/{r["n_cells"]} |'
        )
    md.append('')
    md.append('## c*_val operating-point distribution across cells')
    md.append('')
    md.append('| β | median c*_val | min c*_val | max c*_val |')
    md.append('|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["beta"]:.1f} | {r["median_cstar"]:.2f} | '
            f'{r["min_cstar"]:.2f} | {r["max_cstar"]:.2f} |'
        )
    md.append('')
    md.append('## Interpretation')
    md.append('')
    md.append('β controls the source-val error reduction target used to fit '
              'c*_val. The §5 main text fixes β=0.5 (largest c at which '
              'source-val full-coverage error halves). This appendix shows '
              'population-level conclusions remain stable across β ∈ {0.3, '
              '0.5, 0.7}.')
    md.append('')
    md.append('- **Smaller β (0.3)** demands less reduction → larger c*_val '
              '(more pixels retained) → smaller absolute Δ on held-out test, '
              'but still all cells should improve.')
    md.append('- **Larger β (0.7)** demands more reduction → smaller c*_val '
              '(more pixels abstained) → larger absolute Δ if the model is '
              'genuinely well-calibrated under shift, more fallback cells '
              'if it is not.')
    md.append('- **Fallback cells** (c*_val pinned to 1.0) contribute Δ=0 and '
              'are reported transparently; if any β has a high fallback '
              'count, the criterion was too aggressive for the source-val '
              'error floor on those cells.')
    md.append('')

    out_md = sweep_dir / 'beta_sweep_summary.md'
    with open(out_md, 'w') as f:
        f.write('\n'.join(md))
    print(f'Saved → {out_md}')

    out_json = sweep_dir / 'beta_sweep_summary.json'
    with open(out_json, 'w') as f:
        json.dump({'rows': rows}, f, indent=2, default=str)
    print(f'Saved → {out_json}')


if __name__ == '__main__':
    main()